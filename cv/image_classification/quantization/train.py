import argparse
import copy
import inspect
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    mobilenet_v3_small,
    resnet18,
)
from torchvision.transforms import InterpolationMode

"""
torchao 图像分类量化示例：
1. 先训练或加载一个浮点分类模型。
2. 用 PT2E 路线完成 PTQ（export -> prepare_pt2e -> calibrate -> convert_pt2e）。
3. 用 PT2E 路线完成 QAT（export -> prepare_qat_pt2e -> fine-tune -> convert_pt2e）。
4. 输出 float / PTQ / QAT 的精度、模型大小和 CPU 延迟对比。

这里故意使用 torchvision 的经典 CNN，而不是任意 timm 模型。原因是：
- torchao 当前图像分类 PT2E 教程主要围绕标准 CNN 展开；
- ResNet / MobileNet 这类 Conv2d + BatchNorm + Linear 模型更适合作为 static int8 教学模板；
- 这样能把“量化流程本身”讲清楚，而不是先被模型导出兼容性问题干扰。
"""


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    release_year: int
    summary: str
    recommended_for: str


@dataclass
class MetricResult:
    loss: float
    top1: float
    topk: float


@dataclass
class CompareResult:
    name: str
    top1: float
    topk: float
    size_mb: float | None
    latency_ms: float | None
    notes: str


MODEL_SPECS = {
    "resnet18": ModelSpec(
        alias="resnet18",
        release_year=2015,
        summary="结构简单，量化图里最容易看懂，适合作为 PTQ/QAT 第一套模板。",
        recommended_for="先把 PT2E 量化流程跑通，再迁移到自己的分类模型。",
    ),
    "mobilenet_v3_small": ModelSpec(
        alias="mobilenet_v3_small",
        release_year=2019,
        summary="更偏轻量部署，量化后更接近真实移动端分类场景。",
        recommended_for="希望同时关注量化精度和推理开销。",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="torchao 图像分类量化训练示例：支持 float、PTQ、QAT 和 compare。"
    )
    parser.add_argument(
        "--mode",
        choices=["train_float", "ptq", "qat", "compare"],
        default="compare",
        help="执行模式：训练浮点模型、运行 PTQ、运行 QAT，或顺序对比全部方法。",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["cifar100", "custom"],
        default="cifar100",
        help="数据集类型。教学默认用 CIFAR-100，也支持 ImageFolder 风格自定义数据集。",
    )
    parser.add_argument(
        "--data-root",
        default="./datasets",
        help="CIFAR-100 下载目录，或自定义数据集根目录。",
    )
    parser.add_argument(
        "--train-dir",
        default="",
        help="自定义数据集训练目录，例如 ./datasets/my_cls/train。",
    )
    parser.add_argument(
        "--val-dir",
        default="",
        help="自定义数据集验证目录，例如 ./datasets/my_cls/val。",
    )
    parser.add_argument(
        "--model-name",
        choices=sorted(MODEL_SPECS.keys()),
        default="resnet18",
        help="量化示例默认推荐 ResNet18；也支持 MobileNetV3 Small。",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否加载 ImageNet 预训练权重。默认开启，更适合迁移学习和量化教学。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="训练/验证批大小。",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader worker 数量。",
    )
    parser.add_argument(
        "--float-epochs",
        type=int,
        default=3,
        help="浮点模型微调 epoch 数。",
    )
    parser.add_argument(
        "--qat-epochs",
        type=int,
        default=2,
        help="QAT 微调 epoch 数。通常比重新训练全模型少很多。",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="浮点模型微调学习率。",
    )
    parser.add_argument(
        "--qat-learning-rate",
        type=float,
        default=1e-4,
        help="QAT 阶段学习率。通常比 float 微调更小。",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="优化器权重衰减。",
    )
    parser.add_argument(
        "--calib-batches",
        type=int,
        default=50,
        help="PTQ 校准使用多少个 batch。代表性比数量更重要。",
    )
    parser.add_argument(
        "--train-subset",
        type=int,
        default=0,
        help="仅使用前 N 个训练样本，0 表示全部。适合快速烟测。",
    )
    parser.add_argument(
        "--val-subset",
        type=int,
        default=0,
        help="仅使用前 N 个验证样本，0 表示全部。",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="每个训练 epoch 最多跑多少个 batch，0 表示全部。",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
        help="验证阶段最多评估多少个 batch，0 表示全部。",
    )
    parser.add_argument(
        "--disable-observer-epoch",
        type=int,
        default=1,
        help="QAT 到达该 epoch 后关闭 observer。0 表示从第一轮后就开始关闭。",
    )
    parser.add_argument(
        "--freeze-bn-epoch",
        type=int,
        default=1,
        help="QAT 到达该 epoch 后冻结 BatchNorm 统计量。",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=30,
        help="CPU 延迟基准的正式迭代次数。",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=10,
        help="CPU 延迟基准的预热次数。",
    )
    parser.add_argument(
        "--float-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="浮点模型训练设备。",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/image_classification/torchao_quantization",
        help="checkpoint 和结果输出目录。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。",
    )
    parser.add_argument(
        "--reuse-float-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="存在浮点 checkpoint 时是否复用。默认复用，避免 compare 每次都重训。",
    )
    parser.add_argument(
        "--backend",
        choices=["x86_inductor", "xnnpack"],
        default="x86_inductor",
        help=(
            "PT2E 量化 backend。x86_inductor 依赖更少，适合普通 x86 CPU；"
            "xnnpack 更贴近 torchao 官方图像分类教程，但需要 executorch。"
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="只打印支持的模型，不执行训练和量化。",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前环境没有可用的 CUDA 设备，请改用 --float-device cpu。")
    return torch.device(device_name)


def print_supported_models():
    print("支持的量化教学模型：")
    for key in sorted(MODEL_SPECS.keys()):
        spec = MODEL_SPECS[key]
        print(
            f"- {spec.alias} ({spec.release_year})\n"
            f"  说明: {spec.summary}\n"
            f"  适合: {spec.recommended_for}"
        )


def resolve_model_recipe(model_name, pretrained):
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model_builder = resnet18
    elif model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model_builder = mobilenet_v3_small
    else:
        raise ValueError(f"暂不支持的模型: {model_name}")

    if weights is not None:
        preset = weights.transforms()
        mean = preset.mean
        std = preset.std
        interpolation = preset.interpolation
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        interpolation = InterpolationMode.BILINEAR
    return model_builder, weights, mean, std, interpolation


def build_model(model_name, num_classes, pretrained):
    if model_name == "resnet18":
        _, weights, mean, std, interpolation = resolve_model_recipe(
            model_name,
            pretrained,
        )
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "mobilenet_v3_small":
        _, weights, mean, std, interpolation = resolve_model_recipe(
            model_name,
            pretrained,
        )
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"暂不支持的模型: {model_name}")
    return model, mean, std, interpolation


def maybe_limit_dataset(dataset: Dataset, limit: int) -> Dataset:
    if limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def build_transforms(mean, std, interpolation):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=interpolation),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def build_datasets(args, train_transform, eval_transform):
    if args.dataset_type == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=args.data_root,
            train=True,
            transform=train_transform,
            download=True,
        )
        calib_dataset = datasets.CIFAR100(
            root=args.data_root,
            train=True,
            transform=eval_transform,
            download=True,
        )
        val_dataset = datasets.CIFAR100(
            root=args.data_root,
            train=False,
            transform=eval_transform,
            download=True,
        )
        class_names = train_dataset.classes
    else:
        if not args.train_dir or not args.val_dir:
            raise ValueError(
                "使用自定义数据集时，必须同时提供 --train-dir 和 --val-dir。"
            )
        train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
        calib_dataset = datasets.ImageFolder(args.train_dir, transform=eval_transform)
        val_dataset = datasets.ImageFolder(args.val_dir, transform=eval_transform)
        if train_dataset.class_to_idx != val_dataset.class_to_idx:
            raise ValueError(
                "train_dir 和 val_dir 的类别目录不一致，请确保两个目录的类别名完全相同。"
            )
        class_names = train_dataset.classes

    train_dataset = maybe_limit_dataset(train_dataset, args.train_subset)
    calib_dataset = maybe_limit_dataset(calib_dataset, args.train_subset)
    val_dataset = maybe_limit_dataset(val_dataset, args.val_subset)

    if len(train_dataset) == 0 or len(calib_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("训练集、校准集或验证集为空，请检查数据目录或 subset 设置。")
    return train_dataset, calib_dataset, val_dataset, class_names


def build_loader(dataset, batch_size, num_workers, shuffle):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)


def topk_from_logits(logits, targets, topk):
    _, pred = logits.topk(topk, dim=1, largest=True, sorted=True)
    correct = pred.eq(targets.unsqueeze(1))
    return correct.any(dim=1).sum().item()


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    topk,
    max_batches=0,
    set_training_mode=True,
):
    if set_training_mode and hasattr(model, "train"):
        model.train()
    total_loss = 0.0
    total_samples = 0
    total_top1 = 0
    total_topk = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_top1 += logits.argmax(dim=1).eq(labels).sum().item()
        total_topk += topk_from_logits(logits, labels, topk)

    return MetricResult(
        loss=total_loss / total_samples,
        top1=total_top1 / total_samples,
        topk=total_topk / total_samples,
    )


@torch.inference_mode()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
    topk,
    max_batches=0,
    set_eval_mode=True,
):
    if set_eval_mode and hasattr(model, "eval"):
        model.eval()

    total_loss = 0.0
    total_samples = 0
    total_top1 = 0
    total_topk = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_top1 += logits.argmax(dim=1).eq(labels).sum().item()
        total_topk += topk_from_logits(logits, labels, topk)

    return MetricResult(
        loss=total_loss / total_samples,
        top1=total_top1 / total_samples,
        topk=total_topk / total_samples,
    )


def format_metrics(prefix, metrics, topk_name):
    return (
        f"{prefix} loss={metrics.loss:.4f}, "
        f"top1={metrics.top1:.4f}, "
        f"{topk_name}={metrics.topk:.4f}"
    )


def save_float_checkpoint(path, model, class_names, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "model_name": args.model_name,
        "dataset_type": args.dataset_type,
        "pretrained": args.pretrained,
    }
    torch.save(checkpoint, path)
    return path


def load_float_checkpoint(path, model):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint


def model_size_mb(model):
    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
        torch.save(model.state_dict(), tmp.name)
        return Path(tmp.name).stat().st_size / 1e6


@torch.inference_mode()
def benchmark_cpu_latency(model, example_batch, warmup, iters, set_eval_mode=True):
    batch = example_batch.to("cpu")
    if set_eval_mode and hasattr(model, "eval"):
        model.eval()
    for _ in range(max(0, warmup)):
        model(batch)

    start = time.perf_counter()
    for _ in range(max(1, iters)):
        model(batch)
    end = time.perf_counter()
    return (end - start) * 1000 / max(1, iters)


def print_compare_table(results: Iterable[CompareResult], topk_name):
    print("\n结果对比：")
    print(
        f"{'Method':<12} {'Top1':>10} {'%s' % topk_name:>10} {'Size(MB)':>12} "
        f"{'Latency(ms)':>12}  Notes"
    )
    print("-" * 80)
    for result in results:
        size_str = "-" if result.size_mb is None else f"{result.size_mb:.2f}"
        latency_str = "-" if result.latency_ms is None else f"{result.latency_ms:.2f}"
        print(
            f"{result.name:<12} {result.top1:>10.4f} {result.topk:>10.4f} "
            f"{size_str:>12} {latency_str:>12}  {result.notes}"
        )


def export_with_dynamic_batch(model, example_inputs):
    dynamic_shapes = tuple(
        {0: torch.export.Dim("batch")} if i == 0 else None
        for i in range(len(example_inputs))
    )
    return torch.export.export(
        model,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
    ).module()


def import_pt2e_apis():
    try:
        import torchao
        from torchao.quantization.pt2e.quantize_pt2e import (
            convert_pt2e,
            prepare_pt2e,
            prepare_qat_pt2e,
        )
    except Exception as exc:
        raise RuntimeError(
            "当前环境中的 torchao 无法正常导入。常见原因是 torch 与 torchao 版本不匹配，"
            "或者 torchao 安装不完整。请先校对当前虚拟环境里的 torch / torchao 版本。"
        ) from exc
    return torchao, prepare_pt2e, prepare_qat_pt2e, convert_pt2e


def build_quantizer(backend, is_qat):
    if backend == "xnnpack":
        try:
            from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
                XNNPACKQuantizer,
                get_symmetric_quantization_config,
            )
        except ImportError as exc:
            raise RuntimeError(
                "xnnpack backend 需要额外安装 executorch，例如 `pip install executorch`。"
            ) from exc
        kwargs = {}
        signature = inspect.signature(get_symmetric_quantization_config)
        if "is_qat" in signature.parameters:
            kwargs["is_qat"] = is_qat
        if "is_per_channel" in signature.parameters:
            kwargs["is_per_channel"] = True
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config(**kwargs))
        return quantizer

    if backend == "x86_inductor":
        from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
            X86InductorQuantizer,
            get_default_x86_inductor_quantization_config,
        )

        kwargs = {}
        signature = inspect.signature(get_default_x86_inductor_quantization_config)
        if "is_qat" in signature.parameters:
            kwargs["is_qat"] = is_qat
        if "is_dynamic" in signature.parameters:
            kwargs["is_dynamic"] = False

        quantizer = X86InductorQuantizer()
        quantizer.set_global(get_default_x86_inductor_quantization_config(**kwargs))
        return quantizer

    raise ValueError(f"暂不支持的 backend: {backend}")


def maybe_move_exported_model_to_eval(torchao_module, model):
    move_to_eval = getattr(torchao_module.quantization.pt2e, "move_exported_model_to_eval", None)
    if callable(move_to_eval):
        move_to_eval(model)


def disable_observer_if_supported(torchao_module, model):
    disable_observer = getattr(torchao_module.quantization.pt2e, "disable_observer", None)
    if callable(disable_observer):
        model.apply(disable_observer)


def freeze_bn_stats_in_exported_graph(model):
    batch_norm_targets = {
        torch.ops.aten._native_batch_norm_legit.default,
    }
    if hasattr(torch.ops.aten, "cudnn_batch_norm"):
        batch_norm_targets.add(torch.ops.aten.cudnn_batch_norm.default)

    changed = False
    for node in model.graph.nodes:
        if node.target not in batch_norm_targets:
            continue
        node_args = list(node.args)
        if len(node_args) <= 5:
            continue
        node_args[5] = False
        node.args = tuple(node_args)
        changed = True

    if changed:
        model.graph.lint()
        model.recompile()


def calibrate(prepared_model, dataloader, max_batches):
    with torch.inference_mode():
        for batch_idx, (images, _) in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            prepared_model(images.to("cpu"))


def train_float_model(args, train_loader, val_loader, num_classes, class_names, output_dir):
    checkpoint_path = output_dir / "float_best.pth"
    topk = min(5, num_classes)
    topk_name = f"top{topk}"
    criterion = nn.CrossEntropyLoss()

    model, _, _, _ = build_model(args.model_name, num_classes, args.pretrained)
    float_device = get_device(args.float_device)
    model = model.to(float_device)

    if checkpoint_path.exists() and args.reuse_float_checkpoint:
        print(f"复用已有浮点 checkpoint: {checkpoint_path}")
        load_float_checkpoint(checkpoint_path, model)
        metrics = evaluate(
            model,
            val_loader,
            criterion,
            float_device,
            topk,
            max_batches=args.max_val_batches,
        )
        print(format_metrics("Float", metrics, topk_name))
        return checkpoint_path, metrics

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.float_epochs),
    )

    best_top1 = -1.0
    best_metrics = None

    print(f"开始浮点模型训练，设备: {float_device}")
    for epoch in range(1, args.float_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            float_device,
            topk,
            max_batches=args.max_train_batches,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            float_device,
            topk,
            max_batches=args.max_val_batches,
        )
        scheduler.step()

        print(f"\nFloat Epoch [{epoch}/{args.float_epochs}]")
        print(format_metrics("Train", train_metrics, topk_name))
        print(format_metrics("Val  ", val_metrics, topk_name))

        if val_metrics.top1 > best_top1:
            best_top1 = val_metrics.top1
            best_metrics = val_metrics
            save_float_checkpoint(checkpoint_path, model, class_names, args)
            print(f"保存新的浮点最佳模型到: {checkpoint_path}")

    if best_metrics is None:
        best_metrics = evaluate(
            model,
            val_loader,
            criterion,
            float_device,
            topk,
            max_batches=args.max_val_batches,
        )
        save_float_checkpoint(checkpoint_path, model, class_names, args)

    return checkpoint_path, best_metrics


def build_cpu_float_model_from_checkpoint(args, num_classes, checkpoint_path):
    model, _, _, _ = build_model(args.model_name, num_classes, args.pretrained)
    load_float_checkpoint(checkpoint_path, model)
    model = model.to("cpu")
    model.eval()
    return model


def run_ptq(args, calib_loader, val_loader, num_classes, checkpoint_path):
    torchao, prepare_pt2e, _, convert_pt2e = import_pt2e_apis()
    float_model = build_cpu_float_model_from_checkpoint(args, num_classes, checkpoint_path)
    example_inputs = (torch.randn(2, 3, 224, 224),)

    print(f"\n开始 PTQ，backend: {args.backend}")
    exported_model = export_with_dynamic_batch(float_model, example_inputs)
    quantizer = build_quantizer(args.backend, is_qat=False)
    prepared_model = prepare_pt2e(exported_model, quantizer)

    calibrate(prepared_model, calib_loader, args.calib_batches)
    quantized_model = convert_pt2e(prepared_model)
    maybe_move_exported_model_to_eval(torchao, quantized_model)

    criterion = nn.CrossEntropyLoss()
    topk = min(5, num_classes)
    metrics = evaluate(
        quantized_model,
        val_loader,
        criterion,
        torch.device("cpu"),
        topk,
        max_batches=args.max_val_batches,
        set_eval_mode=False,
    )

    example_batch = next(iter(val_loader))[0]
    latency_ms = benchmark_cpu_latency(
        quantized_model,
        example_batch,
        warmup=args.benchmark_warmup,
        iters=args.benchmark_iters,
        set_eval_mode=False,
    )
    return quantized_model, metrics, latency_ms


def run_qat(args, train_loader, val_loader, num_classes, checkpoint_path, output_dir):
    torchao, _, prepare_qat_pt2e, convert_pt2e = import_pt2e_apis()
    float_model = build_cpu_float_model_from_checkpoint(args, num_classes, checkpoint_path)
    float_model.train()
    example_inputs = (torch.randn(2, 3, 224, 224),)

    print(f"\n开始 QAT，backend: {args.backend}")
    exported_model = export_with_dynamic_batch(float_model, example_inputs)
    quantizer = build_quantizer(args.backend, is_qat=True)
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        prepared_model.parameters(),
        lr=args.qat_learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.qat_epochs),
    )

    topk = min(5, num_classes)
    topk_name = f"top{topk}"
    best_top1 = -1.0
    best_prepared_state = copy.deepcopy(prepared_model.state_dict())
    observer_disabled = False
    bn_frozen = False
    qat_checkpoint_path = output_dir / "qat_prepared_best.pth"

    for epoch in range(args.qat_epochs):
        train_metrics = train_one_epoch(
            prepared_model,
            train_loader,
            criterion,
            optimizer,
            torch.device("cpu"),
            topk,
            max_batches=args.max_train_batches,
            set_training_mode=False,
        )
        scheduler.step()

        current_epoch = epoch + 1
        if not observer_disabled and current_epoch >= max(1, args.disable_observer_epoch):
            disable_observer_if_supported(torchao, prepared_model)
            observer_disabled = True
            print(f"Epoch {epoch + 1}: 已关闭 observer。")

        if not bn_frozen and current_epoch >= max(1, args.freeze_bn_epoch):
            freeze_bn_stats_in_exported_graph(prepared_model)
            bn_frozen = True
            print(f"Epoch {epoch + 1}: 已冻结 BatchNorm 统计量。")

        prepared_model_copy = copy.deepcopy(prepared_model)
        quantized_candidate = convert_pt2e(prepared_model_copy)
        maybe_move_exported_model_to_eval(torchao, quantized_candidate)
        val_metrics = evaluate(
            quantized_candidate,
            val_loader,
            criterion,
            torch.device("cpu"),
            topk,
            max_batches=args.max_val_batches,
            set_eval_mode=False,
        )

        print(f"\nQAT Epoch [{epoch + 1}/{args.qat_epochs}]")
        print(format_metrics("Train", train_metrics, topk_name))
        print(format_metrics("Quant", val_metrics, topk_name))

        if val_metrics.top1 > best_top1:
            best_top1 = val_metrics.top1
            best_prepared_state = copy.deepcopy(prepared_model.state_dict())
            torch.save(best_prepared_state, qat_checkpoint_path)
            print(f"保存新的 QAT prepared checkpoint 到: {qat_checkpoint_path}")

    prepared_model.load_state_dict(best_prepared_state)
    quantized_model = convert_pt2e(prepared_model)
    maybe_move_exported_model_to_eval(torchao, quantized_model)

    final_metrics = evaluate(
        quantized_model,
        val_loader,
        criterion,
        torch.device("cpu"),
        topk,
        max_batches=args.max_val_batches,
        set_eval_mode=False,
    )
    example_batch = next(iter(val_loader))[0]
    latency_ms = benchmark_cpu_latency(
        quantized_model,
        example_batch,
        warmup=args.benchmark_warmup,
        iters=args.benchmark_iters,
        set_eval_mode=False,
    )
    return quantized_model, final_metrics, latency_ms, qat_checkpoint_path


def main():
    args = parse_args()
    if args.list_models:
        print_supported_models()
        return

    set_seed(args.seed)

    _, _, mean, std, interpolation = resolve_model_recipe(
        args.model_name,
        pretrained=args.pretrained,
    )
    train_transform, eval_transform = build_transforms(mean, std, interpolation)

    train_dataset, calib_dataset, val_dataset, class_names = build_datasets(
        args,
        train_transform,
        eval_transform,
    )
    num_classes = len(class_names)
    topk = min(5, num_classes)
    topk_name = f"top{topk}"

    train_loader = build_loader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=True,
    )
    calib_loader = build_loader(
        calib_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=False,
    )
    val_loader = build_loader(
        val_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=False,
    )

    spec = MODEL_SPECS[args.model_name]
    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"数据集类型: {args.dataset_type}")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"校准样本数: {len(calib_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    print(f"类别数: {num_classes}")
    print(f"模型: {spec.alias}")
    print(f"说明: {spec.summary}")
    print(f"建议场景: {spec.recommended_for}")
    print(f"量化 backend: {args.backend}")

    if args.mode == "train_float":
        checkpoint_path, metrics = train_float_model(
            args,
            train_loader,
            val_loader,
            num_classes,
            class_names,
            output_dir,
        )
        float_model = build_cpu_float_model_from_checkpoint(args, num_classes, checkpoint_path)
        latency_ms = benchmark_cpu_latency(
            float_model,
            next(iter(val_loader))[0],
            warmup=args.benchmark_warmup,
            iters=args.benchmark_iters,
        )
        compare_result = CompareResult(
            name="float32",
            top1=metrics.top1,
            topk=metrics.topk,
            size_mb=model_size_mb(float_model),
            latency_ms=latency_ms,
            notes="浮点基线",
        )
        print_compare_table([compare_result], topk_name)
        return

    checkpoint_path, float_metrics = train_float_model(
        args,
        train_loader,
        val_loader,
        num_classes,
        class_names,
        output_dir,
    )
    float_model = build_cpu_float_model_from_checkpoint(args, num_classes, checkpoint_path)
    float_result = CompareResult(
        name="float32",
        top1=float_metrics.top1,
        topk=float_metrics.topk,
        size_mb=model_size_mb(float_model),
        latency_ms=benchmark_cpu_latency(
            float_model,
            next(iter(val_loader))[0],
            warmup=args.benchmark_warmup,
            iters=args.benchmark_iters,
        ),
        notes="浮点基线",
    )

    if args.mode == "ptq":
        ptq_model, ptq_metrics, ptq_latency = run_ptq(
            args,
            calib_loader,
            val_loader,
            num_classes,
            checkpoint_path,
        )
        ptq_result = CompareResult(
            name="ptq_int8",
            top1=ptq_metrics.top1,
            topk=ptq_metrics.topk,
            size_mb=model_size_mb(ptq_model),
            latency_ms=ptq_latency,
            notes="无需再训练，只做校准",
        )
        print_compare_table([float_result, ptq_result], topk_name)
        return

    if args.mode == "qat":
        qat_model, qat_metrics, qat_latency, qat_checkpoint_path = run_qat(
            args,
            train_loader,
            val_loader,
            num_classes,
            checkpoint_path,
            output_dir,
        )
        qat_result = CompareResult(
            name="qat_int8",
            top1=qat_metrics.top1,
            topk=qat_metrics.topk,
            size_mb=model_size_mb(qat_model),
            latency_ms=qat_latency,
            notes=f"额外微调，prepared ckpt: {qat_checkpoint_path.name}",
        )
        print_compare_table([float_result, qat_result], topk_name)
        return

    ptq_model, ptq_metrics, ptq_latency = run_ptq(
        args,
        calib_loader,
        val_loader,
        num_classes,
        checkpoint_path,
    )
    qat_model, qat_metrics, qat_latency, qat_checkpoint_path = run_qat(
        args,
        train_loader,
        val_loader,
        num_classes,
        checkpoint_path,
        output_dir,
    )

    results = [
        float_result,
        CompareResult(
            name="ptq_int8",
            top1=ptq_metrics.top1,
            topk=ptq_metrics.topk,
            size_mb=model_size_mb(ptq_model),
            latency_ms=ptq_latency,
            notes="无需再训练，只做校准",
        ),
        CompareResult(
            name="qat_int8",
            top1=qat_metrics.top1,
            topk=qat_metrics.topk,
            size_mb=model_size_mb(qat_model),
            latency_ms=qat_latency,
            notes=f"额外微调，prepared ckpt: {qat_checkpoint_path.name}",
        ),
    ]
    print_compare_table(results, topk_name)


if __name__ == "__main__":
    main()
