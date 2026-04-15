import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import timm
import torch
from timm.data import resolve_model_data_config
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

"""
图像分类完整流程示例：
1. 使用 CIFAR-100 跑通数据下载、训练、验证、保存模型和预测。
2. 支持切换到用户自定义数据集，要求数据目录满足 ImageFolder 约定：
   train/class_a/xxx.jpg
   train/class_b/xxx.jpg
   val/class_a/yyy.jpg
   val/class_b/yyy.jpg
"""


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    base_name: str
    pretrained_name: str
    release_year: int
    summary: str
    recommended_for: str


MODEL_SPECS = {
    "convnextv2_tiny": ModelSpec(
        alias="convnextv2_tiny",
        base_name="convnextv2_tiny",
        pretrained_name="convnextv2_tiny.fcmae_ft_in1k",
        release_year=2023,
        summary="现代 CNN，效果、稳定性和迁移学习体验比较均衡。",
        recommended_for="首次做图像分类微调，或想要一个稳妥的默认选择。",
    ),
    "mobilenetv4_conv_small": ModelSpec(
        alias="mobilenetv4_conv_small",
        base_name="mobilenetv4_conv_small",
        pretrained_name="mobilenetv4_conv_small.e1200_r224_in1k",
        release_year=2024,
        summary="更轻量，适合低显存、低时延或部署场景。",
        recommended_for="设备资源有限，或者后续想向移动端部署迁移。",
    ),
    "maxvit_tiny": ModelSpec(
        alias="maxvit_tiny",
        base_name="maxvit_tiny_rw_224",
        pretrained_name="maxvit_tiny_rw_224.sw_in1k",
        release_year=2022,
        summary="现代混合架构，兼顾卷积局部建模和全局建模。",
        recommended_for="想体验更强的现代 backbone，但不追求最新论文路线。",
    ),
    "mambaout_tiny": ModelSpec(
        alias="mambaout_tiny",
        base_name="mambaout_tiny",
        pretrained_name="mambaout_tiny.in1k",
        release_year=2024,
        summary="更新的研究路线，适合扩展实验，不建议第一次就拿它当默认模板。",
        recommended_for="已经熟悉基本分类流程，想尝试更新的研究向模型。",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="CIFAR-100 / 自定义数据集图像分类示例"
    )
    parser.add_argument(
        "--dataset-type",
        choices=["cifar100", "custom"],
        default="cifar100",
        help="数据集类型：默认使用 CIFAR-100，也支持自定义目录数据集。",
    )
    parser.add_argument(
        "--data-root",
        default="./datasets",
        help="CIFAR-100 下载目录，或自定义数据集的根目录。",
    )
    parser.add_argument(
        "--train-dir",
        default="",
        help="自定义数据集训练集目录，例如 ./datasets/custom/train。",
    )
    parser.add_argument(
        "--val-dir",
        default="",
        help="自定义数据集验证集目录，例如 ./datasets/custom/val。",
    )
    parser.add_argument(
        "--model-name",
        choices=sorted(MODEL_SPECS.keys()),
        default="convnextv2_tiny",
        help="要使用的模型别名。",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否加载预训练权重。首次训练建议开启；离线环境可用 --no-pretrained。",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="批大小。")
    parser.add_argument("--num-epochs", type=int, default=5, help="训练轮数。")
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="AdamW 学习率。"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="AdamW 权重衰减。"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader 工作进程数。"
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/image_classification",
        help="模型和日志输出目录。",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="训练设备。默认自动选择。",
    )
    parser.add_argument(
        "--predict-samples",
        type=int,
        default=5,
        help="训练结束后展示多少个验证样本的预测结果。",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="仅打印支持的模型及建议，不执行训练。",
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
        raise ValueError("当前环境没有可用的 CUDA 设备，请改用 --device cpu。")
    return torch.device(device_name)


def print_supported_models():
    print("支持的模型：")
    for key in sorted(MODEL_SPECS.keys()):
        spec = MODEL_SPECS[key]
        print(
            f"- {spec.alias} ({spec.release_year})\n"
            f"  说明: {spec.summary}\n"
            f"  适合: {spec.recommended_for}\n"
            f"  timm 基础模型: {spec.base_name}\n"
            f"  timm 预训练权重: {spec.pretrained_name}"
        )


def get_interpolation(name):
    mapping = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "box": InterpolationMode.BOX,
        "hamming": InterpolationMode.HAMMING,
        "lanczos": InterpolationMode.LANCZOS,
    }
    return mapping.get(name, InterpolationMode.BILINEAR)


def build_model(model_name, num_classes, pretrained):
    spec = MODEL_SPECS[model_name]
    timm_name = spec.pretrained_name if pretrained else spec.base_name
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model, spec, timm_name


def build_preview_model(model_name):
    spec = MODEL_SPECS[model_name]
    model = timm.create_model(
        spec.pretrained_name,
        pretrained=False,
        num_classes=100,
    )
    return model


def build_transforms(model):
    data_config = resolve_model_data_config(model)
    input_size = data_config["input_size"]
    image_size = (input_size[-2], input_size[-1])
    mean = data_config["mean"]
    std = data_config["std"]
    crop_pct = data_config.get("crop_pct", 0.875)
    interpolation = get_interpolation(data_config.get("interpolation", "bilinear"))

    resize_size = (
        math.floor(image_size[0] / crop_pct),
        math.floor(image_size[1] / crop_pct),
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                interpolation=interpolation,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform, data_config


def build_datasets(args, train_transform, eval_transform):
    if args.dataset_type == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=args.data_root,
            train=True,
            transform=train_transform,
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
        val_dataset = datasets.ImageFolder(args.val_dir, transform=eval_transform)
        if train_dataset.class_to_idx != val_dataset.class_to_idx:
            raise ValueError(
                "train_dir 和 val_dir 的类别目录不一致，请确保两个目录的类别名完全相同。"
            )
        class_names = train_dataset.classes
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("训练集或验证集为空，请检查数据目录和图片文件。")
    return train_dataset, val_dataset, class_names


def build_dataloaders(train_dataset, val_dataset, batch_size, num_workers):
    use_cuda = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def update_topk_correct(logits, targets, topk):
    topk_indices = logits.topk(topk, dim=1).indices
    matches = topk_indices.eq(targets.unsqueeze(1))
    return matches.any(dim=1).sum().item()


def train_one_epoch(model, dataloader, criterion, optimizer, device, topk):
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_topk_correct = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += logits.argmax(dim=1).eq(labels).sum().item()
        total_topk_correct += update_topk_correct(logits, labels, topk)

    return {
        "loss": total_loss / total_samples,
        "top1": total_correct / total_samples,
        f"top{topk}": total_topk_correct / total_samples,
    }


@torch.inference_mode()
def evaluate(model, dataloader, criterion, device, topk):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_topk_correct = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += logits.argmax(dim=1).eq(labels).sum().item()
        total_topk_correct += update_topk_correct(logits, labels, topk)

    return {
        "loss": total_loss / total_samples,
        "top1": total_correct / total_samples,
        f"top{topk}": total_topk_correct / total_samples,
    }


def format_metrics(prefix, metrics, topk):
    return (
        f"{prefix} loss={metrics['loss']:.4f}, "
        f"top1={metrics['top1']:.4f}, "
        f"top{topk}={metrics[f'top{topk}']:.4f}"
    )


def save_checkpoint(output_dir, model, class_names, spec, timm_name, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pth"
    checkpoint = {
        "model_alias": spec.alias,
        "base_model_name": spec.base_name,
        "timm_model_name": timm_name,
        "release_year": spec.release_year,
        "class_names": class_names,
        "state_dict": model.state_dict(),
        "dataset_type": args.dataset_type,
        "pretrained": args.pretrained,
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def get_sample_reference(dataset, idx, dataset_type):
    if dataset_type == "custom" and hasattr(dataset, "samples"):
        return dataset.samples[idx][0]
    return f"val_sample_index_{idx}"


@torch.inference_mode()
def predict_examples(model, dataset, class_names, device, sample_count, dataset_type):
    model.eval()
    sample_count = min(sample_count, len(dataset))
    print(f"\n展示 {sample_count} 个验证样本的预测结果：")
    for idx in range(sample_count):
        image, label = dataset[idx]
        logits = model(image.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)
        topk = min(3, len(class_names))
        scores, indices = probabilities.topk(topk, dim=1)

        ref = get_sample_reference(dataset, idx, dataset_type)
        gt_name = class_names[label]
        pred_name = class_names[indices[0, 0].item()]
        print(f"- 样本: {ref}")
        print(f"  真实类别: {gt_name}")
        print(f"  预测类别: {pred_name}")
        top_candidates = []
        for score, pred_idx in zip(scores[0], indices[0]):
            top_candidates.append(
                f"{class_names[pred_idx.item()]}({score.item():.4f})"
            )
        print(f"  Top-{topk}: {', '.join(top_candidates)}")


def main():
    args = parse_args()
    if args.list_models:
        print_supported_models()
        return

    set_seed(args.seed)
    device = get_device(args.device)

    # 先用占位类别数构建模型，用来读取模型自己的输入尺寸和归一化配置。
    preview_model = build_preview_model(args.model_name)
    train_transform, eval_transform, data_config = build_transforms(preview_model)
    del preview_model

    train_dataset, val_dataset, class_names = build_datasets(
        args, train_transform, eval_transform
    )
    num_classes = len(class_names)
    topk = min(5, num_classes)

    model, spec, timm_name = build_model(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=args.pretrained,
    )
    model = model.to(device)

    train_loader, val_loader = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.num_epochs),
    )

    output_dir = Path(args.output_dir) / args.model_name
    best_top1 = -1.0
    best_checkpoint = None

    print(f"设备: {device}")
    print(f"数据集类型: {args.dataset_type}")
    print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")
    print(f"类别数: {num_classes}")
    print(f"模型: {spec.alias} ({spec.release_year})")
    print(f"说明: {spec.summary}")
    print(f"建议场景: {spec.recommended_for}")
    print(f"实际加载的 timm 模型名: {timm_name}")
    print(f"模型输入配置: {data_config}")

    start_time = time.time()
    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            topk=topk,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            topk=topk,
        )
        scheduler.step()

        print(f"\nEpoch [{epoch}/{args.num_epochs}]")
        print(format_metrics("Train", train_metrics, topk))
        print(format_metrics("Val  ", val_metrics, topk))

        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            best_checkpoint = save_checkpoint(
                output_dir=output_dir,
                model=model,
                class_names=class_names,
                spec=spec,
                timm_name=timm_name,
                args=args,
            )
            print(f"保存新的最佳模型到: {best_checkpoint}")

    elapsed = time.time() - start_time
    print(f"\n训练完成，总耗时: {elapsed:.1f} 秒")
    if best_checkpoint is not None:
        checkpoint = torch.load(
            best_checkpoint,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["state_dict"])
        print(f"加载最佳模型进行预测演示: {best_checkpoint}")
    predict_examples(
        model=model,
        dataset=val_dataset,
        class_names=class_names,
        device=device,
        sample_count=args.predict_samples,
        dataset_type=args.dataset_type,
    )


if __name__ == "__main__":
    main()
