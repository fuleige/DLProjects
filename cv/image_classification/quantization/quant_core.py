import hashlib
import random
import re
from pathlib import Path

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

try:
    from .quant_types import MetricResult
except ImportError:
    from quant_types import MetricResult


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


def sanitize_tag(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip().lower())
    return cleaned.strip("-") or "default"


def build_experiment_tag(args) -> str:
    weight_tag = "pretrained" if args.pretrained else "scratch"
    if args.dataset_type == "cifar100":
        return f"cifar100_{weight_tag}"

    train_dir = Path(args.train_dir).resolve()
    val_dir = Path(args.val_dir).resolve()
    source_name = train_dir.parent.name if train_dir.name.lower() == "train" else train_dir.name
    source_tag = sanitize_tag(source_name)
    fingerprint = hashlib.sha1(f"{train_dir}|{val_dir}".encode("utf-8")).hexdigest()[:8]
    return f"custom_{source_tag}_{weight_tag}_{fingerprint}"


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
    min_batch_size=1,
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
        if labels.size(0) < min_batch_size:
            continue
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

    if total_samples == 0:
        raise ValueError(
            f"训练阶段没有可用 batch。请检查 batch size 或确保至少存在一个 batch >= {min_batch_size}。"
        )

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
    min_batch_size=1,
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
        if labels.size(0) < min_batch_size:
            continue
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_top1 += logits.argmax(dim=1).eq(labels).sum().item()
        total_topk += topk_from_logits(logits, labels, topk)

    if total_samples == 0:
        raise ValueError(
            f"验证阶段没有可用 batch。请检查 batch size 或确保至少存在一个 batch >= {min_batch_size}。"
        )

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


def get_peak_cuda_memory_mb(device):
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def train_float_model(args, train_loader, val_loader, num_classes, class_names, output_dir):
    checkpoint_path = output_dir / "float_best.pth"
    topk = min(5, num_classes)
    topk_name = f"top{topk}"
    criterion = nn.CrossEntropyLoss()

    model, _, _, _ = build_model(args.model_name, num_classes, args.pretrained)
    float_device = get_device(args.float_device)
    model = model.to(float_device)
    peak_cuda_memory_mb = None

    if float_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(float_device)

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
        peak_cuda_memory_mb = get_peak_cuda_memory_mb(float_device)
        return checkpoint_path, metrics, peak_cuda_memory_mb

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
        peak_cuda_memory_mb = get_peak_cuda_memory_mb(float_device)

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

    peak_cuda_memory_mb = get_peak_cuda_memory_mb(float_device)
    return checkpoint_path, best_metrics, peak_cuda_memory_mb


def build_float_model_from_checkpoint(args, num_classes, checkpoint_path, device, train_mode=False):
    model, _, _, _ = build_model(args.model_name, num_classes, args.pretrained)
    load_float_checkpoint(checkpoint_path, model)
    model = model.to(device)
    if train_mode:
        model.train()
    else:
        model.eval()
    return model
