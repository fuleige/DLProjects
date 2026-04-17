import argparse

try:
    from .quant_types import MODEL_SPECS
except ImportError:
    from quant_types import MODEL_SPECS


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
        default="cuda",
        help="浮点模型训练设备。当前示例默认使用 CUDA，不再静默回退到 CPU。",
    )
    parser.add_argument(
        "--qat-device",
        choices=["auto", "cpu", "cuda"],
        default="cuda",
        help="QAT fake-quant 微调设备。当前示例默认使用 CUDA，最终量化验证和 CPU benchmark 仍在 CPU。",
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


def print_supported_models():
    print("支持的量化教学模型：")
    for key in sorted(MODEL_SPECS.keys()):
        spec = MODEL_SPECS[key]
        print(
            f"- {spec.alias} ({spec.release_year})\n"
            f"  说明: {spec.summary}\n"
            f"  适合: {spec.recommended_for}"
        )
