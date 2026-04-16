# 图像分类示例：CIFAR-100 与自定义数据集

这个目录提供一个“教学导向 + 可落地实战”的图像分类示例：

- 用 `CIFAR-100` 跑通数据下载、训练、验证、保存最佳模型、预测展示。
- 用 `timm` 接入较新的分类模型，而不是只用很老的 baseline。
- 明确说明如果换成用户自定义数据集，数据目录应该怎么整理。

这个目录下的 `quantization/` 需要特别说明一下它的定位：

- 从任务归属看，它属于 `image_classification`，因为量化流程、模型选择和结果解释都围绕图像分类展开。
- 从能力归属看，它也属于“工程化 / 推理部署 / 模型压缩”专题，因为核心目标是把分类模型进一步落到 PTQ / QAT 部署链路。

所以当前仓库采取的是：

- 教程放在 `docs/model_compression/torchao_quantization_guide.md`
- 可运行代码放在 `cv/image_classification/quantization/`

这种组织方式更适合当前阶段，因为它既保留了任务上下文，也把量化明确沉淀成工程化专题，而不是继续塞在任务 README 里讲成长教程。

## 1. 依赖

建议先按 [PyTorch 官方文档](https://pytorch.org/get-started/locally/) 安装与你环境匹配的 `torch` 和 `torchvision`，再安装：

```bash
pip install timm
```

如果你还没有安装 PyTorch，也可以直接一次性安装：

```bash
pip install torch torchvision timm
```

## 2. 先看支持的模型

```bash
python cv/image_classification/example.py --list-models
```

当前主推模型不是 `resnet18` 这类旧 baseline，而是更贴近近几年社区实践的方案：

- `convnextv2_tiny`：2023，默认推荐。适合首次做分类微调，效果和稳定性比较均衡。
- `mobilenetv4_conv_small`：2024，适合轻量化和部署场景。
- `maxvit_tiny`：2022，成熟的现代混合架构。
- `mambaout_tiny`：2024，更偏研究探索，建议作为扩展实验使用。

如果你只是想学最经典的残差网络，可以单独再补一个 `resnet18` baseline；但这个示例不会把它当成“当前推荐模型”。

## 3. 用 CIFAR-100 跑完整流程

```bash
python cv/image_classification/example.py \
  --dataset-type cifar100 \
  --data-root ./datasets \
  --model-name convnextv2_tiny \
  --batch-size 64 \
  --num-epochs 5 \
  --learning-rate 3e-4
```

脚本会自动完成：

- 下载并加载 CIFAR-100
- 根据模型配置构建输入变换
- 训练与验证
- 保存最佳模型到 `./outputs/image_classification/<model_name>/best_model.pth`
- 输出若干验证样本的预测结果

如果你在离线环境运行，或者不希望首次自动下载预训练权重，可以关闭预训练：

```bash
python cv/image_classification/example.py \
  --dataset-type cifar100 \
  --model-name convnextv2_tiny \
  --no-pretrained
```

注意：关闭预训练后，CIFAR-100 上的效果通常会明显下降。教学上建议先体验预训练微调，再比较随机初始化的差异。

## 4. 用户自定义数据集怎么做

这个示例默认支持 `ImageFolder` 风格的目录结构，也就是每个类别一个子目录。

假设你的数据集有 `cat`、`dog` 两个类别，目录应整理成：

```text
datasets/my_animals/
├── train/
│   ├── cat/
│   │   ├── 001.jpg
│   │   └── 002.jpg
│   └── dog/
│       ├── 101.jpg
│       └── 102.jpg
└── val/
    ├── cat/
    │   ├── 201.jpg
    │   └── 202.jpg
    └── dog/
        ├── 301.jpg
        └── 302.jpg
```

然后运行：

```bash
python cv/image_classification/example.py \
  --dataset-type custom \
  --train-dir ./datasets/my_animals/train \
  --val-dir ./datasets/my_animals/val \
  --model-name mobilenetv4_conv_small \
  --batch-size 32 \
  --num-epochs 10
```

这里有几个关键点：

- 类别名直接来自子目录名，不需要额外再手动写 `label2id`。
- 训练集和验证集的类别目录必须一致，否则脚本会报错。
- 这套写法适合单标签图像分类，也就是一张图片只属于一个类别。

如果你的标注现在还是 `CSV`、`Excel` 或数据库格式，不建议直接往这个示例里硬塞读取逻辑。更清晰的做法是先把图片整理成上面的目录结构，再复用这个训练脚本。

## 5. 如何选择模型

可以按下面的思路选：

- 想先做一个稳妥、现代、通用的分类模板：`convnextv2_tiny`
- 算力有限，或者后续要考虑低时延部署：`mobilenetv4_conv_small`
- 想体验更成熟的现代大模型结构：`maxvit_tiny`
- 已经跑通主流程，想尝试更新的研究向模型：`mambaout_tiny`

不要只问“哪个年份最新”，还要看三个现实问题：

- 你的数据量是否足够大
- 你的训练资源是否足够
- 你的最终目标是高精度、低时延，还是教学理解

对大多数入门实战来说，`convnextv2_tiny` 比 `resnet18` 更值得作为默认起点；对部署场景来说，`mobilenetv4_conv_small` 往往更合适。

## 6. 图像分类量化（torchao）

如果你接下来想做的是“把分类模型量化后部署”，建议不要直接在当前这个 `timm` 示例上硬改。更清晰的路线是先用仓库里单独拆出来的量化分支目录：

- 教程：[docs/model_compression/torchao_quantization_guide.md](../../docs/model_compression/torchao_quantization_guide.md)
- 代码目录：[cv/image_classification/quantization/](./quantization/)
- 运行入口：`cv/image_classification/quantization/train.py`
- 实验记录：`cv/image_classification/quantization/BENCHMARK.md`

这套量化示例有两个特点：

- 教学上更清楚：它专门围绕 `ResNet18 / MobileNetV3` 这类更适合讲解 `PTQ/QAT` 的 CNN 写。
- 实战上更稳：直接对齐当前 `torchao` 官方推荐的 `PT2E` 量化路线，而不是旧版 `torch.ao.quantization` API。

如果只想先快速体验完整对比，可以直接运行：

```bash
python3 cv/image_classification/quantization/train.py \
  --mode compare \
  --dataset-type cifar100 \
  --model-name resnet18 \
  --float-epochs 1 \
  --qat-epochs 1 \
  --train-subset 2000 \
  --val-subset 1000 \
  --calib-batches 10
```

这条命令会顺序输出：

- float32 基线
- PTQ int8 结果
- QAT int8 结果

当前这条量化线还采用一个明确原则：

- QAT 微调优先用 GPU
- 最终真实量化验证和 deploy benchmark 再回到 CPU

如果你想理解“为什么图像分类优先用 `PT2E`、什么时候该用 PTQ、什么时候该上 QAT、训练和部署设备为什么要分开”，直接看上面的工程化教程会更完整；如果你只想运行实验，看这个目录下的代码和 benchmark 即可。
