# torchao 量化实验记录

这份记录对应 `cv/image_classification/quantization/train.py` 在当前机器上的一次正式对比，目标是验证：

- float32 基线
- PTQ
- QAT

三种方式在同一套图像分类任务上的精度、部署延迟和资源占用差异。

---

## 1. 实验环境

- 日期：`2026-04-16`
- GPU：`NVIDIA RTX A6000`
- `torch / torchvision / torchao`：`2.11.0+cu130 / 0.26.0+cu130 / 0.17.0`
- backend：`x86_inductor`
- 说明：当前环境未安装 `executorch`，所以没有额外补跑 `xnnpack`

---

## 2. 实验设置

- 模型：`resnet18`
- 数据集：`CIFAR-100`
- 训练子集：`10000`
- 验证子集：`2000`
- float 训练：`3 epochs`
- QAT 微调：`1 epoch`，设备为 `cuda`
- batch size：`128`
- calibration：`20 batches`

float 基线先单独训练完成，然后在同一份 checkpoint 上继续做 PTQ / QAT 对比。
其中：

- float 训练：走 GPU
- QAT fake-quant 微调：走 GPU
- 最终真实量化验证和部署 benchmark：走 CPU

实际运行命令：

```bash
python3 cv/image_classification/quantization/train.py \
  --mode compare \
  --dataset-type cifar100 \
  --data-root ./datasets \
  --model-name resnet18 \
  --float-epochs 3 \
  --qat-epochs 1 \
  --train-subset 10000 \
  --val-subset 2000 \
  --batch-size 128 \
  --num-workers 4 \
  --calib-batches 20 \
  --benchmark-warmup 5 \
  --benchmark-iters 10 \
  --backend x86_inductor \
  --float-device cuda \
  --output-dir ./outputs/image_classification/torchao_quantization_trial \
  --learning-rate 5e-4 \
  --qat-device cuda
```

原始输出文件位于：

- `outputs/image_classification/torchao_quantization_trial/resnet18/cifar100_pretrained/benchmark_results.json`
- `outputs/image_classification/torchao_quantization_trial/resnet18/cifar100_pretrained/benchmark_results.csv`
- `outputs/image_classification/torchao_quantization_trial/resnet18/cifar100_pretrained/benchmark_results.md`

---

## 3. 结果表格

| Method | Top1 | Top5 | Eager(ms) | Deploy(ms) | Speedup vs Float | RSS Delta(MB) | Train Peak CUDA(MB) | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| float32 | 0.6900 | 0.9215 | 1006.52 | 457.10 | 1.00x | 75.10 | 910.06 | 作为精度和速度基线 |
| ptq_int8 | 0.6900 | 0.9215 | 816.14 | 420.29 | 1.09x | 25.77 | - | 这组数据上几乎无精度损失，部署延迟有小幅改善 |
| qat_int8 | 0.7025 | 0.9275 | 1171.86 | 451.45 | 1.01x | 48.00 | 4224.48 | 精度最好，但这次配置下几乎没有拿到额外 CPU deploy 加速 |

补充说明：

- `Deploy(ms)` 是 `torch.compile(inductor)` 之后的单批平均延迟，适合作为最终部署速度主指标。
- `RSS Delta(MB)` 是推理 benchmark 期间采样到的 CPU RSS 增量，比绝对 RSS 更适合做横向比较。
- `Train Peak CUDA(MB)` 记录训练或微调阶段的峰值 GPU 显存。PTQ 没有训练阶段，所以该列为空。

---

## 4. 结果解读

这次测试里最重要的结论有三点：

1. `PTQ` 在这组 `ResNet18 + CIFAR-100` 设置上几乎没有带来可见的精度损失，说明这类 CNN 对 static int8 比较友好。
2. `QAT` 放到 GPU 做 `1` 个 epoch 微调后，`Top1` 从 `0.6900` 提到了 `0.7025`，说明这条线确实可以把精度追回来，甚至略微超过当前 float 基线。
3. 如果当前目标是“先低成本落地”，优先顺序仍然应该是 `float baseline -> PTQ -> 再看是否有必要上 QAT`。

这里也能看出一个很现实的问题：

- QAT 不一定天然比 PTQ 更快
- 即使精度追回来了，最终 CPU deploy 延迟也不一定同步变好
- 量化训练策略和最终 CPU backend kernel 命中，不是同一件事

所以在项目落地里，QAT 更像是“在 PTQ 基础上继续追精度或稳定性”的第二阶段，而不是默认第一选择。

---

## 5. 为什么这份表不直接对比 GPU 量化速度

因为这个目录当前实现的是：

- `torchao PT2E PTQ`
- `torchao PT2E QAT`
- `x86_inductor / xnnpack`

它们主打的是 **CPU / mobile int8 部署链路**，不是 A6000 上的 GPU 低精度推理。

所以这里把资源对比分成了两类：

- 训练阶段：看 float / QAT 的峰值 CUDA 显存
- 部署阶段：看 CPU deploy 延迟和 CPU RSS

如果后面要补 GPU 版量化或低精度对比，更合理的路线会是：

- `float16`
- `bfloat16`
- `float8`
- 或 GPU 友好的 weight-only / dynamic activation 方案

而不是直接拿这套 static int8 CNN 模板去代表 GPU 部署。
