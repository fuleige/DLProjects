# torchao 量化指南：PTQ、QAT 与图像分类落地

这份文档是仓库里的 **工程化专题教程**。

它不绑定某一个目录结构来讲 API，而是把 `torchao` 量化在项目里的常见决策、流程和坑点讲清楚；仓库里对应的具体实现入口是：

- 代码：`cv/image_classification/quantization/`
- 训练入口：`cv/image_classification/quantization/train.py`
- 复现实验：`cv/image_classification/quantization/run_benchmark.sh`

---

## 1. 先说结论：图像分类优先怎么选

如果你的任务是典型图像分类，模型主体是：

- `Conv2d + BatchNorm + ReLU + Linear`
- 例如 `ResNet`、`MobileNet`、`EfficientNet`

那在 `torchao` 里更推荐的主路线是：

- `PTQ`：`torchao.quantization.pt2e.prepare_pt2e`
- `QAT`：`torchao.quantization.pt2e.prepare_qat_pt2e`

也就是当前更稳妥的 `PT2E` 路线。

一个实用速查表：

| 场景 | 更推荐的路线 | 原因 |
| --- | --- | --- |
| CNN 图像分类，目标是 CPU int8 部署 | `PT2E PTQ / PT2E QAT` | 更贴合 `Conv2d` 主导模型和当前教程覆盖范围 |
| `ViT / Swin / MLP-Mixer` 这类 `Linear` 占主导模型 | `quantize_()` 或 weight-only 路线 | `quantize_()` 的主战场更偏 `Linear` |
| 先验证是否能量化落地 | 先跑 `PTQ` | 成本最低 |
| PTQ 掉点不可接受，但部署必须上 int8 | 再补 `QAT` | 通过 fake quant 微调追回精度 |

不要把 `quantize_()` 当成经典 CNN 的默认方案。对标准图像分类 CNN，优先顺序通常是：

1. `float baseline`
2. `PTQ`
3. `QAT`

---

## 2. 三种方案怎么理解

### 2.1 float32 基线

所有量化实验都应该从稳定的 float 模型出发。没有可信的基线，后面对比就没有意义。

### 2.2 PTQ

典型流程：

1. 训练好 float 模型
2. `export`
3. `prepare_pt2e`
4. `calibration`
5. `convert_pt2e`

特点：

- 不需要再训练
- 成本低
- 适合作为第一步部署验证

### 2.3 QAT

典型流程：

1. 从已有 float checkpoint 出发
2. `export`
3. `prepare_qat_pt2e`
4. fake quant 微调
5. `convert_pt2e`

特点：

- 更容易追回 int8 精度
- 训练成本更高
- 需要处理 observer 和 BatchNorm 冻结

---

## 3. 这条线在仓库里怎么落地

仓库里当前的落地示例放在：

- `cv/image_classification/quantization/`

为什么没有直接在 `cv/image_classification/example.py` 上硬扩展：

- `torchao PT2E` 图像分类教程更适合先用标准 CNN 讲清流程
- 任意 `timm` 模型的 `export` 稳定性不一样
- 教学上先把量化链路跑通，比先讨论 backbone 兼容性更重要

当前示例默认支持：

- `resnet18`
- `mobilenet_v3_small`

---

## 4. 环境与 backend

至少需要：

```bash
pip install torch torchvision torchao
```

如果要跑 `xnnpack`：

```bash
pip install executorch
```

当前仓库里推荐先用：

- `--backend x86_inductor`

原因是：

- 依赖更少
- 在普通 x86 CPU 上更容易先跑通

然后再考虑：

- `--backend xnnpack`

需要特别区分一件事：

- `x86_inductor`：当前脚本会给出 `torch.compile(inductor)` 的 deploy 延迟
- `xnnpack`：当前脚本主要记录 eager CPU 延迟；如果要真实 `xnnpack/ExecuTorch` 部署数据，需要到对应运行时里单独测

---

## 5. 训练设备和部署设备不要混为一谈

这是最容易混掉的一点。

当前仓库采用的是更实用的原则：

- float 训练：优先用 GPU
- QAT fake-quant 微调：优先用 GPU
- 最终真实量化验证和 deploy benchmark：回到 CPU

也就是说：

- `QAT` 训练设备可以是 GPU
- 但最终对比表里的部署指标，仍然应该按 CPU 口径来解释

在仓库实现里，训练期和最终部署验证已经拆开：

- 训练中间：GPU 上做 fake-quant 验证，选 best checkpoint
- 最终汇总：把 best prepared state 转到 CPU，`convert_pt2e` 后再做真实量化验证和 CPU benchmark

---

## 6. 仓库里已经补好的工程化能力

当前实现不仅是 API demo，还补了这些实战细节：

- 训练集、校准集、验证集三套视角分离
- QAT 默认优先走 GPU 微调
- 最终 CPU 量化验证单独执行
- 结果自动落盘为 `json / csv / md`
- CPU benchmark 同时记录延迟和 RSS 增量
- 训练/微调阶段记录峰值 CUDA 显存
- `run_benchmark.sh` 提供 `smoke / formal` 两套预设
- 代码已经从单文件拆成多文件，职责按参数、训练、量化、benchmark 分离

---

## 7. 最常用的运行方式

### 7.1 先看支持的模型

```bash
python3 cv/image_classification/quantization/train.py --list-models
```

### 7.2 快速烟测

```bash
bash cv/image_classification/quantization/run_benchmark.sh smoke
```

### 7.3 正式 benchmark

```bash
bash cv/image_classification/quantization/run_benchmark.sh formal
```

如果你想显式指定 QAT 走 GPU：

```bash
python3 cv/image_classification/quantization/train.py \
  --mode compare \
  --dataset-type cifar100 \
  --model-name resnet18 \
  --backend x86_inductor \
  --qat-device cuda
```

---

## 8. 结果应该怎么解读

通常有几种典型情况：

### 8.1 PTQ 和 float 差距很小

说明模型对 static int8 比较友好。通常优先落地 PTQ。

### 8.2 PTQ 掉点明显，QAT 追回来

这是最典型的“QAT 值得上”的情况。

### 8.3 QAT 精度更好，但 CPU deploy 速度并没有明显变快

这也完全正常。

因为：

- 量化训练策略
- 最终 CPU backend kernel 命中

不是同一件事。

所以项目里更现实的判断顺序是：

1. 先看 PTQ 是否够用
2. 如果不够，再看 QAT 是否值得为精度或稳定性付出训练成本
3. 不要默认假设 QAT 一定更快

---

## 9. 常见报错和排查

### 9.1 `No module named torchao`

说明环境里没有装 `torchao`，或者装错了环境。

### 9.2 `No module named executorch`

通常是你切了 `xnnpack`，但没装 `executorch`。

### 9.3 `torch.export` 失败

优先检查：

- 当前模型能不能被 `torch.export`
- 前向里是否有过于动态的 Python 控制流
- batch 动态约束是否和当前设置冲突

### 9.4 PTQ 掉点太多

优先检查：

- calibration 样本是否代表真实输入
- 预处理是否与推理一致
- float 基线本身是否稳定

### 9.5 QAT 后期不稳定

优先检查：

- 学习率是否过大
- observer 是否及时关闭
- BatchNorm 统计量是否及时冻结

---

## 10. 当前仓库里的组织建议

`torchao` 教程本身更像工程化专题，因此放在：

- `docs/model_compression/torchao_quantization_guide.md`

而具体可运行示例继续放在：

- `cv/image_classification/quantization/`

这样更清楚：

- `docs/` 负责解释“方法和工程决策”
- `cv/` 负责承载“具体任务落地”

如果以后仓库里出现更多跨任务复用的量化公共组件，再进一步抽到 `tooling/model_compression/` 会更自然。

---

## 11. 参考资料

- PT2E 总览：https://docs.pytorch.org/ao/stable/pt2e_quantization/index.html
- 图像分类 PTQ 教程：https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_ptq.html
- 图像分类 QAT 教程：https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_qat.html
- `quantize_()` API：https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.quantize_.html
- `QATConfig` API：https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.qat.QATConfig.html
- PyTorch 量化迁移说明：https://docs.pytorch.org/docs/stable/quantization.html
