# torchao 量化路线总览与阅读指南

这份文档不再承担“把全部量化知识点一次讲完”的职责。

它现在只做两件事：

1. 讲清楚 `torchao` 当前主要有哪些量化路线
2. 告诉你这些路线应该怎么和前面的基础知识对应起来

如果你是初学者，**不要** 从 torchao API 名字开始学。

建议阅读顺序是：

1. [量化基础原理：从公式到手写实现](./quantization_fundamentals.md)
2. [量化工作流：PTQ、QAT、Observer 与 Calibration](./quantization_workflows.md)
3. 再回来看这份 torchao 路线总览
4. 最后读 [torchao PT2E 图像分类实战：对应本仓库实现](./torchao_pt2e_image_classification.md)

---

## 1. 先回答一个关键问题：torchao 官方文档到底怎么读

很多人第一次看 torchao 文档会有一个典型困惑：

- 官方文档里确实有量化内容
- 但分散在不同页面里
- 看完一个页面之后，还是不知道整体路线是什么

对初学者来说，可以先把 torchao 官方文档粗分成四类：

### 1.1 基础概念页

- `Quantization Overview`

它负责讲：

- affine quantization
- `scale / zero_point`
- weight-only / dynamic / static 等概念

### 1.2 `quantize_()` 路线

- `Quantized Inference`
- 独立的 `QAT` workflow 页面

这条线更偏：

- eager 量化
- `Linear` 主导模型
- weight-only
- dynamic activation + weight quantization

### 1.3 `PT2E` 路线

- `PT2E Quantization`
- 以及下面的 PTQ / QAT / x86 tutorial

这条线更偏：

- exported graph
- static quantization
- backend quantizer
- CPU / mobile int8 部署

### 1.4 backend 文档

例如：

- `ExecuTorch XNNPACK Quantization`

它负责回答：

- 某个 backend 到底支持什么量化配置
- 最终会落到什么 kernel 或运行时约束

---

## 2. torchao 当前主要有哪两条主路线

对当前稳定版文档来说，初学者可以先把 torchao 理解成两条主路线：

1. `quantize_()` eager 路线
2. `PT2E` 路线

### 2.1 路线 A：`quantize_()` eager 路线

这条线更偏：

- `Linear` 主导模型
- weight-only quantization
- dynamic activation + weight quantization
- LLM / Transformer 推理

常见配置名包括：

- `Int4WeightOnlyConfig`
- `Int8DynamicActivationInt8WeightConfig`
- `Int8DynamicActivationIntxWeightConfig`

如果你之前读过《量化基础原理》，可以这样把它和基础概念对上：

| 基础概念 | 在这条路线里的典型表现 |
| --- | --- |
| weight-only | 直接对应很多 `Int4WeightOnly...` 配置 |
| dynamic quantization | 常见于 `Int8DynamicActivation...` 配置 |
| per-group / per-token | 在 `Linear` / LLM 路线里非常常见 |

### 2.2 路线 B：`PT2E` 路线

可以先记成这 5 步：

1. `torch.export`
2. `prepare_pt2e` 或 `prepare_qat_pt2e`
3. calibration 或 fake-quant 训练
4. `convert_pt2e`
5. lowering 到 backend

这条线更偏：

- static quantization
- 全图量化
- backend pattern matching
- CNN / CPU int8 / mobile 部署

如果你之前读过《量化工作流》，可以这样把它和流程概念对上：

| 工作流概念 | 在这条路线里的位置 |
| --- | --- |
| observer | `prepare_pt2e` 后插入/管理 |
| calibration | PTQ 中间阶段 |
| fake quant | `prepare_qat_pt2e` 后的训练阶段 |
| convert | `convert_pt2e` |
| deploy | `torch.compile(inductor)`、ExecuTorch 等 lowering |

---

## 3. 初学者到底该先看哪条路线

这个问题不能靠“哪个 API 看起来更流行”来决定，而应该看模型结构和部署目标。

### 3.1 如果你做的是经典 CNN 图像分类

例如：

- `ResNet`
- `MobileNet`
- `EfficientNet`

并且目标更偏：

- CPU int8 部署
- static quantization

那优先看：

- `PT2E PTQ`
- `PT2E QAT`

### 3.2 如果你做的是 `Linear` 主导模型

例如：

- `ViT`
- `Transformer`
- `LLM`

并且目标更偏：

- weight-only
- dynamic activation quantization
- 大模型推理节省显存 / 内存

那优先看：

- `quantize_()`

### 3.3 一张表看懂怎么选

| 场景 | 优先路线 | 原因 |
| --- | --- | --- |
| CNN 图像分类，目标是 CPU int8 部署 | `PT2E` | 更贴近 static quantization 和 backend lowering |
| `Linear` 主导模型，目标是大模型推理优化 | `quantize_()` | 官方配置和教程更集中在这条线上 |
| 想最低成本验证量化是否可行 | 先 PTQ | 成本最低 |
| PTQ 掉点明显，必须继续追精度 | 再 QAT | 让模型适应量化误差 |

---

## 4. 为什么这个仓库当前主讲 `PT2E`

仓库当前可运行示例在：

- `cv/image_classification/quantization/`

它的任务背景是：

- 图像分类
- 经典 CNN
- 目标是 PTQ / QAT 对比和 CPU deploy benchmark

所以这里刻意优先讲：

- `PT2E PTQ`
- `PT2E QAT`

而不是把 `quantize_()` 当成默认主线。

这不是因为 `quantize_()` 不重要，而是因为：

- 它更偏 `Linear` 主导模型
- 当前仓库这条示例更适合拿 `PT2E` 讲清 static int8 工作流

---

## 5. 把 torchao API 和基础知识一一对上

如果你想看 API 时不发懵，最有用的是下面这张表。

| 你已经学过的基础概念 | torchao 里常见的对应位置 |
| --- | --- |
| `scale / zero_point` | quantizer 配置、quant primitives、observer 统计结果 |
| 对称/非对称量化 | backend quantizer 或 config 决定的 scheme |
| per-tensor / per-channel / per-group | quantizer / config / backend 支持能力 |
| weight-only | `quantize_()` 常见配置 |
| dynamic quantization | `Int8DynamicActivation...` 这类配置 |
| static quantization | `PT2E` 路线最常见 |
| observer | `prepare_pt2e` / `prepare_qat_pt2e` 后的准备态模型 |
| calibration | PTQ 流程中间阶段 |
| fake quant | QAT 训练阶段 |
| convert | `convert_pt2e` 或相关转换步骤 |

这张表想表达的核心只有一句话：

- **框架 API 不是新知识，它只是把前面那些基础概念组织成了工具接口。**

---

## 6. 当前仓库里最相关的 torchao 路线

如果你只想对照本仓库代码看，最相关的是下面两条。

### 6.1 `PT2E PTQ`

对应仓库代码：

- `cv/image_classification/quantization/quant_pt2e.py`
- `run_ptq(...)`

你可以把它理解成：

- float checkpoint
- `export`
- `prepare_pt2e`
- calibration
- `convert_pt2e`
- CPU benchmark

### 6.2 `PT2E QAT`

对应仓库代码：

- `cv/image_classification/quantization/quant_pt2e.py`
- `run_qat(...)`

你可以把它理解成：

- float checkpoint
- `export`
- `prepare_qat_pt2e`
- fake-quant 微调
- 关闭 observer / 冻结 BN
- `convert_pt2e`
- CPU benchmark

如果你现在想继续往下读，直接转到：

- [torchao PT2E 图像分类实战：对应本仓库实现](./torchao_pt2e_image_classification.md)

---

## 7. 看 torchao 文档时最容易混淆的几件事

### 7.1 把 `quantize_()` 和 `PT2E` 当成同一条路线

不对。

它们解决的问题、依赖的图表示、主场景都不一样。

### 7.2 看见 `QAT` 页面，就以为它一定对应当前仓库示例

也不对。

当前稳定版 torchao 的独立 `QAT` workflow 页面更偏：

- `quantize_()` eager 路线

而本仓库示例走的是：

- `PT2E QAT`

### 7.3 只看 API 名字，不看 backend 约束

也不对。

量化不是只看：

- 配置名字
- prepare / convert 名字

还必须看：

- backend 支持什么
- 最终 lowering 到哪里
- benchmark 口径如何解释

---

## 8. 当前文档和代码应该怎么一起读

推荐顺序：

1. 先读 [量化基础原理](./quantization_fundamentals.md)
2. 再读 [量化工作流](./quantization_workflows.md)
3. 回来读这份 torchao 总览
4. 最后看 [torchao PT2E 图像分类实战](./torchao_pt2e_image_classification.md)
5. 再对照 `cv/image_classification/quantization/` 下的代码

如果你一开始就想直接看代码，也建议至少先建立下面这两个概念：

- static、dynamic、weight-only 的差别
- PTQ、QAT、observer、fake quant 的差别

不然会很容易把代码里几个阶段混成一团。

---

## 9. 参考资料

以下是这份文档最值得顺着读的官方资料：

- `torchao` 文档首页：https://docs.pytorch.org/ao/stable/index.html
- Quantization Overview：https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html
- PT2E Quantization：https://docs.pytorch.org/ao/stable/pt2e_quantization/index.html
- PyTorch 2 Export Post Training Quantization：https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_ptq.html
- PyTorch 2 Export Quantization-Aware Training (QAT)：https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_qat.html
- PyTorch 2 Export Quantization with X86 Backend through Inductor：https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_x86_inductor.html
- Quantized Inference：https://docs.pytorch.org/ao/stable/workflows/inference.html
- Quantization-Aware Training (QAT)：https://docs.pytorch.org/ao/stable/workflows/qat.html
- ExecuTorch XNNPACK Quantization：https://docs.pytorch.org/executorch/1.0/backends/xnnpack/xnnpack-quantization.html
