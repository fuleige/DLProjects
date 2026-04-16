# torchao 量化指南：从基础理论到 PTQ / QAT 图像分类落地

这份文档是仓库里的 **工程化专题教程**。

它不只讲“命令怎么跑”，而是把下面三件事串起来：

- 量化基础理论到底在讲什么
- `torchao` 里这些理论分别落在哪条 API 路线上
- 本仓库的图像分类示例是如何把理论落成可运行代码的

仓库里对应的实现入口是：

- 教程文档：`docs/model_compression/torchao_quantization_guide.md`
- 代码目录：`cv/image_classification/quantization/`
- 训练入口：`cv/image_classification/quantization/train.py`
- 复现实验：`cv/image_classification/quantization/run_benchmark.sh`

---

## 1. 先回答一个问题：`torchao` 官方文档有没有讲量化基础

有，但目前是 **分散讲**，不是一篇“从零讲完”的长教程。

大致可以这样理解：

- `Quantization Overview`
  - 讲 `torchao` 的量化栈、affine quantization、`scale / zero_point`、weight-only / dynamic / static 等基本概念
- `PT2E Quantization`
  - 讲 `torch.export -> prepare -> convert -> lowering` 这条 **静态量化 / 全图量化** 主路线
- `Quantized Inference`
  - 讲 `quantize_()` 这条 **eager 推理量化** 路线，重点覆盖 `Linear` 的 weight-only / dynamic quantization
- `QAT`
  - 讲 fake quantization、`prepare` / `convert`、训练期如何模拟量化数值行为；但最新这页更偏 `quantize_()` 的 `Linear` / LLM 路线
- `PT2E Quantization` 下的 PTQ / QAT 子教程
  - 更直接对应 exported graph、backend quantizer、static quantization 这条路线

也就是说：

- 官方文档 **有基础理论**
- 但组织方式更偏“按子系统拆开讲”
- 如果你是第一次系统学量化，容易知道 API 名字，却没把这些概念连成一张图，也容易把 `quantize_()` 的 eager QAT 和 `PT2E` 的 exported-graph QAT 混在一起

这份仓库文档的目标，就是把这些内容重新按“理论 -> 路线 -> 代码 -> 结果解释”的顺序整理一遍。

---

## 2. 量化到底在做什么

### 2.1 为什么量化

量化的核心目标通常有三个：

1. 降低模型存储和内存占用
2. 降低内存带宽压力
3. 在合适的硬件和 kernel 上加速推理

例如：

- `float32` 每个元素 4 字节
- `int8` 每个元素 1 字节
- `int4` 每个元素约 0.5 字节

所以量化常常先带来：

- 更小的模型
- 更低的内存读写成本

至于速度能否同步提升，还要看：

- 硬件是否支持对应低精度算子
- backend 是否真的命中量化 kernel
- 模型结构是否适合该量化方案

这也是为什么“量化了”不等于“一定更快”。

### 2.2 最基础的数学形式

最常见的是 affine quantization，可以先记住这两步：

```text
q = clamp(round(x / scale + zero_point), qmin, qmax)
x_hat = (q - zero_point) * scale
```

其中：

- `x`：原始高精度值，通常是 `float32 / float16 / bfloat16`
- `q`：量化后的低精度整数值，例如 `int8`
- `x_hat`：反量化后的近似值
- `scale`：把实数区间映射到整数区间的缩放因子
- `zero_point`：让整数 0 对应到某个实数位置的偏移量
- `qmin / qmax`：目标 dtype 的可表示范围

直觉上可以这样理解：

- `scale` 决定“一个整数台阶代表多少实数幅度”
- `zero_point` 决定“实数 0 在整数坐标系里落在哪里”

#### `qmin / qmax` 常见取值

最常见的几组范围是：

- `int8`：`[-128, 127]`
- `uint8`：`[0, 255]`
- `int4`：通常可理解为 `[-8, 7]` 或实现相关变体

但在工程里不要把“某个 dtype 的数值范围”直接等同于“这个 backend 一定这么用”。

真正决定实际行为的还有：

- backend 约束
- quantizer 配置
- weight 和 activation 分别采用什么 scheme

所以正确顺序是：

1. 先理解通用数值范围
2. 再看具体 backend 文档和 quantizer 配置
3. 最后再解释当前模型里的真实量化行为

### 2.3 量化误差从哪里来

量化误差主要有两类：

- 舍入误差：`round(...)`
- 截断误差：`clamp(...)`，也就是超出可表示范围后被裁掉

因此量化的本质不是“无损转换”，而是：

- 用更少 bit 表示原来的值
- 接受一定数值误差
- 换取更小的存储和更高的部署效率

如果某一层激活值分布跨度很大、长尾明显，或者某些通道特别敏感，那么量化误差就更容易放大成精度下降。

### 2.4 `scale` 和 `zero_point` 怎么理解

这是量化里最重要的一组参数。

#### `scale`

`scale` 越小：

- 同样一个整数步长对应的实数间隔越细
- 精度更细
- 但可覆盖的实数范围更窄

`scale` 越大：

- 覆盖范围更宽
- 但离散化更粗

#### `zero_point`

`zero_point` 的意义是让量化后的整数空间更好地对齐原始实数分布。

如果数据分布以 0 为中心且比较对称，通常更适合：

- symmetric quantization

如果数据分布不对称，例如激活值大多非负或存在明显偏移，通常更适合：

- asymmetric quantization

### 2.5 对称量化和非对称量化

#### symmetric quantization

常见特征：

- `zero_point = 0` 或接近 0
- 实现更简单
- 对权重量化很常见

适合：

- 权重分布相对对称的层
- 希望 kernel 更简单、部署更直接的场景

#### asymmetric quantization

常见特征：

- `zero_point` 不一定为 0
- 能更好利用不对称整数范围
- 激活量化里很常见

适合：

- 激活值分布明显偏正或偏移的数据

经验上：

- 权重经常用 symmetric
- 激活经常用 asymmetric 或动态计算量化参数

#### 一个最小数值例子

先看一个最典型的 symmetric 权重量化例子。

假设某层权重范围大致在：

- `[-1.0, 1.0]`

目标量化到 `int8`，并采用 symmetric quantization：

- `qmin = -127`
- `qmax = 127`
- `zero_point = 0`
- `scale = 1.0 / 127`

如果某个 float 权重是：

- `x = 0.30`

那么：

```text
q = round(0.30 / (1/127)) = round(38.1) = 38
x_hat = 38 * (1/127) = 0.2992
```

你可以看到：

- 量化后的整数值是 `38`
- 反量化回来的值不是严格 `0.30`
- 而是一个非常接近的近似值 `0.2992`

再看一个真正带非零 `zero_point` 的 asymmetric 例子。

假设某层激活范围大致在：

- `[-1.0, 5.0]`

目标量化到 `uint8`：

- `qmin = 0`
- `qmax = 255`
- `scale = (5.0 - (-1.0)) / 255 = 6.0 / 255`
- `zero_point = round((0 - (-1.0)) / scale) = round(42.5) = 43`

若激活值：

- `x = 1.0`

则：

```text
q = round(1.0 / (6/255) + 43) = round(85.5) = 86
x_hat = (86 - 43) * (6/255) = 1.0118
```

这个例子想说明的不是“这个值多精确”，而是三件事：

- 量化一定会引入近似误差
- 分布越贴近量化区间，可用整数桶就越充分
- 为什么 activation 的范围估计会直接影响最终精度

### 2.6 量化粒度：per-tensor、per-channel、per-group、per-token

量化不是只能“一整个 Tensor 用一组参数”。

粒度越细，通常精度越好，但实现和存储也更复杂。

#### per-tensor

整块 Tensor 共用一组 `scale / zero_point`。

特点：

- 最简单
- 元信息最少
- 但对分布差异大的通道不够友好

#### per-channel

每个通道各自一组量化参数。

特点：

- 对卷积和线性层权重很常见
- 往往比 per-tensor 更稳

在 CNN 里，权重 per-channel 是非常常见的选择。

#### per-group

把通道再切成若干 group，每个 group 一组参数。

特点：

- 比 per-channel 更灵活
- 在 `int4` / `int8` weight-only 场景里很常见
- 是很多 LLM 权重量化配置的常见折中

#### per-token / per-row

激活在运行时按 token 或按行动态计算量化参数。

特点：

- 常见于 Transformer / LLM 的 `Linear`
- 更偏动态激活量化路线

可以把它们理解成：

- CNN static int8：更常见 `per-channel weight + static activation`
- LLM / Transformer：更常见 `per-token activation + per-group/per-channel weight`

### 2.7 权重量化、动态量化、静态量化分别是什么

这是初学者最容易混的地方。

#### weight-only quantization

只量化权重，输入激活保持较高精度。

特点：

- 工程成本最低
- 对模型改动小
- 很适合 `Linear` 主导模型，尤其是大模型推理

典型收益：

- 模型更小
- 显存 / 内存占用下降明显

#### dynamic quantization

权重预先量化，激活在运行时动态计算量化参数。

特点：

- 不需要离线校准激活统计
- 对输入分布变化更鲁棒
- 常用于 `Linear`

#### static quantization

权重和激活都量化，激活参数通常依赖校准阶段预先估计。

特点：

- 更贴近端到端 int8 部署
- 对 backend 和图模式要求更高
- 对校准数据质量很敏感

CNN 的 CPU int8 部署里，static quantization 非常常见。

### 2.8 PTQ 和 QAT 的区别

#### PTQ：Post-Training Quantization

先训练好 float 模型，再量化。

优点：

- 成本最低
- 不需要再训练或只需要很少额外流程
- 最适合作为第一步可行性验证

缺点：

- 某些模型会明显掉点
- 对校准质量比较敏感

#### QAT：Quantization-Aware Training

在训练或微调阶段就把量化数值行为模拟进去，让模型提前适应量化误差。

优点：

- 通常比 PTQ 更容易保精度
- 对敏感模型更有效

缺点：

- 成本更高
- 训练流程更复杂
- 需要处理 observer、BatchNorm 冻结、学习率等稳定性问题

一个非常实用的顺序是：

1. 先拿到稳定的 float baseline
2. 先做 PTQ
3. PTQ 不够用，再补 QAT

### 2.9 calibration、observer、fake quant 是什么

#### calibration

校准就是拿一批 **有代表性的真实输入** 跑一遍模型，用来估计激活分布，从而得到更合适的量化参数。

重点不是“越多越好”，而是：

- 样本分布要接近真实部署数据
- 预处理要和推理一致

#### observer

observer 可以理解成“统计分布的记录器”。

它通常负责记录：

- min/max
- 或更复杂的统计量

再据此计算：

- `scale`
- `zero_point`

在 static PTQ 里，observer 主要用于校准阶段。

在 QAT 里，observer 往往前期更新统计，后期关闭，避免训练后期量化范围继续漂移。

#### 常见 observer 思路

从思想上看，observer 大致有这几类：

- min/max
  - 最简单，直接记录范围
  - 实现便宜，但容易受 outlier 影响
- moving-average min/max
  - 更适合训练过程中逐步更新统计
  - QAT 里很常见
- histogram / clipping / percentile
  - 会尝试更稳妥地处理长尾和离群点
  - 更复杂，但某些模型上更稳

本仓库当前没有把 observer 类型做成 CLI 参数暴露给用户，而是：

- 交给 backend quantizer 的默认配置

所以这里更重要的是先建立概念：

- observer 的任务是“估计范围”
- calibration / QAT 的任务是“让这个范围估计足够可信”

#### fake quant

fake quant 的关键点是：

- 训练时仍然用 float Tensor 参与计算和反向传播
- 但前向数值会模拟“量化 -> 反量化”的效果

也就是：

- 参数和激活在训练时“看起来像被量化过”
- 但并没有真的全程改成低 bit 整数张量去训练

这正是 QAT 的核心。

#### calibration 选样本时最容易犯的错

最常见的错误有四类：

- 拿训练增强后的样本直接做 calibration
- 样本太少，只覆盖到很窄的输入分布
- 样本很多，但和真实部署分布不一致
- 预处理链路和真实推理不一致

这也是为什么本仓库会特意把：

- `train_dataset`
- `calib_dataset`

拆成两个不同 transform 的视角。

训练时你可以用：

- 随机裁剪
- 随机翻转
- 其他增强

但 calibration 时更应该贴近：

- 真实推理预处理
- 稳定评估态输入

---

## 3. `torchao` 里这些概念分别落在哪

### 3.1 `torchao` 当前主要有两条量化主路线

#### 路线 A：`quantize_()` eager 路线

更偏：

- `Linear` 主导模型
- weight-only quantization
- dynamic activation + weight quantization
- 现代大模型推理

这条路线在 `torchao` 当前官方文档里覆盖得很完整，`Quantized Inference` 和 `QAT` 页面都重点围绕它来展开。

常见例子包括：

- `Int4WeightOnlyConfig`
- `Int8DynamicActivationInt8WeightConfig`
- `Int8DynamicActivationIntxWeightConfig`

如果你的模型主要瓶颈是 `Linear`，这通常是第一考虑对象。

需要补一句：

- 最新 `torchao` 的独立 `QAT` workflow 页面，确实主要是按 `quantize_()` 这条 eager 路线来讲的
- 本仓库用到的 `PT2E QAT`，更应该对照 `PT2E Quantization` 目录下的 QAT 子教程来理解

#### 路线 B：`PT2E` 路线

也就是：

1. `torch.export`
2. `prepare_pt2e` 或 `prepare_qat_pt2e`
3. calibration 或 fake-quant 训练
4. `convert_pt2e`
5. lowering 到具体 backend

这条路线更偏：

- static quantization
- 全图量化
- backend pattern matching
- CPU / mobile int8 部署

### 3.2 为什么这个仓库对图像分类优先讲 `PT2E`

这是结合 `torchao` 官方文档现状和本仓库目标做出的工程判断：

- 当前仓库示例是 `ResNet / MobileNet` 这类经典 CNN
- 目标是图像分类的 CPU int8 部署链路
- 这类场景更贴近 static quantization + backend lowering

所以这里不把 `quantize_()` 当成默认主角，而是优先讲：

- `PT2E PTQ`
- `PT2E QAT`

这不是说 `quantize_()` 不重要，而是说：

- 它当前更像 `Linear` 主导模型的主战场
- 本仓库这个例子更适合拿 `PT2E` 讲清 static int8 的完整流程

### 3.3 一张表看懂怎么选

| 场景 | 更推荐的路线 | 原因 |
| --- | --- | --- |
| CNN 图像分类，目标是 CPU int8 部署 | `PT2E PTQ / PT2E QAT` | 更贴近 static quantization 和 backend lowering |
| `ViT / LLM / MLP` 这类 `Linear` 占主导模型 | `quantize_()` | 官方文档与配置更集中在这条线上 |
| 想先验证“能不能量化落地” | 先 `PTQ` | 成本最低 |
| PTQ 精度掉点不能接受 | 再补 `QAT` | 让模型适应量化误差 |
| 只想先缩模型和省显存 / 内存 | 先看 weight-only | 落地成本通常更低 |

---

## 4. `PT2E` 的完整心智模型

把 `PT2E` 先看成五步：

### 4.1 Step 1: `export`

先把 eager 模型导出成更稳定的图表示。

目的：

- 把模型计算图显式化
- 让后续量化 pass 和 backend lowering 更容易做模式识别

在本仓库里，对应：

- `cv/image_classification/quantization/quant_pt2e.py`
- `export_with_dynamic_batch(...)`

### 4.2 Step 2: `prepare`

对导出的图插入量化相关节点或统计逻辑。

PTQ 对应：

- `prepare_pt2e`

QAT 对应：

- `prepare_qat_pt2e`

这一步的本质是：

- 告诉系统“哪些层、按什么规则量化”
- 为后续 calibration 或 fake quant 做准备

### 4.3 Step 3: calibration 或 QAT 训练

如果是 PTQ：

- 跑校准数据，收集激活统计

如果是 QAT：

- 继续训练 / 微调，让模型适应 fake quant 数值行为

### 4.4 Step 4: `convert`

把准备态模型转换成真实量化模型：

- `convert_pt2e`

这一步之后，模型已经不再只是“收集统计”或“模拟量化”，而是进入真实量化表示。

### 4.5 Step 5: lowering / deploy

最后还要看目标 backend：

- `torch.compile(inductor)`
- `ExecuTorch`
- 其他后端

这一步决定：

- 是否真的融合成量化 kernel
- 最终部署延迟如何

所以量化工作流不能只看 `convert`，还必须看：

- 目标硬件
- 目标 backend
- 实际 benchmark

### 4.6 用一张“伪图”理解 PT2E 到底改了什么

下面这不是精确 IR，只是帮助你建立心智模型。

#### float eager 图

```text
input_fp32
  -> Conv
  -> BatchNorm
  -> ReLU
  -> Linear
  -> logits_fp32
```

#### `prepare_pt2e` 之后

```text
input_fp32
  -> observer
  -> Conv
  -> BatchNorm / fused pattern
  -> ReLU
  -> observer
  -> Linear
  -> observer
  -> logits_fp32
```

这一步的关键变化是：

- 图里插入了 observer 或 fake quant 相关逻辑
- 量化 pass 开始“知道”哪些地方将来要量化
- 在典型 CNN static PTQ 教程里，`prepare_pt2e` 还会对 `Conv + BatchNorm` 做折叠

#### `convert_pt2e` 之后

```text
input_fp32
  -> quantize
  -> int8 kernel / quantized pattern
  -> dequantize or fused boundary
  -> logits_fp32
```

这一步开始，模型已经不再只是“观察范围”，而是进入真实量化表示。

最后到了 lowering / deploy 阶段，还会继续发生：

- pattern matching
- kernel 替换
- backend 专属融合

所以很多人第一次看 PT2E 会困惑：

- 为什么 `prepare`
- `convert`
- `compile / lower`

要分开？

因为它们分别解决的是三件不同的事：

- `prepare`：告诉图“哪里要量化、如何收集信息”
- `convert`：把准备态图变成真实量化图
- `lower / compile`：把量化图真正映射到后端可执行 kernel

---

## 5. 这条线在仓库里怎么落地

### 5.1 目录职责

当前落地示例放在：

- `cv/image_classification/quantization/`

主要文件分工：

- `train.py`
  - 总入口，串起 float / PTQ / QAT / compare
- `quant_args.py`
  - 参数定义和实验配置
- `quant_core.py`
  - 数据、模型、训练、验证、checkpoint
- `quant_pt2e.py`
  - `export`、quantizer、PTQ / QAT 主流程
- `quant_benchmark.py`
  - CPU benchmark、部署 benchmark、结果落盘
- `run_benchmark.sh`
  - `smoke / formal` 一键复现实验

### 5.2 为什么不用“任意 backbone + 任意模型导出”起手

这里故意先选：

- `resnet18`
- `mobilenet_v3_small`

原因是：

- 标准 CNN 更适合作为 static int8 教学模板
- `Conv2d + BatchNorm + ReLU + Linear` 结构更容易解释量化链路
- 先把量化流程跑通，比先讨论任意 backbone 的导出兼容性更重要

### 5.3 数据集为什么拆成训练 / 校准 / 验证三套视角

这是很多教程省略、但工程上很重要的一步。

- 训练集：float 训练和 QAT 微调
- 校准集：PTQ 统计激活分布
- 验证集：统一做最终精度比较

这样可以避免：

- 拿验证集直接充当校准集导致评估口径混乱
- 校准样本和真实部署样本分布不一致却没意识到

还要注意一个和当前实现强相关的细节：

- 这里的“训练 / 校准 / 验证三套视角”不是说一定有三份互斥数据
- 当前代码里，`train_dataset` 和 `calib_dataset` 默认都来自训练集，只是 transform 不同
- `train_dataset` 用训练增强
- `calib_dataset` 用评估态预处理

也就是说，当前仓库强调的是：

- 训练视角
- 校准视角
- 验证视角

而不是“必须先手工切出第三份独立 calibration split”。

另外，当前 CLI 还有一个容易忽略的行为：

- `--train-subset` 会同时裁剪 `train_dataset` 和 `calib_dataset`
- 当前并没有独立的 `--calib-subset`

所以如果你把 `--train-subset` 调得很小，PTQ 的校准覆盖范围也会一起变小。

### 5.4 理论和当前实现是一一怎么对上的

下面这张表是当前仓库最值得直接对照代码看的部分：

| 理论概念 | 当前代码位置 | 当前实现含义 |
| --- | --- | --- |
| static quantization | `quant_pt2e.py` 里的 `build_quantizer(...)` | `x86_inductor` 路径会在支持时显式传 `is_dynamic=False`，说明当前示例走的是 static 路线，不是 dynamic 路线 |
| per-channel 倾向 | `build_quantizer(...)` 的 `xnnpack` 分支 | 如果 API 支持，会显式传 `is_per_channel=True`，说明当前实现优先选择更稳妥的 per-channel 权重量化 |
| calibration 依赖代表性样本 | `build_datasets(...)` + `calibrate(...)` | 校准数据来自训练集视角，但使用评估态 transform，并在 CPU 上跑 observer 统计 |
| QAT 后期稳定化 | `disable_observer_if_supported(...)`、`freeze_bn_stats_in_exported_graph(...)` | 通过 CLI epoch 开关控制 observer 关闭和 BN 统计冻结 |
| 训练设备和部署设备拆开 | `run_qat(...)` + `benchmark_deploy_inference(...)` | QAT 可以在 GPU 微调，但最终 `convert_pt2e` 后统一回 CPU 做真实 benchmark |
| deploy benchmark 口径 | `quant_benchmark.py` | 当前只有 `x86_inductor` 会给出 `torch.compile(inductor)` 的 deploy 延迟，`xnnpack` 不会在这个脚本里直接产出移动端真实时延 |

---

## 6. PTQ 在这个仓库里是怎么实现的

典型流程就是：

1. 训练或加载 float checkpoint
2. `export`
3. `prepare_pt2e`
4. calibration
5. `convert_pt2e`
6. CPU 验证与 benchmark

### 6.1 对应代码入口

主要在：

- `cv/image_classification/quantization/train.py`
- `cv/image_classification/quantization/quant_pt2e.py`

具体函数是：

- `run_ptq(...)`

### 6.2 export 阶段

仓库里使用：

```python
exported_model = export_with_dynamic_batch(
    float_model,
    example_inputs,
    min_batch=1,
    max_batch=args.batch_size,
)
```

这里做的事情是：

- 用 `torch.export` 拿到图
- 给 batch 维加动态约束
- 为后续后端 lowering 提供更稳定的图表示

### 6.3 quantizer 阶段

当前支持两个 backend：

- `x86_inductor`
- `xnnpack`

在代码里由 `build_quantizer(...)` 负责构造。

其中：

- `x86_inductor`
  - 更容易在普通 x86 CPU 上先跑通
- `xnnpack`
  - 更贴近 mobile / edge 场景
  - 需要额外安装 `executorch`

如果你想把“理论里的量化配置”映射到当前仓库实现，最关键的是两点：

#### `x86_inductor` 当前明确走 static 量化

当前代码会在 API 支持时传：

```python
kwargs["is_dynamic"] = False
```

这意味着当前这条示例不是在讲：

- dynamic activation quantization

而是在讲：

- static PTQ / static QAT

按官方 `X86InductorQuantizer` 教程，默认配置本身就是：

- 8-bit activations
- 8-bit weights

所以当前仓库这条示例，你可以先把它理解成：

- CNN 的 static int8 基线模板

#### `xnnpack` 的配置名容易让人误解

当前代码里用的是：

```python
get_symmetric_quantization_config(...)
```

但这不应该被粗暴理解成“所有东西都 symmetric”。

结合 `ExecuTorch XNNPACK` 官方文档，当前 XNNPACK 支持的主方案是：

- 8-bit symmetric weights
- 8-bit asymmetric activations

并且：

- 支持 static 和 dynamic activations
- 支持 per-channel 和 per-tensor
- 不支持 weight-only quantization

所以这里更准确的理解是：

- backend 会暴露一个后端可接受的量化配置入口
- 配置函数名只是入口名
- 真正落地到 activation / weight 的量化形式，还要看该 backend 的文档约束

### 6.4 calibration 阶段

当前实现里：

```python
prepared_model = prepare_pt2e(exported_model, quantizer)
calibrate(prepared_model, calib_loader, args.calib_batches)
quantized_model = convert_pt2e(prepared_model)
```

注意两个原则：

- calibration 要在 **CPU 上按真实预处理** 跑
- 校准样本的“代表性”通常比单纯堆数量更重要

### 6.5 最终评估为什么回到 CPU

因为这条线的目标是：

- CPU / mobile int8 部署

所以仓库里会在 `convert_pt2e` 之后做：

- CPU 精度评估
- CPU eager benchmark
- CPU deploy benchmark

这比在训练 GPU 上看一个“伪部署速度”更符合真实落地。

### 6.6 把当前 PTQ 实现按代码顺序走一遍

如果你想把文档和代码完全对上，当前仓库里的 PTQ 实际顺序是：

1. `train.py` 先得到一份 float checkpoint
2. `run_ptq(...)` 把 float 模型加载到 CPU
3. 用 `torch.randn(2, 3, 224, 224)` 构造 export 示例输入
4. 通过 `export_with_dynamic_batch(...)` 导出图，并约束动态 batch
5. 用 `build_quantizer(...)` 构造 backend quantizer
6. 调 `prepare_pt2e(...)`
7. 用 `calib_loader` 跑 `args.calib_batches` 个 batch 做 calibration
8. 调 `convert_pt2e(...)`
9. 在 CPU 上评估验证集精度
10. 分别记录 eager CPU 延迟和 deploy 延迟

这里有几个非常具体的工程含义：

- PTQ 不会重新训练模型
- PTQ 的质量主要取决于 float baseline、quantizer 配置、calibration 数据
- 当前 deploy 延迟只有 `x86_inductor` 会额外通过 `torch.compile(inductor)` 测一次

如果你读到某个 benchmark 结果想问“这到底是哪一步造成的”，通常就从这 10 步往回定位。

---

## 7. QAT 在这个仓库里是怎么实现的

QAT 的核心不是“直接把训练改成 int8”，而是：

- 在训练期间插入 fake quant 行为
- 让模型参数慢慢适应量化数值误差

### 7.1 对应代码入口

主要函数：

- `run_qat(...)`

### 7.2 典型流程

1. 从 float checkpoint 恢复模型
2. `torch.export`
3. `prepare_qat_pt2e`
4. 在训练 / 微调循环里做 fake quant 训练
5. 训练后加载 best prepared state
6. 转回 CPU
7. `convert_pt2e`
8. 做真实量化模型评估和 benchmark

当前实现还有一个容易被忽略、但很实用的约束：

- QAT 导出时把动态 batch 下界设成了 `min_batch=2`
- 同时训练和验证也会跳过小于 `2` 的 batch
- 所以当前 `QAT` 路径要求 `--batch-size >= 2`

这也是为什么代码里会直接报：

- `QAT 训练图当前要求 batch size >= 2`

### 7.3 为什么 QAT 训练可以走 GPU

因为训练阶段本质上还是 float 计算图，只是数值行为模拟量化。

所以仓库里当前采用的是更现实的策略：

- float 训练：优先走 GPU
- QAT fake-quant 微调：优先走 GPU
- 最终真实量化验证和 deploy benchmark：回到 CPU

这点非常重要：

- GPU 上做的是“训练 / 微调提效”
- CPU 上看的才是“最终 int8 部署表现”

### 7.4 为什么要在后期关闭 observer、冻结 BatchNorm

QAT 后期如果量化范围还在持续漂移，训练会更不稳定。

因此常见做法是：

- 前期让 observer 继续更新统计
- 中后期关闭 observer
- 冻结 BatchNorm 统计量

仓库里对应参数：

- `--disable-observer-epoch`
- `--freeze-bn-epoch`

代码里对应逻辑：

- `disable_observer_if_supported(...)`
- `freeze_bn_stats_in_exported_graph(...)`

### 7.5 为什么最终还要重新 `convert_pt2e`

因为训练阶段的 prepared 模型仍然是：

- 用于 fake quant 的训练态表示

真正部署前必须转成：

- 真实量化图

也就是：

- 训练阶段的“量化感知”
- 和最终部署阶段的“真实量化模型”

不是同一件事。

### 7.6 把当前 QAT 实现按代码顺序走一遍

QAT 在当前仓库里的真实流程比 PTQ 多几步：

1. `run_qat(...)` 从 float checkpoint 恢复模型
2. 把模型放到 `qat_device`
3. 用设备上的示例输入做 `torch.export`
4. 调 `prepare_qat_pt2e(...)`
5. 用 `AdamW + CosineAnnealingLR` 做 fake-quant 微调
6. 每轮结束后在 prepared 模型上做一次 `Val(FakeQ)` 评估
7. 在设定 epoch 后关闭 observer
8. 在设定 epoch 后冻结 BatchNorm 统计量
9. 保存最优 prepared state 到 `qat_prepared_best.pth`
10. 训练完成后把 best prepared model 转回 CPU
11. 调 `convert_pt2e(...)`
12. 对真实量化模型做最终 CPU 评估和 benchmark

这能帮助你理解一个关键差别：

- `Val(FakeQ)` 是训练中的代理指标
- 最终 `convert_pt2e` 后的 CPU 结果，才是部署口径下的真实指标

所以如果你看到：

- 训练中 fake-quant 验证指标很好
- 但最终 CPU int8 结果没有预期那么好

不要惊讶，这正是为什么仓库里把：

- 训练期验证
- 最终 CPU 量化验证

拆成两层来做。

---

## 8. 为什么这个仓库不把 `quantize_()` 当成图像分类默认路线

这部分很关键。

### 8.1 `quantize_()` 更擅长什么

按 `torchao` 当前文档的覆盖重点看，`quantize_()` 更偏：

- `Linear` 主导模型
- weight-only quantization
- dynamic activation + weight quantization
- 大模型 / 推理优化

例如常见配置：

- `Int4WeightOnlyConfig`
- `Int8DynamicActivationInt8WeightConfig`
- `Int8DynamicActivationIntxWeightConfig`

### 8.2 本仓库当前示例更想讲什么

这里想讲的是：

- 图像分类 CNN 的 static int8 量化
- PTQ 与 QAT 的完整工程链路
- backend、部署口径、benchmark 解读

所以默认路线才会是：

1. `float baseline`
2. `PT2E PTQ`
3. `PT2E QAT`

如果以后仓库里补：

- `ViT`
- `LLM`
- `Linear` 主导模型

那时再把 `quantize_()` 作为主线展开会更自然。

---

## 9. 环境与 backend

至少需要：

```bash
pip install torch torchvision torchao
```

如果要跑 `xnnpack`：

```bash
pip install executorch
```

当前仓库更推荐先用：

- `--backend x86_inductor`

原因：

- 依赖更少
- 普通 x86 CPU 更容易先跑通
- 更适合先验证“链路是否正确”

然后再考虑：

- `--backend xnnpack`

需要特别区分一件事：

- `x86_inductor`
  - 当前脚本会给出 `torch.compile(inductor)` 的 deploy 延迟
- `xnnpack`
  - 当前脚本主要给出当前口径下的 CPU 结果
  - 如果要严格对齐移动端运行时数据，仍然建议在对应运行时里单独测

---

## 10. 最常用的运行方式

### 10.1 先看支持的模型

```bash
python3 cv/image_classification/quantization/train.py --list-models
```

### 10.2 快速烟测

```bash
bash cv/image_classification/quantization/run_benchmark.sh smoke
```

### 10.3 正式 benchmark

```bash
bash cv/image_classification/quantization/run_benchmark.sh formal
```

### 10.4 手动指定 QAT 走 GPU

```bash
python3 cv/image_classification/quantization/train.py \
  --mode compare \
  --dataset-type cifar100 \
  --model-name resnet18 \
  --backend x86_inductor \
  --qat-device cuda
```

### 10.5 `compare` 模式到底会做什么

这个模式和“理论 + 实例结合”关系最大，建议明确记住：

1. 先训练或复用 float checkpoint
2. 用同一份 float checkpoint 跑 PTQ
3. 再用同一份 float checkpoint 跑 QAT
4. 最后把三组结果统一写到同一个实验目录

也就是说，当前默认比较口径并不是：

- 三条线各自独立训练出不同起点

而是：

- 同一个 float baseline
- 向下分叉出 PTQ 和 QAT

这能让对比更干净，也更符合工程上“先做 PTQ，不够再上 QAT”的决策顺序。

### 10.6 一次完整实验应该怎么跑

如果你第一次完整复现实验，推荐直接从这个命令开始：

```bash
python3 cv/image_classification/quantization/train.py \
  --mode compare \
  --dataset-type cifar100 \
  --data-root ./datasets \
  --model-name resnet18 \
  --float-epochs 3 \
  --qat-epochs 2 \
  --calib-batches 20 \
  --batch-size 128 \
  --benchmark-warmup 5 \
  --benchmark-iters 10 \
  --backend x86_inductor \
  --qat-device cuda \
  --output-dir ./outputs/image_classification/torchao_quantization_trial
```

这条命令背后的意思可以拆成四组：

#### 任务设置

- `--dataset-type cifar100`
- `--model-name resnet18`

表示先用一个最标准、最容易对照教程的 CNN 分类任务起步。

#### 训练和量化设置

- `--float-epochs 3`
- `--qat-epochs 2`
- `--calib-batches 20`

表示：

- 先得到一个还算像样的 float baseline
- 再用少量 epoch 做 QAT 微调
- PTQ 则只拿 20 个 calibration batch 估计激活范围

#### benchmark 设置

- `--benchmark-warmup 5`
- `--benchmark-iters 10`

表示这是一组“能较快复现但仍有参考意义”的时延采样参数。

#### 部署口径设置

- `--backend x86_inductor`
- `--qat-device cuda`

表示：

- 训练期让 QAT 优先利用 GPU
- 最终部署对比仍按 x86 CPU int8 口径解释

### 10.7 CLI 参数该怎么理解

下面这张表是最常调、也最值得理解的参数：

| 参数 | 主要影响阶段 | 影响什么 | 什么时候优先改它 |
| --- | --- | --- | --- |
| `--train-subset` | float / QAT / PTQ | 同时缩小训练和 calibration 视角 | 想快速烟测全链路时 |
| `--val-subset` | 验证 / benchmark | 缩小最终评估样本量 | 想缩短对比时间时 |
| `--calib-batches` | PTQ | observer 统计覆盖范围 | PTQ 掉点明显时先排查 |
| `--float-epochs` | float baseline | baseline 上限 | 怀疑 baseline 本身不稳时 |
| `--qat-epochs` | QAT | fake-quant 适应程度 | PTQ 不够、QAT 还没追回来时 |
| `--qat-learning-rate` | QAT | 微调稳定性 | QAT 波动大或后期发散时 |
| `--disable-observer-epoch` | QAT | 何时停止更新量化范围 | 训练后期范围漂移时 |
| `--freeze-bn-epoch` | QAT | 何时冻结 BN 统计 | CNN QAT 后期不稳定时 |
| `--backend` | PTQ / QAT / deploy | backend quantizer 和 benchmark 口径 | 要切换 x86 / mobile 解释时 |
| `--benchmark-warmup` | benchmark | 预热充分程度 | 时延波动较大时 |
| `--benchmark-iters` | benchmark | 统计稳定性 | 想要更稳的时延均值时 |
| `--reuse-float-checkpoint` | compare | 是否跳过重复 float 训练 | 反复做 PTQ / QAT 对比时 |

---

## 11. 结果应该怎么解读

### 11.1 PTQ 和 float 差距很小

说明模型对 static int8 友好。

这种情况通常优先落地 PTQ，因为：

- 成本最低
- 链路最简单
- 精度已经足够

### 11.2 PTQ 掉点明显，QAT 追回来

这是最典型的“QAT 值得上”的情况。

说明：

- 量化误差确实存在
- 但模型可以通过 fake quant 微调重新适应这些误差

### 11.3 QAT 精度更好，但 CPU deploy 速度没有明显提升

这完全正常。

因为：

- QAT 解决的是数值适应和精度问题
- backend kernel 命中决定的是最终部署效率

二者不是一回事。

所以更现实的判断顺序是：

1. 先看 PTQ 是否够用
2. 不够再看 QAT 是否值得为精度付出训练成本
3. 不要默认假设 QAT 一定更快

### 11.4 如果模型变小了，但延迟没明显下降

优先排查：

- 是否真的命中了量化 backend
- 算子是否存在频繁反量化 / 回退
- batch size 是否过小，导致 kernel 优势体现不出来
- 性能瓶颈是否其实在数据搬运、非量化层或 Python 开销

### 11.5 输出目录里每个文件怎么看

当前实验跑完后，你最常看到这些文件：

- `float_best.pth`
  - float 基线最优 checkpoint
- `qat_prepared_best.pth`
  - QAT 训练期间保存的最优 prepared state
- `benchmark_results.json`
  - 最适合程序化解析，包含参数、运行环境、结果行
- `benchmark_results.csv`
  - 最适合导入表格和做横向整理
- `benchmark_results.md`
  - 最适合直接贴到实验记录或汇报文档

如果你只想快速判断“要不要继续做 QAT”，通常优先看：

1. `benchmark_results.md`
2. `benchmark_results.json`

前者适合人读，后者适合继续做批量分析。

### 11.6 从结果到下一步：一个实用决策流程

如果你刚跑完 `compare`，可以按下面顺序决策：

#### 情况 A：PTQ 几乎不掉点

优先动作：

- 先确认 deploy 延迟是否有收益
- 如果收益也成立，优先落地 PTQ
- 除非业务对最后一点精度特别敏感，否则没必要急着上 QAT

#### 情况 B：PTQ 掉点明显，但 QAT 追回来

优先动作：

- 保留 PTQ 结果作为低成本基线
- 评估 QAT 训练成本是否值得
- 看 QAT 是否真的带来业务可接受的精度收益

#### 情况 C：PTQ 和 QAT 都不理想

优先动作：

- 先回头检查 float baseline 是否足够强
- 再检查 calibration 视角和样本覆盖
- 然后检查当前 backend / quantizer 配置是否适合这个模型

也就是说，遇到问题时不要一上来就继续堆 QAT epoch。

更稳的顺序是：

1. 先确认 baseline
2. 再确认 calibration
3. 再确认 backend 与模型匹配
4. 最后才去细调 QAT

### 11.7 一份够用的调参顺序表

如果你只想知道“下一步先改什么”，按这个顺序通常最省时间。

#### PTQ 掉点明显时

1. 检查 float baseline 是否可靠
2. 检查 calibration 是否使用评估态预处理
3. 增大 `--calib-batches`
4. 增大 `--train-subset`，避免 calibration 视角太窄
5. 再考虑是否切换 backend 或补 QAT

#### QAT 不稳定时

1. 先降低 `--qat-learning-rate`
2. 检查 `--disable-observer-epoch`
3. 检查 `--freeze-bn-epoch`
4. 适度增加 `--qat-epochs`
5. 最后再考虑更换模型或量化路线

#### benchmark 波动很大时

1. 增大 `--benchmark-warmup`
2. 增大 `--benchmark-iters`
3. 固定 batch size
4. 尽量避免一边训练一边测 deploy 延迟

---

## 12. 常见误区

### 12.1 “量化就是把模型 cast 成 int8”

不对。

量化至少还包含：

- 量化参数选择
- 激活统计或动态量化
- backend lowering
- kernel 命中

只有 dtype 变了，不代表部署链路就成立了。

### 12.2 “QAT 一定比 PTQ 好”

不对。

QAT 往往：

- 更贵
- 更复杂
- 更难调

只有当 PTQ 精度不够时，QAT 的收益才明显。

### 12.3 “GPU 上做了 QAT，所以最终部署也该看 GPU”

不对。

训练设备和部署设备必须分开解释。

本仓库当前这条线的目标是：

- 用 GPU 训练 / 微调
- 用 CPU 解释最终 int8 部署指标

### 12.4 “校准 batch 越多越好”

不对。

比数量更重要的是：

- 是否覆盖真实输入分布
- 是否和真实推理预处理一致

---

## 13. 常见报错和排查

### 13.1 `No module named torchao`

说明环境里没有装 `torchao`，或者装错了环境。

### 13.2 `No module named executorch`

通常是切了 `xnnpack`，但没有安装 `executorch`。

### 13.3 `torch.export` 失败

优先检查：

- 当前模型能不能被 `torch.export`
- 前向里是否有过于动态的 Python 控制流
- batch 动态约束是否和当前设置冲突

### 13.4 PTQ 掉点太多

优先检查：

- calibration 样本是否代表真实输入
- 预处理是否与推理一致
- float 基线本身是否稳定
- 某些敏感层是否需要更细粒度量化配置

### 13.5 QAT 后期不稳定

优先检查：

- 学习率是否过大
- observer 是否及时关闭
- BatchNorm 统计量是否及时冻结
- QAT epoch 是否过少或过多

---

## 14. 当前仓库里的组织建议

`torchao` 教程本身更像工程化专题，因此放在：

- `docs/model_compression/torchao_quantization_guide.md`

而具体可运行示例继续放在：

- `cv/image_classification/quantization/`

这样分工更清楚：

- `docs/` 负责解释“方法、理论和工程决策”
- `cv/` 负责承载“具体任务下可直接运行的脚本和 benchmark”

如果以后仓库里出现更多跨任务复用的量化公共组件，再进一步抽到 `tooling/model_compression/` 会更自然。

### 14.1 当前示例刻意没有覆盖什么

为了让这份教程足够聚焦，当前仓库 **没有** 试图一次讲完所有量化话题。

目前还没有系统展开的部分包括：

- 自定义 observer / histogram observer / percentile clipping 的细粒度配置
- layer-wise quantizer 覆盖策略，例如只改某几层或按模块类型单独设 qconfig
- `ViT / LLM / Embedding / Attention` 为主的 `quantize_()` 路线实战
- 真实移动端 `ExecuTorch / XNNPACK` on-device benchmark
- weight-only、int4、fp8 等更偏大模型或 GPU 的低精度路线

这不是缺陷，而是当前教程的刻意取舍：

- 先把 CNN static int8 的主路径讲清楚
- 再把理论、代码、benchmark 口径统一起来
- 避免一篇文档同时塞进太多互相独立的量化范式

如果后面仓库要继续扩，最自然的增量顺序会是：

1. 补 `quantize_()` 路线的 `Linear / ViT / LLM` 专题
2. 补 `ExecuTorch` 真机 benchmark
3. 补更细粒度的 quantizer / observer 定制教程

---

## 15. 参考资料

以下是这份文档整理时重点参考、并且最值得顺着读的官方资料：

- `torchao` 文档首页：https://docs.pytorch.org/ao/stable/index.html
- Quantization Overview：https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html
- PT2E Quantization：https://docs.pytorch.org/ao/stable/pt2e_quantization/index.html
- PyTorch 2 Export Post Training Quantization：https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_ptq.html
- PyTorch 2 Export Quantization-Aware Training (QAT)：https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_qat.html
- PyTorch 2 Export Quantization with X86 Backend through Inductor：https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_x86_inductor.html
- Quantized Inference：https://docs.pytorch.org/ao/stable/workflows/inference.html
- Quantization-Aware Training (QAT)：https://docs.pytorch.org/ao/stable/workflows/qat.html
- ExecuTorch XNNPACK Quantization：https://docs.pytorch.org/executorch/1.0/backends/xnnpack/xnnpack-quantization.html
- `torchao.quantization` API：https://docs.pytorch.org/ao/stable/api_reference/api_ref_quantization.html
- PyTorch 量化迁移说明：https://docs.pytorch.org/docs/stable/quantization.html

建议阅读顺序：

1. 先看这份仓库教程，把“量化概念图”建立起来
2. 如果你做的是 CNN static int8，优先继续看 `PT2E Quantization` 和它下面的 PTQ / QAT 子教程
3. 如果你做的是 LLM / Transformer，再重点看 `Quantized Inference` 和独立的 `QAT` workflow 页面
