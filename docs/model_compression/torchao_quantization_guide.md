# torchao 量化路线总览与阅读指南

这份文档只回答一类问题：

- **当你开始看 `torchao` 量化文档和代码时，到底该怎么理解它的路线、术语和使用方式。**

它不是一份“把所有量化原理从头讲完”的教材，但会尽量做到：

- 前面把必须听懂的基础概念讲清楚
- 中间把两条主路线讲明白
- 后面把“怎么选、怎么用、怎么读官方文档、怎么对照本仓库代码”讲完整

如果你是第一次接触量化，建议按下面顺序阅读：

1. [量化基础原理：从公式到手写实现](./quantization_fundamentals.md)
2. [量化工作流：PTQ、QAT、Observer 与 Calibration](./quantization_workflows.md)
3. 再回来看这份 `torchao` 路线总览
4. 最后读 [torchao PT2E 图像分类实战：对应本仓库实现](./torchao_pt2e_image_classification.md)

但即使你还没把前两篇完全读熟，也可以先把这篇读完，先建立一张够用的全局地图。

---

## 1. 先说清楚：`torchao` 到底在做什么

很多初学者第一次看到 `torchao` 会有两个误解：

1. 以为它在发明一套全新的量化理论
2. 以为它只是“把模型变成 `int8`”的一组 API

更准确的理解是：

- **`torchao` 主要是在 PyTorch 生态里，把已有的量化原理整理成几条可落地的工程路线。**

它更关心三件事：

- 什么模型适合什么量化方案
- 量化流程该怎么走
- 最终能不能落到目标 backend

所以你看到的很多词，本质上是：

- 量化原理
- 工程流程
- backend 约束

被放进了同一套工具体系里。

这也是为什么官方文档会分成几类页面：

- 有的讲基础概念
- 有的讲模块级量化
- 有的讲导出图后的量化流程
- 有的讲 backend 和部署

不是文档分散，而是量化本来就是一条工程链路。

---

## 2. 先把几个关键名词听懂

这一节只做一件事：

- **把后面反复出现的术语先翻译成大白话。**

你不用一开始就把这些词记得非常精确，但至少要知道它们各自在说什么。

### 2.1 仿射量化（`affine quantization`）是什么

- 它就是最常见的“把浮点数映射成整数”的办法。
- 你前面学过的 `scale / zero_point`，说的就是这件事。

所以这里不用把它当成一条新路线，只要记住：

- **`affine quantization` 是底层数值映射方法，不是路线名。**

更细的公式和推导，放在 [量化基础原理](./quantization_fundamentals.md) 里看更合适。

### 2.2 `weight-only`、`dynamic`、`static` 到底在区分什么

这三个词都在讲“量化谁、量化参数什么时候定”，但重点不同。

| 方案 | 先抓住什么 | 常见场景 |
| --- | --- | --- |
| `weight-only` | 主要量化权重，激活不一定一起量化 | `Linear` 多的模型、LLM 推理优化 |
| `dynamic` | 激活量化参数在推理时再决定 | `Linear` 主导模型 |
| `static` | 激活量化参数提前通过 calibration 估计好 | CNN、CPU `int8` 部署 |

如果你只记一句：

- `weight-only` 在说“量化谁”，`dynamic / static` 在说“激活参数什么时候定”。

### 2.3 直接按模块处理模型的量化方式（`eager quantization`）是什么

- `eager` 可以先理解成“更像按模块处理模型，而不是先整理成一张统一图”。
- 如果你看到某些 `Linear` 模块被直接替换或套上量化配置，这通常就更接近 eager 风格。

所以后面看到：

- `quantize_()`

先把它理解成：

- **围着模块做量化。**

### 2.4 导出后的计算图（`exported graph`）是什么

- `torch.export` 可以先粗略理解成：把原来的模型整理成一张更规整、更适合后端分析和变换的图。
- 所以 `exported graph` 不是“已经量化好的模型”，而是 `PT2E` 路线的起点。

你只要先记住：

- **`PT2E` 是先拿到图，再在图上做量化准备和转换。**

### 2.5 backend、lowering、kernel 在这里分别是什么意思

- `backend`：最终承接执行的后端
- `kernel`：真正做底层计算的实现
- `lowering`：把高层图落到后端能执行的形式

这一组词为什么重要？

因为量化不只是“把数值变小”，还要看：

- 后端支不支持
- 会不会真的匹配到量化 kernel
- benchmark 测到的到底是哪条执行路径

### 2.6 先记住这 5 句话

- `affine quantization` 是底层映射方法
- `weight-only / dynamic / static` 是不同量化方案
- eager 风格更像按模块处理
- `PT2E` 更像先导图再量化
- backend 决定你最后能不能真正跑起来、跑得快

### 2.7 最容易混的一点：这其实是三层概念

初学者最容易把下面三层东西混成一层：

| 你看到的词 | 它在回答什么问题 | 典型例子 |
| --- | --- | --- |
| 路线 | 工程上怎么落地 | `quantize_()`、`PT2E` |
| 工作流 | 训练后做，还是训练中适应 | `PTQ`、`QAT` |
| 量化方案 | 具体量化谁、参数何时确定 | `weight-only`、`dynamic`、`static` |

如果你只记一句：

- **`quantize_() / PT2E` 是路线，`PTQ / QAT` 是工作流，`weight-only / dynamic / static` 是量化方案。**

---

## 3. 为什么 `torchao` 看起来像有两条主路线

先说结论：

- **不是因为有两套不同的量化原理，而是因为有两种很不一样的工程落地方式。**

可以先把它想成两个典型场景。

### 3.1 场景一：我想快速优化 `Linear` 主导模型的推理

这种场景常见于：

- Transformer
- LLM
- 很多以 `Linear` 为主的模型

这时你最关心的可能是：

- 权重能不能压小
- 显存或内存能不能省
- 某些核心模块能不能直接套量化配置

这种时候，思路通常更像：

- 不必先把整个模型变成一张后端友好的全图
- 先看哪些模块值得量化
- 再对这些模块应用量化配置

这就更接近：

- `quantize_()` 这条 eager 风格路线

### 3.2 场景二：我想把一个模型按部署后端要求做成完整量化流程

这种场景常见于：

- CNN 图像分类
- CPU `int8` 部署
- 需要清楚经历 `prepare / calibration / convert / deploy` 全流程的任务

这里这 4 个词可以先这样理解：

- `prepare`：先往模型里插入量化需要的辅助部件。对 `PTQ` 来说，最典型的是插 `observer`，用来记录激活范围；对 `QAT` 来说，还会放入 `fake quant`，用来在训练时模拟量化误差。做完这一步后，模型还不是最终量化模型，但已经从普通 float 模型变成了“可以继续校准或继续训练”的量化准备态
- `calibration`：用一批代表性样本跑前向，收集激活统计，给后续量化参数提供依据
- `convert`：把中间态模型转成最终量化模型
- `deploy`：把量化模型交给具体 backend 执行，并看最终能否跑通、跑快、测得准

如果走的是 `QAT`，中间阶段会更接近：

- 先 `prepare_qat`
- 再做 fake quant 训练
- 最后 `convert` 和 `deploy`

这时你更关心的往往是：

- 模型怎样变成后端更容易处理的图
- observer / calibration 放在哪一步
- `PTQ` 和 `QAT` 分别怎么落地
- 最终能不能对接具体 backend

这种时候，思路通常更像：

1. 先 `export`
2. 再在导出的图上做量化准备
3. `PTQ` 跑 calibration，`QAT` 跑 fake quant 训练
4. 最后 convert 并 lowering

这就更接近：

- `PT2E` 路线

### 3.3 所以“两条路线”真正的差别是什么

一句话概括：

- **`quantize_()` 更像“围着模块做量化”**
- **`PT2E` 更像“围着导出图和后端做量化”**

它们不是：

- 同一件事换了两个 API 名字

也不是：

- 一个先进、一个落后

而是：

- 适合的模型类型不同
- 适合的部署目标不同
- 工作流重点不同

---

## 4. 路线 A：`quantize_()` 与 eager 风格路线

### 4.1 先用一句话理解这条线

你可以把它想成：

- **先盯住模型里的某些模块，尤其是 `Linear`，再按配置把这些模块量化。**

### 4.2 它通常适合什么模型

这条路线通常更适合：

- `Transformer`
- `LLM`
- 其他 `Linear` 比重很高的模型

常见目标是：

- weight-only
- dynamic activation + weight quantization
- 大模型推理中的显存 / 内存优化

### 4.3 一个够用的流程直觉

对初学者来说，可以先把它记成：

```text
float 模型
   |
   -> 选择要量化的模块或配置
   -> 应用 quantize_() 或对应配置
   -> 用量化后的模块做推理
```

它和 `PT2E` 最大的不同是：

- 重点不在“先拿到一张图”
- 而在“哪些模块适合套哪种配置”

### 4.4 配置名字应该怎么看

长配置名先别整串记，先拆三件事：

1. 激活怎么量化
2. 权重怎么量化
3. 位宽是多少

例如：

- `Int4WeightOnlyConfig`：重点是权重 `int4`
- `Int8DynamicActivationInt8WeightConfig`：重点是激活动态 `int8`，权重也是 `int8`

### 4.5 这条路线的优点和局限

优点通常是：

- 对某些模块接入直接
- 对 `Linear` 主导模型很自然
- weight-only / dynamic 这类方案更常见

局限通常是：

- 不一定适合你理解完整的 static `PTQ / QAT` 流程
- 最终效果仍然要看模型结构和 backend 支持

### 4.6 什么时候优先看这条路线

如果你的模型更偏 `Linear` / Transformer / LLM，目标又更偏大模型推理优化，而不是经典 CNN 的 CPU `int8` 部署，通常先看这条路线。

---

## 5. 路线 B：`PT2E` 与导出图路线

### 5.1 先用一句话理解这条线

你可以把它想成：

- **先把模型导出成图，再围绕这张图做量化准备、校准或 QAT，最后交给 backend。**

### 5.2 它通常适合什么模型和部署目标

这条路线通常更适合：

- `ResNet`
- `MobileNet`
- 其他更偏经典 CNN 的图像任务模型

最常见的目标是：

- static quantization
- CPU `int8` 推理
- 需要把 `PTQ / QAT / deploy` 整条链路讲清楚

### 5.3 `PTQ` 和 `QAT` 在这条路线上分别怎么放

如果是 `PTQ`，常见流程是：

```text
float 模型
   -> export
   -> prepare_pt2e
   -> calibration
   -> convert_pt2e
   -> backend / benchmark / deploy
```

如果是 `QAT`，常见流程是：

```text
float 模型
   -> export
   -> prepare_qat_pt2e
   -> fake quant 训练
   -> convert_pt2e
   -> backend / benchmark / deploy
```

这时你前面学过的工作流概念，就能和 API 对上：

| 你学过的概念 | 在 `PT2E` 路线里的对应位置 |
| --- | --- |
| observer | `prepare_pt2e` 后插进去的统计器，用来记录激活范围 |
| calibration | `PTQ` 中间阶段，用代表性输入收集激活统计 |
| fake quant | `QAT` 训练阶段，在训练时模拟量化误差 |
| convert | `convert_pt2e` |
| backend / deploy | convert 之后再看 lowering 和 benchmark |

### 5.4 这条路线的优点和局限

优点通常是：

- 更适合把完整量化工作流讲清楚
- 更贴近 backend 和部署
- 更适合经典 CNN 的 static `int8` 场景

局限通常是：

- 需要理解的阶段更多
- 更依赖 export 和 backend 支持情况

### 5.5 为什么 backend 在这条线里尤其重要

因为 `PT2E` 不是只追求“图里看起来像量化了”，还要关心：

- 这个 backend 支不支持当前 quantizer 配置
- 最后 lowering 到哪里
- benchmark 对应的到底是哪条执行路径

---

## 6. 我到底该先选哪条路线

这个问题不要靠“哪个 API 名字更眼熟”来决定。

最稳的判断顺序是：

1. 先看模型结构
2. 再看部署目标
3. 最后看你需要的是哪类工作流

### 6.1 一张表先做粗选

| 场景 | 优先路线 | 为什么 |
| --- | --- | --- |
| CNN 图像分类，目标是 CPU `int8` 部署 | `PT2E` | 更贴近 static quantization 和 backend 流程 |
| 想系统理解 `PTQ / QAT / observer / calibration / convert` | `PT2E` | 这条线更完整地覆盖工作流 |
| Transformer / LLM 推理优化 | `quantize_()` | 更常见于 `Linear` 主导模型 |
| 重点是 weight-only、节省显存 / 内存 | `quantize_()` | 这条线相关配置更集中 |
| 只是想最低成本验证量化可行性 | 先 `PTQ` | 成本最低，最像第一轮诊断 |
| `PTQ` 掉点明显，还必须追精度 | 再 `QAT` | 让模型在训练中适应量化误差 |

### 6.2 一个更实用的判断口诀

可以先记成：

- **CNN + static + CPU deploy -> 先想 `PT2E`**
- **Linear 多 + weight-only / dynamic + LLM 推理 -> 先想 `quantize_()`**

### 6.3 为什么通常先试 `PTQ`，再决定要不要 `QAT`

不管你最后走哪条量化线，实践里一个很重要的顺序是：

1. 先拿到稳定的 float baseline
2. 先试 `PTQ`
3. `PTQ` 不够，再上 `QAT`

原因很简单：

- `PTQ` 成本最低
- 很多模型其实 `PTQ` 已经够用
- 如果一上来就做 `QAT`，你很容易把“baseline 不稳”和“量化带来的问题”混在一起

所以：

- **`PTQ` 往往不是低配版 `QAT`，而是第一轮诊断工具。**

---

## 7. 两条路线实际使用时，最需要关心什么

这一节不追求 API 细节百科，只抓真正影响选型和落地的点。

### 7.1 使用 `quantize_()` 路线时，要重点看什么

最重要的不是先背 API，而是先搞清楚三件事：

1. 你的模型是不是 `Linear` 主导
2. 你要的是 weight-only、dynamic，还是别的配置
3. 目标 backend 和实际推理路径是否支持这种方案

如果这三件事都没搞清楚，光看配置名会非常容易乱。

### 7.2 使用 `PT2E` 路线时，要重点看什么

这条路线里最常见的误区，是把很多阶段混成一团。

你最好强制自己分清这几步：

1. `export`：先拿到更规整的图
2. `prepare`：往模型里插 observer 或 fake quant 等量化辅助节点，让它从普通 float 模型变成“可校准 / 可 QAT”的准备态
3. calibration 或 `QAT` 训练：让模型真正经历中间阶段
4. `convert`：转成最终量化形式
5. backend / benchmark：确认最终执行落点

如果你把这些阶段混成“反正就是量化”，后面几乎一定会乱。

### 7.3 怎么判断自己现在卡在哪个层面

你可以用下面这张表做快速定位：

| 你现在的困惑 | 你缺的是哪一层理解 |
| --- | --- |
| 看不懂 `scale / zero_point / affine` | 量化基础原理 |
| 分不清 `PTQ / QAT / observer / calibration` | 量化工作流 |
| 分不清 `quantize_()` 和 `PT2E` | `torchao` 路线总览 |
| 不知道仓库代码里流程落在哪 | 本仓库实战文档 |
| 跑通了但不知道 benchmark 是否可信 | backend / deploy 约束 |

---

## 8. torchao 官方文档到底怎么读

很多人卡住，不是完全看不懂，而是不知道先看哪类页面。

对初学者来说，可以把官方文档粗分成四类：

| 页面类型 | 主要解决什么问题 | 什么时候先看 |
| --- | --- | --- |
| 概念页 | 术语是什么意思 | 你还没听懂 `affine / dynamic / static` |
| `quantize_()` 路线页 | 模块级量化怎么做 | 你偏 LLM / `Linear` 主导模型 |
| `PT2E` 路线页 | `export -> prepare -> convert` 怎么走 | 你偏 CNN、`PTQ / QAT`、CPU `int8` |
| backend 页 | 最终支持什么、怎么部署 | 你已经确定路线，开始关心执行落点 |

更稳的阅读顺序是：

1. 先看概念页
2. 再判断自己更接近 `quantize_()` 还是 `PT2E`
3. 深读对应路线页
4. 最后看 backend 文档

---

## 9. 当前仓库里最应该对照哪条路线

如果你现在的目标是：

- 看懂本仓库这套量化示例

那最相关的是：

- `PT2E PTQ`
- `PT2E QAT`

对应代码目录在：

- `cv/image_classification/quantization/`

### 9.1 为什么本仓库当前主讲 `PT2E`

因为当前仓库这条示例线的任务背景是：

- 图像分类
- 经典 CNN
- 目标是 `PTQ / QAT` 对比
- 最后按 CPU deploy 口径解释 benchmark

所以这里刻意优先讲：

- `PT2E PTQ`
- `PT2E QAT`

而不是默认把：

- `quantize_()`

当成主线。

这不是因为 `quantize_()` 不重要，而是因为：

- 它更偏 `Linear` 主导模型
- 当前仓库这条可运行示例，更适合拿 `PT2E` 讲清 static `int8` 工作流

### 9.2 如果你只想先抓住最关键的 API

这一篇不再重复展开文件分工和逐步执行细节，只保留最值得先认的几个入口：

这些入口主要集中在：

- `quant_pt2e.py`

| API / 入口 | 先把它理解成什么 |
| --- | --- |
| `run_ptq(...)` | 本仓库里 `PTQ` 的主入口 |
| `run_qat(...)` | 本仓库里 `QAT` 的主入口 |
| `prepare_pt2e(...)` | `PTQ` 路线里把模型变成量化准备态 |
| `prepare_qat_pt2e(...)` | `QAT` 路线里把模型变成可 fake quant 训练的准备态 |
| `convert_pt2e(...)` | 把准备态模型转成最终量化模型 |
| `build_quantizer(...)` | 决定 quantizer 和 backend 相关配置 |

如果你现在最关心的是：

- “量化到底在哪一步开始发生”

那先盯住：

- `prepare_pt2e(...) / prepare_qat_pt2e(...)`
- `convert_pt2e(...)`

### 9.3 这篇文档和实战文档怎么分工

这一篇只负责把：

- 路线怎么选
- 术语是什么意思
- `torchao` API 大概落在哪

讲清楚。

更细的内容，例如：

- 文件分工
- `PTQ / QAT` 的逐步执行顺序
- 数据流和 benchmark 口径
- 具体参数和实现细节

都放在：

- [torchao PT2E 图像分类实战：对应本仓库实现](./torchao_pt2e_image_classification.md)

如果你接下来要继续深挖代码，直接转到那一篇更合适。

---

## 10. 最容易混淆的几个问题

### 10.1 `quantize_()` 和 `PT2E` 是不是只是两个 API 名字不同

不是。

更准确地说：

- 它们代表两种不同的工程路线

区别主要在模型处理方式、适合的模型结构和部署目标都不同。

### 10.2 `affine quantization` 和 `static quantization` 是不是一回事

不是。

前者更像“怎么算”，后者更像“什么时候把激活量化参数定下来”。

### 10.3 看见 `QAT` 页面，就一定对应当前仓库示例吗

不一定。

当前稳定版 `torchao` 的独立 `QAT` workflow 页面，更偏：

- `quantize_()` eager 风格路线

而本仓库示例走的是：

- `PT2E QAT`

所以看到 `QAT` 时，先确认它挂在哪条路线下面。

### 10.4 `exported graph` 是不是已经量化好了

不是。

`exported graph` 只是表示：

- 模型已经被整理成适合后续处理的图表示

后面还要继续经历 `prepare`、calibration 或 `QAT`、`convert`、lowering。

---

## 11. 如果你现在就要开始动手，最稳的阅读和实践顺序

### 11.1 如果你的目标是看懂本仓库示例

建议按这个顺序：

1. 先读 [量化基础原理](./quantization_fundamentals.md)
2. 再读 [量化工作流](./quantization_workflows.md)
3. 回来读这份 `torchao` 总览
4. 接着读 [torchao PT2E 图像分类实战](./torchao_pt2e_image_classification.md)
5. 最后对照 `cv/image_classification/quantization/` 下的代码

### 11.2 如果你的目标是看大模型推理量化

建议按这个顺序：

1. 先把这份文档里 `weight-only / dynamic / eager` 相关部分看懂
2. 再看 `quantize_()` 路线相关官方页面
3. 最后再去对照具体模型和 backend 支持

---

## 12. 参考资料

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
