# torchao PT2E 图像分类实战：代表性代码讲解

这份文档只讲一件事：

- **本仓库里的 `cv/image_classification/quantization/` 到底有哪些最值得对着看的代表性代码。**

它不是运行手册，也不再展开目录职责、参数表、结果文件说明。

这些内容已经放回：

- [cv/image_classification/quantization/README.md](../../cv/image_classification/quantization/README.md)

在开始之前，建议你至少已经读过：

1. [量化基础原理：从公式到手写实现](./quantization_fundamentals.md)
2. [量化工作流：PTQ、QAT、Observer 与 Calibration](./quantization_workflows.md)
3. [torchao 量化路线总览与阅读指南](./torchao_quantization_guide.md)

---

## 1. 先看总控入口：`main()` 如何把整条线串起来

如果你只想先抓整条主线，先看：

- [train.py](/root/codes/DLProjects/cv/image_classification/quantization/train.py:155) 里的 `main()`

这一段代码做了 4 件事：

1. 解析参数，构造数据和输出目录
2. 训练或复用 float checkpoint
3. 按 `mode` 分叉到 `PTQ`、`QAT` 或 `compare`
4. 把结果整理并落盘

你可以先把总流程记成：

```text
build data
   ->
train or reuse float checkpoint
   ->
run_ptq(...) or run_qat(...)
   ->
build result rows
   ->
write compare artifacts
```

这也是为什么这里推荐先看 `train.py`，再去看 `quant_pt2e.py`。

---

## 2. 代表性代码一：为什么这里会同时构造 `train / calib / val` 三种视角

最值得先看的函数是：

- [quant_core.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_core.py:114) 里的 `build_datasets(...)`

这一段代码最重要的点，不是“读了几个数据集”，而是：

- **它刻意把训练、校准、验证拆成了三种视角。**

对 CIFAR-100 来说，代码实际上会构造：

- `train_dataset`
- `calib_dataset`
- `val_dataset`

其中：

- `train_dataset` 和 `calib_dataset` 默认都来自训练集
- 但 `train_dataset` 用训练增强
- `calib_dataset` 用评估态预处理

这背后的设计意图是：

- 训练视角要更像训练
- 校准视角要更像部署前输入分布
- 验证视角要更像最终评估

所以这里不是在强调“必须单独切第三份数据”，而是在强调：

- **calibration 需要代表部署时的输入分布。**

这也解释了为什么同一个 `--train-subset` 会同时影响：

- 训练样本数量
- calibration 覆盖范围

如果你只对着一段代码看，这一段最适合帮助你把“训练集 / 校准集 / 验证集”三个角色分清。

---

## 3. 代表性代码二：PTQ 的核心链路到底落在哪

PTQ 主入口在：

- [quant_pt2e.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:161) 的 `run_ptq(...)`

如果你只想抓主干，这条链路最值得记住：

```text
float checkpoint
   -> export_with_dynamic_batch(...)
   -> build_quantizer(...)
   -> prepare_pt2e(...)
   -> calibrate(...)
   -> convert_pt2e(...)
   -> evaluate / benchmark
```

这里最有代表性的代码有 4 处。

### 3.1 `export_with_dynamic_batch(...)`

对应函数：

- [quant_pt2e.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:30)

它做的不是量化本身，而是：

- 先把 float 模型导出成更适合 PT2E 处理的图
- 并给 batch 维保留动态范围

可以先把它理解成：

- **后面 `prepare / convert` 能顺利工作，先得有一张 exported graph。**

### 3.2 `build_quantizer(...)`

对应函数：

- [quant_pt2e.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:70)

它主要决定两件事：

1. 你现在选的是哪个 backend
2. 这个 backend 下量化配置该怎么构造

当前这条示例线最重要的一点是：

- 它刻意把 `x86_inductor` 这条线锁在 static int8 场景

所以如果你是第一次看这段代码，不要把它理解成“任意量化配置工厂”，而要理解成：

- **当前教学示例在这里把 backend 约束具体落到了代码里。**

### 3.3 `prepare_pt2e(...)` 和 `calibrate(...)`

`prepare_pt2e(...)` 在 `run_ptq(...)` 里直接调用，`calibrate(...)` 则是：

- [quant_pt2e.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:152)

这里最值得看清的是分工：

- `prepare_pt2e(...)`
  - 把模型从普通 float 图变成“可量化准备态”
- `calibrate(...)`
  - 真正让 observer 看一批代表性输入，收集激活统计

也就是说：

- `prepare` 不是“已经量化好了”
- `calibration` 也不是“继续训练”

它们一个负责把工具装进去，一个负责喂数据收统计。

### 3.4 `convert_pt2e(...)`

`convert_pt2e(...)` 是 PTQ 链路里真正把准备态模型转成最终量化模型的步骤。

在这一套实现里，PTQ 为什么刻意从 CPU 开始、最后也在 CPU 评估？

因为当前示例的目标不是：

- 先在 GPU 上把流程凑齐

而是：

- **让最终量化精度和 deploy benchmark 口径都落在 CPU 上。**

这也是为什么 `run_ptq(...)` 里会：

- 从 CPU 恢复 float checkpoint
- 在 CPU 上 calibration
- 在 CPU 上评估和 benchmark

---

## 4. 代表性代码三：QAT 的核心链路到底和 PTQ 差在哪

QAT 主入口在：

- [quant_pt2e.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:220) 的 `run_qat(...)`

最值得先抓住的差别只有一句：

- **PTQ 的中间阶段是 calibration，QAT 的中间阶段是 fake quant 训练。**

它的主链路可以先记成：

```text
float checkpoint
   -> export_with_dynamic_batch(...)
   -> prepare_qat_pt2e(...)
   -> fake quant fine-tune
   -> disable observer / freeze BN
   -> convert_pt2e(...)
   -> final CPU evaluate / benchmark
```

### 4.1 为什么 QAT 先放到 GPU，再回到 CPU

这段代码最有教学价值的地方，是它把“训练设备”和“部署设备”明确拆开了。

在 `run_qat(...)` 里：

- float checkpoint 先被恢复到 `qat_device`
- fake quant 微调也在 `qat_device` 上进行

但训练完成后，代码又会：

- 把 prepared model 转回 CPU
- 再 `convert_pt2e(...)`
- 再做最终 CPU 评估和 benchmark

这想强调的是：

- **当前示例默认把 QAT fake quant 微调放到 GPU**
- **但最终量化模型的验证和部署口径仍然应该回到目标设备**

### 4.2 `prepare_qat_pt2e(...)` 之后模型发生了什么

这一点在代码里非常关键：

- `prepare_qat_pt2e(...)` 之后，模型已经不是普通 float 模型
- 它变成了一张“可以继续做 fake quant 训练”的准备态图

所以你在这之后看到的训练，不是在训练一个原始 float 模型，而是在训练一个：

- **前向过程中会模拟量化误差的模型。**

### 4.3 为什么会有“关闭 observer”和“冻结 BN”

对应代码在：

- [disable_observer_if_supported(...)](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:122)
- [freeze_bn_stats_in_exported_graph(...)](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:128)

这两步都发生在 QAT 后期。

可以先用最实用的角度理解：

- 训练前期，observer 还需要继续看分布
- 到了后期，希望量化参数和 BN 统计量逐步稳定下来

所以代码会按 epoch：

- 先关闭 observer
- 再冻结 BN 统计量

如果你以前只在文档里见过这两个概念，这一段代码最适合帮助你把它们和真实训练过程对应起来。

### 4.4 为什么这里要求 `batch size >= 2`

这不是随便加的限制。

在当前实现里：

- QAT 导出图时把动态 batch 下界设成了 `min_batch=2`
- 训练和验证也会跳过小于 2 的 batch

所以这个限制不是“拍脑袋的经验值”，而是：

- **当前 QAT 图约束在代码里的直接体现。**

---

## 5. 代表性代码四：为什么 `compare` 模式更像真实工程决策

如果你只想看一个最能体现工程思路的地方，推荐回到：

- [train.py](/root/codes/DLProjects/cv/image_classification/quantization/train.py:155) 的 `main()`

最关键的不是它调用了哪些函数，而是它的分叉方式：

- 先拿到一份 float checkpoint
- 再用这份 checkpoint 跑 PTQ
- 再用同一份 checkpoint 跑 QAT

这意味着：

- 不是三条线各自独立训练、各自比较
- 而是“同一个 float baseline 向下分叉”

这样做的好处是：

- 对比更干净
- 更接近真实项目里先 PTQ、后 QAT 的决策顺序

所以 `compare` 模式真正有代表性的地方，不是它名字叫 compare，而是：

- **它把一条真实的工程决策链写进了代码。**

---

## 6. 代表性代码五：quantizer 和 benchmark 在代码里怎么落地

这条线还有一个很适合对着代码看的点：

- **backend 不是只写在文档里的概念，而是真的在代码里落地成 quantizer 和 benchmark 行为。**

最值得一起看的两个函数是：

- [build_quantizer(...)](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:70)
- [benchmark_deploy_inference(...)](/root/codes/DLProjects/cv/image_classification/quantization/quant_benchmark.py:74)

它们分别回答：

1. 当前 backend 会生成什么量化配置
2. 最终 deploy benchmark 到底怎么测

当前实现里的关键区别是：

- `x86_inductor`
  - 可以尝试用 `torch.compile(inductor)` 测 deploy 延迟
- `xnnpack`
  - 这份脚本不会直接给你真实 on-device 延迟
  - 更适合先看精度和 eager CPU 延迟

所以你在看结果文件时，不应该把所有 backend 的 `Deploy(ms)` 当成同一口径。

---

## 7. 最值得按什么顺序对着源码看

如果你准备把这一套代码真正看明白，推荐顺序是：

1. [train.py](/root/codes/DLProjects/cv/image_classification/quantization/train.py:155) 的 `main()`
2. [quant_core.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_core.py:114) 的 `build_datasets(...)`
3. [quant_core.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_core.py:328) 的 `train_float_model(...)`
4. [quant_pt2e.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:161) 的 `run_ptq(...)`
5. [quant_pt2e.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:220) 的 `run_qat(...)`
6. [quant_benchmark.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_benchmark.py:199) 的 `write_compare_artifacts(...)`

这条顺序对应的是：

- 先看流程入口
- 再看数据和 baseline
- 再看 PTQ / QAT 分叉
- 最后看 benchmark 和结果落盘

---

## 8. 哪些内容故意不在这篇里

为了减少文档重复，这篇刻意不再展开：

- 目录文件分工
- 常用运行命令
- 参数表
- 输出文件逐项说明

这些内容统一放在：

- [cv/image_classification/quantization/README.md](../../cv/image_classification/quantization/README.md)

而更基础的概念和路线选择，仍然放在：

- [量化工作流：PTQ、QAT、Observer 与 Calibration](./quantization_workflows.md)
- [torchao 量化路线总览与阅读指南](./torchao_quantization_guide.md)

如果你已经看完这一篇，下一步最合理的是：

1. 回到 [README](../../cv/image_classification/quantization/README.md) 跑一遍命令
2. 再对照 `train.py`、`quant_pt2e.py` 把流程串起来
