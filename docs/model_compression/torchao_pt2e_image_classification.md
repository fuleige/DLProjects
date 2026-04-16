# torchao PT2E 图像分类实战：对应本仓库实现

这份文档只讲一件事：

- **本仓库里的 `cv/image_classification/quantization/` 到底是怎么把前面的量化基础知识落成可运行代码的。**

它不是量化原理文档，也不是 torchao 全景文档。

在开始之前，建议你至少已经读过：

1. [量化基础原理：从公式到手写实现](./quantization_fundamentals.md)
2. [量化工作流：PTQ、QAT、Observer 与 Calibration](./quantization_workflows.md)
3. [torchao 量化路线总览与阅读指南](./torchao_quantization_guide.md)

---

## 1. 这套示例要解决什么问题

当前仓库这条量化线的目标很明确：

- 先拿到一个稳定的 float 图像分类模型
- 再对比 `PTQ` 和 `QAT`
- 最后按 CPU deploy 口径解释 benchmark

所以它更像一条：

- **教学导向的 CNN static int8 模板**

而不是：

- 任意 backbone 的量化工具箱
- LLM / ViT 的 `quantize_()` 示例
- 真实移动端 on-device benchmark 框架

---

## 2. 当前代码目录怎么分工

目录在：

- `cv/image_classification/quantization/`

主要文件分工如下：

| 文件 | 作用 |
| --- | --- |
| `train.py` | 总入口，串起 float / PTQ / QAT / compare |
| `quant_args.py` | CLI 参数定义 |
| `quant_core.py` | 数据、模型、训练、验证、checkpoint |
| `quant_pt2e.py` | `torch.export`、quantizer、PTQ / QAT 主流程 |
| `quant_benchmark.py` | CPU eager / deploy benchmark 与结果落盘 |
| `run_benchmark.sh` | `smoke / formal` 复现实验入口 |

如果你只想知道“流程从哪开始看”，先看：

- `train.py`

如果你只想知道“量化到底在哪做”，重点看：

- `quant_pt2e.py`

---

## 3. 为什么这里故意选经典 CNN

当前示例默认支持：

- `resnet18`
- `mobilenet_v3_small`

这是刻意选择，不是因为它们最先进，而是因为它们更适合教学。

原因有三个：

1. 它们更贴近 `Conv2d + BatchNorm + ReLU + Linear` 的经典 static int8 场景
2. 图结构更稳定，更适合先把 `PT2E` 流程讲清楚
3. 初学者不应该一开始就被任意 backbone 的 `export` 兼容性打断

所以这条线的优先目标是：

- 先学会量化流程

而不是：

- 一上来就覆盖所有模型家族

---

## 4. 训练集、校准集、验证集在这套实现里分别是什么

这是这套示例里一个很关键的设计点。

### 4.1 三种“视角”，不是三份完全独立的数据

当前代码在 [quant_core.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_core.py:111) 里会构造：

- `train_dataset`
- `calib_dataset`
- `val_dataset`

但要注意：

- `train_dataset` 和 `calib_dataset` 默认都来自训练集
- 区别主要在 transform

具体来说：

- `train_dataset` 使用训练增强
- `calib_dataset` 使用评估态预处理
- `val_dataset` 使用验证集 + 评估态预处理

所以这里的重点不是“硬切第三份 calibration split”，而是：

- 训练视角
- 校准视角
- 验证视角

### 4.2 为什么 calibration 不直接复用训练增强

因为 calibration 的目标不是继续训练模型，而是估计：

- 真实部署时的激活范围

所以它更应该贴近：

- 推理态输入分布
- 评估态 transform

这也是为什么当前实现特意给 `calib_dataset` 用了：

- `eval_transform`

### 4.3 一个容易忽略的细节

当前 CLI 的：

- `--train-subset`

会同时裁剪：

- `train_dataset`
- `calib_dataset`

也就是说，如果你把 `--train-subset` 调得很小：

- 训练样本会变少
- calibration 覆盖范围也会一起变窄

这在烟测时是合理的，但在正式比较时要有意识。

---

## 5. PTQ 在这套实现里是怎么落地的

对应代码主入口：

- [quant_pt2e.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:153) 里的 `run_ptq(...)`

### 5.1 先按流程看一遍

当前 PTQ 实际顺序是：

1. 从 `train.py` 得到或复用一份 float checkpoint
2. 在 `run_ptq(...)` 里把 float 模型加载到 CPU
3. 构造 export 示例输入
4. 调 `export_with_dynamic_batch(...)`
5. 调 `build_quantizer(...)`
6. 调 `prepare_pt2e(...)`
7. 用 `calib_loader` 跑 calibration
8. 调 `convert_pt2e(...)`
9. 在 CPU 上评估验证集
10. 在 CPU 上做 eager 和 deploy benchmark

### 5.2 这和前面学过的工作流怎么对应

| 你学过的概念 | 当前实现里的对应步骤 |
| --- | --- |
| float baseline | `train.py` 先训练或复用 `float_best.pth` |
| observer / prepare | `prepare_pt2e(...)` |
| calibration | `calibrate(prepared_model, calib_loader, ...)` |
| convert | `convert_pt2e(...)` |
| deploy benchmark | `quant_benchmark.py` 里的 benchmark 逻辑 |

### 5.3 quantizer 到底在这里决定了什么

当前 quantizer 由：

- [build_quantizer(...)](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:69)

负责构造。

当前支持两个 backend：

- `x86_inductor`
- `xnnpack`

其中当前更推荐先用：

- `x86_inductor`

因为：

- 依赖更少
- 在普通 x86 CPU 上更容易先跑通

### 5.4 为什么这里强调“当前走的是 static 路线”

因为在 `x86_inductor` 分支里，代码会在 API 支持时显式传：

- `is_dynamic=False`

这意味着当前示例不是在讲：

- dynamic activation quantization

而是在讲：

- static PTQ / static QAT

对初学者来说，这一点非常重要。

否则你会把：

- `quantize_()` 里的 dynamic 路线
- 这里的 `PT2E` static 路线

混成一件事。

---

## 6. QAT 在这套实现里是怎么落地的

对应代码主入口：

- [quant_pt2e.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:208) 里的 `run_qat(...)`

### 6.1 当前 QAT 实际顺序

1. 从 float checkpoint 恢复模型
2. 把模型放到 `qat_device`
3. 用设备上的示例输入做 `torch.export`
4. 调 `prepare_qat_pt2e(...)`
5. 用 `AdamW + CosineAnnealingLR` 做 fake-quant 微调
6. 每轮结束做一次 `Val(FakeQ)` 评估
7. 到设定 epoch 后关闭 observer
8. 到设定 epoch 后冻结 BatchNorm 统计量
9. 保存 best prepared state
10. 训练后转回 CPU
11. 调 `convert_pt2e(...)`
12. 做最终 CPU 评估与 benchmark

### 6.2 为什么 QAT 训练可以放在 GPU

因为训练阶段本质上仍然是：

- float 图上的 fake quant 训练

它不是：

- 真正的 int8 整数训练

所以把 QAT 微调放到 GPU 是合理的。

但要特别注意：

- GPU 上做的是训练/微调
- 最终部署口径仍然是 CPU

这也是为什么当前实现会在训练完成后：

- 把 prepared model 转回 CPU
- 再 `convert_pt2e(...)`
- 再做真实 deploy benchmark

### 6.3 当前实现里一个很具体的约束

在 [run_qat(...)](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:224) 里，当前代码要求：

- `--batch-size >= 2`

原因是：

- QAT 导出时把动态 batch 下界设成了 `min_batch=2`
- 训练和验证也会跳过小于 2 的 batch

所以如果你把 batch size 设成 1：

- 当前这套 QAT 路径会直接报错

---

## 7. `compare` 模式到底做了什么

这一点对理解实验结果非常重要。

对应总入口：

- [train.py](/root/codes/DLProjects/cv/image_classification/quantization/train.py:155)

当前 `compare` 模式的真实逻辑是：

1. 先训练或复用同一份 float checkpoint
2. 用这份 checkpoint 跑 PTQ
3. 再用同一份 checkpoint 跑 QAT
4. 把三组结果统一写到同一个实验目录

这意味着：

- 当前比较口径不是“三条线各自独立训练”
- 而是“同一个 float baseline 向下分叉”

这样做的好处是：

- 对比更干净
- 更符合真实工程决策顺序

也就是：

1. 先做 float baseline
2. 再试 PTQ
3. PTQ 不够再上 QAT

---

## 8. 最常用的运行方式

### 8.1 看支持的模型

```bash
python3 cv/image_classification/quantization/train.py --list-models
```

### 8.2 快速烟测

```bash
bash cv/image_classification/quantization/run_benchmark.sh smoke
```

### 8.3 一次完整对比

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

这条命令的意义是：

- 先训练或复用一份 float baseline
- 再做 PTQ
- 再做 QAT
- 最后按 CPU deploy 口径输出结果

### 8.4 常见参数和代码位置怎么对照

如果你一边跑命令一边看源码，下面这张表最省时间：

| 参数 | 主要代码位置 | 作用 |
| --- | --- | --- |
| `--mode` | [quant_args.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_args.py:13) + [train.py](/root/codes/DLProjects/cv/image_classification/quantization/train.py:203) | 决定只跑 float、PTQ、QAT，还是跑完整 compare |
| `--backend` | [quant_args.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_args.py:177) + [quant_pt2e.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:69) | 决定 quantizer 和 deploy benchmark 口径 |
| `--calib-batches` | [quant_args.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_args.py:94) + [quant_pt2e.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:145) | 控制 PTQ calibration 跑多少个 batch |
| `--qat-device` | [quant_args.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_args.py:154) + [quant_pt2e.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:210) | 决定 QAT fake-quant 微调放在哪个设备上 |
| `--disable-observer-epoch` | [quant_args.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_args.py:124) + [quant_pt2e.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:116) | 控制 QAT 后期何时关闭 observer |
| `--freeze-bn-epoch` | [quant_args.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_args.py:130) + [quant_pt2e.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:122) | 控制 QAT 后期何时冻结 BN 统计量 |
| `--train-subset` / `--val-subset` | [quant_args.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_args.py:100) + [quant_core.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_core.py:146) | 控制烟测时的数据规模 |
| `--benchmark-warmup` / `--benchmark-iters` | [quant_args.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_args.py:136) + [quant_benchmark.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_benchmark.py:35) | 控制时延 benchmark 的预热和采样轮数 |

如果你只想跟一次完整链路，最推荐的源码顺序是：

1. 先看 [train.py](/root/codes/DLProjects/cv/image_classification/quantization/train.py:155)
2. 再看 [quant_core.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_core.py:111) 里的数据和 float baseline
3. 然后看 [quant_pt2e.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:153) 的 PTQ
4. 最后看 [quant_pt2e.py](/root/codes/DLProjects/cv/image_classification/quantization/quant_pt2e.py:208) 的 QAT

---

## 9. 输出目录里的文件怎么读

当前实验跑完后，你最常看到这些文件：

- `float_best.pth`
- `qat_prepared_best.pth`
- `benchmark_results.json`
- `benchmark_results.csv`
- `benchmark_results.md`

它们分别表示：

| 文件 | 用途 |
| --- | --- |
| `float_best.pth` | float 基线最优 checkpoint |
| `qat_prepared_best.pth` | QAT 训练期间保存的最佳 prepared state |
| `benchmark_results.json` | 程序化解析最方便 |
| `benchmark_results.csv` | 最适合表格横向比较 |
| `benchmark_results.md` | 最适合人工阅读和实验汇报 |

如果你只想先快速判断：

- PTQ 是否够用
- QAT 是否值得继续做

最值得先看的是：

1. `benchmark_results.md`
2. `benchmark_results.json`

---

## 10. 看结果之后下一步怎么做

### 10.1 PTQ 和 float 差距很小

这通常说明：

- 当前模型对 static int8 比较友好

优先动作：

- 先确认 deploy 延迟是否也有收益
- 如果有收益，通常优先落地 PTQ

### 10.2 PTQ 掉点明显，但 QAT 追回来

这通常说明：

- 量化误差是真实存在的
- 模型也确实能通过 fake quant 训练适应这种误差

优先动作：

- 保留 PTQ 作为低成本基线
- 再评估 QAT 的训练成本是否值得

### 10.3 PTQ 和 QAT 都不理想

更稳的排查顺序是：

1. 先看 float baseline 是否足够强
2. 再看 calibration 视角是否合理
3. 再看 backend / quantizer 是否适合当前模型
4. 最后才继续细调 QAT

不要一上来就继续堆更多 QAT epoch。

---

## 11. 当前这套示例刻意没有覆盖什么

为了让这条线足够清楚，当前仓库 **没有** 试图一次覆盖所有量化任务。

目前没有系统展开的部分包括：

- `ViT / LLM / Embedding / Attention` 为主的 `quantize_()` 实战
- 更细粒度的 observer / quantizer 定制
- 真实移动端 `ExecuTorch / XNNPACK` on-device benchmark
- weight-only / int4 / fp8 等更偏大模型或 GPU 的专题

这不是遗漏，而是刻意取舍：

- 先把 CNN + PT2E + static int8 这条主路径讲清楚

---

## 12. 下一步该读什么

如果你已经看懂这份文档，下一步通常有两个方向：

1. 回到代码目录，直接对照 `train.py`、`quant_pt2e.py`
2. 如果你想转去大模型 / `Linear` 主导量化，再回看 [torchao 量化路线总览](./torchao_quantization_guide.md)
