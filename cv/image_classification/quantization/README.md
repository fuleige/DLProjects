# 图像分类量化实验目录

这个目录保存的是 `torchao PT2E` 图像分类量化的**可运行实现**。

它主要回答三类实践问题：

1. 这套 `float -> PTQ / QAT -> benchmark` 流程在代码里怎么串起来
2. 常用命令怎么跑，关键参数该怎么选
3. 结果文件落在哪里，应该先看什么

如果你想先补齐原理和路线，再回来跑代码，建议先看：

- [模型压缩文档索引](../../../docs/model_compression/README.md)
- [torchao 量化路线总览](../../../docs/model_compression/torchao_quantization_guide.md)
- [torchao PT2E 图像分类实战：代表性代码讲解](../../../docs/model_compression/torchao_pt2e_image_classification.md)

---

## 1. 这套实现要解决什么问题

当前这条示例线故意只聚焦一个很具体的目标：

- 先拿到一个可靠的 float 图像分类 baseline
- 再对比 `PTQ` 和 `QAT`
- 最后按 CPU deploy 口径解释 benchmark

它更像一条：

- **教学导向的 CNN static int8 模板**

而不是：

- 任意 backbone 的量化工具箱
- LLM / ViT 的 `quantize_()` 示例
- 真实移动端 on-device benchmark 框架

当前实现默认遵守这条设备分工：

- float 训练默认走 GPU
- QAT fake quant 微调默认走 GPU
- 最终真实量化验证和 deploy benchmark 回到 CPU

这能把“训练设备”和“部署设备”分开讲清楚。

如果当前机器没有可用 CUDA，这套 benchmark 脚本不会再默认回退到 CPU 训练；你需要主动改参数，或者换到有 GPU 的环境里运行。

---

## 2. 一条完整流程怎么走

如果你想先建立全局图，可以把这套实现记成下面 5 步：

```text
parse args / build data
    ->
train or reuse float checkpoint
    ->
run PTQ or run QAT
    ->
build compare results
    ->
write benchmark artifacts
```

更具体一点，主流程对应下面这些函数：

1. [train.py](./train.py) 的 `main()`
   - 解析参数
   - 构造 transform、dataset、loader、输出目录

2. [quant_core.py](./quant_core.py) 的 `train_float_model(...)`
   - 训练或复用 float checkpoint

3. [quant_pt2e.py](./quant_pt2e.py) 的 `run_ptq(...)` 或 `run_qat(...)`
   - 承担真正的 PT2E 量化流程

4. [train.py](./train.py) 的 `build_float_result(...)` / `build_ptq_result(...)` / `build_qat_result(...)`
   - 把精度、延迟、显存/内存占用整理成统一结果

5. [quant_benchmark.py](./quant_benchmark.py) 的 `write_compare_artifacts(...)`
   - 落盘 `json / csv / md` 结果文件

### 2.1 PTQ 这条分支怎么走

`PTQ` 的主入口是：

- [quant_pt2e.py](./quant_pt2e.py) 里的 `run_ptq(...)`

它的顺序很固定：

1. 从 float checkpoint 恢复模型
2. `torch.export`
3. `build_quantizer(...)`
4. `prepare_pt2e(...)`
5. 跑 calibration
6. `convert_pt2e(...)`
7. 在 CPU 上评估和 benchmark

### 2.2 QAT 这条分支怎么走

`QAT` 的主入口是：

- [quant_pt2e.py](./quant_pt2e.py) 里的 `run_qat(...)`

它和 PTQ 的最大区别是：

- 中间不是 calibration
- 而是 `prepare_qat_pt2e(...)` 之后继续做 fake quant 微调

它的顺序可以先记成：

1. 从同一个 float checkpoint 恢复模型
2. `torch.export`
3. `prepare_qat_pt2e(...)`
4. 在 `qat_device` 上继续训练
5. 后期关闭 observer、冻结 BN 统计量
6. 把 prepared model 转回 CPU
7. `convert_pt2e(...)`
8. 在 CPU 上评估和 benchmark

### 2.3 `compare` 模式到底在干什么

`compare` 不是三条线各自独立训练。

它做的是：

1. 先训练或复用一份 float checkpoint
2. 用这份 checkpoint 跑 PTQ
3. 再用同一份 checkpoint 跑 QAT
4. 把结果统一写到同一个实验目录

这更符合真实工程里的决策顺序：

- 先 float baseline
- 再 PTQ
- PTQ 不够再 QAT

---

## 3. 目录怎么读

如果你准备直接看代码，下面这张表最省时间：

| 文件 | 主要职责 | 什么时候先看 |
| --- | --- | --- |
| [train.py](./train.py) | 总入口，串起 float / PTQ / QAT / compare | 你想先看整条流程 |
| [quant_args.py](./quant_args.py) | CLI 参数定义 | 你要对照命令行和代码 |
| [quant_core.py](./quant_core.py) | 数据、transform、训练、验证、checkpoint | 你想看 baseline 和数据视角 |
| [quant_pt2e.py](./quant_pt2e.py) | `export`、quantizer、PTQ / QAT 主流程 | 你想看量化到底在哪做 |
| [quant_benchmark.py](./quant_benchmark.py) | CPU eager / deploy benchmark、结果落盘 | 你想看延迟和结果文件怎么来的 |
| [quant_types.py](./quant_types.py) | dataclass 和模型规格 | 你想看结果结构和模型元信息 |
| [run_benchmark.sh](./run_benchmark.sh) | `smoke / formal` 一键入口 | 你想快速复现实验 |
| [BENCHMARK.md](./BENCHMARK.md) | 已有实验记录 | 你想看一份完整结果示例 |

最推荐的阅读顺序是：

1. 先看 [train.py](./train.py)
2. 再看 [quant_core.py](./quant_core.py)
3. 然后看 [quant_pt2e.py](./quant_pt2e.py)
4. 最后看 [quant_benchmark.py](./quant_benchmark.py)

---

## 4. 常用运行方式

下面这些命令默认假设你已经进入了安装好 `torch / torchvision / torchao` 的 Python 环境。

如果当前 shell 里的 `python3` 不是项目环境，可以这样指定解释器：

```bash
PYTHON=/path/to/python bash cv/image_classification/quantization/run_benchmark.sh smoke
```

### 4.1 查看支持的模型

```bash
python3 cv/image_classification/quantization/train.py --list-models
```

### 4.2 快速烟测

```bash
bash cv/image_classification/quantization/run_benchmark.sh smoke
```

适合检查：

- 参数能不能跑通
- 路径和依赖是否正常
- 输出文件是否按预期生成

### 4.3 一次完整对比

```bash
python3 cv/image_classification/quantization/train.py \
  --mode compare \
  --dataset-type cifar100 \
  --data-root ./datasets \
  --model-name resnet18 \
  --backend x86_inductor \
  --float-device cuda \
  --qat-device cuda \
  --output-dir ./outputs/image_classification/torchao_quantization_trial
```

这条命令会：

1. 训练或复用 float checkpoint
2. 跑 PTQ
3. 跑 QAT
4. 输出统一结果文件

### 4.4 只跑 PTQ

```bash
python3 cv/image_classification/quantization/train.py \
  --mode ptq \
  --dataset-type cifar100 \
  --model-name resnet18 \
  --backend x86_inductor \
  --float-device cuda
```

适合先做一轮低成本验证。

### 4.5 只跑 QAT

```bash
python3 cv/image_classification/quantization/train.py \
  --mode qat \
  --dataset-type cifar100 \
  --model-name resnet18 \
  --backend x86_inductor \
  --float-device cuda \
  --qat-device cuda
```

适合你已经确认 PTQ 掉点明显，想看 QAT 能否追回精度。

### 4.6 自定义数据集

目录需要满足 ImageFolder 风格：

```text
your_dataset/
├── train/
│   ├── class_a/
│   └── class_b/
└── val/
    ├── class_a/
    └── class_b/
```

运行方式：

```bash
python3 cv/image_classification/quantization/train.py \
  --mode compare \
  --dataset-type custom \
  --train-dir ./datasets/your_dataset/train \
  --val-dir ./datasets/your_dataset/val \
  --model-name mobilenet_v3_small \
  --backend x86_inductor \
  --float-device cuda \
  --qat-device cuda
```

---

## 5. 最重要的参数

如果你不想一开始就读完整个 `quant_args.py`，先抓住下面这些参数：

| 参数 | 作用 | 什么时候最值得先看 |
| --- | --- | --- |
| `--mode` | 决定只跑 float、PTQ、QAT，还是跑完整 compare | 你在确认流程范围 |
| `--backend` | 选择 PT2E quantizer 和 deploy benchmark 口径 | 你在确认后端目标 |
| `--float-device` | float 训练设备，默认 `cuda`，不会再静默回退到 CPU | 你在调训练资源 |
| `--qat-device` | QAT fake quant 微调设备，默认 `cuda` | 你在调微调资源 |
| `--calib-batches` | PTQ calibration 用多少个 batch | 你在调校准代表性 |
| `--batch-size` | 训练/验证批大小 | 你在调速度或显存；注意 QAT 当前要求 `>= 2` |
| `--train-subset` / `--val-subset` | 只取前 N 个样本 | 你在做快速烟测 |
| `--max-train-batches` / `--max-val-batches` | 每轮最多跑多少个 batch | 你在缩短实验时间 |
| `--reuse-float-checkpoint` | 是否复用已有 float checkpoint | 你在反复对比 PTQ / QAT |
| `--output-dir` | 结果落盘目录 | 你在整理实验输出 |

### 5.1 两个最容易忽略的限制

#### 1. QAT 当前要求 `--batch-size >= 2`

原因是：

- 当前 QAT 导出的动态图约束要求 `min_batch=2`
- 训练和验证也会跳过小于 2 的 batch

#### 2. `--train-subset` 会同时影响训练集和校准集

也就是说你把训练样本裁小了，PTQ calibration 的覆盖范围也会一起变窄。

这很适合烟测，但正式比较时要有意识。

---

## 6. 输出目录里会看到什么

当前实验跑完后，最常见的文件是：

| 文件 | 用途 |
| --- | --- |
| `float_best.pth` | float 基线最优 checkpoint |
| `qat_prepared_best.pth` | QAT 训练期间保存的最佳 prepared state |
| `benchmark_results.json` | 最适合程序化解析 |
| `benchmark_results.csv` | 最适合表格横向比较 |
| `benchmark_results.md` | 最适合人工阅读和实验汇报 |

输出目录一般会像这样组织：

```text
outputs/image_classification/torchao_quantization/<model_name>/<experiment_tag>/
```

其中 `experiment_tag` 来自：

- 数据集类型
- 是否 pretrained
- 自定义数据集时的目录指纹

---

## 7. 怎么读结果

如果你只想快速判断这次实验值不值得继续往下做，优先看：

1. `benchmark_results.md`
2. `benchmark_results.json`

### 7.1 先看哪几列

- `Top1 / TopK`
  - 看精度变化
- `Eager(ms)`
  - 看普通 CPU eager 推理延迟
- `Deploy(ms)`
  - 看更接近部署口径的延迟
- `Speedup`
  - 看相对 float 的速度变化
- `RSS Delta(MB)` / `Train Peak CUDA(MB)`
  - 看内存和训练资源开销

### 7.2 `x86_inductor` 和 `xnnpack` 的结果口径不一样

当前代码里：

- `x86_inductor` 会尝试用 `torch.compile(inductor)` 生成 deploy 延迟
- `xnnpack` 不会在这个脚本里直接产出真正的 on-device deploy 延迟

所以如果你选的是 `xnnpack`：

- 更应该先看 eager CPU 延迟和精度
- 真正的移动端 / ExecuTorch 部署结果需要到对应运行时再测

---

## 8. 这份 README 和文档目录怎么分工

当前推荐分工是：

- [README.md](./README.md)
  - 负责讲怎么跑、怎么读目录、参数和结果怎么看

- [torchao_pt2e_image_classification.md](../../../docs/model_compression/torchao_pt2e_image_classification.md)
  - 只讲最有代表性的代码片段和流程设计

- [torchao_quantization_guide.md](../../../docs/model_compression/torchao_quantization_guide.md)
  - 负责讲 `quantize_()` 和 `PT2E` 的路线差别

- [quantization_workflows.md](../../../docs/model_compression/quantization_workflows.md)
  - 负责讲 `PTQ / QAT / observer / calibration`

如果你接下来要继续深入代码，最建议直接去看：

- [torchao PT2E 图像分类实战：代表性代码讲解](../../../docs/model_compression/torchao_pt2e_image_classification.md)
