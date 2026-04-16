# 图像分类量化实验目录

这个目录保存的是 **可运行实现和实验产物入口**，不再承担完整教程的职责。

如果你想系统理解 `torchao` 的 `PTQ / QAT / PT2E` 路线、训练与部署设备如何拆分、不同 backend 该怎么解读，请直接看工程化教程：

- [torchao 量化指南](../../../docs/model_compression/torchao_quantization_guide.md)
- [模型压缩文档索引](../../../docs/model_compression/README.md)

如果你想直接运行实验，看这个目录即可。

## 1. 目录职责

- `train.py`：总入口，串起 float / PTQ / QAT 流程
- `quant_args.py`：CLI 参数与模型清单
- `quant_core.py`：模型、数据、训练、验证、checkpoint
- `quant_pt2e.py`：PT2E 导出、quantizer、PTQ / QAT 主流程
- `quant_benchmark.py`：CPU benchmark、结果表格、结果落盘
- `quant_types.py`：数据结构和模型规格
- `run_benchmark.sh`：`smoke / formal` 一键复现实验入口
- `BENCHMARK.md`：当前目录下的实验记录和结论

## 2. 当前实现原则

- float 训练优先用 GPU
- QAT fake-quant 微调优先用 GPU
- 最终真实量化验证和 deploy benchmark 再回到 CPU

这个原则对应的是更真实的工程场景：

- 训练和微调尽量利用本机 GPU
- 最终部署指标按 CPU 口径做对比

## 3. 快速开始

先看支持的模型：

```bash
python3 cv/image_classification/quantization/train.py --list-models
```

快速烟测：

```bash
bash cv/image_classification/quantization/run_benchmark.sh smoke
```

正式 benchmark：

```bash
bash cv/image_classification/quantization/run_benchmark.sh formal
```

如果你想手动指定 QAT 走 GPU：

```bash
python3 cv/image_classification/quantization/train.py \
  --mode compare \
  --dataset-type cifar100 \
  --model-name resnet18 \
  --backend x86_inductor \
  --qat-device cuda
```

## 4. 结果文件

每次运行完成后，会在实验输出目录下自动落盘：

- `benchmark_results.json`
- `benchmark_results.csv`
- `benchmark_results.md`

这些文件分别适合：

- `json`：程序化解析
- `csv`：表格整理
- `md`：实验记录和汇报

## 5. 阅读顺序建议

如果你是第一次接触这条线，建议按这个顺序看：

1. 先读 [torchao 量化指南](../../../docs/model_compression/torchao_quantization_guide.md)
2. 再看 `train.py` 如何把 float / PTQ / QAT 串起来
3. 最后看 [BENCHMARK.md](./BENCHMARK.md) 理解结果该怎么解释

这样更符合这个仓库的分工：

- `docs/` 负责讲清楚方法和工程决策
- `cv/.../quantization/` 负责保留具体任务里的可运行实现
