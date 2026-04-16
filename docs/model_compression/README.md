# 模型压缩文档

这个目录收纳“工程化导向”的模型压缩教程与说明文档。

当前先聚焦 **量化**，并按“先原理、再实现、最后落到 torchao 和仓库实战”的顺序拆成多份文档。

## 阅读顺序

如果你是第一次接触量化，建议按这个顺序读：

1. [量化基础原理：从公式到手写实现](./quantization_fundamentals.md)
2. [量化工作流：PTQ、QAT、Observer 与 Calibration](./quantization_workflows.md)
3. [torchao 量化路线总览与阅读指南](./torchao_quantization_guide.md)
4. [torchao PT2E 图像分类实战：对应本仓库实现](./torchao_pt2e_image_classification.md)

## 各文档分工

- [quantization_fundamentals.md](./quantization_fundamentals.md)
  - 面向初学者，只讲量化原理。
  - 重点讲清楚 `scale / zero_point`、对称/非对称量化、量化粒度、weight-only / dynamic / static 到底是什么，以及“如果自己手写，大概会怎么实现”。

- [quantization_workflows.md](./quantization_workflows.md)
  - 只讲 PTQ / QAT 工作流。
  - 重点讲清楚 observer、calibration、fake quant、为什么要 `prepare / convert`、为什么 QAT 训练和最终部署是两件事。

- [torchao_quantization_guide.md](./torchao_quantization_guide.md)
  - 只讲 torchao 路线。
  - 重点讲清楚 `quantize_()` 和 `PT2E` 各自适合什么场景，以及阅读 torchao 官方文档时该怎么对照前两份基础文档。

- [torchao_pt2e_image_classification.md](./torchao_pt2e_image_classification.md)
  - 只讲本仓库的图像分类量化示例。
  - 重点把 `cv/image_classification/quantization/` 的实现和前面的基础知识一一对应起来。

## 和代码目录的关系

当前仓库里对应的可运行示例仍然放在：

- `cv/image_classification/quantization/`

这种组织方式是刻意设计的：

- `docs/model_compression/` 负责讲清楚原理、路线和工程决策
- `cv/.../quantization/` 负责提供具体任务下可直接运行的脚本与 benchmark

如果以后仓库里继续补：

- `ViT / LLM` 的 `quantize_()` 实战
- 更细粒度的 quantizer / observer 定制
- 更通用的 benchmark / 导出组件

再把共性部分进一步抽到 `tooling/model_compression/` 会更自然。
