# 模型压缩文档

这个目录收纳“工程化导向”的模型压缩教程与说明文档。

当前已补充：

- [torchao 量化指南：从基础理论到 PTQ / QAT 图像分类落地](./torchao_quantization_guide.md)

这类文档的定位是：

- 解释基础理论、方法选择、工程决策和常见坑点
- 统一沉淀跨实验可复用的使用指南
- 与具体任务代码目录解耦

当前仓库里对应的可运行示例仍然放在：

- `cv/image_classification/quantization/`

这种组织方式刻意把“教程”和“代码”分开：

- `docs/model_compression/` 负责讲清楚方法和工程化原则
- `cv/.../quantization/` 负责提供具体任务下可直接运行的脚本与 benchmark
