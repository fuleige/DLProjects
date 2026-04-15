# 图像分类中的 torchao 量化实战：PTQ、QAT 与方法对比

这个目录对应的可运行脚本：

- `cv/image_classification/quantization/train.py`

目标不是只讲 API，而是把图像分类里最常见的量化路线讲清楚，并给出一套能直接套到自己项目里的模板。

---

## 1. 先说结论：图像分类优先怎么选

如果你的任务是典型图像分类，模型主体是：

- `Conv2d + BatchNorm + ReLU + Linear`
- 例如 `ResNet`、`MobileNet`、`EfficientNet` 这一类 CNN

那么 `torchao` 下更推荐的主路线是：

- `PTQ`：`torchao.quantization.pt2e.prepare_pt2e`
- `QAT`：`torchao.quantization.pt2e.prepare_qat_pt2e`

也就是官方现在主推的 `PT2E` 路线。

不要把 `quantize_()` 当成图像分类 CNN 的默认主方案。按当前 stable 文档：

- `quantize_()` 的默认 `filter_fn` 主要针对 `Linear`
- `QATConfig` 目前也明确写了，作为 base config 的常见支持仍主要集中在 `Linear/Embedding`

这意味着：

- 对标准 CNN 图像分类，优先用 `PT2E static quantization`
- 对 `ViT/Swin/MLP-Mixer` 这类 `Linear` 占主导的模型，再重点考虑 `quantize_()` 这条线

---

## 2. 三种常见方案怎么理解

### 2.1 FP32 浮点基线

这是所有量化实验的起点。

你先要有一个能正常训练和验证的浮点模型，否则后面的量化结果没有比较意义。

### 2.2 PTQ：Post-Training Quantization

完整路径：

1. 训练好一个 float 模型
2. `export`
3. `prepare_pt2e`
4. 用代表性样本做 `calibration`
5. `convert_pt2e`

优点：

- 不需要重新训练整个模型
- 改造成本低
- 最适合先做部署验证

缺点：

- 精度通常会比 float 低一点
- 如果模型对激活分布很敏感，PTQ 掉点会比较明显

### 2.3 QAT：Quantization-Aware Training

完整路径：

1. 从一个已经训练好的 float checkpoint 出发
2. `export`
3. `prepare_qat_pt2e`
4. 用 fake quant 的方式继续训练若干 epoch
5. `convert_pt2e`

优点：

- 通常比 PTQ 更能恢复 int8 精度
- 对难量化模型更稳

缺点：

- 成本更高，需要继续训练
- 训练流程要处理 observer、BatchNorm 冻结这些细节

---

## 3. 这份示例为什么选 torchvision，不继续沿用当前的 timm 示例

仓库里已经有一个常规分类示例：

- `cv/image_classification/example.py`

它的定位是：

- 展示现代分类模型如何训练和迁移学习
- 默认使用 `timm`

但量化教程故意没有直接基于它扩展，原因很实际：

- `torchao PT2E` 图像分类官方教程主要围绕标准 CNN
- 任意 `timm` 模型的 `export` 和 quantizer 覆盖不一定都稳定
- 教学上先把量化流程讲清楚，比先处理模型导出兼容性更重要

所以这里单独加了一份量化模板，默认支持：

- `resnet18`
- `mobilenet_v3_small`

这两个模型有两个好处：

- 结构清晰，便于理解 `Conv/BN` 为什么适合 static int8
- 量化后更容易看出 PTQ 和 QAT 的差异

---

## 4. 环境要求

建议至少准备：

```bash
pip install torch torchvision torchao
```

如果你想完全跟官方图像分类 PTQ/QAT 教程对齐，用 `xnnpack` backend，还要装：

```bash
pip install executorch
```

另外要注意一件很现实的事：

- `torchao` 必须和当前 `torch` 版本匹配
- 如果导入时出现类似 `torch.int1`、`module has no attribute ...` 这类错误，通常不是脚本写错，而是版本不匹配
- 这时优先先修环境，再跑 PTQ / QAT

这份脚本默认用的是：

- `--backend x86_inductor`

原因是它依赖更少，更适合普通 x86 CPU 环境先把流程跑通。

如果你更关注：

- 与官方 `XNNPACKQuantizer` 教程一致
- 或后续更接近移动端 / ExecuTorch 链路

可以改成：

```bash
--backend xnnpack
```

---

## 5. 代码里已经帮你处理了哪些关键细节

脚本位置：

- `cv/image_classification/quantization/train.py`

它不是只有单个 API demo，而是把真正实战里常遗漏的细节也补上了。

### 5.1 数据集拆成了 3 份视角

- `train_dataset`：训练增强
- `calib_dataset`：校准专用，使用 eval transform
- `val_dataset`：验证集，使用 eval transform

这是因为 PTQ 的 `calibration` 不应该用强数据增强后的分布直接代替推理分布。

### 5.2 compare 模式会自动跑完整对比

它会顺序完成：

1. float 基线训练或加载
2. PTQ
3. QAT
4. 输出对比表

对比项包括：

- `Top1`
- `Top5`
- `state_dict` 大小
- CPU 单批平均延迟

### 5.3 QAT 里补了两个常见细节

- 在指定 epoch 后关闭 observer
- 在指定 epoch 后冻结 BatchNorm 统计量

这两个动作在 QAT 里很常见，不补齐的话，你很难解释为什么 QAT 后期精度不稳定。

### 5.4 QAT 验证用的是“量化后真实图”

不是直接拿 prepared model 的训练图当最终结果，而是：

1. `copy.deepcopy(prepared_model)`
2. `convert_pt2e`
3. 在量化后的模型上做验证

这样更贴近最终部署效果。

---

## 6. 最常用的运行方式

### 6.1 先看支持的模型

```bash
python3 cv/image_classification/quantization/train.py --list-models
```

### 6.2 CIFAR-100 上快速烟测

这条命令适合先确认流程没问题：

```bash
python3 cv/image_classification/quantization/train.py \
  --mode compare \
  --dataset-type cifar100 \
  --data-root ./datasets \
  --model-name resnet18 \
  --float-epochs 1 \
  --qat-epochs 1 \
  --train-subset 2000 \
  --val-subset 1000 \
  --calib-batches 10 \
  --max-train-batches 20 \
  --max-val-batches 10 \
  --batch-size 64 \
  --backend x86_inductor
```

这条命令更像“验证代码可跑通”。

### 6.3 CIFAR-100 上做一版更像样的对比

```bash
python3 cv/image_classification/quantization/train.py \
  --mode compare \
  --dataset-type cifar100 \
  --data-root ./datasets \
  --model-name resnet18 \
  --float-epochs 3 \
  --qat-epochs 2 \
  --calib-batches 50 \
  --batch-size 64 \
  --backend x86_inductor
```

### 6.4 只跑 PTQ

```bash
python3 cv/image_classification/quantization/train.py \
  --mode ptq \
  --dataset-type cifar100 \
  --model-name resnet18 \
  --float-epochs 3 \
  --calib-batches 50
```

适合你已经训练好了 float checkpoint，只想看“只做校准到底会掉多少点”。

### 6.5 只跑 QAT

```bash
python3 cv/image_classification/quantization/train.py \
  --mode qat \
  --dataset-type cifar100 \
  --model-name resnet18 \
  --float-epochs 3 \
  --qat-epochs 2 \
  --qat-learning-rate 1e-4
```

### 6.6 切到官方更常见的 xnnpack 量化器

```bash
python3 cv/image_classification/quantization/train.py \
  --mode compare \
  --dataset-type cifar100 \
  --model-name resnet18 \
  --backend xnnpack
```

如果这里报 `executorch` 缺失，就先安装它。

### 6.7 跑自定义 ImageFolder 数据集

目录结构应满足：

```text
datasets/my_cls/
├── train/
│   ├── class_a/
│   └── class_b/
└── val/
    ├── class_a/
    └── class_b/
```

运行命令：

```bash
python3 cv/image_classification/quantization/train.py \
  --mode compare \
  --dataset-type custom \
  --train-dir ./datasets/my_cls/train \
  --val-dir ./datasets/my_cls/val \
  --model-name mobilenet_v3_small \
  --float-epochs 5 \
  --qat-epochs 2 \
  --backend x86_inductor
```

---

## 7. 这份脚本里的 4 个模式分别干什么

### 7.1 `train_float`

只训练浮点模型并输出 float 基线指标。

适合：

- 你先只想确认分类任务本身能收敛
- 暂时不进入量化

### 7.2 `ptq`

流程是：

1. 训练或加载 float checkpoint
2. 导出 eval 图
3. `prepare_pt2e`
4. 跑 calibration
5. `convert_pt2e`
6. 验证量化模型

适合：

- 先评估部署可行性
- 你不想增加训练成本

### 7.3 `qat`

流程是：

1. 训练或加载 float checkpoint
2. 导出 train 图
3. `prepare_qat_pt2e`
4. 继续训练
5. 周期性将 prepared model 转成真实量化模型做验证
6. 取 best prepared state，再做最终 convert

适合：

- PTQ 掉点明显
- 量化精度要求高

### 7.4 `compare`

这是最推荐的模式。

它会一次性输出：

- float32
- ptq_int8
- qat_int8

让你立刻看到 3 个问题：

- 精度掉了多少
- 模型大小缩了多少
- CPU 延迟有没有改善

---

## 8. PTQ 里最容易忽略的细节

### 8.1 calibration 不是“随便喂几张图”

`calibration` 的意义是让 observer 看到真实推理分布，从而决定量化 scale/zero-point。

所以你应优先保证：

- 样本分布接近真实业务输入
- 图像预处理与推理一致

而不是一味追求 batch 数越多越好。

这也是为什么脚本里专门准备了：

- `calib_dataset = 训练集内容 + eval transform`

### 8.2 PTQ 一定从 eval 图导出

PTQ 是后训练静态量化。

所以它的导出顺序应是：

1. 先把 float model 切到 `eval`
2. 再 `torch.export.export(...)`
3. 再 `prepare_pt2e`

否则 BatchNorm / Dropout 行为会不对。

### 8.3 PTQ 更适合作为第一步

如果你现在刚开始做量化，不建议直接上 QAT。

更合理的顺序是：

1. 先拿 float 基线
2. 直接试 PTQ
3. 如果 PTQ 掉点可以接受，就先落地
4. 只有 PTQ 掉点不可接受，再上 QAT

---

## 9. QAT 里最容易忽略的细节

### 9.1 QAT 不是从随机初始化开始最划算

图像分类里更常见的做法是：

- 先有一个训练好的 float checkpoint
- 再基于它做几轮 QAT 微调

这样成本更低，也更符合部署项目实际。

### 9.2 QAT 导出的是 train 图

QAT 的核心是：

- 训练时在图里插 fake quant

所以它和 PTQ 不同，不是先走 eval 图。

### 9.3 observer 和 BN 冻结要在后期处理

在 QAT 前几轮：

- observer 继续统计
- BN 继续更新

到后面若干轮：

- observer 关闭
- BN 统计冻结

这是因为训练后期你更希望量化参数和归一化统计稳定下来，而不是继续漂移。

### 9.4 不要直接把 prepared model 当最终部署模型

`prepared_model` 只是训练态图。

最终部署前，一定要：

1. `convert_pt2e`
2. 在真实量化模型上验证

---

## 10. 结果应该怎么解读

通常会看到下面几种情况。

### 10.1 PTQ 和 float 差距很小

说明你的模型和数据分布对 static int8 比较友好。

这时通常没有必要为了多追回一点点精度再上 QAT。

### 10.2 PTQ 掉点明显，但 QAT 能追回来

这是最典型的“QAT 值得做”的场景。

尤其是当你：

- 目标平台必须 int8
- 精度损失又不能接受

### 10.3 模型大小明显下降，但延迟收益不明显

这并不奇怪。

模型大小下降主要来自：

- 权重从 float32 变成 int8

但延迟是否改善，还取决于：

- backend
- kernel 是否命中优化路径
- batch size
- 你的 CPU / 移动设备架构

所以“模型变小”不等于“一定更快”。

---

## 11. `x86_inductor` 和 `xnnpack` 怎么选

### 11.1 `x86_inductor`

更适合：

- 先在普通 x86 CPU 环境把流程跑通
- 不想多装 `executorch`

### 11.2 `xnnpack`

更适合：

- 希望更接近 torchao 官方图像分类 PTQ/QAT 教程
- 后续要走 ExecuTorch / 移动端部署链路

一个务实建议是：

1. 本地先用 `x86_inductor` 验证流程
2. 确认方法有效后，再切换 `xnnpack` 看最终部署链路

---

## 12. 那 `quantize_()` 在图像分类里什么时候用

按当前官方文档：

- `quantize_()` 默认更偏 `Linear`
- `QATConfig` 也明确说明它的常见 base config 支持集中在 `Linear/Embedding`

所以在图像分类里，更合适的场景通常是：

- `ViT`
- `Swin`
- `MLP-Mixer`
- 或其他 `Linear` 占大头的模型

如果你是标准 CNN，优先顺序应该是：

1. `PT2E PTQ`
2. `PT2E QAT`

而不是一上来就走 `quantize_()`

---

## 13. 常见报错和排查建议

### 13.1 `No module named torchao`

说明当前 Python 环境里没有装 `torchao`，或者你装到了别的环境。

### 13.2 `No module named executorch`

通常是你把：

- `--backend xnnpack`

打开了，但没有装 `executorch`。

### 13.3 export 失败

先不要急着怀疑量化。

先确认：

- 模型本身能否被 `torch.export.export`
- 前向里有没有非常动态、非常自定义的 Python 控制流

### 13.4 PTQ 精度掉太多

优先排查：

- calibration 样本是否代表真实输入
- 预处理是否和推理一致
- float 基线是否本身就不稳

### 13.5 QAT 后期不稳定

优先排查：

- 学习率是不是过大
- observer 是否及时关闭
- BN 是否及时冻结

---

## 14. 一套建议的落地顺序

如果你要把这套方法迁移到自己的图像分类项目，建议按下面顺序来：

1. 先把 float 基线训练稳定
2. 直接跑 PTQ，看精度损失是否可接受
3. 如果 PTQ 掉点过大，再加 QAT
4. 对比 float / PTQ / QAT 的精度、大小、延迟
5. 最后再迁移到你真正要部署的 backbone

这个顺序比一上来就调一堆量化参数更稳。

---

## 15. 参考资料

以下是我写这份教程时对照的官方资料：

- PT2E 总览：https://docs.pytorch.org/ao/stable/pt2e_quantization/index.html
- 图像分类 PTQ 教程：https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_ptq.html
- 图像分类 QAT 教程：https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_qat.html
- `quantize_()` API：https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.quantize_.html
- `QATConfig` API：https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.qat.QATConfig.html
- PyTorch 量化迁移说明：https://docs.pytorch.org/docs/stable/quantization.html

如果你下一步想做的是：

- 迁移到你自己的分类训练脚本
- 换成 `timm` 模型
- 加上导出到 ONNX / ExecuTorch

最稳妥的做法仍然是先以这份模板为基线，再一点点替换模块，而不是一开始就全量重写。
