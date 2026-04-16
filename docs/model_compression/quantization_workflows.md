# 量化工作流：PTQ、QAT、Observer 与 Calibration

这份文档只回答两个问题：

1. **PTQ 和 QAT 分别是怎么做出来的？**
2. **observer、calibration、fake quant 到底在流程里扮演什么角色？**

如果上一份《量化基础原理》解决的是“量化是什么”，那这一份解决的是：

- **量化工程流程到底怎么落地。**

---

## 1. 先建立整体图

对初学者来说，最重要的是先分清三件事：

1. 你先得有一个可靠的 float baseline
2. PTQ 是“训练后再量化”
3. QAT 是“训练时就让模型适应量化误差”

### 1.1 最基础的决策顺序

通常最稳的顺序是：

1. 先训练或拿到稳定的 float 模型
2. 先试 PTQ
3. PTQ 不够，再试 QAT

原因很简单：

- PTQ 成本最低
- QAT 成本更高
- 很多模型其实 PTQ 就够了

### 1.2 一张工作流总图

你可以先把三条常见路线记成下面这张图：

```text
float baseline
   |
   |--> weight-only
   |      |
   |      -> 量化权重 -> 推理
   |
   |--> PTQ
   |      |
   |      -> prepare -> calibration -> convert -> 推理
   |
   |--> QAT
          |
          -> prepare_qat -> fake quant 训练 -> convert -> 推理
```

这张图想表达的重点是：

- 三条路线都从 float baseline 出发
- PTQ 和 QAT 的区别不在“起点不同”
- 而在于中间到底有没有继续训练

---

## 2. PTQ：训练好以后再量化

PTQ 是 Post-Training Quantization。

它的核心思想是：

- float 模型已经训练好了
- 不再做正式训练
- 只通过 calibration 估计激活范围，然后把模型转换成量化模型

### 2.1 PTQ 的标准流程

可以先记成这 5 步：

1. 训练好 float 模型
2. 准备 calibration 数据
3. 插入 observer 或 prepare 量化图
4. 跑 calibration，收集激活统计
5. convert 成真实量化模型

### 2.1.1 为什么 PTQ 通常先于 QAT

因为 PTQ 最像一个“诊断工具”：

- 成本低
- 路径短
- 能快速告诉你这个模型对量化是否友好

如果一个模型：

- PTQ 几乎不掉点

那通常没必要一上来就付出 QAT 的训练成本。

所以在真实项目里，PTQ 常常不是“低配版 QAT”，而是：

- 第一轮决策工具

### 2.2 calibration 到底在做什么

calibration 本质上是：

- 用一批代表性输入跑模型
- 让 observer 看到真实激活分布
- 由 observer 根据这些统计量计算每层激活的 `scale / zero_point`

它不是训练，因为：

- 不更新权重
- 不做反向传播
- 不做优化器 step

它更像是：

- 给量化图“测量一下真实分布”

### 2.2.1 calibration 数据一定要单独切一份吗

不一定。

真正重要的不是“物理上是不是独立目录”，而是：

- 它是否代表真实部署输入分布
- 它的预处理是否和推理一致

所以现实里常见三种做法：

1. 从训练集里抽一部分样本，但用评估态预处理
2. 单独准备一份 calibration split
3. 直接使用真实线上样本的离线抽样

对初学者来说，先记住这个更重要：

- calibration 的关键是“代表性”
- 而不是“名字叫不叫独立数据集”

### 2.3 observer 到底是什么

observer 是一个“统计器”。

它最常记录这些信息：

- `min / max`
- moving-average min/max
- histogram

这些统计量最后会被转成：

- `scale`
- `zero_point`

所以 observer 的职责非常单纯：

- **先看分布**
- **再算量化参数**

### 2.4 一个最小版 PTQ 伪代码

```python
model = load_float_model()
insert_observers(model)

for x in calib_loader:
    _ = model(x)  # 只前向，不训练

freeze_observer_stats(model)
quantized_model = convert_to_quantized_model(model)
```

初学者最容易漏掉的一点是：

- observer 不是凭空知道量化范围的
- 它必须先看过代表性输入

### 2.5 为什么 PTQ 会掉点

最常见的原因有四类：

1. calibration 数据不够代表真实输入
2. 激活分布长尾明显，简单 min/max 容易被 outlier 拉坏
3. 某些层对量化特别敏感
4. backend 的实际量化配置不适合这个模型

所以 PTQ 的本质问题不是“有没有量化”，而是：

- 量化参数是不是估得足够合理

### 2.6 一个够用的 PTQ 排查顺序

如果 PTQ 掉点明显，建议按这个顺序排查：

1. 先看 float baseline 是否本身就不稳
2. 再看 calibration 样本是否代表真实输入
3. 再看 calibration transform 是否和推理一致
4. 再看当前量化粒度是否过粗
5. 最后才考虑直接上 QAT

这个顺序很重要，因为很多人一看到 PTQ 掉点，就立刻想：

- “那我直接做 QAT 吧”

但现实里更常见的问题是：

- baseline 本身不够强
- calibration 视角不对

---

## 3. QAT：训练时就让模型适应量化误差

QAT 是 Quantization-Aware Training。

它的核心思想是：

- 前向时模拟量化误差
- 让模型在训练期就慢慢适应这些误差
- 训练结束后再转成真实量化模型

### 3.1 QAT 为什么通常更准

因为 PTQ 的模型从头到尾都是按 float 分布学出来的。

一到部署时突然被量化，它会遇到：

- 舍入误差
- 截断误差
- 激活范围变化

而 QAT 会在训练期提前把这些误差“演练”进去。

所以模型参数会逐渐学会：

- 如何在量化误差存在时仍然保持任务精度

### 3.1.1 为什么 QAT 通常从 float checkpoint 继续做

因为 QAT 的目标不是：

- 从零开始重新学一个量化模型

而是：

- 从已经学好的 float 模型继续微调
- 让它适应量化误差

所以 QAT 最常见的工程起点不是：

- 随机初始化

而是：

- 一个已经足够稳定的 float baseline checkpoint

### 3.2 fake quant 到底是什么

QAT 的核心机制是 fake quantization。

它的意思不是：

- 训练时真的把所有张量都换成 int8

而是：

- 前向时，先做一次“量化 -> 反量化”的数值模拟
- 反向时仍然按 float 图训练

也就是：

```text
x_float
  -> fake_quant(x)
  -> x_hat_float
  -> 后续层
```

这里的 `x_hat_float` 仍然是 float Tensor，但它已经带上了量化误差。

### 3.3 一个最小版 fake quant 伪代码

```python
def fake_quantize(x, scale, zero_point, qmin, qmax):
    q = torch.round(x / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)
    x_hat = (q - zero_point) * scale
    return x_hat
```

QAT 训练时，模型前向会不断遇到这种 `x_hat`，所以它会被迫适应：

- “部署后我不再是精确 float 了”

### 3.4 一个最小版 QAT 伪代码

```python
model = load_float_model()
insert_fake_quant_modules(model)
optimizer = AdamW(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for x, y in train_loader:
        logits = model(x)  # 前向里会经过 fake quant
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

quantized_model = convert_to_quantized_model(model)
```

这里最重要的区别是：

- PTQ 不训练
- QAT 会继续训练或微调

### 3.5 QAT 训练时到底在优化什么

从实现角度看，QAT 训练的目标仍然是原始任务损失，例如：

- 分类交叉熵
- 检测损失
- 语言建模损失

并不是另外引入一个“量化损失”。

区别在于：

- 前向数值已经被 fake quant 改变了

所以优化器看到的是：

- “存在量化误差时的任务损失”

这就是为什么 QAT 可以不改任务目标，却仍然学会适应量化部署。

---

## 4. 为什么 QAT 里还要有 observer

很多初学者会疑惑：

- 既然 QAT 已经在训练了，为什么还需要 observer？

因为 fake quant 也需要量化参数。

而量化参数一开始并不知道，所以训练前期依然需要 observer 去估计：

- 当前权重范围
- 当前激活范围

QAT 的常见做法是：

1. 前期 observer 正常更新
2. 中后期关闭 observer
3. 固定量化范围继续训练

为什么后面要关掉？

因为如果 observer 一直更新：

- 量化区间会不停漂移
- 训练后期会更不稳定

这就是很多教程里会出现两个阶段的原因：

1. 前期：继续估范围
2. 后期：固定范围，稳定训练

---

## 5. 为什么 QAT 里经常还要冻结 BatchNorm

这在 CNN 里尤其常见。

原因是：

- BatchNorm 的统计量也会影响数值分布
- 如果 BN 统计和 observer 范围都在后期继续变化
- QAT 训练会变得很不稳定

所以常见做法是：

1. 前期让 BN 正常更新
2. 中后期冻结 BN 统计量
3. 同时让 observer 停止更新

这样模型后期面对的是一个相对稳定的量化环境。

---

## 6. `prepare` 和 `convert` 为什么要分开

这一步是很多框架教程里最容易写得很快、但初学者最容易看懵的。

可以先这么理解：

### 6.1 `prepare`

`prepare` 解决的是：

- 哪些地方要量化
- 量化参数怎么收集
- 图里该插什么 observer / fake quant 逻辑

也就是说，`prepare` 后的模型还不是最终部署模型。

它只是进入了：

- 可校准
- 可做 QAT

的“准备态”。

### 6.2 `convert`

`convert` 解决的是：

- 把准备态模型真正替换成量化表示
- 把 observer / fake quant 等训练期逻辑替换掉
- 得到更接近真实部署的量化图

所以：

- `prepare` 是“插桩”
- `convert` 是“正式替换”

这两步不是一回事。

### 6.3 一张“准备态 / 部署态”对照表

| 阶段 | 模型里主要有什么 | 目的 |
| --- | --- | --- |
| float baseline | 普通 float 模型 | 先学任务本身 |
| prepared PTQ model | observer / 统计逻辑 | 估量化参数 |
| prepared QAT model | fake quant + observer | 模拟量化误差并继续训练 |
| converted quantized model | 真实量化表示 | 接近真实部署 |

初学者最容易混淆的是：

- prepared model
- converted model

如果只记一句话，可以记成：

- prepared 还是“训练/校准态”
- converted 才是“部署态”

---

## 7. PTQ 和 QAT 的实现差异到底在哪

如果只看名字，很多人会觉得二者差很多；其实主干很像，只是中间阶段不同。

| 阶段 | PTQ | QAT |
| --- | --- | --- |
| float baseline | 需要 | 需要 |
| prepare | 需要 | 需要 |
| observer | 需要 | 需要 |
| calibration | 需要 | 通常不走单独 calibration，改为训练期统计 |
| fake quant 训练 | 不需要 | 需要 |
| convert | 需要 | 需要 |

最本质的差别只有一句话：

- PTQ：用数据估范围，不训练参数
- QAT：一边估范围，一边让参数适应量化误差

---

## 8. 从“手写实现”角度看三条线

为了让实现感更强，这里把三种常见工作流放到一张表里看。

| 路线 | 需要训练吗 | 需要 calibration 吗 | 需要 fake quant 吗 | 适合什么问题 |
| --- | --- | --- | --- | --- |
| weight-only | 否 | 否 | 否 | 想先压缩权重 |
| PTQ | 否 | 是 | 否 | 想低成本验证量化可行性 |
| QAT | 是 | 训练期隐含统计 | 是 | PTQ 掉点明显，需要追精度 |

所以如果你问：

- “我到底应该先实现哪条？”

最推荐的学习顺序是：

1. 先手理解 per-tensor / per-channel 的量化
2. 再理解 PTQ 的 observer + calibration
3. 最后再理解 fake quant 和 QAT

### 8.1 为什么不建议一上来就看 QAT 代码

因为 QAT 同时包含：

- 量化原理
- observer
- fake quant
- 训练循环
- prepare / convert

如果前面的基础没立住，你看到的只会是一堆 API 名字。

更稳的顺序是：

1. 先看最小版量化函数
2. 再看 PTQ
3. 最后再看 QAT

---

## 9. 一个最小版从 float 到 PTQ/QAT 的学习路径

如果你想真正学会，不建议一上来就看大而全框架。

更好的顺序是：

### 第一步：手写最小版量化函数

先能自己写出：

- `choose_qparams`
- `quantize`
- `dequantize`

### 第二步：手写最小版静态量化线性层

先能理解：

- 权重怎么量化
- 激活参数怎么固定
- 为什么中间是 `int32` 累加

### 第三步：理解 calibration / observer

先能回答：

- 为什么静态量化需要 calibration
- observer 为什么必须先看数据

### 第四步：理解 fake quant

先能回答：

- 为什么训练时还是 float
- fake quant 为什么能让模型适应部署误差

### 第五步：再去看框架 API

到这时你再看：

- `prepare`
- `convert`
- `quantize_()`
- `PT2E`

会清楚很多，因为你知道这些 API 背后到底在做什么。

---

## 10. 初学者最容易踩的坑

### 10.1 只看概念，不看实现

如果你只记住：

- 对称量化
- 非对称量化
- PTQ
- QAT

但不知道这些东西“代码里大概怎么做”，那你很快就会混乱。

### 10.2 把 PTQ 和 QAT 当成完全不同的世界

它们的主干其实很像。

真正关键的区别是：

- PTQ 不训练，只校准
- QAT 会继续训练，并在前向里模拟量化误差

### 10.3 把 calibration 理解成训练

不对。

calibration 只做：

- 前向
- 统计

不做：

- 反向传播
- 更新权重

### 10.4 看 fake quant 时以为训练已经是 int8

也不对。

训练期通常仍是 float 计算图，只是在前向里注入量化误差的模拟。

### 10.5 把 QAT 结果直接等同于最终部署结果

也不对。

训练中常见的：

- fake-quant 验证指标

和最终：

- converted quantized model 的真实部署结果

不是完全同一件事。

所以一个成熟流程通常会区分：

- 训练期代理指标
- 最终部署口径指标

---

## 11. 下一步该读什么

读完这一份后，建议继续看：

1. [torchao 量化路线总览与阅读指南](./torchao_quantization_guide.md)
2. [torchao PT2E 图像分类实战：对应本仓库实现](./torchao_pt2e_image_classification.md)

如果你发现自己仍然对：

- `scale / zero_point`
- per-channel / per-group
- weight-only / dynamic / static

这些概念模糊，先回去重读上一份《量化基础原理》，不要急着进入框架 API。

## 12. 参考资料

这一份文档最值得继续对照的原始资料有：

- PyTorch / torchao PT2E Quantization：
  https://docs.pytorch.org/ao/stable/pt2e_quantization/index.html
- PyTorch 2 Export Post Training Quantization：
  https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_ptq.html
- PyTorch 2 Export Quantization-Aware Training (QAT)：
  https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_qat.html
- `torchao` Quantization-Aware Training Workflow：
  https://docs.pytorch.org/ao/stable/workflows/qat.html
- Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*：
  https://arxiv.org/abs/1712.05877
- Nagel et al., *A White Paper on Neural Network Quantization*：
  https://arxiv.org/abs/2106.08295
