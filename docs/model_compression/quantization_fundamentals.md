# 量化基础原理：从公式到手写实现

这份文档只回答一个问题：

- **量化到底是什么，如果不依赖现成框架，我大概应该怎么实现它？**

它面向初学者，所以会优先讲清楚：

- 为什么要量化
- `scale / zero_point` 到底在做什么
- 对称/非对称量化、量化粒度分别是什么意思
- weight-only / dynamic / static 量化到底差在哪
- 如果你自己写一个最小版实现，代码会长什么样

如果你读完这一份还不知道：

- 某个量化方案到底量化了谁
- 量化参数什么时候算
- 为什么有 per-tensor / per-channel / per-group

那就说明这份文档没写清楚。

---

## 1. 为什么要做量化

量化的核心目标有三个：

1. 降低模型存储大小
2. 降低推理时的内存带宽压力
3. 在合适的硬件和 kernel 上加速推理

最常见的直觉是：

- `float32` 每个元素 4 字节
- `int8` 每个元素 1 字节
- `int4` 每个元素约 0.5 字节

所以模型参数从 `fp32` 变成 `int8` 后，光看参数存储就可能缩小到原来的四分之一。

但量化并不等于“一定更快”。

速度是否真的提升，还取决于：

- 目标硬件是否支持该低精度算子
- backend 是否真的把图 lowering 到量化 kernel
- 模型结构是否适合该量化方案

所以更准确地说：

- 量化先带来“更小、更省内存”
- 然后才有机会在正确的后端上带来“更快”

---

## 2. 量化的核心公式

最常见的是 affine quantization。

先记住两步：

```text
q = clamp(round(x / scale + zero_point), qmin, qmax)
x_hat = (q - zero_point) * scale
```

其中：

- `x`：原始高精度值，通常是 `float32 / float16 / bfloat16`
- `q`：量化后的整数值
- `x_hat`：反量化后的近似值
- `scale`：缩放因子
- `zero_point`：零点偏移
- `qmin / qmax`：目标整数 dtype 的可表示范围

你可以把它理解成：

- `scale` 决定“一个整数台阶对应多少实数”
- `zero_point` 决定“实数 0 在整数坐标系里放在哪里”

### 2.1 从区间到量化参数

假设我们知道某个 Tensor 的值大致在：

- `[x_min, x_max]`

想把它映射到整数区间：

- `[qmin, qmax]`

最常见的一组近似计算是：

```text
scale = (x_max - x_min) / (qmax - qmin)
zero_point = round(qmin - x_min / scale)
zero_point = clamp(zero_point, qmin, qmax)
```

如果你看到有些地方公式写法不同，不要紧张，本质是同一件事：

- 先确定 float 区间
- 再确定整数区间
- 再把两边对齐

### 2.2 一个最小数值例子

假设某层权重范围大致在：

- `[-1.0, 1.0]`

目标量化到 `int8`，并采用对称量化：

- `qmin = -127`
- `qmax = 127`
- `zero_point = 0`
- `scale = 1.0 / 127`

若某个 float 权重：

- `x = 0.30`

那么：

```text
q = round(0.30 / (1/127)) = round(38.1) = 38
x_hat = 38 * (1/127) = 0.2992
```

可以看到：

- 量化后存的是整数 `38`
- 反量化后不是严格 `0.30`
- 而是一个接近值 `0.2992`

这就是量化误差的来源。

### 2.3 量化误差从哪里来

量化误差主要有两类：

- 舍入误差：`round(...)`
- 截断误差：`clamp(...)`

所以量化不是无损压缩，而是：

- 用更少 bit 表示原来的值
- 接受一定的数值近似
- 换取更低的存储、带宽和部署成本

---

## 3. `scale` 和 `zero_point` 到底怎么理解

### 3.1 `scale`

`scale` 越小：

- 一个整数台阶代表的实数间隔越细
- 表示更精细
- 但覆盖范围更窄

`scale` 越大：

- 覆盖范围更宽
- 但离散化更粗

### 3.2 `zero_point`

`zero_point` 的作用是把“实数 0”对齐到某个整数位置。

为什么要有它？

因为很多实际数据分布并不是围绕 0 完全对称的。

例如某层激活如果大致在：

- `[-1, 5]`

那么它就比：

- `[-5, 5]`

更偏正。

这时如果你强行要求对称量化，可能会浪费一部分整数范围。

所以：

- 对称量化常常令 `zero_point = 0`
- 非对称量化允许 `zero_point != 0`

### 3.3 对称量化和非对称量化

#### 对称量化

常见特征：

- `zero_point = 0` 或接近 0
- 实现更简单
- 权重量化里非常常见

为什么权重常用它？

因为权重往往更接近以 0 为中心的分布。

#### 非对称量化

常见特征：

- `zero_point` 不一定为 0
- 更适合偏移分布
- 激活量化里很常见

为什么激活常用它？

因为激活值经常受：

- `ReLU`
- 偏移输入分布
- 数据增强

等因素影响，不一定围绕 0 对称。

### 3.4 一个真正非零 `zero_point` 的例子

假设激活范围大致在：

- `[-1.0, 5.0]`

量化到 `uint8`：

- `qmin = 0`
- `qmax = 255`
- `scale = 6.0 / 255`
- `zero_point = round((0 - (-1.0)) / scale) = 43`

若激活值：

- `x = 1.0`

则：

```text
q = round(1.0 / (6/255) + 43) = 86
x_hat = (86 - 43) * (6/255) = 1.0118
```

这里最关键的不是 `1.0118` 这个具体数字，而是：

- `zero_point` 真正变成了非零
- 说明量化区间在“向右平移”
- 这样可以更充分利用 `uint8` 的可表示范围

---

## 4. 量化粒度到底是什么

“量化粒度”回答的是另一个问题：

- **一组 `scale / zero_point` 到底服务多大一块数据？**

如果这一点不清楚，看再多名词也会很抽象。

### 4.1 per-tensor

一整个 Tensor 共用一组 `scale / zero_point`。

最简单的伪代码是：

```python
scale, zp = choose_qparams(x)
q = quantize(x, scale, zp)
```

特点：

- 最简单
- 元信息最少
- 但不同通道分布差异大时，误差会比较明显

### 4.2 per-channel

每个通道各用一组量化参数。

先看线性层权重：

- 对 `Linear.weight`，常见形状是 `[out_features, in_features]`
- “每个通道”通常按输出通道，也就是每一行

伪代码可以写成：

```python
for oc in range(W.shape[0]):
    scale[oc], zp[oc] = choose_qparams(W[oc, :])
    W_q[oc, :] = quantize(W[oc, :], scale[oc], zp[oc])
```

再看卷积层权重：

- 对 `Conv2d.weight`，常见形状是 `[out_channels, in_channels, kH, kW]`
- 通常按 `out_channels` 做 per-channel

也就是：

- 每个输出通道一组量化参数

为什么它常见？

因为不同输出通道的权重分布往往差异很大。

如果整层共用一个 `scale`，误差会被最极端的通道主导。

#### 再用一个具体形状看一次

如果：

- `Linear.weight.shape = [3, 4]`

那么它可以写成：

```text
[
  [w00, w01, w02, w03],   <- 输出通道 0
  [w10, w11, w12, w13],   <- 输出通道 1
  [w20, w21, w22, w23],   <- 输出通道 2
]
```

如果按 per-channel 量化，就是：

- 第 0 行一组 `scale_0 / zp_0`
- 第 1 行一组 `scale_1 / zp_1`
- 第 2 行一组 `scale_2 / zp_2`

也就是说，这里的“channel”不是抽象名词，而是：

- **线性层里的一行权重**

再看卷积：

- `Conv2d.weight.shape = [2, 3, 3, 3]`

表示：

- 有 2 个输出通道
- 每个输出通道对应一个 `3 x 3 x 3` 的卷积核张量

如果按 per-channel 量化，就是：

- 第 0 个输出通道对应的整块卷积核，用一组量化参数
- 第 1 个输出通道对应的整块卷积核，用另一组量化参数

对初学者来说，先记住一句最实用的话：

- 在线性层里，per-channel 常常按“每一行权重”
- 在卷积层里，per-channel 常常按“每个输出通道的卷积核”

### 4.3 per-group

per-group 可以理解成：

- 不是整层共用一组
- 也不是每个通道都各自一组
- 而是把数据切成若干组，每组一组参数

以线性层权重为例，假设：

- `W.shape = [4096, 4096]`
- 每行按 `group_size = 128` 切块

那一行会被切成很多 group：

```text
[0:128], [128:256], [256:384], ...
```

伪代码：

```python
for oc in range(W.shape[0]):
    for start in range(0, W.shape[1], group_size):
        end = start + group_size
        block = W[oc, start:end]
        scale, zp = choose_qparams(block)
        W_q[oc, start:end] = quantize(block, scale, zp)
```

它的直觉是：

- per-tensor 太粗
- per-channel 有时还不够细
- per-group 是更细但仍可控的折中

这在 `int4` / `int8` weight-only 场景里非常常见。

#### 先看一个真正会落到代码里的小例子

假设：

- `W.shape = [2, 8]`
- `group_size = 4`

那么它可以理解成：

```text
第 0 行:
  group 0 -> W[0, 0:4]
  group 1 -> W[0, 4:8]

第 1 行:
  group 0 -> W[1, 0:4]
  group 1 -> W[1, 4:8]
```

这时一共会有 4 组量化参数：

- `scale[0, 0], zp[0, 0]`
- `scale[0, 1], zp[0, 1]`
- `scale[1, 0], zp[1, 0]`
- `scale[1, 1], zp[1, 1]`

也就是说，per-group 不是一个抽象概念，而是：

- **先选一条切分轴**
- **再按固定 `group_size` 把它切成多个连续小块**
- **每个小块各算一组量化参数**

#### group_size 到底在控制什么

`group_size` 控制的是：

- 一组量化参数到底管多少个连续元素

如果 `group_size` 很大：

- 更接近 per-channel
- 元信息更少
- 但表达更粗

如果 `group_size` 很小：

- 更细
- 更接近“很多小块各自量化”
- 但量化参数也会更多

所以 `group_size` 本质上是在做一个折中：

- 更细的表达能力
- 更低的元信息和实现成本

可以把它想成：

- per-channel 是“一整行一个刻度尺”
- per-group 是“同一行里分段，每段一个刻度尺”

如果继续往两个极端推，就更容易理解：

- 当 `group_size = in_features` 时，它就退化得非常接近 per-channel
- 当 `group_size = 1` 时，几乎变成“每个元素自己一组量化参数”，表达最细，但通常不值得

### 4.4 per-token / per-row

这类粒度常用于动态激活量化。

例如 Transformer 中一批输入到线性层前，激活可以看成：

- `[batch, seq, hidden]`

如果按 token 做量化，本质上就是：

- 每个 token 的那一行激活，各自现算一组量化参数

伪代码：

```python
for token_idx in range(x.shape[0]):
    scale[token_idx], zp[token_idx] = choose_qparams(x[token_idx, :])
    x_q[token_idx, :] = quantize(x[token_idx, :], scale[token_idx], zp[token_idx])
```

为什么它常见于大模型？

因为：

- 不同 token 的激活分布变化很大
- 运行时动态按 token 算量化参数，通常比整层共用一组更稳

#### 一个具体形状例子

假设某个激活张量是：

- `[batch, seq, hidden] = [2, 3, 4]`

你可以把它看成 6 个 token，每个 token 是一个长度为 4 的向量：

```text
batch 0:
  token 0 -> [a000, a001, a002, a003]
  token 1 -> [a010, a011, a012, a013]
  token 2 -> [a020, a021, a022, a023]

batch 1:
  token 0 -> [a100, a101, a102, a103]
  token 1 -> [a110, a111, a112, a113]
  token 2 -> [a120, a121, a122, a123]
```

如果做 per-token 量化，本质上就是：

- 上面这 6 行，每一行都单独算一组量化参数

这样做的好处是：

- 每个 token 的量化区间都更贴近自己当前分布

坏处是：

- 运行时要算更多组量化参数

### 4.5 量化粒度到底怎么选

一个够用的经验顺序是：

| 粒度 | 精度倾向 | 实现复杂度 | 常见场景 |
| --- | --- | --- | --- |
| per-tensor | 最粗 | 最低 | 最基础教学、简单基线 |
| per-channel | 更稳 | 中等 | CNN 权重量化 |
| per-group | 更细 | 更高 | LLM / int4 weight-only |
| per-token | 激活更稳 | 更高 | Transformer 动态激活量化 |

如果你是初学者，可以先记住：

- CNN 权重：优先理解 per-channel
- 大模型权重：优先理解 per-group
- 大模型激活：优先理解 per-token

#### 如果你还是拿不准，就按这个顺序想

1. 这一层的不同通道分布差异大不大？
2. 你更在意实现简单，还是更在意精度？
3. 这是权重还是激活？
4. 激活是在离线 calibration，还是运行时动态算参数？

一个最够用的判断模板是：

- 先从 per-tensor 起步，确保流程能跑通
- CNN 权重量化时优先尝试 per-channel
- LLM 权重量化时优先理解 per-group
- 动态激活量化时优先理解 per-token / per-row

---

## 5. 到底量化谁：权重、激活、偏置

量化不是只关心“权重有没有变成 int8”。

实际推理里，至少要区分三样东西：

- 权重 `W`
- 激活 `x`
- 偏置 `bias`

### 5.1 权重

权重最容易离线量化，因为：

- 推理前就已经固定
- 可以提前算好量化参数
- 可以提前存成低 bit

### 5.2 激活

激活更麻烦，因为：

- 每次输入不同
- 分布会变
- 有些方案在运行时还要重新算量化参数

所以激活量化通常决定：

- 这是 static 还是 dynamic 路线

### 5.3 偏置

偏置经常仍然保留在更高精度，或者按更高精度规则参与计算。

为什么？

因为矩阵乘法通常会先发生：

- `int8 x int8 -> int32 accumulate`

如果偏置也过度压缩，误差会更明显。

初学者先记住一个非常重要的实现细节：

- 量化推理里，中间累加通常不是 `int8`
- 而是更高精度的 `int32`

---

## 6. 三种最常见方案：weight-only、dynamic、static

这一节是初学者最容易混的地方，所以这里直接讲“怎么实现”。

### 6.1 weight-only quantization

weight-only 的定义很直接：

- **只量化权重**
- 激活仍保持高精度

最小思路：

1. 离线把 `W_float` 量化成 `W_q`
2. 推理时输入 `x` 仍然是高精度
3. 算子内部用特定 kernel 读取量化权重完成计算

简化伪代码：

```python
W_scale, W_zp = choose_qparams(W_float)
W_q = quantize(W_float, W_scale, W_zp)

def forward(x_float):
    return linear_with_quantized_weight(x_float, W_q, W_scale, W_zp, bias)
```

为什么它常见？

因为：

- 落地成本低
- 权重能明显变小
- 不需要专门为激活准备 calibration 数据

### 6.2 dynamic quantization

dynamic quantization 的定义是：

- 权重提前量化
- 激活在运行时动态算量化参数

最小思路：

1. 先量化权重
2. 每次前向时，先根据当前输入 `x` 计算 `x_scale / x_zp`
3. 再把 `x` 量化
4. 然后执行整数计算

伪代码：

```python
W_scale, W_zp = choose_qparams(W_float)
W_q = quantize(W_float, W_scale, W_zp)

def forward(x_float):
    x_scale, x_zp = choose_qparams(x_float)
    x_q = quantize(x_float, x_scale, x_zp)
    y_int32 = int_matmul(x_q, x_zp, W_q, W_zp)
    y_float = dequantize_output(y_int32, x_scale, W_scale, bias)
    return y_float
```

为什么叫 dynamic？

因为激活量化参数不是预先固定的，而是：

- 每次输入来了再动态计算

这里最容易误解的一点是：

- dynamic quantization 不是“每次都重新量化权重”
- 它动态的是激活的量化参数
- 权重量化通常还是离线先做好

### 6.3 static quantization

static quantization 的定义是：

- 权重离线量化
- 激活量化参数也尽量提前定好

那激活参数怎么提前定？

靠：

- calibration
- observer

也就是先拿一批代表性数据跑一遍模型，估计每层激活分布，再把这些量化参数固定下来。

最小思路：

1. 先量化权重
2. 用 calibration 数据统计激活范围
3. 固定每层激活的 `scale / zero_point`
4. 推理时直接按固定参数量化激活

伪代码：

```python
# offline
W_scale, W_zp = choose_qparams(W_float)
W_q = quantize(W_float, W_scale, W_zp)

act_scale, act_zp = run_calibration_and_collect_qparams(calib_loader)

def forward(x_float):
    x_q = quantize(x_float, act_scale, act_zp)
    y_int32 = int_matmul(x_q, act_zp, W_q, W_zp)
    y_float = dequantize_output(y_int32, act_scale, W_scale, bias)
    return y_float
```

为什么 CNN CPU int8 部署里常见它？

因为：

- 它最贴近完整的端到端整数部署
- backend 更容易识别 pattern 并融合
- 对卷积网络很常见

这里也有一个常见误解：

- static quantization 不是把“输入值”预先存死
- 真正预先固定的是激活量化参数
- 推理时来的仍然是新输入，只是它们会按事先确定好的 `scale / zero_point` 被量化

### 6.4 一张表看懂这三者

| 方案 | 权重量化 | 激活量化 | 激活参数何时得到 | 常见场景 |
| --- | --- | --- | --- | --- |
| weight-only | 是 | 否或很弱 | 不需要 | LLM / Linear 推理 |
| dynamic | 是 | 是 | 运行时动态计算 | `Linear` 主导模型 |
| static | 是 | 是 | calibration 后固定 | CNN / CPU int8 部署 |

### 6.5 把同一个 `Linear` 放到三条路线里看

为了避免这三条路线看起来像三个互不相关的名词，我们只看同一个线性层：

```text
y = xW^T + b
```

看看它在三条路线里分别怎么实现。

#### 方案 A：weight-only

```text
离线:
  W_float -> W_q

运行时:
  x_float --------\
                   -> kernel( x_float , W_q ) -> y_float
  W_q ----------/
```

它的核心特点是：

- 只提前量化 `W`
- `x` 仍然保持高精度
- 最适合先压缩权重

#### 方案 B：dynamic quantization

```text
离线:
  W_float -> W_q

运行时:
  x_float -> choose_qparams(x) -> x_q
  x_q , W_q -> int matmul -> int32 accumulate -> dequantize -> y_float
```

它的核心特点是：

- `W` 提前量化
- `x` 的量化参数每次前向现算

#### 方案 C：static quantization

```text
离线:
  W_float -> W_q
  calib data -> observer -> act_scale / act_zp

运行时:
  x_float -> quantize_with_fixed_qparams -> x_q
  x_q , W_q -> int matmul -> int32 accumulate -> dequantize -> y_float
```

它的核心特点是：

- `W` 提前量化
- `x` 也量化
- 但 `x` 的量化参数不是运行时现算，而是 calibration 后固定

### 6.6 再用“预先算什么 / 运行时算什么”总结一次

如果你还是分不清，就只看这一张表：

| 方案 | 预先算好的东西 | 运行时现算的东西 |
| --- | --- | --- |
| weight-only | 权重量化参数、量化权重 | 基本只处理高精度输入 |
| dynamic | 权重量化参数、量化权重 | 输入量化参数、输入量化、整数乘加 |
| static | 权重量化参数、量化权重、激活量化参数 | 输入按固定参数量化、整数乘加 |

这张表非常关键，因为它直接告诉你“怎么实现”：

- 你只要分清哪些东西是 offline 的
- 哪些东西是 online 的

三条路线的本质区别就已经抓住了。

---

## 7. 一个最小版“手写量化”会长什么样

如果你想真正把原理和实现连起来，下面这个最小版伪代码是最值得理解的。

### 7.1 量化参数估计

```python
def choose_qparams_minmax(x, qmin, qmax, eps=1e-8):
    x_min = x.min().item()
    x_max = x.max().item()
    scale = max((x_max - x_min) / (qmax - qmin), eps)
    zero_point = round(qmin - x_min / scale)
    zero_point = max(qmin, min(qmax, zero_point))
    return scale, zero_point
```

### 7.2 量化与反量化

```python
def quantize(x, scale, zero_point, qmin, qmax):
    q = torch.round(x / scale + zero_point)
    q = torch.clamp(q, qmin, qmax)
    return q

def dequantize(q, scale, zero_point):
    return (q - zero_point) * scale
```

### 7.3 一个最小版静态量化线性层

```python
class NaiveStaticQuantLinear:
    def __init__(self, W_float, bias, act_scale, act_zp):
        self.W_scale, self.W_zp = choose_qparams_minmax(W_float, -127, 127)
        self.W_q = quantize(W_float, self.W_scale, self.W_zp, -127, 127)
        self.bias = bias
        self.act_scale = act_scale
        self.act_zp = act_zp

    def __call__(self, x_float):
        x_q = quantize(x_float, self.act_scale, self.act_zp, -128, 127)
        # 真实 kernel 里不会这么天真，这里只是演示数值流程
        x_centered = x_q - self.act_zp
        W_centered = self.W_q - self.W_zp
        y_int32 = x_centered @ W_centered.T
        y_float = y_int32 * (self.act_scale * self.W_scale) + self.bias
        return y_float
```

这段伪代码并不是工业实现，但它已经足够回答初学者最关心的事：

- 量化参数在哪里来
- 权重什么时候量化
- 激活什么时候量化
- 为什么中间是 `int32` 累加
- 为什么输出还要反量化回 float

### 7.4 用一个两输入一输出的 `Linear` 真正算一遍

只看一个最小线性层：

```text
y = x_0 w_0 + x_1 w_1 + b
```

假设：

- `x = [0.20, -0.40]`
- `w = [0.30, -0.10]`
- `b = 0.05`

为了方便手算，我们先人为选一组很整齐的量化参数：

- `x_scale = 0.1, x_zp = 0`
- `w_scale = 0.05, w_zp = 0`

先量化输入：

```text
x_q = [ round(0.20 / 0.1), round(-0.40 / 0.1) ] = [2, -4]
```

再量化权重：

```text
w_q = [ round(0.30 / 0.05), round(-0.10 / 0.05) ] = [6, -2]
```

然后在整数域里做乘加：

```text
y_int32 = 2 * 6 + (-4) * (-2) = 20
```

把它映射回 float：

```text
y_no_bias = y_int32 * (x_scale * w_scale)
          = 20 * (0.1 * 0.05)
          = 0.10
```

最后加上 bias：

```text
y_hat = 0.10 + 0.05 = 0.15
```

再看原始 float 计算：

```text
y = 0.20 * 0.30 + (-0.40) * (-0.10) + 0.05
  = 0.06 + 0.04 + 0.05
  = 0.15
```

这个例子里两边刚好相等，不是因为量化永远无误差，而是因为：

- 我们特意选了能整齐落在量化格点上的数

如果把输入换成：

- `x = [0.23, -0.41]`
- `w = [0.28, -0.09]`

按同样参数量化后，近似值会变成：

- `x_hat = [0.20, -0.40]`
- `w_hat = [0.30, -0.10]`

这时量化输出会接近原始输出，但不再完全相等。这个例子最想让你看到的是：

- **量化线性层的核心流程就是：float -> 整数表示 -> int32 累加 -> 按 scale 恢复输出。**

### 7.5 为什么整数矩阵乘法是成立的

这一步如果不讲清楚，初学者通常会一直有个疑问：

- “既然最后还要反量化回 float，那中间做整数乘法到底有什么意义？”

先从最简单的线性层点积开始。

假设输入和权重分别满足：

```text
x ≈ s_x * (q_x - z_x)
w ≈ s_w * (q_w - z_w)
```

那么一个点积：

```text
y = Σ x_i w_i
```

就可以近似写成：

```text
y ≈ s_x s_w Σ (q_xi - z_x)(q_wi - z_w)
```

把它展开：

```text
y ≈ s_x s_w [
    Σ q_xi q_wi
    - z_w Σ q_xi
    - z_x Σ q_wi
    + K z_x z_w
]
```

其中：

- `K` 是这次点积里参与相乘的元素个数

这条式子最重要的意义不是让你死记，而是让你看清：

- 中间真正做的还是整数乘加
- `zero_point` 的影响可以通过补偿项处理
- 最后再统一乘回 `s_x s_w`

也就是说，量化线性层的核心并不是：

- “把 float 硬转成 int8 再照抄原公式”

而是：

- 先把 float 映射到整数域
- 在整数域里完成主要乘加
- 最后再按 scale 把结果映射回来

### 7.6 为什么中间通常是 `int32`

假设：

- `int8` 的一个元素大致在 `[-128, 127]`

那么两个 `int8` 相乘后，结果范围大致在：

- `[-16384, 16129]`

如果再做很多项累加，一个 `int8` 或 `int16` 很快就不够放了。

所以常见实现会用：

- `int8 x int8`
- `int32 accumulate`

这也是为什么你会经常看到：

- 量化输入和权重是低 bit
- 但累加器不是低 bit

对初学者来说，先记住一个非常实用的判断：

- **低 bit 负责省存储和带宽**
- **更高 bit 累加负责保数值稳定性**

### 7.7 bias 在实现里通常怎么处理

偏置是另一个特别容易被忽略的点。

从公式上看，如果：

```text
y ≈ s_x s_w * y_int32 + bias
```

那 bias 至少要和输出处在可对齐的数值空间里。

常见思路有两种：

1. bias 保留为 float，在最终反量化后再加
2. bias 先按 `s_x * s_w` 量化到更高精度整数，再和 `int32` 累加结果对齐

所以你不应该把 bias 理解成：

- “也像权重一样随便压成 int8 就完了”

更合理的理解是：

- bias 要服务于整数乘加结果的数值对齐

### 7.8 再补一个 dynamic quantized Linear 的最小版

前面那个例子偏 static，这里再给一个 dynamic 的最小版。

```python
class NaiveDynamicQuantLinear:
    def __init__(self, W_float, bias):
        self.W_scale, self.W_zp = choose_qparams_minmax(W_float, -127, 127)
        self.W_q = quantize(W_float, self.W_scale, self.W_zp, -127, 127)
        self.bias = bias

    def __call__(self, x_float):
        x_scale, x_zp = choose_qparams_minmax(x_float, -128, 127)
        x_q = quantize(x_float, x_scale, x_zp, -128, 127)

        x_centered = x_q - x_zp
        W_centered = self.W_q - self.W_zp
        y_int32 = x_centered @ W_centered.T
        y_float = y_int32 * (x_scale * self.W_scale) + self.bias
        return y_float
```

这个例子最想说明的就是：

- 权重量化参数在初始化时就定下来了
- 激活量化参数是在每次前向时现算

这正是 dynamic quantization 的本质。

### 7.9 `Conv2d` 为什么也能按类似思路量化

很多初学者对 `Linear` 例子看懂了，但一到卷积就又断掉。

其实从数值上看，卷积可以理解成：

- 在很多空间位置上重复做局部点积

对一个卷积核权重：

- `W.shape = [out_channels, in_channels, kH, kW]`

对于每个输出通道，本质上都有一组局部滤波器。

在某个空间位置上，这个滤波器会和输入的一个局部 patch 做点积。

所以卷积量化的直觉可以先理解成：

- 卷积不是另一种完全不同的数学
- 它只是“把点积重复很多次”

#### 为什么很多教程会说卷积可以看成矩阵乘法

因为如果你把输入按滑窗展开，每个局部 patch 都可以被摊平成一个向量。

同样，卷积核也可以被摊平成一个向量。

这样一次卷积就可以改写成：

- patch 向量和 kernel 向量的点积

很多框架内部不一定真的用最天真的 `im2col` 去实现，但这个视角非常适合教学，因为它能把：

- `Conv2d`

和：

- `Linear`

统一到同一种“点积/矩阵乘法”语言下。

### 7.10 一个最小版卷积量化心智模型

如果只看一个输出通道 `oc`，可以把它想成：

```python
for oh in range(out_h):
    for ow in range(out_w):
        patch = x[:, oh:oh+kH, ow:ow+kW]
        y[oc, oh, ow] = dot(patch, W[oc])
```

如果再把 patch 和 `W[oc]` 都量化，数值流程和前面的线性层没有本质区别：

1. 输入 patch 量化
2. 卷积核权重量化
3. 做整数乘加
4. 再按 scale 恢复输出尺度

这也是为什么：

- `Linear`
- `Conv2d`

都可以落到类似的量化框架里。

#### 再看一个超小卷积例子

假设：

- 输入只有 1 个通道
- 输入大小是 `4 x 4`
- 卷积核大小是 `2 x 2`
- stride = 1

输入：

```text
x =
[
  [x00, x01, x02, x03],
  [x10, x11, x12, x13],
  [x20, x21, x22, x23],
  [x30, x31, x32, x33],
]
```

卷积核：

```text
w =
[
  [w00, w01],
  [w10, w11],
]
```

第一个输出位置左上角对应的 patch 是：

```text
[
  [x00, x01],
  [x10, x11],
]
```

把它摊平后：

```text
patch_00 = [x00, x01, x10, x11]
kernel   = [w00, w01, w10, w11]
```

这个位置的输出，本质上就是：

```text
y00 = dot(patch_00, kernel)
```

第二个位置又会取另一个 patch：

```text
patch_01 = [x01, x02, x11, x12]
```

然后继续和同一个 kernel 做点积。

所以对这个最小例子来说，卷积可以理解成：

- 取 patch
- 摊平
- 和 kernel 做点积
- 不断重复

一旦你接受这个视角，就很容易理解为什么卷积量化也能沿用：

- 输入量化
- 权重量化
- 整数乘加
- 再按 scale 恢复尺度

这一整套流程。

#### 如果把所有 patch 一次性排成矩阵

上面我们是一次只看一个 patch。

如果把所有滑窗 patch 都摊平并排起来，可以写成：

```text
X_col =
[
  patch_00
  patch_01
  patch_02
  ...
]
```

卷积核也摊平为：

```text
W_row =
[
  kernel_0
  kernel_1
  ...
]
```

那么卷积就可以近似看成：

```text
Y = W_row @ X_col^T
```

这里的维度细节在不同实现里会有转置差别，但教学上最重要的是这句：

- **卷积可以被看成“卷积核矩阵”和“patch 矩阵”的乘法。**

这句话一旦理解了，很多量化实现就会自然很多，因为你可以把它们都归到：

- quantized matmul

这一类数值流程里。

### 7.11 为什么 CNN 权重常用 per-channel

这点对卷积尤其重要。

因为 `Conv2d` 的每个输出通道通常会学到：

- 完全不同的滤波功能
- 完全不同的数值范围

如果整层只共用一组 `scale`：

- 最大范围的那个通道会支配整个量化区间
- 其他通道就会被迫用更粗的步长表示

所以对卷积层来说，per-channel 常常不是“高级优化”，而是：

- 一个非常自然、非常常见的默认选择

#### 再把它和输出通道对应起来看

如果卷积层有：

- `out_channels = 64`

那通常意味着：

- 有 64 个输出特征图
- 也有 64 套卷积核

per-channel 权重量化通常就是：

- 第 0 套卷积核一组量化参数
- 第 1 套卷积核一组量化参数
- ...
- 第 63 套卷积核一组量化参数

这样每个输出通道都能用更贴近自己数值分布的刻度尺。

### 7.12 如果真的手写一个最小版量化 `Conv2d`，需要做哪几步

如果你已经接受了“卷积就是很多次局部点积”，那一个最小版 static quantized `Conv2d` 的实现思路其实已经能写出来了。

先假设：

- 激活使用一组固定的 `x_scale / x_zp`
- 权重按输出通道做 per-channel

那么伪代码大致会是：

```python
class NaiveStaticQuantConv2d:
    def __init__(self, W_float, bias, x_scale, x_zp):
        self.W_scale = []
        self.W_zp = []
        self.W_q = torch.empty_like(W_float)

        for oc in range(W_float.shape[0]):
            scale, zp = choose_qparams_minmax(W_float[oc], -127, 127)
            self.W_scale.append(scale)
            self.W_zp.append(zp)
            self.W_q[oc] = quantize(W_float[oc], scale, zp, -127, 127)

        self.bias = bias
        self.x_scale = x_scale
        self.x_zp = x_zp

    def __call__(self, x_float):
        x_q = quantize(x_float, self.x_scale, self.x_zp, -128, 127)
        y = []

        for oc in range(self.W_q.shape[0]):
            out_map = []
            for oh in range(out_h):
                row = []
                for ow in range(out_w):
                    patch = x_q[:, oh:oh+kH, ow:ow+kW] - self.x_zp
                    kernel = self.W_q[oc] - self.W_zp[oc]
                    acc = torch.sum(patch.reshape(-1) * kernel.reshape(-1))
                    val = acc * (self.x_scale * self.W_scale[oc]) + self.bias[oc]
                    row.append(val)
                out_map.append(row)
            y.append(out_map)
        return y
```

这段代码当然很慢，也省略了：

- stride
- padding
- dilation
- batch 维
- kernel fusion

但它已经把初学者最该看懂的实现骨架写出来了：

1. 权重先离线量化
2. 如果是 per-channel，就对每个输出通道各算一组权重量化参数
3. 输入在前向时按固定激活参数量化
4. 每个输出位置取一个 patch
5. patch 和对应卷积核做整数乘加
6. 再乘回 `x_scale * w_scale[oc]`
7. 最后加该输出通道的 bias

如果你能把这 7 步讲清楚，就已经不是“知道卷积能量化”，而是：

- **知道卷积量化大概该怎么写**

上面这个伪代码是教学版，所以最后直接回到了 float 输出。

真实部署里，backend 往往还会继续处理：

- 输出 requantization
- 算子融合
- 更高效的内存布局

但这些工程细节不会改变这里最核心的数值骨架。

### 7.13 一条适合初学者的手写练习路径

如果你想真正学会，不建议只读公式。

更好的顺序是：

1. 手写 `choose_qparams_minmax`
2. 手写 `quantize / dequantize`
3. 手写一个最小版 static `Linear`
4. 手写一个最小版 dynamic `Linear`
5. 再把一个卷积 patch 看成点积，理解 `Conv2d` 的量化直觉
6. 最后手写一个单样本、无 padding 的 naive static `Conv2d` 骨架

做到这里，你再去看框架代码时，会明显更容易理解：

- 为什么要有 quantizer
- 为什么要有 observer
- 为什么要区分 static / dynamic / weight-only
- 为什么卷积常用 per-channel

### 7.14 一个够用的自测清单

如果你想检查自己是不是真的理解了这份文档，先试着不看答案回答下面几个问题：

1. `scale` 和 `zero_point` 分别在解决什么问题？
2. 为什么 per-channel 通常比 per-tensor 更稳？
3. dynamic quantization 和 static quantization 的根本差别是什么？
4. 为什么中间累加通常要用 `int32`？
5. 为什么卷积也能沿用和线性层类似的量化思路？

如果这五个问题你都能自己讲清楚，说明你已经不只是“背了术语”，而是真的开始理解量化实现了。

---

## 8. 初学者先记住哪几件事

如果你看完一大堆术语后只能记住几件事，我建议先记这几条：

1. 量化不是简单的 `cast(int8)`，而是“确定量化参数 + 量化 + 反量化/整数计算”的完整流程。
2. `scale` 决定分辨率，`zero_point` 决定零点对齐。
3. 量化粒度决定“一组量化参数覆盖多大一块数据”。
4. 权重量化通常比激活量化更容易。
5. static、dynamic、weight-only 的真正差别，在于“激活是否量化、以及激活参数何时得到”。
6. 如果实现上看不清楚，就先从最小版 `Linear` 量化流程理解，而不是一上来就看复杂框架代码。

---

## 9. 下一步该读什么

读完这份文档后，建议继续看：

1. [量化工作流：PTQ、QAT、Observer 与 Calibration](./quantization_workflows.md)
2. [torchao 量化路线总览与阅读指南](./torchao_quantization_guide.md)
3. [torchao PT2E 图像分类实战：对应本仓库实现](./torchao_pt2e_image_classification.md)

## 10. 术语速查

如果你中途看混了，先回来看这张表：

| 术语 | 一句话解释 |
| --- | --- |
| quantization | 把高精度数值映射到低 bit 表示，并接受一定近似误差 |
| dequantization | 把量化后的整数值映射回近似 float |
| `scale` | 一个整数步长对应多少实数幅度 |
| `zero_point` | 实数 0 在整数坐标系里的对齐位置 |
| symmetric quantization | 通常令 `zero_point = 0` 的量化方式 |
| asymmetric quantization | 允许 `zero_point != 0` 的量化方式 |
| per-tensor | 整个 Tensor 共用一组量化参数 |
| per-channel | 每个通道各自一组量化参数 |
| per-group | 每个 group 共用一组量化参数 |
| per-token | 每个 token 或每一行激活各自一组量化参数 |
| weight-only | 只量化权重，不重点量化激活 |
| dynamic quantization | 权重量化，激活量化参数运行时现算 |
| static quantization | 权重和激活都量化，激活参数通常通过 calibration 预先得到 |
| observer | 用来统计分布并计算量化参数的模块/逻辑 |
| calibration | 用代表性输入跑前向，给 observer 收集统计 |
| fake quant | 训练时模拟“量化 -> 反量化”误差的前向过程 |
| PTQ | 训练完成后再量化 |
| QAT | 训练时就让模型适应量化误差 |

## 11. 参考资料

如果你想继续对照原始资料，建议优先看这些：

- PyTorch / torchao Quantization Overview：
  https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html
- `torchao` Quantized Inference：
  https://docs.pytorch.org/ao/stable/workflows/inference.html
- Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*：
  https://arxiv.org/abs/1712.05877
- Nagel et al., *A White Paper on Neural Network Quantization*：
  https://arxiv.org/abs/2106.08295
