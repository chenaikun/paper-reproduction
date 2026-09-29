# ResNet 复现计划：只提供结构、思路和验收标准

本目录不提供可直接运行的 ResNet 实现。目标是你自己从空文件开始写出模型、训练循环和消融实验；这里仅保留结构图、论文理解主线和实验计划。

## 1. 最适合你的模型

你的 RTX 3050 4GB 最适合从 **CIFAR-10 + CIFAR 风格 ResNet-20** 开始：

- 输入为 32×32，训练速度快，单次实验成本低。
- ResNet 的核心机制全部保留：残差分支、shortcut、下采样、BatchNorm、ReLU、全局平均池化。
- ResNet-20 只有约 27 万参数，batch size 128 通常不会有显存压力。
- 学会 ResNet-20 后，ResNet-32/56 只是在同一结构上增加 block 数量。

建议规模：

| 阶段 | 模型 | 建议 batch size | 目的 |
| --- | --- | ---: | --- |
| 第一次手写 | ResNet-20 | 128 | 搞清楚每一层和每个 shape |
| 深度实验 | ResNet-32 | 128 | 观察深度变化 |
| 主要消融 | ResNet-56 | 64 或 128 | 对比 plain/residual、shortcut |
| 可选挑战 | ResNet-110 | 32 或 64 | 只在前面都稳定后尝试 |

不要一开始做 ImageNet ResNet-50。它会把数据准备、训练时间、显存和调参问题混在一起，不利于理解 ResNet 本身。

## 2. 你要自己写出的 ResNet-20 结构

这是 CIFAR 风格结构，不是 ImageNet 的 7×7 卷积加 max-pooling 结构。

```text
输入图像:                 3 × 32 × 32
        │
3×3 Conv, 16 channels, stride=1, padding=1
        │
Stage 1: 3 个 BasicBlock, 16 channels, 32 × 32
        │
Stage 2: 3 个 BasicBlock, 32 channels, 16 × 16
        │  第 1 个 block stride=2，其余 stride=1
        │
Stage 3: 3 个 BasicBlock, 64 channels, 8 × 8
        │  第 1 个 block stride=2，其余 stride=1
        │
Global Average Pooling: 64 × 8 × 8 -> 64
        │
全连接层: 64 -> 10
        │
输出: 10 类 logits
```

每个 BasicBlock：

```text
                 ┌─ 3×3 Conv ─ BN ─ ReLU ─ 3×3 Conv ─ BN ─┐
x ───────────────┤                                           ├─ Add ─ ReLU ─ 输出
                 └──── shortcut(identity 或 projection) ────┘
```

当输入和输出的空间尺寸、通道数都相同时，shortcut 是 identity。当发生 32×32→16×16 或 16→32 通道变化时，shortcut 必须同步改变尺寸。

论文 CIFAR 实验需要重点理解两种处理方法：

- **Option A**：shortcut 做下采样，并对通道做零填充。参数更少，适合先复现论文的 CIFAR 设定。
- **Option B**：使用 1×1 convolution + stride。参数更多，但工程中更常见，也更容易迁移到 ImageNet ResNet。

建议先实现 Option A，再实现 Option B，不能把两者混为一谈。

### 深度为什么是 20

CIFAR BasicBlock 有两个卷积。设每个 stage 有 `n` 个 block：

```text
总深度 = 1 个 stem 卷积 + 3 个 stage × n 个 block × 2 个卷积 + 1 个全连接层
       = 6n + 2
```

因此：

- n=3：ResNet-20
- n=5：ResNet-32
- n=9：ResNet-56
- n=18：ResNet-110

## 3. 论文理解主线

### 问题：网络变深后训练误差反而变大

论文讨论的不是普通的过拟合，而是 **degradation problem**：更深的网络在训练集上的误差都可能更高。

作者的关键假设是：如果浅层网络已经足够好，更深的网络至少应该能复制浅层网络的结果。最理想的情况是新增层学习恒等映射 `H(x)=x`。但让一组普通卷积层精确学成恒等映射并不容易。

### 思路：把学习目标改成残差

普通网络学习：

```text
H(x)
```

ResNet 学习：

```text
H(x) = F(x) + x
```

如果最优结果接近恒等映射，只需让残差分支 `F(x)` 接近 0。这比让多层卷积直接构造恒等变换更容易优化；同时 shortcut 还提供了更直接的梯度路径。

你在复现时必须能回答：

1. 为什么 residual learning 不是简单地“多加一条边”？
2. 为什么 shortcut 默认不使用复杂的非线性变换？
3. 为什么尺寸变化时不能直接做 `out + x`？
4. 为什么“训练误差变大”不能直接解释成过拟合？

## 4. 手写顺序

每一步都先自己写，再运行最小检查；不要先看 torchvision 的实现。

### 阶段 A：只写模型，不训练

1. 写单个 BasicBlock。
2. 用随机输入打印每个中间张量的 shape。
3. 写三个 stage 和 global average pooling。
4. 写参数量统计。
5. 用随机标签做一次 forward/backward。

最低验收标准：

```text
(2, 3, 32, 32) -> (2, 10)
32×32 -> 16×16 -> 8×8 的尺寸变化正确
loss.backward() 后每个可训练参数都有梯度
```

### 阶段 B：先让模型过拟合极小数据

只取 16 或 32 张训练图片，关闭数据增强，反复训练。目标不是泛化，而是确认：

- loss 能接近 0；
- 训练准确率能接近 100%；
- 标签、loss、optimizer、梯度清零都没有写错。

如果小数据都过拟合不了，不要急着调学习率或换模型，先检查 forward 和训练循环。

### 阶段 C：完整 CIFAR-10 基线

建议设置：

- 随机裁剪：padding=4 后裁回 32×32；
- 随机水平翻转；
- CIFAR-10 mean/std 标准化；
- SGD，momentum=0.9，weight decay=1e-4；
- 初始学习率 0.1；
- batch size 128；
- 训练 164 epochs；
- 在大约第 82、123 个 epoch 将学习率乘 0.1；
- 固定随机种子，并保存训练/测试 loss 和 accuracy。

先用 5 epochs 做 smoke test，再跑正式实验。正式实验每次只改一个变量。

## 5. 消融实验计划

### 实验 1：ResNet-20 vs Plain-20

保持卷积层、参数规模、初始化、数据增强和优化器完全一致，只去掉 shortcut 加法。

重点看训练 loss 曲线：残差网络通常更容易优化。不要只比较最终测试准确率。

### 实验 2：Plain-20 vs Plain-56

观察更深的普通网络是否出现 degradation。这个实验用于复现论文提出 ResNet 的动机。

### 实验 3：ResNet-20 的 shortcut

比较：

- Option A：下采样 + 零填充；
- Option B：1×1 projection；
- 错误版本：尺寸变化时仍强行 identity，确认为什么会 shape error。

### 实验 4：去掉 BatchNorm

保持其他配置相同，记录 loss 是否震荡、收敛速度是否变慢。不要同时改变初始化和学习率，否则结论不清楚。

### 实验 5：深度扩展

比较 ResNet-20、ResNet-32、ResNet-56。确认你只改变每个 stage 的 block 数量，而不是无意中改变了 stem 或通道配置。

## 6. 调试清单

遇到不收敛时按以下顺序排查：

1. 单张图片是否能完成 forward。
2. 小数据集是否能过拟合。
3. logits shape 是否为 `(batch_size, 10)`。
4. label 是否是 `LongTensor` 且范围为 0 到 9。
5. `optimizer.zero_grad()` 是否在每个 batch 开头调用。
6. 是否误把测试集放进了训练增强。
7. 下采样时 shortcut 的 shape 是否和 residual branch 一致。
8. 学习率、BN 的 train/eval 状态、数据标准化是否正确。

## 7. 最终完成标准

完成下面这些，你就不只是“会调用 ResNet”，而是掌握了它：

- 不看资料写出 BasicBlock 和 CIFAR ResNet-20。
- 能手算每个 stage 的 shape 和深度公式 `6n+2`。
- 能解释 degradation 为什么促成了 residual learning。
- 能用实验说明 plain network 和 residual network 的训练差异。
- 能实现 Option A 和 Option B shortcut，并解释参数量差异。
- 能把 CIFAR stem 改成 ImageNet stem，并说明为什么输入尺寸变化后结构也要变化。
- 最后再自己实现 Bottleneck，并推导 ResNet-50/101/152 的深度计算方式。

建议学习顺序是：**ResNet-20 → Plain/Residual 对照 → shortcut 消融 → ResNet-56 → ImageNet stem → Bottleneck**。这样每一步都只增加一个新概念，最适合用你的显卡和“自己写代码”的目标完成复现。
