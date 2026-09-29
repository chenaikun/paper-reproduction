# U-Net 复现总结：30/30 数据集实验

## 1. 项目目标

本实验重新使用下载得到的 30/30 图像分割数据集，验证 U-Net 的核心设计，并通过受控消融实验分析以下问题：

1. 增加标注数据后，无 BatchNorm 的 U-Net 是否还会退化为全前景预测；
2. BatchNorm 是否影响模型的训练稳定性和最终性能；
3. 跳跃连接是否能够改善分割结果；
4. Dice Loss 是否比单独的交叉熵损失更有效。

本报告中的结果与旧的 DIC-C2DH-HeLa 小数据实验分开记录，不能直接混为一组指标。

需要特别说明：本文档中的新数据实验是“U-Net 核心结构和训练流程复现”，不是论文全部训练细节的逐项复现。论文中的边界加权损失、SGD 配置、专用权重初始化和弹性变形增强将在第 3.1 节单独说明。

---

## 2. 数据集和划分

数据目录：

```text
data/unmodified-data/
├── train/
│   ├── imgs/       # 30 张图像
│   └── labels/     # 30 张标签
└── test/
    ├── imgs/       # 30 张图像
    └── labels/     # 30 张标签
```

图像和标签均为 `512×512` PNG，标签是二值图，像素值为 `0` 和 `255`。

为了避免测试集参与模型选择，30 张训练图进一步固定划分为：

| 子集 | 数量 | 用途 |
|---|---:|---|
| train | 24 | 更新模型参数 |
| val | 6 | 选择最佳 checkpoint、观察训练过程 |
| test | 30 | 最终一次性评估 |

划分使用固定随机种子 `seed=42`，因此每次运行得到相同的 24/6 划分。

### 2.1 数据处理

训练阶段使用随机水平翻转、垂直翻转和小角度旋转；验证和测试阶段不使用随机增强。

图像输入保持完整的 `512×512`，标签中心裁剪为 `324×324`。原因是当前 U-Net 使用 `padding=0` 的 valid convolution，输入经过多次卷积后，模型输出尺寸为 `324×324`，标签必须与输出严格对齐。

标签读取后由 `0/255` 转换为 `0/1`，供交叉熵损失使用。

---

## 3. 代码实现

新增数据集读取器：

```text
dataset_unmodified.py
```

它负责读取 `train/imgs`、`train/labels` 和 `test/imgs`、`test/labels`，并完成固定划分、增强、归一化和中心裁剪。

训练入口 [train_ablation.py](train_ablation.py) 增加了两个参数：

```bash
--dataset {hela,unmodified}
--data-root PATH
```

旧 HeLa 实验仍然使用默认配置：

```bash
--dataset hela
--data-root ./data/DIC-C2DH-HeLa
```

新数据实验使用：

```bash
--dataset unmodified
--data-root ./data/unmodified-data
```

新数据只有二值标签，没有每个细胞独立的实例编号，因此没有使用论文中依赖实例距离图的 `WeightedCrossEntropyLoss`。本实验使用普通 CE 和 CE+Dice，避免把二值标签错误地当成实例分割标签。

预测可视化脚本：

```text
predict_unmodified.py
```

它可以输出原图、真实标签和预测结果三联图，并同时显示 IoU、Dice 和前景比例。

### 3.1 论文训练组件的复现状态

| 论文组件 | 论文作用 | 当前项目状态 |
|---|---|---|
| 边界加权损失 `w(x)` | 提高相邻目标之间边界像素的权重，减少粘连 | HeLa 代码中已有 `WeightedCrossEntropyLoss`；新数据只有二值标签，无法可靠计算每个实例的 `d1`、`d2`，因此未用于新数据实验 |
| SGD + momentum `0.99` | 论文训练使用的优化器配置 | `train_ablation.py` 支持 SGD；新数据上已完成 `lr=0.01`、`momentum=0.99` 的对照 |
| 权重初始化 `std=sqrt(2/N)` | U-Net 采用的 He/Kaiming 初始化，`N` 为一个神经元的输入连接数 | `train_ablation.py` 已支持 `--init he`，并已完成新数据上的单因素对照 |
| 弹性变形 | 用平滑随机位移扩充医学图像训练样本 | 新数据读取器现已支持平滑随机位移、图像双线性插值和标签最近邻插值；正式训练对照尚未完成 |

论文初始化可以写成：对卷积层权重从均值为 0、标准差为 `sqrt(2/N)` 的高斯分布采样，其中 `N` 是该层一个输出神经元的输入连接数。这是 U-Net 采用的 He/Kaiming 初始化方法，不是 U-Net 论文独创的方法。它与 BatchNorm 不是同一个机制，不能用 BatchNorm 实验结果替代。

论文加权损失的核心形式为：

```text
w(x) = w_c(x) + w_0 * exp(-(d_1(x) + d_2(x))^2 / (2 * sigma^2))
```

其中 `d1` 和 `d2` 是像素到最近两个实例的距离。只有二值前景标签时，不能直接知道前景区域属于哪一个细胞实例。可以尝试通过连通域生成伪实例编号，但这依赖“目标之间已被边界完全分开”的假设，需要先验证，不能直接当作原始实例标注。

---

## 4. 实验配置

除特别说明外，所有实验都使用：

```text
训练集：24 张
验证集：6 张
测试集：30 张
输入：512×512
输出/标签：324×324
优化器：Adam
学习率：5e-4
batch size：1
训练轮数：60
随机种子：42
```

每个实验按照验证集 IoU 保存 `best_model.pth`，测试集只使用最佳 checkpoint 做一次最终评估。

---

## 5. 实验结果

### 5.1 主要消融结果

| 实验 | BatchNorm | 跳跃连接 | 损失 | 最佳验证 IoU | 测试 Mean IoU | 测试 Mean Dice | 测试前景率 |
|---|:---:|:---:|---|---:|---:|---:|---:|
| Baseline | 有 | 有 | CE | 90.30% | 86.09% ± 4.28% | 92.46% ± 2.59% | 0.8119 |
| Baseline + He init | 有 | 有 | CE | 90.12% | **86.77% ± 3.73%** | **92.87% ± 2.21%** | **0.7888** |
| Baseline + SGD | 有 | 有 | CE | 90.08% | 85.27% ± 4.85% | 91.97% ± 2.96% | 0.8295 |
| Baseline + Elastic | 有 | 有 | CE | 89.77% | 86.40% ± 4.57% | 92.64% ± 2.75% | 0.8068 |
| Original | 无 | 有 | CE | 89.23% | 85.62% ± 4.16% | 92.20% ± 2.49% | 0.8104 |
| Baseline | 有 | 有 | CE+Dice | 90.31% | 86.48% ± 4.08% | 92.70% ± 2.47% | 0.7980 |
| No-skip | 有 | 无 | CE | 88.24% | 85.32% ± 4.88% | 92.00% ± 2.97% | 0.8200 |

在目前已经完成的实验中，测试集表现最好的配置是：

```text
Baseline + BatchNorm + 跳跃连接 + He 初始化 + CE
测试 Mean IoU：86.77%
测试 Mean Dice：92.87%
```

### 5.2 训练轮数观察

Baseline + CE 在 20 epoch 和 60 epoch 的结果如下：

| 训练轮数 | 最佳验证 IoU | 测试 Mean IoU | 测试 Mean Dice |
|---:|---:|---:|---:|
| 20 | 89.21% | 85.76% | 92.28% |
| 60 | 90.30% | 86.09% | 92.46% |

从 20 epoch 增加到 60 epoch 后，测试 IoU 只提高 `0.33` 个百分点，说明模型在当前数据集上已经接近性能平台期，继续训练的收益有限。

---

## 6. 结果分析

### 6.1 BatchNorm 的作用

有 BatchNorm 的 Baseline 测试 IoU 为 `86.09%`，无 BatchNorm 的 Original 为 `85.62%`，差值为 `0.47` 个百分点。

在新的 24 张训练图条件下，无 BatchNorm 模型没有退化为全前景，能够正常收敛。这与旧 HeLa 实验不同：旧实验只有 9 张训练图，无 BatchNorm 模型很快坍缩，验证前景率达到接近 `1.0`。

因此更严谨的结论是：

> 数据量增加后，无 BatchNorm 网络的训练稳定性得到改善；BatchNorm 在新数据上仍带来小幅性能优势，但不再是模型能否训练的决定性因素。

由于两套实验的数据来源不同，不能把这个结果解释为“只增加数据量就一定解决了问题”。数据分布、标签格式和训练划分也发生了变化。

### 6.2 跳跃连接的作用

有跳跃连接的 Baseline 测试 IoU 为 `86.09%`，去掉跳跃连接后为 `85.32%`，下降 `0.77` 个百分点。

这说明跳跃连接确实有帮助。它把编码器中的高分辨率细节传给解码器，帮助模型恢复目标边界和空间位置。去掉后模型仍能完成基本分割，但边界细节和整体稳定性变差。

### 6.3 Dice Loss 的作用

Baseline + CE 的测试 IoU 为 `86.09%`，Baseline + CE+Dice 为 `86.48%`，提高 `0.39` 个百分点；测试 Dice 提高 `0.24` 个百分点，平均前景率从 `0.8119` 降至 `0.7980`。

说明 Dice Loss 对区域重叠和前景面积控制有一定帮助，但提升幅度较小。在当前二分类任务上，普通 CE 已经能够取得较好结果，Dice 不是决定性组件。

### 6.4 He 初始化的作用

在 Baseline、Adam、CE 和数据划分完全不变的条件下，He 初始化与默认初始化的结果为：

| 初始化 | 最佳验证 IoU | 测试 Mean IoU | 测试 Mean Dice | 测试前景率 |
|---|---:|---:|---:|---:|
| PyTorch 默认 | 90.30% | 86.09% ± 4.28% | 92.46% ± 2.59% | 0.8119 |
| He/Kaiming | 90.12% | **86.77% ± 3.73%** | **92.87% ± 2.21%** | **0.7888** |

He 初始化的最佳验证 IoU 略低 `0.18` 个百分点，但测试 IoU 提高 `0.68` 个百分点，测试 Dice 提高 `0.41` 个百分点，预测前景率也降低。这说明初始化可能改善了测试集泛化和过预测情况，但当前只有一个随机种子，不能据此断言 He 初始化必然优于默认初始化。

### 6.5 SGD 优化器的作用

在 Baseline、默认初始化、CE 和数据划分保持一致的条件下，将 Adam 换成论文风格的 SGD（`lr=0.01`、`momentum=0.99`）后：

| 优化器 | 最佳验证 IoU | 测试 Mean IoU | 测试 Mean Dice | 测试前景率 |
|---|---:|---:|---:|---:|
| Adam | 90.30% | 86.09% ± 4.28% | 92.46% ± 2.59% | 0.8119 |
| SGD, momentum=0.99 | 90.08% | 85.27% ± 4.85% | 91.97% ± 2.96% | 0.8295 |

SGD 的最佳验证 IoU 与 Adam 接近，仅低 `0.22` 个百分点，但测试 IoU 低 `0.82` 个百分点，测试 Dice 低 `0.49` 个百分点，前景率更高。这说明在当前单次固定划分和学习率配置下，SGD 的训练集/验证集表现尚可，但泛化略差，并且前景过预测更明显。由于 Adam 与 SGD 的学习率尺度不同，这一结果不能单独证明 Adam 在所有配置下都优于 SGD。

### 6.6 弹性变形增强的作用

在 Baseline、Adam、默认初始化和 CE 保持一致的条件下加入弹性变形后：

| 增强方式 | 最佳验证 IoU | 测试 Mean IoU | 测试 Mean Dice | 测试前景率 |
|---|---:|---:|---:|---:|
| 翻转/旋转 | 90.30% | 86.09% ± 4.28% | 92.46% ± 2.59% | 0.8119 |
| 翻转/旋转 + 弹性变形 | 89.77% | 86.40% ± 4.57% | 92.64% ± 2.75% | 0.8068 |

弹性变形使测试 IoU 提高 `0.31` 个百分点、测试 Dice 提高 `0.18` 个百分点，平均前景率下降 `0.0051`。验证集最佳 IoU 反而低于普通 Baseline，说明该增强在当前单次划分上的收益很小，暂时不能称为显著提升。它可能稍微改善了测试泛化和前景过预测，但需要多个随机种子才能确认。

### 6.7 预测结果中的过预测

部分测试图中，预测前景率高于真实前景率。例如某些样本的真实前景率约为 `0.60`，预测前景率约为 `0.83`。这说明当前模型的主要误差是前景过预测，而不是全前景坍缩。

CE+Dice 后平均前景率下降，表明它对该问题有轻微缓解，但没有完全消除误检。

---

## 7. 与旧 HeLa 实验的对照

旧实验使用 DIC-C2DH-HeLa 数据：Seq01 只有 9 张训练帧，Seq02 只有 9 张验证帧。无 BatchNorm 的 Original 模型在该条件下退化为全前景预测。

新数据实验使用 24 张训练图和 30 张独立测试图：无 BatchNorm 模型能够正常训练，测试 IoU 为 `85.62%`。

对照可以说明：

```text
HeLa：9 张训练图，无 BN → 全前景退化
新数据：24 张训练图，无 BN → 正常收敛
```

但这只是现象上的对照，不是严格的单变量因果实验，因为两个数据集的图像内容、标注方式和数据分布都不同。

---

## 8. 复现实验命令

### 8.1 Baseline + CE

```bash
python train_ablation.py \
  --name unmodified_baseline_adam_ce_60epoch \
  --dataset unmodified \
  --data-root ./data/unmodified-data \
  --model baseline \
  --optimizer adam \
  --loss ce \
  --lr 5e-4 \
  --epochs 60 \
  --batch-size 1 \
  --seed 42
```

### 8.2 Baseline + CE+Dice

```bash
python train_ablation.py \
  --name unmodified_baseline_adam_ce_dice_60epoch \
  --dataset unmodified \
  --data-root ./data/unmodified-data \
  --model baseline \
  --optimizer adam \
  --loss ce_dice \
  --lr 5e-4 \
  --epochs 60 \
  --batch-size 1 \
  --seed 42
```

### 8.3 Original，无 BatchNorm

```bash
python train_ablation.py \
  --name unmodified_original_adam_ce_60epoch \
  --dataset unmodified \
  --data-root ./data/unmodified-data \
  --model original \
  --optimizer adam \
  --loss ce \
  --lr 5e-4 \
  --epochs 60 \
  --batch-size 1 \
  --seed 42
```

### 8.4 No-skip

```bash
python train_ablation.py \
  --name unmodified_no_skip_adam_ce_60epoch \
  --dataset unmodified \
  --data-root ./data/unmodified-data \
  --model no_skip \
  --optimizer adam \
  --loss ce \
  --lr 5e-4 \
  --epochs 60 \
  --batch-size 1 \
  --seed 42
```

### 8.5 当前项目中已有的 HeLa 论文风格配置

下面的命令使用无 BatchNorm、SGD、高动量和边界加权损失，适用于带有实例标注的 HeLa 数据，不适用于当前的二值 `unmodified-data` 标签：

```bash
python train_ablation.py \
  --name step4_original_sgd_weighted_ce \
  --dataset hela \
  --data-root ./data/DIC-C2DH-HeLa \
  --model original \
  --optimizer sgd \
  --loss weighted_ce \
  --lr 0.01 \
  --momentum 0.99 \
  --grad-clip 1.0 \
  --w0 10.0 \
  --sigma 5.0 \
  --epochs 60 \
  --batch-size 1 \
  --seed 42
```

这条命令可以说明项目支持论文加权损失和 SGD 配置，但它的结果属于旧 HeLa 实验，不能和新数据的 30 张测试图结果放在同一张表中。

### 8.7 新数据上的弹性增强对照

新数据读取器现在支持 `--elastic-deform`。先运行 5 个 epoch 检查训练是否稳定：

```bash
python train_ablation.py \
  --name unmodified_baseline_elastic_smoke_5epoch \
  --dataset unmodified \
  --data-root ./data/unmodified-data \
  --model baseline \
  --optimizer adam \
  --loss ce \
  --lr 5e-4 \
  --init default \
  --elastic-deform \
  --epochs 5 \
  --batch-size 1 \
  --seed 42
```

确认没有尺寸错误、标签仍为二值且前景率没有长期退化后，再运行 60 epoch 正式实验：

```bash
python train_ablation.py \
  --name unmodified_baseline_elastic_ce_60epoch \
  --dataset unmodified \
  --data-root ./data/unmodified-data \
  --model baseline \
  --optimizer adam \
  --loss ce \
  --lr 5e-4 \
  --init default \
  --elastic-deform \
  --epochs 60 \
  --batch-size 1 \
  --seed 42
```

这组实验只增加弹性变形，其他配置与 `unmodified_baseline_adam_ce_60epoch` 保持一致。

### 8.6 新数据上的初始化对照

训练脚本现在支持：

```bash
--init default
--init he
```

`default` 使用 PyTorch 层的默认初始化，`he` 使用 He/Kaiming normal 初始化。新数据上的正式初始化对照命令为：

```bash
python train_ablation.py \
  --name unmodified_baseline_he_init_ce_60epoch \
  --dataset unmodified \
  --data-root ./data/unmodified-data \
  --model baseline \
  --optimizer adam \
  --loss ce \
  --lr 5e-4 \
  --init he \
  --epochs 60 \
  --batch-size 1 \
  --seed 42
```

该实验只改变初始化方式，其他配置与 `unmodified_baseline_adam_ce_60epoch` 保持一致。结果已写入第 5 节和第 6.4 节。

---

## 9. 汇报时的讲述框架

### 9.1 论文解决了什么问题

医学图像分割需要同时完成语义理解和精确边界定位。下采样可以扩大感受野，但会损失空间细节；而医学图像标注成本高，训练数据有限。U-Net 使用编码器提取上下文信息，使用解码器恢复分辨率，并通过跳跃连接把高分辨率细节传回解码器，从而解决“看得懂但定位不准”的问题。

### 9.2 使用了什么方法，代码如何写

代码由以下部分组成：

1. `DoubleConv`：两个 `3×3` 卷积和 ReLU；
2. `EncoderBlock`：卷积、保存 skip 特征、最大池化；
3. `DecoderBlock`：转置卷积、裁剪对齐、拼接 skip、卷积；
4. `UNet`：组合编码器、瓶颈层和解码器；
5. 数据集类：读取图像和标签、同步增强、归一化和中心裁剪；
6. 训练循环：前向传播、损失计算、反向传播和参数更新；
7. 评估脚本：计算 IoU、Dice、前景率并保存最佳模型。

### 9.3 如何想到，如何思考

可以按照下面的科研过程来讲：

```text
先实现 Baseline
→ 确认数据、模型和训练流程正常
→ 复现无 BN 原始结构
→ 发现全前景预测不能只看 IoU
→ 增加前景率和可视化诊断
→ 控制变量做 BatchNorm、Dice、skip 消融
→ 在更大数据集上重新验证结论
```

最重要的思考是：

> 指标好看不等于模型真的学会了任务；必须同时查看预测图、类别比例和独立测试集结果。

---

## 10. 局限性和后续工作

当前结果仍有以下限制：

1. 只使用了一次固定的 24/6 划分和一个随机种子；
2. 测试集数量为 30 张，规模仍然有限；
3. 没有报告多次重复实验的置信区间；
4. 新数据标签是二值标签，不能严格复现论文中依赖实例距离的边界加权损失；
5. 当前输入输出尺寸是 `512→324`，与论文原文的 `572→388` 不完全相同；
6. 新数据实验与 HeLa 实验的数据来源不同，因此不能做严格的因果归因。
7. 当前新数据主实验没有使用论文的 SGD、高动量、He 初始化和弹性变形组合，因此不能称为论文训练流程的完全复现。

如果继续深入，可以考虑：

- 用 3 个以上随机种子重复 Baseline、Original 和 No-skip；
- 增加概率阈值调节，研究是否可以减少前景过预测；
- 使用真正的实例标签后再测试论文加权损失；
- 实现并单独验证 `std=sqrt(2/N)` 的论文权重初始化；
- 在相同数据划分下比较 Adam 与 SGD(momentum=0.99)；
- 使用连续插值实现更接近论文的弹性变形，并比较增强前后的泛化性能；
- 对测试图逐张保存预测，分析难例和失败模式；
- 在理解 U-Net 后继续学习 ResNet，为目标检测、知识蒸馏和多模态任务打基础。

---

## 11. 最终结论

在下载得到的 30/30 数据集上，U-Net Baseline 能够稳定完成二分类分割。在当前已经完成的对照中，Baseline + He/Kaiming 初始化 + CE 取得最佳测试表现：`86.77%` Mean IoU 和 `92.87%` Mean Dice；Baseline + CE+Dice 也取得了接近的 `86.48%` Mean IoU 和 `92.70%` Mean Dice。SGD 在论文风格配置下能够正常收敛，但本次测试集结果低于 Adam。

消融实验表明：

1. 跳跃连接对恢复空间细节有实际帮助；
2. BatchNorm 在新数据集上带来小幅性能和稳定性优势，但不是模型训练成功的必要条件；
3. Dice Loss 带来小幅区域重叠提升，并略微降低前景过预测；
4. 增加数据后，无 BatchNorm 模型不再像 9 帧 HeLa 实验那样坍缩为全前景；
5. 不能只用单个 IoU 判断模型是否有效，必须结合预测可视化、前景比例、验证集和独立测试集共同判断。

这次复现的主要收获不只是得到一个指标，而是建立了从论文设计、代码实现、训练诊断到控制变量实验的完整流程。
