# 从零读懂 `ld_core_experiment`

> 读者设定：会 Python，学过神经网络训练的基本流程，**没接触过 MMDetection**，第一次读别人的实验仓库。
>
> 读法建议：第 0~3 章按顺序读，这是地基。第 4 章以后可以跳着看。
> 遇到不懂的名词，先翻 **0.6 名词小抄**。

---

## 第 0 章 这个实验到底在干什么（先完全别看代码）

### 0.1 目标检测是什么

给一张图，模型要回答两件事：

1. **图里有什么**（类别：人、狗、车……）
2. **在哪里**（一个矩形框，用左上角 + 右下角四个数表示：`x1, y1, x2, y2`）

一张图里可能有多个物体，所以输出是"一堆框，每个框带一个类别"。

### 0.2 蒸馏是什么：让小学生学大学生

想象两个学生：

- **老师（teacher）**：ResNet101，模型大、跑得慢、通常比学生准。本次机制实验的 teacher 只训练了 1 个 epoch，主要用于验证蒸馏流程和变化方向。
- **学生（student）**：ResNet18，模型小、跑得快，但要重新训练。

**蒸馏（distillation）** 就是：学生除了看标准答案（数据集里人工标注的框，叫 **GT**，ground truth），还要看老师的"解题过程"。

为什么看解题过程有用？因为标准答案只有一个：

> 这个框的左边界 = 372 像素。

而老师的输出是这样的：

> 左边界在 370 的概率 0.05，在 371 的概率 0.15，在 **372 的概率 0.60**，在 373 的概率 0.15，在 374 的概率 0.05……

后者信息多得多：它告诉学生"正确答案在 372，但 371 和 373 也很有可能，别的地方基本不可能"。这种"可能性分布"就是老师脑子里的**知识**，比一个干巴巴的数字更丰富。

用术语说：老师输出的是**软标签（soft label）**，数据集里的是**硬标签（hard label）**。

### 0.3 这篇论文（LD）的核心创新

普通蒸馏只教"这是什么类别"。LD 说：**定位（框在哪）也能教，而且更重要**。

要教定位，得先让模型"能表达不确定性"。这里用的检测器叫 **GFL**，它不直接回归 4 个坐标，而是把每条边表示成**一个 17 维的概率分布**：

```
"左边界的距离" = [0.01, 0.02, ..., 0.60, ..., 0.01]   ← 17 个数，加起来等于 1
                    0    1         7          16     ← 表示距离是 0~16 个 stride
```

于是"教定位"就变成了：**让学生输出的分布去逼近老师输出的分布**。

衡量"两个分布差多少"的数学工具叫 **KL 散度**。这就是 `loss_ld`。

论文还有第二个创新：**VLR（Valuable Localization Region，有价值的定位区域）**。

意思是：不要让所有位置都去学老师。图里大部分位置是背景，学它们纯属浪费；只挑那些"**已经很像、但还差一点**"的位置来学，收益最大。

### 0.4 这个仓库要验证什么

论文的核心论点是：**定位知识比语义知识更重要**。

为了验证，就做四组对照实验，只改"蒸馏什么"，其他全都一样：

| 组 | 教什么 | 类比（考试前复习） |
|---|---|---|
| baseline | 什么都不教，只背标准答案 | 只做真题 |
| kd-only | 只教"这是什么类别" | 只背知识点 |
| loc-only | 只教"框在哪" | 只练解题步骤 |
| full | 两个都教（论文完整方法） | 既背知识点又练步骤 |

如果 **loc-only 比 kd-only 提分多**，就支持论文的观点。

**为什么用 VOC 不用 COCO？** 论文原实验在 COCO 上、8 张显卡跑好几天。这个仓库要在一张显卡上做，所以换成 VOC（20 类、1.6 万张图），分辨率也调小。结果不能和论文的 AP 数字直接比，只能看**趋势**（谁比谁高）。

### 0.5 为什么需要"壳"这一层

你可能会问：直接改官方代码不就行了？

三个原因：

1. **官方仓库是 git submodule**（别人仓库嵌在你的仓库里）。你改了它，别人 clone 下来就复现不了你的实验。
2. **改配置就够**。数据集、分辨率、学习率都是**配置**，不需要动模型代码。
3. **官方代码在新环境跑不起来**（依赖版本冲突）。这个用"打补丁"解决，补丁文件放在你自己的仓库里。

所以整个 `ld_core_experiment` 的角色是：**一个配置生成器 + 一个启动器**。它自己不含任何模型逻辑，所有真正的计算都在 `LD/` 里。

### 0.6 名词小抄

第一次读检测代码，90% 的困惑来自名词。先混个眼熟，不用背：

| 名词 | 大白话 |
|---|---|
| **GT** | ground truth，数据集里人工标注的标准答案 |
| **anchor** | 预先在图上铺好的"参考框"。模型不凭空预测，而是以这些参考框为基准做微调 |
| **logit** | 模型输出的原始分数（还没经过 softmax） |
| **softmax** | 把一堆分数变成"加起来等于 1 的概率" |
| **正样本 / 负样本** | 跟真实物体重合度高的 anchor 叫正样本，基本是背景的叫负样本 |
| **IoU** | 两个框的重叠程度，0=完全不重叠，1=完全重合 |
| **DIoU** | IoU 的改进版，额外考虑两个框中心的距离 |
| **KL 散度** | 衡量两个概率分布差多少。两个分布一样时等于 0 |
| **温度 T** | 蒸馏时的"软化系数"。T 越大，分布越平，小概率也被放大 |
| **epoch** | 把训练集完整过一遍，叫一个 epoch |
| **iteration / iter** | 训练一步（处理一个 batch），叫一个 iteration |
| **batch** | 一次喂给模型的图片数。`samples_per_gpu` 就是它 |
| **checkpoint** | 保存的模型权重文件，后缀 `.pth` |
| **mAP** | 检测任务的标准评分，越高越好。VOC 上一般 70~80 算好 |
| **DFL** | Distribution Focal Loss，GFL 里训练"分布"的损失函数 |
| **reg_max** | 分布有多少个格子。这里是 16，所以每边 17 个格子（0~16） |
| **VLR** | 有价值的定位区域，见 0.3 |
| **config** | 配置文件，本质是一个 Python 字典，见第 1 章 |

---

## 第 1 章 MMDetection 的 config 是什么（零基础版）

### 1.1 MMDetection 是一个"模型零件工厂"

它把检测模型的零件都做好了，注册在一个**花名册**里：

```
backbone 花名册: ResNet, ResNeXt, Swin, ...
neck 花名册:     FPN, PAFPN, ...
head 花名册:     GFLHead, LDHead, ATSSHead, ...
loss 花名册:     QualityFocalLoss, DistributionFocalLoss, ...
检测器花名册:    GFL, FasterRCNN, KnowledgeDistillationSingleStageDetector, ...
```

每个零件都有一个**字符串名字**。

### 1.2 config 就是一个 Python 文件，里面全是 dict

打开 `LD/configs/gfl/gfl_r18_fpn1x_voc.py`，它长这样（精简版）：

```python
model = dict(
    type='GFL',                    # ← 从花名册里挑"GFL"这个检测器
    backbone=dict(type='ResNet', depth=18),
    neck=dict(type='FPN', out_channels=256),
    bbox_head=dict(type='GFLHead', num_classes=20, reg_max=16),
)

optimizer = dict(type='SGD', lr=0.00375)
total_epochs = 4
```

**它没有 class、没有函数调用，只有变量赋值和 dict。** 这是故意的：把"结构描述"和"代码实现"分开，改结构不用改代码。

### 1.3 `type` 这个字符串是怎么变成代码的

程序运行时，`build_detector(cfg.model)` 会做这件事（伪代码）：

```python
def build(cfg):
    类 = 花名册[cfg['type']]          # 'GFL' → GFL 这个 Python 类
    其他参数 = {k: v for k, v in cfg.items() if k != 'type'}
    return 类(**其他参数)              # 实例化
```

如果参数里还有 dict，就**递归**继续 build（所以 `backbone=dict(type='ResNet')` 会再走一遍这句话）。

**记住这一点**：config 里的每个 `dict(type=...)`，运行到最后都会变成一个 `nn.Module`。config 是"图纸"，build 是"照图纸装配"。

### 1.4 必须认识的六个键

| 键 | 管什么 | 类比 |
|---|---|---|
| `model` | 网络结构 | 机器长什么样 |
| `data` | 数据怎么读、怎么预处理、batch 多大 | 喂什么原料 |
| `optimizer` | 用什么优化器、学习率多少 | 怎么调机器 |
| `lr_config` | 学习率什么时候降 | 调机器的节奏 |
| `total_epochs` | 训几轮 | 训多久 |
| `train_cfg` / `test_cfg` | 训练时怎么给 anchor 分配正负样本 / 测试时怎么后处理 | 细节规则 |

`train_cfg` 里的 `ATSSAssigner` 就是个"分配器"：它决定图里那几千个 anchor，哪些算正样本（要学）、哪些算负样本（当背景）。

### 1.5 `_base_`：配置的"继承"

如果不用继承，写一个 VOC 版的 GFL 要抄 100 多行。MMDetection 的做法是：

```python
# configs/gfl_r18_voc_baseline.py  全文就这么几行
_base_ = '../../LD/configs/gfl/gfl_r18_fpn1x_voc.py'
```

意思是：**先把官方那份读完，我这份只写要改的地方**。

合并规则有两条，**必须记住**：

- **dict 是"合并"**：你在子文件里写 `optimizer = dict(lr=0.001)`，合并后变成 `dict(type='SGD', momentum=0.9, weight_decay=0.0001, lr=0.001)`——原来那三个键还在。
- **list 是"整个替换"**：你在子文件里写 `train_pipeline = [...]`，整个列表被换掉，不会一条一条合并。

这个规则解释了很多"为什么我改了没生效"的问题。

### 1.6 跟着读一遍官方 r18 config

打开 `LD/configs/gfl/gfl_r18_fpn1x_voc.py`，一段一段看（用大白话翻译）：

**（1）模型结构**

```python
model = dict(
    type='GFL',
    pretrained='torchvision://resnet18',   # 用 ImageNet 上训好的 ResNet18 当起点，不是从随机数开始
    backbone=dict(type='ResNet', depth=18, frozen_stages=1, norm_eval=True),
    neck=dict(type='FPN', in_channels=[64,128,256,512], out_channels=256,
              start_level=1, add_extra_convs='on_output', num_outs=5),
    bbox_head=dict(type='GFLHead', num_classes=20, ...),
)
```

- `frozen_stages=1`：骨干网最前面那一层不训练（冻结），省算力。
- `norm_eval=True`：BatchNorm 一直用推理模式，不跟着 batch 更新。小 batch 时这很重要，否则统计量会抖。
- `neck` 是 FPN：把不同层的特征图融合成 5 个尺度，用来检测大小不同的物体。
- `num_outs=5`：输出 5 个尺度的特征图，对应下面的 5 个 stride。

**（2）anchor 和 head**

```python
anchor_generator=dict(... strides=[8, 16, 32, 64, 128]),
bbox_head=dict(... reg_max=16, ...)
```

- `strides=[8,16,32,64,128]`：5 层特征图分别相当于原图的 1/8、1/16……采样密度。小物体靠 stride 小的层检测。
- `reg_max=16`：每条边的分布有 17 个格子（0 到 16）。

**（3）三个损失**

```python
loss_cls=dict(type='QualityFocalLoss', ...),      # 分类对不对
loss_dfl=dict(type='DistributionFocalLoss', ...), # 分布学得对不对
loss_bbox=dict(type='CIoULoss', ...),             # 框的位置准不准
```

**（4）训练规则**

```python
train_cfg = dict(assigner=dict(type='ATSSAssigner', topk=9), ...)
```

ATSS 是分配正负样本的方法。`topk=9` 后面讲 VLR 时会再出现。

**（5）怎么训**

```python
optimizer = dict(type='SGD', lr=0.00375, momentum=0.9, weight_decay=0.0001)
lr_config = dict(policy='step', warmup='linear', step=[3])
total_epochs = 4
```

`step=[3]` 意思是"第 3 个 epoch 结束后，把学习率乘 0.1"。所以 4 个 epoch 里，前 3 个用原学习率，最后 1 个用 1/10。（教师那份 R101 配置是 `step=[12]`、`total_epochs=16`，因为是 16 轮的长训练。）

**（6）数据**

```python
data = dict(
    samples_per_gpu=3,      # 每张卡一次喂 3 张图
    workers_per_gpu=2,      # 2 个进程读数据
    train=dict(type='RepeatDataset', times=3, dataset=dict(...)),
)
```

`RepeatDataset(times=3)`：把训练集**重复 3 遍**再喂。VOC 只有 1.6 万张图，16 个 epoch 其实相当于把原数据看了 48 遍。

### 1.7 一份 config 合并后长什么样（手算一遍）

假设官方配置是：

```python
optimizer = dict(type='SGD', lr=0.00375, momentum=0.9, weight_decay=0.0001)
data = dict(samples_per_gpu=3, workers_per_gpu=2, train=dict(type='RepeatDataset', times=3, ...))
```

而这个仓库的壳写成：

```python
optimizer = dict(lr=0.01 * 1 * 1 / 16)     # = 0.000625
data = dict(samples_per_gpu=1, workers_per_gpu=0, train=dict(type='RepeatDataset', times=1, ...))
```

合并后：

```python
optimizer = dict(type='SGD', lr=0.000625, momentum=0.9, weight_decay=0.0001)   # ← 4 个键都在
data = dict(samples_per_gpu=1, workers_per_gpu=0, train=dict(type='RepeatDataset', times=1, ...))
```

**只有你写了的键被改，没写的继承下来。** 这就是为什么 `total_epochs` 会被命令行 `--cfg-options total_epochs=1` 改掉，但 `lr_config.step` 不会跟着变。

---

## 第 2 章 这套代码的三层结构

```text
┌─────────────────────────────────────────────────────┐
│ 第三层  scripts/*.py                                │
│ 干什么：读命令行参数 → 设置环境变量 → 启动官方程序     │
└────────────────────┬────────────────────────────────┘
                     │ 靠 10 个环境变量传话
┌────────────────────▼────────────────────────────────┐
│ 第二层  configs/*.py                                │
│ 干什么：继承官方配置，把数据集/分辨率/学习率改掉       │
└────────────────────┬────────────────────────────────┘
                     │ mmcv 的 Config.fromfile() 读进来
┌────────────────────▼────────────────────────────────┐
│ 第一层  LD/  （官方代码，一行没改，除了补丁）          │
│ 干什么：真正的建模、训练、推理                        │
└─────────────────────────────────────────────────────┘
```

三个文件对应三个职责，记住就行：

| 层 | 代表文件 | 一句话 |
|---|---|---|
| 第三层 | `scripts/train_ld.py` | 决定训哪个变体、训几轮 |
| 第二层 | `configs/ld_r18_voc.py` | 决定网络、数据、蒸馏权重长什么样 |
| 第一层 | `LD/mmdet/models/dense_heads/ld_head.py` | 真正算 loss 的地方 |

---

## 第 3 章 一条命令的完整旅程（慢动作）

### 3.1 你敲下的命令

```bash
python scripts/train_ld.py --variant loc-only \
  --teacher-checkpoint runs/teacher_r101/epoch_16.pth \
  --epochs 4 --seed 0 --deterministic
```

这一行会触发下面一大串事情。我们一步一步跟着走。

### 3.2 第一步：`train_ld.py` 自己做的事

打开 `scripts/train_ld.py`，它会完成下面这些事情：

**① 读参数**（`argparse`）。`--variant loc-only` 表示训练"只教定位"那组。

**② 检查教师权重在不在**：

```python
teacher_checkpoint = args.teacher_checkpoint.expanduser().resolve()
if not teacher_checkpoint.is_file():
    raise SystemExit('Teacher checkpoint not found: {}'.format(teacher_checkpoint))
```

没有教师权重，蒸馏无从谈起，所以这里直接报错退出。**这一步的位置很重要**：它在加载 config 之前，所以报错信息很干净，不会夹在一堆 warning 里。

**③ 决定三样东西**：

```python
config   = configs/ld_r18_voc_loc_only.py      # 用哪份配置
work_dir = runs/ld_loc-only                     # 结果存哪
gpu_ids  = '0'                                  # 用哪张卡
```

**④ 设置环境变量**（下一步细讲）。

**⑤ 启动官方训练程序**（再下一步细讲）。

一句话总结：`train_ld.py` 就是个**前台接待**，检查材料齐不齐、把材料翻译成标准格式，然后把人领到真正的车间（`LD/tools/train.py`）。

### 3.3 第二步：环境变量是怎么变成配置的（关键桥梁）

**为什么要绕这一圈？**

因为官方训练程序的接口是固定的：

```bash
python LD/tools/train.py <config文件路径> --work-dir ... --gpu-ids 0
```

它只认"一个 config 文件路径"。你没法通过命令行告诉它"数据集在这、图片缩小到 640×384、batch 设成 1"。

官方倒是留了个侧门 `--cfg-options data.samples_per_gpu=1`，但要改十几个嵌套的键，命令行会长到没法看。

所以这套壳用了一个**更简单粗暴的办法**：把参数塞进**环境变量**，config 文件在被读取的时候，自己去读环境变量。

流程是这样的：

```text
shell 变量
   ↓  scripts/_common.py 的 build_env()
环境变量字典（VOC_DATA_ROOT=..., LD_PROFILE=tiny, ...）
   ↓  作为 subprocess 的 env 参数传进去
config 文件被 mmcv 读取时
   ↓  configs/_base_/voc_dataset.py 里的 os.environ.get(...)
最终 config 里 data/optimizer/img_scale 被改写
```

具体到数字，如果你不指定 `--profile`，程序会看显卡显存自动选档：

| 档位 | 分辨率 | batch | 读数据的进程 |
|---|---|---|---|
| tiny（显存 < 12G） | 640×384 | 1 | 0 |
| low（12~20G） | 800×480 | 1 | 0 |
| default（≥20G） | 1000×600 | 1 | 2 |
| high（手动） | 1333×800 | 1 | 4 |

你那张 4G 的 3050，会落到 `tiny`。

### 3.4 第三步：真正的训练进程里发生了什么

`subprocess` 启动的新进程里，跑的是 `LD/tools/train.py`。它按顺序做五件事：

**① 读配置**

```python
cfg = Config.fromfile(args.config)      # 把 configs/ld_r18_voc_loc_only.py 读进来
```

这一步会：读壳的 config → 发现 `_base_` → 递归去读官方 config → 合并 → 得到一份完整的配置。同时，壳的 config 里那段 `runpy` 代码会执行，读环境变量，把 `data`、`optimizer` 覆盖掉。

**② 建数据集**

```python
dataset = build_dataset(cfg.data.train)
```

变成一个"能按下标取出一张图 + 它的标注"的对象。数据增强流程（缩放、翻转、归一化）在 config 的 `pipeline` 里定义。

**③ 建模型**

```python
model = build_detector(cfg.model, train_cfg=..., test_cfg=...)
```

按 1.3 说的递归装配。如果是 LD 配置，模型类型是 `KnowledgeDistillationSingleStageDetector`，它会在初始化时**顺便把教师模型也建出来并加载权重**。

**④ 建优化器和训练循环**

```python
optimizer = build_optimizer(model, cfg.optimizer)
runner = EpochBasedRunner(model, ...)
```

**⑤ 开始循环**：取一个 batch → 前向 → 算 loss → 反向传播 → 更新参数。重复几千次。

### 3.5 一个 batch 进模型，经历了什么

假设用 `tiny` 档，输入是 640×384 的图，batch=1。

**第一步：骨干网提取特征**

ResNet18 把图变成特征图，再经过 FPN，得到 **5 张不同大小的特征图**：

| 层 | 尺寸 | 每个格子对应原图大小 |
|---|---|---|
| 1 | 48×80 | 8×8 像素 |
| 2 | 24×40 | 16×16 |
| 3 | 12×20 | 32×32 |
| 4 | 6×10 | 64×64 |
| 5 | 3×5 | 128×128 |

**第二步：每个格子铺一个 anchor**

5 层的 anchor 总数：

```
48×80 + 24×40 + 12×20 + 6×10 + 3×5 = 3840 + 960 + 240 + 60 + 15 = 5115 个
```

也就是说，一张图会产生 **5115 个候选框**。

**第三步：每个 anchor 输出两样东西**

- 分类分数：`[1, 20, H, W]` → 每个 anchor 20 个数（20 个类别各有几分）
- 位置分布：`[1, 68, H, W]` → 每个 anchor 68 个数（4 条边 × 每条边 17 个格子）

粗算一下：`5115 × (20 + 68) ≈ 45 万` 个数字。这就是检测模型一张图的输出量。

**第四步：分配正负样本**

5115 个 anchor 里，绝大部分是背景。ATSS 分配器会挑出**几十个**跟真实物体重合度高的当正样本，其余当负样本。

**第五步：算 loss**

这就是第 5 章要讲的内容。

### 3.6 loss 的骨架（先看结构，细节下一章）

学生要学的有四类东西：

| loss | 学什么 | 谁给的监督 |
|---|---|---|
| `loss_cls` | 这个位置是什么类别 | 数据集（硬标签） |
| `loss_bbox` | 框的位置准不准 | 数据集（硬标签） |
| `loss_dfl` | 每条边的分布形状对不对 | 数据集（由坐标换算出的分布） |
| `loss_ld` / `loss_ld_vlr` | 分布要像老师 | **老师**（软标签） |
| `loss_kd` | 分类分数要像老师 | **老师**（软标签） |

前三个是**常规检测损失**，四个变体都有。后两个（三个）才是蒸馏，四个变体的差异只在这里。

---

## 第 4 章 配置文件逐行读

### 4.1 `configs/_base_/voc_dataset.py`：它在解决什么问题

这个文件是整层壳最重要的文件，但它的目的可以用一句话说清：

> **把"我想怎么跑实验"翻译成"config 里该写什么"。**

它干了六件事，我们一件一件看。

**（1）确定数据集在哪**

```python
PROJECT_ROOT = Path(project_root).resolve()
VOC_DATA_ROOT = Path(os.environ.get('VOC_DATA_ROOT', PROJECT_ROOT / 'data' / 'VOCdevkit'))
DATA_ROOT = str(VOC_DATA_ROOT.resolve()) + os.sep
```

翻译：先看环境变量 `VOC_DATA_ROOT` 有没有设置，没有就用默认的 `<仓库>/2022_cvpr_ld/data/VOCdevkit`。

`os.sep` 是路径分隔符（Windows 上 `\`，Linux 上 `/`）。加它是因为后面全是字符串拼接：`DATA_ROOT + 'VOC2007/ImageSets/...'`。

**（2）选分辨率档位**

```python
RUNTIME_PROFILES = {
    'tiny':    dict(width=640,  height=384, samples_per_gpu=1, workers=0),
    'low':     dict(width=800,  height=480, samples_per_gpu=1, workers=0),
    'default': dict(width=1000, height=600, samples_per_gpu=1, workers=2),
    'high':    dict(width=1333, height=800, samples_per_gpu=1, workers=4),
}
```

就是一张查找表。`auto` 会根据显存自动选（见 3.3 的表）。

为什么这些尺寸都是 32 的倍数？因为 FPN 要下采样 5 次（32 = 2⁵），尺寸不是 32 的倍数会对不齐。

**（3）写数据增强流程**

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),      # 读图
    dict(type='LoadAnnotations', with_bbox=True),  # 读标注
    dict(type='Resize', img_scale=(宽, 高), keep_ratio=True),  # 缩放
    dict(type='RandomFlip', flip_ratio=0.5),        # 50% 概率左右翻转（数据增强）
    dict(type='Normalize', ...),                    # 减均值除标准差
    dict(type='Pad', size_divisor=32),              # 补到 32 的倍数
    dict(type='DefaultFormatBundle'),               # 打包成 tensor
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),  # 只保留这三样
]
```

这就是**数据流水线**：一张图从磁盘到模型输入，要经过这 8 步。

**（4）算学习率**

```python
optimizer = dict(lr=0.01 * SAMPLES_PER_GPU * GPU_COUNT / 16)
```

为什么要这么算？因为**学习率和 batch 大小要成正比**：一次看 16 张图的学习率，和一次看 1 张图的学习率显然不一样。

`/16` 是基准：16 张图时用 0.01。batch=1 时就是 `0.01/16 ≈ 0.000625`。

> 顺带一提：MMDetection 的标准做法是基准用 0.02（即 `0.02 * bs / 16`），这里用了 0.01，是标准值的一半。四组实验都用同一个公式，所以横向比较没问题，但和论文的绝对数值对不齐。

**（5）判断 VOC 数据集是哪种摆放方式**

VOC 数据集有两种常见摆法：

```text
摆法 A（分开）:  VOCdevkit/VOC2007/...  和  VOCdevkit/VOC2012/...
摆法 B（合并）:  VOCdevkit/VOC2007/...  里直接放 16551 张训练图
```

代码靠"**VOC2012 的 trainval.txt 在不在**"来判断：

```python
if VOC2012 的 trainval.txt 存在:
    用两个目录（VOC2007 + VOC2012）
else:
    只用 VOC2007 一个目录（合并版）
```

为什么不看目录在不在？因为合并版可能残留一个空的 `VOC2012/` 目录。看文件更可靠。

**（6）把结果打包返回**

```python
return dict(data=..., optimizer=..., evaluation=..., img_norm_cfg=..., ...)
```

返回一个字典，交给壳的 config 用 `globals().update()` 塞进命名空间——于是这些键就成了 config 的顶层键。

### 4.2 `configs/ld_r18_voc.py`：26 行，每行都重要

```python
_base_ = '../../LD/configs/ld/ld_r18_gflv1_r101_fpn_voc_1x.py'   # ① 继承官方 LD 配置

TEACHER_CHECKPOINT = os.environ.get(                          # ② 教师权重路径
    'TEACHER_CHECKPOINT',
    '{{ fileDirname }}/../runs/teacher_r101/epoch_16.pth')

model = dict(                                                  # ③ 覆盖三个蒸馏权重
    teacher_config='{{ fileDirname }}/gfl_r101_voc_teacher.py',
    teacher_ckpt=TEACHER_CHECKPOINT,
    bbox_head=dict(
        loss_kd=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=10, T=2),
        loss_ld=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10),
        loss_ld_vlr=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=0.25, T=10),
    ))

globals().update(_runpy.run_path(                              # ④ 执行数据覆盖函数
    '{{ fileDirname }}/_base_/voc_dataset.py')['build_voc_overrides'](
        '{{ fileDirname }}/../..'))
```

**① 继承**：官方那份 `ld_r18_gflv1_r101_fpn_voc_1x.py` 里已经写好了学生结构（ResNet18 + FPN + LDHead）、教师结构、优化器、4 个 epoch 的训练计划。这里只是"接着它改"。

**② `{{ fileDirname }}` 是什么**：这是 mmcv 提供的一个特殊变量，会被替换成"当前 config 文件所在的目录"。因为 mmcv 读 config 时会把文件复制到临时目录再执行，写相对路径会指错地方，所以要用它锚定。

**③ 三个蒸馏权重**：

- `loss_kd`：分类蒸馏，权重 10，温度 2
- `loss_ld`：定位蒸馏，权重 0.25，温度 10
- `loss_ld_vlr`：VLR 区域定位蒸馏，权重 0.25，温度 10

**④ `globals().update(...)` 是什么**：把 `voc_dataset.py` 那个函数算出来的字典，塞进当前文件的作用域。效果等价于你在这一行手写了：

```python
data = ...
optimizer = ...
evaluation = ...
```

为什么要这么绕？因为 mmcv 的 `_base_` 只能合并"字典"，不能"调用函数"。这个文件需要调用函数（`build_voc_overrides`）才能算出配置，只好用这种写法绕过限制。

**顺序有讲究**：`model = dict(...)` 必须写在 `globals().update(...)` 之前。因为后者塞进来的字典里没有 `model` 键，不会覆盖；但如果哪天有人往里加了，就会静默盖掉三个蒸馏权重。

### 4.3 三个变体：只是改数字

```python
# kd_only.py —— 只教分类
_base_ = ['./ld_r18_voc.py']
model = dict(bbox_head=dict(
    loss_kd=dict(loss_weight=10, T=2),
    loss_ld=dict(loss_weight=0, T=10),        # ← 改成 0
    loss_ld_vlr=dict(loss_weight=0, T=10),    # ← 改成 0
))
```

```python
# loc_only.py —— 只教定位
_base_ = ['./ld_r18_voc.py']
model = dict(bbox_head=dict(
    loss_kd=dict(loss_weight=0, T=2),         # ← 改成 0
    loss_ld=dict(loss_weight=0.25, T=10),
    loss_ld_vlr=dict(loss_weight=0.25, T=10),
))
```

```python
# full.py —— 论文完整方法
_base_ = ['./ld_r18_voc.py']    # 什么都不改
```

**权重设成 0 意味着什么？** 在 loss 计算里，最后会乘上这个权重，乘 0 就等于这一项不贡献梯度。但注意：**计算还是会发生**（只是结果乘 0），所以并不会省时间。这是很多人第一次读代码会误解的地方。

---

## 第 5 章 蒸馏的数学（从你自己的练习讲起）

### 5.1 先看你已经写过的 `练习1_最小定位蒸馏.py`

你写的那个练习，其实已经是 `loss_ld` 的核心了。对照着看：

**你的代码**：

```python
T_p = torch.tensor([[[0.05,0.10,0.60,0.20,0.05],    # 4 条边，每条 5 个格子
                     [0.10,0.60,0.20,0.05,0.05],
                     [0.05,0.20,0.60,0.10,0.05],
                     [0.05,0.10,0.20,0.60,0.05]]])   # 这是"教师"的分布（已经 softmax 过的概率）

S_l = torch.zeros((3,4,5), requires_grad=True)       # 3 个样本，4 条边，5 个格子（学生 logits）

pos_mask = torch.tensor([True, False, True])         # 只有第 0、2 个样本参与 loss

KL = F.kl_div(log_softmax(S_l), T_p, reduction='none').sum(dim=-1).mean(dim=-1)
KL = KL[pos_mask].mean()
KL.backward()
```

**官方代码**（`ld_head.py` 里的 `loss_single`）：

```python
loss_ld = self.loss_ld(pred_corners, soft_corners,
                       weight=weight_targets, avg_factor=4.0)
```

**两者的差别一共五处**（前三处是"必须补的"，后两处是"锦上添花"）：

| | 你的练习 | 官方代码 | 影响 |
|---|---|---|---|
| 每条边几个格子 | 5 | **17**（reg_max=16） | 只是规模变化 |
| 格子维怎么聚合 | 你 `sum(dim=-1)` 求和 | `mean(dim=1)` 求平均 | 数值差一个 `1/格子数` 的常数（下面有实测） |
| 温度 | 没有（T=1） | **T=10**，且最后乘 `T²` | 决定"学到多少小概率信息" |
| 样本怎么选 | 你手写的 `pos_mask` | 由 ATSS 自动分配；正样本走 `loss_ld`，VLR 区域走 `loss_ld_vlr` | 决定"学谁" |
| 权重 | 全 1 | 用**学生自己的质量分**（VLR 那份用 IoU）加权 | 决定"每个样本的份量" |

实测对比（用你练习 1 里那组数据算的，见 `练习2_把两个练习拼成loss_ld.py`）：

```text
你的写法（格子维求和）      kl = 0.451223
官方写法（格子维求平均）    kl = 0.090245
比值 = 5.0   ← 正好等于格子数 5
```

也就是说，**你写的 KL 本身完全正确**，但官方对格子维取了平均，数值上只有你的 1/5。真要换成 17 个格子，这个比例就变成 1/17。

所以你已经理解了 80%，剩下要补的就是上表后三行。

### 5.2 温度 T 在干什么

温度的作用是**把分布变平**。

假设老师的原始分布是 `[0.90, 0.07, 0.02, 0.01]`（很确信）。

- `T=1`：`softmax(logits/1)` → 还是 `[0.90, 0.07, 0.02, 0.01]`
- `T=10`：`softmax(logits/10)` → 变成 `[0.32, 0.25, 0.22, 0.21]`（平了很多）

**为什么要变平？** 因为 `[0.90, 0.07, 0.02, 0.01]` 里，后三个数太小，梯度几乎为 0，学生学不到。变平之后，"第二可能的位置"也变成 0.28，学生就能从中学到"虽然答案是第一个，但第二个也很像"。

论文里几个 T 的取值：

- 分类蒸馏用 **T=2**（20 个类别，本来就不算太尖）
- 定位蒸馏用 **T=10**（17 个格子，分布很尖，需要更强的软化）

**为什么最后要乘 `T²`？** 因为软化会让梯度变小（除以 T），所以要乘 `T²` 把梯度量级补回来，否则换个 T 就得重新调学习率。

你可以在自己的练习里改一个数字验证：把 `S_l/T` 里的 T 从 1 改成 10，看 loss 和梯度的变化。

### 5.3 三个 loss 的分工

回到那张表，这次加上代码位置：

| loss | 教什么 | 作用在哪些 anchor | 维度 | T | 权重 |
|---|---|---|---|---|---|
| `loss_kd` | 这个位置是什么类别 | 正样本（几十个） | 20 | 2 | 10 |
| `loss_ld` | 这个框的边在哪 | 正样本（几十个） | 68 | 10 | 0.25 |
| `loss_ld_vlr` | 差一点的那些框的边在哪 | VLR 区域（几百个） | 68 | 10 | 0.25 |

用大白话说：

- `loss_kd`：老师看着一个正确的位置说"这是狗，80% 确定，猫只有 15%"，学生跟着学。
- `loss_ld`：老师看着一个正确的位置说"左边界在 372 的概率 60%，371 是 15%"，学生跟着学。
- `loss_ld_vlr`：老师看着一个"IoU 不算高、但已经很接近物体"的位置，也说了一遍它的边界分布，学生跟着学。

### 5.4 VLR：你写的 `算法1` 的升级版

你写的算法 1：

```python
def calculate_val(DIoU_matrix, pos, a):
    return (DIoU_matrix >= a * pos) & (DIoU_matrix <= pos)
```

翻译：**挑出 DIoU 落在 `[a×pos, pos]` 之间的 anchor**。这就是"不够好、但还不错"的那一圈。

官方代码（`atss_assigner.py` 的 `get_vlr_region`）的核心也是这个区间，但有两个升级：

**升级一：`pos` 不是固定 0.5，而是每个 GT 自己算**

```python
threshold = 候选 anchor 的 IoU 均值 + 标准差
```

为什么要自适应？因为大物体和小物体的 IoU 分布差很多。固定阈值在一种物体上合适，在另一种上就不合适了。`均值 + 标准差` 是 ATSS 论文里验证过的自适应阈值。

**升级二：`a = 0.25`，而且判据用 DIoU、权重用 IoU**

```python
is_pos = (DIoU < threshold) & (DIoU >= 0.25 * threshold)
```

为什么用 DIoU 选、用 IoU 加权？DIoU 额外惩罚"中心点偏得远"的情况，选出来的区域更"正"；IoU 更直观，用来当权重（越重合学得越多）。

**这个区间到底在选什么？** 举个具体例子（阈值假设是 0.5）：

- IoU 超过自适应阈值（比如 0.5）的 anchor：已经是正样本了，走 `loss_ld`
- IoU 在 0.125~0.5 之间（即 0.25 倍阈值到阈值之间）：**差一点**，走 `loss_ld_vlr`
- IoU 更低：基本是背景，不管

论文的直觉是：中间这一圈数量远多于正样本，而且它们"只要再挪一点就对"，是最值得学的地方。

### 5.5 官方 `loss_single` 的伪代码

把工程细节去掉，`ld_head.py:116-282` 的骨架是这样的：

```python
def loss_single(anchors, cls_score, bbox_pred, labels, ..., soft_label, soft_target):
    # 1. 把各种张量摊平成 [anchor 数, 特征数]
    # 2. 找出正样本：label 在 0~19 之间的
    pos_inds    = 正样本的下标
    remain_inds = VLR 区域的下标

    # 3. 如果 imitation_method == 'gibox'（默认），先算特征模仿 loss
    #    注意：VOC 配置里它的权重是 0，所以算了也白算

    if 有正样本:
        # 4a. 常规检测 loss（三个）
        loss_bbox = ...
        loss_dfl  = ...

        # 4b. 定位蒸馏：正样本的分布 vs 老师的分布
        loss_ld   = KL(学生的分布, 老师的分布) × 权重

        # 4c. 分类蒸馏：正样本的分类 vs 老师的分类
        loss_kd   = KL(学生的分类, 老师的分类) × 权重
    else:
        全部置 0（防止空图崩掉）

    if 有 VLR 锚点:
        # 5. VLR 定位蒸馏
        loss_ld_vlr = KL(学生的分布, 老师的分布) × IoU权重
    else:
        置 0

    # 6. 分类 loss 用的是数据集标签，不是老师
    loss_cls = ...

    return 各个 loss
```

对照真实代码读的时候，你会发现里面有一行 `loss_kd_neg = 0 * self.loss_kd(...)`——这是作者留的占位，乘 0 了，**不产生任何作用**，看到别慌。

---

## 第 6 章 补丁：知道它存在就行

`patches/ld-mmcv-1.7.2.patch` 改的是官方代码，因为官方代码是 2021 年写的，跑在 2023 年的环境上会报错。

改了 10 个文件，分三类：

1. **版本检查放宽**：官方代码说"只支持 mmcv ≤1.3"，我们用的是 1.7.2，所以把这个检查改成"也允许 1.7.2"。
2. **NumPy 改名**：`np.int` 这种写法在新版 NumPy 里被删了，改成 `np.int64`。一共 10 处，都是机械替换。
3. **API 变了**：比如某个函数以前接受位置参数，现在要求写成关键字参数；某个张量要指定放在哪张卡上。改了 3 处。

**你不需要背这些。** 只要知道：

- 补丁是**幂等**的（重复打不会出错）——脚本靠 `git apply --check --reverse` 判断"是不是已经打过了"。
- 冲突时**一个文件都不改**就报错，不会留下改了一半的乱摊子。

这是仓库里最"工程"的部分，读不懂不影响你理解算法。

---

## 第 7 章 现在的状态：能不能跑

### 7.1 已有的 `runs/` 不是这套代码跑出来的

`runs/` 里有四组 1-epoch 的产物，日志里出现 `mmengine`、`OptimWrapper`、`PackDetInputs`——这些是 **MMDetection 3.x** 的东西。而这套实验层要求的是 **MMDetection 2.10**。

也就是说：那四个目录是**用另一套环境跑的**，和 `ld_core_experiment` 不是同一条路径。里面的 mAP 2.8% 也不能当基线用。

### 7.2 为什么 2.8% 这么低

三个原因叠加：

1. 只训了 1 个 epoch（正常要 16+ 个）
2. 分辨率被压到 640 宽左右（正常 1333×800）
3. 从 ImageNet 初始化开始，模型还没学会检测

这三个都是"冒烟测试"的正常结果，不是 bug。

### 7.3 你这台机器跑不了

| 要求 | 你的机器 |
|---|---|
| Linux | Windows ❌ |
| torch 2.0/2.1 | 2.3.1 ❌ |
| mmcv-full 1.7.2 | mmcv 2.2.0 ❌ |
| mmdet 2.10 | mmdet 3.3.0 ❌ |
| 显存 ≥12G | 4G ❌ |

`install_environment.py` 甚至会直接拒绝非 Linux 系统。

**三条出路**：

1. 租一台 Linux 云 GPU（原设计路径）；
2. 在 WSL2 里重新装一套 Python 3.9 + torch 2.0.1 + mmcv 1.7.2（4G 显存只够 tiny 档，教师 R101 会很吃紧）；
3. 放弃这套壳，用你现有的 MMDet 3 环境重写 LD（工作量最大，且失去了"不改官方代码"的意义）。

**在跑通之前，你完全可以只做"读代码 + 看懂算法"这件事**，这不需要 GPU。

---

## 第 8 章 学习计划（两周）

### 第一周：搞懂"这是什么"

| 天 | 任务 | 产出 |
|---|---|---|
| 1 | 读第 0 章 + 名词小抄 | 能用自己的话解释 LD 在干什么 |
| 2 | 读第 1 章 1.1~1.5，打开官方 r18 config 对照 | 指出 config 里哪个键管什么 |
| 3 | 读第 1.6~1.7，**在纸上**画一遍 config 合并 | 手算出合并后的 optimizer 和 data |
| 4 | 读第 2、3 章 | 说出三层各自的职责 |
| 5 | 读 3.5，**手算** 640×384 的 anchor 数 | 得出 5115 |
| 6-7 | 读第 4 章，对照三个变体文件 | 说出"改哪一行就换了变体" |

### 第二周：搞懂"为什么这么写"

| 天 | 任务 | 产出 |
|---|---|---|
| 8 | 读 5.1~5.2，把 `练习1` 的 T 改成 10 跑一遍 | 观察 loss 变化 |
| 9 | 读 5.3，画出三个 loss 的分工图 | 一张图 |
| 10 | 读 5.4，对照 `算法1` 和 `get_vlr_region` | 说出两个升级点 |
| 11 | 读 5.5 伪代码，对照真实代码逐行 | 能指出 `loss_kd_neg` 是死代码 |
| 12 | 读第 6 章 | 知道补丁在干嘛（不求甚解） |
| 13 | 读第 7 章 | 说清为什么现在跑不了 |
| 14 | 复习 + 做下面的自测题 | 全部答对 |

### 自测题（答不上来就回去看对应章节）

1. 用一句话解释：蒸馏为什么比"只学标准答案"更好？（0.2）
2. `loss_ld` 和 `loss_kd` 分别作用在什么维度上？（5.3）
3. 为什么定位蒸馏的温度用 10，分类用 2？（5.2）
4. 一份 config 写 `optimizer = dict(lr=0.001)`，合并后 `momentum` 还在吗？为什么？（1.5）
5. 640×384 的输入，anchor 总数是多少？怎么算的？（3.5）
6. `loss_weight=0` 的 loss 还会被计算吗？会省时间吗？（4.3）
7. VLR 区域大概是 IoU 在什么范围的 anchor？（5.4）
8. `globals().update(...)` 那一行到底干了什么？为什么要这么写？（4.2）
9. 为什么这套代码不直接改官方代码？（0.5）
10. 四个变体里，哪一个回答了"定位知识比语义知识重要"这个问题？（0.4）

---

## 附录 A：文件对照速查

| 你想知道 | 去看 |
|---|---|
| 实验想验证什么 | 第 0 章 |
| config 怎么读 | `LD/configs/gfl/gfl_r18_fpn1x_voc.py` |
| 超参怎么被改 | `configs/_base_/voc_dataset.py` |
| 一条命令干了什么 | `scripts/train_ld.py` → `scripts/_common.py` |
| 教师怎么挂上去 | `LD/mmdet/models/detectors/kd_one_stage.py` |
| loss 怎么算 | `LD/mmdet/models/dense_heads/ld_head.py:116-282` |
| VLR 怎么选 | `LD/mmdet/core/bbox/assigners/atss_assigner.py:183-298` |
| KL 怎么算 | `LD/mmdet/models/losses/kd_loss.py:12-36` |
| 补丁改了啥 | `patches/ld-mmcv-1.7.2.patch` |

## 附录 B：常见困惑速查

**Q：为什么 config 里写的是字符串 `'GFL'`，不是类名？**
A：为了解耦。换模型只改字符串，不用改 import。

**Q：为什么 loss 要分成好几个，不能合成一个？**
A：因为量级和收敛速度不一样。分类 loss 和定位 loss 的数量级差很多，各自有权重才能平衡。

**Q：`samples_per_gpu=1` 是不是太小了？**
A：是。但 4G 显存只够这么大。检测任务正常用 8~16。小 batch 会让 BatchNorm 统计不稳，所以官方配置里 `norm_eval=True`。

**Q：`RepeatDataset(times=3)` 和 `total_epochs=4` 是什么关系？**
A：`times=3` 让一个 epoch 变成"原数据看 3 遍"，所以 4 个 epoch 实际上看了 12 遍原数据。冒烟测试设 `times=1` 才是真的 1 遍。

**Q：为什么教师权重是 `.pth` 文件，不能直接用教师模型？**
A：因为教师是单独的模型，必须先把权重存下来。学生的 checkpoint 里**不含教师**（代码故意这么设计，省空间）。

**Q：`mAP` 和 `AP50` 有什么区别？**
A：`AP50` 只要求 IoU>0.5 就算检对了。COCO 的 `mAP` 是 IoU 从 0.5 到 0.95 每隔 0.05 算一次再平均，更严格。VOC 的 `mAP` 是在 IoU=0.5 下、对 20 个类别求平均，名字叫 mAP，其实和 COCO 的 AP50 是一回事。看数字时先确认是哪个数据集的口径，否则没法比。

---

## 附录 C：想看工程细节

仓库里还有一份 `ld_core_experiment_代码导读.md`，是给"已经会 MMDetection、想快速搞清这份仓库工程决策"的人写的。内容包括：配置合并的精确机制、环境变量的优先级、补丁的十个文件逐条、单卡限制的原因、已知的十个坑。

**建议你先读完这份《从零读懂》，再去看那份。** 顺序反了会觉得那份很劝退——我第一次写的时候就没考虑到这一点。
