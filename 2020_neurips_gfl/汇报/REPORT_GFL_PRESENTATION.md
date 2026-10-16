# GFL 论文汇报提纲

**Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection**
Xiang Li 等，NeurIPS 2020。

[论文 PDF](papers/gfl_2006.04388.pdf) · [详细备课稿](REPORT_GFL_PRESENTATION_FULL.md) · [演示脚本](debug_loss_single.py) · [逐行注释版](debug_loss_single_annotated.py)

这次讲清楚两个问题、两项设计和对应代码。主要内容是方法理解与张量调试，不展示训练实验。

## 一、


> 我从 CrossKD 读到定位蒸馏 LD，遇到了边界框分布表示这个知识点，于是继续看 GFL。这次重点讲：分类分数怎样表达定位质量，以及四条边的分布怎样预测和训练。

## 二、研究动机：GFL 改了哪两处

![GFL 原文 Figure 4：分类质量联合表示与分布式回归](figures/gfl_original_figure4.png)

*来源：GFL 原文 Figure 4。按上面分类、下面回归的顺序讲。*

### 1. 分类分数与框的质量可能不一致

假设两个框检测同一个人：

| 预测框 | person 分数 | 与真实框的 IoU |
|---|---:|---:|
| A | 0.95 | 0.55 |
| B | 0.85 | 0.90 |

A 更确信“这里是人”，B 的位置却更准。若两框互相重叠超过 NMS 阈值，NMS 可能先保留高分 A，再删除 B。

**原因：普通分类正样本目标通常都是 1，没有直接要求这个分数反映定位质量。**

GFL 的办法：把正样本真实类别上的目标从 1 改成当前框的 IoU，用 **QFL** 学习类别与定位质量的联合分数。

注意三种 IoU 的比较对象：

| 环节 | 比较谁与谁 |
|---|---|
| 当前配置的标准 ATSS 分配 | 预设 anchor 与真实框 |
| QFL 的质量目标 | 当前预测框与匹配的真实框 |
| NMS 去重 | 一个预测框与另一个预测框 |

### 2. 单个距离不能描述分布形状

直接回归只输出一个距离，例如 2.8。GFL 为每条边输出离散分布，再求期望得到连续距离，用 **DFL** 增加对分布的直接监督。

“狄拉克表示”可以通俗理解为：把全部质量放在一个确定位置，最终只表达一个数。**它也能预测小数，不是只能预测整数。**

分布形状可以提供定位不确定性的线索，但不能直接把输出当成经过校准的真实误差概率。DFL 的 0.2、0.8 首先是由距离插值构造的监督权重。

## 三、先讲清参考点、四条边和输出形状

### 1. 从同一个参考点表示真实框和预测框

![参考点、stride=8 网格与预测框解码](figures/gfl_prediction_process.png)

*自绘机制图，不是检测效果图。讲图时用图内数值；下面另用一组小数值手算。*

当前配置每个位置只有一个 anchor，即 A=1。训练时 ATSS 用预设 anchor 挑正样本，回归时取 anchor 中心作为距离参考点。**参考点不必是真实框或预测框的中心。**

假设参考点 P=(56,56)，真实框为 (40,32,80,88)，stride=8：

$$
(l,t,r,b)=(56-40,\ 56-32,\ 80-56,\ 88-56)=(16,24,24,32).
$$

换成特征图单位：

$$
(l,t,r,b)/8=(2,3,3,4).
$$

解码时，若参考点和距离采用相同单位：

$$
\hat B=(x-\hat l,\ y-\hat t,\ x+\hat r,\ y+\hat b).
$$

**左右距离可以不同，因此参考点不必位于框正中央。**

ATSS 只需说明：各层先选中心距离真实框中心最近的 top-k 个候选，再用候选 anchor 的 IoU 均值加标准差设阈值，结合中心在真实框内等条件确定正样本。当前 topk=9；一个真实框可以对应多个正样本位置。

### 2. 为什么是 68 个回归输出

COCO 有 80 类；reg_max=16，每条边有 0—16 共 17 个固定候选距离：

$$
4\times(16+1)=68.
$$

设一层有 K=N×H×W 个位置，其中 M 个正样本：

| 内容 | 原始形状 | 展平或筛选后 |
|---|---|---|
| 分类 logits | (N,80,H,W) | (K,80) |
| 回归 logits | (N,68,H,W) | (K,68) |
| 类别标签 | (N,HW) | (K,) |
| 匹配的真实框 xyxy | (N,HW,4) | (K,4) |
| 正样本四边分布 | (M,68) | (4M,17) |
| 正样本四边距离目标 | (M,4) | (4M,) |

分类对 80 类分别做 sigmoid；回归对每条边的 17 个候选做 softmax。**两种“分类维度”的含义不同。**

### 3. 对应的关键转换代码

来自 [GFLHead.loss_by_feat_single](../mmdetection/mmdet/models/dense_heads/gfl_head.py#L233)，省略无关分支：

~~~python
# 将每个位置的一整组输出放到同一行。
cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
labels = labels.reshape(-1)
bbox_targets = bbox_targets.reshape(-1, 4)  # 正式实现这里是 xyxy，不是距离
anchors = anchors.reshape(-1, 4)

# 前景类别编号为 0—79，背景编号为 80。
pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero().squeeze(1)
pos_bbox_pred = bbox_pred[pos_inds]        # (M,68)
pos_bbox_targets = bbox_targets[pos_inds]  # (M,4)
pos_anchors = anchors[pos_inds]

# anchor 中心转为特征图单位；不能一般地直接取 anchor 前两列。
pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]
pos_decode_bbox_targets = pos_bbox_targets / stride[0]

pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)  # (4M,17)
target_corners = self.bbox_coder.encode(
    pos_anchor_centers, pos_decode_bbox_targets, self.reg_max
).reshape(-1)  # (4M,)
~~~

其中 anchor 中心是 ((x1+x2)/2, (y1+y2)/2)。encode 内部通过 [bbox2distance](../mmdetection/mmdet/structures/bbox/transforms.py#L206) 计算：

~~~python
left = points[..., 0] - bbox[..., 0]
top = points[..., 1] - bbox[..., 1]
right = bbox[..., 2] - points[..., 0]
bottom = bbox[..., 3] - points[..., 1]
~~~

输出顺序是左、上、右、下。正式实现还会 clamp 到合法距离范围，避免右侧整数索引越界。0—16 的单位是相应层 stride，不是像素。

## 四、Integral 与 DFL：一条边从预测到监督

### 1. Integral：把分布变成连续距离——公式（5）

$$
S_i=\frac{e^{z_i}}{\sum_{j=0}^{16}e^{z_j}},
\qquad
\hat y=\sum_{i=0}^{16}iS_i.
$$

z_i 是预测 logit，S_i 是 softmax 概率，i 是固定距离刻度。例：S₂=0.2、S₃=0.8，其余为 0，则期望为 2.8。实际有限 logits 的 softmax 概率均大于 0，这里用理想分布说明。

[Integral.forward](../mmdetection/mmdet/models/dense_heads/gfl_head.py#L42) 的核心只有两行：

~~~python
# (M,68) → (4M,17)，每条边单独归一化。
x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
# project 固定为 [0,1,...,16]；点积求期望，恢复成每框四个距离。
x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
~~~

得到四个距离后，用前面的“左上减、右下加”解码成 xyxy 框。

### 2. DFL：监督概率怎样分配——公式（6）

真实距离 y 位于相邻整数 i、i+1 之间：

$$
L_{\mathrm{DFL}}
=-(i+1-y)\log S_i-(y-i)\log S_{i+1}.
$$

手算例子：

$$
y=2.8,\quad w_l=3-2.8=0.2,\quad w_r=2.8-2=0.8,
$$

$$
L_{\mathrm{DFL}}=-0.2\log S_2-0.8\log S_3.
$$

**0.2、0.8 来自真实距离，不是网络预测。** 两个权重和为 1，并使目标期望为 2×0.2+3×0.8=2.8。

[distribution_focal_loss](../mmdetection/mmdet/models/losses/gfocal_loss.py#L144)：

~~~python
dis_left = label.long()                  # 非负距离向下取整：2.8 → 2
dis_right = dis_left + 1                 # 3
weight_left = dis_right.float() - label  # 0.2
weight_right = label - dis_left.float()  # 0.8

loss = (
    F.cross_entropy(pred, dis_left, reduction='none') * weight_left
    + F.cross_entropy(pred, dis_right, reduction='none') * weight_right
)
~~~

pred 是 logits，不能先手动 softmax 再传给 cross_entropy；交叉熵内部已包含 log-softmax。若五个候选概率都为 0.2，DFL=-log(0.2)≈1.6094。

### 3. 为什么不能只监督期望

| 分布 | 概率分配 | 期望 |
|---|---|---:|
| A | S₂=0.2，S₃=0.8 | 2.8 |
| B | S₀=0.3，S₄=0.7 | 2.8 |

其他三条边相同，两者解码出的框就完全相同。框级损失无法区分它们，而 DFL 会引导概率集中到标注距离附近的 2、3。

**B 并非数学上不合法，也不能说其最终框更差。DFL 增加的是对训练中分布形状的约束；检测收益要由实验验证。（因为是三个损失同时约束）**

## 五、GIoU：监督最后画出的框

设 A 为预测框，B 为真实框，C 为同时包住两框的最小外接矩形：

$$
\operatorname{IoU}=\frac{|A\cap B|}{|A\cup B|},
\qquad
\operatorname{GIoU}
=\operatorname{IoU}-\frac{|C|-|A\cup B|}{|C|},
$$

$$
L_{\mathrm{GIoU}}=1-\operatorname{GIoU}.
$$

为什么是 1-GIoU：框越接近，期望 GIoU 越大；训练最小化损失，所以转换成越小越好。框完全一致时损失为 0。

为什么不只用 IoU：两个框不相交时，IoU 都是 0；GIoU 的外接空隙项仍能反映部分几何差异。

**两种监督互补：GIoU 看最终框，DFL 看形成四条边距离的概率分布。** 回归梯度可以沿“框 → 期望 → softmax → logits”反传。

正式计算入口：[giou_loss](../mmdetection/mmdet/models/losses/iou_loss.py#L122)。

## 六、QFL：让类别分数也学习定位质量

### 1. 标签怎么构造

假设类别为 [人,车,狗]，正样本真实类别是车，当前预测框与匹配真实框 IoU=0.7：

~~~text
普通分类目标：[0, 1.0, 0]
QFL 目标：   [0, 0.7, 0]
背景目标：   [0, 0.0, 0]
~~~

真实类别决定哪一格非零；当前 IoU 决定这一格多大。因此，soft one-hot label 是训练时构造的，质量目标会随预测框改变。

[质量标签生成代码](../mmdetection/mmdet/models/dense_heads/gfl_head.py#L288)：

~~~python
pos_bbox_pred_corners = self.integral(pos_bbox_pred)
pos_decode_bbox_pred = self.bbox_coder.decode(
    pos_anchor_centers, pos_bbox_pred_corners)

# score 初始全零，只为正样本填入当前预测框的 IoU。
score[pos_inds] = bbox_overlaps(
    pos_decode_bbox_pred.detach(),
    pos_decode_bbox_targets,
    is_aligned=True)
~~~

detach 将这次算出的 IoU 当作固定监督目标，QFL 不沿该目标生成路径回传到框坐标。框仍由 GIoU 和 DFL 训练；共享特征仍可接收分类梯度。

### 2. 从 FL 到 QFL——公式（1）、（2）

先用一句话回顾 Focal Loss，忽略类别平衡系数 α：

$$
L_{\mathrm{FL}}=-(1-p_t)^\gamma\log p_t,
\qquad
p_t=\begin{cases}p,&y=1\\1-p,&y=0.\end{cases}
$$

正确类别已经预测得很好时，调制因子减小其贡献。这里 p 是一个类别的 sigmoid 输出，原始标签是 0 或 1。

QFL 将目标推广到连续值 q∈[0,1]：

$$
\operatorname{BCE}(p,q)=-q\log p-(1-q)\log(1-p),
$$

$$
L_{\mathrm{QFL}}=|q-p|^\beta\operatorname{BCE}(p,q),
\qquad \beta=2.
$$

**交叉熵负责对齐软目标，误差调制因子降低已接近目标的项的贡献。** 正样本真实类别的 q 是 IoU，其余类别的 q 为 0。希望学到接近当前 q 的分数，不保证每次参数更新都单调接近。

例：q=0.7，p=0.2 时 QFL≈0.2984；p=0.6 时≈0.0063；p=0.7 时为 0。

### 3. QFL 的核心实现

下面摘自 [quality_focal_loss](../mmdetection/mmdet/models/losses/gfocal_loss.py#L13) 并简化变量初始化；pred 是展平后的分类 logits，label 是类别编号，score 是质量标签：

~~~python
pred_sigmoid = pred.sigmoid()

# 先让所有类别以 0 为监督目标。
zerolabel = torch.zeros_like(pred)
loss = F.binary_cross_entropy_with_logits(
    pred, zerolabel, reduction='none') * pred_sigmoid.pow(beta)

# 再替换正样本“真实类别那一格”的损失。
pos = ((label >= 0) & (label < pred.size(1))).nonzero().squeeze(1)
pos_label = label[pos].long()
scale_factor = score[pos] - pred_sigmoid[pos, pos_label]
loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
    pred[pos, pos_label], score[pos], reduction='none'
) * scale_factor.abs().pow(beta)

loss = loss.sum(dim=1)  # 每个位置把各类别损失相加
~~~

先计算再替换是向量化实现，不是两次训练。推理时没有真实框，直接使用已经学到的 sigmoid 联合分数。

## 七、把三项损失和推理串起来

### 1. 总损失——公式（8）的结构

$$
L=L_{\mathrm{QFL}}+\lambda_0L_{\mathrm{GIoU}}+\lambda_1L_{\mathrm{DFL}}.
$$

[当前配置](../mmdetection/configs/gfl/gfl_r50_fpn_1x_coco.py) 对应权重为 1、2、0.25。这里各项还要完成对应的样本加权和归一化；正式 loss 对象已经乘过配置权重，外部不要再重复乘。

| 损失 | 监督对象 | 最需要记住的一句话 |
|---|---|---|
| QFL | 各类别的联合分数 | 正确类别上的分数要反映当前框的质量 |
| GIoU | 解码后的完整框 | 最后画出的框要贴近真实框 |
| DFL | 四条边的距离分布 | 概率向真实距离附近的相邻刻度集中 |

实现中另有回归样本权重：

~~~python
weight_targets = cls_score.detach().sigmoid()
weight_targets = weight_targets.max(dim=1)[0][pos_inds]  # (M,)
edge_weights = weight_targets[:, None].expand(-1, 4).reshape(-1)  # (4M,)
~~~

这是正样本位置的最大类别联合分数，**不是直接计算的 IoU**。同一框的四条边共用它。不是把三种损失互相相乘，也不是认定低分正样本的标注不可靠。

[单层函数](../mmdetection/mmdet/models/dense_heads/gfl_head.py#L233) 返回三项损失和回归权重和；[多层函数](../mmdetection/mmdet/models/dense_heads/gfl_head.py#L325) 遍历 FPN 层，并对回归损失做跨层权重归一化，框架再汇总各层损失。

### 2. 推理时走哪条路

~~~text
图像 → Backbone → FPN → 各层检测头
                         ├─ 分类 logits → sigmoid → 联合分数
                         └─ 回归 logits → softmax → 期望距离 → 还原像素框
                                          ↓
                        分数筛选 → NMS 去除重复框 → 最终检测结果
~~~

NMS 按分数优先保留框，并抑制与已保留框高度重叠的同类候选。QFL 试图改善这个排序依据，让定位较好的框更有机会被留下。

### 3. 为什么叫 Generalized Focal Loss（备用）

主线讲清后，再补论文公式（7）：

$$
\operatorname{GFL}(p_l,p_r)
=-|y-(y_lp_l+y_rp_r)|^\beta
\big[(y_r-y)\log p_l+(y-y_l)\log p_r\big].
$$

论文统一表述中 p_l+p_r=1，y 位于两个端点之间。

- 端点取 0、1，p_r=p、p_l=1-p，得到 QFL。
- 端点取相邻距离刻度，去掉误差调制因子（β=0），对应 DFL 的两邻居加权交叉熵形式。
- QFL 的目标退化到 0 或 1，就回到忽略 α 的 FL。

**实际 DFL 始终对完整 17 维 softmax 训练；不是训练前就强制只有两个概率非零。**

## 八、现场代码只停这几个地方

打开 [debug_loss_single.py](debug_loss_single.py)，选择 DL 环境，F5 启动；F10 单步跳过，F11 进入自己的函数。进入 PyTorch 底层时用 Shift+F11 返回。

| 停在哪里 | 看什么 | 讲什么 |
|---|---|---|
| 展平与正样本提取 | (32,80)、(32,68)、3 个正样本 | 一个位置一套输出；正样本不是四倍位置数 |
| Integral | (3,68) → (12,17) → (3,4) | 四组概率各自归一化，再求期望 |
| 框解码与 quality | 预测框、真实框、IoU | 左上减、右下加；IoU 是 QFL 目标来源 |
| GIoU 与 DFL | 框损失、相邻刻度与权重 | 分别监督最终框与分布 |
| QFL | 真实类别分数、IoU 目标 | 标签是哪一类，以及该类目标多大 |
| backward 与 step | 梯度和更新结果 | 计算梯度与更新变量是两个步骤 |

~~~python
optimizer.zero_grad()  # 清理旧梯度
total_loss.backward()  # 沿计算图计算梯度
optimizer.step()       # 用梯度更新优化器管理的变量
~~~

运行方式（终端在项目根目录，已激活 DL 环境）：

~~~powershell
python 汇报/debug_loss_single.py --trace
~~~

演示前说明：**这是核心损失的张量模拟。** 随机 logits、手工正样本、单层特征；真实框目标在脚本中直接存为距离。没有运行完整图像检测器或 ATSS，也没有完整多层归一化。脚本直接优化 logits；真实网络通常更新产生 logits 的网络参数。

最后回扣阅读目的：

> GFL 让我理解了定位分布如何表示和训练。接着读 LD，就能理解它为什么蒸馏四条边的分布，而不只是比较最终框坐标。



1. 2.8 对应的 0.2、0.8 从哪里来？——距离插值，不是 IoU。
2. QFL 软标签从哪里来？——类别来自标注与分配，质量来自当前预测框的 IoU。
3. 为什么 GIoU 和 DFL 同时存在？——最终框约束与直接分布监督互补。
4. 去掉 DFL 会怎样？——若仍输出 17 个 logits，GIoU 仍能通过期望训练，QFL 也仍可计算；只是没有直接的相邻刻度分布监督，不会自动变成狄拉克表示。
