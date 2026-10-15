# GFL 论文汇报稿

## 论文信息

**题目：** Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection

**会议：** NeurIPS 2020

**汇报目标：** 讲清楚论文为什么提出 GFL、QFL 和 DFL 分别解决什么问题、论文公式如何对应到 MMDetection 代码，以及当前复现做到什么程度。

**当前复现状态：** 已完成 GFL 模型构建、前向传播、QFL/GIoU/DFL 损失计算、反向传播、参数更新和真实 COCO 小数据训练。尚未完成论文级别的完整 COCO 指标复现。

---

## 一、开场：论文解决什么问题

传统密集目标检测器通常有两个输出分支：

    分类分支：预测目标属于哪个类别
    回归分支：预测目标边界框的位置

传统分类分数通常只表示类别置信度，不能准确表示边界框的定位质量。例如：

    预测框 A：分类分数 0.95，IoU 0.55
    预测框 B：分类分数 0.85，IoU 0.90

如果只按照分类分数排序，定位质量更好的预测框 B 可能排在 A 后面。

GFL 的核心观点是：

> 分类分支输出的分数应该同时反映类别置信度和边界框定位质量。

传统边界框回归直接预测一个确定距离，例如 2.8，无法表达模型对这个预测的确定程度。GFL 因此使用概率分布表示边界框距离。

GFL 包含两个核心方法：

1. **Quality Focal Loss，QFL：** 让分类分数同时表达类别和定位质量；
2. **Distribution Focal Loss，DFL：** 用概率分布表示边界框的四条边距离。

整体流程：

    输入图像
      -> Backbone
      -> FPN 多尺度特征
      -> GFLHead
      -> 分类质量分数 + 边界框距离分布
      -> QFL + GIoU + DFL
      -> NMS

---

## 二、GFL 的检测头结构

GFL 是一个一阶段密集目标检测器。输入图像经过 Backbone 和 FPN 后，检测头在每个特征点输出分类和回归结果。

对于一个 FPN 特征层：

    cls_score:  (N, num_classes, H, W)
    bbox_pred:  (N, 4 * (reg_max + 1), H, W)

当前 COCO 配置：

    num_classes = 80
    reg_max = 16

因此每个特征点输出：

    80 个分类 logits
    4 * (16 + 1) = 68 个回归 logits

68 个回归 logits 的含义：

    左边距离：17 个 logits，对应距离 0 到 16
    上边距离：17 个 logits，对应距离 0 到 16
    右边距离：17 个 logits，对应距离 0 到 16
    下边距离：17 个 logits，对应距离 0 到 16

代码位置：

- GFLHead：../../mmdetection/mmdet/models/dense_heads/gfl_head.py
- Integral：../../mmdetection/mmdet/models/dense_heads/gfl_head.py#L23

---

## 三、边界框的距离表示

对于特征点中心 (x, y) 和真实框 (x1, y1, x2, y2)：

    l = x - x1
    t = y - y1
    r = x2 - x
    b = y2 - y

代码返回顺序：

    [left, top, right, bottom]

代码位置：

- bbox2distance：../../mmdetection/mmdet/structures/bbox/transforms.py#L206
- DistancePointBBoxCoder：../../mmdetection/mmdet/models/task_modules/coders/distance_point_bbox_coder.py

一个正样本对应一个边界框，因此每个正样本对应四个 DFL 回归目标。

---

## 四、Integral：从概率分布得到连续距离

GFL 不直接预测 2.8，而是预测距离 0 到 16 的概率分布：

    P(0), P(1), P(2), ..., P(16)

如果：

    P(2) = 0.2
    P(3) = 0.8

那么：

    y_hat = 2 * 0.2 + 3 * 0.8 = 2.8

一般形式：

    y_hat = sum from i=0 to n of P(y_i) * y_i

代码实现：

    x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
    x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)

解释：

    reshape：把每条边的17个 logits 分开
    softmax：得到距离概率分布
    project：保存 [0, 1, ..., 16]
    F.linear：完成概率和离散距离的加权求和
    reshape：得到四条边的连续距离

代码位置：../../mmdetection/mmdet/models/dense_heads/gfl_head.py#L42

概率集中表示模型比较确定，概率分散表示模型对边界位置不确定。

---

## 五、Distribution Focal Loss

假设真实距离为 y = 2.8，它位于两个相邻整数之间：

    y_left = 2
    y_right = 3

左右权重：

    weight_left = 3 - 2.8 = 0.2
    weight_right = 2.8 - 2 = 0.8

DFL：

    L_DFL = -0.2 log P(2) - 0.8 log P(3)

代码位置：../../mmdetection/mmdet/models/losses/gfocal_loss.py#L143

    dis_left = label.long()
    dis_right = dis_left + 1

    weight_left = dis_right.float() - label
    weight_right = label - dis_left.float()

    loss = F.cross_entropy(pred, dis_left, reduction='none') * weight_left \
         + F.cross_entropy(pred, dis_right, reduction='none') * weight_right

这里的 pred 是 logits，不是 softmax 后的概率。F.cross_entropy 内部会完成 log_softmax 和交叉熵计算。

训练流程：

    logits
      -> CrossEntropy
      -> DFL loss
      -> backward()
      -> logits 产生梯度
      -> 网络参数更新
      -> 下一次前向传播产生新的 logits

optimizer.step() 更新网络参数，不会直接修改已经计算出来的旧 pred。

---

## 六、Quality Focal Loss

QFL 的正负样本目标不同。

负样本没有目标，质量目标为 0，所有类别都被监督为 0。

正样本的真实类别位置使用预测框和真实框之间的 IoU 作为质量目标。

例如：

    真实类别：person
    预测框与真实框 IoU：0.72

person 类别的目标不是 1，而是 0.72；其他类别目标仍然为 0。

QFL 形式：

    L_QFL = -|y - p|^beta [y log p + (1-y) log(1-p)]

其中：

    p：分类分支经过 sigmoid 后的预测概率
    y：类别和定位质量的联合目标
    beta：调制因子，当前配置为 2

代码位置：../../mmdetection/mmdet/models/losses/gfocal_loss.py#L12

    pred_sigmoid = pred.sigmoid()
    zerolabel = pred.new_zeros(pred.shape)

    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none'
    ) * pred_sigmoid.pow(beta)

正样本真实类别位置再替换为 IoU 质量目标：

    loss[pos, pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos, pos_label], score[pos], reduction='none'
    ) * scale_factor.abs().pow(beta)

---

## 七、QFL 中的 IoU 如何得到

在 GFLHead.loss_by_feat_single 中：

    pos_bbox_pred_corners = self.integral(pos_bbox_pred)

四条边距离解码成预测框：

    pos_decode_bbox_pred = self.bbox_coder.decode(
        pos_anchor_centers,
        pos_bbox_pred_corners
    )

计算预测框和真实框 IoU：

    score[pos_inds] = bbox_overlaps(
        pos_decode_bbox_pred.detach(),
        pos_decode_bbox_targets,
        is_aligned=True
    )

score 就是 QFL 正样本的质量标签。

代码位置：../../mmdetection/mmdet/models/dense_heads/gfl_head.py#L280

---

## 八、三种损失的关系

当前配置：

    loss_cls = QualityFocalLoss，权重 1.0
    loss_bbox = GIoULoss，权重 2.0
    loss_dfl = DistributionFocalLoss，权重 0.25

三者作用不同：

    QFL：监督类别和定位质量的联合分数
    GIoU：监督最终预测框和真实框的重叠关系
    DFL：监督四条边距离的概率分布

因此 GIoU 和 DFL 不重复：

    GIoU 关注最终框
    DFL 关注四条边的分布

---

## 九、weight_targets 的作用

代码中：

    weight_targets = cls_score.detach().sigmoid()
    weight_targets = weight_targets.max(dim=1)[0][pos_inds]

它是当前正样本分类分支的最大类别置信度，用于给 GIoU 和 DFL 加权。

需要区分：

    score：
    预测框与真实框的 IoU，用作 QFL 正样本目标

    weight_targets：
    当前分类预测置信度，用于给回归损失加权

DFL 对四条边复制同一个样本权重：

    weight=weight_targets[:, None].expand(-1, 4).reshape(-1)

因为当前实现估计的是整个正样本框的可靠性，而不是分别估计四条边的可靠性。

---

## 十、多尺度 FPN 损失汇总

GFL 使用多个 FPN 尺度：

    高分辨率特征图：适合小目标
    低分辨率特征图：适合大目标

每个尺度调用：

    loss_by_feat_single(...)

每层返回：

    loss_cls
    loss_bbox
    loss_dfl

loss_by_feat 再把所有层组织起来并返回。因此最终日志中的三个损失是多尺度检测头损失。

---

## 十一、推理阶段

训练时，回归分布用于计算 DFL；推理时，分布经过 Integral 转换为四条连续距离：

    bbox_pred = self.integral(bbox_pred) * stride[0]

分类分支经过 sigmoid：

    scores = cls_score.sigmoid()

之后经过：

    置信度筛选
    -> Top-K 筛选
    -> NMS
    -> 最终检测框

代码位置：../../mmdetection/mmdet/models/dense_heads/gfl_head.py#L447

---

## 十二、当前复现实验

当前没有完成论文级别的完整 COCO 复现。完整复现还需要：

    完整 COCO train2017
    完整 COCO val2017
    论文对应的训练周期
    论文对应的代码版本
    一致的数据预处理和超参数
    完整 COCO mAP 对比

当前完成的是真实数据上的训练链路验证。

小型数据集：

    训练图片：16张
    验证图片：4张
    训练标注：112个
    验证标注：31个

实验配置：

- 最小 COCO 配置：../../mmdetection/configs/gfl/gfl_r50_fpn_1x_coco_mini.py
- 最小数据构建脚本：make_mini_coco.py

训练过程：

    COCO 标注读取
    -> 图像加载
    -> Resize
    -> FPN 特征提取
    -> ATSS 正负样本分配
    -> QFL
    -> GIoU Loss
    -> DFL
    -> backward()
    -> optimizer.step()
    -> checkpoint 保存

日志示例：

    第1轮：
    loss      = 2.0774
    loss_cls  = 0.1211
    loss_bbox = 1.3271
    loss_dfl  = 0.6293

    第16轮：
    loss      = 1.8025
    loss_cls  = 0.7065
    loss_bbox = 0.7108
    loss_dfl  = 0.3852

4 张验证图上的 mAP 为 0。原因包括：

    训练集只有16张图片
    验证集只有4张图片
    COCO 仍然包含80个类别
    训练周期很短
    backbone 基本被冻结

因此当前实验只能称为 GFL 代码和训练链路的最小真实数据验证，不能称为论文完整 COCO 结果复现。

---

## 十三、论文内容和代码对应表

| 论文内容 | 代码位置 |
|---|---|
| GFL 检测头 | mmdet/models/dense_heads/gfl_head.py |
| Integral 解码 | gfl_head.py 中 Integral |
| QFL | mmdet/models/losses/gfocal_loss.py |
| DFL | gfocal_loss.py 中 distribution_focal_loss |
| 四条边距离编码 | mmdet/structures/bbox/transforms.py |
| 边界框解码 | distance_point_bbox_coder.py |
| 多尺度损失 | GFLHead.loss_by_feat |
| 推理解码 | GFLHead.predict_by_feat |
| COCO 配置 | configs/gfl/gfl_r50_fpn_1x_coco.py |
| 最小实验配置 | configs/gfl/gfl_r50_fpn_1x_coco_mini.py |

---

## 十四、40 分钟时间安排

| 时间 | 内容 |
|---|---|
| 0-3 分钟 | 论文题目、研究问题和汇报目标 |
| 3-8 分钟 | 分类分数和定位质量不一致 |
| 8-12 分钟 | GFL 总体结构和检测头输出 |
| 12-17 分钟 | 边界框距离分布和 Integral |
| 17-24 分钟 | DFL 公式、y=2.8 示例和代码 |
| 24-30 分钟 | QFL 公式、IoU 质量标签和代码 |
| 30-34 分钟 | GIoU、DFL、QFL 三种损失的关系 |
| 34-37 分钟 | loss_by_feat_single 到多尺度汇总 |
| 37-39 分钟 | 最小真实 COCO 训练实验 |
| 39-40 分钟 | 复现边界、总结和下一步工作 |

建议展示论文中的三类图：

1. 传统分类分数与定位质量不一致的示意图；
2. GFL 检测头和 QFL/DFL 总体结构图；
3. 边界框概率分布通过积分得到连续距离的示意图。

每张图都回答三个问题：

    这张图说明什么问题？
    论文提出了什么解决方法？
    这个方法在代码中在哪里实现？

---

## 十五、导师可能追问的问题

### Q1：QFL 的正样本标签是 1 还是 IoU？

是 IoU，不是 1。预测框和真实框 IoU 为 0.7 时，真实类别位置的质量目标就是 0.7。

### Q2：QFL 和普通 Focal Loss 的区别？

普通 Focal Loss 的正样本目标通常是 1；QFL 的正样本目标是 IoU，因此分类分数同时学习类别和定位质量。

### Q3：DFL 的输入是概率还是 logits？

代码输入是 logits。F.cross_entropy 内部完成 log_softmax 和交叉熵计算。

### Q4：一个边界框有几个 DFL 目标？

四个，分别对应左、上、右、下四条边。

### Q5：为什么回归输出是 68 个通道？

    4条边 × (reg_max + 1)
    = 4 × 17
    = 68

### Q6：GIoU 和 DFL 是否重复？

不重复。GIoU 约束最终预测框，DFL 约束四条边距离的概率分布。

### Q7：当前实验算不算复现成功？

准确说法是：

> 完成了 GFL 训练链路和核心损失的代码级复现，尚未完成论文实验结果级复现。

### Q8：为什么小数据 mAP 是 0？

训练集和验证集过小，类别数仍为 80，训练周期有限，并且 backbone 基本被冻结。该实验目标是验证代码链路，而不是得到有意义的 COCO 指标。

---

## 十六、结论

GFL 不是简单更换一个损失函数，而是重新定义了目标检测中分类和回归的表示方式：

    QFL：学习质量感知的分类分数
    DFL：学习分布式的边界框位置
    GIoU：约束最终预测框
    FPN：处理不同尺度目标

当前已经完成：

    GFL 模型构建
    前向输出检查
    QFL、GIoU、DFL 损失计算
    反向传播
    参数更新
    真实 COCO 小数据训练
    checkpoint 保存

后续工作是扩大数据规模，核对论文对应代码版本、训练策略和超参数，并进行完整 COCO 指标复现。

