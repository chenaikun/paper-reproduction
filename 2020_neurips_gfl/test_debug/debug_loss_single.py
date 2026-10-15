import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# 固定随机种子，便于汇报时重复运行并得到相同的随机预测值。
torch.manual_seed(0)

# ----------------- 1. 解决目录路径导入问题 -----------------
# 获取当前脚本所在目录 (test_debug) 的上一级目录 (2020_neurips_gfl)
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# 将根目录和 mmdetection 路径加入到 sys.path 中
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

mmdet_path = PROJECT_ROOT / "mmdetection"
if mmdet_path.exists() and str(mmdet_path) not in sys.path:
    sys.path.insert(0, str(mmdet_path))

# 验证导入 (根据你的实际类名导入，如果是官方 gfl 可直接导入)
try:
    from mmdet.models.dense_heads import GFLHead
    print("成功导入 MMDetection 模块！")
except ImportError as e:
    print(f"导入提示: {e}，将直接进行纯张量逻辑调试。")


#基础数据

N=2 #batchsize为2
classes_nums=80
H,W=4,4 
reg_max=16  #最大离散值
anchors_num=H*W  #一张特征图全部特征点
stride=(8,8) #下采样倍数


#构造head预测值
cls_score=torch.randn(N,classes_nums,H,W,requires_grad=True)

bbox_pred=torch.randn(N,4*(reg_max+1),H,W,requires_grad=True)

#构造几何基准
y_grid,x_grid=torch.meshgrid(torch.arange(H),torch.arange(W),indexing='ij')
x_centers=(x_grid.float()+0.5)*stride[0]
y_centers=(y_grid.float()+0.5)*stride[1]
points=torch.stack([x_centers.flatten(),y_centers.flatten()],dim=-1)
#.unsqueeze(0):把这叠卡片装进 1 个文件夹里 ---> 得到 [1 个文件夹, 16 张卡片, 4 个坐标]
#.repeat(2, 1, 1):按 (2, 1, 1) 放到复印机里 ---> 得到 [2 个文件夹, 16 张卡片, 4 个坐标](正好对应 Batch 中的 2 张图片，每张图片各分到一个装有 16 张卡片的文件夹)
anchors=torch.cat([points,points],dim=-1).unsqueeze(0).repeat(N,1,1)

#构造监督信息
labels=torch.full((N,anchors_num),classes_nums,dtype=torch.long)  #类别数设置为80  一般用于当作负样本或者背景


#加入几个正样本点
labels[0,5]=1   #第一张图的第6个点类别为1  
labels[1,2]=0   #第二张图的第3个点类别为0
labels[0,0]=1   #第一张图的第1个点类别为1

label_weights = torch.ones((N,anchors_num),dtype=torch.float32)

# 正样本回归目标。这里直接使用特征图坐标单位，便于和中心点相减解码。
bbox_targets=torch.zeros((N,anchors_num,4),dtype=torch.float32)
bbox_targets[0,5]=torch.tensor([3.2, 5.1, 4.0, 2.5])
bbox_targets[1,2]=torch.tensor([8.6, 2.3, 7.4, 6.1])
bbox_targets[0,0]=torch.tensor([4.5, 4.5, 3.0, 3.0])

#控制正样本数量的兜底
#防止正样本出现数据为0
avg_factor=max(int((labels<classes_nums).sum()),1)


# ----------------- 3. 用纯 PyTorch 模拟 GFL 的三个损失 -----------------
# 这些函数是论文公式和 GFLHead 中的调用顺序。



def integral_decode(bbox_logits):
    """Integral: 4 组 (reg_max + 1) 个 logits -> 4 条连续边距离。"""
    # bbox_logits: [num_positive, 4 * 17]
    distributions = bbox_logits.reshape(-1, reg_max + 1)
    probabilities = F.softmax(distributions, dim=1)
    project = torch.arange(
        reg_max + 1, device=bbox_logits.device, dtype=bbox_logits.dtype)
    distances = (probabilities * project).sum(dim=1)
    return distances.reshape(-1, 4)


def distances_to_boxes(centers, distances):
    """把 [left, top, right, bottom] 距离解码为 xyxy 框。"""
    left, top, right, bottom = distances.unbind(dim=1)
    cx, cy = centers.unbind(dim=1)
    return torch.stack([cx - left, cy - top, cx + right, cy + bottom], dim=1)


def aligned_iou(boxes1, boxes2, eps=1e-7):
    """计算一一对应的 IoU；第 i 个 boxes1 只和第 i 个 boxes2 比较。"""
    inter_left = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    inter_top = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    inter_right = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    inter_bottom = torch.minimum(boxes1[:, 3], boxes2[:, 3])
    inter_w = (inter_right - inter_left).clamp(min=0)
    inter_h = (inter_bottom - inter_top).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * \
        (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * \
        (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    return inter_area / (area1 + area2 - inter_area).clamp(min=eps)


def aligned_giou_loss(boxes1, boxes2, eps=1e-7):
    """计算一一对应的 GIoU Loss = 1 - GIoU。"""
    iou = aligned_iou(boxes1, boxes2, eps)

    enc_left = torch.minimum(boxes1[:, 0], boxes2[:, 0])
    enc_top = torch.minimum(boxes1[:, 1], boxes2[:, 1])
    enc_right = torch.maximum(boxes1[:, 2], boxes2[:, 2])
    enc_bottom = torch.maximum(boxes1[:, 3], boxes2[:, 3])
    enc_area = (enc_right - enc_left).clamp(min=0) * \
        (enc_bottom - enc_top).clamp(min=0)

    # GIoU 中的 union_area 需要交集面积；这里重新计算交集，代码更直观。
    inter_left = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    inter_top = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    inter_right = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    inter_bottom = torch.minimum(boxes1[:, 3], boxes2[:, 3])
    inter_area = (inter_right - inter_left).clamp(min=0) * \
        (inter_bottom - inter_top).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * \
        (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * \
        (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union_area = area1 + area2 - inter_area
    giou = iou - (enc_area - union_area) / enc_area.clamp(min=eps)
    return 1 - giou


def quality_focal_loss(logits, labels, quality, beta=2.0):
    """QFL 的最小实现：负样本质量为0，正样本真实类别质量为 IoU。"""
    probabilities = logits.sigmoid()
    zero_target = torch.zeros_like(logits)

    # 先把所有位置当成背景，和 mmdet 的 quality_focal_loss 第 35-40 行一致。
    loss = F.binary_cross_entropy_with_logits(
        logits, zero_target, reduction='none') * probabilities.pow(beta)

    positive = (labels >= 0) & (labels < classes_nums)
    positive_indices = positive.nonzero(as_tuple=False).squeeze(1)
    if positive_indices.numel() > 0:
        positive_labels = labels[positive_indices]
        quality_gap = quality[positive_indices] - probabilities[
            positive_indices, positive_labels]
        positive_loss = F.binary_cross_entropy_with_logits(
            logits[positive_indices, positive_labels],
            quality[positive_indices],
            reduction='none') * quality_gap.abs().pow(beta)
        loss[positive_indices, positive_labels] = positive_loss

    # label_weights 和 avg_factor 对应 GFLHead 对 QFL 的加权与归一化。
    return (loss.sum(dim=1) * label_weights.reshape(-1)).sum() / avg_factor


# ----------------- 4. 调试输出和断点 -----------------
def trace_print(enabled, title, message):
    """只在 --trace 模式打印中间量，避免普通运行时输出过多。"""
    if enabled:
        print(f"\n[{title}]\n{message}")


def pause_if_requested(enabled, title):
    """在 --pdb 模式暂停，让你检查当前作用域中的局部变量。"""
    if enabled:
        print(f"\n即将暂停：{title}")
        print("可输入 p 变量名 查看变量，输入 n 执行下一行，输入 c 继续。")
        breakpoint()


# 核心逻辑计算函数
def run_debug(trace=False, pdb_mode=False):
    print(f"数据加载完毕！\n")
    print(f"anchors:{anchors.shape},cls_score:{cls_score.shape},bbox_pred:{bbox_pred.shape}\n")
    print(f"labels:{labels.shape},bbox_targets:{bbox_targets.shape},avg_factor:{avg_factor}\n")

    trace_print(
        trace,
        "Step 0 / 输入和监督信息",
        "N=2, H=W=4, 类别数=80, reg_max=16\n"
        "每个位置：80 个分类 logits + 68 个回归 logits\n"
        "正样本数量=" + str(int(pos_count := (labels < classes_nums).sum()))
    )
    pause_if_requested(pdb_mode, "Step 0：查看原始 cls_score、bbox_pred、labels")

    # 数据展开：对应 GFLHead.loss_by_feat_single 中的 permute 和 reshape。
    cls_score_flat=cls_score.permute(0,2,3,1).reshape(-1,classes_nums)
    bbox_pred_flat=bbox_pred.permute(0,2,3,1).reshape(-1,(reg_max+1)*4)
    

    #监督数据展开
    labels_flat=labels.reshape(-1)
    bbox_targets_flat=bbox_targets.reshape(-1,4)
    anchors_flat=anchors.reshape(-1,4)

    #提取正样本
    pos_mask=(labels_flat>=0)&(labels_flat<classes_nums)
    pos_bbox_pred=bbox_pred_flat[pos_mask]
    pos_bbox_targets=bbox_targets_flat[pos_mask]
    pos_anchors=anchors_flat[pos_mask]
    pos_centers=pos_anchors[:,:2]/stride[0]

    trace_print(
        trace,
        "Step 1 / 展平并提取正样本",
        f"cls_score_flat: {tuple(cls_score_flat.shape)}\n"
        f"bbox_pred_flat: {tuple(bbox_pred_flat.shape)}\n"
        f"pos_mask 中正样本数量: {int(pos_mask.sum())}\n"
        f"pos_bbox_pred: {tuple(pos_bbox_pred.shape)}\n"
        f"pos_centers:\n{pos_centers.detach()}"
    )
    pause_if_requested(pdb_mode, "Step 1：查看展平后的预测和正样本")

    # 1) Integral：把每个正样本的 68 个回归 logits 解码成四条边距离。
    pos_bbox_pred_corners = integral_decode(pos_bbox_pred)
    print(f"\n预测四条边距离 shape: {pos_bbox_pred_corners.shape}")

    first_distribution = F.softmax(
        pos_bbox_pred[0, :reg_max + 1], dim=0).detach()
    trace_print(
        trace,
        "Step 2 / Integral",
        f"第一个正样本左边的17个概率之和: {first_distribution.sum().item():.6f}\n"
        f"第一个正样本左边概率前5项: {first_distribution[:5]}\n"
        f"预测四条边距离:\n{pos_bbox_pred_corners.detach()}"
    )
    pause_if_requested(pdb_mode, "Step 2：查看 softmax 概率和 Integral 结果")

    # 2) 根据四条边距离得到预测框；真实框也由真实距离目标解码得到。
    pos_decode_bbox_pred = distances_to_boxes(pos_centers, pos_bbox_pred_corners)
    pos_decode_bbox_targets = distances_to_boxes(pos_centers, pos_bbox_targets)

    # 3) 计算预测框和真实框的 IoU。GFL 用它作为 QFL 正样本的质量标签。
    quality = aligned_iou(
        pos_decode_bbox_pred.detach(), pos_decode_bbox_targets)
    print(f"预测框与真实框 IoU: {quality.detach()}")

    trace_print(
        trace,
        "Step 3 / 解码预测框并计算 IoU",
        f"预测框 xyxy:\n{pos_decode_bbox_pred.detach()}\n"
        f"真实框 xyxy:\n{pos_decode_bbox_targets.detach()}\n"
        f"IoU（作为 QFL 正样本质量标签）: {quality.detach()}"
    )
    pause_if_requested(pdb_mode, "Step 3：查看预测框、真实框和 IoU")

    # 分类置信度作为 GIoU/DFL 的样本权重；detach 表示它不反向影响分类分支。
    weight_targets = cls_score_flat.detach().sigmoid().max(dim=1)[0][pos_mask]

    trace_print(
        trace,
        "Step 4 / 样本权重",
        f"weight_targets（正样本最大分类置信度）: {weight_targets}\n"
        "该权重 detach，不会通过回归损失反向更新分类分支。"
    )

    # 4) GIoU Loss：比较最终预测框和真实框，属于框级回归损失。
    loss_bbox_each = aligned_giou_loss(
        pos_decode_bbox_pred, pos_decode_bbox_targets)
    # MMDetection 在这里传 avg_factor=1.0，因此是加权和，不再除以权重和。
    loss_bbox = (loss_bbox_each * weight_targets).sum()

    trace_print(
        trace,
        "Step 5 / GIoU Loss",
        f"每个正样本的 GIoU Loss: {loss_bbox_each.detach()}\n"
        f"加权后的 loss_bbox: {loss_bbox.item():.6f}"
    )
    pause_if_requested(pdb_mode, "Step 5：查看 GIoU Loss")

    # 5) DFL：监督四条边的离散概率分布。
    pred_list=pos_bbox_pred.reshape(-1,(reg_max+1))
    target_list=pos_bbox_targets.reshape(-1)

    dis_left=target_list.long()
    dis_right=dis_left+1
    weight_left=dis_right.float()-target_list
    weight_right=target_list-dis_left.float()

    loss_left=F.cross_entropy(pred_list,dis_left,reduction='none')
    loss_right=F.cross_entropy(pred_list,dis_right,reduction='none')
    # 一个正样本有四条边，因此每条边复制该样本的 weight_targets。
    edge_weights = weight_targets[:, None].expand(-1, 4).reshape(-1)
    dfl_each = loss_left * weight_left + loss_right * weight_right
    # MMDetection 的 avg_factor=4.0，对四条边的损失做平均。
    loss_dfl=(dfl_each * edge_weights).sum() / 4.0

    trace_print(
        trace,
        "Step 6 / DFL",
        f"第一个边目标距离: {target_list[0].item():.4f}\n"
        f"左右整数: {dis_left[0].item()}, {dis_right[0].item()}\n"
        f"左右权重: {weight_left[0].item():.4f}, {weight_right[0].item():.4f}\n"
        f"对应左右交叉熵: {loss_left[0].item():.6f}, {loss_right[0].item():.6f}\n"
        f"DFL: {loss_dfl.item():.6f}"
    )
    pause_if_requested(pdb_mode, "Step 6：查看 DFL 的左右插值")

    # 6) QFL：分类分支的正样本目标是 IoU，负样本目标是0。
    quality_all = torch.zeros_like(labels_flat, dtype=cls_score_flat.dtype)
    quality_all[pos_mask] = quality
    loss_cls = quality_focal_loss(cls_score_flat, labels_flat, quality_all)

    positive_indices = pos_mask.nonzero(as_tuple=False).squeeze(1)
    positive_labels = labels_flat[positive_indices]
    positive_probabilities = cls_score_flat.sigmoid()[
        positive_indices, positive_labels]
    trace_print(
        trace,
        "Step 7 / QFL",
        f"正样本类别: {positive_labels}\n"
        f"正样本类别预测概率: {positive_probabilities.detach()}\n"
        f"正样本 IoU 质量标签: {quality.detach()}\n"
        f"QFL: {loss_cls.item():.6f}"
    )
    pause_if_requested(pdb_mode, "Step 7：查看 QFL 的类别概率和 IoU 目标")

    # 配置中的三个 loss_weight：QFL=1，GIoU=2，DFL=0.25。
    total_loss = loss_cls + 2.0 * loss_bbox + 0.25 * loss_dfl

    print(f"loss_cls (QFL): {loss_cls.item():.6f}")
    print(f"loss_bbox (GIoU): {loss_bbox.item():.6f}")
    print(f"loss_dfl (DFL): {loss_dfl.item():.6f}")
    print(f"total_loss: {total_loss.item():.6f}")

    # 用于展示完整训练链路：三项损失汇总后反向传播，更新 cls_score/bbox_pred。
    optimizer = torch.optim.SGD([cls_score, bbox_pred], lr=0.01)
    old_cls_logit = cls_score.detach().flatten()[0].item()
    old_bbox_logit = bbox_pred.detach().flatten()[0].item()
    optimizer.zero_grad()
    total_loss.backward()
    print(f"cls_score 梯度均值: {cls_score.grad.abs().mean().item():.6e}")
    print(f"bbox_pred 梯度均值: {bbox_pred.grad.abs().mean().item():.6e}")
    optimizer.step()
    print(f"cls_score[0] 更新: {old_cls_logit:.6f} -> {cls_score.detach().flatten()[0].item():.6f}")
    print(f"bbox_pred[0] 更新: {old_bbox_logit:.6f} -> {bbox_pred.detach().flatten()[0].item():.6f}")
    pause_if_requested(pdb_mode, "Step 8：查看梯度和 optimizer.step() 后的参数")
    return total_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='逐步展示 GFL 的 Integral、QFL、GIoU 和 DFL 计算链路')
    parser.add_argument(
        '--trace', action='store_true',
        help='打印每个阶段的关键张量、形状和中间结果')
    parser.add_argument(
        '--pdb', dest='pdb_mode', action='store_true',
        help='在每个阶段进入 Python 调试器；输入 n 单步，c 继续，p 变量查看')
    args = parser.parse_args()
    run_debug(trace=args.trace, pdb_mode=args.pdb_mode)



