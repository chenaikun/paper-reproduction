"""练习2：把你写的两个练习拼成官方 `loss_ld` 的最小版。

这个文件不依赖 mmdet / mmcv，只用 torch，可以直接跑：

    python 2022_cvpr_ld/练习2_把两个练习拼成loss_ld.py

它分三步，每一步都对应你自己的一个练习：

    第 1 步  练习1（KL 蒸馏）      → 补上温度 T、T²、bin 维求平均
    第 2 步  算法1（VLR 区域）     → 改成向量化 + 自适应阈值 + 压成每锚点权重
    第 3 步  两者拼起来            → 做一个 80 个 anchor 的迷你 loss_ld

跑完你会看到官方 `LDHead.loss_single` 里那几个量分别是怎么来的。
"""

import torch
import torch.nn.functional as F


# ============================================================
# 第 0 步：先把你练习1 里的数据集原样搬过来，保证起点一致
# ============================================================
# 形状 (1, 4, 5)：1 个样本、4 条边、每条边 5 个格子，每行加起来是 1
T_P = torch.tensor([[[0.05, 0.10, 0.60, 0.20, 0.05],
                     [0.10, 0.60, 0.20, 0.05, 0.05],
                     [0.05, 0.20, 0.60, 0.10, 0.05],
                     [0.05, 0.10, 0.20, 0.60, 0.05]]], dtype=torch.float32)


def demo1_kl():
    """你的练习1：KL 蒸馏。这里只补三件事。"""
    print("=" * 60)
    print("第 1 步：练习1 的 KL，和官方公式差在哪")
    print("=" * 60)

    # 学生 logits：3 个样本，4 条边，5 个格子（和你练习里一样）
    S_l = torch.zeros(3, 4, 5, requires_grad=True)
    pos_mask = torch.tensor([True, False, True])

    # ---------- 你原来的写法 ----------
    # log_softmax 之后喂给 kl_div，这一步你写对了：
    # kl_div 的第一个参数要的是 log 概率，不是概率
    log_q = F.log_softmax(S_l, dim=-1)                 # (3, 4, 5)
    kl_each = F.kl_div(log_q, T_P, reduction="none")   # (3, 4, 5)  T_P 广播到 3 个样本
    your_kl = kl_each.sum(dim=-1).mean(dim=-1)         # (3,) 先对格子求和，再对 4 条边求平均
    your_kl = your_kl[pos_mask].mean()                 # 只算 mask 里的样本
    print(f"你的写法（格子维求和）        kl = {your_kl.item():.6f}")

    # ---------- 官方写法 ①：格子维求平均 ----------
    official_kl = kl_each.mean(dim=-1).mean(dim=-1)[pos_mask].mean()
    print(f"官方写法（格子维求平均）      kl = {official_kl.item():.6f}")
    print(f"  比值 = {(your_kl / official_kl).item():.1f}   ← 正好等于格子数 5，只差这一个常数")

    # ---------- 官方写法 ②：加温度 ----------
    # 官方的 teacher 输入是 logits，不是概率。这里用 log(T_P) 反推一组等价的 logits：
    # softmax(log(p)) == p，所以 log(T_P) 就是"能还原出 T_P 的那组 logits"
    T_l = torch.log(T_P)

    for T in (1, 2, 10):
        # 注意：这里的张量是 3 维 (样本, 4 条边, 5 个格子)，所以"格子维"是 dim=-1。
        # 官方代码里用的是 dim=1，因为它先把张量压成了 2 维 (N, 17)。两者说的是同一件事。
        target = F.softmax(T_l / T, dim=-1).detach()               # 官方：先除 T 再 softmax
        loss = F.kl_div(F.log_softmax(S_l / T, dim=-1), target,
                        reduction="none").mean(-1) * (T * T)        # 官方：格子维求平均，再乘 T²
        value = loss.mean(-1)[pos_mask].mean()
        print(f"  T={T:<2d}  kl = {value.item():.6f}   "
              f"软化后的教师分布 = {[round(x, 3) for x in target[0, 0].tolist()]}")

    print()
    print("小结：你写的 KL 方向是对的，只差三处")
    print("  ① 格子维：你求和，官方求平均（差一个 1/格子数 的常数）")
    print("  ② 温度：官方先把 logits 除以 T 再做 softmax")
    print("  ③ 缩放：官方最后乘 T*T，把梯度量级补回来")
    print()


# ============================================================
# 第 2 步：你的算法1，改成官方 get_vlr_region 的样子
# ============================================================
def pairwise_iou_diou(anchors, gts):
    """一次性算出所有 anchor × 所有 GT 的 IoU 和 DIoU。

    你的算法1 是双重 for 循环逐对调用；这里用广播一次算完。
    这是"官方代码为什么难读"的第一个原因：所有循环都被拆成了张量运算。
    """
    ax1, ay1, ax2, ay2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
    gx1, gy1, gx2, gy2 = gts[:, 0], gts[:, 1], gts[:, 2], gts[:, 3]

    # 交集：左边取大、右边取小，负数截断成 0（等价于你 is_intersect 里的 w<=0 判断）
    inter_w = (torch.min(ax2[:, None], gx2[None]) - torch.max(ax1[:, None], gx1[None])).clamp(min=0)
    inter_h = (torch.min(ay2[:, None], gy2[None]) - torch.max(ay1[:, None], gy1[None])).clamp(min=0)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1).clamp(min=0) * (ay2 - ay1).clamp(min=0)
    area_g = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    union = (area_a[:, None] + area_g[None] - inter).clamp(min=1e-7)
    iou = inter / union

    # DIoU 的两项：中心点距离 rho²、最小外接矩形对角线 c²
    rho2 = (((ax1 + ax2) / 2)[:, None] - ((gx1 + gx2) / 2)[None]) ** 2 + \
           (((ay1 + ay2) / 2)[:, None] - ((gy1 + gy2) / 2)[None]) ** 2
    cw = torch.max(ax2[:, None], gx2[None]) - torch.min(ax1[:, None], gx1[None])
    ch = torch.max(ay2[:, None], gy2[None]) - torch.min(ay1[:, None], gy1[None])
    c2 = (cw ** 2 + ch ** 2).clamp(min=1e-7)
    diou = iou - rho2 / c2
    return iou, diou


def vlr_region(anchors, gts, topk=9, a=0.25):
    """官方 get_vlr_region 的核心，去掉工程细节。

    返回：
        vlr_value: (A,) 每个 anchor 的 VLR 权重，0 表示不在 VLR 里
        mask:      (A, G) 谁落在谁的 VLR 区间里（和你算法1 输出的布尔矩阵一模一样）
        thr:       (G,) 每个 GT 自己的阈值
    """
    iou, diou = pairwise_iou_diou(anchors, gts)

    # 升级一：阈值不再固定，改成每个 GT 用「最像它的 topk 个 anchor」估一个 μ+σ
    k = min(topk, anchors.size(0))
    topk_iou, _ = iou.topk(k, dim=0)                 # (k, G)
    thr = topk_iou.mean(dim=0) + topk_iou.std(dim=0, unbiased=False)   # (G,)

    # 升级二：a = 0.25，判据用 DIoU（你的算法1 也是用 DIoU，这点你写对了）
    mask = (diou < thr[None, :]) & (diou >= a * thr[None, :])

    # 升级三：把 (A, G) 的布尔矩阵压成 (A,) 的权重
    # 规则：某个 anchor 若落在多个 GT 的 VLR 里，取它和其中 IoU 最大的那个 GT 的 IoU 当权重
    weighted = torch.where(mask, iou, torch.zeros_like(iou))
    vlr_value, _ = weighted.max(dim=1)
    return vlr_value, mask, thr


def demo2_vlr():
    print("=" * 60)
    print("第 2 步：算法1 的 VLR 区域，和官方差在哪")
    print("=" * 60)

    # ---------- 先复现你算法1 里那组数据 ----------
    anchors_small = torch.tensor([[2., 3., 4., 6.],
                                  [1., 5., 2., 8.],
                                  [1., 4., 7., 5.]])
    gts_small = torch.tensor([[2., 5., 6., 8.],
                              [1., 7., 5., 9.]])

    def your_diou(A, B):
        ax1, ay1, ax2, ay2 = A
        bx1, by1, bx2, by2 = B
        iw = min(ax2, bx2) - max(ax1, bx1)
        ih = min(ay2, by2) - max(ay1, by1)
        inter = 0.0 if (iw <= 0 or ih <= 0) else iw * ih
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        iou = inter / union
        rho2 = ((ax1 + ax2) / 2 - (bx1 + bx2) / 2) ** 2 + ((ay1 + ay2) / 2 - (by1 + by2) / 2) ** 2
        cw = max(ax2, bx2) - min(ax1, bx1)
        ch = max(ay2, by2) - min(ay1, by1)
        return iou - rho2 / (cw ** 2 + ch ** 2)

    your_mask = torch.tensor([[your_diou(A, B) < 0.5 and your_diou(A, B) >= 0.5 * 0.5
                               for B in gts_small] for A in anchors_small])
    print("你原来的数据（3 个很小的框，阈值固定 pos=0.5, a=0.5）")
    print("  布尔矩阵 =", your_mask.int().tolist())
    print("  → 全是 0。这就是你当时跑出来的 X_new")
    print("  → 不是 bug：这些框的 DIoU 太小，落不进 [0.25, 0.5] 这个窗口")
    print("  → 也正好说明为什么官方要用『自适应阈值』：固定阈值在真实数据上经常选不到东西")
    print()

    # ---------- 换一组能看出规律的数据 ----------
    # 一个 40×40 的目标，下面一排 anchor 从完全重合到很远
    gts = torch.tensor([[20., 20., 60., 60.]])
    anchors = torch.tensor([
        [20., 20., 60., 60.],     # 0  完全重合
        [24., 24., 64., 64.],     # 1  平移 4
        [28., 28., 68., 68.],     # 2  平移 8
        [32., 32., 72., 72.],     # 3  平移 12
        [36., 36., 76., 76.],     # 4  平移 16
        [40., 40., 80., 80.],     # 5  平移 20
        [48., 48., 88., 88.],     # 6  平移 28
        [80., 80., 120., 120.],   # 7  很远
    ])

    iou, diou = pairwise_iou_diou(anchors, gts)
    vlr_value, mask, thr = vlr_region(anchors, gts, topk=8, a=0.25)
    thr = thr[0]

    print(f"换一组能看出规律的数据（GT = [20,20,60,60]，自适应阈值 thr = {thr:.4f}）")
    print(f"  {'#':>2}  {'IoU':>6}  {'DIoU':>6}   {'你的固定阈值':^12}  {'官方自适应':^10}  角色")
    for k in range(anchors.size(0)):
        your = (diou[k, 0].item() < 0.5) and (diou[k, 0].item() >= 0.25)
        offi = bool(mask[k, 0])
        if iou[k, 0].item() >= thr:
            role = "正样本 → loss_ld"
        elif offi:
            role = "VLR  → loss_ld_vlr"
        else:
            role = "背景（不学）"
        print(f"  {k:>2}  {iou[k,0].item():>6.3f}  {diou[k,0].item():>6.3f}   "
              f"{'选中' if your else '不选':^12}  {'选中' if offi else '不选':^10}  {role}")

    print()
    print("看出差别了吗：")
    print("  · 你的固定窗口是 [0.25, 0.5]（DIoU），跟 GT 大小无关")
    print("  · 官方的窗口是 [0.25·thr, thr)，thr 由这个 GT 自己最像的 9 个 anchor 决定")
    print("  · 官方先用 IoU 把『正样本』挑走，VLR 只收『差一点』的那一圈")
    print("  · 你的输出是 (anchor, GT) 的矩阵；官方最后压成每个 anchor 一个 IoU 当权重")
    print("    因为 loss 要的是『每个锚点一个数』，不是一个矩阵")
    print("  · 注意 1 号锚点：IoU=0.681、阈值 0.681，差 0.0003 就掉进 VLR 了。")
    print("    真实训练里这种边界锚点很随机，不用纠结，但它说明『正样本边界 = 阈值』")
    print()


# ============================================================
# 第 3 步：拼起来，做一个迷你 loss_ld
# ============================================================
def kd_loss(pred_logits, soft_logits, T, weight=None, avg_factor=None):
    """官方 knowledge_distillation_kl_div_loss 的完整逻辑。"""
    target = F.softmax(soft_logits / T, dim=1).detach()
    loss = F.kl_div(F.log_softmax(pred_logits / T, dim=1), target,
                    reduction="none").mean(dim=1) * (T * T)
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        return loss.mean()
    return loss.sum() / avg_factor


def demo3_full():
    print("=" * 60)
    print("第 3 步：拼成一个 80 个 anchor 的迷你 loss_ld")
    print("=" * 60)

    torch.manual_seed(0)

    # 造 80 个 anchor：两层特征图（stride 8 的 8×8，stride 16 的 4×4），中心铺满 64×64 的图
    centers = []
    for stride, size in ((8, 8), (16, 4)):
        for i in range(size):
            for j in range(size):
                cx, cy = (j + 0.5) * stride, (i + 0.5) * stride
                half = 4 * stride          # GFL 的 anchor 边长 = 8 × stride
                centers.append([cx - half, cy - half, cx + half, cy + half])
    anchors = torch.tensor(centers)        # (80, 4)
    gts = torch.tensor([[10., 10., 50., 50.]])   # 一个 40×40 的目标

    # 每个 anchor 的"真值距离"（4 条边），单位是 stride；模拟教师和学生的 17 维分布
    bin_count = 17                          # reg_max=16 → 0~16 共 17 个格子
    teacher_logits = torch.randn(80, 4 * bin_count) * 2
    student_logits = torch.randn(80, 4 * bin_count) * 2

    # 让教师分布有点"真本事"：在真实距离附近放一个峰
    for k, anchor in enumerate(anchors):
        cx, cy = (anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2
        dists = [cx - 10, cy - 10, 50 - cx, 50 - cy]
        for edge, d in enumerate(dists):
            # 把像素距离粗略换算成 0~16 的格子编号（真实代码里是除以 stride）
            idx = int(max(0, min(bin_count - 1, round(float(d) / 16))))
            teacher_logits[k, edge * bin_count + idx] += 6.0

    # ---------- 正样本：IoU 超过该 GT 阈值的 anchor ----------
    iou, _ = pairwise_iou_diou(anchors, gts)
    topk_iou, _ = iou.topk(9, dim=0)
    thr = (topk_iou.mean(dim=0) + topk_iou.std(dim=0, unbiased=False))[0]
    pos_inds = (iou[:, 0] >= thr).nonzero().squeeze(1)

    # ---------- VLR：差一点的那些 ----------
    vlr_value, _, _ = vlr_region(anchors, gts, topk=9, a=0.25)
    vlr_inds = (vlr_value > 0).nonzero().squeeze(1)

    print(f"anchor 总数      : {anchors.size(0)}")
    print(f"正样本数量       : {pos_inds.numel()}   （IoU >= 阈值 {thr:.4f}）")
    print(f"VLR 锚点数量     : {vlr_inds.numel()}   （权重 = 各自被选中的最大 IoU）")
    print(f"背景数量         : {anchors.size(0) - pos_inds.numel() - vlr_inds.numel()}")

    # ---------- 三种 loss 的输入形状（和官方 loss_single 里一致）----------
    # pred_corners: (锚点数 × 4, 17)，每个锚点的 4 条边各一个 17 维分布
    pred_pos = student_logits[pos_inds].view(-1, bin_count)
    soft_pos = teacher_logits[pos_inds].view(-1, bin_count)
    pred_vlr = student_logits[vlr_inds].view(-1, bin_count)
    soft_vlr = teacher_logits[vlr_inds].view(-1, bin_count)
    print(f"\npred_corners 形状 : {tuple(pred_pos.shape)}  = （正样本数 × 4 条边, 17 个格子）")

    # ---------- 权重 ----------
    # 官方 loss_ld 用「学生自己的质量分」当权重；这里没有学生质量分，用 IoU 近似代替
    w_pos = iou[pos_inds, 0].detach()
    w_vlr = vlr_value[vlr_inds]

    # 注意 avg_factor：官方写死成 4.0 和 16.0（不是 ×正样本数），
    # 所以 loss 的量级会随正样本数量变大。这是官方的真实行为，不是笔误。
    loss_ld = kd_loss(pred_pos, soft_pos, T=10, weight=w_pos.repeat_interleave(4),
                      avg_factor=4.0)
    loss_ld_vlr = kd_loss(pred_vlr, soft_vlr, T=10, weight=w_vlr.repeat_interleave(4),
                          avg_factor=16.0)

    print(f"\nloss_ld     = {loss_ld.item():.6f}   （正样本的定位蒸馏，avg_factor=4.0）")
    print(f"loss_ld_vlr = {loss_ld_vlr.item():.6f}   （VLR 区域的定位蒸馏，avg_factor=16.0）")

    # ---------- 自检：学生和教师完全一样时，loss 应该是 0 ----------
    same = kd_loss(student_logits[pos_inds].view(-1, bin_count),
                   student_logits[pos_inds].view(-1, bin_count),
                   T=10, weight=torch.ones(pos_inds.numel() * 4),
                   avg_factor=4.0)
    print(f"\n自检：学生==教师时 loss_ld = {same.item():.2e}   ← 应该接近 0，说明公式没错")

    print()
    print("到这里，官方 loss_single 里那几个量你已经全部见过一遍了：")
    print("  pos_inds     → 正样本下标        （第 3 步）")
    print("  remain_inds  → VLR 下标          （第 2 步 + 第 3 步）")
    print("  pred_corners → (N*4, 17) 的学生分布（第 3 步）")
    print("  soft_corners → (N*4, 17) 的教师分布（第 1 步）")
    print("  weight       → IoU 或质量分      （第 2 步的压缩结果）")
    print("  avg_factor   → 归一化分母        （第 3 步）")


if __name__ == "__main__":
    demo1_kl()
    demo2_vlr()
    demo3_full()
