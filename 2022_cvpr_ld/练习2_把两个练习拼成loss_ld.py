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


def build_scene(gts=None):
    """造一个统一的场景：80 个 anchor + 一个 GT。"""
    centers = []
    for stride, size in ((8, 8), (16, 4)):
        for i in range(size):
            for j in range(size):
                cx, cy = (j + 0.5) * stride, (i + 0.5) * stride
                half = 4 * stride          # GFL 的 anchor 边长 = 8 × stride
                centers.append([cx - half, cy - half, cx + half, cy + half])
    if gts is None:
        gts = torch.tensor([[10., 10., 50., 50.]])
    return torch.tensor(centers), gts


def demo3_full():
    print("=" * 60)
    print("第 3 步：拼成一个 80 个 anchor 的迷你 loss_ld")
    print("=" * 60)

    torch.manual_seed(0)

    anchors, gts = build_scene()          # (80, 4) 和 (1, 4)

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


def demo4_pos_vs_vlr():
    """正样本 和 VLR 到底是不是同一批锚点？"""
    print("=" * 60)
    print("第 4 步：正样本 vs VLR，到底是不是一回事")
    print("=" * 60)

    torch.manual_seed(0)
    # 玩具场景只有 2 个尺度，真实模型有 5 个。这里用 IoU ≥ 阈值 近似 ATSS 的正样本
    # （真实 ATSS 还多一个条件：锚点中心必须落在 GT 内部）
    anchors, gts = build_scene(torch.tensor([[13., 17., 47., 49.]]))
    iou, diou = pairwise_iou_diou(anchors, gts)

    topk_iou, _ = iou.topk(9, dim=0)
    thr = (topk_iou.mean(dim=0) + topk_iou.std(dim=0, unbiased=False))[0]

    pos_mask = iou[:, 0] >= thr
    vlr_val, _, _ = vlr_region(anchors, gts, topk=9, a=0.25)
    vlr_mask = vlr_val > 0

    pos_set = set(pos_mask.nonzero().squeeze(1).tolist())
    vlr_set = set(vlr_mask.nonzero().squeeze(1).tolist())
    both = pos_set & vlr_set

    print(f"阈值 thr          : {thr:.4f}")
    print(f"正样本 pos_inds   : {len(pos_set):>3} 个   （判据：IoU ≥ thr；真实 ATSS 还要求中心在 GT 内）")
    print(f"VLR  remain_inds  : {len(vlr_set):>3} 个   （判据：0.25·thr ≤ DIoU < thr）")
    print(f"  两个都是        : {len(both):>3} 个")
    print(f"  只有 VLR 有      : {len(vlr_set - pos_set):>3} 个   ← 这些根本不是正样本")
    print(f"  只有正样本有      : {len(pos_set - vlr_set):>3} 个")
    print(f"  都是背景        : {anchors.size(0) - len(pos_set | vlr_set):>3} 个")
    print()

    print("前 6 个正样本（IoU 最高的那批）：")
    print(f"  {'#':>3}  {'IoU':>6}  {'DIoU':>6}   在 VLR 里吗")
    for k in sorted(pos_set, key=lambda i: -iou[i, 0].item())[:6]:
        print(f"  {k:>3}  {iou[k,0].item():>6.3f}  {diou[k,0].item():>6.3f}   "
              f"{'是（重叠，官方不互斥）' if k in vlr_set else '否'}")

    print()
    print("前 6 个『只在 VLR 里』的锚点（它们不是正样本）：")
    print(f"  {'#':>3}  {'IoU':>6}  {'DIoU':>6}   角色")
    only_vlr = sorted(vlr_set - pos_set, key=lambda i: -iou[i, 0].item())
    for k in only_vlr[:6]:
        print(f"  {k:>3}  {iou[k,0].item():>6.3f}  {diou[k,0].item():>6.3f}   "
              f"VLR → 只走 loss_ld_vlr")

    # ---------- 为什么正样本也会落进 VLR ----------
    print()
    print("为什么『正样本』也能落进 VLR：因为两个门槛比的不是同一个数")
    print("  正样本的判据：IoU  ≥ thr")
    print("  VLR 的判据  ：DIoU < thr")
    print("  而 DIoU = IoU − ρ²/c²  ≤  IoU   （ρ 是中心距离，c 是最小外接矩形的对角线）")
    print()
    print("  所以只要『中心偏移的惩罚』大于『IoU 高出阈值的余量』，")
    print("  一个锚点就会同时满足 IoU ≥ thr 和 DIoU < thr。")
    print()
    print(f"  {'#':>3}  {'IoU':>6}  {'DIoU':>6}  {'thr':>6}  {'余量':>8}  {'惩罚':>8}  结果")
    overlap = sorted(both, key=lambda i: -iou[i, 0].item())[:6]
    rows = []
    for k in overlap:
        margin = iou[k, 0].item() - thr.item()          # IoU 高出阈值多少
        penalty = iou[k, 0].item() - diou[k, 0].item()  # 中心偏移扣掉多少
        rows.append((k, margin, penalty))
        verdict = "惩罚 > 余量 → 掉进 VLR" if penalty > margin else "惩罚 < 余量 → 不进 VLR"
        print(f"  {k:>3}  {iou[k,0].item():>6.4f}  {diou[k,0].item():>6.4f}  "
              f"{thr.item():>6.4f}  {margin:>8.4f}  {penalty:>8.4f}  {verdict}")
    print()
    if rows:
        k0, m0, p0 = rows[0]
        print(f"  看第 1 行（锚点 {k0}）：IoU 只比阈值高 {m0:.6f}（几乎就是刚好过线），"
              f"但中心偏移扣掉了 {p0:.4f}，")
        print(f"  所以 DIoU 从 {thr.item() + m0:.4f} 掉到 {thr.item() + m0 - p0:.4f}，落在窗口里 ——")
        print("  它既是正样本，又在 VLR 里。")
    print("  真实训练里尺度有 5 个、IoU 拉得开，正样本的『余量』通常远大于『惩罚』，")
    print("  所以重叠少；但边界附近的锚点仍然可能两种都算。")

    print()
    print("把这两个概念纠正一下：")
    print("  ① 正样本 ≠『所有和 GT 相关的锚点』")
    print("     正样本是 ATSS 判出来的：中心在 GT 内 + IoU 超过自适应阈值。")
    print("     一张图通常只有几十个。而 VLR 锚点也和 GT 相关，但它们是另一批。")
    print()
    print("  ② VLR ≠『正样本的一部分』——反了，也错了")
    print(f"     这个例子里 {len(vlr_set)} 个 VLR 锚点中有 {len(vlr_set - pos_set)} 个完全不是正样本。")
    print("     判据也不一样：正样本看 IoU，VLR 看 DIoU 且要求『低于阈值』。")
    print("     VLR 的数量一般是正样本的好几倍，所以它才是定位蒸馏的主力。")
    print()
    print("  ③ 两者可以重叠，官方没有互斥")
    print("     pos_inds 和 remain_inds 是各自独立算出来的，没有『减去正样本』这一步。")
    print("     这个玩具场景里正样本恰好都落进了 VLR 窗口（阈值被算得偏高），")
    print("     真实 ATSS 里正样本 IoU 通常明显高于阈值，重叠会少一些，但确实可能发生。")
    print()
    print("  ④ 各自喂给不同的 loss：")
    print("     pos_inds    → loss_cls / loss_bbox / loss_dfl / loss_ld / loss_kd")
    print("     remain_inds → loss_ld_vlr")
    print("     另外：remain_inds 这个名字里的 remain 不是『正样本剩下的』，别被名字骗了。")
    print()


def demo5_anchor_scale():
    """锚点尺寸和目的大小怎么配对，以及为什么 IoU 会拉得很开。"""
    print("=" * 60)
    print("第 5 步：锚点尺寸和目标大小，是怎么配对的")
    print("=" * 60)

    # GFL 的真实设置：octave_base_scale=8，scales_per_octave=1，strides=[8,16,32,64,128]
    # 每个格子放一个正方形锚点，边长 = 8 × stride
    strides = [8, 16, 32, 64, 128]
    anchor_sides = [8 * s for s in strides]
    print(f"GFL 的 5 层锚点边长 = {anchor_sides} 像素")
    print(f"（配置里 octave_base_scale=8，所以低层锚点小、高层锚点大）")
    print()

    # IoU 的上限只由面积决定：
    #   · 锚点比目标小（完全被目标包住）→ IoU 最大 = 锚点面积 / 目标面积
    #   · 锚点比目标大（完全包住目标）  → IoU 最大 = 目标面积 / 锚点面积
    # 注意这是"上限"，实际因为中心位置、长宽比不同，只会更低
    gt_sides = [16, 32, 64, 128, 256, 512]

    header = "目标边长 ↓ | 锚点 →" + "".join(f"{a:>9}" for a in anchor_sides)
    print(header)
    print("-" * len(header))
    for gt in gt_sides:
        row = f"{gt:>9}    |         "
        for a in anchor_sides:
            area_gt, area_a = gt * gt, a * a
            ceiling = min(area_gt, area_a) / max(area_gt, area_a)
            row += f"{ceiling:>9.3f}"
        print(row)

    print()
    print("读法：每一格是这个目标配这个锚点时，IoU 能达到的『上限』")
    print("  · 对角线附近（目标边长≈锚点边长）→ IoU 上限接近 1")
    print("  · 左下角（大目标配小锚点）      → IoU 上限趋近 0")
    print("  · 右上角（小目标配大锚点）      → IoU 上限也趋近 0")
    print()

    print("举两个具体的：")
    print(f"  · 512×512 的大目标，配上 64×64 的小锚点：")
    print(f"    IoU 最多 = 64²/512² = {64*64/(512*512):.4f}（锚点整个被目标包住，也只能到 0.016）")
    print(f"  · 16×16 的小目标，配上 1024×1024 的大锚点：")
    print(f"    IoU 最多 = 16²/1024² = {16*16/(1024*1024):.6f}（几乎就是 0）")
    print()

    print("所以真实模型里，同一个 GT 的 5 层候选锚点：")
    print("  · 1 层尺寸正好匹配 → IoU 能到 0.5~0.9")
    print("  · 其余 4 层尺寸差太远 → IoU 只有 0.0x，甚至 0.00x")
    print()
    print("这就是为什么 thr = 均值 + 标准差 会明显低于最好的那几个：")
    print("  均值被那 4 层『低分但不算太低』的候选拉下来了，")
    print("  于是尺度匹配的那几个锚点『余量』很大。")
    print()


def demo6_atss_threshold():
    """按 ATSS 的真实流程走一遍：候选怎么选、阈值怎么算、正样本怎么定。"""
    print("=" * 60)
    print("第 6 步：ATSS 的 45 个候选和自适应阈值")
    print("=" * 60)

    img = 640
    gt = torch.tensor([[200., 200., 320., 320.]])   # 一个 120×120 的目标
    strides = [8, 16, 32, 64, 128]
    topk = 9

    # ---------- 造 5 层锚点 ----------
    anchors_per_level, sizes = [], []
    for stride in strides:
        n = img // stride
        ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n), indexing='ij')
        cx = (xs + 0.5) * stride
        cy = (ys + 0.5) * stride
        half = 4 * stride                     # 锚点边长 = 8 × stride
        boxes = torch.stack([cx - half, cy - half, cx + half, cy + half],
                            dim=-1).reshape(-1, 4)
        anchors_per_level.append(boxes)
        sizes.append(boxes.size(0))
    anchors = torch.cat(anchors_per_level)    # 全部锚点
    print(f"5 层锚点总数 = {anchors.size(0)} 个（每层 {sizes}）")
    print()

    # ---------- 第 0 步：先说清坐标系 ----------
    print("第 0 步：锚点和 GT 在同一个坐标系 —— 输入图的像素坐标")
    stride2, boxes2 = 16, anchors_per_level[1]        # 第 2 层：40×40 特征图，锚点 128×128
    n2 = img // stride2
    print(f"  第 2 层的特征图是 {n2}×{n2}（640/16），每个格子对应输入图 {stride2}×{stride2} 像素")
    print(f"  锚点边长 128 = 8 个格子的宽度（8 × stride），不是特征图上的 128 格")
    print()
    print(f"  {'格子(行,列)':>12}  {'中心(输入图)':>14}  {'锚点框(输入图)':>28}  {'IoU':>6}")
    for i, j in [(15, 15), (16, 16), (11, 11)]:
        cx, cy = (j + 0.5) * stride2, (i + 0.5) * stride2
        half = 4 * stride2
        box = torch.tensor([[cx - half, cy - half, cx + half, cy + half]])
        iou_ij, _ = pairwise_iou_diou(box, gt)
        print(f"  ({i:>2},{j:>2}){'':>4}  ({cx:>5.0f},{cy:>5.0f}){'':>3}  "
              f"[{cx-half:>5.0f},{cy-half:>5.0f},{cx+half:>5.0f},{cy+half:>5.0f}]{'':>4}  "
              f"{iou_ij[0,0].item():>6.3f}")
    print()
    print(f"  GT 框自己也是这个坐标系：{gt[0].tolist()}")
    print(f"  所以它们能直接算 IoU —— 如果一个是特征图坐标、一个是原图坐标，算出来毫无意义")
    print()

    # ---------- 第 1 步：每个 GT 在每层挑"中心最近"的 9 个当候选 ----------
    gt_cx, gt_cy = (gt[0, 0] + gt[0, 2]) / 2, (gt[0, 1] + gt[0, 3]) / 2
    cand_idx, start = [], 0
    print("第 1 步：每层按『中心距离』挑最近的 9 个（注意：按距离，不是按 IoU）")
    print(f"  {'层':>2}  {'stride':>6}  {'锚点边长':>8}  {'候选数':>6}  候选的 IoU 范围")
    for level, (stride, boxes) in enumerate(zip(strides, anchors_per_level)):
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        dist = ((cx - gt_cx) ** 2 + (cy - gt_cy) ** 2).sqrt()
        k = min(topk, boxes.size(0))
        idx = dist.topk(k, largest=False).indices + start          # 取距离最小的
        cand_idx.append(idx)
        iou_lv, _ = pairwise_iou_diou(anchors[idx], gt)
        print(f"  {level+1:>2}  {stride:>6}  {8*stride:>8}  {k:>6}  "
              f"{iou_lv.min().item():.3f} ~ {iou_lv.max().item():.3f}")
        start += boxes.size(0)

    cand_idx = torch.cat(cand_idx)            # 5 层 × 9 = 45
    print(f"  候选总数 = {cand_idx.numel()}  （5 层 × 9 个，这就是那个『45』）")
    print()

    # ---------- 第 2 步：算这 45 个候选的 IoU，求 μ+σ ----------
    iou_cand, _ = pairwise_iou_diou(anchors[cand_idx], gt)
    iou_cand = iou_cand[:, 0]
    mu, sigma = iou_cand.mean(), iou_cand.std(unbiased=False)
    thr = mu + sigma
    print("第 2 步：只对这 45 个候选算阈值（不是对全图 8525 个）")
    print(f"  45 个候选的 IoU: 均值 μ = {mu:.4f}, 标准差 σ = {sigma:.4f}")
    print(f"  阈值 thr = μ + σ = {thr:.4f}")
    print(f"  （对照：最好的候选 IoU = {iou_cand.max():.4f}，余量 = "
          f"{(iou_cand.max() - thr).item():.4f}）")
    print()

    # ---------- 第 3 步：候选里 IoU ≥ thr 且中心在 GT 内 → 正样本 ----------
    cand_boxes = anchors[cand_idx]
    cx = (cand_boxes[:, 0] + cand_boxes[:, 2]) / 2
    cy = (cand_boxes[:, 1] + cand_boxes[:, 3]) / 2
    in_gt = (cx > gt[0, 0]) & (cx < gt[0, 2]) & (cy > gt[0, 1]) & (cy < gt[0, 3])
    pos = (iou_cand >= thr) & in_gt
    print("第 3 步：候选里同时满足『IoU ≥ thr』和『中心在 GT 内』的，才是正样本")
    print(f"  正样本数量 = {pos.sum().item()} 个（只可能从这 45 个里出，不会更多）")
    print(f"  它们来自第几层: ", end='')
    layer_of = torch.cat([torch.full((min(topk, s),), i + 1) for i, s in enumerate(sizes)])
    print(sorted(set(layer_of[pos].tolist())))
    print()

    print("三个容易记错的点：")
    print("  ① 候选是按『中心距离』挑的，不是按 IoU 挑 —— 先把离得近的凑一桌，再看谁 IoU 高")
    print("  ② 阈值只用这 45 个算，但 VLR 的判定要用它去筛『全图 8525 个锚点』")
    print("     所以 VLR 能有几百个，正样本却只有几个")
    print("  ③ 每个 GT 各算各的阈值。一张图有 3 个 GT，就有 3 个 thr，互不混合")
    print()


if __name__ == "__main__":
    demo1_kl()
    demo2_vlr()
    demo3_full()
    demo4_pos_vs_vlr()
    demo5_anchor_scale()
    demo6_atss_threshold()
