"""练习3：dim 到底在切哪一维。

跑法：
    C:\\Users\\monkey\\.conda\\envs\\DL\\python.exe 2022_cvpr_ld\\练习3_看懂dim.py

三个小实验：
    1. 二维张量里 dim=1 和 dim=0 的区别（用真实数字看）
    2. 三维张量里 dim=1 可能不是你以为的那一维（我自己踩过的坑）
    3. 官方代码里那个 (N*4, 17) 是怎么变出来的，为什么 dim=1 正好是 17
"""

import torch
import torch.nn.functional as F


def exp1_two_dim():
    print("=" * 64)
    print("实验 1：二维张量，dim=0 和 dim=1 切的是不同方向")
    print("=" * 64)

    # 3 行 4 列。可以想成：3 个样本，每个样本 4 个类别（或 4 个格子）
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                      [1.0, 2.0, 3.0, 4.0],
                      [1.0, 2.0, 3.0, 4.0]])
    print(f"x 的形状 = {tuple(x.shape)}   （行=样本，列=类别/格子）")
    print(x)
    print()

    p_dim1 = F.softmax(x, dim=1)
    print("F.softmax(x, dim=1)  —— 沿着『列』方向归一化，每一行加起来 = 1")
    print(p_dim1)
    print(f"每一行的和 = {p_dim1.sum(dim=1).tolist()}")
    print()

    p_dim0 = F.softmax(x, dim=0)
    print("F.softmax(x, dim=0)  —— 沿着『行』方向归一化，每一列加起来 = 1")
    print(p_dim0)
    print(f"每一列的和 = {p_dim0.sum(dim=0).tolist()}")
    print()

    print("看明白了吗：")
    print("  dim=1 让『同一个样本的 4 个类别』互相竞争，加起来等于 1  ← 这才是概率")
    print("  dim=0 让『同一个类别的 3 个样本』互相竞争               ← 这是跨样本比较，没意义")
    print()


def exp2_three_dim():
    print("=" * 64)
    print("实验 2：三维张量里，dim=1 可能不是你以为的那一维")
    print("=" * 64)

    # 你自己的练习 1 就是这种形状：(3 个样本, 4 条边, 5 个格子)
    x = torch.zeros(3, 4, 5)
    print(f"x 的形状 = {tuple(x.shape)}  = （3 个样本, 4 条边, 5 个格子）")
    print("  下标 0 → 样本  3 个")
    print("  下标 1 → 边    4 条")
    print("  下标 2 → 格子  5 个")
    print()

    print("如果写 dim=1：是对『4 条边』做 softmax → 每条边的 5 个格子加起来不再是 1")
    p = F.softmax(x, dim=1)
    print(f"  softmax(dim=1) 之后，格子方向的和 = {p.sum(dim=-1)[0, 0].tolist()}"
          f"   ← 不是 1，说明归一化做错地方了")
    print()

    print("如果写 dim=2 或 dim=-1：才是对『5 个格子』做 softmax")
    p = F.softmax(x, dim=-1)
    print(f"  softmax(dim=-1) 之后，格子方向的和 = {p.sum(dim=-1)[0, 0].tolist()}"
          f"   ← 是 1，对了")
    print()
    print("结论：张量几维，dim 就有几个可能值。维数越多数越容易数错。")
    print("      所以实战里几乎都写 dim=-1（永远指最后一维），除非你很清楚自己在干嘛。")
    print("      官方写 dim=1 是因为它先把张量压成了二维，见实验 3。")
    print()


def exp3_official_shape():
    print("=" * 64)
    print("实验 3：官方那个 (N*4, 17) 是怎么来的")
    print("=" * 64)

    # 假装是 LDHead 里一个批次的 bbox 输出：1 张图，2×3 的特征图，68 个通道
    n, c, h, w = 1, 68, 3, 2
    bbox_pred = torch.randn(n, c, h, w)
    print(f"① bbox head 的原始输出: {tuple(bbox_pred.shape)}  = (N, 4×(reg_max+1), H, W)")
    print(f"   68 = 4 条边 × 17 个格子（reg_max=16，格子编号 0~16）")

    bbox_pred = bbox_pred.permute(0, 2, 3, 1)
    print(f"② permute(0,2,3,1) 把通道挪到最后: {tuple(bbox_pred.shape)}  = (N, H, W, 68)")

    bbox_pred = bbox_pred.reshape(-1, c)
    print(f"③ reshape(-1, 68) 把前三维压平: {tuple(bbox_pred.shape)}  = (N×H×W, 68)")
    print(f"   此时每一行 = 一个锚点，每一行有 68 个数")

    # 取 5 个"正样本"（假装是 pos_inds）
    pos_bbox_pred = bbox_pred[[0, 1, 2, 3, 4]]
    print(f"④ 挑出正样本: {tuple(pos_bbox_pred.shape)}  = (5, 68)")

    pred_corners = pos_bbox_pred.reshape(-1, 17)
    print(f"⑤ reshape(-1, 17) 拆成每条边一行: {tuple(pred_corners.shape)}  = (5×4, 17)")
    print(f"   此时每一行 = 一个锚点的一条边，每一行 17 个数")

    p = F.softmax(pred_corners, dim=1)
    print(f"⑥ 现在 dim=1 指的就是那 17 个格子: 每行和 = {p.sum(dim=1)[:3].tolist()}")
    print()
    print("所以官方那个 dim=1 不是随便写的：")
    print("  在这个形状里，行 = 一个锚点的一条边，列 = 17 个格子")
    print("  对每条边的 17 个格子做 softmax，才符合『概率分布在格子上』的语义")
    print()
    print("如果写成 dim=0：就变成『同一个格子跨不同锚点归一化』，")
    print("  意思是『这条边在哪个锚点上更可能』——完全不是我们要的东西。")
    print()


if __name__ == "__main__":
    exp1_two_dim()
    exp2_three_dim()
    exp3_official_shape()
