"""画一张图解释：锚点、特征图、坐标系、45 个候选和阈值。

跑法：
    C:\\Users\\monkey\\.conda\\envs\\DL\\python.exe 2022_cvpr_ld\\画图_锚点与45候选.py

输出：2022_cvpr_ld/ld_core_experiment/docs/figures/anchor_explain.png
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
import torch

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'figures')
os.makedirs(OUT_DIR, exist_ok=True)

STRIDES = [8, 16, 32, 64, 128]
SIDES = [8 * s for s in STRIDES]
GT = (200., 200., 320., 320.)      # 120 × 120 的目标（输入图坐标）
IMG = 640


def iou_one(box, gt=GT):
    x1 = max(box[0], gt[0]); y1 = max(box[1], gt[1])
    x2 = min(box[2], gt[2]); y2 = min(box[3], gt[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_g = (gt[2] - gt[0]) * (gt[3] - gt[1])
    return inter / (area_a + area_g - inter)


# ============================================================
# 图 A：三个坐标系
# ============================================================
def panel_a(ax):
    ax.set_xlim(0, 10); ax.set_ylim(2.0, 10.4); ax.axis('off')
    ax.set_title('A. 一张图会经过三个坐标系', fontsize=15, fontweight='bold', pad=10)

    def box(x, y, w, h, text, color):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor='black', lw=1.4))
        ax.text(x + w / 2, y - 0.35, text, ha='center', va='top', fontsize=11)

    # 原图
    box(0.4, 7.6, 2.6, 2.0, '原图\n500 × 375（JPEG 真实尺寸）', '#e8f0fe')
    ax.add_patch(Rectangle((0.9, 8.1), 0.7, 0.55, facecolor='none', edgecolor='#1a73e8', lw=2))

    # 输入图
    box(3.9, 7.4, 2.8, 2.4, '输入图\n640 × 640（resize + pad 后）', '#fff3e0')
    ax.add_patch(Rectangle((4.55, 8.0), 1.0, 0.78, facecolor='none', edgecolor='#e8710a', lw=2))

    # 特征图
    box(7.7, 7.6, 2.0, 2.0, '特征图\n40 × 40（stride 16）', '#e6f4ea')
    for i in range(1, 5):
        ax.plot([7.7 + i * 0.4, 7.7 + i * 0.4], [7.6, 9.6], color='#137333', lw=0.8)
        ax.plot([7.7, 9.7], [7.6 + i * 0.4, 7.6 + i * 0.4], color='#137333', lw=0.8)

    ax.add_patch(FancyArrowPatch((3.05, 8.6), (3.85, 8.6), arrowstyle='-|>',
                                 mutation_scale=18, lw=1.6, color='black'))
    ax.text(3.45, 8.8, 'Resize', ha='center', fontsize=9)
    ax.add_patch(FancyArrowPatch((6.75, 8.6), (7.65, 8.6), arrowstyle='-|>',
                                 mutation_scale=18, lw=1.6, color='black'))
    ax.text(7.2, 8.8, '÷16', ha='center', fontsize=9)

    ax.text(5.0, 5.9, 'GT 框和锚点框都在「输入图」这一层比较',
            ha='center', fontsize=12.5, color='#c5221f', fontweight='bold')
    ax.text(5.0, 5.0, '· GT：数据集标注是原图坐标，pipeline 的 Resize 把它换算到输入图\n'
                      '· 锚点：特征图格子中心 × stride 换算回输入图，再画边框\n'
                      '· 两者单位都是「像素」，所以能直接算 IoU',
            ha='center', va='top', fontsize=10.5, linespacing=1.7)
    ax.text(5.0, 2.6, '如果一个是特征图坐标、一个是原图坐标，算出来的 IoU 毫无意义',
            ha='center', fontsize=10, style='italic', color='#666666')


# ============================================================
# 图 B：一个格子 → 一个锚点框
# ============================================================
def panel_b(ax):
    ax.set_xlim(145, 358); ax.set_ylim(145, 362)
    ax.set_aspect('equal')
    ax.set_title('B. 放大看：一个格子怎么变成一个 128×128 的锚点',
                 fontsize=15, fontweight='bold', pad=10)

    # 每 16 px 一条网格线（256 / 16 = 16 格）
    for k in range(0, 201, 16):
        ax.plot([150 + k, 150 + k], [150, 350], color='#dddddd', lw=0.8, zorder=0)
        ax.plot([150, 350], [150 + k, 150 + k], color='#dddddd', lw=0.8, zorder=0)

    # 格子中心点
    centers = np.arange(158, 350, 16)
    ax.scatter(np.repeat(centers, len(centers)), np.tile(centers, len(centers)),
               s=6, color='#888888', zorder=1)

    # 一个格子（说明 16×16 px）
    ax.add_patch(Rectangle((256, 256), 16, 16, facecolor='#fbbc04', alpha=0.75,
                           edgecolor='none', zorder=2))
    ax.annotate('一个格子\n16 × 16 px', xy=(272, 272), xytext=(292, 232),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    # 锚点框
    ax.add_patch(Rectangle((200, 200), 128, 128, facecolor='none',
                           edgecolor='#e8710a', lw=2.6, zorder=3))
    # GT 框
    ax.add_patch(Rectangle((200, 200), 120, 120, facecolor='#1a73e8', alpha=0.16,
                           edgecolor='#1a73e8', lw=2.2, ls='--', zorder=4))

    ax.annotate('锚点 128×128\n= 8 个格子 × 16 px', xy=(204, 326), xytext=(150, 358),
                fontsize=10.5, color='#e8710a', ha='left', va='top',
                arrowprops=dict(arrowstyle='->', color='#e8710a', lw=1.3))
    ax.annotate('GT 120×120', xy=(316, 214), xytext=(354, 176), fontsize=10.5,
                color='#1a73e8', ha='right', va='top',
                arrowprops=dict(arrowstyle='->', color='#1a73e8', lw=1.3))

    # 边长标注
    ax.annotate('', xy=(200, 190), xytext=(328, 190),
                arrowprops=dict(arrowstyle='<->', color='#e8710a', lw=1.6))
    ax.text(264, 181, '128 px', ha='center', fontsize=10.5, color='#e8710a')

    ax.text(250, 153, 'IoU = 120² / 128² = 0.879', ha='center', fontsize=13,
            fontweight='bold', color='#c5221f')
    ax.set_xlabel('x（像素，输入图坐标）', fontsize=10)
    ax.set_ylabel('y（像素，输入图坐标）', fontsize=10)


# ============================================================
# 图 C：5 层锚点对同一个目标的 IoU 上限
# ============================================================
def panel_c(ax):
    ceilings = [min(120, s) ** 2 / max(120, s) ** 2 for s in SIDES]
    colors = ['#d9d9d9'] * 5
    colors[1] = '#e8710a'
    bars = ax.bar(range(5), ceilings, color=colors, edgecolor='black', lw=0.8)
    for i, (b, c, s) in enumerate(zip(bars, ceilings, SIDES)):
        ax.text(i, c + 0.02, f'{c:.3f}', ha='center', fontsize=11,
                fontweight='bold' if i == 1 else 'normal')
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'第{i+1}层\nstride {s}\n锚点 {s*8}px' for i, s in enumerate(STRIDES)],
                       fontsize=9.5)
    ax.set_ylabel('IoU 上限', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title('C. 同一个 120×120 目标，配 5 种锚点尺寸', fontsize=15,
                 fontweight='bold', pad=10)
    ax.text(2, 0.62, '只有第 2 层尺寸对得上\n其余 4 层天生就是低分',
            fontsize=11, color='#c5221f', ha='center',
            bbox=dict(boxstyle='round', facecolor='#fef7e0', edgecolor='#e8710a'))
    ax.grid(axis='y', ls=':', alpha=0.5)


# ============================================================
# 图 D：45 个候选 + 阈值 + 正样本
# ============================================================
def panel_d(ax):
    gt_cx, gt_cy = 260., 260.
    cand_iou, cand_level = [], []
    for level, (stride, side) in enumerate(zip(STRIDES, SIDES)):
        n = IMG // stride
        ys, xs = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        cx = (xs + 0.5) * stride
        cy = (ys + 0.5) * stride
        dist = np.sqrt((cx - gt_cx) ** 2 + (cy - gt_cy) ** 2).ravel()
        idx = np.argsort(dist)[:9]
        half = side / 2
        for t in idx:
            b = (cx.ravel()[t] - half, cy.ravel()[t] - half,
                 cx.ravel()[t] + half, cy.ravel()[t] + half)
            cand_iou.append(iou_one(b))
            cand_level.append(level)

    cand_iou = np.array(cand_iou); cand_level = np.array(cand_level)
    thr = cand_iou.mean() + cand_iou.std()

    palette = ['#d9d9d9', '#e8710a', '#8bc34a', '#7e57c2', '#29b6f6']
    for level in range(5):
        m = cand_level == level
        ax.scatter(np.arange(45)[m], cand_iou[m], s=70, color=palette[level],
                   edgecolor='black', lw=0.6, zorder=3,
                   label=f'第{level+1}层（锚点 {SIDES[level]}px）')
    ax.axhline(thr, color='#c5221f', lw=2, ls='--', zorder=4)
    ax.text(0.5, thr + 0.02, f'阈值 thr = μ+σ = {thr:.3f}', color='#c5221f',
            fontsize=11.5, fontweight='bold')
    pos = cand_iou >= thr
    ax.scatter(np.arange(45)[pos], cand_iou[pos], s=220, facecolors='none',
               edgecolors='#c5221f', lw=2.4, zorder=5)
    ax.text(44, 0.62, f'圈起来这 {pos.sum()} 个 ≥ thr\n→ 成为正样本',
            ha='right', fontsize=11, color='#c5221f',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#c5221f'))
    ax.set_xlim(-1, 45)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlabel('45 个候选（每层按中心距离取最近 9 个）', fontsize=11)
    ax.set_ylabel('与 GT 的 IoU', fontsize=11)
    ax.set_title('D. 这 45 个候选一起算 μ+σ，得到阈值 thr', fontsize=15,
                 fontweight='bold', pad=10)
    ax.legend(fontsize=8.5, loc='upper left', ncol=2, framealpha=0.9)
    ax.grid(axis='y', ls=':', alpha=0.5)


fig = plt.figure(figsize=(17, 13.5), dpi=150)
gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.18,
                      left=0.05, right=0.97, top=0.94, bottom=0.06)
panel_a(fig.add_subplot(gs[0, 0]))
panel_b(fig.add_subplot(gs[0, 1]))
panel_c(fig.add_subplot(gs[1, 0]))
panel_d(fig.add_subplot(gs[1, 1]))

fig.suptitle('锚点是怎么铺的、为什么只有一个尺度能配上、45 个候选怎么定出阈值',
             fontsize=17, fontweight='bold', y=0.985)

out = os.path.join(OUT_DIR, 'anchor_explain.png')
fig.savefig(out, bbox_inches='tight', facecolor='white')
print('saved:', out)
