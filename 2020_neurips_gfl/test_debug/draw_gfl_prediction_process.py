"""Draw a single-image GFL prediction process for presentation."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np


OUT = Path(__file__).resolve().parent / "gfl_prediction_process.png"


def add_arrow(ax, start, end, color, text, text_offset=(0, 0), lw=2):
    ax.add_patch(FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
    ))
    mid = ((start[0] + end[0]) / 2 + text_offset[0],
           (start[1] + end[1]) / 2 + text_offset[1])
    ax.text(mid[0], mid[1], text, color=color, fontsize=10,
            ha="center", va="center", weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


def draw_image_panel(ax):
    """Draw the original-image view with a GT box, prediction, and reference point."""
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.set_aspect("equal")
    ax.set_facecolor("#e9eef2")
    ax.set_title("① 原图中的一个检测位置", fontsize=14, weight="bold", pad=12)

    # A simple road-like background keeps the diagram visual without implying real data.
    ax.fill_between([0, 640], 320, 480, color="#cbd5dc")
    ax.fill_between([0, 640], 0, 320, color="#f4f6f8")
    ax.plot([0, 640], [320, 320], color="#93a4ae", linewidth=2)
    for x in np.linspace(40, 600, 8):
        ax.plot([x, x + 65], [400, 355], color="#ffffff", linewidth=3, alpha=0.9)

    # The same reference point is used to express both the GT distances and predictions.
    gt = (244, 156, 420, 316)
    pred = (260, 164, 412, 300)
    p = (324, 220)  # (40 + 0.5, 27 + 0.5) * 8
    gt_center = ((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2)
    pred_center = ((pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2)

    # A schematic obstacle, not a detection result from a trained model.
    ax.add_patch(Rectangle((250, 165), 160, 145, facecolor="#d8c5a8",
                           edgecolor="#a79378", linewidth=1, alpha=0.45))
    ax.plot([250, 410], [165, 310], color="#a79378", alpha=0.3)
    ax.plot([250, 410], [310, 165], color="#a79378", alpha=0.3)

    ax.add_patch(Rectangle((gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1],
                           fill=False, linewidth=3, edgecolor="#137c8b",
                           label="真实框"))
    ax.add_patch(Rectangle((pred[0], pred[1]), pred[2] - pred[0], pred[3] - pred[1],
                           fill=False, linewidth=3, edgecolor="#d97706",
                           linestyle="--", label="预测框"))

    ax.scatter(*p, s=95, color="#b42318", zorder=5)
    ax.scatter(*gt_center, s=48, color="#137c8b", marker="x", zorder=5)
    ax.scatter(*pred_center, s=48, color="#d97706", marker="x", zorder=5)

    ax.text(gt[0] + 8, gt[1] - 10, "真实框 GT", color="#137c8b", fontsize=10, weight="bold")
    ax.text(pred[0] + 8, pred[3] + 20, "预测框", color="#d97706", fontsize=10, weight="bold")
    ax.annotate("参考点 P=(324,220)", p, (50, 60), fontsize=10,
                color="#b42318", arrowprops=dict(arrowstyle="->", color="#b42318"))
    ax.annotate("GT中心=(332,236)", gt_center, (440, 380), fontsize=9,
                color="#137c8b", arrowprops=dict(arrowstyle="->", color="#137c8b"))
    ax.annotate("预测框中心=(336,232)", pred_center, (440, 95), fontsize=9,
                color="#d97706", arrowprops=dict(arrowstyle="->", color="#d97706"))

    add_arrow(ax, p, (gt[0], p[1]), "#137c8b", "l=80", (0, -13), 1.7)
    add_arrow(ax, p, (p[0], gt[1]), "#137c8b", "t=64", (24, 0), 1.7)
    add_arrow(ax, p, (gt[2], p[1]), "#137c8b", "r=96", (0, -13), 1.7)
    add_arrow(ax, p, (p[0], gt[3]), "#137c8b", "b=96", (25, 0), 1.7)
    ax.text(25, 440, "真实距离 / 8 = [10,8,12,12]\n预测距离 / 8 = [8,7,11,10]", fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))

    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.set_xlabel("像素坐标 x")
    ax.set_ylabel("像素坐标 y")
    ax.grid(alpha=0.12)


def draw_feature_panel(ax):
    """Draw a local feature-map grid and identify the corresponding point."""
    ax.set_xlim(0, 8)
    ax.set_ylim(6, 0)
    ax.set_aspect("equal")
    ax.set_title("② 下采样 8 倍后的局部特征图", fontsize=14, weight="bold", pad=12)

    for x in range(9):
        ax.plot([x, x], [0, 6], color="#b7c3cc", linewidth=0.8)
    for y in range(7):
        ax.plot([0, 8], [y, y], color="#b7c3cc", linewidth=0.8)
    gx, gy = np.meshgrid(np.arange(8) + 0.5, np.arange(6) + 0.5)
    ax.scatter(gx.ravel(), gy.ravel(), s=10, color="#94a3b8", alpha=0.55)

    selected = (4, 3)
    ax.add_patch(Rectangle(selected, 1, 1, facecolor="#fee2e2", edgecolor="#b42318", linewidth=2.5))
    ax.scatter(selected[0] + 0.5, selected[1] + 0.5, s=90, color="#b42318", zorder=4)
    ax.text(selected[0] + 0.5, selected[1] + 0.5, "P", color="white", ha="center", va="center", weight="bold")

    # This is a local grid for readability; the real 640x480 image would yield 80x60 points.
    ax.annotate("参考点 (40.5,27.5)\n×8 → (324,220) 像素", xy=(4.5, 3.5), xytext=(1.0, 0.9),
                arrowprops=dict(arrowstyle="->", color="#b42318", linewidth=1.5),
                fontsize=10, color="#b42318", weight="bold")
    ax.set_xlabel("列索引（完整特征图 80×60，灰点为参考点）")
    ax.set_ylabel("特征图行索引")
    ax.set_xticks(np.arange(0.5, 8.5), labels=[str(i) for i in range(36, 44)])
    ax.set_yticks(np.arange(0.5, 6.5), labels=[str(i) for i in range(24, 30)])


def draw_flow_panel(ax):
    """Draw the output and decoding flow for the selected point."""
    ax.axis("off")
    ax.set_title("③ 这个位置如何产生一个预测框", fontsize=14, weight="bold", pad=12)

    boxes = [
        (0.06, 0.76, 0.88, 0.13, "检测头输出：4 × 17 个回归 logits\n左 / 上 / 右 / 下，每条边对应距离 0...16"),
        (0.06, 0.54, 0.88, 0.13, "每条边分别 softmax\n得到 17 个候选距离上的概率分布"),
        (0.06, 0.32, 0.88, 0.13, "Integral：Σ 概率 × [0,1,...,16]\n示例 [8,7,11,10] × 8 = [64,56,88,80] 像素"),
        (0.06, 0.10, 0.88, 0.13, "以 P=(324,220) 解码：左上减，右下加\n预测框 = [260,164,412,300]"),
    ]
    colors = ["#dbeafe", "#dcfce7", "#fef3c7", "#ffedd5"]
    for (x, y, w, h, text), color in zip(boxes, colors):
        ax.add_patch(Rectangle((x, y), w, h, transform=ax.transAxes,
                               facecolor=color, edgecolor="#64748b", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#17212b")
    for y in (0.72, 0.50, 0.28):
        ax.annotate("", xy=(0.5, y), xytext=(0.5, y + 0.035),
                    xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-|>", color="#475569", linewidth=1.8))


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(16, 9), dpi=160, facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 0.9], height_ratios=[1.0, 1.0],
                            wspace=0.25, hspace=0.34)
    draw_image_panel(fig.add_subplot(grid[:, 0]))
    draw_feature_panel(fig.add_subplot(grid[0, 1]))
    draw_flow_panel(fig.add_subplot(grid[1, 1]))
    fig.suptitle("GFL：一个特征点从参考点到预测框的完整过程", fontsize=18, weight="bold", y=0.98)
    fig.text(0.5, 0.015,
             "参考点 P 由特征网格决定，不要求是物体中心；真实框和预测框只是共用 P 来表示四条边距离。",
             ha="center", fontsize=11, color="#334155")
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
