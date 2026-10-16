"""Draw numerical teaching examples for the five GFL background papers.

These are constructed examples, not experimental results from any paper.
Run from any directory: python path/to/draw_gfl_background_examples.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


FIGURES = Path(__file__).resolve().parent / "figures"


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=160, layout="constrained")

    # The probability p_t belongs to the correct binary label, including background.
    pt = np.array([0.99, 0.1])
    ce = -np.log(pt)
    fl = (1 - pt) ** 2 * ce
    ax = axes[0, 0]
    x = np.arange(2)
    ax.bar(x - 0.16, ce, width=0.3, color="#64748b", label="交叉熵")
    ax.bar(x + 0.16, fl, width=0.3, color="#137c8b", label="Focal Loss，γ=2")
    ax.set_yscale("log")
    ax.set_ylim(1e-7, 10)
    ax.set_xticks(x, ["容易样本：p_t=0.99", "困难样本：p_t=0.10"])
    ax.set_ylabel("单个样本损失（对数坐标）")
    ax.set_title("Focal Loss：降低容易样本的损失贡献", weight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    ax.text(0.97, 0.95, "忽略 α 权重，以便只观察调制因子", transform=ax.transAxes,
            ha="right", va="top", fontsize=9)

    ax = axes[0, 1]
    coordinates = np.linspace(3, 17, 500)
    for sigma, color in [(0.5, "#137c8b"), (2, "#d97706")]:
        density = np.exp(-0.5 * ((coordinates - 10) / sigma) ** 2) / (
            np.sqrt(2 * np.pi) * sigma)
        ax.plot(coordinates, density, color=color, lw=2,
                label=f"均值 μ=10，标准差 σ={sigma}")
    ax.set_title("不确定性回归：同一均值，可以有不同方差", weight="bold")
    ax.set_xlabel("某条边的归一化坐标（示例单位）")
    ax.set_ylabel("概率密度（曲线下面积为1）")
    ax.legend()
    ax.grid(alpha=0.2)

    # PyTorch .std() uses a sample-standard-deviation correction by default.
    ious = np.arange(1, 7) / 10
    threshold = ious.mean() + ious.std(ddof=1)
    ax = axes[1, 0]
    ax.bar(np.arange(6), ious,
           color=["#137c8b" if z >= threshold else "#94a3b8" for z in ious])
    ax.axhline(threshold, color="#b42318", linestyle="--",
               label=f"均值 + 样本标准差 ≈ {threshold:.3f}")
    ax.set_xticks(np.arange(6), ["P3-a", "P3-b", "P4-a", "P4-b", "P5-a", "P5-b"])
    ax.set_ylim(0, 0.8)
    ax.set_title("ATSS：3层 × 每层2个最近候选（缩小示例）", weight="bold")
    ax.set_ylabel("候选 anchor 与 GT 的 IoU")
    ax.legend(loc="upper left")
    ax.text(0.03, 0.86, "假设六个候选的中心均在 GT 内；正式配置 topk=9",
            transform=ax.transAxes, fontsize=9, va="top")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    for shift, y, color, label in [
        (3, 3.5, "#d97706", "较近：IoU=0，GIoU=-0.2，损失=1.2"),
        (6, 0, "#d97706", "较远：IoU=0，GIoU=-0.5，损失=1.5"),
    ]:
        ax.add_patch(Rectangle((0, y), shift + 2, 2, fill=False,
                               linestyle="--", edgecolor="#64748b", lw=1.5))
        ax.add_patch(Rectangle((0, y), 2, 2, facecolor="#137c8b", alpha=0.7))
        ax.add_patch(Rectangle((shift, y), 2, 2, facecolor=color, alpha=0.7))
        ax.text(0, y + 2.25, label, fontsize=10)
    ax.text(0, 6.3, "左框为GT；右框为预测；虚线为最小外接矩形 C", fontsize=9)
    ax.set_xlim(-0.4, 8.8)
    ax.set_ylim(-0.5, 6.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("GIoU：同样不相交，也能区分包围空隙", weight="bold")

    fig.suptitle("GFL 背景论文：四个构造的小例子", fontsize=17, weight="bold")
    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "gfl_background_examples.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(output)

if __name__ == "__main__":
    main()
