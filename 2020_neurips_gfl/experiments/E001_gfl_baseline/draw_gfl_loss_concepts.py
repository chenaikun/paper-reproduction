"""Draw explanatory QFL and DFL plots from the equations in the GFL paper."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "figures" / "gfl_qfl_dfl_concepts.png"


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), dpi=180)
    fig.patch.set_facecolor("white")

    # QFL at a fixed quality target y=0.5, following the paper's Figure 5(a).
    ax = axes[0]
    p = np.linspace(0.005, 0.995, 500)
    y = 0.5
    bce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    for beta, color in [(1, "#167d8d"), (2, "#d97706"), (4, "#b42318")]:
        qfl = np.abs(y - p) ** beta * bce
        ax.plot(p, qfl, linewidth=2.4, label=fr"$\beta={beta}$", color=color)
    ax.axvline(y, color="#64748b", linestyle="--", linewidth=1.2)
    ax.scatter([y], [0], color="#17212b", zorder=5, s=32)
    ax.annotate("目标质量 y=0.5", (y, 0), xytext=(0.59, 0.23),
                arrowprops=dict(arrowstyle="->", color="#475569"), fontsize=10)
    ax.set_title("QFL：预测分数向质量目标靠近", fontsize=13, weight="bold")
    ax.set_xlabel("预测分数 p（sigmoid 后）")
    ax.set_ylabel("损失（BCE × |y-p|^β）")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper center")

    # Three distributions have the same expected distance 2.8, but different shapes.
    ax = axes[1]
    support = np.arange(5)
    distributions = [
        ("A：集中在 2、3", np.array([0.0, 0.0, 0.2, 0.8, 0.0]), "#167d8d"),
        ("B：远处双峰", np.array([0.3, 0.0, 0.0, 0.0, 0.7]), "#d97706"),
        ("C：邻近更分散", np.array([0.0, 0.0, 0.3, 0.6, 0.1]), "#b42318"),
    ]
    offsets = [-0.22, 0, 0.22]
    width = 0.19
    for (label, probs, color), offset in zip(distributions, offsets):
        ax.bar(support + offset, probs, width=width, color=color, label=label, alpha=0.88)
    ax.axvline(2.8, color="#334155", linestyle="--", linewidth=1.5)
    ax.text(2.86, 0.91, "共同期望 = 2.8", fontsize=10, color="#334155")
    ax.set_title("Integral 相同，不代表分布相同", fontsize=13, weight="bold")
    ax.set_xlabel("离散距离候选")
    ax.set_ylabel("概率")
    ax.set_xticks(support)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.suptitle("GFL 两个关键损失的直观解释", fontsize=16, weight="bold", y=1.02)
    fig.text(0.5, -0.02,
             "左：QFL 用连续质量标签监督分类分数。右：A/B/C 的期望都为 2.8；DFL 还会监督目标附近的概率。",
             ha="center", fontsize=10, color="#334155")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
