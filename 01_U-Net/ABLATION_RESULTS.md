# 受控消融实验结果:从 Baseline 逐步逼近 U-Net 论文原方案

## 背景

代码诊断([diagnosis.html](diagnosis.html))发现旧实验存在严重问题:
Original 架构(无 BN + SGD + CE)的两个模型都退化成了**全前景预测**——
其 67.89% 的"IoU"本质上就是标注中前景像素的占比,没有任何分割能力。
同时旧实验存在无训练/验证划分、Original 关闭了增强、多变量混杂等不公平对比问题。

为此重写了统一流水线(`dataset_unified.py` + `train_ablation.py` + `run_ablation.sh`),
**Seq01 的 9 帧训练、Seq02 的 9 帧验证**(跨序列验证,无时序泄漏),
每步只改一个变量。所有指标均为验证集上的真实泛化指标。

## 实验结果

| Step | 架构 | 优化器 | 损失 | 增强 | 最佳 val IoU | 状态 |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| 0 | Baseline (BN) | Adam 5e-4 | CE+Dice | flip/rot | **87.25%** | 正常 |
| 1 | Baseline (BN) | Adam 5e-4 | CE(去Dice) | flip/rot | **88.56%** | 正常 |
| 2 | Baseline (BN) | SGD 0.01/0.99 | CE | flip/rot | 86.61% | 正常 |
| 3 | Baseline (BN) | SGD 0.01/0.99 | Weighted CE | flip/rot | 84.41% | 正常 |
| 4 | Original (无BN) | SGD 0.01/0.99 | Weighted CE | flip/rot | 73.52%* | ⚠️ 全前景退化 |
| 5 | Original (无BN) | SGD 0.01/0.99 | Weighted CE | flip/rot+弹性变形 | 73.52%* | ⚠️ 全前景退化 |
| 6 | Original (无BN) | SGD 0.001/0.99 | Weighted CE | flip/rot | 73.52%* | ⚠️ 退化(补救:降lr) |
| 7 | Original (无BN) | SGD 0.001/0.99 | Weighted CE | flip/rot+弹性变形 | 73.52%* | ⚠️ 退化(补救:降lr) |
| 8 | Original (无BN) | SGD 0.01/**0.9** | Weighted CE | flip/rot | 73.52%* | ⚠️ 退化(补救:降动量) |
| 9 | Original (无BN) | **Adam** 5e-4 | CE | flip/rot | 74.27%* | ⚠️ 退化(补救:换Adam) |
| 10 | Original (无BN) | **Adam 1e-4** | CE | flip/rot | 73.52%* | ⚠️ 退化(补救:降lr) |

\* 73.52% 是**全前景平凡解在 Seq02 上的本底 IoU**(经实测验证),不是真实的分割能力。
退化判定:验证集预测前景比例 > 0.95。Step 4/5/9 中 59/60 个 epoch 退化,Step 6/7/10 中 58/60 个退化;
所有无 BN 实验都在第 2~4 个 epoch 内坍缩,此后从未恢复。

## 结论

1. **BatchNorm 是决定性因素,而非优化器或损失。**
   Step 9 与 Step 1 唯一区别是去掉 BN,结果从 88.56%(健康)变为坍缩;
   且无 BN 架构在 5 种优化器配置(论文原配置、降 lr、降动量、Adam、更低 Adam)下
   **全部**坍缩到全前景。在 9 帧小数据上,无 BN 的深层网络 logits 迅速饱和,
   梯度消失后锁死在退化解——BN 的逐层归一化阻止了这一坍缩。

2. **这解释了旧实验的"玄学":** 旧 Original + SGD 退化的根因不是 SGD、不是加权损失缺失,
   而是无 BN 架构本身在此数据规模下不可训练。加权损失、梯度裁剪、弹性变形都救不回来。

3. **Dice 损失并非必要:** 去掉后 IoU 反而略升(87.25% → 88.56%),CE 单独已足够。

4. **BN 存在时,SGD(momentum 0.99,论文原配置)完全可以稳定训练**(86.61%),
   与 Adam 差距很小;加权边界损失在 BN 存在时反而略降(84.41%)。

5. **弹性变形在当前规模下无可测增益**(Step 5 = Step 4,Step 7 = Step 6),
   9 帧训练数据 × 60 epoch 不足以体现其收益,且无法阻止无 BN 架构退化。

6. **论文原方案在本文条件下无法直接复现其原始性能:**
   完整复现(无 BN + SGD + 加权 CE + 弹性变形)需要论文量级的数据与 deep supervision 等配套,
   在 9 帧标注的条件下,Baseline(BN + Adam + CE)是更优且稳定的方案。

## 复现

```bash
bash run_ablation.sh        # 运行全部步骤 (0~10)
bash run_ablation.sh 0 5    # 只跑主消融链 (0~5)
bash run_ablation.sh 6 10   # 只跑补救对照 (6~10)
```

每步日志与最佳 checkpoint 保存在 `results/step*/`。
