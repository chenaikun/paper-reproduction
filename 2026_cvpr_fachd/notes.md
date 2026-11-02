# FaCHD 阅读笔记

> Yin et al., *Incremental Object Detection via Future-Aware Decoupled Cross-Head Distillation*, CVPR 2026 (pp. 39935–39944). 西电 De Cheng 组 + 西工大。
> 本文 PDF 文本已抽取到 `paper_text.txt`（10 页，46,894 字符），页码标记与原文一致。

---

## 0. 一句话

在 Faster R-CNN 的 IOD 里，把**学生 backbone 的 ROI 特征**分别喂给**旧教师和"未来/中间教师"的分类头**，用这两个跨头预测去对齐教师输出做蒸馏，于是蒸馏梯度只流进 backbone、不流进学生自己的分类头（解耦），再用**多粒度原型漂移补偿**重训分类头。

模块两个：
- **FaCHD**（Future-aware Decoupled Cross-Head Distillation）—— 管 backbone 的几何一致性；
- **RPSC**（Region Prototype Semantic Correction，正文 3.4 节标题又叫 ...Calibration，两处名字不统一）—— 管分类头决策边界重置。

---

## 1. 它要解决什么

IOD 的两个先天困难（Intro 明确列了，写得比多数 IOD 论文清楚）：

1. 旧类目标在当前阶段**没有标注** → 容易被学成 background；
2. 当前阶段 background 里可能**混着未来类**的目标 → 容易被当成 foreground。

这两点导致跨任务干扰。但作者认为现有 KD-based IOD 忽略了一个更具体的问题：

> **在检测头上，"新类监督"和"旧类蒸馏"是竞争梯度。**

学生分类头同时被两股力拉：GT 说"这是新类"，教师说"这是旧类/背景"。并且头与 backbone 紧耦合 → 头里的"偏向新类"的 bias 会**反传进 backbone 特征**，把特征也带偏，进一步加速遗忘。

所以本文的靶子是 **detection-head bias + model recency**，而不是泛泛的"遗忘"。这是全文的立论点，后面所有设计都服务它。

---

## 2. 方法拆解

### 2.1 记号

- 教师 1：$M_{t-1}$，上一阶段模型，**旧类专家**；
- 教师 2：$M_t^{im}$，"intermediate teacher"，**只用 $D_t$ 训练**，所以它对当前新类有适应性、对旧类几乎无知；
- 学生：$M_t$（从 $M_{t-1}$ 初始化）；
- 分类头分别记 $H_{t-1}$、$H_t^{im}$、$H$（学生）。

**"future-aware" 的落点就在这里**：$M_t^{im}$ 把当前数据里的未来类当背景看，用它来修旧教师的背景概率，等于把"未来类"显式建模进蒸馏目标。

### 2.2 蒸馏区域划分（Eq.1，直接沿用 BPF [29]）

以 $M_{t-1}$ 的 proposal 为蒸馏区域 $R$，按与当前新类 GT 框 $Y_t$ 的 IoU 切成：

- $R_1$：$\forall y_i,\ \mathrm{IoU}(d_j,y_i)\le \lambda_2$ —— 大概率是**旧类**区域；
- $R_2$：IoU $>\lambda_2$ —— 大概率含**新类**。

$R_1$ 用旧教师保旧知识，$R_2$ 用中间教师保新知识，互不污染。⚠️ $\lambda_2$ 的具体取值正文**没给**（BPF 里通常 0.5），而且这个 $\lambda$ 和 Eq.20 里的损失权重 $\lambda=0.4$ **重名**，是写作缺陷，读的时候要自己区分。

### 2.3 背景概率重构（Eq.2–3）

教师侧：
$$\hat p^{c,im}_r = p^{c,im}_r\cdot p^{b,t-1}_r\ (r\in R_1),\qquad
\hat p^{c,t-1}_r = p^{c,t-1}_r\cdot p^{b,im}_r\ (r\in R_2)$$

直觉：**用对方的背景概率去压低自己不该有的前景置信**。$R_1$ 里中间教师可能把旧类误当背景（它的 $p^b$ 大），乘上去正好把这部分前景概率削掉；$R_2$ 反过来。然后拼成教师目标
$$\bar p_r=\mathrm{concat}(\hat p^{c,im}_r,\ p^{c,t-1}_r),\ r\in R_1;\qquad \bar p_r=\mathrm{concat}(\hat p^{c,t-1}_r,\ p^{c,im}_r),\ r\in R_2$$

也就是**目标向量 = [被修正的那一份前景概率] ++ [另一份原始前景概率]**，把两个教师的知识域拼成一个统一标签空间。⚠️ 式(2)(3) 里 $p^b$ 到底是不是标量、维度怎么对齐，正文没写清；实现时基本只能用 BPF 的写法去推。

### 2.4 跨头解码（Eq.4–5）—— 全文关键

学生 ROI 特征 $z$（来自学生 backbone + RPN 的 proposal）分别过**两个教师的分类头**：
$$p^{ch,t-1}=\mathrm{softmax}(H_{t-1}(z)),\qquad p^{ch,im}=\mathrm{softmax}(H^{im}_t(z))$$

学生侧同样做一次背景重构（Eq.6–7）得到 $\bar p^{ch}_r$，最后
$$\mathcal L_{FaCHD}=\frac{1}{|R|}\sum_{r\in R}\mathrm{KL}\big(\bar p_r\ \|\ \bar p^{ch}_r\big)$$

**为什么这能解耦**：$H_{t-1}$、$H^{im}_t$ 都是冻结的，所以 KL 的梯度只能沿 $z$ 回传到学生 backbone/RPN，**根本不会经过学生自己的分类头 $H$**。学生头只吃 $\mathcal L_{cls}$ 的 GT 监督。于是：
- backbone 被塑造成"跟头无关的稳定几何表示"；
- 头上不再有新旧两股力打架。

总损失 $\mathcal L_{total}=\mathcal L_{cls}+\mathcal L_{box}+\alpha\mathcal L_{FaCHD}$（Eq.9）。⚠️ 注意**没有 RPN 蒸馏项、也没有 ROI 特征蒸馏项**——而 ILOD/Faster ILOD 那条线是有的。是省略没写还是真的不加，是复现时第一个要确认的点。

### 2.5 与 CrossKD [36] 的关系（读这篇必须搞清）

机制上就是 **CrossKD 搬到 IOD**：

| | CrossKD (CVPR24) | FaCHD |
|---|---|---|
| 跨头预测 | 学生特征 × **教师头** | 学生 ROI 特征 × **两个教师头** |
| 蒸馏方向 | 教师输出 → 学生的跨头预测（跨头预测是被蒸馏的一方） | 同一个方向 |
| 冻结 | 教师头冻结 | 两个教师头都冻结 |
| 额外 | — | 区域划分 + 背景重构（BPF） + 双教师互补 + RPSC |

结论：**FaCHD 的新意不在"跨头"这个 trick 本身**（这是 [36] 的），而在于"双教师 × 背景重构 × 按 IoU 分区"把它适配到 IOD 的缺失标注/未来类场景，再配一个原型重放模块。作者只在正文用一句"[36]"带过，没正面讨论这个继承关系——审稿意见/组会质疑大概率会打这里。

### 2.6 RPSC：分类头重训（Eq.10–20）

三步：

**(1) 建多粒度原型**
- 全局：$\mu^g_c=\frac{1}{n_c}\sum_i z^c_i$（Eq.11）
- 局部：对每个 RoI 特征取余弦相似度 > γ 的邻域 $S^c_j$（Eq.12），**贪心**选出样本数最多的 top-K 个互不重叠超球，求均值得 $\mu^\ell_c$（Eq.13）。K、γ 正文都没给具体值。

**(2) 估漂移并补偿**（跟 SDC [43]）
$$\delta=z^t-z^{t-1},\qquad
\hat\Delta_c=\frac{\sum_i\alpha_{i,c}\delta_i}{\sum_i\alpha_{i,c}},\quad
\alpha_{i,c}=\exp\!\Big(-\frac{\|z^{t-1}-\mu^{t-1}_c\|^2}{2\sigma_c^2}\Big)$$
即"离旧原型越近的样本，其特征位移越可信"。然后
$$\hat\mu^g_c=\mu^{g,(t-1)}_c+\hat\Delta_c,\qquad \hat\mu^\ell_c=\mu^{\ell,(t-1)}_c+\hat\Delta_c$$
新类原型不补偿（Eq.17），因为它们本来就是当前模型抽的。

**(3) 冻结 backbone+RPN，只用原型重训分类头**
$$\mathcal L_{re}=-\sum_{c\in C_{1:t-1}}y_c\log\hat p^{t-1}_c-\lambda\,y_c\!\!\sum_{c\in C_t}\log\hat p^{t}_c$$
每个 iteration 喂一个 prototype mini-batch 过 $H$，只更新 $H$。

**这一步是"零样本重放"**：不存图、不存特征，只存原型；也是它能号称 rehearsal-free 的原因。代价是——分类头只见过原型分布，没见过真实 RoI 分布，这和后面实验里"新类掉得厉害"可能直接相关（见 §4 的疑点）。

---

## 3. 实验

### 3.1 设置

Faster R-CNN + ResNet-50（ImageNet 预训练）；VOC 2007（20 类，按字母序切分）、COCO 2017（80 类）；单卡 RTX 3090，batch 16，SGD；VOC 10-10 上 $\alpha=20$、$\lambda=0.4$。**不用 exemplar**。

### 3.2 VOC 2007 单步（Table 1，mAP@0.5）

| 设置 | Ours Old / New / All | 最强对手 | 说明 |
|---|---|---|---|
| 10-10 | 76.0 / 75.9 / **75.9** | GDA-IOD 74.9 | 比 GDA-IOD 旧类 +0.9、新类 +1.3 |
| 15-5 | 78.4 / 65.0 / **75.1** | GDA-IOD 73.6 | |
| 19-1 | 76.8 / 58.5 / **75.9** | GMDP-ILOD 73.9 | +2.0 |
| 5-15 | 71.2 / 77.6 / **76.0** | GDA-IOD 74.1 | |

注：`All` 是**按类别数加权**，不是 old/new 的算术平均（例：5-15 的 $(5\times71.2+15\times77.6)/20=76.0$ ✓）。这个验证做一遍，Table 1/3 的数字基本都能对上。

### 3.3 VOC 多步（Table 3）

多步表每组末尾三列都是**最后阶段的 旧类 / 新类 / All**，加权公式同样成立：

| 设置 | Ours 旧 / 新 / All | 主要对手（All） |
|---|---|---|
| 10-5 | 72.2 / 70.8 / **71.5** | BPF 68.7 |
| 5-5 | 63.2 / 68.1 / **66.9** | BPF 62.5、GMDP-ABR 63.3（+4.4 / +3.6） |
| 10-2 | 68.8 / 57.1 / **62.9** | BPF 62.5 |
| 15-5 | 76.5 / **41.3** / **67.7** | BPF 71.5 / 53.1 / 66.9 |

规律很明显：**旧类保持是它的强项，新类 plasticity 是短板**。15-5 那组旧类全场最高 76.5，新类却只有 41.3（对手 53~59），All 只靠旧类加权才勉强领先。这和 RPSC"只用原型重训头"的设计高度自洽——见 §4 第 4 条。

### 3.4 COCO（Table 2，AP）

40+40：**35.5** / 55.7 / 38.9（Joint 40.2；NSGP-RePRE 35.4）
70+10：**36.9** / 57.1 / 40.1

### 3.5 消融

| | FaCHD | RPSC | 10-10 All |
|---|---|---|---|
| (a) baseline (GDA-IOD 风格: 伪标签 + GMM) | – | – | 75.4 |
| (b) | ✓ | – | 75.7 |
| (c) | – | ✓ | 75.8 |
| (d) | ✓ | ✓ | **76.0** |

单模块各 +0.3/+0.4，合用 +0.6 —— **增益很薄**，而且 baseline 已经很强（75.4）。这点要心里有数：FaCHD 在 10-10 上只值 0.3 个点，真正的卖点是 5-5（+4.4）和 19-1（+2.0）这类更难设置。
DwF++ 对比（Table 5）：+0.4 / +0.5 / +0.6 / +0.5，同样是小幅但**一致的**提升。
超参：α=20 最优（5→20 单调升，>20 反而掉，作者归因于收敛受阻）；λ=0.4 最优但对 λ 很不敏感。

---

## 4. 我读出来的疑点（/ 组会会被问的）

1. **$M_t^{im}$ 到底怎么来的？** 正文只说"只用 $D_t$ 优化"，没说初始化（从 $M_{t-1}$ 微调？从 ImageNet 重训？训多少 iter？）。这是复现的**头号阻塞点**，也决定了"future-aware"这个词到底站不站得住。
2. **蒸馏目标里缺 RPN / box 回归项**。$\mathcal L_{total}$ 只有 cls+box+FaCHD。ILOD 系列都蒸馏 RPN 和回归头，这里不蒸馏，定位能力靠什么保？10-10 只加 FaCHD 就涨，是否说明 backbone 特征一致性顺带保住了 RPN？
3. **Table 1 有个明显笔误**：GMDP-ILOD 行 5-15 的 All 写成 **75.6**，但按加权公式 $(5\times65.2+15\times60.5)/20=61.7$。同表其余行都对得上，所以应是排版错误——引用这个数时小心。
4. **强旧类 / 弱新类的系统偏置**。RPSC 只用原型重训头，头被"拽"回旧类分布；Table 3 里 1-15/16-20 新类只有 41.3（对手 50~53），旧类却是全场最高 76.5。作者归因为"更好的稳定性"，但也可能是**过度回放**。论文没给 per-class 分析，也没给遗忘矩阵。
5. **$\lambda$ 符号撞车**（Eq.1 的 IoU 阈值 $\lambda_2$ vs Eq.20 的权重 $\lambda$），且 $\lambda_2$、K、γ 三个超参全没给值。
6. **增益幅度 vs 统计显著性**：主实验涨 0.5~2 点，无多次运行方差，无显著性检验。
7. **名字不统一**：3.4 节标题写 "Compensation"，正文/图 1 写 "Correction / Calibration"；正文还有 "haed"、"stander"、$L_{cls}$ vs $L_{cls}$+$L_{box}$ 下标不一致等笔误——投稿版没收干净。
8. **"解耦"是否彻底**：学生头在 FaCHD 阶段还是被 $\mathcal L_{cls}$ 更新，只是不接蒸馏梯度；但 RPN 是共享的，proposal 分布会变，$R$ 的划分随之漂移。论文没讨论这个反馈回路。

---

## 5. 复现清单（若要动手）

必须先拿到/实现：
- [ ] GDA-IOD [28] 的伪标签 + GMM 方案 —— 它就是 (a) baseline，没有它就无法对齐 75.4
- [ ] BPF [29] 的区域划分 + 背景重构（Eq.1–3, 6–7）原码；FaCHD 的目标构造直接照抄它
- [ ] CrossKD [36] 的跨头蒸馏写法（学生特征 × 教师头的接口）
- [ ] $M_t^{im}$ 的训练脚本（论文缺失，需自己定：多数 IOD 双教师工作是"$M_{t-1}$ 在 $D_t$ 上 finetune 若干 epoch"）
- [ ] RPSC 的超球 top-K 贪心选择 + γ 阈值（缺失，需按 [39] NSGP 反推）
- [ ] 每阶段 iter 数、lr schedule、是否冻结 RPN

工作量估计：核心改动在 ROI head 的 forward 里加两条"教师头旁路"，加上一个 prototype bank + 二阶段重训循环。相对 U-Net 那套复现，工程量中等偏上，主要在数据侧（VOC 按字母序切、COCO 80 类切 40+40）。

---

## 6. 术语速查

| 缩写 | 全称 | 备注 |
|---|---|---|
| IOD | Incremental Object Detection | 本文任务 |
| FaCHD | Future-aware Decoupled Cross-Head Distillation | 主模块 |
| RPSC | Region Prototype Semantic Correction/Calibration | 原型补偿模块 |
| $M_{t-1}$ | 旧类专家教师 | 冻结 |
| $M_t^{im}$ | 中间教师（只训 $D_t$） | 冻结；"future" 的来源 |
| BPF | Bridge Past and Future [29] | 区域划分/背景重构的来源 |
| GDA-IOD | [28] ICCV25 | 直接 baseline，同组前作 |
| SDC | Semantic Drift Compensation [43] | 漂移估计的出处 |
