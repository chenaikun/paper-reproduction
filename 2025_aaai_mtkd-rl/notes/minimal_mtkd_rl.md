# 用最小实验理解 MTKD-RL

运行：

```bash
python paper-reproduction/2025_aaai_mtkd-rl/notes/minimal_mtkd_rl.py
```

## 实验替换了什么

| 论文 | 最小实验 |
|---|---|
| CIFAR-100 图像 | 二维平面上的四类高斯数据 |
| ResNet/WRN/RegNet | 两个 2-16-4 微型 MLP 教师 |
| 学生 CNN | 一个 2-16-4 微型 MLP 学生 |
| 大量教师 | 两个互补教师 |
| 分布式训练 | 普通 NumPy 单进程 |

教师 A 只使用类别 0/1 的训练样本，教师 B 只使用类别 2/3 的训练样本。因此 A 在左半区域可靠，B 在右半区域可靠；学生则同时看到全部四类。

## 公式和代码的对应关系

### 1. 多教师 KD

```text
L = CE(student, y)
  + alpha * sum_m w_logit[m] * KL(student, teacher_m)
  + beta  * sum_m w_feature[m] * MSE(student_feature, teacher_feature_m)
```

对应 `student_kd_step()`。

### 2. State

每个教师提供：

```text
[teacher_feature,
 teacher_logits,
 teacher_CE,
 cosine(student_feature, teacher_feature),
 KL(student_logits, teacher_logits)]
```

所有教师的状态拼接后送入策略网络。对应 `build_state()`。

不同 state 维度（特征、logit、KL）的数值范围不同。策略训练前会对 state 做标准化，否则大数值维度会淹没小数值信号。

### 3. Action

策略网络输出 `2M` 个连续值：

```text
前 M 个值 -> softmax -> logit 蒸馏权重
后 M 个值 -> softmax -> feature 蒸馏权重
```

对应 `GaussianWeightPolicy.act()` 和 `weights()`。

这里使用固定方差的对数正态策略。论文正文直接把策略输出当作权重，代码实现则使用确定性的 softmax 输出。为了让标准 Policy Gradient 的 `log pi(a|s)` 有明确含义，这个小实验从策略分布中采样动作。

### 4. Reward

论文 Eq.(6) 定义的是逐教师奖励：

```text
R_m = -(student_CE
        + alpha * KL(student, teacher_m)
        + beta * feature_MSE(student, teacher_m))
```

源码最终把加权蒸馏项合成为一个标量奖励。为了让 PG 中的 action 直接影响 reward，最小实验也采用这种形式：

```text
reward_i = -(student_CE_i
             + alpha * sum_m w_logit[m] * KL_i_m
             + beta  * sum_m w_feature[m] * feature_MSE_i_m)
```

对应 `weighted_student_reward()`。这是教学实验与论文逐教师公式之间的刻意差异。

### 5. PG

为了让最小实验稳定收敛，代码对 batch 内的 reward 减均值并除以标准差：

```text
advantage = (reward - mean_batch(reward)) / std_batch(reward)
```

然后：

```text
loss_pg = -mean(advantage * log_prob(action | state))
```

对应 `batch_reward_advantages()` 和 `GaussianWeightPolicy.pg_step()`。这与论文 Eq.(8) 都使用了 reward baseline，但比较维度不同，属于最小实验的数值稳定措施。

## 训练流程

```text
训练并冻结两个教师
        ↓
学生使用等权教师蒸馏 warm-up
        ↓
对每个 batch：
  1. 当前学生和教师 -> state
  2. policy 从动作分布采样权重
  3. 用这些权重执行一次加权 KD，更新学生
  4. 用更新后的学生计算每个教师的 reward
  5. 用 REINFORCE 更新 policy
        ↓
重复并在验证集上观察准确率和教师权重
```

## 看输出时关注什么

最重要的不是最终准确率，而是下面两个值：

```text
left_logit_weight_a
right_logit_weight_a
```

如果策略学到了样本相关的教师权重，应满足：

```text
left_logit_weight_a  > right_logit_weight_a
```

也就是说，策略对教师 A 擅长的左半区域给予了更大的 A 权重。这个变化就是论文想通过 RL 学到的“样本级教师权重”。

## 它没有覆盖论文的哪些内容

- 没有真实图像和 CNN 特征。
- 没有每个教师单独的特征转换器。
- 没有 confidence-aware 和 divergence-aware 的额外权重融合。
- 没有 1000 个 batch 的历史缓存和分布式训练。
- 没有检测、分割和 ImageNet 实验。

这些属于工程规模或扩展模块。最小实验保留的是论文最核心的闭环：多教师蒸馏、样本状态、教师权重动作、学生表现奖励和策略梯度更新。
