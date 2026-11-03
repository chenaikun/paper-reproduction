"""最小可跑的 REINFORCE 演示：情境式多臂老虎机（contextual bandit）。

目的：用 40 行看懂 MTKD-RL 的"state -> 教师权重 -> 回报 -> 更新策略"这一循环，
不涉及神经网络、不涉及计算图，纯 numpy。

映射关系：
    3 个臂        <-> 3 个教师
    softmax 权重  <-> 多教师蒸馏权重 w
    选臂          <-> "这一轮主要听哪个教师"
    臂的回报      <-> 学生训完之后的增益（这里用合成回报代替）
    上下文(场景)   <-> 场景级特征（光照/天气/密度）

运行：python notes/rl_bandit_demo.py
"""
import numpy as np

rng = np.random.default_rng(0)

# 两个场景（one-hot 上下文）；场景 0 里教师 0 最好，场景 1 里教师 2 最好
CTX = np.array([[1.0, 0.0],
                [0.0, 1.0]])
MU = np.array([[0.90, 0.50, 0.20],   # 场景 0 的真实回报均值
               [0.20, 0.50, 0.90]])  # 场景 1

W = np.zeros((2, 3))   # 策略参数：2 维场景 -> 3 个教师 logit
b = np.zeros(3)
lr, baseline = 0.10, 0.0

for step in range(6000):
    i = rng.integers(len(CTX))
    s = CTX[i]

    z = s @ W + b
    p = np.exp(z - z.max())
    p /= p.sum()

    a = rng.choice(3, p=p)                  # 按权重采样一个教师
    r = MU[i, a] + rng.normal(0, 0.1)       # 环境返回的回报（含噪声）

    adv = r - baseline                      # 减 baseline：只降方差，不改期望
    grad = -p.copy()
    grad[a] += 1.0                          # dlog p(a|s)/dz = onehot(a) - p
    W += lr * adv * np.outer(s, grad)       # 策略梯度上升
    b += lr * adv * grad
    baseline = 0.99 * baseline + 0.01 * r   # baseline 用滑动平均估计

    if (step + 1) % 2000 == 0:
        print(f"step {step+1:5d}  baseline={baseline:.3f}")

print("\n学到的权重（每行是一个场景）：")
for i, s in enumerate(CTX):
    z = s @ W + b
    p = np.exp(z - z.max()); p /= p.sum()
    print(f"  场景{i}: {np.round(p, 3)}   (最优教师 = {int(np.argmax(MU[i]))})")
