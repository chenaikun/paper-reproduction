"""教学张量演示；使用合成输入，不是论文实验或性能复现。仅依赖 numpy。"""
from pathlib import Path
import json
import numpy as np

def softmax(x, axis=-1):
    ex = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return ex / ex.sum(axis=axis, keepdims=True)

def main():
    p_s = np.array([[.4, .3, .2, .1], [.1, .2, .3, .4]])
    # [B, M, C]
    p_t = np.array([
        [[.7, .1, .1, .1], [.5, .3, .1, .1], [.15, .45, .3, .1]],
        [[.1, .1, .2, .6], [.1, .2, .1, .6], [.1, .1, .6, .2]],
    ])
    y = np.array([0, 3])
    f_s = np.array([[1., 0., 0.], [0., 1., 0.]])
    f_t = np.array([
        [[1., .2, 0.], [.8, .6, 0.], [0., 1., 0.]],
        [[0., 1., .2], [.3, .8, 0.], [1., 0., 0.]],
    ])
    ce_s = -np.log(p_s[np.arange(2), y])
    ce_t = -np.log(p_t[np.arange(2), :, y])
    kl = (p_t * (np.log(p_t) - np.log(p_s[:, None, :]))).sum(-1)
    cosine = (f_s[:, None] * f_t).sum(-1)
    cosine /= np.linalg.norm(f_s, axis=-1)[:, None] * np.linalg.norm(f_t, axis=-1)
    mse = ((f_s[:, None] - f_t)**2).mean(-1)
    # logits = log(probability) 在 T=1 下确实恢复输入概率。
    paper_state = np.concatenate([f_t, np.log(p_t), ce_t[..., None],
                                  cosine[..., None], kl[..., None]], axis=-1)
    w_conf = (1. - softmax(ce_t)) / 2
    w_div_l = softmax(kl)
    w_div_f = softmax(cosine)
    # 仅为解释双头输出而人为设定，未训练任何策略网络。
    w_gen_l = softmax(np.array([[1., 0., -1.], [-.5, .4, .8]]))
    w_gen_f = softmax(np.array([[.3, .8, -.2], [.2, .7, -.1]]))
    w_l = (w_conf + w_div_l + w_gen_l) / 3
    w_f = (w_conf + w_div_f + w_gen_f) / 3
    kd_per_sample = (w_l * kl).sum(-1)
    feature_per_sample = (w_f * mse).sum(-1)
    total = ce_s + kd_per_sample + 5 * feature_per_sample
    paper_reward = -ce_s[:, None] - kl - 5 * mse
    raw_minmax = (paper_reward - paper_reward.min(1, keepdims=True))
    raw_minmax /= np.ptp(paper_reward, axis=1, keepdims=True)
    paper_literal = raw_minmax - paper_reward.mean(1, keepdims=True)
    possible_centered = raw_minmax - raw_minmax.mean(1, keepdims=True)
    code_reward = -total
    code_z = (code_reward - code_reward.mean()) / code_reward.std(ddof=1)
    code_clipped = np.clip(code_z, 0, 1)
    # 模拟学生更新：将 logits 和 embedding 当作独立可训练叶子；
    # 实际网络由共享参数产生这些量，故这不是模型真实参数更新。
    label_onehot = np.eye(4)[y]
    teacher_mixture = (w_l[..., None] * p_t).sum(1)
    d_logits = (p_s - label_onehot) + (p_s - teacher_mixture)
    d_features = (2 * 5 / 3) * (w_f[..., None] * (f_s[:, None] - f_t)).sum(1)
    new_logits = np.log(p_s) - .1 * d_logits / 2
    new_features = f_s - .1 * d_features / 2
    p_new = softmax(new_logits)
    total_new = -np.log(p_new[np.arange(2), y])
    total_new += (w_l * (p_t * (np.log(p_t) - np.log(p_new[:, None]))).sum(-1)).sum(-1)
    total_new += 5 * (w_f * ((new_features[:, None] - f_t)**2).mean(-1)).sum(-1)
    result = {
        "boundary": "T=1, alpha=1, beta=5; synthetic probabilities/features; no training data",
        "p_s": p_s, "p_t": p_t, "labels": y, "f_s": f_s, "f_t": f_t,
        "state_shape": list(paper_state.shape),
        "ce_student": ce_s, "ce_teacher": ce_t, "kl_teacher_to_student": kl,
        "cosine": cosine, "mse": mse, "w_conf": w_conf,
        "w_div_logit": w_div_l, "w_div_feature": w_div_f,
        "w_gen_logit": w_gen_l, "w_gen_feature": w_gen_f,
        "w_logit": w_l, "w_feature": w_f,
        "kd_per_sample": kd_per_sample, "feature_mse_per_sample": feature_per_sample,
        "total_per_sample": total, "mean_total": float(total.mean()),
        "paper_reward_BM": paper_reward, "paper_minmax": raw_minmax,
        "paper_eq8_literal": paper_literal, "candidate_centering_not_paper": possible_centered,
        "code_reward_B": code_reward, "code_reward_z": code_z, "code_reward_clipped": code_clipped,
        "student_gradient_logits_per_sample": d_logits,
        "student_gradient_features_per_sample": d_features,
        "post_leaf_step_total_per_sample": total_new,
        "bce_uniform_M3_R1": float(-np.log(np.array([1/3, 1/3, 1/3])).mean()),
        "bce_peaked_M3_R1": float(-np.log(np.array([.8, .1, .1])).mean()),
    }
    def convert(x):
        return x.tolist() if isinstance(x, np.ndarray) else x
    result = {k: convert(v) for k, v in result.items()}
    assert np.allclose(w_conf.sum(1), 1)
    assert np.allclose(w_l.sum(1), 1) and np.allclose(w_f.sum(1), 1)
    assert np.all(paper_literal >= 0)
    assert np.allclose(paper_state.shape, [2, 3, 10])
    out = Path(__file__).resolve().parent / "模拟张量运行结果.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
