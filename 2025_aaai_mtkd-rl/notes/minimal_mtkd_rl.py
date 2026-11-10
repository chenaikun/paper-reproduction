"""A tiny, runnable MTKD-RL example.

This file is a teaching implementation of the main idea in:

    Multi-Teacher Knowledge Distillation with Reinforcement Learning
    for Visual Recognition

It deliberately replaces images and CNNs with a 2-D, four-class dataset and
small MLPs.  The important data flow is unchanged:

    state -> policy action -> weighted multi-teacher KD -> reward -> PG update

Run:

    python paper-reproduction/2025_aaai_mtkd-rl/notes/minimal_mtkd_rl.py

Only NumPy is required.  The forward and backward passes are written out so
the relationship between the formulas and the code is visible.

Pedagogical simplification:

* The paper finally integrates several agents into one policy.  We use one
  policy with 2*M outputs: M logit-level weights and M feature-level weights.
* To make Policy Gradient well-defined, the policy is a diagonal Gaussian over
  teacher-weight logits.  After sampling, two softmax operations turn these
  logits into logit-level and feature-level weight vectors.
* Both student and teachers expose their hidden feature and output logits.
  They have the same feature width, so no feature translator is needed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Basic neural-network and probability utilities
# ---------------------------------------------------------------------------


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


@dataclass
class MLPCache:
    x: np.ndarray
    hidden_pre: np.ndarray
    hidden: np.ndarray
    logits: np.ndarray


class TinyMLP:
    """Two-layer MLP: input -> tanh hidden feature -> output logits."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rng: np.random.Generator,
        output_scale: float = 1.0,
    ) -> None:
        self.params = {
            "W1": rng.normal(0.0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim),
            "W2": rng.normal(0.0, output_scale / np.sqrt(hidden_dim), (hidden_dim, output_dim)),
            "b2": np.zeros(output_dim),
        }

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, MLPCache]:
        """Return (hidden feature, logits, cache)."""
        hidden_pre = x @ self.params["W1"] + self.params["b1"]
        hidden = np.tanh(hidden_pre)
        logits = hidden @ self.params["W2"] + self.params["b2"]
        return hidden, logits, MLPCache(x, hidden_pre, hidden, logits)

    def backward(
        self, cache: MLPCache, grad_logits: np.ndarray, grad_feature: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """Backpropagate from logits and, optionally, from the hidden feature."""
        grad_hidden = grad_logits @ self.params["W2"].T
        if grad_feature is not None:
            grad_hidden = grad_hidden + grad_feature

        grad_hidden_pre = grad_hidden * (1.0 - cache.hidden**2)
        return {
            "W1": cache.x.T @ grad_hidden_pre,
            "b1": grad_hidden_pre.sum(axis=0),
            "W2": cache.hidden.T @ grad_logits,
            "b2": grad_logits.sum(axis=0),
        }

    def sgd_step(self, grads: dict[str, np.ndarray], learning_rate: float) -> None:
        for name in self.params:
            self.params[name] -= learning_rate * grads[name]


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean cross entropy and its gradient with respect to logits."""
    probs = softmax(logits)
    batch_size = targets.shape[0]
    rows = np.arange(batch_size)
    loss = -np.log(probs[rows, targets] + 1e-12).mean()
    grad = probs
    grad[rows, targets] -= 1.0
    grad /= batch_size
    return float(loss), grad


def per_sample_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    log_probs = log_softmax(logits)
    return -log_probs[np.arange(targets.shape[0]), targets]


def kl_divergence(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    temperature: float = 2.0,
) -> np.ndarray:
    """Per-sample KL(teacher || student), matching the paper's logit KD loss."""
    student_log_probs = log_softmax(student_logits / temperature)
    teacher_log_probs = log_softmax(teacher_logits / temperature)
    teacher_probs = np.exp(teacher_log_probs)
    kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(axis=1)
    return kl * temperature**2


def kl_grad(student_logits: np.ndarray, teacher_logits: np.ndarray, temperature: float = 2.0) -> np.ndarray:
    """Gradient of KL(teacher || student) with respect to student logits."""
    return temperature * (softmax(student_logits / temperature) - softmax(teacher_logits / temperature))


def feature_mse(student_feature: np.ndarray, teacher_feature: np.ndarray) -> np.ndarray:
    """Per-sample feature MSE."""
    return ((student_feature - teacher_feature) ** 2).mean(axis=1)


# ---------------------------------------------------------------------------
# The tiny dataset and complementary teachers
# ---------------------------------------------------------------------------


def make_data(num_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Four Gaussian clusters in 2-D."""
    centers = np.array(
        [
            [-2.0, -2.0],
            [-2.0, 2.0],
            [2.0, -2.0],
            [2.0, 2.0],
        ]
    )
    labels = rng.integers(0, 4, size=num_samples)
    inputs = centers[labels] + 0.65 * rng.normal(size=(num_samples, 2))
    return inputs.astype(np.float64), labels.astype(np.int64)


def accuracy(model: TinyMLP, inputs: np.ndarray, labels: np.ndarray) -> float:
    _, logits, _ = model.forward(inputs)
    return float((logits.argmax(axis=1) == labels).mean())


def train_classifier(
    model: TinyMLP,
    inputs: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    for _ in range(steps):
        index = rng.choice(inputs.shape[0], size=batch_size, replace=False)
        _, logits, cache = model.forward(inputs[index])
        _, grad_logits = cross_entropy(logits, labels[index])
        grads = model.backward(cache, grad_logits)
        model.sgd_step(grads, learning_rate)


# ---------------------------------------------------------------------------
# State and policy
# ---------------------------------------------------------------------------


def build_state(
    student: TinyMLP,
    teachers: list[TinyMLP],
    inputs: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Construct the paper's state for every sample and teacher.

    For each teacher m, the state is:

        [teacher feature, teacher logits, teacher CE,
         student-teacher cosine similarity, student-teacher KL]

    All teacher blocks are concatenated so one policy can decide all M weights.
    """
    student_feature, student_logits, _ = student.forward(inputs)
    teacher_states = []

    for teacher in teachers:
        teacher_feature, teacher_logits, _ = teacher.forward(inputs)
        teacher_ce = per_sample_cross_entropy(teacher_logits, labels)[:, None]
        similarity = (student_feature * teacher_feature).sum(axis=1) / (
            np.linalg.norm(student_feature, axis=1) * np.linalg.norm(teacher_feature, axis=1) + 1e-8
        )
        student_teacher_kl = kl_divergence(student_logits, teacher_logits)[:, None]
        teacher_states.append(
            np.concatenate(
                [
                    teacher_feature,
                    teacher_logits,
                    teacher_ce,
                    similarity[:, None],
                    student_teacher_kl,
                ],
                axis=1,
            )
        )

    return np.concatenate(teacher_states, axis=1)


class GaussianWeightPolicy:
    """Gaussian policy over the logits of the teacher-weight softmax.

    2*M outputs are used:

        action[0:M]   -> softmax -> logit-level teacher weights
        action[M:2M]  -> softmax -> feature-level teacher weights
    """

    def __init__(
        self,
        state_dim: int,
        teacher_count: int,
        hidden_dim: int,
        rng: np.random.Generator,
        noise_std: float = 0.30,
    ) -> None:
        self.teacher_count = teacher_count
        self.network = TinyMLP(state_dim, hidden_dim, 2 * teacher_count, rng, output_scale=0.1)
        self.noise_std = noise_std
        self.state_mean: np.ndarray | None = None
        self.state_std: np.ndarray | None = None

    def fit_state_normalizer(self, state: np.ndarray) -> None:
        """Normalize state dimensions before they enter the policy network."""
        self.state_mean = state.mean(axis=0, keepdims=True)
        self.state_std = state.std(axis=0, keepdims=True) + 1e-8

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        if self.state_mean is None or self.state_std is None:
            return state
        return (state - self.state_mean) / self.state_std

    def act(self, state: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample an action and return (mean, sampled action, log probability)."""
        state = self._normalize_state(state)
        _, mean, _ = self.network.forward(state)
        action = mean + self.noise_std * rng.normal(size=mean.shape)
        std = self.noise_std
        log_prob = (
            -0.5 * ((action - mean) / std) ** 2
            - np.log(std)
            - 0.5 * np.log(2.0 * np.pi)
        ).sum(axis=1)
        return mean, action, log_prob

    def weights(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = self.teacher_count
        logit_weights = softmax(action[:, :m])
        feature_weights = softmax(action[:, m : 2 * m])
        return logit_weights, feature_weights

    def deterministic_weights(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = self._normalize_state(state)
        _, mean, _ = self.network.forward(state)
        return self.weights(mean)

    def pg_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        advantages: np.ndarray,
        learning_rate: float,
    ) -> float:
        """REINFORCE update for the diagonal Gaussian policy."""
        state = self._normalize_state(state)
        _, mean, cache = self.network.forward(state)
        batch_size = state.shape[0]
        std = self.noise_std

        advantages = np.concatenate([advantages, advantages], axis=1)
        diff = (action - mean) / std
        product = advantages * diff
        loss = -product.sum() / batch_size

        # d[-mean(A * log_prob)] / d(mean)
        grad_mean = -advantages * diff / (batch_size * std**2)
        grads = self.network.backward(cache, grad_mean)
        self.network.sgd_step(grads, learning_rate)
        return float(loss)


# ---------------------------------------------------------------------------
# Weighted multi-teacher knowledge distillation
# ---------------------------------------------------------------------------


def student_kd_step(
    student: TinyMLP,
    teachers: list[TinyMLP],
    inputs: np.ndarray,
    labels: np.ndarray,
    logit_weights: np.ndarray,
    feature_weights: np.ndarray,
    learning_rate: float,
    alpha: float,
    beta: float,
) -> float:
    """One SGD step on the paper's weighted multi-teacher KD objective."""
    teacher_data = []
    for teacher in teachers:
        teacher_feature, teacher_logits, _ = teacher.forward(inputs)
        teacher_data.append((teacher_feature, teacher_logits))

    student_feature, student_logits, cache = student.forward(inputs)
    batch_size = inputs.shape[0]
    ce_loss, grad_logits = cross_entropy(student_logits, labels)
    loss = ce_loss
    grad_feature = np.zeros_like(student_feature)

    for idx, (teacher_feature, teacher_logits) in enumerate(teacher_data):
        logit_loss = kl_divergence(student_logits, teacher_logits)
        feature_loss = feature_mse(student_feature, teacher_feature)

        loss += float(
            (
                alpha * logit_weights[:, idx] * logit_loss
                + beta * feature_weights[:, idx] * feature_loss
            ).mean()
        )

        grad_logits += (
            alpha
            * logit_weights[:, idx, None]
            * kl_grad(student_logits, teacher_logits)
            / batch_size
        )
        grad_feature += (
            beta
            * feature_weights[:, idx, None]
            * 2.0
            * (student_feature - teacher_feature)
            / student_feature.shape[1]
            / batch_size
        )

    grads = student.backward(cache, grad_logits, grad_feature)
    student.sgd_step(grads, learning_rate)
    return loss


def weighted_student_reward(
    student: TinyMLP,
    teachers: list[TinyMLP],
    inputs: np.ndarray,
    labels: np.ndarray,
    logit_weights: np.ndarray,
    feature_weights: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Per-sample reward for the actual sampled teacher weights.

    The paper writes teacher-specific rewards in Eq.(6) and compares them
    across teachers in Eq.(8).  The current official-style implementation
    instead combines the weighted KD terms into one scalar per sample and
    uses that as the PG reward.  We follow that aggregate form here because
    it makes the sampled action directly affect the reward.
    """
    student_feature, student_logits, _ = student.forward(inputs)
    student_ce = per_sample_cross_entropy(student_logits, labels)
    reward = -student_ce

    for idx, teacher in enumerate(teachers):
        teacher_feature, teacher_logits, _ = teacher.forward(inputs)
        reward -= (
            alpha * logit_weights[:, idx] * kl_divergence(student_logits, teacher_logits)
            + beta * feature_weights[:, idx] * feature_mse(student_feature, teacher_feature)
        )

    return reward


def batch_reward_advantages(rewards: np.ndarray, teacher_count: int) -> np.ndarray:
    """Batch baseline for the scalar action reward.

    We broadcast the same advantage to both policy heads because the reward is
    a scalar to which logit-level and feature-level weights both contribute.
    """
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    return np.repeat(advantages[:, None], teacher_count, axis=1)


# ---------------------------------------------------------------------------
# Main training procedure
# ---------------------------------------------------------------------------


def evaluate(
    student: TinyMLP,
    teachers: list[TinyMLP],
    policy: GaussianWeightPolicy,
    inputs: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    result = {
        "student_acc": accuracy(student, inputs, labels),
        "teacher_a_acc": accuracy(teachers[0], inputs, labels),
        "teacher_b_acc": accuracy(teachers[1], inputs, labels),
        "teacher_a_good_region_acc": accuracy(teachers[0], inputs[labels < 2], labels[labels < 2]),
        "teacher_b_good_region_acc": accuracy(teachers[1], inputs[labels >= 2], labels[labels >= 2]),
    }

    state = build_state(student, teachers, inputs, labels)
    logit_weights, feature_weights = policy.deterministic_weights(state)

    # Classes 0/1 are teacher A's strong region. Classes 2/3 are B's.
    left = labels < 2
    result["left_logit_weight_a"] = float(logit_weights[left, 0].mean())
    result["right_logit_weight_a"] = float(logit_weights[~left, 0].mean())
    result["left_feature_weight_a"] = float(feature_weights[left, 0].mean())
    result["right_feature_weight_a"] = float(feature_weights[~left, 0].mean())
    return result


def main() -> None:
    seed = 7
    rng = np.random.default_rng(seed)

    teacher_count = 2
    hidden_dim = 16
    alpha = 0.7
    beta = 0.25
    student_lr = 0.06
    policy_lr = 0.05
    batch_size = 64
    warmup_steps = 120
    alternating_steps = 1000

    train_x, train_y = make_data(1600, rng)
    val_x, val_y = make_data(600, rng)

    # Step 1: train and freeze two complementary specialists.
    # Teacher A only sees classes 0/1. Teacher B only sees classes 2/3.
    teacher_a = TinyMLP(2, hidden_dim, 4, rng)
    teacher_b = TinyMLP(2, hidden_dim, 4, rng)
    train_classifier(
        teacher_a,
        train_x[train_y < 2],
        train_y[train_y < 2],
        rng,
        500,
        128,
        0.08,
    )
    train_classifier(
        teacher_b,
        train_x[train_y >= 2],
        train_y[train_y >= 2],
        rng,
        500,
        128,
        0.08,
    )
    teachers = [teacher_a, teacher_b]

    # Step 2: create a student and a policy over teacher weights.
    student = TinyMLP(2, hidden_dim, 4, rng)
    state_dim = teacher_count * (hidden_dim + 4 + 3)
    policy = GaussianWeightPolicy(state_dim, teacher_count, hidden_dim, rng, noise_std=0.25)

    print("Before KD")
    print("  student:", f"{accuracy(student, val_x, val_y):.3f}")
    print("  teacher A:", f"{accuracy(teacher_a, val_x, val_y):.3f}")
    print("  teacher B:", f"{accuracy(teacher_b, val_x, val_y):.3f}")

    # Step 3: warm up with uniform teacher weights, as in Algorithm 1.
    for step in range(warmup_steps):
        index = rng.choice(train_x.shape[0], size=batch_size, replace=False)
        x = train_x[index]
        y = train_y[index]
        uniform_logit = np.full((batch_size, teacher_count), 1.0 / teacher_count)
        uniform_feature = uniform_logit.copy()
        student_kd_step(
            student,
            teachers,
            x,
            y,
            uniform_logit,
            uniform_feature,
            student_lr,
            alpha,
            beta,
        )

    print("\nAfter equal-weight warm-up")
    policy.fit_state_normalizer(build_state(student, teachers, train_x, train_y))
    warmup_metrics = evaluate(student, teachers, policy, val_x, val_y)
    for key, value in warmup_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nAlternating student KD and policy-gradient updates")
    recent_rewards: list[float] = []
    recent_pg_losses: list[float] = []

    for step in range(1, alternating_steps + 1):
        # Explore strongly at first, then exploit the learned policy.
        policy.noise_std = 0.30 * (1.0 - step / alternating_steps) + 0.04

        index = rng.choice(train_x.shape[0], size=batch_size, replace=False)
        x = train_x[index]
        y = train_y[index]

        # State is computed before the student update.
        state = build_state(student, teachers, x, y)

        # Action = sampled continuous teacher-weight logits.
        _, action, _ = policy.act(state, rng)
        logit_weights, feature_weights = policy.weights(action)

        # Train the student with the sampled action.
        student_kd_step(
            student,
            teachers,
            x,
            y,
            logit_weights,
            feature_weights,
            student_lr,
            alpha,
            beta,
        )

        # Reward = negative weighted student KD loss.  It depends directly on
        # the sampled teacher weights, which gives Policy Gradient a signal.
        rewards = weighted_student_reward(
            student,
            teachers,
            x,
            y,
            logit_weights,
            feature_weights,
            alpha,
            beta,
        )
        advantages = batch_reward_advantages(rewards, teacher_count)

        # Policy update: PG uses (state, action, advantage) from this batch.
        pg_loss = policy.pg_step(state, action, advantages, policy_lr)
        recent_rewards.append(float(rewards.mean()))
        recent_pg_losses.append(pg_loss)

        if step % 100 == 0:
            metrics = evaluate(student, teachers, policy, val_x, val_y)
            print(
                f"step {step:4d} | "
                f"student={metrics['student_acc']:.3f} | "
                f"A={metrics['teacher_a_acc']:.3f} | "
                f"B={metrics['teacher_b_acc']:.3f} | "
                f"w_A(left)={metrics['left_logit_weight_a']:.3f} | "
                f"w_A(right)={metrics['right_logit_weight_a']:.3f} | "
                f"reward={np.mean(recent_rewards[-100:]):+.3f} | "
                f"pg_loss={np.mean(recent_pg_losses[-100:]):+.3f}"
            )

    print("\nFinal inspection")
    final_metrics = evaluate(student, teachers, policy, val_x, val_y)
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.4f}")

    print(
        "\nInterpretation:\n"
        "  Teacher A is reliable on classes 0/1 and teacher B on classes 2/3.\n"
        "  A successful run should give A a larger average weight on the left\n"
        "  and B a larger average weight on the right.  This demonstrates the\n"
        "  complete MTKD-RL loop: state -> weights -> weighted KD -> reward -> PG."
    )


if __name__ == "__main__":
    main()
