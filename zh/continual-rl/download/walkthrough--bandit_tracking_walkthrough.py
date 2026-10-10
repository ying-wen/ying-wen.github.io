"""Bandit 逐步算例：同一份经验怎样进入估计、跟踪与评价。

运行：python3 bandit_tracking_walkthrough.py
检查：python3 bandit_tracking_walkthrough.py --test

仅使用标准库。输入是明确给定的八次奖励，不采样环境、不训练策略。
前四次 [1, 0, 1, 0] 可以来自均值 0.5 的 Bernoulli 臂；随后均值
变为 1，故后四次奖励全为 1。动作访问日程固定，用于隔离估计器。
动作编号 0/1/2 对应网页 A/B/C。n 是 B 的访问次数，不是环境步数。
参考 Sutton & Barto (2018), 第二版 §2.1–2.5；本算例独立编写。
"""
import argparse
import json
import math
import unittest


def update_selected(values, counts, action, reward, alpha=None):
    """只更新被选动作。alpha=None 使用该动作本身的样本计数。"""
    if not 0 <= action < len(values) or len(values) != len(counts):
        raise ValueError("values/counts/action 不匹配")
    if alpha is not None and not 0 < alpha <= 1:
        raise ValueError("常数步长必须在 (0, 1] 内")
    if not math.isfinite(reward):
        raise ValueError("reward 必须有限")
    # 先增加访问次数：第一份证据的样本均值步长应当为 1，而非除以 0。
    counts[action] += 1
    step_size = 1 / counts[action] if alpha is None else alpha
    before = values[action]
    values[action] += step_size * (reward - before)
    return {"before": before, "step_size": step_size,
            "error": reward - before, "after": values[action]}


def history_weights(n, alpha=None):
    """Q_1 及 R_1...R_n 在 Q_{n+1} 中的权重，包含初值的剩余权重。"""
    if n < 0 or not isinstance(n, int):
        raise ValueError("n 必须是非负整数")
    if alpha is not None and not 0 < alpha <= 1:
        raise ValueError("常数步长必须在 (0, 1] 内")
    if n == 0:
        return {"initial": 1.0, "rewards": []}
    if alpha is None:
        return {"initial": 0.0, "rewards": [1 / n] * n}
    # 当 alpha=1 时，Python 的 0**0=1 使最后一份奖励获得全部权重。
    return {"initial": (1 - alpha) ** n,
            "rewards": [alpha * (1 - alpha) ** (n - i)
                        for i in range(1, n + 1)]}


def fixed_reward_trace():
    """两个估计器读取完全相同的 B 臂奖励；没有行为选择混杂。"""
    rewards = [1, 0, 1, 0, 1, 1, 1, 1]
    mean_values, constant_values = [0.0] * 3, [0.0] * 3
    mean_counts, constant_counts = [0] * 3, [0] * 3
    rows = []
    for n, reward in enumerate(rewards, 1):
        mean = update_selected(mean_values, mean_counts, 1, reward)
        constant = update_selected(constant_values, constant_counts, 1, reward, 0.5)
        rows.append({"visit": n, "action": 1, "reward": reward,
                     "true_mean": 0.5 if n <= 4 else 1.0,
                     "sample_average": mean, "constant_step": constant,
                     "unselected_values": [mean_values[0], mean_values[2]]})
    return rows


def compare_estimates_and_actions():
    """给定两个估计快照，分别计算估计误差与 ε-greedy 的期望奖励。"""
    truth = [0.2, 0.5, 0.8]
    snapshots = {"X": [0.2, 0.79, 0.78], "Y": [0.0, 0.0, 0.55]}
    result = []
    for label, values in snapshots.items():
        # 固定最小索引打破并列；此处两个快照均无并列最大值。
        greedy = max(range(3), key=lambda a: values[a])
        mse = sum((q - v) ** 2 for q, v in zip(truth, values)) / 3
        probabilities = [0.1 / 3] * 3
        probabilities[greedy] += 0.9
        result.append({"label": label, "values": values, "mse": mse,
                       "greedy_action": greedy, "greedy_reward": truth[greedy],
                       "epsilon_reward": sum(p * q for p, q in zip(probabilities, truth))})
    return result


def walkthrough():
    return {"kind": "deterministic_teaching_calculation", "action_names": ["A", "B", "C"],
            "rewards": [1, 0, 1, 0, 1, 1, 1, 1], "rows": fixed_reward_trace(),
            "weights": {"sample_average": history_weights(8), "constant_step": history_weights(8, 0.5)},
            "evaluation": compare_estimates_and_actions()}


class WalkthroughTests(unittest.TestCase):
    def test_only_selected_action_changes(self):
        values, counts = [0.2, 0.5, 0.8], [2, 4, 2]
        row = update_selected(values, counts, 1, 1)
        self.assertEqual(counts, [2, 5, 2])
        self.assertEqual(values, [0.2, 0.6, 0.8])
        self.assertAlmostEqual(row["step_size"], 0.2)

    def test_first_sample_removes_initialization(self):
        values, counts = [100.0], [0]
        update_selected(values, counts, 0, 0.7)
        self.assertAlmostEqual(values[0], 0.7)

    def test_running_mean_matches_independent_prefix_sums(self):
        rewards = walkthrough()["rewards"]
        for n, row in enumerate(fixed_reward_trace(), 1):
            self.assertAlmostEqual(row["sample_average"]["after"], sum(rewards[:n]) / n)

    def test_constant_step_matches_weighted_sum(self):
        rewards = walkthrough()["rewards"]
        for n, row in enumerate(fixed_reward_trace(), 1):
            weights = history_weights(n, 0.5)
            self.assertAlmostEqual(sum(weights["rewards"]) + weights["initial"], 1)
            self.assertAlmostEqual(row["constant_step"]["after"],
                                   sum(w * r for w, r in zip(weights["rewards"], rewards[:n])))
        self.assertAlmostEqual(fixed_reward_trace()[-1]["constant_step"]["after"], 245 / 256)

    def test_general_weights_and_endpoint(self):
        for alpha in [0.1, 0.5, 1.0]:
            for n in [0, 1, 8, 50]:
                weights = history_weights(n, alpha)
                self.assertAlmostEqual(weights["initial"] + sum(weights["rewards"]), 1)
        self.assertEqual(history_weights(3, 1)["rewards"], [0, 0, 1])

    def test_better_mse_can_make_worse_decision(self):
        x, y = compare_estimates_and_actions()
        self.assertLess(x["mse"], y["mse"])
        self.assertLess(x["greedy_reward"], y["greedy_reward"])
        self.assertAlmostEqual(x["epsilon_reward"], 0.5)
        self.assertAlmostEqual(y["epsilon_reward"], 0.77)

    def test_invalid_rate_is_rejected(self):
        for alpha in [0, -0.1, 1.1]:
            with self.assertRaises(ValueError):
                update_selected([0.0], [0], 0, 1, alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="只执行确定性单元测试")
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=[__file__])
    else:
        print(json.dumps(walkthrough(), ensure_ascii=False, indent=2))
