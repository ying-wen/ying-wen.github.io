#!/usr/bin/env python3
"""从已知梯度到抽样估计：两步任务中的同策略与重要性采样。

只需 Python 标准库。把这个文件下载到任意空目录后运行：
    python3 policy-gradient-sampling.py
    python3 policy-gradient-sampling.py --json --episodes 20000 --seed 2026
    python3 policy-gradient-sampling.py --theta -0.4 --gamma 1 --behavior 0.8

任务和 policy-gradient-walkthrough.py 相同。两个动作 a0、a1 共用参数
theta，目标策略在每个状态选择动作 1 的概率为 p = sigmoid(theta)。
初始状态为 s0；a0 决定下一状态 s1(a0)，所以这仍是一个 Markov 任务。
R1 = 0；只有轨迹 (a0, a1) = (1, 0) 的 R2 = 1，其余 R2 = 0。
目标是初始状态的折扣回报 J(theta) = E_pi[R1 + gamma * R2]。

这里逐步撤去“知道四条轨迹的总体概率”这一信息：先枚举真值，再分别
用目标策略 pi 和行为策略 b 产生经验。b 在两个状态选择动作 1 的概率
均为 --behavior。所有样本在同一 theta 下产生；代码只估计梯度，没有
训练回路。估计完成后才允许调用 propose_update 来形成一次更新提议。
这份独立示例不改变旧枚举脚本的默认输出。
"""

import argparse
import itertools
import json
import math
import random
from decimal import Decimal, InvalidOperation, localcontext


def _number(value, name):
    """用十进制读取参数，拒绝 NaN / Infinity 后再做概率计算。"""
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(name + " must be a finite number") from error
    if not result.is_finite():
        raise ValueError(name + " must be a finite number")
    return result


def support_check(p, behavior):
    """完整 IS 要求 P_pi(tau)>0 时 P_b(tau)>0。

    在此任务，有限且非退化的目标 logit 给四条轨迹正概率。若 b=0
    或 b=1，一些目标轨迹永远不会出现在行为经验中；不能靠加 epsilon
    到分母声称恢复无偏性。检查完整轨迹也让状态到达问题保持可见。
    """
    D = Decimal
    rows = []
    for a0, a1 in itertools.product((0, 1), repeat=2):
        pi_mass = (p if a0 else 1 - p) * (p if a1 else 1 - p)
        b_mass = (behavior if a0 else 1 - behavior) * (behavior if a1 else 1 - behavior)
        rows.append((str(a0) + str(a1), pi_mass, b_mass))
    missing = [name for name, pi_mass, b_mass in rows if pi_mass > D(0) and b_mass == D(0)]
    return {
        "valid": not missing,
        "targetSupport": [name for name, mass, _ in rows if mass > 0],
        "behaviorSupport": [name for name, _, mass in rows if mass > 0],
        "missingTargetTrajectories": missing,
    }


def _objective(theta, gamma):
    """有限差分只查询目标函数，完全不调用 score 梯度代码。"""
    p = 1 / (1 + (-theta).exp())
    return gamma * p * (1 - p)


def _moments(paths, probability_key, estimator_key, analytic):
    D = Decimal
    mean = sum((row[probability_key] * row[estimator_key] for row in paths), D(0))
    variance = sum((row[probability_key] * (row[estimator_key] - mean) ** 2 for row in paths), D(0))
    return {"mean": mean, "variance": variance, "bias": mean - analytic}


def exact_analysis(theta="0.7", gamma="0.5", behavior="0.25"):
    """50 位 Decimal 枚举给读者的尺子；这些概率不会传给抽样器当标签。"""
    with localcontext() as context:
        context.prec = 50
        D = Decimal
        theta = _number(theta, "theta")
        gamma = _number(gamma, "gamma")
        behavior = _number(behavior, "behavior")
        if not (0 <= gamma <= 1):
            raise ValueError("gamma must be in [0, 1]")
        if not (0 <= behavior <= 1):
            raise ValueError("behavior must be in [0, 1]")
        p = 1 / (1 + (-theta).exp())
        # 抽样器使用 float 随机数；其可表示范围也必须保留两个动作。
        if not (0 < p < 1 and 0 < float(p) < 1):
            raise ValueError("theta must give a nondegenerate float sampling policy")
        support = support_check(p, behavior)
        if not support["valid"]:
            missing = ", ".join(support["missingTargetTrajectories"])
            raise ValueError("behavior policy lacks target trajectory support: " + missing
                             + "; require 0 < behavior < 1 for this target policy")
        # Decimal 支持集可能完整，但转成 float 后会舍入为 0 或 1。这是本
        # 抽样演示的数值表示限制；它没有增加重要性采样的数学支持条件。
        if not (0 < float(behavior) < 1):
            raise ValueError("behavior must retain both actions in this float sampling demonstration")

        q = 1 - p
        analytic = gamma * p * q * (1 - 2 * p)
        paths = []
        for a0, a1 in itertools.product((0, 1), repeat=2):
            reward = D(a0 == 1 and a1 == 0)
            pi_actions = [p if a else q for a in (a0, a1)]
            b_actions = [behavior if a else 1 - behavior for a in (a0, a1)]
            scores = [D(a0) - p, D(a1) - p]
            returns = [gamma * reward, reward]  # G0 和 G1 从各自动作时刻开始。
            ratios = [pi / b for pi, b in zip(pi_actions, b_actions)]

            # ∇log pi(a_t|s_t) = a_t-p。外层 gamma^t 保留目标 J 的时间权重。
            # g(tau)=Σ_t gamma^t G_t score_t=gamma*R2*(score0+score1)。
            # G_t 是已观察的回报，作为 score 的系数保持不动（stop-gradient）。
            gradient = returns[0] * scores[0] + gamma * returns[1] * scores[1]
            full_ratio = ratios[0] * ratios[1]

            # 完整 ratio 改变整条经验的权重，包括 a0 导致的状态到达和 a1
            # 导致的终奖。这里估计 ∇J，ratio 是已在冻结 theta 计算的系数；
            # 若用自动微分实现 ratio * G * log pi 代理目标，应 detach ratio
            # 和 G，不能再次对这些系数求导、重复加入 score 路径。
            full_is = full_ratio * gradient

            # 故意错误：每一项只改当前动作分布。t=0 的 G0 仍由行为的 a1
            # 产生；t=1 的 s1(a0) 仍由行为的 a0 到达。两处条件分布都遗漏了。
            single_action = ratios[0] * returns[0] * scores[0] + ratios[1] * gamma * returns[1] * scores[1]
            paths.append({
                "id": str(a0) + str(a1), "actions": [a0, a1], "state1": "s1(" + str(a0) + ")",
                "targetProbability": pi_actions[0] * pi_actions[1],
                "behaviorProbability": b_actions[0] * b_actions[1],
                "reward": reward, "returns": returns, "scores": scores,
                "actionRatios": ratios, "fullRatio": full_ratio,
                "onPolicyGradient": gradient, "fullISGradient": full_is,
                "singleActionRatioGradient": single_action,
            })

        h = D("0.000001")
        finite_difference = (_objective(theta + h, gamma) - _objective(theta - h, gamma)) / (2 * h)
        return {
            "parameters": {"theta": theta, "gamma": gamma, "p": p, "behavior": behavior,
                           "J": _objective(theta, gamma), "analytic": analytic},
            "support": support,
            "trajectories": paths,
            "exact": {
                "analytic": analytic, "finiteDifference": finite_difference,
                "finiteDifferenceStep": h, "finiteDifferenceError": finite_difference - analytic,
                "onPolicy": _moments(paths, "targetProbability", "onPolicyGradient", analytic),
                "fullIS": _moments(paths, "behaviorProbability", "fullISGradient", analytic),
                "singleActionRatio": _moments(paths, "behaviorProbability", "singleActionRatioGradient", analytic),
                # 独立的条件期望式使误差不只出现在枚举表里。
                "singleActionMeanFormula": gamma * p * q * (1 - 2 * behavior),
                "singleActionBiasFormula": 2 * gamma * p * q * (p - behavior),
            },
        }


def _plain(value):
    if isinstance(value, Decimal):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("a reported quantity exceeds float range; choose less extreme probabilities for this sampling demonstration")
        return result
    if isinstance(value, dict):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_plain(child) for child in value]
    return value


class RunningMean:
    """Welford 在线均值和样本方差；不保存、重复使用或优化这批经验。"""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def observe(self, value):
        self.n += 1
        difference = value - self.mean
        self.mean += difference / self.n
        self.m2 += difference * (value - self.mean)

    def summary(self, exact):
        # 一个样本无法估计样本方差。JSON null 保留这项信息，不伪造零误差。
        variance = max(0.0, self.m2 / (self.n - 1)) if self.n > 1 else None
        return {"mean": self.mean, "errorFromAnalytic": self.mean - exact,
                "sampleVariance": variance,
                "standardError": math.sqrt(variance / self.n) if variance is not None else None}


def sample_episode(rng, probability):
    """先 a0 导致 s1(a0)，再在这个状态抽 a1；奖励只由环境规则给出。"""
    a0 = int(rng.random() < probability)
    a1 = int(rng.random() < probability)
    return a0, a1


def episode_estimates(actions, p, gamma, behavior):
    """只读取一条真实采样轨迹及冻结策略，返回三个可观察的梯度估计。"""
    a0, a1 = actions
    reward = float(a0 == 1 and a1 == 0)
    returns = (gamma * reward, reward)
    scores = (a0 - p, a1 - p)
    ratios = tuple((p if action else 1 - p) / (behavior if action else 1 - behavior)
                   for action in actions)
    terms = (returns[0] * scores[0], gamma * returns[1] * scores[1])
    gradient = sum(terms)
    return gradient, ratios[0] * ratios[1] * gradient, sum(ratio * term for ratio, term in zip(ratios, terms))


def _checkpoint_counts(episodes):
    """1、2、5 的对数网格以及最终样本数，便于画逐样本数均值图。"""
    result = {episodes}
    scale = 1
    while scale <= episodes:
        result.update(value * scale for value in (1, 2, 5) if value * scale <= episodes)
        scale *= 10
    return result


def propose_update(theta, gradient, step_size):
    """估计完成后的纯算术提议：不改策略、不采新轨迹、不声称改善 J。

    连续学习需要随后在新 theta 下重新采样或重新计算合适的离策略权重。
    将提议放在估计函数之外，避免“均值的样本来自哪个策略”悄悄改变。
    """
    numbers = [float(value) for value in (theta, gradient, step_size)]
    if not all(math.isfinite(value) for value in numbers) or numbers[2] < 0:
        raise ValueError("finite theta / gradient and nonnegative step_size required")
    theta, gradient, step_size = numbers
    return {"thetaBefore": theta, "gradient": gradient, "stepSize": step_size,
            "thetaProposed": theta + step_size * gradient, "applied": False}


def calculate(theta="0.7", gamma="0.5", behavior="0.25", episodes=20000, seed=2026):
    """解析真值与抽样实验分开执行；theta / p 从头到尾保持同一版本。"""
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    data = _plain(exact_analysis(theta, gamma, behavior))
    parameters = data["parameters"]
    p, gamma, behavior = parameters["p"], parameters["gamma"], parameters["behavior"]
    analytic = parameters["analytic"]
    parameters.update({"episodes": episodes, "seed": seed})

    # 两条可复现但独立的随机流。前者从 pi 产生轨迹，后者从 b 产生轨迹。
    # fullIS 和 singleActionRatio 使用完全相同的行为经验，只改变纠偏式。
    target_rng, behavior_rng = random.Random(seed), random.Random(seed + 1)
    estimates = {name: RunningMean() for name in ("onPolicy", "fullIS", "singleActionRatio")}
    counts = {name: {str(a0) + str(a1): 0 for a0, a1 in itertools.product((0, 1), repeat=2)}
              for name in ("onPolicy", "behavior")}
    checkpoints, checkpoint_counts = [], _checkpoint_counts(episodes)
    for number in range(1, episodes + 1):
        target_actions = sample_episode(target_rng, p)
        behavior_actions = sample_episode(behavior_rng, behavior)
        counts["onPolicy"]["".join(map(str, target_actions))] += 1
        counts["behavior"]["".join(map(str, behavior_actions))] += 1
        target_gradient, _, _ = episode_estimates(target_actions, p, gamma, behavior)
        _, full_is, single_action = episode_estimates(behavior_actions, p, gamma, behavior)
        estimates["onPolicy"].observe(target_gradient)
        estimates["fullIS"].observe(full_is)
        estimates["singleActionRatio"].observe(single_action)
        if number in checkpoint_counts:
            checkpoints.append({"episodes": number, **{name + "Mean": value.mean for name, value in estimates.items()}})

    data["sampling"] = {name: value.summary(analytic) for name, value in estimates.items()}
    data["sampling"].update({"counts": counts, "checkpoints": checkpoints,
                             "onPolicySeed": seed, "behaviorSeed": seed + 1})
    data["evidence"] = {"kind": "frozen-policy-gradient-estimation", "policyFrozen": True,
                        "updatesPerformed": 0, "trainingRun": False}
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="输出枚举与抽样数据，供独立核对或画图")
    parser.add_argument("--theta", default="0.7", help="冻结的共享 Bernoulli logit")
    parser.add_argument("--gamma", default="0.5", help="目标回报的折扣因子，0 <= gamma <= 1")
    parser.add_argument("--behavior", default="0.25", help="行为策略选择动作 1 的概率，0 < b < 1")
    parser.add_argument("--episodes", type=int, default=20000, help="每个数据源的轨迹数；theta 在整批中冻结")
    parser.add_argument("--seed", type=int, default=2026, help="目标随机流种子；行为流使用 seed + 1")
    args = parser.parse_args()
    try:
        data = calculate(args.theta, args.gamma, args.behavior, args.episodes, args.seed)
    except (ValueError, ArithmeticError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
        return
    print("Frozen two-step policy: gradient estimation, no policy updates or training")
    print("Parameters:", data["parameters"])
    print("Analytic / finite difference:", data["exact"]["analytic"], data["exact"]["finiteDifference"])
    print("Estimator                 exact mean          sampled mean           exact bias")
    for name in ("onPolicy", "fullIS", "singleActionRatio"):
        exact, sample = data["exact"][name], data["sampling"][name]
        print("{:23s} {: .10f}        {: .10f}        {: .10f}".format(name, exact["mean"], sample["mean"], exact["bias"]))
    print("Trajectory counts:", data["sampling"]["counts"])
    print("Single-action ratio omits the state-arrival / future-action distribution corrections.")
    print("After this batch, propose_update(theta, gradient, step_size) can form a separate, unapplied update.")


if __name__ == "__main__":
    main()
