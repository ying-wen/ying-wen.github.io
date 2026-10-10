#!/usr/bin/env python3
"""Two prediction diagnostics; Python standard library only.

SPDX-License-Identifier: MIT
Copyright (c) 2026 Ying Wen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Run:
    python3 rlss_prediction_diagnostics.py test
    python3 rlss_prediction_diagnostics.py run --output results.json

Experiment 1 is a deterministic, fully specified two-transition episode.
Experiment 2 samples a scalar follow-on trace, not a trained ETD critic.
Neither experiment establishes a general performance ranking of TD and ETD.
All figure series are returned as data. No plotting package is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys
import unittest


def validate_shared(alpha, episodes):
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Use 0 < alpha < 1 so both TD and ETD episode maps contract.")
    if not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer")


def validate_trace(gamma, probability):
    if not math.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    if not math.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("behavior probability must be finite and in (0, 1)")


# BEGIN shared_feature
def shared_episode(theta, alpha, emphatic=False):
    """Online A -> B -> terminal; r=1 at both transitions, x(A)=x(B)=1.

    At A: gamma_current=0 resets F to 1; gamma_next=1 bootstraps B.
    At B: gamma_current=1 makes F=2; gamma_next=0 stops bootstrapping.
    The two steps use their own CURRENT theta. This timing matters.
    """
    records = []
    followon = 0.0
    for state, gamma_current, gamma_next in (("A", 0.0, 1.0), ("B", 1.0, 0.0)):
        followon = 1.0 + gamma_current * followon  # rho_previous=1, interest=1
        emphasis = followon if emphatic else 1.0
        old_theta = theta
        delta = 1.0 + gamma_next * old_theta - old_theta
        theta = old_theta + alpha * emphasis * delta  # ETD(0), rho_current=1
        records.append({"state": state, "before": old_theta, "after": theta,
                        "delta": delta, "emphasis": emphasis})
    return theta, records


def frozen_episode(theta, alpha, emphatic=False):
    """Known-model diagnostic, NOT an online two-transition implementation.

    Both residuals are evaluated at the SAME old theta, then summed.
    There is no division by two: alpha multiplies the whole episode direction.
    """
    delta_a = 1.0 + theta - theta
    delta_b = 1.0 - theta
    emphasis_b = 2.0 if emphatic else 1.0
    return theta + alpha * (delta_a + emphasis_b * delta_b)


def shared_closed_form(episode, alpha, emphatic=False, online=True, initial=0.0):
    """Independent analytic trajectory for the episode-END parameter."""
    equilibrium = 1.5 if emphatic else 2.0
    if online:
        equilibrium -= alpha
    contraction = 1.0 - (2.0 if emphatic else 1.0) * alpha
    return equilibrium + (initial - equilibrium) * contraction ** episode
# END shared_feature


def uniform_msve(theta):
    """Frozen critic evaluated at BOTH true values; no extra training data."""
    return ((2.0 - theta) ** 2 + (1.0 - theta) ** 2) / 2.0


def shared_feature_experiment(alpha=0.1, episodes=200):
    validate_shared(alpha, episodes)
    methods = (("td_online", False, True), ("etd_online", True, True),
               ("td_frozen", False, False), ("etd_frozen", True, False))
    trajectories, final, cycles = {}, {}, {}
    max_error = 0.0
    for name, emphatic, online in methods:
        theta = 0.0
        rows = [{"episode": 0, "theta": theta, "uniform_msve": uniform_msve(theta)}]
        cycle = []
        cycle_start = max(0, episodes - 4)
        for episode in range(1, episodes + 1):
            if online:
                theta, events = shared_episode(theta, alpha, emphatic)
                if episode > cycle_start:
                    if not cycle:
                        cycle.append({"x": 2 * (episode - 1), "y": events[0]["before"]})
                    cycle.extend({"x": 2 * (episode - 1) + j + 1, "y": event["after"]}
                                 for j, event in enumerate(events))
            else:
                theta = frozen_episode(theta, alpha, emphatic)
            predicted = shared_closed_form(episode, alpha, emphatic, online)
            max_error = max(max_error, abs(theta - predicted))
            rows.append({"episode": episode, "theta": theta,
                         "uniform_msve": uniform_msve(theta)})
        trajectories[name] = rows
        final[name] = {**rows[-1], "exact_finite_episode_theta": predicted,
                       "limit_episode_end_theta": (1.5 if emphatic else 2.0) - (alpha if online else 0.0)}
        if online:
            cycles[name] = cycle
            final[name]["last_after_a"] = events[0]["after"]
            final[name]["last_after_b"] = events[1]["after"]
    return {"trajectories": trajectories, "cycles": cycles, "final": final,
            "closed_form_max_absolute_error": max_error}


# BEGIN followon
def followon_step(previous_followon, previous_ratio, gamma, interest=1.0):
    """Compute incoming F_t. The ratio belongs to action t-1, NOT action t."""
    return interest + gamma * previous_ratio * previous_followon


def finite_trace_moments(time, gamma, probability):
    """Exact moments from F_0=1 and iid previous rho in {0, 1/p}.

    This is deterministic moment propagation, not a sampled trace or a critic.
    The caller uses small time values for second moments to avoid float overflow.
    """
    mean, second = 1.0, 1.0
    for _ in range(time):
        second = 1.0 + 2.0 * gamma * mean + gamma * gamma / probability * second
        mean = 1.0 + gamma * mean
    return mean, second


def sample_trace_batch(seed, paths, horizon, gamma, probability, checkpoints):
    """Independent one-state trajectories, with both actions self-looping.

    pi(a)=1 and b(a)=p. At every action: rho=1/p with probability p, else 0.
    Every path is retained. No clipping, outlier removal, or stopping rule.
    Samples at DIFFERENT times within a path are not independent replicates.
    """
    rng = random.Random(seed)
    values = [1.0] * paths  # F_0=1; no sampled previous action yet
    rows = []
    selected = set(checkpoints)
    target_action_count = 0
    for time in range(horizon + 1):
        if time in selected:
            rows.append({"time": time, **describe(values)})
        if time == horizon:
            break
        for path in range(paths):
            chose_target = rng.random() < probability
            target_action_count += int(chose_target)
            previous_ratio = 1.0 / probability if chose_target else 0.0
            values[path] = followon_step(values[path], previous_ratio, gamma)
    return {"seed": seed, "checkpoints": rows, "target_action_count": target_action_count,
            "actions": paths * horizon, "final_values": values}
# END followon


def describe(values):
    ordered = sorted(values)
    count = len(ordered)
    if not count:
        raise ValueError("Cannot summarize an empty sample")
    # Nearest-rank empirical quantiles, NOT intervals for the population mean.
    quantile = lambda q: ordered[max(0, math.ceil(q * count) - 1)]
    return {"n": count, "mean": math.fsum(ordered) / count,
            "median": quantile(0.5), "q95": quantile(0.95), "q99": quantile(0.99),
            "maximum": ordered[-1], "minimum": ordered[0]}


# BEGIN tail_mass
def stationary_tail(streak, gamma, probability):
    """Exact tail probability and share of the stationary follow-on mean.

    K counts consecutive target actions immediately BEFORE the current state.
    P(K >= k)=p^k and F=sum_{j=0}^K (gamma/p)^j.
    Swapping these nonnegative sums gives
      E[F * 1{K>=k}] = sum_{j<k} p^(k-j)*gamma^j + gamma^k/(1-gamma).
    This avoids division by gamma/p-1, including the gamma=p case.
    """
    validate_trace(gamma, probability)
    if not isinstance(streak, int) or streak < 0:
        raise ValueError("streak must be a nonnegative integer")
    prefix = math.fsum(probability ** (streak - j) * gamma ** j for j in range(streak))
    contribution = prefix + gamma ** streak / (1.0 - gamma)
    return {"streak": streak, "probability": probability ** streak,
            "mean_contribution": contribution,
            "fraction_of_mean": contribution * (1.0 - gamma)}
# END tail_mass


def followon_experiment(paths=2048, horizon=512, seeds=(0, 1, 2, 3), gamma=0.99, probability=1.0 / 7.0):
    validate_trace(gamma, probability)
    if not isinstance(paths, int) or paths < 1 or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("paths and horizon must be positive integers")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty sequence without duplicates")
    # Dense early checkpoints and regular later ones; never chosen from outcomes.
    checkpoints = sorted(set(range(min(33, horizon + 1))) | set(range(40, horizon + 1, 8)) | {horizon})
    batches = [sample_trace_batch(seed, paths, horizon, gamma, probability, checkpoints) for seed in seeds]
    final_values = [value for batch in batches for value in batch["final_values"]]
    pooled = describe(final_values)
    # Store summaries rather than the full pseudo-random stream; seeds reproduce it.
    for batch in batches:
        del batch["final_values"]
    expected = [{"x": time, "y": (1.0 - gamma ** (time + 1)) / (1.0 - gamma)} for time in checkpoints]
    # Only t<=20 here: second moments grow exponentially and are not finite-variance CIs.
    moment_checks = [{"time": t, "mean": finite_trace_moments(t, gamma, probability)[0],
                      "second_moment": finite_trace_moments(t, gamma, probability)[1]} for t in (0, 1, 5, 10, 20)]
    return {"batches": batches, "pooled_final": pooled, "exact_finite_means": expected,
            "short_horizon_moments": moment_checks,
            "stationary_mean": 1.0 / (1.0 - gamma),
            "second_moment_multiplier": gamma * gamma / probability,
            "tail": [stationary_tail(k, gamma, probability) for k in range(13)]}


def series(name, points, role="estimate"):
    return {"name": name, "role": role, "points": points}


def make_report(alpha=0.1, episodes=200, paths=2048, horizon=512,
                seeds=(0, 1, 2, 3), gamma=0.99, probability=1.0 / 7.0):
    shared = shared_feature_experiment(alpha, episodes)
    followon = followon_experiment(paths, horizon, seeds, gamma, probability)
    trajectory_names = {"td_online": "TD：逐步在线", "etd_online": "ETD：逐步在线",
                        "td_frozen": "TD：冻结参数整回合方向", "etd_frozen": "ETD：冻结参数整回合方向"}
    shared_series = [series(trajectory_names[name], [{"x": row["episode"], "y": row["theta"]} for row in rows])
                     for name, rows in shared["trajectories"].items()]
    shared_series.extend(series(name, [{"x": 0, "y": value}, {"x": episodes, "y": value}], "reference")
                         for name, value in (("TD 平均固定点 2", 2.0), ("ETD 平均固定点 1.5", 1.5)))
    cycles = [series(trajectory_names[name], points) for name, points in shared["cycles"].items()]
    ensemble = [series("精确有限时刻 E[F_t]", followon["exact_finite_means"], "reference")]
    ensemble.extend(series("独立批次 seed=" + str(batch["seed"]),
                           [{"x": row["time"], "y": row["mean"]} for row in batch["checkpoints"]])
                    for batch in followon["batches"])
    charts = [
        {"id": "shared-feature-fixed-points", "title": "在线轨迹与冻结参数平均方向不是同一条曲线",
         "xLabel": "回合数（每回合 2 次真实转移）", "yLabel": "回合末共享参数 θ", "xScale": "linear", "yScale": "linear",
         "series": shared_series},
        {"id": "shared-feature-cycle", "title": "最后四个回合仍存在常数步长周期",
         "xLabel": "累计转移数（奇数=A 更新后，偶数=B 更新后）", "yLabel": "更新后的共享参数 θ",
         "xScale": "linear", "yScale": "linear", "series": cycles},
        {"id": "followon-ensemble", "title": "精确期望与有限路径均值：每个批次都保留全部路径",
         "xLabel": "已采样前序动作数 t（F₀=1）", "yLabel": "Follow-on trace F 的均值（对数轴）",
         "xScale": "linear", "yScale": "log", "series": ensemble},
        {"id": "followon-tail-mass", "title": "极少见的长串可以贡献大部分期望",
         "xLabel": "连续目标动作至少 k 次", "yLabel": "占比（对数轴；稳态解析值）", "xScale": "linear", "yScale": "log",
         "series": [series("该事件的概率", [{"x": row["streak"], "y": row["probability"]} for row in followon["tail"]], "reference"),
                    series("该事件贡献的均值占比", [{"x": row["streak"], "y": row["fraction_of_mean"]} for row in followon["tail"]], "reference")]}
    ]
    report = {
        "id": "rlss-prediction", "title": "TD、ETD 与强调权重：两个可手算的诊断实验",
        "scope": "机制诊断与有限步长/采样分布核验；不是原论文性能复现，也不提供一般算法排序。",
        "source": {"download": "/zh/continual-rl/download/rlss_prediction_diagnostics.py",
                   "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "license": "MIT", "python": sys.version.split()[0]},
        "protocol": {
            "shared_feature": {"environment": "A→B→terminal；两步奖励均为1；x(A)=x(B)=1；每回合从A重启",
                               "true_values": {"A": 2.0, "B": 1.0}, "initial_theta": 0.0,
                               "alpha": alpha, "episodes": episodes, "environment_transitions_per_online_method": 2 * episodes,
                               "gamma_on_arrival": {"A": 0, "B": 1, "terminal": 0}, "interest": 1, "lambda": 0, "rho": 1,
                               "evaluation": "每回合末冻结θ；uniform_msve为对A/B真值的均匀平方误差",
                               "frozen_control": "已知模型下同时算两项误差并求和；不是两步在线经验轨迹，不除以2",
                               "uncertainty": "完全确定，无随机种子平均、无置信阴影；回合内周期不是随机误差条"},
            "followon": {"environment": "单状态；两个动作均自循环；目标总选a；行为每步独立选a，概率p；不训练critic",
                         "behavior_target_action_probability": probability, "importance_ratio_support": [0.0, 1.0 / probability],
                         "gamma": gamma, "interest": 1.0, "initial_F0": 1.0, "paths_per_batch": paths,
                         "horizon": horizon, "batch_seeds": list(seeds), "sampled_actions_total": paths * horizon * len(seeds),
                         "variance_convention": "跨独立路径在同一时刻计算；不同时间点不当作独立重复",
                         "quantiles": "nearest rank ceil(q*n)-1；是路径分位数，不是总体均值置信区间",
                         "clipping": False, "outlier_removal": False, "selected_seeds": False,
                         "uncertainty": "展示预先指定的每个批次，不画置信阴影；有限时刻的矩有限，但可能极大",
                         "stationary_tail": "解析稳态几何长串；不假定有限模拟已经采到或准确估计极端尾部"}},
        "checks": {"shared_closed_form_max_absolute_error": shared["closed_form_max_absolute_error"],
                   "shared_closed_form_agrees": shared["closed_form_max_absolute_error"] < 1e-11,
                   "second_moment_multiplier": followon["second_moment_multiplier"],
                   "stationary_finite_second_moment_impossible": followon["second_moment_multiplier"] >= 1.0,
                   "all_trajectories_retained": True, "all_numbers_finite": True},
        "measurements": {"shared_feature": shared, "followon": followon}, "charts": charts,
        "sources": [{"url": "https://arxiv.org/html/1507.01569", "locator": "equations (10)-(13)",
                     "note": "普通 accumulating ETD 的 F/M/e；本实验不是 true-online ETD。"},
                    {"url": "https://arxiv.org/abs/1503.04269", "locator": "stability versus convergence and follow-on weighting",
                     "note": "固定线性设定下的分析；本实验只核验受控小问题。"}],
        "limitations": ["同策略实验只有一个参数与确定性两步回合；不能外推到神经逼近或控制。",
                        "冻结参数方向使用已知模型，不是另一个同预算在线算法。",
                        "第二实验仅采样follow-on，未训练价值函数；trace重尾不能单独证明ETD参数发散。",
                        "固定时刻的样本均值无偏；某次模拟低于精确期望，不证明均值估计器有系统偏差。",
                        "无限稳态二阶矩与每个有限时刻的有限二阶矩必须区分。",
                        "沒有裁剪、调参搜索、seed筛选、任务漂移或表示学习。"]
    }
    # Reject NaN/Infinity rather than serializing invalid or misleading JSON.
    json.dumps(report, allow_nan=False)
    return report


class DiagnosticTests(unittest.TestCase):
    def test_online_timing(self):
        td, a = shared_episode(0.0, 0.1)
        etd, b = shared_episode(0.0, 0.1, True)
        self.assertAlmostEqual(td, 0.19)
        self.assertAlmostEqual(etd, 0.28)
        self.assertEqual([r["emphasis"] for r in a], [1.0, 1.0])
        self.assertEqual([r["emphasis"] for r in b], [1.0, 2.0])
        self.assertAlmostEqual(frozen_episode(0, 0.1), 0.2)
        self.assertAlmostEqual(frozen_episode(0, 0.1, True), 0.3)

    def test_all_four_closed_forms_and_cycle(self):
        for alpha in (0.01, 0.1, 0.7):
            result = shared_feature_experiment(alpha, 120)
            self.assertLess(result["closed_form_max_absolute_error"], 2e-14)
            for emphatic in (False, True):
                end = (1.5 if emphatic else 2.0) - alpha
                observed, events = shared_episode(end, alpha, emphatic)
                self.assertAlmostEqual(observed, end)
                self.assertAlmostEqual(events[0]["after"], end + alpha)

    def test_uniform_msve_is_not_a_sampled_learning_loss(self):
        self.assertEqual(uniform_msve(2), 0.5)
        self.assertEqual(uniform_msve(1.5), 0.25)
        self.assertAlmostEqual(uniform_msve(1.4), 0.26)

    def test_moments_against_exhaustive_action_sequences(self):
        gamma, probability, time = 0.9, 0.2, 8
        first, second, mass = [], [], []
        for actions in itertools.product((False, True), repeat=time):
            value, weight = 1.0, 1.0
            for chose_target in actions:
                ratio = 1.0 / probability if chose_target else 0.0
                value = followon_step(value, ratio, gamma)
                weight *= probability if chose_target else 1.0 - probability
            first.append(weight * value)
            second.append(weight * value * value)
            mass.append(weight)
        m, m2 = finite_trace_moments(time, gamma, probability)
        self.assertAlmostEqual(math.fsum(mass), 1.0)
        self.assertAlmostEqual(math.fsum(first), m)
        self.assertAlmostEqual(math.fsum(second) / m2, 1.0)

    def test_trace_previous_ratio_and_reset(self):
        ratios = [1, 2, 2, 2, 0.25, 1, 0]
        values = [1.0]
        for ratio in ratios:
            values.append(followon_step(values[-1], ratio, 1.0, interest=0.0))
        self.assertEqual(values, [1, 1, 2, 4, 8, 2, 2, 0])

    def test_tail_identity_from_direct_stationary_sum(self):
        # gamma small here so the omitted tail is <1e-15 even in first moments.
        for gamma, probability in ((0.2, 0.3), (0.3, 0.3), (0.0, 0.3)):
            for k in (0, 1, 4):
                terms = []
                for length in range(k, 100):
                    value = sum((gamma / probability) ** j for j in range(length + 1))
                    terms.append((1 - probability) * probability ** length * value)
                tail = stationary_tail(k, gamma, probability)
                self.assertAlmostEqual(math.fsum(terms), tail["mean_contribution"], places=13)
                self.assertGreaterEqual(tail["fraction_of_mean"], 0.0)
                self.assertLessEqual(tail["fraction_of_mean"], 1.0 + 1e-14)

    def test_finite_mean_and_infinite_second_moment_condition(self):
        gamma, probability = 0.99, 1.0 / 7.0
        coefficient = gamma * gamma / probability
        self.assertAlmostEqual(1.0 / (1 - gamma), 100.0)
        self.assertAlmostEqual(coefficient, 6.8607)
        impossible_second = (1 + 2 * gamma / (1 - gamma)) / (1 - coefficient)
        self.assertLess(impossible_second, 0)  # contradicts a finite nonnegative E[F²]

    def test_quantiles_are_nearest_rank_not_confidence_intervals(self):
        stats = describe([1, 2, 100, 3])
        self.assertEqual(stats["median"], 2)
        self.assertEqual(stats["q95"], 100)
        self.assertEqual(stats["mean"], 26.5)

    def test_sampling_reproducible_without_seed_selection(self):
        args = (31, 32, 12, 0.9, 1.0 / 7.0, [0, 1, 12])
        self.assertEqual(sample_trace_batch(*args), sample_trace_batch(*args))
        result = followon_experiment(16, 12, (0, 1), 0.9, 1.0 / 7.0)
        self.assertEqual(result["pooled_final"]["n"], 32)
        self.assertEqual([b["seed"] for b in result["batches"]], [0, 1])

    def test_inputs_and_serialization(self):
        for bad in (0, 1, float("nan")):
            with self.assertRaises(ValueError):
                shared_feature_experiment(bad, 5)
        with self.assertRaises(ValueError):
            followon_experiment(seeds=(0, 0))
        with self.assertRaises(ValueError):
            followon_experiment(gamma=1)
        report = make_report(0.1, 10, 8, 10, (0, 1))
        self.assertEqual(len(report["charts"]), 4)
        self.assertEqual(json.loads(json.dumps(report, allow_nan=False))["id"], "rlss-prediction")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("test", help="run independent analytic and timing checks")
    run = commands.add_parser("run", help="run both diagnostics and write a JSON file")
    run.add_argument("--output", type=Path, required=True, help="JSON output file, not a directory")
    run.add_argument("--alpha", type=float, default=0.1)
    run.add_argument("--episodes", type=int, default=200)
    run.add_argument("--paths", type=int, default=2048, help="independent paths per seed batch")
    run.add_argument("--horizon", type=int, default=512)
    run.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    run.add_argument("--gamma", type=float, default=0.99)
    run.add_argument("--probability", type=float, default=1.0 / 7.0)
    args = parser.parse_args(argv)
    if args.command == "test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(DiagnosticTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    try:
        report = make_report(args.alpha, args.episodes, args.paths, args.horizon,
                             tuple(args.seeds), args.gamma, args.probability)
    except ValueError as error:
        parser.error(str(error))
    if args.output.resolve() == Path(__file__).resolve():
        parser.error("output cannot overwrite the source file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "checks": report["checks"],
                      "shared_final": report["measurements"]["shared_feature"]["final"],
                      "followon_final": report["measurements"]["followon"]["pooled_final"]},
                     ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
