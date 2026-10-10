# SPDX-License-Identifier: MIT
"""A five-state diagnostic of reward-respecting options and model versions.

Python 3.10+, standard library only:
  python rlss_oak_diagnostics.py test
  python rlss_oak_diagnostics.py run --output results.json

This is an original teaching task, NOT an OaK/STOMP implementation or benchmark
reproduction. Features, a two-step horizon, and the reference policy are given.
The option's internal policy and stopping choice are solved exactly. Its outcome
model is either exact or fitted to complete Monte Carlo option executions.
No feature discovery, model-based option learning, or lifelong training claim.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from statistics import fmean, stdev


# BEGIN environment
STATES = ("H", "J", "A", "B", "C")
# (successor, expected EXTERNAL reward). One transition takes one time unit.
# Observed rewards add an independent fair {-1, +1} noise; transitions are exact.
WORLD = {
    "H": {"advance": ("J", 0.0)},
    "J": {"usual": ("A", 3.0), "deferred": ("B", 0.0),
          "feature": ("C", 0.0)},
    "A": {"return": ("H", 0.0)},
    "B": {"return": ("H", 4.0)},
    "C": {"return": ("H", 0.0)},
}
MU = {"H": "advance", "J": "usual", "A": "return",
      "B": "return", "C": "return"}
PHI = {"H": 0.0, "J": 1.0, "A": 0.0, "B": 0.0, "C": 2.0}
HORIZON = 2  # A design restriction, not a discovered stopping guarantee.


def linear_solve(matrix, rhs):
    """Small dense Gaussian elimination with pivoting; no numerical dependency."""
    a = [list(map(float, row)) + [float(y)] for row, y in zip(matrix, rhs)]
    n = len(a)
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(a[i][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("singular reference-policy equations")
        a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        a[col] = [x / scale for x in a[col]]
        for row in range(n):
            if row != col:
                factor = a[row][col]
                a[row] = [x - factor * y for x, y in zip(a[row], a[col])]
    return [row[-1] for row in a]


def gain_bias(policy):
    """Solve g + h(s) = r(s,pi(s)) + h(next), with gauge h(H)=0.

    Every considered policy has a single recurrent cycle reached from all states.
    This uses the Poisson/regenerative bias definition: centered reward up to H.
    Periodic chains need not admit an ordinary infinite-sum limit. No discounted
    value is substituted for this bias.
    """
    n = len(STATES)
    matrix, rewards = [], []
    for i, state in enumerate(STATES):
        successor, reward = WORLD[state][policy[state]]
        row = [0.0] * (n + 1)
        row[i] += 1.0
        row[STATES.index(successor)] -= 1.0
        row[n] = 1.0
        matrix.append(row)
        rewards.append(reward)
    matrix.append([1.0] + [0.0] * n)
    rewards.append(0.0)
    solution = linear_solve(matrix, rewards)
    return solution[n], dict(zip(STATES, solution[:n]))


GAIN, BIAS = gain_bias(MU)  # Freeze these for all subproblem and planning tests.
# END environment


# BEGIN option
@dataclass
class Option:
    kappa: float
    with_terminal_bias: bool
    value: float
    decisions: dict


def solve_option(kappa, with_terminal_bias=True):
    """Exact finite-horizon DP over (state, remaining time).

    Stop is allowed after the first action. At the horizon it is compulsory.
    A stop adds h_mu(s) + kappa*phi(s) ONCE. A continuation adds only R-g_mu.
    The stop action wins ties. This yields beta in {0,1}; it is not sampled TD.
    """
    if kappa < 0 or not math.isfinite(kappa):
        raise ValueError("kappa must be finite and nonnegative")
    decisions = {}

    def terminal(state):
        return (BIAS[state] if with_terminal_bias else 0.0) + kappa * PHI[state]

    @lru_cache(None)
    def value(state, remaining, may_stop):
        if remaining == 0:
            decisions[state, remaining, may_stop] = "stop"
            return terminal(state)
        best = terminal(state) if may_stop else -math.inf
        choice = "stop" if may_stop else None
        for action, (successor, reward) in WORLD[state].items():
            candidate = reward - GAIN + value(successor, remaining - 1, True)
            if candidate > best + 1e-12:
                best, choice = candidate, action
        decisions[state, remaining, may_stop] = choice
        return best

    optimum = value("H", HORIZON, False)
    return Option(kappa, with_terminal_bias, optimum, decisions)


def execute(option, rng=None):
    """A complete option execution. RNG=None gives the exact mean trajectory.

    The world never terminates. The return transition from the endpoint is NOT
    included in this sample: it is represented by the frozen terminal bias.
    """
    state, remaining, total, duration = "H", HORIZON, 0.0, 0
    while True:
        action = option.decisions[state, remaining, duration > 0]
        if action == "stop":
            return total, duration, state
        state, reward = WORLD[state][action]
        if rng is not None:
            reward += 1.0 if rng.random() < 0.5 else -1.0
        total += reward
        duration += 1
        remaining -= 1
# END option


# BEGIN model
@dataclass
class OutcomeModel:
    """MC sufficient statistics: E[external reward], E[duration], P(endpoint).

    Kappa, feature bonuses, and terminal value NEVER enter the reward field.
    A model version is part of the protocol; changing an option changes targets.
    """
    count: int = 0
    reward_sum: float = 0.0
    duration_sum: float = 0.0
    endpoints: dict = field(default_factory=lambda: {s: 0 for s in STATES})

    def observe(self, sample):
        reward, duration, endpoint = sample
        if duration < 1 or endpoint not in self.endpoints:
            raise ValueError("only complete positive-duration option samples")
        self.count += 1
        self.reward_sum += reward
        self.duration_sum += duration
        self.endpoints[endpoint] += 1

    def target(self):
        """One option then return to mu: Rbar - g_mu*tau_bar + Pbar*h_mu.

        h_mu(H)=0, so this also equals its deviation advantage at H.
        This scalar is NOT the new policy's long-run reward rate.
        """
        if not self.count:
            raise ValueError("unobserved model: fall back to the primitive policy")
        return (self.reward_sum - GAIN * self.duration_sum
                + sum(self.endpoints[s] * BIAS[s] for s in STATES)) / self.count

    def record(self):
        if not self.count:
            raise ValueError("no moments without data")
        return {"count": self.count, "external_reward": self.reward_sum / self.count,
                "duration": self.duration_sum / self.count,
                "endpoint_probabilities": {s: self.endpoints[s] / self.count for s in STATES},
                "main_backup": self.target()}


def exact_model(option):
    result = OutcomeModel()
    result.observe(execute(option))
    return result


def repeated_policy_gain(option):
    """Independently evaluate a repeated high-level choice by a regenerative cycle.

    At H execute this option (or one mu action if None); thereafter follow mu
    until H. Repeat. Sum expected external reward / elapsed primitive time.
    Neither the kappa bonus nor any bias term belongs to this true gain.
    """
    if option is None:
        state, total = WORLD["H"][MU["H"]]
        duration = 1
    else:
        total, duration, state = execute(option)
    while state != "H":
        state, reward = WORLD[state][MU[state]]
        total += reward
        duration += 1
    return total / duration


def planned_gain(model, current_option):
    # Compare the option with the one-step mu backup at H, exactly zero here.
    # Ties favor the primitive baseline. Evaluator, not agent, knows the true gain.
    return repeated_policy_gain(current_option if model.target() > 0.0 else None)
# END model


def self_test():
    """Independent enumeration and policy equations, not only self-consistency."""
    close = lambda a, b: math.isclose(a, b, rel_tol=0.0, abs_tol=1e-10)
    assert close(GAIN, 1.0)
    assert all(close(BIAS[s], h) for s, h in zip(STATES, (0, 1, -1, 3, -1)))
    residual = max(abs(GAIN + BIAS[s] - r - BIAS[n])
                   for s in STATES for n, r in [WORLD[s][MU[s]]])
    assert residual < 1e-12
    # Four possible paths; coefficients derived by hand from WORLD and h_mu.
    lines = {"J": (0.0, 1.0), "A": (0.0, 0.0),
             "B": (1.0, 0.0), "C": (-3.0, 2.0)}
    max_error = 0.0
    for k in [i / 20 for i in range(121)]:
        option = solve_option(k)
        candidate_values = {s: a + k * b for s, (a, b) in lines.items()}
        expected = max(candidate_values, key=candidate_values.get)
        sample = execute(option)
        assert sample[2] == expected  # Includes stop-first ties at k=1 and k=3.
        max_error = max(max_error, abs(option.value - candidate_values[expected]))
        model = exact_model(option)
        assert close(option.value, model.target() + k * PHI[sample[2]])
        # A bias gauge shift adds the same constant to all subproblem returns.
        # Raw backups also shift; subtract h(H) to recover invariant advantage.
        offset = 7.25
        shifted_values = {s: v + offset for s, v in candidate_values.items()}
        assert max(shifted_values, key=shifted_values.get) == expected
        reward, duration, endpoint = sample
        shifted_backup = reward - GAIN*duration + BIAS[endpoint] + offset
        assert close(shifted_backup - (BIAS["H"] + offset), model.target() - BIAS["H"])
        # Independent gain check via the new primitive policy's Poisson equations.
        policy = dict(MU)
        if sample[2] not in ("J", "A"):
            policy["J"] = "deferred" if sample[2] == "B" else "feature"
        actual_gain, _ = gain_bias(policy)
        assert close(actual_gain, repeated_policy_gain(option))
    assert max_error < 1e-12
    old, stop, new = [solve_option(k) for k in (0, 2, 4)]
    assert [execute(o)[1:] for o in (old, stop, new)] == [(2, "B"), (1, "J"), (2, "C")]
    assert [exact_model(o).target() for o in (old, stop, new)] == [1.0, 0.0, -3.0]
    assert execute(solve_option(0, False))[2] == "A"
    assert close(repeated_policy_gain(old), 4 / 3)
    assert close(repeated_policy_gain(new), 0)
    assert close(repeated_policy_gain(None), 1)
    assert close(planned_gain(exact_model(old), new), 0)
    assert close(planned_gain(exact_model(new), new), 1)
    # Exact mixing: 64 old endpoints B and 32 new endpoints C predict -1/3.
    mixture = OutcomeModel()
    for _ in range(64):
        mixture.observe(execute(old))
    for _ in range(32):
        mixture.observe(execute(new))
    assert close(mixture.target(), -1 / 3)
    assert close(sum(mixture.record()["endpoint_probabilities"].values()), 1)
    # Enumerate all four equally likely two-step reward-noise sequences.
    class ScriptedNoise:
        def __init__(self, values):
            self.values = iter(values)

        def random(self):
            return next(self.values)

    noisy = OutcomeModel()
    for first in (0.25, 0.75):
        for second in (0.25, 0.75):
            sample = execute(new, ScriptedNoise((first, second)))
            assert sample[1:] == (2, "C")
            assert sample[0] in (-2.0, 0.0, 2.0)
            noisy.observe(sample)
    assert close(noisy.record()["external_reward"], 0.0)
    assert close(noisy.target(), -3.0)
    try:
        OutcomeModel().observe((0, 0, "H"))
        raise AssertionError("zero duration was accepted")
    except ValueError:
        pass
    return {"all_passed": True, "reference_poisson_residual": residual,
            "subproblem_grid_max_error": max_error, "subproblem_grid_points": 121,
            "exact_stale_backup": 1.0, "actual_new_backup": -3.0,
            "gain_stale_selection": 0.0, "gain_fresh_selection": 1.0,
            "gain_old_option_before_change": 4 / 3,
            "same_version_four_noise_outcomes_backup": noisy.target(),
            "bias_gauge_invariance_checked": True,
            "mixed_model_64_old_32_new_backup": mixture.target()}


# BEGIN experiment
def experiment(seeds=64, old_samples=64):
    """After a declared kappa change, fit models with complete independent probes.

    The exact option solver has model access. MC estimators do not: they receive
    only (sum external reward, duration, endpoint). Every probe starts at H.
    This reset/generative diagnostic permission is NOT a strict single lifetime.
    Model refresh is notified by the deliberate option-version change; there is
    no claim of detecting an unknown environmental change.
    """
    if seeds < 2 or old_samples < 1:
        raise ValueError("at least two seeds for standard errors and one old probe")
    checks = self_test()
    before, after = solve_option(0), solve_option(4)
    checkpoints = (1, 2, 4, 8, 16, 24, 32, 64, 128, 256)
    raw = []
    for seed in range(seeds):
        rng = random.Random(1709 + seed)
        old = OutcomeModel()
        for _ in range(old_samples):
            old.observe(execute(before, rng))
        pooled, refreshed = copy.deepcopy(old), OutcomeModel()
        for n in range(1, checkpoints[-1] + 1):
            sample = execute(after, rng)  # Same new sample for both adapting models.
            pooled.observe(sample)
            refreshed.observe(sample)
            if n in checkpoints:
                row = {"seed": seed, "new_samples": n, "new_primitive_steps": 2*n}
                for name, model in (("stale", old), ("pooled", pooled), ("refreshed", refreshed)):
                    row[name] = {"backup": model.target(),
                                 "selected_policy_gain": planned_gain(model, after)}
                raw.append(row)

    summary = []
    for n in checkpoints:
        group = [row for row in raw if row["new_samples"] == n]
        entry = {"new_samples": n, "new_primitive_steps": 2*n}
        for name in ("stale", "pooled", "refreshed"):
            values = [row[name]["backup"] for row in group]
            gains = [row[name]["selected_policy_gain"] for row in group]
            entry[name] = {"mean_backup": fmean(values),
                           "backup_standard_error": stdev(values)/math.sqrt(seeds),
                           "mean_selected_policy_gain": fmean(gains),
                           "gain_standard_error": stdev(gains)/math.sqrt(seeds)}
        entry["exact_expected_pooled_backup"] = -3 + 4*old_samples/(old_samples+n)
        summary.append(entry)

    # Exact subproblem sweep, not a learning curve or an empirical success rate.
    kappas = [i/4 for i in range(25)]
    selected = []
    for k in kappas:
        option = solve_option(k)
        model = exact_model(option)
        selected.append({"kappa": k, "endpoint": execute(option)[2],
                         "duration": execute(option)[1], "subproblem_value": option.value,
                         "main_backup": model.target(), "repeated_policy_gain": repeated_policy_gain(option)})

    def curve(name, values, role="estimate", standard_errors=None):
        points = []
        for i, (x, y) in enumerate(values):
            point = {"x": x, "y": y}
            if standard_errors is not None:
                point.update(low=y-standard_errors[i], high=y+standard_errors[i])
            points.append(point)
        return {"name": name, "role": role,
                "points": points}

    labels = {"stale": "冻结旧模型", "pooled": "混合新旧样本", "refreshed": "分版本重估"}
    charts = [
        {"id": "subproblem-options", "title": "终端奖金改变技能与停止决策",
         "xLabel": "特征奖金强度 κ", "yLabel": "子问题期望回报 Jκ",
         "series": [curve("在 J 停止", [(k, k) for k in kappas]),
                    curve("经 A 后停止", [(k, 0.0) for k in kappas]),
                    curve("经 B 后停止", [(k, 1.0) for k in kappas]),
                    curve("经 C 后停止", [(k, -3+2*k) for k in kappas]),
                    curve("DP 选中的上包络", [(r["kappa"], r["subproblem_value"]) for r in selected], "reference")]},
        {"id": "model-backup-refresh", "title": "旧模型预测的是已经改变的技能",
         "xLabel": "新版本的完整 option 样本数", "yLabel": "相对冻结 μ 的主任务 backup",
         "intervalLabel": "阴影：均值 ± 1 标准误，不是置信区间",
         "series": [curve(labels[name], [(r["new_samples"], r[name]["mean_backup"]) for r in summary],
                          "negative" if name == "stale" else "estimate",
                          [r[name]["backup_standard_error"] for r in summary]) for name in labels]
                   + [curve("当前技能的精确值", [(n, -3.0) for n in checkpoints], "reference"),
                      curve("混合模型的期望", [(r["new_samples"], r["exact_expected_pooled_backup"]) for r in summary], "reference")]},
        {"id": "selection-gain", "title": "模型刷新修复选择，不修复技能本身",
         "xLabel": "新版本的完整 option 样本数", "yLabel": "所选再生策略的实际长期奖励率",
         "intervalLabel": "阴影：均值 ± 1 标准误，不是置信区间",
         "series": [curve(labels[name], [(r["new_samples"], r[name]["mean_selected_policy_gain"]) for r in summary],
                          "negative" if name == "stale" else "estimate",
                          [r[name]["gain_standard_error"] for r in summary]) for name in labels]
                   + [curve("冻结 μ 的奖励率", [(n, GAIN) for n in checkpoints], "reference")]},
    ]
    return {
        "id": "rlss-oak", "title": "差分子问题、技能后果与规划接口",
        "scope": "原创五状态机制诊断；精确 option 求解加 MC 后果拟合；不是完整 OaK 或作者 benchmark 复现。",
        "protocol": {
            "reference_policy": MU, "reference_gain": GAIN, "reference_bias": BIAS,
            "feature": PHI, "option_initiation": ["H"], "max_primitive_duration": HORIZON,
            "stop_tie_break": "stop", "external_reward_noise": "independent fair {-1,+1}",
            "option_before_kappa": 0, "option_after_kappa": 4, "seeds": seeds,
            "seed_values": list(range(1709, 1709+seeds)), "old_complete_probes_per_seed": old_samples,
            "new_complete_probes_per_seed": checkpoints[-1], "primitive_steps_per_probe": 2,
            "probe_permission": "independent restarts at H; probes charged explicitly, not a lifetime benchmark",
            "model_update": "sample means; frozen old, pooled old+new, or new-version-only",
            "planner": "one option deviation then frozen mu; compare with one primitive mu step at H",
            "evaluation": "repeat selected choice at H; follow mu elsewhere; solve true regenerative gain independently",
            "uncertainty": "curves are 64-seed arithmetic means; per-point standard errors and per-seed values included",
            "exact_permissions": "option DP and oracle evaluator know WORLD; MC models see complete outcomes only",
        },
        "checks": checks, "charts": charts,
        "option_versions": [{"kappa": k, "use_terminal_bias": use_bias,
                             "solution": "exact finite-horizon dynamic programming",
                             "decision_at_J": solve_option(k, use_bias).decisions["J", 1, True],
                             "model_source": "exact mean trajectory; count=1 is a synthetic oracle statistic, not an MC observation",
                             "subproblem_value": solve_option(k, use_bias).value,
                             "model": exact_model(solve_option(k, use_bias)).record(),
                             "true_repeated_gain": repeated_policy_gain(solve_option(k, use_bias))}
                            for k, use_bias in ((0, True), (2, True), (4, True), (0, False))],
        "subproblem_sweep": selected, "model_fit_summary": summary, "raw_runs": raw,
        "limitations": [
            "The option policy and beta are solved with an exact finite model, not learned from samples.",
            "The two-step cap, state, feature, main policy, and switch are supplied by the experimenter.",
            "All planners use fixed gain/bias; the selected policy gain is separately evaluated, not assumed optimal.",
            "Refresh receives an explicit option-version change; no autonomous change detection is tested.",
            "Only outcome reward is stochastic here; endpoints and durations are deterministic within each version.",
            "Probes reset to H and their costs are separate from policy-gain evaluation; this is not lifetime net utility.",
        ],
        "sources": ["https://arxiv.org/abs/2202.03466", "https://oaklab.ai/posts/the-oak-architecture"],
        "reproduce": "python3 rlss_oak_diagnostics.py run --output results.json",
    }
# END experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("test", help="exact independent algebra and interface checks")
    run = sub.add_parser("run", help="run the fixed diagnostic protocol")
    run.add_argument("--output", type=Path, required=True, help="JSON file to create")
    args = parser.parse_args()
    if args.command == "test":
        print(json.dumps(self_test(), ensure_ascii=False, indent=2, allow_nan=False))
    else:
        data = experiment()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "checks": data["checks"],
                          "charts": len(data["charts"]), "raw_rows": len(data["raw_runs"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
