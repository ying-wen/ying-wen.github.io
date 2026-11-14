"""A complete, standard-library experiment-design lab (Python 3.10+).

Code: MIT. Run `python experiment_design_lab.py demo --tiny` or `test`.
This is a constructed nonstationary-bandit experiment, not a paper benchmark.
No files are written. Use --raw to include all observed per-step rewards in JSON.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
import statistics
import unittest
from unittest.mock import patch


def finite(values):
    if not all(math.isfinite(x) for x in values):
        raise ValueError("all values must be finite")


def rng_for(seed: int, stream: str) -> random.Random:
    # Stable across Python processes; do not use Python's randomized hash().
    key = hashlib.sha256(f"{seed}:{stream}".encode()).digest()
    return random.Random(int.from_bytes(key[:16], "big"))


@dataclass(frozen=True)
class Config:
    method: str
    alpha: float
    epsilon: float = 0.1
    decay_scale: float = 20.0

    def __post_init__(self):
        finite([self.alpha, self.epsilon, self.decay_scale])
        if self.method not in {"constant", "decay"}:
            raise ValueError("unknown update family")
        if not 0 <= self.alpha <= 1 or not 0 <= self.epsilon <= 1:
            raise ValueError("alpha and epsilon must be in [0, 1]")
        if self.decay_scale <= 0:
            raise ValueError("decay_scale must be positive")


# BEGIN environment
@dataclass(frozen=True)
class Scenario:
    seed: int
    # Controller-only potential outcomes; the learner never receives this table.
    rewards: tuple[tuple[float, float], ...]
    action_uniforms: tuple[tuple[float, float, float], ...]


def make_scenario(seed: int, horizon: int) -> Scenario:
    if horizon < 3 or horizon % 3:
        raise ValueError("horizon must be a positive multiple of three")
    env_rng, action_rng = rng_for(seed, "environment"), rng_for(seed, "actions")
    rewards, uniforms = [], []
    phase_length = horizon // 3
    for t in range(horizon):
        probabilities = (0.2, 0.8) if t // phase_length == 1 else (0.8, 0.2)
        rewards.append(tuple(float(env_rng.random() < p) for p in probabilities))
        # Exactly three action uniforms per time index, regardless of branching.
        uniforms.append(tuple(action_rng.random() for _ in range(3)))
    return Scenario(seed, tuple(rewards), tuple(uniforms))


class Learner:
    def __init__(self, config: Config):
        self.config = config
        self.q = [0.5, 0.5]
        self.counts = [0, 0]

    def choose(self, uniforms):
        explore, arm, tie = uniforms
        if explore < self.config.epsilon:
            return min(1, int(2 * arm))
        best = max(self.q)
        choices = [a for a, value in enumerate(self.q) if value == best]
        return choices[min(len(choices) - 1, int(tie * len(choices)))]

    def update(self, action, reward):
        self.counts[action] += 1
        step = self.config.alpha
        if self.config.method == "decay":
            step /= 1 + (self.counts[action] - 1) / self.config.decay_scale
        self.q[action] += step * (reward - self.q[action])
        if not all(math.isfinite(value) for value in self.q):
            raise FloatingPointError("non-finite value estimate")
# END environment


@dataclass(frozen=True)
class Run:
    seed: int
    config: Config
    horizon: int
    observed_rewards: tuple[float, ...]
    actions: tuple[int, ...]
    utility: tuple[float, ...]
    final_q: tuple[float | None, float | None]
    final_counts: tuple[int, int]
    status: str
    failure: dict | None


# BEGIN lifetime
def run_lifetime(scenario: Scenario, config: Config, limit=None,
                 freeze_at=None, fault_at=None) -> Run:
    horizon = len(scenario.rewards) if limit is None else limit
    if not 1 <= horizon <= len(scenario.rewards):
        raise ValueError("invalid observation budget")
    for boundary in (freeze_at, fault_at):
        if boundary is not None and not 0 <= boundary < horizon:
            raise ValueError("boundary outside run")
    learner = Learner(config)
    observed, actions, failure = [], [], None
    for t in range(horizon):
        action = learner.choose(scenario.action_uniforms[t])
        reward = scenario.rewards[t][action]
        # Score behavior before learning. Unchosen rewards stay controller-only.
        observed.append(reward)
        actions.append(action)
        try:
            if t == fault_at:
                raise FloatingPointError("injected diagnostic fault")
            if freeze_at is None or t < freeze_at:
                learner.update(action, reward)
        except FloatingPointError as error:
            failure = {"after_step": t + 1, "reason": str(error),
                       "injected": fault_at is not None}
            break
    # Predeclared deployment utility: stopped service earns 0 for remaining time.
    # This tail is NOT observed reward; retain both arrays and the failure event.
    utility = tuple(observed + [0.0] * (horizon - len(observed)))
    # JSON has no non-finite numbers. Preserve the failure event and use null for
    # invalid final parameters; actual finite rewards and scores remain intact.
    final_q = tuple(value if math.isfinite(value) else None for value in learner.q)
    return Run(scenario.seed, config, horizon, tuple(observed), tuple(actions),
               utility, final_q, tuple(learner.counts),
               "failed" if failure else "complete", failure)
# END lifetime


def mean_reward(run: Run) -> float:
    return statistics.fmean(run.utility)


def quantile(values, probability):
    if not values or not 0 <= probability <= 1:
        raise ValueError("invalid quantile")
    finite(values)
    data = sorted(values)
    position = (len(data) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    return data[lo] + (position - lo) * (data[hi] - data[lo])


# BEGIN bootstrap
def paired_bootstrap(first, second, reps=2000, seed=917, confidence=0.95):
    """Percentile CI for a fixed pair of configurations, resampling whole runs."""
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("at least two aligned run pairs are required")
    if reps < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    finite(first + second)
    differences = [a - b for a, b in zip(first, second)]
    rng = random.Random(seed)
    n = len(differences)
    samples = [statistics.fmean(differences[rng.randrange(n)] for _ in range(n))
               for _ in range(reps)]
    tail = (1 - confidence) / 2
    return {"mean_difference": statistics.fmean(differences),
            "ci": [quantile(samples, tail), quantile(samples, 1 - tail)],
            "standard_error": statistics.stdev(differences) / math.sqrt(n),
            "pairs": n, "bootstrap_replicates": reps,
            "interval": "run-paired percentile; pointwise, conditional on selection"}
# END bootstrap


def unique_seeds(seeds):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("nonempty, unique run seeds required")


def recovery(run: Run, start: int, end: int, window=30,
             threshold=0.7, hold=2):
    """First reward-based confirmation, with failure distinct from censoring.

    The last observed reward precedes an update failure at the same step. A
    confirmation already reached using those rewards remains an event; otherwise
    a failure at the interval endpoint must not be labeled administrative censoring.
    Run-level status separately retains failures occurring after confirmation.
    """
    if not 0 <= start < end <= run.horizon or window < 1 or hold < 1:
        raise ValueError("invalid recovery interval")
    if not 0 <= threshold <= 1:
        raise ValueError("invalid threshold")
    streak = 0
    for stop in range(start + window, end + 1, window):
        if stop > len(run.observed_rewards):
            return {"status": "failed", "confirmation_delay": None}
        score = statistics.fmean(run.observed_rewards[stop - window:stop])
        streak = streak + 1 if score >= threshold else 0
        if streak >= hold:
            return {"status": "recovered", "confirmation_delay": stop - start}
    failed_by_end = (run.failure is not None and run.failure["after_step"] <= end)
    status = "failed" if len(run.observed_rewards) < end or failed_by_end else "right_censored"
    return {"status": status, "confirmation_delay": None}


def run_metrics(run, window):
    if (run.horizon % 3 or window < 1 or window > run.horizon // 3
            or (run.horizon // 3) % window):
        raise ValueError("metrics need three complete phases and windows dividing each phase")
    length = run.horizon // 3
    windows = [statistics.fmean(run.utility[t:t + window])
               for t in range(0, run.horizon, window)]
    return {"seed": run.seed, "status": run.status, "failure": run.failure,
            "observed_steps": len(run.observed_rewards),
            "lifetime_rate": mean_reward(run), "total_utility": sum(run.utility),
            "phase_rates": [statistics.fmean(run.utility[i * length:(i + 1) * length])
                            for i in range(3)],
            "window_rates": windows, "lowest_window_rate": min(windows),
            "recovery": [recovery(run, length, 2 * length, window),
                         recovery(run, 2 * length, 3 * length, window)]}


# BEGIN selection
def select_on_development(method, candidates, dev_seeds, horizon, prefix):
    """No test seeds, test outcomes or test score are accepted by this function."""
    unique_seeds(dev_seeds)
    if not 1 <= prefix <= horizon or not candidates:
        raise ValueError("invalid development budget")
    records = []
    for alpha in candidates:
        config = Config(method, alpha)
        runs = [run_lifetime(make_scenario(seed, horizon), config, limit=prefix)
                for seed in dev_seeds]
        records.append({"alpha": alpha, "score": statistics.fmean(map(mean_reward, runs)),
                        "failures": sum(run.status == "failed" for run in runs)})
    # Stable candidate order is the predeclared tie-break; no test-set tie-break.
    chosen = max(range(len(records)), key=lambda i: records[i]["score"])
    return Config(method, candidates[chosen]), records


def experiment(dev_seeds, test_seeds, horizon=900, prefix=240,
               candidates=(0.05, 0.15, 0.4), window=30, reps=2000, raw=False):
    unique_seeds(dev_seeds)
    unique_seeds(test_seeds)
    if set(dev_seeds) & set(test_seeds):
        raise ValueError("development and test seeds must be disjoint")
    if (len(test_seeds) < 2 or horizon % 3 or not 1 <= window <= horizon // 3
            or (horizon // 3) % window):
        raise ValueError("invalid test design")
    selected, development = {}, {}
    for method in ("constant", "decay"):
        selected[method], development[method] = select_on_development(
            method, candidates, dev_seeds, horizon, prefix)
    # Lock both configurations before any test scenario is evaluated.
    test_runs = {method: [] for method in selected}
    for seed in test_seeds:
        scenario = make_scenario(seed, horizon)
        for method, config in selected.items():
            test_runs[method].append(run_lifetime(scenario, config))
    records = {method: [run_metrics(run, window) for run in runs]
               for method, runs in test_runs.items()}
    comparison = paired_bootstrap(
        [mean_reward(run) for run in test_runs["constant"]],
        [mean_reward(run) for run in test_runs["decay"]], reps=reps)
    summary = {method: {
        "mean_lifetime_rate": statistics.fmean(item["lifetime_rate"] for item in rows),
        "mean_phase_rates": [statistics.fmean(item["phase_rates"][p] for item in rows)
                             for p in range(3)],
        "failures": sum(item["status"] == "failed" for item in rows),
        "recovery_counts": [{status: sum(item["recovery"][p]["status"] == status
                                         for item in rows)
                             for status in ("recovered", "right_censored", "failed")}
                            for p in range(2)]} for method, rows in records.items()}
    report = {
        "protocol": {"horizon": horizon, "development_prefix": prefix,
                     "dev_seeds": dev_seeds, "test_seeds": test_seeds,
                     "candidates": list(candidates), "epsilon": 0.1,
                     "decay_scale": 20, "window": window,
                     "pairing": "shared time-indexed potential rewards and action uniforms",
                     "estimand": "selected-configuration lifetime utility; not HPO reliability",
                     "failure_utility": "observed reward followed by zero after stopped service",
                     "development_interactions_per_family": len(candidates) * len(dev_seeds) * prefix,
                     "test_interactions_per_family": len(test_seeds) * horizon},
        "selected": {method: asdict(config) for method, config in selected.items()},
        "development_scores": development, "test_summary": summary,
        "constant_minus_decay": comparison, "per_run": records}
    if raw:
        report["raw_runs"] = {method: [asdict(run) for run in runs]
                              for method, runs in test_runs.items()}
    return report
# END selection


def fault_example():
    run = run_lifetime(make_scenario(700001, 90), Config("constant", 0.15), fault_at=29)
    return {"diagnostic_only_not_in_primary_comparison": True,
            "injected_failure": run.failure, "observed_steps": len(run.observed_rewards),
            "utility_steps": len(run.utility), "unobserved_tail_utility": 0.0,
            "lifetime_rate": mean_reward(run)}


class ExperimentTests(unittest.TestCase):
    def test_constant_hand_update(self):
        learner = Learner(Config("constant", 0.2))
        learner.update(0, 1)
        self.assertAlmostEqual(learner.q[0], 0.6)
        self.assertEqual(learner.q[1], 0.5)

    def test_decay_hand_update(self):
        learner = Learner(Config("decay", 0.2, decay_scale=1))
        learner.update(0, 1)
        learner.update(0, 0)
        self.assertAlmostEqual(learner.q[0], 0.54)

    def test_zero_step(self):
        learner = Learner(Config("constant", 0))
        learner.update(1, 1)
        self.assertEqual(learner.q, [0.5, 0.5])

    def test_invalid_config(self):
        for kwargs in ({"alpha": float("nan")}, {"epsilon": -1}, {"decay_scale": 0}):
            with self.assertRaises(ValueError):
                Config("constant", **({"alpha": 0.1} | kwargs))

    def test_scenario_reproducibility(self):
        self.assertEqual(make_scenario(1, 90), make_scenario(1, 90))
        self.assertNotEqual(make_scenario(1, 90), make_scenario(2, 90))

    def test_phase_boundaries_dont_reset_learner(self):
        run = run_lifetime(make_scenario(1, 90), Config("constant", 0.2))
        self.assertEqual(sum(run.final_counts), 90)
        self.assertEqual(len(run.observed_rewards), 90)

    def test_future_rewards_cannot_change_prefix(self):
        scenario = make_scenario(7, 90)
        changed = Scenario(7, scenario.rewards[:30] + ((1.0, 1.0),) * 60,
                           scenario.action_uniforms)
        config = Config("constant", 0.2)
        self.assertEqual(run_lifetime(scenario, config, limit=30),
                         run_lifetime(changed, config, limit=30))

    def test_shared_tape_identical_method(self):
        scenario = make_scenario(12, 90)
        config = Config("constant", 0.2)
        self.assertEqual(run_lifetime(scenario, config), run_lifetime(scenario, config))

    def test_bootstrap_exact_constant_difference(self):
        result = paired_bootstrap([3, 4, 5], [1, 2, 3], reps=100)
        self.assertEqual(result["ci"], [2, 2])
        self.assertEqual(result["standard_error"], 0)

    def test_bootstrap_identity(self):
        result = paired_bootstrap([0.1, 0.5, 0.9], [0.1, 0.5, 0.9], reps=100)
        self.assertEqual(result["ci"], [0, 0])

    def test_bootstrap_rejects_fake_replication(self):
        with self.assertRaises(ValueError):
            paired_bootstrap([1], [2])
        with self.assertRaises(ValueError):
            paired_bootstrap([1, 2], [1])
        with self.assertRaises(ValueError):
            paired_bootstrap([1, float("nan")], [1, 2])

    def test_quantile_boundaries(self):
        self.assertEqual(quantile([0, 10], 0), 0)
        self.assertEqual(quantile([0, 10], 1), 10)
        self.assertEqual(quantile([0, 10], 0.25), 2.5)
        with self.assertRaises(ValueError):
            quantile([], 0.5)

    def test_split_overlap_and_duplicate(self):
        with self.assertRaises(ValueError):
            experiment([1], [1, 2], horizon=90, prefix=30, window=10)
        with self.assertRaises(ValueError):
            experiment([1, 1], [2, 3], horizon=90, prefix=30, window=10)

    def test_selection_tie_break(self):
        selected, _ = select_on_development("constant", (0, 0.2), [2], 90, 1)
        self.assertEqual(selected.alpha, 0)

    def test_test_set_does_not_change_selection(self):
        args = dict(dev_seeds=[1, 2], horizon=90, prefix=30, window=10, reps=20)
        first = experiment(test_seeds=[20, 21], **args)
        second = experiment(test_seeds=[40, 41], **args)
        self.assertEqual(first["selected"], second["selected"])

    def test_failure_retains_reward_and_zero_utility_tail(self):
        run = run_lifetime(make_scenario(5, 90), Config("constant", 0.2), fault_at=9)
        self.assertEqual(run.status, "failed")
        self.assertEqual(len(run.observed_rewards), 10)
        self.assertEqual(run.utility[:10], run.observed_rewards)
        self.assertEqual(run.utility[10:], (0.0,) * 80)
        self.assertEqual(recovery(run, 30, 60, 10)["status"], "failed")

    def test_recovery_confirmation_and_censoring(self):
        base = run_lifetime(make_scenario(5, 90), Config("constant", 0.2))
        success = Run(**(base.__dict__ | {"observed_rewards": (1.0,) * 90}))
        failure = Run(**(base.__dict__ | {"observed_rewards": (0.0,) * 90}))
        self.assertEqual(recovery(success, 30, 60, 10)["confirmation_delay"], 20)
        self.assertEqual(recovery(failure, 30, 60, 10)["status"], "right_censored")

    def test_failure_at_interval_endpoint_is_not_censoring(self):
        base = make_scenario(5, 90)
        scenario = Scenario(base.seed, ((0.0, 0.0),) * 90, base.action_uniforms)
        run = run_lifetime(scenario, Config("constant", 0.2), fault_at=59)
        self.assertEqual(len(run.observed_rewards), 60)
        self.assertEqual(recovery(run, 30, 60, 10)["status"], "failed")

    def test_later_failure_does_not_relabel_earlier_censoring(self):
        base = make_scenario(5, 90)
        scenario = Scenario(base.seed, ((0.0, 0.0),) * 90, base.action_uniforms)
        run = run_lifetime(scenario, Config("constant", 0.2), fault_at=60)
        self.assertEqual(recovery(run, 30, 60, 10)["status"], "right_censored")

    def test_confirmation_precedes_later_update_failure(self):
        base = make_scenario(5, 90)
        scenario = Scenario(base.seed, ((1.0, 1.0),) * 90, base.action_uniforms)
        run = run_lifetime(scenario, Config("constant", 0.2), fault_at=49)
        self.assertEqual(recovery(run, 30, 60, 10),
                         {"status": "recovered", "confirmation_delay": 20})
        self.assertEqual(run.status, "failed")

    def test_window_lengths_must_match(self):
        run = run_lifetime(make_scenario(5, 90), Config("constant", 0.2))
        with self.assertRaises(ValueError):
            run_metrics(run, 25)
        with self.assertRaises(ValueError):
            experiment([1], [2, 3], horizon=90, prefix=30, window=25)
        self.assertEqual(len(run_metrics(run, 10)["window_rates"]), 9)

    def test_numerical_failure_remains_json_serializable(self):
        def broken_update(learner, action, reward):
            learner.q[action] = float("nan")
            raise FloatingPointError("non-finite value estimate")
        with patch.object(Learner, "update", broken_update):
            run = run_lifetime(make_scenario(5, 90), Config("constant", 0.2))
        self.assertEqual(run.status, "failed")
        self.assertFalse(run.failure["injected"])
        self.assertIn(None, run.final_q)
        self.assertEqual(len(run.observed_rewards), 1)
        json.dumps(asdict(run), allow_nan=False)

    def test_freeze_isolated_and_prefix_preserved(self):
        scenario, config = make_scenario(8, 90), Config("constant", 0.2)
        before = run_lifetime(scenario, config, limit=30)
        frozen = run_lifetime(scenario, config, freeze_at=30)
        self.assertEqual(frozen.observed_rewards[:30], before.observed_rewards)
        self.assertEqual(frozen.final_q, before.final_q)
        self.assertEqual(sum(frozen.final_counts), 30)
        self.assertEqual(sum(run_lifetime(scenario, config).final_counts), 90)

    def test_finite_difference_update_gradient(self):
        # Frozen reward target: d[0.5(q-r)^2]/dq = q-r.
        q, reward, h = 0.37, 1.0, 1e-5
        loss = lambda value: 0.5 * (value - reward) ** 2
        numeric = (loss(q + h) - loss(q - h)) / (2 * h)
        self.assertAlmostEqual(numeric, q - reward, places=9)

    def test_budget_accounting_and_finite_report(self):
        report = experiment([1, 2], [10, 11], 90, 30, window=10, reps=20)
        self.assertEqual(report["protocol"]["development_interactions_per_family"], 180)
        self.assertEqual(report["protocol"]["test_interactions_per_family"], 180)
        json.dumps(report, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("demo", "test"))
    parser.add_argument("--tiny", action="store_true", help="90 steps, 4 dev and 8 test runs")
    parser.add_argument("--raw", action="store_true", help="include all observed rewards and actions")
    args = parser.parse_args()
    if args.command == "test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ExperimentTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.tiny:
        report = experiment(list(range(100, 104)), list(range(1000, 1008)),
                            horizon=90, prefix=30, window=10, reps=500, raw=args.raw)
    else:
        report = experiment(list(range(100, 108)), list(range(1000, 1024)), raw=args.raw)
    report["fault_example"] = fault_example()
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
