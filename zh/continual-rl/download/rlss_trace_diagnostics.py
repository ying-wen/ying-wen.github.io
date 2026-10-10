#!/usr/bin/env python3
"""TD(lambda): distinguish credit, projected fixed points, and update scale.

Copyright (c) 2026 Ying Wen. SPDX-License-Identifier: MIT
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

Only Python >=3.10 standard library is required. Run:
  python3 rlss_trace_diagnostics.py test
  python3 rlss_trace_diagnostics.py run --output results.json

This is a deliberately small, stationary, on-policy prediction diagnostic.
It is NOT a reproduction of a Sutton lecture's empirical results. The model
and hidden states belong to the evaluator, not to the online TD learner.
The deterministic mean-field iteration is NOT E[w_t] for adaptive TD driven
by correlated Markov observations. Constant-step-size finite runs do NOT
prove convergence, nor does this example establish an optimal lambda.
"""

from __future__ import annotations

import argparse
import cmath
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


# Environment states 0 and 1 have identical learner features. The evaluator
# retains the underlying Markov state to generate transitions and score value.
P = [[0.8, 0.2, 0.0], [0.0, 0.5, 0.5], [0.3, 0.0, 0.7]]
R = [1.0, 0.0, -1.0]  # Reward R_(t+1) = R[S_t], BEFORE the sampled transition.
PHI = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
GAMMA = 0.9
LAMBDAS = [0.0, 0.5, 0.9, 1.0]
ALPHA = 0.01
STEPS = 20000
SEEDS = list(range(16))
CHECKPOINTS = [0, 10, 30, 100, 300, 1000, 3000, 10000, STEPS]


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


def mv(a, x):
    return [dot(row, x) for row in a]


def transpose(a):
    return [list(column) for column in zip(*a)]


def mm(a, b):
    return [[dot(row, column) for column in transpose(b)] for row in a]


def eye(n):
    return [[float(i == j) for j in range(n)] for i in range(n)]


def solve(a, b):
    """Small dense solve with partial pivoting; no numerical package required."""
    rows = [list(row) + [value] for row, value in zip(a, b)]
    n = len(rows)
    for k in range(n):
        pivot = max(range(k, n), key=lambda i: abs(rows[i][k]))
        rows[k], rows[pivot] = rows[pivot], rows[k]
        if abs(rows[k][k]) < 1e-14:
            raise ValueError("Singular diagnostic system")
        scale = rows[k][k]
        rows[k] = [x / scale for x in rows[k]]
        for i in range(n):
            if i != k:
                scale = rows[i][k]
                rows[i] = [x - scale * y for x, y in zip(rows[i], rows[k])]
    return [row[-1] for row in rows]


def solve_columns(a, b):
    return transpose([solve(a, column) for column in transpose(b)])


def i_minus(c):
    return [[float(i == j) - c * P[i][j] for j in range(3)] for i in range(3)]


def stationary():
    # Two independent stationarity constraints and sum(d)=1.
    a = [[P[j][i] - float(i == j) for j in range(3)] for i in range(2)]
    return solve(a + [[1.0, 1.0, 1.0]], [0.0, 0.0, 1.0])


D = stationary()
VALUE = solve(i_minus(GAMMA), R)
PHI_T_D = [[PHI[s][j] * D[s] for s in range(3)] for j in range(2)]
PROJECTED_W = solve(mm(PHI_T_D, PHI), mv(PHI_T_D, VALUE))
PROJECTED_VALUE = mv(PHI, PROJECTED_W)


def value_distance(x, y):
    return math.sqrt(sum(d * (a - b) ** 2 for d, a, b in zip(D, x, y)))


# BEGIN fixed-point
def projected_system(lam):
    """Stationary, frozen-parameter mean TD drift = b_lambda - A_lambda w.

    D weights hidden states by their stationary probabilities. The inverse is
    a geometric sum of transition powers. No trajectory is sampled here.
    """
    resolvent = i_minus(GAMMA * lam)
    a = mm(PHI_T_D, solve_columns(resolvent, mm(i_minus(GAMMA), PHI)))
    b = mv(PHI_T_D, solve(resolvent, R))
    return a, b, solve(a, b)


def expected_iteration(a, b, alpha, steps):
    """Model-based deterministic drift iteration, NOT a sampled TD run."""
    w = [0.0, 0.0]
    for _ in range(steps):
        aw = mv(a, w)
        w = [x + alpha * (target - current) for x, target, current in zip(w, b, aw)]
    return w


def spectral_radius(a, alpha):
    """rho(I-alpha*A), including complex eigenvalues of a real 2x2 matrix."""
    trace = a[0][0] + a[1][1]
    determinant = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    root = cmath.sqrt(trace * trace - 4 * determinant)
    eigenvalues = [(trace + root) / 2, (trace - root) / 2]
    return max(abs(1 - alpha * eig) for eig in eigenvalues)
# END fixed-point


def sample_index(probabilities, rng):
    draw = rng.random()
    cumulative = 0.0
    for i, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return i
    return len(probabilities) - 1


def transitions(seed, steps):
    """Evaluator samples one continuing lifetime. No episode resets occur."""
    rng = random.Random(seed)
    state = sample_index(D, rng)
    for _ in range(steps):
        next_state = sample_index(P[state], rng)
        # The learner receives ONLY two features, reward and next features.
        yield PHI[state], R[state], PHI[next_state]
        state = next_state


# BEGIN sampled-td
def td_step(w, e, features, reward, next_features, lam, alpha):
    """Accumulating on-policy semi-gradient TD(lambda), fixed linear features.

    Compute both predictions before changing any weight. There is no replay,
    model, hidden state, gradient through the target, or trace reset.
    """
    delta = reward + GAMMA * dot(w, next_features) - dot(w, features)
    trace = [GAMMA * lam * old + feature for old, feature in zip(e, features)]
    next_w = [weight + alpha * delta * z for weight, z in zip(w, trace)]
    return next_w, trace


def run_stream(seed, lam, alpha, steps=STEPS, checkpoints=CHECKPOINTS):
    w, e = [0.0, 0.0], [0.0, 0.0]
    _, _, target = projected_system(lam)
    target_values = mv(PHI, target)
    records = []

    def record(step):
        fitted = mv(PHI, w)
        records.append({"step": step, "weights": list(w),
                        "value_rmse": value_distance(fitted, VALUE),
                        "fixed_point_distance": value_distance(fitted, target_values),
                        "trace_l1": sum(abs(x) for x in e)})

    record(0)
    wanted = set(checkpoints)
    for t, (features, reward, next_features) in enumerate(transitions(seed, steps), 1):
        w, e = td_step(w, e, features, reward, next_features, lam, alpha)
        if not all(math.isfinite(x) for x in w):
            # Never omit a failed run or serialize Infinity as a curve point.
            raise ArithmeticError(f"Nonfinite weights: seed={seed}, lambda={lam}, alpha={alpha}, step={t}")
        if t in wanted:
            record(t)
    return {"seed": seed, "lambda": lam, "alpha": alpha, "checkpoints": records}
# END sampled-td


def check():
    checks = {}

    def close(name, error, tolerance=1e-10):
        assert error < tolerance, (name, error, tolerance)
        checks[name] = {"passed": True, "absolute_error": error, "tolerance": tolerance}

    close("stationarity", max(abs(x - y) for x, y in zip(mv(transpose(P), D), D)))
    close("bellman_truth", max(abs(x - y) for x, y in zip(mv(i_minus(GAMMA), VALUE), R)))
    _, _, w1 = projected_system(1.0)
    close("lambda_one_is_projection", max(abs(x - y) for x, y in zip(w1, PROJECTED_W)))

    # Independent Neumann-series expansion avoids the inverse used above.
    # E[phi_(t-k) (phi_t-gamma*phi_(t+1))^T] = Phi^T D P^k (I-gamma P) Phi.
    for lam in LAMBDAS:
        a, b, w = projected_system(lam)
        power = eye(3)
        sum_matrix = [[0.0] * 3 for _ in range(3)]
        q = GAMMA * lam
        for k in range(400):
            for i in range(3):
                for j in range(3):
                    sum_matrix[i][j] += q ** k * power[i][j]
            power = mm(power, P)
        expanded_a = mm(mm(PHI_T_D, sum_matrix), mm(i_minus(GAMMA), PHI))
        expanded_b = mv(mm(PHI_T_D, sum_matrix), R)
        close(f"neumann_A_{lam}", max(abs(a[i][j] - expanded_a[i][j]) for i in range(2) for j in range(2)))
        close(f"neumann_b_{lam}", max(abs(x - y) for x, y in zip(b, expanded_b)))
        close(f"fixed_point_residual_{lam}", max(abs(x - y) for x, y in zip(mv(a, w), b)))
        fitted = mv(PHI, w)
        floor = value_distance(PROJECTED_VALUE, VALUE)
        gap = value_distance(fitted, PROJECTED_VALUE)
        close(f"orthogonal_error_decomposition_{lam}", abs(value_distance(fitted, VALUE) ** 2 - floor ** 2 - gap ** 2))
        close(f"expected_iteration_converges_{lam}", value_distance(mv(PHI, expected_iteration(a, b, 0.2, 5000)), fitted))

        # On nonnegative one-hot features, trace L1 depends only on q and time.
        e = [0.0, 0.0]
        w = [0.2, -0.4]
        for t, (features, reward, next_features) in enumerate(transitions(2, 100), 1):
            next_w, e = td_step(w, e, features, reward, next_features, lam, 0.01)
            close(f"trace_mass_{lam}_{t}", abs(sum(e) - (1 - q ** t) / (1 - q)))
            delta = reward + GAMMA * dot(w, next_features) - dot(w, features)
            close(f"update_scale_{lam}_{t}", abs(sum(abs(a - b) for a, b in zip(next_w, w)) - 0.01 * abs(delta) * sum(e)))
            w = next_w

    # Stationary mean-field stability cannot establish stability of each sample.
    a0, _, _ = projected_system(0.0)
    a1, _, _ = projected_system(1.0)
    assert spectral_radius(a0, 3.0) < 1 < spectral_radius(a1, 3.0)
    close("eigenvalue_identity", abs(spectral_radius([[1, 0], [0, 2]], 0.2) - 0.8))
    close("complex_eigenvalues", abs(spectral_radius([[1, -1], [1, 1]], 0.2) - math.sqrt(0.68)))
    assert run_stream(7, 0.9, 0.01, 30, [0, 10, 30]) == run_stream(7, 0.9, 0.01, 30, [0, 10, 30])
    # Compact public check summary; all 800 trace identities are still executed.
    identities = [item for name, item in checks.items() if name.startswith(("trace_mass_", "update_scale_"))]
    summary = {name: value for name, value in checks.items() if not name.startswith(("trace_mass_", "update_scale_"))}
    summary["trace_and_update_identities"] = {"passed": True, "cases": len(identities), "max_absolute_error": max(x["absolute_error"] for x in identities)}
    summary["paired_stream_reproducibility"] = {"passed": True, "seed": 7, "steps": 30}
    summary["same_alpha_different_mean_stability"] = {"passed": True, "alpha": 3.0, "rho_lambda_0": spectral_radius(a0, 3.0), "rho_lambda_1": spectral_radius(a1, 3.0)}
    return summary


def series(name, points, role="estimate"):
    return {"name": name, "role": role, "points": [{"x": x, "y": y} for x, y in points]}


def chart(identifier, title, x_label, y_label, lines):
    return {"id": identifier, "title": title, "xLabel": x_label, "yLabel": y_label,
            "xScale": "linear", "yScale": "linear", "series": lines}


def run():
    checks = check()
    floor = value_distance(PROJECTED_VALUE, VALUE)
    analytic = []
    for i in range(101):
        lam = i / 100
        a, b, w = projected_system(lam)
        analytic.append({"lambda": lam, "A": a, "b": b, "weights": w,
                         "value_rmse": value_distance(mv(PHI, w), VALUE),
                         "projection_gap": value_distance(mv(PHI, w), PROJECTED_VALUE)})

    expected = []
    sampled = []
    aggregates = []
    for lam in LAMBDAS:
        a, b, target = projected_system(lam)
        for mode in ["raw", "mass-scaled"]:
            # A diagnostic rescaling, not a claim of optimal or equal-noise alpha.
            alpha = ALPHA if mode == "raw" else ALPHA * (1 - GAMMA * lam)
            expected_w = expected_iteration(a, b, alpha, STEPS)
            expected.append({"lambda": lam, "mode": mode, "alpha": alpha, "iterations": STEPS,
                             "weights": expected_w, "value_rmse": value_distance(mv(PHI, expected_w), VALUE),
                             "fixed_point_distance": value_distance(mv(PHI, expected_w), mv(PHI, target))})
            runs = [dict(run_stream(seed, lam, alpha), mode=mode) for seed in SEEDS]
            sampled.extend(runs)
            values = [row["checkpoints"][-1]["value_rmse"] for row in runs]
            distances = [row["checkpoints"][-1]["fixed_point_distance"] for row in runs]
            aggregates.append({"lambda": lam, "mode": mode, "alpha": alpha, "runs": len(runs),
                               "final_rmse_mean": statistics.mean(values), "final_rmse_sd": statistics.stdev(values),
                               "final_rmse_min": min(values), "final_rmse_max": max(values),
                               "fixed_point_distance_mean": statistics.mean(distances)})

    alpha_grid = [i / 10 for i in range(101)]
    stability = [{"lambda": lam, "alpha": alpha, "spectral_radius": spectral_radius(projected_system(lam)[0], alpha)}
                 for lam in LAMBDAS for alpha in alpha_grid]

    def sampled_series(mode, name):
        return {"name": name, "role": "estimate", "points": [
            {"x": x["lambda"], "y": x["final_rmse_mean"],
             "low": x["final_rmse_min"], "high": x["final_rmse_max"]}
            for x in aggregates if x["mode"] == mode]}

    charts = [
        chart("fixed-point-gap", "固定点离最佳表示有多远", "λ", "稳态加权 RMS 距离", [
            series("TD固定点到最佳投影", [(x["lambda"], x["projection_gap"]) for x in analytic]),
            series("最佳投影本身", [(0, 0), (1, 0)], "reference")]),
        chart("trace-mass", "反复激活使累计迹增大", "已观测转移数", "资格迹 L1 范数", [
            series(f"λ={lam:g}", [(t, (1 - (GAMMA * lam) ** t) / (1 - GAMMA * lam)) for t in range(81)]) for lam in LAMBDAS]),
        chart("expected-stability", "相同步长未必有相同稳定性", "α（完整扫描 0 至 10）", "ρ(I−αAλ)", [
            series(f"λ={lam:g}", [(x["alpha"], x["spectral_radius"]) for x in stability if x["lambda"] == lam]) for lam in LAMBDAS] + [
            series("稳定性边界 ρ=1", [(0, 1), (10, 1)], "reference")]),
        chart("sampled-error", "有限轨迹不等于解析固定点", "λ", "第20000步的价值 RMSE", [
            sampled_series("raw", "固定 α=0.01"),
            sampled_series("mass-scaled", "α=0.01(1−γλ)"),
            series("解析 TD 固定点", [(x["lambda"], x["value_rmse"]) for x in analytic], "reference"),
            series("表示误差下限", [(0, floor), (1, floor)], "reference")]),
    ]
    charts[-1]["intervalLabel"] = "范围带：16 次运行的最小值至最大值，不是置信区间。"
    return {
        "id": "rlss-traces", "title": "λ：信用传播、固定点与更新尺度",
        "scope": "固定平稳 MRP、在策略、线性不充分特征的教学诊断；不复现原课件实验，不证明深度或非平稳 TD 稳定性。",
        "protocol": {
            "environment": {"states": 3, "P": P, "reward_on_departure": R, "gamma": GAMMA,
                            "features": PHI, "initial_state_distribution": D, "terminal_states": [],
                            "resets": "none within each independent seed"},
            "learner_access": "Only phi_t, R_(t+1), phi_(t+1). No model, hidden state, future transitions or replay.",
            "evaluator_access": "P, R, true hidden states and stationary d; used only for exact references and evaluation.",
            "initial_weights": [0, 0], "initial_trace": [0, 0], "lambda_grid": LAMBDAS,
            "sampled_alpha_rules": {"raw": "0.01", "mass-scaled": "0.01*(1-gamma*lambda)"},
            "seeds": SEEDS, "steps_per_run": STEPS, "checkpoints": CHECKPOINTS,
            "sampled_runs": len(sampled), "total_sampled_updates": len(sampled) * STEPS,
            "randomization": "Python random.Random(seed), initial state and each transition sampled in the same order for every condition; paired streams.",
            "reporting": "Every seed and both alpha rules retained; final endpoint mean, SD, min/max, no best-seed or best-alpha selection; SD is not a confidence interval.",
            "expected_iteration": "w <- w+alpha*(b-Aw), with stationary frozen-parameter drift. One iteration is a dense 2x2 model calculation, NOT one environment interaction and NOT E[w_t].",
            "stability_alpha_grid": alpha_grid,
            "stability_scope": "rho(I-alpha*A)<1 characterizes deterministic linear mean-field convergence for every initial weight. It does not characterize sample-path stability or finite-constant-alpha TD bias.",
            "source_file": "rlss_trace_diagnostics.py", "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "references": ["https://web.mit.edu/jnt/www/Papers/J063-97-bvr-td.pdf", "http://incompleteideas.net/book/the-book-2nd.html"],
        },
        "truth": {"stationary": D, "value": VALUE, "projected_weights": PROJECTED_W,
                  "projected_value": PROJECTED_VALUE, "representation_rmse_floor": floor},
        "checks": checks, "charts": charts, "analytic_fixed_points": analytic,
        "deterministic_expected_iterations": expected, "stability_scan": stability,
        "sampled_summary": aggregates, "sampled_runs": sampled,
        "limitations": [
            "A stationary three-state counterexample is a mechanism diagnostic, not a benchmark or an alpha/lambda recommendation.",
            "Reward observations could support a richer history-based state; this experiment deliberately fixes memoryless aliased features.",
            "All stochastic runs use constant alpha and zero initial traces. Finite Markov trajectories need not equal the frozen-parameter stationary expectation.",
            "Scaling alpha by 1-gamma*lambda only normalizes the limiting trace L1 mass for these nonnegative one-hot features. It equalizes neither noise, convergence rate nor policy quality.",
            "Endpoint bands show the observed min/max across 16 runs, not confidence intervals; standard deviations and every checkpoint are retained in this JSON.",
            "No off-policy correction, policy improvement, learned representation, changing environment or nonlinear approximator is tested.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("test")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "test":
        checks = check()
        print(json.dumps({"passed": True, "checks": checks}, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        result = run()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "sampled_runs": len(result["sampled_runs"]),
                          "sampled_updates": result["protocol"]["total_sampled_updates"],
                          "representation_rmse_floor": result["truth"]["representation_rmse_floor"],
                          "summary": result["sampled_summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
