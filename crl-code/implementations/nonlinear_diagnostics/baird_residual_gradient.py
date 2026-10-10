"""Exact-model residual gradient on the same fixed linear Baird problem.

J(w)=1/(2*7) sum_s [gamma*x_lower.w-x_s.w]^2.
Its NEGATIVE gradient is mean_s delta_s*(x_s-gamma*x_lower).
This differs from TD's mean_s delta_s*x_s. The objective and update changed.
"""
from pathlib import Path
try:
    from ._common import BAIRD_X, BAIRD_GAMMA, baird_init, baird_error, dot, log, validate
except ImportError:
    from _common import BAIRD_X, BAIRD_GAMMA, baird_init, baird_error, dot, log, validate

META = {
    "id": "baird_residual_gradient",
    "name": "Baird：精确模型残差梯度",
    "family": "nonlinear_diagnostics",
    "coverage_refs": [],
    "implementation_kind": "component_experiment",
    "task": "baird_expected_diagnostic",
    "baseline": "baird_expected_td",
    "metric": "log10(1 + 七状态价值 RMSE)",
    "unit": "log_value",
    "higher_better": False,
    "budget": "expected_sweeps",
    "scope": "analytic-diagnostic",
    "budget_note": "每步枚举七个状态的期望更新；不是环境步。对照同样七项。",
    "chapter_paths": [
        "foundations/learning-dynamics/",
        "foundations/approximation/prediction/",
        "foundations/approximation/off-policy/"
    ],
    "sources": [
        "https://leemon.com/papers/1995b.pdf"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "同一 Baird 星形问题，对精确 Bellman 残差平方求梯度；每步枚举七状态。",
    "question": "把半梯度改为另一目标的真梯度，能说明什么、不能说明什么？",
    "limitations": "确定性目标转移使此例无需 double sampling。下降 Bellman 残差不等于每步下降价值 RMSE，也不意味着对一般随机模型可直接使用单后继残差梯度。"
}

def objective(w):
    return sum((BAIRD_GAMMA * dot(BAIRD_X[6], w) - dot(x, w))**2 for x in BAIRD_X) / 14

def direction(w):
    residual = [BAIRD_GAMMA * dot(BAIRD_X[6], w) - dot(x, w) for x in BAIRD_X]
    return [sum(residual[s] * (BAIRD_X[s][j]-BAIRD_GAMMA*BAIRD_X[6][j]) for s in range(7)) / 7 for j in range(8)]

def update(w, alpha=0.05):
    g = direction(w)
    return [p + alpha * q for p, q in zip(w, g)]

def run(seed=0, steps=1200, emit=None):
    validate(steps)
    w, rows = baird_init(seed), []
    y, error = baird_error(w)
    log(rows, 0, y, steps, emit, rmse=error, msbe=objective(w))
    for t in range(1, steps + 1):
        w = update(w)
        y, error = baird_error(w)
        log(rows, t, y, steps, emit, rmse=error, msbe=objective(w))
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
