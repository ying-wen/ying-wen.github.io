"""Unbiased MSBE descent by independent successors, with symmetrization.

Use delta(X1)*(1-.5*X2) and delta(X2)*(1-.5*X1).
Independence CONDITIONAL ON THE SAME CURRENT STATE is essential.
The symmetrized mean reuses exactly two model queries, matching the comparator.
"""
import random
from pathlib import Path
try:
    from ._common import log, validate
except ImportError:
    from _common import log, validate

META = {
    "id": "double_sample_residual",
    "name": "独立双后继残差梯度",
    "family": "nonlinear_diagnostics",
    "coverage_refs": [],
    "implementation_kind": "component_experiment",
    "task": "double_sampling_diagnostic",
    "baseline": "single_sample_residual",
    "metric": "真实 MSBE",
    "unit": "squared_value",
    "higher_better": False,
    "budget": "paired_successor_queries",
    "scope": "analytic-diagnostic",
    "budget_note": "每步两次条件独立后继模型查询；两方法都使用全部样本。",
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
    "description": "与单后继法使用相同两个生成模型查询；交叉相乘残差和独立后继梯度，得到 MSBE 无偏梯度估计。",
    "question": "期望的乘积应怎样采样？单条不可回退轨迹缺少什么信息？",
    "limitations": "每步需要同一起点的条件独立后继。严格单轨迹通常不能直接满足；这不是可无条件移植到流式 RL 的方法。有限时间曲线仍含采样方差。"
}

def objective(w):
    return 0.5 * (1.0 - 0.5 * w)**2

def direction(w, x1, x2):
    d1, d2 = 1.0 + (0.5*x1-1.0)*w, 1.0 + (0.5*x2-1.0)*w
    return 0.5*(d1*(1.0-0.5*x2)+d2*(1.0-0.5*x1))

def run(seed=0, steps=1200, emit=None):
    validate(steps)
    rng, w, rows = random.Random(seed), 0.0, []
    log(rows, 0, objective(w), steps, emit, parameter=w, successor_queries=0)
    for t in range(1, steps + 1):
        x1, x2 = 2.0*rng.randrange(2), 2.0*rng.randrange(2)
        w += (0.4 / (1.0 + 0.03*t)) * direction(w, x1, x2)
        log(rows, t, objective(w), steps, emit, parameter=w, successor_queries=2*t)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
