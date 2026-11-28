"""Single-successor residual gradients minimize the WRONG desired objective.

At queried state A, v(A)=w; reward=1, gamma=.5; successor feature X is
0 or 2 with equal probability. J_MSBE=.5*(1-.5*w)^2 has optimum w=2.
E[.5*delta(X)^2]=.25*((1-w)^2+1) instead has optimum w=1.
Both experiments pay for TWO successor queries per update. Here they are
used as two single-sample gradients, then averaged, not discarded.
"""
import random
from pathlib import Path
try:
    from ._common import log, validate
except ImportError:
    from _common import log, validate

META = {
    "id": "single_sample_residual",
    "name": "单后继残差梯度：有偏目标诊断",
    "family": "nonlinear_diagnostics",
    "coverage_refs": [],
    "implementation_kind": "component_experiment",
    "task": "double_sampling_diagnostic",
    "baseline": "double_sample_residual",
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
        "https://leemon.com/papers/1995b.pdf",
        "http://incompleteideas.net/book/the-book-2nd.html"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "生成模型返回两个条件独立后继，特征各为 0 或 2，概率各 1/2。单样本法平均两个各自的残差梯度。",
    "question": "样本 TD 平方误差下降，为什么不代表 Bellman 期望残差下降到零？",
    "limitations": "限定评价起点 A 的采样目标；后继价值使用同一参数特征，并未声称是完整 MDP 的所有状态价值回归。两个方法每步同为两次模型查询。"
}

def objective(w):
    return 0.5 * (1.0 - 0.5 * w)**2

def direction(w, x):
    residual = 1.0 + 0.5 * x * w - w
    return residual * (1.0 - 0.5 * x)

def run(seed=0, steps=1200, emit=None):
    validate(steps)
    rng, w, rows = random.Random(seed), 0.0, []
    log(rows, 0, objective(w), steps, emit, parameter=w, successor_queries=0)
    for t in range(1, steps + 1):
        x1, x2 = 2.0*rng.randrange(2), 2.0*rng.randrange(2)
        # Both residuals use OLD w. Averaging two biased gradients does not remove bias.
        d = 0.5*(direction(w, x1)+direction(w, x2))
        w += (0.4 / (1.0 + 0.03*t)) * d
        log(rows, t, objective(w), steps, emit, parameter=w, successor_queries=2*t)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
