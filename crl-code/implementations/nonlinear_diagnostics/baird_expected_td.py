"""Baird star: the expected semi-gradient TD update can increase value error.

The exact target-policy transition is always to state 6. The behavior stationary
distribution is uniform on 7 states. E_b[rho * delta * x] therefore equals the
enumeration below. This isolates update dynamics: no noisy samples, target
network, optimizer, replay, or clipping. The feature system is LINEAR.
"""
from pathlib import Path
try:
    from ._common import BAIRD_X, BAIRD_GAMMA, baird_init, baird_error, dot, log, validate
except ImportError:
    from _common import BAIRD_X, BAIRD_GAMMA, baird_init, baird_error, dot, log, validate

META = {
    "id": "baird_expected_td",
    "name": "Baird：期望半梯度 TD",
    "family": "nonlinear_diagnostics",
    "coverage_refs": [],
    "implementation_kind": "component_experiment",
    "task": "baird_expected_diagnostic",
    "baseline": "baird_residual_gradient",
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
        "https://leemon.com/papers/1995b.pdf",
        "http://incompleteideas.net/book/the-book-2nd.html"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "七状态八维固定线性特征；行为状态均匀、目标策略总到下状态。每步枚举一次全部状态的期望更新。",
    "question": "没有随机噪声和神经网络时，离策略自举为什么仍可发散？",
    "limitations": "已知模型的期望更新，不是仅一次真实交互的流式算法。特征冗余，参数不唯一。横轴一次 sweep 包含七个状态项；误差未裁剪。"
}

def direction(w):
    next_value = dot(BAIRD_X[6], w)
    delta = [BAIRD_GAMMA * next_value - dot(x, w) for x in BAIRD_X]
    # All seven state terms use the SAME old weights. This is a simultaneous sweep.
    return [sum(delta[s] * BAIRD_X[s][j] for s in range(7)) / 7 for j in range(8)]

def update(w, alpha=0.05):
    g = direction(w)
    return [p + alpha * q for p, q in zip(w, g)]

def run(seed=0, steps=1200, emit=None):
    validate(steps)
    w, rows = baird_init(seed), []
    y, error = baird_error(w)
    log(rows, 0, y, steps, emit, rmse=error)
    for t in range(1, steps + 1):
        w = update(w)
        y, error = baird_error(w)
        log(rows, t, y, steps, emit, rmse=error, parameter_norm=dot(w, w)**0.5)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
