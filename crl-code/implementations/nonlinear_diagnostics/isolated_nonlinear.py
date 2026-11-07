"""Parameter-isolation control. It costs twice as many parameters.

The same observations and labels are used as shared_nonlinear.py. Each context
owns a separate smooth network. Routing is known, not discovered. This baseline
tests absence of cross-updates; it cannot measure benefits of representation
sharing, and its extra capacity must not be omitted from the comparison.
"""
from pathlib import Path
try:
    from ._common import value, gradient, nonlinear_init, regression_context, regression_metrics, log, validate
except ImportError:
    from _common import value, gradient, nonlinear_init, regression_context, regression_metrics, log, validate

META = {
    "id": "isolated_nonlinear",
    "name": "隔离非线性表示：零交叉更新对照",
    "family": "nonlinear_diagnostics",
    "coverage_refs": [],
    "implementation_kind": "component_experiment",
    "task": "nonlinear_interference_diagnostic",
    "baseline": "shared_nonlinear",
    "metric": "两个固定条件目标的均匀 RMSE",
    "unit": "prediction",
    "higher_better": False,
    "budget": "stream_observations",
    "scope": "analytic-diagnostic",
    "budget_note": "每步一个观测，无回放；共享4参数，隔离8参数并预知上下文。",
    "chapter_paths": [
        "foundations/learning-dynamics/",
        "foundations/approximation/prediction/",
        "algorithms/plasticity/"
    ],
    "sources": [
        "https://arxiv.org/abs/2001.06782"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "相同 A/B/A 观测序列，每个上下文有独立四参数 tanh 网络。只更新当前上下文，另一个保持不动。",
    "question": "怎样构造可检验的零干扰参照，并诚实计入额外容量？",
    "limitations": "需要预先知道上下文路由，参数量八而非四。它不展示知识迁移，不是共享模型普遍更差的证据；也不是持续智能体的完整方案。"
}

def update(w, x, target, alpha=0.03):
    residual, g = target-value(w, x), gradient(w, x)
    return [p + alpha*residual*q for p, q in zip(w, g)]

def run(seed=0, steps=1200, emit=None):
    validate(steps)
    initial = nonlinear_init(seed)
    parameters, rows = [initial[:], initial[:]], []
    error, e = regression_metrics(parameters, separate=True)
    log(rows, 0, error, steps, emit, error_A=abs(e[0]), error_B=abs(e[1]), parameter_count=8)
    for t in range(1, steps+1):
        context, phase = regression_context(t, steps)
        other_before = value(parameters[1-context], float(2-context))
        parameters[context] = update(parameters[context], float(context+1), 0.5 if context == 0 else -0.5)
        drift = abs(value(parameters[1-context], float(2-context))-other_before)
        error, e = regression_metrics(parameters, separate=True)
        log(rows, t, error, steps, emit, phase, error_A=abs(e[0]), error_B=abs(e[1]),
            gradient_dot=0.0, unobserved_prediction_drift=drift, parameter_count=8)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
