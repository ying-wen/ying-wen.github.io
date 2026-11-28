"""Shared nonlinear regression isolates interference WITHOUT RL bootstrapping.

f(x)=a*tanh(b*x+c)+d, x_A=1, y_A=.5, x_B=2, y_B=-.5.
A/B/A observation blocks share one four-parameter network. Both targets remain
fixed: only visitation changes. No previous sample is retained for training.
The uniform evaluation knows both targets, but never supplies them to updates.
"""
from pathlib import Path
try:
    from ._common import value, gradient, nonlinear_init, regression_context, regression_metrics, dot, log, validate
except ImportError:
    from _common import value, gradient, nonlinear_init, regression_context, regression_metrics, dot, log, validate

META = {
    "id": "shared_nonlinear",
    "name": "共享非线性表示：干扰诊断",
    "family": "nonlinear_diagnostics",
    "coverage_refs": [],
    "implementation_kind": "component_experiment",
    "task": "nonlinear_interference_diagnostic",
    "baseline": "isolated_nonlinear",
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
        "https://arxiv.org/abs/2001.06782",
        "https://www.nature.com/articles/s41586-024-07711-7"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "两个固定回归目标 +0.5/-0.5，观测依次只来自 A、B、A。一个四参数 tanh 网络共享表示，逐观测 SGD，无回放。",
    "question": "没有自举和策略学习，更新当前样本为何也会破坏另一个样本的预测？",
    "limitations": "构造性监督回归诊断，不是 RL benchmark，也不是长期可塑性丧失的证据。隔离对照用八参数和已知上下文路由，不是同参数量优劣比较。"
}

def update(w, x, target, alpha=0.03):
    # True SGD for one FIXED supervised label, not a TD semi-gradient.
    residual, g = target-value(w, x), gradient(w, x)
    return [p + alpha*residual*q for p, q in zip(w, g)]

def loss_gradient(w, context):
    x, target = float(context+1), 0.5 if context == 0 else -0.5
    return [(value(w, x)-target)*g for g in gradient(w, x)]

def run(seed=0, steps=1200, emit=None):
    validate(steps)
    w, rows = nonlinear_init(seed), []
    error, e = regression_metrics(w)
    log(rows, 0, error, steps, emit, error_A=abs(e[0]), error_B=abs(e[1]), parameter_count=4)
    for t in range(1, steps+1):
        context, phase = regression_context(t, steps)
        # Both diagnostic gradients are measured but ONLY the observed one trains.
        alignment = dot(loss_gradient(w, 0), loss_gradient(w, 1))
        other_before = value(w, float(2-context))
        w = update(w, float(context+1), 0.5 if context == 0 else -0.5)
        unobserved_drift = abs(value(w, float(2-context))-other_before)
        error, e = regression_metrics(w)
        log(rows, t, error, steps, emit, phase, error_A=abs(e[0]), error_B=abs(e[1]),
            gradient_dot=alignment, unobserved_prediction_drift=unobserved_drift, parameter_count=4)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
