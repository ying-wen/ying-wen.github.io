"""Stop-gradient does not mean a target stays fixed across optimization steps.

This file uses analytic derivatives so the intended graph boundary is visible:
delta=r+gamma*v(w,x_next)-v(w,x), update=alpha*delta*grad_v(w,x).
There is NO gamma*grad_v(w,x_next) in the update. Nevertheless recomputing
the bootstrap with new w generally changes the next target.
"""
from pathlib import Path
try:
    from ._common import value, gradient, nonlinear_init, two_state_error, log, validate
except ImportError:
    from _common import value, gradient, nonlinear_init, two_state_error, log, validate

META = {
    "id": "online_nonlinear_td",
    "name": "非线性 TD：即时自举目标",
    "family": "nonlinear_diagnostics",
    "coverage_refs": [],
    "implementation_kind": "component_experiment",
    "task": "nonlinear_target_diagnostic",
    "baseline": "lagged_nonlinear_td",
    "metric": "二状态真实价值 RMSE",
    "unit": "value",
    "higher_better": False,
    "budget": "environment_steps",
    "scope": "analytic-diagnostic",
    "budget_note": "每步一个环境转移与一次梯度更新；冻结目标法额外保存4参数，每50步复制。",
    "chapter_paths": [
        "foundations/learning-dynamics/",
        "foundations/approximation/prediction/",
        "algorithms/plasticity/"
    ],
    "sources": [
        "https://github.com/google-deepmind/dqn",
        "https://arxiv.org/abs/1812.02648"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "A/B 确定性交替，奖励 0/1，gamma=0.8。四参数 tanh 价值函数，单步半梯度 TD；不回放。",
    "question": "stop-gradient 只切断计算图，是否已经让目标不随参数更新变化？",
    "limitations": "小型固定策略预测，不含控制、不含分布变化。例子用于检查时序和目标漂移，不承诺目标网络更优或半梯度必发散。"
}

def update(w, x, reward, next_x, alpha=0.03, gamma=0.8):
    target = reward + gamma*value(w, next_x)
    residual, g = target-value(w, x), gradient(w, x)
    return [p+alpha*residual*q for p, q in zip(w, g)], target

def run(seed=0, steps=1200, emit=None):
    validate(steps)
    w, rows, state = nonlinear_init(seed), [], 0
    log(rows, 0, two_state_error(w), steps, emit, target_drift=0.0, target_age=0)
    for t in range(1, steps+1):
        x, next_x, reward = (-1.0, 1.0, 0.0) if state == 0 else (1.0, -1.0, 1.0)
        w, target_before = update(w, x, reward, next_x)
        drift = abs(reward + 0.8*value(w, next_x)-target_before)
        state = 1-state  # Never terminate merely because an experiment budget ends.
        log(rows, t, two_state_error(w), steps, emit, target_drift=drift, target_age=0)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
