"""Lagged-target semi-gradient prediction: clone storage, explicit copy clock.

Copies happen AFTER update 50, 100, ...; the transition at step 50 still uses
the old target. A target is detached throughout each block, yet jumps when copied.
Target age, target drift, and copy events are logged; evaluation uses online w.
"""
from pathlib import Path
try:
    from ._common import value, gradient, nonlinear_init, two_state_error, log, validate
except ImportError:
    from _common import value, gradient, nonlinear_init, two_state_error, log, validate

META = {
    "id": "lagged_nonlinear_td",
    "name": "非线性 TD：冻结 50 步目标副本",
    "family": "nonlinear_diagnostics",
    "coverage_refs": [],
    "implementation_kind": "component_experiment",
    "task": "nonlinear_target_diagnostic",
    "baseline": "online_nonlinear_td",
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
    "description": "相同二状态流；bootstrap 用独立参数副本，每 50 次更新后复制在线参数。当前预测仍逐步训练。",
    "question": "冻结目标改变了哪一条依赖，又引入了多少滞后？",
    "limitations": "目标副本多保存四参数。没有 replay 或控制；不等于 DQN 实现。冻结期间的局部回归稳定不推出跨复制时刻全局收敛。"
}

def update(w, target_w, x, reward, next_x, alpha=0.03, gamma=0.8):
    # Do not alias target_w to w. Gradients are evaluated only on current prediction.
    target = reward + gamma*value(target_w, next_x)
    residual, g = target-value(w, x), gradient(w, x)
    return [p+alpha*residual*q for p, q in zip(w, g)], target

def run(seed=0, steps=1200, emit=None):
    validate(steps)
    w, rows, state, age = nonlinear_init(seed), [], 0, 0
    target_w = w[:]
    log(rows, 0, two_state_error(w), steps, emit, target_drift=0.0, target_age=0, target_copies=0)
    for t in range(1, steps+1):
        x, next_x, reward = (-1.0, 1.0, 0.0) if state == 0 else (1.0, -1.0, 1.0)
        w, target_before = update(w, target_w, x, reward, next_x)
        age += 1
        if t % 50 == 0:
            target_w, age = w[:], 0
        drift = abs(reward + 0.8*value(target_w, next_x)-target_before)
        state = 1-state
        log(rows, t, two_state_error(w), steps, emit, target_drift=drift, target_age=age, target_copies=t//50)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
