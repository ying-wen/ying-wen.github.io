"""非线性半梯度 TD(0)：每步更新；终止奖励先分配，再清零迹。
记录的 lag-one 梯度差只诊断表示变化，不等于完整历史迹误差。
"""
import random
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "implementations.streaming_composition"
from ._common import Network, features, trace_step, chain_error, norm, check, record, main

META = {
    "id": "neural_td0",
    "name": "非线性半梯度 TD(0)",
    "family": "streaming_composition",
    "task": "sc_neural_delayed_prediction",
    "baseline": "neural_td_trace",
    "metric": "全状态预测 RMSE",
    "unit": "RMSE",
    "higher_better": False,
    "budget": "environment_steps",
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "coverage_refs": [],
    "chapter_paths": [
        "algorithms/credit-assignment/",
        "construction/predictive-knowledge/"
    ],
    "sources": [
        "https://proceedings.mlr.press/v32/seijen14.html",
        "https://arxiv.org/abs/2507.09087"
    ],
    "description": "固定策略六步链，末步 Bernoulli(.7) 奖励，γ=.9；2–6–1 tanh 网络，α=.03，λ=0，逐条转移更新，不重放。记录资格范数与同一旧输入的梯度变化。",
    "limitations": "不是 True-online 非线性扩展，也不是神经 GTD；旧梯度迹与当前网络重算的迹不同。固定 Markov 输入、平稳目标、小型同策略预测。与 MC 的交互预算相同，但梯度计算和更新时刻不同。",
    "question": "当价值梯度随参数改变时，资格迹保存的究竟是哪一时刻的梯度？",
    "defaults": {
        "steps": 1200
    }
}

def run(seed=0, steps=1200, emit=None):
    check(steps)
    rng = random.Random(seed+707)
    net = Network(seed=seed)
    trace = [0.]*len(net.w)
    state = 0
    gamma_in = 0.
    rows = []
    previous_x = previous_gradient = None
    record(rows, 0, steps, chain_error(net), emit, trace_norm=0., gradient_drift=0.)
    for t in range(1, steps+1):
        x = features(state, 5)
        value, gradient = net.value_gradient(x)
        terminal = state == 5
        reward = float(rng.random() < .7) if terminal else 0.
        gamma_out = 0. if terminal else .9
        next_value = 0. if terminal else net.value(features(state+1, 5))
        delta = reward+gamma_out*next_value-value
        # Incoming gamma decays old eligibility. Outgoing gamma controls target.
        trace = trace_step(trace, gradient, gamma_in, 0)
        drift = 0.
        if previous_x is not None:
            now = net.value_gradient(previous_x)[1]
            drift = norm([a-b for a, b in zip(now, previous_gradient)])
        net.add([delta*z for z in trace], .03)
        previous_x, previous_gradient = x, gradient
        trace_norm = norm(trace)
        state = 0 if terminal else state+1
        gamma_in = gamma_out
        # Do not erase earlier states before assigning the final reward.
        if terminal:
            trace = [0.]*len(net.w)
            previous_x = previous_gradient = None
        if t % max(1, steps//20) == 0 or t == steps:
            record(rows, t, steps, chain_error(net), emit, trace_norm=trace_norm,
                   gradient_drift=drift, update_sample_gradients=t,
                   completed_episodes=t//6, retained_transitions=0)
    return rows

if __name__ == "__main__":
    main(META, run)
