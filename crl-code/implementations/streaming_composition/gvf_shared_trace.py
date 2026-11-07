"""共享神经 GVF + 资格迹：独立的实时更新循环。

读取顺序：question → 旧网络预测及梯度 → 各自 ratio/trace → 同时更新。
辅助问题中点变号是人为受控干预，不是一个完整终身机器人任务。
冻结对照不改变 c、γ 或 π；分离对照额外使用一个表示网络。
"""
import random
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "implementations.streaming_composition"
from ._common import Network, features, trace_step, gvf_question, gvf_error, norm, dot, check, record, main

META = {
    "id": "gvf_shared_trace",
    "name": "共享神经 GVF + 资格迹",
    "family": "streaming_composition",
    "task": "sc_neural_gvf_interference",
    "baseline": "gvf_frozen_trace",
    "metric": "固定主问题的全状态预测 RMSE",
    "unit": "RMSE",
    "higher_better": False,
    "budget": "environment_steps",
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "coverage_refs": [],
    "chapter_paths": [
        "construction/predictive-knowledge/",
        "algorithms/credit-assignment/"
    ],
    "sources": [
        "https://sites.ualberta.ca/~amw8/horde.pdf",
        "https://www.jmlr.org/papers/v17/14-488.html"
    ],
    "description": "四状态环上两个不同 π/c/γ 的问题；行为随机一次真实转移，各头普通 IS-TD(0.6) 更新一次。共享 2–6–2 tanh 网络，训练全部参数，α=.01。中点仅辅助 cumulant 变号，主问题不变。",
    "limitations": "不是 Horde 原作者实现或神经 GTD；普通 IS 半梯度无一般稳定保证。共享更新可改变另一问题的预测。 λ 改变信用也可能改变逼近固定点。辅助变号后旧迹不清除；它是环境信号变化，非预测终止。",
    "question": "同一经验流上的多问题预测怎样因共享非线性表示、旧梯度迹和不同策略而互相影响？",
    "defaults": {
        "steps": 1200
    }
}

def run(seed=0, steps=1200, emit=None):
    check(steps)
    rng = random.Random(seed+301)
    networks = [Network(outputs=2, seed=seed)]
    # Each question keeps its own trace and incoming continuation.
    traces = [[0.]*len(networks[0].w) for h in range(2)]
    gamma_in = [0., 0.]
    state = 0
    rows = []
    error = gvf_error(networks, separate=False)
    record(rows, 0, steps, error[0], emit, auxiliary_rmse=error[1],
           trainable_parameters=sum(len(n.w) for n in networks))
    for t in range(1, steps+1):
        # The two prediction questions do NOT each move the environment.
        action = int(rng.random() < .5)
        sn = (state+action) % 4
        sign = 1. if t <= steps//2 else -1.
        directions = [[0.]*len(n.w) for n in networks]
        question_directions = []
        for h in range(2):
            index = 0
            head = h
            net = networks[index]
            c, gamma_out, rho = gvf_question(h, sn, action, sign)
            v, gradient = net.value_gradient(features(state), head, freeze_trunk=False)
            # Semi-gradient: next prediction is a number, not a differentiated target.
            delta = c+gamma_out*net.value(features(sn), head)-v
            traces[h] = trace_step(traces[h], gradient, gamma_in[h], 0.6, rho)
            direction = [delta*z for z in traces[h]]
            question_directions.append(direction)
            directions[index] = [a+b for a, b in zip(directions[index], direction)]
            gamma_in[h] = gamma_out
        # Both deltas and both traces were computed from the SAME old parameters.
        for net, direction in zip(networks, directions):
            net.add(direction, .01)
        state = sn
        if t % max(1, steps//20) == 0 or t == steps:
            error = gvf_error(networks, separate=False, sign=sign)
            shared_dot = dot(question_directions[0][:net.trunk_size], question_directions[1][:net.trunk_size])
            record(rows, t, steps, error[0], emit,
                   phase="before-auxiliary-change" if sign == 1 else "after-auxiliary-change",
                   auxiliary_rmse=error[1], trace_norm=norm(traces[0]),
                   shared_update_inner_product=shared_dot,
                   network_updates=t*len(networks), environment_steps=t)
    return rows

if __name__ == "__main__":
    main(META, run)

