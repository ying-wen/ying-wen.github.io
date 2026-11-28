"""错误诊断：忽略 option 时长。自己的完整 primitive 交互与高层更新循环。

两种程序仅改 continuation。行为均匀，与 critic 参数无关，所以同 seed
生成完全相同轨迹。负对照检验语义，不用于比较两个正确研究算法的优劣。
"""
import random
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "implementations.streaming_composition"
from ._common import Network, features, option_step, option_truth, option_error, check, record, main

META = {
    "id": "neural_option_wrong_duration",
    "name": "错误诊断：忽略 option 时长",
    "family": "streaming_composition",
    "task": "sc_neural_option_duration",
    "baseline": "neural_option_smdp",
    "metric": "相对真实 option Bellman 最优值的 RMSE",
    "unit": "RMSE",
    "higher_better": False,
    "budget": "environment_steps",
    "implementation_kind": "component_experiment",
    "scope": "diagnostic-counterexample",
    "coverage_refs": [],
    "chapter_paths": [
        "construction/options/"
    ],
    "sources": [
        "https://doi.org/10.1016/S0004-3702(99)00052-1"
    ],
    "description": "四状态环，两给定闭环 options，持续时间 1–2 primitive 步；高层均匀探索，2–6–2 tanh 网络进行 option Q-learning，α=.025，无 replay 或 target network。故意将 continuation γ^τ 错写成 γ。固定真实 primitive 步预算，绘图与精确小模型 Q* 比较。",
    "limitations": "这是故意错误的负对照，不是可推荐研究算法。给定 option、Markov 状态、固定表示结构；没有实现 option 发现或神经 Option-Critic。未完成 option 不在预算末尾伪终止。",
    "question": "为什么 option 停止不是环境终止？忽略执行时长为何会改变问题的 Bellman 方程？",
    "defaults": {
        "steps": 1200
    }
}

def run(seed=0, steps=1200, emit=None):
    check(steps)
    rng = random.Random(seed+991)
    net = Network(outputs=2, seed=seed)
    truth = option_truth()
    state = start_state = 0
    option = rng.randrange(2)
    reward_sum, discount, duration, updates = 0., 1., 0, 0
    rows = []
    record(rows, 0, steps, option_error(net, truth), emit, option_updates=0)
    for t in range(1, steps+1):
        sn, reward, option_ended = option_step(state, option)
        reward_sum += discount*reward
        discount *= .9
        duration += 1
        if option_ended:
            value, gradient = net.value_gradient(features(start_state), option)
            # Option ending reopens high-level choice; it does not kill its value.
            next_value = max(net.value(features(sn), o) for o in range(2))
            target = reward_sum+.9*next_value
            net.add([(target-value)*g for g in gradient], .025)
            updates += 1
            start_state = sn
            option = rng.randrange(2)
            reward_sum, discount, duration = 0., 1., 0
        state = sn
        if t % max(1, steps//20) == 0 or t == steps:
            record(rows, t, steps, option_error(net, truth), emit,
                   option_updates=updates, unfinished_duration=duration,
                   primitive_steps=t, incorrect_duration_target=True)
    return rows

if __name__ == "__main__":
    main(META, run)

