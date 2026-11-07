"""非线性 Gradient Monte Carlo：完整回报对照，不是假装流式资格迹。"""
import random
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "implementations.streaming_composition"
from ._common import Network, features, chain_error, check, record, main

META = {
    "id": "neural_gradient_mc",
    "name": "非线性完整回报 Gradient MC",
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
        "foundations/approximation/"
    ],
    "sources": [
        "https://incompleteideas.net/book/the-book-2nd.html"
    ],
    "description": "与 TD(λ) 同一六步链及 2–6–1 tanh 网络；α=.03。保存当前 episode 状态，结束后构造完整回报，按时间正序逐样本更新网络。真实终点无 bootstrap。",
    "limitations": "完整回报对照使用最多六条转移并等待终点；不满足严格无轨迹缓冲的流式约束。预算末尾未完成 episode 不更新、不伪装成终止。固定六步上界才使此例缓冲有界。",
    "question": "去掉 bootstrap 能消除哪一类目标依赖？它又增加了怎样的等待、存储和方差？",
    "defaults": {
        "steps": 1200
    }
}

def run(seed=0, steps=1200, emit=None):
    check(steps)
    rng = random.Random(seed+707)
    net = Network(seed=seed)
    state = 0
    trajectory = []
    updates = 0
    rows = []
    record(rows, 0, steps, chain_error(net), emit, retained_transitions=0)
    for t in range(1, steps+1):
        terminal = state == 5
        reward = float(rng.random() < .7) if terminal else 0.
        trajectory.append((features(state, 5), reward))
        if terminal:
            returns = [0.]*len(trajectory)
            total = 0.
            for j in range(len(trajectory)-1, -1, -1):
                total = trajectory[j][1]+.9*total
                returns[j] = total
            for (x, _), target in zip(trajectory, returns):
                # Each sample recomputes its gradient after earlier MC updates.
                value, gradient = net.value_gradient(x)
                net.add([(target-value)*g for g in gradient], .03)
                updates += 1
            trajectory.clear()
        state = 0 if terminal else state+1
        if t % max(1, steps//20) == 0 or t == steps:
            record(rows, t, steps, chain_error(net), emit,
                   retained_transitions=len(trajectory), max_trajectory_capacity=6,
                   update_sample_gradients=updates, completed_episodes=t//6)
    # An unfinished trajectory is deliberately NOT forced to terminate.
    return rows

if __name__ == "__main__":
    main(META, run)
