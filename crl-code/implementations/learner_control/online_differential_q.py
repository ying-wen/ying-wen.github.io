"""Continual corridor: keep updating differential Q and gain after each step.

The environment, initial parameters, observation memory and action sampler are
identical to frozen_parameters.py. Both learn for the first 400 steps; only
the comparator then loses the permission to write Q/rate. Its memory continues.
Q uses (observed phase, remaining time, selected route, remembered cue).
The cue can become stale after an unannounced change. No theorem is claimed
for this nonstationary agent-state representation.
"""
from pathlib import Path
try:
    from ._common import run_control
except ImportError:
    from _common import run_control

META = {
    "id": "learner_control_online",
    "name": "持续走廊：在线差分 Q 与线索记忆",
    "family": "learner_control",
    "implementation_kind": "component_experiment",
    "coverage_refs": [],
    "task": "continuing_information_recovery_corridor",
    "baseline": "learner_control_frozen",
    "metric": "完整生命期每原始步的实际奖励",
    "unit": "reward_per_step",
    "higher_better": True,
    "budget": "environment_steps",
    "budget_note": "探测、走廊行进与失败恢复全部计入真实步；无外部重置，无免费评测交互。",
    "scope": "independent-component-experiment",
    "chapter_paths": ["algorithms/control/", "foundations/objectives/", "algorithms/average-reward/"],
    "sources": ["https://proceedings.mlr.press/v139/wan21a.html"],
    "defaults": {"steps": 1200},
    "initialization": "27个可达phase/clock/route×cue行的Q=0，gain=0，线索记忆空；初始动作均匀。",
    "algorithm_notes": "实验给定共同预热至400步；对照在400步后冻结，环境在600步改变。在线agent不读取变化标签、未来奖励或全局时钟。两分支不重置环境、记忆和行动RNG。",
    "description": "持续走廊含有成本的信息动作与四步失败恢复；600步路况改变。每个原始步更新差分Q和gain，线索记忆独立递推。",
    "question": "前400步同一历史和参数检查点之后，继续学习怎样改变整个人生的收益？",
    "limitations": "有限教学任务，不是一般偏离遗憾估计或非平稳收敛证明；对照在400步冻结参数而保留记忆及世界；无不可逆陷阱。",
}


def update(q, rate, key, action, reward, next_key, alpha=.1, eta=.05):
    # 1. The actual next observation has already updated memory. No task terminal.
    delta = reward-rate+max(q[next_key])-q[key][action]
    # 2. Both updates use exactly the same OLD-parameter TD error.
    q[key][action] += alpha*delta
    # 3. Estimated optimal gain is not the measured lifetime behavior reward.
    return rate+eta*alpha*delta, delta


def run(seed=0, steps=1200, emit=None):
    return run_control(seed, steps, emit, update, learns=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
