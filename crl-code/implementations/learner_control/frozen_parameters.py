"""Parameter-frozen comparator; observation memory is NOT frozen or cleared.

Both conditions learn identically for the first 400 primitive steps. This
comparator then retains exactly that checkpoint's Q/rate, without resetting
the world, memory, or action RNG. Queried cues still update its memory.
It is a checkpoint-matched diagnostic, not a new learning algorithm.
"""
from pathlib import Path
try:
    from ._common import run_control, FREEZE_STEP
    from .online_differential_q import update as learning_update
except ImportError:
    from _common import run_control, FREEZE_STEP
    from online_differential_q import update as learning_update

META = {
    "id": "learner_control_frozen",
    "name": "持续走廊：400步后冻结参数，保留记忆",
    "family": "learner_control",
    "implementation_kind": "component_experiment",
    "coverage_refs": [],
    "task": "continuing_information_recovery_corridor",
    "baseline": "learner_control_online",
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
    "algorithm_notes": "实验给定共同预热至400步；本分支随后冻结Q/率，环境在600步改变。冻结时刻由实验协议指定，不向agent提供变化标签；不重置世界、记忆或RNG。",
    "description": "前400步与在线方法完全相同；之后冻结该检查点的Q与gain。世界不重置，探测得到的新线索仍进入记忆，真实探测/恢复代价仍计入收益。",
    "question": "冻结参数而保留活动记忆，与重置智能体或冻结全部状态有什么区别？",
    "limitations": "冻结既有参数的检查点匹配诊断，不是所有固定控制器的最优基线；活动记忆仍更新，因此也不是冻结完整内部状态。",
}


def update(q, rate, key, action, reward, next_key):
    # Compute an observable diagnostic without modifying any parameter.
    delta = reward-rate+max(q[next_key])-q[key][action]
    return rate, delta


def run(seed=0, steps=1200, emit=None):
    calls = [0]
    def scheduled_update(q, rate, key, action, reward, next_key):
        calls[0] += 1
        if calls[0] <= FREEZE_STEP:
            return learning_update(q, rate, key, action, reward, next_key)
        return update(q, rate, key, action, reward, next_key)
    return run_control(seed, steps, emit, scheduled_update,
                       learns=lambda t: t <= FREEZE_STEP)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
