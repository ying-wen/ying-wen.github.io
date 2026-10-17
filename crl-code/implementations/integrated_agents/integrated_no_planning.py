"""闭环对照：仅真实更新。
仅移除模型规划；仍学习模型并记录预测误差。
学习步骤和模型契约见 _system.py；本文件是独立可运行的命名配置。
结果记录真实环境时钟。请同时读 model_backups，不能把环境步匹配说成计算预算匹配。
"""
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "implementations.integrated_agents"
from ._system import run_system
from implementations.runtime import run_cli


def run(seed=0, steps=1200, emit=None):
    return run_system(seed, steps, emit, planning=0, recent=True, versioned=True)


META = {
    "id": "integrated_no_planning",
    "name": "闭环对照：仅真实更新",
    "family": "integrated_agents",
    "task": "integrated_continuing_ring",
    "baseline": "integrated_recent_model",
    "metric": "最近100个真实步的平均外部奖励",
    "unit": "reward_per_environment_step",
    "budget": "environment_steps",
    "higher_better": True,
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "coverage_refs": [],
    "chapter_paths": [
        "construction/architectures/",
        "algorithms/average-reward/",
        "construction/options/",
        "construction/predictive-knowledge/"
    ],
    "sources": [
        "https://arxiv.org/abs/2208.11173",
        "https://arxiv.org/abs/2202.03466",
        "https://arxiv.org/abs/2006.16318"
    ],
    "description": "七状态持续随机环；外部奖励位置半程改变，agent 不知道变化时刻。仅移除模型规划；仍学习模型并记录预测误差。原子控制 ε=.2；技能内 ε=.1，最长6步，给定目标1/4。高层α=.1、奖励率增量=.002δ；每次宏动作结束做所设次数规划，并记录模型备份数。",
    "limitations": "原创集成教学原型，非 STOMP/OaK 原论文复现。状态和子目标手工给定，学子任务策略但不学终止规则。常步长非平稳 SMDP 组合无收敛保证。模型预测误差只在已存在模型的完整宏动作上记录，不是无偏全空间误差。",
    "question": "改变模型记忆和技能策略版本管理后，真实收益、预测误差及规划成本怎样变化？",
    "ablation_of": "integrated_recent_model"
}

if __name__ == "__main__":
    run_cli(META, run)

