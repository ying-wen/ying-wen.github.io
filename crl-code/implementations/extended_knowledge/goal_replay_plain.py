"""不重标目标的相同回放基线：透明对照入口，不另冒充算法实现。
核心训练/评价见her_replay.py；只改变hindsight=False，其余任务与预算不变。
"""
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from .her_replay import run as _experiment
from ._common import main

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": True,
    "id": "goal_replay_plain",
    "name": "不重标目标的相同回放基线",
    "task": "ek_her_navigation",
    "baseline": "her_replay",
    "ablation_of": "her_replay",
    "metric": "七目标冻结贪心到达率",
    "unit": "fraction",
    "budget": "environment_steps",
    "coverage_refs": [],
    "chapter_paths": [
        "construction/goals/"
    ],
    "sources": [
        "https://arxiv.org/abs/1707.01495"
    ],
    "description": "透明消融/参照：调用 her_replay.py 的训练流程，仅设置 hindsight=False。7格确定性目标独立链，每回合至多12步；表格Q .2、γ=.95、ε=.4；完成轨迹随机future目标重新算reward/done，每交互另做2次回放；评价42起点目标。",
    "limitations": "baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？",
    "comparison_role": "baseline"
}

def run(seed=0,steps=1200,emit=None):
    return _experiment(seed,steps,emit,hindsight=False)

if __name__=="__main__":
    main(META,run)
