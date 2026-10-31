"""逐目标表格Q基线：透明对照入口，不另冒充算法实现。
核心训练/评价见uvfa_shared.py；只改变tabular=True，其余任务与预算不变。
"""
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from .uvfa_shared import run as _experiment
from ._common import main

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": True,
    "id": "uvfa_tabular",
    "name": "逐目标表格Q基线",
    "task": "ek_goal_navigation",
    "baseline": "uvfa_shared",
    "ablation_of": "uvfa_shared",
    "metric": "七目标冻结贪心到达率",
    "unit": "fraction",
    "budget": "environment_steps",
    "coverage_refs": [],
    "chapter_paths": [
        "construction/goals/"
    ],
    "sources": [
        "https://proceedings.mlr.press/v37/schaul15.html"
    ],
    "description": "透明消融/参照：调用 uvfa_shared.py 的训练流程，仅设置 tabular=True。7格链，随机起点/目标，负一步代价；输入(s,g,s-g,|s-g|)共享MLP4→16→2，SGD .03，γ=.95，ε=.25；最长12步仅截断。每20步冻结遍历42对起点目标。",
    "limitations": "baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？",
    "comparison_role": "baseline"
}

def run(seed=0,steps=1200,emit=None):
    return _experiment(seed,steps,emit,tabular=True)

if __name__=="__main__":
    main(META,run)
