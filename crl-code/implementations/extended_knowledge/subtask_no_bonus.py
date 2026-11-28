"""无停止bonus训练基线：透明对照入口，不另冒充算法实现。
核心训练/评价见subtask_stopping.py；只改变use_bonus=False，其余任务与预算不变。
"""
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from .subtask_stopping import run as _experiment
from ._common import main

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": True,
    "id": "subtask_no_bonus",
    "name": "无停止bonus训练基线",
    "task": "ek_stopping_subtask",
    "baseline": "subtask_stopping",
    "ablation_of": "subtask_stopping",
    "metric": "起点冻结子任务目标回报",
    "unit": "return",
    "budget": "environment_steps",
    "coverage_refs": [],
    "chapter_paths": [
        "construction/goals/"
    ],
    "sources": [
        "https://arxiv.org/abs/2202.03466"
    ],
    "description": "透明消融/参照：调用 subtask_stopping.py 的训练流程，仅设置 use_bonus=False。0..6链，环境一步-.04，6终止得1；状态3到达停止bonus .8。均匀行为采样，α=.2 γ=.9；停止β由z(s')>=maxQ(s')，目标r+βz(s')+γ(1-β)V。冻结起点0评价最多30步。",
    "limitations": "baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？",
    "comparison_role": "baseline"
}

def run(seed=0,steps=1200,emit=None):
    return _experiment(seed,steps,emit,use_bonus=False)

if __name__=="__main__":
    main(META,run)
