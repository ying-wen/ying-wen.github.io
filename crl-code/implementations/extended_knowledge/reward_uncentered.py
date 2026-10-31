"""不中心化的相同折扣TD基线：透明对照入口，不另冒充算法实现。
核心训练/评价见reward_centering.py；只改变centered=False，其余任务与预算不变。
"""
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from .reward_centering import run as _experiment
from ._common import main

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": False,
    "id": "reward_uncentered",
    "name": "不中心化的相同折扣TD基线",
    "task": "ek_centered_prediction",
    "baseline": "reward_centering",
    "ablation_of": "reward_centering",
    "metric": "冻结重构原折扣价值RMSE",
    "unit": "rmse",
    "budget": "environment_steps",
    "coverage_refs": [],
    "chapter_paths": [
        "algorithms/average-reward/"
    ],
    "sources": [
        "https://arxiv.org/abs/2405.09999"
    ],
    "description": "透明消融/参照：调用 reward_centering.py 的训练流程，仅设置 centered=False。三状态确定性continuing环，奖励(9,10,11)，γ=.99 α=.1，on-policy V TD；参照c+=ηαδ (η=.05)，目标r-c+γV(s')；原值重构V+c/(1-γ)，解析真值从环几何级数；没有done或episodes。",
    "limitations": "baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？",
    "comparison_role": "baseline"
}

def run(seed=0,steps=1200,emit=None):
    return _experiment(seed,steps,emit,centered=False)

if __name__=="__main__":
    main(META,run)
