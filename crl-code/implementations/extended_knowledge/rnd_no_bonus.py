"""同预测器无内奖探索基线：透明对照入口，不另冒充算法实现。
核心训练/评价见rnd_exploration.py；只改变use_bonus=False，其余任务与预算不变。
"""
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from .rnd_exploration import run as _experiment
from ._common import main

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": True,
    "id": "rnd_no_bonus",
    "name": "同预测器无内奖探索基线",
    "task": "ek_rnd_chain",
    "baseline": "rnd_exploration",
    "ablation_of": "rnd_exploration",
    "metric": "在线已访问状态比例",
    "unit": "fraction",
    "budget": "environment_steps",
    "coverage_refs": [],
    "chapter_paths": [
        "algorithms/exploration/"
    ],
    "sources": [
        "https://arxiv.org/abs/1810.12894",
        "https://github.com/openai/random-network-distillation"
    ],
    "description": "透明消融/参照：调用 rnd_exploration.py 的训练流程，仅设置 use_bonus=False。12格反射链起点0；固定随机MLP12→12→3与独立预测MLP，MSE SGD .03；Q α=.2 γ=.99 ε=.15，内奖为更新前误差，预测器每交互更新；无外奖，访问覆盖为指标。",
    "limitations": "baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？",
    "comparison_role": "baseline"
}

def run(seed=0,steps=1200,emit=None):
    return _experiment(seed,steps,emit,use_bonus=False)

if __name__=="__main__":
    main(META,run)
