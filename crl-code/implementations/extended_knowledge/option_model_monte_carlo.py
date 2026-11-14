"""完整option蒙特卡洛模型基线：透明对照入口，不另冒充算法实现。
核心训练/评价见option_model.py；只改变td=False，其余任务与预算不变。
"""
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from .option_model import run as _experiment
from ._common import main

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": False,
    "id": "option_model_monte_carlo",
    "name": "完整option蒙特卡洛模型基线",
    "task": "ek_option_model",
    "baseline": "option_model",
    "ablation_of": "option_model",
    "metric": "冻结reward/discounted-endpoint联合RMSE",
    "unit": "rmse",
    "budget": "environment_steps",
    "coverage_refs": [],
    "chapter_paths": [
        "construction/models/"
    ],
    "sources": [
        "https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/sutton%2Bal-1999.pdf"
    ],
    "description": "透明消融/参照：调用 option_model.py 的训练流程，仅设置 td=False。7格链；固定option向右概率.8，β=.3（终点6强制止），γ=.9，r=-.02/终点1；α=.15逐primitive TD学R与Pγ。每次option终止随机起点0..5，与精确固定点RMSE比较。",
    "limitations": "baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？",
    "comparison_role": "baseline"
}

def run(seed=0,steps=1200,emit=None):
    return _experiment(seed,steps,emit,td=False)

if __name__=="__main__":
    main(META,run)
