"""仅primitive的相同模型备份基线：透明对照入口，不另冒充算法实现。
核心训练/评价见option_value_iteration.py；只改变include_options=False，其余任务与预算不变。
"""
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from .option_value_iteration import run as _experiment
from ._common import main

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": False,
    "id": "primitive_value_iteration",
    "name": "仅primitive的相同模型备份基线",
    "task": "ek_option_planning",
    "baseline": "option_value_iteration",
    "ablation_of": "option_value_iteration",
    "metric": "冻结精确最优价值最大误差",
    "unit": "absolute_error",
    "budget": "model_backups",
    "coverage_refs": [],
    "chapter_paths": [
        "construction/planning/"
    ],
    "sources": [
        "https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/sutton%2Bal-1999.pdf"
    ],
    "description": "透明消融/参照：调用 option_value_iteration.py 的训练流程，仅设置 include_options=False。7格链γ=.9，两个primitive及固定随机option（右概率.8，β=.3）；已知R/Pγ有限模型，随机状态原子最大化backup，每step一个state backup。真V由primitive最优右移闭式计算。",
    "limitations": "baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？",
    "comparison_role": "baseline"
}

def run(seed=0,steps=1200,emit=None):
    return _experiment(seed,steps,emit,include_options=False)

if __name__=="__main__":
    main(META,run)
