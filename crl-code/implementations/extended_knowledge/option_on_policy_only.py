"""仅执行option的on-policy更新基线：透明对照入口，不另冒充算法实现。
核心训练/评价见intra_option_q.py；只改变all_options=False，其余任务与预算不变。
"""
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from .intra_option_q import run as _experiment
from ._common import main

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": True,
    "id": "option_on_policy_only",
    "name": "仅执行option的on-policy更新基线",
    "task": "ek_intra_option_chain",
    "baseline": "intra_option_q",
    "ablation_of": "intra_option_q",
    "metric": "冻结call-and-return起点折扣回报",
    "unit": "return",
    "budget": "environment_steps",
    "coverage_refs": [],
    "chapter_paths": [
        "construction/options/"
    ],
    "sources": [
        "https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/sutton%2Bal-1999.pdf"
    ],
    "description": "透明消融/参照：调用 intra_option_q.py 的训练流程，仅设置 all_options=False。6格链起点0终点5，奖励1否则-.02；两固定随机option右动作概率.2/.8，β=.25，α=.1 γ=.9；行为执行active option，终止后ε=.2重新选择。每转移对全部option重要性加权备份，冻结精确call-and-return评价。",
    "limitations": "baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？",
    "comparison_role": "baseline"
}

def run(seed=0,steps=1200,emit=None):
    return _experiment(seed,steps,emit,all_options=False)

if __name__=="__main__":
    main(META,run)
