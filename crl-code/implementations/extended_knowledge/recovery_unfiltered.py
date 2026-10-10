"""不经过恢复过滤的基线：透明对照入口，不另冒充算法实现。
核心训练/评价见recovery_filter.py；只改变filtered=False，其余任务与预算不变。
"""
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from .recovery_filter import run as _experiment
from ._common import main

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": True,
    "id": "recovery_unfiltered",
    "name": "不经过恢复过滤的基线",
    "task": "ek_reset_filter",
    "baseline": "recovery_filter",
    "ablation_of": "recovery_filter",
    "metric": "在线净奖励均值（含恢复失败成本）",
    "unit": "return",
    "budget": "reset_trials",
    "coverage_refs": [],
    "chapter_paths": [
        "algorithms/exploration/"
    ],
    "sources": [
        "https://arxiv.org/abs/1711.06782"
    ],
    "description": "透明消融/参照：调用 recovery_filter.py 的训练流程，仅设置 filtered=False。三种前进行为，收益0/.4/1，真实恢复概率.98/.6/.05，仅环境知道；失败成本2。前30trial轮流收集恢复证据，之后Beta(1,1)后验均值过滤阈值.75、无合格选最高估计；奖励Q均值ε=.2；统计全部trial。",
    "limitations": "baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？",
    "comparison_role": "baseline"
}

def run(seed=0,steps=1200,emit=None):
    return _experiment(seed,steps,emit,filtered=False)

if __name__=="__main__":
    main(META,run)
