"""Option-Critic固定β=.5消融：透明对照入口，不另冒充算法实现。
核心训练/评价见option_critic.py；只改变learn_beta=False，其余任务与预算不变。
"""
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from .option_critic import run as _experiment
from ._common import main

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": True,
    "id": "option_critic_fixed_beta",
    "name": "Option-Critic固定β=.5消融",
    "task": "ek_option_critic_chain",
    "baseline": "option_critic",
    "ablation_of": "option_critic",
    "metric": "冻结call-and-return起点折扣回报",
    "unit": "return",
    "budget": "environment_steps",
    "coverage_refs": [],
    "chapter_paths": [
        "construction/options/"
    ],
    "sources": [
        "https://arxiv.org/abs/1609.05140"
    ],
    "description": "透明消融/参照：调用 option_critic.py 的训练流程，仅设置 learn_beta=False。6格链，r=-.02/终点1，γ=.9；两option softmax内部策略，sigmoid β，每primitive Q_U/QΩ critic α=.15、actor .03、termination .03；ε=.2选择option，termination梯度在arrival state应用，终点跳过。精确冻结策略评价。",
    "limitations": "baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？",
    "comparison_role": "baseline"
}

def run(seed=0,steps=1200,emit=None):
    return _experiment(seed,steps,emit,learn_beta=False)

if __name__=="__main__":
    main(META,run)
