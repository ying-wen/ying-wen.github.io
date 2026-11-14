"""仅动作熵无技能内奖基线：透明对照入口，不另冒充算法实现。
核心训练/评价见diayn_tabular.py；只改变intrinsic=False，其余任务与预算不变。
"""
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from .diayn_tabular import run as _experiment
from ._common import main

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": True,
    "id": "diayn_no_discriminator_reward",
    "name": "仅动作熵无技能内奖基线",
    "task": "ek_diayn_skills",
    "baseline": "diayn_tabular",
    "ablation_of": "diayn_tabular",
    "metric": "冻结技能与状态互信息",
    "unit": "nats",
    "budget": "environment_steps",
    "coverage_refs": [],
    "chapter_paths": [
        "construction/options/"
    ],
    "sources": [
        "https://arxiv.org/abs/1802.06070"
    ],
    "description": "透明消融/参照：调用 diayn_tabular.py 的训练流程，仅设置 intrinsic=False。5格链两skill等先验，长度8 skill固定episode；状态包含remaining-time。表格soft Q α=.2 γ=.95，最大熵actor α=.05 温度.1，discriminator CE α=.15；内奖logq(z|s')-log.5。冻结精确全部8时刻占用，计算真实joint MI。",
    "limitations": "baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？",
    "comparison_role": "baseline"
}

def run(seed=0,steps=1200,emit=None):
    return _experiment(seed,steps,emit,intrinsic=False)

if __name__=="__main__":
    main(META,run)
