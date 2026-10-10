"""仅外奖η=0基线：透明对照入口，不另冒充算法实现。
核心训练/评价见intrinsic_meta_gradient.py；只改变learn_reward=False，其余任务与预算不变。
"""
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from .intrinsic_meta_gradient import run as _experiment
from ._common import main

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": True,
    "id": "intrinsic_fixed_reward",
    "name": "仅外奖η=0基线",
    "task": "ek_intrinsic_meta",
    "baseline": "intrinsic_meta_gradient",
    "ablation_of": "intrinsic_meta_gradient",
    "metric": "冻结更新后外在奖励期望",
    "unit": "return",
    "budget": "meta_updates",
    "coverage_refs": [],
    "chapter_paths": [
        "foundations/reward-design/"
    ],
    "sources": [
        "https://arxiv.org/abs/1804.06459"
    ],
    "description": "透明消融/参照：调用 intrinsic_meta_gradient.py 的训练流程，仅设置 learn_reward=False。二动作bandit外奖(0,1)，内奖η=0即只用外奖。logit θ用精确策略梯度 α=.1 内层一步，外层梯度虽计算但不更新η；θ在线推进。每step一个内层更新，额外梯度评价计数。",
    "limitations": "baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？",
    "comparison_role": "baseline"
}

def run(seed=0,steps=1200,emit=None):
    return _experiment(seed,steps,emit,learn_reward=False)

if __name__=="__main__":
    main(META,run)
