"""仅右动作特征MaxEnt拟合基线：透明对照入口，不另冒充算法实现。
核心训练/评价见maxent_irl.py；只改变full_features=False，其余任务与预算不变。
"""
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from .maxent_irl import run as _experiment
from ._common import main

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": False,
    "id": "maxent_single_feature",
    "name": "仅右动作特征MaxEnt拟合基线",
    "task": "ek_maxent_paths",
    "baseline": "maxent_irl",
    "ablation_of": "maxent_irl",
    "metric": "冻结专家分布轨迹交叉熵",
    "unit": "nats",
    "budget": "gradient_updates",
    "coverage_refs": [],
    "chapter_paths": [
        "foundations/reward-design/"
    ],
    "sources": [
        "https://www.cs.cmu.edu/~bziebart/publications/maximum-entropy-inverse-reinforcement-learning.html"
    ],
    "description": "透明消融/参照：调用 maxent_irl.py 的训练流程，仅设置 full_features=False。4层binary-action确定性DAG，枚举16合法完整轨迹，features=(右边数,转向数)，固定64专家轨迹由θ=(.8,-.5)全局分布采样。每step全批NLL SGD .05，logZ=logsumexp轨迹回报；冻结真实专家分布交叉熵。",
    "limitations": "baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？",
    "comparison_role": "baseline"
}

def run(seed=0,steps=1200,emit=None):
    return _experiment(seed,steps,emit,full_features=False)

if __name__=="__main__":
    main(META,run)
