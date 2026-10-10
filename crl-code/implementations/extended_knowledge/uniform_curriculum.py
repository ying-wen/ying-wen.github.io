"""均匀任务课程基线：透明对照入口，不另冒充算法实现。
核心训练/评价见learning_progress.py；只改变adaptive=False，其余任务与预算不变。
"""
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from .learning_progress import run as _experiment
from ._common import main

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": False,
    "id": "uniform_curriculum",
    "name": "均匀任务课程基线",
    "task": "ek_progress_regression",
    "baseline": "learning_progress",
    "ablation_of": "learning_progress",
    "metric": "冻结三任务期望预测MSE",
    "unit": "mse",
    "budget": "training_examples",
    "coverage_refs": [],
    "chapter_paths": [
        "algorithms/exploration/"
    ],
    "sources": [
        "https://doi.org/10.1109/TEVC.2006.890271"
    ],
    "description": "透明消融/参照：调用 learning_progress.py 的训练流程，仅设置 adaptive=False。三个标量回归任务斜率1/-2/0，第三有σ=1不可约噪声；各任务SGD .05，20误差队列两个10窗均值差绝对值，ε=.2最高progress调度；固定真实斜率只用于环境/冻结MSE评价。",
    "limitations": "baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？",
    "comparison_role": "baseline"
}

def run(seed=0,steps=1200,emit=None):
    return _experiment(seed,steps,emit,adaptive=False)

if __name__=="__main__":
    main(META,run)
