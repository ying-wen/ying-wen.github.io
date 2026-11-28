"""价值迭代：独立、可运行的教学实现。
已知六格链模型，γ=0.95；横轴是同步扫描次数，绝非环境交互。
"""
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "value_iteration",
    "name": "价值迭代",
    "family": "classic",
    "chapter_paths": [
        "algorithms/value/",
        "construction/planning/",
        "foundations/objectives/"
    ],
    "description": "已知六格链模型，γ=0.95；横轴是同步扫描次数，绝非环境交互。",
    "question": "在同一教学任务和预算下，实际更新怎样改变预测或策略？",
    "task": "chain_exact_planning",
    "baseline": "policy_iteration",
    "metric": "最优价值最大绝对误差",
    "unit": "value",
    "higher_better": False,
    "scope": "exact-planning",
    "sources": [
        "http://incompleteideas.net/book/the-book-2nd.html"
    ],
    "defaults": {
        "steps": 1000
    },
    "budget": "model_sweeps",
    "budget_note": "每次同步扫描含5个状态备份；曲线横轴为扫描次数。",
    "initialization": "V=0，策略迭代初始策略总向左；已知确定模型。",
    "limitations": "小型教学任务；曲线不构成一般性能或持续学习优势的证据。"
}

def update(values):
    # 同步扫描：整轮所有目标读取同一份旧 V。
    return {s:max(model_target(s,a,values) for a in range(2)) for s in values}

def run(seed=0, steps=1000, emit=None):
    check_steps(steps)
    values, rows, truth = dict.fromkeys(range(5),0.), [], optimal_values()
    for sweep in range(1,steps+1):
        values = update(values)
        log(rows,sweep,max(abs(values[s]-truth[s]) for s in values),steps,emit,backups=5*sweep)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
