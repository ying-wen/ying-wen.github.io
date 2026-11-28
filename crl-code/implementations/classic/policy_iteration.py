"""策略迭代：独立、可运行的教学实现。
已知六格链模型，γ=0.95；横轴是同步扫描次数，绝非环境交互。
"""
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "policy_iteration",
    "name": "策略迭代",
    "family": "classic",
    "chapter_paths": [
        "algorithms/value/",
        "construction/planning/",
        "foundations/objectives/"
    ],
    "description": "已知六格链模型，γ=0.95；横轴是同步扫描次数，绝非环境交互。",
    "question": "在同一教学任务和预算下，实际更新怎样改变预测或策略？",
    "task": "chain_exact_planning",
    "baseline": "value_iteration",
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

def update(values, policy):
    # 固定策略的同步 Bellman 期望扫描；策略改善不读取半轮新值。
    return {s:model_target(s,policy[s],values) for s in values}

def improve(values):
    return {s:max(range(2),key=lambda a:model_target(s,a,values)) for s in values}

def run(seed=0, steps=1000, emit=None):
    check_steps(steps)
    values, policy, rows = dict.fromkeys(range(5),0.), dict.fromkeys(range(5),0), []
    truth = optimal_values()
    for sweep in range(1,steps+1):
        new = update(values,policy)
        residual = max(abs(new[s]-values[s]) for s in values)
        values = new
        # 评估收敛后才改善；有限预算可以停在评估阶段。
        if residual < 1e-10:
            policy = improve(values)
        log(rows,sweep,max(abs(values[s]-truth[s]) for s in values),steps,emit,backups=5*sweep)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
