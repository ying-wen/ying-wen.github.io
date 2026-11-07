"""Relative Value Iteration
已知不可约非周期两状态模型，奖励0.2/0.8，P(自环)=0.8，其余0.2。初值0，每扫描2个模型备份；测量h1-h0相对于1.5的误差，未归一化对照保留公共线性增长项。
Tv=r+Pv；hnew=Tv−Tv(0)，相对值不受常数坐标变化影响。
时序、边界与未覆盖范围见核心注释及 docs/extended-classic.md。
"""
import math
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "relative_value_iteration",
    "name": "Relative Value Iteration",
    "family": "extended_classic",
    "coverage_refs": [
        "core-rvi"
    ],
    "implementation_kind": "teaching_method",
    "task": "extended_average_known_model",
    "baseline": "unnormalized_value_iteration",
    "metric": "相对偏差差绝对误差",
    "unit": "value",
    "higher_better": False,
    "budget": "model_sweeps",
    "scope": "exact-planning",
    "chapter_paths": [
        "algorithms/average-reward/"
    ],
    "sources": [
        "https://arxiv.org/abs/2006.16318"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "已知不可约非周期两状态模型，奖励0.2/0.8，P(自环)=0.8，其余0.2。初值0，每扫描2个模型备份；测量h1-h0相对于1.5的误差，未归一化对照保留公共线性增长项。",
    "question": "参考状态归一化怎样去除平均奖励公共增长项？",
    "limitations": "有限教学任务，不构成一般性能或原论文实验复现。"
}

def update(values):
    backed=[r+.8*values[s]+.2*values[1-s] for s,r in enumerate([.2,.8])]
    reference=backed[0]
    return [v-reference for v in backed],reference

def run(seed=0,steps=1200,emit=None):
    check(steps)
    rows,v=[],[0.,0.]
    for t in range(1,steps+1):
        v,reference=update(v)
        log(rows,t,abs(v[1]-v[0]-1.5),steps,emit,reference_backup=reference,model_backups=2*t,common_offset=v[0])
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
