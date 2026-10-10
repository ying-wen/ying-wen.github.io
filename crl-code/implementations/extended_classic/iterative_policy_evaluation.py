"""迭代策略评价
已知六格链，从状态0开始，π=(0.2,0.8)，γ=0.95；V初始0。每次预算是5个状态备份的一轮扫描；误差按精确线性方程解测量。
Vnew(s)=Σaπ(a|s)[r+γVold(s')]；整轮读取同一旧V。
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
    "id": "iterative_policy_evaluation",
    "name": "迭代策略评价",
    "family": "extended_classic",
    "coverage_refs": [
        "core-policy-evaluation"
    ],
    "implementation_kind": "teaching_method",
    "task": "extended_fixed_chain_planning",
    "baseline": "policy_evaluation_inplace",
    "metric": "固定策略最大价值误差",
    "unit": "value",
    "higher_better": False,
    "budget": "model_sweeps",
    "scope": "exact-planning",
    "chapter_paths": [
        "foundations/tabular/dynamic-programming/",
        "algorithms/value/"
    ],
    "sources": [
        "http://incompleteideas.net/book/the-book-2nd.html"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "已知六格链，从状态0开始，π=(0.2,0.8)，γ=0.95；V初始0。每次预算是5个状态备份的一轮扫描；误差按精确线性方程解测量。",
    "question": "怎样用同步Bellman期望备份评估固定策略？",
    "limitations": "有限教学任务，不构成一般性能或原论文实验复现。"
}

def update(v):
    result=[0.]*5
    for s in range(5):
        target=0.
        for a,p in enumerate([.2,.8]):
            r,sp=chain(s,a)
            target+=p*(r+(.95*v[sp] if sp is not None else 0.))
        result[s]=target
    return result

def run(seed=0,steps=1200,emit=None):
    check(steps)
    v,rows,truth=[0.]*5,[],fixed_chain_values()
    for t in range(1,steps+1):
        v=update(v)
        log(rows,t,max(abs(a-b) for a,b in zip(v,truth)),steps,emit,model_backups=5*t)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
