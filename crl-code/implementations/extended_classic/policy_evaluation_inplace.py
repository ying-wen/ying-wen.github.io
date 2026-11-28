"""原地策略评价对照
已知六格链，从状态0开始，π=(0.2,0.8)，γ=0.95；V初始0。每次预算是5个状态备份的一轮扫描；误差按精确线性方程解测量。
V(s)←Σaπ(a|s)[r+γV(s')]；同轮已更新的状态可被后续备份读取。
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
    "id": "policy_evaluation_inplace",
    "name": "原地策略评价对照",
    "family": "extended_classic",
    "coverage_refs": [],
    "implementation_kind": "teaching_method",
    "task": "extended_fixed_chain_planning",
    "baseline": "iterative_policy_evaluation",
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
    "question": "原地扫描与同步扫描怎样改变固定策略评价的误差过程？",
    "limitations": "有限教学任务，不构成一般性能或原论文实验复现。"
}

def update(v):
    result=v[:]
    for s in range(5):
        target=0.
        for a,p in enumerate([.2,.8]):
            r,sp=chain(s,a)
            target+=p*(r+(.95*result[sp] if sp is not None else 0.))
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
