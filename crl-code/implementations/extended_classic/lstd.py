"""LSTD
无偏五状态随机游走，从3开始，γ=1，特征[1,s/6]、w=0。冻结当前线性预测测量五状态RMSE；LSTD显式1e-5对角正则，MC只更新完整回合。
A←A+x(x−γx')ᵀ，b←b+xr；求(A+10⁻⁵I)w=b。
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
    "id": "lstd",
    "name": "LSTD",
    "family": "extended_classic",
    "coverage_refs": [
        "core-lstd"
    ],
    "implementation_kind": "teaching_method",
    "task": "extended_linear_walk",
    "baseline": "gradient_mc",
    "metric": "五状态线性价值 RMSE",
    "unit": "value",
    "higher_better": False,
    "budget": "environment_steps",
    "scope": "teaching-prediction",
    "chapter_paths": [
        "foundations/approximation/prediction/",
        "algorithms/value/"
    ],
    "sources": [
        "http://incompleteideas.net/book/the-book-2nd.html",
        "https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/BradtkeBarto-lstd-1996"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "无偏五状态随机游走，从3开始，γ=1，特征[1,s/6]、w=0。冻结当前线性预测测量五状态RMSE；LSTD显式1e-5对角正则，MC只更新完整回合。",
    "question": "经验Bellman正规方程怎样产生线性价值参数？",
    "limitations": "有限教学任务，不构成一般性能或原论文实验复现。"
}

def update(a,b,x,reward,xp):
    # A=sum x(x-γx')^T，b=sum xr；终点xp=0，γ=1。
    for i in range(2):
        b[i]+=x[i]*reward
        for j in range(2):
            a[i][j]+=x[i]*(x[j]-xp[j])

def estimate(a,b,ridge=1e-5):
    # 固定微小对角正则只避免有限前缀奇异；不是原始未正则LSTD。
    regularized=[[v+(ridge if i==j else 0.) for j,v in enumerate(row)] for i,row in enumerate(a)]
    return solve(regularized,b)

def run(seed=0,steps=1200,emit=None):
    check(steps)
    rng,rows,w,s=random.Random(seed),[],[0.,0.],3
    a,b=[[0.,0.],[0.,0.]],[0.,0.]
    for t in range(1,steps+1):
        r,sp=walk(s,rng)
        update(a,b,features(s),r,features(sp))
        w=estimate(a,b)
        s=3 if sp is None else sp
        log(rows,t,linear_error(w),steps,emit)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
