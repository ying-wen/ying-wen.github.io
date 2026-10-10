"""三步 Q(σ)
六格链固定随机π=μ=(0.2,0.8)，γ=0.95，Q初始0，从0开始。保存最长三步前缀，成熟后更新最早Q，终点依次冲洗短目标；预算未成熟尾不伪终止。固定策略评价，不是控制训练；Retrace本实验仅on-policy。
G=r+γ[(σ+(1−σ)π(a'|s'))(Gnext−Q')+σQ'+(1−σ)EπQ]。
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
    "id": "q_sigma",
    "name": "三步 Q(σ)",
    "family": "extended_classic",
    "coverage_refs": [
        "core-q-sigma"
    ],
    "implementation_kind": "teaching_method",
    "task": "extended_fixed_q_prediction",
    "baseline": "tree_backup",
    "metric": "固定策略十状态动作价值 RMSE",
    "unit": "value",
    "higher_better": False,
    "budget": "environment_steps",
    "scope": "teaching-prediction",
    "chapter_paths": [
        "foundations/tabular/multistep/",
        "algorithms/credit-assignment/"
    ],
    "sources": [
        "https://arxiv.org/abs/1703.01327"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "六格链固定随机π=μ=(0.2,0.8)，γ=0.95，Q初始0，从0开始。保存最长三步前缀，成熟后更新最早Q，终点依次冲洗短目标；预算未成熟尾不伪终止。固定策略评价，不是控制训练；Retrace本实验仅on-policy。",
    "question": "采样与期望之间的σ混合怎样形成多步预测目标？",
    "limitations": "有限教学任务，不构成一般性能或原论文实验复现。"
}

def target(q,trajectory,endpoint,endpoint_action,sigma=.5):
    if not 0<=sigma<=1:
        raise ValueError("sigma须在[0,1]")
    # 冻结Q的on-policy n-step递推；σ=0为Tree Backup，σ=1为SARSA。
    # 本实验μ=π，不包含一般off-policy Q(σ)的重要性校正。
    g=q[endpoint][endpoint_action] if endpoint is not None else 0.
    for k in reversed(range(len(trajectory))):
        _,_,r,sp=trajectory[k]
        if sp is None:
            g=r
        else:
            ap=trajectory[k+1][1] if k+1<len(trajectory) else endpoint_action
            expected=dot([.2,.8],q[sp])
            branch=sigma+(1-sigma)*[.2,.8][ap]
            g=r+.95*(branch*(g-q[sp][ap])+sigma*q[sp][ap]+(1-sigma)*expected)
    return g

def run(seed=0,steps=1200,emit=None):
    check(steps)
    rng,rows,q,s=random.Random(seed),[],{j:[0.,0.] for j in range(5)},0
    a=sample([.2,.8],rng)
    pending=[]
    for t in range(1,steps+1):
        r,sp=chain(s,a)
        ap=sample([.2,.8],rng) if sp is not None else 0
        pending.append((s,a,r,sp))
        if len(pending)>=3:
            js,ja,_,_=pending[0]
            g=target(q,pending[:3],sp,ap)
            q[js][ja]+=.1*(g-q[js][ja])
            pending.pop(0)
        if sp is None:
            while pending:
                js,ja,_,_=pending[0]
                g=target(q,pending,None,0)
                q[js][ja]+=.1*(g-q[js][ja])
                pending.pop(0)
        s=0 if sp is None else sp
        a=sample([.2,.8],rng) if sp is None else ap
        log(rows,t,q_error(q),steps,emit)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
