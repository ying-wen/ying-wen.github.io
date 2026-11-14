"""线性半梯度 SARSA
平稳六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1。初始估值0，每预算单位是真实环境转移；冻结贪心策略用终点/循环精确原奖励回报评价。每动作三个共享多项式特征[1,s/5,(s/5)^2]，共6权重；与表格MC容量不同，不能推论同表示优势。
δ=r+γw·x'−w·x，w←w+αδx；终点x'=0。
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
    "id": "semi_gradient_sarsa",
    "name": "线性半梯度 SARSA",
    "family": "extended_classic",
    "coverage_refs": [
        "core-semi-gradient-sarsa"
    ],
    "implementation_kind": "teaching_method",
    "task": "extended_chain_control",
    "baseline": "mc_control",
    "metric": "贪心策略精确折扣回报",
    "unit": "return",
    "higher_better": True,
    "budget": "environment_steps",
    "scope": "teaching-control",
    "chapter_paths": [
        "foundations/approximation/features-control/",
        "algorithms/control/"
    ],
    "sources": [
        "http://incompleteideas.net/book/the-book-2nd.html"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "平稳六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1。初始估值0，每预算单位是真实环境转移；冻结贪心策略用终点/循环精确原奖励回报评价。每动作三个共享多项式特征[1,s/5,(s/5)^2]，共6权重；与表格MC容量不同，不能推论同表示优势。",
    "question": "bootstrap目标停止求导时，共享特征控制怎样更新？",
    "limitations": "有限教学任务，不构成一般性能或原论文实验复现。"
}

def action_features(s,a):
    local=[1.,s/5.,(s/5.)**2]
    return [v if b==a else 0. for b in range(2) for v in local]

def values(w):
    return {s:[dot(w,action_features(s,a)) for a in range(2)] for s in range(5)}

def update(w,x,reward,xp,alpha=.05):
    # bootstrap读取旧w，但梯度只作用当前预测，故称半梯度。
    delta=reward+.95*dot(w,xp)-dot(w,x)
    return [wi+alpha*delta*xi for wi,xi in zip(w,x)],delta

def run(seed=0,steps=1200,emit=None):
    check(steps)
    rng,rows,w,s=random.Random(seed),[],[0.]*6,0
    a=sample(probs(values(w)[s]),rng)
    for t in range(1,steps+1):
        r,sp=chain(s,a)
        ap=sample(probs(values(w)[sp]),rng) if sp is not None else 0
        xp=action_features(sp,ap) if sp is not None else [0.]*6
        w,delta=update(w,action_features(s,a),r,xp)
        s=0 if sp is None else sp
        a=sample(probs(values(w)[s]),rng) if sp is None else ap
        log(rows,t,score(values(w)),steps,emit,td_error=delta)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
