"""三步 SARSA：独立、可运行的教学实现。
六格链在线控制：从0开始，右端终止奖励1，其余奖励-0.01，γ=0.95，ε=0.1。训练与测量分开，测量是冻结当前Q的贪心策略回报。
"""
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "nstep_sarsa",
    "name": "三步 SARSA",
    "family": "classic",
    "chapter_paths": [
        "algorithms/control/",
        "algorithms/value/",
        "foundations/objectives/"
    ],
    "description": "六格链在线控制：从0开始，右端终止奖励1，其余奖励-0.01，γ=0.95，ε=0.1。训练与测量分开，测量是冻结当前Q的贪心策略回报。",
    "question": "在同一教学任务和预算下，实际更新怎样改变预测或策略？",
    "task": "chain_control",
    "baseline": "q_learning",
    "metric": "确定性贪心策略折扣回报",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "sources": [
        "http://incompleteideas.net/book/the-book-2nd.html"
    ],
    "defaults": {
        "steps": 1000
    },
    "budget": "environment_steps",
    "budget_note": "真实转移次数；未完成回合不当作终止；模型规划计算额外单列。",
    "initialization": "Q=0，回合从状态0开始；在线训练，冻结当前贪心策略评价。",
    "limitations": "小型教学任务；曲线不构成一般性能或持续学习优势的证据。"
}

def update(q,s,a,rewards,sp,ap,alpha=.2):
    target = sum(.95**k*r for k,r in enumerate(rewards))
    if sp is not None:
        target += .95**len(rewards)*q[sp][ap]
    q[s][a] += alpha*(target-q[s][a])
    return target

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rng, rows = random.Random(seed), []
    q, s = {j:[0.,0.] for j in range(5)}, 0
    pending = []
    a = action(q[s],rng)
    for t in range(1,steps+1):
        reward, sp = chain(s,a)
        ap = action(q[sp],rng) if sp is not None else 0
        pending.append((s,a,reward))
        if len(pending) >= 3:
            js,ja,_ = pending[0]
            update(q,js,ja,[r for _,_,r in pending[:3]],sp,ap)
            pending.pop(0)
        if sp is None:
            while pending:
                js,ja,_ = pending[0]
                update(q,js,ja,[r for _,_,r in pending],None,0)
                pending.pop(0)
            ap = action(q[0],rng)
        a = ap
        s = 0 if sp is None else sp
        log(rows,t,control_score(q),steps,emit)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
