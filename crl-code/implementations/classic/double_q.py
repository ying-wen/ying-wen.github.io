"""Double Q-learning：独立、可运行的教学实现。
六格链在线控制：从0开始，右端终止奖励1，其余奖励-0.01，γ=0.95，ε=0.1。训练与测量分开，测量是冻结当前Q的贪心策略回报。
"""
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "double_q",
    "name": "Double Q-learning",
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

def update(first,second,s,a,reward,sp,alpha=.2):
    # 用被更新表选择动作，用另一个表评价，避免同一噪声的最大化偏差。
    best = max(range(2),key=lambda b:first[sp][b]) if sp is not None else 0
    target = reward+(.95*second[sp][best] if sp is not None else 0.)
    first[s][a] += alpha*(target-first[s][a])
    return target

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rng, rows = random.Random(seed), []
    q, s = {j:[0.,0.] for j in range(5)}, 0
    q2 = {s:[0.,0.] for s in range(5)}
    for t in range(1,steps+1):
        a = action([q[s][b]+q2[s][b] for b in range(2)],rng)
        reward, sp = chain(s,a)
        if rng.random() < .5:
            update(q,q2,s,a,reward,sp)
        else:
            update(q2,q,s,a,reward,sp)
        measured = {j:[q[j][b]+q2[j][b] for b in range(2)] for j in q}
        s = 0 if sp is None else sp
        log(rows,t,control_score(measured),steps,emit)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
