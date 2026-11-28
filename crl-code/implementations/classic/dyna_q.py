"""Dyna-Q：独立、可运行的教学实现。
六格链在线控制：起点0，终点奖励1，其余-0.01，γ=0.95。环境预算与Q-learning一致，额外模型备份单独记录；不能据此声称相同计算成本。
"""
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "dyna_q",
    "name": "Dyna-Q",
    "family": "classic",
    "chapter_paths": [
        "algorithms/dyna/",
        "construction/planning/",
        "construction/models/"
    ],
    "description": "六格链在线控制：起点0，终点奖励1，其余-0.01，γ=0.95。环境预算与Q-learning一致，额外模型备份单独记录；不能据此声称相同计算成本。",
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

def update(q,s,a,reward,sp,alpha=.2):
    target = reward+(.95*max(q[sp]) if sp is not None else 0.)
    error = target-q[s][a]
    q[s][a] += alpha*error
    return error

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rng, rows = random.Random(seed), []
    q, s, model, backups = {j:[0.,0.] for j in range(5)}, 0, {}, 0
    for t in range(1,steps+1):
        a = action(q[s],rng)
        reward,sp = chain(s,a)
        update(q,s,a,reward,sp)
        model[(s,a)] = (reward,sp)
        # 每个真实转移之后进行5次从已访问模型均匀抽样的备份。
        for _ in range(5):
            js,ja = rng.choice(list(model))
            jr,jp = model[(js,ja)]
            update(q,js,ja,jr,jp)
            backups += 1
        s = 0 if sp is None else sp
        log(rows,t,control_score(q),steps,emit,model_backups=backups)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
