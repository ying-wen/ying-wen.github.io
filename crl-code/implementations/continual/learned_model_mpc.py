"""学习模型滚动规划。
未知六格链从0开始：非终点期望奖励-0.01，终点奖励中点从1变0.5，观测Gaussian噪声σ=0.02。经验均值奖励与频率转移模型初始空，未访问乐观奖励0.05；γ=0.95，ε=0.1。滚动有限时域5步对照1步，两者每转移只执行首动作然后重新规划，额外模型计算不同。
"""
import random
import math
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "learned_model_mpc",
    "name": "学习模型滚动规划",
    "family": "continual",
    "chapter_paths": [
        "construction/models/",
        "construction/planning/",
        "algorithms/dyna/"
    ],
    "description": "未知六格链从0开始：非终点期望奖励-0.01，终点奖励中点从1变0.5，观测Gaussian噪声σ=0.02。经验均值奖励与频率转移模型初始空，未访问乐观奖励0.05；γ=0.95，ε=0.1。滚动有限时域5步对照1步，两者每转移只执行首动作然后重新规划，额外模型计算不同。",
    "question": "五步与一步模型规划怎样响应终点奖励下降？",
    "task": "learned_chain_model_control",
    "baseline": "one_step_model",
    "metric": "原始平均奖励冻结策略折扣回报",
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
    "budget_note": "每步一个真实转移；中点终点奖励均值1变0.5；模型是全历史经验均值；规划计算成本不同。",
    "limitations": "小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。"
}

def update(model,s,a,reward,sp):
    # 只使用真实观测：经验平均奖励和经验转移频率。
    n,total,successors=model.get((s,a),(0,0.,{}))
    successors=successors.copy()
    successors[sp]=successors.get(sp,0)+1
    model[(s,a)]=(n+1,total+reward,successors)

def action_values(model,s,horizon,gamma=.95):
    if horizon<1:
        return [0.,0.]
    values=[]
    for a in range(2):
        if (s,a) not in model:
            # 两个对照共用未访问动作的乐观先验；不能读取真实模型。
            values.append(.05)
            continue
        n,total,successors=model[(s,a)]
        continuation=0.
        if horizon>1:
            for sp,count in successors.items():
                if sp is not None:
                    continuation+=count/n*max(action_values(model,sp,horizon-1,gamma))
        values.append(total/n+gamma*continuation)
    return values

def model_error(model,goal_reward):
    # 真模型仅用于测量奖励预测，不参与计划或学习。
    error=0.
    for s in range(5):
        for a in range(2):
            r,sp=chain(s,a)
            truth=goal_reward if sp is None else r
            n,total,_=model.get((s,a),(1,0.,{}))
            error+=(total/n-truth)**2
    return math.sqrt(error/10)

def evaluate(model,horizon,goal_reward):
    # 冻结学习模型策略，用真实均值奖励精确测量终点或循环。
    def policy(s):
        scores=action_values(model,s,horizon)
        return max(range(2),key=lambda a:scores[a])
    return discounted_chain_score(policy,goal_reward)

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rng,rows,model,s=random.Random(seed),[],{},0
    horizon=5
    for t in range(1,steps+1):
        goal_reward=1. if t<=steps//2 else .5
        scores=action_values(model,s,horizon)
        a=eps_action(scores,rng)
        mean,sp=chain(s,a)
        if sp is None:
            mean=goal_reward
        reward=mean+rng.gauss(0,.02)
        update(model,s,a,reward,sp)
        s=0 if sp is None else sp
        log(rows,t,evaluate(model,horizon,goal_reward),steps,emit,
            model_reward_rmse=model_error(model,goal_reward),planning_horizon=horizon)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
