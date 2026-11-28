"""SMDP 选项 Q-learning。
六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1，Q=0。选项0向左一步，选项1向右最多两步；原始动作对照仅一步。横轴严格累计真实原始转移，任务平稳。未完成且被预算截断的选项不更新，真终点可提前结束选项。
"""
import random
import math
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "option_smdp_q",
    "name": "SMDP 选项 Q-learning",
    "family": "continual",
    "chapter_paths": [
        "construction/options/",
        "algorithms/control/",
        "algorithms/credit-assignment/"
    ],
    "description": "六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1，Q=0。选项0向左一步，选项1向右最多两步；原始动作对照仅一步。横轴严格累计真实原始转移，任务平稳。未完成且被预算截断的选项不更新，真终点可提前结束选项。",
    "question": "两步承诺选项怎样改变相同原始预算下的学习？",
    "task": "option_chain_control",
    "baseline": "primitive_q",
    "metric": "原始奖励贪心策略折扣回报",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "sources": [
        "http://incompleteideas.net/book/the-book-2nd.html",
        "https://www.sciencedirect.com/science/article/pii/S0004370299000521"
    ],
    "defaults": {
        "steps": 1000
    },
    "budget": "environment_steps",
    "budget_note": "累计真实原始转移；任务平稳，无中点变化；选项完成才更新，预算截断不冒充终点；评价遵循call-and-return。",
    "limitations": "小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。"
}

def update(q,s,option,rewards,sp,alpha=.2,gamma=.95):
    # 多步奖励从选项启动时刻折扣；bootstrap必须使用γ^duration。
    duration=len(rewards)
    if duration<1:
        raise ValueError("选项持续时间必须为正")
    target=sum(gamma**i*r for i,r in enumerate(rewards))
    if sp is not None:
        target+=gamma**duration*max(q[sp])
    q[s][option]+=alpha*(target-q[s][option])
    return target

def evaluate(q,gamma=.95):
    # call-and-return：选项未终止时，忽略中间状态的高层贪心选择。
    s,option,remaining=0,0,0
    rewards,visited=[],{}
    while s is not None:
        if remaining==0:
            option=max(range(2),key=lambda o:q[s][o])
            remaining=2 if option==1 else 1
        augmented=(s,option,remaining)
        if augmented in visited:
            start=visited[augmented]
            prefix=sum(gamma**k*r for k,r in enumerate(rewards[:start]))
            cycle=sum(gamma**k*r for k,r in enumerate(rewards[start:]))
            return prefix+gamma**start*cycle/(1-gamma**(len(rewards)-start))
        visited[augmented]=len(rewards)
        reward,s=chain(s,option)
        rewards.append(reward)
        remaining-=1
    return sum(gamma**k*r for k,r in enumerate(rewards))

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rng,rows,q,s,t=random.Random(seed),[],{j:[0.,0.] for j in range(5)},0,0
    while t<steps:
        start=s
        option=eps_action(q[s],rng)
        intended_duration=2 if option==1 else 1
        rewards=[]
        for elapsed in range(intended_duration):
            reward,sp=chain(s,option)
            rewards.append(reward)
            t+=1
            complete=sp is None or elapsed+1==intended_duration
            if complete:
                update(q,start,option,rewards,sp)
            # 选项执行中不重新选择动作，也不把中间状态当作SMDP边界。
            s=0 if sp is None else sp
            log(rows,t,evaluate(q),steps,emit,phase="stationary",
                option_duration=len(rewards),completed_option=int(complete))
            if sp is None or t==steps:
                break
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
