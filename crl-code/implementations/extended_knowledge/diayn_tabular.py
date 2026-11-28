"""DIAYN核心：uniform z在episode固定；discriminator只见状态而不见动作。
ri=log qφ(z|s')-log p(z)，技能策略同时最大化ri与动作熵。
有限表格soft actor-critic实例，Q状态含时间避免把固定8步截断误当无限时域。
"""
import math
import random
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from ._common import walk, softmax, sample, record, check, main

def discriminator_gradient(logits,z):
    p=softmax(logits)
    return [x-float(j==z) for j,x in enumerate(p)]

def intrinsic_reward(discriminator_logits,z,prior=.5):
    if not 0<prior<=1:
        raise ValueError("skill prior须在(0,1]")
    return math.log(max(softmax(discriminator_logits)[z],1e-12))-math.log(prior)

def soft_actor_gradient(logits,q,temperature=.1):
    # J=Σπ(a)(Q(a)-αlogπ(a))，精确微分而非伪REINFORCE常数。
    p=softmax(logits)
    utility=[v-temperature*math.log(max(x,1e-12)) for v,x in zip(q,p)]
    mean=sum(x*u for x,u in zip(p,utility))
    return [x*(u-mean) for x,u in zip(p,utility)]

def frozen_mi(logits):
    # 精确传播各skill各时刻状态分布；不读取discriminator。
    joint=[[0.,0.] for _ in range(5)]
    for z in range(2):
        occupancy=[0.,0.,1.,0.,0.]
        for h in range(8):
            nxt=[0.]*5
            for s,prob in enumerate(occupancy):
                for a,p in enumerate(softmax(logits[h][s][z])):
                    nxt[walk(s,a,5)]+=prob*p
            for s,p in enumerate(nxt):
                joint[s][z]+=.5*p/8
            occupancy=nxt
    total=0.
    for row in joint:
        marginal=sum(row)
        for mass in row:
            if mass>0:
                total+=mass*math.log(mass/(marginal*.5))
    return total

def run(seed=0,steps=1200,emit=None,intrinsic=True):
    check(steps)
    rng=random.Random(seed); rows=[]
    logits=[[[[rng.uniform(-.1,.1) for a in range(2)] for z in range(2)] for s in range(5)] for h in range(8)]
    q=[[[[0.,0.] for a in range(2)] for s in range(5)] for h in range(8)]
    # q[h][s][z][a] has the same dimensions as actor.
    disc=[[rng.uniform(-.01,.01) for z in range(2)] for s in range(5)]
    s=2; h=0; z=rng.randrange(2); episodes=0; reward_sum=0.
    for t in range(1,steps+1):
        p=softmax(logits[h][s][z]); a=sample(p,rng); sn=walk(s,a,5)
        reward=intrinsic_reward(disc[sn],z) if intrinsic else 0.
        dg=discriminator_gradient(disc[sn],z)
        disc[sn]=[v-.15*g for v,g in zip(disc[sn],dg)]
        terminal=h==7
        continuation=0.
        if not terminal:
            np=softmax(logits[h+1][sn][z])
            continuation=sum(x*(v-.1*math.log(max(x,1e-12))) for x,v in zip(np,q[h+1][sn][z]))
        target=reward+.95*continuation
        q[h][s][z][a]+=.2*(target-q[h][s][z][a])
        gradient=soft_actor_gradient(logits[h][s][z],q[h][s][z])
        logits[h][s][z]=[v+.05*g for v,g in zip(logits[h][s][z],gradient)]
        reward_sum+=reward
        if terminal:
            s=2; h=0; z=rng.randrange(2); episodes+=1
        else:
            s=sn; h+=1
        if t==1 or t%20==0 or t==steps:
            record(rows,t,steps,frozen_mi(logits),emit,mean_intrinsic_reward=reward_sum/t,completed_episodes=episodes)
    return rows

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": True,
    "id": "diayn_tabular",
    "name": "DIAYN：表格最大熵技能发现",
    "task": "ek_diayn_skills",
    "baseline": "diayn_no_discriminator_reward",
    "metric": "冻结技能与状态互信息",
    "unit": "nats",
    "budget": "environment_steps",
    "coverage_refs": [
        "core-diayn"
    ],
    "chapter_paths": [
        "construction/options/"
    ],
    "sources": [
        "https://arxiv.org/abs/1802.06070"
    ],
    "description": "5格链两skill等先验，长度8 skill固定episode；状态包含remaining-time。表格soft Q α=.2 γ=.95，最大熵actor α=.05 温度.1，discriminator CE α=.15；内奖logq(z|s')-log.5。冻结精确全部8时刻占用，计算真实joint MI。",
    "limitations": "联合discriminator/skill policy最大熵训练组件，非神经SAC DIAYN或下游迁移实验；观测含position/time，技能可能塌缩；MI不用训练分类器准确率代替。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？"
}

if __name__=="__main__":
    main(META,run)
