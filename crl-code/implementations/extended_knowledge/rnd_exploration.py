"""RND内奖：b(s')=||fθ(s')-f随机(s')||²，目标网络永久冻结。
步骤：真实转移→更新前计算新奇奖励→Q备份→只训练预测网络→统计真实访问。
原文完整实现用PPO等；这里仅CPU小链的神经新奇奖励组件。
"""
import random
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from ._common import MLP, walk, greedy, record, check, main

def observation(s,n=12):
    return [float(j==s) for j in range(n)]

def predictor_step(predictor,target,x,alpha=.03):
    prediction=predictor.forward(x)
    fixed=target.forward(x)
    errors=[a-b for a,b in zip(prediction,fixed)]
    loss=sum(e*e for e in errors)/len(errors)
    predictor.step(x,[2*e/len(errors) for e in errors],alpha)
    return loss

def run(seed=0,steps=1200,emit=None,use_bonus=True):
    check(steps)
    rng=random.Random(seed); rows=[]
    target=MLP(12,12,3,rng); predictor=MLP(12,12,3,rng)
    q=[[0.,0.] for _ in range(12)]; visited={0}; s=0; bonuses=0.
    for t in range(1,steps+1):
        a=greedy(q[s],rng,.15)
        sn=walk(s,a,12); visited.add(sn)
        # 奖励必须来自更新前预测误差；否则刚好拟合可把新奇奖励抹去。
        bonus=predictor_step(predictor,target,observation(sn))
        reward=bonus if use_bonus else 0.
        q[s][a]+=.2*(reward+.99*max(q[sn])-q[s][a])
        bonuses+=bonus; s=sn
        record(rows,t,steps,len(visited)/12,emit,novelty_mse=bonus,mean_novelty=bonuses/t,predictor_updates=t)
    return rows

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": True,
    "id": "rnd_exploration",
    "name": "RND：固定随机MLP探索",
    "task": "ek_rnd_chain",
    "baseline": "rnd_no_bonus",
    "metric": "在线已访问状态比例",
    "unit": "fraction",
    "budget": "environment_steps",
    "coverage_refs": [
        "core-rnd"
    ],
    "chapter_paths": [
        "algorithms/exploration/"
    ],
    "sources": [
        "https://arxiv.org/abs/1810.12894",
        "https://github.com/openai/random-network-distillation"
    ],
    "description": "12格反射链起点0；固定随机MLP12→12→3与独立预测MLP，MSE SGD .03；Q α=.2 γ=.99 ε=.15，内奖为更新前误差，预测器每交互更新；无外奖，访问覆盖为指标。",
    "limitations": "实测RND神经预测/内奖驱动探索组件；无PPO、Atari、双critic或reward/observation归一化，非完整原工程。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？"
}

if __name__=="__main__":
    main(META,run)
