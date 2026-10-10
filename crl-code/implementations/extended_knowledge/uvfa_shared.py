"""UVFA：Q(s,g,a;θ)共同编码状态与目标，而不是每个目标复制一个网络。
设置：七格链，目标到达奖励0，否则-1；成功是真任务终止，12步预算是截断。
公式：δ=r+γ(1-done)max_a Q(s',g,a)-Q(s,g,a)，半梯度平方误差SGD。
步骤：抽取目标→在线探索→计算冻结目标→更新共享参数→独立遍历评价。
原文：Schaul et al. ICML2015；此处只实测目标条件函数逼近组件。
"""
import random
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from ._common import MLP, walk, greedy, record, check, main

def features(s,g):
    return [s/3-1,g/3-1,(s-g)/6,abs(s-g)/6]

def update(net,s,a,sn,g,physical_terminal=False,alpha=.03):
    done = physical_terminal or sn == g
    reward = 0. if sn == g else -1.
    old = net.forward(features(s,g))
    target = reward + (0. if done else .95*max(net.forward(features(sn,g))))
    gradient = [0.,0.]
    gradient[a] = old[a]-target
    net.step(features(s,g), gradient, alpha)
    return target

def success(getq):
    solved=0
    for start in range(7):
        for g in range(7):
            if start == g:
                continue
            s=start
            for _ in range(12):
                s=walk(s,greedy(getq(s,g)))
                if s == g:
                    solved+=1
                    break
    return solved/42

def run(seed=0,steps=1200,emit=None,tabular=False):
    check(steps)
    rng=random.Random(seed)
    net=MLP(4,16,2,rng)
    q={(s,g):[0.,0.] for s in range(7) for g in range(7)}
    rows=[]
    def getq(s,g):
        return q[s,g] if tabular else net.forward(features(s,g))
    s=rng.randrange(7)
    g=rng.choice([x for x in range(7) if x!=s])
    age=0
    for t in range(1,steps+1):
        a=greedy(getq(s,g),rng,.25)
        sn=walk(s,a)
        done=sn==g
        if tabular:
            target=(0. if done else -1.+.95*max(q[sn,g]))
            q[s,g][a]+=.2*(target-q[s,g][a])
        else:
            update(net,s,a,sn,g)
        s=sn
        age+=1
        if done or age==12:
            s=rng.randrange(7)
            g=rng.choice([x for x in range(7) if x!=s])
            age=0
        if t==1 or t%20==0 or t==steps:
            record(rows,t,steps,success(getq),emit,parameter_count=114 if not tabular else 98)
    return rows

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": True,
    "id": "uvfa_shared",
    "name": "UVFA：共享目标条件MLP",
    "task": "ek_goal_navigation",
    "baseline": "uvfa_tabular",
    "metric": "七目标冻结贪心到达率",
    "unit": "fraction",
    "budget": "environment_steps",
    "coverage_refs": [
        "core-uvfa"
    ],
    "chapter_paths": [
        "construction/goals/"
    ],
    "sources": [
        "https://proceedings.mlr.press/v37/schaul15.html"
    ],
    "description": "7格链，随机起点/目标，负一步代价；输入(s,g,s-g,|s-g|)共享MLP4→16→2，SGD .03，γ=.95，ε=.25；最长12步仅截断。每20步冻结遍历42对起点目标。",
    "limitations": "仅目标条件TD共享逼近；不复现原论文低秩分解预训练或未见目标泛化结果。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？"
}

if __name__=="__main__":
    main(META,run)
