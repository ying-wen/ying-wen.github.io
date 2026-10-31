"""Intra-option片段学习：U(s',o)=(1-β)Q(s',o)+βmax合法新option Q(s',o')。
观测同一primitive action后，用ρ=π_o(a|s)/b(a|s)更新所有有支持的option。
continuation不受initiation set限制；只有重新选择option受限制。终点U=0。
"""
import random
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from ._common import walk, greedy, sample, record, check, main

POLICIES=[[.8,.2],[.2,.8]]

def arrival(q,beta,terminal=False,mask=None):
    if terminal:
        return [0.]*len(q)
    mask=[True]*len(q) if mask is None else mask
    legal=[v for v,m in zip(q,mask) if m]
    if not legal:
        raise ValueError("非终止新option选择须有合法项")
    v=max(legal)
    return [(1-b)*x+b*v for x,b in zip(q,beta)]

def update(q,s,a,sn,reward,behavior_probability,terminal=False,active=None,alpha=.1):
    if behavior_probability<=0:
        raise ValueError("动作须有行为支持")
    here=q[s][:]; nxt=q[sn][:]
    u=arrival(nxt,[.25,.25],terminal)
    for o in range(2):
        if active is not None and o!=active:
            continue
        rho=1. if active is not None else POLICIES[o][a]/behavior_probability
        q[s][o]=here[o]+alpha*rho*(reward+.9*u[o]-here[o])

def evaluate(q):
    values=[[0.,0.] for _ in range(6)]
    selected=[greedy(v) for v in q]
    for _ in range(180):
        new=[[0.,0.] for _ in range(6)]
        for s in range(5):
            for o in range(2):
                for a,p in enumerate(POLICIES[o]):
                    sn=walk(s,a,6); terminal=sn==5
                    r=1. if terminal else -.02
                    u=0. if terminal else .75*values[sn][o]+.25*values[sn][selected[sn]]
                    new[s][o]+=p*(r+.9*u)
        values=new
    return values[0][selected[0]]

def run(seed=0,steps=1200,emit=None,all_options=True):
    check(steps)
    rng=random.Random(seed); rows=[]; q=[[0.,0.] for _ in range(6)]
    s=0; o=greedy(q[s],rng,.2); terminations=0
    for t in range(1,steps+1):
        a=sample(POLICIES[o],rng); sn=walk(s,a,6); terminal=sn==5
        r=1. if terminal else -.02
        update(q,s,a,sn,r,POLICIES[o][a],terminal,active=None if all_options else o)
        ended=terminal or rng.random()<.25
        s=0 if terminal else sn
        if ended:
            o=greedy(q[s],rng,.2); terminations+=1
        if t==1 or t%20==0 or t==steps:
            record(rows,t,steps,evaluate(q),emit,option_terminations=terminations,option_updates=t*(2 if all_options else 1))
    return rows

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": True,
    "id": "intra_option_q",
    "name": "Intra-option Q：片段更新",
    "task": "ek_intra_option_chain",
    "baseline": "option_on_policy_only",
    "metric": "冻结call-and-return起点折扣回报",
    "unit": "return",
    "budget": "environment_steps",
    "coverage_refs": [
        "core-intra-option"
    ],
    "chapter_paths": [
        "construction/options/"
    ],
    "sources": [
        "https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/sutton%2Bal-1999.pdf"
    ],
    "description": "6格链起点0终点5，奖励1否则-.02；两固定随机option右动作概率.2/.8，β=.25，α=.1 γ=.9；行为执行active option，终止后ε=.2重新选择。每转移对全部option重要性加权备份，冻结精确call-and-return评价。",
    "limitations": "固定option的片段价值学习组件；不学习内部策略/终止；环境终止不bootstrap，β=.25不是environment done；重要性比需正行为支持。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？"
}

if __name__=="__main__":
    main(META,run)
