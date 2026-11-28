"""R_o(s)=EΣγ^k r_k，P_oγ(s,j)=E[γ^τ 1(endpoint=j)]。
逐步TD：Rtarget=r+γ(1-β)R(s')；
Ptarget(j)=γ[β1(j=s')+(1-β)P(s',j)]。terminal强制β=1。
"""
import math
import random
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from ._common import walk, sample, record, check, main

def targets(r,next_r,next_p,sn,beta,terminal=False,gamma=.9):
    b=1. if terminal else beta
    return r+gamma*(1-b)*next_r, [
        gamma*(b*float(j==sn)+(1-b)*p) for j,p in enumerate(next_p)]

def exact_model():
    r=[0.]*7; p=[[0.]*7 for _ in range(7)]
    for _ in range(300):
        nr=[0.]*7; np=[[0.]*7 for _ in range(7)]
        for s in range(6):
            for a,prob in enumerate([.2,.8]):
                sn=walk(s,a); terminal=sn==6
                rt,pt=targets(1. if terminal else -.02,r[sn],p[sn],sn,.3,terminal)
                nr[s]+=prob*rt
                np[s]=[x+prob*y for x,y in zip(np[s],pt)]
        r,p=nr,np
    return r,p

def run(seed=0,steps=1200,emit=None,td=True):
    check(steps)
    rng=random.Random(seed); rows=[]; r=[0.]*7; p=[[0.]*7 for _ in range(7)]
    truth_r,truth_p=exact_model()
    s=rng.randrange(6); start=s; rewards=[]; episodes=0
    for t in range(1,steps+1):
        a=sample([.2,.8],rng); sn=walk(s,a); terminal=sn==6
        reward=1. if terminal else -.02
        if td:
            rt,pt=targets(reward,r[sn],p[sn][:],sn,.3,terminal)
            r[s]+=.15*(rt-r[s])
            p[s]=[x+.15*(y-x) for x,y in zip(p[s],pt)]
        rewards.append(reward)
        ended=terminal or rng.random()<.3
        if ended:
            if not td:
                rt=sum(.9**k*x for k,x in enumerate(rewards))
                pt=[.9**len(rewards)*float(j==sn) for j in range(7)]
                r[start]+=.15*(rt-r[start])
                p[start]=[x+.15*(y-x) for x,y in zip(p[start],pt)]
            episodes+=1; rewards=[]; s=rng.randrange(6); start=s
        else:
            s=sn
        error=sum((r[i]-truth_r[i])**2+sum((x-y)**2 for x,y in zip(p[i],truth_p[i])) for i in range(6))
        record(rows,t,steps,math.sqrt(error/48),emit,completed_options=episodes,endpoint_row_mass=sum(p[0]),unfinished_option_length=len(rewards))
    return rows

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": False,
    "id": "option_model",
    "name": "Option：奖励与折扣终点模型",
    "task": "ek_option_model",
    "baseline": "option_model_monte_carlo",
    "metric": "冻结reward/discounted-endpoint联合RMSE",
    "unit": "rmse",
    "budget": "environment_steps",
    "coverage_refs": [
        "core-option-model"
    ],
    "chapter_paths": [
        "construction/models/"
    ],
    "sources": [
        "https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/sutton%2Bal-1999.pdf"
    ],
    "description": "7格链；固定option向右概率.8，β=.3（终点6强制止），γ=.9，r=-.02/终点1；α=.15逐primitive TD学R与Pγ。每次option终止随机起点0..5，与精确固定点RMSE比较。",
    "limitations": "固定option模型组件，R仅真实环境reward、不混入子任务bonus；P行和Eγ^τ不归一；预算末未结束option不伪作终止，MC基线不更新该前缀。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？"
}

if __name__=="__main__":
    main(META,run)
