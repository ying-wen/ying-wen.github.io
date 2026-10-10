"""Option-Critic：同时学Q_U、QΩ、内部π和到达后β。
actor ascent：α(Q_U-baseline)∇logπ；termination descent：αβ(1-β)(QΩ-VΩ)。
所有actor/beta梯度使用更新前critic；到达环境terminal无termination梯度/bootstrap。
"""
import random
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from ._common import walk, greedy, sample, sigmoid, softmax, record, check, main

def termination_gradient(logit,q_continue,v_select):
    b=sigmoid(logit)
    return -b*(1-b)*(q_continue-v_select)

def actor_gradient(logits,a,advantage):
    p=softmax(logits)
    return [advantage*(float(j==a)-x) for j,x in enumerate(p)]

def critic_target(reward,q_continue,v_select,beta,terminal=False,gamma=.9):
    return reward if terminal else reward+gamma*((1-beta)*q_continue+beta*v_select)

def update_termination(logit,q_continue,v_select,terminal=False,alpha=.03):
    # 环境done不会通过option终止梯度来学习；它强制停止。
    return logit if terminal else logit+alpha*termination_gradient(logit,q_continue,v_select)

def evaluate(logits,beta,q):
    v=[[0.,0.] for _ in range(6)]
    # 训练option选择有epsilon；冻结报告用确定性贪心选择，内部策略仍随机。
    selected=[greedy(x) for x in q]
    for _ in range(180):
        new=[[0.,0.] for _ in range(6)]
        for s in range(5):
            for o in range(2):
                for a,p in enumerate(softmax(logits[s][o])):
                    sn=walk(s,a,6); terminal=sn==5
                    b=sigmoid(beta[sn][o])
                    u=0. if terminal else (1-b)*v[sn][o]+b*v[sn][selected[sn]]
                    new[s][o]+=p*((1. if terminal else -.02)+.9*u)
        v=new
    return v[0][selected[0]]

def run(seed=0,steps=1200,emit=None,learn_beta=True):
    check(steps)
    rng=random.Random(seed); rows=[]
    logits=[[[rng.uniform(-.1,.1) for a in range(2)] for o in range(2)] for s in range(6)]
    beta=[[0.,0.] for _ in range(6)]
    q=[[0.,0.] for _ in range(6)]
    qu=[[[0.,0.] for o in range(2)] for s in range(6)]
    s=0; o=rng.randrange(2); ended_count=0
    for t in range(1,steps+1):
        p=softmax(logits[s][o]); a=sample(p,rng); sn=walk(s,a,6); terminal=sn==5
        r=1. if terminal else -.02
        b=sigmoid(beta[sn][o])
        # epsilon-greedy policy over options is part of the actual training target.
        vnext=.8*max(q[sn])+.2*sum(q[sn])/2
        old_action=qu[s][o][a]; old_q=q[s][o]
        old_cont=q[sn][o]; old_select=vnext
        target=critic_target(r,old_cont,vnext,b,terminal)
        grad=actor_gradient(logits[s][o],a,old_action-old_q)
        logits[s][o]=[h+.03*g for h,g in zip(logits[s][o],grad)]
        if learn_beta:
            beta[sn][o]=update_termination(beta[sn][o],old_cont,old_select,terminal)
        qu[s][o][a]+=.15*(target-old_action)
        q[s][o]+=.15*(target-old_q)
        ended=terminal or rng.random()<b
        s=0 if terminal else sn
        if ended:
            o=greedy(q[s],rng,.2); ended_count+=1
        if t==1 or t%20==0 or t==steps:
            record(rows,t,steps,evaluate(logits,beta,q),emit,mean_beta=sum(sigmoid(x) for row in beta[:5] for x in row)/10,option_terminations=ended_count)
    return rows

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": True,
    "id": "option_critic",
    "name": "Option-Critic：表格联合学习",
    "task": "ek_option_critic_chain",
    "baseline": "option_critic_fixed_beta",
    "metric": "冻结call-and-return起点折扣回报",
    "unit": "return",
    "budget": "environment_steps",
    "coverage_refs": [
        "core-option-critic"
    ],
    "chapter_paths": [
        "construction/options/"
    ],
    "sources": [
        "https://arxiv.org/abs/1609.05140"
    ],
    "description": "6格链，r=-.02/终点1，γ=.9；两option softmax内部策略，sigmoid β，每primitive Q_U/QΩ critic α=.15、actor .03、termination .03；ε=.2选择option，termination梯度在arrival state应用，终点跳过。精确冻结策略评价。",
    "limitations": "完整小表格actor/critic/termination训练组件；非原深网Atari方法，选项可能退化，未加deliberation cost/entropy正则。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？"
}

if __name__=="__main__":
    main(META,run)
