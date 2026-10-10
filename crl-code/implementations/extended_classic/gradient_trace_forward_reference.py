"""冻结参数前向 GTD2 对照
每episode固定两转移0→1→真终点，末奖励Bernoulli(0.5)，γ=0.9，λ=0.8。V=tanh(w·x)，H=tanh(h·x)，one-hot特征，w=h=0；两网络段内冻结、段间更新，测量真值[0.45,0.5]。只覆盖GTD2冻结参数资格迹方向，不覆盖TDC/TDRC和深度控制。
ε=Gλ−V；Δw=−ΣH∇ε，Δη=Σ(ε−H)∇H；尾bootstrap及其导数均保留。
时序、边界与未覆盖范围见核心注释及 docs/extended-classic.md。
"""
import math
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "gradient_trace_forward_reference",
    "name": "冻结参数前向 GTD2 对照",
    "family": "extended_classic",
    "coverage_refs": [],
    "implementation_kind": "component_experiment",
    "task": "extended_frozen_gtd2_episodes",
    "baseline": "gradient_eligibility_traces",
    "metric": "两状态非线性价值 RMSE",
    "unit": "value",
    "higher_better": False,
    "budget": "episodes",
    "scope": "component-frozen-nonlinear-gradient-traces",
    "chapter_paths": [
        "algorithms/credit-assignment/"
    ],
    "sources": [
        "https://arxiv.org/abs/2507.09087"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "每episode固定两转移0→1→真终点，末奖励Bernoulli(0.5)，γ=0.9，λ=0.8。V=tanh(w·x)，H=tanh(h·x)，one-hot特征，w=h=0；两网络段内冻结、段间更新，测量真值[0.45,0.5]。只覆盖GTD2冻结参数资格迹方向，不覆盖TDC/TDRC和深度控制。",
    "question": "完整求导的冻结λ目标怎样验证后向GTD2总方向？",
    "limitations": "仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。"
}

def prediction(parameter,x):
    value=math.tanh(dot(parameter,x))
    return value,[(1-value*value)*xi for xi in x]

def directions(w,h,features,rewards,gamma=.9,lam=.8):
    # 两网络参数在一条episode内冻结；段间更新不能继承在线前缀等价。
    values,grads=zip(*(prediction(w,x) for x in features))
    hs,hgrads=zip(*(prediction(h,x) for x in features[:-1]))
    dw,dh=[0.]*2,[0.]*2
    target,target_gradient=values[-1],list(grads[-1])
    targets,target_grads=[0.]*len(rewards),[[0.,0.] for _ in rewards]
    for t in reversed(range(len(rewards))):
        target=rewards[t]+gamma*((1-lam)*values[t+1]+lam*target)
        target_gradient=[gamma*((1-lam)*g+lam*old) for g,old in zip(grads[t+1],target_gradient)]
        targets[t],target_grads[t]=target,target_gradient[:]
    for t in range(len(rewards)):
        error=targets[t]-values[t]
        error_grad=[g-old for g,old in zip(target_grads[t],grads[t])]
        dw=[v-hs[t]*g for v,g in zip(dw,error_grad)]
        dh=[v+(error-hs[t])*g for v,g in zip(dh,hgrads[t])]
    return dw,dh

def update(w,h,features,rewards,alpha=.03,beta=.1):
    dw,dh=directions(w,h,features,rewards)
    # 两个方向从相同旧参数求出后才同时应用。
    return [v+alpha*g for v,g in zip(w,dw)],[v+beta*g for v,g in zip(h,dh)]

def run(seed=0,steps=1200,emit=None):
    check(steps)
    rng,rows,w,h=random.Random(seed),[],[0.,0.],[0.,0.]
    xs=[[1.,0.],[0.,1.],[0.,0.]]
    for t in range(1,steps+1):
        reward=float(rng.random()<.5)
        w,h=update(w,h,xs,[0.,reward])
        error=math.sqrt(((math.tanh(w[0])-.45)**2+(math.tanh(w[1])-.5)**2)/2)
        log(rows,t,error,steps,emit,auxiliary_norm=math.sqrt(dot(h,h)),environment_steps=2*t)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
