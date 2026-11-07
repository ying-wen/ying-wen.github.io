"""TD驱动reward centering (Naik等2024)：δ=r-c+γV(s')-V(s)。
同一旧δ更新V与c；γ<1保留折扣目标。报告原值重构RMSE，同时单列centered尺度。
不把参照c说成on-policy reward running mean，也不声称γ<1等于平均奖励。
"""
import math
import random
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from ._common import record, check, main

def centered_step(values,state,next_state,reward,center,alpha=.1,eta=.05,gamma=.99,enabled=True):
    delta=reward-center+gamma*values[next_state]-values[state]
    values[state]+=alpha*delta
    return center+(eta*alpha*delta if enabled else 0.),delta

def exact_values(rewards,gamma=.99):
    n=len(rewards)
    return [sum(gamma**k*rewards[(s+k)%n] for k in range(n))/(1-gamma**n) for s in range(n)]

def run(seed=0,steps=1200,emit=None,centered=True):
    check(steps)
    rng=random.Random(seed); rows=[]; values=[0.,0.,0.]; c=0.; s=rng.randrange(3)
    rewards=[9.,10.,11.]; truth=exact_values(rewards)
    for t in range(1,steps+1):
        sn=(s+1)%3
        c,delta=centered_step(values,s,sn,rewards[s],c,enabled=centered)
        s=sn
        reconstructed=[v+c/.01 for v in values]
        rmse=math.sqrt(sum((x-y)**2 for x,y in zip(reconstructed,truth))/3)
        record(rows,t,steps,rmse,emit,reference=c,td_error=delta,max_centered_value=max(abs(v) for v in values),reconstructed_value0=reconstructed[0])
    return rows

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": False,
    "id": "reward_centering",
    "name": "Reward Centering：TD驱动折扣参照",
    "task": "ek_centered_prediction",
    "baseline": "reward_uncentered",
    "metric": "冻结重构原折扣价值RMSE",
    "unit": "rmse",
    "budget": "environment_steps",
    "coverage_refs": [
        "core-reward-centering"
    ],
    "chapter_paths": [
        "algorithms/average-reward/"
    ],
    "sources": [
        "https://arxiv.org/abs/2405.09999"
    ],
    "description": "三状态确定性continuing环，奖励(9,10,11)，γ=.99 α=.1，on-policy V TD；参照c+=ηαδ (η=.05)，目标r-c+γV(s')；原值重构V+c/(1-γ)，解析真值从环几何级数；没有done或episodes。",
    "limitations": "原2024论文value-based reward centering的表格prediction组件，仍γ<1非Differential TD/平均奖励控制；c的固定点与初始化耦合，不能当行为奖励均值或无偏目标reward-rate。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？"
}

if __name__=="__main__":
    main(META,run)
