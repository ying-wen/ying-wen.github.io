"""平均奖励 SMDP 选项更新
两状态无终点交替流，从0开始，Bernoulli均值0.2/0.8，目标长期奖励率0.5、差分偏差差0.3。价值及率从0开始；原始转移预算。选项持续1或3步，学旧平均时长并归一化更新；只含固定策略预测子机制，非完整平均奖励选项控制。
δ=R−gLold+v(s')−v(s)；Δv=αδ/Lold，Δg=ηΔv，再更新L。
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
    "id": "average_smdp_option",
    "name": "平均奖励 SMDP 选项更新",
    "family": "extended_classic",
    "coverage_refs": [
        "core-smdp-rate"
    ],
    "implementation_kind": "component_experiment",
    "task": "extended_average_prediction",
    "baseline": "differential_td_prediction",
    "metric": "目标奖励率绝对误差",
    "unit": "reward_rate",
    "higher_better": False,
    "budget": "environment_steps",
    "scope": "teaching-average-reward-prediction",
    "chapter_paths": [
        "algorithms/average-reward/",
        "construction/options/"
    ],
    "sources": [
        "https://arxiv.org/abs/2110.13855"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "两状态无终点交替流，从0开始，Bernoulli均值0.2/0.8，目标长期奖励率0.5、差分偏差差0.3。价值及率从0开始；原始转移预算。选项持续1或3步，学旧平均时长并归一化更新；只含固定策略预测子机制，非完整平均奖励选项控制。",
    "question": "旧平均时长怎样归一化平均奖励选项更新？",
    "limitations": "仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。"
}

def update(v,rate,mean_duration,s,total_reward,duration,sp,alpha=.05,eta=.1,length_step=.05):
    if mean_duration<=0 or duration<1:
        raise ValueError("旧平均时长必须为正，实际持续时间必须至少一步")
    # expected-length版本：本次误差使用旧平均时长，不用本次时长替换。
    delta=total_reward-rate*mean_duration+v[sp]-v[s]
    scaled=alpha*delta/mean_duration
    new=v[:]
    new[s]+=scaled
    return new,rate+eta*scaled,mean_duration+length_step*(duration-mean_duration),delta

def run(seed=0,steps=1200,emit=None):
    check(steps)
    rng,rows,v,rate,s,t=random.Random(seed),[],[0.,0.],0.,0,0
    mean_duration=[1.,1.]
    while t<steps:
        start=s
        duration=rng.choice([1,3])
        total=0.
        for elapsed in range(duration):
            total+=float(rng.random()<[.2,.8][s])
            s=1-s
            t+=1
            complete=elapsed+1==duration
            if complete:
                v,rate,mean_duration[start],delta=update(v,rate,mean_duration[start],start,total,duration,s)
            log(rows,t,abs(rate-.5),steps,emit,bias_difference_error=abs(v[1]-v[0]-.3),rate=rate,completed_option=int(complete))
            if t==steps:
                break
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
