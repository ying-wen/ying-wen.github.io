"""学习条件期望资格迹
两个等概率历史汇入完整Markov状态，历史资格为[0.72,0,1]或[0,0.72,1]；后继奖励独立取0或2。主预测参数冻结0，条件期望迹由历史样本均值学习；比较累计信用方向与真实均值[0.36,0.36,1]。不训练完整value网络或递归ET混合。
zbar←zbar+(z−zbar)/N；本条δ先用旧zbar形成方向，再吸收标签。
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
    "id": "expected_eligibility_traces",
    "name": "学习条件期望资格迹",
    "family": "extended_classic",
    "coverage_refs": [
        "core-expected-traces"
    ],
    "implementation_kind": "component_experiment",
    "task": "extended_markov_merge_credit",
    "baseline": "sampled_eligibility_reference",
    "metric": "累计平均信用方向 RMSE",
    "unit": "gradient",
    "higher_better": False,
    "budget": "episodes",
    "scope": "component-expected-credit",
    "chapter_paths": [
        "algorithms/credit-assignment/"
    ],
    "sources": [
        "https://arxiv.org/abs/2007.01839"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "两个等概率历史汇入完整Markov状态，历史资格为[0.72,0,1]或[0,0.72,1]；后继奖励独立取0或2。主预测参数冻结0，条件期望迹由历史样本均值学习；比较累计信用方向与真实均值[0.36,0.36,1]。不训练完整value网络或递归ET混合。",
    "question": "条件期望迹预测器怎样从汇合历史学习信用分配？",
    "limitations": "仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。"
}

def update_trace(mean,count,label):
    # 当前标签只在本次信用方向形成后进入预测器，避免路径标签泄漏。
    count+=1
    return [m+(z-m)/count for m,z in zip(mean,label)],count

def credit(reward,trace):
    return [reward*z for z in trace]

def run(seed=0,steps=1200,emit=None):
    check(steps)
    rng,rows,mean,count,total=random.Random(seed),[],[0.,0.,0.],0,[0.,0.,0.]
    truth=[.36,.36,1.]
    for t in range(1,steps+1):
        branch=rng.randrange(2)
        label=[.72*float(branch==0),.72*float(branch==1),1.]
        reward=2.*float(rng.random()<.5)
        direction=credit(reward,mean)
        total=[a+b for a,b in zip(total,direction)]
        mean,count=update_trace(mean,count,label)
        error=math.sqrt(sum((v/t-target)**2 for v,target in zip(total,truth))/3)
        log(rows,t,error,steps,emit,trace_prediction_error=sum((a-b)**2 for a,b in zip(mean,[.36,.36,1.])))
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
