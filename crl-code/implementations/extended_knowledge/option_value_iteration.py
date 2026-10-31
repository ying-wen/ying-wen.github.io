"""Option VI：V(s)←max_o [R_o(s)+Σ_j P_oγ(s,j)V(j)]。
P已含γ^τ，切忌归一化或重复乘γ。每记录step是真实状态备份次数。
"""
import random
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from ._common import walk, record, check, main
from .option_model import exact_model

def backup(reward,endpoints,values):
    return reward+sum(p*v for p,v in zip(endpoints,values))

def known_models():
    r,p=exact_model(); models=[]
    for s in range(6):
        options=[]
        for a in range(2):
            sn=walk(s,a)
            endpoint=[.9*float(j==sn) for j in range(7)]
            options.append((1. if sn==6 else -.02,endpoint))
        options.append((r[s],p[s]))
        models.append(options)
    return models

def run(seed=0,steps=1200,emit=None,include_options=True):
    check(steps)
    rng=random.Random(seed); rows=[]; values=[0.]*7; models=known_models()
    truth=[sum(.9**k*(-.02) for k in range(5-s))+.9**(5-s) for s in range(6)]+[0.]
    for t in range(1,steps+1):
        s=rng.randrange(6)
        available=models[s] if include_options else models[s][:2]
        values[s]=max(backup(r,p,values) for r,p in available)
        error=max(abs(x-y) for x,y in zip(values,truth))
        record(rows,t,steps,error,emit,phase="model-planning",primitive_backups=t*2,option_backups=t if include_options else 0,value_start=values[0])
    return rows

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": False,
    "id": "option_value_iteration",
    "name": "Option Value Iteration：时间抽象规划",
    "task": "ek_option_planning",
    "baseline": "primitive_value_iteration",
    "metric": "冻结精确最优价值最大误差",
    "unit": "absolute_error",
    "budget": "model_backups",
    "coverage_refs": [
        "core-option-vi"
    ],
    "chapter_paths": [
        "construction/planning/"
    ],
    "sources": [
        "https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/sutton%2Bal-1999.pdf"
    ],
    "description": "7格链γ=.9，两个primitive及固定随机option（右概率.8，β=.3）；已知R/Pγ有限模型，随机状态原子最大化backup，每step一个state backup。真V由primitive最优右移闭式计算。",
    "limitations": "独立有限option VI规划组件；不算环境训练，不把Pγ归一化或只乘一次γ；模型精确求解是预处理，模型估计非此实验内容。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？"
}

if __name__=="__main__":
    main(META,run)
