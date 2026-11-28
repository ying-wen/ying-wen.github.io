"""绝对学习进展课程：|mean(旧窗误差)-mean(新窗误差)|。
只使用已见误差选择任务；不能让真实斜率/未来学习率成为调度输入。
实验含不可约噪声任务，展示单看误差或进展均可能受噪声影响。
"""
import random
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from ._common import greedy, record, check, main

def progress(history,window=10):
    if len(history)<2*window:
        return 0.
    return abs(sum(history[-2*window:-window])/window-sum(history[-window:])/window)

def run(seed=0,steps=1200,emit=None,adaptive=True):
    check(steps)
    rng=random.Random(seed); rows=[]; w=[0.,0.,0.]
    histories=[[],[],[]]; counts=[0,0,0]; slopes=[1.,-2.,0.]
    for t in range(1,steps+1):
        # 未满两个窗先补足证据；之后epsilon保证所有任务仍有机会被访问。
        warm=[j for j,h in enumerate(histories) if len(h)<20]
        task=rng.choice(warm) if adaptive and warm else (
            greedy([progress(h) for h in histories],rng,.2) if adaptive else rng.randrange(3))
        x=rng.uniform(-1,1)
        y=slopes[task]*x+(rng.gauss(0,1) if task==2 else 0.)
        error=w[task]*x-y
        histories[task].append(error*error)
        histories[task]=histories[task][-20:]
        w[task]-=.05*2*error*x
        counts[task]+=1
        # 冻结解析评价：均匀x的E[x²]=1/3，任务2包含真实不可约噪声方差1。
        mse=sum((a-b)**2/3 for a,b in zip(w,slopes))/3+1/3
        record(rows,t,steps,mse,emit,task=task,task0_count=counts[0],task1_count=counts[1],noisy_task_count=counts[2],progress0=progress(histories[0]))
    return rows

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": False,
    "id": "learning_progress",
    "name": "Learning progress：双窗课程",
    "task": "ek_progress_regression",
    "baseline": "uniform_curriculum",
    "metric": "冻结三任务期望预测MSE",
    "unit": "mse",
    "budget": "training_examples",
    "coverage_refs": [
        "core-progress"
    ],
    "chapter_paths": [
        "algorithms/exploration/"
    ],
    "sources": [
        "https://doi.org/10.1109/TEVC.2006.890271"
    ],
    "description": "三个标量回归任务斜率1/-2/0，第三有σ=1不可约噪声；各任务SGD .05，20误差队列两个10窗均值差绝对值，ε=.2最高progress调度；固定真实斜率只用于环境/冻结MSE评价。",
    "limitations": "学习进展调度组件，非原IAC区域分裂机器人系统；噪声可能欺骗progress，不能保证课程优于均匀。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？"
}

if __name__=="__main__":
    main(META,run)
