"""λ-greedy 局部统计量实验
单状态局部混合：固定bootstrap预测0，未来回报N(1,0.25)。每单位含一个统计训练样本和一个独立验证样本；用旧均值/方差选择λ，再看新样本。仅学局部回报矩与λ，不实现整条TD链、状态相关网络和原方法的全部递归估计。
λ=b²/(b²+variance)，均值和二阶中心矩用Welford更新；验证独立采样。
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
    "id": "greedy_adaptive_lambda",
    "name": "λ-greedy 局部统计量实验",
    "family": "extended_classic",
    "coverage_refs": [
        "core-adaptive-lambda"
    ],
    "implementation_kind": "component_experiment",
    "task": "extended_local_lambda_mixture",
    "baseline": "fixed_lambda_reference",
    "metric": "独立验证样本累计混合 MSE",
    "unit": "squared_error",
    "higher_better": False,
    "budget": "sample_pairs",
    "scope": "component-local-bias-variance",
    "chapter_paths": [
        "algorithms/credit-assignment/"
    ],
    "sources": [
        "https://arxiv.org/abs/1607.00446"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "单状态局部混合：固定bootstrap预测0，未来回报N(1,0.25)。每单位含一个统计训练样本和一个独立验证样本；用旧均值/方差选择λ，再看新样本。仅学局部回报矩与λ，不实现整条TD链、状态相关网络和原方法的全部递归估计。",
    "question": "学得的局部回报矩怎样决定bootstrap与MC混合？",
    "limitations": "仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。"
}

def choose_lambda(bias_squared,variance):
    if bias_squared<0 or variance<0:
        raise ValueError("平方偏差和方差不得为负")
    total=bias_squared+variance
    return bias_squared/total if total else 0.

def update_moments(count,mean,m2,value):
    count+=1
    old_error=value-mean
    mean+=old_error/count
    m2+=old_error*(value-mean)
    return count,mean,m2

def run(seed=0,steps=1200,emit=None):
    check(steps)
    rng,rows,count,mean,m2,total=random.Random(seed),[],0,0.,0.,0.
    for t in range(1,steps+1):
        variance=m2/(count-1) if count>1 else 0.
        lam=choose_lambda(mean*mean,variance)
        validation=rng.gauss(1.,.5)
        mixture=(1-lam)*0.+lam*validation
        total+=(mixture-1.)**2
        training=rng.gauss(1.,.5)
        count,mean,m2=update_moments(count,mean,m2,training)
        log(rows,t,total/t,steps,emit,trace_parameter=lam,estimated_variance=variance,observations=2*t)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
