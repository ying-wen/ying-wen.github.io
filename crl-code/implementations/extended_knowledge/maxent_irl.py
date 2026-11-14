"""全局MaxEnt：Pθ(τ)=exp(θ·Φτ)/Zθ，gradient NLL=Eθ[Φ]-Edata[Φ]。
枚举四层DAG所有合法完整path，避免把独立局部softmax误当全轨迹配分函数。
训练专家经验featuremean；teacher参数只生成数据和冻结评价，不提供梯度。
"""
import itertools
import math
import random
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from ._common import softmax, sample, record, check, main
from .preference_reward import trajectory_features

def objective_gradient(theta,features,empirical):
    scores=[sum(w*x for w,x in zip(theta,f)) for f in features]
    m=max(scores); logz=m+math.log(sum(math.exp(v-m) for v in scores))
    probabilities=softmax(scores)
    expected=[sum(p*f[j] for p,f in zip(probabilities,features)) for j in range(len(theta))]
    nll=logz-sum(w*x for w,x in zip(theta,empirical))
    return nll,[x-y for x,y in zip(expected,empirical)],probabilities

def run(seed=0,steps=1200,emit=None,full_features=True):
    check(steps)
    rng=random.Random(seed); rows=[]; theta=[0.,0.]
    features=[trajectory_features(a) for a in itertools.product([0,1],repeat=4)]
    truth=softmax([.8*f[0]-.5*f[1] for f in features])
    demonstrations=[features[sample(truth,rng)] for _ in range(64)]
    empirical=[sum(f[j] for f in demonstrations)/64 for j in range(2)]
    for t in range(1,steps+1):
        _,gradient,_=objective_gradient(theta,features,empirical)
        theta[0]-=.05*gradient[0]
        if full_features:
            theta[1]-=.05*gradient[1]
        _,_,predicted=objective_gradient(theta,features,empirical)
        nll=-sum(p*math.log(max(q,1e-15)) for p,q in zip(truth,predicted))
        record(rows,t,steps,nll,emit,phase="reward-inference",expert_demonstrations=64,right_reward_weight=theta[0],turn_reward_weight=theta[1],feature_gradient_norm=math.sqrt(sum(g*g for g in gradient)))
    return rows

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": False,
    "id": "maxent_irl",
    "name": "MaxEnt IRL：全轨迹配分函数",
    "task": "ek_maxent_paths",
    "baseline": "maxent_single_feature",
    "metric": "冻结专家分布轨迹交叉熵",
    "unit": "nats",
    "budget": "gradient_updates",
    "coverage_refs": [
        "core-maxent-irl"
    ],
    "chapter_paths": [
        "foundations/reward-design/"
    ],
    "sources": [
        "https://www.cs.cmu.edu/~bziebart/publications/maximum-entropy-inverse-reinforcement-learning.html"
    ],
    "description": "4层binary-action确定性DAG，枚举16合法完整轨迹，features=(右边数,转向数)，固定64专家轨迹由θ=(.8,-.5)全局分布采样。每step全批NLL SGD .05，logZ=logsumexp轨迹回报；冻结真实专家分布交叉熵。",
    "limitations": "有限确定性全轨迹MaxEnt IRL奖励学习实例；不是随机动力学causal entropy、无深网/大图采样；64演示数据生成成本另记，可能有采样偏差和奖励不可辨识。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？"
}

if __name__=="__main__":
    main(META,run)
