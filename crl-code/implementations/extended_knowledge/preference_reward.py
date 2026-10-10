"""BT偏好概率：P(A优于B)=σ(w·(ΦA-ΦB))。
负对数似然梯度：(p-y)(ΦA-ΦB)；训练真实随机偏好标签而非拟合已知参数。
冻结评价遍历有限片段对，使用teacher概率计算期望CE，包含不可约偏好噪声。
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

from ._common import sigmoid, record, check, main

def trajectory_features(actions):
    return [float(sum(actions)),float(sum(a!=b for a,b in zip(actions,actions[1:])))]

def loss_gradient(weights,difference,label):
    logit=sum(w*x for w,x in zip(weights,difference))
    loss=max(logit,0)+math.log1p(math.exp(-abs(logit)))-label*logit
    return loss,[(sigmoid(logit)-label)*x for x in difference]

def expected_loss(weights,features):
    total=0.
    for a in features:
        for b in features:
            d=[x-y for x,y in zip(a,b)]
            teacher=sigmoid(d[0]-.7*d[1])
            total+=loss_gradient(weights,d,teacher)[0]
    return total/len(features)**2

def run(seed=0,steps=1200,emit=None,full_features=True):
    check(steps)
    rng=random.Random(seed); rows=[]; weights=[0.,0.]
    features=[trajectory_features(a) for a in itertools.product([0,1],repeat=3)]
    for t in range(1,steps+1):
        a,b=rng.choice(features),rng.choice(features)
        d=[x-y for x,y in zip(a,b)]
        label=float(rng.random()<sigmoid(d[0]-.7*d[1]))
        _,gradient=loss_gradient(weights,d,label)
        weights[0]-=.08*gradient[0]
        if full_features:
            weights[1]-=.08*gradient[1]
        record(rows,t,steps,expected_loss(weights,features),emit,phase="preference-fitting",right_reward_weight=weights[0],turn_reward_weight=weights[1])
    return rows

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": False,
    "id": "preference_reward",
    "name": "Bradley–Terry：偏好奖励训练",
    "task": "ek_preference_reward",
    "baseline": "preference_single_feature",
    "metric": "冻结偏好期望交叉熵",
    "unit": "nats",
    "budget": "preference_labels",
    "coverage_refs": [
        "core-preference-reward"
    ],
    "chapter_paths": [
        "foundations/reward-design/"
    ],
    "sources": [
        "https://arxiv.org/abs/1706.03741"
    ],
    "description": "长度3 binary-action轨迹，features=(右动作数,转向数)，teacher w=(1,-.7)仅生成偏好；每step一个随机轨迹对Bernoulli label，BT SGD .08；所有64轨迹对以teacher分布精确冻结期望交叉熵。",
    "limitations": "独立线性片段偏好奖励拟合组件；不含人类交互收集/主动查询/策略优化闭环，reward平移不可辨识；精确teacher仅评价不用训练梯度。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？"
}

if __name__=="__main__":
    main(META,run)
