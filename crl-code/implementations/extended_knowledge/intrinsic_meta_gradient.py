"""元梯度须穿过真实参数更新而非直接调η。
p=σθ，θ'=θ+α(1+η)p(1-p)，Jexternal(θ')=σθ'。
dJ/dη=σθ'(1-σθ')·αp(1-p)，outer用外奖而不是混合内奖。
"""
import random
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from ._common import sigmoid, record, check, main

def inner(theta,eta,alpha=.1):
    p=sigmoid(theta)
    return theta+alpha*(1+eta)*p*(1-p)

def meta_gradient(theta,eta,alpha=.1):
    p=sigmoid(theta); next_theta=inner(theta,eta,alpha); pn=sigmoid(next_theta)
    return pn*(1-pn)*alpha*p*(1-p)

def run(seed=0,steps=1200,emit=None,learn_reward=True):
    check(steps)
    rng=random.Random(seed); theta=rng.uniform(-.2,.2); eta=0.; rows=[]
    for t in range(1,steps+1):
        # 先根据旧θ/η计算内层更新和穿过该更新的outer梯度。
        next_theta=inner(theta,eta)
        gradient=meta_gradient(theta,eta)
        if learn_reward:
            eta+=.2*gradient
        theta=next_theta
        record(rows,t,steps,sigmoid(theta),emit,phase="meta-training",intrinsic_parameter=eta,meta_gradient=gradient,inner_updates=t,outer_updates=t if learn_reward else 0)
    return rows

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": True,
    "id": "intrinsic_meta_gradient",
    "name": "内奖元梯度：展开一次策略更新",
    "task": "ek_intrinsic_meta",
    "baseline": "intrinsic_fixed_reward",
    "metric": "冻结更新后外在奖励期望",
    "unit": "return",
    "budget": "meta_updates",
    "coverage_refs": [
        "core-intrinsic-meta"
    ],
    "chapter_paths": [
        "foundations/reward-design/"
    ],
    "sources": [
        "https://arxiv.org/abs/1804.06459"
    ],
    "description": "二动作bandit外奖(0,1)，内奖η·1(a=1)。logit θ用精确策略梯度 α=.1 内层一步，外层η沿更新后外奖精确链式梯度 β=.2；θ在线推进，η初值0。每step一个内外更新，额外梯度评价计数。",
    "limitations": "可微内奖reward-design组件；精确有限bandit而非原A2C/PPO/延迟信用系统；不含多步unroll Hessian/state分布项，不能推出复杂任务优势。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？"
}

if __name__=="__main__":
    main(META,run)
