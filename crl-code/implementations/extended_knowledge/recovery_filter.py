"""恢复过滤：提议前进行为→查学习的reset成功概率→拒绝/替换→真实恢复trial。
真实成功率仅用于环境生成结果，不能直接交给过滤器；无合格动作不假称安全。
"""
import random
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from ._common import greedy, record, check, main

def filtered_action(proposed,estimates,threshold=.75):
    safe=[a for a,p in enumerate(estimates) if p>=threshold]
    if proposed in safe:
        return proposed
    candidates=safe or list(range(len(estimates)))
    return max(candidates,key=lambda a:estimates[a])

def run(seed=0,steps=1200,emit=None,filtered=True):
    check(steps)
    rng=random.Random(seed); rows=[]; success=[1.,1.,1.]; failure=[1.,1.,1.]
    totals=[0.,0.,0.]; counts=[0,0,0]; q=[0.,0.,0.]
    env_prob=[.98,.6,.05]; gain=[0.,.4,1.]
    reward_sum=0.; failed=0; rejected=0
    for t in range(1,steps+1):
        # 预先声明的小型数据采集也计入总预算与失败，不能藏掉危险warmup。
        proposed=(t-1)%3 if t<=30 else greedy(q,rng,.2)
        estimates=[s/(s+f) for s,f in zip(success,failure)]
        a=proposed if t<=30 or not filtered else filtered_action(proposed,estimates)
        rejected+=int(a!=proposed)
        ok=rng.random()<env_prob[a]
        success[a]+=int(ok); failure[a]+=int(not ok)
        # 前进任务价值与安全过滤分开；Q估计forward收益，评价净值含失败代价。
        counts[a]+=1; totals[a]+=gain[a]; q[a]=totals[a]/counts[a]
        reward=gain[a]-2*int(not ok); reward_sum+=reward; failed+=int(not ok)
        record(rows,t,steps,reward_sum/t,emit,reset_failure_rate=failed/t,interventions=rejected,estimated_reset_probability=estimates[a])
    return rows

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": True,
    "id": "recovery_filter",
    "name": "恢复概率过滤：学习后再选择",
    "task": "ek_reset_filter",
    "baseline": "recovery_unfiltered",
    "metric": "在线净奖励均值（含恢复失败成本）",
    "unit": "return",
    "budget": "reset_trials",
    "coverage_refs": [
        "core-recovery"
    ],
    "chapter_paths": [
        "algorithms/exploration/"
    ],
    "sources": [
        "https://arxiv.org/abs/1711.06782"
    ],
    "description": "三种前进行为，收益0/.4/1，真实恢复概率.98/.6/.05，仅环境知道；失败成本2。前30trial轮流收集恢复证据，之后Beta(1,1)后验均值过滤阈值.75、无合格选最高估计；奖励Q均值ε=.2；统计全部trial。",
    "limitations": "从实际恢复结果学习的过滤组件；不是完整Leave No Trace双策略/神经reset critic，不提供统计安全证书，初始化和误估可能失败。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？"
}

if __name__=="__main__":
    main(META,run)
