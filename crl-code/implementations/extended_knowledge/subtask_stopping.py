"""奖励尊重子任务：真实奖励r + 到达停止状态时的z，z项不再乘γ。
STOMP时间约定：target=r+β(s')z(s')+γ(1-β(s'))V(s')。
停止是option终止，不是环境done；环境真终止强制无bootstrap。
"""
import random
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from ._common import walk, greedy, record, check, main

def stopping_target(reward,z,next_value,beta,gamma=.9,terminal=False):
    return reward if terminal else reward+beta*z+gamma*(1-beta)*next_value

def score(q,stopping_bonus,objective_bonus=.8):
    s=0; total=0.
    for k in range(30):
        sn=walk(s,greedy(q[s]))
        r=1. if sn==6 else -.04
        total+=.9**k*r
        if sn==6:
            break
        z=stopping_bonus if sn==3 else 0.
        if z>=max(q[sn]):
            # 评价保持同一客观目标，但执行的是该训练条件真正学到的停止规则。
            total+=.9**k*(objective_bonus if sn==3 else 0.)
            break
        s=sn
    return total

def run(seed=0,steps=1200,emit=None,use_bonus=True):
    check(steps)
    rng=random.Random(seed); rows=[]; q=[[0.,0.] for _ in range(7)]
    bonus=.8 if use_bonus else 0.
    for t in range(1,steps+1):
        # 全支持行为使两个方向均可学；不把已知解作为策略输入。
        s=rng.randrange(6); a=rng.randrange(2); sn=walk(s,a)
        terminal=sn==6
        reward=1. if terminal else -.04
        z=bonus if sn==3 else 0.
        beta=float(z>=max(q[sn]))
        target=stopping_target(reward,z,max(q[sn]),beta,terminal=terminal)
        q[s][a]+=.2*(target-q[s][a])
        # baseline也用同一原子问题的bonus .8测量，避免更换评价目标。
        record(rows,t,steps,score(q,bonus),emit,stopping_at_subgoal=int(bonus>=max(q[3])),trained_bonus=bonus)
    return rows

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": True,
    "id": "subtask_stopping",
    "name": "STOMP：奖励尊重子任务停止",
    "task": "ek_stopping_subtask",
    "baseline": "subtask_no_bonus",
    "metric": "起点冻结子任务目标回报",
    "unit": "return",
    "budget": "environment_steps",
    "coverage_refs": [
        "core-subtask"
    ],
    "chapter_paths": [
        "construction/goals/"
    ],
    "sources": [
        "https://arxiv.org/abs/2202.03466"
    ],
    "description": "0..6链，环境一步-.04，6终止得1；状态3到达停止bonus .8。均匀行为采样，α=.2 γ=.9；停止β由z(s')>=maxQ(s')，目标r+βz(s')+γ(1-β)V。冻结起点0评价最多30步。",
    "limitations": "仅STOMP子任务策略/停止组件；bonus不写入主任务环境模型；不含完整STOMP生成/模型/规划闭环。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？"
}

if __name__=="__main__":
    main(META,run)
