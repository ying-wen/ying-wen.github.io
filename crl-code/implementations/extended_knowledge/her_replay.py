"""HER：保留事实(s,a,s')，只换goal并重算奖励/任务终止。
不能修改s'，不能把旧目标成功当physical_terminal；时间截断仍允许bootstrap。
步骤：记录完整实际轨迹→从t之后的已达状态取future目标→重算→混合回放TD→冻结评价。
"""
import random
if __package__ in (None, ""):
    # 同时支持独立文件CLI与python -m；不依赖调用者手动设置PYTHONPATH。
    import sys as _bootstrap_sys
    from pathlib import Path as _BootstrapPath
    _bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
    __package__ = "implementations.extended_knowledge"

from ._common import walk, greedy, record, check, main
from .uvfa_shared import success

def relabel(transition,goal):
    s,a,sn,physical_terminal=transition
    # 成功即终止：s==goal时该目标条件任务已经结束，不能训练“离开成功状态”。
    # 跳过整个无效样本，不改s/a/s'，也不擦除其物理终止事实。
    if s==goal:
        return None
    reward=0. if sn==goal else -1.
    done=physical_terminal or sn==goal
    return s,a,sn,goal,reward,done

def future_replay(trajectory,rng):
    replay=[]
    for i,tr in enumerate(trajectory):
        # 仍只选实际记录的future后继；排除起点已经达成的目标。
        legal=[future[2] for future in trajectory[i:] if future[2]!=tr[0]]
        if not legal:
            continue  # 不伪造未来状态，也不将无效样本伪装成terminal。
        replay.append(relabel(tr,rng.choice(legal)))
    return replay

def update(q,tr,alpha=.2,gamma=.95):
    s,a,sn,g,r,done=tr
    target=r+(0. if done else gamma*max(q[sn,g]))
    q[s,g][a]+=alpha*(target-q[s,g][a])
    return target

def run(seed=0,steps=1200,emit=None,hindsight=True):
    check(steps)
    rng=random.Random(seed)
    q={(s,g):[0.,0.] for s in range(7) for g in range(7)}
    rows=[]; replay=[]; trajectory=[]; backups=0
    s=3; g=rng.choice([0,1,2,4,5,6])
    for t in range(1,steps+1):
        a=greedy(q[s,g],rng,.4)
        sn=walk(s,a)
        factual=(s,a,sn,False)
        trajectory.append(factual)
        update(q,relabel(factual,g)); backups+=1
        s=sn
        if s==g or len(trajectory)==12:
            replay.extend(sample for tr in trajectory
                          if (sample:=relabel(tr,g)) is not None)
            if hindsight:
                replay.extend(future_replay(trajectory,rng))
            trajectory=[]
            s=3; g=rng.choice([0,1,2,4,5,6])
        # 重放量固定；HER改变样本目标而不是伪造更多环境交互。
        for _ in range(2):
            if replay:
                update(q,rng.choice(replay)); backups+=1
        if t==1 or t%20==0 or t==steps:
            record(rows,t,steps,success(lambda s,g:q[s,g]),emit,replay_backups=backups,buffer_size=len(replay))
    return rows

META = {
    "family": "extended_knowledge",
    "defaults": {
        "steps": 1200
    },
    "implementation_kind": "component_experiment",
    "scope": "independent-component-experiment",
    "higher_better": True,
    "id": "her_replay",
    "name": "HER：真实未来目标回放",
    "task": "ek_her_navigation",
    "baseline": "goal_replay_plain",
    "metric": "七目标冻结贪心到达率",
    "unit": "fraction",
    "budget": "environment_steps",
    "coverage_refs": [
        "core-her"
    ],
    "chapter_paths": [
        "construction/goals/"
    ],
    "sources": [
        "https://arxiv.org/abs/1707.01495"
    ],
    "description": "7格确定性目标独立链，每回合至多12步；表格Q .2、γ=.95、ε=.4；完成轨迹随机future目标重新算reward/done，每交互另做2次回放；评价42起点目标。",
    "limitations": "表格HER教学训练，非机器人DDPG/神经HER；未来目标只取已记录后继状态，假设动力学与goal无关；回放备份额外诊断。",
    "question": "在明示的小型设置下，该核心学习机制是否按公式运行？"
}

if __name__=="__main__":
    main(META,run)
