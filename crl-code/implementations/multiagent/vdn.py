"""VDN对照：Q_tot=Q_0+Q_1，任务/局部网络/回放/目标时序与QMIX一致。

集中损失可更新两个局部utility；执行仍是分别argmax。VDN没有状态条件mixer。
每条数据的目标是 r+γ(1-d)Σ_i max_a Q_i,target(o'_i,a)。
这里联合奖励包含交互项，简单加性Q未必能精确拟合整个支付表。
局部greedy一致性不意味着函数表达能力足以表示任意合作问题。
"""
import copy
from pathlib import Path
import torch
try:
    from ._common import seed_torch,local_features,global_features,agent_network,reward,batch,greedy_return,emit_row
except ImportError:
    from _common import seed_torch,local_features,global_features,agent_network,reward,batch,greedy_return,emit_row

META={
    "id":"vdn_cooperative","name":"VDN：加性价值分解对照","family":"multiagent",
    "chapter_paths":["foundations/deep/multi-agent/"],"coverage_refs":[],
    "implementation_kind":"component_experiment","scope":"feedforward-VDN-baseline",
    "description":"QMIX三步合作任务的加性混合对照；局部网络、目标同步与回放批次相同。",
    "question":"将状态条件单调mixer换为求和，能保留什么保证，又失去哪些表示能力？",
    "task":"three_step_cooperative_value_learning","baseline":"qmix_cooperative",
    "metric":"冻结分散贪心策略的折扣回报","unit":"return","higher_better":True,
    "budget":"environment_steps","defaults":{"steps":1200},
    "budget_note":"一个联合转移计一步；预热32步后每步一次batch更新。无额外mixer参数。",
    "initialization":"种子初始化，与QMIX相同局部agent网络、γ=.9、ε=.2和三步回合。",
    "limitations":"前馈小型对照，不是完整SMAC工程。两模型结构容量与计算不同。",
    "sources":["https://arxiv.org/abs/1706.05296","https://proceedings.mlr.press/v80/rashid18a.html"]
}


def update(agent,target,optimizer,data):
    x,s,a,r,xp,sp,done=data
    prediction=agent(x).gather(-1,a.unsqueeze(-1)).squeeze(-1).sum(-1)
    with torch.no_grad(): y=r+.9*(1-done)*target(xp).max(-1).values.sum(-1)
    loss=(prediction-y).square().mean()
    optimizer.zero_grad();loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(),10.)
    optimizer.step()
    return float(loss.detach())


def run(seed=0,steps=1200,emit=None):
    if type(steps) is not int or steps<1: raise ValueError("steps must be positive")
    rng=seed_torch(seed);agent=agent_network();target=copy.deepcopy(agent)
    optimizer=torch.optim.Adam(agent.parameters(),lr=.003)
    replay=[];rows=[];bits=[rng.randrange(2),rng.randrange(2)];clock=0;updates=0;loss=0.
    emit_row(rows,0,steps,greedy_return(agent),emit,gradient_updates=0,td_loss=0.)
    for t in range(1,steps+1):
        x=local_features(bits,clock);s=global_features(bits,clock)
        with torch.no_grad(): actions=agent(x).argmax(-1).tolist()
        actions=[rng.randrange(2) if rng.random()<.2 else a for a in actions]
        r=reward(bits,actions);done=clock==2
        next_clock=0 if done else clock+1
        replay.append((x,s,actions,r,local_features(bits,next_clock),global_features(bits,next_clock),done))
        if len(replay)>2048: replay.pop(0)
        if len(replay)>=32:
            loss=update(agent,target,optimizer,batch(rng.sample(replay,32)));updates+=1
            if updates%50==0: target.load_state_dict(agent.state_dict())
        if done: bits=[rng.randrange(2),rng.randrange(2)]
        clock=next_clock
        emit_row(rows,t,steps,greedy_return(agent),emit,gradient_updates=updates,td_loss=loss)
    return rows


if __name__=="__main__":
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
