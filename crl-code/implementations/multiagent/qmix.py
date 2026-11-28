"""QMIX：三步合作任务中的神经单调混合、回放和目标网络。

每个agent只输入自己的比特、时间和身份；集中mixer额外输入两个比特。
Q_tot(s,a)=w2(s)^T ELU(Q_local(a)^T W1(s)+b1(s))+b2(s)。
W1,w2通过绝对值限制为非负，ELU导数非负，故∂Q_tot/∂Q_i≥0。
固定s时，各agent的局部argmax的组合也是联合Q_tot的argmax。
这不是任意联合Q都可表示：非单调支付表构成反例。

每步：ε-greedy联合动作→存转移→随机32条回放→构造冻结目标
y=r+.9(1-d) Q_tot,target(s',argmax_a' Q_local,target(o',a'))。
再最小化均方TD误差，每50训练更新复制agent与mixer目标网络。
终止发生在第三步；总预算停止不是额外终止。评价只用局部贪心agent，
枚举4种context，不使用mixer的全局信息来选动作。
此例保留QMIX结构与TD学习，但没有循环网络、SMAC或长时部分可观测。
"""
import copy
from pathlib import Path
import torch
from torch import nn
try:
    from ._common import seed_torch,local_features,global_features,agent_network,reward,batch,greedy_return,emit_row
except ImportError:
    from _common import seed_torch,local_features,global_features,agent_network,reward,batch,greedy_return,emit_row

META = {
    "id":"qmix_cooperative","name":"QMIX：神经单调混合与TD训练",
    "family":"multiagent","chapter_paths":["foundations/deep/multi-agent/"],
    "coverage_refs":["core-qmix"],"implementation_kind":"component_experiment",
    "description":"两agent三步Dec-POMDP，集中状态条件mixer、局部Q网络、经验回放与冻结目标。",
    "question":"非负混合权重怎样保证分散贪心一致性，并参与真正的TD训练？",
    "task":"three_step_cooperative_value_learning","baseline":"vdn_cooperative",
    "metric":"冻结分散贪心策略的折扣回报","unit":"return","higher_better":True,
    "scope":"feedforward-QMIX-small-Dec-POMDP","budget":"environment_steps","defaults":{"steps":1200},
    "budget_note":"一个联合转移计一步，32条预热后每步一次batch更新；QMIX比VDN多mixer参数与计算。",
    "initialization":"Torch种子初始化，目标网络复制在线参数，ε=.2；γ=.9，回合固定三步。评价无训练随机数。",
    "limitations":"可运行的小型前馈QMIX实例；未复现循环SMAC系统，不证明持续多智能体泛化或相同计算成本。",
    "sources":["https://proceedings.mlr.press/v80/rashid18a.html",
               "https://github.com/oxwhirl/pymarl/blob/master/src/modules/mixers/qmix.py"]
}


class Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hyper_w1=nn.Linear(7,16)
        self.hyper_w2=nn.Linear(7,8)
        self.b1=nn.Linear(7,8)
        self.b2=nn.Sequential(nn.Linear(7,8),nn.ReLU(),nn.Linear(8,1))

    def forward(self,chosen_q,state):
        w1=self.hyper_w1(state).abs().reshape(-1,2,8)
        w2=self.hyper_w2(state).abs().reshape(-1,8,1)
        hidden=torch.nn.functional.elu(torch.bmm(chosen_q.unsqueeze(1),w1)+self.b1(state).unsqueeze(1))
        return (torch.bmm(hidden,w2).flatten()+self.b2(state).flatten())


def update(agent,mixer,target_agent,target_mixer,optimizer,data):
    x,s,a,r,xp,sp,done=data
    chosen=agent(x).gather(-1,a.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_local=target_agent(xp).max(-1).values
        y=r+.9*(1-done)*target_mixer(next_local,sp)
    error=mixer(chosen,s)-y
    loss=error.square().mean()
    optimizer.zero_grad();loss.backward()
    torch.nn.utils.clip_grad_norm_(list(agent.parameters())+list(mixer.parameters()),10.)
    optimizer.step()
    return float(loss.detach())


def run(seed=0,steps=1200,emit=None):
    if type(steps) is not int or steps<1: raise ValueError("steps must be positive")
    rng=seed_torch(seed)
    agent=agent_network();mixer=Mixer()
    target_agent=copy.deepcopy(agent);target_mixer=copy.deepcopy(mixer)
    optimizer=torch.optim.Adam(list(agent.parameters())+list(mixer.parameters()),lr=.003)
    replay=[];rows=[];bits=[rng.randrange(2),rng.randrange(2)];clock=0;updates=0;loss=0.
    emit_row(rows,0,steps,greedy_return(agent),emit,gradient_updates=0,td_loss=0.)
    for t in range(1,steps+1):
        x=local_features(bits,clock);s=global_features(bits,clock)
        with torch.no_grad(): actions=agent(x).argmax(-1).tolist()
        actions=[rng.randrange(2) if rng.random()<.2 else a for a in actions]
        r=reward(bits,actions);done=clock==2
        # Terminal next encoding is a placeholder only; done masks its value.
        next_clock=0 if done else clock+1
        replay.append((x,s,actions,r,local_features(bits,next_clock),global_features(bits,next_clock),done))
        if len(replay)>2048: replay.pop(0)
        if len(replay)>=32:
            loss=update(agent,mixer,target_agent,target_mixer,optimizer,batch(rng.sample(replay,32)))
            updates+=1
            if updates%50==0:
                target_agent.load_state_dict(agent.state_dict())
                target_mixer.load_state_dict(mixer.state_dict())
        if done: bits=[rng.randrange(2),rng.randrange(2)]
        clock=next_clock
        emit_row(rows,t,steps,greedy_return(agent),emit,gradient_updates=updates,td_loss=loss)
    return rows


if __name__=="__main__":
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
