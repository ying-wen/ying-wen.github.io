"""COMA 反事实优势：一步 Dec-POMDP 中的完整表格 actor–critic 学习。

问题：两智能体各看到一个独立均匀比特 o_i，选择 a_i∈{0,1}。
团队奖励是 .2 I[a_0=o_0]+.2 I[a_1=o_1]+.6 I[两者都对]。
集中 critic 看见联合状态 s=(o_0,o_1)；执行时 actor i 只用 o_i。
目标 J(θ)=E_{s,a~πθ}[r(s,a)]，不是个体奖励最大化。

步骤1：用更新前 π_i 采样联合动作，保存这次的动作概率。
步骤2：用采样前的critic计算 b_i=Σ_u π_i(u|o_i)Q(s,(a_-i,u))，A_i=Q(s,a)-b_i。
步骤3：Q(s,a)←Q(s,a)+α_c[r-Q(s,a)]；一步回合故无 bootstrap。
步骤4：θ_i[o_i,k]←θ_i[o_i,k]+α_a A_i(I[k=a_i]-π_i(k|o_i))。
两个 actor 都使用同一份旧策略和采样前 critic，不让本次奖励污染基线。
基线对 a_i 不依赖，所以 E_{a_i~π_i}[b_i∇logπ_i]=0。
实验估计 critic 后训练 actor；上式只说明基线不增加偏差，不消除critic误差。
本例无多步 TD(λ)、循环神经网络或 StarCraft；不是原论文工程复现。
"""
import random
from pathlib import Path
try:
    from ._common import probabilities, reward, policy_return, emit_row
except ImportError:
    from _common import probabilities, reward, policy_return, emit_row

META = {
    "id":"coma_contextual", "name":"COMA：反事实优势与策略训练",
    "family":"multiagent", "chapter_paths":["foundations/deep/multi-agent/"],
    "coverage_refs":["core-coma"], "implementation_kind":"component_experiment",
    "description":"两智能体一步合作任务；局部策略、学习的联合critic与反事实基线。",
    "question":"边际化自身动作的基线怎样进入可运行的集中训练、分散执行过程？",
    "task":"contextual_cooperative_actor_critic", "baseline":"joint_policy_gradient",
    "metric":"冻结随机联合策略的精确期望奖励", "unit":"reward", "higher_better":True,
    "scope":"tabular-COMA-in-one-step-Dec-POMDP",
    "budget":"environment_steps", "defaults":{"steps":1200},
    "budget_note":"每次联合动作是一条转移和一个完整回合；两个actor更新额外记录。",
    "initialization":"actor logits与集中critic均为零；context均匀采样。评价枚举4种状态与4种联合动作。",
    "limitations":"COMA的一步表格特例；无循环actor、多步critic或原论文任务。两方法共享critic学习规则；比较不是等梯度方差保证。",
    "sources":["https://arxiv.org/abs/1705.08926",
               "https://github.com/oxwhirl/pymarl/blob/master/src/learners/coma_learner.py"]
}


def advantages(q, probabilities_by_agent, actions):
    chosen = q[2*actions[0]+actions[1]]
    baseline0 = sum(probabilities_by_agent[0][u]*q[2*u+actions[1]] for u in range(2))
    baseline1 = sum(probabilities_by_agent[1][u]*q[2*actions[0]+u] for u in range(2))
    return [chosen-baseline0, chosen-baseline1]


def actor_update(theta, bits, actions, pi, advantage, alpha=.04):
    for i in range(2):
        for k in range(2):
            theta[i][bits[i]][k] += alpha*advantage[i]*((k==actions[i])-pi[i][k])


def run(seed=0,steps=1200,emit=None):
    if type(steps) is not int or steps<1:
        raise ValueError("steps must be positive")
    rng=random.Random(seed)
    theta=[[[0.,0.] for _ in range(2)] for _ in range(2)]
    critic=[[0.]*4 for _ in range(4)]
    rows=[]
    emit_row(rows,0,steps,policy_return(theta),emit,critic_squared_error=0.,actor_updates=0)
    for t in range(1,steps+1):
        bits=[rng.randrange(2),rng.randrange(2)]
        pi=[probabilities(theta[i][bits[i]]) for i in range(2)]
        actions=[int(rng.random()>=pi[i][0]) for i in range(2)]
        r=reward(bits,actions); q=critic[2*bits[0]+bits[1]]
        index=2*actions[0]+actions[1]
        error=r-q[index]
        advantage=advantages(q,pi,actions)
        q[index] += .15*error
        actor_update(theta,bits,actions,pi,advantage)
        emit_row(rows,t,steps,policy_return(theta),emit,
                 critic_squared_error=error*error,actor_updates=2*t)
    return rows


if __name__=="__main__":
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
