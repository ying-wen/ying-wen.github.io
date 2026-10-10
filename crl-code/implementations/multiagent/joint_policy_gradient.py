"""联合critic策略梯度对照：保留COMA任务/critic，移除反事实基线。

每次保存采样前集中Q作为 A_i，再分别更新critic和两个局部actor。
不是独立奖励学习：所有actor仍共享团队奖励与集中critic。
相比COMA，唯一算法干预是基线从边际化自身动作改为零。
两者的critic误差、采样分布和训练后策略会不同，不能只看回报就断言降方差。
"""
import random
from pathlib import Path
try:
    from ._common import probabilities,reward,policy_return,emit_row
except ImportError:
    from _common import probabilities,reward,policy_return,emit_row

META = {
    "id":"joint_policy_gradient","name":"联合critic策略梯度：无反事实基线",
    "family":"multiagent","chapter_paths":["foundations/deep/multi-agent/"],
    "coverage_refs":[],"implementation_kind":"component_experiment",
    "description":"与COMA教学实验相同的联合critic、局部actor和一步合作任务，基线为零。",
    "question":"保留集中critic但去掉反事实基线，学习曲线与梯度有何变化？",
    "task":"contextual_cooperative_actor_critic","baseline":"coma_contextual",
    "metric":"冻结随机联合策略的精确期望奖励","unit":"reward","higher_better":True,
    "scope":"ablation-baseline", "budget":"environment_steps","defaults":{"steps":1200},
    "budget_note":"每步一个联合动作；两actor各写入一次。枚举评价不记为训练样本。",
    "initialization":"actor logits和critic均为零；局部观测与COMA相同。",
    "limitations":"一步表格对照；并非对所有多智能体actor–critic的性能结论。",
    "sources":["https://arxiv.org/abs/1705.08926"]
}


def run(seed=0,steps=1200,emit=None):
    if type(steps) is not int or steps<1: raise ValueError("steps must be positive")
    rng=random.Random(seed); rows=[]
    theta=[[[0.,0.] for _ in range(2)] for _ in range(2)]
    critic=[[0.]*4 for _ in range(4)]
    emit_row(rows,0,steps,policy_return(theta),emit,critic_squared_error=0.,actor_updates=0)
    for t in range(1,steps+1):
        bits=[rng.randrange(2),rng.randrange(2)]
        pi=[probabilities(theta[i][bits[i]]) for i in range(2)]
        actions=[int(rng.random()>=pi[i][0]) for i in range(2)]
        q=critic[2*bits[0]+bits[1]]; index=2*actions[0]+actions[1]
        error=reward(bits,actions)-q[index]
        advantage=q[index]
        q[index] += .15*error
        # A_i is stop-gradient Q; this explicit tabular update has no graph to detach.
        for i in range(2):
            for k in range(2):
                theta[i][bits[i]][k] += .04*advantage*((k==actions[i])-pi[i][k])
        emit_row(rows,t,steps,policy_return(theta),emit,
                 critic_squared_error=error*error,actor_updates=2*t)
    return rows


if __name__=="__main__":
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
