"""Intentional TD + Intentional Policy Gradient 的实际流式actor-critic组件。
独立linear V与softmax策略，lambda=.8，gamma=1；环境真实终止后清eligibility。
优化器顺序：旧delta -> RMS梯度尺度 -> 当前梯度sigma -> trace/sigma EMA ->
alpha -> delta RMS裁剪 -> actor额外L1delta归一化 -> 参数更新 -> 清trace。
actor gradient=grad[logpi + .01*H*sign(旧delta)]，符号固定，熵进入trace/RMS。
critic与actor都读旧参数，同一delta；不是仅仅NLMS标量核或完整论文benchmark。
启用作者默认RMS/clip/norm衰减；eta使用小任务设置value=.2、policy=.03。
"""
import math
import random
import torch
from torch import nn
try:
    from ._common import *
except ImportError:
    from _common import *
META = {
    "id": "extended-intentional",
    "name": "Intentional AC流式组件",
    "family": "extended_adaptation",
    "task": "linear-trace-chain",
    "baseline": "extended-trace_ac",
    "metric": "frozen_greedy_return",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "implementation_kind": "component_experiment",
    "coverage_refs": [
        "core-intentional"
    ],
    "chapter_paths": [
        "algorithms/streaming/"
    ],
    "description": "Intentional AC流式组件的实际CPU教学实验；设置、公式和边界详见本文件。",
    "question": "实际学习或状态更新怎样改变此教学任务的独立测量？",
    "sources": [
        "https://github.com/sharifnassab/Intentional_RL"
    ],
    "defaults": {
        "steps": 600
    },
    "budget": "environment_steps",
    "budget_note": "按实际环境转移计步；各算法更新/回放频率见本文件循环，环境步相同不等于相同计算成本。",
    "limitations": "保留RMSProp/trace/sigma/剪裁与归一化核心的线性离散actor-critic；不复现连续控制/Atari系统。冻结贪心回报不等同随机策略目标。"
}

class IntentionalOptimizer:
    def __init__(self,size,eta,q=.8,policy=False):
        self.eta,self.q,self.policy=eta,q,policy
        self.t=0; self.sigma=self.delta_sq=self.delta_abs=0.
        self.v=torch.zeros(size); self.trace=torch.zeros(size)
    def step(self,gradient,delta,terminal=False):
        self.t+=1
        self.v=.999*self.v+.001*gradient.square()
        rho=1/(torch.sqrt(self.v/(1-.999**self.t))+1e-8)
        current=float((rho*gradient.square()).sum())
        self.trace=self.q*self.trace+gradient
        self.sigma=self.q*self.sigma+(1-self.q)*current
        sigma=self.sigma/(1-self.q**self.t)
        mass=float((rho*self.trace.square()).sum())
        alpha=self.eta/max(math.sqrt(max(0.,sigma*mass)),1e-8)
        self.delta_sq=.9998*self.delta_sq+.0002*delta*delta
        cap=20*math.sqrt(self.delta_sq/(1-.9998**self.t))
        safe=math.copysign(min(abs(delta),cap),delta)
        if self.policy:
            self.delta_abs=.9998*self.delta_abs+.0002*abs(safe)
            safe/=max(self.delta_abs/(1-.9998**self.t),1e-12)
        update=alpha*safe*rho*self.trace
        if terminal:self.trace.zero_()
        return update,alpha

def policy_gradient(probabilities,a,delta,entropy_coefficient=.01):
    """熵梯度乘 sign(delta)；零TD误差时必须是0而非copysign返回的+1。"""
    score=torch.nn.functional.one_hot(torch.tensor(a),2).float()-probabilities
    logp=probabilities.clamp(min=1e-8).log()
    entropy=-(probabilities*logp).sum()
    entropy_gradient=-probabilities*(logp+entropy)
    sign=1. if delta>0 else -1. if delta<0 else 0.
    return score+entropy_coefficient*entropy_gradient*sign,float(entropy)

def run(seed=0,steps=600,emit=None):
    rng=setup(seed,steps); value=torch.zeros(6); policy=torch.zeros(2,6)
    critic_opt=IntentionalOptimizer(6,.2); actor_opt=IntentionalOptimizer(12,.03,policy=True)
    env=Chain(); x=env.reset(); rows=[]; ticks=grid(steps)
    actor=lambda x:policy@x
    log(rows,0,chain_score(actor),emit)
    for t in range(1,steps+1):
        probabilities=(policy@x).softmax(0)
        a=int(rng.random()<float(probabilities[1]))
        xp,r,d=env.step(a)
        delta=float(r+(0 if d else 1)*(value@xp)-value@x)
        score_gradient,entropy=policy_gradient(probabilities,a,delta)
        actor_gradient=score_gradient[:,None]*x
        # 两个优化器都读旧delta和旧模型梯度；梯度不穿过bootstrap。
        dv,critic_alpha=critic_opt.step(x,delta,d)
        dp,actor_alpha=actor_opt.step(actor_gradient.reshape(-1),delta,d)
        value+=dv; policy+=dp.reshape(2,6)
        x=env.reset() if d else xp
        if t in ticks:
            log(rows,t,chain_score(actor),emit,td_error=delta,value_step=critic_alpha,
                policy_step=actor_alpha,entropy=entropy)
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
