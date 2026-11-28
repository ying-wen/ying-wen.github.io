"""TRPO：共轭梯度、精确离散 KL 的 HVP 与受约束回溯线搜索。

线性 surrogate 最大化与局部二次 KL 近似产生自然梯度方向：
    g = grad mean[(pi_new/pi_old)*A], (H+damping I)d=g。
不构造完整 Hessian：对 KL 梯度与向量的内积再求梯度得到 Hv。
候选全步为 sqrt(2*delta/(d^T(H+damping I)d))*d。
局部二次近似不能保证实际 KL；必须逐个候选检查精确 batch KL。
只有 surrogate 提升且 KL(old||new)<=delta 的候选才被接受。
全部候选失败时完整恢复旧 actor 参数，critic 可继续独立拟合。

当前版本是离散动作、单 worker、全批次的教学特例，GAE lambda=.95，
gamma=1 对应有限 DeadlineChain 的总回报；没有标准 benchmark 复现声明。
KL 约束只针对本批状态上的平均 KL，不是所有状态上的策略变化保证。
来源：https://spinningup.openai.com/en/latest/algorithms/trpo.html
"""
import math
import torch
from torch.distributions import Categorical, kl_divergence
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "deep-trpo",
    "name": "TRPO",
    "family": "deep",
    "chapter_paths": [
        "foundations/deep/trust-region/",
        "algorithms/policy/",
    ],
    "description": "共轭梯度与 KL 回溯约束的实际 TRPO 教学训练",
    "question": "约束策略更新能否在有限预算内提高冻结评价回报？",
    "baseline": "deep-vpg",
    "metric": "frozen_evaluation_return",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "task": "deadline-chain",
    "sources": [
        "https://spinningup.openai.com/en/latest/algorithms/trpo.html",
        "https://arxiv.org/abs/1502.05477",
    ],
    "defaults": {
        "steps": 1000,
    },
    "budget": "environment_steps",
    "limitations": "小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。",
}

def flat_parameters(module):
    return torch.cat([p.detach().reshape(-1) for p in module.parameters()]).clone()

def set_parameters(module,flat):
    # 原地 copy 保留 Parameter identity；回滚不会丢掉 optimizer 的引用。
    offset=0
    with torch.no_grad():
        for p in module.parameters():
            n=p.numel(); p.copy_(flat[offset:offset+n].view_as(p)); offset+=n
    if offset!=flat.numel(): raise ValueError("parameter vector shape mismatch")

def flat_gradient(scalar,parameters,create_graph=False):
    return torch.cat([g.reshape(-1) for g in
                      torch.autograd.grad(scalar,parameters,create_graph=create_graph)])

def conjugate_gradient(matvec,b,iterations=10,tolerance=1e-10):
    """解 SPD 系统 Ad=b；残差已足够小或非正曲率时立即停止。"""
    direction=torch.zeros_like(b); residual=b.clone(); search=residual.clone()
    squared=torch.dot(residual,residual)
    for _ in range(iterations):
        if float(squared)<=tolerance: break
        product=matvec(search); curvature=torch.dot(search,product)
        if not torch.isfinite(curvature) or float(curvature)<=0: break
        alpha=squared/curvature
        direction+=alpha*search; residual-=alpha*product
        new_squared=torch.dot(residual,residual)
        search=residual+(new_squared/squared)*search; squared=new_squared
    return direction

def hessian_vector_product(actor,x,old_probabilities,vector,damping=.1):
    """旧分布必须冻结；H 是 KL(old||current) 对 current 参数的 Hessian。"""
    params=list(actor.parameters())
    old=Categorical(probs=old_probabilities.detach())
    new=Categorical(logits=actor(x))
    kl=kl_divergence(old,new).mean()
    first=flat_gradient(kl,params,create_graph=True)
    second=flat_gradient(torch.dot(first,vector.detach()),params).detach()
    return second+damping*vector

def policy_update(actor,x,actions,old_logp,old_probabilities,advantage,
                  delta=.01,damping=.1,backtrack_iters=10):
    # 步骤 1：快照 actor；固定旧 logp、旧全动作分布和优势。
    previous=flat_parameters(actor)
    old_logp=old_logp.detach(); old_probabilities=old_probabilities.detach()
    advantage=advantage.detach()
    advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    def objective():
        logp=Categorical(logits=actor(x)).log_prob(actions)
        return ((logp-old_logp).exp()*advantage).mean()
    baseline=float(objective().detach())
    gradient=flat_gradient(objective(),list(actor.parameters())).detach()
    # 步骤 2：用 HVP 解自然梯度方向；damping 只用于数值条件改善。
    matvec=lambda vector: hessian_vector_product(actor,x,old_probabilities,vector,damping)
    direction=conjugate_gradient(matvec,gradient)
    curvature=torch.dot(direction,matvec(direction))
    accepted=False; final_kl=0.; improvement=0.; fraction=0.
    # 步骤 3：二次模型给出初始步长，真实 KL 才是最终接受判据。
    if torch.isfinite(curvature) and float(curvature)>1e-12:
        step=direction*math.sqrt(2*delta/float(curvature))
        try:
            for index in range(backtrack_iters):
                candidate_fraction=.8**index
                set_parameters(actor,previous+candidate_fraction*step)
                with torch.no_grad():
                    current=Categorical(logits=actor(x))
                    actual_kl=float(kl_divergence(Categorical(probs=old_probabilities),current).mean())
                    gain=float(objective())-baseline
                if math.isfinite(actual_kl) and math.isfinite(gain) and actual_kl<=delta and gain>0:
                    accepted=True; final_kl=actual_kl; improvement=gain
                    fraction=candidate_fraction
                    break
        finally:
            # 步骤 4：失败/异常必须回滚，不能留在线搜索的最后一个拒绝候选。
            if not accepted: set_parameters(actor,previous)
    return dict(actor_updated=int(accepted),actual_kl=final_kl,
                surrogate_improvement=improvement,accepted_fraction=fraction)

def run(seed=0,steps=1000,emit=None):
    setup(seed); actor,critic=mlp(6,2),mlp(6,1)
    value_opt=torch.optim.Adam(critic.parameters(),lr=.01)
    env=DeadlineChain(); x=env.reset(); rows=[]; ticks=grid(steps)
    xs=[]; actions=[]; logps=[]; probabilities=[]; rs=[]; vs=[]; nvs=[]; ds=[]
    info={}; batches=accepted=0; record(rows,0,actor,emit)
    for t in range(1,steps+1):
        # 步骤 5：固定策略采集批次，缓存整分布供精确 categorical KL。
        with torch.no_grad():
            dist=Categorical(logits=actor(x)); action=dist.sample()
            logp=dist.log_prob(action); probs=dist.probs
            value=float(critic(x).item())
        xp,r,d=env.step(int(action))
        with torch.no_grad(): next_value=float(critic(xp).item())
        xs.append(x); actions.append(action); logps.append(logp); probabilities.append(probs)
        rs.append(r); vs.append(value); nvs.append(next_value); ds.append(d)
        x=env.reset() if d else xp
        if t in ticks:
            # 步骤 6：采样截止 bootstrap；真正终止截断 GAE，环境不被批次重置。
            adv,returns=advantages(rs,vs,nvs,ds,lam=.95)
            observations=torch.stack(xs)
            info=policy_update(actor,observations,torch.stack(actions),
                               torch.stack(logps),torch.stack(probabilities),adv)
            # 步骤 7：critic 的优化与 actor 的 KL 约束独立。
            for _ in range(4):
                loss=(critic(observations).squeeze(-1)-returns).square().mean()
                value_opt.zero_grad(); loss.backward(); value_opt.step()
            batches+=1; accepted+=info["actor_updated"]
            info["value_loss"]=float(loss.detach())
            xs=[]; actions=[]; logps=[]; probabilities=[]; rs=[]; vs=[]; nvs=[]; ds=[]
            record(rows,t,actor,emit,samples=t,updated_batches=batches,
                   actor_updates=accepted,**info)
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
