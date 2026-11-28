"""仅共享环境、网络、采样与评估；算法更新留在各自文件中。"""
import random
import torch
from torch import nn
from torch.distributions import Normal
torch.set_num_threads(1)

def setup(seed):
    torch.manual_seed(seed)
    return random.Random(seed)

def mlp(n, m):
    return nn.Sequential(nn.Linear(n,32), nn.Tanh(), nn.Linear(32,m))

class DeadlineChain:
    """位置 one-hot 加剩余时间：截止日是任务终止，而非外部截断。"""
    def reset(self):
        self.p,self.left=0,12
        return self.obs()
    def obs(self):
        return torch.tensor([float(i==self.p) for i in range(5)]+[self.left/12.])
    def step(self,a):
        self.p=min(4,max(0,self.p+(1 if a else -1)))
        self.left-=1
        done=self.p==4 or self.left==0
        return self.obs(),1. if self.p==4 else -.02,done

class BoundedLQ:
    """有界一维线性二次控制，时间也是状态。40 步为真实有限时域终止。"""
    def __init__(self,seed=0):
        self.rng=random.Random(seed)
    def reset(self):
        self.x=self.rng.uniform(-1.,1.)
        self.left=40
        return self.obs()
    def obs(self):
        return torch.tensor([self.x,self.left/40.])
    def step(self,a):
        u=float(torch.as_tensor(a).reshape(-1)[0].clamp(-1,1))
        reward=-(self.x*self.x+.05*u*u)
        self.x=max(-3.,min(3.,.92*self.x+.3*u))
        self.left-=1
        return self.obs(),reward,self.left==0

class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=mlp(3,1)
    def forward(self,x,a):
        return self.net(torch.cat([x,a],-1)).squeeze(-1)

class GaussianActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=mlp(2,2)
    def forward(self,x,deterministic=False):
        mean,logstd=self.net(x).chunk(2,-1)
        dist=Normal(mean,logstd.clamp(-5,2).exp())
        z=mean if deterministic else dist.rsample()
        a=z.tanh()
        # tanh 变量变换的稳定 Jacobian，不能用未修正的 Gaussian logp。
        correction=2*(0.6931471805599453-z-nn.functional.softplus(-2*z))
        return a,(dist.log_prob(z)-correction).sum(-1)

def batch(replay,rng,n=32):
    rows=rng.sample(list(replay),n)
    x,a,r,xp,d=zip(*rows)
    return torch.stack(x),torch.stack(a),torch.tensor(r),torch.stack(xp),torch.tensor(d,dtype=torch.float32)

def evaluate(actor,continuous=False,stochastic=False):
    # 独立环境种子；确定性评估不消耗训练随机流。
    scores=[]
    with torch.no_grad():
        env=BoundedLQ(991) if continuous else DeadlineChain()
        for _ in range(12):
            x=env.reset(); score=0.; done=False
            while not done:
                a=actor(x,True)[0] if stochastic else actor(x)
                a=a.tanh() if continuous and not stochastic else a
                if not continuous: a=int(a.argmax())
                x,r,done=env.step(a); score+=r
            scores.append(score)
    return sum(scores)/len(scores)

def grid(steps):
    return set([0,steps]+list(range(max(1,(steps+19)//20),steps,max(1,(steps+19)//20))))

def record(rows,step,actor,emit=None,continuous=False,stochastic=False,**diagnostics):
    row=dict(step=step,value=evaluate(actor,continuous,stochastic),phase="evaluation",**diagnostics)
    rows.append(row)
    if emit: emit(row)

def polyak(online,target,tau=.02):
    with torch.no_grad():
        for src,dst in zip(online.parameters(),target.parameters()):
            dst.lerp_(src,tau)

def advantages(rewards,values,next_values,terminals,lam=.95):
    # 终止遮罩关闭 bootstrap；批次截止仍使用 next_values，且不重置环境。
    adv=[]; tail=0.
    for r,v,n,d in reversed(list(zip(rewards,values,next_values,terminals))):
        delta=r+(1-float(d))*n-v
        tail=delta+lam*(1-float(d))*tail
        adv.append(tail)
    adv=torch.tensor(list(reversed(adv)))
    return adv,adv+torch.tensor(values)

def fixed_offline_data():
    """512 个固定行为日志转移；与算法、训练 seed 无关。

    两种离线算法调用这个确定性采样函数，训练中再不与环境交互。
    行为策略以 .65 概率向右；这不是专家数据，也不保证充分覆盖。
    """
    rng=random.Random(2026)
    env=DeadlineChain(); x=env.reset(); data=[]
    for _ in range(512):
        a=int(rng.random()<.65)
        xp,r,d=env.step(a)
        data.append((x,torch.tensor(a),r,xp,d))
        x=env.reset() if d else xp
    return data
