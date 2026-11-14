"""只共享任务、网络、采样与测量，算法更新在独立文件中。"""
import math
import random
import torch
from torch import nn
torch.set_num_threads(1)

def setup(seed,steps):
    if not isinstance(steps,int) or steps<2: raise ValueError("steps >= 2 required")
    torch.manual_seed(seed)
    return random.Random(seed)

def grid(steps):
    stride=max(1,math.ceil(steps/20))
    return set([0,steps]+list(range(stride,steps,stride)))

def log(rows,t,value,emit=None,phase="evaluation",**info):
    row=dict(step=t,value=float(value),phase=phase,**info)
    rows.append(row)
    if emit: emit(row)

def mlp(n,m,hidden=16):
    return nn.Sequential(nn.Linear(n,hidden),nn.Tanh(),nn.Linear(hidden,m))

class Chain:
    """有限时域截止属于真实终止，剩余时间可见。"""
    def reset(self):
        self.s=0; self.left=12
        return self.obs()
    def obs(self):
        return torch.tensor([float(j==self.s) for j in range(5)]+[self.left/12])
    def step(self,a):
        self.s=min(4,max(0,self.s+(1 if a else -1))); self.left-=1
        done=self.s==4 or self.left==0
        return self.obs(),1. if self.s==4 else -.02,done

def chain_score(actor):
    env=Chain(); x=env.reset(); total=0.
    with torch.no_grad():
        while True:
            x,r,d=env.step(int(actor(x).argmax())); total+=r
            if d:return total

def sequence(rng):
    """延迟线索：首位±1，后7位干扰；最终目标为首位。"""
    cue=rng.choice([-1.,1.])
    return [cue]+[rng.uniform(-.2,.2) for _ in range(7)],cue

def sequence_score(predict):
    rng=random.Random(8181); losses=[]
    with torch.no_grad():
        for _ in range(32):
            xs,y=sequence(rng); losses.append((float(predict(xs))-y)**2)
    return sum(losses)/len(losses)

def hmm_stream(rng,state):
    state=state if rng.random()<.9 else 1-state
    observation=state if rng.random()<.8 else 1-state
    return state,observation

def regression_data(rng,changed):
    x=torch.tensor([rng.uniform(-1,1) for _ in range(4)])
    sign=-1. if changed else 1.
    y=sign*(x[0]+.5*x[1]-.3*x[2])+.2*x[3].square()
    return x,y

class ReLUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(4,16); self.output=nn.Linear(16,1)
    def forward(self,x):
        return self.output(self.hidden(x).relu()).squeeze(-1)

def regression_scores(net,changed):
    rng=random.Random(393); current=[]; old=[]
    with torch.no_grad():
        for _ in range(64):
            x,y=regression_data(rng,changed); _,oldy=regression_data(random.Random(0),False)
            original=x[0]+.5*x[1]-.3*x[2]+.2*x[3].square()
            prediction=net(x)
            current.append(float((prediction-y).square()))
            old.append(float((prediction-original).square()))
    return sum(current)/len(current),sum(old)/len(old)

def alternating_value_error(values,amplitude):
    truth=torch.tensor([amplitude/(1-.8**2),.8*amplitude/(1-.8**2)])
    return float((values-truth).square().mean())
def ac_terms(weights,x,a,reward,xp,terminal):
    """仅计算当前网络输出/梯度，更新规则仍由独立文件定义。"""
    value=weights[:6]@x
    logits=weights[6:].reshape(2,6)@x
    probs=logits.softmax(0)
    next_value=weights[:6]@xp
    delta=float(reward+(0. if terminal else 1.)*next_value-value)
    score=(torch.nn.functional.one_hot(torch.tensor(a),2).float()-probs)[:,None]*x
    gradient_u=torch.cat([x,.5*score.reshape(-1)])
    delta_gradient=torch.cat([(0. if terminal else 1.)*xp-x,torch.zeros(12)])
    return delta,gradient_u,delta_gradient

def ac_actor(weights):
    return lambda x: weights[6:].reshape(2,6)@x
