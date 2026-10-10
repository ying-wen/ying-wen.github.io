"""共用任务与测量；不包含任何学习算法更新。"""
import math
import random

def check_steps(steps):
    if not isinstance(steps,int) or steps < 2:
        raise ValueError("steps必须为至少2的整数，以定义两段任务")

def log(rows,t,value,steps,emit,phase=None,**diagnostics):
    if t % max(1,math.ceil(steps/80)) and t != steps:
        return
    row = dict(step=t,value=float(value),phase=phase or ("before_change" if t<=steps//2 else "after_change"),**diagnostics)
    rows.append(row)
    if emit:
        emit(row)

def dot(w,x):
    return sum(a*b for a,b in zip(w,x))

def eps_action(q,rng):
    if rng.random()<.1:
        return rng.randrange(len(q))
    best = [i for i,v in enumerate(q) if v == max(q)]
    return rng.choice(best)

def signal(rng,t,steps):
    x = [1.,rng.uniform(-1.,1.)]
    target_w = [.2,.7] if t<=steps//2 else [-.2,-.7]
    return x,dot(target_w,x)+rng.gauss(0,.03),target_w

def regression_error(w,target):
    # x_1恒1，x_2均匀[-1,1]；解析无噪声预测MSE。
    return (w[0]-target[0])**2+(w[1]-target[1])**2/3

def recurrent_sequence(rng,t,steps,length=8):
    xs=[rng.uniform(-1.,1.) for _ in range(length)]
    teacher=(.8,.4,0.) if t<=steps//2 else (.8,-.4,0.)
    h=0.
    for x in xs:
        h=math.tanh(teacher[0]*h+teacher[1]*x+teacher[2])
    return xs,h

def chain(s,a):
    sp = max(0,s-1) if a==0 else s+1
    return (1.,None) if sp==5 else (-.01,sp)

def chain_score(q):
    return discounted_chain_score(lambda s:max(range(2),key=lambda a:q[s][a]))

def discounted_chain_score(policy,goal_reward=1.,gamma=.95):
    # 冻结确定性策略：终点有限求和，重复状态无限循环精确求和。
    s,rewards,visited=0,[],{}
    while s is not None:
        if s in visited:
            start=visited[s]
            prefix=sum(gamma**k*r for k,r in enumerate(rewards[:start]))
            cycle=sum(gamma**k*r for k,r in enumerate(rewards[start:]))
            return prefix+gamma**start*cycle/(1-gamma**(len(rewards)-start))
        visited[s]=len(rewards)
        a=policy(s)
        r,s=chain(s,a)
        rewards.append(goal_reward if s is None else r)
    return sum(gamma**k*r for k,r in enumerate(rewards))
