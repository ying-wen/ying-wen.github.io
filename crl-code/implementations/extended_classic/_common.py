"""只共享环境、抽样、数值线性代数和测量；更新与训练循环在独立文件。"""
import math
import random

def check(steps):
    if type(steps) is not int or steps<1:
        raise ValueError("steps必须为正整数")

def log(rows,t,value,steps,emit,**diagnostics):
    if t % max(1,math.ceil(steps/80)) and t!=steps:
        return
    row=dict(step=t,value=float(value),phase="stationary",**diagnostics)
    rows.append(row)
    if emit:
        emit(row)

def dot(a,b):
    return sum(x*y for x,y in zip(a,b))

def solve(a,b):
    n=len(b)
    rows=[list(row)+[rhs] for row,rhs in zip(a,b)]
    for j in range(n):
        pivot=max(range(j,n),key=lambda k:abs(rows[k][j]))
        if abs(rows[pivot][j])<1e-12:
            raise ValueError("线性系统奇异")
        rows[j],rows[pivot]=rows[pivot],rows[j]
        scale=rows[j][j]
        rows[j]=[v/scale for v in rows[j]]
        for k in range(n):
            if k!=j:
                scale=rows[k][j]
                rows[k]=[v-scale*w for v,w in zip(rows[k],rows[j])]
    return [row[-1] for row in rows]

def walk(s,rng):
    sp=s+(1 if rng.random()<.5 else -1)
    return float(sp==6), None if sp in (0,6) else sp

def features(s):
    return [0.,0.] if s is None else [1.,s/6.]

def value_error(v):
    return math.sqrt(sum((v[s]-s/6.)**2 for s in range(1,6))/5)

def linear_error(w):
    return value_error({s:dot(w,features(s)) for s in range(1,6)})

def chain(s,a):
    sp=max(0,s-1) if a==0 else s+1
    return (1.,None) if sp==5 else (-.01,sp)

def probs(q,epsilon=.1):
    best=[a for a,v in enumerate(q) if v==max(q)]
    return [epsilon/len(q)+(1-epsilon)/len(best) if a in best else epsilon/len(q) for a in range(len(q))]

def sample(p,rng):
    u=rng.random()
    for a,v in enumerate(p):
        u-=v
        if u<0:
            return a
    return len(p)-1

def score(q):
    s,rewards,seen=0,[],{}
    while s is not None:
        if s in seen:
            start=seen[s]
            prefix=sum(.95**k*r for k,r in enumerate(rewards[:start]))
            cycle=sum(.95**k*r for k,r in enumerate(rewards[start:]))
            return prefix+.95**start*cycle/(1-.95**(len(rewards)-start))
        seen[s]=len(rewards)
        a=max(range(2),key=lambda a:q[s][a])
        r,s=chain(s,a)
        rewards.append(r)
    return sum(.95**k*r for k,r in enumerate(rewards))

def fixed_chain_values():
    matrix=[[float(i==j) for j in range(5)] for i in range(5)]
    rhs=[0.]*5
    for s in range(5):
        for a,p in enumerate([.2,.8]):
            r,sp=chain(s,a)
            rhs[s]+=p*r
            if sp is not None:
                matrix[s][sp]-=.95*p
    return solve(matrix,rhs)

def fixed_q_truth():
    v=fixed_chain_values()
    return {s:[r+(.95*v[sp] if sp is not None else 0.) for r,sp in [chain(s,0),chain(s,1)]] for s in range(5)}

def q_error(q):
    truth=fixed_q_truth()
    return math.sqrt(sum((q[s][a]-truth[s][a])**2 for s in q for a in range(2))/10)

def ope_episode(rng):
    actions=[rng.randrange(2) for _ in range(3)]
    rewards=[float(rng.random()<[.1,.9][a]) for a in actions]
    ratios=[[.2,.8][a]/.5 for a in actions]
    return actions,rewards,ratios

def sigmoid(x):
    return 1/(1+math.exp(-x)) if x>=0 else math.exp(x)/(1+math.exp(x))

