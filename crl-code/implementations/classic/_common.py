"""所有实验共用的任务、采样与测量；算法更新在各自文件中。
随机游走预测的真值为 s/6；控制任务是有终点的六格链。
这些是可检查的小型教学任务，不是大规模性能基准。
"""
import math
import random

BOOK = "http://incompleteideas.net/book/the-book-2nd.html"

def log(rows, step, value, steps, emit, **diagnostics):
    if step % max(1, math.ceil(steps/80)) and step != steps:
        return
    row = dict(step=step, value=float(value), phase="stationary", **diagnostics)
    rows.append(row)
    if emit:
        emit(row)

def check_steps(steps):
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps 必须为正整数")

def epsilon_prob(q, epsilon=.1):
    best = [i for i,v in enumerate(q) if v == max(q)]
    return [epsilon/len(q)+(1-epsilon)/len(best) if i in best
            else epsilon/len(q) for i in range(len(q))]

def sample(p, rng):
    u = rng.random()
    for i, mass in enumerate(p):
        u -= mass
        if u < 0:
            return i
    return len(p)-1

def action(q, rng):
    return sample(epsilon_prob(q), rng)

def softmax(h):
    weights = [math.exp(x-max(h)) for x in h]
    return [x/sum(weights) for x in weights]

def walk(s, rng):
    sp = s + (1 if rng.random() < .5 else -1)
    return (1. if sp == 6 else 0.), (None if sp in (0,6) else sp)

def rmse(v):
    return math.sqrt(sum((v[s]-s/6)**2 for s in range(1,6))/5)

def chain(s, a):
    sp = max(0, s-1) if a == 0 else s+1
    return (1., None) if sp == 5 else (-.01, sp)

def control_score(q):
    s, rewards, visited = 0, [], {}
    while s is not None:
        if s in visited:
            # 首次重复状态起形成确定性循环；几何级数补齐无限尾项。
            start = visited[s]
            prefix = sum(.95**k*r for k,r in enumerate(rewards[:start]))
            cycle = sum(.95**k*r for k,r in enumerate(rewards[start:]))
            return prefix+.95**start*cycle/(1-.95**(len(rewards)-start))
        visited[s] = len(rewards)
        # 确定性 tie break 是测量约定；训练时并列动作均匀采样。
        a = max(range(2), key=lambda a: q[s][a])
        r, s = chain(s,a)
        rewards.append(r)
    return sum(.95**k*r for k,r in enumerate(rewards))

def features(s):
    # 两个共享特征：不能为五个状态各设一个独立参数。
    return [0.,0.] if s is None else [1.,s/6.]

def dot(w,x):
    return sum(a*b for a,b in zip(w,x))

def linear_error(w):
    return rmse({s:dot(w,features(s)) for s in range(1,6)})

def model_target(s,a,v):
    r,sp = chain(s,a)
    return r + (.95*v[sp] if sp is not None else 0.)

def optimal_values():
    return {s:sum(-.01*.95**k for k in range(4-s))+.95**(4-s) for s in range(5)}
