"""扩展机制共享数值/日志工具；不隐藏任何学习算法。"""
import math
import random

def check(steps):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps须为正整数")

def sigmoid(x):
    return 1 / (1 + math.exp(-max(-60., min(60., x))))

def softmax(xs):
    e = [math.exp(x-max(xs)) for x in xs]
    return [v/sum(e) for v in e]

def sample(p, rng):
    u = rng.random()
    for i, v in enumerate(p):
        u -= v
        if u <= 0:
            return i
    return len(p)-1

def greedy(q, rng=None, epsilon=0.):
    if rng is not None and rng.random() < epsilon:
        return rng.randrange(len(q))
    best = max(q)
    ids = [i for i, v in enumerate(q) if v == best]
    return rng.choice(ids) if rng is not None else ids[0]

def record(rows, t, steps, value, emit, phase="mechanism-training", **diagnostics):
    if t == 1 or t % 20 == 0 or t == steps:
        row = dict(step=t, value=float(value), phase=phase, **diagnostics)
        rows.append(row)
        if emit:
            emit(row.copy())

def walk(s, a, n=7):
    return max(0, min(n-1, s+(-1 if a == 0 else 1)))

def main(meta, run):
    from implementations.runtime import run_cli
    run_cli(meta, run)

class MLP:
    """单隐层tanh网络；输出线性，全部参数可训练或冻结，显式反向传播。"""
    def __init__(self, inputs, hidden, outputs, rng):
        self.w = [[rng.uniform(-.6,.6) for _ in range(inputs)] for _ in range(hidden)]
        self.b = [0.] * hidden
        self.v = [[rng.uniform(-.6,.6) for _ in range(hidden)] for _ in range(outputs)]
        self.c = [0.] * outputs
    def forward(self, x):
        h = [math.tanh(sum(a*b for a,b in zip(w,x))+b) for w,b in zip(self.w,self.b)]
        return [sum(a*b for a,b in zip(v,h))+c for v,c in zip(self.v,self.c)]
    def gradients(self, x, output_grad):
        h = [math.tanh(sum(a*b for a,b in zip(w,x))+b) for w,b in zip(self.w,self.b)]
        dh = [(1-h[j]**2)*sum(output_grad[k]*self.v[k][j] for k in range(len(self.v))) for j in range(len(h))]
        return ([[d*z for z in x] for d in dh], dh,
                [[d*z for z in h] for d in output_grad], output_grad[:])
    def step(self, x, output_grad, alpha):
        dw,db,dv,dc = self.gradients(x,output_grad)
        for i in range(len(self.w)):
            self.w[i] = [a-alpha*d for a,d in zip(self.w[i],dw[i])]
            self.b[i] -= alpha*db[i]
        for i in range(len(self.v)):
            self.v[i] = [a-alpha*d for a,d in zip(self.v[i],dv[i])]
            self.c[i] -= alpha*dc[i]
