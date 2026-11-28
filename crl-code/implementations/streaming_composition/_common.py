"""Small environments and explicit tanh derivatives, not hidden training loops.

All algorithms use a flat parameter vector. This makes the distinction between
the current gradient and a stored eligibility vector directly inspectable.
No automatic-differentiation graph, replay buffer, or target network is used.
The update_sample_gradients diagnostic counts gradients used by learning, not
total Jacobian evaluations: forward/diagnostic calls may also compute derivatives.
"""
import math
import random


class Network:
    """v_j(x)=sum_h W2[j,h] tanh(W1[h]·x+b1[h])+b2[j]."""
    def __init__(self, inputs=2, hidden=6, outputs=1, seed=0):
        self.inputs, self.hidden, self.outputs = inputs, hidden, outputs
        self.trunk_size = hidden * (inputs + 1)
        rng = random.Random(seed)
        self.w = [rng.uniform(-.4, .4) for _ in range(self.trunk_size + outputs*(hidden+1))]

    def value_gradient(self, x, head=0, freeze_trunk=False):
        if len(x) != self.inputs or not 0 <= head < self.outputs:
            raise ValueError("Invalid network input/head")
        gradient = [0.] * len(self.w)
        base = self.trunk_size + head*(self.hidden+1)
        value = self.w[base+self.hidden]
        gradient[base+self.hidden] = 1.
        for h in range(self.hidden):
            start = h*(self.inputs+1)
            activation = math.tanh(sum(self.w[start+i]*x[i] for i in range(self.inputs)) + self.w[start+self.inputs])
            value += self.w[base+h]*activation
            gradient[base+h] = activation
            if not freeze_trunk:
                derivative = self.w[base+h]*(1-activation*activation)
                for i in range(self.inputs):
                    gradient[start+i] = derivative*x[i]
                gradient[start+self.inputs] = derivative
        return value, gradient

    def value(self, x, head=0):
        return self.value_gradient(x, head)[0]

    def add(self, direction, alpha):
        self.w = [w+alpha*d for w, d in zip(self.w, direction)]
        if not all(math.isfinite(w) for w in self.w):
            raise FloatingPointError("Non-finite parameter: fail rather than hide divergence")


def features(s, last=3):
    u = 2*s/last-1
    return [u, u*u]


def trace_step(old, gradient, gamma_in, lam, rho=1.):
    """Ordinary state-value IS convention: current rho multiplies BOTH terms."""
    return [rho*(gamma_in*lam*z+g) for z, g in zip(old, gradient)]


def norm(v):
    return math.sqrt(sum(x*x for x in v))


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def check(steps):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")


def record(rows, t, steps, value, emit, phase="training", **diagnostics):
    if t == 0 or t % max(1, steps//20) == 0 or t == steps:
        row = dict(step=t, value=float(value), phase=phase, **diagnostics)
        rows.append(row)
        if emit:
            emit(row.copy())


def main(meta, run):
    from implementations.runtime import run_cli
    run_cli(meta, run)


def gvf_question(head, state_next, action, sign=1.):
    """c, gamma_out, pi(a)/b(a); b(right)=.5, question pi=(.8,.2)."""
    probability_right = (.8, .2)[head]
    rho = 2*(probability_right if action else 1-probability_right)
    if head == 0:
        return float(state_next == 3), (0. if state_next == 3 else .9), rho
    return sign*(1. if state_next in (1, 3) else -1.), .8, rho


def gvf_truth(head, sign=1.):
    """Independent full-state Bellman iteration, NOT another learner's output."""
    value = [0.]*4
    p = (.8, .2)[head]
    for _ in range(350):
        new = []
        for s in range(4):
            answer = 0.
            for a, probability in ((0, 1-p), (1, p)):
                sn = (s+a) % 4
                c, gamma, _ = gvf_question(head, sn, a, sign)
                answer += probability*(c+gamma*value[sn])
            new.append(answer)
        value = new
    return value


def gvf_error(networks, separate=False, sign=1.):
    errors = []
    for h in range(2):
        net = networks[h] if separate else networks[0]
        truth = gvf_truth(h, sign)
        errors.append(math.sqrt(sum((net.value(features(s), 0 if separate else h)-truth[s])**2 for s in range(4))/4))
    return errors


def chain_error(net, success=.7):
    return math.sqrt(sum((net.value(features(s, 5))-success*.9**(5-s))**2 for s in range(6))/6)


def finite_lambda_targets(values, rewards, discounts, lam):
    """Frozen-parameter finite forward view; final value is an explicit tail."""
    out = [0.]*len(rewards)
    tail = values[-1]
    for t in range(len(rewards)-1, -1, -1):
        tail = rewards[t]+discounts[t]*((1-lam)*values[t+1]+lam*tail)
        out[t] = tail
    return out


def option_step(state, option):
    """Ring: option 0 = one left; option 1 = feedback policy reaching beacon 3.

    Beacon policy goes left at 0 or 3, right at 1 or 2. It terminates on arrival
    at 3. Thus duration is 1 or 2 and depends on the state, not an action repeat.
    The environment never terminates.
    """
    action = -1 if option == 0 or state in (0, 3) else 1
    sn = (state+action) % 4
    reward = -.05 + .4*float(sn == 3) + .12*float(sn == 0)
    return sn, reward, option == 0 or sn == 3


def option_target(rewards, next_value, gamma=.9, terminal=False, wrong_duration=False):
    if not rewards:
        raise ValueError("An option must execute at least one primitive step")
    accumulated = sum(gamma**k*r for k, r in enumerate(rewards))
    continuation = 0. if terminal else (gamma if wrong_duration else gamma**len(rewards))*next_value
    return accumulated+continuation


def option_model(state, option):
    rewards = []
    while True:
        state, r, ended = option_step(state, option)
        rewards.append(r)
        if ended:
            return state, rewards


def option_truth():
    q = [[0., 0.] for _ in range(4)]
    for _ in range(350):
        new = []
        for s in range(4):
            row = []
            for o in range(2):
                sn, rewards = option_model(s, o)
                row.append(option_target(rewards, max(q[sn])))
            new.append(row)
        q = new
    return q


def option_error(net, truth):
    return math.sqrt(sum((net.value(features(s), o)-truth[s][o])**2 for s in range(4) for o in range(2))/8)
