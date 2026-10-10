"""Small cooperative environments and observation encodings; no learning updates."""
import itertools
import math
import random


def probabilities(logits):
    maximum = max(logits)
    weights = [math.exp(x-maximum) for x in logits]
    return [x/sum(weights) for x in weights]


def reward(bits, actions):
    correct = [a == b for a, b in zip(actions, bits)]
    return .2*correct[0]+.2*correct[1]+.6*all(correct)


def policy_return(theta):
    """Exact frozen stochastic return on all four equally likely contexts."""
    result = 0.
    for bits in itertools.product(range(2), repeat=2):
        pi = [probabilities(theta[i][bits[i]]) for i in range(2)]
        for actions in itertools.product(range(2), repeat=2):
            result += .25*pi[0][actions[0]]*pi[1][actions[1]]*reward(bits, actions)
    return result


def emit_row(rows, t, steps, value, emit, **diagnostics):
    if t and t != steps and t % max(1, steps//20):
        return
    row = dict(step=t, value=float(value), phase='stationary', **diagnostics)
    rows.append(row)
    if emit:
        emit(dict(row))


def seed_torch(seed):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    return random.Random(seed)


def local_features(bits, t):
    """Actor i gets only its own bit, shared clock, and agent identity."""
    import torch
    rows = []
    for i in range(2):
        rows.append([float(bits[i] == j) for j in range(2)] +
                    [float(t == j) for j in range(3)] +
                    [float(i == j) for j in range(2)])
    return torch.tensor(rows)


def global_features(bits, t):
    import torch
    index = 2*bits[0]+bits[1]
    return torch.tensor([float(index == j) for j in range(4)] +
                        [float(t == j) for j in range(3)])


def agent_network():
    import torch
    return torch.nn.Sequential(torch.nn.Linear(7,16), torch.nn.ReLU(),
                               torch.nn.Linear(16,2))


def greedy_return(net):
    """Deterministic evaluation is exact, read-only, and consumes no train RNG."""
    import torch
    result = 0.
    with torch.no_grad():
        for bits in itertools.product(range(2),repeat=2):
            result += .25*sum(.9**t*reward(bits,net(local_features(bits,t)).argmax(-1).tolist())
                             for t in range(3))
    return result


def batch(transitions):
    import torch
    fields = list(zip(*transitions))
    return (torch.stack(fields[0]), torch.stack(fields[1]), torch.tensor(fields[2]),
            torch.tensor(fields[3]), torch.stack(fields[4]), torch.stack(fields[5]),
            torch.tensor(fields[6],dtype=torch.float32))
