"""Seven original standard-library tabular RL labs, not benchmark reproductions.
Run: python3 tabular_textbook_lab.py demo (or a chapter name); test runs checks.
"""
import argparse
import heapq
import json
import math
import random
import unittest

# BEGIN bandits
def softmax(values):
    top = max(values)
    weights = [math.exp(v - top) for v in values]
    return [v / sum(weights) for v in weights]

def choose(probabilities, rng):
    u, total = rng.random(), 0.0
    for i, p in enumerate(probabilities):
        total += p
        if u < total:
            return i
    return len(probabilities) - 1

def epsilon_probs(values, epsilon):
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0,1]")
    best = max(range(len(values)), key=lambda i: values[i])
    probabilities = [epsilon / len(values)] * len(values)
    probabilities[best] += 1.0 - epsilon
    return probabilities

def ucb_action(values, counts, time):
    unseen = [a for a in range(len(values)) if counts[a] == 0]
    if unseen:
        return unseen[0]
    return max(range(len(values)), key=lambda a:
               values[a] + math.sqrt(2 * math.log(time) / counts[a]))

def bandit(method, steps=4000, seed=7):
    """Stationary Bernoulli arms. Gradient baseline uses the OLD mean."""
    if steps < 1:
        raise ValueError("steps must be positive")
    rng = random.Random(seed)
    means = [0.2, 0.5, 0.8]
    q, counts, preferences = [0.0]*3, [0]*3, [0.0]*3
    baseline, total = 0.0, 0.0
    for t in range(1, steps + 1):
        if method == "epsilon":
            action = choose(epsilon_probs(q, 0.1), rng)
        elif method == "ucb":
            action = ucb_action(q, counts, t)
        elif method == "gradient":
            probabilities = softmax(preferences)
            action = choose(probabilities, rng)
        else:
            raise ValueError(method)
        reward = float(rng.random() < means[action])
        total += reward
        counts[action] += 1
        q[action] += (reward - q[action]) / counts[action]
        if method == "gradient":
            for a in range(3):
                preferences[a] += 0.1 * (reward - baseline) * (
                    float(a == action) - probabilities[a])
        baseline += (reward - baseline) / t
    if method == "ucb":
        action = ucb_action(q, counts, steps+1)
        next_policy = [float(a == action) for a in range(3)]
    else:
        next_policy = (softmax(preferences) if method == "gradient"
                       else epsilon_probs(q, 0.1))
    return {"counts": counts, "mean_reward": total / steps, "q": q,
            "next_action_probabilities": next_policy}
# END bandits

# BEGIN mdps
# model[s][a]: (probability, reward, next_state) outcomes.
# None means TRUE terminal, not a budget truncation.
MODEL = {0: [[(1., 1., None)], [(1., 0., 1)]],
         1: [[(1., 2., None)], [(1., .5, 0)]]}
CONTROL_MODEL = {0: [[(1., .2, None)], [(1., 0., 1)]],
                 1: [[(1., 1., None)], [(1., -1., None)]]}

def returns(rewards, gamma, tail=0.0):
    result, value = [0.0]*len(rewards), tail
    for t in reversed(range(len(rewards))):
        value = rewards[t] + gamma * value
        result[t] = value
    return result

def action_values(model, state, values, gamma):
    return [sum(p * (r + gamma * (0 if sp is None else values[sp]))
                for p, r, sp in outcomes) for outcomes in model[state]]

def step(model, state, action, rng):
    outcomes = model[state][action]
    _, reward, successor = outcomes[choose([x[0] for x in outcomes], rng)]
    return reward, successor
# END mdps

# BEGIN dynamic_programming
def evaluate_policy(model, policy, gamma=0.9, tolerance=1e-12):
    if not 0 <= gamma < 1:
        raise ValueError("this discounted solver requires gamma < 1")
    values = {s: 0.0 for s in model}
    for _ in range(100000):
        new = {s: sum(p*q for p, q in zip(
            policy[s], action_values(model, s, values, gamma))) for s in model}
        difference = max(abs(new[s]-values[s]) for s in model)
        values = new
        if difference < tolerance:
            return values
    raise RuntimeError("evaluation did not converge")

def greedy_policy(model, values, gamma):
    policy = {}
    for s in model:
        q = action_values(model, s, values, gamma)
        best = max(range(len(q)), key=lambda a: q[a])
        policy[s] = [float(a == best) for a in range(len(q))]
    return policy

def policy_iteration(model=MODEL, gamma=0.9):
    policy = {s: [1.] + [0.]*(len(model[s])-1) for s in model}
    history = []
    while True:
        values = evaluate_policy(model, policy, gamma)
        history.append({"actions": [row.index(1.) for row in policy.values()],
                        "values": list(values.values())})
        improved = greedy_policy(model, values, gamma)
        if improved == policy:
            return values, policy, history
        policy = improved

def value_iteration(model=MODEL, gamma=0.9, tolerance=1e-12):
    if not 0 <= gamma < 1:
        raise ValueError("this discounted solver requires gamma < 1")
    values = {s: 0.0 for s in model}
    for sweep in range(100000):
        new = {s: max(action_values(model, s, values, gamma)) for s in model}
        difference = max(abs(new[s]-values[s]) for s in model)
        values = new
        if difference < tolerance:
            return values, greedy_policy(model, values, gamma), sweep+1
    raise RuntimeError("value iteration did not converge")
# END dynamic_programming

# BEGIN monte_carlo
def mc_visits(states, rewards, gamma=1.0, first_visit=True):
    if len(states) != len(rewards):
        raise ValueError("states exclude terminal")
    totals, counts, seen = {}, {}, set()
    for state, target in zip(states, returns(rewards, gamma)):
        if first_visit and state in seen:
            continue
        seen.add(state)
        counts[state] = counts.get(state, 0)+1
        totals[state] = totals.get(state, 0.)+target
    return {s: totals[s]/counts[s] for s in totals}

def importance_ratio(target_probabilities, behavior_probabilities):
    if len(target_probabilities) != len(behavior_probabilities):
        raise ValueError("probability sequences must align")
    weight = 1.
    for pi, b in zip(target_probabilities, behavior_probabilities):
        if b <= 0:
            raise ValueError("sampled action needs positive behavior probability")
        weight *= pi/b
    return weight

def is_estimates(samples):
    numerator = sum(g*w for g, w in samples)
    weight = sum(w for _, w in samples)
    return numerator/len(samples), (numerator/weight if weight else None)

def mc_control(episodes=6000, seed=7, epsilon=0.1):
    rng = random.Random(seed)
    q = {s: [0.]*len(CONTROL_MODEL[s]) for s in CONTROL_MODEL}
    counts = {s: [0]*len(q[s]) for s in q}
    for _ in range(episodes):
        trajectory, rewards, state = [], [], 0
        while state is not None:
            action = choose(epsilon_probs(q[state], epsilon), rng)
            reward, sp = step(CONTROL_MODEL, state, action, rng)
            trajectory.append((state, action))
            rewards.append(reward)
            state = sp
        seen = set()
        for (s, a), target in zip(trajectory, returns(rewards, .9)):
            if (s, a) not in seen:
                seen.add((s, a))
                counts[s][a] += 1
                q[s][a] += (target-q[s][a])/counts[s][a]
    return q
# END monte_carlo

# BEGIN temporal_difference
def td_prediction(episodes=1000, alpha=0.1):
    values = [0., 0.]  # 0 --r0--> 1 --r1--> terminal
    for _ in range(episodes):
        values[0] += alpha*(.9*values[1]-values[0])
        values[1] += alpha*(1.-values[1])
    return values

def control_target(kind, reward, next_values, probabilities=None,
                   next_action=None, gamma=0.9):
    if next_values is None:
        return reward
    if kind == "sarsa":
        tail = next_values[next_action]
    elif kind == "expected":
        tail = sum(p*v for p, v in zip(probabilities, next_values))
    elif kind == "q":
        tail = max(next_values)
    else:
        raise ValueError(kind)
    return reward+gamma*tail

def td_control(kind, episodes=6000, seed=7, alpha=0.1, epsilon=0.1):
    rng = random.Random(seed)
    q1 = {s: [0.]*len(CONTROL_MODEL[s]) for s in CONTROL_MODEL}
    q2 = {s: [0.]*len(q1[s]) for s in q1}
    for _ in range(episodes):
        state = 0
        action = choose(epsilon_probs([x+y for x, y in zip(q1[state], q2[state])],
                                      epsilon), rng)
        while state is not None:
            reward, sp = step(CONTROL_MODEL, state, action, rng)
            next_action, probabilities = None, None
            if sp is not None:
                probabilities = epsilon_probs([x+y for x, y in zip(q1[sp], q2[sp])],
                                              epsilon)
                next_action = choose(probabilities, rng)
            if kind == "double":
                update, other = (q1, q2) if rng.random()<.5 else (q2, q1)
                target = reward
                if sp is not None:
                    selected = max(range(len(update[sp])), key=lambda a: update[sp][a])
                    target += .9*other[sp][selected]
                update[state][action] += alpha*(target-update[state][action])
            else:
                target = control_target(kind, reward, None if sp is None else q1[sp],
                                        probabilities, next_action)
                q1[state][action] += alpha*(target-q1[state][action])
            # Keep SARSA's already-selected next action.
            state, action = sp, next_action
    return {s: [(x+y)/2 if kind=="double" else x for x, y in zip(q1[s], q2[s])]
            for s in q1}
# END temporal_difference

# BEGIN multistep
def nstep_episode(states, rewards, values, n, alpha=0.1, gamma=1.0):
    """Terminal trajectory, online update order and terminal flush.
    Full trajectory is supplied for tests; live code needs only an n+1 ring buffer.
    """
    if n<1 or len(states)!=len(rewards)+1 or states[-1] is not None:
        raise ValueError("need n>=1 and true-terminal trajectory")
    terminal, values = len(rewards), values.copy()
    for t in range(terminal+n-1):
        tau = t-n+1
        if tau<0:
            continue
        end = min(tau+n, terminal)
        target = sum(gamma**(k-tau)*rewards[k] for k in range(tau, end))
        if tau+n<terminal:
            target += gamma**n*values[states[tau+n]]
        s = states[tau]
        values[s] += alpha*(target-values[s])
    return values

def qsigma_target(states, actions, rewards, q, policy, sigma, gamma=0.9):
    """Frozen-Q, ON-POLICY finite-horizon Q(sigma) target.
    A nonterminal endpoint requires its sampled action. No general off-policy
    importance correction is implemented. sigma=0: tree backup; sigma=1: Sarsa.
    """
    if not 0<=sigma<=1 or len(states)!=len(rewards)+1:
        raise ValueError("invalid sigma or trajectory")
    horizon = len(rewards)
    g = 0. if states[-1] is None else q[states[-1]][actions[-1]]
    for k in reversed(range(horizon)):
        sp = states[k+1]
        if sp is None:
            if k!=horizon-1:
                raise ValueError("terminal must be last")
            g = rewards[k]
        else:
            ap = actions[k+1]
            expected = sum(p*v for p, v in zip(policy[sp], q[sp]))
            branch = sigma+(1.-sigma)*policy[sp][ap]
            g = rewards[k]+gamma*(branch*(g-q[sp][ap])
                +sigma*q[sp][ap]+(1.-sigma)*expected)
    return g

def random_walk(n, episodes=5000, seed=7):
    rng = random.Random(seed)
    values = {s: .5 for s in range(1,6)}
    for _ in range(episodes):
        state, states, rewards = 3, [3], []
        while state not in (0,6):
            sp = state+rng.choice([-1,1])
            rewards.append(float(sp==6))
            states.append(None if sp in (0,6) else sp)
            state = sp
        values = nstep_episode(states, rewards, values, n, alpha=.02)
    return {"values": values, "reference": {s:s/6 for s in values},
            "rmse": math.sqrt(sum((values[s]-s/6)**2 for s in values)/5)}
# END multistep

# BEGIN planning
def dyna(planning_steps=5, real_steps=4000, seed=7):
    """Deterministic chain: a last-outcome model is exact after observation."""
    rng = random.Random(seed)
    q = {s:[0.,0.] for s in range(4)}
    model, state, simulated, terminals = {}, 0, 0, 0
    def update(s, a, reward, sp):
        target = reward+(.9*max(q[sp]) if sp is not None else 0.)
        q[s][a] += .2*(target-q[s][a])
    for _ in range(real_steps):
        action = choose(epsilon_probs(q[state], .5), rng)
        position = max(0, state+(1 if action else -1))
        reward, sp = float(position==4), None if position==4 else position
        update(state, action, reward, sp)
        model[state,action] = reward,sp
        for _ in range(planning_steps):
            s,a = rng.choice(list(model))
            r,sn = model[s,a]
            update(s,a,r,sn)
            simulated += 1
        terminals += int(sp is None)
        state = 0 if sp is None else sp
    return {"right_values":[q[s][1] for s in q], "reference":[.729,.81,.9,1],
            "real_steps":real_steps,"model_updates":simulated,"episodes":terminals}

def prioritized_chain():
    """Exact one-action model: changed terminal reward launches a reverse wave."""
    model = {s:(float(s==3),None if s==3 else s+1) for s in range(4)}
    predecessors = {s:[] for s in model}
    for s,(_,sp) in model.items():
        if sp is not None:
            predecessors[sp].append(s)
    values,heap,sequence = {s:0. for s in model},[],[]
    def target(s):
        reward,sp = model[s]
        return reward+(.9*values[sp] if sp is not None else 0)
    for s in model:
        error = abs(target(s)-values[s])
        if error>1e-12:
            heapq.heappush(heap,(-error,s))
    while heap:
        _,s = heapq.heappop(heap)
        if abs(target(s)-values[s])<=1e-12:
            continue  # stale priority
        values[s] = target(s)
        sequence.append(s)
        for predecessor in predecessors[s]:
            error = abs(target(predecessor)-values[predecessor])
            if error>1e-12:
                heapq.heappush(heap,(-error,predecessor))
    return {"backup_order":sequence,"values":values}
# END planning

def demonstrations(mode):
    if mode=="bandits":
        return {k:bandit(k) for k in ("epsilon","ucb","gradient")}
    if mode=="mdps":
        return {"returns":returns([1,2,3],.5),"truncated_tail":returns([1],.9,10),
                "uniform_values":evaluate_policy(MODEL,{s:[.5,.5] for s in MODEL})}
    if mode=="dynamic-programming":
        _,_,history = policy_iteration()
        v,_,sweeps = value_iteration()
        return {"policy_iteration":history,"value_iteration":v,"sweeps":sweeps}
    if mode=="monte-carlo":
        rng,samples = random.Random(7),[]
        for _ in range(10000):
            a = rng.randrange(2)
            reward = float(rng.random()<[.2,.8][a])
            samples.append((reward,[.2,.8][a]/.5))
        return {"first":mc_visits([0,0],[1,2]),
                "every":mc_visits([0,0],[1,2],first_visit=False),
                "ordinary_weighted_IS":is_estimates(samples),"IS_reference":.68,
                "epsilon_soft_control":mc_control()}
    if mode=="temporal-difference":
        return {"prediction":td_prediction(),
                "control":{k:td_control(k) for k in ("sarsa","expected","q","double")}}
    if mode=="multistep":
        q,pi = {0:[0],1:[2,4]},{0:[1],1:[.25,.75]}
        return {"qsigma":{str(s):qsigma_target([0,1,None],[0,0],[1,3],q,pi,s)
                          for s in (0,.5,1)},
                "random_walk":{str(n):random_walk(n) for n in (1,3,8)}}
    if mode=="planning":
        return {"dyna":{str(n):dyna(n) for n in (0,5)},
                "prioritized":prioritized_chain()}
    raise ValueError(mode)

class Checks(unittest.TestCase):
    def test_softmax_shift_and_extreme(self):
        self.assertEqual(softmax([1000,1000]),[.5,.5])
        self.assertAlmostEqual(softmax([-1000,-999])[1],softmax([0,1])[1])
    def test_epsilon(self):
        self.assertEqual(epsilon_probs([2,1],.2),[.9,.1])
        with self.assertRaises(ValueError): epsilon_probs([1,2],2)
    def test_bandit_sampling(self):
        for k in ("epsilon","ucb","gradient"):
            r=bandit(k)
            self.assertEqual(sum(r["counts"]),4000)
            self.assertGreater(r["counts"][2],r["counts"][0])
    def test_ucb_initialization_and_report(self):
        self.assertEqual(ucb_action([999,0],[1,0],2),1)
        r=bandit("ucb",steps=2)
        self.assertEqual(r["next_action_probabilities"],[0,0,1])
        r=bandit("ucb")
        a=ucb_action(r["q"],r["counts"],4001)
        self.assertEqual(r["next_action_probabilities"],
                         [float(i==a) for i in range(3)])
    def test_bandit_budget(self):
        with self.assertRaises(ValueError): bandit("epsilon",steps=0)
    def test_returns(self):
        self.assertEqual(returns([1,2,3],.5),[2.75,3.5,3])
        self.assertEqual(returns([1],.9,10),[10])
    def test_terminal(self):
        self.assertEqual(action_values(MODEL,0,{0:999,1:10},.9),[1,9])
    def test_dp(self):
        v,p,h=policy_iteration()
        self.assertEqual([r["actions"] for r in h],[[0,0],[1,0],[1,1]])
        self.assertAlmostEqual(v[0],45/19,places=9)
        self.assertAlmostEqual(v[1],50/19,places=9)
        for l,r in zip(h,h[1:]):
            self.assertTrue(all(a<=b+1e-12 for a,b in zip(l["values"],r["values"])))
        other,p2,_=value_iteration()
        self.assertEqual(p,p2)
        self.assertAlmostEqual(other[0],v[0],places=9)
    def test_contraction(self):
        u,v={0:-3,1:8},{0:7,1:-2}
        d=max(abs(max(action_values(MODEL,s,u,.9))-max(action_values(MODEL,s,v,.9)))
              for s in MODEL)
        self.assertLessEqual(d,.9*max(abs(u[s]-v[s]) for s in u))
    def test_uniform_policy(self):
        v=evaluate_policy(MODEL,{s:[.5,.5] for s in MODEL})
        self.assertAlmostEqual(v[0],425/319,places=9)
        self.assertAlmostEqual(v[1],590/319,places=9)
    def test_gamma(self):
        with self.assertRaises(ValueError): value_iteration(gamma=1)
    def test_first_every(self):
        self.assertEqual(mc_visits([0,0],[1,2]),{0:3})
        self.assertEqual(mc_visits([0,0],[1,2],first_visit=False),{0:2.5})
    def test_is(self):
        ordinary,weighted=is_estimates([(1,.4),(3,1.6)])
        self.assertAlmostEqual(ordinary,2.6)
        self.assertAlmostEqual(weighted,2.6)
        self.assertEqual(is_estimates([(9,0)]),(0,None))
    def test_ratio(self):
        self.assertAlmostEqual(importance_ratio([.2,.8],[.5,.5]),.64)
        with self.assertRaises(ValueError): importance_ratio([1],[0])
    def test_mc_control(self):
        q=mc_control()
        self.assertAlmostEqual(q[0][0],.2)
        self.assertLess(abs(q[0][1]-.81),.06)
    def test_td_prediction(self):
        self.assertEqual(td_prediction(1),[0,.1])
        self.assertAlmostEqual(td_prediction()[0],.9)
    def test_targets(self):
        self.assertAlmostEqual(control_target("sarsa",0,[2,0],next_action=1),0)
        self.assertAlmostEqual(control_target("expected",0,[2,0],[.9,.1]),1.62)
        self.assertAlmostEqual(control_target("q",0,[2,0]),1.8)
        self.assertEqual(control_target("q",3,None),3)
    def test_control(self):
        self.assertAlmostEqual(td_control("q")[0][1],.9,places=6)
        self.assertAlmostEqual(td_control("double")[0][1],.9,places=6)
        self.assertAlmostEqual(td_control("expected")[0][1],.81,places=6)
        self.assertLess(abs(td_control("sarsa")[0][1]-.81),.2)
    def test_double_roles(self):
        q,other=[5,4],[1,3]
        a=max(range(2),key=lambda i:q[i])
        self.assertEqual(.9*other[a],.9)
        self.assertAlmostEqual(.9*max(other),2.7)
    def test_nstep_one(self):
        r=nstep_episode([0,1,None],[0,1],{0:.2,1:.4},1,.1,.9)
        self.assertAlmostEqual(r[0],.216)
        self.assertAlmostEqual(r[1],.46)
    def test_flush(self):
        for n in (2,8):
            self.assertEqual(nstep_episode([0,1,None],[0,1],{0:0,1:0},n,1,.9),
                             {0:.9,1:1})
    def test_nstep_validation(self):
        with self.assertRaises(ValueError): nstep_episode([0,None],[1],{0:0},0)
    def test_qsigma_endpoints(self):
        q,pi={0:[0],1:[2,4]},{0:[1],1:[.25,.75]}
        for s,ref in zip((0,.5,1),(4.375,4.0375,3.7)):
            self.assertAlmostEqual(qsigma_target([0,1,None],[0,0],[1,3],q,pi,s),ref)
    def test_qsigma_one_step(self):
        q,pi={0:[0],1:[2,4]},{0:[1],1:[.25,.75]}
        self.assertAlmostEqual(qsigma_target([0,1],[0,0],[1],q,pi,0),4.15)
        self.assertAlmostEqual(qsigma_target([0,1],[0,0],[1],q,pi,1),2.8)
    def test_qsigma_terminal(self):
        self.assertEqual(qsigma_target([0,None],[0],[7],{0:[0]},{0:[1]},.5),7)
    def test_deterministic_policy(self):
        q,pi={0:[0],1:[2,4]},{0:[1],1:[1,0]}
        self.assertAlmostEqual(qsigma_target([0,1,None],[0,0],[1,3],q,pi,0),
                               qsigma_target([0,1,None],[0,0],[1,3],q,pi,1))
    def test_walk(self):
        for n in (1,3,8): self.assertLess(random_walk(n)["rmse"],.15)
    def test_prioritized(self):
        r=prioritized_chain()
        self.assertEqual(r["backup_order"],[3,2,1,0])
        for a,b in zip(r["values"].values(),[.729,.81,.9,1]): self.assertAlmostEqual(a,b)
    def test_dyna(self):
        r=dyna()
        self.assertEqual(r["model_updates"],5*r["real_steps"])
        self.assertLess(max(abs(a-b) for a,b in zip(r["right_values"],r["reference"])),.01)
    def test_expected_sample(self):
        self.assertEqual(sum([1,1,1,5])/4,.75*1+.25*5)

MODES=["bandits","mdps","dynamic-programming","monte-carlo",
       "temporal-difference","multistep","planning"]
if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=MODES+["demo","test"])
    args=parser.parse_args()
    if args.mode=="test":
        unittest.main(argv=["tabular_textbook_lab.py"],verbosity=2)
    else:
        result={m:demonstrations(m) for m in MODES} if args.mode=="demo" else demonstrations(args.mode)
        print(json.dumps(result,ensure_ascii=False,indent=2))
