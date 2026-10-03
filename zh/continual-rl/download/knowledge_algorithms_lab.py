#!/usr/bin/env python3
"""Executable mechanisms for goals, options, models and planning.

Python >= 3.10; standard library only. Original teaching code, MIT.
These small exact/tabular experiments are not reproductions of paper benchmarks.
Run: python3 knowledge_algorithms_lab.py all | goals | options | models | planning | test
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import math
import random
import unittest


# BEGIN goals
def relabel_goal(next_position: int, goal: int, physical_terminal: bool = False):
    """Our goal task ENDS on success; another task may not use this convention."""
    success = next_position == goal
    reward = 0.0 if success else -1.0
    terminal = physical_terminal or success
    return reward, terminal


def goal_q_update(q, state, action, next_state, goal, alpha=0.5, gamma=0.9,
                  physical_terminal=False):
    reward, terminal = relabel_goal(next_state, goal, physical_terminal)
    next_value = 0.0 if terminal else max(q.get((next_state, a, goal), 0.0)
                                          for a in (-1, 1))
    key = (state, action, goal)
    old = q.get(key, 0.0)
    target = reward + gamma * next_value
    q[key] = old + alpha * (target - old)
    return reward, terminal, target, q[key]


def future_her(trajectory, original_goal, rng):
    """trajectory: (s,a,s_next,physical_terminal) in one recorded episode.

    Outputs (s,a,s_next,goal,physical_terminal), preserving physical termination
    for both original and hindsight goals. Goal success must never erase it.
    Dynamics are deterministic and goal independent in this teaching example.
    """
    replay = []
    for t, (s, a, sn, physical_terminal) in enumerate(trajectory):
        replay.append((s, a, sn, original_goal, physical_terminal))
        future_index = rng.randrange(t, len(trajectory))
        hindsight_goal = trajectory[future_index][2]
        replay.append((s, a, sn, hindsight_goal, physical_terminal))
    return replay


def subtask_td(value, state, next_state, reward, stopping_value, beta,
               alpha=0.1, gamma=0.9, rho=1.0):
    """STOMP Eq.(5) timing; beta is supplied, not optimized by this function."""
    old_here, old_next = value[state], value[next_state]
    target = reward + beta * stopping_value + gamma * (1.0 - beta) * old_next
    value[state] = old_here + alpha * rho * (target - old_here)
    return target


def stop_rules(stopping_value, continuation_value, gamma=0.9):
    paper_rule = stopping_value >= continuation_value  # STOMP Eq.(9)
    greedy_one_step = stopping_value >= gamma * continuation_value
    return paper_rule, greedy_one_step


def intermediate_goals(success_counts, attempt_counts, low=0.2, high=0.8):
    """Transparent finite curriculum; not the Goal GAN training algorithm."""
    return [g for g in success_counts if attempt_counts[g] > 0
            and low <= success_counts[g] / attempt_counts[g] <= high]
# END goals


# BEGIN options
def option_target(rewards, gamma, next_value, terminal=False):
    if not rewards:
        raise ValueError("An option must execute at least one primitive step")
    accumulated = sum(gamma**k * r for k, r in enumerate(rewards))
    return accumulated + (0.0 if terminal else gamma**len(rewards) * next_value)


def arrival_value(q_continue, v_select, beta, terminal=False):
    return 0.0 if terminal else (1.0 - beta) * q_continue + beta * v_select


def intra_option_update(q, state, next_state, reward, beta_next,
                        target_action_prob, behavior_action_prob,
                        alpha=0.1, gamma=0.9, terminal=False,
                        next_initiation_mask=None):
    """One observed action updates every fixed option with support.

    Dictionaries q[s] are vectors over options. q snapshots make targets
    simultaneous, including the case state == next_state. The initiation mask
    restricts NEW option selections at next_state, not continuation of an
    already active option. Omit it only if every option can initiate there.
    """
    if behavior_action_prob <= 0.0:
        raise ValueError("Observed action must have positive behavior probability")
    here, nxt = q[state][:], q[next_state][:]
    mask = [True] * len(nxt) if next_initiation_mask is None else next_initiation_mask
    if len(mask) != len(nxt):
        raise ValueError("Initiation mask must match the number of options")
    if not terminal and not any(mask):
        raise ValueError("A nonterminal selection state needs a legal option")
    value = 0.0 if terminal else max(v for v, legal in zip(nxt, mask) if legal)
    for o in range(len(here)):
        rho = target_action_prob[o] / behavior_action_prob
        continuation = arrival_value(nxt[o], value, beta_next[o], terminal)
        delta = reward + gamma * continuation - here[o]
        q[state][o] = here[o] + alpha * rho * delta
    return q[state]


def softmax(logits):
    maximum = max(logits)
    exps = [math.exp(x - maximum) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def option_critic_actor_step(logits, chosen_action, q_action, baseline,
                             termination_logit, q_continue, v_select,
                             alpha_actor=0.1, alpha_beta=0.1):
    """Local sampled ascent directions; caller supplies OLD critic estimates.

    At environment terminal the caller must skip the termination update.
    This is not a complete deep Option-Critic training system.
    """
    probs = softmax(logits)
    advantage_action = q_action - baseline
    new_logits = [h + alpha_actor * advantage_action *
                  ((1.0 if a == chosen_action else 0.0) - probs[a])
                  for a, h in enumerate(logits)]
    beta = sigmoid(termination_logit)
    advantage_option = q_continue - v_select
    new_termination = termination_logit - alpha_beta * beta * (1.0 - beta) * advantage_option
    return new_logits, new_termination


def skill_intrinsic_rewards(log_q_z_given_s, log_prior_z,
                            log_q_next_given_skill, log_mixture_next):
    """Mechanism only: no discriminator, SAC, density training or entropy term."""
    return (log_q_z_given_s - log_prior_z,
            log_q_next_given_skill - log_mixture_next)
# END options


# BEGIN models
def option_episode_target(rewards, gamma, endpoint, n_states):
    reward_target = sum(gamma**k * r for k, r in enumerate(rewards))
    endpoint_target = [0.0] * n_states
    endpoint_target[endpoint] = gamma**len(rewards)
    return reward_target, endpoint_target


def model_td_update(reward_model, endpoint_model, state, next_state, reward,
                    beta_next, alpha=0.1, gamma=0.9, rho=1.0, terminal=False):
    """Fixed option; n is a discounted endpoint distribution, not normalized.

    True environment termination forces stopping, independently of beta.
    The explicit terminal endpoint has V=0 during planning; its discounted
    probability mass is retained. A rollout cutoff is not true termination.
    All right-hand sides use pre-update values (self-loops included).
    """
    old_reward, next_reward = reward_model[state], reward_model[next_state]
    old_row, next_row = endpoint_model[state][:], endpoint_model[next_state][:]
    stop = 1.0 if terminal else beta_next
    r_target = reward + gamma * (1.0 - stop) * next_reward
    p_target = [gamma * (stop * float(j == next_state)
                        + (1.0 - stop) * next_row[j])
                for j in range(len(old_row))]
    reward_model[state] = old_reward + alpha * rho * (r_target - old_reward)
    endpoint_model[state] = [v + alpha * rho * (target - v)
                             for v, target in zip(old_row, p_target)]
    return r_target, p_target


def model_backup(reward, discounted_endpoints, values):
    return reward + sum(p * v for p, v in zip(discounted_endpoints, values))


def mixed_duration_model(outcomes, gamma, n_states):
    """outcomes: (probability, rewards, endpoint). Keep time/end correlation."""
    if not math.isclose(sum(p for p, _, _ in outcomes), 1.0):
        raise ValueError("Outcome probabilities must sum to one")
    r_bar, p_bar = 0.0, [0.0] * n_states
    for probability, rewards, endpoint in outcomes:
        r, p = option_episode_target(rewards, gamma, endpoint, n_states)
        r_bar += probability * r
        p_bar = [old + probability * new for old, new in zip(p_bar, p)]
    return r_bar, p_bar


def successor_feature_step(psi, state, action, features, next_state, next_action_probs,
                           alpha=0.1, gamma=0.9, terminal=False):
    """Action-conditioned SF for a fixed target policy, with known feature signal.

    The observed action conditions the prediction, so the next action is averaged
    under the target policy; no current-action importance ratio is needed here.
    psi[s][a] is a feature vector. Snapshot before mutation handles self-loops.
    """
    old = psi[state][action][:]
    next_vectors = [row[:] for row in psi[next_state]]
    next_features = [sum(pr * vec[j] for pr, vec in zip(next_action_probs, next_vectors))
                     for j in range(len(features))]
    target = [x + (0.0 if terminal else gamma * xn)
              for x, xn in zip(features, next_features)]
    psi[state][action] = [x + alpha * (y - x) for x, y in zip(old, target)]
    return target


def gpi_action(successor_feature_bank, state, reward_weights):
    """Greedy policy improvement over a bank of evaluated fixed policies."""
    action_scores = [max(sum(x*w for x,w in zip(psi[state][a], reward_weights))
                         for psi in successor_feature_bank)
                     for a in range(len(successor_feature_bank[0][state]))]
    return max(range(len(action_scores)), key=action_scores.__getitem__), action_scores


def learn_tiny_sf():
    bank = []
    for fixed_action in (0, 1):
        psi = [[[0.0, 0.0] for _ in range(2)] for _ in range(2)]
        probabilities = [float(a == fixed_action) for a in range(2)]
        for _ in range(400):
            for s in range(2):
                for a in range(2):
                    successor_feature_step(psi, s, a, [float(a==0), float(a==1)],
                                           a, probabilities, alpha=1.0)
        bank.append(psi)
    return bank
# END models


# BEGIN planning
def q_learning_update(q, s, a, reward, sn, terminal, alpha=1.0, gamma=0.9):
    target = reward + (0.0 if terminal else gamma * max(q[sn]))
    delta = target - q[s][a]
    q[s][a] += alpha * delta
    return delta


def dyna_q(seed=4, planning_steps=5, episodes=40):
    """Deterministic 5-state line, reward 1 on entry to terminal state 4.

    This small episodic mechanism test permits reset. It is not single-life CRL.
    Model key=(s,a), value=(reward,next_state,terminal); last observation is
    correct here only because the environment is deterministic and stationary.
    """
    rng = random.Random(seed)
    q = [[0.0, 0.0] for _ in range(5)]
    model = {}
    lengths = []
    for _ in range(episodes):
        s = 0
        for length in range(1, 1001):
            if rng.random() < 0.2:
                a = rng.randrange(2)
            else:
                best = max(q[s])
                a = rng.choice([i for i, val in enumerate(q[s]) if val == best])
            sn = max(0, min(4, s + (-1 if a == 0 else 1)))
            terminal = sn == 4
            reward = float(terminal)
            q_learning_update(q, s, a, reward, sn, terminal)
            model[s, a] = (reward, sn, terminal)
            for _ in range(planning_steps):
                ms, ma = rng.choice(list(model))
                mr, msn, mt = model[ms, ma]
                q_learning_update(q, ms, ma, mr, msn, mt)
            s = sn
            if terminal:
                lengths.append(length)
                break
        else:
            raise RuntimeError("Episode exceeded safety cap")
    return q, model, lengths


def prioritized_sweeping(q, model, seeds, max_backups=100, threshold=1e-10, gamma=0.9):
    """Deterministic tabular predecessor graph and lazy priority queue.

    Recompute residual when popped; stale entries are allowed but do not
    trigger a backup after their residual has fallen below threshold.
    """
    predecessors = {s: set() for s in range(len(q))}
    for (s, a), (_, sn, _) in model.items():
        predecessors[sn].add((s, a))
    heap = []
    counter = itertools.count()

    def enqueue(key):
        s, a = key
        r, sn, terminal = model[key]
        target = r + (0.0 if terminal else gamma * max(q[sn]))
        residual = abs(target - q[s][a])
        if residual > threshold:
            heapq.heappush(heap, (-residual, next(counter), key))

    for key in seeds:
        enqueue(key)
    updates = []
    while heap and len(updates) < max_backups:
        _, _, (s, a) = heapq.heappop(heap)
        r, sn, terminal = model[s, a]
        target = r + (0.0 if terminal else gamma * max(q[sn]))
        if abs(target - q[s][a]) <= threshold:
            continue
        q_learning_update(q, s, a, r, sn, terminal, gamma=gamma)
        updates.append((s, a, q[s][a]))
        for predecessor in predecessors[s]:
            enqueue(predecessor)
    return updates


def option_value_iteration(models, values=None, tolerance=1e-12, cap=10000):
    """models[s] contains (r_o, discounted endpoint weights) for legal options."""
    values = [0.0] * len(models) if values is None else values[:]
    for iteration in range(1, cap + 1):
        new_values = [max(model_backup(r, p, values) for r, p in actions)
                      for actions in models]
        if max(abs(a - b) for a, b in zip(values, new_values)) < tolerance:
            return new_values, iteration
        values = new_values
    raise RuntimeError("No convergence within cap; inspect model row masses")


def finite_horizon_mpc(position, goal, horizon=3):
    """Enumerate actions in a known deterministic model, execute first only."""
    best_score, best_sequence = -math.inf, None
    for sequence in itertools.product((-1, 0, 1), repeat=horizon):
        x, score = position, 0.0
        for action in sequence:
            x += action
            score -= (x - goal)**2 + 0.01 * action**2
        if score > best_score:
            best_score, best_sequence = score, sequence
    return best_sequence[0], best_sequence, best_score
# END planning


def demo_goals():
    q = {(1, 1, 4): -2.0, (1, 1, 2): -2.0,
         (2, -1, 4): -2.0, (2, 1, 4): -1.0}
    print("original (r,done,target,newQ):", goal_q_update(q, 1, 1, 2, 4))
    print("HER      (r,done,target,newQ):", goal_q_update(q, 1, 1, 2, 2))
    values = [2.0, 6.0]
    target = subtask_td(values, 0, 1, reward=1.0, stopping_value=4.0, beta=0.25)
    print("subtask target, updated v:", target, values[0])
    print("STOMP rule / one-step greedy at z=9.5,v=10:", stop_rules(9.5, 10.0))
    print("intermediate goals:", intermediate_goals({"easy":9,"learnable":5,"hard":0},
                                                     {"easy":10,"learnable":10,"hard":10}))


def demo_options():
    print("two-step SMDP target:", option_target([1.0, 2.0], 0.9, 10.0))
    print("arrival value:", arrival_value(2.0, 6.0, 0.25))
    q = [[0.0, 0.0], [2.0, 6.0]]
    print("two compatible options, updated Q:",
          intra_option_update(q, 0, 1, 1.0, [0.25, 1.0], [1.0, 0.5], 0.5))
    logits, termination = option_critic_actor_step([0.0, 0.0], 0, 3.0, 1.0,
                                                   0.0, 2.0, 6.0)
    print("policy logits, termination logit:", logits, termination)
    print("probability of action 0 / termination:", softmax(logits)[0], sigmoid(termination))


def demo_models():
    reward_model = [0.0, 0.0, 0.0]
    endpoints = [[0.0] * 3 for _ in range(3)]
    model_td_update(reward_model, endpoints, 1, 2, 2.0, 1.0, alpha=1.0)
    model_td_update(reward_model, endpoints, 0, 1, 1.0, 0.0, alpha=1.0)
    print("two-step reward and discounted endpoints:", reward_model[0], endpoints[0])
    print("backup with endpoint value 10:", model_backup(reward_model[0], endpoints[0], [0,0,10]))
    r, p = mixed_duration_model([(0.5, [0.0], 0), (0.5, [0.0]*3, 1)], 0.9, 2)
    print("joint time/endpoint weights and backup:", p, model_backup(r, p, [10,0]))
    print("nonlinear counterexample V(E[X])=0, E[V(X)]=1 for X=+-1, V=x^2")
    print("SF/GPI action and scores, reward weights=(1,2):", gpi_action(learn_tiny_sf(),0,[1,2]))


def demo_planning():
    q, _, lengths = dyna_q()
    print("Dyna greedy values:", [round(max(row), 6) for row in q])
    print("Dyna final episode length:", lengths[-1])
    chain = {(0,0):(0.0,1,False), (1,0):(0.0,2,False), (2,0):(1.0,3,True)}
    qp = [[0.0] for _ in range(4)]
    print("prioritized backward wave:", prioritized_sweeping(qp, chain, [(2,0)]))
    values, iterations = option_value_iteration([[(2.8, [0.81])]])
    print("option VI value / analytic:", values[0], 2.8/(1-0.81), "iterations:", iterations)
    print("MPC first action / proposed sequence / score:", finite_horizon_mpc(0,2))


class MechanismTests(unittest.TestCase):
    def test_her_recomputes_reward_and_terminal(self):
        self.assertEqual(relabel_goal(2,4),(-1.0,False))
        self.assertEqual(relabel_goal(2,2),(0.0,True))
        self.assertEqual(relabel_goal(2,4,True),(-1.0,True))

    def test_goal_targets(self):
        q={(1,1,4):-2.0,(2,1,4):-1.0,(2,-1,4):-2.0,(1,1,2):-2.0}
        self.assertAlmostEqual(goal_q_update(q,1,1,2,4)[3],-1.95)
        self.assertAlmostEqual(goal_q_update(q,1,1,2,2)[3],-1.0)

    def test_future_her_uses_reached_future(self):
        t=[(0,1,1,False),(1,1,2,False),(2,1,3,True)]
        replay=future_her(t,4,random.Random(1))
        self.assertEqual(len(replay),6)
        for index in range(3):
            self.assertIn(replay[2*index+1][3],[x[2] for x in t[index:]])
            self.assertEqual(replay[2*index][4],t[index][3])
            self.assertEqual(replay[2*index+1][4],t[index][3])

    def test_future_her_physical_terminal_end_to_end(self):
        replay=future_her([(1,1,2,True)],4,random.Random(1))
        q={(2,-1,4):100.0,(2,1,4):100.0,
           (2,-1,2):100.0,(2,1,2):100.0}
        results=[]
        for s,a,sn,g,physical_terminal in replay:
            results.append(goal_q_update(q,s,a,sn,g,alpha=1.0,
                                         physical_terminal=physical_terminal))
        self.assertEqual(results[0],(-1.0,True,-1.0,-1.0))
        self.assertEqual(results[1],(0.0,True,0.0,0.0))

    def test_subtask_convention(self):
        v=[2.0,6.0]
        self.assertAlmostEqual(subtask_td(v,0,1,1,4,0.25),6.05)
        self.assertAlmostEqual(v[0],2.405)
        self.assertAlmostEqual(subtask_td([0,6],0,1,1,4,1),5)
        self.assertAlmostEqual(subtask_td([0,6],0,1,1,4,0),6.4)

    def test_stop_conventions_are_not_identical(self):
        self.assertEqual(stop_rules(9.5,10.0),(False,True))
        # Two-step STOMP terminal contribution is gamma^(K-1)*z,
        # whereas an option model multiplies a continuation value by gamma^K.
        self.assertNotEqual(0.9**1*10,0.9**2*10)

    def test_curriculum_intermediate_only(self):
        self.assertEqual(intermediate_goals({0:1,1:5,2:9},{0:10,1:10,2:10}),[1])

    def test_smdp_discount_and_terminal(self):
        self.assertAlmostEqual(option_target([1,2],0.9,10),10.9)
        self.assertAlmostEqual(option_target([1,2],0.9,10,True),2.8)
        self.assertAlmostEqual(option_target([1],0.9,10),10)
        with self.assertRaises(ValueError):
            option_target([],0.9,10)

    def test_arrival_endpoints(self):
        self.assertEqual(arrival_value(2,6,0),2)
        self.assertEqual(arrival_value(2,6,1),6)
        self.assertEqual(arrival_value(2,6,0.25),3)
        self.assertEqual(arrival_value(2,6,0.25,True),0)

    def test_intra_option_support_and_old_snapshot(self):
        q=[[2.0,6.0]]
        result=intra_option_update(q,0,0,1,[0.25,1],[1,0],0.5)
        self.assertAlmostEqual(result[0],2.34)
        self.assertEqual(result[1],6)

    def test_intra_option_reselection_obeys_initiation_set(self):
        q=[[0.0,0.0],[1.0,100.0]]
        result=intra_option_update(q,0,1,0,[1,1],[1,1],1,
                                   alpha=1,next_initiation_mask=[True,False])
        self.assertAlmostEqual(result[0],.9)
        self.assertAlmostEqual(result[1],.9)

    def test_intra_option_continuation_ignores_initiation_set(self):
        q=[[0.0,0.0],[1.0,100.0]]
        result=intra_option_update(q,0,1,0,[1,0],[1,1],1,
                                   alpha=1,next_initiation_mask=[True,False])
        self.assertAlmostEqual(result[0],.9)
        self.assertAlmostEqual(result[1],90.0)

    def test_intra_option_terminal_needs_no_legal_successor(self):
        q=[[0.0,0.0],[1.0,100.0]]
        result=intra_option_update(q,0,1,2,[1,1],[1,1],1,alpha=1,
                                   terminal=True,next_initiation_mask=[False,False])
        self.assertEqual(result,[2,2])

    def test_sigmoid_extreme_logits(self):
        self.assertEqual(sigmoid(-1000.0),0.0)
        self.assertEqual(sigmoid(1000.0),1.0)
        self.assertEqual(sigmoid(0.0),.5)

    def test_actor_and_termination_finite_difference(self):
        logits=[0.2,-0.1]
        h=0.3; eps=1e-6
        new_logits,new_h=option_critic_actor_step(logits,0,3,1,h,2,6,1,1)
        objective=lambda x: math.log(softmax(x)[0])*2
        for i in range(2):
            plus=logits[:]; minus=logits[:]
            plus[i]+=eps; minus[i]-=eps
            numeric=(objective(plus)-objective(minus))/(2*eps)
            self.assertAlmostEqual(new_logits[i]-logits[i],numeric,places=7)
        arrival=lambda z: arrival_value(2,6,sigmoid(z))
        numeric=(arrival(h+eps)-arrival(h-eps))/(2*eps)
        self.assertAlmostEqual(new_h-h,numeric,places=7)

    def test_skill_rewards(self):
        result=skill_intrinsic_rewards(math.log(.8),math.log(.5),math.log(.4),math.log(.2))
        self.assertAlmostEqual(result[0],math.log(1.6))
        self.assertAlmostEqual(result[1],math.log(2))

    def test_full_termination_gradient_uses_discounted_arrivals(self):
        # One continuing state, two options, fixed high-level policy (1/2,1/2).
        # Option rewards are 0 and 1. Only option 0's stop logit is varied.
        gamma, beta1, logit = .9, .3, .2
        def solve2(a, b, c, d, x, y):
            det = a*d-b*c
            return ((d*x-b*y)/det, (a*y-c*x)/det)
        def values(h):
            beta0 = sigmoid(h)
            # I - gamma * option transition matrix under call-and-return.
            a, b = 1-gamma*(1-beta0/2), -gamma*beta0/2
            c, d = -gamma*beta1/2, 1-gamma*(1-beta1/2)
            q0, q1 = solve2(a,b,c,d,0,1)
            # Discounted occupancy before action, starting with option 0.
            occupancy0, _ = solve2(a,c,b,d,1,0)
            return q0,q1,occupancy0
        q0,q1,occupancy0 = values(logit)
        beta0 = sigmoid(logit)
        advantage0 = q0-(q0+q1)/2
        gradient = -gamma*occupancy0*beta0*(1-beta0)*advantage0
        eps = 1e-5
        numeric = (values(logit+eps)[0]-values(logit-eps)[0])/(2*eps)
        self.assertAlmostEqual(gradient,numeric,places=7)
        self.assertGreater(abs(gradient/gamma-numeric),1e-3)

    def test_episode_model(self):
        r,p=option_episode_target([1,2],.9,2,3)
        self.assertAlmostEqual(r,2.8)
        self.assertEqual(p[:2],[0,0])
        self.assertAlmostEqual(p[2],.81)

    def test_model_bellman_recursion(self):
        r=[0.0]*3;p=[[0.0]*3 for _ in range(3)]
        model_td_update(r,p,1,2,2,1,alpha=1)
        model_td_update(r,p,0,1,1,0,alpha=1)
        self.assertAlmostEqual(r[0],2.8)
        self.assertAlmostEqual(p[0][2],.81)
        self.assertAlmostEqual(model_backup(r[0],p[0],[0,0,10]),10.9)

    def test_model_self_loop_uses_old_values(self):
        r=[1.0];p=[[0.5]]
        model_td_update(r,p,0,0,2,.2,alpha=.5)
        self.assertAlmostEqual(r[0],1+.5*(2+.9*.8-1))
        self.assertAlmostEqual(p[0][0],.5+.5*(.9*(.2+.8*.5)-.5))

    def test_model_true_terminal_overrides_option_beta(self):
        r=[0.0,5.0];p=[[0.0,0.0],[3.0,4.0]]
        target,end=model_td_update(r,p,0,1,1.0,0.0,alpha=1.0,terminal=True)
        self.assertEqual(target,1.0)
        self.assertEqual(end,[0.0,.9])
        self.assertEqual(r[0],1.0)
        self.assertEqual(model_backup(r[0],p[0],[10.0,0.0]),1.0)

    def test_model_rollout_cutoff_keeps_continuation(self):
        r=[0.0,5.0];p=[[0.0,0.0],[0.0,.5]]
        target,end=model_td_update(r,p,0,1,1.0,0.0,alpha=1.0,terminal=False)
        self.assertEqual(target,5.5)
        self.assertEqual(end,[0.0,.45])

    def test_random_duration_joint(self):
        r,p=mixed_duration_model([(.5,[0],0),(.5,[0,0,0],1)],.9,2)
        self.assertAlmostEqual(sum(p),.8145)
        self.assertAlmostEqual(model_backup(r,p,[10,0]),4.5)
        self.assertNotAlmostEqual(sum(p)*5,4.5)

    def test_linear_expectation_identity(self):
        weights=[2,-3];outcomes=[(.25,[1,2]),(.75,[-1,4])]
        dot=lambda a,b:sum(x*y for x,y in zip(a,b))
        mean=[sum(pr*x[j] for pr,x in outcomes) for j in range(2)]
        self.assertAlmostEqual(dot(weights,mean),sum(pr*dot(weights,x) for pr,x in outcomes))
        self.assertNotEqual(((1+(-1))/2)**2,(1**2+(-1)**2)/2)

    def test_successor_features_and_gpi(self):
        bank=learn_tiny_sf()
        self.assertAlmostEqual(bank[0][0][1][0],9.0)
        self.assertAlmostEqual(bank[0][0][1][1],1.0)
        action,scores=gpi_action(bank,0,[1,2])
        self.assertEqual(action,1)
        self.assertAlmostEqual(scores[0],19.0)
        self.assertAlmostEqual(scores[1],20.0)
        self.assertEqual(gpi_action(bank,0,[2,1])[0],0)

    def test_successor_features_terminal(self):
        psi=[[[3.0,4.0]]]
        target=successor_feature_step(psi,0,0,[1,2],0,[1],alpha=1,terminal=True)
        self.assertEqual(target,[1,2])
        self.assertEqual(psi[0][0],[1,2])

    def test_dyna_matches_optimal_chain(self):
        q,_,_=dyna_q()
        for s in range(4):
            self.assertAlmostEqual(max(q[s]),.9**(3-s))

    def test_prioritized_wave(self):
        model={(0,0):(0,1,False),(1,0):(0,2,False),(2,0):(1,3,True)}
        q=[[0.0] for _ in range(4)]
        wave=prioritized_sweeping(q,model,[(2,0)])
        self.assertEqual([s for s,a,v in wave],[2,1,0])
        self.assertAlmostEqual(q[0][0],.81)

    def test_priority_budget(self):
        model={(0,0):(0,1,False),(1,0):(1,2,True)}
        q=[[0.0] for _ in range(3)]
        self.assertEqual(len(prioritized_sweeping(q,model,[(1,0)],max_backups=1)),1)
        self.assertEqual(q[0][0],0)

    def test_option_vi(self):
        values,_=option_value_iteration([[(2.8,[.81])]])
        self.assertAlmostEqual(values[0],2.8/(1-.81),places=9)

    def test_planning_reward_error_bound(self):
        true_value=2.8/(1-.81)
        wrong_value=2.9/(1-.81)
        self.assertAlmostEqual(wrong_value-true_value,.1/(1-.81))

    def test_mpc_returns_first_action_not_full_open_loop(self):
        action,sequence,_=finite_horizon_mpc(0,2)
        self.assertEqual(action,1)
        self.assertEqual(sequence,(1,1,0))
        self.assertEqual(finite_horizon_mpc(2,2)[0],0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lesson",choices=["goals","options","models","planning","all","test"])
    args=parser.parse_args()
    if args.lesson=="test":
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(MechanismTests)
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(not result.wasSuccessful())
    demos={"goals":demo_goals,"options":demo_options,"models":demo_models,"planning":demo_planning}
    for name,demo in demos.items():
        if args.lesson in (name,"all"):
            print(f"\n[{name}]")
            demo()


if __name__=="__main__":
    main()
