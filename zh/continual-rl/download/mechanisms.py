#!/usr/bin/env python3
"""CRL mechanism labs. Python 3.10+, standard library only; no network/files/GPU.

Original teaching implementations, not copies of upstream experiment code.
Code: MIT, Copyright (c) 2026 Ying Wen. See https://opensource.org/license/mit.
Run: python3 mechanisms.py all     or: python3 mechanisms.py test
Subcommands: gvf, state, average, options, models, planning, meta, goals,
streaming, retention, plasticity, exploration, architectures
Small analytic tasks check mechanisms, not published benchmark performance.
"""
import argparse
import json
import math
import random
import unittest


def dot(x, y):
    return sum(a * b for a, b in zip(x, y))


# BEGIN gvf
def gvf_td(w, x, xp, cumulant, gamma_next, rho, alpha):
    """Linear IS semi-gradient TD(0); target parameters are held fixed."""
    delta = cumulant + gamma_next * dot(w, xp) - dot(w, x)
    return [wi + alpha * rho * delta * xi for wi, xi in zip(w, x)]


def gtd2(w, h, x, xp, cumulant, gamma_next, rho, alpha, beta):
    """GTD2(0), NOT TDC/GTD(lambda). All RHS use OLD w and h."""
    delta = cumulant + gamma_next * dot(w, xp) - dot(w, x)
    projection = dot(x, h)
    wn = [wi + alpha * rho * (xi - gamma_next * xpi) * projection
          for wi, xi, xpi in zip(w, x, xp)]
    hn = [hi + beta * (rho * delta - projection) * xi
          for hi, xi in zip(h, x)]
    return wn, hn
# END gvf


# BEGIN state
def belief_step(belief, transition, likelihood):
    """Known-model Bayes filter. transition[old][new], evidence already observed."""
    prior = [sum(belief[i] * transition[i][j] for i in range(len(belief)))
             for j in range(len(belief))]
    raw = [p * l for p, l in zip(prior, likelihood)]
    total = sum(raw)
    if total <= 0:
        raise ValueError('Observation has zero likelihood under this model')
    return [r / total for r in raw]


def rtrl_step(state, jacobian, observation, theta):
    """Scalar tanh RNN, theta=(recurrent weight, input weight, bias).

    Exact derivative along this sequence when theta is fixed. If theta is
    changed at every step without recomputing history, this becomes an online
    sensitivity approximation, not the exact gradient of a replayed trajectory.
    """
    recurrent, input_weight, bias = theta
    sn = math.tanh(recurrent * state + input_weight * observation + bias)
    local = [state, observation, 1.0]
    jn = [(1 - sn * sn) * (d + recurrent * j)
          for d, j in zip(local, jacobian)]
    return sn, jn
# END state


# BEGIN average
def differential_td(values, rate, s, reward, sp, alpha, eta):
    delta = reward - rate + values[sp] - values[s]
    vn = values.copy()
    vn[s] += alpha * delta
    return vn, rate + eta * alpha * delta


def differential_q(q, rate, s, a, reward, sp, alpha, eta):
    delta = reward - rate + max(q[sp]) - q[s][a]
    qn = [row.copy() for row in q]
    qn[s][a] += alpha * delta
    return qn, rate + eta * alpha * delta


def average_option_step(q, rate, length, reward_sum, duration, next_max,
                        alpha, eta, beta):
    """Wan et al. 2021 eqs 6-9: OLD expected length in BOTH denominators."""
    if length <= 0:
        raise ValueError('Expected option duration must be positive')
    delta = reward_sum - rate * length + next_max - q
    return (q + alpha * delta / length,
            rate + eta * alpha * delta / length,
            length + beta * (duration - length))
# END average


# BEGIN options
def option_target(reward, gamma, beta_next, q_continue, v_switch):
    # beta_next is option termination, NOT environment termination.
    return reward + gamma * ((1 - beta_next) * q_continue + beta_next * v_switch)


def termination_logit_step(logit, advantage, alpha):
    beta = 1 / (1 + math.exp(-logit))
    return logit - alpha * beta * (1 - beta) * advantage


def smdp_target(rewards, gamma, next_value):
    return sum(gamma ** k * r for k, r in enumerate(rewards)) + gamma ** len(rewards) * next_value
# END options


# BEGIN models
def option_model_sweep(reward, transition, beta, gamma, r_model, p_model):
    """Synchronous expectation backup for one fixed option policy.

    reward[s] = E[R|s,o], transition = P_pi_o, beta on ARRIVAL state.
    p_model includes discount gamma^duration; rows need not sum to one.
    The toy has no environment terminal states.
    """
    n = len(reward)
    rn, pn = [], []
    for s in range(n):
        rn.append(reward[s] + gamma * sum(transition[s][j] * (1-beta[j]) * r_model[j]
                                        for j in range(n)))
        pn.append([gamma * sum(transition[s][j] *
                    (beta[j] * float(j == k) + (1-beta[j]) * p_model[j][k])
                    for j in range(n)) for k in range(n)])
    return rn, pn
# END models


# BEGIN planning
def model_plan(rewards, discounted_endpoints, sweeps=100):
    """V(s) <- max_o [r_o(s) + sum_j p_o(s,j)V(j)] for fixed models.

    rewards[s][o]; discounted_endpoints[s][o][next_state].
    Synchronous backups, no additional gamma. Planner does not fit the models.
    """
    values = [0.0] * len(rewards)
    for _ in range(sweeps):
        values = [max(r + dot(p, values) for r, p in zip(rs, ps))
                  for rs, ps in zip(rewards, discounted_endpoints)]
    return values
# END planning


# BEGIN meta
def idbd_step(w, log_alpha, trace, x, target, meta_rate):
    """Linear supervised IDBD, diagonal approximation, positive trace factor.

    Order: old error/trace -> new log steps -> new weights/trace.
    This is NOT TIDBD or actor-critic Metatrace.
    """
    error = target - dot(w, x)
    bn = [b + meta_rate * error * xi * hi for b, xi, hi in zip(log_alpha, x, trace)]
    alphas = [math.exp(b) for b in bn]
    wn = [wi + a * error * xi for wi, a, xi in zip(w, alphas, x)]
    hn = [hi * max(0.0, 1-a*xi*xi) + a * error * xi
          for hi, a, xi in zip(trace, alphas, x)]
    return wn, bn, hn


def scalar_meta_gradient(log_alpha, train_targets, validation_target):
    """Exact forward derivative of a FIXED log step size through SGD updates."""
    alpha, w, sensitivity = math.exp(log_alpha), 0.0, 0.0
    for target in train_targets:
        error = target - w
        sensitivity = (1-alpha) * sensitivity + alpha * error
        w += alpha * error
    loss = 0.5 * (w-validation_target)**2
    return loss, (w-validation_target) * sensitivity
# END meta


# BEGIN goals
def goal_reward(achieved, goal):
    return 0.0 if achieved == goal else -1.0


def subtask_target(reward, gamma, beta_next, terminal_bonus, next_subtask_value):
    """STOMP convention: immediate stopping bonus is NOT multiplied by gamma."""
    return reward + beta_next * terminal_bonus + gamma * (1-beta_next) * next_subtask_value
# END goals


# BEGIN streaming
def td_trace_step(w, e, x, xp, reward, gamma_current, gamma_next, lam, alpha):
    delta = reward + gamma_next * dot(w,xp) - dot(w,x)
    en = [gamma_current*lam*ei + xi for ei,xi in zip(e,x)]
    wn = [wi + alpha*delta*ei for wi,ei in zip(w,en)]
    return wn,en
# END streaming


# BEGIN retention
def ewc_quadratic_step(w, new_target, old_anchor, precision, strength, alpha):
    # Known scalar curvature, NOT an empirical Fisher estimate from an RL policy.
    gradient = (w-new_target) + strength*precision*(w-old_anchor)
    return w-alpha*gradient
# END retention


# BEGIN plasticity
def recycle_relu_unit(incoming, outgoing, biases, index, new_incoming):
    """One hidden-layer mechanism: replace incoming weights, zero outgoing.

    Selection, maturity and optimizer-state reset are intentionally not modeled.
    It preserves output exactly only if the old unit contributed zero.
    """
    wi=[row.copy() for row in incoming]; wo=outgoing.copy(); bs=biases.copy()
    wi[index]=new_incoming.copy(); wo[index]=0.0; bs[index]=0.0
    return wi,wo,bs
# END plasticity


# BEGIN exploration
def rnd_scalar_step(predictor, frozen_target, observation, alpha):
    # A deliberately scalar RND-mechanism illustration, not a deep RND agent.
    target = frozen_target * observation
    error = target - predictor * observation
    intrinsic_reward = error * error  # Measure BEFORE fitting this sample.
    predictor += alpha * error * observation
    return predictor,intrinsic_reward
# END exploration


# BEGIN architectures
def agent_transition(q, rate, prediction, s, action, reward, sp, behavior_prob):
    """One experience, two questions, separate learner states.

    Controller: average-reward Differential Q.
    Predictor: discounted GVF for the fixed policy 'always action 0'.
    State is supplied by a HAND-DESIGNED cue memory. No autonomous state/skill
    discovery and no claim to implement the OaK architecture.
    """
    if behavior_prob <= 0: raise ValueError('Behavior probability must be positive')
    x=[float(i==s) for i in range(len(prediction))]
    xp=[float(i==sp) for i in range(len(prediction))]
    rho=1/behavior_prob if action==0 else 0.0
    new_prediction=gvf_td(prediction,x,xp,reward,.9,rho,.005)
    new_q,new_rate=differential_q(q,rate,s,action,reward,sp,.05,.05)
    return new_q,new_rate,new_prediction
# END architectures


def demo_gvf(seed=7):
    # One state, two actions. Behavior is uniform; target always chooses action 1.
    # Cumulant=action, gamma=.5 -> target GVF value = 1/(1-.5)=2.
    rng, w, wg, h = random.Random(seed), [0.0], [0.0], [0.0]
    for t in range(12000):
        a = rng.randrange(2)
        rho = 2.0 if a == 1 else 0.0
        w = gvf_td(w, [1.0], [1.0], a, .5, rho, .02)
        wg, h = gtd2(wg, h, [1.0], [1.0], a, .5, rho, .003, .02)
    return {'analytic_target': 2.0, 'is_td': w[0], 'gtd2': wg[0],
            'behavior_value_if_no_IS': 1.0}


def demo_state(seed=7):
    # Cue +/-1, then 3 zero observations; only sequence-end label is used.
    # theta held fixed during each short sequence; update at its end.
    rng, theta = random.Random(seed), [.6, .5, 0.0]
    for _ in range(2500):
        cue = rng.choice([-1.0, 1.0])
        state, jac = 0.0, [0.0]*3
        for obs in [cue, 0.0, 0.0, 0.0]:
            state, jac = rtrl_step(state, jac, obs, theta)
        theta = [p - .03 * (state - .8*cue) * j for p, j in zip(theta, jac)]
    predictions = []
    for cue in [-1.0, 1.0]:
        state, jac = 0.0, [0.0]*3
        for obs in [cue, 0.0, 0.0, 0.0]:
            state, jac = rtrl_step(state, jac, obs, theta)
        predictions.append(state)
    posterior = belief_step([.5,.5], [[1,0],[0,1]], [.9,.1])
    return {'last_observation_both_cases': 0, 'rnn_predictions': predictions,
            'theta': theta, 'known_model_belief_after_cue': posterior,
            'protocol': 'short resettable supervised sequences; not a lifelong RL benchmark'}


def demo_average(seed=7):
    v, rate = [0.0, 0.0], 0.0
    for t in range(12000):
        s = t % 2
        v, rate = differential_td(v, rate, s, 2.0*s, 1-s, .03, .1)
    rng, q, qrate = random.Random(seed), [[0.0, 0.0]], 0.0
    for _ in range(12000):
        # Behavior chooses a suboptimal action with probability 1/2.
        a = rng.randrange(2)
        q, qrate = differential_q(q, qrate, 0, a, float(a), 0, .03, .1)
    return {'td_rate': rate, 'analytic_rate': 1, 'relative_value_gap': v[1]-v[0],
            'q_rate': qrate, 'q_action_gap': q[0][1]-q[0][0], 'behavior_reward_mean': .5}


def demo_options(seed=7):
    return {'continue_or_switch_target': option_target(1,.9,.25,4,6),
            'two_step_smdp_target': smdp_target([1,2],.9,10),
            'termination_logit_when_advantage_negative': termination_logit_step(0,-2,.1)}


def demo_models(seed=7):
    # 0 -> 1 -> 2, terminate on arrival at 2. State 2 self-transitions.
    r, p = [0.0]*3, [[0.0]*3 for _ in range(3)]
    for _ in range(12):
        r, p = option_model_sweep([1,2,0], [[0,1,0],[0,0,1],[0,0,1]], [0,0,1], .9, r, p)
    return {'r_from_0': r[0], 'discounted_endpoint_from_0': p[0],
            'backup_with_endpoint_value_10': r[0] + dot(p[0], [0,0,10]),
            'analytic': {'r': 2.8, 'endpoint_weight': .81, 'backup': 10.9}}


def demo_planning(seed=7):
    # Same endpoint (the single decision state). Primitive reward=1; skill=2.8.
    # Primitive takes 1 step; skill takes 2, hence .9 vs .81.
    v = model_plan([[1.0,2.8]], [[[.9],[.81]]], 200)[0]
    return {'planned_value': v, 'analytic_value': 2.8/(1-.81),
            'wrong_if_discount_applied_twice': 2.8/(1-.9*.81)}


def demo_meta(seed=7):
    rng, w, b, h = random.Random(seed), [0.,0.], [math.log(.05)]*2, [0.,0.]
    sums = [0.0,0.0]
    for t in range(4000):
        x = [rng.choice([-1.,1.]), rng.choice([-1.,1.])]
        target = (1 if t < 2000 else -1)*x[0] + rng.gauss(0,.1)
        sums[t//2000] += (target-dot(w,x))**2
        w,b,h = idbd_step(w,b,h,x,target,.005)
    loss, derivative = scalar_meta_gradient(math.log(.2), [1.,2.], 3.)
    return {'phase_mse': [s/2000 for s in sums], 'final_weights': w,
            'final_step_sizes': [math.exp(z) for z in b],
            'two_step_meta_loss': loss, 'd_loss_d_log_alpha': derivative}


def demo_goals(seed=7):
    return {'original_goal_B_achieved_A_reward': goal_reward('A','B'),
            'relabeled_goal_A_reward': goal_reward('A','A'),
            'stopping_subtask_target': subtask_target(-1,.9,1,5,8),
            'continuing_subtask_target': subtask_target(-1,.9,0,5,8),
            'note': 'Relabeling changes the question; it does not generate a physical success.'}


def demo_streaming(seed=7):
    # A -> B -> terminal, only final reward=1. One episode, not single-life.
    results={}
    for lam in [0.,.8]:
        w,e=[0.,0.],[0.,0.]
        w,e=td_trace_step(w,e,[1,0],[0,1],0,1,.9,lam,.1)
        w,e=td_trace_step(w,e,[0,1],[0,0],1,.9,0,lam,.1)
        results[str(lam)]={'weights':w,'trace':e}
    return results


def demo_retention(seed=7):
    result={}
    for strength in [0.,1.,5.]:
        w=1.
        for _ in range(300): w=ewc_quadratic_step(w,-1,1,1,strength,.05)
        result[str(strength)]={'weight':w,'analytic':(strength-1)/(strength+1),
                              'old_loss':.5*(w-1)**2,'new_loss':.5*(w+1)**2}
    return result


def demo_plasticity(seed=7):
    wi,wo,bs=[[-1.],[1.]],[3.,2.],[-2.,0.]
    def output(iw,ow,bias):
        return sum(o*max(0,dot(row,[1.])+bi) for row,o,bi in zip(iw,ow,bias))
    before=output(wi,wo,bs)
    wn,on,bn=recycle_relu_unit(wi,wo,bs,0,[.5])
    return {'before':before,'after_recycling_zero_contribution_unit':output(wn,on,bn),
            'next_outgoing_gradient_for_target_3':(before-3)*.5,
            'scope':'interface check, not a test of maintained plasticity'}


def demo_exploration(seed=7):
    predictor,history=0.,[]
    for _ in range(15):
        predictor,reward=rnd_scalar_step(predictor,2.,1.,.2)
        history.append(reward)
    forgotten_reward=(2.-0.)**2
    return {'first_intrinsic_reward':history[0],'last_intrinsic_reward':history[-1],
            'if_predictor_is_reset_intrinsic_reward':forgotten_reward,
            'frozen_target':2.}


def demo_architectures(seed=7):
    rng=random.Random(seed); q=[[0.,0.] for _ in range(6)]; rate=0.; prediction=[0.]*6
    cue,phase=rng.randrange(2),0; totals=[0.,0.,0.]; epsilon=.2; steps=18000
    for t in range(steps):
        # At the entry (phase 0), a cue becomes visible. It is not visible later.
        # Hand-designed memory retains it; we do not give the controller hidden task ID.
        s=2*phase+cue
        best=0 if q[s][0]>=q[s][1] else 1
        action=rng.randrange(2) if rng.random()<epsilon else best
        prob=epsilon/2 + (1-epsilon if action==best else 0)
        mapping=cue if t<6000 else 1-cue
        reward=float(phase==2 and action==mapping)
        totals[t//6000]+=reward
        next_phase=(phase+1)%3
        # A new cue is sampled before the next action at the entry; model the
        # actual next entry cue now so next state is consistent with the next loop.
        next_cue=rng.randrange(2) if next_phase==0 else cue
        sp=2*next_phase+next_cue
        q,rate,prediction=agent_transition(q,rate,prediction,s,action,reward,sp,prob)
        phase,cue=next_phase,next_cue
    return {'steps':steps,'env_reset_calls':0,'learner_resets':0,
            'reward_per_step_by_block':[r/6000 for r in totals],
            'reward_rate_estimate':rate,'gvf_always_action_0':prediction,
            'scope':'hand-designed memory + two learned heads, not an autonomous architecture'}


class MechanismTests(unittest.TestCase):
    def test_recurrent_jacobian_product_order(self):
        # X_t = F_t X_(t-1) + theta B_t; the F matrices do not commute.
        matrices=[[[1.,0.],[0.,1.]],[[1.,2.],[0.,1.]],[[1.,0.],[3.,1.]]]
        inputs=[[1.,1.],[2.,-1.],[.5,.1]]
        def matvec(matrix,vector):
            return [dot(row,vector) for row in matrix]
        def forward(theta):
            state=[0.,0.]
            for matrix,b in zip(matrices,inputs):
                state=[v+theta*bi for v,bi in zip(matvec(matrix,state),b)]
            return state
        jacobian=[0.,0.]
        for matrix,b in zip(matrices,inputs):
            jacobian=[v+bi for v,bi in zip(matvec(matrix,jacobian),b)]
        ordered=matvec(matrices[2],matvec(matrices[1],inputs[0]))
        reversed_order=matvec(matrices[1],matvec(matrices[2],inputs[0]))
        self.assertNotEqual(ordered,reversed_order)
        expanded=[a+b+c for a,b,c in zip(ordered,matvec(matrices[2],inputs[1]),inputs[2])]
        for j,exact,p,m in zip(jacobian,expanded,forward(.2+1e-6),forward(.2-1e-6)):
            self.assertAlmostEqual(j,exact)
            self.assertAlmostEqual(j,(p-m)/2e-6,places=7)

    def test_trace_credit(self):
        r=demo_streaming()
        self.assertEqual(r['0.0']['weights'],[0.,.1])
        self.assertAlmostEqual(r['0.8']['weights'][0],.072)

    def test_retention_analytic(self):
        for result in demo_retention().values():
            self.assertAlmostEqual(result['weight'],result['analytic'],places=5)

    def test_recycle_zero_contribution(self):
        r=demo_plasticity()
        self.assertEqual(r['before'],r['after_recycling_zero_contribution_unit'])
        self.assertEqual(r['next_outgoing_gradient_for_target_3'],-.5)

    def test_rnd_reward_before_update(self):
        predictor,reward=rnd_scalar_step(0,2,1,.2)
        self.assertEqual(reward,4); self.assertEqual(predictor,.4)

    def test_separate_head_questions(self):
        q,rate,p=agent_transition([[0.,0.]],0.,[0.],0,1,1.,0,.5)
        self.assertEqual(p,[0.]); self.assertGreater(q[0][1],0)
        self.assertGreater(rate,0)

    def test_gvf_terminal(self):
        self.assertEqual(gvf_td([2],[1],[1],3,0,1,.1),[2.1])

    def test_gvf_zero_ratio(self):
        self.assertEqual(gvf_td([2],[1],[1],3,.9,0,.1),[2])

    def test_gtd2_old_auxiliary(self):
        w,h = gtd2([1],[2],[1],[1],3,.5,2,.1,.2)
        self.assertAlmostEqual(w[0],1.2)
        self.assertAlmostEqual(h[0],2.6)

    def test_gtd2_expected_direction_matches_mspbe_gradient(self):
        # Uniform fixed behavior distribution over four synthetic transitions.
        data=[([1.,0.],[0.,1.],1.,.8,1.5),([0.,1.],[1.,0.],-.3,.8,.5),
              ([1.,1.],[0.,1.],.4,.5,1.),([1.,-1.],[1.,0.],.2,.3,1.)]
        w=[.2,-.1]; n=len(data)
        C=[[sum(x[i]*x[j] for x,xp,r,g,rho in data)/n for j in range(2)] for i in range(2)]
        A=[[sum(rho*x[i]*(x[j]-g*xp[j]) for x,xp,r,g,rho in data)/n for j in range(2)] for i in range(2)]
        b=[sum(rho*r*x[i] for x,xp,r,g,rho in data)/n for i in range(2)]
        def solve(y):
            determinant=C[0][0]*C[1][1]-C[0][1]*C[1][0]
            return [(C[1][1]*y[0]-C[0][1]*y[1])/determinant,
                    (C[0][0]*y[1]-C[1][0]*y[0])/determinant]
        def objective(params):
            residual=[bi-dot(row,params) for bi,row in zip(b,A)]
            return .5*dot(residual,solve(residual))
        h=solve([bi-dot(row,w) for bi,row in zip(b,A)])
        updates=[gtd2(w,h,x,xp,r,g,rho,1.,.1)[0] for x,xp,r,g,rho in data]
        for i in range(2):
            plus,minus=w.copy(),w.copy();plus[i]+=1e-6;minus[i]-=1e-6
            gradient=(objective(plus)-objective(minus))/2e-6
            direction=sum(u[i]-w[i] for u in updates)/n
            self.assertAlmostEqual(direction,-gradient,places=7)

    def test_gvf_known_solution(self):
        r=demo_gvf()
        self.assertAlmostEqual(r['is_td'],2,places=6)
        self.assertAlmostEqual(r['gtd2'],2,places=3)

    def test_belief_and_neutral_observation(self):
        b=belief_step([.5,.5],[[1,0],[0,1]],[.9,.1])
        self.assertEqual(b,[.9,.1])
        self.assertEqual(belief_step(b,[[1,0],[0,1]],[1,1]),b)

    def test_belief_impossible(self):
        with self.assertRaises(ValueError): belief_step([.5,.5],[[1,0],[0,1]],[0,0])

    def test_rtrl_finite_difference(self):
        observations=[1.,0.,-.2,.4]; theta=[.7,.3,-.1]
        def forward(params):
            s,j=0.,[0.]*3
            for o in observations: s,j=rtrl_step(s,j,o,params)
            return s,j
        _,jac=forward(theta)
        for k in range(3):
            plus,minus=theta.copy(),theta.copy(); plus[k]+=1e-6; minus[k]-=1e-6
            self.assertAlmostEqual(jac[k],(forward(plus)[0]-forward(minus)[0])/2e-6,places=7)

    def test_state_delayed_cue(self):
        left,right=demo_state()['rnn_predictions']
        self.assertLess(left,-.7); self.assertGreater(right,.7)

    def test_differential_same_delta(self):
        q,g=differential_q([[2],[4]],1,0,0,3,1,.1,.5)
        self.assertEqual(q[0],[2.4]); self.assertAlmostEqual(g,1.2)

    def test_average_known_solutions(self):
        r=demo_average()
        for key in ['td_rate','relative_value_gap','q_rate','q_action_gap']:
            self.assertAlmostEqual(r[key],1,places=5)

    def test_average_shift_invariance(self):
        a,g=differential_td([1,3],.4,0,2,1,.1,.2)
        b,h=differential_td([101,103],.4,0,2,1,.1,.2)
        self.assertAlmostEqual(g,h)
        self.assertAlmostEqual(b[0]-a[0],100)

    def test_option_old_duration(self):
        q,g,length=average_option_step(2,1,4,8,10,3,.2,.5,.1)
        self.assertAlmostEqual(q,2.25); self.assertAlmostEqual(g,1.125)
        self.assertAlmostEqual(length,4.6)

    def test_option_invalid_duration(self):
        with self.assertRaises(ValueError): average_option_step(0,0,0,1,1,0,.1,.1,.1)

    def test_option_termination_limits(self):
        self.assertAlmostEqual(option_target(1,.9,0,4,6),4.6)
        self.assertAlmostEqual(option_target(1,.9,1,4,6),6.4)

    def test_termination_gradient(self):
        self.assertLess(termination_logit_step(0,2,.1),0)
        self.assertGreater(termination_logit_step(0,-2,.1),0)
        self.assertEqual(termination_logit_step(0,0,.1),0)

    def test_smdp_discount(self):
        self.assertAlmostEqual(smdp_target([1,2],.9,10),10.9)

    def test_model_equals_sample_path(self):
        r=demo_models()
        self.assertAlmostEqual(r['r_from_0'],2.8)
        self.assertAlmostEqual(r['discounted_endpoint_from_0'][2],.81)
        self.assertAlmostEqual(r['backup_with_endpoint_value_10'],10.9)

    def test_model_one_step(self):
        r,p=option_model_sweep([1,2],[[0,1],[1,0]],[1,1],.9,[99,99],[[99,99],[99,99]])
        self.assertEqual(r,[1,2]); self.assertEqual(p,[[0,.9],[.9,0]])

    def test_planning_known_value(self):
        r=demo_planning()
        self.assertAlmostEqual(r['planned_value'],r['analytic_value'],places=8)

    def test_meta_finite_difference(self):
        b=math.log(.2); ys=[1.,2.,-.5]
        _,grad=scalar_meta_gradient(b,ys,3.)
        numeric=(scalar_meta_gradient(b+1e-6,ys,3.)[0]-scalar_meta_gradient(b-1e-6,ys,3.)[0])/2e-6
        self.assertAlmostEqual(grad,numeric,places=7)

    def test_idbd_first_step_and_irrelevant_feature(self):
        w,b,h=idbd_step([0,0],[math.log(.1)]*2,[0,0],[1,0],2,.01)
        self.assertAlmostEqual(w[0],.2); self.assertEqual(w[1],0)
        self.assertAlmostEqual(h[0],.2); self.assertEqual(h[1],0)

    def test_goal_relabel(self):
        self.assertEqual(goal_reward('A','A'),0)
        self.assertEqual(goal_reward('A','B'),-1)

    def test_subtask_not_option_endpoint(self):
        self.assertEqual(subtask_target(-1,.9,1,5,8),4)
        self.assertAlmostEqual(subtask_target(-1,.9,0,5,8),6.2)
        self.assertNotEqual(subtask_target(-1,.9,1,5,8),smdp_target([-1],.9,5))


DEMOS={name:globals()['demo_'+name] for name in ['gvf','state','average','options','models','planning','meta','goals','streaming','retention','plasticity','exploration','architectures']}
if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('lab',choices=['all','test']+list(DEMOS),nargs='?',default='all')
    parser.add_argument('--seed',type=int,default=7)
    args=parser.parse_args()
    if args.lab=='test': unittest.main(argv=['mechanisms.py'],verbosity=2)
    else:
        selected=DEMOS if args.lab=='all' else {args.lab:DEMOS[args.lab]}
        print(json.dumps({name:fn(args.seed) for name,fn in selected.items()},indent=2,ensure_ascii=False))
