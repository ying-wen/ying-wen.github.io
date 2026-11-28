"""MIT-licensed, standard-library mechanism labs; not paper benchmark reproductions.

Python 3.10+: python lifelong_algorithms_lab.py {average,streaming,retention,
plasticity,exploration,architectures,all,test}. No files or network are used.
Each region is embedded verbatim in its tutorial; tests check algebra and timing.
"""
from __future__ import annotations

import argparse
import math
import random
import unittest
from dataclasses import dataclass


# BEGIN average
def differential_td(values, rate, state, reward, next_state,
                    alpha=0.1, eta=0.1, rho=1.0):
    """Tabular fixed-policy prediction. Compute BOTH updates from old values."""
    if alpha < 0 or eta < 0 or rho < 0:
        raise ValueError("nonnegative steps and importance ratio required")
    error = reward - rate + values[next_state] - values[state]
    result = list(values)
    result[state] += alpha * rho * error
    return result, rate + eta * alpha * rho * error, error


def differential_q(q, rate, state, action, reward, next_state,
                   alpha=0.1, eta=0.1):
    """Off-policy control: max target; no behavior-reward EMA."""
    error = reward - rate + max(q[next_state]) - q[state][action]
    result = [row[:] for row in q]
    result[state][action] += alpha * error
    return result, rate + eta * alpha * error, error


def rvi_sweep(h, transitions, reference=0):
    """Known model: transitions[s][a] contains (probability,reward,next_state)."""
    backed = [max(sum(prob * (reward + h[sp]) for prob, reward, sp in action)
                  for action in state) for state in transitions]
    reference_value = backed[reference]
    return [v - reference_value for v in backed], reference_value


def option_rate_step(q, rate, mean_duration, total_reward, duration,
                     next_best, alpha=0.1, eta=0.1, duration_alpha=0.1):
    """Duration-normalized expected-length variant; use OLD positive length."""
    if mean_duration <= 0 or duration <= 0:
        raise ValueError("option durations must be positive")
    error = total_reward - rate * mean_duration + next_best - q
    scaled = alpha * error / mean_duration
    return (q + scaled, rate + eta * scaled,
            mean_duration + duration_alpha * (duration - mean_duration))


def average_demo():
    values, rate = [0.0, 0.0], 0.0
    for t in range(20000):
        state = t % 2
        values, rate, _ = differential_td(
            values, rate, state, 2.0 * state, 1 - state, 0.02, 0.1)
    q, optimal_rate = [[0.0, 0.0]], 0.0
    for t in range(10000):
        action = t % 2  # behavior reward rate is exactly 0.5
        q, optimal_rate, _ = differential_q(q, optimal_rate, 0, action,
                                            float(action), 0, 0.02, 0.1)
    print("average", {"prediction_rate": round(rate, 6),
          "h1_minus_h0": round(values[1] - values[0], 6),
          "control_rate": round(optimal_rate, 6), "behavior_rate": 0.5,
          "option_step": option_rate_step(1, 0.5, 2, 5, 4, 3)})
    q_centered, reference = 0.0, 0.0
    for _ in range(1000):
        q_centered, reference, _ = centered_single_state_step(
            q_centered, reference, 1.0, gamma=0.9, eta=0.1, alpha=0.3)
    print("discounted_centering", {"q": round(q_centered, 6),
          "reference_c": round(reference, 6), "true_reward_rate": 1.0,
          "invariant_c_minus_eta_q": round(reference - 0.1*q_centered, 12)})
# END average


# BEGIN centering
def centered_single_state_step(q, reference, reward, gamma=0.9,
                               eta=0.1, alpha=0.1):
    """One-state diagnostic, not a complete reward-centering implementation.

    Discounted TD reference c need not equal the actual reward rate.
    Both writes use the SAME old-parameter error. gamma=1 is the
    differential limiting comparison, not discounted policy equivalence.
    """
    if not 0 <= gamma <= 1 or eta < 0 or alpha < 0:
        raise ValueError("invalid discount or nonnegative update scale")
    delta = reward - reference - (1 - gamma)*q
    return q + alpha*delta, reference + eta*alpha*delta, delta


def centered_single_state_fixed_point(reward, gamma, eta, q0=0.0, c0=0.0):
    """Solve c-eta*q invariant and zero TD error; not a stability claim."""
    if not 0 <= gamma <= 1 or eta < 0 or eta + 1 - gamma <= 0:
        raise ValueError("a positive fixed-point denominator is required")
    invariant = c0 - eta*q0
    q = (reward - invariant)/(eta + 1 - gamma)
    return q, invariant + eta*q
# END centering


# BEGIN streaming
@dataclass
class Welford:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def observe(self, x):
        self.count += 1
        old_delta = x - self.mean
        self.mean += old_delta / self.count
        self.m2 += old_delta * (x - self.mean)
        return self.mean, self.variance

    @property
    def variance(self):
        return self.m2 / (self.count - 1) if self.count > 1 else 1.0


def td_lambda_step(w, trace, x, xp, reward, gamma_current,
                   gamma_next, lam=0.8, alpha=0.1, kappa=None):
    """Linear TD: gamma_current decays past trace; gamma_next bootstraps.
    kappa enables the ObGD algebra, NOT a general nonlinear safety theorem.
    """
    if not (len(w) == len(trace) == len(x) == len(xp)):
        raise ValueError("dimension mismatch")
    v = sum(a*b for a, b in zip(w, x))
    vp = sum(a*b for a, b in zip(w, xp))
    error = reward + gamma_next * vp - v
    z = [gamma_current * lam * old + feature
         for old, feature in zip(trace, x)]
    step = alpha
    if kappa is not None:
        mass = alpha * kappa * max(abs(error), 1.0) * sum(map(abs, z))
        step = alpha / max(1.0, mass)  # defined also when z == 0
    return [a + step * error * b for a, b in zip(w, z)], z, error, step


def streaming_demo():
    outcomes = {}
    for lam in (0.0, 0.8):
        w, z, _, _ = td_lambda_step([0., 0.], [0., 0.], [1., 0.],
                                    [0., 1.], 0, 0, 0.9, lam)
        w, z, _, _ = td_lambda_step(w, z, [0., 1.], [0., 0.],
                                    1, 0.9, 0, lam)
        outcomes[str(lam)] = [round(v, 6) for v in w]
    stats = Welford()
    for x in (1., 2., 3.):
        stats.observe(x)
    bounded = td_lambda_step([0.], [0.], [10.], [0.], 1, 0, 0,
                             alpha=0.1, kappa=2)
    print("streaming", {"two_step_credit": outcomes,
          "mean_variance": (stats.mean, stats.variance),
          "obgd_illustration_weight": bounded[0],
          "warning": "gradient 10 violates the <=1 heuristic bound"})
# END streaming


# BEGIN retention
def reservoir_insert(buffer, item, seen, capacity, rng):
    """seen is the number INCLUDING this item; O(capacity) retained data."""
    if capacity < 0 or seen < 1:
        raise ValueError("invalid budget or sample counter")
    if len(buffer) < capacity:
        buffer.append(item)
    elif capacity:
        index = rng.randrange(seen)
        if index < capacity:
            buffer[index] = item


def ewc_scalar_optimum(new_target, old_weight, importance, strength):
    if importance < 0 or strength < 0:
        raise ValueError("curvature and strength must be nonnegative")
    k = importance * strength
    return (new_target + k * old_weight) / (1.0 + k)


def ewc_update(weights, current_gradient, old_weights, importance,
               strength=1.0, alpha=0.1):
    """One SGD update; current_gradient excludes the consolidation penalty."""
    if not (len(weights) == len(current_gradient) == len(old_weights) == len(importance)):
        raise ValueError("EWC dimensions differ")
    if strength < 0 or any(f < 0 for f in importance):
        raise ValueError("EWC requires nonnegative importance")
    return [w - alpha * (g + strength*f*(w-old))
            for w, g, old, f in zip(weights, current_gradient, old_weights, importance)]


def clone_loss(teacher_prob, student_prob, teacher_value, student_value):
    """CLEAR-like replay cloning terms only, not the complete RL objective."""
    if len(teacher_prob) != len(student_prob):
        raise ValueError("policy dimensions differ")
    if any(p < 0 for p in teacher_prob) or any(q <= 0 for q in student_prob):
        raise ValueError("teacher >=0, student >0 required")
    if not math.isclose(sum(teacher_prob), 1.) or not math.isclose(sum(student_prob), 1.):
        raise ValueError("policies must sum to one")
    kl = sum(p * math.log(p / q) for p, q in zip(teacher_prob, student_prob) if p)
    return kl, (student_value - teacher_value) ** 2


def vtrace_target(rewards, values, rhos, gamma=0.9, rho_cap=1., c_cap=1.):
    """Finite unroll value target. values includes the final bootstrap value."""
    if len(values) != len(rewards) + 1 or len(rhos) != len(rewards):
        raise ValueError("unroll dimensions differ")
    if min(rho_cap, c_cap) < 0 or c_cap > rho_cap or any(rho < 0 for rho in rhos):
        raise ValueError("require 0 <= c_cap <= rho_cap and nonnegative ratios")
    correction, product = 0., 1.
    for t, reward in enumerate(rewards):
        delta = min(rho_cap, rhos[t]) * (reward + gamma * values[t+1] - values[t])
        correction += product * delta
        product *= gamma * min(c_cap, rhos[t])
    return values[0] + correction


def retention_demo():
    buffer, rng = [], random.Random(7)
    for seen in range(1, 101):
        reservoir_insert(buffer, seen, seen, 5, rng)
    weight = [1.]
    for _ in range(100):
        weight = ewc_update(weight, [weight[0]+1.], [1.], [1.], strength=1.)
    print("retention", {"fixed_budget_reservoir": buffer,
          "ewc_optima_k_0_1_5": [ewc_scalar_optimum(-1, 1, 1, k) for k in (0, 1, 5)],
          "ewc_sgd_k1": round(weight[0], 8),
          "clone_terms": clone_loss([0.8, 0.2], [0.5, 0.5], 2., 1.),
          "two_step_vtrace": vtrace_target([0, 1], [0, 0, 0], [1, 1])})
# END retention


# BEGIN plasticity
def redo_indices(mean_abs_activations, threshold=0.1):
    """Relative activity score; an all-zero layer is entirely dormant."""
    if not mean_abs_activations:
        return []
    denominator = sum(mean_abs_activations) / len(mean_abs_activations)
    if denominator == 0:
        return list(range(len(mean_abs_activations)))
    return [i for i, a in enumerate(mean_abs_activations)
            if a / denominator <= threshold]


def cbp_candidates(utilities, ages, maturity, rate, credit=0.0):
    """Accumulated-budget variant: protect young units, rank eligible utility."""
    if not 0 <= rate <= 1:
        raise ValueError("replacement rate outside [0,1]")
    eligible = [i for i, age in enumerate(ages) if age > maturity]
    credit += rate * len(eligible)
    count = min(len(eligible), int(credit))
    selected = sorted(eligible, key=lambda i: (utilities[i], i))[:count]
    return selected, credit - count


def cbp_contribution_utility(old_utility, mean_absolute_activation,
                             mean_absolute_outgoing, age, decay=0.99):
    """EMA plus age correction for the explicitly chosen contribution variant."""
    if age < 1 or not 0 <= decay < 1:
        raise ValueError("increment age before updating utility")
    instantaneous = mean_absolute_activation * mean_absolute_outgoing
    updated = decay * old_utility + (1. - decay) * instantaneous
    return updated, updated / (1. - decay**age)


def recycle_one_unit(incoming, outgoing, first_moment, second_moment,
                     index, new_incoming):
    """Scalar-input, scalar-output hidden unit; reset its incoming/outgoing state.
    first/second_moment[i] each contain [incoming_parameter, outgoing_parameter].
    """
    incoming[index] = new_incoming
    outgoing[index] = 0.0
    first_moment[index] = [0.0, 0.0]
    second_moment[index] = [0.0, 0.0]


def relu_prediction(incoming, outgoing, x=1.0):
    return sum(v * max(0., u*x) for u, v in zip(incoming, outgoing))


def train_relu_toy(incoming, outgoing, target=1., steps=20, alpha=0.1):
    incoming, outgoing = incoming[:], outgoing[:]
    for _ in range(steps):
        error = target - relu_prediction(incoming, outgoing)
        old_u, old_v = incoming[:], outgoing[:]
        for i in range(len(incoming)):
            incoming[i] += alpha * error * old_v[i] * (old_u[i] > 0)
            outgoing[i] += alpha * error * max(0., old_u[i])
    return 0.5 * (target - relu_prediction(incoming, outgoing)) ** 2


def plasticity_demo():
    u, v = [-1., 1.], [0.5, 2.]
    before = relu_prediction(u, v)
    moments1, moments2 = [[3., 4.], [3., 4.]], [[5., 6.], [5., 6.]]
    recycle_one_unit(u, v, moments1, moments2, 0, 0.5)
    print("plasticity", {"redo_selected": redo_indices([0., 2.]),
          "prediction_before_after": (before, relu_prediction(u, v)),
          "cleared_optimizer_state": moments1[0] + moments2[0],
          "matched_probe_loss_aged": train_relu_toy([-1.], [1.]),
          "matched_probe_loss_fresh": train_relu_toy([0.5], [0.]),
          "scope": "synthetic dead-ReLU mechanism, not an empirical LoP diagnosis"})
# END plasticity


# BEGIN exploration
def count_bonus(count, scale=1.):
    if count < 0:
        raise ValueError("count must be nonnegative")
    return scale / math.sqrt(count + 1.)


def rnd_step(prediction, fixed_target, alpha=0.25):
    """One table feature: novelty is measured BEFORE fitting this sample."""
    error = fixed_target - prediction
    return error * error, prediction + alpha * error


def progress_score(old_error, new_error, floor=0.):
    """Positive progress, not raw surprise; both errors use a matched probe."""
    return max(floor, old_error - new_error)


def recovery_filter(proposed_action, recovery_probabilities, threshold):
    """Toy known-probability filter; learned estimates give no safety guarantee."""
    if recovery_probabilities[proposed_action] >= threshold:
        return proposed_action
    feasible = [a for a, prob in enumerate(recovery_probabilities)
                if prob >= threshold]
    return max(feasible, key=lambda a: recovery_probabilities[a]) if feasible else None


def exploration_demo():
    predictor = 0.0
    novelty = []
    for _ in range(4):
        bonus, predictor = rnd_step(predictor, 2.0)
        novelty.append(round(bonus, 6))
    print("exploration", {"rnd_bonus_before_fit": novelty,
          "fake_novelty_if_predictor_reset": rnd_step(0., 2.)[0],
          "count_0_3_8": [count_bonus(n) for n in (0, 3, 8)],
          "noisy_vs_learnable_progress": [progress_score(4., 4.), progress_score(2., 1.)],
          "filtered_action": recovery_filter(1, [0.99, 0.2], 0.9)})
# END exploration


# BEGIN architectures
class ModularAgent:
    """Hand-designed Markov state; tabular control/GVF/one-step model.
    The model stores empirical counts of (reward,next_state) per state/action.
    Planning changes q only. Reward-rate learning uses REAL data only here.
    """
    def __init__(self, states=6, alpha=0.05, eta=0.05, gamma=0.8):
        self.q = [[0., 0.] for _ in range(states)]
        self.gvf = [0.] * states
        self.rate = 0.
        self.model = {}
        self.planning_cursor = 0
        self.alpha, self.eta, self.gamma = alpha, eta, gamma

    def action_probabilities(self, state, epsilon=0.2):
        best = max(range(2), key=lambda a: self.q[state][a])
        prob = [epsilon / 2., epsilon / 2.]
        prob[best] += 1. - epsilon
        return prob

    def observe(self, state, action, reward, next_state, behavior_probability,
                planning_budget=0):
        if behavior_probability <= 0 or planning_budget < 0:
            raise ValueError("invalid recorded probability or planning budget")
        # All REAL targets use one parameter snapshot before any mutation.
        q_error = reward - self.rate + max(self.q[next_state]) - self.q[state][action]
        prediction_error = reward + self.gamma*self.gvf[next_state] - self.gvf[state]
        ratio = (1.0 if action == 1 else 0.0) / behavior_probability
        self.q[state][action] += self.alpha*q_error
        self.rate += self.eta*self.alpha*q_error
        self.gvf[state] += self.alpha*ratio*prediction_error
        # Finite toy outcome slots, not an unlimited-lifetime byte bound:
        # Python counts/cursor grow in bit width; new reward values add keys.
        outcomes = self.model.setdefault((state, action), {})
        outcomes[reward, next_state] = outcomes.get((reward, next_state), 0) + 1
        # A deterministic scheduler makes the simulated-update budget inspectable.
        keys = sorted(self.model)
        for _ in range(planning_budget):
            s, a = keys[self.planning_cursor % len(keys)]
            self.planning_cursor += 1
            outcomes = self.model[s, a]
            count = sum(outcomes.values())
            delta = sum(n * (r - self.rate + max(self.q[sp]) - self.q[s][a])
                        for (r, sp), n in outcomes.items()) / count
            self.q[s][a] += self.alpha*delta
        return {"control_error": q_error, "prediction_error": prediction_error,
                "ratio": ratio, "planning_updates": planning_budget}


def architecture_run(seed=7, planning_budget=0, steps=18000):
    rng, agent = random.Random(seed), ModularAgent()
    cue, phase = rng.randrange(2), 0
    rewards, trace = [], None
    for t in range(steps):
        # State is a hand-coded (phase, remembered cue), not a discovered representation.
        state = 2*phase + cue
        probabilities = agent.action_probabilities(state)
        action = 0 if rng.random() < probabilities[0] else 1
        mapping = 0 if t < steps // 2 else 1  # hidden change, never supplied to agent
        reward = float(action == (cue ^ mapping)) if phase == 1 else 0.0
        if phase == 2:
            phase, cue = 0, rng.randrange(2)
        else:
            phase += 1
        next_state = 2*phase + cue
        trace = agent.observe(state, action, reward, next_state,
                              probabilities[action], planning_budget)
        rewards.append(reward)
    block = steps // 3
    means = [sum(rewards[i*block:(i+1)*block]) / block for i in range(3)]
    return {"window_reward_rates": [round(v, 6) for v in means],
            "optimal_rate_estimate": round(agent.rate, 6),
            "model_entries": len(agent.model), "last_update": trace}


def architectures_demo():
    print("architectures", {"without_planning": architecture_run(),
          "one_simulated_update_per_step": architecture_run(planning_budget=1),
          "scope": "fixed-budget toy integration; not an OaK implementation"})
# END architectures


class MechanismTests(unittest.TestCase):
    def test_centering_reference_is_not_true_reward_rate(self):
        q, c = 0.0, 0.0
        for _ in range(1000):
            q, c, _ = centered_single_state_step(q, c, 1., alpha=.3)
        self.assertAlmostEqual(q, 5.)
        self.assertAlmostEqual(c, .5)
        self.assertNotAlmostEqual(c, 1.)

    def test_centering_invariant_with_arbitrary_initialization(self):
        q, c, eta = 2., -.3, .2
        invariant = c-eta*q
        for reward in [1., -2., 3., 0.]:
            q, c, _ = centered_single_state_step(q, c, reward, .8, eta, .3)
            self.assertAlmostEqual(c-eta*q, invariant)
        fixed_q, fixed_c = centered_single_state_fixed_point(1., .8, eta, 2., -.3)
        self.assertAlmostEqual(1.-fixed_c-.2*fixed_q, 0.)
        self.assertAlmostEqual(fixed_c-eta*fixed_q, invariant)

    def test_centering_differential_limit(self):
        _, c = centered_single_state_fixed_point(3., 1., .2, 4., -1.)
        self.assertAlmostEqual(c, 3.)

    def test_centering_rejects_invalid_settings(self):
        with self.assertRaises(ValueError):
            centered_single_state_step(0., 0., 1., gamma=1.1)
        with self.assertRaises(ValueError):
            centered_single_state_fixed_point(1., 1., 0.)

    def test_average_both_updates_use_old_error(self):
        v, g, d = differential_td([1., 3.], .5, 0, 2., 1)
        self.assertEqual(d, 3.5)
        self.assertAlmostEqual(v[0], 1.35)
        self.assertAlmostEqual(g, .535)

    def test_average_constant_offset(self):
        a = differential_td([1., 3.], .5, 0, 2., 1)
        b = differential_td([11., 13.], .5, 0, 2., 1)
        self.assertEqual(a[1:], b[1:])

    def test_average_zero_ratio_no_update(self):
        v, g, _ = differential_td([1., 2.], .5, 0, 4., 1, rho=0.)
        self.assertEqual(v, [1., 2.]); self.assertEqual(g, .5)

    def test_control_not_behavior_ema(self):
        q, rate = [[0., 0.]], 0.
        for t in range(10000):
            q, rate, _ = differential_q(q, rate, 0, t % 2, float(t % 2), 0, .02, .1)
        self.assertAlmostEqual(rate, 1., places=5)
        self.assertAlmostEqual(q[0][1] - q[0][0], 1., places=5)

    def test_rvi_known_solution(self):
        h, g = rvi_sweep([7.], [[[(1., 0., 0)], [(1., 1., 0)]]])
        self.assertEqual(h, [0.]); self.assertEqual(g, 8.)
        h, g = rvi_sweep(h, [[[(1., 0., 0)], [(1., 1., 0)]]])
        self.assertEqual(g, 1.)

    def test_option_uses_old_length(self):
        q, g, length = option_rate_step(1., .5, 2., 5., 4., 3.)
        self.assertAlmostEqual(q, 1.3)
        self.assertAlmostEqual(g, .53)
        self.assertAlmostEqual(length, 2.2)
        with self.assertRaises(ValueError):
            option_rate_step(0, 0, 0, 1, 1, 0)

    def test_duration_ratio_not_mean_of_ratios(self):
        rewards, durations = [1., 9.], [1., 3.]
        self.assertEqual(sum(rewards)/sum(durations), 2.5)
        self.assertEqual(sum(r/t for r, t in zip(rewards, durations))/2, 2.)

    def test_two_step_trace_hand_calculation(self):
        w, z, _, _ = td_lambda_step([0., 0.], [0., 0.], [1., 0.], [0., 1.], 0, 0, .9)
        w, z, _, _ = td_lambda_step(w, z, [0., 1.], [0., 0.], 1, .9, 0)
        self.assertAlmostEqual(w[0], .072); self.assertAlmostEqual(w[1], .1)

    def test_lambda_zero_and_terminal_trace(self):
        w, z, _, _ = td_lambda_step([0., 0.], [1., 0.], [0., 1.], [0., 0.], 1, .9, 0, lam=0)
        self.assertEqual(w, [0., .1]); self.assertEqual(z, [0., 1.])

    def test_welford(self):
        stats = Welford()
        for x in [1, 2, 3]: stats.observe(x)
        self.assertEqual(stats.mean, 2.)
        self.assertEqual(stats.variance, 1.)

    def test_obgd_zero_trace_and_bound(self):
        w, _, _, step = td_lambda_step([0.], [0.], [0.], [0.], 1, 0, 0, kappa=2)
        self.assertEqual(w, [0.]); self.assertEqual(step, .1)
        w, _, _, step = td_lambda_step([0.], [0.], [1.], [0.], 100, 0, 0, alpha=1, kappa=2)
        self.assertAlmostEqual(w[0], .5); self.assertAlmostEqual(step, .005)

    def test_obgd_not_unconditional_guarantee(self):
        w, _, _, _ = td_lambda_step([0.], [0.], [10.], [0.], 1, 0, 0, alpha=.1, kappa=2)
        self.assertGreater(w[0]*10, 1.)  # a deliberate negative example

    def test_reservoir_budget_and_zero(self):
        buffer, rng = [], random.Random(3)
        for i in range(1, 1001): reservoir_insert(buffer, i, i, 7, rng)
        self.assertEqual(len(buffer), 7); self.assertEqual(len(set(buffer)), 7)
        buffer = []; reservoir_insert(buffer, 1, 1, 0, rng)
        self.assertEqual(buffer, [])

    def test_ewc_stationary_gradient(self):
        for k in [0., 1., 5.]:
            w = ewc_scalar_optimum(-1, 1, 1, k)
            self.assertAlmostEqual((w+1) + k*(w-1), 0.)

    def test_ewc_update_uses_current_gradient_and_old_anchor(self):
        self.assertEqual(ewc_update([1.], [2.], [1.], [5.]), [.8])
        result = ewc_update([0.], [1.], [1.], [1.], strength=1.)
        self.assertEqual(result, [0.])

    def test_clone_identity_and_kl_direction(self):
        self.assertEqual(clone_loss([.8,.2], [.8,.2], 2, 2), (0., 0.))
        kl, mse = clone_loss([.8,.2], [.5,.5], 2, 1)
        self.assertAlmostEqual(kl, .8*math.log(1.6)+.2*math.log(.4))
        self.assertEqual(mse, 1)

    def test_vtrace_limits(self):
        self.assertAlmostEqual(vtrace_target([0,1], [0,0,0], [1,1]), .9)
        self.assertEqual(vtrace_target([5,9], [2,3,4], [0,1]), 2)

    def test_redo_zero_layer_and_relative_activity(self):
        self.assertEqual(redo_indices([0,2]), [0])
        self.assertEqual(redo_indices([0,0]), [0,1])

    def test_cbp_maturity_and_fractional_budget(self):
        selected, credit = cbp_candidates([0., .1, 1.], [0, 21, 21], 20, .25)
        self.assertEqual(selected, []); self.assertEqual(credit, .5)
        selected, credit = cbp_candidates([0., .1, 1.], [1, 22, 22], 20, .25, credit)
        self.assertEqual(selected, [1]); self.assertEqual(credit, 0.)

    def test_cbp_utility_age_correction(self):
        raw, corrected = cbp_contribution_utility(0., 2., 3., 1)
        self.assertAlmostEqual(raw, .06)
        self.assertAlmostEqual(corrected, 6.)
        self.assertEqual(cbp_candidates([0.], [100], 20, 0.), ([], 0.))

    def test_recycle_clears_state_and_zero_contribution(self):
        u, v, mom, var = [-1.,1.], [.5,2.], [[2.,3.],[2.,3.]], [[4.,5.],[4.,5.]]
        before = relu_prediction(u,v)
        recycle_one_unit(u,v,mom,var,0,.5)
        self.assertEqual(relu_prediction(u,v), before)
        self.assertEqual(mom[0]+var[0], [0.,0.,0.,0.])

    def test_reset_active_unit_changes_output(self):
        u, v, mom, var = [1.], [2.], [[0.,0.]], [[0.,0.]]
        before = relu_prediction(u,v)
        recycle_one_unit(u,v,mom,var,0,.5)
        self.assertEqual(before-relu_prediction(u,v), 2.)

    def test_aged_fresh_same_probe(self):
        self.assertEqual(train_relu_toy([-1.], [1.]), .5)
        self.assertLess(train_relu_toy([.5], [0.]), .1)

    def test_rnd_preupdate_reward_and_frozen_target(self):
        bonus, prediction = rnd_step(0, 2)
        self.assertEqual(bonus, 4.); self.assertEqual(prediction, .5)
        self.assertEqual(rnd_step(prediction, 2)[0], 2.25)
        self.assertEqual(rnd_step(2, 2), (0., 2.))

    def test_count_progress_recovery(self):
        self.assertEqual(count_bonus(3), .5)
        self.assertEqual(progress_score(4, 4), 0)
        self.assertEqual(recovery_filter(1, [.99,.2], .9), 0)
        self.assertIsNone(recovery_filter(1, [.1,.2], .9))

    def test_architecture_targets_snapshot_and_ratio(self):
        agent = ModularAgent(states=2, alpha=.1, eta=.1)
        agent.q = [[1.,2.], [3.,4.]]
        agent.gvf = [2.,5.]; agent.rate = .5
        result = agent.observe(0,1,1,1,.25)
        self.assertEqual(result["control_error"], 2.5)
        self.assertEqual(result["prediction_error"], 3.)
        self.assertEqual(result["ratio"], 4.)
        self.assertAlmostEqual(agent.gvf[0], 3.2)
        self.assertAlmostEqual(agent.rate, .525)

    def test_planning_cannot_update_real_data_rate(self):
        a, b = ModularAgent(states=2), ModularAgent(states=2)
        ra = a.observe(0,1,1,1,.5,0)
        rb = b.observe(0,1,1,1,.5,5)
        self.assertEqual(a.rate, b.rate)
        self.assertEqual(a.gvf, b.gvf)
        self.assertEqual(rb["planning_updates"], 5)
        self.assertNotEqual(a.q, b.q)

    def test_probabilities_full_support(self):
        probs = ModularAgent().action_probabilities(0)
        self.assertAlmostEqual(sum(probs), 1.)
        self.assertGreater(min(probs), 0.)

    def test_model_counts_and_planning_budget(self):
        agent = ModularAgent(states=2)
        agent.observe(0, 1, 1, 1, .5, 2)
        agent.observe(0, 1, 0, 0, .5, 3)
        self.assertEqual(agent.model[0, 1], {(1, 1): 1, (0, 0): 1})
        self.assertEqual(agent.planning_cursor, 5)


DEMOS = {"average": average_demo, "streaming": streaming_demo,
         "retention": retention_demo, "plasticity": plasticity_demo,
         "exploration": exploration_demo, "architectures": architectures_demo}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter", choices=[*DEMOS, "all", "test"])
    args = parser.parse_args()
    if args.chapter == "test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(MechanismTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    elif args.chapter == "all":
        for demo in DEMOS.values(): demo()
    else:
        DEMOS[args.chapter]()


if __name__ == "__main__":
    main()
