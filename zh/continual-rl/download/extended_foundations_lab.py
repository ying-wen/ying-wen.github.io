"""Seven parallel RL problem axes: independent standard-library mechanisms.

python3 extended_foundations_lab.py demo
python3 extended_foundations_lab.py test

Original small numerical examples, not neural benchmark reproductions.
"""
import argparse
import itertools
import json
import math
import random
import unittest


def dot(x, y):
    return sum(a*b for a, b in zip(x, y))


def probabilities(values):
    if not values or any(v < 0 for v in values) or not math.isclose(sum(values), 1., abs_tol=1e-10):
        raise ValueError('a normalized probability vector is required')


# BEGIN partial_observability
def belief_update(prior, transition, likelihood):
    """Transition[s][s_next], likelihood[s_next] for the observation received."""
    probabilities(prior)
    n = len(prior)
    if len(transition) != n or len(likelihood) != n or any(len(row) != n for row in transition):
        raise ValueError('matching state dimensions required')
    for row in transition:
        probabilities(row)
    if any(not 0 <= p <= 1 for p in likelihood):
        raise ValueError('observation likelihood must be in [0,1]')
    predicted = [sum(prior[s]*transition[s][sp] for s in range(n)) for sp in range(n)]
    evidence = dot(predicted, likelihood)
    if evidence <= 0:
        raise ValueError('impossible observation under this model')
    return [p*l/evidence for p, l in zip(predicted, likelihood)], evidence


def information_value(prior, sensor, rewards, sensing_cost=0.):
    """One sensing step, then choose one terminal action.

    sensor[state][observation]; rewards[action][state]; hidden state is static.
    """
    probabilities(prior)
    for row in sensor:
        probabilities(row)
    before = max(dot(prior, r) for r in rewards)
    after = 0.
    for o in range(len(sensor[0])):
        joint = [p*row[o] for p, row in zip(prior, sensor)]
        evidence = sum(joint)
        if evidence:
            posterior = [x/evidence for x in joint]
            after += evidence*max(dot(posterior, r) for r in rewards)
    return before, after-sensing_cost, after-sensing_cost-before
# END partial_observability


# BEGIN exploration
def beta_posterior(successes, failures, alpha=1., beta=1.):
    a, b = alpha+successes, beta+failures
    if min(a, b) <= 0 or min(successes, failures) < 0:
        raise ValueError('positive prior and nonnegative counts required')
    return a, b, a/(a+b), a*b/((a+b)**2*(a+b+1))


def thompson_action(counts, seed):
    rng = random.Random(seed)
    samples = [rng.betavariate(*beta_posterior(s, f)[:2]) for s, f in counts]
    return max(range(len(samples)), key=samples.__getitem__), samples


def ucb_scores(means, counts, time):
    if time < 1 or len(means) != len(counts) or any(n < 0 for n in counts):
        raise ValueError('valid counts and time required')
    return [float('inf') if n == 0 else q+math.sqrt(2*math.log(time)/n)
            for q, n in zip(means, counts)]


def coherent_chain_probability(length):
    """Only all-right succeeds; unbiased independent actions vs one episode head."""
    if length < 1:
        raise ValueError('positive chain length required')
    return 0.5**length, 0.5
# END exploration


# BEGIN distributional
def categorical_projection(reward, gamma, terminal, support, masses):
    probabilities(masses)
    n = len(support)
    if n < 2 or len(masses) != n or not 0 <= gamma <= 1:
        raise ValueError('two or more atoms and valid discount required')
    spacing = (support[-1]-support[0])/(n-1)
    if spacing <= 0 or any(not math.isclose(z, support[0]+i*spacing) for i, z in enumerate(support)):
        raise ValueError('equally spaced ascending support required')
    output = [0.]*n
    for atom, mass in zip(support, masses):
        target = reward+(0. if terminal else gamma*atom)
        target = min(support[-1], max(support[0], target))
        position = min(n-1, max(0., (target-support[0])/spacing))
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            output[lower] += mass
        else:
            output[lower] += mass*(upper-position)
            output[upper] += mass*(position-lower)
    return output


def quantile_loss_gradient(theta, samples, tau):
    if not samples or not 0 < tau < 1:
        raise ValueError('nonempty samples and interior quantile required')
    loss = sum((tau-float(y-theta < 0))*(y-theta) for y in samples)/len(samples)
    # At equality this selects one valid subgradient.
    gradient = sum(float(y < theta)-tau for y in samples)/len(samples)
    return loss, gradient


def lower_cvar(values, masses, fraction):
    probabilities(masses)
    if len(values) != len(masses) or not 0 < fraction <= 1:
        raise ValueError('valid mass fraction required')
    remaining, total = fraction, 0.
    for value, mass in sorted(zip(values, masses)):
        used = min(mass, remaining)
        total += used*value
        remaining -= used
        if remaining <= 1e-14:
            break
    return total/fraction
# END distributional


# BEGIN offline
def importance_ratio(target, behavior):
    if not 0 <= target <= 1 or not 0 < behavior <= 1:
        raise ValueError('positive recorded behavior probability required')
    return target/behavior


def per_decision_is(rewards, ratios, gamma=1.):
    if len(rewards) != len(ratios):
        raise ValueError('one ratio per reward required')
    product, estimate = 1., 0.
    for t, (reward, ratio) in enumerate(zip(rewards, ratios)):
        product *= ratio
        estimate += gamma**t*product*reward
    return estimate


def sequential_dr(rewards, ratios, q_hat, v_hat, gamma=1.):
    """v_hat contains estimates at each state and a terminal zero."""
    n = len(rewards)
    if len(ratios) != n or len(q_hat) != n or len(v_hat) != n+1 or v_hat[-1] != 0:
        raise ValueError('complete finite trajectory and zero terminal value required')
    estimate = 0.
    for t in reversed(range(n)):
        estimate = v_hat[t]+ratios[t]*(rewards[t]+gamma*estimate-q_hat[t])
    return estimate


def expectile(values, tau, weights=None):
    if not values or not 0 < tau < 1:
        raise ValueError('nonempty values and interior expectile required')
    weights = [1.]*len(values) if weights is None else weights
    if len(weights) != len(values) or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError('nonnegative observation weights required')
    low, high = min(values), max(values)
    for _ in range(100):
        middle = (low+high)/2
        derivative_sign = sum(w*(tau if q >= middle else 1-tau)*(q-middle)
                              for q, w in zip(values, weights))
        if derivative_sign > 0:
            low = middle
        else:
            high = middle
    return (low+high)/2


def iql_weighted_actor(q_values, counts, tau=.75, inverse_temperature=1.):
    """Exact weighted behavioral-cloning solution for a one-state categorical actor."""
    value = expectile(q_values, tau, counts)
    logits = [inverse_temperature*(q-value) for q in q_values]
    offset = max(z for n, z in zip(counts, logits) if n > 0)
    weights = [n*math.exp(z-offset) if n > 0 else 0. for n, z in zip(counts, logits)]
    return value, [w/sum(weights) for w in weights]


def cql_penalty_gradient(q_values, data_probabilities):
    probabilities(data_probabilities)
    if len(q_values) != len(data_probabilities):
        raise ValueError('matching action dimensions required')
    maximum = max(q_values)
    exponentials = [math.exp(q-maximum) for q in q_values]
    normalizer = sum(exponentials)
    loss = maximum+math.log(normalizer)-dot(q_values, data_probabilities)
    return loss, [x/normalizer-p for x, p in zip(exponentials, data_probabilities)]
# END offline


# BEGIN model_based
def fit_scalar_gain(actions, increments):
    if len(actions) != len(increments) or not actions:
        raise ValueError('paired action and state-increment samples required')
    denominator = dot(actions, actions)
    if denominator == 0:
        raise ValueError('no excitation: gain is not identifiable')
    return dot(actions, increments)/denominator


def model_rollout(state, sequence, gain=1., drift=0.):
    states = []
    for action in sequence:
        state = state+gain*action+drift
        states.append(state)
    return states


def mpc_action(state, goal, actions=(-1., 0., 1.), horizon=2,
               gain=1., action_cost=.1, terminal_weight=0.):
    """Enumerate open-loop candidates; return only the first action to execute."""
    if horizon < 1 or not actions:
        raise ValueError('positive horizon and nonempty actions required')
    best = None
    for sequence in itertools.product(actions, repeat=horizon):
        states = model_rollout(state, sequence, gain)
        cost = sum((s-goal)**2+action_cost*a*a for s, a in zip(states, sequence))
        cost += terminal_weight*(states[-1]-goal)**2
        if best is None or cost < best[0]:
            best = cost, sequence, states
    return best[1][0], best


def rollout_error_bound(one_step_error, lipschitz, horizon):
    """Same actions, same initial state; error_{k+1} <= L error_k + epsilon."""
    if min(one_step_error, lipschitz, horizon) < 0:
        raise ValueError('nonnegative inputs required')
    error, errors = 0., []
    for _ in range(horizon):
        error = lipschitz*error+one_step_error
        errors.append(error)
    return errors


def imagined_lambda_return(rewards, values, gamma=.9, lam=.8):
    """values[k] = V(s_k); final value supplies the finite imagination tail."""
    if len(values) != len(rewards)+1:
        raise ValueError('one more state value than reward required')
    carry, returns = values[-1], []
    for k in reversed(range(len(rewards))):
        carry = rewards[k]+gamma*((1-lam)*values[k+1]+lam*carry)
        returns.append(carry)
    return list(reversed(returns))
# END model_based


# BEGIN constraints
def occupancy_residual(occupancy, transition, initial, gamma):
    """Normalized discounted occupancy x[s][a] and P[s][a][next_state]."""
    probabilities(initial)
    if not 0 <= gamma < 1:
        raise ValueError('discount below one required')
    n = len(initial)
    return [sum(occupancy[s])-(1-gamma)*initial[s]
            -gamma*sum(occupancy[sp][a]*transition[sp][a][s]
                       for sp in range(n) for a in range(len(occupancy[sp])))
            for s in range(n)]


def constrained_two_action(rewards, costs, budget):
    """One-state normalized occupancy LP; costs[1] > costs[0]."""
    if costs[1] <= costs[0]:
        raise ValueError('ordered distinct costs required')
    if budget < costs[0]:
        raise ValueError('infeasible budget')
    largest = min(1., (budget-costs[0])/(costs[1]-costs[0]))
    p = largest if rewards[1] > rewards[0] else 0.
    return p, (1-p)*rewards[0]+p*rewards[1], (1-p)*costs[0]+p*costs[1]


def primal_dual_step(logit, multiplier, reward_gap, cost0, cost_gap, budget,
                     actor_step=.1, dual_step=.1):
    probability = 1/(1+math.exp(-logit))
    gradient = probability*(1-probability)*(reward_gap-multiplier*cost_gap)
    cost = cost0+cost_gap*probability
    # Both directions use the same old policy and multiplier.
    return logit+actor_step*gradient, max(0., multiplier+dual_step*(cost-budget))
# END constraints


# BEGIN multi_agent
def matrix_value(matrix, row_policy, column_policy):
    probabilities(row_policy); probabilities(column_policy)
    return sum(row_policy[i]*column_policy[j]*matrix[i][j]
               for i in range(len(row_policy)) for j in range(len(column_policy)))


def minimax_2x2(matrix):
    """Row maximizes, column minimizes. Optimize the lower envelope of two lines."""
    a, b = matrix[0]
    c, d = matrix[1]
    candidates = [0., 1.]
    denominator = a-c-b+d
    if denominator != 0:
        crossing = (d-c)/denominator
        if 0 <= crossing <= 1:
            candidates.append(crossing)
    lower = lambda p: min(p*a+(1-p)*c, p*b+(1-p)*d)
    p = max(candidates, key=lower)
    return [p, 1-p], lower(p)


def zero_sum_gap(matrix, row_policy, column_policy):
    row_best = max(dot(row, column_policy) for row in matrix)
    column_best = min(sum(row_policy[i]*matrix[i][j] for i in range(len(matrix)))
                      for j in range(len(matrix[0])))
    return row_best-column_best


def counterfactual_advantage(matrix, row_action, column_action, row_policy):
    probabilities(row_policy)
    baseline = sum(row_policy[i]*matrix[i][column_action] for i in range(len(row_policy)))
    return matrix[row_action][column_action]-baseline


def monotone_joint_greedy(local_values, weights):
    if len(local_values) != len(weights) or any(w < 0 for w in weights):
        raise ValueError('nonnegative mixing weights required')
    local_choice = tuple(max(range(len(q)), key=q.__getitem__) for q in local_values)
    joint = list(itertools.product(*(range(len(q)) for q in local_values)))
    value = lambda acts: sum(w*q[a] for w, q, a in zip(weights, local_values, acts))
    return local_choice, value(local_choice), max(map(value, joint))
# END multi_agent


class Checks(unittest.TestCase):
    def test_bayes_prediction_and_evidence(self):
        posterior, evidence = belief_update([.6, .4], [[.9, .1], [.2, .8]], [.8, .2])
        self.assertAlmostEqual(evidence, .572)
        self.assertAlmostEqual(posterior[0], .496/.572)
        self.assertAlmostEqual(sum(posterior), 1.)

    def test_impossible_observation(self):
        with self.assertRaises(ValueError):
            belief_update([1., 0.], [[1., 0.], [0., 1.]], [0., 1.])

    def test_information_value(self):
        before, after, improvement = information_value([.5, .5], [[.8, .2], [.2, .8]], [[1., -1.], [-1., 1.]], .1)
        self.assertAlmostEqual(before, 0)
        self.assertAlmostEqual(after, .5)
        self.assertAlmostEqual(improvement, .5)

    def test_uninformative_sensor(self):
        self.assertAlmostEqual(information_value([.5, .5], [[.5, .5], [.5, .5]], [[1., -1.], [-1., 1.]], .1)[2], -.1)

    def test_beta_posterior(self):
        a, b, mean, variance = beta_posterior(3, 1)
        self.assertEqual((a, b), (4, 2))
        self.assertAlmostEqual(mean, 2/3)
        self.assertAlmostEqual(variance, 2/63)

    def test_thompson_seed(self):
        self.assertEqual(thompson_action([(3, 1), (1, 3)], 42), thompson_action([(3, 1), (1, 3)], 42))

    def test_ucb_unvisited(self):
        self.assertEqual(ucb_scores([.9, .1], [10, 0], 10)[1], float('inf'))

    def test_coherent_exploration(self):
        self.assertEqual(coherent_chain_probability(10), (1/1024, .5))

    def test_projection_mass_and_mean(self):
        projected = categorical_projection(.5, .5, False, [-1., 0., 1.], [.2, .5, .3])
        for actual, expected in zip(projected, [0., .45, .55]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(sum(projected), 1.)
        self.assertAlmostEqual(dot(projected, [-1., 0., 1.]), .55)

    def test_projection_terminal_atom(self):
        self.assertEqual(categorical_projection(0, .9, True, [-1., 0., 1.], [.2, .5, .3]), [0., 1., 0.])

    def test_projection_clipping(self):
        self.assertEqual(categorical_projection(5, 1, False, [-1., 0., 1.], [.2, .5, .3]), [0., 0., 1.])

    def test_quantile_gradient(self):
        theta, samples, tau, eps = .3, [0., 2.], .75, 1e-6
        _, gradient = quantile_loss_gradient(theta, samples, tau)
        finite = (quantile_loss_gradient(theta+eps, samples, tau)[0]-quantile_loss_gradient(theta-eps, samples, tau)[0])/(2*eps)
        self.assertAlmostEqual(gradient, -.25)
        self.assertAlmostEqual(gradient, finite)

    def test_risk_not_mean(self):
        self.assertEqual(lower_cvar([0., 4.], [.5, .5], .5), 0.)
        self.assertEqual(lower_cvar([2.], [1.], .5), 2.)
        self.assertEqual(lower_cvar([0., 4.], [.5, .5], 1.), 2.)

    def test_cvar_partial_atom(self):
        self.assertAlmostEqual(lower_cvar([0., 4.], [.5, .5], .75), 4/3)

    def test_is_bandit_expectation(self):
        estimates = [per_decision_is([r], [importance_ratio(p, .5)])
                     for r, p in zip([1., 3.], [.8, .2])]
        self.assertAlmostEqual(sum(estimates)/2, 1.4)

    def test_dr_perfect_model(self):
        estimates = [sequential_dr([r], [importance_ratio(p, .5)], [r], [1.4, 0.])
                     for r, p in zip([1., 3.], [.8, .2])]
        self.assertEqual(estimates, [1.4, 1.4])

    def test_dr_equals_is_zero_model(self):
        rewards, ratios = [1., 2.], [2., .5]
        self.assertAlmostEqual(sequential_dr(rewards, ratios, [0., 0.], [0., 0., 0.], .9),
                               per_decision_is(rewards, ratios, .9))

    def test_dr_two_step_expectation(self):
        target = [[.75, .25], [.25, .75]]
        reward = [[1., 2.], [0., 4.]]
        q_model = [[2., 0.], [1., 2.]]
        estimates = []
        for a0, a1 in itertools.product(range(2), repeat=2):
            estimates.append(sequential_dr(
                [reward[0][a0], reward[1][a1]],
                [target[0][a0]/.5, target[1][a1]/.5],
                [q_model[0][a0], q_model[1][a1]], [1.5, 1.75, 0.], gamma=.5))
        # Four equally likely behavior trajectories; direct target expectation.
        oracle = .75*1+.25*2+.5*(.25*0+.75*4)
        self.assertAlmostEqual(sum(estimates)/4, oracle)

    def test_missing_behavior_support(self):
        with self.assertRaises(ValueError):
            importance_ratio(.2, 0.)

    def test_expectile_not_quantile(self):
        self.assertAlmostEqual(expectile([0., 2.], .75), 1.5)
        self.assertAlmostEqual(expectile([0., 2.], .5), 1.)

    def test_iql_weighted_clone(self):
        value, policy = iql_weighted_actor([0., 2.], [1., 1.])
        self.assertAlmostEqual(value, 1.5)
        self.assertAlmostEqual(policy[1], 1/(1+math.exp(-2)))
        self.assertEqual(iql_weighted_actor([0., 2.], [1., 0.])[1], [1., 0.])
        self.assertEqual(iql_weighted_actor([0., 10000.], [1., 0.])[1], [1., 0.])

    def test_cql_finite_difference(self):
        q, data, eps = [.2, 1.], [.8, .2], 1e-6
        _, gradient = cql_penalty_gradient(q, data)
        for i in range(2):
            plus, minus = list(q), list(q)
            plus[i] += eps; minus[i] -= eps
            fd = (cql_penalty_gradient(plus, data)[0]-cql_penalty_gradient(minus, data)[0])/(2*eps)
            self.assertAlmostEqual(gradient[i], fd)

    def test_mpc_plan(self):
        action, (cost, sequence, states) = mpc_action(0., 2.)
        self.assertEqual(action, 1.)
        self.assertEqual(sequence, (1., 1.))
        self.assertEqual(states, [1., 2.])
        self.assertAlmostEqual(cost, 1.2)

    def test_model_gain_identification(self):
        self.assertAlmostEqual(fit_scalar_gain([-1., 1., 2.], [-.5, .5, 1.]), .5)
        with self.assertRaises(ValueError):
            fit_scalar_gain([0., 0.], [0., 0.])

    def test_model_error_growth(self):
        actual = model_rollout(0., [1.]*3, gain=.5)
        predicted = model_rollout(0., [1.]*3, gain=1.)
        bound = rollout_error_bound(.5, 1., 3)
        self.assertEqual([abs(x-y) for x, y in zip(actual, predicted)], bound)

    def test_imagined_return(self):
        self.assertEqual(imagined_lambda_return([1., 2.], [0., 3., 4.], gamma=1., lam=1.), [7., 6.])
        self.assertEqual(imagined_lambda_return([1., 2.], [0., 3., 4.], gamma=1., lam=0.), [4., 6.])

    def test_occupancy_flow(self):
        self.assertAlmostEqual(occupancy_residual([[.7, .3]], [[[1.], [1.]]], [1.], .9)[0], 0.)

    def test_two_state_occupancy_flow(self):
        residual = occupancy_residual([[0., .5], [.5]],
                                      [[[1., 0.], [0., 1.]], [[0., 1.]]], [1., 0.], .5)
        for value in residual:
            self.assertAlmostEqual(value, 0.)

    def test_constraint_mixture(self):
        p, reward, cost = constrained_two_action([1., 3.], [0., 2.], .6)
        self.assertAlmostEqual(p, .3)
        self.assertAlmostEqual(reward, 1.6)
        self.assertAlmostEqual(cost, .6)

    def test_constraint_infeasible(self):
        with self.assertRaises(ValueError):
            constrained_two_action([1., 3.], [1., 2.], .6)

    def test_dual_direction(self):
        _, multiplier = primal_dual_step(math.log(4), 0., 2., 0., 2., .6)
        self.assertAlmostEqual(multiplier, .1)

    def test_primal_gradient_finite_difference(self):
        theta, multiplier, eps = .3, .7, 1e-6
        objective = lambda z: 2/(1+math.exp(-z))-multiplier*(2/(1+math.exp(-z))-.6)
        updated, _ = primal_dual_step(theta, multiplier, 2, 0, 2, .6, actor_step=1.)
        self.assertAlmostEqual(updated-theta, (objective(theta+eps)-objective(theta-eps))/(2*eps))

    def test_matching_pennies(self):
        policy, value = minimax_2x2([[1., -1.], [-1., 1.]])
        self.assertEqual(policy, [.5, .5])
        self.assertEqual(value, 0.)
        self.assertEqual(zero_sum_gap([[1., -1.], [-1., 1.]], policy, policy), 0.)

    def test_saddle_gap(self):
        self.assertEqual(zero_sum_gap([[1., -1.], [-1., 1.]], [1., 0.], [1., 0.]), 2.)

    def test_minimax_dominated_row(self):
        self.assertEqual(minimax_2x2([[3., 2.], [1., 0.]]), ([1., 0.], 2.))

    def test_minimax_nonuniform(self):
        policy, value = minimax_2x2([[3., -1.], [0., 2.]])
        self.assertAlmostEqual(policy[0], 1/3)
        self.assertAlmostEqual(value, 1.)

    def test_counterfactual_baseline(self):
        matrix, policy = [[3., 0.], [0., 2.]], [.5, .5]
        self.assertAlmostEqual(counterfactual_advantage(matrix, 0, 0, policy), 1.5)
        self.assertAlmostEqual(sum(policy[i]*counterfactual_advantage(matrix, i, 0, policy) for i in range(2)), 0.)

    def test_monotone_greedy(self):
        choice, greedy, optimal = monotone_joint_greedy([[0., 2.], [3., 1.]], [.5, 2.])
        self.assertEqual(choice, (1, 0))
        self.assertEqual(greedy, optimal)


def demo():
    return {
        'partial_observability': {
            'belief': belief_update([.6, .4], [[.9, .1], [.2, .8]], [.8, .2]),
            'information': information_value([.5, .5], [[.8, .2], [.2, .8]], [[1., -1.], [-1., 1.]], .1)},
        'exploration': {'posterior': beta_posterior(3, 1), 'chain': coherent_chain_probability(10)},
        'distributional': {'projection': categorical_projection(.5, .5, False, [-1., 0., 1.], [.2, .5, .3]),
                           'lower_cvar': lower_cvar([0., 4.], [.5, .5], .5)},
        'offline': {'expectile_and_actor': iql_weighted_actor([0., 2.], [1., 1.]),
                    'dr': sequential_dr([1.], [1.6], [1.], [1.4, 0.])},
        'model_based': {'mpc': mpc_action(0., 2.), 'error': rollout_error_bound(.5, 1., 3)},
        'constraints': {'mixture': constrained_two_action([1., 3.], [0., 2.], .6)},
        'multi_agent': {'minimax': minimax_2x2([[1., -1.], [-1., 1.]])}
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['demo', 'test'])
    args = parser.parse_args()
    if args.command == 'test':
        unittest.main(argv=['extended_foundations_lab.py'], verbosity=2)
    else:
        print(json.dumps(demo(), indent=2))
