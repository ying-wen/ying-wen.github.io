"""Small independent checks of equations discussed in the textbook.

These are mathematical diagnostics, not upstream training reproductions.
"""
import math
import unittest


def solve_small(matrix, rhs):
    """Independent pivoted elimination for the three-state textbook example."""
    rows = [list(row)+[b] for row,b in zip(matrix, rhs)]
    n = len(rhs)
    for i in range(n):
        pivot = max(range(i, n), key=lambda j: abs(rows[j][i]))
        rows[i], rows[pivot] = rows[pivot], rows[i]
        scale = rows[i][i]
        if abs(scale) < 1e-12:
            raise ValueError('singular test system')
        rows[i] = [v/scale for v in rows[i]]
        for j in range(n):
            if j != i:
                amount = rows[j][i]
                rows[j] = [x-amount*y for x,y in zip(rows[j], rows[i])]
    return [row[-1] for row in rows]


def corridor_prediction(p, gamma, centering=0.):
    # This is the analytical three-state example, not the separate running
    # continuing-information-recovery implementation.
    transition = [[0., p, 1-p], [.1, .9, 0.], [.1, 0., .9]]
    reward = [4*p, 0., 1.]
    matrix = [[float(i == j)-gamma*transition[i][j] for j in range(3)]
              for i in range(3)]
    return solve_small(matrix, [r-centering for r in reward])


class ReadingDerivations(unittest.TestCase):
    def test_average_lambda_operator_keeps_the_poisson_fixed_point(self):
        p = .4
        transition = [[0., p, 1-p], [.1, .9, 0.], [.1, 0., .9]]
        reward, stationary = [4*p, 0., 1.], [1/11, 10*p/11, 10*(1-p)/11]
        gain = sum(d*r for d,r in zip(stationary, reward))
        bias = [0., -10*gain, 10*(1-gain)]
        offset = sum(d*h for d,h in zip(stationary, bias))
        bias = [h-offset for h in bias]  # d^T h = 0, not h(0) = 0
        self.assertAlmostEqual(sum(d*h for d,h in zip(stationary, bias)), 0.)
        ph = [sum(a*h for a,h in zip(row, bias)) for row in transition]
        for h,r,next_h in zip(bias, reward, ph):
            self.assertAlmostEqual(gain+h, r+next_h)
        # T_lambda h = (I-lambda P)^-1 [r-g+(1-lambda)Ph].
        # Lambda mixes undiscounted n-step targets; it does not discount reward.
        for decay in (0., .3, .9, .99):
            matrix = [[float(i == j)-decay*transition[i][j]
                       for j in range(3)] for i in range(3)]
            rhs = [r-gain+(1-decay)*next_h for r,next_h in zip(reward, ph)]
            result = solve_small(matrix, rhs)
            for actual, expected in zip(result, bias):
                self.assertAlmostEqual(actual, expected, places=10)

    def test_discount_and_average_reward_can_reverse_policy_order(self):
        va, vb = corridor_prediction(1., .5), corridor_prediction(0., .5)
        self.assertAlmostEqual(va[0], 88/21)
        self.assertAlmostEqual(vb[0], 20/21)
        self.assertGreater(va[0], vb[0])
        self.assertLess(4/11, 10/11)  # long-run gains select the other policy
        self.assertAlmostEqual(corridor_prediction(1., .9)[0], 760/109)
        self.assertAlmostEqual(corridor_prediction(0., .9)[0], 900/109)
        self.assertAlmostEqual(corridor_prediction(1., 20/23)[0],
                               corridor_prediction(0., 20/23)[0])

    def test_centering_finite_discount_does_not_create_differential_value(self):
        p, gamma = .4, .5
        gain = (10-6*p)/11
        v = corridor_prediction(p, gamma)
        u = corridor_prediction(p, gamma, gain)
        for vi, ui in zip(v, u):
            self.assertAlmostEqual(ui, vi-gain/(1-gamma))
        h = [0., -10*gain, 10*(1-gain)]  # Poisson solution with h(0)=0
        self.assertNotAlmostEqual(u[1]-u[0], h[1])
        almost_one = corridor_prediction(p, .999999)
        for i in range(3):
            self.assertAlmostEqual(almost_one[i]-almost_one[0], h[i], places=3)

    def test_discounted_critic_can_reverse_average_actor_direction(self):
        p, eps = .4, 1e-6
        gain = lambda z: (10-6*z)/11
        derivative = (gain(p+eps)-gain(p-eps))/(2*eps)
        self.assertAlmostEqual(derivative, -6/11, places=9)
        v = corridor_prediction(p, .5)
        discounted_q_difference = 4.+.5*(v[1]-v[2])
        state_zero_mass = 1/11
        surrogate_direction = state_zero_mass*discounted_q_difference
        self.assertAlmostEqual(surrogate_direction, 34/121)
        self.assertLess(derivative, 0.)
        self.assertGreater(surrogate_direction, 0.)
        # Subtracting a common exact gain changes no action advantage here.
        u = corridor_prediction(p, .5, gain(p))
        self.assertAlmostEqual(4.+.5*(u[1]-u[2]), discounted_q_difference)

    def test_gvf_return_is_not_the_external_reward_return(self):
        # One physical trajectory, four explicitly different questions.
        # The charge-arrival event ends auxiliary returns, not the world.
        energy, delivered, charge = [2., 1., 1.], [0., 0., 1.], [0., 1., 0.]
        rewards = [10*d-e for d,e in zip(delivered, energy)]
        def gvf(cumulants, continuations):
            total, weight = 0., 1.
            for c, gamma in zip(cumulants, continuations):
                total += weight*c
                weight *= gamma
            return total, weight
        stopping = [1-a for a in charge]
        self.assertEqual(gvf(energy, stopping), (3., 0.))
        self.assertEqual(gvf([1., 1., 1.], stopping), (2., 0.))
        self.assertEqual(gvf(charge, stopping), (1., 0.))
        partial, tail = gvf(rewards, [.9]*3)
        self.assertAlmostEqual(partial, 4.39)
        self.assertAlmostEqual(tail, .729)  # still multiplies the unknown future
        self.assertAlmostEqual(gvf(charge, [.9*(1-a) for a in charge])[0], .9)

    def test_gvf_termination_removes_tail_not_arrival_signal(self):
        def update(value, next_value, cumulant, continuation):
            delta = cumulant+continuation*next_value-value
            return value+.1*delta, delta
        main, delta_main = update(1., 5., -1., .9)
        energy, delta_energy = update(2., 100., 1., 0.)
        hit, _ = update(.4, 100., 1., 0.)
        self.assertAlmostEqual(main, 1.25)
        self.assertAlmostEqual(energy, 1.9)
        self.assertAlmostEqual(hit, .46)
        self.assertGreater(delta_main, 0.)  # negative reward, positive TD error
        self.assertLess(delta_energy, 0.)  # positive cumulant, negative TD error

    def test_option_subtask_reward_is_not_the_planning_reward(self):
        external, gamma, next_value = [-1., -1.], .9, 10.
        reward_model = sum(gamma**i*r for i,r in enumerate(external))
        endpoint_kernel = gamma**len(external)
        main_target = reward_model+endpoint_kernel*next_value
        subtask_return = -.01+.95*1.
        self.assertAlmostEqual(reward_model, -1.9)
        self.assertAlmostEqual(endpoint_kernel, .81)
        self.assertAlmostEqual(main_target, 6.2)
        self.assertAlmostEqual(2.+.1*(main_target-2.), 2.42)
        self.assertAlmostEqual(subtask_return, .94)
        self.assertNotAlmostEqual(subtask_return, main_target)
        self.assertNotAlmostEqual(reward_model+gamma*endpoint_kernel*next_value, main_target)
        # Ending an option retains future value; ending the environment does not.
        self.assertNotEqual(reward_model, main_target)

    def test_discounted_endpoint_gvf_counts_the_last_transition_discount(self):
        gamma = .9
        for duration in (1, 2, 5):
            answer, weight = 0., 1.
            for t in range(duration):
                stopped = float(t == duration-1)
                # B is the sampled stopping event, not environment termination.
                answer += weight*gamma*stopped
                weight *= gamma*(1-stopped)
            self.assertAlmostEqual(answer, gamma**duration)
            self.assertEqual(weight, 0.)

    def test_potential_shaping_keeps_its_finite_trajectory_boundary(self):
        phi, gamma = [2., -1., 5., 3.], .9
        change = sum(gamma**t*(gamma*phi[t+1]-phi[t]) for t in range(3))
        self.assertAlmostEqual(change, -phi[0]+gamma**3*phi[3])
        self.assertNotAlmostEqual(change, -phi[0])
        # The remaining boundary disappears only under the applicable
        # infinite discounted limit or appropriate terminal-potential rule.

    def test_termination_advantage_matches_arrival_mixture_derivative(self):
        sigmoid = lambda x: 1/(1+math.exp(-x))
        h, eps, q, v = .3, 3e-6, 2., 5.
        local_value = lambda x: (1-sigmoid(x))*q+sigmoid(x)*v
        fd = (local_value(h+eps)-local_value(h-eps))/(2*eps)
        gradient = -sigmoid(h)*(1-sigmoid(h))*(q-v)
        self.assertAlmostEqual(fd, gradient, places=9)
        self.assertGreater(gradient, 0.)
        # With a greedy high-level policy and a re-available current option,
        # the continuation advantage cannot be positive.
        for values in ([2., 5.], [7., 5.], [2., 2.]):
            self.assertLessEqual(values[0]-max(values), 0.)

    def test_double_sampling_requires_outer_state_average(self):
        # Two current states have different conditional means/products.
        d = [.2, .8]
        means_delta, means_direction = [1., 3.], [2., -1.]
        correct = sum(p*a*b for p,a,b in zip(d, means_delta, means_direction))
        wrong = sum(p*a for p,a in zip(d, means_delta))*sum(p*b for p,b in zip(d, means_direction))
        self.assertAlmostEqual(correct, -2.)
        self.assertNotAlmostEqual(correct, wrong)

    def test_nonlinear_tangent_objective_hessian_correction(self):
        # Smooth one-parameter family; exact enumeration, no Monte Carlo noise.
        x, y, d, rewards = [1., -.4], [.3, 1.], [.4, .6], [.2, -.1]
        p, gamma = [[.8, .2], [.3, .7]], .8
        def moments(theta):
            t = math.tanh(theta)
            v = [theta*theta*a+t*b for a,b in zip(x,y)]
            phi = [2*theta*a+(1-t*t)*b for a,b in zip(x,y)]
            hess = [2*a-2*t*(1-t*t)*b for a,b in zip(x,y)]
            c = sum(ds*f*f for ds,f in zip(d,phi))
            delta = [[rewards[s]+gamma*v[sp]-v[s] for sp in range(2)] for s in range(2)]
            b = sum(d[s]*p[s][sp]*delta[s][sp]*phi[s] for s in range(2) for sp in range(2))
            return phi,hess,delta,b,c
        theta, eps = .7, 1e-6
        def objective(t):
            *_,b,c = moments(t)
            return .5*b*b/c
        phi,hess,delta,b,c = moments(theta)
        u = b/c
        negative_gradient = sum(d[s]*p[s][sp]*(
            (phi[s]-gamma*phi[sp])*phi[s]*u
            -(delta[s][sp]-phi[s]*u)*hess[s]*u)
            for s in range(2) for sp in range(2))
        finite_difference = (objective(theta+eps)-objective(theta-eps))/(2*eps)
        self.assertAlmostEqual(negative_gradient, -finite_difference, places=8)

    def test_lambda_return_alignment_counterexample(self):
        # Nonterminal two-step rollout, rewards 0, Q(s1)=2, Q(s2)=10.
        # Compare the standard recursion with a carry initialized at s2 when
        # the last transition was already removed from the reversed scan.
        gamma, lam = .9, .5
        tail = gamma*10.
        correct = gamma*((1-lam)*2.+lam*tail)
        misaligned = gamma*10.+gamma*lam*(tail-10.)
        self.assertAlmostEqual(correct, 4.95)
        self.assertAlmostEqual(misaligned, 8.55)

    def test_smdp_rate_is_total_reward_over_total_time(self):
        rewards, durations = [1., 4.], [1., 2.]
        rate = sum(rewards)/sum(durations)
        ratios = [r/t for r,t in zip(rewards,durations)]
        simple_harmonic = len(ratios)/sum(1/z for z in ratios)
        weighted_arithmetic = sum(t/sum(durations)*z for t,z in zip(durations,ratios))
        self.assertAlmostEqual(rate, 5/3)
        self.assertAlmostEqual(rate, weighted_arithmetic)
        self.assertAlmostEqual(simple_harmonic, 4/3)
        # No independence assumption was needed for the exact finite sums.


if __name__ == '__main__':
    unittest.main()
