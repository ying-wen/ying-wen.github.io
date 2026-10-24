"""Independent checks of average-reward equations, clocks, gauges and coverage."""
import math
import unittest
from implementations import runtime
from implementations.average_systems import _common as c
from implementations.average_systems import differential_td_onpolicy as on
from implementations.average_systems import differential_td_offpolicy as off
from implementations.average_systems import wrong_behavior_mean_td as wrong
from implementations.average_systems import differential_q_multistate as dq
from implementations.average_systems import differential_dyna as dyna
from implementations.average_systems import rvi_multistate as rvi
from implementations.average_systems import unnormalized_vi_multistate as vi


class AverageSystemsTests(unittest.TestCase):
    def test_two_state_poisson_hand_calculation(self):
        # Periodic chain: a gain and Poisson solution still exist.
        g, h = c.evaluate([[1.], [1.]], [[[0., 1.]], [[1., 0.]]], [[0.], [2.]])
        self.assertAlmostEqual(g, 1.)
        self.assertAlmostEqual(h[0], 0.)
        self.assertAlmostEqual(h[1], 1.)

    def test_three_state_solution_residual_and_stationarity(self):
        g, h = c.evaluate(c.TARGET)
        pp = [[sum(c.TARGET[s][a]*c.P[s][a][sp] for a in range(2))
               for sp in range(3)] for s in range(3)]
        # Independent stationary-distribution solve, rather than the gain/bias solve.
        mat = [[pp[j][i]-float(i == j) for j in range(3)] for i in range(2)]
        mat.append([1., 1., 1.])
        d = c.solve(mat, [0., 0., 1.])
        expected = sum(d[s]*sum(c.TARGET[s][a]*c.REWARDS[s][a] for a in range(2))
                       for s in range(3))
        self.assertAlmostEqual(g, expected)
        for s in range(3):
            rp = sum(c.TARGET[s][a]*c.REWARDS[s][a] for a in range(2))
            self.assertAlmostEqual(g+h[s], rp+sum(pp[s][sp]*h[sp] for sp in range(3)))

    def test_differential_update_uses_old_parameters(self):
        h, rate, delta = on.update([1., 3., 0.], .5, 0, 2., 1, alpha=.1, eta=.1)
        self.assertEqual(delta, 3.5)
        self.assertAlmostEqual(h[0], 1.35)
        self.assertAlmostEqual(rate, .535)

    def test_offset_invariance_and_coupled_invariant(self):
        h, shifted = [1., 3., 0.], [101., 103., 100.]
        a, ga, da = off.update(h, .5, 0, 2., 1, 1.4)
        b, gb, db = off.update(shifted, .5, 0, 2., 1, 1.4)
        self.assertAlmostEqual(da, db)
        self.assertAlmostEqual(ga, gb)
        self.assertAlmostEqual(ga-.2*sum(a), .5-.2*sum(h))
        for x, y in zip(a, b):
            self.assertAlmostEqual(y-x, 100.)

    def test_importance_ratio_zero_and_conditional_correction(self):
        h, g, delta = off.update([1., 2., 3.], .5, 0, 2., 1, 0.)
        self.assertEqual(h, [1., 2., 3.])
        self.assertEqual(g, .5)
        target_gain, bias = c.evaluate(c.TARGET)
        for s in range(3):
            expected = 0.
            for a in range(2):
                for sp in range(3):
                    rho = c.TARGET[s][a]/c.BEHAVIOR[s][a]
                    delta = c.REWARDS[s][a]-target_gain+bias[sp]-bias[s]
                    expected += c.BEHAVIOR[s][a]*c.P[s][a][sp]*rho*delta
            self.assertAlmostEqual(expected, 0.)

    def test_missing_behavior_coverage_rejected(self):
        with self.assertRaises(ValueError):
            c.coverage(c.TARGET, [[1., 0.]]*3)

    def test_behavior_rate_negative_control_is_wrong(self):
        gp, h = c.evaluate(c.TARGET)
        gb, _ = c.evaluate(c.BEHAVIOR)
        self.assertGreater(abs(gp-gb), .14)
        # At the true bias, wrong gain leaves nonzero target conditional residual.
        for s in range(3):
            residual = sum(c.TARGET[s][a]*(c.REWARDS[s][a]-gb+
                           sum(c.P[s][a][sp]*h[sp] for sp in range(3))-h[s])
                           for a in range(2))
            self.assertAlmostEqual(residual, gp-gb)
        row = wrong.run(7, 12000)[-1]
        self.assertAlmostEqual(row['gain'], row['experienced_reward_rate'])
        self.assertGreater(row['gain_abs_error'], .1)

    def test_no_terminal_mask_at_run_limit(self):
        # A reporting limit is not an absorbing terminal transition.
        q = [[1., 2.], [3., 4.], [5., 6.]]
        _, _, delta = dq.update(q, .5, 0, 0, 2., 2)
        self.assertEqual(delta, 2.-.5+6.-1.)
        self.assertNotEqual(delta, 2.-.5-1.)

    def test_model_counts_and_expectation_match_sample_mixture(self):
        model = {}
        dyna.model_update(model, 0, 1, 2., 1)
        dyna.model_update(model, 0, 1, 4., 2)
        entry = model[(0, 1)]
        self.assertEqual(entry, {'count': 2, 'reward_sum': 6., 'next': [0, 1, 1]})
        q = [[1., 2.], [3., 4.], [5., 6.]]
        updated, rate, delta = dyna.planning_update(q, .5, (0, 1), entry)
        self.assertEqual(delta, 3.-.5+(4.+6.)/2-2.)
        self.assertAlmostEqual(updated[0][1], 2.+.08*delta)
        self.assertAlmostEqual(rate, .5+.1*.08*delta)

    def test_model_rejects_unobserved_pairs(self):
        with self.assertRaises(ValueError):
            dyna.planning_update([[0., 0.]]*3, 0., (0, 0),
                                 {'count': 0, 'reward_sum': 0., 'next': [0, 0, 0]})

    def test_planning_budget_is_separate_from_real_data(self):
        row = dyna.run(0, 13)[-1]
        self.assertEqual(row['environment_updates'], 13)
        self.assertEqual(row['model_backups'], 65)
        self.assertEqual(row['total_backups'], 78)
        self.assertEqual(rvi.run(0, 13)[-1]['model_backups'], 78)
        self.assertEqual(rvi.run(0, 13)[-1]['environment_updates'], 0)

    def test_rvi_and_unanchored_vi_same_relative_solution(self):
        a, b = [0., 0., 0.], [0., 0., 0.]
        for _ in range(100):
            a, _ = rvi.update(a)
            b, _ = vi.update(b)
            self.assertEqual(a[0], 0.)
            for x, y in zip(a, b):
                self.assertAlmostEqual(x, y-b[0])
        self.assertGreater(b[0], 50.)

    def test_rvi_optimality_equation(self):
        h = [0., 0., 0.]
        for _ in range(150):
            h, g = rvi.update(h)
        best, _, _ = c.optimal()
        self.assertAlmostEqual(g, best)
        for s in range(3):
            target = max(c.REWARDS[s][a]+sum(c.P[s][a][sp]*h[sp] for sp in range(3))
                         for a in range(2))
            self.assertAlmostEqual(g+h[s], target)

    def test_multichain_scalar_gain_not_silently_solved(self):
        with self.assertRaises(ValueError):
            c.evaluate([[1.], [1.]], [[[1., 0.]], [[0., 1.]]], [[0.], [2.]])

    def test_periodic_poisson_solution_does_not_ensure_rvi_convergence(self):
        h, observed = [0., 0.], []
        for _ in range(4):
            backed = [h[1], 2.+h[0]]
            h = [v-backed[0] for v in backed]
            observed.append(h[1])
        self.assertEqual(observed, [2., 0., 2., 0.])

    def test_common_mixing_component_gives_span_contraction(self):
        # The common uniform .2 mass cancels in state differences.
        x, y = [10., -2., 3.], [1., 7., -4.]
        tx, _ = rvi.update(x)
        ty, _ = rvi.update(y)
        delta = [a-b for a, b in zip(x, y)]
        backed_delta = [a-b for a, b in zip(tx, ty)]
        self.assertLessEqual(max(backed_delta)-min(backed_delta),
                             .8*(max(delta)-min(delta))+1e-12)

    def test_all_registered_modules_complete_reproduce_and_compare(self):
        ids = [k for k, p in runtime.discover().items() if p.parent.name == 'average_systems']
        self.assertEqual(len(ids), 8)
        for key in ids:
            module = runtime.load(key)
            rows = module.run(3, 41)
            runtime.validate_rows(rows, 41)
            self.assertEqual(rows, module.run(3, 41))
            runtime.comparable([module.META, runtime.load(module.META['baseline']).META])
            self.assertTrue(all(math.isfinite(row['value']) for row in rows))


if __name__ == '__main__':
    unittest.main()
