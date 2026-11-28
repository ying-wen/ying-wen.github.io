"""Analytic checks, not claims of general algorithm performance."""
import math
import unittest

from implementations import runtime
from implementations.nonlinear_diagnostics import (
    _common as c, baird_expected_td as td, baird_residual_gradient as rg,
    double_sample_residual as double, single_sample_residual as single,
    shared_nonlinear as shared, isolated_nonlinear as isolated,
    online_nonlinear_td as online, lagged_nonlinear_td as lagged,
)


def derivative(f, w, j, epsilon=1e-6):
    plus, minus = w[:], w[:]
    plus[j] += epsilon
    minus[j] -= epsilon
    return (f(plus) - f(minus)) / (2 * epsilon)


class BairdTests(unittest.TestCase):
    def test_features_have_unit_norm(self):
        for x in c.BAIRD_X:
            self.assertAlmostEqual(c.dot(x, x), 1.0)

    def test_target_true_value_zero(self):
        self.assertEqual(td.direction([0.0] * 8), [0.0] * 8)
        self.assertEqual(rg.objective([0.0] * 8), 0.0)

    def test_expected_importance_sampling_equals_enumeration(self):
        # For every s: b(solid)=1/7, pi(solid)=1, rho=7.
        # b(dashed)=6/7, pi(dashed)=0, so its weighted contribution vanishes.
        w = c.baird_init(3)
        actual = td.direction(w)
        expected = [0.0] * 8
        for x in c.BAIRD_X:
            delta = c.BAIRD_GAMMA * c.dot(c.BAIRD_X[6], w) - c.dot(x, w)
            for j in range(8):
                expected[j] += (1 / 7) * (1 / 7) * 7 * delta * x[j]
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b)

    def test_residual_gradient_matches_finite_difference(self):
        w = c.baird_init(2)
        for j, direction in enumerate(rg.direction(w)):
            self.assertAlmostEqual(direction, -derivative(rg.objective, w, j), places=8)

    def test_residual_step_decreases_its_objective(self):
        w = c.baird_init(0)
        for _ in range(50):
            new = rg.update(w)
            self.assertLessEqual(rg.objective(new), rg.objective(w) + 1e-12)
            w = new

    def test_linear_td_failure_is_preserved_not_clipped(self):
        rows = td.run(0, 1200)
        self.assertGreater(rows[-1]["rmse"], 100 * rows[0]["rmse"])
        self.assertAlmostEqual(rows[-1]["value"], math.log10(1 + rows[-1]["rmse"]))

    def test_td_is_not_residual_gradient(self):
        w = c.baird_init(0)
        self.assertGreater(sum((a-b)**2 for a, b in zip(td.direction(w), rg.direction(w))), 1e-3)


class DoubleSamplingTests(unittest.TestCase):
    def test_independent_expectation_is_negative_msbe_gradient(self):
        for w in (-2.0, 0.0, 1.0, 2.0, 5.0):
            expected = sum(double.direction(w, x, y) / 4 for x in (0.0, 2.0) for y in (0.0, 2.0))
            self.assertAlmostEqual(expected, 0.5 - 0.25*w)

    def test_same_successor_bias_and_distinct_optima(self):
        for w in (0.0, 1.0, 2.0, 5.0):
            expected = sum(single.direction(w, x) / 2 for x in (0.0, 2.0))
            self.assertAlmostEqual(expected, 0.5 - 0.5*w)
        self.assertEqual(sum(single.direction(1.0, x)/2 for x in (0.0, 2.0)), 0.0)
        self.assertEqual(single.objective(1.0), 0.125)
        self.assertEqual(double.objective(2.0), 0.0)

    def test_correlated_resampling_does_not_fix_bias(self):
        self.assertAlmostEqual(sum(double.direction(2.0, x, x)/2 for x in (0.0, 2.0)), -0.5)

    def test_equal_model_query_budget(self):
        for algorithm in (single, double):
            rows = algorithm.run(2, 100)
            self.assertEqual(rows[-1]["successor_queries"], 200)
        runtime.comparable([single.META, double.META])


class NonlinearTests(unittest.TestCase):
    def test_smooth_network_jacobian(self):
        w = [0.8, -0.4, 0.3, -0.7]
        for x in (-1.0, 1.0, 2.0):
            for j, g in enumerate(c.gradient(w, x)):
                self.assertAlmostEqual(g, derivative(lambda z: c.value(z, x), w, j), places=8)

    def test_shared_loss_gradient_finite_difference(self):
        w = c.nonlinear_init(0)
        for context in (0, 1):
            target = 0.5 if context == 0 else -0.5
            f = lambda z: 0.5*(c.value(z, context+1)-target)**2
            for j, g in enumerate(shared.loss_gradient(w, context)):
                self.assertAlmostEqual(g, derivative(f, w, j), places=8)

    def test_opposing_gradients_increase_unobserved_loss_locally(self):
        w = c.nonlinear_init(0)
        alignment = c.dot(shared.loss_gradient(w, 0), shared.loss_gradient(w, 1))
        self.assertLess(alignment, 0.0)
        alpha = 1e-5
        new = shared.update(w, 1.0, 0.5, alpha)
        increase = 0.5*(c.value(new, 2.0)+0.5)**2 - 0.5*(c.value(w, 2.0)+0.5)**2
        self.assertAlmostEqual(increase / alpha, -alignment, places=5)

    def test_isolation_has_zero_cross_update_but_more_parameters(self):
        rows = isolated.run(3, 90)
        self.assertTrue(all(row.get("unobserved_prediction_drift", 0.0) == 0.0 for row in rows))
        self.assertEqual(rows[-1]["parameter_count"], 8)
        self.assertEqual(shared.run(3, 90)[-1]["parameter_count"], 4)

    def test_jacobian_changes_when_representation_changes(self):
        old = c.nonlinear_init(0)
        new = shared.update(old, 1.0, 0.5, alpha=0.5)
        self.assertGreater(sum((a-b)**2 for a, b in zip(c.gradient(old, 1.0), c.gradient(new, 1.0))), 1e-4)

    def test_two_state_analytic_values(self):
        va, vb = 20 / 9, 25 / 9
        self.assertAlmostEqual(va, 0.8*vb)
        self.assertAlmostEqual(vb, 1.0+0.8*va)

    def test_semigradient_is_fixed_target_loss_gradient(self):
        w, alpha = c.nonlinear_init(0), 0.03
        new, target = online.update(w, -1.0, 0.0, 1.0, alpha)
        for j in range(4):
            fixed_gradient = derivative(lambda z: 0.5*(target-c.value(z, -1.0))**2, w, j)
            self.assertAlmostEqual((new[j]-w[j])/alpha, -fixed_gradient, places=8)
        residual_gradient = [derivative(lambda z: 0.5*(0.8*c.value(z, 1.0)-c.value(z, -1.0))**2, w, j) for j in range(4)]
        self.assertGreater(sum(((new[j]-w[j])/alpha+residual_gradient[j])**2 for j in range(4)), 1e-5)

    def test_lagged_target_is_not_mutated(self):
        w, target_w = c.nonlinear_init(2), c.nonlinear_init(3)
        snapshot = target_w[:]
        new, target = lagged.update(w, target_w, -1.0, 0.0, 1.0)
        self.assertEqual(target_w, snapshot)
        self.assertNotEqual(new, w)
        self.assertEqual(target, 0.8*c.value(snapshot, 1.0))

    def test_copy_clock_and_logged_target_age(self):
        rows = lagged.run(0, 51)
        for row in rows:
            self.assertEqual(row["target_age"], row["step"] % 50)
            self.assertEqual(row["target_copies"], row["step"] // 50)
            if row["step"] % 50:
                self.assertEqual(row["target_drift"], 0.0)
        self.assertGreater(next(row for row in rows if row["step"] == 50)["target_drift"], 0.0)


class RuntimeTests(unittest.TestCase):
    def test_every_entry_is_reproducible_finite_and_complete(self):
        modules = (td, rg, double, single, shared, isolated, online, lagged)
        for module in modules:
            with self.subTest(module=module.META["id"]):
                rows = module.run(4, 81)
                runtime.validate_rows(rows, 81)
                self.assertEqual(rows, module.run(4, 81))
                self.assertEqual(module.META["implementation_kind"], "component_experiment")
                runtime.comparable([module.META, runtime.load(module.META["baseline"]).META])
                with self.assertRaises(ValueError):
                    module.run(0, 0)


if __name__ == "__main__":
    unittest.main()
