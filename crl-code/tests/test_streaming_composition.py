"""Algebra, boundary, protocol and actual-learning checks; no benchmark claims."""
import importlib
import math
import subprocess
import sys
import unittest
from pathlib import Path

from implementations.streaming_composition._common import (
    Network, features, trace_step, finite_lambda_targets, gvf_question,
    gvf_truth, option_target, option_model, option_step, option_truth,
)
from implementations import runtime

IDS = ["gvf_shared_trace", "gvf_frozen_trace", "gvf_separate_trace", "gvf_shared_td0",
       "neural_td_trace", "neural_td0", "neural_gradient_mc",
       "neural_option_smdp", "neural_option_wrong_duration"]


class NeuralDerivativeTests(unittest.TestCase):
    def test_all_parameter_derivatives_both_heads(self):
        n = Network(outputs=2, seed=5)
        for head in range(2):
            _, gradient = n.value_gradient([-.3, .6], head)
            for i, exact in enumerate(gradient):
                old = n.w[i]
                n.w[i] = old+1e-6
                plus = n.value([-.3, .6], head)
                n.w[i] = old-1e-6
                minus = n.value([-.3, .6], head)
                n.w[i] = old
                self.assertAlmostEqual(exact, (plus-minus)/2e-6, places=8)

    def test_freezing_only_trunk(self):
        n = Network(outputs=2)
        value, full = n.value_gradient([.4, .16], 1)
        frozen_value, frozen = n.value_gradient([.4, .16], 1, True)
        self.assertEqual(value, frozen_value)
        self.assertEqual(frozen[:n.trunk_size], [0.]*n.trunk_size)
        self.assertEqual(full[n.trunk_size:], frozen[n.trunk_size:])
        self.assertGreater(sum(x*x for x in full[:n.trunk_size]), 0.)

    def test_semigradient_treats_target_as_constant(self):
        n = Network(seed=7)
        x, xp = [-.4, .16], [.2, .04]
        v, g = n.value_gradient(x)
        vp, gp = n.value_gradient(xp)
        target = .3+.9*vp
        descent = [(target-v)*x for x in g]
        residual_descent = [(target-v)*(a-.9*b) for a, b in zip(g, gp)]
        self.assertGreater(sum((a-b)**2 for a, b in zip(descent, residual_descent)), 1e-6)
        for i in range(len(n.w)):
            old = n.w[i]
            n.w[i] = old+1e-6
            plus = .5*(target-n.value(x))**2
            n.w[i] = old-1e-6
            minus = .5*(target-n.value(x))**2
            n.w[i] = old
            self.assertAlmostEqual(descent[i], -(plus-minus)/2e-6, places=8)

    def test_old_gradient_is_not_current_gradient(self):
        n = Network(seed=9)
        old = n.value_gradient([-.5, .25])[1]
        update = n.value_gradient([.5, .25])[1]
        n.add(update, .4)
        current = n.value_gradient([-.5, .25])[1]
        self.assertGreater(sum((a-b)**2 for a, b in zip(old, current)), 1e-5)


class TraceTests(unittest.TestCase):
    def test_ratio_multiplies_new_and_old_eligibility(self):
        first = trace_step([0., 0.], [1., 0.], 0., .8, 2.)
        second = trace_step(first, [0., 1.], .9, .8, .5)
        self.assertAlmostEqual(second[0], .72)
        self.assertAlmostEqual(second[1], .5)
        self.assertEqual(trace_step(second, [1., 1.], .9, .8, 0.), [0., 0.])

    def test_lambda_zero_and_gamma_zero(self):
        self.assertEqual(trace_step([9., 8.], [1., 2.], .9, 0., 2.), [2., 4.])
        self.assertEqual(trace_step([9., 8.], [1., 2.], 0., .8), [1., 2.])

    def test_terminal_reward_precedes_clearing(self):
        e = trace_step([1., 0.], [0., 1.], .9, .8)
        self.assertAlmostEqual(e[0], .72)
        # gamma_out=0 removes bootstrap, not the preceding state's eligibility.
        reward, v = 1., 0.
        delta = reward-v
        self.assertAlmostEqual(.1*delta*e[0], .072)
        self.assertEqual(trace_step(e, [1., 0.], 0., .8), [1., 0.])

    def test_frozen_nonlinear_forward_backward_all_lambda(self):
        n = Network(seed=4)
        xs = [features(s, 5) for s in (0, 2, 1, 4)]
        rewards = [.1, -.2, .4]
        discounts = [.9, .7, 0.]
        values = [n.value(x) for x in xs]
        gradients = [n.value_gradient(x)[1] for x in xs[:-1]]
        for lam in (0., .4, 1.):
            targets = finite_lambda_targets(values, rewards, discounts, lam)
            forward = [sum((targets[t]-values[t])*gradients[t][i] for t in range(3)) for i in range(len(n.w))]
            backward = [0.]*len(n.w)
            e = [0.]*len(n.w)
            for t in range(3):
                delta = rewards[t]+discounts[t]*values[t+1]-values[t]
                e = trace_step(e, gradients[t], 0. if t == 0 else discounts[t-1], lam)
                backward = [a+delta*z for a, z in zip(backward, e)]
            for a, b in zip(forward, backward):
                self.assertAlmostEqual(a, b, places=12)

    def test_finite_nonterminal_tail_is_not_zero(self):
        self.assertEqual(finite_lambda_targets([1., 10.], [2.], [.9], 1.), [11.])
        self.assertEqual(finite_lambda_targets([1., 10.], [2.], [0.], 1.), [2.])

    def test_online_updates_break_frozen_reference(self):
        n = Network(seed=4)
        x0, x1 = features(0, 5), features(1, 5)
        v0, g0 = n.value_gradient(x0)
        v1, g1 = n.value_gradient(x1)
        d0, d1 = 1.+.9*v1-v0, 2.-v1
        frozen = [w+.3*(d0*a+d1*(.9*.8*a+b)) for w, a, b in zip(n.w, g0, g1)]
        n.add([d0*g for g in g0], .3)
        new_v1, new_g1 = n.value_gradient(x1)
        e = trace_step(g0, new_g1, .9, .8)
        n.add([(2.-new_v1)*z for z in e], .3)
        self.assertGreater(sum((a-b)**2 for a, b in zip(n.w, frozen)), 1e-5)


class SemanticTests(unittest.TestCase):
    def test_gvf_pseudotermination_and_ratio(self):
        c, gamma, rho = gvf_question(0, 3, 1)
        self.assertEqual((c, gamma, rho), (1., 0., 1.6))
        self.assertEqual(gvf_question(1, 3, 1)[1], .8)
        self.assertEqual((3+1)%4, 0)  # the world continues after question 0 ends

    def test_gvf_truth_bellman_residual(self):
        for h, p in enumerate((.8, .2)):
            for sign in (-1., 1.):
                v = gvf_truth(h, sign)
                for s in range(4):
                    target = 0.
                    for a, probability in ((0, 1-p), (1, p)):
                        sn = (s+a)%4
                        c, gamma, _ = gvf_question(h, sn, a, sign)
                        target += probability*(c+gamma*v[sn])
                    self.assertAlmostEqual(v[s], target, places=10)

    def test_option_duration_and_terminal(self):
        self.assertAlmostEqual(option_target([1., 2.], 10.), 10.9)
        self.assertAlmostEqual(option_target([1., 2.], 10., wrong_duration=True), 11.8)
        self.assertAlmostEqual(option_target([1., 2.], 10., terminal=True), 2.8)
        self.assertEqual(option_target([2.], 10., gamma=0.), 2.)
        self.assertEqual(option_target([2.], 10.), option_target([2.], 10., wrong_duration=True))
        with self.assertRaises(ValueError):
            option_target([], 1.)

    def test_option_has_feedback_and_state_dependent_duration(self):
        self.assertEqual(len(option_model(0, 1)[1]), 1)
        self.assertEqual(len(option_model(1, 1)[1]), 2)
        self.assertEqual(option_step(0, 1)[0], 3)  # left at 0
        self.assertEqual(option_step(1, 1)[0], 2)  # right at 1
        self.assertEqual(option_step(3, 1)[2], False)  # executes before stopping

    def test_option_truth_independent_bellman_residual(self):
        q = option_truth()
        for s in range(4):
            for o in range(2):
                sn, rewards = option_model(s, o)
                self.assertAlmostEqual(q[s][o], option_target(rewards, max(q[sn])), places=10)


class ProtocolTests(unittest.TestCase):
    def test_all_nine_real_loops_reproducible_and_comparable(self):
        for algorithm in IDS:
            module = importlib.import_module("implementations.streaming_composition."+algorithm)
            events = []
            rows = module.run(seed=2, steps=37, emit=events.append)
            self.assertEqual(rows, module.run(seed=2, steps=37))
            self.assertEqual(rows, events)
            runtime.validate_rows(rows, 37)
            runtime.comparable([module.META, runtime.load(module.META["baseline"]).META])
            self.assertTrue(all(math.isfinite(row["value"]) for row in rows))

    def test_mc_incomplete_episode_not_used_as_terminal(self):
        from implementations.streaming_composition.neural_gradient_mc import run
        row = run(steps=5)[-1]
        self.assertEqual(row["update_sample_gradients"], 0)
        self.assertEqual(row["retained_transitions"], 5)
        row = run(steps=7)[-1]
        self.assertEqual(row["update_sample_gradients"], 6)
        self.assertEqual(row["retained_transitions"], 1)

    def test_gvf_shared_world_clock(self):
        from implementations.streaming_composition.gvf_shared_trace import run
        row = run(steps=40)[-1]
        self.assertEqual(row["environment_steps"], 40)
        self.assertEqual(row["network_updates"], 40)
        self.assertEqual(row["phase"], "after-auxiliary-change")

    def test_independent_file_cli(self):
        root = Path(__file__).resolve().parents[1]
        file = root/"implementations/streaming_composition/neural_td_trace.py"
        result = subprocess.run([sys.executable, str(file), "--help"], capture_output=True, text=True, cwd="/tmp")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("steps", result.stdout)


if __name__ == "__main__":
    unittest.main()
