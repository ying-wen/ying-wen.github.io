import copy
import math
import unittest
from unittest.mock import patch

from implementations.learner_control import frozen_parameters, online_differential_q
from implementations.learner_control._common import (
    CHANGE_STEP, FREEZE_STEP, Controller, Corridor, Observation, all_rows,
)
from implementations.runtime import comparable, discover, validate_rows


class LearnerControlTests(unittest.TestCase):
    def test_real_time_good_route(self):
        env = Corridor()
        reward, obs = env.step(1)
        self.assertEqual(obs, Observation('travel', 2, 1))
        self.assertEqual(reward, -.02)
        reward, obs = env.step(0)
        self.assertEqual((reward, obs.remaining), (-.02, 1))
        reward, obs = env.step(0)
        self.assertEqual((reward, obs), (1., Observation('hub')))
        self.assertEqual(env.time, 3)
        self.assertAlmostEqual(env.total_reward, .96)

    def test_recovery_cost_and_no_free_reset(self):
        env = Corridor()
        env.step(2); env.step(0)
        reward, obs = env.step(0)
        self.assertEqual((reward, obs.phase, obs.remaining), (-.4, 'recovery', 4))
        for remaining in (3, 2, 1, 0):
            reward, obs = env.step(0)
            self.assertEqual(reward, -.05)
            self.assertEqual(obs.remaining, remaining)
        self.assertEqual(obs.phase, 'hub')
        self.assertEqual((env.time, env.failures, env.recovery_steps), (7, 1, 4))
        self.assertAlmostEqual(env.total_reward, -.64)

    def test_information_cost_and_unannounced_fixed_change(self):
        env = Corridor()
        self.assertIsNone(env.observation().cue)
        reward, obs = env.step(0)
        self.assertEqual((reward, obs.cue), (-.05, 1))
        env.time = CHANGE_STEP-1
        # Arrival is judged at its actual time, so the change is not a label
        # that is handed to the controller in a normal unqueried observation.
        env.step(1)
        self.assertIsNone(env.observation().cue)
        env.step(0)
        reward, obs = env.step(0)
        self.assertEqual(reward, -.4)
        self.assertIsNone(obs.cue)

    def test_frozen_parameters_do_not_freeze_memory(self):
        agent = Controller(0)
        initial = copy.deepcopy(agent.q)
        key = agent.observe(Observation('hub', cue=1))
        next_key = agent.observe(Observation('hub', cue=2))
        rate, _ = frozen_parameters.update(agent.q, agent.rate, key, 0, -.05, next_key)
        self.assertEqual((agent.memory, agent.memory_writes, rate), (2, 2, 0.))
        self.assertEqual(agent.q, initial)
        self.assertNotEqual(key, next_key)

    def test_same_initial_policy_and_legal_observation_keys(self):
        first, second = Controller(9), Controller(9)
        self.assertEqual(len(all_rows()), 27)
        for key, row in first.q.items():
            self.assertEqual(first.probabilities(key), second.probabilities(key))
            self.assertAlmostEqual(sum(first.probabilities(key)), 1.)
            self.assertEqual(len(row), 3 if key[0] == 'hub' else 1)
        env = Corridor()
        env.step(1)
        with self.assertRaises(ValueError):
            env.step(2)
        self.assertEqual(env.time, 1)  # rejected actions do not consume time

    def test_hand_computed_update_and_old_rate(self):
        agent = Controller(0)
        key = ('hub', 0, 0, None)
        successor = ('travel', 2, 1, None)
        agent.q[key][1], agent.q[successor][0] = .4, .8
        rate, delta = online_differential_q.update(
            agent.q, .2, key, 1, -.02, successor, alpha=.1, eta=.05)
        self.assertAlmostEqual(delta, .18)
        self.assertAlmostEqual(agent.q[key][1], .418)
        self.assertAlmostEqual(rate, .2009)

    def test_snapshot_freeze_keeps_existing_parameters(self):
        agent = Controller(0)
        key = ('hub', 0, 0, None)
        agent.q[key] = [.2, .7, -.1]
        saved = copy.deepcopy(agent.q)
        rate, _ = frozen_parameters.update(agent.q, .3, key, 0, -.05, key)
        self.assertEqual(agent.q, saved)
        self.assertEqual(rate, .3)

    def test_budget_prefix_does_not_move_change(self):
        for module in (online_differential_q, frozen_parameters):
            short = module.run(seed=3, steps=620)
            long = module.run(seed=3, steps=1200)
            self.assertEqual(short, [row for row in long if row['step'] <= 620])
            self.assertEqual(next(r for r in long if r['step'] == 600)['phase'],
                             'after-change')

    def test_identical_history_until_freeze_then_memory_continues(self):
        online = online_differential_q.run(seed=3, steps=1200)
        frozen = frozen_parameters.run(seed=3, steps=1200)
        self.assertEqual([r for r in online if r['step'] <= FREEZE_STEP],
                         [r for r in frozen if r['step'] <= FREEZE_STEP])
        checkpoint = next(r for r in frozen if r['step'] == FREEZE_STEP)
        for row in frozen:
            if row['step'] > FREEZE_STEP:
                self.assertEqual(row['q_norm'], checkpoint['q_norm'])
                self.assertEqual(row['estimated_gain'], checkpoint['estimated_gain'])
                self.assertEqual(row['parameter_updates'], FREEZE_STEP)
        self.assertGreater(frozen[-1]['memory_writes'], checkpoint['memory_writes'])

    def test_exact_checkpoint_parameters_and_hidden_future_causality(self):
        captured = []
        def make_agent(seed):
            agent = Controller(seed)
            captured.append(agent)
            return agent
        with patch('implementations.learner_control._common.Controller', make_agent):
            online = online_differential_q.run(seed=4, steps=FREEZE_STEP)
            frozen = frozen_parameters.run(seed=4, steps=FREEZE_STEP)
            with patch('implementations.learner_control._common.CHANGE_STEP', 900):
                changed_future = online_differential_q.run(seed=4, steps=FREEZE_STEP)
        self.assertEqual(online, frozen)
        self.assertEqual(online, changed_future)
        for agent in captured[1:]:
            self.assertEqual(agent.q, captured[0].q)
            self.assertEqual(agent.rate, captured[0].rate)
            self.assertEqual(agent.memory, captured[0].memory)
            self.assertEqual(agent.rng.getstate(), captured[0].rng.getstate())

    def test_runtime_contract_and_counters(self):
        comparable([online_differential_q.META, frozen_parameters.META])
        ids = discover()
        self.assertIn('learner_control_online', ids)
        self.assertIn('learner_control_frozen', ids)
        for module, learns in ((online_differential_q, True), (frozen_parameters, False)):
            emitted = []
            rows = module.run(seed=2, steps=1200, emit=emitted.append)
            validate_rows(rows, 1200)
            self.assertEqual(rows, emitted)
            for row in rows:
                self.assertTrue(math.isfinite(row['value']))
                self.assertAlmostEqual(row['value']*row['step'], row['lifetime_reward'])
                self.assertEqual(row['environment_resets'], 0)
                expected = row['step'] if learns else min(row['step'], FREEZE_STEP)
                self.assertEqual(row['parameter_updates'], expected)
                self.assertEqual(row['memory_writes'], row['probes'])
            self.assertGreater(rows[-1]['q_norm'], 0.)
            self.assertGreater(rows[-1]['memory_writes'], 0)

    def test_no_budget_terminal_or_invalid_budget(self):
        for module in (online_differential_q, frozen_parameters):
            for steps in (1, 2, 3, 17):
                rows = module.run(seed=0, steps=steps)
                validate_rows(rows, steps)
                self.assertEqual(rows[-1]['parameter_updates'], steps)
            for invalid in (True, 0, -1, 1.5):
                with self.assertRaises(ValueError):
                    module.run(steps=invalid)


if __name__ == '__main__':
    unittest.main()
