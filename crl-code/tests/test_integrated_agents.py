"""Interface identities, invalidation and real environment clocks, not win tests."""
import unittest
from implementations.integrated_agents._system import (
    Agent, OptionModel, GOALS, N, available, smdp_error, subtask_update, run_system,
)


class IntegratedAgentTests(unittest.TestCase):
    def test_duration_counts_real_time(self):
        self.assertAlmostEqual(smdp_error(3., 4, .5, 2., 1.), 2.)
        with self.assertRaises(ValueError):
            smdp_error(1, 0, 1, 1, 1)

    def test_model_joint_moments_and_planning(self):
        model = OptionModel(recent=False)
        model.update(0, 2, 1., 2, 1)
        model.update(0, 2, 3., 4, 4)
        cell = model.cells[0, 2]
        self.assertEqual(cell[:3], [2, 2., 3.])
        self.assertAlmostEqual(sum(cell[3]), 1.)
        q = [[float(s)]*4 for s in range(N)]
        self.assertAlmostEqual(model.target(0, 2, q, .5), 2.-.5*3.+.5*(1+4))

    def test_external_model_never_uses_subtask_reward(self):
        q = [[0., 0.] for _ in range(N)]
        subtask_update(q, 0, 1, GOALS[0], GOALS[0], alpha=1.)
        self.assertEqual(q[0][1], 1.)
        model = OptionModel()
        model.update(0, 2, -.02, 1, GOALS[0])
        self.assertEqual(model.cells[0, 2][1], -.02)

    def test_no_zero_duration_option(self):
        for i, goal in enumerate(GOALS):
            self.assertNotIn(i+2, available(goal))

    def test_policy_commit_invalidates_all_dependents(self):
        agent = Agent(0)
        agent.model.update(0, 2, 1., 2, 1)
        agent.model.update(0, 0, 0., 1, 6)
        agent.q[0][2] = 4.
        agent.subq[0][0][1] = 1.
        agent.commit_policies()
        self.assertNotIn((0, 2), agent.model.cells)
        self.assertIn((0, 0), agent.model.cells)
        self.assertTrue(all(row[2] == 0. for row in agent.q))
        self.assertEqual(agent.versions[0], 1)

    def test_negative_control_keeps_stale_model(self):
        agent = Agent(0, versioned=False)
        agent.model.update(0, 2, 1., 2, 1)
        agent.subq[0][0][1] = 1.
        agent.commit_policies()
        self.assertIn((0, 2), agent.model.cells)
        self.assertEqual(agent.versions[0], 1)

    def test_planning_does_not_change_real_update_clock(self):
        agent = Agent(0, planning=4, primitive_only=True)
        agent.complete(0, 1, .98, 1, 1)
        self.assertEqual((agent.real_backups, agent.model_backups, agent.model.updates), (1, 4, 1))
        self.assertAlmostEqual(agent.rate, .002*.98)

    def test_single_lifetime_deterministic_population(self):
        for seed in range(5):
            for config in ({}, {'recent': False}, {'planning': 0},
                           {'versioned': False}, {'primitive_only': True}):
                rows = run_system(seed, 127, **config)
                self.assertEqual(rows, run_system(seed, 127, **config))
                self.assertEqual(rows[-1]['step'], 127)
                self.assertLessEqual(rows[-1]['model_cells'], N*4)
                self.assertEqual(rows[-1]['model_backups'],
                                 config.get('planning', 4)*rows[-1]['real_backups'])
                if not config.get('primitive_only', False):
                    self.assertEqual(rows[-1]['subtask_backups'], 2*127)
                self.assertTrue(all(-.02-1e-12 <= r['value'] <= .98+1e-12 for r in rows))


if __name__ == '__main__':
    unittest.main()
