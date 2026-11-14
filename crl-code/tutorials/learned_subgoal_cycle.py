#!/usr/bin/env python3
"""Learn a proposed subgoal's policy from REAL steps, then model its execution.

Python 3.10+, standard library only. Run in this file's directory, or replace
the filename below with its absolute path when invoking from another directory:
  python3 learned_subgoal_cycle.py --practice-episodes 4
  python3 learned_subgoal_cycle.py --practice-episodes 3
  python3 learned_subgoal_cycle.py --c-probe-reward 0
  python3 learned_subgoal_cycle.py --test

Original finite teaching execution, not a STOMP reproduction or performance
experiment. The six-site delivery world, visible local action menus, one-hot
state identities, frontier score, practice scheduler and reset permission are
given. ONLY World reads hidden transition outcomes. Learner learns an observed
graph from actual steps. BFS scores candidate locations; it never supplies a
policy. GoalQ starts at zero and updates one action value per real practice
step; no replay, path solver or model backup trains these internal policies.

The practice behavior tries the least-practiced action, restricted to actions
whose physical outcomes were already observed. Ties follow the local menu.
This is a designed coverage rule, NOT a learned exploration strategy or safety
guarantee. Unknown probe buttons are reserved for one real trial after a learned
skill reaches its proposed subgoal. Given practice budgets do not guarantee
success: three attempts leave the C policy unsuccessful in this example.

Each greedy policy is frozen as version 1. Only a subsequently executed complete
skill supplies a reward/discounted-endpoint model sample. Physical termination
is a valid FAILED completion when the subgoal was not reached; it must not be
relabeled as success. An observation-limited prefix supplies no complete model.
The declared directed acyclic world naturally stops any attempt in <=3 steps;
there is no hidden timeout or fabricated return at a sampling boundary.

External reward is never replaced by the internal arrival indicator. Planning
uses only observed primitive rows and completed frozen-skill models. The final
skill choice is read by an actual subsequent delivery, not merely printed.
All seed, baseline, practice, model-collection, probe and deployment steps and
every reset are separately charged. A reset is an allowed external operation,
not a free physical action or a continuing-environment result. JSON histories
are the evaluator's log, not claimed bounded internal agent memory.
"""
from collections import Counter, deque
from copy import deepcopy
from fractions import Fraction as F
import argparse
import json

GAMMA = F(9, 10)


class World:
    def __init__(self, c_probe_reward=2):
        self._rows = {
            'S': {'upper': ('A', 0), 'lower': ('B', 0)},
            'A': {'exit': ('X', 1), 'probe': ('Y', 0)},
            'B': {'forward': ('C', 0)},
            'C': {'exit': ('X', 1), 'probe': ('Y', c_probe_reward)},
            'X': {}, 'Y': {},
        }
        self.state = 'S'

    def observe(self):
        return dict(state=self.state, actions=list(self._rows[self.state]),
                    physical_terminal=not self._rows[self.state])

    def step(self, action):
        state = self.state
        self.state, reward = self._rows[state][action]
        return dict(state=state, action=action, reward=F(reward),
                    successor=self.state, **self._arrival())

    def _arrival(self):
        observation = self.observe()
        return dict(next_actions=observation['actions'],
                    physical_terminal=observation['physical_terminal'])

    def reset(self):
        self.state = 'S'
        return self.observe()


class Learner:
    def __init__(self):
        self.actions, self.rows, self.counts = {}, {}, Counter()

    def observe(self, observation):
        self.actions[observation['state']] = list(observation['actions'])

    def ingest(self, record):
        key = (record['state'], record['action'])
        self.rows[key] = (record['successor'], record['reward'], record['physical_terminal'])
        self.counts[key] += 1
        self.observe(dict(state=record['successor'], actions=record['next_actions']))

    def tried_actions(self, state):
        return [a for a in self.actions.get(state, []) if self.counts[state, a] > 0]

    def discover(self):
        distance, queue = {'S': 0}, deque(['S'])
        while queue:
            state = queue.popleft()
            for action in self.tried_actions(state):
                successor = self.rows[state, action][0]
                if successor not in distance:
                    distance[successor] = distance[state] + 1
                    queue.append(successor)
        candidates = []
        for state, actions in self.actions.items():
            unknown = [a for a in actions if self.counts[state, a] == 0]
            if unknown and state in distance:
                candidates.append(dict(goal=state, distance=distance[state],
                                       score=F(1, distance[state] + 1), untried=unknown))
        return sorted(candidates, key=lambda row: (-row['score'], row['goal']))

    def primitive_plan(self):
        """Synchronous value iteration over OBSERVED rows, never hidden World."""
        values = {state: F(0) for state in self.actions}
        backups = 0
        for sweep in range(1, 32):
            updated = {}
            for state in values:
                candidates = [reward + (0 if done else GAMMA * values[successor])
                              for (source, _), (successor, reward, done) in self.rows.items()
                              if source == state]
                backups += len(candidates)
                updated[state] = max(candidates, default=F(0))
            if updated == values:
                values = updated
                break
            values = updated
        else:
            raise RuntimeError('finite teaching planner did not converge')
        policy = {}
        for state in values:
            available = self.tried_actions(state)
            if available:
                policy[state] = max(available, key=lambda a: self.rows[state, a][1] +
                                   (0 if self.rows[state, a][2] else GAMMA * values[self.rows[state, a][0]]))
        return dict(values=values, policy=policy, sweeps=sweep, row_backups=backups)


class GoalQ:
    def __init__(self, goal, allowed_actions):
        self.goal = goal
        self.actions = deepcopy(allowed_actions)
        self.q = {(s, a): F(0) for s, actions in self.actions.items() for a in actions}
        self.visits = Counter()
        self.updates = 0

    def behavior(self, state):
        # This sampler reads visit counts, not a graph distance or hidden path.
        return min(self.actions[state], key=lambda a: self.visits[state, a])

    def update(self, record):
        state, action, successor = record['state'], record['action'], record['successor']
        if state == self.goal or not self.actions.get(state):
            raise ValueError('no internal task action after its stopping state')
        done = record['physical_terminal'] or successor == self.goal
        signal = F(int(successor == self.goal))
        old = self.q[state, action]
        tail = F(0) if done else max((self.q[successor, a] for a in self.actions.get(successor, [])), default=F(0))
        target = signal + GAMMA * tail
        self.q[state, action] = target  # alpha=1 in this deterministic finite example
        self.visits[state, action] += 1
        self.updates += 1
        return dict(goal=self.goal, signal=signal, task_stopped=done, old=old,
                    next_value=tail, target=target, delta=target-old, new=target,
                    start_values=self.state_values('S'), updates=self.updates)

    def state_values(self, state):
        return {a: self.q[state, a] for a in self.actions.get(state, [])}

    def freeze(self, version=1):
        policy = {s: max(actions, key=lambda a: self.q[s, a])
                  for s, actions in self.actions.items() if actions and s != self.goal}
        return dict(goal=self.goal, version=version, initiation=['S'], policy=policy,
                    stop_on_goal_or_physical_terminal=True,
                    q={s: self.state_values(s) for s in self.actions},
                    real_td_updates=self.updates)


def complete_model_sample(skill, episode):
    """Return None for a prefix; record both successful and failed completions."""
    if not episode['completed']:
        return None
    if episode['skill_version'] != skill['version'] or episode['goal'] != skill['goal']:
        raise ValueError('model evidence does not match this frozen skill version')
    if not episode['records']:
        raise ValueError('a skill must take at least one real action')
    records, path = episode['records'], episode['path']
    if len(path) != len(records) + 1 or path[0] not in skill['initiation']:
        raise ValueError('invalid start or path length for skill execution')
    for index, row in enumerate(records):
        if row['state'] != path[index] or row['successor'] != path[index + 1]:
            raise ValueError('model evidence is not a continuous actual path')
        if index and row['t'] != records[index - 1]['t'] + 1:
            raise ValueError('model evidence skips a real time step')
        if row['skill_version'] != skill['version'] or row['goal'] != skill['goal']:
            raise ValueError('record has another skill specification')
        if skill['policy'][row['state']] != row['action']:
            raise ValueError('practice behavior is not the frozen skill policy')
        if index < len(records) - 1 and (row['physical_terminal'] or row['successor'] == skill['goal']):
            raise ValueError('execution continued after skill stopping')
    last = records[-1]
    physical, success = last['physical_terminal'], last['successor'] == skill['goal']
    if not (physical or success):
        raise ValueError('completed flag cannot turn a nonstopping prefix into a model')
    if episode['physical_terminal'] != physical or episode['goal_success'] != success:
        raise ValueError('completion label disagrees with the real last transition')
    total = sum((GAMMA ** i * row['reward'] for i, row in enumerate(records)), F(0))
    if episode['external_return'] != total:
        raise ValueError('model reward must be the actual external reward sum')
    duration = len(records)
    return dict(goal=skill['goal'], version=skill['version'], start=episode['path'][0],
                reward=episode['external_return'], duration=duration,
                endpoint=episode['path'][-1], endpoint_weight=GAMMA ** duration,
                physical_terminal=episode['physical_terminal'],
                goal_success=episode['goal_success'], samples=1,
                evidence_steps=[row['t'] for row in episode['records']])


def model_score(model, values):
    tail = 0 if model['physical_terminal'] else values[model['endpoint']]
    return model['reward'] + model['endpoint_weight'] * tail


def run(practice_episodes=4, c_probe_reward=2):
    if isinstance(practice_episodes, bool) or not isinstance(practice_episodes, int) or practice_episodes < 0:
        raise ValueError('practice_episodes must be a nonnegative integer')
    world, learner = World(c_probe_reward), Learner()
    learner.observe(world.observe())
    events, episodes = [], []
    steps, resets = 0, 0
    phase_steps = Counter()

    def episode(phase, choose, goal=None, trainer=None, skill_version=None, observation_limit=None):
        nonlocal steps, resets
        if episodes:
            before = world.state
            learner.observe(world.reset())
            resets += 1
            events.append(dict(type='reset', t=steps, reset_index=resets,
                               from_state=before, to_state='S', next_phase=phase))
        path, records, total = [world.state], [], F(0)
        completed, physical, success = False, False, False
        while True:
            if observation_limit is not None and len(records) >= observation_limit:
                break
            action = choose(world.state)
            record = world.step(action)
            learner.ingest(record)
            steps += 1
            phase_steps[phase] += 1
            record = dict(t=steps, phase=phase, goal=goal, skill_version=skill_version, **record)
            total += GAMMA ** len(records) * record['reward']
            if trainer is not None:
                record['td'] = trainer.update(record)
            records.append(record)
            events.append(dict(type='real', **record))
            path.append(record['successor'])
            physical, success = record['physical_terminal'], record['successor'] == goal
            if physical or success:
                completed = True
                break
            if len(records) >= 32:
                raise RuntimeError('unexpected nonterminating path; no model sample emitted')
        result = dict(phase=phase, goal=goal, skill_version=skill_version, path=path,
                      records=records, external_return=total, completed=completed,
                      physical_terminal=physical, goal_success=success,
                      stop_reason='goal' if success else 'physical_terminal' if physical else 'observation_limit')
        episodes.append(result)
        events.append(dict(type='episode_end', t=steps, phase=phase, goal=goal,
                           path=path, external_return=total, completed=completed,
                           goal_success=success, stop_reason=result['stop_reason']))
        return result

    for actions in [('upper', 'exit'), ('lower', 'forward', 'exit')]:
        iterator = iter(actions)
        episode('seed', lambda state: next(iterator))
    seed_candidates = learner.discover()
    before_plan = learner.primitive_plan()
    baseline = episode('baseline', lambda state: before_plan['policy'][state])
    rounds, skills, models = [], {}, {}
    for _ in range(2):
        candidates = learner.discover()
        if not candidates:
            break
        goal = candidates[0]['goal']
        events.append(dict(type='candidate', t=steps, candidates=candidates, chosen=goal))
        allowed = {s: learner.tried_actions(s) for s in learner.actions}
        trainer = GoalQ(goal, allowed)
        initial = trainer.freeze()
        practices = [episode('practice', trainer.behavior, goal=goal, trainer=trainer)
                     for _ in range(practice_episodes)]
        skill = trainer.freeze()
        skills[goal] = skill
        events.append(dict(type='freeze', t=steps, **deepcopy(skill)))
        execution = episode('model_collection', lambda state: skill['policy'][state],
                            goal=goal, skill_version=skill['version'])
        model = complete_model_sample(skill, execution)
        if model is not None:
            models[goal] = model
            events.append(dict(type='complete_model', t=steps, model=deepcopy(model)))
        probe = None
        # Probe is a separate HIGH-LEVEL choice after arrival, not part of skill.
        unknown = [a for a in learner.actions[world.state] if learner.counts[world.state, a] == 0]
        if execution['goal_success'] and unknown:
            action = unknown[0]
            record = world.step(action)
            learner.ingest(record)
            steps += 1
            phase_steps['probe'] += 1
            probe = dict(t=steps, phase='probe', goal=goal, skill_version=None, **record)
            events.append(dict(type='real', **probe))
        exploration = dict(path=execution['path'] + ([] if probe is None else [probe['successor']]),
                           external_return=execution['external_return'] +
                           (0 if probe is None else GAMMA ** len(execution['records']) * probe['reward']),
                           primitive_steps=len(execution['records']) + int(probe is not None))
        rounds.append(dict(goal=goal, candidates=candidates, initial_skill=initial,
                           practice=practices, skill=skill, model_execution=execution,
                           model=model, probe=probe, exploration=exploration,
                           next_candidates=learner.discover()))
    final_plan = learner.primitive_plan()
    scores = {goal: model_score(model, final_plan['values']) for goal, model in models.items()}
    chosen = max(sorted(scores), key=scores.get)
    events.append(dict(type='plan', t=steps, primitive_values=final_plan['values'],
                       option_scores=scores, chosen=chosen))
    skill = skills[chosen]
    deployment_skill = episode('deployment', lambda state: skill['policy'][state],
                               goal=chosen, skill_version=skill['version'])
    deployment_path = list(deployment_skill['path'])
    deployment_return = deployment_skill['external_return']
    deployment_tail = []
    while learner.actions[world.state]:
        action = final_plan['policy'][world.state]
        record = world.step(action)
        learner.ingest(record)
        steps += 1
        phase_steps['deployment'] += 1
        record = dict(t=steps, phase='deployment_tail', goal=None, skill_version=None, **record)
        deployment_return += GAMMA ** (len(deployment_path) - 1) * record['reward']
        deployment_tail.append(record)
        events.append(dict(type='real', **record))
        deployment_path.append(record['successor'])
    return dict(schema_version=1, config=dict(gamma=GAMMA, practice_episodes=practice_episodes,
                alpha=1, initial_q=0, reset_permission=True,
                practice_actions='only previously tried actions; least local practice count; menu-order ties',
                c_probe_reward_evaluator_only=c_probe_reward),
                seed_candidates=seed_candidates, before_plan=before_plan, baseline=baseline,
                rounds=rounds, models=deepcopy(models), final_plan=final_plan,
                option_scores=scores, chosen=chosen,
                deployment=dict(skill_episode=deployment_skill, tail=deployment_tail,
                                path=deployment_path, external_return=deployment_return),
                budgets=dict(real_steps=steps, phase_steps=dict(phase_steps), resets=resets,
                             episodes_started=len(episodes), unique_observed_rows=len(learner.rows),
                             actual_goal_td_updates=sum(r['skill']['real_td_updates'] for r in rounds),
                             primitive_planning_backups=before_plan['row_backups']+final_plan['row_backups'],
                             option_model_queries=len(scores), model_complete_samples=len(models)),
                remaining_candidates=learner.discover(), events=events)


def self_test():
    full, short = run(4), run(3)
    assert [c['goal'] for c in full['seed_candidates']] == ['A', 'C']
    assert full['rounds'][1]['skill']['q']['S']['lower'] == F(9, 10)
    assert short['rounds'][1]['skill']['q']['S']['lower'] == 0
    assert full['models']['C']['goal_success'] and not short['models']['C']['goal_success']
    assert short['models']['C']['endpoint'] == 'X' and short['models']['C']['reward'] == F(9, 10)
    assert full['baseline']['path'] == ['S', 'A', 'X']
    assert full['deployment']['path'] == ['S', 'B', 'C', 'Y']
    assert full['deployment']['external_return'] == F(81, 50)
    assert full['budgets']['real_steps'] == 31 and full['budgets']['resets'] == 13
    assert run(4, c_probe_reward=0)['chosen'] == 'A'
    prefix = deepcopy(full['rounds'][1]['model_execution'])
    prefix['completed'] = False
    assert complete_model_sample(full['rounds'][1]['skill'], prefix) is None
    return dict(ok=True, checks=11)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--practice-episodes', type=int, default=4)
    parser.add_argument('--c-probe-reward', type=int, default=2,
                        help='environment-only counterfactual; unknown until actually probed')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    result = self_test() if args.test else run(args.practice_episodes, args.c_probe_reward)
    print(json.dumps(result, indent=2, default=lambda x: float(x) if isinstance(x, F) else str(x)))
