#!/usr/bin/env python3
"""Six-place model aging: exact, finite closed-loop experiment; standard library.

Download this file alone, then run:
    python3 model-change-walkthrough.py
    python3 model-change-walkthrough.py --json

All probabilities and returns use Fraction. The independent planner enumerates
empirical outcome paths to terminal, rather than the browser module's ordered
Bellman backups. A count of five equivalent row backups is reported per plan;
it is an algorithmic budget for the browser planner, not Python operation count.
"""
import argparse
from collections import Counter
from fractions import Fraction as F
from math import comb
import json

GAMMA = F(9, 10)
BUDGET, CLOSE, REOPEN, AGE = 96, 12, 48, 12
MODES = ('cumulative-greedy', 'recent-greedy', 'cumulative-recheck', 'recent-recheck')
ACTIONS = {'S': ('upper', 'lower'), 'A': ('cross',), 'B': ('forward',),
           'C': ('forward',), 'X': ('reset',), 'Y': ('reset',)}


class World:
    """Only an executed action returns a transition; no phase observation."""
    def __init__(self, scenario):
        assert scenario in ('reopen', 'stationary', 'closed')
        self.scenario, self.clock, self.state = scenario, -28, 'S'

    def step(self, action):
        s, t = self.state, self.clock
        assert action in ACTIONS[s]
        reset = action == 'reset'
        if s in ('X', 'Y'):
            nxt = 'S'
        elif s == 'A':
            blocked = self.scenario != 'stationary' and t >= CLOSE
            blocked = blocked and (self.scenario == 'closed' or t < REOPEN)
            nxt = 'Y' if blocked else 'X'
        else:
            nxt = {('S', 'upper'): 'A', ('S', 'lower'): 'B',
                   ('B', 'forward'): 'C', ('C', 'forward'): 'X'}[s, action]
        self.clock, self.state = t + 1, nxt
        return dict(t=t + 1, s=s, a=action, next=nxt,
                    reward=int(nxt == 'X' and not reset),
                    done=nxt in ('X', 'Y'), reset=reset)


class EmpiricalModel:
    def __init__(self, recent):
        self.recent = recent
        self.counts, self.last_outcome, self.last_start = {}, {}, {}
        self.version = 0

    def observe(self, row):
        if row['reset']:
            return
        k = row['s'], row['a']
        outcome = row['next'], row['reward'], row['done']
        self.counts.setdefault(k, Counter())[outcome] += 1
        self.last_outcome[k] = outcome
        self.version += 1
        if row['s'] == 'S':
            self.last_start[row['a']] = row['t']

    def outcomes(self, s, a):
        k = s, a
        counts = Counter({self.last_outcome[k]: 1}) if self.recent else self.counts[k]
        total = sum(counts.values())
        return [(nxt, reward, done, F(n, total))
                for (nxt, reward, done), n in counts.items()]

    def path_values(self, state, action):
        """Enumerate probability/discounted-return pairs, with no truth query."""
        paths = []
        for nxt, reward, done, prob in self.outcomes(state, action):
            if done:
                paths.append((prob, F(reward)))
            else:
                # This observed graph has one action after leaving S.
                assert len(ACTIONS[nxt]) == 1
                for tail_prob, tail_return in self.path_values(nxt, ACTIONS[nxt][0]):
                    paths.append((prob * tail_prob, reward + GAMMA * tail_return))
        return paths

    def plan(self):
        return {a: sum((prob * ret for prob, ret in self.path_values('S', a)), F(0))
                for a in ('upper', 'lower')}

    def p_x(self):
        return sum((p for nxt, _, _, p in self.outcomes('A', 'cross') if nxt == 'X'), F(0))


def run(mode, scenario):
    model, world = EmpiricalModel(mode.startswith('recent')), World(scenario)
    for route in ['upper'] * 8 + ['lower']:
        actions = ['upper', 'cross', 'reset'] if route == 'upper' else ['lower', 'forward', 'forward', 'reset']
        for action in actions:
            model.observe(world.step(action))
    assert world.clock == 0 and model.version == 19
    decisions, trace, episodes = [], [], []
    reward = resets = 0
    episode = None
    for t in range(BUDGET):
        state = world.state
        if state == 'S':
            values = model.plan()
            greedy = 'upper' if values['upper'] > values['lower'] else 'lower'
            ages = {a: t - model.last_start[a] for a in ('upper', 'lower')}
            stale = sorted((a for a in ages if ages[a] >= AGE), key=lambda a: (-ages[a], a))
            action = stale[0] if mode.endswith('recheck') and stale else greedy
            decisions.append(dict(t=t, choice=action, greedy=greedy, values=values,
                                  modelVersion=model.version, ages=ages))
            episode = dict(start=t, choice=action, path=['S'], steps=0, ret=F(0))
        else:
            action = ACTIONS[state][0]
        row = world.step(action)
        model.observe(row)
        reward += row['reward']
        resets += row['reset']
        if not row['reset']:
            episode['ret'] += GAMMA ** episode['steps'] * row['reward']
            episode['steps'] += 1
            episode['path'].append(row['next'])
            if row['done']:
                episode['end'] = row['t']
                episodes.append(episode)
                episode = None
        trace.append(dict(**row, totalReward=reward, pX=model.p_x(), modelVersion=model.version))
    metrics = dict(interactionTicks=BUDGET, physicalActions=BUDGET-resets, resets=resets,
                   modelBackups=5*len(decisions), modelUpdates=model.version-19, reward=reward,
                   completedEpisodes=len(episodes), failedEpisodes=sum(e['ret'] == 0 for e in episodes),
                   recheckOverrides=sum(d['choice'] != d['greedy'] for d in decisions),
                   discountedEpisodeReturn=sum((e['ret'] for e in episodes), F(0)),
                   unfinishedEpisode=episode is not None, terminalAtBudget=world.state in ('X', 'Y'))
    return dict(mode=mode, scenario=scenario, decisions=decisions, trace=trace, metrics=metrics)


def noise_counterfactual():
    """Exact finite binomial enumeration, a separate stationary IID problem."""
    p, n = F(19, 20), 20
    mean_error = sum((F(comb(n, k)) * p**k * (1-p)**(n-k) * (F(k, n)-p)**2
                      for k in range(n+1)), F(0))
    last_error = p*(1-p)**2 + (1-p)*p**2
    assert mean_error == p*(1-p)/n and last_error == p*(1-p)
    return dict(p=p, n=n, sampleMeanMSE=mean_error, lastObservationMSE=last_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--test', action='store_true', help='run exact assertions and print PASS')
    args = parser.parse_args()
    results = [run(m, s) for s in ('reopen', 'stationary', 'closed') for m in MODES]
    expected = [23, 24, 22, 25, 32, 32, 30, 30, 23, 24, 19, 20]
    assert [r['metrics']['reward'] for r in results] == expected
    # Without a real upper-route visit, reopening and staying closed produce
    # identical observable histories for both greedy controllers.
    for index in (0, 1):
        assert results[index]['trace'] == results[index+8]['trace']
    noise = noise_counterfactual()
    if args.test:
        print('PASS: exact closed-loop results, coverage equivalence, and binomial MSE assertions.')
    elif args.json:
        print(json.dumps(dict(results=results, stationaryNoise=noise),
                         default=lambda x: float(x) if isinstance(x, F) else str(x)))
    else:
        print('Exact finite run: 28 warm-up ticks + 96 comparison ticks per controller.')
        print('scenario    estimator/controller      reward  fails  reset  row-backups')
        for r in results:
            m = r['metrics']
            print(f"{r['scenario']:11} {r['mode']:25} {m['reward']:6} {m['failedEpisodes']:6} {m['resets']:6} {m['modelBackups']:12}")
        print('Stationary IID counterfactual MSE:', {k: str(v) for k, v in noise.items()})
        print('Fraction assertions passed; no random training or hidden-phase controller input.')


if __name__ == '__main__':
    main()
