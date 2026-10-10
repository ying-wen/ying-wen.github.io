"""One continuing world, explicit information actions, and recoverable costs.

No episodes or environment reset. A bad route consumes four recovery steps.
Only an inspect action reveals the current good route. The remembered cue
continues to update even when value/rate parameters are frozen. A hidden
change happens at primitive step 600, independently of the requested budget.

This is a finite teaching comparison of complete learning processes, not
a general deviation-regret estimator, an irreversible-world benchmark, or
a convergence claim for nonstationary/non-Markov agent-state TD.
"""
import math
import random
from collections import deque
from typing import NamedTuple, Optional

CHANGE_STEP = 600
FREEZE_STEP = 400


class Observation(NamedTuple):
    phase: str
    remaining: int = 0
    route: int = 0
    cue: Optional[int] = None


class Corridor:
    """Hub actions: 0 inspect, 1 left, 2 right. Else only action 0 (wait).

    Entering a route costs .02 and takes a real step. Two further travel
    steps follow. The intermediate step costs .02. Arrival gives +1 when
    the route is good; otherwise -.4 and four forced -.05 recovery steps.
    Inspect costs .05, takes one step, stays at the hub and reveals a cue.
    The good route is left before step 600 and right from step 600 onward.
    Route success uses the condition at arrival, not departure.
    """
    def __init__(self):
        self.time = 0
        self.phase, self.remaining, self.route = 'hub', 0, 0
        self.probes = self.successes = self.failures = self.recovery_steps = 0
        self.total_reward = 0.

    def observation(self, cue=None):
        return Observation(self.phase, self.remaining, self.route, cue)

    def step(self, action):
        legal = (0, 1, 2) if self.phase == 'hub' else (0,)
        if action not in legal:
            raise ValueError('action is not available in the observed phase')
        self.time += 1
        good_route = 1 if self.time < CHANGE_STEP else 2
        cue = None
        if self.phase == 'hub':
            if action == 0:
                reward, cue = -.05, good_route
                self.probes += 1
            else:
                self.phase, self.remaining, self.route = 'travel', 2, action
                reward = -.02
        elif self.phase == 'travel':
            self.remaining -= 1
            if self.remaining:
                reward = -.02
            elif self.route == good_route:
                reward = 1.
                self.successes += 1
                self.phase, self.route = 'hub', 0
            else:
                reward = -.4
                self.failures += 1
                self.phase, self.remaining, self.route = 'recovery', 4, 0
        else:
            reward = -.05
            self.recovery_steps += 1
            self.remaining -= 1
            if self.remaining == 0:
                self.phase = 'hub'
        self.total_reward += reward
        # There is no terminal flag. Logging/budget boundaries are not resets.
        return reward, self.observation(cue)


def all_rows():
    """Enumerate all legal observable phase/clock/route × remembered-cue rows.

    Preallocation lets the frozen condition preserve its complete parameter
    dictionary, not merely refrain from changing already visited rows.
    """
    observations = [Observation('hub')]
    observations += [Observation('travel', n, route)
                     for n in (1, 2) for route in (1, 2)]
    observations += [Observation('recovery', n) for n in (1, 2, 3, 4)]
    return {(o.phase, o.remaining, o.route, memory):
            [0.] * (3 if o.phase == 'hub' else 1)
            for o in observations for memory in (None, 1, 2)}


class Controller:
    """Persistent observation memory and parameters are different state.

    The controller gets Observation, not the world, global clock, change
    indicator, unqueried good-route bit, or evaluation statistics.
    """
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.q, self.rate = all_rows(), 0.
        self.memory = None
        self.memory_writes = 0

    def observe(self, observation):
        if observation.cue is not None:
            self.memory = observation.cue
            self.memory_writes += 1
        return (observation.phase, observation.remaining,
                observation.route, self.memory)

    def probabilities(self, key, epsilon=.15):
        values = self.q[key]
        best = [i for i, v in enumerate(values) if v == max(values)]
        return [epsilon/len(values)+(1-epsilon)/len(best) if i in best
                else epsilon/len(values) for i in range(len(values))]

    def choose(self, key):
        u = self.rng.random()
        for action, probability in enumerate(self.probabilities(key)):
            u -= probability
            if u < 0:
                return action
        return len(self.q[key])-1


def run_control(seed, steps, emit, update, learns):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    world, agent = Corridor(), Controller(seed)
    key = agent.observe(world.observation())
    rows, recent, parameter_updates = [], deque(maxlen=100), 0
    for t in range(1, steps+1):
        action = agent.choose(key)
        reward, observation = world.step(action)
        next_key = agent.observe(observation)
        # Snapshot ordering: old rate/Q, actual reward, new observation memory.
        agent.rate, delta = update(agent.q, agent.rate, key, action, reward, next_key)
        writes = learns(t) if callable(learns) else learns
        parameter_updates += int(writes)
        key = next_key
        recent.append(reward)
        if t == 1 or t % 20 == 0 or t == steps:
            row = {
                'step': t, 'value': world.total_reward/t,
                'phase': 'before-change' if t < CHANGE_STEP else 'after-change',
                'lifetime_reward': world.total_reward,
                'recent_reward_rate': sum(recent)/len(recent),
                'estimated_gain': agent.rate,
                'parameter_updates': parameter_updates,
                'parameters_frozen': int(not writes),
                'memory_writes': agent.memory_writes,
                'probes': world.probes, 'successful_routes': world.successes,
                'failed_routes': world.failures,
                'recovery_steps': world.recovery_steps,
                'environment_resets': 0,
                'q_norm': math.sqrt(sum(v*v for row in agent.q.values() for v in row)),
                'last_td_error': delta,
                'pending_phase': observation.phase,
                'pending_steps': observation.remaining,
            }
            rows.append(row)
            if emit:
                emit(row.copy())
    # No fake terminal update for an unfinished route or recovery at the budget.
    return rows
