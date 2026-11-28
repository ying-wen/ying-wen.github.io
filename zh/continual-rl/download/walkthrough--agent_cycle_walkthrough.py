"""One continuing agent, two clocks, and predictions that change decisions.

Python 3.10+, standard library only; run from ANY directory:
    python3 /path/to/agent_cycle_walkthrough.py --steps 32
    python3 /path/to/agent_cycle_walkthrough.py --test

This finite, original teaching example reuses the seven-state ring geometry of
integrated_agents/_system.py, not its experiment or learned skills. The world
never resets. Left/right skills are GIVEN, version 1, and stop on arrival at
their target (4/1 respectively) or after two primitive steps. Starting at a
target does not stop a skill before its first action. Each move stalls with
probability .1; arrival at 1 earns +1, and every primitive step costs .02.

What is learned, and who reads it?
  * w[o, remaining, s] predicts external reward UNTIL this skill stops. Its
    on-policy primitive TD(lambda) updates use a real eligibility trace.
    The planner reads w[o, 2, s] as its reward model Rhat(s,o).
  * The last K=4 COMPLETE outcomes per (s,o) estimate duration Dhat and endpoint
    probabilities Phat. These are two more terminating GVF questions, learned
    by Monte Carlo, not by the reward head's TD rule. Stored macro outcomes are
    bounded model memory; this is not a no-stored-experience protocol.
  * Differential control reads R - g*tau + max Q(end) on a real completion.
    Planning reads Rhat - g*Dhat + Phat @ max Q, at most B=2 times per completion.
    Planning changes Q only; it cannot advance real time or create evidence.
  * Each (s,o) has beta=log(alpha). Alpha stays fixed within an execution.
    The NEXT actual execution of that same pair validates the previous local
    reward-model write, using its cached pre-execution prediction and the new
    complete external return. This delayed gradient is evaluated at the beta
    used for the PREVIOUS write and applied to the current beta; the log saves
    both versions. The resulting beta is read at the next start.

The finite-horizon augmented coordinate (o, remaining, s) is never revisited
within an execution. Thus its current TD residuals do not depend on this
execution's alpha: earlier writes only affect already-visited coordinates.
For a fixed starting table and trajectory, h = alpha*sum(lambda**i * delta_i)
is the exact derivative of its START-coordinate final write wrt beta. This is
not a derivative of the entire lifetime: prior tables, targets, selection and
sampling dependencies are outside the local graph, and no gradient crosses
planning. The later Monte Carlo validation return is a fixed sample target.
Likewise, in this special acyclic tabular execution, the actual online trace
matches the frozen-start forward lambda-return. That equivalence must not be
carried over to general online shared-feature or revisiting TD(lambda).

The control update uses the direct differential SMDP extension Q+=.1*delta,
g+=.01*delta, without replacing elapsed time by a discounted surrogate. Its
constant steps and combination with the learned model are a finite teaching
example, not a convergence claim or a published complete CRL/STOMP/OaK
algorithm. Skills, state representation, goals and budgets are fixed by us.
The default four warm-up choices are explicitly prescribed; later choices
are greedy with ties to R. Seed 0 fixes one illustrative environmental path,
not a performance comparison. --steps truncates observation, NOT the skill.
An unfinished skill keeps its trace but does not create a complete model row.

The JSON is an evaluator's full teaching log, not the agent's internal memory.
The agent uses fixed tables and at most 14*4 two-number macro-outcome records;
there is no replay-training loop. Counters describe operations, not wall time,
and Python integer bit length is not a constant-memory theorem.

Source connections (the combined protocol and example are original):
  Sutton & Barto, second edition, sections 8.2 and 12.1-12.2:
  http://incompleteideas.net/book/the-book-2nd.html
  Wan, Naik & Sutton, Average-Reward Learning and Planning with Options,
  section 3 equations (3)-(5), versus normalized equations (6)-(9):
  https://arxiv.org/html/2110.13855
The local derivative is derived above and checked against this file's actual
updates; it is not labelled IDBD, Metatrace, or a full return meta-gradient.
"""

import argparse
from collections import deque
import copy
import json
import math
import random

N = 7
OPTIONS = ("L", "R")
TARGETS = {"L": 4, "R": 1}
VERSION = 1
CAP = 2
WINDOW = 4
LAMBDA = .8
INITIAL_ALPHA = .25
CONTROL_ALPHA = .1
RATE_ALPHA = .01
META_ALPHA = .05
ALPHA_MIN, ALPHA_MAX = .02, .8


def reward_key(option, remaining, state):
    return f"{option}:{remaining}:{state}"


def pair_key(state, option):
    return f"{state}:{option}"


def greedy(row):
    """A declared deterministic tie break, not an exploratory policy."""
    return "L" if row[0] > row[1] else "R"


def reward_td_step(weights, trace, option, remaining, state, reward,
                   successor, ended, alpha, lam=LAMBDA):
    """All residual reads precede writes; terminal delta sees the ACTIVE trace.

    Continuation from the preceding transition is 1 within this execution;
    the current arrival's continuation=0 closes its bootstrap, not its credit.
    The caller clears trace only AFTER this last delta has updated weights.
    """
    key = reward_key(option, remaining, state)
    next_key = None if ended else reward_key(option, remaining - 1, successor)
    old, tail = weights[key], 0. if ended else weights[next_key]
    before_trace = dict(trace)
    for coordinate in trace:
        trace[coordinate] *= lam
    trace[key] = trace.get(key, 0.) + 1.
    delta = reward + tail - old
    writes = []
    for coordinate, eligibility in sorted(trace.items()):
        before = weights[coordinate]
        weights[coordinate] += alpha * delta * eligibility
        writes.append(dict(key=coordinate, before=before,
                           increment=alpha * delta * eligibility,
                           after=weights[coordinate]))
    return dict(delta=delta, target=reward + tail,
                reads=dict(key=key, value=old, next_key=next_key, tail=tail,
                           reward=reward, continuation=0 if ended else 1),
                trace_before=before_trace, trace_active=dict(trace), writes=writes)


class Ring:
    """Only the evaluator sees the RNG draw; the agent receives (s,a,r,s')."""
    def __init__(self, seed=0):
        self.state = 0
        self.rng = random.Random(seed)

    def step(self, option):
        uniform = self.rng.random()
        state = self.state
        self.state = state if uniform < .1 else (state + (-1 if option == "L" else 1)) % N
        reward = float(self.state == 1) - .02
        return state, reward, self.state, uniform


class Agent:
    def __init__(self, planning_budget=2):
        if isinstance(planning_budget, bool) or not isinstance(planning_budget, int) or not 0 <= planning_budget <= 2:
            raise ValueError("planning_budget must be 0, 1 or 2")
        self.budget = planning_budget
        self.q = [[0., 0.] for _ in range(N)]
        self.rate = 0.
        self.weights = {reward_key(o, n, s): 0.
                        for o in OPTIONS for n in (1, 2) for s in range(N)}
        self.beta = {pair_key(s, o): math.log(INITIAL_ALPHA)
                     for s in range(N) for o in OPTIONS}
        self.sensitivity = {}  # previous local derivative and its evaluation beta
        self.outcomes = {}     # at most 14 deques, each maxlen=4
        self.trace = {}
        self.pending = None
        self._stop_endpoint = None  # set only by an actual terminating observation
        self.t = self.k = self.completed = self.model_revision = 0

    def model(self, state, option):
        key = pair_key(state, option)
        data = self.outcomes.get(key)
        if not data:
            return None  # no empirical D/P: never plan from this row
        endpoints = [sum(end == s for _, end in data) / len(data) for s in range(N)]
        return dict(reward=self.weights[reward_key(option, 2, state)],
                    duration=sum(duration for duration, _ in data) / len(data),
                    endpoint=endpoints, evidence_in_window=len(data),
                    skill_version=VERSION, representation_version=VERSION,
                    model_revision=self.model_revision)

    def snapshot(self):
        return dict(t=self.t, k=self.k, q=copy.deepcopy(self.q), rate=self.rate,
                    reward_gvf=dict(self.weights), trace=dict(self.trace),
                    beta=dict(self.beta), local_sensitivity=dict(self.sensitivity),
                    model_revision=self.model_revision,
                    model_window_sizes={key: len(value) for key, value in sorted(self.outcomes.items())},
                    completed=self.completed,
                    greedy=[greedy(row) for row in self.q])

    def start(self, state, option, reason):
        if self.pending is not None:
            raise ValueError("cannot start before the current skill stops")
        key = pair_key(state, option)
        self.trace.clear()
        self.pending = dict(start=state, option=option, duration=0, total=0.,
                            alpha=math.exp(self.beta[key]), beta_used=self.beta[key],
                            cached_prediction=self.weights[reward_key(option, CAP, state)],
                            previous_h=self.sensitivity.get(key, {}).get("h"),
                            previous_beta_used=self.sensitivity.get(key, {}).get("beta_used"), h=0.,
                            model_before=self.model(state, option))
        return dict(type="option_start", t=self.t, k=self.k, state=state,
                    option=option, reason=reason, skill_version=VERSION,
                    representation_version=VERSION,
                    beta_used=self.pending["beta_used"], alpha_used=self.pending["alpha"],
                    prediction_before=self.pending["cached_prediction"],
                    previous_h=self.pending["previous_h"], snapshot=self.snapshot())

    def observe(self, state, reward, successor):
        p = self.pending
        if p is None:
            raise ValueError("start a skill before observing")
        if self._stop_endpoint is not None:
            raise ValueError("complete the stopped skill before observing again")
        remaining = CAP - p["duration"]
        ended = successor == TARGETS[p["option"]] or remaining == 1
        update = reward_td_step(self.weights, self.trace, p["option"], remaining,
                                state, reward, successor, ended, p["alpha"])
        p["h"] += p["alpha"] * LAMBDA ** p["duration"] * update["delta"]
        p["duration"] += 1
        p["total"] += reward
        self._stop_endpoint = successor if ended else None
        self.t += 1
        self.model_revision += 1  # reward head changed from ONE real observation
        event = dict(type="atomic", t=self.t, k=self.k, state=state,
                     action=p["option"], successor=successor, reward=reward,
                     remaining_before=remaining, option_ended=ended,
                     stop_reason=("target" if successor == TARGETS[p["option"]]
                                  else "cap") if ended else None,
                     skill_version=VERSION, alpha_used=p["alpha"],
                     cumulative_reward=p["total"], duration=p["duration"],
                     local_h=p["h"], **update)
        # Keep the active terminal trace visible in the atomic event; callers
        # can verify it was used. complete() clears it after the final write.
        return event

    def complete(self, successor):
        p = self.pending
        if (p is None or p["duration"] < 1 or self._stop_endpoint is None
                or self._stop_endpoint != successor):
            raise ValueError("completion needs the observed stop endpoint of a nonempty skill")
        start, option, total, duration = p["start"], p["option"], p["total"], p["duration"]
        key, oi = pair_key(start, option), OPTIONS.index(option)
        rate_before, value_before = self.rate, self.q[start][oi]
        tail = max(self.q[successor])
        target = total - rate_before * duration + tail
        delta = target - value_before
        self.q[start][oi] += CONTROL_ALPHA * delta
        self.rate += RATE_ALPHA * delta
        control = dict(reward=total, duration=duration, rate_before=rate_before,
                       next_value=tail, before=value_before, target=target, delta=delta,
                       after=self.q[start][oi], rate_after=self.rate,
                       rate_rule="rate += 0.01 * delta")
        # The validation prediction was saved BEFORE the current execution's
        # TD updates: comparing its post-update value would reuse this target.
        old_beta = self.beta[key]
        error = p["cached_prediction"] - total
        gradient = None if p["previous_h"] is None else error * p["previous_h"]
        raw_beta = old_beta if gradient is None else old_beta - META_ALPHA * gradient
        self.beta[key] = min(math.log(ALPHA_MAX), max(math.log(ALPHA_MIN), raw_beta))
        meta = dict(key=key, validation_prediction=p["cached_prediction"],
                    heldout_return=total, validation_error=error,
                    loss=.5 * error * error, previous_h=p["previous_h"], gradient=gradient,
                    differentiated_beta=p["previous_beta_used"],
                    beta_used_this_execution=p["beta_used"], beta_before=old_beta,
                    beta_unclipped=raw_beta, beta_after=self.beta[key],
                    clipped=self.beta[key] != raw_beta,
                    alpha_next=math.exp(self.beta[key]), saved_local_h=p["h"],
                    scope="delayed gradient at previous beta; frozen initial table and sample path")
        self.sensitivity[key] = dict(h=p["h"], beta_used=p["beta_used"])
        self.outcomes.setdefault(key, deque(maxlen=WINDOW)).append((duration, successor))
        self.model_revision += 1  # complete D/P observation; not another real tick
        self.completed += 1
        self.trace.clear()
        self.pending = None
        self._stop_endpoint = None
        return dict(type="option_end", t=self.t, k=self.k, start=start,
                    option=option, successor=successor, reward_sum=total, duration=duration,
                    model_before=p["model_before"], model_after=self.model(start, option),
                    control=control, meta=meta, preplan=self.snapshot())

    def plan(self, state, option):
        model = self.model(state, option)
        if model is None:
            raise ValueError("planning needs a complete observed D/P row")
        oi = OPTIONS.index(option)
        before = self.q[state][oi]
        next_values = [max(row) for row in self.q]
        endpoint_value = sum(prob * value for prob, value in zip(model["endpoint"], next_values))
        target = model["reward"] - self.rate * model["duration"] + endpoint_value
        self.q[state][oi] += CONTROL_ALPHA * (target - before)
        self.k += 1
        return dict(type="planning", t=self.t, k=self.k, state=state, option=option,
                    model=model, rate_read=self.rate, endpoint_values=next_values,
                    endpoint_value=endpoint_value, before=before, target=target,
                    after=self.q[state][oi], snapshot=self.snapshot())

    def planning_keys(self, decision_state, latest):
        # Search control: first available rows at the imminent decision state,
        # then the latest completed row, then lexicographic known rows.
        pairs = [(int(key.split(":")[0]), key.split(":")[1]) for key in self.outcomes]
        return sorted(pairs, key=lambda pair: (pair[0] != decision_state, pair != latest, pair))


def run(steps=32, seed=0, planning_budget=2, warmup=("R", "L", "R", "L")):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if any(option not in OPTIONS for option in warmup):
        raise ValueError("unknown warmup option")
    env, agent, events = Ring(seed), Agent(planning_budget), []
    reward_total = 0.
    decision = 0

    def choose():
        return (warmup[decision], "prescribed_warmup") if decision < len(warmup) else (greedy(agent.q[env.state]), "learned_greedy")

    option, reason = choose()
    events.append(agent.start(env.state, option, reason))
    for _ in range(steps):
        state, reward, successor, uniform = env.step(agent.pending["option"])
        reward_total += reward
        atomic = agent.observe(state, reward, successor)
        atomic["environment_uniform_evaluator_only"] = uniform
        events.append(atomic)
        if atomic["option_ended"]:
            end = agent.complete(successor)
            events.append(end)
            before_choice = greedy(agent.q[successor])
            keys = agent.planning_keys(successor, (end["start"], end["option"]))
            for index in range(agent.budget):
                events.append(agent.plan(*keys[index % len(keys)]))
            decision += 1
            option, reason = choose()
            events.append(dict(type="decision", t=agent.t, k=agent.k, state=successor,
                               preplan_greedy=before_choice,
                               postplan_greedy=greedy(agent.q[successor]),
                               next_option=option, reason=reason,
                               will_execute=agent.t < steps, postplan=agent.snapshot()))
            if agent.t < steps:
                events.append(agent.start(successor, option, reason))
    return dict(schema_version=1, description="finite continuing agent mechanism walkthrough",
                config=dict(states=N, start_state=0, reward_location=1, step_cost=.02,
                            stall_probability=.1, targets=TARGETS, skill_cap=CAP,
                            skill_version=VERSION, representation_version=VERSION,
                            lambda_=LAMBDA, initial_alpha=INITIAL_ALPHA,
                            control_alpha=CONTROL_ALPHA, rate_alpha=RATE_ALPHA,
                            meta_alpha=META_ALPHA, window=WINDOW,
                            planning_budget_per_completion=planning_budget,
                            seed=seed, steps=steps, warmup=list(warmup)),
                events=events, final=agent.snapshot(),
                pending=copy.deepcopy(agent.pending),
                evaluation=dict(real_steps=agent.t, planning_calls=agent.k,
                                lifetime_reward=reward_total,
                                observed_reward_per_step=reward_total / agent.t,
                                environment_resets=0,
                                log_is_external_to_agent=True),
                resource_bounds=dict(reward_parameters=28, control_parameters=14,
                                     beta_parameters=14, sensitivity_slots=14,
                                     sensitivity_scalars_per_slot=2,
                                     model_rows=14, complete_outcome_slots=56,
                                     outcome_scalars_per_slot=2,
                                     nonzero_trace_coordinates=2,
                                     planning_calls_per_completion=planning_budget,
                                     planning_calls_per_real_step_at_most=planning_budget))


def self_test():
    """Small checks here; independent path/forward/gradient checks in JS tests."""
    data = run()
    assert data["evaluation"]["real_steps"] == 32
    assert data["evaluation"]["environment_resets"] == 0
    assert data["final"]["k"] == 2 * data["final"]["completed"]
    assert any(event["type"] == "option_end" and event["meta"]["gradient"] is not None
               for event in data["events"])
    assert all(event["model"]["evidence_in_window"] <= WINDOW
               for event in data["events"] if event["type"] == "planning")
    # An unfinished two-step skill is not converted into a terminal sample.
    partial = run(steps=1, warmup=("L",))
    assert partial["pending"]["duration"] == 1
    assert partial["final"]["completed"] == 0 and partial["final"]["k"] == 0
    assert partial["final"]["trace"] == {"L:2:0": 1.}
    # Target at start still permits one action; arriving at it stops afterwards.
    agent = Agent()
    agent.start(1, "R", "test")
    assert not agent.observe(1, -.02, 2)["option_ended"]
    return {"ok": True, "checks": 8}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planning-budget", type=int, default=2)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    data = self_test() if args.test else run(args.steps, args.seed, args.planning_budget)
    print(json.dumps(data, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
