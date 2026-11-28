"""Protocol v2: an inspectable continual-control implementation, without study results.

Python 3 standard library. In the directory containing this downloaded file:
    python3 continual_protocol.py
    python3 continual_protocol.py --mode template
    python3 continual_protocol.py --mode fixture
    python3 continual_protocol.py --mode fixture --events

Default mode validates a specification; it never starts an environment. The fixture
uses short, explicit uniform tapes, not sampled performance trials. There is no
performance-run CLI. A future study must freeze its design and add a study runner.
The constant-step differential controller is a teaching prototype, not a
convergence claim or a reproduction of Dyna-Q, STOMP, or OaK.
Its real update uses the unscaled sampled-duration form discussed in Wan et al.
(NeurIPS 2021), equations (3-5), not their expected-duration-scaled algorithm (6-9):
https://papers.neurips.cc/paper_files/paper/2021/file/c058f544c737782deacefa532d9add4c-Paper.pdf
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys

N, GOALS, CAP = 7, (1, 4), 6
SCHEMA = "continual-ring-protocol-v2"
CONFIGURATIONS = (
    {"id": "recent-planning", "memory": "recent", "planning": 4},
    {"id": "cumulative-planning", "memory": "cumulative", "planning": 4},
    {"id": "recent-no-planning", "memory": "recent", "planning": 0},
    {"id": "cumulative-no-planning", "memory": "cumulative", "planning": 0},
)
PENDING = [
    "sealed study design, custody, study runner, failures and uncertainty analysis",
    "serializable learner/pending-macro/RNG checkpoint with tested resume",
    "isolated frozen diagnostic using a complete learner-state copy",
    "wall-clock, CPU, peak-memory, hardware and runtime measurement protocol",
    "compute-matched comparison and learned revisit/planning scheduling",
]


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def positive_integer(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(label + " must be a positive integer")


def protocol_template():
    """An implementation contract, separate from the fillable study design."""
    return {
        "schema": SCHEMA,
        "environment_steps": 1200,
        "configurations": copy.deepcopy(list(CONFIGURATIONS)),
        "environment_interface": "separate-evaluator-only-json",
        "rng_streams": ["environment", "behavior", "planning"],
        "research_status": "not-run",
        "results": None,
    }


def environment_example():
    """Public validation example; never describes a sealed test set."""
    return {"schema": SCHEMA + "-environment", "reward_schedule": [
        {"start_step": 1, "rewarding_position": 1},
        {"start_step": 601, "rewarding_position": 4},
    ]}


def validate_environment(specification):
    if not isinstance(specification, dict) or set(specification) != {"schema", "reward_schedule"}:
        raise ValueError("environment needs exactly schema and reward_schedule")
    if specification["schema"] != SCHEMA + "-environment":
        raise ValueError("unsupported environment schema")
    schedule = specification["reward_schedule"]
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("reward_schedule must be a nonempty list")
    previous = 0
    for row in schedule:
        if not isinstance(row, dict) or set(row) != {"start_step", "rewarding_position"}:
            raise ValueError("each change needs start_step and rewarding_position")
        start, goal = row["start_step"], row["rewarding_position"]
        positive_integer(start, "start_step")
        if start <= previous or (previous == 0 and start != 1):
            raise ValueError("schedule must start at 1 and increase strictly")
        if type(goal) is not int or not 0 <= goal < N:
            raise ValueError("rewarding_position must be an integer in [0, 6]")
        previous = start
    # Future entries are allowed: increasing the budget never moves a change.
    return copy.deepcopy(specification)


def validate_protocol(protocol, environment=None):
    expected = protocol_template()
    if not isinstance(protocol, dict) or set(protocol) != set(expected):
        raise ValueError("protocol fields must match the versioned implementation contract")
    positive_integer(protocol["environment_steps"], "environment_steps")
    for key in expected:
        if key != "environment_steps" and protocol[key] != expected[key]:
            raise ValueError("unsupported protocol field: " + key)
    # Strictly reject bool/float planning values which compare equal to integers.
    for config in protocol["configurations"]:
        if type(config["planning"]) is not int:
            raise ValueError("planning must be an integer")
    if environment is not None:
        validate_environment(environment)
    return {"status": "validated-implementation-contract", "schema": SCHEMA,
            "protocol_sha256": digest(protocol), "environment_validated": environment is not None,
            "environment_sha256": digest(environment) if environment is not None else None,
            "environment_steps": protocol["environment_steps"], "configuration_count": 4,
            "environment_executed": False, "study_design_validated": False,
            "research_status": "not-run", "results": None, "pending": list(PENDING)}


class UniformStream:
    """One named stream; every draw is counted. Injected fixtures never wrap."""
    def __init__(self, name, *, seed=None, values=None):
        if (seed is None) == (values is None):
            raise ValueError("provide exactly one of seed and values")
        self.name, self.draws = name, 0
        if values is not None:
            values = tuple(values)
            if not values or any(type(x) not in (int, float) or not math.isfinite(x)
                                 or not 0 <= x < 1 for x in values):
                raise ValueError("uniform tape entries must be finite numbers in [0, 1)")
            self.values, self.rng = values, None
            self.identity = "fixed-tape:" + digest(values)
        else:
            if type(seed) is not int or seed < 0:
                raise ValueError("seed must be a nonnegative integer")
            derived = digest({"schema": SCHEMA, "seed": seed, "stream": name})
            self.values, self.rng = None, random.Random(int(derived, 16))
            self.identity = "domain-separated-seed:" + derived

    def random(self):
        if self.values is not None:
            if self.draws == len(self.values):
                raise ValueError("exhausted fixed tape: " + self.name)
            value = self.values[self.draws]
        else:
            value = self.rng.random()
        self.draws += 1
        return value

    def choice(self, choices):
        if not choices:
            raise ValueError("cannot choose from an empty sequence")
        return choices[int(self.random() * len(choices))]

    def descriptor(self):
        state = self.rng.getstate() if self.rng is not None else {"index": self.draws}
        return {"stream": self.name, "identity": self.identity, "draws": self.draws,
                "state_sha256": digest(state)}


class Environment:
    """Evaluator owns the change table. Agent receives no Environment reference."""
    def __init__(self, specification, stream):
        self.specification = validate_environment(specification)
        self.stream, self.t, self.state, self.phase = stream, 0, 0, 0

    def step(self, action):
        if type(action) is not int or action not in (0, 1):
            raise ValueError("primitive action must be 0 or 1")
        self.t += 1
        schedule = self.specification["reward_schedule"]
        while self.phase + 1 < len(schedule) and schedule[self.phase + 1]["start_step"] <= self.t:
            self.phase += 1
        uniform = self.stream.random()
        if uniform >= .1:
            self.state = (self.state + (-1 if action == 0 else 1)) % N
        reward = float(self.state == schedule[self.phase]["rewarding_position"]) - .02
        return self.state, reward


def available(state):
    return [0, 1] + [2 + i for i, goal in enumerate(GOALS) if state != goal]


class EventLog:
    def __init__(self, run_id, emit=None):
        self.run_id, self.emit, self.count = run_id, emit, 0

    def add(self, event, **fields):
        self.count += 1
        row = {"event_id": self.count, "run_id": self.run_id, "event": event, **fields}
        if self.emit is not None:
            self.emit(copy.deepcopy(row))


class Agent:
    """Shared learning backbone. Only memory rule and planning count vary."""
    def __init__(self, configuration, behavior, planning, log):
        if (not isinstance(configuration, dict) or configuration not in CONFIGURATIONS
                or type(configuration.get("planning")) is not int):
            raise ValueError("unknown factorial configuration")
        if behavior is planning or behavior.name != "behavior" or planning.name != "planning":
            raise ValueError("behavior and planning require separate named streams")
        self.configuration = dict(configuration)
        self.behavior, self.planning, self.log = behavior, planning, log
        self.q = [[0.] * 4 for _ in range(N)]
        self.subq = [[[0., 0.] for _ in range(N)] for _ in GOALS]
        self.policies = [[0] * N for _ in GOALS]
        self.versions, self.model, self.rate = [0, 0], {}, 0.
        self.counts = dict(real_backups=0, model_backups=0, subtask_backups=0,
                           model_writes=0, invalidations=0)

    def version(self, option):
        return self.versions[option - 2] if option >= 2 else 0

    def choose(self, state):
        choices = available(state)
        if self.behavior.random() >= .2:
            best = max(self.q[state][a] for a in choices)
            choices = [a for a in choices if self.q[state][a] == best]
        return self.behavior.choice(choices)

    def primitive(self, state, option):
        if option < 2:
            return option
        if self.behavior.random() < .1:
            return self.behavior.choice([0, 1])
        return self.policies[option - 2][state]

    def observe(self, state, action, successor, t, macro_id):
        for i, goal in enumerate(GOALS):
            before = self.subq[i][state][action]
            target = 1. if successor == goal else -.01 + .95 * max(self.subq[i][successor])
            self.subq[i][state][action] += .25 * (target - before)
            self.counts["subtask_backups"] += 1
            self.log.add("subtask_update", t=t, macro_id=macro_id, goal=goal,
                         cell=[state, action], q_before=before, target=target,
                         q_after=self.subq[i][state][action], alpha=.25)

    def complete(self, macro, successor, t):
        state, option = macro["start_state"], macro["option_id"]
        reward_sum, duration = macro["reward_sum"], macro["duration"]
        positive_integer(duration, "completed macro duration")
        if self.version(option) != macro["executed_policy_version"]:
            raise ValueError("policy changed during macro execution")
        before, rate_before = self.q[state][option], self.rate
        next_value = max(self.q[successor][a] for a in available(successor))
        target = reward_sum - self.rate * duration + next_value
        delta = target - before
        self.q[state][option] += .1 * delta
        self.rate += .002 * delta
        self.counts["real_backups"] += 1
        common = {"t": t, "macro_id": macro["macro_id"]}
        self.log.add("real_update", **common, cell=[state, option],
                     model_version=self.version(option), reward_sum=reward_sum, duration=duration,
                     next_value=next_value, target=target, delta=delta, q_before=before,
                     q_after=self.q[state][option], rate_before=rate_before, rate_after=self.rate)
        key = (state, option)
        old = copy.deepcopy(self.model.get(key))
        count = old["count"] if old else 0
        cell = self.model.setdefault(key, {"count": 0, "reward": 0., "duration": 0.,
                                          "endpoints": [0.] * N, "version": self.version(option)})
        alpha = 1. if not count else (.2 if self.configuration["memory"] == "recent" else 1. / (count + 1))
        cell["count"] += 1
        cell["reward"] += alpha * (reward_sum - cell["reward"])
        cell["duration"] += alpha * (duration - cell["duration"])
        for s in range(N):
            cell["endpoints"][s] += alpha * (float(s == successor) - cell["endpoints"][s])
        self.counts["model_writes"] += 1
        self.log.add("model_write", **common, cell=list(key), alpha=alpha,
                     model_count_before=count, model_version=self.version(option), model_before=old,
                     observed_target={"reward_sum": reward_sum, "duration": duration, "successor": successor},
                     prediction_error_before=None if old is None else abs(reward_sum - old["reward"]),
                     model_after=cell)
        keys = sorted(self.model)
        for _ in range(self.configuration["planning"]):
            s, o = self.planning.choice(keys)
            model = self.model[s, o]
            q_before = self.q[s][o]
            endpoint_values = [max(self.q[x][a] for a in available(x)) for x in range(N)]
            target = model["reward"] - self.rate * model["duration"] + sum(
                p * v for p, v in zip(model["endpoints"], endpoint_values))
            self.q[s][o] += .1 * (target - q_before)
            self.counts["model_backups"] += 1
            self.log.add("planning_update", **common, cell=[s, o], model_version=model["version"],
                         model_read=model, endpoint_values=endpoint_values, rate=self.rate,
                         target=target, q_before=q_before, q_after=self.q[s][o],
                         planning_draw=self.planning.draws)
        # Models/Q refer to the policy just executed; commit only after planning.
        for i, goal in enumerate(GOALS):
            new = [max(range(2), key=lambda a: row[a]) for row in self.subq[i]]
            if any(new[s] != self.policies[i][s] for s in range(N) if s != goal):
                o = i + 2
                deleted = [{"cell": list(k), "model": self.model[k]} for k in sorted(self.model) if k[1] == o]
                old_policy, old_q = self.policies[i], [row[o] for row in self.q]
                self.policies[i] = new
                self.versions[i] += 1
                for item in deleted:
                    del self.model[tuple(item["cell"])]
                for row in self.q:
                    row[o] = 0.
                self.counts["invalidations"] += 1
                self.log.add("policy_commit", **common, option_id=o, policy_before=old_policy,
                             policy_after=new, version_before=self.versions[i] - 1,
                             version_after=self.versions[i], invalidated_cells=deleted,
                             q_column_before=old_q, q_column_after=[0.] * N)


def run_fixture(configuration, *, steps=16, environment=None, tapes=None, emit=None):
    """Short semantic fixture, bounded to 64 steps; not a performance study API.

    The environment uses a supplied uniform per atomic step. Behavior/planning
    use separate supplied tapes indexed by their own calls; differing trajectories
    need not consume behavior draws at corresponding environment steps.
    """
    positive_integer(steps, "fixture steps")
    if steps > 64:
        raise ValueError("fixture limit is 64; a research runner is not implemented")
    environment = environment if environment is not None else {
        "schema": SCHEMA + "-environment", "reward_schedule": [
            {"start_step": 1, "rewarding_position": 1},
            {"start_step": 7, "rewarding_position": 4},
            {"start_step": 13, "rewarding_position": 1},
        ]}
    if tapes is None:
        tapes = {"environment": [.8, .05, .8, .8] * 16,
                 "behavior": [.3, .6, .8, .9, .4, .2, .7, .1] * 64,
                 "planning": [.0, .3, .6, .9] * 64}
    if set(tapes) != {"environment", "behavior", "planning"}:
        raise ValueError("fixture needs exactly environment, behavior and planning tapes")
    streams = {name: UniformStream(name, values=values) for name, values in tapes.items()}
    log = EventLog("fixture:" + configuration["id"], emit)
    env = Environment(environment, streams["environment"])
    agent = Agent(configuration, streams["behavior"], streams["planning"], log)
    effective_contract = protocol_template()
    effective_contract["environment_steps"] = steps
    fixture_specification = {"environment_steps": steps, "configuration": configuration,
                             "evaluator_environment": environment, "uniform_tapes": tapes}
    log.add("run_start", schema=SCHEMA, evidence_status="fixed-sequence-semantics-only",
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            protocol_sha256=digest(effective_contract), effective_contract=effective_contract,
            fixture_specification=fixture_specification, fixture_specification_sha256=digest(fixture_specification),
            configuration=configuration,
            configuration_sha256=digest(configuration), fixture_steps=steps,
            evaluator_environment_sha256=digest(environment), fixture_tapes_sha256=digest(tapes),
            streams={name: stream.descriptor() for name, stream in streams.items()})
    macro, macro_id, cumulative_reward = None, 0, 0.
    for t in range(1, steps + 1):
        if macro is None:
            macro_id += 1
            option = agent.choose(env.state)
            macro = dict(macro_id=macro_id, start_t=t, start_state=env.state, option_id=option,
                         executed_policy_version=agent.version(option), reward_sum=0., duration=0)
            log.add("macro_start", t=t, **macro, behavior_draws=agent.behavior.draws)
        state, option = env.state, macro["option_id"]
        action = agent.primitive(state, option)
        successor, reward = env.step(action)
        cumulative_reward += reward
        macro["reward_sum"] += reward
        macro["duration"] += 1
        log.add("atomic_step", t=t, macro_id=macro_id, state_before=state,
                primitive_action=action, successor=successor, external_reward=reward,
                cumulative_reward=cumulative_reward, active_option=option,
                option_version=macro["executed_policy_version"],
                environment_draws=env.stream.draws, behavior_draws=agent.behavior.draws,
                evaluator_phase=env.phase)
        agent.observe(state, action, successor, t, macro_id)
        reason = ("primitive" if option < 2 else "subgoal" if successor == GOALS[option - 2]
                  else "duration-cap" if macro["duration"] == CAP else None)
        if reason is not None:
            log.add("macro_boundary", **macro, end_t=t, successor=successor,
                    stop_reason=reason, censored=False, model_sample=True)
            agent.complete(macro, successor, t)
            macro = None
        log.add("resource_counts", t=t, environment_steps=t, **agent.counts,
                model_cells=len(agent.model), pending_duration=0 if macro is None else macro["duration"])
    if macro is not None:
        log.add("macro_boundary", **macro, end_t=steps, successor=env.state,
                stop_reason="budget-truncation", censored=True, model_sample=False)
    summary = {"configuration_id": configuration["id"], "environment_steps": steps,
               "cumulative_reward": cumulative_reward, "pending_duration": 0 if macro is None else macro["duration"],
               **agent.counts, "model_cells": len(agent.model),
               "streams": {name: stream.descriptor() for name, stream in streams.items()},
               "evidence_status": "fixed-sequence-semantics-only", "research_status": "not-run"}
    log.add("run_end", **summary, q=agent.q, subq=agent.subq, policies=agent.policies,
            versions=agent.versions, rate=agent.rate,
            model=[{"cell": list(k), **v} for k, v in sorted(agent.model.items())],
            pending_macro=macro, checkpoint_restorable=False, pending=list(PENDING))
    return {**summary, "event_count": log.count}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("validate", "template", "environment-example", "fixture"), default="validate")
    parser.add_argument("--protocol", type=Path, help="implementation contract JSON, used only by validate")
    parser.add_argument("--environment", type=Path, help="separate evaluator-owned change table JSON, used only by validate")
    parser.add_argument("--events", action="store_true", help="fixture only: emit complete JSONL event stream")
    args = parser.parse_args(argv)
    if args.mode != "validate" and (args.protocol or args.environment):
        parser.error("--protocol and --environment are only accepted in validate mode")
    if args.events and args.mode != "fixture":
        parser.error("--events requires --mode fixture")
    try:
        if args.mode == "template":
            result = protocol_template()
        elif args.mode == "environment-example":
            result = environment_example()
        elif args.mode == "validate":
            protocol = json.loads(args.protocol.read_text(encoding="utf-8")) if args.protocol else protocol_template()
            environment = json.loads(args.environment.read_text(encoding="utf-8")) if args.environment else None
            result = validate_protocol(protocol, environment)
        else:
            emit = (lambda row: print(json.dumps(row, ensure_ascii=False, allow_nan=False))) if args.events else None
            summaries = [run_fixture(config, emit=emit) for config in CONFIGURATIONS]
            if args.events:
                return 0
            result = {"schema": SCHEMA, "status": "fixed-sequence-fixtures-completed",
                      "evidence_status": "semantic-checks-only", "research_status": "not-run",
                      "results": None, "fixtures": summaries, "pending": list(PENDING)}
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print("protocol error: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
