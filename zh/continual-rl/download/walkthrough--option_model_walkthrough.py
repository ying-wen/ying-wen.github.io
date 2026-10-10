#!/usr/bin/env python3
"""A fixed option, its two models, and planning in the textbook four rooms.

Run: python3 tutorials/option_model_walkthrough.py
     python3 tutorials/option_model_walkthrough.py --json

Standard library only. All outputs are exact deterministic teaching calculations,
not sampled learning curves. Fixed records are used to inspect model updates; the
script neither trains an agent nor measures discovery/sample efficiency.

Reference: Sutton, Precup & Singh (1999), sections 2, 3, 5. In this file a
terminal environment transition retains its reward but has zero endpoint mass.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from math import isclose

State = tuple[int, int]
MOVES = ((0, -1), (1, 0), (0, 1), (-1, 0))
NAMES = ("up", "right", "down", "left")
GAMMA = 0.9
GOAL = (6, 0)
DOORS = ((3, 1), (5, 3), (3, 5), (1, 3))
STATES = tuple((x, y) for y in range(7) for x in range(7)
               if (x != 3 and y != 3) or (x, y) in DOORS)


def key(state: State) -> str:
    return f"{state[0]},{state[1]}"


@dataclass(frozen=True)
class Transition:
    state: State
    action: int
    reward: float
    next: State
    terminal: bool
    stop: bool


@dataclass(frozen=True)
class Option:
    name: str
    room: frozenset[State]
    initiation: frozenset[State]
    policy: dict[State, int]
    target: State

    def beta(self, arrival: State) -> bool:
        # Check AFTER an action. Starting at the other doorway is permitted,
        # although beta is 1 there: the first action enters the room.
        return arrival not in self.room


def transition(state: State, action: int, reward_mode: str) -> tuple[State, float, bool]:
    if state == GOAL:
        raise ValueError("No transition should be sampled after environment termination")
    dx, dy = MOVES[action]
    candidate = (state[0] + dx, state[1] + dy)
    arrival = candidate if candidate in STATES else state
    terminal = arrival == GOAL
    if reward_mode not in ("cost", "goal"):
        raise ValueError("Choose 'cost' or 'goal' explicitly")
    reward = -1.0 if reward_mode == "cost" else float(terminal)
    return arrival, reward, terminal


def make_options() -> list[Option]:
    options = []
    for name, x0, y0, doors in (
        ("NW", 0, 0, ((3, 1), (1, 3))),
        ("NE", 4, 0, ((3, 1), (5, 3))),
        ("SW", 0, 4, ((1, 3), (3, 5))),
        ("SE", 4, 4, ((5, 3), (3, 5))),
    ):
        room = frozenset(s for s in STATES if x0 <= s[0] < x0 + 3 and y0 <= s[1] < y0 + 3)
        allowed = room | frozenset(doors)
        for target in doors:
            distance, queue = {target: 0}, deque([target])
            while queue:
                s = queue.popleft()
                for dx, dy in MOVES:
                    n = (s[0] + dx, s[1] + dy)
                    if n in allowed and n not in distance:
                        distance[n] = distance[s] + 1
                        queue.append(n)
            policy = {}
            initiation = allowed - {target, GOAL}
            for s in sorted(initiation):
                # Among shortest paths, align the doorway row first.
                preferred = (2 if s[1] < target[1] else 0) if s[1] != target[1] else (1 if s[0] < target[0] else 3)
                for a in (preferred, 0, 1, 2, 3):
                    dx, dy = MOVES[a]
                    n = (s[0] + dx, s[1] + dy)
                    if n in distance and distance[n] == distance[s] - 1:
                        policy[s] = a
                        break
            options.append(Option(f"{name}:{key(target)}", room, initiation, policy, target))
    return options


def execute(option: Option, start: State, reward_mode: str = "cost") -> list[Transition]:
    if start not in option.initiation:
        raise ValueError("Illegal option start")
    s, rows = start, []
    for _ in range(50):
        a = option.policy[s]
        n, reward, terminal = transition(s, a, reward_mode)
        stop = terminal or option.beta(n)
        rows.append(Transition(s, a, reward, n, terminal, stop))
        if stop:
            return rows
        s = n
    # A budget cutoff is not a successful termination or complete model sample.
    raise ValueError("Incomplete option: do not fit a full-outcome model")


def outcome(rows: list[Transition], gamma: float = GAMMA) -> dict:
    """Complete-call sample. Gamma is already contained in the endpoint kernel."""
    if not rows or not rows[-1].stop or any(t.stop for t in rows[:-1]):
        raise ValueError("Need exactly one complete option call")
    reward, discount = 0.0, 1.0
    for t in rows:
        reward += discount * t.reward
        discount *= gamma
    return {"reward": reward, "kernel": {} if rows[-1].terminal else {key(rows[-1].next): discount},
            "duration": len(rows), "end": key(rows[-1].next), "terminal": rows[-1].terminal}


def boundary_update(old: dict, sample: dict, alpha: float) -> dict:
    """Update only the start of this completed call, not its intermediate states."""
    keys = old["kernel"].keys() | sample["kernel"].keys()
    return {"reward": old["reward"] + alpha * (sample["reward"] - old["reward"]),
            "kernel": {x: old["kernel"].get(x, 0) + alpha * (sample["kernel"].get(x, 0) - old["kernel"].get(x, 0)) for x in sorted(keys)}}


def intra_pass(option: Option, rows: list[Transition], rewards: dict, kernels: dict,
               alpha: float = 1.0, gamma: float = GAMMA) -> None:
    """One chronological pass through GIVEN records: SPS (1999), eqs. 18–19.

    The fixed deterministic option must choose the observed action. Targets are
    cached before either model is changed; this also makes self-loops correct.
    This example makes no stochastic off-policy or neural convergence claim.
    """
    for t in rows:
        if option.policy[t.state] != t.action:
            raise ValueError("Incompatible action for this deterministic option")
        s, n = key(t.state), key(t.next)
        beta = option.beta(t.next)
        if t.terminal:
            reward_target, kernel_target = t.reward, {}
        elif beta:
            reward_target, kernel_target = t.reward, {n: gamma}
        else:
            reward_target = t.reward + gamma * rewards.get(n, 0)
            kernel_target = {x: gamma * p for x, p in kernels.get(n, {}).items()}
        new = boundary_update({"reward": rewards.get(s, 0), "kernel": kernels.get(s, {})},
                              {"reward": reward_target, "kernel": kernel_target}, alpha)
        rewards[s], kernels[s] = new["reward"], new["kernel"]


def backup(model: dict, values: dict[str, float]) -> float:
    # No additional gamma here: the duration has already entered the kernel.
    return model["reward"] + sum(p * values.get(x, 0) for x, p in model["kernel"].items())


def plan(options: list[Option], sweeps: int, with_options: bool) -> dict:
    """Synchronous exact-model VI, goal reward +1, other rewards 0.

    Count candidate backups separately from sweeps. Building the option models
    also queries primitive dynamics; none of these are agent experience samples.
    """
    models, model_queries = {}, 0
    if with_options:
        for s in STATES:
            for option in options:
                if s in option.initiation:
                    rows = execute(option, s, "goal")
                    models[(s, option.name)] = outcome(rows)
                    model_queries += len(rows)
    values = {key(s): 0.0 for s in STATES}
    backups, history = 0, [values.copy()]
    for _ in range(sweeps):
        new = values.copy()
        for s in STATES:
            if s == GOAL:
                continue
            candidates = []
            for a in range(4):
                n, reward, terminal = transition(s, a, "goal")
                candidates.append(reward + (0 if terminal else GAMMA * values[key(n)]))
            if with_options:
                candidates.extend(backup(models[(s, o.name)], values) for o in options if s in o.initiation)
            new[key(s)] = max(candidates)
            backups += len(candidates)
        values = new
        history.append(values.copy())
    return {"history": history, "backups": backups, "model_build_queries": model_queries}


def calculate() -> dict:
    options = make_options()
    option = next(o for o in options if o.name == "NW:3,1")
    rows = execute(option, (0, 0))
    sample = outcome(rows)
    smdp = boundary_update({"reward": 0, "kernel": {}}, sample, 1)
    rewards, kernels, passes = {}, {}, []
    for k in range(4):
        intra_pass(option, rows, rewards, kernels)
        passes.append({"pass": k + 1, "rewards": rewards.copy(),
                       "kernels": {s: p.copy() for s, p in kernels.items()}})
    # Same map, same termination, different deployed policy: six steps to g.
    detour_path = ((0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (3, 1))
    detour_policy = {s: MOVES.index((n[0]-s[0], n[1]-s[1])) for s, n in zip(detour_path, detour_path[1:])}
    changed = Option("NW:3,1@detour", option.room, option.initiation,
                     {**option.policy, **detour_policy}, option.target)
    fresh = outcome(execute(changed, (0, 0)))
    data = {
        "provenance": "Exact deterministic teaching calculations; fixed-record updates, no training run.",
        "gamma": GAMMA, "states": [key(s) for s in STATES],
        "trace": [{"state": key(t.state), "action": NAMES[t.action], "reward": t.reward,
                   "next": key(t.next), "stop": t.stop, "terminal": t.terminal} for t in rows],
        "sample": sample, "boundary": smdp, "intra": passes,
        "planning": {"primitive": plan(options, 5, False), "augmented": plan(options, 5, True)},
        "version_change": {"old_path": [key(rows[0].state)] + [key(t.next) for t in rows],
                           "new_path": [key(s) for s in detour_path], "old": sample, "new": fresh,
                           "old_target": backup(sample, {"3,1": 10}), "new_target": backup(fresh, {"3,1": 10})},
    }
    assert isclose(sample["reward"], -3.439)
    assert isclose(passes[-1]["kernels"]["0,0"]["3,1"], 0.6561)
    assert isclose(data["planning"]["augmented"]["history"][-1]["0,0"], 0.9**7)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print exact data for figure regeneration")
    args = parser.parse_args()
    data = calculate()
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    print("Fixed four-room option: I/π/β are supplied; no discovery is run.")
    for k, t in enumerate(data["trace"]):
        print(f"t={k}: {t['state']} --{t['action']}--> {t['next']}  R={t['reward']:+g}  stop={t['stop']}  environment_done={t['terminal']}")
    print("Complete sample:", data["sample"])
    for item in data["intra"]:
        print(f"Fixed-record pass {item['pass']}: reward(A)={item['rewards']['0,0']:.6f}, kernel(A,g)={item['kernels']['0,0'].get('3,1', 0):.6f}")
    for name, run in data["planning"].items():
        print(f"{name}: V5(A)={run['history'][-1]['0,0']:.7f}, candidate_backups={run['backups']}, model_build_queries={run['model_build_queries']}")
    v = data["version_change"]
    print(f"Changed policy, same endpoint: stale target={v['old_target']:.6f}; current target={v['new_target']:.6f}")


if __name__ == "__main__":
    main()
