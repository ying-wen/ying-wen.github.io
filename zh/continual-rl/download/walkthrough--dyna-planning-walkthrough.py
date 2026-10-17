#!/usr/bin/env python3
"""Standalone exact Dyna mechanism, Python 3 standard library only.

Run from any directory: python3 dyna-planning-walkthrough.py --json
                       python3 dyna-planning-walkthrough.py --test
No training seeds or performance claims; all arithmetic uses Fraction.
"""
import argparse
from fractions import Fraction as F
import json

GAMMA = F(9, 10)
ACTIONS = {"S": ("go", "exit"), "A": ("cross",), "G": (), "B": (), "D": ()}
OLD = [("S", "go", F(0), "A", False), ("A", "cross", F(1), "G", True),
       ("S", "exit", F(3, 5), "B", True)]
CHANGED = [OLD[0], ("A", "cross", F(0), "D", True)]


def target(q, row, gamma=GAMMA):
    s, a, reward, nxt, done = row
    if a not in ACTIONS[s]:
        raise ValueError("invalid action")
    return reward if done else reward + gamma * max(q[nxt, b] for b in ACTIONS[nxt])


class Learner:
    def __init__(self):
        self.q = {(s, a): F(0) for s in ACTIONS for a in ACTIONS[s]}
        self.model = {}
        self.pred = {s: set() for s in ACTIONS}
        self.counts = dict.fromkeys(("environmentSteps", "resets", "directBackups", "modelWrites",
                                    "modelCalls", "planningBackups", "priorityReads", "queuePops"), 0)

    def observe(self, row, direct=True):
        s, a, reward, nxt, done = row
        pair = s, a
        self.counts["environmentSteps"] += 1
        if direct:
            self.q[pair] = target(self.q, row)
            self.counts["directBackups"] += 1
        if pair in self.model:
            self.pred[self.model[pair][3]].remove(pair)
        self.model[pair] = row
        self.pred[nxt].add(pair)
        self.counts["modelWrites"] += 1

    def read(self, pair, priority=False):
        if pair not in self.model:
            raise ValueError("unobserved pair")
        self.counts["modelCalls"] += 1
        self.counts["priorityReads"] += int(priority)
        return self.model[pair]

    def plan(self, pair):
        self.q[pair] = target(self.q, self.read(pair))
        self.counts["planningBackups"] += 1

    def enqueue_preds(self, state, queue):
        for pair in sorted(self.pred[state]):
            residual = abs(target(self.q, self.read(pair, True)) - self.q[pair])
            if residual > 0:
                queue[pair] = max(queue.get(pair, F(0)), residual)

    def pop(self, queue):
        pair = sorted(queue, key=lambda p: (-queue[p], p))[0]
        del queue[pair]
        self.counts["queuePops"] += 1
        row = self.read(pair)
        if target(self.q, row) == self.q[pair]:
            return pair, True
        self.q[pair] = target(self.q, row)
        self.counts["planningBackups"] += 1
        self.enqueue_preds(pair[0], queue)
        return pair, False


def warmup():
    # The learner sees three actual rows; the map is never inserted as a truth table.
    # One modeled backup, after observing the old reward, propagates it to S.
    l = Learner()
    l.observe(OLD[0])
    l.observe(OLD[1])
    l.plan(("S", "go"))
    l.counts["resets"] += 1
    l.observe(OLD[2])
    return l


def choose(q):
    return "go" if q["S", "go"] > q["S", "exit"] else "exit"


def delta(l, base):
    return {k: v - base[k] for k, v in l.counts.items()}


def q_json(q):
    return {f"{s}:{a}": float(v) for (s, a), v in q.items()}


def walkthrough():
    # Phase counters below exclude warmup. Resets are counted separately from transitions.
    base = warmup().counts.copy()
    scenarios = {}
    for name in ("none", "scheduled", "hybrid", "textbookPriority"):
        l = warmup()
        l.counts["resets"] += 1
        queue = {}
        for row in CHANGED:
            # Textbook priority first writes the model and queues the observed residual;
            # a separate direct backup here would erase that residual before insertion.
            l.observe(row, direct=name != "textbookPriority")
            if name == "textbookPriority":
                pair = row[:2]
                priority = abs(target(l.q, row) - l.q[pair])
                if priority:
                    queue[pair] = priority
        if name == "scheduled":
            for pair in (("S", "exit"), ("A", "cross"), ("S", "go")):
                l.plan(pair)
        if name == "hybrid":
            # Direct-first Dyna has already changed A, so seed from its predecessors.
            l.enqueue_preds("A", queue)
            l.pop(queue)
        if name == "textbookPriority":
            # A finite two-pop planning budget is sufficient for this two-link graph.
            for _ in range(2):
                if not queue:
                    break
                l.pop(queue)
        result = {"q": q_json(l.q), "choice": choose(l.q), "counts": delta(l, base)}
        if name != "textbookPriority":
            first = choose(l.q)
            rows = [OLD[2]] if first == "exit" else CHANGED
            # World generates actual rows according to the chosen action.
            l.counts["resets"] += 1
            for row in rows:
                l.observe(row)
            result["nextExperience"] = {"firstAction": first,
                "route": ["S", "B"] if first == "exit" else ["S", "A", "D"],
                "rewards": [float(r[2]) for r in rows],
                "return": float(sum(GAMMA ** i * r[2] for i, r in enumerate(rows))),
                "steps": len(rows)}
            result["countsAfterNext"] = delta(l, base)
            result["finalQ"] = q_json(l.q)
        scenarios[name] = result
    return {"evidence": "exact-deterministic-mechanism", "sampledRuns": 0,
            "gamma": float(GAMMA), "oldQ": q_json(warmup().q), "warmupCounts": base,
            "scenarios": scenarios, "stochasticCheck": {"expectation": .75, "variance": .1875}}


def self_test():
    d = walkthrough()
    # Independent path enumeration: rewards occur on transitions, terminal continuation is zero.
    routes = {"old_go": [F(0), F(1)], "new_go": [F(0), F(0)], "exit": [F(3, 5)]}
    returns = {name: sum(GAMMA ** i * r for i, r in enumerate(rs)) for name, rs in routes.items()}
    assert returns == {"old_go": F(9, 10), "new_go": F(0), "exit": F(3, 5)}
    assert d["oldQ"]["S:go"] == float(returns["old_go"])
    for name in ("scheduled", "hybrid", "textbookPriority"):
        assert d["scenarios"][name]["q"]["S:go"] == float(returns["new_go"])
        assert d["scenarios"][name]["choice"] == "exit"
    assert d["scenarios"]["none"]["nextExperience"]["return"] == 0
    assert d["scenarios"]["hybrid"]["nextExperience"]["return"] == float(returns["exit"])
    assert d["scenarios"]["hybrid"]["counts"]["modelCalls"] == 2
    assert d["scenarios"]["textbookPriority"]["counts"]["modelCalls"] == 3
    assert d["scenarios"]["textbookPriority"]["counts"]["directBackups"] == 0
    l = warmup()
    l.observe(CHANGED[1])
    assert not l.pred["G"] and l.pred["D"] == {("A", "cross")}
    queue = {}
    l.enqueue_preds("A", queue)
    assert queue == {("S", "go"): F(9, 10)}
    l.pop(queue)
    queue[("S", "go")] = F(9, 10)
    before = l.counts["planningBackups"]
    assert l.pop(queue)[1]
    assert l.counts["planningBackups"] == before  # stale pop still costs a model read
    try:
        Learner().plan(("S", "go"))
    except ValueError:
        pass
    else:
        raise AssertionError("unobserved model access accepted")
    assert target(warmup().q, OLD[0], F(0)) == 0  # gamma=0 removes delayed value
    probabilities = [F(3, 4), F(1, 4)]
    values = [F(1), F(0)]
    mean = sum(p*v for p,v in zip(probabilities, values))
    assert mean == F(3, 4) and sum(p*(v-mean)**2 for p,v in zip(probabilities, values)) == F(3, 16)
    return "PASS: exact routes, model/predecessor replacement, queue accounting, stale/unknown/zero-discount checks"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        status = self_test()
        if not args.json:
            print(status)
    if args.json or not args.test:
        print(json.dumps(walkthrough(), ensure_ascii=False, indent=2))
