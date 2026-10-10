"""Exact control targets on one supplied episode. Python 3, standard library.

No random sampling or training. JS uses recursive branch targets; this file checks
Q(sigma) independently by a frozen-table TD-error expansion with Fraction arithmetic.
"""
import argparse
import json
from fractions import Fraction as F

GAMMA, ALPHA = F(9, 10), F(1, 4)
Q = {"A": [F(2), F(0)], "B": [F(1), F(5)], "C": [F(4), F(0)]}
Q2 = {"A": [F(2), F(0)], "B": [F(4), F(2)], "C": [F(2), F(3)]}
PI = {"A": [F(3, 4), F(1, 4)], "B": [F(1, 4), F(3, 4)], "C": [F(3, 4), F(1, 4)]}
STATES, ACTIONS, REWARDS = ["A", "B", "C", "T"], [0, 0, 0], [F(1), F(2), F(3)]


def greedy(row):
    return max(range(len(row)), key=lambda a: row[a])  # first action wins a tie


def expected(state, q=Q):
    return sum(p * v for p, v in zip(PI[state], q[state]))


def update(before, target):
    return before + ALPHA * (target - before)


def one_step(action=0, terminated=False):
    if terminated:
        return {k: F(1) for k in ("sarsa", "expected", "q_learning", "double_q1", "double_q2", "double_coin_mean")}
    d1 = F(1) + GAMMA * Q2["B"][greedy(Q["B"])]
    d2 = F(1) + GAMMA * Q["B"][greedy(Q2["B"])]
    return {"sarsa": F(1) + GAMMA * Q["B"][action], "expected": F(1) + GAMMA * expected("B"),
            "q_learning": F(1) + GAMMA * max(Q["B"]), "double_q1": d1, "double_q2": d2, "double_coin_mean": (d1 + d2) / 2}


def n_step(n=2, endpoint="sample", length=3, terminated=True, q=Q):
    h = min(n, length)
    target = sum(GAMMA**k * REWARDS[k] for k in range(h))
    if h < length or not terminated:
        target += GAMMA**h * (expected(STATES[h], q) if endpoint == "expected" else q[STATES[h]][ACTIONS[h]])
    return target


def error_expansion(n=2, sigma=F(0), version="paper"):
    """G-Q(root) = sum of frozen TD errors times discounted branch coefficients."""
    h = min(n, 3)
    target, coefficient = Q["A"][0], F(1)
    for k in range(h):
        s = STATES[k]
        if k + 1 == 3:
            delta = REWARDS[k] - Q[s][ACTIONS[k]]
        else:
            nxt, a = STATES[k + 1], ACTIONS[k + 1]
            tail = expected(nxt) if version == "book_cv" else sigma * Q[nxt][a] + (1 - sigma) * expected(nxt)
            delta = REWARDS[k] + GAMMA * tail - Q[s][ACTIONS[k]]
        target += coefficient * delta
        if k + 1 < h:
            p = PI[STATES[k + 1]][ACTIONS[k + 1]]
            coefficient *= GAMMA * (sigma + (1 - sigma) * p)
    return target


def delayed_updates(n=2):
    q, version, events = {s: list(row) for s, row in Q.items()}, 0, []
    for clock in range(1, 3 + n):
        start = clock - n
        if start < 0:
            events.append({"clock": clock, "observed_transition": clock if clock <= 3 else None, "waiting": True, "read_version": version})
            continue
        h = min(start + n, 3)
        target = sum(GAMMA**(k-start) * REWARDS[k] for k in range(start, h))
        if h < 3:
            target += GAMMA**(h-start) * q[STATES[h]][ACTIONS[h]]
        state, action = STATES[start], ACTIONS[start]
        before, after = q[state][action], update(q[state][action], target)
        events.append({"clock": clock, "observed_transition": clock if clock <= 3 else None, "waiting": False,
                       "state": state, "action": action, "horizon": h, "bootstrap": h < 3,
                       "read_version": version, "target": target, "before": before, "after": after})
        q[state][action], version = after, version + 1
    return {"events": events, "final_q": q}


def double_bias():
    rows = [[F(-1), F(0)], [F(-1), F(2)], [F(1), F(0)], [F(1), F(2)]]
    return {"true_q": [0, 1], "single_max_mean": sum(max(r) for r in rows)/4,
            "double_mean": sum(e[greedy(s)] for s in rows for e in rows)/16,
            "selected_truth_mean": sum([0, 1][greedy(s)] for s in rows)/4, "true_max": 1}


def diagnostics():
    targets = one_step()
    q_after_c = {**Q, "C": [update(F(4), F(3)), F(0)]}
    return {"one_step": {"targets": targets, "updates": {k: update(F(2), v) for k, v in targets.items() if k != "double_coin_mean"},
                         "double_role_expectations": {"mean_target": targets["double_coin_mean"],
                                                      "selected_table_after": update(F(2), targets["double_coin_mean"]),
                                                      "two_table_average_after": F(2) + ALPHA * (targets["double_coin_mean"]-2)/2},
                         "update_scope": "double_q1/double_q2 denote the selected table; double_role_expectations separately tracks the two-table average",
                         "sarsa_conditional_mean": sum(PI["B"][a]*one_step(a)["sarsa"] for a in range(2)), "double_bias": double_bias()},
            "multistep": {"two_step": {"sarsa": n_step(), "endpoint_expected": n_step(endpoint="expected"),
                                       "tree_backup": error_expansion(), "paper_sigma_half": error_expansion(sigma=F(1, 2)),
                                       "book_cv_sigma_half": error_expansion(sigma=F(1, 2), version="book_cv")},
                          "three_step": {"sarsa": n_step(3), "tree_backup": error_expansion(3)}},
            "timing": {"n2": delayed_updates(2), "n3": delayed_updates(3), "truncated_prefix_target": n_step(length=2, terminated=False),
                       "wrong_terminal_prefix_target": n_step(length=2), "terminal_full_return": n_step(3),
                       "wrong_late_boundary_target": n_step(q=q_after_c), "boundary_q_before": 4, "boundary_q_after": F(15, 4)}}


def checks():
    assert one_step()["sarsa"] == F(19, 10)
    assert one_step()["expected"] == sum(PI["B"][a]*one_step(a)["sarsa"] for a in range(2))
    assert one_step()["double_q1"] == F(14, 5)
    assert n_step() == F(151, 25)
    assert error_expansion() == F(2173, 400)
    assert error_expansion(3) == F(8449, 1600)
    for n in (1, 2, 3, 5):
        assert error_expansion(n, F(1)) == n_step(n)
    assert n_step(length=2, terminated=False) - n_step(length=2) == F(81, 25)
    assert len([e for e in delayed_updates(5)["events"] if not e["waiting"]]) == 3
    assert double_bias()["double_mean"] == F(3, 4) < double_bias()["true_max"]
    return 10


def json_value(value):
    if isinstance(value, F):
        return float(value)
    raise TypeError(type(value).__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        print(f"PASS: {checks()} exact deterministic checks; sampled_runs=0")
    else:
        print(json.dumps(diagnostics(), default=json_value, indent=2, ensure_ascii=False))
