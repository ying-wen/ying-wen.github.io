"""Original complete every-visit weighted-IS MC control, standard library only.

Run from any directory: python3 mc-offpolicy-control-walkthrough.py --test
or --json (default also prints JSON). No random training or third-party imports.
Sutton & Barto (2020), Sections 5.1, 5.6, 5.7, pp.92,109-111.
The two-step task is shared with the preceding on-policy tutorial. A separate
proper one-state loop exposes repeat visits. Exact values are reader's oracles,
never inputs to update_weighted_control. Fractions keep all enumerations exact.
"""
import argparse
from fractions import Fraction
from itertools import product
import json
import unittest

F = Fraction
GAMMA, EPSILON = F(1, 2), F(1, 5)
REWARDS = ((2, 0), (4, 0))


def rational(x):
    return x if isinstance(x, F) else F(str(x))


def greedy(q):
    return [0 if row[0] > row[1] else 1 for row in q]


def behavior(q, epsilon=EPSILON):
    epsilon = rational(epsilon)
    if not 0 < epsilon <= 1:
        raise ValueError("Positive epsilon is needed for support")
    return [[epsilon / 2 + (1 - epsilon if a == g else 0) for a in (0, 1)]
            for g in greedy(q)]


def environment_step(state, action):
    if state not in (0, 1) or action not in (0, 1):
        raise ValueError("Invalid state/action")
    return state + 1, REWARDS[state][action], state == 1


def episode_from_actions(policy, actions):
    if len(policy) != 2 or len(actions) != 2:
        raise ValueError("Two decisions are required")
    if any(len(row) != 2 or min(row) <= 0 or sum(row) != 1 for row in policy):
        raise ValueError("Positive action probabilities must sum to one")
    state, rewards, logged = 0, [], []
    for a in actions:
        if a not in (0, 1):
            raise ValueError("Illegal action")
        logged.append(policy[state][a])
        state, reward, terminated = environment_step(state, a)
        rewards.append(reward)
    return {"path": "".join("AB"[a] for a in actions), "states": [0, 1],
            "actions": list(actions), "rewards": rewards,
            "behavior": [list(row) for row in policy], "logged_probabilities": logged,
            "probability": logged[0] * logged[1], "terminated": terminated, "cutoff": False}


def sample_episode(policy, uniforms):
    if len(uniforms) != 2 or any(not 0 <= rational(u) < 1 for u in uniforms):
        raise ValueError("Two uniform values in [0,1) are required")
    actions = [0 if rational(u) < policy[s][0] else 1 for s, u in enumerate(uniforms)]
    return dict(episode_from_actions(policy, actions), uniforms=list(map(rational, uniforms)))


def update_weighted_control(q, c, episode, gamma=GAMMA):
    if not episode["terminated"] or episode["cutoff"]:
        raise ValueError("A cutoff is not true termination; MC return is unavailable")
    gamma = rational(gamma)
    if not 0 <= gamma <= 1:
        raise ValueError("Invalid discount")
    Q, C = [list(row) for row in q], [list(row) for row in c]
    target, trace, g, w, break_time = greedy(Q), [], F(0), F(1), None
    T = len(episode["states"])
    if not T or any(len(episode[k]) != T for k in ("actions", "rewards", "logged_probabilities")):
        raise ValueError("Incomplete transition log")
    for t in reversed(range(T)):
        s, a = episode["states"][t], episode["actions"][t]
        p = rational(episode["logged_probabilities"][t])
        if s not in range(len(Q)) or a not in (0, 1) or not 0 < p <= 1:
            raise ValueError("Invalid sampled transition")
        before = {"q": [list(row) for row in Q], "c": [list(row) for row in C], "target": list(target)}
        g = rational(episode["rewards"][t]) + gamma * g
        C[s][a] += w
        Q[s][a] += w / C[s][a] * (g - Q[s][a])
        target[s] = 0 if Q[s][0] > Q[s][1] else 1
        mismatch = a != target[s]
        next_w = F(0) if mismatch else w / p
        trace.append({"t": t, "state": s, "action": a, "reward": episode["rewards"][t],
            "return": g, "weight": w, "logged_probability": p, "before": before,
            "q_after": [list(row) for row in Q], "c_after": [list(row) for row in C],
            "target_after": list(target), "break": mismatch, "next_weight": next_w})
        if mismatch:
            break_time = t
            break
        w = next_w
    return {"q": Q, "c": C, "target": target, "trace": trace,
            "break_time": break_time, "skipped_times": list(range(break_time or 0))}


def exact_values(q):
    """Closed Bellman algebra, not a learner label."""
    target, b = greedy(q), behavior(q)
    return {"target": 2 * (target[0] == 0) + 2 * (target[1] == 0),
            "behavior": 2 * b[0][0] + 2 * b[1][0],
            "current_target_q": [[2 + 2 * (target[1] == 0), 2 * (target[1] == 0)], [4, 0]]}


def adaptive_branches(episodes):
    if episodes not in (1, 2):
        raise ValueError("Enumerate one or two episodes")
    nodes = [{"paths": [], "probability": F(1), "q": [[0, 0], [0, 0]], "c": [[0, 0], [0, 0]]}]
    for _ in range(episodes):
        children = []
        for node in nodes:
            for actions in product((0, 1), repeat=2):
                row = episode_from_actions(behavior(node["q"]), actions)
                result = update_weighted_control(node["q"], node["c"], row)
                children.append({"paths": node["paths"] + [row["path"]],
                    "probability": node["probability"] * row["probability"],
                    "q": result["q"], "c": result["c"], "target": result["target"],
                    "exact": exact_values(result["q"])})
        nodes = children
    return {"episodes": episodes, "branches": nodes,
        "probability_sum": sum(n["probability"] for n in nodes),
        "expected_target_value": sum(n["probability"] * n["exact"]["target"] for n in nodes),
        "expected_behavior_value": sum(n["probability"] * n["exact"]["behavior"] for n in nodes),
        "target_value_distribution": [{"value": v, "probability": sum(n["probability"] for n in nodes if n["exact"]["target"] == v)} for v in (0, 2, 4)]}


def sample_repeated_episode(action_uniforms, transition_uniforms):
    if not action_uniforms or len(action_uniforms) != len(transition_uniforms) or any(not 0 <= rational(u) < 1 for u in (*action_uniforms, *transition_uniforms)):
        raise ValueError("Matching uniform inputs in [0,1) are required")
    states, actions, rewards, logged, probability, terminated = [], [], [], [], F(1), False
    for t, u in enumerate(action_uniforms):
        a = 0 if rational(u) < F(1, 2) else 1
        states.append(0)
        actions.append(a)
        rewards.append(1 if a == 0 else 0)
        logged.append(F(1, 2))
        terminated = a == 1 or rational(transition_uniforms[t]) < F(1, 2)
        probability *= F(1, 2) * (F(1, 2) if a == 0 else 1)
        if terminated:
            break
    return {"path": "".join("AB"[a] for a in actions), "states": states, "actions": actions,
        "rewards": rewards, "logged_probabilities": logged, "terminated": terminated, "cutoff": not terminated,
        "action_uniforms": list(map(rational, action_uniforms[:len(actions)])),
        "transition_uniforms": list(map(rational, transition_uniforms[:len(actions)])), "probability": probability}


def repeated_visit():
    row = sample_repeated_episode((F(1, 10), F(1, 5), F(3, 10)), (F(4, 5), F(7, 10), F(1, 5)))
    return {"task": {"gamma": 1, "A_reward": 1, "A_terminal_probability": F(1, 2),
        "B_reward": 0, "B_terminal_probability": 1, "behavior_A_probability": F(1, 2), "behavior_value": F(2, 3), "target_A_value": 2},
        "episode": row, "returns": [3, 2, 1], "first_visit_mean": 3, "every_visit_mean": 2,
        "incorrect_backward_seen_set": 1, "first_visit_weighted_action_value": 3,
        "first_visit_suffix_weight": 4,
        "every_visit_weighted_control": update_weighted_control([[0, 0]], [[0, 0]], row, 1)}


def walkthrough():
    q, c, rows = [[0, 0], [0, 0]], [[0, 0], [0, 0]], []
    for i, u in enumerate(((F(1, 2), F(1, 20)), (F(1, 20), F(19, 20)), (F(1, 20), F(1, 2)))):
        row = dict(sample_episode(behavior(q), u), episode=i + 1,
            q_before=[list(r) for r in q], c_before=[list(r) for r in c],
            target_before=greedy(q), exact_before=exact_values(q))
        result = update_weighted_control(q, c, row)
        q, c = result["q"], result["c"]
        row.update(result)
        row.update(behavior_after=behavior(q), exact_after=exact_values(q))
        rows.append(row)
    return {"kind": "exact_finite_offpolicy_control",
        "task": {"gamma": GAMMA, "epsilon": EPSILON, "rewards": REWARDS, "tie_action": "B"},
        "assigned_episodes": rows, "adaptive_enumerations": [adaptive_branches(1), adaptive_branches(2)],
        "repeated_visit": repeated_visit(),
        "limitations": {"selected_paths_are_not_expected_gain": True, "no_random_training": True,
            "cutoff_rejected": True, "changing_target_not_fixed_policy_unbiasedness": True}}


def json_numbers(x):
    if isinstance(x, F):
        return float(x)
    if isinstance(x, dict):
        return {k: json_numbers(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_numbers(v) for v in x]
    return x


class ControlTests(unittest.TestCase):
    def test_backward_weighted_sums_and_visit_selection(self):
        r = repeated_visit()
        trace = r["every_visit_weighted_control"]["trace"]
        self.assertEqual([t["weight"] for t in trace], [1, 2, 4])
        self.assertEqual([t["q_after"][0][0] for t in trace], [1, F(5, 3), F(17, 7)])
        self.assertEqual(trace[-1]["c_after"][0][0], 7)
        self.assertEqual(sum(w * g for w, g in zip((4, 2, 1), (3, 2, 1))) / F(7), F(17, 7))
        self.assertEqual((r["first_visit_mean"], r["every_visit_mean"], r["incorrect_backward_seen_set"]), (3, 2, 1))
        # Bellman equation q(A)=1+q(A)/2 gives 2, distinct from one selected sample.
        self.assertNotEqual(trace[-1]["q_after"][0][0], r["task"]["target_A_value"])

    def test_update_then_greedy_then_break(self):
        rows = walkthrough()["assigned_episodes"]
        self.assertEqual([r["path"] for r in rows], ["BA", "AB", "AA"])
        self.assertEqual(rows[0]["trace"][0]["before"]["target"], [1, 1])
        self.assertFalse(rows[0]["trace"][0]["break"])
        self.assertEqual(rows[0]["trace"][1]["weight"], 10)
        self.assertEqual(rows[1]["trace"][0]["c_after"][1][1], 1)
        self.assertEqual(rows[1]["skipped_times"], [0])
        self.assertEqual(rows[2]["c"][0][0], F(10, 9))
        self.assertEqual(rows[2]["q"], [[4, 2], [4, 0]])
        # Changing current behavior never changes row[0]'s recorded .1.
        self.assertEqual(rows[0]["logged_probabilities"], [F(9, 10), F(1, 10)])
        self.assertEqual(rows[0]["behavior_after"][1][0], F(9, 10))

    def test_exhaustive_episodes_and_closed_oracle(self):
        first = adaptive_branches(1)
        self.assertEqual(first["probability_sum"], 1)
        self.assertEqual(first["expected_target_value"], F(2, 5))
        self.assertEqual([n["exact"]["target"] for n in first["branches"]], [4, 2, 2, 0])
        # Two independent terminal A discoveries have failure .9^2.
        # Start A succeeds only after matched tail: independently enumerated in JS tests.
        second = adaptive_branches(2)
        self.assertEqual(len(second["branches"]), 16)
        self.assertEqual(second["probability_sum"], 1)
        self.assertEqual(sum(r["probability"] for r in second["target_value_distribution"]), 1)

    def test_boundaries(self):
        row = episode_from_actions(behavior([[0, 0], [0, 0]]), (0, 0))
        with self.assertRaises(ValueError):
            update_weighted_control([[0, 0], [0, 0]], [[0, 0], [0, 0]], dict(row, terminated=False, cutoff=True))
        with self.assertRaises(ValueError):
            behavior([[0, 0], [0, 0]], 0)
        with self.assertRaises(ValueError):
            sample_episode(behavior([[0, 0], [0, 0]]), (0, 1))
        self.assertEqual(behavior([[4, 0]], 1), [[F(1, 2), F(1, 2)]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print exact finite calculation as numeric JSON")
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=["mc-offpolicy-control-walkthrough.py"], verbosity=2)
    else:
        print(json.dumps(json_numbers(walkthrough()), ensure_ascii=False, indent=2))
