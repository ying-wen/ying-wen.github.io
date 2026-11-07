"""Exact teaching calculations for distributed RL; standard library only.

Run: python3 distributed_systems_lab.py test
     python3 distributed_systems_lab.py demo

These are estimator and scheduling checks, not a distributed training benchmark.
Original teaching code: MIT. See the chapter for IMPALA and GEAR source papers.
"""
import json
import math
import sys
import unittest


# BEGIN vtrace
def vtrace(rewards, discounts, values, behavior_probs, target_probs,
           rho_cap=1.0, c_cap=1.0, pg_cap=1.0):
    """One unpadded trajectory of T transitions; values has T+1 entries.

    Values and target probabilities use the same fixed learner snapshot. A true
    terminal transition has discount 0. A rollout cut keeps its discount and
    supplies the bootstrap value. Behavior probabilities are saved at action
    selection, not recomputed using current parameters. Returned targets are
    constants (stop-gradient) when used in an autodiff actor-critic loss.
    """
    T = len(rewards)
    if T < 1 or len(values) != T + 1 or any(
        len(a) != T for a in (discounts, behavior_probs, target_probs)
    ):
        raise ValueError("one probability/discount per transition; T+1 values")
    arrays = (rewards, discounts, values, behavior_probs, target_probs)
    if not all(math.isfinite(x) for a in arrays for x in a):
        raise ValueError("inputs must be finite")
    if not all(0 <= d <= 1 for d in discounts):
        raise ValueError("discounts must lie in [0,1]")
    if not all(0 < p <= 1 for p in behavior_probs):
        raise ValueError("a recorded action must have positive behavior support")
    if not all(0 <= p <= 1 for p in target_probs):
        raise ValueError("target action probabilities must lie in [0,1]")
    if not all(math.isfinite(c) and c > 0 for c in (rho_cap, c_cap, pg_cap)):
        raise ValueError("caps must be positive and finite")
    if c_cap > min(1.0, rho_cap):
        raise ValueError("this implementation uses c_cap <= min(1, rho_cap)")
    ratios = [p / mu for p, mu in zip(target_probs, behavior_probs)]
    rhos = [min(rho_cap, r) for r in ratios]
    cs = [min(c_cap, r) for r in ratios]
    deltas = [rhos[t] * (rewards[t] + discounts[t] * values[t+1] - values[t])
              for t in range(T)]
    targets = [0.0] * T
    correction = 0.0  # At the rollout boundary v_T = V_T.
    for t in reversed(range(T)):
        correction = deltas[t] + discounts[t] * cs[t] * correction
        targets[t] = values[t] + correction
    next_targets = targets[1:] + [values[-1]]
    advantages = [min(pg_cap, ratios[t]) * (
        rewards[t] + discounts[t] * next_targets[t] - values[t])
        for t in range(T)]
    return {"ratios": ratios, "deltas": deltas,
            "targets": targets, "pg_advantages": advantages}
# END vtrace


def queue_trace(ticks=6, arrivals=2, service=1, capacity=None):
    """Deterministic FIFO: arrivals first, drop oldest if full, then serve.

    Each abstract tick publishes one version, independent of training. This
    isolates queue residence time; ticks are not measured seconds or SGD steps.
    """
    if any(not isinstance(x, int) or x < 1 for x in (ticks, arrivals, service)):
        raise ValueError("ticks/arrivals/service must be positive integers")
    if capacity is not None and (not isinstance(capacity, int) or capacity < 1):
        raise ValueError("capacity must be a positive integer")
    queue, events, next_id, discarded = [], [], 0, 0
    for tick in range(ticks):
        for _ in range(arrivals):
            queue.append({"id": next_id, "birth": tick, "version": tick})
            next_id += 1
        if capacity is not None and len(queue) > capacity:
            n = len(queue) - capacity
            queue = queue[n:]
            discarded += n
        used = queue[:service]
        queue = queue[len(used):]
        events.append({"tick": tick, "queued": len(queue), "discarded": discarded,
                       "used": [{**x, "age": tick-x["birth"],
                                 "version_lag": tick-x["version"]} for x in used]})
    return events


def demo():
    return {
        "kind": "exact_calculation_not_training",
        "vtrace": vtrace([1, 2], [.9, 0], [.4, .8, 0], [.5, .5], [.25, .75]),
        "fifo": queue_trace(), "bounded_fifo": queue_trace(capacity=2),
        "shards": {"sizes": [1, 3], "equal_shard_sampling": [.5, 1/6, 1/6, 1/6],
                   "size_weighted_shards": [.25] * 4},
    }


class Checks(unittest.TestCase):
    def test_hand_calculation(self):
        result = demo()["vtrace"]
        for actual, expected in zip(result["targets"], [1.6, 2.]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(result["pg_advantages"][0], 1.2)

    def test_on_policy_telescopes_to_bootstrapped_return(self):
        r, d, v = [1., -2., 3.], [.9, .8, .7], [.4, -.3, .2, 4.]
        result = vtrace(r, d, v, [.5]*3, [.5]*3)
        expected, tail = [], v[-1]
        for reward, discount in reversed(list(zip(r, d))):
            tail = reward + discount * tail
            expected.insert(0, tail)
        for a, b in zip(result["targets"], expected):
            self.assertAlmostEqual(a, b)

    def test_reverse_recursion_equals_explicit_forward_sum(self):
        r, d, v, mu, pi = [1, -1, 2], [.8, .9, 0], [.2, .6, .3, 0], [.5]*3, [.1, .9, .3]
        result = vtrace(r, d, v, mu, pi)
        for s in range(3):
            acc, weight = v[s], 1.
            for t in range(s, 3):
                acc += weight * result["deltas"][t]
                weight *= d[t] * min(1., pi[t]/mu[t])
            self.assertAlmostEqual(acc, result["targets"][s])

    def test_terminal_and_rollout_cut_are_different(self):
        terminal = vtrace([1], [0], [0, 100], [1], [1])
        cut = vtrace([1], [.9], [0, 100], [1], [1])
        self.assertEqual(terminal["targets"], [1])
        self.assertEqual(cut["targets"], [91])

    def test_rho_clipping_changes_expected_fixed_point(self):
        # One-state terminal bandit: target policy value=.8; clipped value=.5.
        mu, pi, rewards = [.8, .2], [.2, .8], [0, 1]
        mass = [min(a, b) for a, b in zip(mu, pi)]
        fixed = sum(w*r for w, r in zip(mass, rewards))/sum(mass)
        drift = sum(mu[a]*min(1, pi[a]/mu[a])*(rewards[a]-fixed) for a in range(2))
        self.assertAlmostEqual(fixed, .5)
        self.assertAlmostEqual(drift, 0)
        self.assertAlmostEqual(sum(pi[a]*rewards[a] for a in range(2)), .8)

    def test_queue_conservation_and_age(self):
        for capacity in (None, 2):
            events = queue_trace(capacity=capacity)
            served = 0
            for e in events:
                served += len(e["used"])
                self.assertEqual(2*(e["tick"]+1), served+e["queued"]+e["discarded"])
            self.assertEqual(events[-1]["used"][0]["age"], 3 if capacity is None else 0)

    def test_unequal_shards(self):
        # Pick shard proportional to size, then uniformly within shard.
        probabilities = [size/4/size for size in [1, 3] for _ in range(size)]
        self.assertEqual(probabilities, [.25]*4)

    def test_invalid_probability_and_lengths(self):
        for mu in (0, -.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                vtrace([1], [.9], [0, 0], [mu], [.5])
        with self.assertRaises(ValueError):
            vtrace([1], [.9], [0], [.5], [.5])


if __name__ == '__main__':
    if sys.argv[1:] == ['test']:
        unittest.main(argv=[sys.argv[0]])
    elif sys.argv[1:] == ['demo']:
        print(json.dumps(demo(), ensure_ascii=False, indent=2))
    else:
        raise SystemExit(__doc__)
