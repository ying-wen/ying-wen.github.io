"""Exact expectations for learning-time reward and a frozen diagnostic.

Run: python3 lifetime_evaluation_walkthrough.py
Only the standard library is required. No random training is performed.

The environment has three locations J, L, R. A circle consists of a chosen
J -> side transition with Bernoulli reward and a forced side -> J transition
with zero reward. Only the first transition updates the direction bit.
The schedule and reward probabilities belong to the environment, not the agent.
"""
from collections import defaultdict
from fractions import Fraction as F
import json


def next_side(side, reward, learning=True):
    """The complete learner update: no clock, phase, or unchosen reward input."""
    if side not in ("L", "R") or reward not in (0, 1):
        raise ValueError("side must be L/R and arrival reward must be 0/1")
    return ("R" if side == "L" else "L") if learning and reward == 0 else side


def expectation(good=F(9, 10), block=8, diagnostic_circles=8):
    """Enumerate probability mass over (next direction, cumulative reward).

    Merging histories with the same pair is exact for this learner. Fractions
    prevent numerical roundoff. These masses are analyst-side computations;
    a deployed agent stores only its one direction bit.
    """
    good = F(good)
    if not F(1, 2) <= good <= 1 or block < 1 or diagnostic_circles < 1:
        raise ValueError("require 1/2 <= good <= 1 and positive circle counts")
    mass = {("L", 0): F(1)}
    fixed_total = F(0)
    rows = []
    curves = [{"step": 0, "learner": F(0), "fixed_left": F(0)}]
    for circle, phase in enumerate(["A"] * block + ["B"] * block + ["A"] * block, 1):
        probabilities = {"L": good if phase == "A" else 1-good,
                         "R": 1-good if phase == "A" else good}
        left_before = sum((p for (side, _), p in mass.items() if side == "L"), F(0))
        following = defaultdict(F)
        expected_reward = F(0)
        for (side, total), probability in mass.items():
            for reward in (0, 1):
                event = probabilities[side] if reward else 1-probabilities[side]
                weight = probability * event
                following[(next_side(side, reward), total + reward)] += weight
                expected_reward += weight * reward
        mass = dict(following)
        assert sum(mass.values(), F(0)) == 1
        cumulative = sum((p * total for (_, total), p in mass.items()), F(0))
        fixed_total += probabilities["L"]
        left_after = sum((p for (side, _), p in mass.items() if side == "L"), F(0))
        rows.append(dict(circle=circle, phase=phase, left_before=left_before,
                         expected_reward=expected_reward, left_after=left_after,
                         cumulative=cumulative, fixed_left_cumulative=fixed_total))
        # Return-to-J is a real zero-reward step, NOT another lose-shift update.
        curves.extend([dict(step=2*circle-1, learner=cumulative, fixed_left=fixed_total),
                       dict(step=2*circle, learner=cumulative, fixed_left=fixed_total)])

    # Fork into a fixed-A diagnostic. Freeze the bit within each history; do not
    # resample a direction every diagnostic circle or feed outcomes back to mass.
    checkpoint = mass.copy()
    diagnostic_mean = sum((p * (good if side == "L" else 1-good)
                           for (side, _), p in checkpoint.items()), F(0))
    result = dict(kind="exact-expectation", sampled_runs=0,
                  training_steps=6*block, diagnostic_steps=2*diagnostic_circles,
                  good_reward_probability=good, rows=rows, curves=curves,
                  lifetime_learner=rows[-1]["cumulative"], lifetime_fixed_left=fixed_total,
                  checkpoint_left_probability=rows[-1]["left_after"],
                  diagnostic_learner_per_circle=diagnostic_mean,
                  diagnostic_fixed_left_per_circle=good,
                  diagnostic_learner_total=diagnostic_circles*diagnostic_mean,
                  diagnostic_fixed_left_total=diagnostic_circles*good)
    assert checkpoint == mass  # The diagnostic cannot write into the main life.
    return result


def numeric(value):
    if isinstance(value, F):
        return float(value)
    if isinstance(value, dict):
        return {key: numeric(item) for key, item in value.items()}
    if isinstance(value, list):
        return [numeric(item) for item in value]
    return value


def test():
    d = expectation()
    assert d["lifetime_learner"] == F(462, 25)  # 18.48
    assert d["lifetime_fixed_left"] == F(76, 5)  # 15.2
    assert d["diagnostic_learner_total"] == F(164, 25)  # 6.56
    assert d["diagnostic_fixed_left_total"] == F(36, 5)  # 7.2
    assert d["rows"][8]["expected_reward"] == F(9, 50)
    assert d["rows"][16]["expected_reward"] == F(9, 50)
    assert d["checkpoint_left_probability"] == F(9, 10)
    for row in d["rows"]:
        ql = F(9, 10) if row["phase"] == "A" else F(1, 10)
        qr = 1 - ql
        assert row["left_after"] == row["left_before"]*ql + (1-row["left_before"])*(1-qr)
    for first, second in zip(d["curves"][1::2], d["curves"][2::2]):
        assert first["learner"] == second["learner"]
    deterministic = expectation(F(1))
    assert deterministic["lifetime_learner"] == 22
    assert deterministic["lifetime_fixed_left"] == 16
    assert deterministic["diagnostic_learner_per_circle"] == 1
    uninformative = expectation(F(1, 2))
    assert uninformative["lifetime_learner"] == uninformative["lifetime_fixed_left"] == 12
    assert uninformative["diagnostic_learner_total"] == 4
    assert next_side("L", 0) == "R"
    assert next_side("L", 0, learning=False) == "L"


if __name__ == "__main__":
    test()
    print(json.dumps(numeric(expectation()), indent=2))
