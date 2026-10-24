"""Exact small games for understanding MARL learning objectives (MIT).

No neural algorithm or paper benchmark is reproduced here. Each experiment
isolates one question using a fully specified finite game. All expectations
are enumerated, so the plots have no sampling confidence bands.

Run: python3 marl_objectives_lab.py test
     python3 marl_objectives_lab.py demo --out results/marl-objectives
Only the explicitly selected output directory receives generated artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import unittest


def distribution(values):
    if not values or any(v < 0 for v in values) or abs(sum(values)-1) > 1e-10:
        raise ValueError("Expected a probability distribution")
    return list(values)


def value(matrix, row, col):
    distribution(row)
    distribution(col)
    if len(matrix) != len(row) or any(len(r) != len(col) for r in matrix):
        raise ValueError("Matrix and policy dimensions differ")
    return sum(row[i]*col[j]*matrix[i][j]
               for i in range(len(row)) for j in range(len(col)))


# BEGIN structured_exploration
def exploration_probabilities(agents):
    """Reward is 1 iff all agents choose 1 in a one-step cooperative game.

    Both policies have Bernoulli(1/2) marginals. The second protocol permits
    a shared random bit Z visible to every actor before acting. This is an
    information/coordination assumption, not free extra power for CTDE.
    """
    if type(agents) is not int or agents < 1:
        raise ValueError("agents must be a positive integer")
    return {"independent": 2.0**(-agents), "shared_latent": .5}
# END structured_exploration


# BEGIN credit_variance
def credit_variances():
    """Enumerate four equally likely joint actions, Q(a,b)=a+10b.

    Agent one's Bernoulli logit is zero. Its score is a-1/2. Teammate
    b is sampled independently. E[g]=1/4 for all three baselines.
    A conditional COMA baseline averages ONLY our unobserved action:
    b_COMA(b)=sum_a pi(a)Q(a,b)=1/2+10b.
    This special additive game gives zero variance; that is not a general
    COMA guarantee, nor a causal interpretation of a learned critic.
    """
    result = {}
    for name in ("none", "state_value", "counterfactual"):
        samples = []
        for a in (0, 1):
            for b in (0, 1):
                baseline = {"none": 0., "state_value": 5.5,
                            "counterfactual": .5+10*b}[name]
                samples.append((a-.5)*(a+10*b-baseline))
        mean = sum(samples)/4
        result[name] = {"mean": mean,
                        "variance": sum((x-mean)**2 for x in samples)/4}
    return result
# END credit_variance


# BEGIN sequential_updates
def team_update(matrix, action, sequential):
    """Exact coordinate best response; ties retain the incumbent action.

    Simultaneous updates evaluate both changes against the OLD teammate.
    Sequential updates evaluate the second against the NEW first agent.
    Fixed reward, exact values and independently controlled policies are
    essential here. PPO with shared parameters does not satisfy this test.
    """
    a, b = action
    best_a = max(range(len(matrix)), key=lambda i: (matrix[i][b], i == a))
    other_a = best_a if sequential else a
    best_b = max(range(len(matrix[0])),
                 key=lambda j: (matrix[other_a][j], j == b))
    return best_a, best_b


def update_paths(rounds=8):
    matrix = [[0., 1.], [1., 0.]]  # Team succeeds iff actions differ.
    paths = {}
    for name, sequential in [("simultaneous", False), ("sequential", True)]:
        action = (0, 0)
        rows = []
        for k in range(rounds+1):
            rows.append({"round": k, "actions": list(action),
                         "return": matrix[action[0]][action[1]]})
            action = team_update(matrix, action, sequential)
        paths[name] = rows
    return paths
# END sequential_updates


RPS = [[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]]


# BEGIN population_evaluation
def saddle_gap(matrix, row, col):
    """Full-game zero-sum gap, not win rate against one training opponent."""
    distribution(row)
    distribution(col)
    upper = max(sum(a*b for a, b in zip(r, col)) for r in matrix)
    lower = min(sum(row[i]*matrix[i][j] for i in range(len(row)))
                for j in range(len(col)))
    return upper-lower


def fictitious_play(rounds=300):
    """Symmetric RPS: BR to opponent's empirical average, not latest action.

    Start with one rock observation. Both roles have the same population
    by game symmetry; a best response is computed by full enumeration.
    Tie-breaking selects the lowest action index. Report BOTH the latest
    pure policy and the empirical mixture, which are different objects.
    This is exact matrix-game fictitious play, not NFSP or neural PSRO.
    """
    counts = [1, 0, 0]
    rows = []
    for k in range(rounds+1):
        mixture = [c/sum(counts) for c in counts]
        scores = [sum(a*b for a, b in zip(r, mixture)) for r in RPS]
        action = max(range(3), key=lambda i: scores[i])
        pure = [float(i == action) for i in range(3)]
        rows.append({"round": k, "mean_gap": saddle_gap(RPS, mixture, mixture),
                     "latest_gap": saddle_gap(RPS, pure, pure),
                     "mixture": mixture, "best_response": action})
        counts[action] += 1
    return rows


def partner_shift():
    """Coordination reward I[a=b]. Fixed and changing objectives differ."""
    matrix = [[1., 0.], [0., 1.]]
    policies = {"train_specialist": [1., 0.], "balanced": [.5, .5]}
    partners = {"training": [.9, .1], "held_out": [0., 1.]}
    return {name: {group: value(matrix, pi, mu) for group, mu in partners.items()}
            for name, pi in policies.items()}


def expanding_pool_counterexample():
    """Exact double-oracle expansion can increase full-game exploitability.

    Both pools initially contain action 0 only. Action 1 is each player's
    exact best response. The expanded 2x2 restricted equilibrium is (1,1).
    But the absent action 2 exploits it with magnitude 10. Pool inclusion
    alone does not make the CURRENT restricted equilibrium's gap monotonic.
    """
    matrix = [[0., -1., 1.], [1., 0., -10.], [-1., 10., 0.]]
    return {"matrix": matrix, "initial_gap": saddle_gap(matrix, [1., 0., 0.], [1., 0., 0.]),
            "expanded_gap": saddle_gap(matrix, [0., 1., 0.], [0., 1., 0.])}
# END population_evaluation


# BEGIN conditional_response
def conditional_response():
    """Isolate conditional vs marginal prediction, not the PR2 optimizer.

    rho(b|a) is a supplied hypothetical response model. It is not a claim
    that a simultaneous opponent observes a before choosing b. Fitting,
    variational inference and policy learning are deliberately not included.
    """
    q = [[4., 0.], [1., 2.]]
    rho = [[.1, .9], [.9, .1]]
    marginal = [.5, .5]  # rho averaged with a uniform prior over our action.
    return {"payoffs": q, "response": rho, "marginal": marginal,
            "marginal_values": [sum(x*y for x, y in zip(r, marginal)) for r in q],
            "conditional_values": [sum(x*y for x, y in zip(r, p))
                                   for r, p in zip(q, rho)]}
# END conditional_response


# BEGIN advantage_decomposition
def advantage_decomposition(matrix, row, col, action):
    """Two-agent telescoping identity at a FIXED reference joint policy."""
    a, b = action
    baseline = value(matrix, row, col)
    prefix = sum(col[j]*matrix[a][j] for j in range(len(col)))
    increments = [prefix-baseline, matrix[a][b]-prefix]
    return increments, matrix[a][b]-baseline
# END advantage_decomposition


def experiments():
    return {"schema_version": 1,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "method": "Exact finite enumeration; no sampled learning curves",
            "exploration": [{"agents": n, **exploration_probabilities(n)}
                            for n in range(1, 13)],
            "credit": credit_variances(), "updates": update_paths(),
            "population": fictitious_play(), "partners": partner_shift(),
            "pool_expansion": expanding_pool_counterexample(),
            "response": conditional_response(),
            "advantage": advantage_decomposition([[3., 0.], [0., 2.]],
                                                 [.5, .5], [.5, .5], (0, 0))}


def line_svg(title, x_label, y_label, series, ymax, ymin=0.):
    """Small reproducible vector plot. Inputs are computed, not hand-drawn."""
    width, height = 760, 370
    left, top, w, h = 80, 58, 620, 230
    xmax = max(p[0] for _, points in series for p in points)
    colors = ["#245a81", "#a44725", "#537c53"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
           f'<title>{html.escape(title)}</title>',
           '<rect width="760" height="370" fill="#ffffff"/>',
           '<g font-family="system-ui, sans-serif" font-size="14" fill="#20394c">',
           f'<text x="80" y="28" font-size="18">{html.escape(title)}</text>']
    for tick in range(5):
        y = top+h-tick*h/4
        val = ymin+(ymax-ymin)*tick/4
        out += [f'<path d="M {left} {y} H {left+w}" stroke="#dee6eb"/>',
                f'<text x="68" y="{y+5}" text-anchor="end">{val:.3g}</text>']
    for tick in range(5):
        x = left+tick*w/4
        out.append(f'<text x="{x}" y="310" text-anchor="middle">{xmax*tick/4:g}</text>')
    for idx, (name, points) in enumerate(series):
        poly = " ".join(f'{left+x/xmax*w:.3f},{top+h-(y-ymin)/(ymax-ymin)*h:.3f}' for x, y in points)
        color = colors[idx % len(colors)]
        out += [f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.5"/>',
                f'<text x="{left+idx*220}" y="355" fill="{color}">{html.escape(name)}</text>']
    out += [f'<text x="390" y="330" text-anchor="middle">{html.escape(x_label)}</text>',
            f'<text x="18" y="175" transform="rotate(-90 18 175)" text-anchor="middle">{html.escape(y_label)}</text>',
            '</g></svg>']
    return "\n".join(out)


def save_results(directory):
    directory.mkdir(parents=True, exist_ok=True)
    data = experiments()
    (directory/"results.json").write_text(json.dumps(data, indent=2)+"\n")
    plots = {
        "exploration.svg": line_svg("A shared latent changes joint exploration", "Number of agents", "P(all choose 1)",
            [("Independent", [(r["agents"], r["independent"]) for r in data["exploration"]]),
             ("Shared random bit", [(r["agents"], r["shared_latent"]) for r in data["exploration"]])], .5),
        "updates.svg": line_svg("Same payoff, different policy update order", "Update round", "Team reward",
            [(k, [(r["round"], r["return"]) for r in rows]) for k, rows in data["updates"].items()], 1.),
        "population.svg": line_svg("Evaluate a policy and a population separately", "Fictitious-play round", "Full-game saddle gap",
            [("Latest pure policy", [(r["round"], r["latest_gap"]) for r in data["population"]]),
             ("Empirical mixture", [(r["round"], r["mean_gap"]) for r in data["population"]])], 2.)}
    for name, svg in plots.items():
        (directory/name).write_text(svg)
    return {"output": str(directory), "files": ["results.json", *plots]}


class MechanismTests(unittest.TestCase):
    def test_distribution_rejects_negative(self):
        with self.assertRaises(ValueError): distribution([-.1, 1.1])

    def test_exploration_probability(self):
        self.assertEqual(exploration_probabilities(6), {"independent": 1/64, "shared_latent": .5})

    def test_exploration_rejects_bad_count(self):
        with self.assertRaises(ValueError): exploration_probabilities(0)

    def test_credit_unbiased(self):
        for r in credit_variances().values(): self.assertAlmostEqual(r["mean"], .25)

    def test_credit_conditional_variance(self):
        r = credit_variances()
        self.assertEqual(r["counterfactual"]["variance"], 0.)
        self.assertGreater(r["state_value"]["variance"], 0.)

    def test_sequential_improvement(self):
        rows = update_paths()["sequential"]
        self.assertEqual(rows[0]["return"], 0.)
        self.assertTrue(all(r["return"] == 1. for r in rows[1:]))

    def test_simultaneous_cycle(self):
        rows = update_paths()["simultaneous"]
        self.assertEqual(rows[0]["actions"], rows[2]["actions"])
        self.assertTrue(all(r["return"] == 0. for r in rows))

    def test_sequential_all_states(self):
        matrix = [[3., 1.], [0., 2.]]
        for a in range(2):
            for b in range(2):
                x, y = team_update(matrix, (a, b), True)
                self.assertGreaterEqual(matrix[x][y], matrix[a][b])

    def test_rps_equilibrium(self):
        self.assertAlmostEqual(saddle_gap(RPS, [1/3]*3, [1/3]*3), 0.)

    def test_rps_pure_gap(self):
        self.assertEqual(saddle_gap(RPS, [1., 0., 0.], [1., 0., 0.]), 2.)

    def test_fictitious_play_average(self):
        rows = fictitious_play()
        self.assertTrue(all(r["latest_gap"] == 2. for r in rows))
        self.assertLess(rows[-1]["mean_gap"], .25)

    def test_partner_objectives_differ(self):
        r = partner_shift()
        self.assertGreater(r["train_specialist"]["training"], r["balanced"]["training"])
        self.assertLess(r["train_specialist"]["held_out"], r["balanced"]["held_out"])

    def test_pool_expansion_not_monotone_exploitability(self):
        r = expanding_pool_counterexample()
        self.assertEqual((r["initial_gap"], r["expanded_gap"]), (2., 20.))

    def test_conditional_changes_choice(self):
        r = conditional_response()
        self.assertGreater(r["marginal_values"][0], r["marginal_values"][1])
        self.assertLess(r["conditional_values"][0], r["conditional_values"][1])

    def test_advantage_telescopes(self):
        for a in range(2):
            for b in range(2):
                terms, joint = advantage_decomposition([[3., 0.], [0., 2.]], [.4, .6], [.7, .3], (a, b))
                self.assertAlmostEqual(sum(terms), joint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["test", "demo"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "test":
        unittest.main(argv=["marl_objectives_lab.py"], verbosity=2)
    elif args.out:
        print(json.dumps(save_results(args.out), indent=2))
    else:
        print(json.dumps(experiments(), indent=2))
