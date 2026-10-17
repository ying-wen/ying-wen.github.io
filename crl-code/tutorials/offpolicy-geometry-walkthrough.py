"""Independent Fraction reference for the two-state off-policy example.

Run: python3 offpolicy-geometry-walkthrough.py [--json]
No dependencies, simulation, random draw, or policy optimization.
"""
from fractions import Fraction as Q
import argparse
import json


def calculate():
    gamma, alpha, beta = Q(9, 10), Q(1, 100), Q(1, 20)
    behavior, x = [Q(9, 10), Q(1, 10)], [Q(1), Q(2)]
    # Matrix definitions independently of the JS event-update implementation.
    C = sum(d * f * f for d, f in zip(behavior, x))
    A = sum(d * f * (f - gamma * x[1]) for d, f in zip(behavior, x))
    H = -A / C
    events = []
    for state in range(2):
        for action in range(2):
            rho = Q(10) if action == 1 else Q(0)
            delta, hx = gamma * x[action] - x[state], x[state] * H
            events.append(dict(state=state, action=action,
                probability=behavior[state] * behavior[action], rho=rho, delta=delta, hx=hx,
                td=rho * delta * x[state], gtd2=rho * (x[state] - gamma * x[action]) * hx,
                tdc=rho * (delta * x[state] - gamma * x[action] * hx),
                auxiliary=(rho * delta - hx) * x[state]))
    expected = {key: sum(e['probability'] * e[key] for e in events)
                for key in ['td', 'gtd2', 'tdc', 'auxiliary']}
    # Solve m = d_b + gamma P_pi^T m: m_A=d_A; m_B=d_B+gamma*(m_A+m_B).
    m = [behavior[0], (behavior[1] + gamma * behavior[0]) / (1 - gamma)]
    CM = sum(d * f * f for d, f in zip(m, x))
    AM = sum(d * f * (f - gamma * x[1]) for d, f in zip(m, x))
    fixed = [Q(9, 13), Q(-45, 34), Q(45, 146)]
    # Closed linear recurrence matrices, not probability-averaging JS updates.
    trajectories = {}
    for name in ['gtd2', 'tdc']:
        w, h = Q(1), Q(0)
        rows = [dict(t=0, w=w, h=h)]
        for t in range(1, 601):
            if name == 'gtd2':
                new_w = w + alpha * A * h
            else:
                new_w = w + alpha * (-A * w - (C - A) * h)
            new_h = h + beta * (-A * w - C * h)
            w, h = new_w, new_h
            rows.append(dict(t=t, w=w, h=h))
        trajectories[name] = rows
    # Follow-on uses the PREVIOUS action ratio. The value update uses old w.
    recorded, state, w, F, previous_rho = [], 0, Q(1), Q(0), Q(0)
    for t, action in enumerate([1, 1, 0, 1]):
        old_w = w
        F = Q(1) + gamma * previous_rho * F
        rho = Q(10) if action == 1 else Q(0)
        trace = rho * F * x[state]
        delta = gamma * x[action] * w - x[state] * w
        w += alpha * delta * trace
        recorded.append(dict(t=t, state=state, action=action, oldW=old_w,
                             F=F, rho=rho, trace=trace, delta=delta, w=w))
        state, previous_rho = action, rho
    result = dict(moments=dict(A=A, C=C, b=Q(0), bootstrapCross=C-A), events=events,
        directionsAtAuxiliaryEquilibrium=dict(w=Q(1), h=H, **expected),
        emphasis=dict(mass=m, normalized=[d/sum(m) for d in m], conditionalFollowon=[m[i]/behavior[i] for i in range(2)],
                      moments=dict(A=AM, C=CM, b=Q(0), bootstrapCross=CM-AM), recordedPath=recorded),
        rewardVariant=dict(rewards=[Q(1), Q(0)], trueValues=[Q(1), Q(0)],
            fixedPoints=[dict(name=name, w=w, values=[w, 2*w], VEbehavior=behavior[0]*(w-1)**2+behavior[1]*(2*w)**2)
                for name, w in zip(['value-regression', 'behavior-MSPBE', 'emphatic-projection'], fixed)]),
        trajectories=trajectories)
    assert A == Q(-17, 25) and C == Q(13, 10)
    assert expected['gtd2'] == expected['tdc'] == Q(-578, 1625)
    assert m == [Q(9, 10), Q(91, 10)] and AM == Q(73, 25)
    assert [r['F'] for r in recorded] == [1, 10, 91, 1]
    assert [r['w'] for r in recorded] == [Q(27, 25), Q(81, 125), Q(81, 125), Q(2187, 3125)]
    return result


def numeric(value):
    if isinstance(value, Q):
        return float(value)
    if isinstance(value, list):
        return [numeric(item) for item in value]
    if isinstance(value, dict):
        return {key: numeric(item) for key, item in value.items()}
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='print all exact-reference values as JSON floats')
    args = parser.parse_args()
    result = calculate()
    if args.json:
        print(json.dumps(numeric(result), ensure_ascii=False))
    else:
        print('Zero-reward model: A=-17/25, C=13/10, h*=34/65 at w=1')
        print('At h*: TD=17/25; GTD2=TDC=-578/1625; auxiliary direction=0')
        for event in result['events']:
            print({key: str(value) for key, value in event.items()})
        print('ETD stationary emphasis masses:', result['emphasis']['mass'])
        print('Given ETD path:', [(r['F'], r['trace'], r['w']) for r in result['emphasis']['recordedPath']])
        print('Reward variant r=(1,0), true v=(1,0):')
        print([(p['name'], str(p['w']), str(p['VEbehavior'])) for p in result['rewardVariant']['fixedPoints']])
        print('600 fixed-data-law updates:', {name: numeric(rows[-1]) for name, rows in result['trajectories'].items()})
