#!/usr/bin/env python3
"""Two-step settlement: Fraction Bellman/projection/CVaR and one logit step.

Save this file in an empty directory. Python standard library only.
python3 distributional-walkthrough.py [--test | --json]
No RNG, neural training, or benchmark scores are produced.
"""
import argparse
import json
import math
from fractions import Fraction as F

SUPPORT = tuple(map(F, range(5)))


def law(action, advance=F(1, 2), gamma=F(1, 2), q=F(1, 2)):
    if not 0 <= gamma <= 1 or not 0 <= q <= 1:
        raise ValueError("invalid discount or probability")
    if action == "fixed":
        return [(advance + gamma * 2, F(1))]
    if action == "floating":
        return [(advance, 1 - q), (advance + gamma * 4, q)]
    raise ValueError("unknown action")


def expectation(outcomes):
    return sum(z * p for z, p in outcomes)


def cvar(outcomes, eta=F(1, 2)):
    """Integrate the inverse CDF over [0, eta], including partial atoms."""
    if not 0 < eta <= 1:
        raise ValueError("invalid tail mass")
    if sum(p for _, p in outcomes) != 1 or any(p < 0 for _, p in outcomes):
        raise ValueError("invalid distribution")
    area, cumulative = F(0), F(0)
    for z, p in sorted(outcomes):
        right = cumulative + p
        overlap = max(F(0), min(eta, right) - min(eta, cumulative))
        area += z * overlap
        cumulative = right
    return area / eta


def cvar_optimization(outcomes, eta):
    """Independent variational identity: maximize b-E[(b-Z)+]/eta."""
    return max(b - sum(p * max(F(0), b-z) for z, p in outcomes) / eta
               for b in {z for z, _ in outcomes})


def project(reward, next_masses, gamma=F(1, 2), terminal=False, support=SUPPORT):
    """Fraction floor/ceiling implementation, independent of JS tent kernel."""
    if len(support) < 2 or len(next_masses) != len(support):
        raise ValueError("matching uniform support required")
    delta = support[1]-support[0]
    if delta <= 0 or any(z != support[0]+i*delta for i,z in enumerate(support)):
        raise ValueError("uniform increasing support required")
    if sum(next_masses) != 1 or any(p < 0 for p in next_masses):
        raise ValueError("invalid masses")
    raw = [reward + gamma * (not terminal) * z for z in support]
    clipped = [max(support[0], min(support[-1], z)) for z in raw]
    result = [F(0)]*len(support)
    for y,p in zip(clipped, next_masses):
        b = (y-support[0])/delta
        lo,hi = math.floor(b), math.ceil(b)
        if lo == hi:
            result[lo] += p
        else:
            result[lo] += p*(hi-b)
            result[hi] += p*(b-lo)
    return dict(raw=raw, clipped=clipped, next_masses=list(next_masses), projected=result,
                raw_mean=sum(z*p for z,p in zip(raw,next_masses)),
                projected_mean=sum(z*p for z,p in zip(support,result)),
                clipped_mass=sum(p for z,y,p in zip(raw,clipped,next_masses) if z!=y))


def softmax(logits):
    weights = [math.exp(x-max(logits)) for x in logits]
    return [w/sum(weights) for w in weights]


def loss(logits, target):
    return -sum(m*math.log(p) for m,p in zip(target,softmax(logits)) if m)


def logit_step(target):
    logits = [0.]*5
    before = softmax(logits)
    gradient = [p-float(m) for p,m in zip(before,target)]
    after_logits = [-g for g in gradient]
    return dict(rate=1,before_logits=logits,target=target,before=before,gradient=gradient,
                after_logits=after_logits,after=softmax(after_logits),
                before_loss=loss(logits,target),after_loss=loss(after_logits,target))


def choose(advance=F(1,2), criterion="mean", eta=F(1,2)):
    score = expectation if criterion == "mean" else lambda d: cvar(d,eta)
    scores = dict(fixed=score(law("fixed")),floating=score(law("floating",advance)))
    return dict(criterion=criterion,scores=scores,
                action="floating" if scores["floating"] > scores["fixed"] else "fixed")


def uncertainty(alpha,beta):
    q = F(alpha,alpha+beta)
    var = F(alpha*beta,(alpha+beta)**2*(alpha+beta+1))
    return dict(alpha=alpha,beta=beta,success_probability=q,variance_of_parameter=var,
                predictive_mean=F(1,2)+2*q,expected_conditional_return_variance=4*(q*(1-q)-var),
                variance_of_conditional_mean=4*var,predictive_return_variance=4*q*(1-q))


def data():
    floating = project(F(1,2),[F(1,2),0,0,0,F(1,2)])
    fixed = project(F(1,2),[0,0,1,0,0])
    packed = lambda d: dict(atoms=[z for z,_ in d],masses=[p for _,p in d])
    return dict(kind="constructed-exact-settlement",random_rollouts=0,neural_training_steps=0,exact_logit_steps=1,
        task=dict(gamma=F(1,2),support=SUPPORT,advance=F(1,2),fixed_final_reward=2,
                  floating_final_rewards=[0,4],floating_final_masses=[F(1,2)]*2,terminal_future_return=0),
        nominal=dict(fixed=packed(law("fixed")),floating=packed(law("floating"))),
        bonus=dict(fixed=packed(law("fixed")),floating=packed(law("floating",F(3,5)))),
        projections=dict(fixed=fixed,floating=floating,overflow=project(F(3),[F(1,2),0,0,0,F(1,2)])),
        sample_update=logit_step(floating["projected"]),
        choices=dict(nominal_mean=choose(),bonus_mean=choose(F(3,5)),bonus_risk=choose(F(3,5),"lower_cvar")),
        risk=dict(true_fixed=F(3,2),projected_fixed=cvar(list(zip(SUPPORT,fixed["projected"]))),
                  true_floating=F(1,2),projected_floating=cvar(list(zip(SUPPORT,floating["projected"])))),
        uncertainty=[uncertainty(1,1),uncertainty(50,50)])


def run_checks():
    checks = 0
    # Complete two-reward trajectories independently check recursive labels.
    for gamma in [F(0),F(1,2),F(1)]:
        for q in [F(0),F(1,4),F(1,2),F(1)]:
            recursive = expectation(law("floating",gamma=gamma,q=q))
            enumerated = (1-q)*(F(1,2)+gamma*0)+q*(F(1,2)+gamma*4)
            assert recursive == enumerated
            checks += 1
    for reward in [F(-2),F(0),F(1,2),F(1),F(3),F(5)]:
        for terminal in [False,True]:
            p = project(reward,[F(1,2),0,0,0,F(1,2)],terminal=terminal)
            assert sum(p["projected"]) == 1
            assert p["projected_mean"] == sum(z*m for z,m in zip(p["clipped"],p["next_masses"]))
            if terminal:
                assert len(set(p["raw"])) == 1
            checks += 1
    assert project(F(2),[F(1,2),0,0,0,F(1,2)],terminal=True)["projected"] == [0,0,1,0,0]
    assert project(F(1,2),[0,0,1,0,0],gamma=F(0))["projected"] == [F(1,2),F(1,2),0,0,0]
    for outcomes in [law("fixed"),law("floating"),[(F(-1),F(1,4)),(F(3),F(3,4))]]:
        for eta in [F(1,8),F(1,2),F(3,4),F(1)]:
            assert cvar(outcomes,eta) == cvar_optimization(outcomes,eta)
            checks += 1
    target = data()["projections"]["floating"]["projected"]
    for logits in [[0.]*5,[.1,-.2,.3,.4,-.5]]:
        gradient = [p-float(m) for p,m in zip(softmax(logits),target)]
        for i,g in enumerate(gradient):
            h=1e-5;plus=logits.copy();minus=logits.copy();plus[i]+=h;minus[i]-=h
            assert abs((loss(plus,target)-loss(minus,target))/(2*h)-g)<1e-8
            checks += 1
    assert choose()["action"] == "fixed"
    assert choose(F(3,5))["action"] == "floating"
    assert choose(F(3,5),"lower_cvar")["action"] == "fixed"
    assert data()["projections"]["overflow"]["raw_mean"] == 4
    assert data()["projections"]["overflow"]["projected_mean"] == F(7,2)
    for alpha,beta in [(1,1),(50,50)]:
        d=uncertainty(alpha,beta)
        assert d["expected_conditional_return_variance"]+d["variance_of_conditional_mean"] == d["predictive_return_variance"] == 1
        checks += 1
    print(f"{checks+7} checks passed; exact mechanisms only, zero random rollouts.")


def numeric(value):
    if isinstance(value,F):
        return float(value)
    if isinstance(value,dict):
        return {k:numeric(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [numeric(v) for v in value]
    return value


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test",action="store_true")
    parser.add_argument("--json",action="store_true")
    args=parser.parse_args()
    if args.test:
        run_checks()
    elif args.json:
        print(json.dumps(numeric(data()),ensure_ascii=False,indent=2))
    else:
        d=data()
        print("Known two-step settlement; gamma=1/2, support=0,1,2,3,4")
        for key,p in d["projections"].items():
            print(key,"raw:",list(map(str,p["raw"])),"target:",list(map(str,p["projected"])),"means:",p["raw_mean"],"->",p["projected_mean"])
        print("one frozen-target logit step:",numeric(d["sample_update"]))
        print("true-law choices:",numeric(d["choices"]))
        print("true / projected lower CVaR:",numeric(d["risk"]))
        print("same predictive variance, different parameter uncertainty:",numeric(d["uncertainty"]))
