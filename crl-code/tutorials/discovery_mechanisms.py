#!/usr/bin/env python3
"""Inspect question discovery and spectral skill construction, without training.

Run from any directory: python3 discovery_mechanisms.py [--test]
Standard library only. These are exact teaching mechanisms, not reproductions
of the distributed GVF discovery or neural eigenoption experiments.
"""
import json
import math
import sys
import unittest


def question_update(eta, weight=.4, alpha=.2, x=1., next_x=.5, test_x=1., target=1.):
    """One semigradient update; differentiate the actual update in eta.

    The incoming weight and sampled transition are fixed. The bootstrap value
    is a numerical constant for the inner weight gradient, but continuation
    and cumulant remain differentiable in the question parameters.
    The outer target is a fixed downstream prediction label, not a return
    gradient estimator and not the new question's own training error.
    """
    cumulant = eta[0]
    # Stable logistic, including large finite eta values.
    gamma = 1 / (1 + math.exp(-eta[1])) if eta[1] >= 0 else math.exp(eta[1]) / (1 + math.exp(eta[1]))
    tail = weight * next_x
    delta = cumulant + gamma * tail - weight * x
    updated = weight + alpha * delta * x
    error = updated * test_x - target
    sensitivity = [alpha * x, alpha * x * tail * gamma * (1-gamma)]
    gradient = [error * test_x * value for value in sensitivity]
    return dict(cumulant=cumulant, continuation=gamma, old_weight=weight,
                delta=delta, new_weight=updated, loss=.5*error*error,
                sensitivity=sensitivity, meta_gradient=gradient)


def path_coordinate(n=4):
    """Analytic first nonconstant random-walk mode of an undirected path.

    u = sqrt(D) e has Euclidean norm one; e is the state coordinate.
    No sampled graph discovery is performed here. The path is supplied.
    """
    if n < 3:
        raise ValueError('Use at least three states')
    degree = [1] + [2]*(n-2) + [1]
    raw = [math.cos(math.pi*s/(n-1)) for s in range(n)]
    norm = math.sqrt(sum(d*x*x for d,x in zip(degree,raw)))
    coordinate = [x/norm for x in raw]
    return coordinate, degree, 1-math.cos(math.pi/(n-1))


def skill(coordinate, direction=1, gamma=.9):
    """Solve internal control with a zero-value stop action.

    Synchronous backups read the old value vector. At each state the first
    action is stop, so a tie at zero stops. Primitive left/right actions
    reflect at the boundary. Original task rewards are not consulted.
    """
    if direction not in (-1, 1) or not 0 <= gamma < 1:
        raise ValueError('Require a signed direction and discounted internal task')
    e = [direction*x for x in coordinate]
    n = len(e)
    values = [0.]*n
    def candidates(s, old):
        return [('stop', 0.)] + [(a, e[sp]-e[s]+gamma*old[sp])
                               for a,sp in [('left',max(0,s-1)),('right',min(n-1,s+1))]]
    for _ in range(1000):
        new = [max(v for _,v in candidates(s,values)) for s in range(n)]
        gap = max(abs(a-b) for a,b in zip(new,values))
        values = new
        if gap < 1e-13:
            break
    else:
        raise RuntimeError('Bellman iteration did not meet tolerance')
    policy = [max(candidates(s,values), key=lambda pair: pair[1])[0] for s in range(n)]
    rewards = [dict(state=s, left=e[max(0,s-1)]-e[s],
                    right=e[min(n-1,s+1)]-e[s], stop=0.) for s in range(n)]
    return dict(direction=direction, intrinsic_rewards=rewards, value=values, policy=policy,
                initiation=[s for s,a in enumerate(policy) if a != 'stop'],
                termination=[s for s,a in enumerate(policy) if a == 'stop'])


def execute(option, start):
    s, path = start, [start]
    for _ in range(100):
        a = option['policy'][s]
        if a == 'stop':
            return path
        s += -1 if a == 'left' else 1
        path.append(s)
    raise RuntimeError('A discovered policy failed to stop')


def run():
    e, degree, eigenvalue = path_coordinate()
    # Machado et al. (2017) use the normalized-Laplacian vector directly.
    # e and u have different differences on a non-regular graph.
    u = [math.sqrt(d)*x for d,x in zip(degree,e)]
    options = [skill(u,sign) for sign in [1,-1]]
    for option in options:
        option['path_from_state_1'] = execute(option,1)
    return dict(question=question_update([.2,0.]), graph='given four-state path',
                random_walk_coordinate=e, eigenpurpose_coordinate=u,
                degree=degree,eigenvalue=eigenvalue,options=options)


class Mechanisms(unittest.TestCase):
    def test_question_gradient_by_independent_finite_difference(self):
        eta = [.2, -.4]
        answer = question_update(eta)
        for j, exact in enumerate(answer['meta_gradient']):
            plus, minus = eta[:], eta[:]
            plus[j] += 1e-6
            minus[j] -= 1e-6
            numerical = (question_update(plus)['loss']-question_update(minus)['loss'])/2e-6
            self.assertAlmostEqual(exact,numerical,places=8)

    def test_no_learning_no_meta_path(self):
        self.assertEqual(question_update([.2,0.],alpha=0.)['meta_gradient'],[0.,0.])

    def test_no_consumer_no_meta_path(self):
        self.assertEqual(question_update([.2,0.],test_x=0.)['meta_gradient'],[0.,0.])

    def test_independent_graph_laplacian(self):
        for n in [4,7]:
            e,d,lam = path_coordinate(n)
            u = [math.sqrt(degree)*x for degree,x in zip(d,e)]
            self.assertAlmostEqual(sum(x*x for x in u),1.)
            self.assertAlmostEqual(sum(degree*x for degree,x in zip(d,e)),0.)
            energy = 0.
            for s in range(n):
                neighbors = [j for j in [s-1,s+1] if 0 <= j < n]
                lu = u[s]-sum(u[j]/math.sqrt(d[s]*d[j]) for j in neighbors)
                self.assertAlmostEqual(lu,lam*u[s])
                for j in neighbors:
                    energy += .5*(e[s]-e[j])**2
            self.assertAlmostEqual(energy,lam)

    def test_values_equal_exhaustive_finite_paths_with_stopping(self):
        coordinate,degree,_ = path_coordinate()
        e = [math.sqrt(d)*x for d,x in zip(degree,coordinate)]
        # Enumerate every sequence, including loops and early stops. This
        # calculation never reads the Bellman value vector under test.
        def enumerate_paths(s,sign,remaining):
            returns = [0.]  # Stop now.
            if remaining:
                for sp in [max(0,s-1),min(3,s+1)]:
                    returns += [sign*(e[sp]-e[s])+.9*g
                                for g in enumerate_paths(sp,sign,remaining-1)]
            return returns
        for sign in [1,-1]:
            option = skill(e,sign)
            for s,v in enumerate(option['value']):
                self.assertAlmostEqual(v,max(enumerate_paths(s,sign,8)))

    def test_discovered_components_and_execution(self):
        coordinate,degree,_ = path_coordinate()
        e = [math.sqrt(d)*x for d,x in zip(degree,coordinate)]
        positive,negative = skill(e,1),skill(e,-1)
        self.assertEqual(positive['termination'],[0])
        self.assertEqual(negative['termination'],[3])
        self.assertEqual(positive['initiation'],[1,2,3])
        self.assertEqual(execute(positive,1),[1,0])
        self.assertEqual(execute(negative,1),[1,2,3])

    def test_exported_rewards_use_original_normalized_laplacian_vector(self):
        result = run()
        e, u = result['random_walk_coordinate'], result['eigenpurpose_coordinate']
        self.assertNotAlmostEqual(e[0]-e[1], u[0]-u[1])
        for option in result['options']:
            self.assertAlmostEqual(option['intrinsic_rewards'][1]['left'],
                                   option['direction']*(u[0]-u[1]))

    def test_zero_direction_stops_everywhere(self):
        self.assertEqual(skill([0.,0.,0.,0.])['termination'],[0,1,2,3])


if __name__ == '__main__':
    if '--test' in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        print(json.dumps(run(),ensure_ascii=False,indent=2))
