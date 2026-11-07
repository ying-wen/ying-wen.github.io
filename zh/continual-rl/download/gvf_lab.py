"""GVF prediction laboratory: TD(0), TD(lambda), GTD2, TDC, GTD(lambda), ETD(lambda).

Original educational implementation, MIT. Python 3.10+, standard library only.
No paper-scale benchmark claim. Run: python3 gvf_lab.py compare --steps 40000
Each GVF has its own question, parameters, traces and continuation state.
"""
import argparse
import json
import math
import random
import unittest


def dot(x, y):
    return sum(a*b for a,b in zip(x,y))


# BEGIN question
def question(name, state, action, next_state):
    """Return cumulant, gamma_next, target probability of the OBSERVED action.

    'arrival' predicts a discounted next-arrival signal, not an undiscounted
    eventual-arrival probability. 'slow-arrival' asks about a different policy.
    Arrival is a transition into cell 2, including staying there for one step.
    """
    forward = .2 if name == 'slow-arrival' else .8
    pi_observed = forward if action == 1 else 1-forward
    arrived = next_state == 2
    cumulant = 1. if name == 'energy' else float(arrived)
    gamma_next = 0. if arrived else .9
    return cumulant, gamma_next, pi_observed
# END question


# BEGIN learner
class LinearGVF:
    """Fixed-feature prediction. Right-hand sides always use OLD w and h.

    gamma_current belongs to the transition entering x; gamma_next to x -> xp.
    GTD(lambda) here is the TDC-style form used in RLPark, not GTD2(lambda).
    ETD uses a follow-on trace and interest, not a second learned value vector.
    """
    methods = ('td0','tdlambda','gtd2','tdc','gtdlambda','etd')

    def __init__(self, dimension, method='td0', alpha=.01, beta=.05, lam=.6):
        if method not in self.methods or dimension < 1:
            raise ValueError('Invalid method or dimension')
        if not all(math.isfinite(v) for v in (alpha,beta,lam)) or alpha<=0 or beta<=0 or not 0<=lam<=1:
            raise ValueError('Positive finite step sizes and lambda in [0,1] required')
        self.method,self.alpha,self.beta = method,alpha,beta
        self.lam = 0. if method in ('td0','gtd2','tdc') else lam
        self.w,self.h,self.e = [[0.]*dimension for _ in range(3)]
        self.gamma_current,self.rho_previous,self.followon = 0.,0.,0.

    def step(self, x, xp, cumulant, gamma_next, rho, interest=1.):
        if len(x)!=len(self.w) or len(xp)!=len(x):
            raise ValueError('Feature dimension mismatch')
        if not all(math.isfinite(v) for v in [*x,*xp,cumulant,gamma_next,rho,interest]):
            raise ValueError('Nonfinite input')
        if not 0<=gamma_next<=1 or rho<0 or interest<0:
            raise ValueError('Invalid discount, ratio or interest')
        old_w,old_h = self.w[:],self.h[:]
        delta = cumulant + gamma_next*dot(old_w,xp) - dot(old_w,x)
        if self.method == 'etd':
            self.followon = interest + self.gamma_current*self.rho_previous*self.followon
            emphasis = self.lam*interest + (1-self.lam)*self.followon
        else:
            emphasis = 1.
        self.e = [rho*(self.gamma_current*self.lam*ei + emphasis*xi)
                  for ei,xi in zip(self.e,x)]
        if self.method == 'gtd2':
            xh = dot(x,old_h)
            dw = [rho*(xi-gamma_next*xpi)*xh for xi,xpi in zip(x,xp)]
        elif self.method in ('tdc','gtdlambda'):
            eh = dot(self.e,old_h)
            dw = [delta*ei-gamma_next*(1-self.lam)*xpi*eh for ei,xpi in zip(self.e,xp)]
        else:
            dw = [delta*ei for ei in self.e]
        self.w = [wi+self.alpha*di for wi,di in zip(old_w,dw)]
        if self.method in ('gtd2','tdc','gtdlambda'):
            xh = dot(x,old_h)
            self.h = [hi+self.beta*(delta*ei-xh*xi) for hi,ei,xi in zip(old_h,self.e,x)]
        self.gamma_current,self.rho_previous = gamma_next,rho
        return delta

    def external_reset(self):
        """Use only when the data protocol actually breaks the trajectory.

        A GVF gamma_next=0 already terminates its own trace on the next step;
        it does not require resetting the physical environment or weights.
        """
        self.e = [0.]*len(self.w)
        self.gamma_current,self.rho_previous,self.followon = 0.,0.,0.
# END learner


def solve(matrix, rhs):
    """Tiny pivoted Gaussian elimination for an independent Bellman reference."""
    n=len(rhs); a=[list(row)+[y] for row,y in zip(matrix,rhs)]
    for j in range(n):
        pivot=max(range(j,n),key=lambda i:abs(a[i][j]))
        a[j],a[pivot]=a[pivot],a[j]
        if abs(a[j][j])<1e-12: raise ValueError('Singular reference system')
        scale=a[j][j]; a[j]=[v/scale for v in a[j]]
        for i in range(n):
            if i!=j:
                scale=a[i][j]; a[i]=[v-scale*u for v,u in zip(a[i],a[j])]
    return [row[-1] for row in a]


def exact_values(name):
    # v = r + P_gamma v; transition rows include the arrival-dependent gamma.
    matrix=[[float(i==j) for j in range(3)] for i in range(3)]; rhs=[0.]*3
    for s in range(3):
        for action in (0,1):
            sp=(s+action)%3
            c,g,p=question(name,s,action,sp)
            rhs[s]+=p*c; matrix[s][sp]-=p*g
    return solve(matrix,rhs)


# BEGIN horde
def run(method='gtdlambda', steps=40000, seed=7, alpha=.01, beta=.05, lam=.6, use_is=True):
    """A complete multi-question loop, sharing data but NOT traces or weights."""
    rng=random.Random(seed)
    names=('energy','arrival','slow-arrival')
    heads={name:LinearGVF(3,method,alpha,beta,lam) for name in names}
    features=[[float(i==j) for j in range(3)] for i in range(3)]
    state=0
    for _ in range(steps):
        # The real behavior is sampled exactly ONCE for all questions.
        action=int(rng.random()<.5); behavior_probability=.5
        next_state=(state+action)%3
        for name,head in heads.items():
            c,g,pi_observed=question(name,state,action,next_state)
            rho=pi_observed/behavior_probability if use_is else 1.
            head.step(features[state],features[next_state],c,g,rho)
        state=next_state  # no env.reset() when a question ends
    result={}
    for name,head in heads.items():
        truth=exact_values(name)
        result[name]={'prediction':head.w,'reference':truth,
                      'rmse':math.sqrt(sum((w-v)**2 for w,v in zip(head.w,truth))/3)}
    return result
# END horde


# BEGIN counterexample
def counterexample(iterations=300, alpha=.05, beta=.1):
    """Expected updates in an explicitly specified, stationary two-state MDP.

    x(0)=1, x(1)=2. Action a sets the next state to a. b(1)=.1, pi(1)=1.
    Thus d_b=(.9,.1), gamma=.9, all cumulants=0, and true w=0.
    Exact expectation removes sampling noise; this is not an empirical rank.
    """
    A=.9*1*(1-.9*2)+.1*2*(2-.9*2)  # -.68
    C=.9*1**2+.1*2**2  # 1.3
    td,gtd,h=1.,1.,0.
    for _ in range(iterations):
        td+=alpha*(-A*td)
        old_gtd,old_h=gtd,h
        gtd=old_gtd+alpha*A*old_h
        h=old_h+beta*(-A*old_gtd-C*old_h)
    return {'A':A,'C':C,'true_weight':0.,'td_weight':td,'gtd2_weight':gtd,
            'scope':'exact expected updates; not a stochastic convergence proof'}
# END counterexample


class Tests(unittest.TestCase):
    def test_bellman_reference(self):
        for name in ('energy','arrival','slow-arrival'):
            v=exact_values(name)
            for s in range(3):
                backup=0.
                for a in (0,1):
                    sp=(s+a)%3;c,g,p=question(name,s,a,sp);backup+=p*(c+g*v[sp])
                self.assertAlmostEqual(v[s],backup)

    def test_question_semantics(self):
        self.assertEqual(question('energy',1,1,2),(1.,0.,.8))
        self.assertEqual(question('arrival',0,0,0),(0.,.9,1-.8))
        self.assertNotEqual(exact_values('arrival'),exact_values('slow-arrival'))

    def test_is_conditional_expectation(self):
        for s in range(3):
            expected=sum(question('arrival',s,a,(s+a)%3)[2]*(a+3) for a in (0,1))
            estimated=sum(.5*(question('arrival',s,a,(s+a)%3)[2]/.5)*(a+3) for a in (0,1))
            self.assertAlmostEqual(expected,estimated)

    def test_td_lambda_zero(self):
        a=LinearGVF(2,'td0');b=LinearGVF(2,'tdlambda',lam=0)
        for _ in range(3):
            for x,xp,c,g,r in [([1,0],[0,1],1,.9,2),([0,1],[1,0],2,0,.5)]:
                a.step(x,xp,c,g,r);b.step(x,xp,c,g,r)
        self.assertEqual(a.w,b.w)

    def test_gtd_lambda_zero_is_tdc(self):
        a=LinearGVF(2,'tdc');b=LinearGVF(2,'gtdlambda',lam=0)
        for _ in range(4):
            a.step([1,0],[0,1],1,.5,1.7);b.step([1,0],[0,1],1,.5,1.7)
        self.assertEqual((a.w,a.h),(b.w,b.h))

    def test_gtd2_uses_old_h(self):
        a=LinearGVF(1,'gtd2',alpha=.1,beta=.2);a.w=[1.];a.h=[2.]
        a.step([1],[1],3,.5,2)
        self.assertAlmostEqual(a.w[0],1.2);self.assertAlmostEqual(a.h[0],2.6)

    def test_gtd_trace_two_discounts(self):
        a=LinearGVF(1,'gtdlambda',alpha=.1,beta=.2,lam=.4)
        a.w=[1.];a.h=[2.];a.e=[3.];a.gamma_current=.5
        a.step([1],[2],3,.9,2)
        e=2*(.5*.4*3+1);delta=3+.9*2-1
        self.assertAlmostEqual(a.e[0],e)
        self.assertAlmostEqual(a.w[0],1+.1*(delta*e-.9*.6*2*(e*2)))
        self.assertAlmostEqual(a.h[0],2+.2*(delta*e-2))

    def test_emphasis_uses_previous_ratio(self):
        a=LinearGVF(1,'etd',lam=.5)
        a.step([1],[1],0,.8,3,interest=2)
        a.step([1],[1],0,.6,4,interest=1)
        self.assertAlmostEqual(a.followon,1+.8*3*2)
        self.assertAlmostEqual(a.e[0],4*(.8*.5*6 + .5+.5*5.8))

    def test_emphatic_lambda_one_is_is_td_lambda(self):
        a=LinearGVF(1,'etd',lam=1);b=LinearGVF(1,'tdlambda',lam=1)
        for rho in (1.5,.5,2):
            a.step([1],[1],1,.8,rho);b.step([1],[1],1,.8,rho)
        self.assertEqual(a.w,b.w)

    def test_question_termination_keeps_terminal_credit(self):
        a=LinearGVF(2,'tdlambda',alpha=.1,lam=.5)
        a.step([1,0],[0,1],0,.8,1)
        a.step([0,1],[1,0],1,0,1)
        self.assertEqual(a.w,[.04000000000000001,.1])
        a.step([1,0],[0,1],0,.8,1)
        self.assertEqual(a.e,[1.,0.])

    def test_external_reset_keeps_knowledge(self):
        a=LinearGVF(1,'etd');a.step([1],[1],1,.9,1)
        before=a.w[:];a.external_reset()
        self.assertEqual(a.w,before);self.assertEqual(a.e,[0.]);self.assertEqual(a.followon,0.)

    def test_stationary_expected_counterexample(self):
        r=counterexample();self.assertLess(r['A'],0)
        self.assertGreater(r['td_weight'],1000);self.assertLess(abs(r['gtd2_weight']),.1)

    def test_tabular_smoke(self):
        for method in LinearGVF.methods:
            for r in run(method,steps=40000).values():
                self.assertLess(r['rmse'],.25,(method,r))

    def test_missing_is_answers_wrong_policy(self):
        without=run('td0',steps=40000,use_is=False)
        self.assertGreater(without['energy']['rmse'],.3)

    def test_dimensions_and_nonfinite(self):
        a=LinearGVF(2)
        with self.assertRaises(ValueError):a.step([1],[1],1,.9,1)
        with self.assertRaises(ValueError):a.step([1,0],[0,1],1,.9,float('nan'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment',choices=['compare','counterexample','test']+list(LinearGVF.methods),nargs='?',default='compare')
    p.add_argument('--steps',type=int,default=40000);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--alpha',type=float,default=.01);p.add_argument('--beta',type=float,default=.05)
    p.add_argument('--lam',type=float,default=.6);p.add_argument('--without-is',action='store_true')
    args=p.parse_args()
    if args.steps<1:p.error('--steps must be positive')
    if args.experiment=='test':unittest.main(argv=['gvf_lab.py'],verbosity=2)
    elif args.experiment=='counterexample':print(json.dumps(counterexample(),indent=2))
    else:
        methods=LinearGVF.methods if args.experiment=='compare' else [args.experiment]
        print(json.dumps({m:run(m,args.steps,args.seed,args.alpha,args.beta,args.lam,not args.without_is) for m in methods},indent=2))
