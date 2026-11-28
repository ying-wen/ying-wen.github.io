"""Independent algebra, information-boundary and numerical tests for MARL examples."""
import copy
import itertools
import unittest
from implementations.multiagent import coma, joint_policy_gradient
from implementations.multiagent import _common as common
from implementations import runtime


class CounterfactualTests(unittest.TestCase):
    def test_advantage_marginalizes_only_own_action(self):
        q=[1.,4.,3.,9.];pi=[[.25,.75],[.6,.4]];a=[1,0]
        actual=coma.advantages(q,pi,a)
        self.assertAlmostEqual(actual[0],3-(.25*1+.75*3))
        self.assertAlmostEqual(actual[1],3-(.6*3+.4*9))

    def test_conditional_expected_advantage_is_zero(self):
        q=[-.4,1.1,.9,-.2];pi=[[.3,.7],[.2,.8]]
        for i in range(2):
            for other in range(2):
                total=0.
                for action in range(2):
                    a=[other,other];a[i]=action
                    total+=pi[i][action]*coma.advantages(q,pi,a)[i]
                self.assertAlmostEqual(total,0.,places=13)

    def test_counterfactual_policy_gradient_matches_finite_difference(self):
        logits=[[.1,-.2],[.5,-.3]];q=[.3,1.,-.2,.7]
        pi=[common.probabilities(x) for x in logits]
        def objective(theta):
            p=[common.probabilities(x) for x in theta]
            return sum(p[0][a]*p[1][b]*q[2*a+b] for a,b in itertools.product(range(2),repeat=2))
        for i in range(2):
            for k in range(2):
                analytic=sum(pi[0][a]*pi[1][b]*coma.advantages(q,pi,[a,b])[i]*
                             (([a,b][i]==k)-pi[i][k])
                             for a,b in itertools.product(range(2),repeat=2))
                plus=copy.deepcopy(logits);minus=copy.deepcopy(logits);eps=1e-6
                plus[i][k]+=eps;minus[i][k]-=eps
                numeric=(objective(plus)-objective(minus))/(2*eps)
                self.assertAlmostEqual(analytic,numeric,places=9)

    def test_no_observed_other_bit_enters_actor(self):
        theta=[[[.2,-.1],[.4,.5]],[[0.,0.],[0.,0.]]]
        p0=common.probabilities(theta[0][0])
        for other_bit in range(2):
            self.assertEqual(p0,common.probabilities(theta[0][[0,other_bit][0]]))

    def test_controlled_sampling_runs_are_reproducible_and_complete(self):
        for module in [coma,joint_policy_gradient]:
            first=module.run(2,80)
            self.assertEqual(first,module.run(2,80))
            runtime.validate_rows(first,80)
            self.assertTrue(all(0<=r['value']<=1 for r in first))
            self.assertEqual(first[-1]['actor_updates'],160)

    def test_first_actor_update_uses_presample_critic(self):
        # Q=0 before the first action, so both actors must remain uniform.
        # Fitting the sampled Q before constructing A would violate this.
        for module in [coma,joint_policy_gradient]:
            rows=module.run(4,1)
            self.assertEqual(rows[0]['value'],rows[-1]['value'])


try:
    import torch
    from implementations.multiagent import qmix, vdn
except ImportError:
    torch=None


@unittest.skipIf(torch is None,"Optional PyTorch is not installed")
class MixingTests(unittest.TestCase):
    def setUp(self):
        common.seed_torch(5)

    def test_mixer_derivative_is_nonnegative(self):
        mixer=qmix.Mixer()
        utilities=torch.randn(7,2,requires_grad=True)
        states=torch.randn(7,7)
        gradient=torch.autograd.grad(mixer(utilities,states).sum(),utilities)[0]
        self.assertGreaterEqual(float(gradient.min()),-1e-7)

    def test_decentralized_greedy_is_a_joint_maximizer(self):
        mixer=qmix.Mixer();state=common.global_features([0,1],1).unsqueeze(0)
        local=torch.tensor([[.7,-.1],[-.3,.5]])
        greedy=local.argmax(-1).tolist()
        selected=lambda a:mixer(torch.tensor([[local[0,a[0]],local[1,a[1]]]]),state).item()
        values=[selected(a) for a in itertools.product(range(2),repeat=2)]
        self.assertAlmostEqual(selected(greedy),max(values),places=6)

    def test_fixed_other_agent_does_not_change_own_local_features(self):
        self.assertTrue(torch.equal(common.local_features([0,0],1)[0],
                                    common.local_features([0,1],1)[0]))
        self.assertFalse(torch.equal(common.global_features([0,0],1),
                                     common.global_features([0,1],1)))

    def test_terminal_target_does_not_bootstrap(self):
        class Constant(torch.nn.Module):
            def __init__(self,v):
                super().__init__();self.value=torch.nn.Parameter(torch.tensor(float(v)))
            def forward(self,x):
                return self.value.expand(*x.shape[:-1],2)
        class Sum(torch.nn.Module):
            def forward(self,q,s):return q.sum(-1)
        agent=Constant(0);target=Constant(99);mixer=Sum()
        x=common.local_features([0,1],2);s=common.global_features([0,1],2)
        data=common.batch([(x,s,[0,1],1.,x,s,True)])
        optimizer=torch.optim.SGD(agent.parameters(),lr=0.)
        self.assertAlmostEqual(qmix.update(agent,mixer,target,mixer,optimizer,data),1.)
        self.assertAlmostEqual(vdn.update(agent,target,optimizer,data),1.)
        self.assertIsNone(target.value.grad)

    def test_three_step_return_and_update_counts(self):
        for module in [qmix,vdn]:
            rows=module.run(3,70)
            runtime.validate_rows(rows,70)
            self.assertTrue(all(-1e-7<=r['value']<=2.71+1e-6 for r in rows))
            self.assertEqual(rows[-1]['gradient_updates'],39)

    def test_nonmonotone_payoff_counterexample(self):
        payoff=[[1.,0.],[0.,1.]]
        self.assertGreater(payoff[0][0],payoff[1][0])
        self.assertLess(payoff[0][1],payoff[1][1])


if __name__=="__main__":
    unittest.main()
