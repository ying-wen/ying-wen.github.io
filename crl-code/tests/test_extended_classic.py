"""独立算法真实运行与数学边界检查；测试成功不等于效能证据。"""
import ast
import importlib
import itertools
import json
import math
from pathlib import Path
import random
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from implementations import runtime

def module(name):
    return importlib.import_module("implementations.extended_classic."+name)


class ExtendedClassicTests(unittest.TestCase):
    def test_literal_registry_baselines_and_seed_population(self):
        paths={i:p for i,p in runtime.discover().items() if p.parent.name=="extended_classic"}
        self.assertEqual(len(paths),32)
        registry={x["id"] for x in json.loads((ROOT/"integrations/algorithm_coverage.json").read_text())["entries"]}
        refs=set()
        for key,path in paths.items():
            with self.subTest(algorithm=key):
                meta=runtime.metadata(path)
                self.assertEqual(meta,module(key).META)
                self.assertEqual(meta["family"],"extended_classic")
                self.assertIn(meta["implementation_kind"],("teaching_method","component_experiment"))
                self.assertTrue(set(meta["coverage_refs"])<=registry)
                refs.update(meta["coverage_refs"])
                baseline=runtime.metadata(paths[meta["baseline"]])
                runtime.comparable([meta,baseline])
                for seed in (0,3,8):
                    emitted=[]
                    rows=module(key).run(seed=seed,steps=1200,emit=emitted.append)
                    self.assertEqual(rows,emitted)
                    self.assertLessEqual(len(rows),100)
                    runtime.validate_rows(rows,1200)
                self.assertEqual(module(key).run(seed=0,steps=17),module(key).run(seed=0,steps=17))
                with self.assertRaises(ValueError):
                    module(key).run(steps=0)
        self.assertEqual(len(refs),23)

    def test_policy_evaluation_bellman_boundary(self):
        m=module("iterative_policy_evaluation")
        self.assertAlmostEqual(m.update([0.]*5)[4],.798)
        self.assertAlmostEqual(m.update([0.]*5)[3],-.01)
        self.assertLess(m.run(steps=500)[-1]["value"],1e-9)

    def test_mc_visit_definitions(self):
        for name,expected_count,expected_value in (
            ("every_visit_mc",2,1.5),("first_visit_mc_reference",1,2.)):
            v,counts={1:0.,2:0.},{1:0,2:0}
            module(name).update(v,counts,[1,2,1],[1.,0.,1.])
            self.assertEqual(counts[1],expected_count)
            self.assertEqual(v[1],expected_value)

    def test_mc_control_discount_and_first_pair(self):
        q,counts={0:[0.,0.],1:[0.,0.]},{0:[0,0],1:[0,0]}
        module("mc_control").update(q,counts,[(0,1,0.),(1,1,1.)])
        self.assertAlmostEqual(q[0][1],.95)
        self.assertEqual(q[1][1],1.)

    def test_importance_weights_and_zero_mass(self):
        for name in ("ordinary_is","weighted_is","per_decision_is","sequential_dr"):
            m=module(name)
            self.assertEqual(m.importance_ratio(.8,.5),1.6)
            self.assertEqual(m.importance_ratio(0.,.5),0.)
            with self.assertRaises(ValueError):
                m.importance_ratio(.8,0.)
        m=module("ordinary_is")
        contribution,weight=m.sample_return([1.,2.],[2.,.5],gamma=.5)
        self.assertEqual(weight,1.)
        self.assertEqual(contribution,2.)
        state=module("weighted_is").update(0.,0.,0,[1.],[0.])
        self.assertEqual(state[-1],0.)
        self.assertEqual(module("per_decision_is").sample_return([1.,2.],[2.,.5],gamma=.5),3.)

    def test_ope_expectation_and_dr_model_independence(self):
        ordinary,pdis,dr=0.,0.,0.
        gamma=.9
        truth=.74*(1+gamma+gamma**2)
        for actions in itertools.product(range(2),repeat=3):
            rewards=[[.1,.9][a] for a in actions]
            ratios=[[.2,.8][a]/.5 for a in actions]
            ordinary+=.5**3*module("ordinary_is").sample_return(rewards,ratios)[0]
            pdis+=.5**3*module("per_decision_is").sample_return(rewards,ratios)
            # 故意有偏、但Vhat与Qhat一致的冻结模型，准确比率仍保证期望。
            vhat=[.3*(1+gamma+gamma**2),.3*(1+gamma),.3]
            qhat=vhat[:]
            dr+=.5**3*module("sequential_dr").sample_return(rewards,ratios,qhat,vhat)
        self.assertAlmostEqual(ordinary,truth)
        self.assertAlmostEqual(pdis,truth)
        self.assertAlmostEqual(dr,truth)

    def test_dr_zero_model_is_pdis(self):
        r,rho=[1.,2.,3.],[.4,1.6,.4]
        dr=module("sequential_dr").sample_return(r,rho,[0.]*3,[0.]*3)
        self.assertAlmostEqual(dr,module("per_decision_is").sample_return(r,rho))

    def test_qsigma_tree_backup_and_sarsa_endpoints(self):
        q={0:[.2,.4],1:[.5,.7],2:[.9,1.]}
        trajectory=[(0,1,.1,1),(1,0,.2,2)]
        m=module("q_sigma")
        tree=module("tree_backup").target(q,trajectory,2,1)
        self.assertAlmostEqual(m.target(q,trajectory,2,1,sigma=0.),tree)
        sarsa=.1+.95*.2+.95**2*q[2][1]
        self.assertAlmostEqual(m.target(q,trajectory,2,1,sigma=1.),sarsa)
        terminal=[(0,1,.1,1),(1,0,1.,None)]
        self.assertAlmostEqual(m.target(q,terminal,None,0,sigma=1.),.1+.95)
        with self.assertRaises(ValueError):
            m.target(q,trajectory,2,1,sigma=2.)

    def test_retrace_lambda_zero_expected_td(self):
        q={0:[.2,.4],1:[.5,.7],2:[.9,1.]}
        trajectory=[(0,1,.1,1),(1,0,.2,2)]
        expected=.1+.95*(.2*.5+.8*.7)
        self.assertAlmostEqual(module("retrace").target(q,trajectory,2,1,lam=0.),expected)

    def test_gradient_mc_true_derivative(self):
        w=module("gradient_mc").update([.1,.2],[1.,.5],1.,alpha=.1)
        self.assertAlmostEqual(w[0],.18)
        self.assertAlmostEqual(w[1],.24)

    def test_lstd_accumulation_and_solution(self):
        m=module("lstd")
        a,b=[[0.,0.],[0.,0.]],[0.,0.]
        m.update(a,b,[1.,0.],0.,[0.,1.])
        m.update(a,b,[0.,1.],1.,[0.,0.])
        self.assertEqual(a,[[1.,-1.],[0.,1.]])
        self.assertEqual(b,[0.,1.])
        self.assertEqual(m.estimate(a,b,ridge=0.),[1.,1.])
        with self.assertRaises(ValueError):
            m.estimate([[0.,0.],[0.,0.]],[0.,0.],ridge=0.)

    def test_semigradient_sarsa_true_terminal(self):
        m=module("semi_gradient_sarsa")
        w,delta=m.update([.2,.4],[1.,0.],1.,[0.,0.],alpha=.1)
        self.assertAlmostEqual(delta,.8)
        self.assertAlmostEqual(w[0],.28)
        self.assertEqual(w[1],.4)

    def test_differential_td_and_smdp_old_duration(self):
        v,rate,delta=module("differential_td_prediction").update([1.,2.],.5,0,1.,1,alpha=.1,eta=.2)
        self.assertEqual(delta,1.5)
        self.assertAlmostEqual(rate,.53)
        v,rate,length,delta=module("average_smdp_option").update(
            [1.,2.],.5,2.,0,3.,3,1,alpha=.1,eta=.2,length_step=.1)
        self.assertEqual(delta,3.)
        self.assertAlmostEqual(v[0],1.15)
        self.assertAlmostEqual(rate,.53)
        self.assertAlmostEqual(length,2.1)
        self.assertEqual(module("average_smdp_option").run(steps=1)[-1]["step"],1)

    def test_rvi_reference_shift_invariance(self):
        m=module("relative_value_iteration")
        a,_=m.update([.2,.8])
        b,_=m.update([4.2,4.8])
        for x,y in zip(a,b):
            self.assertAlmostEqual(x,y)
        self.assertEqual(a[0],0.)
        self.assertLess(m.run(steps=120)[-1]["value"],1e-12)

    def test_watkins_cuts_after_current_error_and_keeps_ties(self):
        m=module("watkins_q_lambda")
        q={0:[0.,0.],1:[1.,2.]}
        e={0:[1.,0.],1:[0.,0.]}
        m.update(q,e,0,1,1.,1,0,alpha=.1)
        self.assertGreater(q[0][0],0.)
        self.assertEqual(e,{0:[0.,0.],1:[0.,0.]})
        q={0:[0.,0.],1:[2.,2.]}
        e={0:[0.,0.],1:[0.,0.]}
        m.update(q,e,0,1,1.,1,0,alpha=.1)
        self.assertAlmostEqual(e[0][1],.95*.8)

    def test_local_lambda_minimum_and_moments(self):
        m=module("greedy_adaptive_lambda")
        self.assertEqual(m.choose_lambda(4.,1.),.8)
        self.assertEqual(m.choose_lambda(0.,0.),0.)
        loss=lambda x:(1-x)**2*4+x*x
        self.assertLess(loss(.8),loss(.7))
        count,mean,m2=0,0.,0.
        for x in (1.,2.,3.):
            count,mean,m2=m.update_moments(count,mean,m2,x)
        self.assertEqual((count,mean,m2),(3,2.,2.))

    def test_expected_trace_predictor_and_conditional_credit(self):
        m=module("expected_eligibility_traces")
        mean,count=[0.,0.,0.],0
        for z in ([.72,0.,1.],[0.,.72,1.]):
            mean,count=m.update_trace(mean,count,z)
        self.assertEqual(mean,[.36,.36,1.])
        self.assertEqual(m.credit(2.,mean),[.72,.72,2.])

    def test_gradient_trace_forward_backward_and_finite_difference(self):
        back=module("gradient_eligibility_traces")
        forward=module("gradient_trace_forward_reference")
        w,h=[.1,-.2],[.3,.1]
        xs=[[1.,.2],[.1,1.],[.2,.3]]
        rewards=[.2,.8]
        dw,dh=back.directions(w,h,xs,rewards)
        fw,fh=forward.directions(w,h,xs,rewards)
        for a,b in zip(dw+dh,fw+fh):
            self.assertAlmostEqual(a,b,places=12)
        def objective(w,h):
            values=[math.tanh(sum(v*x for v,x in zip(w,row))) for row in xs]
            targets=[0.,0.]
            tail=values[-1]
            for t in (1,0):
                tail=rewards[t]+.9*(.2*values[t+1]+.8*tail)
                targets[t]=tail
            total=0.
            for t in (0,1):
                auxiliary=math.tanh(sum(v*x for v,x in zip(h,xs[t])))
                total+=auxiliary*(targets[t]-values[t])-.5*auxiliary**2
            return total
        for parameters,direction,sign in ((w,dw,-1.),(h,dh,1.)):
            for j in range(2):
                plus,minus=parameters[:],parameters[:]
                plus[j]+=1e-6
                minus[j]-=1e-6
                if parameters is w:
                    numerical=(objective(plus,h)-objective(minus,h))/2e-6
                else:
                    numerical=(objective(w,plus)-objective(w,minus))/2e-6
                self.assertAlmostEqual(direction[j],sign*numerical,places=8)

    def test_thompson_posterior_observations(self):
        successes,failures=[0,0],[0,0]
        m=module("thompson_sampling")
        m.update(successes,failures,1,1.)
        m.update(successes,failures,1,0.)
        self.assertEqual((successes,failures),([0,1],[0,1]))
        self.assertIn(m.select(successes,failures,random.Random(0)),(0,1))

    def test_primal_dual_uses_old_multiplier_and_projects(self):
        m=module("cmdp_primal_dual")
        logit,multiplier=m.update(0.,2.,1,1.,1.,1)
        self.assertAlmostEqual(logit,-.075)
        self.assertAlmostEqual(multiplier,2.09)
        _,multiplier=m.update(0.,0.,0,0.,0.,1)
        self.assertEqual(multiplier,0.)

    def test_minimax_matching_pennies_pure_and_degenerate(self):
        m=module("minimax_2x2")
        row,value=m.minimax([[1.,-1.],[-1.,1.]])
        self.assertEqual(row,[.5,.5])
        self.assertEqual(value,0.)
        row,value=m.minimax([[2.,2.],[1.,1.]])
        self.assertEqual(row,[1.,0.])
        self.assertEqual(value,2.)
        row,value=m.minimax([[0.,0.],[0.,0.]])
        self.assertEqual(value,0.)
        self.assertAlmostEqual(sum(row),1.)


if __name__=="__main__":
    unittest.main()
