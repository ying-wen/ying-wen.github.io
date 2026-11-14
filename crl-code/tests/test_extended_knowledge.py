"""扩展知识：梯度、真实终止、时间抽象及完整多seed人口检查。"""
import copy
import math
import random
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from implementations.runtime import discover, load, comparable, validate_rows
from implementations.extended_knowledge._common import MLP, softmax, sigmoid
from implementations.extended_knowledge import (
    rnd_exploration as rnd, learning_progress as curriculum,
    recovery_filter as recovery, uvfa_shared as uvfa, her_replay as her,
    subtask_stopping as stopping, intra_option_q as intra,
    option_critic as critic, diayn_tabular as diayn, option_model as model,
    option_value_iteration as planning, preference_reward as preference,
    maxent_irl as irl, intrinsic_meta_gradient as meta, reward_centering as centering,
)

def derivative(fn,x,epsilon=1e-6):
    return (fn(x+epsilon)-fn(x-epsilon))/(2*epsilon)

class GradientChecks(unittest.TestCase):
    def test_mlp_all_parameter_groups_finite_difference(self):
        net=MLP(2,3,2,random.Random(4)); x=[.2,-.4]; direction=[.7,-.3]
        grads=net.gradients(x,direction)
        def objective():
            return sum(a*b for a,b in zip(net.forward(x),direction))
        for attr,gradient in zip(("w","b","v","c"),grads):
            array=getattr(net,attr)
            if attr in ("w","v"):
                for i,row in enumerate(array):
                    for j,old in enumerate(row[:]):
                        row[j]=old+1e-6; hi=objective()
                        row[j]=old-1e-6; lo=objective(); row[j]=old
                        self.assertAlmostEqual((hi-lo)/2e-6,gradient[i][j],places=7)
            else:
                for i,old in enumerate(array[:]):
                    array[i]=old+1e-6; hi=objective()
                    array[i]=old-1e-6; lo=objective(); array[i]=old
                    self.assertAlmostEqual((hi-lo)/2e-6,gradient[i],places=7)

    def test_rnd_fixed_target_and_fit(self):
        rng=random.Random(3); target=MLP(12,12,3,rng); predictor=MLP(12,12,3,rng)
        frozen=copy.deepcopy(target.__dict__)
        x=rnd.observation(5); initial=rnd.predictor_step(predictor,target,x)
        for _ in range(100):
            final=rnd.predictor_step(predictor,target,x)
        self.assertLess(final,initial)
        self.assertEqual(target.__dict__,frozen)

    def test_preference_gradient(self):
        w=[.2,-.3]; d=[2.,-1.]; y=.7
        _,gradient=preference.loss_gradient(w,d,y)
        for j in range(2):
            def fn(x):
                z=w[:]; z[j]=x
                return preference.loss_gradient(z,d,y)[0]
            self.assertAlmostEqual(gradient[j],derivative(fn,w[j]),places=7)

    def test_maxent_gradient_and_normalization(self):
        features=[[0.,0.],[1.,0.],[1.,2.],[2.,1.]]
        w=[.3,-.2]; empirical=[1.,.75]
        _,gradient,p=irl.objective_gradient(w,features,empirical)
        self.assertAlmostEqual(sum(p),1.)
        for j in range(2):
            def fn(x):
                z=w[:]; z[j]=x
                return irl.objective_gradient(z,features,empirical)[0]
            self.assertAlmostEqual(gradient[j],derivative(fn,w[j]),places=7)

    def test_intrinsic_reward_unrolled_gradient(self):
        for theta in [-2.,0.,1.]:
            for eta in [-1.,.3,2.]:
                fn=lambda e:sigmoid(meta.inner(theta,e))
                self.assertAlmostEqual(meta.meta_gradient(theta,eta),derivative(fn,eta),places=8)

    def test_option_critic_sampled_actor_gradient(self):
        logits=[.2,-.4]; advantage=.7; chosen=1
        grad=critic.actor_gradient(logits,chosen,advantage)
        for j in range(2):
            def fn(x):
                z=logits[:]; z[j]=x
                return advantage*math.log(softmax(z)[chosen])
            self.assertAlmostEqual(grad[j],derivative(fn,logits[j]),places=7)

    def test_option_critic_termination_gradient(self):
        h=.3; q=2.; v=4.
        fn=lambda x:(1-sigmoid(x))*q+sigmoid(x)*v
        self.assertAlmostEqual(critic.termination_gradient(h,q,v),derivative(fn,h),places=7)
        self.assertGreater(critic.update_termination(h,q,v),h)
        self.assertLess(critic.update_termination(h,v,q),h)

    def test_diayn_discriminator_gradient(self):
        logits=[.2,-.3]; grad=diayn.discriminator_gradient(logits,1)
        for j in range(2):
            def fn(x):
                z=logits[:]; z[j]=x
                return -math.log(softmax(z)[1])
            self.assertAlmostEqual(grad[j],derivative(fn,logits[j]),places=7)

    def test_diayn_exact_soft_actor_gradient(self):
        logits=[.2,-.3]; q=[1.,-.5]; grad=diayn.soft_actor_gradient(logits,q)
        for j in range(2):
            def fn(x):
                z=logits[:]; z[j]=x; p=softmax(z)
                return sum(a*(b-.1*math.log(a)) for a,b in zip(p,q))
            self.assertAlmostEqual(grad[j],derivative(fn,logits[j]),places=7)

class SemanticsChecks(unittest.TestCase):
    def test_uvfa_terminal_vs_continuing(self):
        net=MLP(4,3,2,random.Random(2))
        net.v=[[0.]*3,[0.]*3]; net.c=[2.,3.]
        self.assertEqual(uvfa.update(net,0,1,1,1,alpha=0.),0.)
        self.assertEqual(uvfa.update(net,0,1,1,6,physical_terminal=True,alpha=0.),-1.)
        self.assertAlmostEqual(uvfa.update(net,0,1,1,6,alpha=0.),-1+.95*3.)

    def test_her_relabel_preserves_facts_and_physical_terminal(self):
        factual=(2,1,3,True)
        tr=her.relabel(factual,6)
        self.assertEqual(tr[:3],factual[:3])
        self.assertTrue(tr[-1])
        self.assertEqual(tr[-2],-1.)
        self.assertTrue(her.relabel((2,1,3,False),3)[-1])
        self.assertFalse(her.relabel((2,1,3,False),6)[-1])

    def test_her_future_goal_is_recorded_future(self):
        trajectory=[(0,1,1,False),(1,1,2,False),(2,0,1,False)]
        replay=her.future_replay(trajectory,random.Random(7))
        for i,tr in enumerate(replay):
            self.assertIn(tr[3],[x[2] for x in trajectory[i:]])
            self.assertEqual(tr[:3],trajectory[i][:3])
            self.assertNotEqual(tr[0],tr[3])

    def test_her_success_source_is_not_a_legal_training_sample(self):
        trajectory=[(3,0,2,False),(2,1,3,False),(3,1,4,False)]
        self.assertIsNone(her.relabel(trajectory[0],3))
        for seed in range(20):
            replay=her.future_replay(trajectory,random.Random(seed))
            self.assertEqual(len(replay),3)
            for i,tr in enumerate(replay):
                self.assertNotEqual(tr[0],tr[3])
                self.assertEqual(tr[:3],trajectory[i][:3])
                self.assertIn(tr[3],[x[2] for x in trajectory[i:]])

    def test_her_no_legal_future_goal_skips_without_fabricating(self):
        self.assertEqual(her.future_replay([(3,0,3,False)],random.Random(0)),[])
        self.assertIsNone(her.relabel((3,0,2,True),3))
        # 对保留的合法重标样本，物理终止仍保留，不能由新目标覆盖。
        valid=her.relabel((3,0,2,True),4)
        self.assertTrue(valid[-1])
        self.assertEqual(valid[:3],(3,0,2))

    def test_her_bootstrap_and_success(self):
        q={(s,g):[10.,20.] for s in range(3) for g in range(3)}
        terminal=her.relabel((0,1,1,False),1)
        self.assertEqual(her.update(q,terminal,alpha=0.),0.)
        truncated=her.relabel((0,1,1,False),2)
        self.assertAlmostEqual(her.update(q,truncated,alpha=0.),18.)

    def test_subtask_stopping_bonus_time(self):
        self.assertEqual(stopping.stopping_target(1.,4.,99.,1.),5.)
        self.assertEqual(stopping.stopping_target(1.,4.,10.,0.),10.)
        self.assertEqual(stopping.stopping_target(1.,4.,99.,1.,terminal=True),1.)

    def test_subtask_evaluation_keeps_learned_stopping_rule(self):
        q=[[0.,1.] for _ in range(7)]
        q[3]=[0.,.5]
        # 相同评价bonus不能让no-bonus基线凭空获得候选的停止策略。
        self.assertAlmostEqual(stopping.score(q,.8),-.04-.9*.04+.81*(-.04+.8))
        self.assertAlmostEqual(stopping.score(q,0.),sum(.9**k*(-.04) for k in range(5))+.9**5)

    def test_intra_option_initiation_vs_continuation(self):
        self.assertEqual(intra.arrival([10.,2.],[0.,1.],mask=[False,True]),[10.,2.])
        self.assertEqual(intra.arrival([10.,2.],[1.,1.],mask=[False,True]),[2.,2.])
        self.assertEqual(intra.arrival([10.,2.],[0.,0.],terminal=True,mask=[False,False]),[0.,0.])

    def test_intra_option_old_snapshot_self_loop(self):
        q=[[1.,2.]]
        intra.update(q,0,0,0,.5,.5,alpha=.1)
        # both targets use old [1,2], not the first updated value.
        expected0=1+.1*(.8/.5)*(.5+.9*(.75*1+.25*2)-1)
        expected1=2+.1*(.2/.5)*(.5+.9*2-2)
        self.assertAlmostEqual(q[0][0],expected0)
        self.assertAlmostEqual(q[0][1],expected1)

    def test_intra_option_terminal_and_support(self):
        q=[[0.,0.],[100.,100.]]
        intra.update(q,0,1,1,2.,.5,terminal=True,alpha=.1)
        self.assertAlmostEqual(q[0][0],.08)
        self.assertAlmostEqual(q[0][1],.32)
        with self.assertRaises(ValueError):
            intra.update(q,0,1,1,2.,0.)

    def test_option_critic_terminal(self):
        self.assertEqual(critic.critic_target(2.,100.,200.,.3,terminal=True),2.)
        self.assertEqual(critic.update_termination(.4,100.,200.,terminal=True),.4)
        self.assertAlmostEqual(critic.critic_target(2.,10.,20.,.25),13.25)

    def test_option_model_discounted_endpoint_not_probability(self):
        r,p=model.targets(1.,10.,[.2,.3,.4],2,1.,gamma=.9)
        self.assertEqual(r,1.)
        self.assertEqual(p,[0.,0.,.9])
        # two primitive steps: R=r0+γr1; endpoint mass γ².
        r2,p2=model.targets(2.,r,p,1,0.,gamma=.9)
        self.assertAlmostEqual(r2,2.9)
        self.assertAlmostEqual(sum(p2),.81)
        self.assertAlmostEqual(planning.backup(r2,p2,[0.,0.,10.]),11.)

    def test_option_model_terminal_forces_stop(self):
        r,p=model.targets(2.,100.,[100.,100.],1,0.,terminal=True)
        self.assertEqual(r,2.)
        self.assertEqual(p,[0.,.9])

    def test_option_value_iteration_time_discount(self):
        self.assertAlmostEqual(planning.backup(1.,[0.,.81],[0.,10.]),9.1)

    def test_diayn_intrinsic_and_independence(self):
        self.assertAlmostEqual(diayn.intrinsic_reward([0.,0.],0),0.)
        logits=[[[[0.,0.] for z in range(2)] for s in range(5)] for h in range(8)]
        self.assertAlmostEqual(diayn.frozen_mi(logits),0.)
        for h in range(8):
            for s in range(5):
                logits[h][s][0]=[100.,-100.]
                logits[h][s][1]=[-100.,100.]
        self.assertAlmostEqual(diayn.frozen_mi(logits),math.log(2),places=7)

    def test_progress_uses_two_complete_windows(self):
        self.assertEqual(curriculum.progress([1.]*19),0.)
        self.assertEqual(curriculum.progress([2.]*10+[.5]*10),1.5)
        self.assertEqual(curriculum.progress([.5]*10+[2.]*10),1.5)

    def test_filter_unknown_not_certified_safe(self):
        self.assertEqual(recovery.filtered_action(2,[.95,.8,.1]),0)
        self.assertEqual(recovery.filtered_action(1,[.95,.8,.1]),1)
        self.assertEqual(recovery.filtered_action(2,[.2,.3,.1]),1)

    def test_reward_centering_old_delta_and_invariant(self):
        v=[2.,3.]; c,delta=centering.centered_step(v,0,1,10.,1.,alpha=.1,eta=.05)
        self.assertAlmostEqual(delta,10-1+.99*3-2)
        self.assertAlmostEqual(c-.05*sum(v),1-.05*5)
        uncentered=[2.,3.]
        c2,d2=centering.centered_step(uncentered,0,1,10.,0.,enabled=False)
        self.assertEqual(c2,0.)
        self.assertAlmostEqual(d2,10+.99*3-2)

    def test_reward_centering_true_continuing_values(self):
        rewards=[9.,10.,11.]; v=centering.exact_values(rewards)
        for s in range(3):
            self.assertAlmostEqual(v[s],rewards[s]+.99*v[(s+1)%3],places=9)

class PopulationChecks(unittest.TestCase):
    def test_every_direct_file_cli_imports_without_pythonpath(self):
        env=os.environ.copy()
        env.pop("PYTHONPATH",None)
        with tempfile.TemporaryDirectory() as tmp:
            for key,path in discover().items():
                if path.parent.name!="extended_knowledge":
                    continue
                command=subprocess.run([sys.executable,str(path),"--help"],cwd=tmp,env=env,capture_output=True,text=True)
                self.assertEqual(command.returncode,0,key+": "+command.stderr)
                self.assertIn("--steps",command.stdout)

    def test_direct_cli_trains_all_shared_import_path_types(self):
        # 数值共享、跨算法评价、跨算法模型/feature以及baseline共享训练四类入口。
        ids=["reward_centering","her_replay","option_value_iteration","maxent_irl","option_critic_fixed_beta"]
        paths=discover()
        env=os.environ.copy(); env.pop("PYTHONPATH",None)
        with tempfile.TemporaryDirectory() as tmp:
            for key in ids:
                out=Path(tmp)/key
                command=subprocess.run([sys.executable,str(paths[key]),"--steps","5","--seeds","11","23","--out",str(out)],cwd=tmp,env=env,capture_output=True,text=True)
                self.assertEqual(command.returncode,0,key+": "+command.stdout+command.stderr)
                manifest=json.loads((out/"manifest.json").read_text())
                self.assertEqual(manifest["status"],"complete")
                self.assertEqual(len(manifest["algorithms"]),2)
                self.assertEqual(len(manifest["runs"]),4)
                self.assertTrue(all(r["status"]=="complete" for r in manifest["runs"]))
                self.assertTrue((out/"learning-curves.svg").exists())
                events=(out/key/"seed-11"/"events.jsonl").read_text().splitlines()
                self.assertEqual(json.loads(events[-1])["step"],5)

    def test_all_literal_metadata_comparable_and_reproducible(self):
        ids=[k for k,p in discover().items() if p.parent.name=="extended_knowledge"]
        self.assertEqual(len(ids),30)
        refs=[]
        for key in ids:
            module=load(key); metadata=module.META
            comparable([metadata,load(metadata["baseline"]).META])
            self.assertEqual(metadata["implementation_kind"],"component_experiment")
            refs.extend(metadata["coverage_refs"])
            for seed in [11,23]:
                emitted=[]
                rows=module.run(seed,37,emitted.append)
                validate_rows(rows,37)
                self.assertEqual(rows,emitted)
                self.assertEqual(rows,module.run(seed,37))
                self.assertEqual(rows[-1]["step"],37)
                self.assertTrue(all(math.isfinite(r["value"]) for r in rows))
            with self.assertRaises(ValueError):
                module.run(0,False)
        self.assertEqual(len(refs),15)
        self.assertEqual(len(set(refs)),15)

if __name__=="__main__":
    unittest.main()
