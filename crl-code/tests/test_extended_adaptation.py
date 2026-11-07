"""扩展组件：公式手算、完整梯度、时序以及多种子有限运行；不是性能测试。"""
import importlib.util
import unittest
if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("extended adaptation 使用可选依赖 torch")
import importlib
import math
import pathlib
import random
import subprocess
import sys
import tempfile
import torch
from implementations.extended_adaptation import gae,vtrace,bayes_filter,bptt,rtu,maml,metatrace,clear,redo,intentional,meta_gradient,obgd2024,continual_backprop
from implementations.extended_adaptation._common import ac_terms

class ExtendedTests(unittest.TestCase):
    def finite_difference(self,fn,x):
        gradient=torch.zeros_like(x)
        for j in range(len(x)):
            up=x.detach().clone(); down=up.clone()
            up[j]+=1e-5; down[j]-=1e-5
            gradient[j]=(fn(up)-fn(down))/(2e-5)
        return gradient

    def test_gae_terminal_and_cutoff(self):
        a,target=gae.estimate([1.,2.],[.3,.4],[.4,10.],[False,True],lam=.5)
        torch.testing.assert_close(a,torch.tensor([1.9,1.6]))
        a,_=gae.estimate([1.],[.3],[2.],[False])
        self.assertAlmostEqual(float(a[0]),2.7,6)

    def test_vtrace_hand(self):
        target=vtrace.targets([1.,2.],[.3,.4],[.4,99.],[2.,.5],[False,True],gamma=.9)
        # residual1=.5*(2-.4)=.8； residual0=2*(1+.9*.4-.3)=2.12。
        torch.testing.assert_close(target,torch.tensor([.3+2.12+.9*.8,1.2]))

    def test_vtrace_rho_and_c_separate(self):
        target=vtrace.targets([1.,2.],[0.,0.],[0.,0.],[3.,3.],[False,True],gamma=.5,rho_bar=2.,c_bar=.25)
        # rho clips residual to2；c only clips propagated tail to.25。
        torch.testing.assert_close(target,torch.tensor([2.+.5*.25*4.,4.]))
        terminal=vtrace.targets([1.,2.],[0.,0.],[9.,9.],[3.,3.],[True,True],gamma=.5,rho_bar=2.,c_bar=.25)
        torch.testing.assert_close(terminal,torch.tensor([2.,4.]))

    def test_bayes_normalization(self):
        p=bayes_filter.filter_step([.8,.2],1)
        prior=.1*.8+.9*.2
        self.assertAlmostEqual(p[1],.8*prior/(.8*prior+.2*(1-prior)))
        self.assertAlmostEqual(sum(p),1.)

    def test_full_bptt_finite_difference(self):
        x=torch.tensor([.7,.1,-.1,.7,.3,-.2,.4,-.2,0.],dtype=torch.double,requires_grad=True)
        inputs=[1.,.2,-.1,0.,.1]
        loss,g=bptt.loss_gradient(x,inputs,1.)
        expected=self.finite_difference(lambda z:.5*(bptt.predict(z,inputs)-1).square(),x)
        torch.testing.assert_close(g,expected,atol=1e-7,rtol=1e-5)
        _,short=bptt.loss_gradient(x,inputs,1.,True)
        self.assertGreater(float((g-short).abs().max()),1e-5)

    def test_rtu_all_parameters_finite_difference(self):
        x=torch.tensor([math.log(-math.log(.8)),math.log(.2),.3,-.2,.4,-.2,0.],dtype=torch.double)
        inputs=[1.,.2,-.1,0.,.1]
        _,g=rtu.loss_gradient(x,inputs,1.)
        expected=self.finite_difference(lambda z:.5*(rtu.predict(z,inputs)-1).square(),x)
        torch.testing.assert_close(g,expected,atol=1e-7,rtol=1e-5)

    def test_maml_second_order(self):
        x=torch.tensor([.1,.2,-.3],dtype=torch.double,requires_grad=True)
        support=torch.eye(3,dtype=torch.double)*2
        sy=torch.tensor([1.,2.,3.],dtype=torch.double)
        query=torch.eye(3,dtype=torch.double)
        qy=torch.zeros(3,dtype=torch.double)
        g=torch.autograd.grad(maml.objective(x,support,sy,query,qy),x)[0]
        def fn(z):
            z=z.requires_grad_()
            return maml.objective(z,support,sy,query,qy).detach()
        torch.testing.assert_close(g,self.finite_difference(fn,x),atol=1e-7,rtol=1e-5)

    def test_metatrace_reads_old_sensitivity(self):
        w,h,z=torch.zeros(2),torch.tensor([.1,.2]),torch.zeros(2)
        result=metatrace.step(w,h,z,0.,math.log(.03),0.,0.,torch.ones(2),torch.zeros(2),2.)
        self.assertAlmostEqual(result[3],.3,6)
        torch.testing.assert_close(result[0],result[-1]*2*torch.ones(2))
        self.assertLessEqual(result[-1]*2,1.)

    def test_metatrace_linear_sensitivity_finite_difference(self):
        # 固定beta且约束不活跃的线性critic：忽略trace Jacobian在这里精确成立。
        def trajectory(beta):
            w=torch.tensor([.2]); h=torch.zeros(1); z=torch.zeros(1)
            zb=v=u=0.
            for reward in (1.,.2,-.3):
                w,h,z,zb,beta,v,u,_=metatrace.step(w,h,z,zb,beta,v,u,torch.ones(1),-torch.ones(1),reward-float(w[0]),mu=0.)
            return float(w[0]),float(h[0])
        beta=math.log(.03)
        _,h=trajectory(beta)
        numerical=(trajectory(beta+.001)[0]-trajectory(beta-.001)[0])/.002
        self.assertAlmostEqual(h,numerical,places=5)

    def test_reservoir_capacity_and_recycler(self):
        buffer=[]; rng=random.Random(1)
        for t in range(1,101):clear.reservoir(buffer,t,t,rng,8)
        self.assertEqual(len(buffer),8)
        self.assertNotEqual(buffer,list(range(93,101)))
        net=redo.ReLUNet()
        before=net.hidden.weight[1].detach().clone()
        redo.replace_units(net,[0],rng)
        torch.testing.assert_close(net.hidden.weight[1],before)
        self.assertEqual(float(net.output.weight[:,0].detach().abs().sum()),0.)
        self.assertEqual(float(net.hidden.bias[0].detach()),0.)

    def test_obgd_target_trace_scale_order(self):
        parameter=torch.tensor([1.,2.],requires_grad=True)
        trace,alpha=obgd2024.step([parameter],torch.tensor([2.,1.]),torch.tensor([1.,-2.]),3.,alpha=.2,q=.5,kappa=2.)
        torch.testing.assert_close(trace,torch.tensor([2.,-1.5]))
        self.assertAlmostEqual(alpha,.2/(.2*2*3*3.5),7)
        torch.testing.assert_close(parameter.detach(),torch.tensor([1.,2.])+alpha*3*trace)

    def test_score_function_gradient(self):
        logits=torch.tensor([.2,-.1],requires_grad=True)
        loss=-2.*logits.log_softmax(0)[1]
        gradient=torch.autograd.grad(loss,logits)[0]
        torch.testing.assert_close(gradient,-2*(torch.tensor([0.,1.])-logits.detach().softmax(0)))

    def test_cbp_maturity_protection(self):
        # 无成熟单元时不能回收；预算10 < maturity20。
        rows=continual_backprop.run(0,10)
        self.assertEqual(rows[-1]['replaced_units'],0)
        util=torch.tensor([0.,.1,.2]); age=torch.tensor([20.,21.,100.])
        selected,credit=continual_backprop.select_units(util,age,.9,rate=.1)
        # immature unit0 excluded; corrected utility unit2 < unit1 despite larger raw EMA。
        self.assertEqual(selected,[2])
        self.assertAlmostEqual(credit,.1)

    def test_clear_clone_teacher_detached(self):
        actor=torch.nn.Linear(2,2,bias=False); critic=torch.nn.Linear(2,1,bias=False)
        with torch.no_grad():actor.weight.zero_();critic.weight.zero_()
        x=torch.tensor([1.,0.]); old_p=torch.tensor([.8,.2]); old_v=torch.tensor(.4)
        item=(x,0,1.,old_p,old_v)
        base=clear.replay_loss(actor,critic,item,False)
        cloned=clear.replay_loss(actor,critic,item,True)
        expected=.1*(old_p*(old_p.log()-math.log(.5))).sum()+.1*.4**2
        torch.testing.assert_close(cloned-base,expected)
        cloned.backward()
        self.assertFalse(old_p.requires_grad)
        self.assertTrue(torch.isfinite(actor.weight.grad).all())

    def test_each_module_direct_cli(self):
        root=pathlib.Path(__file__).parents[1]
        with tempfile.TemporaryDirectory(prefix='extended-cli-') as output:
            for path in sorted((root/'implementations'/'extended_adaptation').glob('*.py')):
                if path.name.startswith('_'):continue
                result=subprocess.run([sys.executable,str(path),'--steps','40','--seeds','0','--out',str(pathlib.Path(output)/path.stem)],cwd=root,capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,0,(path.name,result.stderr))

    def test_intentional_functional_first_step_and_terminal(self):
        opt=intentional.IntentionalOptimizer(2,.2)
        gradient=torch.tensor([2.,-1.])
        update,alpha=opt.step(gradient,3.,True)
        # 首步 sigma 与 trace 质量相同，g^T Δw=eta*delta。
        self.assertAlmostEqual(float(gradient@update),.6,6)
        self.assertEqual(float(opt.trace.abs().sum()),0.)
        self.assertGreater(float(opt.v.sum()),0.)  # 终止只清 trace，不清尺度统计。

    def test_intentional_zero_delta_entropy_sign(self):
        probabilities=torch.tensor([.9,.1])
        g,_=intentional.policy_gradient(probabilities,1,0.)
        score=torch.tensor([0.,1.])-probabilities
        torch.testing.assert_close(g,score)
        # 零delta只积累真实score，不向trace/RMS混入有偏熵项。
        opt=intentional.IntentionalOptimizer(2,.03,policy=True)
        update,_=opt.step(g,0.)
        torch.testing.assert_close(opt.trace,score)
        torch.testing.assert_close(opt.v,.001*score.square())
        self.assertEqual(float(update.abs().sum()),0.)

    def test_clear_switch_boundary(self):
        rows=clear.run(0,10)
        midpoint=next(r for r in rows if r['step']==5)
        self.assertEqual(midpoint['value'],midpoint['old_return'])

    def test_actor_critic_gradient_domains_and_terminal_target(self):
        w=torch.arange(18).float()*.01
        x=torch.tensor([1.,0.,0.,0.,0.,1.])
        xp=torch.tensor([0.,1.,0.,0.,0.,.5])
        delta,g_u,g_delta=ac_terms(w,x,1,.7,xp,True)
        self.assertAlmostEqual(delta,float(.7-w[:6]@x),6)
        torch.testing.assert_close(g_delta[:6],-x)
        torch.testing.assert_close(g_delta[6:],torch.zeros(12))
        torch.testing.assert_close(g_u[:6],x)
        policy=w[6:].reshape(2,6).clone().requires_grad_()
        expected=torch.autograd.grad((policy@x).log_softmax(0)[1],policy)[0]
        torch.testing.assert_close(g_u[6:].reshape(2,6),.5*expected)
        _,_,continued=ac_terms(w,x,1,.7,xp,False)
        torch.testing.assert_close(continued[:6],xp-x)

    def test_recyclers_no_stale_sgd_state(self):
        for module in (redo,continual_backprop):
            net=module.ReLUNet(); opt=torch.optim.SGD(net.parameters(),lr=.03)
            opt.zero_grad(); net(torch.ones(4)).square().backward(); opt.step()
            self.assertEqual(len(opt.state),0)
            module.replace_units(net,[2],random.Random(1))
            self.assertEqual(float(net.output.weight[:,2].detach().abs().sum()),0.)
            self.assertEqual(len(opt.state),0)

    def test_meta_gradient_finite_difference(self):
        train=[(3,0.,4,False),(4,0.,5,False),(5,1.,6,True)]
        validation=[(3,0.,2,False),(2,0.,1,False),(1,0.,0,True)]
        theta=torch.tensor([.1,.2,.3,.4,.5],requires_grad=True)
        beta=torch.tensor(.3,requires_grad=True)
        _,_,_,derivative=meta_gradient.meta_step(theta,beta,train,validation)
        def outer(b):
            th=theta.detach().clone().requires_grad_()
            return meta_gradient.meta_step(th,torch.tensor(b,requires_grad=True),train,validation)[2]
        expected=(outer(.301)-outer(.299))/.002
        self.assertAlmostEqual(derivative,expected,places=5)

    def test_meta_gradient_nonterminal_validation_target(self):
        train=[(3,0.,4,False),(4,0.,5,False),(5,1.,6,True)]
        validation=[(3,0.,4,False),(4,0.,5,False)]
        theta=torch.tensor([.1,.2,.3,.4,.5],requires_grad=True)
        beta=torch.tensor(.3,requires_grad=True)
        updated,_,outer,derivative=meta_gradient.meta_step(theta,beta,train,validation)
        target=meta_gradient.returns(updated,validation,torch.tensor(1.)).detach()
        # lambda'=1的cutoff bootstrap应是更新后的V(5)，不是旧theta[4]。
        torch.testing.assert_close(target,updated.detach()[4].repeat(2))
        self.assertNotAlmostEqual(float(target[0]),float(theta.detach()[4]),6)
        expected=.5*(updated.detach()[torch.tensor([2,3])]-target).square().mean()
        self.assertAlmostEqual(outer,float(expected),7)
        def fixed_target_outer(b):
            th=theta.detach().clone().requires_grad_()
            targets=meta_gradient.returns(th,train,torch.tensor(b).sigmoid())
            inner=.5*(th[torch.tensor([2,3,4])]-targets).square().mean()
            grad=torch.autograd.grad(inner,th)[0]
            new=th-.1*grad
            return float((.5*(new[torch.tensor([2,3])]-target).square().mean()).detach())
        numerical=(fixed_target_outer(.301)-fixed_target_outer(.299))/.002
        self.assertAlmostEqual(derivative,numerical,places=5)

    def test_all_methods_small_multiseed(self):
        root=pathlib.Path(__file__).parents[1]/"implementations"/"extended_adaptation"
        modules={}
        for path in root.glob("*.py"):
            if path.name.startswith("_"):continue
            module=importlib.import_module("implementations.extended_adaptation."+path.stem)
            modules[module.META["id"]]=module
            for seed in (0,1,2):
                emitted=[]
                rows=module.run(seed,40,emitted.append)
                self.assertEqual(rows,emitted)
                self.assertEqual(rows[0]["step"],0)
                self.assertEqual(rows[-1]["step"],40)
                self.assertTrue(all(math.isfinite(r["value"]) for r in rows))
                self.assertLessEqual(len(rows),100)
        for module in modules.values():
            base=modules[module.META["baseline"]]
            for key in ("task","family","budget","metric","unit","higher_better"):
                self.assertEqual(module.META[key],base.META[key],(module.META["id"],key))

if __name__=="__main__":unittest.main()
