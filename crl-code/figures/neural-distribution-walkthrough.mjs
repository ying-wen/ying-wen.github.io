/** A tiny shared nonlinear QR predictor. Four deterministic expectation updates; no RNG. */
import {midpointTaus,pairwise,readout,inverseCDFDistance} from './quantile-walkthrough.mjs';
export const TAUS=Object.freeze(midpointTaus(4));
export const TRUE_FIXED=Object.freeze([1.5,1.5,1.5,1.5]);
export const TRUE_FLOATING=Object.freeze([.5,.5,2.5,2.5]);
function validate(p){
 if(!p||!Number.isFinite(p.a)||!Number.isFinite(p.c)||p.w?.length!==4||p.b?.length!==4||[...p.w,...p.b].some(x=>!Number.isFinite(x)))throw new RangeError('finite ten-parameter network required');
}
export function network(p,x){
 validate(p);if(!Number.isFinite(x))throw new RangeError('finite input required');
 const h=Math.tanh(p.a*x+p.c);return {h,atoms:p.b.map((b,i)=>b+p.w[i]*h)};
}
export function fitTwoInputs(fixed,floating){
 if(fixed.length!==4||floating.length!==4||[...fixed,...floating].some(x=>!Number.isFinite(x)))throw new RangeError('two finite four-head targets required');
 return {a:.5,c:.5,w:floating.map((v,i)=>(v-fixed[i])/Math.tanh(1)),b:[...fixed]};
}
export function lossGradient(p,x,target){
 const predicted=network(p,x),qr=pairwise(predicted.atoms,TAUS,target);
 const hidden=qr.gradient.reduce((s,d,i)=>s+d*p.w[i],0)*(1-predicted.h**2);
 return {...predicted,loss:qr.loss,output_gradient:qr.gradient,gradient:{a:x*hidden,c:hidden,w:qr.gradient.map(d=>d*predicted.h),b:[...qr.gradient]}};
}
export function step(p,x,target,rate=.5){
 if(!Number.isFinite(rate)||rate<=0)throw new RangeError('positive finite rate required');
 const before=lossGradient(p,x,target),g=before.gradient;
 const after={a:p.a-rate*g.a,c:p.c-rate*g.c,w:p.w.map((v,i)=>v-rate*g.w[i]),b:p.b.map((v,i)=>v-rate*g.b[i])};
 return {rate,before,after,after_loss:lossGradient(after,x,target).loss};
}
export function jacobian(p,x){
 const {h}=network(p,x),sech=1-h*h;
 return p.w.map((w,i)=>[w*sech*x,w*sech,...p.w.map((_,j)=>i===j?h:0),...p.b.map((_,j)=>i===j?1:0)]);
}
export function crossJacobian(p,left,right){
 const a=jacobian(p,left),b=jacobian(p,right);
 return a.map(row=>b.map(other=>row.reduce((s,v,k)=>s+v*other[k],0)));
}
const quarter=[.25,.25,.25,.25];
function prediction(p,x,target){
 const atoms=network(p,x).atoms;
 return {atoms,...readout(atoms),loss:pairwise(atoms,TAUS,target).loss,w1:inverseCDFDistance(atoms,quarter,target,quarter)};
}
export function finiteSampleEnumeration(){
 const sequences=[];
 for(let bits=0;bits<16;bits++){
  const rewards=Array.from({length:4},(_,i)=>(bits>>i)&1?4:0),returns=rewards.map(r=>.5+.5*r);
  const countLow=rewards.filter(r=>r===0).length;
  sequences.push({rewards,returns,count_low:countLow,empirical_cdf_at_one:countLow/4,absolute_cdf_error:Math.abs(countLow/4-.5),mass:1/16});
 }
 return {specified:[0,0,0,4],specified_returns:[.5,.5,.5,2.5],specified_cdf_at_one:.75,true_cdf_at_one:.5,
  sequences,counts:Array.from({length:5},(_,k)=>sequences.filter(s=>s.count_low===k).length),
  expected_cdf:sequences.reduce((v,s)=>v+s.mass*s.empirical_cdf_at_one,0),
  expected_absolute_cdf_error:sequences.reduce((v,s)=>v+s.mass*s.absolute_cdf_error,0),
  expected_squared_cdf_error:sequences.reduce((v,s)=>v+s.mass*(s.empirical_cdf_at_one-.5)**2,0)};
}
export function neuralDistributionData(){
 const initial=fitTwoInputs(TRUE_FIXED,[1,1,2,2]),snapshots=[];let p=initial;
 for(let k=0;k<=4;k++){
  snapshots.push({updates:k,parameters:p,fixed:prediction(p,-1,TRUE_FIXED),floating:prediction(p,1,TRUE_FLOATING)});
  if(k<4)p=step(p,1,TRUE_FLOATING).after;
 }
 // A separate successor network sees terminal labels 0 / 4. It is not the parent predictor above.
 const successorBefore=fitTwoInputs([2,2,2,2],[1,1,3,3]),successorStep=step(successorBefore,1,[0,0,4,4]);
 const targetBefore=network(successorBefore,1).atoms.map(z=>.5+.5*z),targetAfter=network(successorStep.after,1).atoms.map(z=>.5+.5*z);
 const heldParent=[1,1,2,2];
 return {kind:'exact-shared-tanh-quantile-mechanism',random_rollouts:0,optimizer_updates:4,successor_optimizer_updates:1,stochastic_training_runs:0,
  task:{gamma:.5,advance:.5,fixed_final_reward:2,floating_final_rewards:[0,4],floating_final_masses:[.5,.5]},
  inputs:{fixed:-1,floating:1},taus:TAUS,true_fixed:TRUE_FIXED,true_floating:TRUE_FLOATING,
  initial,first_step:step(initial,1,TRUE_FLOATING),snapshots,exact_fit:fitTwoInputs(TRUE_FIXED,TRUE_FLOATING),
  initial_cross_jacobian:crossJacobian(initial,-1,1),
  aliased_inputs:{input:1,truth_distance:1,minimum_sum_w1_lower_bound:1},
  bootstrap:{successor_before:network(successorBefore,1).atoms,successor_after:network(successorStep.after,1).atoms,target_before:targetBefore,target_after:targetAfter,
   frozen_target:[...targetBefore],held_parent:heldParent,parent_loss_before:pairwise(heldParent,TAUS,targetBefore).loss,parent_loss_after:pairwise(heldParent,TAUS,targetAfter).loss,
   target_w1_before:inverseCDFDistance(targetBefore,quarter,TRUE_FLOATING,quarter),target_w1_after:inverseCDFDistance(targetAfter,quarter,TRUE_FLOATING,quarter),
   successor_step:successorStep},finite_sample:finiteSampleEnumeration()};
}
