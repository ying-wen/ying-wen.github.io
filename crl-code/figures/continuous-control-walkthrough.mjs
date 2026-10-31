/** Constructed two-step actuator example; no random rollouts or fitted networks. */
export const lastQ=a=>-((.25+a-1)**2)-.1*a*a;
export const lastSlope=a=>1.5-2.2*a;
export const critic=(a,amplitude=.8)=>lastQ(a)+amplitude*Math.exp(-.5*((a-1.4)/.12)**2);
export const criticSlope=a=>lastSlope(a)+.8*Math.exp(-.5*((a-1.4)/.12)**2)*(1.4-a)/.12**2;
export const targetAction=(a,e)=>Math.max(-2,Math.min(2,a+Math.max(-.2,Math.min(.2,e))));
const normal=e=>Math.exp(-.5*(e/.15)**2)/(.15*Math.sqrt(2*Math.PI));
const integrate=(f,l,r,n=2000)=>{
  const h=(r-l)/n;let total=f(l)+f(r);
  for(let i=1;i<n;i++)total+=(i%2?4:2)*f(l+i*h);
  return total*h/3;
};
// Compute tail mass by normalizing the central interval, independent of Python erfc.
export function smoothedValue(a=1.4){
  const tail=(1-integrate(normal,-.2,.2))/2;
  return tail*(critic(targetAction(a,-.2),.3)+critic(targetAction(a,.2),.3))+
    integrate(e=>normal(e)*critic(targetAction(a,e),.3),-.2,.2);
}
export function continuousControlData(){
  const action=1.2,theta=Math.atanh(action/2),jac=2*(1-(action/2)**2),gradient=criticSlope(action)*jac;
  const nextAction=2*Math.tanh(theta+.05*gradient),r=-.56875;
  const mean=.3,logStd=-.7,epsilon=.4,sigma=Math.exp(logStd),u=mean+sigma*epsilon;
  const z=Math.tanh(u),a=2*z,lj=2*(Math.log(2)-Math.abs(u)-Math.log1p(Math.exp(-2*Math.abs(u))));
  const lpNormal=-.5*epsilon**2-logStd-.5*Math.log(2*Math.PI),lpNorm=lpNormal-lj,lp=lpNorm-Math.log(2);
  const entropyMean=.4*z,valueMean=-lastSlope(a)*2*(1-z*z);
  const entropyStd=.2*(-1+2*z*sigma*epsilon),valueStd=valueMean*sigma*epsilon;
  const smooth=smoothedValue(),targetEntropy=.1+Math.log(2);
  return {
    kind:'constructed-numerical-example',random_rollouts:0,optimizer_steps:0,hypothetical_analytic_steps:1,
    task:{initial:[0,2],observed_action:.25,next_state:[.25,1],reward:r,terminal:false,gamma:.9,action_bounds:[-2,2]},
    actor:{action,theta,da_dtheta:jac,true_slope:lastSlope(action),critic_slope:criticSlope(action),gradient,step_size:.05,
      next_action:nextAction,true_before:lastQ(action),true_after:lastQ(nextAction),estimated_before:critic(action),estimated_after:critic(nextAction)},
    targets:{base_action:1.4,sigma:.15,noise_clip:.2,q1_center:critic(1.4),q2_center:critic(1.4,.3),min_smoothed:smooth,
      ddpg:r+.9*critic(1.4),twin_only:r+.9*critic(1.4,.3),td3_expectation:r+.9*smooth,
      gaussian_tail_mass:(1-integrate(normal,-.2,.2))/2,terminal_target:r,quadrature_intervals:2000},
    sac:{mean,log_std:logStd,epsilon,sigma,u,normalized_action:z,action:a,logp_normal:lpNormal,log_jacobian:lj,
      logp_normalized:lpNorm,logp:lp,q:lastQ(a),loss:.2*lp-lastQ(a),da_dmean:2*(1-z*z),
      gradient_mean:entropyMean+valueMean,gradient_log_std:entropyStd+valueStd,entropy_mean:entropyMean,value_mean:valueMean,
      entropy_log_std:entropyStd,value_log_std:valueStd,alpha:.2,target:r+.9*(lastQ(a)-.2*lp),
      target_entropy_normalized:.1,target_entropy_physical:targetEntropy,temperature_residual:lp+targetEntropy,
      beta_gradient:-.2*(lp+targetEntropy),beta_gradient_unshifted:-.2*(lp+.1)},
    curves:Array.from({length:181},(_,i)=>{const a=i/100;return [a,lastQ(a),critic(a),critic(a,.3)];}),
  };
}
