/** Constructed finite trajectory and candidate policy; no sampled training. */
export function gaePpoData(){
  const rewards=[0,0,1],values=[.2,.8,.4,0],gamma=1,lam=.5,epsilon=.2;
  const delta=rewards.map((r,t)=>r+gamma*values[t+1]-values[t]);
  const advantage=[0,0,0];let carry=0;
  for(let t=2;t>=0;t--){carry=delta[t]+gamma*lam*carry;advantage[t]=carry;}
  const old_probability=[.5,.5,.5],probability=[.7,.3,.55];
  const ratio=probability.map((p,t)=>p/old_probability[t]);
  const term=ratio.map((r,t)=>Math.min(r*advantage[t],Math.max(1-epsilon,Math.min(1+epsilon,r))*advantage[t]));
  const logp_derivative=ratio.map((r,t)=>((advantage[t]>0&&r>1+epsilon)||(advantage[t]<0&&r<1-epsilon))?0:r*advantage[t]);
  return {kind:'constructed-exact-example',gamma,lam,epsilon,rewards,values,delta,advantage,
    lambda_target:advantage.map((a,t)=>a+values[t]),reward_to_go:[1,1,1],old_probability,probability,ratio,term,logp_derivative,
    exact_evaluation:{old_return:.5**3,candidate_return:probability.reduce((a,b)=>a*b,1)},random_rollouts:0,optimizer_steps:0};
}
