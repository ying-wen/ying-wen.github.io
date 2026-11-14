/** Fixed two-step support example. No interaction, fitted networks or random data. */
export const GAMMA=.9;
export const BEHAVIOR=[.5,.5,0];
export const GIVEN_Q=[0,2,4];
export const TRUE_REWARDS=[0,2,-2];
export const valueTarget=(reward,terminal,nextValue)=>terminal?reward:reward+GAMMA*nextValue;
export const softmax=q=>{
  const peak=Math.max(...q),weights=q.map(v=>Math.exp(v-peak)),total=weights.reduce((a,b)=>a+b,0);
  return weights.map(v=>v/total);
};
const checkProb=p=>{
  if(!p.length||p.some(v=>!Number.isFinite(v)||v<0)||Math.abs(p.reduce((a,b)=>a+b,0)-1)>1e-12)throw new RangeError('probabilities must sum to one');
};
export function checkSupport(pi,mu=BEHAVIOR){
  checkProb(pi);checkProb(mu);
  if(pi.length!==mu.length||pi.some((v,i)=>v>0&&mu[i]===0))throw new RangeError('target has an unsupported action');
}
export function expectile(q,p,tau){
  checkProb(p);
  if(q.length!==p.length||q.some((v,i)=>p[i]>0&&!Number.isFinite(v))||!(tau>0&&tau<1))throw new RangeError('finite supported values and 0 < tau < 1 required');
  const supported=q.filter((_,i)=>p[i]>0);let low=Math.min(...supported),high=Math.max(...supported);
  for(let k=0;k<90;k++){
    const v=(low+high)/2,balance=q.reduce((total,x,i)=>p[i]===0?total:total+p[i]*(x<v?1-tau:tau)*(x-v),0);
    if(balance>0)low=v;else high=v;
  }
  return (low+high)/2;
}
export const expectileLoss=(v,q=GIVEN_Q,p=BEHAVIOR,tau=.75)=>q.reduce((sum,x,i)=>p[i]===0?sum:sum+p[i]*(x<v?1-tau:tau)*(x-v)**2,0);
export function extractedPolicy(q=GIVEN_Q,p=BEHAVIOR,v=1.5,beta=1){
  checkProb(p);
  if(q.length!==p.length||!Number.isFinite(beta)||beta<0)throw new RangeError('nonnegative finite inverse temperature required');
  const peak=Math.max(...q.filter((_,i)=>p[i]>0).map(x=>beta*(x-v)));
  const w=q.map((x,i)=>p[i]===0?0:p[i]*Math.exp(beta*(x-v)-peak)),z=w.reduce((a,b)=>a+b,0);
  return w.map(x=>x/z);
}
/** The s contribution to the full eight-transition mean, with frozen terminal labels. */
export function cqlLoss(q,alpha=1,stateMass=.5){
  const peak=Math.max(...q),logsum=peak+Math.log(q.reduce((sum,x)=>sum+Math.exp(x-peak),0));
  const bellman=.5*BEHAVIOR.reduce((sum,p,i)=>p===0?sum:sum+p*(q[i]-TRUE_REWARDS[i])**2,0);
  const reg=logsum-BEHAVIOR.reduce((sum,p,i)=>sum+p*q[i],0);
  return stateMass*(bellman+alpha*reg);
}
export const cqlGradient=(q,alpha=1,stateMass=.5)=>{
  const s=softmax(q);
  return q.map((x,i)=>stateMass*((BEHAVIOR[i]===0?0:BEHAVIOR[i]*(x-TRUE_REWARDS[i]))+alpha*(s[i]-BEHAVIOR[i])));
};
export function dataset(unseenReward=-2){
  return [0,0,1,1].map(action=>[
    {state:'start',action:'go',reward:0,next_state:'s',terminal:false,mu:1},
    {state:'s',action,reward:[0,2,unseenReward][action],next_state:null,terminal:true,mu:.5},
  ]);
}
export function pdis(episodes,pi){
  checkSupport(pi);
  return episodes.reduce((total,episode)=>{
    let w=1,value=0;
    episode.forEach((step,t)=>{w*=step.state==='start'?1:pi[step.action]/step.mu;value+=GAMMA**t*w*step.reward;});
    return total+value;
  },0)/episodes.length;
}
export function doublyRobust(episodes,pi,q=GIVEN_Q){
  checkSupport(pi);
  const v=pi.reduce((sum,p,i)=>sum+p*q[i],0),qStart=GAMMA*v;
  return episodes.reduce((sum,episode)=>{
    let result=0;
    for(const step of [...episode].reverse()){
      const qs=step.state==='start'?qStart:q[step.action],vs=step.state==='start'?qStart:v;
      const rho=step.state==='start'?1:pi[step.action]/step.mu;
      result=vs+rho*(step.reward+GAMMA*result-qs);
    }
    return sum+result;
  },0)/episodes.length;
}
export function offlineSupportData(){
  const q=[...GIVEN_Q],prob=softmax(q),g=cqlGradient(q),next=q.map((x,i)=>x-g[i]);
  const v=expectile(q,BEHAVIOR,.75),pi=extractedPolicy(q,BEHAVIOR,v,1),episodes=dataset();
  const candidate=[.1,.8,.1];
  return {
    kind:'constructed-exact-example',random_rollouts:0,optimizer_steps:0,hypothetical_gradient_steps:1,
    task:{gamma:GAMMA,states:['start','s'],start_action:'go',start_reward:0,terminal_rewards:TRUE_REWARDS,unseen_reward_known_to_learner:false},
    dataset:{episodes,episode_count:4,transition_count:8,state_mass:{start:.5,s:.5},action_counts:[2,2,0],behavior:BEHAVIOR},
    bootstrap:{q,greedy_action:2,observed_terminal_squared_error:0,start_target:valueTarget(0,false,Math.max(...q)),true_greedy_return:GAMMA*TRUE_REWARDS[2],
      hypothetical_online_feedback:{action:2,reward:-2,terminal:true,step_size:1,next_q:[0,2,-2],next_start_target:1.8}},
    cql:{alpha:1,state_mass:.5,step_size:1,softmax:prob,conditional_regularizer_gradient:prob.map((x,i)=>x-BEHAVIOR[i]),
      full_mean_gradient:g,next_q:next,next_start_target:GAMMA*Math.max(...next),loss_before:cqlLoss(q),loss_after:cqlLoss(next)},
    iql:{tau:.75,beta:1,value:v,value_loss:expectileLoss(v),q_start_target:valueTarget(0,false,v),
      advantages:q.map((x,i)=>BEHAVIOR[i]===0?null:x-v),actor_probability:pi,true_actor_return:GAMMA*pi.reduce((sum,p,i)=>sum+p*TRUE_REWARDS[i],0),
      expectile_curve:Array.from({length:81},(_,i)=>{const x=-.5+i*3/80;return [x,expectileLoss(x)];}),
      tau_examples:[.5,.75,.9,.99,.999999].map(tau=>[tau,expectile(q,BEHAVIOR,tau)])},
    ope:{fixed_policy:pi,pdis:pdis(episodes,pi),dr_exact_supported_model:doublyRobust(episodes,pi),dr_zero_model:doublyRobust(episodes,pi,[0,0,0]),
      unsupported_candidate:candidate,same_logs_in_both_worlds:JSON.stringify(dataset(-2))===JSON.stringify(dataset(4)),
      true_candidate_returns:[-2,4].map(r=>GAMMA*(candidate[1]*2+candidate[2]*r)),
      naive_missing_mass_pdis:GAMMA*candidate[1]*2,naive_uncorrected_model:GAMMA*(candidate[1]*2+candidate[2]*4)},
  };
}
