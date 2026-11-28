// The calculation is separate from SVG rendering. No random training or
// access to exact labels inside sampleEpisode/update. Python is standalone;
// tests compare these two implementations and use a third algebraic oracle.
const rewards=[[2,0],[4,0]],gamma=.5,epsilon=.2;
const initialPolicy=[[.5,.5],[.5,.5]],uniforms=[[.4,.8],[.95,.05],[.05,.5]];
const copy=value=>structuredClone(value);
export function policyPair(policy){
  if(!Array.isArray(policy)||policy.length!==2||policy.some(row=>!Array.isArray(row)||row.length!==2||row.some(p=>!Number.isFinite(p)||p<0)||Math.abs(row[0]+row[1]-1)>1e-12))throw new RangeError('Two legal action distributions are required');
  return copy(policy);
}
export function epsilonGreedy(q,e=epsilon){
  if(!Number.isFinite(e)||e<=0||e>1)throw new RangeError('0 < epsilon <= 1');
  return q.map(row=>{const greedy=row[0]>row[1]?0:1;return row.map((_,a)=>e/2+(a===greedy?1-e:0));});
}
export function episodeFromActions(policy,actions){
  policy=policyPair(policy);
  if(actions.length!==2||actions.some(a=>a!==0&&a!==1))throw new RangeError('Two legal actions are needed');
  const logged=actions.map((a,t)=>policy[t][a]);
  if(logged.some(p=>p===0))throw new RangeError('Episode is outside policy support');
  const r=actions.map((a,t)=>rewards[t][a]);
  return {path:actions.map(a=>'AB'[a]).join(''),actions:[...actions],rewards:r,returns:[r[0]+gamma*r[1],r[1]],probability:logged[0]*logged[1],policy,logged_probabilities:logged};
}
export function sampleEpisode(policy,u){
  policy=policyPair(policy);
  if(u.length!==2||u.some(v=>!Number.isFinite(v)||v<0||v>=1))throw new RangeError('Two uniform numbers in [0,1) are needed');
  return {...episodeFromActions(policy,u.map((v,t)=>v<policy[t][0]?0:1)),uniforms:[...u]};
}
export function update(q,counts,episode,e=epsilon){
  const Q=copy(q),N=copy(counts);
  episode.actions.forEach((a,t)=>{N[t][a]++;Q[t][a]+=(episode.returns[t]-Q[t][a])/N[t][a];});
  return {q:Q,counts:N,policy:epsilonGreedy(Q,e)};
}
export function exactValue(policy){
  policy=policyPair(policy);
  const values=[0,0,0],q=[[0,0],[0,0]];
  for(const s of [1,0]){q[s]=rewards[s].map(r=>r+gamma*values[s+1]);values[s]=q[s].reduce((v,x,a)=>v+policy[s][a]*x,0);}
  return {values:values.slice(0,2),q};
}
export function adaptiveBranches(episodes=2,e=epsilon){
  if(!Number.isInteger(episodes)||episodes<1||episodes>3)throw new RangeError('Enumerate one to three episodes');
  let nodes=[{paths:[],probability:1,q:[[0,0],[0,0]],counts:[[0,0],[0,0]],policy:copy(initialPolicy)}];
  for(let i=0;i<episodes;i++)nodes=nodes.flatMap(node=>[0,1].flatMap(a=>[0,1].map(b=>{
    const episode=episodeFromActions(node.policy,[a,b]),next=update(node.q,node.counts,episode,e);
    return {paths:[...node.paths,episode.path],probability:node.probability*episode.probability,...next};
  })));
  nodes.forEach(node=>node.true_current_value=exactValue(node.policy).values[0]);
  return {episodes,branches:nodes,probability_sum:nodes.reduce((v,n)=>v+n.probability,0),expected_current_value:nodes.reduce((v,n)=>v+n.probability*n.true_current_value,0),below_initial_probability:nodes.reduce((v,n)=>v+(n.true_current_value<2-1e-12?n.probability:0),0)};
}
export function frozenRegimeMixture(){
  const old=.5,current=.9,batches=[0,1].flatMap(a=>[0,1].map(b=>{
    const returns=[2+gamma*rewards[1][a],2+gamma*rewards[1][b]];
    return {actions_at_S1:[a,b],probability:(a===0?old:1-old)*(b===0?current:1-current),returns,running_average:(returns[0]+returns[1])/2};
  }));
  return {old_A_probability:old,current_A_probability:current,old_q:2+2*old,current_q:2+2*current,batches,expected_running_average:batches.reduce((v,r)=>v+r.probability*r.running_average,0)};
}
export function suffixWeights(p=.9,lengths=[1,2,5,10,20,50],budget=100){
  if(!Number.isFinite(p)||p<=0||p>1||lengths.some(h=>!Number.isInteger(h)||h<1))throw new RangeError('Positive support and integer lengths required');
  return lengths.map(h=>({remaining_decisions:h,match_probability:p**h,nonzero_weight:p**(-h),target_mean:1,weight_variance:p**(-h)-1,expected_matches:budget*p**h,zero_matches_probability:(1-p**h)**budget}));
}
export function mcControlWalkthroughData(){
  let q=[[0,0],[0,0]],counts=[[0,0],[0,0]],policy=copy(initialPolicy);
  const rows=uniforms.map((u,i)=>{
    const row={...sampleEpisode(policy,u),episode:i+1,q_before:copy(q),counts_before:copy(counts),exact_before:exactValue(policy)};
    ({q,counts,policy}=update(q,counts,row));
    return {...row,q_after:copy(q),counts_after:copy(counts),policy_after:copy(policy),exact_after:exactValue(policy)};
  });
  return {kind:'exact_finite_control',gamma,epsilon,rewards:copy(rewards),initial_policy:copy(initialPolicy),initial_value:exactValue(initialPolicy).values[0],assigned_episodes:rows,adaptive_enumerations:[adaptiveBranches(1),adaptiveBranches(2)],frozen_regime_mixture:frozenRegimeMixture(),coverage_cost:{optimal_soft_policy:[[.9,.1],[.9,.1]],soft_value:3.6,greedy_value:4,value_cost:.4,nonpreferred_action_probability:.1,mean_wait_episodes:10,no_B_in_10_probability:.9**10},long_suffix:{behavior_A_probability:.9,budget:100,rows:suffixWeights(),uniform_behavior_H20:suffixWeights(.5,[20])[0]}};
}
