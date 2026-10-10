// Original finite calculation for Sutton & Barto §5.7. No random training.
// The controller only reads sampled transitions and logged behavior probabilities.
const clone=x=>structuredClone(x);
export const task={gamma:.5,epsilon:.2,rewards:[[2,0],[4,0]],tie_action:'B'};
export const greedy=q=>q.map(row=>row[0]>row[1]?0:1);
export function behavior(q,epsilon=task.epsilon){
  if(!Number.isFinite(epsilon)||epsilon<=0||epsilon>1)throw new RangeError('0 < epsilon <= 1 is required for support');
  return greedy(q).map(g=>[0,1].map(a=>epsilon/2+(a===g?1-epsilon:0)));
}
function probabilities(p){
  if(!Array.isArray(p)||p.some(row=>!Array.isArray(row)||row.length!==2||row.some(v=>!Number.isFinite(v)||v<=0)||Math.abs(row[0]+row[1]-1)>1e-12))throw new RangeError('Two positive action probabilities must sum to one');
}
export function episodeFromActions(policy,actions){
  probabilities(policy);
  if(policy.length!==2||actions.length!==2||actions.some(a=>a!==0&&a!==1))throw new RangeError('This task has exactly two decisions');
  const rewards=actions.map((a,s)=>task.rewards[s][a]),logged=actions.map((a,s)=>policy[s][a]);
  return {path:actions.map(a=>'AB'[a]).join(''),states:[0,1],actions:[...actions],rewards,behavior:clone(policy),logged_probabilities:logged,probability:logged[0]*logged[1],terminated:true,cutoff:false};
}
export function sampleEpisode(policy,uniforms){
  probabilities(policy);
  if(uniforms.length!==2||uniforms.some(u=>!Number.isFinite(u)||u<0||u>=1))throw new RangeError('Two uniform values in [0,1) are required');
  return {...episodeFromActions(policy,uniforms.map((u,s)=>u<policy[s][0]?0:1)),uniforms:[...uniforms]};
}
export function updateWeightedControl(q,c,episode,gamma=task.gamma){
  const Q=clone(q),C=clone(c),target=greedy(Q),trace=[];
  if(!episode.terminated||episode.cutoff)throw new RangeError('A cutoff is not true termination; complete MC return is unavailable');
  if(!Number.isFinite(gamma)||gamma<0||gamma>1)throw new RangeError('0 <= gamma <= 1');
  const T=episode.states.length;
  if(!T||['actions','rewards','logged_probabilities'].some(k=>episode[k].length!==T))throw new RangeError('Incomplete transition log');
  let G=0,W=1,breakTime=null;
  for(let t=T-1;t>=0;t--){
    const s=episode.states[t],a=episode.actions[t],p=episode.logged_probabilities[t];
    if(!Q[s]||(a!==0&&a!==1)||!Number.isFinite(p)||p<=0||p>1||!Number.isFinite(episode.rewards[t]))throw new RangeError('Invalid sampled transition');
    const before={q:clone(Q),c:clone(C),target:[...target]};
    G=episode.rewards[t]+gamma*G;
    C[s][a]+=W;
    Q[s][a]+=W/C[s][a]*(G-Q[s][a]);
    target[s]=Q[s][0]>Q[s][1]?0:1;
    const mismatch=a!==target[s],nextWeight=mismatch?0:W/p;
    trace.push({t,state:s,action:a,reward:episode.rewards[t],return:G,weight:W,logged_probability:p,before,q_after:clone(Q),c_after:clone(C),target_after:[...target],break:mismatch,next_weight:nextWeight});
    if(mismatch){breakTime=t;break;}
    W=nextWeight;
  }
  return {q:Q,c:C,target,trace,break_time:breakTime,skipped_times:breakTime===null?[]:Array.from({length:breakTime},(_,i)=>i)};
}
export function exactValues(q){
  const target=greedy(q),b=behavior(q);
  // Reader's algebraic reference, never passed to the controller.
  return {target:2*Number(target[0]===0)+2*Number(target[1]===0),behavior:2*b[0][0]+2*b[1][0],current_target_q:[[2+2*Number(target[1]===0),2*Number(target[1]===0)],[4,0]]};
}
export function adaptiveBranches(episodes){
  if(!Number.isInteger(episodes)||episodes<1||episodes>2)throw new RangeError('Enumerate one or two episodes');
  let nodes=[{paths:[],probability:1,q:[[0,0],[0,0]],c:[[0,0],[0,0]]}];
  for(let j=0;j<episodes;j++)nodes=nodes.flatMap(node=>[0,1].flatMap(a=>[0,1].map(b=>{
    const episode=episodeFromActions(behavior(node.q),[a,b]),next=updateWeightedControl(node.q,node.c,episode);
    return {paths:[...node.paths,episode.path],probability:node.probability*episode.probability,q:next.q,c:next.c,target:next.target,exact:exactValues(next.q)};
  })));
  const distribution=[0,2,4].map(value=>({value,probability:nodes.reduce((sum,n)=>sum+(n.exact.target===value?n.probability:0),0)}));
  return {episodes,branches:nodes,probability_sum:nodes.reduce((sum,n)=>sum+n.probability,0),expected_target_value:nodes.reduce((sum,n)=>sum+n.probability*n.exact.target,0),expected_behavior_value:nodes.reduce((sum,n)=>sum+n.probability*n.exact.behavior,0),target_value_distribution:distribution};
}
export function sampleRepeatedEpisode(actionUniforms,transitionUniforms){
  if(actionUniforms.length!==transitionUniforms.length||!actionUniforms.length||[...actionUniforms,...transitionUniforms].some(u=>!Number.isFinite(u)||u<0||u>=1))throw new RangeError('Matching uniform inputs in [0,1) are required');
  const states=[],actions=[],rewards=[],logged=[];let probability=1,terminated=false;
  for(let t=0;t<actionUniforms.length;t++){
    const a=actionUniforms[t]<.5?0:1;
    states.push(0);actions.push(a);rewards.push(a===0?1:0);logged.push(.5);
    terminated=a===1||transitionUniforms[t]<.5;
    probability*=.5*(a===0?.5:1);
    if(terminated)break;
  }
  return {path:actions.map(a=>'AB'[a]).join(''),states,actions,rewards,logged_probabilities:logged,terminated,cutoff:!terminated,action_uniforms:actionUniforms.slice(0,actions.length),transition_uniforms:transitionUniforms.slice(0,actions.length),probability};
}
export function repeatedVisit(){
  // New one-state task: A pays 1 then returns S or terminates (each 1/2);
  // B exits with 0. All policies terminate a.s.; no hidden time-limit state.
  const episode=sampleRepeatedEpisode([.1,.2,.3],[.8,.7,.2]);
  const result=updateWeightedControl([[0,0]],[[0,0]],episode,1);
  return {task:{gamma:1,A_reward:1,A_terminal_probability:.5,B_reward:0,B_terminal_probability:1,behavior_A_probability:.5,behavior_value:2/3,target_A_value:2},episode,returns:[3,2,1],first_visit_mean:3,every_visit_mean:2,incorrect_backward_seen_set:1,first_visit_weighted_action_value:3,first_visit_suffix_weight:4,every_visit_weighted_control:result};
}
export function mcOffpolicyControlData(){
  let q=[[0,0],[0,0]],c=[[0,0],[0,0]];
  const inputs=[[.5,.05],[.05,.95],[.05,.5]];
  const assigned=inputs.map((u,i)=>{
    const row={episode:i+1,...sampleEpisode(behavior(q),u),q_before:clone(q),c_before:clone(c),target_before:greedy(q),exact_before:exactValues(q)};
    const result=updateWeightedControl(q,c,row);q=result.q;c=result.c;
    return {...row,...result,behavior_after:behavior(q),exact_after:exactValues(q)};
  });
  return {kind:'exact_finite_offpolicy_control',task:clone(task),assigned_episodes:assigned,adaptive_enumerations:[adaptiveBranches(1),adaptiveBranches(2)],repeated_visit:repeatedVisit(),limitations:{selected_paths_are_not_expected_gain:true,no_random_training:true,cutoff_rejected:true,changing_target_not_fixed_policy_unbiasedness:true}};
}
