/** Finite corridor mechanism kernel. No network or benchmark training. */
export const config=Object.freeze({length:10,horizon:10,episodes:3,prior:.5,safe:.25,low:.2,high:.8,gamma:1});
const probability=p=>{if(!Number.isFinite(p)||p<0||p>1)throw new RangeError('probability must be in [0,1]');return p;};
const positiveInt=(n,name)=>{if(!Number.isInteger(n)||n<1)throw new RangeError(name+' must be a positive integer');return n;};
const modelProbability=theta=>{if(!['low','high'].includes(theta))throw new RangeError('unknown model');return config[theta];};
export function posterior(q,y){
 probability(q);if(![0,1].includes(y))throw new RangeError('endpoint observation must be binary');
 const high=y?config.high:1-config.high,low=y?config.low:1-config.low;
 const evidence=q*high+(1-q)*low;
 return {q:q*high/evidence,evidence,predictive:config.low+(config.high-config.low)*q};
}
export const predictive=q=>config.low+(config.high-config.low)*probability(q);

/** Independent L/R geometry diagnostic, with an absorbing right endpoint. */
export function directionDP(length=10,horizon=length){
 positiveInt(length,'length');positiveInt(horizon,'horizon');
 let mass=Array(length+1).fill(0);mass[0]=1;const occupancy=[mass.slice()];
 for(let t=0;t<horizon;t++){
  const next=Array(length+1).fill(0);
  for(let x=0;x<=length;x++)if(x===length)next[x]+=mass[x];else{
   next[Math.max(0,x-1)]+=mass[x]/2;next[x+1]+=mass[x]/2;
  }
  mass=next;occupancy.push(mass.slice());
 }
 return {reach:mass[length],occupancy};
}
export function enumerateDirections(length=10,horizon=length){
 positiveInt(length,'length');positiveInt(horizon,'horizon');
 if(horizon>16)throw new RangeError('finite enumeration supports at most 16 actions');
 let reaches=0;
 for(let bits=0;bits<2**horizon;bits++){
  let x=0;for(let t=0;t<horizon&&x<length;t++)x=(bits>>t)&1?x+1:Math.max(0,x-1);
  if(x===length)reaches++;
 }
 return {reach:reaches/2**horizon,paths:2**horizon,reaching_paths:reaches};
}

/** Finite-horizon optimal policy in one supplied candidate model, not true theta. */
export function candidatePlan(theta,length=10,horizon=length,safe=config.safe){
 const p=modelProbability(theta);positiveInt(length,'length');positiveInt(horizon,'horizon');
 if(!Number.isFinite(safe)||safe<0)throw new RangeError('safe reward must be nonnegative');
 const values=[Array(length).fill(0)],policies=[Array(length).fill('cutoff')],qs=[[]];
 for(let h=1;h<=horizon;h++){
  const v=[],policy=[],row=[];
  for(let x=0;x<length;x++){
   const q={safe,left:values[h-1][Math.max(0,x-1)],right:x+1===length?p:values[h-1][x+1]};
   let best='safe';for(const a of ['left','right'])if(q[a]>q[best]+1e-12)best=a;
   v.push(q[best]);policy.push(best);row.push(q);
  }
  values.push(v);policies.push(policy);qs.push(row);
 }
 return {values,policies,qs};
}

/** Simulated actions cause observations; theta is only the environment's input. */
export function runEpisode(theta,q,{mode='held',uTape=[.4],rewardQuantile=.4,length=10,horizon=length}={}){
 modelProbability(theta);probability(q);probability(rewardQuantile);
 if(!['held','step'].includes(mode))throw new RangeError('unknown sampling clock');
 if(!Array.isArray(uTape)||!uTape.length)throw new RangeError('a nonempty quantile tape is required');
 uTape.forEach(probability);
 const plans={low:candidatePlan('low',length,horizon),high:candidatePlan('high',length,horizon)};
 const before=q,held=uTape[0]<q?'high':'low';let x=0,total=0,end='cutoff';const records=[];
 for(let t=0;t<horizon;t++){
  const sampled=mode==='held'?held:(uTape[t%uTape.length]<q?'high':'low');
  const remaining=horizon-t,action=plans[sampled].policies[remaining][x];
  const next=action==='right'?x+1:action==='left'?Math.max(0,x-1):x;
  const terminal=action==='safe'||next===length;
  const observation=next===length?Number(rewardQuantile<config[theta]):null;
  const reward=action==='safe'?config.safe:observation??0;
  const after=observation===null?q:posterior(q,observation).q;
  records.push({t:t+1,x,remaining,sampled,action,next,reward,terminal,endpoint_observation:observation,q_before:q,q_after:after});
  x=next;total+=reward;q=after;
  if(terminal){end=action==='safe'?'safe':'endpoint';break;}
 }
 return {mode,q_before:before,q_after:q,total,end,primitive_actions:records.length,records};
}
export function runTrials(theta,{mode='held',uTapes=[[.4],[.4],[.1]],rewardQuantiles=[.4,.4,.4],prior=config.prior}={}){
 if(uTapes.length!==rewardQuantiles.length)throw new RangeError('trial tapes must have equal length');
 let q=prior;const episodes=[];
 uTapes.forEach((uTape,i)=>{const run=runEpisode(theta,q,{mode,uTape,rewardQuantile:rewardQuantiles[i]});q=run.q_after;episodes.push(run);});
 return {episodes,q_final:q,total:episodes.reduce((sum,run)=>sum+run.total,0),endpoint_visits:episodes.filter(run=>run.end==='endpoint').length};
}

/** Exact branch recursion for repeated trials; ordinary zeroes never enter Bayes. */
export function trialDP(episodes,q=.5,theta='low',mode='held',length=10){
 if(!Number.isInteger(episodes)||episodes<0)throw new RangeError('episodes must be nonnegative');
 probability(q);positiveInt(length,'length');const p=modelProbability(theta);
 if(!['held','step'].includes(mode))throw new RangeError('unknown sampling clock');
 if(episodes===0)return {value:0,endpoint_visits:0,primitive_actions:0};
 const reach=mode==='held'?q:q**length,branches=[{mass:1-reach,reward:config.safe,q,visits:0},{mass:reach*p,reward:1,q:posterior(q,1).q,visits:1},{mass:reach*(1-p),reward:0,q:posterior(q,0).q,visits:1}];
 let value=0,visits=0,actions=0;
 for(const branch of branches){const tail=trialDP(episodes-1,branch.q,theta,mode,length);value+=branch.mass*(branch.reward+tail.value);visits+=branch.mass*(branch.visits+tail.endpoint_visits);}
 // step sampling stops at the first low model; success uses all length actions.
 const currentActions=mode==='held'?(1-q)+length*q:Array.from({length},(_,i)=>q**i).reduce((a,b)=>a+b,0);
 for(const branch of branches)actions+=branch.mass*trialDP(episodes-1,branch.q,theta,mode,length).primitive_actions;
 return {value,endpoint_visits:visits,primitive_actions:actions+currentActions};
}

/** A table probe separates data masking, TD bootstrapping and fixed prior. */
export function headTrainingProbe(observedTerminalReward=0){
 if(![0,1].includes(observedTerminalReward))throw new RangeError('terminal reward must be binary');
 const alpha=.5,mask=[1,0],priors=[.3,-.1],residualBefore=[.5,.3];
 const qBefore=residualBefore.map((f,k)=>f+priors[k]);
 const residualAfter=residualBefore.map((f,k)=>f+alpha*mask[k]*(observedTerminalReward-qBefore[k]));
 const ownTargets=[.8,.2],nonterminalBefore=[.5,.5];
 return {alpha,mask,terminal_reward:observedTerminalReward,terminal_target:[observedTerminalReward,observedTerminalReward],prior_before:priors,prior_after:priors.slice(),residual_before:residualBefore,residual_after:residualAfter,q_before:qBefore,q_after:residualAfter.map((f,k)=>f+priors[k]),
  own_target_values:ownTargets,nonterminal_before:nonterminalBefore,nonterminal_after:nonterminalBefore.map((v,k)=>v+alpha*(ownTargets[k]-v)),neural_training_steps:0};
}
export function deepExplorationData(){
 const failure=runTrials('low'),success=runTrials('high',{uTapes:[[.4],[.4]],rewardQuantiles:[.1,.9]});
 const curves=Array.from({length:10},(_,i)=>({length:i+1,step_reach:directionDP(i+1).reach,held_reach:.5,model_step_reach:.5**(i+1)}));
 const exact=Object.fromEntries(['low','high'].map(theta=>[theta,Object.fromEntries(['held','step'].map(mode=>[mode,trialDP(3,.5,theta,mode)]))]));
 const known=runEpisode('high',1,{rewardQuantile:.9});
 return {metadata:{kind:'exact',clock:'primitive actions within three episodes, at most ten decisions per episode',unknown:'fixed endpoint model low/high; all transitions, candidate likelihoods and safe reward supplied',actor_observation:'position, remaining action budget, arrived reward and saved q; true theta unavailable',resets:'first episode initialized at x=0; at most two resets restore position and budget only; theta and belief persist; no simulator queries by actor',random_sampled_runs:0,neural_training_steps:0},config,curves,directions:{dp:directionDP(),enumeration:enumerateDirections()},plans:{low:candidatePlan('low'),high:candidatePlan('high')},closed_loop:{failure,success},exact,
  noise_control:{known_high:known,known_high_failure_probability:1-config.high,predictive_variance_known:config.high*(1-config.high),parameter_variance_known:0},training_probe:headTrainingProbe(failure.episodes[0].records.at(-1).reward)};
}
