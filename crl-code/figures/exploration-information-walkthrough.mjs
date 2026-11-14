/** A finite corridor with a hidden, fixed useful side.
 * Only environmentStep and the analysis enumerator receive theta. The controller
 * receives a posterior q=P(theta=R), a physical location, and remaining time.
 * All published probabilities are dyadic rationals, represented exactly in JS.
 * No random samples, model fitting, terminal reset, or empirical learning curve.
 */
export const config=Object.freeze({horizon:6,recovery:4,prior:.5,discount:1});
const probability=q=>{if(!Number.isFinite(q)||q<0||q>1)throw new RangeError('q must be in [0,1]');};
const count=(n,label)=>{if(!Number.isInteger(n)||n<0)throw new RangeError(`${label} must be a nonnegative integer`);};
export function posterior(q,channel,z){
 probability(q);
 if(![-1,1].includes(z))throw new RangeError('the observation must be -1 or +1');
 if(channel==='screen')return q;
 if(channel!=='sign')throw new RangeError('unknown observation channel');
 const evidence=z===1?q:1-q;
 if(evidence===0)throw new RangeError('zero-probability sign observation');
 return z===1?1:0;
}
export function predictionRisk(q,channel,mean){
 probability(q);
 if(!Number.isFinite(mean))throw new RangeError('finite prediction required');
 if(!['sign','screen'].includes(channel))throw new RangeError('unknown observation channel');
 const p=channel==='sign'?q:.5;
 return (1-p)*(-1-mean)**2+p*(1-mean)**2;
}
export function entropyBits(q){probability(q);return q===0||q===1?0:-q*Math.log2(q)-(1-q)*Math.log2(1-q);}
export function informationGain(q,channel){probability(q);if(channel==='screen')return 0;if(channel==='sign')return entropyBits(q);throw new RangeError('unknown observation channel');}

const cache=new Map();
/** Closed-form branch values after a route observation reveals the hidden side.
 * On a wrong route, that first zero-reward step is followed by k recovery steps.
 * k=0 is a different transition law: the wrong action itself returns to J.
 */
export function decision(h,q,k){
 count(h,'horizon');count(k,'recovery');probability(q);
 const key=`${h}/${q}/${k}`;if(cache.has(key))return cache.get(key);
 if(h===0)return {value:0,action:null,values:{}};
 const afterWrong=Math.max(0,h-1-k);
 const values={left:(1-q)*h+q*afterWrong,right:q*h+(1-q)*afterWrong,sign:h-1,screen:decision(h-1,q,k).value};
 // Ties prefer a direct action: L, then R, then sign, then screen.
 const action=Object.keys(values).reduce((best,a)=>values[a]>values[best]?a:best,'left');
 const result={value:values[action],action,values};cache.set(key,result);return result;
}
export function environmentStep(state,action,theta,z=-1,k=config.recovery){
 if(!['L','R'].includes(theta))throw new RangeError('theta must be L or R');
 count(k,'recovery');
 if(state.place==='G'){
  if(action!=='collect')throw new RangeError('G requires collect');
  return {state:{place:'G',remaining:0},reward:1,observation:null};
 }
 if(state.place==='D'){
  if(action!=='recover'||state.remaining<1)throw new RangeError('D requires a pending recovery step');
  return {state:{place:state.remaining===1?'J':'D',remaining:state.remaining-1},reward:0,observation:null};
 }
 if(state.place!=='J')throw new RangeError('unknown physical state');
 if(action==='sign'||action==='screen'){
  if(action==='screen'&&![-1,1].includes(z))throw new RangeError('invalid screen symbol');
  return {state:{place:'J',remaining:0},reward:0,observation:{channel:action,z:action==='sign'?(theta==='R'?1:-1):z}};
 }
 if(!['left','right'].includes(action))throw new RangeError('invalid J action');
 const good=(action==='right')===(theta==='R');
 return {state:{place:good?'G':k===0?'J':'D',remaining:good?0:k},reward:good?1:0,observation:{channel:'route',side:action==='right'?'R':'L',good}};
}
export function updateBelief(q,observation){
 if(!observation)return q;
 if(observation.channel==='route')return (observation.side==='R')===observation.good?1:0;
 return posterior(q,observation.channel,observation.z);
}
export function runHistory(theta,{horizon=config.horizon,recovery=config.recovery,prior=config.prior,firstAction=null,screenTape=[],disconnectSign=false}={}){
 count(horizon,'horizon');count(recovery,'recovery');probability(prior);
 let q=prior,controlQ=prior,state={place:'J',remaining:0},events=[];
 for(let t=0;t<horizon;t++){
  const h=horizon-t;
  let action=state.place==='G'?'collect':state.place==='D'?'recover':decision(h,controlQ,recovery).action;
  if(t===0&&firstAction)action=firstAction;
  // Disconnect only the sign-to-controller interface. The learner still writes
  // q correctly; the deliberately faulty consumer retains its old controlQ.
  if(disconnectSign&&t>0&&state.place==='J'&&action==='sign')action=controlQ>.5?'right':'left';
  const observed=environmentStep(state,action,theta,screenTape[t]??-1,recovery),nextQ=updateBelief(q,observed.observation);
  const nextControlQ=disconnectSign&&observed.observation?.channel==='sign'?controlQ:updateBelief(controlQ,observed.observation);
  events.push({step:t+1,from:state.place,action,reward:observed.reward,to:observed.state.place,recovery_remaining:observed.state.remaining,q_before:q,q_after:nextQ,control_q_before:controlQ,control_q_after:nextControlQ,observation:observed.observation});
  q=nextQ;controlQ=nextControlQ;state=observed.state;
 }
 return {theta,events,total:events.reduce((sum,e)=>sum+e.reward,0),final_state:state,final_q:q};
}
/** Enumerate every hidden world and every possible length-H screen tape. */
export function enumerateHistories(options={}){
 const h=options.horizon??config.horizon,q=options.prior??config.prior;
 count(h,'horizon');probability(q);if(h>12)throw new RangeError('this teaching enumerator is limited to 12 steps');
 let total=0,mass=0;const outcomes={};
 for(const theta of ['L','R'])for(let bits=0;bits<2**h;bits++){
  const weight=(theta==='R'?q:1-q)/2**h;
  if(weight===0)continue;
  const screenTape=Array.from({length:h},(_,i)=>(bits>>i)&1?1:-1);
  const run=runHistory(theta,{...options,screenTape});
  total+=weight*run.total;mass+=weight;outcomes[run.total]=(outcomes[run.total]??0)+weight;
 }
 return {value:total,mass,outcomes};
}
export function walkthroughData(){
 const h=config.horizon,k=config.recovery,q=config.prior;
 const choices=Object.fromEntries(['left','right','sign','screen'].map(firstAction=>[firstAction,enumerateHistories({firstAction})]));
 const matrix=[];
 for(let horizon=0;horizon<=8;horizon++)for(let recovery=0;recovery<=6;recovery++)for(const prior of [0,.25,.5,.75,1])matrix.push({horizon,recovery,prior,...decision(horizon,prior,recovery)});
 return {metadata:{kind:'exact',sampled_runs:0,clock:'primitive environment steps; H is an evaluation cutoff, not termination',unknown:'fixed theta L/R; prior and observation/transition laws given'},config,
  prediction:{before:{sign:predictionRisk(q,'sign',0),screen:predictionRisk(q,'screen',0)},after:{sign:0,screen:predictionRisk(q,'screen',0)},information_bits:{sign:informationGain(q,'sign'),screen:informationGain(q,'screen')},overfit_screen:{old_mean:0,new_mean:1,same_sample_before:1,same_sample_after:0,fresh_frame_before:predictionRisk(q,'screen',0),fresh_frame_after:predictionRisk(q,'screen',1)}},
  choices,decision:decision(h,q,k),trajectories:{sign_L:runHistory('L',{firstAction:'sign'}),sign_R:runHistory('R',{firstAction:'sign'}),direct_L:runHistory('L',{firstAction:'left'}),direct_R:runHistory('R',{firstAction:'left'}),screen_R:runHistory('R',{firstAction:'screen'})},
  contrasts:{zero_extra_recovery:decision(h,q,0),one_step_left:decision(1,q,k),known_world:decision(h,1,k),disconnected_consumer:enumerateHistories({firstAction:'sign',disconnectSign:true})},matrix};
}
