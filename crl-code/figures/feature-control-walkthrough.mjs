/** Exact finite Sarsa walkthrough: actual weight writes and subsequent real actions.
 * Environment truth is used only for analysis; sarsaStep sees one observed transition.
 * No hash collisions, model queries by the learner, RNG, or performance curves.
 */
export const config=Object.freeze({gamma:1,alpha:.6,epsilon:0,episodes:2,dimension:3,tie:'continue'});
export const coordinates=Object.freeze({A:.3,B:.6});
export const task=Object.freeze({A:Object.freeze([{action:'continue',next:'B',reward:2},{action:'exit',next:'T',reward:0}]),B:Object.freeze([{action:'continue',next:'T',reward:-1}])});
export const dot=(x,y)=>x.reduce((s,v,i)=>s+v*y[i],0);
export const tileKeys=(s,h=.5,count=2)=>{
 if(!Number.isFinite(s)||!(h>0)||!Number.isInteger(count)||count<1)throw new RangeError('invalid tile configuration');
 return Array.from({length:count},(_,k)=>`${k}:${Math.floor(s/h+k/count)}`);
};
export function featureMap(name,scale=1){
 if(!(scale>0)||!Number.isFinite(scale))throw new RangeError('positive finite scale required');
 const r=1/Math.sqrt(2),maps={separate:{A:[1,0,0],B:[0,1,0]},tiles:{A:[r,0,r],B:[0,r,r]},aggregate:{A:[1,0,0],B:[1,0,0]}};
 if(!maps[name])throw new RangeError('unknown representation');
 return Object.fromEntries(Object.entries(maps[name]).map(([s,x])=>[s,x.map(v=>v*scale)]));
}
export const qValue=(w,phi,s,a=0)=>dot(w[a],phi[s]);
export const values=(w,phi)=>({A:[qValue(w,phi,'A'),qValue(w,phi,'A',1)],B:[qValue(w,phi,'B')]});
export function greedy(q){return q.indexOf(Math.max(...q));}
export function epsilonProbabilities(q,epsilon){
 if(epsilon<0||epsilon>1)throw new RangeError('epsilon must lie in [0,1]');
 const max=Math.max(...q),ties=q.map((v,i)=>v===max?i:-1).filter(i=>i>=0);
 return q.map((_,i)=>epsilon/q.length+(ties.includes(i)?(1-epsilon)/ties.length:0));
}
export function sarsaStep(w,phi,{state,action,reward,next,nextAction,terminated},alpha=config.alpha){
 const oldQ=qValue(w,phi,state,action);
 // A terminal outcome never asks for a terminal feature or next action.
 const bootstrap=terminated?0:qValue(w,phi,next,nextAction);
 const target=reward+config.gamma*bootstrap,delta=target-oldQ;
 const after=w.map(row=>[...row]);
 after[action]=after[action].map((v,i)=>v+alpha*delta*phi[state][i]);
 return {weights:after,old_q:oldQ,bootstrap,target,delta};
}
export function runCorridor(name,{scale=1,alpha=config.alpha,episodes=config.episodes}={}){
 if(!Number.isInteger(episodes)||episodes<1)throw new RangeError('positive episodes required');
 const phi=featureMap(name,scale);let w=[[0,0,0],[0,0,0]],events=[],first=null;
 for(let episode=1;episode<=episodes;episode++){
  let state='A',action=greedy(values(w,phi).A);
  while(state!=='T'){
   const before=values(w,phi),outcome=task[state][action],terminated=outcome.next==='T';
   // Choose and retain the next actual action before changing parameters.
   const nextAction=terminated?null:greedy(values(w,phi)[outcome.next]);
   const step=sarsaStep(w,phi,{state,action,reward:outcome.reward,next:outcome.next,nextAction,terminated},alpha);
   w=step.weights;
   events.push({episode,state,action,action_name:outcome.action,reward:outcome.reward,next:outcome.next,terminated,next_action:nextAction,before,old_q:step.old_q,bootstrap:step.bootstrap,target:step.target,delta:step.delta,after:values(w,phi)});
   state=outcome.next;action=nextAction;
  }
  if(episode===1)first=values(w,phi);
 }
 return {name,scale,alpha,features:phi,gram:[[dot(phi.A,phi.A),dot(phi.A,phi.B)],[dot(phi.B,phi.A),dot(phi.B,phi.B)]],events,after_first_episode:first,next_real_transition:events.find(e=>e.episode===2),final:values(w,phi)};
}
export function featureControlWalkthroughData(){
 const representations=Object.fromEntries(['separate','tiles','aggregate'].map(name=>[name,runCorridor(name)]));
 return {metadata:{kind:'exact',purpose:'finite on-policy control and representation/scale diagnostics',sampled_runs:0,source:'Sutton & Barto (2020), §§9.3,9.5.3–4,10.1; original course tilecoding.py'},config,coordinates,task,
  analysis_truth:{A:[1,0],B:[-1]},raw_tile_keys:{A:tileKeys(coordinates.A),B:tileKeys(coordinates.B)},representations,
  scales:{base:representations.tiles,uncompensated:runCorridor('tiles',{scale:10}),compensated:runCorridor('tiles',{scale:10,alpha:config.alpha/100})},
  epsilon_point_two:Object.fromEntries(Object.entries(representations).map(([name,r])=>[name,{probabilities:epsilonProbabilities(r.after_first_episode.A,.2),expected_first_reward:epsilonProbabilities(r.after_first_episode.A,.2)[0]*2}]))};
}
