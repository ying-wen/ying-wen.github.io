/** Exact finite-deadline parcel task; no sampled training or learned model. */
export const CONFIG = Object.freeze({gamma:.9,horizon:3,bonus:3,startPotential:1,nearPotential:3});
export const MODES = Object.freeze(['true','proxy','shaping','bad-boundary','bad-discount']);
export const PATHS = Object.freeze([
 {id:'early',label:'第2步送达',actions:['approach','deliver']},
 {id:'late',label:'第3步送达',actions:['approach','wait','deliver']},
 {id:'timeout',label:'门口等待至超时',actions:['approach','wait','wait']},
]);
export const terminal=(s,c=CONFIG)=>s.x==='G'||s.t>=c.horizon;
export function potential(s,c=CONFIG,keepTimeout=false){
 if(s.x==='G'||(s.t>=c.horizon&&!keepTimeout))return 0;
 return s.x==='S'?c.startPotential:c.nearPotential;
}
export function transitions(s,c=CONFIG){
 if(terminal(s,c))return [];
 return (s.x==='S'?['approach']:['deliver','wait']).map(a=>{
  const next={t:s.t+1,x:a==='deliver'?'G':'A'};
  return {s:{...s},a,next,reward:Number(a==='deliver'),done:terminal(next,c)};
 });
}
export function trainingReward(row,mode,c=CONFIG){
 if(!MODES.includes(mode))throw new RangeError('Unknown reward mode');
 if(mode==='true')return row.reward;
 if(mode==='proxy')return row.reward+(row.next.x==='A'?c.bonus:0);
 const factor=mode==='bad-discount'?1:c.gamma;
 return row.reward+factor*potential(row.next,c,mode==='bad-boundary')-potential(row.s,c);
}
/** Backward induction on the clock-augmented state (t,x). */
export function solve(mode,c=CONFIG){
 const states={};
 function visit(s){
  if(terminal(s,c))return 0;
  const key=`${s.t}:${s.x}`;
  if(states[key])return states[key].value;
  const q=transitions(s,c).map(row=>({action:row.a,value:trainingReward(row,mode,c)+c.gamma*visit(row.next)}));
  const best=q.reduce((a,b)=>b.value>a.value+1e-12?b:a);
  states[key]={value:best.value,choice:best.action,q};return best.value;
 }
 const value=visit({t:0,x:'S'});return {mode,value,states};
}
export function execute(actions,mode,c=CONFIG){
 let s={t:0,x:'S'},trainReturn=0,trueReturn=0;const trace=[];
 for(const action of actions){
  const row=transitions(s,c).find(r=>r.a===action);
  if(!row)throw new RangeError('Action unavailable or task already ended');
  const training=trainingReward(row,mode,c),weight=c.gamma**s.t;
  trainReturn+=weight*training;trueReturn+=weight*row.reward;
  trace.push({...row,training,weight,cumulativeTraining:trainReturn,cumulativeTrue:trueReturn});s=row.next;
 }
 if(!terminal(s,c))throw new RangeError('Path must reach delivery or the specified deadline');
 return {actions:[...actions],trace,trainReturn,trueReturn,delivered:s.x==='G',steps:actions.length,end:s};
}
export function rollout(solution,c=CONFIG){
 let s={t:0,x:'S'};const actions=[];
 while(!terminal(s,c)){
  const a=solution.states[`${s.t}:${s.x}`].choice;
  actions.push(a);s=transitions(s,c).find(row=>row.a===a).next;
 }
 return execute(actions,solution.mode,c);
}
export function walkthroughData(c=CONFIG){
 const results=MODES.map(mode=>{
  const solution=solve(mode,c),run=rollout(solution,c);
  const paths=PATHS.map(p=>({...p,...execute(p.actions,mode,c)}));
  return {mode,solution,paths,selected:paths.find(p=>p.actions.join()===run.actions.join()).id,run};
 });
 return {schema:'reward-design-walkthrough-v1',evidence:'exact finite model calculation',config:{...c},
  objective:'Discounted delivered-parcel reward; genuine deadline after three actions; state is (t,x).',
  information:'Planner knows all transitions and specified rewards; no reward learning, random sampling, or training run.',
  budget:'Each displayed policy has one complete episode: 2 or 3 real actions. Planning is exhaustive; no learning-speed claim.',
  results};
}
