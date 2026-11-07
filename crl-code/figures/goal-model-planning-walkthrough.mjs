/** Deterministic teaching fixture. The planner receives observations, never truth. */
export const gamma = 0.9;
export const actions = Object.freeze({S:['upper','lower'],A:['cross'],B:['forward'],C:['forward'],X:[],Y:[]});
export const states = Object.keys(actions);
export const key = (s,a)=>`${s}:${a}`;
// Two previously observed episodes, S-A-X and S-B-C-X; one reset between them.
export const history = Object.freeze([
 {s:'S',a:'upper',next:'A',physicalDone:false},
 {s:'A',a:'cross',next:'X',physicalDone:true},
 {s:'S',a:'lower',next:'B',physicalDone:false},
 {s:'B',a:'forward',next:'C',physicalDone:false},
 {s:'C',a:'forward',next:'X',physicalDone:true},
]);
// After a second reset, actually traverse S-A-Y. Only the last row changes the model.
export const newHistory = Object.freeze([
 {s:'S',a:'upper',next:'A',physicalDone:false},
 {s:'A',a:'cross',next:'Y',physicalDone:true},
]);
export function learnModel(records,initial={}) {
 const model=structuredClone(initial);
 for(const row of records) {
  if(!actions[row.s]?.includes(row.a)||!states.includes(row.next)||typeof row.physicalDone!=='boolean') throw new Error('Invalid observation');
  model[key(row.s,row.a)]={next:row.next,physicalDone:row.physicalDone};
 }
 return model;
}
export function label(row,goal) {
 if(!states.includes(goal))throw new Error('Unknown goal');
 if(row.s===goal||actions[row.s]?.length===0)throw new Error('No action after task termination');
 return {reward:Number(row.next===goal),done:row.physicalDone||row.next===goal};
}
export const zeroQ=()=>Object.fromEntries(states.flatMap(s=>actions[s].map(a=>[key(s,a),0])));
export function value(q,s,goal){return s===goal||!actions[s].length?0:Math.max(...actions[s].map(a=>q[key(s,a)]));}
export function sweep(model,q,goal) {
 const next={...q};let calls=0;
 for(const s of states)for(const a of actions[s]) {
  if(s===goal){next[key(s,a)]=0;continue;}
  const outcome=model[key(s,a)];if(!outcome)throw new Error(`Unobserved pair ${key(s,a)}`);
  calls++;
  const l=label({s,a,...outcome},goal);
  next[key(s,a)]=l.reward+gamma*Number(!l.done)*value(q,outcome.next,goal);
 }
 return {q:next,calls};
}
export function trace(model,goal,rounds=4,initial=zeroQ()) {
 if(!Number.isInteger(rounds)||rounds<0)throw new Error('Invalid planning budget');
 let q={...initial},calls=0;const rows=[];
 for(let k=0;k<=rounds;k++) {
  rows.push({k,calls,q:{...q},values:Object.fromEntries(states.map(s=>[s,value(q,s,goal)])),
   choice:q['S:upper']>q['S:lower']?'upper':'lower'});
  if(k<rounds){const r=sweep(model,q,goal);q=r.q;calls+=r.calls;}
 }
 return rows;
}
// Read-only evaluator. Current truth is available here only, not to sweep/learnModel.
export function evaluateFirstAction(first,closed,goal='X') {
 const route=first==='upper'?['S','A',closed?'Y':'X']:['S','B','C','X'];
 let result=0,steps=0;
 for(let i=1;i<route.length;i++){steps++;if(route[i]===goal){result=gamma**(i-1);break;}}
 return {return:result,steps,route:route.slice(0,steps+1)};
}
export function walkthroughData() {
 const old=learnModel(history),updated=learnModel(newHistory,old);
 const x=trace(old,'X'),a=trace(old,'A');
 const repaired=trace(updated,'X',3,x.at(-1).q);
 const stale=trace(old,'X',3,x.at(-1).q);
 const diagnostic=t=>t.map(row=>({...row,currentReturn:evaluateFirstAction(row.choice,true).return}));
 return {evidence:'exact-deterministic-model-and-backups',sampledRuns:0,gamma,actions,history,newHistory,
  protocol:{oldEnvironmentSteps:5,newEnvironmentSteps:2,totalEnvironmentSteps:7,resets:2,
   informativeChangedRows:1,callsPerXSweep:5,callsPerASweep:4,tieBreak:'lower',
   knowledge:'Known state labels and action sets; transition model receives observed rows only. Current truth is evaluator-only.',
   diagnostic:'Exact route evaluation, no independent sampled evaluation episodes.'},
  oldModel:old,updatedModel:updated,goals:{X:x,A:a},
  labels:{X:label(history[0],'X'),A:label(history[0],'A')},
  afterChange:{stale:diagnostic(stale),repaired:diagnostic(repaired)},
  currentTransitionL1Mean:{stale:2/5,repaired:0},
  physicalRoutes:{upper:history.slice(0,2),lower:history.slice(2)}};
}
