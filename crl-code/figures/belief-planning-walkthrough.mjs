/** Finite warehouse POMDP: a single optional scan, then terminal collection.
 * Known task model / deterministic enumeration; no neural training or sampling.
 */
export const locations=Object.freeze(['L','C','R']);
export const positions=Object.freeze([-1,0,1]);
export const task=Object.freeze({horizon:2,gamma:1,scanCost:.4,successReward:4,failureReward:-2,scanAccuracy:.8});
export function validateBelief(b){
 if(!Array.isArray(b)||b.length!==3||!b.every(x=>Number.isFinite(x)&&x>=0)||Math.abs(b.reduce((a,x)=>a+x,0)-1)>1e-12)throw new RangeError('three normalized nonnegative probabilities required');
 return b;
}
export function sensor(q=.8){
 if(!Number.isFinite(q)||q<0||q>1)throw new RangeError('sensor accuracy in [0,1] required');
 return locations.map((_,s)=>locations.map((_,o)=>s===o?q:(1-q)/2));
}
export function posterior(b,o,q=.8){
 validateBelief(b);if(!Number.isInteger(o)||o<0||o>2)throw new RangeError('observation index required');
 const O=sensor(q),joint=b.map((x,s)=>x*O[s][o]),evidence=joint.reduce((a,x)=>a+x,0);
 if(evidence===0)throw new RangeError('observation has zero evidence');
 return {evidence,belief:joint.map(x=>x/evidence)};
}
export function collectValues(b){return validateBelief(b).map(p=>task.failureReward+(task.successReward-task.failureReward)*p);}
export function terminal(b){const values=collectValues(b),value=Math.max(...values);return {action:values.indexOf(value),value,values};}
export function beliefMean(b){return validateBelief(b).reduce((x,p,i)=>x+p*positions[i],0);}
export function scanBackup(b,q=.8,cost=.4){
 validateBelief(b);if(!Number.isFinite(cost)||cost<0)throw new RangeError('nonnegative finite scan cost required');
 const O=sensor(q),branches=[];let gross=0;
 for(let o=0;o<3;o++){
  const evidence=b.reduce((sum,p,s)=>sum+p*O[s][o],0);
  if(evidence===0){branches.push({observation:o,evidence,belief:null,action:null,value:0,values:null});continue;}
  const updated=posterior(b,o,q),choice=terminal(updated.belief);
  branches.push({observation:o,...updated,...choice});gross+=evidence*choice.value;
 }
 return {q,cost,branches,gross,value:gross-cost};
}
/** Evaluate a conditional plan by summing the joint (hidden,observation) law. */
export function planValue(b,plan,q=.8,cost=.4){
 validateBelief(b);if(!Array.isArray(plan)||plan.length!==3||!plan.every(a=>Number.isInteger(a)&&a>=0&&a<3))throw new RangeError('one collection action per observation required');
 if(!Number.isFinite(cost)||cost<0)throw new RangeError('nonnegative finite scan cost required');
 const O=sensor(q);let value=-cost;
 for(let s=0;s<3;s++)for(let o=0;o<3;o++)value+=b[s]*O[s][o]*(plan[o]===s?4:-2);
 return value;
}
export function enumeratePlans(b,q=.8,cost=.4){
 const plans=[];for(let a=0;a<3;a++)for(let c=0;c<3;c++)for(let r=0;r<3;r++){const plan=[a,c,r];plans.push({plan,value:planValue(b,plan,q,cost)});}
 const value=Math.max(...plans.map(p=>p.value));return {count:plans.length,value,best:plans.filter(p=>Math.abs(p.value-value)<1e-12)};
}
export function fitSensor(counts){
 if(!Array.isArray(counts)||counts.length!==3||!counts.every(row=>Array.isArray(row)&&row.length===3&&row.every(x=>Number.isInteger(x)&&x>=0)))throw new RangeError('3 by 3 nonnegative integer counts required');
 const total=counts.flat().reduce((a,x)=>a+x,0);if(!total)throw new RangeError('empty calibration records');
 return {total,correct:counts.reduce((a,row,s)=>a+row[s],0),q:counts.reduce((a,row,s)=>a+row[s],0)/total};
}
export function walkthroughData(){
 const ambiguous=[.5,0,.5],centered=[0,1,0],calibrationCounts=[[8,1,1],[1,8,1],[1,1,8]],fit=fitSensor(calibrationCounts),scan=scanBackup(ambiguous,fit.q),oldScan=scanBackup(ambiguous,.4);
 return {task,locations,positions,observationMatrix:sensor(),ambiguous,centered,means:[beliefMean(ambiguous),beliefMean(centered)],direct:terminal(ambiguous),centeredDirect:terminal(centered),scan,oldScan,openLoop:[0,1,2].map(a=>({action:a,value:planValue(ambiguous,[a,a,a])})),enumeration:enumeratePlans(ambiguous),calibrationCounts,fit,control:{oldChosen:'collect-L',oldPredicted:1,oldActual:1,calibratedChosen:'scan',calibratedActual:scan.value,loss:scan.value-1},accuracyCurve:Array.from({length:21},(_,i)=>{const q=1/3+2*i/60;return {q,value:scanBackup(ambiguous,q).value};})};
}
