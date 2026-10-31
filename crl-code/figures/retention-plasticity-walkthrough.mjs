/** Fixed-data diagnostic, not a deep-RL training experiment.
 * Two-layer linear network f(x)=v*u*x, nonlinear in its two parameters.
 * Training/evaluation use the complete deterministic support x in {-1,+1}.
 * There is no RNG, replay, optimizer momentum, or hidden state.
 */
export const fixture=Object.freeze({alpha:1/4,steps:8,inputs:[-1,1],oldTarget:1,newTarget:-1});
export const loss=([u,v],target)=>(u*v-target)**2/2;
export const gradient=([u,v],target)=>[(u*v-target)*v,(u*v-target)*u];
export function step(weights,target,alpha=fixture.alpha){
 const g=gradient(weights,target);
 // Both coordinates use the same pre-update parameter version.
 return weights.map((w,i)=>w-alpha*g[i]);
}
export function trajectory(initial,target,steps=fixture.steps,alpha=fixture.alpha){
 if(!Number.isInteger(steps)||steps<0||!Number.isFinite(alpha)||alpha<0)throw new RangeError('Invalid update budget');
 if(initial.length!==2||!initial.every(Number.isFinite)||!Number.isFinite(target))throw new RangeError('Invalid predictor');
 let weights=[...initial];const rows=[];
 for(let k=0;k<=steps;k++){
  rows.push({k,weights:[...weights],product:weights[0]*weights[1],oldLoss:loss(weights,1),newLoss:loss(weights,-1),targetLoss:loss(weights,target)});
  if(k<steps)weights=step(weights,target,alpha);
 }
 return rows;
}
export function walkthroughData(){
 const starts={balanced:[1,1],reparameterized:[2,1/2],fresh:[1,0]};
 const branches={};
 for(const [name,initial] of Object.entries(starts)){
  const adaptation=trajectory(initial,-1);
  // Probe branches never write into the original B checkpoint.
  const checkpoint=[...adaptation.at(-1).weights];
  const returnProbe=trajectory(checkpoint,1);
  branches[name]={initial:[...initial],adaptation,returnProbe,
   newImprovement:adaptation[0].newLoss-adaptation.at(-1).newLoss,
   oldReadOnly:loss(checkpoint,1),returnImprovement:returnProbe[0].oldLoss-returnProbe.at(-1).oldLoss};
 }
 return {evidence:'exact-deterministic-iteration',sampledRuns:0,...fixture,
  protocol:{trainingExamplesPerUpdate:2,newUpdates:8,returnProbeUpdates:8,readOnlyEvaluationsPerTrace:9,
   evaluation:'complete two-point support, no independent generalization estimate',
   fresh:'fixed reference (1,0), not a random initialization ensemble',
   state:'two parameters only; all probes copied; no optimizer state or RNG'},
  branches,freshOldProbe:trajectory([1,0],1)};
}
