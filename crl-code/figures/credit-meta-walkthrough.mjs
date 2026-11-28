/** A fixed, observed three-transition stream; no environment simulation. */
export const creditFixture = Object.freeze({features:[[1,0],[1,1],[1,0],[0,0]],rewards:[1,0,2],initial:[0,0],gamma:1,lambda:.5,alpha:.5});
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const add=(a,b,k=1)=>a.map((v,i)=>v+k*b[i]);
function inputs(options={}){
 const d={...creditFixture,...options};
 if(d.features.length!==d.rewards.length+1||!d.rewards.length||d.initial.length===0)throw new RangeError('one more feature than reward is required');
 if(d.features.some(x=>x.length!==d.initial.length)||[...d.features.flat(),...d.rewards,...d.initial,d.alpha,d.gamma,d.lambda].some(v=>!Number.isFinite(v)))throw new RangeError('finite, matching dimensions required');
 if(d.alpha<0||d.gamma<0||d.gamma>1||d.lambda<0||d.lambda>1)throw new RangeError('invalid alpha, gamma or lambda');
 return d;
}
export function tdPrefixes(options={},trueOnline=false){
 const d=inputs(options),history=[d.initial.slice()],rows=[];
 let w=d.initial.slice(),z=w.map(()=>0),oldPrediction=0;
 for(let k=0;k<d.rewards.length;k++){
  const x=d.features[k],next=d.features[k+1],v=dot(w,x),vNext=dot(w,next),delta=d.rewards[k]+d.gamma*vNext-v;
  const versionGap=v-oldPrediction,oldTrace=z.slice();
  z=add(z.map(v=>d.gamma*d.lambda*v),x,trueOnline?1-d.alpha*d.gamma*d.lambda*dot(oldTrace,x):1);
  const increment=trueOnline?add(z.map(v=>d.alpha*(delta+versionGap)*v),x,-d.alpha*versionGap):z.map(v=>d.alpha*delta*v);
  w=add(w,increment);history.push(w.slice());
  rows.push({t:k,delta,trace:z.slice(),prediction:v,nextPrediction:vNext,oldPrediction,versionGap,increment,weight:w.slice()});
  oldPrediction=vNext; // PRE-update prediction, including at repeated states.
 }
 return {history,rows};
}
export function onlineForward(options={}){
 const d=inputs(options),history=[d.initial.slice()],targets=[];
 for(let h=1;h<=d.rewards.length;h++){
  const g=new Array(h);let tail=dot(history[h-1],d.features[h]);
  // Independent of the Python n-step summation: recursively mix known returns.
  for(let k=h-1;k>=0;k--){
   tail=d.rewards[k]+d.gamma*((1-d.lambda)*dot(history[k],d.features[k+1])+d.lambda*tail);
   g[k]=tail;
  }
  let helper=d.initial.slice();
  for(let k=0;k<h;k++)helper=add(helper,d.features[k],d.alpha*(g[k]-dot(helper,d.features[k])));
  history.push(helper);targets.push(g);
 }
 return {history,targets};
}
/** Differentiate the actual accumulating-TD update, by reverse adjoints. */
export function metaGradient(options={},evaluationFeature=[1,1],target=2){
 const d=inputs(options),run=tdPrefixes(d),n=d.rewards.length;
 function reverse(endpoint,stopBootstrap=false){
  let adjoint=endpoint.slice();const paths=new Array(n);
  for(let k=n-1;k>=0;k--){
   const z=run.rows[k].trace,delta=run.rows[k].delta,q=dot(z,adjoint);
   paths[k]=delta*q;
   const errorGradient=d.features[k].map((x,j)=>(stopBootstrap?0:d.gamma*d.features[k+1][j])-x);
   adjoint=add(adjoint,errorGradient,d.alpha*q);
  }
  return paths;
 }
 const prediction=dot(run.history.at(-1),evaluationFeature),residual=prediction-target;
 const perStepPredictionDerivative=reverse(evaluationFeature),perStepGradient=perStepPredictionDerivative.map(v=>residual*v);
 const sensitivity=d.initial.map((_,j)=>reverse(d.initial.map((__,i)=>i===j?1:0)).reduce((a,b)=>a+b,0));
 const reusedResidual=dot(run.history.at(-1),d.features[n-1])-d.rewards[n-1];
 return {evaluationFeature,target,prediction,residual,loss:residual**2/2,sensitivity,
  full:perStepGradient.reduce((a,b)=>a+b,0),direct:perStepGradient.at(-1),
  stopBootstrap:residual*reverse(evaluationFeature,true).reduce((a,b)=>a+b,0),
  perStepPredictionDerivative,perStepGradient,
  reusedTrainLoss:reusedResidual**2/2,reusedTrainGradient:reusedResidual*dot(sensitivity,d.features[n-1])};
}
export function creditMetaData(){
 return {kind:'exact',sampled_runs:0,fixture:creditFixture,traditional:tdPrefixes(),trueOnline:tdPrefixes({},true),forward:onlineForward(),meta:metaGradient(),
  sensitivityHistory:[[0,0],...creditFixture.rewards.map((_,i)=>metaGradient({features:creditFixture.features.slice(0,i+2),rewards:creditFixture.rewards.slice(0,i+1)}).sensitivity)],
  lossCurve:Array.from({length:61},(_,i)=>{const alpha=.2+i*.01,m=metaGradient({alpha});return {alpha,loss:m.loss,reusedTrainLoss:m.reusedTrainLoss};})};
}
