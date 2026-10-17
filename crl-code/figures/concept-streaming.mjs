/** Original, deterministic protocol examples; no benchmark measurements. */
export const dot=(a,b)=>a.reduce((sum,x,i)=>sum+x*b[i],0);
export function tdStep(w,z,{x,next,reward,gammaCurrent=.9,gammaNext=.9},{alpha=.1,lambda=.8}={}){
 const delta=reward+gammaNext*dot(w,next)-dot(w,x);
 const trace=z.map((v,i)=>gammaCurrent*lambda*v+x[i]);
 const increment=trace.map(v=>alpha*delta*v);
 return {delta,trace,increment,weights:w.map((v,i)=>v+increment[i])};
}
export function protocolExample(shared=false){
 const transitions=[{x:[1,0],next:shared?[1,1]:[0,1],reward:0},{x:shared?[1,1]:[0,1],next:[0,0],reward:1,gammaNext:0}];
 const run=lambda=>{let w=[0,0],z=[0,0];const rows=[];for(const t of transitions){const r=tdStep(w,z,t,{lambda});rows.push(r);w=r.weights;z=r.trace;}return {rows,weights:w,afterResetTrace:[0,0],updates:2,retainedTransitions:0};};
 const traces=run(.8),td0=run(0);
 const replayStep=tdStep(td0.weights,[0,0],transitions[0],{lambda:0});
 const gradients=transitions.map(t=>tdStep([0,0],[0,0],t,{lambda:0}).increment);
 const batchWeights=[0,1].map(i=>gradients.reduce((sum,g)=>sum+g[i],0)/transitions.length);
 return {transitions,traces,td0,replay:{...replayStep,updates:3,retainedTransitions:2},batch:{weights:batchWeights,updates:1,peakRetainedTransitions:2}};
}
export function schedule({arrivals=[20,40,60,80],period=20,duration=12,updatesPerArrival=1}={}){
 let free=0,k=0;
 return arrivals.map((arrival,t)=>{
  const start=Math.max(arrival,free),updates=[];
  for(let j=0;j<updatesPerArrival;j++){updates.push({k:++k,start:start+j*duration,end:start+(j+1)*duration});}
  free=start+updatesPerArrival*duration;
  return {t,arrival,deadline:arrival+period,start,end:free,miss:free>arrival+period,updates};
 });
}
export function computeStreamingData(){return {protocol:protocolExample(),shared:protocolExample(true),clocks:{strict:schedule(),repeat:schedule({updatesPerArrival:2}),batch:{arrival:60,start:60,end:72,updates:1,samples:3}}};}
