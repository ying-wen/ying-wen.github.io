/** Four trajectories, exact expectations, and fixed critic errors; no sampling. */
export function sigmoid(theta){
 if(!Number.isFinite(theta))throw new RangeError('theta must be finite');
 return theta>=0?1/(1+Math.exp(-theta)):Math.exp(theta)/(1+Math.exp(theta));
}
export function policyGradientData(theta=.7,gamma=.5){
 const p=sigmoid(theta),q=1-p;
 if(!(p>0&&p<1&&gamma>=0&&gamma<=1))throw new RangeError('nondegenerate policy and gamma in [0,1] required');
 const values=[gamma*p*q,0,q],localBaselines=[gamma*q*q,0,p];
 const trajectories=[];
 for(let a0=0;a0<2;a0++)for(let a1=0;a1<2;a1++){
  const reward=Number(a0===1&&a1===0),probability=(a0?p:q)*(a1?p:q);
  trajectories.push({actions:[a0,a1],probability,reward,returns:[gamma*reward,reward],scores:[a0-p,a1-p],state1:a0+1});
 }
 const sum=(f,rows=trajectories)=>rows.reduce((z,r)=>z+r.probability*f(r),0);
 const moments=baseline=>{
  const rows=trajectories.map(r=>{const g0=(r.returns[0]-baseline[0])*r.scores[0],g1=gamma*(r.returns[1]-baseline[r.state1])*r.scores[1];return {...r,g0,g1,gradient:g0+g1};});
  const expectation=f=>rows.reduce((z,r)=>z+r.probability*f(r),0),mean0=expectation(r=>r.g0),mean1=expectation(r=>r.g1);
  const variance0=expectation(r=>(r.g0-mean0)**2),variance1=expectation(r=>(r.g1-mean1)**2),covariance=expectation(r=>(r.g0-mean0)*(r.g1-mean1));
  return {baseline,mean:mean0+mean1,variance0,variance1,covariance,variance:expectation(r=>(r.gradient-mean0-mean1)**2),gradients:rows.map(r=>r.gradient)};
 };
 const localVariance=(state,b)=>{
  const time=state===0?0:1,rows=trajectories.filter(r=>state===0||r.state1===state),mass=rows.reduce((z,r)=>z+r.probability,0);
  const mean=sum(r=>(r.returns[time]-b)*r.scores[time],rows)/mass;
  return sum(r=>((r.returns[time]-b)*r.scores[time]-mean)**2,rows)/mass;
 };
 const states=[{name:'s0',mass:1,value:values[0],actionValues:[0,gamma*q]},
  {name:'s1(0)',mass:gamma*q,value:0,actionValues:[0,0]},
  {name:'s1(1)',mass:gamma*p,value:q,actionValues:[1,0]}];
 for(const s of states){s.localScore=q*(-p)*s.actionValues[0]+p*q*s.actionValues[1];s.contribution=s.mass*s.localScore;}
 const analytic=gamma*p*q*(1-2*p);
 const baselineResults=[['zero',[0,0,0]],['value',values],['local-optimum',localBaselines],['constant-3',[3,3,3]]].map(([name,b])=>({name,...moments(b)}));
 const tdMean=errors=>{
  const V=values.map((v,i)=>v+errors[i]);
  return sum(r=>(gamma*V[r.state1]-V[0])*r.scores[0]+gamma*(r.reward-V[r.state1])*r.scores[1]);
 };
 const criticResults=[[0,0,0],[7,0,0],[0,.3,.3],[0,0,.5]].map(errors=>({errors,tdMean:tdMean(errors),mcMean:moments(values.map((v,i)=>v+errors[i])).mean,bias:gamma*p*q*(errors[2]-errors[1])}));
 return {parameters:{theta,gamma,p,J:gamma*p*q,analytic},trajectories,states,
  occupancyGradient:states.reduce((z,s)=>z+s.contribution,0),Z:1+gamma,
  baselineResults,localBaselines,localVariances:states.map((s,i)=>({state:i,zero:localVariance(i,0),value:localVariance(i,values[i]),optimal:localVariance(i,localBaselines[i])})),
  criticResults,misuse:{missingOuterDiscount:sum(r=>r.returns[0]*r.scores[0]+r.returns[1]*r.scores[1]),sameSampleBaseline:sum(r=>(r.returns[0]-r.returns[0])*r.scores[0]+gamma*(r.returns[1]-r.returns[1])*r.scores[1])},
  baselineCurve:Array.from({length:81},(_,i)=>[i/80,localVariance(2,i/80)]),
  criticCurve:Array.from({length:81},(_,i)=>{const error=-.5+i/80;return [error,tdMean([0,0,error])];}),
  evidence:'Exact enumeration under a frozen policy; conditional and episode variances; no random training.'};
}
