/** Exact source-defined toy calculations, separated from their SVG layouts. */

export function trueOnlineScalar(features,rewards,{alpha=.5,gamma=1,lambda=.5,initial=0}={}){
  let w=initial,e=0,oldValue=0;const steps=[],history=[w];
  for(let t=0;t<rewards.length;t++){
    const x=features[t],xp=features[t+1],v=w*x,vp=w*xp;
    const delta=rewards[t]+gamma*vp-v,oldTrace=e;
    const carry=gamma*lambda*oldTrace,add=(1-alpha*carry*x)*x;
    e=carry+add;
    const td=alpha*delta*e,correction=alpha*(v-oldValue)*(e-x);
    const before=w;w+=td+correction;
    steps.push({t,x,xp,reward:rewards[t],before,v,vp,oldValue,oldTrace,carry,add,e,delta,td,correction,after:w});
    oldValue=vp;history.push(w);
  }return {steps,history};
}

export function creditKnowledgeData(){
  const tree={gamma:.9,reward:1,probabilities:[.25,.75],q:[2,4],sampled:0,deepReturn:6};
  tree.expectedOld=tree.probabilities.reduce((s,p,i)=>s+p*tree.q[i],0);
  tree.replacement=tree.probabilities[0]*(tree.deepReturn-tree.q[0]);
  tree.leaves=[tree.probabilities[0]*tree.deepReturn,tree.probabilities[1]*tree.q[1]];
  tree.target=tree.reward+tree.gamma*tree.leaves.reduce((a,b)=>a+b,0);
  tree.sarsa=tree.reward+tree.gamma*tree.deepReturn;
  const trace={alpha:.5,gamma:1,lambda:.5,features:[1,1,0],rewards:[1,2]};
  Object.assign(trace,trueOnlineScalar(trace.features,trace.rewards,trace));
  trace.dutchOnly=trace.steps[1].before+trace.steps[1].td;
  let accumulatingWeight=0,accumulatingTrace=0;
  for(let t=0;t<trace.rewards.length;t++){
    const x=trace.features[t],xp=trace.features[t+1];
    const delta=trace.rewards[t]+trace.gamma*accumulatingWeight*xp-accumulatingWeight*x;
    accumulatingTrace=trace.gamma*trace.lambda*accumulatingTrace+x;
    accumulatingWeight+=trace.alpha*delta*accumulatingTrace;
  }
  trace.accumulating=accumulatingWeight;
  const emphasis={states:['A','B','C'],gamma:[0,.5,.5],interest:[1,0,0],rho:[2,4,1],cumulant:[0,1,0],lambda:0,rows:[]};
  let F=0;
  emphasis.states.forEach((s,t)=>{
    const previousRatio=t?emphasis.rho[t-1]:0,carry=emphasis.gamma[t]*previousRatio*F;
    F=emphasis.interest[t]+carry;
    emphasis.rows.push({state:s,t,previousRatio,carry,F,M:F,currentRatio:emphasis.rho[t],traceCoefficient:emphasis.rho[t]*F});
  });
  const option={rate:1,values:{G:4,F:0},versions:[{version:1,path:[[0,0],[1,0],[2,0]],endpoint:'G'},{version:2,path:[[0,0],[0,1],[1,1],[2,1]],endpoint:'F'}]};
  option.versions=option.versions.map(v=>{
    const duration=v.path.length-1,rewards=Array.from({length:duration},(_,i)=>-1+(i===duration-1&&v.endpoint==='G'?5:0)),reward=rewards.reduce((a,b)=>a+b,0),tail=option.values[v.endpoint];
    return {...v,rewards,duration,reward,tail,backup:reward-option.rate*duration+tail};
  });
  option.staleError=option.versions[0].backup-option.versions[1].backup;
  return {kind:'exact',tree,trace,emphasis,option};
}
