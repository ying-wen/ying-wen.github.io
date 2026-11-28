/** Exact, original teaching examples; none of these arrays is a sampled training run. */
export const sigmoid=x=>1/(1+Math.exp(-x));
export function banditSnapshot(){
 const q=[.2,.5,.8],estimates=[.6,.5,.4],epsilon=.1,greedy=0;
 const probability=estimates.map((_,a)=>epsilon/3+(a===greedy?1-epsilon:0));
 const selected=2,reward=1,alpha=.1,updated=[...estimates];
 updated[selected]+=alpha*(reward-updated[selected]);
 return {q,estimates,epsilon,greedy,probability,expectedReward:probability.reduce((v,p,a)=>v+p*q[a],0),sample:{selected,reward,alpha,updated},sampling:'given estimate snapshot; one possible reward; no random training'};
}
export function importanceAtoms(){
 const q=[.2,.8],behavior=[.5,.5],target=[.2,.8],ratio=target.map((p,a)=>p/behavior[a]);
 const atoms=q.flatMap((success,a)=>[1,0].map(reward=>({action:a,reward,behaviorMass:behavior[a]*(reward?success:1-success),targetMass:target[a]*(reward?success:1-success),ratio:ratio[a]})));
 return {q,behavior,target,ratio,atoms,behaviorValue:atoms.reduce((v,a)=>v+a.behaviorMass*a.reward,0),targetValue:atoms.reduce((v,a)=>v+a.targetMass*a.reward,0),sampling:'exact enumeration of probability mass, not a sample dataset'};
}
export function projectedValues(){
 const gamma=.5,features=[1,3],distribution=[.5,.5],truth=[2/3,4/3];
 const dot=(x,y)=>x.reduce((v,a,i)=>v+a*y[i],0),project=v=>dot(features,v)/dot(features,features),bellman=v=>[gamma*v[1],1+gamma*v[0]];
 const mcWeight=project(truth),tdWeight=3/7,mc=features.map(x=>x*mcWeight),td=features.map(x=>x*tdWeight),backup=bellman(mc),projected=features.map(x=>x*project(backup));
 return {gamma,features,distribution,truth,mcWeight,tdWeight,mc,td,backup,projected,normal:{C:5,A:3.5,b:1.5},sampling:'exact model and orthogonal projection; no training budget or seed'};
}
export function stabilityRecursions({alpha=.01,budget=300}={}){
 const features=[1,2],behavior=[.9,.1],gamma=.9;
 // Target action always moves to B. State weights stay d_b after action IS.
 const A=behavior.reduce((sum,p,s)=>sum+p*features[s]*(features[s]-gamma*features[1]),0);
 const onPolicyA=features[1]*(features[1]-gamma*features[1]),noBootstrapA=behavior.reduce((sum,p,s)=>sum+p*features[s]**2,0);
 const cases=[{id:'off-policy',A},{id:'on-policy',A:onPolicyA},{id:'no-bootstrap',A:noBootstrapA}].map(c=>({...c,factor:1-alpha*c.A,history:Array.from({length:budget+1},(_,k)=>({k,w:(1-alpha*c.A)**k}))}));
 return {features,behavior,gamma,alpha,budget,initial:1,cases,sampling:'deterministic frozen-law mean recursions, not conditional online weight expectations or sampled learning curves'};
}
export function twoStepPolicy({theta=.7,gamma=.5,alpha=5,baseline=0}={}){
 const p=sigmoid(theta),trajectories=[0,1].flatMap(first=>[0,1].map(second=>{
  const probability=(first?p:1-p)*(second?p:1-p),reward=first===1&&second===0?1:0;
  const gradient=(first-p)*(gamma*reward-baseline)+gamma*(second-p)*(reward-baseline);
  return {actions:[first,second],probability,reward,gradient:gradient===0?0:gradient};
 }));
 const objective=gamma*p*(1-p),gradient=trajectories.reduce((v,t)=>v+t.probability*t.gradient,0),nextTheta=theta+alpha*gradient,nextP=sigmoid(nextTheta);
 return {theta,gamma,alpha,baseline,p,trajectories,objective,gradient,nextTheta,nextP,nextObjective:gamma*nextP*(1-nextP),curve:Array.from({length:101},(_,i)=>({theta:i/100,J:gamma*sigmoid(i/100)*(1-sigmoid(i/100))})),sampling:'four trajectories enumerated exactly; one expected-gradient step, not stochastic training'};
}
export function frozenDQN(){
 const reward=1,gamma=.9,currentNext=[5,4],targetNext=[2,6],current=2;
 const selected=currentNext.indexOf(Math.max(...currentNext)),dqn=reward+gamma*Math.max(...targetNext),double=reward+gamma*targetNext[selected];
 const huberGradient=x=>Math.max(-1,Math.min(1,x));
 // One scalar output coordinate, frozen Double target, illustrational alpha=.1.
 const alpha=.1,copyEvery=3,history=[{k:0,online:current,target:current}];let online=current,target=current;
 for(let k=1;k<=6;k++){online-=alpha*huberGradient(online-double);if(k%copyEvery===0)target=online;history.push({k,online,target});}
 return {reward,gamma,currentNext,targetNext,current,selected,dqn,double,terminalTarget:reward,dqnHuberGradient:huberGradient(current-dqn),doubleHuberGradient:huberGradient(current-double),alpha,copyEvery,history,sampling:'prescribed network outputs and scalar frozen-label calculation; not learned maze values or a DQN training run'};
}
export const computeClassicMechanisms=()=>({bandit:banditSnapshot(),importance:importanceAtoms(),projection:projectedValues(),stability:stabilityRecursions(),policy:twoStepPolicy(),dqn:frozenDQN()});
