/** Exact, prescribed one-step calculation. No sampling or training run. */
const finite=(x,name)=>{if(!Number.isFinite(x))throw new TypeError(`${name} must be finite`);return x;};
export function qValues(theta,x){
  if(!Array.isArray(theta)||theta.length!==2)throw new TypeError('theta = [w, v]');
  const [w,v]=theta.map((n,i)=>finite(n,`theta[${i}]`));finite(x,'x');
  const h=Math.max(0,w*x),left=v*h;
  return [left,3-left];
}
export function qGradient(theta,x,action){
  qValues(theta,x);if(action!==0&&action!==1)throw new RangeError('action = 0 or 1');
  const [w,v]=theta,sign=action===0?1:-1;
  // The derivative convention at ReLU(0) is zero, as in the companion code.
  return w*x>0?[sign*v*x,sign*w*x]:[0,0];
}
export const greedy=values=>values[0]>=values[1]?0:1;
export function bootstrapTarget(online,target,experience,{gamma=.5,double=false}={}){
  if(!(gamma>=0&&gamma<=1))throw new RangeError('gamma in [0,1]');
  const {reward,nextX,terminated}=experience;finite(reward,'reward');
  if(typeof terminated!=='boolean')throw new TypeError('terminated must be boolean');
  if(terminated)return reward;
  const targetValues=qValues(target,nextX);
  const selected=greedy(qValues(double?online:target,nextX));
  return reward+gamma*targetValues[selected];
}
export function frozenTargetStep(theta,experience,label,alpha=.5){
  finite(label,'label');if(!(alpha>=0&&Number.isFinite(alpha)))throw new RangeError('alpha >= 0');
  const prediction=qValues(theta,experience.x)[experience.action];
  const error=prediction-label,gradient=qGradient(theta,experience.x,experience.action).map(g=>error*g);
  return {prediction,label,error,gradient,theta:theta.map((v,i)=>v-alpha*gradient[i])};
}
export function sharedFeatureWalkthrough(){
  const experience={x:1,action:0,reward:0,nextX:2,terminated:false};
  const theta0=[.5,1],target0=[...theta0],gamma=.5,alpha=.5;
  const label=bootstrapTarget(theta0,target0,experience,{gamma});
  const update=frozenTargetStep(theta0,experience,label,alpha),theta1=update.theta;
  const snapshot=(phase,online,target)=>({phase,online:[...online],target:[...target],
    current:qValues(online,experience.x),successor:qValues(online,experience.nextX),
    targetSuccessor:qValues(target,experience.nextX),selected:greedy(qValues(online,experience.nextX)),
    dqn:bootstrapTarget(online,target,experience,{gamma}),
    double:bootstrapTarget(online,target,experience,{gamma,double:true})});
  const semiGradient=[...update.gradient];
  const qGrad=qGradient(theta0,1,0),tailGrad=qGradient(theta0,2,1);
  return {kind:'exact',scope:'One prescribed SGD update; no environment rollout or learned performance.',
    experience,gamma,alpha,update,
    phases:[snapshot('before',theta0,target0),snapshot('updated',theta1,target0),snapshot('copied',theta1,theta1)],
    gradientPaths:{semiGradient,sharedParameterResidual:qGrad.map((g,i)=>update.error*(g-gamma*tailGrad[i]))}};
}
