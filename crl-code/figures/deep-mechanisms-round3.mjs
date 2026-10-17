/** Deterministic teaching calculations, not sampled learning experiments. */
const finite=(x,name)=>{if(!Number.isFinite(x))throw new TypeError(`${name} must be finite`);return x;};
const probability=(x,name)=>{finite(x,name);if(x<0||x>1)throw new RangeError(`${name} must lie in [0,1]`);return x;};
const distribution=(xs,name)=>{
  if(!Array.isArray(xs)||!xs.length)throw new TypeError(`${name} must be a nonempty distribution`);
  xs.forEach(x=>probability(x,name));
  if(Math.abs(xs.reduce((s,x)=>s+x,0)-1)>1e-12)throw new RangeError(`${name} must sum to one`);
};

/** Row-stochastic T moves prior mass; likelihood conditions the next state. */
export function bayesFilter({prior,transition,likelihood}) {
  distribution(prior,'prior');const size=prior.length;
  if(!Array.isArray(transition)||transition.length!==size||!Array.isArray(likelihood)||likelihood.length!==size)throw new RangeError('Belief dimensions mismatch');
  transition.forEach(row=>{if(row.length!==size)throw new RangeError('Transition dimensions mismatch');distribution(row,'transition row');});
  likelihood.forEach(x=>probability(x,'likelihood'));
  const flows=prior.map((p,i)=>transition[i].map(q=>p*q));
  const prediction=prior.map((_,j)=>flows.reduce((sum,row)=>sum+row[j],0));
  const joint=prediction.map((p,j)=>p*likelihood[j]),evidence=joint.reduce((s,p)=>s+p,0);
  if(!(evidence>0))throw new RangeError('Observation has zero evidence under this model');
  return {prior,transition,likelihood,flows,prediction,joint,evidence,posterior:joint.map(p=>p/evidence)};
}

/** A reflected left endpoint. Only ten right moves reach 10 in ten steps. */
export function corridorTrajectory(actions,horizon=actions.length) {
  if(!Number.isInteger(horizon)||horizon<1)throw new RangeError('horizon must be a positive integer');
  let position=0;const path=[position];
  for(const action of actions) {
    if(action!==-1&&action!==1)throw new RangeError('Corridor action must be -1 or 1');
    position=Math.min(horizon,Math.max(0,position+action));path.push(position);
  }
  return path;
}

export function corridorExample(horizon=10) {
  if(!Number.isInteger(horizon)||horizon<1||horizon>20)throw new RangeError('horizon must be in 1..20');
  const initial=Array(horizon+1).fill(0);initial[0]=1;
  const independent=[initial],committed=[initial];
  for(let t=1;t<=horizon;t++) {
    const next=Array(horizon+1).fill(0);
    independent.at(-1).forEach((p,x)=>{
      next[Math.max(0,x-1)]+=p/2;next[Math.min(horizon,x+1)]+=p/2;
    });
    independent.push(next);
    const persistent=Array(horizon+1).fill(0);persistent[0]=.5;persistent[t]+=.5;committed.push(persistent);
  }
  const actions=Array.from({length:horizon},(_,i)=>[1,1,-1,1,-1,1,1,-1,1,1][i%10]);
  return {horizon,independent,committed,illustrative:{actions,path:corridorTrajectory(actions,horizon)},
    fullRight:corridorTrajectory(Array(horizon).fill(1),horizon),
    successIndependent:independent.at(-1)[horizon],successCommitted:committed.at(-1)[horizon]};
}

export function criticValue(action){finite(action,'action');return -((action-.8)**2);}
export function deterministicActorStep({theta=0,stepSize=.1}={}) {
  finite(theta,'theta');finite(stepSize,'stepSize');if(stepSize<0)throw new RangeError('stepSize must be nonnegative');
  const gradient=-2*(theta-.8),next=theta+stepSize*gradient;
  return {theta,stepSize,gradient,next,before:criticValue(theta),after:criticValue(next)};
}
export function noisyAction({mean,noise,noiseClip=Infinity,low=-1,high=1}) {
  [mean,noise,low,high].forEach((x,i)=>finite(x,['mean','noise','low','high'][i]));
  if(!(high>low))throw new RangeError('Action bounds must be increasing');
  if(!(noiseClip>=0)||Number.isNaN(noiseClip))throw new RangeError('noiseClip must be nonnegative');
  const perturbation=Math.min(noiseClip,Math.max(-noiseClip,noise)),unbounded=mean+perturbation;
  return {mean,noise,noiseClip:Number.isFinite(noiseClip)?noiseClip:null,perturbation,unbounded,action:Math.min(high,Math.max(low,unbounded)),low,high};
}
export function actorExample(){return {step:deterministicActorStep(),
  behavior:noisyAction({mean:0,noise:.3}),target:noisyAction({mean:.9,noise:3,noiseClip:.2}),
  targetCritics:[3,2],reward:1,gamma:.9,targetValue:1+.9*2};}

export function gaussianLogDensity(u,{mean=0,sigma=1}={}) {
  finite(u,'u');finite(mean,'mean');finite(sigma,'sigma');
  if(!(sigma>0))throw new RangeError('sigma must be positive');
  return -.5*Math.log(2*Math.PI)-Math.log(sigma)-.5*((u-mean)/sigma)**2;
}
const softplus=x=>Math.max(0,x)+Math.log1p(Math.exp(-Math.abs(x)));
/** Evaluate in pre-squash coordinates, avoiding log(1-tanh(u)^2) cancellation. */
export function squashDensity(u,{mean=0,sigma=1,scale=1,offset=0}={}) {
  finite(scale,'scale');finite(offset,'offset');if(scale===0)throw new RangeError('scale must be nonzero');
  const logGaussian=gaussianLogDensity(u,{mean,sigma});
  const normalized=Math.tanh(u),logJacobian=2*(Math.log(2)-u-softplus(-2*u));
  const logDensity=logGaussian-logJacobian-Math.log(Math.abs(scale));
  return {u,normalized,action:offset+scale*normalized,logGaussian,logJacobian,logDensity,density:Math.exp(logDensity)};
}
/** Physical-coordinate density on the open support; zero at/outside endpoints. */
export function actionDensity(a,{mean=0,sigma=1,scale=1,offset=0}={}) {
  finite(a,'action');squashDensity(0,{mean,sigma,scale,offset});
  const z=(a-offset)/scale;
  return Math.abs(z)>=1?0:squashDensity(Math.atanh(z),{mean,sigma,scale,offset}).density;
}
export function densityExample() {
  const sample=Array.from({length:1201},(_,i)=>-6+i/100);
  return {scale:2,mean:0,sigma:1,interval:[-.5,.5],
    normalizedInterval:[Math.tanh(-.5),Math.tanh(.5)],physicalInterval:[2*Math.tanh(-.5),2*Math.tanh(.5)],
    atZero:{gaussian:gaussianLogDensity(0),normalized:squashDensity(0),physical:squashDensity(0,{scale:2})},
    curves:{gaussian:sample.filter(u=>Math.abs(u)<=3).map(u=>[u,Math.exp(gaussianLogDensity(u))]),
      normalized:[[-1,0],...sample.map(u=>{const p=squashDensity(u);return [p.action,p.density];}),[1,0]],
      physical:[[-2,0],...sample.map(u=>{const p=squashDensity(u,{scale:2});return [p.action,p.density];}),[2,0]]}};
}
export function computeDeepMechanismsRound3(){return {
  belief:bayesFilter({prior:[.6,.4],transition:[[.9,.1],[.2,.8]],likelihood:[.8,.2]}),
  exploration:corridorExample(),actor:actorExample(),density:densityExample()};}
