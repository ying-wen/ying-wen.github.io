/**
 * Exact teaching mechanisms. These calculations are not training experiments.
 * Keep environmental termination, sample boundaries and autodiff boundaries
 * separate: each changes a different part of the computation.
 */

/** Two offset, collision-free one-dimensional tilings from the chapter. */
export function activeTiles(state, {width=1, count=2}={}) {
  return Array.from({length:count},(_,k)=>`${k}:${Math.floor(state/width+k/count)}`);
}

export function tileExample() {
  const state=.25, alpha=.1, delta=1, active=activeTiles(state);
  const probes=[-.25,.25,.75,1.25].map(s=>{
    const keys=activeTiles(s), shared=keys.filter(key=>active.includes(key)).length;
    return {state:s,keys,shared,change:alpha*delta*shared};
  });
  return {state,alpha,delta,width:1,count:2,active,probes,
    intervals:[{left:-.5,right:0,change:.1},{left:0,right:.5,change:.2},
      {left:.5,right:1,change:.1},{left:1,right:1.5,change:0}]};
}

/** One backward GAE step. nextAdvantage must belong to the same trajectory. */
export function gaeStep({reward,value,nextValue,gamma,lambda,bootstrap,carry,nextAdvantage=0}) {
  const delta=reward+gamma*bootstrap*nextValue-value;
  return {delta,advantage:delta+gamma*lambda*carry*nextAdvantage};
}

export function gaeExample() {
  const common={reward:1,value:.5,nextValue:1,gamma:.9,lambda:.8,nextAdvantage:1};
  const cases=[
    {id:'terminal',label:'任务真正终止',bootstrap:0,carry:0},
    {id:'timeout',label:'人工截断后重置',bootstrap:1,carry:0},
    {id:'continuous',label:'下一行仍是同一轨迹',bootstrap:1,carry:1},
  ].map(v=>({...v,...gaeStep({...common,...v})}));
  // At an ordinary batch edge no following residual has been collected.
  const batchEnd=gaeStep({...common,bootstrap:1,carry:0,nextAdvantage:0});
  return {common,cases,batchEnd};
}

/** h is a numerical memory; sensitivity tracks dh/dtheta on this tape. */
export function recurrentSequence({theta=.2,a=.5,inputs=[1,0,0],boundary='none'}={}) {
  let h=0,sensitivity=0;
  return inputs.map((input,index)=>{
    // The boundary is between h1 and the computation of h2.
    if(index===1&&boundary!=='none') {
      sensitivity=0;
      if(boundary==='reset') h=0;
    }
    h=a*h+theta*input;
    sensitivity=a*sensitivity+input;
    return {step:index+1,input,h,sensitivity};
  });
}

export function memoryExample() {
  return {theta:.2,a:.5,inputs:[1,0,0],cases:[
    {id:'none',label:'保留完整梯度',steps:recurrentSequence()},
    {id:'detach',label:'仅 detach',steps:recurrentSequence({boundary:'detach'})},
    {id:'reset',label:'清空记忆',steps:recurrentSequence({boundary:'reset'})},
  ]};
}

/** Categorical projection onto equally spaced support, including exact atoms. */
export function categoricalProjection({support,probabilities,reward,gamma,terminal=false}) {
  if(support.length!==probabilities.length||support.length<2) throw new Error('Support mismatch');
  const lower=support[0],upper=support.at(-1),spacing=(upper-lower)/(support.length-1);
  if(!(spacing>0)||support.some((z,i)=>Math.abs(z-(lower+i*spacing))>1e-12)) throw new Error('Support must be equally spaced');
  if(probabilities.some(p=>!Number.isFinite(p)||p<0)||Math.abs(probabilities.reduce((a,b)=>a+b,0)-1)>1e-12) throw new Error('Invalid probability mass');
  const mass=support.map(()=>0),flows=[];
  const locations=support.map(z=>reward+(terminal?0:gamma)*z);
  locations.forEach((raw,j)=>{
    const z=Math.max(lower,Math.min(upper,raw)),b=(z-lower)/spacing;
    const lo=Math.floor(b),hi=Math.ceil(b);
    if(lo===hi) {
      mass[lo]+=probabilities[j];
      flows.push({from:j,to:lo,mass:probabilities[j]});
    } else {
      for(const [to,weight]of [[lo,hi-b],[hi,b-lo]]) {
        const amount=probabilities[j]*weight;
        mass[to]+=amount;flows.push({from:j,to,mass:amount});
      }
    }
  });
  return {support,probabilities,reward,gamma,terminal,locations,mass,flows,
    rawMean:locations.reduce((s,z,j)=>s+z*probabilities[j],0),
    projectedMean:support.reduce((s,z,j)=>s+z*mass[j],0)};
}

export function distributionExample() {
  const base={support:[-1,0,1],probabilities:[.2,.5,.3],reward:.5,gamma:.5};
  return {...categoricalProjection(base),terminalCase:categoricalProjection({...base,reward:0,terminal:true}),
    clippedCase:categoricalProjection({...base,reward:5})};
}

export function computeClassicVisualDepth() {
  return {tile:tileExample(),gae:gaeExample(),memory:memoryExample(),distribution:distributionExample()};
}
