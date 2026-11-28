/** One learned write gate in the continuing noisy-cue task.
 * Deterministic finite enumeration, not sampled training. Only the environment
 * reads z. Every loss treats the already-carried boundary activity as constant.
 */
export const settings = Object.freeze({eta:Math.log(1/3),alpha:1,accuracy:.8,delays:[4,8,4,4,8,4]});
export const example = [1,1,-1,1,1,-1].map((cue,i)=>({cue,z:cue,delay:settings.delays[i]}));
export const sigmoid = x => x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));

export function environment(draws){
  let visit=0,steps=0,place='C',elapsed=0;
  const read=i=>{const d=typeof draws==='function'?draws(i):draws[i];if(!d||![-1,1].includes(d.z)||![-1,1].includes(d.cue)||!Number.isInteger(d.delay)||d.delay<1)throw Error('invalid draw');return {...d};};
  let draw=read(0);
  const observation=(signal=0,reward=0)=>Object.freeze({place,signal,reward,terminated:false});
  return {initial:observation(draw.cue),get steps(){return steps;},step(action){
    if(['C','M'].includes(place)&&action==='forward'){elapsed++;place=elapsed===draw.delay?'J':'M';steps++;return observation();}
    if(place==='J'&&[-1,1].includes(action)){place=action===1?'L':'R';steps++;return observation(0,action*draw.z);}
    if(['L','R'].includes(place)&&action==='return'){draw=read(++visit);elapsed=0;place='C';steps++;return observation(draw.cue);}
    throw Error('illegal action');
  }};
}

/** The callbacks store optional diagnostics outside the constant-size learner. */
export function learner(env,options={},onActivity=()=>{},onDecision=()=>{}){
  let eta=options.eta??settings.eta,h=0,S=0,place,count=0,age=0,gate=0,oldH=0,cue=0;
  const alpha=options.alpha??settings.alpha,skip=options.skipUpdate??-1;
  function observe(o){
    place=o.place;
    if(place==='C'){
      // Boundary h is retained numerically, but its old parameter history is detached.
      oldH=h;cue=o.signal;gate=sigmoid(eta);h=(1-gate)*oldH+gate*cue;
      S=gate*(1-gate)*(cue-oldH);age=0;
    }else if(['M','J'].includes(place)){age++;/* h'=h; J=1, B=0, S'=S */}
    onActivity({step:env.steps,place,h,S,eta,signal:o.signal,reward:o.reward});
  }
  observe(env.initial);
  function advance(){
    const action=['C','M'].includes(place)?'forward':place==='J'?(h>=0?1:-1):'return';
    const saved=place==='J'?{visit:count+1,oldH,cue,gate,usedEta:eta,h,S,delay:age,action,prediction:action*h}:null;
    const o=env.step(action);observe(o);
    if(saved){
      const gradient=(saved.prediction-o.reward)*saved.action*saved.S,applied=count!==skip;
      if(applied)eta-=alpha*gradient;count++;
      onDecision({...saved,reward:o.reward,rewardStep:env.steps,gradient,applied,updatedEta:eta,nextGate:sigmoid(eta),hAfterUpdate:h});
    }
    return o;
  }
  return {advance,snapshot:()=>({eta,h,S,place,decisions:count,environmentSteps:env.steps,initializationCalls:1,resetCalls:0,terminated:false,gradientScalars:1})};
}

export function run(draws=example,options={}){
  const activity=[],decisions=[],env=environment(draws),agent=learner(env,options,r=>activity.push(r),r=>decisions.push(r));
  while(decisions.length<draws.length)agent.advance();
  return {...agent.snapshot(),activity,decisions,totalReward:decisions.reduce((v,r)=>v+r.reward,0)};
}

/** Independent per-passage oracle: expand h as a weighted sum of old cues.
 * This does not use the environment, learner, or forward activity recurrence.
 */
export function expandedOracle(draws,options={}){
  const gates=[],cues=[],rows=[];let eta=options.eta??settings.eta,steps=-1;
  const sum=()=>cues.reduce((s,c,j)=>s+c*gates[j]*gates.slice(j+1).reduce((p,g)=>p*(1-g),1),0);
  for(const [i,d] of draws.entries()){
    const oldH=sum(),gate=sigmoid(eta);gates.push(gate);cues.push(d.cue);
    const h=sum(),action=h>=0?1:-1,reward=action*d.z,S=gate*(1-gate)*(d.cue-oldH);
    // d*r recovers the mode only after the action; this is an oracle simplification.
    const gradient=(h-d.z)*S;
    if(i!==(options.skipUpdate??-1))eta-=(options.alpha??1)*gradient;
    steps+=d.delay+2;rows.push({h,oldH,gate,S,action,reward,gradient,updatedEta:eta,rewardStep:steps});
  }
  return rows;
}

export function enumerate(options={}){
  const delays=options.delays??settings.delays,p=options.accuracy??settings.accuracy;
  let probability=0,pathCount=0;const expectedRewards=delays.map(()=>0),expectedGate=delays.map(()=>0),expectedLoss=delays.map(()=>0);
  function visit(draws,mass){
    if(draws.length===delays.length){
      const result=run(draws,options);probability+=mass;pathCount++;
      for(const [i,r] of result.decisions.entries()){expectedRewards[i]+=mass*r.reward;expectedGate[i]+=mass*r.gate;expectedLoss[i]+=mass*.5*(r.prediction-r.reward)**2;}
      return;
    }
    for(const z of [-1,1])for(const cue of [-1,1])visit([...draws,{z,cue,delay:delays[draws.length]}],mass*.5*(cue===z?p:1-p));
  }
  visit([],1);return {pathCount,probability,expectedRewards,expectedGate,expectedLoss,expectedTotal:expectedRewards.reduce((a,b)=>a+b,0)};
}

export function conditionalLoss(eta,oldH,cue,action,reward){const g=sigmoid(eta),h=(1-g)*oldH+g*cue;return .5*(action*h-reward)**2;}
export function learnedMemoryData(){
  const live=run(),omitted=run(example,{skipUpdate:2}),frozen=run(example,{alpha:0}),oldH=live.decisions[2].oldH;
  const sweep=Array.from({length:65},(_,i)=>{const eta=-8+i/4,gate=sigmoid(eta);return {eta,gate,h:(1-gate)*oldH-gate,sensitivity:-(1+oldH)*gate*(1-gate)};});
  return {kind:'exact-finite-enumeration',sampledRuns:0,settings,example,live,omitted,frozen,sweep,
    expectation:enumerate(),omittedExpectation:enumerate({skipUpdate:2}),frozenExpectation:enumerate({alpha:0}),
    uninformativeExpectation:enumerate({accuracy:.5}),saturated:run(example,{eta:-8}),
    conditionalMeanCoefficient:.6,optimalConditionalLoss:.32,overwriteConditionalLoss:.4};
}
if(typeof process!=='undefined'&&process.argv[1]&&import.meta.url===new URL('file://'+process.argv[1]).href)console.log(JSON.stringify(learnedMemoryData(),null,2));
