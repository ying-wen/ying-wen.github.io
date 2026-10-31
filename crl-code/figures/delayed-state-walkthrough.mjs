/** Independent new door modes, finite memory, and one delayed reward loss.
 * Deterministic enumeration, no sampled training. The environment alone reads z.
 */
export const settings = Object.freeze({b: -.5, rho: 1, alpha: 1, accuracy: .8, window: 2});
export const example = Object.freeze([{z:1,cue:1,delay:4},{z:-1,cue:-1,delay:8},{z:1,cue:1,delay:4}]);

export function delayedEnvironment(draws) {
  let visit=0, steps=0, place='C', elapsed=0;
  const getDraw=i=>{
    const x=typeof draws==='function'?draws(i):draws[i];
    if(!x||![-1,1].includes(x.z)||![-1,1].includes(x.cue)||!Number.isInteger(x.delay)||x.delay<1) throw new RangeError('each arrival needs z,cue in ±1 and a positive integer delay');
    return {...x};
  };
  let draw=getDraw(0);
  const observation=(signal=0,reward=0)=>Object.freeze({place,signal,reward,terminated:false});
  const initial=observation(draw.cue);
  return {initial,get steps(){return steps;},step(action){
    if(['C','M'].includes(place)&&action==='forward'){
      elapsed++;place=elapsed===draw.delay?'J':'M';steps++;return observation();
    }
    if(place==='J'&&[-1,1].includes(action)){
      place=action===1?'L':'R';steps++;return observation(0,action*draw.z);
    }
    if(['L','R'].includes(place)&&action==='return'){
      draw=getDraw(++visit);elapsed=0;place='C';steps++;return observation(draw.cue);
    }
    throw new Error('illegal transition');
  }};
}

/** A real bounded tape: each item is one local (J,B) operation, ending at J.
 * The cue write counts as one operation, followed by D corridor operations.
 */
export function backwardTape(tape){
  let adjoint=1,derivative=0;
  for(let i=tape.length-1;i>=0;i--){derivative+=adjoint*tape[i].B;adjoint*=tape[i].J;}
  return derivative;
}

/** Callbacks are optional external diagnostic recorders, not learner memory. */
export function delayedLearner(env,options={},onActivity=()=>{},onDecision=()=>{}){
  let b=options.b??settings.b,h=0,S=0,place,count=0,age=0;
  const rho=options.rho??settings.rho,alpha=options.alpha??settings.alpha;
  const method=options.method??'rtrl',K=options.window??settings.window,tape=[];
  if(!['rtrl','tbptt'].includes(method)||!Number.isInteger(K)||K<1)throw new RangeError('invalid gradient method/window');
  const sensitivity=()=>method==='rtrl'?S:backwardTape(tape);
  function observe(o){
    place=o.place;
    if(place==='C'||place==='M'||place==='J'){
      const write=place==='C',J=write?0:rho,B=write?o.signal:0;
      h=write?b*o.signal:rho*h;
      age=write?0:age+1;
      if(method==='rtrl')S=J*S+B;
      else {tape.push({J,B});if(tape.length>K)tape.shift();}
    }
    onActivity({step:env.steps,place,signal:o.signal,reward:o.reward,h,sensitivity:sensitivity(),b,age,tapeItems:tape.length});
  }
  observe(env.initial);
  function advance(){
    const action=['C','M'].includes(place)?'forward':place==='J'?(h>=0?1:-1):'return';
    const saved=place==='J'?{step:env.steps,delay:age,usedParameter:b,h,sensitivity:sensitivity(),action,prediction:action*h}:null;
    const observation=env.step(action);observe(observation);
    if(saved){
      const gradient=(saved.prediction-observation.reward)*saved.action*saved.sensitivity;
      b-=alpha*gradient;count++;
      onDecision({...saved,rewardStep:env.steps,reward:observation.reward,gradient,updatedParameter:b});
    }
    return observation;
  }
  return {advance,snapshot:()=>({b,h,sensitivity:sensitivity(),place,decisions:count,environmentSteps:env.steps,
    initializationCalls:1,resetCalls:0,terminated:false,gradientMemoryItems:method==='rtrl'?1:tape.length})};
}

export function runDelayed(draws,options={}){
  const activity=[],decisions=[],env=delayedEnvironment(draws);
  const learner=delayedLearner(env,options,r=>activity.push(r),r=>decisions.push(r));
  const count=options.decisions??draws.length;
  while(decisions.length<count)learner.advance();
  return {...learner.snapshot(),activity,decisions,totalReward:decisions.reduce((s,x)=>s+x.reward,0)};
}

/** Fixed recent-observation controller. No trained b and no gradient tape.
 * It stores the last W cue/zero signals, then follows a remaining cue, tie L.
 */
export function runRecentWindow(draws,W=2){
  if(!Number.isInteger(W)||W<1)throw new RangeError('W must be positive');
  const env=delayedEnvironment(draws),memory=[],decisions=[];
  let place,h=0;
  const observe=o=>{place=o.place;memory.push(o.signal);if(memory.length>W)memory.shift();h=[...memory].reverse().find(x=>x!==0)??0;};
  observe(env.initial);
  while(decisions.length<draws.length){
    const action=['C','M'].includes(place)?'forward':place==='J'?(h>=0?1:-1):'return';
    const saved=place==='J'?{h,action,memory:[...memory]}:null,o=env.step(action);observe(o);
    if(saved)decisions.push({...saved,reward:o.reward,rewardStep:env.steps});
  }
  return {decisions,totalReward:decisions.reduce((s,x)=>s+x.reward,0),environmentSteps:env.steps,resetCalls:0};
}

/** Independent closed form. No state or sensitivity recurrence. */
export function passageOracle(b,cue,z,delay,options={}){
  const rho=options.rho??1,alpha=options.alpha??1,K=options.window??2;
  const attenuation=rho**delay,h=b*attenuation*cue,action=h>=0?1:-1,reward=action*z;
  const sensitivity=options.method==='tbptt'&&delay>=K?0:attenuation*cue;
  const gradient=(action*h-reward)*action*sensitivity;
  return {h,action,reward,sensitivity,gradient,updatedParameter:b-alpha*gradient};
}

export function enumerateDelayed(delays=[4,8,4],options={}){
  const p=options.accuracy??settings.accuracy,paths=[];
  function visit(draws,probability){
    if(draws.length===delays.length){
      const result=options.controller==='recent'?runRecentWindow(draws,options.stateWindow??2):runDelayed(draws,options);
      paths.push({draws,probability,...result});return;
    }
    for(const z of [-1,1])for(const cue of [-1,1])visit([...draws,{z,cue,delay:delays[draws.length]}],probability*.5*(cue===z?p:1-p));
  }
  visit([],1);
  const expectedRewards=delays.map((_,i)=>paths.reduce((s,x)=>s+x.probability*x.decisions[i].reward,0));
  return {paths,probability:paths.reduce((s,x)=>s+x.probability,0),expectedRewards,
    expectedTotal:expectedRewards.reduce((a,b)=>a+b,0),environmentSteps:paths[0].environmentSteps};
}

/** Conditional expectation over correctness signs e=Z*c, independent each visit.
 * For full/tape controllers this uses 2^n branches, not the 4^n trajectory engine.
 */
export function expectedOracle(delays=[4,8,4],options={}){
  const p=options.accuracy??.8,rewards=delays.map(()=>0);
  function recurse(i,b,prob){
    if(i===delays.length)return;
    for(const [e,mass] of [[1,p],[-1,1-p]]){
      const row=passageOracle(b,1,e,delays[i],options);
      // When b=0, actual tie L has E[r]=0 after averaging the two cue signs.
      rewards[i]+=prob*mass*(b===0?0:row.reward);
      recurse(i+1,row.updatedParameter,prob*mass);
    }
  }
  recurse(0,options.b??settings.b,1);
  return {expectedRewards:rewards,expectedTotal:rewards.reduce((a,b)=>a+b,0)};
}

export function delayCurve(){
  return Array.from({length:16},(_,i)=>{
    const delay=i+1,base=passageOracle(-.5,1,1,delay),decay=passageOracle(-.5,1,1,delay,{rho:.5}),cut=passageOracle(-.5,1,1,delay,{method:'tbptt',window:2});
    return {delay,held:base,decayed:decay,truncated:cut,recentCueRetained:delay<2};
  });
}

export function delayedStateData(){
  const summary=o=>{const {paths,...rest}=enumerateDelayed([4,8,4],o);return rest;};
  return {kind:'exact-mechanism',sampledRuns:0,settings,example,
    live:runDelayed(example),truncated:runDelayed(example,{method:'tbptt'}),decayed:runDelayed(example,{rho:.5}),
    frozen:runDelayed(example,{alpha:0}),recent:runRecentWindow(example),
    enumeration:enumerateDelayed(),truncatedExpectation:summary({method:'tbptt'}),decayedExpectation:summary({rho:.5}),
    recentExpectation:summary({controller:'recent'}),noCueExpectation:summary({accuracy:.5}),
    oracle:expectedOracle(),curve:delayCurve()};
}

if(typeof process!=='undefined'&&process.argv[1]&&import.meta.url===new URL('file://'+process.argv[1]).href){
  console.log(JSON.stringify(delayedStateData(),null,2));
}
