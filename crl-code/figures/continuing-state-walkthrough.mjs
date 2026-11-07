/** A continuing, partially observed door loop. Exact finite-window arithmetic, not training. */
export const settings = Object.freeze({a: -.5, alpha: .5, accuracy: .8, decisions: 2});

/** z stays private for the whole lifetime. Cues are consumed only on actual returns to C. */
export function continuingEnvironment(z, cues) {
  if (![-1, 1].includes(z)) throw new RangeError('hidden mode must be ±1');
  let place = 'C', cueIndex = 0, steps = 0;
  const getCue = () => {
    const cue = typeof cues === 'function' ? cues(cueIndex++) : cues[cueIndex++];
    if (![-1, 1].includes(cue)) throw new RangeError('a further cue is required');
    return cue;
  };
  const observation = (signal, reward = 0) => Object.freeze({place, signal, reward, terminated: false});
  const initial = observation(getCue());
  return {initial, get steps() { return steps; }, step(action) {
    if (place === 'C' && action === 'forward') { place = 'J'; steps++; return observation(0); }
    if (place === 'J' && [-1, 1].includes(action)) {
      place = action === 1 ? 'L' : 'R'; steps++; return observation(action * z, action * z);
    }
    if (['L', 'R'].includes(place) && action === 'return') {
      const cue = getCue(); place = 'C'; steps++; return observation(cue);
    }
    throw new Error('illegal transition');
  }};
}

/** Only public observations, actions and arrived rewards enter this learner. */
export function continuingLearner(env, options = {}) {
  let a = options.a ?? settings.a, h = 0, sensitivity = 0, place;
  const alpha = options.alpha ?? settings.alpha, boundary = options.boundary ?? 'carry';
  if (!['carry', 'detach', 'erase'].includes(boundary)) throw new RangeError('unknown diagnostic boundary');
  const activity = [], decisions = [];
  function observe(o) {
    const input = options.eraseCue && o.place === 'C' ? 0 : o.signal;
    const oldH = h, oldSensitivity = sensitivity;
    h = a * oldH + input; sensitivity = oldH + a * oldSensitivity; place = o.place;
    activity.push({step: env.steps, place, input, h, sensitivity, usedParameter: a, reward: o.reward, terminated: o.terminated});
  }
  observe(env.initial);
  function advance() {
    const action = place === 'C' ? 'forward' : place === 'J' ? (options.fixedAction ?? (h >= 0 ? 1 : -1)) : 'return';
    const saved = place === 'J' ? {decisionState: h, decisionSensitivity: sensitivity, action,
      prediction: action * h, derivative: action * sensitivity, usedParameter: a} : null;
    const o = env.step(action);
    observe(o); // Arrived activity uses the old parameter, including the reward observation.
    if (saved) {
      const gradient = (saved.prediction - o.reward) * saved.derivative;
      const row = {...saved, reward: o.reward, gradient, arrivedState: h, arrivedSensitivity: sensitivity,
        updatedParameter: a - alpha * gradient, step: env.steps};
      decisions.push(row); a = row.updatedParameter;
      // Deliberate learner interventions at the first reward, never an environment reset.
      if (decisions.length === 1 && boundary !== 'carry') { sensitivity = 0; if (boundary === 'erase') h = 0; }
    }
    return o;
  }
  return {advance, snapshot: () => ({a, h, sensitivity, place, activity: activity.map(x=>({...x})),
    decisions: decisions.map(x=>({...x})), totalReward: decisions.reduce((s,d)=>s+d.reward,0),
    environmentSteps: env.steps, initializationCalls: 1, resetCalls: 0, terminated: false,
    artificialStateClears: boundary === 'erase' && decisions.length ? 1 : 0})};
}

export function runContinuing(z, cues, options = {}) {
  const learner = continuingLearner(continuingEnvironment(z, cues), options);
  const n = options.decisions ?? settings.decisions;
  for (let i = 0; i < 3 * n - 1; i++) learner.advance();
  return learner.snapshot(); // Observation window only: the same learner can keep advancing.
}

/** Fixed recorded parameter schedule and inputs. No update or policy is rerun here. */
export function scheduleOutput(activity, epsilon = 0) {
  return activity.reduce((h, row) => (row.usedParameter + epsilon) * h + row.input, 0);
}

/** Independent sum over input-to-output paths, differentiating each parameter occurrence. */
export function pathSensitivity(activity) {
  let derivative = 0;
  for (let i = 0; i < activity.length; i++) {
    for (let j = i + 1; j < activity.length; j++) {
      let path = activity[i].input;
      for (let k = i + 1; k < activity.length; k++) if (k !== j) path *= activity[k].usedParameter;
      derivative += path;
    }
  }
  return derivative;
}

/** Analysis only. Eight full paths; the learner does not receive these latent draws. */
export function enumerateContinuing(options = {}) {
  const p = options.accuracy ?? settings.accuracy, paths = [];
  for (const z of [-1, 1]) for (const c1 of [-1, 1]) for (const c2 of [-1, 1]) {
    const probability = .5 * (c1 === z ? p : 1-p) * (c2 === z ? p : 1-p);
    paths.push({z, cues: [c1,c2], probability, ...runContinuing(z,[c1,c2],options)});
  }
  const expectedRewards = [0,1].map(i=>paths.reduce((sum,row)=>sum+row.probability*row.decisions[i].reward,0));
  return {paths, probability: paths.reduce((s,r)=>s+r.probability,0), expectedRewards,
    expectedTotal: expectedRewards[0]+expectedRewards[1],
    secondSuccess: paths.reduce((s,r)=>s+r.probability*(r.decisions[1].reward===1),0)};
}

/** Independent two-decision closed form and conditional expectation tree; no activity trace. */
export function conditionalOracle({a = settings.a, alpha = settings.alpha, accuracy: p = settings.accuracy,
  eraseCue = false, boundary = 'carry', fixedAction} = {}) {
  const branches = [];
  let first = 0, second = 0;
  for (const z of [-1,1]) for (const c1 of [-1,1]) {
    const x = eraseCue ? 0 : c1, hJ = a*x;
    const d1 = fixedAction ?? (hJ>=0?1:-1), r1=d1*z;
    const nextA=a-alpha*(d1*hJ-r1)*d1*x;
    const carry = boundary==='erase'?0:a*a*x+r1;
    const branchProbability=.5*(c1===z?p:1-p);
    let nextReward = 0;
    for (const c2 of [-1,1]) {
      const hJ2=nextA*nextA*carry+nextA*(eraseCue?0:c2);
      const d2=fixedAction??(hJ2>=0?1:-1);
      nextReward+=(c2===z?p:1-p)*d2*z;
    }
    first+=branchProbability*r1; second+=branchProbability*nextReward;
    branches.push({z,c1,probability:branchProbability,nextA,conditionalSecondReward:nextReward});
  }
  return {expectedRewards:[first,second],expectedTotal:first+second,branches};
}

export function continuingStateData() {
  const summary=options=>{const {paths,...rest}=enumerateContinuing(options);return rest;};
  return {settings, live:runContinuing(1,[1,1]), frozen:runContinuing(1,[1,1],{alpha:0}),
    detached:runContinuing(1,[1,1],{boundary:'detach'}), erased:runContinuing(1,[1,1],{boundary:'erase'}),
    fixedRight:runContinuing(1,[1,1],{fixedAction:-1}), enumeration:enumerateContinuing(),
    frozenEnumeration:summary({alpha:0}), noCueEnumeration:summary({eraseCue:true}),
    uninformativeCueEnumeration:summary({accuracy:.5}), largeStepEnumeration:summary({alpha:2}),
    oracle:conditionalOracle()};
}
