/** Exact, finite episodic control example. Environment secrets never enter learnerEpisode. */
export const defaults = Object.freeze({a: -.5, alpha: .5, accuracy: .8});

export function doorEnvironment(z, cue) {
  if (![z, cue].every(v => v === -1 || v === 1)) throw new RangeError('z and cue must be signed bits');
  let place = 'C';
  return {
    initial: Object.freeze({place: 'C', signal: cue, reward: 0, terminal: false}),
    step(action) {
      if (place === 'C' && action === 'forward') {
        place = 'J'; return {place, signal: 0, reward: 0, terminal: false};
      }
      if (place !== 'J' || ![-1, 1].includes(action)) throw new Error('invalid or post-terminal action');
      place = action === 1 ? 'L' : 'R';
      const reward = action * z;
      return {place, signal: reward, reward, terminal: true};
    },
  };
}

/** The learner receives public observations/rewards only, never z or cue accuracy. */
export function learnerEpisode(env, a, {alpha = defaults.alpha, recordedAction = null, eraseCue = false} = {}) {
  let h = 0, sensitivity = 0;
  const history = [], activity = [];
  const observe = observation => {
    const input = eraseCue && observation.place === 'C' ? 0 : observation.signal;
    const oldH = h;
    h = a * oldH + input;
    sensitivity = oldH + a * sensitivity;
    history.push({observation: {...observation}});
    activity.push({place: observation.place, input, h, sensitivity, usedParameter: a});
  };
  observe(env.initial);
  history.push({action: 'forward'});
  observe(env.step('forward'));
  const decisionState = h, decisionSensitivity = sensitivity;
  const selectedAction = h >= 0 ? 1 : -1; // A tie selects L.
  const action = recordedAction ?? selectedAction;
  if (![-1, 1].includes(action)) throw new RangeError('door action must be ±1');
  const prediction = action * h, derivative = action * sensitivity;
  history.push({action});
  const outcome = env.step(action);
  observe(outcome); // The final activity uses the episode's old parameter.
  const error = outcome.reward - prediction;
  const gradient = -error * derivative;
  const updatedParameter = a - alpha * gradient;
  return {usedParameter: a, decisionState, decisionSensitivity, selectedAction, action,
    prediction, derivative, reward: outcome.reward, error, gradient, updatedParameter,
    terminalState: h, terminalSensitivity: sensitivity, activity, history,
    environmentSteps: 2, resetCalls: 1, terminated: true, truncated: false};
}

export function runEpisodes(cases, options = {}) {
  let a = options.a ?? defaults.a;
  const episodes = cases.map(({z, cue}) => {
    const episode = learnerEpisode(doorEnvironment(z, cue), a, options);
    a = episode.updatedParameter;
    return episode;
  });
  return {episodes, finalParameter: a, totalReward: episodes.reduce((s, e) => s + e.reward, 0),
    environmentSteps: 2 * episodes.length, resetCalls: episodes.length};
}

/** A fixed, observable-only dataset collected by the always-R behavior policy. */
export function collectRightRecords(cases) {
  return cases.map(({z, cue}) => {
    const env = doorEnvironment(z, cue);
    return {initial: env.initial, transitions: [
      {action: 'forward', observation: env.step('forward')},
      {action: -1, observation: env.step(-1)},
    ]};
  });
}

export function trainOnRecords(records, options = {}) {
  let a = options.a ?? defaults.a;
  const episodes = records.map(record => {
    let index = 0;
    const source = {initial: record.initial, step(action) {
      const item = record.transitions[index++];
      if (!item || item.action !== action) throw new Error('requested action differs from fixed record');
      return {...item.observation};
    }};
    const row = learnerEpisode(source, a, {...options, recordedAction: record.transitions[1].action});
    row.environmentSteps = 0; row.resetCalls = 0; row.recordedTransitions = 2;
    a = row.updatedParameter;
    return row;
  });
  return {episodes, finalParameter: a, totalReward: episodes.reduce((s,e)=>s+e.reward,0),
    environmentSteps: 0, resetCalls: 0, recordedTransitions: 2*records.length,
    collectionEnvironmentSteps: 2*records.length, collectionResetCalls: records.length};
}

export function latentCases(accuracy = defaults.accuracy) {
  if (accuracy < 0 || accuracy > 1) throw new RangeError('accuracy must be a probability');
  return [-1, 1].flatMap(z => [-1, 1].map(cue => ({z, cue,
    probability: .5 * (z === cue ? accuracy : 1 - accuracy)})));
}

/** Analysis only: enumerate hidden draws outside the learner. */
export function enumerateTwoEpisodes(options = {}) {
  const cases = latentCases(options.accuracy ?? defaults.accuracy);
  const paths = cases.flatMap(first => cases.map(second => ({probability: first.probability * second.probability,
    draws: [first, second], ...runEpisodes([first, second], options)})));
  const weighted = f => paths.reduce((sum, path) => sum + path.probability * f(path), 0);
  return {paths, probability: weighted(() => 1),
    expectedRewards: [0, 1].map(i => weighted(p => p.episodes[i].reward)),
    secondSuccess: weighted(p => p.episodes[1].reward === 1 ? 1 : 0),
    expectedTotal: weighted(p => p.totalReward)};
}

export function recurrentControlData() {
  const selected = [{z: 1, cue: 1}, {z: 1, cue: 1}];
  const summary = options => {
    const {paths, ...rest} = enumerateTwoEpisodes(options); return rest;
  };
  return {settings: defaults, live: runEpisodes(selected),
    record: trainOnRecords(collectRightRecords(selected)),
    erased: runEpisodes(selected, {eraseCue: true}),
    frozen: runEpisodes(selected, {alpha: 0}),
    misleading: runEpisodes([{z: -1, cue: 1}, {z: 1, cue: 1}]),
    enumeration: summary({}), frozenEnumeration: summary({alpha: 0}),
    noCueEnumeration: summary({eraseCue: true}), noInformationEnumeration: summary({accuracy: .5}),
    branchMasses: [.8 * .8, .8 * .2, .2 * .2, .2 * .8]};
}
