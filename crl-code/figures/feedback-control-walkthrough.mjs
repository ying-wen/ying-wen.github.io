/**
 * Exact teaching examples, not sampled motor data or controller training.
 * State is measured before the action; the unobserved disturbance is then added.
 * Every run keeps its gain and observation delay fixed.
 */
const finite = (x, name) => {
  if (!Number.isFinite(x)) throw new TypeError(`${name} must be finite`);
  return x;
};

export function feedbackPoles(gain, delay = 0) {
  finite(gain, 'gain');
  if (![0, 1].includes(delay)) throw new RangeError('delay must be zero or one step');
  let roots;
  if (delay === 0) roots = [{real: 1 - gain, imaginary: 0}];
  else {
    const discriminant = 1 - 4 * gain;
    if (discriminant >= 0) {
      const r = Math.sqrt(discriminant);
      roots = [{real: (1 + r) / 2, imaginary: 0}, {real: (1 - r) / 2, imaginary: 0}];
    } else {
      const i = Math.sqrt(-discriminant) / 2;
      roots = [{real: .5, imaginary: i}, {real: .5, imaginary: -i}];
    }
  }
  roots = roots.map(root => ({...root, modulus: Math.hypot(root.real, root.imaginary)}));
  return {gain, delay, roots, spectralRadius: Math.max(...roots.map(root => root.modulus)), asymptoticallyStable: roots.every(root => root.modulus < 1)};
}

export function simulateFixedFeedback({gain, delay = 0, initial = 0, previous = initial, disturbances, stepSeconds = .1}) {
  finite(gain, 'gain'); finite(initial, 'initial'); finite(previous, 'previous');
  finite(stepSeconds, 'stepSeconds');
  if (stepSeconds <= 0) throw new RangeError('stepSeconds must be positive');
  if (![0, 1].includes(delay)) throw new RangeError('delay must be zero or one step');
  if (!Array.isArray(disturbances) || !disturbances.length) throw new TypeError('disturbances must be a nonempty array');
  disturbances.forEach(x => finite(x, 'disturbance'));
  let current = initial, old = previous;
  const states = [{step: 0, seconds: 0, position: current}], rows = [];
  disturbances.forEach((disturbance, step) => {
    const observed = delay === 0 ? current : old;
    const action = -gain * observed;
    const next = current + action + disturbance;
    rows.push({step, seconds: step * stepSeconds, positionBefore: current, observed, gain, action, disturbance, positionAfter: next});
    old = current; current = next;
    states.push({step: step + 1, seconds: (step + 1) * stepSeconds, position: current});
  });
  return {gain, delay, stepSeconds, initial, previous, disturbances: [...disturbances], poles: feedbackPoles(gain, delay), states, rows};
}

/** Minimum squared action subject to reaching x1+x2=target in one step. */
export function minimumEffortCorrection(state, target = 1) {
  if (!Array.isArray(state) || state.length !== 2) throw new TypeError('state must contain two coordinates');
  state.forEach(x => finite(x, 'coordinate')); finite(target, 'target');
  const error = target - (state[0] + state[1]), action = [error / 2, error / 2];
  return {state: [...state], target, error, action, after: state.map((x, i) => x + action[i]), actionSquared: action.reduce((sum, u) => sum + u * u, 0)};
}

export function taskPerturbation(disturbance, initial = [.5, .5]) {
  if (!Array.isArray(disturbance) || disturbance.length !== 2) throw new TypeError('disturbance must contain two coordinates');
  disturbance.forEach(x => finite(x, 'disturbance'));
  if (!Array.isArray(initial) || initial.length !== 2) throw new TypeError('initial must contain two coordinates');
  initial.forEach(x => finite(x, 'initial coordinate'));
  const afterDisturbance = initial.map((x, i) => x + disturbance[i]);
  const correction = minimumEffortCorrection(afterDisturbance);
  const states = [[...initial], afterDisturbance, correction.after];
  return {initial: [...initial], disturbance: [...disturbance], disturbanceSquared: disturbance.reduce((sum, x) => sum + x * x, 0), ...correction, states, errors: states.map(x => 1 - (x[0] + x[1]))};
}

export function feedbackControlWalkthrough() {
  const disturbances = [0, 0, 1, 0, 0, 0, 0, 0, 0];
  return {
    evidenceKind: 'exact', sampledRuns: 0, randomGeneratorUsed: false,
    scalarTask: {goal: 'hold position at zero',positionUnit: 'cm',actionUnit: 'cm per step',disturbanceUnit: 'cm per step',stepSeconds: .1,update: 'x[t+1]=x[t]+u[t]+d[t]',order: ['observe', 'act', 'add disturbance'],initial: 0,previous: 0,disturbanceObservedBeforeAction: false},
    cases: [
      {id: 'current',label: '即时 k=1.25',...simulateFixedFeedback({gain: 1.25, disturbances})},
      {id: 'large',label: '过大 k=2.25',...simulateFixedFeedback({gain: 2.25, disturbances})},
      {id: 'delayed',label: '旧 x，k=1.25',...simulateFixedFeedback({gain: 1.25, delay: 1, disturbances})},
    ],
    redundantTask: {goal: 'x1+x2=1',unit: 'dimensionless',dynamics: 'xAfter=x+u',objective: 'minimize u1^2+u2^2 subject to reaching the task line',parametersLearned: false},
    perturbations: [
      {id: 'relevant',label: '任务相关扰动',...taskPerturbation([.2, .2])},
      {id: 'redundant',label: '冗余方向扰动',...taskPerturbation([.2, -.2])},
    ],
  };
}
