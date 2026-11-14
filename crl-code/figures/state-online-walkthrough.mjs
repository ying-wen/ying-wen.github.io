/** A finite deterministic online learner, plus three explicit interventions.
 * There is one real parameter update after observing h2; h and S are retained.
 * No random training, optimizer unrolling approximation, or control experiment.
 */
export const settings = Object.freeze({a: .5, b: 1, alpha: .5, earlyTarget: 1, finalTarget: 1});
export const observations = Object.freeze([1, 0, 0, 0]);

export function runOnline(overrides = {}) {
  const c = {...settings, ...overrides};
  let a = c.a, b = c.b, h = 0, s = [0, 0];
  const rows = [];
  for (const [i, o] of observations.entries()) {
    const before = h, used = [a, b];
    h = a * before + b * o;
    s = [before + a * s[0], o + a * s[1]];
    let write = [0, 0], updateGradient = null;
    if (i === 1) {
      updateGradient = s.map(v => (h - c.earlyTarget) * v);
      write = updateGradient.map(v => -c.alpha * v);
      a += write[0]; b += write[1];
    }
    rows.push({t: i + 1, observation: o, old_h: before, used, h, trace: [...s], updateGradient, write, after: [a, b]});
  }
  return {config: c, rows, current: [a, b], h, trace: s, loss: .5 * (h - c.finalTarget) ** 2};
}

// The schedule is an input here. Its generating update rule is not rerun.
export function replaySchedule(schedule, offset = [0, 0]) {
  let h = 0;
  for (const [i, theta] of schedule.entries()) h = (theta[0] + offset[0]) * h + (theta[1] + offset[1]) * observations[i];
  return h;
}

// Independent expansion of each parameter occurrence, without the S recurrence.
export function occurrenceContributions(rows) {
  return rows.map((r, i) => {
    let downstream = 1;
    for (let j = i + 1; j < rows.length; j++) downstream *= rows[j].used[0];
    return {t: r.t, derivative: [downstream * r.old_h, downstream * r.observation]};
  });
}

// Closed form for differentiating the entire one-update learner with respect to
// initial a,b and alpha. b+ has no forward effect in the remaining zero inputs.
export function learnerOracle(overrides = {}) {
  const {a, b, alpha, earlyTarget: y, finalTarget} = {...settings, ...overrides};
  const h2 = a * b, aPlus = a - alpha * (h2 - y) * b;
  const da = 1 - alpha * b * b, db = -alpha * (2 * a * b - y);
  const stateDerivative = [aPlus * aPlus * b + 2 * aPlus * h2 * da,
    aPlus * aPlus * a + 2 * aPlus * h2 * db];
  const alphaDerivative = -2 * aPlus * h2 * (h2 - y) * b;
  const h = aPlus * aPlus * h2, error = h - finalTarget;
  return {h, stateDerivative, alphaDerivative, lossGradient: stateDerivative.map(v => error * v), lossAlphaDerivative: error * alphaDerivative};
}

export function onlineData() {
  const actual = runOnline(), schedule = actual.rows.map(r => r.used);
  const [a, b] = actual.current, hReplay = a ** 3 * b;
  const traceDerivative = actual.trace, replayDerivative = [3 * a * a * b, a ** 3];
  const learner = learnerOracle();
  const modes = {
    schedule: {h: actual.h, derivative: traceDerivative, gradient: traceDerivative.map(v => (actual.h - 1) * v)},
    replay: {h: hReplay, derivative: replayDerivative, gradient: replayDerivative.map(v => (hReplay - 1) * v)},
    learner: {h: learner.h, derivative: learner.stateDerivative, gradient: learner.lossGradient},
  };
  const curves = Array.from({length: 51}, (_, i) => {
    const epsilon = -.125 + i / 200;
    return {epsilon, schedule: replaySchedule(schedule, [epsilon, 0]),
      replay: replaySchedule(observations.map(() => actual.current), [epsilon, 0]),
      learner: runOnline({a: settings.a + epsilon}).h};
  });
  return {kind: 'exact-mechanism', sampled_runs: 0, actual, modes, learner,
    occurrenceContributions: occurrenceContributions(actual.rows), curves};
}
