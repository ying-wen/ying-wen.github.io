/** Deterministic recurrent replay mechanism; no training loop or dependencies. */
export const settings = Object.freeze({
  old: Object.freeze({w: -.8, rho: .8}),
  current: Object.freeze({w: .8, rho: .8}),
  target: Object.freeze({w: .6, rho: .7}),
  gamma: .9, alpha: .01, decisionIndex: 4, burnStart: 2, learnStart: 4,
});
const checkParams = p => {
  if (!p || !Number.isFinite(p.w) || !Number.isFinite(p.rho)) throw new TypeError('finite w and rho required');
};

/** x is the observed label, never an unobserved parcel position. */
export function unroll(observations, params, {initialState = 0, detachAt = [], resetBefore = []} = {}) {
  checkParams(params);
  if (!Number.isFinite(initialState)) throw new TypeError('finite initialState required');
  let h = initialState, dw = 0, drho = 0;
  return observations.map((x, index) => {
    if (!Number.isFinite(x)) throw new TypeError('finite observation required');
    const reset = resetBefore.includes(index), detached = detachAt.includes(index);
    if (reset) {h = 0; dw = 0; drho = 0;}
    if (detached) {dw = 0; drho = 0;}
    const previous = h;
    h = Math.tanh(params.rho * previous + params.w * x);
    const jacobian = 1 - h * h;
    dw = jacobian * (params.rho * dw + x);
    drho = jacobian * (params.rho * drho + previous);
    return {index, x, previous, h, dw, drho, jacobian, resetBefore: reset, detachedBefore: detached};
  });
}

export const doorQ = h => ({L: 1 + 3 * h, R: 1 - 3 * h});
export const forwardQ = h => 1 + 3 * h * h;
export const greedyDoor = h => doorQ(h).L >= doorQ(h).R ? 'L' : 'R';
export const parcelReward = (parcel, action) => {
  if (!['L', 'R'].includes(parcel) || !['L', 'R'].includes(action)) throw new RangeError('L/R required');
  return parcel === action ? 4 : -2;
};

export function tdTarget({reward, terminated, nextOnlineState, nextTargetState, gamma = settings.gamma}) {
  if (!Number.isFinite(reward) || gamma < 0 || gamma > 1) throw new RangeError('valid reward and gamma required');
  if (typeof terminated !== 'boolean') throw new TypeError('terminated must be explicit');
  // A real terminal has no next decision; do not query a fictitious next Q.
  if (terminated) return {y: reward, nextAction: null, continuation: 0};
  if (!Number.isFinite(nextOnlineState) || !Number.isFinite(nextTargetState)) throw new TypeError('nonterminal successor states required');
  const nextAction = greedyDoor(nextOnlineState), continuation = doorQ(nextTargetState)[nextAction];
  return {y: reward + gamma * continuation, nextAction, continuation};
}

export function terminalLoss(row, action = 'R', reward = -2, alpha = settings.alpha, params = settings.current) {
  if (!['L', 'R'].includes(action)) throw new RangeError('recorded action required');
  const sign = action === 'L' ? 1 : -1, q = doorQ(row.h)[action], residual = q - reward;
  const gradient = {w: residual * sign * 3 * row.dw, rho: residual * sign * 3 * row.drho};
  return {action, reward, y: reward, q, residual, loss: .5 * residual * residual, gradient,
    updated: {w: params.w - alpha * gradient.w, rho: params.rho - alpha * gradient.rho}};
}

export function replayVariants(cue = 1) {
  if (![-1, 1].includes(cue)) throw new RangeError('cue must be ±1');
  const observations = [cue, 0, 0, 0, 0];
  const old = unroll(observations, settings.old), current = unroll(observations, settings.current);
  const target = unroll(observations, settings.target);
  const variants = {
    full: current,
    prefixDetached: unroll(observations, settings.current, {detachAt: [settings.learnStart]}),
    storedBurnDetached: unroll(observations.slice(2), settings.current, {initialState: old[1].h, detachAt: [2]}),
    storedBurnAttached: unroll(observations.slice(2), settings.current, {initialState: old[1].h}),
    zeroBurn: unroll(observations.slice(2), settings.current, {detachAt: [2]}),
    zeroAtLearn: unroll(observations.slice(4), settings.current),
    storedAtLearn: unroll(observations.slice(4), settings.current, {initialState: old[3].h}),
  };
  const parcel = cue === 1 ? 'L' : 'R', recordedAction = greedyDoor(old[4].h);
  const reward = parcelReward(parcel, recordedAction);
  const results = Object.fromEntries(Object.entries(variants).map(([name, rows]) => {
    const last = rows.at(-1), action = greedyDoor(last.h);
    return [name, {rows, h: last.h, dw: last.dw, drho: last.drho, q: doorQ(last.h),
      action, freshReward: parcelReward(parcel, action), stateError: Math.abs(last.h - current[4].h),
      sample: terminalLoss(last, recordedAction, reward)}];
  }));
  return {cue, observations, old, current, target, recordedAction, recordedReward: reward,
    oldAction: recordedAction, currentAction: greedyDoor(current[4].h),
    oldFreshReward: reward, currentFreshReward: parcelReward(parcel, greedyDoor(current[4].h)),
    results, bootstrap: {prediction: forwardQ(current[3].h),
      ...tdTarget({reward: 0, terminated: false, nextOnlineState: current[4].h, nextTargetState: target[4].h}),
      wrongSharedStateTarget: tdTarget({reward: 0, terminated: false, nextOnlineState: current[4].h, nextTargetState: current[4].h}).y},
  };
}

/** Compare initializations through the same grey suffix, holding parameters fixed. */
export function initialStateErrors(rho, steps = 8) {
  const params = {w: .8, rho}, currentInitial = Math.tanh(.8), oldInitial = -currentInitial;
  const current = [currentInitial, ...unroll(Array(steps).fill(0), params, {initialState: currentInitial}).map(r => r.h)];
  const stored = [oldInitial, ...unroll(Array(steps).fill(0), params, {initialState: oldInitial}).map(r => r.h)];
  const zero = [0, ...unroll(Array(steps).fill(0), params).map(r => r.h)];
  return current.map((h, B) => ({B, reference: h, stored: stored[B], zero: zero[B],
    storedError: Math.abs(h - stored[B]), zeroError: Math.abs(h - zero[B]),
    storedAction: greedyDoor(stored[B]), referenceAction: greedyDoor(h)}));
}

/** A sampled chunk boundary keeps memory/continuation; an actual reset changes memory. */
export function boundaryData() {
  const chunk = unroll([1, 0, 0, 0, 0], settings.current, {detachAt: [2]});
  const falseReset = unroll([1, 0, 0, 0, 0], settings.current, {resetBefore: [2]});
  const newEpisode = unroll([1, 0, -1, 0], settings.current, {resetBefore: [2]});
  const next = replayVariants().current[4].h, target = replayVariants().target[4].h;
  return {chunk, falseReset, newEpisode,
    chunkTarget: tdTarget({reward: 0, terminated: false, nextOnlineState: next, nextTargetState: target}).y,
    falseTerminalTarget: tdTarget({reward: 0, terminated: true}).y,
    realTerminalTarget: tdTarget({reward: -2, terminated: true}).y};
}

export function recurrentReplayData() {
  const branches = [-1, 1].map(replayVariants);
  return {settings, positive: branches[1], branches,
    initialErrors: {contractive: initialStateErrors(.8), noncontractive: initialStateErrors(1.2)},
    missingCue: {observedSuffix: [0, 0, 0], zeroStates: branches.map(b => b.results.zeroBurn.h),
      zeroActions: branches.map(b => b.results.zeroBurn.action),
      meanFreshReward: branches.reduce((s, b) => s + .5 * b.results.zeroBurn.freshReward, 0),
      fullMeanFreshReward: branches.reduce((s, b) => s + .5 * b.results.full.freshReward, 0)},
    boundaries: boundaryData(),
    scope: 'Fixed nonlinear recurrent Q mechanism; old/current/target weights are given; no random training or benchmark reproduction.',
  };
}
