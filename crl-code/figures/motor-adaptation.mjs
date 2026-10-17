/** Deterministic teaching calculations; no participant data, fitting or training. */
export const motorSources = Object.freeze({
  smith: 'https://doi.org/10.1371/journal.pbio.0040179',
  heald: 'https://doi.org/10.1038/s41586-021-04129-3',
});

export function compensationSlope(ideal, measured) {
  if (ideal.length !== measured.length || !ideal.length) throw new RangeError('matching nonempty force samples required');
  let numerator = 0, denominator = 0;
  ideal.forEach((v, i) => {
    if (!Number.isFinite(v) || !Number.isFinite(measured[i])) throw new TypeError('finite samples required');
    numerator += v * measured[i]; denominator += v * v;
  });
  if (!denominator) throw new RangeError('nonzero ideal force required');
  return numerator / denominator;
}

export function motorAdaptationData() {
  const duration = .5, distance = .12, gain = 15;
  const forceProfiles = Array.from({length: 101}, (_, k) => {
    const time = duration * k / 100;
    const velocity = distance * Math.PI / (2 * duration) * Math.sin(Math.PI * time / duration);
    return {time, velocity, distance: distance * (1 - Math.cos(Math.PI * time / duration)) / 2,
      ideal: gain * velocity, noCompensation: 0, retainedCompensation: .6 * gain * velocity};
  });
  // Smith et al. chose these retention factors for their schematic Figs 1–2,
  // not as the empirical best fit. Initial states here are independently chosen.
  const retention = {fast: .92, slow: .996}, initial = {fast: -.4, slow: .4};
  const rebound = Array.from({length: 151}, (_, k) => {
    const fast = initial.fast * retention.fast ** k, slow = initial.slow * retention.slow ** k;
    return {trial: k, fast, slow, total: fast + slow,
      equalRateFast: initial.fast * retention.slow ** k, equalRateSlow: slow, equalRateTotal: 0};
  });
  const prior = .5, cueReliability = .8, feedbackReliability = .9;
  const responsibility = [{cue: 1, feedback: 1}, {cue: 2, feedback: 1},
    {cue: 1, feedback: -1}, {cue: 2, feedback: -1}].map(({cue, feedback}) => {
      const cue1 = cue === 1 ? cueReliability : 1 - cueReliability;
      const feed1 = feedback === 1 ? feedbackReliability : 1 - feedbackReliability;
      const unnormalized1 = prior * cue1 * feed1;
      const unnormalized2 = (1 - prior) * (1 - cue1) * (1 - feed1);
      return {cue, feedback, probability: unnormalized1 / (unnormalized1 + unnormalized2)};
    });
  return {
    kind: 'independent-deterministic-teaching-calculations', sources: motorSources,
    apparatus: {duration, distance, gain, specifiedCompensation: .6, forceProfiles},
    recovery: {retention, initial, errorClampedTo: 0, rebound},
    inference: {prior, cueReliability, feedbackReliability, responsibility},
    protocol: {source: motorSources.heald, phases: [
      {id: 'null', label: '空场', trials: 50}, {id: 'original', label: '原场', trials: 125},
      {id: 'opposite', label: '反场', trials: 15}, {id: 'channel', label: '通道', trials: 150},
    ], channelProbesInFirstTwoPhases: true, evokerChannelTrialNumbers: [3, 4]},
  };
}
