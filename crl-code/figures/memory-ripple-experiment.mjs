/**
 * Source-grounded protocol and explicit teaching calculations, not animal data.
 * The synthetic signal below illustrates timing only. Its amplitude, shape,
 * duration and the chosen 100 ms delay are not extracted LFP measurements.
 */
export const memoryRippleSources = Object.freeze({
  girardeau: 'https://www.nature.com/articles/nn.2384',
  mainText: 'https://buzsakilab.com/content/PDFs/Girardeau2009NatNeurosci.pdf',
  supplement: 'https://buzsakilab.com/content/PDFs/Girardeau2009NatNeurosciSupp.pdf',
  mattarDaw: 'https://doi.org/10.1038/s41593-018-0232-z',
});

export function memoryRippleProtocol() {
  return {
    kind: 'original-protocol-schematic', sources: memoryRippleSources,
    task: {arms: 8, baitedArmCount: 3, trialsPerDay: 3, maxTrialMinutes: 3,
      interTrialRestMinutes: 3, postTrainingRestMinutes: 60,
      referenceCues: 'room-wall visual cues', restSite: 'flowerpot at maze center'},
    recording: {site: 'CA1 pyramidal layer', onlineDetector: 'ripple-band threshold crossing',
      onlineDetectionPercent: 86, onlineDetectionSemPercent: 1.3,
      offlineBandHz: [100, 200], offlinePeakThresholdSd: 5, offlineMaxDurationMs: 100},
    stimulation: {site: 'ventral hippocampal commissure', pulseMs: .5, maxPulsesPerSecond: 5,
      delayRangeMs: [80, 120]},
    groups: [
      {id: 'immediate', animals: 7, implanted: true, delayMs: 0},
      {id: 'delayed', animals: 7, implanted: true, delayRangeMs: [80, 120]},
      {id: 'unimplanted', animals: 12, implanted: false, delayMs: null},
    ],
    drawing: {kind: 'schematic', timeDomainMs: [-40, 140],
      syntheticEventStartMs: -18, syntheticEventEndMs: 54,
      illustratedDelayMs: 100, baitedArmIndices: [0, 4, 5]},
  };
}

export function schematicRippleTrace({interrupted = false} = {}) {
  const {drawing: d} = memoryRippleProtocol();
  return Array.from({length: 361}, (_, i) => {
    const timeMs = d.timeDomainMs[0] + i / 2;
    const active = timeMs >= d.syntheticEventStartMs && timeMs <= d.syntheticEventEndMs
      && (!interrupted || timeMs < 0);
    const phase = (timeMs - d.syntheticEventStartMs) /
      (d.syntheticEventEndMs - d.syntheticEventStartMs);
    const amplitude = active ? Math.sin(Math.PI * phase) * Math.sin(2 * Math.PI * timeMs / 8) : 0;
    return [timeMs, amplitude];
  });
}

/** Estimated local gain: evaluate both policies with the same updated Q vector. */
export function estimatedLocalGain(qAfter, policyBefore, policyAfter) {
  if (!qAfter.length || policyBefore.length !== qAfter.length || policyAfter.length !== qAfter.length
    || !qAfter.every(Number.isFinite)) throw new TypeError('matching finite action vectors required');
  for (const policy of [policyBefore, policyAfter]) {
    if (!policy.every(p => Number.isFinite(p) && p >= 0 && p <= 1)
      || Math.abs(policy.reduce((a, b) => a + b, 0) - 1) > 1e-10)
      throw new RangeError('policy must be a probability distribution');
  }
  return qAfter.reduce((sum, q, a) => sum + q * (policyAfter[a] - policyBefore[a]), 0);
}
