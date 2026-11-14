/**
 * Independent teaching calculations for learning-experiments figures.
 * No animal measurements or fitted parameters are stored here.
 * Sources for the experimental designs (not these numerical examples):
 * Steinberg et al. (2013), doi:10.1038/nn.3413, Figures 1–2 and Methods.
 * Hattori et al. (2023), doi:10.1038/s41593-023-01485-3, Figures 1–3 and 5.
 */
export function baitAvailability(assignmentProbability, unvisitedTrials) {
  if (!Number.isFinite(assignmentProbability) || assignmentProbability < 0 || assignmentProbability > 1)
    throw new RangeError('assignment probability must be in [0, 1]');
  if (!Number.isInteger(unvisitedTrials) || unvisitedTrials < 0)
    throw new RangeError('unvisited trials must be a nonnegative integer');
  // Initially empty; independent assignment opportunities; a single available
  // reward persists without accumulating extra portions until collection.
  return 1 - (1 - assignmentProbability) ** unvisitedTrials;
}

export function cueResponse({windowSeconds, baselineSeconds, cueSeconds}) {
  if (!Number.isFinite(windowSeconds) || windowSeconds <= 0)
    throw new RangeError('observation window must be positive');
  if (![baselineSeconds, cueSeconds].every(v => Number.isFinite(v) && v >= 0 && v <= windowSeconds))
    throw new RangeError('port occupancy must fit the observation window');
  return {
    differenceSeconds: cueSeconds - baselineSeconds,
    differencePercentagePoints: 100 * (cueSeconds - baselineSeconds) / windowSeconds,
  };
}

export function learningExperimentsData() {
  const occupancy = {windowSeconds: 30, baselineSeconds: 6, cueSeconds: 18};
  return {
    kind: 'independent-teaching-calculations',
    occupancy: {...occupancy, ...cueResponse(occupancy)},
    baiting: {
      assignmentProbability: .1,
      unvisitedTrials: [1, 2, 3],
      availability: [1, 2, 3].map(n => baitAvailability(.1, n)),
      initialRewardAvailable: false,
      extraPortionsAccumulate: false,
    },
  };
}
