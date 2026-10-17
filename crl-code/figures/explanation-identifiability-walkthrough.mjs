/** Controlled feedback, exact probabilities, and finite-count likelihoods.
 * Original teaching example; no participant data and no sampled training.
 * Every probe belongs to a separate run initialized at Q(L)=Q(R)=0.
 */
const finite = (x, name) => {if (!Number.isFinite(x)) throw new TypeError(`${name} must be finite`);};
export function sigmoid(z) {
  finite(z, 'logit');
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}
export function logit(p) {
  if (!(p > 0 && p < 1)) throw new RangeError('probability must lie strictly between zero and one');
  return Math.log(p) - Math.log1p(-p);
}
export function feedbackValues(alpha, n) {
  if (!(alpha > 0 && alpha <= 1)) throw new RangeError('alpha must lie in (0,1]');
  if (!Number.isInteger(n) || n < 0) throw new RangeError('feedback count must be a nonnegative integer');
  const values = [0];
  for (let t = 0; t < n; t++) values.push(values.at(-1) + alpha * (1 - values.at(-1)));
  return values;
}
export function controlledRun(alpha, beta, n) {
  finite(beta, 'beta');
  if (!(beta > 0)) throw new RangeError('beta must be positive');
  const values = feedbackValues(alpha, n);
  return {n, alpha, beta, initial: [0, 0], forcedAction: 'L', rewards: Array(n).fill(1),
    values, rightValue: 0, probe: {feedback: false, update: false, qBefore: values.at(-1),
      qAfter: values.at(-1), probabilityLeft: sigmoid(beta * values.at(-1))}};
}
export function rewardSequencePrediction(alpha, beta, rewards) {
  feedbackValues(alpha, 0); finite(beta, 'beta');
  if (!(beta > 0) || !Array.isArray(rewards) || !rewards.length || !rewards.every(r => r === 0 || r === 1)) throw new RangeError('positive beta and a nonempty binary reward sequence required');
  let q = 0;
  const values = [q];
  for (const reward of rewards) {q += alpha * (reward - q); values.push(q);}
  return {rewards: rewards.slice(), values, probabilityLeft: sigmoid(beta * q)};
}
/** Invert oracle probabilities. Empirical frequencies need not be feasible. */
export function identifyFromProbabilities(p1, p2) {
  if (!(p1 > .5 && p1 < 1 && p2 > .5 && p2 < 1)) return {identified: false, reason: 'probability boundary'};
  const z1 = logit(p1), z2 = logit(p2), ratio = z2 / z1;
  if (!(ratio >= 1 && ratio < 2)) return {identified: false, reason: 'outside the stipulated model', ratio};
  const alpha = 2 - ratio;
  return {identified: true, alpha, beta: z1 / alpha, ratio, z1, z2};
}
function validateCounts(rows) {
  if (!Array.isArray(rows) || !rows.length) throw new TypeError('count rows are required');
  for (const {n, total, left} of rows) if (!Number.isInteger(n) || n < 1 || !Number.isInteger(total) || total < 1 || !Number.isInteger(left) || left < 0 || left > total) throw new RangeError('invalid feedback/count row');
}
const softplus = z => Math.max(0, z) + Math.log1p(Math.exp(-Math.abs(z)));
/** Omits the binomial coefficients, which are constant across parameters. */
export function countLogLikelihood(alpha, beta, rows) {
  validateCounts(rows); finite(beta, 'beta');
  if (beta < 0) throw new RangeError('beta must be nonnegative; zero is the fitting boundary');
  return rows.reduce((ll, row) => {
    const z = beta * feedbackValues(alpha, row.n).at(-1);
    return ll - row.left * softplus(-z) - (row.total - row.left) * softplus(z);
  }, 0);
}
/** Concavity in beta makes this score-root search global for each fixed alpha. */
export function profileAtAlpha(alpha, rows) {
  validateCounts(rows);
  const q = rows.map(row => feedbackValues(alpha, row.n).at(-1));
  const score = beta => rows.reduce((s, row, i) => s + q[i] * (row.left - row.total * sigmoid(beta * q[i])), 0);
  if (score(0) <= 0) return {alpha, beta: 0, logLikelihood: countLogLikelihood(alpha, 0, rows), boundary: 'beta=0 limit'};
  if (rows.every(row => row.left === row.total)) return {alpha, beta: null, logLikelihood: 0, boundary: 'beta infinite supremum'};
  let high = 1;
  while (score(high) > 0) high *= 2;
  let low = 0;
  for (let i = 0; i < 100; i++) {const mid = (low + high) / 2; if (score(mid) > 0) low = mid; else high = mid;}
  const beta = (low + high) / 2;
  return {alpha, beta, logLikelihood: countLogLikelihood(alpha, beta, rows), boundary: null};
}
export function binomialProbabilities(total, probability) {
  if (!Number.isInteger(total) || total < 1) throw new RangeError('total must be a positive integer');
  if (!(probability > 0 && probability < 1)) throw new RangeError('probability must be interior');
  let logCoefficient = 0;
  return Array.from({length: total + 1}, (_, k) => {
    if (k) logCoefficient += Math.log(total - k + 1) - Math.log(k);
    return Math.exp(logCoefficient + k * Math.log(probability) + (total - k) * Math.log1p(-probability));
  });
}
/** A prespecified classifier for TWO FIXED generators, not fitted model families.
 * Rows are generating rules, columns are selected rules. Integrate all counts.
 */
export function exactModelRecovery(total = 20, threshold = 2) {
  if (!Number.isInteger(threshold)) throw new RangeError('threshold must be an integer');
  const p1 = sigmoid(1), p2 = sigmoid(1.8);
  const generators = [{id: 'accumulating', p1, p2}, {id: 'first-feedback-latch', p1, p2: p1}];
  const confusion = generators.map(generator => {
    const first = binomialProbabilities(total, generator.p1), second = binomialProbabilities(total, generator.p2);
    const mass = [0, 0];
    for (let k1 = 0; k1 <= total; k1++) for (let k2 = 0; k2 <= total; k2++) mass[k2 - k1 >= threshold ? 0 : 1] += first[k1] * second[k2];
    return mass;
  });
  return {totalPerCondition: total, threshold, rule: 'select accumulating iff k2-k1 >= threshold', generators, confusion,
    sampledRuns: 0, interpretation: 'exact count-distribution recovery for two fixed generators only'};
}
export function identifiabilityWalkthrough() {
  const candidates = [{id: 'A', alpha: .2, beta: 5}, {id: 'B', alpha: .5, beta: 2}];
  const counts = [{n: 1, total: 100, left: 73}, {n: 2, total: 100, left: 86}];
  const fitted = identifyFromProbabilities(.73, .86);
  const maximum = countLogLikelihood(fitted.alpha, fitted.beta, counts);
  const profile = Array.from({length: 100}, (_, i) => {
    const alpha = (i + 1) / 100, row = profileAtAlpha(alpha, counts);
    return {...row, gap: maximum - row.logLikelihood, singleConditionGap: 0};
  });
  return {kind: 'original exact teaching example', participantCount: 0, sampledRuns: 0, randomGeneratorUsed: false,
    protocol: {resetEveryRun: true, initialValues: [0, 0], probePerRun: 1, probeFeedback: false,
      conditions: [1, 2], independentUnit: 'one reset-feedback-probe run', stableParametersAssumed: true},
    candidates: candidates.map(candidate => ({...candidate, runs: [1, 2].map(n => controlledRun(candidate.alpha, candidate.beta, n)),
      logLikelihood: countLogLikelihood(candidate.alpha, candidate.beta, counts)})),
    counts, countOrigin: 'fixed illustrative counts, not sampled or observed participants', fitted,
    maximumLogLikelihood: maximum, profile, modelRecovery: exactModelRecovery(),
    distinguishingSequence: {rewards: [1, 0], accumulating: rewardSequencePrediction(.2, 5, [1, 0]),
      latchProbabilityLeft: sigmoid(1), boundaryAccumulating: rewardSequencePrediction(1, 1, [1, 0])},
    probabilityGeometry: {p1Range: [.5, .97], alphaSlices: [1, .5, .2, .02],
      upperBoundary: 'p2 = sigmoid(2 logit(p1)); alpha=0 is excluded',
      lowerBoundary: 'p2 = p1; alpha=1 intersects the first-feedback latch'}};
}
