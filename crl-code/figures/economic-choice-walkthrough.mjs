/** Exact lottery arithmetic and deliberately constructed mechanisms, not behavior fitting. */
export const source = Object.freeze({
  title: 'Kahneman & Tversky (1979), Prospect Theory',
  url: 'https://www.dmi-ida.org/download-pdf/pdf/Kahneman-ProspectTheoryAnalysis-1979.pdf',
  doi: 'https://doi.org/10.2307/1914185',
  locations: {procedure: 'pp. 264–265', commonConsequence: 'Problems 1–2, pp. 265–266', commonRatio: 'Problems 3–4, p. 266', reference: 'Problems 11–12, p. 273', weights: 'pp. 280–282'},
});

// Reported questionnaire percentages are stored as source facts, never computed.
export const observations = Object.freeze({
  evidence: 'reported hypothetical questionnaire choices',
  commonConsequence: {n: 72, p1B: 82, p2C: 83, reportedJointBC: 61, validJointBC: null, jointStatus: 'printed joint percentage conflicts with printed marginals, even allowing rounding; not a valid frequency', sameRespondents: true},
  commonRatio: {n: 95, p3B: 80, p4C: 65, sameRespondents: true, jointPercentage: null},
  reference: {p11N: 70, p11B: 84, p12N: 68, p12C: 69, sameRespondents: false},
});

// [money in Israeli pounds, probability]. The ordering exposes the common .66 branch.
export const lotteries = Object.freeze({
  p1A: [[2500, .33], [2400, .66], [0, .01]],
  p1B: [[2400, .33], [2400, .66], [2400, .01]],
  p2C: [[2500, .33], [0, .66], [0, .01]],
  p2D: [[2400, .33], [0, .66], [2400, .01]],
  p3A: [[4000, .8], [0, .2]], p3B: [[3000, 1]],
  p4C: [[4000, .2], [0, .8]], p4D: [[3000, .25], [0, .75]],
  p11A: [[1000, .5], [0, .5]], p11B: [[500, 1]],
  p12C: [[-1000, .5], [0, .5]], p12D: [[-500, 1]],
});

export function expected(lottery, utility = x => x) {
  const mass = lottery.reduce((s, [, p]) => s + p, 0);
  if (lottery.some(([x, p]) => !Number.isFinite(x) || !Number.isFinite(p) || p < 0) || Math.abs(mass - 1) > 1e-12) throw new RangeError('Lottery probabilities must sum to one');
  return lottery.reduce((s, [x, p]) => s + p * utility(x), 0);
}

// A transparent monotone, piecewise linear teaching example, not the paper's fit.
export const weightNodes = Object.freeze([[0, 0], [.2, .25], [.25, .28], [.8, .65], [1, 1]]);
export function weight(p) {
  if (!Number.isFinite(p) || p < 0 || p > 1) throw new RangeError('p must lie in [0,1]');
  for (let i = 1; i < weightNodes.length; i++) {
    const [x1, y1] = weightNodes[i - 1], [x2, y2] = weightNodes[i];
    if (p <= x2) return y1 + (p - x1) * (y2 - y1) / (x2 - x1);
  }
  return 1;
}

// x is a change in thousands of pounds. Only the reference mechanism is illustrated.
export function referenceValue(x) { return x >= 0 ? Math.sqrt(x) : -2 * Math.sqrt(-x); }

export function economicChoiceData() {
  const means = Object.fromEntries(Object.entries(lotteries).map(([name, lottery]) => [name, expected(lottery)]));
  const weighted = {p3A: weight(.8), p3B: .75, p4C: weight(.2), p4D: weight(.25) * .75};
  const referenceScores = Object.fromEntries(['p11A', 'p11B', 'p12C', 'p12D'].map(name => [name, expected(lotteries[name], x => referenceValue(x / 1000))]));
  return {
    evidence: 'exact finite-lottery enumeration and constructed illustrations', source, observations, lotteries, means,
    constructed: {weightNodes, weighted, referenceScores, valueFunction: 'sqrt(x) for x>=0; -2 sqrt(-x) otherwise; x in thousands of pounds'},
    probabilityCurve: Array.from({length: 101}, (_, i) => [i / 100, weight(i / 100)]),
    valueCurve: Array.from({length: 201}, (_, i) => [(i - 100) / 100, referenceValue((i - 100) / 100)]),
  };
}
