/**
 * Deterministic cue-integration teaching calculations.
 * Protocol metadata comes from Ernst & Banks (2002), pp. 429–433, Fig. 1–3,
 * Methods. Every numerical curve below is a constructed example, not human data.
 * Heights / standard deviations are in mm; variances / covariance in mm^2.
 */
const finite=(v,label)=>{if(!Number.isFinite(v))throw new TypeError(label+' must be finite');return v;};
const positive=(v,label)=>{finite(v,label);if(v<=0)throw new RangeError(label+' must be positive');return v;};

export const cueIntegrationProtocol=Object.freeze({
  source:'Ernst & Banks (2002), Nature 415, 429–433, Fig. 1–3 and Methods',
  sourceURL:'https://www.cns.nyu.edu/~david/courses/perceptionGrad/Readings/ErnstBanks-Nature2002.pdf',
  kind:'schematic',
  observers:4,
  apparatus:{visual:'CRT reflected in opaque mirror; binocular shutter glasses',haptic:'two PHANToM force-feedback devices; unseen right index finger and thumb'},
  bar:{widthMM:150,standardHeightMM:55,depthStepMM:30,positionRandomized:true},
  visualNoise:{distribution:'uniform depth displacement parallel to line of sight',rangeAsPercentOfDepthStep:[0,67,133,200],newDotsEachPresentation:true},
  hapticNoiseAdded:false,
  procedure:{task:'two-interval forced choice: select taller bar',intervalOrderRandomized:true,presentationSeconds:1,feedback:false,
    singleModalityBeforeFraction:.5,singleModalityAfterFraction:.5,
    combinedOnset:'both fingertips contact; visual and haptic presentation end together'},
  joint:{standardMeanMM:55,conflictMM:[-6,-3,0,3,6],comparisonHeightsRangeMM:[47,63],comparisonConflictMM:0,conflictAndNoiseRandomized:true},
  measuredPsychometricData:null,
});

export function gaussianDensity(z,mean,sigma){
  finite(z,'height');finite(mean,'mean');positive(sigma,'sigma');
  return Math.exp(-.5*((z-mean)/sigma)**2)/(sigma*Math.sqrt(2*Math.PI));
}

/** Standard normal CDF (absolute approximation error < 8e-8). */
export function normalCDF(z){
  finite(z,'z');
  if(z===0)return .5;
  const x=Math.abs(z),t=1/(1+.2316419*x);
  const tail=gaussianDensity(x,0,1)*t*(.319381530+t*(-.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))));
  return z>0?1-tail:tail;
}

/**
 * Best unbiased linear combination under a positive-definite covariance matrix.
 * A correlated optimum need not lie in [0,1]; no clipping is imposed here.
 */
export function minimumVarianceFusion({visual,haptic,sigmaVisual,sigmaHaptic,covariance=0}){
  finite(visual,'visual reading');finite(haptic,'haptic reading');
  positive(sigmaVisual,'visual sigma');positive(sigmaHaptic,'haptic sigma');finite(covariance,'covariance');
  const vv=sigmaVisual**2,vh=sigmaHaptic**2;
  if(Math.abs(covariance)>=sigmaVisual*sigmaHaptic)throw new RangeError('covariance matrix must be positive definite');
  const denominator=vv+vh-2*covariance;
  const weightVisual=(vh-covariance)/denominator,weightHaptic=1-weightVisual;
  return {visual,haptic,sigmaVisual,sigmaHaptic,covariance,weightVisual,weightHaptic,
    mean:weightVisual*visual+weightHaptic*haptic,
    variance:(vv*vh-covariance**2)/denominator,
    sigma:Math.sqrt((vv*vh-covariance**2)/denominator)};
}

export function linearFusionVariance(w,sigmaVisual,sigmaHaptic,covariance=0){
  finite(w,'weight');positive(sigmaVisual,'visual sigma');positive(sigmaHaptic,'haptic sigma');finite(covariance,'covariance');
  if(Math.abs(covariance)>=sigmaVisual*sigmaHaptic)throw new RangeError('covariance matrix must be positive definite');
  return w*w*sigmaVisual**2+(1-w)**2*sigmaHaptic**2+2*w*(1-w)*covariance;
}

/** Two independent intervals, each with equal single-presentation variance. */
export function psychometricProbability(comparison,{pse=55,sigma}={}){
  finite(comparison,'comparison');finite(pse,'PSE');positive(sigma,'sigma');
  return normalCDF((comparison-pse)/(Math.SQRT2*sigma));
}
export function psychometricThreshold(sigma){return Math.SQRT2*positive(sigma,'sigma');}

export function standardWithConflict(delta,mean=55){
  finite(delta,'conflict');finite(mean,'mean');
  return {visual:mean+delta/2,haptic:mean-delta/2,delta,mean};
}
export function conflictPSE(delta,weightVisual,mean=55){
  finite(weightVisual,'visual weight');const s=standardWithConflict(delta,mean);
  return weightVisual*s.visual+(1-weightVisual)*s.haptic;
}

export function cueIntegrationWalkthrough(){
  const conditions=[1,4].map(sigmaVisual=>{
    const f=minimumVarianceFusion({visual:52,haptic:56,sigmaVisual,sigmaHaptic:3});
    const densities=Array.from({length:261},(_,i)=>{
      const height=42+i*.1;
      return {height,visual:gaussianDensity(height,52,sigmaVisual),haptic:gaussianDensity(height,56,3),combined:gaussianDensity(height,f.mean,f.sigma)};
    });
    return {...f,densities,thresholdVisual:psychometricThreshold(sigmaVisual),thresholdHaptic:psychometricThreshold(3),thresholdCombined:psychometricThreshold(f.sigma),
      conflictPredictions:cueIntegrationProtocol.joint.conflictMM.map(delta=>({delta,...standardWithConflict(delta),pse:conflictPSE(delta,f.weightVisual)}))};
  });
  const psychometrics=[{id:'visual-clear',sigma:1},{id:'haptic',sigma:3},{id:'visual-noisy',sigma:4}].map(c=>({...c,pse:55,threshold:psychometricThreshold(c.sigma),
    points:Array.from({length:261},(_,i)=>{const height=42+i*.1;return {height,probability:psychometricProbability(height,c)};})}));
  return {kind:'exact',provenance:'Constructed independent Gaussian calculations, not participant data',heightUnit:'mm',densityUnit:'1/mm',
    densityDomains:{height:[42,68],density:[0,.5]},psychometricDomains:{height:[42,68],probability:[0,1]},
    conditions,psychometrics,protocol:cueIntegrationProtocol};
}
