/** Original teaching calculations. These are not participant or neural data. */
export function gaussianFusion({visual=52,haptic=56,visualSD=1,hapticSD=3}={}) {
 if(![visual,haptic,visualSD,hapticSD].every(Number.isFinite)||visualSD<=0||hapticSD<=0)throw new RangeError('finite measurements and positive standard deviations required');
 const visualPrecision=1/visualSD**2,hapticPrecision=1/hapticSD**2;
 const weight=visualPrecision/(visualPrecision+hapticPrecision);
 return {visual,haptic,visualSD,hapticSD,weight,mean:weight*visual+(1-weight)*haptic,variance:1/(visualPrecision+hapticPrecision)};
}
export function normalDensity(x,mean,sd) {
 if(![x,mean,sd].every(Number.isFinite)||sd<=0)throw new RangeError('positive finite Gaussian scale required');
 return Math.exp(-.5*((x-mean)/sd)**2)/(sd*Math.sqrt(2*Math.PI));
}
export function responsePatterns() {
 // Toy significant-source masks. They illustrate spread and differentiation;
 // no entropy normalization, Lempel–Ziv estimate, or clinical PCI is computed.
 const local=[[1,1,1,0,0,0,0,0],...Array.from({length:5},()=>Array(8).fill(0))];
 const repeated=Array.from({length:6},()=>[1,1,0,0,1,0,0,0]);
 const diverse=[
  [1,1,0,0,1,0,0,0],[0,1,1,0,0,1,0,0],
  [0,0,1,1,0,0,1,0],[0,0,0,1,1,0,0,1],
  [1,0,0,0,1,1,0,0],[0,1,0,0,0,1,1,0],
 ];
 return [local,repeated,diverse].map((mask,i)=>({
  id:['local','repeated','diverse'][i],mask,
  activeSources:mask.filter(r=>r.some(Boolean)).length,
  activeBins:mask.flat().reduce((a,b)=>a+b,0),
  distinctActiveRows:new Set(mask.filter(r=>r.some(Boolean)).map(r=>r.join(''))).size,
 }));
}
export function causalComparison({inputProbability=.5,intervention=null}={}) {
 if(!Number.isFinite(inputProbability)||inputProbability<0||inputProbability>1)throw new RangeError('input probability must be in [0,1]');
 if(intervention!==null&&intervention!==0&&intervention!==1)throw new RangeError('intervention must be null, 0 or 1');
 return ['common','chain'].map(model=>{
  const outcomes=[0,1].map(u=>{
   const x=intervention===null?u:intervention;
   const y=model==='common'?u:x;
   return {u,x,y,p:u?inputProbability:1-inputProbability};
  });
  return {model,outcomes,meanY:outcomes.reduce((s,o)=>s+o.p*o.y,0)};
 });
}
export function computeMindInference() {
 return {
  provenance:'Original explanatory examples, not empirical recordings.',
  fusion:[gaussianFusion(),gaussianFusion({visualSD:4})],
  responses:responsePatterns(),
  observation:causalComparison(),intervention:causalComparison({intervention:0}),
 };
}
