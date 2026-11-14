/** Known two-step settlement task; exact mechanisms, no sampled training. */
export const SUPPORT=Object.freeze([0,1,2,3,4]);
export const GAMMA=.5;
const finite=(v,name)=>{if(!Number.isFinite(v))throw new RangeError(name+' must be finite');return v;};
function distribution(atoms,masses){
  if(!Array.isArray(atoms)||!Array.isArray(masses)||!atoms.length||atoms.length!==masses.length)throw new RangeError('matching nonempty atoms and masses required');
  atoms.forEach(v=>finite(v,'atom'));masses.forEach(v=>{finite(v,'mass');if(v<0)throw new RangeError('negative mass');});
  if(Math.abs(masses.reduce((a,b)=>a+b,0)-1)>1e-10)throw new RangeError('masses must sum to one');
}
export function expectation(atoms,masses){distribution(atoms,masses);return atoms.reduce((sum,z,i)=>sum+z*masses[i],0);}
export function lowerCVaR(atoms,masses,eta=.5){
  distribution(atoms,masses);finite(eta,'tail mass');if(eta<=0||eta>1)throw new RangeError('tail mass must lie in (0,1]');
  let remaining=eta,total=0;
  for(const [z,p] of atoms.map((z,i)=>[z,masses[i]]).sort((a,b)=>a[0]-b[0])){
    const used=Math.min(remaining,p);total+=z*used;remaining-=used;if(remaining<=1e-14)break;
  }
  return total/eta;
}
/** Tent-kernel form of the categorical projection; exact grid points retain mass. */
export function categoricalTarget({reward,gamma=GAMMA,terminal=false,nextAtoms=SUPPORT,nextMasses,support=SUPPORT}){
  finite(reward,'reward');finite(gamma,'discount');if(gamma<0||gamma>1)throw new RangeError('discount outside [0,1]');
  if(typeof terminal!=='boolean')throw new RangeError('terminal must be boolean');
  distribution(nextAtoms,nextMasses);
  if(!Array.isArray(support)||support.length<2)throw new RangeError('at least two support points required');
  support.forEach(v=>finite(v,'support'));const delta=support[1]-support[0];
  if(delta<=0||support.some((v,i)=>Math.abs(v-support[0]-i*delta)>1e-10))throw new RangeError('uniform increasing support required');
  const raw=nextAtoms.map(z=>reward+gamma*(terminal?0:1)*z);
  const clipped=raw.map(z=>Math.max(support[0],Math.min(support.at(-1),z)));
  const projected=support.map(z=>clipped.reduce((sum,y,j)=>sum+nextMasses[j]*Math.max(0,1-Math.abs(y-z)/delta),0));
  return {raw,clipped,next_masses:[...nextMasses],projected,raw_mean:expectation(raw,nextMasses),projected_mean:expectation(support,projected),clipped_mass:nextMasses.reduce((sum,p,j)=>sum+(raw[j]!==clipped[j]?p:0),0)};
}
export function softmax(logits){
  if(!Array.isArray(logits)||!logits.length)throw new RangeError('nonempty logits required');logits.forEach(v=>finite(v,'logit'));
  const largest=Math.max(...logits),weights=logits.map(v=>Math.exp(v-largest)),sum=weights.reduce((a,b)=>a+b,0);return weights.map(v=>v/sum);
}
export function categoricalLoss(logits,target){const p=softmax(logits);distribution(logits,target);return -target.reduce((sum,m,i)=>sum+(m?m*Math.log(p[i]):0),0);}
export function logitStep(logits,target,rate=1){
  finite(rate,'rate');if(rate<0)throw new RangeError('nonnegative rate required');
  const before=softmax(logits);distribution(logits,target);const gradient=before.map((p,i)=>p-target[i]);
  const afterLogits=logits.map((v,i)=>v-rate*gradient[i]);
  return {rate,before_logits:[...logits],target:[...target],before,gradient,after_logits:afterLogits,after:softmax(afterLogits),before_loss:categoricalLoss(logits,target),after_loss:categoricalLoss(afterLogits,target)};
}
export function settlementLaw(action,{advance=.5,gamma=GAMMA,highProbability=.5}={}){
  finite(advance,'advance');finite(gamma,'discount');finite(highProbability,'probability');
  if(gamma<0||gamma>1||highProbability<0||highProbability>1)throw new RangeError('discount and probability must lie in [0,1]');
  if(action==='fixed')return {atoms:[advance+gamma*2],masses:[1]};
  if(action==='floating')return {atoms:[advance,advance+gamma*4],masses:[1-highProbability,highProbability]};
  throw new RangeError('unknown action');
}
export function chooseSettlement({floatingAdvance=.5,criterion='mean',eta=.5,tie='fixed'}={}){
  if(!['mean','lower_cvar'].includes(criterion)||!['fixed','floating'].includes(tie))throw new RangeError('unknown criterion or tie action');
  const laws={fixed:settlementLaw('fixed'),floating:settlementLaw('floating',{advance:floatingAdvance})};
  const scores=Object.fromEntries(Object.entries(laws).map(([a,d])=>[a,criterion==='mean'?expectation(d.atoms,d.masses):lowerCVaR(d.atoms,d.masses,eta)]));
  return {criterion,scores,action:Math.abs(scores.fixed-scores.floating)<1e-12?tie:scores.fixed>scores.floating?'fixed':'floating'};
}
export function uncertaintyComponents(alpha,beta){
  finite(alpha,'alpha');finite(beta,'beta');if(alpha<=0||beta<=0)throw new RangeError('positive beta parameters required');
  const q=alpha/(alpha+beta),parameterVariance=alpha*beta/((alpha+beta)**2*(alpha+beta+1));
  return {alpha,beta,success_probability:q,variance_of_parameter:parameterVariance,predictive_mean:.5+2*q,
    expected_conditional_return_variance:4*(q*(1-q)-parameterVariance),variance_of_conditional_mean:4*parameterVariance,predictive_return_variance:4*q*(1-q)};
}
export function distributionalData(){
  const floating=categoricalTarget({reward:.5,nextMasses:[.5,0,0,0,.5]});
  const fixed=categoricalTarget({reward:.5,nextMasses:[0,0,1,0,0]});
  const overflow=categoricalTarget({reward:3,nextMasses:[.5,0,0,0,.5]});
  const update=logitStep([0,0,0,0,0],floating.projected);
  return {kind:'constructed-exact-settlement',random_rollouts:0,neural_training_steps:0,exact_logit_steps:1,
    task:{gamma:GAMMA,support:[...SUPPORT],advance:.5,fixed_final_reward:2,floating_final_rewards:[0,4],floating_final_masses:[.5,.5],terminal_future_return:0},
    nominal:{fixed:settlementLaw('fixed'),floating:settlementLaw('floating')},
    bonus:{fixed:settlementLaw('fixed'),floating:settlementLaw('floating',{advance:.6})},
    projections:{fixed,floating,overflow},sample_update:update,
    choices:{nominal_mean:chooseSettlement(),bonus_mean:chooseSettlement({floatingAdvance:.6}),bonus_risk:chooseSettlement({floatingAdvance:.6,criterion:'lower_cvar'})},
    risk:{true_fixed:1.5,projected_fixed:lowerCVaR(SUPPORT,fixed.projected),true_floating:.5,projected_floating:lowerCVaR(SUPPORT,floating.projected)},
    uncertainty:[uncertaintyComponents(1,1),uncertaintyComponents(50,50)]};
}
