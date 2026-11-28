/** Deterministic two-step settlement quantiles. No RNG or neural training. */
const finite=(x,label)=>{if(!Number.isFinite(x))throw new TypeError(label+' must be finite');return x;};
const levels=taus=>{if(!taus.length||taus.some(t=>!Number.isFinite(t)||t<=0||t>=1))throw new RangeError('levels must lie in (0,1)');};
function distribution(atoms,masses){
 if(!atoms.length||atoms.length!==masses.length||atoms.some(x=>!Number.isFinite(x))||masses.some(p=>!Number.isFinite(p)||p<0)||Math.abs(masses.reduce((s,p)=>s+p,0)-1)>1e-12)throw new RangeError('valid unit mass distribution required');
 return atoms.map((z,i)=>({z,p:masses[i]})).sort((a,b)=>a.z-b.z);
}
export function midpointTaus(n){if(!Number.isInteger(n)||n<1)throw new RangeError('positive integer N required');return Array.from({length:n},(_,i)=>(i+.5)/n);}
export function inverseCDF(atoms,masses,tau){
 finite(tau,'tau');if(tau<=0||tau>1)throw new RangeError('quantile level must lie in (0,1]');
 let mass=0;for(const {z,p} of distribution(atoms,masses)){mass+=p;if(p>0&&mass>=tau-1e-14)return z;}throw new Error('missing quantile');
}
export function cdfInterval(atoms,masses,theta,tau){
 finite(theta,'theta');levels([tau]);const law=distribution(atoms,masses);
 return [law.filter(o=>o.z<theta).reduce((s,o)=>s+o.p,0)-tau,law.filter(o=>o.z<=theta).reduce((s,o)=>s+o.p,0)-tau];
}
export function quantileIntegral(atoms,masses,left=0,right=1){
 if(!Number.isFinite(left)||!Number.isFinite(right)||left<0||right>1||left>right)throw new RangeError('integration interval must lie in [0,1]');
 let total=0,cumulative=0;for(const {z,p} of distribution(atoms,masses)){
  total+=z*Math.max(0,Math.min(right,cumulative+p)-Math.max(left,cumulative));cumulative+=p;
 }return total;
}
export function readout(atoms,masses=atoms.map(()=>1/atoms.length),eta=.5){
 finite(eta,'tail mass');if(eta<=0||eta>1)throw new RangeError('tail mass must lie in (0,1]');
 return {mean:quantileIntegral(atoms,masses),lower_cvar:quantileIntegral(atoms,masses,0,eta)/eta};
}
export function pinball(u,tau){finite(u,'residual');levels([tau]);return u*(tau-(u<0?1:0));}
export function quantileHuber(u,tau,kappa=1){
 finite(u,'residual');levels([tau]);finite(kappa,'kappa');if(kappa<=0)throw new RangeError('positive kappa required');
 const magnitude=Math.abs(u),h=magnitude<=kappa?.5*u*u:kappa*(magnitude-.5*kappa);
 return Math.abs(tau-(u<0?1:0))*h/kappa;
}
export function pairwise(theta,taus,target,{kind='pinball',kappa=1}={}){
 if(!theta.length||theta.length!==taus.length||!target.length||[...theta,...target].some(x=>!Number.isFinite(x)))throw new RangeError('matching current values/levels and nonempty target required');
 levels(taus);if(!['pinball','huber'].includes(kind))throw new RangeError('unknown loss');
 if(kind==='huber'&&(!Number.isFinite(kappa)||kappa<=0))throw new RangeError('positive kappa required');
 const residuals=theta.map(x=>target.map(y=>y-x)),headGradient=theta.map(()=>0);let loss=0;
 residuals.forEach((row,i)=>row.forEach(u=>{
  loss+=(kind==='pinball'?pinball(u,taus[i]):quantileHuber(u,taus[i],kappa))/(theta.length*target.length);
  // For pinball at equality choose 0, a legal member of [-tau,1-tau].
  const derivative=kind==='pinball'?(u===0?0:(u<0?1:0)-taus[i]):-Math.abs(taus[i]-(u<0?1:0))*Math.max(-1,Math.min(1,u/kappa));
  headGradient[i]+=derivative/target.length;
 }));
 return {kind,kappa:kind==='huber'?kappa:null,loss,residuals,head_gradient:headGradient,gradient:headGradient.map(g=>g/theta.length)};
}
export function quantileStep(theta,taus,target,{rate=theta.length,...options}={}){
 finite(rate,'rate');if(rate<=0)throw new RangeError('positive rate required');const before=pairwise(theta,taus,target,options);
 const after=theta.map((x,i)=>x-rate*before.gradient[i]);
 return {...before,rate,before:[...theta],target:[...target],after,after_loss:pairwise(after,taus,target,options).loss};
}
export function bellmanTargets({reward,gamma=.5,terminal=false,successor}){
 finite(reward,'reward');finite(gamma,'gamma');if(gamma<0||gamma>1||typeof terminal!=='boolean'||!successor?.length||successor.some(x=>!Number.isFinite(x)))throw new RangeError('valid discount, terminal and successor required');
 return successor.map(z=>reward+gamma*(terminal?0:z));
}
export function selectMeanAction(actions,{tie='fixed'}={}){
 const entries=Object.entries(actions);if(!entries.length||!Object.hasOwn(actions,tie))throw new RangeError('tie action must exist');
 const scores=Object.fromEntries(entries.map(([a,atoms])=>[a,readout(atoms).mean]));
 let action=tie;for(const [a] of entries)if(scores[a]>scores[action]+1e-12)action=a;
 return {scores,action};
}
export function inverseCDFDistance(atoms,masses,otherAtoms,otherMasses){
 const a=distribution(atoms,masses),b=distribution(otherAtoms,otherMasses),boundaries=[0,1];
 for(const law of [a,b]){let c=0;law.forEach(o=>{c+=o.p;boundaries.push(c);});}
 const cuts=[...new Set(boundaries)].sort((x,y)=>x-y);let total=0;
 for(let i=1;i<cuts.length;i++)if(cuts[i]>cuts[i-1]){const mid=(cuts[i]+cuts[i-1])/2;total+=(cuts[i]-cuts[i-1])*Math.abs(inverseCDF(atoms,masses,mid)-inverseCDF(otherAtoms,otherMasses,mid));}
 return total;
}
export function quantileData(){
 const taus=midpointTaus(4),fixed={atoms:[1.5],masses:[1]},floating={atoms:[.5,2.5],masses:[.5,.5]};
 const target=bellmanTargets({reward:.5,successor:[0,0,4,4]}),initial=[1,1,2,2];
 const step=quantileStep(initial,taus,target),singleLow=quantileStep(initial,taus,[.5]);
 const snapshots=[];let current=[...initial];
 for(let k=0;k<=32;k++){
  if([0,1,4,32].includes(k))snapshots.push({updates:k,atoms:[...current],...readout(current),w1:inverseCDFDistance(current,current.map(()=>.25),floating.atoms,floating.masses),loss:pairwise(current,taus,target).loss});
  if(k<32)current=quantileStep(current,taus,target,{rate:4/Math.sqrt(k+1)}).after;
 }
 const smooth=taus.map(t=>t<.5?.5+t/(1-t):2.5-(1-t)/t);
 const coarse={atoms:[.5,2.5],masses:[.2,.8]};const coarsePoints=taus.map(t=>inverseCDF(coarse.atoms,coarse.masses,t));
 const bonus={fixed:[1.5,1.5,1.5,1.5],floating:[.6,.6,2.6,2.6]};
 return {kind:'constructed-exact-settlement-quantiles',random_rollouts:0,neural_training_steps:0,exact_expected_updates:32,
  task:{gamma:.5,advance:.5,fixed_final_reward:2,floating_final_rewards:[0,4],floating_final_masses:[.5,.5],terminal_future_return:0},
  taus,true_laws:{fixed,floating},true_midpoints:{fixed:[1.5,1.5,1.5,1.5],floating:target},
  transition:{state:'s',action:'floating',reward:.5,next_state:'v',terminal:false,frozen_successor:[0,0,4,4],target},
  sample_step:step,single_low_sample:singleLow,snapshots,
  huber:{kappa:1,step:quantileStep(initial,taus,target,{kind:'huber'}),stationary_points:smooth,stationary_readout:readout(smooth),at_true:pairwise(target,taus,target,{kind:'huber'})},
  atom_conditions:{at_low:cdfInterval(floating.atoms,floating.masses,.5,.125),median_interval:[.5,2.5],canonical_median:inverseCDF(floating.atoms,floating.masses,.5)},
  finite_resolution:{law:coarse,midpoints:coarsePoints,true_readout:readout(coarse.atoms,coarse.masses),approximate_readout:readout(coarsePoints)},
  bonus:{advance:.6,atoms:bonus,mean_choice:selectMeanAction(bonus),risk_scores:{fixed:readout(bonus.fixed).lower_cvar,floating:readout(bonus.floating).lower_cvar},risk_action:'fixed'},
  iqn_specified_levels:{current:[.15,.7],target:[.2,.4,.9],decision:[.1,.3,.6,.8],true_target_samples:[.5,.5,2.5],current_true_values:[.5,2.5]},
 };
}
