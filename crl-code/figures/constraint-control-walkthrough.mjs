/** Exact delivery-route CMDP. All model entries and action quantiles are given. */
export const GAMMA=.9;
export const BUDGET=.3;
export const FAST_REWARD=9.2;
export const DETOUR_REWARD=10;
const finite=(v,label)=>{if(!Number.isFinite(v))throw new RangeError(label+' must be finite');return v;};
const probability=p=>{finite(p,'probability');if(p<0||p>1)throw new RangeError('probability outside [0,1]');return p;};
const discount=g=>{finite(g,'discount');if(g<0||g>=1)throw new RangeError('discount must lie in [0,1)');return g;};
export const projectProbability=p=>Math.max(0,Math.min(1,p));
export const projectMultiplier=v=>Math.max(0,v);

export function policyMetrics(p,gamma=GAMMA){
  probability(p);discount(gamma);
  const reward=(1-p)*gamma*DETOUR_REWARD+p*FAST_REWARD,cost=p;
  const occupancy={junction_detour:(1-gamma)*(1-p),junction_fast:(1-gamma)*p,
    detour_go:(1-gamma)*gamma*(1-p),terminal_wait:gamma*p+gamma**2*(1-p)};
  return {p,reward,cost,normalized_reward:(1-gamma)*reward,normalized_cost:(1-gamma)*cost,
    probability_of_any_cost:p,occupancy};
}
export function optimalProbability(budget=BUDGET,gamma=GAMMA){
  finite(budget,'budget');discount(gamma);
  if(budget<0)throw new RangeError('infeasible budget: even detour costs zero');
  return FAST_REWARD-gamma*DETOUR_REWARD>0?Math.min(1,budget):0;
}
export function lagrangian(p,lambda,budget=BUDGET){
  probability(p);finite(lambda,'multiplier');if(lambda<0)throw new RangeError('nonnegative multiplier required');
  const {reward,cost}=policyMetrics(p);return reward-lambda*(cost-budget);
}
/** Both directions read old p and old lambda; supplied cost estimate is frozen. */
export function primalDualStep(p,lambda,{etaP=.5,etaLambda=2,estimatedCost=p,budget=BUDGET}={}){
  probability(p);finite(lambda,'multiplier');finite(estimatedCost,'cost estimate');finite(budget,'budget');
  if(lambda<0||!Number.isFinite(etaP)||!Number.isFinite(etaLambda)||etaP<0||etaLambda<0)throw new RangeError('nonnegative finite rates and multiplier required');
  const primalGradient=FAST_REWARD-GAMMA*DETOUR_REWARD-lambda;
  const rawP=p+etaP*primalGradient,rawLambda=lambda+etaLambda*(estimatedCost-budget);
  return {old_p:p,old_lambda:lambda,estimated_cost:estimatedCost,primal_gradient:primalGradient,
    raw_p:rawP,raw_lambda:rawLambda,p:projectProbability(rawP),lambda:projectMultiplier(rawLambda)};
}
export function givenQuantileTrajectory(p,u=.55){
  probability(p);finite(u,'quantile');if(u<0||u>=1)throw new RangeError('quantile must lie in [0,1)');
  return u<p?
    [{state:'junction',action:'fast',reward:FAST_REWARD,cost:1,next_state:'terminal',terminal:true}]:
    [{state:'junction',action:'detour',reward:0,cost:0,next_state:'detour',terminal:false},
      {state:'detour',action:'go',reward:DETOUR_REWARD,cost:0,next_state:'terminal',terminal:true}];
}
export function meanKL(candidate,old=.2){
  probability(candidate);probability(old);if(old===0||old===1)throw new RangeError('interior old probability required');
  const term=(q,r)=>q===0?0:q*Math.log(q/r);
  return (1-GAMMA)*(term(candidate,old)+term(1-candidate,1-old));
}
export function localConstrainedStep(old=.2,budget=BUDGET,delta=.005){
  probability(old);finite(budget,'budget');finite(delta,'KL radius');
  if(old===0||old===1||delta<0)throw new RangeError('interior old policy and nonnegative KL radius required');
  const fisher=(1-GAMMA)/(old*(1-old)),radius=Math.sqrt(2*delta/fisher);
  const lower=Math.max(0,old-radius),upper=Math.min(1,old+radius),feasibleUpper=Math.min(upper,budget);
  if(feasibleUpper<lower)throw new RangeError('local feasible intersection is empty');
  const candidate=feasibleUpper;
  return {old_p:old,budget,delta,fisher,radius,trust_lower:lower,trust_upper:upper,
    reward_only_candidate:upper,constrained_candidate:candidate,
    exact_mean_kl:meanKL(candidate,old),reward_only_exact_mean_kl:meanKL(upper,old)};
}
export function constraintControlData(){
  let state={p:.6,lambda:0};const snapshots=[{k:0,...state,trajectory:givenQuantileTrajectory(state.p)}],updates=[];
  for(let k=0;k<3;k++){
    const next=primalDualStep(state.p,state.lambda);updates.push(next);state={p:next.p,lambda:next.lambda};
    snapshots.push({k:k+1,...state,trajectory:givenQuantileTrajectory(state.p)});
  }
  return {kind:'constructed-exact-cmdp',random_rollouts:0,neural_training_steps:0,exact_update_steps:3,
    task:{gamma:GAMMA,initial_distribution:{junction:1},budget:BUDGET,normalized_budget:(1-GAMMA)*BUDGET,
      fast_reward:FAST_REWARD,detour_terminal_reward:DETOUR_REWARD,absorbing_reward:0,absorbing_cost:0},
    policies:[0,.3,.6,1].map(p=>policyMetrics(p)),optimal_p:optimalProbability(),
    updates,snapshots,given_action_quantile:.55,local:localConstrainedStep(),
    feasible_curve:Array.from({length:101},(_,i)=>{const p=i/100;return [p,policyMetrics(p).reward];})};
}
