/** Deterministic two-ply search, not an AlphaZero implementation or training run. */
export const PAYOFFS = Object.freeze([Object.freeze([1,-1]),Object.freeze([0,1])]);
export const ROOT_PRIOR = Object.freeze([.8,.2]);

export function searchLabels(simulations=32){
  if(!Number.isInteger(simulations)||simulations<1)throw new RangeError('positive integer simulations required');
  const n=[0,0],w=[0,0],childN=[[0,0],[0,0]],childW=[[0,0],[0,0]],trace=[];
  const select=(counts,sums,prior)=>{
    const total=counts.reduce((a,b)=>a+b,0);
    const score=counts.map((v,a)=>(v?sums[a]/v:0)+prior[a]*Math.sqrt(1+total)/(1+v));
    return score[1]>score[0]?1:0; // ties choose first action
  };
  for(let k=0;k<simulations;k++){
    const a=select(n,w,ROOT_PRIOR),b=select(childN[a],childW[a],[.5,.5]),z=PAYOFFS[a][b];
    n[a]++;w[a]+=z;childN[a][b]++;childW[a][b]-=z;
    trace.push({simulation:k+1,actions:[a,b],rootOutcome:z,rootVisits:[...n]});
  }
  const policy=n.map(v=>v/simulations);
  return {simulations,payoffs:PAYOFFS,prior:ROOT_PRIOR,counts:n,childCounts:childN,
    actionValues:w.map((v,a)=>n[a]?v/n[a]:0),policy,minimax:PAYOFFS.map(row=>Math.min(...row)),
    trace,exampleGame:{actions:[0,1],valueLabels:[-1,1]},
    policyLogitGradient:ROOT_PRIOR.map((p,a)=>p-policy[a])};
}
