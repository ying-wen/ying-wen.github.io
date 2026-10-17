/** Exact matrix and one-step analytic geometry examples; not an RL run. */
export const rpsPayoff = Object.freeze([[0,-1,1],[1,0,-1],[-1,1,0]].map(Object.freeze));
export const discPayoff=(v,w)=>v[0]*w[1]-v[1]*w[0];
const mean=points=>[0,1].map(k=>points.reduce((s,p)=>s+p[k],0)/points.length);
const effectiveDiversity=points=>points.reduce((sum,v)=>sum+points.reduce((s,w)=>s+Math.max(0,discPayoff(v,w)),0),0)/points.length**2;

export function populationGeometryData(){
  const radius=.5,step=.6;
  const original=[[radius,0],[-radius/2,Math.sqrt(3)*radius/2],[-radius/2,-Math.sqrt(3)*radius/2]];
  const gradients=original.map(v=>original.reduce((grad,w)=>{
    // The current population has exact ties with self. Their ReLU subgradient is 0.
    if(discPayoff(v,w)>1e-12){grad[0]+=w[1]/3;grad[1]-=w[0]/3;}
    return grad;
  },[0,0]));
  const updated=original.map((v,i)=>v.map((x,k)=>x+step*gradients[i][k]));
  const smallMix=[1/3,2/3,0];
  return {
    kind:'deterministic-matrix-and-one-gradient-step',
    source:'https://arxiv.org/abs/1901.08106',
    rps:{matrix:rpsPayoff,names:['R','P','S'],projection:rpsPayoff.map(row=>row.slice(0,2)),
      nash:[1/3,1/3,1/3],smallMix,smallSecurity:Math.min(...[0,1,2].map(j=>smallMix.reduce((s,x,i)=>s+x*rpsPayoff[i][j],0))),
      duplicatedOpponentMatrix:[[1,1,1],[-1,-1,-1]],duplicatedOpponentValue:1},
    disc:{radius,step,original,gradients,updated,meanBefore:mean(original),meanAfter:mean(updated),
      radiusAfter:Math.hypot(...updated[0]),relativePopulationValue:0,
      diversityBefore:effectiveDiversity(original),diversityAfter:effectiveDiversity(updated),
      exploitBefore:discPayoff(original[0],original[1]),exploitAfter:discPayoff(updated[0],original[1])},
  };
}
