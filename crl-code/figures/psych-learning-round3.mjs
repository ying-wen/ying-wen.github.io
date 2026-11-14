/** Exact independent teaching examples. No empirical subject measurements. */
export function psychLearningData(){
  const predicted=[3,0],observed=[0,3],utility=[1,1];
  const dot=(a,b)=>a.reduce((sum,x,i)=>sum+x*b[i],0);
  const chains=[['A1','A2','A3','A4'],['B1','B2','B3','B4']];
  const visual=['A3','B1','A1','B4','A4','B2','A2','B3'];
  const edges=items=>items.slice(1).map((x,i)=>`${items[i]}→${x}`);
  const trueEdges=chains.flatMap(edges),visualEdges=edges(visual);
  const memories=[1,0],contextWeights=[[.9,.1],[.1,.9],[.9,.1]];
  return {
    kind:'independent-teaching-calculations',
    identity:{predicted,observed,utility,featureError:observed.map((x,i)=>x-predicted[i]),scalarError:dot(observed,utility)-dot(predicted,utility)},
    devaluation:{outcomeProbability:1,utilityBefore:1,utilityAfter:0,cachedActionValue:1,recomputedValue:0},
    replay:{chains,visual,trueEdges,visualEdges,sharedEdges:visualEdges.filter(e=>trueEdges.includes(e))},
    renewal:{memories,contextWeights,predictions:contextWeights.map(w=>dot(w,memories)),overwrittenMemory:0,frozenOverwritePredictions:[0,0]},
  };
}
