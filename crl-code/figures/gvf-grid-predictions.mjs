/** Pure finite-model calculations; no sampling, training, or private inputs.
 * The GVF convention is C[t+1] + gamma[t+1] G[t+1]: entering a
 * terminal door contributes its cumulant before continuation becomes zero.
 * Original teaching example; definition follows Sutton & Barto, §17.1.
 */
export const questions=[
 {id:'vf',name:'任务 VF：折扣成功',policy:'left',signal:'success indicator',continuation:.9,unit:'discounted success'},
 {id:'left',name:'GVF：向左能否成功',policy:'left',signal:'success indicator',continuation:1,unit:'probability'},
 {id:'right',name:'GVF：向右能否成功',policy:'right',signal:'success indicator',continuation:1,unit:'probability'},
 {id:'steps',name:'GVF：还要走几步',policy:'left',signal:'one per transition',continuation:1,unit:'steps'},
];
export const positions=['C','M','J'];
export function beliefForCue(cue,accuracy=.8){return cue==='left'?accuracy:1-accuracy;}
export function exactAnswer(question,position,belief){
 const remaining=3-positions.indexOf(position);
 if(question.signal==='one per transition')return remaining;
 const success=question.policy==='left'?belief:1-belief;
 return success*question.continuation**(remaining-1);
}

export function leftPosterior({prior=.5,accuracy=.8}={}){
 if(![prior,accuracy].every(v=>Number.isFinite(v)&&v>=0&&v<=1))throw new RangeError('probabilities must be in [0,1]');
 const evidence=accuracy*prior+(1-accuracy)*(1-prior);
 if(evidence===0)throw new RangeError('left cue has zero probability');
 return accuracy*prior/evidence;
}

export function questionMapsData({prior=.5,accuracy=.8,specs=questions}={}){
 const belief=leftPosterior({prior,accuracy});
 const predictions=specs.map(q=>{
  // Backward Bellman substitution in the finite acyclic chain C -> M -> J.
  // The final move enters a door; both success and failure end the question.
  const last=q.signal==='one per transition'?1:(q.policy==='left'?belief:1-belief);
  const values={L:0,R:0,J:last};
  for(const [s,next] of [['M','J'],['C','M']])values[s]=(q.signal==='one per transition'?1:0)+q.continuation*values[next];
  return {...q,values,domain:q.id==='steps'?[0,3]:[0,1],displayUnit:q.id==='steps'?'步':q.id==='vf'?'奖励单位':'概率'};
 });
 return {kind:'exact-finite-model',prior,accuracy,condition:'left cue observed; later position observations add no mode evidence',belief,predictions,terminalContinuation:0,terminalFutureValue:0,finalTransitionCounted:true};
}
