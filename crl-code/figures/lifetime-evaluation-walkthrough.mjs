/** Independent probability recursion; Python uses exact joint-distribution DP. */
export function lifetimeEvaluationData(good=.9, block=8, diagnosticCircles=8){
  if(!(good>=.5&&good<=1&&Number.isInteger(block)&&block>0&&Number.isInteger(diagnosticCircles)&&diagnosticCircles>0))throw new RangeError('invalid task');
  let p=1,total=0,fixed=0;
  const rows=[],curves=[{step:0,learner:0,fixed_left:0}];
  const schedule=[...Array(block).fill('A'),...Array(block).fill('B'),...Array(block).fill('A')];
  schedule.forEach((phase,i)=>{
    const ql=phase==='A'?good:1-good,qr=1-ql,mean=p*ql+(1-p)*qr;
    const after=p*ql+(1-p)*(1-qr);total+=mean;fixed+=ql;
    rows.push({circle:i+1,phase,left_before:p,expected_reward:mean,left_after:after,cumulative:total,fixed_left_cumulative:fixed});
    curves.push({step:2*i+1,learner:total,fixed_left:fixed},{step:2*i+2,learner:total,fixed_left:fixed});
    p=after;
  });
  const diagnostic=p*good+(1-p)*(1-good);
  return {kind:'exact-expectation',sampled_runs:0,training_steps:6*block,diagnostic_steps:2*diagnosticCircles,
    good_reward_probability:good,rows,curves,lifetime_learner:total,lifetime_fixed_left:fixed,
    checkpoint_left_probability:p,diagnostic_learner_per_circle:diagnostic,diagnostic_fixed_left_per_circle:good,
    diagnostic_learner_total:diagnosticCircles*diagnostic,diagnostic_fixed_left_total:diagnosticCircles*good};
}
