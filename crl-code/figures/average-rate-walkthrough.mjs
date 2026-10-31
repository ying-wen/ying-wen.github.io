/** Exact deterministic semi-MDP. Rewards arrive at transition completion.
 * The event counter and physical seconds are deliberately separate.
 * No stochastic training or algorithm-convergence experiment is performed.
 */
export const states = Object.freeze(['E', 'H', 'A', 'B']);
export function transitions(slowReturn = 3) {
  if (!Number.isInteger(slowReturn) || slowReturn < 1) throw new RangeError('positive integer duration required');
  return {
    E: [{action:'enter', next:'H', reward:-3, duration:2}],
    H: [{action:'quick', next:'A', reward:0, duration:1}, {action:'slow', next:'B', reward:0, duration:1}],
    A: [{action:'return', next:'H', reward:4, duration:1}],
    B: [{action:'return', next:'H', reward:9, duration:slowReturn}],
  };
}
export function evaluate(slowProbability = 0, slowReturn = 3) {
  transitions(slowReturn);
  if (!Number.isFinite(slowProbability) || slowProbability < 0 || slowProbability > 1) throw new RangeError('probability in [0,1] required');
  const p = slowProbability, cycleReward = 4*(1-p)+9*p, cycleTime = 2*(1-p)+(1+slowReturn)*p;
  const gain = cycleReward/cycleTime;
  return {slowProbability:p, gain, cycleReward, cycleTime,
    bias:{E:-3-2*gain, H:0, A:4-gain, B:9-slowReturn*gain},
    eventStationary:{E:0, H:.5, A:(1-p)/2, B:p/2}};
}
export function residual(edge, gain, bias, state) {
  return edge.reward-gain*edge.duration+bias[edge.next]-bias[state];
}
export function actionValues(evaluation, slowReturn = 3) {
  return transitions(slowReturn).H.map(e=>({action:e.action, value:e.reward-evaluation.gain*e.duration+evaluation.bias[e.next]}));
}
export function lifetime(policy = 'quick', horizon = 18, slowReturn = 3) {
  if (!['quick','slow'].includes(policy) || !Number.isInteger(horizon) || horizon < 0) throw new RangeError('policy and integer horizon required');
  const model=transitions(slowReturn), points=[[0,0]], events=[];
  let state='E', time=0, reward=0, n=0;
  while(time<horizon){
    const edge=model[state].find(e=>state!=='H'||e.action===policy), origin=state, start=time;
    for(let elapsed=1;elapsed<=edge.duration&&time<horizon;elapsed++){
      time++; if(elapsed===edge.duration){reward+=edge.reward;state=edge.next;n++;events.push({n,start,time,state,origin,reward:edge.reward,duration:edge.duration,total:reward});}
      points.push([time,reward]);
    }
  }
  return {points,events,physicalSeconds:horizon,completedEvents:n,totalReward:reward};
}
/** Wan et al. inter-option update, specialized to a known deterministic L.
 * The caller supplies nextValue from a frozen prediction or control snapshot.
 * alpha has seconds as units; eta has inverse seconds as units.
 */
export function durationBackup({reward,duration,rate,current,nextValue,alpha=.3,eta=.5}) {
  if(![reward,duration,rate,current,nextValue,alpha,eta].every(Number.isFinite)||duration<=0||alpha<0||eta<0)throw new RangeError('finite inputs and positive duration required');
  const delta=reward-rate*duration+nextValue-current, increment=alpha*delta/duration;
  return {delta,increment,value:current+increment,rate:rate+eta*increment};
}
export function walkthroughData() {
  const quick=evaluate(0),slow=evaluate(1),mixed=evaluate(.5),changedQuick=evaluate(0,4),changedSlow=evaluate(1,4);
  return {kind:'exact',sampledRuns:0,units:{time:'seconds',reward:'reward',gain:'reward/second',bias:'reward'},
    transitions:transitions(),quick,slow,mixed,changedQuick,changedSlow,
    quickActionValues:actionValues(quick),slowActionValues:actionValues(slow),
    changedQuickActionValues:actionValues(changedQuick,4),
    durationMean:{quick:2,slow:(0/1+9/3)/2},
    predictionBackup:durationBackup({reward:9,duration:3,rate:2,current:2,nextValue:0}),
    quickLifetime:lifetime('quick'),slowLifetime:lifetime('slow'),
    changedSlowLifetime:lifetime('slow',18,4)};
}
