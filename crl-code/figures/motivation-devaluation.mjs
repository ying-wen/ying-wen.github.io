/**
 * Pure, deterministic teaching calculations for outcome revaluation.
 * No animal data, fitted parameters, simulation or training is used here.
 *
 * Two disjoint, two-transition routes initially connect A→C→E and B→D→F.
 * Only arrival at terminal E or F pays a reward; the terminal has no tail value.
 * Reward exchange changes terminal rewards. Transition exchange changes C and D.
 * The initial scalar values and initial terminal-occupancy rows are retained.
 */
export function routeValue(start, next, rewards, gamma=1) {
  if(!Number.isFinite(gamma)||gamma<0||gamma>1)throw new RangeError('gamma must be in [0,1]');
  let state=start, discount=1, value=0;
  const visited=new Set();
  while(Object.hasOwn(next,state)){
    if(visited.has(state))throw new RangeError('teaching route must terminate');
    visited.add(state);
    state=next[state];
    // Reward belongs to the transition arriving here, not to a later action.
    const reward=rewards[state]??0;
    if(!Number.isFinite(reward))throw new TypeError('reward must be finite');
    value+=discount*reward;
    discount*=gamma;
  }
  return value;
}

export function terminalOccupancy(start,next,terminals=['E','F'],gamma=1){
  return terminals.map(terminal=>routeValue(start,next,{[terminal]:1},gamma));
}

export function revaluationData(gamma=1){
  const initialNext={A:'C',B:'D',C:'E',D:'F'};
  const initialRewards={E:4,F:1};
  const starts=['A','B'];
  const oldRows=starts.map(s=>terminalOccupancy(s,initialNext,['E','F'],gamma));
  const initialValues=starts.map(s=>routeValue(s,initialNext,initialRewards,gamma));
  const conditions=[
    {id:'initial',label:'原路线',next:initialNext,rewards:initialRewards},
    {id:'reward',label:'只交换终点奖励',next:initialNext,rewards:{E:1,F:4}},
    {id:'transition',label:'只交换中段出口',next:{A:'C',B:'D',C:'F',D:'E'},rewards:initialRewards},
  ].map(condition=>({
    ...condition,
    values:{
      cached:[...initialValues],
      occupancy:oldRows.map(row=>row[0]*condition.rewards.E+row[1]*condition.rewards.F),
      model:starts.map(s=>routeValue(s,condition.next,condition.rewards,gamma)),
    },
  }));
  return {kind:'exact',gamma,starts,terminalRewardConvention:'reward on arrival, zero tail',
    oldRows,conditions,permissions:{cached:'no upstream update',occupancy:'fixed start rows; new rewards',model:'updated one-step transitions; full recomputation'}};
}

/** Protocol metadata prevents the two historical experiments being conflated. */
export const devaluationProtocol={
  source:'Adams & Dickinson (1981), Experiment I',
  kind:'schematic',
  levers:1,
  outcomes:['instrumental O1','noncontingent O2'],
  training:'different days; outcome identities counterbalanced',
  aversion:{cycles:3,daysPerCycle:2,treatmentsByDay:['LiCl','saline'],leverAvailable:false,
    target:{LiCl:'O1',saline:'O2'},control:{LiCl:'O2',saline:'O1'}},
  test:{hoursAfterLastAversionSession:24,minutes:20,binMinutes:5,leverAvailable:true,outcomesDelivered:false},
  measuredResponseRates:null,
};
