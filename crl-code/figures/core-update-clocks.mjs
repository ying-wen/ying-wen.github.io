/** Exact teaching calculations. No environment, neural network, or training run. */
export function mcTdSnapshots({alpha=.1,gamma=.9}={}) {
  const td={A:0,B:0,T:0},mc={A:0,B:0,T:0},snapshots=[];
  for(let episode=1;episode<=2;episode++) {
    const history=[];
    for(const [state,next,reward] of [['A','B',0],['B','T',1]]) {
      const beforeTD={...td},beforeMC={...mc};
      const target=reward+gamma*td[next];
      td[state]+=alpha*(target-td[state]);
      history.push({state,reward});
      const mcTargets={};
      if(next==='T') {
        let G=0;
        for(const item of [...history].reverse()) {
          G=item.reward+gamma*G;
          mcTargets[item.state]=G;
          mc[item.state]+=alpha*(G-mc[item.state]);
        }
      }
      snapshots.push({episode,state,next,reward,beforeTD,beforeMC,target,mcTargets,td:{...td},mc:{...mc}});
    }
  }
  return {alpha,gamma,snapshots};
}

/** Mirrors train_dqn's counter conditions; e_1 selection is a specified illustration. */
export function dqnClockSchedule(steps=101) {
  const records=[];let updates=0,targetVersion=0;
  for(let environment=1;environment<=steps;environment++) {
    const codeIndex=environment-1;
    const optimized=environment>=32 && codeIndex%2===0;
    const targetUsed=optimized?targetVersion:null;
    if(optimized)updates++;
    const copied=environment%100===0;
    if(copied)targetVersion=updates;
    records.push({environment,codeIndex,sampleBirth:1,sampleAge:environment-1,
      optimized,updates,targetUsed,copied,targetVersion});
  }
  return records;
}
