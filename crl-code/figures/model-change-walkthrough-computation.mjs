// Original finite closed-loop example. Environment truth is private to step().
// The planner consumes observed empirical rows, never a change notification.
export const GAMMA = 0.9;
export const CONFIG = Object.freeze({budget:96, closeAt:12, reopenAt:48, recheckAge:12, warmUpper:8});
export const ACTIONS = Object.freeze({S:['upper','lower'],A:['cross'],B:['forward'],C:['forward'],X:['reset'],Y:['reset']});
export const ROWS = ['S:upper','A:cross','S:lower','B:forward','C:forward'];
export const MODES = ['cumulative-greedy','recent-greedy','cumulative-recheck','recent-recheck'];
export function environment(scenario='reopen',start=-28) {
  if(!['reopen','closed','stationary'].includes(scenario))throw new Error('Unknown scenario');
  let t=start,s='S';
  return {step(a) {
    if(!ACTIONS[s].includes(a))throw new Error(`Unavailable action ${s}:${a}`);
    const from=s, reset=a==='reset';
    const closed=t>=CONFIG.closeAt&&(scenario==='closed'||scenario==='reopen'&&t<CONFIG.reopenAt);
    if(reset)s='S';
    else if(from==='S')s=a==='upper'?'A':'B';
    else if(from==='A')s=closed?'Y':'X';
    else if(from==='B')s='C';
    else s='X';
    const row={t:t+1,s:from,a,next:s,reward:Number(!reset&&s==='X'),done:s==='X'||s==='Y',reset};
    t++;return row;
  }};
}
export function newModel(kind='cumulative') {
  if(!['cumulative','recent'].includes(kind))throw new Error('Unknown estimator');
  return {kind,version:0,rows:{},lastStart:{upper:null,lower:null}};
}
export function observe(model,row) {
  if(row.reset)return;
  const key=`${row.s}:${row.a}`,r=model.rows[key]??{count:0,outcomes:{},lastSeen:null};
  r.count++;
  // Outcome identity includes reward and termination, not only the next state.
  const id=JSON.stringify([row.next,row.reward,row.done]);
  if(model.kind==='recent')r.outcomes={};
  r.outcomes[id]=(r.outcomes[id]??0)+1;
  r.lastSeen=row.t;model.rows[key]=r;model.version++;
  if(row.s==='S')model.lastStart[row.a]=row.t;
}
export function probabilities(model,key) {
  const row=model.rows[key];if(!row)throw new Error(`No observations for ${key}`);
  const n=Object.values(row.outcomes).reduce((a,b)=>a+b,0);
  return Object.entries(row.outcomes).map(([id,count])=>{
    const [next,reward,done]=JSON.parse(id);return {next,reward,done,p:count/n};
  });
}
export function plan(model) {
  // The common initial coverage establishes this acyclic order. One expected
  // backup per observed action row; changing phase is not an input.
  const V={X:0,Y:0},Q={};let backups=0;
  for(const [s,as] of [['C',['forward']],['B',['forward']],['A',['cross']],['S',['upper','lower']]]) {
    for(const a of as){
      const key=`${s}:${a}`;
      Q[key]=probabilities(model,key).reduce((v,o)=>v+o.p*(o.reward+(o.done?0:GAMMA*V[o.next])),0);
      backups++;
    }
    V[s]=Math.max(...as.map(a=>Q[`${s}:${a}`]));
  }
  return {modelVersion:model.version,Q,V,backups};
}
export function decide(model,t,recheck=false) {
  const p=plan(model),greedy=p.Q['S:upper']>p.Q['S:lower']+1e-12?'upper':'lower';
  const ages=Object.fromEntries(['upper','lower'].map(a=>[a,t-model.lastStart[a]]));
  const stale=['upper','lower'].filter(a=>ages[a]>=CONFIG.recheckAge)
    .sort((a,b)=>ages[b]-ages[a]||a.localeCompare(b));
  const choice=recheck&&stale.length?stale[0]:greedy;
  return {...p,t,ages,greedy,choice,recheckOverride:choice!==greedy};
}
export function warmup(env,model) {
  const rows=[];
  for(const route of [...Array(CONFIG.warmUpper).fill('upper'),'lower']) {
    const actions=route==='upper'?['upper','cross','reset']:['lower','forward','forward','reset'];
    for(const a of actions){const row=env.step(a);rows.push(row);observe(model,row);}
  }
  return rows;
}
export function run(mode,scenario='reopen') {
  if(!MODES.includes(mode))throw new Error('Unknown controller');
  const model=newModel(mode.startsWith('recent')?'recent':'cumulative');
  const env=environment(scenario),seed=warmup(env,model),initial=structuredClone(model);
  const decisions=[],trace=[],episodes=[];
  let s='S',reward=0,backups=0,resets=0,episode=null;
  for(let t=0;t<CONFIG.budget;t++){
    let a;
    if(s==='S'){
      const d=decide(model,t,mode.endsWith('recheck'));decisions.push(d);backups+=d.backups;a=d.choice;
      episode={start:t,choice:a,greedy:d.greedy,modelVersion:d.modelVersion,path:['S'],return:0,steps:0};
    }else a=ACTIONS[s][0];
    const row=env.step(a);observe(model,row);reward+=row.reward;resets+=Number(row.reset);
    if(!row.reset){
      episode.return+=GAMMA**episode.steps*row.reward;episode.steps++;episode.path.push(row.next);
      if(row.done){episodes.push({...episode,end:row.t});episode=null;}
    }
    const pX=probabilities(model,'A:cross').filter(o=>o.next==='X').reduce((n,o)=>n+o.p,0);
    trace.push({...row,totalReward:reward,pX,modelVersion:model.version,backups,
      // Reader-only diagnostic, never sent to decide()/observe().
      truePX:scenario==='stationary'||row.t<CONFIG.closeAt||scenario==='reopen'&&row.t>=CONFIG.reopenAt?1:0});
    s=row.next;
  }
  return {mode,scenario,initial,seed,decisions,trace,episodes,finalModel:model,
    metrics:{interactionTicks:CONFIG.budget,physicalActions:CONFIG.budget-resets,resets,modelBackups:backups,
      modelUpdates:model.version-initial.version,reward,completedEpisodes:episodes.length,
      failedEpisodes:episodes.filter(e=>e.return===0).length,recheckOverrides:decisions.filter(d=>d.recheckOverride).length,
      discountedEpisodeReturn:episodes.reduce((n,e)=>n+e.return,0),unfinishedEpisode:episode!==null,
      changedOutletObservations:trace.filter(r=>r.s==='A').map(r=>({t:r.t,next:r.next})),
      terminalAtBudget:s==='X'||s==='Y'}};
}
export function walkthroughData(){
  return {kind:'exact finite deterministic closed-loop experiment',config:CONFIG,gamma:GAMMA,
    protocol:{warmupTicks:28,warmupPhysicalActions:19,warmupResets:9,warmupReward:9,
      environmentClock:'completed real interactions including charged terminal reset',
      computationClock:'expected row backups; five per S decision; planning consumes no environment tick',
      controlObjective:'discounted goal-X return for each trip; terminal tail zero',
      reportedObjective:'raw lifetime reward at fixed 96 additional interaction ticks, reset overhead included',
      known:'state/action identities, reward convention and terminal reset; observed warmup graph fixes backup order',
      hidden:'outlet transitions, change schedule and current true phase',
      evidence:'No random seeds, confidence intervals or external performance claim; exact constructed counterexample.'},
    stationaryNoise:{p:0.95,n:20,lastObservationMSE:0.95*0.05,sampleMeanMSE:0.95*0.05/20},
    results:['reopen','stationary','closed'].flatMap(s=>MODES.map(m=>run(m,s)))};
}
