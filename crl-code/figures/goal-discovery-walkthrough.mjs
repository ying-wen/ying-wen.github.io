// Original, bounded frontier-goal teaching mechanism; not a STOMP reproduction.
// Only act() has access to hidden outcomes. Discovery/learning read observed rows.
export const gamma = 0.9;
export const key = (s,a) => `${s}:${a}`;
const physical = {
  S: {upper:['A',0],lower:['B',0]},
  A: {exit:['X',1],probe:['Y',0]}, // Y pays 2, but this road costs 2.
  B: {forward:['C',0]}, C: {exit:['X',1],probe:['Y',2]}, X:{}, Y:{},
};
export function act(s,a) {
  const outcome=physical[s]?.[a];
  if(!outcome)throw new Error(`Unavailable physical action ${key(s,a)}`);
  const [next,reward]=outcome;
  return {s,a,next,reward,terminal:Object.keys(physical[next]).length===0,
    actions:Object.keys(physical[s]),nextActions:Object.keys(physical[next])};
}
export function newMemory(){return {rows:{},counts:{},actions:{},terminal:[],observations:0};}
export function observe(memory,row){
  const k=key(row.s,row.a),old=memory.rows[k];
  if(old&&(old.next!==row.next||old.reward!==row.reward))throw new Error('Stationary deterministic example only');
  memory.rows[k]={...row}; memory.counts[k]=(memory.counts[k]??0)+1;
  memory.actions[row.s]=[...row.actions];memory.actions[row.next]=[...row.nextActions];
  if(row.terminal&&!memory.terminal.includes(row.next))memory.terminal.push(row.next);
  memory.observations++;
}
export function seedMemory(){
  const m=newMemory();
  for(const [s,a] of [['S','upper'],['A','exit'],['S','lower'],['B','forward'],['C','exit']])observe(m,act(s,a));
  return m;
}
export function discover(memory,start='S'){
  const distance={[start]:0},queue=[start];
  for(const s of queue)for(const row of Object.values(memory.rows).filter(r=>r.s===s)){
    if(distance[row.next]===undefined){distance[row.next]=distance[s]+1;queue.push(row.next);}
  }
  return Object.entries(memory.actions).filter(([s])=>!memory.terminal.includes(s)&&distance[s]!==undefined)
    .map(([goal,actions])=>({goal,untried:actions.filter(a=>!memory.counts[key(goal,a)]),distance:distance[goal]}))
    .filter(c=>c.untried.length)
    // One probe is executed per call: coverage score = 1 new action row / travel+probe steps.
    .map(c=>({...c,score:1/(c.distance+1)}))
    .sort((a,b)=>b.score-a.score||a.goal.localeCompare(b.goal));
}
export function learnGoal(memory,goal){
  const rows=Object.values(memory.rows).filter(r=>r.s!==goal&&!memory.terminal.includes(r.s));
  let q=Object.fromEntries(rows.map(r=>[key(r.s,r.a),0])),snapshots=[{...q}],backups=0;
  for(let sweep=0;sweep<20;sweep++){
    const old=q;
    q=Object.fromEntries(rows.map(r=>{
      const tail=Math.max(0,...rows.filter(n=>n.s===r.next).map(n=>old[key(n.s,n.a)]));
      backups++;
      return [key(r.s,r.a),r.next===goal?1:r.terminal?0:gamma*tail];
    }));
    snapshots.push({...q});
    if(Object.keys(q).every(k=>Math.abs(q[k]-old[k])<1e-12))break;
  }
  const policy={};
  for(const s of Object.keys(memory.actions)){
    const choices=rows.filter(r=>r.s===s).sort((a,b)=>q[key(b.s,b.a)]-q[key(a.s,a.a)]||a.a.localeCompare(b.a));
    if(choices.length&&q[key(s,choices[0].a)]>0)policy[s]=choices[0].a;
  }
  return {goal,q,policy,snapshots,backups,stop:[goal,...memory.terminal],
    // This family fixes first-hit termination; only its goal parameter is discovered.
    terminationLearned:false};
}
export function optionModel(memory,option,start='S'){
  let s=start,reward=0,discount=1;const path=[s],records=[];
  while(!option.stop.includes(s)){
    const row=memory.rows[key(s,option.policy[s])];
    if(!row)throw new Error('Option has no data-supported action');
    records.push(key(s,row.a));reward+=discount*row.reward;discount*=gamma;
    s=row.next;path.push(s);if(path.length>20)throw new Error('Nonterminating option');
  }
  return {goal:option.goal,path,records,reward,duration:path.length-1,endpoint:s,discount};
}
export function execute(memory,option,{probe=false}={}){
  let s='S',G=0,discount=1;const rows=[];
  while(!option.stop.includes(s)){
    const row=act(s,option.policy[s]);rows.push(row);observe(memory,row);
    G+=discount*row.reward;discount*=gamma;s=row.next;
  }
  if(probe){
    const a=memory.actions[s].find(a=>!memory.counts[key(s,a)]);
    if(!a)throw new Error('No untried action at frontier');
    const row=act(s,a);rows.push(row);observe(memory,row);G+=discount*row.reward;
  }
  return {path:['S',...rows.map(r=>r.next)],rows,externalReturn:G,steps:rows.length};
}
export function externalValues(memory){
  let values=Object.fromEntries(Object.keys(memory.actions).map(s=>[s,0])),backups=0;
  for(let i=0;i<20;i++){
    const old=values;
    values=Object.fromEntries(Object.keys(old).map(s=>{
      const rows=Object.values(memory.rows).filter(r=>r.s===s);backups+=rows.length;
      return [s,rows.length?Math.max(...rows.map(r=>r.reward+(r.terminal?0:gamma*old[r.next]))):0];
    }));
    if(Object.keys(old).every(s=>Math.abs(old[s]-values[s])<1e-12))break;
  }
  return {values,backups};
}
export function walkthroughData(){
  const memory=seedMemory(),initial=structuredClone(memory),before=externalValues(memory),rounds=[],options=[];
  for(let round=0;round<2;round++){
    const candidates=discover(memory),selected=candidates[0],option=learnGoal(memory,selected.goal);
    const model=optionModel(memory,option),execution=execute(memory,option,{probe:true});
    options.push(option);
    rounds.push({candidates,selected:option.goal,option,model,execution,afterCandidates:discover(memory),
      observedRows:Object.keys(memory.rows).length,externalValues:externalValues(memory)});
  }
  const after=externalValues(memory),consumer=options.map(option=>{
    const model=optionModel(memory,option);
    return {goal:option.goal,...model,value:model.reward+model.discount*after.values[model.endpoint]};
  }).sort((a,b)=>b.value-a.value);
  const chosen=options.find(o=>o.goal===consumer[0].goal),prefix=execute(memory,chosen);
  const end=prefix.path.at(-1),row=Object.values(memory.rows).filter(r=>r.s===end)
    .sort((a,b)=>b.reward+(b.terminal?0:gamma*after.values[b.next])-a.reward-(a.terminal?0:gamma*after.values[a.next]))[0];
  const finalRow=act(end,row.a);observe(memory,finalRow);
  const deployment={goal:chosen.goal,path:[...prefix.path,finalRow.next],
    externalReturn:prefix.externalReturn+gamma**prefix.steps*finalRow.reward,steps:prefix.steps+1};
  return {gamma,initial,before,rounds,after,consumer,deployment,finalMemory:memory,
    budget:{seedSteps:5,explorationSteps:rounds.reduce((n,r)=>n+r.execution.steps,0),deploymentSteps:deployment.steps,
      episodes:5,resets:4,uniqueRows:Object.keys(memory.rows).length,optionCount:options.length,
      goalBackups:options.reduce((n,o)=>n+o.backups,0)},
    claim:'Exact deterministic mechanism with given features, action affordances, first-hit family and exploration budget; no general efficacy claim.'};
}
