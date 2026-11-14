/** Original finite branching mechanism: exact model, declared primitive budget, no training. */
export const gamma = .9;
export const leaves = Object.freeze(['C1','C2','C3','C4']);
export const actions = Object.freeze({S:['go','exit'],A:['cross'],C1:['collect','wait'],C2:['collect','wait'],C3:['collect','wait'],C4:['collect','wait'],G:[],D:[],B:[]});
export const pair = (s,a) => `${s}:${a}`;
export function exactModel(exitReward=.6,rewards=[2,2,0,0]) {
 if(!Number.isFinite(exitReward)||rewards.length!==4||rewards.some(r=>!Number.isFinite(r)))throw Error('Invalid task');
 const model={'S:go':[{next:'A',reward:0,done:false,p:1}],'S:exit':[{next:'B',reward:exitReward,done:true,p:1}],
  'A:cross':leaves.map(next=>({next,reward:0,done:false,p:.25}))};
 leaves.forEach((s,i)=>{model[pair(s,'collect')]=[{next:'G',reward:rewards[i],done:true,p:1}];model[pair(s,'wait')]=[{next:'D',reward:0,done:true,p:1}];});
 return model;
}
export function initialQ(exitReward=.6) {
 const q=Object.fromEntries(Object.entries(actions).flatMap(([s,as])=>as.map(a=>[pair(s,a),0])));q['S:exit']=exitReward;return q;
}
export function counters() {return {environmentTransitions:0,modelWrites:0,modelQueries:0,expandedOutcomes:0,actionValueReads:0,qWrites:0,
 scheduleAppends:0,scheduleBudgetChecks:0,sampleSelections:0,queuePushes:0,queuePops:0,work:0};}
export const workOf = c => c.modelQueries+c.expandedOutcomes+c.actionValueReads+c.qWrites;
export const choice = q => q['S:go']>q['S:exit']+1e-12?'go':'exit';
/** Cost metadata are known model topology; no reward or true-value oracle is consulted. */
export function backupCost(model,k,mode='expected',draw=0) {
 const rows=model[k];if(!rows)throw Error(`Unknown model pair ${k}`);
 if(mode!=='expected'&&mode!=='sample')throw Error('Invalid backup mode');
 if(mode==='sample'&&(!Number.isInteger(draw)||draw<0||draw>=rows.length))throw Error('Invalid model draw');
 const expanded=mode==='expected'?rows:[rows[draw]];
 return {modelQueries:1,expandedOutcomes:expanded.length,actionValueReads:expanded.reduce((sum,r)=>sum+(r.done?0:actions[r.next].length),0),qWrites:1};
}
export function backup(q,model,k,{mode='expected',draw=0,alpha=1,g=gamma}={},counts=counters()) {
 if(!(alpha>0&&alpha<=1)||!(g>=0&&g<1))throw Error('Invalid update');
 const cost=backupCost(model,k,mode,draw),rows=mode==='expected'?model[k]:[model[k][draw]];
 const y=rows.reduce((sum,r)=>{
  const continuation=r.done?0:g*Math.max(...actions[r.next].map(a=>q[pair(r.next,a)]));
  return sum+(mode==='expected'?r.p:1)*(r.reward+continuation);
 },0);
 const before=q[k];q[k]+=alpha*(y-before);
 for(const [key,value]of Object.entries(cost))counts[key]+=value;
 if(mode==='sample')counts.sampleSelections++;
 counts.work=workOf(counts);
 return {pair:k,mode,draw:mode==='sample'?draw:null,alpha,before,target:y,after:q[k],cost};
}
export function compileSchedule({order='backward',mode='expected',draws=[0]}={}) {
 if(!['backward','forward','moving'].includes(order)||!['expected','sample'].includes(mode))throw Error('Invalid schedule');
 const leafSteps=leaves.map(s=>({pair:pair(s,'collect'),mode:'expected',alpha:1}));
 const crossSteps=mode==='expected'?[{pair:'A:cross',mode,alpha:1}]:draws.map((draw,i)=>({pair:'A:cross',mode,draw,alpha:1/(i+1)}));
 if(mode==='sample'&&!crossSteps.length)throw Error('At least one sample required');
 const go={pair:'S:go',mode:'expected',alpha:1};
 if(order==='forward')return [go,...crossSteps,...leafSteps];
 if(order==='moving') {
  if(mode!=='sample'||crossSteps.length!==2)throw Error('Moving schedule requires two samples');
  return [crossSteps[0],...leafSteps,crossSteps[1],go];
 }
 return [...leafSteps,...crossSteps,go];
}
/** Atomic backups stop at the first item that cannot fit; unused budget is retained. */
export function runSchedule({order='backward',mode='expected',draws=[0],budget=30,exitReward=.6,rewards=[2,2,0,0],g=gamma}={}) {
 if(!Number.isInteger(budget)||budget<0)throw Error('Invalid budget');
 const model=exactModel(exitReward,rewards),q=initialQ(exitReward),counts=counters(),schedule=compileSchedule({order,mode,draws}),events=[];
 counts.scheduleAppends=schedule.length;
 const snapshot=()=>({q:{...q},work:counts.work,leafMean:leaves.reduce((sum,s)=>sum+Math.max(q[pair(s,'collect')],q[pair(s,'wait')]),0)/4});
 const snapshots=[snapshot()];let stoppedAt=null;
 for(const step of schedule) {
  counts.scheduleBudgetChecks++;
  const cost=backupCost(model,step.pair,step.mode,step.draw??0);
  if(counts.work+workOf(cost)>budget){stoppedAt=step.pair;break;}
  events.push(backup(q,model,step.pair,{...step,g},counts));snapshots.push(snapshot());
 }
 return {order,mode,budget,q,choice:choice(q),counts,unusedBudget:budget-counts.work,stoppedAt,complete:events.length===schedule.length,events,snapshots};
}
/** Reader-only truth: add rewards along complete environment paths, without a backup recurrence. */
export function rewardPathOracle(exitReward=.6,rewards=[2,2,0,0],g=gamma) {
 const paths=rewards.map((r,i)=>({route:['S','A',leaves[i],'G'],probability:.25,rewards:[0,0,r],return:g*g*r,realTransitions:3}));
 return {paths,go:paths.reduce((sum,p)=>sum+p.probability*p.return,0),exit:exitReward};
}
export function actualEpisode(q,{leaf=0,exitReward=.6,rewards=[2,2,0,0],g=gamma}={}) {
 if(!Number.isInteger(leaf)||leaf<0||leaf>3)throw Error('Invalid real outcome');
 const firstAction=choice(q);
 if(firstAction==='exit')return {firstAction,route:['S','B'],rewards:[exitReward],return:exitReward,environmentTransitions:1};
 const s=leaves[leaf],lastAction=q[pair(s,'collect')]>=q[pair(s,'wait')]?'collect':'wait';
 const rs=[0,0,lastAction==='collect'?rewards[leaf]:0];
 return {firstAction,lastAction,route:['S','A',s,lastAction==='collect'?'G':'D'],rewards:rs,return:rs.reduce((sum,r,i)=>sum+g**i*r,0),environmentTransitions:3};
}
/** Exhaust every modeled successor sequence; no random-number generator is run. */
export function enumerateSamples(n,{budget=16+5*n,exitReward=.6,order='backward'}={}) {
 if(!Number.isInteger(n)||n<1||n>6)throw Error('Enumeration supports 1..6 draws');
 const bins=Array.from({length:n+1},(_,positive)=>({positive,sequences:0,probability:0,meanEstimatedGo:0,estimateRange:[Infinity,-Infinity],chooseGoProbability:0}));
 const truth=rewardPathOracle(exitReward);let chooseGoProbability=0,meanGo=0,mseGo=0,meanTrueReturn=0,meanNextRealTransitions=0;
 let representative=null;
 for(let index=0;index<4**n;index++) {
  const draws=Array.from({length:n},(_,i)=>Math.floor(index/4**i)%4),positive=draws.filter(i=>i<2).length;
  const run=runSchedule({order,mode:'sample',draws,budget,exitReward});representative??=run;
  const p=1/4**n,go=run.choice==='go',v=run.q['S:go'];
  bins[positive].sequences++;bins[positive].probability+=p;bins[positive].meanEstimatedGo+=v;
  bins[positive].estimateRange=[Math.min(bins[positive].estimateRange[0],v),Math.max(bins[positive].estimateRange[1],v)];
  if(go){chooseGoProbability+=p;bins[positive].chooseGoProbability+=p;}
  meanGo+=p*v;mseGo+=p*(v-truth.go)**2;meanTrueReturn+=p*(go?truth.go:truth.exit);meanNextRealTransitions+=p*(go?3:1);
 }
 bins.forEach(b=>b.meanEstimatedGo/=b.sequences);
 return {n,enumeratedSequences:4**n,bins,chooseGoProbability,meanGo,mseGo,meanTrueReturn,meanNextRealTransitions,
  counts:representative.counts,unusedBudget:representative.unusedBudget,complete:representative.complete};
}
export function budgetComparison(budget,exitReward=.6) {
 const n=Math.max(1,Math.floor((budget-16)/5));
 const sample=enumerateSamples(n,{budget,exitReward}),expected=runSchedule({budget,exitReward}),truth=rewardPathOracle(exitReward);
 return {budget,n,sample,expected:{choice:expected.choice,qGo:expected.q['S:go'],chooseGoProbability:expected.choice==='go'?1:0,
  trueReturn:expected.choice==='go'?truth.go:truth.exit,counts:expected.counts,unusedBudget:expected.unusedBudget,complete:expected.complete}};
}
export function walkthroughData() {
 const forward=runSchedule({order:'forward'}),backward=runSchedule(),oneSample=runSchedule({mode:'sample',draws:[0],budget:21});
 return {evidence:'exact-model-and-finite-enumeration',randomTrainingRuns:0,gamma,exitReward:.6,branchRewards:[2,2,0,0],actions,
  model:exactModel(),initialQ:initialQ(),truth:rewardPathOracle(),costs:{leaf:backupCost(exactModel(),'C1:collect'),
   sampleCross:backupCost(exactModel(),'A:cross','sample',0),expectedCross:backupCost(exactModel(),'A:cross'),go:backupCost(exactModel(),'S:go')},
  order:{forward,backward},sameOrder:{expected:backward,samplePositive:oneSample,sampleZero:runSchedule({mode:'sample',draws:[2],budget:21})},
  frozen:{expectedCross:.9,targetVariance:.81,samples:[1,2,3,4].map(n=>enumerateSamples(n))},
  moving:enumerateSamples(2,{order:'moving'}),budgets:[12,20,21,25,26,29,30,31,35,36,40].map(b=>budgetComparison(b)),
  nearTie:{exitReward:.82,truth:rewardPathOracle(.82),budgets:[21,26,30,31,36].map(b=>budgetComparison(b,.82))},
  nextReal:{forward:actualEpisode(forward.q),backwardPositive:actualEpisode(backward.q,{leaf:0}),backwardZero:actualEpisode(backward.q,{leaf:2})},
  protocol:{model:'Given exact distribution model, independent variant; no earlier empirical records are reused. Reward-path truth is reader-only.',
   initial:'All Q zero except known exit payoff. Wait Q zero is correct here; collect values are computed by leaf backups. Ties at S choose exit; leaf ties choose collect.',
   budget:'W=model interface queries + joint outcomes processed + successor action-value reads + Q writes. Atomic backup; halt before an item that cannot fit.',
   schedule:'Static topology-based leaf order C1,C2,C3,C4, then cross, then go; it does not inspect rewards or true values to choose leaf order.',
   overhead:'Count schedule appends, metadata/budget checks, and modeled sample selections separately. W excludes those operations, RNG, arithmetic, memory allocations and wall-clock time. No priority queue is used.',
   sampling:'Supplied successor selectors enumerate iid uniform model draws exactly. After leaf backups, hold leaf Q fixed; cross uses alpha_k=1/k; go reads the resulting cross once.',
   moving:'First cross sample precedes all leaf updates; second follows them. Same n=2 primitive work, but targets have different conditional means.',
   action:'Only after planning stops does S compare go with exit. Next real episode is separately charged and does not feed back into the frozen comparisons.'}};
}
