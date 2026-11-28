/** Empirical joint-outcome Dyna mechanism. No random training or hidden truth labels. */
export const gamma = .9;
export const actions = Object.freeze({S:['go','exit'],A:['cross'],G:[],D:[],B:[]});
export const pair = (s,a) => `${s}:${a}`;
export const zeroQ = () => ({'S:go':0,'S:exit':0,'A:cross':0});
export const outcome = next => ({s:'A',a:'cross',next,reward:next==='G'?1:0,done:true});
export const goRow = Object.freeze({s:'S',a:'go',next:'A',reward:0,done:false});
export const exitRow = Object.freeze({s:'S',a:'exit',next:'B',reward:.6,done:true});
export const teachingOutcomes = Object.freeze(['D','G','G','G']);
export function learner(){
 return {q:zeroQ(),model:{},counts:{environmentSteps:0,resets:0,modelWrites:0,
  directBackups:0,modelQueries:0,outcomeEvaluations:0,planningBackups:0}};
}
function validate(row){
 if(!actions[row.s]?.includes(row.a)||!(row.next in actions)||!Number.isFinite(row.reward)||typeof row.done!=='boolean')throw Error('Invalid outcome');
 if(!row.done&&!actions[row.next].length)throw Error('Terminal outcome must be marked done');
}
export function target(q,row,g=gamma){
 validate(row);
 return row.reward+(row.done?0:g*Math.max(...actions[row.next].map(a=>q[pair(row.next,a)])));
}
export function writeQ(q,row,y,alpha=1){
 if(!(alpha>=0&&alpha<=1)||!Number.isFinite(y))throw Error('Invalid update');
 const k=pair(row.s,row.a),before=q[k];q[k]+=alpha*(y-before);
 return {pair:k,before,target:y,after:q[k]};
}
export function observe(l,row){
 validate(row);l.counts.environmentSteps++;l.counts.directBackups++;
 const event=writeQ(l.q,row,target(l.q,row));
 const k=pair(row.s,row.a),entry=l.model[k]??={n:0,outcomes:[]};
 // Reward, successor and termination are counted jointly, never independently recombined.
 let item=entry.outcomes.find(o=>o.next===row.next&&o.reward===row.reward&&o.done===row.done);
 if(!item){item={next:row.next,reward:row.reward,done:row.done,count:0};entry.outcomes.push(item);}
 item.count++;entry.n++;entry.outcomes.sort((a,b)=>a.next.localeCompare(b.next)||a.reward-b.reward||Number(a.done)-Number(b.done));
 l.counts.modelWrites++;return event;
}
export function distribution(l,k){
 const entry=l.model[k];if(!entry?.n)throw Error(`Unobserved pair ${k}`);
 return entry.outcomes.map(o=>({...o,probability:o.count/entry.n}));
}
export function rewardMean(l,k){return distribution(l,k).reduce((sum,o)=>sum+o.probability*o.reward,0);}
export function modelTarget(l,k,{mode='expected',u=.5,g=gamma}={}){
 const rows=distribution(l,k),[s,a]=k.split(':');l.counts.modelQueries++;
 if(mode==='expected'){
  l.counts.outcomeEvaluations+=rows.length;
  return rows.reduce((sum,o)=>sum+o.probability*target(l.q,{s,a,...o},g),0);
 }
 if(mode!=='sample'||!(u>=0&&u<1))throw Error('Invalid model query');
 // A supplied quantile selects a possible modeled draw, not a new real transition.
 let mass=0,selected=rows.at(-1);
 for(const row of rows){mass+=row.probability;if(u<mass){selected=row;break;}}
 l.counts.outcomeEvaluations++;return target(l.q,{s,a,...selected},g);
}
export function planningBackup(l,k,options={}){
 const y=modelTarget(l,k,options),[s,a]=k.split(':');
 l.counts.planningBackups++;return writeQ(l.q,{s,a},y,options.alpha??1);
}
export const choice = q => q['S:go']>q['S:exit']+1e-12?'go':'exit';
export function planRoad(l){return [planningBackup(l,'A:cross'),planningBackup(l,'S:go')];}
const copy = x => JSON.parse(JSON.stringify(x));
const delta = (after,before) => Object.fromEntries(Object.keys(after).map(k=>[k,after[k]-before[k]]));
export function snapshot(l,n){
 const rows=distribution(l,'A:cross');
 return {n,successes:rows.find(r=>r.next==='G')?.count??0,distribution:rows,
  rewardMean:rewardMean(l,'A:cross'),q:{...l.q},choice:choice(l.q),counts:{...l.counts}};
}
export function acquire(outcomes=teachingOutcomes){
 const l=learner(),prefixes=[];observe(l,exitRow);
 for(const next of outcomes){
  // These episodes are prescribed exploratory visits. They are not induced by greedy Q.
  l.counts.resets++;observe(l,goRow);observe(l,outcome(next));planRoad(l);
  prefixes.push({...snapshot(l,prefixes.length+1),realRecords:[{...goRow},outcome(next)],acquisition:'prescribed-exploration'});
 }
 return {learner:l,prefixes};
}
export function actualEpisode(l,next='D'){
 const before={...l.counts},firstAction=choice(l.q);l.counts.resets++;
 const rows=firstAction==='go'?[goRow,outcome(next)]:[exitRow];
 rows.forEach(row=>observe(l,row));
 const returned=rows.reduce((sum,row,i)=>sum+gamma**i*row.reward,0);
 // A declared two-pair planning schedule follows the completed real episode.
 planRoad(l);
 return {firstAction,route:rows.map(r=>r.s).concat(rows.at(-1).next),rewards:rows.map(r=>r.reward),
  return:returned,realSteps:rows.length,counts:delta(l.counts,before),after:snapshot(l,l.model['A:cross'].n)};
}
export function frozenComparisons(){
 const base=acquire(['D','G']).learner,out={};
 for(const [name,mode,u]of [['sampleD','sample',.25],['sampleG','sample',.75],['expected','expected',.5]]){
  const l=copy(base),before={...l.counts};
  const cross=planningBackup(l,'A:cross',{mode,u}),go=planningBackup(l,'S:go');
  const counts=delta(l.counts,before),q={...l.q},firstAction=choice(q);
  const nextExperience=actualEpisode(l,'D');
  out[name]={cross,go,q,choice:firstAction,counts,nextExperience};
 }
 return {frozenQ:{...base.q},model:copy(base.model),realCounts:{...base.counts},branches:out};
}
/** Analytical reader reference: exhaust all 2^4 real outcome sequences, not model draws. */
export function enumerateFour(){
 const sequences=[];
 for(let mask=0;mask<16;mask++){
  const outcomes=Array.from({length:4},(_,i)=>(mask>>i)&1?'G':'D');
  const successes=outcomes.filter(s=>s==='G').length;
  const probability=(3/4)**successes*(1/4)**(4-successes);
  const l=acquire(outcomes).learner,selected=choice(l.q);
  sequences.push({outcomes,successes,probability,choice:selected,estimatedGo:l.q['S:go'],
   trueNextEpisodeValue:selected==='go'?.675:.6,environmentSteps:l.counts.environmentSteps});
 }
 const successBins=Array.from({length:5},(_,k)=>({successes:k,
  probability:sequences.filter(s=>s.successes===k).reduce((sum,s)=>sum+s.probability,0),
  choice:k>=3?'go':'exit'}));
 const chooseGoProbability=sequences.filter(s=>s.choice==='go').reduce((sum,s)=>sum+s.probability,0);
 return {sequences,successBins,chooseGoProbability,wrongChoiceProbability:1-chooseGoProbability,
  expectedNextEpisodeValue:sequences.reduce((sum,s)=>sum+s.probability*s.trueNextEpisodeValue,0),
  expectedNextRealSteps:1+chooseGoProbability};
}
export function walkthroughData(){
 const acquired=acquire(),full=copy(acquired.learner),closure=actualEpisode(full,'D');
 const stopped=acquire(['D','G']).learner,stoppedBefore=copy(stopped.model['A:cross']);
 const greedyEpisode=actualEpisode(stopped);const beforeImagined={...stopped.counts};
 // Current greedy model trajectories only visit S:exit. No call writes outcome counts.
 for(let i=0;i<10;i++)planningBackup(stopped,'S:exit',{mode:'sample',u:.5});
 return {evidence:'exact-enumeration-and-prescribed-records',sampledRuns:0,gamma,alpha:1,
  actions,initialRealRecord:{...exitRow},teachingOutcomes,prefixes:acquired.prefixes,afterFour: snapshot(acquired.learner,4),closure,
  frozen:frozenComparisons(),enumeration:enumerateFour(),
  greedyTrap:{crossBefore:stoppedBefore,crossAfter:copy(stopped.model['A:cross']),greedyEpisode,
   tenImaginedEpisodes:delta(stopped.counts,beforeImagined)},
  readerTruth:{cross:[{next:'G',reward:1,done:true,probability:.75},{next:'D',reward:0,done:true,probability:.25}],
   values:{'A:cross':.75,'S:go':.675,'S:exit':.6},greedyThreshold:2/3},
  protocol:{variant:'Independent stationary stochastic variant; no deterministic old-phase records are reused.',
   acquisition:'Real exit once, then four prescribed exploratory go/cross episodes; D,G,G,G is one possible outcome record.',
   model:'Empirical counts of joint reward/successor/termination outcomes; unseen outcomes receive zero empirical mass, not a certainty claim.',
   update:'Each real row: direct alpha=1 update and one model write; after each crossing/real episode: expected cross, then expected go.',
   frozen:'Only the initial cross target comparison reads the common frozen Q; propagation to go reads the branch updated cross.',
   modelQuery:'One interface request; expectation returns all observed joint outcomes, sample returns one. Outcome evaluations are charged separately.',
   counts:'Real steps, resets, model writes, direct backups, model queries, expanded outcomes and planning backups are separate units.',
   tie:'Mathematical ties choose exit; JS uses 1e-12 tolerance and independent Fraction Python uses exact rationals.',
   enumeration:'All 16 sequences of four iid true cross outcomes; five acquired episodes cost nine real steps. Next autonomous episodes cost one or two steps.',
   closure:'After D,G,G,G, greedy go actually receives D; the empirical model changes from 3/4 to 3/5 and replanning changes go from .675 to .54.'}};
}
