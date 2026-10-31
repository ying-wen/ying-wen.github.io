/** Exact, known-model calculations; no environment sampling or training. */
export const config=Object.freeze({cols:4,rows:3,walls:[[1,1]],goal:[3,0],start:[0,2],
 gamma:.9,step_reward:-.04,goal_reward:1,actions:['up','right','down','left']});
const moves=[[0,-1],[1,0],[0,1],[-1,0]];
export const key=cell=>cell.join(',');
export function gridModel(c=config){
 const states=[];const blocked=new Set(c.walls.map(key));
 for(let y=0;y<c.rows;y++)for(let x=0;x<c.cols;x++)if(!blocked.has(key([x,y]))&&key([x,y])!==key(c.goal))states.push(key([x,y]));
 const model={};
 for(const s of states){
  const [x,y]=s.split(',').map(Number);
  model[s]=moves.map(([dx,dy])=>{
   let next=[x+dx,y+dy];
   if(next[0]<0||next[0]>=c.cols||next[1]<0||next[1]>=c.rows||blocked.has(key(next)))next=[x,y];
   const terminal=key(next)===key(c.goal);
   return [{probability:1,reward:terminal?c.goal_reward:c.step_reward,next:terminal?null:key(next),terminal}];
  });
 }
 return {states,model};
}
export function actionValues(model,s,V,gamma=config.gamma){
 return model[s].map(outcomes=>outcomes.reduce((z,o)=>z+o.probability*(o.reward+gamma*(o.terminal?0:V[o.next])),0));
}
export function uniformPolicy(states){return Object.fromEntries(states.map(s=>[s,[.25,.25,.25,.25]]));}
export function greedyPolicy(states,model,V,gamma=config.gamma){
 return Object.fromEntries(states.map(s=>{
  const q=actionValues(model,s,V,gamma),best=Math.max(...q),a=q.findIndex(v=>v>=best-1e-12);
  return [s,q.map((_,i)=>i===a?1:0)];
 }));
}
function linearSolve(A,b){
 const n=b.length,M=A.map((row,i)=>[...row,b[i]]);
 for(let col=0;col<n;col++){
  let pivot=col;for(let i=col+1;i<n;i++)if(Math.abs(M[i][col])>Math.abs(M[pivot][col]))pivot=i;
  if(Math.abs(M[pivot][col])<1e-14)throw new Error('singular policy system');
  [M[col],M[pivot]]=[M[pivot],M[col]];
  const scale=M[col][col];for(let j=col;j<=n;j++)M[col][j]/=scale;
  for(let i=0;i<n;i++)if(i!==col){const f=M[i][col];for(let j=col;j<=n;j++)M[i][j]-=f*M[col][j];}
 }
 return M.map(row=>row[n]);
}
export function evaluatePolicy(states,model,policy,gamma=config.gamma){
 if(gamma<0||gamma>=1)throw new RangeError('finite discounted solver requires 0 <= gamma < 1');
 const index=new Map(states.map((s,i)=>[s,i])),A=states.map((_,i)=>states.map((_,j)=>+(i===j))),b=states.map(()=>0);
 states.forEach((s,i)=>model[s].forEach((outcomes,a)=>outcomes.forEach(o=>{
  const weight=policy[s][a]*o.probability;b[i]+=weight*o.reward;
  if(!o.terminal)A[i][index.get(o.next)]-=gamma*weight;
 })));
 const solved=linearSolve(A,b);
 return Object.fromEntries(states.map((s,i)=>[s,solved[i]]));
}
export function residual(states,model,V,policy=null,gamma=config.gamma){
 return Math.max(...states.map(s=>{
  const q=actionValues(model,s,V,gamma),target=policy?q.reduce((z,v,a)=>z+policy[s][a]*v,0):Math.max(...q);
  return Math.abs(target-V[s]);
 }));
}
export function goalDistances(states,model){
 const distance={};let changed=true;
 while(changed){changed=false;for(const s of states){
  const candidate=Math.min(...model[s].flat().map(o=>o.terminal?1:distance[o.next]===undefined?Infinity:1+distance[o.next]));
  if(Number.isFinite(candidate)&&(distance[s]===undefined||candidate<distance[s])){distance[s]=candidate;changed=true;}
 }}
 if(states.some(s=>distance[s]===undefined))throw new Error('example requires every state to reach the goal');
 return distance;
}
export function shortestValues(states,distance,gamma=config.gamma){
 return Object.fromEntries(states.map(s=>{
  const d=distance[s],cost=Array.from({length:d-1},(_,i)=>gamma**i).reduce((a,b)=>a+b,0);
  return [s,config.step_reward*cost+gamma**(d-1)*config.goal_reward];
 }));
}
export function backupTrace(states,model,{inPlace=false,order=states,budgets=[10,30,60],gamma=config.gamma}={}){
 if(order.length!==states.length||new Set(order).size!==states.length||order.some(s=>!states.includes(s)))throw new Error('order must contain every active state once');
 if(!budgets.length||budgets.some(b=>!Number.isInteger(b)||b<=0||b%states.length))throw new Error('snapshots require positive whole-sweep budgets');
 let V=Object.fromEntries(states.map(s=>[s,0]));const snapshots=[],rows=[];let backups=0;
 while(backups<Math.max(...budgets)){
  const read=inPlace?V:{...V},next=inPlace?V:{...V};
  for(const s of order){
   const before=V[s],q=actionValues(model,s,read,gamma),after=Math.max(...q);next[s]=after;
   rows.push({backup:++backups,state:s,before,after,action_values:q});
   if(inPlace&&budgets.includes(backups))snapshots.push({backups,values:{...V},residual:residual(states,model,V,null,gamma)});
   if(backups===Math.max(...budgets))break;
  }
  V=next;
  if(!inPlace&&budgets.includes(backups))snapshots.push({backups,values:{...V},residual:residual(states,model,V,null,gamma)});
 }
 return {order:[...order],in_place:inPlace,snapshots,rows};
}
export function discountedReturn(rewards,gamma,tail=0){return rewards.reduceRight((g,r)=>r+gamma*g,tail);}
export function mdpDpWalkthroughData(){
 const {states,model}=gridModel(),distances=goalDistances(states,model),optimal=shortestValues(states,distances);
 const uniform=uniformPolicy(states),vUniform=evaluatePolicy(states,model,uniform),greedy=greedyPolicy(states,model,vUniform),vGreedy=evaluatePolicy(states,model,greedy);
 const order=[...states].sort((a,b)=>distances[a]-distances[b]||states.indexOf(a)-states.indexOf(b));
 const trajectory=[[0,2],[0,1],[0,0],[1,0],[2,0],[3,0]],rewards=[-.04,-.04,-.04,-.04,1];
 const tail=optimal['0,0'];
 return {kind:'exact-known-model',config:{...config},states,model,distances,optimal,
  prediction:{policy:uniform,values:vUniform,residual:residual(states,model,vUniform,uniform)},
  improvement:{policy:greedy,values:vGreedy,residual:residual(states,model,vGreedy,greedy),
   one_step_from_start:actionValues(model,key(config.start),vUniform)},
  propagation:{synchronous:backupTrace(states,model),near_first:backupTrace(states,model,{inPlace:true,order}),
   far_first:backupTrace(states,model,{inPlace:true,order:[...order].reverse()})},
  boundary:{trajectory,rewards,cut_after:2,tail_state:'0,0',tail,
   full_return:discountedReturn(rewards,config.gamma),truncated_target:discountedReturn(rewards.slice(0,2),config.gamma,tail),
   mistaken_terminal:discountedReturn(rewards.slice(0,2),config.gamma),terminal_reward:1,terminal_tail:0},
  budget:{state_backups:[10,30,60],outcomes_per_backup:4,environment_transitions:0,scheduling_cost_included:false},random_rollouts:0};
}
