/** Original deterministic teaching examples, not empirical performance data. */
export const key=(x,y)=>`${x},${y}`;
export const moves=[{name:'up',dx:0,dy:-1,glyph:'↑'},{name:'right',dx:1,dy:0,glyph:'→'},{name:'down',dx:0,dy:1,glyph:'↓'},{name:'left',dx:-1,dy:0,glyph:'←'}];
export const dot=(a,b)=>a.reduce((v,x,i)=>v+x*b[i],0);
export const add=(a,b)=>a.map((v,i)=>v+b[i]);
export const scale=(a,c)=>a.map(x=>c*x);

export function fourRooms(){
 const doors=['3,1','5,3','3,5','1,3'],states=[];
 for(let y=0;y<7;y++)for(let x=0;x<7;x++)if((x!==3&&y!==3)||doors.includes(key(x,y)))states.push(key(x,y));
 const floor=new Set(states),goal='6,0',gamma=.9;
 const rooms=[{id:'NW',x:0,y:0,doors:['3,1','1,3']},{id:'NE',x:4,y:0,doors:['3,1','5,3']},{id:'SW',x:0,y:4,doors:['1,3','3,5']},{id:'SE',x:4,y:4,doors:['5,3','3,5']}].map(r=>({...r,cells:states.filter(s=>{const [x,y]=s.split(',').map(Number);return x>=r.x&&x<r.x+3&&y>=r.y&&y<r.y+3;})}));
 const transition=(s,a)=>{
  if(s===goal)return {next:s,reward:0,terminal:true};
  const [x,y]=s.split(',').map(Number),m=moves[a],candidate=key(x+m.dx,y+m.dy),next=floor.has(candidate)?candidate:s;
  return {next,reward:next===goal?1:0,terminal:next===goal};
 };
 const options=rooms.flatMap(room=>room.doors.map(target=>{
  const allowed=new Set([...room.cells,...room.doors]),distance={[target]:0},queue=[target];
  for(let i=0;i<queue.length;i++){
   const [x,y]=queue[i].split(',').map(Number);
   for(const m of moves){const s=key(x+m.dx,y+m.dy);if(allowed.has(s)&&distance[s]===undefined){distance[s]=distance[queue[i]]+1;queue.push(s);}}
  }
  // Prefer aligning to the doorway's row, then column; ties remain deterministic.
  const policy={};
  for(const s of allowed){
   if(s===target||s===goal)continue;
   const [x,y]=s.split(',').map(Number),[tx,ty]=target.split(',').map(Number);
   const preferred=[y!==ty?(y<ty?2:0):(x<tx?1:3),0,1,2,3];
   policy[s]=preferred.find(a=>{const m=moves[a],n=key(x+m.dx,y+m.dy);return allowed.has(n)&&distance[n]===distance[s]-1;});
  }
  const initiation=[...allowed].filter(s=>s!==target&&s!==goal);
  const beta=s=>!room.cells.includes(s);
  const execute=start=>{
   if(!initiation.includes(start))throw new Error('Illegal option start '+start);
   let s=start,discount=1,reward=0;const rows=[];
   for(let t=0;t<50;t++){
    const action=policy[s],o=transition(s,action),stop=o.terminal||beta(o.next);
    reward+=discount*o.reward;discount*=gamma;
    rows.push({t,state:s,action: moves[action].name,next:o.next,reward:o.reward,beta:beta(o.next)?1:0,terminal:o.terminal,stop});s=o.next;
    if(stop)return {start,end:s,duration:rows.length,reward,kernel:o.terminal?0:discount,terminal:o.terminal,rows};
   }
   throw new Error('Option did not terminate');
  };
  return {id:`${room.id}:${target}`,room:room.id,target,initiation,policy,beta,execute};
 }));
 return {width:7,height:7,states,doors,rooms,goal,gamma,transition,options};
}

export function planningData(sweeps=5){
 const env=fourRooms(),nonterminal=env.states.filter(s=>s!==env.goal);
 const models=Object.fromEntries(nonterminal.map(s=>[s,env.options.filter(o=>o.initiation.includes(s)).map(o=>({option:o.id,...o.execute(s)}))]));
 const run=withOptions=>{
  let V=Object.fromEntries(env.states.map(s=>[s,0])),backups=0;const history=[{...V}];
  for(let k=0;k<sweeps;k++){
   const next={...V};
   for(const s of nonterminal){
    const q=moves.map((_,a)=>{backups++;const t=env.transition(s,a);return t.reward+(t.terminal?0:env.gamma*V[t.next]);});
    if(withOptions)for(const model of models[s]){backups++;q.push(model.reward+model.kernel*V[model.end]);}
    next[s]=Math.max(...q);
   }
   V=next;history.push({...V});
  }
  return {values:V,history,backups};
 };
 return {gamma:env.gamma,sweeps,states:env.states,goal:env.goal,start:'0,0',doors:env.doors,models,primitive:run(false),augmented:run(true)};
}

export function creditData(){
 const gamma=.9,lambda=.8,alpha=.1,decay=gamma*lambda;
 const states=['A','B','A','B','C'],rewards=[0,0,0,0,1],features={A:[1,0,0],B:[0,1,0],C:[0,0,1],D:[0,0,0]};
 let z=[0,0,0];
 const rows=states.map((state,t)=>{z=add(scale(z,decay),features[state]);return {t,state,reward:rewards[t],delta:rewards[t],trace:[...z],increment:scale(z,alpha*rewards[t])};});
 const values=rows.reduce((sum,r)=>add(sum,r.increment),[0,0,0]);
 const forward=states.map((state,t)=>{
  const n=states.length-t;
  const components=Array.from({length:n},(_,i)=>{
   const length=i+1,weight=length<n?(1-lambda)*lambda**i:lambda**i;
   let target=0;for(let j=0;j<length;j++)target+=gamma**j*rewards[t+j];
   return {length,weight,target,contribution:weight*target};
  });
  return {t,state,components,target:components.reduce((v,c)=>v+c.contribution,0)};
 });
 const sharedFeatures={A:[1,0],B:[1,1],C:[0,1],D:[1,0]};
 let sharedTrace=[0,0];
 for(const s of states)sharedTrace=add(scale(sharedTrace,decay),sharedFeatures[s]);
 const sharedWeights=scale(sharedTrace,alpha),sharedValues=Object.fromEntries(Object.entries(sharedFeatures).map(([s,x])=>[s,dot(sharedWeights,x)]));
 const input={A:1,B:2,C:3,D:1},old=[1,1],current=[2,.5];
 const value=(theta,x)=>theta[0]*Math.tanh(theta[1]*x);
 const gradient=(theta,x)=>{const h=Math.tanh(theta[1]*x);return [h,theta[0]*x*(1-h*h)];};
 let historical=[0,0],recomputed=[0,0];
 const nonlinear=states.map((s,t)=>{
  const theta=t<2?old:current,g=gradient(theta,input[s]),now=gradient(current,input[s]);
  historical=add(scale(historical,decay),g);recomputed=add(scale(recomputed,decay),now);
  return {t,state:s,theta,g,currentGradient:now,historical:[...historical],recomputed:[...recomputed]};
 });
 return {gamma,lambda,alpha,decay,states,rewards,rows,values,forward,sharedFeatures,sharedTrace,sharedWeights,sharedValues,nonlinear,old,current,input,value,gradient};
}

export function computeConceptData(){
 const env=fourRooms(),option=env.options.find(o=>o.id==='NW:3,1'),execution=option.execute('0,0');
 const costReward=execution.rows.reduce((s,_,t)=>s-env.gamma**t,0),kernel=env.gamma**execution.duration;
 return {provenance:'Original exact teaching calculations; deterministic known model, no sampled performance curves.',execution:{...execution,reward:costReward,rows:execution.rows.map(row=>({...row,reward:-1})),costReward,kernel,continuation:10,target:costReward+kernel*10,policy:option.policy,initiation:option.initiation},planning:planningData(),credit:creditData()};
}
