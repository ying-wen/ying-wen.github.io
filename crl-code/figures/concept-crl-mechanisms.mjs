/** Exact, deliberately small teaching examples. No paper performance measurements. */
export function averageCycle(steps=8){
 const rewards=Array.from({length:steps},(_,t)=>t%2?2:0),gain=1,bias=[0,1];
 return {rewards,gain,bias,centered:rewards.map(r=>r-gain),residuals:[0-gain+bias[1]-bias[0],2-gain+bias[0]-bias[1]]};
}
export function modelBackups(){
 const gamma=.9,values=[0,0,0,0],frames=[values.slice()];
 // Edges 0->1 and 1->2 were previously observed. The latest real step reveals 2->G.
 for(const s of [2,1,0]){values[s]=(s===2?1:0)+gamma*values[s+1];frames.push(values.slice());}
 return {gamma,frames,order:[2,1,0],realTransitions:1,modelBackups:2,terminal:3};
}
export function sharedExperience(){
 const q=[[1,2],[3,4]],v=[2,5],gain=.5,alpha=.1,eta=.1,reward=1,behaviorProbability=.25;
 const controlError=reward-gain+Math.max(...q[1])-q[0][1],predictionError=reward+.8*v[1]-v[0],rho=1/behaviorProbability;
 return {before:{q,v,gain},after:{q:[[1,q[0][1]+alpha*controlError],q[1].slice()],v:[v[0]+alpha*rho*predictionError,v[1]],gain:gain+eta*alpha*controlError},controlError,predictionError,rho,behaviorProbability,reward,model:{from:0,to:1,action:1,countBefore:0,countAfter:1},alpha,eta};
}
export function neuronFrames(){
 const evaluate=(u,v)=>{const h=Math.max(0,u),f=v*h,error=f-1;return {u,v,h,f,loss:.5*error*error,gradU:error*v*(u>0?1:0),gradV:error*h};};
 const aged=evaluate(-1,1),recycled=evaluate(.5,0),frames=[aged,recycled];
 let current=recycled;
 for(let i=0;i<2;i++){current=evaluate(current.u-.1*current.gradU,current.v-.1*current.gradV);frames.push(current);}
 return {x:1,y:1,alpha:.1,frames};
}
export function hindsightUpdate(){
 const before=-2,alpha=.5,gamma=.9;
 const original={goal:4,reward:-1,done:false,nextValue:-1},relabeled={goal:2,reward:0,done:true,nextValue:-1};
 for(const row of [original,relabeled]){row.target=row.reward+gamma*(row.done?0:row.nextValue);row.after=before+alpha*(row.target-before);}
 return {from:1,to:2,before,alpha,gamma,original,relabeled};
}
export const mechanismData=()=>({average:averageCycle(),planning:modelBackups(),architecture:sharedExperience(),plasticity:neuronFrames(),goals:hindsightUpdate()});
