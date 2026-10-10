/** Exact, fixed-data-law teaching calculations; no random training. */
export const offpolicyFixture=Object.freeze({states:['A','B'],features:[1,2],behavior:[.9,.1],target:[0,1],gamma:.9,alpha:.01,beta:.05,initial:[1,0],steps:600});

export function eventDirections({w=1,h=34/65,state=0,action=1,rewards=[0,0]}={}){
 const {features:x,behavior:b,target:pi,gamma}=offpolicyFixture;
 const current=x[state],next=x[action],rho=pi[action]/b[action];
 const delta=rewards[state]+gamma*next*w-current*w,hx=current*h;
 return {state,action,probability:b[state]*b[action],rho,delta,hx,
  td:rho*delta*current,gtd2:rho*(current-gamma*next)*hx,
  tdc:rho*(delta*current-gamma*next*hx),auxiliary:(rho*delta-hx)*current};
}

export function modelMoments(rewards=[0,0],weights=offpolicyFixture.behavior){
 const {features:x,gamma}=offpolicyFixture;
 return {C:weights.reduce((z,d,i)=>z+d*x[i]**2,0),
  A:weights.reduce((z,d,i)=>z+d*x[i]*(x[i]-gamma*x[1]),0),
  b:weights.reduce((z,d,i)=>z+d*x[i]*rewards[i],0),
  bootstrapCross:weights.reduce((z,d,i)=>z+d*gamma*x[1]*x[i],0)};
}

export function expectedDirections(w,h,rewards=[0,0]){
 const totals={td:0,gtd2:0,tdc:0,auxiliary:0};
 for(let state=0;state<2;state++)for(let action=0;action<2;action++){
  const e=eventDirections({w,h,state,action,rewards});
  for(const key of Object.keys(totals))totals[key]+=e.probability*e[key];
 }
 return totals;
}

export function fixedLawTrajectory(method,steps=offpolicyFixture.steps){
 if(!['gtd2','tdc'].includes(method))throw new RangeError('method must be gtd2 or tdc');
 let [w,h]=offpolicyFixture.initial;
 const rows=[{t:0,w,h}];
 for(let t=1;t<=steps;t++){
  const d=expectedDirections(w,h);
  // All four events use the same OLD w,h, then their increments are averaged.
  const nextW=w+offpolicyFixture.alpha*d[method],nextH=h+offpolicyFixture.beta*d.auxiliary;
  w=nextW;h=nextH;rows.push({t,w,h});
 }
 return rows;
}

export function emphaticRecordedPath(){
 // A legal, deliberately given path under b; not drawn as a random sample.
 const actions=[1,1,0,1],rows=[];
 let state=0,w=1,F=0,previousRho=0;
 for(let t=0;t<actions.length;t++){
  const action=actions[t],oldW=w;
  F=1+offpolicyFixture.gamma*previousRho*F;
  const e=eventDirections({w,h:0,state,action});
  const trace=e.rho*F*offpolicyFixture.features[state];
  w+=offpolicyFixture.alpha*e.delta*trace;
  rows.push({t,state,action,oldW,F,rho:e.rho,trace,delta:e.delta,w});
  state=action;previousRho=e.rho;
 }
 return rows;
}

export function offpolicyGeometryData(){
 const moments=modelMoments(),hStar=-moments.A/moments.C;
 const events=[];for(let state=0;state<2;state++)for(let action=0;action<2;action++)events.push(eventDirections({state,action,h:hStar}));
 const v=[1,2],bellman=[1.8,1.8],projectedWeight=99/65,projected=[projectedWeight,2*projectedWeight];
 // sqrt(D_b) makes the D_b norm Euclidean. Draw equal scale on both axes.
 const whiten=a=>a.map((q,i)=>Math.sqrt(offpolicyFixture.behavior[i])*q);
 const emphasis=[.9,9.1],emphaticMoments=modelMoments([0,0],emphasis);
 const rewardMoments=modelMoments([1,0]),rewardEmphatic=modelMoments([1,0],emphasis);
 const VE=w=>.9*(w-1)**2+.1*(2*w)**2;
 const fixedPoints=[['value-regression',9/13],['behavior-MSPBE',rewardMoments.b/rewardMoments.A],['emphatic-projection',rewardEmphatic.b/rewardEmphatic.A]].map(([name,w])=>({name,w,values:[w,2*w],VEbehavior:VE(w)}));
 return {fixture:offpolicyFixture,moments,events,correctedStateMass:[.9,.1],targetStateMass:[0,1],
  directionsAtAuxiliaryEquilibrium:{w:1,h:hStar,...expectedDirections(1,hStar)},
  firstCoupledUpdate:{gtd2:fixedLawTrajectory('gtd2',1)[1],tdc:fixedLawTrajectory('tdc',1)[1]},
  geometry:{v,bellman,projected,projectedWeight,whitenedV:whiten(v),whitenedBellman:whiten(bellman),whitenedProjected:whiten(projected),
   MSBE:.58,MSPBE:578/1625,halfMSPBE:289/1625,negativeGradient:-578/1625},
  objectiveCurve:Array.from({length:81},(_,i)=>{const w=-1+i/40;return [w,(moments.A*w)**2/(2*moments.C)];}),
  emphasis:{interest:[1,1],lambda:0,mass:emphasis,normalized:[.09,.91],conditionalFollowon:[1,91],moments:emphaticMoments,
   recordedPath:emphaticRecordedPath()},
  trajectories:{gtd2:fixedLawTrajectory('gtd2'),tdc:fixedLawTrajectory('tdc')},
  rewardVariant:{rewards:[1,0],trueValues:[1,0],moments:rewardMoments,emphaticMoments:rewardEmphatic,fixedPoints},
  evidence:'Exact enumeration and deterministic fixed-data-law recurrences; given ETD path; no random training or control-performance claim.'};
}
