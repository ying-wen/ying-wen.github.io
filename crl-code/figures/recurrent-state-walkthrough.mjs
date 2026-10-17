/** Independent reverse-mode calculation. No learner update inside the sequence. */
export const config=Object.freeze({a:.5,b:1,w:1,target:1,alpha:.1,h0:0,finite_difference_epsilon:1e-6});
export const observations=Object.freeze([1,0,0,0]);

export function forward(inputs=observations,{a=config.a,b=config.b,tanh=false}={}){
 let h=0;
 return inputs.map((observation,i)=>{
  const old_h=h,z=a*old_h+b*observation;
  h=tanh?Math.tanh(z):z;
  return {t:i+1,observation,old_h,h,slope:tanh?1-h*h:1};
 });
}

export function bptt(inputs=observations,{a=config.a,b=config.b,w=config.w,target=config.target,window=inputs.length,tanh=false}={}){
 const rows=forward(inputs,{a,b,tanh}),final_h=rows.at(-1).h;
 let adjoint=(w*final_h-target)*w;
 const gradient_theta=[0,0],contributions=[];
 for(let i=rows.length-1;i>=Math.max(0,rows.length-window);i--){
  const r=rows[i],local=adjoint*r.slope;
  const contribution=[local*r.old_h,local*r.observation];
  gradient_theta[0]+=contribution[0];gradient_theta[1]+=contribution[1];
  contributions.unshift({t:r.t,adjoint,contribution});
  adjoint=local*a;
 }
 return {final_h,gradient_theta,gradient_w:(w*final_h-target)*final_h,contributions};
}

// Unroll every parameter-to-state path. This does not reuse the RTRL recurrence.
export function pathSensitivity(rows,a,end=rows.length,start=0){
 const result=[0,0];
 for(let origin=start;origin<end;origin++){
  let factor=rows[origin].slope;
  for(let later=origin+1;later<end;later++)factor*=a*rows[later].slope;
  result[0]+=factor*rows[origin].old_h;
  result[1]+=factor*rows[origin].observation;
 }
 return result;
}

export function loss(inputs=observations,parameters={}){
 const {w=config.w,target=config.target}=parameters;
 return .5*(w*forward(inputs,parameters).at(-1).h-target)**2;
}

export function recurrentStateData(){
 const raw=forward(),rows=raw.map((r,i)=>({t:r.t,observation:r.observation,old_h:r.old_h,h:r.h,sensitivity:pathSensitivity(raw,config.a,i+1)}));
 const final_h=raw.at(-1).h,residual=config.w*final_h-config.target,modes={};
 for(const [name,window] of [['full',4],['last_two',2],['detach_each',1]]){
  const {gradient_theta}=bptt(observations,{window});
  modes[name]={window,final_h,sensitivity:pathSensitivity(raw,config.a,4,4-window),gradient_theta,write_theta:gradient_theta.map(g=>-config.alpha*g),write_w:0};
 }
 const gradient_w=bptt().gradient_w;
 modes.readout_only={final_h,gradient_w,write_theta:[0,0],write_w:-config.alpha*gradient_w};
 const eps=config.finite_difference_epsilon;
 const finite_difference=['a','b'].map(key=>(loss(observations,{[key]:config[key]+eps})-loss(observations,{[key]:config[key]-eps}))/(2*eps));
 const erased=forward(observations.map(()=>0)),erasedPrediction=config.w*erased.at(-1).h;
 return {kind:'exact-mechanism',sampled_runs:0,config:{...config},observations:[...observations],rows,prediction:config.w*final_h,residual,loss:loss(),modes,finite_difference,
  erased_cue:{final_h:erased.at(-1).h,sensitivity:pathSensitivity(erased,config.a),prediction_for_positive_target:erasedPrediction,prediction_for_negative_target:erasedPrediction}};
}
