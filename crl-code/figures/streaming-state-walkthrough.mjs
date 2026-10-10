/** Original finite state example. Diagnostics never become learner inputs. */
export const config = Object.freeze({gamma:.9,lam:.8,alpha:.1,beta:.9,eps:.1,retention:.5});
export const stream = Object.freeze([
 {next_observation:0,reward:2,terminal:false},
 {next_observation:0,reward:0,terminal:false},
 {next_observation:0,reward:1,terminal:true},
]);
export const initialState=()=>({w:0,h:1,z:0,max_v:0,gamma_in:0,k:0,cursor:0});
export function step(old,transition,{resetBeforeTerminal=false}={}){
 const terminal=transition.terminal;
 const next_h=terminal?0:config.retention*old.h+transition.next_observation;
 const gamma_out=terminal?0:config.gamma;
 const delta=transition.reward+gamma_out*old.w*next_h-old.w*old.h;
 const previous_z=terminal&&resetBeforeTerminal?0:old.z;
 const trace_used=old.gamma_in*config.lam*previous_z+old.h;
 const increment=delta*trace_used;
 const max_v=Math.max(config.beta*old.max_v,Math.abs(increment));
 const displacement=config.alpha*increment/(max_v+config.eps);
 const after={w:old.w+displacement,h:terminal?0:next_h,z:terminal?0:trace_used,
  max_v,gamma_in:gamma_out,k:old.k+1,cursor:old.cursor+1};
 return {after,row:{t:old.cursor,k:after.k,h:old.h,next_h,old_w:old.w,
  reward:transition.reward,delta,trace_used,increment,max_v,displacement,
  w:after.w,terminal,after:{...after}}};
}
export function suffix(start,{resetBeforeTerminal=false}={}){
 let state={...start};const rows=[];
 for(const transition of stream.slice(start.cursor)){
  const r=step(state,transition,{resetBeforeTerminal});state=r.after;rows.push(r.row);
 }
 return {rows,final:{...state}};
}
function permissions(){
 // These fixed features are retained ONLY by the replay variant.
 const features=[[1,.5,2,.9],[.5,.25,0,.9],[.25,0,1,0]];
 let w=0;
 for(const [x,next,reward,gamma] of features)w+=.1*(reward+gamma*w*next-w*x)*x;
 const [x,next,reward,gamma]=features[0],replay_delta=reward+gamma*w*next-w*x;
 return {strict_w:w,replay_delta,replay_w:w+.1*replay_delta*x,
  environment_transitions:3,strict_updates:3,replay_updates:4,
  strict_retained_transitions:0,replay_retained_transitions:3};
}
export function streamingStateData(){
 const first=step(initialState(),stream[0]),resumed=JSON.parse(JSON.stringify(first.after));
 const starts={full:resumed,clear_trace:{...resumed,z:0},clear_scale:{...resumed,max_v:0},
  clear_activity:{...resumed,h:0},clear_weight:{...resumed,w:0},
  weights_only:{w:resumed.w,h:0,z:0,max_v:0,gamma_in:resumed.gamma_in,k:1,cursor:1}};
 return {kind:'exact-mechanism',config:{...config},stream:[...stream],first:first.row,
  checkpoint:{...first.after},branches:Object.fromEntries(Object.entries(starts).map(([name,state])=>[name,suffix(state)])),
  wrong_terminal_reset:suffix(resumed,{resetBeforeTerminal:true}),permissions:permissions(),random_rollouts:0};
}
