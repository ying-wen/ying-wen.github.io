/** Exact, supplied-data diagnostics. No RNG, policy training, or environment rollouts. */
export const config = Object.freeze({gamma:.9,alpha:.25,epsilon:.5});
export const oldQ = Object.freeze({A:Object.freeze([2,0]),B:Object.freeze([1,5]),C:Object.freeze([4,0])});
export const secondQ = Object.freeze({A:Object.freeze([2,0]),B:Object.freeze([4,2]),C:Object.freeze([2,3])});
export const policy = Object.freeze({A:Object.freeze([.75,.25]),B:Object.freeze([.25,.75]),C:Object.freeze([.75,.25])});
export const task = Object.freeze({A:Object.freeze([{action:'go',next:'B',reward:1},{action:'exit',next:'T',reward:0}]),B:Object.freeze([{action:'fast',next:'C',reward:2},{action:'detour',next:'T',reward:4}]),C:Object.freeze([{action:'finish',next:'T',reward:3},{action:'quit',next:'T',reward:-1}])});
export const trace = Object.freeze({states:Object.freeze(['A','B','C','T']),actions:Object.freeze([0,0,0]),rewards:Object.freeze([1,2,3])});
export const greedy = q => q.indexOf(Math.max(...q));
export const expected = (q,p) => q.reduce((sum,v,a)=>sum+p[a]*v,0);
export const update = (before,target,alpha=config.alpha) => before+alpha*(target-before);

export function oneStepTargets({reward=1,next='B',action=0,terminated=false,q=oldQ,q2=secondQ,pi=policy}={}){
 if(terminated)return {sarsa:reward,expected:reward,q_learning:reward,double_q1:reward,double_q2:reward,double_coin_mean:reward};
 const g1=greedy(q[next]),g2=greedy(q2[next]);
 const double_q1=reward+config.gamma*q2[next][g1],double_q2=reward+config.gamma*q[next][g2];
 return {sarsa:reward+config.gamma*q[next][action],expected:reward+config.gamma*expected(q[next],pi[next]),q_learning:reward+config.gamma*Math.max(...q[next]),double_q1,double_q2,double_coin_mean:(double_q1+double_q2)/2};
}

export function nStepTarget({start=0,n=2,endpoint='sample',q=oldQ,pi=policy,path=trace,terminated=true}={}){
 if(!Number.isInteger(n)||n<1)throw new RangeError('n must be a positive integer');
 const h=Math.min(start+n,path.rewards.length);
 let g=0;for(let k=start;k<h;k++)g+=config.gamma**(k-start)*path.rewards[k];
 if(h<path.rewards.length||!terminated){
  const s=path.states[h];
  if(!q[s])throw new RangeError('a nonterminal boundary needs a value table');
  if(endpoint==='sample'&&path.actions[h]===undefined)throw new RangeError('sample endpoint needs a recorded boundary action');
  g+=config.gamma**(h-start)*(endpoint==='expected'?expected(q[s],pi[s]):q[s][path.actions[h]]);
 }
 return g;
}

/** Paper mixed-error recursion, or book (7.17) with its on-policy control variate. */
export function branchTarget({start=0,n=2,sigma=0,version='paper',q=oldQ,pi=policy,path=trace,terminated=true}={}){
 if(sigma<0||sigma>1)throw new RangeError('sigma must lie in [0,1]');
 if(!['paper','book_cv'].includes(version))throw new RangeError('unknown Q(sigma) version');
 const h=Math.min(start+n,path.rewards.length),terminal=terminated&&h===path.rewards.length;
 let g=terminal?path.rewards[h-1]:q[path.states[h]][path.actions[h]];
 for(let k=terminal?h-2:h-1;k>=start;k--){
  const s=path.states[k+1],a=path.actions[k+1],v=expected(q[s],pi[s]),p=pi[s][a];
  const continuation=version==='book_cv'?v+(sigma+(1-sigma)*p)*(g-q[s][a]):sigma*g+(1-sigma)*(v+p*(g-q[s][a]));
  g=path.rewards[k]+config.gamma*continuation;
 }
 return g;
}

/** Delayed online Sarsa on an already supplied episode; never reads future rewards. */
export function delayedUpdates(n=2){
 if(!Number.isInteger(n)||n<1)throw new RangeError('n must be a positive integer');
 const q=Object.fromEntries(Object.entries(oldQ).map(([s,row])=>[s,[...row]]));
 const T=trace.rewards.length,events=[];let version=0;
 for(let clock=1;clock<=T+n-1;clock++){
  const start=clock-n;
  if(start<0){events.push({clock,observed_transition:clock<=T?clock:null,waiting:true,read_version:version});continue;}
  const h=Math.min(start+n,T),g=trace.rewards.slice(start,h).reduce((v,r,k)=>v+config.gamma**k*r,0)+(h<T?config.gamma**(h-start)*q[trace.states[h]][trace.actions[h]]:0);
  const state=trace.states[start],action=trace.actions[start],before=q[state][action],after=update(before,g);
  events.push({clock,observed_transition:clock<=T?clock:null,waiting:false,state,action,horizon:h,bootstrap:h<T,read_version:version,target:g,before,after});
  q[state][action]=after;version++;
 }
 return {events,final_q:q};
}

/** Independent finite enumeration: unbiased evaluation of the selected action can underestimate max q. */
export function doubleBiasDiagnostic(){
 const selectors=[[-1,0],[-1,2],[1,0],[1,2]],evaluators=[[-1,0],[-1,2],[1,0],[1,2]];
 let single=0,double=0,selected_truth=0;
 for(const q1 of selectors){const a=greedy(q1);single+=Math.max(...q1)/4;selected_truth+=[0,1][a]/4;for(const q2 of evaluators)double+=q2[a]/16;}
 return {true_q:[0,1],single_max_mean:single,double_mean:double,selected_truth_mean:selected_truth,true_max:1};
}

export function controlTargetsWalkthroughData(){
 const one=oneStepTargets(),two={sarsa:nStepTarget(),endpoint_expected:nStepTarget({endpoint:'expected'}),tree_backup:branchTarget(),paper_sigma_half:branchTarget({sigma:.5}),book_cv_sigma_half:branchTarget({sigma:.5,version:'book_cv'})};
 const prefix={states:['A','B','C'],actions:[0,0,0],rewards:[1,2]};
 const qAfterC={...oldQ,C:[update(4,3),0]};
 const qTrue={C:[3,-1],B:[3.8,4],A:[4.555,0]},qOptimal={C:[3,-1],B:[4.7,4],A:[5.23,0]};
 return {metadata:{kind:'exact',sampled_runs:0,purpose:'fixed-data targets, boundary and update-order diagnostics',source:'Sutton & Barto (2020), §§6.4–6.7,7.1–7.6; De Asis et al. (2018), Eqs.13–14'},config,task,trace,old_q:oldQ,second_q:secondQ,policy,
  one_step:{targets:one,updates:Object.fromEntries(Object.entries(one).filter(([name])=>name!=='double_coin_mean').map(([name,g])=>[name,update(2,g)])),double_role_expectations:{mean_target:one.double_coin_mean,selected_table_after:update(2,one.double_coin_mean),two_table_average_after:2+config.alpha*(one.double_coin_mean-2)/2},update_scope:'double_q1/double_q2 denote the selected table; double_role_expectations separately tracks the two-table average',sarsa_conditional_mean:policy.B.reduce((v,p,a)=>v+p*oneStepTargets({action:a}).sarsa,0),double_bias:doubleBiasDiagnostic()},
  multistep:{two_step:two,three_step:{sarsa:nStepTarget({n:3}),tree_backup:branchTarget({n:3})},q_true_under_frozen_behavior:qTrue,q_optimal:qOptimal},
  timing:{n2:delayedUpdates(2),n3:delayedUpdates(3),truncated_prefix_target:nStepTarget({path:prefix,terminated:false}),wrong_terminal_prefix_target:nStepTarget({path:prefix}),terminal_full_return:nStepTarget({n:3}),wrong_late_boundary_target:nStepTarget({q:qAfterC}),boundary_q_before:4,boundary_q_after:3.75}};
}
