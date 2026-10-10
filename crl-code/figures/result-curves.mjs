/** Re-layout recorded means and sample SDs; never fit or resample a curve. */
import {palette as C, text, line, circle, frame} from './visual-kit.mjs';

const styles={
 bandit_sample_average:[C.blue,''],bandit_constant_step:[C.orange,'8 5'],
 bandit_ucb:[C.purple,'8 5'],bandit_gradient:[C.teal,'3 5'],
 td0:[C.blue,''],td_lambda:[C.purple,'3 5'],mc_prediction:[C.orange,'8 5'],
 gvf_td:[C.blue,''],gvf_gtd_lambda:[C.orange,'8 5'],
 option_smdp_q:[C.blue,''],primitive_q:[C.orange,'8 5'],
 option_model:[C.blue,''],option_model_monte_carlo:[C.orange,'8 5'],
 option_value_iteration:[C.blue,''],primitive_value_iteration:[C.orange,'8 5'],
 linear_td:[C.blue,''],gtd2:[C.orange,'8 5'],true_online_td:[C.teal,'8 5'],
 'deep-dqn':[C.blue,''],'deep-double_dqn':[C.orange,'8 5'],
 'deep-c51':[C.purple,'8 5'],'deep-qr_dqn':[C.orange,'8 5'],
 'deep-ddpg':[C.blue,''],'deep-td3':[C.orange,'8 5'],'deep-sac':[C.purple,'3 5'],
 shared_nonlinear:[C.blue,''],isolated_nonlinear:[C.orange,'8 5'],
 lagged_nonlinear_td:[C.purple,'8 5'],online_nonlinear_td:[C.blue,''],
 neural_td_trace:[C.purple,'8 5'],neural_td0:[C.blue,''],
 dyna_q:[C.blue,''],prioritized_sweeping:[C.purple,'3 5'],q_learning:[C.orange,'8 5'],
 differential_td_offpolicy:[C.blue,''],wrong_behavior_mean_td:[C.orange,'8 5'],
 rvi_multistate:[C.blue,''],unnormalized_vi_multistate:[C.orange,'8 5'],
 differential_dyna:[C.blue,''],differential_q_multistate:[C.orange,'8 5'],
 mc_control:[C.blue,''],watkins_q_lambda:[C.orange,'8 5'],
 rtrl:[C.blue,''],tbptt:[C.orange,'8 5'],
 integrated_recent_model:[C.blue,''],integrated_cumulative_model:[C.orange,'8 5'],integrated_no_planning:[C.purple,'3 5'],
};
const clocks={environment_steps:'真实环境步',model_backups:'模型备份次数',model_sweeps:'模型扫描轮数（每轮6备份）',stream_observations:'收到的观测数',sequences:'序列数（每段8个输入）'};
const units={reward:'奖励',value:'价值',return:'回报',rmse:'联合误差',absolute_error:'价值误差',prediction:'预测值',RMSE:'预测值',reward_rate:'奖励/环境步',reward_per_environment_step:'奖励/环境步',squared_error:'预测误差平方'};
const metricLabels={frozen_evaluation_return:'冻结策略的外部回报'};
// Short legend labels keep mechanism comparisons readable on a phone; the
// full catalogue names and source identities remain on the surrounding page.
const labels={
 dyna_q:'Dyna-Q',prioritized_sweeping:'优先扫描',q_learning:'Q-learning',
 differential_td_offpolicy:'离策略差分 TD',wrong_behavior_mean_td:'负对照：行为奖励均值',
 rvi_multistate:'RVI：减去参考值',unnormalized_vi_multistate:'VI：不减参考值',
 differential_dyna:'Differential Dyna',differential_q_multistate:'Differential Q',
 mc_control:'首次访问 MC',watkins_q_lambda:'Watkins Q(λ)',
 'deep-c51':'C51','deep-qr_dqn':'QR-DQN','deep-dqn':'DQN',
 rtrl:'RTRL',tbptt:'TBPTT（3步梯度）',
 integrated_recent_model:'近期模型',integrated_cumulative_model:'累计模型',integrated_no_planning:'不作模型规划',
};
const f=x=>Number(x.toFixed(3)).toString();
// Axis labels need less precision than the underlying observations. Keep the
// recorded points unchanged, and reserve space for the longest displayed tick.
const tick=x=>Number(x.toPrecision(3)).toString();

/** Wrap display labels without shrinking fonts; CJK glyphs occupy one unit. */
export function wrapLabel(value,limit){
 const rows=[];let current='',width=0;
 for(const char of value){
  const size=/[\u0000-\u007f]/.test(char)?.53:1;
  if(width+size>limit&&current){rows.push(current);current='';width=0;}
  current+=char;width+=size;
 }
 if(current)rows.push(current);
 return rows;
}

export function checkedSeries(curves){
 const series=Object.entries(curves).map(([id,points])=>{
  if(!styles[id]||!Array.isArray(points)||points.length===0)throw new Error('Unsupported or empty series: '+id);
  let last=-Infinity;
  for(const p of points){
   if(![p.step,p.mean,p.std].every(Number.isFinite)||p.std<0||p.step<=last||!Number.isInteger(p.n)||p.n<2)
    throw new Error('Invalid recorded point: '+id);
   last=p.step;
  }
  return {id,points};
 });
 if(!series.length)throw new Error('No series');
 return series;
}

export function resultDomains(curves){
 const series=checkedSeries(curves),p=series.flatMap(s=>s.points);
 const low=Math.min(0,...p.map(d=>d.mean-d.std));
 const high=Math.max(0,...p.map(d=>d.mean+d.std));
 const span=high-low||1;
 // Preserve the complete SD band, including negative endpoints for a
 // nonnegative metric. A sample-SD band is not its support or a confidence set.
 return {x:[0,Math.max(...p.map(d=>d.step))],y:[low<0?low-span*.04:0,high+span*.08]};
}

export function renderResultCurves({id,name,metric,unit,budget,steps,curves,names},mobile=false){
 const series=checkedSeries(curves),domain=resultDomains(curves);
 if(!clocks[budget]||!units[unit])throw new Error('Unreviewed clock or unit');
 const clockLabel=metric==='frozen_evaluation_return'&&budget==='environment_steps'?'训练环境步（评估另计）':clocks[budget];
 const yTicks=Array.from({length:5},(_,i)=>domain.y[0]+(domain.y[1]-domain.y[0])*i/4);
 const width=mobile?360:720,left=Math.max(mobile?62:78,14+10*Math.max(...yTicks.map(y=>tick(y).length))),right=width-24;
 const labelSize=mobile?18:20,lineHeight=mobile?24:26;
 let body='',cursor=35;
 const addLines=(label,color=C.ink)=>{
  for(const row of wrapLabel(label,mobile?17:32)){
   body+=text(20,cursor,row,{size:labelSize,color});cursor+=lineHeight;
  }
 };
 addLines((labels[id]||name)+'：对照结果');cursor+=5;
 addLines((metricLabels[metric]||metric)+'（'+units[unit]+'）',C.muted);cursor+=8;
 const changeSequence=['rtrl','tbptt'].includes(id)?Math.floor(steps/2):null;
 if(changeSequence!==null){
  if(!Number.isInteger(steps)||steps<2)throw new Error('Sequence-change annotation needs the original budget');
  addLines('第'+changeSequence+'段后更换教师；EMA保留',C.muted);cursor+=5;
 }
 const changeStep=['integrated_recent_model','integrated_cumulative_model'].includes(id)?Math.floor(steps/2):null;
 if(changeStep!==null){
  if(!Number.isInteger(steps)||steps<2)throw new Error('Reward-change annotation needs the original budget');
  addLines('第'+changeStep+'步后奖励位置改变',C.muted);
  addLines('100步滑动窗口；变化时不清空',C.muted);cursor+=5;
 }
 for(const s of series){
  const [color,dash]=styles[s.id],rows=wrapLabel(labels[s.id]||names[s.id]||s.id,mobile?14:29);
  body+=line(22,cursor-6,51,cursor-6,{color,dash,width:3});
  for(const row of rows){body+=text(61,cursor,row,{size:18});cursor+=24;}
  cursor+=4;
 }
 body+=text(20,cursor+3,'均值 ± 1 个样本标准差',{size:16,color:C.muted});
 const clockRows=wrapLabel(clockLabel,mobile?15:32);
 const top=cursor+27,bottom=top+(mobile?240:275),height=bottom+105+23*(clockRows.length-1);
 const X=x=>left+(right-left)*x/domain.x[1];
 const Y=y=>bottom-(bottom-top)*(y-domain.y[0])/(domain.y[1]-domain.y[0]);
 for(let i=0;i<=4;i++){
  const y=yTicks[i];
  body+=line(left,Y(y),right,Y(y),{width:1,color:C.line});
  body+=text(left-9,Y(y)+5,tick(y),{size:16,anchor:'end',color:C.muted});
 }
 for(let i=0;i<=(mobile?2:4);i++){
  const x=domain.x[1]*i/(mobile?2:4);
  body+=line(X(x),bottom,X(x),bottom+5,{color:C.ink,width:1});
  body+=text(X(x),bottom+25,f(x),{size:16,anchor:'middle',color:C.muted});
 }
 const path=points=>points.map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(3)+' '+Y(p[1]).toFixed(3)).join('');
 // Draw both bands first, then both center lines, so a later band does not
 // obscure the focal line. Zero SD produces a zero-area polygon naturally.
 for(const s of series){
  const points=[...s.points.map(p=>[p.step,p.mean+p.std]),...s.points.toReversed().map(p=>[p.step,p.mean-p.std])];
  body+='<path data-series-band="'+s.id+'" d="'+path(points)+'Z" fill="'+styles[s.id][0]+'" opacity="0.12"/>';
 }
 if(changeSequence!==null)body+='<g data-change-after-sequence="'+changeSequence+'">'+line(X(changeSequence),top,X(changeSequence),bottom,{color:C.muted,dash:'3 5',width:1.5})+'</g>';
 if(changeStep!==null)body+='<g data-change-after-step="'+changeStep+'">'+line(X(changeStep),top,X(changeStep),bottom,{color:C.muted,dash:'3 5',width:1.5})+'</g>';
 for(const s of series){
  const [color,dash]=styles[s.id];
  body+='<path data-series="'+s.id+'" data-points="'+s.points.length+'" d="'+path(s.points.map(p=>[p.step,p.mean]))+'" fill="none" stroke="'+color+'" stroke-width="2.5"'+(dash?' stroke-dasharray="'+dash+'"':'')+'/>';
  const last=s.points.at(-1);body+=circle(X(last.step),Y(last.mean),3,{fill:color});
 }
 body+=line(left,top,left,bottom,{color:C.ink,width:1.5})+line(left,bottom,right,bottom,{color:C.ink,width:1.5});
 clockRows.forEach((label,i)=>{body+=text((left+right)/2,bottom+54+i*23,label,{size:18,anchor:'middle'});});
 const ns=[...new Set(series.flatMap(s=>s.points.map(p=>p.n)))].join('/');
 body+=text(20,height-17,'每点 n='+ns+'；逐点连接，未平滑',{size:16,color:C.muted});
 return frame({title:name+' 的已记录对照结果',description:metric+'，横轴为'+clockLabel+'。原始记录的均值及一个样本标准差；每条曲线保留全部点，同图使用相同坐标。',width,height,body,kind:'measured',heading:false});
}
