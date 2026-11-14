/**
 * Draw the supplied finite diagnostic, with no training or fitted curves.
 * Run from any directory: node /path/to/knowledge-transport.mjs
 * Source numbers: python3 ../tutorials/knowledge_transport.py demo
 */
import {execFileSync} from 'node:child_process';
import {writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {palette as C,text,line,arrow,rect,circle,frame} from './visual-kit.mjs';

const t=(x,y,value,size=20,color=C.ink,anchor='start',weight=400)=>text(x,y,value,{size,color,anchor,weight});
const n=(value,digits=4)=>Number(value.toFixed(digits)).toString();
const group=(x,y,body)=>`<g transform="translate(${x} ${y})">${body}</g>`;

export function knowledgeTransportData(){
 const script=fileURLToPath(new URL('../tutorials/knowledge_transport.py',import.meta.url));
 return JSON.parse(execFileSync('python3',[script,'demo'],{encoding:'utf8',maxBuffer:1024*1024}));
}

function checked(data){
 if(data.schema!=='crl-knowledge-transport-v1'||data.evidence_kind!=='exact_finite_diagnostic')throw Error('Expected the exact finite transport diagnostic');
 const paths=data.paths.reference;
 if(paths.length!==2||paths[0].states.join(',')!=='S,A,X'||paths[1].states.join(',')!=='S,B,C,Y')throw Error('The task figure requires the specified two continuations');
 if(data.paths.old.some(p=>p.physical_terminal)||paths.some(p=>!p.physical_terminal))throw Error('Option stops and physical terminals must remain distinct');
 if(paths.some(p=>p.rewards.slice(0,-1).some(r=>r!==0)))throw Error('The task figure labels all nonterminal rewards as zero');
 return data;
}

function stateNode(name,x,y,{stop=false,terminal=false,radius=28,color=C.blue}={}){
 let b='';
 if(terminal)b+=rect(x-radius,y-radius,radius*2,radius*2,{fill:C.ink,stroke:C.ink,radius:3})+t(x,y+7,name,22,C.white,'middle',600);
 else {
  b+=circle(x,y,radius,{fill:C.white,stroke:color,strokeWidth:2.5});
  if(stop)b+=circle(x,y,radius+6,{fill:'none',stroke:color,strokeWidth:2});
  b+=t(x,y+7,name,22,color,'middle',600);
 }
 return b;
}
function connect(from,to,{fromRadius=29,toRadius=29,...style}={}){
 const [x,y]=from,[u,v]=to,dx=u-x,dy=v-y,len=Math.hypot(dx,dy);
 return arrow(x+dx/len*fromRadius,y+dy/len*fromRadius,u-dx/len*toRadius,v-dy/len*toRadius,{head:10,width:3,...style});
}

function task(data,mobile){
 const width=mobile?400:900;
 const positions=mobile?{S:[54,310],A:[190,200],B:[140,460],C:[286,460],X:[350,200],Y:[286,610]}:
  {S:[110,270],A:[410,160],B:[320,370],C:[560,370],X:[790,160],Y:[790,370]};
 const stops=new Set(data.paths.old.map(p=>p.states.at(-1))),terminals=new Set(data.paths.reference.map(p=>p.states.at(-1)));
 let b=mobile?t(24,38,'同一世界：option 停止以后',20)+t(24,67,'任务仍然继续',20):'';
 b+=t(24,mobile?106:77,`六个物理状态 · γ=${n(data.gamma)}`,mobile?18:20,C.muted);
 b+=t(24,mobile?135:106,'旧策略；每条箭头是一条原子转移',mobile?17:20,C.muted);
 for(const path of data.paths.reference)for(let i=1;i<path.states.length;i++){
  const from=path.states[i-1],to=path.states[i],continued=stops.has(from);
  b+=connect(positions[from],positions[to],{fromRadius:stops.has(from)?35:29,toRadius:stops.has(to)?36:29,color:continued?C.teal:C.blue});
 }
 for(const state of data.states)b+=stateNode(state,...positions[state],{stop:stops.has(state),terminal:terminals.has(state)});
 const oldUpper=data.paths.old[0],oldLower=data.paths.old[1],rewards=data.paths.reference.map(p=>p.rewards.at(-1));
 if(mobile){
  b+=t(95,214,'upper',18,C.blue,'middle')+t(95,238,`p=${n(oldUpper.probability)}`,17,C.blue,'middle');
  b+=t(32,410,'lower',18,C.blue)+t(32,434,`p=${n(oldLower.probability)}`,17,C.blue);
  b+=t(213,420,'forward',18,C.blue,'middle');
  b+=t(270,165,`exit · +${n(rewards[0])}`,18,C.teal,'middle');
  b+=t(308,540,'probe',18,C.teal)+t(308,567,`+${n(rewards[1])}`,20,C.teal);
  b+=t(190,254,`τ=${oldUpper.duration}`,19,C.blue,'middle')+t(242,512,`τ=${oldLower.duration}`,19,C.blue,'end');
  b+=t(350,254,'终止',18,C.ink,'middle')+t(286,665,'终止',18,C.ink,'middle');
  b+=arrow(25,706,62,706,{color:C.blue})+t(77,713,'route option 内部行为',18);
  b+=arrow(25,743,62,743,{color:C.teal})+t(77,750,'给定续行；外部奖励见箭头',17)+t(77,777,'其余边外部奖励为 0',17);
  b+=stateNode('A',43,823,{stop:true,radius:13})+t(77,820,'双圈：option 停止',17)+t(77,847,'物理任务仍继续',17);
  b+=t(24,894,'τ：原子转移 / 次 option 调用',17,C.muted);
  b+=t(24,923,'给定策略的精确诊断；无训练结果',17,C.muted);
 }else{
  b+=t(225,179,'upper',19,C.blue,'middle')+t(225,206,`p=${n(oldUpper.probability)}`,18,C.blue,'middle');
  b+=t(176,353,'lower',19,C.blue,'middle')+t(176,379,`p=${n(oldLower.probability)}`,18,C.blue,'middle');
  b+=t(440,344,'forward',19,C.blue,'middle');
  b+=t(600,128,`exit · +${n(rewards[0])}`,20,C.teal,'middle');
  b+=t(677,340,`probe · +${n(rewards[1])}`,20,C.teal,'middle');
  b+=t(410,220,`τ=${oldUpper.duration}`,20,C.blue,'middle')+t(560,429,`τ=${oldLower.duration}`,20,C.blue,'middle');
  b+=t(790,220,'episode 终止',18,C.ink,'middle')+t(790,429,'episode 终止',18,C.ink,'middle');
  b+=arrow(28,480,67,480,{color:C.blue})+t(82,487,'option 内部行为',18);
  b+=arrow(310,480,349,480,{color:C.teal})+t(364,487,'给定续行；非零外部奖励见箭头，其余为 0',18);
  b+=stateNode('A',43,529,{stop:true,radius:13})+t(78,536,'双圈仅表示 option 停止',18);
  b+=t(380,536,'τ：原子转移 / 次调用；给定策略，无训练结果',18,C.muted);
 }
 return frame({width,height:mobile?954:565,kind:'exact',title:'同一世界：option 停止以后，任务仍继续',description:`六个物理状态S、A、B、C、X、Y，只画给定的S到A到X和S到B到C到Y。蓝箭头是route option内部行为，A与C的双圈是option停止位置，调用时长分别为${oldUpper.duration}和${oldLower.duration}条原子转移；青箭头是在停止之后的给定续行，外部奖励分别为${n(rewards[0])}和${n(rewards[1])}，其余图中边奖励零。黑色方形X和Y才结束物理任务。概率为给定旧策略，数据由knowledge_transport.py demo输出；这是精确有限诊断，不是训练或性能实验。`,body:b,heading:mobile?false:{x:24,y:39,size:26}});
}

function reversiblePanel(data,w){
 const d=data.scenarios.bijective_recoding,left=50,right=w-50,yt=128,yb=244;
 let b=t(16,34,'1 · 只换坐标',22,C.blue)+t(16,68,'物理行为不变，读出同步搬运',16,C.muted);
 b+=t(left,99,'旧坐标',17,C.muted,'middle')+t(right,99,'新坐标',17,C.muted,'middle');
 for(const [state,y,color] of [['A',yt,C.blue],['C',yb,C.teal]]){
  const mapped=d.mapping[state],newY=mapped==='A'?yt:yb;
  b+=connect([left,y],[right,newY],{fromRadius:27,toRadius:27,color,width:2.5});
  b+=stateNode(state,left,y,{radius:23,color})+stateNode(mapped,right,newY,{radius:23,color});
  b+=t(left,y+47,`读出 ${n(d.old.continuation_readout[state])}`,17,color,'middle');
  b+=t(right,newY+47,`读出 ${n(d.transported.continuation_readout[mapped])}`,17,color,'middle');
 }
 b+=t(16,329,'只展开 A / C 的端点坐标',16,C.muted);
 const oldK=d.old.discounted_endpoint_kernel,newK=d.transported.discounted_endpoint_kernel;
 b+=t(16,365,`旧 K：(${n(oldK.A)}, ${n(oldK.C)})`,16);
 b+=t(16,395,`新 K：(${n(newK.A)}, ${n(newK.C)})`,16);
 const barX=54,barW=w-91,scale=barW/1.6;
 for(const [i,label,value] of [[0,'旧',d.old.discounted_backup],[1,'新',d.transported.discounted_backup]]){
  const y=438+i*43;
  b+=t(16,y+17,label,18)+rect(barX,y,value*scale,22,{fill:C.blue,stroke:'none',radius:2})+t(barX+value*scale+7,y+17,n(value),17,C.blue);
 }
 b+=t(16,539,`Q：${n(d.old.discounted_backup)} → ${n(d.transported.discounted_backup)}`,22,C.blue);
 b+=t(16,572,'相同颜色追踪相同物理端点',16,C.muted);
 return b;
}

function policyPanel(data,w){
 const d=data.scenarios.policy_change,oldP=d.old.endpoint_probability.A,newP=d.physical_true.endpoint_probability.A;
 const x=18,barW=w-36;
 let b=t(16,34,'2 · 同坐标，行为改变',21,C.orange)+t(16,68,'upper 概率决定到达 A 或 C',16,C.muted);
 for(const [i,prefix,p] of [[0,'旧',oldP],[1,'当前',newP]]){
  const y=117+i*85;
  b+=t(x,y,`${prefix}：P(upper | S)=${n(p)}`,17);
  b+=rect(x,y+16,barW*p,25,{fill:C.blue,stroke:'none',radius:0})+rect(x+barW*p,y+16,barW*(1-p),25,{fill:C.teal,stroke:'none',radius:0});
  b+=t(x+barW*p/2,y+35,'A',17,C.white,'middle')+t(x+barW*(p+(1-p)/2),y+35,'C',17,C.white,'middle');
 }
 const X=v=>62+(w-111)*v/1.6,reference=d.current_decision.reference;
 b+=t(16,302,`reference = ${n(reference)}`,18,C.muted);
 b+=line(X(reference),320,X(reference),493,{color:C.muted,dash:'5 5',width:2});
 for(const [i,prefix,value,color] of [[0,'旧',d.old.discounted_backup,C.blue],[1,'当前',d.physical_true.discounted_backup,C.orange]]){
  const y=335+i*99;
  b+=t(16,y+18,prefix,17)+rect(X(0),y,X(value)-X(0),25,{fill:color,stroke:'none',radius:2})+t(X(0)+9,y+19,n(value),17,C.white);
  b+=t(62,y+58,i?'选择 route':'选择 reference',17,color);
 }
 b+=line(X(0),510,X(1.6),510,{color:C.ink})+t(X(0),535,'0',16,C.muted,'middle');
 b+=t(16,572,'条长为 Q；虚线是同一参照值',16,C.muted);
 return b;
}

function lossyPanel(data,w){
 const d=data.scenarios.lossy_merge,left=54,right=w-54,mid=w/2,y=130;
 const low=d.physical_readouts.A,high=d.physical_readouts.C,c=d.minimax_readout,error=d.unavoidable_max_readout_error;
 let b=t(16,34,'3 · A / C 合并成 Z',21,C.purple)+t(16,68,'行为未改；两个条件答案',16,C.muted)+t(16,92,'压到同一坐标',16,C.muted);
 b+=connect([left,y],[mid,250],{fromRadius:27,toRadius:29,color:C.purple})+connect([right,y],[mid,250],{fromRadius:27,toRadius:29,color:C.purple});
 b+=stateNode('A',left,y,{radius:24,color:C.blue})+stateNode('C',right,y,{radius:24,color:C.teal});
 b+=t(left,y+48,`读出 ${n(low)}`,18,C.blue,'middle')+t(right,y+48,`读出 ${n(high)}`,18,C.teal,'middle');
 b+=stateNode(d.mapping.A,mid,250,{radius:27,color:C.purple})+t(mid,302,'只能共用一个读出 c',18,C.purple,'middle');
 const X=v=>32+(w-64)*(v-low)/(high-low);
 b+=line(X(low),365,X(high),365,{color:C.ink});
 for(const [v,color,label] of [[low,C.blue,'A'],[high,C.teal,'C']])b+=circle(X(v),365,6,{fill:color})+t(X(v),342,`${label}=${n(v)}`,17,color,'middle');
 b+=circle(X(c),365,7,{fill:C.purple})+line(X(c),375,X(c),416,{color:C.purple,width:2})+t(X(c),445,`c=${n(c)}`,19,C.purple,'middle');
 b+=line(X(low),394,X(c),394,{color:C.purple,width:2})+line(X(c),394,X(high),394,{color:C.purple,width:2});
 for(const v of [low,c,high])b+=line(X(v),388,X(v),400,{color:C.purple,width:2});
 b+=t((X(low)+X(c))/2,422,n(c-low),16,C.purple,'middle')+t((X(c)+X(high))/2,422,n(high-c),16,C.purple,'middle');
 b+=t(mid,491,`max(|c−${n(low)}|, |c−${n(high)}|)`,18,C.purple,'middle');
 b+=t(mid,529,`最小可能最大误差 ≥ ${n(error)}`,20,C.purple,'middle');
 b+=t(16,572,'中点给出最小最大读出误差',16,C.muted);
 return b;
}

function comparison(data,mobile){
 const width=mobile?400:900,w=mobile?352:274,startY=mobile?134:114,step=mobile?626:0;
 const panels=[reversiblePanel,policyPanel,lossyPanel];
 let b=mobile?t(24,37,'坐标、行为与信息损失',21)+t(24,67,'改变的是不同对象',21):'';
 b+=t(24,mobile?104:78,'同一有限任务 · Q：外部奖励单位',mobile?17:20,C.muted);
 panels.forEach((draw,i)=>{
  const x=mobile?24:24+i*289,y=startY+i*step;
  b+=rect(x,y,w,602,{fill:C.bg,stroke:'none',radius:4})+group(x,y,draw(data,w));
 });
 const footer=mobile?startY+step*2+641:760;
 if(mobile)b+=t(24,footer,'K 已包含 γ^τ',18,C.muted)+t(24,footer+29,'所有量由同一脚本精确计算',17,C.muted)+t(24,footer+58,'给定策略与读出；不是训练结果',17,C.muted);
 else b+=t(24,footer,'K 已包含 γ^τ；所有量由同一脚本精确计算',19,C.muted)+t(24,footer+30,'给定策略与读出；不是训练曲线或方法优势',19,C.muted);
 return frame({width,height:footer+(mobile?84:54),kind:'exact',title:'坐标、行为与信息损失，改变的是不同对象',description:`左面板在可逆坐标交换后同时搬运端点核与续行读出，Q从${n(data.scenarios.bijective_recoding.old.discounted_backup)}保持为${n(data.scenarios.bijective_recoding.transported.discounted_backup)}；颜色跟踪物理端点，交叉箭头表示坐标交换。中面板坐标固定，upper概率从${n(data.scenarios.policy_change.old.endpoint_probability.A)}变为${n(data.scenarios.policy_change.physical_true.endpoint_probability.A)}，A和C概率的面积条随之变化；Q条长由${n(data.scenarios.policy_change.old.discounted_backup)}变为${n(data.scenarios.policy_change.physical_true.discounted_backup)}，跨过同一reference ${n(data.reference.discounted_backup)}，指定两选项的排序反转。右面板将A和C映到同一Z，共享读出无法同时给出${n(data.scenarios.lossy_merge.physical_readouts.A)}和${n(data.scenarios.lossy_merge.physical_readouts.C)}，数轴中点显示最小可能最大误差为${n(data.scenarios.lossy_merge.unavoidable_max_readout_error)}。数值来自Python精确有限诊断，未训练任何学习器。手机版将同样三个面板竖直排列。`,body:b,heading:mobile?false:{x:24,y:39,size:26}});
}

export function knowledgeTransportFigures(data){
 checked(data);
 return {
  'knowledge-transport-task.svg':task(data,false),
  'knowledge-transport-task.mobile.svg':task(data,true),
  'knowledge-transport-comparison.svg':comparison(data,false),
  'knowledge-transport-comparison.mobile.svg':comparison(data,true),
  'knowledge-transport-data.json':JSON.stringify(data,null,2)+'\n',
 };
}

if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const assets=knowledgeTransportFigures(knowledgeTransportData());
 for(const [name,body] of Object.entries(assets))writeFileSync(new URL('../../crl-figures/'+name,import.meta.url),body);
 console.log(JSON.stringify({evidence_kind:'exact',generated:Object.keys(assets)},null,2));
}
