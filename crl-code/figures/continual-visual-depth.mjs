/** Four exact, deliberately small mechanisms. No sampled performance claims. */
import {palette as C,text,line,arrow,rect,circle,frame} from './visual-kit.mjs';

export function continualDepthData(){
  const loops=[{name:'A',reward:2,duration:1},{name:'B',reward:3,duration:3}].map(o=>({...o,rate:o.reward/o.duration,totalAt12:12/o.duration*o.reward}));
  const gvfAlpha=1,mcUpdate=(old,completeReturn)=>old+gvfAlpha*(completeReturn-old);
  const gvf={bonus:5,alpha:gvfAlpha,routes:[{name:'左路',energy:[1,1]},{name:'右路',energy:[1,2,1]}].map(r=>{
    const hit=r.energy.map((_,i)=>Number(i===r.energy.length-1));
    const continuation=hit.map(h=>1-h);
    const rewards=r.energy.map((e,i)=>5*hit[i]-e);
    const energyReturn=r.energy.reduce((a,b)=>a+b,0);
    return {...r,hit,continuation,rewards,energyReturn,hitReturn:1,taskReturn:rewards.reduce((a,b)=>a+b,0),learnedEnergy:mcUpdate(0,energyReturn),learnedHit:mcUpdate(0,1)};
  })};
  const gamma=.9;
  const options=[{name:'快路',duration:2,success:.6},{name:'稳路',duration:4,success:.9}].map(o=>{
    const reward=-Array.from({length:o.duration},(_,k)=>gamma**k).reduce((a,b)=>a+b,0);
    const mass=gamma**o.duration;
    const goalWeight=mass*o.success,otherWeight=mass*(1-o.success);
    return {...o,reward,mass,goalWeight,otherWeight,tail:10*goalWeight,value:reward+10*goalWeight};
  });
  const x=1,phi=1,target=2,alpha=1,gradient=(phi*x-target)*x,newPhi=phi-alpha*gradient;
  const heads=(p,scale=1)=>({feature:p*x,prediction:2/scale*p*x,q:[1/scale*p*x,1.5],model:.5/scale*p*x});
  const architecture={x,phi,target,alpha,gradient,newPhi,before:heads(phi),after:heads(newPhi),compensated:heads(newPhi,newPhi/phi)};
  return {kind:'exact-calculation',loops,mixture:{reward:5,duration:4,rate:5/4,meanPerDecisionRate:1.5},gvf,options:{gamma,goalValue:10,otherValue:0,items:options},architecture};
}

const f=(v,k=3)=>Number(v.toFixed(k)).toString();
const txt=(x,y,v,color=C.ink,size=18,anchor='start')=>text(x,y,String(v),{color,size,anchor});
const mid=(x,y,v,color=C.ink,size=18)=>txt(x,y,v,color,size,'middle');
const path=(d,color=C.ink,width=2,dash='')=>`<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}"${dash?` stroke-dasharray="${dash}"`:''}/>`;
const group=(x,y,body)=>`<g transform="translate(${x},${y})">${body}</g>`;
const dot=(x,y,color=C.teal,r=7)=>circle(x,y,r,{fill:color,stroke:color});
const panel=(x,y,w,title,body)=>group(x,y,txt(0,0,title,C.ink,21)+body+line(0,28,w,28,{color:C.line,width:1}));
const coin=(x,y,color=C.orange,r=7)=>circle(x,y,r,{fill:color,stroke:color})+line(x,y-r/2,x,y+r/2,{color:C.white,width:1.5});
const star=(x,y,color=C.orange)=>`<path d="M${x},${y-10} l3,7 8,1 -6,5 2,8 -7,-4 -7,4 2,-8 -6,-5 8,-1 Z" fill="${color}"/>`;

function loopPanel(o,w){
  const cx=70,cy=105,color=o.name==='A'?C.teal:C.blue;
  let s=circle(cx,cy,25,{fill:C.white,stroke:color})+mid(cx,cy+6,'S',color,21);
  s+=path(`M${cx-17},${cy-20} C${cx-48},${cy-76} ${cx+91},${cy-76} ${cx+68},${cy+5} C${cx+48},${cy+63} ${cx-17},${cy+64} ${cx-28},${cy+20}`,color,3);
  s+=arrow(cx-29,cy+30,cx-27,cy+20,{color,width:3});
  for(let i=0;i<o.reward;i++)s+=coin(w-95+i*21,76);
  s+=txt(w-110,113,`${o.reward} 奖励`,C.orange,20)+txt(w-110,149,`${o.duration} 秒`,color,20);
  s+=mid(w/2,210,`${o.reward} ÷ ${o.duration} = ${o.rate} / 秒`,color,24);
  return s;
}
function clockPanel(loops,w){
  const start=43,span=w-80,dx=span/12;
  let s='';loops.forEach((o,i)=>{
    const y=100+i*140,color=i===0?C.teal:C.blue;
    s+=txt(0,y+6,o.name,color,23)+line(start,y,start+span,y,{color:C.line,width:2});
    for(let t=0;t<=12;t++){
      const x=start+t*dx;s+=line(x,y-4,x,y+4,{color:C.line,width:1});
      if(t>0&&t%o.duration===0)for(let r=0;r<o.reward;r++)s+=coin(x,y-14-r*13,color,5);
    }
    s+=txt(start,y+33,'0',C.muted,18)+txt(start+span,y+33,'12 秒',C.muted,18,'end');
    s+=txt(w,y-50,`Σ ${o.totalAt12}`,color,22,'end');
  });return s;
}
function rewardRate(d,mobile){
  const width=mobile?360:720,w=mobile?312:316,body=mobile?
    panel(24,96,w,'1 · A：快而小',loopPanel(d.loops[0],w))+panel(24,355,w,'2 · B：慢而大',loopPanel(d.loops[1],w))+panel(24,614,w,'3 · 放到同一段真实时间',clockPanel(d.loops,w)):
    panel(28,96,w,'1 · A：快而小',loopPanel(d.loops[0],w))+panel(376,96,w,'2 · B：慢而大',loopPanel(d.loops[1],w))+panel(28,365,664,'3 · 放到同一段真实时间',clockPanel(d.loops,664));
  return frame({title:'按次数还是按时间？',description:'两个半马尔可夫自环都回到S，没有外部重置。A每1秒得2，B每3秒得3。每次决策奖励B较高，但每秒奖励A较高。底部相同12秒内A累计24，B累计12，圆点每枚表示1奖励。',width,height:mobile?925:675,kind:'exact',body,heading:mobile?{x:24,y:42,size:24}:undefined});
}

function trajectory(r,w){
  let s='',left=18,dx=(w-36)/3;
  for(let i=0;i<4;i++){
    const x=left+i*dx;
    s+=circle(x,75,16,{fill:i===3?C.bg:C.white,stroke:i===3?C.orange:C.blue});
    s+=mid(x,82,i===0?'S':i===3?'G':String(i),i===3?C.orange:C.ink,18);
    if(i<3){
      s+=arrow(x+19,75,x+dx-19,75,{color:C.blue,width:3});
      s+=mid(x+dx/2,50,`E=${r.energy[i]}`,C.teal,18)+mid(x+dx/2,117,`R=${r.rewards[i]}`,C.orange,18);
    }
  }
  s+=txt(0,160,'π右：沿右路到 G',C.blue,19)+txt(w,160,'到达后停止累计',C.muted,18,'end');
  if(w<400){s=s.replace(txt(w,160,'到达后停止累计',C.muted,18,'end'),txt(0,190,'到达后停止累计',C.muted,18));}
  return s;
}
function questionPanel(r,w){
  const x0=67,dx=(w-135)/2;
  let s=txt(0,63,'E',C.teal,22)+txt(0,117,'到达',C.orange,18)+txt(0,171,'γ',C.muted,22);
  r.energy.forEach((c,i)=>{
    const x=x0+i*dx;
    s+=rect(x-20,39,40,32,{fill:C.bg,stroke:C.teal,radius:4})+mid(x,62,c,C.teal,20);
    s+=(r.hit[i]?star(x,111):circle(x,111,8,{fill:C.white,stroke:C.orange}));
    s+=mid(x,170,r.continuation[i],C.muted,20);
  });
  s+=arrow(65,190,65,221,{color:C.teal,width:3})+arrow(w-65,190,w-65,221,{color:C.orange,width:3});
  s+=mid(65,253,'能耗 4',C.teal,23)+mid(w-65,253,'到达 1',C.orange,23);
  s+=mid(w/2,294,'v ← 完整累计量（α=1）',C.ink,18);
  return s;
}
function choicePanel(g,w){
  let s=txt(0,61,'左路也有独立经验',C.muted,18),ys=[132,226];
  g.routes.forEach((r,i)=>{
    const y=ys[i],color=i===0?C.teal:C.blue;
    s+=txt(0,y-37,r.name,color,20)+line(58,y,w-28,y,{color:C.line,width:2});
    for(let k=0;k<=r.energy.length;k++)s+=dot(58+k*(w-95)/r.energy.length,y,color,6);
    s+=txt(w,y-37,`${g.bonus}×1−${r.energyReturn}=${r.taskReturn}`,color,22,'end');
    if(i===0)s+=arrow(14,y+1,41,y+1,{color:C.teal,width:4});
  });
  return s+mid(w/2,294,'比较任务回报，选择左路',C.ink,19);
}
function gvfQuestions(d,mobile){
  const w=mobile?312:316,width=mobile?360:720;
  const body=mobile?
    panel(24,96,w,'1 · 同一段右路经验',trajectory(d.gvf.routes[1],w))+panel(24,350,w,'2 · 两个给定的问题',questionPanel(d.gvf.routes[1],w))+panel(24,700,w,'3 · 答案用于动作比较',choicePanel(d.gvf,w)):
    panel(28,96,664,'1 · 同一段右路经验',trajectory(d.gvf.routes[1],664))+panel(28,310,w,'2 · 两个给定的问题',questionPanel(d.gvf.routes[1],w))+panel(376,310,w,'3 · 答案用于动作比较',choicePanel(d.gvf,w));
  return frame({title:'同一经验，回答不同问题',description:'确定性右路能耗依次1、2、1，到达指示0、0、1。设计者指定策略沿右路和延续1、1、0。完整样本以MC步长1把起点的能耗答案改为4、到达答案改为1。左路的独立经验给能耗2、到达1。任务奖励每步为5倍到达减能耗，两路线总回报为3和1，因此选择左路。问题规格没有被这个答案更新自动发现。',width,height:mobile?1035:645,kind:'exact',body,heading:mobile?{x:24,y:42,size:24}:undefined});
}

function optionPaths(items,w){
  let s='';items.forEach((o,i)=>{
    const y=90+i*120,color=i===0?C.blue:C.teal,x0=28,x1=w-83;
    s+=txt(0,y-42,o.name,color,20)+txt(w,y-42,`τ=${o.duration}`,C.muted,19,'end');
    const span=x1-x0;
    for(let k=0;k<o.duration-1;k++){
      const x=x0+k*span/(o.duration-1);
      s+=circle(x,y,7,{fill:k===0?C.white:color,stroke:color});
      s+=arrow(x+11,y,x0+(k+1)*span/(o.duration-1)-10,y,{color,width:2});
    }
    s+=circle(x1,y,7,{fill:color,stroke:color});
    s+=arrow(x1,y,x1+24,y-24,{color,width:2})+arrow(x1,y,x1+24,y+24,{color,width:2});
    s+=star(x1+32,y-28,C.orange)+circle(x1+32,y+28,9,{fill:C.white,stroke:C.muted});
    s+=txt(w,y-22,f(o.success),C.orange,18,'end')+txt(w,y+35,f(1-o.success),C.muted,18,'end');
  });return s+mid(w/2,277,'末步分流；到 G 或 F 都停止',C.muted,18);
}
function optionKernel(items,w){
  let s='';items.forEach((o,i)=>{
    const y=96+i*120,color=i===0?C.blue:C.teal;
    s+=txt(0,y-42,o.name,color,20);
    const gX=82,fX=w-66;
    // Circle areas, not radii, encode the discounted endpoint masses.
    s+=circle(gX,y,41*Math.sqrt(o.goalWeight),{fill:C.bg,stroke:color})+star(gX,y,C.orange);
    s+=circle(fX,y,41*Math.sqrt(o.otherWeight),{fill:C.white,stroke:color})+mid(fX,y+6,'F',C.muted,18);
    s+=mid(gX,y+55,f(o.goalWeight,5),color,19)+mid(fX,y+55,f(o.otherWeight,5),C.muted,19);
  });return s;
}
function backupPanel(items,w){
  let s='',x0=w<400?125:230,scale=w<400?22:53;
  const rows=items.map((o,i)=>({o,y:90+i*100,color:i===0?C.blue:C.teal}));
  s+=line(x0,44,x0,222,{color:C.muted,width:1});
  rows.forEach(({o,y,color})=>{
    s+=rect(x0+o.reward*scale,y-15,-o.reward*scale,30,{fill:C.orange,stroke:C.orange,radius:2});
    s+=rect(x0,y-15,o.tail*scale,30,{fill:color,stroke:color,radius:2});
    s+=txt(0,y-37,o.name,color,20)+txt(w,y-37,`${f(o.value,4)}`,color,25,'end');
    s+=txt(x0-6,y+44,f(o.reward,3),C.orange,18,'end')+txt(x0+6,y+44,`+ ${f(o.tail,4)}`,color,18);
  });
  s+=txt(0,273,'段内成本',C.orange,18)+txt(w,273,'停止后价值 × 终点权重',C.blue,18,'end');
  if(w<400)s=s.replace(txt(w,273,'停止后价值 × 终点权重',C.blue,18,'end'),txt(0,305,'停止后价值 × 终点权重',C.blue,18));
  return s;
}
function optionComparison(d,mobile){
  const width=mobile?360:720,w=mobile?312:316,items=d.options.items;
  const body=mobile?
    panel(24,96,w,'1 · 内部执行',optionPaths(items,w))+panel(24,420,w,'2 · 折扣终点核',optionKernel(items,w))+panel(24,765,w,'3 · 高层备份',backupPanel(items,w)):
    panel(28,96,w,'1 · 内部执行',optionPaths(items,w))+panel(376,96,w,'2 · 折扣终点核',optionKernel(items,w))+panel(28,438,664,'3 · 高层备份',backupPanel(items,664));
  return frame({title:'到达同处，为何价值不同',description:'两条固定option每原始步都获得负1。快路2步后以0.6到G，否则到F；稳路4步后以0.9到G，否则到F。折扣0.9，G后续值10，F后续值0；两终点都只是option停止，不是世界终止。圆面积编码折扣终点质量。快路备份值2.96，稳路2.4659，成功概率更高没有抵消额外时间成本。',width,height:mobile?1110:755,kind:'exact',body,heading:mobile?{x:24,y:42,size:24}:undefined});
}

function encoderPanel(a,w){
  let s=rect(15,59,54,54,{fill:C.bg,stroke:C.line,radius:5});
  for(let k=0;k<4;k++)s+=line(23+10*k,67,23+10*k,105,{color:C.ink,width:4});
  s+=mid(42,144,'x=1',C.muted,19)+arrow(77,87,128,87,{color:C.ink,width:3});
  const y=157,x1=170,x2=w-39;
  s+=line(143,y,w,y,{color:C.line,width:2})+rect(x1-17,y-38,34,38,{fill:C.blue,stroke:C.blue,radius:2})+rect(x2-17,y-76,34,76,{fill:C.teal,stroke:C.teal,radius:2});
  s+=mid(x1,105,'1',C.blue,21)+mid(x2,66,'2',C.teal,21)+arrow(x1+24,121,x2-22,99,{color:C.orange,width:3});
  s+=txt(0,209,'仅更新编码器 φ：1 → 2',C.ink,20)+txt(0,246,'辅助目标：z = 2',C.muted,19);
  return s;
}
function headsPanel(a,w){
  const labels=['预测','控制','模型'],x0=86,x1=w-27,span=x1-x0;
  let s='';
  for(let i=0;i<3;i++){
    const y=77+i*76;s+=txt(0,y+7,labels[i],C.ink,20)+line(x0,y,x1,y,{color:C.line,width:2});
    if(i===0){
      s+=circle(x0+span/2,y,7,{fill:C.blue,stroke:C.blue})+dot(x1,y,C.teal)+arrow(x0+span/2+10,y-15,x1-9,y-15,{color:C.orange,width:2});
      s+=mid(x0+span/2,y+29,'2',C.blue,18)+mid(x1,y+29,'4',C.teal,18);
    }else if(i===1){
      s+=circle(x0+span/2,y,7,{fill:C.blue,stroke:C.blue})+dot(x1,y,C.teal);
      s+=path(`M${x0+span*.75},${y-15} v30`,C.muted,2,'4 3');
      s+=mid(x0+span/2,y+30,'L:1',C.blue,18)+mid(x1,y+30,'2',C.teal,18)+mid(x0+span*.75,y-23,'R:1.5',C.muted,18);
    }else{
      s+=circle(x0+span/2,y,9,{fill:C.white,stroke:C.blue})+dot(x1,y,C.teal);
      s+=arrow(x0+span/2+13,y,x1-12,y,{color:C.orange,width:2})+mid(x0+span/2,y+30,'0.5',C.blue,18)+mid(x1,y+30,'1',C.teal,18);
    }
  }return s;
}
function compensationPanel(a,w){
  let s='';const mobile=w<400;
  const xs=mobile?[52,157,265]:[90,332,574],labels=['预测','控制','模型'],after=['2z','z','0.5z'],fixed=['z','0.5z','0.25z'],results=['2','(1, 1.5)','0.5'];
  xs.forEach((x,i)=>{
    s+=mid(x,58,labels[i],C.ink,20)+mid(x,96,after[i],C.muted,20)+arrow(x,110,x,139,{color:C.orange,width:3})+mid(x,169,fixed[i],C.teal,20)+mid(x,212,results[i],C.teal,21);
  });
  s+=mid(w/2,255,'所有输入权重 ÷ 2，原答案恢复',C.ink,mobile?18:21);
  return s;
}
function featureDrift(d,mobile){
  const width=mobile?360:720,w=mobile?312:316,a=d.architecture;
  const body=mobile?
    panel(24,96,w,'1 · 同一观测，坐标改变',encoderPanel(a,w))+panel(24,391,w,'2 · 头部没更新，答案变了',headsPanel(a,w))+panel(24,735,w,'3 · 已知缩放可同步补偿',compensationPanel(a,w)):
    panel(28,96,w,'1 · 同一观测，坐标改变',encoderPanel(a,w))+panel(376,96,w,'2 · 头部没更新，答案变了',headsPanel(a,w))+panel(28,425,664,'3 · 已知缩放可同步补偿',compensationPanel(a,664));
  return frame({title:'表示变了，谁的答案也变',description:'标量观测x=1，经编码器z=φx，φ从1更新为2。保持三个读出参数不变，预测2z由2到4，左动作值z由1到2而右动作值仍1.5，预测物理后继0.5z由0.5到1。环境未变。纯坐标缩放可把三个输入权重同时减半恢复答案，但一般非线性表示改变不一定有这样的可逆补偿。',width,height:mobile?1025:720,kind:'exact',body,heading:mobile?{x:24,y:42,size:24}:undefined});
}

export function renderContinualDepth(d=continualDepthData()){
  const assets={};
  for(const [name,render] of Object.entries({'average-clock':rewardRate,'gvf-question-answer':gvfQuestions,'option-model-backup':optionComparison,'shared-feature-drift':featureDrift})){
    assets[`concept-depth-${name}.svg`]=render(d,false);
    assets[`concept-depth-${name}.mobile.svg`]=render(d,true);
  }
  assets['concept-depth-data.json']=JSON.stringify(d,null,2)+'\n';
  return assets;
}
