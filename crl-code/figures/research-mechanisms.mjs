/** Deterministic teaching examples, not benchmark measurements. No dependencies. */
import {palette, text as drawText, line as drawLine, arrow as drawArrow, circle as drawCircle, rect as drawRect, frame} from './visual-kit.mjs';
export function computeMechanisms(){
 const x=[-1,1],v=x=>x*x;
 const retention=[0,1,5].map(k=>{const w=(k-1)/(k+1);return {k,w,oldLoss:.5*(w-1)**2,newLoss:.5*(w+1)**2};});
 const scoresA=[.7,.5,.6],scoresB=[.6,.6,.4],differences=scoresA.map((v,i)=>v-scoresB[i]);
 const learners=[0,.1,1].map(alpha=>{const q=[1,.5],steps=[];for(let t=0;t<10;t++){const a=q[0]>=q[1]?0:1,r=a;const before=[...q];q[a]+=alpha*(r-q[a]);steps.push({t,a,r,before,after:[...q]});}return {alpha,steps,total:steps.reduce((s,v)=>s+v.r,0)};});
 const policies=[.2,1,3].map(tau=>{const q=[0,1,2],z=q.reduce((s,x)=>s+Math.exp(x/tau),0),p=q.map(x=>Math.exp(x/tau)/z);return {tau,q,p,entropy:-p.reduce((s,x)=>s+x*Math.log(x),0)};});
 const machine=actions=>{let state=0;return actions.map(action=>{const before=state,reward=action==='D'&&state===1?1:0;if(action==='K'&&state===0)state=1;else if(reward)state=2;return {action,before,after:state,reward};});};
 return {model:{outcomes:x,mean:0,valueOfMean:v(0),meanValue:x.reduce((s,x)=>s+v(x),0)/2},retention,exploration:{noise:[4,4],learnable:[2,1]},experiments:{scoresA,scoresB,differences,mean:differences.reduce((s,x)=>s+x,0)/3,resample:[2,2,0],resampled:(differences[2]*2+differences[0])/3},learners,policies,reward:{KD:machine(['K','D']),DK:machine(['D','K'])}};
}
const C=palette;
// Positional adapters preserve the examples while using one visual vocabulary.
const txt=(x,y,s,n=20,c=C.ink,a='start')=>drawText(x,y,s,{size:n,color:c,anchor:a});
const line=(x,y,xx,yy,c=C.line,w=2,d='')=>drawLine(x,y,xx,yy,{color:c,width:w,dash:d?'6 5':''});
const dot=(x,y,c=C.blue,r=8)=>drawCircle(x,y,r,{fill:c});
const rect=(x,y,w,h,c='#eef5fa')=>drawRect(x,y,w,h,{fill:c,stroke:'none',radius:5});
const arrow=(x,y,xx,yy,c=C.blue)=>drawArrow(x,y,xx,yy,{color:c,width:3});
const num=v=>Number(v.toFixed(3));
function svg(title,desc,h,b){return frame({title,description:desc,height:h,body:b,kind:title.startsWith('高误差')?'schematic':'exact'});}
function key(x,y,c=C.teal){return `<circle cx="${x}" cy="${y}" r="10" fill="none" stroke="${c}" stroke-width="4"/><path d="M${x+10} ${y}h24m-6 0v9m-9 -9v7" stroke="${c}" stroke-width="4"/>`;}
function door(x,y,open=false){return `<path d="M${x} ${y+42}V${y}h32v42" fill="${open?'#d7eee7':'#d8e3eb'}" stroke="${open?C.teal:C.muted}" stroke-width="3"/>`+dot(x+25,y+25,open?C.teal:C.muted,3);}
export function renderMechanisms(d=computeMechanisms()){
 const out={};let b=txt(28,80,'后果是随机的；价值计算可能是非线性的。',20,C.muted);
 const X=x=>360+x*230,Y=v=>440-v*250;
 b+=line(80,440,650,440)+line(360,465,360,145);
 for(const x of [-1,0,1])b+=txt(X(x),474,x,19,C.muted,'middle');
 const pts=Array.from({length:121},(_,i)=>{const x=-1.1+i*2.2/120;return `${X(x)},${Y(x*x)}`;}).join(' ');
 b+=`<polyline points="${pts}" fill="none" stroke="${C.blue}" stroke-width="4"/>`+txt(510,141,'V(x) = x²',23,C.blue);
 b+=line(130,190,590,190,C.teal,3)+line(130,440,130,190,C.muted,2,'dash')+line(590,440,590,190,C.muted,2,'dash')+dot(130,190,C.teal)+dot(590,190,C.teal)+dot(360,440,C.orange,10);
 b+=txt(110,520,'−1',23,C.teal)+txt(587,520,'+1',23,C.teal)+txt(130,553,'各 1/2 概率',20,C.muted)+arrow(153,515,321,515,C.orange)+arrow(568,515,399,515,C.orange)+txt(360,522,'0',25,C.orange,'middle');
 b+=rect(65,590,270,94,'#fff1e6')+txt(85,626,'先平均后果，再算价值',20,C.orange)+txt(85,661,'V(E[X]) = 0',24,C.orange)+rect(375,590,280,94,'#e5f2ee')+txt(395,626,'先算价值，再平均',20,C.teal)+txt(395,661,'E[V(X)] = 1',24,C.teal);
 out['concept-research-model-expectation.svg']=svg('平均位置，不等于平均价值','随机后果为负一或正一，各占一半。抛物线价值为x平方；均值位置零的价值为零，两个真实后果的平均价值为一。',718,b);
 b=txt(28,80,'同一个参数 w，旧目标在 +1，新目标在 −1。',20,C.muted);
 const px=x=>355+x*220,py=y=>425-y*110;
 b+=line(63,425,655,425)+txt(61,133,'未加权任务损失',20,C.muted);
 for(const [center,c]of [[1,C.blue],[-1,C.orange]]){const p=Array.from({length:101},(_,i)=>{const x=-1.2+2.4*i/100;return `${px(x)},${py(.5*(x-center)**2)}`;}).join(' ');b+=`<polyline points="${p}" fill="none" stroke="${c}" stroke-width="4"/>`;}
 b+=txt(85,469,'新目标 −1',21,C.orange)+txt(535,469,'旧目标 +1',21,C.blue)+txt(448,128,'旧损失',20,C.blue)+txt(448,158,'新损失',20,C.orange);
 d.retention.forEach((r,i)=>{const y=528+i*89;b+=txt(30,y+7,`κ=${r.k}`,22)+line(141,y,575,y,C.line,3)+dot(px(r.w),y,C.teal,11)+txt(px(r.w),y+36,`w=${num(r.w)}`,18,C.teal,'middle');});
 b+=txt(28,831,'增大保留权重，会改变折中解的位置。',21,C.muted);
 out['concept-research-retention-conflict.svg']=svg('保留与适应怎样拉扯同一参数','精确二次例子。旧目标w等于一，新目标w等于负一。保留权重为零、一、五时，联合最优点分别为负一、零、三分之二。图中两条曲线是未加权任务损失。',868,b);
 b=txt(28,80,'重复看见一个区域：误差大，还是确实在学会？',20,C.muted);
 for(let region=0;region<2;region++){
  const y=134+region*334;b+=txt(28,y,region?'可学习的固定图案':'不可预测的下一帧',23,region?C.teal:C.orange);
  for(let f=0;f<3;f++){const x=38+f*110;for(let r=0;r<5;r++)for(let c=0;c<5;c++){const on=region?c===r||c===4-r:((r*7+c*11+f*13+r*c*f)%5<2);b+=rect(x+c*16,y+27+r*16,14,14,on?(region?C.teal:C.orange):'#eef2f4');}}
  const ex=[445,633],ey=val=>y+151-val*29;b+=line(433,y+151,657,y+151)+line(433,y+17,433,y+151);
  const arr=region?d.exploration.learnable:d.exploration.noise;
  b+=line(ex[0],ey(arr[0]),ex[1],ey(arr[1]),region?C.teal:C.orange,4);
  arr.forEach((v,i)=>{b+=dot(ex[i],ey(v),region?C.teal:C.orange)+txt(ex[i],ey(v)-14,v,21,C.ink,'middle');});
  b+=txt(435,y+188,'前',18,C.muted)+txt(622,y+188,'后',18,C.muted)+txt(437,y+229,`误差下降 = ${arr[0]-arr[1]}`,20,region?C.teal:C.orange);
 }
 b+=txt(28,814,'这里比较预测误差与进展；不是把噪声都视为新知识。',19,C.muted);
 out['concept-research-exploration-progress.svg']=svg('高误差与学习进展的区别','上方噪声图案不断改变，构造的误差一直为四。下方固定图案可学习，误差从二降到一。数值是解释信号差异的算例，不是图像模型训练结果。',850,b);
 b=txt(28,80,'任务：先拿钥匙，再开门。每次动作后回到同一位置。',19,C.muted);
 for(let row=0;row<2;row++){
  const seq=row?d.reward.DK:d.reward.KD,y=155+row*280;b+=txt(28,y,row?'D → K':'K → D',24);
  seq.forEach((s,i)=>{const x=170+i*240;b+=rect(x,y-30,187,154);if(s.action==='K')b+=key(x+66,y+26);else b+=door(x+70,y+5,s.reward===1);b+=txt(x+94,y+94,`奖励 ${s.reward}`,22,s.reward?C.teal:C.muted,'middle');if(!i)b+=arrow(x+192,y+35,x+230,y+35);});
  b+=txt(170,y+158,'记忆',18,C.muted);
  const states=[0,...seq.map(x=>x.after)];states.forEach((s,i)=>{b+=dot(271+i*122,y+152,[C.muted,C.blue,C.teal][s],13)+txt(271+i*122,y+194,['未拿钥匙','已有钥匙','完成'][s],18,C.ink,'middle');if(i<2)b+=arrow(289+i*122,y+152,368+i*122,y+152);});
 }
 b+=txt(28,736,'K、D 各出现一次：只数动作，两个顺序得到同样总分。',19)+txt(28,775,'记住“已经拿到钥匙”，才能让开门奖励取决于顺序。',19,C.blue);
 out['concept-research-reward-memory.svg']=svg('奖励为什么也需要状态','两段动作序列KD和DK具有相同动作计数。保存奖励自动机状态后，只有先拿钥匙再开门得到一。当前物理位置不变，记忆状态改变奖励含义。',816,b);
 b=txt(28,80,'一次完整生命期是一份样本；同一行的 A、B 成对。',20,C.muted);
 const cols=[C.blue,C.orange,C.teal];
 d.experiments.scoresA.forEach((a,i)=>{const y=155+i*144;b+=txt(30,y+25,`世界 ${i+1}`,19)+rect(129,y-18,230,44,'#edf5fa')+rect(129,y+36,230,44,'#fff1e6');for(let j=0;j<8;j++)b+=dot(145+j*28,y+4,cols[i],4)+dot(145+j*28,y+58,cols[i],4);b+=txt(384,y+9,`A: ${a}`,21,C.blue)+txt(384,y+63,`B: ${d.experiments.scoresB[i]}`,21,C.orange)+arrow(476,y+31,535,y+31)+txt(552,y+39,`${d.experiments.differences[i]>0?'+':''}${num(d.experiments.differences[i])}`,25,cols[i]);});
 b+=txt(28,624,'重采样完整的配对索引',22)+txt(28,662,'例：[3, 3, 1]',21,C.muted);
 d.experiments.resample.forEach((i,j)=>{b+=rect(281+j*128,602,104,81,'#f1f5f7')+dot(305+j*128,625,cols[i],8)+txt(334+j*128,668,num(d.experiments.differences[i]),22,cols[i],'middle');});
 b+=txt(28,736,`原均值差 = ${num(d.experiments.mean)}`,23)+txt(28,778,`本次重采样均值 = ${num(d.experiments.resampled)}`,23,C.teal)+txt(28,826,'不是把一条曲线的时间点当成独立运行。',20,C.muted);
 out['concept-research-experiment-pairing.svg']=svg('实验重复与 bootstrap：抽取的是什么？','三对独立生命期的两方法分数为零点七零点五零点六与零点六零点六零点四。配对差为零点一负零点一零点二。按完整配对索引三三一重采样，均值变为六分之一。',865,b);
 b=txt(28,80,'初始 Q=(1, 0.5)，现在的真实奖励是 (0, 1)。',20,C.muted);
 d.learners.forEach((l,i)=>{const y=162+i*193;b+=txt(28,y,`α=${l.alpha}`,23);l.steps.forEach((s,j)=>{const x=150+j*48;b+=rect(x,y-27,40,53,s.a?'#e5f2ee':'#fff1e6')+txt(x+20,y+8,s.a,23,s.a?C.teal:C.orange,'middle')+dot(x+20,y+59,s.r?C.teal:'#d6dfe5',s.r?11:5);});b+=txt(151,y+114,`十步奖励和：${l.total}`,21);});
 b+=txt(150,119,'动作序列',19,C.muted)+txt(28,756,'第一步都选动作 0。差别在于收到 0 后怎样更新。',20)+txt(28,797,'仅比较当前冻结策略，会漏掉未来学习的差异。',20,C.blue);
 out['concept-research-control-learning.svg']=svg('当前动作一样，未来表现为何不同？','一状态确定性bandit中的精确贪心更新。相同初始价值，不同学习率零、零点一、一导致十步奖励零、三、九。并列时选择动作零。每格是实际执行的一步。',835,b);
 b=txt(28,80,'给定三个动作价值 Q=(0,1,2)，温度改变最优混合。',19,C.muted);
 d.policies.forEach((r,i)=>{const y=180+i*216;b+=txt(28,y-26,`τ=${r.tau}`,23);r.p.forEach((p,j)=>{const x=176+j*146;b+=rect(x,y+80-p*135,65,p*135,[C.blue,C.orange,C.teal][j])+txt(x+32,y+113,`a${j}`,19,C.muted,'middle')+txt(x+32,y+61-p*135,p<.001?p.toFixed(6):num(p),p<.001?17:20,C.ink,'middle');});b+=line(140,y+80,636,y+80)+txt(28,y+162,`熵 H = ${num(r.entropy)}`,19,C.muted);});
 b+=txt(28,845,'低温接近贪心；高温更分散，但这不是信息增益。',20,C.blue);
 out['concept-research-soft-policy.svg']=svg('最大熵控制如何改变行动分布？','精确离散软策略，概率正比于exp(Q除以温度)。三组柱形图使用相同Q，仅温度为零点二、一、三，展示策略熵与价值偏好的权衡。',881,b);
 out['concept-research-mechanisms-data.json']=JSON.stringify(d,null,2)+'\n';return out;
}
