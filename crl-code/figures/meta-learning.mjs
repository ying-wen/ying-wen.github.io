/** Small, exact calculations for the meta-learning chapter.
 * node public/crl-code/figures/meta-learning.mjs
 * These are teaching examples, not DiscoRL or MGSC performance experiments.
 */
import {writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

export function oneStep(alpha,evaluationTarget,{w=0,trainingTarget=1}={}){
 if(![alpha,evaluationTarget,w,trainingTarget].every(Number.isFinite))throw Error('Finite inputs required');
 const sensitivity=trainingTarget-w,updated=w+alpha*sensitivity;
 return {alpha,evaluationTarget,updated,loss:.5*(updated-evaluationTarget)**2,
  sensitivity,gradient:(updated-evaluationTarget)*sensitivity,
  logStepGradient:alpha*(updated-evaluationTarget)*sensitivity};
}
export const sigmoid=x=>1/(1+Math.exp(-x));
// Target is held fixed when differentiating this local MGSC-style surrogate.
export function planningAllocation(logit,target=[.3,.1],alpha=.2){
 const p=sigmoid(logit),w=[alpha*p,alpha*(1-p)];
 const loss=w.reduce((sum,v,i)=>sum+(v-target[i])**2,0);
 const gradient=2*alpha*p*(1-p)*((w[0]-target[0])-(w[1]-target[1]));
 return {p,w,loss,gradient};
}
export const data={kind:'exact-teaching-examples',rows:[oneStep(.2,1),oneStep(.2,0)],
 comparison:[oneStep(.4,1),oneStep(.4,0)],planning:planningAllocation(0)};
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const t=(x,y,s,size=21,color='#233c55',anchor='start')=>`<text x="${x}" y="${y}" text-anchor="${anchor}" font-size="${size}" fill="${color}">${esc(s)}</text>`;
const line=(x1,y1,x2,y2,color='#bbcbd5',width=2)=>`<path d="M${x1} ${y1}L${x2} ${y2}" fill="none" stroke="${color}" stroke-width="${width}"/>`;
const dot=(x,y,r=7,c='#237ba1')=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${c}"/>`;
const arr=(x1,y1,x2,y2,color='#54738a')=>`<path d="M${x1} ${y1}L${x2} ${y2}" fill="none" stroke="${color}" stroke-width="3" marker-end="url(#arrow)"/>`;
const curve=(d,color='#54738a',width=3,arrow=false)=>`<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}"${arrow?' marker-end="url(#arrow)"':''}/>`;
const wrap=(title,height,desc,body)=>`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 ${height}" role="img" aria-labelledby="title desc"><title id="title">${esc(title)}</title><desc id="desc">${esc(desc)}</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10" fill="context-stroke"/></marker></defs><rect width="920" height="${height}" fill="#fff"/><g font-family="system-ui,sans-serif">${t(32,43,title,27)}${body}</g></svg>`;
// Small pictures encode the objects: positions in a maze, transitions on a tape,
// network weights and a target. Long explanations remain in the chapter.
const maze=(x,y,size=26,goal=2)=>{
 let out='';for(let j=0;j<3;j++)for(let i=0;i<3;i++)out+=`<rect x="${x+i*size}" y="${y+j*size}" width="${size}" height="${size}" fill="${i===1&&j===1?'#aebfc6':'#f1f6f8'}" stroke="#fff" stroke-width="2"/>`;
 out+=dot(x+size/2,y+size*2.5,size*.19);out+=`<path d="M${x+size*(goal+.25)} ${y+size*.3}h${size*.5}v${size*.4}h-${size*.5}z" fill="#e6ad49"/>`;
 return out+curve(`M${x+size*.5} ${y+size*2.1}V${y+size*.5}H${x+size*(goal+.2)}`,'#4d9eaa',3,true);
};
const net=(x,y,scale=1,color='#237ba1')=>{
 const layers=[[0,28,56],[0,28,56],[14,42]],xs=[0,45,90];let out='';
 for(let l=0;l<2;l++)for(const a of layers[l])for(const b of layers[l+1])out+=line(x+xs[l]*scale,y+a*scale,x+xs[l+1]*scale,y+b*scale,color,1.5);
 for(let l=0;l<3;l++)for(const a of layers[l])out+=dot(x+xs[l]*scale,y+a*scale,7*scale,color);
 return out;
};
const tape=(x,y,n=4,color='#237ba1')=>Array.from({length:n},(_,i)=>`<rect x="${x+i*23}" y="${y}" width="19" height="32" rx="3" fill="${color}" opacity="${.35+.6*i/(n-1)}"/>`+dot(x+i*23+9.5,y+16,3,'#fff')).join('');
const target=(x,y,r=27)=>[1,.65,.3].map((q,i)=>`<circle cx="${x}" cy="${y}" r="${r*q}" fill="${i%2?'white':'#c25e2b'}"/>`).join('');
const lock=(x,y)=>`<path d="M${x+5} ${y+15}v-6a8 8 0 0 1 16 0v6" fill="none" stroke="#596779" stroke-width="4"/><rect x="${x}" y="${y+13}" width="26" height="23" rx="4" fill="#596779"/>`;
export const figures={};

let one=t(40,98,'一次预测更新',22)+t(762,98,'训练目标',18)+
 line(80,158,825,158)+dot(80,158,10,'#98aab5')+dot(229,158,11)+target(825,158,17)+
 arr(100,130,215,130)+t(145,115,'α = 0.2',19,'#237ba1','middle')+
 t(80,197,'w = 0',21,'#233c55','middle')+t(229,197,'w′ = 0.2',21,'#237ba1','middle')+t(825,197,'1',21,'#233c55','middle');
for(const [i,y] of [1,0].entries()){
 const x0=75+i*445,y0=546,width=325,height=230,color=i?'#c25e2b':'#247d67';
 one+=t(x0,263,`后来收到 ${y}`,24,color)+line(x0,y0,x0+width+15,y0)+line(x0,y0,x0,y0-height-15)+t(x0+width+8,y0+35,'α',20)+t(x0-8,y0-height-25,'后续误差 J',18);
 const d=Array.from({length:51},(_,k)=>{const a=k/50;return `${k?'L':'M'}${x0+a*width} ${y0-oneStep(a,y).loss/.5*height}`;}).join('');
 one+=curve(d,color,4);
 for(const a of [0,.2,.4,1])one+=line(x0+a*width,y0,x0+a*width,y0+6)+t(x0+a*width,y0+27,String(a),17,'#536e80','middle');
 const current=oneStep(.2,y),cx=x0+.2*width,cy=y0-current.loss/.5*height;
 one+=dot(cx,cy,8,color)+line(cx,cy,cx,y0,color,1)+t(cx+15,cy-(i?48:12),`J = ${current.loss.toFixed(2)}`,19,color);
 one+=arr(cx,y0+66,cx+(i?-45:80),y0+66,color)+t(x0+180,y0+73,i?'步长减小':'步长增大',21,color);
 one+=t(x0,y0+118,`dJ/dα = ${current.gradient}`,22,color);
}
figures['concept-meta-one-update.svg']=wrap('同一次学习，为什么步长会向相反方向调整？',715,'上方数轴画出预测从0移到0.2。下方是两个后续目标下精确平方误差曲线，当前点的斜率分别为负0.8和正0.2。',one);

// Five visual interventions on the same inner learning loop.
let design=t(460,92,'同一个学习过程',20,'#536e80','middle')+
 maze(75,145,31)+t(120,270,'环境',22,'#233c55','middle')+
 arr(175,190,290,190)+tape(306,174,5)+t(361,270,'经验',22,'#233c55','middle')+
 arr(431,190,540,190)+net(565,160,1.35)+t(625,270,'学习器',22,'#233c55','middle')+
 arr(704,190,818,190)+t(787,166,'动作',20)+curve('M842 207V306H120V255','#8da6b5',2.5,true);
const xs=[102,282,462,642,822],titles=['初值','更新幅度','训练目标','计算分配','内部记忆'],methods=['MAML','IDBD / Metatrace','LPG / DiscoRL','MGSC','RL² / PEARL'];
for(let i=0;i<xs.length;i++){design+=t(xs[i],386,titles[i],22,'#233c55','middle')+t(xs[i],551,methods[i],18,'#536e80','middle');}
design+=curve('M48 440Q102 520 156 440','#90a6b5',3)+dot(68,463,8)+dot(131,470,5,'#b7c6d0')+arr(130,433,80,455,'#237ba1');
design+=line(227,472,337,472)+dot(227,472,6,'#a8b9c5')+arr(234,450,258,450,'#237ba1')+arr(234,496,324,496,'#c25e2b')+dot(265,472,7)+dot(333,472,7,'#c25e2b');
design+=target(462,471,35)+arr(406,428,457,466,'#233c55');
design+=tape(594,450,5)+`<circle cx="619" cy="466" r="26" fill="none" stroke="#c25e2b" stroke-width="4"/>`+line(638,485,657,504,'#c25e2b',5);
design+=net(777,442,1,'#7a68a3')+curve('M858 435C888 395 750 395 781 436','#7a68a3',2.5,true);
design+=curve('M102 343V323H625V293','#a9bbc7',2)+curve('M282 343V327H625V293','#a9bbc7',2)+curve('M462 343V331H625V293','#a9bbc7',2)+curve('M642 343V335H361V290','#a9bbc7',2)+curve('M822 343V319H668V292','#a9bbc7',2);
design+=line(40,597,880,597)+t(460,637,'怎样得到这些设计？',21,'#233c55','middle')+t(460,674,'元梯度 · 黑盒搜索 · 蒸馏 · 程序搜索',21,'#536e80','middle');
figures['concept-meta-design-space.svg']=wrap('元学习可以改动学习过程中的哪些位置？',712,'上方为环境、经验、学习器、动作的循环；下方用起点、步长箭头、靶心、样本选择和循环记忆表示五种设计对象。搜索方法是另一维度。',design);

let discovery=t(35,95,'发现阶段：规则参数 η 也学习',23)+
 maze(45,136,23,2)+maze(45,230,23,0)+maze(45,324,23,2)+
 net(173,147,.85,'#247d67')+net(173,241,.85,'#c25e2b')+net(173,335,.85,'#7a68a3')+
 arr(116,169,153,169)+arr(116,263,153,263)+arr(116,357,153,357)+
 t(203,409,'多个 agent',19,'#233c55','middle')+
 curve('M259 170H306V259H351','#247d67',2.5,true)+arr(259,263,351,263,'#c25e2b')+curve('M259 357H306V269H351','#7a68a3',2.5,true)+
 net(374,224,1.5,'#7a68a3')+t(441,203,'规则 Uη',23,'#7a68a3','middle')+t(437,337,'轨迹、预测',18,'#536e80','middle')+
 arr(527,264,590,264)+target(626,264,31)+t(626,337,'训练目标',19,'#233c55','middle')+
 arr(662,264,713,264)+net(734,229,1.25)+t(791,337,'更新 w',20,'#237ba1','middle')+
 curve('M844 314V435H442V352','#c25e2b',4,true)+
 t(643,466,'后续回报 → 改进 η',22,'#c25e2b','middle')+
 line(35,505,885,505)+t(35,552,'使用阶段：固定 η，新 agent 继续学习',23)+
 maze(54,603,27)+arr(145,640,261,640)+
 net(290,605,1.15,'#7a68a3')+lock(399,617)+t(346,709,'固定规则',20,'#7a68a3','middle')+
 arr(438,640,534,640)+net(565,605,1.2)+t(622,709,'w 持续更新',20,'#237ba1','middle')+
 arr(687,640,751,640)+line(765,678,873,678)+line(765,678,765,597)+curve('M767 671L785 667L802 647L817 657L833 625L850 620L873 603','#247d67',3)+t(814,709,'学习表现',20,'#247d67','middle');
figures['concept-meta-discovery.svg']=wrap('学习一条规则，再让新智能体使用它',750,'三种环境中的智能体产生轨迹，共同训练规则网络；规则生成目标用于更新智能体。后续回报改进规则参数。下半图锁住规则参数，新智能体的权重继续更新。小曲线只表示评价位置。',discovery);

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const dir=new URL('../../crl-figures/',import.meta.url);
 for(const [name,svg] of Object.entries(figures))writeFileSync(new URL(name,dir),svg);
 writeFileSync(new URL('concept-meta-data.json',dir),JSON.stringify(data,null,2)+'\n');
 console.log('Generated three meta-learning figures and exact examples');
}
