/** Original teaching calculations, not fitted biological data.
 * Run: node public/crl-code/figures/mind-decisions.mjs
 * No random sampling or external package is used. Figures and JSON are regenerated.
 */
import {writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {resolve} from 'node:path';

export function minimumCorrection(x1,x2,target=1){
 if(![x1,x2,target].every(Number.isFinite))throw Error('Finite states and target required');
 const error=target-x1-x2;
 const u=[error/2,error/2];
 return {x:[x1,x2],u,next:[x1+u[0],x2+u[1]],energy:u[0]**2+u[1]**2};
}
export function informationValue(q,cost=1){
 if(!Number.isFinite(q)||q<.5||q>1||!Number.isFinite(cost)||cost<0)throw Error('q in [0.5,1], nonnegative finite cost required');
 // Symmetric binary likelihoods and equal priors give posterior P(high|signal)=q or 1-q.
 const after=.5*Math.max(6,10*q)+.5*Math.max(6,10*(1-q));
 return {q,before:6,after,gross:after-6,net:after-6-cost,cost};
}
const escape=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const text=(x,y,s,size=18,fill='#243f53')=>`<text x="${x}" y="${y}" font-size="${size}" fill="${fill}">${escape(s)}</text>`;
const line=(x1,y1,x2,y2,color='#6d8294',extra='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="2" ${extra}/>`;
const frame=(title,desc,body)=>`<svg xmlns="http://www.w3.org/2000/svg" width="920" height="520" viewBox="0 0 920 520" role="img" aria-labelledby="title desc"><title id="title">${escape(title)}</title><desc id="desc">${escape(desc)}</desc><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#bd643f"/></marker></defs><rect width="920" height="520" fill="#fbfcfd"/><g font-family="Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif">${text(28,38,title,23)}${body}</g></svg>\n`;
export function figures(){
 const states=[[.2,.2],[.8,.6],[.9,.1]].map(x=>minimumCorrection(...x));
 const values=Array.from({length:101},(_,i)=>informationValue(.5+i/200));
 let b=text(30,69,'一步、无噪声、无动作约束；目标只要求 x₁ + x₂ = 1',17);
 const px=x=>65+330*x,py=y=>435-330*y;
 for(const t of [0,.2,.4,.6,.8,1]){
  b+=line(px(t),py(0),px(t),py(1),'#e1e7eb')+line(px(0),py(t),px(1),py(t),'#e1e7eb');
  b+=text(px(t)-8,py(0)+23,t.toFixed(1),14)+text(px(0)-35,py(t)+5,t.toFixed(1),14);
 }
 b+=line(px(0),py(1),px(1),py(0),'#23817a');
 b+=text(415,444,'x₁',18)+text(40,100,'x₂',18);
 for(const s of states){
  const [x,y]=s.x,[nx,ny]=s.next;
  if(s.energy>1e-12)b+=line(px(x),py(y),px(nx),py(ny),'#bd643f','marker-end="url(#arrow)"');
  b+=`<circle cx="${px(x)}" cy="${py(y)}" r="6" fill="#bd643f"/>`;
  b+=text(px(x)+9,py(y)-9,`(${x}, ${y})`,15);
 }
 b+=text(490,122,'同一任务可以有许多成功状态',21);
 b+=text(490,165,'(0.2, 0.2) → (0.5, 0.5)');
 b+=text(490,196,'动作 = (0.3, 0.3)，代价 = 0.18');
 b+=text(490,249,'(0.8, 0.6) → (0.6, 0.4)');
 b+=text(490,280,'动作 = (−0.2, −0.2)，代价 = 0.08');
 b+=text(490,333,'(0.9, 0.1) 已在目标线上');
 b+=text(490,364,'无需纠正；动作和代价都为零');
 b+=text(490,418,'沿绿色线的差异，在此任务中冗余',17,'#23817a');
 b+=text(28,493,'解析教学例；不是生物运动数据。改变目标或约束后，冗余方向也会改变。',16);
 const control=frame('最小干预：纠正任务误差，不追踪唯一姿态','目标线与三个精确计算的反馈动作。',b);
 b=text(28,71,'A 确定得 6；B 以相同概率得 10 或 0；一次信息操作成本为 1',17);
 b+=text(32,124,'立即选择',20)+text(32,157,'A：6 > E[B] = 5');
 b+=text(32,224,'消息完全可靠（q = 1）',20);
 b+=line(65,255,65,342)+line(65,268,125,268)+line(65,342,125,342);
 b+=text(135,276,'高，概率 ½ → 选 B，得 10');
 b+=text(135,350,'低，概率 ½ → 选 A，得 6');
 b+=text(32,410,'净收益 = ½×10 + ½×6 − 1 = 7',18);
 b+=text(32,443,'相对立即选择的净增益：1',18,'#23817a');
 const qx=q=>530+(q-.5)*660,qy=v=>407-(v+1)*106;
 for(const q of [.5,.6,.7,.8,.9,1])b+=line(qx(q),qy(-1),qx(q),qy(2),'#e1e7eb')+text(qx(q)-12,435,q.toFixed(1),14);
 for(const y of [-1,0,1,2])b+=line(qx(.5),qy(y),qx(1),qy(y),y===0?'#798e9e':'#e1e7eb')+text(500,qy(y)+5,y,15);
 const points=key=>values.map(v=>qx(v.q)+','+qy(v[key])).join(' ');
 b+=`<polyline points="${points('gross')}" fill="none" stroke="#23817a" stroke-width="3"/><polyline points="${points('net')}" fill="none" stroke="#bd643f" stroke-width="3"/>`;
 b+=text(525,102,'相对立即行动的价值',18)+text(650,468,'消息正确率 q',17);
 b+=text(548,137,'绿色：毛增益',16,'#23817a')+text(548,165,'橙色：扣除成本后的净增益',16,'#bd643f');
 b+=text(28,499,'精确计算；消息误报对称，先验各半。内部推演不自动等于获得这样的外部消息。',16);
 return {'mind-control-task-directions.svg':control,'mind-economics-value-of-information.svg':frame('信息何时值得获取？先看它能否改变行动','决策树和正确率从0.5到1时的信息毛收益与净收益。',b),'mind-decisions-data.json':JSON.stringify({scope:'Original deterministic teaching calculations; no human data; no random seed',control:{target:1,states},information:{A:6,B:[0,10],prior:[.5,.5],cost:1,values}},null,2)+'\n'};
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 for(const [name,content] of Object.entries(figures())){
  const destination=new URL('../../crl-figures/'+name,import.meta.url);
  writeFileSync(destination,content);
  process.stdout.write(fileURLToPath(destination)+'\n');
 }
}
