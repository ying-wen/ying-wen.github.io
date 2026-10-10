/** Exact one-step expectile example. No neural recording or fitted participant data.
 * Run from the website root: node public/crl-code/figures/mind-distributional.mjs
 * Positive errors use tau; negative errors use 1-tau. Solve expected drift = 0.
 */
import {writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

export function drift(value,tau,rewards){
 if(!(tau>0&&tau<1)||!Number.isFinite(value)||!rewards.length||!rewards.every(Number.isFinite))throw Error('Invalid expectile inputs');
 return rewards.reduce((sum,r)=>sum+(r>=value?tau:1-tau)*(r-value),0)/rewards.length;
}
export function expectile(tau,rewards){
 drift(0,tau,rewards);
 let lo=Math.min(...rewards),hi=Math.max(...rewards);
 for(let i=0;i<70;i++){const mid=(lo+hi)/2;if(drift(mid,tau,rewards)>0)lo=mid;else hi=mid;}
 return (lo+hi)/2;
}
export const data={kind:'exact-one-step-model',rewards:[0,.5,1],comparison:[.5],taus:[.2,.5,.8]};
data.rows=data.taus.map(tau=>({tau,value:expectile(tau,data.rewards),constant:expectile(tau,data.comparison),errorAtHalf:.5-expectile(tau,data.rewards)}));
const text=(x,y,t,size=21,fill='#24415c')=>`<text x="${x}" y="${y}" font-size="${size}" fill="${fill}">${t}</text>`;
const lines=[];
for(const [k,title,rewards] of [[0,'每次都是 0.5',[.5]],[1,'0、0.5、1 等概率',[0,.5,1]]]){
 const x=40+k*440;lines.push(text(x,104,title,24));
 lines.push(`<path d="M${x+20} 225H${x+340}" stroke="#8094a8"/>`);
 for(const r of rewards){const h=90/rewards.length;lines.push(`<rect x="${x+15+r*300}" y="${225-h}" width="26" height="${h}" rx="2" fill="${k?'#237ba1':'#357e69'}"/>`);}
 for(const r of [0,.5,1])lines.push(text(x+16+r*300,253,String(r),18));
 lines.push(text(x,287,'平均奖励 = 0.5',21));
}
for(const [i,row] of data.rows.entries()){
 const y=390+i*65;lines.push(text(40,y,`τ = ${row.tau}`));
 lines.push(text(237,y,row.constant.toFixed(2),24));
 lines.push(`<path d="M480 ${y-8}H820" stroke="#c6d4df"/>`);
 const cx=480+340*row.value;lines.push(`<circle cx="${cx}" cy="${y-8}" r="7" fill="#237ba1"/>`);
 lines.push(text(cx+14,y,row.value.toFixed(2),23));
}
export const svg=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 715" role="img" aria-labelledby="title desc"><title id="title">同样的平均奖励，不同的结果分布</title><desc id="desc">原创一步预测算例：确定奖励0.5和三点等概率奖励的均值相同，但不同expectile坐标有所不同。</desc><rect width="920" height="715" fill="white"/><g font-family="system-ui, sans-serif">${text(40,43,'一个均值遗漏了哪些结果差别？',28)}${lines.join('')}${text(40,337,'正误差的权重',18)}${text(202,337,'确定奖励的预测',18)}${text(480,337,'三点奖励的预测（expectile）',18)}<rect x="35" y="551" width="850" height="129" rx="9" fill="#edf4f8"/>${text(55,582,'同样收到 r = 0.5，三个通道的误差为 +0.25、0、−0.25。',22)}${text(55,619,'它们以不同的正／负误差权重学习，不同坐标一起描述结果分布。',20)}${text(55,653,'独立推导的精确算例；均匀三点分布；不是神经数据或分位数。',18)}</g></svg>`;
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const directory=new URL('../../crl-figures/',import.meta.url);
 writeFileSync(new URL('mind-distributional-expectiles.svg',directory),svg);
 writeFileSync(new URL('mind-distributional-data.json',directory),JSON.stringify(data,null,2)+'\n');
 console.log('Generated exact expectile figure and data');
}
