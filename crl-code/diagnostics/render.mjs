/** MIT. Render the textbook's numerical diagnostics without external packages.
 * Usage: node render.mjs results.json
 * Writes one SVG per chart beside the input JSON. These are numerical checks,
 * not author benchmark reproductions. Run the corresponding Python lab first.
 */
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const escape=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
const widthOf=text=>[...String(text)].reduce((n,c)=>n+(c.charCodeAt(0)>255?18:9.5),0);
const number=value=>value===0?'0':(Math.abs(value)>=1e5||Math.abs(value)<.001)?value.toExponential(1):Number(value.toPrecision(4)).toString();
const palette=['#285d82','#ad5733','#69803f','#805c85','#b08728'];
const dashes=['','9 5','3 4','12 4 3 4','6 3'];

function tickStep(span,target=5){
 const raw=span/target,power=10**Math.floor(Math.log10(raw));
 return ([1,2,2.5,5,10].find(m=>m*power>=raw)||10)*power;
}
function linearTicks(lo,hi,target=5){
 const step=tickStep(hi-lo,target),first=Math.ceil(lo/step-1e-10)*step;
 return Array.from({length:Math.floor((hi-first)/step+1e-9)+1},(_,i)=>{
  const value=first+i*step;return Math.abs(value)<step*1e-10?0:value;
 });
}

function extent(values,log=false){
 let lo=Math.min(...values),hi=Math.max(...values);
 if(log){
  if(lo<=0)throw Error('Logarithmic axis requires strictly positive values; do not silently hide zero.');
  lo=Math.floor(Math.log10(lo));hi=Math.ceil(Math.log10(hi));
  if(lo===hi){lo-=1;hi+=1;}
  return [lo,hi];
 }
 if(lo>=0)lo=0;
 if(hi<=0)hi=0;
 if(lo===hi)return [lo-1,hi+1];
 const step=tickStep(hi-lo);
 return [Math.floor(lo/step)*step,Math.ceil(hi/step)*step];
}

export function renderChart(chart,scope=''){
 if(!/^[a-z0-9][a-z0-9-]*$/.test(chart.id))throw Error('Invalid chart id');
 if(!chart.title||!chart.xLabel||!chart.yLabel)throw Error('A chart needs title and axis labels');
 if(chart.xScale&&chart.xScale!=='linear')throw Error('Only linear x axes are supported');
 if(chart.yScale&&!['linear','log'].includes(chart.yScale))throw Error('Unknown y scale');
 if(!Array.isArray(chart.series)||!chart.series.length||chart.series.length>8)throw Error('Expected 1–8 series');
 for(const series of chart.series){
  if(!series.name||!Array.isArray(series.points)||!series.points.length)throw Error('Empty series');
  for(let i=0;i<series.points.length;i++){
   const {x,y}=series.points[i];
   if(!Number.isFinite(x)||!Number.isFinite(y))throw Error('Non-finite point');
   const {low,high}=series.points[i];
   if(low!==undefined||high!==undefined){
    if(!Number.isFinite(low)||!Number.isFinite(high)||low>y||high<y)throw Error('Invalid estimate bounds');
    if(!chart.intervalLabel)throw Error('An uncertainty interval needs its meaning in intervalLabel');
   }
   if(i&&x<series.points[i-1].x)throw Error('Points must be ordered by numeric x');
  }
 }
 const W=900,left=96,right=36,plotH=320;
 // Each long legend gets its own row. Short entries wrap instead of shrinking.
 let legendX=left,legendY=104;
 const legends=chart.series.map((series,i)=>{
  const itemW=64+widthOf(series.name);
  if(itemW>W-left-right)throw Error('Legend label is too long; shorten it and explain it in the caption');
  if(legendX>left&&legendX+itemW>W-right){legendX=left;legendY+=30;}
  const item={series,i,x:legendX,y:legendY};legendX+=itemW+24;
  return item;
 });
 const top=legendY+(chart.intervalLabel?58:34),bottom=top+plotH,H=bottom+90;
 const points=chart.series.flatMap(s=>s.points);
 const xs=points.map(p=>p.x),ys=points.flatMap(p=>p.low===undefined?[p.y]:[p.low,p.y,p.high]),log=chart.yScale==='log';
 let xlo=Math.min(...xs),xhi=Math.max(...xs);
 if(xlo===xhi){xlo-=.5;xhi+=.5;}
 const [ylo,yhi]=extent(ys,log);
 const sx=x=>left+(x-xlo)/(xhi-xlo)*(W-left-right);
 const sy=y=>bottom-((log?Math.log10(y):y)-ylo)/(yhi-ylo)*plotH;
 const color=i=>chart.series[i].role==='reference'?'#4d5660':palette[i%palette.length];
 const dash=i=>chart.series[i].role==='reference'?['7 5','2 4','10 4 2 4'][chart.series.slice(0,i).filter(s=>s.role==='reference').length%3]:dashes[i%dashes.length];
 const out=[`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="title description">`,
  `<title id="title">${escape(chart.title)}</title><desc id="description">${escape(scope)} ${escape(chart.xLabel)}；${escape(chart.yLabel)}。原始数值见同目录 results.json。</desc>`,
  '<rect width="100%" height="100%" fill="#fff"/>',
  '<g font-family="system-ui, -apple-system, Noto Sans CJK SC, sans-serif" fill="#233748">',
  `<text x="${left}" y="36" font-size="23" font-weight="650">${escape(chart.title)}</text>`,
  `<text x="${left}" y="68" font-size="16">${escape(chart.yLabel)}${log&&!/对数|log/i.test(chart.yLabel)?'（对数坐标）':''}</text>`];
 if(widthOf(chart.title)*23/18>W-left-right)throw Error('Chart title is too long');
 for(const {series,i,x,y} of legends){
  out.push(`<path d="M${x} ${y-6}h32" fill="none" stroke="${color(i)}" stroke-width="2.6" stroke-dasharray="${dash(i)}"/><text x="${x+43}" y="${y}" font-size="18">${escape(series.name)}</text>`);
 }
 if(chart.intervalLabel){
  if(widthOf(chart.intervalLabel)*16/18>W-left-right)throw Error('Interval label is too long');
  out.push(`<text x="${left}" y="${legendY+27}" font-size="16" fill="#586574">${escape(chart.intervalLabel)}</text>`);
 }
 // Common scales across all series; ticks use the same transform as the marks.
 const logStep=Math.max(1,Math.ceil((yhi-ylo)/6));
 const yTicks=log?Array.from({length:Math.floor((yhi-ylo)/logStep)+1},(_,i)=>ylo+i*logStep):linearTicks(ylo,yhi,6);
 if(log&&yTicks.at(-1)!==yhi)yTicks.push(yhi);
 for(const tick of yTicks){
  const value=log?10**tick:tick,y=sy(value);
  out.push(`<path d="M${left} ${y}H${W-right}" stroke="${!log&&Math.abs(value)<1e-12?'#a4b2c0':'#e2e7eb'}" fill="none"/><text x="${left-12}" y="${y+6}" text-anchor="end" font-size="17">${escape(number(value))}</text>`);
 }
 const uniqueX=[...new Set(xs)].sort((a,b)=>a-b);
 let xTicks=uniqueX.length<=7?uniqueX:linearTicks(xlo,xhi,6);
 // Include both reviewed endpoints, while avoiding two nearly coincident labels.
 if(xTicks[0]!==xlo){if((xTicks[0]-xlo)/(xhi-xlo)<.07)xTicks.shift();xTicks.unshift(xlo);}
 if(xTicks.at(-1)!==xhi){if((xhi-xTicks.at(-1))/(xhi-xlo)<.07)xTicks.pop();xTicks.push(xhi);}
 for(const tick of xTicks){
  const x=sx(tick);
  out.push(`<path d="M${x} ${bottom}v6" stroke="#677789"/><text x="${x}" y="${bottom+30}" text-anchor="middle" font-size="17">${escape(number(tick))}</text>`);
 }
 out.push(`<path d="M${left} ${top}V${bottom}H${W-right}" fill="none" stroke="#677789" stroke-width="1.2"/>`);
 for(let i=0;i<chart.series.length;i++){
  const bounded=chart.series[i].points.filter(p=>p.low!==undefined);
  if(bounded.length&&bounded.length!==chart.series[i].points.length)throw Error('Bounds must cover the full series');
  if(bounded.length){
   const coords=[...bounded.map(p=>`${sx(p.x)},${sy(p.high)}`),...[...bounded].reverse().map(p=>`${sx(p.x)},${sy(p.low)}`)].join(' ');
   out.push(`<polygon points="${coords}" fill="${color(i)}" fill-opacity="0.10" stroke="none"/>`);
  }
 }
 for(let i=0;i<chart.series.length;i++){
  const series=chart.series[i],d=series.points.map((p,j)=>`${j?'L':'M'}${sx(p.x).toFixed(3)} ${sy(p.y).toFixed(3)}`).join(' ');
  out.push(`<path d="${d}" fill="none" stroke="${color(i)}" stroke-width="2.6" stroke-linejoin="round" stroke-dasharray="${dash(i)}"/>`);
  // Marks locate actual observations on sparse curves; no interpolated samples.
  if(series.points.length<=35)for(const p of series.points){
   const x=sx(p.x),y=sy(p.y);
   if(i%2)out.push(`<rect x="${x-3.5}" y="${y-3.5}" width="7" height="7" fill="#fff" stroke="${color(i)}" stroke-width="1.7"/>`);
   else out.push(`<circle cx="${x}" cy="${y}" r="3.5" fill="${color(i)}"/>`);
  }
 }
 out.push(`<text x="${left+(W-left-right)/2}" y="${bottom+65}" text-anchor="middle" font-size="18">${escape(chart.xLabel)}</text></g></svg>`);
 return out.join('\n')+'\n';
}

export function renderResults(input){
 const result=JSON.parse(fs.readFileSync(input,'utf8'));
 if(!result.id||!result.protocol||!result.checks||!result.scope)throw Error('Missing experiment provenance or checks');
 if(!Array.isArray(result.charts)||!result.charts.length)throw Error('No charts to render');
 const seen=new Set();
 const rendered=result.charts.map(chart=>{
  if(seen.has(chart.id))throw Error('Duplicate chart id');seen.add(chart.id);
  return {id:chart.id,svg:renderChart(chart,result.scope)};
 });
 // Validate every figure before writing any output.
 return rendered.map(({id,svg})=>{
  const target=path.join(path.dirname(input),id+'.svg');
  fs.writeFileSync(target,svg,'utf8');return target;
 });
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 if(process.argv.length!==3){console.error('Usage: node render.mjs results.json');process.exitCode=2;}
 else for(const target of renderResults(process.argv[2]))console.log(target);
}
