/** Pure SVG rendering of supplied exact numbers. No training or hidden solver. */
import {palette as C,text,line,arrow,rect,circle,frame} from './visual-kit.mjs';
const t=(x,y,s,size=20,color=C.ink,anchor='start')=>text(x,y,s,{size,color,anchor});
const n=v=>Number(v.toFixed(3)).toString();
function plane(x,y,w,h,xd,yd,xlabel,ylabel){
 const X=v=>x+(v-xd[0])/(xd[1]-xd[0])*w,Y=v=>y+h-(v-yd[0])/(yd[1]-yd[0])*h;
 let body=line(X(xd[0]),Y(0),X(xd[1]),Y(0))+line(X(0),Y(yd[0]),X(0),Y(yd[1]));
 body+=t(X(xd[1]),Y(0)+48,xlabel,18,C.muted,'end')+t(X(0),y-16,ylabel,18,C.muted);
 for(const q of [0,1])if(q>=xd[0]&&q<=xd[1])body+=line(X(q),Y(0)-4,X(q),Y(0)+4)+t(X(q),Y(0)+24,q,16,C.muted,'middle');
 for(const q of [-1,1])if(q>=yd[0]&&q<=yd[1])body+=line(X(0)-4,Y(q),X(0)+4,Y(q))+t(X(0)-9,Y(q)+5,q,16,C.muted,'end');
 return {body,X,Y,vec:(a,b,color,dash='')=>arrow(X(a[0]),Y(a[1]),X(b[0]),Y(b[1]),{color,dash,width:3,head:9}),
  point:(a,color)=>circle(X(a[0]),Y(a[1]),5,{fill:color,stroke:C.white})};
}
function corridor(width,y){
 const xs=width===360?[48,176,305]:[118,360,602];
 let b='';
 ['A','B','⊥'].forEach((name,i)=>{
  if(i<2)b+=arrow(xs[i]+23,y,xs[i+1]-23,y,{color:C.blue,head:9})+t((xs[i]+xs[i+1])/2,y-17,i? '−1':'+2',18,C.ink,'middle');
  b+=circle(xs[i],y,22,{fill:C.white,stroke:C.blue})+t(xs[i],y+7,name,20,C.ink,'middle');
  if(i===2)b+=circle(xs[i],y,27,{fill:'none',stroke:C.blue,strokeWidth:1});
 });
 return b;
}
function geometry(d,mobile){
 const width=mobile?360:720;let b=corridor(width,106);
 b+=t(24,159,'γ=1；完整回报 G(A)=1，G(B)=−1',mobile?16:19,C.muted);
 const plots=[{x:60,y:253},{x:mobile?74:414,y:mobile?615:253}];
 let p=plane(plots[0].x,plots[0].y,230,230,[0,1.3],[0,1.3],'x₁','x₂');
 b+=t(24,205,'共享方向：两个单位特征',20);
 b+=p.body+p.vec([0,0],d.provenance.features[0],C.blue)+p.vec([0,0],d.provenance.features[1],C.teal);
 b+=line(p.X(.8),p.Y(.6),p.X(.8),p.Y(0),{dash:'5 5'});
 b+=t(p.X(1)-4,p.Y(0)-17,'x(A)',18,C.blue,'end');
 b+=t(p.X(.8)+10,p.Y(.6)-12,'x(B)',18,C.teal);
 b+=t(p.X(.8),p.Y(0)+49,'内积 0.8',18,C.teal,'middle');
 p=plane(plots[1].x,plots[1].y,215,275,[-.25,1.25],[-1.3,.62],'v̂(A)','v̂(B)');
 b+=t(mobile?24:385,mobile?567:205,'预测同时移动；方框是真值',20);
 b+=p.body;
 const vals=[[0,0],...d.mc.map(r=>r.predictions)];
 for(let i=1;i<vals.length;i++)b+=p.vec(vals[i-1],vals[i],i===1?C.blue:C.orange);
 vals.forEach((v,i)=>{b+=p.point(v,i===2?C.orange:C.blue)+t(p.X(v[0])+(i===1?10:-11),p.Y(v[1])+(i===2?28:-11),String(i),18,i===2?C.orange:C.blue);});
 b+=rect(p.X(1)-6,p.Y(-1)-6,12,12,{fill:C.white,stroke:C.teal,radius:0,strokeWidth:2});
 b+=t(p.X(1)-7,p.Y(-1)-16,'(1,−1)',18,C.teal,'end');
 const y=mobile?940:587;
 b+=t(24,y,'按 A、B 顺序回归；α=0.5，w₀=0',mobile?17:20);
 b+=t(24,y+32,'均匀价值误差：0.5 → 0.5525 → 0.4034',mobile?16:19,C.muted);
 return frame({title:'一次更新怎样牵动两个预测？',description:'原创精确计算。上方为两步走廊，奖励加2和减1，折扣1。左图特征有相同单位长度，内积0.8由虚线投影显示。右图以A、B预测为两条坐标轴，圆点0、1、2表示从零开始在完整回报可用后按A、B顺序进行MC回归的预测，方框是真值(1,−1)。第一步让A更准却使B更差，均匀半平方误差从0.5增加到0.5525。坐标与箭头均由脚本数值生成；不是学习曲线。',width,height:y+57,body:b,kind:'exact',heading:{x:24,y:40,size:mobile?20:26}});
}
function memory(d,mobile){
 const width=mobile?360:720;let b=corridor(width,105);
 b+=t(24,160,'到达终点时：先衰减 A，再加入 B',mobile?17:21);
 const acc=d.accumulating[1], dutch=d.true_online[1];
 for(let i=0;i<2;i++){
  const x=mobile?61:61+i*347,y=mobile?242+i*390:242,row=i?dutch:acc;
  const p=plane(x,y,235,210,[0,1.5],[0,1.34],'迹分量 1','迹分量 2');
  b+=t(mobile?24:24+i*347,y-43,i?'Dutch：新方向系数 0.8':'累积：新方向系数 1',20,i?C.purple:C.blue)+p.body;
  b+=p.vec([0,0],row.faded,C.muted)+p.vec(row.faded,row.trace,C.teal);
  b+=p.vec([0,0],row.trace,i?C.purple:C.blue,'6 5');
  b+=t(p.X(.25),p.Y(0)+48,'0.5 x(A)',18,C.muted,'middle');
  b+=t(p.X(row.trace[0])-8,p.Y(row.trace[1])-18,'('+row.trace.map(n).join(', ')+')',18,i?C.purple:C.blue,'end');
  b+=t(mobile?24:24+i*347,y+288,i?'0.5 x(A) + 0.8 x(B)':'0.5 x(A) + x(B)',18,C.teal);
 }
 const y=mobile?982:592;
 b+=t(24,y,'实线首尾相接；虚线是合成梯度记忆',mobile?17:20);
 b+=t(24,y+32,'两条迹都保存参数方向，而非两份状态值',mobile?16:19,C.muted);
 return frame({title:'资格迹把过去的方向留到现在',description:'原创精确向量图。A到B到终点，奖励2、−1，gamma1、lambda0.5、alpha0.5。累积迹将旧A特征衰减为(0.5,0)并加上B特征(0.8,0.6)，得到(1.3,0.6)。Dutch迹将新B特征乘以1−0.5×0.5×0.8=0.8，得到(1.14,0.48)。两图坐标尺度一致。灰色实线为衰减历史，青色实线为新特征贡献，蓝或紫虚线是向量和，不表示模型生成数据。',width,height:y+58,body:b,kind:'exact',heading:{x:24,y:40,size:mobile?19:26}});
}
function online(d,mobile){
 const width=mobile?360:720,p=plane(mobile?68:106,140,mobile?245:500,mobile?335:310,[-.3,1.2],[-.75,.15],'参数 1','参数 2');
 let b=t(24,84,'首步已经更新：w₁=(1,0)，v̂(B)=0.8',mobile?16:20)+p.body;
 const start=d.true_online[0].after,acc=d.accumulating[1].after,tr=d.true_online[1];
 const mid=start.map((v,i)=>v+tr.td_increment[i]);
 b+=p.vec([0,0],start,C.blue)+p.vec(start,acc,C.red)+p.vec(start,mid,C.purple)+p.vec(mid,tr.after,C.orange);
 b+=p.point(start,C.blue)+t(p.X(start[0])-2,p.Y(start[1])-18,'第一步',18,C.blue,'end');
 b+=p.point(acc,C.red)+t(p.X(acc[0])+7,p.Y(acc[1])+28,'累积迹',18,C.red);
 b+=p.point(mid,C.purple)+t(p.X(mid[0])-10,p.Y(mid[1])-20,'仅 Dutch',18,C.purple,'end');
 b+=p.point(tr.after,C.orange)+t(p.X(tr.after[0])+12,p.Y(tr.after[1])-13,'完整修正',18,C.orange);
 const y=mobile?561:531;
 b+=t(24,y,'累积： (−0.17, −0.54)',mobile?18:20,C.red);
 b+=t(24,y+34,'仅 Dutch： (−0.026, −0.432)',mobile?17:20,C.purple);
 b+=t(24,y+68,'完整修正： (0.11, −0.48)',mobile?18:20,C.orange);
 b+=t(24,y+112,'在线前向重算也到达 (0.11, −0.48)',mobile?16:20,C.muted);
 return frame({title:'在线参数变化还需要一次补偿',description:'原创精确参数路径图，固定走廊与同一首步权重(1,0)。第二步TD误差为−1.8，保存的旧B预测是0而当前B预测是0.8。红箭头使用普通累积迹到(−0.17,−0.54)。紫箭头只使用Dutch迹的TD增量到(−0.026,−0.432)，橙箭头再加预测变化修正(0.136,−0.048)到(0.11,−0.48)。这个终点与逐前缀在线前向定义一致。箭头表示参数更新，不表示环境动作或性能排名。',width,height:y+143,body:b,kind:'exact',heading:{x:24,y:40,size:mobile?19:26}});
}
export function sharedGradientFigures(data){
 const assets={};
 for(const mobile of [false,true])for(const [name,render] of Object.entries({geometry,memory,online})){
  assets['concept-shared-gradient-'+name+(mobile?'.mobile.svg':'.svg')]=render(data,mobile);
 }
 return assets;
}
