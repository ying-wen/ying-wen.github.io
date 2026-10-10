#!/usr/bin/env python3
"""Small, inspectable RL algorithms and a continual-learning diagnostic.

Python 3 standard library only. Examples:
  python3 examples/rl_foundations.py prediction --seeds 5 --episodes 1000 --out prediction
  python3 examples/rl_foundations.py control --seeds 5 --episodes 800 --switch 400 --out control
  python3 examples/rl_foundations.py policy --seeds 5 --episodes 800 --switch 400 --out policy
  python3 examples/rl_foundations.py dyna --seeds 5 --episodes 800 --switch 400 --out dyna
  python3 examples/rl_foundations.py consolidation --seeds 1 --out consolidation

Writes PREFIX.csv and PREFIX.html, refusing to overwrite either. These are
tabular teaching experiments, not deep-network or single-life benchmarks.
Prediction: balanced random walk, terminal payoff 0/1, gamma=1, start=3.
Control/policy/Dyna: episodic corridor 0..6, start=3, cost=-.01 per step;
left/right terminal bonuses .1/1, swapped AFTER --switch episodes if nonzero.
The agent sees reward, not the switch flag. Episodes reset position and traces,
not learned values, policy, critic or model. Separate RNG for Dyna planning.
Consolidation: deterministic scalar Gaussian-loss analogy, NOT an RL experiment.
"""
import argparse
import csv
import html
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def softmax(logits):
    maximum = max(logits)
    values = [math.exp(x-maximum) for x in logits]
    return [x/sum(values) for x in values]


def policy_step(logits, probabilities, action, advantage, alpha):
    """For softmax logits: d log pi(a)/d theta_j = 1[j=a] - pi(j)."""
    for j in range(len(logits)):
        logits[j] += alpha*advantage*(float(j==action)-probabilities[j])


def td_target(reward, gamma, next_value, terminal):
    return reward if terminal else reward+gamma*next_value


def q_step(q, state, action, reward, next_state, terminal, alpha, gamma,
           next_action=None):
    """next_action=None gives Q-learning; an actual next action gives SARSA."""
    boot = max(q[next_state]) if next_action is None else q[next_state][next_action]
    target = td_target(reward, gamma, boot, terminal)
    q[state][action] += alpha*(target-q[state][action])


def greedy_epsilon(q, state, epsilon, rng):
    if rng.random()<epsilon:
        return rng.randrange(2)
    best=max(q[state])
    return rng.choice([a for a in range(2) if q[state][a]==best])


def corridor(state, action, flipped=False):
    next_state=state+(-1 if action==0 else 1)
    terminal=next_state in (0,6)
    bonus=(1. if flipped else .1) if next_state==0 else (.1 if flipped else 1.)
    return next_state, -.01+(bonus if terminal else 0.), terminal


def row(seed, method, index, metric, value):
    return dict(seed=seed,method=method,index=index,metric=metric,value=value)


def prediction(seed=0, episodes=1000, alpha=.1, lam=.8, **_):
    # Pre-generate the same fixed-policy trajectories for all three estimators.
    rng=random.Random(seed)
    trajectories=[]
    for _ in range(episodes):
        state,trajectory=3,[]
        while state not in (0,6):
            nxt=state+rng.choice((-1,1))
            trajectory.append((state,float(nxt==6),nxt))
            state=nxt
        trajectories.append(trajectory)
    result=[]
    for method in ('every_visit_mc','td0','td_lambda'):
        values=[0.,.5,.5,.5,.5,.5,0.]
        for episode,trajectory in enumerate(trajectories,1):
            if method=='every_visit_mc':
                returns=[0.]*len(trajectory)
                g=0.
                for t in reversed(range(len(trajectory))):
                    g=trajectory[t][1]+g
                    returns[t]=g
                for (state,_,_),g in zip(trajectory,returns):
                    values[state]+=alpha*(g-values[state])
            else:
                trace=[0.]*7
                for state,reward,nxt in trajectory:
                    delta=td_target(reward,1.,values[nxt],nxt in (0,6))-values[state]
                    trace=[lam*x for x in trace] if method=='td_lambda' else [0.]*7
                    trace[state]+=1.
                    for s in range(1,6):
                        values[s]+=alpha*delta*trace[s]
            if episode%20==0 or episode==episodes:
                rmse=math.sqrt(sum((values[s]-s/6)**2 for s in range(1,6))/5)
                result.append(row(seed,method,episode,'value_rmse',rmse))
    return result


def control(seed=0, episodes=800, alpha=.1, epsilon=.1, switch=0,
            planning=10, experiment='control', **_):
    result=[]
    methods=('sarsa','q_learning') if experiment=='control' else ('q_learning','dyna_q')
    for method in methods:
        rng,plan_rng=random.Random(seed),random.Random(seed+7919)
        q,model,window=[ [0.,0.] for _ in range(7)],{},[]
        env_steps,updates=0,0
        for episode in range(1,episodes+1):
            state,total=3,0.
            action=greedy_epsilon(q,state,epsilon,rng)
            for _ in range(10000):
                nxt,reward,terminal=corridor(state,action,bool(switch and episode>switch))
                total+=reward;env_steps+=1
                next_action=greedy_epsilon(q,nxt,epsilon,rng) if method=='sarsa' and not terminal else 0
                q_step(q,state,action,reward,nxt,terminal,alpha,1.,next_action if method=='sarsa' else None)
                updates+=1
                if method=='dyna_q':
                    # Most recent observed deterministic transition; not an oracle.
                    model[state,action]=(nxt,reward,terminal)
                    keys=list(model)
                    for _ in range(planning):
                        ms,ma=plan_rng.choice(keys)
                        ns,r,d=model[ms,ma]
                        q_step(q,ms,ma,r,ns,d,alpha,1.)
                        updates+=1
                if terminal:break
                state=nxt
                action=next_action if method=='sarsa' else greedy_epsilon(q,state,epsilon,rng)
            else:
                raise RuntimeError('Episode exceeded safety guard; no fabricated terminal target was applied.')
            window.append(total)
            if episode%20==0 or episode==switch or episode==episodes:
                result.extend([row(seed,method,episode,'episode_return',statistics.mean(window)),
                               row(seed,method,episode,'environment_steps',env_steps),
                               row(seed,method,episode,'update_count',updates)])
                window=[]
    return result


def policy(seed=0, episodes=800, alpha=.05, switch=0, **_):
    result=[]
    for method in ('reinforce','actor_critic'):
        rng=random.Random(seed)
        logits,values=[ [0.,0.] for _ in range(7)],[0.]*7
        window=[]
        for episode in range(1,episodes+1):
            state,trajectory,total=3,[],0.
            for _ in range(10000):
                probabilities=softmax(logits[state])
                action=int(rng.random()>=probabilities[0])
                nxt,reward,terminal=corridor(state,action,bool(switch and episode>switch))
                total+=reward
                if method=='actor_critic':
                    delta=td_target(reward,1.,values[nxt],terminal)-values[state]
                    # Actor's direction uses the pre-update critic and current policy.
                    policy_step(logits[state],probabilities,action,delta,alpha)
                    values[state]+=.1*delta
                else:
                    trajectory.append((state,action,reward,probabilities))
                if terminal:break
                state=nxt
            else:
                raise RuntimeError('Episode exceeded safety guard.')
            if method=='reinforce':
                g=0.
                # All gradients use cached probabilities from the rollout policy.
                for state,action,reward,probabilities in reversed(trajectory):
                    g=reward+g
                    policy_step(logits[state],probabilities,action,g,alpha)
            window.append(total)
            if episode%20==0 or episode==switch or episode==episodes:
                result.append(row(seed,method,episode,'episode_return',statistics.mean(window)))
                result.append(row(seed,method,episode,'start_right_probability',softmax(logits[3])[1]))
                window=[]
    return result


def consolidation(seed=0, strength=1., **_):
    """L_A=.5(w-1)^2, L_B=.5(w+1)^2; known curvature F=1.
    Replay optimizes equal-weight losses; EWC adds strength/2*(w-1)^2.
    No estimated neural Fisher, task discovery, or deep RL is implemented.
    """
    result=[]
    for method in ('fine_tune','equal_rehearsal','quadratic_ewc'):
        w=1.
        for step in range(1,201):
            gradient=w+1
            if method=='equal_rehearsal':gradient=.5*(w+1)+.5*(w-1)
            if method=='quadratic_ewc':gradient+=strength*(w-1)
            w-=.05*gradient
            if step%5==0:
                result.extend([row(seed,method,step,'old_loss',.5*(w-1)**2),
                               row(seed,method,step,'new_loss',.5*(w+1)**2)])
    return result


def html_report(rows, config):
    colors=['#2563eb','#8896aa','#ad8244']
    methods=list(dict.fromkeys(r['method'] for r in rows))
    metrics=list(dict.fromkeys(r['metric'] for r in rows))
    titles={'value_rmse':'价值估计 RMSE（越低越好）','episode_return':'每段 episode 的平均回报',
            'environment_steps':'累计真实交互步数','update_count':'累计价值更新次数',
            'start_right_probability':'起点选择右侧的概率','old_loss':'旧目标损失','new_loss':'新目标损失'}
    charts=[];tables=[]
    for metric in metrics:
        data=[r for r in rows if r['metric']==metric]
        maximum=max(r['value'] for r in data);minimum=min(0.,min(r['value'] for r in data))
        top=maximum if maximum>minimum else minimum+1
        xmax=max(r['index'] for r in data)
        def point(index,value):return f'{48+index/xmax*340:.2f},{230-(value-minimum)/(top-minimum)*175:.2f}'
        svg=['<svg viewBox="0 0 420 280" role="img" aria-label="'+titles[metric]+'">']
        for fraction in (0,.5,1):
            value=minimum+fraction*(top-minimum);y=230-fraction*175
            svg.append(f'<path d="M48 {y}H388" stroke="#e2e8f0"/><text x="2" y="{y+4}">{value:.2g}</text>')
        switch=config.get('switch',0) if config['experiment'] in ('control','policy','dyna') else 0
        if switch:
            x=48+switch/xmax*340
            svg.append(f'<path d="M{x} 40V230" stroke="#9a682d" stroke-dasharray="4 4"/>')
        for m,method in enumerate(methods):
            group=[r for r in data if r['method']==method]
            by_step=defaultdict(list)
            for seed in sorted({r['seed'] for r in group}):
                points=sorted((r['index'],r['value']) for r in group if r['seed']==seed)
                for t,v in points:by_step[t].append(v)
                svg.append('<polyline points="'+' '.join(point(t,v) for t,v in points)+f'" stroke="{colors[m]}" opacity=".15" fill="none"/>')
            svg.append('<polyline points="'+' '.join(point(t,statistics.mean(v)) for t,v in sorted(by_step.items()))+f'" stroke="{colors[m]}" stroke-width="2.5" fill="none"/>')
            last=[r['value'] for r in group if r['index']==xmax]
            sd=f'{statistics.stdev(last):.4g}' if len(last)>1 else '—'
            tables.append(f'<tr><td>{titles[metric]}</td><td>{method}</td><td>{statistics.mean(last):.4g}</td><td>{sd}</td></tr>')
        for tick in sorted({0,xmax//2,xmax}):
            svg.append(f'<text x="{48+tick/xmax*340}" y="251" text-anchor="middle">{tick}</text>')
        axis='更新次数' if config['experiment']=='consolidation' else 'Episode（不是环境步数）'
        svg.append(f'<text x="216" y="276" text-anchor="middle">{axis}</text></svg>')
        charts.append('<section><h2>'+titles[metric]+'</h2>'+''.join(svg)+'</section>')
    legend='　'.join(f'<span style="color:{colors[i]}">━ {m}</span>' for i,m in enumerate(methods))
    return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RL 算法教学实验</title><link rel="icon" href="data:,"><style>
body{max-width:1020px;margin:36px auto;padding:0 18px;color:#24334b;background:#fafbfc;font:15px/1.8 system-ui}h1{font-size:28px}h2{font-size:18px}.plots{display:grid;grid-template-columns:1fr 1fr;gap:16px}section{border:1px solid #dde4ee;background:white;padding:20px;border-radius:8px}svg{width:100%;font:12px system-ui;fill:#64748b}pre{overflow:auto;font-size:12px;background:#f1f5f9;padding:18px}table{border-collapse:collapse;font-size:13px}td,th{padding:10px;border-bottom:1px solid #e2e8f0;text-align:left}.scroll{overflow:auto}@media(max-width:680px){.plots{grid-template-columns:1fr}}@media print{body{background:white}.plots{display:block}section{break-inside:avoid}}
</style></head><body><main>'''+f'''<h1>{config['experiment']} · 公式到代码</h1><p>{legend}</p><p>粗线为 seed 均值，浅线为单次运行。竖虚线（若有）是奖励切换，学习器不接收切换通知。横轴单位见各图；同样的 episode 数不保证相同环境步数或更新量。标准差反映 seed 波动，不是置信区间。</p><div class="plots">{''.join(charts)}</div><h2>最后一个记录点</h2><div class="scroll"><table><tr><th>指标</th><th>方法</th><th>均值</th><th>样本 SD</th></tr>{''.join(tables)}</table></div><h2>解释之前先检查</h2><ol><li>有没有把样本数、更新次数与 episode 数混为一谈？</li><li>指标改善来自目标变化、额外计算，还是新的学习规则？</li><li>consolidation 是已知曲率的标量反例；不能据此评价深度 EWC 或 replay。</li></ol><details><summary>完整运行配置</summary><pre>{html.escape(json.dumps(config,ensure_ascii=False,indent=2))}</pre></details></main></body></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment',choices=['prediction','control','policy','dyna','consolidation'])
    p.add_argument('--seeds',type=int,help='Default: 1 for consolidation; 5 otherwise.')
    p.add_argument('--episodes',type=int,default=800)
    p.add_argument('--alpha',type=float,help='Default: 0.05 for policy; 0.1 otherwise.')
    p.add_argument('--epsilon',type=float,default=.1)
    p.add_argument('--lam',type=float,default=.8)
    p.add_argument('--switch',type=int,default=0,help='Reward reversal after this episode; 0 disables.')
    p.add_argument('--planning',type=int,default=10)
    p.add_argument('--strength',type=float,default=1.)
    p.add_argument('--out',type=Path,required=True,help='Output prefix, not a CSV filename.')
    a=p.parse_args()
    if a.seeds is None:a.seeds=1 if a.experiment=='consolidation' else 5
    if a.alpha is None:a.alpha=.05 if a.experiment=='policy' else .1
    if a.seeds<1 or a.episodes<1 or not 0<a.alpha<=1 or not 0<a.epsilon<=1 or not 0<=a.lam<=1 or not 0<=a.switch<a.episodes or a.planning<0 or not 0<=a.strength<=10:
        p.error('Invalid parameters (epsilon must be positive in the corridor).')
    if not all(math.isfinite(v) for v in (a.alpha,a.epsilon,a.lam,a.strength)):
        p.error('Parameters must be finite.')
    if a.experiment=='consolidation' and a.seeds!=1:
        p.error('Consolidation is deterministic; use --seeds 1, not repeated identical evidence.')
    if a.experiment in ('prediction','consolidation') and a.switch:
        p.error('--switch is only used by control, policy and dyna.')
    csv_path,report_path=Path(str(a.out)+'.csv'),Path(str(a.out)+'.html')
    if csv_path.exists() or report_path.exists():p.error('Output already exists; choose a new prefix.')
    if not csv_path.parent.is_dir():p.error('Output parent directory does not exist; create it first.')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    function=control if a.experiment in ('control','dyna') else globals()[a.experiment]
    rows=[r for seed in range(a.seeds) for r in function(seed=seed,**vars(a))]
    if not all(math.isfinite(r['value']) for r in rows):raise ValueError('Non-finite output: inspect stability.')
    document=html_report(rows,config)
    with csv_path.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','method','index','metric','value','config'])
        writer.writeheader();writer.writerows(dict(r,config=json.dumps(config,sort_keys=True)) for r in rows)
    with report_path.open('x',encoding='utf-8') as f:f.write(document)
    print(f'{len(rows)} rows -> {csv_path}\nOpen {report_path.resolve()} in a browser.')


if __name__=='__main__':main()
