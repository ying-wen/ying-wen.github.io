# 学习与规划：Dyna、优先扫描和执行时搜索

真实经验既能直接改进价值，也能训练后果模型。规划使用这个模型继续计算，关键是模型语义、backup 的成本以及计算应分配到哪里。

## 本章内容

- 实现 Dyna 的真实交互、模型更新与模拟更新循环。
- 推导优先扫描的前驱传播，区别 sampling 与 expectation backup。
- 理解 rollout、MCTS 与持续环境中模型错误的边界。

<a id="problem-definition"></a>

## 本章的问题定义

世界模型可查询或可由经验估计。真实交互有限，额外模型计算用于改进价值和决策。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 模型 $\hat p$ 与每次真实交互之间允许的备份预算 $B$。

### 需要求解的对象

在预算内安排模型备份，并从改进后的价值或搜索结果选择动作。

### 信息与数据权限

真实样本与模型生成样本来源不同；模型计算不会增加真实环境的信息量。

$$
(\hat T_*Q)(s,a)=\mathbb E_{\hat p}[R+\gamma\max_{a^{\prime}}Q(S^{\prime},a^{\prime})\mid s,a],\qquad N_{\rm backup}\le B
$$

模型备份逼近在模型中的最优性递推；最终仍按真实环境收益评价。模型内残差降低不代表真实控制误差降低。

### 成立条件与解的含义

- 模型语义、终止和奖励必须与外部过程一致。
- 精确模型下的 DP 结论不能直接用于有偏学习模型。

判断准则：分别计数真实步、模型查询和耗时，并区分采样误差、模型误差与规划不足。

### 适用边界

- 把模型 rollout 当作无成本的真实经验。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [Dyna 与模型学习](https://yingwen.io/zh/continual-rl/algorithms/dyna/)：本章把模型备份与真实更新、模型估计组合成 Dyna 循环，分别检查三者误差。

- 特例：增加条件 · [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)：本章限定为原子动作的表格备份；时间抽象规划还需技能持续时间和终点模型，是相关章更广的接口。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

计算预算不足以遍历全部状态；模型本身又可能过时或错误。

### 本章的核心思路

区分模型如何学习、一次备份如何执行，以及备份应分配到哪里。

1. [决定求期望还是采样](#lesson-derive)：两种备份在模型正确时指向同一对象，但方差与计算成本不同。

2. [把模型接入真实学习循环](#planning-dyna)：每次真实经验既更新价值也更新模型，再调用预算内的模型更新。

3. [把有限计算放到误差会传播的位置](#planning-priority)：前驱和优先队列加速传播，但队列本身不保证模型正确。

结论与条件：精确备份和充分覆盖可连接 DP 理论；有限预算、有偏模型需单独测量控制后果。

### 相关方法改变了什么

- Dyna / MPC：Dyna 将模型计算写入持久价值；MPC 为当前动作重新规划。

- 均匀采样 / 优先扫描：改变备份分配，不改变模型真实性条件。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 概率与期望

概率非负且总和为一；期望是按概率加权的平均，不是单次结果。条件期望只比较满足已知条件的情形。

### 递推与赋值

递推通过已有估计计算新估计。下标表示时间；更新符号表示程序中的赋值，不是数学恒等式。

<a id="lesson-setting"></a>

## 1 · 什么是模型与规划

分布模型给出所有可能后果及其概率；采样模型每次返回一个随机后果。它们都不是价值函数：模型描述动作之后会发生什么，价值结合奖励目标与继续行为评价这些后果。规划通过模型而非真实执行取得更新信息。

本章先使用有限平稳 MDP、折扣 $\gamma<1$。真实转移为 $(s,a,r,s')$，终止单独标记。若模型恰好等于环境，规划收敛指向环境中的相应价值；若模型错误，规划求解的是模型所定义的问题。

<a id="lesson-derive"></a>

## 2 · 期望 backup 与采样 backup

$$
y_{\rm exp}(s,a)=\hat r(s,a)+\gamma\sum_{s'}\hat p(s'\mid s,a)\max_bQ(s',b)
$$

模型期望枚举所有后果。若奖励与后继相关，也可以直接对联合模型求和；r̂ 是相应即时奖励期望。

$$
(\tilde r,\tilde s')\sim\hat p(\cdot,\cdot\mid s,a),\qquad y_{\rm sample}=\tilde r+\gamma\max_bQ(\tilde s',b)
$$

采样一次后果。固定 Q 且采样模型正确时，样本 target 的期望等于期望 backup；一次样本方差一般不为零。

期望 backup 的成本随分支数增长；采样 backup 每次便宜，但需要重复才能降低随机性。比较应匹配计算成本或模型调用次数，而不是把“一次更新”当作跨算法相同单位。对确定性环境，两者在后果计算上可以相同。

<a id="planning-dyna"></a>

## 3 · Dyna 的三条更新路径

真实经验可以直接训练 Q，也可以训练模型。模型学得的后果再生成模拟转移，继续使用同一种价值更新。Dyna 不是先学完模型再独立规划，而是把行动、直接学习、模型学习和规划交错执行。

**算法：Dyna-Q 的完整循环**

1. 初始化 Q 与空模型。
1. 每次真实交互：
  1. 按 Q 的探索策略选动作，执行后得到奖励、后继与终止。
  1. 对这条真实经验做一次 Q-learning 更新。
  1. 把后果加入模型。
  1. 重复 `n_plan` 次：
    1. 从已建模的状态动作中选择一个。
    1. 查询模型取得后果，再做一次价值更新。
  1. 继续真实交互；真正终止时按环境协议开始下一 episode。

本例环境确定，模型保存最后一次后果即可。随机环境中只保存最后样本会把随机分布误当确定事实，需要计数分布、条件生成模型或其他估计。还要注意模型中的 terminated，不应在模拟中把终点当成普通后继状态。

<a id="planning-priority"></a>

## 4 · 优先扫描与前驱传播

一个终点奖励首次被发现后，最值得更新的往往是能到达它的前驱，而不是所有已知状态均匀抽样。优先扫描记录模型的前驱关系，用预期 backup 改变量作为优先级，弹出高优先级项更新，再检查它的前驱。

$$
P(s,a)=\left|\hat r(s,a)+\gamma\sum_{s'}\hat p(s'\mid s,a)V(s')-Q(s,a)\right|
$$

优先级衡量当前模型 target 与旧估计的差，不是动作真实价值或样本不确定性的估计。

**算法：算法伪代码**

1. 对有变化的状态动作计算优先级；超过阈值则入队。
1. 在规划预算内：
  1. 弹出优先级最高的项目，重新检查其当前误差。
  1. 若已过时且误差很小则跳过；否则执行 backup。
  1. 查找能转移到这个状态的前驱项目。
  1. 为这些前驱重新计算优先级，必要时加入队列。

模型改变时前驱图也要更新，例如移除不再成立的确定性边。队列里重复或过时的优先级必须处理；队列算法正确不意味着模型正确。随机转移时，后继变化对前驱的影响还受转移概率加权。

<a id="planning-search"></a>

## 5 · 从后台规划到执行时搜索

后台 Dyna 更新长期保留的价值表，执行时可快速查表。执行时规划则围绕当前状态花计算：rollout 比较候选首动作后，沿某个默认策略模拟；MPC 规划有限动作序列，执行首动作后用真实观测重新规划。

MCTS 维护当前搜索树，通常循环选择、扩展、模拟或叶值评价、回传统计四步。UCT 用访问统计权衡树上探索与利用，它借鉴 bandit，但树节点的回报分布会随搜索策略和子树增长变化，不能直接把独立平稳老虎机假设照搬。

$$
a\in\arg\max_b\left[\widehat Q(s,b)+c\sqrt{\frac{\log N(s)}{N(s,b)}}\right]
$$

这是常见 UCT 选择形式；未访问边单独优先处理。最终执行哪个根动作、是否保留搜索树、叶值来自 rollout 还是网络，仍是算法定义的一部分。

搜索必须拥有可查询的模型或仿真器。能任意恢复仿真状态，不等于真实机器人拥有这种 reset 权限。策略价值网络、模型学习与搜索可组合，但搜索本身不能验证模型外推到未见区域是否正确。

<a id="lesson-example"></a>

## 6 · 手算：奖励的反向传播

四个非终止状态 0→1→2→3→终点，只有最后一步奖励 1，γ=.9，初值全零。已知确定性模型下，最初只有状态 3 的误差为 1。更新它后，状态 2 的误差变 .9，然后状态 1 为 .81，状态 0 为 .729。

| 第几次 backup | 更新状态 | 新值 |
| --- | --- | --- |
| 1 | 3 | 1 |
| 2 | 2 | .9 |
| 3 | 1 | .81 |
| 4 | 0 | .729 |

真实 Dyna 实验允许左右动作，从状态 0 出发，左边界停在 0。探索 ε=.5，4000 次真实交互；每步规划 5 次时增加 20000 次模型更新。向右价值接近 (.729,.81,.9,1)。这个最终一致性不是“规划比无规划更快”的充分证据，还需要整个学习曲线与计算对照。

<a id="lesson-code"></a>

## 7 · 可运行模型循环

Dyna 的真实交互与模拟更新计数，以及含过时优先级检查的反向扫描。

```python
def dyna(planning_steps=5, real_steps=4000, seed=7):
    """Deterministic chain: a last-outcome model is exact after observation."""
    rng = random.Random(seed)
    q = {s:[0.,0.] for s in range(4)}
    model, state, simulated, terminals = {}, 0, 0, 0
    def update(s, a, reward, sp):
        target = reward+(.9*max(q[sp]) if sp is not None else 0.)
        q[s][a] += .2*(target-q[s][a])
    for _ in range(real_steps):
        action = choose(epsilon_probs(q[state], .5), rng)
        position = max(0, state+(1 if action else -1))
        reward, sp = float(position==4), None if position==4 else position
        update(state, action, reward, sp)
        model[state,action] = reward,sp
        for _ in range(planning_steps):
            s,a = rng.choice(list(model))
            r,sn = model[s,a]
            update(s,a,r,sn)
            simulated += 1
        terminals += int(sp is None)
        state = 0 if sp is None else sp
    return {"right_values":[q[s][1] for s in q], "reference":[.729,.81,.9,1],
            "real_steps":real_steps,"model_updates":simulated,"episodes":terminals}

def prioritized_chain():
    """Exact one-action model: changed terminal reward launches a reverse wave."""
    model = {s:(float(s==3),None if s==3 else s+1) for s in range(4)}
    predecessors = {s:[] for s in model}
    for s,(_,sp) in model.items():
        if sp is not None:
            predecessors[sp].append(s)
    values,heap,sequence = {s:0. for s in model},[],[]
    def target(s):
        reward,sp = model[s]
        return reward+(.9*values[sp] if sp is not None else 0)
    for s in model:
        error = abs(target(s)-values[s])
        if error>1e-12:
            heapq.heappush(heap,(-error,s))
    while heap:
        _,s = heapq.heappop(heap)
        if abs(target(s)-values[s])<=1e-12:
            continue  # stale priority
        values[s] = target(s)
        sequence.append(s)
        for predecessor in predecessors[s]:
            error = abs(target(predecessor)-values[predecessor])
            if error>1e-12:
                heapq.heappush(heap,(-error,predecessor))
    return {"backup_order":sequence,"values":values}
```

planning 模式分别运行零规划与每步五次规划，再输出优先扫描顺序 [3,2,1,0]。源码范围是表格 Dyna 和小链上的优先扫描；MCTS、世界模型与机器人控制还需要相应的搜索结构、模型接口和环境约束。

规划和行为当前使用同一随机数发生器，因此两个规划预算消耗随机数的顺序不同；它们不是严格共同随机数的配对实验。比较统计性能时，应先决定共享哪些环境随机事件、哪些采样过程独立，并相应构造随机流。

<a id="lesson-branches"></a>

## 8 · 模型错误与持续变化

未访问的后果不会因增加规划次数被自动发现。若路径突然堵塞，旧模型可能反复建议已失效的路线；如果新通道出现但从不尝试，也不会更新模型。Dyna-Q+ 在规划奖励中加入久未实际尝试的探索奖励，是主动重新检验模型的一种思路。

$$
\tilde r=\hat r+\kappa\sqrt{\tau(s,a)}
$$

τ 表示距离上次真实尝试的时间，不能每次模拟调用就清零。这是探索启发式，改变用于规划的奖励，不是对真实环境奖励的无偏估计；本节代码未实现该扩展。

持续规划要平衡新信息获取与已有信息利用，另需限制模型大小、旧知识失效和每步计算。更长模拟并不总更好，因为错误可以积累并被控制器利用。模型准确性、规划计算误差与真实控制收益应分开评估。

<a id="lesson-check"></a>

## 9 · 带答案练习

问：一个随机模型以 .75 概率 target=1、.25 概率 target=5，期望 backup 是多少？答：2。单次采样只能看到 1 或 5，但固定 Q 下其期望为 2。

问：把模拟步数算进环境样本数会怎样？答：混淆信息来源和计算成本。应单列真实交互、模型训练、模型调用与 backup 次数。

问：未发现终点奖励时，给零奖励旧模型做百万次规划能学会正确路线吗？答：一般不能。计算无法替代缺失事实；需要实际探索或额外先验。



<a id="chapter-code"></a>

## 下载与运行

Python 3，仅标准库。下载完整文件后在其目录运行；页面展示核心区域。小环境和解析检验用于理解机制，不是论文基准复现。

[下载 tabular_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/tabular_textbook_lab.py)

```sh
python3 tabular_textbook_lab.py planning
python3 tabular_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：对应第 2–8 章。以下推导、例子和标准库实现独立编写；原书用于核对定义、算法关系与适用条件。

- [David Silver · UCL 强化学习课程](https://davidstarsilver.wordpress.com/teaching/)：原课程页面提供 MDP、动态规划、预测、控制、探索和规划的讲义及视频。

- [Sutton · Dyna: An Integrated Architecture for Learning, Planning, and Reacting](https://doi.org/10.1145/122344.122377)：真实学习、模型学习和规划的原始架构。

<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [Dyna 与模型学习](https://yingwen.io/zh/continual-rl/algorithms/dyna/)
- [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)
- [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)
- [持续学习的智能体架构](https://yingwen.io/zh/continual-rl/construction/architectures/)

对应原始材料：8.1–8.11。本文为原创讲解，原书、论文与上游代码保留各自许可。
