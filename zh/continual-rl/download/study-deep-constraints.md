# 约束强化学习：占据测度、拉格朗日与可行策略

“回报高且代价不超过预算”与“每一步都安全”之间差了哪些条件？

## 本章内容

- 从折扣访问频率推导 CMDP 的占据流约束。
- 解释随机策略为何可能必要及拉格朗日乘子的方向。
- 区分平均代价、概率约束、硬约束和训练期安全。

<a id="problem-definition"></a>

## 本章的问题定义

除了收益，还规定累计成本预算；允许的策略必须同时满足目标和约束。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 成本 $C_{t+1}$、预算 $b$，$J_C(\pi)=\mathbb E_\pi[\sum_{t\ge0}\gamma^tC_{t+1}]$。

### 需要求解的对象

在可行策略集合中最大化收益，必要时采用随机化策略。

### 信息与数据权限

成本是额外观测信号；执行时安全约束若要求每步成立，需要另给可验证信息。

$$
\max_\pi J(\pi)\quad\text{s.t.}\quad J_C(\pi)\le b
$$

这是期望累计成本约束。它不等价于每条轨迹、每个时刻都安全，也不等价于预设一个固定惩罚系数。

### 成立条件与解的含义

- 可行集非空；成本尺度与时间聚合明确。
- 强对偶及线性规划解释依赖相应有限凸表示条件。

判断准则：同时报告收益、约束违反量、置信区间和可行运行比例。

### 适用边界

- 从平均约束满足推出零事故或逐步安全保证。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)：本章从一般问题规格中选取有期望累计成本预算的 CMDP；逐步安全等其他约束不由这个特例涵盖。

- 改变评价目标 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：相对仅按收益排序的控制，本章增加成本可行性条件，再在可行集合内最大化收益；固定惩罚权重不是同一个问题。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

最高收益策略可能不可行；固定惩罚难以事先对应指定预算。

### 本章的核心思路

用占据测度定义可行策略，再将约束违反反馈给乘子和策略更新。

1. [把策略转成长期访问量](#lesson-notation)：占据流约束连接局部转移与整个策略表现。

2. [在可行集合里求最优](#lesson-derive)：线性例子展示为何随机化可以成为必要条件。

3. [把预算误差反馈给惩罚强度](#lesson-dual)：乘子方向来自对偶问题，不能随意将负成本当成另一种奖励。

结论与条件：期望 CMDP 保证、CPO 局部近似与执行时安全是不同强度的结论。

### 相关方法改变了什么

- 固定 penalty / dual learning：前者指定权衡，后者根据违反预算调节权衡。

- CMDP / safety shield：前者约束长期统计，后者在执行时限制具体动作。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 条件概率与期望

区分随机变量、一次样本与条件分布；可以使用 Bayes 法则和全期望公式。

### MDP 与价值

理解状态、动作、转移、奖励、策略和 Bellman 递推；学习数据可以来自不同策略。

### 优化与梯度

知道固定目标、参数梯度、约束和期望目标的含义；神经网络只是函数表示的一种选择。

<a id="lesson-setting"></a>

## 1 · 约束不是另一个奖励名称

$$
\max_\pi J_r(\pi),\qquad J_c(\pi)=\mathbb E_\pi\sum_{t=0}^\infty\gamma^t C_{t+1}\le d
$$

CMDP 在回报之外声明代价预算。代价可以表示能耗、碰撞次数或其他事件，但其定义由任务提供。

给奖励减去固定惩罚系数，是优化一个标量化目标；满足给定预算是另一个问题。除非有额外分析，不能从“惩罚很大”推出预算合格。设计者提供成本传感器、约束阈值、初始分布和事故定义；这些量并不是算法从奖励标量自动推断出的。

这里先考虑有限状态动作、平稳转移和折扣期望代价。期望累计代价小，仍允许某些轨迹发生严重事故。若目标是每次执行都不越界、事故概率低于阈值或不可逆状态永不进入，需要不同约束与模型条件。

<a id="lesson-notation"></a>

## 2 · 用占据测度表示整个策略

$$
x_\pi(s,a)=(1-\gamma)\mathbb E_\pi\sum_{t=0}^\infty\gamma^t\mathbf1[S_t=s,A_t=a],\qquad x_\pi\ge0
$$

规范化后的折扣占据测度总和为一。它记录状态动作被使用多少，而不只记录每个状态下的动作概率。

$$
\sum_ax(s,a)=(1-\gamma)\nu(s)+\gamma\sum_{\tilde s,a}P(s\mid\tilde s,a)x(\tilde s,a)
$$

把时间零与随后时刻的访问拆开。后续访问等于来自所有前驱的折扣流入，得到线性流守恒。

$$
\pi(a\mid s)=\frac{x(s,a)}{\sum_{a'}x(s,a')}
$$

状态占据非零时可以恢复平稳随机策略；零占据状态的动作可另作定义，不能由零除法确定。

<a id="lesson-derive"></a>

## 3 · CMDP 的线性规划与随机化

$$
(1-\gamma)J_r=\sum_{s,a}x(s,a)r(s,a),\qquad(1-\gamma)J_c=\sum_{s,a}x(s,a)c(s,a)
$$

归一化占据对应归一化收益与代价，因此预算也必须乘相同因子。

$$
\max_{x\ge0}\sum_{s,a}x(s,a)r(s,a)\quad\text{s.t. flow},\qquad\sum_{s,a}x(s,a)c(s,a)\le(1-\gamma)d
$$

已知有限模型下，目标与约束对占据测度是线性的。这个精确规划形式不等于神经策略优化已经解决可行性。

没有约束时，有限折扣 MDP 可以找到最优确定策略；加入成本预算后，最优可行解可能需要随机化。随机化是为了在回报和代价之间满足预算，不只是训练时探索。若所有策略都超出预算，问题本身不可行，优化器不能创造一条不存在的安全策略。

<a id="lesson-dual"></a>

## 4 · 乘子怎样把违反预算反馈给策略

$$
\mathcal L(\pi,\lambda)=J_r(\pi)-\lambda(J_c(\pi)-d),\qquad\lambda\ge0
$$

策略对拉格朗日函数做最大化，乘子对其做最小化。固定乘子相当于使用奖励减成本的目标。

$$
\begin{gathered}\theta\leftarrow\theta+\eta_\theta(\nabla J_r-\lambda\nabla J_c)\\\lambda\leftarrow[\lambda+\eta_\lambda(\widehat J_c-d)]_+\end{gathered}
$$

代价超预算时乘子上升，使下一轮策略更重视代价。正部投影保证乘子非负。

$$
\lambda^*(J_c(\pi^*)-d)=0
$$

互补松弛描述理想最优解：未激活的约束可有零乘子；正乘子对应预算恰好激活。它不是每个学习迭代都应满足的恒等式。

有限已知 CMDP 的占据线性规划可以分析对偶与可行性；神经参数化、采样噪声和同时更新可能破坏这些简洁性质。乘子振荡、成本 critic 滞后和策略表示不足都会影响实际学习，不能只看最后一次 multiplier。

<a id="lesson-safety"></a>

## 5 · CPO 与执行时安全的区别

CPO 在策略更新中近似约束代价变化，并限制 KL 信任域，试图在提高回报时维持约束。其分析依赖真实或受控的估计、局部近似和具体约束形式；样本实现并不因此成为任意轨迹的无事故证书。

$$
\max_\Delta g_r^\top\Delta\quad\text{s.t.}\quad \hat J_c+g_c^\top\Delta\le d,\qquad\tfrac12\Delta^\top F\Delta\le\delta
$$

这是理解局部 constrained update 的近似优化问题。若当前策略已不可行，需恢复步或其他处理，不能默认约束一定有满足的上升方向。

执行时 shield 或 barrier 方法限制具体动作的可行集合，常需要可靠动力学、状态估计与不变集条件。平均约束、鲁棒控制与这些执行过滤机制可以组合，但它们保护的对象不同。训练时违反多少约束，也应与最终冻结策略的代价分开报告。

<a id="lesson-algorithm"></a>

## 6 · 同时记录收益、代价和可行性

**算法：本页数值核采用旧策略与旧乘子同时计算两方向，避免时间顺序含糊。**

1. 明确代价事件、预算、折扣与初始分布。
1. 检查是否存在已知可行基线；没有时不能假设问题可行。
1. 采样策略数据，分别估计 reward return 与 cost return。
1. 固定旧策略估计，计算策略与乘子的方向。
1. 投影乘子到非负区间，按声明的顺序更新策略。
1. 记录每轮预算违反、训练期累计代价与独立评估。
1. 需要硬执行保证时另行验证安全模型和动作过滤器。

<a id="lesson-example"></a>

## 7 · 一个必须混合的最优策略

单状态两动作，奖励为 $(1,3)$，代价为 $(0,2)$。使用规范化代价预算 $0.6$，选择动作一的概率为 $p$。奖励为 $1+2p$，代价为 $2p$，所以最优可行概率 $p=0.3$，收益为 $1.6$。确定选高奖励动作不可行，确定选低成本动作可行但次优。

若 $\gamma=0.9$，上述规范化预算 0.6 对应未规范化预算六，收益 1.6 对应折扣总收益十六。把 0.6 与未规范化成本二十比较，是单位错误。单状态占据 $(0.7,0.3)$ 满足流守恒。

固定惩罚 $\lambda<1$ 时仍倾向完全选择高成本动作；$\lambda>1$ 时倾向低成本动作；$\lambda=1$ 时两动作标量价值相同。正确乘子不独自指定混合比例，还需可行性和策略解。

<a id="lesson-code"></a>

## 8 · 占据流、解析混合与 primal–dual 核

规范化流残差、两动作精确可行解以及同时计算的策略/乘子一步更新。

```python
def occupancy_residual(occupancy, transition, initial, gamma):
    """Normalized discounted occupancy x[s][a] and P[s][a][next_state]."""
    probabilities(initial)
    if not 0 <= gamma < 1:
        raise ValueError('discount below one required')
    n = len(initial)
    return [sum(occupancy[s])-(1-gamma)*initial[s]
            -gamma*sum(occupancy[sp][a]*transition[sp][a][s]
                       for sp in range(n) for a in range(len(occupancy[sp])))
            for s in range(n)]


def constrained_two_action(rewards, costs, budget):
    """One-state normalized occupancy LP; costs[1] > costs[0]."""
    if costs[1] <= costs[0]:
        raise ValueError('ordered distinct costs required')
    if budget < costs[0]:
        raise ValueError('infeasible budget')
    largest = min(1., (budget-costs[0])/(costs[1]-costs[0]))
    p = largest if rewards[1] > rewards[0] else 0.
    return p, (1-p)*rewards[0]+p*rewards[1], (1-p)*costs[0]+p*costs[1]


def primal_dual_step(logit, multiplier, reward_gap, cost0, cost_gap, budget,
                     actor_step=.1, dual_step=.1):
    probability = 1/(1+math.exp(-logit))
    gradient = probability*(1-probability)*(reward_gap-multiplier*cost_gap)
    cost = cost0+cost_gap*probability
    # Both directions use the same old policy and multiplier.
    return logit+actor_step*gradient, max(0., multiplier+dual_step*(cost-budget))
```

运行 test 检查流守恒、最优混合、不可行预算拒绝和 primal 梯度有限差分。代码没有实现 CPO 的共轭梯度、线搜索或安全执行层。jachiam/cpo 提供原作者研究代码；Safety Gym 是代价环境，不是安全性证明工具。

<a id="lesson-branches"></a>

## 9 · 约束在持续任务中也会变化

- 风险漏记：成本传感器未覆盖的事故，不会被成本 critic 自动补充。
- 尾部不足：低频严重事故可能在平均样本里消失；要与尾部风险和置信边界区分。
- 非平稳阈值或动力学：旧可行策略可能变得不可行，历史成本回放也可能不再代表当前风险。

CRL 的单生命期要求将探索成本、适应阶段事故和恢复能力计入评价，而非只测训练后的策略。状态预测、模型规划与约束优化相互影响：不能观察到危险状态的 agent，即使优化器正确，也可能无法执行所需安全控制。

<a id="lesson-check"></a>

## 10 · 练习与答案

- 问：代价期望小于预算，是否保证每条轨迹安全？答：不是；这是平均约束。
- 问：提高固定惩罚一定能找到最优可行策略吗？答：不保证；混合策略、可行性及优化误差仍需处理。
- 问：所有动作成本至少一而预算 0.6，会发生什么？答：此单状态问题不可行，程序应明确拒绝。
- 实验：将预算从 0.6 改为一，解析最优 p 应从 0.3 变为 0.5。再检查规范化与未规范化指标是否同步改变。



<a id="chapter-code"></a>

## 下载与运行

标准库解析机制实验；不包含完整神经网络训练或大型 benchmark。

[下载 extended_foundations_lab.py](https://yingwen.io/zh/continual-rl/download/extended_foundations_lab.py)

```sh
python3 extended_foundations_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Altman · Constrained Markov Decision Processes](https://www.routledge.com/Constrained-Markov-Decision-Processes/Altman/p/book/9781315140223)：原作者专著；占据测度、线性规划与拉格朗日方法的系统来源。

- [Achiam et al. · Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a.html)：CMDP、约束策略改进与局部信任域近似。

- [Achiam · CPO 原作者代码](https://github.com/jachiam/cpo)：原研究工程；算法实现与环境依赖应分别阅读。

- [OpenAI · Safety Gym](https://github.com/openai/safety-gym)：原团队的带成本环境，用于约束学习评估；不提供普遍安全保证。

- [Kochenderfer、Wheeler、Wray · Algorithms for Decision Making](https://algorithmsbook.com/)：作者书站提供决策、信念状态、模型与规划教材及配套 Julia 代码入口。

<a id="study-connections"></a>

## 与教材主线的衔接

本章是可按问题选择的并列研究分支，不要求先读完其他深度研究分支。

[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)
- [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

对应原始材料：Constrained Markov Decision Processes；Achiam et al.：Constrained Policy Optimization。本文为原创讲解，原书、论文与上游代码保留各自许可。
