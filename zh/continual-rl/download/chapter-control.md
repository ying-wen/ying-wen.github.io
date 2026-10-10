# 持续控制：比较策略与学习智能体

一个智能体当前做得好，不代表它以后仍能学得好。持续控制要评价完整的行动—学习过程：行动改变世界和数据，学习改变后续行动，有限记忆与计算又限制了这个过程。本章从这些依赖出发，定义可以比较的对象，并用可解析反例检验不同评价标准。

## 本章内容

- 区分固定策略、历史依赖智能体、学习规则与外部设计者，写出完整交互时序。
- 说明价值在持续学习中如何定义，以及当前冻结策略的价值遗漏了什么。
- 逐步计算短期与长期排序反转、相同策略的不同学习能力、不可逆后果三个反例。
- 区分可实现比较者、知晓未来的 oracle、偏离遗憾与从初始世界出发的比较。
- 运行精确枚举、重要性采样和有限差分测试，为自己的 CRL 实验写明目标、信息与资源边界。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [强化学习问题的形式化：交互、目标与持续学习](https://yingwen.io/zh/continual-rl/foundations/objectives/)：区分固定策略的表现与整个学习过程的收益。


### 期望与条件信息

$\mathbb E[X\mid h]$ 表示已经知道历史 $h$ 后，对仍未知的后果取平均。不同条件信息对应不同的预测问题。

### 策略与学习规则

策略决定现在怎样行动；学习规则决定收到经验后，未来的策略怎样改变。学习率、优化器状态、记忆与表示更新都会影响后者。

### 一次环境转移

时刻 $t$ 的智能体先选 $A_t$，随后收到 $R_{t+1},O_{t+1}$，再更新内部状态。动作不能使用尚未收到的后果。

<a id="problem-definition"></a>

## 本章的问题定义

行动与学习共同决定后续数据和行为。持续控制评价完整学习智能体，冻结策略评价只是其中一个局部工具。

### 给定条件与符号

- 世界条件规律或声明的世界分布、初始化与评价时域。
- 允许的智能体/比较器集合、资源限制、预训练与人工重置权限。

### 需要求解的对象

在声明条件内选择和比较完整的因果行动—学习过程；经典策略改善只解决其中固定表示、固定任务的局部控制。

### 信息与数据权限

世界由 $e(r,o'\mid h,a)$ 描述，$h$ 为已到达历史；完整内部状态 $Z_t$ 含参数、记忆和优化器状态，更新 $U$ 只能在收到后果后执行。

$$
J_{T,\gamma}(\mathcal A,e)=\mathbb E_{\mathcal A,e}\!\left[\sum_{t=0}^{T-1}\gamma^tR_{t+1}\right],\qquad \sup_{\mathcal A\in\mathfrak A_B}J_{T,\gamma}(\mathcal A,e)
$$

$\mathcal A$ 是行为、更新与初始化组成的完整算法，$\mathfrak A_B$ 是预算 $B$ 内且权限一致的因果候选集合，$T$ 为寿命，$\gamma$ 为权重。此式定义比较目标，不主张未知大世界存在可计算的全局最优学习器；实际历史条件下的偏离比较是另一估计对象。

### 成立条件与解的含义

- 未来信息不可用于已封存的设计；比较器不得暗中具有真模型、任务标签或额外重置。
- 价值存在需可积；日志反事实估计另需覆盖与正确行为概率，单次真实生命期不提供所有分支。

判断准则：精确复算短长时域排序、同当前策略不同更新规则和不可逆失败三个反例；独立生命期比较累计收益、恢复和永久失败，局部偏离差不能替代初始世界比较。

### 适用边界

- 无重置部署不等于无预训练、无历史数据或严格流式。
- 局部偏离遗憾小不证明过去没有不可逆损害。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)：固定策略价值支持局部评价；持续学习器的未来还包含更新与探索。

- 组合不同学习问题 · [持续学习的智能体架构](https://yingwen.io/zh/continual-rl/construction/architectures/)：完整控制对象要求将模块状态、共享资源和更新时序落实为一个算法。

- 改变信息或数据协议 · [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)：生命期比较需封存设计、独立测试和允许的比较器，避免测试未来回流到调参。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

相同当前动作分布可有不同未来学习能力；事后局部比较还可能遗漏过去毁坏的可达性。

### 本章的核心思路

把更新规则与内部状态纳入比较对象，再按具体信息权限选择可识别的价值或差值。

1. [完整描述学习过程](#control-agent)：因为网络权重不能决定所有未来更新，将记忆、优化器、初始化与更新顺序一起写成算法。

2. [选择可实现比较器](#control-comparators)：因为知晓未来的oracle改变了问题，先固定预算和信息，再定义从初始世界或实际历史出发的比较。

3. [估计并检验比较边界](#control-deviation)：因为每条真实历史只记录一个分支，偏离估计要检查行为支持；以不可逆反例检验它遗漏的初始决策后果。

结论与条件：给定世界和完整算法时可定义可积条件价值；重要性采样正确性依赖支持与概率。反例和局部估计器不证明通用CRL算法最优。

### 相关方法改变了什么

- 冻结策略评价/GPI：固定后续行为或逐次评价改善，遗漏后续学习能力。

- 实际历史偏离比较：从已发生的世界条件评价可行偏离，无法自动反映过去毁坏。

- 初始世界生命期比较：保留行动造成的长期分支，需要独立可重复世界或额外识别假设。


<a id="lesson-setting"></a>

## 1 · 持续控制的对象

前面的[目标与评价章](/zh/continual-rl/foundations/objectives/#lesson-noisy-lifetime)已经在同一岔路迷宫中给出一个问题：胜留败换与始终向左的学习器，在 48 个真实步内的期望收益为 18.48 与 15.20；末态复制到 A 并冻结方向更新，8 圈诊断却是 6.56 与 7.20。两种排序都算对了，因为一项评价交互中的学习过程，另一项评价留下的当前行为。本章从这个差别继续问：怎样定义完整的比较对象，未来允许它更新什么，又能依据哪些经验估计它的表现？

Q-learning 和策略梯度本来就从交互中学习。经典推导常先固定环境与评价目标，分析某一策略的价值或参数化策略的改善方向。持续控制把学习期间的代价和以后继续学习的能力也放进比较：固定策略评价问“以后一直按 $\pi$ 行动会怎样”；完整学习器评价让探索、参数更新、表示与记忆管理按声明的规则继续发生。两者使用的仍是明确的回报准则，区别在未来行为由谁决定。

| 比较对象 | 固定的部分 | 允许变化的部分 |
| --- | --- | --- |
| 冻结策略 | 当前策略映射及其参数 | 世界状态、由行动生成的轨迹 |
| 持续学习智能体 | 完整算法与初始化协议 | 参数、内部记忆、行为和访问分布 |
| 算法设计方案 | 资源与信息约束、测试协议 | 设计者在开发阶段选择的结构与超参数 |

这三个对象可以在同一个世界中研究。外界规律固定时，有限容量的学习器仍可能持续改写知识；外界变化时，也要检查有限的变化类型是否已被同一个固定行为规则覆盖。以下先把算法内部状态和未来更新写清楚，再区分从生命期起点比较、从实际历史比较，以及比较者额外知道未来的情形。是否要求永不停止学习，另由所采用的形式化定义决定。

<a id="control-designer"></a>

## 2 · 智能体、外部设计者与大世界

运行时的闭环只有智能体与世界：智能体行动，世界返回后果，智能体更新。外部设计者位于这个闭环之外，选择奖励接口、初始表示、网络结构、预训练数据、调参方式和可用资源。设计者若在运行中改变这些选择，就引入了新的外部干预；该干预必须属于明确的协议，而不能隐含在算法描述中。

| 层次 | 例子 | 必须说明的问题 |
| --- | --- | --- |
| 智能体内部的因果更新 | 根据刚收到的 TD 误差调整学习率 | 更新依赖哪些已到达数据？用了多少计算？ |
| 设计阶段的选择 | 比较多个学习率后选一个 | 选择时看过多少环境、多少未来轨迹？ |
| 运行中的外部干预 | 失败后人工重置、指定新目标 | 谁决定干预？成本是否计入生命周期？ |

MDP 是环境与决策信息的一种数学描述，它本身并不要求外部设计者参与每一步。这里区分三方，是为了追踪知识、计算和控制权限的来源。若把设计者在完整测试生命期上找到的最佳参数当成智能体自行适应，就混合了两种能力。奖励表达了哪种偏好，以及奖励如何由设计者给定或由智能体建模，在“奖励假设与设计”一章单独讨论。

因此，比较算法之前至少要固定：世界分布或世界类别、初始化与预训练权限、在线可见的信息、是否允许重置、每步计算和记忆预算。允许先用其他世界做开发并不等于违反因果性；关键是测试世界的未来信息不能回流到已经封存的设计选择。

<a id="control-agent"></a>

## 3 · 从历史到学习规则

记 $h_t=(o_0,a_0,r_1,o_1,\ldots,a_{t-1},r_t,o_t)$。世界通过条件分布 $e(r,o'\mid h_t,a)$ 给出下一次后果，不要求当前观测是 Markov 状态。历史依赖智能体是映射 $\lambda:h\mapsto\Delta(\mathcal A)$。这一定义只描述可见行为；要讨论学习过程，还需要说明行为如何由策略与学习规则组成。

$$
\begin{aligned}
 S&:\mathcal H\to\mathcal S,\qquad
 \pi:\mathcal S\to\Delta(\mathcal A),\\
 \sigma&:\mathcal H\to\Delta(\Pi),\\
 \lambda(a\mid h)&=\mathbb E_{\pi\sim\sigma(h)}
       [\,\pi(a\mid S(h))\,].
 \end{aligned}
$$

这是 Elelimy 等人的表示方式：固定状态映射 $S$ 与策略类 $\Pi$ 后，学习规则 $\sigma$ 根据历史给出当前策略的分布。相同的外部行为可能有不同分解，因此讨论学习必须交代所选的表示与策略类。原文固定有限状态表示；下面的工程内部状态写法进一步容纳持续变化的表示。

$$
\begin{aligned}
 A_t&\sim\pi_{Z_t}(\cdot),\\
 (R_{t+1},O_{t+1})&\sim e(\cdot\mid h_t,A_t),\\
 Z_{t+1}&=U(Z_t,A_t,R_{t+1},O_{t+1}).
 \end{aligned}
$$

Z 是完整内部状态，U 是预先指定的更新规则。Z 可包含观测编码、参数、优化器动量、资格迹、回放内容、随机数发生器状态和计数器。为简化公式，这里将内部随机性包含在初始随机状态中，使 U 写为确定性映射。

完整算法记为 $\mathcal A=(\pi,U,\operatorname{Law}(Z_0))$，并附带资源与信息约束。仅复制网络权重不一定复制了同一个智能体；仅给出损失函数也没有说明同一步中哪些量使用旧参数、先更新哪个模块、哪些经验被保留。形式化的 $\sigma(h)$ 可以隐式编码整个学习过程，工程实现则必须把这些状态和时序落实。

**算法：完整学习智能体的交互接口；U 可以包含多个子更新，但顺序属于算法**

1. 初始化内部状态 $Z_0$；环境只按协议初始化一次。
1. 每个真实时间步 $t$：
  1. 用当前 $Z_t$ 形成动作分布；保存实际行为概率。
  1. 选择并执行 $A_t$，收到 $R_{t+1},O_{t+1}$。
  1. 记录奖励、资源消耗、失败与外部干预。
  1. 按规定时序执行 $Z_{t+1}=U(Z_t,A_t,R_{t+1},O_{t+1})$。
  1. 在新状态上选择下一动作，不暗中冻结学习或恢复初始参数。

<a id="lesson-derive"></a>

## 4 · 价值仍然可以定义

现在可以把目标章的条件价值用于完整学习器。给定世界的条件规律、算法及当前内部状态，未来仍含随机后果，却已经规定了收到这些后果时怎样更新。回报可积时，对这段未来取条件期望，就是继续学习的价值。冻结诊断则另行关闭指定更新，评价改变后的行为规则。

$$
V^{\mathcal A,e}_{t,H}(h,z)
 =\mathbb E_{\mathcal A,e}\!\left[
 \sum_{k=0}^{H-1}\gamma^kR_{t+k+1}
 \,\middle|\,h_t=h,Z_t=z\right]
$$

H 是从当前时刻起的评价长度。该价值固定世界的条件规律与完整算法，未来更新 U 仍继续执行。历史和内部状态有时冗余；同时写出是为了区分世界信息与算法状态。

$$
\begin{aligned}
 V_{t,0}(h,z)&=0,\\
 V_{t,H}(h,z)
 &=\sum_a\pi_z(a)\sum_{r,o'}e(r,o'\mid h,a)\\
 &\quad\cdot\bigl[r+\gamma V_{t+1,H-1}
    (h\mathbin{\|}(a,r,o'),\,U(z,a,r,o'))\bigr].
 \end{aligned}
$$

将第一步奖励从和式中拆出，再对当前动作和后果取条件期望，就得到递推。符号 ∥ 表示追加历史。求和写法假设有限或可数后果；连续变量改为积分。这里没有把未来参数冻结，也没有假设观测本身满足 Markov 性。

完整历史在形式上足以携带过去的信息，但长度不断增长。把历史和优化器状态写进价值定义，不会自动提供有限维表示、数据覆盖、低方差估计或有效规划算法。定义可用与问题已解决，是两件不同的事。若模型未知，递推式中的条件分布本身还需要学习。

$$
v_\pi=T_\pi v_\pi,\qquad
 T_{\pi'}v_\pi\ge v_\pi\ \Longrightarrow\ v_{\pi'}\ge v_\pi
$$

在有限、平稳、折扣小于一的 MDP 中，准确评价固定策略后，逐状态满足单步改善条件，仍可用算子的单调性与压缩性证明策略改进。持续控制没有推翻这个定理；它提醒我们，冻结策略不是唯一的比较对象，而带有限资源的整个学习器也不自动满足该定理的条件。

[第一册动态规划的策略改善推导](/zh/continual-rl/foundations/tabular/dynamic-programming/#dp-improvement)以同一 MDP 中两个固定策略为比较对象；上面的学习器递推则把未来更新也放进条件期望。GPI 仍可作为内部算法，但它每次改善当前策略的结论，尚未计入为了学到这个策略所用的数据、探索损失和计算时间。这些量需要在完整运行过程中评价。

<a id="control-objectives"></a>

## 5 · 目标、时间尺度与资源约束

$$
\begin{aligned}
 J_T(\mathcal A)&=\mathbb E\!\left[\sum_{t=0}^{T-1}R_{t+1}\right],\\
 J_\gamma(\mathcal A)&=\mathbb E\!\left[\sum_{t\ge0}\gamma^tR_{t+1}\right],\\
 g(\mathcal A)&=\lim_{T\to\infty}\frac{J_T(\mathcal A)}{T},
 \quad\text{若极限存在}.
 \end{aligned}
$$

有限生命期总回报、从初始时刻折扣的回报、平均奖励率，是三个不同目标。无穷折扣要求 γ<1 且通常假设奖励有界；一般非平稳过程的平均率可能不存在，需要另行选择 liminf、有限窗口或其他明确标准。

迷宫中的 $18.48$ 与 $15.20$ 对应第一行的有限 $J_{48}$；除以 48 得到这段生命期的每步均值，尚未取无限时间极限。从时间零折扣又会使很晚发生的适应按 $\gamma^t$ 降权。第 10 节改为在每个实际历史上比较未来 $H$ 步，再对这些起点等权平均：这换了比较问题，不能用原先的生命期排序直接回答。

$$
\mathcal A\in\mathfrak A_{B,C},\qquad
 |Z_t|_{\rm bytes}\le B,\quad
 \operatorname{cost}(U_t)\le C
$$

资源约束定义可实现的算法类。B 可限制持久记忆，C 可限制每次真实交互允许的计算或延迟。实际研究应给出对应的计量方式，而非只限制网络参数量。

探索会付出低奖励或风险成本；表示与策略更换可能需要停机或校准；规划消耗真实时间。若这些成本会影响任务，应放入奖励、单独约束或并列指标中，并说明选择理由。迷宫中的失败尝试与返回 J 已计入 48 个真实步，额外冻结诊断使用的副本另列预算。换成机器人时，收费重置、恢复过程与等待计算也须按同样原则说明。[平均奖励章](/zh/continual-rl/algorithms/average-reward/)进一步区分长期奖励率、真实时长与有限启动损失。

<a id="lesson-example"></a>

## 6 · 反例一：短期较好，不代表长期较好

在初始时刻选择一次方案，之后不能切换。安全方案每步奖励 1；投资方案第一步奖励 −2，以后每步奖励 2。世界完全已知，没有估计误差，也不需要神经网络。只改变评价时间尺度，就足以改变排序。

$$
J_H(\mathrm{safe})=H,\qquad
 J_H(\mathrm{invest})=-2+2(H-1)=2H-4
$$

H≥1 时，投资减去安全的差为 H−4。因此 H=3 时安全较好，H=4 相同，H=5 起投资较好。这不是算法收敛速度的争议，而是评价目标不同。

| 评价长度 | 安全 | 投资 | 较好方案 |
| --- | --- | --- | --- |
| 3 | 3 | 2 | 安全 |
| 4 | 4 | 4 | 相同 |
| 5 | 5 | 6 | 投资 |

$$
\begin{aligned}
 J_\gamma(\mathrm{invest})-J_\gamma(\mathrm{safe})
 &=-2+\frac{2\gamma}{1-\gamma}-\frac1{1-\gamma}\\
 &=\frac{4\gamma-3}{1-\gamma}.
 \end{aligned}
$$

对 $0\le\gamma<1$，排序在 $\gamma=3/4$ 处反转。平均奖励则分别为 1 与 2；最初的代价在无限时间平均中消失。不同目标确实在回答不同问题。

$$
\begin{aligned}
 p_\theta&=(1+e^{-\theta})^{-1},\\
 J_H(\theta)&=p_\theta J_H(\mathrm{invest})
            +(1-p_\theta)J_H(\mathrm{safe}),\\
 \frac{\partial J_H}{\partial\theta}
 &=p_\theta(1-p_\theta)(H-4).
 \end{aligned}
$$

若策略仅在初始时刻按 p 选择投资，梯度方向也随 H 反转。代码用中心有限差分核验这个解析导数；没有把梯度正确当成某种目标普遍合理的证据。

不同评价长度、折扣与初始随机策略的解析梯度

```python
def investment_returns(horizon, gamma=1.0):
    """Choose once: safe gives 1 forever; invest gives -2, then 2 forever."""
    check_horizon(horizon)
    safe = [1.0] * horizon
    invest = [-2.0] + [2.0] * (horizon - 1) if horizon else []
    return {"safe": discounted(safe, gamma), "invest": discounted(invest, gamma)}


def sigmoid(theta):
    if theta >= 0:
        return 1 / (1 + math.exp(-theta))
    z = math.exp(theta)
    return z / (1 + z)


def investment_objective(theta, horizon):
    p = sigmoid(theta)
    returns = investment_returns(horizon)
    objective = p * returns["invest"] + (1 - p) * returns["safe"]
    gradient = p * (1 - p) * (returns["invest"] - returns["safe"])
    return objective, gradient
```

<a id="control-learning-potential"></a>

## 7 · 反例二：相同当前策略，不同未来学习

![相同初始动作价值的三个学习器，按学习率零、零点一、一执行十步；彩色格是动作，圆点表示奖励。](https://yingwen.io/crl-figures/concept-research-control-learning.svg)

从左向右读实际动作。三者第一步完全相同；新反馈对价值的影响不同，导致切换时刻不同。每个实心绿圆对应一次奖励 1，总数分别为 0、3、9。脚本 research-mechanisms.mjs 按动作、反馈、更新的顺序逐步计算。

迷宫已经表明末次策略与生命期收益可以排序不同。这里进一步把当前估计也设成完全相同，只改变更新速度。考虑奖励刚刚切换的一状态、两动作环境：动作 0 的奖励现在为 0，动作 1 为 1。几个智能体都保存 $Q_0=(1,0.5)$，按最大 Q 贪心行动，并在并列时选动作 0。它们当前的动作分布完全相同，差别只有收到奖励后的更新规则。

$$
Q_{t+1}(A_t)=Q_t(A_t)+
 \alpha[\,R_{t+1}-Q_t(A_t)\,]
$$

这里 Q 估计一步奖励均值，不是把折扣 Q-learning 的后继项漏掉。世界是一状态 bandit，足以隔离学习能力与当前动作的差别。未选择动作的估计保持不变。

| 学习率 | 前十步行为 | 前十步奖励和 |
| --- | --- | --- |
| 0 | 一直选动作 0 | 0 |
| 1 | 先选一次 0，之后都选 1 | 9 |
| 0.1 | 先选七次 0，之后三次选 1 | 3 |

$$
Q_k(0)=(1-\alpha)^k,\qquad
 k_{\rm switch}=\min\{k\ge1:(1-\alpha)^k<0.5\}
$$

只要尚未切换动作，动作 0 的估计按指数衰减。严格小于来自并列时选 0 的规则。α=0 永不切换；α=1 更新一次即可切换；α=0.1 时 0.9 的六次方仍大于 .5，七次方才小于 .5。

如果冻结这些智能体当前的策略，未来都会一直选 0，价值相同且为零。继续运行学习规则时，它们的十步收益却不同，因而比较对象需要包含更新规则。噪声奖励下，大学习率也会更快响应偶然结果；旧情境回来时还会遇到保留与重学的取舍。目标与评价章的带噪迷宫正好提供这层检查，不能把这里的确定性排序直接推广过去。

同一初始化、不同更新规则；fixed_stream_fit 另行展示被动预测而非闭环控制

```python
@dataclass
class GreedyLearner:
    alpha: float
    q: list = field(default_factory=lambda: [1.0, 0.5])
    updates: int = 0

    def __post_init__(self):
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must lie in [0, 1]")
        self.q = list(self.q)

    def action(self):
        # Tie convention matters: choose action 0 when the two values are equal.
        return max(range(2), key=lambda a: self.q[a])

    def observe(self, action, reward):
        self.q[action] += self.alpha * (reward - self.q[action])
        self.updates += 1


def reversal_lifetime(alpha, horizon=10):
    """After a change, reward(0)=0, reward(1)=1. No resets inside a lifetime."""
    check_horizon(horizon)
    learner = GreedyLearner(alpha)
    actions, rewards = [], []
    for _ in range(horizon):
        action = learner.action()
        reward = float(action == 1)
        actions.append(action)
        rewards.append(reward)
        learner.observe(action, reward)
    return {"actions": actions, "rewards": rewards, "total": sum(rewards),
            "final_q": learner.q, "updates": learner.updates}


def fixed_stream_fit(alpha, actions=(0, 1, 0, 1)):
    """Learner is a passive predictor; logged actions do not come from its policy."""
    learner = GreedyLearner(alpha)
    for action in actions:
        learner.observe(action, float(action == 1))
    return {"final_q": learner.q, "next_action": learner.action()}
```

<a id="experiment-learner_control_online"></a>

### 实验：实验 · 同一检查点后继续学习，还是仅保留活动记忆

已有一套能行动的参数后，环境改变时继续更新参数有什么作用？

**环境与可用信息。** 单次生命的走廊世界。大厅可花 1 步、付出 0.05 奖励成本探测当前正确路线，或选择左/右路线。进入路线花 1 步、奖励 −0.02；再走 2 步，中间奖励 −0.02。正确到达得 1 并回大厅；错误到达得 −0.4，随后必须经过 4 个各得 −0.05 的恢复步。观测含阶段、剩余行进/恢复时间与已选路线；只有探测会返回路况线索。600 步起正确路线由左改右，不提供变化通知。

**设置。** 1200 个原始步，Q 与率全零，线索记忆初始为空，ε=0.15，差分 Q 步长 0.1，率步长 0.005。前 400 步两分支完全相同；随后对照保留该检查点的 Q/率并禁写，在线分支继续更新。两者都不重置世界、记忆和 RNG；新探测线索仍更新记忆。冻结与变化时刻由实验预先固定，与总预算无关。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** online_differential_q.py 每个真实步用奖励减率加后继最大 Q 的差分误差同时更新 Q 和率。frozen_parameters.py 在 400 步后仅禁用这两项写入。动作依赖观测阶段与记忆线索；冻结参数不等于冻结活动状态，更不等于清空记忆。

**测量。** 主图是从第 1 步起累加的真实奖励除以原始步数。探测、行进、失败和恢复都计入。另看最近 100 步收益、恢复步数、参数更新次数与记忆写入次数；没有免费评估回合。

```bash
python3 implementations/learner_control/online_differential_q.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/learner_control_online/curves.svg)

横轴：environment_steps。纵轴：完整生命期每原始步的实际奖励。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 400 步时两者全程奖励率均为 0.23686，参数和历史逐项一致；600 步时都为 0.24734。变化后在线分支在 800 步降到 0.20124，1200 步回到 0.22632；冻结分支最终为 0.12787。末尾 100 步奖励率分别为 0.27234、0.02152。冻结分支仍平均写入记忆 60 次，失败不是因为记忆被清空。

**结论边界。** 这是共同检查点之后的有限持续控制比较，不是一般偏离遗憾估计。环境只有可恢复代价，没有不可逆陷阱；不提供外部重置。观察记忆可能过时，非平稳条件下差分 Q 没有在此得到收敛证明；一个冻结检查点也不代表最优固定控制器。

**继续实验。** 检查同 seed 在 400 步前的 Q、率、记忆与 RNG 完全一致。把未来变化从 600 改到 900，400 步前必须不变。再单独冻结记忆写入，明确那是新增消融，不能与冻结参数混为一谈。

[源码](https://yingwen.io/crl-code/implementations/learner_control/online_differential_q.py) · [逐种子记录](https://yingwen.io/crl-code/results/learner_control_online/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/learner_control_online/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/learner_control_online/curves.json)



<a id="experiment-bandit_constant_step"></a>

### 实验：实验 · 相同初始行动分布，不同学习规则

从同样的初始策略出发，仅改变更新规则，会改变整个学习过程的所得奖励吗？

**环境与可用信息。** 无状态三臂 Bernoulli 赌博机，三动作奖励 1 的概率固定为 0.2、0.5、0.8，其他情况奖励 0。无回合终止、无变化信号；每次拉臂计一个真实交互。

**设置。** 1200 次拉臂，初始动作估计和计数为 0。两者都用 ε=0.1 的 ε-greedy；固定步长为 0.1，样本均值对被选动作使用其访问次数的倒数。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** 两个控制器开始时都是均匀行动分布，但一个用常数步长保留近期敏感性，另一个逐渐降低步长。估计改变下一次行动，因此评价对象是行动与更新组成的完整过程。

**测量。** 纵轴为从第一步到当前步的实际奖励总和除以拉臂次数。它包含早期探索成本，不是最后冻结策略的最优动作概率。

```bash
python3 implementations/classic/bandit_constant_step.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-figures/result-bandit_constant_step.svg)

横轴：environment_steps。纵轴：累计平均奖励。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 行为策略实际收到的外部奖励。value 是从第1步至当前 step 的奖励总和除以 step，已经包含这段交互的探索代价。

**step：怎样计时。** step 是实际拉臂次数；每次选择只给所选臂增加一个样本。UCB 的上界项只参与动作选择，不加入环境奖励。梯度 bandit 每次用旧策略概率和收到本次奖励前的基线更新全部偏好，再更新奖励基线。一次环境步不等于只改一个参数。

**怎样汇总。** 每条记录已是累计均值。末点给全程平均奖励，不能再把所有记录点的 value 平均当成全程平均。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** 末点 value × step 可恢复该运行的累计奖励；两个检查点的累计量相减，可恢复这段区间的奖励总和。稀疏检查点不能恢复区间内每一步奖励、访问计数、动作价值或策略概率。各臂真实均值固定为0.2、0.5、0.8；这些曲线不测量变化后的适应。

计算位置：[classic/bandit_sample_average.py](https://yingwen.io/crl-code/implementations/classic/bandit_sample_average.py) · [classic/bandit_constant_step.py](https://yingwen.io/crl-code/implementations/classic/bandit_constant_step.py) · [classic/bandit_ucb.py](https://yingwen.io/crl-code/implementations/classic/bandit_ucb.py) · [classic/bandit_gradient.py](https://yingwen.io/crl-code/implementations/classic/bandit_gradient.py) · [classic/_common.py](https://yingwen.io/crl-code/implementations/classic/_common.py)

</details>

**结果分析。** 1200 次交互后的累计平均奖励为 0.73917；样本均值对照为 0.76283。本平稳任务未显示常数步长优势。保持适应能力有潜在价值，不等于任何静止任务上都应获得更高收益。

**结论边界。** 这是持续控制的最小特例，不包含状态转移、信息动作、不可逆后果或变化后再适应。它只能说明完整学习规则会影响收益，不能作为一般 CRL 或偏离遗憾实验。

**继续实验。** 将两个控制器在同一历史处复制，分别冻结估计和保留更新。冻结不得同时重置已有估计或 RNG。报告从该历史之后的实际收益，而不是重新开始一个更有利的世界。

[源码](https://yingwen.io/crl-code/implementations/classic/bandit_constant_step.py) · [逐种子记录](https://yingwen.io/crl-code/results/bandit_constant_step/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/bandit_constant_step/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/bandit_constant_step/curves.json)

<a id="control-history"></a>

## 8 · 观测价值为什么不一定够用

另一个世界先展示一个公平随机比特，随后只显示空白观测。智能体此时选择 0 或 1，选中先前比特获得奖励 1，否则为 0。两种历史的当前观测都相同，但它们需要不同动作。

$$
\begin{aligned}
 Q(h^{(0)},0)&=1,\quad Q(h^{(0)},1)=0,\\
 Q(h^{(1)},0)&=0,\quad Q(h^{(1)},1)=1.
 \end{aligned}
$$

历史上标表示先前看到的比特。能记住比特的智能体每次都得到 1；完全丢失该信息且只有空白输入的策略，平均只能得到 .5。

固定总选 0 的策略时，仍可定义 $V(O=\mathrm{blank})=0.5$，它是两种历史的混合平均，并非没有数学意义。但该量不能区分历史价值 1 与 0，也不足以为具有记忆的智能体选择动作。混合权重还取决于历史分布。把观测当作状态，隐含地忽略了这层条件信息。

为此可以保存历史、构造 belief，或学习有限容量的 agent state。哪些信息值得保留，取决于后续预测和控制用途。状态增广提供了一种描述，不保证智能体已经找到了合适的压缩方式；agent-state 章讨论其学习问题。

<a id="control-comparators"></a>

## 9 · 可实现比较者与知晓未来的 oracle

控制研究常用参考表现与算法表现之差，衡量遗憾或性能差距。参考对象必须单独定义：它与智能体有相同信息吗？能看未来吗？它在自己的轨迹上行动，还是在智能体实际到达的历史上作局部比较？这些选择决定差距意味着什么。

考虑独立公平比特 $B_t$，第 $t$ 步的奖励是 $\mathbf1\{A_t=B_t\}$。智能体选动作后才看到本步奖励；过去反馈足以推知过去比特，但不能预测新的独立比特。任何只依赖过去的因果算法都满足下式。

$$
\mathbb P(A_t=B_t\mid h_t)=\tfrac12,\qquad
 \mathbb E\!\left[\sum_{t=0}^{T-1}R_{t+1}\right]=T/2
$$

当前动作怎样依赖过去都无济于事，因为本步比特与过去独立。若 oracle 在行动前看到本步比特，它每步得 1，因此差距为 T/2。线性差距不表示实现有错，而表示参考对象拥有算法不可能取得的信息。

这只是无结构快速变化下的反例，并不是说所有动态遗憾目标都不可实现。非平稳 RL 理论通常限制环境总变化量、切换次数或其他规律。Cheung 等人的工作在相应有限 MDP、可达性与变化预算条件下研究滑动窗口、乐观模型及置信区间扩宽。其动态比较标准与这里逐步知晓比特的 oracle 不应混为一谈。

| 参考对象 | 能够说明什么 | 不能直接说明什么 |
| --- | --- | --- |
| 同等预算的因果学习器 | 实际可替换设计之间的表现差异 | 是否达到全知最优 |
| 已知当前模型的最优策略 | 与模型已知基准的差距 | 该基准能否在线估计并及时实现 |
| 知晓未来的 oracle | 环境或信息限制下的理想上界 | 任何算法都能逼近该上界 |
| 从同一起点运行另一算法 | 完整生命周期方案的差别 | 单次真实世界同时观察到两个反事实 |

历史条件价值与所有公平比特序列的精确枚举

```python
def history_values():
    """A fair bit is revealed once, then the current observation becomes blank."""
    frozen_action_zero = {"bit_0_then_blank": 1.0, "bit_1_then_blank": 0.0}
    memory_policy = {"bit_0_then_blank": 1.0, "bit_1_then_blank": 1.0}
    return {"observation": "blank", "frozen_action_zero": frozen_action_zero,
            "observation_mixture_for_action_zero": 0.5,
            "remembering_agent": memory_policy,
            "remembering_mean": 1.0, "memoryless_mean": 0.5}


def unpredictable_comparator(horizon=4, rule=lambda history: 0.5):
    """Each independent fair bit identifies that step's rewarding action.

    rule receives only earlier bits, which binary reward/action feedback reveals.
    It returns the probability of action 1 BEFORE the current bit is revealed.
    """
    check_horizon(horizon)
    if horizon > 12:
        raise ValueError("exact enumeration limited to 12 steps")
    total = 0.0
    for bits in itertools.product((0, 1), repeat=horizon):
        reward = 0.0
        for t, bit in enumerate(bits):
            p_one = rule(bits[:t])
            if not 0 <= p_one <= 1:
                raise ValueError("causal rule must return a probability")
            reward += p_one if bit else 1 - p_one
        total += 2 ** (-horizon) * reward
    return {"causal_expected_reward": total, "clairvoyant_reward": float(horizon),
            "dynamic_regret": horizon - total}
```

<a id="control-deviation"></a>

## 10 · 沿实际历史比较：偏离遗憾

Elelimy 等人提出另一种比较方式。给定学习规则 $\sigma$ 与策略变换 $\phi:\Pi\to\Pi$，把每个历史上将得到的策略改成 $\phi(\pi)$。这得到偏离后的规则 $\phi\circ\sigma$。它可以代表偏向另一动作，或始终采用某个固定策略的 external deviation。偏离作用于策略生成规则，不是复制另一条 rollout 已经发生的动作序列。

$$
\begin{aligned}
 \Delta_{t,H}^{\phi}(h_t)
 &=V^{\phi\circ\sigma,e}_{t,H}(h_t)
   -V^{\sigma,e}_{t,H}(h_t),\\
 \rho_{T,H}^{\phi}
 &=\frac1T\sum_{t=0}^{T-1}\Delta_{t,H}^{\phi}(h_t).
 \end{aligned}
$$

两项从原智能体已产生的同一历史出发，各自向未来继续 H 步。偏离之后，后续动作、世界状态、收到的数据和学习过程仍会共同变化。只有比较起点固定，未来轨迹没有被强制固定。正值表示这个偏离可以改善表现。

这不同于从生命期起点分别运行两个算法再比较总回报：后者通常会到达不同历史。此处比较的是“已经来到这里，此后换一种系统性的行为方式会怎样”。还要声明偏离集合 $\Phi$；对一个很小的集合没有改善空间，不意味着对所有可能学习器都最优。论文把这一形式用于讨论持续改进，不把它等同于全部 CRL 定义。

$$
\begin{aligned}
 W_{t,H}^{\phi}
 &=\prod_{j=t}^{t+H-1}
   \frac{\lambda^\phi(A_j\mid h_j)}
        {\lambda(A_j\mid h_j)},\\
 G_{t,H}&=\sum_{k=0}^{H-1}\gamma^kR_{t+k+1},\\
 \widehat\Delta_{t,H}^{\phi}
 &=(W_{t,H}^{\phi}-1)G_{t,H}.
 \end{aligned}
$$

轨迹重要性权重把行为分布下的段回报转换为偏离分布的期望。分子必须是在该实际前缀上重新计算的偏离规则条件概率；如果规则包含参数更新，不能始终使用一个错误的冻结概率。世界条件概率在两条轨迹密度之比中相消，是该估计成立的关键。

在目标动作受到行为策略覆盖、回报可积时，对给定起始历史有 $\mathbb E[\widehat\Delta_{t,H}^{\phi}\mid h_t]=\Delta_{t,H}^{\phi}$。固定有限 $H$、有界奖励与所有动作概率的统一正下界，是论文有限时域一致性结果中的重要条件。长轨迹的权重乘积仍可能产生巨大方差。估计器需要足够完整的未来窗口；窗口互相重叠，不是独立样本，不能直接拿窗口数当置信区间的独立运行数。

本页符号按原论文式 (1)：偏离回报减去原智能体回报。该文 Algorithm 1 的末行按其变量定义写出了相反的减法顺序；下方实现使用式 (1) 的正号含义，并用解析枚举核验。这个区别影响“正值表示有改进空间”的解释。

<a id="control-irreversible"></a>

## 11 · 反例三：局部没有改善空间，不代表过去没有损害

世界初始健康。安全动作每步奖励 1，并保持健康；冒险动作立即奖励 4，但使世界永久损坏，以后任何动作奖励都为 0。比较安全智能体与第一步冒险的智能体。环境从不重置，后者的失败没有从记录中删掉。

$$
J_T(\mathrm{safe})=T,\qquad
 J_T(\mathrm{risky})=4,\qquad T\ge1
$$

T=100 时，两个从初始世界出发的方案相差 96。这个差距包含第一次动作造成的所有未来后果。

现在改用实际冒险轨迹上的局部偏离比较，令偏离始终选安全动作，取 $\gamma=1,H=6$。初始健康历史的未来六步回报差是 $6-4=2$；从第二个历史开始，世界已经损坏，任何偏离都无法恢复，差为零。因此以下平均局部差随着生命期长度趋近于零。

$$
\rho_{T,6}^{\mathrm{safe}}=\frac{2}{T}
 \longrightarrow0,\qquad
 \frac{J_T(\mathrm{safe})-J_T(\mathrm{risky})}{T}
 \longrightarrow1
$$

左侧问沿实际遭遇的世界还有多少局部改善空间；右侧问最初采用另一方案能否避免破坏。两者不矛盾，但不能互相替代。局部未来窗口始终长六步，世界在记录结束后仍继续，不能把最后几步擅自截短。

本例使用已知模拟器精确计算两种后果，不声称从一条确定性冒险轨迹识别出未尝试的安全反事实。它说明评价标准必须与研究关心的损害相匹配。真实单生命期中，恢复、安全约束和对不可逆区域的事前判断可能比事后局部最优更重要。

为了满足重要性采样的覆盖假设而让真实机器人以正概率尝试每个危险动作，也不是通用方案。数学估计的条件与允许的探索权限可能冲突，必须限定安全动作集合、加入先验或承认该反事实无法从现有数据可靠估计。Single-Life RL 工作研究没有测试期人工重置的适应，但可以使用训练数据和预训练；“单生命期”不等于“没有任何先前知识”。

无重置的破坏过程与两种比较标准；仅作可解析教学反例

```python
def trap_lifetime(risky_first, horizon=10):
    """Safe action yields 1; risky yields 4 ONCE and destroys all future reward."""
    check_horizon(horizon)
    destroyed = False
    rewards, states = [], []
    for t in range(horizon):
        states.append("destroyed" if destroyed else "healthy")
        if destroyed:
            reward = 0.0
        elif risky_first and t == 0:
            reward, destroyed = 4.0, True
        else:
            reward = 1.0
        rewards.append(reward)
    return {"rewards": rewards, "states": states, "total": sum(rewards),
            "destroyed": destroyed}


def trap_diagnostics(lifetime=100, local_horizon=6):
    check_horizon(lifetime, allow_zero=False)
    check_horizon(local_horizon, allow_zero=False)
    risky, safe = trap_lifetime(True, lifetime), trap_lifetime(False, lifetime)
    # Known-model diagnostic from each history actually visited by risky agent.
    # Only the initial healthy history admits a beneficial deviation.
    local_gap = (local_horizon - 4.0) / lifetime
    return {"risky_total": risky["total"], "safe_total": safe["total"],
            "from_initial_world_gap": safe["total"] - risky["total"],
            "mean_local_deviation_gap": local_gap,
            "post_destruction_deviation_gap": 0.0,
            "access": "exact known simulator, not estimated from a real counterfactual"}
```

<a id="control-mechanisms"></a>

## 12 · 从控制问题连接到算法条线

上述例子提供了分解算法的依据。当前行为不佳可能来自估计错误，也可能来自忘记了关键历史、更新过慢、表示失去可塑性、探索没有取得信息，或曾经进入不可恢复的状态。这些原因需要不同机制，不能都归为“非平稳性”。

| 控制中的困难 | 机制改变什么 | 对应的可检验问题 |
| --- | --- | --- |
| 发现并适应变化 | 常数步长、变化检测、上下文与元学习改变更新速度 | 在相同新数据下，响应延迟是否缩短？ |
| 重用旧能力 | 回放、蒸馏、模块和快慢学习保留部分经验或函数 | 旧情境回来时，恢复收益能否抵消保留成本？ |
| 长时间后还能学 | 特征替换、激活设计与优化器处理影响可塑性 | 旧网络与同预算新网络对新问题的学习速度是否不同？ |
| 取得有用信息 | 探索、GVF 与目标构建改变未来证据 | 多获得了什么信息？为此付出什么奖励与风险？ |
| 把知识变成行为 | option、转移模型与规划改变决策单位和前瞻深度 | 相同环境步数与计算预算下，真实行为是否改善？ |
| 记住任务相关历史 | agent state 与预测表征改变条件信息 | 相同观测但不同历史能否产生正确决策？ |

这些机制可以结合，但每增加一个模块，都需要明确它改变了哪个接口与资源项。Anand 与 Precup 的持续预测与控制工作将价值分为较持久与较快速变化的成分，以不同时间尺度保留与适应。它说明价值方法仍是 CRL 的重要算法工具；同时，其具体保证与实验设置不自动覆盖任意大世界、全部表示变化或不可逆风险。

一个有效的控制消融应在相同外部设计协议下替换模块，再观察整个闭环。冻结数据流适合测更新机制；若机制原本旨在改变探索，则还必须允许它改变数据。把这两类实验并列，比只报告一个最终分数更能定位原因。

<a id="lesson-code"></a>

## 13 · 可运行的估计器与精确测试

有限完整窗口的轨迹重要性采样；返回有符号的偏离优势，不截成非负

```python
def validate_distribution(probabilities):
    if not probabilities or any(not math.isfinite(p) or p < 0 for p in probabilities):
        raise ValueError("invalid probability distribution")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-12):
        raise ValueError("probabilities must sum to 1")


def segment_deviation(records, gamma=0.9):
    """One complete H-step segment: (action, reward, behavior_probs, target_probs).

    Probabilities must be conditional on the actual prefix at that step. The
    target is the declared deviation applied at that prefix, not an action label
    copied from another rollout. Positive means deviation return exceeds agent
    return (Eq. 1 convention in Elelimy et al., 2025).
    """
    if not records:
        raise ValueError("empty segment")
    weight, rewards = 1.0, []
    for action, reward, behavior, target in records:
        validate_distribution(behavior)
        validate_distribution(target)
        if len(behavior) != len(target) or not 0 <= action < len(behavior):
            raise ValueError("action dimensions differ")
        if any(q > 0 and b == 0 for b, q in zip(behavior, target)):
            raise ValueError("target action lacks behavior support")
        if behavior[action] == 0 or not math.isfinite(reward):
            raise ValueError("impossible logged action or nonfinite reward")
        weight *= target[action] / behavior[action]
        rewards.append(reward)
    realized_return = discounted(rewards, gamma)
    return (weight - 1) * realized_return


def deviation_estimate(records, horizon=1, gamma=0.9):
    """Use complete windows only. Overlapping windows are NOT independent runs."""
    check_horizon(horizon, allow_zero=False)
    if len(records) < horizon:
        raise ValueError("not enough data for a complete window")
    values = [segment_deviation(records[t:t + horizon], gamma)
              for t in range(len(records) - horizon + 1)]
    return sum(values) / len(values)


def exact_deviation_check(horizon=2, gamma=0.9):
    """Enumerate a one-state bandit: b(1)=.5, deviation(1)=.75, r=a."""
    check_horizon(horizon, allow_zero=False)
    if horizon > 12:
        raise ValueError("exact enumeration limited to 12 steps")
    expectation = 0.0
    for actions in itertools.product((0, 1), repeat=horizon):
        records = [(a, float(a), (0.5, 0.5), (0.25, 0.75)) for a in actions]
        expectation += 0.5 ** horizon * segment_deviation(records, gamma)
    known = discounted([0.75 - 0.5] * horizon, gamma)
    return {"IS_expectation": expectation, "known_difference": known}
```

解析检验使用一状态 bandit：奖励等于动作标签，行为概率 $\lambda(1)=0.5$，偏离概率 $\lambda^\phi(1)=0.75$。取 $H=2,\gamma=0.9$，真实差为 $(0.75-0.5)(1+0.9)=0.475$。程序枚举四个动作序列，按行为概率加权，其重要性采样估计的期望也为 0.475。

下载本章单文件运行；Python 3.10+，仅标准库

```sh
python3 continual_control_lab.py demo
python3 continual_control_lab.py test
```

| 输出项 | 可核验数值 | 检验的内容 |
| --- | --- | --- |
| horizon_reversal | H=3：3 对 2；H=5：5 对 6 | 评价长度改变排序 |
| same_initial_policy | α=0/0.1/1：总奖励 0/3/9 | 行动之后才更新，下一步使用新估计 |
| unpredictable_oracle | T=4：因果期望 2，oracle 为 4 | 未来信息带来的不可实现优势 |
| deviation_IS | 期望 0.475，解析差 0.475 | 重要性比率、完整窗口与符号 |
| irreversible | 起点差 96，局部均值 .02 | 真实无重置轨迹与比较对象 |

29 个单元测试覆盖零长度、零折扣、并列动作、不同实例的状态隔离、支持集缺失、非法概率、不完整窗口、负偏离值保留、解析期望与梯度有限差分。实际日志必须保存动作选择时的行为概率；事后用更新过的策略计算分母会改变估计对象。

脚本中的偏离概率作为每一步已正确计算的条件分布输入；它没有实现任意神经网络学习规则的重演系统。脚本也不复现论文的大规模基准、QWALE 或非平稳乐观规划算法。它验证本章反例与估计公式，使各项权限和条件能够逐个检查。

<a id="lesson-branches"></a>

## 14 · 怎样设计持续控制比较

| 实验层次 | 保持不变 | 允许变化 | 结论范围 |
| --- | --- | --- | --- |
| 固定经验流 | 输入顺序与经验内容 | 学习参数、记忆、表示 | 对这条流的预测与适应能力 |
| 闭环模拟世界 | 环境生成机制与预算 | 行动、访问分布、学习轨迹 | 该世界类别中的完整算法表现 |
| 冻结诊断 | 检查点参数与诊断任务 | 只进行规定的评估动作 | 当前能力，非未来学习能力 |
| 真实单生命期 | 真实部署历史与干预协议 | 允许的因果学习和行动 | 实际经验；未观察后果需额外识别假设 |

独立模拟世界允许从相同初始分布运行不同算法。相同随机种子可以安排共同外生随机数，但不同动作仍可能引向不同状态；不能要求两个闭环算法收到完全相同的后续观测，再声称测到了探索能力。真实世界中通常只能实际走向其中一个分支。日志重加权也不能凭空修复支持缺失、隐藏干预或错误的世界假设。

完整生命周期比较应保留探索损失、切换期间的低谷和永久失败。报告累计与分段收益、响应和恢复时间、旧能力保持、不可恢复失败率，以及记忆、梯度更新数、模型查询和真实时间。某项诊断改善但总体收益不变，是有效的机制证据或限制，而不是必须隐藏的结果。

开发阶段可以用声明的世界和预算选择算法。封存后，测试期间保持算法设计不变，但其内部学习继续运行。不要把“封存算法”误解为冻结所有参数。Mesbahi 等人的 lifetime tuning 立场论文专门指出：设计者遍历完整生命期后再挑超参数，会利用本应属于未来的信息，并可能改变算法排名。使用开发前缀或独立开发世界时，要同时报告其数据成本与测试世界的关系。

不同形式化回答不同问题。Abel 等人用 agent basis 描述隐式策略搜索，并相对于该 basis 定义最优智能体必须持续学习的问题；Elelimy 等人强调学习规则、实际历史与偏离比较；非平稳遗憾理论用受限制的环境变化类研究学习代价；单生命期协议强调部署时不能靠人工重置恢复。这些视角可以互补，但不能省掉条件后合并成一个通用定理。

<a id="research-reset-control-protocol"></a>

## 重置是转移、动作还是外部资源？

Wan、Korenkevych 与 Zhu 的 continuing-task 研究区分无重置、预设重置和智能体控制重置。continuing 指“结束”后的收益仍有意义；持续学习还需判断环境、知识或能力是否不断要求适应。两个维度可以组合，不能凭 wrapper 名称相互替代。

| 协议 | 闭环里发生什么 | 必须计入 |
| --- | --- | --- |
| 无重置 | 状态自然演化，失败后自行恢复 | 恢复时间、不可逆失败与未来数据。 |
| 预设重置 | 环境条件触发回到初始分布 | 重置成本、耗时及之后收益。 |
| 智能体控制重置 | 动作决定是否重置 | 选择权限、频率与真实代价。 |
| 回合式冻结评测 | 评测者反复从指定起点测试 | 该问题与真实训练生命的区别。 |

$$
g_\pi=\frac{\mathbb E[G_{\rm task}-C_{\rm reset}]}{\mathbb E[\tau_{\rm task}+\tau_{\rm reset}]}
$$

再生循环中的奖励率。要求返回同一再生分布、可积收益、有限正期望时长；不可逆环境不能随意套用。分母按原始步或真实时间计算。

A 每轮任务收益 10、用时 10，B 收益 6、用时 3；二者重置成本为 2、用时为 2。只看回合收益 A 更高，计整个循环则 A 的奖励率为 8/12，B 为 4/5，排序反转。这是比较目标与资源不同，不是网络能力差异。

$$
y_t=r_{t+1}-c\,d_{t+1}+\gamma V(s_{t+1}^{\rm actual})
$$

若 reset 是持续过程中的真实转移，actual 是 reset 后实际状态，不能因 d=1 自动删未来项。吸收终止另有边界；重置若耗时不止一步，则显式计时或采用 SMDP。

DeepRL-continuing-tasks 的 experiments/no_resets_mujoco、predefined_resets_mujoco、agent_resets_mujoco 对应三个协议。检查 time-limit 隐含重置和新增动作维度的权限；仅将 done 改成 False，没有定义 reset 后状态与恢复。

**算法：目标、协议与方法的三轴比较**

1. 固定 reset 协议、外部奖励与真实时间单位
1. 比较原始折扣、折扣+中心化、平均奖励控制
1. 匹配采样、调参和失败恢复权限
1. 分开报告全程学习收益、冻结策略率、reset 次数和恢复耗时
1. 诊断 reset 仅在副本上发生，不改变在线生命

MoReFree 则以真实动作返回初始区域，训练无需调用外部 reset，但主要仍以重置后的目标到达能力评测。应同时记录能力获取速度与真实生命的累计收益，两种结果回答不同问题。

<a id="lesson-check"></a>

## 15 · 习题与答案

问题一：两个智能体在当前所有可观测输入上动作分布相同，能否认为未来价值相同？若冻结这些策略且未来信息处理也相同，可以评价同一行为规律。若它们持续更新参数，则还要比较内部状态与更新规则。第 7 节的两个学习率给出总回报 0 与 9 的反例。

问题二：把优化器与历史加入状态后，是否已经解决 CRL？没有。它让条件分布和价值的描述完整，但历史可能无限增长，优化目标仍可能难估计，资源仍有限。构造一个足够且可学习的状态表示，是算法问题，不是记号替换。

问题三：偏离估计值为负，要截成零吗？本章不截。负值表示该偏离的估计结果更差。对有限个偏离再取最大值属于另一个统计选择操作，会引入选择偏差；报告时应区别单个有符号差值与集合上的最大差。

问题四：行为从不选择动作 1，却要评价总选 1 的偏离，可以把分母加一个小常数吗？不能由此获得正确反事实。缺的是支持与数据，数值平滑不提供缺失的环境后果。必须增加合法覆盖、使用有根据的模型假设，或明确无法识别。

问题五：平均局部偏离遗憾趋近于零，能否保证没有造成不可逆损害？不能。破坏反例中，事后世界已经无法改善，但最初选择安全可以避免损害。该结论要求另一种从初始世界比较的目标或显式安全约束。

问题六：在完整测试生命期上挑出的最佳学习率，是智能体具有元学习能力的证据吗？不是。那是外部设计者利用完整数据做选择。在线元学习必须给出因果更新规则，并把其状态、数据和计算加入完整算法，再用封存的协议比较。

研究练习：为你的算法写四句话——比较哪一种完整对象；允许它看到什么；以什么时间尺度评价；与谁比较。再写一个会使这四句话产生不同答案的小世界。先验证反例与公式，再把同一协议用于更复杂的任务。

## 本章的实验设计

分别比较固定策略和完整学习器。保持初始化、环境历史与资源条件一致，记录信息获取和不可逆动作对生命期收益的影响。

设定：使用本章的两步学习率、信息动作和不可恢复状态反例。复制同一合法历史下的智能体与世界状态，分别运行原学习规则、冻结参数和允许的偏离规则。

- 当前动作分布相同不保证未来学习器价值相同。
- 偏离收益减原收益为正表示该偏离更好；不要反用符号。
- 比较器只能使用协议允许的信息、重置与计算。
- 冻结参数仍可保留记忆递推；两个开关分别处理。

对照：固定策略、冻结参数与完整学习器分列；资源和历史匹配的局部偏离；小型可枚举模型与真实交互结果对应

记录：完整生命期累计奖励与每原始时间奖励率；信息动作的短期成本和后续收益；失败后的可恢复性、探索数据与每步计算

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-question)

## 学习与研究衔接

固定策略、记忆递推和完整学习器是不同评价对象。CRL 可定义历史条件价值，但不能不加条件地沿用固定 MDP 的策略排序。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-control) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=control) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=control)

## 从本章进入实践

[策略梯度与控制](https://yingwen.io/zh/continual-rl/code/#practice-policy-control)：优化器确实降低了损失，为什么行动仍可能变差？

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

教材可以先给定目标和技能集合；持续构造还要决定哪些行为值得练习、维护或放弃。谱结构、路径奖励、时间距离和语言先验提供不同候选偏置。先固定候选比较选择与组合，再改变生成器，才能辨认下游收益究竟来自哪一步。

- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)
- [Posterior Sampling for Continuing Environments](https://yingwen.io/zh/continual-rl/research/#recent-cpsrl-continuing-exploration)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

给定模型可研究怎样规划；模型也在学习时，规划会选择性地查询误差，并改变以后的数据。Dreamer、STOMP 和 DRAGO 分别研究想象控制、随机时长行为模型和旧知识保留。新的比较应固定规划查询与总预算，检验哪些后果误差真正改变选择，哪些维护值得继续。

- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [RVI-SAC: Average Reward Off-Policy Deep Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-rvi-sac-average-control)
- [Reward Centering](https://yingwen.io/zh/continual-rl/research/#recent-reward-centering-discounted)
- [An Empirical Study of Deep Reinforcement Learning in Continuing Tasks](https://yingwen.io/zh/continual-rl/research/#recent-continuing-task-deep-study)
- [Posterior Sampling for Continuing Environments](https://yingwen.io/zh/continual-rl/research/#recent-cpsrl-continuing-exploration)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [An Empirical Study of Deep Reinforcement Learning in Continuing Tasks](https://yingwen.io/zh/continual-rl/research/#recent-continuing-task-deep-study)

### RVI-SAC: Average Reward Off-Policy Deep Reinforcement Learning

Yukinari Hisaki, Isao Ono

ICML 2024 · 2024 · 支持方法与理论

#### 研究问题

深度连续控制若最终按单位时间收益评测，训练能否直接采用平均奖励而非有限折扣？

#### 关键机制

RVI-SAC 将相对价值参照项加入 soft critic，以平均奖励的 soft policy improvement 构造 actor，并用额外 reset critic 与可学习成本控制重置频率。完整实现包含双 critic、经验重放、目标网络和温度更新。

#### 证据

论文给出平均奖励最大熵控制推导，并在 MuJoCo 运动任务中比较；公开实现可核对重置转移是否继续 bootstrap。

#### 条件与限制

理论的表格或精确评价条件不自动覆盖所有神经网络训练。最大熵奖励率、外部奖励率与带 reset 成本的奖励率是三个量；不可将有限折扣 reward centering 当作同一算法。

#### 阅读与实验

逐项对应 critic 参照、actor 分布、reset 指示与 reset 后状态；评价保留外部原始奖励、实际时长、重置次数和训练修正目标。

#### 原文与相关入口

- [ICML 2024 正式论文](https://proceedings.mlr.press/v235/hisaki24a.html)：平均奖励 soft improvement、RVI 与自动 reset cost。

#### 作者代码

[作者仓库 README 标明 reference code 与同名原论文。](https://github.com/yhisaki/average-reward-drl)

average_reward_drl/algorithms/rvi_sac.py 及其参照项、固定 reset cost 变体。

### Reward Centering

Abhishek Naik, Yi Wan, Manan Tomar, Richard S. Sutton

RLC 2024 / RLJ · 2024 · 支持方法与理论

#### 研究问题

接近一的折扣为何使共同价值偏移很大，中心化能改善什么、又不能改变什么？

#### 关键机制

从折扣价值的共同偏移与相对价值分解出发，移除奖励参照量；on-policy 可估计行为奖励均值，off-policy 提出 TD 驱动的参照更新。保留小于一的折扣时，中心化没有消除折扣对策略排序的影响。

#### 证据

原文给出理论动机与表格、线性、非线性控制实验，检验折扣及奖励常数平移。深度 continuing-task 后续研究扩大了算法与环境范围。

#### 条件与限制

TD 中心化中的标量在有限折扣下不必精确等于真实奖励率。训练期的联合参照/价值更新与固定常数下的平移恒等式需分别分析；真实终止改变平移条件。

#### 阅读与实验

用单状态常奖励问题解出联合更新固定点，再用多动作问题检查策略排序；同时记录参照量与直接观测的外部奖励率。

#### 原文与相关入口

- [RLC 2024 原文](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_261.pdf)：中心化分解、on/off-policy 区别及收敛讨论。
- [RLJ 论文记录](https://rlj.cs.umass.edu/2024/papers/Paper261.html)：正式题名、作者与会议年份。

### An Empirical Study of Deep Reinforcement Learning in Continuing Tasks

Yi Wan, Dmytro Korenkevych, Zheqing Zhu

arXiv 预印本 · 2025 · 评价与实验协议

#### 研究问题

把环境作为持续的转移过程后，无重置、预设重置和智能体控制重置怎样改变学习难点？

#### 关键机制

构造三类 continuing 协议，将重置后的收益纳入同一条持续过程；对深度控制算法及不同 reward centering 方法进行比较。重置权限属于环境/接口设计，而不是一个可以隐藏的评测便利。

#### 证据

作者公开 MuJoCo 与 Atari testbeds、训练和评价配置。论文报告中心化在多种方法中的收益，同时保留大折扣及无重置恢复困难等限制。

#### 条件与限制

continuing 指非回合式持续交互，不自动意味着环境任意非平稳或无限容量学习。仓库 citation 中的 2024 草稿年与 arXiv 2025 发布年不同；此处按可核验预印本记录，不指定未确认的会议。

#### 阅读与实验

先固定重置转移、成本和时间，再比较目标与算法；分别评价全程学习收益、冻结策略奖励率及失败恢复。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2501.06937)：三类持续协议与深度中心化实验；2025 年 arXiv 首稿。

#### 作者代码

[论文对应 Meta 作者团队的研究仓库，README 明确区分三个 reset 协议。](https://github.com/facebookresearch/DeepRL-continuing-tasks)

testbeds、Pearl 算法、experiments 配置与评测/作图。

### Reset-free Reinforcement Learning with World Models

Zhao Yang, Thomas M. Moerland, Mike Preuss, Aske Plaat, Edward S. Hu

TMLR 2025 · 2025 · 支持方法与理论

#### 研究问题

不能靠外部重置回到起点时，怎样兼顾探索新状态与持续获得对任务有用的经验？

#### 关键机制

MoReFree 在 goal-conditioned world-model 系统中交替练习评测目标、返回初始分布与探索目标；模型内的策略训练也偏向任务相关目标。返回行为通过真实动作实现，调度块结束不会将物理世界 reset。

#### 证据

作者在八个 reset-free 任务中与模型自由及模型式基线比较；公开环境、探索调度和 imagination training 实现。

#### 条件与限制

训练无 reset，但主要评价仍使用可重置的 episodic 测试。已给定初始与目标状态分布、世界模型和 replay 都是资源；这不是任意非平稳 CRL 或真实安全的完整保证。

#### 阅读与实验

把返回成本计入总步数，分别消融数据获取目标与模型内训练目标；检查外部 reward-free 是否仍依赖设计者提供目标示例。

#### 原文与相关入口

- [作者论文 v3](https://arxiv.org/html/2408.09807v3)：训练与评价协议、back-and-forth exploration 与目标分布。
- [TMLR 作者项目页](https://yangzhao-666.github.io/morefree/)：正式发表状态与作者代码链接。

#### 作者代码

[TMLR 作者项目页明确链接的官方实现。](https://github.com/yangzhao-666/MoReFree)

resetfree/env.py、goal_picker_wrapper.py、Dreamer/PEG 与目标条件实验。

### Posterior Sampling for Continuing Environments

Wanqiao Xu, Shi Dong, Benjamin Van Roy

RLC 2024 / RLJ · 2024 · 支持方法与理论

#### 研究问题

没有自然回合边界，后验采样探索应在什么时候更换整条行动假设？

#### 关键机制

CPSRL 以独立随机时钟重采样模型并规划，而不等待真实 reset 或逐状态计数翻倍。几何持续时间把策略试验的未折扣收益与相应折扣规划目标联系起来；改变的是探索承诺的时间尺度。

#### 证据

论文在有限平稳 MDP 条件下分析 Bayesian regret，得到含奖励平均时间 $\tau$ 的 $\widetilde O(\tau S\sqrt{AT})$ 量级，并给出模拟。

#### 条件与限制

定理依赖正确后验、规划与平均时间条件；深网 ensemble 只是一种近似，不直接继承表格界。重采样不重置世界；平稳后验也不会自动遗忘已过时的动力学。

#### 阅读与实验

比较每步换假设、几何时钟与固定时钟，控制同一模型学习预算；在漂移实验中另外定义后验遗忘，避免误用平稳遗憾保证。

#### 原文与相关入口

- [RLC 2024 原文](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_277.pdf)：随机重采样、折扣联系与 Bayesian regret 假设。
- [RLJ 论文记录](https://rlj.cs.umass.edu/2024/papers/Paper277.html)：作者、会议与理论结果。


<a id="chapter-code"></a>

## 下载与运行

Python 3.10+，仅标准库。五组可解析持续控制反例、轨迹重要性采样、精确枚举与 29 个测试；教学模型不是论文 benchmark 或大规模性能复现。

[下载 continual_control_lab.py](https://yingwen.io/zh/continual-rl/download/continual_control_lab.py)

```sh
python3 continual_control_lab.py demo
python3 continual_control_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Elelimy et al. · Rethinking the Foundations for Continual Reinforcement Learning](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_243.pdf)：RLC / RLJ 2025。学习规则、策略偏离、实际历史条件下的比较及重要性采样。正文式 (1) 与 Algorithm 1 末行减法顺序不同；本章按式 (1) 约定正值为偏离优于原智能体。

- [Abel et al. · A Definition of Continual Reinforcement Learning](https://david-abel.github.io/papers/neurips2023_crl.pdf)：NeurIPS 2023。以 agent basis 与隐式搜索形式化持续学习；“永不停止”是相对于所选 basis 的性质，不是参数一直有微小变化。

- [Mesbahi et al. · Position: Lifetime tuning is incompatible with continual reinforcement learning](https://proceedings.mlr.press/v267/mesbahi25a.html)：ICML 2025 立场论文。区分外部设计者利用完整生命期调参与智能体内部的因果适应，讨论前缀调参及评价协议。

- [Cheung, Simchi-Levi & Zhu · Reinforcement Learning for Non-Stationary MDPs: The Blessing of (More) Optimism](https://proceedings.mlr.press/v119/cheung20a.html)：ICML 2020。变化预算、滑动窗口、乐观模型与置信区间扩宽。动态遗憾的参考对象、环境条件和信息假设须一起阅读。

- [Chen et al. · You Only Live Once: Single-Life Reinforcement Learning](https://arxiv.org/abs/2210.08863)：NeurIPS 2022。测试期无人工重置的适应与恢复，可使用先前数据；不要把此部署协议等同于没有预训练。

- [Chen et al. · Single-Life RL 作者代码](https://github.com/anniesch/single-life-rl)：QWALE 的原始实验工程，包括训练、智能体、环境与先前数据接口。代码的预训练和测试权限与本章解析反例不同。

- [Anand & Precup · Prediction and Control in Continual Reinforcement Learning](https://papers.neurips.cc/paper_files/paper/2023/file/c94bbbef466ab1b2cfa100e41413b3a8-Paper-Conference.pdf)：NeurIPS 2023。以持久与暂态价值成分连接保留和适应；是具体价值学习方法，不主张冻结策略价值已经完整评价所有学习器。

- [Bowling et al. · Settling the Reward Hypothesis](https://david-abel.github.io/papers/icml2023_settling_the_rh.pdf)：ICML 2023。历史上的偏好与奖励表示的公理条件。用于理解目标的来源；详细推导见奖励假设与设计章。

- [Sutton & Barto · Reinforcement Learning, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：第 3–8 章建立 MDP、动态规划、MC、TD 与规划；第 9–13 章把这些更新扩展到函数逼近、资格迹和策略梯度。按本章的问题找相应章节，不必从头重读。

- [ICML 2024 正式论文](https://proceedings.mlr.press/v235/hisaki24a.html)：平均奖励 soft improvement、RVI 与自动 reset cost。

- [RVI-SAC: Average Reward Off-Policy Deep Reinforcement Learning · 作者实现](https://github.com/yhisaki/average-reward-drl)：average_reward_drl/algorithms/rvi_sac.py 及其参照项、固定 reset cost 变体。 作者仓库 README 标明 reference code 与同名原论文。

- [RLC 2024 原文](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_261.pdf)：中心化分解、on/off-policy 区别及收敛讨论。

- [RLJ 论文记录](https://rlj.cs.umass.edu/2024/papers/Paper261.html)：正式题名、作者与会议年份。

- [作者论文](https://arxiv.org/abs/2501.06937)：三类持续协议与深度中心化实验；2025 年 arXiv 首稿。

- [An Empirical Study of Deep Reinforcement Learning in Continuing Tasks · 作者实现](https://github.com/facebookresearch/DeepRL-continuing-tasks)：testbeds、Pearl 算法、experiments 配置与评测/作图。 论文对应 Meta 作者团队的研究仓库，README 明确区分三个 reset 协议。

- [作者论文 v3](https://arxiv.org/html/2408.09807v3)：训练与评价协议、back-and-forth exploration 与目标分布。

- [TMLR 作者项目页](https://yangzhao-666.github.io/morefree/)：正式发表状态与作者代码链接。

- [Reset-free Reinforcement Learning with World Models · 作者实现](https://github.com/yangzhao-666/MoReFree)：resetfree/env.py、goal_picker_wrapper.py、Dreamer/PEG 与目标条件实验。 TMLR 作者项目页明确链接的官方实现。

- [RLC 2024 原文](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_277.pdf)：随机重采样、折扣联系与 Bayesian regret 假设。

- [RLJ 论文记录](https://rlj.cs.umass.edu/2024/papers/Paper277.html)：作者、会议与理论结果。
