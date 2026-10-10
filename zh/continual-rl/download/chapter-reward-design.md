# 奖励假设与奖励设计

什么样的目标可以表示为奖励？智能体学会最大化奖励，是否就实现了设计者的意图？

## 本章内容

- 区分偏好、评价准则、奖励信号和辅助学习信号。
- 说明奖励表示定理的条件，并构造 Markov 奖励不能表达的排序。
- 推导势函数塑形、偏好学习和有限轨迹 MaxEnt IRL 的更新。
- 设计可以区分目标错误、奖励模型错误与优化错误的实验。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [强化学习问题的形式化：交互、目标与持续学习](https://yingwen.io/zh/continual-rl/foundations/objectives/)：先确定比较行为与学习过程的准则，再问奖励怎样表达这一目标。


### 历史与策略

历史包含已经发生的观测和动作。策略或完整学习器诱导历史上的概率分布。设计者也可能观察智能体看不到的变量。

### 回报

本章用有限时域折扣回报推导。平均奖励和风险准则会另行说明。

$$
G_T=\sum_{t=0}^{T-1}\gamma^t R_{t+1}
$$

### 概率模型与梯度

知道条件概率、期望与链式法则即可。对数配分函数的梯度会在正文推导。

<a id="problem-definition"></a>

## 本章的问题定义

设计者偏好与智能体收到的标量信号可能不同；需要分别研究目标能否表达、奖励能否推断、有限学习器能否利用它。

### 给定条件与符号

- 评价者可见的结果、偏好或示范数据，以及允许的奖励输入。
- 奖励模型类、反馈预算、训练环境和独立评价准则。

### 需要求解的对象

符合声明偏好或有明确误差边界的奖励机制；辅助奖励还需通过外部评价证明其学习用途。

### 信息与数据权限

$\bar H$ 表示评价者可见的完整结果历史；智能体只见其自身历史。奖励参数 $\psi$ 的更新只能使用已获得的反馈。

$$
A\succeq B\ \Longleftrightarrow\ \mathbb E_A[U(\bar H)]\ge\mathbb E_B[U(\bar H)]
$$

$A,B$ 是结果历史的概率分布，$\succeq$ 是设计者偏好，$U$ 是其效用表示。进一步寻找可累加奖励，需要额外时间一致性和输入表达条件；偏好交叉熵、IRL似然和辅助奖励外层回报分别是不同的求解目标。

### 成立条件与解的含义

- 期望效用和逐步奖励表示各自要求相应公理，不能将任意偏好直接当成Markov奖励。
- 偏好或示范推断需明确评价噪声、片段长度和行为模型；塑形需明确折扣与终点势。

判断准则：在声明的偏好域检验排序一致性；对奖励推断报告未用于拟合的反馈误差与新策略结果；塑形以望远镜边界项检验目标保持。

### 适用边界

- 比较训练标签拟合良好不证明新行为符合偏好。
- 不将奖励假说的表示结论解释为任意有限智能体都能成功优化。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)：目标章给定比较准则，本章检查该准则到奖励接口的表示与推断。

- 组合不同学习问题 · [元学习与学习规则的适应](https://yingwen.io/zh/continual-rl/algorithms/meta/)：学习辅助奖励时，奖励参数通过内层更新影响外层表现，成为元学习问题。

- 组合不同学习问题 · [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)：本章区分外部评价与学习信号，探索可提供新奇或信息信号；二者组合不必改变最终外部目标，辅助信号的用途须单独验证。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

奖励错误、奖励模型错误和控制失败可以产生相同的低外部收益。

### 本章的核心思路

为每条信号追踪其来源、允许输入和被评价的对象；分别验证表示、估计和行为。

1. [检验表达能力](#lesson-expressivity)：因为局部奖励未必区分历史顺序，先检查偏好公理与Markov反例，必要时增广状态。

2. [选择保持目标或推断目标的机制](#lesson-derive)：若目标已知而信号稀疏，势差塑形处理反馈；若只有比较标签，偏好模型处理目标信息，二者保证不同。

3. [将学习奖励接回外部评价](#lesson-intrinsic)：因为内部奖金可以诱导投机，沿更新求外层梯度并单独评估外部结果与反馈成本。

结论与条件：势函数目标保持需满足所写折扣、边界与时序；偏好/IRL拟合只在观测模型及覆盖范围内有解释。

### 相关方法改变了什么

- 势函数塑形：在条件下保持原策略排序，改善信号形态。

- 偏好学习与IRL：分别由结果比较与示范行为推断奖励，需要不同的观测模型。

- 最优内部奖励：选择信号使受限学习器提高外部评价，一般无策略不变性保证。


<a id="lesson-setting"></a>

## 1. 先确定谁的目标、谁的观测、谁的奖励

考虑一个送货机器人。设计者希望它按时、安全地完成送货。机器人收到的奖励 $R_{t+1}$ 却可能只由距离变化、送达计数和碰撞惩罚组成。希望实现的目标与实际奖励不是同一个对象。奖励设计研究二者怎样关联。控制学习则研究给定这些信号之后怎样行动。

| 对象 | 本章记号 | 送货例子 | 需要检查 |
| --- | --- | --- | --- |
| 设计者偏好 | ≽ | 安全完成比冒险抢时更好 | 对哪些结果、概率和时间作比较？ |
| 评价准则 | U 或 J | 完整生命期的交付量与事故限制 | 评价者能看到哪些变量？ |
| 奖励机制 | r 或 rψ | 程序或奖励模型输出的标量 | 能否被行为操纵？版本是否变化？ |
| 学习用信号 | r̃ | 奖励加塑形或探索奖金 | 它是否保持原目标？ |

把送货问题缩到一件包裹和三个动作。机器人从 S 接近门口 A，之后可以送达 G，也可以留在 A 等待；每个动作恰好耗时一步。送达立即结束，第三个动作后仍未送达也真正结束。状态必须包含时钟 $(t,x)$：$(2,A)$ 还可送达，$(3,A)$ 已超时。我们约定评价只关心折扣后的送达：送达奖励 1，其余 0，$\gamma=0.9$。本例用这项评价表达截止前尽早送达的目标。

![包裹从 S 到门口 A，随后可送达或等待；时钟达到3便真正终止。图展开全部三条可行轨迹，并标出势值。](https://yingwen.io/crl-figures/reward-design-walkthrough-task.svg)

沿箭头可列出全部结果：第2步送达、第3步送达、等待至超时。双圈是终端，红色终端表示未送达。门口势在非终端为3、截止时为0；因此物理位置相同，带时钟的状态及边界条件不同。原创确定性算例；代码精确枚举这些有限路径。

| 完整轨迹 | 逐步原奖励 | 原任务折扣回报 |
| --- | --- | --- |
| S → A → G | [0, 1] | 0.9 |
| S → A → A → G | [0, 0, 1] | 0.81 |
| S → A → A → A（超时） | [0, 0, 0] | 0 |

现在给每次到达或停留门口的动作额外奖金 3，包括最后一次超时动作。这个传感器容易实现，却只检测位置。三条路径的训练回报变成 $3+0.9=3.9$、$3+0.9\times3+0.9^2=6.51$、$3+0.9\times3+0.9^2\times3=8.13$。知道全部转移的精确规划器会选择连续等待，实际送达数为0。它已找到代理奖励下的最优行为；继续提高求解精度无法修复这个规格错误。后面沿同一任务检查：怎样增加逐步信号，又保留尽早送达的排序？

$$
h_t=(o_0,a_0,\ldots,o_t),\qquad R_{t+1}=r_\psi(\bar h_{t+1}),\qquad J_U(L)=\mathbb E_{P_L}[U(\bar H)]
$$

L 是包含学习规则和记忆的智能体。带横线的历史是评价者可见的信息。它可以不同于智能体历史。$P_L$ 是交互产生的分布。

运行时仍可使用“智能体—世界”的二元接口。外部设计者决定奖励通道和评价协议。在这个三方描述中，必须记录设计者还提供了哪些任务标签、重置和人工反馈。奖励学习不会自动消除这些外部工作。

<a id="lesson-hypotheses"></a>

## 2. 三个不同命题：可表达、足以驱动、容易学会

| 命题 | 实际问题 | 不能由它推出 |
| --- | --- | --- |
| 奖励假设的表示问题 | 一组偏好能否由期望累积标量奖励表示？ | 给定观察上的一个简单奖励一定存在。 |
| Reward is Enough 的研究假说 | 追求奖励是否能够促成广泛的智能能力？ | 某个当前算法已有这些能力；给任何奖励都能学成。 |
| 奖励设计与可学习性 | 有限经验、有限计算下，哪些信号能诱导所需行为？ | 最优策略相同，就有相同学习速度或安全性。 |

Silver、Singh、Precup 与 Sutton 的 Reward is Enough 提出一个关于智能能力来源的研究假说。Bowling、Martin、Abel 与 Dabney 的 Settling the Reward Hypothesis 则研究偏好的表示条件。它们回答不同的问题。本章不把前者写成定理，也不把后者解释为“任意目标都有一个简单奖励”。

一个最优性结果还可能隐藏极大的计算量、探索成本或表示需求。持续强化学习尤其需要把这些成本放回问题中。能写出奖励与能在一次生命期内学会行为，是两层结论。

<a id="lesson-representation"></a>

## 3. 从结果偏好到累积奖励：条件在哪里

先考虑有限结果集合及其概率混合。完备性允许比较任意两个彩票。传递性排除循环偏好。连续性排除某些无限优先级。独立性要求与同一第三种彩票作相同比例混合时，原有排序保持。满足相应条件，才可以用期望效用表示偏好。

$$
A\succeq B\iff \mathbb E_A[u(H)]\geq\mathbb E_B[u(H)]
$$

这是对结果分布的表示。它没有要求 u 能逐步相加，也没有要求智能体能计算它。风险厌恶可以通过结果效用的形状表达；不能简单说标量效用只允许风险中性。

要把效用分解成局部奖励，还需要时间上的一致性。在 Bowling 等人的有限历史与有限支持彩票框架中，四个期望效用公理加上 temporal γ-indifference，刻画了逐转移奖励与转移依赖折扣的表示。其关键递推如下。

$$
u(\epsilon)=0,\qquad u(x\cdot h)=r(x)+\gamma(x)u(h),\qquad 0\leq\gamma(x)\leq1
$$

x 是原文中的一个转移符号，x·h 表示把它接在历史之前。不是预先假定某个物理状态已经足够。前缀对两个后续历史的效用差只能按同一 γ(x) 缩放。

这条限制可直接检验：如果收到同一个前缀后，对两种未来的相对偏好发生了无法由同一非负比例解释的变化，那么当前表示不满足这条递推。结论是该表示或偏好假设需要改变，而不是宣布这个目标“不理性”。对无限历史、平均奖励和有限内存的扩展还需要各自的条件。

<a id="lesson-expressivity"></a>

## 4. 一个两步反例：Markov 奖励依赖状态的选择

![钥匙和门的两种动作顺序，以及未拿钥匙、已有钥匙、完成三个奖励记忆状态。](https://yingwen.io/crl-figures/concept-research-reward-memory.svg)

两个顺序都有一次 K 和一次 D。只数动作时无法分出先后；增加下方的记忆状态后，开门能否得奖取决于此前有没有拿钥匙。图中物理位置不变，变化的是奖励自动机状态。数值与正文两步例子一致。

世界只有一个状态 s。两个动作记为 K 和 D，每次都回到 s。时域为两步，γ=1。设计者要求先 K 后 D。对任何固定的 r(s,a,s)，KD 与 DK 的回报都是 r(K)+r(D)。因此它们必然并列，不能实现严格偏好 KD≻DK。这里的问题不是网络不够大，而是奖励的输入丢失了顺序。

$$
G_r(\tau)=\sum_x n_\tau(x)r(x),\qquad n_{KD}=n_{DK}\ \Longrightarrow\ G_r(KD)=G_r(DK)
$$

nτ 是转移计数向量。任何仅依赖这些计数的线性奖励，都不能区分计数相同的轨迹。折扣、时钟或更大的状态会改变这个条件，必须显式声明。

引入自动机状态 q∈{start,has-key,done}。K 把 start 变成 has-key。此后第一次 D 才获得 1 并进入 done。在增广状态 (s,q) 上，奖励可以是 Markov 的。Reward Machines 将这类奖励结构显式表示，并用于学习。

可执行的顺序反例与三状态奖励自动机。代码把 K/D 命名为 key/door。

```python
def additive_return(events, weights):
    return sum(weights.get(event, 0.0) for event in events)


def ordered_goal(events):
    """Finite-state reward: collect key before door; pay once."""
    mode, reward = "start", 0
    for event in events:
        if mode == "start" and event == "key":
            mode = "has-key"
        elif mode == "has-key" and event == "door":
            mode, reward = "done", reward + 1
    return reward
```

增加记忆修复的是这个信息缺失。它不证明任意偏好都有有限状态表示，也不自动修复彩票偏好违反独立性的问题。Abel 等人的 Markov 奖励表达能力研究应当放在这一层理解。

<a id="research-openmind-ungrounded-alignment"></a>

## 奖励里的“那个对象”是谁：未接地的目标与感知编码

写出“拾起垃圾得 1 分”已经确定了一部分偏好，却还没有告诉系统哪段传感器数据表示垃圾。如果相机输入被置换、换了感知模态，或内部表示在学习中重排，奖励公式中的变量可能失去原来的指代。这里的问题先发生在概念与感知的对应关系上；一个优化完全正确的控制器也可能在错误事件上得到奖励。

Pickett、Nain、Modayil 与 Jones 的 The Ungrounded Alignment Problem（2024 预印本，ICDL 2025）用一个刻意简化的问题隔离这层困难：系统预先知道要识别的字母序列，却不知道字母将以何种图像编码出现，训练时没有图像到字母的标签。作者固定英语字符的 bigram 关系表，让共享编码器输出的相邻字符分布与该表给出的关系相吻合。

$$
e_t=E_\theta(x_t),\qquad d_{t+1}(y)=\sum_{x\in\Sigma}B(y\mid x)e_t(x)
$$

这里用统一记号重写关系传播：E 把未知感知编码映成字母概率，B 是预先给定且冻结的字符转移知识，d 是由前一字符推得的下一字符分布。学习再用批内对比损失约束 d 与下一图像的编码；不是由奖励大小直接学习字母名称。

原文在置换 EMNIST 字符序列上的触发词 fnord 检测达到 99.88%，但在 CIFAR26 编码上为 85.49% ± 1.59%。不能只用前者概括跨感知编码的结果。实验预先提供字符关系，进行 64 次随机重启以缓解局部最优，并用触发词出现概率设定阈值。它是一项接地问题的受控解法，不能据此宣称已解决人类价值对齐，也不是固定预算单生命期流式 RL 的实证。

接回奖励设计，可以把任务写成两个待检查的映射：先从经验识别事件，再由事件计算奖励。下面的分解是教学用的接口，不是该论文提出的新 RL 算法。奖励权重可以保持不变，而事件识别器需要随传感器变化重新接地；反过来，识别正确而权重改变，才是在改变评价偏好。

$$
Z_{t+1}=g_{\eta_t}(H_{t+1}),\qquad R_{t+1}=\rho_\psi(Z_{t+1})
$$

g 负责“发生了什么”的识别，ρ 负责“这个事件值多少”的评分。若二者都可学习，必须保存其版本和各自监督来源。正确识别一个事件不自动证明给它的分数合适。

思考：让感知映射变化而事件关系固定，再让事件关系变化而感知固定，会分别破坏上面哪一项假设？如果两类事件在所有已知关系中完全对称，仅靠无标签关系匹配能否唯一确定它们的名字？需要加入什么经验或先验才能打破对称？

<a id="lesson-derive"></a>

## 5. 势函数塑形与边界条件

设原目标是折扣奖励，γ∈[0,1)。选择一个势函数 Φ。它可用来提示哪些状态看起来更有希望。不能直接把 Φ(s′) 加到奖励上。正确的差分塑形项同时减去当前势。

$$
F_t=\gamma\Phi_{t+1}(s_{t+1})-\Phi_t(s_t),\qquad \widetilde R_{t+1}=R_{t+1}+F_t
$$

先允许势显式依赖时间。固定势是 Φt=Φ 的特例。相邻项使用相同边界上的势值，才能消去。

$$
\begin{aligned}\widetilde G_T&=G_T+\sum_{t=0}^{T-1}\gamma^{t+1}\Phi_{t+1}(s_{t+1})-\sum_{t=0}^{T-1}\gamma^t\Phi_t(s_t)\\&=G_T-\Phi_0(s_0)+\gamma^T\Phi_T(s_T).\end{aligned}
$$

中间每一项都出现一次正号和一次负号。只剩起点与终点。

无限折扣任务中，若势一致有界，末项趋于零。从同一起点出发，各策略的价值减去同一个常数，最优动作排序保持。有限回合中，把真正终端的势设为零也得到这个结论。若有限截断保留了不同的终点势，就不能忽略边界项。

$$
\widetilde V^\pi(s)=V^\pi(s)-\Phi(s),\qquad \widetilde Q^\pi(s,a)=Q^\pi(s,a)-\Phi(s)
$$

这是固定势、无限折扣、边界消失时的关系。它说明目标排序保持，不保证函数逼近或有限样本训练得到相同策略。

逐轨迹验证望远镜求和。测试同时覆盖 γ=0、有限 γ=1 和非零终点势。

```python
def discounted_sum(rewards, gamma):
    return sum(gamma ** t * r for t, r in enumerate(rewards))


def shape(rewards, potentials, gamma):
    """potentials[t] is Phi_t(s_t); retain the final boundary potential."""
    if len(potentials) != len(rewards) + 1:
        raise ValueError("One potential is required at each boundary.")
    return [r + gamma * potentials[t + 1] - potentials[t]
            for t, r in enumerate(rewards)]


def shaping_residual(rewards, potentials, gamma):
    lhs = discounted_sum(shape(rewards, potentials, gamma), gamma)
    rhs = (discounted_sum(rewards, gamma) - potentials[0]
           + gamma ** len(rewards) * potentials[-1])
    return lhs - rhs
```

回到三步送货任务，取 $\Phi(0,S)=1$，所有非终端门口状态的势为3，送达与截止状态的势都为0。尽早送达的塑形奖励为 $[0.9\times3-1,\;1-3]=[1.7,-2]$，其折扣回报是 $1.7+0.9\times(-2)=-0.1$。延迟送达得到 $[1.7,-0.3,-2]$，回报 $-0.19$；超时得到 $[1.7,-0.3,-3]$，回报 $-1$。每条完整路径恰好比原回报少起点势1，所以尽早送达仍最好。

$$
(G_{\rm early},G_{\rm late},G_{\rm timeout})=(0.9,0.81,0),\qquad (\widetilde G_{\rm early},\widetilde G_{\rm late},\widetilde G_{\rm timeout})=(-0.1,-0.19,-1).
$$

负的塑形回报仍可对应正确行动：控制比较从同一状态出发的各动作，公共偏移不改变差值。这里用已知模型精确求解，尚未测量任何学习速度。

这个结论也揭示塑形解决不了什么。若给刚才的错误代理奖励再加同一个合法势差，三个代理回报都减1，超时仍以7.13胜出。势差保持的是所提供奖励的排序；它不会把“门口停留”自动改写成“完成送达”。Ng、Harada 与 Russell 的原论文给出固定势的策略不变性及其条件；上面的带时钟有限任务通过逐轨迹边界恒等式直接核对。

平均奖励下可用未折扣势差。若势有界，T 步平均塑形奖励为 [Φ(sT)−Φ(s0)]/T，极限为零。这里保持的是长期奖励率；暂态收益与差分价值可以改变。

<a id="experiment-potential_shaping"></a>

### 实验：实验 · 势差奖励的保证与短预算失败

正确的势函数塑形为什么仍可能使有限预算结果更差？

**环境与可用信息。** 六格确定性链：从 0 开始，左右动作，左边界停留。到位置 5 得 1 并终止，其他转移得 −0.01。观测就是位置。真实终止后从 0 开始；折扣为 0.95，任务没有中途变化。

**设置。** 各运行 1200 个真实环境步；Q 全零，学习率 0.2，ε-greedy 的 ε=0.1，并列最优训练动作均匀选择。两者只有训练奖励不同。势函数为非终止状态的 −(5−s)/5，终点势为 0。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** shape 使用原奖励加 0.95 倍后继势减当前势；真实终点势为 0。Q-learning 在塑形奖励上训练，chain_score 始终在原奖励上评分。

**测量。** 比较原任务冻结贪心折扣回报。训练塑形奖励会改变数值尺度，不能把它和未塑形回报直接画成同一性能指标。

```bash
python3 implementations/continual/potential_shaping.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/potential_shaping/curves.svg)

横轴：environment_steps。纵轴：原始奖励贪心策略折扣回报。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 第 1200 步塑形均值为 0.58193，未塑形基线为 0.77741；塑形标准差约 0.43711。现有结果不是正向成功案例，应保留为“策略集合保持不等于有限样本加速”的实例。

**结论边界。** 两者从相同的零 Q 开始，并未采用理论上对应的势移初值。探索路径和访问频率会因此改变；这是平稳短链，不是自动设计奖励或跨任务泛化。

**继续实验。** 逐轨迹重算塑形回报与原回报之差，核对终点项。随后分别改变势的方向、尺度和初始化，判断哪一项改变目标，哪一项只改变学习过程。

[源码](https://yingwen.io/crl-code/implementations/continual/potential_shaping.py) · [逐种子记录](https://yingwen.io/crl-code/results/potential_shaping/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/potential_shaping/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/potential_shaping/curves.json)

<a id="lesson-example"></a>

## 6. 三个会改变结论的细节

先只改变送货代码中的一项约定。若程序在超时后停止，却仍把 $(3,A)$ 当作势为3的普通门口状态，最后一步塑形奖励就从 $-3$ 变成 $-0.3$。超时路径留下 $0.9^3\times3=2.187$ 的终点项，训练回报从 $-1$ 变为1.187，超过尽早送达的 $-0.1$。时间上限本身没有违背塑形恒等式；错误来自丢掉了恒等式中仍然存在的末项。

![三步等待轨迹的累计折扣塑形回报：前两步重合，正确终点势的蓝线最后降至负1，保留门口势的红虚线终至1.187。](https://yingwen.io/crl-figures/reward-design-walkthrough-boundary.svg)

横轴是已执行动作数；纵轴是该前缀已经收到的累计折扣塑形奖励。两条线只在最后一笔奖励处分开，差额2.187恰为未消去的终点项。本任务的三步截止是规定的真正终止，后面没有补齐动作或隐藏奖励。

另一个错误是沿用未折扣势差 $\Phi(s_{t+1})-\Phi(s_t)$，但仍以 $\gamma=0.9$ 累加回报，并正确处理终端势。此时尽早送达得 $[2,-2]$，回报0.2；延迟送达得 $[2,0,-2]$，回报0.38。等在高势的门口一拍，使负奖励更晚出现，规划器于是推迟送达，原任务回报降至0.81。折扣不仅出现在回报外层，还须进入塑形差分本身。

| 交给规划器的信号 | 早送达 / 晚送达 / 超时：训练回报 | 真正执行 | 原任务回报 |
| --- | --- | --- | --- |
| 原任务奖励 | 0.9 / 0.81 / 0 | 第2步送达 | 0.9 |
| 加门口占用奖金 | 3.9 / 6.51 / 8.13 | 等待至超时 | 0 |
| 正确势差 | −0.1 / −0.19 / −1 | 第2步送达 | 0.9 |
| 遗漏超时边界 | −0.1 / −0.19 / 1.187 | 等待至超时 | 0 |
| 遗漏塑形折扣 | 0.2 / 0.38 / −0.43 | 第3步送达 | 0.81 |

![五种奖励信号下，晚送达与超时相对早送达的训练回报差；各行同时列出所选行为和其原任务回报。](https://yingwen.io/crl-figures/reward-design-walkthrough-decisions.svg)

所有行共用差值坐标：0线代表尽早送达，向右的路径才比它有更高训练回报。正确塑形与原任务的两个差值完全相同。每行执行一条完整回合，真实动作数分别为2、3、2、3、3；这是一组已知模型的精确选择，未把它当作匹配训练预算的性能实验。

若只是收集到第1步就截断数据，包裹其实还可送达，便不能把门口变成真正终端。继续立即送达的原尾值是 $V(1,A)=1$，塑形尾值是 $\widetilde V(1,A)=1-3=-2$。完整目标应为 $1.7+0.9\times(-2)=-0.1$；只保留前缀1.7会改变评价。上节的零终端势处理适用于本来就规定结束的任务；采样截止需要保留相应尾值。其他常见边界也可以用同一个检查方法辨认：

| 操作 | 小例子 | 正确结论 |
| --- | --- | --- |
| 每步加常数 | 本例 γ=1。短路线奖励 [1]；长路线 [0,0,0.9]。各步加 0.2 后，1.2<1.5。 | 可变时长任务中，常数奖励可能改变最优策略。 |
| 终点势不归零 | 一步奖励 2 与 1，终点势分别 0 与 10；γ=0.9。塑形后 2<10。 | 必须保留边界项。time-limit truncation 不是天然的任务终止。 |
| 在线改变势，却混用时间索引 | 上一转移用了 $Φ_0(s_1)=2$；下一转移减去 $Φ_1(s_1)=5$。 | 中间项不能相消。动态塑形要明确生成和保存势值的时序。 |

正比例缩放奖励在相同回报准则下保持策略排序，但可能改变步长、熵温度和裁剪阈值的有效尺度。把奖励归一化称为“无害预处理”是不充分的。若归一化统计随行为变化，还需说明实际优化的目标。

稠密奖励也不天然优于稀疏奖励。距离奖金可能让智能体反复接近而不完成任务。先证明或检查目标是否改变，再测是否更快学会。训练回报上升和原任务成功率上升需要分别报告。

<a id="lesson-preference"></a>

## 7. 从比较中学习奖励：Bradley–Terry 模型

给评价者看两段轨迹 A、B。令 y=1 表示偏好 A，y=0 表示偏好 B。y=0.5 可表示平局的软标签。先用奖励模型对每段求和，再用两个分数之差预测选择概率。

$$
S_\psi(A)=\sum_{t\in A}r_\psi(o_t,a_t,o_{t+1}),\quad z=\frac{S_\psi(A)-S_\psi(B)}{\tau},\quad p_\psi=\frac1{1+e^{-z}}
$$

τ>0 是选择噪声的温度。若奖励尺度和温度同时未知，两者不能仅凭这个似然分别确定。片段求和、折扣和片段长度都属于模型假设。

$$
\ell=-y\log p_\psi-(1-y)\log(1-p_\psi),\qquad \nabla_\psi\ell=(p_\psi-y)\frac{\nabla_\psi S_\psi(A)-\nabla_\psi S_\psi(B)}{\tau}
$$

先对 logit 求导得到 p−y，再沿两个片段分数反向传播。若 A 已被偏好，但预测 p 很低，更新应提高 A 相对 B 的分数。

一维线性奖励的稳定交叉熵与梯度。测试用有限差分检查符号，并覆盖极端 logit。

```python
def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def preference_loss_gradient(theta, feature_difference, label):
    """label=1 prefers A. Difference is f(A)-f(B); temperature is fixed at 1."""
    if not 0.0 <= label <= 1.0:
        raise ValueError("Preference label must lie in [0, 1].")
    z = theta * feature_difference
    loss = max(z, 0.0) - label * z + math.log1p(math.exp(-abs(z)))
    gradient = (sigmoid(z) - label) * feature_difference
    return loss, gradient
```

**算法：偏好学习与控制的交替循环。第 4 步是否重标记，是协议的一部分。**

1. 1. 用当前策略收集轨迹；保存原始观测与奖励模型版本。
1. 2. 从候选片段中选比较对；记录选择规则与反馈成本。
1. 3. 收集偏好 y；最小化比较损失。
1. 4. 用更新后的奖励模型重标记训练经验。
1. 5. 按约定预算改进策略；继续收集新的片段。
1. 6. 在未参与拟合的比较与原任务指标上评价。

Christiano 等人的工作把片段偏好学习与深度 RL 结合。PEBBLE 进一步使用无监督预训练和经验重标记以提高反馈利用率。二者都不能仅凭训练比较准确率证明奖励在新行为上可靠。策略会主动寻找模型给高分的行为，因而会改变奖励模型的输入分布。

<a id="experiment-preference_reward"></a>

### 实验：实验 · 从有噪声的片段比较学习奖励

只知道两个轨迹片段谁更好，能否学到解释偏好的奖励特征？

**环境与可用信息。** 全部长度 3 的二动作序列，共 8 个片段。输入特征是右动作数与转向次数；不存在在线环境终止或控制器。标签由两片段特征差的线性得分经 sigmoid 产生 Bernoulli 偏好，生成权重为 (1,−0.7)。

**设置。** 每步随机取两个片段并获得 1 个偏好标签，共 1200 个标签；奖励权重从 (0,0) 开始，SGD 步长 0.08。完整模型学习两维，基线冻结转向权重为 0。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** preference_reward.py 优化 Bradley–Terry 交叉熵，梯度由预测偏好概率减真实随机标签，再乘片段特征差得到。已知生成权重只用于数据生成和隔离评价，不直接提供训练梯度。

**测量。** 枚举全部 64 个有序片段对，以真实偏好概率计算期望交叉熵。它包含不可约标签噪声，正确模型的交叉熵也不必为 0。

```bash
python3 implementations/extended_knowledge/preference_reward.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/preference_reward/curves.svg)

横轴：preference_labels。纵轴：冻结偏好期望交叉熵。每种方法 1200 preference_labels；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 第 1200 个标签后，完整特征模型的平均交叉熵为 0.55666，单特征基线为 0.59108。完整模型第 600 步为 0.55282，后面略升，符合固定步长随机估计可能波动的现象。

**结论边界。** 这是奖励拟合组件，不含人类标注、主动查询、神经奖励模型或策略优化闭环。较低偏好交叉熵不证明训练出的控制器有更高真实收益。

**继续实验。** 给所有片段添加同一常数奖励，偏好概率会变吗？再构造训练中未出现的转向模式，分别测奖励模型误差和最终策略收益。

[源码](https://yingwen.io/crl-code/implementations/extended_knowledge/preference_reward.py) · [逐种子记录](https://yingwen.io/crl-code/results/preference_reward/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/preference_reward/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/preference_reward/curves.json)

<a id="lesson-irl"></a>

## 8. 从示范中推断奖励：一个可手算的 MaxEnt IRL

偏好学习给出比较标签。逆强化学习给出示范行为。二者都要对人的行为与噪声作假设。这里使用有限、可枚举、等基准权重的可行轨迹集合，以及线性奖励。这个例子避开随机动力学中的因果熵问题。

$$
r_\theta=\theta^\top f,\qquad F(\tau)=\sum_t f_t,\qquad P_\theta(\tau)=\frac{\exp(\theta^\top F(\tau))}{Z(\theta)}
$$

Z 是所有可行轨迹指数分数的和。固定环境和轨迹集合。若轨迹具有不同基准概率，应把它们纳入分布，而不是当作奖励。

$$
\widehat F=\frac1M\sum_{i=1}^M F(\tau_i),\qquad\mathcal L(\theta)=\log Z(\theta)-\theta^\top\widehat F,\qquad \nabla\mathcal L=\sum_\tau P_\theta(\tau)F(\tau)-\widehat F
$$

$\tau_1,\ldots,\tau_M$ 是给定的 $M\ge1$ 条示范，$\widehat F$ 是每条示范的累计特征再取样本平均，而不是对所有时间步重新平均。这里 $\mathcal L$ 是平均负对数似然。对 $\log Z$ 求导得到模型期望特征，所以梯度是“模型特征−示范特征”。梯度下降使两者靠近。

有限轨迹 MaxEnt 负对数似然与精确梯度。测试不是完整随机 MDP 的 MaxCausalEnt 实现。

```python
def maxent_loss_gradient(theta, features, empirical_mean):
    """Finite, equally weighted feasible trajectories; deterministic toy model."""
    logits = [theta * f for f in features]
    peak = max(logits)
    unnormalized = [math.exp(x - peak) for x in logits]
    total = sum(unnormalized)
    probs = [x / total for x in unnormalized]
    log_z = peak + math.log(total)
    expected_feature = sum(p * f for p, f in zip(probs, features))
    return log_z - theta * empirical_mean, expected_feature - empirical_mean
```

相同行为可能由多个奖励解释。势函数变换就是一类不可辨识性来源。示范也可能受动作限制、错误信念或有限计算影响。CIRL 把人和机器人放进一个合作的部分信息博弈。Inverse Reward Design 则把手写代理奖励及其训练环境当作有关真实目标的证据。这些方法改变了推断问题，不是直接读出人的“真正奖励”。

<a id="experiment-maxent_irl"></a>

### 实验：实验 · 全轨迹配分函数与示范特征匹配

从有限示范拟合 MaxEnt 轨迹分布时，特征遗漏与采样误差怎样出现？

**环境与可用信息。** 四层二动作确定性 DAG，枚举 16 条完整轨迹，走满 4 步后结束。特征为右动作数与转向次数；专家全轨迹概率正比于 exp(0.8×右动作数−0.5×转向数)。这是奖励推断，不执行新的控制策略。

**设置。** 每个 seed 先固定采样 64 条专家示范，共 256 个示范动作；参数从 (0,0) 开始。做 1200 次全批梯度更新，步长 0.05。对照只学习右动作特征；此处横轴是梯度更新，不是新增环境数据。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** maxent_irl.py 精确枚举全轨迹 log-sum-exp，梯度为模型期望特征减示范经验特征。不能用逐个状态独立归一化的局部 softmax 替代这个全轨迹配分函数。

**测量。** 纵轴是对真实专家全轨迹分布的冻结交叉熵，而不是对 64 条训练示范的损失。后者下降不保证前者逐步下降。

```bash
python3 implementations/extended_knowledge/maxent_irl.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/maxent_irl/curves.svg)

横轴：gradient_updates。纵轴：冻结专家分布轨迹交叉熵。每种方法 1200 gradient_updates；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 第 1200 次更新，完整模型的平均交叉熵为 2.19096，单特征基线为 2.24758；第 600 次已分别约 2.19095 和 2.24758。继续优化几乎不再改善当前评价，新增迭代不能替代新增示范。

**结论边界。** 有限确定性轨迹枚举，不是随机动力学中的 causal entropy IRL。64 条示范带来估计误差；奖励不可辨识也未因优化收敛而消失。没有在线策略学习结果。

**继续实验。** 固定示范不变，比较更多更新与更多示范。再给所有完整轨迹奖励加同一常数，验证概率不变，并解释为何奖励参数不一定唯一。

[源码](https://yingwen.io/crl-code/implementations/extended_knowledge/maxent_irl.py) · [逐种子记录](https://yingwen.io/crl-code/results/maxent_irl/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/maxent_irl/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/maxent_irl/curves.json)

<a id="lesson-constraints"></a>

## 9. 标量化、多目标、约束与风险

多目标问题先给出奖励向量。固定权重将它变成一个标量任务，但选择权重本身就是设计决策。安全约束则限制允许的策略集合。期望成本限制与“每次都安全”不是同一要求。

$$
\max_\pi J_r(\pi)\quad\text{s.t.}\quad J_c(\pi)\leq d,\qquad \mathcal L(\pi,\lambda)=J_r(\pi)-\lambda(J_c(\pi)-d),\quad\lambda\geq0
$$

在适当的有限 MDP、随机策略和占用测度条件下，可研究线性规划与对偶。实际神经优化、未知动力学及有限样本还会带来误差。

单步例子：安全动作奖励 1、成本 0；风险动作奖励 3、成本 1。若允许期望成本不超过 0.2，则最优风险概率为 0.2，期望奖励为 1.4。取 λ=2 时，两个动作的惩罚后奖励相等。任意混合都最优，其中很多违反约束。因此“找到一个惩罚系数”不等于“任意优化器都返回可行策略”。

解析求解单步 CMDP。测试明确指出期望约束不是逐次安全保证。

```python
def constrained_mixture(budget):
    """Safe action: (reward,cost)=(1,0); risky: (3,1).

    Choose risky with probability p. E[cost]<=budget; this is NOT a
    per-trajectory guarantee. The risk limit is an expectation constraint.
    """
    if not 0 <= budget <= 1:
        raise ValueError("Budget must lie in [0, 1].")
    p = budget
    return {"p_risky": p, "expected_reward": 1 + 2 * p,
            "expected_cost": p}
```

风险准则可能涉及回报分布的分位数、尾部损失或失败概率。把总回报送入非线性效用后，通常不能直接沿用原始 Markov 状态上的加性 Bellman 方程。可以增广累计量或使用相应风险递推，但必须重新给出假设。

<a id="lesson-failures"></a>

## 10. 奖励投机、篡改与错误泛化

| 失败类型 | 怎么发生 | 区分它的实验 |
| --- | --- | --- |
| 规格错误 | 奖励准确实现了错误代理目标，例如只数箱子移动次数。 | 保持奖励程序不变，独立测真正交付与副作用。 |
| 奖励模型外推错误 | 训练片段拟合良好，新行为获得虚假高分。 | 在新策略产生的片段上取得独立标签；分布外误差单独报告。 |
| 奖励通道篡改 | 行为改变评分程序、传感器或输入，间接提高分数。 | 固定世界结果，干预评分通道；检查激励路径。 |
| 优化失败 | 正确奖励下仍未找到好策略。 | 给小问题的精确求解器或已知可行策略作对照。 |

不可把所有失败都称为 reward hacking。任务指标、所给奖励与实际行为需要分别记录。AI Safety Gridworlds 提供奖励投机和安全问题的小型诊断环境。它不能代表所有真实部署风险。奖励篡改的因果分析还区分修改奖励函数与修改奖励函数输入。

<a id="lesson-continual"></a>

## 11. 持续交互使奖励设计多了哪些问题

- 奖励在变，还是对同一目标的估计在变？前者改变任务；后者改变学习信号。实验必须分开。
- 新状态、新物体和新技能出现后，旧奖励模型可能没有定义或校准。测试应包含这种表示扩展。
- 偏好反馈有成本和延迟。询问也是一个行动。评价者可用的信息和时间属于协议。
- 旧经验若按新模型重标记，就不再使用当时的奖励估计。保存原始事件、反馈和模型版本，允许重算。
- 内在动机可以服务于探索。子任务奖励可以服务于技能学习。最终评价仍要说明它们怎样帮助外部目标。
- 生命期回报要计入失败探索、人工纠正和奖励模型训练成本。最后的高分策略不能抹去这些成本。

一个可检验的问题是：性能下降究竟来自世界变化、奖励模型漂移，还是控制器失去适应能力？固定其中两项、只改变第三项。再进行联合变化实验。直接一起更换环境、奖励模型和控制器，无法定位原因。

<a id="lesson-intrinsic"></a>

## 12. 奖励也可以学习：有限智能体的内外层目标

为什么内部学习奖励不必等于最终评价？考虑一个学习时间有限的智能体。只在成功时给奖可能使它来不及发现关键行为。Optimal Rewards 路线据此区分任务评价与促成学习的信号：选择内部奖励，使有限学习过程获得更好的外部结果。这不是取消外部目标，而是把奖励设计写成优化问题。

$$
r_\eta^{\mathrm{train}}=r^{\mathrm{ext}}+r_\eta^{\mathrm{int}},\qquad \eta^*\in\arg\max_\eta J_{\mathrm{ext}}(L_\eta)
$$

Lη 表示使用该训练奖励的完整学习器。外层可以评价一次更新后的策略，也可以评价整个生命期；这两种目标不同。

一类可微方法沿策略更新求奖励参数的梯度。为看清链式法则，先只考虑一次更新，固定本次数据，并假设更新前的 θ 不依赖 η。

$$
\theta'=\theta+\alpha g(\theta,\eta),\qquad \nabla_\eta J_{\mathrm{ext}}(\theta')=\alpha\left(\frac{\partial g}{\partial\eta}\right)^\top\nabla_{\theta'}J_{\mathrm{ext}}(\theta')
$$

如果 θ 本来依赖 η，需要保留更早的敏感度。若求完整交互期望的梯度，还需处理采样分布对 η 的依赖；固定轨迹上的链式法则不自动等于无偏的完整元梯度。

一个精确可验的特例是两动作 bandit。动作 1 的外部奖励为 1，动作 0 为 0。令 p=σ(θ)，内部奖励差为 η。一次期望策略梯度更新是 θ′=θ+αp(1−p)(1+η)。外层只评价 σ(θ′)，不把内部奖金计入成功。

两动作 bandit 上的一步奖励元梯度。有限差分验证的是这个解析例子，不是 LIRPG 全部实验。

```python
def intrinsic_meta_gradient(theta, eta, alpha):
    """One expected-gradient bandit update; theta is independent of eta here.

    Extrinsic rewards are (0, 1); intrinsic reward difference is eta.
    The outer objective is the new probability of action 1, not its bonus.
    """
    p = sigmoid(theta)
    sensitivity = alpha * p * (1 - p)
    next_theta = theta + sensitivity * (1 + eta)
    outer_return = sigmoid(next_theta)
    gradient = outer_return * (1 - outer_return) * sensitivity
    return outer_return, gradient
```

| 研究路线 | 优化的对象 | 必须保留的区别 |
| --- | --- | --- |
| Optimal Rewards；Singh、Lewis、Barto 等 | 有限能力智能体使用的奖励机制 | 好的内部信号取决于智能体限制与环境分布。 |
| PGRD；Sorg、Lewis、Singh | 通过奖励参数影响受限规划器或决策过程 | 奖励影响行为的路径不必是神经策略的一次学习更新。 |
| LIRPG；Zheng、Oh、Singh | 通过策略学习更新优化内在奖励 | 把外部评价对策略更新的依赖接回奖励模型。 |
| What Can Learned Intrinsic Rewards Capture? | 跨生命期学习内在奖励 | 元训练的生命期、计算与先验必须计入，不能称为完全从零的单生命期学习。 |

这条线与元学习、探索和目标构造相交。它与势函数塑形不同：一般学得的内在奖励没有策略不变性保证。判断是否有用，要看外部评价、训练预算和迁移条件，而不只是内部奖励增长。

<a id="lesson-code"></a>

## 13. 运行、阅读原始代码与选择基准

先用 [三步送货单文件程序](/crl-code/tutorials/reward-design-walkthrough.py) 复算本章主线。下载该文件到任意空目录即可运行，无第三方依赖；`--test` 检查所有有限路径、边界恒等式及折扣退化情形，`--json` 输出每个信号下的三条路径和真正执行的选择。也可读取 [逐步数值 JSON](/crl-figures/reward-design-walkthrough-data.json)。

输出按原奖励、位置奖金、正确势差、错误边界、错误折扣依次给出选择；预期为 early、timeout、early、timeout、late。

```sh
python3 reward-design-walkthrough.py
python3 reward-design-walkthrough.py --test
python3 reward-design-walkthrough.py --json
```

网站的 [计算模块](/crl-code/figures/reward-design-walkthrough.mjs) 在带时钟的状态上做后向动态规划；下载版先枚举完整路径，再用 Fraction 有理数求和，并以闭式表达式核对。二者共享任务规格，求值路径不同。规划器获得全部转移与指定奖励；这项检查支持回报与选择的计算，不包含奖励模型学习、未知环境探索或持续学习效果的结论。

原有配套程序 `reward_design_lab.py` 同样只依赖 Python 标准库。运行 `python3 reward_design_lab.py demo` 看其余章节算例，再运行 `python3 reward_design_lab.py test` 检查顺序反例和梯度。它不训练深度奖励网络，不宣称复现下面论文的完整实验。

| 入口 | 读哪部分 | 怎样开始 |
| --- | --- | --- |
| B-Pref / PEBBLE 作者代码 | reward_model.py、train_PEBBLE.py、replay_buffer.py | 追踪片段分数、标签方向、查询预算和重标记时点。 |
| Reward Machines 作者代码 | 奖励自动机、任务文件与学习循环 | 画出一个任务的自动机；再删除一个记忆状态观察语义变化。 |
| imitation 的 preference comparisons | 官方文档和教程 5 | 它是库作者提供的实现，不是 2017 年论文原作者代码。先完成小环境教程。 |
| AI Safety Gridworlds | README 中 reward gaming 与相关环境 | 分别保存环境给分和独立 performance 指标。原仓库已归档，先核对依赖。 |

复现实验至少需要三组预算：环境交互、人的反馈或模拟标签数、优化与奖励模型计算。使用脚本化偏好标签时，称为模拟偏好；不能由此声称已经验证真实人的偏好学习。

<a id="lesson-branches"></a>

## 14. 研究条线怎样连接

| 研究问题 | 代表路线 | 下一步读法 |
| --- | --- | --- |
| 目标可表达吗 | 奖励假设、Markov 表达能力、奖励自动机 | 先明确偏好域、状态与时间，再看表示定理。 |
| 怎样帮助有限预算的学习 | 势函数塑形、内在动机、子任务奖励 | 区分目标保持与优化加速。接目标、探索与 options 章。 |
| 怎样从人获得目标信息 | IRL、偏好学习、CIRL、逆奖励设计 | 分开观测模型、反馈选择、奖励推断和控制。 |
| 怎样防止代理失效 | 奖励投机、奖励篡改、分布外评价 | 比较训练信号与独立评价；使用可干预的小问题。 |
| 怎样长期更新目标知识 | 奖励模型持续学习、奖励感知状态与反馈预算 | 接状态、控制、可塑性和实验章。先隔离漂移来源。 |

这些路线可以组合，但不是同一种算法的版本。奖励自动机规定时间结构；偏好学习估计奖励；塑形改变学习信号；CMDP 规定允许的策略。组合之前先确定各自处理的对象和保证。

<a id="lesson-check"></a>

## 15. 练习与可反驳的实验

先遮住送货表格最后两列再作预测：把门口势由3改成0，哪些错误会消失，代理占用奖金的错误是否仍在？把折扣改为1，早、晚送达的原目标为何并列，漏写塑形折扣为何也变得无影响？最后在正确塑形下只收集第一步，分别用零尾值和 $1-\Phi(1,A)$ 作尾值，核对它们比较的是哪个任务。

- 推导 γ<1 时，每步加常数 b 在无限持续任务中的价值偏移。解释为何同一推导不能直接用于可变长度终止回合。
- 把 KD/DK 例子改为有时钟的状态。构造一个能区分顺序的奖励。指出新增的信息。
- 令奖励势在线更新。写出一个相邻时间索引一致的塑形记录格式。再构造混用旧势与新势的反例。
- 对两段长度不同的轨迹，各步加同一常数，比较偏好概率是否改变。相同长度时结果又如何？
- 用代码检查偏好梯度和 MaxEnt 梯度。交换 A/B 后，标签必须怎样变化？
- 比较无塑形、距离奖金、势差塑形。匹配交互与调参预算，分别报告原奖励、训练奖励、成功率和行为长度。
- 在奖励模型持续更新的条件下，比较经验不重标记、全部重标记、按版本重标记。先提出会导致它们不同的机制，再选择任务。

离开本章前，应能回答：我的奖励表达了什么？我增加了什么假设？保证属于目标表示、最优性、学习速度还是安全性？代码中的哪项测试能够推翻我的实现？

## 本章的实验设计

先分清训练奖励、设计者评价与辅助信号。用反例检查目标保持，再测有限预算学习。

设定：单状态两步顺序任务、可变时长路线与片段偏好。固定真实目标，分别更改奖励表示、塑形和估计模型。

- 顺序不同但转移计数相同时，固定加性奖励不能严格排序。
- 势函数塑形逐轨迹满足望远镜恒等式；非零终点势不能省略。
- 偏好交叉熵和有限轨迹 MaxEnt 梯度通过有限差分。

对照：原奖励、距离奖金与势差塑形；相同数据和反馈预算的奖励模型；固定奖励模型与在线更新；重标记与不重标记

记录：原任务评价与实际训练奖励分别记录；偏好查询、标签噪声与分布外预测误差；真实交互、重标记和优化计算成本

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-classic)

## 学习与研究衔接

偏好、奖励机制、回报与辅助信号不是同一对象。保持最优策略、加快学习与符合设计者意图需要不同证据。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-reward-design) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=reward-design) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=reward-design)

## 从本章进入实践

[策略梯度与控制](https://yingwen.io/zh/continual-rl/code/#practice-policy-control)：优化器确实降低了损失，为什么行动仍可能变差？

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

给定状态后可以估计价值；观测不足时，还要学习保留哪些历史。GVF 规定预测什么，RTRL 计算递归敏感度，资格迹组织时间信用。应分别检验信息是否进入状态、反馈能否教会这种保留，以及有限预测预算怎样分配，而不是把三者当作替代算法。

- [The Ungrounded Alignment Problem](https://yingwen.io/zh/continual-rl/research/#recent-openmind-ungrounded-alignment)

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

教材可以先给定目标和技能集合；持续构造还要决定哪些行为值得练习、维护或放弃。谱结构、路径奖励、时间距离和语言先验提供不同候选偏置。先固定候选比较选择与组合，再改变生成器，才能辨认下游收益究竟来自哪一步。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [Reward-Aware Proto-Representations in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-reward-aware-proto-representations)
- [MaestroMotif: Skill Design from Artificial Intelligence Feedback](https://yingwen.io/zh/continual-rl/research/#recent-maestromotif-semantic-skills)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

给定模型可研究怎样规划；模型也在学习时，规划会选择性地查询误差，并改变以后的数据。Dreamer、STOMP 和 DRAGO 分别研究想象控制、随机时长行为模型和旧知识保留。新的比较应固定规划查询与总预算，检验哪些后果误差真正改变选择，哪些维护值得继续。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)

### Reward-Respecting Subtasks for Model-Based Reinforcement Learning

Richard S. Sutton, Marlos C. Machado, G. Zacharias Holland, David Szepesvari, Finbarr Timbers, Brian Tanner, Adam White

Artificial Intelligence · 2023 · 支持方法与理论

#### 研究问题

学到一个能到达子目标的技能之后，为什么它仍可能不适合主任务规划？

#### 关键机制

STOMP 把子任务、option、模型和规划连起来。子任务保留原任务的路径奖励，并用带有特征偏好的终止价值表达目标；学习得到策略和终止规则后，再预测该行为的累计奖励与折扣终点。这样，技能不会因为只追求到达子目标而忽略途中代价。

#### 证据

论文用小问题展示奖励感知子任务怎样产生可用于规划的行为与后果模型。实验将各阶段依次进行，从而能够分清子任务设计、option 学习、模型学习和规划各自的作用。

#### 条件与限制

这些实验没有同时运行并更新全部阶段。特征选择、子任务淘汰和规划计算分配仍需算法；终止收益属于子任务规格，不能随意换成固定终点奖励，也不能混入真实奖励模型。

#### 阅读与实验

先在同一绕路环境比较两种子任务，并计算奖励模型、折扣终点模型和一次备份。再固定候选与容量，检验下游规划用途能否指导技能保留和模型重学；这第二步是拟议研究，不是原论文已证实的闭环。

#### 原文与相关入口

- [期刊论文](https://doi.org/10.1016/j.artint.2023.104001)：STOMP 与奖励感知子任务的正式论文。
- [作者预印本](https://arxiv.org/abs/2202.03466)：最初预印本早于期刊年份；阅读停止收益的精确定义。

### Reward-Aware Proto-Representations in Reinforcement Learning

Hon Tik Tse, Siddarth Chandrasekar, Marlos C. Machado

NeurIPS 2025 · 2025 · 支持方法与理论

#### 研究问题

仅编码可达关系的表示，怎样进一步反映奖励与行动成本？

#### 关键机制

论文研究 default representation，将奖励或成本纳入对未来状态关系的表示，并给出动态规划与 TD 学习方法。由此提取的谱特征可以参与技能发现、奖励塑形和迁移。它沿着 SR 的后果预测思路前进，但不再把奖励完全留到最后的线性读出阶段。

#### 证据

作者提供表格问题中的推导，并用表示、技能和迁移实验展示奖励信息如何改变学得的结构。代码包含 SR、DR 的计算和在线表示学习实验。

#### 条件与限制

把奖励纳入表示会改变迁移边界：奖励或内部成本变化后，原表示可能需要重学。论文结果不能解释为任意新奖励下都能免费零样本迁移。

#### 阅读与实验

固定转移图，只改变一处通行成本，比较 SR 与 DR 的谱方向。随后检查新的 eigenoption 是改变了可达性，还是改变了对路径代价的偏好。

#### 原文与相关入口

- [论文与版本记录](https://arxiv.org/abs/2505.16217)：NeurIPS 2025；后续版本修订不改变会议年份。
- [作者实现](https://github.com/httse9/Reward-Aware-Proto-Representations)：从 minigrid_basics/examples 的表示计算与技能实验开始。

#### 作者代码

[原论文作者仓库。](https://github.com/httse9/Reward-Aware-Proto-Representations)

奖励感知表示、谱特征与相关 MiniGrid 实验。

### MaestroMotif: Skill Design from Artificial Intelligence Feedback

Martin Klissarov, Mikael Henaff, Roberta Raileanu, Shagun Sodhani, Pascal Vincent, Amy Zhang, Pierre-Luc Bacon, Doina Precup, Marlos C. Machado, Pierluca D’Oro

ICLR 2025 · 2025 · 支持方法与理论

#### 研究问题

语言描述如何变成可训练的技能奖励，并进一步组织成一个层次策略？

#### 关键机制

设计者先给出技能描述。语言模型的偏好反馈被用于训练奖励模型，再用生成的代码规定技能启动、终止和组合方式；强化学习负责学习实际执行行为。这把语义先验、奖励学习和时间抽象串成了具体训练流程。

#### 证据

论文在 NetHack 学习环境中检验复杂技能与任务组合。作者仓库同时包含偏好、代码生成和 RL 训练模块，可以追踪自然语言到环境动作的完整依赖。

#### 条件与限制

语义知识、技能描述和语言模型来自外部设计过程。该证据并不说明智能体仅凭自身交互就能产生同样的技能体系；偏好模型也可能与真实目标不一致。

#### 阅读与实验

选择一项技能，分别列出描述、偏好标签、训练奖励、终止条件和下游用途。移除语义描述或改变奖励模型时，要单独计量额外查询与人工成本。

#### 原文与相关入口

- [ICLR 2025 原文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2dc5a0faac8102fd47363795f71126ee-Abstract-Conference.html)：技能设计、奖励学习与组合实验。
- [作者实现](https://github.com/mklissa/maestromotif)：偏好学习、代码生成和执行策略的不同模块。

#### 作者代码

[原论文作者仓库。](https://github.com/mklissa/maestromotif)

MaestroMotif 的偏好处理、技能组织与 RL 实验。

### The Ungrounded Alignment Problem

Marc Pickett, Aakash Kumar Nain, Joseph Modayil, Llion Jones

ICDL 2025（预印本 2024） · 2025 · 支持方法与理论

#### 研究问题

即使目标已经写明，智能体怎样知道其中的概念对应哪一段未知感知输入？

#### 关键机制

利用固定的字符转移关系知识，对未知图像编码进行无标签对齐。共享编码器与对比学习把感知模式连接到既有关系结构。

#### 证据

原文在置换像素的字符序列上检验触发词识别。这隔离了概念指代与感知编码的问题，而非直接训练一般 RL 控制器。

#### 条件与限制

预先给定的关系、重启训练与阈值选择是实验条件。受控接地任务的准确率不能当作通用价值对齐或单生命期适应的证据。

#### 阅读与实验

分别改变感知编码和事件关系，区分事件识别失败、预测失败与奖励偏好错误。

#### 原文与相关入口

- [原文](https://arxiv.org/html/2408.04242v1)：问题定义、关系传播与训练条件。
- [机构发表目录](https://www.openmindresearch.org/research)：列为 ICDL 2025。

#### 作者代码

[原论文给出的作者仓库。](https://github.com/EmergenceAI/babybeaver)

作者字符接地任务实现；不是流式 actor–critic 代码。


<a id="chapter-code"></a>

## 下载与运行

原创建模反例、望远镜求和、偏好与 MaxEnt 梯度、单步约束优化。不是大规模算法复现。

[下载 reward_design_lab.py](https://yingwen.io/zh/continual-rl/download/reward_design_lab.py)

```sh
python3 reward_design_lab.py demo
python3 reward_design_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto — Reinforcement Learning: An Introduction, 2nd ed., §§3.2, 17.4](http://incompleteideas.net/book/the-book-2nd.html)：目标与奖励信号的区别；稀疏奖励、先验引导与受限学习器的奖励设计。本章三步送货为原创算例。

- [Singh, Lewis & Barto — Where Do Rewards Come From? (CogSci 2009)](https://web.eecs.umich.edu/~baveja/Papers/singh-lewis-barto-2009-cogsci.pdf)：内在奖励、外部评价与受限智能体。原文从最优奖励框架讨论奖励来源。

- [Sorg, Lewis & Singh — Reward Design via Online Gradient Ascent (NeurIPS 2010)](https://proceedings.neurips.cc/paper_files/paper/2010/hash/168908dd3227b8358eababa07fcaf091-Abstract.html)：PGRD：奖励参数通过受限决策过程影响外层表现。

- [Zheng, Oh & Singh — On Learning Intrinsic Rewards for Policy Gradient Methods (NeurIPS 2018)](https://proceedings.neurips.cc/paper/2018/hash/51de85ddd068f0bc787691d356176df9-Abstract.html)：LIRPG 原文。从外部回报沿策略更新求内在奖励梯度。

- [LIRPG：原论文作者实现](https://github.com/Hwhitetooth/lirpg)：原论文脚注给出的仓库。基于历史 TensorFlow / Baselines；用独立环境核对依赖，不与本站标准库示例混用。

- [Zheng et al. — What Can Learned Intrinsic Rewards Capture? (ICML 2020)](https://proceedings.mlr.press/v119/zheng20b.html)：跨多个生命期学习内在奖励；注意内外层时域与迁移条件。

- [Silver, Singh, Precup & Sutton — Reward is Enough (2021)](https://www.sciencedirect.com/science/article/pii/S0004370221000862)：研究假说：奖励追求与智能能力。与奖励表示定理分开读。

- [Bowling, Martin, Abel & Dabney — Settling the Reward Hypothesis (ICML 2023)](https://proceedings.mlr.press/v202/bowling23a.html)：读公理 1–5、定理 4.1 与设计者目标一节。注意历史域和时间一致性条件。

- [Abel et al. — On the Expressivity of Markov Reward (NeurIPS 2021)](https://arxiv.org/abs/2111.00876)：不同任务描述下的奖励表达限制。状态与奖励函数的允许输入非常关键。

- [Ng, Harada & Russell — Policy Invariance under Reward Transformations (ICML 1999)](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)：作者镜像，§§2–3、Theorem 1。有限状态折扣无限时域或满足 proper 条件的未折扣吸收任务；必要性是在未知转移和奖励下统一保证，不能解释为任一非势函数都会破坏每个给定任务。

- [Toro Icarte et al. — Reward Machines：作者论文与代码](https://github.com/RodrigoToroIcarte/reward_machines)：作者仓库提供奖励结构、任务与算法。适合把顺序任务从文字变成状态机。

- [Christiano et al. — Deep Reinforcement Learning from Human Preferences (2017)](https://arxiv.org/abs/1706.03741)：片段偏好、奖励预测与策略学习。区分真实反馈和模拟偏好实验。

- [Lee, Smith & Abbeel — PEBBLE (2021)](https://arxiv.org/abs/2106.05091)：无监督预训练、反馈选择与经验重标记如何组合。

- [B-Pref / PEBBLE：研究团队官方代码](https://github.com/rll-research/BPref)：含 PEBBLE 训练脚本与基准。先核对依赖、标签噪声与查询预算。

- [imitation — Preference Comparisons 官方教程](https://imitation.readthedocs.io/en/latest/algorithms/preference_comparisons.html)：学习库的文档与实现入口。不是 Christiano 等人原论文的作者仓库。

- [Ziebart et al. — Maximum Entropy Inverse Reinforcement Learning (AAAI 2008)](https://www.cs.cmu.edu/~bziebart/publications/maximum-entropy-inverse-reinforcement-learning.html)：原作者页面含论文和修正提示。随机动力学需继续读 maximum causal entropy。

- [Hadfield-Menell et al. — Cooperative Inverse Reinforcement Learning (2016)](https://arxiv.org/abs/1606.03137)：合作、目标不确定性与人的信息优势。不是把示范直接当最优动作标签。

- [Hadfield-Menell et al. — Inverse Reward Design (2017)](https://arxiv.org/abs/1711.02827)：手写代理奖励只是在特定训练环境中的证据，不能无条件外推。

- [Everitt et al. — Reward Tampering: A Causal Influence Diagram Perspective](https://arxiv.org/abs/1908.04734)：奖励函数篡改与输入篡改的因果区别。

- [Google DeepMind — Specification Gaming](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)：用具体失败区分规格漏洞与控制学习失败。

- [AI Safety Gridworlds：原始环境代码](https://github.com/google-deepmind/ai-safety-gridworlds)：小型安全诊断环境。仓库已归档；独立 performance 指标不是智能体的训练奖励。

- [Pickett et al. — The Ungrounded Alignment Problem](https://arxiv.org/html/2408.04242v1)：2024 预印本；Openmind 与 ICDL 官方日程列为 ICDL 2025。关系先验、无标签感知接地与触发词检测。

- [EmergenceAI — babybeaver 作者代码](https://github.com/EmergenceAI/babybeaver)：论文直接给出的原作者仓库；检查训练重启、关系表和触发阈值，不能误标为流式 RL 实现。
