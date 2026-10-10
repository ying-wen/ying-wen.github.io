# 深度价值学习：DQN、Double DQN 与目标的时间顺序

现代深度强化学习 · 第 1 章

把表格 Q-learning 换成网络后，损失、数据与目标为什么都需要重新组织？

## 本章内容

- 在同一个共享特征算例中算出预测、动作与旧经验标签的变化。
- 写出 DQN 与 Double DQN 的不同目标，区分冻结副本与停止梯度。
- 区分环境终止、采样截断和目标网络更新。
- 把标准库手算对应到完整 PyTorch 交互与训练循环。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [特征、泛化与半梯度控制](https://yingwen.io/zh/continual-rl/foundations/approximation/features-control/)：从半梯度动作价值更新进入神经网络控制。
- [离策略函数逼近：覆盖、发散与稳定更新](https://yingwen.io/zh/continual-rl/foundations/approximation/off-policy/)：理解离策略、自举与共享表示共同带来的稳定性问题。


### Bellman 递推

当前价值等于即时奖励加折扣后的后继价值；预测目标与最优控制目标不同。

### 函数与梯度

神经网络把参数和输入映射为输出。链式法则必须指明哪些量参与求导，哪些量作为固定目标。

### 经验分布

on-policy 数据由当前策略产生；off-policy 数据可来自旧策略，但能否正确复用取决于算法目标和数据覆盖。

<a id="problem-definition"></a>

## 本章的问题定义

在离散动作控制中，用神经网络近似最优动作价值；环境模型未知，数据来自不断变化的行为。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 在线网络 $Q_\theta$、目标网络 $Q_{\bar\theta}$、经验分布 $\mathcal D$；$m=0$ 表示真实终止，否则 $m=1$。

### 需要求解的对象

学习能产生有效贪心动作的价值近似；最终目标仍是外部期望回报。

### 信息与数据权限

可重放已保存转移，但不能由 replay 自动获得未覆盖动作的真实结果。

$$
y=r+\gamma m\max_{a^{\prime}}Q_{\bar\theta}(s^{\prime},a^{\prime}),\qquad L(\theta)=\mathbb E_{\mathcal D}[(Q_\theta(s,a)-\operatorname{sg}(y))^2]
$$

这是在固定目标版本和数据分布下的回归损失，$\operatorname{sg}$ 表示停止梯度。它是逼近 Bellman 最优性更新的机制，不是等同于最大化 J 的恒等式。

### 成立条件与解的含义

- 有限可枚举动作；明确终止、目标网络同步和 replay 采样规则。
- 网络容量、覆盖和优化误差均可能限制解；一般非线性 DQN 没有表格式全局收敛保证。

判断准则：先在已知小问题检查 target 和动作选择；再报告实际收益、价值尺度及覆盖，不能只看训练 loss。

### 适用边界

- 把固定 target 下的回归下降解释为整个非平稳控制过程单调改善。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 限制表示或采用近似 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：本章用神经动作价值与指定贪心提取规则近似折扣控制；表示、覆盖和求解误差限制实际行为，回归损失不是完整学习器收益。

- 改变信息或数据协议 · [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)：本章允许 replay 和目标网络滞后；相对严格流式协议，它增加数据复用、持久内存及更新调度。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

自举目标、采样分布与网络参数同时变化，且最大化会选择正向估计噪声。

### 本章的核心思路

将采样、回归和目标更新放在不同时间尺度，并分离动作选择与评价。

1. [固定一个回归目标再求梯度](#lesson-derive)：停止对 target 反传，明确每次优化实际使用的目标版本。

2. [拆开选择和评价](#lesson-double)：Double DQN 由在线网络选动作，再由目标网络评值。

3. [规定三个时钟的执行顺序](#lesson-algorithm)：环境步、优化步和目标同步步不是同一计数器。

结论与条件：目标网络与 replay 是稳定化机制；它们不消除函数逼近、自举和离策略耦合的所有风险。

### 相关方法改变了什么

- DQN / Double DQN：改变后继目标中的动作选择器，不改变外部奖励目标。

- Replay / 严格在线更新：改变数据复用和相关性，也改变内存与每步成本。


<a id="lesson-setting"></a>

## 1 · 学习对象、数据和三个时间尺度

表格方法给每个状态动作对一个独立的数。状态很多时，既存不下所有条目，也等不到每个条目都获得足够经验。函数逼近让相似输入共享学习结果；深度网络进一步让“怎样共享”也随经验改变。图像中的位置、速度或物体关系不必全由设计者预先编码，但训练信号必须让网络保留对预测和行动有用的差别。网络更大，并不能使未观察到的信息自动出现。

在离散动作 MDP 中，网络 $Q_\theta(s,a)$ 近似动作价值。它不是一个动作分类器：输出可以为负，也不需要和为一。行为策略通常采用 $\epsilon$-greedy，而学习目标使用贪心后继动作，因此行为策略与目标策略不同。

这带来一个闭环：更新网络会同时改变许多预测；这些预测生成下一批自举标签，也决定接下来访问哪里。一次监督回归通常可以先固定数据和标签，DQN 则必须在学习过程中组织它们。Replay 暂时固定可重用的经验，target network 暂时固定标签生成器。理解这两项设计，要从这个相互影响的过程出发，而不能只把它们当作网络训练的默认附件。

三种时间需要分开。环境时钟产生转移；优化时钟从 replay 抽样更新网络；目标网络时钟更新用于 bootstrap 的参数。每个环境步做多少次优化，和每隔多少次更新同步目标，都会改变算法。只报告学习率不足以复现实验。

![DQN 的真实环境、replay 抽样、在线与目标网络、停止梯度和定期复制组成的数据流；前向数据、梯度与参数复制用不同路径表示。](https://yingwen.io/crl-figures/concept-classic-dqn-dataflow.svg)

原创算法结构图，数值沿用本章后面的给定输出：当前预测 2、目标后继 $(2,6)$、$r=1,\gamma=0.9$，普通 DQN 标签为 6.4。迷宫只作流程背景，这些输出不是迷宫训练结果。图画普通 DQN 的 target max；Double DQN 另由在线后继 $(5,4)$ 选动作零，再取目标值 2，得到 2.8。复制以优化步计时，具体间隔 $C$ 由实现规定。

| 符号 | 含义 |
| --- | --- |
| θ / θ⁻ | 当前 Q 网络 / 目标网络参数 |
| D | 保存 transition 的有限 replay buffer |
| d / b | 真正终止标志 / bootstrap mask |
| B | 一个 minibatch 的样本数，不是时间跨度 |

<a id="lesson-derive"></a>

## 2 · 从 Bellman 目标到停止梯度的回归

$$
Q^*(s,a)=\mathbb E[R_{t+1}+\gamma\max_{a'}Q^*(S_{t+1},a')\mid s,a]
$$

已知最优价值必须满足这个固定点关系。但采样更新只能使用一个后继和一个当前估计。

$$
\begin{gathered}y_i=r_i+\gamma(1-d_i)\max_{a'}Q_{\theta^-}(s_i',a')\\L(\theta)=\frac1B\sum_i\ell\!\left(Q_\theta(s_i,a_i)-\operatorname{sg}(y_i)\right)\end{gathered}
$$

sg 表示该次优化中停止梯度。目标会在未来改变，但在当前 backward 中被视为固定标签。

若 $\ell(e)=\tfrac12e^2$，梯度为 $B^{-1}\sum_i(Q_\theta-y_i)\nabla_\theta Q_\theta$。当目标副本固定时，这是当前回归损失对在线参数的完整梯度。把标签也由当前估计生成的整体自举过程来看，它采用半梯度原则：不沿后继价值求导。若使用同一参数并对后继目标求导，就变成另一种残差优化；随机转移下，样本平方 TD 误差的期望也不同于期望 Bellman 残差的平方，后者的梯度涉及双采样问题。

还要区分评价误差与训练损失。给定关心的状态动作分布 $\mu$，$\mathbb E_\mu[(Q_\theta-Q^*)^2]$ 评价预测离真值多远；DQN 的损失比较的是 replay 样本与当下自举标签。标签不是真值、replay 不必服从 $\mu$。因此 TD loss 下降既不等于价值误差下降，也不等于策略回报提高。这是函数逼近中必须先声明分布的原因。

$$
\ell_{\mathrm{Huber}}(e)=\begin{cases}\tfrac12e^2,&|e|\le1,\\|e|-\tfrac12,&|e|>1.\end{cases}
$$

教学实现使用 Huber 损失：小误差保留二次梯度，大误差的导数幅度限制为一。它减少单个异常目标的影响，但不解决错误目标或数据覆盖不足。

限制大误差也可能改变所拟合的统计量。只考虑一个终止动作：奖励以 0.9 的概率为零、0.1 的概率为十，真实期望价值为一。平方损失的最优常数是一；上述阈值为一的 Huber 损失在 $0<q<1$ 时满足 $0.9q-0.1=0$，最优常数为 $1/9$。这是损失选择的取舍，不能把 Huber 回归无条件解释为精确的 Bellman 均值备份。

网络表示共享使一个动作的更新也能改变其他状态动作的价值。表格中的局部更新性质消失了。下面只用两个参数，算出这种联动怎样继续改变行为和自举标签。

<a id="course-neural-coupling"></a>

## 2.1 · 一次 TD 更新为什么会改变未采样状态

先固定一个学习目标 $y_t$，令 $x=(s,a)$，$g_t(x)=\nabla_\theta Q_{\theta_t}(x)$。平方损失给出的参数更新为 $\Delta\theta=\alpha\delta_tg_t(x_t)$，其中 $\delta_t=y_t-Q_{\theta_t}(x_t)$。在足够小且不跨越激活折点的平滑邻域，对另一个输入 $x$ 做一阶展开，可直接看到共享表示的作用。

$$
Q_{\theta_t+\Delta\theta}(x)-Q_{\theta_t}(x)=\alpha\delta_t\underbrace{g_t(x)^\top g_t(x_t)}_{K_t(x,x_t)}+O(\|\Delta\theta\|^2)
$$

K 是当前参数处的梯度内积。表格表示对不同条目给出零内积；共享线性特征或神经网络通常不为零。非线性网络中的 K 本身还会随学习改变。

这个内积可以为正、为负或接近零。正值表示当前更新把另一个预测推向相同方向，并不自动表示正迁移：另一个输入的正确目标可能恰好要求相反方向。负值也不自动是坏事。必须同时看两处目标，而不能只看特征相似度。

$$
Q_w(s)=w,\quad Q_w(s')=2w,\quad r=0,\quad\gamma=0.9
\quad\Longrightarrow\quad \delta=0.8w,\qquad w_{k+1}=(1+0.8\alpha)w_k
$$

反例：固定数据只反复提供 s→s′ 这条转移，后继没有独立训练目标。半梯度 TD 对任意正步长都会放大非零 w。两个真实价值可以都是零，但共享参数与数据覆盖使这一更新离零越来越远。

这个反例不是完整、充分探索的表格 on-policy 过程。如果冻结目标参数 $\bar w$ 并把内层回归充分拟合，得到的也是 $w\approx1.8\bar w$；再同步目标，外层仍会放大。固定标签的回归暂时好做，不等于反复自举的整个过程收敛。

| 结论成立的对象 | 已经知道什么 | 不能直接推出什么 |
| --- | --- | --- |
| 表格 Bellman 最优算子 | 折扣小于一时，在最大范数中压缩 | 任意神经网络梯度更新也压缩 |
| 固定特征、固定策略的线性 TD | 适当 on-policy 分布与步长条件下可分析固定点 | 特征不断变化时沿用同一个矩阵证明 |
| 固定 target 的回归 | 当前监督目标与梯度可明确写出 | 下一轮 target 或行为改变后仍改进真实回报 |

与持续学习的连接：应记录更新对其他预测的干扰，以及梯度几何怎样随时间改变。一个网络同时学习奖励价值、GVF 和模型时，误差是否下降要按每个预测问题分别检查。共享容量带来迁移，也引入耦合；它不是免费的知识共享。

<a id="lesson-shared-feature"></a>

## 2.1 · 只更新一个动作，为什么后继动作也换了？

只看 replay 中一条未终止的旧经验 $e=(s,L,0,s',0)$：在 $s$ 执行动作 $L$，奖励为零，到达 $s'$。给两个输入指定数值 $x(s)=1,x(s')=2$。网络先计算一个可学习特征 $h_w(x)=\max(0,wx)$，再读出两个动作价值；只有 $\theta=(w,v)$ 可学习，常数三固定。

$$
\begin{aligned}Q_\theta(x,L)&=v h_w(x),\\Q_\theta(x,R)&=3-v h_w(x),\\\theta_0&=(\tfrac12,1),\quad\theta^-_0=\theta_0.\end{aligned}
$$

这两个输出共用同一个特征和读出参数。这个刻意受限的小网络用于查看参数联动；它没有假定任何任务的真实动作价值之和必须为三。

取 $\gamma=1/2$。初始在线网络在 $s$ 给出 $(1/2,5/2)$，在 $s'$ 给出 $(1,2)$。目标副本也给出 $(1,2)$，因此旧经验的普通 DQN 标签为 $y=0+\tfrac12\max(1,2)=1$。我们只回归已记录的动作 $L$，损失为 $\tfrac12(Q_\theta(s,L)-1)^2$。

$$
\begin{aligned}\nabla_\theta Q_{\theta_0}(s,L)&=(1,\tfrac12),\\\nabla_\theta L(\theta_0)&=(\tfrac12-1)(1,\tfrac12)\\&=(-\tfrac12,-\tfrac14),\\\theta_1&=\theta_0-\tfrac12\nabla_\theta L\\&=(\tfrac34,\tfrac98).\end{aligned}
$$

此处 ReLU 输入为正，故 $\nabla_\theta Q_\theta(x,L)=(vx,wx)$；代入 $x=1,\theta_0$ 得到第一行。两参数梯度同时在更新前计算，步长为 $1/2$。误差幅度小于一，采用本页的 Huber 损失也得到同一步。

![同一共享 ReLU 特征从两个状态读出价值：只回归 s 的 L 动作，但 s 的 R 和后继 s′ 的两个价值都改变，后继贪心动作从 R 改为 L。](https://yingwen.io/crl-figures/concept-dqn-shared-feature.svg)

原创精确算例：箭头连接相同输入、相同动作的更新前后值；圆点表示更新前，方点表示更新后，共用 0–3 价值轴。红色箭头只标本次损失直接作用的预测；其他值随共享参数联动。计算来自 [独立数值模块](/crl-code/figures/dqn-shared-feature.mjs)，没有采样训练。

更新后 $Q_{\theta_1}(s,L)=\tfrac34\times\tfrac98=27/32$。同时，未作为回归项的 $Q(s,R)$ 从 $5/2$ 降到 $69/32$；后继的两个输出从 $(1,2)$ 变成 $(27/16,21/16)$。若下一次访问 $s'$ 时采用贪心行为，动作将从 $R$ 改为 $L$。这一选择会影响接下来产生什么经验，因而参数经由行动继续影响 replay 的内容。

目标副本在这一步仍是原来的两个参数；本例前向确定、输入固定，所以其输出仍为 (1,2)。冻结副本隔开的是在线参数到普通 DQN 标签的即时反馈；在线特征、其他动作预测和行为选择仍在改变。上例只说明一次更新的几何关系，没有用新的环境回报评价动作改变是否有益。

<a id="lesson-gradient-paths"></a>

## 2.2 · 固定标签，与让标签参与求导

用这个例子可以检查代码里究竟停掉了哪条梯度。若去掉独立副本，并把同一个 $Q_\theta$ 放到预测和目标两边，初始时 $s'$ 的最大动作仍为 $R$。在这个最大动作不变的邻域内，$y(\theta)=\tfrac12(3-2wv)=\tfrac32-wv$，其参数导数为 $(-1,-1/2)$。

$$
\begin{gathered}\nabla_\theta\tfrac12\left[Q_\theta(s,L)-y(\theta)\right]^2\\=(Q_\theta-y)\left[\nabla_\theta Q_\theta-\nabla_\theta y\right]\\=(-1,-\tfrac12).\end{gathered}
$$

与上节的 (−1/2,−1/4) 相比，这次对后继目标也求了导，得到另一个更新。数值相差两倍是本例的结果，不是一般比例。

因此，停止梯度规定的是这次损失的求导路径；目标复制规定的是之后用哪一版函数重新生成标签。若目标网已经是独立副本，且不在优化器中，即使意外为它建立梯度图，也不会自动改变在线网的梯度或更新目标参数；但会产生不必要的梯度。若两边实际共用参数，或把目标参数也交给优化器，更新语义就会改变。代码要同时检查参数是否独立、优化器持有哪些参数，以及目标前向是否停止梯度。

[标准库算例](/crl-code/tutorials/dqn_shared_feature_walkthrough.py) 用有限差分分别核对固定 $y=1$ 的损失与上述共享参数残差，得到这两组导数。这个检查验证求导对象是否写对；非线性自举控制的收敛还需要另外的条件，不能从一个样本的 loss 下降推出。

<a id="lesson-double"></a>

## 3 · Double DQN：把选择和评价分开

如果两个动作的真实价值相同，而估计误差有噪声，max 倾向挑中被高估的那个。DQN 在目标网络里同时选动作和估值。Double DQN 用当前网络选择动作，再让目标网络评价这个被选中的动作。

手算噪声即可看见选择偏差：两个真值都为零，各估计独立地以同等概率误差 $-1$ 或 $+1$。四组等概率结果的最大值为 $(-1,1,1,1)$，平均为 $1/2$，不是零。若选择网与评价网的误差完全独立，给定被选动作后评价误差均值仍为零，因而这个玩具情形的 Double 评价无偏；若两网误差相同，则又回到 $1/2$。实际网络介于这些统计关系之间，拆开选择与评价不是独立性的证明。

$$
a_i^*=\arg\max_a Q_\theta(s_i',a),\qquad y_i^{\mathrm{Double}}=r_i+\gamma(1-d_i)Q_{\theta^-}(s_i',a_i^*)
$$

不是对两个 Q 值取 min，也不是交替更新两个完全独立的网络。当前网与目标网有关联，因此它并不使过估计严格归零，也可能产生低估。

回到同一条旧经验。更新前，两网都在 $s'$ 选择 $R$，DQN 和 Double 的标签都为一，所以前面那一步也可作为两种方法的共同第一步。在线网更新后改选 $L$，冻结副本对 $L$ 的评价仍为一。此时普通 DQN 继续取副本最大值二，标签仍为一；Double 改取副本的 $L$ 输出，标签变成 $1/2$。没有新增经验，目标副本也没有更新，变化来自“评价哪一个动作”。

![同一条旧经验在更新前、在线更新后、目标复制后的三次标签计算。普通 DQN 标签依次为 1、1、27/32，Double 标签依次为 1、1/2、27/32。](https://yingwen.io/crl-figures/concept-dqn-target-relabel.svg)

原创精确计算，所有面板始终使用 $e=(s,L,0,s',0)$ 与 $\gamma=1/2$。每行两根等比例柱为 $s'$ 的 L/R 价值，橙色环指出选中的动作；“Double”环按在线网排序，却读取目标副本的柱高。最后一格只复制参数、不再执行梯度步。图中的三个标签是三次重算的结果，不是把已经计算好的数值追溯改写。

现在执行硬拷贝 $\theta^-\leftarrow\theta_1$。副本在 $s'$ 的输出变成 $(27/16,21/16)$，两种标签均重新算为 $27/32$。奖励零和原来的后继输入都没变，改变的是从那个后继继续行动的估计。普通 replay 保存转移事实，并不把 bootstrap 标签永久存成真值。若继续训练，需在下一次抽到它时用当时的网络重新计算。

目标网络降低标签在连续梯度步之间的变化速度。Replay 将连续经验重新抽样，提高数据复用并减弱相邻样本的相关性。二者没有把数据变成来自真实分布的独立样本，也没有消除策略导致的覆盖偏差。

<a id="experiment-deep-double_dqn"></a>

### 实验：实验 · 分开选动作与评价动作的 Double DQN

同一网络、回放和目标同步下，更换 bootstrap 的动作选择方式会怎样？

**环境与可用信息。** DeadlineChain：位置 0–4、左右动作、边界截断。观测为五维位置 one-hot 加剩余时间比例。到位置 4 得 1 并终止；其他步得 −0.02。12 步截止也是任务真实终止，且剩余时间可观测。每回合重新从位置 0 开始，网络跨回合保留。

**设置。** 1200 个真实步。两者用 6→32 tanh→2 网络、相同 seed 的 PyTorch 默认随机初始化和复制的目标网络；Adam 0.003，γ=0.99，Huber 损失，梯度范数上限 10。replay 容量 4000、批量 32，每两步一次更新，每 100 步硬同步。ε 从 1 降到最低 0.05。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** double_dqn.py 用在线网络选择后继动作，用目标网络评价该动作；dqn.py 对目标网络直接取最大。两者都停止整个 critic target 的梯度。

**测量。** 每 60 步冻结网络，在隔离环境执行 12 个贪心回合。图中是未折扣环境回报；训练 target 使用 γ=0.99。此图没有直接测量 Q 高估偏差。

```bash
python3 implementations/deep/double_dqn.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-figures/result-deep-double_dqn.svg)

横轴：训练环境步（独立评估交互另计）。纵轴：冻结策略的外部回报。每种方法 1200 训练环境步；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 冻结当前网络，以 argmax 选择动作，在独立 DeadlineChain 中跑12次完整回合，取未折扣外部回报的平均。这里起点相同且评价过程确定，12次并非12个独立训练种子。

**step：怎样计时。** step 只数训练环境转移。评价交互另计；任务中的12步期限是真正终止条件。

**怎样汇总。** 取该记录时刻的值；不先对曲线上的时间点求平均。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** CSV 的 phase=evaluation。可以重算冻结评估曲线；未保存逐步训练奖励，不能从这些评估点反推训练全程收益。

计算位置：[deep/dqn.py](https://yingwen.io/crl-code/implementations/deep/dqn.py) · [deep/double_dqn.py](https://yingwen.io/crl-code/implementations/deep/double_dqn.py) · [deep/_common.py](https://yingwen.io/crl-code/implementations/deep/_common.py)

</details>

**结果分析。** 第 600 步两者平均冻结回报都是 0.94；第 1200 步 Double DQN 为 0.94，DQN 为 0.936。短链上两者几乎打平，这个小差异不能证明普遍优势或已解决高估。

**结论边界。** 任务是可观测有限时域小链，不是 Atari。网络是 tanh MLP，不是卷积网络；模型大小、回放次数及更新比固定。真正的 deadline 终止与外部时间截断不能混用。

**继续实验。** 先对同一小批次打印选中动作和两种 target，再在相同模型下测对解析 Q 的误差。只比较环境回报，能否判断差异来自高估？

[源码](https://yingwen.io/crl-code/implementations/deep/double_dqn.py) · [逐种子记录](https://yingwen.io/crl-code/results/deep-double_dqn/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/deep-double_dqn/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/deep-double_dqn/curves.json)

<a id="course-replay-time"></a>

## 3.1 · Replay、target 和表示分别改变哪一个问题

$$
L_t(\theta;\bar\theta)=\mathbb E_{(s,a,r,s')\sim\nu_t}
\frac12\left[Q_\theta(s,a)-\operatorname{sg}\!\left(r+\gamma\max_{a'}Q_{\bar\theta}(s',a')\right)\right]^2
$$

这里先省略终止 mask。ν 是抽样分布，横跨哪些历史经验；bar θ 决定自举标签；θ 决定可表示的函数与更新几何。这三者必须分别说明。

| 设计选择 | 直接改变的对象 | 平稳任务中的作用 | 持续任务中的代价 |
| --- | --- | --- | --- |
| Replay 大小与抽样规则 | ν：训练哪些经验、各占多少权重 | 复用数据，改善覆盖，改变时间相关性 | 旧奖励或旧动力学可能反复进入目标 |
| Target 同步或 Polyak 系数 | 标签生成器及其滞后 | 减慢短期自举反馈 | 旧价值需要时间才能反映新经验 |
| Encoder 与归一化器更新 | 表示坐标和梯度内积 | 学习更有用的特征与尺度 | 旧模型、旧隐状态和旧优化统计可能失配 |
| 每交互步的梯度步数 | 优化时钟相对经验时钟的速度 | 更充分利用有限数据 | 可能更快拟合旧分布，而非更快适应世界 |

手算：单状态单动作，自循环，$\gamma=0.9$。奖励从 $+1$ 变为 $-1$。当前问题的正确价值是 $-1/(1-0.9)=-10$。若 replay 恰有一半旧奖励和一半新奖励，平均目标满足 $q=0+0.9q$，得到零。回归可以拟合得很稳定，却仍不回答当前问题。目标网络不校正这种数据失配。

旧行为策略与旧世界也不同。在同一平稳 MDP 中，Q-learning 可以使用其他策略收集的转移，但仍需要覆盖与相应收敛条件。世界的奖励或转移已经改变时，旧转移连条件分布都不同。动作重要性比只校正行为概率，不能把旧动力学变成新动力学。

优先回放还有一层分布选择。设当前 buffer 有 N 条经验，第 i 条被抽中的概率为 $q_i>0$，其当前半梯度为 $g_i$。大 TD error 可以让一条经验更常被训练，但它也可能来自不可预测的奖励噪声，不能等同于可获得的学习进展。

$$
\mathbb E_{i\sim q}\!\left[\frac{g_i}{Nq_i}\right]=\frac1N\sum_{i=1}^{N}g_i
$$

条件于当前 buffer、参数与抽样概率，权重 $1/(Nq_i)$ 恢复均匀 buffer 更新的期望。常用 $(Nq_i)^{-\beta}$ 在 $\beta<1$ 时只作部分校正；计算此次梯度时权重固定。

这个等式没有恢复当前策略的占据分布，也没有替换旧奖励或旧动力学。把两个样本以 0.9 和 0.1 的概率抽取时，未校正平均是 $0.9g_1+0.1g_2$；完全校正后才是各占一半。抽样、校正与 buffer 保留哪些经验，因而是三个不同选择。

在流式协议中，当前样本用完即丢弃，不代表算法必须遗忘所有信息。参数、资格迹、归一化统计和元参数仍可递推保存。去掉 replay 改变数据访问权限；还必须重新设计单步尺度、信用分配和变化检测。可先比较相同交互预算与相同计算预算的两组对照，不把二者混为一种“样本效率”。

<a id="lesson-algorithm"></a>

## 4 · 一次交互与一次优化的精确次序

**算法：算法伪代码**

1. 初始化 $Q_\theta$，复制 $\theta^-=\theta$，清空 replay。
1. 每个环境步：
  1. 用当前 $Q_\theta$ 的 $\epsilon$-greedy 行为选动作。
  1. 获取真实的 $(s,a,r,s',d)$，再决定是否重置环境。
  1. 将原来的 $s'$ 写入 replay，不能写成 reset 后的状态。
  1. 样本足够时，抽取 batch，先计算所有停止梯度的目标。
  1. 用 gather 取出当前动作预测，反传 loss，只更新 $\theta$。
  1. 达到目标同步时钟后，复制或平滑更新 $\theta^-$。
  1. 用独立环境评估，不把评估转移放进训练 replay。

循环中的“步”需要带上计数器。记 $t$ 为已经产生的环境转移数，$k$ 为已经完成的梯度更新数；一条经验 $e_\tau$ 产生于环境时刻 $\tau$，在时刻 $t$ 的年龄是 $t-\tau$ 个环境步。目标副本还保存一次同步时的在线参数版本。于是同一条旧经验可以在不同 $k$ 被重复使用，也可以在不同目标版本下得到新的 bootstrap 标签。年龄是两个环境时刻之差，不是第四个独立递增的优化计数器。

![DQN 的经验生成、同一旧样本的年龄、在线梯度更新与目标硬拷贝在不同事件上发生。](https://yingwen.io/crl-figures/concept-core-update-clocks-dqn.svg)

上图横向、手机版纵向排列环境事件，省略段明确标出，距离不代表相同时间间隔。按本页 train_dqn 的循环条件解析生成调度；蓝色抽样箭头指定相应 batch 含第 1 步的经验 $e_1$，是一条可发生的抽样路径，并非训练日志。网络下标是已完成的梯度步数，不是 Q 值。目标复制机制见 [Mnih 等 Methods / Algorithm 1](https://deepmind-media.storage.googleapis.com/dqn/DQNNaturePaper.pdf#page=7)；[调度计算代码](/crl-code/figures/core-update-clocks.mjs)。

具体看配套实现：循环索引从零开始；积累至少 32 条经验后，只在偶数索引优化。因此第 33 个环境转移后做第 1 次梯度更新，第 35 个之后做第 2 次，第 99 个之后做第 34 次。第 34 步虽然不优化，却仍产生新经验，旧样本 $e_1$ 的年龄也从 32 增至 33。样本多久以前产生，与网络已经从它或其他样本学过多少次，是不同的问题。

本页训练代码每 100 个环境步硬拷贝一次目标。第 100 步本身不优化，复制的是此时已有的 $\theta_{34}$；第 99 步的更新仍使用初始副本 $\theta_0$，第 101 步的第 35 次更新才开始使用 $\theta_{34}$。开头数据流图以每 $C$ 个优化步同步作另一种示意约定，实施时必须明确采用哪一个计数器。增加每条经验对应的优化次数，会改变两种约定下目标的实际滞后。

以图中重复抽到的 $e_1=(s_1,a_1,r_1,s'_1,d_1)$ 为例，奖励和终止标记一直来自原记录。在第 $k$ 次更新时，先用当前目标副本重新形成 $y_1^{(k)}=r_1+\gamma(1-d_1)\max_a Q_{\theta^-}(s'_1,a)$，再用更新前的 $Q_{\theta_{k-1}}(s_1,a_1)$ 计算梯度，得到 $\theta_k$。固定样本与固定副本时，普通 DQN 的标签不随在线梯度步改变；硬拷贝之后它才可能改变。Double DQN 还用在线网络选后继动作，即使副本未变，所选动作改变也可能改变标签。两种情况下，当前反传均不穿过标签。

真正终止时 $b=1-d=0$。日志窗口或人工采样上限不是终止，通常仍需从截断前的最后观测 bootstrap。有限时域任务的最后一步可以是真终止，但若剩余时间会影响最优决策，它应作为状态输入。本页小环境有可观察的十二步期限，期限耗尽属于任务终止。

<a id="lesson-example"></a>

## 5 · 一个 batch 元素的手算

设奖励为 $1$，折扣为 $0.9$。后继当前网络输出为 $(5,4)$，目标网络输出为 $(2,6)$。DQN 目标是 $1+0.9\times6=6.4$。Double DQN 先选择动作零，所以目标是 $1+0.9\times2=2.8$。若该转移真正终止，两者目标都退化为 $1$。

若当前动作预测为 $2$，Double 目标下误差为 $-0.8$，Huber 梯度系数也是 $-0.8$，下降步骤会提高该预测。DQN 目标下误差为 $-4.4$，Huber 梯度系数截到 $-1$。这是损失梯度的限制，不是把目标值限制为某个范围。

独立计算 DQN/Double 目标与两类 mask。

```python
def dqn_target(reward, gamma, terminated, next_online, next_target,
               double=True):
    if not next_online or len(next_online) != len(next_target):
        raise ValueError('matching nonempty action vectors required')
    if terminated:
        return reward
    if double:
        selected = max(range(len(next_online)), key=next_online.__getitem__)
        continuation = next_target[selected]
    else:
        continuation = max(next_target)
    return reward + gamma * continuation


def masks(terminated, boundary):
    # boundary includes terminal, timeout, or the end of this rollout.
    return float(not terminated), float(not (terminated or boundary))
```

<a id="lesson-code"></a>

## 6 · 实际 PyTorch 更新与完整小任务训练

先下载 [dqn_shared_feature_walkthrough.py](/crl-code/tutorials/dqn_shared_feature_walkthrough.py)，执行 `python3 dqn_shared_feature_walkthrough.py` 查看三次快照；执行 `python3 dqn_shared_feature_walkthrough.py --test` 核对精确分数、有限差分、参数副本与终止退化情形。单文件只用 Python 标准库。

| 手算中做的事 | 完整实现中的位置 |
| --- | --- |
| 先生成当次标签 | dqn_update 的 no_grad：普通 max，或 online argmax 后 target gather |
| 对已记录动作回归 | online(x).gather → smooth_l1_loss → backward → optimizer.step |
| 产生和重用经验 | train_dqn 的真实 env.step、replay.append 与 random.sample |
| 让下一次旧样本使用新版估计 | train_dqn 的 target.load_state_dict |

标准库例子把一次 SGD 写到可手算的程度；下面完整工程采用小 MLP、Adam、梯度范数限制和实际交互。两者的更新顺序可逐项对应，具体参数轨迹不相同。

batch 维度始终为 B；目标处 no_grad；gather 后去掉单维。

```python
def dqn_update(online, target, optimizer, batch, gamma=.99, double=True):
    observations, actions, rewards, next_observations, terminated = zip(*batch)
    x, xp = torch.stack(observations), torch.stack(next_observations)
    action = torch.tensor(actions, dtype=torch.long)
    reward, terminal = torch.tensor(rewards), torch.tensor(terminated, dtype=torch.float32)
    prediction = online(x).gather(1, action[:, None]).squeeze(1)
    with torch.no_grad():
        target_values = target(xp)
        if double:
            selected = online(xp).argmax(dim=1)
            tail = target_values.gather(1, selected[:, None]).squeeze(1)
        else:
            tail = target_values.max(dim=1).values
        y = reward + gamma*(1-terminal)*tail
    assert prediction.shape == y.shape == (len(batch),)
    loss = nn.functional.smooth_l1_loss(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    return float(loss.detach())
```

完整交互、replay 抽样、优化、目标同步及独立 greedy 评估。

```python
def train_dqn(steps=2000, seed=0, double=True):
    random.seed(seed)
    torch.manual_seed(seed)
    online = mlp(6, 2)
    target = copy.deepcopy(online).requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=.003)
    replay = deque(maxlen=4000)
    env, initial_score = DeadlineChain(), evaluate(online)
    observation, losses = env.reset(), []
    for t in range(steps):
        epsilon = max(.05, 1-t/max(1, steps*.7))
        with torch.no_grad():
            action = random.randrange(2) if random.random() < epsilon else int(online(observation).argmax())
        next_observation, reward, terminated = env.step(action)
        replay.append((observation, action, reward, next_observation, terminated))
        observation = env.reset() if terminated else next_observation
        if len(replay) >= 32 and t % 2 == 0:
            batch = random.sample(list(replay), 32)
            losses.append(dqn_update(online, target, optimizer, batch, double=double))
        if (t+1) % 100 == 0:
            target.load_state_dict(online.state_dict())
    return {'algorithm': 'Double DQN' if double else 'DQN', 'seed': seed,
            'environment_steps': steps, 'updates': len(losses),
            'initial_greedy_return': initial_score, 'final_greedy_return': evaluate(online),
            'last_loss': losses[-1] if losses else None, 'scope': 'DeadlineChain only'}
```

将 deep_textbook_lab.py 与 deep_textbook_train.py 放在同一目录。执行 python3 deep_textbook_train.py dqn --steps 2000 --seed 0；加 --ordinary-dqn 切换普通 DQN。内置 DeadlineChain 使用小 MLP，不下载数据，不使用 Gym 或 GPU。DQN 用折扣 0.99，显示的 greedy return 是便于读数的未折扣任务收益；它不是跨算法公平评测报告。

<a id="lesson-branches"></a>

## 7 · 失败条件与进入 CRL 的接口

- 形状错误：预测为 [B,1] 而目标为 [B] 会广播成 [B,B]；loss 仍能下降，却优化了错误配对。
- 目标泄漏：缺少 detach 会改变求导对象；把 reset 后观测存入旧 transition 会虚构动力学。
- 数据陈旧：环境改变后，replay 仍保存当年的奖励和转移。重新计算 bootstrap 标签不会改写这些事实；若要按当前奖励重新标记，必须另有可用的奖励规则。
- 表示干扰：目标网络不能防止已有特征失去可塑性；增加更新/样本比也可能放大过拟合。

原始 Atari DQN 还裁剪环境奖励，这与 Huber 限制误差导数不同。若两条两步路径的奖励分别为 $(100,0)$ 和 $(1,1)$，$\gamma=0.9$ 时原始收益为 100 与 1.9；把奖励裁到 $[-1,1]$ 后变为 1 与 1.9，最优选择反转。奖励变换可能帮助统一数值尺度，也改变所学目标；报告原始游戏得分时，应同时说明训练奖励。

教材 §16.5 强调，Atari 案例在各游戏中采用相同结构和超参数，但每个游戏重新初始化并分别学习。这证明同一学习设计能适用于不同问题，并未证明一个网络在这些游戏上依次学习时能保留旧技能并持续学会新技能。后一个要求才把表示共享、经验保留与可塑性放进同一持续学习问题。

研究持续深度价值学习时，可先固定网络和数据，分别改变 replay 的年龄分布、目标更新速度与特征回收。若三个模块同时改变，回报差异无法说明是哪一种机制有效。

<a id="lesson-check"></a>

## 8 · 练习与答案

- 练习：在共享特征例子中固定 $w=1/2$，只更新 $v$，其余条件不变。答：$v=9/8$，$Q(s,L)=9/16$，$s'$ 输出为 $(9/8,15/8)$，仍选 $R$，故目标副本未同步时 Double 标签仍为一。可学习特征的改变参与了原例中的动作翻转。
- 练习：在线参数已经更新为 $(3/4,9/8)$，但目标尚未同步。下一次使用这条旧经验时，两种方法的误差 $Q-y$ 各是多少？答：普通 DQN 为 $-5/32$，Double 为 $11/32$，对当前预测的更新方向相反。这是标签选择的后果，尚未说明哪个方向更接近真实价值。
- 练习：为什么复制后本例的 TD 误差恰为零？答：后继选择 $L$ 时，$x(s')=2x(s)$ 且 $\gamma=1/2,r=0$，使目标等于当前预测。这个代数巧合没有检验其他转移，不能当作学会任务的证据。

- 问：训练窗口在第 100 步结束，但任务没有结束，bootstrap mask 是多少？答：一。需用真正最后观测的价值；不能把采样边界误写为终止。
- 问：目标网络当前输出为 (2,6)，Double 一定取 6 吗？答：不一定。当前网络负责选择，例子中选择了目标值为 2 的动作。
- 问：replay 与 target 哪个解决长期遗忘？答：二者都不提供一般保证。Replay 保存部分数据，target 降低短期标签变化；保留能力还依赖覆盖、容量和更新规则。
- 实验：把目标同步间隔改为 1、100、1000，在相同交互预算记录 TD loss、目标变化和回报。解释过快与过慢同步各自可能带来的问题，而不是只选最好种子。

## 从本章进入实践

[预测与控制](https://yingwen.io/zh/continual-rl/code/#practice-prediction)：学会预测更多事情，什么时候会改变行动？



<a id="chapter-code"></a>

## 下载与运行

CPU DeadlineChain 完整训练；需要同目录 deep_textbook_lab.py 和 PyTorch。

[下载 deep_textbook_train.py](https://yingwen.io/zh/continual-rl/download/deep_textbook_train.py)

```sh
python3 deep_textbook_train.py dqn --steps 2000 --seed 0
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning, second edition](http://incompleteideas.net/book/the-book-2nd.html)：§9.3 pp.201–203：标签依赖与半梯度；§16.5 pp.439–440：共享特征、replay 与固定目标副本。

- [Mnih et al. · Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)：Methods 的损失、Training algorithm 与 Algorithm 1：replay、延迟目标和误差裁剪。

- [DeepMind · 原始 DQN NeuralQLearner.lua](https://github.com/google-deepmind/dqn/blob/master/dqn/NeuralQLearner.lua#L167)：getQUpdate 生成目标，qLearnMinibatch 只对在线网 backward，perceive 按计数复制目标。原作者 Lua/Torch 工程。

- [van Hasselt et al. · Double DQN](https://arxiv.org/abs/1509.06461)：2015 预印本 v3 / AAAI 2016：§Double DQN 分开在线选择与目标评价，保留定期复制。

- [Elsayed et al. · Streaming Deep Reinforcement Learning Finally Works](https://arxiv.org/abs/2410.14606)：Stream-X 将神经预测与控制放回逐样本更新；应逐项检查归一化、初始化、资格迹和步长控制，不把移除 replay 当作完整算法。

- [Schaul et al. · Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)：§3.3–3.4：抽样概率与重要性权重；校正目标是当前 buffer 的均匀经验分布，不是任意当前世界分布。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-deep-value#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-deep-value#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-deep-deep-value)
<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 预测的量是什么，误差又是什么？

先固定策略。价值是回报的条件期望。MC 使用完整回报；TD 用下一时刻的预测替代未观察的余项。两者不能仅按同一批样本上的 TD error 排序。

函数逼近与深度方法：共享参数限制了可表示的函数。采样权重决定在哪些状态上拟合。最小价值误差、最小 Bellman 残差和 TD 固定点一般不同；神经网络又使可表示的局部方向随参数改变。

持续学习中的研究问题：多个 GVF 共用表示时，哪些预测值得占用容量？应分别检查问题定义是否改变、数据是否覆盖，以及回答该问题的误差是否降低。

[MC 与 TD](https://yingwen.io/zh/continual-rl/foundations/tabular/temporal-difference/) → [投影与半梯度](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/) → [神经价值更新](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/) → [GVF 的问题与答案](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)


### 可进一步检验的问题

- [04 · 旧价值什么时候应该复用，什么时候应该快速改写？](https://yingwen.io/zh/continual-rl/research/#research-continual-control)：Replay 中的样本年龄与目标网络的参数版本会延缓价值改写；环境变化后，需要判断这些滞后何时有益、何时妨碍适应。
- [06 · 不存 replay、每步只处理新经验时，怎样避免更新失稳？](https://yingwen.io/zh/continual-rl/research/#research-streaming-stability)：DQN 的采样、回放与优化时钟明确了数据复用权限；撤掉 replay 后，需要重新检查相关性和每步更新幅度。
- [12 · 怎样保留旧能力，而不把过时知识强加给新任务？](https://yingwen.io/zh/continual-rl/research/#research-retention-transfer)：Replay 保存旧样本，目标网络保存较慢变化的函数副本；这些保存对象不同，不能用一个遗忘分数解释所有作用。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [深度价值学习](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)
- [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)
- [可塑性与特征更新](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)

对应原始材料：Sutton & Barto §6.5；§9.2–9.3：预测目标与半梯度；§11.3：离策略风险；§16.5：DQN 的实验条件。本文为原创讲解，原书、论文与上游代码保留各自许可。
