# 持续控制与平均奖励

函数逼近与经典进阶方法 · 第 3 章

智能体没有自然回合终点时，怎样定义和学习长期控制目标？

## 本章内容

- 区分折扣价值、平均奖励率与差分价值。
- 从 Poisson 方程得到差分 TD 和 Sarsa 的两组同步更新。
- 在同一服务站中手算真实时钟下的奖励率、启动 bias 和一次时长归一化更新。
- 说明 unichain、communicating 与函数逼近保证的边界。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [MDP、回报与价值：序列决策的数学对象](https://yingwen.io/zh/continual-rl/foundations/tabular/mdps/)：区分任务目标、回报和策略价值。
- [特征、泛化与半梯度控制](https://yingwen.io/zh/continual-rl/foundations/approximation/features-control/)：理解半梯度控制，再把折扣目标换为长期奖励率。


### 持续交互

任务本身不要求在某个时间终止。训练日志可以分段，但分段不等于环境终止，也不自动提供无成本重置。

### 差分价值

先扣除长期奖励率，再比较从不同状态出发的暂态优势。差分价值允许任意共同加法常数，需要参考状态或其他归一化约定。

<a id="problem-definition"></a>

## 本章的问题定义

交互没有自然终点，评价单位原始时间内的长期收益，而非折扣累计量。

### 给定条件与符号

- 平稳 MDP、策略类 $\Pi$、每一步原始奖励 $R_{t+1}$。
- 平均奖励 $g_\pi$ 与差分价值 $h_\pi$；近似动作价值使用 $q_w(s,a)$。

### 需要求解的对象

在规定策略类中提高平均奖励，同时估计相对未来价值。

### 信息与数据权限

只有连续样本流；人为日志分段不产生新的问题终止。

$$
\sup_{\pi\in\Pi}g_\pi,\qquad g_\pi=\lim_{T\to\infty}\frac1T\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}R_{t+1}\right]
$$

本章先讨论极限存在且与起点无关的情形。差分价值刻画有限暂态与长期奖励率的偏差，不是发散的无折扣总回报。

### 成立条件与解的含义

- 策略诱导过程需具有保证平均奖励定义适用的链结构。
- 差分价值只确定到加法常数，需要参考规范。

判断准则：真实奖励率、学习器内部奖励率估计、差分 Bellman 残差分别报告。

### 适用边界

- 用渐近奖励率概括所有有限生命期损失或不可逆风险。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)：本章从一般准则中选取极限存在且与起点无关的平均奖励设定；它增加适用条件，而非推广所有准则。

- 限制表示或采用近似 · [平均奖励与差分价值](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)：本章用共享特征近似差分动作价值并采用差分 Sarsa；相关平均奖励章还讨论表格预测、离策略与规划。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

无折扣累计回报可能发散，价值的共同偏移也不可辨识。

### 本章的核心思路

每步减去奖励率，学习相对价值，再把奖励率估计与价值更新耦合。

1. [从长期速率推导差分方程](#lesson-derive)：先定义中心化的回报对象，避免对发散总和直接求值。

2. [确定两个估计的更新时序](#average-algorithm)：差分 Sarsa 同时更新价值和奖励率；两者不应被误认成独立真值。

3. [检验目标是否真正相同](#average-objective)：构造折扣与速率排序不同的例子，阻止把一种目标的实验结论外推到另一种。

结论与条件：链结构和函数逼近条件决定可用结论；表格定理不能自动保证近似控制。

### 相关方法改变了什么

- 平均奖励 / 折扣回报：前者强调长期速率，后者为相同时间距离设置几何权重。

- 差分更新 / 折扣奖励中心化：有限折扣下的中心化参照不一定等于真实奖励率。


<a id="lesson-setting"></a>

## 1 · 长期速率与折扣和是不同目标

仓储机器人一天接一天地执行任务，不一定存在一个自然终点。在这种问题中，可以继续采用折扣回报，也可以直接问“长期每单位时间完成多少工作”。后一问题引入平均奖励。它强调持续运行的速率，同时让有限的启动代价在无限时间平均中消失；因此选择这一目标之前，必须确认这种取舍符合任务。

函数逼近使这个选择更重要。共享参数可能让一个状态的改善伴随另一状态的退步，不能只要求“每处都变好”就完成策略比较。我们必须说明：特别关心某个起点，还是按智能体长期访问的状态来评价？这两种加权方式，会给折扣带来不同的含义。

$$
g_\pi(s)=\lim_{T\to\infty}\frac1T\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}R_{t+1}\mid S_0=s\right].
$$

平均奖励率衡量每个真实时间步的长期收益。存在此极限以及是否依赖初始状态，都是问题的条件，而不是符号定义自动提供的性质。

固定策略若具有一个常返类、其余状态至多是暂态，通常称为 unichain。在有限状态、适当奖励条件下，不同初始状态最终进入同一常返类，长期奖励率可以相同。若策略产生多个互不连通的常返类，奖励率可能依赖起点，不能用一个标量概括所有状态。

Communicating 描述 MDP 层面的可达性：不同状态之间存在某些策略可以建立到达关系。它不等于每一个策略都只有一个常返类。相关最优平均奖励结果还需要区分策略类、探索条件和算法假设。实践中先检查有没有不可逆动作和吸收陷阱。

$$
\begin{aligned}v_\gamma^\pi(s)&=\mathbb E_\pi\sum_{t\ge0}\gamma^tR_{t+1},\\ g_\pi&=\lim_{\gamma\uparrow1}(1-\gamma)v_\gamma^\pi(s).\end{aligned}
$$

极限关系在适当有限遍历条件下成立，但固定 $\gamma<1$ 仍定义另一个目标；把它设得很接近 1 也会放大价值尺度并延长有效预测时域。

<a id="lesson-derive"></a>

## 2 · 从长期速率到差分 Bellman 方程

平均奖励率只描述长期斜率，无法区分获得同一速率但暂态收益不同的状态。差分价值补充这种信息。为避免在周期链上把一个不收敛的普通无限和当作定义，本章直接用 Poisson 方程定义差分价值，并固定一个参考值。

$$
g_\pi\mathbf1+h_\pi=r_\pi+P_\pi h_\pi,\qquad h_\pi(s_{\rm ref})=0.
$$

在适当 unichain 条件下，$g_\pi$ 是公共奖励率，$h_\pi$ 在加法常数之外确定。式子表示：当前位置的暂态优势，等于当前奖励超出平均的部分，加上下个状态的优势。

$$
\delta_t=R_{t+1}-\bar r_t+x_{t+1}^\top w_t-x_t^\top w_t.
$$

用在线估计 $\bar r_t$ 替代未知 $g_\pi$，再用线性近似替代 $h_\pi$，得到样本残差。这里不是在折扣 TD 中简单令 $\gamma=1$；新增的奖励率项负责去掉随时间累积的趋势。

差分价值的数值可以为负。它不表示负的长期收益，只表示相对参考状态的暂态劣势。将所有差分价值加上同一常数不会改变 TD 残差，也不会改变按动作价值差选择动作的结果。原始参数均值漂移和价值差失真应分别诊断。

$$
\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}R_{t+1}\mid S_0=s\right]=Tg_\pi+h_\pi(s)-\mathbb E_\pi[h_\pi(S_T)\mid S_0=s].
$$

对 Poisson 方程沿轨迹取条件期望并求和，中间的差分价值项相消，得到有限 T 的精确恒等式。有限状态下 h 有界，除以 T 后两端的差分价值贡献消失，只剩长期奖励率。

这条等式解释了 gain 与 bias 的分工。gain 是长期累计收益的斜率；bias 是由起点和到达过程造成的有界修正。gain-optimal 只要求最大化斜率。同样的长期率可以容许完全不同的前期损失。若要进一步比较这些策略，就需要偏差最优、有限寿命目标或另外给定的约束，而不是声称平均奖励已经评价了所有时间尺度。

例如一个决策态可选择“立即获得1并进入每步奖励1的吸收循环”，也可“先付出10再进入同一循环”。两策略的 gain 都是1。令循环状态 h=0，则决策态的 bias 分别为0与−11。无论从哪一个较长但有限的时限看，后一策略都少11；单看极限奖励率则把它们并列。这个例子不需要非平稳性，就足以说明策略评价准则必须先说清。

<a id="average-rate-cycles"></a>

### 2.1 · 同一个服务站：决策次数与真实秒数

先在一个持续运行的服务系统中把 gain 与 bias 都算出来。系统只在启动时从 E 进入服务站 H，耗时 2 秒、奖励 −3。以后每次回到 H，都能选择快环或慢环，分别经 A 或 B 完成作业；回站是世界自身的转移。所有奖励在对应动作结束时到账，执行途中没有额外奖励或决策。

| 当前位置与选择 | 后继状态 | 累计奖励 R | 持续时间 τ（秒） |
| --- | --- | --- | --- |
| E：启动 | H | −3 | 2 |
| H：快环 | A | 0 | 1 |
| A：返回 | H | 4 | 1 |
| H：慢环 | B | 0 | 1 |
| B：返回 | H | 9 | 3 |

一次快环需要两次转移、2 秒，获得 4；一次慢环同样需要两次转移，却要 4 秒，获得 9。固定策略 $\pi_F$ 每次选快环，$\pi_L$ 每次选慢环。启动只出现一次，所以它不改变无限时间平均的斜率：

$$
g_F=\frac{0+4}{1+1}=2,\qquad g_L=\frac{0+9}{1+3}=\frac94\quad\text{奖励/秒}.
$$

这是一种半马尔可夫描述：在决策时刻记录状态、总奖励与耗时。把长动作展开为每秒一次的倒计时状态，就能得到同一个真实过程的普通 MDP。

看图前先预测：两条路线都恰好执行两次动作，按每次动作的奖励平均，能否直接得到每秒收益？再试一种看似已经修正时长的做法：先算每次动作的奖励除以秒数，再把这两个数等权平均。

![启动E经两秒到H；快环H到A再到H为两秒得4，慢环H到B再到H为四秒得9，轨迹长度使用同一秒数坐标。](https://yingwen.io/crl-figures/average-rate-walkthrough-cycles.svg)

沿横轴读真实秒数；每个实心小圆点代表 1 奖励，奖励在终点到账。两个循环的决策次数相同，真实耗时不同。原创确定性算例；[计算核](/crl-code/figures/average-rate-walkthrough.mjs)与[精确分数教程](/crl-code/tutorials/average_rate_walkthrough.py)使用同一转移表。

$$
\underbrace{\frac{0/1+4/1}{2}}_{\text{快环：逐次率平均}}=2,\qquad\underbrace{\frac{0/1+9/3}{2}}_{\text{慢环：逐次率平均}}=\frac32.
$$

后一种算法错误地偏爱快环。慢环的返回动作占了 3/4 的时间，却只得到 1/2 的平均权重。真实总奖励除以真实总时长，才得到 $9/4$。按动作平均奖励则为 2 和 9/2，单位是奖励/决策，也不是所需目标。

随机选择两条路线时也要保留这个分母。每次到 H 各以 1/2 的概率选择，平均一圈奖励为 $(4+9)/2$，平均一圈时间为 $(2+4)/2$，所以 $g=13/6$；直接平均两条路线的率却是 $17/8$。较长路线在真实生命中占据更大的时间份额。

<a id="average-rate-bias"></a>

### 2.2 · 扣去等待期间本可获得的收益

长期率已经知道，但从 A、B 或尚未完成启动的 E 出发，接下来会遇到不同的等待与奖励。令 $n$ 数高层转移，$T_n$ 数真实秒，$\tau_n=T_{n+1}-T_n$。在一次转移期间，基准收益应为 $g_\pi\tau_n$，因而单位时间的 Poisson 方程变为：

$$
h_\pi(s)=\mathbb E_\pi[R_n-g_\pi\tau_n+h_\pi(S_{n+1})\mid S_n=s],\qquad h_\pi(H)=0.
$$

$R_n$ 是这整个动作的奖励，不是每秒奖励；$h$ 的单位是奖励，$g$ 的单位是奖励/秒。这里以回站 H 为共同参考，用 Poisson 方程定义 bias。固定策略下的循环有周期，不假设普通无限中心化回报收敛。

先评价快环。A 返回 H 得 4、花 1 秒，所以 $h_F(A)=4-2=2$；从 B 返回虽然得 9，但占用 3 秒，所以 $h_F(B)=9-2\times3=3$。E 启动要付出 3，并失去 2 秒按基准工作的机会，因此 $h_F(E)=-3-2\times2=-7$。把这些数代回 H 的方程，得到 $0=0-2+2$。

| 固定策略 | g（奖励/秒） | h(E) | h(H) | h(A) | h(B) |
| --- | --- | --- | --- | --- | --- |
| 每次快环 | 2 | −7 | 0 | 2 | 3 |
| 每次慢环 | 9/4 | −15/2 | 0 | 7/4 | 9/4 |

第二行用同样四个方程求得。例如慢环的 H 方程为 $0=0-9/4+9/4$，B 方程为 $9/4=9-3\times9/4$。$h_L(E)$ 更负，是因为同样启动等待按更高的奖励率计入机会成本；单拿这个负数并不能断言慢环策略较差。每个策略自己的 gain 与 bias 必须一起读。

$$
\mathbb E_\pi\!\left[\sum_{n=0}^{N-1}R_n\right]-g_\pi\mathbb E_\pi[T_N]=h_\pi(S_0)-\mathbb E_\pi[h_\pi(S_N)],\qquad T_0=0.
$$

将 N 条方程相加，中间的价值相消。这是固定转移次数 N 的精确关系。启动后恰好回到 H 时，快环的累计奖励等于 $2T_N-7$，慢环等于 $9T_N/4-15/2$；两个式子各在自己的回站时刻成立。

这也解释了启动状态的特殊地位。精确模型能算出 E 的价值，但一条没有外部重置的生命只经过 E 一次，在线学习器没有无限多次机会把这项估计学准。回站 H 会反复访问，启动 E 不会；算法的覆盖条件必须区分二者。

<a id="average-algorithm"></a>

## 3 · 差分半梯度 Sarsa 的更新

$$
\begin{aligned}\delta_t&=R_{t+1}-\bar r_t+\hat q(S_{t+1},A_{t+1},w_t)-\hat q(S_t,A_t,w_t),\\ w_{t+1}&=w_t+\alpha\delta_t\nabla_w\hat q(S_t,A_t,w_t),\\ \bar r_{t+1}&=\bar r_t+\beta\delta_t.\end{aligned}
$$

两组更新使用同一个旧参数下的误差。先更新奖励率再重新计算价值误差，会得到不同算法。

**算法：差分半梯度 Sarsa**

1. 初始化动作价值权重与奖励率估计，按探索策略选择动作
1. 执行动作，得到下一状态与奖励；按旧策略选择下一动作
1. 保存两个旧动作价值，计算 $\delta=R-\bar r+q'-q$
1. 用这一个 $\delta$ 更新权重和奖励率；推进到已选的下一动作
1. 持续记录真实奖励，不因报告窗口结束而重置价值或奖励率

在一般动态控制过程中，策略、状态分布和奖励率估计都在变化。上式是教材中的算法构造，不能把固定策略表格评价或特定 Differential Q-learning 的定理直接作为它对任意线性或神经表示的收敛保证。

若下一动作项改为 $\max_a q(S_{t+1},a)$，便进入差分 Q-learning 的控制路线。RVI 方法则通过参考函数减去一个标量来规范相对价值。它们都处理平均奖励问题，但规范方式和收敛条件并不相同，不能混用变量更新。

<a id="rlss-average-rate-identification"></a>

## 奖励率与价值常数：同一个 TD 残差能否辨认两者

差分 TD 与中心化折扣 TD 都引入一个奖励率估计，但它们对常数平移的反应不同。这个差别决定了能否仅靠一个残差同时识别奖励率和 critic 的基线。先冻结策略，采用表格表示，并令 P 为随机转移矩阵。

$$
\delta_g(u)=r-g\mathbf1+\gamma Pu-u,\qquad \widetilde u=u+c\mathbf1,\qquad \widetilde g=g-(1-\gamma)c,\qquad \delta_{\widetilde g}(\widetilde u)=\delta_g(u).
$$

因为 P1=1，平移价值造成的 −(1−γ)c 恰好被奖励率的改动抵消。固定 γ<1 时，仅让残差为零并不能同时确定正确 g 和价值常数。

最小例子只有一个状态，自循环且每步奖励为 2。取 γ=0.9。残差 2−g−0.1u=0 同时允许 (g,u)=(2,0) 和 (1,10)。后一个 critic 的残差完全为零，奖励率却错了。若先用旧残差同步更新 g 和 u，算法会停在哪个组合，还受初始化、步长比例或其他约束影响。

$$
d_\pi^\top(r-g\mathbf1+Pu-u)=g_\pi-g.
$$

在 γ=1、固定策略且按其平稳分布取期望时，价值差的期望相消；无论固定 u 多不准确，平均残差仍能提供奖励率误差信号。此处的冻结和分布条件不能省去。

中心化折扣预测可额外固定一个规范，例如 dπ 加权价值均值为零；也可在固定行为策略下用真实奖励直接估计其平均率，再学习中心化 critic。直接奖励平均估计的是实际行为的率，不会因为 critic 中出现 max 就自动成为最优策略的率。离策略控制必须另作论证。

| 更新形式 | 必须区分的含义 | 不能互换的细节 |
| --- | --- | --- |
| 差分 Sarsa | 评价并改善实际探索行为 | 下一动作来自同一行为轨迹 |
| Differential Q-learning | 以最大后继动作价值构造控制误差 | 原算法在每次有效转移上更新奖励率；需其特定条件 |
| Greedy-gated R-learning 风格 | 只在满足贪心门控时更新奖励率 | 门控及其判断使用更新前还是更新后 Q，都是算法定义 |
| 固定策略的奖励滑动平均 | 追踪当前实际获得的每步奖励 | 不是离策略最优奖励率估计器 |

这些差异不是命名问题。代码里多一个 if，就可能改掉平均更新与适用定理。先在单状态例子检查奖励率可辨识性，再做两周期控制和多链反例，最后加入函数逼近。把固定折扣逐渐调到 1 是一种求解思路；不由上述恒等式自动获得非平稳控制保证。

<a id="lesson-example"></a>

## 4 · 手算：周期链与奖励率同步更新

A 到 B 的奖励为 0，B 到 A 的奖励为 2。两个状态确定性交替，每两个时间步共得到 2，所以平均奖励率为 1。选择 $h(A)=0$，则 A 的方程给出 $h(B)=1$；B 的方程也成立。这个周期例子有平均奖励和 Poisson 解，却不应直接假设普通中心化无穷回报逐点收敛。

$$
1+0=0+1,\qquad1+1=2+0.
$$

这是两个状态的差分 Bellman 方程。

现在令估计 $w=(0,1)$、$\bar r=0.5$，观察 A 到 B。步长 $\alpha=0.1,\beta=0.2$。误差为 $0-0.5+1-0=0.5$。因此 $w_A^+=0.05,w_B^+=1,\bar r^+=0.6$。

若错误地先令奖励率变为 0.6，再给价值计算误差，就会把误差变为 0.4，得到 $w_A^+=0.04$。这就是一项能区分实现时序的单步测试；仅看长时间曲线可能不容易发现。

<a id="average-rate-backup"></a>

### 4.1 · 在慢环上手算一次预测更新

仍固定每次选慢环。取更新前估计 $\bar g=2,\hat h(B)=2,\hat h(H)=0$。观察到 B 返回 H，奖励为 9、耗时为 3。因为每个访问状态在固定策略下只有一个选定动作，可以把对应的动作价值记为 $\hat h$。采用 inter-option Differential Q-evaluation 的已知确定性时长特例，先只计算一次残差：

$$
\begin{gathered}\delta=9-2\times3+0-2=1,\qquad\Delta=\alpha\delta/L=\frac3{10}\frac13=\frac1{10},\\\hat h^+(B)=\frac{21}{10},\qquad\bar g^+=2+\frac12\Delta=\frac{41}{20}.\end{gathered}
$$

这里 $L=3$ 秒是已知的期望时长，$\alpha=0.3$ 秒、$\eta=0.5/\text{秒}$；增量为 $\Delta=\alpha\delta/L$，率更新为 $\eta\Delta$。把一秒作为时间单位后，可以直接使用这些数值。价值与率共用旧残差，随后才处理下一条转移。

如果先将率改为 2.05，再重算残差，就得到 $9-2.05\times3-2=0.85$；价值因此变为 2.085，而非 2.1。若把全部价值加 100，正确残差仍为 1。这两个手算检查分别针对更新次序与常数规范。

同一旧参数快照；一次确定性时长归一化备份

```python
rate, h_b, h_h, duration = 2.0, 2.0, 0.0, 3.0
alpha, eta = 0.3, 0.5
delta = 9.0 - rate * duration + h_h - h_b
increment = alpha * delta / duration
h_b, rate = h_b + increment, rate + eta * increment
print(delta, h_b, rate)  # 1.0, 2.1, 2.05
```

上述更新对应 Wan、Naik 与 Sutton 的 options 论文 §3，式 (6)–(10)，这里仅演算一次。随机时长下应维护期望长度 L，不能把随机观测到的 τ 同时塞进分母并沿用原固定点结论；单步动作论文的收敛证明也不能直接覆盖任意半马尔可夫更新。

接下来允许 H 的动作变化，就要比较两条路线在同一旧策略基准下的后果。第三册的[同一服务站控制与规划](/zh/continual-rl/algorithms/average-reward/#average-rate-control)继续计算一次策略改善，并让时长模型过时，检查规划为什么会选错。

<a id="average-objective"></a>

## 5 · 折扣目标为什么可能改变动作排序

设初始决策可以进入两个不可逆分支。分支甲先支付 100，之后每步奖励为 2；分支乙无初始成本，之后每步奖励为 1。长期平均收益分别为 2 和 1，平均奖励偏好甲。有限折扣却可能偏好乙，因为初始成本不会被按时间平均稀释。

$$
\begin{aligned}v_\gamma(\text{甲})&=-100+\frac{2\gamma}{1-\gamma},\\ v_\gamma(\text{乙})&=\frac{\gamma}{1-\gamma},\\ v_\gamma(\text{甲})>v_\gamma(\text{乙})&\iff\gamma>\frac{100}{101}.\end{aligned}
$$

这是目标差异的反例，不是满足 communicating 假设的收敛示例。不可逆分支也提醒我们，最优长期率可能掩盖前期成本和安全代价。

上例从同一个决策起点比较未来。换一种比较方式会怎样：对每个策略，按它自己长期访问各状态的频率，平均这些状态的折扣价值？有限状态、固定策略、有界奖励下，令 dπ 为该策略的平稳分布，且每个策略具有单一常返类。于是长期率不依赖起点，可以写成下面的标量。这里不要求状态分布逐步收敛；周期链也可以有平稳分布。

$$
\begin{aligned}d_\pi^\top P_\pi&=d_\pi^\top,\qquad g_\pi=d_\pi^\top r_\pi,\qquad 0\le\gamma<1,\\ J_\gamma^{\rm stat}(\pi)&=d_\pi^\top v_\gamma^\pi=d_\pi^\top(r_\pi+\gamma P_\pi v_\gamma^\pi)\\ &=g_\pi+\gamma J_\gamma^{\rm stat}(\pi),\\ J_\gamma^{\rm stat}(\pi)&=\frac{g_\pi}{1-\gamma}.\end{aligned}
$$

把 Bellman 方程按平稳分布加权，后继状态仍服从同一分布，因而后继价值的平均与当前相同。固定同一个 γ 比较策略时，只多了一个公共正因子，排序恰好与平均奖励一致。这是原书 §10.4 所讨论的目标。

这与前面的起点反例并不矛盾。固定初始分布 $\mu$ 下的 $J_\mu(\pi)=\mu^\top v_\gamma^\pi$ 仍是合法目标，折扣一般会影响排序；$d_\pi$ 则随策略改变，已经是另一种评价。在优化 $J_\gamma^{\rm stat}$ 时，不能冻结 $d_\pi$ 后只提高当前常见状态的价值，就声称仍在优化这个完整目标。

因此，持续交互与函数逼近并未使所有折扣控制都失去意义。需要重新检查的是：所用状态加权是否对应我们想比较的交互过程，实际更新又是否改善那个目标。若目标已选为长期奖励率，求解方法中再使用折扣，则它承担估计或计算的作用；仅从上述恒等式，推不出某个折扣 TD 控制算法会优化奖励率。

平均奖励也有局限。有限生命期里付出巨额不可恢复成本可能不可接受，即使渐近奖励率更高。真正研究单生命期系统时，需要同时记录累计奖励、坏事件概率与恢复时间，不能只报告理论上的无限时域平均率。

如果动作或 option 的持续时间不同，应按真实时间归一化。一个持续十步的宏动作获得奖励 5，不能与持续一步获得奖励 2 按“每个决策点的奖励”直接比较。半马尔可夫残差中要扣除奖励率乘持续时间，而非只减一次奖励率。

<a id="lesson-code"></a>

## 6 · 可运行的差分更新与周期实验

差分 Sarsa 更新核与两状态固定策略评价

```python
def differential_sarsa(weights, rate, x, reward, xp, alpha, beta):
    delta = reward - rate + dot(weights, xp) - dot(weights, x)
    # One old delta drives BOTH updates. There is no terminal discount mask.
    next_weights = [w + alpha * delta * value for w, value in zip(weights, x)]
    next_rate = rate + beta * delta
    return next_weights, next_rate, delta


def average_demo():
    # A -> B gives 0; B -> A gives 2. Exact gain=1, h(B)-h(A)=1.
    weights, rate = [0.0, 0.0], 0.0
    for t in range(12000):
        s = t % 2
        x, xp = [0.0, 0.0], [0.0, 0.0]
        x[s], xp[1 - s] = 1.0, 1.0
        weights, rate, _ = differential_sarsa(weights, rate, x, 2.0 * s, xp, 0.05, 0.01)
    return {"gain": rate, "relative_values": [0.0, weights[1] - weights[0]],
            "exact_gain": 1.0, "exact_relative_values": [0.0, 1.0]}
```

示例每个状态只有一个动作，因此它验证的是差分评价，而不是已解决非平稳控制。运行后奖励率接近 1，相对价值接近 [0,1]。测试还检查给价值共同加常数后的残差不变，以及奖励率与价值是否确实共用旧误差。

运行差分评价及边界测试

```sh
python3 approximation_textbook_lab.py average-control
python3 approximation_textbook_lab.py test
```

服务站算例的[完整标准库教程](/crl-code/tutorials/average_rate_walkthrough.py)用分数高斯消元独立解五个方程，并按一秒一格展开真实奖励。将这个单文件保存为 average_rate_walkthrough.py，在文件所在目录运行：

得到 gain/bias、一次备份、有限时间收益与模型改变后的动作比较

```bash
python3 average_rate_walkthrough.py
```

<a id="experiment-differential_dyna"></a>

### 实验：实验 · 学得模型的五次规划值得多少真实经验

相同真实交互下，经验模型备份能否更早形成好的策略？

**环境与可用信息。** 三状态、双动作、完全可观测的持续 MDP。状态 0/1/2 的两动作奖励依次为 (0.1,−0.05)、(0.5,0.05)、(0.2,1.2)。优先后继依次为 (1,2)、(0,2)、(0,1)：转移以 0.8 概率取该后继，以 0.2 概率在三状态中均匀选择。无终止、无外部重置。控制器只能看到实际转移；模型由各状态动作的奖励均值和后继频数学习，不能访问真实转移矩阵。

**设置。** 1200 个真实步；状态 0、Q=0、率=0 起步。ε=0.25，Q 步长 0.08，率步长 0.008。每个真实步后做 5 次已访问状态动作的模型期望备份；模型自由对照没有这些备份。规划 RNG 与环境 RNG 分开。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** 先做真实差分 Q 更新，再写入经验模型，再用 planning_update 更新 Q 和率。模型不能从未访问状态动作生成知识；模拟奖励不计入真实生命期奖励。

**测量。** 主图是冻结当前贪心策略后，用真实模型精确解得的 gain。另看真实交互奖励率和总备份数：最终策略能力、学习过程所得收益和计算开销是三个量。

```bash
python3 implementations/average_systems/differential_dyna.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-figures/result-differential_dyna.svg)

横轴：environment_steps。纵轴：冻结贪心策略的精确平均奖励。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 固定当前 Q 所选的贪心策略，在真实三状态模型上求其长期平均奖励。真实模型只用于评价。value 不是算法内部的 gain_estimate，也不是带探索的行为所收到的平均奖励。

**step：怎样计时。** step 是真实交互数。两方法每步都直接更新 Q 和率；Differential Dyna 另做5次经验模型期望备份。environment_updates、model_backups、total_backups 分别记录这两类更新及其和，尚未计模型维护与求和的全部运行时间。

**怎样汇总。** 取该记录时刻的值；不先对曲线上的时间点求平均。 全程行为奖励率另看 experienced_reward_rate：它已把从第1步开始的探索与学习成本纳入平均，不应再次平均各稀疏检查点。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** CSV 可重算奖励率差 optimal_gain − value 和各更新计数。experienced_reward_rate × step 给出累计真实奖励；模型奖励不加入该总量。日志未保存 Q 表与每步奖励，不能仅凭冻结策略的 value 恢复策略或逐步行为轨迹。

计算位置：[average_systems/differential_dyna.py](https://yingwen.io/crl-code/implementations/average_systems/differential_dyna.py) · [average_systems/differential_q_multistate.py](https://yingwen.io/crl-code/implementations/average_systems/differential_q_multistate.py) · [average_systems/_common.py](https://yingwen.io/crl-code/implementations/average_systems/_common.py)

</details>

**结果分析。** 第 300 步，Dyna 的冻结策略 gain 均值约 0.58763，对照约 0.42844；第 1200 步两者均为最优 0.59704。真实全程奖励率分别约 0.50207、0.45040，但总备份数分别为 7200 和 1200。相同交互不等于相同计算。

**结论边界。** 这是有限经验模型的组件实验，不是非平稳或神经 Dyna 的收敛证明。两者最后达到同一策略，不能仅凭终点图说明规划无用；也不能忽略六倍备份成本。

**继续实验。** 比较真实步预算相同和总备份预算相同两种口径。另冻结模型学习但保留规划，说明旧模型何时无法提供新世界的证据。

[源码](https://yingwen.io/crl-code/implementations/average_systems/differential_dyna.py) · [逐种子记录](https://yingwen.io/crl-code/results/differential_dyna/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/differential_dyna/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/differential_dyna/curves.json)

<a id="lesson-branches"></a>

## 7 · 非平稳性、时间尺度与实验指标

固定奖励率是平稳问题中的对象。环境发生变化时，常数步长奖励率估计可以追踪局部变化，但它不再估计同一个全局常数。奖励率跟踪过慢会把均值变化误认为状态优势；过快则可能吸收本应由价值解释的差异。两种步长需要共同研究。

控制实验应保留完整交互轨迹，报告总体每步奖励、变化前后窗口收益、恢复时间和失败次数。把轨迹人为切成回合再给每回合相同权重，可能改变时间加权；删除崩溃段则会夸大奖励率。

与 CRL 主线的连接有两层：平均奖励规定长期控制目标，流式学习规定经验与计算的使用权限。使用平均奖励并不自动满足有限内存；严格流式也不要求采用平均奖励目标。研究时应把这两个维度分开。

<a id="lesson-check"></a>

## 8 · 练习与答案

- 只把折扣 TD 的折扣设为 1，会得到差分 TD 吗？答案：不会；还需要奖励率估计来消除线性增长趋势。
- 周期链能否具有平均奖励？答案：可以。时间平均与状态分布逐点收敛是不同性质，本章交替链就是例子。
- 奖励率已经正确，差分价值是否必然正确？答案：不必然。奖励率只有一个标量，无法描述不同状态的暂态优势。
- 持续 4 步、总奖励 7 的 option，在奖励率估计为 1.5 时，其未加终点价值的差分目标是多少？答案：$7-4\times1.5=1$。



<a id="chapter-code"></a>

## 下载与运行

标准库教学实现。命令运行本章的可解析小问题和全文件的公式测试；不依赖深度学习库，不等同于论文中的大规模实验。

[下载 approximation_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/approximation_textbook_lab.py)

```sh
python3 approximation_textbook_lab.py average-control
python3 approximation_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：Part II 第 9–13 章。原书建立函数逼近预测、控制、离策略稳定性、资格迹与策略梯度的共同框架。

- [Wan, Naik & Sutton · Learning and Planning in Average-Reward MDPs](https://proceedings.mlr.press/v139/wan21a.html)：Differential Q-learning 等算法的原始论文。其表格控制理论不应直接替代任意函数逼近的保证。

- [Wan, Naik & Sutton · Average-Reward Learning and Planning with Options](https://arxiv.org/abs/2110.13855)：§2 的真实时间奖励率与半马尔可夫 Bellman 方程；§3 式 (6)–(10) 的期望时长归一化更新。本文服务站为原创确定性算例。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-approximation-average-control#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-approximation-average-control#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-approximation-average-control)
<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 优化一段折扣回报，还是长期单位时间收益？

回报定义决定策略排序。折扣奖励、有限时域和平均奖励是不同目标；把折扣取大，只是在某些条件下接近相应极限。

函数逼近与深度方法：持续任务的相对价值需要奖励率和定标条件。训练中更新策略时，奖励率也在变化。动作时长不同，还要区分每次决策与单位物理时间的收益。

持续学习中的研究问题：长期收益率忽略有限的启动损失；单生命期不能忽略。怎样同时报告生命期收益、适应成本和后期表现，并让预测、控制与模型使用一致的时间单位？

[平均奖励控制基础](https://yingwen.io/zh/continual-rl/foundations/approximation/average-control/) → [熵如何改变目标](https://yingwen.io/zh/continual-rl/foundations/deep/entropy-control/) → [平均奖励的预测、控制与规划](https://yingwen.io/zh/continual-rl/algorithms/average-reward/) → [完整学习器的评价](https://yingwen.io/zh/continual-rl/algorithms/control/)


### 可进一步检验的问题

- [01 · 什么目标能够评价一个始终在学习的智能体？](https://yingwen.io/zh/continual-rl/research/#research-lifetime-objective)：周期链的奖励率与暂态价值说明长期极限会忽略有限前缀，由此可问适应成本何时改变实际部署期的排序。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [平均奖励与差分价值](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)
- [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)
- [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

对应原始材料：第 10 章：平均奖励、折扣目标与 differential semi-gradient Sarsa。本文为原创讲解，原书、论文与上游代码保留各自许可。
