# 持续探索：新奇、不确定性、学习进展与恢复

外部奖励稀疏、世界持续改变时，怎样获得有用的新经验，而不是追逐永远无法学会的噪声？

## 本章内容

- 在同一走廊中计算预测误差、后验更新与学习进展，辨认哪些不确定性可以被经验减少。
- 跟踪信息怎样改变后续动作、真实经验和有限时间收益，并把恢复的原始步数计入比较。
- 从计数与固定随机目标理解 count bonus 和 RND，明确奖励时序以及它们与信息价值的关系。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [持续控制：比较策略与学习智能体](https://yingwen.io/zh/continual-rl/algorithms/control/)：将试探的收益和成本计入真实交互。


### 外部奖励与内部奖励

外部奖励是任务学习器接收的评价信号；内部奖励由系统构造，用于引导数据获取。外部奖励是否表达设计者意图，仍是奖励设计问题。提高内部奖励不自动意味着外部任务表现更好。

### 认识不确定性与随机性

认识不确定性可通过更多数据减少；环境固有随机性可能不能减少。单次预测误差往往混合两者。

### 目标条件策略

$\pi(a\mid s,g)$ 把目标 $g$ 当作输入；课程决定练习的目标，控制器决定到达方法。课程调度与底层策略学习处于不同层次。

<a id="problem-definition"></a>

## 本章的问题定义

行为既获得外部收益也决定未来能学到什么；在稀疏反馈、漂移或无免费重置条件下选择有用且可恢复的经验。

### 给定条件与符号

- 外部任务、可达动作、反馈及恢复/重置权限。
- 探索先验或内部信号类、可保存统计、训练与恢复预算。

### 需要求解的对象

服务声明外部目标的数据获取策略；新奇、不确定性、进展与技能多样性是不同候选代理。

### 信息与数据权限

$r_t^{\rm ext}$ 是外部反馈，$r_t^{\rm int}$ 由已到达经验和当前统计生成；训练信号在更新前或后计算的约定必须明确。

$$
J^{\rm ext}_T(L)=\mathbb E_L\!\left[\sum_{t=0}^{T-1}r_t^{\rm ext}\right],\qquad r_t^{\rm train}=r_t^{\rm ext}+\beta_t r_t^{\rm int}
$$

$L$ 是完整探索与学习过程，$T$ 为寿命，$\beta_t$ 为内部信号权重。第一式定义此处有限寿命外部评价，第二式只是常见训练代理；内部回报增加不能代替外部收益、可达性或信息增益。

### 成立条件与解的含义

- 认识不确定性与环境随机性分别讨论；预测误差包含模型限制和遗忘。
- 无重置问题明确先前知识、恢复权限和不可逆失败；可重置基准不能免费证明single-life效果。

判断准则：匹配预算记录真实覆盖、原始外部收益、恢复步数和不可逆失败；在随机噪声区/可学习区对照代理，并用固定probe区分新奇与预测器遗忘。

### 适用边界

- 随机参数或动作噪声不自动构成校准后验。
- 高RND误差不等于高信息增益；恢复数据也不能假装免费重置。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：探索改变后续世界和训练分布，完整控制需保留探索损失与长期收益。

- 组合不同学习问题 · [目标条件化与子任务构造](https://yingwen.io/zh/continual-rl/construction/goals/)：课程选择目标改变真实数据获取，底层目标条件策略另学如何达成。

- 组合不同学习问题 · [元学习与学习规则的适应](https://yingwen.io/zh/continual-rl/algorithms/meta/)：探索信号或选择规则可由外部寿命评价学习，但需计入元训练预算。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

外部奖励无法指出应访问哪里；高预测误差可能来自不能学会的噪声或内部遗忘。

### 本章的核心思路

为所需数据价值选择对应代理，再用覆盖、学习与恢复的独立量检查其用途。

1. [估计访问不足或熟悉程度](#lesson-rnd)：因为尚未覆盖的行为可能有价值，计数与RND分别用访问统计和固定目标拟合构造代理，计时先定义。

2. [区分误差与可学习进展](#lesson-progress)：因为永远噪声可保持高误差，用匹配probe的误差/成功率变化选择练习目标。

3. [将恢复与不可逆代价纳入](#lesson-recovery)：因为无法免费回到起点，探索还需检查返回安全集的能力，原始收益计入真实恢复过程。

结论与条件：计数递推/RND梯度可核验，但bonus与误差只是代理；后验或探索保证需额外模型与采样条件，single-life结果限于声明权限。

### 相关方法改变了什么

- 计数/RND：分别代理覆盖与随机函数熟悉程度，可能受表示漂移或遗忘影响。

- 后验/不确定性探索：面向相容价值假设，需区分认识与回报随机性。

- 进展课程/恢复策略：前者分配可学习目标，后者处理真实可达性和失败成本。


<a id="lesson-setting"></a>

## 1 · 探索的数据获取目标

在长走廊尽头才有奖励，ε-greedy 的零散随机动作可能始终无法形成连续前进。若环境还会改变，以前足够的经验也会过时。但用新奇奖励强迫 agent 去任何陌生地方，同样可能进入死胡同，或者一辈子盯着随机闪烁的屏幕。真正的问题是：哪些行动会带来未来仍有用、可学习、可恢复的经验？

$$
r_t^{\rm train}=r_t^{\rm ext}+\beta_t r_t^{\rm int}
$$

这是常见实现接口，而非所有探索算法的定义。若 β 随时间变，训练目标也在变；报告原始外部收益时不能把内部分数混进去。

| 要估计的对象 | 典型信号 | 容易混淆的失败 |
| --- | --- | --- |
| 访问不足 | 计数或密度模型更新幅度 | 表示每次变化产生假新奇 |
| 尚未学会 | 固定随机目标的预测误差 | 预测器遗忘使旧状态再次显得陌生 |
| 正在学会 | 固定 probe 上误差/成功率改进 | 噪声波动被当成进展 |
| 不同技能 | 状态对 latent skill 的可识别性 | 多样但不一定对任务有用 |
| 可恢复性 | 返回安全集的概率/成本 | 估计乐观却进入不可逆失败 |

计数估计覆盖程度，RND 估计随机函数的熟悉程度，课程选择练习目标，恢复机制限制可接受风险。探索的最终价值来自它怎样改变以后可用的知识和行为。用这些代理信号取代无法直接计算的信息价值，是算法设计选择；代理变大本身不是收益改善的证明。组合时需分别定义它们的目标、更新数据和资源预算。

<a id="lesson-information"></a>

### 1.1 · 同样难预测，哪条经验会留下知识？

先把走廊缩短为一个分岔 $J$。左、右两侧恰有一侧通向补给区 $G$，另一侧通向需要恢复的区域 $D$。正确侧记为 $\Theta\in\{L,R\}$；在这一段交互中它固定不变，智能体最初认为两种世界各有 $1/2$ 的可能。智能体知道下面的任务规则、观测规律和先验，唯一未知的是当前世界的 $\Theta$。它没有可以查询真实侧别的接口。

分岔旁有一块路标和一块屏幕。读路标花一个真实步，仍留在 J，外部奖励为 0；路标显示 −1 表示左侧正确，+1 表示右侧正确，读过之后内容保持不变。看屏幕也花一个步、奖励为 0，但每次显示的 −1 或 +1 都独立重抽，各占一半，而且与正确侧无关。这一已知的独立性很重要：若屏幕的分布或它与道路的关系也未知，看屏幕就可能帮助学习另一项规律。

选左或选右同样花一个步。选对时，当步得到 1 并到达 G；之后留在 G 的每一步仍可领取 1，G 没有终止这一段交互。选错时，当步得到 0 并到达 D；这个后果已经揭示了正确侧，但还要执行 k 个零奖励恢复步才能回到 J。返回后再次选路仍需一个步。先取 k=4，只比较从起点开始的 H=6 个原始步，截取之后的奖励不计入本次评价；全过程没有重置操作。

先暂停在行动前：如果两个显示目标都取 ±1，最好的初始预测都是 0。只看这时的平方预测误差，能决定该读路标还是看屏幕吗？

![智能体在分岔 J，左右道路被遮住；固定路标与随机屏幕均需一个观察步，正确路通向持续供给区 G，错误路需四个恢复步返回。](https://yingwen.io/crl-figures/exploration-information-walkthrough-task.svg)

上方保持智能体尚不知道的左右对应关系；下方展开已知任务规则。每个恢复圆点代表一个真实步。H=6 是评价窗口，G 仍可继续获得奖励。图片与以下数值来自同一个原创有限任务。

把已有的实际经验记为 $\mathcal H_t$，维护 $q_t=P(\Theta=R\mid\mathcal H_t)$。路标的预测均值是 $m_t^{\rm sign}=2q_t-1$；屏幕的下一帧均值始终是 $m_t^{\rm screen}=0$。平方损失下，条件均值给出最小的期望误差。初始 $q_0=1/2$，所以两个目标的最小期望平方误差都是 1。

$$
\begin{aligned}
 E_t^{\rm sign}&=(1-q_t)(-1-m_t^{\rm sign})^2+q_t(1-m_t^{\rm sign})^2=4q_t(1-q_t),\\
 E_t^{\rm screen}&=\tfrac12(-1-0)^2+\tfrac12(1-0)^2=1.
 \end{aligned}
$$

E 是给定当前经验、预测下一次同类输出的期望平方误差；路标会重复同一固定符号，屏幕会产生一枚新的独立符号。这里没有从训练曲线猜测这些值，而是按给定模型计算。

现在实际观察到 $+1$。若来自路标，在左侧正确的世界中这个结果不可能发生，因此 Bayes 更新得到 $q_{t+1}=1$，预测均值变为 1，下次再读同一路标的误差为 0。若来自屏幕，两种世界都以 $1/2$ 的概率产生这个结果，似然相同，后验仍为 $1/2$，下一帧误差仍为 1。看到 $-1$ 时，对称地得到路标后验 0、屏幕后验不变。

$$
q_{t+1}=\frac{q_tP(Z_{t+1}=z\mid\Theta=R,a_t)}{q_tP(z\mid\Theta=R,a_t)+(1-q_t)P(z\mid\Theta=L,a_t)}.
$$

$a_t$ 指本次选择读标还是看屏。控制器更新时只读取所选行动产生的 $z$；未读路标的内容不会成为训练标签。道路的成功或失败也通过各自的观测似然更新同一个 $q$。

这把相同的初始误差分成了两种来源。路标的内容在真实世界里是确定的，误差来自智能体尚不知道哪一种世界正在发生；屏幕的规律已经知道，误差来自下一枚符号本身的随机性。换一种预测器，不能从这些历史中推断一枚独立公平符号的取值。若只把误差大的地方设为行动目标，屏幕会一直有分数，即使关于道路的知识再也没有增加。

可以直接量化关于 $\Theta$ 的不确定性。令二元熵 $H_b(q)=-q\log_2q-(1-q)\log_2(1-q)$，并约定 $0\log_2 0=0$。一次观察的预期信息增益为 $I(\Theta;Z\mid\mathcal H_t,a)=H_b(q_t)-\mathbb E_Z[H_b(q_{t+1})]$。从均匀先验出发，读标把熵从 1 bit 降到 0，信息增益为 1 bit；屏幕的增益为 0。这里的信息对象是固定侧别，而非已经显示出来的一枚屏幕符号。VIME 使用后验与先验之间的 KL 变化构造探索奖励，并以变分模型近似难以精确维护的后验；这个两世界例子让相应更新可以逐项算出。

![观察到同一个 +1 后，路标使左右侧别的后验从各半变为右侧概率 1，而随机屏幕后验保持各半；未来预测误差分别从 1 降到 0、从 1 保持 1。](https://yingwen.io/crl-figures/exploration-information-walkthrough-posterior.svg)

概率条高表示关于正确侧的信念；下方条长表示下一次同类目标的期望平方误差。两种通道使用相同输出尺度。读标带来 1 bit 信息和 1 单位预测进展，屏幕两者均为 0。

误差下降还必须说明在哪些目标上测量。对屏幕采用一个故意简单的更新 $m_+=Z_t$，相当于半平方损失的一次步长 1 更新。若刚见到 $Z_t=+1$，该样本上的误差从 1 降到 0；然而新的独立帧 $Z'$ 仍各半取 ±1，$\mathbb E[(Z'-m_+)^2]=\tfrac12(1-1)^2+\tfrac12(-1-1)^2=2$。同样本的“进展”为 1，未来预测却从误差 1 退到了 2。保持已知最优均值 0 的预测器，其误差才一直是 1。这两个预测器不能混成同一条曲线。

本例先在两种可能世界及屏幕输出上枚举，得到供读者核算的风险；真实学习器并不会收到未选择动作的观测。实际估计学习进展时，也应让更新前后的预测器面对可比较的评价目标，避免把记住一次噪声当成改善。RND 则另外规定了预测目标：同一输入对应同一个冻结随机网络输出。它并没有要求预测下一枚随机屏幕符号，后文会回到这个差别。

<a id="lesson-information-value"></a>

### 1.2 · 知道以后怎样行动，决定信息值多少

有了可减少的不确定性，还要回答是否值得现在花一步去消除它。先只看后续行动：读标得到 +1，q 变为 1，智能体下一步选择右侧，获得 1 并到达 G；接着四步继续领取，总收益为 5。读到 −1 则选左，收益同样为 5。这里是更新后的 q 改变了实际动作，再由真实道路产生下一条奖励。

直接选路也能学习，不能把它的后续行为固定在最初猜测上。先验并列时约定选左：若左侧正确，六步都能得到 1，共得 6；若右侧正确，第一个零奖励立即把 q 更新为 1，但第 2 至第 5 步仍要恢复，第 6 步才重新从 J 选右并获得 1。这两种世界各占一半，直接选左的期望总收益是 (6+1)/2=3.5。知道了答案，没有让身体立即回到分岔。

看屏幕后 q 保持不变，却只剩五个步。此时再读路标、选正确侧并持续领取，能得到 4；若直接选路，期望只有 2.5。因此“先看屏幕”的最优后续行为仍会转去读路标。它少得到的 1 来自已经消耗的一步，而非某种人为给噪声设置的负奖励。先预测下面那条走错的轨迹：知道右侧正确之后，还会经过几个零奖励步？

![六个原始步的完整行动与奖励：读标后选正确侧得 5；直接选左在两个世界分别得 6 与 1，走错后四步恢复；看屏后再读标得 4。](https://yingwen.io/crl-figures/exploration-information-walkthrough-control.svg)

每个圆是一原始步，圆下是该步外部奖励。直接选左的两行以先验各半求期望；观察策略在另一世界只交换左右，收益不变。G 的后续领取和 D 的恢复均实际计时，没有只统计成功片段。

这个任务足够小，可以写出动作消费者的全部计算。令 $V(h,q;k)$ 表示处在 J、还比较 $h$ 个步、当前后验为 q 时的最优期望外部收益，$V(0,q;k)=0$。记 $[x]_+=\max(0,x)$。选择正确侧后，这 h 个步都能取得奖励；选错会先损失一个步，再花 k 个恢复步，最多还剩 $[h-1-k]_+$ 个领取步。于是：

$$
\begin{aligned}
 Q_L(h,q;k)&=(1-q)h+q[h-1-k]_+,\\
 Q_R(h,q;k)&=qh+(1-q)[h-1-k]_+,\\
 Q_{\rm sign}(h,q;k)&=h-1,\\
 Q_{\rm screen}(h,q;k)&=V(h-1,q;k),\\
 V(h,q;k)&=\max\{Q_L,Q_R,Q_{\rm sign},Q_{\rm screen}\},\qquad h\geq1.
 \end{aligned}
$$

这是在正确已知的两世界模型内进行的有限时间规划。q 来自观测，h 由实际消耗的步数递减，k 是本例给定的恢复时长；策略不读取真实 Θ。路标精确揭示侧别，故读标后 h−1 个步都可获 1。屏幕既无任务奖励也不改 q，所以只留下更短的同一问题。

在原条件下，四个首行动值依次为 3.5、3.5、5、4，控制器选择读标。现在做一个接口对照：花一步读标并正确更新 q，却让控制器保留读标前的各半信念，随后直接选左；道路反馈仍正常更新它的信念。这样两世界分别只得 5 与 0，期望 2.5。把知识写进一个变量与让行动读取这个变量，是两个需要分别核对的环节。

| 只改变的条件 | 直接选路的期望收益 | 先读路标的收益 | 较优首行动 |
| --- | --- | --- | --- |
| H=6，k=4 | 3.5 | 5 | 读标 |
| H=6，k=0 | 5.5 | 5 | 直接选路 |
| H=1，k=4 | 0.5 | 0 | 直接选路 |

k=0 指另一种物理条件：选错的那个动作在得到 0 后就结束在 J，不再需要额外恢复步。它仍消耗一个真实步，并非调用免费的 reset。此时直接选路的两种收益是 6 和 5，平均 5.5，已经超过先读标的 5。只剩一个步时，读标仍带来完整的 1 bit 信息，但本次评价里已没有行动可使用它。

因此同一条路标、同一个先验可以有相同的信息增益和预测进展，却有不同的外部收益价值。以上比较已经计入获取信息的时间，也允许直接试路后学习和恢复；不能把一比特直接换算成一个固定奖励。持续交互还会改变知识的有效期：若侧别随后可能变化，旧后验要怎样更新、何时值得重新检查，便成为新的问题。当前算例的 Θ 固定，只隔离信息获得、实际消费者与恢复时间这三个环节。

<a id="lesson-derive"></a>

## 2 · Count bonus：少见为什么值得再看？

在独立同分布、有限方差的样本均值估计中，标准误差按 $1/\sqrt N$ 缩小。表格探索由此把状态动作访问次数作为不确定性的粗代理，对少见选择增加奖励。完整的探索保证还需要奖励界、转移模型、置信水平与规划条件。

$$
b_t(s,a)=\frac{\beta}{\sqrt{N_t(s,a)+1}},\qquad N_{t+1}(s,a)=N_t(s,a)+\mathbf1[(S_t,A_t)=(s,a)]
$$

此约定先用访问前计数 Nt 计算本次奖励，再计数加一；+1 定义首次访问。若先加一，就应相应改写公式，避免 off-by-one。

像素状态几乎不重复，直接计数会把每帧当作首次访问。可在固定表示上哈希计数，或者用密度模型提供会泛化的伪计数。令 $p(x)$ 是学习当前 $x$ 之前的概率，$p'(x)$ 是更新一次之后的概率；假想二者分别等于 $\widehat N/\widehat n$ 与 $(\widehat N+1)/(\widehat n+1)$，解这两个方程得到：

$$
\widehat N(x)=\frac{p(x)(1-p'(x))}{p'(x)-p(x)}
$$

需要 p′≥p 的 learning-positive 条件使计数非负；p′=p 的极限对应无学习增量，不能直接除零。深度密度优化不总满足这个条件，实际实现必须处理。

如果表示也不停学习，同一个物理状态可能被重新映射到未访问位置。此时 bonus 反映的是编码漂移，不一定是真实知识增加；固定随机表示、冻结评测编码和原始状态覆盖可帮助分离这两者。

<a id="experiment-count_bonus"></a>

### 实验：实验 · 新奇奖励可以暂时偏离原任务

简单链已能用 ε-greedy 探索时，额外访问奖励是否仍会更快找到原任务的最优行为？

**环境与可用信息。** 六格链从状态 0 出发，左右移动，到第 5 格终止并奖励 1，其余转移奖励 −0.01。环境平稳，折扣 0.95。五个非终止状态各有两个 Q 值，初值全零。

**设置。** 五种子各 1200 个真实转移。Q-learning 步长 0.2，ε=0.1。每次访问先将状态动作计数加 1，再给训练目标加入 0.2/sqrt(N)；无 bonus 对照其余规则相同。终点之后重置到 0。

**检验的机制。** 少访问的状态动作会得到较大正奖励，策略可能因此停留或绕行。随着计数增加，人工奖励衰减。它改变训练目标，但评价始终使用原环境奖励。

**测量。** 纵轴是冻结贪心策略从 0 出发的原奖励折扣回报。代码对确定性策略中的循环求精确无限和；没有把卡住的轨迹按某个有利截止删掉。最短路径的值约为 0.7774。

```bash
python3 implementations/continual/count_bonus.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/count_bonus/curves.svg)

横轴：environment_steps。纵轴：原始奖励贪心策略折扣回报。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 第 615 步，有 bonus 的平均原任务回报为 0.1910，无 bonus 对照五条都已达到 0.7774。第 1200 步两者都达到 0.7774。此处额外新奇奖励延迟了原任务行为形成，末尾没有差异。

**结论边界。** 短小平稳链没有困难的稀疏探索瓶颈。这不是密度模型 pseudo-count 的验证，也不是对所有探索奖励的否定。计数新奇不等于信息增益或外部效用。

**继续实验。** 先分别记录访问覆盖与原任务回报，观察二者能否反向变化。再增加链长度、改变 bonus 系数，明确在哪个任务难度区间 bonus 的探索收益超过目标偏移的代价。

[源码](https://yingwen.io/crl-code/implementations/continual/count_bonus.py) · [逐种子记录](https://yingwen.io/crl-code/results/count_bonus/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/count_bonus/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/count_bonus/curves.json)

<a id="lesson-rnd"></a>

## 3 · RND：把熟悉程度变成一个可训练的预测任务

Random Network Distillation 初始化两个网络：目标网络 $f_\xi$ 随机生成后永久冻结，预测网络 $f_\theta$ 学习拟合它。对刚看到的观测，两个输出的差作为新奇信号；重复观察后预测误差逐渐下降。目标固定，是它与预测随机环境下一帧的关键区别。

$$
\begin{aligned}r_t^{\rm int}&=\|f_\theta(o_{t+1})-f_\xi(o_{t+1})\|_2^2,\qquad \xi\ \text{冻结},\\L_{\rm pred}&=\tfrac12\|f_\theta(o_{t+1})-f_\xi(o_{t+1})\|_2^2,\\\theta_+&=\theta-\alpha J_\theta(o_{t+1})^\top[f_\theta(o_{t+1})-f_\xi(o_{t+1})].\end{aligned}
$$

Jθ 为预测网络输出的 Jacobian。intrinsic reward 先由更新前 θ 计算并保存，再训练预测器；不能在反向传播后重算奖励却不说明目标时序。

**算法：算法伪代码**

1. 初始化冻结的随机 target ξ、可训练 predictor θ 和行为学习器
1. 循环收集经验：
  1. 行为策略给动作，环境返回 o′ 与外部奖励
  1. 按规定在线统计归一化 o′；target 始终 stop-gradient
  1. 用更新前 predictor 计算并保存 $r_{int}=\|\mathrm{predictor}(o^{\prime})-\mathrm{target}(o^{\prime})\|^2$
  1. 以 $r_{ext},r_{int}$ 及各自 discount 更新价值/策略目标
  1. 用指定数量样本和更新次数训练 predictor
  1. target 保持冻结；跨回合保留新奇统计

原始 RND 使用 actor–critic/PPO 系统，并区分外部与内部奖励的价值估计。完整实现还包含观测归一化、内部奖励缩放、预测器更新比例、两个折扣因子、终止语义与优势混合。本页表格示例只保留预测误差与更新次序，完整神经网络实验使用文末作者代码。

固定 target 降低了一类“预测环境随机下一帧”问题，却不消除全部噪声陷阱：随机像素可以产生不断变化的输入，预测器容量和覆盖仍有限。更重要的是，持续学习中 predictor 也会遗忘；旧状态预测误差重新升高可能只是内部退化。评价应同时检查覆盖和预测器保留，而不是把每一次高误差都叫信息增益。

<a id="experiment-rnd_exploration"></a>

### 实验：实验 · 预测误差作为内奖，未必优于无偏随机游走

RND 的误差确实参与了 Q 更新，是否意味着访问覆盖一定扩大？

**环境与可用信息。** 12 格反射边界链，从 0 开始，无终止也无外部奖励。观测为 one-hot。固定随机目标网络与独立预测网络均为 12–12–3 tanh MLP；权重取 U(−0.6,0.6)，偏置为零。

**设置。** 五种子各 1200 个转移，Q 初值为零，Q 步长 0.2、γ=0.99、ε=0.15。预测器每步以均方误差 SGD 更新，步长 0.03。内奖使用本次预测器更新前的误差。对照仍训练相同预测器，但不将误差交给 Q。

**检验的机制。** 新奇误差随训练变化，Q 又需要追踪这些变化。无内奖时全零 Q 的平局随机打破，使行为接近随机游走；这在小反射链中本身就是强覆盖基线。

**测量。** 纵轴为截至当前真正到达过的状态数除以 12，范围为 0 到 1。它不是 return，也不是模型不确定性的校准误差。原始日志另有 novelty_mse 和 predictor_updates。

```bash
python3 implementations/extended_knowledge/rnd_exploration.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/rnd_exploration/curves.svg)

横轴：environment_steps。纵轴：在线已访问状态比例。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 第 600 步，RND 平均覆盖率为 0.85，对照为 1.00；末尾为 0.90 对 1.00。因此这个小任务没有显示内奖收益。预测误差作为奖励的机制可以正确运行，同时控制效果仍不理想。

**结论边界。** 本组件没有 PPO、双 critic、观测归一化和内奖归一化，不是 RND Atari 系统。访问覆盖也不足以证明探索了任务相关信息。

**继续实验。** 先比较 bonus 的衰减速度与 Q 的追踪速度，排查旧内奖残留。随后加入明确的稀有远端事件，仍用原事件发现率评价，不通过提高内奖累计值来定义成功。

[源码](https://yingwen.io/crl-code/implementations/extended_knowledge/rnd_exploration.py) · [逐种子记录](https://yingwen.io/crl-code/results/rnd_exploration/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/rnd_exploration/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/rnd_exploration/curves.json)

<a id="lesson-uncertainty"></a>

## 4 · LSAC：通过价值函数的不确定性探索

新奇程度与决策不确定性不同。一个从未见过但显然不可达的画面可能有很高 RND 误差；两个动作的长期收益难以区分，即使画面熟悉，也可能值得进一步尝试。后验采样用一组与数据相容的价值函数表示这种认识不确定性，采样其中一个，再让策略根据它选择行动。

Langevin Monte Carlo 为这种思路提供一种参数空间采样方法。设固定数据上的能量函数为 $L(w)$，逆温度为 $\beta>0$。沿损失梯度下降并加入高斯噪声：

$$
w_{k+1}=w_k-\eta\nabla L(w_k)+\sqrt{2\eta/\beta}\,\epsilon_k,\qquad\epsilon_k\sim\mathcal N(0,I).
$$

连续时间扩散在适当正则条件下以 $p(w)\propto\exp[-\beta L(w)]$ 为不变分布。有限步长离散化存在偏差；只有能量确实对应负对数后验、且采样充分时，才具有相应的贝叶斯后验解释。RL 的目标还会改变，因此实际使用是近似。

以一维能量为例：$L(w)=(w-2)^2/2$，取 $w=0$、$\eta=0.1$、$\beta=10$。梯度项把参数移到 0.2，噪声标准差为 $\sqrt{0.02}$。噪声使多次独立运行保留不同假设；这与把同一个高斯噪声直接加到环境动作上不同。

ICLR 2025 的 Langevin Soft Actor-Critic（LSAC）把这一思路用于分布式回报 critic，再让 SAC actor 面对采样出的 critic 更新。完整方法还使用自适应 Langevin 漂移、多个参数链，以及扩散模型生成的样本与动作梯度修正。论文称其多链方案为简化 parallel tempering，但实际采用同温、不同初始化且不交换副本；它与经典不同温度的交换算法有所区别。

**算法：算法伪代码**

1. 维护真实经验库、生成样本库、多个 critic 及其目标网络
1. 从真实与生成经验中构造训练 batch
1. 对每个 critic 计算分布式 Bellman 损失，执行自适应 Langevin 参数更新
1. 从多个 critic 中采样一个，更新最大熵 actor 和温度参数
1. 按配置更新生成器、目标网络和样本库
1. 分别记录外部回报、critic 差异、计算量与实际访问覆盖

需要区分两类方差：一个状态下随机回报的方差可以是不可减少的环境随机性；不同可信参数模型之间的差异才可能反映认识不确定性。分布式 critic 不会自动完成这种分解。LSAC 的实验证据来自可重置的连续控制任务；经验库、生成器和多 critic 的资源开销，以及这些估计在长期漂移中的有效性，仍需在 CRL 条件下单独研究。

<a id="lesson-progress"></a>

## 5 · 学习进展与自动课程

路标例子中的误差从 1 降到 0，下一次读取同一路标时仍准确；已知公平屏幕的最优预测误差则保持 1。这提供了一个调度思路：把练习投向能使未来表现改善的地方。对一个目标 k，先比较学习前后的预测误差；若表现采用成功率或回报，改善方向要相应改成上升。

$$
\operatorname{LP}_k=E_{k,\rm before}-E_{k,\rm after},\qquad \operatorname{ALP}_k=|E_{k,\rm before}-E_{k,\rm after}|
$$

E 应在匹配的任务/目标和可比较的 probe 上估计。正向 LP 强调改善；绝对进展也会把退步当作需要重新关注的信号。二者不是同一种调度规则。

读过一次路标后，再读它的进展已经为 0，调度者应寻找其他尚能改善的目标。不过，进展也会受估计噪声、优化速度和表征变化影响。前面的屏幕更新展示了最直接的误判：同样本损失下降与未来误差下降方向相反。用有限数据估计进展时，需要可比较的 probe、适当的平滑或不确定性估计；此外，目标能进步多少仍未告诉我们到达它要花多少步。

离散课程的最小实现是：维护每个目标的近期与较早成功率；按进展设置采样权重，并保留非零随机探索概率；学生完成一次练习后更新该目标统计，再选下一个。ALP-GMM 将这一思想扩展到连续环境参数空间：用附近历史参数的表现估计绝对进展，在参数—进展空间拟合混合模型，偏向有进展的区域采样，同时保留随机探索。

**算法：算法伪代码**

1. 维护目标集合或可采样的连续参数空间
1. 初始化各目标的历史表现、近期表现与非零探索权重
1. 每轮：按学习进展 + 探索机制选目标/环境参数 g
  1. 学生用相同固定预算练习 g，记录外部成功率或回报
  1. 与可比较的历史表现计算 LP / ALP
  1. 更新课程模型；下一轮再采样
1. 评测始终使用预先约定的目标分布，而不是只评教师偏爱的任务

在仿真中教师可以生成新地图并 reset 到起点；现实单生命期通常没有这种权限。将课程移植到 single-life，必须把“选目标”变成当前可达目标的选择，加入到达和恢复的真实成本。随机参数生成器提供的环境控制权是一项资源，不能隐去。

<a id="lesson-skills"></a>

## 6 · 技能发现与时间一致的探索

有时需要的不是更高的一步 bonus，而是能连续执行很久的行为。技能发现可以使不同 latent z 对应不同可辨认的访问分布。以 DIAYN 风格目标为例，固定先验 p(z)，训练判别器 qψ(z|s) 分辨当前状态来自哪个技能；策略获得 log qψ(z|s)−log p(z) 的内部奖励，并以最大熵控制鼓励技能内部的动作多样性。

$$
I(S;Z)=\mathbb E[\log p(z\mid s)-\log p(z)]\ \geq\ \mathbb E[\log q_\psi(z\mid s)-\log p(z)]
$$

下界来自条件 KL 非负。判别器通过带技能标签的状态训练，技能策略把该下界中的项作为奖励；这提供多样性目标，不保证技能对应人类语义或外部奖励。

技能本身可以提高 temporally extended exploration，但何时开始、何时终止、如何选择技能仍需控制器；固定 latent 一个回合与可学习 option termination 不是同一个机制。判断技能是否有用，应测外部任务学习加速和模型规划收益，而非只展示不同轨迹颜色。

<a id="lesson-recovery"></a>

## 7 · 单生命期探索与恢复约束

走廊里选错之后，智能体立即学会了正确方向，损失却继续持续四个恢复步。这说明信息获取与恢复要在同一时间轴上评价。本例的恢复时长完全已知且总能成功；现实中恰恰还要学习能否返回、需要多久，以及这个估计在新区域是否可靠。

在模拟器中，跌进坑之后 reset() 即可；在现实中，卡住、断电或不可逆损坏会改变此后全部数据分布。探索可以维护一个恢复策略与恢复价值，估计从候选动作后能否回到安全或可维持后续交互的状态集合。若风险过大则切换到恢复动作，或停止冒险并请求外部干预。

$$
\mathcal A_{\rm admissible}(s)=\{a:\widehat P(\text{恢复成功}\mid s,a)\geq 1-\varepsilon\}
$$

这只是一个简化的决策接口。用学习估计 P̂ 过滤并不自动给真实安全保证；需要校准、保守不确定性界、分布外检测或外部安全机制。

“Leave no Trace”同时学习前向任务与 reset 行为，展示了恢复应成为学习系统的一部分。不同任务中的失败集与允许干预必须明确。研究报告至少记录恢复成功率、恢复耗时、不可逆失败和人工干预次数；不能只保留 forward policy 的成功回合。

<a id="experiment-recovery_filter"></a>

### 实验：实验 · 前向收益高，可能只是把恢复成本藏起来了

只最大化前进行为收益时，加入从实际结果学习的恢复过滤能否减少全流程损失？

**环境与可用信息。** 三种前进行为的收益为 0、0.4、1，环境中的真实恢复概率分别为 0.98、0.6、0.05。恢复失败扣除 2。过滤器看不到真实概率，只收到实际成功或失败结果。每个 trial 允许重新开展下一次试验。

**设置。** 五种子各 1200 个恢复 trial。每个动作的恢复后验从 Beta(1,1) 开始；前 30 次轮流收集证据，危险尝试同样计入总分。随后 ε=0.2 的前向收益提议器选动作，过滤器要求后验均值至少 0.75；无合格动作时选估计最高者。

**检验的机制。** 前向 Q 只估计动作收益，过滤器另管恢复可行性。过滤使提议被替换为恢复概率较高的行为。它不提供置信下界，也不把“相对最安全”误称为“保证安全”。

**测量。** 纵轴是所有 trial 的累计净奖励均值，包含预热和失败成本。日志另记 reset_failure_rate 与 interventions。若只报告前向收益，会遗漏本实验的主要问题。

```bash
python3 implementations/extended_knowledge/recovery_filter.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/recovery_filter/curves.svg)

横轴：reset_trials。纵轴：在线净奖励均值（含恢复失败成本）。每种方法 1200 reset_trials；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 末尾五种子平均净奖励为 −0.0517，无过滤对照为 −0.7945。过滤减少了损失，但自己的均值仍然为负；该结果不能被写成零失败或安全证书。

**结论边界。** 这是三动作恢复概率组件，不包含学习一个多步恢复策略，也不是真正不可重置的单生命期。对照的提议器没有直接优化净奖励，因此差异部分来自目标分工，而非新估计器必然优越。

**继续实验。** 加入直接学习净奖励的强基线，再比较后验均值过滤与保守置信下界过滤。若要研究单生命期，应让某些失败真正结束交互，并把丢失的后续学习机会计入评价。

[源码](https://yingwen.io/crl-code/implementations/extended_knowledge/recovery_filter.py) · [逐种子记录](https://yingwen.io/crl-code/results/recovery_filter/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/recovery_filter/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/recovery_filter/curves.json)

<a id="lesson-example"></a>

## 8 · 手算：新奇奖励的衰减

把 RND 缩小为某个表格状态的一个标量输出：冻结 target=2，predictor 初值 0，半平方损失梯度步长 α=0.25。第一次看见，先得到 bonus=4，再更新预测为 0.5；第二次 $bonus=(2-0.5)^2=2.25$，更新为 0.875；之后两次 bonus 分别为 1.265625、0.711914。

若误把 predictor 每回合清零，相同旧状态又得到 4。这个高分不是环境提供了新知识，而是实现制造的遗忘。同样，count bonus 在访问前 N=0、3、8 时分别为 1、0.5、1/3；更新计数的时刻必须固定。

恢复例子：动作 0 的已知恢复概率是 0.99，动作 1 是 0.2，阈值 0.9。即便探索器提议动作 1，示例 filter 也只能返回动作 0；如果所有动作都不满足阈值，它返回 None，而不是随意宣称最不坏动作安全。这一保守接口应保留到真实系统。

<a id="lesson-code"></a>

## 9 · 奖励时序实现与探索诊断

先运行本章的[路标与屏幕独立教程](/crl-code/tutorials/exploration-information-walkthrough.py)。下载后可在任意目录执行，只需 Python 标准库。它用 Fraction 在原始步上递推，再枚举固定左右世界与所有屏幕符号历史；终端输出每个首行动的期望收益、具体六步事件、不同 H/k/先验下的决策。

默认输出 left/right=7/2、sign=5、screen=4；--test 独立检查原始步递推、整段历史枚举、恢复时钟与噪声拟合失败。

```bash
python3 exploration-information-walkthrough.py
python3 exploration-information-walkthrough.py --test
```

读输出时先看 direct_R：第一步失败已经把 q 从 1/2 改为 1，第五步才返回 J，第六步选右得到 1。再只把恢复时长改成 0，检查直接行动如何反超读标。图中的[精确数据](/crl-figures/exploration-information-walkthrough-data.json)保留全部枚举比较；这些是给定任务的解析检查，没有新神经训练或采样误差带。下方保留计数、RND 奖励时序与恢复过滤器的原实践入口。

计数、更新前 RND reward、正向进展和已知恢复概率的最小实现。

```python
def count_bonus(count, scale=1.):
    if count < 0:
        raise ValueError("count must be nonnegative")
    return scale / math.sqrt(count + 1.)


def rnd_step(prediction, fixed_target, alpha=0.25):
    """One table feature: novelty is measured BEFORE fitting this sample."""
    error = fixed_target - prediction
    return error * error, prediction + alpha * error


def progress_score(old_error, new_error, floor=0.):
    """Positive progress, not raw surprise; both errors use a matched probe."""
    return max(floor, old_error - new_error)


def recovery_filter(proposed_action, recovery_probabilities, threshold):
    """Toy known-probability filter; learned estimates give no safety guarantee."""
    if recovery_probabilities[proposed_action] >= threshold:
        return proposed_action
    feasible = [a for a, prob in enumerate(recovery_probabilities)
                if prob >= threshold]
    return max(feasible, key=lambda a: recovery_probabilities[a]) if feasible else None


def exploration_demo():
    predictor = 0.0
    novelty = []
    for _ in range(4):
        bonus, predictor = rnd_step(predictor, 2.0)
        novelty.append(round(bonus, 6))
    print("exploration", {"rnd_bonus_before_fit": novelty,
          "fake_novelty_if_predictor_reset": rnd_step(0., 2.)[0],
          "count_0_3_8": [count_bonus(n) for n in (0, 3, 8)],
          "noisy_vs_learnable_progress": [progress_score(4., 4.), progress_score(2., 1.)],
          "filtered_action": recovery_filter(1, [0.99, 0.2], 0.9)})
```

预期 RND bonus 为 4、2.25、1.265625、0.711914；代码不执行真实危险动作

```sh
python lifelong_algorithms_lab.py exploration
python lifelong_algorithms_lab.py test
```

本页示例实现奖励顺序、冻结目标、相同状态的学习进展和无可行动作的边界。完整 RND、ALP-GMM 与 LSAC 训练系统由文末作者仓库提供；这些系统的神经网络训练与性能结果不属于本页小实验的复现范围。

- 将 intrinsic reward、extrinsic reward、状态覆盖、预测误差和实际学习进展分别记录。
- 固定 predictor 训练次数和容量；否则更慢的预测器可能仅因“总是没学会”得到更高内部奖励。
- 在重复访问旧区域时评测 RND 是否发生假新奇；比较冻结表征与学习表征。
- 课程基线至少包含随机目标、均匀目标、固定难度，统一学生训练预算和最终评价目标分布。
- 单生命期报告全程数据：成功、恢复、失败和干预，不只挑可 reset 的局部片段。

<a id="lesson-branches"></a>

## 10 · 方法选择与实验条件

| 观察到的瓶颈 | 优先切入 | 关键对照 |
| --- | --- | --- |
| 同一少量状态循环 | count/RND 与覆盖度 | 内部奖励大小不等于真正覆盖 |
| 远处奖励到不了 | 时间一致的探索技能 / options | 同样原始步预算，不只比技能决策次数 |
| 困难任务总学不会 | learning-progress curriculum | 区分不可学与只是暂时困难 |
| 旧区域反复显得新奇 | 预测器保留/表征漂移 | 冻结 target 仍可能 predictor 遗忘 |
| 一次失败就无法继续 | 恢复、风险和可达目标 | 干预和 reset 费用计入整个生命期 |

<a id="research-cpsrl-resampling"></a>

## CPSRL：世界不重置，探索假设可重采样

每步换一个可信模型，行动可能相互抵消；永久坚持初始抽样，又可能长期错过新证据。CPSRL（RLC 2024）以随机时钟决定更换整条探索假设，时钟不会调用环境 reset。

$$
\Pr(L=\ell)=p(1-p)^{\ell-1},\quad\Pr(L>k)=(1-p)^k,\quad\mathbb E[\sum_{k=0}^{L-1}R_{t+k+1}]=\mathbb E[\sum_{k\ge0}(1-p)^kR_{t+k+1}]
$$

L 为至少一的几何持续时间。右侧奖励来自永久保持本次抽样策略的反事实轨迹，而非重采样之后的实际新策略；时钟独立于该轨迹。奖励有界等条件允许按存活概率交换求和。

所以 $\gamma=1-p$ 的规划对应随机长度试验的期望未折扣收益。$p=0.1$ 时平均承诺 10 步，第 5 个奖励纳入概率为 $0.9^4$。p 控制算法节奏，外部评价仍可使用整个真实流的未折扣收益。

**算法：探索时钟接口；规划目标与重采样时间需匹配**

1. 初始化先验与抽样模型
1. 按当前抽样模型规划并行动，真实后果更新 posterior
1. 独立时钟决定下一步是否重新抽样模型/策略
1. 不清零网络、经验或物理状态
1. ensemble 替代 posterior 时，另说明近似方式

$$
\gamma=1-\sqrt{SA/T},\qquad\mathbb E[\operatorname{Regret}(T)]=\widetilde O(\tau S\sqrt{AT})
$$

原文 Theorem 4.1 的已知时域选择；T≥SA 使 γ 非负。S、A 是表格规模。未知时域采用原文 Appendix B、C 的调度，不能让任意固定 p 承担这个渐近保证。

$$
\left|\mathbb E_{\pi^*_{\mathcal E}}\!\left[\sum_{t=0}^{T-1}R_{t+1}\mid\mathcal E,S_0=s\right]-Tg^*_{\mathcal E}\right|\le\tau\quad\text{for all }T,s
$$

τ 是先验支持环境中最优平均奖励策略的统一 reward-averaging bound（原文 Assumption 3.2）。它约束期望累计收益与长期线性收益之差；不是所有策略的 mixing time，也不是样本均值的置信收敛时间。

该理论依赖平稳有限、弱连通 MDP，正确后验与规划，以及上述统一有界条件。任意固定 p 的深网 ensemble 不直接继承表格界。Bayesian regret 也不等于任意漂移路径上的动态遗憾。

- 已知小 MDP 用精确后验/规划，比较逐步换、固定长度和几何长度；记录远奖励到达概率。
- 匹配模型更新次数、动作预算与规划精度，计入重规划成本。
- 漂移实验另定义 posterior 遗忘或变点机制；只改时钟不会删除过时证据。
- 未确认原作者完整仓库，此处仅提供原文及算法，不猜测代码链接。

<a id="research-morefree-data-and-goals"></a>

## MoReFree：真实探索与模型内目标分布

无 reset 世界中，最大化覆盖可能长期停留在任务无关区域。MoReFree（TMLR 2025）同时改变真实目标调度与 imagination training：任务目标、返回初始区域和探索目标彼此配合。返回由真实动作实现，日志块边界不会将物理世界复位。

$$
\rho_{\rm imag}={\alpha\over2}\rho_{g^*}+{\alpha\over2}\rho_0+(1-\alpha)\rho_{\rm replay}
$$

模型内 goal-conditioned policy 的采样分布：评测目标、初始状态与 replay 状态。它不是实际状态占用分布。

真实采样又不同：概率 α 执行前向目标与返回目标的一组 Go-Explore，概率 1−α 执行探索目标的 Go-Explore。相关 pair 耗时可为单个探索块的两倍，因此调度事件概率不等于原始时间占比。

$$
\operatorname{TimeShare}_{\rm relevant}\approx{\alpha\mathbb E[T_{\rm pair}]\over\alpha\mathbb E[T_{\rm pair}]+(1-\alpha)\mathbb E[T_{\rm explore}]}
$$

可积、稳定调度循环下的时间比例诊断。pair 耗时两倍、α=0.2 时份额约为 1/3；重尾或不稳定恢复应直接统计真实日志。

作者 resetfree/env.py 的分块 done 让 PEG 切换阶段，但保留物理环境；goal_picker_wrapper.py 决定调度。它是算法阶段边界，不能直接当作外部任务真实终止，也不应称为环境 reset。

| 消融 | 机制问题 |
| --- | --- |
| 只改真实调度 | 是否获得更多任务相关数据？ |
| 只改模型内目标 | 相同数据是否训练更有用的返回与前向策略？ |
| 共同改变 | 是否存在数据与训练目标的交互？ |
| 匹配原始步与模型计算 | 收益是否只是 pair 更长或虚拟训练更多？ |

**算法：区分论文能力评价与新增全生命评价**

1. 一次初始化后保留物理世界与学习器
1. 调度前向+返回 pair 或探索块，记录真实耗时
1. 保存数据并更新 world model，按指定分布训练条件策略
1. 评测副本：按论文的给定起点测试目标到达
1. 补充 CRL 评测：原生命收益、返回率、恢复时间与不可逆失败

已有证据支持 reset-free 训练与目标到达，但目标/初始分布由设计者给定、主要评测仍 episodic，world model 与 replay 均占资源。进一步 CRL 研究应检验未通知变化后如何更新返回区域与探索价值；真实机器人恢复保证还需独立验证。

<a id="lesson-check"></a>

## 11 · 自测与动手题

- 在走廊中只剩一个步时，路标仍减少多少不确定性？为什么最优动作却是直接试路？答案：1 bit，但读标后本次窗口没有行动读取它；直接行动期望为 0.5。
- 已知屏幕公平且独立，预测器设成刚看到的 +1 时，当前样本误差与下一帧期望误差各是多少？答案：0 与 2；保持均值 0 时下一帧期望误差为 1。
- 直接选错后 q 已经正确，为什么 H=6、k=4 时仍只得到 1？答案：错误动作和四个恢复步共消耗五步，最后一步才可重新选路。

- RND error 就是 Bayesian 信息增益吗？不是，它是固定随机函数的拟合误差，受表示、容量、优化和遗忘影响。
- 环境噪声大就一定值得探索吗？不一定；固有随机性可能不提供可减少的不确定性。
- ALP 为何会关注退步区域？绝对变化将遗忘或能力退化也当作重新练习的理由；正向 LP 则不这样做。
- 技能可辨认就一定对任务有用吗？不一定，必须验证其外部效用和复用接口。
- 有恢复概率估计是否意味着安全？不意味着，模型可能失准或遇到分布外状态。

动手题：给标量 RND 增加两组状态 A、B，长时间只训练 B，再返回 A，检查预测器是否因共享参数发生干扰；比较高误差来自真正未见状态，还是来自遗忘。然后以相同训练预算比较“选最大误差”与“选最大误差下降”的行为。

## 本章的实验设计

探索阶段的交互也属于总成本。区分访问覆盖、新预测学得速度与实际奖励。

设定：构造可学习的稀疏反馈区域和不可预测噪声区域，保持外部任务奖励不变。比较探索信号与真实行为结果。

- 高预测误差不被直接解释成高学习进步。
- 内在奖励和外在奖励分别记录。
- 探索与课程生成交互计入总预算。

对照：随机探索与固定探索强基线；噪声关闭/开启的受控条件；固定数据检查信号，再做闭环行为

记录：访问覆盖、目标发现时间；预测误差下降或学习进步；完整任务收益、恢复与探索成本

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-control)

## 学习与研究衔接

经验获取决定之后能学习什么。预测误差大可能只是噪声，未必代表学习进展或控制价值。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-exploration) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=exploration) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=exploration)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

教材可以先给定目标和技能集合；持续构造还要决定哪些行为值得练习、维护或放弃。谱结构、路径奖励、时间距离和语言先验提供不同候选偏置。先固定候选比较选择与组合，再改变生成器，才能辨认下游收益究竟来自哪一步。

- [Proper Laplacian Representation Learning](https://yingwen.io/zh/continual-rl/research/#recent-proper-laplacian-representations)
- [Reward-Aware Proto-Representations in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-reward-aware-proto-representations)
- [METRA: Scalable Unsupervised RL with Metric-Aware Abstraction](https://yingwen.io/zh/continual-rl/research/#recent-metra-skills)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)
- [Posterior Sampling for Continuing Environments](https://yingwen.io/zh/continual-rl/research/#recent-cpsrl-continuing-exploration)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

给定模型可研究怎样规划；模型也在学习时，规划会选择性地查询误差，并改变以后的数据。Dreamer、STOMP 和 DRAGO 分别研究想象控制、随机时长行为模型和旧知识保留。新的比较应固定规划查询与总预算，检验哪些后果误差真正改变选择，哪些维护值得继续。

- [Knowledge Retention in Continual Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-drago-model-retention)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [Posterior Sampling for Continuing Environments](https://yingwen.io/zh/continual-rl/research/#recent-cpsrl-continuing-exploration)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [Knowledge Retention in Continual Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-drago-model-retention)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [The Cell Must Go On: Agar.io for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-agarcl)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文规定对象与条件，架构路线提出组织方式，算法实验检验局部机制。撤掉阶段间冻结后，一个模块会改变另一个模块的学习问题；有限预算应优先维护哪条知识，成为新的决策。先检验两模块反馈和资源分配，再扩大整机，而不是由组件分别有效推断长期组合收益。

- [Plasticity as the Mirror of Empowerment](https://yingwen.io/zh/continual-rl/research/#recent-plasticity-mirror-empowerment)

### Proper Laplacian Representation Learning

Diego Gomez, Michael Bowling, Marlos C. Machado

ICLR 2024 · 2024 · 支持方法与理论

#### 研究问题

技能发现需要一组确定的谱方向，为什么仅学到低频子空间还不够？

#### 关键机制

图上的平滑性目标倾向保留缓慢变化的特征，但旋转后的同一子空间未必给出可解释、排序明确的单个特征向量。ALLO 使用增广 Lagrangian、正交条件与对称性破除，同时恢复特征向量和特征值，从而为 eigenoption 的方向构造提供更明确的输入。

#### 证据

论文分析优化目标，并在多个环境中检验谱表示的恢复质量和下游使用。作者仓库包含表示学习训练程序。

#### 条件与限制

谱结构依赖采样行为诱导的图和覆盖程度，不是脱离数据分布的环境真值。低频方向也不自动等于有奖励价值的技能；这正是奖励感知表示要继续处理的问题。

#### 阅读与实验

先在小图上直接求特征分解，再比较学习特征的子空间误差和逐向量误差。两种指标不等价，后者才揭示任意旋转问题。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2310.10833)：ICLR 2024 论文的公开版本。
- [ALLO 作者代码](https://github.com/tarod13/laplacian_dual_dynamics)：增广 Lagrangian 的实际优化与实验入口。

#### 作者代码

[论文作者的 ALLO 实现。](https://github.com/tarod13/laplacian_dual_dynamics)

Laplacian 表示学习和论文实验。

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

### METRA: Scalable Unsupervised RL with Metric-Aware Abstraction

Seohong Park, Oleh Rybkin, Sergey Levine

ICLR 2024 · 2024 · 支持方法与理论

#### 研究问题

没有外部任务奖励时，怎样发现能产生长距离、有区别状态变化的技能？

#### 关键机制

METRA 学习反映时间距离的潜在表示，并让技能方向 $z$ 最大化内在奖励 $r_z=(\phi(s')-\phi(s))^\top z$。邻接状态间的距离约束阻止编码器靠任意放大数值提高奖励。表示学习和技能策略相互影响，因此它不同于先固定一个表示、再单独训练 option。

#### 证据

论文在视觉与状态输入的运动、操纵任务中研究无监督技能学习和下游使用。作者代码包括约束优化、技能策略和相应实验配置。

#### 条件与限制

预训练技能加下游任务不等于技能库在单次生命内持续维护。理论距离约束与源码中的均方尺度、松弛量截断需要分别对照，不能只照抄一个简化公式重现。

#### 阅读与实验

观察表示范数、约束残差和实际位移三条曲线。若内在回报上升而位移不变，应先检查尺度和约束，而不是直接解释为探索改善。

#### 原文与相关入口

- [ICLR 原文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/516593a423838642a2eb4e9c5b9c7f44-Abstract-Conference.html)：方法与技能评价。
- [作者代码](https://github.com/seohongpark/METRA)：核心方法在 iod/metra.py；同时检查约束的归一化与截断。

#### 作者代码

[作者提供的论文实现。](https://github.com/seohongpark/METRA)

METRA、技能训练与下游评价。

### Knowledge Retention in Continual Model-Based Reinforcement Learning

Haotian Fu, Yixiang Sun, Michael Littman, George Konidaris

ICML 2025 · 2025 · 直接研究持续学习

#### 研究问题

当前任务不再访问旧区域时，世界模型怎样避免忘记那些区域的动力学？

#### 关键机制

DRAGO 将生成式旧经验、旧模型知识和探索结合起来。模型学习不只跟随当前奖励驱动的数据分布，还尝试维持对曾经学过区域的预测能力。它关注知识保留发生在模型里，而不只是保存旧策略输出。

#### 证据

论文在 MiniGrid 和连续控制任务中检验模型保留与任务表现，并给出原作者实现。关键设定包括共享状态与动力学、变化的任务奖励。

#### 条件与限制

已知任务切换、旧模型和数据资源是协议的一部分。共享动力学下的保留不能直接外推到动力学本身任意变化，也不是禁止重放的严格流式算法。

#### 阅读与实验

分别测量旧区域模型误差、旧任务回报与新任务学习速度。若模型误差改善而控制没有改善，再检查规划是否真正查询了被保留的知识。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/fu25f.html)：任务设定、模型保留和探索机制。
- [作者代码](https://github.com/YixiangSun/drago)：原文链接的 DRAGO 实验实现。

#### 作者代码

[作者 Yixiang Sun 的仓库；不使用同名第三方项目。](https://github.com/YixiangSun/drago)

DRAGO 的模型、重放、探索与实验。

### Plasticity as the Mirror of Empowerment

David Abel, Michael Bowling, Andre Barreto, Will Dabney, Shi Dong, Steven Hansen, Anna Harutyunyan, Khimya Khetarpal, Clare Lyle, Razvan Pascanu, Georgios Piliouras, Doina Precup, Jonathan Richens, Mark Rowland, Tom Schaul, Satinder P. Singh

NeurIPS 2025 · 2025 · 定义与架构观点

#### 研究问题

环境改变智能体的能力，与智能体改变环境的能力，能否放在统一的信息论框架中？

#### 关键机制

论文用广义有向信息描述两个方向：环境对智能体的影响对应一种可塑性，智能体对环境的影响对应赋能。统一表达使二者的关系和权衡可以被形式化，而不仅用神经元休眠或短期奖励间接描述。

#### 证据

贡献主要是概念定义和理论关系，提供研究长期交互的新坐标。它没有把信息量指标直接等同于某个具体神经网络算法的长期回报。

#### 条件与限制

信息论可塑性与“新目标拟合速度”不是相同估计量，也不等于参数变化越大越好。有限数据下怎样稳健估计这些信息量，需要额外方法。

#### 阅读与实验

分别举出高环境影响但低奖励、高赋能但不学习的过程。说明为什么两类能力与任务成功都需要独立评价。

#### 原文与相关入口

- [NeurIPS 2025 原文](https://papers.nips.cc/paper_files/paper/2025/hash/f04957cc30544d62386f402e1da0b001-Abstract-Conference.html)：统一定义、理论关系与解释。
- [作者预印本](https://arxiv.org/abs/2505.10361)：便于检索定义和证明。

### The Cell Must Go On: Agar.io for Continual Reinforcement Learning

Mohamed A. Mohamed, Kateryna Nekhomiazh, Vedant Vyas, Marcos M. José, Andrew Patterson, Marlos C. Machado

arXiv 预印本 · 2025 · 评价与实验协议

#### 研究问题

如何在持续、动态的高维交互里，同时研究记忆、探索、信用分配与学习能力保持？

#### 关键机制

AgarCL 提供持续运行的游戏环境，并用分解的小任务暴露不同困难。完整环境把这些机制放回同一交互循环，小任务则便于定位失败原因。游戏中的复活事件与把整个世界和智能体都重新开始不是同一种重置。

#### 证据

论文提供环境、基线与可塑性方法比较；部分常见修复在其测试中改善有限。这说明保持可塑性并不能单独代替记忆、探索和长期信用分配。

#### 条件与限制

一个游戏不能代表全部真实持续问题。小任务与完整游戏的协议需要分别阅读；此处仅按可确认的预印本状态收录，不把投稿信息写成会议录用。

#### 阅读与实验

先在一个小任务中验证机制，再检验它在完整环境中的作用是否仍存在。将世界重置、角色复活、参数重置和数据清空分开记录。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2505.18347)：环境设计、分解任务与基线结果。
- [作者环境仓库](https://github.com/machado-research/AgarCL)：环境安装、接口和运行示例；算法基线与环境本体分开。

#### 作者代码

[Machado 研究团队的环境实现。](https://github.com/machado-research/AgarCL)

AgarCL 环境与示例，非所有算法结果的单一训练脚本。

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

内部奖励/进展/恢复接口的确定性检查；非 RND、课程或安全算法完整性能复现。

[下载 lifelong_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)

```sh
python lifelong_algorithms_lab.py exploration
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, §2.1–2.2](http://incompleteideas.net/book/the-book-2nd.html)：从同一行动同时影响当前收益与未来可用信息的困难出发；本章走廊是原创有限任务。

- [Houthooft et al. · VIME: Variational Information Maximizing Exploration](https://arxiv.org/pdf/1605.09674)：§2.2–2.3，式 (2)–(7)：关于动力学参数的后验变化、信息增益与变分近似；本章两世界更新可精确枚举。

- [Bellemare et al. · Unifying Count-Based Exploration and Intrinsic Motivation](https://proceedings.neurips.cc/paper_files/paper/2016/hash/afda332245e2af431fb7b672a68b659d-Abstract.html)：从密度更新推导 pseudo-count；需满足学习正性并处理退化分母。

- [Burda et al. · Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)：冻结随机目标、预测器及内部/外部价值设置。

- [RND 作者代码](https://github.com/openai/random-network-distillation)：完整 Atari 训练系统；重点阅读 ppo_agent.py、policies、归一化与 predictor 更新。

- [Portelas et al. · Teacher algorithms for curriculum learning](https://proceedings.mlr.press/v100/portelas20a.html)：ALP-GMM 的连续环境参数调度，不等于无环境生成权限的 single-life 问题。

- [ALP-GMM 作者实现 · teachDeepRL](https://github.com/flowersteam/teachDeepRL)：包含 toy teacher 环境和参数化 BipedalWalker；从 toy_env 开始核对课程机制。

- [Eysenbach et al. · Diversity is All You Need](https://arxiv.org/abs/1802.06070)：技能可辨认性与最大熵策略，不应把多样性视为外部效用保证。

- [Eysenbach et al. · Leave no Trace](https://arxiv.org/abs/1711.06782)：将前向学习与恢复策略共同建模，理解自主学习中的 reset 成本。

- [Ishfaq et al. · Langevin Soft Actor-Critic](https://proceedings.iclr.cc/paper_files/paper/2025/file/2420e12b7af4ac1e411fa6000576ffbd-Paper-Conference.pdf)：ICLR 2025：近似 critic 参数采样、分布式回报与生成样本；需区分认识不确定性与回报噪声。

- [LSAC 作者实现](https://github.com/hmishfaq/LSAC)：从 lsac.py 及 configs/gym_exp.json 对应 critic、生成器与训练预算；需要论文指定的连续控制依赖。

- [作者论文 v3](https://arxiv.org/html/2408.09807v3)：训练与评价协议、back-and-forth exploration 与目标分布。

- [TMLR 作者项目页](https://yangzhao-666.github.io/morefree/)：正式发表状态与作者代码链接。

- [Reset-free Reinforcement Learning with World Models · 作者实现](https://github.com/yangzhao-666/MoReFree)：resetfree/env.py、goal_picker_wrapper.py、Dreamer/PEG 与目标条件实验。 TMLR 作者项目页明确链接的官方实现。

- [RLC 2024 原文](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_277.pdf)：随机重采样、折扣联系与 Bayesian regret 假设。

- [RLJ 论文记录](https://rlj.cs.umass.edu/2024/papers/Paper277.html)：作者、会议与理论结果。
