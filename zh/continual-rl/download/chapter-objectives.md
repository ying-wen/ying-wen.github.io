# 强化学习问题的形式化：交互、目标与持续学习

一个长期运行的智能体应当优化什么？这个选择怎样影响状态、价值函数、学习算法和评价？

## 本章内容

- 先写出定义域、给定量、未知量、信息权限、候选解和目标，再选择求解方法。
- 从奖励序列推导回报与价值函数，比较回合总奖励、折扣目标和平均奖励。
- 说明随机停止解释和势函数塑形的成立条件，计算条件失效时的反例。
- 在同一资源预算下比较整个学习过程与冻结策略，区分 continuing、continual 和 non-stationary。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [MDP、回报与价值：序列决策的数学对象](https://yingwen.io/zh/continual-rl/foundations/tabular/mdps/)：明确交互、回报和策略价值，再把比较对象扩展为持续更新的学习器。


### 期望

同一策略可能产生多条轨迹。期望按这些轨迹的概率加权，而不是只选成功轨迹。

$$
\mathbb E[X]=\sum_x P(X=x)x
$$

### 条件概率

给定已经看到的历史和选定动作，描述下一条经验的分布。条件中不能包含尚未到达的信息。

$$
P(O_{t+1},R_{t+1}\mid H_t,A_t)
$$

### 几何级数

固定比例衰减的无限和有闭式解。折扣回报、随机停止和周期算例都用到它。

$$
\sum_{k=0}^{\infty}\gamma^k=\frac1{1-\gamma},\qquad0\le\gamma<1
$$

### 马尔可夫状态

若状态保留了预测下一步奖励和状态所需的历史信息，就能支持一步递推。当前图像不一定满足这个条件。

<a id="problem-definition"></a>

## 本章的问题定义

长期交互之前，先决定评价谁、从什么起点、在多长时间内比较哪些后果。奖励接口、决策信息和求解方法分别规定。

### 给定条件与符号

- 智能体—世界边界、观测与动作接口、奖励生成规则。
- 初始世界分布、评价长度、折扣或平均准则，以及记忆、计算、预训练和重置权限。

### 需要求解的对象

一个能比较完整学习智能体的评价准则；随后才定义与该准则相符的预测和控制问题。

### 信息与数据权限

$H_t=(O_0,A_0,R_1,O_1,\ldots,O_t)$ 是动作前已经收到的历史；学习智能体 $\Lambda$ 只能由允许的历史产生 $A_t$。

$$
J_{T,\gamma}(\Lambda)=\mathbb E_{\Lambda}\!\left[\sum_{t=0}^{T-1}\gamma^tR_{t+1}\right]
$$

$T$ 是评价长度，$0\le\gamma\le1$ 是时间权重，$R_{t+1}$ 是执行动作后收到的奖励。有限 $T$ 的总奖励、无限折扣回报和存在极限时的长期平均奖励是不同选择；此式是其中一个规格，不是默认适用于全部章节。

### 成立条件与解的含义

- 期望必须存在；无限折扣的有界奖励是一个充分条件。
- 比较对象的信息与资源权限一致；训练中止、真实终止、任务切换和人工重置分别定义。

判断准则：同一组手算奖励序列能按选定准则复算排序；短期与长期排序反例、终止边界和资源成本均能由规格解释。

### 适用边界

- 不从一个奖励公式推出其符合设计者意图。
- 不因价值可以定义就假定有限资源智能体能够精确求解。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [奖励假设与奖励设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/)：评价准则提出希望实现的偏好，奖励设计进一步检查实际学习信号是否表达它。

- 推广：放宽条件 · [平均奖励与差分价值](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)：本章讨论更一般的评价选择，平均奖励只是其中的长期单位时间准则；有限寿命和折扣目标也在本章范围内。

- 组合不同学习问题 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：控制把明确的准则应用于策略或完整学习器的可行比较。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

不同时间权重、终止语义和比较对象会改变同一轨迹的排序。

### 本章的核心思路

先固定评价对象和边界，再从回报拆分推出价值；算法损失必须沿这条定义链解释。

1. [将任务与信息接口分开](#lesson-setting)：因为观测只描述可见信息，先给定奖励与历史接口，避免用状态编码替代目标。

2. [从奖励序列定义回报](#lesson-derive)：因为价值依赖未来行为和时间权重，先拆分回报，再对指定行为取条件期望。

3. [用排序反例检验准则](#lesson-lifetime)：因为折扣、平均奖励和寿命表现可能冲突，用同一后果序列与完整学习过程检查所选准则。

结论与条件：回报递推和可积条件期望是定义层面的结论；不承诺行为可学、可达或最优。

### 相关方法改变了什么

- 折扣回报：按时间位置赋予几何权重，适合明确采用这种偏好的任务。

- 平均奖励：比较长期单位时间收益，需要极限和链结构条件。

- 有限寿命评价：保留启动、探索与恢复成本，长度本身是协议的一部分。


<a id="lesson-setting"></a>

## 1. 先定义问题：接口、未知量、可用信息与解

讨论怎样求解之前，先要明确问题：允许哪些交互、决策时知道什么、比较哪些候选对象、什么结果算更好。价值函数、网络、TD 更新和规划都是在这些条件下求解的工具。改变其中任意一项条件，都应重新检查原来的结论。

$$
\begin{aligned}
 &O_t\in\mathcal O,\quad A_t\in\mathcal A,\quad R_{t+1}\in\mathcal R\subseteq\mathbb R,\\
 &\mathcal H_t=\mathcal O\times(\mathcal A\times\mathcal R\times\mathcal O)^t,\qquad
 \mathcal H=\bigcup_{t\ge0}\mathcal H_t,\\
 &H_t=(O_0,A_0,R_1,O_1,\ldots,A_{t-1},R_t,O_t),\\
 &O_0\sim\nu_0,\qquad A_t\sim\Lambda(\cdot\mid H_t),\qquad
 (O_{t+1},R_{t+1})\sim P_t(\cdot\mid H_t,A_t).
 \end{aligned}
$$

$\mathcal O,\mathcal A,\mathcal R$ 是观测、动作和标量奖励的定义域；$\nu_0$ 是初始观测分布。$P=(P_t)_{t\ge0}$ 是环境响应核，$\Lambda$ 是因果行为规则，输出动作的概率分布。$H_t$ 只含动作前已到达的信息。集合可以有限、可数或连续；连续情形要求核可测，公式中的求和相应改成积分。

这个一般历史过程允许环境有记忆、只被部分观测，或随时间改变响应。$P_t$ 上的时间下标只是允许响应随时间变化，并不证明环境非平稳。完整历史总能作为形式上的状态，但历史空间会随时间增长；智能体未必能保存完整历史，也未必能计算它的精确价值。

| 问题规格 | 必须明确的内容 | 解的含义 |
| --- | --- | --- |
| 定义域与给定量 | 动作、观测、奖励、初始条件、时间单位、评价期限与终止规则。 | 候选对象必须在此接口与时间边界下运行。 |
| 未知量 | 真实响应核 $P$、隐藏状态或奖励参数可未知；给出允许的环境族 $\mathcal E$。 | 未知模型的学习问题与已知模型的规划问题不同。 |
| 可用信息 | 部署时可见的历史、任务标签、模型调用、预训练数据和重置权限。 | 动作及更新不得读取未来奖励、隐藏真状态或未授权测试数据。 |
| 候选集合 | 指定固定策略类 $\Pi$，或完整实现集合 $\mathfrak L$；预算可限制后者。 | 最优只相对于规定的候选集合，不能默认其中包含可以使用任意计算量的理想智能体。 |
| 目标与比较规则 | 给出 $J_P$，并说明逐环境保证、环境先验平均、最坏情形或指定比较器。 | 一次运行的得分、期望性能与跨环境保证不同。 |
| 假设与精度 | 可测性、奖励可积性、Markov 性、平稳性、可达性及允许误差。 | 若上确界不能达到，应给近似解或性能保证，不能默认最优解存在。 |

$$
J_P(\mathcal L)=\mathbb E_{P,\mathcal L}[U(\tau)],\qquad
 J_P^*=\sup_{\mathcal L\in\mathfrak L}J_P(\mathcal L),\qquad
 J_P(\mathcal L_\varepsilon)\ge J_P^*-\varepsilon
$$

$\mathcal L$ 是含初始化、行动、学习和记忆更新的可执行智能体；它诱导行为规则 $\Lambda$。$\tau$ 是完整交互轨迹，$U$ 是规定的轨迹评分，要求期望存在；$\varepsilon\ge0$ 是容许性能差距。该式在一个固定 $P$ 中定义比较标准，并未允许未知模型的学习器预先读取 $P$。平均奖励等准则在后文直接定义，不必都写成某个无限轨迹评分的期望。

若要设计能用于不同未知环境的同一个学习器，还要规定跨环境的要求。例如给定环境先验 $\eta$，优化 $\mathbb E_{P\sim\eta}[J_P(\mathcal L)]$；或在环境族 $\mathcal E$ 中优化 $\inf_{P\in\mathcal E}J_P(\mathcal L)$。前者是先验平均，后者是最坏情形；按这两种要求选出的学习器一般不同。单个真实环境的最优值可以作理论比较器，但学习器并不会因此获得额外信息。

考虑一个反复经过岔路口的小迷宫。“在接下来的 48 步里获得更多奖励”规定目标。“保留最近哪边有奖励”规定信息摘要。“失败后换边，成功后留在原边”规定学习规则。下面让左、右路线的奖励规律交替出现，观察同一个迷宫怎样产生策略求解、跨任务比较与持续适应这几种问法。

智能体与环境的边界是建模边界。它把所研究的决策和学习过程，与提供观测、奖励及动作后果的其余过程分开。边界不必沿机器人的外壳划定。若研究高层导航，电机控制器可以属于环境；若研究电流控制，它就属于智能体。奖励生成逻辑相对这个学习器位于环境一侧，不能由当前策略任意改写评分规则。

| 问题 | 数学对象 | 典型研究内容 |
| --- | --- | --- |
| 哪些结果更好？ | 奖励与轨迹评价 $J$ | 折扣、平均奖励、风险约束、生命期收益。 |
| 过去的哪些信息要保留？ | 历史摘要 $h_t$ | Agent state、部分可观测性、记忆。 |
| 未来会发生什么？ | 条件预测、价值函数和模型 | GVF、预测校准、转移模型。 |
| 怎样改变行为？ | 策略、学习规则和规划过程 | TD、actor–critic、元学习、模型规划。 |

Sutton 与 Barto 的 reward hypothesis 把目标和目的理解为期望累计标量奖励的最大化。这是研究与建模主张，不是无需条件便成立的数学定理。它没有指定奖励的具体数值，也没有声称任意给出的奖励都准确表达设计者的意图。Bowling 等在 ICML 2023 研究了更精确的表述：给定对经验结果的偏好关系，哪些条件使奖励表示成为可能。其结论依赖所列假设，不等于所有现实偏好都满足这些假设。

一个任务定义至少要写出动作、观测、奖励、时间单位、终止或重置规则，以及评价方式。安全约束、允许使用的信息和计算预算也需要明示。“是否完成任务”与“每秒完成量减去能耗”不是同一个目标。

理解这些关系，要先看问题的哪一项条件发生了变化。问题包含只说明某类问题能嵌入另一类；替换目标可能改变解；方法中的近似可能引入误差；协议限制则改变可行集合。这四种关系不能用一根“越来越先进”的箭头替代。

先看一个会反复经过的岔路口 J。向左或右走到侧格，领取奖励后再走回 J，每圈两步。规则 A 让左边奖励为 1、右边为 0，规则 B 则反过来。只运行 A 时，位置足以描述这个平稳有限 MDP。若按 A、B、A 切换，既可问每种规则的好策略是什么，也可问一个持续更新的学习器怎样利用旧经验适应下一段。两种问法共享迷宫，却比较不同对象。

![四层嵌套区域：一般历史交互、平稳有限隐藏状态的 POMDP、平稳有限 MDP、平稳独立 bandit；每层给出一个可视化场景。](https://yingwen.io/crl-figures/concept-crl-setting-axes.svg)

从外向内依次增加模型条件。揭示 POMDP 的完整状态得到 MDP；再限制为单状态和独立拉臂奖励，得到这里的标准 bandit。图中的网格是例子，不是模型定义。深度网络、持续学习和多任务没有画成新的内层集合，因为它们分别改变求解方式、学习与评价问题、任务安排。

环境模型描述后果如何产生，任务安排描述何时遇到哪些控制问题，学习问题则规定要比较哪些会行动和更新的智能体实现。这些条件可以组合：把本例的位置与周期时钟写成真状态 $X_t$，整个世界具有固定的 Markov 核；若接口只给出位置，智能体就看不到这个时钟。即使给出了这样的形式状态描述，仍要说明智能体怎样得到相关信息、可保存多少知识以及怎样评价它。

这个迷宫也能用来说明多任务与持续学习的区别。一种多任务实验允许在 A、B 两种规则下反复采样，并把当前任务标签交给策略。策略可以学成“见 A 选左，见 B 选右”，然后在两种任务上冻结测试。共享知识是否改善各任务的表现，是这种实验的一个主要问题。

换成这里的持续部署协议：智能体按既定顺序遇到 A、B、A，不能为了补习而请求切回 A，也没有任务标签。为了发现规律已经变化，它至少需要新的证据；每次试错都计入收益。此时要比较的是同一个学习器怎样使用过去经验、何时改道，以及在整个过程中获得多少奖励。仅凭学习结束后的两个测试分数，无法还原这些过程信息。

这些是两份具体协议，而不是对所有多任务或持续学习研究的规定。任务标签、旧任务访问、参数重置、回放和计分时段都可以单独改变；给出标签的顺序任务也能研究持续学习。区别因此不在“任务多不多”，而在经验怎样到来、哪些知识必须保留，以及评价最终策略还是完整学习过程。

![五组画面对比完整视野与局部视野、重复试验与单条生命轨迹、经验回放与统计记忆、暂停与持续运行的时钟、显式任务标签与奖励反馈。](https://yingwen.io/crl-figures/concept-crl-setting-permissions.svg)

逐行比较，跨行组合。例如，一个完全可观测 MDP 也可以禁止外部重置并要求严格流式更新；一个多任务实验也可以不给任务标签。撤去一项便利，会改变需要推断的信息或可执行的算法。最后还要规定计分对象：冻结策略，或继续行动和更新的完整学习器。

| 两个概念 | 关系类型 | 成立条件 / 改变量 | 带来的结构与边界 |
| --- | --- | --- | --- |
| 一般历史过程 → MDP | 附加环境结构 | 存在可观测充分状态 $S_t$；响应只依赖 $(S_t,A_t)$。若核还不随时间改变，得到平稳 MDP。 | 带来状态上的 Bellman 递推；不保证状态有限，也不保证模型已知。 |
| MDP 与 POMDP | 信息结构与问题包含 | POMDP 的隐藏状态满足 Markov 条件，但接口只提供观测；MDP 是状态被完全揭示的特例。 | 在有限已知模型下，信念是充分统计量；它通常属于连续空间，计算仍可能困难。 |
| MDP 与 bandit | 去掉受动作控制的后续状态 | 标准无上下文 bandit 可写成一个观测状态、每次拉臂生成奖励的 MDP；一般 bandit 的奖励过程还需单列假设。 | 未知臂收益仍需探索；没有受控状态的长程后果，不等于没有长期学习目标。 |
| 折扣 ↔ 有限期限 ↔ 平均奖励 | 替换评价准则 | 对同一奖励过程，改变各时刻奖励的汇总方式。 | 一般不能保证最优行为不变；有限启动代价与任务目标章的周期路线给出反例。 |
| 完整历史 → 有限记忆 / 神经网络 | 限制表示或求解方法 | 把历史压缩到可实现摘要，并用参数共享近似价值、策略或模型。 | 摘要未必充分；网络变深不构成新的环境类别，也不保证最优值可表示。 |
| 平稳 ↔ 外部非平稳 | 改变响应假设 | 相对于已声明的状态，转移或奖励核是否随时间改变；时间扩张可恢复形式上的齐次性。 | 扩张可能需要无界、不可观测或未知的状态；策略变化引起的数据漂移不自动是环境变化。 |
| 一般学习器 → 严格流式 / single-life | 限制交互与实现协议 | 分别限制经验回放与外部重置；两者可独立设置。 | 可行学习器集合缩小，最优值可能下降；不是新的奖励准则。 |
| 有界智能体 + Big World | 预算限制与研究假设 | 前者精确定义可用内存和计算；后者主张关注环境复杂性超过智能体能力的问题。 | 有限容量不能在逻辑上推出永久学习的必要性；是否需要持续适应还取决于环境、目标与候选集合。 |
| 一般 RL → Abel 等的 CRL | 相对基底的性质判定 | 固定环境、性能、候选智能体及生成它们的 agent basis，再判断所有最优者是否永不到达基底。 | 不是 MDP → 深度 RL → CRL 的升级链，也不取消价值函数。 |

下面先选择评价准则与评价对象，再说明 Markov 假设带来的递推结构。第 3–7 节检查期限、停止、奖励率和奖励变换；第 8 节区分辅助问题。第 9 节把学习与资源限制纳入候选对象，随后回到这个岔路迷宫，逐步算出全程收益、冻结诊断、保持与迁移。

<a id="lesson-derive"></a>

## 2. 选择准则与评价对象，再构造价值递推

明确交互条件后，还要选择怎样评价奖励序列。有限期限适合已知部署时长；折扣给远期收益较小权重；平均奖励关注长期单位步收益。自然终止、重置和采样截断规定了过程的边界，不能从评价公式反推。

$$
\begin{aligned}
 J_{P,T}(\Lambda)&=\mathbb E_{P,\Lambda}\!\left[\sum_{t=0}^{T-1}R_{t+1}\right],\\
 J_{P,\gamma}(\Lambda)&=\mathbb E_{P,\Lambda}\!\left[\sum_{t=0}^{\infty}\gamma^tR_{t+1}\right],\quad0\le\gamma<1,\\
 g_P(\Lambda)&=\liminf_{T\to\infty}\frac1T\mathbb E_{P,\Lambda}\!\left[\sum_{t=0}^{T-1}R_{t+1}\right].
 \end{aligned}
$$

$T\ge1$ 为指定期限，$\gamma$ 为折扣系数；$P$ 与 $\Lambda$ 共同诱导轨迹。这里 $g_P$ 使用期望平均的下极限，避免预设普通极限存在。奖励有界是这些量有限的一组充分条件。若每步实际时长不同，单位时间评价需另外定义。这些准则均可用于固定策略或完整学习器；选择评价对象与选择评价准则，是两件独立的事。

| 选择 | 突出什么 | 仍需补充什么 |
| --- | --- | --- |
| 有限期限累计奖励 | 计入期限内探索、启动与恢复成本。 | 期限和初始分布；随机自然终止还需检查可积性。 |
| 折扣回报 | 区分近期与远期收益。 | 折扣与真实时间的关系；固定折扣一般不能替代动作相关生存概率。 |
| 长期平均奖励 | 比较持续运行的长期产出。 | 链结构或下极限约定；有限启动损失可能被抹去。 |

不等价的最小反例：在初始状态选择路线，随后不可切换。A 每步奖励 1；B 第一步奖励 $-100$，此后每步奖励 2。期限 $T=10$ 时，A 得 10，B 得 $-82$；折扣 $\gamma=0.9$ 时，A 值为 10，B 值为 $-100+2\gamma/(1-\gamma)=-82$；长期平均却分别为 1 和 2。另两条路线 A 恒得 1、B 周期得到 $(0,0,4)$，可使有限期限、不同折扣与平均奖励各自改变排名；任务与目标章的周期路线算例给出完整计算。

策略评价、控制和学习器设计回答三个不同的问题。策略评价给定未来行为，求其收益预测；控制在给定策略类中选择行为；学习器设计则比较产生行为和持续更新的整个可执行过程。学习器虽然会改变参数，但其初始化和更新程序可以固定，因此在一般历史空间上仍诱导一个固定的行为规则。

| 问题 | 给定 / 未知 | 求什么 | 核心求解难点 |
| --- | --- | --- | --- |
| 固定策略评价 | 给定 $\pi$、目标准则与模型或样本；真实价值未知。 | 计算或估计条件回报。 | 未知环境带来采样误差；表示限制带来近似误差；不是直接最大化收益。 |
| 控制 | 给定环境接口、评价与策略类 $\Pi$；优良策略未知。 | 求 $\sup_{\pi\in\Pi}J_P(\pi)$ 或近似最优策略。 | 动作改变未来状态、收益与信息；价值估计与行为改善相互作用。 |
| 完整学习器设计 | 给定信息权限、环境族与实现预算；候选 $\mathcal L$ 诱导未来更新。 | 按指定的跨环境规则，比较实际整段交互中的表现。 | 探索、更新、记忆及规划都产生资源成本和行为后果，最后的参数不足以代表整个过程的表现。 |

$$
\begin{aligned}
 V_{P,\Lambda}^{\gamma}(h)&=\mathbb E_{P,\Lambda}\!\left[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}\mid H_t=h\right],\\
 V_{P,\Lambda}^{\gamma}(h)&=\sum_a\Lambda(a\mid h)\int\!
 \left[r+\gamma V_{P,\Lambda}^{\gamma}(h\mathbin{\Vert}(a,r,o))\right]P_t(do,dr\mid h,a).
 \end{aligned}
$$

$h\in\mathcal H_t$ 为可实现历史，$h\mathbin{\Vert}(a,r,o)$ 表示接上一条经验，$o,r$ 为下一观测和奖励的取值。动作求和写于可数动作情形，连续动作改为积分；连续历史的条件期望用正规条件分布并按几乎处处意义理解。奖励有界且 $\gamma<1$ 时，一步拆分和条件期望给出递推。若 $\Lambda$ 来自学习器，未来更新已经包含在其后续历史行为中；这是学习器的历史价值，不是冻结当前参数后的价值。

因此，CRL 可以有价值函数；变化的是条件信息、未来行为规则及可行比较器。写出历史递推，还没有得到紧凑、可计算的解。Bellman 等式成立，并不等于 TD 能在有限表示中准确求解。冻结参数时，还必须说明记忆递推、探索、模型和优化器是否也被冻结，才能明确“固定策略”指什么。

奖励 $R_{t+1}$ 是一次转移后收到的标量。回报 $G_t$ 是对未来奖励进行聚合的随机变量。策略 $\pi$ 规定动作分布。价值函数则是给定未来行为和当前信息后的期望回报。它们分别是信号、聚合规则、行为规则和预测量。下面的 $v_\pi,q_\pi$ 使用充分 Markov 状态 $S_t$ 与固定平稳策略 $\pi(a\mid s)$；一般历史行为的价值已经在上式定义。

$$
G_t^\gamma=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1},\qquad0\le\gamma<1
$$

若 $|R_t|\le R_{\max}$，则 $|G_t^\gamma|\le R_{\max}/(1-\gamma)$。较远奖励权重更小是目标定义的一部分，并不只是数值优化技巧。

$$
\begin{aligned}G_t^\gamma&=R_{t+1}+\gamma\sum_{k=0}^{\infty}\gamma^kR_{t+k+2}=R_{t+1}+\gamma G_{t+1}^\gamma,\\v_\pi(s)&=\mathbb E_\pi[G_t^\gamma\mid S_t=s],\\q_\pi(s,a)&=\mathbb E_\pi[G_t^\gamma\mid S_t=s,A_t=a].\end{aligned}
$$

第一行只把第一项从级数中取出。后两行引入期望与条件。$q_\pi$ 表示当前执行指定动作后，未来继续使用 $\pi$；它不要求这个动作是 $\pi$ 最常选择的动作。

若决策时能获得状态 $S_t=f(H_t)$，且任意可实现历史都满足以下条件，就可以用状态核描述环境响应。若未来行为也只依据这个状态与固定策略，才可进一步将历史价值压缩成状态价值。这里 $f$ 是状态构造函数，$\mathcal S$ 是状态空间；状态是否充分，要根据条件分布判断，称它为“state”并不能保证这一点。

$$
\Pr(S_{t+1}=s',R_{t+1}=r\mid H_t,A_t=a)
 =p(s',r\mid S_t,a)
$$

左侧的条件律由环境响应与状态构造共同决定：先生成 $(O_{t+1},R_{t+1})$，再令 $S_{t+1}=f(H_{t+1})$。这里并非直接把观测核 $P_t$ 的输出改名。右侧的 $p$ 是不随时间变化的转移与奖励核。此式同时声明 Markov 性和环境平稳性；若右侧改为 $p_t$，仍可研究时间不齐次的 Markov 问题。有限 MDP 还要求 $\mathcal S,\mathcal A$ 有限。已知 $p$ 时可直接规划，未知 $p$ 时则需经验、模型估计或无模型更新。

POMDP 假设隐藏状态 $X_t\in\mathcal X$ 满足 Markov 演化，但可见的 $O_t$ 由观测机制生成。即使潜在核固定，单独的观测也未必 Markov。若有限隐藏状态模型和初始分布已知，可以用信念 $b_t(x)=\Pr(X_t=x\mid H_t)$ 决策；奖励若提供状态信息，也必须进入信念更新。未知模型时，仅保存状态信念通常还不够，可能需要模型参数的后验。

$$
\begin{aligned}
 K(x',o,r\mid x,a)&=\Pr(X_{t+1}=x',O_{t+1}=o,R_{t+1}=r\mid X_t=x,A_t=a),\\
 b_{t+1}(x')&=\frac{\sum_x b_t(x)K(x',O_{t+1},R_{t+1}\mid x,A_t)}
 {\sum_{y,x}b_t(x)K(y,O_{t+1},R_{t+1}\mid x,A_t)}.
 \end{aligned}
$$

$K$ 是已知的联合潜在转移、观测与奖励核；$x,x',y\in\mathcal X$，$o\in\mathcal O$，$r\in\mathcal R$。分母要求观测事件有正概率；连续观测或奖励使用相应密度或正规条件分布。信念在已知模型下是充分统计量，但通常有连续取值，精确更新和控制未必在预算内。

无上下文、平稳、独立拉臂的 bandit 是更简单的特例：环境只有一个可见状态，动作 $a$ 从未知奖励分布 $\rho_a$ 取得一次奖励，再回到同一状态。若 $\rho_a$ 已知，控制可直接选均值最大的臂；若未知，完整学习器还需权衡当次收益与信息。Contextual、非平稳或相关奖励 bandit 需要各自声明上下文生成、漂移和依赖条件，不能全部套入这个特例。

历史价值已经有一步递推。进一步使用充分 Markov 状态、平稳环境与固定平稳策略 $\pi(a\mid s)$，才能把递推压缩到同一个状态函数 $v_\pi$。对第一步动作与结果分组求期望，得到下面的 Bellman 方程。若剩余期限影响决策，应把必要时间信息纳入状态，或使用时间索引的价值函数。

$$
\begin{aligned}v_\pi(s)&=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)[r+\gamma v_\pi(s')],\\q_\pi(s,a)&=\sum_{s',r}p(s',r\mid s,a)\left[r+\gamma\sum_{a'}\pi(a'\mid s')q_\pi(s',a')\right].\end{aligned}
$$

有限集合对应求和，连续量对应积分。方程描述价值应满足的关系；TD 和动态规划则是求解它的方法。

$$
J_\gamma(\pi;d_0)=\mathbb E_{S_0\sim d_0}[v_\pi(S_0)],\qquad\pi^*\in\operatorname*{arg\,max}_{\pi\in\Pi}J_\gamma(\pi;d_0)
$$

初始分布 $d_0$ 和可用策略集合 $\Pi$ 也是问题定义的一部分。真实值 $v_\pi$ 与网络估计 $v_w$ 不同。降低预测误差是在改进估计，不直接等于提高策略收益。

在有限 MDP、有限动作、有界奖励和 $\gamma<1$ 下，允许全部平稳策略时，Bellman 最优算子在最大范数下是 $\gamma$ 收缩；其唯一不动点可由价值迭代逼近，贪心策略达到最优。这是“附加结构怎样帮助求解”的具体结果。若把策略限制为某个网络类，或把历史压缩为不充分的摘要，逐状态贪心策略未必可表示，原收缩结论也不自动适用于参数更新。

$$
(\mathcal T_*v)(s)=\max_a\sum_{s',r}p(s',r\mid s,a)[r+\gamma v(s')],\qquad
   \|\mathcal T_*v-\mathcal T_*u\|_\infty\le\gamma\|v-u\|_\infty
$$

$\mathcal T_*$ 为最优 Bellman 算子，$u,v:\mathcal S\to\mathbb R$ 为候选状态价值，$\|v\|_\infty=\max_s|v(s)|$。这是精确有限问题的结构；TD、函数逼近和 actor–critic 各自仍需分析数据、投影和更新的条件。

固定策略问题是一个基础对象。在线学习时，未来行为还受尚未发生的更新影响。评价整个学习器应把学习状态纳入过程。当前冻结策略的价值，不能完整描述这个学习器今后的实际收益。

<a id="lesson-boundaries"></a>

## 3. 回合、持续任务和采样截断

Episodic 任务给出真实结束时刻 $T$，例如一局棋的胜负终局。未折扣回报是到终止为止的奖励和。奖励有界且期望 episode 长度有限，是保证期望回报有限的一组充分条件。不能仅凭“最终会结束”就默认所有期望存在。

$$
G_t=\sum_{k=t}^{T-1}R_{k+1},\qquad v_\pi(S_T)=0
$$

终止后没有待计入的任务奖励，因此尾值为零。折扣版 episodic 目标还给各项乘 $\gamma^{k-t}$。终止规则与折扣系数是两个独立选择。

Continuing 任务没有自然的最终结束。机器人充电、服务器等待请求、失败后恢复，都可能是持续过程中的普通转移。是否重置位置、是否清空记忆、重置需要多少时间和代价，要分别规定。把这些过程删掉会改变所评价的系统。

采样截断只表示记录或计算到此为止。例如保存 $n$ 步 rollout，但任务还会继续。此时后续收益没有消失。对固定策略，用真实价值补齐尾部有下面的恒等式；实际算法使用估计值，所以还会有估计误差。

$$
v_\pi(s)=\mathbb E_\pi\!\left[\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^n v_\pi(S_{t+n})\mid S_t=s\right]
$$

真正终止时尾值为零；单纯 timeout 通常保留尾值。若时间上限本来就是任务定义，应把剩余时间纳入状态，并在期限处使用终止边界。

每步奖励恒为 1、$\gamma=0.9$ 时，持续价值为 10。只记录一步，应有 $1+0.9\times10=10$。把采样边界错当终止，只得到 1。差值来自目标边界错误，与网络容量无关。

<a id="lesson-stopping"></a>

## 4. 折扣何时等价于随机停止

先假想一条完整的潜在奖励轨迹，再独立抽取停止时间 $N\ge1$。第一条奖励一定计入。之后每收完一条奖励，以概率 $\gamma$ 继续，因此 $P(N>k)=\gamma^k$。这个独立性是推导的关键。

$$
\begin{aligned}\mathbb E\!\left[\sum_{k=0}^{N-1}R_{t+k+1}\right]&=\sum_{k=0}^{\infty}\mathbb E[\mathbf1\{N>k\}R_{t+k+1}]\\&=\sum_{k=0}^{\infty}P(N>k)\mathbb E[R_{t+k+1}]\\&=\mathbb E[G_t^\gamma].\end{aligned}
$$

有界奖励和 $\gamma<1$ 保证可以交换和与期望。第二行使用停止事件与潜在奖励轨迹独立。若失败概率依赖动作或状态，通常需要状态相关的 continuation，而非统一常数 $\gamma$。

$$
\mathbb E[N]=\sum_{k=0}^{\infty}P(N>k)=\frac1{1-\gamma}
$$

这是有效时间尺度，不是硬截断。$\gamma=0.99$ 时平均 100 步，不表示第 101 步之后的奖励权重为零。还要说明一步对应多少真实时间。

潜在奖励为 $(1,2,3)$，之后为零，$\gamma=0.5$。第一步后停止的概率为 0.5，累计奖励为 1；第二步后停止的概率为 0.25，累计奖励为 3；至少运行三步的概率为 0.25，累计奖励为 6。期望为 $0.5\times1+0.25\times3+0.25\times6=2.75$，等于折扣和 $1+0.5\times2+0.25\times3$。

独立性失败的例子：第一步奖励为零。公平硬币同时决定是否继续，以及下一条潜在奖励是 2 还是 0；只有奖励为 2 时才继续。实际第二步贡献的期望为 1。若用继续概率 0.5 乘无条件奖励均值 1，则错误地得到 0.5。动作相关的失败风险不能直接由全局折扣替代。

枚举停止位置，独立计算期望；不调用折扣求和函数。

```python
def stopped_return_exact(rewards, gamma):
    """Expected undiscounted sum with an independent geometric stopping time.

    Collect the first reward for sure, then survive after each reward with
    probability gamma. Rewards beyond the supplied list are zero. The last
    outcome combines all stopping times at or beyond the end of the list.
    """
    discount(gamma)
    rewards = finite(rewards, 'rewards')
    expectation = 0.0
    partial_sum = 0.0
    for index, reward in enumerate(rewards):
        partial_sum += reward
        survival = gamma ** index
        probability = survival if index == len(rewards) - 1 else (1 - gamma) * survival
        expectation += probability * partial_sum
    return expectation
```

<a id="lesson-average"></a>

## 5. 平均奖励与差分价值

若任务关注长期单位时间产出，可以使用平均奖励。它把有限长度的累计收益除以经过的时间，再取极限。它不是未折扣的无限奖励和，也不是在折扣回报公式中直接令折扣等于一。

$$
g_\pi(s)=\lim_{T\to\infty}\frac1T\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}R_{t+1}\mid S_0=s\right]
$$

极限需要存在。有限单常返类等条件可使长期奖励率与初始状态无关，这时简写为 $g_\pi$。多常返类可能具有不同奖励率。每步时长不等时，还需用总真实时长作分母。

差分价值 $h_\pi$ 描述相对长期平均水平的瞬态优势。对固定策略，把增长项 $Tg_\pi$ 从有限时域价值中分离，可推导 Poisson 方程。在有限不可约且非周期链中，适当常数规范下有下面的渐近展开。周期链不一定有这个逐点展开，但可以直接求最后的方程。

$$
\begin{aligned}V_T(s)&=\mathbb E_\pi[R_{t+1}+V_{T-1}(S_{t+1})\mid S_t=s],\\V_T(s)&=Tg_\pi+h_\pi(s)-\sum_x d_\pi(x)h_\pi(x)+o(1),\\g_\pi+h_\pi(s)&=\mathbb E_\pi[R_{t+1}+h_\pi(S_{t+1})\mid S_t=s].\end{aligned}
$$

$V_T$ 是未来 $T$ 步未折扣奖励的期望，$d_\pi$ 是有限不可约、非周期链的唯一平稳分布，$x$ 遍历状态。偏差项要减去平稳均值；只有选择 $\sum_xd_\pi(x)h_\pi(x)=0$ 的规范，才能简写成 $Tg_\pi+h_\pi(s)+o(1)$。最后一式只包含 $h$ 的差值，也可以另选参考状态为零，但不能同时省略相应平稳均值。平均奖励算法章进一步推导 Differential TD/Q 与 RVI 如何估计这些量。

$$
\lim_{\gamma\uparrow1}(1-\gamma)v_{\pi,\gamma}(s)=g_\pi(s)
$$

对有限 MDP 中的固定平稳策略，此极限联系了折扣价值与平均奖励。它不是说任意接近一的折扣都会给出相同策略排名，也不能据此任意交换极限、策略优化和函数逼近。

一个策略可能支付很大的启动代价后获得更高稳态收益。平均奖励会忽略有限启动代价，但有限预算任务可能无法承担它。因此目标函数与瞬态指标都需要明示。最终奖励率不能概括整段学习过程。

<a id="lesson-example"></a>

## 6. 同一环境中的策略偏好反转

环境有两条路线。初始时选择 A 或 B，随后只能沿所选路线的环行走。A 每一步得到 1。B 的周期奖励为 0、0、4，并从第一条零奖励开始。奖励与转移均不随时间变化。两条路线没有额外未计入的入场步。此例比较对应的两条确定性策略。

$$
v_A(\gamma)=\frac1{1-\gamma},\qquad v_B(\gamma)=\frac{4\gamma^2}{1-\gamma^3},\qquad g_A=1,\quad g_B=\frac43
$$

B 在第 3、6、9 等步发放奖励，所以折扣和为周期首项乘周期比例的几何级数。平均奖励则直接用周期收益除以周期长度。

$$
\begin{aligned}v_B>v_A&\iff4\gamma^2>1+\gamma+\gamma^2\\&\iff3\gamma^2-\gamma-1>0\\&\iff\gamma>\frac{1+\sqrt{13}}6\approx0.76759.\end{aligned}
$$

两个分母都为正，所以可以相乘比较。平均奖励总是偏好 B，低折扣却可能偏好 A；差别来自评价标准，不是学习算法出错。

| 评价规则 | A | B | 更优路线 |
| --- | --- | --- | --- |
| 折扣 $\gamma=0.5$ | 2 | 1.142857 | A |
| 折扣 $\gamma=0.9$ | 10 | 11.955720 | B |
| 每步长期平均 | 1 | 1.333333 | B |
| 从规定相位开始的前 5 步总奖励 | 5 | 4 | A |

B 的奖励率为 $4/3$。令第一状态的差分价值为零，Poisson 方程给出 $h=(0,4/3,8/3)$。第一式为 $4/3+0=0+4/3$，最后一式为 $4/3+8/3=4+0$。较大差分值表示更接近下一笔收益。周期性不妨碍方程有解，但中心化奖励的有限部分和会振荡。

有限回报后向计算、周期折扣闭式解与平均奖励 Poisson 方程；终止和截断尾值由调用方明确传入。

```python
def discounted_return(rewards, gamma, bootstrap=0.0):
    """Sum gamma**t * rewards[t], plus gamma**T * bootstrap.

    bootstrap=0 describes a true terminal boundary. For a truncated continuing
    trajectory, the caller can supply its continuation value instead.
    """
    discount(gamma)
    rewards = finite(rewards, 'rewards')
    if not math.isfinite(bootstrap):
        raise ValueError('bootstrap must be finite')
    value = float(bootstrap)
    for reward in reversed(rewards):
        value = reward + gamma * value
    return value


def periodic_value(cycle, gamma, phase=0):
    """Exact value of an infinite deterministic reward cycle at a given phase."""
    discount(gamma, continuing=True)
    cycle = finite(cycle, 'cycle')
    if not cycle:
        raise ValueError('cycle must be non-empty')
    if not isinstance(phase, int):
        raise ValueError('phase must be an integer')
    rewards = cycle[phase % len(cycle):] + cycle[:phase % len(cycle)]
    return discounted_return(rewards, gamma) / (1 - gamma ** len(cycle))


def periodic_average_and_bias(cycle):
    """Solve g+h(i)=r(i)+h(i+1) with h(0)=0, including periodic chains."""
    cycle = finite(cycle, 'cycle')
    if not cycle:
        raise ValueError('cycle must be non-empty')
    rate = sum(cycle) / len(cycle)
    bias = [0.0]
    for reward in cycle[:-1]:
        bias.append(bias[-1] + rate - reward)
    return rate, bias


def prefix_rewards(cycle, steps):
    cycle = finite(cycle, 'cycle')
    if not cycle or not isinstance(steps, int) or steps < 0:
        raise ValueError('non-empty cycle and nonnegative integer steps required')
    return [cycle[t % len(cycle)] for t in range(steps)]
```

<a id="lesson-shaping"></a>

## 7. 奖励塑形的边界条件

辅助奖励可以让反馈更密集，但一般会改变任务。势函数塑形给出一种有条件的保序方式。选择固定势函数 $\Phi$，在每次转移上加 $F(s,s')=\gamma\Phi(s')-\Phi(s)$。其中 $\gamma$ 必须与回报的折扣一致。

$$
\begin{aligned}R'_{t+1}&=R_{t+1}+\gamma\Phi(S_{t+1})-\Phi(S_t),\\G'_{0:T}&=\sum_{t=0}^{T-1}\gamma^tR'_{t+1}\\&=G_{0:T}+\sum_{t=0}^{T-1}[\gamma^{t+1}\Phi(S_{t+1})-\gamma^t\Phi(S_t)]\\&=G_{0:T}-\Phi(S_0)+\gamma^T\Phi(S_T).\end{aligned}
$$

相邻势函数项逐项抵消，只留下端点。是否保留策略排序，取决于剩余边界项，而不是奖励是否看起来有帮助。

无限持续折扣任务中，若 $\Phi$ 有界且 $\gamma<1$，尾项趋于零。固定起点后，每条策略的价值都减去相同的 $\Phi(s)$。有限 episode 中，一个充分条件是所有真实终止状态的势为零。这样无论终止时间如何变化，都只有起点偏移。

$$
v'_\pi(s)=v_\pi(s)-\Phi(s),\qquad q'_\pi(s,a)=q_\pi(s,a)-\Phi(s)
$$

等式使用上述边界条件及相同策略、转移和折扣。它保留原问题的策略比较，不保证任意函数逼近算法具有相同的学习轨迹或收敛速度。

原奖励为 $(0,2)$，沿途势为 $(3,1,0)$，折扣为 0.9。塑形奖励为 $(-2.1,1)$，回报为 $-2.1+0.9=-1.2$。原回报为 1.8，两者恰好相差起点势 3。奖励可以变负，但排序仍可保持。

反例：两种动作均一步终止，A 的原奖励为 1，B 为 0。起点势为 0，A 终点势为 0，B 终点势为 2，$\gamma=0.9$。塑形后 A 得 1，B 得 1.8，排序颠倒。即使终点势都是同一个非零常数，不同终止长度仍可能通过 $\gamma^T$ 产生不同偏移。

采用吸收状态的无限延伸时，必须计入吸收后的塑形奖励；不能使用无限和的结论，却在代码中删掉这些项。若只是 rollout 截断，应保留相应的塑形尾值。零终点、非零终点和截断尾值在配套测试中分别处理。

$$
\frac1T\sum_{t=0}^{T-1}(R'_{t+1}-R_{t+1})=\frac{\Phi(S_T)-\Phi(S_0)}T\longrightarrow0
$$

平均奖励中取 $F=\Phi(s')-\Phi(s)$。若势有界，端点差除以时间后消失，长期奖励率不变。有限生命收益仍有端点差；这与折扣塑形是不同的评价推导。

常数平移也要检查边界。无限持续折扣任务中，$r'=ar+b$ 且 $a>0$ 给出 $v'=av+b/(1-\gamma)$，因此保留排序。长度可变的未折扣 episode 却增加 $bT$，可能改变终止偏好。奖励裁剪、时间变化的探索 bonus、任意“靠近目标就加分”，也不自动满足势函数条件。

保留全部端点势，比较逐步回报与望远镜恒等式。

```python
def shape_rewards(rewards, potentials, gamma):
    """F_t=gamma*Phi(s[t+1])-Phi(s[t]); retain the final potential explicitly."""
    discount(gamma)
    rewards = finite(rewards, 'rewards')
    potentials = finite(potentials, 'potentials')
    if len(potentials) != len(rewards) + 1:
        raise ValueError('one potential per state, including both endpoints, required')
    return [reward + gamma * potentials[t + 1] - potentials[t]
            for t, reward in enumerate(rewards)]


def shaping_comparison(rewards, potentials, gamma):
    shaped = shape_rewards(rewards, potentials, gamma)
    original = discounted_return(rewards, gamma)
    boundary = -potentials[0] + gamma ** len(rewards) * potentials[-1]
    return dict(original=original, shaped=discounted_return(shaped, gamma),
                boundary=boundary, expected=original + boundary)
```

<a id="experiment-potential_shaping"></a>

### 实验：实验 · 保持最优策略，不等于保持学习轨迹

势函数塑形保持正确目标时，有限预算曲线是否必须更好？

**环境与可用信息。** 六格确定性链：从 0 开始，左右动作，左边界停留。到位置 5 得 1 并终止，其他转移得 −0.01。观测就是位置。真实终止后从 0 开始；折扣为 0.95，任务没有中途变化。

**设置。** 各运行 1200 个真实环境步；Q 全零，学习率 0.2，ε-greedy 的 ε=0.1，并列最优训练动作均匀选择。两者只有训练奖励不同。势函数为非终止状态的 −(5−s)/5，终点势为 0。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** potential_shaping.py 仅以势差改变训练奖励；评价仍使用未塑形环境奖励。两者都从 Q=0 起步，这不等同于把两者初始 Q 按势函数做对应平移。

**测量。** 主图是冻结贪心策略从状态 0 出发的精确折扣回报，包含可能无限循环的几何尾项，不是训练奖励或完整生命期所得。

```bash
python3 implementations/continual/potential_shaping.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/potential_shaping/curves.svg)

横轴：environment_steps。纵轴：原始奖励贪心策略折扣回报。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 15 步时塑形的均值为 0.38644，对照为 −0.2；到 1200 步，塑形为 0.58193，对照为 0.77741。塑形的末端标准差约 0.43711。早期较好、晚期反差说明目标保持不保证任意初始化与有限探索下的学习优势。

**结论边界。** 固定势、正确终点势和静止链满足本实验所用边界约定。有限预算的一条失败曲线不是对最优策略保持结论的反例；本图也没有测真实学习过程的总收益。

**继续实验。** 把初始 Q 按势函数对应平移，再检查相同行为随机数下两种方法的动作和更新是否对应。改变终点势后先推导边界项，再运行。

[源码](https://yingwen.io/crl-code/implementations/continual/potential_shaping.py) · [逐种子记录](https://yingwen.io/crl-code/results/potential_shaping/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/potential_shaping/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/potential_shaping/curves.json)

<a id="lesson-subtasks"></a>

## 8. 总任务、目标条件子任务和预测知识

“目标”有两种常见用法。总任务目标规定什么结果对整个智能体更好。目标条件策略的 goal 则是输入，例如“到充电站”或“使某个特征变大”。后者可以帮助求解总任务，但不自动等于总任务。

$$
\pi(a\mid s,z),\qquad r_z(s,a,s'),\qquad J_{\rm ext}(\Lambda)
$$

$z$ 是子任务描述，$r_z$ 是训练它所用的奖励，$J_{\rm ext}$ 是外部评价。运货过程中选择“去充电”只是中间决策，不必改变整个系统的订单收益目标。

将“到达充电站”训练得可靠，不保证订单产出提高。高层选择器还需判断何时执行，以及耗时和后果。子任务选择、option 终止和模型学习因此各有问题。外部评价应计入训练与执行子任务的交互，而不只统计技能成功率。

预测问题又不同。GVF 用 cumulant $c$、continuation $\gamma$ 和策略 $\pi$ 定义待预测的经验量，例如“持续前进会遇到多少碰撞”。预测学习要求把该量估计准确；控制器是否应避免碰撞，要由任务奖励和约束决定。$c$ 不必是外部奖励。

预测知识可以供控制、规划和状态构造使用，但不必全部成为状态坐标。状态也不必全部由可解释预测构成。总任务、内部子任务和知识预测分别定义后，才能分析它们是帮助、冲突还是增加了无用计算。Alberta Plan 的基本智能体也区分主要策略、其他策略、价值函数和转移模型；它们通过构造的状态交换信息。

<a id="lesson-lifetime"></a>

## 9. 完整学习器、资源限制与“持续”的定义

固定策略评价分析一个明确的行为规则。持续学习还需要评价完整智能体，包括初始化、探索、参数更新、记忆管理和规划。它在变好之前已经消耗时间，也已经产生收益或损失。

$$
J_T(\Lambda)=\mathbb E_\Lambda\!\left[\sum_{t=0}^{T-1}R_{t+1}\right],\qquad\overline J_T(\Lambda)=J_T(\Lambda)/T
$$

所有学习期动作都在累计和中。对固定 $T$，累计收益与平均收益排序相同。期望涵盖环境、探索和学习器的随机性。

$$
J_{\rm frozen}(\Lambda,T)=\mathbb E[J(\pi_{w_T};d_{\rm eval})]
$$

这里在训练至 $T$ 后冻结参数，再从指定分布 $d_{\rm eval}$ 评价策略。外层期望来自训练随机性。这是有用的诊断，但通常不同于 $J_T$；额外评价环境的交互量也应单列。

在平稳确定性三臂 bandit 中，三个动作分别给 0、0.6 和 1。过程 A 十步里一直选第二臂，在线收益为 6，冻结末次动作后的每步收益为 0.6。过程 B 前八步选第一臂，最后两步选第三臂，在线收益只有 2，但冻结后收益为 1。这里是两个预设行为过程的评价反例，不是学习算法的性能比较。

相同环境中的在线收益和冻结末次动作收益，可以产生相反排序。

```python
def evaluate_schedule(arm_rewards, action_schedule):
    """Compare online reward with the value of freezing the final bandit action.

    This is an evaluator of prescribed action schedules, not a bandit learner.
    The stationary deterministic environment returns arm_rewards[action].
    """
    arm_rewards = finite(arm_rewards, 'arm_rewards')
    actions = tuple(action_schedule)
    if not arm_rewards or not actions:
        raise ValueError('non-empty arms and action schedule required')
    if any(not isinstance(a, int) or not 0 <= a < len(arm_rewards) for a in actions):
        raise ValueError('each action must index an arm')
    rewards = [arm_rewards[a] for a in actions]
    cumulative = sum(rewards)
    return dict(cumulative=cumulative, online_mean=cumulative / len(rewards),
                frozen_final_mean=arm_rewards[actions[-1]])
```

资源限制决定哪些智能体可以公平比较。需要规定持久内存、每个交互步的计算量、数据保存范围、重置权限和预训练预算。若规划会延迟真实行动，墙钟时间也进入任务。若统一用环境步计分，至少要另报算力和延迟。

$$
\begin{aligned}
   \mathfrak L(B)&=\{\mathcal L: M(\mathcal L)\le B_M,\ C_t(\mathcal L)\le B_C\text{ for every }t,\ \mathcal L\text{ obeys the information protocol}\},\\
   J_P^*(B)&=\sup_{\mathcal L\in\mathfrak L(B)}J_P(\mathcal L),\qquad
   \mathfrak L(B_1)\subseteq\mathfrak L(B_2)\ \Longrightarrow\ J_P^*(B_1)\le J_P^*(B_2).
   \end{aligned}
$$

$B=(B_M,B_C)$ 是这里选定的内存与每步计算预算，$M$ 为所有可实现运行中的最大持久内存，$C_t$ 为第 $t$ 步计算量的最坏情形上界；改用期望预算需另行声明。预算也可扩展为延迟、能耗、预训练和开发成本，但需约定计量。放宽同一问题的预算只扩大候选集合，因此上确界不会下降；这不意味着每个更大的网络都学得更好。

Big World 假设进一步选择环境相对智能体过于复杂的问题。它激励用有限状态与知识跟踪当前相关情况，研究何时保留、替换、探索和规划。平稳性只约束环境核，不能保证有限智能体已掌握所有有用情况；反过来，有限容量本身也不能证明永久适应必然优于固定行为。需要在所规定的环境、准则和预算下展示差距。

Alberta Plan 强调长期交互、有限计算和时间一致性。学习与规划是运行过程的一部分，而非奖励不计分的特殊准备阶段。有限生命评价体现这种关注，但不是该计划唯一指定的评分公式。不同应用仍需说明自己的时间聚合方式。

| 术语 | 含义 | 不能据此推出的结论 |
| --- | --- | --- |
| Continuing task | 没有自然最终终止。 | 不强制平均奖励，也不要求非平稳环境。 |
| Stationary environment | 给定充分状态和动作后，动力学与奖励核不随时刻改变。 | 不表示观测不变化、数据独立或智能体已学会。 |
| 持续更新的学习器 | 参数、知识或其他学习状态在运行中适应经验。 | 非零更新不证明所有最优智能体都必须永久学习。 |
| Abel 等的 CRL 定义 | 相对于给定 agent basis，最优智能体持续隐含搜索，而不最终停在其中一个基础智能体。 | 不单凭任务切换次数或非平稳标签来分类。 |

Abel 等的形式化还要给定候选历史智能体集合 $\mathfrak A$ 与非空基底 $\mathfrak B\subseteq\mathfrak A$。基底“生成”候选集合，是指每个候选者都能通过依历史选择基底元素，重现它在环境中可实现历史上的动作分布。基底元素本身也可以是历史策略；它不等于一组数值参数。随后用行为上的“到达”而非参数变化量判定是否停止隐含搜索。

$$
\begin{aligned}
   \operatorname{Reach}_P(\lambda,\mathfrak B)
   &\iff\exists h\in\mathcal H^{P,\lambda}\ \exists\beta\in\mathfrak B\ \forall h'\in\mathcal H_h^{P,\lambda}:\lambda(\cdot\mid hh')=\beta(\cdot\mid hh'),\\
   \mathfrak A^*&=\operatorname*{arg\,max}_{\lambda\in\mathfrak A}J_P(\lambda),\\
   \text{CRL relative to }\mathfrak B
   &\iff\forall\lambda^*\in\mathfrak A^*: \neg\operatorname{Reach}_P(\lambda^*,\mathfrak B).
   \end{aligned}
$$

$\lambda,\beta$ 为行为规则；$\mathcal H^{P,\lambda}$ 是有非零概率的可实现历史，$\mathcal H_h^{P,\lambda}$ 是可接在 $h$ 后的可实现后缀，$hh'$ 为连接。此处按原文的可数接口表述，并假定基底生成候选集合且最优者存在。到达要求存在一个历史，从那里开始在所有可实现延续中永远与某个基底元素相同；“永不到达”否定这一命题。它不等于概率一收敛、数值近似或梯度最终为零。

这一定义是相对于四项给定量的性质：环境、性能、候选集合与基底。换基底或改变预算可能改变判定；只观察有限日志无法证明无限时间永不到达。它与本章的有限生命期评分互补：前者定义持续性的结构性质，后者比较指定期限内的实际表现，不能相互代替。

未知但平稳的有限 bandit 仍需要学习，而每个已确定的环境都存在最优固定拉臂策略。不过，存在一个使用真实臂均值的 oracle 策略，并不说明未知模型的有限预算学习器能立即执行它；也不能只凭这一事实完成 Abel 定义中的判定。要一起分析信息、准则、候选集合和基底，而不只检查转移函数有没有时间下标。

<a id="experiment-bandit_constant_step"></a>

### 实验：实验 · 把探索期间的代价计入评价

评价整个学习器时，为何不能只看最后认为哪个动作最好？

**环境与可用信息。** 无状态三臂 Bernoulli 赌博机，三动作奖励 1 的概率固定为 0.2、0.5、0.8，其他情况奖励 0。无回合终止、无变化信号；每次拉臂计一个真实交互。

**设置。** 1200 次拉臂，初始动作估计和计数为 0。两者都用 ε=0.1 的 ε-greedy；固定步长为 0.1，样本均值对被选动作使用其访问次数的倒数。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** 控制器每次先选动作、获得真实奖励，再更新该动作估计。日志中的累计奖励保留了所有早期探索成本，没有用最后的最优动作均值回填历史。

**测量。** 纵轴为从起点至当前时刻的实际平均奖励。它和固定动作 2 的期望奖励 0.8 是不同评价对象：前者包含学习，后者是固定策略的参照。

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

**结果分析。** 1200 步时固定步长与样本均值的全程平均奖励分别为 0.73917、0.76283，均未达到始终选择已知最优臂的参照 0.8。差距包含合法探索和估计错误，不能直接全称为算法实现故障。

**结论边界。** 没有冻结策略的独立评测曲线，因此不能从现有图计算“最终能力减去学习代价”。已知最优臂是分析参照，未作为控制器的初始知识。

**继续实验。** 新增隔离的冻结分支评估最终动作分布，并保留原来的全程收益指标。若冻结评测数据被用于选择在线动作，评价协议发生了什么变化？

[源码](https://yingwen.io/crl-code/implementations/classic/bandit_constant_step.py) · [逐种子记录](https://yingwen.io/crl-code/results/bandit_constant_step/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/bandit_constant_step/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/bandit_constant_step/curves.json)



<a id="experiment-learner_control_online"></a>

### 实验：实验 · 评价完整学习过程，而不是重置后的最终策略

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

<a id="rlss-temporal-uniformity"></a>

## 时间上的统一性：持续更新与计算代价

Alberta Plan 所说的 temporal uniformity，指学习、规划、表征构造不依赖一个特殊且不计代价的训练阶段。它不是“环境不变化”，也不是“每一步必须执行完全相同数量的梯度更新”。条件触发的更新、休眠和自适应步长都可以由同一个持续运行的规则决定。

$$
Z_{t+1}=\mathcal U(Z_t,A_t,O_{t+1},R_{t+1}),\qquad A_{t+1}\sim\Pi(\cdot\mid Z_{t+1}).
$$

Z 包括用于行动的状态、模型参数、资格迹、统计量、元参数和规划队列。规则 U 持续适用；它的输出与计算路径可以依赖当前信息。这是有限学习器的记账表达，不是某一种固定架构。

“从不关掉学习”也不充分。若步长只能减小、不能在变化后恢复，更新虽然非零，也可能几乎无法改变预测。反过来，稳定时期暂时不更新某个模型，不必违背持续学习；需要说明何种可观测事件会重新启动它，以及监测本身占多少资源。

| 计量 | 适用情形 | 不能漏掉的成本 |
| --- | --- | --- |
| 真实交互步 | 仿真器等待学习器，环境步定义固定 | 每步多少更新、模型查询与持久存储 |
| 真实经过时间 | 世界在计算时仍演化 | 响应延迟、等待动作、逾期或恢复 |
| 有限生命期收益 | 学习成本属于智能体表现 | 探索、技能学习与表示替换期间的损失 |

例子：两个方法使用同样一千条真实转移，一个每步做一次更新，另一个做一百次模型 backup。按交互步画图可以比较经验利用；不能由此声称第二个方法计算更高效。若环境每十毫秒必须接收动作，额外规划还可能推迟动作，进而改变实际获得的数据。

$$
g_{\rm time}(\Lambda)=\lim_{n\to\infty}\frac{\mathbb E_\Lambda[\sum_{k=0}^{n-1}R_{k+1}]}{\mathbb E_\Lambda[\sum_{k=0}^{n-1}\tau_k]}.
$$

这是一个需要极限及相应长期条件的单位时长评价。τ 是真实经过时间。它不是 E[R/τ]，也不是每个决策步等权的奖励平均。任务若采用有限时限，应直接报告该时限内实际得到的总收益。

**算法：从研究设定到可执行的资源协议**

1. 先规定世界时钟：学习期间是否继续演化？
1. 再规定动作截止时间、等待动作及逾期后果。
1. 记录实际交互步、实际时长、更新数、模型查询数与峰值内存。
1. 区分上线前数据/调参成本与计入生命期的学习成本。
1. 比较算法时保持同一评价协议；另用同数据诊断定位更新机制。

因此，大世界观点并不要求把原来每项保证全部抛弃。应先标出保证依赖的条件，再逐项改变信息、表示、数据权限或计算预算。这样，失败时才能区分目标变了、信息不足、估计不稳，还是来不及完成决策。

<a id="lesson-maze-evaluation"></a>

## 9.1 回到岔路迷宫：把学习过程和评价逐步算出来

沿用图中的 J、L、R 三格。智能体在 J 选左或右，抵达侧格时领取 1 或 0，再用一个强制动作回到 J，返回奖励为 0。前 8 圈是 A，接着 8 圈是 B，再接着 8 圈是 A；之后重复这一周期。这里比较前 24 圈，也就是 48 个原始环境步。返回 J 是普通转移，过程中没有外部重置；第 48 步只是本次统计截止，世界仍可继续。

先运行一个最小学习器。它保存一个方向位，初始为左；领取 1 就保留方向，领取 0 就把下一圈的方向翻转。它只读取实际选中路线的奖励，既没有模式标签，也不用计数推断切换日程。这个“胜留败换”规则利用了两条路线确定性、互为好坏的任务结构；加入奖励噪声或更多路线后，应重新选择学习规则。

**算法：实际计算图中蓝线的完整更新规则；侧格仅有返回动作。**

1. side = "L"
1. for each circle:
    1. move from J to side; observe reward
    1. if reward == 0:
        1. side = "R" if side == "L" else "L"
    1. return to J; receive 0
    1. record both environment steps

冻结左策略也从左开始，却一直不改变方向。图中的 oracle 预先知道哪一段是 A 或 B，每圈都选有奖励的路线；它提供收益上界，额外日程信息使它不属于与普通学习器相同的信息条件。先预测这三条累计曲线在哪些位置分开，再读图。

![三格往返迷宫、48 原始步累计奖励曲线，以及胜留败换每圈奖励的圆点序列。](https://yingwen.io/crl-figures/concept-crl-lifetime-curves.svg)

上方先看两种奖励规则。曲线横轴是原始环境步，纵轴累计所有学习期奖励；侧格返回时曲线保持水平。蓝线由胜留败换实际逐步运行，虚线是冻结策略与知日程 oracle。底部绿点为奖励 1，小橙点为 0。[逐步数据](/crl-figures/concept-crl-setting-data.json) 和[计算模块](/crl-code/figures/crl-setting.mjs) 可独立核算。

第 9 圈进入 B 时，学习器仍选左，收到 0 后才改选右，所以 B 段获得 $0+1+\cdots+1=7$。第 17 圈回到 A 时同样先损失一次，再改回左。全程奖励为 $8+7+7=22$；冻结左策略得到 $8+0+8=16$。按原始步计的平均分别为 $22/48$ 与 $16/48$。二者在第 24 圈都选左，在 A 中冻结测试都得每圈 1；末次策略相同，却留下不同的生命期收益。

下一步问未来学习，而非过去总收益。在第 8 圈复制学习器：两份当前方向都是左，一份关闭方向更新，另一份保留更新。把副本放入固定 B，分别运行 8 圈。它们从相同当前动作规则出发，差别只在新经验能否改变后续行为。

![第 8 圈检查点分成两个副本，左列始终向左无奖，右列首次失败后改向右并连续获奖。](https://yingwen.io/crl-figures/concept-crl-freeze-continue.svg)

从上向下读八次选择。箭头表示方向，绿点表示奖励 1；每列执行 16 个原始诊断步。冻结的是方向更新，返回 J 的转移仍照常执行；副本状态和数据不回写主生命期。

当前策略只需要方向位就能指定动作，完整学习器还要规定如何更新这个位。本例不使用权重、优化器、回放或模型；复杂实现若使用它们，就应将这些持久量连同递归活动和资格迹纳入实现状态 $Z_t$。观测 $O_t$ 是这次收到的位置和奖励，历史 $H_t$ 是全部已见经验的数学描述，实际保存哪些信息则由预算决定。冻结权重时，其他状态仍可能递推，因此需逐项声明冻结范围。

适应、保持和前向迁移把同一条过程中的不同问题分开。适应看切换后的实际奖励；保持把旧情境重新呈现给冻结副本；前向迁移比较有旧经验的副本与 fresh 学习器在同一新情境、同一更新规则和预算下的表现。下面采用“新情境前 8 圈累计收益之差”作为窗口前向迁移指标，其他文献也可能采用零样本成绩或学习速度，需要先对齐定义。

![逐圈奖励、旧任务冻结表现、新任务学习收益差和全生命期累计收益的四幅小图。](https://yingwen.io/crl-figures/concept-crl-evaluation-probes.svg)

B 中快速改道获得 7，并未保住旧 A 的冻结行为；与 fresh 的同预算 B 窗口比较为 7−7=0。第一、四项来自原轨迹，第二、三项使用额外复制的诊断世界。测量对象和访问权限不同，四个数字不能相互替代。

第 8 圈的方向为左，第 16 圈则为右。在固定 A 中停止方向更新，两个检查点每圈的奖励分别为 1 和 0，所以旧 A 的冻结表现下降量为 1。原轨迹第 17 圈返回 A 后重新改道，是再适应，不是一直保留了旧行为。另一方面，第 8 圈副本和初始选左的 fresh 学习器都在 B 的 8 圈里得到 7，窗口前向迁移为 0。这个学习器能及时追踪，却只保存最近的一项选择。

| 测量 | 本例的数值与单位 | 需要的权限 |
| --- | --- | --- |
| 有限生命期收益 | 22；48 原始步内的奖励总和。 | 完整训练日志；所有动作计分。 |
| B 段适应 | 首圈损失 1，后 7 圈均得 1。 | 原轨迹日志；明确切换时刻用于分析。 |
| 旧 A 冻结保持 | 每圈奖励从 1 降至 0。 | 复制检查点并在 A 探测；禁止回写。 |
| 窗口前向迁移 | B 的前 8 圈收益差：7−7=0。 | aged 与 fresh 副本；同新情境、预算和规则。 |
| 末次冻结表现 | 两普通策略在 A 都得每圈 1。 | 额外 A 诊断；与训练收益单列。 |

在可复制模拟器中，可以用多个独立随机种子或独立任务序列重复完整生命，再计算事先指定的收益统计与不确定性。随机性若来自不同任务顺序，就要把顺序作为实验条件。相邻时间步来自同一个有记忆的过程，不是独立重复实验；共同整数 seed 也要核对实际共享了哪些随机事件。

只有一个不可重来的真实生命时，日志仍可给出实际收益和发生过的切换后表现。未再次出现的旧任务保持、未运行的 fresh 学习器和另一策略的收益，则需要额外干预、模型或离策略估计条件。允许复制的诊断结果可以帮助理解机制，但应与这个真实生命的在线成绩分开报告。

本章用有限窗口比较实现，Abel 等的定义则用给定环境、性能、候选集合与基底判定无限延续中的持续性。A/B/A 的名字和几次非零更新本身不完成那个判定。要研究周期重复世界中的无限持续性，可以进一步选定性能准则和基底，并分析何时固定基底行为会错过后续机会。Kumar 等从计算约束和长期平均收益组织问题；有限窗口保留了其长期准则可能忽略的启动与恢复损失。

<a id="lesson-noisy-lifetime"></a>

## 9.2 奖励带噪声时，哪一个学习器更好？

现在只改奖励，不改地图和时间安排。在 A 中，抵达 L 时以 0.9 的概率得 1，抵达 R 时以 0.1 的概率得 1；否则得 0。B 的两个概率互换。各次奖励在给定所选路线和当前规律后独立。仍从 J 出发，按 A、B、A 各走 8 圈，每圈包括去、回两步。智能体不知道规律标签、切换日程和奖励概率。

胜留败换仍只保存一个方向位，初始向左。它在所选侧格收到 0 后翻转方向，收到 1 后保留方向；返回 J 的零奖励不触发第二次更新。噪声使这条规则偶尔误判：好路线也会给 0，坏路线也会给 1。始终向左则不会受这些偶然结果影响，却不能适应 B。哪一种损失更大，需要沿完整过程计算。

![带随机奖励的 J/L/R 迷宫，A/B/A 生命期的两条期望累计收益曲线，以及末态冻结后固定 A 中的两条诊断条形。](https://yingwen.io/crl-figures/lifetime-evaluation-ranking.svg)

每圈包含两个原始环境步。全程期望收益为 18.48 与 15.20；末态复制到 A，关闭方向更新后，再测 8 圈，期望收益却是 6.56 与 7.20。这里画的是全部随机历史的精确期望，不是一次训练曲线。末态的 90%/10% 指不同生命留下左/右方向的概率，每个冻结副本内部不再换边。[Python 逐步计算](/crl-code/tutorials/lifetime_evaluation_walkthrough.py) · [作图数据](/crl-figures/lifetime-evaluation-data.json)。

用 $n$ 表示圈数，$p_n$ 表示第 $n$ 圈开始时学习器向左的概率。$p_1=1$。分析者知道该圈左、右路线的获奖概率 $q_L(n),q_R(n)$。下一圈仍向左只有两种情况：本圈向左且获奖，或本圈向右且未获奖。因而

$$
\begin{aligned}p_{n+1}&=p_n q_L(n)+(1-p_n)[1-q_R(n)],\\ \mathbb E[R_{2n-1}]&=p_n q_L(n)+(1-p_n)q_R(n),\qquad R_{2n}=0.\end{aligned}
$$

$R_{2n-1}$ 是抵达侧格的奖励，$R_{2n}$ 是返回奖励。$p_n$ 是分析随机生命的概率，不是智能体保存的置信度；智能体实际只保存 L 或 R。

在 A 中，上式给出 $p_{n+1}=0.9$；在 B 中给出 $p_{n+1}=0.1$。第一圈必选左，期望奖励为 0.9。每次刚切换时，大多数生命仍朝向上一段的好路线，所以两次切换的首圈期望奖励均为 $0.9\times0.1+0.1\times0.9=0.18$。其余 21 圈均为 $0.9^2+0.1^2=0.82$。

$$
J_{48}(\Lambda_{\rm WSLS})=0.9+2(0.18)+21(0.82)=18.48,\qquad J_{48}(\Lambda_{\rm left})=16(0.9)+8(0.1)=15.2.
$$

所有探索、误判和切换后的损失均已计入。按原始步归一化，两者分别为 0.385 与约 0.3167；这里是有限 48 步的收益比较，不是无限平均奖励的结论。

到第 48 步，胜留败换的方向在 90% 的生命中为左，10% 中为右。复制这个检查点，放到固定 A 中做 8 圈诊断，并关闭方向位更新。于是有些副本一直左，有些一直右。始终向左的对照在所有副本里都向左。诊断奖励的期望分别为

$$
J_{\rm probe}(\Lambda_{\rm WSLS})=8[0.9(0.9)+0.1(0.1)]=6.56,\qquad J_{\rm probe}(\Lambda_{\rm left})=8(0.9)=7.2.
$$

诊断共使用 16 个额外环境步。副本状态和奖励不回写原生命期。方向在副本内固定，因此同一副本各圈共享同一个方向，不能把 90%/10% 误当作逐圈独立抽取的随机策略。

排名反转来自评价的问题不同。生命期收益问：经过这三个阶段时，整个学习过程收获多少？末次冻结诊断问：留下来的当前行为在 A 中表现怎样？后者指出胜留败换容易被噪声误导；前者还计入它在 B 中适应的收益。两项测量都有用途，却不能互相替代。这个例子没有证明胜留败换最优；保留较长证据、检测变化或预测日程都可能产生更好的学习器，也会改变内存和计算需求。

把 A/B 标签和周期位置加入环境的充分状态后，仍可写出 Markov 模型。智能体未必看到这些变量。这个有限例子用来分离评价对象与信息条件，不要求放弃 MDP 的数学工具，也不单凭几次切换判定永久学习是否必要。下一章的流式例子将把这个方向位扩展为可逐项检查的递归状态、权重、资格迹和尺度统计。

[流式学习器保存了什么？](/zh/continual-rl/algorithms/streaming/#lesson-learner-state) 中会看到：恢复相同权重，不一定恢复同一个学习器。评价之前，应先明确哪些状态在运行、哪些状态被冻结。

下载上图链接的脚本后运行。仅 Python 标准库；先执行断言，再输出各圈概率、两个原始步的收益记录和独立诊断。

```sh
python3 lifetime_evaluation_walkthrough.py
```

练习：把好路线的概率改为 1，应恢复上一节的 22 与 16；改为 0.5，两种学习器的 24 圈期望都应为 12。再改变每段圈数，观察适应收益何时不足以弥补误判。脚本用有理数合并“方向、累计奖励”相同的历史，作图模块独立使用上面的概率递推；两者逐圈交叉检查。

<a id="lesson-code"></a>

## 10. 运行与检查

下载 objectives_lab.py 后运行；Python 3.10+，仅标准库。

```sh
python3 objectives_lab.py all
python3 objectives_lab.py preferences
python3 objectives_lab.py stopping
python3 objectives_lab.py shaping
python3 objectives_lab.py lifetime
python3 objectives_lab.py test
```

preferences 输出折扣价值、奖励率和前五步收益。stopping 独立计算停止位置期望，并展示误把截断当终止的差异。shaping 输出原回报、塑形回报和解析边界项。lifetime 比较两个评价方式的排名。每个实验都有明确输入和确定性结果。

**算法：任务与评价的计算流程，不是某个控制算法的训练步骤**

1. 给定环境、允许信息、资源上限和候选行为过程。
1. 选择回合、折扣、平均奖励或有限生命目标。
1. 分别写出真实终止、重置和采样截断规则。
1. 在同一初始条件下生成奖励序列，保留全部学习期奖励。
1. 若采用奖励塑形：
  1. 固定势函数，匹配折扣，检查起点与终点项。
  1. 分别记录原始奖励与训练奖励。
1. 报告在线收益，以及需要的冻结策略诊断。
1. 改变目标时，重新计算策略排序。

解析实验不依赖采样估计，因此数值一致性只检验定义和实现。进入随机环境后，还要报告独立运行数、不确定性和失败轨迹。单次真实生命中，则需说明哪些统计只从日志离线计算，哪些干预不允许执行。

<a id="lesson-branches"></a>

## 11. 后续研究问题

| 问题 | 要学习的机制 | 与本章的连接 |
| --- | --- | --- |
| 未知状态的长期价值如何估计？ | TD、资格迹、off-policy 预测。 | 固定策略、奖励、折扣与边界。 |
| 没有自然终点时如何提高收益？ | Differential TD/Q、RVI、平均奖励控制。 | 明确奖励率、链结构和时间单位。 |
| 当前观测不足以决策？ | 信念、RNN、预测状态、在线递归梯度。 | 保持目标，比较历史摘要丢失的信息。 |
| 哪些知识值得长期保存？ | GVF、问题发现、预测表示。 | 明确预测对象及其对外部目标的用途。 |
| 怎样分解长程行为？ | 目标条件策略、option、子任务发现。 | 子任务奖励与总评价分开，计入时长。 |
| 学习很久后如何继续适应？ | 步长适应、可塑性、元学习、资源管理。 | 评价完整学习过程，不只测最后参数。 |
| 应当计划哪些未来？ | 模型学习、规划和搜索控制。 | 按决策影响分析模型误差，计入规划成本。 |

清楚的研究可以只改变一个选择。保持环境和策略类不变，比较不同折扣与平均奖励；或者保持外部目标不变，比较有明确边界条件的奖励塑形。用可解析环境确定预期策略后，再进入函数逼近实验，才能区分任务变化、估计误差和学习机制的影响。

<a id="lesson-check"></a>

## 12. 习题与诊断

- 把 B 改成 $(0,0,5)$，重新求折扣阈值。答案：$4\gamma^2-\gamma-1>0$，阈值为 $(1+\sqrt{17})/8$；前五步 A 与 B 都得 5。
- $B=(0,0,4)$，但从发放 4 的相位开始，平均奖励是否变？答案：不变；折扣价值变为 $4/(1-\gamma^3)$。
- 起点势为零、终点势都为 10，$\gamma=0.9$，A 一步奖励 1，B 两步奖励 $(0,1.5)$，是否保序？答案：原回报 1 与 1.35，塑形后 10 与 9.45，排序反转。
- 碰撞 GVF 预测准确，是否定义了避免碰撞的任务？答案：没有；预测内容与控制评价中的碰撞代价不同。
- 平稳 MDP 运行百万步且一直更新网络，是否满足 Abel 等的 CRL 定义？答案：不能据此判断；还需指定 agent basis，并分析最优智能体是否能停止搜索。

| 现象 | 优先检查 |
| --- | --- |
| 训练损失降低但收益不升 | 是否把预测目标、内部奖励或辅助损失当成外部评价。 |
| 增大折扣后偏好不同动作 | 目标本来可能变化；先解小模型检查策略排序。 |
| timeout 附近价值突降 | 是否把采样截断写成真实终止。 |
| 塑形后循环刷奖励 | 是否满足势差形式，终点和折扣是否匹配。 |
| 最终策略好，长期运行却损失大 | 是否遗漏学习初期、探索、恢复和重置成本。 |
| 收益接近但计算量差异很大 | 交互与计算预算是否一致，延迟是否影响真实动作。 |

## 本章的实验设计

先明确比较冻结策略，还是比较持续更新的学习器。两者具有不同的估计对象。

设定：同一可控任务分别评价完整在线生命期和最终冻结策略。保留原始奖励、动作时间与所有重置事件。

- 两种评价的收益口径和数据来源可以分别重算。
- 冻结副本不会改变主训练的参数、统计、记忆与 RNG。
- 奖励变换若改变优化目标，明确标注而非沿用原目标名称。

对照：相同初始化的在线更新与冻结参数；相同协议下的未变换奖励基线；部署允许与禁止重置的分别报告

记录：完整生命期累计奖励或每原始时间奖励率；冻结策略条件表现；重置、评价及在线更新成本

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-question)

## 学习与研究衔接

折扣与平均奖励是目标选择。持续学习不能仅靠把回合接长来定义。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-objectives) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=objectives) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=objectives)

## 从本章进入实践

[策略梯度与控制](https://yingwen.io/zh/continual-rl/code/#practice-policy-control)：优化器确实降低了损失，为什么行动仍可能变差？

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

给定模型可研究怎样规划；模型也在学习时，规划会选择性地查询误差，并改变以后的数据。Dreamer、STOMP 和 DRAGO 分别研究想象控制、随机时长行为模型和旧知识保留。新的比较应固定规划查询与总预算，检验哪些后果误差真正改变选择，哪些维护值得继续。

- [Distributional Model Equivalence for Risk-Sensitive Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-distributional-model-equivalence)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [An Idiosyncrasy of Time-discretization in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-openmind-time-discretization)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [Rethinking the Foundations for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-rethinking-crl-foundations)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文规定对象与条件，架构路线提出组织方式，算法实验检验局部机制。撤掉阶段间冻结后，一个模块会改变另一个模块的学习问题；有限预算应优先维护哪条知识，成为新的决策。先检验两模块反馈和资源分配，再扩大整机，而不是由组件分别有效推断长期组合收益。

- [Rethinking the Foundations for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-rethinking-crl-foundations)
- [Plasticity as the Mirror of Empowerment](https://yingwen.io/zh/continual-rl/research/#recent-plasticity-mirror-empowerment)

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

### Rethinking the Foundations for Continual Reinforcement Learning

Esraa Elelimy, David Szepesvari, Martha White, Michael Bowling

RLC 2025 / RLJ · 2025 · 定义与架构观点

#### 研究问题

如果智能体终生交互且世界不断变化，传统形式化和评价对象遗漏了什么？

#### 关键机制

论文重新审视状态、时间与累计奖励评价中的隐含假设，并讨论以交互历史和偏离遗憾等对象描述持续学习。研究重点从“在固定任务上最终收敛到什么”转向“在持续过程里，什么样的行为比较才有意义”。

#### 证据

这是形式化与研究基础的论证，提出可继续研究的定义和问题；不是一个已经完成全部工程验证的通用智能体。

#### 条件与限制

对常见形式化局限的讨论不意味着 MDP、折扣回报或平均奖励在各自条件下无效。评价框架还需要与具体可计算算法和实验协议连接。

#### 阅读与实验

选择一个有不可逆代价的环境，分别写出最终任务分数、终生在线收益和比较策略集合。检验它们是否会给同一行为排出不同顺序。

#### 原文与相关入口

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_243.pdf)：形式化动机、定义与论证。
- [RLJ 论文入口](https://rlj.cs.umass.edu/2025/papers/Paper243.html)：作者与正式收录信息。

### Distributional Model Equivalence for Risk-Sensitive Reinforcement Learning

Tyler Kastner, Murat A. Erdogdu, Amir-massoud Farahmand

NeurIPS 2023 · 2023 · 支持方法与理论

#### 研究问题

模型正确预测期望回报，能否同时支持避开低概率灾难的决策？

#### 关键机制

论文证明 proper value equivalence 对风险敏感规划不足，再以回报分布与统计摘要定义更强的模型等价。完整分布覆盖更多风险度量，有限摘要则限制可支持的风险目标；相应 Bellman 闭合性质决定摘要能否递推。

#### 证据

正式原文包含理论、表格反例与大规模实验，并直接给出 distribution-equivalence 作者仓库。它检验的是特定风险敏感目标下的模型学习与规划接口。

#### 条件与限制

正确均值和方差不自动保证尾部概率或 CVaR；有限 quantile 表示与投影也有近似误差。静态模型等价不保证新环境中的风险校准，更不等于安全约束保证。

#### 阅读与实验

构造均值相同、尾部不同的两动作，先验证期望控制无法区分，再用指定风险度量评价。训练分布、投影和风险目标必须匹配，不能在评估时随意换风险函数。

#### 原文与相关入口

- [NeurIPS 2023 原文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b0cd0e8027309ea050951e758b70d60e-Abstract-Conference.html)：proper VE 的不足、统计摘要与 Bellman 闭合。
- [作者实现](https://github.com/tylerkastner/distribution-equivalence)：原文第 7 节直接链接的实验代码。

#### 作者代码

[正式原文第 7 节提供的作者仓库。](https://github.com/tylerkastner/distribution-equivalence)

分布模型等价与风险敏感实验；不提供任意任务的安全证书。

### An Idiosyncrasy of Time-discretization in Reinforcement Learning

Kris De Asis, Richard S. Sutton

RLC 2024 / RLJ · 2024 · 支持方法与理论

#### 研究问题

同样的物理奖励流，为什么会因奖励和折扣放在区间的不同位置而得到不同目标？

#### 关键机制

从连续时间回报的右端点近似出发，让区间奖励与后继价值按到达时间共同折扣。固定间隔时只差一个比例，不等间隔时这个比例一般无法提出求和。

#### 证据

原文给出时间离散化分析与实验。教材用恒定奖励率的两段时间计算，比较左右端点近似与精确积分。

#### 条件与限制

奖励率采样与已经积分的区间奖励不同。该修正不能消除动作延迟或低采样率遗漏事件，也不是任意 SMDP 接口都应照搬的公式。

#### 阅读与实验

固定每秒的目标而非每步折扣，再改变采样周期及抖动。报告积分误差、每秒更新次数和控制收益。

#### 原文与相关入口

- [RLC 正式记录](https://rlj.cs.umass.edu/2024/papers/Paper164.html)：正式出版入口。
- [原文推导](https://arxiv.org/html/2406.14951v2)：式 7 与不均匀时间步的回报定义。


<a id="chapter-code"></a>

## 下载与运行

标准库解析实验：周期策略偏好、随机停止、终止与截断、势函数塑形、在线与冻结评价。

[下载 objectives_lab.py](https://yingwen.io/zh/continual-rl/download/objectives_lab.py)

```sh
python3 objectives_lab.py all
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 2nd ed.](https://incompleteideas.net/book/the-book-2nd.html)：第 3 章的智能体边界、奖励、回报与价值；第 10 章的持续任务平均奖励。

- [MIT Press · Reinforcement Learning, second edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)：正式第二版出版信息与作者入口。

- [Bowling, Martin, Abel & Dabney · Settling the Reward Hypothesis · ICML 2023](https://proceedings.mlr.press/v202/bowling23a.html)：将经验偏好可由奖励表示的条件形式化，而非无条件肯定所有目标都能由任意标量奖励表达。

- [Sutton, Bowling & Pilarski · The Alberta Plan for AI Research](https://arxiv.org/html/2208.11173v3)：长期交互、时间一致性、有限计算与基本智能体；状态、主要策略、其他策略、价值函数和模型有不同作用。

- [Abel 等 · A Definition of Continual Reinforcement Learning · NeurIPS 2023](https://arxiv.org/html/2307.11046v2)：通过 agent basis、生成和到达关系定义持续学习，而非仅凭非平稳性分类。

- [Ng, Harada & Russell · Policy Invariance under Reward Transformations · ICML 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)：固定势函数塑形与策略不变性；有限轨迹推导需要保留边界项。

- [Wan, Naik & Sutton · Learning and Planning in Average-Reward MDPs · ICML 2021](https://proceedings.mlr.press/v139/wan21a.html)：从奖励率与差分价值进入可执行的学习与规划算法。

- [平均奖励作者代码 · average-reward-methods](https://github.com/abhisheknaik96/average-reward-methods)：原论文预测、控制和实验配置；周期计算用于理解目标，后续平均奖励章解释更新算法。

- [Cassandra, Kaelbling & Littman · Acting Optimally in Partially Observable Stochastic Domains · AAAI 1994](https://cdn.aaai.org/AAAI/1994/AAAI94-157.pdf)：隐藏 Markov 状态、观测与信念；已知模型下把信息获取和环境控制放入同一优化问题。

- [Javed & Sutton · The Big World Hypothesis and its Ramifications for Artificial Intelligence](https://oaklab.ai/posts/the-big-world-hypothesis)：问题选择与容量不匹配假设，不是关于一切环境的定理。

- [Kumar et al. · Continual Learning as Computationally Constrained Reinforcement Learning](https://arxiv.org/html/2307.04345v3)：第 2 节明确比较完整智能体的平均奖励与计算限制，同时指出平均奖励不能区分所有有限时间损失；第 3 节区分计算、信息和物理容量。

- [Khetarpal、Riemer、Rish 与 Precup · Towards Continual Reinforcement Learning: A Review and Perspectives](https://arxiv.org/pdf/2012.13490)：第 6.2–6.3 节将累计回报与保持、迁移、技能复用等探测指标分开讨论；本章的窗口指标和小迷宫数值由原创脚本独立定义、计算。

- [Sutton、Bowling、Pilarski · The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173)：Research Vision 区分经验、时间统一、计算约束和其他智能体；本文预算例子是教学分析。
