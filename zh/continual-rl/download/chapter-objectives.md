# 强化学习问题的形式化：交互、目标与持续学习

一个长期运行的智能体应当优化什么？这个选择怎样影响状态、价值函数、学习算法和评价？

## 本章内容

- 先写出定义域、给定量、未知量、信息权限、候选解和目标，再选择求解方法。
- 从奖励序列推导回报与价值函数，比较回合总奖励、折扣目标和平均奖励。
- 说明随机停止解释和势函数塑形的成立条件，计算条件失效时的反例。
- 在同一资源预算下比较整个学习过程与冻结策略，区分 continuing、continual 和 non-stationary。

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


<a id="chapter-prerequisites"></a>

## 预备知识与符号

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

<a id="lesson-setting"></a>

## 1. 先定义问题：接口、未知量、可用信息与解

本书先形式化要解决的问题，再讨论怎样求解。读者先要确定：允许哪些交互、决策时知道什么、比较哪些候选对象、什么结果算更好。价值函数、网络、TD 更新和规划随后才作为求解工具进入。改变这些条件中的任意一项，都应重新检查原来的结论。

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

这个一般历史过程允许环境有记忆、部分可观测及随时间变化的响应。$P_t$ 上的时间下标只是保留这种可能性，并不证明环境非平稳。完整历史总能作为形式上的状态，但其空间会随时间增长；这不意味着智能体能保存它，或能计算它的精确价值。

| 问题规格 | 必须明确的内容 | 解的含义 |
| --- | --- | --- |
| 定义域与给定量 | 动作、观测、奖励、初始条件、时间单位、评价期限与终止规则。 | 候选对象必须在此接口与时间边界下运行。 |
| 未知量 | 真实响应核 $P$、隐藏状态或奖励参数可未知；给出允许的环境族 $\mathcal E$。 | 未知模型的学习问题与已知模型的规划问题不同。 |
| 可用信息 | 部署时可见的历史、任务标签、模型调用、预训练数据和重置权限。 | 动作及更新不得读取未来奖励、隐藏真状态或未授权测试数据。 |
| 候选集合 | 指定固定策略类 $\Pi$，或完整实现集合 $\mathfrak L$；预算可限制后者。 | 最优只相对于所规定的集合，不能默认包含任意计算量的理想智能体。 |
| 目标与比较规则 | 给出 $J_P$，并说明逐环境保证、环境先验平均、最坏情形或指定比较器。 | 一次运行的得分、期望性能与跨环境保证不同。 |
| 假设与精度 | 可测性、奖励可积性、Markov 性、平稳性、可达性及允许误差。 | 若上确界不能达到，应给近似解或性能保证，不能默认最优解存在。 |

$$
J_P(\mathcal L)=\mathbb E_{P,\mathcal L}[U(\tau)],\qquad
 J_P^*=\sup_{\mathcal L\in\mathfrak L}J_P(\mathcal L),\qquad
 J_P(\mathcal L_\varepsilon)\ge J_P^*-\varepsilon
$$

$\mathcal L$ 是含初始化、行动、学习和记忆更新的可执行智能体；它诱导行为规则 $\Lambda$。$\tau$ 是完整交互轨迹，$U$ 是规定的轨迹评分，要求期望存在；$\varepsilon\ge0$ 是容许性能差距。该式在一个固定 $P$ 中定义比较标准，并未允许未知模型的学习器预先读取 $P$。平均奖励等准则在后文直接定义，不必都写成某个无限轨迹评分的期望。

若希望在未知环境中设计同一个学习器，还要规定跨环境的要求。例如给定环境先验 $\eta$，优化 $\mathbb E_{P\sim\eta}[J_P(\mathcal L)]$；或在环境族 $\mathcal E$ 中优化 $\inf_{P\in\mathcal E}J_P(\mathcal L)$。前者是先验平均，后者是最坏情形；它们一般选择不同学习器。单个真实环境的最优值可以作理论比较器，但不是学习器得到的额外信息。

考虑一台长期运送物品的机器人。“提高每小时完成的订单数”规定任务。“记住是否载货”规定决策需要的信息。“使用循环网络和 TD 更新”规定求解方法。这些选择互相影响，但不能互相替代。换一个状态编码器，不会自动改变订单收益；改变失败惩罚，则可能改变最优行为。

智能体与环境的边界是建模边界。它把所研究的决策和学习过程，与提供观测、奖励及动作后果的其余过程分开。边界不必沿机器人的外壳划定。若研究高层导航，电机控制器可以属于环境；若研究电流控制，它就属于智能体。奖励生成逻辑相对这个学习器位于环境一侧，不能由当前策略任意改写评分规则。

| 问题 | 数学对象 | 典型研究内容 |
| --- | --- | --- |
| 哪些结果更好？ | 奖励与轨迹评价 $J$ | 折扣、平均奖励、风险约束、生命期收益。 |
| 过去的哪些信息要保留？ | 历史摘要 $h_t$ | Agent state、部分可观测性、记忆。 |
| 未来会发生什么？ | 条件预测、价值函数和模型 | GVF、预测校准、转移模型。 |
| 怎样改变行为？ | 策略、学习规则和规划过程 | TD、actor–critic、元学习、模型规划。 |

Sutton 与 Barto 的 reward hypothesis 把目标和目的理解为期望累计标量奖励的最大化。这是研究与建模主张，不是无需条件便成立的数学定理。它没有指定奖励的具体数值，也没有声称任意给出的奖励都准确表达设计者的意图。Bowling 等在 ICML 2023 研究了更精确的表述：给定对经验结果的偏好关系，哪些条件使奖励表示成为可能。其结论依赖所列假设，不等于所有现实偏好都满足这些假设。

一个任务定义至少要写出动作、观测、奖励、时间单位、终止或重置规则，以及评价方式。安全约束、允许使用的信息和计算预算也需要明示。“是否完成任务”与“每秒完成量减去能耗”不是同一个目标。

理解关系时，先问改变了哪一部分规格。问题包含只说明某类问题能嵌入另一类；目标替换可能改变解；方法近似可能引入误差；协议限制则改变可行集合。这四种关系不能用一根“越来越先进”的箭头替代。

| 两个概念 | 关系类型 | 成立条件 / 改变量 | 带来的结构与边界 |
| --- | --- | --- | --- |
| 一般历史过程 → MDP | 附加环境结构 | 存在可观测充分状态 $S_t$；响应只依赖 $(S_t,A_t)$。若核还不随时间改变，得到平稳 MDP。 | 带来状态上的 Bellman 递推；不保证状态有限，也不保证模型已知。 |
| MDP 与 POMDP | 信息结构与问题包含 | POMDP 的隐藏状态满足 Markov 条件，但接口只提供观测；MDP 是状态被完全揭示的特例。 | 在有限已知模型下，信念是充分统计量；它通常属于连续空间，计算仍可能困难。 |
| MDP 与 bandit | 去掉受动作控制的后续状态 | 标准无上下文 bandit 可写成一个观测状态、每次拉臂生成奖励的 MDP；一般 bandit 的奖励过程还需单列假设。 | 未知臂收益仍需探索；没有受控状态的长程后果，不等于没有长期学习目标。 |
| 折扣 ↔ 有限期限 ↔ 平均奖励 | 替换评价准则 | 对同一奖励过程改变时间聚合。 | 一般不保留最优行为；有限启动代价与任务目标章的周期路线给出反例。 |
| 完整历史 → 有限记忆 / 神经网络 | 限制表示或求解方法 | 把历史压缩到可实现摘要，并用参数共享近似价值、策略或模型。 | 摘要未必充分；网络变深不构成新的环境类别，也不保证最优值可表示。 |
| 平稳 ↔ 外部非平稳 | 改变响应假设 | 相对于已声明的状态，转移或奖励核是否随时间改变；时间扩张可恢复形式上的齐次性。 | 扩张可能需要无界、不可观测或未知的状态；策略变化引起的数据漂移不自动是环境变化。 |
| 一般学习器 → 严格流式 / single-life | 限制交互与实现协议 | 分别限制经验回放与外部重置；两者可独立设置。 | 可行学习器集合缩小，最优值可能下降；不是新的奖励准则。 |
| 有界智能体 + Big World | 预算限制与研究假设 | 前者精确定义可用内存和计算；后者主张关注环境复杂性超过智能体能力的问题。 | 有限容量不逻辑蕴含永久学习；是否需要持续适应还取决于环境、目标与候选集合。 |
| 一般 RL → Abel 等的 CRL | 相对基底的性质判定 | 固定环境、性能、候选智能体及生成它们的 agent basis，再判断所有最优者是否永不到达基底。 | 不是 MDP → 深度 RL → CRL 的升级链，也不取消价值函数。 |

下面沿这些区别展开：第 2 节选择评价准则与评价对象，并说明 Markov 假设带来的可解结构；第 3–7 节检查期限、停止、奖励率和奖励变换是否保持原问题；第 8 节区分辅助问题；第 9 节把学习与资源限制纳入候选对象。第 10 节的计算因此是对形式化定义的检验，而不是先选算法再寻找任务。

<a id="lesson-derive"></a>

## 2. 选择准则与评价对象，再构造价值递推

有了交互规格，下一步才选择怎样评价奖励序列。有限期限适合已知部署时长；折扣给远期收益较小权重；平均奖励关注长期单位步收益。自然终止、重置和采样截断是过程边界，不能从评价公式反推。

$$
\begin{aligned}
 J_{P,T}(\Lambda)&=\mathbb E_{P,\Lambda}\!\left[\sum_{t=0}^{T-1}R_{t+1}\right],\\
 J_{P,\gamma}(\Lambda)&=\mathbb E_{P,\Lambda}\!\left[\sum_{t=0}^{\infty}\gamma^tR_{t+1}\right],\quad0\le\gamma<1,\\
 g_P(\Lambda)&=\liminf_{T\to\infty}\frac1T\mathbb E_{P,\Lambda}\!\left[\sum_{t=0}^{T-1}R_{t+1}\right].
 \end{aligned}
$$

$T\ge1$ 为指定期限，$\gamma$ 为折扣系数；$P$ 与 $\Lambda$ 共同诱导轨迹。这里 $g_P$ 使用期望平均的下极限，避免预设普通极限存在。奖励有界是这些量有限的一组充分条件。若每步实际时长不同，单位时间评价需另外定义。它们均可用于固定策略或完整学习器，评价对象与准则是两条独立轴。

| 选择 | 突出什么 | 仍需补充什么 |
| --- | --- | --- |
| 有限期限累计奖励 | 计入期限内探索、启动与恢复成本。 | 期限和初始分布；随机自然终止还需检查可积性。 |
| 折扣回报 | 区分近期与远期收益。 | 折扣与真实时间的关系；固定折扣一般不能替代动作相关生存概率。 |
| 长期平均奖励 | 比较持续运行的长期产出。 | 链结构或下极限约定；有限启动损失可能被抹去。 |

不等价的最小反例：在初始状态选择路线，随后不可切换。A 每步奖励 1；B 第一步奖励 $-100$，此后每步奖励 2。期限 $T=10$ 时，A 得 10，B 得 $-82$；折扣 $\gamma=0.9$ 时，A 值为 10，B 值为 $-100+2\gamma/(1-\gamma)=-82$；长期平均却分别为 1 和 2。另两条路线 A 恒得 1、B 周期得到 $(0,0,4)$，可使有限期限、不同折扣与平均奖励各自改变排名；任务与目标章的周期路线算例给出完整计算。

先区分三个问题。策略评价给定未来行为，求其收益预测；控制在给定策略类中选择行为；学习器设计比较产生行为和持续更新的整个可执行过程。学习器虽然改变参数，其初始化和更新程序可以固定，因此在一般历史空间上仍诱导一个固定的行为规则。

| 问题 | 给定 / 未知 | 求什么 | 核心求解难点 |
| --- | --- | --- | --- |
| 固定策略评价 | 给定 $\pi$、目标准则与模型或样本；真实价值未知。 | 计算或估计条件回报。 | 未知环境带来采样误差；表示限制带来近似误差；不是直接最大化收益。 |
| 控制 | 给定环境接口、评价与策略类 $\Pi$；优良策略未知。 | 求 $\sup_{\pi\in\Pi}J_P(\pi)$ 或近似最优策略。 | 动作改变未来状态、收益与信息；价值估计与行为改善相互作用。 |
| 完整学习器设计 | 给定信息权限、环境族与实现预算；候选 $\mathcal L$ 诱导未来更新。 | 按指定跨环境规则比较实际整段交互。 | 探索、更新、记忆及规划都产生资源成本和行为后果，末次参数不足以代表表现。 |

$$
\begin{aligned}
 V_{P,\Lambda}^{\gamma}(h)&=\mathbb E_{P,\Lambda}\!\left[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}\mid H_t=h\right],\\
 V_{P,\Lambda}^{\gamma}(h)&=\sum_a\Lambda(a\mid h)\int\!
 \left[r+\gamma V_{P,\Lambda}^{\gamma}(h\mathbin{\Vert}(a,r,o))\right]P_t(do,dr\mid h,a).
 \end{aligned}
$$

$h\in\mathcal H_t$ 为可实现历史，$h\mathbin{\Vert}(a,r,o)$ 表示接上一条经验，$o,r$ 为下一观测和奖励的取值。动作求和写于可数动作情形，连续动作改为积分；连续历史的条件期望用正规条件分布并按几乎处处意义理解。奖励有界且 $\gamma<1$ 时，一步拆分和条件期望给出递推。若 $\Lambda$ 来自学习器，未来更新已经包含在其后续历史行为中；这是学习器的历史价值，不是冻结当前参数后的价值。

所以 CRL 可以有价值函数。变化的是条件信息、未来行为规则及可行比较器。历史递推本身不提供紧凑、可计算的解；Bellman 等式成立与 TD 在有限表示中准确求解，是两个命题。冻结参数时还必须说明记忆递推、探索、模型和优化器是否冻结，否则连“固定策略”指什么也不清楚。

奖励 $R_{t+1}$ 是一次转移后收到的标量。回报 $G_t$ 是对未来奖励进行聚合的随机变量。策略 $\pi$ 规定动作分布。价值函数则是给定未来行为和当前信息后的期望回报。它们分别是信号、聚合规则、行为规则和预测量。下面的 $v_\pi,q_\pi$ 使用充分 Markov 状态 $S_t$ 与固定平稳策略 $\pi(a\mid s)$；一般历史行为的价值已经在上式定义。

$$
G_t^\gamma=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1},\qquad0\le\gamma<1
$$

若 $|R_t|\le R_{\max}$，则 $|G_t^\gamma|\le R_{\max}/(1-\gamma)$。较远奖励权重更小是目标定义的一部分，并不只是数值优化技巧。

$$
\begin{aligned}G_t^\gamma&=R_{t+1}+\gamma\sum_{k=0}^{\infty}\gamma^kR_{t+k+2}=R_{t+1}+\gamma G_{t+1}^\gamma,\\v_\pi(s)&=\mathbb E_\pi[G_t^\gamma\mid S_t=s],\\q_\pi(s,a)&=\mathbb E_\pi[G_t^\gamma\mid S_t=s,A_t=a].\end{aligned}
$$

第一行只把第一项从级数中取出。后两行引入期望与条件。$q_\pi$ 表示当前执行指定动作后，未来继续使用 $\pi$；它不要求这个动作是 $\pi$ 最常选择的动作。

若决策时能获得状态 $S_t=f(H_t)$，且对任意可实现历史满足以下条件，环境响应可压缩到状态核。若未来行为也只依据这个状态与固定策略，才可进一步将历史价值压缩成状态价值。这里 $f$ 是状态构造函数，$\mathcal S$ 是状态空间；是否充分由条件分布判定，不由“state”这个名称判定。

$$
\Pr(S_{t+1}=s',R_{t+1}=r\mid H_t,A_t=a)
 =p(s',r\mid S_t,a)
$$

左侧是环境响应与状态构造共同诱导的条件律：先生成 $(O_{t+1},R_{t+1})$，再令 $S_{t+1}=f(H_{t+1})$。它不是直接把观测核 $P_t$ 的输出改名。右侧的 $p$ 是不随时间变化的转移与奖励核。此式同时声明 Markov 性和环境平稳性；若右侧改为 $p_t$，仍可研究时间不齐次的 Markov 问题。有限 MDP 还要求 $\mathcal S,\mathcal A$ 有限。已知 $p$ 可直接规划，未知 $p$ 则需经验、模型估计或无模型更新。

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

[分册导读](https://yingwen.io/zh/continual-rl/start/classic-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-objectives) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=objectives) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=objectives)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

模型可提取性的理论说明某类能力需要什么知识，不指定唯一网络。Dreamer 研究潜在想象控制，STOMP 研究随机时长行为模型，DRAGO 研究旧模型知识保留。模型误差、查询策略和规划收益之间仍需实验连接。

- [Distributional Model Equivalence for Risk-Sensitive Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-distributional-model-equivalence)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [Rethinking the Foundations for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-rethinking-crl-foundations)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文提供定义和条件，架构讲座提出模块组织，算法论文检验特定机制。完整系统还要明确智能体、外部设计者和世界各自承担的工作；组件成立不自动意味着组合后的长期收益成立。

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
