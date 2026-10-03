# 持续强化学习：正文阅读顺序

## I · 强化学习问题与目标

从智能体与世界的持续交互出发，区分外部设计者的偏好、实际奖励和评价准则。奖励是假设与设计的对象，不是结果好坏的天然真值。折扣、时域和平均奖励进一步规定跨时间的比较。

- [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)
- [奖励假设与奖励设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/)
- [平均奖励与差分价值](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)

目标规定要改善什么。奖励机制传递学习信号。状态规定决策时可以使用什么信息。三者需要分别定义。

## II · 状态构造与表征

环境状态通常不可直接获得。智能体从观测、动作和奖励历史构造内部状态。状态更新、特征学习和参数学习是不同的计算过程。

- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)

给定状态后，可以定义关于未来的预测。状态是否充分，取决于它需要支持哪些预测与决策。

## III · 预测与预测知识

先固定策略，再学习回报预测。MC、TD 和资格迹提供不同的估计方法。GVF 进一步指定累积信号、延续条件和目标策略。预测不自动产生更好的行为。

- [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)
- [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)

控制需要使用预测来改变策略。策略改变后，数据分布和需要预测的未来也会改变。

## IV · 控制与策略改善

先区分固定策略的改善与完整学习器的生命期表现。持续控制中，动作同时改变世界、未来数据和后续学习。经典 GPI、动作价值和策略梯度提供局部工具；历史条件、可行比较器与资源限制决定这些工具能支持什么结论。

- [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)
- [深度价值学习](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)
- [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)
- [最大熵控制](https://yingwen.io/zh/continual-rl/algorithms/soft-control/)

这些控制方法都需要分配信用并选择更新尺度。持续交互要求学习过程也能适应。

## V · 时间信用分配

多步回报从未来反馈构造学习目标。资格迹在反馈到来时更新过去相关的参数方向。递归网络还需要传播参数对内部状态的影响。信用分配可以用于预测，也可以用于控制。

- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

信用分配规定反馈如何作用于过去。流式协议进一步限制数据保存与每步计算。

## VI · 流式学习与更新稳定性

流式学习规定经验何时使用、能否重放。归一化和更新尺度控制处理数值稳定性。这些约束与机制不等于信用分配，也不要求使用元学习。

- [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

固定的更新规则可以依据当前数据调整尺度。元学习进一步用经验改善学习规则本身。

## VII · 元学习与学习规则的适应

在线元梯度估计学习参数对后续表现的影响。跨任务元强化学习则利用任务分布，学习初始化、上下文推断或更新规则。两者的训练协议与评价对象需要分别定义。

- [元学习与学习规则的适应](https://yingwen.io/zh/continual-rl/algorithms/meta/)

有了学习机制，还需要规定可复用行为的目标、启动条件与终止规则。

## VIII · 子任务与时间抽象

任务的总体目标与子任务目标需要区分。目标条件化策略描述一族控制问题。Option 则规定行为的启动条件、内部策略和终止规则。技能发现决定哪些行为进入可用集合。

- [目标条件化与子任务构造](https://yingwen.io/zh/continual-rl/construction/goals/)
- [Options 与技能发现](https://yingwen.io/zh/continual-rl/construction/options/)

能够执行一个行为，不等于知道它的后果。规划还需要这个行为的模型。

## IX · 模型学习与规划

Dyna 用真实经验学习模型，再用模型更新价值。Option 模型预测随机时长行为的奖励和终点。规划决定在哪些状态、对哪些行为、使用多少次模型。

- [Dyna 与模型学习](https://yingwen.io/zh/continual-rl/algorithms/dyna/)
- [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)
- [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)

模型、状态、技能与策略都会变化。接下来需要研究这些变化怎样影响保留、适应与探索。

## X · 长期适应、保留与探索

遗忘指已有能力下降。可塑性损失指学习新能力的速度下降。探索改变后续经验。这三类问题需要不同的对照实验，不能用同一个训练误差代替。

- [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)
- [可塑性与特征更新](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)
- [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)

在完整智能体中，这些机制共享经验与计算资源。它们的相互作用需要单独研究。

## XI · 持续学习的智能体架构

状态为预测、策略和模型提供输入。子任务产生可学习的行为。模型支持规划。架构需要定义各模块的更新接口，也需要确定有限资源下的保留与替换规则。

- [持续学习的智能体架构](https://yingwen.io/zh/continual-rl/construction/architectures/)

架构提出机制假设。实验检验实现、机制与长期收益。

## XII · 强化学习实验方法

先定义要估计的量，再选择任务、基线与随机单位。开发数据用于选择配置。独立测试用于估计表现。持续学习还要区分在线收益、冻结诊断和恢复过程。

- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

实验设计贯穿全部章节。每学完一个更新规则，就可以运行小问题、提出反例并检验假设。

---

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


---

# 奖励假设与奖励设计

什么样的目标可以表示为奖励？智能体学会最大化奖励，是否就实现了设计者的意图？

## 本章内容

- 区分偏好、评价准则、奖励信号和辅助学习信号。
- 说明奖励表示定理的条件，并构造 Markov 奖励不能表达的排序。
- 推导势函数塑形、偏好学习和有限轨迹 MaxEnt IRL 的更新。
- 设计可以区分目标错误、奖励模型错误与优化错误的实验。

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


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 历史与策略

历史包含已经发生的观测和动作。策略或完整学习器诱导历史上的概率分布。设计者也可能观察智能体看不到的变量。

### 回报

本章用有限时域折扣回报推导。平均奖励和风险准则会另行说明。

$$
G_T=\sum_{t=0}^{T-1}\gamma^t R_{t+1}
$$

### 概率模型与梯度

知道条件概率、期望与链式法则即可。对数配分函数的梯度会在正文推导。

<a id="lesson-setting"></a>

## 1. 先确定谁的目标、谁的观测、谁的奖励

考虑一个送货机器人。设计者希望它按时、安全地完成送货。机器人收到的奖励 $R_{t+1}$ 却可能只由距离变化、送达计数和碰撞惩罚组成。希望实现的目标与实际奖励不是同一个对象。奖励设计研究二者怎样关联。控制学习则研究给定这些信号之后怎样行动。

| 对象 | 本章记号 | 送货例子 | 需要检查 |
| --- | --- | --- | --- |
| 设计者偏好 | ≽ | 安全完成比冒险抢时更好 | 对哪些结果、概率和时间作比较？ |
| 评价准则 | U 或 J | 完整生命期的交付量与事故限制 | 评价者能看到哪些变量？ |
| 奖励机制 | r 或 rψ | 程序或奖励模型输出的标量 | 能否被行为操纵？版本是否变化？ |
| 学习用信号 | r̃ | 奖励加塑形或探索奖金 | 它是否保持原目标？ |

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

平均奖励下可用未折扣势差。若势有界，T 步平均塑形奖励为 [Φ(sT)−Φ(s0)]/T，极限为零。这里保持的是长期奖励率；暂态收益与差分价值可以改变。

<a id="lesson-example"></a>

## 6. 三个会改变结论的细节

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

<a id="lesson-irl"></a>

## 8. 从示范中推断奖励：一个可手算的 MaxEnt IRL

偏好学习给出比较标签。逆强化学习给出示范行为。二者都要对人的行为与噪声作假设。这里使用有限、可枚举、等基准权重的可行轨迹集合，以及线性奖励。这个例子避开随机动力学中的因果熵问题。

$$
r_\theta=\theta^\top f,\qquad F(\tau)=\sum_t f_t,\qquad P_\theta(\tau)=\frac{\exp(\theta^\top F(\tau))}{Z(\theta)}
$$

Z 是所有可行轨迹指数分数的和。固定环境和轨迹集合。若轨迹具有不同基准概率，应把它们纳入分布，而不是当作奖励。

$$
\mathcal L(\theta)=\log Z(\theta)-\theta^\top\widehat F,\qquad \nabla\mathcal L=\sum_\tau P_\theta(\tau)F(\tau)-\widehat F
$$

对 log Z 求导：先对指数求导，再除以 Z，得到模型期望特征。负对数似然的梯度就是“模型特征−示范特征”。梯度下降使两者靠近。

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

配套程序只依赖 Python 标准库。先运行 demo 看数值，再运行 test 检查反例和梯度。它不训练深度奖励网络，不宣称复现下面论文的完整实验。

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

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

Laplacian 描述行为图结构，奖励感知表示加入路径价值，METRA 学习有区别的行为，HIQL 利用离线目标轨迹，MaestroMotif 引入语言先验。它们承担不同的设计工作；生成技能、选择技能与组合技能需要分别评价。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [Reward-Aware Proto-Representations in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-reward-aware-proto-representations)
- [MaestroMotif: Skill Design from Artificial Intelligence Feedback](https://yingwen.io/zh/continual-rl/research/#recent-maestromotif-semantic-skills)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

模型可提取性的理论说明某类能力需要什么知识，不指定唯一网络。Dreamer 研究潜在想象控制，STOMP 研究随机时长行为模型，DRAGO 研究旧模型知识保留。模型误差、查询策略和规划收益之间仍需实验连接。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)

### Reward-Respecting Subtasks for Model-Based Reinforcement Learning

Richard S. Sutton, Marlos C. Machado, G. Zacharias Holland, David Szepesvari, Finbarr Timbers, Brian Tanner, Adam White

Artificial Intelligence · 2023 · 支持方法与理论

#### 研究问题

学到一个能到达子目标的技能之后，为什么它仍可能不适合主任务规划？

#### 关键机制

STOMP 把子任务、option、模型和规划连起来。子任务保留原任务的路径奖励，并用带有特征偏好的终止价值表达目标；学习得到策略和终止规则后，再预测该行为的累计奖励与折扣终点。这样，技能不会因为只追求到达子目标而忽略途中代价。

#### 证据

论文用明确的小问题展示奖励感知子任务如何产生更有用的行为和规划模型。它提供的是可分析的构造链，而非只比较一个技能执行成功率。

#### 条件与限制

终止收益的约定是子任务定义的一部分，不能随意换成固定终点奖励。特征和子任务候选的选择尚不等于完整自主发现机制；实验也不构成整个 OaK 架构的验证。

#### 阅读与实验

在同一个绕路环境中比较“最短到达目标”和“保留路径奖励”的子任务。分别计算 option 的奖励模型、折扣终点模型与一次规划备份。

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

- [Singh, Lewis & Barto — Where Do Rewards Come From? (CogSci 2009)](https://web.eecs.umich.edu/~baveja/Papers/singh-lewis-barto-2009-cogsci.pdf)：内在奖励、外部评价与受限智能体。原文从最优奖励框架讨论奖励来源。

- [Sorg, Lewis & Singh — Reward Design via Online Gradient Ascent (NeurIPS 2010)](https://proceedings.neurips.cc/paper_files/paper/2010/hash/168908dd3227b8358eababa07fcaf091-Abstract.html)：PGRD：奖励参数通过受限决策过程影响外层表现。

- [Zheng, Oh & Singh — On Learning Intrinsic Rewards for Policy Gradient Methods (NeurIPS 2018)](https://proceedings.neurips.cc/paper/2018/hash/51de85ddd068f0bc787691d356176df9-Abstract.html)：LIRPG 原文。从外部回报沿策略更新求内在奖励梯度。

- [LIRPG：原论文作者实现](https://github.com/Hwhitetooth/lirpg)：原论文脚注给出的仓库。基于历史 TensorFlow / Baselines；用独立环境核对依赖，不与本站标准库示例混用。

- [Zheng et al. — What Can Learned Intrinsic Rewards Capture? (ICML 2020)](https://proceedings.mlr.press/v119/zheng20b.html)：跨多个生命期学习内在奖励；注意内外层时域与迁移条件。

- [Silver, Singh, Precup & Sutton — Reward is Enough (2021)](https://www.sciencedirect.com/science/article/pii/S0004370221000862)：研究假说：奖励追求与智能能力。与奖励表示定理分开读。

- [Bowling, Martin, Abel & Dabney — Settling the Reward Hypothesis (ICML 2023)](https://proceedings.mlr.press/v202/bowling23a.html)：读公理 1–5、定理 4.1 与设计者目标一节。注意历史域和时间一致性条件。

- [Abel et al. — On the Expressivity of Markov Reward (NeurIPS 2021)](https://arxiv.org/abs/2111.00876)：不同任务描述下的奖励表达限制。状态与奖励函数的允许输入非常关键。

- [Ng, Harada & Russell — Policy Invariance under Reward Transformations (ICML 1999)](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)：势函数塑形的原始论文。阅读时明确终止、折扣和保持最优策略的条件。

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


---

# 平均奖励：奖励率、差分价值与持续控制

没有自然终点时，怎样同时学习每单位时间的收益、状态的相对价值，以及最优行为？

## 本章内容

- 从时间平均目标推导 Poisson / Bellman 方程，解释为何值函数只确定到常数。
- 独立实现 Differential TD、Differential Q 与已知模型的 RVI，明确每条更新使用哪个旧值。
- 把原子动作推广到随机时长的 option，分清目标、数据协议与深度实现假设。

<a id="problem-definition"></a>

## 本章的问题定义

持续运行没有自然终点，关注每个原始时间步的长期收益；预测与控制分别求指定策略奖励率或最优奖励率。

### 给定条件与符号

- 固定MDP、原始步计时、有界奖励和可用动作。
- 预测时给定目标策略；控制时给定探索、访问与更新预算。

### 需要求解的对象

指定策略的奖励率与差分价值，或使奖励率尽可能大的策略及相应相对动作价值。

### 信息与数据权限

数据为 $(S_t,A_t,R_{t+1},S_{t+1})$；离策略预测另需实际行为 $b(A_t\mid S_t)$，目标策略为 $\pi$。技能更新另记录原始持续时间 $\tau$。

$$
g_\pi=\lim_{T\to\infty}\frac1T\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}R_{t+1}\right],\qquad g_* =\sup_{\pi\in\Pi}g_\pi
$$

$T$ 是原始环境步数，$\Pi$ 是允许的策略集合。预测只估计固定 $\pi$ 的 $g_\pi$；控制才比较 $g_*$。差分价值 $h_\pi$ 描述去掉奖励率后的相对收益，只在加常数意义下确定。

### 成立条件与解的含义

- 本章基础预测先假设奖励率不依赖初始状态，例如有限不可约策略链；控制需对应算法的通信和访问条件。
- 差分方程与相对值的锚定需要明确；深网、非平稳世界和随机技能不能直接继承表格收敛结论。

判断准则：小MDP上核对奖励率和Poisson/Bellman方程残差；比较相对价值差而非任意偏移；技能按原始时间计收益并检验时长扣除。

### 适用边界

- 有限窗口平均不等于已证明存在的无限时域奖励率。
- 奖励中心化参照量不自动等于精确平均奖励。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 改变评价目标 · [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)：将折扣累计量改成长期奖励率，预测对象随之变为奖励率与差分价值。

- 改变评价目标 · [Options 与技能发现](https://yingwen.io/zh/continual-rl/construction/options/)：相对options章的折扣控制，本章采用每个原始步的长期奖励率；同样的随机时长技能需扣除奖励率乘时长，而不使用折扣尾值。

- 组合不同学习问题 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：固定策略奖励率是局部控制工具；完整学习器仍需计入有限寿命的适应与探索成本。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

无折扣总奖励随时间增长，普通价值无法直接作为有限相对量；技能还改变了决策间隔。

### 本章的核心思路

估计共同增长的奖励率与剩余相对价值，并保持原始时间单位；联合TD更新与参考函数锚定是两种实现。

1. [减去长期增长项](#lesson-derive)：因为总奖励随时间线性增长，Poisson方程用奖励率分离增长与相对价值；Differential TD/Q以同一旧误差更新两种估计。

2. [锚定价值的平移自由度](#lesson-rvi)：因为相对值加常数仍满足方程，RVI用参考状态/函数约束坐标；这是另一实现，不要求Differential TD也固定参考状态。

3. [将机会成本按技能时长计算](#lesson-duration)：因为一个option消耗多个原始步，理想半Markov方程扣除奖励率乘实际时长；本章更新变体先用旧期望长度估计扣除并归一化，再更新长度，不能任意换成随机时长分母。

结论与条件：表格差分TD/Q与RVI有各自的链结构、步长及覆盖条件；本章深度骨架不提供普遍收敛或重置免费保证。

### 相关方法改变了什么

- Differential TD/Q：由同一TD误差联合更新奖励率与价值。

- RVI：用参考函数提供相对价值的锚定。

- 奖励中心化：处理共同奖励偏移；折扣联合更新的中心不应预先当成奖励率。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 状态与马尔可夫性

给定当前状态和动作，下一步奖励与状态的条件分布不再依赖更早历史；观测不充分时，需要先构造带记忆的 agent state。

### 策略评估与控制

策略评估求固定策略 $\pi$ 的价值；控制则寻找收益更高的策略。控制样本可由探索策略 $b$ 产生，因此样本奖励均值未必是最优策略的奖励率。

### TD 与半梯度

用当前估计的下一状态价值补齐未知未来，把它视作本步固定目标再更新当前估计；这不是对整个 Bellman residual 求全梯度。

<a id="lesson-setting"></a>

## 1 · 每单位时间的收益

考虑一个长期运行的排队服务器。每一步可以接收或拒绝请求，接收高价值请求有收益，但占用容量可能妨碍未来工作。不存在一个自然的“游戏结束”时刻。我们要比较单位真实时间的收益，而不是人为截断后的一回合总分。为先把目标讲清，本页从有限、平稳的 MDP 开始，假定所评估策略有唯一长期奖励率；学习期间环境改变的问题放在诊断一节单独讨论。

$$
g_\pi=\lim_{T\to\infty}\frac1T\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}R_{t+1}\right]
$$

$g_\pi$ 是每个原始环境步的平均收益。本章假设它与初始状态无关；例如有限不可约策略链满足这一点。多个互不连通的常返类可能具有不同奖励率，因而需要更一般的、依赖状态的表述。

未折扣的无限奖励和通常发散；平均奖励目标使用时间平均，不能仅由折扣回报中代入 $\gamma=1$ 得到。与此同时，没有自然终点的环境仍可采用折扣目标。目标函数、是否重放数据、是否允许外部重置，是三个需要分别指定的选择。

| 量 | 含义 | 在算法中何时改变 |
| --- | --- | --- |
| $g_\pi$ / $g_*$ | 固定策略 / 最优策略的真实奖励率 | 问题的性质，未知 |
| $\bar g$ | 奖励率估计 | 每条实际更新后改变 |
| $h(s)$ / $q(s,a)$ | 差分价值，又称 bias | 根据 TD 误差更新 |
| $\alpha$、$\eta$ | 价值步长、奖励率相对步长 | 本页固定；非平稳场景可研究自适应 |

奖励率衡量长期稳态表现，相对价值衡量从当前状态出发，在进入稳态之前比平均水平多赚或少赚多少。两者都需要：两条策略可能长期奖励率相同，但在前几百步的收益相差很大。对有限预算研究还要报告累计收益、变化后恢复时间，不能只报告最终 g。

<a id="lesson-derive"></a>

## 2 · 从时间平均到 Differential TD / Q

固定策略的有限时域价值记为 $V_T(s)$。先将第一步与剩下 $T-1$ 步分开。在有限、不可约且非周期的链上，采用适当的常数规范，可得到 $V_T(s)=Tg_\pi+h_\pi(s)+o(1)$。代入递推并消去线性增长项，便得到下面的方程。周期链不一定有这个逐点渐近形式，但仍可以直接研究相应的 Poisson 方程。

$$
\begin{aligned}V_T(s)&=\mathbb E_\pi[R_{t+1}+V_{T-1}(S_{t+1})\mid S_t=s],\\g_\pi+h_\pi(s)&=\mathbb E_\pi[R_{t+1}+h_\pi(S_{t+1})\mid S_t=s].\end{aligned}
$$

这是 Poisson 方程的状态形式。令 $P_\pi$ 为策略转移矩阵、$r_\pi$ 为期望即时奖励向量，矩阵形式为 $(I-P_\pi)h=r_\pi-g_\pi\mathbf1$。由于 $(I-P_\pi)\mathbf1=0$，$h$ 与 $h+c\mathbf1$ 满足同一方程。

差分价值 $h$ 的绝对数值依赖常数规范。可以指定某个参考状态的价值为零，也可以允许算法得到由初始化决定的偏移；跨运行比较关注差值、策略与 Bellman 残差。Differential TD 的基本版本不固定某个参考状态。

$$
\begin{aligned}\delta_t&=R_{t+1}-\bar g_t+h_t(S_{t+1})-h_t(S_t),\\h_{t+1}(S_t)&=h_t(S_t)+\alpha_t\rho_t\delta_t,\\\bar g_{t+1}&=\bar g_t+\eta\alpha_t\rho_t\delta_t,\qquad \rho_t=\frac{\pi(A_t\mid S_t)}{b(A_t\mid S_t)}.\end{aligned}
$$

On-policy 时 $\rho=1$。Off-policy 预测要求 $\pi$ 可能选择的动作在 $b$ 下也有正概率。两个更新共用旧参数下的 TD 误差，即计算一次 $\delta_t$ 后同时用于价值与奖励率。

奖励率也由 TD 误差更新，是因为在正确解处，对目标策略求条件期望后，奖励、奖励率与差分价值变化相抵消。只平均真实奖励则估计行为策略 $b$ 的奖励率；当 $\pi\ne b$ 时，它可能与价值头的目标不同。重要性比修正动作分布，但不提供任意非线性函数逼近与任意状态分布下的收敛保证。

$$
\begin{aligned}q_*(s,a)&=\mathbb E[R_{t+1}-g_*+\max_{a'}q_*(S_{t+1},a')\mid s,a],\\\delta_t&=R_{t+1}-\bar g_t+\max_{a'}Q_t(S_{t+1},a')-Q_t(S_t,A_t),\\Q_{t+1}(S_t,A_t)&=Q_t(S_t,A_t)+\alpha_t\delta_t,\\\bar g_{t+1}&=\bar g_t+\eta\alpha_t\delta_t.\end{aligned}
$$

Differential Q 用 max 定义控制目标；采到的 (s,a) 在自身转移分布下更新，因此表格单步控制式不再乘评估式的 π/b。仍需要充分访问状态动作与适当步长。

**算法：Differential Q-learning；统计窗口不改变学习器状态**

1. 初始化 $Q(s,a)=0$、$\bar g=0$，选择 $\alpha>0$、$\eta>0$
1. 在每个环境步：
  1. 按 $\varepsilon$-greedy 策略选择动作，观察 $(s,a,r,s')$
  1. 用更新前参数计算 $\delta=r-\bar g+\max_{a'}Q(s',a')-Q(s,a)$
  1. 更新 $Q(s,a)\leftarrow Q(s,a)+\alpha\delta$
  1. 更新 $\bar g\leftarrow\bar g+\eta\alpha\delta$
  1. 令 $s\leftarrow s'$，继续交互

在理论证明里常使用满足随机逼近条件的递减步长并要求覆盖；在持续漂移中通常保留常数步长追踪新目标，代价是稳态方差和不同的保证。将线性特征换成深网也不是无条件继承表格收敛性。

<a id="lesson-rvi"></a>

## 3 · 相对价值迭代与 RVI Q-learning

转移分布与奖励均值已知时，可以直接计算 Bellman 更新。它与折扣算子不同：给所有 $h$ 加同一常数，更新结果也整体平移，不会由 $\gamma<1$ 压缩这个方向。RVI 每轮减去参考状态更新后的值，保持一个约定的原点。

$$
\begin{aligned}(Th)(s)&=\max_a\sum_{s',r}p(s',r\mid s,a)[r+h(s')],\\h_{k+1}(s)&=(Th_k)(s)-(Th_k)(s_{\rm ref}).\end{aligned}
$$

初始化 h(sref)=0；在适当条件下稳定后，(Th)(sref) 给出 g*。不要把第一轮任意偏移下的参考值立即当作真实奖励率；周期性也可能导致普通同步迭代振荡。

$$
Q_{t+1}(S_t,A_t)=Q_t(S_t,A_t)+\alpha_t\big[R_{t+1}-f(Q_t)+\max_a Q_t(S_{t+1},a)-Q_t(S_t,A_t)\big]
$$

RVI Q-learning 是采样形式：用满足所选理论条件的参考函数 f(Q) 代替另学一个 ḡ。一个常见选项是参考状态动作的 Q 值。参考函数影响数值锚定与瞬态行为，不能随意换成任何神经网络统计量后仍引用原证明。

| 方法 | 奖励率 / 原点从何而来 | 样本或模型 |
| --- | --- | --- |
| Differential TD/Q | 由同一 TD error 更新 ḡ；h/Q 可留常数偏移 | 单步实际样本 |
| RVI | 每轮减去参考状态的 backup | 完整已知模型 |
| RVI Q | f(Q) 提供参照；理论限制 f 的性质 | 访问到的状态动作样本 |
| Differential planning | 模型提供真实转移分布的模拟样本或期望 backup | 需另外检查模型误差和更新调度 |

<a id="lesson-duration"></a>

## 4 · 随机持续时间与半马尔可夫更新

执行一次导航技能，可能持续 2 步，也可能持续 200 步。若每次 option 结束只平均奖励，慢技能可能因为单次收益高而被偏爱。单位原始步奖励应为总奖励除以总耗时。在独立重复执行、均值有限且平均时长为正的例子中，它是 $\mathbb E[R]/\mathbb E[\tau]$，一般不同于 $\mathbb E[R/\tau]$。

$$
q_*(s,o)=\mathbb E\!\left[R_{t:t+\tau}-g_*\tau+\max_{o'}q_*(S_{t+\tau},o')\mid s,o\right]
$$

$R_{t:t+\tau}$ 是 option 内未折扣的外部奖励总和，$\tau$ 是原始步数，最大值只对终点可用的 options 取。时间扣除项为 $g_*\tau$，反映技能执行期间的机会成本。

下面采用期望时长归一化变体，维护严格正的长度估计 $L(s,o)$。用旧 $L$ 计算奖励率扣除与步长归一化后，再更新 $L$；当 $L=\mathbb E[\tau\mid s,o]$ 时，期望 TD 误差与上述半马尔可夫方程一致。该更新假设 option 终止且期望长度有限。

$$
\begin{aligned}\delta&=R-\bar g L_{\rm old}+\max_{o'}Q(s',o')-Q(s,o),\\Q(s,o)&\leftarrow Q(s,o)+\alpha\delta/L_{\rm old},\\\bar g&\leftarrow\bar g+\eta\alpha\delta/L_{\rm old},\\L(s,o)&\leftarrow L_{\rm old}+\alpha_L(\tau-L_{\rm old}).\end{aligned}
$$

若改为采样 τ 直接扣除、同时又用随机 τ 作分母，噪声与期望可能改变；必须追踪相应算法及其假设，不能把 L 和 τ 任意互换。

<a id="lesson-example"></a>

## 5 · 手算：奖励率、控制目标与更新时序

例一，确定性两状态环：$0\to1$ 的奖励为 0，$1\to0$ 的奖励为 2。每两步收益为 2，因此 $g=1$。取 $h(0)=0$，由 $1+h(0)=h(1)$ 得 $h(1)=1$，另一个方程 $1+1=2+0$ 也成立。另取一条用于单步计算的样本：$h=[1,3]$、$\bar g=0.5$，转移 $0\to1$、奖励为 2，则 $\delta=2-0.5+3-1=3.5$。当 $\alpha=\eta=0.1$ 时，更新后 $h(0)=1.35$、$\bar g=0.535$。这条单步样本的奖励与前述固定环不同。

例二，单状态双动作：动作 0 给奖励 0，动作 1 给奖励 1，然后返回原状态。探索行为各做一半，实际奖励率为 $0.5$；最优奖励率为 $1$，最优方程要求 $Q(1)-Q(0)=1$。Differential Q 的 $\bar g$ 估计的是最优目标，因此应接近 1。这个例子把控制目标与行为奖励滑动平均明确区分开。

例三，当前 $Q=1$、$\bar g=0.5$、旧 $L=2$；option 累计奖励为 5，实际长度为 4，终点最大价值为 3。由 $\delta=5-0.5\times2+3-1=6$，取 $\alpha=\eta=0.1$ 得更新后 $Q=1.3$、$\bar g=0.53$；最后取 $\alpha_L=0.1$ 得 $L=2.2$。三个结果均按旧长度构造，更新顺序因此是算法定义的一部分。

<a id="lesson-code"></a>

## 6 · 完整更新、运行命令与输出诊断

可执行源码：函数返回新表格，避免原地修改使后续目标读到半更新参数。

```python
def differential_td(values, rate, state, reward, next_state,
                    alpha=0.1, eta=0.1, rho=1.0):
    """Tabular fixed-policy prediction. Compute BOTH updates from old values."""
    if alpha < 0 or eta < 0 or rho < 0:
        raise ValueError("nonnegative steps and importance ratio required")
    error = reward - rate + values[next_state] - values[state]
    result = list(values)
    result[state] += alpha * rho * error
    return result, rate + eta * alpha * rho * error, error


def differential_q(q, rate, state, action, reward, next_state,
                   alpha=0.1, eta=0.1):
    """Off-policy control: max target; no behavior-reward EMA."""
    error = reward - rate + max(q[next_state]) - q[state][action]
    result = [row[:] for row in q]
    result[state][action] += alpha * error
    return result, rate + eta * alpha * error, error


def rvi_sweep(h, transitions, reference=0):
    """Known model: transitions[s][a] contains (probability,reward,next_state)."""
    backed = [max(sum(prob * (reward + h[sp]) for prob, reward, sp in action)
                  for action in state) for state in transitions]
    reference_value = backed[reference]
    return [v - reference_value for v in backed], reference_value


def option_rate_step(q, rate, mean_duration, total_reward, duration,
                     next_best, alpha=0.1, eta=0.1, duration_alpha=0.1):
    """Duration-normalized expected-length variant; use OLD positive length."""
    if mean_duration <= 0 or duration <= 0:
        raise ValueError("option durations must be positive")
    error = total_reward - rate * mean_duration + next_best - q
    scaled = alpha * error / mean_duration
    return (q + scaled, rate + eta * scaled,
            mean_duration + duration_alpha * (duration - mean_duration))


def average_demo():
    values, rate = [0.0, 0.0], 0.0
    for t in range(20000):
        state = t % 2
        values, rate, _ = differential_td(
            values, rate, state, 2.0 * state, 1 - state, 0.02, 0.1)
    q, optimal_rate = [[0.0, 0.0]], 0.0
    for t in range(10000):
        action = t % 2  # behavior reward rate is exactly 0.5
        q, optimal_rate, _ = differential_q(q, optimal_rate, 0, action,
                                            float(action), 0, 0.02, 0.1)
    print("average", {"prediction_rate": round(rate, 6),
          "h1_minus_h0": round(values[1] - values[0], 6),
          "control_rate": round(optimal_rate, 6), "behavior_rate": 0.5,
          "option_step": option_rate_step(1, 0.5, 2, 5, 4, 3)})
    q_centered, reference = 0.0, 0.0
    for _ in range(1000):
        q_centered, reference, _ = centered_single_state_step(
            q_centered, reference, 1.0, gamma=0.9, eta=0.1, alpha=0.3)
    print("discounted_centering", {"q": round(q_centered, 6),
          "reference_c": round(reference, 6), "true_reward_rate": 1.0,
          "invariant_c_minus_eta_q": round(reference - 0.1*q_centered, 12)})
```

下载本页实验文件后运行；Python 3.10+，仅标准库

```sh
python lifelong_algorithms_lab.py average
python lifelong_algorithms_lab.py test
```

预期：两状态环 prediction_rate≈1、h1_minus_h0≈1；单状态控制 control_rate≈1，而 behavior_rate=0.5；option_step=(1.3,0.53,2.2)。测试额外验证值函数整体平移不改变 TD error、ρ=0 时不更新、零时长被拒绝、随机时长的“比值均值”反例。

作者仓库 average-reward-methods 将预测与控制分别放在 prediction_agents.py 和 control_agents.py。阅读时先对应奖励率与价值更新，再检查参考函数、探索策略、步长与环境重置规则。本页脚本用于表格机制实验；作者的统计实验还需要运行相应环境、多个随机种子及完整训练配置。

<a id="lesson-centering"></a>

## 7 · 奖励中心化：连接折扣价值与平均奖励

为什么 $\gamma$ 接近 $1$ 时，折扣价值往往很大，而动作之间的差异相对很小？设每步奖励减去固定常数 $c$。由于几何级数之和为 $1/(1-\gamma)$，所有策略、状态和动作的折扣价值都平移同一个量：

$$
Q_{r-c,\gamma}^{\pi}(s,a)=Q_{r,\gamma}^{\pi}(s,a)-\frac{c}{1-\gamma}
$$

此式假设无限持续交互、固定折扣且每一步均减去 $c$。因此固定 $c$ 不改变动作排序。若只在长度可变的 episode 内减去常数，累计平移依赖终止时间，这个结论就不再直接成立。

Naik 等人的 Reward Centering（RLC 2024）据此将共同的奖励偏移从价值学习中分离。它与 Differential TD/Q 共用一个重要思想：同时学习价值差异与标量参照量。但保留 $\gamma<1$ 时，学习的仍是中心化的折扣价值，而不是自动改成平均奖励控制。

$$
\delta_t=R_{t+1}-c_t+\gamma\max_{a'}Q_t(S_{t+1},a')-Q_t(S_t,A_t),\qquad c_{t+1}=c_t+\eta\alpha_t\delta_t
$$

这是 TD 驱动的控制中心化形式，$Q$ 同时按 $\alpha_t\delta_t$ 更新。$c$ 是联合学习的参照量；当 $\gamma<1$ 时，不应预先把它等同于精确的平均奖励率。固定常数的平移恒等式与这个联合学习过程是不同结论。

反例：单状态、单动作、每步奖励 1，取 $\gamma=0.9$、$\eta=0.1$、$Q_0=c_0=0$。两条更新使 $c_t-\eta Q_t$ 恒为零；稳定固定点为 $Q_*=5,c_*=0.5$，真实奖励率却为 1。此时 Q 正确表示中心化奖励 0.5 的折扣价值。详细推导见本章的 TD 中心化固定点研究节。on-policy 奖励均值、TD 参照与平均奖励 Differential TD 需分别命名和评价。

例如 $c=2$、$\gamma=0.99$ 时，共同价值偏移是 $200$；移除它可让网络更多容量用于区分状态和动作。Wan、Korenkevych 与 Zhu 的 continuing-task 研究（2025）进一步比较了无重置、预设重置和智能体控制重置的环境，并发现中心化不能完全消除大折扣带来的性能下降。它解决部分数值与估计问题，并不消除恢复困难或探索不足。

- 对照实验一：给所有真实奖励加同一个常数，比较中心化前后的学习曲线和价值尺度。
- 对照实验二：保留奖励但改变重置规则，区分数值问题与恢复能力。
- 对照实验三：分别采用行为奖励滑动平均、TD 驱动参照量和指定参考状态，检查 off-policy 目标是否一致。

<a id="lesson-branches"></a>

## 8 · 从差分预测到深度平均奖励控制

平均奖励 off-policy 预测加函数逼近后，数据分布与目标占用分布不一致会破坏简单 TD 的稳定性。Differential GQ 等方法引入辅助变量与投影目标；它们解决的是策略评估，不等于完成了未知目标策略的深度控制。本页先把表格目标和时序固定，避免将数学对象、求解器和网络结构混为一谈。

$$
\begin{aligned}\widetilde V_Q(s')&=\mathbb E_{a'\sim\pi_\theta}[Q(s',a')-\tau_{\!H}\log\pi_\theta(a'\mid s')],\\y&=r-f(Q)+\widetilde V_Q(s'),\qquad L_Q=\mathbb E[(Q(s,a)-\operatorname{stopgrad}(y))^2].\end{aligned}
$$

这是 soft average-reward critic 的概念骨架，不是 RVI-SAC 全部实现。熵温度 τH 与 option 时长 τ 不同；具体算法还需要目标网络、双 critic、参考项估计及 reset 成本等。

RVI-SAC 将参考项估计与最大熵控制结合。训练收益可包含熵和重置成本，评测的外部奖励率则可能不含这些项。若失败后的重置是持续过程中的真实转移，价值目标需要计入其后续收益；若它是问题定义中的吸收终止，边界条件便不同。重置的动作、代价与时间决定了目标的形式。

- 研究目标改变：比较平均奖励、折扣目标和 reward centering。保留 γ<1 的 centering 主要改变数值与 baseline，不自动变成平均奖励。
- 研究可追踪性：在不向 agent 提供切换时间的情况下改变奖励或转移，报告变化后累计损失与估计延迟。
- 研究函数逼近：固定特征、线性辅助变量、深网三层递进；先确定失效来自覆盖、投影还是表示漂移。
- 研究 SMDP：控制时长分布、奖励与时长相关性，并统一按原始环境步计算分母。

<a id="research-centering-fixed-point"></a>

## 研究专题 A · TD 中心化标量为何不总是奖励率？

固定 c 时，中心化折扣价值只发生共同平移。TD 中心化却同时更新 q 与 c，二者相互影响。下面用常奖励的单状态系统求解其固定点，检查 c 是否等于真实平均奖励率。

$$
\delta_t=r-c_t-(1-\gamma)q_t,\quad q_{t+1}=q_t+\alpha\delta_t,\quad c_{t+1}=c_t+\eta\alpha\delta_t
$$

单状态、单动作、常奖励 r，γ<1。所有右侧使用旧参数；c 是 TD 参照量，不预先称作真实 g。

$$
c_t-\eta q_t=c_0-\eta q_0=:k,\quad q_*={r-k\over\eta+1-\gamma},\quad c_*={\eta r+(1-\gamma)k\over\eta+1-\gamma}
$$

两种增量成比例，所以 c−ηq 不变；联立 δ=0 得固定点。确定性误差倍率为 1−α(η+1−γ)，还需满足其绝对值小于 1 的稳定条件。

取 $r=1$、$\gamma=0.9$、$\eta=0.1$、初值全零，得 $q_*=5,c_*=0.5$，实际奖励率却是 1。q 正是中心化奖励 0.5 的折扣价值 $0.5/(1-0.9)=5$。若令 $\gamma=1$，这个单状态方程才要求 $c_*=r$。

Reward Centering 的 TD 驱动参照、on-policy 行为奖励均值与 Differential TD 是不同对象。保留 γ<1 可改善共同数值尺度，却不能证明有限折扣与平均奖励在任意策略上排序相同。off-policy 场景也不能把真实行为均值直接当作目标策略的奖励率。

| 量 | 含义 | 检查 |
| --- | --- | --- |
| c | TD 学习参照 | γ、初始化约束及联合固定点。 |
| reward/time | 实际行为外部奖励率 | 完整奖励和真实时间。 |
| g | 平均奖励方法的目标奖励率 | 策略对象、覆盖与收敛条件。 |

**算法：参照语义的验证方案**

1. 常奖励单状态：逐步更新 q,c，核对 c−ηq 不变量
1. 改变 γ、奖励常数偏移和初值，核对解析固定点
1. 多动作控制：保持同一 reset 协议，比较排序和外部率
1. 深度比较：同预算分别加入行为均值与 TD 中心化

可运行的单状态更新与独立固定点参考；本章 average 命令打印反例，test 命令核对不变量、固定点与差分极限。

```python
def centered_single_state_step(q, reference, reward, gamma=0.9,
                               eta=0.1, alpha=0.1):
    """One-state diagnostic, not a complete reward-centering implementation.

    Discounted TD reference c need not equal the actual reward rate.
    Both writes use the SAME old-parameter error. gamma=1 is the
    differential limiting comparison, not discounted policy equivalence.
    """
    if not 0 <= gamma <= 1 or eta < 0 or alpha < 0:
        raise ValueError("invalid discount or nonnegative update scale")
    delta = reward - reference - (1 - gamma)*q
    return q + alpha*delta, reference + eta*alpha*delta, delta


def centered_single_state_fixed_point(reward, gamma, eta, q0=0.0, c0=0.0):
    """Solve c-eta*q invariant and zero TD error; not a stability claim."""
    if not 0 <= gamma <= 1 or eta < 0 or eta + 1 - gamma <= 0:
        raise ValueError("a positive fixed-point denominator is required")
    invariant = c0 - eta*q0
    q = (reward - invariant)/(eta + 1 - gamma)
    return q, invariant + eta*q
```

原始 Reward Centering 论文与 DeepRL-continuing-tasks 的 rc 配置可追踪具体变体。函数逼近、随机采样及不同更新时间尺度带来额外误差；单状态不变量用于发现语义混淆，不能直接推广成深网守恒律。

<a id="research-rvi-sac-reset-cost"></a>

## 研究专题 B · RVI-SAC：平均奖励、soft 参照与重置成本

RVI-SAC（ICML 2024）直接面向平均奖励最大熵控制。在 soft 后继价值中减去参考项，不使用小于一的环境折扣；另用 reset critic 和成本控制重置频率。完整系统不是仅把 SAC 的 γ 改成 1。

$$
\bar v(s')=\mathbb E_{a'\sim\pi_\theta}[\min_k\bar Q_k(s',a')-\tau_H\log\pi_\theta(a'\mid s')],\quad y=r-cd-f+\bar v(s')
$$

target critics、策略与温度构造 y 后停止梯度；d 是本次 reset 指示，s′ 是真实后继。τH 是熵温度，不是任务耗时。

$$
L_Q=\sum_k\mathbb E[(Q_k(s,a)-\operatorname{sg}(y))^2],\quad f^+=(1-\kappa)f+\kappa\zeta\mathbb E_{\rm batch}[\bar v(s')],\quad L_\pi=\mathbb E[\tau_H\log\pi_\theta(a\mid s)-\min_k Q_k(s,a)]
$$

f 式对应作者当前 rvi_sac.py 的移动参考，ζ 对应 fq_gain；其他 reference 变体需逐文件区分。actor 使用可重参数化动作，温度另行训练。

$$
y_d=d-f_d+\bar Q_d(s',a'),\quad L_{Q_d}=\mathbb E[(Q_d(s,a)-\operatorname{sg}(y_d))^2],\quad c^+=\max\{0,c+\alpha_c(f_d-p_0)\}
$$

reset critic 使用 reset 指示作为信号，fd 按对应移动参考更新。c 式是普通 dual 梯度步解释；作者实际用 Adam 再投影非负，不能把此式称作完整 Adam。p0 为目标 reset 频率。

若 fd=0.03、p0=0.01，梯度增大重置成本；若 fd=0.005，则减小至非负边界。成本改变 critic 与 actor，actor 又改变真实重置频率。这是依赖准确估计的反馈，并非一次失败便固定加罚。

**算法：对应 average_reward_drl/algorithms/rvi_sac.py 与 train.py；固定版本核对**

1. 从 replay 采样，以旧 f/fd/c 和 target networks 构造两个 target
1. 更新双任务 critic 与 reset critic，再更新 reference
1. 更新 actor 和温度；更新非负 reset cost
1. 最后更新 target networks
1. 采用 reset scheme 时：训练循环先实际 reset，再保存新状态为后继

| 分别报告 | 理由 |
| --- | --- |
| 外部奖励率 | 应用收益不混入 entropy。 |
| 熵及 reset 成本修正目标 | 与原始外部奖励数值不同。 |
| 重置率、恢复时间与真实耗时 | 仿真一步 reset 不等于即时免费复位。 |

作者 yhisaki/average-reward-drl 的固定成本与 reference 变体适合机制对照。精确平均奖励 soft improvement 的条件不自动覆盖重放、目标网络、非凸逼近及成本反馈联合学习。CRL 还需检查变化后的参照滞后和过时 replay。

<a id="lesson-check"></a>

## 9 · 习题与诊断

| 问题 | 应当能给出的解释 |
| --- | --- |
| h 全体加 100，策略会变吗？ | max 的相对次序和 TD 差分不变；参考规范化会改变存储值。 |
| 为何控制实验中 ḡ≈1，但真实流只有 0.5？ | ḡ 对应最优 Bellman 目标；实际探索仍选择低奖励动作。二者必须分别报告。 |
| 只有一次不可重置生命，能否用 g 的渐近目标解释全部价值？ | 目标仍可用，但有限生命损失、不可逆风险与恢复成本不能由极限奖励率单独概括。 |
| 算法收敛证明能否覆盖突然改变的深网实验？ | 不直接覆盖；需要重新陈述平稳性、函数逼近、覆盖和步长假设。 |

动手题：把两状态环的一条奖励改为 4，保留所有学习器状态。分别比较递减步长与常数步长的追踪速度；预测新 g=2、h(1)−h(0)=2。再把日志分段，但绝不在分段边界清零参数。将“记录一个新窗口”和“重新开始学习”严格分开。

## 本章的实验设计

记录每单位原始时间的收益。技能调用次数不能代替经过的时间。分别报告暂态和长期表现。

设定：先比较两个持续时间不同的固定技能：累计奖励分别为 2、9，时长分别为 1、9；再接小型非回合式 MDP 的 Differential TD/Q。用常奖励单状态反例检查有限折扣下的 TD 中心化。

- 总奖励率为 11/10，而不是两个技能奖励率的平均数。
- 每单位原始时间的奖励加同一常数后，奖励率按相同常数平移。
- 差分价值的常数偏移不改变所比较的动作优势。
- 中心化代码保持 c−ηq 不变量；r=1、γ=0.9、η=0.1、零初值时收敛到 q=5、c=0.5，而真实奖励率为 1。

对照：固定策略的解析奖励率；匹配动作持续时间的 primitive 对照；单独列出的折扣目标基线；行为奖励均值、TD 中心化和 γ=1 差分极限分别测试

记录：总原始奖励除以总原始时长；奖励率估计误差与暂态表现；差分 Bellman 残差及各状态访问量；中心化参照、实际奖励率与联合固定点误差分别报告

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-modules)

## 学习与研究衔接

平均奖励按原始时间计收益。随机时长 option 要使用半马尔可夫时间口径。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-average) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=average) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=average)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

Laplacian 描述行为图结构，奖励感知表示加入路径价值，METRA 学习有区别的行为，HIQL 利用离线目标轨迹，MaestroMotif 引入语言先验。它们承担不同的设计工作；生成技能、选择技能与组合技能需要分别评价。

- [Posterior Sampling for Continuing Environments](https://yingwen.io/zh/continual-rl/research/#recent-cpsrl-continuing-exploration)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [RVI-SAC: Average Reward Off-Policy Deep Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-rvi-sac-average-control)
- [Reward Centering](https://yingwen.io/zh/continual-rl/research/#recent-reward-centering-discounted)
- [An Empirical Study of Deep Reinforcement Learning in Continuing Tasks](https://yingwen.io/zh/continual-rl/research/#recent-continuing-task-deep-study)
- [Posterior Sampling for Continuing Environments](https://yingwen.io/zh/continual-rl/research/#recent-cpsrl-continuing-exploration)

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

标准库表格机制实验与单步公式测试；不是深度算法或论文 benchmark 复现。

[下载 lifelong_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)

```sh
python lifelong_algorithms_lab.py average
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Wan, Naik & Sutton · Learning and Planning in Average-Reward MDPs](https://proceedings.mlr.press/v139/wan21a.html)：表格 Differential TD/Q 与规划的原论文；对应本文奖励率由 TD error 驱动的核心。

- [作者实验仓库 · average-reward-methods](https://github.com/abhisheknaik96/average-reward-methods)：从 agents/prediction_agents.py 和 control_agents.py 对照更新，再读各实验配置；非本页教学代码的性能保证。

- [Wan, Naik & Sutton · Average-Reward Learning and Planning with Options](https://arxiv.org/abs/2110.13855)：不同 option 学习/规划更新与时长归一化；本页展示的是明确标记的 expected-duration 变体。

- [Wan & Sutton · Weakly Communicating MDPs](https://arxiv.org/abs/2209.15141)：扩展理解平均奖励控制的链结构和收敛条件；不可外推为任意非平稳深网定理。

- [ICML 2024 正式论文](https://proceedings.mlr.press/v235/hisaki24a.html)：平均奖励 soft improvement、RVI 与自动 reset cost。

- [RVI-SAC: Average Reward Off-Policy Deep Reinforcement Learning · 作者实现](https://github.com/yhisaki/average-reward-drl)：average_reward_drl/algorithms/rvi_sac.py 及其参照项、固定 reset cost 变体。 作者仓库 README 标明 reference code 与同名原论文。

- [RLJ 论文记录](https://rlj.cs.umass.edu/2024/papers/Paper261.html)：正式题名、作者与会议年份。

- [作者论文](https://arxiv.org/abs/2501.06937)：三类持续协议与深度中心化实验；2025 年 arXiv 首稿。

- [An Empirical Study of Deep Reinforcement Learning in Continuing Tasks · 作者实现](https://github.com/facebookresearch/DeepRL-continuing-tasks)：testbeds、Pearl 算法、experiments 配置与评测/作图。 论文对应 Meta 作者团队的研究仓库，README 明确区分三个 reset 协议。

- [RLC 2024 原文](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_261.pdf)：中心化分解、on/off-policy 区别及收敛讨论。

- [RLC 2024 原文](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_277.pdf)：随机重采样、折扣联系与 Bayesian regret 假设。

- [RLJ 论文记录](https://rlj.cs.umass.edu/2024/papers/Paper277.html)：作者、会议与理论结果。


---

# Agent state：部分可观测性、递归记忆与在线信用分配

任务目标给定以后，智能体应当保留哪些历史信息，才能预测未来并选择动作？

## 本章内容

- 区分真实环境状态、观测、历史、智能体状态、信念和可学习参数，识别观测混叠。
- 从条件概率推导 belief 更新，从链式法则推导 RTRL、BPTT 和 TBPTT，写清楚各自保留与丢弃的梯度。
- 读懂 RNN、GRU、LSTM、预测状态、GVFN、RTU 分别改变了什么，运行能逐项检查的记忆实验。
- 区分状态更新与参数学习，设计无任务边界、固定资源预算的 CRL 状态构造实验。

<a id="problem-definition"></a>

## 本章的问题定义

当前观测不足以预测后果或选择动作，需要从历史构造有限、可更新的决策信息。

### 给定条件与符号

- 观测、动作、奖励的因果流，以及指定的预测问题或控制评价。
- 状态容量、每步计算预算、递归结构和允许的训练数据。

### 需要求解的对象

可递推的历史摘要及其参数，使声明的后果预测或决策所需信息得到保留；不是重建全部历史。

### 信息与数据权限

$H_t$ 是完整已到达历史；$z_t=f_\phi(z_{t-1},a_{t-1},o_t)$ 是实际保存的摘要，$\phi$ 为状态更新参数。隐藏环境状态不作为免费输入。

$$
\operatorname{Law}(Y\mid H_t=h,a)=\operatorname{Law}(Y\mid z_t=z(h),a)
$$

$Y$ 是本任务指定的未来后果，$a$ 是当前干预动作，$z(h)$ 是历史的状态编码。此式表达相对该后果族的理想充分性；实际网络用预测损失、Bellman目标或控制目标近似检验，不宣称有限状态总能满足它。

### 成立条件与解的含义

- 充分性必须相对后果、未来行为和时间尺度定义；任意多预测坐标不自动构成充分状态。
- RTRL固定参数全历史敏感度、在线参数变化和截断BPTT分别说明，不能共用精确梯度称谓。

判断准则：在同观测但不同历史的别名反例上保持不同预测/动作；用匹配历史探针测预测误差，敏感度对固定参数有限差分吻合，并报告内存与每步时间。

### 适用边界

- 资格迹不能代替行动时使用的记忆状态。
- 状态充分性或预测精度不直接证明控制最优。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)：GVF可提供预测坐标，但其题目集合是否保留决策信息仍需验证。

- 组合不同学习问题 · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)：状态构造处理存什么；递归敏感度处理后来的误差怎样更新早先记忆参数。

- 改变信息或数据协议 · [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)：严格流式预算限制保存历史及展开计算图，影响可选递归结构和导数近似。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

历史含有有用线索但持续增长；学习记忆又需要计算参数通过过去活动影响当前输出的路径。

### 本章的核心思路

先指定摘要必须保留的后果，再选择递归结构及其可负担的敏感度计算。

1. [从历史别名识别缺失信息](#lesson-derive)：因为同一观测可对应不同未来，先用历史条件分布和预测状态区分需要保留的线索。

2. [把活动与参数求导分开](#lesson-rtrl)：因为递归活动是运行状态而参数是学习对象，RTRL分别递推活动和全历史敏感度。

3. [按预算选择结构或导数近似](#lesson-online-approximations)：因为一般敏感度昂贵，RTU限制耦合结构，BPTT/UORO分别截断或压缩导数；误差和资源分别检验。

结论与条件：敏感度恒等式限定在所写固定参数/计算图；RTU成本依赖局部结构，近似导数的无偏性不保证低方差或回报增益。

### 相关方法改变了什么

- 信念状态或PSR：有相应模型/可识别性条件时定义充分信息，未必低成本可学。

- RTRL/RTU：递推敏感度；RTU通过结构降低成本而非通用精确RTRL的免费替代。

- 截断BPTT/UORO：分别丢弃长路径或随机压缩导数，产生不同偏差与方差。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 条件概率

对未观察到的变量保留分布；新证据到来后按似然重新加权并归一化。不是把观测当成真实状态。

$$
P(x\mid o)=\frac{P(o\mid x)P(x)}{\sum_{x'}P(o\mid x')P(x')}
$$

### 链式法则

当前状态依赖旧状态，旧状态也依赖参数；求导必须同时包含当前直接影响和历史间接影响。

$$
\frac{d f_\theta(h_\theta)}{d\theta}=\frac{\partial f}{\partial\theta}+\frac{\partial f}{\partial h}\frac{dh_\theta}{d\theta}
$$

### TD 半梯度

把 bootstrap target 暂时视为常数，沿当前预测的梯度更新；这不等于完整地对平方 Bellman 残差求导。

$$
\theta^+=\theta+\alpha\bigl(r+\gamma v_\theta(h')-v_\theta(h)\bigr)\nabla_\theta v_\theta(h)
$$

<a id="lesson-setting"></a>

## 1. 问题设定：当前观测为什么不足以决定动作

任务目标规定哪些结果更好。状态构造规定决策时保留哪些信息。本章固定任务目标，研究第二个问题。智能体状态是可递归更新的历史摘要。它不等于当前观测，也不等于控制器要追求的目标。

设想走廊入口闪一次红灯或蓝灯，随后四个时刻看到的都是同一面灰墙。走到岔路，红灯要求向左，蓝灯要求向右。只接收当前灰墙图像的策略没有任何变量能区分这两段经历；在两种提示等概率、奖励只取决于最终方向时，再大的前馈网络也不能把平均正确率提高到 50% 以上。增加训练步数不能恢复已经被输入接口丢掉的信息。

| 对象 | 记号 | 定义与作用 |
| --- | --- | --- |
| 环境状态 | $X_t$ | 使环境动力学具有 Markov 性的变量，通常不可见；仿真器内部状态不属于默认的策略输入。 |
| 观测 | $O_t$ | 传感器当时给出的信息；同一观测可以来自不同环境状态。 |
| 历史 | $H_t=(O_0,A_0,R_1,\ldots,O_t)$ | 原则上可区分已见过的全部经历，但存储量随生命长度增长。 |
| 智能体状态 | 固定构造器时 $h_t=F_\theta(H_t)$ | 供预测与控制使用的有限历史摘要；可以是向量、分布、预测集或检索记忆。 |
| 学习参数 | $\theta_t$ | 决定状态更新规则的持久参数；与当时的活动状态不是同一种记忆。 |
| 梯度记忆 | $E_t$ | 保存历史如何影响参数梯度；它不同于活动状态，也不同于价值学习的资格迹。 |

$$
h_{t+1}=f_{\theta_t}(h_t,A_t,R_{t+1},O_{t+1}),\qquad A_t\sim\pi_{w_t}(\cdot\mid h_t)
$$

动作由 $h_t$ 产生，执行动作后得到奖励与新观测，再构造 $h_{t+1}$。讨论神经网络时，将动作、奖励和观测编码后合为输入 $u_{t+1}$。选择 $A_t$ 时尚未得到 $R_{t+1}$，时序必须保持一致。

$h_t$ 是当前活动，$\theta_t$ 是活动更新规则的参数。固定 $\theta$ 时，活动仍可随经验变化；这提供记忆，但不是参数学习。在线改变 $\theta_t$ 时，当前活动由过去各时刻的参数共同产生，通常不等于用最新参数重新处理全部历史的结果。分析整个学习器的运行状态时，还需包含参数、优化器统计量和梯度记忆。本章用 $h_t$ 专指感知与决策接口中的历史摘要。

状态有两类不同要求。预测充分性要求摘要保留某类未来预测需要的信息。控制充分性只要求摘要足以实现给定任务的最优行为。全面预测未来通常比完成一个特定任务要求更强。资源有限时，需要明确压缩要保留哪种能力。

$$
P(O_{t+1},R_{t+1}\mid H_t,A_t)=P(O_{t+1},R_{t+1}\mid h_t,A_t)
$$

若条件对所有相关历史和动作成立，且摘要能由自身与新经验按固定规则递归更新，则可递推得到多步动作条件的观测与奖励预测。这里要求所有动作条件，而不只是当前行为策略常见的动作。

$$
\exists\,\bar\pi\quad\text{s.t.}\quad\pi^*(a\mid H)=\bar\pi(a\mid f(H))\quad\text{for all relevant }H,a
$$

这是给定任务的一种控制充分性定义：至少存在一个最优策略 $\pi^*$ 能通过摘要 $f(H)$ 实现。它不要求重建全部未来观测。定义中的目标、可用动作和历史范围都必须固定；任务改变后，原摘要未必仍然充分。

例如未来屏幕背景色可由历史精确预测，但背景色不影响奖励、动作后果或决策。控制摘要可以丢掉它，仍实现最优行为，却不再足以预测完整图像。反过来，在当前策略下把某个价值预测准确，不保证能比较尚未尝试的动作，更不保证支持新的奖励任务。

精确信念提供预测充分状态的参照。RNN 指定可学习的摘要函数族。RTRL 和 BPTT 决定怎样训练参数。预测状态给摘要坐标规定经验语义。这些方法改变不同对象，可以组合，但每种组合仍需检查信息损失和计算成本。

- 允许信息：过去与当前的观测、自己的动作、已经到达的奖励，以及事先声明的模型。
- 不默认允许：环境隐藏状态、任务 ID、变化时刻、随意 reset、将未来奖励输入过去状态。
- 优化目标：最终仍是长期控制表现；预测误差、记忆任务准确率和状态重构误差是诊断，不是同一个目标。

<a id="lesson-derive"></a>

## 2. 有已知模型时，状态构造就是 Bayes 过滤

在有限 POMDP 中，$X_t$ 是隐藏状态，转移矩阵 $P_a(i,j)$ 给出执行动作 $a$ 后从 $i$ 到 $j$ 的概率，似然 $L_a(o\mid j)$ 给出到达 $j$ 后看到观测 $o$ 的概率。信念 $b_t(i)=P(X_t=i\mid H_t)$ 保留完整的状态不确定性，而不是只保留最可能的状态标签。两种信念即使具有相同的最大概率状态，也可能需要不同的辨识动作。

$$
\bar b_{t+1}(j)=\sum_i P_{A_t}(i,j)b_t(i)
$$

第一步预测：尚未用新观测，先将旧分布通过动作条件动力学向前传播。这里求和是对旧隐藏状态边缘化。

$$
b_{t+1}(j)=\frac{L_{A_t}(O_{t+1}\mid j)\bar b_{t+1}(j)}{\sum_k L_{A_t}(O_{t+1}\mid k)\bar b_{t+1}(k)}
$$

第二步校正：以新观测似然重加权预测分布，再除以证据概率。若奖励也包含隐藏状态信息，需要一并条件化；依赖旧状态和新状态的奖励由后面的联合核公式处理。

**算法：观测只依赖到达状态、奖励不另外提供信息时的 Bayes 过滤**

1. 初始化 $b$ 为已知初始分布。
1. 每次转移：
  1. 根据 $b$ 选动作 $a$；执行后获得 $o$。
  1. $\bar b(j)\leftarrow\sum_i b(i)P_a(i,j)$
  1. $u(j)\leftarrow\bar b(j)L_a(o\mid j)$
  1. 若 $\sum_j u(j)=0$，报告模型下不可能的观测。
  1. $b(j)\leftarrow u(j)/\sum_k u(k)$
  1. 用新信念做下一步预测与控制。

其依据是 Markov 条件：历史对下一步的影响先通过 $X_t$，再通过转移传给 $X_{t+1}$，因而知道 $b_t$ 就可以对未知旧状态积分。如果奖励还取决于旧状态，不能简单把它并入只依赖新状态的似然；应直接使用联合核 $P(j,o,r\mid i,a)$。下面的通式也涵盖奖励与观测相关的情况。

$$
b_{t+1}(j)=\frac{\sum_i b_t(i)P(j,O_{t+1},R_{t+1}\mid i,A_t)}{\sum_{k,i}b_t(i)P(k,O_{t+1},R_{t+1}\mid i,A_t)}
$$

已知、正确的有限状态模型使 belief 成为递归充分状态。模型未知、隐状态空间巨大或动力学变化时，仍需解决模型学习、近似推断与跟踪问题。

转移预测、似然校正与不可能观测检查；对应前面的简化观测模型。

```python
def belief_step(prior, transition, likelihood):
    """P[i][j] = P(next=j | current=i, selected_action)."""
    n = len(prior)
    if len(transition) != n or len(likelihood) != n:
        raise ValueError("inconsistent state dimensions")
    if any(len(row) != n for row in transition):
        raise ValueError("transition must be square")
    if any(v < 0 for v in prior + likelihood):
        raise ValueError("negative probability")
    if not math.isclose(sum(prior), 1.0):
        raise ValueError("prior must sum to one")
    if any(any(v < 0 for v in row) or not math.isclose(sum(row), 1.0)
           for row in transition):
        raise ValueError("each transition row must be a probability distribution")
    predicted = [sum(prior[i] * transition[i][j] for i in range(n))
                 for j in range(n)]
    unnormalized = [predicted[j] * likelihood[j] for j in range(n)]
    evidence = sum(unnormalized)
    if evidence <= 0:
        raise ValueError("observation has zero probability under this model")
    return [v / evidence for v in unnormalized]
```

<a id="lesson-recurrence"></a>

## 3. 学习递归状态：RNN、GRU 与 LSTM

不建立完整 POMDP 模型，也可以学习从历史到有用预测的递归压缩。最简单的 RNN 将旧状态与新输入送入非线性函数。设 $h\in\mathbb R^n$、$u\in\mathbb R^d$，$W\in\mathbb R^{n\times n}$、$U\in\mathbb R^{n\times d}$，每步活动更新的运算量约为 $O(n^2+nd)$。读出头既可以预测奖励或未来传感器，也可以输出价值与策略。

$$
z_t=Wh_{t-1}+Uu_t+b,\qquad h_t=\tanh(z_t),\qquad \hat y_t=g_\psi(h_t)
$$

状态构造器 $\theta=(W,U,b)$ 决定保留什么，读出头 $\psi$ 决定如何使用状态。只训练读出头而冻结 $\theta$，是在学习如何使用固定记忆特征。

GRU 和 LSTM 提供可调节的保留、擦除与写入路径。以下 GRU 采用 $z_t$ 越大越保留旧状态的约定；另一些实现使用互补门，或先做隐藏状态线性变换再乘重置门，参数不能直接逐项互换。门函数 $\sigma$ 是 sigmoid，$\odot$ 表示逐元素相乘。

$$
\begin{aligned}z_t&=\sigma(W_z h_{t-1}+U_z u_t+b_z),\\r_t&=\sigma(W_r h_{t-1}+U_r u_t+b_r),\\\tilde h_t&=\tanh(W_h(r_t\odot h_{t-1})+U_hu_t+b_h),\\h_t&=z_t\odot h_{t-1}+(1-z_t)\odot\tilde h_t.\end{aligned}
$$

保留门 $z$ 在旧记忆与候选状态间插值；重置门 $r$ 控制生成候选状态时参考多少旧记忆。门的参数也需要由训练信号学习。

$$
\begin{aligned}i_t,f_t,o_t&=\sigma(W_{i,f,o}h_{t-1}+U_{i,f,o}u_t+b_{i,f,o}),\\\tilde c_t&=\tanh(W_c h_{t-1}+U_c u_t+b_c),\\c_t&=f_t\odot c_{t-1}+i_t\odot\tilde c_t,\\h_t&=o_t\odot\tanh(c_t).\end{aligned}
$$

LSTM 区分 cell 记忆 $c$ 与读出活动 $h$。固定门时，直接保留路径的导数是 $\partial c_t/\partial c_{t-1}=\operatorname{diag}(f_t)$；完整 Jacobian 还包括门通过旧活动依赖旧 cell 的路径。

若 $f_t$ 接近 1，LSTM 的直接记忆通道衰减较慢；若 $z_t$ 接近 0，GRU 可以快速重写状态。但表达长记忆的能力不等于学会长记忆：在长度 8 的片段上训练，仍可能学不会相隔 200 步的因果关系。网络可表示什么、当前活动保留了什么、梯度能否追溯到记忆写入，必须分别分析。

<a id="lesson-rtrl"></a>

## 4. 从链式法则完整推导 RTRL

先固定整段序列使用同一个参数 $\theta$，并把输入序列视为给定。设 $h_t=f_\theta(h_{t-1},u_t)$，初始 $h_0$ 与 $\theta$ 无关。定义 $E_t=\partial h_t/\partial\theta$、$J_t=\partial f/\partial h_{t-1}$、$B_t=\partial f/\partial\theta$；求 $B_t$ 时保持旧状态不变。链式法则将导数分成当前直接作用 $B_t$ 与历史传播 $J_tE_{t-1}$。

$$
E_t=J_t E_{t-1}+B_t,\qquad E_0=0
$$

$E\in\mathbb R^{n\times p}$，$p$ 为参数数目。RTRL 在活动向前推进时同步维护敏感度，目标到达后不需要为求导重放全部旧输入。

$$
\nabla_\theta\ell_t=E_t^\top\nabla_{h_t}\ell_t+\left.\nabla_\theta\ell_t\right|_{h_t}
$$

最后一项只在损失直接使用 $\theta$ 时出现，例如读出与递归共享参数。若递归参数仅通过 $h_t$ 影响损失，这项为零。bootstrap 目标是否 stop-gradient 则决定这里求导的损失。

$$
\begin{aligned}D_t&=\operatorname{diag}(1-h_t\odot h_t),\qquad J_t=D_tW,\\\frac{\partial h_{t,k}}{\partial W_{ij}}&=(1-h_{t,k}^2)\left[\mathbf1_{k=i}h_{t-1,j}+\sum_l W_{kl}\frac{\partial h_{t-1,l}}{\partial W_{ij}}\right].\end{aligned}
$$

权重 $W_{ij}$ 直接连接旧节点 $j$ 与新节点 $i$，也通过旧活动继续传递历史影响。只保留括号内第一项是一步局部梯度。

**算法：RTRL 的前向敏感度计算；在线每步变参时保留递推，但改变了其精确导数解释**

1. 初始化 $h=0$、$E=0$。对固定参数 $\theta$ 的输入序列：
  1. 保存 $h_{\rm old}$、$E_{\rm old}$。
  1. $h\leftarrow f_\theta(h_{\rm old},u)$
  1. $J\leftarrow\partial f/\partial h_{\rm old}$
  1. $B\leftarrow\partial f/\partial\theta$，其中 $h_{\rm old}$ 固定。
  1. $E\leftarrow JE_{\rm old}+B$
  1. 目标到达时，用 $E^\top\nabla_h\ell$ 计算递归参数梯度。
  1. 当前片段结束后更新参数，以核验固定参数的完整梯度。

稠密 $n$ 单元 RNN 通常有 $p\approx n^2$ 个递归参数。敏感度的空间为 $O(np)=O(n^3)$，直接计算 $JE$ 的每步时间为 $O(n^2p)=O(n^4)$。这是普通稠密实现的成本，不是利用结构、稀疏性或随机低秩估计后的必然下界。

在线每步学习时，当前 $h_t$ 是历史上不同参数值产生的，而新参数不会重新计算全部过去活动。沿这条已实现轨迹维护的 RTRL trace，不等于“用当前 $\theta_t$ 重新运行整个历史”的导数。固定参数的有限差分检验可以发现微分实现错误；时变参数条件下，更新频率与状态滞后则是另外的研究问题。

标量 RNN 的三个参数（递归权重、输入权重、偏置）逐项维护敏感度。

```python
def rtrl(theta, inputs, initial=0.0):
    """Exact derivatives for one FIXED parameter vector and fixed initial state.

    h[t] = tanh(a*h[t-1] + b*x[t] + c), theta=(a,b,c).
    Parameters are not changed while processing the sequence.
    """
    a, b, c = theta
    h, eligibility = initial, [0.0, 0.0, 0.0]
    states, derivatives = [], []
    for x in inputs:
        old_h = h
        h = math.tanh(a * old_h + b * x + c)
        local = [old_h, x, 1.0]
        eligibility = [(1.0 - h * h) * (a * e + d)
                       for e, d in zip(eligibility, local)]
        states.append(h)
        derivatives.append(eligibility[:])
    return states, derivatives


def final_loss_grad(theta, inputs, target, initial=0.0):
    states, jacobians = rtrl(theta, inputs, initial)
    error = states[-1] - target
    return 0.5 * error * error, [error * e for e in jacobians[-1]]
```

<a id="lesson-bptt"></a>

## 5. BPTT 与 TBPTT：同一梯度，另一种计算顺序

RTRL 向前传播状态对参数的敏感度；BPTT 保存前向活动，再反向传播损失对状态的伴随量。设总损失为各步损失之和，$a_t$ 表示总损失对 $h_t$ 的导数。从最后时刻开始，逐步加上后续损失经状态递归传回的影响。

$$
\begin{aligned}a_t&=\nabla_{h_t}\ell_t+J_{t+1}^\top a_{t+1},\qquad a_{T+1}=0,\\\nabla_\theta L&=\sum_{t=1}^{T}B_t^\top a_t+\sum_{t=1}^{T}\left.\nabla_\theta\ell_t\right|_{h_t}.\end{aligned}
$$

在相同固定参数、相同目标、相同初始状态和完整序列上，BPTT 与 RTRL 给出同一梯度；区别是保存的中间对象和计算时序，不是优化了不同的数学目标。

完整 BPTT 对 $T$ 步序列的活动存储随 $T$ 增长。TBPTT 只在有限窗口内反传，并在片段边界 detach：状态数值继续传入下一片段，但跨边界的导数被切断。reset 则改变状态数值。二者对应不同实验：detach 限制训练信用范围，reset 还会删除当前已有的活动记忆。

$$
E_t^{(K)}=\sum_{k=0}^{K-1}\left(\prod_{j=0}^{k-1}J_{t-j}\right)B_{t-k}
$$

这是针对当前损失、最近 $K$ 步的截断敏感度；空乘积为单位矩阵，更早项置零。分块 TBPTT 中，各时刻距块边界不同，因此每个损失的有效窗口不一定都等于 $K$。

| 方法 | 保存什么 | 理想化成本 | 遗漏什么 |
| --- | --- | --- | --- |
| 完整 BPTT | T 步活动/计算图 | 时间 $O(Tn^2)$，活动内存 $O(Tn)$，另加参数 | 固定参数且完整反传时不截断历史梯度。 |
| TBPTT | K 步活动和边界状态 | 每块 $O(Kn^2)$，活动内存 $O(Kn)$ | 跨截断边界的参数影响；数值记忆仍可继续。 |
| 稠密 RTRL | 当前状态与 n×p 敏感度 | 每步 $O(n^2p)$，敏感度内存 $O(np)$ | 固定参数时不截断；在线变参时有历史不一致。 |
| 结构化 RTRL | 受限结构的局部敏感度 | 可显著低于稠密成本 | 先限制递归连接，再获得便宜的精确结构内导数。 |

同一个 RNN 的反向推导；window=2 切断导数而不清空旧状态。

```python
def bptt_final(theta, inputs, target, window=None, initial=0.0):
    """Final-state loss; window=None is full BPTT.

    A finite window holds the earlier boundary state numerically intact but
    treats it as a constant during differentiation (detach, not state reset).
    """
    states, _ = rtrl(theta, inputs, initial)
    a = theta[0]
    left = 0 if window is None else max(0, len(inputs) - window)
    adjoint = states[-1] - target
    grad = [0.0, 0.0, 0.0]
    for t in range(len(inputs) - 1, left - 1, -1):
        previous = initial if t == 0 else states[t - 1]
        dz = adjoint * (1.0 - states[t] ** 2)
        for j, feature in enumerate([previous, inputs[t], 1.0]):
            grad[j] += dz * feature
        adjoint = a * dz
    return grad
```

<a id="lesson-predictive"></a>

## 6. 预测状态与 GVFN：给记忆坐标一个可检验的含义

预测知识与状态摘要是两个对象。知识回答指定的未来问题。状态保存下一个预测或决策所需的信息。知识可以被读出使用，也可以选一部分预测直接作为状态坐标。后一选择形成预测状态结构，但不是所有预测系统都必须这样构造。

普通隐藏向量的第 7 维通常没有独立语义。预测状态的思路是用“如果执行指定行动，之后会观察到什么”的答案表示历史。一个有限 test 指定动作序列和观测序列，其预测是该观测序列在干预动作序列下出现的概率。若一组 core tests 的预测能够推出所有相关 test 的预测，这组数就可作为状态；这是一项充分性条件，不是随便放几个预测头都会满足。

$$
q_i(H_t)=P(o^{(i)}_{1:k_i}\mid H_t,\operatorname{do}(a^{(i)}_{0:k_i-1}))
$$

动作序列在这里是条件干预；概率描述执行该动作序列后出现指定观测的可能性，而非行为策略选择该序列的概率。预测状态避免显式命名隐藏 X，但仍需要学会稳定地更新这些预测。

GVF 用 cumulant $c$、continuation $\gamma$ 和目标策略 $\pi$ 定义预测问题。$c$ 可以是接触、红色像素或另一个定义明确的信号；$\gamma$ 指定预测结束或衰减的方式；$\pi$ 指定假设采取什么行动。将多组 GVF 的预测值直接作为递归状态分量，就是 GVFN 的结构思路。

$$
\begin{aligned}G_t^{(i)}&=c_{t+1}^{(i)}+\gamma_{t+1}^{(i)}G_{t+1}^{(i)},\\v_i(H_t)&=\mathbb E_{\pi_i}[G_t^{(i)}\mid H_t],\\h_t&=f_\theta(h_{t-1},u_t)\approx(v_1(H_t),\ldots,v_n(H_t)).\end{aligned}
$$

例如第 1 个分量预测持续前进直到碰墙前是否见红，第 2 个分量预测随机转向下未来 20 步的碰撞累计量。它们的时间尺度、动作条件和可用监督都不同，不是同一个奖励值函数的复制。

一个直接的半梯度基线将每个分量作为 TD 预测：由同一组旧参数计算 $h_t$ 与 $h_{t+1}$，构造误差，把下一步 target 视为常数，再沿 $h_t$ 的递归梯度更新共享参数。它解释预测语义如何约束记忆。原 GVFN 进一步分析 Bellman 网络投影误差及梯度校正；这些算法针对的目标和稳定性问题比下式更广。

$$
\begin{aligned}\delta_t^{(i)}&=c_{t+1}^{(i)}+\gamma_{t+1}^{(i)}h_{t+1,i}-h_{t,i},\\\theta^+&=\theta+\alpha\sum_i\rho_t^{(i)}\delta_t^{(i)}\nabla_\theta h_{t,i},\qquad \rho_t^{(i)}=\frac{\pi_i(A_t\mid H_t)}{b(A_t\mid H_t)}.\end{aligned}
$$

目标策略与行为策略不同时，动作重要性比是必要的纠偏对象之一，但非线性递归、bootstrapping 与 off-policy 的组合并不因此自动稳定。若目标动作在行为策略下概率为零，问题无法仅靠比率修复。

**算法：算法伪代码**

1. 为每个预测分量明确 $c_i,γ_i,π_i$；初始化递归网络与所选梯度记忆。
1. 每次真实转移：
  1. 用行为策略 b 选动作，并记录真实选取概率。
  1. 用旧网络状态与新输入计算 `h_next`；同时推进 RTRL 或 TBPTT。
  1. 逐问题计算 $c_i,γ_i,ρ_i$，以及 $δ_i$。
  1. 固定 bootstrap target，组合各分量的梯度更新共享 θ。
  1. 更新控制头；将 `h_next` 作为后续活动状态。
  1. 定期检查每个预测的校准、方差，以及对控制的增益。

如果只是把 GVF 头挂在一个自由隐藏层后面，那么隐藏状态本身不必等于预测：这是辅助任务结构。两种结构都值得比较，但不能混称。设计研究时还应设置坏问题对照：大量准确却与决策无关的预测，可能耗费预算而不改善控制。预测充分性、易学性和控制实用性需要分别测量。

<a id="lesson-rtu"></a>

## 7. RTU：通过递归结构降低敏感度计算成本

在稠密 RTRL 中，一个参数可经所有递归单元影响其他单元，形成昂贵的敏感度传播。结构化递归让小块独立演化：线性 RTU 的一个块可表示为二维缩放旋转，多块并行运行，由输入投影和输出头混合信息。于是结构内的敏感度具有局部性，可以保留长期递推而降低成本。

$$
\begin{aligned}z_t&=rM(\varphi)z_{t-1}+\sqrt{1-r^2}\,Wu_t,\\M(\varphi)&=\begin{pmatrix}\cos\varphi&-\sin\varphi\\\sin\varphi&\cos\varphi\end{pmatrix},\qquad 0<r<1.\end{aligned}
$$

$z$ 的两个实数分量等价于一个复数递归单元。$r$ 控制衰减，$\varphi$ 控制相位；平方根项归一化输入尺度，与 RL discount $\gamma$ 无关。两个新状态分量都必须使用同一个旧 $z$。

$$
\begin{aligned}e_t^r&=rM e_{t-1}^r+M z_{t-1}-\frac{r}{\sqrt{1-r^2}}Wu_t,\\e_t^\varphi&=rM e_{t-1}^\varphi+rM'(\varphi)z_{t-1}.\end{aligned}
$$

第一式包含输入归一化因子对 $r$ 的导数。每个块维护其两个分量对本地参数的导数；固定输入维度时，这部分成本随块数线性增加。若参数跨块共享，或输入来自另一个可训练递归层，则需要重新分析敏感度结构。

RTU（NeurIPS 2024）沿用 LRU 的结构化递归思路，并针对在线 RL 设计 trace 计算。作者实现对半径与角度进一步参数化，使用自定义梯度接口将敏感度接入学习。非线性的位置很重要：放在读出头与放进递归方程，会产生不同的 Jacobian。配套实验单独实现线性块对 $r$、$\varphi$ 的敏感度，用于检验归一化导数与更新时序。

可检查的旋转块核心：保留归一化导数，并用中心有限差分核验两类敏感度。

```python
def rotation_trace(inputs, radius=0.9, angle=0.4, input_weights=(0.7, -0.2)):
    """One linear RTU-style rotation block and exact r/angle sensitivities.

    Radius is differentiated directly; a paper implementation usually learns
    log-log radius parameters instead. No actor, critic, optimizer, or benchmark
    is included. Both new components use OLD recurrent components.
    """
    if not 0.0 < radius < 1.0:
        raise ValueError("radius must be strictly inside (0,1)")
    cs, sn = math.cos(angle), math.sin(angle)
    norm = math.sqrt(1.0 - radius * radius)
    h, dr, dp = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]

    def rotate(v):
        return [cs * v[0] - sn * v[1], sn * v[0] + cs * v[1]]

    for x in inputs:
        rotated, er, ep = rotate(h), rotate(dr), rotate(dp)
        angle_direct = [-sn * h[0] - cs * h[1],
                        cs * h[0] - sn * h[1]]
        dr = [radius * er[j] + rotated[j]
              - radius / norm * input_weights[j] * x for j in range(2)]
        dp = [radius * ep[j] + radius * angle_direct[j] for j in range(2)]
        h = [radius * rotated[j] + norm * input_weights[j] * x for j in range(2)]
    return h, dr, dp
```

原始仓库把递归网络、actor–critic/PPO 和实验配置分开。linear_rtus.py 中的 carry 同时包含活动与梯度记忆；src/agents 与 src/algorithms 将它接到控制损失；configs 决定部分可观测包装、网络宽度和预算。因此，网络结构能够逐步更新，并不意味着所有使用它的实验都没有 rollout、环境 reset 或其他控制算法状态。

<a id="lesson-online-approximations"></a>

## 8. 降低在线梯度成本：近似导数，还是限制状态结构

降低 RTRL 成本有两类不同选择。其一保留一般递归函数，近似庞大的敏感度 $E$；其二限制递归连接，使该结构内的导数本来就便宜。UORO（ICLR 2018）属于前者，LRU 与 RTU 的局部敏感度计算属于后者。前者承担估计方差或偏差，后者承担表达结构的限制；二者也可以组合。

随机低秩估计的核心可用一个恒等式理解。给定需要累加的两个外积 $ab^\top+cd^\top$，令随机符号 $\epsilon\in\{-1,+1\}$ 等概率取值。下面构造只存两个向量，却在条件期望上等于原来的秩二矩阵，因为交叉项包含均值为零的 $\epsilon$。在采样符号前选定正比例系数 $\rho$，可以调节方差而不改变这一期望。

$$
\widehat E=(\rho a+\epsilon c)(\rho^{-1}b+\epsilon d)^\top,\qquad\mathbb E_\epsilon[\widehat E]=ab^\top+cd^\top
$$

展开后交叉项为 $\epsilon\rho ad^\top+\epsilon\rho^{-1}cb^\top$。这解释随机压缩为何可以保留条件期望，却不保证每一步估计接近真值。将它用于 $E_t=J_tE_{t-1}+B_t$ 还需要处理 $B_t$ 的投影与尺度选择；不能把这个恒等式当成完整 UORO 算法。

梯度估计无偏不保证优化效果良好。长序列中的随机敏感度可能具有很大方差；有限样本训练可能不稳定。相反，TBPTT 有明确的截断偏差，却可能因为方差小、实现高效而表现更好。比较时既要看小网络中的梯度误差与方差，也要看相同计算和内存预算下的控制表现。原始 UORO 论文附有实现材料，方差分析文献进一步解释了这种权衡。

| 状态/训练路线 | 保留的结构 | 主要代价或近似 | 适合怎样的对照 |
| --- | --- | --- | --- |
| GRU/LSTM + TBPTT | 非线性门控递归 | 窗口之外的时间梯度截断；存储窗口内活动。 | 固定前向状态，改变反传窗口，隔离信用分配误差。 |
| 一般递归 + UORO 类估计 | 不要求递归矩阵按小块独立 | 随机敏感度的方差；在线变参仍有轨迹解释问题。 | 小网络完整 RTRL 为参照，分别比较均值、方差与算力。 |
| LRU/RTU + 结构化 trace | 线性/局部块递归，适当输入与输出混合 | 表达族和跨块耦合方式受限。 | 与同结构 BPTT、不同结构 TBPTT 分别比较。 |
| GVFN + 递归梯度方法 | 状态坐标受预测问题约束 | 预测问题是否有用、off-policy 学习是否稳定。 | 固定训练方法，替换状态语义；固定语义，再替换梯度方法。 |

实验可以从短的提示—延迟—决策任务进入 POPGym，再进入屏蔽位置或速度的连续控制。RTU 作者仓库已经包含 POPGym 的 JAX 实现与部分可观测 Brax 包装，适合追踪从记忆诊断到控制的变化。但这些仍有各自的 episode 和观测协议；研究持续状态构造时，应另外设置无任务标签的变化，保持学习器状态连续，并报告资源上限。

<a id="lesson-example"></a>

## 9. 完整算例：证据累计与延迟信用

算例 A：隐藏颜色不变，初始红/蓝各半，看到红提示的概率分别为 0.8 与 0.2。第一次看到红后：未归一化权重为 0.4、0.1，因此红的后验为 0.8。第二次仍看到红：权重为 0.64、0.04，后验为 16/17≈0.941176。若只记最后一个提示，每次都会回到 0.8，无法累计证据；若第二次观测与第一次完全相关而仍当作独立，0.941176 则是错误的过度自信。

$$
\frac{P(X=\mathrm{red}\mid o,o)}{P(X=\mathrm{blue}\mid o,o)}=\frac{0.5}{0.5}\times\frac{0.8}{0.2}\times\frac{0.8}{0.2}=16
$$

这里隐藏状态恒定、两次提示给定颜色后条件独立，所以后验赔率可连续乘似然比。一般有状态转移时必须先执行预测步骤，不能无条件累乘。

算例 B：标量 RNN 为 $h_t=\tanh(0.8h_{t-1}+0.3u_t)$，$h_0=0$，输入为 $(1,0,0,0)$，最终目标为 0.8。前向活动保留了第一步的正信号；完整 RTRL 与 BPTT 都得到损失对输入权重的导数约 $-0.275922$。若只反传最后两步，这两步输入为零，输入权重的梯度也恰好为零。活动记忆仍在，训练信号却无法归因到最初写入记忆的权重。

| 计算 | 递归权重 a 的梯度 | 输入权重 b 的梯度 | 偏置的梯度 |
| --- | --- | --- | --- |
| RTRL / 完整 BPTT | −0.339975 | −0.275922 | −1.792498 |
| 只对最后两步 TBPTT | −0.230183 | 0 | −1.139690 |

用完整梯度训练两种等概率提示，可以得到接近 −0.8 与 +0.8 的预测。该重置式监督任务将表示能力、活动记忆和梯度可达性分开，因而适合诊断训练循环。进入控制问题后，还需要分析探索、bootstrap 目标和状态分布变化，监督任务的结果本身不能替代这些实验。

<a id="lesson-code"></a>

## 10. 从公式到运行循环

下载本页配套脚本后运行；仅需 Python 3.10+ 标准库。

```sh
python3 state_meta_lab.py state
python3 state_meta_lab.py test
```

state 子命令依次运行信念更新、RTRL/BPTT/TBPTT 对照、延迟提示训练和旋转块敏感度。中心有限差分采用 $[L(\theta+\varepsilon)-L(\theta-\varepsilon)]/(2\varepsilon)$，$\varepsilon=10^{-6}$。它检查给定程序的解析导数；预测泛化、在线变参稳定性与控制回报需要另外的评测。

真实训练与评估循环：权重在整个短序列之后更新，状态重置权限显式写在代码中。

```python
def delayed_cue(epochs=1400, seed=3):
    """Controlled, resettable SUPERVISED sequence experiment, not lifelong RL."""
    rng = random.Random(seed)
    theta = [0.8, 0.3, 0.0]
    for _ in range(epochs):
        cue = rng.choice([-1.0, 1.0])
        inputs, target = [cue, 0.0, 0.0, 0.0], 0.8 * cue
        _, grad = final_loss_grad(theta, inputs, target)
        # Sequence starts from h=0; weights change only AFTER its full gradient.
        theta = [w - 0.08 * g for w, g in zip(theta, grad)]
    predictions = [rtrl(theta, [cue, 0.0, 0.0, 0.0])[0][-1]
                   for cue in [-1.0, 1.0]]
    return theta, predictions


def state_demo():
    prior = [0.5, 0.5]
    identity = [[1.0, 0.0], [0.0, 1.0]]
    first = belief_step(prior, identity, [0.8, 0.2])
    second = belief_step(first, identity, [0.8, 0.2])
    print("belief, one/two red cues:", first, second)
    theta, inputs, target = [0.8, 0.3, 0.0], [1.0, 0.0, 0.0, 0.0], 0.8
    loss, forward = final_loss_grad(theta, inputs, target)
    backward = bptt_final(theta, inputs, target)
    truncated = bptt_final(theta, inputs, target, window=2)
    print("RTRL gradient:", forward)
    print("BPTT gradient:", backward)
    print("TBPTT-2 gradient:", truncated, "(cue input-weight gradient is zero)")
    learned, predictions = delayed_cue()
    print("trained cue predictions:", predictions)
    print("rotation block state, d/dr, d/dangle:", rotation_trace([1, 0, 0]))
    print("scope: known-model filtering + fixed-parameter derivatives + supervised sequences")
```

- 实验一：把四步延迟改成二十步；分别记录预测误差、梯度范数和运行时间，不只看最终准确率。
- 实验二：令 TBPTT 窗口从 1 增至序列长度，检查输入权重梯度何时非零。保持相同前向活动，才能隔离反传窗口的影响。
- 实验三：将半径改为 0.5、0.9、0.99；比较衰减时间尺度以及归一化导数的大小。靠近 1 时也要监测数值条件。
- 实验四：引入偶发错误提示，比较最后一次提示、精确 belief 与训练 RNN；不要在 policy 输入中放入生成提示的隐藏颜色。

<a id="lesson-branches"></a>

## 11. 状态构造的研究分支

| 研究问题 | 改变的对象与代表路线 | 关键对照与局限 |
| --- | --- | --- |
| 历史信息是否够用？ | belief、PSR、GVFN：分别用隐状态分布、未来测试概率、长期条件预测描述历史。 | 比较对未训练动作与奖励变化的预测；单一行为策略上的低误差不等于状态充分。 |
| 怎样保存长时信息？ | RNN、GRU、LSTM、结构化状态空间/RTU：改变递归动力学。 | 匹配参数数目、单步算力与记忆；门控结构仍需要可达的训练信号。 |
| 怎样在线分配历史信用？ | BPTT、TBPTT、RTRL、低秩/局部近似、结构化精确 trace。 | 固定参数的梯度正确性与在线时变参数近似需要分别检验；计算吞吐量与统计效率也应分别报告。 |
| 应当预测哪些东西？ | 辅助预测、GVF 问题发现、representation meta-learning。 | 问题选择本身消耗数据与算力；要保留无关预测、随机预测和固定问题的对照。 |
| 状态怎样接入控制？ | recurrent actor-critic、预测状态输入、belief-conditioned policy。 | 冻结控制头测试状态漂移；冻结状态测试控制学习，排查二者相互追逐。 |
| 知识会不会过时？ | 状态构造器持续更新、变化检测、记忆管理。 | 区分正常动力学、环境变化与策略诱导分布变化；未通知变化不提供记忆重置权限。 |

GVFN 作者仓库中的问题定义、递归单元与学习方法应分别阅读：改变 GVF 集合是在改变状态语义，改变递归结构是在改变表示族，改变更新方法是在改变训练信用或稳定性。这三者需要独立消融。将 RTU 与预测状态结合时也相同：低成本长期梯度并不会自动选出有用的预测问题。

<a id="research-memory-training-interface"></a>

## 研究专题 A · Memoroids：记忆能保存多久，梯度能学习多久？

“隐状态能保留很久”和“参数能从早期事件学到什么”是两个问题。前者取决于递归动力学，后者还取决于训练中的梯度路径。即使活动保留了线索，在短块之间停止梯度，也可能无法教会网络哪些输入值得记住。Memoroids（NeurIPS 2024）提供一个重要对照：保持线性递归模型，改变长序列运算和训练批处理方式。

$$
h_t=A_t h_{t-1}+b_t,\qquad (A_1,b_1)\star(A_2,b_2)=(A_2A_1,A_2b_1+b_2)
$$

每次输入产生仿射变换，第二个变换作用在第一个之后。单位元为 (I,0)，一般不交换；对角 A 可降低合并成本。

$$
((A_1,b_1)\star(A_2,b_2))\star(A_3,b_3)=(A_3A_2A_1,A_3A_2b_1+A_3b_2+b_3)
$$

两种括号顺序结果相同，所以可以用树状 scan 计算前缀。固定大小合并的并行深度可为 O(log T)，总工作量仍为 O(T)；稠密矩阵乘法成本没有消失。

标量例子：$h_0=0$，三个变换为 $(0.5,1)$、$(0.5,2)$、$(0.5,3)$。逐步得到 $1,2.5,4.25$；组合为 $(0.125,4.25)$。若第三次输入来自新的独立回合，规定状态归零，则改成 $(0,3)$，前面活动被屏蔽，结果为 3。非零初始状态则将 $A_t h_{\rm init}$ 吸收入新偏置。

Tape-Based Batching 将多个完整回合存入一条 tape，以 begin/reset 信息阻断回合之间的状态传递，减少固定分段的补零和梯度截断。它仍保存序列并执行反传，既不属于严格逐步 RTRL，也不能让任意非线性门控模型采用同样的结合运算。

**算法：验证方案；不是论文实验已在本地运行的报告**

1. 固定一个早期线索、延迟决策的 POMDP和同一记忆模型
1. A：固定长度片段，保留活动但停止跨片段梯度
1. B：完整回合 tape，用 begin 标记阻断真实回合边界
1. C：允许相同数据协议时，对照结构化在线敏感度
1. 记录决策表现、早期观测敏感度、训练峰值内存与每步延迟

| 对象 | 作者实现入口 | 检查 |
| --- | --- | --- |
| 结合递归与边界 | proroklab/memoroids 的 memory、modules.py | 逐步、scan 与 begin 标记是否一致？ |
| 存储与切分 | buffer.py、segment_dqn.py、tape_dqn.py | 采样是否含完整回合？活动缓存来自哪版参数？ |
| 损失与回报 | losses.py、returns.py | target、padding 和终止定义是否匹配？ |

CRL 的进一步问题是参数变化时的状态一致性：实际活动由历次参数生成，从完整历史用当前参数重算则得到另一状态。缓存旧活动、burn-in、tape 重算和实时敏感度各有资源与近似边界。先固定参数验证代数等价，再开放参数更新测状态差异与控制表现；固定参数的恒等式不能证明不同学习时序的智能体等价。

<a id="research-state-query-sufficiency"></a>

## 研究专题 B · 状态充分性要相对于未来问题检验

把一张画面编码成漂亮的潜在向量，与从历史形成足够预测和行动的 agent state，是两个问题。DINO-WM 使用冻结视觉 patch 特征加观测历史做动作后果预测，V-JEPA 2-AC 也利用视频表示与动作条件预测器。它们提醒我们先检查表示接口中有哪些历史与运动信息，再讨论“模型理解了世界”。单帧自监督特征本身不保证隐藏速度、门锁状态或先前指令可被恢复。

$$
\begin{aligned}H_t&=(O_0,A_0,R_1,O_1,\ldots,A_{t-1},R_t,O_t),\quad Z_t=f_\theta(H_t),\\\mathcal Q&=\{(\pi_q,C_q,\gamma_q)\},\qquad v_q(h)=\mathbb E_{\pi_q}[G_q\mid H_t=h],\\Z(h)=Z(\tilde h)&\ \Rightarrow\ \begin{cases}\pi_q(\cdot\mid h)=\pi_q(\cdot\mid\tilde h),\\v_q(h)=v_q(\tilde h),\end{cases}\quad\forall q\in\mathcal Q.\end{aligned}
$$

这是目标策略可由该状态执行、且状态对查询族理想充分的联合规格，不是上述论文对任意历史的保证。每个 q 必须在两条历史上采用同一声明的策略规则；若规则要求已被状态丢弃的历史信息，先违反的是行为条件兼容性。查询族越丰富，允许丢弃的信息通常越少，但有限查询相容不等于任意控制的 Markov 充分性。

例子：两个历史都以“机器人站在门前”结束，但其中一个历史刚执行过解锁。若预测任务只问附近墙壁颜色，两种历史可以具有完全相同的正确答案；若增加“执行推门动作能否通过”的查询，二者必须区分。增加与门锁无关的预测数量无法补回这个信息。这个例子也说明，预测充分性必须包括行为条件，单纯预测自然视频的下一帧不足以验证反事实动作后果。

| 表示路线 | 怎样形成训练信号 | 须另外检验的条件 |
| --- | --- | --- |
| GVF / successor 表示 | 指定策略和时域下的未来量或占用 | 目标策略覆盖、查询族是否区分控制相关历史 |
| HILP 时间距离表示 | 离线目标价值约束潜在距离 | 有向距离、有限维嵌入和历史混叠 |
| DINO-WM 冻结视觉表示 | 动作条件的未来 patch 特征 | 视觉特征是否保留任务事件，历史窗口是否足够 |
| V-JEPA 视频与 2-AC | 视频潜在预测，再学机器人动作条件预测 | 动作坐标语义、预训练与部署视角、记忆跨度 |

**算法：从表示诊断推进到控制检验；不是将隐藏状态作为训练输入**

1. 受控状态实验（拟议）：
  1. 构造同观测、不同历史的成对起点；只把观测/动作历史交给 agent
  1. 固定未来行为策略，收集成对的后续轨迹
  1. 分别用单帧特征、固定历史窗口、可学习递归状态预测同一查询集
  1. 在冻结副本上测：隐藏条件可读性、未来预测误差、线性/非线性读出差
  1. 再用相同控制器和计算预算测试动作选择与全程回报
  1. 分开改变观测外观、隐藏动力学和奖励，保留失败结果

研究空缺在于：一个持续变化的查询族怎样反过来帮助构造状态，又怎样发现当前查询尚未区分的历史？可以提出新问题后只使用新发生的轨迹检验，但这仍需要合适的行为覆盖与在线信用分配。SF² 的状态动作特征、HILP 的离线距离、视频预训练都不能直接替代这一递归状态构造过程。

<a id="lesson-check"></a>

## 12. 诊断、自测与研究问题

| 现象 | 优先检查 | 不能直接下的结论 |
| --- | --- | --- |
| 同一观测的动作标签冲突 | 是否遗漏历史、动作或奖励；构造对照历史 | “加深网络就能解决”。 |
| 训练预测很好，控制很差 | 预测是否与动作选择相关；是否覆盖反事实策略 | “预测状态没有价值”。 |
| 梯度为零 | 饱和、截断边界、提前 detach、无监督信号 | “环境没有长期依赖”。 |
| replay 后状态失配 | 旧参数存的 h 与当前 θ 是否兼容；是否做 burn-in | “replay 必然不能用于 RNN”。 |
| 换任务就崩溃 | 是否借用了 task reset；状态与参数哪个需要适应 | “模型容量不足”。 |

- 问：RTRL 是否允许无限记忆却不增加运行内存？答：固定维度网络和固定参数维度下，敏感度内存不随时间长度增长；但信息压缩和梯度稳定性仍有限，不能称为无损无限记忆。
- 问：网络权重不更新时，h 变化算不算适应？答：是基于经验的活动/上下文适应，但不是参数学习。比较 RL² 与在线梯度学习时尤其必须区分。
- 问：把边界状态 detach 是否等于智能体忘了过去？答：不等于；detach 只改变求导图。只有数值重置或动力学衰减才直接改变当时存的记忆。
- 问：有一百个准确 GVF，是否已经有充分状态？答：没有这种数量保证；关键是问题集合是否区分了控制相关历史，以及动作条件是否覆盖。

可开始的研究题：在相同单步算力和持久内存下，逐渐增加提示到决策的延迟，再加入未通知的提示规则变化。比较 GRU+TBPTT、结构化递归+trace、手工充分记忆与观测基线。记录终生回报、变化后恢复时间、预测校准、每步延迟与内存。手工记忆是可达上界诊断，不是可学习算法的公平替代；变化时刻只供评测统计，不传入智能体。

## 本章的实验设计

冻结参数后，内部状态仍可继续递推。用这一诊断区分记忆计算与参数适应。

设定：构造两个当前观测完全相同、早期线索不同且正确动作相反的历史。线索只在合法时间出现，研究者真状态用于分析。

- 无历史模型不能凭额外隐藏标签区分这两个历史。
- 固定权重时循环活动仍随观测变化。
- 改变隔离的诊断相位，不改变同一合法输入下的动作。

对照：当前观察、普通 GRU 与候选状态模型；相同持久状态容量的简单递推；冻结权重、清记忆与保留记忆

记录：延迟线索任务的分条件成功率；固定参数分支的行为与记忆读出；状态、敏感度和每步计算量

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-checkpoint)

## 学习与研究衔接

深度网络不能自动消除部分可观测性。必须区分观测编码、历史状态递推与参数学习。

[分册导读](https://yingwen.io/zh/continual-rl/start/deep-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-state) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=state) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=state)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

状态是支持后续计算的内部信息；GVF 指定一个预测问题；RTRL 和资格迹规定信用如何传播。三者可以组合，但不是相互替代的算法名称。先理解给定策略的预测，再讨论预测怎样改善控制。

- [Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks](https://yingwen.io/zh/continual-rl/research/#recent-columnar-constructive-networks)
- [Real-Time Recurrent Learning using Trace Units in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-real-time-trace-units)
- [Towards model-free RL algorithms that scale well with unstructured data](https://yingwen.io/zh/continual-rl/research/#recent-nibbler-predictive-features)
- [When does Self-Prediction help? Understanding Auxiliary Tasks in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-self-prediction-auxiliary-tasks)
- [Recurrent Reinforcement Learning with Memoroids](https://yingwen.io/zh/continual-rl/research/#recent-memoroids-sequence-learning)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)
- [Does Zero-Shot Reinforcement Learning Exist?](https://yingwen.io/zh/continual-rl/research/#recent-zero-shot-forward-backward)
- [Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations](https://yingwen.io/zh/continual-rl/research/#recent-successor-flow-features)

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [Expected Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-expected-eligibility-traces)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)
- [Recurrent Reinforcement Learning with Memoroids](https://yingwen.io/zh/continual-rl/research/#recent-memoroids-sequence-learning)

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

Laplacian 描述行为图结构，奖励感知表示加入路径价值，METRA 学习有区别的行为，HIQL 利用离线目标轨迹，MaestroMotif 引入语言先验。它们承担不同的设计工作；生成技能、选择技能与组合技能需要分别评价。

- [Proper Laplacian Representation Learning](https://yingwen.io/zh/continual-rl/research/#recent-proper-laplacian-representations)
- [METRA: Scalable Unsupervised RL with Metric-Aware Abstraction](https://yingwen.io/zh/continual-rl/research/#recent-metra-skills)
- [HIQL: Offline Goal-Conditioned RL with Latent States as Actions](https://yingwen.io/zh/continual-rl/research/#recent-hiql-hierarchical-goals)
- [Foundation Policies with Hilbert Representations](https://yingwen.io/zh/continual-rl/research/#recent-hilbert-foundation-policies)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

模型可提取性的理论说明某类能力需要什么知识，不指定唯一网络。Dreamer 研究潜在想象控制，STOMP 研究随机时长行为模型，DRAGO 研究旧模型知识保留。模型误差、查询策略和规划收益之间仍需实验连接。

- [Mastering diverse control tasks through world models](https://yingwen.io/zh/continual-rl/research/#recent-dreamerv3-world-models)
- [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://yingwen.io/zh/continual-rl/research/#recent-dino-wm-feature-planning)
- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://yingwen.io/zh/continual-rl/research/#recent-vjepa2-action-conditioned)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Real-Time Recurrent Learning using Trace Units in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-real-time-trace-units)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Learning from experience instead of curated datasets](https://yingwen.io/zh/continual-rl/research/#recent-oak-network-idbd)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [The Cell Must Go On: Agar.io for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-agarcl)
- [OGBench: Benchmarking Offline Goal-Conditioned RL](https://yingwen.io/zh/continual-rl/research/#recent-ogbench-goal-evaluation)
- [Recurrent Reinforcement Learning with Memoroids](https://yingwen.io/zh/continual-rl/research/#recent-memoroids-sequence-learning)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文提供定义和条件，架构讲座提出模块组织，算法论文检验特定机制。完整系统还要明确智能体、外部设计者和世界各自承担的工作；组件成立不自动意味着组合后的长期收益成立。

- [The OaK Architecture: A Vision of SuperIntelligence from Experience](https://yingwen.io/zh/continual-rl/research/#recent-oak-architecture)
- [Towards model-free RL algorithms that scale well with unstructured data](https://yingwen.io/zh/continual-rl/research/#recent-nibbler-predictive-features)
- [Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations](https://yingwen.io/zh/continual-rl/research/#recent-successor-flow-features)

### Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks

Khurram Javed, Haseeb Shah, Richard S. Sutton, Martha White

JMLR 24 · 2023 · 支持方法与理论

#### 研究问题

如果每次观测只处理一次，如何学习包含历史信息的状态，而不保存一段序列做反向传播？

#### 关键机制

一般递归网络的实时递归学习需要维护庞大的参数—状态敏感度。CCN 限制列之间的递归依赖，并逐步构造新特征，使敏感度可以局部计算。它通过改变网络结构和构造过程降低求导成本，而不是把任意稠密 RNN 的完整导数免费变小。

#### 证据

论文分析受限结构的计算性质，并在动物学习启发的预测问题和 Atari 策略评价中检验预测效率。这里的 Atari 结果主要是预测已有策略的回报，不等于从头训练完整控制智能体。

#### 条件与限制

结构约束、构造顺序和被冻结的旧特征共同限制函数类。监督预测和策略评价上的优势，还需要在会主动改变数据分布的控制闭环中检验。

#### 阅读与实验

先写出递归状态对参数的敏感度递推，再检查哪些跨列项被结构消除。比较 CCN、截断 BPTT 与 RTU 时，同时计入状态、梯度缓存和每步计算。

#### 原文与相关入口

- [JMLR 原文与论文入口](https://www.jmlr.org/papers/v24/23-0367.html)：从网络结构、敏感度传播与预测实验三部分阅读。

### Real-Time Recurrent Learning using Trace Units in Reinforcement Learning

Esraa Elelimy, Adam White, Michael Bowling, Martha White

NeurIPS 2024 · 2024 · 支持方法与理论

#### 研究问题

递归状态既要保存长时信息，又要在在线强化学习中以可控成本更新，怎样设计其递归结构？

#### 关键机制

RTU 使用有结构的递归连接，并维护状态关于参数的在线敏感度。复杂的递归动力学可以用实值运算实现。其关键是让状态更新与梯度迹具有相容的计算结构，减少一般 RTRL 的高阶成本；这与仅给 TD 误差加一条资格迹不同。

#### 证据

论文在部分可观测任务中与常见递归网络比较预测与控制表现。作者代码包含 RTU、其他递归基线、实时 actor–critic 以及部分可观测环境配置。

#### 条件与限制

计算优势依赖特定递归参数化，不能外推为任意记忆问题上的表达能力优势。PPO 版本和严格逐步更新版本的经验协议不同，应分别比较。

#### 阅读与实验

在同一部分可观测任务中固定隐状态维度，再比较完整运行内存和每步更新时间。检查 actor、critic 与递归状态的参数更新是否共享同一条敏感度。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1e616bde0438cb10cb6adf076ae7d336-Abstract-Conference.html)：结构、在线导数与实验协议。
- [作者代码](https://github.com/esraaelelimy/rtus)：从 src/nets、src/agents 和实验配置追踪递归状态到控制更新。

#### 作者代码

[论文作者维护的实现。](https://github.com/esraaelelimy/rtus)

RTU 网络、实时学习器与论文实验配置。

### Towards model-free RL algorithms that scale well with unstructured data

Joseph Modayil, Zaheer Abbas

arXiv 预印本 · 2023 · 支持方法与理论

#### 研究问题

大量原始观测中只有少数局部组合与奖励有关，智能体能否逐步构造有用的预测特征？

#### 关键机制

Nibbler 将预测问题的构造和预测结果的复用结合起来：选择局部输入、学习与奖励相关的通用价值预测，再将预测作为后续学习的特征。GVF 在这里不是一个新的优化器，而是描述“预测什么、在什么行为下预测”的问题接口。

#### 证据

作者在组合式合成环境中增加观测规模，报告了利用任务结构的样本效率。环境可以具有指数增长的状态组合，但学习器不必显式枚举全部状态。

#### 条件与限制

这不是对任意高维观测的线性样本复杂度保证。局部可分解结构、候选问题与特征构造规则仍是关键条件；从该实验族迁移到视觉控制需要额外验证。

#### 阅读与实验

把一条预测完整写成累积量、延续条件、目标策略和输入特征四项，再指出它如何进入主任务的价值函数。区分问题生成带来的收益与增加参数量带来的收益。

#### 原文与相关入口

- [作者预印本](https://arxiv.org/abs/2311.02215)：问题族、Nibbler 构造过程与扩展性实验。

### When does Self-Prediction help? Understanding Auxiliary Tasks in Reinforcement Learning

Claas A. Voelcker, Tyler Kastner, Igor Gilitschenski, Amir-massoud Farahmand

RLC 2024 / RLJ · 2024 · 支持方法与理论

#### 研究问题

预测下一潜在状态、重建观测和学习价值，为什么会产生不同的表示？

#### 关键机制

论文在含干扰因素的线性问题中分析辅助目标的学习动力学。潜在状态自预测与价值学习共同作用时可能保留决策相关结构，但单独训练同一目标未必得到最有用的特征。目标的作用取决于它和 TD 目标怎样共享表示。

#### 证据

线性分析给出可检查的条件，并用神经网络实验检验部分预测。结果不支持“任何自监督预测都能改善 RL”这种无条件判断。

#### 条件与限制

线性分析中的观测映射、优化过程与神经网络控制并不完全等价。项目仓库入口不等于已经提供完整可复现实验实现，因此这里不列为可运行代码。

#### 阅读与实验

固定编码器容量，分别比较仅 TD、仅辅助任务和联合训练。记录价值误差与任务收益，不要只用辅助损失下降评价表示。

#### 原文与相关入口

- [RLC 2024 论文入口](https://rlj.cs.umass.edu/2024/papers/Paper197.html)：原文、线性假设与神经网络实验。
- [作者预印本](https://arxiv.org/abs/2406.17718)：便于追踪论文版本。

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

### HIQL: Offline Goal-Conditioned RL with Latent States as Actions

Seohong Park, Dibya Ghosh, Benjamin Eysenbach, Sergey Levine

NeurIPS 2023 · 2023 · 支持方法与理论

#### 研究问题

只拿到已有轨迹时，长距离目标为什么适合拆成高层子目标和低层动作？

#### 关键机制

HIQL 学习目标条件价值，并以潜在状态作为高层动作。高层提出中间目标，低层输出环境动作；两层利用优势加权回归学习。时间分解让低层面对较短的控制距离，而不是要求一个策略直接消化所有远距离价值误差。

#### 证据

论文在离线长时域目标任务中检验层次结构，并提供原始实现。作者后来在 OGBench 中提供更统一的实现，二者适合不同用途：原实验复现和统一基线比较。

#### 条件与限制

数据覆盖和行为分布约束仍然存在。目标采样、层级时间间隔与离线轨迹由外部流程提供，不能把效果解释为在线自主目标生成已经解决。

#### 阅读与实验

对一段轨迹明确标记最终目标、中间目标和当前动作。逐一检查价值目标、优势权重和高层标签的停止梯度边界。

#### 原文与相关入口

- [NeurIPS 2023 原文](https://papers.nips.cc/paper_files/paper/2023/file/6d7c4a0727e089ed6cdd3151cbe8d8ba-Paper-Conference.pdf)：离线目标学习和两层回归目标。
- [HIQL 原始实现](https://github.com/seohongpark/HIQL)：README 区分原始实验与 OGBench 中的新实现。

#### 作者代码

[作者仓库；更新的统一基线另见 OGBench。](https://github.com/seohongpark/HIQL)

HIQL 原论文的离线训练与评价。

### OGBench: Benchmarking Offline Goal-Conditioned RL

Seohong Park, Kevin Frans, Benjamin Eysenbach, Sergey Levine

ICLR 2025 · 2025 · 评价与实验协议

#### 研究问题

一个目标条件算法表现不好，是长时域、轨迹拼接、视觉表示还是随机性造成的？

#### 关键机制

OGBench 用不同环境类型与数据集分别施加这些困难，并提供统一的目标条件基线。固定离线数据让算法面对相同经验，从而将学习机制的差异与在线探索能力的差异暂时分离。

#### 证据

论文提供八类环境、八十五个数据集和六类算法实现。价值在于可复用的实验接口与困难分解，而不只是汇总一个排行榜。

#### 条件与限制

固定数据不检验智能体如何主动获得未来经验，也不直接检验单次生命的灾难性变化、恢复或长期资源管理。它适合 CRL 子问题实验，不是完整 CRL 的替代品。

#### 阅读与实验

先选择只改变一种困难的两个数据集，再比较 HIQL 与平坦目标策略。把观察到的差异写成可检验机制假设，而不是直接归因于“层次更好”。

#### 原文与相关入口

- [论文](https://arxiv.org/abs/2410.20092)：ICLR 2025；环境、数据与基线定义。
- [作者基准库](https://github.com/seohongpark/ogbench)：数据获取、环境与统一算法实现。

#### 作者代码

[基准作者维护的官方实现。](https://github.com/seohongpark/ogbench)

离线目标环境、数据集与标准化基线。

### Learning from experience instead of curated datasets

Oak Lab

Oak Lab 技术博文 · 2026 · 支持方法与理论

#### 研究问题

有用信号稀疏且大量输入是噪声时，在线学习规则如何分配不同方向的更新能力？

#### 关键机制

博文从含稀有有效特征的线性预测问题出发，对比统一步长与 IDBD 的逐权重适应，再展示 NetworkIDBD 在非线性带噪观测中的例子。核心主张是让长期学习效果影响信用和步长分配，而不只依据当前梯度幅度归一化。

#### 证据

公开页面提供受控噪声特征任务和 NoisyMNIST 示例。它们是机制演示，便于理解有效信号密度与输入规模的关系。

#### 条件与限制

该页面不是完整 CRL 控制论文，也未给出可直接复现所有图表的完整代码和算法推导。监督噪声任务的结果不能证明一般 SGD 或所有深度 RL 都无法从经验学习。

#### 阅读与实验

先复现线性噪声特征问题，分开改变有效特征稀疏度与噪声维数。进入控制前，再加入策略改变数据分布这一因素。

#### 原文与相关入口

- [Oak Lab 原始博文](https://oaklab.ai/posts/learning-from-experience-instead-of-curated-datasets)：2026 年 7 月 13 日；受控实验、NetworkIDBD 示例与研究动机。

### Mastering diverse control tasks through world models

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap

Nature · 2025 · 支持方法与理论

#### 研究问题

同一套世界模型训练与控制方法，能否减少跨任务重新设计损失和超参数的需求？

#### 关键机制

DreamerV3 从经验学习递归潜在状态、奖励和延续预测，再在潜在想象轨迹上学习 actor 和 critic。尺度稳健的表示与损失设计使同一配置可以适用于多种任务。模型是用于决策的学习接口，不必生成完整真实世界。

#### 证据

论文在大量视觉和状态控制任务上报告了广泛表现。关键含义是共享算法配置；这些结果主要来自分别训练的任务智能体，不是一个智能体按顺序学会全部任务。

#### 条件与限制

经验重放、批量训练和模型想象都有资源成本。模型偏差、表示遗忘与长期任务切换仍需要专门实验，不能由多任务覆盖范围自动推出持续学习能力。

#### 阅读与实验

把状态更新、模型训练、想象起点和策略更新四种分布分别写清。比较真实交互步数之外，还应记录想象步数和优化次数。

#### 原文与相关入口

- [Nature 原文](https://www.nature.com/articles/s41586-025-08744-2)：方法和任务协议；区分共享配置与单智能体持续学习。
- [作者维护的实现](https://github.com/danijar/dreamerv3)：公开实现的版本与论文实验环境应分别记录。

#### 作者代码

[作者发布的重实现；不把当前分支当作原论文实验的冻结快照。](https://github.com/danijar/dreamerv3)

DreamerV3 的作者维护公开实现及运行配置。

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

### The OaK Architecture: A Vision of SuperIntelligence from Experience

Richard S. Sutton

RLC 2025 讲座 / Oak Lab · 2025 · 定义与架构观点

#### 研究问题

持续学习是否只是在一个现成 actor–critic 上加入抗遗忘机制，还是需要重新安排知识构造与使用？

#### 关键机制

OaK 提出从经验持续形成状态、预测知识、子任务、时间抽象与模型，并让这些知识服务规划的架构方向。这里的重点是模块之间怎样产生可复用知识，而不只是保留某个固定策略网络的参数。

#### 证据

官方页面提供 Richard Sutton 的架构讲座与相关研究入口。STOMP、预测学习和在线特征学习等论文可以检验其中具体组件，但不能自动验证整体架构。

#### 条件与限制

这是研究愿景与架构讲解，不是一套已公布完整训练配方、统一基准结果和可复现端到端代码的系统。资源分配、问题生成、知识替换与模块相互干扰仍需明确算法。

#### 阅读与实验

为每个模块写出输入、输出、更新频率和资源上限。再选择一个双模块接口做可证伪实验，例如技能模型改善是否真的减少规划误差。

#### 原文与相关入口

- [Oak Lab 官方讲座页面](https://oaklab.ai/posts/the-oak-architecture)：讲座入口与架构研究方向。
- [Oak Lab 研究主页](https://oaklab.ai/)：区分已发表研究、技术文章和仍在预告中的项目。

### Recurrent Reinforcement Learning with Memoroids

Steven Morad, Chris Lu, Ryan Kortvelesy, Stephan Liwicki, Jakob Foerster, Amanda Prorok

NeurIPS 2024 · 2024 · 支持方法与理论

#### 研究问题

当记忆网络能够保存信息时，训练序列的切分是否仍会阻止学习器给早期信息分配信用？

#### 关键机制

Memoroids 将一类线性递归模型写成结合运算，利用并行 scan 处理长序列；Tape-Based Batching 将多个完整回合接入同一条 tape，用显式边界处理状态重置，减少分段、补零和截断反传带来的问题。

#### 证据

论文在 POPGym 等部分可观测任务和循环价值学习中比较分段与 tape 训练，并研究观测敏感度、样本效率及运行时间。

#### 条件与限制

并行 scan 和长序列反传使用保存的序列与批处理资源，不属于严格逐步、每条经验只使用一次的 RTRL。结合结构也不使任意非线性 RNN 都能采用同样的 scan。

#### 阅读与实验

固定同一种记忆模型，对照截断长度、完整回合和流式在线导数；分别检查活动能记多久、梯度能传多久、持久内存与训练峰值内存。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://papers.nips.cc/paper_files/paper/2024/file/19f7f755908372efb25826d61959cdf9-Paper-Conference.pdf)：结合运算、inline reset、Tape-Based Batching 与实验。
- [作者公开版本](https://arxiv.org/html/2402.09900v3)：附录给出不同递归模型与回报的 memoroid 写法。

#### 作者代码

[论文附录原链接 memory-monoids 对应作者 Prorok Lab 的现有 memoroids 仓库；README 标明论文。](https://github.com/proroklab/memoroids)

memory 模型、buffer、losses 与 segment_dqn/tape_dqn 对照。

### Expected Eligibility Traces

Hado van Hasselt, Sephora Madjiheurem, Matteo Hessel, David Silver, André Barreto, Diana Borsa

AAAI 2021（2020预印本） · 2021 · 支持方法与理论

#### 研究问题

当前误差能否同时更新本次未走过、但也可能到达当前状态的过去路径？

#### 关键机制

学习给定当前状态的资格迹条件均值，再用当前TD误差更新该均值所指向的过去预测。递归混合在实际轨迹迹与预测的期望迹之间插值；预测对象是过去资格，而非未来奖励。

#### 证据

原文在Markov状态与相应条件下证明更新均值相同、逐分量方差不增，并在路径汇合问题检验预测效率。信用章精确枚举一个正例和一个状态混叠反例。

#### 条件与限制

不完整观察、参数漂移和近似迹预测器会破坏无偏条件。全参数期望迹预测还有输出维度和计算成本；小实验不复现作者的神经实验。

#### 阅读与实验

保持奖励边际分布一致，仅改变奖励是否依赖隐藏的过去路径。先测信用均值与方差，再研究agent state能否恢复条件独立。

#### 原文与相关入口

- [作者原文](https://arxiv.org/html/2007.01839)：Lemma 1、Proposition 1及ET(λ,η)递归混合。
- [AAAI发表版本](https://ojs.aaai.org/index.php/AAAI/article/view/17200)：正式会议年份为2021。

### Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning

Noah Farr, Aryaman Reddi, Carlo D’Eramo, Jan Peters

arXiv预印本（2026-07-07 v2） · 2026 · 支持方法与理论

#### 研究问题

严格逐步更新的智能体怎样同时学习递归记忆、分配延迟信用并控制计算？

#### 关键机制

将RTU结构的RTRL敏感度接入QRC与流式actor–critic。敏感度给出当前输出对记忆参数的导数，资格迹再组合过去输出的回报信用；两条递推保持分工。

#### 证据

v2在MemoryChain、五项POPGym和masked MuJoCo上报告5-seed结果，另用KMemoryChain比较在线敏感度与当前参数重算参考，并检验Taylor修正。

#### 条件与限制

masked MuJoCo仍落后批量PPO。固定参数精确RTRL不代表在线变参敏感度始终等于当前参数重算；诊断保存整个episode，须计为额外评价资源。尚未确认作者公开代码。

#### 阅读与实验

在相同递归容量下独立改变记忆跨度、回报λ与参数步幅，同时测敏感度误差和回报。诊断改善不能单独当作控制改进证据。

#### 原文与相关入口

- [2026年v2原文](https://arxiv.org/html/2605.24709v2)：方法、5-seed实验、masked MuJoCo负边界与staleness诊断。

### Does Zero-Shot Reinforcement Learning Exist?

Ahmed Touati, Jérémy Rapin, Yann Ollivier

ICLR 2023 · 2023 · 支持方法与理论

#### 研究问题

没有事先指定奖励时，怎样学一套预测表示，日后接收新奖励就能选行为？

#### 关键机制

Forward–Backward 表示联合学习行为条件的未来占用与奖励读出，而非先固定任意编码器再学习 successor features。新奖励被映射到任务向量，策略根据这个向量直接行动；该论文系统比较 FB 与多种 SF 基础特征。

#### 证据

原文在固定离线 replay buffers 上比较零样本任务迁移，借此把表示学习与探索数据的质量分开。不同特征与数据覆盖产生显著差异，不能仅靠“所有奖励”的理论目标预测实际效果。

#### 条件与限制

假定共享动力学与可用经验覆盖。无下游梯度更新不等于无预训练成本；有限秩、近似训练和奖励估计都有误差。新动力学、历史混叠和严格一次使用经验均须另测。

#### 阅读与实验

同一 buffer 对比随机特征、谱特征与联合 FB，再独立换 buffer。奖励读出误差、占用误差与新任务回报分别报告，避免把数据覆盖优势记成表示优势。

#### 原文与相关入口

- [作者原文](https://arxiv.org/abs/2209.14935)：2022 首稿，ICLR 2023；比较奖励表示、SF 与 FB。
- [作者研究平台](https://github.com/facebookresearch/controllable_agent)：README 直接关联两篇 FB 论文；该仓库已经归档。

#### 作者代码

[论文作者团队仓库；归档工程，依赖和旧环境需单独核验。](https://github.com/facebookresearch/controllable_agent)

FB 与 SF 的训练、固定数据实验及奖励查询示例。

### Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations

Haosen Shi, Jianda Chen, Sinno Jialin Pan

ICLR 2026 · 2026 · 支持方法与理论

#### 研究问题

能否直接学习多步未来状态的分布，并把它压缩为适合控制学习的特征？

#### 关键机制

SF² 以 flow matching 估计 successor measure，将条件向量场分解为未来位置及生成时间的投影与当前状态动作特征的乘积。特征进入 TD3/SAC 的 critic；线性的是向量场对条件特征的分解，critic 本身可以非线性。

#### 证据

正式原文给出 mixture Bellman 结构、生成式 bootstrap 与控制实验，并提供作者 JAX/Brax 仓库。实验研究在线收集数据下的 off-policy 控制，并使用 replay、批次与目标网络。

#### 条件与限制

“online policy learning”不代表 strict streaming。生成时间不是环境时间；向量场线性不保证任意奖励价值线性。文中与 SR 的小生成时间联系是近似动机，未证明递归 agent state 或任意持续变化下的充分性。

#### 阅读与实验

对齐模型调用与梯度预算，拆分直接预测、bootstrap、critic 联合训练。冻结特征后比较线性与非线性读出，再测新奖励和动力学变化，才能检验预测知识的可复用程度。

#### 原文与相关入口

- [ICLR 2026 正式原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48acf4b231771e693f42305b4c9b4c9f-Abstract-Conference.html)：第 2–3 节和算法附录；区分 flow 时间、环境时间与近似 SR 联系。
- [原文链接的作者实现](https://github.com/Shiien/successor-flow-representation-implementation)：SAC/TD3、flow 特征、对照和 sweep 配置。

#### 作者代码

[正式论文摘要直接链接的作者代码。](https://github.com/Shiien/successor-flow-representation-implementation)

基于 JAX/Brax 的 SF² 控制实验；不包含自动 GVF 问题发现或完整持续架构。

### Foundation Policies with Hilbert Representations

Seohong Park, Tobias Kreiman, Sergey Levine

ICML 2024 · 2024 · 支持方法与理论

#### 研究问题

如何从无任务标签的离线轨迹形成既能按方向调用、又能用于目标任务的策略接口？

#### 关键机制

HILP 先学习近似保存时间距离的 Hilbert 表示，再以潜在位移与方向的内积训练方向条件策略。新任务通过奖励回归、目标方向或分层调用选择策略条件，结构表示也支持测试时规划。

#### 证据

ICML 原文与作者项目包含零样本 RL、离线目标条件 RL 及规划实验；官方仓库将 zero-shot 与 goal-conditioned 两套实现分开。

#### 条件与限制

精确时间距离不总能无损嵌入有限维对称欧氏距离，尤其有向不可逆行为；理论充分条件与近似神经实验需区分。方向条件策略没有自动获得任意停止条件或完整技能后果模型。

#### 阅读与实验

固定离线数据分别测距离误差、方向执行误差、奖励可表达误差与高层收益。让同一视觉观测对应不同历史，检查仅观测编码是否足够，之后再讨论 CRL 状态维护。

#### 原文与相关入口

- [ICML 2024 原文](https://proceedings.mlr.press/v235/park24g.html)：Hilbert 距离、策略提示和定理前提；不是 ICLR 论文。
- [作者项目与公式](https://seohong.me/projects/hilp/)：时间距离与方向奖励接口。
- [官方实现](https://github.com/seohongpark/HILP)：hilp_zsrl 与 hilp_gcrl 对应不同实验。

#### 作者代码

[作者项目直接链接并标为 official implementation。](https://github.com/seohongpark/HILP)

离线预训练、零样本奖励适配及目标条件实验。

### DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning

Gaoyue Zhou, Hengkai Pan, Yann LeCun, Lerrel Pinto

ICML 2025 · 2025 · 支持方法与理论

#### 研究问题

预训练视觉表示能否直接成为动作后果预测与目标规划的接口？

#### 关键机制

冻结 DINOv2 空间 patch 特征，用离线动作轨迹学习未来特征预测器；测试时优化动作序列，让预测特征接近目标图像特征。没有重建图像、奖励模型或逆模型，不表示没有动作条件的动力学训练。

#### 证据

ICML 原文在六类环境检验视觉目标规划，作者仓库公开数据、部分检查点、训练与 CEM 规划入口。零样本指给定已训练模型后解决目标，无额外任务策略训练。

#### 条件与限制

依赖视觉预训练与离线交互覆盖；patch 相近不总等于任务完成或风险相同。原实验不证明冻结视觉表示能适应长期新物体、新动作语义或隐藏状态。

#### 阅读与实验

分别改变背景、物体属性、控制动力学与目标分布。把冻结 encoder 和联合更新 encoder 分开，对照视觉距离、真实成功与模型误差，观察表示漂移的依赖成本。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/zhou25t.html)：正式发表入口；早期 ICLR 投稿页不能替代此状态。
- [作者项目代码](https://github.com/gaoyuezhou/dino_wm)：train.py、plan.py、数据与已公开模型检查点范围。

#### 作者代码

[原作者 Gaoyue Zhou 的论文配套仓库。](https://github.com/gaoyuezhou/dino_wm)

DINO 特征预测、离线环境数据与目标规划；README 公开部分环境检查点。

### V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, Nicolas Ballas

arXiv 预印本（此处采用 2025 首稿） · 2025 · 支持方法与理论

#### 研究问题

无动作标注的视频预训练，与能接受机器人动作的规划模型之间还缺哪一步？

#### 关键机制

V-JEPA 2 先学被遮蔽视频的潜在特征预测；V-JEPA 2-AC 冻结编码器，再用机器人轨迹训练动作条件预测器。控制以目标图像的特征差为代价进行 MPC；视频理解、动作条件预测和真实控制是三个独立证据层。

#### 证据

2025 首稿报告以大规模视频预训练，再用不到 62 小时 DROID 交互视频后训练，在两个实验室以图像目标做真实机器人规划。论文单独讨论相机位置、长程规划与图像目标的局限。

#### 条件与限制

无任务奖励并不等于无动作、无机器人状态或无外部数据。零样本部署未持续更新模型，也未发现和维护 options。官方仓库现含 V-JEPA 2.1，复现首稿须记录配置和模型版本。

#### 阅读与实验

按视觉编码、动作坐标、后果模型、目标代价逐项做迁移检验。若引入在线更新，记录模型更新使旧目标接口失效的程度，测未来交互收益，而非仅用 frozen probe 证明 CRL。

#### 原文与相关入口

- [2025 首稿](https://arxiv.org/abs/2506.09985v1)：action-free 预训练、2-AC 后训练、真实规划与第 4.3 节限制；此处不赋予未核实会议状态。
- [Meta FAIR 官方实现](https://github.com/facebookresearch/vjepa2)：包含 V-JEPA 2、2-AC 和较新的 2.1；版本不能混用。

#### 作者代码

[Meta FAIR 官方仓库；首稿模型与后续版本需按配置区分。](https://github.com/facebookresearch/vjepa2)

官方视频表征与动作条件模型；数据、机器人部署条件与检查点分别核验。


<a id="chapter-code"></a>

## 下载与运行

Bayes 过滤、标量 RTRL/BPTT/TBPTT、重置式监督提示实验、单个线性 RTU 风格旋转块；不是原论文 RL 基准复现。

[下载 state_meta_lab.py](https://yingwen.io/zh/continual-rl/download/state_meta_lab.py)

```sh
python3 state_meta_lab.py state
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Williams & Zipser：RTRL 原始论文](https://doi.org/10.1162/neco.1989.1.2.270)：理解固定参数下的全历史敏感度；本页给出独立的标量推导与实现。

- [Williams & Zipser：递归网络梯度与复杂度](https://web.stanford.edu/class/psych209a/ReadingsByDate/02_25/Williams%20Zipser95RecNets.pdf)：补读精确求导与持续在线权重变化之间的区别。

- [Littman、Sutton、Singh：Predictive Representations of State](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf)：原始 PSR 研究；关注 core tests 与条件更新，不把任意预测集合都称作充分状态。

- [Abel、Hershkowitz、Littman：Near Optimal Behavior via Approximate State Abstraction](https://proceedings.mlr.press/v48/abel16.html)：按保留的价值、模型和行为性质区分抽象，分析近似抽象造成的控制损失。

- [Schlegel 等：General Value Function Networks](https://arxiv.org/abs/1807.06763)：预测坐标约束、Bellman 网络目标、递归训练与截断敏感性。

- [GVFN 作者仓库](https://github.com/mkschleg/GVFN)：Julia 实验代码，按预测问题、递归网络、学习更新三部分恢复原实验。

- [Elelimy 等：Real-Time Recurrent Learning using Trace Units（NeurIPS 2024）](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e616bde0438cb10cb6adf076ae7d336-Paper-Conference.pdf)：线性/非线性 RTU 的结构、参数化与在线敏感度；结合作者配置辨识控制协议。

- [RTU 作者源码：linear_rtus.py](https://github.com/esraaelelimy/rtus/blob/be54e13b91edcd7988dd1764f8f2d412ca2db856/src/nets/rtus/linear_rtus.py)：固定版本入口：活动与梯度 carry、自定义 VJP、输入投影敏感度。

- [Hochreiter & Schmidhuber：Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735)：门控长期记忆的原始工作；本页采用常用现代门控写法，不把全部工程变体归于原式。

- [Cho 等：Learning Phrase Representations using RNN Encoder–Decoder](https://aclanthology.org/D14-1179/)：GRU 的原始来源之一，给出更新门与候选状态的定义。

- [Tallec & Ollivier：Unbiased Online Recurrent Optimization（ICLR 2018）](https://openreview.net/pdf?id=rJQDjk-0b)：随机低秩敏感度、在线无偏近似及其条件；原论文附有实现材料。

- [Cooijmans & Martens：On the Variance of UORO](https://arxiv.org/abs/1902.02405)：理解低秩无偏估计的方差；无偏性与有限预算训练效果需要分别讨论。

- [Orvieto 等：Resurrecting Recurrent Neural Networks for Long Sequences](https://arxiv.org/abs/2303.06349)：LRU 的线性/对角递归、初始化与尺度设计；长序列建模结果不是 streaming RL 的直接证据。

- [RTU 原始实验与配置](https://github.com/esraaelelimy/rtus)：网络、实时 actor–critic/PPO、POPGym 与部分可观测 Brax 的入口和配置。

- [NeurIPS 2024 原文](https://papers.nips.cc/paper_files/paper/2024/file/19f7f755908372efb25826d61959cdf9-Paper-Conference.pdf)：结合运算、inline reset、Tape-Based Batching 与实验。

- [作者公开版本](https://arxiv.org/html/2402.09900v3)：附录给出不同递归模型与回报的 memoroid 写法。

- [Recurrent Reinforcement Learning with Memoroids · 作者实现](https://github.com/proroklab/memoroids)：memory 模型、buffer、losses 与 segment_dqn/tape_dqn 对照。 论文附录原链接 memory-monoids 对应作者 Prorok Lab 的现有 memoroids 仓库；README 标明论文。

- [作者原文](https://arxiv.org/abs/2209.14935)：2022 首稿，ICLR 2023；比较奖励表示、SF 与 FB。

- [作者研究平台](https://github.com/facebookresearch/controllable_agent)：README 直接关联两篇 FB 论文；该仓库已经归档。

- [ICLR 2026 正式原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48acf4b231771e693f42305b4c9b4c9f-Abstract-Conference.html)：第 2–3 节和算法附录；区分 flow 时间、环境时间与近似 SR 联系。

- [原文链接的作者实现](https://github.com/Shiien/successor-flow-representation-implementation)：SAC/TD3、flow 特征、对照和 sweep 配置。

- [ICML 2024 原文](https://proceedings.mlr.press/v235/park24g.html)：Hilbert 距离、策略提示和定理前提；不是 ICLR 论文。

- [作者项目与公式](https://seohong.me/projects/hilp/)：时间距离与方向奖励接口。

- [官方实现](https://github.com/seohongpark/HILP)：hilp_zsrl 与 hilp_gcrl 对应不同实验。

- [ICML 2025 原文](https://proceedings.mlr.press/v267/zhou25t.html)：正式发表入口；早期 ICLR 投稿页不能替代此状态。

- [作者项目代码](https://github.com/gaoyuezhou/dino_wm)：train.py、plan.py、数据与已公开模型检查点范围。

- [2025 首稿](https://arxiv.org/abs/2506.09985v1)：action-free 预训练、2-AC 后训练、真实规划与第 4.3 节限制；此处不赋予未核实会议状态。

- [Meta FAIR 官方实现](https://github.com/facebookresearch/vjepa2)：包含 V-JEPA 2、2-AC 和较新的 2.1；版本不能混用。


---

# 价值预测与时间差分学习

价值预测估计给定策略的期望回报。回报的定义确定预测对象，MC、TD 和多步方法则提供不同的估计方式。本章从回报递推式推导这些方法，并讨论资格迹怎样分配时间上的信用。

## 本章内容

- 从 return 逐行推到 Bellman 方程，辨认模型期望和经验采样。
- 独立实现 MC 与 TD，解释为什么同一条轨迹给出的第一次更新不同。
- 理解 $\lambda$ 在传播信用中做什么，以及为什么普通在线 TD($\lambda$) 与固定参数前向视图不能无条件画等号。

<a id="problem-definition"></a>

## 本章的问题定义

环境与目标策略固定，询问按该策略行动的期望回报；本章基础数据也由该策略产生。

### 给定条件与符号

- Markov状态、固定目标策略、奖励与延续/终止定义。
- 经验流、价值表示类及更新预算；已知模型是DP的额外权限。

### 需要求解的对象

指定策略的价值函数及在给定表示下的估计；策略本身不是本章待学对象。

### 信息与数据权限

在 $S_t$ 按 $\pi$ 选择 $A_t$，得到 $R_{t+1},S_{t+1}$；估计 $V_t$ 只能使用已经收到的经验。

$$
v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s],\qquad G_t=R_{t+1}+\gamma_{t+1}G_{t+1}
$$

$\pi$ 是给定策略，$G_t$ 是随机回报，$\gamma_{t+1}$ 是当前转移后的延续因子；真实终止为0。$V_t$ 是估计而不是答案本身；TD平方误差与真实价值误差不相等。

### 成立条件与解的含义

- 基础设定为有限、固定MDP、有界奖励、非终止处固定折扣小于1；无折扣终止问题需另给可积终止条件。
- 表格收敛还需访问与步长条件；共享非线性表示、离策略及不断漂移不自动继承该结论。

判断准则：两步链上预测趋近起点0.9、后继1；一般任务以独立回报或解析解测价值误差，训练样本TD误差无需逐条为零。

### 适用边界

- 不以对动作取最大值替代给定策略的动作平均。
- 不以训练TD损失下降宣称策略收益改善。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)：本章将累计信号限定为任务奖励，并采用普通折扣/终止规则，因此是更一般GVF预测规格的特例。

- 组合不同学习问题 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：预测可以评价一个候选策略；控制另需策略改善和行为数据更新。

- 组合不同学习问题 · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)：多步和资格迹决定同一预测问题的反馈如何作用于过去，不改变给定策略的题目。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

完整回报尚未到来；一步自举可立即更新，却依赖当前后续价值估计。

### 本章的核心思路

从同一回报递推式选择模型期望、完整样本或一步自举；比较的是估计方法而非三个任务。

1. [从回报推到条件期望](#lesson-derive)：因为预测对象是指定行为的未来，Bellman方程按目标策略平均动作而不取最大值。

2. [选择等待长度与信用路径](#value-traces)：因为短目标依赖估计、长目标等待更多数据，多步混合与资格迹在同一对象下改变传播。

3. [以解析值检查完整循环](#lesson-code)：因为一次正确梯度不保证整个时序正确，真实终止尾值、旧参数和访问更新一并用两步链验证。

结论与条件：正确有限折扣模型的Bellman算子收缩；表格MC/TD还需各自采样与步长条件。普通在线迹不与冻结前向视图无条件精确等价。

### 相关方法改变了什么

- DP：已知后果模型时计算条件期望，需要模型权限。

- MC：用完整回报采样，减少自举依赖但等待结果。

- TD与多步：用后继估计补足未来，改变目标偏差、方差与反馈延迟。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 一条经验

在 $S_{t}$ 做 $A_{t}$ 后得到 $R_{t+1}$、$S_{t+1}$。奖励属于这次转移，不是到达状态之前另一轮的奖励。

### 折扣与终止

$\gamma$ 控制未来相对权重；真实终止后的价值设为 0。本章记 $\gamma_{t+1}$=$\gamma$(1−terminated)。训练脚本的时间截断不一定是任务终止。

### 表格与函数逼近

表格每个状态或状态动作一组独立参数；函数逼近让不同输入共享参数。更新一个输入可能改变其他输入的预测。

<a id="lesson-setting"></a>

## 1 · 给定策略的预测问题

本章固定策略 $\pi(a\mid s)$，估计按它持续行动的后果，不改变动作选择规则。假设状态与动作有限、转移规律固定、奖励有界且 $0\le\gamma<1$。数据由目标策略产生。若当前观测不足以条件化未来，需要先引入历史或适当的状态表示；不能直接将观测当作 Markov 状态。

价值预测与控制使用相同的回报定义，但求解的对象不同。预测以策略为输入，以价值为答案；控制根据价值或其他后果估计改善策略。先独立理解预测，可以分清一个控制算法的问题来自评价误差、决策规则，还是收集到的数据。通用价值函数则进一步扩展预测的信号和延续规则，仍不自动承担策略改善。

$$
G_t=R_{t+1}+\gamma_{t+1}G_{t+1},\qquad v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s]
$$

$G_t$ 是一次实现的随机回报；$v_\pi(s)$ 是同一起点、同一策略下回报的条件期望。真实终止令后继延续因子为零，只删掉未来项，不删掉进入终点时的奖励。

算法维护估计 $V$；真实价值通常未知。解析小 MDP 可以直接计算价值误差；一般任务可用独立采样回报进行评估。训练中的 TD error 与价值估计误差是不同的量。

<a id="lesson-derive"></a>

## 2 · 从回报拆分到 Bellman，再到三种更新

$$
v_\pi(s)=\sum_a\pi(a\mid s)\sum_{s\prime,r}p(s\prime,r\mid s,a)[r+\gamma(s,a,s\prime)v_\pi(s\prime)]
$$

第一步把 G 拆成当前奖励和未来；第二步对动作与转移取条件期望；Markov 性允许用下一状态而非完整历史描述后半段。这里尚未提出任何学习算法。

$$
V_{k+1}(s)=(T_\pi V_k)(s),\qquad \|T_\pi u-T_\pi v\|_\infty\le\gamma\|u-v\|_\infty
$$

已知模型时，可以对所有后果求和。这是动态规划的期望备份；$\gamma$<1 带来压缩性。在正确表格模型下重复备份趋向唯一固定点。$\gamma$=1 的终止任务需要另一套适当终止条件。

$$
V(S_t)\leftarrow V(S_t)+\alpha[G_t-V(S_t)]
$$

MC 用完整实际结果作为监督目标。需等到回报可计算；对固定策略、可积回报，它直接对正确条件期望采样。环境中途不断变化、轨迹没有结束或非常长时，等待完整回报代价很大。

$$
\delta_t=R_{t+1}+\gamma_{t+1}V_t(S_{t+1})-V_t(S_t),\qquad V_{t+1}(S_t)=V_t(S_t)+\alpha_t\delta_t
$$

TD(0) 只等一步，把剩下的未来交给当前估计。单个 target 通常不是对真实 v 的无偏样本，因为 V 还不准；但正确条件下长期固定点可以正确。因此，需要分析其长期固定点，而不只分析单次目标的偏差。

| 方法 | 需要什么 | 一条转移能否立即更新 | 误差从哪里来 |
| --- | --- | --- | --- |
| DP | 转移与奖励模型 | 可以，但使用模型期望 | 模型误差、备份不足 |
| MC | 完整采样回报 | 通常不能 | 有限样本方差 |
| TD | 一步经验与旧价值 | 可以 | 自举估计、有限样本、表示限制 |

<a id="value-traces"></a>

## 3 · 多步与 $\lambda$：奖励应该传回多远？

$$
G_t^{(n)}=\sum_{k=1}^n\gamma^{k-1}R_{t+k}+\gamma^n V(S_{t+n})
$$

终止时截到终点。n 小更依赖价值估计，n 大使用更多实际结果，也等待更久、通常有更大方差。具体的误差权衡取决于奖励噪声、价值估计和轨迹长度。

$$
G_t^\lambda=(1-\lambda)\sum_{n\ge1}\lambda^{n-1}G_t^{(n)},\qquad G_t^\lambda-V(S_t)=\sum_{k\ge0}(\gamma\lambda)^k\delta_{t+k}
$$

在无限折扣轨迹、固定 V 下，展开 n-step，按相同奖励与 V 项收集，价值项逐项抵消，就得到右边的 TD error 加权和。有限终止轨迹的最后一项吸收剩余权重。

$$
e_t=\gamma\lambda e_{t-1}+x_t,\qquad w_{t+1}=w_t+\alpha\delta_t e_t
$$

线性 $V=w^Tx$ 时，把未来误差回传改写为记录过去特征。e 是“哪些参数对近期预测有影响”的信用痕迹，不是能在动作选择时回忆往事的隐状态。表格取 one-hot 特征。

这一步前向/后向等价的推导固定了参数。普通在线 TD($\lambda$) 每步都改参数，有限步长下不能声称严格等价于该固定参数目标；true-online TD($\lambda$) 用 Dutch trace 和额外修正实现相应在线前向视图。离策略时还要处理目标策略与行为策略的差别，相应修正将在通用价值函数一章中推导。

<a id="lesson-example"></a>

## 4 · 两步轨迹：为什么 MC 比 TD 更早改动起点？

A --奖励 0→ B --奖励 1→终点。$\gamma$=.9，初始 V(A)=V(B)=0，$\alpha$=.1。真实值为 $v_\pi(A)=0.9$、$v_\pi(B)=1$。

| 第一次经历 | A 的 target / 更新后值 | B 的 target / 更新后值 |
| --- | --- | --- |
| MC | 完整回报 .9 → .09 | 完整回报 1 → .1 |
| TD(0) | 0+.9×0=0 → 0 | 1+0−0 → .1 |
| TD($\lambda$=.8) | 第二步迹 .9×.8=.72 → .072 | 第二步当前特征迹 1 → .1 |

第二回合 TD 在 A 的 target 已是 .9×.1=.09，因此 V(A)=.009。延迟奖励通过多轮自举逐渐传播到起点。因此，MC 与 TD 即使具有相同的极限值，有限数据下的学习过程也不同。

<a id="lesson-code"></a>

## 5 · 核心实现与一次完整训练循环

**算法：线性、on-policy、回合制的 accumulating TD(λ)**

1. 初始化 $w$；给定步长 $\alpha$、迹参数 $\lambda$ 和特征函数 $x(s)$。
1. 每个回合开始时令 $e=0$，观察起始状态 $S$。
1. 每次转移：
  1. 按目标策略执行动作，观察 $R,S'$。
  1. 若真实终止，令 $\gamma'=0$；否则令 $\gamma'=\gamma$。
  1. $\delta\leftarrow R+\gamma'w^\top x(S')-w^\top x(S)$
  1. $e\leftarrow\gamma\lambda e+x(S)$
  1. $w\leftarrow w+\alpha\delta e$
  1. $S\leftarrow S'$；真实终止时结束本回合。

相同环境与步长，分别用整段结果和一步自举

```python
def prediction(method="td", episodes=200, alpha=0.1):
    """A --0--> B --1--> terminal; gamma=.9, exact values [.9, 1]."""
    values = [0.0, 0.0, 0.0]
    trajectory = [(0, 0.0, 1, 0.9), (1, 1.0, 2, 0.0)]
    for _ in range(episodes):
        if method == "mc":
            ret = 0.0
            for state, reward, _, discount in reversed(trajectory):
                ret = reward + discount * ret
                values[state] += alpha * (ret - values[state])
        elif method == "td":
            for state, reward, nxt, discount in trajectory:
                delta = reward + discount * values[nxt] - values[state]
                values[state] += alpha * delta
        else:
            raise ValueError("Choose mc or td")
    return values[:2]
```

- MC 从终点向前计算 return；本例每个状态每回合仅访问一次，因此 first-visit 与 every-visit 没有差别。一般轨迹中须明确选择。
- TD 在收到下一状态后立刻更新，后继值使用更新前可用的估计。终点值为 0。
- 运行 value 后两者应趋近 [.9, 1]。将 `episodes` 设为 1，可以比较首次更新与本节的数值计算。

<a id="lesson-branches"></a>

## 6 · 向持续预测的推广

| 分支 | 实际改变 | 需要继续检查 |
| --- | --- | --- |
| 常数步长跟踪 | 让新数据持续改变估计 | 方差、变化速度、访问频率 |
| GVF | 把任务奖励改为任意累积信号，并明确策略与延续 | 预测题目是否定义正确 |
| GTD / emphatic TD | 改变离策略函数逼近的更新几何或加权 | 覆盖、方差、线性理论条件 |
| 平均奖励 TD | 去掉长期增长的奖励率，学习差分价值 | 奖励率与价值的联合估计 |
| 神经网络 TD | 用共享非线性表示 | 自举、离策略、函数逼近的耦合 |

常数步长不追求在静止问题里把噪声彻底平均掉，而是保留对新规律的响应。有效样本权重随年龄呈指数衰减；真正的变化检测、滑动窗口选择和历史情境复用则是进一步的问题，不是 TD 自动具备的功能。

<a id="lesson-check"></a>

## 7 · 习题与讨论

问题：学习 TD 时把 target 中的下一状态价值也求导，会更“完整”吗？不一定。它改变了算法：TD 的半梯度更新是固定 target 后的回归方向；完整样本残差梯度优化另一目标。对期望 Bellman 残差平方求无偏梯度还涉及双采样。先写目标，再判断哪个梯度正确。

问题：为什么训练 TD error 不为零仍可能已经学对？随机奖励和随机转移会带来不可约的单样本误差；正确价值满足条件期望误差为零，不要求每条样本误差都零。用已知解析值或独立 rollout 估值，不能只看训练 loss。

## 本章的实验设计

在小型马尔可夫奖励过程中计算解析价值。再检查样本更新和参数误差。TD 误差不能代替价值误差。

设定：两状态奖励过程：A 无奖励到 B，B 获得 1 后回 A，γ=0.5。固定策略的价值为 v(A)=2/3、v(B)=4/3。

- 解析解满足两个 Bellman 方程。
- 零步长不写入，γ=0 只拟合下一奖励。
- 终端样本不读取无效后继值。

对照：MC、TD(0) 与指定版本 TD(λ)；相同轨迹与独立交互两种面板；固定表征及共同步长搜索预算

记录：对解析价值的均方误差；Bellman 残差与样本 TD 误差分别记录；固定数据量下的误差曲线

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-classic)

## 学习与研究衔接

表格 TD 的局部更新推广为共享特征上的参数更新。GVF 再改变预测问题，而非只改变网络。

[分册导读](https://yingwen.io/zh/continual-rl/start/classic-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-value) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=value) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=value)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

状态是支持后续计算的内部信息；GVF 指定一个预测问题；RTRL 和资格迹规定信用如何传播。三者可以组合，但不是相互替代的算法名称。先理解给定策略的预测，再讨论预测怎样改善控制。

- [When does Self-Prediction help? Understanding Auxiliary Tasks in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-self-prediction-auxiliary-tasks)

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)
- [Safe and Efficient Off-Policy Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-retrace-safe-offpolicy)
- [Convergent Tree Backup and Retrace with Function Approximation](https://yingwen.io/zh/continual-rl/research/#recent-convergent-tree-retrace)
- [Multi-Step Reinforcement Learning: A Unifying Algorithm](https://yingwen.io/zh/continual-rl/research/#recent-q-sigma-backups)
- [A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning](https://yingwen.io/zh/continual-rl/research/#recent-lambda-greedy)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [Reward Centering](https://yingwen.io/zh/continual-rl/research/#recent-reward-centering-discounted)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning](https://yingwen.io/zh/continual-rl/research/#recent-lambda-greedy)

### When does Self-Prediction help? Understanding Auxiliary Tasks in Reinforcement Learning

Claas A. Voelcker, Tyler Kastner, Igor Gilitschenski, Amir-massoud Farahmand

RLC 2024 / RLJ · 2024 · 支持方法与理论

#### 研究问题

预测下一潜在状态、重建观测和学习价值，为什么会产生不同的表示？

#### 关键机制

论文在含干扰因素的线性问题中分析辅助目标的学习动力学。潜在状态自预测与价值学习共同作用时可能保留决策相关结构，但单独训练同一目标未必得到最有用的特征。目标的作用取决于它和 TD 目标怎样共享表示。

#### 证据

线性分析给出可检查的条件，并用神经网络实验检验部分预测。结果不支持“任何自监督预测都能改善 RL”这种无条件判断。

#### 条件与限制

线性分析中的观测映射、优化过程与神经网络控制并不完全等价。项目仓库入口不等于已经提供完整可复现实验实现，因此这里不列为可运行代码。

#### 阅读与实验

固定编码器容量，分别比较仅 TD、仅辅助任务和联合训练。记录价值误差与任务收益，不要只用辅助损失下降评价表示。

#### 原文与相关入口

- [RLC 2024 论文入口](https://rlj.cs.umass.edu/2024/papers/Paper197.html)：原文、线性假设与神经网络实验。
- [作者预印本](https://arxiv.org/abs/2406.17718)：便于追踪论文版本。

### Deep Reinforcement Learning with Gradient Eligibility Traces

Esraa Elelimy, Brett Daley, Andrew Patterson, Marlos C. Machado, Adam White, Martha White

RLC 2025 / RLJ · 2025 · 支持方法与理论

#### 研究问题

资格迹怎样与明确的梯度目标结合，而不是直接把线性半梯度规则搬到深度网络？

#### 关键机制

论文从广义投影 Bellman 误差出发构造多步目标，推导带资格迹的梯度学习方法。前向视角连接多步回报与经验重放，后向视角通过递推迹分配信用。目标函数、辅助估计器和迹的更新共同决定算法，不只是选择一个较大的 λ。

#### 证据

作者给出多种算法并在 MuJoCo、MinAtar 等任务中比较。代码同时提供相关梯度算法与实验设置，可以把推导中的量映射到实际更新。

#### 条件与限制

线性 GTD 的收敛条件不能自动赋予非线性实现全局收敛保证。重放版本与流式版本的数据使用预算也不能混为一谈。

#### 阅读与实验

从一段短轨迹分别计算前向多步目标和后向迹。随后对照原代码检查辅助网络、目标与主网络参数使用的是更新前还是更新后的值。

#### 原文与相关入口

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_302.pdf)：目标、算法推导与实验。
- [作者算法库](https://github.com/esraaelelimy/gtd_algos)：论文提供的梯度 TD 与资格迹实现。

#### 作者代码

[原论文链接的作者仓库。](https://github.com/esraaelelimy/gtd_algos)

论文梯度算法、资格迹和实验配置。

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

### Safe and Efficient Off-Policy Reinforcement Learning

Rémi Munos, Tom Stepleton, Anna Harutyunyan, Marc G. Bellemare

NeurIPS 2016 · 2016 · 支持方法与理论

#### 研究问题

目标与行为策略不一致时，如何保留多步信用而避免重要性比率乘积爆炸？

#### 关键机制

统一多步目标为目标策略TD误差的加权和，Retrace采用λmin(1,π/μ)传播系数。近同策略时保留长迹，目标概率较低的动作则减少传播；一步误差仍使用目标动作期望。

#### 证据

论文分析表格算子的收缩性质，给出条件下的评价与控制收敛，并报告Atari实验。信用章独立检查传播系数和有限轨迹恒等式。

#### 条件与限制

表格安全性不是任意线性或神经逼近的稳定性保证。行为覆盖、变化策略与投影条件仍需检查；代码小实验不复现Atari。

#### 阅读与实验

在同样轨迹与表示上，分别改变策略差异和动作随机性，比较Tree-backup与Retrace的信用长度、方差和预测误差。

#### 原文与相关入口

- [原论文](https://arxiv.org/html/1606.02647)：统一算子、传播系数及理论条件。

### Convergent Tree Backup and Retrace with Function Approximation

Ahmed Touati, Pierre-Luc Bacon, Doina Precup, Pascal Vincent

ICML 2018 · 2018 · 支持方法与理论

#### 研究问题

传播系数已经截断，为什么函数逼近下的Tree-backup和Retrace仍可能发散？

#### 关键机制

分析函数逼近与off-policy多步bootstrap的学习算子，展示线性反例，再把相应目标写成二次凸凹鞍点问题，构造梯度版本。

#### 证据

原文给出线性不稳定例子、梯度方法收敛保证与有限样本界。它直接限定了从Retrace表格结论外推到逼近算法的范围。

#### 条件与限制

凸凹线性问题的保证不能自动覆盖学习表示的深度网络。稳定目标、更新速度与控制性能还需分别验证。

#### 阅读与实验

先检查固定表示下的期望更新矩阵，再将半梯度和梯度版本按相同样本、步数与计算预算比较。

#### 原文与相关入口

- [ICML原文](https://proceedings.mlr.press/v80/touati18a.html)：理论反例、鞍点方法和保证条件。

### Multi-Step Reinforcement Learning: A Unifying Algorithm

Kristopher De Asis, J. Fernando Hernandez-Garcia, G. Zacharias Holland, Richard S. Sutton

AAAI 2018 · 2018 · 支持方法与理论

#### 研究问题

多步动作价值目标必须始终采样下一动作，或始终对动作取期望吗？

#### 关键机制

Q(σ)逐处混合Sarsa的采样动作与Expected Sarsa的动作期望，并同步改变后续误差传播。σ控制采样程度，与控制回报长度的λ不同。

#### 证据

原文给出统一n-step表达、off-policy修正和实验比较。信用章小程序核验冻结on-policy几何λ混合的两个端点。

#### 条件与限制

原文n-step和本章λ混合参考具有不同实现范围。只改一步误差却不改多步传播或策略修正，不能称为完整Q(σ)。

#### 阅读与实验

把采样噪声、目标长度和策略差异分开改变，避免把σ与λ的作用归到同一“更长信用”解释。

#### 原文与相关入口

- [原文](https://arxiv.org/html/1703.01327)：式13–15：混合误差、传播及off-policy修正。

### A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning

Martha White, Adam White

arXiv预印本 · 2016 · 支持方法与理论

#### 研究问题

不同状态的预测可靠性不同，固定λ是否浪费了多步信用？

#### 关键机制

将下一处bootstrap选择写成局部偏差平方与回报方差的折中，得到$λ=b^2/(b^2+\operatorname{Var}(G))$。完整λ-greedy还用在线预测器估计回报均值和二阶矩。

#### 证据

原文给出状态相关λ的目标、增量算法和多个预测设置的实验。信用章仅核对已知统计量下的局部最优与变量λ恒等式。

#### 条件与限制

局部贪心目标不是整条轨迹的联合最优。逼近误差、统计滞后和非平稳性会影响λ估计；辅助资源需要计入比较。

#### 阅读与实验

先让噪声方差变化，再让bootstrap可靠性变化。比较固定λ、已知统计参照和在线估计，分别观察目标偏差与适应速度。

#### 原文与相关入口

- [作者原文](https://arxiv.org/html/1607.00446)：局部目标、状态λ、均值／二阶矩预测与完整算法。


<a id="chapter-code"></a>

## 下载与运行

Python 3.10+，仅标准库。实现表格 MC/TD 完整小实验；资格迹的前后向推导与实现见时间信用分配章。

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

```sh
python3 foundations_detail_lab.py value
python3 foundations_detail_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：第 3–8 章建立 MDP、动态规划、MC、TD 与规划；第 9–13 章把这些更新扩展到函数逼近、资格迹和策略梯度。按本章的问题找相应章节，不必从头重读。

- [Sutton · Learning to Predict by the Methods of Temporal Differences](https://doi.org/10.1007/BF00115009)：TD 的原始问题动机与多步预测。读“如何利用尚未结束的经验”，并理解其更新规则。

- [van Seijen & Sutton · True Online TD($\lambda$)](https://proceedings.mlr.press/v32/seijen14.html)：检查在线前向视图、Dutch trace 与修正项；它解决的不是简单加大 $\lambda$。


---

# 通用价值函数与预测知识

通用价值函数把一个预测问题写成目标策略、累积信号与延续规则。以“沿墙行走至充电点的能耗”为例，本章从问题定义推导 Bellman 方程，再逐步构造 on-policy 与 off-policy 学习算法。

## 本章内容

- 从一个自然语言问题写出 cumulant、continuation、目标策略和状态条件，分清问题与学习器。
- 独立推导 Bellman 方程、线性 TD、资格迹、GTD2/TDC、GTD($\lambda$) 与 Emphatic TD 的更新。
- 逐行运行多问题共享经验的学习循环，检查解析解、off-policy 发散反例与实现时序。

<a id="problem-definition"></a>

## 本章的问题定义

在指定行为条件下预测某种信号的累计量，例如到充电点前的能耗；信号不必是任务奖励。

### 给定条件与符号

- 每个问题的目标策略、cumulant、延续规则和条件状态。
- 真实行为数据及行为概率；固定特征/网络类和更新预算。

### 需要求解的对象

各个给定预测题目的条件期望；稳定估计和预测发现是另外需要声明的子问题。

### 信息与数据权限

$b$ 生成动作，$\pi$ 定义假想未来行为；由已到达转移构造 $C_{t+1}$ 与 $\gamma_{t+1}$。行为支持目标动作时才可计算 $\rho_t=\pi(A_t\mid S_t)/b(A_t\mid S_t)$。

$$
v_{\pi,c,\gamma}(s)=\mathbb E_\pi\!\left[\sum_{k=0}^{\infty}\left(\prod_{j=1}^{k}\gamma_{t+j}\right)C_{t+k+1}\,\middle|\,S_t=s\right]
$$

$C$ 是由信号规则 $c$ 生成的累计信号，$\gamma$ 为转移延续因子，空乘积为1。事件上停止仍保留该步信号。线性GTD的投影Bellman目标是求解代理，未必等于最小真实预测误差。

### 成立条件与解的含义

- 分析期间环境、目标策略与表示固定，状态Markov且累计量存在；延续矩阵谱半径小于1提供唯一解条件。
- 离策略需要覆盖；GTD/ETD稳定性须满足相应线性、遍历和步长条件，不推广为任意深网定理。

判断准则：有限题目直接解线性Bellman系统核对预测与单位；在离策略反例上分开测价值误差、发散和重要性比方差。

### 适用边界

- 离策略预测稳定不等于得到全局最优控制。
- 事件终止预测不要求重置真实环境。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 推广：放宽条件 · [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)：本章将任务奖励推广为指定累计信号，并允许转移依赖的延续规则；目标策略仍须单独给定。

- 组合不同学习问题 · [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)：预测可作状态坐标，但覆盖少量问题不证明所有相关历史已被保留。

- 组合不同学习问题 · [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)：设计奖励与折扣终点信号可以预测模型输出；普通单个GVF不等于完整后果模型。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

自然语言题目容易混淆累计信号与终止计时；实际行为又可能不同于假想行为。

### 本章的核心思路

先明确问题三元组，再分开处理Bellman递推、行为纠偏与逼近稳定性。

1. [固定题目语义和计时](#gvf-semantics)：因为能耗、到达概率和折扣到达量不同，先由cumulant与延续写出累计量和Bellman方程。

2. [由真实行为估计目标行为](#gvf-offpolicy)：因为样本动作由行为策略产生，用记录概率的比率纠偏并检查支持，缺失覆盖不能靠加小常数修复。

3. [以辅助量估计投影目标方向](#gvf-gtd)：因为重要性比不保证共享线性参数稳定，GTD用辅助向量估计投影Bellman目标所需的条件量，并保留所写主/辅助更新。

4. [用强调权重处理另一种逼近](#gvf-etd)：因为状态分布也影响稳定性与逼近解，ETD递推follow-on与强调权重；它不只是替换GTD的迹，需分别检验加权固定点和方差。

结论与条件：可积且延续矩阵满足条件时题目有确定解；线性GTD/ETD的理论依赖其假设，神经递归GVF不因此自动稳定。

### 相关方法改变了什么

- 普通TD：低成本一步自举，离策略共享逼近下可能发散。

- GTD2/TDC：借助辅助量优化投影Bellman相关目标，需区分目标与更新式。

- Emphatic TD：调整历史和状态强调权重，目标权重及方差与GTD不同。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 价值函数与条件期望

$v_\pi(s)$ 是在状态 s 出发、以后按 $\pi$ 行动时，一个指定未来累计量的条件期望。改变后续行为或累计信号，就改变了问题，即使物理状态相同。

### Bootstrap 与半梯度

用当前预测的下一状态值构造训练目标叫 bootstrap。更新当前预测时，把这个目标暂时当常数叫半梯度；不是把整条 Bellman 残差对全部参数求导。

### 固定特征与线性逼近

x(s) 是给定的 d 维特征，w 是学习的 d 维参数。one-hot 特征让每个状态有自己的参数，退化为表格；一般特征会让不同状态的更新相互影响。

$$
\hat v_w(s)=w^\top x(s)
$$

### 行为策略与目标策略

b 实际选动作并产生数据；$\pi$ 是问题中假想后续采用的策略。预测“若一直向右”的后果，不要求真实机器人永远向右，但需要相应经验覆盖。

<a id="lesson-setting"></a>

## 1 · 通用价值函数的定义

General Value Function（通用价值函数）把通常只问“未来有多少奖励”的价值预测，扩展成“在某种行为条件下，未来某种信号会累计成什么”。GVF 定义预测对象，学习算法估计这个对象。二者可以分别选择：一个网络可以表示多个 GVF，同一个 GVF 则可以由表格 TD、线性 GTD 或其他预测方法学习。

本章属于预测，而不是策略改善。普通价值预测先固定任务奖励与策略；GVF 进一步允许指定其他累积信号和延续规则。两者都在回答给定行为条件下会发生什么。控制则还要规定行为的优化目标，并据预测改变动作选择。行为策略可以在实际运行中不断变化，但每个 GVF 的目标策略仍须单独定义。

| 概念 | 描述的对象 | 与 GVF 的关系 |
| --- | --- | --- |
| 预测问题 | 要估计的条件期望 | GVF 用目标策略、信号与延续给出这个问题的规格 |
| 学习算法 | 经验如何更新估计 | TD、GTD、ETD 等是求解这些预测问题的算法 |
| Agent state | 决策与预测使用的历史摘要 | 多个预测可作为其特征，但预测数量多不保证状态满足 Markov 性 |
| 环境模型 | 动作或 option 导致的奖励与后继信息 | 适当设计的一组 GVF 可表达模型的部分输出；任意一个 GVF 不等于完整转移模型 |
| 控制规则 | 根据后果选择何种行为 | 需要另行定义奖励目标、选择规则与探索机制 |

| 组成 | 要决定什么 | 充电点例子 |
| --- | --- | --- |
| 状态条件 s | 从什么信息出发提问 | 机器人当前区域和足以决策的历史摘要 |
| 目标策略 $\pi(a\mid s)$ | 后续假设采取什么行为 | 以 0.8 的概率向前，否则等待 |
| 累计信号 $C_{t+1}$ | 每次转移累计什么量 | 这一步的耗电量 |
| 延续因子 $\gamma_{t+1}\in[0,1]$ | 到达后是否继续累计，或者如何折扣 | 到充电点为 0，其他地方为 1 |
| 答案 v(s) | 给定上述规格，未来累计量的期望 | 到下次充电点的期望总耗电 |

本章先假定状态 s 是 Markov 的，环境、目标策略和特征在分析期间固定，且累计量存在。Markov 指给定当前状态和动作后，预测下一步不再需要完整历史。若只输入不充分的摄像头画面，不能直接沿用后面的精确 Bellman 方程或线性收敛结论。我们最后再讨论表示和预测问题本身不断变化的 CRL 情形。

数据接口是一条条 $(S_t,A_t,S_{t+1})$、传感器信号和行为概率 $b(A_t\mid S_t)$。一个 question 函数把每条真实经验变成自己的 $(C_{t+1},\gamma_{t+1},\pi(A_t\mid S_t))$。真实奖励可以是传感器信号之一，但不必是这个 GVF 的 cumulant。

<a id="gvf-semantics"></a>

## 2 · 累积信号、延续与预测语义

| 要问的内容 | $C_{t+1}$ | $\gamma_{t+1}$ | 答案的单位和含义 |
| --- | --- | --- | --- |
| 到充电点的步数 | 每一步为 1，包括到达那一步 | 到达为 0，否则 1 | 步；须保证目标策略下到达时间有有限期望 |
| 到充电点的能耗 | 该步耗电 | 到达为 0，否则 1 | 能量；不自动等于步数 |
| 最终是否会到达 | 到达指示为 1，否则 0 | 到达为 0，否则 1 | 首次到达概率；事件后停止计数，不能反复加 1 |
| 较近的到达有多大可能 | 到达指示 | 到达为 0，否则 0.9 | 折扣到达量 E[0.9^(T−1)1{到达}]，一般不是最终到达概率 |

$$
G_t=C_{t+1}+\gamma_{t+1}C_{t+2}+\gamma_{t+1}\gamma_{t+2}C_{t+3}+\cdots
$$

第一笔信号不折扣。$\gamma_{t+1}$ 作用于这次转移之后的余项，不作用于已经收到的 $C_{t+1}$。因此在到达事件上把 $\gamma$ 设为 0，仍会保留到达那一步的信号。

$$
G_t=\sum_{k=0}^{\infty}\left(\prod_{j=1}^{k}\gamma_{t+j}\right)C_{t+k+1},\qquad v_{\pi,c,\gamma}(s)=\mathbb E_\pi[G_t\mid S_t=s]
$$

k=0 的空乘积为 1。$\gamma$ 可以依赖转移；本页程序按到达状态计算。固定 $\gamma$<1 是熟悉的指数折扣；事件终止可以让某些地方 $\gamma$=1，也可以完全不重置真实环境。

若 C 是传感器温度而 $\gamma$=0.9，预测的是折扣温度总和，量级约等于温度的十倍，不是“十步以后的温度”。若想问第十步温度，需要重新设计预测结构/时间状态；若想近似未来平均温度，常数折扣下可考虑 $(1-\gamma)v$，但仍不是固定十步窗口平均。

旧 Horde/RLPark 接口还显式提供普通信号 r 和停止值 z，组合为 $c=r+(1-\gamma)z$。读这类源码时应先转换为统一 cumulant，再比较更新式。例如到达时 $\gamma$=0，才把 z 完整加入。不能既把 z 加进 c，又在 TD target 里重复加一次。

<a id="lesson-derive"></a>

## 3 · Bellman 方程

$$
G_t=C_{t+1}+\gamma_{t+1}G_{t+1}
$$

把无限和拆成第一项与剩下的同一类问题。这一步只用代数，还没有学习算法。

$$
v(s)=\sum_a\pi(a\mid s)\,\mathbb E[C_{t+1}+\gamma_{t+1}v(S_{t+1})\mid S_t=s,A_t=a]
$$

在状态 s 上取条件期望；Markov 条件允许用 $v(S_{t+1})$ 代替未来余项的条件期望。这里是按 $\pi$ 加权，不是对动作取 max：我们在评估一个给定行为问题，还没有优化行为。

$$
\begin{gathered}r_c(s)=\mathbb E_\pi[C_{t+1}\mid s],\quad M(s,s')=\mathbb E_\pi[\gamma_{t+1}\mathbf1\{S_{t+1}=s'\}\mid s]\\v=r_c+Mv,\qquad v=(I-M)^{-1}r_c\end{gathered}
$$

M 是已经含延续权重的转移矩阵，不一定每行和为 1。若谱半径小于 1，逆矩阵存在。统一 $\gamma\le\gamma_{\max}<1$ 是一个充分条件；事件终止时可通过转移结构满足该条件，不要求每个 $\gamma$ 都小于 1。

“解方程”适用于已知模型。本章用它给小实验提供独立正确答案。机器人通常不知道 M 和 $r_c$，因此需要从经验估计固定点。TD 的意义正是每收到一条转移就改进一次答案，而不等模型和完整回报都准备好。

$$
q(s,a)=\mathbb E[C_{t+1}+\gamma_{t+1}\sum_{a'}\pi(a'\mid S_{t+1})q(S_{t+1},a')\mid s,a]
$$

如果问题还条件于“第一步选 a”，就得到 action-value 形式；下一步仍对指定 $\pi$ 求期望。把这个期望换成 max，会把固定行为的预测问题变成控制问题。

<a id="gvf-td"></a>

## 4 · 线性 TD(0)

$$
\begin{gathered}y_t=C_{t+1}+\gamma_{t+1}w_t^\top x_{t+1},\quad\delta_t=y_t-w_t^\top x_t\\\widetilde L_t(w)=\tfrac12(y_t-w^\top x_t)^2,\qquad w_{t+1}=w_t+\alpha\delta_t x_t\end{gathered}
$$

先用旧参数得到 target，随后只对当前预测求导。$\delta$>0 说明这次 target 比预测高，要提高相关特征的权重。one-hot 时只改变当前状态的一个表项。

- 初始化 w（例如全 0）；观察当前状态并计算 $x_{t}$。
- 按行为策略选一次动作；保存动作概率，执行后收到传感器和下一状态。
- 用 question 算 c、$\gamma_{t+1}$、目标动作概率。
- 用旧 w 算当前预测、下一预测和 $\delta$；一次性更新 w。
- 转到下一状态继续；GVF 停止只影响它的余项，不自动重置机器人。

对 y 中的 w 也求导会得到 residual-gradient 型更新，方向含 $x_t-\gamma_{t+1}x_{t+1}$。这并不是同一算法。进一步要把单次转移的平方残差与期望 Bellman 残差区分开：对后者构造无偏梯度涉及独立下一状态样本等问题。因此，相同的单样本 TD error 可以对应不同的优化目标与更新方向。

<a id="gvf-traces"></a>

## 5 · 多步预测与资格迹

$$
G_t^{(n)}=\sum_{k=0}^{n-1}\left(\prod_{j=1}^{k}\gamma_{t+j}\right)C_{t+k+1}+\left(\prod_{j=1}^{n}\gamma_{t+j}\right)\hat v(S_{t+n})
$$

n-step target 累积 n 步真实信号后才接预测。n 大时通常减少对 bootstrap 的依赖，但需要更多未来数据，方差和等待时间也会改变。

$$
\begin{gathered}G_t^\lambda=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}\\G_t^\lambda-\hat v(S_t)=\sum_{k=0}^{\infty}\left(\prod_{j=1}^{k}\gamma_{t+j}\right)\lambda^k\delta_{t+k}\end{gathered}
$$

这里先取固定预测参数、on-policy 和收敛的级数；第二式由 bootstrap 项相消得到。$\lambda$=0 只剩当前 TD error。有限终止轨迹的 $\lambda$=1 对应完整回报极限，不能把任意无限无折扣问题也照搬。

$$
e_t=\gamma_t\lambda e_{t-1}+x_t,\qquad w_{t+1}=w_t+\alpha\delta_t e_t
$$

把同一个未来误差应对过去特征产生的影响，压缩到资格迹 e。迹延续到当前使用 $\gamma_{t}$；target 离开当前使用 $\gamma_{t+1}$。固定参数的前后向恒等式不意味着普通在线 accumulating TD($\lambda$) 在有限步长下逐次等于所有 forward-view 更新；true-online 方法正是处理这种在线等价性的一条路线。

| 两步轨迹，$\alpha$=.1、$\lambda$=.8 | e | $\delta$ | w |
| --- | --- | --- | --- |
| 初始化；s0→s1，c=0，$\gamma_{t+1}$=.9 | (1,0) | 0 | (0,0) |
| s1→终点，c=1，$\gamma_{t+1}$=0 | (.72,1) | 1 | (.072,.1) |
| 下一次问题开始，$\gamma_t$=0 | 旧迹不再延续 | 重新由当前样本决定 | 已学 w 不清零 |

$\lambda$ 不是“记住更久的状态”。它影响学习信用；当前输入仍可能不包含历史。一个只看当前画面的函数加上资格迹，不会因此自动变成能辨别所有历史的状态表示。

<a id="gvf-offpolicy"></a>

## 6 · 离策略预测与重要性采样

$$
\rho_t=\frac{\pi(A_t\mid S_t)}{b(A_t\mid S_t)},\qquad \mathbb E_{A\sim b}[\rho f(A)\mid s]=\sum_a\pi(a\mid s)f(a)
$$

条件是 $\pi(a\mid s)$>0 时 b(a|s)>0。若问题问一个真实行为永不尝试的动作，仅乘比率不能制造缺失信息。比率使用采样时的行为概率；更新策略后重新计算这个概率会改变含义。

$$
e_t=\rho_t(\gamma_t\lambda e_{t-1}+x_t),\qquad w_{t+1}=w_t+\alpha\delta_t e_t
$$

这是本章 ordinary importance-sampled TD($\lambda$) 的约定。$\lambda$=0 退化为 $\alpha\rho\delta x$ 更新。对旧迹继续乘当前 $\rho$，会形成历史比率乘积，可能产生很大方差。不同 off-policy return / trace 设计不是任意可互换的。

这个换测度只把给定状态下的动作分布 b 改成 $\pi$；状态仍由真实行为的长期访问分布 $d_b$ 提供。在表格覆盖充分的小问题上可能工作得很好，但函数逼近、bootstrap 和 off-policy 组合后，不保证稳定。先把“回答正确策略的问题”和“数值学习能稳定”分成两件事。

例：一个状态、两动作，动作 1 的 cumulant 为 1，动作 0 为 0，$\gamma$=.5；$\pi$ 永远选 1，b 均匀。问题答案是 2，不用 IS 则通常学到行为策略的答案 1。这个例子揭示换题错误，却还不是发散反例。

<a id="gvf-projection"></a>

## 7 · 线性函数逼近下的发散

$$
\begin{gathered}A=\mathbb E_b[\rho x(x-\gamma'x')^\top],\quad b_c=\mathbb E_b[\rho C'x],\quad C_x=\mathbb E_b[xx^\top]\\\mathbb E[\Delta w\mid w]=\alpha(b_c-Aw)\end{gathered}
$$

$b_c$ 是向量，不是行为策略 b；$C_x$ 是特征二阶矩，不是累计信号。表格可能精确满足 Bellman 方程，但固定低维表示通常不能，于是期望 TD 更新寻找的是特定采样权重下的投影固定点。矩阵 A 的稳定性决定这条动力学会不会远离解。

构造一个无需复杂神经网络的反例：两状态 s0、s1，特征分别为 1、2；动作 a 直接把下一状态设为 $s_a$。行为以 .9/.1 选择动作 0/1，因此 $d_b$=(.9,.1)；目标永远选动作 1，$\gamma$=.9，所有 cumulant 为 0。真实答案处处为 0，w=0 能精确表示。

$$
\begin{gathered}A=.9\cdot1(1-.9\cdot2)+.1\cdot2(2-.9\cdot2)=-.68\\\mathbb E[\Delta w]=+.68\alpha w\end{gathered}
$$

非零 w 的绝对值在期望更新下越来越大，尽管真实答案可表示、比率算对了、世界完全平稳。配套 `counterexample` 实验执行这一期望递推，分离更新方向与采样噪声的作用。

<a id="gvf-gtd"></a>

## 8 · 投影 Bellman 误差与 GTD2

$$
\begin{gathered}\Pi=\Phi(\Phi^\top D_b\Phi)^{-1}\Phi^\top D_b,\quad J(w)=\tfrac12\|\Pi(T\Phi w-\Phi w)\|_{D_b}^2\\J(w)=\tfrac12(b_c-Aw)^\top C_x^{-1}(b_c-Aw)\end{gathered}
$$

$\Phi$ 的每一行是一个状态特征；$D_b$ 用行为访问权重给误差加权；$\Pi$ 投影回特征能表示的函数空间。代入线性 Bellman 残差得到第二式，前提是 $C_x$ 可逆。这里最小化投影 Bellman 误差，不是直接最小化到真实 v 的平方误差。

$$
-\nabla_wJ=A^\top C_x^{-1}(b_c-Aw),\qquad h^*(w)=C_x^{-1}(b_c-Aw)
$$

把昂贵的矩阵求逆改为辅助预测：对固定 w，让 h 求解 $C_xh=b_c-Aw$。h 的作用是估计纠正方向，不是另一套预测目标网络。

$$
\begin{gathered}h_{t+1}=h_t+\beta\,[\rho_t\delta_t-x_t^\top h_t]x_t\\w_{t+1}=w_t+\alpha\rho_t(x_t-\gamma_{t+1}x_{t+1})(x_t^\top h_t)\end{gathered}
$$

第一行的期望是 $\beta(b_c-Aw-C_xh)$，第二行的期望是 $\alpha A^\top h$。辅助量跟踪得足够好时主更新逼近下降方向。两行都用旧 w、旧 h；源代码先缓存，不先改 h 再计算主方向。

在前面的两状态例子，$C_x$=1.3，理想下降方向为 $-(.68^2/1.3)w$，与普通 TD 的 +.68w 方向相反。步长缩放不改变方向；GTD2 改变的是期望更新本身。随机收敛理论还需固定策略/特征、覆盖、矩阵条件和合适步长；不能把这个期望例子推广成任意神经网络的收敛保证。

<a id="gvf-tdc"></a>

## 9 · TDC 与 GTD($\lambda$)

$$
A^\top h=C_xh-\mathbb E_b[\rho\gamma'x'x^\top]h\ \approx\ b_c-Aw-\mathbb E_b[\rho\gamma'x'x^\top]h
$$

由于 $\mathbb E[\rho\mid s]=1$，$A^T$ 的第一项可写为 $C_x$。只有当 h 接近自己的固定点时，$C_xh$ 才可用 $b_c-Aw$ 替代。这样获得 TDC 的期望方向，不意味着它与 GTD2 每一步相同。

$$
\begin{gathered}w_{t+1}=w_t+\alpha\rho_t[\delta_t x_t-\gamma_{t+1}x_{t+1}(x_t^\top h_t)]\\h_{t+1}=h_t+\beta[\rho_t\delta_t-x_t^\top h_t]x_t\end{gathered}
$$

这是 TDC(0)：主项保留 TD 更新，再加入梯度校正；h 与 GTD2(0) 的更新相同。第一步 h=0 时 GTD2 主参数不动，TDC 一般会动——一个很简单的实现区分测试。

$$
\begin{gathered}e_t=\rho_t(\gamma_t\lambda e_{t-1}+x_t)\\w_{t+1}=w_t+\alpha[\delta_t e_t-\gamma_{t+1}(1-\lambda)x_{t+1}(e_t^\top h_t)]\\h_{t+1}=h_t+\beta[\delta_t e_t-x_t(x_t^\top h_t)]\end{gathered}
$$

这里称 GTD($\lambda$) 的是 RLPark 使用的 TDC-style 形式。$\lambda$=0 时 $e=\rho x$，正好退化为前面的 TDC，而不是 GTD2。辅助更新的投影项仍是 $x(x^Th)$，不能把其中每个 x 都替换成 e。

- 保存旧 w、h 和上一时刻的 $\gamma$、迹。
- 用当前 c 和 $\gamma_{t+1}$ 计算 $\delta$。
- 用当前 $\rho$、进入当前的 $\gamma$ 更新 e。
- 从旧 h 计算 $e^Th$ 与 $x^Th$，再生成两个增量。
- 应用增量；最后保存 $\gamma_{t+1}$ 供下一个转移使用。

本章 $\lambda$ 为常数。若使用状态相关 $\lambda$，下一状态 continuation 中的 $\lambda$ 下标也要按该算法原定义重新核对，不同下标约定对应不同的前向回报与资格迹。GTD 系列牺牲了额外向量与步长调节成本，以处理特定离策略函数逼近问题；它并不总是在有限样本上最快。

<a id="gvf-etd"></a>

## 10 · Emphatic TD 与状态加权

另一条路线不显式估计 MSPBE 梯度，而是调整更新的有效状态权重。interest $i_{t}$≥0 表示在哪些状态更关心预测；它不是奖励，也不改变 GVF 的 cumulant。Follow-on trace F 追踪目标行为从这些关心的状态出发会到达哪里；emphasis M 再结合 bootstrap 程度分配更新。

$$
\begin{gathered}F_t=i_t+\gamma_t\rho_{t-1}F_{t-1}\\M_t=\lambda i_t+(1-\lambda)F_t\\e_t=\rho_t(\gamma_t\lambda e_{t-1}+M_t x_t),\qquad w_{t+1}=w_t+\alpha\delta_t e_t\end{gathered}
$$

F 用上一动作的 $\rho_{t-1}$；e 用当前动作的 $\rho_{t}$。第一次没有前驱，置 $\gamma_t$=0、F=0；i=1 是常用起点。$\lambda$=1 时 M=i，i=1 下退化为相应 IS-TD(1) 迹；$\lambda$=0 时 M=F。

为什么强调有效？在固定策略下，$\lambda$=0 的期望 follow-on 权重满足 $f=D_bi+M^\top f$，其中此处的 M 是第 3 节折扣转移矩阵，不是标量 emphasis $M_{t}$。它把起点关注度沿目标动力学传播，使线性系统获得与普通 $d_b$ 投影不同的权重。更换权重也会更换逼近误差在状态之间的取舍。

代价是方差：连续的大重要性比可以让 F 和 e 很大。理论中的稳定期望方向、随机算法的收敛条件和短期数值表现是不同命题。随意裁剪 F/$\rho$ 会改变算法；可以作为实验变体，但不能继续直接引用未裁剪版本的结论。

<a id="lesson-example"></a>

## 11 · 共享经验的三个预测问题

实验环境是三个区域的环：动作 0 留在原地，动作 1 前进到下一区域；到区域 2 视为到达充电点。环境本身持续运行。行为以 .5 概率选各动作。energy 问目标 $\pi$(前进)=.8 下的折扣累计步成本；arrival 问相同行为下的折扣到达；slow-arrival 问 $\pi$(前进)=.2 下的折扣到达。到达时 $\gamma_{t+1}$=0，其余 .9。

| 真实转移 | energy 的 (c,$\gamma_{t+1}$,$\rho$) | arrival 的 (c,$\gamma_{t+1}$,$\rho$) | slow-arrival 的 (c,$\gamma_{t+1}$,$\rho$) |
| --- | --- | --- | --- |
| s0 —前进→ s1 | (1,.9,1.6) | (0,.9,1.6) | (0,.9,.4) |
| s1 —前进→ s2 | (1,0,1.6) | (1,0,1.6) | (1,0,.4) |
| s2 —前进→ s0 | (1,.9,1.6) | (0,.9,1.6) | (0,.9,.4) |

先按 TD(0)、$\alpha$=.1、各 w=0 手算 energy。第一步 $\delta$=1，w(s0)=.16。第二步 $\gamma_{t+1}$=0，因此不管 s2 预测多少，$\delta$=1，w(s1)=.16。第三步 $\delta$=1+.9×.16−0=1.144，w(s2)=.18304。第二步结束的是一个预测累计段，不是清空整个经验流；第三步仍可以从充电点出发问下次到达。

若换 GTD2，初始 h=0 导致第一步 w 不动，但 h(s0)=$\beta$×1.6；下一次相关特征出现时，辅助信息才影响 w。若换 GTD($\lambda$)，第二步的误差还能沿 e 回传给 s0。它们回答的是同一 question，瞬时学习路径不同。

$$
v_{\rm energy}(s_1)=1+.2\cdot.9v_{\rm energy}(s_1),\qquad v_{\rm energy}(s_0)=1+.9[.2v(s_0)+.8v(s_1)]
$$

到 s2 的余项为 0。因此 v(s1)=1/.82；再解出 v(s0)。程序直接构造 I−M 并用独立消元法求所有状态、所有 question 的参考答案，而不是拿某个学习器当真值。

<a id="lesson-code"></a>

## 12 · 算法实现

**算法：固定特征下的 Horde 式并行预测**

1. 为每个问题 $j$ 指定累积信号 $c^j$、延续因子 $\gamma^j$、目标策略 $\pi^j$ 和学习器。
1. 初始化各学习器的权重、辅助向量及资格迹。
1. 每个真实环境步：
  1. 按行为策略 $b(\cdot\mid S_t)$ 采样一次动作，保存 $b(A_t\mid S_t)$。
  1. 执行动作，观察 $S_{t+1}$ 和各传感器信号。
  1. 使用同一份更新前特征，对每个问题 $j$：
    1. 计算 $C^j_{t+1}$、$\gamma^j_{t+1}$、$\rho^j_t=\pi^j(A_t\mid S_t)/b(A_t\mid S_t)$。
    1. 由旧权重计算 $\delta^j_t=C^j_{t+1}+\gamma^j_{t+1}v^j(S_{t+1})-v^j(S_t)$。
    1. 按指定的 TD、GTD 或 ETD 规则，更新该问题独立的参数和迹。
    1. 保存该问题的延续系数及所需历史量。
  1. 继续真实交互；预测问题的终止不触发环境重置。

下面三段来自同一个独立脚本：question 接口、学习器、共享数据的完整 Horde-style 循环。Python 列表只是为了看清每个量，复杂度为每个 head 每步 O(d)；多 head 版本为 O(md)。这些向量不随生命期增长。

① Question：把真实 transition 转成三个预测问题各自的信号、延续与策略概率

```python
def question(name, state, action, next_state):
    """Return cumulant, gamma_next, target probability of the OBSERVED action.

    'arrival' predicts a discounted next-arrival signal, not an undiscounted
    eventual-arrival probability. 'slow-arrival' asks about a different policy.
    Arrival is a transition into cell 2, including staying there for one step.
    """
    forward = .2 if name == 'slow-arrival' else .8
    pi_observed = forward if action == 1 else 1-forward
    arrived = next_state == 2
    cumulant = 1. if name == 'energy' else float(arrived)
    gamma_next = 0. if arrived else .9
    return cumulant, gamma_next, pi_observed
```

② 六种固定特征学习器：delta 与 correction 使用旧参数；各 head 独立保存迹

```python
class LinearGVF:
    """Fixed-feature prediction. Right-hand sides always use OLD w and h.

    gamma_current belongs to the transition entering x; gamma_next to x -> xp.
    GTD(lambda) here is the TDC-style form used in RLPark, not GTD2(lambda).
    ETD uses a follow-on trace and interest, not a second learned value vector.
    """
    methods = ('td0','tdlambda','gtd2','tdc','gtdlambda','etd')

    def __init__(self, dimension, method='td0', alpha=.01, beta=.05, lam=.6):
        if method not in self.methods or dimension < 1:
            raise ValueError('Invalid method or dimension')
        if not all(math.isfinite(v) for v in (alpha,beta,lam)) or alpha<=0 or beta<=0 or not 0<=lam<=1:
            raise ValueError('Positive finite step sizes and lambda in [0,1] required')
        self.method,self.alpha,self.beta = method,alpha,beta
        self.lam = 0. if method in ('td0','gtd2','tdc') else lam
        self.w,self.h,self.e = [[0.]*dimension for _ in range(3)]
        self.gamma_current,self.rho_previous,self.followon = 0.,0.,0.

    def step(self, x, xp, cumulant, gamma_next, rho, interest=1.):
        if len(x)!=len(self.w) or len(xp)!=len(x):
            raise ValueError('Feature dimension mismatch')
        if not all(math.isfinite(v) for v in [*x,*xp,cumulant,gamma_next,rho,interest]):
            raise ValueError('Nonfinite input')
        if not 0<=gamma_next<=1 or rho<0 or interest<0:
            raise ValueError('Invalid discount, ratio or interest')
        old_w,old_h = self.w[:],self.h[:]
        delta = cumulant + gamma_next*dot(old_w,xp) - dot(old_w,x)
        if self.method == 'etd':
            self.followon = interest + self.gamma_current*self.rho_previous*self.followon
            emphasis = self.lam*interest + (1-self.lam)*self.followon
        else:
            emphasis = 1.
        self.e = [rho*(self.gamma_current*self.lam*ei + emphasis*xi)
                  for ei,xi in zip(self.e,x)]
        if self.method == 'gtd2':
            xh = dot(x,old_h)
            dw = [rho*(xi-gamma_next*xpi)*xh for xi,xpi in zip(x,xp)]
        elif self.method in ('tdc','gtdlambda'):
            eh = dot(self.e,old_h)
            dw = [delta*ei-gamma_next*(1-self.lam)*xpi*eh for ei,xpi in zip(self.e,xp)]
        else:
            dw = [delta*ei for ei in self.e]
        self.w = [wi+self.alpha*di for wi,di in zip(old_w,dw)]
        if self.method in ('gtd2','tdc','gtdlambda'):
            xh = dot(x,old_h)
            self.h = [hi+self.beta*(delta*ei-xh*xi) for hi,ei,xi in zip(old_h,self.e,x)]
        self.gamma_current,self.rho_previous = gamma_next,rho
        return delta

    def external_reset(self):
        """Use only when the data protocol actually breaks the trajectory.

        A GVF gamma_next=0 already terminates its own trace on the next step;
        it does not require resetting the physical environment or weights.
        """
        self.e = [0.]*len(self.w)
        self.gamma_current,self.rho_previous,self.followon = 0.,0.,0.
```

③ 完整训练与评估：环境只前进一步，每个预测问题各更新一次

```python
def run(method='gtdlambda', steps=40000, seed=7, alpha=.01, beta=.05, lam=.6, use_is=True):
    """A complete multi-question loop, sharing data but NOT traces or weights."""
    rng=random.Random(seed)
    names=('energy','arrival','slow-arrival')
    heads={name:LinearGVF(3,method,alpha,beta,lam) for name in names}
    features=[[float(i==j) for j in range(3)] for i in range(3)]
    state=0
    for _ in range(steps):
        # The real behavior is sampled exactly ONCE for all questions.
        action=int(rng.random()<.5); behavior_probability=.5
        next_state=(state+action)%3
        for name,head in heads.items():
            c,g,pi_observed=question(name,state,action,next_state)
            rho=pi_observed/behavior_probability if use_is else 1.
            head.step(features[state],features[next_state],c,g,rho)
        state=next_state  # no env.reset() when a question ends
    result={}
    for name,head in heads.items():
        truth=exact_values(name)
        result[name]={'prediction':head.w,'reference':truth,
                      'rmse':math.sqrt(sum((w-v)**2 for w,v in zip(head.w,truth))/3)}
    return result
```

下载本页脚本后，在文件所在目录运行；只需 Python 3.10+ 标准库

```sh
python3 gvf_lab.py compare --steps 40000 --seed 7
python3 gvf_lab.py gtdlambda --lam 0.6
python3 gvf_lab.py td0 --without-is
python3 gvf_lab.py counterexample
python3 gvf_lab.py test
```

compare 输出每个学习器的三个预测向量、解析 reference 和均匀状态加权 RMSE。RMSE 是对这个已知小环境的价值误差，不是 TD loss，也不是机器人控制回报。有限样本和常数步长让不同 seed 的结果不同；本实验不用于排出六种算法的普遍优劣。

counterexample 用完整期望更新复现第 7 节反例：TD 权重远离 0，GTD2 逼近 0。compare 的 one-hot 表示则用于接口和解析答案检查；表格例子的稳定表现不能取消反例。--without-is 会让不同目标策略的到达问题错误地趋向同一个行为策略答案。

<a id="gvf-diagnostics"></a>

## 13 · 数值性质与实验设计

| 检查 | 应该看到什么 | 否则先查什么 |
| --- | --- | --- |
| $\gamma_{t+1}$=0 | 当前 cumulant 保留，next value 消失 | 把当前奖励也乘了 $\gamma$，或 terminal mask 用错 |
| $\lambda$=0 | TD($\lambda$)→TD(0)，本章 GTD($\lambda$)→TDC | 错误地把 GTD($\lambda$) 对齐到 GTD2 |
| $\rho$=0 | 普通 IS-TD 主更新为 0；GTD 辅助量仍可能有投影衰减 | 把两条辅助更新都整体乘 $\rho$ |
| 问题到达终止事件 | 本次误差仍给旧迹信用；下一步旧迹不再延续 | 在终点更新之前就提前清掉 e |
| 所有 head 交换顺序 | 固定共享特征时训练结果不变 | 不小心共享了可变 w、e、F 或重复前进环境 |
| C 同时乘 10 | 真值相应乘 10，但固定步长数值表现未必不变 | 把尺度敏感性误认为问题语义变化 |
| 新 target policy 无行为覆盖 | 无法可靠回答该问题 | 给分母加数值常数不能弥补缺失的行为覆盖 |

实验可逐步放宽假设：先固定表示与问题，再分别改变策略覆盖、噪声、预测时域或特征混叠，最后考虑表示与控制的联合学习。预测精度衡量回答问题的能力；后续控制任务的样本需求衡量这些预测的用途。两类指标应分别报告。

<a id="lesson-branches"></a>

## 14 · 预测方法的关系

| 路线 | 实际改变的对象 | 关键区别 |
| --- | --- | --- |
| MC / n-step / TD($\lambda$) / true-online | 回报估计、信用时域与在线等价性 | 不改变给定 c、$\gamma$、$\pi$ 所定义的真值；有限表示下 fixed point/逼近可能随算法而变 |
| Ordinary IS-TD / GTD2 / TDC / GTD($\lambda$) / ETD | 离策略估计、梯度方向或有效状态权重 | 不是同一套迹加不同算法名 |
| LSTD / LSPE 等最小二乘预测 | 累计线性系统并求解 | 通常需 $O(d^2)$ 存储/计算及求解开销；遗忘因子和非平稳跟踪另需设计 |
| Horde / nexting | 问题集合和共享经验的调度 | Horde 是组织方式，nexting 强调多个近未来尺度；都不是新的单个 loss |
| TD networks / GVFN | 预测之间的依赖及递归表示 | 如果 cumulant 或输入依赖其他可学习预测，目标会移动，还需要跨预测/时间的信用 |
| Successor features | 把 cumulant 扩成特征向量，并按奖励权重复用 | 每个坐标可视为预测；适用的奖励族和动力学条件要成立 |
| 深度 GVF / auxiliary tasks | 非线性共享表示与学习信号 | stop-gradient、target network、replay 可成为工程选择，但不能直接继承固定线性理论 |

要使用预测进行控制，可以把多头输出作为决策输入，或用它们构造子目标、评估风险、近似模型。这个接口必须显式写出：下游究竟用了哪些预测，删去它们有什么变化？“预测到风险”也不等于保证安全，预测错误本身可能在新情境中最大。

这些分支分别放宽固定问题、固定表示或固定数据分布的假设。下一节讨论预测问题本身的选择，以及预测学习与行为学习的相互影响。

<a id="gvf-discovery"></a>

## 15 · 从学习预测到选择预测

前面假定预测问题和行为策略由外部给定。持续智能体还需要决定学哪些问题，以及采取什么行为才能改善这些预测。这两种选择处在不同位置：问题生成改变预测的语义，行为学习改变可获得的数据。

Veeriah 等的 Discovery of Useful Questions as Auxiliary Tasks（NeurIPS 2019）用后续主任务的学习效果评价预测问题。问题参数控制累积信号等内容；辅助预测更新共享表示；外层梯度通过这些更新反传，选择对主任务有用的问题。其关键区别是评价“学习该预测带来的效果”，而非仅选择容易预测的信号。

$$
\theta^+(\eta)=\theta-\alpha\nabla_\theta L_{\mathrm{aux}}(\theta;\eta),\qquad
\nabla_\eta L_{\mathrm{task}}(\theta^+)
=\left(\frac{\partial\theta^+}{\partial\eta}\right)^\top
\nabla_{\theta^+}L_{\mathrm{task}}
$$

这是问题发现的单步链式法则示意。问题参数影响辅助更新，辅助更新影响后续任务损失。多步展开还需累积中间更新的参数依赖；具体预测结构与梯度截断属于方法设定。

McLeod 等的 Continual Auxiliary Task Learning（NeurIPS 2021）保留一组辅助预测，进一步学习行为策略以收集有助于这些预测的数据。预测器不断改进会改变学习进展信号；行为变化又改变预测器的训练分布。论文使用 successor features 将动力学相关预测与变化的奖励权重分开，研究这种耦合系统的跟踪能力。

Modayil 与 Abbas 的 Nibbler（2023）研究另一种结构选择：从无结构的观测特征中选择与奖励相关的累积信号，为不同预测器选择局部输入，再把预测器生成的非线性特征用于主任务价值学习。其多组合环境用于检验观测维数增长时的计算与样本需求；结论依赖该问题族和特征选择机制。

Voelcker 等的 When does Self-Prediction Help?（RLC 2024）解释了为什么“预测准确”不足以决定表示是否适合控制。观测重构、潜在状态自预测与 TD 共同学习时，对特征的要求不同；奖励无关的干扰变量尤其会改变这种关系。其线性分析与 MinAtar 实验把“独立训练表示”和“辅助主任务训练”区分开来。

$$
z_t=\phi_\theta(o_t),\qquad
L_{\mathrm{latent}}=\|f_\psi(z_t,a_t)-\operatorname{sg}[\phi_\theta(o_{t+1})]\|^2,\qquad
L_{\mathrm{joint}}=L_{\mathrm{TD}}+\xi L_{\mathrm{latent}}
$$

上式给出常见潜在自预测的结构：预测目标是下一观测的表示，并在目标端停止梯度。这个一步辅助目标可以帮助解释表示学习，但它并不自动定义任意时域、任意目标策略下的 GVF。

因此，研究预测知识至少包含三个可分别检验的问题：预测对象能否表达所需信息，学习器能否在当前经验分布下准确跟踪，以及这些预测是否改善决策。它们对应问题设计、预测算法和下游使用三类实验。

<a id="research-gvf-measures-and-readouts"></a>

## 研究专题 A · 从有限预测向量到可查询的未来占用

GVF 的基本单位是一个明确的问题；successor features 将有限个信号在同一策略下的未来累计组成向量。当未来奖励尚未知时，新的问题是：有限信号族遗漏了什么，能否学一个可被更多信号查询的未来占用？FB 和 SF² 分别给出低秩占用与生成式 successor measure 两条路线。它们扩展预测对象，而不取消目标策略、时域和覆盖要求。

$$
\mu_\gamma^\pi(B\mid s,a)=(1-\gamma)\sum_{k=0}^{\infty}\gamma^k\Pr_\pi(S_{t+k+1}\in B\mid S_t=s,A_t=a),\qquad Q_c^\pi(s,a)=\frac{1}{1-\gamma}\int c(x)\,\mu_\gamma^\pi(dx\mid s,a)
$$

此处明确采用从下一状态开始、归一化的占用约定，且 cumulant 为到达状态函数 c(x)、固定 0≤γ<1。一般转移 cumulant 或动作相关奖励须扩展所占用的对象；状态相关 continuation 也不能机械使用这条固定折扣归一化。

若每步信号是“到达充电区”，占用积分给折扣访问累计；它可以反复计数，不是首次到达概率。若希望在首次事件停止，必须把 stopping 规则写进问题，或扩展状态为尚未到达/已到达。给 γ=0.9 时，归一化占用对充电区的质量为 0.2，累计访问期望便为 2；把质量 0.2 当成累计值会少掉因子 10。

$$
\mu_\gamma^\pi(\cdot\mid s,a)=(1-\gamma)P(\cdot\mid s,a)+\gamma\,\mathbb E_{S'\sim P,\,A'\sim\pi}[\mu_\gamma^\pi(\cdot\mid S',A')]
$$

measure Bellman 方程是分布的混合：部分目标来自真实一步后果，部分来自下一状态策略条件的长期预测。生成式 bootstrap 仍会传播估计误差；无需显式长 rollout 不代表没有长期误差。

$$
\psi^\pi(s,a)=\frac{1}{1-\gamma}\int\phi(x)\,\mu_\gamma^\pi(dx\mid s,a),\qquad r_w(x)=\phi(x)^\top w\Rightarrow Q_w^\pi(s,a)=\psi^\pi(s,a)^\top w
$$

SF 是占用分布在有限特征上的投影。奖励张成空间、固定策略与不变动力学一起决定复用边界；这条线性价值恒等式不能直接推广到任意 learned embedding。

Does Zero-Shot RL Exist?（ICLR 2023）将 FB 与不同 SF 基础特征放在同一固定数据上比较，说明联合学习可读出的占用结构与任意自监督特征并不等价。其最优性目标和有限神经训练结果需分开看：良好的 buffer 覆盖是迁移结果的一部分，不是凭空从零样本查询产生的经验。

实验可从固定策略、固定动力学开始：学习占用或 SF 后冻结表示，公布一组未参与表示训练的 cumulants；分别测查询读出误差、行为覆盖和下游控制。再改变策略、动力学或 continuation，每次只放宽一个复用前提。研究空缺是有限预算下怎样选择需保留的查询、检测新查询超出表示范围，并用未来经验补齐。

<a id="research-gvf-flow-feature-boundaries"></a>

## 研究专题 B · SF² 的线性结构究竟在哪一层

SF²（ICLR 2026）将未来占用作为生成式预测问题，再让控制器使用压缩特征。这条链值得逐层核对：它不是将所有 GVF 统一为一个线性 TD 网络，也不是自动学习完整 agent state。原文的结构约束加在生成向量场上，而不是最终的 reward readout 或 critic 上。

$$
u_\theta(x,k,s,a)=\zeta_\theta(x,k)^\top\psi_\theta(s,a),\qquad \frac{dx_k}{dk}=u_\theta(x_k,k,s,a),\qquad Q_\omega(s,a)=g_\omega(\psi_\theta(s,a))
$$

x 是生成的未来状态位置，k∈[0,1] 是噪声到目标分布的生成时间，ψ 是当前状态动作的条件特征，ζ 是随 x 与 k 改变的矩阵投影。g 可以非线性；k 不等于环境原始步数，也不是 GVF 的 discount。

$$
\mathcal L_{\rm flow}=(1-\gamma)\mathcal L_{\rm one\ step}+\gamma\mathcal L_{\rm bootstrap}
$$

两项分别拟合真实下一状态的 flow target 与下一个状态动作条件的目标向量场，具体采样路径、目标网络及停止梯度按原文算法实现。这里表示混合结构，不声称两项可由普通标量 TD error 直接替代。

即使 u 对 ψ 线性，求解 ODE 时 x 随 ψ 改变，而 ζ 又依赖 x；因此最终样本与其统计量一般可非线性依赖 ψ。由“生成向量场线性”跳到“任意新奖励的 Q 都可线性读出”，少了一个实质性证明或实验。原文在小生成时间下给出的 SR-like 更新联系也被明确限定为近似解释。

| 待证明的命题 | 原文提供什么 | 可增加什么检验 |
| --- | --- | --- |
| 能预测多步未来占用 | flow 与 successor mixture 的训练结构 | 独立未来轨迹上的分布距离和多模态覆盖 |
| ψ 适合原任务控制 | 与 TD3/SAC 联合的控制实验 | 同 backbone、相同 update ratio 与模型调用预算的对照 |
| ψ 支持线性新查询 | 不由向量场分解自动推出 | 冻结 ψ，以线性 head 预测新 cumulants；对照非线性 head |
| ψ 能作为递归 agent state | 不是该工作直接研究的对象 | 同观测不同历史、递归更新和梯度信用实验 |
| 适合 strict streaming | 作者算法含 replay、批次与目标网络 | 单次经验协议需要另行设计并报告 |

**算法：不同证据层逐次扩展，保留作者协议与新 CRL 协议的区别**

1. 读作者实现时（不在教材中声称已复现）：
  1. 追踪 networks 中 ψ、ζ 与非线性 Q 的维度
  1. 追踪 losses 中 one-step、bootstrap、critic loss 的梯度路径
  1. 核对 target ψ/ζ 更新时序、γ 与 ODE 积分步数
  1. 固定 replay 与 update ratio，先复现原任务对照
  1. 冻结 ψ 后做新 cumulant 的线性/非线性读出
  1. 最后才改变动力学或改为流式经验协议

对持续预测的启发是将“预测内容丰富”与“下游可便宜查询”分开优化。生成器可以保留多模态未来，而有限特征有助于低成本控制；两者之间是否形成可持续维护、可迁移且资源有界的知识接口，仍是可检验的研究问题。

<a id="lesson-check"></a>

## 16 · 习题与讨论

| 问题 | 推理与答案 |
| --- | --- |
| 预测“未来十步内碰撞概率”，直接用碰撞指示和 $\gamma$=.9 对吗？ | 不对。它通常是折扣碰撞累计量。要先定义首次事件、有限窗口与终止规则；固定折扣不能精确代表十步截断。 |
| 每个 GVF 的 $\gamma$=0 都要 env.reset() 吗？ | 不需要。它只终止自己的累计问题和后续迹延续；真实交互仍可继续，其他 GVF 也可以不停。 |
| 既然用了正确 $\rho$，为什么 TD 仍会发散？ | 动作条件期望被修正，函数逼近下的状态加权和 bootstrap 几何仍可能不稳定；第 7 节给出 A<0 的例子。 |
| 所有 GVF 都预测很准，能断言状态足够好了吗？ | 不能。若没有一个问题区分影响动作的历史，预测完全准确仍可能丢失控制所需信息；必须测表示充分性与下游用途。 |

练习：为一个具体的传感器事件定义累积信号、延续因子和目标策略。给出两条转移上的更新，并构造一个具有解析答案的小环境，比较价值误差与 TD error 的变化。

## 本章的实验设计

固定目标策略和累积信号。用解析预测或独立轨迹检验问题的真值。再改变行为策略检查覆盖与重要性采样。

设定：固定一个二动作预测问题，目标概率为 (0.8,0.2)，行为概率为 (0.4,0.6)。分别预测一次事件和多步累积量，再连接固定控制器。

- 动作 0 的重要性比为 2；加权期望与目标策略期望一致。
- 目标正概率而行为零概率时，程序拒绝声称可从现有数据识别。
- γ=0 预测下一 cumulant；目标策略或信号变化有独立版本。

对照：相同维度随机特征与固定 GVF；在线学习 GVF 与冻结已学 GVF；固定下游容量、数据权限与计算

记录：解析或独立目标策略下的预测误差；各问题尺度、覆盖、重要性比尾部；使用预测前后的原任务收益

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-modules)

## 学习与研究衔接

标量奖励价值是 GVF 的一种特例。定义多个问题不等于已经学出有用状态或控制策略。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-gvf) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=gvf) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=gvf)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

状态是支持后续计算的内部信息；GVF 指定一个预测问题；RTRL 和资格迹规定信用如何传播。三者可以组合，但不是相互替代的算法名称。先理解给定策略的预测，再讨论预测怎样改善控制。

- [Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks](https://yingwen.io/zh/continual-rl/research/#recent-columnar-constructive-networks)
- [Towards model-free RL algorithms that scale well with unstructured data](https://yingwen.io/zh/continual-rl/research/#recent-nibbler-predictive-features)
- [When does Self-Prediction help? Understanding Auxiliary Tasks in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-self-prediction-auxiliary-tasks)
- [Does Zero-Shot Reinforcement Learning Exist?](https://yingwen.io/zh/continual-rl/research/#recent-zero-shot-forward-backward)
- [Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations](https://yingwen.io/zh/continual-rl/research/#recent-successor-flow-features)

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)
- [Expected Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-expected-eligibility-traces)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

模型可提取性的理论说明某类能力需要什么知识，不指定唯一网络。Dreamer 研究潜在想象控制，STOMP 研究随机时长行为模型，DRAGO 研究旧模型知识保留。模型误差、查询策略和规划收益之间仍需实验连接。

- [The Value Equivalence Principle for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-value-equivalence-models)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文提供定义和条件，架构讲座提出模块组织，算法论文检验特定机制。完整系统还要明确智能体、外部设计者和世界各自承担的工作；组件成立不自动意味着组合后的长期收益成立。

- [The OaK Architecture: A Vision of SuperIntelligence from Experience](https://yingwen.io/zh/continual-rl/research/#recent-oak-architecture)
- [Towards model-free RL algorithms that scale well with unstructured data](https://yingwen.io/zh/continual-rl/research/#recent-nibbler-predictive-features)
- [The Value Equivalence Principle for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-value-equivalence-models)
- [Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations](https://yingwen.io/zh/continual-rl/research/#recent-successor-flow-features)

### Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks

Khurram Javed, Haseeb Shah, Richard S. Sutton, Martha White

JMLR 24 · 2023 · 支持方法与理论

#### 研究问题

如果每次观测只处理一次，如何学习包含历史信息的状态，而不保存一段序列做反向传播？

#### 关键机制

一般递归网络的实时递归学习需要维护庞大的参数—状态敏感度。CCN 限制列之间的递归依赖，并逐步构造新特征，使敏感度可以局部计算。它通过改变网络结构和构造过程降低求导成本，而不是把任意稠密 RNN 的完整导数免费变小。

#### 证据

论文分析受限结构的计算性质，并在动物学习启发的预测问题和 Atari 策略评价中检验预测效率。这里的 Atari 结果主要是预测已有策略的回报，不等于从头训练完整控制智能体。

#### 条件与限制

结构约束、构造顺序和被冻结的旧特征共同限制函数类。监督预测和策略评价上的优势，还需要在会主动改变数据分布的控制闭环中检验。

#### 阅读与实验

先写出递归状态对参数的敏感度递推，再检查哪些跨列项被结构消除。比较 CCN、截断 BPTT 与 RTU 时，同时计入状态、梯度缓存和每步计算。

#### 原文与相关入口

- [JMLR 原文与论文入口](https://www.jmlr.org/papers/v24/23-0367.html)：从网络结构、敏感度传播与预测实验三部分阅读。

### Towards model-free RL algorithms that scale well with unstructured data

Joseph Modayil, Zaheer Abbas

arXiv 预印本 · 2023 · 支持方法与理论

#### 研究问题

大量原始观测中只有少数局部组合与奖励有关，智能体能否逐步构造有用的预测特征？

#### 关键机制

Nibbler 将预测问题的构造和预测结果的复用结合起来：选择局部输入、学习与奖励相关的通用价值预测，再将预测作为后续学习的特征。GVF 在这里不是一个新的优化器，而是描述“预测什么、在什么行为下预测”的问题接口。

#### 证据

作者在组合式合成环境中增加观测规模，报告了利用任务结构的样本效率。环境可以具有指数增长的状态组合，但学习器不必显式枚举全部状态。

#### 条件与限制

这不是对任意高维观测的线性样本复杂度保证。局部可分解结构、候选问题与特征构造规则仍是关键条件；从该实验族迁移到视觉控制需要额外验证。

#### 阅读与实验

把一条预测完整写成累积量、延续条件、目标策略和输入特征四项，再指出它如何进入主任务的价值函数。区分问题生成带来的收益与增加参数量带来的收益。

#### 原文与相关入口

- [作者预印本](https://arxiv.org/abs/2311.02215)：问题族、Nibbler 构造过程与扩展性实验。

### When does Self-Prediction help? Understanding Auxiliary Tasks in Reinforcement Learning

Claas A. Voelcker, Tyler Kastner, Igor Gilitschenski, Amir-massoud Farahmand

RLC 2024 / RLJ · 2024 · 支持方法与理论

#### 研究问题

预测下一潜在状态、重建观测和学习价值，为什么会产生不同的表示？

#### 关键机制

论文在含干扰因素的线性问题中分析辅助目标的学习动力学。潜在状态自预测与价值学习共同作用时可能保留决策相关结构，但单独训练同一目标未必得到最有用的特征。目标的作用取决于它和 TD 目标怎样共享表示。

#### 证据

线性分析给出可检查的条件，并用神经网络实验检验部分预测。结果不支持“任何自监督预测都能改善 RL”这种无条件判断。

#### 条件与限制

线性分析中的观测映射、优化过程与神经网络控制并不完全等价。项目仓库入口不等于已经提供完整可复现实验实现，因此这里不列为可运行代码。

#### 阅读与实验

固定编码器容量，分别比较仅 TD、仅辅助任务和联合训练。记录价值误差与任务收益，不要只用辅助损失下降评价表示。

#### 原文与相关入口

- [RLC 2024 论文入口](https://rlj.cs.umass.edu/2024/papers/Paper197.html)：原文、线性假设与神经网络实验。
- [作者预印本](https://arxiv.org/abs/2406.17718)：便于追踪论文版本。

### Deep Reinforcement Learning with Gradient Eligibility Traces

Esraa Elelimy, Brett Daley, Andrew Patterson, Marlos C. Machado, Adam White, Martha White

RLC 2025 / RLJ · 2025 · 支持方法与理论

#### 研究问题

资格迹怎样与明确的梯度目标结合，而不是直接把线性半梯度规则搬到深度网络？

#### 关键机制

论文从广义投影 Bellman 误差出发构造多步目标，推导带资格迹的梯度学习方法。前向视角连接多步回报与经验重放，后向视角通过递推迹分配信用。目标函数、辅助估计器和迹的更新共同决定算法，不只是选择一个较大的 λ。

#### 证据

作者给出多种算法并在 MuJoCo、MinAtar 等任务中比较。代码同时提供相关梯度算法与实验设置，可以把推导中的量映射到实际更新。

#### 条件与限制

线性 GTD 的收敛条件不能自动赋予非线性实现全局收敛保证。重放版本与流式版本的数据使用预算也不能混为一谈。

#### 阅读与实验

从一段短轨迹分别计算前向多步目标和后向迹。随后对照原代码检查辅助网络、目标与主网络参数使用的是更新前还是更新后的值。

#### 原文与相关入口

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_302.pdf)：目标、算法推导与实验。
- [作者算法库](https://github.com/esraaelelimy/gtd_algos)：论文提供的梯度 TD 与资格迹实现。

#### 作者代码

[原论文链接的作者仓库。](https://github.com/esraaelelimy/gtd_algos)

论文梯度算法、资格迹和实验配置。

### The OaK Architecture: A Vision of SuperIntelligence from Experience

Richard S. Sutton

RLC 2025 讲座 / Oak Lab · 2025 · 定义与架构观点

#### 研究问题

持续学习是否只是在一个现成 actor–critic 上加入抗遗忘机制，还是需要重新安排知识构造与使用？

#### 关键机制

OaK 提出从经验持续形成状态、预测知识、子任务、时间抽象与模型，并让这些知识服务规划的架构方向。这里的重点是模块之间怎样产生可复用知识，而不只是保留某个固定策略网络的参数。

#### 证据

官方页面提供 Richard Sutton 的架构讲座与相关研究入口。STOMP、预测学习和在线特征学习等论文可以检验其中具体组件，但不能自动验证整体架构。

#### 条件与限制

这是研究愿景与架构讲解，不是一套已公布完整训练配方、统一基准结果和可复现端到端代码的系统。资源分配、问题生成、知识替换与模块相互干扰仍需明确算法。

#### 阅读与实验

为每个模块写出输入、输出、更新频率和资源上限。再选择一个双模块接口做可证伪实验，例如技能模型改善是否真的减少规划误差。

#### 原文与相关入口

- [Oak Lab 官方讲座页面](https://oaklab.ai/posts/the-oak-architecture)：讲座入口与架构研究方向。
- [Oak Lab 研究主页](https://oaklab.ai/)：区分已发表研究、技术文章和仍在预告中的项目。

### Expected Eligibility Traces

Hado van Hasselt, Sephora Madjiheurem, Matteo Hessel, David Silver, André Barreto, Diana Borsa

AAAI 2021（2020预印本） · 2021 · 支持方法与理论

#### 研究问题

当前误差能否同时更新本次未走过、但也可能到达当前状态的过去路径？

#### 关键机制

学习给定当前状态的资格迹条件均值，再用当前TD误差更新该均值所指向的过去预测。递归混合在实际轨迹迹与预测的期望迹之间插值；预测对象是过去资格，而非未来奖励。

#### 证据

原文在Markov状态与相应条件下证明更新均值相同、逐分量方差不增，并在路径汇合问题检验预测效率。信用章精确枚举一个正例和一个状态混叠反例。

#### 条件与限制

不完整观察、参数漂移和近似迹预测器会破坏无偏条件。全参数期望迹预测还有输出维度和计算成本；小实验不复现作者的神经实验。

#### 阅读与实验

保持奖励边际分布一致，仅改变奖励是否依赖隐藏的过去路径。先测信用均值与方差，再研究agent state能否恢复条件独立。

#### 原文与相关入口

- [作者原文](https://arxiv.org/html/2007.01839)：Lemma 1、Proposition 1及ET(λ,η)递归混合。
- [AAAI发表版本](https://ojs.aaai.org/index.php/AAAI/article/view/17200)：正式会议年份为2021。

### Does Zero-Shot Reinforcement Learning Exist?

Ahmed Touati, Jérémy Rapin, Yann Ollivier

ICLR 2023 · 2023 · 支持方法与理论

#### 研究问题

没有事先指定奖励时，怎样学一套预测表示，日后接收新奖励就能选行为？

#### 关键机制

Forward–Backward 表示联合学习行为条件的未来占用与奖励读出，而非先固定任意编码器再学习 successor features。新奖励被映射到任务向量，策略根据这个向量直接行动；该论文系统比较 FB 与多种 SF 基础特征。

#### 证据

原文在固定离线 replay buffers 上比较零样本任务迁移，借此把表示学习与探索数据的质量分开。不同特征与数据覆盖产生显著差异，不能仅靠“所有奖励”的理论目标预测实际效果。

#### 条件与限制

假定共享动力学与可用经验覆盖。无下游梯度更新不等于无预训练成本；有限秩、近似训练和奖励估计都有误差。新动力学、历史混叠和严格一次使用经验均须另测。

#### 阅读与实验

同一 buffer 对比随机特征、谱特征与联合 FB，再独立换 buffer。奖励读出误差、占用误差与新任务回报分别报告，避免把数据覆盖优势记成表示优势。

#### 原文与相关入口

- [作者原文](https://arxiv.org/abs/2209.14935)：2022 首稿，ICLR 2023；比较奖励表示、SF 与 FB。
- [作者研究平台](https://github.com/facebookresearch/controllable_agent)：README 直接关联两篇 FB 论文；该仓库已经归档。

#### 作者代码

[论文作者团队仓库；归档工程，依赖和旧环境需单独核验。](https://github.com/facebookresearch/controllable_agent)

FB 与 SF 的训练、固定数据实验及奖励查询示例。

### Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations

Haosen Shi, Jianda Chen, Sinno Jialin Pan

ICLR 2026 · 2026 · 支持方法与理论

#### 研究问题

能否直接学习多步未来状态的分布，并把它压缩为适合控制学习的特征？

#### 关键机制

SF² 以 flow matching 估计 successor measure，将条件向量场分解为未来位置及生成时间的投影与当前状态动作特征的乘积。特征进入 TD3/SAC 的 critic；线性的是向量场对条件特征的分解，critic 本身可以非线性。

#### 证据

正式原文给出 mixture Bellman 结构、生成式 bootstrap 与控制实验，并提供作者 JAX/Brax 仓库。实验研究在线收集数据下的 off-policy 控制，并使用 replay、批次与目标网络。

#### 条件与限制

“online policy learning”不代表 strict streaming。生成时间不是环境时间；向量场线性不保证任意奖励价值线性。文中与 SR 的小生成时间联系是近似动机，未证明递归 agent state 或任意持续变化下的充分性。

#### 阅读与实验

对齐模型调用与梯度预算，拆分直接预测、bootstrap、critic 联合训练。冻结特征后比较线性与非线性读出，再测新奖励和动力学变化，才能检验预测知识的可复用程度。

#### 原文与相关入口

- [ICLR 2026 正式原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48acf4b231771e693f42305b4c9b4c9f-Abstract-Conference.html)：第 2–3 节和算法附录；区分 flow 时间、环境时间与近似 SR 联系。
- [原文链接的作者实现](https://github.com/Shiien/successor-flow-representation-implementation)：SAC/TD3、flow 特征、对照和 sweep 配置。

#### 作者代码

[正式论文摘要直接链接的作者代码。](https://github.com/Shiien/successor-flow-representation-implementation)

基于 JAX/Brax 的 SF² 控制实验；不包含自动 GVF 问题发现或完整持续架构。

### The Value Equivalence Principle for Model-Based Reinforcement Learning

Christopher Grimm, André Barreto, Satinder Singh, David Silver

NeurIPS 2020 · 2020 · 支持方法与理论

#### 研究问题

模型容量有限时，必须预测全部状态细节，还是只须保持规划会查询的量？

#### 关键机制

Value equivalence 以策略集合和函数集合定义模型规格：模型对这些函数进行这些策略的 Bellman backup，应与真实环境相同。扩大查询族会缩小可接受模型族；它把“决策相关”从口号变成有条件的等价关系。

#### 证据

论文给出等价模型类的性质及有限实验，并解释若干隐式模型方法。后续 Proper Value Equivalence（NeurIPS 2021）研究策略价值固定点等价及规划充分性。

#### 条件与限制

少数当前 critic 的 backup 相同，不说明所有新奖励、新策略或风险目标都相同。精确算子等价与神经损失在样本上较小不同；奖励或查询族变化后须重新验证。

#### 阅读与实验

保存独立的 planner 查询集，直接测 target 误差和动作排序。用未参与模型拟合的价值函数检验迁移，并与像素误差对照，找出模型实际保留的信息。

#### 原文与相关入口

- [NeurIPS 2020 原文](https://papers.nips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)：VE 依赖策略与函数集合。
- [Proper Value Equivalence · NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/400e5e6a7ce0c754f281525fae75a873-Abstract.html)：多步算子、固定点与规划充分性；不是任意潜在网络的保证。


<a id="chapter-code"></a>

## 下载与运行

六个线性学习器、三个共享经验的预测问题、解析 Bellman 参照与一个 off-policy 期望发散反例。

[下载 gvf_lab.py](https://yingwen.io/zh/continual-rl/download/gvf_lab.py)

```sh
python3 gvf_lab.py compare
python3 gvf_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Veeriah et al. · Discovery of Useful Questions as Auxiliary Tasks（NeurIPS 2019）](https://proceedings.neurips.cc/paper/2019/file/10ff0b5e85e5b85cc3095d431d8c08b4-Paper.pdf)：用多步元梯度学习 GVF 问题；区分问题参数、辅助预测更新和主任务评价。

- [McLeod et al. · Continual Auxiliary Task Learning（NeurIPS 2021）](https://papers.nips.cc/paper/2021/hash/68331ff0427b551b68e911eebe35233b-Abstract.html)：研究行为策略与辅助预测共同学习时的非平稳性，以及 successor features 的作用。

- [Modayil & Abbas · Towards model-free RL algorithms that scale well with unstructured data（2023）](https://arxiv.org/html/2311.02215v1)：Nibbler 的问题选择、局部输入选择与特征复用；算法 1–5 给出完整更新次序。

- [Voelcker et al. · When does Self-Prediction Help?（RLC 2024）](https://openreview.net/forum?id=izAJ8sHF5q)：比较观测重构与潜在自预测在独立表示学习和辅助 TD 学习中的不同作用。

- [Understanding Auxiliary Tasks · 作者项目说明](https://github.com/adaptive-agents-lab/understanding_auxiliary_tasks)：对应 RLC 2024 论文；仓库目前只有项目说明，未提供可运行实验源码。论文中的线性分析和神经网络实验不能视为已在此仓库公开实现。

- [Sutton et al. · Horde (2011)](https://sites.ualberta.ca/~amw8/horde.pdf)：原始 question/answer 分工、共享经验架构与停止信号约定。

- [Sutton et al. · Fast gradient-descent methods for TD learning (2009)](https://doi.org/10.1145/1553374.1553501)：GTD2 与 TDC 的目标和更新；注意两者不是逐次相同。

- [Sutton, Mahmood & White · An Emphatic Approach (2016)](https://www.jmlr.org/papers/v17/14-488.html)：interest、follow-on、emphasis 与线性稳定性；随机收敛需相应附加条件。

- [Sutton & Barto · 第 9、11、12 章](http://incompleteideas.net/book/the-book-2nd.html)：函数逼近、离策略预测、资格迹的完整教材背景。

- [RLPark · GTDLambda.java 固定版本](https://github.com/rlpark/rlpark/blob/2baf7389c75a31831e5b1a68539771c1947b7f09/rlpark.plugin.rltoys/jvsrc/rlpark/plugin/rltoys/algorithms/predictions/td/GTDLambda.java)：对照 update 中 $\gamma$t / $\gamma$t+1、$\rho$、correction 与旧辅助参数。Java 使用 v 为主权重、w 为辅助权重；本章使用 w、h。

- [RLPark · Horde.java](https://github.com/rlpark/rlpark/blob/2baf7389c75a31831e5b1a68539771c1947b7f09/rlpark.plugin.rltoys/jvsrc/rlpark/plugin/rltoys/horde/Horde.java)：看调度器怎样把同一 transition 分发给不同 demon；该实现面向原机器人系统。

- [GVFN · 作者代码](https://github.com/mkschleg/GVFN)：问题参与递归状态之后，需要额外处理表示与预测依赖。

- [GVFHordes.jl · 作者问题库](https://github.com/mkschleg/GVFHordes.jl)：按 cumulant / discount / policy 组织问题接口；可与本页 question 函数逐项对应。

- [作者原文](https://arxiv.org/abs/2209.14935)：2022 首稿，ICLR 2023；比较奖励表示、SF 与 FB。

- [作者研究平台](https://github.com/facebookresearch/controllable_agent)：README 直接关联两篇 FB 论文；该仓库已经归档。

- [ICLR 2026 正式原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48acf4b231771e693f42305b4c9b4c9f-Abstract-Conference.html)：第 2–3 节和算法附录；区分 flow 时间、环境时间与近似 SR 联系。

- [原文链接的作者实现](https://github.com/Shiien/successor-flow-representation-implementation)：SAC/TD3、flow 特征、对照和 sweep 配置。

- [NeurIPS 2020 原文](https://papers.nips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)：VE 依赖策略与函数集合。

- [Proper Value Equivalence · NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/400e5e6a7ce0c754f281525fae75a873-Abstract.html)：多步算子、固定点与规划充分性；不是任意潜在网络的保证。

- [Barreto et al. · Successor Features for Transfer in Reinforcement Learning](https://arxiv.org/abs/1606.05312)：有限奖励特征、固定策略 SF 与 GPI 的经典机制桥梁。

- [Touati & Ollivier · Learning One Representation to Optimize All Rewards](https://arxiv.org/abs/2103.07945)：FB 表示的理论出发点；探索/经验覆盖、近似误差及奖励查询约定。

- [Sutton, Bowling & Pilarski · The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173)：状态、预测、时间抽象与规划的研究纲领，不是完成全闭环的报告。


---

# 持续控制：比较策略与学习智能体

一个智能体当前做得好，不代表它以后仍能学得好。持续控制要评价完整的行动—学习过程：行动改变世界和数据，学习改变后续行动，有限记忆与计算又限制了这个过程。本章从这些依赖出发，定义可以比较的对象，并用可解析反例检验不同评价标准。

## 本章内容

- 区分固定策略、历史依赖智能体、学习规则与外部设计者，写出完整交互时序。
- 说明价值在持续学习中如何定义，以及当前冻结策略的价值遗漏了什么。
- 逐步计算短期与长期排序反转、相同策略的不同学习能力、不可逆后果三个反例。
- 区分可实现比较者、知晓未来的 oracle、偏离遗憾与从初始世界出发的比较。
- 运行精确枚举、重要性采样和有限差分测试，为自己的 CRL 实验写明目标、信息与资源边界。

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

3. [估计并检验比较边界](#control-deviation)：因为每条真实历史只经历一个分支，偏离估计要检查行为支持；以不可逆反例检验它遗漏的初始决策后果。

结论与条件：给定世界和完整算法时可定义可积条件价值；重要性采样正确性依赖支持与概率。反例和局部估计器不证明通用CRL算法最优。

### 相关方法改变了什么

- 冻结策略评价/GPI：固定后续行为或逐次评价改善，遗漏后续学习能力。

- 实际历史偏离比较：从已发生的世界条件评价可行偏离，无法自动反映过去毁坏。

- 初始世界生命期比较：保留行动造成的长期分支，需要独立可重复世界或额外识别假设。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 期望与条件信息

$\mathbb E[X\mid h]$ 表示已经知道历史 $h$ 后，对仍未知的后果取平均。不同条件信息对应不同的预测问题。

### 策略与学习规则

策略决定现在怎样行动；学习规则决定收到经验后，未来的策略怎样改变。学习率、优化器状态、记忆与表示更新都会影响后者。

### 一次环境转移

时刻 $t$ 的智能体先选 $A_t$，随后收到 $R_{t+1},O_{t+1}$，再更新内部状态。动作不能使用尚未收到的后果。

<a id="lesson-setting"></a>

## 1 · 持续控制的对象

普通策略评价提出一个明确的反事实问题：如果从这里开始，一直按给定策略 $\pi$ 行动，会得到什么？持续学习提出的另一个问题是：如果从这里开始，继续运行这个学习智能体，会得到什么？两者都可以有数学定义，但它们一般不是同一个量。第二个问题包括以后会发生的探索、参数更新、表示变化、记忆淘汰与技能获取。

例如，两个机器人现在都选择同一条路线。其中一个会从故障中更新模型，另一个的更新已经停止。当前动作分布无法区分它们，但后续表现可能截然不同。反过来，两个当前价值相近的机器人也可能因记忆容量、更新延迟或探索权限不同，获得不同的未来经验。

| 比较对象 | 固定的部分 | 允许变化的部分 |
| --- | --- | --- |
| 冻结策略 | 当前策略映射及其参数 | 世界状态、由行动生成的轨迹 |
| 持续学习智能体 | 完整算法与初始化协议 | 参数、内部记忆、行为和访问分布 |
| 算法设计方案 | 资源与信息约束、测试协议 | 设计者在开发阶段选择的结构与超参数 |

持续控制不等于把一个静止任务训练更久，也不等于只考察环境变化。即使外界规律固定，智能体也可能因容量有限而不断改写知识；即使外界变化，有限任务集合也可能被一次记住。这里关注的是完整生命周期中，行动与学习怎样共同影响表现。具体任务是否要求永不停止学习，还取决于采用的形式化定义。

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

“当前策略的价值不够”不等于“CRL 不能定义价值”。只要目标、未来行为规律和条件信息明确，且回报可积，就能定义条件期望。问题在于究竟评价冻结策略，还是评价持续适应的智能体，以及所使用的状态是否足以表达这个条件。

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

传统 GPI 的评价—改善循环是有用的算法组织方式，但它不直接评价学习规则未来的适应能力。详细的策略迭代、价值迭代及采样控制推导属于经典控制分册。本章继续保留期望回报的严格定义，同时把评价对象扩展到实际运行的学习过程。

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

从时间零折扣意味着很晚发生的适应按 $\gamma^t$ 降权。另一种选择是在每个实际时刻都比较未来 $H$ 步，再对这些时刻等权平均。它关注沿途各局部世界中的持续改进，不等同于初始世界的终身总回报。不能因为两者都写有“长期”就混用其结论。

$$
\mathcal A\in\mathfrak A_{B,C},\qquad
 |Z_t|_{\rm bytes}\le B,\quad
 \operatorname{cost}(U_t)\le C
$$

资源约束定义可实现的算法类。B 可限制持久记忆，C 可限制每次真实交互允许的计算或延迟。实际研究应给出对应的计量方式，而非只限制网络参数量。

探索会付出低奖励或风险成本；表示与策略更换可能需要停机或校准；规划消耗真实时间。若这些成本会影响任务，应放入奖励、单独约束或并列指标中，并说明选择理由。平均奖励相同也可能有不同的启动损失和恢复时间。增益与差分价值的区分将在平均奖励章推导；这里只强调：没有脱离目标与约束的统一“最好算法”。

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

考虑奖励刚刚切换的一状态、两动作环境：动作 0 的奖励现在为 0，动作 1 为 1。两个智能体都保存 $Q_0=(1,0.5)$，都按最大 Q 贪心行动，并在并列时选动作 0。它们当前的动作分布完全相同，差别只有收到奖励后的更新规则。

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

如果冻结两个智能体当前的策略，未来都会一直选 0，价值相同且为零。继续运行学习规则时，它们的价值不同。因此当前冻结策略价值不能完整表征未来可学习性。这个例子不证明越大学习率越好：若奖励有噪声或旧情境会回来，还需要比较方差、遗忘和重学成本。它证明的是比较对象必须包含更新规则。

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

控制研究常把算法表现减去某个参考表现，称为遗憾或性能差距。参考对象必须单独定义：它与智能体有相同信息吗？能看未来吗？它在自己的轨迹上行动，还是在智能体实际到达的历史上作局部比较？这些选择决定差距意味着什么。

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

两项从原智能体实际经历的同一历史出发，各自向未来继续 H 步。偏离之后，后续动作、世界状态、收到的数据和学习过程仍会共同变化。只有比较起点固定，未来轨迹没有被强制固定。正值表示这个偏离可以改善表现。

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
| 真实单生命期 | 真实部署历史与干预协议 | 允许的因果学习和行动 | 实际经历；未观察后果需额外识别假设 |

独立模拟世界允许从相同初始分布运行不同算法。相同随机种子可以安排共同外生随机数，但不同动作仍可能引向不同状态；不能要求两个闭环算法收到完全相同的后续观测，再声称测到了探索能力。真实世界通常只能经历其中一个分支。日志重加权也不能凭空修复支持缺失、隐藏干预或错误的世界假设。

完整生命周期比较应保留探索损失、切换期间的低谷和永久失败。报告累计与分段收益、响应和恢复时间、旧能力保持、不可恢复失败率，以及记忆、梯度更新数、模型查询和真实时间。某项诊断改善但总体收益不变，是有效的机制证据或限制，而不是必须隐藏的结果。

开发阶段可以用声明的世界和预算选择算法。封存后，测试期间保持算法设计不变，但其内部学习继续运行。不要把“封存算法”误解为冻结所有参数。Mesbahi 等人的 lifetime tuning 立场论文专门指出：设计者遍历完整生命期后再挑超参数，会利用本应属于未来的信息，并可能改变算法排名。使用开发前缀或独立开发世界时，要同时报告其数据成本与与测试世界的关系。

不同形式化回答不同问题。Abel 等人用 agent basis 描述隐式策略搜索，并相对于该 basis 定义最优智能体必须持续学习的问题；Elelimy 等人强调学习规则、实际历史与偏离比较；非平稳遗憾理论用受限制的环境变化类研究学习代价；单生命期协议强调部署时不能靠人工重置恢复。这些视角可以互补，但不能省掉条件后合并成一个通用定理。

<a id="research-reset-control-protocol"></a>

## 研究专题 A · 重置是转移、动作还是外部资源？

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

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

Laplacian 描述行为图结构，奖励感知表示加入路径价值，METRA 学习有区别的行为，HIQL 利用离线目标轨迹，MaestroMotif 引入语言先验。它们承担不同的设计工作；生成技能、选择技能与组合技能需要分别评价。

- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)
- [Posterior Sampling for Continuing Environments](https://yingwen.io/zh/continual-rl/research/#recent-cpsrl-continuing-exploration)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

模型可提取性的理论说明某类能力需要什么知识，不指定唯一网络。Dreamer 研究潜在想象控制，STOMP 研究随机时长行为模型，DRAGO 研究旧模型知识保留。模型误差、查询策略和规划收益之间仍需实验连接。

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


---

# 深度价值学习：DQN 与 Double DQN

DQN 用神经网络近似动作价值。在共享参数下，一次更新会影响多个状态；自举目标又依赖当前估计。经验回放与目标网络分别调整数据使用方式和目标变化速度，Double DQN 则分开动作选择与动作评估。

## 本章内容

- 能从 Bellman 最优方程写出 DQN / Double DQN 的 target 与半梯度。
- 实现两层 ReLU Q 网络及其反向传播。
- 知道 buffer、更新比率、目标滞后和非平稳环境之间的冲突。

<a id="problem-definition"></a>

## 本章的问题定义

离散动作折扣控制中，以共享神经网络估计动作价值；数据行为、回放分布和目标网络具有不同时间尺度。

### 给定条件与符号

- 固定Markov任务、奖励、真实终止语义和可选离散动作。
- 网络、回放容量、采样规则、目标同步、探索及更新预算。

### 需要求解的对象

可产生高回报动作的近似最优动作价值；每批训练只拟合给定Bellman标签。

### 信息与数据权限

行为策略收集 $(s,a,r,s',d)$，$d$ 只表示真实终止；回放分布 $D$ 决定本批样本，在线参数 $\theta$ 与目标参数 $\theta^-$ 的更新时间各自规定。

$$
Q^*(s,a)=\mathbb E\!\left[R+\gamma(1-d)\max_{a'}Q^*(S',a')\mid s,a\right]
$$

$R,S'$ 是真实条件后果，$0\le\gamma<1$。这是理想最优价值固定点；本批DQN损失是 $\tfrac12\mathbb E_D[(Q_\theta(s,a)-\operatorname{sg}(Y))^2]$，$Y$ 为旧目标网络构造的标签，$\operatorname{sg}$ 表示停止梯度。二者不是同一个优化问题。

### 成立条件与解的含义

- 基础题目固定且状态Markov；数据必须覆盖决策所需动作与状态。
- 任意非线性网络、回放与自举的组合没有本章给出的全局收敛保证。

判断准则：两状态解析问题上核对最优价值[[0.9,0.1],[1,−1]]、终止标签和网络梯度；复杂任务用独立行为收益并记录真实步、梯度步和回放年龄。

### 适用边界

- Double DQN不保证完全消除高估或总有更高回报。
- batch size为1不构成严格流式协议。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 限制表示或采用近似 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：DQN解决固定离散折扣任务的局部价值控制，并不完整评价持续学习过程。

- 组合不同学习问题 · [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)：回放重用经验，但是否保留未来需要的旧知识还需历史采样与回访评价。

- 组合不同学习问题 · [可塑性与特征更新](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)：共享网络长期可学习性是额外问题，较低Bellman标签损失不能诊断全部退化。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

共享估计同时改变多个输入，目标又依赖价值估计，产生相关样本与追逐标签的反馈。

### 本章的核心思路

分别控制数据重用、目标移动和选择—评估耦合，不把三个机制混成一项收敛保证。

1. [从最优固定点构造冻结标签](#lesson-derive)：因为网络不能直接枚举真实期望，用目标网络生成本批Bellman标签并停止其梯度。

2. [分开动作选择和评估](#lesson-derive)：因为最大值会偏爱估计偏高的动作，Double DQN用在线网络选、目标网络评估；两网络仍可能相关。

3. [验证共享梯度与三种时钟](#lesson-code)：因为一处参数更新影响多个输出，先检查固定标签梯度，再接回放、真实交互和目标同步循环。

结论与条件：精确有限折扣Bellman算子有唯一固定点；这不构成神经DQN训练的收敛证明，目标网络与回放是有限协议下的稳定化机制。

### 相关方法改变了什么

- 表格Q-learning：独立参数消除共享逼近干扰，但不能扩展到任意高维输入。

- DQN：目标网络同时选与评估下一动作。

- Double DQN：分开选择与评估来源，减少一类最大化偏差而不消除所有误差。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 一条经验

在 $S_{t}$ 做 $A_{t}$ 后得到 $R_{t+1}$、$S_{t+1}$。奖励属于这次转移，不是到达状态之前另一轮的奖励。

### 折扣与终止

$\gamma$ 控制未来相对权重；真实终止后的价值设为 0。本章记 $\gamma_{t+1}$=$\gamma$(1−terminated)。训练脚本的时间截断不一定是任务终止。

### 表格与函数逼近

表格每个状态或状态动作一组独立参数；函数逼近让不同输入共享参数。更新一个输入可能改变其他输入的预测。

### 链式法则与 stop-gradient

同一个数可以参与前向计算但不参与本次求导。DQN 把本次目标当作固定标签；这不是遗漏反向传播。

<a id="lesson-setting"></a>

## 1 · 学习对象仍然是最优动作价值

先讨论离散动作、固定 MDP、折扣回报。$Q_\theta(s,a)$ 输出每个动作的值。参数共享使一次更新能泛化到未访问输入，但也会改变过去已经正确的值。环境经验来自当前 $\epsilon$-greedy 策略，训练 minibatch 来自 replay 分布，目标又来自另一个滞后的参数快照 $\theta^-$。三种时间尺度同时存在。

| 持久对象 | 存什么 | 何时改 |
| --- | --- | --- |
| 在线网络 $\theta$ | 当前动作价值 | 每次梯度更新 |
| 目标网络 $\theta^-$ | 较慢移动的 target | 每 C 步硬复制，或明确的软更新 |
| replay D | 真实经验及终止语义 | 每步加入、按容量淘汰 |
| 优化器状态 | 动量、二阶矩等 | 每次参数更新；本章用 SGD 便于检查 |

本章不把 replay 称为 CRL 的完整记忆方案。均匀缓冲区回答的是“从现有数据分布抽样”，不是“哪些知识以后仍值得保留”。

<a id="lesson-derive"></a>

## 2 · Bellman target 到半梯度，再到 Double DQN

$$
Y=r+\gamma(1-d)\max_{a\prime}Q_{\theta^-}(s\prime,a\prime),\qquad L(\theta)=\frac1B\sum_{i=1}^B\frac12[Q_\theta(s_i,a_i)-\operatorname{sg}(Y_i)]^2
$$

d 仅表示真实任务终止。目标网络不参与梯度；经验分布也被当作这次优化给定的数据。网络训练的是拟合一批 Bellman target 的代理问题，不是直接最小化全局真实价值误差。

$$
\theta\leftarrow\theta+\frac{\alpha}{B}\sum_i[Y_i-Q_\theta(s_i,a_i)]\nabla_\theta Q_\theta(s_i,a_i)
$$

从平方损失求导即可得到。若使用 Huber loss，大误差区的导数会截到常数幅度；这改变鲁棒性与梯度尺度，不改变 target 定义。本教学实现用半平方误差，便于有限差分。

$$
a^*=\operatorname{argmax}_a Q_\theta(s\prime,a),\qquad Y^{\mathrm{Double}}=r+\gamma(1-d)Q_{\theta^-}(s\prime,a^*)
$$

DQN 同一目标网络既选动作又评估；Double DQN 用在线网络选择、目标网络评估。两网络仍相关，不能声称完全消除高估或保证性能更好。

为什么目标要慢一点？若预测和 target 同时追着彼此跑，当前网络误差会立刻成为新的标签。目标网络暂时固定这层反馈；replay 则减弱相邻经验相关性并重用样本。二者是实践稳定化手段，不是对任意非线性网络收敛的证明。

<a id="dqn-derivative"></a>

## 3 · 两层网络的梯度推导

$$
h=\operatorname{ReLU}(W_1x+b_1),\quad q=W_2h+b_2,\quad e=q_a-Y
$$

只对实际采取动作 a 的输出拟合 target；其余输出的直接梯度为零，但共享隐藏层更新后可能间接改变它们。

$$
\frac{\partial L}{\partial W_{2,a:}}=e h^\top,\quad \frac{\partial L}{\partial b_{2,a}}=e,\quad g_h=eW_{2,a:}^\top,\quad g_z=g_h\odot\mathbf1_{W_1x+b_1>0},\quad \frac{\partial L}{\partial W_1}=g_zx^\top
$$

所有右侧必须使用同一次前向传播的旧参数。若先改 $W_{2}$ 再用新 $W_{2}$ 算 $W_{1}$ 梯度，就不再是所写损失的梯度。ReLU 恰好为 0 不可微；测试避开拐点，用常规子梯度 0。

<a id="lesson-example"></a>

## 4 · 一条样本，三个容易错的 target

当前 Q(s,a)=2，r=1，$\gamma$=.9；下一状态在线输出 [3,2]，目标输出 [1,4]。DQN target=1+.9×4=4.6；Double DQN 在线选动作 0，目标评估为 1，因此 target=1.9。它们给出相反更新方向完全可能。

若此转移真实终止，两个 target 都是 1，且根本不必调用下一状态网络；若只是采样器时间上限，任务本身还能继续，则要从真实最后观测 bootstrap。不能误用自动 reset 后新回合的初始观测。

把 r 改成 10 后，TD error 增大并不自动意味着环境发生了变化：也可能只是首次访问高奖励状态。诊断变化需要固定条件或独立检测设计。

<a id="lesson-code"></a>

## 5 · 核心实现：网络、反向传播、回放与目标同步

**算法：DQN 与 Double DQN；每次运行选择一种目标**

1. 初始化在线网络 $Q_\theta$、目标网络 $Q_{\bar\theta}$ 与回放池 $\mathcal D$。
1. 每个真实步：
  1. 按 $\epsilon$-greedy 策略行动，将 $(s,a,r,s',d)$ 加入 $\mathcal D$。
  1. 抽取小批量转移，并使用更新前的网络计算固定目标：
    1. DQN：$Y=r+\gamma(1-d)\max_bQ_{\bar\theta}(s',b)$
    1. Double DQN：$a^*=\arg\max_bQ_\theta(s',b)$，$Y=r+\gamma(1-d)Q_{\bar\theta}(s',a^*)$
  1. 对 $\tfrac12\operatorname{mean}[(Q_\theta(s,a)-\operatorname{sg}(Y))^2]$ 做一次梯度下降。
  1. 每隔给定的真实步数，复制 $\bar\theta\leftarrow\theta$。
  1. 真实终止时按协议重置环境，保留网络和回放池。

无需深度框架也能运行的两层 ReLU DQN / Double DQN

```python
class TinyQ:
    """Two-layer ReLU Q-network: 2 state features -> 8 hidden -> 2 actions."""
    def __init__(self, rng, hidden=8):
        self.w1 = [[rng.uniform(-0.5, 0.5) for _ in range(2)] for _ in range(hidden)]
        self.b1 = [0.1] * hidden
        self.w2 = [[rng.uniform(-0.2, 0.2) for _ in range(hidden)] for _ in range(2)]
        self.b2 = [0.0, 0.0]

    def forward(self, state):
        # An integer state is a one-hot input; terminal state is never evaluated.
        hidden = [max(0.0, row[state] + b) for row, b in zip(self.w1, self.b1)]
        q = [sum(w * h for w, h in zip(row, hidden)) + b
             for row, b in zip(self.w2, self.b2)]
        return hidden, q

    def gradient(self, state, action, target):
        hidden, q = self.forward(state)
        error = q[action] - target  # gradient of 1/2 * squared error
        g = {"w1": [[0.0] * 2 for _ in hidden], "b1": [0.0] * len(hidden),
             "w2": [[0.0] * len(hidden) for _ in range(2)], "b2": [0.0, 0.0]}
        g["b2"][action] = error
        for j, h in enumerate(hidden):
            g["w2"][action][j] = error * h
            dh = error * self.w2[action][j] * (h > 0.0)
            g["w1"][j][state] = dh
            g["b1"][j] = dh
        return g

    def train_batch(self, frozen_samples, alpha):
        # All gradients use the SAME pre-update parameters.
        gradients = [self.gradient(s, a, y) for s, a, y in frozen_samples]
        for name in ("w1", "w2", "b1", "b2"):
            param = getattr(self, name)
            for i in range(len(param)):
                if isinstance(param[i], list):
                    for j in range(len(param[i])):
                        param[i][j] -= alpha * sum(g[name][i][j] for g in gradients) / len(gradients)
                else:
                    param[i] -= alpha * sum(g[name][i] for g in gradients) / len(gradients)


def dqn_target(online, target, reward, discount, next_state, double):
    if discount == 0.0:
        return reward  # do not even index a terminal observation
    _, target_q = target.forward(next_state)
    if double:
        _, online_q = online.forward(next_state)
        selected = max(range(2), key=lambda a: online_q[a])
        return reward + discount * target_q[selected]
    return reward + discount * max(target_q)


def train_dqn(steps=5000, seed=7, double=True):
    rng = random.Random(seed)
    online = TinyQ(rng)
    target = copy.deepcopy(online)
    replay, state = [], 0
    for t in range(steps):
        _, q = online.forward(state)
        action = rng.randrange(2) if rng.random() < 0.2 else max(range(2), key=lambda a: q[a])
        if state == 0 and action == 0:
            reward, discount, nxt = 0.0, 0.9, 1
        else:
            reward = 0.1 if state == 0 else (1.0 if action == 0 else -1.0)
            discount, nxt = 0.0, None
        replay.append((state, action, reward, discount, nxt))
        replay = replay[-256:]
        if len(replay) >= 16:
            batch = rng.sample(replay, 16)
            frozen = [(s, a, dqn_target(online, target, r, g, sp, double))
                      for s, a, r, g, sp in batch]
            online.train_batch(frozen, alpha=0.03)
        if (t + 1) % 25 == 0:
            target = copy.deepcopy(online)
        state = 0 if discount == 0.0 else nxt
    estimates = [online.forward(s)[1] for s in range(2)]
    reference = [[0.9, 0.1], [1.0, -1.0]]
    return {"Q": estimates, "reference": reference,
            "max_error": max(abs(estimates[s][a] - reference[s][a]) for s in range(2) for a in range(2))}
```

环境只有两个非终止状态。状态 0 动作 0 以奖励 0 到状态 1，动作 1 以 .1 终止；状态 1 两动作分别以 +1/−1 终止。最优 Q 为 [[.9,.1],[1,−1]]。每轮真实交互写入容量 256 的 buffer，抽 16 条经验，先冻结所有 target，再基于同一旧网络求 minibatch 梯度；每 25 个真实步同步目标。

输入用 one-hot 不代表表格算法：八个 ReLU 隐藏单元和两个输出共享参数。实验用中心有限差分检查网络梯度，以解析动作价值检查训练结果。环境只有两个非终止状态，适合观察目标网络、回放和梯度更新的作用。

- 读输出先检查 `max_error`，再改变 double、target 同步频率、buffer 容量。每次只改一项。
- 若加入奖励切换，记录 buffer 样本的年龄和切换前后比例；不要只报告容量。
- 比较更多梯度更新是否有利时，同时报告环境步数与梯度步数。

<a id="lesson-branches"></a>

## 6 · DQN 家族的不同改动在解决什么

| 分支 | 改什么 | 并不自动解决 |
| --- | --- | --- |
| Double DQN | 动作选择与评估耦合 | 持续漂移、可塑性 |
| Dueling network | V 与 advantage 的输出分解 | 任务身份、记忆 |
| Prioritized replay | 采样更高 TD error 的经验并校正权重 | 高误差是否只是噪声 |
| n-step / distributional RL | 目标传播距离 / 回报分布表示 | 任意环境变化 |
| ReDo / resets / continual backprop | 长时间训练后的特征与优化状态 | 旧知识保留与安全 |
| Recurrent DQN | 把历史编码成决策状态 | replay 中隐状态是否过时 |

Rainbow 将若干改动组合，在其任务与预算中评估。研究 CRL 时不应把“使用 Rainbow”当作所有机制已处理：对变化场景，旧缓冲区、目标网络、特征和状态都可能以不同速度过时。

<a id="lesson-check"></a>

## 7 · 怎样知道是实现错还是学习困难？

先检查网络能否过拟合固定 target 的一小批监督样本；再固定目标网络检查 Bellman target；再加入 replay 与环境循环。如果连固定数据回归都失败，就不应把结果解释为 CRL 可塑性损失。

为何 batch size=1 不等于严格 streaming？因为样本仍可能来自旧 replay，并可能对同一经验重复训练。数据权限、重用次数、目标快照和每步延迟都需单独描述。

## 本章的实验设计

分别检查终止处理、目标网络和梯度方向。通过这些测试后，再比较跨种子的回报。

设定：固定小批次：在线后继 Q=(3,2)，目标后继 Q=(4,20)，奖励 1，γ=0.9；另加入真实 terminal 和外部 truncation。

- Double Q 选择动作 0，并得到目标 4.6。
- 目标网络不从该损失收到非预期梯度。
- 自动重置时 bootstrap 使用最终有效观测而非新初态。

对照：普通 DQN 与 Double DQN 的同基座核；相同 replay、网络及更新比；各完整方法的独立合理调参

记录：逐样本 target、TD 残差和梯度；在线/目标参数版本与同步事件；回放访问、梯度步和独立 run 回报

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-deep)

## 学习与研究衔接

DQN 保留 TD 目标。网络、回放与目标网络增加新的时间尺度，长期训练时也可能引入陈旧数据与可塑性问题。

[分册导读](https://yingwen.io/zh/continual-rl/start/deep-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-deep-value) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=deep-value) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=deep-value)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

状态是支持后续计算的内部信息；GVF 指定一个预测问题；RTRL 和资格迹规定信用如何传播。三者可以组合，但不是相互替代的算法名称。先理解给定策略的预测，再讨论预测怎样改善控制。

- [Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations](https://yingwen.io/zh/continual-rl/research/#recent-successor-flow-features)

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [Safe and Efficient Off-Policy Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-retrace-safe-offpolicy)
- [Convergent Tree Backup and Retrace with Function Approximation](https://yingwen.io/zh/continual-rl/research/#recent-convergent-tree-retrace)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

模型可提取性的理论说明某类能力需要什么知识，不指定唯一网络。Dreamer 研究潜在想象控制，STOMP 研究随机时长行为模型，DRAGO 研究旧模型知识保留。模型误差、查询策略和规划收益之间仍需实验连接。

- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://yingwen.io/zh/continual-rl/research/#recent-tdmpc2-decision-time-model)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Revisiting Adam for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-revisiting-streaming-adam)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [The Dormant Neuron Phenomenon in Deep Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-redo-dormant-neurons)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [Revisiting Adam for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-revisiting-streaming-adam)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文提供定义和条件，架构讲座提出模块组织，算法论文检验特定机制。完整系统还要明确智能体、外部设计者和世界各自承担的工作；组件成立不自动意味着组合后的长期收益成立。

- [Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations](https://yingwen.io/zh/continual-rl/research/#recent-successor-flow-features)

### The Dormant Neuron Phenomenon in Deep Reinforcement Learning

Ghada Sokar, Rishabh Agarwal, Pablo Samuel Castro, Utku Evci

ICML 2023 · 2023 · 支持方法与理论

#### 研究问题

网络参数数量没有变，为什么越来越多隐藏单元不再对输出产生有效贡献？

#### 关键机制

ReDo 用相对激活量识别低活跃单元，重新初始化其输入连接，并处理输出连接，使被回收单元可以重新参与学习。它针对的是可用表示容量，而不是直接惩罚旧任务表现变化。

#### 证据

论文记录深度 RL 中的休眠单元现象，并比较回收机制对多个任务学习的影响。实现进入作者所在团队的 Dopamine 代码库。

#### 条件与限制

低激活只是可塑性问题的一种诊断，不能覆盖曲率变化、优化器状态和负迁移。回收也可能损坏低频但重要的旧知识，需要与保留指标共同评价。

#### 阅读与实验

同时记录休眠比例、新目标拟合速度与旧任务冻结表现。三者发生不同方向变化时，不要用单个表示指标替代整个持续学习结论。

#### 原文与相关入口

- [ICML 2023 原文](https://proceedings.mlr.press/v202/sokar23a.html)：休眠定义、回收规则与实验。
- [Dopamine ReDo 实现](https://github.com/google/dopamine/tree/master/dopamine/labs/redo)：作者团队公开代码中的 ReDo 模块。

#### 作者代码

[论文作者团队发布的实现，不是本教材的简化版本。](https://github.com/google/dopamine/tree/master/dopamine/labs/redo)

Dopamine 中的 ReDo 神经元回收与实验实现。

### Revisiting Adam for Streaming Reinforcement Learning

Florin Gogianu, Luțu Adrian-Cătălin, Razvan Pascanu

RLC 2026 / RLJ 预会议版 · 2026 · 支持方法与理论

#### 研究问题

流式 RL 的不稳定来自 Adam 本身，还是目标导数、方差与超参数的组合？

#### 关键机制

论文重新分析自适应更新的信噪比，将 Adam 的稳定项与目标导数尺度联系起来，并研究有界导数的回报分布学习及多步更新。它改变的是目标与更新的配合，而非简单沿用批量训练时的默认配置。

#### 证据

作者在大规模 Atari 流式实验中展示了具有竞争力的结果，并重新比较早期流式方法。正式 RLJ 入口收录为 RLC 2026 预会议论文。

#### 条件与限制

主体实验采用经典回合式 Atari 的流式学习协议，不是任意非平稳终生适应的证据。这些结果也不否定归一化、资格迹或更新约束在其他任务中的价值。版本、调参预算和目标分布必须对齐。

#### 阅读与实验

建立二维对照：固定目标换优化器，固定优化器换目标。将调参种子与最终测试分开，再判断改进来自哪一个因素。

#### 原文与相关入口

- [RLC 2026 论文入口](https://rlj.cs.umass.edu/2026/papers/Paper131.html)：会议收录信息与论文。
- [作者预印本](https://arxiv.org/abs/2605.06764)：Adam 尺度分析、回报分布目标与实验协议。

### Safe and Efficient Off-Policy Reinforcement Learning

Rémi Munos, Tom Stepleton, Anna Harutyunyan, Marc G. Bellemare

NeurIPS 2016 · 2016 · 支持方法与理论

#### 研究问题

目标与行为策略不一致时，如何保留多步信用而避免重要性比率乘积爆炸？

#### 关键机制

统一多步目标为目标策略TD误差的加权和，Retrace采用λmin(1,π/μ)传播系数。近同策略时保留长迹，目标概率较低的动作则减少传播；一步误差仍使用目标动作期望。

#### 证据

论文分析表格算子的收缩性质，给出条件下的评价与控制收敛，并报告Atari实验。信用章独立检查传播系数和有限轨迹恒等式。

#### 条件与限制

表格安全性不是任意线性或神经逼近的稳定性保证。行为覆盖、变化策略与投影条件仍需检查；代码小实验不复现Atari。

#### 阅读与实验

在同样轨迹与表示上，分别改变策略差异和动作随机性，比较Tree-backup与Retrace的信用长度、方差和预测误差。

#### 原文与相关入口

- [原论文](https://arxiv.org/html/1606.02647)：统一算子、传播系数及理论条件。

### Convergent Tree Backup and Retrace with Function Approximation

Ahmed Touati, Pierre-Luc Bacon, Doina Precup, Pascal Vincent

ICML 2018 · 2018 · 支持方法与理论

#### 研究问题

传播系数已经截断，为什么函数逼近下的Tree-backup和Retrace仍可能发散？

#### 关键机制

分析函数逼近与off-policy多步bootstrap的学习算子，展示线性反例，再把相应目标写成二次凸凹鞍点问题，构造梯度版本。

#### 证据

原文给出线性不稳定例子、梯度方法收敛保证与有限样本界。它直接限定了从Retrace表格结论外推到逼近算法的范围。

#### 条件与限制

凸凹线性问题的保证不能自动覆盖学习表示的深度网络。稳定目标、更新速度与控制性能还需分别验证。

#### 阅读与实验

先检查固定表示下的期望更新矩阵，再将半梯度和梯度版本按相同样本、步数与计算预算比较。

#### 原文与相关入口

- [ICML原文](https://proceedings.mlr.press/v80/touati18a.html)：理论反例、鞍点方法和保证条件。

### Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations

Haosen Shi, Jianda Chen, Sinno Jialin Pan

ICLR 2026 · 2026 · 支持方法与理论

#### 研究问题

能否直接学习多步未来状态的分布，并把它压缩为适合控制学习的特征？

#### 关键机制

SF² 以 flow matching 估计 successor measure，将条件向量场分解为未来位置及生成时间的投影与当前状态动作特征的乘积。特征进入 TD3/SAC 的 critic；线性的是向量场对条件特征的分解，critic 本身可以非线性。

#### 证据

正式原文给出 mixture Bellman 结构、生成式 bootstrap 与控制实验，并提供作者 JAX/Brax 仓库。实验研究在线收集数据下的 off-policy 控制，并使用 replay、批次与目标网络。

#### 条件与限制

“online policy learning”不代表 strict streaming。生成时间不是环境时间；向量场线性不保证任意奖励价值线性。文中与 SR 的小生成时间联系是近似动机，未证明递归 agent state 或任意持续变化下的充分性。

#### 阅读与实验

对齐模型调用与梯度预算，拆分直接预测、bootstrap、critic 联合训练。冻结特征后比较线性与非线性读出，再测新奖励和动力学变化，才能检验预测知识的可复用程度。

#### 原文与相关入口

- [ICLR 2026 正式原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48acf4b231771e693f42305b4c9b4c9f-Abstract-Conference.html)：第 2–3 节和算法附录；区分 flow 时间、环境时间与近似 SR 联系。
- [原文链接的作者实现](https://github.com/Shiien/successor-flow-representation-implementation)：SAC/TD3、flow 特征、对照和 sweep 配置。

#### 作者代码

[正式论文摘要直接链接的作者代码。](https://github.com/Shiien/successor-flow-representation-implementation)

基于 JAX/Brax 的 SF² 控制实验；不包含自动 GVF 问题发现或完整持续架构。

### TD-MPC2: Scalable, Robust World Models for Continuous Control

Nicklas Hansen, Hao Su, Xiaolong Wang

ICLR 2024 · 2024 · 支持方法与理论

#### 研究问题

如何让短期动力学与长期价值分工，并在动作选择时继续使用模型？

#### 关键机制

TD-MPC2 学习无需观测 decoder 的潜在动力学、奖励、价值与策略先验。决策时优化有限动作序列，用终点价值补上未展开的后果；执行第一步后，利用新观测重新规划。

#### 证据

正式会议原文报告 104 个在线任务和单一大型多任务智能体的实验。官方仓库包含模型训练与计划接口，适合与 Dreamer 的想象 actor 学习比较计算位置。

#### 条件与限制

跨任务共享超参数和多任务能力不是单条生命流中持续适应的证据。replay、任务条件、模型更新、决策延迟等成本需进入 CRL 协议；长程 critic 错误不能被短期模型精度自动修复。

#### 阅读与实验

固定模型，对比无终点价值、不同 horizon 和不同规划预算；再固定预算比较部署 actor 与决策时搜索。环境变化后同时记录模型校准、critic 误差和恢复收益。

#### 原文与相关入口

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html)：短期预测、终点价值、多任务协议。
- [作者实现](https://github.com/nicklashansen/tdmpc2)：训练、模型与 plan 函数分别阅读。

#### 作者代码

[作者维护的原论文代码。](https://github.com/nicklashansen/tdmpc2)

TD-MPC2 的单任务/多任务训练和决策时规划。


<a id="chapter-code"></a>

## 下载与运行

Python 3.10+，仅标准库。包含两层网络及手写反向传播、replay 与目标网络。实验环境为可解析的小型 MDP。

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

```sh
python3 foundations_detail_lab.py deep-value
python3 foundations_detail_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Mnih et al. · Human-level control through deep RL](https://www.nature.com/articles/nature14236)：DQN 原论文，包含网络结构、输入预处理、经验回放、目标网络与 Atari 训练协议。

- [van Hasselt et al. · Deep RL with Double Q-learning](https://arxiv.org/abs/1509.06461)：追踪选择动作和评估动作分别来自哪套参数。

- [CleanRL · DQN 单文件实现与说明](https://docs.cleanrl.dev/rl-algorithms/dqn/)：可运行的现代复现工程，不是 DQN 原论文作者代码。可对照 replay、final observation、target 同步与训练频率。

- [DeepMind · 原始 DQN 工程](https://github.com/google-deepmind/dqn)：历史作者实现，Torch/Lua 与旧 Atari 依赖；用于核对原方法，运行需要相应历史依赖。


---

# 策略梯度、Actor–Critic 与 PPO

策略梯度直接优化参数化策略的期望回报。REINFORCE 使用采样回报估计梯度；actor–critic 引入价值预测以分配信用；PPO 在旧策略采集的数据上优化一个局部代理目标。

## 本章内容

- 从轨迹概率推出策略梯度与 baseline 的零期望。
- 由 TD error 递推计算 GAE，正确处理终止、截断和 rollout 边界。
- 手算 PPO 的正负 advantage 裁剪，并运行有真实采样循环的最小实现。

<a id="problem-definition"></a>

## 本章的问题定义

直接改善参数化随机策略；策略变化也改变后续数据分布。先以固定有限时域推导，再区分自举优势和旧数据代理。

### 给定条件与符号

- 初始分布、环境接口、有限时域与回报准则。
- 可微策略类、采样长度、优势估计器、优化次数和数据权限。

### 需要求解的对象

策略参数及指定回报目标的梯度估计；critic为估计辅助量，PPO裁剪目标为局部代理。

### 信息与数据权限

$\tau$ 是由策略 $\pi_\theta$ 产生的轨迹。PPO保存采样时旧动作概率，不能在每轮优化中重算分母；时间截断与真实终止分别处理。

$$
\max_\theta J(\theta),\qquad J(\theta)=\mathbb E_{\tau\sim p_\theta}\!\left[\sum_{t=0}^{T-1}R_{t+1}\right]
$$

$\theta$ 为策略参数，$p_\theta$ 为策略诱导的轨迹分布，$T$ 为固定有限时域。本章先用不折扣目标；若改为从起点严格折扣，梯度需相应时间/占用权重。PPO旧数据裁剪代理最大化不等于精确最大化此 $J$。

### 成立条件与解的含义

- 环境与初始分布不依赖策略参数；score求导需可交换期望与求导等正则条件。
- baseline不依赖当前动作且actor将其视为固定权重；近似critic、GAE和数据重用引入的误差分别声明。

判断准则：小bandit上梯度方向与精确期望/有限差分一致，正负优势裁剪分支及GAE边界正确；收益以新交互评估，不由actor loss替代。

### 适用边界

- PPO裁剪不提供所有状态上的硬KL信赖域。
- 局部梯度方向不保证有限大步后回报单调增加。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 限制表示或采用近似 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：参数化策略梯度提供局部策略改善，不穷举完整有限资源学习器。

- 组合不同学习问题 · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)：GAE和多步优势为动作梯度分配时间信用，不是独立控制目标。

- 改变评价目标 · [最大熵控制](https://yingwen.io/zh/continual-rl/algorithms/soft-control/)：相对最大熵控制，本章基本目标只累计外部奖励；若另外加入熵项就改变该基本目标。PPO与SAC的数据协议差异还需另行说明。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

环境奖励不可直接沿动作反传，完整回报的梯度估计又有较大方差；重复优化会离开采样策略。

### 本章的核心思路

先对轨迹概率求导，再用因果性和baseline减少无关噪声，最后明确控制旧数据代理的偏移。

1. [沿概率而非环境奖励求导](#lesson-derive)：因为动作改变轨迹分布，用log-derivative将真实回报转为采样score权重。

2. [用critic与多步优势降低等待](#policy-gae)：因为完整回报长且噪声大，TD与GAE用估计补尾；critic误差和混合长度带来相应偏差。

3. [限制旧数据的局部优化激励](#policy-ppo)：因为新策略会偏离旧采样分布，保存旧概率并按优势符号裁剪PPO代理，再以新交互验证。

结论与条件：精确score和合格baseline保持期望梯度；近似critic/均匀rollout/裁剪代理需各自解释，PPO本章实现没有全局最优或硬信赖域保证。

### 相关方法改变了什么

- REINFORCE：完整采样回报提供梯度权重，等待和方差较大。

- Actor–critic/GAE：以自举价值与多步优势替代完整回报，依赖critic质量。

- PPO/TRPO：分别通过裁剪代理与约束近似管理策略变化，求解成本和保证不同。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 一条经验

在 $S_{t}$ 做 $A_{t}$ 后得到 $R_{t+1}$、$S_{t+1}$。奖励属于这次转移，不是到达状态之前另一轮的奖励。

### 折扣与终止

$\gamma$ 控制未来相对权重；真实终止后的价值设为 0。本章记 $\gamma_{t+1}$=$\gamma$(1−terminated)。训练脚本的时间截断不一定是任务终止。

### 表格与函数逼近

表格每个状态或状态动作一组独立参数；函数逼近让不同输入共享参数。更新一个输入可能改变其他输入的预测。

### 随机策略

$\pi_\theta$(a|s) 是动作概率。二动作可用 p=$\sigma$($\theta$)，softmax 推广到多动作；连续动作通常是参数化概率密度。

### 梯度上升与损失下降

要最大化回报 J，就沿 ∇J 上升；框架常定义 loss=−代理目标再下降，符号必须一致。

<a id="lesson-setting"></a>

## 1 · 先用有限时域把梯度推清楚

设轨迹 $\tau$=($S_{0}$,$A_{0}$,$R_{1}$,…,$S_T$)，环境动力学和初始状态分布不依赖策略参数 $\theta$。先取有限时域不折扣总回报。策略产生数据，因此 $\theta$ 改变的不只是预测，还会改变访问哪些状态与轨迹。网络 loss 的下降不是环境回报上升的直接证据。

$$
J(\theta)=\mathbb E_{\tau\sim p_\theta}[\sum_{t=0}^{T-1}R_{t+1}],\qquad p_\theta(\tau)=p_0(s_0)\prod_t\pi_\theta(a_t\mid s_t)p(s_{t+1},r_{t+1}\mid s_t,a_t)
$$

目标是对自己诱导的数据分布求期望。接下来对该分布求导，而不是把采到的奖励当作可对 $\theta$ 直接反向传播的环境函数。

<a id="lesson-derive"></a>

## 2 · 对数导数、因果性与 baseline

$$
\nabla J=\sum_\tau p_\theta(\tau)R(\tau)\nabla\log p_\theta(\tau)=\mathbb E[\sum_t\nabla\log\pi_\theta(A_t\mid S_t)R(\tau)]
$$

使用 ∇p=p∇log p；环境项与 $\theta$ 无关，只有策略项留下。连续空间将求和换积分，还需要可交换求导与积分等正则条件。

$$
\nabla J=\mathbb E[\sum_t\nabla\log\pi_\theta(A_t\mid S_t)G_t],\qquad G_t=\sum_{k=t}^{T-1}R_{k+1}
$$

当前动作不能影响之前已经发生的奖励。条件在历史上，策略 score 的期望为零，因此可从权重中删掉过去奖励而保持同一期望梯度。

$$
\sum_a\pi_\theta(a\mid s)\nabla\log\pi_\theta(a\mid s)b(s)=b(s)\nabla\sum_a\pi_\theta(a\mid s)=0
$$

任意不依赖当前动作的 baseline 都不改变期望。取 $b=V_\pi$，权重成为 advantage；若 baseline 共享参数，actor 更新中须将 baseline 当作固定权重，不额外沿它求导。

$$
A_\pi(s,a)=Q_\pi(s,a)-V_\pi(s),\qquad \delta_t=R_{t+1}+\gamma_{t+1}V(S_{t+1})-V(S_t)
$$

真实 V 与 on-policy 数据下，TD residual 的条件期望给 advantage。使用近似 V 会引入估计误差；actor–critic 就是在用较低方差、可能有偏的 critic 信号替代完整回报。

若 J 是从固定起点的严格折扣目标，轨迹求和形式中需要相应 $\gamma$^t 权重，或等价地在折扣占用分布下取样。很多实现直接对 rollout 的每个时刻均匀平均，再使用 $\gamma$ 折扣的 critic；这是常见代理方式，不能不说明就称与任意 J 完全同一梯度。

<a id="policy-gae"></a>

## 3 · GAE 把多步 advantage 写成反向递推

$$
\hat A_t=\delta_t+\gamma_{t+1}\lambda b_t\hat A_{t+1},\qquad \hat R_t=\hat A_t+V_{\mathrm{old}}(S_t)
$$

$b_{t}$ 表示后继样本是否仍在同一条轨迹内；真实终止、reset 或采样块边界时置 0，避免把新轨迹的误差接回来。它与控制 bootstrap 的 $\gamma_{t+1}$ 分开。从末尾向前算即可。$\lambda$=0 只用一步 TD，$\lambda$ 接近 1 更多依赖真实后续奖励；终止后 tail 为零。value target 是优势加旧 value，不要把更新中的新 value 混进去。

三个边界必须拆开：真实终止：bootstrap=0 且 trace 停；time-limit 截断：若原任务继续，bootstrap 使用最终观测 value，但 trace 不跨进 reset 的新轨迹；rollout 采样块结束：bootstrap 使用块尾状态 value，未采到的 GAE tail 截断为 0。这些不是一个 done 布尔变量能无歧义表达的。

训练 actor 前，advantage、旧 log-prob 和 critic target 都冻结。本章不强制标准化 advantage；若采用标准化，负正号与尺度可能改变，必须在实现说明中写清。

<a id="policy-ppo"></a>

## 4 · 从重要性比率到 PPO 的单样本分段函数

$$
r_t(\theta)=\exp[\log\pi_\theta(A_t\mid S_t)-\log\pi_{\mathrm{old}}(A_t\mid S_t)],\qquad L^{\mathrm{clip}}=\mathbb E_t[\min(r_t\hat A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)]
$$

比率修正旧状态上的动作分布；它不是完整轨迹重要性采样，也没修正当前策略诱导的新状态分布。PPO 因此是局部代理优化，不能把每次 loss 降低解释成严格策略改善。

| 优势符号 | 停止继续奖励的区间 | 仍然保留梯度的情形 |
| --- | --- | --- |
| A>0 | r>1+$\epsilon$：已经增加够多 | r 太低时继续鼓励恢复 |
| A<0 | r<1−$\epsilon$：已经减少够多 | r 太高时继续惩罚错误增加 |

注意 min 的方向：负 advantage 乘以较大的 ratio 更负。因此不能先把 ratio 双向裁剪再简单乘 A；那会在应该纠错的方向也抹掉梯度。裁剪也不是硬 KL 约束，整个网络共享参数仍可能改变其他动作，通常还要监测 KL 并限制优化轮数。

**算法：PPO 完整循环；`old_logprob` 在 K 轮里始终不变**

1. 给定策略参数 $\theta$、价值参数 $\phi$、裁剪范围 $\epsilon$ 与优化轮数 $K$。
1. 每轮采样：
  1. 固定行为策略 $\pi_{\rm old}$，采集轨迹，保存动作对数概率和价值预测。
  1. 根据真实终止与时间截断计算 GAE，固定 $\widehat A_t$ 与价值目标 $\widehat V_t$。
  1. 重复 $K$ 轮，在轨迹中抽取小批量样本：
    1. $r_t\leftarrow\exp[\log\pi_\theta(A_t\mid S_t)-\log\pi_{\rm old}(A_t\mid S_t)]$
    1. $L_\pi\leftarrow-\operatorname{mean}\{\min[r_t\widehat A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\widehat A_t]\}$
    1. $L_V\leftarrow\operatorname{mean}[(V_\phi(S_t)-\widehat V_t)^2]$
    1. 按给定权重优化策略、价值和熵正则项。
    1. 根据预先设定的 KL 阈值决定是否提前停止本轮优化。
  1. 用更新后的策略采集下一轮轨迹。

<a id="lesson-example"></a>

## 5 · PPO 裁剪与 GAE 的数值例子

旧策略选择动作 1 的概率 .5，新概率 .75，ratio=1.5，$\epsilon$=.2。若 A=2，两个候选值为 3 和 2.4，取 2.4，该样本正向改善梯度已为 0；若 A=−2，则取 min(−3,−2.4)=−3，仍推动降低该动作概率。

两步奖励 [0,1]，旧 V=[.2,.5]，真实终止后的 value=0，$\gamma$=.9、$\lambda$=.8。$\delta_{1}$=1−.5=.5，$\delta_{0}$=0+.9×.5−.2=.25；$A_{1}$=.5，$A_{0}$=.25+.72×.5=.61。critic targets 为 [.81,1]。这和把 value target 简单设成 reward 不同。

<a id="lesson-code"></a>

## 6 · 核心实现：裁剪、边界递推与采样训练

二动作 Bernoulli PPO 与独立 GAE 函数

```python
def sigmoid(theta):
    if theta >= 0:
        return 1.0 / (1.0 + math.exp(-theta))
    exp_theta = math.exp(theta)
    return exp_theta / (1.0 + exp_theta)


def ppo_sample(theta, old_probability, action, advantage, epsilon=0.2):
    p = sigmoid(theta)
    probability = p if action == 1 else 1.0 - p
    ratio = probability / old_probability
    clipped = min(1.0 + epsilon, max(1.0 - epsilon, ratio))
    objective = min(ratio * advantage, clipped * advantage)
    inactive = (advantage > 0 and ratio > 1.0 + epsilon) or (advantage < 0 and ratio < 1.0 - epsilon)
    derivative = 0.0 if inactive else advantage * ratio * (action - p)
    return objective, derivative


def gae(rewards, values, next_values, discounts, trace_continues, lam=0.95):
    # At time-limit truncation, bootstrap may survive but the trace must not
    # jump into the next reset episode. next_values uses FINAL observation.
    out, tail = [0.0] * len(rewards), 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + discounts[t] * next_values[t] - values[t]
        tail = delta + discounts[t] * lam * trace_continues[t] * tail
        out[t] = tail
    return out


def train_ppo_bandit(rounds=100, seed=7):
    rng, theta = random.Random(seed), 0.0
    for _ in range(rounds):
        old_p = sigmoid(theta)
        batch = []
        for _ in range(64):
            action = int(rng.random() < old_p)
            # Exact old-policy baseline for this reward function.
            batch.append((old_p if action else 1.0 - old_p, action, action - old_p))
        for _ in range(4):
            grad = sum(ppo_sample(theta, *sample)[1] for sample in batch) / len(batch)
            theta += 0.1 * grad
    return {"probability_good_action": sigmoid(theta)}
```

最小 bandit 动作 1 得 1、动作 0 得 0，真实 $V_{\rm old}=p_{\rm old}$，因此 baseline 不需要另训练一个 critic。它隔离 PPO 的采样、旧概率冻结、多个 epoch 与裁剪机制；多步 actor–critic 的 GAE 是独立函数，并测试 time-limit 是否错误串到新回合。

运行后好动作概率应从 .5 上升。测试用中心有限差分核对裁剪之外及平坦区的导数，另检查正负 advantage 的不对称。这个 bandit 实验隔离策略裁剪的作用；多步控制还需要价值回归、轨迹边界处理和环境交互。

<a id="lesson-branches"></a>

## 7 · 为什么这些不是一条“后者替代前者”的序列

| 方法 | 主要改变 | 关键边界 |
| --- | --- | --- |
| REINFORCE | 完整 return 的 Monte Carlo 梯度 | 方差、等待长度 |
| actor–critic / A2C / A3C | 用 critic 与自举优势 | critic 误差、同步方式 |
| GAE | 优势估计器的多步组合 | 不是独立控制算法 |
| TRPO | 约束或近似约束策略改变 | 二阶近似、求解成本 |
| PPO | 裁剪或惩罚局部代理目标 | 不提供任意更新的硬信赖域 |
| streaming actor–critic | 每条新经验立即更新 | 尺度、迹、近似梯度时序 |
| SAC | 离策略熵正则控制 | 目标与数据协议都改变 |

CRL 中固定 rollout 收集长短、更新 epoch 数和每步延迟可能决定适应速度。增加 epochs 得到更低旧数据 loss，却可能更不适应变化后的分布；同时报告环境互动与优化计算。

<a id="lesson-check"></a>

## 8 · 习题与讨论

为何 `actor_loss` 变低不代表回报更高？这是固定旧数据上的局部代理；策略更新后数据分布改变，过多优化可能损害未见状态。必须实际评测交互收益和变化后表现。

为何必须保存 `old_logprob`？若每个 epoch 都用新网络重算分母，ratio 会不断回到 1，裁剪失去对原采样策略的参照。只保存动作、但不保存对应旧分布，是常见的隐蔽错误。

## 本章的实验设计

梯度检查使用固定轨迹。性能评价则必须重新采样。两种实验不能互相替代。

设定：固定短 rollout，分别给正负优势并跨越 PPO 裁剪边界；然后在可枚举小策略上对期望目标核导数。

- ratio=1.4、A=2、clip=0.2 的最大化目标为 2.4。
- ratio=1.4、A=−2 时目标为 −2.8，不能统一先裁剪 ratio。
- 外部截断可以 bootstrap，但不得把新 episode 的 GAE 接回旧轨迹。

对照：REINFORCE 与带基线估计器；固定样本和真实重采样分开；相同旧策略数据与明确 epoch 制度

记录：概率比、clip fraction、KL 与熵；固定样本梯度误差；独立交互收益及 rollout/更新成本

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-deep)

## 学习与研究衔接

从 REINFORCE 到 actor–critic，评论家提供低方差学习信号。PPO 的批量多轮更新与严格流式协议不同。

[分册导读](https://yingwen.io/zh/continual-rl/start/deep-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-policy) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=policy) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=policy)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://yingwen.io/zh/continual-rl/research/#recent-vtrace-impala)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Intentional Updates for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-intentional-updates)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Intentional Updates for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-intentional-updates)

### Intentional Updates for Streaming Reinforcement Learning

Arsalan Sharifnassab, Mohamed Elsayed, Kris De Asis, A. Rupam Mahmood, Richard S. Sutton

ICML 2026 · 2026 · 支持方法与理论

#### 研究问题

能否先规定本次更新应产生多大作用，再反推合适的参数更新尺度？

#### 关键机制

Intentional 方法以局部线性近似连接参数变化和预测变化。critic 以减少一定比例的 TD 误差为目标，actor 控制策略输出变化的局部代理量；再结合资格迹和逐坐标尺度，求出这一次更新的强度。这是有目标的局部更新控制，不是对长期表现求导的元梯度。

#### 证据

论文给出推导和流式控制比较，ICML 2026 正式论文入口与作者实现均可用。实现将优化器与 actor–critic 交互区分开，便于检查更新时序。

#### 条件与限制

Taylor 近似在大更新时可能失准。采样动作上的对数概率变化不等于精确的全分布 KL 上界；熵项与 TD 误差符号也必须按原算法处理。

#### 阅读与实验

在一次更新前后直接测量预测变化，并与线性估计比较。分别测试正、负 TD 误差和很小梯度的情形，不要只检查参数是否有限。

#### 原文与相关入口

- [ICML 2026 原文](https://proceedings.mlr.press/v306/sharifnassab26a.html)：正式会议版本与更新意图的定义。
- [作者实现](https://github.com/sharifnassab/Intentional_RL)：重点对照 optimizer.py 与 intentional_ac.py。

#### 作者代码

[原论文作者提供的实现。](https://github.com/sharifnassab/Intentional_RL)

Intentional 更新与流式 actor–critic。

### IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures

Lasse Espeholt, Hubert Soyer, Rémi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, Koray Kavukcuoglu

ICML 2018 · 2018 · 支持方法与理论

#### 研究问题

actor采样策略落后于learner时，如何校正状态价值与策略更新？

#### 关键机制

V-trace用截断ρ校正当前TD误差，用独立截断c控制后续误差传播，再用下一状态V-trace目标构造actor优势。ρ上限还决定表格固定点对应的截断策略。

#### 证据

原文分析固定点并检验分布式多任务训练。固定版本作者代码明确区分clipped_rhos、cs、反向scan与pg_advantages。

#### 条件与限制

IMPALA保存短轨迹并批量训练，不属于严格单样本流式协议。截断后价值可能对应不同于原目标的策略；信用章bandit示例显示0.8变为0.5。

#### 阅读与实验

独立改变策略滞后、ρ上限和c上限。记录目标策略变化与传播长度，不把两种截断都只解释为方差控制。

#### 原文与相关入口

- [ICML原文](https://proceedings.mlr.press/v80/espeholt18a.html)：V-trace固定点与分布式实验。
- [作者固定实现](https://github.com/google-deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/vtrace.py)：from_importance_weights与下一状态actor目标。

#### 作者代码

[原作者团队仓库的固定版本。](https://github.com/google-deepmind/scalable_agent/tree/6c0c8a701990fab9053fb338ede9c915c18fa2b1)

IMPALA原始TensorFlow实现与V-trace；运行需要原项目环境。


<a id="chapter-code"></a>

## 下载与运行

Python 3.10+，仅标准库。执行 PPO bandit 完整采样/优化循环，独立实现多步 GAE；不声称包含网络 PPO 的通用训练工程。

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

```sh
python3 foundations_detail_lab.py policy
python3 foundations_detail_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Williams · Simple statistical gradient-following algorithms](https://doi.org/10.1007/BF00992696)：REINFORCE 的原始梯度估计思想。把 score function 与数据分布的关系读清楚。

- [Schulman et al. · Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)：区分 $\gamma$ 带来的目标/估计变化和 $\lambda$ 引入的优势估计权衡。

- [Schulman et al. · Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)：逐项对照 surrogate、clipping、训练轮数与采样协议。

- [Spinning Up · Policy optimization 推导与代码](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)：以轨迹概率、log derivative 和 baseline 连起数学与 PyTorch。

- [OpenAI Spinning Up · PPO PyTorch 作者教学实现](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py)：追 `buffer.finish_path`、`compute_loss_pi`、update 三部分；工程的旧环境 API 需单独适配。


---

# 最大熵控制与 Soft Actor–Critic

把多样性纳入目标，会怎样改变 Bellman 方程、策略改善和实际训练？先用离散精确解推清楚，再给出连续 SAC 的完整接口。

## 本章内容

- 从带熵约束的最优化推导 softmax 与 log-sum-exp。
- 辨认 soft Q、策略熵、双 critic 和自动温度分别起什么作用。
- 实现离散熵正则 actor 更新，并能检查连续 SAC 的 log-prob 和梯度路径。

<a id="problem-definition"></a>

## 本章的问题定义

将动作分布熵作为明确的优化收益，控制目标随之改变；不是给普通控制算法附加一个无影响的探索技巧。

### 给定条件与符号

- 固定折扣任务、策略类、动作坐标与熵定义。
- 温度或目标熵、回放与双critic配置、计算预算。

### 需要求解的对象

熵正则策略和soft价值；SAC近似学习这些量并可另行适应温度。

### 信息与数据权限

真实经验生成回放；critic标签停止梯度。actor更新冻结critic参数，但保留其对动作输入的导数。连续动作密度必须包含变换Jacobian。

$$
J_\tau(\pi)=\mathbb E_\pi\!\left[\sum_{t=0}^{\infty}\gamma^t\{R_{t+1}+\tau\mathcal H(\pi(\cdot\mid S_t))\}\right]
$$

$\gamma<1$ 是折扣，$\tau>0$ 是熵温度，$\mathcal H$ 为离散熵或明确坐标下的微分熵。$\tau$ 固定时，这是区别于纯外部回报的目标；自动温度另有目标熵约定。SAC的critic与actor loss是估计和改善该目标的代理。

### 成立条件与解的含义

- 有限离散精确softmax推导要求各动作价值有限；连续积分、微分熵和重参数化需要相应可积/可微条件。
- 奖励尺度、动作尺度与温度共同决定目标；深网、双critic最小值与回放不自动保证收敛。

判断准则：离散Q=[0,1]、温度0.5时动作1概率约0.880797、soft value约1.063464；连续实现检查变换密度与梯度路径，外部收益和熵收益分别报告。

### 适用边界

- soft value不是纯外部回报的价值。
- 连续微分熵不与离散熵直接数值比较。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 改变评价目标 · [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)：熵进入回报，而actor优化和数据分布也按SAC协议改变。

- 组合不同学习问题 · [平均奖励与差分价值](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)：最大熵准则可以结合平均奖励，但需重新定义奖励率、差分critic与参照项。

- 组合不同学习问题 · [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)：熵鼓励分布多样性，但不等于访问新区域、信息增益或恢复能力。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

普通贪心选择忽略目标中的熵；连续策略还需可微采样和正确概率密度。

### 本章的核心思路

先由熵正则最优化推到softmax/soft Bellman，再将评价、改善和温度分别落实。

1. [从熵收益推导策略改善](#lesson-derive)：因为确定贪心不再最优，拉格朗日推导得到softmax与log-sum-exp，并明确温度尺度。

2. [构造soft评价与actor梯度](#sac-targets)：因为后续收益含熵，critic标签扣对数概率；actor通过重参数化动作保留动作价值梯度。

3. [安排各模块的冻结边界](#sac-loop)：因为同批数据上critic、actor和温度互相依赖，明确标签停止梯度、critic参数冻结及目标软更新次序。

结论与条件：有限离散精确局部熵优化有解析解；近似双critic/SAC训练不继承任意网络的全局最优保证，自动温度也需其目标熵可行。

### 相关方法改变了什么

- 普通贪心控制：只优化外部回报，不支付熵收益。

- 精确soft策略迭代：已知或精确价值下执行soft评价与改善。

- SAC：以回放、双critic和重参数化actor近似实现，含额外估计和工程误差。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 一条经验

在 $S_{t}$ 做 $A_{t}$ 后得到 $R_{t+1}$、$S_{t+1}$。奖励属于这次转移，不是到达状态之前另一轮的奖励。

### 折扣与终止

$\gamma$ 控制未来相对权重；真实终止后的价值设为 0。本章记 $\gamma_{t+1}$=$\gamma$(1−terminated)。训练脚本的时间截断不一定是任务终止。

### 表格与函数逼近

表格每个状态或状态动作一组独立参数；函数逼近让不同输入共享参数。更新一个输入可能改变其他输入的预测。

### 熵与温度

H($\pi$)=−Σ$\pi$ log$\pi$ 衡量离散分布不确定性；$\tau$>0 是熵权重。本章用 $\tau$ 避免与学习率 $\alpha$ 混淆。连续密度的微分熵依赖坐标尺度，不与离散熵直接数值比较。

<a id="lesson-setting"></a>

## 1 · 熵正则化控制目标

$$
J_\tau(\pi)=\mathbb E_\pi[\sum_{t\ge0}\gamma^t(R_{t+1}-\tau\log\pi(A_t\mid S_t))]
$$

期望下 −log$\pi$ 就是熵。$\tau$ 大更愿意保留动作多样性；$\tau$→0 才回到普通奖励最大化的相应极限。用 entropy bonus 改变了优化问题，而不仅是修复梯度。

先假设离散动作、已知一组当前 Q(s,a)，求一个状态上最好的概率分布。之后再将这个策略改善步与从 replay 学 Q 交替。soft Q 的常见约定包含当前外部奖励与未来熵，但不包含当前动作自身的 −$\tau$log$\pi$；soft V 才在该状态对 Q−$\tau$log$\pi$ 求期望。

<a id="lesson-derive"></a>

## 2 · 用拉格朗日乘子推导软策略改善

$$
\max_{p_a\ge0,\,\sum_a p_a=1}\sum_a p_a Q_a-\tau\sum_a p_a\log p_a
$$

这是期望价值加熵的优化。对正概率的内部解加乘子 $\eta$，令导数 $Q_a$−$\tau$(log $p_a$+1)+$\eta$=0。

$$
p_a^*=\frac{\exp(Q_a/\tau)}{\sum_b\exp(Q_b/\tau)},\qquad V^*(s)=\tau\log\sum_a\exp(Q(s,a)/\tau)
$$

先解出 p∝exp(Q/$\tau$)，再归一化。把 logp=Q/$\tau$−logZ 代回目标，Q 项抵消，剩下 $\tau$logZ。$\tau$>0 下解严格正；计算时减掉最大 Q 防止溢出。

$$
Q^\pi(s,a)=\mathbb E[R_{t+1}+\gamma V^\pi(S_{t+1})],\quad V^\pi(s)=\mathbb E_{a\sim\pi}[Q^\pi(s,a)-\tau\log\pi(a\mid s)]
$$

这是策略评价；前面的 softmax 是给定 Q 的策略改善。两者交替构成 soft policy iteration 的基础。学习中的近似 Q 与有限网络容量会让真实训练偏离精确运算。

$$
L_\pi=\mathbb E_{s\sim D,a\sim\pi_\theta}[\tau\log\pi_\theta(a\mid s)-Q(s,a)]
$$

给定 critic，对这个损失下降等价于向 exp(Q/$\tau$) 的分布靠近（加上与 $\theta$ 无关的归一化常数，可写 KL）。离散时能枚举所有动作；连续时通常借可微重参数化采样。

<a id="sac-targets"></a>

## 3 · SAC 的 critic、actor 与温度不是同一个更新

$$
a\prime\sim\pi_\theta(\cdot\mid s\prime),\quad Y=r+\gamma(1-d)[\min_{j=1,2}Q_{\bar\phi_j}(s\prime,a\prime)-\tau\log\pi_\theta(a\prime\mid s\prime)]
$$

这是现代常用双 Q SAC 的 target；使用 target critics 和当前 actor 采样后继动作，整个 Y 在 critic 回归中停止梯度。min 减少某些高估影响，也可能引入低估；不是无误差真值。

$$
L_{Q_j}=\mathbb E_D[\tfrac12(Q_{\phi_j}(s,a)-\operatorname{sg}Y)^2],\quad L_\pi=\mathbb E[\tau\log\pi_\theta(a_\theta\mid s)-\min_jQ_{\phi_j}(s,a_\theta)]
$$

critic 更新不动 actor；actor 更新冻结 critic 参数，但必须保留 Q 对动作输入的梯度，才能把“哪个动作好”传回 actor。直接对 Q 的整个输出 detach 会把这条路径剪断。

$$
u=\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon,\quad\epsilon\sim\mathcal N(0,I),\quad a=\tanh u,\quad\log\pi(a\mid s)=\log\mathcal N(u;\mu,\sigma)-\sum_i\log(1-\tanh^2u_i)
$$

tanh 改变密度，必须扣 Jacobian。动作若进一步映射到环境区间，还要处理尺度常数。连续多维动作的 log-prob 对动作维求和；不是留一个 [batch, action_dim] 与 [batch] 错误广播。

$$
L_\tau(\log\tau)=\mathbb E[-\tau\operatorname{sg}(\log\pi(a\mid s)+\mathcal H_{\mathrm{target}})]
$$

这是常见温度目标的一种参数化。当实际熵低于目标时，梯度下降会增大 $\tau$。许多实现用 −log$\tau$ 乘同一停止梯度项，方向相同但步长尺度不同。固定 $\tau$ 与自动 $\tau$ 是不同实验配置；目标熵是超参数而不是环境真值。

<a id="sac-loop"></a>

## 4 · 完整训练顺序与一次更新的冻结边界

**算法：固定温度或自动温度二选一；更新次数与次序属于算法设定**

1. 初始化策略 $\pi_\theta$、两个 critic $Q_{\phi_1},Q_{\phi_2}$、目标 critic 与回放池。
1. 每个真实环境步：
  1. 按探索规则采样动作；保存 $(s,a,r,s',d)$，其中 $d$ 只表示真实终止。
  1. 从回放池抽取小批量样本。
  1. 在停止梯度的上下文中计算下一动作和 soft target $Y$。
  1. 分别最小化两个 critic 的平方误差。
  1. 固定 critic 参数，保留 critic 输出对动作的导数。
  1. 通过重参数化动作，最小化 $\tau\log\pi_\theta(a\mid s)-\min_jQ_{\phi_j}(s,a)$。
  1. 若使用自适应温度，以固定的策略对数概率更新温度参数。
  1. 更新目标参数：$\bar\phi_j\leftarrow(1-\eta)\bar\phi_j+\eta\phi_j$。
  1. 环境重置由任务协议决定；网络权重与回放池跨回合保留。

软更新系数 $\eta$ 的不同工程约定可能相反，有的写 polyak 接近 1 作为旧参数保留率。本章式子中 $\eta$ 是新参数占比；照抄变量名却不对式子是常见错误。

<a id="lesson-example"></a>

## 5 · 两个动作的解析答案

Q=[0,1]，$\tau$=.5。最优动作 1 概率为 exp(2)/(1+exp(2))≈.880797，不是 1；soft value=.5 log(1+exp(2))≈1.063464，大于最大外部 Q=1，因为目标还包含熵收益。

若两动作 Q 都加 3，概率不变、soft value 加 3；若把奖励整体乘 10 而 $\tau$ 不变，则策略更接近贪心。奖励尺度与温度不能完全分开比较。目标 critic 的两个值也须先对同一动作取 min，不能先对各 critic 分别最大化。

<a id="lesson-code"></a>

## 6 · 核心实现：精确离散备份与可检查 actor 梯度

离散动作 soft value、双 critic target 与分类 actor 的完整梯度下降

```python
def softmax(logits):
    shifted = [math.exp(x - max(logits)) for x in logits]
    return [x / sum(shifted) for x in shifted]


def soft_value(q, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    largest = max(q)
    return largest + temperature * math.log(sum(math.exp((x - largest) / temperature) for x in q))


def discrete_sac_target(reward, discount, probabilities, q1, q2, temperature):
    if discount == 0.0:
        return reward
    expectation = sum(p * (min(a, b) - temperature * math.log(p))
                      for p, a, b in zip(probabilities, q1, q2) if p > 0)
    return reward + discount * expectation


def categorical_actor(logits, q, temperature):
    probabilities = softmax(logits)
    top = max(logits)
    log_z = math.log(sum(math.exp(z - top) for z in logits))
    log_probs = [z - top - log_z for z in logits]
    cost = [temperature * logp - val for logp, val in zip(log_probs, q)]
    objective = sum(p * c for p, c in zip(probabilities, cost))
    # Gradient of sum pi(a) [temperature log pi(a) - Q(a)].
    gradient = [p * (c - objective) for p, c in zip(probabilities, cost)]
    return objective, gradient


def train_soft_bandit(temperature=0.5, iterations=2000):
    q, logits = [0.0, 1.0], [0.0, 0.0]
    for _ in range(iterations):
        _, grad = categorical_actor(logits, q, temperature)
        logits = [z - 0.1 * g for z, g in zip(logits, grad)]
    return {"learned": softmax(logits), "exact": softmax([v / temperature for v in q]),
            "soft_value": soft_value(q, temperature)}
```

此实验先固定 Q=[0,1]，从均匀概率更新 actor，输出 learned 与解析 exact；测试每个 logit 的梯度，并验证 log-sum-exp 的平移性质。它实现 SAC 中可独立核查的 soft policy improvement 与 target，不包含连续环境的全部 SAC 工程。

扩展到连续网络实现时，按本章第 3–4 节逐项对齐张量、停止梯度与训练顺序；不能把离散 softmax 的枚举公式直接换成未归一化的连续概率。章节末的作者实现提供完整优化器与环境循环。

<a id="lesson-branches"></a>

## 7 · 熵正则、探索与持续学习的关系

| 概念 | 和 SAC 的关系 | 必须分开的问题 |
| --- | --- | --- |
| 最大熵控制 | 定义 Q / V / actor 的目标 | 不是任意探索都等价于熵 |
| DDPG / TD3 | 相邻的离策略连续控制路线 | 确定性 actor、平滑 target、双 critic 的设计不同 |
| 离散 SAC | 动作期望可精确枚举 | 算法与连续重参数化实现不同 |
| 内在奖励 / RND | 在外部奖励之外增加学习信号 | 新奇不等于策略熵 |
| CRL replay / plasticity | 影响长期训练数据与可训练能力 | SAC 本身不保证长期适应与保留 |

在持续环境中，陈旧 critic 会让 actor 追逐过时值；自动温度只能调整随机性，不能识别哪部分世界已改变。先用固定数据测试价值跟踪，再分析策略诱导分布，能避免把所有失败归因于熵系数。

<a id="lesson-check"></a>

## 8 · 两个梯度检查题

actor 更新时冻结 critic，是不是要对 min Q detach？不是。冻结的是 critic 参数的梯度，但 Q 对 a 的导数仍需流向 actor。critic 更新时才对整个 target Y stop-gradient。

把 $\tau$ 降到 0 是不是仍需计算 log$\pi$ 的 tanh Jacobian？当实现中还训练温度或报告熵时当然需要；即使固定 $\tau$ 恰为 0，算法也退化成不同的极限目标，不能继续用“相同 SAC 配置”的名义比较。

## 本章的实验设计

区分外部奖励与带熵的优化目标。固定评价口径后，再比较温度、样本量和计算成本。

设定：固定连续动作样本与重参数化噪声，检查动作缩放、tanh 密度修正、双 critic 与温度。性能面板使用相同外部奖励定义。

- 真正终止时不读取后继策略和 critic 输出。
- tanh 后动作的 log-prob 包含相应 Jacobian 修正。
- actor、critic、温度及目标网络的更新时钟可独立追踪。

对照：固定温度与可学习温度；相同网络、动作范围及奖励尺度；完整 SAC/TD3 与共同基座消融分列

记录：原始奖励与熵正则目标分别记录；温度、动作饱和、critic 偏差与策略 KL；回放更新比、数据量和计算

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-deep)

## 学习与研究衔接

连续动作中，策略承担动作搜索。熵、双评论家与回放各有作用，不能合并为一个“稳定化技巧”。

[分册导读](https://yingwen.io/zh/continual-rl/start/deep-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-soft-control) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=soft-control) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=soft-control)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [RVI-SAC: Average Reward Off-Policy Deep Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-rvi-sac-average-control)

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


<a id="chapter-code"></a>

## 下载与运行

Python 3.10+，仅标准库。运行精确离散 actor 优化和公式检查；连续 SAC 的完整训练顺序在正文，作者工程在章末。

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

```sh
python3 foundations_detail_lab.py soft-control
python3 foundations_detail_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Haarnoja et al. · Soft Actor-Critic](https://proceedings.mlr.press/v80/haarnoja18b.html)：原始最大熵 actor–critic；原版含 value 网络，与后来常见无独立 value 网络的实现需区分。

- [Haarnoja et al. · Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)：自动温度与实际算法版本，核对目标熵和温度参数化。

- [作者工程 · softlearning](https://github.com/rail-berkeley/softlearning)：原作者维护的最大熵 RL 工程；环境与框架依赖应在单独环境中固定版本。

- [Spinning Up · SAC PyTorch 实现](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py)：教学实现入口，重点追 `compute_loss_q`、`compute_loss_pi`、critic 参数冻结与 Polyak 更新。

- [Spinning Up · SAC 公式与算法说明](https://spinningup.openai.com/en/latest/algorithms/sac.html)：作为实现导览，区分固定温度示例和自动温度扩展。


---

# 时间信用分配：从资格迹到深度梯度学习

结果到来时，怎样更新过去的预测、动作和记忆参数？策略变化、表示变化和部分可观测性会怎样改变信用与等价条件？

## 本章内容

- 区分时间信用、网络结构信用、流式约束与元学习。
- 推导 n-step、λ-return 和冻结参数的前向／后向等价，定位在线参数变化造成的差异。
- 实现 true-online TD(λ)，逐前缀核验荷兰迹与预测差修正。
- 用同一误差传播式比较 Tree-backup、Retrace，并区分 V-trace 的误差截断与传播截断。
- 推导状态相关 λ、Q(σ) 与期望资格迹，检查各自的目标、条件和反例。
- 从广义投影 Bellman 误差推导 GTD2／TDC／TDRC 的三条迹，核对非线性冻结总和等价。
- 连接 actor–critic、GAE 和 RTRL／RTU，区分回报信用、状态敏感度与参数变化造成的过时导数。

<a id="problem-definition"></a>

## 本章的问题定义

延迟反馈到来时，计算它应怎样改变早先预测、动作和记忆参数。时间传播与结构求导是两个耦合但不同的对象。

### 给定条件与符号

- 预测/控制的基础目标、轨迹和终止/窗口边界。
- 目标与行为策略、表示、迹参数、导数路径及可用内存。

### 需要求解的对象

与声明的前向目标、投影目标或策略目标相符的更新估计器，并说明在线变化和近似造成的差异。

### 信息与数据权限

起点 $S_t$ 的完整前向目标要等后续奖励到来；资格迹 $e_k$ 在时刻 $k$ 保存过去更新方向的统计量。它不是供行动回忆历史的agent state。

$$
\Delta w_{\rm forward}=\alpha\sum_{t=0}^{T-1}\bigl(G_t^\lambda-v_w(S_t)\bigr)\nabla_wv_w(S_t)
$$

$w$ 是本段冻结的预测参数，$T$ 是真实终点，$G_t^\lambda$ 是本章定义的多步混合目标，$\lambda$ 为混合权重，$\alpha$ 为步长。这是用于检验后向信用的理想总更新；在线true-online视图、离策略修正、投影Bellman梯度和actor敏感度各自使用对应对象，不全是此式。

### 成立条件与解的含义

- 冻结前后向总和需同一参数和边界；在线精确等价需true-online的线性条件与相应在线前向定义。
- 重要性比需要支持；传播截断、误差截断、状态相关迹以及递归敏感度均需各自计时和目标说明。

判断准则：在有限轨迹逐项核对冻结总和或每个在线前缀；检查单步、全回报、零比率和终止端点；策略/递归梯度以对应固定计算图有限差分检验。

### 适用边界

- 更新信用不等于通用因果责任归属。
- 资格迹、RTRL和元敏感度虽都递推，追踪对象并不相同。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)：同一固定策略预测对象可采用一步、多步或迹；信用机制改变如何利用反馈。

- 组合不同学习问题 · [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)：记忆参数的结构敏感度连接过去活动与当前输出，不由回报迹自动给出。

- 改变信息或数据协议 · [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)：后向迹可满足有限内存；能否重放、等待和展开仍由协议限定。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

前向目标依赖尚未发生的反馈；参数和行为持续变化又破坏简单冻结等价。

### 本章的核心思路

先写反馈传播的前向对象，再推出后向统计，并逐项修正在线参数变化、行为差异与结构导数。

1. [展开并抵消未来反馈项](#lesson-equivalence)：因为多步目标共享奖励和bootstrap，固定参数下将混合目标改写为TD误差加权和。

2. [补偿在线预测的变化](#lesson-true-online)：因为普通迹用不断变化的参数，true-online以Dutch trace和预测差修正对齐线性在线前向视图。

3. [处理不同目标行为的传播](#lesson-off-policy)：因为行为轨迹不等于目标轨迹，Tree-backup用动作期望分支，Retrace截断传播比率；有界传播本身不保证函数逼近稳定。

4. [对明确的投影目标求梯度](#lesson-gradient-traces)：因为半梯度自举未必对应稳定目标，辅助误差预测与多条迹分别估计广义投影目标中的条件均值和传播方向。

5. [接入actor和记忆敏感度](#lesson-actor-recurrent)：因为动作梯度与递归参数不是价值输出本身，分别用优势和结构敏感度连接相应目标。

结论与条件：冻结求和是代数恒等式；true-online精确等价限于相应线性视图。GTD/TDRC与离策略控制各有额外条件，不推广到任意深网或变化世界。

### 相关方法改变了什么

- 多步前向目标：等待有限窗口再计算，数据和延迟成本明确。

- 普通迹/true-online：前者低成本在线近似；后者修正线性在线前向视图。

- Retrace/Tree-backup/V-trace：行为纠偏与传播系数不同，误差截断可改变有效目标策略。

- GTD2/TDC/TDRC：以辅助误差预测和梯度迹连接指定投影目标，目标、曲率与正则路径需各自检查。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 价值预测

固定策略的价值 $v_\pi(s)$ 是未来回报的条件期望。网络 $v_w$ 只是估计。以下先讨论固定策略的预测问题。

### TD 误差

一步奖励与下一状态预测形成目标，再减去当前预测。两次预测使用同一组更新前参数。

$$
\delta_t=R_{t+1}+\gamma v_w(S_{t+1})-v_w(S_t)
$$

### 线性逼近

固定特征 $x_t$ 与权重 $w$ 的内积构成预测。对权重的梯度是 $x_t$；学习特征的情况需要额外求导。

$$
v_w(S_t)=w^\top x_t,\qquad\nabla_wv_w(S_t)=x_t
$$

### 终止与截断

真正终止的尾值为零。记录窗口结束不一定是终止；若任务仍继续，尾部应保留 bootstrap。

<a id="lesson-setting"></a>

## 1. 信用分配要回答的问题

智能体先记住一条提示，再经过多个状态，最后获得奖励。此时至少有三个学习问题。哪些过去的价值预测需要修正？哪些动作概率应该改变？哪些负责保存提示的参数应当更新？只说“把奖励反传回去”没有说明所用的目标、路径和近似。

| 层次 | 问题 | 典型计算 |
| --- | --- | --- |
| 时间信用 | 较晚的结果应当怎样影响较早的预测或动作？ | 多步回报、资格迹、动作优势估计。 |
| 结构信用 | 当前输出对哪些参数和内部计算敏感？ | 反向传播、链式法则、递归敏感度。 |
| 计算与数据约束 | 能保存多少历史？每次交互可以计算多久？ | rollout、重放、截断反传、固定内存的流式更新。 |
| 学习规则的适应 | 步长、迹长度或目标应当怎样随经验改变？ | IDBD、Metatrace、meta-gradient RL。 |

这些层次可以组合。资格迹是一种时间信用机制。它可以用于逐步更新，也可以用于离线等价分析。流式协议限制数据和计算，不定义哪段历史应该得到信用。元学习调整学习规则；即使它也使用“迹”，所追踪的可能是步长对未来权重的影响，而不是过去状态对当前 TD 更新的资格。

先固定行为策略，观察一条 episode：$(S_0,R_1,S_1,\ldots,R_T,S_T)$。取常数 $\gamma,\lambda\in[0,1]$。终点特征为 $x_T=0$。推导前半部分冻结参数 $w$；之后才允许每步更新。随后分别放松同策略、固定迹长度、已知 Markov 状态和线性表示这些条件。每次放松只引入一个新的问题。

Continuing task 指没有内在终点的持续交互任务；continual learning 指在长期经验中继续学习与适应。一个固定环境的 continuing task 未必包含分布变化；一串有限 episode 也可以构成持续学习。本文用“持续交互任务”指前者。日志窗口结束、环境终止和研究协议的任务切换需要分别记录。

这里的信用是一个指定学习目标下的更新分配，不是对行动因果责任的通用解释。TD 迹先解决预测更新。策略梯度还需说明动作怎样影响回报分布。两者不能只因都乘了奖励就被视作同一方法。

<a id="lesson-derive"></a>

## 2. n-step 与 λ-return 的前向视角

一步 TD 只使用下一条真实奖励，其余部分由价值估计补齐。$n$ 步目标使用更多真实奖励，再从第 $n$ 个后继状态 bootstrap。增加 $n$ 减少了对近处估计的直接依赖，也延迟了目标形成；在有噪声时，它通常改变偏差与方差。

$$
G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^n v_w(S_{t+n}),\qquad1\le n\le T-t
$$

本节所有 bootstrap 都用同一固定 $w$。当 $n=T-t$ 时，终点价值为零，目标等于完整 episode return。对于非终止的采样窗口，最后一项不能自动删掉。

不必只选一个 $n$。$\lambda$-return 对各个长度的目标作几何混合。有限 episode 的最后一个目标接收尚未分配的全部权重。这个尾项保证权重之和为一。

$$
G_t^\lambda=(1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1}G_t^{(n)}+\lambda^{T-t-1}G_t^{(T-t)}
$$

$\lambda=0$ 给出一步 TD 目标；$\lambda=1$ 给出完整 return。最后一步只有一个目标，其权重为一。$\gamma$ 决定任务中未来奖励的权重，$\lambda$ 决定学习时怎样混合不同 bootstrap 长度；二者职责不同。

$$
G_t^\lambda=R_{t+1}+\gamma\left[(1-\lambda)v_w(S_{t+1})+\lambda G_{t+1}^\lambda\right],\qquad G_T^\lambda=0
$$

把每个多步目标的第一条奖励分离后，剩下的部分仍是同类混合，所以可以后向递推。非终止窗口的边界应取其 bootstrap 值，而不取零。

完整前向目标需要未来数据。它适合解释学习方向，也适合在收集好的短轨迹上计算监督目标。若不能等待完整 episode，就需要有限窗口目标或后向机制。这个计算限制不改变“信用应分配给什么”的问题本身。

<a id="lesson-equivalence"></a>

## 3. 冻结参数时，资格迹来自交换求和

记固定预测为 $v_t=v_w(S_t)$。将前向递推减去 $v_t$，把一项 $\gamma v_{t+1}$ 加入并减去，便能用 TD 误差替代整段奖励。反复展开直到边界，就得到误差和。

$$
\begin{aligned}G_t^\lambda-v_t&=\delta_t+\gamma\lambda(G_{t+1}^\lambda-v_{t+1}),\\\delta_t&=R_{t+1}+\gamma v_{t+1}-v_t,\\G_t^\lambda-v_t&=\sum_{k=t}^{T-1}(\gamma\lambda)^{k-t}\delta_k.\end{aligned}
$$

终点的回报与预测均为零，所以最后没有残余项。若边界不是终止，只要前向目标与 TD 分解使用同一个固定尾值，这个有限窗口恒等式仍成立。

令 $g_t=\nabla_wv_w(S_t)$。对所有时刻的半梯度增量求和，且在整个求和期间不改变 $w$。交换“过去预测”和“后来误差”的求和次序，就得到后向资格迹。

$$
\begin{aligned}\Delta w&=\alpha\sum_{t=0}^{T-1}(G_t^\lambda-v_t)g_t\\&=\alpha\sum_{k=0}^{T-1}\delta_k\underbrace{\sum_{t=0}^{k}(\gamma\lambda)^{k-t}g_t}_{z_k},\\z_k&=\gamma\lambda z_{k-1}+g_k,\qquad z_{-1}=0.\end{aligned}
$$

这是固定参数下的总增量恒等式。前向计算使用未来误差修正早期预测；后向计算在误差到达时，将它乘以过去预测梯度的衰减和。对线性预测 $g_t=x_t$。

资格迹积累的是输出梯度，不是过去已经完成的损失梯度。当前 $\delta_k$ 到来后才乘整条迹。Momentum 则积累各时刻已经乘上各自误差的更新，两者时间结构不同。迹也不是样本缓冲区：它压缩了某种加权和，不能从中恢复任意旧转移。

前向 λ-return 与后向误差迹各自独立计算总增量，始终使用冻结权重。

```python
def lambda_returns(rewards, values, gamma, lam):
    """Finite-horizon forward targets; values[-1] is zero only at true terminal."""
    if len(values) != len(rewards) + 1 or not values:
        raise ValueError('T+1 values required')
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError('gamma and lambda must lie in [0,1]')
    targets = [0.0] * len(rewards)
    tail = values[-1]
    for t in range(len(rewards) - 1, -1, -1):
        tail = rewards[t] + gamma * ((1 - lam) * values[t + 1] + lam * tail)
        targets[t] = tail
    return targets


def frozen_forward_backward(features, rewards, weights, gamma, lam, alpha):
    """Two independently accumulated episode increments using ONE fixed w."""
    validate(features, rewards, weights, gamma, lam, alpha)
    values = [dot(weights, x) for x in features]
    targets = lambda_returns(rewards, values, gamma, lam)
    forward, backward, trace = [[0.0] * len(weights) for _ in range(3)]
    for t, x in enumerate(features[:-1]):
        for j in range(len(weights)):
            forward[j] += alpha * (targets[t] - values[t]) * x[j]
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        trace = [gamma * lam * z + xj for z, xj in zip(trace, x)]
        backward = [b + alpha * delta * z for b, z in zip(backward, trace)]
    return dict(targets=targets, forward=forward, backward=backward)
```

<a id="lesson-online"></a>

## 4. 传统在线 TD(λ) 为什么不再精确等价

实际 TD(λ) 通常每收到一次转移就改变权重。于是下一步预测已经不是冻结推导中的预测；非线性网络的梯度也会变化。继续使用同样的递推仍然定义一个合理算法，但不能继续引用冻结参数的总增量恒等式来证明它与原前向目标完全一致。

$$
\begin{aligned}\delta_t&=R_{t+1}+\gamma w_t^\top x_{t+1}-w_t^\top x_t,\\z_t&=\gamma\lambda z_{t-1}+x_t,\\w_{t+1}&=w_t+\alpha\delta_tz_t.\end{aligned}
$$

这里是固定线性特征的 accumulating TD(λ)。每步更新立即生效。小步长时与某些冻结近似的差别可能较小，但不能将近似写成任意步长下的精确等价。

若使用状态相关 continuation，必须区分到达当前状态的 $\gamma_t$ 与离开当前状态的 $\gamma_{t+1}$。前者衰减旧迹，后者构造 bootstrap。真正终止的 $\gamma_{t+1}=0$ 不意味着应当在使用终点奖励之前删除过去资格。

$$
z_t=\gamma_t\lambda z_{t-1}+g_t,\qquad\delta_t=R_{t+1}+\gamma_{t+1}v_{w_t}(S_{t+1})-v_{w_t}(S_t)
$$

配套代码先采用常数折扣，并将终点特征设为零。终止奖励完成更新后，新 episode 才重置资格迹。参数通常保留。单纯日志分段不提供清迹权限。

Replacing trace 是另一种选择。例如二值特征被重新激活时，将其迹置一，而不是再加一。它避免重复访问使该分量持续累加，但改变了更新定义。它也不等于下面的 Dutch trace；“都是资格迹”不意味着前向解释相同。

<a id="lesson-true-online"></a>

## 5. True-online TD(λ)：定义新的在线前向目标

精确在线等价需要先定义一个只使用已到达数据的前向算法。设当前只观察到时刻 $h$。对每个旧时刻 $t<h$，构造截至 $h$ 的 interim $\lambda$-return。第 $n$ 步 bootstrap 使用当时尚未处理该转移前的真实权重 $w_{t+n-1}$。这些权重不是后来重算的临时权重。

$$
\begin{aligned}G_t^{(n),\mathrm{on}}&=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^n w_{t+n-1}^\top x_{t+n},\\G_t^{\lambda\mid h}&=(1-\lambda)\sum_{n=1}^{h-t-1}\lambda^{n-1}G_t^{(n),\mathrm{on}}+\lambda^{h-t-1}G_t^{(h-t),\mathrm{on}}.\end{aligned}
$$

只有到达时刻 $h$ 之前的信息参与目标。若 $h$ 不是真实终点，最后目标仍有 bootstrap。新数据到来后，旧时刻的 interim 目标也随之变化。

字面实现每来一步，就从本 episode 的初始参数重新顺序执行全部旧状态的更新。它只使用已经到达的数据，但保存与重算成本很高。这个算法用于定义精确参考，不是推荐的流式实现。

$$
w_0^{[h]}=w_{\rm init},\qquad w_{t+1}^{[h]}=w_t^{[h]}+\alpha\left(G_t^{\lambda\mid h}-(w_t^{[h]})^\top x_t\right)x_t,\qquad w_h=w_h^{[h]}
$$

上标 $[h]$ 指当前重算的临时序列。必须从同一个 $w_{\rm init}$ 开始；不能在上一轮临时末值上再重复加整段更新。算法在每个观察前缀末端只保留对角结果 $w_h$。

直接重算所有 interim 目标的参考程序；用来逐前缀验证，不满足固定历史存储预算。

```python
def online_forward_reference(features, rewards, initial, gamma, lam, alpha):
    """Literal interim lambda-return algorithm; deliberately expensive.

    At each horizon h, recompute all interim targets and replay their sequential
    updates from initial. The n-step bootstrap uses the actual weights at time
    t+n-1, NOT the temporary helper weights used in this replay. This reference
    uses only the prefix observed so far, but stores and recomputes old data.
    """
    validate(features, rewards, initial, gamma, lam, alpha)
    history = [list(initial)]
    for horizon in range(1, len(rewards) + 1):
        helper = list(initial)
        for t in range(horizon):
            target, reward_sum = 0.0, 0.0
            remaining = horizon - t
            for n in range(1, remaining + 1):
                reward_sum += gamma ** (n - 1) * rewards[t + n - 1]
                bootstrap = dot(history[t + n - 1], features[t + n])
                n_step = reward_sum + gamma ** n * bootstrap
                mixture = lam ** (n - 1)
                if n < remaining:
                    mixture *= 1 - lam
                target += mixture * n_step
            error = target - dot(helper, features[t])
            helper = [w + alpha * error * x for w, x in zip(helper, features[t])]
        history.append(helper)
    return history
```

线性特征使一次更新关于旧权重呈仿射形式，其线性部分为 $A_t=I-\alpha x_tx_t^\top$。新数据对旧目标产生的修正，在到达当前权重前会经过这些更新矩阵。因此旧资格需要先乘 $A_t$，不能只乘标量衰减。

$$
\begin{aligned}z_t&=x_t+\gamma\lambda A_tz_{t-1}\\&=\gamma\lambda z_{t-1}+\left(1-\alpha\gamma\lambda z_{t-1}^\top x_t\right)x_t.\end{aligned}
$$

这就是荷兰迹。内积使用旧迹。减去的项处理先前更新与当前特征重叠造成的影响；它不是任意选择的稳定性裁剪。

迹修正还不够。记 $V=w_t^\top x_t$、$V'=w_t^\top x_{t+1}$，并保留上一转移计算的 $V_{\rm old}=w_{t-1}^\top x_t$。它是对当前状态的旧预测。新目标的变化涉及 $R+\gamma V'-V_{\rm old}=\delta+(V-V_{\rm old})$。加上旧更新已造成的预测变化补偿，得到完整更新。

$$
\begin{aligned}\delta&=R+\gamma V'-V,\\w^+&=w+\alpha(\delta+V-V_{\rm old})z-\alpha(V-V_{\rm old})x,\\V_{\rm old}&\leftarrow V'.\end{aligned}
$$

最后保存的是更新前计算的下一状态预测，不能重新用新权重计算。第一步可初始化旧预测为零，因为此时迹等于当前特征，两项预测差修正恰好相消。

**算法：固定特征、常数步长的 true-online TD(λ)；终点特征为零。**

1. 初始化 $w=w_{\rm init}$，$z=0$，$V_{\rm old}=0$。
1. 每条转移 $(x,R,x')$：
  1. 用旧权重计算 $V=w^\top x$、$V'=w^\top x'$。
  1. 计算 $\delta=R+\gamma V'-V$。
  1. 保存旧迹内积 $b=z^\top x$。
  1. 更新 $z\leftarrow\gamma\lambda z+(1-\alpha\gamma\lambda b)x$。
  1. 更新 $w\leftarrow w+\alpha(\delta+V-V_{\rm old})z-\alpha(V-V_{\rm old})x$。
  1. 保存 $V_{\rm old}\leftarrow V'$。
1. 真正 episode 结束后：保留权重，清零资格迹和旧预测。

在这组线性条件下，true-online TD(λ) 在每个前缀都与上述在线前向算法相等。这不同于冻结前向目标。对于 $d$ 维稠密特征，它每步使用 $O(d)$ 时间和额外迹存储；字面参考在第 $h$ 步使用 $O(h^2d)$ 时间与 $O(hd)$ 历史存储。非线性网络、在线变化的特征和任意优化器不能直接继承这一精确结论。

同一函数分别实现传统 accumulating TD 与荷兰迹加完整修正；返回每条转移后的权重。

```python
def td_episode(features, rewards, initial, gamma, lam, alpha, true_online=True):
    """Online linear TD(lambda), returning weights after EVERY transition.

    Terminal features must be zero. A nonzero last feature denotes a truncated
    continuing segment with its current value used as bootstrap. No artificial
    parameter reset occurs inside the segment.
    """
    validate(features, rewards, initial, gamma, lam, alpha)
    weights = list(initial)
    trace = [0.0] * len(weights)
    old_value = 0.0
    history = [list(weights)]
    for t, reward in enumerate(rewards):
        x, next_x = features[t], features[t + 1]
        value, next_value = dot(weights, x), dot(weights, next_x)
        delta = reward + gamma * next_value - value
        if true_online:
            # The overlap is computed from the OLD trace.
            overlap = dot(trace, x)
            trace = [gamma * lam * z + (1 - alpha * gamma * lam * overlap) * xj
                     for z, xj in zip(trace, x)]
            correction = value - old_value
            weights = [w + alpha * (delta + correction) * z - alpha * correction * xj
                       for w, z, xj in zip(weights, trace, x)]
        else:
            trace = [gamma * lam * z + xj for z, xj in zip(trace, x)]
            weights = [w + alpha * delta * z for w, z in zip(weights, trace)]
        old_value = next_value  # PRE-update prediction, not the new prediction.
        history.append(list(weights))
    return history
```

<a id="lesson-example"></a>

## 6. 两个手算例子

先看冻结计算。两步轨迹特征为 $(1,0)$、$(0,1)$、终点 $(0,0)$。奖励为 0、1。固定权重为 $(0.2,0.4)$，取 $\gamma=0.9$、$\lambda=0.8$、$\alpha=0.1$。末步目标为 1；首步目标为 $0.9[(1-0.8)0.4+0.8\times1]=0.792$。前向总增量为 $(0.0592,0.06)$。

$$
\begin{aligned}\delta_0&=0+0.9\times0.4-0.2=0.16,\qquad\delta_1=1-0.4=0.6,\\z_0&=(1,0),\qquad z_1=(0.72,1),\\\Delta w&=0.1[0.16(1,0)+0.6(0.72,1)]=(0.0592,0.06).\end{aligned}
$$

两种求和次序得到同一个增量。这里尚未在第一步真正改变权重。

再看重复特征的在线例子。$x_0=x_1=1$，$x_2=0$，奖励为 1、2，初始 $w=0$，取 $\gamma=\lambda=1$、$\alpha=0.5$。第一步两种在线算法都得到 $w=0.5$。第二步传统迹为 2，TD 误差为 1.5，所以最终 $w=0.5+0.5\times1.5\times2=2$。

True-online 第二步的荷兰迹为 $1+(1-0.5\times1)=1.5$。当前预测为 0.5，保存的旧预测为 0。完整更新为 $0.5+0.5(1.5+0.5)1.5-0.5\times0.5=1.75$。在线前向参考先用完整目标 3 把初值更新到 1.5，再用第二个目标 2 更新到 1.75。

| 算法或参考 | 第一步后 | 第二步后／总增量 | 为何不同 |
| --- | --- | --- | --- |
| 冻结前向与冻结后向 | 暂不应用增量 | 2.5 | 所有预测保持为零，总增量为 0.5×(3+2)。 |
| 传统在线 TD(1) | 0.5 | 2.0 | 权重改变，但仍使用 accumulating trace。 |
| 在线前向参考 | 0.5 | 1.75 | 每个前缀更新目标，并从初值顺序重算。 |
| True-online TD(1) | 0.5 | 1.75 | 用荷兰迹与修正高效实现同一在线参考。 |

这些数字区分的是算法定义。它们不能单独说明哪个算法在所有任务中学得更好。若只验证最终权重，也可能遗漏错误抵消；配套测试同时比较每个前缀，并覆盖非零初始化、稠密特征和非终止窗口。

<a id="lesson-control"></a>

## 7. 从预测到 SARSA 与 Watkins 控制

SARSA 把预测对象换为实际策略的动作价值。下一动作 $A_{t+1}$ 来自将要执行的行为策略，目标使用对应 $Q(S_{t+1},A_{t+1})$。表格 accumulating 迹为每个状态动作对保存一项资格。

$$
\begin{aligned}\delta_t^{\rm Sarsa}&=R_{t+1}+\gamma Q_t(S_{t+1},A_{t+1})-Q_t(S_t,A_t),\\z_t(s,a)&=\gamma\lambda z_{t-1}(s,a)+\mathbf1\{s=S_t,a=A_t\},\\Q_{t+1}(s,a)&=Q_t(s,a)+\alpha\delta_t^{\rm Sarsa}z_t(s,a).\end{aligned}
$$

策略随 Q 改变时，固定策略预测的前向解释成为局部解释。实际下一动作需要在明确的参数时序下选择。真正终止不选择下一动作，bootstrap 直接为零。

Q-learning 的一步目标却使用贪心延续 $\max_aQ(S_{t+1},a)$。若后续实际执行非贪心动作，沿这条轨迹继续累计奖励就不再直接对应贪心策略。Watkins Q(λ) 用剪迹处理这种不匹配：保留当前一步 max 目标更新，再阻止后续非贪心延续的误差继续传给更早的状态动作。

**算法：Watkins accumulating Q(λ)；剪迹发生在当前更新之后。**

1. 资格表 $z$ 保存已由上一转移衰减或剪断的旧资格。
1. 执行当前动作，获得奖励与下一状态。
1. 若非终止，用更新前 $Q$ 选择下一动作 $a'$。
1. 用同一旧 $Q$ 判断 $a'$ 是否达到下一状态的最大值；并列最大也算贪心。
1. 计算 $\delta=R+\gamma\max_bQ(s',b)-Q(s,a)$。
1. 令 $z(s,a)\leftarrow z(s,a)+1$。
1. 对全部状态动作对更新 $Q\leftarrow Q+\alpha\delta z$。
1. 若真正终止或下一动作非贪心，令 $z=0$；否则令 $z\leftarrow\gamma\lambda z$。
1. 执行已选定的下一动作。

例如当前 Q 为 0，下一状态两动作价值为 2 和 1，实际选择价值为 1 的动作，奖励为 0，$\gamma=0.9$、$\alpha=0.1$。SARSA 将当前值更新为 0.09，并保留衰减迹；Watkins 更新为 0.18，然后剪断旧迹。若在当前更新前清零，会连合法的一步信用也删掉。随机探索恰好选到并列最大动作时，不应仅因它来自探索分支就剪迹。

SARSA 与 Watkins 单步更新共用表格接口；下一动作贪心性在更新前判断。

```python
def control_trace_step(q, trace, state, action, reward, next_state, next_action,
                       gamma=0.9, lam=0.8, alpha=0.1, watkins=False, terminal=False):
    """Tabular accumulating Sarsa(lambda) or Watkins Q(lambda), one transition.

    Input trace is already decayed/cut by the preceding step. next_action must
    be selected using PRE-update q. For Watkins, determine whether it ties for
    the maximum before changing q. Cutting affects FUTURE credit only.
    """
    key = (state, action)
    if key not in q:
        raise ValueError('current state-action pair missing from q')
    greedy_next = False
    if terminal:
        target = reward
    else:
        next_values = [v for (s, _), v in q.items() if s == next_state]
        if not next_values or (next_state, next_action) not in q:
            raise ValueError('next state-action values required')
        greedy_next = q[(next_state, next_action)] == max(next_values)
        bootstrap = max(next_values) if watkins else q[(next_state, next_action)]
        target = reward + gamma * bootstrap
    delta = target - q[key]
    active = {pair: trace.get(pair, 0.0) for pair in q}
    active[key] += 1.0
    updated = {pair: value + alpha * delta * active[pair] for pair, value in q.items()}
    keep = not terminal and (not watkins or greedy_next)
    carry = {pair: gamma * lam * value if keep else 0.0 for pair, value in active.items()}
    return dict(q=updated, trace=carry, delta=delta, greedy_next=greedy_next)
```

频繁探索会使 Watkins 的有效信用范围很短。其他 off-policy 多步方法使用重要性比、截断比率或期望分支来处理策略差异。它们有不同目标与稳定性条件。不能只把 SARSA 的下一动作值改成 max，同时无条件保留所有旧迹，就称为 Watkins Q(λ)。

<a id="lesson-off-policy"></a>

## 8. Tree-backup 与 Retrace：改变误差的传播系数

设数据由行为策略 $\mu$ 产生，要学习目标策略 $\pi$ 的动作价值。先冻结 $Q$ 和两种策略。一步误差用目标策略的动作期望；延长目标时，还要决定后续真实动作对更早预测有多大影响。记 $Q_t=Q(S_t,A_t)$、$\rho_t=\pi(A_t\mid S_t)/\mu(A_t\mid S_t)$。行为策略必须覆盖所需的目标动作；没有采到的动作不能靠截断比率补出来。

$$
\begin{aligned}\bar Q(S_{t+1})&=\sum_a\pi(a\mid S_{t+1})Q(S_{t+1},a),\\\delta_t^\pi&=R_{t+1}+\gamma\bar Q(S_{t+1})-Q_t,\\G_t^c-Q_t&=\sum_{k=t}^{T-1}\gamma^{k-t}\left(\prod_{i=t+1}^{k}c_i\right)\delta_k^\pi.\end{aligned}
$$

空乘积为一，所以当前一步误差不乘后续动作比率。终止时动作期望为零；非终止窗口保留尾部期望。这里的c是信用传播系数，不是折扣，也不是奖励。

这不是把未来奖励简单相加。每个 $\delta_k^\pi$ 已经减去实际动作的当前Q，并加入目标动作期望。传播系数再决定沿行为轨迹采到的后续误差可以传多远。交换求和后，旧预测梯度在到达当前动作时乘 $\gamma c_t$。

$$
G_t^c-Q_t=\delta_t^\pi+\gamma c_{t+1}(G_{t+1}^c-Q_{t+1}),\qquad z_t=\gamma c_tz_{t-1}+\nabla Q_t
$$

前向使用进入下一动作的c；后向使用进入当前动作的c。两个下标不能互换。对于冻结Q，误差和与梯度迹给出同一个总增量。逐步改变参数后的等价性另需证明。

| 方法 | 传播系数c | 效果与条件 |
| --- | --- | --- |
| 逐决策重要性采样 | $\lambda\rho_t$ | 策略差异被显式校正，但比率乘积可能迅速增大。 |
| Tree-backup(λ) | $\lambda\pi(A_t\mid S_t)$ | 期望分支不需要除以行为概率；即使完全同策略，随机动作概率仍会衰减旧信用。 |
| Retrace(λ) | $\lambda\min(1,\rho_t)$ | 近同策略时保留长信用；目标概率小于行为概率时减少传播。 |

例如 $\lambda=0.8$、$\pi=\mu$，采到的动作概率为0.1。Tree-backup的系数为0.08；Retrace为0.8。若 $\pi=0.8$、$\mu=0.01$，重要性采样系数变成64，Retrace仍为0.8。截断传播减少长比率乘积，没有删掉每条误差中的目标动作期望。

Munos等的Retrace原文证明了相应表格算子的收缩性质与条件下的控制收敛。这个“安全”有具体数学含义。Touati等随后给出了Tree-backup和Retrace在线性函数逼近中仍可发散的例子，并构造梯度版本。把c限制在一以内不能独自解决函数逼近、bootstrap和off-policy的联合不稳定。

固定误差的独立求和、Tree-backup／Retrace系数、on-policy Q(σ)与V-trace的有限轨迹计算。

```python
def error_kernel_targets(values, deltas, incoming, gamma):
    """Q targets from fixed errors and incoming c_t; c_0 is never used.

    G_t - Q_t = delta_t + gamma*c_(t+1)*(G_(t+1)-Q_(t+1)).
    This is an independently written finite-path calculation, not a learner.
    """
    n = len(deltas)
    if len(values) != n or len(incoming) != n:
        raise ValueError('equal-length values, errors, and incoming coefficients required')
    if not 0 <= gamma <= 1 or any(c < 0 or not math.isfinite(c) for c in incoming):
        raise ValueError('gamma in [0,1] and finite nonnegative coefficients required')
    targets, correction = [0.0] * n, 0.0
    for t in range(n - 1, -1, -1):
        carry = gamma * incoming[t + 1] if t + 1 < n else 0.0
        correction = deltas[t] + carry * correction
        targets[t] = values[t] + correction
    return targets


def offpolicy_trace_coefficients(target_probs, behavior_probs, lam, method):
    if len(target_probs) != len(behavior_probs) or not 0 <= lam <= 1:
        raise ValueError('matching action probabilities and lambda in [0,1] required')
    coefficients = []
    for pi, mu in zip(target_probs, behavior_probs):
        if not 0 <= pi <= 1 or not 0 < mu <= 1:
            raise ValueError('sampled actions require positive behavior probability')
        ratio = pi / mu
        if method == 'is':
            coefficient = lam * ratio
        elif method == 'tree':
            coefficient = lam * pi
        elif method == 'retrace':
            coefficient = lam * min(1.0, ratio)
        else:
            raise ValueError('method must be is, tree, or retrace')
        coefficients.append(coefficient)
    return coefficients


def q_sigma_targets(q_taken, expected_next, rewards, next_probs, gamma, lam, sigma):
    """Frozen ON-POLICY Q(sigma) with geometric lambda mixing.

    sigma controls sampled/expected actions; lambda controls return length.
    No off-policy Q(sigma) correction is implemented here.
    """
    n = len(rewards)
    if (len(q_taken) != n + 1 or len(expected_next) != n
            or len(next_probs) != n or not 0 <= sigma <= 1 or not 0 <= lam <= 1):
        raise ValueError('T+1 sampled Q and T expectations/probabilities required')
    deltas = [r + gamma * (sigma * q_taken[t + 1] + (1 - sigma) * expected_next[t])
              - q_taken[t] for t, r in enumerate(rewards)]
    incoming = [0.0] + [lam * (sigma + (1 - sigma) * next_probs[t])
                         for t in range(n - 1)]
    return error_kernel_targets(q_taken[:-1], deltas, incoming, gamma)


def vtrace_targets(rewards, values, discounts, ratios, rho_cap=1.0, c_cap=1.0):
    """Finite V-trace target and actor advantage, with explicit tail bootstrap.

    The continuation coefficient c_t is indexed at the CURRENT action, unlike
    the incoming coefficient c_(t+1) in the action-value kernel above.
    rho_cap=None leaves the error correction ratio unclipped.
    """
    n = len(rewards)
    if len(values) != n + 1 or len(discounts) != n or len(ratios) != n:
        raise ValueError('T+1 values and T rewards/discounts/ratios required')
    if c_cap <= 0 or (rho_cap is not None and rho_cap < c_cap):
        raise ValueError('0 < c_cap <= rho_cap required')
    if any(not 0 <= d <= 1 for d in discounts) or any(r < 0 for r in ratios):
        raise ValueError('discounts in [0,1] and nonnegative ratios required')
    clipped = [min(rho_cap, r) if rho_cap is not None else r for r in ratios]
    vs, correction = [0.0] * n, 0.0
    for t in range(n - 1, -1, -1):
        delta = rewards[t] + discounts[t] * values[t + 1] - values[t]
        correction = clipped[t] * delta + discounts[t] * min(c_cap, ratios[t]) * correction
        vs[t] = values[t] + correction
    next_targets = vs[1:] + [values[-1]]
    advantages = [rho * (r + d * tail - value)
                  for rho, r, d, tail, value in zip(clipped, rewards, discounts,
                                                    next_targets, values[:-1])]
    return dict(targets=vs, advantages=advantages)


def clipped_policy(target, behavior, rho_cap):
    """Tabular V-trace fixed-point policy, not the original target policy."""
    if len(target) != len(behavior) or not target or rho_cap <= 0:
        raise ValueError('matching nonempty distributions and positive rho cap required')
    if (any(p < 0 for p in target) or any(p <= 0 for p in behavior)
            or abs(sum(target) - 1) > 1e-10 or abs(sum(behavior) - 1) > 1e-10):
        raise ValueError('normalized distributions with behavior support required')
    masses = [min(pi, rho_cap * mu) for pi, mu in zip(target, behavior)]
    return [mass / sum(masses) for mass in masses]
```

<a id="lesson-vtrace-sigma"></a>

## 9. 两种不同的选择：Q(σ)的动作期望与V-trace的策略校正

$\lambda$ 混合回报长度，$\sigma$ 混合后继动作的采样与期望。先取on-policy、冻结Q的情形。记 $\bar Q_{t+1}$ 为目标策略动作期望。Q(σ)的一步误差和多步传播同时改变，而不是只修改其中一个。

$$
\begin{aligned}\delta_t^\sigma&=R_{t+1}+\gamma\left[\sigma_{t+1}Q_{t+1}+(1-\sigma_{t+1})\bar Q_{t+1}\right]-Q_t,\\c_i^\sigma&=\lambda\left[\sigma_i+(1-\sigma_i)\pi(A_i\mid S_i)\right],\\G_t^{\sigma,\lambda}-Q_t&=\sum_{k=t}^{T-1}\gamma^{k-t}\left(\prod_{i=t+1}^{k}c_i^\sigma\right)\delta_k^\sigma.\end{aligned}
$$

这是在原n-step Q(σ)传播式上作几何λ混合的冻结on-policy表达。σ=1得到Sarsa的采样延续；σ=0得到Tree-backup的期望分支。λ=0只保留一步，但σ仍决定一步目标。

De Asis等原文还给出了off-policy Q(σ)的修正。上式和本章代码只实现on-policy端点；不能在行为与目标策略不同时直接使用它。做比较时至少固定回报长度、σ、λ、策略匹配程度四项。改变σ通常是在改变采样噪声，改变λ还会改变bootstrap跨度。

V-trace解决另一个问题。分布式actor用较旧策略收集轨迹，learner用新策略训练状态价值。每条TD误差需要当前动作的校正；误差继续向前面的状态传播时还需要另一个系数。两类截断分工不同。

$$
\begin{aligned}\bar\rho_t&=\min(\rho_{\max},\rho_t),\qquad c_t=\min(c_{\max},\rho_t),\\\delta_t^V&=\bar\rho_t\left[R_{t+1}+\gamma_{t+1}V_{t+1}-V_t\right],\\v_t^{\rm VT}-V_t&=\delta_t^V+\gamma_{t+1}c_t\left(v_{t+1}^{\rm VT}-V_{t+1}\right).\end{aligned}
$$

取0<cmax≤ρmax；窗口边界取vVT=V。这里的 $c_t$ 乘当前动作比率，与上一节动作价值形式中的 $c_{t+1}$ 不同。原IMPALA常取两个上限为一。

在表格评价条件下，$\rho_{\max}$ 会改变V-trace固定点对应的策略，而 $c_{\max}$ 主要影响传播长度与收敛速度。固定点策略为：

$$
\pi_{\rho_{\max}}(a\mid s)=\frac{\min\{\rho_{\max}\mu(a\mid s),\pi(a\mid s)\}}{\sum_b\min\{\rho_{\max}\mu(b\mid s),\pi(b\mid s)\}}
$$

ρmax足够大时恢复目标策略；截断较强时可能得到介于行为与目标之间的策略。不能把两个截断都称为“只降低方差而不改变目标”。

单步bandit就能看出目标变化。取 $\mu=(0.8,0.2)$、$\pi=(0.2,0.8)$，两动作奖励为0、1。$\rho_{\max}=1$ 时截断质量为 $(0.2,0.2)$，归一化策略为 $(0.5,0.5)$，价值固定点为0.5；原目标价值为0.8。因为没有后继步，改变c根本不能改变这个例子的固定点。

$$
\widehat A_t^{\rm VT}=\bar\rho_t\left[R_{t+1}+\gamma_{t+1}v_{t+1}^{\rm VT}-V_t\right]
$$

actor使用下一状态的V-trace目标构造动作回报。它不等于直接使用$v_t^{VT}-V_t$。实践中策略梯度比率上限还可以单独设置；本文小程序令它与ρmax一致。

IMPALA作者代码先计算行为与目标动作log-prob的差，再用反向scan形成窗口目标，最后stop_gradient。它保存短轨迹并进行批量训练。反向scan在数组上从后向前计算，不等于严格流式后向资格迹；前者依赖已保存的窗口，后者在新误差到达时更新一条压缩迹。

<a id="lesson-state-lambda"></a>

## 10. 状态相关与自适应λ：先写清目标，再决定长度

固定λ把每一处bootstrap都视为同样可靠。状态相关 $\lambda(S_t)$ 可以让可信状态提前截断，让预测偏差较大的状态依赖更多真实奖励。先冻结预测和λ函数，并采用到达状态的下标。

$$
\begin{aligned}G_t^\lambda&=R_{t+1}+\gamma_{t+1}\left[(1-\lambda_{t+1})V_{t+1}+\lambda_{t+1}G_{t+1}^\lambda\right],\\G_t^\lambda-V_t&=\sum_{k=t}^{T-1}\left(\prod_{i=t+1}^{k}\gamma_i\lambda_i\right)\delta_k,\\z_t&=\gamma_t\lambda_tz_{t-1}+\nabla V_t.\end{aligned}
$$

到达当前状态的γtλt衰减旧资格；离开当前状态的γt+1λt+1形成前向延续。终止折扣为零会删除尾值，但终止奖励仍要乘此前已经存在的资格。

在可精确表示的表格问题中，不同合法λ可以共享真实价值固定点。投影和函数逼近介入后，λ会影响逼近解，因而不能只把它解释为优化速度参数。在线适应λ还会使目标随学习改变；在旧状态分布上估计得很好的长度函数，在变化后的分布上可能已经过时。

White与White的λ-greedy从一个局部选择出发：下一状态的当前预测有平方偏差 $b^2$，完整未来回报方差为 $s_G^2$。把下一处bootstrap与未来Monte Carlo回报混合；在固定预测、相应条件独立约定下，需要最小化的λ相关项是：

$$
L(\lambda)=(1-\lambda)^2b^2+\lambda^2s_G^2,\qquad\frac{dL}{d\lambda}=-2(1-\lambda)b^2+2\lambda s_G^2,\qquad\lambda^*=\frac{b^2}{b^2+s_G^2}
$$

这是单处混合的局部最优，不是联合优化整条轨迹的证明。两项都为零时任何λ等价；配套程序选择零作为约定。

若 $b^2=4$、$s_G^2=1$，最优λ为0.8，局部目标为0.8；λ=0和λ=1的目标分别为4和1。实际算法没有真值，需要额外预测未来回报均值与二阶矩，再估计偏差和方差。本章小实验只核对已知统计量下的最小化，不声称复现完整λ-greedy。噪声、混叠或非平稳性会使这些统计估计本身成为学习问题。

Meta-gradient RL采用另一条路线：一次参数更新依赖λ，再用后来经验评价更新后的参数，并对这条学习路径求导。元目标、未来数据与导数截断必须明确。它与直接减少同一批次上的λ-return拟合误差不同；后者可能把λ调到更容易拟合但更偏的目标。

变量折扣／λ的前后向数值恒等式；局部bias–variance目标的已知统计量最优点。

```python
def variable_lambda_equivalence(features, rewards, weights, discounts, lambdas):
    """Frozen forward/backward identity with arrival-indexed gamma_t/lambda_t."""
    n = len(rewards)
    if len(discounts) != n + 1 or len(lambdas) != n + 1:
        raise ValueError('T+1 arrival-indexed discounts and lambdas required')
    validate(features, rewards, weights, 1.0, 1.0)
    if any(not 0 <= v <= 1 for v in [*discounts, *lambdas]):
        raise ValueError('discounts and lambdas must lie in [0,1]')
    values = [dot(weights, x) for x in features]
    targets, tail = [0.0] * n, values[-1]
    for t in range(n - 1, -1, -1):
        tail = rewards[t] + discounts[t + 1] * (
            (1 - lambdas[t + 1]) * values[t + 1] + lambdas[t + 1] * tail)
        targets[t] = tail
    forward, backward, trace = [[0.0] * len(weights) for _ in range(3)]
    for t in range(n):
        delta = rewards[t] + discounts[t + 1] * values[t + 1] - values[t]
        trace = [discounts[t] * lambdas[t] * z + x for z, x in zip(trace, features[t])]
        forward = [a + (targets[t] - values[t]) * x for a, x in zip(forward, features[t])]
        backward = [a + delta * z for a, z in zip(backward, trace)]
    return dict(targets=targets, forward=forward, backward=backward)


def greedy_lambda(bias_squared, return_variance):
    """Oracle optimum of the LOCAL bias/variance surrogate, not full lambda-greedy.

    White & White's algorithm must also learn return moments online. We supply
    those statistics as inputs here so the closed-form optimization is testable.
    """
    if any(not math.isfinite(v) or v < 0 for v in (bias_squared, return_variance)):
        raise ValueError('finite nonnegative squared bias and variance required')
    total = bias_squared + return_variance
    return bias_squared / total if total else 0.0
```

<a id="lesson-expected-traces"></a>

## 11. Expected Eligibility Traces：信用能否复用于另一条过去路径？

普通迹只包含本次走过的状态。设两条路径在同一个状态汇合，后面的随机奖励与此前走哪条路径无关。当前奖励可以同时更新那些可能到达汇合状态的路径。期望资格迹学习的是“到达此状态时，过去资格通常是什么”，不是预测未来奖励的successor feature。

$$
\bar z(s)=\mathbb E[z_t\mid S_t=s],\qquad\Delta w=\alpha\delta_t\bar z(S_t),\qquad\min_\eta\ \frac12\|z_t-\bar z_\eta(S_t)\|^2
$$

保留普通迹作为训练标签，再用状态条件预测器估计它。η在这一式中仅表示预测器参数；下面的混合系数另记ξ，以免两种用途混淆。

van Hasselt等的均值与方差分析依赖 Markov 状态、固定预测参数与固定 Markov 策略 $\pi(a\mid s)$。当前 TD 误差的条件分布不能再依赖未纳入状态的历史。此时，给定完整状态，当前转移产生的 $\delta_t$ 与到达它之前产生的 $z_t$ 条件独立。因此：

$$
\mathbb E[\delta_tz_t\mid s]=\mathbb E[\delta_t\mid s]\mathbb E[z_t\mid s]=\mathbb E[\delta_t\bar z(s)\mid s]
$$

替换为精确条件均值后，保留条件平均更新，并消去历史路径随机性的一部分。方差不增是逐分量陈述；任意近似神经预测器和漂移参数不能自动继承无偏性。

最小数值例子：两条等概率历史的资格为 $(0.72,0,1)$ 和 $(0,0.72,1)$，因此期望资格为 $(0.36,0.36,1)$。汇合后的TD误差独立地取0或2。两种更新均值都为 $(0.36,0.36,1)$；前两分量的方差由0.3888降为0.1296。最后分量只受奖励噪声影响，方差仍为1。

反例同样重要。若表面观察相同，但走第一条历史时奖励必为2、走第二条必为0，那么普通迹平均更新是 $(0.72,0,1)$，期望迹变成 $(0.36,0.36,1)$。隐藏历史决定未来奖励，条件独立不成立，期望迹把信用分给了错误的路径。改善agent state可以恢复条件，单纯加长或平均迹不能解决状态混叠。

$$
y_t=(1-\xi)\bar z_\eta(S_t)+\xi\left(\gamma_t\lambda y_{t-1}+\nabla V_t\right),\qquad\xi\in[0,1]
$$

原文ET(λ,η)用η表示这处混合系数。这里改记ξ：ξ=1恢复普通迹，ξ=0使用预测的期望迹。它是递归混合，不是将两条独立算好的完整迹一次凸组合。

实现时需要选择预测器的状态输入、资格标签、更新时序与统计遗忘速度。全参数迹预测器的输出维度和主网络参数数目相同；它可能非常昂贵。表示学习还会改变资格所在的参数坐标，旧迹标签可能失效。应先在固定特征、路径汇合的小问题验证条件，再研究低维近似和状态变化。

精确枚举四种路径／奖励组合，不依赖随机采样；另保留状态混叠导致均值改变的反例。

```python
def expected_trace_enumeration(hidden_history=False):
    """Exact finite enumeration at a merging state, including an aliasing failure.

    Two equiprobable histories have traces (.72,0,1) and (0,.72,1).
    Markov case: reward in {0,2} is independent of the incoming history.
    Aliased case: the hidden incoming history determines reward 2 vs 0.
    Predictions and features are frozen; this is not an ET training benchmark.
    """
    traces = [[0.72, 0.0, 1.0], [0.0, 0.72, 1.0]]
    expected_trace = [sum(z[j] for z in traces) / 2 for j in range(3)]
    cases = [(0.5, 2.0, traces[0]), (0.5, 0.0, traces[1])] if hidden_history else [
        (0.25, reward, trace) for trace in traces for reward in (0.0, 2.0)]
    result = {}
    for name, use_expected in [('instantaneous', False), ('expected', True)]:
        updates = [(prob, [reward * z for z in (expected_trace if use_expected else trace)])
                   for prob, reward, trace in cases]
        mean = [sum(prob * row[j] for prob, row in updates) for j in range(3)]
        variance = [sum(prob * (row[j] - mean[j]) ** 2 for prob, row in updates)
                    for j in range(3)]
        result[name] = dict(mean=mean, variance=variance)
    return result
```

<a id="lesson-gradient-traces"></a>

## 12. 深度梯度资格迹：目标、辅助预测与三条递推

半梯度TD把bootstrap目标当常数，用 $\delta_tz_t$ 更新预测。非线性、off-policy和bootstrap组合时，这个方向不一定来自一个稳定的整体目标。Elelimy等的2025年工作先指定广义投影Bellman误差，再推导多步梯度算法。下面先取固定策略、on-policy、固定 $\gamma,\lambda$ 的情形；这是能够逐项检验的起点。

记 $\epsilon_t^\lambda=G_t^\lambda-V_w(S_t)$，其条件均值为 $\bar\epsilon_w^\lambda(s)$。辅助函数 $H_\eta(s)$ 估计这个条件平均误差。它既不是第二个价值目标网络，也不是任意保存梯度的变量。平方有共轭形式 $u^2=\max_h(2uh-h^2)$，于是限制辅助函数类后得到：

$$
\mathcal E_\lambda(w)=\max_{H\in\mathcal H}\ \mathbb E_{s\sim d}\left[2\bar\epsilon_w^\lambda(s)H(s)-H(s)^2\right]
$$

辅助函数类如果包含所有状态函数，会恢复均方条件Bellman误差；受限函数类定义相应的广义投影目标。状态分布d、策略、λ和函数类都是目标的一部分。

为什么不直接最小化一次采样TD误差的平方？条件误差平方的梯度涉及两个条件期望的乘积。用同一随机下一状态替代两个独立样本，通常得到额外协方差项。辅助预测把一个条件期望变成可学习的函数，避免要求环境从同一状态再独立采一次。它仍带来辅助估计误差与更新速度的选择。

将目标按二分之一缩放，用 $H_t=H_\eta(S_t)$。GTD2的前向主参数方向为 $-H_t\nabla_w\epsilon_t^\lambda$，辅助参数方向为 $(\epsilon_t^\lambda-H_t)\nabla_\eta H_t$。前者对包含bootstrap的误差求导；后者让辅助预测拟合多步误差。

$$
\begin{aligned}\nabla_w\delta_t&=\gamma\nabla_wV_w(S_{t+1})-\nabla_wV_w(S_t),\\z_t^H&=\gamma\lambda z_{t-1}^H+H_t,\\z_t^\eta&=\gamma\lambda z_{t-1}^\eta+\nabla_\eta H_t,\\\Delta w_t^{\rm GTD2}&=-z_t^H\nabla_w\delta_t,\\\Delta\eta_t&=\delta_tz_t^\eta-H_t\nabla_\eta H_t.\end{aligned}
$$

两条迹承担不同角色：标量迹累积辅助误差预测，参数迹累积辅助网络的输出梯度。主网络方向仍对当前一步误差完整求导；不能对下一状态预测stop-gradient后宣称实现同一GTD2。

推导后向形式仍然靠交换求和。固定 $w,\eta$ 后，$\nabla\epsilon_t^\lambda=\sum_{k\ge t}(\gamma\lambda)^{k-t}\nabla\delta_k$。有限窗口在 $T$ 处使用同一尾值 $G_T=V_w(S_T)$，并对这个尾值求导，因此 $\epsilon_T=0$ 且 $\nabla_w\epsilon_T=0$；真实终止则将尾值及其导数设为零。所有旧时刻的 $H_t$ 在当前误差梯度之前合并成标量迹；辅助网络的梯度同理合并。TDC在此基础上加入 $(\epsilon_t^\lambda-H_t)\nabla V_t$ 修正，后向形式还需第三条价值梯度迹。

$$
\begin{aligned}z_t^w&=\gamma\lambda z_{t-1}^w+\nabla_wV_t,\\\Delta w_t^{\rm TDC}&=\delta_tz_t^w-H_t\nabla_wV_t-z_t^H\nabla_w\delta_t,\\\Delta\eta_t^{\rm TDRC}&=\delta_tz_t^\eta-H_t\nabla_\eta H_t-\beta\eta_t.\end{aligned}
$$

TDC 使用同一辅助更新；TDRC 进一步加入辅助参数正则。主网络不能只保留 $\delta_tz_t^w$ 而删掉后两项。TDC 的校正依赖辅助估计，非线性情况下没有自动的全局收敛保证。

**算法：两个网络的更新方向先共同算完，避免第二个方向意外使用第一个网络的新参数。**

1. 收到转移后，先用更新前的主网络与辅助网络：
  1. 计算 $V_t,V_{t+1},H_t,\delta_t$，以及 $\nabla_wV_t,\nabla_w\delta_t,\nabla_\eta H_t$。
  1. 更新标量迹 $z_t^H$、辅助梯度迹 $z_t^\eta$、主梯度迹 $z_t^w$。
  1. 从同一组旧参数计算两个参数方向。
  1. 选择 GTD2／TDC／TDRC 规则，再分别应用主、辅助步长。
  1. 若真实终止，先完成奖励信用，再按算法边界清迹。
1. 控制版本还必须指定目标动作、策略不一致处理和剪迹时序。

原文Theorem 6.1明确假设两个参数集合在episode内不变，证明的是总增量相同。它不等于线性true-online的每前缀等价。本文用tanh主预测和tanh辅助预测独立计算前向目标及其导数，再与三条迹比较。GTD2还用中心有限差分核对双网络鞍点方向。数值结果检验代数与实现，不检验深度控制性能。

作者QRC实现把主预测换成动作价值，完整求导max-bootstrap误差，并保存`h_trace`、`grad_h_trace`、`grad_q_trace`。配置中的`gradient_correction` 和 `reg_coeff`分别控制梯度校正与辅助正则。代码在完成当前更新后，对终止、环境截断或指定的非贪心事件清迹；环境截断仍可保留bootstrap。这是该实验实现的协议，不能推广为所有日志分段都需要清迹。

论文的MuJoCo前向方法使用保存的窗口与PPO框架；MinAtar后向方法逐步更新。两者不能直接当作相同存储协议。MinAtar的QRC与StreamQ对照还涉及JAX／PyTorch实现差异，作者明确说明每秒步数比较有框架混杂。研究时应分别检查学习曲线、数据复用、参数量、辅助计算和更新时间。

冻结非线性GTD2／TDC／TDRC总增量检查，及主／辅助参数有限差分；不是作者基准复现。

```python
def gradient_trace_equivalence(features, rewards, weights, auxiliary, gamma, lam,
                               correction=False, regularization=0.0):
    """Frozen nonlinear GTD2/TDC/TDRC total-increment identities, on-policy.

    V=tanh(w dot x), H=tanh(theta dot x). Independently differentiate the
    recursive lambda target for the forward view, then accumulate the backward
    scalar-H trace and parameter traces. No parameters change during this call.
    The saddle objective is half-scaled: sum(H*delta_lambda - H**2/2).
    This is a mathematical unit experiment, not the authors' deep RL benchmark.
    """
    validate(features, rewards, weights, gamma, lam)
    if len(auxiliary) != len(weights) or regularization < 0:
        raise ValueError('matching auxiliary dimension and nonnegative penalty required')
    n, d = len(rewards), len(weights)
    value_rows = [nonlinear_value_gradient(weights, x) for x in features]
    values, gradients = zip(*value_rows)
    h_rows = [nonlinear_value_gradient(auxiliary, x) for x in features[:-1]]
    targets, target_gradients = [0.0] * n, [[0.0] * d for _ in range(n)]
    tail, tail_gradient = values[-1], list(gradients[-1])
    for t in range(n - 1, -1, -1):
        tail = rewards[t] + gamma * ((1 - lam) * values[t + 1] + lam * tail)
        tail_gradient = [gamma * ((1 - lam) * g + lam * old)
                         for g, old in zip(gradients[t + 1], tail_gradient)]
        targets[t], target_gradients[t] = tail, list(tail_gradient)
    fw, fh, bw, bh, trace_v, trace_h = [[0.0] * d for _ in range(6)]
    scalar_trace, objective = 0.0, 0.0
    for t, (h, grad_h) in enumerate(h_rows):
        grad_v = gradients[t]
        error = targets[t] - values[t]
        grad_error = [g - v for g, v in zip(target_gradients[t], grad_v)]
        objective += h * error - 0.5 * h * h - 0.5 * regularization * dot(auxiliary, auxiliary)
        fw = [a - h * ge + ((error - h) * gv if correction else 0.0)
              for a, ge, gv in zip(fw, grad_error, grad_v)]
        fh = [a + (error - h) * gh - regularization * theta
              for a, gh, theta in zip(fh, grad_h, auxiliary)]
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        grad_delta = [gamma * gn - gv for gn, gv in zip(gradients[t + 1], grad_v)]
        scalar_trace = gamma * lam * scalar_trace + h
        trace_h = [gamma * lam * z + g for z, g in zip(trace_h, grad_h)]
        trace_v = [gamma * lam * z + g for z, g in zip(trace_v, grad_v)]
        bw = [a - scalar_trace * gd + (delta * zv - h * gv if correction else 0.0)
              for a, gd, zv, gv in zip(bw, grad_delta, trace_v, grad_v)]
        bh = [a + delta * zh - h * gh - regularization * theta
              for a, zh, gh, theta in zip(bh, trace_h, grad_h, auxiliary)]
    return dict(forward_w=fw, backward_w=bw, forward_h=fh, backward_h=bh,
                objective=objective)
```

<a id="lesson-actor-recurrent"></a>

## 13. 深度actor–critic、GAE与RTRL／RTU的接口

critic 的迹累积价值梯度。actor 的迹累积动作对数概率梯度，二者维度和含义可能不同。为明确折扣约定，先取 episodic 目标 $J=\mathbb E[\sum_t\gamma^tR_{t+1}]$，并在一条轨迹内固定策略参数。令 $\psi_t=\nabla_\theta\log\pi_\theta(A_t\mid S_t)$。

$$
\nabla_\theta J=\mathbb E\!\left[\sum_t\gamma^t\bigl(G_t-b(S_t)\bigr)\psi_t\right]
$$

动作无关的 baseline 不改变期望梯度。这个表达包含策略产生数据的概率路径；不是对环境奖励数值直接求动作导数。

以固定 critic 的 TD 误差和来构造优势估计，再交换求和，可以得到 actor 的后向形式。以下明确保留初始时刻目标中的 $\gamma^t$ 权重。若改成平均奖励或不同状态采样目标，应重新推导权重，而不是静默删掉它。

$$
\begin{aligned}\widehat A_t^\lambda&=\sum_{k=t}^{T-1}(\gamma\lambda_\theta)^{k-t}\delta_k,\\z_t^\theta&=\gamma\lambda_\theta z_{t-1}^\theta+\gamma^t\psi_t,\\\sum_t\gamma^t\widehat A_t^\lambda\psi_t&=\sum_t\delta_tz_t^\theta,\qquad\theta^+=\theta+\alpha_\theta\delta_tz_t^\theta.\end{aligned}
$$

最后一行的总和恒等式使用固定轨迹内的预测和梯度。逐步改变 actor 与 critic 后是在线算法近似。有限 episode 中 λ=1 给出 Monte Carlo 优势；λ<1 时近似 critic 的 bootstrap 误差会进入优势估计。

这组优势误差和就是GAE的基本计算。批量actor–critic通常先保存窗口，以冻结预测反向计算优势，再做多轮更新；流式actor–critic在误差到来时使用过去的动作梯度迹。相同误差和不意味着相同学习协议。窗口bootstrap、策略更新次数、概率比率与目标的状态权重都会改变实际方向。

固定score与critic的折扣初始状态目标：独立前向优势与后向actor迹总和。

```python
def actor_trace_equivalence(scores, rewards, values, gamma, lam):
    """Frozen discounted-start-state policy scores; actor is not updated here."""
    if len(scores) != len(rewards) or not scores or len(values) != len(rewards) + 1:
        raise ValueError('T score vectors/rewards and T+1 values required')
    d = len(scores[0])
    if not d or any(len(score) != d for score in scores):
        raise ValueError('score dimensions must match')
    advantages = [g - v for g, v in zip(lambda_returns(rewards, values, gamma, lam), values)]
    forward, backward, trace = [[0.0] * d for _ in range(3)]
    for t, score in enumerate(scores):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        trace = [gamma * lam * z + gamma ** t * s for z, s in zip(trace, score)]
        forward = [a + gamma ** t * advantages[t] * s for a, s in zip(forward, score)]
        backward = [a + delta * z for a, z in zip(backward, trace)]
    return dict(forward=forward, backward=backward, advantages=advantages)
```

深度实现每步先计算主网络当前输出的梯度，再将数值梯度加入独立的资格缓冲。普通半梯度迹不保留历史自动求导图；否则内存会随时间增长。共享actor／critic主干时，两个目标的方向要明确组合。对过去各条梯度先分别做Adam变换，再累加，与先累加资格方向后交给Adam通常不同；优化器选择属于算法定义。

若输入是递归状态 $h_t=f_\theta(h_{t-1},u_t)$，当前输出的梯度还要经过状态构造。RTRL 保存 $E_t=\partial h_t/\partial\theta$。它回答旧输入通过递归计算怎样影响当前输出；资格迹则进一步把不同输出时刻的梯度按回报信用组合。两种记忆不能互相替代。

$$
\begin{aligned}E_t&=\frac{\partial f_\theta}{\partial h_{t-1}}E_{t-1}+\left.\frac{\partial f_\theta}{\partial\theta}\right|_{h_{t-1}},\\g_t&=E_t^\top\nabla_hv_\theta(h_t)+\left.\nabla_\theta v_\theta(h_t)\right|_{h_t},\\z_t&=\gamma\lambda z_{t-1}+g_t.\end{aligned}
$$

这些导数以固定参数的计算图为精确参照。训练中参数不断改变时，过去活动与梯度均有其生成时刻；TBPTT 的 detach 又会截断部分路径。不能因保存了长资格迹，就认定被截断的递归参数梯度已经恢复。

最小例子：$h_t=0.5h_{t-1}+bu_t$，$h_0=0$，输入依次为 1、0、0。最终 $h=0.25b$，所以 $\partial h/\partial b=0.25$。取 $b=0.2$、目标 1，平方损失梯度为 $(0.05-1)0.25=-0.2375$。若在第一次输入后 detach，保留的活动仍是 0.05，但后两步输入为零，截断梯度变成零。

固定参数的最小递归敏感度及 detach 对照；测试用中心有限差分核验完整梯度。

```python
def recurrent_sensitivity(inputs, b, target=1.0, decay=0.5, truncate_after=None):
    """Fixed-parameter h_t=decay*h_(t-1)+b*u_t, and exact dh_t/db.

    truncate_after cuts the derivative before that zero-based input index.
    The numerical hidden state is retained, demonstrating detach vs reset.
    """
    h, sensitivity = 0.0, 0.0
    for t, value in enumerate(inputs):
        if t == truncate_after:
            sensitivity = 0.0
        h = decay * h + b * value
        sensitivity = decay * sensitivity + value
    loss = 0.5 * (h - target) ** 2
    return dict(h=h, sensitivity=sensitivity, loss=loss,
                gradient=(h - target) * sensitivity)
```

RTU的“trace”指递归状态对参数的敏感度。两维实值旋转块实现复值对角递归，使特定结构下的RTRL更便宜。作者实现将状态活动与grad_memory共同递推，并用自定义求导接到当前输出。这个敏感度先给出价值或动作score的梯度，回报资格迹再把不同时刻的输出梯度组合。两块变量的维度、用途与重置条件都需分别规定。

2026年Farr等把RTU-RTRL接入QRC与流式actor–critic，并单独研究参数不断改变时的敏感度过时问题。固定参数下的精确RTRL不意味着沿更新中的参数轨迹仍等于“把全部历史重新用当前参数运行一次”的导数。原文用保存轨迹的重算参考做诊断；这项额外存储应计为评价成本。Masked MuJoCo结果仍低于批量PPO，因此它支持特定部分可观测设置下的流式可行性，尚不支持全面性能优势。

<a id="lesson-code"></a>

## 14. 实验入口与原始实现

Python 3.10+，仅标准库；每个入口对应正文中的一个可检查问题。

```sh
python3 credit_assignment_lab.py all
python3 credit_assignment_lab.py online
python3 credit_assignment_lab.py offpolicy
python3 credit_assignment_lab.py adaptive
python3 credit_assignment_lab.py expected
python3 credit_assignment_lab.py gradient
python3 credit_assignment_lab.py actor
python3 credit_assignment_lab.py recurrent
python3 credit_assignment_lab.py test
```

| 入口 | 核验对象 | 不据此声称 |
| --- | --- | --- |
| frozen / online | 冻结总和恒等式；true-online每前缀等价；传统TD差异。 | 神经网络的true-online等价。 |
| control / offpolicy | 合法一步信用与剪迹时序；传播系数；Q(σ)端点；V-trace固定点策略。 | 完整分布式IMPALA或Atari复现。 |
| adaptive | 变量折扣／λ恒等式；局部已知统计量的最优混合。 | 完整在线λ-greedy或meta-gradient复现。 |
| expected | Markov汇合状态的均值／逐分量方差；混叠反例。 | 任意近似期望迹都无偏。 |
| gradient | 非线性冻结双网络总增量与GTD2有限差分。 | 深度全局收敛或QRC性能优势。 |
| actor / recurrent | 固定score误差和；递归敏感度有限差分与detach反例。 | 在线变参梯度或RTU完整基准复现。 |

作者实验仓库的 totd.py 把步长 $\alpha$ 折进迹变量，并用 predprev 保存旧预测。配套代码使用迹外步长；在这里的常数步长条件下，二者通过 $z_{\rm author}=\alpha z$ 对应。比较源码时应先统一这个定义，再比较修正项。实验入口、随机 MDP 配置与结果处理位于同一作者仓库；完整论文实验需要其依赖和配置。

<a id="lesson-branches"></a>

## 15. 可区分机制的研究路线

| 研究问题 | 直接改变的对象 | 不能混同的对象 |
| --- | --- | --- |
| 奖励很晚，更新只到最近一步 | 回报长度、λ 混合、资格迹。 | 不是单纯增加网络深度。 |
| 记忆存在，但最初写入参数没梯度 | RTRL、BPTT、截断边界。 | 加长 TD 迹不能恢复已删掉的结构梯度。 |
| 同一轨迹上前后向更新不一致 | 参数冻结约定、在线目标、荷兰迹与修正。 | 不能用数值接近替代精确等价。 |
| 没有样本缓冲仍超出实时预算 | 迹存储、梯度计算、递归结构与数值尺度。 | 流式约束不等于某一种信用算法。 |
| 固定步长或 λ 在变化后失效 | 步长适应、meta-gradient、Metatrace。 | 元迹追踪学习参数影响，和价值资格迹分开存。 |
| off-policy 长迹方差或偏差很大 | 目标策略修正、截断、梯度 TD 或 emphatic 方法。 | 一条未经修正的 on-policy 迹不提供稳定性保证。 |

同一算法可以同时属于几条研究线。Metatrace 就为步长建立元时间信用；流式 actor–critic 可以同时使用价值迹、策略迹和递归敏感度。教材按这些对象分别讲解，是为了明确接口，不是把算法划进互不相交的名词类别。

| 可证伪问题 | 受控改变 | 判断与竞争解释 |
| --- | --- | --- |
| 长迹改善来自更快奖励传播，还是更大更新？ | 固定表示，增加奖励延迟；匹配实际更新尺度。 | 比较首状态预测误差和回报；仅迹范数增大不足以支持机制。 |
| Retrace保留长信用是否提高效率？ | 固定Q和轨迹覆盖，仅改变π与μ差异；比较Tree-backup。 | 记录信用衰减、方差、近同策略学习速度；加进逼近后重新检查稳定性。 |
| 期望迹减少路径噪声，还是把不同隐藏状态混到一起？ | Markov汇合与观察混叠两组环境；保持相同奖励边际分布。 | 分别测均值偏差和方差；均方误差降低不证明信用无偏。 |
| 辅助误差预测改善了目标方向吗？ | 同样λ与算力，比较半梯度、GTD2、TDC与TDRC。 | 同时记录辅助误差、主预测误差、实际参数步幅；额外网络容量是竞争解释。 |
| 远处提示学不到，是状态遗失还是梯度被截断？ | 固定总参数预算，分别改变记忆跨度、RTRL导数与回报λ。 | 记录提示可解码性、敏感度误差、回报；这些量不能互相替代。 |
| 自适应λ在环境改变后仍有效吗？ | 同一生命改变噪声与预测可靠性；计入辅助统计的适应速度。 | 比较固定λ、已知统计的局部上限参照和在线估计；未来真值只能用于评价。 |

先完成有限轨迹的恒等式与反例，再进入固定策略预测，随后才让策略和表示共同学习。后一阶段还需报告新能力形成、旧能力保留、恢复速度及每步资源。资格迹解决了哪一条信用路径，应由受控改变检验，而不是由算法名称或一条更高的奖励曲线推断。

<a id="lesson-check"></a>

## 16. 失败边界与思考题

| 观察到的错误 | 检查方法 |
| --- | --- |
| 终点奖励只更新最后状态 | 是否在使用该奖励前清空了旧迹。 |
| True-online 在第二步才出错 | 保存的是否是更新前 V′；内积是否使用旧迹；是否遗漏预测差修正。 |
| 随机动作导致所有迹都被切断 | 实际动作是否仍为并列最大；不要仅检查探索分支标志。 |
| 长迹在重复特征上造成大步更新 | 积累次数、特征尺度和 α；精确前向等价不是稳定性定理。 |
| 参数不更新时测试通过，训练中却失配 | 在线变参、特征漂移和策略采样变化是否超出了证明条件。 |
| Trace 很长却学不会远处提示 | 状态是否保存提示，以及递归梯度是否被 detach。 |
| 所有比率已截断，线性预测仍发散 | 表格算子结论是否被外推到投影与函数逼近；是否需要梯度目标。 |
| V-trace训练值稳定，却与原目标价值不同 | ρ上限诱导的固定点策略是否已改变。 |
| 期望迹均值变了 | 条件输入是否Markov；旧标签是否仍在同一参数坐标。 |
| 辅助网络学习很好，主网络却实现不同规则 | 是否把∇δ中的下一预测detach；是否遗漏H项、标量迹或更新时序。 |

- 把手算的冻结参数例子改成 λ=0。答案：首步目标变成 0.36，首权重增量为 0.016；第二权重增量仍为 0.06。
- True-online 的 λ=0 为什么退化为普通 TD(0)？答案：迹等于当前特征，两个 V−Vold 修正相消，只剩 αδx。
- 能否只使用荷兰迹而删掉预测差修正？答案：不能；它一般不再实现这里定义的在线前向算法。
- Watkins 的 λ=1 是否必然使用整回合 Monte Carlo return？答案：不是；非贪心动作仍会剪断延续并使用 bootstrap。
- RTRL 敏感度与 TD 资格迹是否都是“历史压缩”，因而可以共用一块变量？答案：它们压缩不同导数对象，通常维度、递推和重置条件都不同。
- π=μ时Tree-backup和Retrace是否完全相同？答案：随机策略下不相同；前者仍乘动作概率，后者的比率为一。
- 截断V-trace的ρ与c是否都只改变方差？答案：不是；ρ上限决定固定点策略，c主要决定后续误差的传播。
- Expected Traces的低方差是否允许忽略agent state？答案：不允许；隐藏历史决定未来转移时，信用均值也可能改变。
- 双网络非线性GTD迹的总和测试通过，是否意味着在线每一步等价？答案：不是；测试冻结参数，原定理也明确这一条件。

一个可开始的研究是固定状态表示与环境，分别增加奖励延迟和重复特征比例。比较传统迹、true-online 与短窗口目标，同时记录预测误差、迹范数、每步计算和内存。第二阶段再引入表示学习，观察原先的精确等价如何失效。两个阶段分开，才能定位变化来自时间目标还是结构梯度。

## 本章的实验设计

先验证固定参数的前后向恒等式、在线 true-online 参考和离策略比率。再检验条件期望迹及非线性梯度迹，最后做学习速度实验。

设定：固定短轨迹核对线性在线等价和离策略传播，再用路径汇合/观测混叠检查 Expected Traces，用双 tanh 网络检查冻结非线性梯度总和。

- λ=0 退化为相应单步更新；真实终端反馈先沿迹传播再清理。
- Retrace 与 V-trace 的传播下标和两类比率分别核对。
- 条件期望迹的均值/方差结论需要状态条件独立；混叠反例应破坏结论。
- GTD2 主、辅助方向分别通过有限差分；TDC/TDRC 对应各自修正。
- 冻结参数总和等价不冒充逐前缀 true-online 等价。

对照：固定参数与在线更新分开；同策略与离策略、Markov 与混叠成对；固定 λ、状态 λ 与截断长度；GTD2/TDC/TDRC 主辅助网络预算匹配

记录：每步迹、比率、误差、参数及网络版本；独立前向参考与有限差分误差；条件均值/方差、信用长度与计算量；匹配资源下的延迟任务学习曲线

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-estimator)

## 学习与研究衔接

先分开前向目标、后向计算与离策略校正，再比较 Expected Traces 和梯度迹的估计对象。RTRL 传播递归敏感度；元学习传播更新规则的敏感度。

[分册导读](https://yingwen.io/zh/continual-rl/start/classic-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-credit) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=credit) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=credit)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

状态是支持后续计算的内部信息；GVF 指定一个预测问题；RTRL 和资格迹规定信用如何传播。三者可以组合，但不是相互替代的算法名称。先理解给定策略的预测，再讨论预测怎样改善控制。

- [Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks](https://yingwen.io/zh/continual-rl/research/#recent-columnar-constructive-networks)
- [Real-Time Recurrent Learning using Trace Units in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-real-time-trace-units)
- [Recurrent Reinforcement Learning with Memoroids](https://yingwen.io/zh/continual-rl/research/#recent-memoroids-sequence-learning)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)
- [Expected Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-expected-eligibility-traces)
- [Safe and Efficient Off-Policy Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-retrace-safe-offpolicy)
- [Convergent Tree Backup and Retrace with Function Approximation](https://yingwen.io/zh/continual-rl/research/#recent-convergent-tree-retrace)
- [Multi-Step Reinforcement Learning: A Unifying Algorithm](https://yingwen.io/zh/continual-rl/research/#recent-q-sigma-backups)
- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://yingwen.io/zh/continual-rl/research/#recent-vtrace-impala)
- [A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning](https://yingwen.io/zh/continual-rl/research/#recent-lambda-greedy)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)
- [Recurrent Reinforcement Learning with Memoroids](https://yingwen.io/zh/continual-rl/research/#recent-memoroids-sequence-learning)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Streaming Deep Reinforcement Learning Finally Works](https://yingwen.io/zh/continual-rl/research/#recent-stream-x)
- [Intentional Updates for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-intentional-updates)
- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)
- [Real-Time Recurrent Learning using Trace Units in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-real-time-trace-units)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Intentional Updates for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-intentional-updates)
- [A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning](https://yingwen.io/zh/continual-rl/research/#recent-lambda-greedy)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [Recurrent Reinforcement Learning with Memoroids](https://yingwen.io/zh/continual-rl/research/#recent-memoroids-sequence-learning)

### Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks

Khurram Javed, Haseeb Shah, Richard S. Sutton, Martha White

JMLR 24 · 2023 · 支持方法与理论

#### 研究问题

如果每次观测只处理一次，如何学习包含历史信息的状态，而不保存一段序列做反向传播？

#### 关键机制

一般递归网络的实时递归学习需要维护庞大的参数—状态敏感度。CCN 限制列之间的递归依赖，并逐步构造新特征，使敏感度可以局部计算。它通过改变网络结构和构造过程降低求导成本，而不是把任意稠密 RNN 的完整导数免费变小。

#### 证据

论文分析受限结构的计算性质，并在动物学习启发的预测问题和 Atari 策略评价中检验预测效率。这里的 Atari 结果主要是预测已有策略的回报，不等于从头训练完整控制智能体。

#### 条件与限制

结构约束、构造顺序和被冻结的旧特征共同限制函数类。监督预测和策略评价上的优势，还需要在会主动改变数据分布的控制闭环中检验。

#### 阅读与实验

先写出递归状态对参数的敏感度递推，再检查哪些跨列项被结构消除。比较 CCN、截断 BPTT 与 RTU 时，同时计入状态、梯度缓存和每步计算。

#### 原文与相关入口

- [JMLR 原文与论文入口](https://www.jmlr.org/papers/v24/23-0367.html)：从网络结构、敏感度传播与预测实验三部分阅读。

### Real-Time Recurrent Learning using Trace Units in Reinforcement Learning

Esraa Elelimy, Adam White, Michael Bowling, Martha White

NeurIPS 2024 · 2024 · 支持方法与理论

#### 研究问题

递归状态既要保存长时信息，又要在在线强化学习中以可控成本更新，怎样设计其递归结构？

#### 关键机制

RTU 使用有结构的递归连接，并维护状态关于参数的在线敏感度。复杂的递归动力学可以用实值运算实现。其关键是让状态更新与梯度迹具有相容的计算结构，减少一般 RTRL 的高阶成本；这与仅给 TD 误差加一条资格迹不同。

#### 证据

论文在部分可观测任务中与常见递归网络比较预测与控制表现。作者代码包含 RTU、其他递归基线、实时 actor–critic 以及部分可观测环境配置。

#### 条件与限制

计算优势依赖特定递归参数化，不能外推为任意记忆问题上的表达能力优势。PPO 版本和严格逐步更新版本的经验协议不同，应分别比较。

#### 阅读与实验

在同一部分可观测任务中固定隐状态维度，再比较完整运行内存和每步更新时间。检查 actor、critic 与递归状态的参数更新是否共享同一条敏感度。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1e616bde0438cb10cb6adf076ae7d336-Abstract-Conference.html)：结构、在线导数与实验协议。
- [作者代码](https://github.com/esraaelelimy/rtus)：从 src/nets、src/agents 和实验配置追踪递归状态到控制更新。

#### 作者代码

[论文作者维护的实现。](https://github.com/esraaelelimy/rtus)

RTU 网络、实时学习器与论文实验配置。

### Deep Reinforcement Learning with Gradient Eligibility Traces

Esraa Elelimy, Brett Daley, Andrew Patterson, Marlos C. Machado, Adam White, Martha White

RLC 2025 / RLJ · 2025 · 支持方法与理论

#### 研究问题

资格迹怎样与明确的梯度目标结合，而不是直接把线性半梯度规则搬到深度网络？

#### 关键机制

论文从广义投影 Bellman 误差出发构造多步目标，推导带资格迹的梯度学习方法。前向视角连接多步回报与经验重放，后向视角通过递推迹分配信用。目标函数、辅助估计器和迹的更新共同决定算法，不只是选择一个较大的 λ。

#### 证据

作者给出多种算法并在 MuJoCo、MinAtar 等任务中比较。代码同时提供相关梯度算法与实验设置，可以把推导中的量映射到实际更新。

#### 条件与限制

线性 GTD 的收敛条件不能自动赋予非线性实现全局收敛保证。重放版本与流式版本的数据使用预算也不能混为一谈。

#### 阅读与实验

从一段短轨迹分别计算前向多步目标和后向迹。随后对照原代码检查辅助网络、目标与主网络参数使用的是更新前还是更新后的值。

#### 原文与相关入口

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_302.pdf)：目标、算法推导与实验。
- [作者算法库](https://github.com/esraaelelimy/gtd_algos)：论文提供的梯度 TD 与资格迹实现。

#### 作者代码

[原论文链接的作者仓库。](https://github.com/esraaelelimy/gtd_algos)

论文梯度算法、资格迹和实验配置。

### Streaming Deep Reinforcement Learning Finally Works

Mohamed Elsayed, Elena Sorina Lupu, Gautham Vasan, A. Rupam Mahmood

arXiv（2024 首稿；2026 v3） · 2026 · 直接研究持续学习

#### 研究问题

不保存经验重放、不使用目标网络或训练批次时，深度 RL 能否逐步稳定学习？

#### 关键机制

Stream-X 把信号归一化、表示初始化、资格迹和受控更新尺度组织为一组流式学习方法。各组件处理的是不同问题：奖励尺度、激活与梯度传播、延迟信用，以及一次更新造成的输出变化。去掉重放并不意味着这些问题会自动消失。

#### 证据

2026 年第三版扩展到 Atari、控制与机器人等实验，并包含持续变化设置。论文和代码经历过版本变化，比较结果时需要同时标明论文版本和算法实现。

#### 条件与限制

广泛任务上的流式可行性不等于所有非平稳问题都已解决。不能把旧版较弱 Adam 基线推广成对所有流式 Adam 方法的否定；后续研究专门检验了这一点。代码许可证也应独立于本教材许可证处理。

#### 阅读与实验

按归一化、资格迹、更新控制分别做消融，并保持每步算力一致。先验证严格一次使用经验，再研究长期变化，而不是仅把小批量大小改成一。

#### 原文与相关入口

- [2026 年第三版论文](https://arxiv.org/abs/2410.14606v3)：作者名单、任务范围与算法版本以该版为准。
- [作者代码版本](https://github.com/mohmdelsayed/streaming-drl/tree/9326fc3e23a401f28087ae2e41b635888740586b)：固定实现版本，避免把不同年份的更新规则混在一起。

#### 作者代码

[原作者仓库的固定版本。](https://github.com/mohmdelsayed/streaming-drl/tree/9326fc3e23a401f28087ae2e41b635888740586b)

Stream-X 算法、变换、优化器及实验；使用前阅读仓库许可证。

### Intentional Updates for Streaming Reinforcement Learning

Arsalan Sharifnassab, Mohamed Elsayed, Kris De Asis, A. Rupam Mahmood, Richard S. Sutton

ICML 2026 · 2026 · 支持方法与理论

#### 研究问题

能否先规定本次更新应产生多大作用，再反推合适的参数更新尺度？

#### 关键机制

Intentional 方法以局部线性近似连接参数变化和预测变化。critic 以减少一定比例的 TD 误差为目标，actor 控制策略输出变化的局部代理量；再结合资格迹和逐坐标尺度，求出这一次更新的强度。这是有目标的局部更新控制，不是对长期表现求导的元梯度。

#### 证据

论文给出推导和流式控制比较，ICML 2026 正式论文入口与作者实现均可用。实现将优化器与 actor–critic 交互区分开，便于检查更新时序。

#### 条件与限制

Taylor 近似在大更新时可能失准。采样动作上的对数概率变化不等于精确的全分布 KL 上界；熵项与 TD 误差符号也必须按原算法处理。

#### 阅读与实验

在一次更新前后直接测量预测变化，并与线性估计比较。分别测试正、负 TD 误差和很小梯度的情形，不要只检查参数是否有限。

#### 原文与相关入口

- [ICML 2026 原文](https://proceedings.mlr.press/v306/sharifnassab26a.html)：正式会议版本与更新意图的定义。
- [作者实现](https://github.com/sharifnassab/Intentional_RL)：重点对照 optimizer.py 与 intentional_ac.py。

#### 作者代码

[原论文作者提供的实现。](https://github.com/sharifnassab/Intentional_RL)

Intentional 更新与流式 actor–critic。

### Recurrent Reinforcement Learning with Memoroids

Steven Morad, Chris Lu, Ryan Kortvelesy, Stephan Liwicki, Jakob Foerster, Amanda Prorok

NeurIPS 2024 · 2024 · 支持方法与理论

#### 研究问题

当记忆网络能够保存信息时，训练序列的切分是否仍会阻止学习器给早期信息分配信用？

#### 关键机制

Memoroids 将一类线性递归模型写成结合运算，利用并行 scan 处理长序列；Tape-Based Batching 将多个完整回合接入同一条 tape，用显式边界处理状态重置，减少分段、补零和截断反传带来的问题。

#### 证据

论文在 POPGym 等部分可观测任务和循环价值学习中比较分段与 tape 训练，并研究观测敏感度、样本效率及运行时间。

#### 条件与限制

并行 scan 和长序列反传使用保存的序列与批处理资源，不属于严格逐步、每条经验只使用一次的 RTRL。结合结构也不使任意非线性 RNN 都能采用同样的 scan。

#### 阅读与实验

固定同一种记忆模型，对照截断长度、完整回合和流式在线导数；分别检查活动能记多久、梯度能传多久、持久内存与训练峰值内存。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://papers.nips.cc/paper_files/paper/2024/file/19f7f755908372efb25826d61959cdf9-Paper-Conference.pdf)：结合运算、inline reset、Tape-Based Batching 与实验。
- [作者公开版本](https://arxiv.org/html/2402.09900v3)：附录给出不同递归模型与回报的 memoroid 写法。

#### 作者代码

[论文附录原链接 memory-monoids 对应作者 Prorok Lab 的现有 memoroids 仓库；README 标明论文。](https://github.com/proroklab/memoroids)

memory 模型、buffer、losses 与 segment_dqn/tape_dqn 对照。

### Expected Eligibility Traces

Hado van Hasselt, Sephora Madjiheurem, Matteo Hessel, David Silver, André Barreto, Diana Borsa

AAAI 2021（2020预印本） · 2021 · 支持方法与理论

#### 研究问题

当前误差能否同时更新本次未走过、但也可能到达当前状态的过去路径？

#### 关键机制

学习给定当前状态的资格迹条件均值，再用当前TD误差更新该均值所指向的过去预测。递归混合在实际轨迹迹与预测的期望迹之间插值；预测对象是过去资格，而非未来奖励。

#### 证据

原文在Markov状态与相应条件下证明更新均值相同、逐分量方差不增，并在路径汇合问题检验预测效率。信用章精确枚举一个正例和一个状态混叠反例。

#### 条件与限制

不完整观察、参数漂移和近似迹预测器会破坏无偏条件。全参数期望迹预测还有输出维度和计算成本；小实验不复现作者的神经实验。

#### 阅读与实验

保持奖励边际分布一致，仅改变奖励是否依赖隐藏的过去路径。先测信用均值与方差，再研究agent state能否恢复条件独立。

#### 原文与相关入口

- [作者原文](https://arxiv.org/html/2007.01839)：Lemma 1、Proposition 1及ET(λ,η)递归混合。
- [AAAI发表版本](https://ojs.aaai.org/index.php/AAAI/article/view/17200)：正式会议年份为2021。

### Safe and Efficient Off-Policy Reinforcement Learning

Rémi Munos, Tom Stepleton, Anna Harutyunyan, Marc G. Bellemare

NeurIPS 2016 · 2016 · 支持方法与理论

#### 研究问题

目标与行为策略不一致时，如何保留多步信用而避免重要性比率乘积爆炸？

#### 关键机制

统一多步目标为目标策略TD误差的加权和，Retrace采用λmin(1,π/μ)传播系数。近同策略时保留长迹，目标概率较低的动作则减少传播；一步误差仍使用目标动作期望。

#### 证据

论文分析表格算子的收缩性质，给出条件下的评价与控制收敛，并报告Atari实验。信用章独立检查传播系数和有限轨迹恒等式。

#### 条件与限制

表格安全性不是任意线性或神经逼近的稳定性保证。行为覆盖、变化策略与投影条件仍需检查；代码小实验不复现Atari。

#### 阅读与实验

在同样轨迹与表示上，分别改变策略差异和动作随机性，比较Tree-backup与Retrace的信用长度、方差和预测误差。

#### 原文与相关入口

- [原论文](https://arxiv.org/html/1606.02647)：统一算子、传播系数及理论条件。

### Convergent Tree Backup and Retrace with Function Approximation

Ahmed Touati, Pierre-Luc Bacon, Doina Precup, Pascal Vincent

ICML 2018 · 2018 · 支持方法与理论

#### 研究问题

传播系数已经截断，为什么函数逼近下的Tree-backup和Retrace仍可能发散？

#### 关键机制

分析函数逼近与off-policy多步bootstrap的学习算子，展示线性反例，再把相应目标写成二次凸凹鞍点问题，构造梯度版本。

#### 证据

原文给出线性不稳定例子、梯度方法收敛保证与有限样本界。它直接限定了从Retrace表格结论外推到逼近算法的范围。

#### 条件与限制

凸凹线性问题的保证不能自动覆盖学习表示的深度网络。稳定目标、更新速度与控制性能还需分别验证。

#### 阅读与实验

先检查固定表示下的期望更新矩阵，再将半梯度和梯度版本按相同样本、步数与计算预算比较。

#### 原文与相关入口

- [ICML原文](https://proceedings.mlr.press/v80/touati18a.html)：理论反例、鞍点方法和保证条件。

### Multi-Step Reinforcement Learning: A Unifying Algorithm

Kristopher De Asis, J. Fernando Hernandez-Garcia, G. Zacharias Holland, Richard S. Sutton

AAAI 2018 · 2018 · 支持方法与理论

#### 研究问题

多步动作价值目标必须始终采样下一动作，或始终对动作取期望吗？

#### 关键机制

Q(σ)逐处混合Sarsa的采样动作与Expected Sarsa的动作期望，并同步改变后续误差传播。σ控制采样程度，与控制回报长度的λ不同。

#### 证据

原文给出统一n-step表达、off-policy修正和实验比较。信用章小程序核验冻结on-policy几何λ混合的两个端点。

#### 条件与限制

原文n-step和本章λ混合参考具有不同实现范围。只改一步误差却不改多步传播或策略修正，不能称为完整Q(σ)。

#### 阅读与实验

把采样噪声、目标长度和策略差异分开改变，避免把σ与λ的作用归到同一“更长信用”解释。

#### 原文与相关入口

- [原文](https://arxiv.org/html/1703.01327)：式13–15：混合误差、传播及off-policy修正。

### IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures

Lasse Espeholt, Hubert Soyer, Rémi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, Koray Kavukcuoglu

ICML 2018 · 2018 · 支持方法与理论

#### 研究问题

actor采样策略落后于learner时，如何校正状态价值与策略更新？

#### 关键机制

V-trace用截断ρ校正当前TD误差，用独立截断c控制后续误差传播，再用下一状态V-trace目标构造actor优势。ρ上限还决定表格固定点对应的截断策略。

#### 证据

原文分析固定点并检验分布式多任务训练。固定版本作者代码明确区分clipped_rhos、cs、反向scan与pg_advantages。

#### 条件与限制

IMPALA保存短轨迹并批量训练，不属于严格单样本流式协议。截断后价值可能对应不同于原目标的策略；信用章bandit示例显示0.8变为0.5。

#### 阅读与实验

独立改变策略滞后、ρ上限和c上限。记录目标策略变化与传播长度，不把两种截断都只解释为方差控制。

#### 原文与相关入口

- [ICML原文](https://proceedings.mlr.press/v80/espeholt18a.html)：V-trace固定点与分布式实验。
- [作者固定实现](https://github.com/google-deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/vtrace.py)：from_importance_weights与下一状态actor目标。

#### 作者代码

[原作者团队仓库的固定版本。](https://github.com/google-deepmind/scalable_agent/tree/6c0c8a701990fab9053fb338ede9c915c18fa2b1)

IMPALA原始TensorFlow实现与V-trace；运行需要原项目环境。

### A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning

Martha White, Adam White

arXiv预印本 · 2016 · 支持方法与理论

#### 研究问题

不同状态的预测可靠性不同，固定λ是否浪费了多步信用？

#### 关键机制

将下一处bootstrap选择写成局部偏差平方与回报方差的折中，得到$λ=b^2/(b^2+\operatorname{Var}(G))$。完整λ-greedy还用在线预测器估计回报均值和二阶矩。

#### 证据

原文给出状态相关λ的目标、增量算法和多个预测设置的实验。信用章仅核对已知统计量下的局部最优与变量λ恒等式。

#### 条件与限制

局部贪心目标不是整条轨迹的联合最优。逼近误差、统计滞后和非平稳性会影响λ估计；辅助资源需要计入比较。

#### 阅读与实验

先让噪声方差变化，再让bootstrap可靠性变化。比较固定λ、已知统计参照和在线估计，分别观察目标偏差与适应速度。

#### 原文与相关入口

- [作者原文](https://arxiv.org/html/1607.00446)：局部目标、状态λ、均值／二阶矩预测与完整算法。

### Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning

Noah Farr, Aryaman Reddi, Carlo D’Eramo, Jan Peters

arXiv预印本（2026-07-07 v2） · 2026 · 支持方法与理论

#### 研究问题

严格逐步更新的智能体怎样同时学习递归记忆、分配延迟信用并控制计算？

#### 关键机制

将RTU结构的RTRL敏感度接入QRC与流式actor–critic。敏感度给出当前输出对记忆参数的导数，资格迹再组合过去输出的回报信用；两条递推保持分工。

#### 证据

v2在MemoryChain、五项POPGym和masked MuJoCo上报告5-seed结果，另用KMemoryChain比较在线敏感度与当前参数重算参考，并检验Taylor修正。

#### 条件与限制

masked MuJoCo仍落后批量PPO。固定参数精确RTRL不代表在线变参敏感度始终等于当前参数重算；诊断保存整个episode，须计为额外评价资源。尚未确认作者公开代码。

#### 阅读与实验

在相同递归容量下独立改变记忆跨度、回报λ与参数步幅，同时测敏感度误差和回报。诊断改善不能单独当作控制改进证据。

#### 原文与相关入口

- [2026年v2原文](https://arxiv.org/html/2605.24709v2)：方法、5-seed实验、masked MuJoCo负边界与staleness诊断。


<a id="chapter-code"></a>

## 下载与运行

标准库：31项检查覆盖冻结／在线等价、策略校正、Q(σ)端点、V-trace目标变化、状态λ、期望迹混叠反例、非线性梯度迹有限差分及actor／递归敏感度；不是完整论文复现。

[下载 credit_assignment_lab.py](https://yingwen.io/zh/continual-rl/download/credit_assignment_lab.py)

```sh
python3 credit_assignment_lab.py all
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton · Learning to Predict by the Methods of Temporal Differences · 1988](https://jmvidal.cse.sc.edu/library/sutton88a.pdf)：TD 与资格迹的原始研究，使用预测变化构造增量学习信号。

- [Sutton & Barto · Reinforcement Learning, Chapter 12](https://incompleteideas.net/book/the-book-2nd.html)：前向／后向视角、控制资格迹与 true-online 的教材讨论。

- [van Seijen & Sutton · True Online TD(λ) · ICML 2014](https://proceedings.mlr.press/v32/seijen14.html)：定义逐前缀在线前向参考，并推导精确等价的后向算法。

- [van Seijen 等 · True Online Temporal-Difference Learning · JMLR 2016](https://www.jmlr.org/papers/v17/15-599.html)：Algorithm 2 的迹外步长形式、线性等价分析，以及 TD／Sarsa 实验。

- [作者随机 MDP 实验 · totd-rndmdp-experiments](https://github.com/armahmood/totd-rndmdp-experiments)：论文作者 Rupam Mahmood 提供的 accumulating、replacing、true-online 对照与随机 MDP 实验。

- [作者 true-online 更新 · 固定版本 totd.py](https://github.com/armahmood/totd-rndmdp-experiments/blob/4b6d459890a0e9da898f6cf8b849ba7870ea044a/pysrc/algorithms/tdprediction/onpolicy/totd.py)：step 将 α 折入迹；predprev 与预测差修正对应原文更新。

- [Watkins · Learning from Delayed Rewards · 1989](https://www.repository.cam.ac.uk/items/f1ce2aa9-2e01-4a08-92b7-1eec76b652ed)：Q-learning 与延迟奖励控制的原始博士论文。

- [Sutton 等 · Policy Gradient Methods with Function Approximation · NIPS 1999](https://papers.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)：策略梯度、价值辅助与 actor–critic 的理论基础；需区分所选目标的状态权重。

- [Williams & Zipser · A Learning Algorithm for Continually Running Fully Recurrent Neural Networks · 1989](https://doi.org/10.1162/neco.1989.1.2.270)：递归敏感度的原始方法；用于理解网络内的历史梯度路径，而非替代回报资格迹。

- [Precup, Sutton & Singh · Eligibility Traces for Off-Policy Policy Evaluation · ICML 2000](https://web.eecs.umich.edu/~baveja/Papers/OffPolicy.pdf)：作者站点原文；重要性采样、Tree-backup与目标／行为策略的分工。

- [Munos等 · Safe and Efficient Off-Policy Reinforcement Learning · NeurIPS 2016](https://arxiv.org/html/1606.02647)：统一误差传播系数；Retrace算子收缩及表格评价／控制的条件。

- [Touati等 · Convergent Tree Backup and Retrace with Function Approximation · ICML 2018](https://proceedings.mlr.press/v80/touati18a.html)：线性函数逼近中的不稳定反例，以及鞍点形式的梯度方法。

- [De Asis等 · Multi-Step Reinforcement Learning: A Unifying Algorithm · AAAI 2018](https://arxiv.org/html/1703.01327)：Q(σ)采样／期望混合；正文数值代码仅实现冻结on-policy几何λ混合。

- [Espeholt等 · IMPALA · ICML 2018](https://proceedings.mlr.press/v80/espeholt18a.html)：V-trace两类截断、固定点策略与actor目标；区分轨迹窗口和流式资格迹。

- [IMPALA作者V-trace实现 · 固定版本](https://github.com/google-deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/vtrace.py)：from_importance_weights：clipped_rhos、cs、反向scan、下一目标的actor优势和stop_gradient。

- [White & White · A Greedy Approach to Adapting the Trace Parameter · 2016](https://arxiv.org/html/1607.00446)：状态相关λ；局部bias–variance目标；回报均值和二阶矩的在线估计。

- [Xu, van Hasselt & Silver · Meta-Gradient Reinforcement Learning · NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/2715518c875999308842e3455eda2fe3-Abstract.html)：通过学习更新和后续评价适应回报定义；区别于同样本直接选择容易拟合的目标。

- [van Hasselt等 · Expected Eligibility Traces · AAAI 2021](https://arxiv.org/html/2007.01839)：条件期望迹、Markov均值／逐分量方差结论、递归混合与特征混叠边界；2020年预印本。

- [Elelimy等 · Deep Reinforcement Learning with Gradient Eligibility Traces · RLC 2025](https://arxiv.org/html/2507.09087v1)：广义投影目标、GTD2／TDC／TDRC三条迹；Theorem 6.1冻结双网络参数，Table 2与QRC公式中的标量H迹相对应。

- [GTD资格迹作者QRC代码 · 固定版本](https://github.com/esraaelelimy/gtd_algos/blob/76293dea9b2129d55e08bfb4178618a0a26c2dd8/gtd_algos/src/algorithms/qrc.py)：update_step先共同计算主／辅助方向，再更新参数；检查完整∇δ、三条迹、正则与更新后的清迹。

- [Schulman等 · High-Dimensional Continuous Control Using Generalized Advantage Estimation · ICLR 2016](https://arxiv.org/html/1506.02438)：TD误差和的优势解释、bias–variance及折扣目标约定；GAE不等于完整actor迹学习协议。

- [Elelimy等 · Real-Time Recurrent Learning using Trace Units in Reinforcement Learning · NeurIPS 2024](https://arxiv.org/html/2409.01449)：特定递归结构的在线敏感度、固定参数精确导数与变参时过时问题。

- [RTU作者递归实现](https://github.com/esraaelelimy/rtus/blob/main/src/nets/rtus/non_linear_rtus.py)：FwdRealTimeNonLinearRTUs推进活动与grad_memory；自定义求导使用已存敏感度。

- [Farr等 · Streaming RL under Partial Observability with RTRL · 2026 v2](https://arxiv.org/html/2605.24709v2)：2026-07-07预印本；RTU与QRC／流式AC组合、5-seed实验、masked MuJoCo负边界及需保存轨迹的敏感度诊断；未确认作者代码。


---

# 流式强化学习：交互协议与更新稳定性

流式学习首先约束数据和计算：新经验到来后及时更新，不重放历史转移，并限制每步时间与持久内存。在这些约束下，本章研究怎样组织资格迹、尺度统计和更新规则，使逐步学习可实现、可检查。

## 本章内容

- 明确流式数据权限、单步计算与持久内存预算，区分流式协议、持续任务和不可重置生命期。
- 在流式协议中实现 TD(λ)，检查痕迹与终止时序；将时间信用分配与更新稳定性分开分析。
- 推导并比较 ObGD、StreamingOptimizer 与 Intentional Updates 的尺度机制、保证范围和失败条件。

<a id="problem-definition"></a>

## 本章的问题定义

每次真实经验到来后及时学习，不重放历史转移，持久内存和每步计算有明确预算；该协议独立于是否允许环境重置。

### 给定条件与符号

- 基础预测或控制目标、新经验因果流、持久字节预算和单步时间预算。
- 网络、资格迹、尺度统计与更新控制器；真实终止清理规则。

### 需要求解的对象

满足协议且可持续执行的学习更新；更新尺度局部可检查，长期预测或控制质量另行评价。

### 信息与数据权限

允许保存参数、优化器状态、资格迹和在线统计，不保存供再次训练的历史转移。当前奖励统计、目标计算与更新顺序都需明确。

$$
M_t=U(M_{t-1},\xi_t),\qquad \operatorname{bytes}(M_t)\le B,\qquad C_t\le C_{\max}
$$

$\xi_t$ 是刚收到的经验，$M_t$ 是全部持久学习状态，$U$ 是更新映射，$B$ 为内存预算；$C_t$ 是包括动作选择、学习与规划的本步总计算，$C_{\max}$ 为上限。历史转移不得被重取训练。基础任务目标沿用预测或控制规格，这些资源约束不是新的奖励函数。

### 成立条件与解的含义

- 预算包含网络、优化器、迹和统计；batch size、无GPU或常数缓存大小不单独证明协议。
- 局部输出线性化、范数代理和逐坐标位移界的前提分别声明；终止不自动清除长期尺度统计。

判断准则：日志证明每条转移使用权限与延迟，持久内存不随寿命增长；手算更新/尺度边界正确；长期实验报告非有限更新、恢复及外部收益，参数位移界不替代收益。

### 适用边界

- 小参数更新不保证TD误差下降、策略改善或单生命期安全。
- 不把观测归一化和自适应统计当作无成本且不影响函数的预处理。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 改变信息或数据协议 · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)：流式预算允许迹这种统计机制，却限制长窗口、重放和完整反传。

- 组合不同学习问题 · [元学习与学习规则的适应](https://yingwen.io/zh/continual-rl/algorithms/meta/)：预设尺度控制可逐步执行；依据后续表现学习规则还需额外敏感度和资源。

- 改变信息或数据协议 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：完整控制比较需加入流式资源约束；无重置与无重放是两项独立权限。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

没有回放可反复纠正当前过冲；奖励、特征和迹尺度变化会放大单步更新。

### 本章的核心思路

用可递推统计保留信用并按声明的局部尺度限制更新，分别检验协议和任务表现。

1. [以迹替代保存训练历史](#lesson-derive)：因为预算禁止重取转移，资格迹保存过去方向统计并在新误差到来时更新。

2. [定位尺度与过冲](#lesson-scale)：因为同一名义步长随特征尺度产生不同输出变化，先在线性例上推有效步长，再辨认ObGD的范数代理条件。

3. [选择可检查的尺度控制器](#lesson-current-streamx)：因为整体代理与逐坐标边界处理的量不同，对照2026控制器的参数位移界与Intentional Updates的输出线性化。

4. [按输出变化解释学习尺度](#lesson-output-steps)：因为参数步长未直接指定预测改变，Intentional方法利用梯度内积决定局部尺度，非线性与统计近似仍需测量。

结论与条件：有限增量和正稳定项下可检验逐坐标位移界；线性固定标签的输出关系只约束本次更新。Stream-X/Intentional深度组合不提供无条件长期收敛。

### 相关方法改变了什么

- ObGD：以TD误差与迹范数做廉价整体缩放，依赖代理尺度解释。

- 逐坐标StreamingOptimizer：用衰减最大增量提供坐标位移边界，不保证输出收缩。

- Intentional Updates：按局部输出变化选尺度，含预条件、迹和在线平均近似。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 价值函数

$v(s)$ 预测从状态 $s$ 出发按固定策略累积的折扣奖励。学习器根据不断到来的经验修正这个预测。

### 半梯度 TD

误差 $\delta=r+\gamma v(s')-v(s)$ 中的两个预测使用同一份旧参数；更新方向只对当前预测求导。

### 资格迹

与参数同维的向量，积累过去预测对参数的敏感性。它保存信用分配所需的统计量，而不保存原始样本。

<a id="lesson-setting"></a>

## 1 · 流式学习的数据与计算约束

机器人每 20 毫秒拿到一条观测并必须作出下一步决定。若学习需要等一批数据、回放百万帧或者反复重算整段历史，就不符合这个实时任务的预算。这里采用严格的一步流式协议：每条新 transition 到来后立即更新，不保存原始 transition 供以后重复采样；允许维护固定大小的参数、痕迹、运行统计与 agent state。

这并不等于“算法没有记忆”：资格迹、RNN 隐状态与优化器二阶矩都占内存，必须记账。也不等于“环境永不变化”或“环境没有终止”。本章先研究固定策略预测，随后区分 Sarsa / Q / actor–critic。网络是否深、目标是否平均奖励、世界是否单生命期，是另外三个独立选择。

本章讨论的是学习协议及其数值实现。时间信用分配决定当前反馈影响哪些过去预测或决策；学习规则的适应则用经验改进步长、目标或更新方式。资格迹是可在流式预算内使用的一种信用机制，在线元梯度也可能采用流式实现，但两者都不构成 streaming 的定义。

| 约束或对象 | 保存内容或度量 | 边界 |
| --- | --- | --- |
| 严格 streaming 数据 | 当前样本；固定维参数和统计 | 不重放旧转移，不等待未来样本组成训练 batch |
| 资格迹 | 每个参数一个历史敏感性统计 | 不能无损恢复历史样本 |
| 实时计算预算 | 一次或固定少数次前后向 | 限制每步计算与历史回溯 |
| 学习评价 | 实际每步延迟、内存、累计奖励 | GPU 吞吐量不能单独代表实时控制能力 |

把 DQN 的 replay capacity 改成 1 是可用的诊断基线，却未必保留 DQN 的稳定性。原方法依靠的样本混合、target network 和批平均同时改变了。研究问题应是“哪些机制能在这个协议下工作”，而不是只换一个 buffer 参数。

<a id="lesson-derive"></a>

## 2 · 流式协议中的资格迹

时间信用分配研究延迟误差应当影响哪些过去预测、动作或参数。它既可以使用在线资格迹，也可以借助保存的轨迹与反向传播，因此不等于流式学习。本节只保留实现流式 TD 所需的前向—后向关系与边界；本书《时间信用分配》章独立讨论更完整的算法条线。

$$
G_t^{(n)}=\sum_{k=1}^{n}\gamma^{k-1}R_{t+k}+\gamma^n v(S_{t+n})
$$

n-step target 比一步 TD 看得更远，却要等 n 步才能构造。有限回合末尾的 bootstrap 为零。

先固定价值参数。将 $\delta_t,\gamma\delta_{t+1},\ldots$ 相加时，相邻两项中的状态价值相消，只留下首项 $-v(S_t)$ 和末尾的 bootstrap。这一望远镜求和说明了多步回报与 TD 误差序列的关系。

$$
G_t^{(n)}-v(S_t)=\sum_{k=0}^{n-1}\gamma^k\delta_{t+k}
$$

等式要求右侧所有 δ 使用与左侧同一套固定价值。在线每步变参数时，普通 accumulating trace 与这个冻结参数前向视角只有近似对应；不能把有限步长下的等价说成精确。

$$
\begin{aligned}G_t^\lambda&=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)},\\G_t^\lambda-v(S_t)&=\sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k}.\end{aligned}
$$

无限混合式先取 $0\le\lambda<1$，并要求回报和右侧级数收敛；$\lambda=1$ 用相应极限或完整回报定义，不直接代入第一行的零乘无限和。有限回合让最后一个 Monte Carlo return 承接剩余权重。$\lambda=0$ 退化为一步 TD，接近 1 则使较远的 TD error 参与信用分配。

前向视角考虑未来哪些 $\delta$ 影响当前预测；后向视角在误差到来时，给过去相关的预测分配信用。交换求和顺序，就需要保存过去梯度按 $\gamma\lambda$ 衰减后的总和。对于线性 $v=w^\top x$，梯度就是特征 $x$；对于网络则使用各时刻参数下的输出梯度。

$$
\begin{aligned}z_t&=\gamma_t\lambda z_{t-1}+\nabla_w v(S_t;w_t),\\\delta_t&=R_{t+1}+\gamma_{t+1}v(S_{t+1};w_t)-v(S_t;w_t),\\w_{t+1}&=w_t+\alpha_t\delta_t z_t.\end{aligned}
$$

γt 属于到达 St 的转移，用来衰减旧痕迹；γt+1 属于当前刚完成的转移，用来构造 target。常数 γ 时区别被隐藏，终止和 GVF 的状态依赖 discount 会把错误暴露出来。

若 $S_{t+1}$ 真正终止，$\gamma_{t+1}=0$，不再 bootstrap；但本次奖励仍通过 $z_t$ 分给本回合的过去状态。痕迹在完成终点更新后、新回合开始前清空。若只是时间限制截断而真实过程仍会继续，价值目标保留 bootstrap；若随后环境另行重置，资格迹则不能跨越这次非连续转移。

**算法：线性 accumulating TD(λ)；终点奖励在清迹之前更新**

1. 初始化参数 $w$、$z=0$，选择 $\lambda$ 与 $\alpha$
1. 每步接收 $(x,r,x')$ 和 $\gamma_{\rm current},\gamma_{\rm next}$：
  1. 保存旧参数 $w_{\rm old}$
  1. 计算 $v=x^\top w_{\rm old}$、$v'=x'^\top w_{\rm old}$
  1. 计算 $\delta=r+\gamma_{\rm next}v'-v$
  1. 更新 $z\leftarrow\gamma_{\rm current}\lambda z+x$
  1. 更新 $w\leftarrow w_{\rm old}+\alpha\delta z$
1. 新回合开始时令 $z=0$

| 方法 | 保存的痕迹 | 关键限制 |
| --- | --- | --- |
| TD(λ) | 过去 ∇v 的衰减和 | 普通在线 accumulating 版本不是任意步长下精确前向等价 |
| Momentum SGD | 过去 loss gradient 的衰减和 | 每步历史误差已乘在各自梯度上；不同于当前 δ 乘全部痕迹 |
| True online TD(λ) | Dutch trace 和预测差修正 | 对指定线性在线 λ-return 有精确等价；不能直接宣称任意深网同样成立 |

<a id="lesson-scale"></a>

## 3 · 数值尺度与更新过冲

考虑一维固定标签回归：预测为 $wx$，误差为 $e=y-wx$，更新为 $w_+=w+\alpha ex$。代入后得到 $e_+=e(1-\alpha x^2)$。即使 $\alpha=0.1$，只要 $x=10$，有效步长也达到 10，误差变号且放大 9 倍。步长的作用取决于特征尺度。

$$
\xi=\frac{e-e_+}{e}=\alpha x^2,\qquad 0<\xi\leq1\ \Longrightarrow\ \text{本次固定标签误差不越过零点}
$$

这是当前样本的代数性质，不等于新样本泛化、长期控制稳定性或单生命期安全。e=0 时用更新为零处理，不做 0/0。

TD 更新会同时改变当前预测和下一状态 bootstrap。线性情况下直接展开，得到 $\xi=\alpha z^\top(x-\gamma x')$，而非 $\alpha x^\top x$。它可以为负，即同一样本的 TD 误差变大；这体现半梯度 TD 与固定标签回归的区别。

$$
\delta_+=\delta-\alpha\delta\,z^\top(x-\gamma x'),\qquad \xi=\alpha z^\top(x-\gamma x')
$$

式中 z 在这一步更新中视为固定。深网使用局部线性化后只能得到近似；若再用更便宜的范数代理，还需要额外梯度尺度假设。

$$
\begin{aligned}M&=\alpha\kappa\max(|\delta|,1)\|z\|_1,\\\alpha_{\rm used}&=\frac{\alpha}{\max(1,M)},\\w_+&=w+\alpha_{\rm used}\delta z.\end{aligned}
$$

这是 ObGD 的代数更新形式，κ 是安全缩放系数。它用廉价统计抑制大更新；对任意网络、任意输入并不存在无条件不越界保证。零痕迹时分母至少 1，不会除零。

一个尺度反例是 $y=1$、$x=10$、$w=0$、$\alpha=0.1$、$\kappa=2$。此时 $z=10$、$M=2$，所以 $\alpha_{\rm used}=0.05$，更新后 $w_+=0.5$，预测为 5，仍然超过目标 1。这里梯度尺度不满足范数代理所需条件，说明该更新的实际性质依赖输入与网络尺度。

在线输入统计采用 Welford 递推：先用旧均值计算差，再更新新均值，最后用新旧两侧差的乘积更新 M2。所有统计只能使用已经到来的观测。全数据预先归一化会泄露未来；累积统计在漂移中会越来越迟钝，指数滑动统计则是另一种估计器，需要独立比较。

$$
\begin{aligned}n&\leftarrow n+1,\quad d=x-\mu_{\rm old},\\\mu&\leftarrow\mu_{\rm old}+d/n,\\M_2&\leftarrow M_2+d(x-\mu),\quad \sigma^2=M_2/(n-1).\end{aligned}
$$

n=1 时必须定义方差初始化和 ε，避免第一步除零。归一化统计的改变也在改变价值函数输入坐标，因此不是与学习完全无关的预处理。

<a id="lesson-streamx"></a>

## 4 · Stream-X：流式深度控制的组件组合

Stream-X 是将同一组稳定化机制应用到不同基础学习器的方法。2024 版使用 ObGD、资格迹、在线观测与奖励尺度统计、无可学习仿射参数的 LayerNorm，以及稀疏初始化。2026 修订版保留这一组织方式，但更换了更新控制器并修改初始化；两版的优化器需要分别理解。

$$
\bar o_{t,i}=\frac{o_{t,i}-\mu_{t,i}}{\sqrt{\sigma_{t,i}^2+\varepsilon}},\qquad U_t=\gamma U_{t-1}+r_t,\qquad \bar r_t=\frac{r_t}{\sqrt{\operatorname{Var}_t(U)+\varepsilon}}
$$

观测按各坐标中心化并缩放；奖励只缩放，不减均值。奖励迹使用已到来的奖励，是未来 return 尺度的因果代理。2026 实现先更新统计并缩放当前奖励，再在 episode 边界清零奖励迹；累计均值和方差保留。

LayerNorm 在单个观测的一层预激活上计算均值和方差，因而不需要 batch。稀疏初始化则改变参数更新对不同输入的共享程度。二者不相同：前者控制层内尺度，后者改变初始连接。2024 的逐层固定稀疏率在低输入维度下可能将一层全部置零；2026 的 layer-aware 初始化分配全网非零连接预算，复现时应连同 sparse_init.py 一起使用。

| 分支 | 当前误差或方向 | 必须保留的时序 |
| --- | --- | --- |
| Stream TD(λ) | 固定策略 δ=r+γv′−v | 价值输出梯度累积后再 ObGD |
| Stream Sarsa(λ) | δ=r+γq(s′,a′)−q(s,a) | 下一动作来自实际目标策略 |
| Stream Q(λ) | max 下一动作目标；Watkins 痕迹语义 | 执行非贪心动作时截断此前不再匹配贪心延续的痕迹 |
| Stream AC(λ) | critic δ 配合 ∇logπ 的 actor trace | 行为动作按更新前策略采样；actor/critic 采用同一旧 δ，熵项另核对 |

Stream-X 的控制实验允许环境终止与重置；2026 修订还研究了变化摩擦条件、机器人与资源受限设备。其结果需要按照各自的数据与环境协议解释。一次交互更新、没有经验重放，并不意味着所有实验都采用不可重置的单生命期。

作者 2024 实现先对 $-v$ 和负 log-probability 求梯度，再由 optim.py 做减法；本文采用正输出梯度加更新，两种符号约定一致。应避免先对平方 TD 损失求导，再把 $\delta$ 额外乘入优化器。原 actor 的熵项还含 $\operatorname{sign}(\delta)$，经本步乘误差后，其即时熵方向与 $|\delta|$ 成比例；这与直接加一个固定权重熵梯度不同。

<a id="lesson-current-streamx"></a>

## 5 · 2026 更新控制器：从整体缩放到逐坐标边界

ObGD 用全网 $\|z\|_1$ 形成一个共同步长。若许多坐标同时具有非零痕迹，网络越大，每个坐标获得的更新可能越小。修订版 StreamingOptimizer 转而对完整增量 $u_t=\delta_t z_t$ 的每个坐标维护衰减最大值：

$$
v_{t,i}=\max\{\beta v_{t-1,i},|u_{t,i}|\},\qquad w_{t+1,i}=w_{t,i}+\alpha\frac{u_{t,i}}{v_{t,i}+\varepsilon}
$$

初始化最大值为零，保留比例满足 $0\leq\beta<1$。这里的 $v_{t,i}$ 是优化器统计，不是状态价值；代码中名为 max_v。更新后若 episode 结束，只清资格迹，保留这个尺度统计。

$$
|\Delta w_{t,i}|=\alpha\frac{|u_{t,i}|}{v_{t,i}+\varepsilon}\leq\alpha\frac{|u_{t,i}|}{|u_{t,i}|+\varepsilon}<\alpha
$$

有限增量、$\alpha>0$ 与 $\varepsilon>0$ 下，单坐标位移严格小于名义步长。这个结论不需要前一节的 TD 梯度尺度假设，但它只约束参数位移，并不保证 TD 误差减小、策略改善或无长期发散。

**算法：StreamingOptimizer 的 TD 形式，对应作者 2026 optimizer.py**

1. 初始化资格迹 $z=0$ 和逐坐标最大值 $v=0$
1. 每条转移：
  1. 用旧参数计算 $\delta$ 和输出梯度 $g$
  1. 更新 $z\leftarrow\gamma\lambda z+g$
  1. 形成完整增量 $u\leftarrow\delta z$
  1. 更新 $v\leftarrow\max(\beta v,|u|)$（逐坐标）
  1. 更新 $w\leftarrow w+\alpha u/(v+\varepsilon)$
  1. 若将开始新 episode，清零 $z$，保留 $v$

手算一个尖峰：上一最大值为 $2$，$\beta=0.9$，新增量为 $100$，则新最大值立即变成 $100$，位移约为 $\alpha$。下一步增量降为 $1$ 时，最大值仍为 $90$，位移约为 $\alpha/90$。所以它立即限制尖峰，而后逐步恢复更新尺度。

2026 的 Stream-RAC 还采用重参数化 actor：通过动作对 soft Q 目标求导，而不是把 TD 误差乘 log-policy 梯度。作者实现每步清 actor trace，critic 则使用 soft next-action target 与资格迹。它仍没有 replay 和 target network，但不能当作 Stream-AC 只换一个函数名。

<a id="lesson-output-steps"></a>

## 6 · Intentional Updates：按输出变化选择步长

Intentional-AC 属于 ICML 2026 的 Intentional Updates for Streaming Reinforcement Learning。其起点是输出变化：已选定参数方向 $d$，希望标量输出 $y(w)$ 改变 $\Delta$。一阶展开给出 $y(w+\alpha d)-y(w)\approx\alpha\nabla y^\top d$，因此可以反解步长。这里不追踪参数对过去步长的敏感度，也不建立元参数的资格迹；“步长依数据变化”与“元梯度学习步长”是不同机制。

$$
\alpha\approx\frac{\Delta}{\nabla y(w)^\top d},\qquad d=\delta\nabla V_w(s),\qquad\Delta=\eta\delta\quad\Longrightarrow\quad\alpha=\frac{\eta}{\|\nabla V_w(s)\|^2}
$$

令 $\delta=r+\gamma V_w(s')-V_w(s)$，这里目标 $r+\gamma V_w(s')$ 在本次更新中固定。在线性 TD(0)、梯度非零且不加裁剪时，当前预测向这个固定目标移动恰好 $\eta\delta$。对非线性网络，它只是局部近似；对重新计算过 bootstrap 的新 TD 误差，也没有相同收缩等式。

例如 $w=0.5$、当前特征 $x=2$、下一特征 $x'=1$、奖励 $r=1.5$、$\gamma=0.9$、$\eta=0.4$。旧目标为 $1.95$，误差为 $0.95$；步长为 $0.4/4=0.1$，更新后 $w^+=0.69$。到旧目标的误差是 $1.95-1.38=0.57=0.6\times0.95$，但重新计算 bootstrap 得到 $1.5+0.9\times0.69-1.38=0.741$。区分两个误差是理解其保证范围的关键。

资格迹使更新方向包含过去梯度，逐坐标缩放又改变了几何。记当前输出梯度为 $g_t$，逐坐标缩放矩阵为 $D_t=\operatorname{diag}(\rho_t)$，资格迹为 $z_t=qz_{t-1}+g_t$，且 $q=\gamma\lambda<1$。对过去预测变化的加权平方和应用 Cauchy–Schwarz，可得到涉及 $z_t^\top D_tz_t$ 与历史梯度范数的尺度。作者实现用偏差校正的指数平均跟踪后者，而不是储存所有过去梯度。

$$
\begin{aligned}\nu_t&=\beta_2\nu_{t-1}+(1-\beta_2)g_t^2,\qquad \rho_t=\left(\sqrt{\nu_t/(1-\beta_2^t)}+\varepsilon\right)^{-1},\\\sigma_t&=g_t^\top D_tg_t,\qquad s_t=qs_{t-1}+(1-q)\sigma_t,\qquad\bar\sigma_t=\frac{s_t}{1-q^t},\\\alpha_t&=\frac{\eta}{\max\{\sqrt{\bar\sigma_t\,z_t^\top D_tz_t},\varepsilon\}},\qquad w^+=w+\alpha_t\widetilde\delta_tD_tz_t.\end{aligned}
$$

平方、开方和倒数均逐坐标进行。$\bar\sigma$ 是平均而非未归一化的历史和，不能将推导中的历史求和与实现直接等同；$D_t$ 还随时间改变，因此历史统计量也是在线代理。分母只确定局部更新尺度，不给任意长期训练轨迹提供收敛保证。

完整实现先用原始 $\delta^2$ 的偏差校正指数平均给出裁剪幅度，得到 $\widetilde\delta$。critic 使用它更新价值；actor 再除以裁剪后误差绝对值的运行平均，以限制奖励尺度的影响。对于无迹 actor，$g=\nabla\log\pi(a\mid s)$ 时，目标是让已采样动作的对数概率变化约为 $\eta\widetilde A/\overline{|A|}$。这既不是精确的全分布 KL trust region，也不保证每个动作概率都只改变固定百分比；动作相关步长还会重加权期望策略梯度。

$$
\begin{aligned}c_t&=\beta_c c_{t-1}+(1-\beta_c)\delta_t^2,\qquad B_t=C\sqrt{c_t/(1-\beta_c^t)},\\\widetilde\delta_t&=\operatorname{clip}(\delta_t,-B_t,B_t),\\a_t&=\beta_a a_{t-1}+(1-\beta_a)|\widetilde\delta_t|,\qquad \widehat A_t=\frac{\widetilde\delta_t}{\max\{a_t/(1-\beta_a^t),10^{-12}\}}.\end{aligned}
$$

critic 使用 $\widetilde\delta_t$，actor 使用 $\widehat A_t$。作者默认 $C=20$、$\beta_c=\beta_a=0.9998$。统计量包含当前误差，并采用裁剪后的误差更新 actor 尺度；交换顺序会得到不同算法。

$$
g_t^{\rm actor}=\nabla_\theta\left[\log\pi_\theta(a_t\mid s_t)+\xi\,\mathcal H(\pi_\theta(\cdot\mid s_t))\,\operatorname{sg}\!\left(\operatorname{sign}(\delta_t)\right)\right]
$$

这是作者 actor 实际送入优化器的梯度。TD 误差由更新前的 critic 计算，符号作为常数，不沿 critic 反传；裁剪与正尺度归一化保持符号，因此也可用处理后优势的符号。该梯度一起进入 RMSProp 统计与资格迹，随后再乘归一化优势。无迹时，符号修正使熵贡献乘上优势的绝对值；有迹时还混合了过去的熵梯度，不能据此断言每次更新都增加当前状态熵。它不同于在最终更新中另加一项不经过资格迹的熵梯度。

**算法：Intentional optimizer 的运算顺序；网络与行为策略由 actor–critic 主循环提供**

1. 用旧 critic 计算 $\delta$，固定它的符号。
1. critic 使用 $g=\nabla V(s)$；actor 使用 $g=\nabla[\log\pi+\xi\mathcal H\,\operatorname{sign}(\delta)]$。
1. 二者有独立优化器状态：
  1. 更新梯度二阶矩，得到 $D$。
  1. 更新资格迹 $z$、当前梯度范数 $\sigma$ 和偏差校正平均 $\bar\sigma$。
  1. 根据 $\sqrt{\bar\sigma\,z^\top Dz}$ 计算步长。
  1. 更新误差裁剪统计，得到 $\widetilde\delta$。
  1. actor 额外更新 $|\widetilde\delta|$ 的尺度统计并归一化。
  1. 执行参数更新；若是真正 episode 边界，在本次更新后清空资格迹。

作者默认归一化顺序的标准库实现，以及能精确分析的未归一化 TD(0) 特例。

```python
@dataclass
class IntentionalStep:
    """Vector optimizer matching the full author's default normalization order.

    The caller supplies +grad V for the critic. For the author's actor use
    +grad_theta[log pi(a|s) + xi*H(pi(.|s))*sign(delta)], with the sign of the
    pre-update critic's delta held constant. Entropy enters the trace and the
    RMSProp statistics with the score gradient; it is not added after the step.
    Clipping and positive-scale normalization preserve delta's sign. Omitting
    this sign reverses the immediate entropy contribution for negative delta.
    With nonzero traces, past entropy terms may have different signs; this is
    not a guarantee that each update increases the current state's entropy.
    'policy' additionally normalizes the clipped TD error by its running L1 scale.
    No environment, neural network, or author benchmark is reproduced here.
    """
    size: int
    eta: float = 0.5
    q: float = 0.72  # gamma*lambda, must be in [0, 1)
    beta2: float = 0.999
    beta_clip: float = 0.9998
    beta_norm: float = 0.9998
    clip_mult: float = 20.0
    policy: bool = False
    t: int = 0
    sigma: float = 0.0
    delta_sq: float = 0.0
    delta_abs: float = 0.0
    v: list = field(default_factory=list)
    e: list = field(default_factory=list)

    def __post_init__(self):
        if self.size < 1 or not 0 <= self.q < 1:
            raise ValueError('positive size and 0 <= q < 1 required')
        if any(not 0 <= b < 1 for b in (self.beta2, self.beta_clip, self.beta_norm)):
            raise ValueError('EMA decays must lie in [0,1)')
        self.v, self.e = [0.0] * self.size, [0.0] * self.size

    def step(self, g, delta, reset=False):
        if len(g) != self.size or not all(math.isfinite(x) for x in [*g, delta]):
            raise ValueError('finite gradient with declared dimension required')
        self.t += 1
        self.v = [self.beta2 * v + (1 - self.beta2) * gi * gi
                  for v, gi in zip(self.v, g)]
        rho = [1 / (math.sqrt(v / (1 - self.beta2 ** self.t)) + 1e-8)
               for v in self.v]
        sigma_now = sum(gi * gi * ri for gi, ri in zip(g, rho))
        self.e = [self.q * e + gi for e, gi in zip(self.e, g)]
        self.sigma = self.q * self.sigma + (1 - self.q) * sigma_now
        sigma_bc = self.sigma / (1 - self.q ** self.t)
        trace_norm = sum(e * e * ri for e, ri in zip(self.e, rho))
        alpha = self.eta / max(math.sqrt(sigma_bc * trace_norm), 1e-8)
        self.delta_sq = self.beta_clip * self.delta_sq + (1 - self.beta_clip) * delta**2
        cap = self.clip_mult * math.sqrt(self.delta_sq / (1 - self.beta_clip ** self.t))
        safe = math.copysign(min(abs(delta), cap), delta)
        if self.policy:
            self.delta_abs = self.beta_norm * self.delta_abs + (1 - self.beta_norm) * abs(safe)
            norm = self.delta_abs / (1 - self.beta_norm ** self.t)
            safe /= max(norm, 1e-12)
        update = [alpha * safe * ri * e for ri, e in zip(rho, self.e)]
        if reset:
            self.e = [0.0] * self.size
        return dict(update=update, alpha=alpha, safe_delta=safe, sigma=sigma_bc)


def intentional_td0(w, x, reward, next_x, gamma, eta):
    """Unpreconditioned, unclipped TD(0): freeze the old bootstrap target."""
    target = reward + gamma * dot(w, next_x)
    delta = target - dot(w, x)
    norm_sq = dot(x, x)
    if norm_sq == 0:
        return list(w), target, delta
    new_w = [wi + eta * delta * xi / norm_sq for wi, xi in zip(w, x)]
    return new_w, target, delta
```

偏差校正中的 $t$ 从优化器第一次更新开始计数，初始二阶矩、误差统计和梯度范数统计均为零。episode 结束时，在完成本次更新之后清空 $z$；其余幅度统计与更新计数保留。actor 与 critic 分别维护这些状态，不能共用一个尺度估计。

比较这些方法时，应保持网络、资格迹、奖励处理和交互预算相同：固定步长提供基线；Metatrace 增加对学习过程的敏感度与元信用分配；Intentional updates 改变每步输出尺度；Stream-X 则是一套为 streaming 深度控制设计的组件组合。任何一条的正结果都不足以单独判定其他组件无关。

<a id="lesson-example"></a>

## 7 · 两步链的流式执行

状态特征为 $x_0=(1,0)$、$x_1=(0,1)$；转移 $s_0\to s_1$ 的奖励为 0，$s_1\to\mathrm{terminal}$ 的奖励为 1。初始化 $w=z=(0,0)$，取 $\gamma=0.9$、$\lambda=0.8$、$\alpha=0.1$。第一步 $\delta_0=0$、$z_0=(1,0)$，权重不变。第二步 $\gamma_{\rm next}=0$，但 $\gamma_{\rm current}=0.9$，因此 $z_1=(0.72,1)$、$\delta_1=1$；最终 $w=(0.072,0.1)$。

若 $\lambda=0$，同一趟只更新第二个状态，得到 $w=(0,0.1)$；起点要等下次访问才利用新的后继预测。资格迹使用同一个 $\delta$ 乘保存的历史敏感性，将一次误差信号分配给过去，无需重放终点样本。

两种常见错误可以直接手算定位：在 terminal 更新前清零 z，会丢掉起点的 0.072；把当前 terminal 的 γnext=0 用来衰减过去痕迹，也会得到相同错误。测试把这两个 discount 分为独立参数，避免常数 γ 隐藏索引问题。

<a id="lesson-code"></a>

## 8 · 机制实验与作者实现

线性 TD(λ)、Welford 与 2024 ObGD 的可执行机制实验。

```python
@dataclass
class Welford:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def observe(self, x):
        self.count += 1
        old_delta = x - self.mean
        self.mean += old_delta / self.count
        self.m2 += old_delta * (x - self.mean)
        return self.mean, self.variance

    @property
    def variance(self):
        return self.m2 / (self.count - 1) if self.count > 1 else 1.0


def td_lambda_step(w, trace, x, xp, reward, gamma_current,
                   gamma_next, lam=0.8, alpha=0.1, kappa=None):
    """Linear TD: gamma_current decays past trace; gamma_next bootstraps.
    kappa enables the ObGD algebra, NOT a general nonlinear safety theorem.
    """
    if not (len(w) == len(trace) == len(x) == len(xp)):
        raise ValueError("dimension mismatch")
    v = sum(a*b for a, b in zip(w, x))
    vp = sum(a*b for a, b in zip(w, xp))
    error = reward + gamma_next * vp - v
    z = [gamma_current * lam * old + feature
         for old, feature in zip(trace, x)]
    step = alpha
    if kappa is not None:
        mass = alpha * kappa * max(abs(error), 1.0) * sum(map(abs, z))
        step = alpha / max(1.0, mass)  # defined also when z == 0
    return [a + step * error * b for a, b in zip(w, z)], z, error, step


def streaming_demo():
    outcomes = {}
    for lam in (0.0, 0.8):
        w, z, _, _ = td_lambda_step([0., 0.], [0., 0.], [1., 0.],
                                    [0., 1.], 0, 0, 0.9, lam)
        w, z, _, _ = td_lambda_step(w, z, [0., 1.], [0., 0.],
                                    1, 0.9, 0, lam)
        outcomes[str(lam)] = [round(v, 6) for v in w]
    stats = Welford()
    for x in (1., 2., 3.):
        stats.observe(x)
    bounded = td_lambda_step([0.], [0.], [10.], [0.], 1, 0, 0,
                             alpha=0.1, kappa=2)
    print("streaming", {"two_step_credit": outcomes,
          "mean_variance": (stats.mean, stats.variance),
          "obgd_illustration_weight": bounded[0],
          "warning": "gradient 10 violates the <=1 heuristic bound"})
```

仅标准库；包含 ObGD 尺度条件不满足时的过冲反例

```sh
python lifelong_algorithms_lab.py streaming
python lifelong_algorithms_lab.py test
```

输出中 $\lambda=0$ 为 $[0,0.1]$，$\lambda=0.8$ 为 $[0.072,0.1]$；输入 $1,2,3$ 的均值为 $2$、样本方差为 $1$。尺度反例的更新后权重为 $0.5$。下载脚本覆盖这些机制；深度控制实验使用下方作者实现。2024 分支对应 ObGD，2026 分支对应 StreamingOptimizer，二者的配置不能混用。

Intentional 输出尺度实验及回归测试；仅标准库

```sh
python3 meta_frontier_lab.py intentional
python3 meta_frontier_lab.py test
```

IntentionalStep 实现梯度缩放、资格迹、误差裁剪和 actor 误差归一化。它接收网络梯度与旧 TD 误差，不包含神经网络和环境；无迹线性特例用于检查输出变化，带迹与熵项的测试用于检查更新顺序。完整控制结果需使用下方作者训练入口。

- 诊断量一：每步 |δ|、trace 的 L1/L2 norm、实际参数更新 norm 和 αused/α，分别看是目标、历史累积还是步长造成尖峰。
- 诊断量二：原始及归一化观测/奖励的分位数，LayerNorm 前后的激活统计；不要只记录平均值。
- 诊断量三：峰值内存、单步 p50/p95/p99 延迟；用严格相同计算预算比较是否允许 replay 的方法。
- 消融顺序：线性 TD0→线性 traces→网络 TD0→network traces→逐个稳定化组件；每次只增加一个新的故障来源。

<a id="lesson-branches"></a>

## 9 · 从局部更新到持续学习问题

| 实际问题 | 首个可解释基线 | 不应跳过的下一步 |
| --- | --- | --- |
| 短期信用难 | TD(λ) / Sarsa(λ) | λ 与 trace 截断；固定参数前向/后向核对 |
| Off-policy 预测发散 | 重要性比 TD 与梯度 TD 对照 | 覆盖、trace 方差与投影目标；不能只减步长 |
| 输入数值尺度改变 | 在线统计和线性尺度实验 | 统计漂移是否改变目标，是否有未来泄露 |
| 深网每步不稳定 | 完整 Stream-X 组件消融 | 初始化、网络、优化器、归一化作为整体记录 |
| 平均收益目标 | differential error 与独立 rate learner | γ=1 的 trace 管理、平均奖励理论不能照搬折扣保证 |

研究“仅一次真实交互”时，还要把动作延迟带来的环境状态变化纳入协议。若一次复杂更新占用了十个控制周期，却仍只按一条 transition 计时，它与固定时钟世界不是同一问题。

<a id="lesson-check"></a>

## 10 · 习题与答案线索

- 为什么不把资格迹等同于 Adam 的一阶矩？答：痕迹积累的是输出梯度；当前 δ 到来时才分配误差。Adam 的矩积累的是各个时刻已经乘各自误差的 loss gradient。
- 为什么 terminal 的 γnext=0 不应清掉本步全部历史信用？答：terminal 奖励属于刚结束的这一串决策，仍应沿到达当前状态的历史痕迹传播。
- 为何一个方法没有 buffer，却可能不满足固定预算？答：RTRL 的敏感性矩阵、长历史重算或无界模型增长可能消耗随规模增长的内存和时间。
- 能否用归一化证明长期稳定？答：归一化只控制部分尺度，目标漂移、off-policy 投影、非线性与策略反馈仍然存在。

动手题：在两步链加入第三个延迟状态，手算起点更新为 $α(γλ)^2$；再将特征整体乘 10，比较固定 α、按平方缩放 α 与 ObGD。将同一样本误差改变量和长期预测误差分开画，观察它们何时不一致。

## 本章的实验设计

同时记录交互次数、每步延迟和持久内存。均值延迟可能掩盖超时步骤。

设定：一条连续数据流逐步到达，限定每步计算和持久内存。改变未来后缀、特征尺度和控制周期，检查学习前缀及超时行为。

- 未来奖励与标签改变不影响此前已完成的更新。
- 归一化只使用协议允许的当前和过去信息。
- 声明的数据重访次数、状态大小及 fallback 与日志一致。

对照：调好的固定步长与简单尺度控制；Stream-X/Intentional 组件的同基座消融；同交互与同控制时钟分别比较

记录：每步延迟分布和超时率；持久状态大小、样本读取与梯度次数；完整生命期收益和变化后恢复

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-streaming)

## 学习与研究衔接

在线交互不等于严格 streaming。必须分别说明回放、批量、每步计算与持久存储。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-streaming) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=streaming) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=streaming)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

状态是支持后续计算的内部信息；GVF 指定一个预测问题；RTRL 和资格迹规定信用如何传播。三者可以组合，但不是相互替代的算法名称。先理解给定策略的预测，再讨论预测怎样改善控制。

- [Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks](https://yingwen.io/zh/continual-rl/research/#recent-columnar-constructive-networks)
- [Real-Time Recurrent Learning using Trace Units in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-real-time-trace-units)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Streaming Deep Reinforcement Learning Finally Works](https://yingwen.io/zh/continual-rl/research/#recent-stream-x)
- [Intentional Updates for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-intentional-updates)
- [Revisiting Adam for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-revisiting-streaming-adam)
- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)
- [Real-Time Recurrent Learning using Trace Units in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-real-time-trace-units)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)
- [Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-upgd-utility-protection)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-upgd-utility-protection)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Step-size Optimization for Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-step-size-optimization)
- [Learning from experience instead of curated datasets](https://yingwen.io/zh/continual-rl/research/#recent-oak-network-idbd)
- [Intentional Updates for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-intentional-updates)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [Revisiting Adam for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-revisiting-streaming-adam)

### Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks

Khurram Javed, Haseeb Shah, Richard S. Sutton, Martha White

JMLR 24 · 2023 · 支持方法与理论

#### 研究问题

如果每次观测只处理一次，如何学习包含历史信息的状态，而不保存一段序列做反向传播？

#### 关键机制

一般递归网络的实时递归学习需要维护庞大的参数—状态敏感度。CCN 限制列之间的递归依赖，并逐步构造新特征，使敏感度可以局部计算。它通过改变网络结构和构造过程降低求导成本，而不是把任意稠密 RNN 的完整导数免费变小。

#### 证据

论文分析受限结构的计算性质，并在动物学习启发的预测问题和 Atari 策略评价中检验预测效率。这里的 Atari 结果主要是预测已有策略的回报，不等于从头训练完整控制智能体。

#### 条件与限制

结构约束、构造顺序和被冻结的旧特征共同限制函数类。监督预测和策略评价上的优势，还需要在会主动改变数据分布的控制闭环中检验。

#### 阅读与实验

先写出递归状态对参数的敏感度递推，再检查哪些跨列项被结构消除。比较 CCN、截断 BPTT 与 RTU 时，同时计入状态、梯度缓存和每步计算。

#### 原文与相关入口

- [JMLR 原文与论文入口](https://www.jmlr.org/papers/v24/23-0367.html)：从网络结构、敏感度传播与预测实验三部分阅读。

### Real-Time Recurrent Learning using Trace Units in Reinforcement Learning

Esraa Elelimy, Adam White, Michael Bowling, Martha White

NeurIPS 2024 · 2024 · 支持方法与理论

#### 研究问题

递归状态既要保存长时信息，又要在在线强化学习中以可控成本更新，怎样设计其递归结构？

#### 关键机制

RTU 使用有结构的递归连接，并维护状态关于参数的在线敏感度。复杂的递归动力学可以用实值运算实现。其关键是让状态更新与梯度迹具有相容的计算结构，减少一般 RTRL 的高阶成本；这与仅给 TD 误差加一条资格迹不同。

#### 证据

论文在部分可观测任务中与常见递归网络比较预测与控制表现。作者代码包含 RTU、其他递归基线、实时 actor–critic 以及部分可观测环境配置。

#### 条件与限制

计算优势依赖特定递归参数化，不能外推为任意记忆问题上的表达能力优势。PPO 版本和严格逐步更新版本的经验协议不同，应分别比较。

#### 阅读与实验

在同一部分可观测任务中固定隐状态维度，再比较完整运行内存和每步更新时间。检查 actor、critic 与递归状态的参数更新是否共享同一条敏感度。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1e616bde0438cb10cb6adf076ae7d336-Abstract-Conference.html)：结构、在线导数与实验协议。
- [作者代码](https://github.com/esraaelelimy/rtus)：从 src/nets、src/agents 和实验配置追踪递归状态到控制更新。

#### 作者代码

[论文作者维护的实现。](https://github.com/esraaelelimy/rtus)

RTU 网络、实时学习器与论文实验配置。

### Deep Reinforcement Learning with Gradient Eligibility Traces

Esraa Elelimy, Brett Daley, Andrew Patterson, Marlos C. Machado, Adam White, Martha White

RLC 2025 / RLJ · 2025 · 支持方法与理论

#### 研究问题

资格迹怎样与明确的梯度目标结合，而不是直接把线性半梯度规则搬到深度网络？

#### 关键机制

论文从广义投影 Bellman 误差出发构造多步目标，推导带资格迹的梯度学习方法。前向视角连接多步回报与经验重放，后向视角通过递推迹分配信用。目标函数、辅助估计器和迹的更新共同决定算法，不只是选择一个较大的 λ。

#### 证据

作者给出多种算法并在 MuJoCo、MinAtar 等任务中比较。代码同时提供相关梯度算法与实验设置，可以把推导中的量映射到实际更新。

#### 条件与限制

线性 GTD 的收敛条件不能自动赋予非线性实现全局收敛保证。重放版本与流式版本的数据使用预算也不能混为一谈。

#### 阅读与实验

从一段短轨迹分别计算前向多步目标和后向迹。随后对照原代码检查辅助网络、目标与主网络参数使用的是更新前还是更新后的值。

#### 原文与相关入口

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_302.pdf)：目标、算法推导与实验。
- [作者算法库](https://github.com/esraaelelimy/gtd_algos)：论文提供的梯度 TD 与资格迹实现。

#### 作者代码

[原论文链接的作者仓库。](https://github.com/esraaelelimy/gtd_algos)

论文梯度算法、资格迹和实验配置。

### Streaming Deep Reinforcement Learning Finally Works

Mohamed Elsayed, Elena Sorina Lupu, Gautham Vasan, A. Rupam Mahmood

arXiv（2024 首稿；2026 v3） · 2026 · 直接研究持续学习

#### 研究问题

不保存经验重放、不使用目标网络或训练批次时，深度 RL 能否逐步稳定学习？

#### 关键机制

Stream-X 把信号归一化、表示初始化、资格迹和受控更新尺度组织为一组流式学习方法。各组件处理的是不同问题：奖励尺度、激活与梯度传播、延迟信用，以及一次更新造成的输出变化。去掉重放并不意味着这些问题会自动消失。

#### 证据

2026 年第三版扩展到 Atari、控制与机器人等实验，并包含持续变化设置。论文和代码经历过版本变化，比较结果时需要同时标明论文版本和算法实现。

#### 条件与限制

广泛任务上的流式可行性不等于所有非平稳问题都已解决。不能把旧版较弱 Adam 基线推广成对所有流式 Adam 方法的否定；后续研究专门检验了这一点。代码许可证也应独立于本教材许可证处理。

#### 阅读与实验

按归一化、资格迹、更新控制分别做消融，并保持每步算力一致。先验证严格一次使用经验，再研究长期变化，而不是仅把小批量大小改成一。

#### 原文与相关入口

- [2026 年第三版论文](https://arxiv.org/abs/2410.14606v3)：作者名单、任务范围与算法版本以该版为准。
- [作者代码版本](https://github.com/mohmdelsayed/streaming-drl/tree/9326fc3e23a401f28087ae2e41b635888740586b)：固定实现版本，避免把不同年份的更新规则混在一起。

#### 作者代码

[原作者仓库的固定版本。](https://github.com/mohmdelsayed/streaming-drl/tree/9326fc3e23a401f28087ae2e41b635888740586b)

Stream-X 算法、变换、优化器及实验；使用前阅读仓库许可证。

### Intentional Updates for Streaming Reinforcement Learning

Arsalan Sharifnassab, Mohamed Elsayed, Kris De Asis, A. Rupam Mahmood, Richard S. Sutton

ICML 2026 · 2026 · 支持方法与理论

#### 研究问题

能否先规定本次更新应产生多大作用，再反推合适的参数更新尺度？

#### 关键机制

Intentional 方法以局部线性近似连接参数变化和预测变化。critic 以减少一定比例的 TD 误差为目标，actor 控制策略输出变化的局部代理量；再结合资格迹和逐坐标尺度，求出这一次更新的强度。这是有目标的局部更新控制，不是对长期表现求导的元梯度。

#### 证据

论文给出推导和流式控制比较，ICML 2026 正式论文入口与作者实现均可用。实现将优化器与 actor–critic 交互区分开，便于检查更新时序。

#### 条件与限制

Taylor 近似在大更新时可能失准。采样动作上的对数概率变化不等于精确的全分布 KL 上界；熵项与 TD 误差符号也必须按原算法处理。

#### 阅读与实验

在一次更新前后直接测量预测变化，并与线性估计比较。分别测试正、负 TD 误差和很小梯度的情形，不要只检查参数是否有限。

#### 原文与相关入口

- [ICML 2026 原文](https://proceedings.mlr.press/v306/sharifnassab26a.html)：正式会议版本与更新意图的定义。
- [作者实现](https://github.com/sharifnassab/Intentional_RL)：重点对照 optimizer.py 与 intentional_ac.py。

#### 作者代码

[原论文作者提供的实现。](https://github.com/sharifnassab/Intentional_RL)

Intentional 更新与流式 actor–critic。

### Revisiting Adam for Streaming Reinforcement Learning

Florin Gogianu, Luțu Adrian-Cătălin, Razvan Pascanu

RLC 2026 / RLJ 预会议版 · 2026 · 支持方法与理论

#### 研究问题

流式 RL 的不稳定来自 Adam 本身，还是目标导数、方差与超参数的组合？

#### 关键机制

论文重新分析自适应更新的信噪比，将 Adam 的稳定项与目标导数尺度联系起来，并研究有界导数的回报分布学习及多步更新。它改变的是目标与更新的配合，而非简单沿用批量训练时的默认配置。

#### 证据

作者在大规模 Atari 流式实验中展示了具有竞争力的结果，并重新比较早期流式方法。正式 RLJ 入口收录为 RLC 2026 预会议论文。

#### 条件与限制

主体实验采用经典回合式 Atari 的流式学习协议，不是任意非平稳终生适应的证据。这些结果也不否定归一化、资格迹或更新约束在其他任务中的价值。版本、调参预算和目标分布必须对齐。

#### 阅读与实验

建立二维对照：固定目标换优化器，固定优化器换目标。将调参种子与最终测试分开，再判断改进来自哪一个因素。

#### 原文与相关入口

- [RLC 2026 论文入口](https://rlj.cs.umass.edu/2026/papers/Paper131.html)：会议收录信息与论文。
- [作者预印本](https://arxiv.org/abs/2605.06764)：Adam 尺度分析、回报分布目标与实验协议。

### Step-size Optimization for Continual Learning

Thomas Degris, Khurram Javed, Arsalan Sharifnassab, Yuxin Liu, Richard S. Sutton

arXiv 预印本 · 2024 · 支持方法与理论

#### 研究问题

误差变大时，应该减小步长过滤噪声，还是增大步长追踪真实变化？

#### 关键机制

论文区分梯度归一化与步长优化。IDBD 类方法以 $\alpha_i=\exp(\beta_i)$ 保证步长为正，并用权重对过去步长的敏感度估计改变 $\beta_i$ 是否有利。持续学习中，静止的无关方向适合很小步长，而持续变化的有用方向需要保留追踪能力。

#### 证据

作者用权重翻转和带噪追踪等线性学习问题比较机制，显示相似的误差幅度可以要求相反的步长反应。

#### 条件与限制

这些可分析任务不是深度控制上的普适优越性证据。元步长、近似敏感度与输入尺度仍会影响结果；步长自适应并没有消除全部外部设计参数。

#### 阅读与实验

分别增加观测噪声和目标漂移速度，检查步长是否采取不同反应。若只记录平均误差，就看不到噪声过滤与追踪之间的区别。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2401.17401)：步长优化与归一化的对照实验。

### Learning from experience instead of curated datasets

Oak Lab

Oak Lab 技术博文 · 2026 · 支持方法与理论

#### 研究问题

有用信号稀疏且大量输入是噪声时，在线学习规则如何分配不同方向的更新能力？

#### 关键机制

博文从含稀有有效特征的线性预测问题出发，对比统一步长与 IDBD 的逐权重适应，再展示 NetworkIDBD 在非线性带噪观测中的例子。核心主张是让长期学习效果影响信用和步长分配，而不只依据当前梯度幅度归一化。

#### 证据

公开页面提供受控噪声特征任务和 NoisyMNIST 示例。它们是机制演示，便于理解有效信号密度与输入规模的关系。

#### 条件与限制

该页面不是完整 CRL 控制论文，也未给出可直接复现所有图表的完整代码和算法推导。监督噪声任务的结果不能证明一般 SGD 或所有深度 RL 都无法从经验学习。

#### 阅读与实验

先复现线性噪声特征问题，分开改变有效特征稀疏度与噪声维数。进入控制前，再加入策略改变数据分布这一因素。

#### 原文与相关入口

- [Oak Lab 原始博文](https://oaklab.ai/posts/learning-from-experience-instead-of-curated-datasets)：2026 年 7 月 13 日；受控实验、NetworkIDBD 示例与研究动机。

### Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning

Mohamed Elsayed, A. Rupam Mahmood

ICLR 2024 · 2024 · 支持方法与理论

#### 研究问题

同一网络里，哪些方向应当保护，哪些方向应当获得更强的新学习与扰动？

#### 关键机制

UPGD 用移除权重或特征的反事实损失变化定义效用，并以 Taylor 近似在线估计。平滑、缩放后的效用同时调制梯度与随机扰动，让近期高效用方向变化较小、低效用方向更活跃。

#### 证据

主体证据包括未知边界的非平稳流式监督任务；另外包含长时间 PPO 实验。两类证据应分别理解，不能把监督任务数量写成 RL 任务覆盖。

#### 条件与限制

近期分布上的效用不保证稀有旧知识的重要性；一阶和二阶近似、权重级和特征级版本不同。PPO 仍使用 rollout 与重复更新，不因 optimizer 在线就成为严格流式 RL。

#### 阅读与实验

用可精确消融的小网络检查效用估计，再拆开保护梯度、保护噪声和 weight decay 三种作用；独立报告新学习与旧功能。

#### 原文与相关入口

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/file/8e5f0591943d8dae5702af12dcdcd2f6-Paper-Conference.pdf)：效用定义、近似、不同 UPGD 变体与 PPO 实验。
- [作者预印本](https://arxiv.org/abs/2404.00781)：流式监督协议与 RL 证据范围。

#### 作者代码

[论文首页明确链接的作者仓库；README 的短实现是一个指定变体。](https://github.com/mohmdelsayed/upgd)

权重/特征效用实验、流式任务及 PPO 实现。

### Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning

Noah Farr, Aryaman Reddi, Carlo D’Eramo, Jan Peters

arXiv预印本（2026-07-07 v2） · 2026 · 支持方法与理论

#### 研究问题

严格逐步更新的智能体怎样同时学习递归记忆、分配延迟信用并控制计算？

#### 关键机制

将RTU结构的RTRL敏感度接入QRC与流式actor–critic。敏感度给出当前输出对记忆参数的导数，资格迹再组合过去输出的回报信用；两条递推保持分工。

#### 证据

v2在MemoryChain、五项POPGym和masked MuJoCo上报告5-seed结果，另用KMemoryChain比较在线敏感度与当前参数重算参考，并检验Taylor修正。

#### 条件与限制

masked MuJoCo仍落后批量PPO。固定参数精确RTRL不代表在线变参敏感度始终等于当前参数重算；诊断保存整个episode，须计为额外评价资源。尚未确认作者公开代码。

#### 阅读与实验

在相同递归容量下独立改变记忆跨度、回报λ与参数步幅，同时测敏感度误差和回报。诊断改善不能单独当作控制改进证据。

#### 原文与相关入口

- [2026年v2原文](https://arxiv.org/html/2605.24709v2)：方法、5-seed实验、masked MuJoCo负边界与staleness诊断。


<a id="chapter-code"></a>

## 下载与运行

流式 TD、在线统计与 ObGD 的机制检查；meta_frontier_lab.py 提供 Intentional 输出尺度实验。不包含完整神经网络控制训练。

[下载 lifelong_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)

```sh
python lifelong_algorithms_lab.py streaming
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [配套章节：时间信用分配](/zh/continual-rl/algorithms/credit-assignment/)：系统讨论多步回报、资格迹和跨时间信用；流式学习只是其中部分方法可以满足的数据协议。

- [配套章节：学习规则的适应](/zh/continual-rl/algorithms/meta/)：在线元梯度使用学习过程的敏感度调整规则；与本章预设的更新尺度控制区分。

- [Sharifnassab 等：Intentional Updates（ICML 2026）](https://proceedings.mlr.press/v306/sharifnassab26a.html)：Intentional-TD、Intentional-Q 与 Intentional-AC；按预期输出变化选择局部步长。

- [Intentional-AC 作者优化器源码](https://github.com/sharifnassab/Intentional_RL/blob/e86e26fd8613ac212e9a52c3fed8a01d0a31f685/optimizer.py)：二阶矩、sigma 偏差校正、误差裁剪、policy 归一化及更新后 trace reset 的具体顺序。

- [Intentional-AC 作者 actor–critic 主循环](https://github.com/sharifnassab/Intentional_RL/blob/e86e26fd8613ac212e9a52c3fed8a01d0a31f685/intentional_ac.py#L79)：update_params 用旧 critic 构造 TD 误差；actor 熵项乘该误差的符号，随后将合成梯度送入独立的 policy 优化器。

- [Sutton & Barto · Reinforcement Learning, Chapter 12](http://incompleteideas.net/book/the-book-2nd.html)：前向 λ-return、资格迹与控制中的 trace 语义；本页已给出所需核心推导。

- [van Seijen & Sutton · True Online TD Learning](https://proceedings.mlr.press/v32/seijen14.html)：线性在线前向/后向精确等价的适用范围，区别于普通 accumulating traces。

- [Elsayed, Vasan & Mahmood · Stream-X，2024 版](https://arxiv.org/abs/2410.14606v2)：ObGD 的有效步长动机、资格迹与网络稳定化组合；与修订版区分。

- [Stream-X · 2024 作者实现](https://github.com/mohmdelsayed/streaming-drl/tree/407dca7a8b584c1c20bc649053557f66e270b1e6)：optim.py 使用 ObGD；从 stream_ac_continuous.py 核对输出梯度、熵项和误差乘法。

- [Elsayed, Lupu, Vasan & Mahmood · Stream-X，2026 修订版](https://arxiv.org/abs/2410.14606v3)：逐坐标 StreamingOptimizer、layer-aware 初始化及 Stream-RAC；附录 B.2 解释相对于早期方法的变化。

- [Stream-X · 2026 作者实现](https://github.com/mohmdelsayed/streaming-drl/tree/9326fc3e23a401f28087ae2e41b635888740586b)：依次阅读 obs_reward_transforms.py、sparse_init.py、optimizer.py 和相应 agent；代码采用 CC BY-NC 4.0 许可证。

- [Welford · Note on a Method for Calculating Corrected Sums of Squares and Products](https://doi.org/10.1080/00401706.1962.10490022)：在线均值与方差递推的原始统计方法。


---

# 学习规则的适应：在线元梯度与跨任务元学习

学习规则的适应研究经验怎样改变学习速度、训练目标或更新方式。在线元梯度在同一经验流中评价过去更新的影响；跨任务元学习则利用训练任务中的适应经历，改进新任务上的适应。本章分别定义这两种设定，再推导其算法。

## 本章内容

- 明确内层状态、元参数、外层目标与数据权限，区分在线元梯度、跨任务元学习和预设尺度控制。
- 从多步参数更新推导精确敏感度；说明 IDBD/TIDBD、截断元梯度和一阶 MAML 丢掉了哪些项。
- 完整理解 MAML、RL²、PEARL、meta-gradient RL 与 learned update rules 的训练/测试循环及 reset 边界。
- 用可运行的有限差分、手算与机制反例验证代码，并设计适用于持续交互而非仅任务重置的评测。

<a id="problem-definition"></a>

## 本章的问题定义

更新规则或初始化也需由经验改善。单一流的在线元梯度与跨任务元训练具有不同数据、重置和评价单位。

### 给定条件与符号

- 内层学习状态、可微更新映射和可选元参数。
- 固定外部评价、适应长度、后续评价数据、元训练/测试任务与重置权限。

### 需要求解的对象

使指定外层评价改善的初始化、步长、目标或更新规则；不能通过改写评价标准降低外层损失。

### 信息与数据权限

内层状态 $w_t$ 含权重及影响更新的优化器/迹；经验 $\xi_t$ 已到达。跨任务训练允许声明的训练任务，测试任务未来不得参与外层选择。

$$
w_{t+1}=F_\eta(w_t,\xi_t),\qquad \min_\eta\mathcal J(\eta)=\mathbb E[J(w_{t+K},\xi_{t+K:t+K+M};\eta_{\rm eval})]
$$

$\eta$ 为元参数，$F_\eta$ 为更新规则，$K$ 为适应长度，$M$ 为后续评价长度；$J$ 是外层损失（例如负回报），$\eta_{\rm eval}$ 固定评价约定。期望所覆盖的任务、随机性和生命周期必须声明；适应后表现与全程收益不同。

### 成立条件与解的含义

- 固定数据导数需可微更新；完整RL期望导数还包括采样分布依赖，不能由固定轨迹链式法则自动得到。
- 对角、截断和一阶近似明确丢弃哪些敏感度；任务边界、参数/context reset和元训练成本计入协议。

判断准则：小问题多步元敏感度与有限差分一致；在独立后续数据或未见任务上、匹配适应与计算预算评价固定外层目标；在线协议无未来回流。

### 适用边界

- 在完整测试寿命上挑超参数不是智能体在线元学习。
- 跨任务快速适应不自动证明单一无重置生命期持续学习。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)：元敏感度追踪更新规则对后续学习的影响，不是过去预测的普通资格迹。

- 改变信息或数据协议 · [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)：元梯度若缓存rollout或展开图，不自动满足严格流式；预算限制可用近似。

- 组合不同学习问题 · [奖励假设与奖励设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/)：内在奖励或训练目标可作为元参数，由固定外部评价选择。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

一次规则改变会通过许多后续更新影响表现；改变策略还改变未来样本。

### 本章的核心思路

把内层全部状态纳入更新映射，先求完整敏感度，再声明哪些路径为降低成本而近似。

1. [传播规则对未来状态的影响](#lesson-derive)：因为元参数既直接改变本次更新又经旧状态间接传播，Jacobian递推包含两条路径。

2. [按在线预算压缩敏感度](#lesson-idbd)：因为完整矩阵昂贵，IDBD保留对角，TIDBD/Metatrace还需处理bootstrap和迹；它们不是精确链式法则。

3. [对齐训练/测试适应单位](#lesson-meta-rl)：因为MAML、context与学习更新规则改变不同内层对象，分别声明任务分布、reset和外层封存，独立评价适应。

结论与条件：固定数据和元参数下完整敏感度递推是链式法则；截断/一阶/对角更新改变导数。没有相应采样路径估计时不能称为完整RL无偏元梯度。

### 相关方法改变了什么

- IDBD/TIDBD/Metatrace：在线适应学习参数，以结构近似减少敏感度成本。

- MAML：跨任务学习适合少量梯度更新的初始化。

- RL²/PEARL与learned rules：前者主要适应活动/context，后者学习更新信号，参数和reset边界不同。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 两种时间尺度

内层变量随当前经验适应；外层变量决定内层怎样适应，并由后续表现训练。“外层”不必在另一台机器，也不必慢到离线。

### 梯度与 Hessian

梯度描述损失随参数的一阶变化；Hessian 描述梯度本身如何改变。对一次梯度下降再求导，自然会出现 Hessian，而不是额外加的技巧。

$$
\frac{\partial (w-\alpha\nabla L(w))}{\partial w}=I-\alpha\nabla^2L(w)
$$

### 固定数据条件下求导

可微优化循环通常先把采样轨迹当常量。RL 的策略变化还会改变轨迹分布；是否估计这条路径必须另行说明。

<a id="lesson-setting"></a>

## 1. 问题设定：哪些变量在适应，适应后的表现如何评价

环境变化可以要求不同形式的适应。例如，地面摩擦系数改变后，机器人可以更新控制参数、推断新的动力学情境，或改变参数学习的速度。三种方法分别改变模型参数、活动状态和学习规则，所依赖的数据与评价目标也不同。

本章有两种主要数据设定。在线元梯度可在一个持续运行的问题上工作：权重不断学习，步长或学习目标也根据后续误差调整，不要求出现一组可重置的任务。跨任务 meta-RL 则先规定任务分布与任务内适应过程，用训练任务中的经历训练初始化、推断器或更新规则，再检验新任务适应。两者都涉及学习过程，但评价单位与信息权限不同。

| 设定 | 在线元梯度 | 跨任务 meta-RL |
| --- | --- | --- |
| 数据组织 | 同一经验流中的先前更新与后续评价 | 训练任务中的适应数据和适应后评价数据 |
| 持久状态 | 权重与元敏感度通常跨更新保留；按真实边界处理资格迹 | 按任务或 trial 协议复制初值、清理 context 或循环活动 |
| 部署时变化 | 权重与元参数均可继续更新 | 外层参数常固定，任务内权重或活动状态继续适应 |
| 主要检验 | 长寿命中的累计误差、控制表现与变化后适应 | 未见任务上，在规定数据与 reset 预算内的适应 |
| 额外条件 | 元梯度展开、数据复用与每步预算 | 任务分布、元训练成本、适应次数与测试任务隔离 |

在线元梯度也可能缓存 rollout 或展开计算图，因此不自动满足严格流式协议。跨任务训练得到的规则也可以用于持续交互，但跨任务适应表现不能替代单次长生命中的持续学习证据。资格迹解决的是时间信用分配；它既可服务普通 TD，也可服务元目标，不以元学习为前提。

| 路线 | 内层适应什么 | 外层学什么 | 典型监督/边界 |
| --- | --- | --- | --- |
| 在线超参数学习：IDBD/TIDBD、meta-gradient RL | 预测器或策略权重 $w$ | 步长 $\alpha$、discount、$\lambda$、更新系数等 $\eta$ | 后续误差或回报；可在单一流上工作，不必有任务 ID。 |
| 优化型 Meta-RL：MAML | 从共同初值出发的任务内权重 $w_i$ | 适合少量梯度适应的初始化 $\theta$ | 任务分布、适应数据与适应后数据；通常有明确 trial/reset。 |
| 上下文型 Meta-RL：RL²、PEARL | 循环活动 $h$ 或任务潜变量 $z$ | 推断与控制网络参数 $\theta,\phi$ | 训练任务分布；测试时可仅更新上下文，不更新网络参数。 |
| 学习更新规则：LPG、DiscoRL 等 | 被训练的 agent 参数与学习器状态 | 产生目标/损失/更新信号的规则 $\eta$ | 跨环境、跨训练轨迹的外层表现；元训练预算必须单列。 |

统一记经验为 $\xi_t$，内层学习状态为 $w_t$，元参数为 $\eta$，更新映射为 $F_\eta$，外层损失为 $J$。这里的“状态”包括所有影响后续学习的量：除了网络权重，还可能有优化器动量、资格迹和归一化统计量。将它们纳入更新映射，才能定义需要求导的过程；把某些状态视为常数，则对应一个明确的梯度近似。

$$
w_{t+1}=F_\eta(w_t,\xi_t),\qquad \mathcal J(\eta)=\mathbb E\bigl[J(w_{t+K},\xi_{t+K:t+K+M};\eta_{\rm eval})\bigr]
$$

$K$ 是适应长度，$M$ 是后续评价长度；$\eta_{\rm eval}$ 表示固定的评价约定。训练代理目标可以学习，最终评价标准仍需要事先定义。

- 先说明数据协议：独立任务、同一任务内多 episode，还是一条未分段的生命流。
- 再说明评价目标：预测误差、适应后回报、训练全过程回报、变化后恢复速度，不能彼此代替。
- 最后说明测试权限：能否重置参数/上下文/环境，是否拿到任务边界，是否继续更新外层参数。

<a id="lesson-derive"></a>

## 2. 多步更新的敏感度与链式法则

先固定 $\eta$ 和采样数据，定义 $H_t=\partial w_t/\partial\eta$。对更新 $F$ 求导，要包含 $F$ 对 $\eta$ 的直接依赖，以及 $\eta$ 通过旧参数 $w_t$ 产生的间接依赖。若 $\eta$ 就是初始化，则 $H_0=I$；若 $\eta$ 是独立步长而初始 $w_0$ 与它无关，则 $H_0=0$。

$$
H_{t+1}=\frac{\partial F_\eta}{\partial w_t}H_t+\frac{\partial F_\eta}{\partial\eta}
$$

这是固定元参数、固定经验序列下的精确前向敏感度。若 $F=w+f$，传播矩阵为 $I+\partial f/\partial w$，旧敏感度通常会被当前更新的 Jacobian 改变。

$$
\nabla_\eta J=H_{t+K}^{\top}\nabla_w J+\left.\nabla_\eta J\right|_w
$$

若评价损失只通过适应后的 $w$ 依赖 $\eta$，右端第二项为零。若评价目标也含可学习 discount，第二项可能通过缩短评价时间尺度降低损失，而没有改善原定的长期目标。

$$
\begin{aligned}F_\beta(w)&=w-\alpha\nabla_w L(w),\qquad \alpha=e^\beta,\\H_{t+1}&=(I-\alpha\nabla_w^2L_t)H_t-\alpha\nabla_wL_t.\end{aligned}
$$

对数步长 $\beta$ 保证 $\alpha>0$。第一项传播已有敏感度，第二项表示步长对本次更新的直接影响。敏感度矩阵 $H$ 的行数是内层状态维度，列数是元参数维度；二者都很大时需要结构或近似。

只看一次内层更新，且 $H_t=0$ 时，外层对 $\beta$ 的导数为 $-\alpha g_{\rm train}^{\top}g_{\rm eval}$。两个梯度方向一致，适当增大步长会局部改善后续损失；方向相反，则倾向缩小步长。这是局部导数的解释，不是任意大步长下的单调性保证。样本噪声和梯度相关性还会影响该信号的可信度。

$$
\frac{\partial J(w-\alpha g_{\rm train})}{\partial\beta}=-\alpha\,g_{\rm eval}^{\top}g_{\rm train}
$$

$g_{\rm eval}$ 在更新后的参数上计算，$g_{\rm train}$ 在更新前计算。把两者都在旧参数上计算，得到的是另一种近似。

RL 还有采样路径：$\eta$ 改变策略，策略改变动作及后续数据。完整期望回报的导数一般包含轨迹分布的 score-function 项。固定数据的解析实验能够隔离优化器微分；采样路径是否无偏，则需要在随机过程的层面另外检验。

两步及多步学习的完整标量敏感度；旧 w 和旧 H 一致使用，可由有限差分独立核对。

```python
def fixed_beta_training(beta, train_targets, validation_target, initial=0.0):
    """Exact unrolled log-step-size derivative for FIXED beta and fixed data.

    w_next = w + exp(beta)*(target-w). H=dw/dbeta, H0=0.
    Includes every inner update Jacobian; it is not an online changing-beta
    total derivative and does not differentiate the data collection policy.
    """
    alpha = math.exp(beta)
    w, h = initial, 0.0
    trace = []
    for target in train_targets:
        error = target - w
        h = (1.0 - alpha) * h + alpha * error  # use old w and H
        w = w + alpha * error
        trace.append((w, h))
    loss = 0.5 * (w - validation_target) ** 2
    hypergradient = (w - validation_target) * h
    return loss, hypergradient, trace
```

<a id="lesson-idbd"></a>

## 3. IDBD：从多步链式法则到每个特征一个步长

IDBD 考虑线性监督预测：输入 $x_t$、标签 $y_t$、预测 $w_t^\top x_t$，误差为 $\delta_t=y_t-w_t^\top x_t$。每个特征 $i$ 有自己的步长 $\alpha_i=\exp(\beta_i)$。一个方向的更新持续同向，可能表示跟踪速度不足；频繁反向，则可能是更新过冲。IDBD 通过权重对过去步长的敏感度 $h_i$ 估计这种关系，而不只统计梯度幅度。

$$
\delta_t=y_t-w_t^\top x_t,\qquad w_{i,t+1}=w_{i,t}+\alpha_{i,t}\delta_t x_{i,t}
$$

这是内层 LMS/线性 SGD 更新。所有特征共享误差 $\delta$，但各坐标的 $\alpha_i$ 不同，因此改变的不只是总更新长度，还包括学习的几何方向。

精确敏感度是矩阵 $\partial w_j/\partial\beta_i$：改变一个权重会改变公共误差，再影响其他权重。IDBD 只保存 $h_i\approx\partial w_i/\partial\beta_i$，忽略 $i\ne j$ 的交叉影响。在固定 $\beta$ 的对角近似下，对权重更新求导得到保留因子 $1-\alpha_i x_i^2$，以及直接作用项 $\alpha_i\delta x_i$。

$$
\begin{aligned}\beta_i^+&=\beta_i+\mu\delta x_i h_i,\qquad \alpha_i=e^{\beta_i^+},\\w_i^+&=w_i+\alpha_i\delta x_i,\\h_i^+&=h_i\max(0,1-\alpha_i x_i^2)+\alpha_i\delta x_i.\end{aligned}
$$

$\delta$ 来自更新前的权重，$\beta$ 使用旧 $h$ 更新，之后才计算新 $\alpha$、$w$ 与 $h$。正截断避免保留因子为负，是稳定化操作而非精确链式法则。$\mu$ 是另需指定的元步长。

**算法：算法伪代码**

1. 初始化 $w=0$、$h=0$、$\beta_i=\log\alpha_{i,0}$。
1. 每个样本 $(x,y)$：
  1. $\delta\leftarrow y-w^\top x$              # 更新前的预测误差
  1. $\beta_i\leftarrow\beta_i+\mu\delta x_i h_i$ # 使用旧敏感度
  1. $\alpha_i\leftarrow\exp\beta_i$
  1. $w_i\leftarrow w_i+\alpha_i\delta x_i$
  1. $h_i\leftarrow h_i\max(0,1-\alpha_i x_i^2)+\alpha_i\delta x_i$
1. 用更新前误差评价当前样本，下一样本再评价学习后的预测。

初始 $h=0$，第一步元更新为零，因为尚无过去步长影响当前权重的历史。普通权重更新产生非零 $h$ 后，后续误差才能评价这种影响。特征尺度、强相关性、噪声与过大的元步长 $\mu$ 都可能使适应不稳定，因此仍需监测步长及敏感度。

线性监督预测的 IDBD：保存旧误差与旧敏感度，依次更新对数步长、权重和敏感度。

```python
def idbd_step(weights, beta, history, features, target, meta_rate=0.01):
    """Linear supervised IDBD, with diagonal sensitivity and positive clipping."""
    error = target - sum(w * x for w, x in zip(weights, features))
    new_beta = [b + meta_rate * error * x * h
                for b, x, h in zip(beta, features, history)]
    alpha = [math.exp(b) for b in new_beta]
    new_weights = [w + a * error * x
                   for w, a, x in zip(weights, alpha, features)]
    new_history = [h * max(0.0, 1.0 - a * x * x) + a * error * x
                   for h, a, x in zip(history, alpha, features)]
    return new_weights, new_beta, new_history, error
```

<a id="lesson-tidbd"></a>

## 4. TIDBD：bootstrap 与资格迹改变了什么？

TD 的目标不是外部固定标签 $y$，而是奖励加下一状态的当前预测。因此，对 TD 误差求导时必须指定是否对 bootstrap 目标求导。以下采用 2018 TIDBD Algorithm 1：线性、on-policy、常数 $\gamma$、累积资格迹的半梯度版本。后续 TIDBD 和 AutoTIDBD 的完整误差导数、步长归一化等扩展，需要分别识别。

$$
\begin{aligned}\delta_t&=R_{t+1}+\gamma w_t^\top x_{t+1}-w_t^\top x_t,\\z_t&=\gamma\lambda z_{t-1}+x_t,\\w_{i,t+1}&=w_{i,t}+\alpha_{i,t}\delta_t z_{i,t}.\end{aligned}
$$

$z$ 是内层资格迹，$h$ 是步长敏感度；$\gamma$ 定义预测时间尺度，$\lambda$ 控制信用回溯。资格迹在真实 episode 间如何清零由协议决定，未通知的分布变化不提供额外清零信号。

$$
\begin{aligned}\beta_i^+&=\beta_i+\mu\delta_t x_{i,t}h_{i,t},\qquad\alpha_i=e^{\beta_i^+},\\h_i^+&=h_i\max(0,1-\alpha_i x_{i,t}z_{i,t})+\alpha_i\delta_t z_{i,t}.\end{aligned}
$$

外层局部误差的半梯度使用当前 $x$，内层更新通过资格迹 $z$ 分配信用。因此 $\beta$ 更新中是 $x$，敏感度传播和权重更新中则同时出现 $z$。

$$
\left.\frac{\partial\delta_t}{\partial w_i}\right|_{\rm semi}=-x_{i,t},\qquad\left.\frac{\partial\delta_t}{\partial w_i}\right|_{\rm full}=\gamma x_{i,t+1}-x_{i,t}
$$

若改为完整 TD-error 导数，外层方向涉及 $x_t-\gamma x_{t+1}$，$h$ 的传播因子也相应变化。目标导数约定、敏感度递推与步长归一化应成套指定。

当 $\gamma=0$、旧资格迹不再贡献且 $z=x$ 时，更新退化为 IDBD。真正 terminal 的 bootstrap 为零，可以将下一状态特征置零，在完成本次更新后清空 $z$；采样预算造成的 timeout 不自动等于 terminal。自适应步长本身也不改变 off-policy TD 的稳定性条件。

TIDBD 的全部一转移更新。调用者负责真实 episode 的资格迹边界；脚本不自行猜任务变化。

```python
def tidbd_step(weights, beta, history, eligibility, features, next_features,
               reward, gamma=0.9, lam=0.8, meta_rate=0.01):
    """2018 TIDBD(lambda) semi-gradient version, on-policy, constant gamma.

    This is NOT the later full TD-error-gradient/normalization variants.
    When the protocol declares a genuine new episode, the CALLER clears z.
    """
    value = sum(w * x for w, x in zip(weights, features))
    next_value = sum(w * x for w, x in zip(weights, next_features))
    delta = reward + gamma * next_value - value
    new_beta = [b + meta_rate * delta * x * h
                for b, x, h in zip(beta, features, history)]
    alpha = [math.exp(b) for b in new_beta]
    new_z = [gamma * lam * z + x for z, x in zip(eligibility, features)]
    new_weights = [w + a * delta * z
                   for w, a, z in zip(weights, alpha, new_z)]
    new_history = [h * max(0.0, 1.0 - a * x * z) + a * delta * z
                   for h, a, x, z in zip(history, alpha, features, new_z)]
    return new_weights, new_beta, new_history, new_z, delta
```

<a id="lesson-metatrace"></a>

## 5. Metatrace：过去的哪些步长影响了今天的控制误差

TIDBD 主要从一步预测误差出发。带资格迹的 actor–critic 则把一个控制目标的信用分散到许多时刻：今天的 TD 误差会更新过去状态的价值与动作概率。Metatrace（Young、Wang、Taylor，IJCAI 2019）相应地为步长建立一条元资格迹。区别不只是把 TIDBD 接上 actor，而是外层目标包含各时刻实际使用的不同权重 $w_k$，因此要保留各自的敏感度 $h_k$。

$$
\mathcal J=\tfrac12\sum_k\left[\bigl(\operatorname{sg}(G_k^\lambda)-V_{w_k}(S_k)\bigr)^2-\log\pi_{w_k}(A_k\mid S_k)\operatorname{sg}\bigl(G_k^\lambda-V_{w_k}(S_k)\bigr)\right]
$$

这是用于推导的时变权重半梯度目标：critic 拟合多步 return，actor 使用固定的优势评价。在线 return 写成 $G_k^\lambda=V_{w_k}(S_k)+\sum_{t\ge k}(\gamma\lambda)^{t-k}\delta_t$；求导时按 $\operatorname{sg}$ 约定停止目标路径。权重梯度因此组合为下面的 $\nabla U$。

$$
\begin{aligned}U_w(s,a)&=V_w(s)+\tfrac12\log\pi_w(a\mid s),\\g_t&=\nabla_w U_{w_t}(S_t,A_t),\qquad q=\gamma\lambda,\\z_t&=qz_{t-1}+g_t,\qquad \delta_t=R_{t+1}+\gamma V_{w_t}(S_{t+1})-V_{w_t}(S_t),\\w_{t+1}&=w_t+\alpha_t\delta_tz_t.\end{aligned}
$$

$U$ 把原文选择的 critic 与 actor 权重合并；$1/2$ 来自其联合目标的系数，并非任意 actor–critic 都必须采用的常数。$z$ 是权重更新的资格迹。目标策略随权重变化，以下元导数仍在给定采样路径下计算。

设 $h_k\approx\partial w_k/\partial\beta$，$\alpha=\exp\beta$。对时变权重的多步半梯度目标应用链式法则，再交换时间求和，当前 TD 误差应乘过去各时刻 $g_k^\top h_k$ 的衰减和。这导出第二条迹 $z_{\beta,t}$。如果错误地使用 $z_t^\top h_t$，就把今天的敏感度替换到了过去所有时刻，改变了目标的时间结构。

$$
\begin{aligned}z_{\beta,t}&=\sum_{k=0}^{t}q^{t-k}g_k^\top h_k=qz_{\beta,t-1}+g_t^\top h_t,\\\widetilde\nabla_\beta\mathcal J&=-\sum_t\delta_t z_{\beta,t},\qquad \beta^+=\beta+\mu\delta_tz_{\beta,t},\\h_{t+1}&\approx h_t+e^{\beta^+}z_t\left[\delta_t+\left(\gamma\nabla V_{w_t}(S_{t+1})-\nabla V_{w_t}(S_t)\right)^\top h_t\right].\end{aligned}
$$

外层目标对 return 采取半梯度约定；敏感度递推却使用 $\delta$ 对权重的完整导数，因为它要追踪实际更新怎样依赖步长。递推省略了 $\partial z_t/\partial w$，也未计算动作分布改变的采样路径。即使 critic 线性，策略对数概率一般仍非线性，因而整体不是精确超梯度。

**算法：无熵项、无归一化的标量步长 Metatrace 核心**

1. 初始化 $w$、$\beta$；$z=0$，$h=0$，$z_\beta=0$。
1. 每个转移，用旧 $w$ 计算 $\delta$、$g=\nabla U$ 和 $d=\gamma\nabla V'-\nabla V$。
  1. $z\leftarrow qz+g$
  1. $z_\beta\leftarrow qz_\beta+g^\top h$        # 此时 h 仍是旧敏感度
  1. $\beta\leftarrow\beta+\mu\delta z_\beta$
  1. $\alpha\leftarrow\exp(\beta)$
  1. $h_{\rm new}\leftarrow h+\alpha z(\delta+d^\top h)$
  1. $w\leftarrow w+\alpha\delta z$；$h\leftarrow h_{\rm new}$
1. 真正 episode 结束后清空 $z,z_\beta$；保留 $w,\beta,h$。

手算两步：令 $q=0.5$、$g_0=g_1=1$、$h_0=0$、$h_1=0.1$。第二步 $z_1=1.5$，但 $z_{\beta,1}=0.5\times0+1\times0.1=0.1$，不是 $z_1h_1=0.15$。差别在于第一步发生时敏感度仍为零。延长资格迹不会创造过去并不存在的步长影响。

原文还给出逐参数版本和混合版本：逐参数版本保存对角敏感度，混合版本以 $\alpha_i=\exp(\widehat\beta+\beta_i)$ 结合全局与局部适应。实用算法先把元增量除以运行幅度估计，再限制有效步长；若有熵正则 $\psi\mathcal H(\pi)$，元增量中还包括 $\psi\nabla\mathcal H^\top h$。这些状态必须与权重资格迹分开保存。

$$
\begin{aligned}\Delta_\beta&=\delta z_\beta+\psi\nabla\mathcal H^\top h,\\v&\leftarrow\max\{|\Delta_\beta|,\ v+\mu(|\Delta_\beta|-v)\},\\\beta&\leftarrow\beta+\mu\Delta_\beta/\widetilde v,\qquad \widetilde v=\begin{cases}v&v>0\\1&v=0,\end{cases}\\u&\leftarrow\max\{e^\beta\|g\|^2,\ u+(1-q)(e^\beta\|g\|^2-u)\},\\\beta&\leftarrow\beta-\log\max(u,1).\end{aligned}
$$

这是原文标量归一化的结构。末行只有在有效步长估计超过 1 时才缩小 $\alpha$，不是对任意非线性网络误差的全局收缩定理。加入熵正则时，权重和敏感度更新也需要对应熵项。

$$
\begin{aligned}w^+&=w+\alpha(z\delta+\psi\nabla\mathcal H),\\h^+&\approx h+\alpha\left[z(\delta+\nabla\delta^\top h)+\psi\nabla\mathcal H\right].\end{aligned}
$$

这里 $\alpha$ 使用归一化后的新对数步长。敏感度保留熵更新对步长的直接依赖，但与资格迹的高阶导数一样，省略熵梯度对权重的 Hessian 项。

各状态的生命周期不同。完整标量算法在训练开始时初始化 $h=0$、$v=0$ 和给定的 $\beta_0$；每个 episode 开始时将 $z,z_\beta,u$ 清零，保留权重 $w$、步长参数 $\beta$、敏感度 $h$ 和元增量幅度估计 $v$。其中 $u$ 约束当前资格迹涉及的状态，episode 结束后这些状态不再进入新轨迹；$h$ 则追踪仍被保留的权重如何依赖步长，不应随资格迹一起清空。下面的无归一化代码没有 $u,v$，但保持相同的 $h$ 生命周期。

线性 critic 特例的完整核心；固定步长时敏感度可与有限差分逐项比较，在线更新步长时使用局部近似。

```python
@dataclass
class LinearMetatrace:
    """Scalar, critic-only, unnormalized Metatrace core (no entropy term).

    U(w,x)=V(w,x)=w*x. For mu=0 and fixed data, h is exact dw/d(log alpha).
    With online beta updates, h is the local sensitivity approximation.
    """
    w: float = 0.0
    beta: float = math.log(0.1)
    mu: float = 0.01
    gamma: float = 0.9
    lam: float = 0.8
    z: float = 0.0
    h: float = 0.0
    z_beta: float = 0.0

    def step(self, x, reward, next_x, terminal=False):
        gamma = 0.0 if terminal else self.gamma
        delta = reward + gamma * self.w * next_x - self.w * x
        # Old h belongs to the weights that generated the current prediction.
        old_h = self.h
        self.z = self.gamma * self.lam * self.z + x
        self.z_beta = self.gamma * self.lam * self.z_beta + x * old_h
        self.beta += self.mu * delta * self.z_beta
        alpha = math.exp(self.beta)
        delta_derivative = gamma * next_x - x
        self.h = old_h + alpha * self.z * (delta + delta_derivative * old_h)
        self.w += alpha * delta * self.z
        result = dict(w=self.w, h=self.h, alpha=alpha, delta=delta,
                      z=self.z, z_beta=self.z_beta)
        if terminal:
            # Genuine episodic boundary; parameters and step-size persist.
            self.z = self.z_beta = 0.0
        return result
```

<a id="lesson-meta-rl"></a>

## 6. Meta-gradient RL：根据后续表现选择学习目标

discount 和 $\lambda$ 影响学习目标的时间尺度与偏差—方差权衡。Meta-gradient RL 将这些选择作为内层学习的元参数 $\eta$：先用一段经验更新 agent，再用后续经验和固定评价约定计算元损失。外层并不知道一个监督标签意义上的“正确 $\gamma$”，而是评价由该 $\gamma$ 产生的参数更新。

$$
G_t^\lambda=R_{t+1}+\gamma\left[(1-\lambda)v_w(S_{t+1})+\lambda G_{t+1}^\lambda\right]
$$

在有限 rollout 内从末端向前计算，末端 $G$ 由 bootstrap 值或真正 terminal 的零值初始化。固定 $w$、轨迹与末端值，可以单独推导 return 对超参数的直接导数。

$$
\begin{aligned}\frac{\partial G_t^\lambda}{\partial\gamma}&=(1-\lambda)v_{t+1}+\lambda G_{t+1}^\lambda+\gamma\lambda\frac{\partial G_{t+1}^\lambda}{\partial\gamma},\\\frac{\partial G_t^\lambda}{\partial\lambda}&=\gamma\left(G_{t+1}^\lambda-v_{t+1}+\lambda\frac{\partial G_{t+1}^\lambda}{\partial\lambda}\right).\end{aligned}
$$

改变 $\gamma$ 或 $\lambda$，既影响当前组合，也影响后续 return。若用 sigmoid 将它们限制在 $(0,1)$，链式法则还需乘 sigmoid 导数。若末端值依赖 $\eta$，边界导数也必须保留。

将 return 导数代入 actor/critic 的更新函数 $f$，即可得到直接导数 $\partial f/\partial\eta$，再经敏感度递推影响后续参数。原 Meta-gradient RL 使用衰减 trace 降低成本：把完整的 Jacobian 传播替换为一个标量衰减。

$$
\begin{aligned}H^+&=\left(I+\frac{\partial f}{\partial w}\right)H+\frac{\partial f}{\partial\eta},\\z^+&=\mu z+\frac{\partial f}{\partial\eta},\qquad\eta^+=\eta-\beta\,z^{+\top}\nabla_{w^+}J_{\rm eval}.\end{aligned}
$$

第二行以标量衰减近似完整 Jacobian 传播；$\mu=0$ 只考虑当前更新。这一节的 $z$ 是元敏感度近似，不是前文的权重资格迹。在线变化的 $\eta$ 又引入局部近似，因此它不等于固定元参数下的全历史总导数。

**算法：算法伪代码**

1. 保存 agent 参数 $w$、元参数 $\eta$ 和元敏感度 $z$。
1. 内层 rollout：
  1. 用 $\eta$ 构造 return，计算更新 $f$。
  1. 计算 $\partial f/\partial\eta$，推进完整 $H$ 或近似 $z$。
  1. $w^+\leftarrow w+f$
1. 后续评价 rollout：
  1. 用固定 $\eta_{\rm eval}$ 计算 $J_{\rm eval}$。
  1. 沿 $w^+$ 对 $\eta$ 的影响更新元参数。
1. 继续交互；元梯度截断与状态重置按预先声明的协议执行。

相邻 rollout 通常不是真正独立样本，验证经验也受策略变化影响。研究时需要报告数据复用、是否用新策略采样、元梯度截断长度，以及是否将策略采样路径计入估计。训练误差降低本身不足以说明学习规则更适合长期控制。

$\lambda$-return 对 $\gamma$ 和 $\lambda$ 的直接导数：固定边界值，执行后向递推，并由有限差分检验。

```python
def lambda_return_sensitivity(rewards, next_values, bootstrap, gamma=0.9, lam=0.8):
    """Lambda return and exact direct derivatives for fixed values/boundary."""
    if len(rewards) != len(next_values):
        raise ValueError("one next-state value is required per reward")
    g, dg, dl = bootstrap, 0.0, 0.0
    for reward, value in reversed(list(zip(rewards, next_values))):
        mixture = (1.0 - lam) * value + lam * g
        new_dg = mixture + gamma * lam * dg
        new_dl = gamma * (g - value + lam * dl)
        g = reward + gamma * mixture
        dg, dl = new_dg, new_dl
    return g, dg, dl
```

<a id="lesson-output-steps"></a>

## 7. 在线元梯度与预设更新规则的边界

步长随数据变化，不足以判断一个方法是否在学习更新规则。IDBD、TIDBD 和 Metatrace 保存参数对过去步长的敏感度，用后续误差评价这种影响，再改变步长参数。Stream-X 与 Intentional Updates 则按预先规定的归一化或输出变化公式计算本次更新，不估计这条元敏感度。它们属于流式更新稳定化方法，可以作为元学习的对照或与之组合。

| 机制 | 保留的历史 | 本次更新回答的问题 |
| --- | --- | --- |
| 资格迹 | 过去输出梯度的衰减统计 | 当前误差应分给哪些过去预测或决策？ |
| Stream-X / Intentional 的尺度控制 | 梯度、误差或完整增量的幅度统计 | 在给定方向上，本次参数或输出允许改变多少？ |
| 在线元梯度 | 内层参数对学习规则参数的敏感度 | 过去的规则选择怎样影响了后续误差，应如何调整？ |

流式学习章完整推导 ObGD、StreamingOptimizer 与 Intentional-AC，并保留实现、手算及边界检查。时间信用分配章讨论误差在时间上的分配。这里继续研究学习规则本身的适应；这三个问题可以在同一个 agent 中共同出现，但不是同一种算法分类。

<a id="lesson-maml"></a>

## 8. MAML：学习一个适合少量更新的初始化

MAML 换了问题设定：从任务分布 $p(T)$ 采样任务，每个任务允许收集少量适应经验，再评价适应后策略。外层变量是共享初始化 $\theta$。每个任务从同一初值复制出 $w_i$，执行若干梯度更新；外层评价这些更新后的表现。直接平均各任务未适应的损失，则是不同的多任务学习目标。

$$
\begin{aligned}w_i&=\theta-\alpha\nabla_\theta L_i^{\rm train}(\theta),\\\mathcal J(\theta)&=\mathbb E_{T_i\sim p(T)}\left[L_i^{\rm test}(w_i)\right],\\\nabla_\theta\mathcal J_i&=\left(I-\alpha\nabla^2_\theta L_i^{\rm train}\right)^\top\nabla_{w_i}L_i^{\rm test}.\end{aligned}
$$

train/test 在这里指同一任务内的适应与评价数据，不是把最终测试任务反复拿来调外层参数。两阶段数据区分能避免只优化对同一批样本的过拟合。

多步版本重复 $w^{k+1}=w^k-\alpha\nabla L^k$，元梯度经过每一步 Jacobian 的乘积。FOMAML 将曲率修正近似为单位矩阵，仍然使用内层适应后的评价梯度。Reptile 则令 $\theta\leftarrow\theta+\varepsilon(w_i-\theta)$，将初始化向任务内训练后的参数移动；两者都避免显式二阶导数，但并非同一更新规则。

**算法：算法伪代码**

1. 每轮外层更新：
  1. 从训练任务分布采样任务 $T_i$。
  1. 对每个任务复制初始化：$w_i\leftarrow\theta$。
    1. 收集适应数据 $D_{\rm train}$，估计策略梯度损失。
    1. 执行 $K$ 次内层更新，保留计算图。
    1. 用适应后的策略收集 $D_{\rm test}$，计算外层损失。
  1. 合并外层梯度，沿内层更新反传到 $\theta$。
1. 测试新任务：从 $\theta$ 出发，按相同适应预算更新 $w$，不更新 $\theta$。

在 RL 中，对策略梯度 surrogate 再做自动微分，并不自动得到期望元回报的正确高阶导数。适应轨迹的分布依赖 $\theta$，评价轨迹的分布依赖 $w_i$；数值相同的 surrogate 可能具有不同的高阶导数。因而需要明确 likelihood-ratio、baseline 和 stop-gradient 的位置。二次函数实验可以隔离优化器链式法则，下面的随机例子再分析采样路径。

解析二次任务：完整 MAML 与 FO 近似的数值差别可直接检查。

```python
def scalar_maml(initialization, train_target, test_target, curvature=2.0, alpha=0.1):
    """One task, deterministic quadratic losses, exact MAML and FO approximation.

    L_train = curvature/2*(w-train_target)^2; L_test = (w-test_target)^2/2.
    This illustrates optimizer differentiation, not policy-gradient sampling.
    """
    adapted = initialization - alpha * curvature * (initialization - train_target)
    outer_gradient = adapted - test_target
    exact = (1.0 - alpha * curvature) * outer_gradient
    first_order = outer_gradient
    return 0.5 * outer_gradient ** 2, exact, first_order, adapted
```

$$
\nabla_\theta\mathbb E_{D\sim p_\theta}[J(U_\theta(D))]=\mathbb E_D\left[\left.\nabla_\theta J(U_\theta(D))\right|_D+J(U_\theta(D))\nabla_\theta\log p_\theta(D)\right]
$$

第一项在给定适应数据下求导，第二项反映初值改变了适应数据的分布。若 $J$ 还通过新策略采样估计，它自身也需要正确的策略梯度；sampled reward 的数值通常没有可直接反传的动作导数。

下面穷举一个 Bernoulli 动作，比较只保留更新路径与同时保留采样 score 项。由于动作只有两种，期望可以精确求和，有限差分不含 Monte Carlo 噪声。DiCE 等工作进一步处理随机计算图的高阶梯度；它们解决的是估计器问题，而不是替代 MAML 的内外层目标。

固定数据微分遗漏了什么：用一个能精确穷举的随机适应过程检查两条梯度路径。

```python
def stochastic_adaptation(theta, alpha=0.2, target=1.0):
    """Enumerate a Bernoulli sample's TWO gradient paths exactly.

    a~Bernoulli(sigmoid(theta)); w=(1-alpha)*theta+alpha*a.
    The toy outer loss is (w-target)^2/2. No Monte Carlo noise is present.
    """
    probability = 1.0 / (1.0 + math.exp(-theta))
    expected_loss = path_gradient = score_gradient = 0.0
    for action, mass in [(0.0, 1.0 - probability), (1.0, probability)]:
        adapted = (1.0 - alpha) * theta + alpha * action
        loss = 0.5 * (adapted - target) ** 2
        expected_loss += mass * loss
        path_gradient += mass * (adapted - target) * (1.0 - alpha)
        score_gradient += mass * loss * (action - probability)
    return expected_loss, path_gradient, score_gradient
```

<a id="lesson-context"></a>

## 9. RL² 与 PEARL：在活动状态中进行任务推断

另一种办法是把“学到了哪个任务”放进活动状态或潜变量，而把推断规则留在固定网络权重里。RL² 将前一动作、奖励和 episode 结束信号与当前观测输入 RNN；内层适应发生在 hidden state 的变化，外层用 RL 训练整条 trial 的累计表现。若一个 trial 包含同一任务的多个 episode，hidden state 在 episode 间保留，在新的任务 trial 开始才重置；这是任务协议的一部分。

$$
h_{t+1}=f_\theta(h_t,O_{t+1},A_t,R_{t+1},d_{t+1}),\qquad A_{t+1}\sim\pi_\theta(\cdot\mid h_{t+1})
$$

测试时可以冻结 $\theta$，而 $h$ 仍随经验改变。活动变化支持探索、记忆和适应，也可以与参数学习组合；两种变化应分别记录。

**算法：算法伪代码**

1. RL² 元训练：
  1. 采样训练任务，初始化 trial 活动状态 $h$。
  1. 在同一任务运行多个 episode；环境重置时保留跨 episode 的 $h$。
  1. 将已到达的动作、奖励和 done 输入 RNN，累计 trial 回报。
  1. 用序列梯度更新共享参数 $\theta$。
1. 测试：冻结 $\theta$；新任务 trial 才初始化 $h$，适应协议与训练一致。

PEARL 引入任务潜变量 $z$，从转移集合 $c$ 推断 $q_\phi(z\mid c)$，策略和价值函数均以 $z$ 为条件。其 context encoder 将各条转移编码成高斯因子后相乘，因此不依赖 context 内转移的排列。这适合任务在 context 范围内固定的设定；混合变化前后的证据，可能推断出并不存在的折中任务。

$$
q_\phi(z\mid c)\propto\prod_{\xi\in c}\Psi_\phi(z\mid\xi),\qquad \sigma^{-2}=\sum_j\sigma_j^{-2},\qquad \mu=\sigma^2\sum_j\frac{\mu_j}{\sigma_j^2}
$$

右式是单坐标高斯因子乘积，其他坐标可逐项计算；作者实现还通过 KL 正则将编码分布约束到先验附近。不要在已有因子定义之外又无说明地重复乘同一个先验。

$$
\begin{aligned}L_Q&=\mathbb E\left[\left(Q_\omega(s,a,z)-\operatorname{sg}(y)\right)^2\right],\\y&=r+\gamma\,\bar V(s',z),\\L_{\rm enc}&=L_Q+\beta_{\rm KL}\,D_{\rm KL}\bigl(q_\phi(z\mid c)\Vert p(z)\bigr).\end{aligned}
$$

原实现使用 SAC 的 $V$ 网络版本：慢更新网络提供目标价值，terminal 时停止 bootstrap。编码器由 critic 误差与 KL 训练；原作者代码在 policy 通路对 $z$ detach，使该通路不直接更新 encoder。

**算法：算法伪代码**

1. PEARL 元训练：
  1. 为训练任务维护 replay；从 context 推断 $q_\phi(z\mid c)$ 并采样 $z$。
  1. 用 $\pi(a\mid s,z)$ 交互，将经验加入该任务 replay/context。
  1. 采样 critic 训练转移和 context，生成可重参数化的 $z$。
  1. 更新 $Q$ 与 encoder（包含 KL）；更新条件策略和价值/target 网络。
1. 元测试：冻结网络参数，从先验采样 $z$，交互收集 context。
  1. 重新推断 posterior 并采样 $z$；不输入真实任务参数。

解析例子采用 $y=z+\epsilon$，先验 $z\sim\mathcal N(0,1)$，独立噪声方差为 1。没有证据时后验仍为先验；一条 $y=1$ 后为 $\mathcal N(0.5,0.5)$；三条相同观测后为 $\mathcal N(0.75,0.25)$。即使模型参数不变，条件推断仍改变预测。这说明 context adaptation 的性质；PEARL 另外需要学习推断网络及控制策略。

已知高斯任务的可运行 context 推断；明确展示先验与观测因子的不同角色。

```python
def gaussian_context(observations, noise_variance=1.0, prior_mean=0.0,
                     prior_variance=1.0):
    """Exact Bayesian latent-task inference for y_i=z+Gaussian noise.

    It demonstrates context-driven adaptation with NO parameter gradient.
    It is not PEARL's learned encoder or SAC training loop.
    """
    if noise_variance <= 0 or prior_variance <= 0:
        raise ValueError("variances must be positive")
    precision = 1.0 / prior_variance + len(observations) / noise_variance
    natural_mean = prior_mean / prior_variance + sum(observations) / noise_variance
    return natural_mean / precision, 1.0 / precision
```

<a id="lesson-learned-rules"></a>

## 10. 学习完整更新规则：LPG、DiscoRL 与外层优化

单一标量步长在给定方向下缩放更新，逐参数步长还会改变更新的几何方向。更一般的 learned update rule 让元网络读取奖励、动作、agent 输出、预测误差或训练统计量，产生目标、损失系数或更新信号。agent 参数 $w$ 按内层流程变化，元网络 $\eta$ 根据训练轨迹的外层表现优化。状态构造、预测目标与学习规则可以联合学习，但分别具有自己的状态和梯度路径。

$$
\begin{aligned}(y_t^{\rm target},m_{t+1})&=U_\eta(\xi_{t:t+k},p_{w_t},m_t),\\w_{t+1}&=w_t-\alpha\nabla_w\ell\bigl(p_w,y_t^{\rm target}\bigr),\\\eta^+&=\eta-\beta\nabla_\eta\,\mathcal J(w_{t+K}).\end{aligned}
$$

这是一个通用接口，具体方法的损失仍有差异。$m$ 为元学习器活动状态，$p_w$ 为 agent 的策略或辅助预测。target 对内层 $w$ 和外层 $\eta$ 的导数可以采用不同约定；整块 detach 会同时删除外层希望学习的通路。

**算法：算法伪代码**

1. 元训练：
  1. 采样训练环境与 agent 初值，初始化 learner state。
  1. 重复内层更新：
    1. 采集交互数据，运行 $U_\eta$，得到目标或损失。
    1. 更新 $w$，推进 optimizer 与 meta state。
  1. 用外部回报目标评价训练过程，按指定估计器更新 $\eta$。
1. 部署固定规则：冻结 $\eta$，新 agent 仍按规则持续更新 $w$。
1. 若部署时也更新 $\eta$，另行定义其数据、目标和预算。

DiscoRL（Nature 2025）延续 LPG 的可学习更新规则路线，并扩大算法发现的训练环境集合。其规则不仅选择步长，还利用 agent 的预测输出、TD/advantage 统计量和元网络状态产生训练信号。内层 agent 不断学习，外层改进产生这些信号的规则。评价时需要分别讨论规则在未见环境上的迁移，以及单个 agent 在长时间、非平稳交互中的持续学习；这是两个不同的泛化轴。

这类方法的关键风险是外层训练分布与预算：规则可能记住特定奖励尺度、时间范围、终止模式或架构。除了新环境回报，还应测试更长生命、未通知变化、不同观测尺度、优化器状态持续保留等场景。一个规则在训练 1000 步后很好，不代表继续运行 100 万步还稳定。

<a id="lesson-frontier"></a>

## 11. 前沿问题：元目标短视、规则表示与发现预算

第一条问题是元目标的时间范围。只沿 $K$ 步内层更新求导，容易偏好立即降低误差的更新，而忽视更久以后的表示学习或探索。Bootstrapped Meta-Learning（ICLR 2022）在可微展开得到 $w_K$ 后，执行额外内层更新以产生未来目标 $\widehat w$，再让 $w_K$ 在选定的距离下接近这个目标。计算图仍只穿过前 $K$ 步，目标分支停止梯度；额外未来学习需要计算与数据，但不必保存同等长度的反传图。

$$
w_K=F_\eta^{(K)}(w_0),\qquad \widehat w=\operatorname{sg}\bigl(T^{(L)}(w_K)\bigr),\qquad J_{\rm BMG}=D\bigl(\widehat w,w_K\bigr)
$$

$T^{(L)}$ 表示构造目标的额外更新，可与 $F_\eta$ 不同；$D$ 可以是参数距离，也可以是策略分布距离。这样改变的是外层目标的几何与监督范围；梯度仍只穿过前 $K$ 步更新，不穿过未来目标分支。目标如果自身很差，匹配它也未必有益。

第二条问题是外层应该怎样搜索规则。RLC 2025 的 How Should We Meta-Learn Reinforcement Learning Algorithms? 比较黑盒进化学习、神经蒸馏、符号蒸馏和 LLM 代码发现等方法，涉及更新规则、优化器和策略损失等不同对象。其意义在于把“规则的表示形式”与“搜索规则的方法”拆开：可解释公式不一定必须由人工写出，神经更新规则也不一定必须靠元梯度训练。

| 研究条线 | 最小实验设计 | 应回答的机制问题 |
| --- | --- | --- |
| 在线元梯度：IDBD → TIDBD → Metatrace | 固定特征/小网络，比较完整低维敏感度、对角近似和元资格迹；再引入未通知漂移。 | 收益来自步长幅度、跨时间信用，还是额外归一化？同等内存下能否保留？ |
| 短视与目标几何：meta-gradient → BMG | 固定内层算法与总交互，分别增加反传窗口和停止梯度的目标生成窗口。 | 更多未来信息与更多反传计算哪一个更重要？目标滞后造成什么偏差？ |
| 规则发现：LPG → DiscoRL；RLC 2025 方法比较 | 把元训练环境、规则参数量、种子和外层预算独立控制，保留未见任务与长生命测试。 | 学到的是可迁移更新机制，还是某组训练环境/奖励尺度/训练长度的特化？ |

这些实验不应只比较最后一个 checkpoint 的分数。需要同时记录全过程回报、漂移后恢复曲线、参数与步长范数、资格迹统计、每步延迟、持久内存和外层总预算。对算法发现尤其要区分：发现一次规则的成本、训练一个新 agent 的成本，以及部署期间是否继续改规则。原始代码能帮助恢复这些训练协议，不能用单个优化器函数替代整个实验定义。

<a id="lesson-example"></a>

## 12. 两个可手算的元学习过程

算例 A：$w_0=0$，两个训练标签都是 2，$\alpha=0.1$，$\beta=\log(0.1)$，外层标签为 3。内层得到 $w_1=0.2$、$w_2=0.38$。由 $H_0=0$，先有 $H_1=0.2$，再有 $H_2=0.9\times0.2+0.1\times1.8=0.36$。外层损失为 $(0.38-3)^2/2=3.4322$，元梯度为 $(0.38-3)\times0.36=-0.9432$。梯度下降因而增大 $\beta$，使后续学习更快地向正目标移动。

| 两步超梯度方案 | 第二步敏感度 $H_2$ | 外层对 $\beta$ 的梯度 | 改变了什么 |
| --- | --- | --- | --- |
| 精确固定 $\beta$ 多步递推 | 0.36 | −0.9432 | 传播了第一次更新经过第二次更新的影响。 |
| 只保留当前更新，H 从零开始 | 0.18 | −0.4716 | 完全删掉过去影响。 |
| 忽略曲率、直接累计 | 0.38 | −0.9956 | 保留过去，但漏掉第二次更新的 0.9 收缩。 |

算例 B：MAML 初值 $\theta=0$，内层损失 $(w-2)^2$，步长 0.1，外层损失 $(w-3)^2/2$。内层曲率为 2，适应后 $w=0.4$。完整元梯度为 $(1-0.1\times2)(0.4-3)=-2.08$；FOMAML 为 $-2.6$。二者在这个例子中同向但不等；曲率更大时连符号也可能不同。有限差分应扰动初值 $\theta$，再重新执行内层更新。

若评价目标自身可以通过缩短时间尺度改变，元学习器有可能把 $\gamma$ 推向零，以回避长时误差。这说明评价目标应先于训练代理目标定义；降低一个可变目标的数值，不直接意味着改善了原定任务。

<a id="lesson-code"></a>

## 13. 可运行代码与控制变量实验

脚本只依赖 Python 3.10+ 标准库；直接运行即可打印全部手算对应值。

```sh
python3 state_meta_lab.py meta
python3 state_meta_lab.py test
```

基础实验依次给出两步超梯度、IDBD/TIDBD、MAML 与 FO、随机采样的额外梯度项、$\lambda$-return 导数和 context 后验。固定数据的实验用有限差分检验导数；随机例子穷举全部动作以检验期望的两条求导路径。TD 退化关系与边界测试检查更新时序。这里采用低维解析问题，便于观察每一个内部量。

Metatrace 机制实验与配套回归测试；标准库即可运行

```sh
python3 meta_frontier_lab.py metatrace
python3 meta_frontier_lab.py test
```

进阶脚本中的 Metatrace 是无熵正则、无归一化的线性 critic 特例，用来检查元信用的时间索引和敏感度传播。它不包含完整神经网络与环境。复现原论文回报，需要使用作者训练入口、网络、环境版本与统计协议。配套脚本中的 IntentionalStep 属于流式章的输出尺度实验，不作为本章的元学习方法。

完整机制实验驱动代码：每一种“适应”都有独立可观察的对象。

```python
def meta_demo():
    loss, grad, trace = fixed_beta_training(math.log(0.1), [2.0, 2.0], 3.0)
    print("two inner steps (w,H):", trace)
    print("validation loss, d/dbeta:", loss, grad)
    w, b, h = [0.0], [math.log(0.1)], [0.0]
    for target in [2.0, 2.0, -2.0]:
        w, b, h, delta = idbd_step(w, b, h, [1.0], target)
        print("IDBD: target,w,alpha,H:", target, w[0], math.exp(b[0]), h[0])
    result = tidbd_step([0.0, 0.0], [math.log(0.1)] * 2, [0.0] * 2,
                       [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], 1.0)
    print("TIDBD: weights, traces:", result[0], result[3])
    print("MAML: loss, exact, FO, adapted:", scalar_maml(0.0, 2.0, 3.0))
    print("stochastic adaptation loss,path,score:", stochastic_adaptation(0.3))
    print("lambda return, d/dgamma, d/dlambda:",
          lambda_return_sensitivity([1, 2], [0.3, 0.4], 0.4))
    for context in [[], [1.0], [1.0, 1.0, 1.0]]:
        print("context posterior (mean,var):", context, gaussian_context(context))
    print("scope: analytic mechanism tests, not meta-RL benchmark performance")
```

- 修改一：把两个训练标签改为 (2,−2)，保留外层标签 3；记录精确、多步截断与直接累计的符号和幅度，检查“连续梯度同向”的直觉何时失效。
- 修改二：改变 MAML 的内层曲率与步长，让 α×曲率大于 1；解释完整元梯度的符号为什么可能与 FO 相反。
- 修改三：将 context 后半部分的目标从 +1 改为 −1；累积全部证据与只使用近期证据的估计为何不同？这需要推断协议的改变，不是简单调大 actor 步长。
- 修改四：先用完全相同的数据核验导数，再开放策略影响采样。若性能改变，分别检查估计器偏差、探索差异与计算预算，不将它们混为元梯度精度。

<a id="lesson-branches"></a>

## 14. 算法条线与 CRL 的适用条件

| 路线 | 可继续推进的 CRL 问题 | 不能默认继承的假设 |
| --- | --- | --- |
| IDBD/TIDBD | 不同特征、预测问题和环境阶段需要不同适应速度。 | 线性、近似对角敏感度、采样与半梯度稳定条件；仍有元步长。 |
| Metatrace | 带资格迹的在线控制，显式追踪历史步长影响。 | 半梯度外层目标、忽略资格迹高阶导数和策略采样路径；标量/向量/混合变体不同。 |
| Meta-gradient RL | 在线适应回报尺度、λ、学习目标或更新系数。 | 外层评价可信、元梯度时序正确、不会靠改变目标投机。 |
| MAML/FOMAML | 过去任务是否能提供更易适应的参数起点？ | 独立任务采样、可回到共同初值、有清晰适应预算。 |
| RL²/PEARL | 上下文推断能否在固定参数下识别变化？ | 训练任务分布、context 内任务稳定、task/trial reset 权限。 |
| Learned update rules | 能否学到跨环境、跨生命长度的通用学习机制？ | 外层预算、分布外泛化、时间尺度与持久状态稳定性。 |
| 持续元学习 | 外层规则本身也要一生更新，且不能遗忘旧适应能力。 | 不能用无限保留过去所有训练轨迹作为默认实现。 |

这些方法处理不同的适应对象，没有单一的替代关系。上下文推断器可以使用在线自适应步长训练，预测网络的 GVF 问题参数可以接受元梯度，结构化 RNN 也可以成为 learned update rule 的记忆。组合方法时，需要重新定义内层与外层状态、数据来源和导数截断边界，以明确各模块之间传递哪些信息和梯度。

原始实现中，MAML-RL 的 maml_vpg.py 连接适应前后 surrogate 与内层计算图；PEARL 的 agent.py 实现 context、posterior 与潜变量采样，sac.py 组织 critic/encoder 更新；DiscoRL 的 disco.py 定义更新规则接口与持久元状态。检查这些接口后，再恢复配置中的环境、采样量和外层预算，才能比较完整算法。

<a id="research-trac-scale-adaptation"></a>

## 研究专题 A · TRAC：参考位移与多时间尺度的在线适应

IDBD 追踪步长如何影响后续误差；TRAC 的适应对象是参数相对参考点的位移尺度。它将基础优化器与一维在线 tuner 组合，让近期证据决定离初始化多远。参考点是正则化锚，不是已证明安全的策略。

$$
\theta_{t+1}=\theta_{\rm ref}+s_{t+1}(\theta^{\rm Base}_{t+1}-\theta_{\rm ref}),\quad s_{t+1}=\sum_j s_{t+1,j}
$$

论文 Algorithm 1 的参数化。只有 s∈[0,1] 才是凸组合，范围之外会外推；典型实测范围不能被写成普适约束。

$$
z_t=\langle g_t,\theta_t-\theta_{\rm ref}\rangle,\quad v_{t,j}=\beta_j^2v_{t-1,j}+z_t^2,\quad u_{t,j}=\beta_j u_{t-1,j}-z_t
$$

采用损失下降梯度约定。不同 β 保留不同时间范围；正内积表示增大位移的局部损失代价，负内积支持增大位移。回报上升梯度需改符号。

$$
s_{t+1,j}=\frac{\epsilon}{\operatorname{erfi}(1/\sqrt2)}\operatorname{erfi}\!\left(\frac{u_{t,j}}{\sqrt{2v_{t,j}}+\epsilon}\right)
$$

论文 tuner 的决策形式。稳定 erfi、初期尺度与裁剪都是实现条件；任意 sigmoid 不能替代后仍称原算法。

若参考点为 0，基础点为 2，尺度 0.2 时部署点为 0.4，尺度 1 时为 2。同一方向产生不同偏移。此时 $g=+1$ 表示继续增大位移局部有害，$g=-1$ 则相反。这个例子解释反馈符号，并不证明任意 RL 梯度下都选择最优正则。

原作者 trac.py 从已缩放的部署点重建未缩放位移，加入基础 optimizer 增量，再重新缩放；inner product 使用重建方向并含非负尺度处理。它与抽象 Algorithm 1 的参数化需分别核对。基础动量、参考参数和 tuner 统计都需跨日志分段保留，复现应固定代码版本。

- 先冻结梯度序列，打印基础/部署位移及各 u/v/s；检查零梯度、符号和状态保存恢复。
- 同一 Adam 与学习率下比较固定尺度、固定 L2、单 tuner 与多个 tuner；匹配 warm-start 和调参预算。
- 未通知变化后同时测适应、旧功能与初期代价。凸在线损失遗憾不等于非凸、策略依赖采样的深度 RL 终生回报保证。

<a id="research-rule-discovery-resources"></a>

## 研究专题 B · 算法表示、规则搜索与部署成本

RLC 2025 的 How Should We Meta-Learn Reinforcement Learning Algorithms? 将学习的组件与发现它的方法分开比较。更新规则可以是神经函数、符号公式或代码；它可以通过进化搜索、蒸馏、代码提案等过程产生。表示的可解释性不能代替发现预算与部署效果。

$$
\eta^*=\operatorname*{arg\,max}_{\eta\in\mathcal U}\mathbb E_{E\sim\mathcal D_{\rm train}}[J_H(\mathcal A_\eta,E)],\qquad J_H=\mathbb E[\sum_{t=0}^{H-1}R_{t+1}]
$$

统一比较接口，不是论文全部方法共用的具体损失。U 是规则类，H 是外层生命长度；输入权限、状态与规则容量改变都会改变 U。

| 层次 | 应固定或记录 | 混淆 |
| --- | --- | --- |
| 发现 | 训练环境、模拟步、搜索次数、设备时间 | 用更多搜索发现的规则却不报告搜索资源。 |
| 部署 | 每步延迟、规则网络与持久 meta-state | 只计 agent 权重，不计学习规则本身。 |
| 泛化 | 未见任务、奖励尺度、漂移与生命长度 | 看完整测试未来再选规则，称作在线适应。 |

最小研究可只学习 critic 更新中的一个小函数，固定 actor、网络、数据权限及内层预算，比较不同发现方式。封存规则后，测试十倍生命长度和未通知变化。这样能识别收益来自规则结构、搜索方法还是外部资源。作者 AlexGoldie/learn-rl-algorithms 按发现方法与评价流程组织实现。

**算法：算法发现比较流程**

1. 注册被替换组件、可读变量、训练环境与发现预算
1. 独立记录每种搜索方法的全部提案和失败
1. 封存规则及外层选择，不读取测试未来轨迹
1. 新 agent 按相同初始化运行，报告全程收益和部署成本

若规则本身也要在单生命内持续学习，还必须定义在线外层目标、旧/新经验分配、外层状态与内层 optimizer 的相容性。跨任务规则迁移是一个研究轴，部署智能体终生修改规则是另一个闭环，不能从前者直接宣称后者已解决。

<a id="lesson-check"></a>

## 15. 诊断、自测与研究起点

| 看到的症状 | 要检查的具体量 |
| --- | --- |
| 元参数不动 | H 是否被清零/截断；target 是否把 η 路径一起 detach；第一步 H=0 是否被误认为 bug。 |
| 步长突然爆炸 | β、α、特征尺度、元梯度范数、clip 触发频率；不能只画平滑后的 return。 |
| 测试适应特别快 | 是否拿到 task ID/边界、重置先验、测试轨迹参与外层更新；与基线权限是否相同。 |
| context 很自信却一直错 | 环境已变化、经验高度相关、后验模型失配；小方差不等于正确推断。 |
| 外层损失下降但控制无改善 | 评价目标是否与实际回报错位；是否只改善同批样本或代理目标。 |
| 精确梯度测试过了但 RL 效果差 | 采样路径、方差、截断偏差、外层过拟合、额外算力，逐项隔离。 |

- 问：Adam 是不是就等于本页的元学习？答：Adam 的统计量改变预设更新规则，但没有在这里所定义的外层后续目标上学习这些规则参数；“有自适应状态”不是充分定义。
- 问：测试时冻结权重就不可能 Meta-RL 适应吗？答：RL² 的活动和 PEARL 的 posterior 仍能变；必须指出变化发生在何处。
- 问：把 IDBD 的 y 换成 TD target 就已经完成 TIDBD 推导吗？答：没有，还要处理 target 导数约定与资格迹，特别是 β 用 x、内层更新用 z 的区别。
- 问：有限差分能证明 RL 元梯度无偏吗？答：只能验证给定可微程序的导数；轨迹分布的导数是否被正确估计，是另一个统计问题。

可开始的研究题：固定预测问题与 feature，先比较常数步长、IDBD/TIDBD 及精确低维敏感度在未通知变化下的终生预测误差，再扩展到 actor-critic。预算应同时限制环境交互和每步计算；变化检测只用于离线统计，不给智能体任务标签。记录变化前误差、恢复积分误差、长期稳定性、步长与 trace 轨迹，并保留元学习无益或有害的情形。这样得到的是可解释的机制研究，而不只是多一个可调超参数的学习曲线。

## 本章的实验设计

外层更新、超参数搜索和基线调参都消耗预算。先锁定开发程序，再用独立运行比较表现。

设定：先用两步标量学习核元敏感度，再选择在线步长学习或跨任务适应协议。两种设定分别声明重置、任务分布与测试权限。

- 完整元梯度与固定随机路径的有限差分一致。
- 第一步零历史敏感度不被误判为实现故障。
- 测试经验不得通过未声明外层更新泄漏到方法选择。

对照：调好的固定率、简单日程与元率；完整导数、截断和停止梯度；相同任务分布及额外计算对照

记录：元目标、敏感度、步长及实际参数写入；适应前后收益和所需数据；元训练、搜索与部署开销

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-selection)

## 学习与研究衔接

IDBD 适应步长，MAML 学初始化，context-based meta-RL 推断任务；内外层目标与数据权限不同。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-meta) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=meta) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=meta)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning](https://yingwen.io/zh/continual-rl/research/#recent-lambda-greedy)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Intentional Updates for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-intentional-updates)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline](https://yingwen.io/zh/continual-rl/research/#recent-reset-and-distill)
- [Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-fame-fast-meta-learners)
- [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-trac-online-regularization)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Step-size Optimization for Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-step-size-optimization)
- [Learning from experience instead of curated datasets](https://yingwen.io/zh/continual-rl/research/#recent-oak-network-idbd)
- [Discovering state-of-the-art reinforcement learning algorithms](https://yingwen.io/zh/continual-rl/research/#recent-disco-rl)
- [Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-fame-fast-meta-learners)
- [Intentional Updates for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-intentional-updates)
- [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-trac-online-regularization)
- [How Should We Meta-Learn Reinforcement Learning Algorithms?](https://yingwen.io/zh/continual-rl/research/#recent-meta-algorithm-search-comparison)
- [A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning](https://yingwen.io/zh/continual-rl/research/#recent-lambda-greedy)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [Position: Lifetime tuning is incompatible with continual reinforcement learning](https://yingwen.io/zh/continual-rl/research/#recent-lifetime-tuning)
- [How Should We Meta-Learn Reinforcement Learning Algorithms?](https://yingwen.io/zh/continual-rl/research/#recent-meta-algorithm-search-comparison)

### Intentional Updates for Streaming Reinforcement Learning

Arsalan Sharifnassab, Mohamed Elsayed, Kris De Asis, A. Rupam Mahmood, Richard S. Sutton

ICML 2026 · 2026 · 支持方法与理论

#### 研究问题

能否先规定本次更新应产生多大作用，再反推合适的参数更新尺度？

#### 关键机制

Intentional 方法以局部线性近似连接参数变化和预测变化。critic 以减少一定比例的 TD 误差为目标，actor 控制策略输出变化的局部代理量；再结合资格迹和逐坐标尺度，求出这一次更新的强度。这是有目标的局部更新控制，不是对长期表现求导的元梯度。

#### 证据

论文给出推导和流式控制比较，ICML 2026 正式论文入口与作者实现均可用。实现将优化器与 actor–critic 交互区分开，便于检查更新时序。

#### 条件与限制

Taylor 近似在大更新时可能失准。采样动作上的对数概率变化不等于精确的全分布 KL 上界；熵项与 TD 误差符号也必须按原算法处理。

#### 阅读与实验

在一次更新前后直接测量预测变化，并与线性估计比较。分别测试正、负 TD 误差和很小梯度的情形，不要只检查参数是否有限。

#### 原文与相关入口

- [ICML 2026 原文](https://proceedings.mlr.press/v306/sharifnassab26a.html)：正式会议版本与更新意图的定义。
- [作者实现](https://github.com/sharifnassab/Intentional_RL)：重点对照 optimizer.py 与 intentional_ac.py。

#### 作者代码

[原论文作者提供的实现。](https://github.com/sharifnassab/Intentional_RL)

Intentional 更新与流式 actor–critic。

### Step-size Optimization for Continual Learning

Thomas Degris, Khurram Javed, Arsalan Sharifnassab, Yuxin Liu, Richard S. Sutton

arXiv 预印本 · 2024 · 支持方法与理论

#### 研究问题

误差变大时，应该减小步长过滤噪声，还是增大步长追踪真实变化？

#### 关键机制

论文区分梯度归一化与步长优化。IDBD 类方法以 $\alpha_i=\exp(\beta_i)$ 保证步长为正，并用权重对过去步长的敏感度估计改变 $\beta_i$ 是否有利。持续学习中，静止的无关方向适合很小步长，而持续变化的有用方向需要保留追踪能力。

#### 证据

作者用权重翻转和带噪追踪等线性学习问题比较机制，显示相似的误差幅度可以要求相反的步长反应。

#### 条件与限制

这些可分析任务不是深度控制上的普适优越性证据。元步长、近似敏感度与输入尺度仍会影响结果；步长自适应并没有消除全部外部设计参数。

#### 阅读与实验

分别增加观测噪声和目标漂移速度，检查步长是否采取不同反应。若只记录平均误差，就看不到噪声过滤与追踪之间的区别。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2401.17401)：步长优化与归一化的对照实验。

### Discovering state-of-the-art reinforcement learning algorithms

Junhyuk Oh, Gregory Farquhar, Iurii Kemaev, Dan A. Calian, Matteo Hessel, Luisa Zintgraf, Satinder Singh, Hado van Hasselt, David Silver

Nature · 2025 · 支持方法与理论

#### 研究问题

除了学习策略，能否从大量学习过程里学出更有效的 RL 更新规则？

#### 关键机制

DiscoRL 用外层优化评价执行若干内层更新后的行为表现，学习价值、策略与辅助预测之间的更新方式。被训练的对象是学习算法本身，而不仅是某个任务的策略参数。内外两层有各自的数据、时间尺度与计算预算。

#### 证据

论文报告跨环境发现更新规则与迁移到未见环境的结果，并公开配套算法实现。它展示了自动算法发现的可能性，但依赖大规模外层训练。

#### 条件与限制

外层在大量环境和设备上的搜索属于设计者侧资源，不能记作测试智能体单次生命内的自主学习。公开规则的执行成本与发现该规则的成本应分别报告。

#### 阅读与实验

画出内层参数和外层参数的更新依赖，再列出测试时哪些量被冻结。与在线 IDBD 比较时，先区分跨任务算法发现和单流步长追踪。

#### 原文与相关入口

- [Nature 原文](https://www.nature.com/articles/s41586-025-09761-x)：算法发现过程、外层资源与泛化实验。
- [作者实现](https://github.com/google-deepmind/disco_rl)：配套代码与发现的更新规则。

#### 作者代码

[Google DeepMind 的论文配套仓库。](https://github.com/google-deepmind/disco_rl)

DiscoRL 配套实现与学习到的更新规则；具体训练资源以仓库说明为准。

### Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning

Ke Sun, Hongming Zhang, Jun Jin, Chao Gao, Xi Chen, Wulong Liu, Linglong Kong

ICLR 2026 · 2026 · 直接研究持续学习

#### 研究问题

快速学习新任务和整合旧知识，能否由不同学习器承担并以明确目标连接？

#### 关键机制

FAME 的快速学习器适应当前任务，元学习器整合此前知识。论文按旧策略的重要访问分布度量价值或策略变化，再据此构造减少遗忘的整合目标。自适应预热决定如何利用旧知识初始化或约束早期行为，以减少负迁移。

#### 证据

论文分析价值型和策略型版本，并在像素与连续控制任务序列中比较。作者提供官方实现，可追踪快速适应与知识整合两个阶段。

#### 条件与限制

设定要求相同状态与动作空间、已知任务边界以及额外整合计算。这里的 meta learner 主要是知识整合模块，不应因名称就当作通过长期回报反向求导的在线元梯度算法。脑机制类比也不是神经科学实验证据。

#### 阅读与实验

分别报告新任务前向迁移、旧任务保留和两个学习阶段的计算量。改变任务相似性，检验自适应预热是否确实避免有害旧知识。

#### 原文与相关入口

- [ICLR 2026 原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2230ffcd5da10015ce0c6ce588fc2936-Abstract-Conference.html)：任务边界假设、遗忘度量与快慢知识机制。
- [FAME 官方实现](https://github.com/datake/FAME)：论文链接的快速学习与知识整合代码。

#### 作者代码

[论文与仓库均注明为官方实现。](https://github.com/datake/FAME)

FAME 的价值型、策略型持续学习实验。

### Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline

Hongjoon Ahn, Jinu Hyeon, Youngmin Oh, Bosun Hwang, Taesup Moon

ICLR 2025 · 2025 · 直接研究持续学习

#### 研究问题

一个网络还能拟合新目标，为什么先前训练仍可能让它在新任务上学得更慢？

#### 关键机制

论文把任务之间的负迁移与一般可塑性损失区分开。Reset & Distill 在新任务开始时重置在线 actor 和 critic，避免旧初始化阻碍学习；随后离线蒸馏当前策略与旧专家的动作分布以整合知识。适应和保留通过不同过程实现。

#### 证据

作者在控制与游戏任务中分析负迁移，并在长 MetaWorld 序列上检验该基线。原文直接提供实现地址。

#### 条件与限制

任务边界、在线网络重置、旧专家和离线蒸馏都需要资源。它不能直接当作无边界、不能重置、禁止回放的单次生命方案。

#### 阅读与实验

除了与连续微调比较，还要与同等预算的从头训练比较。若新任务表现低于从头训练，先检查负迁移，再判断是否属于单纯容量损失。

#### 原文与相关入口

- [ICLR 2025 原文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ba9e3d60610f3525717665966d86e0cd-Abstract-Conference.html)：负迁移诊断、Reset & Distill 机制与边界。
- [原文代码入口](https://github.com/hongjoon0805/Reset-Distill)：论文首页提供的作者实现。

#### 作者代码

[ICLR 正式论文首页明确链接的代码。](https://github.com/hongjoon0805/Reset-Distill)

Reset & Distill 以及任务序列实验。

### Learning from experience instead of curated datasets

Oak Lab

Oak Lab 技术博文 · 2026 · 支持方法与理论

#### 研究问题

有用信号稀疏且大量输入是噪声时，在线学习规则如何分配不同方向的更新能力？

#### 关键机制

博文从含稀有有效特征的线性预测问题出发，对比统一步长与 IDBD 的逐权重适应，再展示 NetworkIDBD 在非线性带噪观测中的例子。核心主张是让长期学习效果影响信用和步长分配，而不只依据当前梯度幅度归一化。

#### 证据

公开页面提供受控噪声特征任务和 NoisyMNIST 示例。它们是机制演示，便于理解有效信号密度与输入规模的关系。

#### 条件与限制

该页面不是完整 CRL 控制论文，也未给出可直接复现所有图表的完整代码和算法推导。监督噪声任务的结果不能证明一般 SGD 或所有深度 RL 都无法从经验学习。

#### 阅读与实验

先复现线性噪声特征问题，分开改变有效特征稀疏度与噪声维数。进入控制前，再加入策略改变数据分布这一因素。

#### 原文与相关入口

- [Oak Lab 原始博文](https://oaklab.ai/posts/learning-from-experience-instead-of-curated-datasets)：2026 年 7 月 13 日；受控实验、NetworkIDBD 示例与研究动机。

### Position: Lifetime tuning is incompatible with continual reinforcement learning

Golnaz Mesbahi, Parham Mohammad Panahi, Olya Mastikhina, Steven Tang, Martha White, Adam White

ICML 2025 Position Paper · 2025 · 评价与实验协议

#### 研究问题

如果设计者用完整未来生命反复调参，实验还在测智能体面对未知变化的能力吗？

#### 关键机制

论文限制调参可访问的生命阶段，并比较这种选择方式与利用完整生命回报挑选配置的差异。外部设计者掌握未来变化信息，可能使一个并不自适应的固定算法显得适应良好。核心改变发生在评价协议，而不是 TD 更新公式。

#### 证据

作者用持续、非平稳设置中的深度 RL 实验说明超参数选择可以改变方法比较。该工作属于立场论文，论证与示例用于推动更符合问题目标的评价。

#### 条件与限制

允许多少开发阶段经验需要按应用规定，不存在由该论文推出的普适固定比例。仅限制时间前缀也不能替代独立测试种子和计算预算控制。

#### 阅读与实验

对同一配置集合分别按开发前缀和完整生命选择超参数，再在独立测试生命上比较。报告两种选择使用了哪些未来信息。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/mesbahi25a.html)：调参协议、论证与示例实验。

### Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning

Aneesh Muppidi, Zhiyu Zhang, Heng Yang

NeurIPS 2024 · 2024 · 直接研究持续学习

#### 研究问题

未知环境变化时间和速度时，怎样在线决定参数应离参考初始化多远？

#### 关键机制

TRAC 在基础优化器外维护一组具有不同遗忘时间尺度的一维 tuner，根据梯度与参考方向的内积调整参数位移尺度。它通过数据驱动的缩放联系到正则化，而不是对未来任务回报进行长窗口元梯度反传。

#### 证据

作者在 Procgen、Atari 与 Gym Control 变化序列中比较适应与可塑性，并分析在线凸优化对该设计的启发。

#### 条件与限制

凸在线优化中的遗憾理论不等于非凸、策略依赖采样的深度 RL 收敛定理。“parameter-free”不表示没有基础学习率、初始化、时间尺度网格、warm-start 或协议选择。

#### 阅读与实验

记录 tuner 尺度、距参考点的位移、旧分布干扰与变化后适应。用相同基础优化器比较固定尺度、单时间尺度和多时间尺度。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/5b76d77e7095c6480ed827b85f0c2878-Paper-Conference.pdf)：Algorithm 1–2、正则化联系与持续实验。
- [作者论文 v3](https://arxiv.org/html/2405.16642v3)：区分凸理论、RL 经验结果与初期表现限制。

#### 作者代码

[作者项目页与仓库均明确标为官方实现。](https://github.com/ComputationalRobotics/TRAC)

trac.py、PyTorch/JAX optimizer 包与控制/视觉实验。

### How Should We Meta-Learn Reinforcement Learning Algorithms?

Alexander David Goldie, Zilin Wang, Jaron Cohen, Jakob Foerster, Shimon Whiteson

RLC 2025 / RLJ · 2025 · 评价与实验协议

#### 研究问题

算法表示、发现算法的方法和测试智能体的学习成本，应该如何独立比较？

#### 关键机制

对 RL 流程的不同组件进行算法发现，比较黑盒学习、神经/符号蒸馏与 LLM 代码提案。学习器的表示形式和搜索过程分开定义，才能识别泛化、可解释性和成本之间的取舍。

#### 证据

论文直接比较元训练、元测试、样本成本、训练时间与可解释性，作者代码按发现方法和评价入口组织。

#### 条件与限制

跨环境发现规则主要发生在设计者侧，不等于运行智能体已能终生修改规则。论文的训练任务和预算范围不支持所有算法发现方法的普适排序。

#### 阅读与实验

固定被学习组件、输入权限、元训练数据和发现预算；封存规则后再检验未见环境、长生命与未通知漂移。

#### 原文与相关入口

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_218.pdf)：比较对象、元训练/测试与多维成本。
- [RLJ 论文记录](https://rlj.cs.umass.edu/2025/papers/Paper218.html)：作者与正式会议收录。

#### 作者代码

[论文提供、仓库标为官方的作者实现。](https://github.com/AlexGoldie/learn-rl-algorithms)

learning_algorithms 中各发现方法与独立 evaluation 流程。

### A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning

Martha White, Adam White

arXiv预印本 · 2016 · 支持方法与理论

#### 研究问题

不同状态的预测可靠性不同，固定λ是否浪费了多步信用？

#### 关键机制

将下一处bootstrap选择写成局部偏差平方与回报方差的折中，得到$λ=b^2/(b^2+\operatorname{Var}(G))$。完整λ-greedy还用在线预测器估计回报均值和二阶矩。

#### 证据

原文给出状态相关λ的目标、增量算法和多个预测设置的实验。信用章仅核对已知统计量下的局部最优与变量λ恒等式。

#### 条件与限制

局部贪心目标不是整条轨迹的联合最优。逼近误差、统计滞后和非平稳性会影响λ估计；辅助资源需要计入比较。

#### 阅读与实验

先让噪声方差变化，再让bootstrap可靠性变化。比较固定λ、已知统计参照和在线估计，分别观察目标偏差与适应速度。

#### 原文与相关入口

- [作者原文](https://arxiv.org/html/1607.00446)：局部目标、状态λ、均值／二阶矩预测与完整算法。


<a id="chapter-code"></a>

## 下载与运行

多步超梯度、IDBD/TIDBD、标量 MAML 与解析 context；meta_frontier_lab.py 提供 Metatrace 线性特例。

[下载 state_meta_lab.py](https://yingwen.io/zh/continual-rl/download/state_meta_lab.py)

```sh
python3 state_meta_lab.py meta
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [配套章节：时间信用分配](/zh/continual-rl/algorithms/credit-assignment/)：多步回报、资格迹与序列信用；区分权重更新的信用与元参数影响的信用。

- [配套章节：流式学习](/zh/continual-rl/algorithms/streaming/#lesson-output-steps)：数据与计算协议、Stream-X 和 Intentional 的完整尺度推导、实现与对照。

- [Sutton：Adapting Bias by Gradient Descent（IDBD）](https://cdn.aaai.org/AAAI/1992/AAAI92-027.pdf)：原始 IDBD：对数步长、逐特征适应与敏感度历史。

- [Kearney 等：TIDBD 2018 版本](https://arxiv.org/abs/1804.03334)：本页代码具体对应 Algorithm 1 的 semi-gradient、累积迹版本。

- [Kearney 等：Learning Feature Relevance Through Step Size Adaptation](https://arxiv.org/abs/1903.03252)：后续 TIDBD 文献；比较时必须逐式区分版本与归一化/梯度选择。

- [Vivek Veeriah 的 IDBD 实验实现](https://github.com/vivekveeriah/IncrementalDeltaBarDelta)：TIDBD 合作者提供的 IDBD 实验代码；不是 Sutton 1992 官方发布，也不是 TIDBD 完整复现仓库。

- [Young、Wang、Taylor：Metatrace Actor-Critic（IJCAI 2019）](https://www.ijcai.org/proceedings/2019/0581.pdf)：时变权重的元目标、scalar/vector/mixed Metatrace；重点对照元资格迹与敏感度中的完整 TD-error 导数。

- [Metatrace 扩展原文](https://arxiv.org/abs/1805.04514)：含归一化与熵项细节；配套脚本为独立实现的线性特例。

- [Xu、van Hasselt、Silver：Meta-Gradient Reinforcement Learning](https://arxiv.org/abs/1805.09801)：内层更新、后续评价、精确敏感度与衰减 trace 近似的原文。

- [Finn、Abbeel、Levine：MAML](https://proceedings.mlr.press/v70/finn17a.html)：明确任务分布、适应前后目标与可微内层更新。

- [Foerster 等：DiCE](https://proceedings.mlr.press/v80/foerster18a.html)：随机计算图的高阶梯度：普通一阶 surrogate 反复微分可能遗漏或产生错误项。

- [MAML-RL 作者源码](https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/algos/maml_vpg.py)：固定版本的策略梯度适应实现；旧 TensorFlow/rllab 栈，不能假设现代依赖直接兼容。

- [Duan 等：RL²](https://arxiv.org/abs/1611.02779)：循环活动承载快速学习，外层 RL 学习整个 trial 内的适应行为。

- [Rakelly 等：PEARL](https://proceedings.mlr.press/v97/rakelly19a.html)：概率 context、off-policy 数据与任务推断；注意任务稳定范围与 reset 权限。

- [PEARL 作者源码：agent.py](https://github.com/katerakelly/oyster/blob/44e20fddf181d8ca3852bdf9b6927d6b8c6f48fc/rlkit/torch/sac/agent.py)：高斯因子乘积、reparameterized sample、clear_z 与 policy 输入的 detach 实现。

- [PEARL 作者源码：sac.py](https://github.com/katerakelly/oyster/blob/44e20fddf181d8ca3852bdf9b6927d6b8c6f48fc/rlkit/torch/sac/sac.py)：原实现的 Q、V、策略及 encoder/KL 更新时序。

- [Oh 等：Discovering Reinforcement Learning Algorithms](https://arxiv.org/abs/2007.08794)：将学习目标与更新机制本身作为可学习对象的原始工作。

- [Flennerhag 等：Bootstrapped Meta-Learning（ICLR 2022）](https://arxiv.org/abs/2109.04504)：用未来学习产生停止梯度的目标，区分元目标的有效时间范围与反传展开长度。

- [RLJ 论文记录](https://rlj.cs.umass.edu/2025/papers/Paper218.html)：作者与正式会议收录。

- [How Should We Meta-Learn Reinforcement Learning Algorithms? · 作者实现](https://github.com/AlexGoldie/learn-rl-algorithms)：learning_algorithms 中各发现方法与独立 evaluation 流程。 论文提供、仓库标为官方的作者实现。

- [Discovering state-of-the-art reinforcement learning algorithms（Nature 2025）](https://www.nature.com/articles/s41586-025-09761-x)：DiscoRL 的原论文；区分环境间规则迁移和单一 agent 的持续学习。

- [DiscoRL 作者仓库](https://github.com/google-deepmind/disco_rl)：元训练、评价入口与配置；区分规则发现预算和新 agent 的训练预算。

- [DiscoRL 固定版本更新规则源码](https://github.com/google-deepmind/disco_rl/blob/9059a29f7121d60948f25ef165e08e050e9399c8/disco_rl/update_rules/disco.py)：重点读 agent 输出契约、规则输入、辅助预测目标与持久 meta-state。

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/5b76d77e7095c6480ed827b85f0c2878-Paper-Conference.pdf)：Algorithm 1–2、正则化联系与持续实验。

- [作者论文 v3](https://arxiv.org/html/2405.16642v3)：区分凸理论、RL 经验结果与初期表现限制。

- [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning · 作者实现](https://github.com/ComputationalRobotics/TRAC)：trac.py、PyTorch/JAX optimizer 包与控制/视觉实验。 作者项目页与仓库均明确标为官方实现。

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_218.pdf)：比较对象、元训练/测试与多维成本。


---

# 目标与子任务：条件控制、经验重用与技能设计

机器人学会到达一个充电站以后，怎样学习一族目标、利用未成功的尝试，并提出对未来控制有用的子任务？

## 本章内容

- 区分目标表示、目标条件控制、重标记经验、课程选择与子任务发现五个问题。
- 独立写出 goal-conditioned Q-learning 与 HER 的数据循环，知道何时必须重算 reward 和 terminal。
- 推导带 stopping value 的子任务 target，区分 STOMP 原文约定与普通 option 后续价值。
- 用真实任务收益与规划收益评价子任务，而非只看目标到达率。

<a id="problem-definition"></a>

## 本章的问题定义

同一动力学下需要应对一族结果目标，并决定经验如何共享、下一目标如何选择、哪些子任务有长期用途。

### 给定条件与符号

- 目标空间、目标奖励和成功/终止语义，目标分布。
- 目标无关动力学、真实转移及可重算标签的信息，数据和重置预算。

### 需要求解的对象

给定目标的策略/价值；HER改变训练标签，课程改变练习分布，子任务构造另选择有用的行为规格。

### 信息与数据权限

$g$ 是当前指定目标；保存状态、动作、后继、实际达到结果与物理终止。事后目标 $g'$ 只能重算评价标签，不能改写已发生转移。

$$
\max_\pi\ \mathbb E_{g\sim p_G}\mathbb E_{\pi(\cdot\mid s,g)}\!\left[\sum_{t=0}^{T_g-1}\gamma^t r_g(S_t,A_t,S_{t+1})\right]
$$

$p_G$ 是声明的目标评价分布，$r_g$ 为该目标奖励，$T_g$ 为按目标语义定义的停止时刻，$\gamma$ 为折扣。UVFA学习一族价值；重标记分布不等于 $p_G$。STOMP另以沿途cumulant与停止价值定义子任务，非普通到达目标。

### 成立条件与解的含义

- 重标记要求动力学不因目标改变；必要标签可由保留信息重算。
- 随机环境中按实际结果选目标可能条件化噪声；标准HER不普遍无偏。子任务停止价值计时遵从明确的return约定。

判断准则：在原先指定且未用于选择的目标上测成功率、步数和收益；重标记检验奖励与终止重算；子任务还需测其技能/模型对主任务规划的用途。

### 适用边界

- hindsight成功标签不等于原真实目标已经成功。
- 子任务到达率高不证明有主任务价值。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：相对完整持续控制，本章限定目标参数索引的任务族和可共享动力学；目标输入、重标记权限与目标分布是这类控制问题的具体结构。

- 组合不同学习问题 · [Options 与技能发现](https://yingwen.io/zh/continual-rl/construction/options/)：子任务是行为评价规格，option是包含执行与停止的可复用解。

- 组合不同学习问题 · [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)：课程选择改变真实经验分配，不能由目标条件控制或HER自动完成。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

稀疏目标使失败轨迹缺少反馈；共享目标可能引入分布偏差，局部子目标还可能伤害主任务。

### 本章的核心思路

把目标规格、标签操作、课程与子任务停止收益分别定义，再检查各自连接外部收益的路径。

1. [从每个目标推导共享控制](#lesson-derive)：因为目标只改变评价，先写每个目标的Bellman方程，再以UVFA共享参数。

2. [重算事实的评价而非事实](#lesson-derive)：因为失败轨迹可能完成别的目标，HER保留转移并重算奖励/目标终止，同时检查随机选择偏差。

3. [用沿途收益约束子任务](#lesson-subtasks)：因为最短到达可能忽略危险代价，reward-respecting子任务保留沿途奖励并以停止价值表达可复用后果。

结论与条件：固定目标恢复普通控制定义；HER有效性取决于动力学和重标记条件，STOMP停止规则及折扣计时不能凭普通option公式替换。

### 相关方法改变了什么

- UVFA：共享一族目标价值，不决定练习目标。

- HER：重用已发生经验的目标标签，不改变实际交互成功。

- 课程/STOMP：课程选择经验分配；STOMP提出有沿途收益与停止价值的技能子任务。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 状态、动作、目标

$s$ 是此刻用于决策的信息，$a$ 是可执行动作，$g$ 是希望达到的结果规格。目标可以用坐标、特征、图像或奖励参数表达；表示之外，还需要明确成功条件或奖励函数。

### Q-learning 的含义

$Q(s,a,g)$ 估计先做动作 $a$、随后尽力完成目标 $g$ 的累计回报。一次真实转移提供当前奖励和下一状态；最大值对应后续选择价值最高的动作。

$$
Q(s,a,g)\leftarrow Q(s,a,g)+\alpha\left[r_g+\gamma(1-d_g)\max_{a'}Q(s',a',g)-Q(s,a,g)\right]
$$

### Bootstrap 与半梯度

用自己对未来的估计构造当前标签叫 bootstrap。神经网络训练时通常对 target 停止梯度，只对当前 Q 的输出求导；target network 可减缓标签与拟合器同时变化。

### Option 与子任务不是同一物

子任务规定评价某种行为的标准；option 是一种可执行的解，由内部策略 π 与停止概率 β（以及允许启动的位置）组成。一个子任务可能有多个解，也可能始终解不好。

<a id="lesson-setting"></a>

## 1. 目标如何进入一个强化学习问题

想象一辆仓库小车，能够左右移动，当前位置是 s，目的货架是 g。它在一次尝试中没有到达货架 4，却经过了位置 2。最直接的问题是“以后给定任意 g，怎样行动”；另一个问题是“这次失败能否教会到达 2”；再一个问题是“下一次应该练习哪个目标”。三者分别改变价值函数的输入、训练样本的标签、真实交互的分配，不能用一个 HER 名称包办。

| 研究对象 | 输入与输出 | 没有被顺带解决的事 |
| --- | --- | --- |
| 目标表达 | 由 g 定义奖励 $r_g$、成功条件与可选的终止条件 | 并不保证目标可达或可观察 |
| 目标条件控制 / UVFA | 给 s,a,g，预测 Q；给 s,g，输出动作 | 不决定下一次练哪个 g |
| HER / 重标记 | 给已发生的轨迹，构造另一 g 下的训练转移 | 不把失败事实改成真实外部成功 |
| 目标生成 / 课程 | 给学习进度与经验，选择下一批 g | 不等于已学到可靠技能或它的模型 |
| 子任务 / STOMP | 给沿途 cumulant 与停止价值，学习行为及停止 | 学到行为以后还要学习后果模型并检查规划用途 |

先考虑有限、可观察、动力学不依赖目标的 MDP：环境转移 $P(s'|s,a)$ 不因事后改写 $g$ 而改变。真实数据包含 $(s,a,r,s')$；为了重算目标奖励，还要保存 achieved goal（实际达到的结果）、物理终止与必要的附加信息。下面允许经验重放和环境重置。扩展到 CRL 时，还需明确存储和重置限制，以及目标分布、动力学和目标语义分别怎样变化。

判断是否需要这一条线：若问题只是一个固定目标的回报优化，先把普通控制做好；若同一物理世界要反复应对不同结果，goal-conditioned 控制可以共享经验；若目标奖励过稀疏，考虑 HER；若经验总用于过易或根本不可达的目标，再考虑课程；若希望长期积累可组合的行为与模型，才进一步构造子任务。

<a id="lesson-derive"></a>

## 2. 从一族控制问题推到 UVFA 与 HER

对每个目标 g 先固定奖励 $r_g$ 和终止 $d_g$。最常见的到达任务令成功奖励为 0、未成功为 −1；成功立即结束时，回报衡量到达前的折扣步数。也可以成功后继续交互，只在奖励里反映成功。这两种任务的 Bellman 方程不同，不能把 success、环境 terminated 和时间上限 truncated 混成一个 done。

$$
Q_g^*(s,a)=\mathbb E\!\left[r_g(s,a,S')+\gamma(1-d_g)\max_{a'}Q_g^*(S',a')\mid s,a\right]
$$

这是对固定 g 的普通最优 Bellman 方程。期望来自真实动力学；g 告诉我们怎样评价结果，不负责改变已发生的转移。

UVFA 让不同目标 $g$ 共享一个函数近似器。最简单实现把状态与目标特征拼接后输入网络，也可以分别编码再组合；每个目标仍遵循自己的 Bellman 方程。共享使相近目标互相提供训练信号，也可能产生干扰。泛化能力取决于数据覆盖和表示，需在未用于训练的目标上单独评价。

$$
Q_\theta(s,a,g),\qquad y=r_g+\gamma(1-d_g)\max_{a'}Q_{\bar\theta}(s',a',g)
$$

θ 是在线网络，θ̄ 是慢更新的目标网络；表格实现不需要两个网络。这里展示离散动作版本；连续动作可使用目标 actor 提供下一动作。

$$
\begin{aligned}L(\theta)&=\tfrac12\left(Q_\theta(s,a,g)-\operatorname{stopgrad}(y)\right)^2\\ \theta^+&=\theta+\alpha\left(y-Q_\theta(s,a,g)\right)\nabla_\theta Q_\theta(s,a,g)\end{aligned}
$$

对平方误差求导就得到更新。target 被视作本次拟合的固定标签，因此这是半梯度；并不是对完整 Bellman 残差两侧一起求导。

HER 再增加一个数据操作：先真实执行原目标 g，保留轨迹；抽到其中一步时，选一个后续实际达到的结果 g′；状态、动作、下一状态保持不变，只按 g′ 重算奖励和目标相关的终止。于是“没完成 g”仍能为“怎样完成 g′”提供监督。HER 要配合能利用其他行为策略数据的 off-policy 学习器；它不是直接把 PPO 的 on-policy 轨迹标签改掉就仍保留原保证。

**算法：算法伪代码**

1. 收集一段 ($s_t,a_t,s_{t+1}$，$\mathrm{achieved}_{t+1}$，`physical_done`，`info`)
1. 对抽到的时刻 t：
  1. 以一定概率保留原目标 g；否则从未来 $\mathrm{achieved}_{t+1:T}$ 采样 g′
  1. 不改 $s_t,a_t,s_{t+1}$
  1. 重算 $r_{g^{\prime}}(s_t,a_t,s_{t+1})$
  1. 按所定义任务重算 $d_{g^{\prime}}$，物理终止仍保留
  1. 构造目标 y；更新 Q（以及连续动作时的 actor）
1. 独立评估：只用预先规定的真实目标，不用 hindsight 标签

在确定性且目标无关的环境中，原转移仍是新目标下同一个状态—动作对的有效结果。随机环境里，按后来实际发生的结果选择新目标，可能条件化转移噪声：例如偶然成功的随机结果，被过度表示为可稳定达成的目标。此时标准 HER 不普遍给出无偏 Bellman 样本。课程分布还会改变训练目标的权重，因此重标记后的训练误差与原目标分布上的成功率需要分别评价。

<a id="lesson-subtasks"></a>

## 3. 从到达目标到有停止规则的子任务

外部给定目标只要求解决一个任务；子任务构造还要决定什么行为值得长期保留。考虑通向门口的两条路：一条两步但经过危险区，每步真实代价 −5；另一条四步但无危险。仅奖励“最短到门口”可能学到对主任务有害的技能。reward-respecting 的出发点是保留沿途真实奖励，同时用停止价值表达“这个结果有时值得准备”。

$$
G^{c,z}=\sum_{k=1}^{K}\gamma^{k-1}C_{t+k}+\gamma^{K-1}z(S_{t+K})
$$

K≥1 是本次子任务停止时刻，C 是累积信号，z 是停止价值。这采用 STOMP 原文式(2)的计时：停止价值与最后一次 cumulant 具有同一折扣指数，不是再过一个原始时间步。

将第一步单独拿出。到 $s'$ 后，以 $\beta(s')$ 的概率停止并获得 $z(s')$；否则从 $s'$ 的下一步继续积累，其全部未来相对当前多乘一个 $\gamma$。对这两个分支取期望，得到以下递推。停止的是子任务回报的累积，物理环境仍可继续运行。

$$
\begin{aligned}v(s)&=\mathbb E_\pi\!\left[C+\beta(s')z(s')+\gamma(1-\beta(s'))v(s')\mid s\right]\\ \delta&=C+\beta(s')z(s')+\gamma(1-\beta(s'))v(s')-v(s)\end{aligned}
$$

β=1 时 target=C+z；β=0 时 target=C+γv(s′)。真实环境终止使用 β=1 且 z=0。两个端点是移植实现时最值得先写的测试。

$$
z^i(s)=w^\top x(s)-w_i x_i(s)+\bar w^i x_i(s)
$$

STOMP 的 feature-attainment 子任务把某一特征的价值权重替换为一个乐观 bonus weight。它不是任意给 feature 一个大奖励：其他特征当前的价值仍保留，沿途 cumulant 也保持真实奖励。

固定目标策略与停止函数时，可以直接做 TD；行为由另一策略 $b$ 产生时，一步半梯度更新可乘 $\rho=\pi(a|s)/b(a|s)$。分母使用产生动作时记录的概率，且目标动作需要行为支持。这个比率修正当前动作分布；函数逼近与 off-policy bootstrap 的稳定性仍需另行分析。

$$
\begin{aligned}v(s)&\leftarrow v(s)+\alpha\rho\delta\\ \theta&\leftarrow\theta+\alpha_\pi\rho\delta\nabla_\theta\log\pi_\theta(a|s)\end{aligned}
$$

这里取 $\lambda=0$，同一 TD 误差 $\delta$ 在 critic 更新前缓存。重要性比修正当前动作分布；actor 所见的状态分布仍由采样过程决定，完整目标的梯度解释还取决于相应占用权重。

停止收益的计时还会影响停止规则。STOMP 式(9)采用 $\beta(s)=\mathbf1[z(s)\geq v(s)]$；若保持当前 $v$ 不变、单独最大化前面的单步 target，比较量却是 $z(s)$ 与 $\gamma v(s)$。例如 $z=9.5$、$v=10$、$\gamma=0.9$ 时，前一规则继续，后一规则停止。因此，复现该论文应保留其规则；设计另一种最优停止问题则应明确 return 与边界条件，再重新推导，不能仅凭符号相似互换。

**算法：算法伪代码**

1. 初始化每个子任务的价值 $v_i$、策略 $π_i$、停止价值规格 $z_i$
1. 每个真实转移：
  1. 记录旧价值、旧策略概率、行为概率 b(a|s)
  1. 由选定规则计算 $β_i(s^{\prime})$，环境终止强制 $β_i=1,z_i=0$
  1. 对每个 i 计算 $δ_i = r + β_i z_i + γ(1-β_i)v_i(s^{\prime}) - v_i(s)$
  1. 更新 critic；用已缓存 $δ_i$ 更新 actor
  1. 另行学习此 option 的真实 reward/end-state 模型
  1. 评价该模型用于主任务规划的改善，不能只看子任务成功率

<a id="lesson-curriculum"></a>

## 4. 哪个目标值得现在学：显式课程到持续目标生成

先做一个透明的有限目标课程：每个目标维护尝试次数、成功次数和近期成功率，优先采样既非几乎必成、也非长期完全失败的目标。这是“适中难度”的可运行基线，不是 Goal GAN 的完整实现。Goal GAN 再学习一个生成器来覆盖适中难度目标区域；分类器用当前策略对目标的成功概率构造标签，生成器提出下一批练习目标。它增加的是目标分布学习，而不改变低层控制的 Bellman 方程。

$$
\hat p_t(g)=\frac{\text{近期成功次数}}{\text{近期尝试次数}},\qquad \mathcal G_{\rm learnable}=\{g:\ell\leq\hat p_t(g)\leq u\}
$$

近期窗口避免历史累计均值完全淹没近期变化；阈值与置信度、窗口长度共同决定课程。探索保底应保证暂时失败的目标并非永远被排除。

学习进度是另一种信号：比较同一目标在两个相邻窗口的成功率或预测误差改善。纯粹选进步最快的目标可能遗忘旧目标，也可能把噪声造成的波动当作进步；所以应保留复习预算、未探索目标预算和难度不确定性。CRL 中目标语义与环境可达性一起变化时，还需要区分“技能退化”与“世界变了”。这应成为实验中的独立变量，而非让课程默默选择更容易的任务。

| 分支 | 实际改变的对象 | 必要对照 |
| --- | --- | --- |
| 固定目标集 + 均匀采样 | 只控制训练目标分布 | 先确认控制器能解单目标 |
| 适中难度 / Goal GAN | 采样或生成可学习目标 | 相同交互预算下与均匀目标比较 |
| 学习进度驱动 | 按能力变化分配练习 | 区分噪声、遗忘和真正进步 |
| 奖励相关子任务 | 改变行为学习的 cumulant / stopping value | 与最短路径、随机技能比较规划收益 |
| 持续目标发现 | 候选产生、评估、保留与淘汰共同在线变化 | 计入候选学习与模型维护的算力/内存成本 |

<a id="lesson-example"></a>

## 5. 一次经验如何产生两个不同但合法的更新

小车从 $1$ 向右到 $2$，原目标为 $4$，$\gamma=0.9$、$\alpha=0.5$。当前动作价值为 $-2$，下一状态对目标 $4$ 的最大动作价值为 $-1$。任务成功即终止，成功奖励为 $0$，其余为 $-1$。原目标未达成，故 target 为 $-1+0.9\times(-1)=-1.9$，新值为 $-2+0.5\times0.1=-1.95$。

把同一转移重标为目标 $2$：物理过程不变，但目标已达成，$r'=0$、$d'=1$，不再 bootstrap。若当前动作对目标 $2$ 的价值原为 $-2$，更新后为 $-2+0.5\times(0-(-2))=-1$。两个值回答不同的条件问题。若任务规定成功后继续，$d'$ 则不自动变为 $1$，应保留相应后续价值。

| 本次计算 | 原目标 4 | 重标记目标 2 |
| --- | --- | --- |
| reward | −1 | 0 |
| 目标终止 | 否 | 是 |
| target | −1.9 | 0 |
| 更新前 Q | −2 | −2 |
| 更新后 Q | −1.95 | −1 |

子任务例子中，$v(s)=2$、$v(s')=6$、$r=1$、$z=4$、$\beta=0.25$。停止分支贡献 $0.25\times4=1$，继续分支贡献 $0.9\times0.75\times6=4.05$，target 合计 $6.05$。取 $\alpha=0.1$、$\rho=1$，新价值为 $2.405$。这里的 $z$ 是子任务停止价值，区别于主任务中“换一个 option 继续”的高层价值。

<a id="lesson-code"></a>

## 6. 实现与实验：目标标签和终止条件

目标重标记、表格 goal-Q、future HER、子任务 TD 与有限目标课程。

```python
def relabel_goal(next_position: int, goal: int, physical_terminal: bool = False):
    """Our goal task ENDS on success; another task may not use this convention."""
    success = next_position == goal
    reward = 0.0 if success else -1.0
    terminal = physical_terminal or success
    return reward, terminal


def goal_q_update(q, state, action, next_state, goal, alpha=0.5, gamma=0.9,
                  physical_terminal=False):
    reward, terminal = relabel_goal(next_state, goal, physical_terminal)
    next_value = 0.0 if terminal else max(q.get((next_state, a, goal), 0.0)
                                          for a in (-1, 1))
    key = (state, action, goal)
    old = q.get(key, 0.0)
    target = reward + gamma * next_value
    q[key] = old + alpha * (target - old)
    return reward, terminal, target, q[key]


def future_her(trajectory, original_goal, rng):
    """trajectory: (s,a,s_next,physical_terminal) in one recorded episode.

    Outputs (s,a,s_next,goal,physical_terminal), preserving physical termination
    for both original and hindsight goals. Goal success must never erase it.
    Dynamics are deterministic and goal independent in this teaching example.
    """
    replay = []
    for t, (s, a, sn, physical_terminal) in enumerate(trajectory):
        replay.append((s, a, sn, original_goal, physical_terminal))
        future_index = rng.randrange(t, len(trajectory))
        hindsight_goal = trajectory[future_index][2]
        replay.append((s, a, sn, hindsight_goal, physical_terminal))
    return replay


def subtask_td(value, state, next_state, reward, stopping_value, beta,
               alpha=0.1, gamma=0.9, rho=1.0):
    """STOMP Eq.(5) timing; beta is supplied, not optimized by this function."""
    old_here, old_next = value[state], value[next_state]
    target = reward + beta * stopping_value + gamma * (1.0 - beta) * old_next
    value[state] = old_here + alpha * rho * (target - old_here)
    return target


def stop_rules(stopping_value, continuation_value, gamma=0.9):
    paper_rule = stopping_value >= continuation_value  # STOMP Eq.(9)
    greedy_one_step = stopping_value >= gamma * continuation_value
    return paper_rule, greedy_one_step


def intermediate_goals(success_counts, attempt_counts, low=0.2, high=0.8):
    """Transparent finite curriculum; not the Goal GAN training algorithm."""
    return [g for g in success_counts if attempt_counts[g] > 0
            and low <= success_counts[g] / attempt_counts[g] <= high]
```

下载脚本后运行；仅需 Python 3.10+ 标准库

```sh
python3 knowledge_algorithms_lab.py goals
python3 knowledge_algorithms_lab.py test
```

运行结果中，原目标的新值为 $-1.95$，HER 新值为 $-1$，子任务 target 为 $6.05$；两种停止比较在 $z=9.5$ 时分别返回继续与停止。physical_terminal 随原样本和重标记样本一起传递。把物理终止后状态的 $Q$ 人为设成 $100$，未达成目标的转移 target 仍为 $-1$，而非错误 bootstrap 得到的 $89$。这个例子同时检查了“目标是否成功”和“物理过程是否结束”两个独立条件。

- 实验 A：把“成功即终止”改为“成功后仍继续”。同时修改 relabel_goal 和对应测试，观察若只改 reward、不改 bootstrap 会怎样。
- 实验 B：为相同 (s,a) 加随机分支，观察 future relabel 后的成功结果是否过度代表真实概率；分别在原目标评估与重放 loss 上作图。
- 实验 C：固定两条通向同一目标的路径，一条短但有负奖励。比较最短路径子任务与真实奖励加 stopping value 的排序，同时记录到达率和沿途代价。

OpenAI Baselines 的 her_sampler.py 展示了“选未来 achieved goal → 替换 desired goal → 调用 reward_fun”的数据流程；随后可读 ddpg.py 的 target 和 replay buffer。原工程采用自己的定长 episode 与成功处理约定，移植到成功即终止任务时，需要同步调整 bootstrap。Goal GAN 的 rllab-curriculum 则按目标采样、成功率标签、策略更新三个环节阅读。两套历史工程适合在独立依赖环境中运行。

<a id="lesson-branches"></a>

## 7. 如何组合目标学习的几个组成部分

目标相关方法可以组合，因为它们修改不同接口：UVFA 表示一族价值函数，HER 重用经验，课程分配交互，子任务定义评价，option 实现行为。一个系统可以同时包含它们，也可能只需要其中一个。组合并不自动有效：HER 的训练目标分布可能与课程提出的目标失配；新生成目标可能无法由现有表示判断成功；有用行为如果没有准确后果模型，仍不能用于可靠规划。

| 你观察到的失败 | 优先检查的量 | 可比较的方法线 |
| --- | --- | --- |
| 目标一换就从零学 | Q 是否以目标为条件，是否共享可迁移表示 | UVFA / goal-conditioned actor-critic |
| 几乎没有成功样本 | 奖励可否从记录数据重算，动力学是否目标无关 | HER / relabeling |
| 只会练容易任务 | 课程覆盖、成功率估计、探索保底 | 均匀采样 / 适中难度 / 学习进度 |
| 到达子目标却损害主任务 | 沿途真实 reward 是否被删掉 | shortest-path / reward-respecting |
| 技能很多而规划没收益 | 模型精度、技能适用性、维护预算 | STOMP 的行为—模型—规划联合评价 |

<a id="lesson-offline-goals"></a>

## 8. 目标很远时：从平坦价值函数到 HIQL

前面的 HER 假设能继续交互，错误的价值还有机会被新经验纠正。离线目标条件 RL 只给定一个已有轨迹集，学习者不能随时尝试一个看起来很好的动作。如果目标很远，好坏动作的价值差很小，估值噪声就可能盖过真正的控制信号。分层的一个作用是把“一步选哪个动作”改成“先到哪个较近的中间状态”。

HIQL（NeurIPS 2023）使用一个目标条件状态价值 $V(s,g)$，然后提取两个策略：高层 $\pi_h(z|s,g)$ 预测中间状态的潜在表示，低层 $\pi_\ell(a|s,z)$ 产生到达这个中间目标的动作。这里的高层动作不是环境按钮，而是数据中未来状态的表示；低层才承担可执行控制。

$$
\begin{aligned}\omega_j&=\operatorname{sg}\!\left[\min\{\exp(\eta_j A_j),100\}\right],\qquad j\in\{\ell,h\},\\ z_{\rm target}&=\operatorname{sg}\!\left[\phi(s_{t+k})\right],\\ J_\ell&=\mathbb E_{\mathcal D}\left[\omega_\ell\log\pi_\ell(a_t\mid s_t,\phi(g_\ell))\right],\\ J_h&=\mathbb E_{\mathcal D}\left[\omega_h\log\pi_h(z_{\rm target}\mid s_t,g)\right].\end{aligned}
$$

两层 actor 做优势加权回归；$\eta_\ell$ 与 $\eta_h$ 分别控制权重尖锐程度。这里写出作者实现的上限 100。$\operatorname{sg}$ 保持数值、停止该通路的梯度：actor 拟合固定权重和固定高层标签，不通过改变它们来降低回归损失。

价值为两层策略提供评价，作者实现可以在同一批更新中联合计算价值、低层 actor 与高层 actor 的损失。回归权重使用价值差：低层 $A_\ell=V(s_{t+1},g_\ell)-V(s_t,g_\ell)$，高层 $A_h=V(s_{t+k},g)-V(s_t,g)$。计算这些权重及高层标签时使用更新前参数，并相对于本次 actor 求导保持固定；价值仍由自己的损失学习。目标表示还可依赖当前位置，前式用 $\phi(g)$ 简写该接口。低层策略输入中的目标编码是否接受 actor 梯度由作者代码的 policy_train_rep 配置决定，不能与停止高层回归标签梯度混为一谈。

状态序列可训练价值和高层，但低层动作策略仍需要带动作的数据。这里利用无动作数据扩大的是对“哪些状态连接到哪里”的知识；执行动作仍须通过动作数据学习。把数据来源分开，是评价这类方法与纯视频学习、完全离线控制之间关系的必要条件。

HIQL 与 HER 的关系是互补的：HER 处理目标标签和经验使用，HIQL 处理长期控制中的策略结构与离线提取。它与可变时长 Option-Critic 也不同：中间目标和训练间隔给出了分层结构，并不自动学习同样的终止梯度。用于持续 RL 时，还需研究新经验怎样更新旧价值和目标表示，而不是仅换一批测试目标。

OGBench（ICLR 2025）适合检验这类区别。它把长时域到达、轨迹拼接、像素输入和随机性等挑战放进统一的离线目标学习基准，并提供多种基线实现。起步时选一个状态输入任务，固定数据与目标抽样，比较平坦目标策略和 HIQL；再依次改变数据覆盖、子目标间隔和观测形式。先解释哪一种困难导致差距，再扩大任务规模。轨迹边界标记与 bootstrap mask 在数据接口中具有不同含义，时间截断不应自动当作真实任务终止。

<a id="lesson-language-skills"></a>

## 9. 人类如何指定技能：MaestroMotif 的语言到控制过程

有些子任务来自图结构，有些来自人的语义要求。例如“寻找食物”和“回避危险”比一个坐标更适合描述 NetHack 中的长期行为，却难以直接手写一个可靠的数值奖励。MaestroMotif（ICLR 2025）将语言模型放在技能设计环节：把语言要求转换成可学习奖励、启动与终止规则、以及训练时的技能调度，然后仍由 RL 学习执行技能的低层策略。

第一步，给定技能描述 $x_i$，让语言模型比较两段候选观察中哪个更符合要求，得到偏好标签 $y$；随后训练奖励模型 $r_i$。一个基本的成对偏好模型是令奖励差决定偏好概率。

$$
\begin{aligned}p_i(s_1\succ s_2)&=\sigma(r_i(s_1)-r_i(s_2))\\ \mathcal L_i&=-\mathbb E\left[y\log p_i+(1-y)\log(1-p_i)\right]\end{aligned}
$$

这将语言反馈蒸馏成低层训练时可反复查询的标量信号。损失只约束相对偏好，奖励平移本身不可辨识；其尺度、偏好一致性及分布外可靠性仍影响控制。

第二步，依据技能规格生成 $I_i$、$\beta_i$ 的程序和训练时的高层调度程序。第三步，在调度所产生的状态分布中，以学得的 $r_i$ 训练技能条件低层策略。第四步，面对下游任务描述，生成组合这些既有技能的程序。学习奖励、写执行规则、学原始动作和组合技能是四种不同操作。

**算法：MaestroMotif 的流程；语言反馈规定行为语义，强化学习承担行为执行**

1. 人类给出技能描述与技能组合要求
1. 语言模型比较观察对 → 训练每个技能的奖励模型
1. 语言模型生成启动、终止与训练调度程序
1. 在调度产生的交互中，用 RL 训练技能条件低层策略
1. 下游任务描述 → 组合既有技能的程序 → 执行并评估外部任务

这种方法与 STOMP 的差别在奖励来源：STOMP 保留环境沿途奖励并改变停止价值，MaestroMotif 则依据语言偏好学习技能奖励，还借助人类规格和预训练语言知识。其下游组合无需重新训练某个任务的低层策略，不意味着技能本身没有经过 RL 训练，也不意味着技能规格已经由智能体自主发现。

作者仓库可按三个目录阅读：preference 对应偏好与奖励学习，code_generation 对应规则及组合程序，rl_baseline 与 sample_factory 对应技能执行训练。一个有辨识力的实验是固定技能策略，分别改变奖励模型、训练调度和下游组合程序：技能失败究竟来自语义评价、访问不到适合学习的状态，还是高层组合不当？这比只比较最终任务得分更能定位改进方向。

<a id="lesson-check"></a>

## 10. 失败诊断与自测答案

- Q 总是接近零而真实成功率不升：检查重标记比例过高、成功标签泄漏，以及评估是否错误采用事后目标。
- HER 后回报上升但原目标失败：检查训练目标覆盖；重标记只证明某些结果可学习，不保证原目标可达。
- 子任务全部立刻停止：检查 stopping value 尺度、初始化、是否错误地把 β 当作环境 done，以及是否允许零步启动就停止。
- CRL 性能忽高忽低：分别记录目标分布、环境变化和模型版本，并按这些条件分析成功率。

自测 1：为何 HER 不能随意改 s′？答：状态转移是事实证据；改变目标是在同一事实下提出新问题，改变结果则是在伪造没有经历过的转移。自测 2：把 stopping bonus 写进主任务模型会怎样？答：规划会把学习辅助奖励当真实收益，重复调用技能可能获得虚假的“奖励”。自测 3：目标编码共享的失败就是灾难性遗忘吗？答：不一定，也可能是目标本来不可区分、训练分布变了或价值估计尚未收敛，应通过固定目标保留集和表示可辨识性实验区分。

## 本章的实验设计

子任务成功率与主任务收益需要分别测量。设计固定目标、随机目标和学得目标三个对照。

设定：在小型地图固定低层技能和高层控制器，比较固定目标、随机目标与学习目标。研究者图结构只用于可达性诊断。

- 目标成功判定只用当时可得信号。
- 后见目标标签用于训练，不作为过去行动时已经知道的输入。
- 不可达目标有时限、失败与退出语义。

对照：数量和尺度匹配的随机目标；固定手工目标的特权诊断；固定技能只改目标；再固定目标只改技能

记录：目标到达率、到达时间与失败率；新任务收益和目标使用频率；全部目标生成和训练成本

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-modules)

## 学习与研究衔接

目标定义偏好，状态保留信息。学会达到给定目标，不等于学会构建有用目标。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-goals) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=goals) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=goals)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

状态是支持后续计算的内部信息；GVF 指定一个预测问题；RTRL 和资格迹规定信用如何传播。三者可以组合，但不是相互替代的算法名称。先理解给定策略的预测，再讨论预测怎样改善控制。

- [Does Zero-Shot Reinforcement Learning Exist?](https://yingwen.io/zh/continual-rl/research/#recent-zero-shot-forward-backward)

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

Laplacian 描述行为图结构，奖励感知表示加入路径价值，METRA 学习有区别的行为，HIQL 利用离线目标轨迹，MaestroMotif 引入语言先验。它们承担不同的设计工作；生成技能、选择技能与组合技能需要分别评价。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [Laplacian Keyboard: Beyond the Linear Span](https://yingwen.io/zh/continual-rl/research/#recent-laplacian-keyboard)
- [HIQL: Offline Goal-Conditioned RL with Latent States as Actions](https://yingwen.io/zh/continual-rl/research/#recent-hiql-hierarchical-goals)
- [MaestroMotif: Skill Design from Artificial Intelligence Feedback](https://yingwen.io/zh/continual-rl/research/#recent-maestromotif-semantic-skills)
- [Foundation Policies with Hilbert Representations](https://yingwen.io/zh/continual-rl/research/#recent-hilbert-foundation-policies)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

模型可提取性的理论说明某类能力需要什么知识，不指定唯一网络。Dreamer 研究潜在想象控制，STOMP 研究随机时长行为模型，DRAGO 研究旧模型知识保留。模型误差、查询策略和规划收益之间仍需实验连接。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [General Agents Contain World Models](https://yingwen.io/zh/continual-rl/research/#recent-general-agents-world-models)
- [Laplacian Keyboard: Beyond the Linear Span](https://yingwen.io/zh/continual-rl/research/#recent-laplacian-keyboard)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [OGBench: Benchmarking Offline Goal-Conditioned RL](https://yingwen.io/zh/continual-rl/research/#recent-ogbench-goal-evaluation)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文提供定义和条件，架构讲座提出模块组织，算法论文检验特定机制。完整系统还要明确智能体、外部设计者和世界各自承担的工作；组件成立不自动意味着组合后的长期收益成立。

- [General Agents Contain World Models](https://yingwen.io/zh/continual-rl/research/#recent-general-agents-world-models)
- [The OaK Architecture: A Vision of SuperIntelligence from Experience](https://yingwen.io/zh/continual-rl/research/#recent-oak-architecture)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)

### Reward-Respecting Subtasks for Model-Based Reinforcement Learning

Richard S. Sutton, Marlos C. Machado, G. Zacharias Holland, David Szepesvari, Finbarr Timbers, Brian Tanner, Adam White

Artificial Intelligence · 2023 · 支持方法与理论

#### 研究问题

学到一个能到达子目标的技能之后，为什么它仍可能不适合主任务规划？

#### 关键机制

STOMP 把子任务、option、模型和规划连起来。子任务保留原任务的路径奖励，并用带有特征偏好的终止价值表达目标；学习得到策略和终止规则后，再预测该行为的累计奖励与折扣终点。这样，技能不会因为只追求到达子目标而忽略途中代价。

#### 证据

论文用明确的小问题展示奖励感知子任务如何产生更有用的行为和规划模型。它提供的是可分析的构造链，而非只比较一个技能执行成功率。

#### 条件与限制

终止收益的约定是子任务定义的一部分，不能随意换成固定终点奖励。特征和子任务候选的选择尚不等于完整自主发现机制；实验也不构成整个 OaK 架构的验证。

#### 阅读与实验

在同一个绕路环境中比较“最短到达目标”和“保留路径奖励”的子任务。分别计算 option 的奖励模型、折扣终点模型与一次规划备份。

#### 原文与相关入口

- [期刊论文](https://doi.org/10.1016/j.artint.2023.104001)：STOMP 与奖励感知子任务的正式论文。
- [作者预印本](https://arxiv.org/abs/2202.03466)：最初预印本早于期刊年份；阅读停止收益的精确定义。

### Laplacian Keyboard: Beyond the Linear Span

Siddarth Chandrasekar, Marlos C. Machado

arXiv 预印本 · 2026 · 支持方法与理论

#### 研究问题

从一组谱技能出发，能否解决超出原特征线性奖励空间的新任务？

#### 关键机制

Laplacian 特征先定义行为基，并借助后继特征预测各行为的后果。固定任务权重的价值组合受特征张成空间限制；论文进一步使用随状态变化的元策略，在不同位置组合已有行为。关键变化是组合规则从一组全局固定权重变为状态相关的行为选择。

#### 证据

论文对行为基与任务组合给出理论分析，并报告有限环境中的组合实验。它延续 eigenoptions 与 successor features 的路线，同时解释了为什么单纯线性读出会遇到表达边界。

#### 条件与限制

理论结论依赖具体的行为基、近似误差和任务条件。技能集合的长期生成、淘汰与非平稳模型维护仍是另外的问题；此处按预印本收录，不指定未经确认的会议。

#### 阅读与实验

构造一个必须在中途切换方向的奖励任务。分别比较固定权重的技能选择与状态相关切换，并解释性能差异来自哪里。

#### 原文与相关入口

- [作者预印本](https://arxiv.org/abs/2602.07730)：阅读线性张成空间的限制及状态相关组合机制。

### HIQL: Offline Goal-Conditioned RL with Latent States as Actions

Seohong Park, Dibya Ghosh, Benjamin Eysenbach, Sergey Levine

NeurIPS 2023 · 2023 · 支持方法与理论

#### 研究问题

只拿到已有轨迹时，长距离目标为什么适合拆成高层子目标和低层动作？

#### 关键机制

HIQL 学习目标条件价值，并以潜在状态作为高层动作。高层提出中间目标，低层输出环境动作；两层利用优势加权回归学习。时间分解让低层面对较短的控制距离，而不是要求一个策略直接消化所有远距离价值误差。

#### 证据

论文在离线长时域目标任务中检验层次结构，并提供原始实现。作者后来在 OGBench 中提供更统一的实现，二者适合不同用途：原实验复现和统一基线比较。

#### 条件与限制

数据覆盖和行为分布约束仍然存在。目标采样、层级时间间隔与离线轨迹由外部流程提供，不能把效果解释为在线自主目标生成已经解决。

#### 阅读与实验

对一段轨迹明确标记最终目标、中间目标和当前动作。逐一检查价值目标、优势权重和高层标签的停止梯度边界。

#### 原文与相关入口

- [NeurIPS 2023 原文](https://papers.nips.cc/paper_files/paper/2023/file/6d7c4a0727e089ed6cdd3151cbe8d8ba-Paper-Conference.pdf)：离线目标学习和两层回归目标。
- [HIQL 原始实现](https://github.com/seohongpark/HIQL)：README 区分原始实验与 OGBench 中的新实现。

#### 作者代码

[作者仓库；更新的统一基线另见 OGBench。](https://github.com/seohongpark/HIQL)

HIQL 原论文的离线训练与评价。

### OGBench: Benchmarking Offline Goal-Conditioned RL

Seohong Park, Kevin Frans, Benjamin Eysenbach, Sergey Levine

ICLR 2025 · 2025 · 评价与实验协议

#### 研究问题

一个目标条件算法表现不好，是长时域、轨迹拼接、视觉表示还是随机性造成的？

#### 关键机制

OGBench 用不同环境类型与数据集分别施加这些困难，并提供统一的目标条件基线。固定离线数据让算法面对相同经验，从而将学习机制的差异与在线探索能力的差异暂时分离。

#### 证据

论文提供八类环境、八十五个数据集和六类算法实现。价值在于可复用的实验接口与困难分解，而不只是汇总一个排行榜。

#### 条件与限制

固定数据不检验智能体如何主动获得未来经验，也不直接检验单次生命的灾难性变化、恢复或长期资源管理。它适合 CRL 子问题实验，不是完整 CRL 的替代品。

#### 阅读与实验

先选择只改变一种困难的两个数据集，再比较 HIQL 与平坦目标策略。把观察到的差异写成可检验机制假设，而不是直接归因于“层次更好”。

#### 原文与相关入口

- [论文](https://arxiv.org/abs/2410.20092)：ICLR 2025；环境、数据与基线定义。
- [作者基准库](https://github.com/seohongpark/ogbench)：数据获取、环境与统一算法实现。

#### 作者代码

[基准作者维护的官方实现。](https://github.com/seohongpark/ogbench)

离线目标环境、数据集与标准化基线。

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

### General Agents Contain World Models

Jonathan Richens, David Abel, Alexis Bellot, Tom Everitt

ICML 2025 · 2025 · 支持方法与理论

#### 研究问题

能完成足够丰富的目标集合，是否意味着智能体内部已经包含可提取的环境预测知识？

#### 关键机制

论文在形式化条件下，将广泛多步目标上的行为能力与环境模型的可提取性联系起来。通过查询智能体对不同目标的行为，可以恢复关于环境后果的信息；目标集合和性能要求越强，所要求的预测知识也越强。

#### 证据

主要证据是给定假设下的理论结果，而不是某个世界模型架构在所有任务上击败无模型算法的实验。

#### 条件与限制

可提取模型不等于智能体显式保存一个 RSSM，也不意味着所有实用任务都需要重建全部环境。必要知识的结论不能代替如何高效学到它的算法。

#### 阅读与实验

列出定理要求的目标丰富性和查询能力，再尝试构造一个只会单一任务的反例。由此区分任务专门知识与支持广泛目标的预测模型。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2506.01622)：形式化设定、模型可提取性与证明。
- [David Abel 论文目录](https://david-abel.github.io/papers.html)：作者提供的 ICML 2025 发表信息及相关研究。

### The OaK Architecture: A Vision of SuperIntelligence from Experience

Richard S. Sutton

RLC 2025 讲座 / Oak Lab · 2025 · 定义与架构观点

#### 研究问题

持续学习是否只是在一个现成 actor–critic 上加入抗遗忘机制，还是需要重新安排知识构造与使用？

#### 关键机制

OaK 提出从经验持续形成状态、预测知识、子任务、时间抽象与模型，并让这些知识服务规划的架构方向。这里的重点是模块之间怎样产生可复用知识，而不只是保留某个固定策略网络的参数。

#### 证据

官方页面提供 Richard Sutton 的架构讲座与相关研究入口。STOMP、预测学习和在线特征学习等论文可以检验其中具体组件，但不能自动验证整体架构。

#### 条件与限制

这是研究愿景与架构讲解，不是一套已公布完整训练配方、统一基准结果和可复现端到端代码的系统。资源分配、问题生成、知识替换与模块相互干扰仍需明确算法。

#### 阅读与实验

为每个模块写出输入、输出、更新频率和资源上限。再选择一个双模块接口做可证伪实验，例如技能模型改善是否真的减少规划误差。

#### 原文与相关入口

- [Oak Lab 官方讲座页面](https://oaklab.ai/posts/the-oak-architecture)：讲座入口与架构研究方向。
- [Oak Lab 研究主页](https://oaklab.ai/)：区分已发表研究、技术文章和仍在预告中的项目。

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

### Does Zero-Shot Reinforcement Learning Exist?

Ahmed Touati, Jérémy Rapin, Yann Ollivier

ICLR 2023 · 2023 · 支持方法与理论

#### 研究问题

没有事先指定奖励时，怎样学一套预测表示，日后接收新奖励就能选行为？

#### 关键机制

Forward–Backward 表示联合学习行为条件的未来占用与奖励读出，而非先固定任意编码器再学习 successor features。新奖励被映射到任务向量，策略根据这个向量直接行动；该论文系统比较 FB 与多种 SF 基础特征。

#### 证据

原文在固定离线 replay buffers 上比较零样本任务迁移，借此把表示学习与探索数据的质量分开。不同特征与数据覆盖产生显著差异，不能仅靠“所有奖励”的理论目标预测实际效果。

#### 条件与限制

假定共享动力学与可用经验覆盖。无下游梯度更新不等于无预训练成本；有限秩、近似训练和奖励估计都有误差。新动力学、历史混叠和严格一次使用经验均须另测。

#### 阅读与实验

同一 buffer 对比随机特征、谱特征与联合 FB，再独立换 buffer。奖励读出误差、占用误差与新任务回报分别报告，避免把数据覆盖优势记成表示优势。

#### 原文与相关入口

- [作者原文](https://arxiv.org/abs/2209.14935)：2022 首稿，ICLR 2023；比较奖励表示、SF 与 FB。
- [作者研究平台](https://github.com/facebookresearch/controllable_agent)：README 直接关联两篇 FB 论文；该仓库已经归档。

#### 作者代码

[论文作者团队仓库；归档工程，依赖和旧环境需单独核验。](https://github.com/facebookresearch/controllable_agent)

FB 与 SF 的训练、固定数据实验及奖励查询示例。

### Foundation Policies with Hilbert Representations

Seohong Park, Tobias Kreiman, Sergey Levine

ICML 2024 · 2024 · 支持方法与理论

#### 研究问题

如何从无任务标签的离线轨迹形成既能按方向调用、又能用于目标任务的策略接口？

#### 关键机制

HILP 先学习近似保存时间距离的 Hilbert 表示，再以潜在位移与方向的内积训练方向条件策略。新任务通过奖励回归、目标方向或分层调用选择策略条件，结构表示也支持测试时规划。

#### 证据

ICML 原文与作者项目包含零样本 RL、离线目标条件 RL 及规划实验；官方仓库将 zero-shot 与 goal-conditioned 两套实现分开。

#### 条件与限制

精确时间距离不总能无损嵌入有限维对称欧氏距离，尤其有向不可逆行为；理论充分条件与近似神经实验需区分。方向条件策略没有自动获得任意停止条件或完整技能后果模型。

#### 阅读与实验

固定离线数据分别测距离误差、方向执行误差、奖励可表达误差与高层收益。让同一视觉观测对应不同历史，检查仅观测编码是否足够，之后再讨论 CRL 状态维护。

#### 原文与相关入口

- [ICML 2024 原文](https://proceedings.mlr.press/v235/park24g.html)：Hilbert 距离、策略提示和定理前提；不是 ICLR 论文。
- [作者项目与公式](https://seohong.me/projects/hilp/)：时间距离与方向奖励接口。
- [官方实现](https://github.com/seohongpark/HILP)：hilp_zsrl 与 hilp_gcrl 对应不同实验。

#### 作者代码

[作者项目直接链接并标为 official implementation。](https://github.com/seohongpark/HILP)

离线预训练、零样本奖励适配及目标条件实验。

### Constructing an Optimal Behavior Basis for the Option Keyboard

Lucas N. Alegre, Ana L. C. Bazzan, André Barreto, Bruno C. da Silva

NeurIPS 2025 · 2025 · 支持方法与理论

#### 研究问题

状态相关的技能组合足够强时，是否仍须为每个新奖励保存一条完整策略？

#### 关键机制

OKB 联合扩充基础策略与 Option Keyboard 的元策略。线性支持方法选择需要补齐的任务权重，先训练现有基础上的组合，再检查无法表达的动作并新增基础，移除冗余项。优化的是可组合的行为基，而非仅增加技能数量。

#### 证据

正式原文分析基础数量与 convex coverage set 的关系，并在多任务领域检验规模与表现。附录提供元策略、新基础训练和角点枚举等实现细节；论文声明实验代码在 Supplemental Material。

#### 条件与限制

保证假定 NewPolicy 返回最优策略，且 TrainOK 能达到可表达的最优组合。近似 critic、有限训练、变化动力学不直接继承保证；非线性任务结论仅覆盖最优行为可由相应子策略组合的类别。

#### 阅读与实验

分别比较“增加基础”“只训练组合”和“去除冗余”。保留一组未用于基构造的奖励权重，记录基础规模、元策略成本、SF 误差与迁移回报。持续淘汰仍需未来任务效用检验。

#### 原文与相关入口

- [NeurIPS 2025 原文与补充材料入口](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0ab48777def88e73b50746a6011be0b0-Abstract-Conference.html)：算法 1–3、附录 A.3 的两个最优子程序假设及代码声明；未在本教材运行补充代码。
- [Option Keyboard 的经典桥梁](https://proceedings.neurips.cc/paper/2019/file/251c5ffd6b62cc21c446c963c76cf214-Paper.pdf)：cumulant 组合、GPE/GPI 与技能接口。


<a id="chapter-code"></a>

## 下载与运行

目标重标记、子任务 TD 与有限目标课程的表格实验；深度方法另附原论文和作者工程。

[下载 knowledge_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/knowledge_algorithms_lab.py)

```sh
python3 knowledge_algorithms_lab.py goals
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Schaul et al. — Universal Value Function Approximators](https://proceedings.mlr.press/v37/schaul15.html)：目标作为价值函数输入，状态与目标通过共享表示实现泛化；比较原文表示分解与直接拼接输入的实现。

- [Andrychowicz et al. — Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)：原文。对照 future 策略、off-policy 学习器与稀疏奖励实验；重标记不等于改变真实成功率。

- [OpenAI Baselines — her_sampler.py](https://github.com/openai/baselines/blob/master/baselines/her/her_sampler.py)：作者团队原工程入口。逐项追未来 achieved goal、目标替换和 reward_fun；依赖与 done 约定需同时检查。

- [Florensa et al. — Automatic Goal Generation for RL Agents](https://proceedings.mlr.press/v80/florensa18a.html)：原文与项目入口。Goal GAN 学的是适中难度目标分布；它不是 HER，也不是低层控制器。

- [Goal generation — 作者 rllab-curriculum](https://github.com/florensacc/rllab-curriculum)：课程层与策略层的实现：目标生成、成功率标签、策略训练各自改变什么。旧依赖适合放在独立环境。

- [Sutton et al. — Reward-Respecting Subtasks](https://arxiv.org/html/2202.03466v3)：式(2)、(5)、(9)分别给出 return、TD target 和停止规则；式(12)之后另定义真实 option model。

- [Park et al. — HIQL · NeurIPS 2023](https://arxiv.org/abs/2307.11949)：从同一个目标价值提取高层子目标与低层动作，分析长期价值噪声对平坦和分层策略的影响。

- [HIQL 作者实现](https://github.com/seohongpark/HIQL)：比较目标采样、价值学习、低层优势和高层跨步优势；way_steps 对应中间目标间隔。

- [HIQL 核心文件 — src/agents/hiql.py](https://github.com/seohongpark/HIQL/blob/master/src/agents/hiql.py)：compute_actor_loss 与 compute_high_actor_loss 分别构造一步和跨步价值差；compute_value_loss 做 expectile 更新；pretrain_update 组合三项损失。

- [OGBench · ICLR 2025](https://arxiv.org/abs/2410.20092)：离线目标条件 RL 的任务和数据设计；把长时域、拼接、观测和随机性等困难分开。

- [OGBench 作者基准与基线](https://github.com/seohongpark/ogbench)：环境、数据接口与算法实现；尤其注意 masks 与 terminals 分别表示什么。

- [Klissarov et al. — MaestroMotif · ICLR 2025](https://arxiv.org/abs/2412.08542)：技能描述、偏好奖励、程序化规则及低层 RL 的完整分工；实验重点是 NetHack 中技能的训练与组合。

- [MaestroMotif 作者实现](https://github.com/mklissa/maestromotif)：preference、code_generation、rl_baseline 和 sample_factory 对应奖励、组合规则与执行策略学习。


---

# Options：多步决策、技能发现与可复用行为

怎样把连续多步的行为当成可复用的决策单位，同时仍能在每个原始时间步学习？

## 本章内容

- 从原始回报拆出 SMDP target，而不是机械地把动作 a 换成 option o。
- 理解 SMDP Q-learning、intra-option learning 和 Option-Critic 分别更新什么。
- 推导 option 内策略与终止函数的两种梯度，正确处理停止、换技能和环境结束。
- 按覆盖、可区分性、可预测性、主任务价值区分发现技能的算法线。

<a id="problem-definition"></a>

## 本章的问题定义

以闭环多步行为为决策单位；给定技能时选择技能，学习技能时还要更新内部动作与终止。

### 给定条件与符号

- Markov任务、原始奖励和折扣，以及可启动技能集合或技能参数化。
- 每步转移、当前技能标识、开始状态、累计折扣奖励、时长与计算预算。

### 需要求解的对象

给定技能集合上的高层策略/价值，或在声明发现准则下学习内部策略和停止函数；技能发现准则不自动等于主任务目标。

### 信息与数据权限

$o=(I_o,\pi_o,\beta_o)$ 分别规定启动集、内部动作策略与到达后的停止概率；高层 $\mu$ 只在技能停止后重选，$\tau\ge1$ 为原始步时长。

$$
Q^*_{\mathcal O}(s,o)=\mathbb E_o\!\left[\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}+\gamma^\tau\max_{o'\in\mathcal O(S_{t+\tau})}Q^*_{\mathcal O}(S_{t+\tau},o')\mid S_t=s\right]
$$

$\mathcal O(s)$ 为在状态 $s$ 可启动的固定技能集合，$\gamma$ 按原始步折扣，真实终止的尾项为0。此最优性限于集合；评价给定 $\mu$ 时将最大值换成其动作平均。Option-Critic改变技能参数，模型和发现问题需另定义。

### 成立条件与解的含义

- 基础有限折扣任务中每个非终止状态有可启动行为；完整技能样本需相应终止/可积条件。
- Intra-option的行为纠偏需要动作支持；技能参数持续改变时原固定技能理论不直接适用。

判断准则：两步奖励1、2和终点价值10、折扣0.9时跨步target为10.9；一步技能退化为普通控制；检查启动mask、终止梯度方向、技能多样性与主任务收益。

### 适用边界

- 技能停止不等于环境终止。
- 扩大技能集合在精确问题中的潜在收益不保证有限学习和规划成本后的收益。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：相对完整持续控制，本章先限定Markov任务和具有启动、执行、停止接口的行为类；它在这个局部设定内再将一步动作推广为随机时长技能。

- 组合不同学习问题 · [目标条件化与子任务构造](https://yingwen.io/zh/continual-rl/construction/goals/)：子任务给出行为评价标准，技能将它落实为策略、启动与停止。

- 组合不同学习问题 · [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)：可执行技能还需后果模型才能用于模型规划；技能改变会改变模型题目。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

高层跨多步才收到反馈，内部行为又在每步执行；技能终止与环境终止不能使用同一边界。

### 本章的核心思路

从原始回报按技能边界拆分，再把继续/停止分支展开为一步接口。

1. [保留随机时长折扣](#lesson-derive)：因为技能消耗多个原始步，SMDP标签使用内部折扣奖励与实际时长的尾折扣。

2. [每步估计相容技能](#lesson-intra)：因为不用等技能完整结束，一步arrival value混合继续和高层重选，动作比率处理其他技能的行为差异。

3. [分别优化内部动作与停止](#lesson-critic)：因为执行什么与何时交回高层是两种选择，Option-Critic用动作score与继续—切换优势构造不同梯度。

结论与条件：固定技能精确SMDP与相应表格学习有明确条件；Option-Critic为参数化目标的梯度结构，不保证技能多样性、全局最优或可迁移。

### 相关方法改变了什么

- SMDP Q-learning：技能完整结束后更新高层价值，等待时间明确。

- Intra-option：每个原始步更新相容技能的价值，仍可固定内部策略。

- Option-Critic/发现方法：前者用主任务梯度学策略与停止；谱/互信息等发现采用另外的准则。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### Primitive action 与 option

primitive action 是环境接收的一步指令。option $o=(I_o,\pi_o,\beta_o)$ 包含三个部分：$I_o$ 规定允许启动的状态；$\pi_o(a|s)$ 选择当前动作；$\beta_o(s')$ 给出到达下一状态后停止的概率。停止 option 不等于结束环境。

### Call-and-return 执行

高层策略 $\mu(o|s)$ 选择一个 option；低层反复按 $\pi_o$ 行动，直到按 $\beta_o$ 停止；随后高层重新选择。高层决策之间可能间隔一个或多个原始时间步。

### 价值与动作优势

$Q(s,o)$ 是从 $s$ 启动 $o$ 然后按高层策略继续的价值；$V(s)=\sum_o\mu(o|s)Q(s,o)$。给定高层策略的评价使用这个加权平均；高层最优控制才使用合法 options 上的最大值。

### Log-derivative

策略概率对参数的导数可写成概率乘 log 概率的导数，因此能用采样动作估计求和。优势 baseline 不依赖当前动作时，可减少方差而不改变局部期望。

$$
\nabla_\theta\pi_\theta(a|s)=\pi_\theta(a|s)\nabla_\theta\log\pi_\theta(a|s)
$$

<a id="lesson-setting"></a>

## 1. 从一步动作到闭环的多步行为

仓库导航中，“向右一次”只能跨一个格子；“走到门口”可能经过十个格子，而且途中每次观测后都调整动作。因此 option 不是预先固定、不可反馈的动作序列：它通常是闭环策略与停止规则。若门被堵住，同一 option 的执行路径和耗时都会改变。时间抽象把高层的搜索与信用分配单位变粗，但低层仍然每步感知和控制。

先考虑有限 Markov 状态、有界奖励和固定折扣 $0\leq\gamma<1$。每个 option 至少执行一步，每个非终止状态有可启动的 option；使用完整轨迹估计时，所执行的 option 还须几乎必然停止。数据包含每个原始步的 $(s,o,a,r,s')$，以及本次 option 的开始状态、累计折扣奖励和持续时间。后两项决定了高层决策的正确更新。

| 阶段 | 已知什么 | 学习什么 |
| --- | --- | --- |
| 给定技能的高层控制 | I、π、β 固定 | 在当前状态选哪个 option |
| Intra-option 估值 | 一个真实动作可与多个 option 相容 | 每步更新多个 option 的价值 |
| Option-Critic | 指定技能数及可微参数化 | 内部策略、停止函数与高层价值 |
| 技能发现 | 只给世界经验和发现准则 | 哪些不同的行为值得进入技能库 |

在 CRL 中，技能的复用取决于变化发生在哪一层。奖励改变可能只需重新估价；环境动力学改变可能要求重学技能；技能策略改变会使旧后果模型过期；表示改变可能同时影响低层策略与模型输入。因此，持续维护技能库还包括评价技能适用性、更新后果模型，以及替换失效或冗余的技能。

<a id="lesson-derive"></a>

## 2. SMDP：由原始回报推导跨多步更新

假设在 $t$ 启动 $o$，在 $t+\tau$ 停止。把回报沿这个边界拆开，前半段是执行 option 期间得到的奖励，后半段是从停止状态开始的后续回报。后半段隔了 $\tau$ 个原始步，因此乘 $\gamma^\tau$。持续时间可以随机；每次样本都使用其实际 $\tau$。

$$
\begin{aligned}G_t&=\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}+\gamma^\tau G_{t+\tau}\\ Q(s,o)&=\mathbb E\!\left[\widehat R_o+\gamma^\tau V(S_{t+\tau})\mid S_t=s,O_t=o\right]\end{aligned}
$$

第二行对第一行取条件期望。R̂_o 是本次 option 内的折扣奖励和，不是平均奖励，也不是子任务内在奖励，除非你明确改变了目标。

$$
Q(s,o)\leftarrow Q(s,o)+\alpha\left[\widehat R_o+\gamma^\tau\max_{o'\in I(s')}Q(s',o')-Q(s,o)\right]
$$

这是给定 options 时的 SMDP Q-learning。I(s′) 表示在 s′ 可启动的 options。真实环境终止时后项为 0；仅 option 结束时后项仍存在。

**算法：算法伪代码**

1. 初始化 Q；每次高层选择合法 o，保存 `s_start`
1. `R_sum=0`；`discount=1`；`duration=0`
1. 循环：
  1. 按 $π_o(a|s)$ 行动，观察 r,s′,environment_done
  1. `R_sum += discount*r`；$\mathrm{discount}←γ\mathrm{discount}$；`duration += 1`
  1. 若 environment_done，或在 s′ 按 $β_o(s^{\prime})$ 抽样结束：
    1. continuation = 0（环境终止）否则 $\max_{o^{\prime}\in\mathcal O(s^{\prime})} Q(s^{\prime},o^{\prime})$
    1. $Q(s_{start},o)←Q(s_{start},o)+α(R_{sum}+\mathrm{discount}\,\mathrm{continuation}-Q(s_{start},o))$
    1. 若环境未结束，在 s′ 重新选 option
  1. 否则保留当前 option；s=s′

这种估值要等 option 结束才完成一次更新；长技能会延迟信用，几乎不停止的技能也可能妨碍高层探索。若每个 option 恰好执行一步，该算法退化为普通 Q-learning：内部奖励仅有一项，后续折扣恰为 $\gamma$。这个特例可以用来检查奖励和时间索引。

<a id="lesson-intra"></a>

## 3. Intra-option：不等执行完，立即学习多个技能

到达 $s'$ 后，以概率 $1-\beta_o(s')$ 继续当前 $o$，以概率 $\beta_o(s')$ 结束并交回高层。对两个分支求平均，得到 arrival value $U$。因此它的定义直接来自 option 的执行规则。

$$
\begin{aligned}U(s',o)&=(1-\beta_o(s'))Q(s',o)+\beta_o(s')V(s')\\ Q(s,o)&=\sum_a\pi_o(a|s)\mathbb E\!\left[R+\gamma U(S',o)\mid s,a\right]\end{aligned}
$$

只走一个原始步，故这里乘一个 γ。β=0 恢复继续同一 option；β=1 恢复马上让高层选择。环境终止则 U=0，无论 β 预测多少。

若真实行为的动作概率为 $b(a|s)$，用这个动作估计另一 option $\pi_o$ 的动作平均时，可乘 $\rho_o=\pi_o(a|s)/b(a|s)$。确定性 options 中，只有同样会选择刚才动作的候选有贡献，其他候选的比率为零。在表格、固定 options 的条件下，经验因而可在多个行为问题间共享。

$$
\begin{aligned}\delta_o&=r+\gamma U(s',o)-Q(s,o)\\ Q(s,o)&\leftarrow Q(s,o)+\alpha\rho_o\delta_o,\qquad \rho_o=\frac{\pi_o(a|s)}{b(a|s)}\end{aligned}
$$

所有 target 先使用同一份旧 Q 计算，特别是 s=s′ 自环时。ρ 修正动作分布；访问不到的状态不会凭空学到，函数逼近与 off-policy 的稳定性也需另行处理。

高层控制使用 $V(s')=\max_{o':s'\in I_{o'}}Q(s',o')$，最大值只遍历在 $s'$ 合法启动的 options。继续当前 option 的 $Q(s',o)$ 不受启动 mask 限制，因为它已在更早的合法位置开始执行；只有终止后重新选择的分支受限。若评价给定 $\mu$，则在合法集合上按 $\mu$ 求加权平均。Intra-option 更新的是固定技能的价值，内部策略是否有用、如何学习，还要由下一节解决。

<a id="lesson-critic"></a>

## 4. Option-Critic：内部动作与何时停止是两个梯度

为了学习内部动作，定义 $Q_U(s,o,a)$：当前处于 $o$ 时，先执行 $a$，再遵守 option 的继续、停止与高层选择规则的回报。它比 $Q(s,o)$ 多条件化了当前动作；$U$ 则描述到达下一状态后的混合价值。三者可以共享网络表示，但对应不同条件。

$$
\begin{aligned}Q_U(s,o,a)&=\mathbb E[R+\gamma U(S',o)\mid s,a]\\ Q(s,o)&=\sum_a\pi_{o,\theta}(a|s)Q_U(s,o,a)\end{aligned}
$$

第一式给 critic 的 TD target，第二式把内部策略的动作平均还原成 option 价值。实际工程也常用 r+γU 作为当前 $Q_U$ 的样本估计，未必单独存一个完整三维表。

固定 critic 作为局部评价器，对第二式中的当前动作概率求导，再将以后状态的递归影响展开，得到沿状态-option 占用分布加权的策略梯度。我们不需要显式微分环境转移，但必须在正确的轨迹/占用分布上采样。以下写出精确目标所对应的结构，再写常用的单样本方向。

$$
\begin{aligned}\nabla_\theta J&=\sum_{s,o}d_\gamma(s,o)\sum_a\nabla_\theta\pi_{o,\theta}(a|s)\,Q_U(s,o,a)\\ \Delta\theta&=\alpha_\theta\nabla_\theta\log\pi_{o,\theta}(a|s)\,[\widehat Q_U(s,o,a)-b(s,o)]\end{aligned}
$$

$d_\gamma$ 是从指定起点出发的折扣占用权重；$b$ 不依赖当前动作。第一式对应精确目标，第二式给出单个样本的更新方向；其无偏解释需要相应采样权重，普通逐步在线实现常以实际访问分布近似。

终止梯度可直接从混合式推导。保持当前 $Q$ 与 $V$ 不动，对 $U=(1-\beta)Q+\beta V$ 求终止参数导数，得到 $\nabla\beta(V-Q)$。当 $Q<V$ 时，增大停止概率能提高局部价值；反之应鼓励延续。递归展开这些贡献后得到终止梯度定理，其占用权重对应到达状态。

$$
\begin{aligned}A_\Omega(s',o)&=Q(s',o)-V(s'),\\ U_{\rm loc}(s',o;\vartheta)&=(1-\beta_{o,\vartheta}(s'))\operatorname{sg}(Q(s',o))+\beta_{o,\vartheta}(s')\operatorname{sg}(V(s')),\\ \nabla_\vartheta U_{\rm loc}(s',o;\vartheta)&=-\nabla_\vartheta\beta_{o,\vartheta}(s')\,\operatorname{sg}(A_\Omega(s',o)),\\ \Delta\vartheta&=-\alpha_\beta\nabla_\vartheta\beta_{o,\vartheta}(s')\,\widehat A_\Omega(s',o).\end{aligned}
$$

$\operatorname{sg}$ 表示在当前局部更新中固定 critic。这个局部偏导不是包含未来 $Q,V$ 参数依赖的完整 $\nabla_\vartheta U$。若 $\beta=\sigma(h)$，则 $\partial\beta/\partial h=\beta(1-\beta)$。当前 option 优势为负时，更新增大 $h$，使它更容易停止；优势为正时则鼓励继续。真实环境终止后不再存在继续与停止的选择。

$$
\nabla_\vartheta J=-\sum_{s',o}d_\gamma^{\rm arrival}(s',o)\,\nabla_\vartheta\beta_{o,\vartheta}(s')\,A_\Omega(s',o)
$$

$d_\gamma^{\rm arrival}(s',o)=\sum_{t\ge0}\gamma^{t+1}\Pr(S_{t+1}=s',\Omega_t=o)$ 是从指定初始化出发、在非终止到达状态上定义的未归一化折扣占用；$\Omega_t$ 表示正在执行的 option。该式把未来重复出现的局部终止选择展开后才得到完整目标梯度。求导时固定高层和内部动作策略参数；若共享参数，还需合并相应梯度路径。

**算法：算法伪代码**

1. 初始化 $π_o,β_o,Q$ 或 $Q_U$；选择当前 option o
1. 每一步：
  1. 按旧 $π_o$ 行动，记录 s,o,a,r,s′ 与真实 terminal
  1. 缓存旧 $Q(s^{\prime},o),V(s^{\prime}),β_o(s^{\prime})$，构造 $U$ 和 $y=r+γU$
  1. 用 y 更新 critic（真实 terminal 时 y=r）
  1. 用缓存的动作优势更新 $π_o$ 的 log-probability
  1. 若非 terminal，用 $-∇β_o(s^{\prime})[Q(s^{\prime},o)-V(s^{\prime})]$ 更新停止参数
  1. 按明确规定的 β 版本抽样停止；停止才按高层 μ 重新选 o
  1. s=s′；持续记录技能长度、选择频率、动作熵与主任务回报

实现中可先缓存更新前的 $\beta$ 与 critic，利用同一次转移构造两个 actor 的更新，再按缓存的终止概率决定是否换技能。这样执行规则和训练标签具有一致的时间顺序。动作熵、终止正则和切换代价可以减少技能塌缩或频繁切换，但它们改变的是优化目标，须与上面的基本梯度分别记录。

<a id="lesson-example"></a>

## 5. 从原始轨迹到两个梯度：完整手算

跨步估值：某 option 连续得到奖励 $1,2$，两步后停止；$\gamma=0.9$，停止状态的高层价值为 $10$。内部奖励和为 $1+0.9\times2=2.8$，后续贡献为 $0.9^2\times10=8.1$，总 target 为 $10.9$。若环境也在此终止，总 target 才变为 $2.8$。

一步 intra-option：$Q(s',o_1)=2$、$V(s')=6$、$\beta_1=0.25$，故 $U_1=0.75\times2+0.25\times6=3$。奖励为 $1$ 时 target 为 $3.7$。若 $Q(s,o_1)=0$、$b(a|s)=0.5$、$\pi_1(a|s)=1$、$\alpha=0.1$，则 $\rho_1=2$，更新后价值为 $0.74$。第二个 option 若 $\beta_2=1$、$\pi_2(a|s)=0.5$，则 target 为 $6.4$、$\rho_2=1$，更新后价值为 $0.64$。同一真实动作产生了两个不同控制问题的样本更新。

内部策略梯度：两个动作 logits 都为 0，故概率各为 0.5；抽到动作 0，动作价值估计为 3、baseline 为 1。softmax 的 log 概率梯度为 (0.5,−0.5)，乘优势 2 与步长 0.1，得到新 logits (0.1,−0.1)，动作 0 的概率变为约 0.549834。

终止梯度：到达状态的 $Q=2$、$V=6$，option 优势为 $-4$。终止 logit $h=0$，$\beta=0.5$、导数为 $0.25$；步长 $0.1$ 给出 $\Delta h=-0.1\times0.25\times(-4)=0.1$，新停止概率约 $0.524979$。内部 actor 在 $s$ 更新动作选择，终止 actor 在 $s'$ 更新是否继续。

<a id="lesson-code"></a>

## 6. 实现与实验：检查三个时间尺度

跨步 target、arrival value、多个 option 的一步估值、两种 actor 更新与技能内在奖励的最小实现。

```python
def option_target(rewards, gamma, next_value, terminal=False):
    if not rewards:
        raise ValueError("An option must execute at least one primitive step")
    accumulated = sum(gamma**k * r for k, r in enumerate(rewards))
    return accumulated + (0.0 if terminal else gamma**len(rewards) * next_value)


def arrival_value(q_continue, v_select, beta, terminal=False):
    return 0.0 if terminal else (1.0 - beta) * q_continue + beta * v_select


def intra_option_update(q, state, next_state, reward, beta_next,
                        target_action_prob, behavior_action_prob,
                        alpha=0.1, gamma=0.9, terminal=False,
                        next_initiation_mask=None):
    """One observed action updates every fixed option with support.

    Dictionaries q[s] are vectors over options. q snapshots make targets
    simultaneous, including the case state == next_state. The initiation mask
    restricts NEW option selections at next_state, not continuation of an
    already active option. Omit it only if every option can initiate there.
    """
    if behavior_action_prob <= 0.0:
        raise ValueError("Observed action must have positive behavior probability")
    here, nxt = q[state][:], q[next_state][:]
    mask = [True] * len(nxt) if next_initiation_mask is None else next_initiation_mask
    if len(mask) != len(nxt):
        raise ValueError("Initiation mask must match the number of options")
    if not terminal and not any(mask):
        raise ValueError("A nonterminal selection state needs a legal option")
    value = 0.0 if terminal else max(v for v, legal in zip(nxt, mask) if legal)
    for o in range(len(here)):
        rho = target_action_prob[o] / behavior_action_prob
        continuation = arrival_value(nxt[o], value, beta_next[o], terminal)
        delta = reward + gamma * continuation - here[o]
        q[state][o] = here[o] + alpha * rho * delta
    return q[state]


def softmax(logits):
    maximum = max(logits)
    exps = [math.exp(x - maximum) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def option_critic_actor_step(logits, chosen_action, q_action, baseline,
                             termination_logit, q_continue, v_select,
                             alpha_actor=0.1, alpha_beta=0.1):
    """Local sampled ascent directions; caller supplies OLD critic estimates.

    At environment terminal the caller must skip the termination update.
    This is not a complete deep Option-Critic training system.
    """
    probs = softmax(logits)
    advantage_action = q_action - baseline
    new_logits = [h + alpha_actor * advantage_action *
                  ((1.0 if a == chosen_action else 0.0) - probs[a])
                  for a, h in enumerate(logits)]
    beta = sigmoid(termination_logit)
    advantage_option = q_continue - v_select
    new_termination = termination_logit - alpha_beta * beta * (1.0 - beta) * advantage_option
    return new_logits, new_termination


def skill_intrinsic_rewards(log_q_z_given_s, log_prior_z,
                            log_q_next_given_skill, log_mixture_next):
    """Mechanism only: no discriminator, SAC, density training or entropy term."""
    return (log_q_z_given_s - log_prior_z,
            log_q_next_given_skill - log_mixture_next)
```

标准库运行；测试包含局部梯度，以及双 option 链完整折扣目标的终止梯度有限差分

```sh
python3 knowledge_algorithms_lab.py options
python3 knowledge_algorithms_lab.py test
```

完整梯度测试使用一个持续状态和两个 option。内部动作每步分别产生 $0$ 与 $1$，高层以相同概率重新选择，初始执行 option 0。先精确解出二阶 Bellman 线性方程，再对初始收益作参数有限差分。对照量是到达占用加权的终止梯度，不是单个状态上的局部偏导；遗漏到达前那一步的 $\gamma$ 会使测试失败。

运行结果对应前面的手算：跨步 target 为 $10.9$，arrival value 为 $3$，两个估值为 $[0.74,0.64]$，动作 logits 为 $[0.1,-0.1]$，终止 logit 为 $0.1$。还应检查启动集合：下一状态价值为 $[1,100]$、只有第一个 option 可启动时，重新选择的价值是 $1$；第二个 option 若早已开始且 $\beta=0$，继续执行的价值仍可为 $100$。代码中的 sigmoid 分支避免了极端 logit 的指数溢出。

这份表格实现适合逐步观察更新顺序。进一步研究深度 Option-Critic 时，作者 Atari 仓库提供卷积网络、目标网络、策略与终止损失以及完整交互循环；可沿相同的三个接口阅读：跨步选技能、原始步选动作、到达状态判断停止。

- 改动 A：把所有 β 设成 1，并限制一个 option 对应一个 primitive action；验证 target 退化到 Q-learning。
- 改动 B：把当前 option 优势从 −4 改为 +4；停止 logit 应下降。再把其设为 0，更新应为零。
- 改动 C：构造 s=s′ 自环，依次更新两个 options；比较未缓存旧 Q 的实现与本页同时 target 的差异。
- 改动 D：增加 option 平均长度，固定真实步数和计算预算，比较估值误差与控制回报；不能只按“高层决策次数”比较样本效率。

<a id="lesson-branches"></a>

## 7. 技能发现首先要选择什么行为值得学习

Option-Critic 直接用当前任务回报优化技能；另一类发现方法在没有当前任务奖励时，先积累覆盖环境或具有可区分效果的行为。它们与 option 形式兼容，但并不都学习可变的终止条件，也不都提供规划所需的 reward/end-state 模型。比较之前先问：发现信号是什么，技能怎样被调用，何时结束，下游需要什么接口？

结构发现、互信息发现和任务回报优化回答的问题不同。图结构方法寻找能跨越环境大尺度区域的行为；DIAYN 寻找可被区分的行为；DADS 进一步要求行为后果可预测；Option-Critic 则直接改进当前任务回报。先看后两种无监督准则，再推导 Machado 一线如何从环境结构得到技能。

$$
r_{\rm DIAYN}(s,z)=\log q_\phi(z|s)-\log p(z)
$$

DIAYN 采样技能标识 z，让条件策略产生能被判别器区分的状态。qφ 猜“哪个技能产生了这个状态”，固定先验 p 保持技能使用多样性；策略训练还包含动作熵项。这里不是外部目标奖励，也不自动约束一个技能的终点。

其最小训练循环是：采样 z 并固定一段交互；保存 (s,a,s′,z)；训练判别器预测 z；用当前判别器构造内在奖励训练条件策略；在统一外部任务上评价技能复用。需警惕技能仅凭无关背景区分、所有技能起点泄漏标签、判别器与策略相互追逐。技能可分辨并不自动意味着有用。

$$
r_{\rm DADS}(s,z,s')=\log q_\phi(s'|s,z)-\log\!\left(\sum_{z'}p(z')q_\phi(s'|s,z')\right)
$$

DADS 比较特定技能下状态变化的可预测性与混合技能模型。离散求和也可用先验采样近似；连续技能需要相应积分/采样。它同时学习技能动力学，因而比只有技能判别器更直接提供模型控制接口。

| 准则 | 怎样学习 | 对 CRL 的价值与缺口 |
| --- | --- | --- |
| 主任务回报 / Option-Critic | critic + 内部策略梯度 + 终止梯度 | 能面向当前任务优化；可能缺少将来任务需要的多样性 |
| 图结构 / eigenoptions | 图特征 → 内在差分奖励 → 策略与停止 | 有利于结构覆盖；不自动保留真实沿途奖励 |
| 可区分性 / DIAYN | 技能条件策略 + 状态判别器 + 熵 | 获得不同技能；没有自动得到完整规划模型 |
| 可预测后果 / DADS | 技能条件动力学 + 内在奖励 + 下游模型控制 | 使行为更易预测；仍需检查模型覆盖与任务奖励接口 |
| 未来可能有用 / reward-respecting | 真实奖励 + feature stopping value → option → model | 显式连接规划用途；候选特征与维护预算仍是研究问题 |

<a id="lesson-eigenoptions"></a>

## 8. Eigenoptions：把环境的慢变化方向变成行为

在有多个房间的迷宫里，随机游走很容易反复访问同一房间，却很久才穿过狭窄门口。因此，探索需要的不只是不同动作，还包括改变所在大区域的多步行为。Machado、Bellemare 与 Bowling 的 Laplacian option discovery（ICML 2017）用转移图的低频方向描述这种大尺度变化，再让策略主动沿这些方向移动。

$$
L_{\rm sym}=I-D^{-1/2}AD^{-1/2},\qquad L_{\rm sym}u_i=\lambda_i u_i,\qquad 0=\lambda_0\leq\lambda_1\leq\cdots
$$

A 是无向转移图的邻接权重，D 的对角线是节点度数。常数对应的平凡方向需按所用归一化转换后去除。低频特征在相连状态间变化缓慢，跨瓶颈却能形成明显差异；它描述的是图上的连通结构，而非观测像素距离。

选定一个非平凡状态特征 $e_i(s)$ 后，把沿该特征上升定义为子任务。对正、负两个方向分别训练，可以得到向相反区域移动的技能。策略仍按 RL 学习，并不是直接对特征做梯度就能得到环境动作。

$$
r_i(s,a,s')=e_i(s')-e_i(s),\qquad Q_i(s,a)\leftarrow Q_i(s,a)+\alpha\left[r_i+\gamma_i\max_{b\in\mathcal A\cup\{\perp\}}Q_i(s',b)-Q_i(s,a)\right]
$$

终止动作 ⊥ 的后续收益规定为零；当所有继续动作的内在价值都不大于零时，可以选择停止。技能的启动区域是仍值得继续的状态。$γ_i$ 是该发现问题自己的折扣，未必等于主任务折扣。

这里的差分奖励不是“保持原任务最优策略不变”的一般奖励塑形。策略不变塑形通常使用 $\gamma\Phi(s')-\Phi(s)$；eigenoption 则有意定义一个新的内在控制问题。它的价值在于生成长程行为，不在于保证该行为已经优化外部任务。

**算法：Eigenoptions 的表格流程；图特征、技能学习与下游评价是三个不同阶段**

1. 构造状态转移图，求低频非平凡特征 $e_i$
1. 对每个特征及其相反方向：
  1. 定义内在奖励 $r_i=e_i(s\prime)-e_i(s)$
  1. 加入收益为零的终止动作 $\perp$
  1. 用控制算法学习内部策略，并由继续价值决定停止区域
1. 将学到的 options 加入行为集合
1. 在相同真实交互预算下测量覆盖，再测下游任务回报

完整转移图在像素和连续状态中不可得。Machado 等人的 Deep Successor Representation（ICLR 2018）转向从经验学习多步后果表示。对固定策略，SR 的每一行记录未来折扣状态访问；它可以通过 TD 学习，而不必先收集所有边再完整求图。

$$
M^\pi=\sum_{k=0}^{\infty}\gamma^k(P^\pi)^k=(I-\gamma P^\pi)^{-1},\qquad M^\pi(s,:)\leftarrow M^\pi(s,:)+\alpha\left[\mathbf e_s+\gamma M^\pi(s',:)-M^\pi(s,:)\right]
$$

这里采用包含当前状态的访问约定，$e_s$ 是当前状态的 one-hot 向量。神经表示用特征替换 one-hot，再学习其未来累计。SR 与转移算子共享适当的谱结构；与对称 Laplacian 的对应还需要可逆性或合适的对称化，不能对任意有向动力学直接当作同一个矩阵。

由此得到一条清楚的变化：2017 年先有图再发现技能，2018 年开始从行为数据学习发现技能所需的结构。代价是表示依赖采样策略；没有访问过的房间，不会因为使用深度网络就自动出现在可靠的结构表示中。

<a id="lesson-discovery-cycle"></a>

## 9. 从静态特征到在线发现：ROD、DCEO 与 ALLO

技能改变探索，探索又改变表示，因而“先学表示、以后永久冻结技能”的分阶段方案并非唯一选择。Machado、Barreto、Precup 与 Bowling 在 JMLR 2023 的 Temporal Abstraction with the Successor Representation 中把这条反馈链称为 representation-driven option discovery：经验形成表示，表示定义子任务，子任务产生 options，options 再改变后续经验。

| 反馈环节 | 需要观察的量 | 可能失效的原因 |
| --- | --- | --- |
| 经验 → 表示 | 转移覆盖、谱方向、表示误差 | 只看已访问区域，遗漏窄门另一侧 |
| 表示 → 子任务 | 内在奖励分布、方向差异 | 冗余特征产生重复行为 |
| 子任务 → option | 成功率、执行长度、停止位置 | 低层训练不足或终止不合理 |
| option → 新经验 | 新区域访问与外部任务收益 | 高层过早偏爱熟悉技能，进一步缩小覆盖 |

DCEO（Deep Covering Eigenoptions，ICML 2023）把结构表示与技能发现放入深度 RL 的交互循环：从收集的转移学习 Laplacian 表示，以其方向构造探索技能，同时利用这些技能继续收集数据。它所推进的是大状态空间中在线获得覆盖性技能的问题；获得探索收益后，仍需单独学习外部任务价值或技能模型，才能讨论规划和任务复用。作者 mklissa/dceo 仓库提供了与深度 RL 基线相连的实现。

但“学到正确的低频子空间”还不等于“学到每一条明确的特征方向”。设两个坐标 $u_1,u_2$ 张成正确子空间，对它们作旋转仍可能得到相同的子空间损失；若技能奖励分别由两个坐标构造，旋转却会改变每个技能的行为。需要特征值缩放距离时，仅得到一个任意旋转的空间也不够。

$$
\min_{u_1,\ldots,u_k}\sum_{i=1}^{k}\langle u_i,Lu_i\rangle\quad\text{subject to}\quad\langle u_i,u_j\rangle=\delta_{ij}
$$

谱学习的基本形式同时要求平滑与正交。只最小化这一无序子空间目标，不能指定每个输出神经元对应哪个有序特征向量；这正是从子空间估计走向特征对估计时的区别。

Proper Laplacian Representation Learning（ALLO，ICLR 2024）通过带次序的约束、增广拉格朗日原始—对偶更新和特定停止梯度处理，学习有序的特征向量及特征值。它不是另一套 option 策略梯度，而是修复上游表示学习，使下游每个方向的含义与尺度更明确。作者 laplacian_dual_dynamics 的训练入口为 train_laprepr.py；读代码时应同时看向量损失、正交约束乘子和特征值读取，不能只截取平滑损失。

继续把任务价值加入高层选择，会发生什么？Value-Aware Eigenoptions（RLC 2025 Inductive Biases workshop）发现，固定 eigenoptions 能帮助信用分配，但在线发现时按任务价值选择技能也可能削弱探索覆盖。其核心实验把“已有技能怎么使用”和“后续技能从哪些经验中发现”分开。这个负面结果说明，任务回报、结构覆盖和长期技能维护之间需要明确的采样分工，而不是把高层控制器换成贪心就自然完成持续发现。

<a id="lesson-skill-composition"></a>

## 10. 从单方向技能到可组合技能：METRA 与 Laplacian Keyboard

Eigenoptions 常把一个特征方向对应为一个技能。如果下游目标位于两个方向之间，或者需要先沿一条方向再沿另一条方向运动，应如何组合？可以离散切换已有 options，也可以把连续的方向向量作为低层策略输入。后者把技能库变成可查询的条件策略族，但还需要规定方向的意义与执行时长。

METRA（ICLR 2024）从“可区分的技能可能仍然只在原地摆动”出发，让技能在与时间距离相关的潜在空间中产生大幅位移。它不是对 Laplacian 特征直接求特征分解，而是联合学习表示与方向条件策略，并用相邻状态的距离约束防止表示任意放大。

$$
\max_{\phi,\pi}\ \mathbb E\left[(\phi(S')-\phi(S))^\top Z\right],\qquad \|\phi(s')-\phi(s)\|_2\leq1\ \text{on observed adjacent states}
$$

这是理解 METRA 的受约束结构；实际算法用对偶与松弛处理约束。固定方向 Z 时，内积给出一步内在奖励；约束让累计潜在位移与需要经过的步数相联系。方向在采集一段轨迹时保持固定。

$$
\begin{aligned}\Delta\phi&=\phi(s')-\phi(s),\qquad d=\dim\phi(s),\\ c(s,s')&=1-\frac{\|\Delta\phi\|_2^2}{d},\qquad \widetilde c=\min\{c,\varepsilon_{\rm slack}\},\\ \mathcal L_\phi&=-\mathbb E\left[(\Delta\phi)^\top Z+\operatorname{sg}(e^b)\widetilde c\right],\\ \mathcal L_b&=b\,\mathbb E\left[\operatorname{sg}(\widetilde c)\right].\end{aligned}
$$

这是作者 iod/metra.py 中连续技能、dual_dist='one' 分支的实际尺度：平方差用维度平均而非求和，因此零残差对应 $\|\Delta\phi\|_2=\sqrt d$。$b$ 是对偶乘子的对数，$\varepsilon_{\rm slack}$ 对应 dual_slack；clamp(max=dual_slack) 只截断过大的正残差，不截断违反约束时的负残差，也不是双侧裁剪。两项损失分别最小化；负残差会增大 $b$，加强表示约束。理论逐对约束在实现中由样本上的软惩罚近似，不能将上述残差与单位欧氏球尺度直接等同。

METRA 在状态与像素控制任务中研究无奖励预训练和后续复用；这与长期变化环境中同时维护旧技能和学习新技能的完整 CRL 协议不同。作者 Seohong Park 的 METRA 仓库包含表示更新和 SAC 技能训练；实验时可比较“技能可区分但位移小”与“覆盖扩大”这两种不同结果。

Laplacian Keyboard（2026）把 ALLO 提供的谱特征与 successor features 联系起来。先在无任务奖励的经验上学习特征，再对不同权重向量训练相应的低层策略及其未来特征预测；下游既可由奖励回归得到一个固定权重，也可训练高层策略按当前状态选择权重，分段调用低层行为。

$$
\begin{aligned}r_w(s,a,s')&=w^\top\phi(s')\\ \psi(s,a,w)&=\mathbb E_{\pi_w}\left[\sum_{k=0}^{\infty}\gamma^k\phi(S_{t+k+1})\mid s,a\right]\\ Q^{\pi_w}_w(s,a)&=w^\top\psi(s,a,w)\end{aligned}
$$

每个 w 同时规定一个奖励方向和对应的目标策略。预测 ψ 时必须沿同一个 $π_w$ 递归，不能对各特征分量分别取最大后再组合。这里特征计在到达状态，避免与上一节包含当前状态的 SR 约定混淆。

**算法：Laplacian Keyboard 的分层接口；高层选择连续技能参数，低层实现原始动作**

1. 无奖励阶段：
  1. 从经验学习 Laplacian 特征 $\phi$
  1. 采样权重 $w$，训练奖励为 $w^\top\phi(s\prime)$ 的低层策略与 SF
1. 有奖励下游阶段：
  1. 高层在当前状态选 $w$；低层按 $\pi_w$ 执行一段时间
  1. 累加真实外部奖励，记录实际长度 $\tau$
  1. 段结束时用 $R_{\mathrm{sum}}+\gamma^\tau V(s\prime)$ 更新高层
  1. 再根据新状态选择下一个 $w$

为什么还要高层学习？若外部奖励不能由有限特征线性表示，固定 $w$ 的直接迁移会产生表达误差。即使拟合奖励的误差不超过 $\varepsilon$，对固定策略也只能得到价值误差至多 $\varepsilon/(1-\gamma)$：逐步奖励误差沿折扣级数累加。状态相关的分段组合扩大了行为表达能力，却不等于有限个谱特征已覆盖所有任务的最优策略。

与原始 Option Keyboard 的 GPI 组合相比，这里连续权重不仅用于给旧策略重新打分，还成为低层策略族的条件与高层的动作。与 Option-Critic 相比，其技能意义主要来自预训练的谱奖励，而不是完全由当前外部回报端到端塑造。论文附录 H 给出了低层训练和高层 SMDP 训练的伪代码，可重点追踪累计奖励、真实段长和环境终止三项。

至此，技能发现有了三条可分别检验的研究问题：表示是否保留环境长程结构；行为是否沿这些结构可靠执行；调用方式是否真正帮助新任务。接着还需要第四个问题：能否预测这些行为的后果，并据此规划？reward-respecting 子任务保留真实沿途奖励，option models 预测真实后果，STOMP 将两者接入规划；这些接口与纯覆盖性技能互补，而非简单的新旧替代。

<a id="research-options-behavior-basis"></a>

## 研究专题 A · 从给定技能库到自动补齐行为基

Option-Critic 优化当前任务中的内部动作和停止；谱发现提供覆盖性的候选行为；Option Keyboard 则问已有行为怎样组合。这里还缺一个问题：组合器已经训练充分，却依然无法产生某个必要动作时，是组合学习不足，还是基础行为缺失？OKB（NeurIPS 2025）把这一区别变成增量构造行为基的准则。

$$
\pi_{\rm OK}(s,w;\Pi)\in\arg\max_a\max_{\pi_i\in\Pi}\psi^{\pi_i}(s,a)^\top\omega(s,w)
$$

与固定新奖励权重 w 的 GPI 相比，元策略 ω 根据状态和任务选择组合方向。这里的 max 对固定基础策略的 SF 做评价；ω 的训练使用真实目标回报，不是让各 SF 坐标独立选择自己的最优未来。

例如一个递送问题需要先穿门，再向充电区移动。为整个任务选择单一奖励方向可能过早偏向充电；状态相关 ω 可以先选择过门方向，进门后再切换。若基础策略的 SF 在所有方向上都把“开门”排在其他动作后面，任何 ω 都无法恢复该动作，组合器的表达范围就成为瓶颈。此时增加训练步数与增加必要基础是不同操作。

$$
\mathcal A_{\Pi}(s)=\bigcup_{z\in\mathcal Z}\arg\max_a\max_i\psi^{\pi_i}(s,a)^\top z;\qquad A^{\pi_{\rm OK}}_w(s,a)=Q^{\pi_{\rm OK}}_w(s,a)-V^{\pi_{\rm OK}}_w(s)
$$

A_Π 描述当前基础通过任意方向能表达的动作。论文用已训练组合策略的正优势动作识别仍值得改善的行为；把这种诊断解释成缺失基础，要求组合训练已达到其表达范围内的最优，有限训练的正优势也可能仅是未学充分。

**算法：教学摘要；角点枚举、判定与子程序细节见原文算法和附录**

1. OKB 的结构：
  1. 从一个任务训练初始基础策略与 SF
  1. 根据现有 SF 的线性支持角点挑选任务权重
  1. 固定基础，训练状态/任务条件的组合元策略
  1. 检查仍无法表达的必要行为；若存在，训练一个新基础并加入
  1. 移除不再必要的基础，再更新支持集合
  1. 只有原文最优子程序和充分检查条件成立时，采用其最优基结论

定理中的 NewPolicy(w) 必须返回最优策略，TrainOK 必须找到可表达的最优组合；深度 actor–critic 的有限训练不能默认为满足这两个条件。原文非线性任务的扩展也要求最优行为可由相关线性任务的子策略构成，不能写成有限技能覆盖所有未来任务。

实验应将随机加基础、按覆盖加基础、仅训练组合与 OKB 构造分开，计入基础训练、SF 估计和元策略全部经验。若进入 CRL，再固定技能容量并引入未知新奖励和通道变化：旧基础是否还必要，旧 SF 是否过期，基淘汰是否损害后来恢复？原文的静态最优构造提供起点，有限资源的终生维护尚需另做验证。

<a id="research-options-directional-policy-contract"></a>

## 研究专题 B · 方向条件策略如何成为真正的时间抽象

HILP（ICML 2024）从离线时间距离表示学方向条件行为。它与 METRA 都使用潜在位移的方向奖励，但表示来源不同：HILP 从离线目标价值约束距离，METRA 在技能交互中联合约束表示与行为。条件策略 π(a|s,z) 本身只定义当前动作，必须再规定调用、停止和时长，才能成为 planner 可使用的 option。

$$
d^*(s,g)\approx\|\phi(s)-\phi(g)\|_2,\qquad r_z(s,a,s')=(\phi(s')-\phi(s))^\top z,\quad \|z\|_2=1
$$

HILP 的结构将时间距离与方向行为联系起来。精确欧氏嵌入要求距离结构相容；一般有向控制的 d*(s,g) 与 d*(g,s) 可不同，不能同时被同一对称欧氏距离精确表示。近似训练结果与理论条件须分别报告。

一扇只允许从左到右通过的门给出了反例：左右两边在像素上近，单向到达很容易，反向到达却不可能。仅以对称潜在距离宣布“两个方向同样可执行”会掩盖控制限制。正确的接口应实际测各方向策略成功率，并让启动集合排除无法可靠执行的起点。

$$
o_z=(I_z,\pi_z,\beta_z),\quad \widehat R_z=\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1},\quad y_{\rm high}=\widehat R_z+\gamma^\tau V(S_{t+\tau})
$$

高层使用真实外部奖励与真实段长。方向奖励用于训练低层，除非改变主任务，否则不能代替 R。固定时长 K 是一种明确的 β/时钟约定；目标到达终止又是另一种约定，不能只标为同一个“skill”。

| 缺少的接口 | 最小可实现选择 | 必须测什么 |
| --- | --- | --- |
| 启动条件 | 仅从数据覆盖且方向成功的状态调用 | 数据外起点的失败与拒绝调用率 |
| 停止 | 固定 K 步或检测子目标到达 | 长度分布、停止错误与切换开销 |
| 后果模型 | 拟合外部奖励、联合折扣终点与时长 | 新后续价值下的 backup 误差 |
| 版本 | 方向表示与策略变化时标记模型过期 | 同名 z 的后果是否已变 |
| 高层控制 | 按段收集真实 target 学选择 z | 计入原始步数的收益与计算延迟 |

实验可冻结同一个 HILP 低层，比较直接目标方向、无模型高层与 option-model 规划，匹配真实交互和调用预算。再仅改变停止条件，检验收益来自更好的行为还是更合适的时间尺度。研究空缺是新经验改变距离和方向语义时，怎样同步维护启动、停止和后果模型；预训练的通用方向接口并未自动完成这条闭环。

<a id="lesson-check"></a>

## 11. 诊断与自测

- 技能全部长度为 1：检查终止梯度符号、critic 初始化、β 是否被当作环境 done，以及高层选择是否有过强的即时切换优势。
- 所有技能完全一样：检查各技能是否获得不同学习信号、初始化/探索能否破坏对称；增加技能数量并不自动增加有效能力。
- 长期 option 价值偏高：检查 $γ^τ$ 是否误写为 γ，是否漏掉沿途负奖励，以及技能模型是否过期。
- 只在训练目标有效：分别评价固定技能后重新学习高层的速度、技能覆盖、维护开销与长期遗忘。

自测 1：$\beta=1$ 是否使 $V(s')=0$？答：只是重新选择 option，真实环境终止才清零后续价值。自测 2：为何 intra-option 能学习没被执行的 option？答：共享已观察的一步后果，用动作兼容性或概率比修正当前动作分布，以 bootstrap 表示后续行为。自测 3：SF 是否就是 option？答：SF 预测固定策略的未来特征，option 定义行为与停止；一个 option 可以有 SF 模型，两者的职责不同。

## 本章的实验设计

计入技能发现和执行的原始环境步。将技能覆盖、使用频率和主任务收益分别记录。

设定：一个 option 连续得到奖励 1、2，γ=0.9，终点价值 3。随后加入随机持续时间、真实终端与外部截断。

- 两步目标为 5.23；真实任务终端时为 2.8。
- τ=1 退化到对应单步 backup。
- option 终止只让高层重选，不自动清整个 learner。

对照：primitive-only 与等动作保持时长；固定技能、随机技能和学习技能；冻结旧库、在线维护与重新学习

记录：primitive steps、option 次数和时长分布；终止地点、覆盖、失败退出和使用集中度；原任务收益与技能发现成本

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-modules)

## 学习与研究衔接

单步动作推广为可变持续时间的策略。发现、学习、选择和终止 option 是不同子问题。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-options) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=options) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=options)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

Laplacian 描述行为图结构，奖励感知表示加入路径价值，METRA 学习有区别的行为，HIQL 利用离线目标轨迹，MaestroMotif 引入语言先验。它们承担不同的设计工作；生成技能、选择技能与组合技能需要分别评价。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [Proper Laplacian Representation Learning](https://yingwen.io/zh/continual-rl/research/#recent-proper-laplacian-representations)
- [Reward-Aware Proto-Representations in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-reward-aware-proto-representations)
- [Laplacian Keyboard: Beyond the Linear Span](https://yingwen.io/zh/continual-rl/research/#recent-laplacian-keyboard)
- [METRA: Scalable Unsupervised RL with Metric-Aware Abstraction](https://yingwen.io/zh/continual-rl/research/#recent-metra-skills)
- [HIQL: Offline Goal-Conditioned RL with Latent States as Actions](https://yingwen.io/zh/continual-rl/research/#recent-hiql-hierarchical-goals)
- [MaestroMotif: Skill Design from Artificial Intelligence Feedback](https://yingwen.io/zh/continual-rl/research/#recent-maestromotif-semantic-skills)
- [Foundation Policies with Hilbert Representations](https://yingwen.io/zh/continual-rl/research/#recent-hilbert-foundation-policies)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

模型可提取性的理论说明某类能力需要什么知识，不指定唯一网络。Dreamer 研究潜在想象控制，STOMP 研究随机时长行为模型，DRAGO 研究旧模型知识保留。模型误差、查询策略和规划收益之间仍需实验连接。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [Laplacian Keyboard: Beyond the Linear Span](https://yingwen.io/zh/continual-rl/research/#recent-laplacian-keyboard)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文提供定义和条件，架构讲座提出模块组织，算法论文检验特定机制。完整系统还要明确智能体、外部设计者和世界各自承担的工作；组件成立不自动意味着组合后的长期收益成立。

- [The OaK Architecture: A Vision of SuperIntelligence from Experience](https://yingwen.io/zh/continual-rl/research/#recent-oak-architecture)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)

### Reward-Respecting Subtasks for Model-Based Reinforcement Learning

Richard S. Sutton, Marlos C. Machado, G. Zacharias Holland, David Szepesvari, Finbarr Timbers, Brian Tanner, Adam White

Artificial Intelligence · 2023 · 支持方法与理论

#### 研究问题

学到一个能到达子目标的技能之后，为什么它仍可能不适合主任务规划？

#### 关键机制

STOMP 把子任务、option、模型和规划连起来。子任务保留原任务的路径奖励，并用带有特征偏好的终止价值表达目标；学习得到策略和终止规则后，再预测该行为的累计奖励与折扣终点。这样，技能不会因为只追求到达子目标而忽略途中代价。

#### 证据

论文用明确的小问题展示奖励感知子任务如何产生更有用的行为和规划模型。它提供的是可分析的构造链，而非只比较一个技能执行成功率。

#### 条件与限制

终止收益的约定是子任务定义的一部分，不能随意换成固定终点奖励。特征和子任务候选的选择尚不等于完整自主发现机制；实验也不构成整个 OaK 架构的验证。

#### 阅读与实验

在同一个绕路环境中比较“最短到达目标”和“保留路径奖励”的子任务。分别计算 option 的奖励模型、折扣终点模型与一次规划备份。

#### 原文与相关入口

- [期刊论文](https://doi.org/10.1016/j.artint.2023.104001)：STOMP 与奖励感知子任务的正式论文。
- [作者预印本](https://arxiv.org/abs/2202.03466)：最初预印本早于期刊年份；阅读停止收益的精确定义。

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

### Laplacian Keyboard: Beyond the Linear Span

Siddarth Chandrasekar, Marlos C. Machado

arXiv 预印本 · 2026 · 支持方法与理论

#### 研究问题

从一组谱技能出发，能否解决超出原特征线性奖励空间的新任务？

#### 关键机制

Laplacian 特征先定义行为基，并借助后继特征预测各行为的后果。固定任务权重的价值组合受特征张成空间限制；论文进一步使用随状态变化的元策略，在不同位置组合已有行为。关键变化是组合规则从一组全局固定权重变为状态相关的行为选择。

#### 证据

论文对行为基与任务组合给出理论分析，并报告有限环境中的组合实验。它延续 eigenoptions 与 successor features 的路线，同时解释了为什么单纯线性读出会遇到表达边界。

#### 条件与限制

理论结论依赖具体的行为基、近似误差和任务条件。技能集合的长期生成、淘汰与非平稳模型维护仍是另外的问题；此处按预印本收录，不指定未经确认的会议。

#### 阅读与实验

构造一个必须在中途切换方向的奖励任务。分别比较固定权重的技能选择与状态相关切换，并解释性能差异来自哪里。

#### 原文与相关入口

- [作者预印本](https://arxiv.org/abs/2602.07730)：阅读线性张成空间的限制及状态相关组合机制。

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

### HIQL: Offline Goal-Conditioned RL with Latent States as Actions

Seohong Park, Dibya Ghosh, Benjamin Eysenbach, Sergey Levine

NeurIPS 2023 · 2023 · 支持方法与理论

#### 研究问题

只拿到已有轨迹时，长距离目标为什么适合拆成高层子目标和低层动作？

#### 关键机制

HIQL 学习目标条件价值，并以潜在状态作为高层动作。高层提出中间目标，低层输出环境动作；两层利用优势加权回归学习。时间分解让低层面对较短的控制距离，而不是要求一个策略直接消化所有远距离价值误差。

#### 证据

论文在离线长时域目标任务中检验层次结构，并提供原始实现。作者后来在 OGBench 中提供更统一的实现，二者适合不同用途：原实验复现和统一基线比较。

#### 条件与限制

数据覆盖和行为分布约束仍然存在。目标采样、层级时间间隔与离线轨迹由外部流程提供，不能把效果解释为在线自主目标生成已经解决。

#### 阅读与实验

对一段轨迹明确标记最终目标、中间目标和当前动作。逐一检查价值目标、优势权重和高层标签的停止梯度边界。

#### 原文与相关入口

- [NeurIPS 2023 原文](https://papers.nips.cc/paper_files/paper/2023/file/6d7c4a0727e089ed6cdd3151cbe8d8ba-Paper-Conference.pdf)：离线目标学习和两层回归目标。
- [HIQL 原始实现](https://github.com/seohongpark/HIQL)：README 区分原始实验与 OGBench 中的新实现。

#### 作者代码

[作者仓库；更新的统一基线另见 OGBench。](https://github.com/seohongpark/HIQL)

HIQL 原论文的离线训练与评价。

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

### The OaK Architecture: A Vision of SuperIntelligence from Experience

Richard S. Sutton

RLC 2025 讲座 / Oak Lab · 2025 · 定义与架构观点

#### 研究问题

持续学习是否只是在一个现成 actor–critic 上加入抗遗忘机制，还是需要重新安排知识构造与使用？

#### 关键机制

OaK 提出从经验持续形成状态、预测知识、子任务、时间抽象与模型，并让这些知识服务规划的架构方向。这里的重点是模块之间怎样产生可复用知识，而不只是保留某个固定策略网络的参数。

#### 证据

官方页面提供 Richard Sutton 的架构讲座与相关研究入口。STOMP、预测学习和在线特征学习等论文可以检验其中具体组件，但不能自动验证整体架构。

#### 条件与限制

这是研究愿景与架构讲解，不是一套已公布完整训练配方、统一基准结果和可复现端到端代码的系统。资源分配、问题生成、知识替换与模块相互干扰仍需明确算法。

#### 阅读与实验

为每个模块写出输入、输出、更新频率和资源上限。再选择一个双模块接口做可证伪实验，例如技能模型改善是否真的减少规划误差。

#### 原文与相关入口

- [Oak Lab 官方讲座页面](https://oaklab.ai/posts/the-oak-architecture)：讲座入口与架构研究方向。
- [Oak Lab 研究主页](https://oaklab.ai/)：区分已发表研究、技术文章和仍在预告中的项目。

### Foundation Policies with Hilbert Representations

Seohong Park, Tobias Kreiman, Sergey Levine

ICML 2024 · 2024 · 支持方法与理论

#### 研究问题

如何从无任务标签的离线轨迹形成既能按方向调用、又能用于目标任务的策略接口？

#### 关键机制

HILP 先学习近似保存时间距离的 Hilbert 表示，再以潜在位移与方向的内积训练方向条件策略。新任务通过奖励回归、目标方向或分层调用选择策略条件，结构表示也支持测试时规划。

#### 证据

ICML 原文与作者项目包含零样本 RL、离线目标条件 RL 及规划实验；官方仓库将 zero-shot 与 goal-conditioned 两套实现分开。

#### 条件与限制

精确时间距离不总能无损嵌入有限维对称欧氏距离，尤其有向不可逆行为；理论充分条件与近似神经实验需区分。方向条件策略没有自动获得任意停止条件或完整技能后果模型。

#### 阅读与实验

固定离线数据分别测距离误差、方向执行误差、奖励可表达误差与高层收益。让同一视觉观测对应不同历史，检查仅观测编码是否足够，之后再讨论 CRL 状态维护。

#### 原文与相关入口

- [ICML 2024 原文](https://proceedings.mlr.press/v235/park24g.html)：Hilbert 距离、策略提示和定理前提；不是 ICLR 论文。
- [作者项目与公式](https://seohong.me/projects/hilp/)：时间距离与方向奖励接口。
- [官方实现](https://github.com/seohongpark/HILP)：hilp_zsrl 与 hilp_gcrl 对应不同实验。

#### 作者代码

[作者项目直接链接并标为 official implementation。](https://github.com/seohongpark/HILP)

离线预训练、零样本奖励适配及目标条件实验。

### Constructing an Optimal Behavior Basis for the Option Keyboard

Lucas N. Alegre, Ana L. C. Bazzan, André Barreto, Bruno C. da Silva

NeurIPS 2025 · 2025 · 支持方法与理论

#### 研究问题

状态相关的技能组合足够强时，是否仍须为每个新奖励保存一条完整策略？

#### 关键机制

OKB 联合扩充基础策略与 Option Keyboard 的元策略。线性支持方法选择需要补齐的任务权重，先训练现有基础上的组合，再检查无法表达的动作并新增基础，移除冗余项。优化的是可组合的行为基，而非仅增加技能数量。

#### 证据

正式原文分析基础数量与 convex coverage set 的关系，并在多任务领域检验规模与表现。附录提供元策略、新基础训练和角点枚举等实现细节；论文声明实验代码在 Supplemental Material。

#### 条件与限制

保证假定 NewPolicy 返回最优策略，且 TrainOK 能达到可表达的最优组合。近似 critic、有限训练、变化动力学不直接继承保证；非线性任务结论仅覆盖最优行为可由相应子策略组合的类别。

#### 阅读与实验

分别比较“增加基础”“只训练组合”和“去除冗余”。保留一组未用于基构造的奖励权重，记录基础规模、元策略成本、SF 误差与迁移回报。持续淘汰仍需未来任务效用检验。

#### 原文与相关入口

- [NeurIPS 2025 原文与补充材料入口](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0ab48777def88e73b50746a6011be0b0-Abstract-Conference.html)：算法 1–3、附录 A.3 的两个最优子程序假设及代码声明；未在本教材运行补充代码。
- [Option Keyboard 的经典桥梁](https://proceedings.neurips.cc/paper/2019/file/251c5ffd6b62cc21c446c963c76cf214-Paper.pdf)：cumulant 组合、GPE/GPI 与技能接口。


<a id="chapter-code"></a>

## 下载与运行

SMDP、intra-option、动作与终止梯度的表格实验；深度技能训练另附原论文和作者工程。

[下载 knowledge_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/knowledge_algorithms_lab.py)

```sh
python3 knowledge_algorithms_lab.py options
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton, Precup & Singh — Between MDPs and semi-MDPs](https://doi.org/10.1016/S0004-3702(99)00052-1)：原始 options 框架。重点是 SMDP 最优方程、intra-option learning 与 option models，而不只读三元组定义。

- [Bacon, Harb & Precup — The Option-Critic Architecture](https://arxiv.org/html/1609.05140)：原文式(1)–(3)、内部策略梯度及终止梯度。对照 arrival 状态与动作采样状态，区分理论占用权重和在线实现。

- [Jean Harb — Option-Critic 作者 Atari 实现](https://github.com/jeanharb/option_critic)：作者工程；沿 critic target、policy loss、termination loss 读，并独立检查旧框架环境依赖。

- [Machado et al. — A Laplacian Framework for Option Discovery](https://arxiv.org/abs/1703.00956)：原文。比较覆盖结构的发现目标与基于主任务回报的技能优化。

- [Marlos Machado — options](https://github.com/mcmachado/options)：作者源码；分别追踪表示、内在奖励和技能求解，不将此仓库当作所有 options 方法的统一实现。

- [Machado et al. — Eigenoption Discovery through the Deep Successor Representation · ICLR 2018](https://arxiv.org/abs/1710.11089)：从完整图上的谱发现转向从经验学习未来特征；重点比较 SR 的策略依赖和原始图结构的关系。

- [Machado et al. — Temporal Abstraction in RL with the Successor Representation · JMLR 2023](https://jmlr.org/papers/v24/21-1213.html)：系统解释表示—子任务—技能—经验的反馈，以及 SR 支持时间抽象的机制。

- [Klissarov & Machado — Deep Covering Options · ICML 2023](https://proceedings.mlr.press/v202/klissarov23a.html)：DCEO 原文：深度表示、覆盖性技能与在线探索如何连接。

- [DCEO 作者实现](https://github.com/mklissa/dceo)：深度探索技能实现；先定位表示学习、内在奖励和行为采样，再与普通深度 RL 基线对照。

- [Proper Laplacian Representation Learning · ICLR 2024](https://arxiv.org/abs/2310.10833)：ALLO 原文：从低频子空间到有序特征对，解释为何方向与特征值对后续技能和规划重要。

- [ALLO 作者实现](https://github.com/tarod13/laplacian_dual_dynamics)：train_laprepr.py 是训练入口；同时阅读原始变量、对偶乘子与正交约束，而非仅看平滑损失。

- [Value-Aware Eigenoptions · RLC 2025 workshop](https://arxiv.org/abs/2507.09127)：区分固定技能对信用分配的帮助与在线价值驱动发现可能损害探索的结果；该工作属于 workshop。

- [Park, Rybkin & Levine — METRA · ICLR 2024](https://arxiv.org/abs/2310.08887)：时间距离约束下的方向条件技能；与仅优化可区分性的目标比较。

- [METRA 作者实现](https://github.com/seohongpark/METRA)：包含潜在表示与技能策略的联合训练；阅读时分别追踪内在奖励、距离约束和 SAC 更新。

- [METRA 核心文件 — iod/metra.py](https://github.com/seohongpark/METRA/blob/master/iod/metra.py)：_update_rewards 构造方向内在奖励；_update_loss_te 与 _update_loss_dual_lam 处理表示和约束；_optimize_op 进入技能策略训练。

- [Laplacian Keyboard · 2026](https://arxiv.org/abs/2602.07730)：谱特征、连续奖励权重、SF 与高层分段控制；附录 H 提供作者算法伪代码。

- [Machado — Deep RL Course 讲座讲义](https://deeprlcourse.github.io/assets/guests/marlos_machado.pdf)：结合图示复习环境表示、技能发现和探索反馈；可在读完谱发现部分后使用。

- [Eysenbach et al. — Diversity Is All You Need](https://arxiv.org/abs/1802.06070)：原文。关注互信息代理、技能先验与动作熵；可区分性不是下游任务价值。

- [Ben Eysenbach — SAC / DIAYN 作者代码](https://github.com/ben-eysenbach/sac)：作者历史工程；DIAYN 在 SAC 上加入技能输入、判别器和内在奖励，依次阅读这三个改动。

- [Sharma et al. — Dynamics-Aware Unsupervised Discovery of Skills](https://arxiv.org/abs/1907.01657)：原文。理解技能条件后果模型如何既提供发现奖励又进入下游控制。

- [Google Research — DADS 作者代码](https://github.com/google-research/dads)：原工程含技能学习与技能空间 MPC；入口 unsupervised_skill_learning/dads_off.py，按其配置区分训练与评估。

- [ICML 2024 原文](https://proceedings.mlr.press/v235/park24g.html)：Hilbert 距离、策略提示和定理前提；不是 ICLR 论文。

- [作者项目与公式](https://seohong.me/projects/hilp/)：时间距离与方向奖励接口。

- [官方实现](https://github.com/seohongpark/HILP)：hilp_zsrl 与 hilp_gcrl 对应不同实验。

- [NeurIPS 2025 原文与补充材料入口](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0ab48777def88e73b50746a6011be0b0-Abstract-Conference.html)：算法 1–3、附录 A.3 的两个最优子程序假设及代码声明；未在本教材运行补充代码。

- [Option Keyboard 的经典桥梁](https://proceedings.neurips.cc/paper/2019/file/251c5ffd6b62cc21c446c963c76cf214-Paper.pdf)：cumulant 组合、GPE/GPI 与技能接口。


---

# Dyna：模型学习与规划

直接学习只使用真实发生的一步；模型学会了后果以后，可以从旧状态重新思考。Dyna 的关键是把真实学习、模型学习、规划分开再连接。

## 本章内容

- 能区别 transition model、Q 函数与规划更新。
- 实现表格 Dyna-Q，并解释规划步数为何不是免费样本。
- 理解模型陈旧、分布选择与优先规划的失效方式。

<a id="problem-definition"></a>

## 本章的问题定义

真实交互昂贵时，从真实经验学后果模型，再用预算内的模型备份传播价值。

### 给定条件与符号

- 固定折扣控制任务、真实经验、模型表示和可规划状态动作。
- 每个真实步的规划次数或总计算预算；模型更新与真实终止规则。

### 需要求解的对象

共享价值上的真实学习与模型规划组合；目标是控制收益，模型拟合与模型内收敛是中间子问题。

### 信息与数据权限

真实经验更新模型 $\hat M=(\hat r,\hat P)$；模拟后果仅用于价值/策略计算。规划不得将自己的预测重新记为独立真实事实。

$$
(\hat TQ)(s,a)=\hat r(s,a)+\gamma\sum_{s'}\hat P(s'\mid s,a)\max_bQ(s',b)
$$

$\hat r$ 是奖励模型，$\hat P$ 是转移模型，$\gamma<1$ 是折扣，$Q$ 为动作价值。此式是当前模型的控制备份，不是新获得的真实经验；真实任务的最优值对应真实算子 $T$。规划预算有限且 $\hat M$ 可能有偏。

### 成立条件与解的含义

- 静态界要求真实与模型的有界折扣算子均满足相应收缩条件。
- 随机世界不能把最后一个后果当精确分布；非平稳世界必须检查模型年龄和真实验证。

判断准则：两步链在模型保存后的一次起点备份传播0.9；固定模型下检查备份残差及解析值；匹配真实步和总备份两种预算分别比较，并报告模型偏差。

### 适用边界

- 增加模拟备份不增加真实证据。
- 模型内收敛不证明真实最优或变化后已经适应。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)：模型学习提供后果接口，Dyna将它接到真实学习与模拟备份。

- 特例：增加条件 · [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)：Dyna是规划的一种学习期价值更新方式；行动时搜索和想象训练采用不同接口。

- 组合不同学习问题 · [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)：旧模型无法自行发现外界变化，Dyna-Q+类真实再验证机制需要探索。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

稀少真实奖励传播慢，但模型偏差可被反复规划放大。

### 本章的核心思路

真实数据决定模型事实，模型备份重用其后果计算；对真实性与传播速度分开检验。

1. [复用同一个价值备份](#lesson-derive)：因为真实与模型后果都能构造Bellman标签，直接学习和规划共享价值，但区别后果来源。

2. [落实真实—模型—规划顺序](#dyna-steps)：因为只有真实转移带来新证据，先真实备份和建模，再预算内选择已观察起点规划。

3. [把计算投向传播前驱](#dyna-priorities)：因为价值变化只影响相关前驱，用残差队列与前驱索引调度，而模型真实性仍单独验证。

结论与条件：固定精确有限折扣模型可按Bellman收缩求解；学得模型留下误差，更多备份只能减少当前模型内的求解误差。

### 相关方法改变了什么

- 无规划Q-learning：只用真实后果更新，可作为零规划退化对照。

- 均匀Dyna：在已建模状态动作上分配备份。

- Prioritized sweeping/Dyna-Q+：前者调度计算传播，后者鼓励真实再尝试陈旧行为，解决不同不足。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 一条经验

在 $S_{t}$ 做 $A_{t}$ 后得到 $R_{t+1}$、$S_{t+1}$。奖励属于这次转移，不是到达状态之前另一轮的奖励。

### 折扣与终止

$\gamma$ 控制未来相对权重；真实终止后的价值设为 0。本章记 $\gamma_{t+1}$=$\gamma$(1−terminated)。训练脚本的时间截断不一定是任务终止。

### 表格与函数逼近

表格每个状态或状态动作一组独立参数；函数逼近让不同输入共享参数。更新一个输入可能改变其他输入的预测。

<a id="lesson-setting"></a>

## 1 · 三个部件不是三个互斥算法

设离散状态动作、先从确定性固定环境开始，优化折扣回报。直接学习更新 Q；模型记下每个已访问 (s,a) 的奖励和后继；规划从模型选一个状态动作再进行备份。真实经验仍是模型知识的来源，规划不会凭空获得环境已经变化的证据。

| 部件 | 输入 | 输出 / 持久状态 |
| --- | --- | --- |
| 直接学习 | 真实 (s,a,r,s′) | 改善当前 Q |
| 模型学习 | 相同真实转移 | M(s,a)=(r,$\gamma_{t+1}$,s′) |
| 规划 | 模型转移及当前 Q | 再次改善 Q |
| 行为 | 当前状态与 Q | 实际动作，决定下一条真实经验 |

这个简单模型只适用于确定性环境。随机环境若只记最后一个后继，会把噪声当作确定规律；应保存条件分布、经验样本集合或其他概率模型。用神经网络并不会取消建模这个分布的责任。

<a id="lesson-derive"></a>

## 2 · 从真实 Q-learning 备份到模型备份

$$
Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma_{t+1}\max_b Q(s\prime,b)-Q(s,a)]
$$

真实更新与 Q-learning 相同。模型预测一个后果后，可把预测的 r、s′代入相同备份。改变的是备份的数据来源。价值函数与转移模型仍是不同的学习对象。

$$
\hat TQ(s,a)=\hat r(s,a)+\gamma\sum_{s\prime}\hat P(s\prime\mid s,a)\max_b Q(s\prime,b)
$$

这是概率模型的期望备份；也可从模型采样一个后果进行 sample backup。期望备份计算更贵、单次方差更小；采样备份可用较低成本集中在更有用的状态动作。

$$
\|Q^*_{\hat M}-Q^*_M\|_\infty\le\frac{\|\hat TQ^*_M-TQ^*_M\|_\infty}{1-\gamma}
$$

两个模型的 Bellman 最优算子均是 $\gamma$ 压缩且有界时，由加减 T̂Q*M 和三角不等式得到。它说明即使规划完全收敛，错误模型仍会留下由 1/(1−$\gamma$) 放大的误差；更多规划不等于更多真实知识。

这条不等式是固定模型、固定 MDP 下的分析。CRL 中模型、奖励和表示同时改变，误差项随时间变，需要重新衡量旧模型与当前世界的偏差，不能拿静态收敛替代适应分析。

<a id="dyna-steps"></a>

## 3 · Dyna-Q 的完整更新过程

**算法：Dyna-Q 的三个学习步骤共享 Q，但模型只由真实数据更新**

1. 初始化动作价值 $Q$ 与空模型 $\mathcal M$；给定每步规划次数 $n$。
1. 每个真实环境步：
  1. 按 $\epsilon$-greedy 策略选择 $a$，观察 $r,s'$ 与终止标记。
  1. 使用真实转移进行一次 Q-learning 更新。
  1. 更新模型 $\mathcal M(s,a)\leftarrow(r,\gamma',s')$。
  1. 重复 $n$ 次：
    1. 从已观察的状态—动作对中选择 $(\tilde s,\tilde a)$。
    1. 由模型得到 $(\hat r,\hat\gamma,\hat s')$。
    1. $Q(\tilde s,\tilde a)\leftarrow Q(\tilde s,\tilde a)+\alpha[\hat r+\hat\gamma\max_bQ(\hat s',b)-Q(\tilde s,\tilde a)]$。
  1. 继续真实交互；回合边界由任务协议决定。

不能把规划生成的转移重新当成真实样本去更新同一个模型，否则会用自己的想象强化自己的错误；也不能把 n 次规划算作 n 个真实环境步。在报告中分别记录 real steps、model updates、planning backups 和实际运行时间。

<a id="lesson-example"></a>

## 4 · 两步链说明规划为什么能传播新奖励

A→B 的奖励为 0，B→终点奖励为 1，$\gamma$=.9，Q 初始 0，$\alpha$=1。第一次真实经过 A 时，Q(A)=0；经过 B 后，Q(B)=1。若模型已保存 A→B，在下一次真实从 A 经过之前，规划一次 A 就能得 Q(A)=.9。

现在世界把 B 的奖励改成 −1，但智能体尚未访问 B。模型仍保存 +1，规划一千次也只会把旧答案强化得更一致。只有获得变化证据、采用先验不确定性或主动再探索，才可能发现新事实。

若 n=0，本算法退回相同真实更新的 Q-learning。若模型是 oracle，比较的是规划调度；若模型也学习，结果同时混合了建模与规划质量。最好分别做这两个对照。

<a id="lesson-code"></a>

## 5 · 核心实现：五状态链的真实学习与规划

可独立运行的 Dyna-Q，记录真实与规划预算

```python
def q_backup(q, s, a, reward, discount, nxt, alpha):
    bootstrap = max(q[nxt]) if discount else 0.0
    q[s][a] += alpha * (reward + discount * bootstrap - q[s][a])


def dyna_chain(planning=5, steps=1000, seed=7):
    """Deterministic 5-state chain. Right at 4 gives 1 and truly terminates."""
    rng, q, model, state = random.Random(seed), [[0.0, 0.0] for _ in range(5)], {}, 0
    for _ in range(steps):
        greedy = [a for a in range(2) if q[state][a] == max(q[state])]
        action = rng.randrange(2) if rng.random() < 0.2 else rng.choice(greedy)
        nxt = max(0, state - 1) if action == 0 else state + 1
        reward, discount = (1.0, 0.0) if nxt == 5 else (0.0, 0.9)
        q_backup(q, state, action, reward, discount, nxt, 0.1)
        model[(state, action)] = (reward, discount, nxt)
        for _ in range(planning):
            simulated_s, simulated_a = rng.choice(list(model))
            r, g, sp = model[(simulated_s, simulated_a)]
            q_backup(q, simulated_s, simulated_a, r, g, sp, 0.1)
        state = 0 if not discount else nxt
    return {"Q": q, "right_action_reference": [0.9 ** (4 - s) for s in range(5)],
            "real_steps": steps, "planning_updates": planning * steps}
```

五个状态，右移到第 5 个边界得奖励 1 并终止，左移不越过 0。$\gamma$=.9，最优右移动作值为 $[.9^4,.9^3,.9^2,.9,1]$。测试对这个解析向量检查，以此区分数值误差与有限样本误差。

运行 dyna 后检查 `right_action_reference` 与各行的右移 Q。把 planning 从 5 改为 0，比同样真实步数下的传播速度；再比同样总 backup 次数，结论可能不同。若引入随机转移，先换成正确的概率模型，再谈 Dyna 是否有效。

<a id="dyna-priorities"></a>

## 6 · 有限规划预算应该花在哪里？

$$
p(s,a)=|\hat r+\hat\gamma\max_bQ(\hat s\prime,b)-Q(s,a)|
$$

prioritized sweeping 用模型预测的备份残差安排更新；后继值改变后，沿 predecessor 关系把影响传播到前驱。这里的 priority 是规划更新优先级，不等于 replay 的采样权重。

- 维护每个状态有哪些前驱 (s,a)，并在模型结构变化时更新这个反向索引。
- 真实观察后计算残差，大于阈值则入优先队列；每次弹出最大项做备份。
- 该状态 value 变化后，重新计算其前驱残差并入队；避免无限重复、陈旧队列值和重复预算统计。
- 如果规划误差很大只是因为模型本身错误，优先更新可能把错误放大；应同时监测模型真实性和价值残差。

Dyna-Q+ 给长期未尝试的行为加随时间增长的探索奖励，鼓励重新检查世界；它不是通过旧模型自行检测变化。奖励的时间单位、未尝试动作的初始化、访问时间更新必须一致，才能解释实验。

<a id="lesson-branches"></a>

## 7 · 从 Dyna 到世界模型和时间抽象规划

| 路线 | 复用的共同思想 | 新增难点 |
| --- | --- | --- |
| 表格 Dyna / prioritized sweeping | 真实经验学模型，模型改善价值 | 规划起点与预算 |
| MPC | 模型预测短期候选动作序列，每步重规划 | 模型误差、约束、在线搜索代价 |
| Dreamer / latent imagination | 在学得潜在模型中训练行为 | 表示与动力学的联合误差、想象分布 |
| MuZero | 为搜索学习决策相关模型 | 不要求重建完整观测；训练与搜索协议不同 |
| Option model + planning | 一个模型后果覆盖随机多步行为 | 奖励累计、时长折扣、终止分布 |
| 持续世界模型 | 在变化中更新模型并保存有用规律 | 旧数据、潜在状态漂移、模型与策略分别遗忘 |

这些方法不是只有网络大小不同。预测完整观测、预测奖励与转移、预测 option 后果、为搜索构造潜在状态，是不同模型接口；判断方法前应把输入、输出、使用者和训练信号写出来。

<a id="lesson-check"></a>

## 8 · 习题与讨论

更多规划后真实回报下降，是不是说明模型学习没用？不能。可能是模型不准、规划分布选错、价值逼近不稳，或计算延迟占用了真实行动机会。分别用 oracle 模型、固定 Q 目标、匹配预算隔离原因。

为什么不能把 model loss 下降直接等同于控制收益？模型可能在大量容易预测但与动作选择无关的观测上变准，却仍把关键奖励或终止事件预测错。应记录决策相关误差，并做固定策略或固定模型的消融。

## 本章的实验设计

同时匹配真实交互预算和规划计算预算。增加备份次数本身也可能提高回报。

设定：在小迷宫提供可准确维护的表格模型，比较每个真实步后执行 0、少量或较多模拟 backup；再引入模型错误。

- 真实转移与模型生成转移进入各自计数。
- 规划 backup 使用指定模型和价值版本。
- 关闭模型更新与关闭规划是不同开关。

对照：无规划 Q-learning；准确模型与学习模型；同计算的额外真实数据复用

记录：真实交互收益与模拟 backup 数；模型奖励/转移误差；墙钟、规划占比和决策延迟

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-planning)

## 学习与研究衔接

一次真实经验既更新模型，也支持额外规划。持续环境中应同时测模型陈旧程度和规划收益。

[分册导读](https://yingwen.io/zh/continual-rl/start/classic-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-dyna) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=dyna) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=dyna)



<a id="chapter-code"></a>

## 下载与运行

Python 3.10+，仅标准库。完整表格 Dyna-Q 与解析参照；概率模型、优先队列与神经世界模型需按正文接口进一步实现。

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

```sh
python3 foundations_detail_lab.py dyna
python3 foundations_detail_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：第 3–8 章建立 MDP、动态规划、MC、TD 与规划；第 9–13 章把这些更新扩展到函数逼近、资格迹和策略梯度。按本章的问题找相应章节，不必从头重读。

- [Sutton · Dyna, an integrated architecture](https://doi.org/10.1145/122344.122377)：把行动、模型学习、直接学习与规划整合的原始架构思想。

- [Moore & Atkeson · Prioritized Sweeping](https://doi.org/10.1007/BF00993104)：理解前驱索引与异步更新调度，而不只记“按误差排序”。

- [Sutton & Barto 教材复现代码](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)：社区维护的教材实现，不是 Dyna 原论文历史代码；chapter08 可比较 Dyna-Q 与 Dyna-Q+。


---

# 模型与后果预测：学什么，才能用于下一次决策？

执行一个动作或技能以后，会积累多少奖励、何时到哪里；这些预测怎样支持规划与任务变化后的迁移？

## 本章内容

- 区分样本模型、分布模型、期望模型、option model 与 successor features。
- 从随机持续时间的回报推导 reward/end-state 模型与一步 TD 学习。
- 知道期望模型为何在线性价值下足够，以及对非线性价值为什么会失败。
- 独立实现 SF 的向量 TD 与 GPI，并理解它们和 option/世界模型的边界。

<a id="problem-definition"></a>

## 本章的问题定义

预测动作或技能后果，以支持可改变的下游价值和规划；输出应由用途确定，而非统一要求重建全部观测。

### 给定条件与符号

- 被建模的固定动作/技能、奖励与原始时间约定。
- 状态/特征、下游价值函数类、真实数据与模型容量预算。

### 需要求解的对象

足以计算指定后果备份的奖励及终点统计；完整分布、样本模型、期望特征模型和SF预测的是不同对象。

### 信息与数据权限

技能 $o$ 的策略与停止函数给定，$\tau$ 为原始时长。改变技能或表示即改变题目；离策略一步学习须记录行为概率并检查支持。

$$
\mathcal B_oV(s)=\mathbb E_o\!\left[\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}+\gamma^\tau V(S_{t+\tau})\mid s\right],\qquad \hat{\mathcal B}_oV\approx\mathcal B_oV\quad(V\in\mathcal V)
$$

$\mathcal V$ 为声明的下游价值类，$\gamma$ 为折扣。模型充分性以能否复算该类备份判断；奖励模型 $r_o$ 和折扣终点模型 $p_o^\gamma$ 是实现接口。若 $V_w(s)=w^\top\phi(s)$，折扣特征期望足够；一般非线性价值不满足此交换。

### 成立条件与解的含义

- 固定环境、表示与被建模行为；奖励有界，$0\le\gamma<1$。终点模型假设技能停止或真实终止几乎必然在有限时间发生；折扣时长与终点联合建模，真实终止的后续价值固定为零。
- SF重加权需相同动力学、策略、折扣和特征语义，奖励变化限于给定特征的线性张成。

判断准则：已知技能上检验奖励和折扣终点向量，并对多组未拟合下游价值比较备份误差；包含随机终点均值失败反例及时间—终点相关例。

### 适用边界

- 平均下一状态代入非线性价值通常不等于价值期望。
- 模型重建损失下降不直接证明动作选择改善。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [Options 与技能发现](https://yingwen.io/zh/continual-rl/construction/options/)：技能规定执行和停止，模型估计该行为的奖励、时长及终点后果。

- 组合不同学习问题 · [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)：规划使用模型统计计算候选价值，其需求决定模型必须保留什么。

- 组合不同学习问题 · [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)：模型统计可化为特定预测题目；SF也预测固定策略的累计特征而非任意新行为。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

完整后果分布昂贵；均值压缩可能丢失下游最大值、非线性或随机时长所需的信息。

### 本章的核心思路

相对下游函数类定义足够统计，将内部奖励与联合折扣终点分开建模。

1. [从技能回报拆出后果接口](#lesson-derive)：因为未来价值可变，奖励和折扣终点分别建模，使同一模型可重用于不同尾值。

2. [检验压缩是否保留所需期望](#lesson-expectation)：因为非线性不能随意与期望交换，先在线性价值下推特征期望，再用均值反例界定适用范围。

3. [为奖励迁移预测累计特征](#lesson-successors)：因为只换奖励权重时可复用固定策略后果，SF保留累计特征，GPI再比较候选策略的重加权价值。

结论与条件：线性价值与固定特征下的期望备份等价是恒等式；SF/GPI保证需相同动力学等条件及价值误差控制，不覆盖任意奖励或技能变化。

### 相关方法改变了什么

- 完整分布/样本模型：保留多后果或提供样本，可用于一般尾值但成本与采样方差不同。

- 期望特征模型：对指定线性尾值充分，不能只凭均值支持任意非线性控制。

- Successor features：预测固定策略的累计特征，用于奖励重加权，不等于技能终点模型。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 条件模型

模型回答“从这个状态，假如做这个动作/遵循这个技能，会发生什么”。它必须包含条件行为；把经验中的平均下一状态当作所有动作共同的预测，无法比较行动。

### Option

一个可执行技能包含内部策略 $π_o$ 和停止概率 $β_o$。τ 是至少为 1 的原始步数。模型预测这个已指定行为的后果；改变其策略或停止函数，就是改变被建模的对象。

### 期望与采样

分布模型描述全部可能后果，样本模型随机生成一个后果，期望模型只输出某些统计量。平均值足够与否由下游计算决定，不由模型名字决定。

### 线性价值

φ(s) 是固定特征，w 是权重。线性指价值对特征线性；φ 本身可以是非线性编码。若编码也在学习，关于固定模型/表示的推导要重新检查。

$$
V_w(s)=w^\top\phi(s)
$$

<a id="lesson-setting"></a>

## 1. 模型要预测什么，取决于怎样使用它

“模型好不好”必须相对用途回答。预测下一张画面很准，可能仍错过决定动作的稀有碰撞；平均位移很准，可能把绕障碍的左右两条安全路径平均成穿墙。反过来，一个无法还原像素的模型，只要准确预测所需奖励和后续价值，也可能支持有效控制。先明确下游要计算哪个量，才能选择训练标签。

| 对象 | 模型的输出 | 直接用途 |
| --- | --- | --- |
| 一步分布模型 | P(s′,r\|s,a) | 对任意后续价值求期望，或采样模拟 |
| 一步样本模型 | 给 (s,a) 产生一次随机 (r,s′) | Dyna / rollout；需多样本处理随机性 |
| Option model | 内部折扣奖励 + 折扣终点后果 | 以技能为单位做 Bellman backup |
| 期望模型 | 下游所需特征的条件期望 | 线性价值的精确期望 backup |
| Successor features | 固定策略下未来特征的累计期望 | 奖励权重变化时重估该策略，再做 GPI |
| Latent/world model | 潜在状态、奖励、继续概率等 | 潜在轨迹上的控制学习或搜索 |

先假设环境是固定 MDP，状态或固定特征可观察，奖励有界，技能 $\pi_o,\beta_o$ 固定，且 $0\le\gamma<1$。本页的终点模型假设技能停止或真实终止几乎必然在有限时间发生，所以终点随机变量有定义。训练数据可以来自实际执行技能的轨迹，也可以来自支持其动作的其他行为策略。若同时改变环境、技能和表示，模型的预测目标也会漂移，除了静态收敛，还需要分析跟踪误差。

以 option 为例，模型接收 $(s,o)$，输出内部奖励 $r_o(s)$ 与折扣终点分布 $p_o^\gamma(\cdot|s)$；规划器用当前价值 $V$ 计算 $r_o+p_o^\gamma V$。这个分工允许价值改变时复用同一个后果模型。直接预测某个固定策略的完整回报也有用，但那是价值预测，不能替代面向不同后续价值的后果接口。

<a id="lesson-derive"></a>

## 2. Reward model 与折扣终点模型从哪里来

$$
Q(s,o)=\mathbb E_o\!\left[\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}+\gamma^\tau V(S_{t+\tau})\mid S_t=s\right]
$$

先执行指定 option，再按下游策略继续。其内奖励的和与终点的后续价值可以分别建模，因为期望具有线性。

$$
\begin{aligned}r_o(s)&=\mathbb E_o\!\left[\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}\mid s\right]\\ p_o^\gamma(j|s)&=\mathbb E_o\!\left[\gamma^\tau\mathbf1\{S_{t+\tau}=j\}\mid s\right]\\ Q(s,o)&=r_o(s)+\sum_jp_o^\gamma(j|s)V(j)\end{aligned}
$$

$p_o^γ$ 已把时间与终点联合加权。它不是普通归一化转移概率，行和是 $E[γ^τ]$。规划时不能再乘一次 γ，也不能把行重新归一化到 1。

完整执行可直接提供监督：观察奖励 $r_1,\ldots,r_\tau$ 后计算折扣和，为终点构造 one-hot 向量并乘 $\gamma^\tau$，再分别拟合条件均值。这不需要 bootstrap，但须等技能停止；长技能的估计方差和更新延迟都可能增大。下面用一步递推来学习相同对象。

考虑刚到 $s'$：若 option 停止，后续内部奖励为零，终点就是 $s'$；若继续，剩余奖励由 $r_o(s')$ 描述，剩余折扣终点由 $p_o^\gamma(\cdot|s')$ 描述。真实环境终止也必须停止技能。令 $D\in\{0,1\}$ 为真实终止标志，$\widetilde\beta=D+(1-D)\beta_o(S')$ 为合并后的停止概率。这里的 $\tau$ 取技能停止与真实终止中先发生的时间；采样窗口截断不算真实终止。

$$
\begin{aligned}r_o(s)&=\mathbb E_{\pi_o}[R+\gamma(1-\widetilde\beta)r_o(S')\mid s]\\ p_o^\gamma(j|s)&=\mathbb E_{\pi_o}\!\left[\gamma\{\widetilde\beta\mathbf1_{S'=j}+(1-\widetilde\beta)p_o^\gamma(j|S')\}\mid s\right]\end{aligned}
$$

两条分支都已经过一个原始步，所以终点项与继续项都乘折扣。奖励模型的停止分支没有额外奖励。这里保留显式终止状态，并固定其后续价值为零；终点模型仍记录到达它的折扣质量。

$$
\begin{aligned}\widetilde\beta'&=d+(1-d)\beta_o(s')\\ \delta_r&=R+\gamma(1-\widetilde\beta')\hat r_o(s')-\hat r_o(s)\\ \delta_{p,j}&=\gamma[\widetilde\beta'\mathbf1_{s'=j}+(1-\widetilde\beta')\hat p_o^\gamma(j|s')]-\hat p_o^\gamma(j|s)\\ \hat r_o(s)&\leftarrow\hat r_o(s)+\alpha_r\rho_o\delta_r\\ \hat p_o^\gamma(j|s)&\leftarrow\hat p_o^\gamma(j|s)+\alpha_p\rho_o\delta_{p,j}\end{aligned}
$$

d 是本次经验中的真实终止标志。状态值式模型对当前动作取 $π_o$ 平均，off-policy 时 $ρ_o=π_o(a|s)/b(a|s)$；真实执行该 option 时 ρ=1。所有右侧使用旧参数；神经网络版本对 target 停止梯度，并对当前输出的梯度做半梯度更新。

**算法：算法伪代码**

1. 固定 option $π_o$、$β_o$ 和表示版本；初始化 reward/end-state 模型
1. 每次看到原始 transition s,a,r,s′,d：
  1. 缓存旧 $r_o(s),r_o(s^{\prime}),p_o(s),p_o(s^{\prime})$
  1. 读取到达 s′ 后的 β；计算记录动作时的 $π_o(a|s)/b(a|s)$
  1. `stop = d + (1-d)*beta`
  1. `reward_target = r + gamma*(1-stop)*old_reward_next`
  1. `endpoint_target = gamma*(stop*one_hot(next_state) + (1-stop)*old_endpoint_next)`
  1. 分别更新 reward 和 endpoint 输出
1. 规划调用时：$r_o(s)$ + `dot(p_o(s), 当前 V)`；终止状态的 V 固定为零
1. 若 option/表示改变，标记模型过期并重新采样或持续更新

终止反例：本步奖励为 $1$、$\gamma=0.9$、技能本身的 $\beta_o(s')=0$，旧奖励模型在终止状态误估为 $5$。遗漏 $d$ 会得到目标 $1+0.9\times5=5.5$；正确目标为 $1$。即使控制价值在终止状态固定为零，也不能替代奖励模型自己的停止屏蔽。

<a id="lesson-expectation"></a>

## 3. 期望模型为什么有时够用，有时必错

状态很多时，显式存每个终点的概率昂贵。若下游价值对特征线性，就可以把求和移入特征期望，预测一个固定维度的向量 $m_o(s)$。这不是近似技巧，而是给定前提下的恒等式；但前提本身非常重要。

$$
\begin{aligned}m_o(s)&=\mathbb E_o[\gamma^\tau\phi(S_{t+\tau})\mid s]\\ \mathbb E_o[\gamma^\tau V_w(S_{t+\tau})\mid s]&=w^\top m_o(s)\\ y_m&=\gamma[\widetilde\beta'\phi(s')+(1-\widetilde\beta')\hat m_o(s')]\end{aligned}
$$

m 的 TD 递推只需把 one-hot 终点替换为特征，并沿用真实终止与技能停止的合并概率。状态可以高维，但下游 V 必须对这套特征线性，才能把期望与价值计算交换。

若采用终止状态零特征的约定，令 $\phi(s')=0$ 当 $d=1$。目标可等价写为 $y_m=\gamma(1-d)[\beta_o(s')\phi(s')+(1-\beta_o(s'))\hat m_o(s')]$。这时真实终止的目标向量为零；它与显式终止状态的 one-hot 约定输出不同，但在终止后价值为零的规划中一致。

反例：$X$ 以相同概率为 $-1$ 或 $+1$，$V(x)=x^2$。均值模型给 $\mathbb E[X]=0$，代入价值得到 $0$，而真正的 $\mathbb E[V(X)]=1$。只保存均值会丢掉方差信息。扩展特征为 $(x,x^2)$、学习完整分布或采样多个后果，分别提供了恢复所需信息的途径。

控制还包含最大值运算。即使每个动作的 $Q$ 对特征线性，$V(s)=\max_aQ(s,a)$ 通常仍非线性，所以 $\mathbb E[\max_aQ(S',a)]$ 不等于 $\max_aQ(\mathbb E[S'],a)$。采用线性状态价值 $V_w$、再在当前状态比较不同动作模型，是另一种规划接口；预测中的线性等价不能直接推广到任意 Q 控制。

随机时间与终点也可能相关。假设一半概率走一步到价值 $10$ 的 A，另一半概率走三步到价值 $0$ 的 B，$\gamma=0.9$。正确贡献是 $0.5\times0.9\times10=4.5$；若把 $\mathbb E[\gamma^\tau]=0.8145$ 与平均终点价值 $5$ 相乘，得到 $4.0725$。用平均时长 $2$ 得到的 $\gamma^2\times5=4.05$ 也不同。需要保存的是时间与终点的联合折扣后果。

<a id="lesson-successors"></a>

## 4. Successor features 与 GPI：不重学动力学地换奖励

另一个复用问题是：环境和行为策略不变，奖励的偏好改变。比如同一路线会产生“耗时、电量、采到的资源”三个信号；早上偏好快，晚上偏好省电。如果奖励是这些信号的线性组合，就能先预测整段未来信号，再乘新的偏好权重。这是 SF 的基本分解，不是预测“下一状态均值”。

$$
\begin{aligned}r_w(s,a,s')&=\phi(s,a,s')^\top w\\ \psi^\pi(s,a)&=\mathbb E_\pi\!\left[\sum_{k=0}^\infty\gamma^k\phi(S_{t+k},A_{t+k},S_{t+k+1})\mid S_t=s,A_t=a\right]\\ Q_w^\pi(s,a)&=\psi^\pi(s,a)^\top w\end{aligned}
$$

φ 是每步可观察的特征信号，ψ 是同一目标策略下的向量预测；每个分量都是一个价值预测问题。等式需要同样的动力学、策略、折扣与特征语义，奖励变化仅发生在 w。

将累计和拆成第一步与后续，就得到向量 Bellman 方程：标量奖励换为向量 $\phi$，标量价值换为向量 $\psi$。下一动作按同一个目标策略平均或采样。若对每个特征分量分别取最大，不同分量可能对应彼此冲突的行为，就不再是这个策略的未来特征预测。

$$
\begin{aligned}y_\psi&=\phi(s,a,s')+\gamma(1-d)\sum_{a'}\pi(a'|s')\hat\psi^\pi(s',a')\\ \hat\psi^\pi(s,a)&\leftarrow\hat\psi^\pi(s,a)+\alpha[y_\psi-\hat\psi^\pi(s,a)]\end{aligned}
$$

这是动作条件的表格 expected-TD：当前动作已作为条件给定，环境转移的样本不需要再乘当前动作概率比；后续动作明确按 π 求平均。如果改成状态式 ψ(s) 预测，当前动作平均和 off-policy 修正就必须相应改变。

只有一个旧策略时，重新加权得到的是该策略的新价值。为了改善控制，可保存多个 $\pi_i$ 的 SF，先计算每个候选在新奖励下的价值，再在每个状态选择候选中评价最高的动作，这就是 generalized policy improvement。它可以逐状态组合动作，而非只在 episode 开头选一个旧策略并始终照做。

$$
\pi_{\rm GPI}(s)\in\arg\max_a\max_i\left[\hat\psi^{\pi_i}(s,a)^\top w_{\rm new}\right]
$$

顺序是先沿特征维度与 w 做点积，再沿策略维度取 max，最后沿动作维度选 argmax。把两个 max 或特征求和维度弄错，会产生另一种算法。

**算法：算法伪代码**

1. 准备固定策略集合 $π_1,\ldots,π_n$ 与各自 SF 估计
1. 每个观察 transition：
  1. 对每个 i，用 $π_i$ 的下一动作分布构造向量 TD target，更新 $ψ_i$
1. 若奖励权重变为 $w_{new}$：
  1. 不改 SF，先计算 $Q_i(s,a)=ψ_i(s,a)^T w_{new}$
  1. $\mathrm{score}(a)=\max_i Q_i(s,a)$
  1. 执行 $\arg\max_a\mathrm{score}(a)$，并继续收集真实数据
1. 如果学习了新的专门策略，把它连同其 SF 加入集合；预算有限时须选择保留项

在精确 $Q$、相同动力学和折扣下，GPI 不劣于被比较的各个策略；近似保证取决于统一价值误差界。奖励不在当前 $\phi$ 的线性张成空间，或动力学改变导致旧 $\psi$ 失效时，需要重新估计相应误差。SF 预测策略产生的累计特征，option 定义执行与停止，option model 预测停止时后果；三者可以组合，但承担不同职责。

<a id="lesson-example"></a>

## 5. 两次原始转移怎样构造完整技能模型

技能从状态 $0$ 经过 $1$ 到 $2$，在 $2$ 停止，奖励分别为 $1,2$，$\gamma=0.9$。模型初值为零，步长为一。按从后往前的顺序更新，可以在两次 backup 中算出结果；在线从前往后更新时，还需再次访问或重放，才能把后方的新信息传到起点。

| 更新 | reward target | discounted endpoint target |
| --- | --- | --- |
| 1→2，β(2)=1 | 2 + 0 = 2 | 0.9 × one-hot(2) |
| 0→1，β(1)=0 | 1 + 0.9×2 = 2.8 | 0.9 × [0,0,0.9] = [0,0,0.81] |
| 给 V(2)=10 做一次规划 | 2.8 | 0.81×10 = 8.1，合计 10.9 |

这个 backup 与直接计算两步回报相同，无需重新模拟技能内的动作。若把终点向量错误归一化为 $[0,0,1]$，会得到 $12.8$；若额外再乘 $\gamma$，会得到 $10.09$。两个错误分别丢掉或重复计算了时间折扣。

SF 例子：两个动作分别产生特征 $(1,0)$ 和 $(0,1)$，随后进入动作编号对应的状态，$\gamma=0.9$。策略 $\pi_0$ 始终选动作 $0$，$\pi_1$ 始终选动作 $1$。先做动作 $1$ 再跟 $\pi_0$，SF 为 $(9,1)$；先做动作 $0$ 再跟 $\pi_1$，SF 为 $(1,9)$。当 $w=(1,2)$ 时，GPI 对两个动作的分数为 $19,20$，故选动作 $1$；改为 $w=(2,1)$ 后，无需重学 SF 就会改选动作 $0$。这里复用的是不变动力学和不变目标策略下的预测。

<a id="lesson-code"></a>

## 6. 实现与实验：期望、时间和策略条件

Monte Carlo/TD option model、随机时长联合模型、SF 向量更新及 GPI；数组和更新时序都可直接检查。

```python
def option_episode_target(rewards, gamma, endpoint, n_states):
    reward_target = sum(gamma**k * r for k, r in enumerate(rewards))
    endpoint_target = [0.0] * n_states
    endpoint_target[endpoint] = gamma**len(rewards)
    return reward_target, endpoint_target


def model_td_update(reward_model, endpoint_model, state, next_state, reward,
                    beta_next, alpha=0.1, gamma=0.9, rho=1.0, terminal=False):
    """Fixed option; n is a discounted endpoint distribution, not normalized.

    True environment termination forces stopping, independently of beta.
    The explicit terminal endpoint has V=0 during planning; its discounted
    probability mass is retained. A rollout cutoff is not true termination.
    All right-hand sides use pre-update values (self-loops included).
    """
    old_reward, next_reward = reward_model[state], reward_model[next_state]
    old_row, next_row = endpoint_model[state][:], endpoint_model[next_state][:]
    stop = 1.0 if terminal else beta_next
    r_target = reward + gamma * (1.0 - stop) * next_reward
    p_target = [gamma * (stop * float(j == next_state)
                        + (1.0 - stop) * next_row[j])
                for j in range(len(old_row))]
    reward_model[state] = old_reward + alpha * rho * (r_target - old_reward)
    endpoint_model[state] = [v + alpha * rho * (target - v)
                             for v, target in zip(old_row, p_target)]
    return r_target, p_target


def model_backup(reward, discounted_endpoints, values):
    return reward + sum(p * v for p, v in zip(discounted_endpoints, values))


def mixed_duration_model(outcomes, gamma, n_states):
    """outcomes: (probability, rewards, endpoint). Keep time/end correlation."""
    if not math.isclose(sum(p for p, _, _ in outcomes), 1.0):
        raise ValueError("Outcome probabilities must sum to one")
    r_bar, p_bar = 0.0, [0.0] * n_states
    for probability, rewards, endpoint in outcomes:
        r, p = option_episode_target(rewards, gamma, endpoint, n_states)
        r_bar += probability * r
        p_bar = [old + probability * new for old, new in zip(p_bar, p)]
    return r_bar, p_bar


def successor_feature_step(psi, state, action, features, next_state, next_action_probs,
                           alpha=0.1, gamma=0.9, terminal=False):
    """Action-conditioned SF for a fixed target policy, with known feature signal.

    The observed action conditions the prediction, so the next action is averaged
    under the target policy; no current-action importance ratio is needed here.
    psi[s][a] is a feature vector. Snapshot before mutation handles self-loops.
    """
    old = psi[state][action][:]
    next_vectors = [row[:] for row in psi[next_state]]
    next_features = [sum(pr * vec[j] for pr, vec in zip(next_action_probs, next_vectors))
                     for j in range(len(features))]
    target = [x + (0.0 if terminal else gamma * xn)
              for x, xn in zip(features, next_features)]
    psi[state][action] = [x + alpha * (y - x) for x, y in zip(old, target)]
    return target


def gpi_action(successor_feature_bank, state, reward_weights):
    """Greedy policy improvement over a bank of evaluated fixed policies."""
    action_scores = [max(sum(x*w for x,w in zip(psi[state][a], reward_weights))
                         for psi in successor_feature_bank)
                     for a in range(len(successor_feature_bank[0][state]))]
    return max(range(len(action_scores)), key=action_scores.__getitem__), action_scores


def learn_tiny_sf():
    bank = []
    for fixed_action in (0, 1):
        psi = [[[0.0, 0.0] for _ in range(2)] for _ in range(2)]
        probabilities = [float(a == fixed_action) for a in range(2)]
        for _ in range(400):
            for s in range(2):
                for a in range(2):
                    successor_feature_step(psi, s, a, [float(a==0), float(a==1)],
                                           a, probabilities, alpha=1.0)
        bank.append(psi)
    return bank
```

无第三方依赖；所有目标都可用上面的数字核算

```sh
python3 knowledge_algorithms_lab.py models
python3 knowledge_algorithms_lab.py test
```

运行得到 reward model 为 $2.8$，终点向量为 $[0,0,0.81]$，backup 为 $10.9$；随机时间与终点联合贡献为 $4.5$。SF/GPI 在 $w=(1,2)$ 时选择动作 $1$，两个动作的分数约为 $[19,20]$。逐项改变价值、持续时间或奖励权重，可以观察模型究竟保留了哪一种可复用信息。

- 改动 A：只改变 endpoint value，冻结 option 与模型，验证模型可被不同价值函数重复调用。
- 改动 B：令短路径终点与长路径终点交换，验证即使平均时长和未加权终点分布没变，正确价值仍可改变。
- 改动 C：保持线性 reward 但改变动力学，冻结 SF；观察立即迁移不再正确，再比较持续更新 ψ 的恢复速度。
- 改动 D：给 critic 加非线性，分别比较均值输入、特征扩展和多样本估计；报告 target 偏差而非只看训练 loss。

<a id="lesson-reward-aware"></a>

## 7. 当环境结构也需要考虑代价：Default Representation

SR 和 Laplacian 表示主要刻画在某种默认行为下哪些状态容易互相到达，但相同的图结构可以对应完全不同的实际代价。两条通道几何长度相同，一条却持续消耗大量资源：仅根据连通性构造的技能仍可能偏爱这条通道。Reward-Aware Proto-Representations（NeurIPS 2025）研究如何让用于技能与奖励塑形的结构表示反映沿途奖励。

理解 Default Representation（DR）要先限定控制问题。设非终止状态集合为 $N$、终止集合为 $T$，默认策略诱导转移 $P^{\pi_d}$，非终止奖励 $r(s)<0$。在线性可解控制中，控制器改变下一状态分布，同时为偏离默认分布付出 KL 代价，温度 $\lambda>0$ 控制偏离的价格。它不是任意标准动作 MDP 都自动具备的结构。

$$
v^*(s)=r(s)+\lambda\log\sum_{s'}P^{\pi_d}(s'|s)\exp\!\left(v^*(s')/\lambda\right)
$$

这个软 Bellman 方程来自最大化“期望后续价值减去对默认转移的 KL 代价”。默认分布没有支持的后果不能凭空控制出来；边界状态的价值由终止收益给定。

令 desirability 为 $z(s)=\exp(v^*(s)/\lambda)$，把两侧指数化。非线性的对数求和变成 $z$ 的线性递推，再把非终止状态和终止状态分开，就得到 DR。

$$
\begin{aligned}D_r&=\operatorname{diag}\!\left(\exp(r_N/\lambda)\right)\\ z_N&=D_r\left(P_{NN}^{\pi_d}z_N+P_{NT}^{\pi_d}z_T\right)\\ Z_{NN}&=\left[D_r^{-1}-P_{NN}^{\pi_d}\right]^{-1}\\ z_N&=Z_{NN}P_{NT}^{\pi_d}z_T,\qquad z_T=\exp(r_T/\lambda)\end{aligned}
$$

Z 把内部动力学与沿途代价编码在一起；固定这两项而改变终点收益时，可以重用 Z。负的非终止奖励使 $D_r$ 的对角元素小于 1，有助于保证所需逆和递推收敛。终点收益并未混进内部模型。

DR 与 SR 的联系可直接算出。若所有非终止状态的奖励都等于 $\lambda\log\gamma$，则 $D_r=\gamma I$，因而 $Z_{NN}=\gamma(I-\gamma P_{NN}^{\pi_d})^{-1}$，即 SR 的常数倍。奖励不均匀时，每经过一个状态都受到不同的指数权重，表示便能区分低代价和高代价区域。

$$
Z=D_r+D_rPZ,\qquad Z(s,:)\leftarrow Z(s,:)+\alpha\left[e^{r(s)/\lambda}\left(\mathbf e_s+Z(s',:)\right)-Z(s,:)\right]
$$

这是在默认策略下采样、只考虑非终止行列时的递推；进入终止状态后，内部矩阵的后续行取零。若行为不同，需明确所估计的默认转移或使用相应分布修正。它近似 SR 的向量 TD，但当前状态奖励决定的缩放同时作用于 one-hot 与后续行，不能只替换 SR 的 $\gamma$。

手算一条通道：默认行为从状态 0 必然进入状态 1，再进入终点；令两个非终止状态的指数奖励权重均为 0.5。则内部矩阵第一行为 (0.5,0.25)，第二行为 (0,0.5)。若状态 0 的代价增大，使权重降为 0.25，第一行变成 (0.25,0.125)。到达状态 1 的默认路径没有变，但跨过代价区的权重下降了。这正是奖励感知结构与纯连通结构的差异。

论文将 DR 用于构造奖励感知的谱特征，再用于奖励塑形、技能发现与迁移实验。它提供的是另一种构造长期结构的准则，而非把所有模型都换成一个新矩阵。若环境奖励变化，DR 本身通常也要更新；相反，奖励线性权重变化而动力学不变时，固定策略 SF 可以保持不变。两者分别把奖励放在不同位置，因此具有不同的复用边界。

| 表示 | 固定哪些条件才能复用 | 直接改变什么 |
| --- | --- | --- |
| SR / SF | 动力学、目标策略、折扣、特征语义 | 用新奖励权重重新评价同一策略 |
| DR | 默认动力学、内部奖励、KL 控制约定 | 通过终点收益或奖励感知结构重新求解 |
| Option model | 技能策略与停止规则、动力学、奖励约定 | 用新的后续价值评价同一技能 |
| STOMP 子任务与模型 | 子任务可改变，后果模型须匹配当前技能 | 由奖励相关子任务产生行为，再预测真实后果 |

作者 Reward-Aware-Proto-Representations 仓库适合按“默认转移与奖励 → DR → 特征 → 技能/塑形 → 外部回报”阅读。实验中应分别改变内部代价和终点奖励：前者测试结构能否重新适应，后者测试已有结构能否复用。若两者同时变化，就难以解释改进来自哪一种能力。

<a id="lesson-branches"></a>

## 8. 从表格模型到潜在世界模型

潜在世界模型把观测历史压缩为 z，再学从 z 和动作预测下一潜在状态。一个常见训练结构包含：利用当前观测得到后验表示、仅凭过去表示和动作得到预测先验、重建或预测观测、预测 reward 与继续概率，并用一致性/KL 项约束先验后验。目标是让不接触未来真实观测的想象轨迹仍保持决策相关的信息，而不是在训练时用未来观测泄漏答案。

$$
\begin{aligned}z_t&\sim q_\theta(z_t\mid h_t,o_t),\qquad h_{t+1}=f_\theta(h_t,z_t,a_t)\\ \hat z_{t+1}&\sim p_\theta(z_{t+1}\mid h_{t+1})\\ \mathcal L_{\rm model}&=\mathcal L_{\rm obs}+\mathcal L_{\rm reward}+\mathcal L_{\rm continue}+\mathcal L_{\rm regularize}\end{aligned}
$$

这四项说明观测、奖励、继续概率和潜在先验—后验之间的分工；各项的实际权重、分布形式与梯度路径由具体算法规定。重建项使用真实观测，想象轨迹只能使用预测先验。

Dreamer 类方法在真实序列上学习模型，再从后验状态启动想象 rollout 来训练 actor/critic；部署可直接执行学到的 actor。MuZero 类方法学习对 reward、value、policy 有用的潜在递推，并在决策时做搜索，不要求重建全部像素。这两类模型“学来做什么”不同，不能以画面是否逼真作为统一分数。

持续学习增加三种独立失效：环境漂移使过去 P 不再正确；技能漂移使同名 o 的后果改变；表示漂移使旧 replay 中的潜在坐标与当前模型不兼容。最小研究协议应分别开关三者。模型带上技能/编码器版本、从原始经验重新编码、近期数据加权、保留校准集、缩短想象跨度，都是可比较的设计变量，但没有一种可在所有变化下保证模型可靠。

| 模型路线 | 特别擅长的复用 | 需要额外检查 |
| --- | --- | --- |
| Tabular option model | 技能后果被任意新 V 调用 | 持续时间、终止、技能版本 |
| Expectation model | 线性值函数改变时快速重估 | 特征充分性、非线性和 max 的交换 |
| SF / GPI | 同动力学下换奖励权重 | 目标策略固定、reward 特征可表达性 |
| Generative latent model | 多步想象与分布性后果 | 表示充分性、分布外误差、多步滚动 |
| Value-equivalent / decision-oriented model | 保留下游决策所需量 | 保证通常相对于某类策略/价值，不是全世界精确模型 |

<a id="research-model-query-equivalence"></a>

## 研究专题 A · 模型充分性由规划查询与风险目标共同决定

本章的期望模型已说明线性价值下哪些统计足够。Value Equivalence（NeurIPS 2020）进一步把模型规格写成查询集合：指定哪些策略与后续函数，要求模型在这些查询上生成正确 backup。模型可以舍弃不影响这些计算的细节；当规划器或奖励改变时，原先可忽略的细节也可能变成必要知识。

$$
(T_M^\pi v)(s)=\mathbb E_{M,\pi}[R+\gamma v(S')\mid s],\qquad \widehat M\equiv_{\Pi,\mathcal V}M\ \Longleftrightarrow\ T_{\widehat M}^\pi v=T_M^\pi v\quad\forall\pi\in\Pi,\ v\in\mathcal V
$$

等价针对指定策略与函数族。用一个当前 critic 拟合 targets，只检验一个有限且会变化的查询集合；VE 不是说任意小模型都足够，也不是把 reward-only 预测称为完整环境模型。

Proper Value Equivalence（NeurIPS 2021）考察多步算子与策略价值固定点，给出适当策略族下的规划充分性。Distributional Model Equivalence（NeurIPS 2023）揭示它的另一边界：保持期望值不足以保持风险敏感决策。后者需保留回报分布或与指定风险目标相容的统计摘要。

手算反例：动作 A 确定获得 1；动作 B 以各半概率获得 −9 或 11；两者期望均为 1。只保持均值的模型可以把它们当作同一动作，但最差一半的平均回报分别为 1 和 −9。若目标由最大期望改为最大 lower-tail CVaR，旧模型无法回答新问题。这个反例不需要非平稳环境，改变查询规格本身就能造成模型不足。

$$
\mathcal S(\nu)=(\mathbb E_\nu[G],\mathbb E_\nu[G^2],\ldots),\qquad \mathcal S(\mathcal T^\pi\eta)=\mathcal T_{\mathcal S}^\pi\mathcal S(\eta)
$$

第二式表达摘要的 Bellman 闭合要求：更新后的统计应能由已有统计正确计算。完整分布保留更多信息；有限矩、分位点与投影各有局限，均值加方差通常不能识别任意尾部风险。不要把任意摘要都默认为满足这条闭合式。

**算法：模型 loss、查询误差和决策结果是三类证据**

1. 模型用途实验（拟议）：
  1. 固定真实数据、编码器、模型容量与优化预算
  1. 分别训练状态预测模型、当前价值等价模型、指定分布摘要模型
  1. 在未用于拟合的策略/后续价值/奖励查询上测 target 误差
  1. 对相同候选动作同时报告均值、选定尾部指标与真实动作排序
  1. 只改变风险目标，再只改变动力学，区分规格不足与环境漂移

持续模型研究可以由此提出清楚的假设：固定容量模型应按未来规划查询而非仅按观测频率分配表示。怎样发现新查询、保留旧风险事件，并及时判断模型的等价规格失效，仍是开放问题。模型等价定理也不提供有限采样下的安全保证。

<a id="research-model-frozen-visual-dynamics"></a>

## 研究专题 B · 冻结视觉特征后的动力学：DINO-WM 与 V-JEPA 2-AC

两条近年的视觉模型路线都把“看到什么”与“执行动作后发生什么”分开，但外部经验来源不同。DINO-WM（ICML 2025）在 DINOv2 patch 特征上用离线行为轨迹拟合动力学；V-JEPA 2（2025 首稿）先以无动作标注的视频学潜在预测，再冻结编码器，用机器人交互轨迹训练 2-AC 动作条件预测器。没有像素 decoder，并不意味着没有动力学数据。

$$
z_t=e_{\rm frozen}(o_t),\qquad \hat z_{t+1}=f_\theta(z_{t-L+1:t},a_{t-L+1:t}),\qquad \mathcal L_{\rm pred}=\sum_{k=1}^{H}\ell(\hat z_{t+k},\operatorname{sg}[e_{\rm frozen}(o_{t+k})])
$$

这是两类方法的教学性共同接口，历史长度、动作/机器人状态输入、损失距离与预测 rollout 方式以各原文为准。冻结 e 稳定了模型输出坐标；训练看到真实未来观测，不意味着测试规划可以读取它。

例子：同一物体的视觉颜色改变，冻结 encoder 可能依然给出相近的任务特征，模型因而迁移；摩擦系数改变却会使相同动作产生不同位置，即使视觉编码完全稳定，f 也必须更新。若把 e 也在线更新，第三种问题出现：旧模型预测的坐标与当前目标图像的坐标可能不再相同。三类变化需要独立实验。

| 数据与接口 | DINO-WM | V-JEPA 2 / 2-AC |
| --- | --- | --- |
| 预训练来源 | DINOv2 图像特征 | 大规模图像/视频的潜在预测 |
| 动作条件阶段 | 离线行为轨迹上预测 patch 特征 | 机器人轨迹上后训练动作条件模型 |
| 目标 | 观测目标的特征距离 | 机器人图像目标的潜在距离 |
| 部署使用 | 优化动作序列并滚动重规划 | 2-AC 模型支持图像目标 MPC |
| 持续更新证据 | 原文主要检验离线学习后规划 | 首稿主要检验冻结模型零样本机器人部署 |

先在作者公开检查点和已支持环境上验证训练/规划接口，再用单变量扰动做预测误差与真实目标成功率的联合评价。图像目标很近时仍可能物理碰撞；视觉目标代价也可能漏掉执行中的负奖励。因此若研究奖励控制，还需要额外 reward/风险接口，不能把视觉相似度默认成环境目标。

两个工作给 CRL 的机会是稳定且可迁移的起始表示，但未来学习必须另行检验。可比较冻结 e 仅更新 f、联合更新 e/f、以及使用兼容约束的三种方案，固定每步训练预算，测新后果学习、旧查询保留与控制恢复。预训练资源与在线维护资源应各自记录；首稿结果不是自动问题发现、option 构造或终生世界模型维护的证明。

<a id="lesson-check"></a>

## 9. 失败诊断与自测答案

- 模型 loss 很低，规划却变坏：检查训练分布与 planner 查询分布、是否漏掉罕见高代价事件、是否只预测均值而下游非线性。
- 长技能被过分偏爱：检查折扣终点模型是否错误归一化、是否用平均 τ 替换随机 τ、是否漏记沿途 reward。
- 换奖励后 SF 迁移失败：检查新奖励是否真在特征张成空间、旧策略 SF 是否准确、动力学是否同时变了。
- 同名技能更新后旧模型失效：不是随机异常，而是条件行为已改变；要么重估模型，要么让版本与参数成为显式条件。

自测 1：$p_o^\gamma$ 的行和小于一是否代表概率遗漏？答：它包含时间折扣，行和为 $\mathbb E[\gamma^\tau]$。自测 2：把每个奖励特征各自的最优累计值相加，能否得到新奖励的最优值？答：各分量的最优策略可能冲突，SF 必须条件化于同一策略。自测 3：为什么 stopping bonus 不进入 $r_o$？答：它是训练技能的辅助目标，而主任务模型预测执行技能真正得到的环境奖励。

## 本章的实验设计

模型预测误差与决策损失需要分别测量。对规划无关变量的准确预测，不一定改善策略。

设定：固定同一真实数据集训练后果模型，再在独立轨迹和规划实际访问的状态上评价。加入随机时长与终点相关的 option。

- 折扣后果模型保留持续时间与终点的相关性。
- 奖励、continuation 和后继特征目标分别核对。
- 模型生成的虚拟奖励不计作真实环境收益。

对照：准确模型诊断与学习模型；随机分布测试与规划访问分布测试；固定消费者只替换模型

记录：一步及多步预测误差；动作排序、价值误差和模型使用分布；真实执行表现及模型计算成本

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-planning)

## 学习与研究衔接

模型不必重建全部观测。应预测规划真正需要的量，并测试模型误差如何改变决策。

[分册导读](https://yingwen.io/zh/continual-rl/start/deep-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-models) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=models) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=models)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

状态是支持后续计算的内部信息；GVF 指定一个预测问题；RTRL 和资格迹规定信用如何传播。三者可以组合，但不是相互替代的算法名称。先理解给定策略的预测，再讨论预测怎样改善控制。

- [Does Zero-Shot Reinforcement Learning Exist?](https://yingwen.io/zh/continual-rl/research/#recent-zero-shot-forward-backward)
- [Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations](https://yingwen.io/zh/continual-rl/research/#recent-successor-flow-features)

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

Laplacian 描述行为图结构，奖励感知表示加入路径价值，METRA 学习有区别的行为，HIQL 利用离线目标轨迹，MaestroMotif 引入语言先验。它们承担不同的设计工作；生成技能、选择技能与组合技能需要分别评价。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [Reward-Aware Proto-Representations in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-reward-aware-proto-representations)
- [Laplacian Keyboard: Beyond the Linear Span](https://yingwen.io/zh/continual-rl/research/#recent-laplacian-keyboard)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

模型可提取性的理论说明某类能力需要什么知识，不指定唯一网络。Dreamer 研究潜在想象控制，STOMP 研究随机时长行为模型，DRAGO 研究旧模型知识保留。模型误差、查询策略和规划收益之间仍需实验连接。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [Mastering diverse control tasks through world models](https://yingwen.io/zh/continual-rl/research/#recent-dreamerv3-world-models)
- [Knowledge Retention in Continual Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-drago-model-retention)
- [General Agents Contain World Models](https://yingwen.io/zh/continual-rl/research/#recent-general-agents-world-models)
- [Laplacian Keyboard: Beyond the Linear Span](https://yingwen.io/zh/continual-rl/research/#recent-laplacian-keyboard)
- [The Value Equivalence Principle for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-value-equivalence-models)
- [Distributional Model Equivalence for Risk-Sensitive Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-distributional-model-equivalence)
- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://yingwen.io/zh/continual-rl/research/#recent-tdmpc2-decision-time-model)
- [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://yingwen.io/zh/continual-rl/research/#recent-dino-wm-feature-planning)
- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://yingwen.io/zh/continual-rl/research/#recent-vjepa2-action-conditioned)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [Knowledge Retention in Continual Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-drago-model-retention)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文提供定义和条件，架构讲座提出模块组织，算法论文检验特定机制。完整系统还要明确智能体、外部设计者和世界各自承担的工作；组件成立不自动意味着组合后的长期收益成立。

- [General Agents Contain World Models](https://yingwen.io/zh/continual-rl/research/#recent-general-agents-world-models)
- [The OaK Architecture: A Vision of SuperIntelligence from Experience](https://yingwen.io/zh/continual-rl/research/#recent-oak-architecture)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)
- [The Value Equivalence Principle for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-value-equivalence-models)
- [Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations](https://yingwen.io/zh/continual-rl/research/#recent-successor-flow-features)

### Reward-Respecting Subtasks for Model-Based Reinforcement Learning

Richard S. Sutton, Marlos C. Machado, G. Zacharias Holland, David Szepesvari, Finbarr Timbers, Brian Tanner, Adam White

Artificial Intelligence · 2023 · 支持方法与理论

#### 研究问题

学到一个能到达子目标的技能之后，为什么它仍可能不适合主任务规划？

#### 关键机制

STOMP 把子任务、option、模型和规划连起来。子任务保留原任务的路径奖励，并用带有特征偏好的终止价值表达目标；学习得到策略和终止规则后，再预测该行为的累计奖励与折扣终点。这样，技能不会因为只追求到达子目标而忽略途中代价。

#### 证据

论文用明确的小问题展示奖励感知子任务如何产生更有用的行为和规划模型。它提供的是可分析的构造链，而非只比较一个技能执行成功率。

#### 条件与限制

终止收益的约定是子任务定义的一部分，不能随意换成固定终点奖励。特征和子任务候选的选择尚不等于完整自主发现机制；实验也不构成整个 OaK 架构的验证。

#### 阅读与实验

在同一个绕路环境中比较“最短到达目标”和“保留路径奖励”的子任务。分别计算 option 的奖励模型、折扣终点模型与一次规划备份。

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

### Laplacian Keyboard: Beyond the Linear Span

Siddarth Chandrasekar, Marlos C. Machado

arXiv 预印本 · 2026 · 支持方法与理论

#### 研究问题

从一组谱技能出发，能否解决超出原特征线性奖励空间的新任务？

#### 关键机制

Laplacian 特征先定义行为基，并借助后继特征预测各行为的后果。固定任务权重的价值组合受特征张成空间限制；论文进一步使用随状态变化的元策略，在不同位置组合已有行为。关键变化是组合规则从一组全局固定权重变为状态相关的行为选择。

#### 证据

论文对行为基与任务组合给出理论分析，并报告有限环境中的组合实验。它延续 eigenoptions 与 successor features 的路线，同时解释了为什么单纯线性读出会遇到表达边界。

#### 条件与限制

理论结论依赖具体的行为基、近似误差和任务条件。技能集合的长期生成、淘汰与非平稳模型维护仍是另外的问题；此处按预印本收录，不指定未经确认的会议。

#### 阅读与实验

构造一个必须在中途切换方向的奖励任务。分别比较固定权重的技能选择与状态相关切换，并解释性能差异来自哪里。

#### 原文与相关入口

- [作者预印本](https://arxiv.org/abs/2602.07730)：阅读线性张成空间的限制及状态相关组合机制。

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

### Mastering diverse control tasks through world models

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap

Nature · 2025 · 支持方法与理论

#### 研究问题

同一套世界模型训练与控制方法，能否减少跨任务重新设计损失和超参数的需求？

#### 关键机制

DreamerV3 从经验学习递归潜在状态、奖励和延续预测，再在潜在想象轨迹上学习 actor 和 critic。尺度稳健的表示与损失设计使同一配置可以适用于多种任务。模型是用于决策的学习接口，不必生成完整真实世界。

#### 证据

论文在大量视觉和状态控制任务上报告了广泛表现。关键含义是共享算法配置；这些结果主要来自分别训练的任务智能体，不是一个智能体按顺序学会全部任务。

#### 条件与限制

经验重放、批量训练和模型想象都有资源成本。模型偏差、表示遗忘与长期任务切换仍需要专门实验，不能由多任务覆盖范围自动推出持续学习能力。

#### 阅读与实验

把状态更新、模型训练、想象起点和策略更新四种分布分别写清。比较真实交互步数之外，还应记录想象步数和优化次数。

#### 原文与相关入口

- [Nature 原文](https://www.nature.com/articles/s41586-025-08744-2)：方法和任务协议；区分共享配置与单智能体持续学习。
- [作者维护的实现](https://github.com/danijar/dreamerv3)：公开实现的版本与论文实验环境应分别记录。

#### 作者代码

[作者发布的重实现；不把当前分支当作原论文实验的冻结快照。](https://github.com/danijar/dreamerv3)

DreamerV3 的作者维护公开实现及运行配置。

### General Agents Contain World Models

Jonathan Richens, David Abel, Alexis Bellot, Tom Everitt

ICML 2025 · 2025 · 支持方法与理论

#### 研究问题

能完成足够丰富的目标集合，是否意味着智能体内部已经包含可提取的环境预测知识？

#### 关键机制

论文在形式化条件下，将广泛多步目标上的行为能力与环境模型的可提取性联系起来。通过查询智能体对不同目标的行为，可以恢复关于环境后果的信息；目标集合和性能要求越强，所要求的预测知识也越强。

#### 证据

主要证据是给定假设下的理论结果，而不是某个世界模型架构在所有任务上击败无模型算法的实验。

#### 条件与限制

可提取模型不等于智能体显式保存一个 RSSM，也不意味着所有实用任务都需要重建全部环境。必要知识的结论不能代替如何高效学到它的算法。

#### 阅读与实验

列出定理要求的目标丰富性和查询能力，再尝试构造一个只会单一任务的反例。由此区分任务专门知识与支持广泛目标的预测模型。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2506.01622)：形式化设定、模型可提取性与证明。
- [David Abel 论文目录](https://david-abel.github.io/papers.html)：作者提供的 ICML 2025 发表信息及相关研究。

### The OaK Architecture: A Vision of SuperIntelligence from Experience

Richard S. Sutton

RLC 2025 讲座 / Oak Lab · 2025 · 定义与架构观点

#### 研究问题

持续学习是否只是在一个现成 actor–critic 上加入抗遗忘机制，还是需要重新安排知识构造与使用？

#### 关键机制

OaK 提出从经验持续形成状态、预测知识、子任务、时间抽象与模型，并让这些知识服务规划的架构方向。这里的重点是模块之间怎样产生可复用知识，而不只是保留某个固定策略网络的参数。

#### 证据

官方页面提供 Richard Sutton 的架构讲座与相关研究入口。STOMP、预测学习和在线特征学习等论文可以检验其中具体组件，但不能自动验证整体架构。

#### 条件与限制

这是研究愿景与架构讲解，不是一套已公布完整训练配方、统一基准结果和可复现端到端代码的系统。资源分配、问题生成、知识替换与模块相互干扰仍需明确算法。

#### 阅读与实验

为每个模块写出输入、输出、更新频率和资源上限。再选择一个双模块接口做可证伪实验，例如技能模型改善是否真的减少规划误差。

#### 原文与相关入口

- [Oak Lab 官方讲座页面](https://oaklab.ai/posts/the-oak-architecture)：讲座入口与架构研究方向。
- [Oak Lab 研究主页](https://oaklab.ai/)：区分已发表研究、技术文章和仍在预告中的项目。

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

### Does Zero-Shot Reinforcement Learning Exist?

Ahmed Touati, Jérémy Rapin, Yann Ollivier

ICLR 2023 · 2023 · 支持方法与理论

#### 研究问题

没有事先指定奖励时，怎样学一套预测表示，日后接收新奖励就能选行为？

#### 关键机制

Forward–Backward 表示联合学习行为条件的未来占用与奖励读出，而非先固定任意编码器再学习 successor features。新奖励被映射到任务向量，策略根据这个向量直接行动；该论文系统比较 FB 与多种 SF 基础特征。

#### 证据

原文在固定离线 replay buffers 上比较零样本任务迁移，借此把表示学习与探索数据的质量分开。不同特征与数据覆盖产生显著差异，不能仅靠“所有奖励”的理论目标预测实际效果。

#### 条件与限制

假定共享动力学与可用经验覆盖。无下游梯度更新不等于无预训练成本；有限秩、近似训练和奖励估计都有误差。新动力学、历史混叠和严格一次使用经验均须另测。

#### 阅读与实验

同一 buffer 对比随机特征、谱特征与联合 FB，再独立换 buffer。奖励读出误差、占用误差与新任务回报分别报告，避免把数据覆盖优势记成表示优势。

#### 原文与相关入口

- [作者原文](https://arxiv.org/abs/2209.14935)：2022 首稿，ICLR 2023；比较奖励表示、SF 与 FB。
- [作者研究平台](https://github.com/facebookresearch/controllable_agent)：README 直接关联两篇 FB 论文；该仓库已经归档。

#### 作者代码

[论文作者团队仓库；归档工程，依赖和旧环境需单独核验。](https://github.com/facebookresearch/controllable_agent)

FB 与 SF 的训练、固定数据实验及奖励查询示例。

### Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations

Haosen Shi, Jianda Chen, Sinno Jialin Pan

ICLR 2026 · 2026 · 支持方法与理论

#### 研究问题

能否直接学习多步未来状态的分布，并把它压缩为适合控制学习的特征？

#### 关键机制

SF² 以 flow matching 估计 successor measure，将条件向量场分解为未来位置及生成时间的投影与当前状态动作特征的乘积。特征进入 TD3/SAC 的 critic；线性的是向量场对条件特征的分解，critic 本身可以非线性。

#### 证据

正式原文给出 mixture Bellman 结构、生成式 bootstrap 与控制实验，并提供作者 JAX/Brax 仓库。实验研究在线收集数据下的 off-policy 控制，并使用 replay、批次与目标网络。

#### 条件与限制

“online policy learning”不代表 strict streaming。生成时间不是环境时间；向量场线性不保证任意奖励价值线性。文中与 SR 的小生成时间联系是近似动机，未证明递归 agent state 或任意持续变化下的充分性。

#### 阅读与实验

对齐模型调用与梯度预算，拆分直接预测、bootstrap、critic 联合训练。冻结特征后比较线性与非线性读出，再测新奖励和动力学变化，才能检验预测知识的可复用程度。

#### 原文与相关入口

- [ICLR 2026 正式原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48acf4b231771e693f42305b4c9b4c9f-Abstract-Conference.html)：第 2–3 节和算法附录；区分 flow 时间、环境时间与近似 SR 联系。
- [原文链接的作者实现](https://github.com/Shiien/successor-flow-representation-implementation)：SAC/TD3、flow 特征、对照和 sweep 配置。

#### 作者代码

[正式论文摘要直接链接的作者代码。](https://github.com/Shiien/successor-flow-representation-implementation)

基于 JAX/Brax 的 SF² 控制实验；不包含自动 GVF 问题发现或完整持续架构。

### Constructing an Optimal Behavior Basis for the Option Keyboard

Lucas N. Alegre, Ana L. C. Bazzan, André Barreto, Bruno C. da Silva

NeurIPS 2025 · 2025 · 支持方法与理论

#### 研究问题

状态相关的技能组合足够强时，是否仍须为每个新奖励保存一条完整策略？

#### 关键机制

OKB 联合扩充基础策略与 Option Keyboard 的元策略。线性支持方法选择需要补齐的任务权重，先训练现有基础上的组合，再检查无法表达的动作并新增基础，移除冗余项。优化的是可组合的行为基，而非仅增加技能数量。

#### 证据

正式原文分析基础数量与 convex coverage set 的关系，并在多任务领域检验规模与表现。附录提供元策略、新基础训练和角点枚举等实现细节；论文声明实验代码在 Supplemental Material。

#### 条件与限制

保证假定 NewPolicy 返回最优策略，且 TrainOK 能达到可表达的最优组合。近似 critic、有限训练、变化动力学不直接继承保证；非线性任务结论仅覆盖最优行为可由相应子策略组合的类别。

#### 阅读与实验

分别比较“增加基础”“只训练组合”和“去除冗余”。保留一组未用于基构造的奖励权重，记录基础规模、元策略成本、SF 误差与迁移回报。持续淘汰仍需未来任务效用检验。

#### 原文与相关入口

- [NeurIPS 2025 原文与补充材料入口](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0ab48777def88e73b50746a6011be0b0-Abstract-Conference.html)：算法 1–3、附录 A.3 的两个最优子程序假设及代码声明；未在本教材运行补充代码。
- [Option Keyboard 的经典桥梁](https://proceedings.neurips.cc/paper/2019/file/251c5ffd6b62cc21c446c963c76cf214-Paper.pdf)：cumulant 组合、GPE/GPI 与技能接口。

### The Value Equivalence Principle for Model-Based Reinforcement Learning

Christopher Grimm, André Barreto, Satinder Singh, David Silver

NeurIPS 2020 · 2020 · 支持方法与理论

#### 研究问题

模型容量有限时，必须预测全部状态细节，还是只须保持规划会查询的量？

#### 关键机制

Value equivalence 以策略集合和函数集合定义模型规格：模型对这些函数进行这些策略的 Bellman backup，应与真实环境相同。扩大查询族会缩小可接受模型族；它把“决策相关”从口号变成有条件的等价关系。

#### 证据

论文给出等价模型类的性质及有限实验，并解释若干隐式模型方法。后续 Proper Value Equivalence（NeurIPS 2021）研究策略价值固定点等价及规划充分性。

#### 条件与限制

少数当前 critic 的 backup 相同，不说明所有新奖励、新策略或风险目标都相同。精确算子等价与神经损失在样本上较小不同；奖励或查询族变化后须重新验证。

#### 阅读与实验

保存独立的 planner 查询集，直接测 target 误差和动作排序。用未参与模型拟合的价值函数检验迁移，并与像素误差对照，找出模型实际保留的信息。

#### 原文与相关入口

- [NeurIPS 2020 原文](https://papers.nips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)：VE 依赖策略与函数集合。
- [Proper Value Equivalence · NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/400e5e6a7ce0c754f281525fae75a873-Abstract.html)：多步算子、固定点与规划充分性；不是任意潜在网络的保证。

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

### TD-MPC2: Scalable, Robust World Models for Continuous Control

Nicklas Hansen, Hao Su, Xiaolong Wang

ICLR 2024 · 2024 · 支持方法与理论

#### 研究问题

如何让短期动力学与长期价值分工，并在动作选择时继续使用模型？

#### 关键机制

TD-MPC2 学习无需观测 decoder 的潜在动力学、奖励、价值与策略先验。决策时优化有限动作序列，用终点价值补上未展开的后果；执行第一步后，利用新观测重新规划。

#### 证据

正式会议原文报告 104 个在线任务和单一大型多任务智能体的实验。官方仓库包含模型训练与计划接口，适合与 Dreamer 的想象 actor 学习比较计算位置。

#### 条件与限制

跨任务共享超参数和多任务能力不是单条生命流中持续适应的证据。replay、任务条件、模型更新、决策延迟等成本需进入 CRL 协议；长程 critic 错误不能被短期模型精度自动修复。

#### 阅读与实验

固定模型，对比无终点价值、不同 horizon 和不同规划预算；再固定预算比较部署 actor 与决策时搜索。环境变化后同时记录模型校准、critic 误差和恢复收益。

#### 原文与相关入口

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html)：短期预测、终点价值、多任务协议。
- [作者实现](https://github.com/nicklashansen/tdmpc2)：训练、模型与 plan 函数分别阅读。

#### 作者代码

[作者维护的原论文代码。](https://github.com/nicklashansen/tdmpc2)

TD-MPC2 的单任务/多任务训练和决策时规划。

### DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning

Gaoyue Zhou, Hengkai Pan, Yann LeCun, Lerrel Pinto

ICML 2025 · 2025 · 支持方法与理论

#### 研究问题

预训练视觉表示能否直接成为动作后果预测与目标规划的接口？

#### 关键机制

冻结 DINOv2 空间 patch 特征，用离线动作轨迹学习未来特征预测器；测试时优化动作序列，让预测特征接近目标图像特征。没有重建图像、奖励模型或逆模型，不表示没有动作条件的动力学训练。

#### 证据

ICML 原文在六类环境检验视觉目标规划，作者仓库公开数据、部分检查点、训练与 CEM 规划入口。零样本指给定已训练模型后解决目标，无额外任务策略训练。

#### 条件与限制

依赖视觉预训练与离线交互覆盖；patch 相近不总等于任务完成或风险相同。原实验不证明冻结视觉表示能适应长期新物体、新动作语义或隐藏状态。

#### 阅读与实验

分别改变背景、物体属性、控制动力学与目标分布。把冻结 encoder 和联合更新 encoder 分开，对照视觉距离、真实成功与模型误差，观察表示漂移的依赖成本。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/zhou25t.html)：正式发表入口；早期 ICLR 投稿页不能替代此状态。
- [作者项目代码](https://github.com/gaoyuezhou/dino_wm)：train.py、plan.py、数据与已公开模型检查点范围。

#### 作者代码

[原作者 Gaoyue Zhou 的论文配套仓库。](https://github.com/gaoyuezhou/dino_wm)

DINO 特征预测、离线环境数据与目标规划；README 公开部分环境检查点。

### V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, Nicolas Ballas

arXiv 预印本（此处采用 2025 首稿） · 2025 · 支持方法与理论

#### 研究问题

无动作标注的视频预训练，与能接受机器人动作的规划模型之间还缺哪一步？

#### 关键机制

V-JEPA 2 先学被遮蔽视频的潜在特征预测；V-JEPA 2-AC 冻结编码器，再用机器人轨迹训练动作条件预测器。控制以目标图像的特征差为代价进行 MPC；视频理解、动作条件预测和真实控制是三个独立证据层。

#### 证据

2025 首稿报告以大规模视频预训练，再用不到 62 小时 DROID 交互视频后训练，在两个实验室以图像目标做真实机器人规划。论文单独讨论相机位置、长程规划与图像目标的局限。

#### 条件与限制

无任务奖励并不等于无动作、无机器人状态或无外部数据。零样本部署未持续更新模型，也未发现和维护 options。官方仓库现含 V-JEPA 2.1，复现首稿须记录配置和模型版本。

#### 阅读与实验

按视觉编码、动作坐标、后果模型、目标代价逐项做迁移检验。若引入在线更新，记录模型更新使旧目标接口失效的程度，测未来交互收益，而非仅用 frozen probe 证明 CRL。

#### 原文与相关入口

- [2025 首稿](https://arxiv.org/abs/2506.09985v1)：action-free 预训练、2-AC 后训练、真实规划与第 4.3 节限制；此处不赋予未核实会议状态。
- [Meta FAIR 官方实现](https://github.com/facebookresearch/vjepa2)：包含 V-JEPA 2、2-AC 和较新的 2.1；版本不能混用。

#### 作者代码

[Meta FAIR 官方仓库；首稿模型与后续版本需按配置区分。](https://github.com/facebookresearch/vjepa2)

官方视频表征与动作条件模型；数据、机器人部署条件与检查点分别核验。


<a id="chapter-code"></a>

## 下载与运行

表格 option model、SF/GPI 与条件反例；不包含完整深度世界模型训练。

[下载 knowledge_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/knowledge_algorithms_lab.py)

```sh
python3 knowledge_algorithms_lab.py models
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton, Precup & Singh — Options 的 Bellman 模型](https://doi.org/10.1016/S0004-3702(99)00052-1)：原始框架。先比较完整轨迹定义与一步递推，再核对 discount 已包含在 transition model 中。

- [Sutton et al. — Reward-Respecting Subtasks，模型部分](https://arxiv.org/html/2202.03466v3)：原文第 4 节将 reward model 与 expectation model 分开；子任务 stopping bonus 不能混入环境 reward model。

- [Wan et al. — Planning with Expectation Models](https://arxiv.org/abs/1904.01191)：原文。理解线性状态价值下期望模型的充分性及函数逼近前提。

- [Planning with Expectation Models for Control](https://arxiv.org/abs/2104.08543)：控制与期望模型的接口；重点比较线性预测的等价条件与包含非线性 Q/max 的控制更新。

- [Barreto et al. — Successor Features for Transfer in RL](https://arxiv.org/abs/1606.05312)：原文。先查奖励分解，再查目标策略条件与 GPI 改善保证的精确/近似前提。

- [DeepMind — Option Keyboard 作者工程](https://github.com/google-deepmind/deepmind-research/tree/f5de0ede8430809180254ee957abf36ed62579ef/option_keyboard)：研究团队代码；keyboard_agent.py 的策略、cumulant、动作三个维度与本页 GPI 计算顺序对照。不是所有 SF 方法的统一实现。

- [Reward-Aware Proto-Representations in RL · NeurIPS 2025](https://arxiv.org/abs/2505.16217)：DR 的定义、线性可解控制推导与谱特征用途；注意内部奖励固定和终点收益改变是不同的迁移条件。

- [Reward-Aware Proto-Representations 作者实现](https://github.com/httse9/Reward-Aware-Proto-Representations)：从默认转移和奖励构造 DR，比较奖励感知特征、奖励塑形与技能发现；读取数值求解与谱处理设置。

- [DR 矩阵构造 — rep_utils.py](https://github.com/httse9/Reward-Aware-Proto-Representations/blob/master/minigrid_basics/examples/rep_utils.py)：compute_SR 与 compute_DR 并排给出两个矩阵的区别；get_representation 中的对称化决定了随后分解的谱对象。

- [DR 在线表示与发现 — ROD_DR.py](https://github.com/httse9/Reward-Aware-Proto-Representations/blob/master/minigrid_basics/examples/ROD_DR.py)：learn_representation 对应 DR 的采样递推；compute_eigenvector 从访问过的状态计算谱特征。结合数据采集循环阅读，辨认其经验转移实际对应的默认行为分布。

- [Hafner et al. — DreamerV3](https://arxiv.org/abs/2301.04104)：原文。区分真实序列模型学习与想象序列 actor/critic 学习，保留完整损失与超参数。

- [Danijar Hafner — DreamerV3 作者维护实现](https://github.com/danijar/dreamerv3)：作者维护的公开重实现；README 明确其为 reimplementation，并非原内部训练代码。先追 model/imagined rollout/actor 三个接口。

- [Schrittwieser et al. — MuZero](https://arxiv.org/abs/1911.08265)：原文。预测 reward、policy、value 的潜在模型与决策时树搜索；不以观测重建作为必要接口。

- [作者原文](https://arxiv.org/abs/2209.14935)：2022 首稿，ICLR 2023；比较奖励表示、SF 与 FB。

- [作者研究平台](https://github.com/facebookresearch/controllable_agent)：README 直接关联两篇 FB 论文；该仓库已经归档。

- [ICLR 2026 正式原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48acf4b231771e693f42305b4c9b4c9f-Abstract-Conference.html)：第 2–3 节和算法附录；区分 flow 时间、环境时间与近似 SR 联系。

- [原文链接的作者实现](https://github.com/Shiien/successor-flow-representation-implementation)：SAC/TD3、flow 特征、对照和 sweep 配置。

- [NeurIPS 2025 原文与补充材料入口](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0ab48777def88e73b50746a6011be0b0-Abstract-Conference.html)：算法 1–3、附录 A.3 的两个最优子程序假设及代码声明；未在本教材运行补充代码。

- [Option Keyboard 的经典桥梁](https://proceedings.neurips.cc/paper/2019/file/251c5ffd6b62cc21c446c963c76cf214-Paper.pdf)：cumulant 组合、GPE/GPI 与技能接口。

- [NeurIPS 2020 原文](https://papers.nips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)：VE 依赖策略与函数集合。

- [Proper Value Equivalence · NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/400e5e6a7ce0c754f281525fae75a873-Abstract.html)：多步算子、固定点与规划充分性；不是任意潜在网络的保证。

- [NeurIPS 2023 原文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b0cd0e8027309ea050951e758b70d60e-Abstract-Conference.html)：proper VE 的不足、统计摘要与 Bellman 闭合。

- [作者实现](https://github.com/tylerkastner/distribution-equivalence)：原文第 7 节直接链接的实验代码。

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html)：短期预测、终点价值、多任务协议。

- [作者实现](https://github.com/nicklashansen/tdmpc2)：训练、模型与 plan 函数分别阅读。

- [ICML 2025 原文](https://proceedings.mlr.press/v267/zhou25t.html)：正式发表入口；早期 ICLR 投稿页不能替代此状态。

- [作者项目代码](https://github.com/gaoyuezhou/dino_wm)：train.py、plan.py、数据与已公开模型检查点范围。

- [2025 首稿](https://arxiv.org/abs/2506.09985v1)：action-free 预训练、2-AC 后训练、真实规划与第 4.3 节限制；此处不赋予未核实会议状态。

- [Meta FAIR 官方实现](https://github.com/facebookresearch/vjepa2)：包含 V-JEPA 2、2-AC 和较新的 2.1；版本不能混用。


---

# 规划：把模型中的经验转成更好的决策

真实交互很贵、计算预算有限时，怎样决定想象什么、更新什么，以及何时应该不再相信模型？

## 本章内容

- 独立实现真实学习—模型学习—模型规划三条交织的 Dyna 循环。
- 理解 prioritized sweeping 如何沿前驱传播变化，并与 prioritized replay 区分。
- 把 option model 变成正确 backup，推导收缩和模型误差放大。
- 解释 MPC、MCTS/MuZero 与 Dreamer 在何时计算、更新对象、误差来源上的差异。

<a id="problem-definition"></a>

## 本章的问题定义

已有规则模拟器或学得后果模型，在有限额外计算内改善价值或当前决策。模型查询不作为真实环境证据。

### 给定条件与符号

- 奖励与转移/技能模型、当前状态及可能的尾值和行为先验。
- 候选行为、搜索时域、备份/模拟次数和行动延迟预算。

### 需要求解的对象

当前模型下可用的价值/动作近似，以及明确分离模型、截断和求解误差的实际控制。

### 信息与数据权限

规划只查询已给定或由过去经验学得的模型；真实行动后获得新观测，再根据协议重规划或更新模型。

$$
\hat J_H(a_{0:H-1};s)=\mathbb E_{\hat P}\!\left[\sum_{k=0}^{H-1}\gamma^k\hat r(S_k,a_k)+\gamma^H\hat V(S_H)\mid S_0=s\right]
$$

$\hat P,\hat r$ 为当前模型，$H$ 为规划时域，$\hat V$ 为尾值，$\gamma$ 为原始步折扣。这是MPC型局部模型目标；Dyna/option迭代求模型固定点，MCTS自适应搜索，想象训练将计算存入actor，不能全部视作同一求解器。

### 成立条件与解的含义

- 收缩结论需固定合法模型和相同候选集；option模型已含时长折扣，不能再乘一次。
- 模型误差、模型外查询与尾值误差均影响真实后果；静态离线覆盖不能免费推广到长期漂移。

判断准则：小链优先传播按前驱得到1、0.9、0.81；option固定点约14.736842，枚举MPC首动作+1；一般任务同时报告模型内值、真实收益和行动延迟。

### 适用边界

- 规划次数增加不保证真实回报增加。
- 区域谱距离不自动等于最优单向到达时间。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 推广：放宽条件 · [Dyna 与模型学习](https://yingwen.io/zh/continual-rl/algorithms/dyna/)：规划还包含行动时搜索、技能备份与想象训练，Dyna仅是其中一种组合。

- 组合不同学习问题 · [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)：模型的下游误差决定规划结果；更多求解计算不能消除错误模型。

- 组合不同学习问题 · [Options 与技能发现](https://yingwen.io/zh/continual-rl/construction/options/)：技能模型将多个原始步压缩成一次备份，同时引入合法启动、随机时长与学习成本。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

短视搜索漏掉远期收益，长展开积累模型误差；计算应投向哪些分支也是决策。

### 本章的核心思路

将模型、尾值和计算调度分别定义，选择展开或摊销机制并保留真实反馈校准。

1. [为价值传播分配有限备份](#lesson-priority)：因为后继变化只影响部分前驱，用优先队列传播而非重算全部状态。

2. [按行为跨度拆分备份](#lesson-options)：因为技能执行跨越多个原始步，option备份使用已经联合折扣的终点后果，不能再折扣一次。

3. [将短期搜索与长期尾值连接](#lesson-search)：因为长展开会累积模型误差，MPC用短展开加尾值，MCTS把有限模拟分配给所需分支。

4. [按部署延迟摊销模型计算](#lesson-imagination)：因为部署不总能支付树搜索，想象actor把模型计算存入参数；需检验模型偏差是否被固化。

结论与条件：固定精确有限折扣option模型在给定集合上可收缩；MPC/MCTS有限求解与学得深度模型没有本章提供的真实最优保证。

### 相关方法改变了什么

- Dyna/优先传播：学习时更新价值，调度备份而非搜索全部动作序列。

- MPC/MCTS：行动时分别优化序列或分配树搜索，支付当前决策延迟。

- 想象训练actor：模型计算摊销进策略，部署快但可能固化模型偏差。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 规划的操作性定义

用模型得到的后果，而非新发生的一次真实转移，改善价值、策略或当前动作选择。模型可以是已知模拟器，也可以是从经验学来的；“有网络”不意味着“有模型”。

### Bellman backup

把即时 reward 与下一状态的估计价值合成当前标签。一次 backup 可以是赋值（value iteration）或小步拟合（TD）。

$$
y=r+\gamma(1-d)\max_{a'}Q(s',a'),\qquad Q(s,a)\leftarrow Q(s,a)+\alpha[y-Q(s,a)]
$$

### 模型接口

一步模型给 $(r,s',d)$ 的样本或分布；option 模型给技能内折扣奖励 $r_o$ 和已经含 $\gamma^\tau$ 的折扣终点权重 $p_o^\gamma$。后者做 backup 时不应再乘一次 $\gamma$。

### 真实数据预算与计算预算

一次真实动作和一次模型 rollout 不是同一成本。比较规划方法时应同时计入环境步数、模型调用次数、更新次数、内存和决策延迟。

<a id="lesson-setting"></a>

## 1. 用已有的后果知识改善决策

一条走廊尽头的奖励由 1 变成 10。直接 TD 必须再次走过前面的状态，才能逐步把价值变化向前传播；若已学会哪些动作通向哪里，就能在不额外消耗机器人电量的情况下，把新 reward 沿模型反向传播。规划的价值是把已有知识转成新决策。它没有创造新的环境证据：如果门已经关上但模型仍说门开着，更多想象只会更加确信一条错误路线。

| 算法线 | 模型用于什么 | 主要修改对象 |
| --- | --- | --- |
| Dyna | 生成经验式 target | 跨很多状态的 Q/价值/策略 |
| Prioritized sweeping | 寻找变化影响的前驱，优先 backup | 有限计算预算的更新顺序 |
| Option planning | 一次跨多个原始步预测后果 | 规划所用的时间单位 |
| MPC | 比较有限未来动作序列 | 当前一步的动作决定 |
| MCTS / MuZero | 在当前状态展开选择性搜索树 | 当前动作分布，及其后训练标签 |
| Dreamer 式想象学习 | 在潜在模型中产生 rollout | 可直接部署的 actor 和 critic |

先从确定、平稳的有限 MDP 开始，允许环境重置和表格存储。进入持续学习以后，模型与目标可能一起变化，规划还会通过动作选择影响后续数据。因此，除了评价模型上的计算结果，也要记录每步计算预算、知识更新时间和真实环境回报。

<a id="lesson-derive"></a>

## 2. Dyna-Q：一个循环里的三种学习

先看没有规划的 Q-learning：做一个真实动作，用实际 reward 和 next state 更新当前 Q。Dyna 保留这一步，另把真实 transition 用来更新环境模型，随后从已经见过的状态动作中抽样，调用模型产生额外 target，再用同一个 Q 更新规则学习。这三种操作分别是 direct RL、model learning、planning，不能只把 replay 叫作模型学习。

$$
\begin{aligned}y_{\rm real}&=R+\gamma(1-D)\max_aQ(S',a)\\ y_{\rm model}&=\hat R+\gamma(1-\hat D)\max_aQ(\hat S',a)\end{aligned}
$$

两种更新使用同一个价值函数，区别在后果来源。模型可以输出一个随机样本，也可以在小状态空间显式求期望；不能在随机环境仅记最后一次后果并当作精确动力学。

**算法：算法伪代码**

1. 初始化 Q=0、空 model、规划预算 n
1. 每个真实原始步：
  1. 1. 按 ε-greedy(Q(s,·)) 选择动作，打破相等价值时随机选
  1. 2. 与环境交互，得到 r,s′,true_terminal
  1. 3. 用真实 target 更新 Q(s,a)
  1. 4. model(s,a) ← 本次观察（仅确定平稳环境可直接覆盖）
  1. 5. 重复 n 次：
       1. 从已见过的 $(s_m,a_m)$ 抽样
       1. 从 model 得到 $r_m,s_m^{\prime}$,`terminal_m`
       1. 用模型 target 更新 $Q(s_m,a_m)$
  1. 6. 到 s′；只有环境真的终止且协议允许才 reset
1. 评估：冻结探索/学习开关或明确在线评估协议，按真实交互统计回报

在随机环境中，模型可以维护转移次数和奖励均值，或学习可采样的概率分布。如果只维护 next state 均值，再把非线性 max Q 作用在这个均值上，会改变 Bellman target。确定性演示里的字典覆盖只是最小可运行设计，不能不加修改地移植到随机机器人。

规划预算 $n=0$ 时退化为 Q-learning；预算增加时，价值更充分地接近当前模型的固定点，模型误差的影响也更充分。经验重放和 Dyna 都可重复训练；区别在于 Dyna 提供可查询的后果预测。若确定性模型仅存储并抽取旧转移，它在这一层面就与重放接近；概率、泛化或期望模型则提供了额外预测能力。

<a id="lesson-priority"></a>

## 3. Prioritized sweeping：为什么从后往前算

设状态 0→1→2→终点，只有最后一步奖励 1。最初所有值为零；看到最后一步后，2 的值需要改变。更新 0 暂时没有作用，因为它的后继 1 还没变化。优先更新 2，再更新能到达 2 的前驱 1，再更新 0，可以把一次新信息在三个 backup 中传到底。更新顺序由模型中的前驱依赖关系决定。

$$
P(s,a)=\left|\hat r(s,a)+\gamma(1-\hat d)\max_{a'}Q(\hat s',a')-Q(s,a)\right|
$$

这里的 P 是 priority，不是转移概率。表格确定性版本用当前 Bellman 残差作为队列优先级；小于阈值 θ 的项不进入队列。

**算法：算法伪代码**

1. 维护 model 与 predecessor(s′) = 所有可能转移到 s′ 的 (s,a)
1. 真实数据更新模型后，计算被影响 (s,a) 的残差并入最大优先队列
1. 每个规划预算步：
  1. 弹出优先级最大的 (s,a)
  1. 重新计算残差（队列中的旧优先级可能过时）
  1. 若仍显著，用当前 model 做 backup
  1. 对 predecessor(s) 中每项计算新的残差，符合阈值则入队
1. 队列空或预算耗尽即停止；保留剩余队列到下一次真实交互

随机模型中，某个后继价值的变化会按转移概率影响多个前驱，需要相应的期望 backup/优先级设计。模型改变了某个动作的终点时，前驱表也必须更新，不能只在最初建一次。本页实现为确定性已知前驱图，允许队列里有重复旧项，弹出后重新检查残差；这样避免陈旧优先级无限消耗有效 backup。

Prioritized replay 按训练样本的 TD error 调整采样机会；prioritized sweeping 还利用模型找到受影响的前驱并传播变化。两者可以结合，但使用的信息和计算过程不同，比较时需要分别记录模型知识、样本访问与 backup 预算。

<a id="lesson-options"></a>

## 4. Option planning 与收缩：粗时间尺度为什么能加速

$$
(TV)(s)=\max_{o\in\mathcal O(s)}\left[r_o(s)+\sum_jp_o^\gamma(j|s)V(j)\right]
$$

$r_o$ 是技能内真实折扣 reward，$p_o^γ$ 是 $E[γ^τ 1\{S_{t+τ}=j\}]$。有 primitive actions 时也可把它们作为 τ=1 的 options 放入同一集合；这是统一接口，不是强制用长技能替代所有短动作。

若有准确的“走到门口”模型，一次 backup 就可把门外的新价值传到门内起点，而 primitive-action backup 通常要逐步传播。模型的学习和维护当然也要花经验与算力；不能只报告规划阶段少了几步，把技能发现成本全部藏掉。

$$
\begin{aligned}\kappa&=\sup_{s,o}\sum_jp_o^\gamma(j|s)=\sup_{s,o}\mathbb E[\gamma^\tau|s,o]\leq\gamma<1\\ |(TV)(s)-(TW)(s)|&\leq\max_o\sum_jp_o^\gamma(j|s)|V(j)-W(j)|\\ \|TV-TW\|_\infty&\leq\kappa\|V-W\|_\infty\end{aligned}
$$

第一步使用 τ≥1，第二步使用两个 max 的差不大于候选值的最大差，最后把最大值差提出求和。固定精确模型、相同 option 集合下，反复 value iteration 收敛到唯一固定点。

长 options 的折扣行和可能更小；若集合里仍有一步动作，统一收缩率上界 $\kappa$ 却仍可能等于 $\gamma$。此外，技能后果的学习成本、候选数量和与任务的相关性都会影响实际速度。因此，收缩率解释了某一种传播优势，但系统效率还要计入其余成本。

$$
\begin{aligned}\|\hat V^*-V^*\|_\infty&\leq\kappa\|\hat V^*-V^*\|_\infty+\|(\hat T-T)V^*\|_\infty\\ \|\hat V^*-V^*\|_\infty&\leq\frac{\varepsilon_r+\varepsilon_p\|V^*\|_\infty}{1-\kappa}\end{aligned}
$$

这里要求学得的算子也具有不超过 κ 的收缩率；每个候选 reward 误差至多 $ε_r$，折扣终点向量的 L1 误差至多 $ε_p$。由加减 T̂V* 与三角不等式得到第一行，再移项。该界是同一有限 option 集合上的模型误差传播，不是任意神经网络规划的保证。

平均奖励目标需要另一种分析。跨技能的 differential backup 包含持续时间成本：$Q(s,o)=\mathbb E[R_{\rm sum}-g\tau+h(S_{\rm end})]$，其中 $g$ 是每个原始时间步的奖励率，$h$ 是相对价值。取 $\gamma=1$ 后，上面的折扣收缩证明不再成立，还需要参考状态、归一化或其他结构条件。按技能调用次数而非原始时间计算平均奖励，会改变优化目标。

<a id="lesson-search"></a>

## 5. MPC 与 MCTS：在行动前计算什么

Dyna 主要把计算存进长期价值/策略参数；MPC 主要为眼前动作做有限时域优化。当前状态固定，给定候选动作序列 $a_0,\ldots,a_{H-1}$，用模型 rollout 计算其收益，再找最好的序列；只执行第一步，拿到新的真实状态后重新优化，称 receding horizon。反馈来自每次重规划，而不是把整段序列不加修正地执行完。

$$
\max_{a_{0:H-1}}\ \mathbb E_{\hat P}\!\left[\sum_{k=0}^{H-1}\gamma^k\hat r(s_k,a_k)+\gamma^H\hat V(s_H)\right]
$$

H 是计划跨度，终点价值补偿截断以后未展开的后果。没有终点价值、跨度太短时，会产生短视；有终点价值但估计不准时，也会误导优化。

**算法：算法伪代码**

1. MPC：
  1. 从当前真实状态/后验开始
  1. 产生 K 个动作序列（枚举、随机采样或 CEM 等优化器）
  1. 用同一模型对每个序列 rollout；随机环境可评估多个粒子
  1. 比较累计 reward + terminal value，选最好序列
  1. 仅执行首个动作；下次用新观测重新估计状态并规划
1. CEM 近似：保留高分 elite 序列，拟合其分布，再采样和筛选若干轮

MCTS 则根据已展开节点的结果，把有限预算投向更有希望或更不确定的分支。一次 simulation 通常分为 selection、expansion/evaluation、backup 三步。Q 估计告诉我们已知收益，先验概率和访问次数给探索项。下面是常见 PUCT 型选择结构，不是所有 MCTS 的唯一定义。

$$
a=\arg\max_a\left[Q(s,a)+c\,P_{\rm prior}(a|s)\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}\right]
$$

N 是当前搜索树的访问次数，$P_{prior}$ 是网络或其他来源给出的先验。它们是搜索期的局部状态，不等于训练 replay 的采样频率。

**算法：算法伪代码**

1. MCTS：
  1. 从当前根节点沿 Q + exploration bonus 选分支
  1. 走到未展开叶子时，用模型产生下一节点，并估计 reward、prior、leaf value
  1. 沿路径反向累计 G ← r + γG，更新边的访问次数和平均价值
  1. 重复至 simulation/时间预算耗尽
  1. 根据根的访问次数分布或最大访问次数选择真实动作
1. 若训练网络：用搜索后的根分布作策略监督，用实际/估计 return 作价值监督

MuZero 把观测历史编码成根 latent，用可学习 dynamics 递推 latent 与 reward，再预测 policy/value 供搜索；不要求 decoder 还原每个像素。学到的模型仍可能被搜索利用其误差。mctx 是研究团队的 JAX 搜索库，提供 model/recurrent 接口与搜索算法；它不是完整 MuZero 训练系统，也不自动附带环境、replay 与网络训练。

<a id="lesson-tdmpc"></a>

## 6. TD-MPC2：短时域模型与长期价值分工

纯 MPC 有一个两难：视野短时看不到远处收益，视野长时模型误差累积。终点价值提供一条折中路线——只让模型预测它较可靠的短期，再用学得的价值补足剩余长期后果。TD-MPC2（ICLR 2024）把这一路线扩展到较大规模的连续控制与多任务训练，联合学习潜在状态、短期动力学、奖励和长期价值。

$$
\hat z_{k+1}=f_\theta(\hat z_k,a_k),\qquad \hat J(a_{0:H-1})=\sum_{k=0}^{H-1}\gamma^k\hat r_\theta(\hat z_k,a_k)+\gamma^H\hat V_\theta(\hat z_H)
$$

这是短期预测加终点价值的规划分工。模型在潜在状态中递推，不需要重建原始观测的每个像素；实际 TD-MPC2 使用其规定的价值估计、采样优化和策略先验。这里写成 V 便于与本章 MPC 公式对应。

训练时，模型预测的下一潜在状态与真实下一观测的编码对齐，同时拟合奖励和 TD 价值目标。行动时，在潜在模型中比较候选动作序列；策略先验帮助提出较好的候选，再执行选定序列的首动作。长期信息同时进入终点价值和候选分布，因而无需把所有远期细节都展开成很长的模型轨迹。

这与 Dreamer 的主要差别不是“有没有 world model”，而是决策计算放在哪里：TD-MPC2 在行动时继续优化动作序列，Dreamer 主要将想象训练所得的能力存入 actor。多任务训练覆盖广也不等于已经解决持续适应；若用于 CRL，需进一步考察旧数据保留、任务切换、模型漂移及在线规划延迟。作者 tdmpc2 仓库可沿模型损失、计划函数、策略先验三个位置阅读。

<a id="lesson-alps"></a>

## 7. ALPS：用谱结构确定远路，用短 MPC 走近路

现在考虑一座弯曲的迷宫：起点和目标在坐标上很近，却隔着一堵长墙。以原始位置距离作为 MPC 成本，短视控制会一直靠近墙；简单增大预测时域，又会遇到模型误差累积。ALPS（Laplacian Representations for Decision-Time Planning，ICML 2026）利用环境连通结构定义距离并产生中间目标，再分别处理长程路线和局部控制。

这个方法采用离线目标条件设定：已有状态—动作轨迹，先从数据学表示、一步前向模型和行为先验，再在给定目标下做决策时规划。这里的 Laplacian 表示不是一个生成未来观测的世界模型；它用于衡量距离和划分区域。前向动力学则在原始状态空间训练，两种对象承担不同任务。

$$
\psi_i(s)=\frac{\phi_i(s)}{\sqrt{\lambda_i}},\qquad d_k^2(s,g)=\sum_{i=1}^{k}\frac{(\phi_i(s)-\phi_i(g))^2}{\lambda_i}
$$

ALLO 学到非平凡特征 $φ_i$ 及其特征值 $λ_i$；按特征值缩放以后，低频的长程结构获得相应权重。这里用 ψ 表示规划坐标，与 successor features 中同名符号的定义不同。

缩放并非仅为好看。对连通无向图，采用组合 Laplacian 的完整非零特征系，可将随机游走的往返时间写成谱距离；这给出了为什么结构距离能反映绕墙难度的依据。实际神经估计、特征截断和数据覆盖会引入误差，有向不可逆控制也不自动满足同样的精确等式。

$$
C(s,g)=\operatorname{vol}(G)\sum_{i=1}^{|\mathcal S|-1}\frac{(u_i(s)-u_i(g))^2}{\lambda_i}
$$

C 是随机游走从 s 到 g 再返回的期望步数，vol(G) 是图的总度。它不是最优策略的单向最短到达时间。换用归一化 Laplacian 时，还需按该定义调整坐标；实际使用少数方向时只保留近似。

接下来，在缩放表示空间对经验状态聚类，把相互容易到达的区域作为图节点，用数据里的跨区域转移连接这些节点。高层用 Dijkstra 选择通向目标区域的路线和下一个中间目标；低层用短时域 CEM，在一步前向模型中比较动作序列，使预测状态接近这个中间目标。行为先验引导候选动作，减轻离线数据覆盖不足时任意搜索的风险。

**算法：ALPS 将长期路径选择与短期模型控制分开；区域图和动力学模型不是同一个对象**

1. 离线准备：
  1. 用轨迹学习 ALLO 特征和特征值，得到缩放坐标 $\psi$
  1. 在原始状态空间学习一步模型；学习动作行为先验
  1. 在 $\psi$ 空间聚类，由数据转移构建区域图
1. 每个真实决策：
  1. 找到当前位置与目标所在区域
  1. 高层在区域图中求路线，选下一中间目标
  1. 低层用行为先验引导 CEM，做短时域模型预测
  1. 按到中间目标的结构距离比较候选序列
  1. 执行首动作，用新观测重新定位并规划

把它与 eigenoptions 对照，就能看清 Machado 这条研究线的延伸。Eigenoptions 把谱方向转成可执行的闭环策略；ALPS 把谱坐标用作规划距离与区域划分，不必先为每个方向训练一个 option。两者共用“从转移中发现长程结构”的出发点，但把结构交给了不同的控制机制。

它与 STOMP 的联系也不是替代关系。STOMP 先构造奖励相关子任务，学出 options，再学习技能后果模型做 backup；ALPS 以结构子目标组织决策时短期搜索。前者要维护技能策略和长程模型，后者要维护区域连接、局部模型和在线搜索预算。门突然关闭时，这两种系统需要修正的对象不同。

论文在 OGBench 的部分离线目标任务上比较了规划与模型无关基线。作为起步实验，可保持同一个局部模型，只依次移除特征值缩放、高层区域图或行为先验，观察失败分别来自距离失真、长程路线缺失还是动作搜索离开数据支持。若研究持续 RL，再增加通道关闭与新区域开放，分别测图、表示和模型更新的速度；静态离线结果本身不回答这些适应问题。

<a id="lesson-imagination"></a>

## 8. Dreamer 式想象：把规划的成果存进 actor

想象学习不一定每次决策都搜索一棵树。先用真实序列拟合 world model，从真实经验后验得到起点 latent；在模型中按 actor 产生多个动作和下一 latent；用预测 reward、继续概率和 critic 构造 return；用这些目标训练 actor/critic。部署时 actor 的一次前向计算就可输出动作。它更接近把许多模型计算摊销进参数，区别于每次行动重新做大规模搜索。

$$
\begin{aligned}G_k^\lambda&=\hat r_{k+1}+\gamma\hat c_{k+1}\left[(1-\lambda)V(\hat z_{k+1})+\lambda G_{k+1}^\lambda\right]\\ G_H^\lambda&=V(\hat z_H)\end{aligned}
$$

这里用一个通用 imagined λ-return 说明信用分配：ĉ 预测下一步是否继续，λ 决定依赖多长的模型 rollout；最后以 critic 截断。λ=0 是一步 target，λ=1 是到 H 的模型多步 return。不同 Dreamer 版本在 return 归一化、分布输出、梯度路径与损失上有具体区别。

critic 拟合停止梯度的 return；actor 可通过可微模型的路径梯度、likelihood-ratio 梯度或二者组合获得训练信号。模型参数是否冻结、潜在状态路径是否允许反传、优势是否停止梯度，共同规定了具体估计器。模型预测误差、梯度估计偏差与真实回报改善需要分别分析。

**算法：算法伪代码**

1. 通用 imagined actor-critic：
  1. 真实 replay → 编码后验状态 → 更新 world model
  1. 固定本轮模型参数，从真实后验状态启动 H 步 actor rollout
  1. 预测每步 reward 和 continue；从尾到头计算 λ-return
  1. critic 拟合停止梯度的 return
  1. actor 用规定的梯度路径优化 return/advantage，并加入明确的熵项
  1. 返回真实交互收集新数据；持续检查模型误差和分布覆盖

如果一步误差在 rollout 中逐步累积，增大 $H$ 可能让 actor 更偏爱模型预测过于乐观的行为。比较不同想象长度时，应同时测量真实验证误差、预测回报和真实回报。DreamerV3 中的分布损失、尺度处理和数据—更新比例也影响这一过程，阅读作者实现时需连同 imagined actor 的梯度路径一起看。

<a id="lesson-example"></a>

## 9. 三个可以手工复算的规划过程

优先传播：已知 $0\to1\to2\to$ 终点，只有最后一步奖励为 $1$，$\gamma=0.9$，初值全为零。队列从最后一条边开始，依次把三个状态的值更新为 $1$、$0.9$、$0.81$；终点保持零。若最后奖励改为 $2$，变化仍可从最后一条边沿前驱传播，无需再真实走完整条链。

Option value iteration：只有一个抽象状态和一个 option，内部折扣奖励为 $2.8$，返回该状态的折扣权重为 $0.81$。更新 $v_{k+1}=2.8+0.81v_k$；从 $v_0=0$ 开始，依次为 $2.8,5.068,6.90508$，极限 $v^*=2.8/(1-0.81)=14.736842\ldots$。若模型奖励误为 $2.9$，极限误差为 $0.1/(1-0.81)=0.526316\ldots$，增加规划次数不能消除此偏差。

MPC：一维位置从 $0$ 去目标 $2$，动作取 $-1,0,+1$，模型 $x'=x+a$，时域 $H=3$。使用有限时域无折扣得分 $-(x'-2)^2-0.01a^2$，枚举 $27$ 条动作序列，最优为 $(1,1,0)$，总分 $-1.02$。实际只执行首个 $+1$；若出现执行偏移，下次从新观测位置重新规划。

<a id="lesson-code"></a>

## 10. 实现与实验：Dyna、优先传播与 MPC

包含确定性 Dyna-Q 的完整交互循环、优先队列、option value iteration 与有限时域 MPC。

```python
def q_learning_update(q, s, a, reward, sn, terminal, alpha=1.0, gamma=0.9):
    target = reward + (0.0 if terminal else gamma * max(q[sn]))
    delta = target - q[s][a]
    q[s][a] += alpha * delta
    return delta


def dyna_q(seed=4, planning_steps=5, episodes=40):
    """Deterministic 5-state line, reward 1 on entry to terminal state 4.

    This small episodic mechanism test permits reset. It is not single-life CRL.
    Model key=(s,a), value=(reward,next_state,terminal); last observation is
    correct here only because the environment is deterministic and stationary.
    """
    rng = random.Random(seed)
    q = [[0.0, 0.0] for _ in range(5)]
    model = {}
    lengths = []
    for _ in range(episodes):
        s = 0
        for length in range(1, 1001):
            if rng.random() < 0.2:
                a = rng.randrange(2)
            else:
                best = max(q[s])
                a = rng.choice([i for i, val in enumerate(q[s]) if val == best])
            sn = max(0, min(4, s + (-1 if a == 0 else 1)))
            terminal = sn == 4
            reward = float(terminal)
            q_learning_update(q, s, a, reward, sn, terminal)
            model[s, a] = (reward, sn, terminal)
            for _ in range(planning_steps):
                ms, ma = rng.choice(list(model))
                mr, msn, mt = model[ms, ma]
                q_learning_update(q, ms, ma, mr, msn, mt)
            s = sn
            if terminal:
                lengths.append(length)
                break
        else:
            raise RuntimeError("Episode exceeded safety cap")
    return q, model, lengths


def prioritized_sweeping(q, model, seeds, max_backups=100, threshold=1e-10, gamma=0.9):
    """Deterministic tabular predecessor graph and lazy priority queue.

    Recompute residual when popped; stale entries are allowed but do not
    trigger a backup after their residual has fallen below threshold.
    """
    predecessors = {s: set() for s in range(len(q))}
    for (s, a), (_, sn, _) in model.items():
        predecessors[sn].add((s, a))
    heap = []
    counter = itertools.count()

    def enqueue(key):
        s, a = key
        r, sn, terminal = model[key]
        target = r + (0.0 if terminal else gamma * max(q[sn]))
        residual = abs(target - q[s][a])
        if residual > threshold:
            heapq.heappush(heap, (-residual, next(counter), key))

    for key in seeds:
        enqueue(key)
    updates = []
    while heap and len(updates) < max_backups:
        _, _, (s, a) = heapq.heappop(heap)
        r, sn, terminal = model[s, a]
        target = r + (0.0 if terminal else gamma * max(q[sn]))
        if abs(target - q[s][a]) <= threshold:
            continue
        q_learning_update(q, s, a, r, sn, terminal, gamma=gamma)
        updates.append((s, a, q[s][a]))
        for predecessor in predecessors[s]:
            enqueue(predecessor)
    return updates


def option_value_iteration(models, values=None, tolerance=1e-12, cap=10000):
    """models[s] contains (r_o, discounted endpoint weights) for legal options."""
    values = [0.0] * len(models) if values is None else values[:]
    for iteration in range(1, cap + 1):
        new_values = [max(model_backup(r, p, values) for r, p in actions)
                      for actions in models]
        if max(abs(a - b) for a, b in zip(values, new_values)) < tolerance:
            return new_values, iteration
        values = new_values
    raise RuntimeError("No convergence within cap; inspect model row masses")


def finite_horizon_mpc(position, goal, horizon=3):
    """Enumerate actions in a known deterministic model, execute first only."""
    best_score, best_sequence = -math.inf, None
    for sequence in itertools.product((-1, 0, 1), repeat=horizon):
        x, score = position, 0.0
        for action in sequence:
            x += action
            score -= (x - goal)**2 + 0.01 * action**2
        if score > best_score:
            best_score, best_sequence = score, sequence
    return best_sequence[0], best_sequence, best_score
```

Python 3.10+；不需要环境库或 GPU

```sh
python3 knowledge_algorithms_lab.py planning
python3 knowledge_algorithms_lab.py test
```

Dyna 的五状态小链会学到贪心状态值 $[0.729,0.81,0.9,1,0]$。训练仍含 $\varepsilon$ 探索，因此最后一条轨迹的长度会有波动。优先传播依次更新后三条边，所得值为 $1$、$0.9$、$0.81$；option 迭代接近 $14.736842$；MPC 的首动作是 $+1$。这些中间量比只看一次 episode 的总回报更容易定位实现问题。

- 实验 A：比较 planning_steps=0、1、5、20，在多个固定种子下同时记录真实交互数与总 backup 数；是否仍可称为“更省成本”？
- 实验 B：走廊中途关闭一条旧通路，比较“只覆盖模型”“近期数据模型”“带探索奖励/陈旧度探索”三种恢复策略；仅增加 planning 预算会不会更糟？
- 实验 C：优先队列预算设为 1，逐次观察价值变化走到哪一层；解释为什么不是所有状态同时变化。
- 实验 D：给 MPC 的模型增加有方向的偏差，比较不同 H 与重规划频率；更长的搜索是否总是更好？

从表格实现转向深度规划时，mctx 的 root/recurrent 接口对应 reward、discount、prior 与 value；DreamerV3 对应真实数据模型更新、想象数据和 actor 更新；ALPS 对应谱表示、区域图与短 MPC。按这些数据流阅读，可以识别算法究竟把一次额外计算用于什么。

<a id="lesson-branches"></a>

## 11. 持续学习中怎样分配规划计算

| 实际问题 | 合适的起点 | 关键失败模式 |
| --- | --- | --- |
| 数据贵，模型小而可靠 | Dyna / prioritized sweeping | 用旧模型重复强化错误 |
| 有可复用的闭环技能 | option model backups | 技能改变但模型未更新 |
| 当前决策可花较多计算 | MPC / MCTS | 分布外搜索、决策延迟、有限视野 |
| 部署动作必须很快 | 模型想象训练 actor | 模型误差被长期固化进策略 |
| 奖励变化而动力学稳定 | 模型重估价值 / SF-GPI | 忘记检查奖励表达与策略覆盖条件 |
| 环境长期变化且无任务边界 | 近期校准 + 预算化规划 + 主动验证 | 模型、表示、技能同时漂移无法归因 |

非平稳 Dyna 的经典动机是：模型未报告变化，不代表世界未变化。可用距离上次尝试的时间等信号鼓励真实验证被长期忽略的动作，例如 Dyna-Q+ 类方法在模型规划 reward 中增加与未尝试时长相关的 bonus。但该 bonus 是探索设计，不是新的真实 reward；评估仍应采用环境原 reward，并把额外探索成本计入。

研究上更有辨识力的问题是：“每步有限 B 次计算，应优先验证哪个模型、更新哪个技能模型、还是改进哪个价值？”这连接了变化检测、价值相关模型误差、元学习计算分配与长期知识维护。把所有预算都放进更大模型，并不能自动解决这一调度问题。

<a id="research-planning-allocation-and-error"></a>

## 研究专题 A · 规划预算应分给可靠且会改变决策的查询

TD-MPC2 的短模型加终点价值、DINO-WM 的视觉目标搜索、Dreamer 的 imagined actor 学习，都把预测变成决策，但将计算放在不同位置。CRL 还要问：新经验改变了哪项知识，有限计算应重算哪些决策？扩大 horizon、增加候选序列和增加梯度更新，分别消耗不同资源，不能只写成统一的“更多规划”。

$$
\hat J(a_{0:H-1})=\sum_{k=0}^{H-1}\gamma^k\hat r(\hat z_k,a_k)+\gamma^H\hat V(\hat z_H),\qquad \hat z_{k+1}=f(\hat z_k,a_k)
$$

短期模型承担 H 步后果，critic 承担剩余长期价值。改变 H 会同时改变模型误差、价值误差和计算成本；用 V 书写为通用接口，具体 TD-MPC2 的价值估计和策略先验须照原算法。

$$
|\hat J-J|\leq\sum_{k=0}^{H-1}\gamma^k\varepsilon_{r,k}+\gamma^H\varepsilon_{V,H}
$$

这是对于同一固定候选序列，在各步奖励贡献与终点价值贡献已有相应误差界时的直接三角不等式。ε 包括状态预测造成的偏差，不是仅在真实状态上测到的 one-step loss；候选经优化后走到数据外，原先的界也可能不适用。

手算：两个候选的真实收益为 1.0 与 0.9，如果每个候选估计误差不超过 0.02，排序可靠；若误差上界为 0.1，搜索器可能稳定地选择较差动作。选择更多候选还可能发现更多能利用模型误差的轨迹。只有预测总体平均误差低，没有 planner 实际查询上的校准，不能断言增加搜索一定有益。

| 预算变量 | 改善的潜在瓶颈 | 必须同时观察 |
| --- | --- | --- |
| 模型 horizon H | 终点价值短视 | rollout 偏差与远期 critic 误差 |
| 候选数 / 优化轮次 | 动作序列搜索不足 | 查询是否离开数据支持，决策延迟 |
| 想象 actor 更新 | 部署策略尚未吸收模型知识 | 模型偏差写入参数后能否由真实数据纠正 |
| option model backup | 原始步传播过慢 | 技能版本、真实时长与维护成本 |
| 近期模型再训练 | 变化后旧模型过期 | 旧区域知识是否被不必要地忘掉 |

**算法：测试规划计算的收益，也计入计算对交互频率的影响**

1. 计算匹配的规划实验（拟议）：
  1. 给每个环境步固定总毫秒/模型调用预算
  1. 对同一状态保存真实执行后的结果，记录所选候选的预测偏差
  1. 分别改变 H、候选数和 actor 更新数；其余预算匹配
  1. 在隐藏动力学变化后统计首批错误、恢复时间及全程收益
  1. 记录动作等待造成的真实时间损失；模拟次数不能算成环境证据

研究空缺是由真实后续数据学习预算分配，而不是始终固定一个大搜索。预测不确定性只是一种候选信号；它须与动作排序敏感性、模型更新时间和实际延迟共同验证。该分配方案属于本教材的研究提案，不是 TD-MPC2 或视觉 world-model 原文已证明的持续学习机制。

<a id="research-planning-objective-sensitive-contract"></a>

## 研究专题 B · 更换目标后，哪些规划知识仍可复用

将“换任务”视为一个统一事件会掩盖接口差异。SF/GPI 在共享动力学与线性奖励族中复用未来特征；VE 模型只保持特定策略/价值的 backup；视觉目标模型按潜在距离搜索；风险敏感规划还要求对应回报分布。规划复用的前提，必须从新目标会查询什么反推。

| 变化 | 可能直接复用 | 先失效的对象 |
| --- | --- | --- |
| 只换线性奖励权重 | 固定策略 SF，及准确的奖励特征累计 | 若新奖励不在张成空间，读出不够 |
| 只换后续价值函数 | 相同技能的 reward/endpoint model | 仅对旧 critic 等价的压缩模型可能不足 |
| 从期望改为尾部风险 | 保留相应分布/摘要的模型 | 均值价值等价模型 |
| 更换视觉目标 | 稳定 encoder 与覆盖目标的动作模型 | 潜在距离未反映任务或目标不在支持内 |
| 动力学变化 | 可保留未受影响区域/技能知识 | 旧 SF、局部模型和区域可达图 |
| 技能策略/停止变化 | 环境的一步动力学模型可能可复用 | 旧 option 后果与时长模型 |

$$
Q_o(s;V,g)=\mathbb E\!\left[R_{\rm sum}-g\tau+V(S_{\rm end})\mid s,o\right]
$$

平均奖励的技能选择以原始时间的机会成本 gτ 计价。这与折扣 terminal weight 接口不同；研究新的风险目标还须指定对整个随机回报怎样取风险函数，不能先对每一部分随意取 CVaR 再相加。

例子：技能 A 保证 10 步到目标，技能 B 平均 8 步但偶尔耗时 100 步。平均时间目标与 deadline 失败概率可以偏好不同技能，即使二者终点相同。只学“成功率”或“平均持续时间”无法同时回答所有问题；planner 应先给出查询，再选择需要维护的后果统计。

实验设计以同一批冻结技能为起点，分别改变奖励权重、deadline、风险度量和动力学，比较重用、局部重学与全部重学。模型查询误差与实际收益应成对记录；奖励改变适应快，并不能替代动力学变化恢复快的证据。Distributional Model Equivalence 提供风险目标的理论反例，持续技能后果的尾部维护则仍需未来数据和校准实验。

<a id="lesson-check"></a>

## 12. 诊断与自测答案

- 规划步越多越差：固定 policy 检查模型校准，再看 planner 查询是否远离训练分布；区分模型误差与 critic 优化问题。
- 终点奖励不能向前传播：检查 predecessor 方向、队列 threshold、是否把真实 terminal 后的 bootstrap 清零但保留进入 terminal 的 reward。
- 长 option 价值异常：检查模型内已折扣的 endpoint 是否被再次乘 γ，或被错误归一化。
- 模拟回报高、真实回报低：区分模型目标误差、规划优化器误差和执行反馈误差，三者要分别做对照。

自测 1：MCTS 是否必须先学模型？答：棋类可以直接使用规则模拟器。自测 2：Dyna 的 $n\to\infty$ 是否逼近真实最优策略？答：它至多充分求解当前模型，覆盖和模型误差仍存在。自测 3：增加 option 是否保证更好？答：保留原动作且模型精确时，扩大候选集不会降低理论最优值；有限计算、模型误差与学习成本仍可能降低实际表现。自测 4：平均奖励为何不能沿用 $1/(1-\gamma)$ 的误差界？答：$\gamma=1$ 时折扣收缩论证失效，需要相对价值和其他条件。

## 本章的实验设计

报告模型调用次数与决策耗时。用准确模型、学得模型和无规划对照区分误差来源。

设定：先在准确小模型中验证搜索可以改变动作，再用同一模型快照扫描深度和预算。最后接入在线模型。

- 一次搜索的模型版本一致，或有明确并发语义。
- 超时有预定动作而不是事后无限延长。
- 有限搜索的准确模型对照不自动称为最优上界。

对照：无规划、准确模型规划与学习模型规划；同数据和同部署时间分别比较；固定/在线模型×有/无规划

记录：真实回报、模型调用和搜索节点；深度、候选数、价值查询与延迟；计划预测与真实执行差值

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-planning)

## 学习与研究衔接

训练时想象与决策时搜索不是同一算法接口。比较时要同时限定模型调用和环境交互。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-planning) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=planning) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=planning)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

Laplacian 描述行为图结构，奖励感知表示加入路径价值，METRA 学习有区别的行为，HIQL 利用离线目标轨迹，MaestroMotif 引入语言先验。它们承担不同的设计工作；生成技能、选择技能与组合技能需要分别评价。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [Laplacian Keyboard: Beyond the Linear Span](https://yingwen.io/zh/continual-rl/research/#recent-laplacian-keyboard)
- [Foundation Policies with Hilbert Representations](https://yingwen.io/zh/continual-rl/research/#recent-hilbert-foundation-policies)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

模型可提取性的理论说明某类能力需要什么知识，不指定唯一网络。Dreamer 研究潜在想象控制，STOMP 研究随机时长行为模型，DRAGO 研究旧模型知识保留。模型误差、查询策略和规划收益之间仍需实验连接。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [Mastering diverse control tasks through world models](https://yingwen.io/zh/continual-rl/research/#recent-dreamerv3-world-models)
- [Laplacian Keyboard: Beyond the Linear Span](https://yingwen.io/zh/continual-rl/research/#recent-laplacian-keyboard)
- [The Value Equivalence Principle for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-value-equivalence-models)
- [Distributional Model Equivalence for Risk-Sensitive Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-distributional-model-equivalence)
- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://yingwen.io/zh/continual-rl/research/#recent-tdmpc2-decision-time-model)
- [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://yingwen.io/zh/continual-rl/research/#recent-dino-wm-feature-planning)
- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://yingwen.io/zh/continual-rl/research/#recent-vjepa2-action-conditioned)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文提供定义和条件，架构讲座提出模块组织，算法论文检验特定机制。完整系统还要明确智能体、外部设计者和世界各自承担的工作；组件成立不自动意味着组合后的长期收益成立。

- [The OaK Architecture: A Vision of SuperIntelligence from Experience](https://yingwen.io/zh/continual-rl/research/#recent-oak-architecture)
- [The Value Equivalence Principle for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-value-equivalence-models)

### Reward-Respecting Subtasks for Model-Based Reinforcement Learning

Richard S. Sutton, Marlos C. Machado, G. Zacharias Holland, David Szepesvari, Finbarr Timbers, Brian Tanner, Adam White

Artificial Intelligence · 2023 · 支持方法与理论

#### 研究问题

学到一个能到达子目标的技能之后，为什么它仍可能不适合主任务规划？

#### 关键机制

STOMP 把子任务、option、模型和规划连起来。子任务保留原任务的路径奖励，并用带有特征偏好的终止价值表达目标；学习得到策略和终止规则后，再预测该行为的累计奖励与折扣终点。这样，技能不会因为只追求到达子目标而忽略途中代价。

#### 证据

论文用明确的小问题展示奖励感知子任务如何产生更有用的行为和规划模型。它提供的是可分析的构造链，而非只比较一个技能执行成功率。

#### 条件与限制

终止收益的约定是子任务定义的一部分，不能随意换成固定终点奖励。特征和子任务候选的选择尚不等于完整自主发现机制；实验也不构成整个 OaK 架构的验证。

#### 阅读与实验

在同一个绕路环境中比较“最短到达目标”和“保留路径奖励”的子任务。分别计算 option 的奖励模型、折扣终点模型与一次规划备份。

#### 原文与相关入口

- [期刊论文](https://doi.org/10.1016/j.artint.2023.104001)：STOMP 与奖励感知子任务的正式论文。
- [作者预印本](https://arxiv.org/abs/2202.03466)：最初预印本早于期刊年份；阅读停止收益的精确定义。

### Laplacian Keyboard: Beyond the Linear Span

Siddarth Chandrasekar, Marlos C. Machado

arXiv 预印本 · 2026 · 支持方法与理论

#### 研究问题

从一组谱技能出发，能否解决超出原特征线性奖励空间的新任务？

#### 关键机制

Laplacian 特征先定义行为基，并借助后继特征预测各行为的后果。固定任务权重的价值组合受特征张成空间限制；论文进一步使用随状态变化的元策略，在不同位置组合已有行为。关键变化是组合规则从一组全局固定权重变为状态相关的行为选择。

#### 证据

论文对行为基与任务组合给出理论分析，并报告有限环境中的组合实验。它延续 eigenoptions 与 successor features 的路线，同时解释了为什么单纯线性读出会遇到表达边界。

#### 条件与限制

理论结论依赖具体的行为基、近似误差和任务条件。技能集合的长期生成、淘汰与非平稳模型维护仍是另外的问题；此处按预印本收录，不指定未经确认的会议。

#### 阅读与实验

构造一个必须在中途切换方向的奖励任务。分别比较固定权重的技能选择与状态相关切换，并解释性能差异来自哪里。

#### 原文与相关入口

- [作者预印本](https://arxiv.org/abs/2602.07730)：阅读线性张成空间的限制及状态相关组合机制。

### Mastering diverse control tasks through world models

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap

Nature · 2025 · 支持方法与理论

#### 研究问题

同一套世界模型训练与控制方法，能否减少跨任务重新设计损失和超参数的需求？

#### 关键机制

DreamerV3 从经验学习递归潜在状态、奖励和延续预测，再在潜在想象轨迹上学习 actor 和 critic。尺度稳健的表示与损失设计使同一配置可以适用于多种任务。模型是用于决策的学习接口，不必生成完整真实世界。

#### 证据

论文在大量视觉和状态控制任务上报告了广泛表现。关键含义是共享算法配置；这些结果主要来自分别训练的任务智能体，不是一个智能体按顺序学会全部任务。

#### 条件与限制

经验重放、批量训练和模型想象都有资源成本。模型偏差、表示遗忘与长期任务切换仍需要专门实验，不能由多任务覆盖范围自动推出持续学习能力。

#### 阅读与实验

把状态更新、模型训练、想象起点和策略更新四种分布分别写清。比较真实交互步数之外，还应记录想象步数和优化次数。

#### 原文与相关入口

- [Nature 原文](https://www.nature.com/articles/s41586-025-08744-2)：方法和任务协议；区分共享配置与单智能体持续学习。
- [作者维护的实现](https://github.com/danijar/dreamerv3)：公开实现的版本与论文实验环境应分别记录。

#### 作者代码

[作者发布的重实现；不把当前分支当作原论文实验的冻结快照。](https://github.com/danijar/dreamerv3)

DreamerV3 的作者维护公开实现及运行配置。

### The OaK Architecture: A Vision of SuperIntelligence from Experience

Richard S. Sutton

RLC 2025 讲座 / Oak Lab · 2025 · 定义与架构观点

#### 研究问题

持续学习是否只是在一个现成 actor–critic 上加入抗遗忘机制，还是需要重新安排知识构造与使用？

#### 关键机制

OaK 提出从经验持续形成状态、预测知识、子任务、时间抽象与模型，并让这些知识服务规划的架构方向。这里的重点是模块之间怎样产生可复用知识，而不只是保留某个固定策略网络的参数。

#### 证据

官方页面提供 Richard Sutton 的架构讲座与相关研究入口。STOMP、预测学习和在线特征学习等论文可以检验其中具体组件，但不能自动验证整体架构。

#### 条件与限制

这是研究愿景与架构讲解，不是一套已公布完整训练配方、统一基准结果和可复现端到端代码的系统。资源分配、问题生成、知识替换与模块相互干扰仍需明确算法。

#### 阅读与实验

为每个模块写出输入、输出、更新频率和资源上限。再选择一个双模块接口做可证伪实验，例如技能模型改善是否真的减少规划误差。

#### 原文与相关入口

- [Oak Lab 官方讲座页面](https://oaklab.ai/posts/the-oak-architecture)：讲座入口与架构研究方向。
- [Oak Lab 研究主页](https://oaklab.ai/)：区分已发表研究、技术文章和仍在预告中的项目。

### Foundation Policies with Hilbert Representations

Seohong Park, Tobias Kreiman, Sergey Levine

ICML 2024 · 2024 · 支持方法与理论

#### 研究问题

如何从无任务标签的离线轨迹形成既能按方向调用、又能用于目标任务的策略接口？

#### 关键机制

HILP 先学习近似保存时间距离的 Hilbert 表示，再以潜在位移与方向的内积训练方向条件策略。新任务通过奖励回归、目标方向或分层调用选择策略条件，结构表示也支持测试时规划。

#### 证据

ICML 原文与作者项目包含零样本 RL、离线目标条件 RL 及规划实验；官方仓库将 zero-shot 与 goal-conditioned 两套实现分开。

#### 条件与限制

精确时间距离不总能无损嵌入有限维对称欧氏距离，尤其有向不可逆行为；理论充分条件与近似神经实验需区分。方向条件策略没有自动获得任意停止条件或完整技能后果模型。

#### 阅读与实验

固定离线数据分别测距离误差、方向执行误差、奖励可表达误差与高层收益。让同一视觉观测对应不同历史，检查仅观测编码是否足够，之后再讨论 CRL 状态维护。

#### 原文与相关入口

- [ICML 2024 原文](https://proceedings.mlr.press/v235/park24g.html)：Hilbert 距离、策略提示和定理前提；不是 ICLR 论文。
- [作者项目与公式](https://seohong.me/projects/hilp/)：时间距离与方向奖励接口。
- [官方实现](https://github.com/seohongpark/HILP)：hilp_zsrl 与 hilp_gcrl 对应不同实验。

#### 作者代码

[作者项目直接链接并标为 official implementation。](https://github.com/seohongpark/HILP)

离线预训练、零样本奖励适配及目标条件实验。

### The Value Equivalence Principle for Model-Based Reinforcement Learning

Christopher Grimm, André Barreto, Satinder Singh, David Silver

NeurIPS 2020 · 2020 · 支持方法与理论

#### 研究问题

模型容量有限时，必须预测全部状态细节，还是只须保持规划会查询的量？

#### 关键机制

Value equivalence 以策略集合和函数集合定义模型规格：模型对这些函数进行这些策略的 Bellman backup，应与真实环境相同。扩大查询族会缩小可接受模型族；它把“决策相关”从口号变成有条件的等价关系。

#### 证据

论文给出等价模型类的性质及有限实验，并解释若干隐式模型方法。后续 Proper Value Equivalence（NeurIPS 2021）研究策略价值固定点等价及规划充分性。

#### 条件与限制

少数当前 critic 的 backup 相同，不说明所有新奖励、新策略或风险目标都相同。精确算子等价与神经损失在样本上较小不同；奖励或查询族变化后须重新验证。

#### 阅读与实验

保存独立的 planner 查询集，直接测 target 误差和动作排序。用未参与模型拟合的价值函数检验迁移，并与像素误差对照，找出模型实际保留的信息。

#### 原文与相关入口

- [NeurIPS 2020 原文](https://papers.nips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)：VE 依赖策略与函数集合。
- [Proper Value Equivalence · NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/400e5e6a7ce0c754f281525fae75a873-Abstract.html)：多步算子、固定点与规划充分性；不是任意潜在网络的保证。

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

### TD-MPC2: Scalable, Robust World Models for Continuous Control

Nicklas Hansen, Hao Su, Xiaolong Wang

ICLR 2024 · 2024 · 支持方法与理论

#### 研究问题

如何让短期动力学与长期价值分工，并在动作选择时继续使用模型？

#### 关键机制

TD-MPC2 学习无需观测 decoder 的潜在动力学、奖励、价值与策略先验。决策时优化有限动作序列，用终点价值补上未展开的后果；执行第一步后，利用新观测重新规划。

#### 证据

正式会议原文报告 104 个在线任务和单一大型多任务智能体的实验。官方仓库包含模型训练与计划接口，适合与 Dreamer 的想象 actor 学习比较计算位置。

#### 条件与限制

跨任务共享超参数和多任务能力不是单条生命流中持续适应的证据。replay、任务条件、模型更新、决策延迟等成本需进入 CRL 协议；长程 critic 错误不能被短期模型精度自动修复。

#### 阅读与实验

固定模型，对比无终点价值、不同 horizon 和不同规划预算；再固定预算比较部署 actor 与决策时搜索。环境变化后同时记录模型校准、critic 误差和恢复收益。

#### 原文与相关入口

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html)：短期预测、终点价值、多任务协议。
- [作者实现](https://github.com/nicklashansen/tdmpc2)：训练、模型与 plan 函数分别阅读。

#### 作者代码

[作者维护的原论文代码。](https://github.com/nicklashansen/tdmpc2)

TD-MPC2 的单任务/多任务训练和决策时规划。

### DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning

Gaoyue Zhou, Hengkai Pan, Yann LeCun, Lerrel Pinto

ICML 2025 · 2025 · 支持方法与理论

#### 研究问题

预训练视觉表示能否直接成为动作后果预测与目标规划的接口？

#### 关键机制

冻结 DINOv2 空间 patch 特征，用离线动作轨迹学习未来特征预测器；测试时优化动作序列，让预测特征接近目标图像特征。没有重建图像、奖励模型或逆模型，不表示没有动作条件的动力学训练。

#### 证据

ICML 原文在六类环境检验视觉目标规划，作者仓库公开数据、部分检查点、训练与 CEM 规划入口。零样本指给定已训练模型后解决目标，无额外任务策略训练。

#### 条件与限制

依赖视觉预训练与离线交互覆盖；patch 相近不总等于任务完成或风险相同。原实验不证明冻结视觉表示能适应长期新物体、新动作语义或隐藏状态。

#### 阅读与实验

分别改变背景、物体属性、控制动力学与目标分布。把冻结 encoder 和联合更新 encoder 分开，对照视觉距离、真实成功与模型误差，观察表示漂移的依赖成本。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/zhou25t.html)：正式发表入口；早期 ICLR 投稿页不能替代此状态。
- [作者项目代码](https://github.com/gaoyuezhou/dino_wm)：train.py、plan.py、数据与已公开模型检查点范围。

#### 作者代码

[原作者 Gaoyue Zhou 的论文配套仓库。](https://github.com/gaoyuezhou/dino_wm)

DINO 特征预测、离线环境数据与目标规划；README 公开部分环境检查点。

### V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, Nicolas Ballas

arXiv 预印本（此处采用 2025 首稿） · 2025 · 支持方法与理论

#### 研究问题

无动作标注的视频预训练，与能接受机器人动作的规划模型之间还缺哪一步？

#### 关键机制

V-JEPA 2 先学被遮蔽视频的潜在特征预测；V-JEPA 2-AC 冻结编码器，再用机器人轨迹训练动作条件预测器。控制以目标图像的特征差为代价进行 MPC；视频理解、动作条件预测和真实控制是三个独立证据层。

#### 证据

2025 首稿报告以大规模视频预训练，再用不到 62 小时 DROID 交互视频后训练，在两个实验室以图像目标做真实机器人规划。论文单独讨论相机位置、长程规划与图像目标的局限。

#### 条件与限制

无任务奖励并不等于无动作、无机器人状态或无外部数据。零样本部署未持续更新模型，也未发现和维护 options。官方仓库现含 V-JEPA 2.1，复现首稿须记录配置和模型版本。

#### 阅读与实验

按视觉编码、动作坐标、后果模型、目标代价逐项做迁移检验。若引入在线更新，记录模型更新使旧目标接口失效的程度，测未来交互收益，而非仅用 frozen probe 证明 CRL。

#### 原文与相关入口

- [2025 首稿](https://arxiv.org/abs/2506.09985v1)：action-free 预训练、2-AC 后训练、真实规划与第 4.3 节限制；此处不赋予未核实会议状态。
- [Meta FAIR 官方实现](https://github.com/facebookresearch/vjepa2)：包含 V-JEPA 2、2-AC 和较新的 2.1；版本不能混用。

#### 作者代码

[Meta FAIR 官方仓库；首稿模型与后续版本需按配置区分。](https://github.com/facebookresearch/vjepa2)

官方视频表征与动作条件模型；数据、机器人部署条件与检查点分别核验。


<a id="chapter-code"></a>

## 下载与运行

表格 Dyna、优先传播、option value iteration 与枚举 MPC；深度规划和想象训练另附原论文及作者工程。

[下载 knowledge_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/knowledge_algorithms_lab.py)

```sh
python3 knowledge_algorithms_lab.py planning
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto — Reinforcement Learning，第 8 章](http://incompleteideas.net/book/the-book-2nd.html)：Dyna、变化环境和 prioritized sweeping 的系统教材入口；本页用独立小链把各循环展开。

- [Moore & Atkeson — Prioritized Sweeping](https://doi.org/10.1007/BF00993104)：原论文。关键在模型前驱与优先调度，不能与只提高旧样本抽样频率的 prioritized replay 混为一谈。

- [Sutton et al. — Linear Dyna and Prioritized Sweeping](https://proceedings.mlr.press/r6/sutton08a.html)：原文。展示从前驱状态到前驱特征的扩展；收敛结论有明确线性模型/策略评估条件。

- [Sutton et al. — Reward-Respecting Subtasks](https://arxiv.org/html/2202.03466v3)：第 5 节以 learned option models 做规划；模型和发现成本应与规划收益一起评价。

- [Schrittwieser et al. — MuZero](https://arxiv.org/abs/1911.08265)：原文的 representation/dynamics/prediction 三个网络与树搜索；预测的是决策需要的量。

- [Google DeepMind — mctx](https://github.com/google-deepmind/mctx)：研究团队 JAX 搜索库。提供搜索接口及示例，不是完整 MuZero 训练代码。

- [Hafner et al. — DreamerV3](https://arxiv.org/abs/2301.04104)：原文。想象轨迹训练 actor/critic，与行动时搜索要区分；具体稳定化细节属于方法的一部分。

- [Danijar Hafner — DreamerV3 公开重实现](https://github.com/danijar/dreamerv3)：作者维护且 README 明确称 reimplementation。配置、模型训练比例和 imagined actor 梯度路径需一起读。

- [Sharma et al. — DADS 作者实现](https://github.com/google-research/dads)：技能动力学用于 latent skill MPC 的具体工程入口；比较“在原始动作空间规划”与“在已学技能空间规划”。

- [Hansen, Su & Wang — TD-MPC2 · ICLR 2024](https://arxiv.org/abs/2310.16828)：短期潜在模型、终点价值与策略先验如何共同支持决策时优化；区分多任务预训练与持续适应。

- [作者实现](https://github.com/nicklashansen/tdmpc2)：训练、模型与 plan 函数分别阅读。

- [Shehmar et al. — Laplacian Representations for Decision-Time Planning · ICML 2026](https://proceedings.mlr.press/v306/shehmar26a.html)：ALPS 正式论文：特征值缩放、区域子目标和短时域规划；实验设定为离线目标条件 RL。

- [ALPS 原文与算法细节](https://arxiv.org/abs/2602.05031)：第 4 节及附录给出 ALLO、原始状态前向模型、行为先验、图路径与 CEM 的分工。

- [Machado Research — ALPS 作者实现](https://github.com/machado-research/ALPS)：原始工程；分别追踪表示训练、模型训练、区域图和评估时规划，并保持作者任务与配置。

- [ALPS 高层路径 — planner/hierarchical.py](https://github.com/machado-research/ALPS/blob/main/planner/hierarchical.py)：plan 管理当前区域与中间目标；compute_cluster_path 调用图最短路。该实现不指定边权时比较的是跨越区域的次数。

- [ALPS 低层优化 — planner/optimizer.py](https://github.com/machado-research/ALPS/blob/main/planner/optimizer.py)：rollout_prior_mean 生成先验引导的动作序列；_cem_core 按代价筛选 elite 并更新分布；optimize_trajectory 将这些步骤接起来。

- [ICML 2024 原文](https://proceedings.mlr.press/v235/park24g.html)：Hilbert 距离、策略提示和定理前提；不是 ICLR 论文。

- [作者项目与公式](https://seohong.me/projects/hilp/)：时间距离与方向奖励接口。

- [官方实现](https://github.com/seohongpark/HILP)：hilp_zsrl 与 hilp_gcrl 对应不同实验。

- [NeurIPS 2020 原文](https://papers.nips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)：VE 依赖策略与函数集合。

- [Proper Value Equivalence · NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/400e5e6a7ce0c754f281525fae75a873-Abstract.html)：多步算子、固定点与规划充分性；不是任意潜在网络的保证。

- [NeurIPS 2023 原文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b0cd0e8027309ea050951e758b70d60e-Abstract-Conference.html)：proper VE 的不足、统计摘要与 Bellman 闭合。

- [作者实现](https://github.com/tylerkastner/distribution-equivalence)：原文第 7 节直接链接的实验代码。

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html)：短期预测、终点价值、多任务协议。

- [ICML 2025 原文](https://proceedings.mlr.press/v267/zhou25t.html)：正式发表入口；早期 ICLR 投稿页不能替代此状态。

- [作者项目代码](https://github.com/gaoyuezhou/dino_wm)：train.py、plan.py、数据与已公开模型检查点范围。

- [2025 首稿](https://arxiv.org/abs/2506.09985v1)：action-free 预训练、2-AC 后训练、真实规划与第 4.3 节限制；此处不赋予未核实会议状态。

- [Meta FAIR 官方实现](https://github.com/facebookresearch/vjepa2)：包含 V-JEPA 2、2-AC 和较新的 2.1；版本不能混用。


---

# 知识保留：经验重放、参数约束与模型记忆

新经验要求适应，旧技能又可能重新有用；如何在固定预算内管理它们的冲突？

## 本章内容

- 区分任务再次出现时的遗忘、分布改变后的必要适应，以及网络失去学习能力。
- 从目标函数推导 replay、EWC、策略蒸馏与 CLEAR，并把算法落实为完整存储/采样/更新流程。
- 用固定内存、相同数据与计算预算设计保留—适应对照，而不让“存得更多”掩盖算法贡献。

<a id="problem-definition"></a>

## 本章的问题定义

新经验要求改变行为，而过去能力可能再有用途；在固定存储和更新预算内控制保留—适应冲突。

### 给定条件与符号

- 当前经验、历史能力的独立评价分布和未来回访协议。
- 固定内存、更新预算、可保存的经验/参数/教师输出及是否提供任务边界。

### 需要求解的对象

在当前适应与声明历史能力上满足可检验取舍的学习器；保留对象可为经验、参数近似或功能输出。

### 信息与数据权限

旧数据和教师输出来自当时行为；重放需处理行为差异，动力学变化不能仅由动作概率比修复。评测副本不得反馈数据或重置主智能体。

$$
\min L_{\rm now}(\theta)\quad\text{s.t.}\quad L_{\rm hist}(\theta)-L_{\rm hist}(\theta_{\rm ref})\le\varepsilon,\quad M\le B
$$

$\theta$ 为当前参数，$\theta_{\rm ref}$ 是历史参考，$L_{\rm now}$ 为当前声明损失，$L_{\rm hist}$ 是固定历史诊断分布的损失，$\varepsilon$ 为允许退化，$M$ 为持久内存，$B$ 为预算。此式定义一种可操作取舍；replay/EWC/CLEAR以不同代理近似它，RL最终还需在线外部收益。

### 成立条件与解的含义

- 历史诊断保持同一任务语义；永久失效的旧行为不应自动当作必须保持的能力。
- 固定预算包含buffer、教师、参数锚与统计；旧任务边界和真标签是额外权限。

判断准则：A→B→A或等价独立probe记录A的下降、B学习速度与A恢复；匹配容量、数据和更新数，报告全程收益，区分遗忘、负迁移和必要适应。

### 适用边界

- 保留不是无条件阻止参数变化或复制旧错误。
- 历史训练损失低不能替代旧能力在固定分布上的诊断。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [可塑性与特征更新](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)：保留测旧能力下降，可塑性测新学习速度；两者可能同时出现但需不同对照。

- 组合不同学习问题 · [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)：模型/SF也可保留结构知识，但模型过期与重估成本须另测。

- 改变信息或数据协议 · [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)：原始经验replay超出严格不重放协议，参数统计和功能约束也需记入预算。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

当前梯度会覆盖旧行为，而保存全部经验或每任务完整模型不可持续扩展。

### 本章的核心思路

先确定未来需保留什么，再用预算内的经验、局部几何或教师输出近似其约束。

1. [选择代表的历史经验分布](#lesson-derive)：因为FIFO与reservoir保留不同年龄分布，明确保存与训练采样规则，匹配所需历史评价。

2. [纠偏历史学习并约束功能漂移](#lesson-clear)：因为旧行为与当前策略不同，CLEAR用V-trace处理旧经验并用旧输出克隆限制漂移，两项目的分开。

3. [以局部参数几何压缩历史](#lesson-ewc)：因为不能总保留数据，EWC保存锚点与对角Fisher近似；检查边界触发、容量和局部近似误差。

结论与条件：reservoir的时间索引保留概率可精确检验；EWC是局部曲率代理，CLEAR依赖覆盖和off-policy条件，均不提供任意变化任务的无遗忘保证。

### 相关方法改变了什么

- Replay：保留真实输入和经验，采样分布与行为纠偏决定用途。

- EWC：保留参数附近的局部损失几何，忽略部分相关方向。

- 蒸馏/CLEAR：保留指定状态上的功能输出，并可用RL项继续改进，强约束也会保存错误。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 训练分布与评价分布

当前数据分布由环境和行为策略共同产生；评价可以关注当前环境、历史环境混合或整个生命期收益，需要事先指定。

### KL 与策略蒸馏

$D_{\rm KL}(p\Vert q)=\sum_a p(a)\log[p(a)/q(a)]$。蒸馏在指定状态上保持教师分布 $p$，学生参数仍可改变。

### 二阶近似

在旧解附近，用曲率加权的二次函数近似旧损失；该近似描述局部敏感性。

<a id="lesson-setting"></a>

## 1 · 保留的对象与评价目标

一个导航智能体曾学会走左门。门的开关暂时改变，之后可能恢复原状。若新训练破坏了左门技能，回来时重新学习很慢，这是知识保留问题；若门永久封死，强迫新策略继续走左门反而有害。持续学习不是无条件保留一切，而是明确未来会如何使用历史知识。

用 A→B→A 的重复环境可以观察保留，但它只是一个研究协议：是否给任务 ID、是否外部 reset、每段多长、是否重新初始化优化器，都影响解释。真实 single-life 不能随时回 A 测试；可以用独立评测副本诊断，但副本不能向在线 agent 提供额外训练数据或重置其主状态。

| 现象 | 可检验问题 | 不能混称为 |
| --- | --- | --- |
| 遗忘 | A 的表现训练 B 后是否下降？ | 不一定是 B 学不会 |
| 负迁移 | 有历史的 agent 学 B 是否比 fresh 慢？ | 不一定是旧 A 被遗忘 |
| 可塑性丧失 | 相同新数据与优化预算下 aged 学习能力是否下降？ | 仅凭低回报不能确诊 |
| 必要适应 | 旧行为已失效时能否及时改变？ | 不是所有旧性能下降都应阻止 |

$$
L_t(\theta)=(1-\omega)L_{\rm now}(\theta)+\omega L_{\rm retained}(\theta),\qquad 0\leq\omega\leq1
$$

这是说明稳定—适应取舍的目标骨架。历史损失可以由真实 replay、参数近似或旧模型输出构造；不同构造保留的是不同信息。

先固定预算：参数、目标网络、教师网络、buffer 中的观测/动作/概率/价值、统计量都按字节计算；还要固定每个环境步允许的梯度更新数。一个方法多训练十次或保存整个历史，不能只与只用一条新样本的方法比较“算法更好”。

<a id="lesson-derive"></a>

## 2 · 经验重放与历史采样分布

Replay 使已经不再出现的状态继续影响梯度。FIFO buffer 强调近期数据；reservoir sampling 则在固定容量 $M$ 下，让前 $t$ 条样本具有相同的保留概率。两者代表的训练分布不同，应由未来评价需求决定。

**算法：Reservoir sampling；保存分布与训练时的采样分布分别定义**

1. 初始化样本计数 $t=0$，buffer 容量为 $M$
1. 每来一条 transition：
  1. 令 $t\leftarrow t+1$
  1. 若 buffer 未满，追加样本
  1. 否则均匀采样 $j\in\{0,\ldots,t-1\}$
    1. 若 $j<M$，以新样本替换 buffer 的第 $j$ 项
1. 训练时按指定的新旧样本比例采样并更新

当 $t>M$ 时，新样本被接收的概率为 $M/t$。任一旧样本此前被保留的概率为 $M/(t-1)$，本步被替换的条件概率为 $1/t$；于是存活概率为 $[M/(t-1)](1-1/t)=M/t$。均匀的是时间索引，并不意味着每个任务、状态或稀有事件等权。

$$
\mathbb E[\widehat g]= (1-\omega)\mathbb E_{d_t}[\nabla\ell]+\omega\mathbb E_{d_B}[\nabla\ell]
$$

dB 由保存与采样共同决定。损失权重、batch 中新旧比例和重用次数都能改变实际有效权重，不能只报告 buffer 容量。

强化学习还有一个监督学习没有的困难：旧 transition 由旧策略 μ 产生，当前要改进的是 π。旧奖励和转移样本在平稳环境中仍可有效，但策略梯度与多步 bootstrap 需要处理行为差异；环境动力学若也变了，动作重要性比并不能把旧世界样本变成新世界样本。

<a id="lesson-clear"></a>

## 3 · CLEAR：新经验学习、旧经验校正、旧输出克隆

CLEAR 结合三类更新：当前经验支持适应；历史经验通过 off-policy actor–critic 继续改善价值与行为；旧状态上的策略和价值克隆约束输出漂移。为此，存储项除转移外还包含旧策略分布与旧价值输出。该方法不要求任务边界，但保留效果取决于 replay 覆盖。

$$
\begin{aligned}\bar\rho_t&=\min(\rho_{\max},\pi(A_t\mid S_t)/\mu(A_t\mid S_t)),\\c_t&=\min(c_{\max},\pi(A_t\mid S_t)/\mu(A_t\mid S_t)),\\d_t&=\bar\rho_t[R_{t+1}+\gamma V(S_{t+1})-V(S_t)],\\v_s^{\rm trace}&=V(S_s)+\sum_{t=s}^{s+n-1}\gamma^{t-s}\!\left(\prod_{i=s}^{t-1}c_i\right)d_t.\end{aligned}
$$

这是常数折扣下长度为 $n$ 的 V-trace 目标，空乘积为 1。$\bar\rho_t$ 调整本步误差，其截断影响固定点对应的有效策略；$c_t$ 控制跨步传播与方差，在原理论条件下不改变该固定点。真实终止位置用零折扣截断，unroll 的非终止末端保留价值 bootstrap。

$$
\begin{aligned}L_V&=\tfrac12(V_\theta(s)-\operatorname{stopgrad}(v_s^{\rm trace}))^2,\\L_\pi&=-\log\pi_\theta(a_s\mid s)\operatorname{stopgrad}\!\left[\bar\rho_s(r+\gamma v_{s+1}^{\rm trace}-V_\theta(s))\right],\\L_{\rm clone}&=\beta_\pi D_{\rm KL}(\mu_{\rm stored}\Vert\pi_\theta)+\beta_V(V_\theta-V_{\rm stored})^2.\end{aligned}
$$

RL 项应用于新旧数据，cloning 项用于 replay；再按原配置加入熵项和各损失权重。旧 logits、旧值与 trace target 不通过梯度回流到历史模型。actor 的整个重要性加权 advantage 都要 stop-gradient；若把 ρ̄ 留在外面，会额外对概率比求导，改变政策梯度估计器。

**算法：算法伪代码**

1. 动作采样前保存 μ(a|s)、完整策略输出和 Vstored(s)
1. 环境返回 transition；将新 unroll 与历史 unroll 按规定比例组成训练数据
1. 用当前网络计算 π、V；用存储 μ 计算裁剪的重要性比
1. 自后向前计算 V-trace targets；将 targets stop-gradient
1. actor 使用整体 detach 的 weighted advantage（包括重要性比）；计算 critic、entropy；只对 replay 加策略/价值 cloning
1. 合并损失做一次优化；按容量策略存入新经验
1. 记账 buffer 字节数、unroll 长度、每步更新数与 replay 年龄

旧策略未必最优，因此克隆与 RL 更新有不同作用：前者限制偏离，后者利用奖励继续改进。强克隆可能保存旧错误，弱克隆则可能无法保留旧能力。比较方法时需要同时观察当前适应与历史回访表现。

<a id="lesson-ewc"></a>

## 4 · EWC：用参数曲率近似旧知识

无法保存旧数据时，可以保留旧解 $\theta_A$ 与每个参数的重要性。若旧损失在 $\theta_A$ 的梯度近似为零，二阶展开中的 Hessian 描述沿哪些方向移动代价大。EWC 用对角 Fisher 近似这种局部几何，从而让重要参数较难改变。

$$
\begin{aligned}L_A(\theta)&\approx L_A(\theta_A)+\tfrac12(\theta-\theta_A)^\top H_A(\theta-\theta_A),\\L_{\rm EWC}(\theta)&=L_B(\theta)+\frac{\kappa}{2}\sum_iF_i(\theta_i-\theta_{A,i})^2,\\\nabla_iL_{\rm EWC}&=\nabla_iL_B+\kappa F_i(\theta_i-\theta_{A,i}).\end{aligned}
$$

Fisher 来自指定概率模型的 score 外积期望：$F_i=\mathbb E[(\partial_{\theta_i}\log p_\theta(y\mid x))^2]$。真实 Fisher 在模型分布上取期望；经验 Fisher 改用观测标签。两者与实际 RL 损失的 Hessian 不普遍相等，因此样本、动作及分布都是估计器定义的一部分。

完整流程是：在约定的 consolidation 时刻保存 θA；在相应数据分布上估计并冻结 F；后续每步把二次惩罚加入当前损失。若没有任务边界，要定义 consolidation 的触发或连续累计规则。每个任务另存一个 F 与参数快照会随任务数增长；online EWC 将历史重要性折叠成固定规模统计，但其遗忘系数与近似误差需要评估。

EWC 保留的是参数附近的旧损失近似，不是完整旧数据，也不保证旧动作概率不变。网络可用不同参数实现同一功能；对角近似忽略参数之间的相关方向。因此功能空间的蒸馏与参数空间的正则可以产生明显不同的行为。

<a id="lesson-model-memory"></a>

## 5 · DRAGO：保留世界模型，而非只保留旧策略

如果任务改变的只是奖励与起点，不同任务会让智能体看到同一世界的不同区域。此时值得保留的是动力学知识：旧任务的最优策略未必适用于新任务，但旧房间的布局仍然有效。DRAGO（Fu 等，ICML 2025）研究这一设定，允许已知任务切换、当前任务 buffer，以及一个旧模型快照；不保存往期任务的原始数据。

先用生成器 $G$ 表示过去访问过的状态—动作联合分布。从中采样 $(\hat s,\hat a)$，由冻结的旧动力学模型 $T_{\rm old}$ 产生后继标签。当前模型同时拟合真实新经验与这些合成旧经验：

$$
\begin{aligned}(\hat s,\hat a)&\sim p_G,\qquad \hat s'=\operatorname{stopgrad}(T_{\rm old}(\hat s,\hat a)),\\L_{\rm dyn}(\psi)&=\mathbb E_{(s,a,s')\sim D_i}\|T_\psi(s,a)-s'\|^2+\lambda\mathbb E_{(\hat s,\hat a)\sim p_G}\|T_\psi(\hat s,\hat a)-\hat s'\|^2.\end{aligned}
$$

第一项校正当前可见区域，第二项保护过去区域的模型输出。与 CLEAR 的区别在于教师输出是后继状态预测，输入则由生成器产生。

生成器自身也会遗忘，因此用当前真实状态—动作对与上一生成器产生的样本共同训练新生成器。DRAGO 还用真实探索重新连接旧区域；这一部分补充了生成数据无法证明当前真实可达性的缺口。保留与重新获取知识是两种互补操作。

**算法：DRAGO 的知识保留流程；策略、价值、奖励头及探索器仍有各自训练目标**

1. 在已知任务边界保存冻结副本 $T_{\rm old}$ 与 $G_{\rm old}$
1. 每个训练阶段：
  1. 收集当前任务的真实转移，维护当前 buffer
  1. 从生成器采样旧状态—动作对，计算冻结模型的后继标签
  1. 以真实项与合成项联合更新当前动力学模型
  1. 以当前真实输入和 $G_{\rm old}$ 的样本训练当前生成器
  1. 用任务行为与记忆恢复行为收集新的真实经验
1. 下次任务切换时更新冻结副本

这个方法的边界来自问题设定：若真实动力学永久改变，旧模型标签可能已经错误，保持它会妨碍适应；若生成器遗漏了某个区域，蒸馏便没有对应输入。实验应分别测模型保持误差、跨区域规划与新任务适应，并计入生成器和冻结副本的存储、训练成本。作者 drago 仓库提供完整模型式实现；本页脚本仍聚焦 reservoir、EWC 和 CLEAR 的数值机制。

<a id="lesson-example"></a>

## 6 · 稳定性与适应性的二次例子

$$
L(w)=\tfrac12(w+1)^2+\tfrac\kappa2(w-1)^2,\qquad \frac{dL}{dw}=(w+1)+\kappa(w-1),\qquad w_*=\frac{\kappa-1}{\kappa+1}
$$

旧目标是 w=1，新目标是 w=−1，F=1。κ=0、1、5 分别得到 −1、0、2/3；这是精确二次例子，不是深网保证。

κ=0 完全适应新任务；κ=1 平均折中，但两个任务都不完美；κ=5 更接近旧解，却留下较大的新误差。这说明“新任务回报低”可能是强保留目标的结果，不一定是网络失去学习能力。

功能克隆的例子：旧策略 $\mu=(0.8,0.2)$，新策略 $\pi=(0.5,0.5)$，故 $D_{\rm KL}(\mu\Vert\pi)=0.8\log1.6+0.2\log0.4\approx0.192745$。旧值为 2、新值为 1 时，值克隆平方误差为 1。交换 KL 方向会改变数值与梯度，教师和学生的角色须固定。

两步 V-trace 样本的奖励为 $[0,1]$，价值全零、$\gamma=0.9$，所有比率与裁剪系数为 1，则第一状态目标为 $0.9$。若第一步目标策略概率为零，$\bar\rho_0=c_0=0$，第一状态本次没有轨迹修正，保持原值。这展示了当前误差与后续误差乘积的边界。

<a id="lesson-code"></a>

## 7 · 可运行实现与实验设计

固定容量 reservoir、EWC 精确小例子、克隆项与有限 unroll V-trace；不是完整 CLEAR 神经网络代理。

```python
def reservoir_insert(buffer, item, seen, capacity, rng):
    """seen is the number INCLUDING this item; O(capacity) retained data."""
    if capacity < 0 or seen < 1:
        raise ValueError("invalid budget or sample counter")
    if len(buffer) < capacity:
        buffer.append(item)
    elif capacity:
        index = rng.randrange(seen)
        if index < capacity:
            buffer[index] = item


def ewc_scalar_optimum(new_target, old_weight, importance, strength):
    if importance < 0 or strength < 0:
        raise ValueError("curvature and strength must be nonnegative")
    k = importance * strength
    return (new_target + k * old_weight) / (1.0 + k)


def ewc_update(weights, current_gradient, old_weights, importance,
               strength=1.0, alpha=0.1):
    """One SGD update; current_gradient excludes the consolidation penalty."""
    if not (len(weights) == len(current_gradient) == len(old_weights) == len(importance)):
        raise ValueError("EWC dimensions differ")
    if strength < 0 or any(f < 0 for f in importance):
        raise ValueError("EWC requires nonnegative importance")
    return [w - alpha * (g + strength*f*(w-old))
            for w, g, old, f in zip(weights, current_gradient, old_weights, importance)]


def clone_loss(teacher_prob, student_prob, teacher_value, student_value):
    """CLEAR-like replay cloning terms only, not the complete RL objective."""
    if len(teacher_prob) != len(student_prob):
        raise ValueError("policy dimensions differ")
    if any(p < 0 for p in teacher_prob) or any(q <= 0 for q in student_prob):
        raise ValueError("teacher >=0, student >0 required")
    if not math.isclose(sum(teacher_prob), 1.) or not math.isclose(sum(student_prob), 1.):
        raise ValueError("policies must sum to one")
    kl = sum(p * math.log(p / q) for p, q in zip(teacher_prob, student_prob) if p)
    return kl, (student_value - teacher_value) ** 2


def vtrace_target(rewards, values, rhos, gamma=0.9, rho_cap=1., c_cap=1.):
    """Finite unroll value target. values includes the final bootstrap value."""
    if len(values) != len(rewards) + 1 or len(rhos) != len(rewards):
        raise ValueError("unroll dimensions differ")
    if min(rho_cap, c_cap) < 0 or c_cap > rho_cap or any(rho < 0 for rho in rhos):
        raise ValueError("require 0 <= c_cap <= rho_cap and nonnegative ratios")
    correction, product = 0., 1.
    for t, reward in enumerate(rewards):
        delta = min(rho_cap, rhos[t]) * (reward + gamma * values[t+1] - values[t])
        correction += product * delta
        product *= gamma * min(c_cap, rhos[t])
    return values[0] + correction


def retention_demo():
    buffer, rng = [], random.Random(7)
    for seen in range(1, 101):
        reservoir_insert(buffer, seen, seen, 5, rng)
    weight = [1.]
    for _ in range(100):
        weight = ewc_update(weight, [weight[0]+1.], [1.], [1.], strength=1.)
    print("retention", {"fixed_budget_reservoir": buffer,
          "ewc_optima_k_0_1_5": [ewc_scalar_optimum(-1, 1, 1, k) for k in (0, 1, 5)],
          "ewc_sgd_k1": round(weight[0], 8),
          "clone_terms": clone_loss([0.8, 0.2], [0.5, 0.5], 2., 1.),
          "two_step_vtrace": vtrace_target([0, 1], [0, 0, 0], [1, 1])})
```

运行后先核对手算，再增加新旧分布和 buffer 预算

```sh
python lifelong_algorithms_lab.py retention
python lifelong_algorithms_lab.py test
```

脚本检查固定容量、零容量、二次目标的驻点、教师与学生相同时的零克隆损失，以及 V-trace 的目标策略零概率边界。完整 CLEAR 还包含序列收集、分布式 actor 与神经网络更新；这些应在机制函数正确后逐层加入。

- 最小研究矩阵：online-only、FIFO replay、reservoir replay、replay+clone、EWC；固定总字节数与每步梯度计算。
- 评价至少三项：当前适应曲线、旧任务保持/重学速度、全生命期累计收益；另记真实回访与诊断评测的区别。
- 当环境永久改变，增加“有意遗忘”的对照：统一历史 replay 是否持续施加过时约束？
- 把任务 ID、教师 checkpoint、离线旧数据访问列为显式资源，不与无需这些信息的方法直接混淆。

<a id="lesson-branches"></a>

## 8 · 数据、参数、功能、模块与快慢学习

| 条线 | 保存对象和更新机制 | 主要代价 |
| --- | --- | --- |
| Replay / CLEAR | 旧样本与旧输出；混合 RL 学习及功能约束 | 观测存储、旧分布偏差、多步行为校正 |
| EWC / online EWC | 旧参数与重要性；加二次惩罚 | 局部/对角近似、consolidation 调度 |
| Distillation | 旧策略或表示输出；在指定输入上对齐 | 需要有代表性的输入；教师错误也会保留 |
| Progressive / modular | 把新知识放在新或路由模块中，限制相互干扰 | 模块增长、任务识别与路由、知识复用成本 |
| Progress & Compress | 快的 active column 学习，再蒸馏进慢的 knowledge base | 双系统预算、压缩时间、压缩时旧知识保护 |

快慢系统的本质是时间尺度分离：快速部分吸收近期证据，慢速部分整合可复用信息。若压缩只在新状态上进行，旧功能可能没有监督；如果每次任务都增长模块，就必须把无限寿命下的容量问题列出来。认知科学的互补学习系统可提供设计动机，但动机并不能替代算法和实验的因果证据。

一个可检验的研究问题是：在无任务 ID 的循环环境中，总内存固定为 10 MB，将一部分观测存储预算改为保存旧 logits，是否改善全程收益与回访表现？相同原则也适用于 DRAGO：生成器、旧模型和少量真实 replay，应在同一存储预算下比较。

<a id="research-upgd-utility"></a>

## 研究专题 A · UPGD：保留与可塑性耦合在同一更新中

固定重要参数可以保护旧功能，却可能阻碍必要适应；无选择地扰动又可能破坏仍有用的功能。UPGD 根据近期效用，让一些方向少变化，另一些方向更容易改变。效用的干预方式与数据范围是定义的一部分，不能直接叫作永久知识重要性。

$$
U_i=L(w-w_ie_i)-L(w)\approx-w_i\frac{\partial L}{\partial w_i}+\frac12w_i^2\frac{\partial^2L}{\partial w_i^2}
$$

U 是将第 i 个权重置零造成的损失变化。右侧为局部二阶 Taylor 近似；大幅移除时可失准，特征级与权重级干预也不相同。

取 $L(w)=(w-1)^2/2$。在 $w=1$，精确移除效用为 0.5，一阶项却为零，二阶项补足 0.5。在 $w=-1$，精确效用为 −1.5，一阶项为 −2，二阶近似为 −1.5。负效用只说明这个分布上移除有益，不说明未来永远无用。

$$
\bar U_{t,i}=\beta\bar U_{t-1,i}+(1-\beta)U_{t,i},\qquad w_{t+1,i}=(1-\alpha\lambda)w_{t,i}-2\alpha(1-\widetilde U_{t,i})(g_{t,i}+\xi_{t,i})
$$

对应作者 README 的全局缩放、一阶、权重级短实现骨架。Ũ 经偏差修正、全局参照与 sigmoid 得到，ξ 是指定尺度扰动。论文另有效用/特征变体，不能混用。

同一保护系数同时乘在梯度和噪声上。因此高效用坐标不仅少受随机破坏，也可能更慢完成新任务更新；低效用坐标可能承担更大优化方差。weight decay 是第三条独立变化路径。只消融噪声不能解释全部效果。

| 对照 | 检验 |
| --- | --- |
| 统一乘子 vs 效用乘子 | 总体降低学习率还是方向选择？ |
| 只调梯度、只调噪声、同时调节 | 功能保护与参数多样性的不同作用。 |
| 精确效用、一阶、二阶 | 近似在漂移或饱和处何时失真？ |
| 近期分布、旧 probe、真实回访 | 近期有用与历史稀有重要知识是否相同？ |

**算法：从反事实效用到持续控制的验证路径**

1. 两坐标回归：精确计算逐坐标置零效用，固定相同噪声序列
1. A→B→A 数据流：不给切换标记，不清零效用统计
1. 报告当前误差、新目标速度、A 回访与整个过程损失
1. 再接入 PPO，明确 rollout 长度与重复更新预算

作者 mohmdelsayed/upgd 的 experiments/weight_utility.py 对应效用实验，core/run/rl 对应 PPO。需要核对缩放分母退化、梯度与参数是否来自同一步、统计是否跨任务保留。主体流式监督结果与 PPO 证据分别报告；换用在线 optimizer 不能使 PPO 自动满足严格一次使用经验协议。

<a id="lesson-check"></a>

## 9 · 习题与下一步

- 旧任务分数下降、新任务学习很快，说明什么？可能是遗忘或必要适应，不足以判断可塑性丧失。
- Reservoir 是否让每个任务等权？否，它让各时刻样本等概率；较长任务会占更多份额。
- 已知 μ(a|s) 能否修正所有历史偏差？不能；动作比率不修正环境转移本身已经改变，也不恢复从未保存的状态。
- 为什么保存 KL 教师也要计内存？每条样本的 logits/value 与教师 checkpoint 都是真实资源。
- 如果 EWC 的 F 全零会怎样？二次约束消失，退化为当前任务学习；若 κ 极大，则会压制必要适应。

动手题：用同一组 A→B→A 样本分别做 FIFO 和 reservoir，比较 buffer 内各阶段比例；再把 B 的长度扩大十倍，检查所谓“平衡历史”是否仍成立。先用确定性样本查清保存分布，再把真实 RL 策略反馈加进去。

## 本章的实验设计

冻结副本测试保留能力。保持更新的副本测试再适应。副本中的反馈不能写回主运行。

设定：使用 A→B→A，并在回访前复制隔离的评测分支。一个分支冻结，另一个允许新增学习。加入未见 C 以防“少遗忘但学不动”。

- A 的初次习得达到预定非平凡标准。
- 冻结回访样本不进入主 buffer 和归一化。
- 保存、回放与模型增长计入固定预算。

对照：无保留机制、固定容量 replay；相同存储与更新预算的 EWC/蒸馏；独立任务顺序与相同信息权限

记录：冻结保留与允许重学后的恢复曲线；B/C 获取速度和完整生命期回报；缓存、参数、蒸馏与梯度成本

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-continual)

## 学习与研究衔接

记住旧任务和快速学习新任务可能冲突。冻结诊断与保持更新的再适应实验分别回答不同问题。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-retention) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=retention) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=retention)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

模型可提取性的理论说明某类能力需要什么知识，不指定唯一网络。Dreamer 研究潜在想象控制，STOMP 研究随机时长行为模型，DRAGO 研究旧模型知识保留。模型误差、查询策略和规划收益之间仍需实验连接。

- [Knowledge Retention in Continual Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-drago-model-retention)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-upgd-utility-protection)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [Loss of plasticity in deep continual learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-backpropagation)
- [Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn](https://yingwen.io/zh/continual-rl/research/#recent-c-chain-churn)
- [Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline](https://yingwen.io/zh/continual-rl/research/#recent-reset-and-distill)
- [Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-fame-fast-meta-learners)
- [Knowledge Retention in Continual Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-drago-model-retention)
- [Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-vla-simple-recipe)
- [Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-upgd-utility-protection)
- [Parseval Regularization for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-parseval-continual-geometry)
- [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-trac-online-regularization)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-fame-fast-meta-learners)
- [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-trac-online-regularization)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-vla-simple-recipe)

### Loss of plasticity in deep continual learning

Shibhansh Dohare, J. Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A. Rupam Mahmood, Richard S. Sutton

Nature · 2024 · 直接研究持续学习

#### 研究问题

一个长期训练的网络如何保留继续形成新特征的能力？

#### 关键机制

Continual Backpropagation 在梯度学习之外持续生成并测试特征。它估计单元的效用和成熟度，少量替换低效用的成熟单元，并协调新单元的输入、输出和相关状态。维护新的可学习方向是一个持续过程，而不是等到任务切换后整体重启。

#### 证据

论文在长序列监督学习与强化学习问题中展示可塑性损失，并检验特征替换的作用。作者仓库包含 generate-and-test 与优化器状态处理。

#### 条件与限制

有限序列上的学习保持不保证无限生命中的任意适应。替换率、效用定义与成熟度条件仍需选择；新任务学习速度和旧能力保留必须分开测量。

#### 阅读与实验

逐项消融“成熟度筛选”“效用筛选”“随机替换”。比较相同替换预算，检验收益究竟来自定向回收还是一般参数扰动。

#### 原文与相关入口

- [Nature 原文](https://doi.org/10.1038/s41586-024-07711-7)：长期可塑性实验与 continual backpropagation。
- [作者代码](https://github.com/shibhansh/loss-of-plasticity)：关注 lop/algos/gnt.py 及替换时的优化器状态。

#### 作者代码

[论文作者公开的实验实现。](https://github.com/shibhansh/loss-of-plasticity)

论文任务、持续反向传播与 generate-and-test。

### Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn

Hongyao Tang, Johan Obando-Ceron, Pablo Samuel Castro, Aaron Courville, Glen Berseth

ICML 2025 · 2025 · 直接研究持续学习

#### 研究问题

一次局部更新为什么会在其他输入上引发大幅预测变化，并损害后续学习？

#### 关键机制

C-CHAIN 抑制相对于近期参考网络的函数输出变化，降低一次更新在其他样本上造成的 churn。论文把该现象与经验神经切线核及学习动力学联系起来。正则化对象是函数变化，不是直接把所有参数锁在旧值附近。

#### 证据

作者在持续 Gym Control、ProcGen、DMC 和 MinAtar 序列中比较，并提供对应环境和算法代码。

#### 条件与限制

近期函数稳定性不等于长期任务知识保留；参考样本和参考网络也占资源。若环境突然发生真实变化，过强抑制输出变化可能延迟必要适应。

#### 阅读与实验

将 churn 按旧分布、新分布分别计算，并同时画适应速度。这样才能区分“减少无关干扰”和“阻止有用改变”。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/tang25g.html)：机制、理论分析与持续实验。
- [作者代码](https://github.com/bluecontra/C-CHAIN)：四类持续环境的基线和 C-CHAIN 对照实现。

#### 作者代码

[作者仓库，README 说明依赖的 TRAC、CleanRL 与 MinAtar 基础实现。](https://github.com/bluecontra/C-CHAIN)

持续控制环境与 C-CHAIN 对照实验。

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

### Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning

Ke Sun, Hongming Zhang, Jun Jin, Chao Gao, Xi Chen, Wulong Liu, Linglong Kong

ICLR 2026 · 2026 · 直接研究持续学习

#### 研究问题

快速学习新任务和整合旧知识，能否由不同学习器承担并以明确目标连接？

#### 关键机制

FAME 的快速学习器适应当前任务，元学习器整合此前知识。论文按旧策略的重要访问分布度量价值或策略变化，再据此构造减少遗忘的整合目标。自适应预热决定如何利用旧知识初始化或约束早期行为，以减少负迁移。

#### 证据

论文分析价值型和策略型版本，并在像素与连续控制任务序列中比较。作者提供官方实现，可追踪快速适应与知识整合两个阶段。

#### 条件与限制

设定要求相同状态与动作空间、已知任务边界以及额外整合计算。这里的 meta learner 主要是知识整合模块，不应因名称就当作通过长期回报反向求导的在线元梯度算法。脑机制类比也不是神经科学实验证据。

#### 阅读与实验

分别报告新任务前向迁移、旧任务保留和两个学习阶段的计算量。改变任务相似性，检验自适应预热是否确实避免有害旧知识。

#### 原文与相关入口

- [ICLR 2026 原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2230ffcd5da10015ce0c6ce588fc2936-Abstract-Conference.html)：任务边界假设、遗忘度量与快慢知识机制。
- [FAME 官方实现](https://github.com/datake/FAME)：论文链接的快速学习与知识整合代码。

#### 作者代码

[论文与仓库均注明为官方实现。](https://github.com/datake/FAME)

FAME 的价值型、策略型持续学习实验。

### Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline

Hongjoon Ahn, Jinu Hyeon, Youngmin Oh, Bosun Hwang, Taesup Moon

ICLR 2025 · 2025 · 直接研究持续学习

#### 研究问题

一个网络还能拟合新目标，为什么先前训练仍可能让它在新任务上学得更慢？

#### 关键机制

论文把任务之间的负迁移与一般可塑性损失区分开。Reset & Distill 在新任务开始时重置在线 actor 和 critic，避免旧初始化阻碍学习；随后离线蒸馏当前策略与旧专家的动作分布以整合知识。适应和保留通过不同过程实现。

#### 证据

作者在控制与游戏任务中分析负迁移，并在长 MetaWorld 序列上检验该基线。原文直接提供实现地址。

#### 条件与限制

任务边界、在线网络重置、旧专家和离线蒸馏都需要资源。它不能直接当作无边界、不能重置、禁止回放的单次生命方案。

#### 阅读与实验

除了与连续微调比较，还要与同等预算的从头训练比较。若新任务表现低于从头训练，先检查负迁移，再判断是否属于单纯容量损失。

#### 原文与相关入口

- [ICLR 2025 原文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ba9e3d60610f3525717665966d86e0cd-Abstract-Conference.html)：负迁移诊断、Reset & Distill 机制与边界。
- [原文代码入口](https://github.com/hongjoon0805/Reset-Distill)：论文首页提供的作者实现。

#### 作者代码

[ICLR 正式论文首页明确链接的代码。](https://github.com/hongjoon0805/Reset-Distill)

Reset & Distill 以及任务序列实验。

### Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning

Jiaheng Hu, Jay Shim, Chen Tang, Yoonchang Sung, Bo Liu, Peter Stone, Roberto Martín-Martín

RLC 2026 · 2026 · 直接研究持续学习

#### 研究问题

大规模预训练的视觉—语言—动作模型，是否仍需要复杂机制才能顺序学习控制任务？

#### 关键机制

论文研究对预训练 VLA 进行顺序强化学习，并以低秩适配等相对简单的训练流程检验持续学习。预训练表示、可更新参数子空间和 RL 目标共同决定迁移与遗忘，不能只把结果归因于单一保留正则项。

#### 证据

作者在多种 VLA 与长期任务基准上比较，并提供实验代码。RLJ 的 RLC 2026 论文页与论文脚注分别给出正式入口和作者仓库。

#### 条件与限制

预训练数据和算力属于外部资源，任务与重置协议也影响难度。该结果不意味着从零训练的网络不会遗忘，更不意味着任意无边界任务流只需微调。

#### 阅读与实验

固定预训练模型，分别改变可训练参数量和任务顺序。报告预训练资源、每任务在线数据、回放或重置条件，再与传统 CRL 方法比较。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2603.11653)：VLA 持续学习设置、机制与比较。
- [RLC 2026 / RLJ 论文页](https://rlj.cs.umass.edu/2026/papers/Paper84.html)：会议原文入口；PDF 脚注链接作者代码。
- [作者实现](https://github.com/UT-Austin-RobIn/continual-vla-rl)：持续 VLA 的训练与评价代码。

#### 作者代码

[UT Austin RobIn 实验室的原论文仓库。](https://github.com/UT-Austin-RobIn/continual-vla-rl)

预训练 VLA 的顺序 RL 训练和论文评价。

### Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning

Aneesh Muppidi, Zhiyu Zhang, Heng Yang

NeurIPS 2024 · 2024 · 直接研究持续学习

#### 研究问题

未知环境变化时间和速度时，怎样在线决定参数应离参考初始化多远？

#### 关键机制

TRAC 在基础优化器外维护一组具有不同遗忘时间尺度的一维 tuner，根据梯度与参考方向的内积调整参数位移尺度。它通过数据驱动的缩放联系到正则化，而不是对未来任务回报进行长窗口元梯度反传。

#### 证据

作者在 Procgen、Atari 与 Gym Control 变化序列中比较适应与可塑性，并分析在线凸优化对该设计的启发。

#### 条件与限制

凸在线优化中的遗憾理论不等于非凸、策略依赖采样的深度 RL 收敛定理。“parameter-free”不表示没有基础学习率、初始化、时间尺度网格、warm-start 或协议选择。

#### 阅读与实验

记录 tuner 尺度、距参考点的位移、旧分布干扰与变化后适应。用相同基础优化器比较固定尺度、单时间尺度和多时间尺度。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/5b76d77e7095c6480ed827b85f0c2878-Paper-Conference.pdf)：Algorithm 1–2、正则化联系与持续实验。
- [作者论文 v3](https://arxiv.org/html/2405.16642v3)：区分凸理论、RL 经验结果与初期表现限制。

#### 作者代码

[作者项目页与仓库均明确标为官方实现。](https://github.com/ComputationalRobotics/TRAC)

trac.py、PyTorch/JAX optimizer 包与控制/视觉实验。

### Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning

Mohamed Elsayed, A. Rupam Mahmood

ICLR 2024 · 2024 · 支持方法与理论

#### 研究问题

同一网络里，哪些方向应当保护，哪些方向应当获得更强的新学习与扰动？

#### 关键机制

UPGD 用移除权重或特征的反事实损失变化定义效用，并以 Taylor 近似在线估计。平滑、缩放后的效用同时调制梯度与随机扰动，让近期高效用方向变化较小、低效用方向更活跃。

#### 证据

主体证据包括未知边界的非平稳流式监督任务；另外包含长时间 PPO 实验。两类证据应分别理解，不能把监督任务数量写成 RL 任务覆盖。

#### 条件与限制

近期分布上的效用不保证稀有旧知识的重要性；一阶和二阶近似、权重级和特征级版本不同。PPO 仍使用 rollout 与重复更新，不因 optimizer 在线就成为严格流式 RL。

#### 阅读与实验

用可精确消融的小网络检查效用估计，再拆开保护梯度、保护噪声和 weight decay 三种作用；独立报告新学习与旧功能。

#### 原文与相关入口

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/file/8e5f0591943d8dae5702af12dcdcd2f6-Paper-Conference.pdf)：效用定义、近似、不同 UPGD 变体与 PPO 实验。
- [作者预印本](https://arxiv.org/abs/2404.00781)：流式监督协议与 RL 证据范围。

#### 作者代码

[论文首页明确链接的作者仓库；README 的短实现是一个指定变体。](https://github.com/mohmdelsayed/upgd)

权重/特征效用实验、流式任务及 PPO 实现。

### Parseval Regularization for Continual Reinforcement Learning

Wesley Chung, Lynn Cherif, David Meger, Doina Precup

NeurIPS 2024 · 2024 · 直接研究持续学习

#### 研究问题

仅在初始化时保持良好的权重几何，是否足以让很晚出现的新任务仍容易学习？

#### 关键机制

在选定隐藏层加入 $\lambda\|WW^\top-sI\|_F^2$，持续约束行向量的范数与角度；输出层及额外尺度设计保留表达能力。它维护学习的几何条件，并不直接保存旧任务标签或预测。

#### 证据

作者在 Gridworld、CARL、MetaWorld 任务序列中检验，并拆分范数与角度约束。稳定秩、Jacobian 与熵属于诊断量，不单独构成可塑性或保留的因果证明。

#### 条件与限制

约束会限制函数类；输出行数大于输入维度时，全部行正交不可实现。非线性门控仍能切断梯度。有限任务序列的结果不保证无限生命内有效，也不是无任务信息的万能机制。

#### 阅读与实验

同预算比较仅初始化正交、持续范数约束、持续角度约束和完整正则；同时记录新目标拟合、真实回报、旧功能与额外计算。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/e6df4efa20adf8ef9acb80e94072a429-Paper-Conference.pdf)：目标函数、容量限制、角度/范数消融及持续任务协议。
- [作者版本记录](https://arxiv.org/abs/2412.07224)：正式会议年份为 2024。

#### 作者代码

[仓库明确标为 NeurIPS 2024 官方实现。](https://github.com/wechu/parseval_reg)

PPO、任务序列、正则化与网络结构消融。


<a id="chapter-code"></a>

## 下载与运行

存储策略、目标项和 trace 数值实验；完整 CLEAR 训练还需要神经网络、序列收集与环境配置。

[下载 lifelong_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)

```sh
python lifelong_algorithms_lab.py retention
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Rolnick et al. · Experience Replay for Continual Learning (CLEAR)](https://papers.nips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html)：新旧经验混合、off-policy 学习与行为/价值克隆的原论文。

- [Espeholt et al. · IMPALA](https://proceedings.mlr.press/v80/espeholt18a.html)：V-trace 原始推导及裁剪系数所控制的偏差、方差与目标策略。

- [Kirkpatrick et al. · Overcoming catastrophic forgetting in neural networks](https://doi.org/10.1073/pnas.1611835114)：EWC 的概率解释与 Fisher 近似；注意不同 Fisher 估计器的语义。

- [Schwarz et al. · Progress & Compress](https://proceedings.mlr.press/v80/schwarz18a.html)：固定架构大小的 active/knowledge 双系统与 online EWC 压缩。

- [Fu et al. · Knowledge Retention in Continual Model-Based RL · ICML 2025](https://proceedings.mlr.press/v267/fu25f.html)：DRAGO 的生成回放与真实记忆恢复；共享动力学、已知任务边界，当前任务数据仍可重放。

- [DRAGO 作者实现](https://github.com/YixiangSun/drago)：对应生成器、冻结旧动力学模型和任务/回访行为；与仅用 reservoir 的模型自由方法有不同预算。

- [AGI-Labs continual_rl](https://github.com/AGI-Labs/continual_rl)：可扩展 CRL 比较框架，包含统一接口下的方法复现；实现来源与原论文作者仓库分别标识。

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/5b76d77e7095c6480ed827b85f0c2878-Paper-Conference.pdf)：Algorithm 1–2、正则化联系与持续实验。

- [作者论文 v3](https://arxiv.org/html/2405.16642v3)：区分凸理论、RL 经验结果与初期表现限制。

- [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning · 作者实现](https://github.com/ComputationalRobotics/TRAC)：trac.py、PyTorch/JAX optimizer 包与控制/视觉实验。 作者项目页与仓库均明确标为官方实现。

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/file/8e5f0591943d8dae5702af12dcdcd2f6-Paper-Conference.pdf)：效用定义、近似、不同 UPGD 变体与 PPO 实验。

- [作者预印本](https://arxiv.org/abs/2404.00781)：流式监督协议与 RL 证据范围。

- [Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning · 作者实现](https://github.com/mohmdelsayed/upgd)：权重/特征效用实验、流式任务及 PPO 实现。 论文首页明确链接的作者仓库；README 的短实现是一个指定变体。

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/e6df4efa20adf8ef9acb80e94072a429-Paper-Conference.pdf)：目标函数、容量限制、角度/范数消融及持续任务协议。

- [作者版本记录](https://arxiv.org/abs/2412.07224)：正式会议年份为 2024。

- [Parseval Regularization for Continual Reinforcement Learning · 作者实现](https://github.com/wechu/parseval_reg)：PPO、任务序列、正则化与网络结构消融。 仓库明确标为 NeurIPS 2024 官方实现。


---

# 可塑性：梯度通路、有效学习率与预测干扰

网络不是不会做旧事，而是越来越难学会新事；怎样诊断、定位并恢复这种能力？

## 本章内容

- 用 matched aged/fresh probe 区分能力丧失、探索失败和普通遗忘。
- 从梯度通路推导 dormant-unit 机制，具体实现 ReDo 与 CBP 的选择、替换、成熟期和优化器状态处理。
- 理解 CReLU、正则化、网络重置、plasticity injection 等分支的不同作用及保留代价。

<a id="problem-definition"></a>

## 本章的问题定义

长时间训练后，网络在同样新数据与优化预算下变得难以学习；需先排除探索和任务难度差异。

### 给定条件与符号

- 同容量aged/fresh学习器、匹配新目标、数据顺序和优化预算。
- 固定probe评价、优化器与归一化协议，以及允许的单元替换/参数扰动预算。

### 需要求解的对象

可学习性退化的可识别诊断及恢复机制，并量化恢复对旧功能和在线收益的代价。

### 信息与数据权限

probe数据对各条件一致且不向主在线学习器提供额外世界经验；重置参数、优化器和环境是不同干预。

$$
\mathcal P_K(\theta;\mathcal D)=L_{\mathcal D}(\theta)-L_{\mathcal D}(U^K(\theta;\mathcal D))
$$

$\theta$ 为probe起点参数，$U$ 指定优化器与训练序列，$K$ 为更新预算，$L_{\mathcal D}$ 为固定独立且匹配的probe损失。比较改进量需控制初始损失；此诊断不是在线回报目标，机制还要回到真实控制评价。

### 成立条件与解的含义

- 容量、目标难度、数据、优化器与统计更新匹配，初始误差或可比较误差区间受控。
- dormant、特征秩和梯度范数是诊断代理；低活动不证明单元对所有未来状态无用。

判断准则：匹配probe上测固定预算误差曲线及aged/fresh差异；选择性重置检查精确输出扰动、优化器清理和年龄；联合报告旧能力损失与在线恢复。

### 适用边界

- dormant比例下降不自动证明在线收益增加。
- 部分网络重置不等于新的所有坐标都拥有全新优化器语义。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)：重新获得新学习能力可能删除旧贡献，需要保留与适应两个评价。

- 组合不同学习问题 · [深度价值学习](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)：DQN共享表示与自举可带来训练老化，但低回报不单独确诊可塑性。

- 组合不同学习问题 · [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)：matched probe与单独重置优化器等干预提供机制识别，而非只看相关指标。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

相同标称步长可因梯度通路、尺度、曲率或优化器历史产生不同学习速度。

### 本章的核心思路

先用匹配数据隔离可学性，再针对被识别的通路或几何故障干预并测保留代价。

1. [用匹配probe确定退化](#lesson-setting)：因为在线回报还混入探索与新任务难度，固定新数据和优化预算比较aged/fresh改进。

2. [恢复缺失的可用特征](#lesson-cbp)：因为低活动/低效用单元可能阻断梯度，ReDo/CBP分别按活动或效用选替换对象，并处理输出扰动和成熟期。

3. [检验权重尺度与有效步长](#lesson-normalization)：因为单元仍活动时也会学慢，NaP在相应归一化结构中控制权重尺度，使有效步长可解释。

4. [约束参考状态的预测干扰](#lesson-churn)：因为一处梯度会改变其他输入，C-CHAIN以近期冻结函数限制跨状态扰动；过强约束也会阻碍必要适应。

结论与条件：单隐藏层选择性替换的输出差和理想尺度不变层的SGD尺度关系可代数检验；不构成普遍可塑性恢复或无遗忘定理。

### 相关方法改变了什么

- ReDo/CBP：按不同评分检测并替换特征，需测稀有状态覆盖和成熟期。

- NaP/正则：改变更新几何及有效学习尺度，不等于单元替换。

- 优化器重置/参数重置：分别干预历史尺度与表示，matched对照可定位原因而代价不同。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 梯度链式法则

网络输出对某个权重的梯度由下游权重、激活导数与输入相乘；任一环节接近零都可能使参数难以变化。

### ReLU

$\operatorname{ReLU}(z)=\max(0,z)$；$z<0$ 时导数为零。一个单元在当前数据上不激活，并不意味着它在所有未来状态上都无用。

### 优化器状态

Adam 保存一阶矩、二阶矩及时间计数；只重置参数而不处理这些状态，可能让新参数继承旧方向和尺度。

<a id="lesson-setting"></a>

## 1 · 可塑性的操作性定义

设两个同样结构的网络接收到完全相同的新目标和训练数据：一个已经训练很久，另一个刚初始化。若前者在相同学习预算下改进更慢，便有可塑性下降的证据。反之，仅看到在线 reward 下降，还不能排除探索不足、环境更难、价值估计错误或任务本身无解。

$$
\mathcal P_K(\theta;\mathcal D)=L_{\mathcal D}(\theta)-L_{\mathcal D}(U^K(\theta;\mathcal D))
$$

这是一种可操作的 $K$ 步改进量：$U$ 指定优化器、步长、训练序列和统计更新。评价采用独立但匹配的 probe 数据，并控制初始损失；仅比较不同起点的绝对改进量会混入任务难度差异。

完整 aged/fresh 对照至少固定网络容量、目标难度、数据顺序、更新数、优化器和归一化。为了拆解原因，还应增加“只重置优化器”“只重置部分权重”“保留 replay 但重置网络”等条件。若 fresh 用额外探索或外部 task reset，就不再是同一学习能力测试。

| 诊断量 | 能提示什么 | 不能单独证明什么 |
| --- | --- | --- |
| 新目标上的固定预算 loss reduction | 局部可学习性 | 在线探索是否成功 |
| Dormant 比例与梯度范数 | 激活或梯度通路变化 | 全部 plasticity loss 都由死 ReLU 导致 |
| 特征矩阵的有效秩 | 表征多样性退化 | 秩越大一定回报越高 |
| Hessian/曲率代理、权重范数 | 优化几何与尺度变差 | 一个相关曲线就是因果机制 |

CRL 中即使环境外部不变，策略、bootstrap target 与所访问状态也持续变化。因此可塑性问题不要求人为切任务；但人为目标切换能帮助在小实验里隔离机制。

<a id="lesson-derive"></a>

## 2 · 隐藏单元的梯度通路

$$
\begin{aligned}f(x)&=\sum_i v_i h_i(x),\qquad h_i(x)=\max(0,u_i^\top x),\\L&=\tfrac12(f(x)-y)^2,\\\nabla_{u_i}L&=(f-y)v_i\mathbf1[u_i^\top x>0]x,\\\partial_{v_i}L&=(f-y)h_i(x).\end{aligned}
$$

若当前数据上 $u_i^\top x<0$，这个单元的输入权重与输出权重梯度均为零。参数容量与能够通过梯度使用的容量因而不同。

这个例子刻意简单：真实可塑性下降也可能出现于没有饱和单元的网络，与目标尺度、曲率、优化器状态和特征相关性有关。因而 ReDo/CBP 是针对某类故障的机制，不是所有网络老化现象的定义。

$$
s_i=\frac{\mathbb E_{x\sim D}[|h_i(x)|]}{\frac1H\sum_{j=1}^H\mathbb E_{x\sim D}[|h_j(x)|]}
$$

ReDo 用层内相对活动度辨认 dormant 单元，H 为该层单元数，D 是检测用数据分布。若整层全零，分母为零，实现必须明确处理，而不是让 NaN 静默决定选择。

一个活动度低的单元可能在稀有但重要的状态上有用。D 的覆盖、检测频率和阈值决定了“无用”的含义。把检测 batch 换成当前一个观测，会极大改变噪声和选择；这不是不需要评估的实现小节。

<a id="lesson-redo"></a>

## 3 · ReDo：检测休眠，再恢复可用的梯度通路

**算法：算法伪代码**

1. 按既定周期收集检测 batch；前向记录各层隐藏激活
1. 对每层计算 `mean(abs(h_i))` 和相对 score $s_i$
1. 选择 $s_i$ ≤阈值 的单元；阈值与检测频率是超参数
1. 对选中单元：
  1. 输入权重按原初始化分布重新采样，输入 bias 重新初始化
  1. 对应输出权重置零，避免新随机特征立即注入任意输出
  1. 清理相关优化器状态；核对 target network 与参数同步规则
1. 恢复正常 RL 更新；记录替换数量与替换前后的预测变化

“输出权重置零”只保证新随机特征在替换瞬间不产生新贡献，不保证旧输出完全保留。因为旧单元的贡献也被删掉了；只有旧贡献恰好为零时，替换瞬间才严格不变。低活动度通常减小扰动，但不是函数等价变换。

$$
f_{\rm after}(x)-f_{\rm before}(x)=-\sum_{i\in\mathcal R}v_i h_i(x)
$$

假定其余参数和 bias 不变且新输出权重设为零，R 是替换集合。这是精确的单隐藏层输出差，可用于检查重置实现有没有删除未选中的单元。

下一个样本上，若新激活 $h_{\rm new}\ne0$ 且误差非零，即使输出权重为零，输出权重仍有梯度。输出连接建立后，输入权重才重新获得下游梯度。新生单元需要学习时间，频繁再次替换会打断这一过程。

<a id="lesson-cbp"></a>

## 4 · CBP：特征的持续生成与检验

Continual Backpropagation 在常规反向传播之外，持续维护候选特征的效用与年龄，优先替换已经成熟却低效用的单元。效用不必等同于活动度：高活动但没有下游用途的特征，也可能值得替换。作者实现提供 contribution、zero-contribution、adaptable-contribution 等选项；必须写清实际使用哪一种。

$$
\begin{aligned}c_{i,t}&=\mathbb E_B[|h_i|]\,\operatorname{mean}_j|W_{{\rm out},ji}|,\\u_{i,t}&=\beta u_{i,t-1}+(1-\beta)c_{i,t},\\\widehat u_{i,t}&=u_{i,t}/(1-\beta^{a_i}).\end{aligned}
$$

这是 contribution 效用的一个明确版本；ai 是自出生以来的更新次数。它同时考虑激活和下游权重，并用年龄进行 EMA 偏差修正。不要把此式泛称为所有 CBP 实验的唯一评分。

$$
c_{i,t}^{\rm adapt}=\frac{\mathbb E_B[|h_i-\widehat\mu_i|]\,\operatorname{mean}_j|W_{{\rm out},ji}|}{\operatorname{mean}_k|W_{{\rm in},ik}|+\varepsilon}
$$

adaptable-contribution 风格还考虑非恒定贡献及输入权重尺度。本式显式加入 ε 以定义零分母；对应原代码/实验的均值、范数及 ε 选择应按实际配置复核。

**算法：算法伪代码**

1. 1. 用当前样本完成普通梯度更新；保留检测所需激活
1. 2. 单元年龄 ai += 1；更新激活均值与效用 EMA
1. 3. 只让 ai>成熟阈值 的单元进入候选集合 E
1. 4. replacement_credit += replacement_rate × |E|
1. 5. 本步替换 floor(replacement_credit) 个候选，扣掉整数部分
1. 6. 在候选内按 bias-corrected utility 从小到大选择
1. 7. 重采输入参数，输出连接归零；必要的 bias 补偿按指定变体处理
1. 8. 清零新生单元的 age、utility、activation statistics
1. 9. 清理相关 optimizer moments / coordinate step；继续反向传播

步骤 4–5 是累计预算变体，也可用概率方式实现小于一个单元的期望替换率。成熟期保护新单元的学习机会。替换率为零时，算法退化成不替换的原学习器。

Adam 的一阶矩、二阶矩以及时间计数都与参数年龄有关。某些作者实现使用可逐坐标清零的 AdamGnT；普通框架只给整个张量一个 step 时，局部重置无法等同于所有坐标的全新 Adam。复现时应保留原优化器语义，或把替代策略作为一个新的实验条件。

<a id="lesson-normalization"></a>

## 5 · NaP：归一化后的有效学习率

单元保持活动时，网络仍可能逐渐学得更慢。一个原因是归一化改变了参数尺度与函数变化的关系。考虑无偏置、忽略数值稳定项的理想尺度不变层：把进入归一化的权重乘以正数，输出保持不变。设损失也具有这种不变性，用链式法则可得：

$$
L(cw)=L(w),\quad \nabla L(cw)=c^{-1}\nabla L(w),\quad c>0.\qquad \eta_{\rm eff}^{\rm SGD}=\frac{\eta}{\|w\|^2}.
$$

令单位方向为 $w/\|w\|$，一次普通 SGD 的方向变化与权重范数的平方成反比。这里的有效学习率公式针对普通梯度步；归一化梯度的对应尺度为 $\eta/\|w\|$，不能直接把两者等同于所有 Adam 更新。

因此，同样的标称步长下，权重范数从 2 增长到 4，理想 SGD 的有效步长变为原来的四分之一。这个效应不会被“休眠单元比例仍然很低”排除。Lyle 等在 NeurIPS 2024 的 Normalize-and-Project（NaP）将非线性之前的归一化，与定期恢复每层初始权重范数结合，使学习率调度显式化。

$$
\widetilde W_\ell\leftarrow\operatorname{OptimizerStep}(W_\ell),\qquad W_\ell^+\leftarrow\frac{\|W_{\ell,0}\|_F}{\|\widetilde W_\ell\|_F}\widetilde W_\ell.
$$

式中假定投影前范数非零。投影保持权重方向，仅恢复范数。归一化的可学习缩放与偏置是另一组参数，需要按具体变体单独约束；它们不自动满足这里的尺度不变条件。

归一化还有第二个作用：均值和方差使单元之间的梯度耦合，位于 ReLU 前的归一化可以给部分不激活的预激活量传递其他单元的梯度。这与直接替换单元是不同机制。有效学习率保持恒定也并非总是最优：价值估计需要一定程度的收敛，论文中的部分 Rainbow 实验仍需要显式衰减。研究问题因而是适应速度与估计噪声的调度，而非无条件维持最大更新幅度。

<a id="lesson-churn"></a>

## 6 · C-CHAIN：控制更新在参考状态上的影响

共享参数让一次局部学习同时改变其他输入的预测。称这种改变为 prediction churn。设当前训练输入为 $x$，参考输入为 $\bar x$，标量网络为 $f_\theta$，一次梯度步为 $\Delta\theta=-\eta\nabla_\theta L_x$。对参考预测做一阶展开：

$$
\begin{aligned}\Delta f(\bar x)&\approx\nabla_\theta f_\theta(\bar x)^\top\Delta\theta\\&=-\eta\underbrace{\nabla_\theta f_\theta(\bar x)^\top\nabla_\theta f_\theta(x)}_{K_\theta(\bar x,x)}\frac{\partial L_x}{\partial f_\theta(x)}.\end{aligned}
$$

$K_\theta$ 是两个输入的梯度内积，即经验神经切线核的一个元素。它可以为正或负；在 $x$ 上降低损失并不意味着在 $\bar x$ 上也降低损失。该一阶式在更新较小时解释局部干扰，并非有限大步长下的精确等式。

ICML 2025 的 C-CHAIN 在参考状态上约束这种变化。其值函数形式可写成下面的函数空间正则：当前网络在参考状态的全部动作价值，靠近近期冻结网络的输出。它与 EWC 的参数距离不同，也与长期保存旧任务教师不同，主要控制训练过程中的预测变化。

$$
L(\theta)=L_{\rm TD}(\theta)+\lambda\,\mathbb E_{s\sim D_{\rm ref}}\|Q_\theta(s,\cdot)-\operatorname{sg}[Q_{\theta^-}(s,\cdot)]\|_2^2.
$$

$\theta^-$ 是近期参数快照，$\operatorname{sg}$ 表示停止梯度。作者 MinAtar 实现独立采样 TD batch 与参考 batch，因此样本可能重叠；它还维护有界的历史网络，并根据损失尺度调节正则系数。

**算法：算法伪代码**

1. 从经验库采样 TD batch，构造停止梯度的 bootstrap target
1. 独立采样参考状态，选择一个近期冻结网络 $Q_{\theta^-}$
1. 计算 TD 损失与参考状态上的全动作价值差
1. 对两项加权和反向传播并更新当前网络
1. 按配置保存近期网络快照、淘汰超出窗口的快照
1. 用运行中的损失统计调整正则强度，记录预测改变与新目标学习速度

实现不需要显式构造完整神经切线核；核只用于解释干扰机制。过强的约束会妨碍必要的价值修正，参考分布遗漏的状态也不受保护。参考 batch、历史网络和额外前向传播均有成本，所以该方法的经验重放条件与严格流式学习不同。一个直接实验是匹配更新数与内存预算，在固定 probe 上比较替换、范数投影和短期函数约束分别改善哪类故障。

<a id="lesson-example"></a>

## 7 · 手算：替换扰动与学习恢复

网络只有两个单元，x=1，输入权重 u=[−1,1]、输出权重 v=[0.5,2]。激活 h=[0,1]，输出 f=2；活动度的层均值为 0.5，scores=[0,2]，阈值 0.1 只选第一个。把它的输入改成 0.5、输出改成 0，输出仍为 2，因为旧贡献为零。

若误选第二个单元，它的旧贡献为 2，输出会从 2 降为 0。这个变化不是数值误差，而是更新定义直接造成的。对神经网络实验应记录预测扰动、策略 KL 和短期回报下降，不能只记录 dormant 数量减少。

再用单单元学习 y=1。aged 网络 u=−1、v=1，在 x=1 上所有相关梯度为零，平方损失一直为 0.5。fresh 网络 u=0.5、v=0，第一步 α=0.1 更新输出权重 v→0.05，输入 u 暂时不变；此后输出通路非零，输入也能学习。两者用相同 20 次样本训练，本页程序验证后者损失下降。这个人为构造只证明死 ReLU 机制，不代表所有 aged 网络都会失效。

成熟期例子：三个单元效用 [0,0.1,1]、年龄 [0,21,21]，阈值 20，rate=0.25。最年轻的单元虽效用最低也不进入候选；两个候选每步积累 0.5 替换名额，第二步才替换效用 0.1 的成熟单元。

<a id="lesson-code"></a>

## 8 · 替换器实现与诊断实验

单输入/单输出隐藏层的机制实现；moment 数组分别表示被替换输入/输出权重的状态。完整网络还需处理 bias、卷积轴和 target 参数。

```python
def redo_indices(mean_abs_activations, threshold=0.1):
    """Relative activity score; an all-zero layer is entirely dormant."""
    if not mean_abs_activations:
        return []
    denominator = sum(mean_abs_activations) / len(mean_abs_activations)
    if denominator == 0:
        return list(range(len(mean_abs_activations)))
    return [i for i, a in enumerate(mean_abs_activations)
            if a / denominator <= threshold]


def cbp_candidates(utilities, ages, maturity, rate, credit=0.0):
    """Accumulated-budget variant: protect young units, rank eligible utility."""
    if not 0 <= rate <= 1:
        raise ValueError("replacement rate outside [0,1]")
    eligible = [i for i, age in enumerate(ages) if age > maturity]
    credit += rate * len(eligible)
    count = min(len(eligible), int(credit))
    selected = sorted(eligible, key=lambda i: (utilities[i], i))[:count]
    return selected, credit - count


def cbp_contribution_utility(old_utility, mean_absolute_activation,
                             mean_absolute_outgoing, age, decay=0.99):
    """EMA plus age correction for the explicitly chosen contribution variant."""
    if age < 1 or not 0 <= decay < 1:
        raise ValueError("increment age before updating utility")
    instantaneous = mean_absolute_activation * mean_absolute_outgoing
    updated = decay * old_utility + (1. - decay) * instantaneous
    return updated, updated / (1. - decay**age)


def recycle_one_unit(incoming, outgoing, first_moment, second_moment,
                     index, new_incoming):
    """Scalar-input, scalar-output hidden unit; reset its incoming/outgoing state.
    first/second_moment[i] each contain [incoming_parameter, outgoing_parameter].
    """
    incoming[index] = new_incoming
    outgoing[index] = 0.0
    first_moment[index] = [0.0, 0.0]
    second_moment[index] = [0.0, 0.0]


def relu_prediction(incoming, outgoing, x=1.0):
    return sum(v * max(0., u*x) for u, v in zip(incoming, outgoing))


def train_relu_toy(incoming, outgoing, target=1., steps=20, alpha=0.1):
    incoming, outgoing = incoming[:], outgoing[:]
    for _ in range(steps):
        error = target - relu_prediction(incoming, outgoing)
        old_u, old_v = incoming[:], outgoing[:]
        for i in range(len(incoming)):
            incoming[i] += alpha * error * old_v[i] * (old_u[i] > 0)
            outgoing[i] += alpha * error * max(0., old_u[i])
    return 0.5 * (target - relu_prediction(incoming, outgoing)) ** 2


def plasticity_demo():
    u, v = [-1., 1.], [0.5, 2.]
    before = relu_prediction(u, v)
    moments1, moments2 = [[3., 4.], [3., 4.]], [[5., 6.], [5., 6.]]
    recycle_one_unit(u, v, moments1, moments2, 0, 0.5)
    print("plasticity", {"redo_selected": redo_indices([0., 2.]),
          "prediction_before_after": (before, relu_prediction(u, v)),
          "cleared_optimizer_state": moments1[0] + moments2[0],
          "matched_probe_loss_aged": train_relu_toy([-1.], [1.]),
          "matched_probe_loss_fresh": train_relu_toy([0.5], [0.]),
          "scope": "synthetic dead-ReLU mechanism, not an empirical LoP diagnosis"})
```

输出应包括替换前后 (2,2)、清零的 optimizer state，以及 aged/fresh probe loss

```sh
python lifelong_algorithms_lab.py plasticity
python lifelong_algorithms_lab.py test
```

示例覆盖全零层、相对活动度、成熟期、分数替换预算与替换扰动。fresh 网络是固定数据诊断的对照条件。作者 ReDo 代码中的 weight_recyclers.py、CBP 的 lop/algos/gnt.py 分别给出完整框架中的选择与生成流程；C-CHAIN 的 MinAtar agent 展示参考采样和快照队列。本页小实验实现替换机制，不包含这些论文的完整深度 RL 训练。

- 先做固定数据 probe，分离探索与学习能力；再返回在线 RL 看是否仍有实际收益。
- 匹配参数量与 FLOPs，特别是 CReLU 改变特征维度、模块扩容和 injection 增加容量的情况。
- 与随机等量替换、只清 optimizer state、全网络重置和只改激活函数对照，辨认选择规则是否真的有用。
- 同时报告新任务适应、旧技能保留、替换后短期损失和每步额外成本，避免只选择一种有利指标。

<a id="lesson-branches"></a>

## 9 · 干预位置与选择依据

$$
\operatorname{CReLU}(z)=[\max(0,z),\max(0,-z)]
$$

对 z≠0 至少一支对 z 有非零导数，但最终输出梯度仍取决于下游权重与损失。它不是整个网络梯度永远不为零的保证，并且会改变宽度/参数预算。

| 方法线 | 干预对象 | 应排除的替代解释 |
| --- | --- | --- |
| ReDo | 当前分布下低活动单元周期性回收 | 收益是否仅来自额外随机扰动？ |
| CBP / generate-and-test | 低效用成熟特征的持续替换 | 是否只需随机替换？成熟期与效用是否都重要？ |
| CReLU / 激活改造 | 激活及梯度通路 | 增加的特征维度是否解释提升？ |
| LayerNorm / 参数正则 | 尺度、曲率与优化几何 | 是否只是更好的超参数而非长期机制？ |
| Primacy-bias resets | 周期重置一部分网络，保留经验 | 收益可能依赖 replay 重建能力，不自动适于无 replay 流式学习 |
| Plasticity injection | 接入可训练新分支并控制初始输出 | 额外参数和 optimizer state 如何计入固定预算？ |

最重要的研究逻辑是先诊断，再选干预。若 aged 与 fresh 在同一数据上学得同样快，但在线 aged 不再到达新状态，应转向探索或 agent state；若输出尺度越来越大而 dormant 比例不变，优先检查优化几何；若旧技能特别珍贵，则回收规则要加入保留约束。

<a id="research-parseval-geometry"></a>

## 研究专题 A · Parseval：持续维护尺度与方向几何

ReDo/CBP 改动单元，Parseval regularization 则让仍在使用的矩阵保持较好的几何条件。它不等待某个单元完全休眠才干预，而是持续约束不同输出方向的相关性与尺度。其机制应与 NaP 的有效学习率和 C-CHAIN 的函数变化分别检验。

$$
\Omega(W)=\lambda\|WW^\top-sI\|_F^2,\qquad\nabla_W\Omega=4\lambda(WW^\top-sI)W
$$

W 为输出维度×输入维度。对平方 Frobenius 范数微分得到该梯度；s>0。它约束行内积与范数，不把坐标锁到旧值。

$$
\|WW^\top-sI\|_F^2=\sum_i(\|w_i\|^2-s)^2+\sum_{i\ne j}\langle w_i,w_j\rangle^2
$$

范数与方向是两个独立作用。weight decay 不能保证方向分开；每行归一化也不能消除两行重合。

两行均为 $(1,0)$、$s=1$，范数已正确，但正则仍为 $2\lambda$；改成 $(1,0)$ 与 $(0,1)$ 后为零。若 W 有 3 行只有 2 列，$\operatorname{rank}(WW^\top)\le2$，不可能等于 $sI_3$。需改变约束侧、分组或层宽，不能把不可实现的零残差当目标。

非零奇异值靠近同一尺度不等于整个非线性网络等距。激活导数、输入分布、残差与输出层仍决定完整 Jacobian。作者保留输出层自由度，并检验额外尺度与层等容量补偿，说明可学习性与函数表达之间存在取舍。

**算法：目标级骨架；并非独立参数 reset**

1. 按基础 RL 方法构造 actor/critic 代理损失与停止梯度 target
1. 对选定隐藏层计算 `W @ W.T`，并加入 $λ\|WW^T-sI\|^2$
1. 同一次反传更新任务与正则项；保留 optimizer 状态
1. 记录任务误差、Gram 残差、奇异值和新目标 probe

| 消融 | 隔离的因素 |
| --- | --- |
| 仅正交初始化 vs 持续正则 | 有利几何是否在训练中流失。 |
| 仅范数、仅非对角项、完整项 | 尺度与方向的不同贡献。 |
| 固定宽度 vs 容量补偿 | 改善是否依赖新增表达能力。 |

可从 wechu/parseval_reg 的 agent.py 对应正则，main.py 对应任务序列。验证时匹配交互、梯度次数、参数量和调参预算，并比较 aged/fresh 在同一新目标数据上的拟合速度及旧功能。稳定秩提高本身不证明任意未来目标可学习；几何改善而真实控制无改善也应保留。

<a id="lesson-check"></a>

## 10 · 自测与研究练习

- 为什么 low-utility 不是“永远无用”？效用根据某个近期分布估计；稀有历史状态和未来任务可能改变结论。
- 为什么新输出权重置零后仍能成长？第一步输出权重有非零激活梯度，后续输入权重才得到非零下游信号。
- 清零全张量 Adam step 可以当作局部重置吗？不可以，它也改变未替换坐标的偏差修正；必须明确这项额外干预。
- ReDo 能否证明可塑性下降就是 dead neurons？不能。存在其他机制，也存在休眠是结果而非根因的可能。

动手题：在 toy 网络中让旧单元的小贡献从 0 缓慢变到 0.1，逐次核对替换前后输出差正好等于删掉的贡献。再把随机 reset 与 score-based reset 设置为相同替换次数，用 matched probe 比较改善，避免将替换率不同误判为选择规则更好。

## 本章的实验设计

使用新任务学习速度检验可塑性。保留原参数与重新初始化的对照必须具有相同的新增数据预算。

设定：让 learner 具有不同训练年龄，再在相同新增数据下学习新信号。用 fresh 模型、优化器重置与不同替换机制拆解原因。

- 新信号对普通强基线确实可学。
- 替换包含按算法约定处理入边、出边和 optimizer 状态。
- 诊断探针引入的任务变化与原主轨迹分开。

对照：完全 fresh、仅重置优化器；相同替换数量的随机替换；相同新增数据、容量与总计算

记录：新任务达到标准的样本量；aged/fresh 学习曲线与失败；激活、梯度、秩及替换成本

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-continual)

## 学习与研究衔接

不遗忘不意味着还能学习。应在相同新数据预算下测试老网络、新网络与局部重置。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-plasticity) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=plasticity) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=plasticity)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Streaming Deep Reinforcement Learning Finally Works](https://yingwen.io/zh/continual-rl/research/#recent-stream-x)
- [Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-upgd-utility-protection)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [The Dormant Neuron Phenomenon in Deep Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-redo-dormant-neurons)
- [Understanding Plasticity in Neural Networks](https://yingwen.io/zh/continual-rl/research/#recent-understanding-plasticity)
- [Loss of plasticity in deep continual learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-backpropagation)
- [Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn](https://yingwen.io/zh/continual-rl/research/#recent-c-chain-churn)
- [Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline](https://yingwen.io/zh/continual-rl/research/#recent-reset-and-distill)
- [Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-fame-fast-meta-learners)
- [Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-vla-simple-recipe)
- [Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-upgd-utility-protection)
- [Parseval Regularization for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-parseval-continual-geometry)
- [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-trac-online-regularization)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Step-size Optimization for Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-step-size-optimization)
- [Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-fame-fast-meta-learners)
- [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-trac-online-regularization)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [The Cell Must Go On: Agar.io for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-agarcl)
- [Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-vla-simple-recipe)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文提供定义和条件，架构讲座提出模块组织，算法论文检验特定机制。完整系统还要明确智能体、外部设计者和世界各自承担的工作；组件成立不自动意味着组合后的长期收益成立。

- [Plasticity as the Mirror of Empowerment](https://yingwen.io/zh/continual-rl/research/#recent-plasticity-mirror-empowerment)

### The Dormant Neuron Phenomenon in Deep Reinforcement Learning

Ghada Sokar, Rishabh Agarwal, Pablo Samuel Castro, Utku Evci

ICML 2023 · 2023 · 支持方法与理论

#### 研究问题

网络参数数量没有变，为什么越来越多隐藏单元不再对输出产生有效贡献？

#### 关键机制

ReDo 用相对激活量识别低活跃单元，重新初始化其输入连接，并处理输出连接，使被回收单元可以重新参与学习。它针对的是可用表示容量，而不是直接惩罚旧任务表现变化。

#### 证据

论文记录深度 RL 中的休眠单元现象，并比较回收机制对多个任务学习的影响。实现进入作者所在团队的 Dopamine 代码库。

#### 条件与限制

低激活只是可塑性问题的一种诊断，不能覆盖曲率变化、优化器状态和负迁移。回收也可能损坏低频但重要的旧知识，需要与保留指标共同评价。

#### 阅读与实验

同时记录休眠比例、新目标拟合速度与旧任务冻结表现。三者发生不同方向变化时，不要用单个表示指标替代整个持续学习结论。

#### 原文与相关入口

- [ICML 2023 原文](https://proceedings.mlr.press/v202/sokar23a.html)：休眠定义、回收规则与实验。
- [Dopamine ReDo 实现](https://github.com/google/dopamine/tree/master/dopamine/labs/redo)：作者团队公开代码中的 ReDo 模块。

#### 作者代码

[论文作者团队发布的实现，不是本教材的简化版本。](https://github.com/google/dopamine/tree/master/dopamine/labs/redo)

Dopamine 中的 ReDo 神经元回收与实验实现。

### Understanding Plasticity in Neural Networks

Clare Lyle, Zeyu Zheng, Evgenii Nikishin, Bernardo Avila Pires, Razvan Pascanu, Will Dabney

ICML 2023 · 2023 · 支持方法与理论

#### 研究问题

学习变慢一定意味着网络已饱和或特征秩下降吗？

#### 关键机制

论文通过新目标拟合实验研究可塑性，并分析优化几何与曲率的影响。某些表示统计与学习能力下降会同时出现，却不是所有设置中的充分解释。评价对象从“网络看起来是否健康”转向“在受控更新预算内还能学会什么”。

#### 证据

受控探针与 RL 实验展示了不同机制之间的区别，并检验网络设计和优化过程的作用。它为可塑性研究提供诊断方式，而不是单一通用修复算法。

#### 条件与限制

探针目标、优化器和步数会改变测得的可塑性。相关性不等于所有控制任务中的因果机制；探针训练也不能写回被评价的在线智能体。

#### 阅读与实验

复制同一个检查点，在副本上拟合两类新目标。保持训练预算一致，并报告探针过程与真实环境回报之间的区别。

#### 原文与相关入口

- [ICML 2023 原文](https://proceedings.mlr.press/v202/lyle23b.html)：可塑性探针、优化几何与诊断边界。

### Loss of plasticity in deep continual learning

Shibhansh Dohare, J. Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A. Rupam Mahmood, Richard S. Sutton

Nature · 2024 · 直接研究持续学习

#### 研究问题

一个长期训练的网络如何保留继续形成新特征的能力？

#### 关键机制

Continual Backpropagation 在梯度学习之外持续生成并测试特征。它估计单元的效用和成熟度，少量替换低效用的成熟单元，并协调新单元的输入、输出和相关状态。维护新的可学习方向是一个持续过程，而不是等到任务切换后整体重启。

#### 证据

论文在长序列监督学习与强化学习问题中展示可塑性损失，并检验特征替换的作用。作者仓库包含 generate-and-test 与优化器状态处理。

#### 条件与限制

有限序列上的学习保持不保证无限生命中的任意适应。替换率、效用定义与成熟度条件仍需选择；新任务学习速度和旧能力保留必须分开测量。

#### 阅读与实验

逐项消融“成熟度筛选”“效用筛选”“随机替换”。比较相同替换预算，检验收益究竟来自定向回收还是一般参数扰动。

#### 原文与相关入口

- [Nature 原文](https://doi.org/10.1038/s41586-024-07711-7)：长期可塑性实验与 continual backpropagation。
- [作者代码](https://github.com/shibhansh/loss-of-plasticity)：关注 lop/algos/gnt.py 及替换时的优化器状态。

#### 作者代码

[论文作者公开的实验实现。](https://github.com/shibhansh/loss-of-plasticity)

论文任务、持续反向传播与 generate-and-test。

### Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn

Hongyao Tang, Johan Obando-Ceron, Pablo Samuel Castro, Aaron Courville, Glen Berseth

ICML 2025 · 2025 · 直接研究持续学习

#### 研究问题

一次局部更新为什么会在其他输入上引发大幅预测变化，并损害后续学习？

#### 关键机制

C-CHAIN 抑制相对于近期参考网络的函数输出变化，降低一次更新在其他样本上造成的 churn。论文把该现象与经验神经切线核及学习动力学联系起来。正则化对象是函数变化，不是直接把所有参数锁在旧值附近。

#### 证据

作者在持续 Gym Control、ProcGen、DMC 和 MinAtar 序列中比较，并提供对应环境和算法代码。

#### 条件与限制

近期函数稳定性不等于长期任务知识保留；参考样本和参考网络也占资源。若环境突然发生真实变化，过强抑制输出变化可能延迟必要适应。

#### 阅读与实验

将 churn 按旧分布、新分布分别计算，并同时画适应速度。这样才能区分“减少无关干扰”和“阻止有用改变”。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/tang25g.html)：机制、理论分析与持续实验。
- [作者代码](https://github.com/bluecontra/C-CHAIN)：四类持续环境的基线和 C-CHAIN 对照实现。

#### 作者代码

[作者仓库，README 说明依赖的 TRAC、CleanRL 与 MinAtar 基础实现。](https://github.com/bluecontra/C-CHAIN)

持续控制环境与 C-CHAIN 对照实验。

### Streaming Deep Reinforcement Learning Finally Works

Mohamed Elsayed, Elena Sorina Lupu, Gautham Vasan, A. Rupam Mahmood

arXiv（2024 首稿；2026 v3） · 2026 · 直接研究持续学习

#### 研究问题

不保存经验重放、不使用目标网络或训练批次时，深度 RL 能否逐步稳定学习？

#### 关键机制

Stream-X 把信号归一化、表示初始化、资格迹和受控更新尺度组织为一组流式学习方法。各组件处理的是不同问题：奖励尺度、激活与梯度传播、延迟信用，以及一次更新造成的输出变化。去掉重放并不意味着这些问题会自动消失。

#### 证据

2026 年第三版扩展到 Atari、控制与机器人等实验，并包含持续变化设置。论文和代码经历过版本变化，比较结果时需要同时标明论文版本和算法实现。

#### 条件与限制

广泛任务上的流式可行性不等于所有非平稳问题都已解决。不能把旧版较弱 Adam 基线推广成对所有流式 Adam 方法的否定；后续研究专门检验了这一点。代码许可证也应独立于本教材许可证处理。

#### 阅读与实验

按归一化、资格迹、更新控制分别做消融，并保持每步算力一致。先验证严格一次使用经验，再研究长期变化，而不是仅把小批量大小改成一。

#### 原文与相关入口

- [2026 年第三版论文](https://arxiv.org/abs/2410.14606v3)：作者名单、任务范围与算法版本以该版为准。
- [作者代码版本](https://github.com/mohmdelsayed/streaming-drl/tree/9326fc3e23a401f28087ae2e41b635888740586b)：固定实现版本，避免把不同年份的更新规则混在一起。

#### 作者代码

[原作者仓库的固定版本。](https://github.com/mohmdelsayed/streaming-drl/tree/9326fc3e23a401f28087ae2e41b635888740586b)

Stream-X 算法、变换、优化器及实验；使用前阅读仓库许可证。

### Step-size Optimization for Continual Learning

Thomas Degris, Khurram Javed, Arsalan Sharifnassab, Yuxin Liu, Richard S. Sutton

arXiv 预印本 · 2024 · 支持方法与理论

#### 研究问题

误差变大时，应该减小步长过滤噪声，还是增大步长追踪真实变化？

#### 关键机制

论文区分梯度归一化与步长优化。IDBD 类方法以 $\alpha_i=\exp(\beta_i)$ 保证步长为正，并用权重对过去步长的敏感度估计改变 $\beta_i$ 是否有利。持续学习中，静止的无关方向适合很小步长，而持续变化的有用方向需要保留追踪能力。

#### 证据

作者用权重翻转和带噪追踪等线性学习问题比较机制，显示相似的误差幅度可以要求相反的步长反应。

#### 条件与限制

这些可分析任务不是深度控制上的普适优越性证据。元步长、近似敏感度与输入尺度仍会影响结果；步长自适应并没有消除全部外部设计参数。

#### 阅读与实验

分别增加观测噪声和目标漂移速度，检查步长是否采取不同反应。若只记录平均误差，就看不到噪声过滤与追踪之间的区别。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2401.17401)：步长优化与归一化的对照实验。

### Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning

Ke Sun, Hongming Zhang, Jun Jin, Chao Gao, Xi Chen, Wulong Liu, Linglong Kong

ICLR 2026 · 2026 · 直接研究持续学习

#### 研究问题

快速学习新任务和整合旧知识，能否由不同学习器承担并以明确目标连接？

#### 关键机制

FAME 的快速学习器适应当前任务，元学习器整合此前知识。论文按旧策略的重要访问分布度量价值或策略变化，再据此构造减少遗忘的整合目标。自适应预热决定如何利用旧知识初始化或约束早期行为，以减少负迁移。

#### 证据

论文分析价值型和策略型版本，并在像素与连续控制任务序列中比较。作者提供官方实现，可追踪快速适应与知识整合两个阶段。

#### 条件与限制

设定要求相同状态与动作空间、已知任务边界以及额外整合计算。这里的 meta learner 主要是知识整合模块，不应因名称就当作通过长期回报反向求导的在线元梯度算法。脑机制类比也不是神经科学实验证据。

#### 阅读与实验

分别报告新任务前向迁移、旧任务保留和两个学习阶段的计算量。改变任务相似性，检验自适应预热是否确实避免有害旧知识。

#### 原文与相关入口

- [ICLR 2026 原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2230ffcd5da10015ce0c6ce588fc2936-Abstract-Conference.html)：任务边界假设、遗忘度量与快慢知识机制。
- [FAME 官方实现](https://github.com/datake/FAME)：论文链接的快速学习与知识整合代码。

#### 作者代码

[论文与仓库均注明为官方实现。](https://github.com/datake/FAME)

FAME 的价值型、策略型持续学习实验。

### Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline

Hongjoon Ahn, Jinu Hyeon, Youngmin Oh, Bosun Hwang, Taesup Moon

ICLR 2025 · 2025 · 直接研究持续学习

#### 研究问题

一个网络还能拟合新目标，为什么先前训练仍可能让它在新任务上学得更慢？

#### 关键机制

论文把任务之间的负迁移与一般可塑性损失区分开。Reset & Distill 在新任务开始时重置在线 actor 和 critic，避免旧初始化阻碍学习；随后离线蒸馏当前策略与旧专家的动作分布以整合知识。适应和保留通过不同过程实现。

#### 证据

作者在控制与游戏任务中分析负迁移，并在长 MetaWorld 序列上检验该基线。原文直接提供实现地址。

#### 条件与限制

任务边界、在线网络重置、旧专家和离线蒸馏都需要资源。它不能直接当作无边界、不能重置、禁止回放的单次生命方案。

#### 阅读与实验

除了与连续微调比较，还要与同等预算的从头训练比较。若新任务表现低于从头训练，先检查负迁移，再判断是否属于单纯容量损失。

#### 原文与相关入口

- [ICLR 2025 原文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ba9e3d60610f3525717665966d86e0cd-Abstract-Conference.html)：负迁移诊断、Reset & Distill 机制与边界。
- [原文代码入口](https://github.com/hongjoon0805/Reset-Distill)：论文首页提供的作者实现。

#### 作者代码

[ICLR 正式论文首页明确链接的代码。](https://github.com/hongjoon0805/Reset-Distill)

Reset & Distill 以及任务序列实验。

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

### Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning

Jiaheng Hu, Jay Shim, Chen Tang, Yoonchang Sung, Bo Liu, Peter Stone, Roberto Martín-Martín

RLC 2026 · 2026 · 直接研究持续学习

#### 研究问题

大规模预训练的视觉—语言—动作模型，是否仍需要复杂机制才能顺序学习控制任务？

#### 关键机制

论文研究对预训练 VLA 进行顺序强化学习，并以低秩适配等相对简单的训练流程检验持续学习。预训练表示、可更新参数子空间和 RL 目标共同决定迁移与遗忘，不能只把结果归因于单一保留正则项。

#### 证据

作者在多种 VLA 与长期任务基准上比较，并提供实验代码。RLJ 的 RLC 2026 论文页与论文脚注分别给出正式入口和作者仓库。

#### 条件与限制

预训练数据和算力属于外部资源，任务与重置协议也影响难度。该结果不意味着从零训练的网络不会遗忘，更不意味着任意无边界任务流只需微调。

#### 阅读与实验

固定预训练模型，分别改变可训练参数量和任务顺序。报告预训练资源、每任务在线数据、回放或重置条件，再与传统 CRL 方法比较。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2603.11653)：VLA 持续学习设置、机制与比较。
- [RLC 2026 / RLJ 论文页](https://rlj.cs.umass.edu/2026/papers/Paper84.html)：会议原文入口；PDF 脚注链接作者代码。
- [作者实现](https://github.com/UT-Austin-RobIn/continual-vla-rl)：持续 VLA 的训练与评价代码。

#### 作者代码

[UT Austin RobIn 实验室的原论文仓库。](https://github.com/UT-Austin-RobIn/continual-vla-rl)

预训练 VLA 的顺序 RL 训练和论文评价。

### Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning

Aneesh Muppidi, Zhiyu Zhang, Heng Yang

NeurIPS 2024 · 2024 · 直接研究持续学习

#### 研究问题

未知环境变化时间和速度时，怎样在线决定参数应离参考初始化多远？

#### 关键机制

TRAC 在基础优化器外维护一组具有不同遗忘时间尺度的一维 tuner，根据梯度与参考方向的内积调整参数位移尺度。它通过数据驱动的缩放联系到正则化，而不是对未来任务回报进行长窗口元梯度反传。

#### 证据

作者在 Procgen、Atari 与 Gym Control 变化序列中比较适应与可塑性，并分析在线凸优化对该设计的启发。

#### 条件与限制

凸在线优化中的遗憾理论不等于非凸、策略依赖采样的深度 RL 收敛定理。“parameter-free”不表示没有基础学习率、初始化、时间尺度网格、warm-start 或协议选择。

#### 阅读与实验

记录 tuner 尺度、距参考点的位移、旧分布干扰与变化后适应。用相同基础优化器比较固定尺度、单时间尺度和多时间尺度。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/5b76d77e7095c6480ed827b85f0c2878-Paper-Conference.pdf)：Algorithm 1–2、正则化联系与持续实验。
- [作者论文 v3](https://arxiv.org/html/2405.16642v3)：区分凸理论、RL 经验结果与初期表现限制。

#### 作者代码

[作者项目页与仓库均明确标为官方实现。](https://github.com/ComputationalRobotics/TRAC)

trac.py、PyTorch/JAX optimizer 包与控制/视觉实验。

### Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning

Mohamed Elsayed, A. Rupam Mahmood

ICLR 2024 · 2024 · 支持方法与理论

#### 研究问题

同一网络里，哪些方向应当保护，哪些方向应当获得更强的新学习与扰动？

#### 关键机制

UPGD 用移除权重或特征的反事实损失变化定义效用，并以 Taylor 近似在线估计。平滑、缩放后的效用同时调制梯度与随机扰动，让近期高效用方向变化较小、低效用方向更活跃。

#### 证据

主体证据包括未知边界的非平稳流式监督任务；另外包含长时间 PPO 实验。两类证据应分别理解，不能把监督任务数量写成 RL 任务覆盖。

#### 条件与限制

近期分布上的效用不保证稀有旧知识的重要性；一阶和二阶近似、权重级和特征级版本不同。PPO 仍使用 rollout 与重复更新，不因 optimizer 在线就成为严格流式 RL。

#### 阅读与实验

用可精确消融的小网络检查效用估计，再拆开保护梯度、保护噪声和 weight decay 三种作用；独立报告新学习与旧功能。

#### 原文与相关入口

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/file/8e5f0591943d8dae5702af12dcdcd2f6-Paper-Conference.pdf)：效用定义、近似、不同 UPGD 变体与 PPO 实验。
- [作者预印本](https://arxiv.org/abs/2404.00781)：流式监督协议与 RL 证据范围。

#### 作者代码

[论文首页明确链接的作者仓库；README 的短实现是一个指定变体。](https://github.com/mohmdelsayed/upgd)

权重/特征效用实验、流式任务及 PPO 实现。

### Parseval Regularization for Continual Reinforcement Learning

Wesley Chung, Lynn Cherif, David Meger, Doina Precup

NeurIPS 2024 · 2024 · 直接研究持续学习

#### 研究问题

仅在初始化时保持良好的权重几何，是否足以让很晚出现的新任务仍容易学习？

#### 关键机制

在选定隐藏层加入 $\lambda\|WW^\top-sI\|_F^2$，持续约束行向量的范数与角度；输出层及额外尺度设计保留表达能力。它维护学习的几何条件，并不直接保存旧任务标签或预测。

#### 证据

作者在 Gridworld、CARL、MetaWorld 任务序列中检验，并拆分范数与角度约束。稳定秩、Jacobian 与熵属于诊断量，不单独构成可塑性或保留的因果证明。

#### 条件与限制

约束会限制函数类；输出行数大于输入维度时，全部行正交不可实现。非线性门控仍能切断梯度。有限任务序列的结果不保证无限生命内有效，也不是无任务信息的万能机制。

#### 阅读与实验

同预算比较仅初始化正交、持续范数约束、持续角度约束和完整正则；同时记录新目标拟合、真实回报、旧功能与额外计算。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/e6df4efa20adf8ef9acb80e94072a429-Paper-Conference.pdf)：目标函数、容量限制、角度/范数消融及持续任务协议。
- [作者版本记录](https://arxiv.org/abs/2412.07224)：正式会议年份为 2024。

#### 作者代码

[仓库明确标为 NeurIPS 2024 官方实现。](https://github.com/wechu/parseval_reg)

PPO、任务序列、正则化与网络结构消融。


<a id="chapter-code"></a>

## 下载与运行

替换、optimizer 状态与合成 aged/fresh 机制诊断；不是 ReDo/CBP 全论文复现。

[下载 lifelong_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)

```sh
python lifelong_algorithms_lab.py plasticity
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sokar et al. · The Dormant Neuron Phenomenon in Deep RL](https://proceedings.mlr.press/v202/sokar23a.html)：ReDo 活动度定义、重置流程与实验。

- [ReDo 作者实现 · Google Dopamine](https://github.com/google/dopamine/tree/master/dopamine/labs/redo)：作者发布的 Dopamine 版本；核心文件为 weight_recyclers.py 及各 recycled agent。

- [Dohare et al. · Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7)：长期可塑性与 continual backpropagation 的研究。

- [CBP 作者代码 · loss-of-plasticity](https://github.com/shibhansh/loss-of-plasticity)：lop/algos/gnt.py 与 lop/utils/AdamGnT.py；效用变体、成熟期、预算和 optimizer 状态均须按配置核对。

- [Lyle et al. · Understanding Plasticity in Neural Networks](https://proceedings.mlr.press/v202/lyle23b.html)：强调曲率等机制，避免将全部 plasticity loss 简化为 dead ReLU。

- [Abbas et al. · Loss of Plasticity in Continual Deep RL](https://proceedings.mlr.press/v232/abbas23a.html)：循环 Atari 与 CReLU 的原始研究，配合预算匹配理解激活改造。

- [Nikishin et al. · The Primacy Bias in Deep RL](https://proceedings.mlr.press/v162/nikishin22a.html)：保留经验但周期重置部分网络的机制；与严格 streaming 条件不同。

- [Lyle et al. · Normalization and effective learning rates in reinforcement learning](https://papers.nips.cc/paper_files/paper/2024/hash/c04d37be05ba74419d2d5705972a9d64-Abstract-Conference.html)：NeurIPS 2024：NaP、归一化的梯度耦合，以及显式与隐式学习率调度。

- [Tang et al. · Mitigating Plasticity Loss in Continual RL by Reducing Churn](https://arxiv.org/abs/2506.00592)：ICML 2025：用参考状态的函数空间约束控制预测改变。

- [C-CHAIN 作者实现 · MinAtar Double DQN](https://github.com/bluecontra/C-CHAIN/blob/main/crl_minatar/agents/double_dqn_c_chain.py)：参考 batch、全动作 Q 正则、近期网络队列和损失尺度自适应；完整运行还需对应环境及训练配置。

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/5b76d77e7095c6480ed827b85f0c2878-Paper-Conference.pdf)：Algorithm 1–2、正则化联系与持续实验。

- [作者论文 v3](https://arxiv.org/html/2405.16642v3)：区分凸理论、RL 经验结果与初期表现限制。

- [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning · 作者实现](https://github.com/ComputationalRobotics/TRAC)：trac.py、PyTorch/JAX optimizer 包与控制/视觉实验。 作者项目页与仓库均明确标为官方实现。

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/file/8e5f0591943d8dae5702af12dcdcd2f6-Paper-Conference.pdf)：效用定义、近似、不同 UPGD 变体与 PPO 实验。

- [作者预印本](https://arxiv.org/abs/2404.00781)：流式监督协议与 RL 证据范围。

- [Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning · 作者实现](https://github.com/mohmdelsayed/upgd)：权重/特征效用实验、流式任务及 PPO 实现。 论文首页明确链接的作者仓库；README 的短实现是一个指定变体。

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/e6df4efa20adf8ef9acb80e94072a429-Paper-Conference.pdf)：目标函数、容量限制、角度/范数消融及持续任务协议。

- [作者版本记录](https://arxiv.org/abs/2412.07224)：正式会议年份为 2024。

- [Parseval Regularization for Continual Reinforcement Learning · 作者实现](https://github.com/wechu/parseval_reg)：PPO、任务序列、正则化与网络结构消融。 仓库明确标为 NeurIPS 2024 官方实现。


---

# 持续探索：新奇、不确定性、学习进展与恢复

外部奖励稀疏、世界持续改变时，怎样获得有用的新经验，而不是追逐永远无法学会的噪声？

## 本章内容

- 从计数不确定性和预测误差分别推导 count bonus 与 RND，明确 reward 在更新前还是更新后计算。
- 区分新奇、信息增益、学习进展、技能多样性和自动课程这些不同目标。
- 将探索放回无免费 reset 的世界：记录恢复成本、可达性与真实外部收益，而不是只累计 intrinsic reward。

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


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 外部奖励与内部奖励

外部奖励定义任务成功；内部奖励由学习器构造，引导获得数据。提高内部奖励不自动意味着外部目标更好。

### 认识不确定性与随机性

认识不确定性可通过更多数据减少；环境固有随机性可能不能减少。单次预测误差往往混合两者。

### 目标条件策略

$\pi(a\mid s,g)$ 把目标 $g$ 当作输入；课程决定练习的目标，控制器决定到达方法。课程调度与底层策略学习处于不同层次。

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

计数估计覆盖程度，RND 估计随机函数的熟悉程度，课程选择练习目标，恢复机制限制可接受风险。这些方法作用于不同决策层次；组合时需分别定义其目标、更新数据和资源预算。

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

$$
\operatorname{LP}_k=E_{k,\rm before}-E_{k,\rm after},\qquad \operatorname{ALP}_k=|E_{k,\rm before}-E_{k,\rm after}|
$$

E 应在匹配的任务/目标和可比较的 probe 上估计。正向 LP 强调改善；绝对进展也会把退步当作需要重新关注的信号。二者不是同一种调度规则。

想象两个区域：随机噪声区误差一直是 4，学习进展为 0；一个可学习区误差从 2 降到 1，进展为 1。若只追求误差，会偏向噪声区；若追求进展，会偏向可学习区。但有限样本中的噪声涨跌也会伪造进展，必须平滑、匹配评测数据或估计置信区间。

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

在模拟器中，跌进坑之后 reset() 即可；在现实中，卡住、断电或不可逆损坏会改变此后全部数据分布。探索可以维护一个恢复策略与恢复价值，估计从候选动作后能否回到安全或可维持后续交互的状态集合。若风险过大则切换到恢复动作，或停止冒险并请求外部干预。

$$
\mathcal A_{\rm admissible}(s)=\{a:\widehat P(\text{恢复成功}\mid s,a)\geq 1-\varepsilon\}
$$

这只是一个简化的决策接口。用学习估计 P̂ 过滤并不自动给真实安全保证；需要校准、保守不确定性界、分布外检测或外部安全机制。

“Leave no Trace”同时学习前向任务与 reset 行为，展示了恢复应成为学习系统的一部分。不同任务中的失败集与允许干预必须明确。研究报告至少记录恢复成功率、恢复耗时、不可逆失败和人工干预次数；不能只保留 forward policy 的成功回合。

<a id="lesson-example"></a>

## 8 · 手算：新奇奖励的衰减

把 RND 缩小为某个表格状态的一个标量输出：冻结 target=2，predictor 初值 0，半平方损失梯度步长 α=0.25。第一次看见，先得到 bonus=4，再更新预测为 0.5；第二次 $bonus=(2-0.5)^2=2.25$，更新为 0.875；之后两次 bonus 分别为 1.265625、0.711914。

若误把 predictor 每回合清零，相同旧状态又得到 4。这个高分不是环境提供了新知识，而是实现制造的遗忘。同样，count bonus 在访问前 N=0、3、8 时分别为 1、0.5、1/3；更新计数的时刻必须固定。

恢复例子：动作 0 的已知恢复概率是 0.99，动作 1 是 0.2，阈值 0.9。即便探索器提议动作 1，示例 filter 也只能返回动作 0；如果所有动作都不满足阈值，它返回 None，而不是随意宣称最不坏动作安全。这一保守接口应保留到真实系统。

<a id="lesson-code"></a>

## 9 · 奖励时序实现与探索诊断

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

## 研究专题 A · CPSRL：世界不重置，探索假设可重采样

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

## 研究专题 B · MoReFree：真实探索与模型内目标分布

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

Laplacian 描述行为图结构，奖励感知表示加入路径价值，METRA 学习有区别的行为，HIQL 利用离线目标轨迹，MaestroMotif 引入语言先验。它们承担不同的设计工作；生成技能、选择技能与组合技能需要分别评价。

- [Proper Laplacian Representation Learning](https://yingwen.io/zh/continual-rl/research/#recent-proper-laplacian-representations)
- [Reward-Aware Proto-Representations in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-reward-aware-proto-representations)
- [METRA: Scalable Unsupervised RL with Metric-Aware Abstraction](https://yingwen.io/zh/continual-rl/research/#recent-metra-skills)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)
- [Posterior Sampling for Continuing Environments](https://yingwen.io/zh/continual-rl/research/#recent-cpsrl-continuing-exploration)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

模型可提取性的理论说明某类能力需要什么知识，不指定唯一网络。Dreamer 研究潜在想象控制，STOMP 研究随机时长行为模型，DRAGO 研究旧模型知识保留。模型误差、查询策略和规划收益之间仍需实验连接。

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

形式化论文提供定义和条件，架构讲座提出模块组织，算法论文检验特定机制。完整系统还要明确智能体、外部设计者和世界各自承担的工作；组件成立不自动意味着组合后的长期收益成立。

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


---

# 持续智能体架构：模块接口、更新调度与长期评价

各模块单独能学，不代表接在一起就能持续改善；它们究竟交换什么、何时更新、如何共享有限计算？

## 本章内容

- 用状态、参数、数据接口与更新调度描述一个持续学习系统。
- 追踪同一条 experience 如何服务控制、GVF 与模型学习，明确每个目标和概率的参数版本。
- 在固定预算下集成 agent，分析表示漂移、模型偏差与各模块的作用。

<a id="problem-definition"></a>

## 本章的问题定义

状态、预测、控制、技能、模型和规划共享有限资源；各模块单独可学并不保证组合后目标与时序一致。

### 给定条件与符号

- 各模块的输入/输出、目标、状态和更新规则。
- 真实交互接口、共享参数、总内存和计算预算、设计/测试权限。

### 需要求解的对象

完整可执行智能体与可反驳的模块协同假设，明确每条真实经验被哪些学习过程如何使用。

### 信息与数据权限

$M_t$ 是全部持久内部信息；行为 $b(\cdot\mid M_t)$ 产生动作，记录采样时概率。预测的目标策略、模型题目和控制目标各自固定或按声明规则变化。

$$
M_{t+1}=U(M_t,A_t,R_{t+1},O_{t+1}),\qquad J_T(U,b)=\mathbb E\!\left[\sum_{t=0}^{T-1}R_{t+1}\right]
$$

$U$ 是包含全部模块调度的更新，$b$ 是行为规则，$T$ 为预定寿命；评价还附带资源和干预成本。模块局部损失只支持相应子问题，不能简单相加就称等于此整体寿命目标。

### 成立条件与解的含义

- 各模块计时、目标参数版本、行为概率和表示版本明确，更新依赖已到达数据。
- 共享表示改变时需检查旧价值、模型和技能坐标；总成本包含规划、teacher与统计。

判断准则：同一条样本的控制与固定策略预测分别符合手算target；快照、模型折扣与资源记账一致；以等预算消融检验完整寿命收益、失败和模块漂移。

### 适用边界

- Alberta/OaK研究纲领不等于已验证的通用智能体。
- 模块loss各自下降不证明组合寿命收益上升。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)：状态给各模块提供决策输入，表示漂移会使多个接口同时变化。

- 组合不同学习问题 · [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)：技能与模型接口规定可重用后果，规划不得重复折扣或使用过期技能模型。

- 组合不同学习问题 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：架构必须作为完整行动与学习过程评价，而非只看最终冻结策略。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

不同目标共享经验和表示，更新顺序可隐式改变标签；有限计算还决定哪些模块能及时适应。

### 本章的核心思路

把目标和快照版本落实为显式接口，再把真实更新、模型更新与规划按预算调度。

1. [为同一经验标注不同题目](#lesson-derive)：因为控制最大化与GVF目标策略评价不同，先缓存采样概率和旧参数，分别构造误差。

2. [使调度成为算法的一部分](#lesson-schedule)：因为先改哪个模块会改变后续标签，明确真实学习、模型更新和规划次序，并分别记账。

3. [维护表示与技能模型一致性](#lesson-drift)：因为表示或技能改变会使旧后果坐标过期，增加版本和校准诊断，以等预算模块消融检验协同。

结论与条件：教学表格组合可验证接口与时序；不将各模块的局部理论相加成任意共享深网的收敛或通用智能保证。

### 相关方法改变了什么

- 表格Dyna组合：接口可解析，适合验证真实学习和规划分工。

- STOMP式链路：从子任务到技能、后果模型与规划，需完整计入发现与维护。

- 共享深度模块：减少部分重复表示，却引入梯度冲突、表示漂移和版本一致性问题。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### Agent state

智能体从历史递推得到的决策输入 $z_t$；不必等于环境真实状态，但必须保留任务需要的信息。

### Prediction / control / model

预测回答指定策略下未来会怎样；控制选择外部收益高的行为；模型预测行为后果以便模拟计算。三者具有不同目标。

### Planning

使用模型产生的预测进行价值或策略计算，不直接增加真实环境经验；算力和模型错误须计入评价。

<a id="lesson-setting"></a>

## 1 · 持续学习模块的相互依赖

设一个机器人要持续感知、工作、学习新技能并适应磨损。它需要用历史判断当前情况，对多种未来结果形成预测，选择当前行为，形成可重复使用的技能，再用模型推演这些技能。若每个模块都依赖其他模块已经正确，整个系统就会在启动时陷入循环。架构工作的核心，是指定这些尚不准确的模块怎样共同成长。

考虑一个有限系统：数据来自一条交互流，内部包含六个手工记忆状态、两个原子动作、一个控制表、一个固定策略 GVF 和一个经验转移模型。它足以说明数据与参数的依赖关系。自动状态构造、技能发现以及跨模块元学习是在这些接口之上的进一步研究问题。

| 模块 | 输入 | 输出 / 持久状态 | 所需信息约束 |
| --- | --- | --- | --- |
| State constructor | 旧内部状态、动作、新观测 | 内部状态、表示参数、可选敏感性迹 | 不读取真实隐藏状态或任务切换 ID |
| Behavior / control | 内部状态、控制价值或 actor | 动作及采样时行为概率 | 概率必须对应实际采样策略 |
| Predictive knowledge | 转移、累积量、折扣、目标策略 | 一组 GVF 值与各自参数 | 每个问题有独立的预测定义 |
| Model learning | 状态动作与真实后果 | 奖励、终点、时长模型 | 区分已学模型与环境真模型 |
| Planning | 已学模型、价值、计算预算 | 更新后的价值或行为偏好 | 模拟计算不增加真实交互数 |
| Meta / resource allocation | 误差、后续验证信号、成本 | 步长、特征、问题及规划预算 | 后续数据只在发生后用于更新 |

说“同一条 experience 多种用途”不等于所有模块共享一个 loss。它们可以共享表示，但各自预测什么、控制什么、何时 detach 都必须明确。否则一个有用的辅助任务可能被误当成外部目标，或者目标策略概率被错误地从更新后的 actor 读取。

<a id="lesson-derive"></a>

## 2 · 一次交互的参数与目标

$$
z_t=f_{\phi_t}(z_{t-1},a_{t-1},o_t),\quad a_t\sim b_{\theta_t}(\cdot\mid z_t),\quad e_t=(z_t,a_t,r_{t+1},o_{t+1},b_t(a_t\mid z_t))
$$

状态形成后，智能体用此刻参数采样动作，并保存其行为概率。后续策略更新不改变这条经验的采样分布。

收到新观测后，在约定的旧表示版本下形成 $z_{t+1}$，计算本次真实数据的目标，再更新参数。本例控制采用平均奖励目标；GVF 的目标策略固定为 $\pi(a=1\mid z)=1$，折扣为 $0.8$。二者虽然使用相同奖励，预测对象仍然不同。

$$
\begin{aligned}\delta_t^{\rm control}&=r_{t+1}-\bar g_t+\max_a Q_t(z_{t+1},a)-Q_t(z_t,a_t),\\\delta_t^{\rm pred}&=r_{t+1}+0.8\,v_t(z_{t+1})-v_t(z_t),\\\rho_t&=\mathbf1[a_t=1]/b_t(a_t\mid z_t).\end{aligned}
$$

控制的 max 对应最优动作目标；GVF 的 ρ 对应固定 target policy。行为率、最优奖励率和 GVF 预测值必须分别命名与记录。

$$
\begin{aligned}Q(z_t,a_t)&\leftarrow Q_t(z_t,a_t)+\alpha\delta_t^{\rm control},\\\bar g&\leftarrow\bar g_t+\eta\alpha\delta_t^{\rm control},\\v(z_t)&\leftarrow v_t(z_t)+\alpha\rho_t\delta_t^{\rm pred}.\end{aligned}
$$

这三项使用同一更新前快照。将来若共享神经网络，需要明确定义 loss 合并、detach 和梯度冲突处理，不能依靠执行顺序偶然决定目标。

经验模型记录每个 (z,a) 后看到各 (r,z′) 的次数，形成经验条件分布。它不同于只记最后一次转移：后者会把真实随机性误当作不断变化的确定性结果。本例的状态、动作与奖励取值有限，因此表格槽位数固定。但 Python 整数计数与规划游标的位宽仍可随运行时间增长；槽位固定不等于无限运行时固定字节预算。若奖励连续，以每个不同奖励作为字典键还会增加槽位数。

严格的资源限制需要另行规定有限精度计数、固定容量的近期统计或参数化模型，并明确溢出与淘汰规则。这些选择会改变模型估计。本例保留累计计数，以便观察旧经验如何延缓对环境变化的适应；它不是严格固定字节预算的终生实现。

$$
\widehat p_t(r,z'\mid z,a)=\frac{N_t(z,a,r,z')}{\sum_{\tilde r,\tilde z}N_t(z,a,\tilde r,\tilde z)},\quad \delta^{\rm plan}=\sum_{r,z'}\widehat p_t(r,z'\mid z,a)[r-\bar g+\max_{a'}Q(z',a')]-Q(z,a)
$$

规划选择已观察的 (z,a) 做模型期望 backup。这里是教学性的表格 Dyna 组合；奖励率只从真实数据学习，规划只改 Q，这项选择写进接口并测试。

另一些 differential planning 方法也从模型更新奖励率，因此上述接口是一项具体设计选择。模型样本的分布与真实时钟不同；若扩展为规划也更新 $\bar g$，需要说明模型采样分布、时长和更新比例，不能将模拟次数直接当成实际经历的时间。

<a id="lesson-schedule"></a>

## 3 · 真实交互与规划的更新调度

**算法：算法伪代码**

1. 初始化 state memory、Q、ḡ、GVF、model 与固定规划预算 B
1. 每个真实环境步：
  1. 1. 用当前 agent state 计算行为分布；采样动作并保存 b(a|z)
  1. 2. 环境推进一次，返回真实 reward 和 observation
  1. 3. 用明确的表示版本计算 z′，保留本步必要中间量
  1. 4. 用更新前参数同时计算 control/GVF/model targets
  1. 5. 更新各学习器；模型只吸收此次真实 experience
  1. 6. 做恰好 B 次模型 backup；轮转选已知 state-action
  1. 7. 记录真实收益、预测误差、模型误差、更新成本
  1. 8. 若启用表示/技能重构，按版本规则处理依赖模块
  1. 9. 携带 memory 到下一步；不在记录窗口或隐藏变化时 reset

第 4 步的“同时”指语义上共同读取旧快照，不要求并行硬件执行。第 6 步的每次 planning 可以读取前一次 planning 更新后的 Q，这是异步迭代；要区别于第 4 步对真实经验目标的快照约定。代码显式写出这两个边界，避免一边修改 Q 一边意外改变本步 GVF 目标。

如果 GVF 数量 K、技能数 O 或网络规模持续增长，每步成本也可能增长。一个终生系统需要给新对象分配预算并淘汰低价值对象：例如只更新部分预测问题、限制模型缓存、分配固定 planning calls。自动增长不等于固定预算下的持续学习。

<a id="lesson-construction"></a>

## 4 · 从 Dyna 扩展到 subtask → option → model → planning

Dyna 的基础链路是“真实数据学价值和模型，模型供模拟价值更新”。时间抽象扩展还需要四个不同对象。Subtask 定义为何要学某种行为；option 定义具体如何执行以及何时停止；option model 描述实际执行产生的外部奖励、终点与时长；planner 用这些后果与当前主任务价值组合。学习技能用的人工奖励不能不加区分地写进主任务模型。

$$
\begin{aligned}r_o(z)&=\mathbb E\!\left[\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}\mid z,o\right],\\p_o(z,z')&=\mathbb E[\gamma^\tau\mathbf1(Z_{t+\tau}=z')\mid z,o],\\(T_oV)(z)&=r_o(z)+\sum_{z'}p_o(z,z')V(z').\end{aligned}
$$

折扣 option model 已把 $γ^τ$ 包进终点权重，因此 planner 不能再乘一次 γ。τ 随机时也不能把 $E[γ^τV]$ 简化成 $γ^{Eτ}E[V]$。平均奖励 option model 则保留奖励、时长和未折扣转移，使用 R−gτ。

例如学习“到门口”为了形成可复用 option，终止奖励可用于驱动这个 subtask；但主任务可能是递送物品，门口本身没有外部奖励。Planner 必须知道此技能真实消耗多少步、沿路得到什么外部收益、到达哪里，才能判断其价值。STOMP 的研究价值在于把这个链路作为可连接的学习问题，而不是在图中简单把 option 画成一条箭头。

Successor features 提供另一种可复用接口：若外部奖励近似 $r=φ(s,a,s^{\prime})^Tw$，则策略的 successor feature 预测折扣特征累积，价值近似 $ψ_π^Tw$。它方便奖励改变时快速重算价值，但已知特征线性分解、固定策略及动力学变化都是重要边界；并非等价于任意世界模型。

<a id="lesson-drift"></a>

## 5 · 表示漂移与模块一致性

一个模型昨天学会“特征第 3 维高意味着门已开”，今天 state constructor 重构后第 3 维代表速度；即使 model 参数没变，语义也失效了。将 learned state 当作普通固定数组，会掩盖这类依赖。需要同时考虑表示误差、模型误差和控制目标，而不是分别验证每个模块 loss 下降。

| 设计选择 | 怎样保持接口 | 代价或未解决点 |
| --- | --- | --- |
| 固定状态编码 | 整个实验中保持坐标语义 | 无法研究自动 state construction |
| 保存原始经验重编码 | 新表示版本上重算状态和训练模型 | 额外存储/计算，严格 streaming 下可能不允许 |
| 慢表示 / 快预测器 | 限制表示变化，让依赖者追踪 | 时间尺度需要调节，仍无任意漂移保证 |
| 显式版本与兼容检查 | 模型记录表示版本，失配时作废或迁移 | 作废丢知识，迁移本身也是学习问题 |
| 联合一致性训练 | 共享或约束 representation/model/value | 损失冲突、梯度泄漏和新优化问题 |

处理网络单元回收也类似：如果一个隐藏特征被替换，依赖它的预测头、模型输入输出、eligibility trace 以及元梯度敏感性是否仍有旧语义？应明确哪些状态清零、哪些变换、哪些保留。把一个塑性机制接到架构里，影响不只发生在该层权重。

<a id="lesson-example"></a>

## 6 · 手算同一条样本的两个不同目标

设当前状态 z0 的 Q=[1,2]，下一状态 z1 的 Q=[3,4]，ḡ=0.5；GVF 值为 v(z0)=2、v(z1)=5。实际采动作 1 的行为概率为 0.25，收到奖励 1。控制 error=1−0.5+4−2=2.5；预测 error=1+0.8×5−2=3；目标策略固定选动作 1，故 ρ=4。

取 α=0.1、η=0.1：Q(z0,1) 从 2 变成 2.25，ḡ 从 0.5 变成 0.525，GVF 从 2 变成 3.2。三个量共用一次 experience，却遵循不同目标。若采的是动作 0，控制依然学习该动作后果，而该固定目标策略 GVF 的重要性比为 0，本步不更新。

若随后做一次模型规划，Q 可再次改变；本页约定 ḡ 与 GVF 不从这条 imagined update 改变。测试通过同时运行 B=0 与 B=5，验证额外规划仅改变允许改变的模块。这类模块隔离测试比只看最终 reward 曲线更容易发现隐藏耦合。

<a id="lesson-code"></a>

## 7 · 不间断交互的集成实验

明确的参数快照、保存行为概率、经验分布模型和固定预算 round-robin planning。

```python
class ModularAgent:
    """Hand-designed Markov state; tabular control/GVF/one-step model.
    The model stores empirical counts of (reward,next_state) per state/action.
    Planning changes q only. Reward-rate learning uses REAL data only here.
    """
    def __init__(self, states=6, alpha=0.05, eta=0.05, gamma=0.8):
        self.q = [[0., 0.] for _ in range(states)]
        self.gvf = [0.] * states
        self.rate = 0.
        self.model = {}
        self.planning_cursor = 0
        self.alpha, self.eta, self.gamma = alpha, eta, gamma

    def action_probabilities(self, state, epsilon=0.2):
        best = max(range(2), key=lambda a: self.q[state][a])
        prob = [epsilon / 2., epsilon / 2.]
        prob[best] += 1. - epsilon
        return prob

    def observe(self, state, action, reward, next_state, behavior_probability,
                planning_budget=0):
        if behavior_probability <= 0 or planning_budget < 0:
            raise ValueError("invalid recorded probability or planning budget")
        # All REAL targets use one parameter snapshot before any mutation.
        q_error = reward - self.rate + max(self.q[next_state]) - self.q[state][action]
        prediction_error = reward + self.gamma*self.gvf[next_state] - self.gvf[state]
        ratio = (1.0 if action == 1 else 0.0) / behavior_probability
        self.q[state][action] += self.alpha*q_error
        self.rate += self.eta*self.alpha*q_error
        self.gvf[state] += self.alpha*ratio*prediction_error
        # Finite toy outcome slots, not an unlimited-lifetime byte bound:
        # Python counts/cursor grow in bit width; new reward values add keys.
        outcomes = self.model.setdefault((state, action), {})
        outcomes[reward, next_state] = outcomes.get((reward, next_state), 0) + 1
        # A deterministic scheduler makes the simulated-update budget inspectable.
        keys = sorted(self.model)
        for _ in range(planning_budget):
            s, a = keys[self.planning_cursor % len(keys)]
            self.planning_cursor += 1
            outcomes = self.model[s, a]
            count = sum(outcomes.values())
            delta = sum(n * (r - self.rate + max(self.q[sp]) - self.q[s][a])
                        for (r, sp), n in outcomes.items()) / count
            self.q[s][a] += self.alpha*delta
        return {"control_error": q_error, "prediction_error": prediction_error,
                "ratio": ratio, "planning_updates": planning_budget}


def architecture_run(seed=7, planning_budget=0, steps=18000):
    rng, agent = random.Random(seed), ModularAgent()
    cue, phase = rng.randrange(2), 0
    rewards, trace = [], None
    for t in range(steps):
        # State is a hand-coded (phase, remembered cue), not a discovered representation.
        state = 2*phase + cue
        probabilities = agent.action_probabilities(state)
        action = 0 if rng.random() < probabilities[0] else 1
        mapping = 0 if t < steps // 2 else 1  # hidden change, never supplied to agent
        reward = float(action == (cue ^ mapping)) if phase == 1 else 0.0
        if phase == 2:
            phase, cue = 0, rng.randrange(2)
        else:
            phase += 1
        next_state = 2*phase + cue
        trace = agent.observe(state, action, reward, next_state,
                              probabilities[action], planning_budget)
        rewards.append(reward)
    block = steps // 3
    means = [sum(rewards[i*block:(i+1)*block]) / block for i in range(3)]
    return {"window_reward_rates": [round(v, 6) for v in means],
            "optimal_rate_estimate": round(agent.rate, 6),
            "model_entries": len(agent.model), "last_update": trace}


def architectures_demo():
    print("architectures", {"without_planning": architecture_run(),
          "one_simulated_update_per_step": architecture_run(planning_budget=1),
          "scope": "fixed-budget toy integration; not an OaK implementation"})
```

Python 3.10+，无外部环境、无 GPU、无网络

```sh
python lifelong_algorithms_lab.py architectures
python lifelong_algorithms_lab.py test
```

环境每三个原始步骤形成一个自然周期：阶段 0 显示二值 cue，阶段 1 必须依据记住的 cue 选动作并获得奖励，阶段 2 过渡到下一 cue。Agent state 是手工编码的 (phase,cue)，共六种。半程隐藏地翻转 cue→正确动作的映射；程序不把切换时间或映射传给 agent，也不清参数、记忆或模型。这个自然周期不是一个对 agent reset 的训练回合。

对比每步零次和一次模型更新。最优外部奖励率为 $1/3$；使用 $\epsilon=0.2$ 探索的实际率接近 $0.3$。固定随机种子 7、运行 18000 步，无规划的三个窗口实际率约为 $[0.299667,0.2895,0.300833]$；有规划时为 $[0.299667,0.202167,0.300833]$，变化附近反而更差。规划条件下最终 $\bar g\approx0.31847$，无规划约为 $1/3$。奖励率估计与行为实际率的对象不同，因此不能将这两个量互换。累积模型仍保留旧映射，可能解释适应迟缓；需通过改变模型记忆长度、多个种子和预算匹配进一步检验。

这个实验包含手工记忆、一个固定 GVF 和原子动作模型。它适合学习模块更新关系；自动记忆学习、预测问题发现、option 构造与元学习需要额外算法。研究时可逐项加入模块，测量它究竟改善了状态区分、预测准确性、规划效率还是控制回报。

<a id="lesson-evaluation"></a>

## 8 · 从固定策略评价到持续学习过程

传统实验常在训练结束后冻结策略，再评价若干回合。这个协议回答最终策略有多好，却不能完整描述一个始终学习的智能体。后者的对象包括参数更新规则、记忆、优化器状态与资源分配；相同当前策略可能因为内部学习状态不同而具有不同的未来表现。

$$
H_t=(O_0,A_0,R_1,O_1,\ldots,O_t),\qquad M_{t+1}=U(M_t,A_t,R_{t+1},O_{t+1}),\qquad A_t\sim b(\cdot\mid M_t).
$$

$H_t$ 是完整交互历史，$M_t$ 是智能体实际保存的有限信息，包含递归状态、参数及学习状态。环境可由历史条件分布描述；智能体必须用有限 $M_t$ 近似利用它，而不能存储无限历史。

Elelimy、Szepesvari、White 与 Bowling 在 RLC 2025 提出以 history process 和面向持续学习的 deviation regret 重新讨论这一评价对象。其启发是把学习中的变化和偏离行为的后果纳入形式化，而不只比较一个与时间无关的最优策略。这是一条研究立场，并不意味着有限 MDP 或平均奖励理论失效；关键是说明模型表达了环境什么信息，以及评价量是否覆盖学习的代价与收益。

| 评价条件 | 回答的问题 | 实现时的约束 |
| --- | --- | --- |
| 冻结参数的策略评测 | 已学行为在指定分布上有多好 | 不把这条曲线当作完整生命期表现 |
| 持续学习的全程收益 | 适应期间的损失能否被后续收益补偿 | 恢复、探索、失败期间同样计时 |
| 相同数据的学习 probe | 内部学习能力是否改变 | 固定数据顺序、容量与更新预算 |
| 相同实时计算预算 | 更多规划或预测是否值得其延迟 | 报告峰值内存、每步耗时及动作频率 |

AgarCL 将这些耦合问题放在同一环境中：细胞的质量改变移动速度与视觉尺度，部分可观测性要求记忆，吞噬与避敌要求长期控制。这里可以区分两种变化：固定规则下状态相关的动力学仍可构成平稳的完整状态 MDP，而智能体看到的分布与控制尺度会随行为持续改变。仅从观测变化不能推出环境转移核本身随时间变化。

平台的完整游戏没有统一回合重置，但被吞噬的细胞会局部重生，其余世界状态继续存在。动作包含连续方向和分裂等离散选择；DQN 基线将方向离散化，PPO/SAC 使用混合动作接口。论文测试的可塑性方法未在完整游戏中建立持续胜任能力，并用小游戏分离探索、信用分配等困难。因此，一个有用的复现路线是先核对小游戏中的模块机制，再进入完整游戏，保留失败、重生和学习成本。

<a id="lesson-branches"></a>

## 9 · 模块假设与研究路线

| 假设 | 只改哪一处 | 必需的机制检查 |
| --- | --- | --- |
| 更好的 state 减少混叠 | memory / representation | 同观测不同历史的预测与动作是否可区分 |
| GVF 作为知识改善控制 | 共享特征或决策输入 | 固定控制预算，比较真实预测与随机辅助任务 |
| 新技能节省规划深度 | option 和 model 接口 | 模型奖励/时长是否为真实外部后果 |
| 规划帮助适应 | 模型与 planning budget | 模型误差、旧数据权重、每步真实计算成本 |
| Meta-learning 改善未来学习 | 步长/目标/更新规则 | 是否穿过未来数据，是否多用元训练分布 |
| 替换机制维持长期能力 | 特征或技能的 generate-and-test | 被替换对象的全部依赖状态是否一致处理 |

Alberta Plan 与 OaK 将状态、预测、控制、规划、时间抽象和元学习组织为相互依赖的研究方向。它们是理解长期目标的研究纲领；具体实验仍需给出可执行的子系统、接口与预算。完整整合及统一评估属于持续推进的开放问题。

“架构 A 比 B 好”很容易混入更多计算、更多先验、更多模型查询或更丰富目标。更有解释力的问题是：“在相同真实步数、固定模型容量和固定每步更新预算下，自动 option model 比原子模型减少多少规划误差或适应延迟？”先给出可测量接口，才有可积累的结论。

<a id="research-architecture-knowledge-contracts"></a>

## 研究专题 A · 用查询规格连接预测、技能与规划

完整架构中最容易遗漏的是知识的条件。相同的“ψ”可能指 SF 累计、flow 条件特征或 Laplacian 规划坐标；相同的“零样本”可能指无新奖励训练、无任务策略训练或无部署环境数据。模块名称不足以保证彼此兼容。可将每项知识记录为一份可检查的查询规格：它在什么状态版本、行为、时域与目标下预测什么，谁会使用它。

$$
K_j=(q_j,\,\mathcal D_j,\,\nu^{\rm state}_j,\,\nu^{\rm behavior}_j,\,\widehat y_j,\,\epsilon_j,\,c_j),\qquad q_j=(\pi_j,C_j,\gamma_j,\text{readout}_j)
$$

这是拟议的架构元数据规格：查询、验证域、状态与行为版本、预测器、所用校准误差及成本。ε 只有在明示验证协议下才有含义，不默认是全环境的数学上界；readout 指下游要从该知识算什么。

| 模块知识 | 明确依赖 | 允许的下游查询 | 不能据此宣称 |
| --- | --- | --- | --- |
| SF / FB | 动力学、策略族、特征与奖励读出 | 被表示覆盖的奖励价值或任务条件行为 | 任意动力学变化立即迁移 |
| SF² | 条件特征、flow 投影、目标策略及生成器 | 未来占用采样，或给非线性 critic 的特征 | 任意 GVF 可线性读出 |
| 方向技能 / OKB | 基础策略、SF、元策略及调用规则 | 组合执行；额外模型可支持规划 | 技能组合自动得到准确后果模型 |
| DINO-WM / 2-AC | 冻结视觉编码、动作语义、离线动力学数据 | 给定目标下的动作条件特征预测与搜索 | 在线自动状态/技能构造已完成 |
| VE / 分布等价模型 | 策略/价值或统计摘要查询族 | 该族范围内的 Bellman 或风险计算 | 模型还原全部世界，或保证任意新风险目标 |

例如编码器升级后，旧 DINO 特征终点与新目标坐标不能直接比较；低层停止规则改变后，旧 option model 的 τ 与终点不再对应当前执行；主任务改成风险规避后，旧 mean-value 等价模型的规格不够。失配来自明确依赖，而不是看到回报下降后笼统称“遗忘”。

**算法：这是一项可实现的集成研究设计，不等于现成 OaK 实现**

1. 接口维护骨架（拟议）：
  1. 真实 transition 只推进环境一次，保存采样时行为与状态版本
  1. 分发给各自明确的 prediction/model/control learner
  1. 问题或技能更新后标记依赖知识过期，安排真实后续验证
  1. planner 只调用查询条件与版本匹配的知识；失配时退回已有合法接口
  1. 记录旧知识重新校准、局部迁移与作废的成本
  1. 固定总容量与每步更新预算，比较接口检查对失败/恢复的作用

OaK 与 Alberta Plan 提出经验产生预测、子任务、options、模型与规划的长期研究链。上述近年工作分别强化其中若干接口，但外部数据、固定表示或已给定奖励族仍可能参与启动。研究价值在于明确这些模块怎样互相提供学习信号，并检验未见变化下的真实收益，而非将多个论文模块接在一起就宣称完成完整架构。

<a id="research-architecture-future-utility-and-budget"></a>

## 研究专题 B · 从预训练能力到有限预算下的持续知识维护

预训练模型降低在线学习的起点成本，持续架构还需要决定学什么、何时补齐、何时淘汰。OKB 的行为基补齐、GVF 问题发现、generate-and-test 的特征维护针对不同对象，可以共享“未来是否有用”的评价思想，但不能把它们直接视为同一个算法。特别是低预测误差可能仅说明问题太容易，高误差也可能来自不可约噪声。

$$
U_j(W)=\underbrace{J_W(\text{with }K_j)-J_W(\text{without }K_j)}_{\text{同协议下的未来收益差}}-\lambda_C\Delta C_j-\lambda_M\Delta M_j,\qquad \sum_jM_j\leq M_{\max},\quad \sum_jC_{j,t}\leq C_{\max}
$$

这是资源受限知识效用的拟议检验量，不是新引用定理。$J_W$ 必须明确是未来真实收益、查询误差减少或其他指标，不能混用；with/without 需要配对随机性、冻结副本或独立控制实验，不能读取同一生命的反事实结果。

例子：一个新预测头使充电风险估计更准确，却增加每步延迟，以致机器人错过控制频率；另一个技能提高覆盖，但长期不被规划器查询。这两者可能有预测或探索价值，却未必有净控制效用。只有在同样真实时间、容量与更新预算下测到未来收益，才能判断应否保留。

| 加入候选后的检验 | 回答的问题 | 应保留的边界 |
| --- | --- | --- |
| 未来固定行为预测 | 新知识是否减少未见后果误差 | 不把 replay 拟合当作未来检验 |
| 冻结副本读出 | 现有表示是否让查询可低成本恢复 | probe 不写回正在评价的 agent |
| 影子规划 | 知识是否改变正确的动作排序 | 不把模型里的收益当作真实环境收益 |
| 配对控制 | 实际调用是否改善未来互动 | 计入探索、失败和恢复成本 |
| 预算压力与淘汰 | 有限容量是否仍能持续补齐 | 记录旧能力损失与未来再学习成本 |

从 DINO-WM 或 V-JEPA 2-AC 启动的系统，可先冻结 encoder 只持续更新后果预测，之后才加入状态、问题或技能生成；从 SF² 启动则可先测试新查询可读出程度，再决定是否需扩容。每次新增机制都应有同预算对照，避免把更多数据、更多参数或更多规划混进架构结论。

尚未闭合的研究问题包括：任务和查询不断新增时如何分配信用；知识版本改变时怎样保留兼容接口；新动力学需要探索时怎样支付机会成本；被淘汰知识在未来再次出现时怎样恢复。这些是全局持续学习的问题，与某个静态基准上最终分数提高不同，应以一条完整经验流的收益与资源轨迹验证。

<a id="lesson-check"></a>

## 10 · 自测与集成实验

- 为什么行为概率必须随经验保存？它描述采样当时分布；更新后策略不能倒过来解释历史动作。
- 同一个 reward 能否同时用于平均奖励控制和折扣 GVF？可以，但 discount、target policy 与意义不同，需要两个目标和估计器。
- 为什么模型有 loss 下降，planning 仍可能伤害行为？训练分布、规划查询分布和非平稳旧数据不同；小局部误差可以被反复 backup 放大。
- 在日志上切成多个窗口，能清空 recurrent state 吗？不应无声明清空，否则改变了任务协议。
- 只有多个模块接在一起就算 OaK 实现吗？不算；需逐项给出目标构造、学习器、接口、调度、预算以及开放环节。

动手题：把经验模型从终生累计计数改成带固定容量的近期统计，保持规划次数不变，研究隐藏变化后恢复速度与稳态方差。再把 cue memory 去掉，让两个不同历史映射到同一状态，观察加更多 planning 是否能修复信息丢失。预期不能：计算无法凭空恢复被状态表示丢弃的信息。

## 本章的实验设计

每次消融都对应一个接口假设。报告中间量和长期收益。完整系统的改进不能直接归因于其中一个模块。

设定：先固定技能和消费者检验预测模块，再比较预测开关×技能开关；在明确收益链后接模型或规划。

- 每个输出有明确消费者、参数版本与可用时刻。
- 改变诊断 oracle 不会通过中间模块进入部署输入。
- 完整快照包含各模块状态和事件队列；恢复能力按真实支持声明。

对照：四格交互设计与同容量基座；已训练系统断链与无模块重新训练分别报告；准确模块诊断及固定/在线版本

记录：预测准确、技能成功与真实回报三层指标；表示/目标/模型版本漂移；全部模块的计算、存储及延迟

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-planning)

## 学习与研究衔接

连接模块后才出现的数据分布反馈，需要单独验证。MARL 还要控制其他智能体变化与训练信息权限。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-architectures) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=architectures) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=architectures)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

状态是支持后续计算的内部信息；GVF 指定一个预测问题；RTRL 和资格迹规定信用如何传播。三者可以组合，但不是相互替代的算法名称。先理解给定策略的预测，再讨论预测怎样改善控制。

- [Towards model-free RL algorithms that scale well with unstructured data](https://yingwen.io/zh/continual-rl/research/#recent-nibbler-predictive-features)

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://yingwen.io/zh/continual-rl/research/#recent-vtrace-impala)

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

Laplacian 描述行为图结构，奖励感知表示加入路径价值，METRA 学习有区别的行为，HIQL 利用离线目标轨迹，MaestroMotif 引入语言先验。它们承担不同的设计工作；生成技能、选择技能与组合技能需要分别评价。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [MaestroMotif: Skill Design from Artificial Intelligence Feedback](https://yingwen.io/zh/continual-rl/research/#recent-maestromotif-semantic-skills)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

模型可提取性的理论说明某类能力需要什么知识，不指定唯一网络。Dreamer 研究潜在想象控制，STOMP 研究随机时长行为模型，DRAGO 研究旧模型知识保留。模型误差、查询策略和规划收益之间仍需实验连接。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [Mastering diverse control tasks through world models](https://yingwen.io/zh/continual-rl/research/#recent-dreamerv3-world-models)
- [General Agents Contain World Models](https://yingwen.io/zh/continual-rl/research/#recent-general-agents-world-models)
- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://yingwen.io/zh/continual-rl/research/#recent-tdmpc2-decision-time-model)
- [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://yingwen.io/zh/continual-rl/research/#recent-dino-wm-feature-planning)
- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://yingwen.io/zh/continual-rl/research/#recent-vjepa2-action-conditioned)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [Rethinking the Foundations for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-rethinking-crl-foundations)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [Loss of plasticity in deep continual learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-backpropagation)
- [Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-vla-simple-recipe)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Learning from experience instead of curated datasets](https://yingwen.io/zh/continual-rl/research/#recent-oak-network-idbd)
- [Discovering state-of-the-art reinforcement learning algorithms](https://yingwen.io/zh/continual-rl/research/#recent-disco-rl)
- [How Should We Meta-Learn Reinforcement Learning Algorithms?](https://yingwen.io/zh/continual-rl/research/#recent-meta-algorithm-search-comparison)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [Position: Lifetime tuning is incompatible with continual reinforcement learning](https://yingwen.io/zh/continual-rl/research/#recent-lifetime-tuning)
- [The Cell Must Go On: Agar.io for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-agarcl)
- [Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-vla-simple-recipe)
- [How Should We Meta-Learn Reinforcement Learning Algorithms?](https://yingwen.io/zh/continual-rl/research/#recent-meta-algorithm-search-comparison)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文提供定义和条件，架构讲座提出模块组织，算法论文检验特定机制。完整系统还要明确智能体、外部设计者和世界各自承担的工作；组件成立不自动意味着组合后的长期收益成立。

- [Rethinking the Foundations for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-rethinking-crl-foundations)
- [Plasticity as the Mirror of Empowerment](https://yingwen.io/zh/continual-rl/research/#recent-plasticity-mirror-empowerment)
- [General Agents Contain World Models](https://yingwen.io/zh/continual-rl/research/#recent-general-agents-world-models)
- [The OaK Architecture: A Vision of SuperIntelligence from Experience](https://yingwen.io/zh/continual-rl/research/#recent-oak-architecture)
- [Towards model-free RL algorithms that scale well with unstructured data](https://yingwen.io/zh/continual-rl/research/#recent-nibbler-predictive-features)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)

### Towards model-free RL algorithms that scale well with unstructured data

Joseph Modayil, Zaheer Abbas

arXiv 预印本 · 2023 · 支持方法与理论

#### 研究问题

大量原始观测中只有少数局部组合与奖励有关，智能体能否逐步构造有用的预测特征？

#### 关键机制

Nibbler 将预测问题的构造和预测结果的复用结合起来：选择局部输入、学习与奖励相关的通用价值预测，再将预测作为后续学习的特征。GVF 在这里不是一个新的优化器，而是描述“预测什么、在什么行为下预测”的问题接口。

#### 证据

作者在组合式合成环境中增加观测规模，报告了利用任务结构的样本效率。环境可以具有指数增长的状态组合，但学习器不必显式枚举全部状态。

#### 条件与限制

这不是对任意高维观测的线性样本复杂度保证。局部可分解结构、候选问题与特征构造规则仍是关键条件；从该实验族迁移到视觉控制需要额外验证。

#### 阅读与实验

把一条预测完整写成累积量、延续条件、目标策略和输入特征四项，再指出它如何进入主任务的价值函数。区分问题生成带来的收益与增加参数量带来的收益。

#### 原文与相关入口

- [作者预印本](https://arxiv.org/abs/2311.02215)：问题族、Nibbler 构造过程与扩展性实验。

### Reward-Respecting Subtasks for Model-Based Reinforcement Learning

Richard S. Sutton, Marlos C. Machado, G. Zacharias Holland, David Szepesvari, Finbarr Timbers, Brian Tanner, Adam White

Artificial Intelligence · 2023 · 支持方法与理论

#### 研究问题

学到一个能到达子目标的技能之后，为什么它仍可能不适合主任务规划？

#### 关键机制

STOMP 把子任务、option、模型和规划连起来。子任务保留原任务的路径奖励，并用带有特征偏好的终止价值表达目标；学习得到策略和终止规则后，再预测该行为的累计奖励与折扣终点。这样，技能不会因为只追求到达子目标而忽略途中代价。

#### 证据

论文用明确的小问题展示奖励感知子任务如何产生更有用的行为和规划模型。它提供的是可分析的构造链，而非只比较一个技能执行成功率。

#### 条件与限制

终止收益的约定是子任务定义的一部分，不能随意换成固定终点奖励。特征和子任务候选的选择尚不等于完整自主发现机制；实验也不构成整个 OaK 架构的验证。

#### 阅读与实验

在同一个绕路环境中比较“最短到达目标”和“保留路径奖励”的子任务。分别计算 option 的奖励模型、折扣终点模型与一次规划备份。

#### 原文与相关入口

- [期刊论文](https://doi.org/10.1016/j.artint.2023.104001)：STOMP 与奖励感知子任务的正式论文。
- [作者预印本](https://arxiv.org/abs/2202.03466)：最初预印本早于期刊年份；阅读停止收益的精确定义。

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

### Loss of plasticity in deep continual learning

Shibhansh Dohare, J. Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A. Rupam Mahmood, Richard S. Sutton

Nature · 2024 · 直接研究持续学习

#### 研究问题

一个长期训练的网络如何保留继续形成新特征的能力？

#### 关键机制

Continual Backpropagation 在梯度学习之外持续生成并测试特征。它估计单元的效用和成熟度，少量替换低效用的成熟单元，并协调新单元的输入、输出和相关状态。维护新的可学习方向是一个持续过程，而不是等到任务切换后整体重启。

#### 证据

论文在长序列监督学习与强化学习问题中展示可塑性损失，并检验特征替换的作用。作者仓库包含 generate-and-test 与优化器状态处理。

#### 条件与限制

有限序列上的学习保持不保证无限生命中的任意适应。替换率、效用定义与成熟度条件仍需选择；新任务学习速度和旧能力保留必须分开测量。

#### 阅读与实验

逐项消融“成熟度筛选”“效用筛选”“随机替换”。比较相同替换预算，检验收益究竟来自定向回收还是一般参数扰动。

#### 原文与相关入口

- [Nature 原文](https://doi.org/10.1038/s41586-024-07711-7)：长期可塑性实验与 continual backpropagation。
- [作者代码](https://github.com/shibhansh/loss-of-plasticity)：关注 lop/algos/gnt.py 及替换时的优化器状态。

#### 作者代码

[论文作者公开的实验实现。](https://github.com/shibhansh/loss-of-plasticity)

论文任务、持续反向传播与 generate-and-test。

### Discovering state-of-the-art reinforcement learning algorithms

Junhyuk Oh, Gregory Farquhar, Iurii Kemaev, Dan A. Calian, Matteo Hessel, Luisa Zintgraf, Satinder Singh, Hado van Hasselt, David Silver

Nature · 2025 · 支持方法与理论

#### 研究问题

除了学习策略，能否从大量学习过程里学出更有效的 RL 更新规则？

#### 关键机制

DiscoRL 用外层优化评价执行若干内层更新后的行为表现，学习价值、策略与辅助预测之间的更新方式。被训练的对象是学习算法本身，而不仅是某个任务的策略参数。内外两层有各自的数据、时间尺度与计算预算。

#### 证据

论文报告跨环境发现更新规则与迁移到未见环境的结果，并公开配套算法实现。它展示了自动算法发现的可能性，但依赖大规模外层训练。

#### 条件与限制

外层在大量环境和设备上的搜索属于设计者侧资源，不能记作测试智能体单次生命内的自主学习。公开规则的执行成本与发现该规则的成本应分别报告。

#### 阅读与实验

画出内层参数和外层参数的更新依赖，再列出测试时哪些量被冻结。与在线 IDBD 比较时，先区分跨任务算法发现和单流步长追踪。

#### 原文与相关入口

- [Nature 原文](https://www.nature.com/articles/s41586-025-09761-x)：算法发现过程、外层资源与泛化实验。
- [作者实现](https://github.com/google-deepmind/disco_rl)：配套代码与发现的更新规则。

#### 作者代码

[Google DeepMind 的论文配套仓库。](https://github.com/google-deepmind/disco_rl)

DiscoRL 配套实现与学习到的更新规则；具体训练资源以仓库说明为准。

### Learning from experience instead of curated datasets

Oak Lab

Oak Lab 技术博文 · 2026 · 支持方法与理论

#### 研究问题

有用信号稀疏且大量输入是噪声时，在线学习规则如何分配不同方向的更新能力？

#### 关键机制

博文从含稀有有效特征的线性预测问题出发，对比统一步长与 IDBD 的逐权重适应，再展示 NetworkIDBD 在非线性带噪观测中的例子。核心主张是让长期学习效果影响信用和步长分配，而不只依据当前梯度幅度归一化。

#### 证据

公开页面提供受控噪声特征任务和 NoisyMNIST 示例。它们是机制演示，便于理解有效信号密度与输入规模的关系。

#### 条件与限制

该页面不是完整 CRL 控制论文，也未给出可直接复现所有图表的完整代码和算法推导。监督噪声任务的结果不能证明一般 SGD 或所有深度 RL 都无法从经验学习。

#### 阅读与实验

先复现线性噪声特征问题，分开改变有效特征稀疏度与噪声维数。进入控制前，再加入策略改变数据分布这一因素。

#### 原文与相关入口

- [Oak Lab 原始博文](https://oaklab.ai/posts/learning-from-experience-instead-of-curated-datasets)：2026 年 7 月 13 日；受控实验、NetworkIDBD 示例与研究动机。

### Mastering diverse control tasks through world models

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap

Nature · 2025 · 支持方法与理论

#### 研究问题

同一套世界模型训练与控制方法，能否减少跨任务重新设计损失和超参数的需求？

#### 关键机制

DreamerV3 从经验学习递归潜在状态、奖励和延续预测，再在潜在想象轨迹上学习 actor 和 critic。尺度稳健的表示与损失设计使同一配置可以适用于多种任务。模型是用于决策的学习接口，不必生成完整真实世界。

#### 证据

论文在大量视觉和状态控制任务上报告了广泛表现。关键含义是共享算法配置；这些结果主要来自分别训练的任务智能体，不是一个智能体按顺序学会全部任务。

#### 条件与限制

经验重放、批量训练和模型想象都有资源成本。模型偏差、表示遗忘与长期任务切换仍需要专门实验，不能由多任务覆盖范围自动推出持续学习能力。

#### 阅读与实验

把状态更新、模型训练、想象起点和策略更新四种分布分别写清。比较真实交互步数之外，还应记录想象步数和优化次数。

#### 原文与相关入口

- [Nature 原文](https://www.nature.com/articles/s41586-025-08744-2)：方法和任务协议；区分共享配置与单智能体持续学习。
- [作者维护的实现](https://github.com/danijar/dreamerv3)：公开实现的版本与论文实验环境应分别记录。

#### 作者代码

[作者发布的重实现；不把当前分支当作原论文实验的冻结快照。](https://github.com/danijar/dreamerv3)

DreamerV3 的作者维护公开实现及运行配置。

### General Agents Contain World Models

Jonathan Richens, David Abel, Alexis Bellot, Tom Everitt

ICML 2025 · 2025 · 支持方法与理论

#### 研究问题

能完成足够丰富的目标集合，是否意味着智能体内部已经包含可提取的环境预测知识？

#### 关键机制

论文在形式化条件下，将广泛多步目标上的行为能力与环境模型的可提取性联系起来。通过查询智能体对不同目标的行为，可以恢复关于环境后果的信息；目标集合和性能要求越强，所要求的预测知识也越强。

#### 证据

主要证据是给定假设下的理论结果，而不是某个世界模型架构在所有任务上击败无模型算法的实验。

#### 条件与限制

可提取模型不等于智能体显式保存一个 RSSM，也不意味着所有实用任务都需要重建全部环境。必要知识的结论不能代替如何高效学到它的算法。

#### 阅读与实验

列出定理要求的目标丰富性和查询能力，再尝试构造一个只会单一任务的反例。由此区分任务专门知识与支持广泛目标的预测模型。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2506.01622)：形式化设定、模型可提取性与证明。
- [David Abel 论文目录](https://david-abel.github.io/papers.html)：作者提供的 ICML 2025 发表信息及相关研究。

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

### Position: Lifetime tuning is incompatible with continual reinforcement learning

Golnaz Mesbahi, Parham Mohammad Panahi, Olya Mastikhina, Steven Tang, Martha White, Adam White

ICML 2025 Position Paper · 2025 · 评价与实验协议

#### 研究问题

如果设计者用完整未来生命反复调参，实验还在测智能体面对未知变化的能力吗？

#### 关键机制

论文限制调参可访问的生命阶段，并比较这种选择方式与利用完整生命回报挑选配置的差异。外部设计者掌握未来变化信息，可能使一个并不自适应的固定算法显得适应良好。核心改变发生在评价协议，而不是 TD 更新公式。

#### 证据

作者用持续、非平稳设置中的深度 RL 实验说明超参数选择可以改变方法比较。该工作属于立场论文，论证与示例用于推动更符合问题目标的评价。

#### 条件与限制

允许多少开发阶段经验需要按应用规定，不存在由该论文推出的普适固定比例。仅限制时间前缀也不能替代独立测试种子和计算预算控制。

#### 阅读与实验

对同一配置集合分别按开发前缀和完整生命选择超参数，再在独立测试生命上比较。报告两种选择使用了哪些未来信息。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/mesbahi25a.html)：调参协议、论证与示例实验。

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

### Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning

Jiaheng Hu, Jay Shim, Chen Tang, Yoonchang Sung, Bo Liu, Peter Stone, Roberto Martín-Martín

RLC 2026 · 2026 · 直接研究持续学习

#### 研究问题

大规模预训练的视觉—语言—动作模型，是否仍需要复杂机制才能顺序学习控制任务？

#### 关键机制

论文研究对预训练 VLA 进行顺序强化学习，并以低秩适配等相对简单的训练流程检验持续学习。预训练表示、可更新参数子空间和 RL 目标共同决定迁移与遗忘，不能只把结果归因于单一保留正则项。

#### 证据

作者在多种 VLA 与长期任务基准上比较，并提供实验代码。RLJ 的 RLC 2026 论文页与论文脚注分别给出正式入口和作者仓库。

#### 条件与限制

预训练数据和算力属于外部资源，任务与重置协议也影响难度。该结果不意味着从零训练的网络不会遗忘，更不意味着任意无边界任务流只需微调。

#### 阅读与实验

固定预训练模型，分别改变可训练参数量和任务顺序。报告预训练资源、每任务在线数据、回放或重置条件，再与传统 CRL 方法比较。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2603.11653)：VLA 持续学习设置、机制与比较。
- [RLC 2026 / RLJ 论文页](https://rlj.cs.umass.edu/2026/papers/Paper84.html)：会议原文入口；PDF 脚注链接作者代码。
- [作者实现](https://github.com/UT-Austin-RobIn/continual-vla-rl)：持续 VLA 的训练与评价代码。

#### 作者代码

[UT Austin RobIn 实验室的原论文仓库。](https://github.com/UT-Austin-RobIn/continual-vla-rl)

预训练 VLA 的顺序 RL 训练和论文评价。

### The OaK Architecture: A Vision of SuperIntelligence from Experience

Richard S. Sutton

RLC 2025 讲座 / Oak Lab · 2025 · 定义与架构观点

#### 研究问题

持续学习是否只是在一个现成 actor–critic 上加入抗遗忘机制，还是需要重新安排知识构造与使用？

#### 关键机制

OaK 提出从经验持续形成状态、预测知识、子任务、时间抽象与模型，并让这些知识服务规划的架构方向。这里的重点是模块之间怎样产生可复用知识，而不只是保留某个固定策略网络的参数。

#### 证据

官方页面提供 Richard Sutton 的架构讲座与相关研究入口。STOMP、预测学习和在线特征学习等论文可以检验其中具体组件，但不能自动验证整体架构。

#### 条件与限制

这是研究愿景与架构讲解，不是一套已公布完整训练配方、统一基准结果和可复现端到端代码的系统。资源分配、问题生成、知识替换与模块相互干扰仍需明确算法。

#### 阅读与实验

为每个模块写出输入、输出、更新频率和资源上限。再选择一个双模块接口做可证伪实验，例如技能模型改善是否真的减少规划误差。

#### 原文与相关入口

- [Oak Lab 官方讲座页面](https://oaklab.ai/posts/the-oak-architecture)：讲座入口与架构研究方向。
- [Oak Lab 研究主页](https://oaklab.ai/)：区分已发表研究、技术文章和仍在预告中的项目。

### How Should We Meta-Learn Reinforcement Learning Algorithms?

Alexander David Goldie, Zilin Wang, Jaron Cohen, Jakob Foerster, Shimon Whiteson

RLC 2025 / RLJ · 2025 · 评价与实验协议

#### 研究问题

算法表示、发现算法的方法和测试智能体的学习成本，应该如何独立比较？

#### 关键机制

对 RL 流程的不同组件进行算法发现，比较黑盒学习、神经/符号蒸馏与 LLM 代码提案。学习器的表示形式和搜索过程分开定义，才能识别泛化、可解释性和成本之间的取舍。

#### 证据

论文直接比较元训练、元测试、样本成本、训练时间与可解释性，作者代码按发现方法和评价入口组织。

#### 条件与限制

跨环境发现规则主要发生在设计者侧，不等于运行智能体已能终生修改规则。论文的训练任务和预算范围不支持所有算法发现方法的普适排序。

#### 阅读与实验

固定被学习组件、输入权限、元训练数据和发现预算；封存规则后再检验未见环境、长生命与未通知漂移。

#### 原文与相关入口

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_218.pdf)：比较对象、元训练/测试与多维成本。
- [RLJ 论文记录](https://rlj.cs.umass.edu/2025/papers/Paper218.html)：作者与正式会议收录。

#### 作者代码

[论文提供、仓库标为官方的作者实现。](https://github.com/AlexGoldie/learn-rl-algorithms)

learning_algorithms 中各发现方法与独立 evaluation 流程。

### IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures

Lasse Espeholt, Hubert Soyer, Rémi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, Koray Kavukcuoglu

ICML 2018 · 2018 · 支持方法与理论

#### 研究问题

actor采样策略落后于learner时，如何校正状态价值与策略更新？

#### 关键机制

V-trace用截断ρ校正当前TD误差，用独立截断c控制后续误差传播，再用下一状态V-trace目标构造actor优势。ρ上限还决定表格固定点对应的截断策略。

#### 证据

原文分析固定点并检验分布式多任务训练。固定版本作者代码明确区分clipped_rhos、cs、反向scan与pg_advantages。

#### 条件与限制

IMPALA保存短轨迹并批量训练，不属于严格单样本流式协议。截断后价值可能对应不同于原目标的策略；信用章bandit示例显示0.8变为0.5。

#### 阅读与实验

独立改变策略滞后、ρ上限和c上限。记录目标策略变化与传播长度，不把两种截断都只解释为方差控制。

#### 原文与相关入口

- [ICML原文](https://proceedings.mlr.press/v80/espeholt18a.html)：V-trace固定点与分布式实验。
- [作者固定实现](https://github.com/google-deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/vtrace.py)：from_importance_weights与下一状态actor目标。

#### 作者代码

[原作者团队仓库的固定版本。](https://github.com/google-deepmind/scalable_agent/tree/6c0c8a701990fab9053fb338ede9c915c18fa2b1)

IMPALA原始TensorFlow实现与V-trace；运行需要原项目环境。

### Constructing an Optimal Behavior Basis for the Option Keyboard

Lucas N. Alegre, Ana L. C. Bazzan, André Barreto, Bruno C. da Silva

NeurIPS 2025 · 2025 · 支持方法与理论

#### 研究问题

状态相关的技能组合足够强时，是否仍须为每个新奖励保存一条完整策略？

#### 关键机制

OKB 联合扩充基础策略与 Option Keyboard 的元策略。线性支持方法选择需要补齐的任务权重，先训练现有基础上的组合，再检查无法表达的动作并新增基础，移除冗余项。优化的是可组合的行为基，而非仅增加技能数量。

#### 证据

正式原文分析基础数量与 convex coverage set 的关系，并在多任务领域检验规模与表现。附录提供元策略、新基础训练和角点枚举等实现细节；论文声明实验代码在 Supplemental Material。

#### 条件与限制

保证假定 NewPolicy 返回最优策略，且 TrainOK 能达到可表达的最优组合。近似 critic、有限训练、变化动力学不直接继承保证；非线性任务结论仅覆盖最优行为可由相应子策略组合的类别。

#### 阅读与实验

分别比较“增加基础”“只训练组合”和“去除冗余”。保留一组未用于基构造的奖励权重，记录基础规模、元策略成本、SF 误差与迁移回报。持续淘汰仍需未来任务效用检验。

#### 原文与相关入口

- [NeurIPS 2025 原文与补充材料入口](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0ab48777def88e73b50746a6011be0b0-Abstract-Conference.html)：算法 1–3、附录 A.3 的两个最优子程序假设及代码声明；未在本教材运行补充代码。
- [Option Keyboard 的经典桥梁](https://proceedings.neurips.cc/paper/2019/file/251c5ffd6b62cc21c446c963c76cf214-Paper.pdf)：cumulant 组合、GPE/GPI 与技能接口。

### TD-MPC2: Scalable, Robust World Models for Continuous Control

Nicklas Hansen, Hao Su, Xiaolong Wang

ICLR 2024 · 2024 · 支持方法与理论

#### 研究问题

如何让短期动力学与长期价值分工，并在动作选择时继续使用模型？

#### 关键机制

TD-MPC2 学习无需观测 decoder 的潜在动力学、奖励、价值与策略先验。决策时优化有限动作序列，用终点价值补上未展开的后果；执行第一步后，利用新观测重新规划。

#### 证据

正式会议原文报告 104 个在线任务和单一大型多任务智能体的实验。官方仓库包含模型训练与计划接口，适合与 Dreamer 的想象 actor 学习比较计算位置。

#### 条件与限制

跨任务共享超参数和多任务能力不是单条生命流中持续适应的证据。replay、任务条件、模型更新、决策延迟等成本需进入 CRL 协议；长程 critic 错误不能被短期模型精度自动修复。

#### 阅读与实验

固定模型，对比无终点价值、不同 horizon 和不同规划预算；再固定预算比较部署 actor 与决策时搜索。环境变化后同时记录模型校准、critic 误差和恢复收益。

#### 原文与相关入口

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html)：短期预测、终点价值、多任务协议。
- [作者实现](https://github.com/nicklashansen/tdmpc2)：训练、模型与 plan 函数分别阅读。

#### 作者代码

[作者维护的原论文代码。](https://github.com/nicklashansen/tdmpc2)

TD-MPC2 的单任务/多任务训练和决策时规划。

### DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning

Gaoyue Zhou, Hengkai Pan, Yann LeCun, Lerrel Pinto

ICML 2025 · 2025 · 支持方法与理论

#### 研究问题

预训练视觉表示能否直接成为动作后果预测与目标规划的接口？

#### 关键机制

冻结 DINOv2 空间 patch 特征，用离线动作轨迹学习未来特征预测器；测试时优化动作序列，让预测特征接近目标图像特征。没有重建图像、奖励模型或逆模型，不表示没有动作条件的动力学训练。

#### 证据

ICML 原文在六类环境检验视觉目标规划，作者仓库公开数据、部分检查点、训练与 CEM 规划入口。零样本指给定已训练模型后解决目标，无额外任务策略训练。

#### 条件与限制

依赖视觉预训练与离线交互覆盖；patch 相近不总等于任务完成或风险相同。原实验不证明冻结视觉表示能适应长期新物体、新动作语义或隐藏状态。

#### 阅读与实验

分别改变背景、物体属性、控制动力学与目标分布。把冻结 encoder 和联合更新 encoder 分开，对照视觉距离、真实成功与模型误差，观察表示漂移的依赖成本。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/zhou25t.html)：正式发表入口；早期 ICLR 投稿页不能替代此状态。
- [作者项目代码](https://github.com/gaoyuezhou/dino_wm)：train.py、plan.py、数据与已公开模型检查点范围。

#### 作者代码

[原作者 Gaoyue Zhou 的论文配套仓库。](https://github.com/gaoyuezhou/dino_wm)

DINO 特征预测、离线环境数据与目标规划；README 公开部分环境检查点。

### V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, Nicolas Ballas

arXiv 预印本（此处采用 2025 首稿） · 2025 · 支持方法与理论

#### 研究问题

无动作标注的视频预训练，与能接受机器人动作的规划模型之间还缺哪一步？

#### 关键机制

V-JEPA 2 先学被遮蔽视频的潜在特征预测；V-JEPA 2-AC 冻结编码器，再用机器人轨迹训练动作条件预测器。控制以目标图像的特征差为代价进行 MPC；视频理解、动作条件预测和真实控制是三个独立证据层。

#### 证据

2025 首稿报告以大规模视频预训练，再用不到 62 小时 DROID 交互视频后训练，在两个实验室以图像目标做真实机器人规划。论文单独讨论相机位置、长程规划与图像目标的局限。

#### 条件与限制

无任务奖励并不等于无动作、无机器人状态或无外部数据。零样本部署未持续更新模型，也未发现和维护 options。官方仓库现含 V-JEPA 2.1，复现首稿须记录配置和模型版本。

#### 阅读与实验

按视觉编码、动作坐标、后果模型、目标代价逐项做迁移检验。若引入在线更新，记录模型更新使旧目标接口失效的程度，测未来交互收益，而非仅用 frozen probe 证明 CRL。

#### 原文与相关入口

- [2025 首稿](https://arxiv.org/abs/2506.09985v1)：action-free 预训练、2-AC 后训练、真实规划与第 4.3 节限制；此处不赋予未核实会议状态。
- [Meta FAIR 官方实现](https://github.com/facebookresearch/vjepa2)：包含 V-JEPA 2、2-AC 和较新的 2.1；版本不能混用。

#### 作者代码

[Meta FAIR 官方仓库；首稿模型与后续版本需按配置区分。](https://github.com/facebookresearch/vjepa2)

官方视频表征与动作条件模型；数据、机器人部署条件与检查点分别核验。


<a id="chapter-code"></a>

## 下载与运行

六状态、双动作、手工记忆的集成教学骨架；不是完整 OaK/STOMP 性能复现。

[下载 lifelong_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)

```sh
python lifelong_algorithms_lab.py architectures
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton · Dyna, an integrated architecture for learning, planning, and reacting](https://doi.org/10.1145/122344.122377)：模型学习、直接学习与模拟规划的经典接口；本页骨架受此启发，并明确另外加入平均奖励与 GVF。

- [Sutton, Bowling & Pilarski · The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173)：状态、预测、时间抽象与规划的研究纲领，不是完成全闭环的报告。

- [Sutton et al. · Reward-Respecting Subtasks / STOMP](https://arxiv.org/abs/2202.03466)：Subtasks、Options、Models、Planning 的学习链路及 reward-respecting 构造。

- [Barreto et al. · Successor Features for Transfer in Reinforcement Learning](https://arxiv.org/abs/1606.05312)：有限奖励特征、固定策略 SF 与 GPI 的经典机制桥梁。

- [Oak Lab · The OaK Architecture](https://oaklab.ai/posts/the-oak-architecture)：Richard Sutton 的架构研究纲领与讲座入口。

- [作者代码 · average-reward-methods](https://github.com/abhisheknaik96/average-reward-methods)：control_agents.py 中的直接学习与 planning_update，适合比较奖励率更新调度。

- [Elelimy et al. · Rethinking the Foundations for Continual RL](https://rlj.cs.umass.edu/2025/papers/Paper243.html)：RLC 2025：history process、deviation regret 与持续学习评价对象。

- [Mohamed et al. · The Cell Must Go On: Agar.io for Continual RL](https://arxiv.org/abs/2505.18347)：2025 年提出、后续修订的 AgarCL：完整环境与分解小游戏，讨论探索、记忆、信用分配和可塑性。

- [AgarCL 作者环境](https://github.com/machado-research/AgarCL)：C++ 仿真与 Python 接口；局部重生、观测和混合动作需按环境配置理解。

- [AgarCL 作者基线 · PPO 混合动作实现](https://github.com/machado-research/AgarCL-benchmark/blob/main/PPO_multi_heads_full_action.py)：同仓库包含 DQN_full_action_set.py、SAC_full_action_set.py 及 recurrent 版本；完整游戏实验需匹配动作、参数搜索与运行预算。

- [NeurIPS 2025 原文与补充材料入口](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0ab48777def88e73b50746a6011be0b0-Abstract-Conference.html)：算法 1–3、附录 A.3 的两个最优子程序假设及代码声明；未在本教材运行补充代码。

- [Option Keyboard 的经典桥梁](https://proceedings.neurips.cc/paper/2019/file/251c5ffd6b62cc21c446c963c76cf214-Paper.pdf)：cumulant 组合、GPE/GPI 与技能接口。

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html)：短期预测、终点价值、多任务协议。

- [作者实现](https://github.com/nicklashansen/tdmpc2)：训练、模型与 plan 函数分别阅读。

- [ICML 2025 原文](https://proceedings.mlr.press/v267/zhou25t.html)：正式发表入口；早期 ICLR 投稿页不能替代此状态。

- [作者项目代码](https://github.com/gaoyuezhou/dino_wm)：train.py、plan.py、数据与已公开模型检查点范围。

- [2025 首稿](https://arxiv.org/abs/2506.09985v1)：action-free 预训练、2-AC 后训练、真实规划与第 4.3 节限制；此处不赋予未核实会议状态。

- [Meta FAIR 官方实现](https://github.com/facebookresearch/vjepa2)：包含 V-JEPA 2、2-AC 和较新的 2.1；版本不能混用。

- [Touati & Ollivier · Learning One Representation to Optimize All Rewards](https://arxiv.org/abs/2103.07945)：FB 表示的理论出发点；探索/经验覆盖、近似误差及奖励查询约定。


---

# 实验设计：从更新正确到持续学习证据

一个算法通过测试、曲线更高，分别能说明什么？怎样用有限预算得到可以重复检验的结论？

## 本章内容

- 明确估计对象、独立随机单位、信息权限与资源预算。
- 完成开发选择、独立测试、整次运行配对 bootstrap 的可执行实验。
- 分别解释在线收益、冻结诊断、恢复、失败和跨任务推广。

<a id="problem-definition"></a>

## 本章的问题定义

确定有限数据能够支持哪种算法结论；实现一致、机制解释、独立比较和适用范围分别需要证据。

### 给定条件与符号

- 完整算法、封存配置、预定寿命与世界/任务分布。
- 开发与测试拆分、随机单位、基线、指标、资源和特权信息预算。

### 需要求解的对象

声明评价量的估计、差值与不确定性，以及与该估计范围匹配的结论。

### 信息与数据权限

学习器只见所选动作的已到达后果；实验控制器可管理种子与诊断，但变化时间、潜在未选奖励和测试排序不得回流到封存算法选择。

$$
\Delta_T=\mathbb E_\xi[X(A,h_A,\xi;T)-X(B,h_B,\xi;T)],\qquad X(A,h,\xi;T)=\frac1T\sum_{t=0}^{T-1}R_{t+1}
$$

$A,B$ 为完整算法，$h_A,h_B$ 为开发后封存的配置，$\xi$ 为运行随机性，$T$ 为寿命。目标是有限寿命在线平均差；无限奖励率、最佳调参器性能和最后冻结策略分数是其他量。整次运行是本章的独立单位。

### 成立条件与解的含义

- 独立测试不参与设计选择；配对需有合理共同外生随机性并保留各自闭环轨迹。
- 区间依赖独立单位和重采样协议；时间点不作为独立重复，缺失失败和非有限运行有预定处理。

判断准则：手算/有限差分/边界测试与所写更新一致；开发测试无重叠；整次运行差值及区间可重算，保留负差、失败和全部预算，结论限定于覆盖条件。

### 适用边界

- 测试通过不是算法长期有效或回报优势的证据。
- 开发集最高分和tiny有符号差不能直接称为已证优势。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 改变信息或数据协议 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：完整生命期控制比较要求独立世界、封存设计和可实现比较器。

- 组合不同学习问题 · [可塑性与特征更新](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)：可塑性假设需要匹配probe与干预，在线奖励曲线不足以定位原因。

- 组合不同学习问题 · [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)：冻结诊断、恢复和在线表现估计不同对象，副本评价不回流训练。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

研究者选择、时间依赖与信息不公平会使高曲线看似有优势，即使差异来自预算或噪声。

### 本章的核心思路

先固定估计对象与独立单位，再分离选择和测试；实现与机制证据单独连接到回报比较。

1. [隔离配置选择与性能估计](#lesson-derive)：因为开发赢家带有选择噪声，用独立测试评价封存配置；若要评价搜索器，需重复整个搜索。

2. [以完整生命期处理依赖](#lesson-randomness)：因为同曲线时间点共享参数和探索历史，整次运行作为独立单位，配对按允许的共同随机性定义。

3. [报告差值和范围而非赢家标签](#lesson-intervals)：因为有限样本有不确定性，对整次运行重采样并保留负差、失败与选择规则，再限制结论范围。

结论与条件：独立测试可避免复用选择噪声；bootstrap区间依赖数据和采样假设，不证明未知任务普适优势或机制因果性。

### 相关方法改变了什么

- 实现测试：核对更新与边界，不能替代环境收益实验。

- 受控机制实验：用干预与匹配资源检验具体解释，范围通常较窄。

- 独立寿命比较：估计封存算法的在线差值及不确定性，推广需追加未见条件。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 随机变量与样本均值

相同算法多次运行会产生不同收益。单次结果为 $X_i$，总体期望为 $\mu=\mathbb E[X]$，样本均值为 $\bar X=n^{-1}\sum_iX_i$。

### 在线学习

动作依据当前可得历史产生。环境反馈发生后，学习器才更新参数。测试一个持续学习算法时，学习本身通常仍然开启。

### 实验控制器

它管理随机分配、预算、记录与评价。环境提供反馈，学习器只接收协议允许的信息。统计分析可以读取真实变化时间，但不能将其回传给学习器。

<a id="lesson-setting"></a>

## 1 · 问题、估计对象与四级证据

考虑一个需要长期选择服务策略的系统。方法 A 启动慢，但后期表现好；方法 B 较早学会，却在环境变化后退化。最后一段收益、全程收益和冻结策略得分可能给出不同排序。实验需要先决定哪个问题重要，再决定记录与聚合方式。

$$
X(A,h,\xi;T)=\frac1T\sum_{t=0}^{T-1}R_{t+1},\qquad \mu(A,h;T)=\mathbb E_\xi[X(A,h,\xi;T)].
$$

$A$ 是完整学习算法，$h$ 是固定配置，$\xi$ 包含环境和学习器的随机性，$T$ 是预定寿命。这里估计有限寿命的在线平均奖励，不是已经证明存在的无限时域奖励率。

Patterson 等的 Empirical Design in Reinforcement Learning 将评价量、变异、超参数和研究者选择放在同一实验流程中。可以据此把证据分成四级。它们支持不同强度的主张，不能互相替代。

| 证据层级 | 核心问题 | 具体实验 | 能够支持的结论 |
| --- | --- | --- | --- |
| 更新与实现 | 计算是否符合所写算法？ | 手算、有限差分、边界、恢复运行 | 指定输入与条件下实现一致 |
| 机制 | 改动是否按假设改变学习过程？ | 控制变量、反例、等容量与等计算对照 | 在受控条件下支持特定解释 |
| 独立比较 | 收益差是否能在新运行重现？ | 锁定配置、独立重复、差值区间 | 指定分布和预算下的差异 |
| 适用范围 | 改变任务与时间尺度后是否仍成立？ | 未见任务、长寿命、资源压力测试 | 覆盖到的条件及已知失效边界 |

较小的 TD 误差是机制量。较高的行为奖励是控制结果。一个数值恒等式是实现证据。三者之间需要实验连接。例如减小步长可以减少更新尖峰，也可能让变化后的恢复更慢。仅展示更新范数下降，无法判断完整学习器是否更好。

<a id="lesson-derive"></a>

## 2 · 配置性能与选择过程的性能

固定配置的性能，与搜索后所得方法的性能，是不同统计对象。设开发数据用于从候选配置中选择一个配置。测试必须在选择完成后独立进行。否则同一批幸运噪声既帮某配置胜出，又被拿来报告其优势。

$$
\widehat h=\arg\max_{h\in\mathcal H}\overline X_{\rm dev}(h),\qquad \widehat\mu_{\rm test}=\frac1n\sum_{i=1}^n X(A,\widehat h,\xi_i^{\rm test};T).
$$

测试随机性必须独立于选择过程。该均值估计的是已选配置 $\widehat h$ 的性能。重复整个搜索流程的期望则为 $\Psi(B)=\mathbb E_{D_{\rm dev},\xi_{\rm search}}[\mu(A,\widehat h_B;T)]$，需要把搜索作为外层重复单位。

$$
\mathbb E\!\left[\max_h\overline X_{\rm dev}(h)\right]\geq\max_h\mathbb E[\overline X_{\rm dev}(h)].
$$

最大值运算保留了向上的噪声。这个不等式说明为何开发集上的最高均值通常过于乐观；独立测试消除重复使用这批选择噪声的问题，但不会证明选到了未知的全局最优配置。

| 数据层 | 允许的用途 | 禁止的信息流 |
| --- | --- | --- |
| 训练交互 | 每个 run 内选择动作和更新 | 读取未发生的奖励或未选择动作的反馈 |
| 开发与选择 | 查错、设计候选、选择参数 | 将反复查看的数据仍称为独立测试 |
| 独立测试 | 评价锁定配置的完整学习过程 | 依据测试排序再换参数、换指标或只加跑赢家 |

RL 的“测试”不必意味着冻结参数。若评价的是持续学习算法，测试 run 从初始化开始，按锁定的更新规则持续学习；研究者不再根据这批结果修改规则。若评价的是固定部署策略，才冻结策略参数。两种协议对应不同对象。

Jordan 等将参数设置或自适应机制纳入完整算法定义，关注新任务上的可用性与调参成本。Mesbahi 等进一步讨论 lifetime tuning：在整个未来寿命上反复搜索，可能掩盖长期适应困难。限制开发可见前缀是一种具体研究协议，前缀比例并没有统一最佳值。

本章实验只用开发生命期的早期前缀选择参数，再运行独立测试生命期的全部三个阶段。变化时间不提供给学习器。这个设计检验的是早期选择后能否适应后续变化；它不等价于已经评价了整个调参器的可靠性。

<a id="lesson-implementation"></a>

## 3 · 更新测试与梯度测试

实现测试应从能手算的对象开始。固定目标的标量回归足以检查误差符号、步长与更新顺序。随后再加入 bootstrap、重要性比、资格迹、目标网络和终止语义。每增加一种依赖，就增加一项能区分错误版本的测试。

$$
L(q)=\tfrac12(q-r)^2,\qquad \nabla_qL=q-r,\qquad q^+=q+\alpha(r-q).
$$

取 $q=0.5$、$r=1$、$\alpha=0.2$，应得到 $q^+=0.6$。本章 bandit 的更新就是这个式子，因此可以把手算、代码和梯度检查逐项对齐。

$$
g_j^{\rm FD}=\frac{L(\theta+\varepsilon e_j)-L(\theta-\varepsilon e_j)}{2\varepsilon}.
$$

中心差分应固定样本与随机噪声，并尝试多个 $\varepsilon$。过大有截断误差，过小有舍入误差。若检查 TD 半梯度，bootstrap 标签在两侧扰动中也必须固定；重新计算标签检查的是另一个梯度。

| 测试 | 本章实例 | 扩展到深度 RL |
| --- | --- | --- |
| 单步精确值 | 常数步长后为 0.6；递减式在 decay_scale=1、第二次奖励为 0 时得到 0.54 | 检查全部辅助变量的旧值与新值 |
| 退化条件 | 零步长不改变价值 | 零 trace、零重要性比、关闭正则 |
| 边界 | 运行结束、无恢复、失败后停止 | terminated、truncated、批末与自动重置 |
| 因果性 | 修改未来奖励不改变过去前缀 | 归一化、目标构造及元梯度不使用未来信息 |
| 统计接口 | 重复种子、拆分重叠、非有限值被拒绝 | 计划 run 与实际产物一一对应 |

测试通过说明实现满足这些检查。算法长期稳定、回报优势和迁移能力仍属于后续证据。反例也应保留：如果一种步长控制只在有限特征尺度下成立，超出条件后的失败能帮助读者理解其适用范围。

<a id="lesson-budget"></a>

## 4 · 信息权限与公平预算

相同环境步数回答样本效率问题。相同墙钟时间回答特定硬件上的计算效率问题。相同梯度步数只匹配部分优化成本。三种约束通常不能同时满足。主比较选定一个约束，再报告其他资源坐标。

| 资源或权限 | 需要声明的内容 | 典型混杂 |
| --- | --- | --- |
| 交互 | 原始环境步、帧跳过、恢复、额外评测 | 把一个 option 决策与一个原始动作视为等量数据 |
| 计算 | 网络前后向、规划、模型滚动、生成样本 | 更高更新比带来的收益归给新公式 |
| 内存 | 参数、优化器、trace、replay、模型、历史副本 | 只统计网络参数 |
| 调参 | 候选数、每个候选的 run 数、可见前缀 | 只报告胜出配置的训练成本 |
| 特权信息 | 任务 ID、真实状态、变化时间、真模型 | 把信息优势归给学习能力 |

本实验的两个方法都保持两个动作价值和两个计数，每步只使用一个实际反馈。候选集合、探索概率、初始化、开发种子及选择目标相同。差别是步长是否随某个动作的访问次数递减。环境控制器可以生成所有动作的潜在奖励，但只把所选动作的奖励交给学习器。

标准运行每个方法使用三个候选、八个开发 run、每个前缀 240 步，共 5760 次开发交互。独立测试为 24 个 run，每个 900 步，共 21600 次测试交互。环境配对不减少实际部署成本；它只是模拟实验中的方差控制。日志与奖励表属于实验控制器，不能据此声称整个研究程序只有常数内存。

<a id="lesson-randomness"></a>

## 5 · 完整生命期作为随机单位

同一学习曲线中的相邻奖励相互依赖。它们共享参数、探索历史与先前经验。因此 900 个时间点不等于 900 个独立算法样本。这里一次从初始化到结束的完整生命期是一个样本。开发 run 和测试 run 使用互不重叠的种子集合。

仅给两个算法相同的整数 seed，不能保证它们面对相同随机事件。不同控制流可能消耗不同数量的随机数。本实验按时间和动作预先生成外生 Bernoulli 奖励表；按时间再生成固定三个动作随机数。不同方法在同一时间选择同一动作，会得到同一潜在反馈。方法间仍可选择不同动作和形成不同历史。

$$
R_{t+1}(a)=\mathbf1[U_{t,a}<p_t(a)],\qquad U_{t,a}\overset{\rm iid}{\sim}\operatorname{Uniform}(0,1).
$$

一个配对共享同一张 $U$ 表及同一组动作随机数。不同配对独立生成。共享外生情景保持各方法自己的边际分布；它不使两个方法的轨迹相同，也不保证差值方差一定降低。

环境的潜在反馈与学习器分开；学习器只接收所选动作的奖励。

```python
@dataclass(frozen=True)
class Scenario:
    seed: int
    # Controller-only potential outcomes; the learner never receives this table.
    rewards: tuple[tuple[float, float], ...]
    action_uniforms: tuple[tuple[float, float, float], ...]


def make_scenario(seed: int, horizon: int) -> Scenario:
    if horizon < 3 or horizon % 3:
        raise ValueError("horizon must be a positive multiple of three")
    env_rng, action_rng = rng_for(seed, "environment"), rng_for(seed, "actions")
    rewards, uniforms = [], []
    phase_length = horizon // 3
    for t in range(horizon):
        probabilities = (0.2, 0.8) if t // phase_length == 1 else (0.8, 0.2)
        rewards.append(tuple(float(env_rng.random() < p) for p in probabilities))
        # Exactly three action uniforms per time index, regardless of branching.
        uniforms.append(tuple(action_rng.random() for _ in range(3)))
    return Scenario(seed, tuple(rewards), tuple(uniforms))


class Learner:
    def __init__(self, config: Config):
        self.config = config
        self.q = [0.5, 0.5]
        self.counts = [0, 0]

    def choose(self, uniforms):
        explore, arm, tie = uniforms
        if explore < self.config.epsilon:
            return min(1, int(2 * arm))
        best = max(self.q)
        choices = [a for a, value in enumerate(self.q) if value == best]
        return choices[min(len(choices) - 1, int(tie * len(choices)))]

    def update(self, action, reward):
        self.counts[action] += 1
        step = self.config.alpha
        if self.config.method == "decay":
            step /= 1 + (self.counts[action] - 1) / self.config.decay_scale
        self.q[action] += step * (reward - self.q[action])
        if not all(math.isfinite(value) for value in self.q):
            raise FloatingPointError("non-finite value estimate")
```

$$
D_i=X_{A,i}-X_{B,i},\quad \widehat\Delta=\frac1n\sum_iD_i,\quad \operatorname{Var}(D)=\operatorname{Var}(X_A)+\operatorname{Var}(X_B)-2\operatorname{Cov}(X_A,X_B).
$$

配对的价值取决于两个方法结果的协方差。正相关可能降低差值方差。配对规则须在看结果前固定；不能事后挑选看起来最有利的匹配。

<a id="lesson-intervals"></a>

## 6 · 配对 bootstrap 与区间含义

把每个完整 run 汇总成预定主指标，得到成对分数。一次 bootstrap 从这些配对索引中有放回抽取与原样本相同数量的索引，再计算差值均值。重复这个过程，用重采样分布的分位数构造 percentile 区间。

$$
I_1^{(b)},\ldots,I_n^{(b)}\overset{\rm iid}{\sim}\operatorname{Uniform}\{1,\ldots,n\},\qquad\widehat\Delta^{(b)}=\frac1n\sum_{j=1}^nD_{I_j^{(b)}}.
$$

95% percentile 区间取这些均值的 2.5% 与 97.5% 分位数。重采样的是整对 run；不独立重采样同一曲线中的时间点，也不分别打散两个方法的配对。

运行级配对重采样，返回差值均值、区间与配对差的标准误。

```python
def paired_bootstrap(first, second, reps=2000, seed=917, confidence=0.95):
    """Percentile CI for a fixed pair of configurations, resampling whole runs."""
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("at least two aligned run pairs are required")
    if reps < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    finite(first + second)
    differences = [a - b for a, b in zip(first, second)]
    rng = random.Random(seed)
    n = len(differences)
    samples = [statistics.fmean(differences[rng.randrange(n)] for _ in range(n))
               for _ in range(reps)]
    tail = (1 - confidence) / 2
    return {"mean_difference": statistics.fmean(differences),
            "ci": [quantile(samples, tail), quantile(samples, 1 - tail)],
            "standard_error": statistics.stdev(differences) / math.sqrt(n),
            "pairs": n, "bootstrap_replicates": reps,
            "interval": "run-paired percentile; pointwise, conditional on selection"}
```

置信区间描述估计程序的不确定性，不是下一次运行的分数范围。标准差与经验分位数描述运行差异。很窄的均值区间可以与很不稳定的单次表现同时存在。Bootstrap 次数只控制重采样的数值精度；它不会增加真实 run 数，也无法创造尚未观察到的稀有失败。

本章只有一个构造环境。少量 run 的区间是教学性的近似，不保证标称覆盖率精确成立。Agarwal 等的 rliable 面向多任务基准，使用任务内运行重采样和 IQM 等汇总。固定任务集通常保持任务不变；要推广到新任务总体，还需定义任务抽样层级。运行数不同的任务也不能无说明直接摊平。

| 输出 | 回答的问题 | 不直接回答 |
| --- | --- | --- |
| 均值差及区间 | 锁定配置的平均收益差 | 哪个方法在每次运行都更好 |
| 全部 run 分数 | 波动、多峰与尾部表现 | 未知总体的完整失败模式 |
| IQM 或性能剖面 | 指定任务混合上的稳健表现 | 任意未见任务上的优势 |
| 逐时间点区间 | 特定时刻均值的不确定性 | 整条曲线同时有 95% 覆盖 |

<a id="lesson-example"></a>

## 7 · 可复算的非平稳 bandit

环境分三个等长阶段。两个动作的成功概率依次是 (0.8,0.2)、(0.2,0.8)、(0.8,0.2)。奖励为 0 或 1。智能体在阶段切换时不重置，也不接收阶段编号。两个价值都初始化为 0.5，采用探索概率 0.1 的 epsilon-greedy；并列最大值随机选择。

$$
Q_n(a)=Q_{n-1}(a)+\alpha_n[R_n-Q_{n-1}(a)],\qquad \alpha_n^{\rm constant}=\alpha_0,\quad\alpha_n^{\rm decay}=\frac{\alpha_0}{1+(n-1)/20}.
$$

$n$ 是所选动作含本次在内的访问次数，$R_n$ 是该动作第 $n$ 次被选择时的奖励。每个方法均从 $\alpha_0\in\{0.05,0.15,0.4\}$ 选择。递减式不是样本均值公式；它是一个有明确速度参数的对照更新。

常数步长将旧奖励的权重按几何速度衰减。若某个动作持续被采样，旧估计的贡献在后续若干次更新后越来越小。递减步长则使后期新反馈影响更弱。这给出“适应较慢”的机制假设，但控制结果还取决于探索是否重新访问已变好的动作。

$$
Q_n=(1-\alpha)^nQ_0+\alpha\sum_{j=1}^n(1-\alpha)^{n-j}R_j.
$$

这是同一动作按常数步长更新的精确展开。它描述访问次数下的记忆长度；若策略很少再选该动作，真实时间中的恢复仍可能很慢。

手算一次配对分析：三个独立生命期得到 A=[0.7,0.5,0.6]、B=[0.6,0.6,0.4]。差值为 [0.1,−0.1,0.2]，均值为 1/15。一次索引重采样 [3,3,1] 得到均值 1/6；[2,2,1] 得到 −1/30。重采样变动的是完整运行的组成，而非重新运行算法。

| tiny 命令的输出 | constant | decay |
| --- | --- | --- |
| 开发选择的初始步长 | 0.05 | 0.05 |
| 独立测试全程平均收益 | 0.595833 | 0.562500 |
| 第一阶段平均收益 | 0.700000 | 0.712500 |
| 反转阶段平均收益 | 0.504167 | 0.300000 |
| 恢复原概率阶段平均收益 | 0.583333 | 0.675000 |

tiny 模式使用四个开发 run、八个独立测试 run，每次 90 步。选择只看开发的前 30 步。固定种子得到平均差 0.033333，500 次配对 bootstrap 的区间约为 [−0.002118,0.060451]。区间包含零，因此这个小例子没有确定平均差的符号。第三阶段又出现递减方法较高的结果，显示全程均值之外还存在阶段取舍。

这些数字是可执行示例的输出。它们不支持深度网络或任意 CRL 环境的结论。公开固定种子有利于学习和复现；读者若据此反复修改算法，这批种子便成为开发数据，新的确认实验需要另外预定的运行。

<a id="lesson-code"></a>

## 8 · 选择、测试与结果保存

选择函数不接收测试数据；两种配置确定后才开始测试，所有候选和逐 run 指标均保留。

```python
def select_on_development(method, candidates, dev_seeds, horizon, prefix):
    """No test seeds, test outcomes or test score are accepted by this function."""
    unique_seeds(dev_seeds)
    if not 1 <= prefix <= horizon or not candidates:
        raise ValueError("invalid development budget")
    records = []
    for alpha in candidates:
        config = Config(method, alpha)
        runs = [run_lifetime(make_scenario(seed, horizon), config, limit=prefix)
                for seed in dev_seeds]
        records.append({"alpha": alpha, "score": statistics.fmean(map(mean_reward, runs)),
                        "failures": sum(run.status == "failed" for run in runs)})
    # Stable candidate order is the predeclared tie-break; no test-set tie-break.
    chosen = max(range(len(records)), key=lambda i: records[i]["score"])
    return Config(method, candidates[chosen]), records


def experiment(dev_seeds, test_seeds, horizon=900, prefix=240,
               candidates=(0.05, 0.15, 0.4), window=30, reps=2000, raw=False):
    unique_seeds(dev_seeds)
    unique_seeds(test_seeds)
    if set(dev_seeds) & set(test_seeds):
        raise ValueError("development and test seeds must be disjoint")
    if (len(test_seeds) < 2 or horizon % 3 or not 1 <= window <= horizon // 3
            or (horizon // 3) % window):
        raise ValueError("invalid test design")
    selected, development = {}, {}
    for method in ("constant", "decay"):
        selected[method], development[method] = select_on_development(
            method, candidates, dev_seeds, horizon, prefix)
    # Lock both configurations before any test scenario is evaluated.
    test_runs = {method: [] for method in selected}
    for seed in test_seeds:
        scenario = make_scenario(seed, horizon)
        for method, config in selected.items():
            test_runs[method].append(run_lifetime(scenario, config))
    records = {method: [run_metrics(run, window) for run in runs]
               for method, runs in test_runs.items()}
    comparison = paired_bootstrap(
        [mean_reward(run) for run in test_runs["constant"]],
        [mean_reward(run) for run in test_runs["decay"]], reps=reps)
    summary = {method: {
        "mean_lifetime_rate": statistics.fmean(item["lifetime_rate"] for item in rows),
        "mean_phase_rates": [statistics.fmean(item["phase_rates"][p] for item in rows)
                             for p in range(3)],
        "failures": sum(item["status"] == "failed" for item in rows),
        "recovery_counts": [{status: sum(item["recovery"][p]["status"] == status
                                         for item in rows)
                             for status in ("recovered", "right_censored", "failed")}
                            for p in range(2)]} for method, rows in records.items()}
    report = {
        "protocol": {"horizon": horizon, "development_prefix": prefix,
                     "dev_seeds": dev_seeds, "test_seeds": test_seeds,
                     "candidates": list(candidates), "epsilon": 0.1,
                     "decay_scale": 20, "window": window,
                     "pairing": "shared time-indexed potential rewards and action uniforms",
                     "estimand": "selected-configuration lifetime utility; not HPO reliability",
                     "failure_utility": "observed reward followed by zero after stopped service",
                     "development_interactions_per_family": len(candidates) * len(dev_seeds) * prefix,
                     "test_interactions_per_family": len(test_seeds) * horizon},
        "selected": {method: asdict(config) for method, config in selected.items()},
        "development_scores": development, "test_summary": summary,
        "constant_minus_decay": comparison, "per_run": records}
    if raw:
        report["raw_runs"] = {method: [asdict(run) for run in runs]
                              for method, runs in test_runs.items()}
    return report
```

行为收益在更新前记录。原始观测与失败后的部署效用分开保存。

```python
def run_lifetime(scenario: Scenario, config: Config, limit=None,
                 freeze_at=None, fault_at=None) -> Run:
    horizon = len(scenario.rewards) if limit is None else limit
    if not 1 <= horizon <= len(scenario.rewards):
        raise ValueError("invalid observation budget")
    for boundary in (freeze_at, fault_at):
        if boundary is not None and not 0 <= boundary < horizon:
            raise ValueError("boundary outside run")
    learner = Learner(config)
    observed, actions, failure = [], [], None
    for t in range(horizon):
        action = learner.choose(scenario.action_uniforms[t])
        reward = scenario.rewards[t][action]
        # Score behavior before learning. Unchosen rewards stay controller-only.
        observed.append(reward)
        actions.append(action)
        try:
            if t == fault_at:
                raise FloatingPointError("injected diagnostic fault")
            if freeze_at is None or t < freeze_at:
                learner.update(action, reward)
        except FloatingPointError as error:
            failure = {"after_step": t + 1, "reason": str(error),
                       "injected": fault_at is not None}
            break
    # Predeclared deployment utility: stopped service earns 0 for remaining time.
    # This tail is NOT observed reward; retain both arrays and the failure event.
    utility = tuple(observed + [0.0] * (horizon - len(observed)))
    # JSON has no non-finite numbers. Preserve the failure event and use null for
    # invalid final parameters; actual finite rewards and scores remain intact.
    final_q = tuple(value if math.isfinite(value) else None for value in learner.q)
    return Run(scenario.seed, config, horizon, tuple(observed), tuple(actions),
               utility, final_q, tuple(learner.counts),
               "failed" if failure else "complete", failure)
```

Python 3.10+，仅标准库。所有输出写到标准输出，不创建或覆盖文件。

```sh
python experiment_design_lab.py test
python experiment_design_lab.py demo --tiny
python experiment_design_lab.py demo
python experiment_design_lab.py demo --raw
```

test 检查手算、有限差分、零步长、未来数据隔离、拆分重叠、相同算法配对、恢复删失、失败记录及冻结副本。demo 的标准配置是 900 步、八个开发 run 和 24 个测试 run。--raw 额外输出每步实际奖励、动作、效用与最终学习状态，便于重新计算曲线。

报告同时包含开发候选得分、选择结果、测试种子、每个 run 的分段与窗口收益，以及主差值区间。失败示范通过单独的故障注入演示停机处理，明确标为 diagnostic_only_not_in_primary_comparison，不参与正常两方法的性能比较。

<a id="lesson-curves"></a>

## 9 · 曲线、失败与恢复

全程收益包含初始探索、变化后的低谷和重新学习。阶段均值帮助定位变化。最低窗口与原始曲线暴露平均值掩盖的崩溃。窗口宽度及步长应在查看结果前确定，平滑仅用于显示，不能据此改变主统计量。本章程序要求窗口长度整除每个阶段的长度，使窗口等长且不跨越变化边界；不将较短的尾窗与完整窗口混合取最低值。

$$
\tau_{\rm confirm}=\min\{kW:k\geq L,\ \bar R_j\geq c\ \text{for all }j\in\{k-L+1,\ldots,k\}\}.
$$

从变化时刻开始按长度 $W$ 的不重叠窗口计数。连续 $L$ 个窗口达到阈值 $c$ 后，在第 $k$ 个窗口末确认恢复。本实验用 $c=0.7$、$L=2$。这是观测收益的操作性指标，受奖励噪声影响。

下一次变化或观察期结束前未达标，记录 right_censored，并报告截至该时刻的恢复比例。不能只平均成功恢复者的时间。运行中崩溃或资源耗尽则是失败事件，不能自动当作与性能无关的删失。tiny 模式中，反转阶段两方法分别只有 1/8 和 0/8 达到确认条件，这些未恢复 run 全部保留。

若更新恰好在阶段最后一步失败，且此前未确认恢复，仍记为失败，而不是因为奖励数组长度刚好填满而记为删失。确认依据的是更新前已记录的奖励；已发生的首次确认不会被之后的故障抹去，但运行级失败状态始终单独保留。

本 bandit 的奖励非负，并把停机后不再提供服务的收益定义为零。程序分别保存已观测奖励和补齐预算后的效用，记录失败时刻与原因。对任意真实任务，填零不一定合理；若停机有成本或奖励无自然下界，应预先定义失败代价，或提供多个合理代价下的敏感性分析。

单个运行的日志窗口不增加独立重复数。需要整条曲线的区间时，以 run 为单位重采样整条轨迹。大量时刻逐点查看显著性会增加偶然发现；主指标与主要比较在实验前确定，其他曲线承担描述与诊断作用。

<a id="lesson-freeze"></a>

## 10 · 在线评价与冻结诊断

在线主评价问：这个学习过程在真实经历中获得了多少回报？冻结诊断问：从同一个历史起点出发，接下来继续更新是否有帮助？在可复制模拟器中，可以复制学习器与环境状态，再分别保持更新和冻结参数。诊断交互不回灌主生命期。

| 需要隔离的对象 | 正确做法 | 改变它会引入的混杂 |
| --- | --- | --- |
| 网络参数与优化器 | 声明哪些更新被停止 | 冻结权重但继续改变归一化仍有适应 |
| 循环记忆 | 通常允许其随观测递推 | 清空记忆同时删除了历史信息 |
| 环境与随机数流 | 复制环境或使用匹配独立生命期 | 副本评测消耗主 RNG 会改变主轨迹 |
| replay 与统计 | 副本独立维护或明确冻结 | 评测样本进入主 buffer 增加训练数据 |

本章 bandit 没有循环状态和归一化。run_lifetime 的 freeze_at 仅停止价值和计数更新，动作仍按当前价值与相同的探索规则产生。测试比较冻结前缀和独立重建的同一前缀，并检查主运行计数未受影响。这是较简单的冻结语义；深度 agent 需要列出更多持久状态。

冻结后变差，支持该分支时段保持更新有益。冻结后不变差，可能因为策略已足够好、环境未再挑战它，或学习器本来就没有学会。旧任务回访也只是诊断：若真实世界不能重新访问旧情境，完整遗忘矩阵无法仅凭单条在线轨迹识别。

<a id="lesson-branches"></a>

## 11 · 从小实验进入研究

| 研究问题 | 保持不变 | 有针对性的扩展 |
| --- | --- | --- |
| 追踪能力 | 数据权限与总交互数 | 改变反转频率、幅度和奖励噪声 |
| 探索与记忆的分工 | 价值更新规则 | 固定行为流与闭环行为对照 |
| 部分可观测性 | 环境动态和奖励 | 隐去线索、延迟线索，比较记忆模型 |
| 元学习的作用 | 元训练成本及开发权限 | 固定步长、在线自适应、额外计算对照 |
| 深度可塑性 | 容量、梯度预算和探针数据 | aged/fresh、随机替换、只重置优化器 |
| 可用性与调参 | 总搜索预算 | 重复整个选择过程，报告所得独立性能 |

进入多任务 benchmark 前，写出任务权重与归一化规则。一个任务的高分单位不能任意支配总均值。rliable 的矩阵接口按运行×任务组织数据；IQM 去掉上下各 25% 的分数质量，故尾部失败仍需单列。库提供统计实现，实验对象与重采样层级仍由研究问题决定。

一份可复核交付至少包括：问题与指标、环境版本和信息权限、完整算法与依赖、候选与选择规则、所有计划 run 的终态、原始曲线、分析入口、主差值与区间，以及不支持优势的条件。结论应落在这组证据覆盖的范围内。

<a id="lesson-check"></a>

## 12 · 习题与检查

- 把开发前缀从 240 步改为全寿命，需要改变哪句结论？选择器已看到后期变化，估计对象从前缀选择后的持续适应变为完整寿命调参后的性能。
- 同一策略测试 100 个回合，可以称作 100 次独立算法运行吗？它们主要描述同一训练产物的执行噪声，不能替代独立训练。
- 为什么 tiny 区间跨零不能证明两方法等价？区间也包含有实际意义的正差异；等价需要预定容忍范围和相应分析。
- 把八次生命期切成 24 个阶段后做 bootstrap 会怎样？共同参数和历史造成相关，独立样本数被夸大。
- 在测试后给失败方法加大搜索范围，还能沿用原区间吗？新方法已利用原测试信息，需要另外确认。

动手题：保持测试集不变，仅用不同开发 seed 组重复选择，观察所选步长怎样变化。此时这些测试分数可用来探索选择器不稳定性，但新的主张仍需重新设计外层独立重复。另做固定行为随机动作对照，判断常数步长的变化优势来自更快估计，还是来自估计与探索的反馈。

## 学习与研究衔接

从精确测试到受控学习曲线，再到多运行基准。每一层支持不同强度的结论。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-experiments) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=experiments) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=experiments)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

状态是支持后续计算的内部信息；GVF 指定一个预测问题；RTRL 和资格迹规定信用如何传播。三者可以组合，但不是相互替代的算法名称。先理解给定策略的预测，再讨论预测怎样改善控制。

- [Recurrent Reinforcement Learning with Memoroids](https://yingwen.io/zh/continual-rl/research/#recent-memoroids-sequence-learning)
- [Does Zero-Shot Reinforcement Learning Exist?](https://yingwen.io/zh/continual-rl/research/#recent-zero-shot-forward-backward)

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [Recurrent Reinforcement Learning with Memoroids](https://yingwen.io/zh/continual-rl/research/#recent-memoroids-sequence-learning)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Revisiting Adam for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-revisiting-streaming-adam)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [Rethinking the Foundations for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-rethinking-crl-foundations)
- [An Empirical Study of Deep Reinforcement Learning in Continuing Tasks](https://yingwen.io/zh/continual-rl/research/#recent-continuing-task-deep-study)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [Understanding Plasticity in Neural Networks](https://yingwen.io/zh/continual-rl/research/#recent-understanding-plasticity)
- [Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn](https://yingwen.io/zh/continual-rl/research/#recent-c-chain-churn)
- [Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline](https://yingwen.io/zh/continual-rl/research/#recent-reset-and-distill)
- [Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-vla-simple-recipe)
- [Parseval Regularization for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-parseval-continual-geometry)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Discovering state-of-the-art reinforcement learning algorithms](https://yingwen.io/zh/continual-rl/research/#recent-disco-rl)
- [How Should We Meta-Learn Reinforcement Learning Algorithms?](https://yingwen.io/zh/continual-rl/research/#recent-meta-algorithm-search-comparison)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [Position: Lifetime tuning is incompatible with continual reinforcement learning](https://yingwen.io/zh/continual-rl/research/#recent-lifetime-tuning)
- [The Cell Must Go On: Agar.io for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-agarcl)
- [OGBench: Benchmarking Offline Goal-Conditioned RL](https://yingwen.io/zh/continual-rl/research/#recent-ogbench-goal-evaluation)
- [Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-vla-simple-recipe)
- [Revisiting Adam for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-revisiting-streaming-adam)
- [An Empirical Study of Deep Reinforcement Learning in Continuing Tasks](https://yingwen.io/zh/continual-rl/research/#recent-continuing-task-deep-study)
- [How Should We Meta-Learn Reinforcement Learning Algorithms?](https://yingwen.io/zh/continual-rl/research/#recent-meta-algorithm-search-comparison)
- [Recurrent Reinforcement Learning with Memoroids](https://yingwen.io/zh/continual-rl/research/#recent-memoroids-sequence-learning)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文提供定义和条件，架构讲座提出模块组织，算法论文检验特定机制。完整系统还要明确智能体、外部设计者和世界各自承担的工作；组件成立不自动意味着组合后的长期收益成立。

- [Rethinking the Foundations for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-rethinking-crl-foundations)

### OGBench: Benchmarking Offline Goal-Conditioned RL

Seohong Park, Kevin Frans, Benjamin Eysenbach, Sergey Levine

ICLR 2025 · 2025 · 评价与实验协议

#### 研究问题

一个目标条件算法表现不好，是长时域、轨迹拼接、视觉表示还是随机性造成的？

#### 关键机制

OGBench 用不同环境类型与数据集分别施加这些困难，并提供统一的目标条件基线。固定离线数据让算法面对相同经验，从而将学习机制的差异与在线探索能力的差异暂时分离。

#### 证据

论文提供八类环境、八十五个数据集和六类算法实现。价值在于可复用的实验接口与困难分解，而不只是汇总一个排行榜。

#### 条件与限制

固定数据不检验智能体如何主动获得未来经验，也不直接检验单次生命的灾难性变化、恢复或长期资源管理。它适合 CRL 子问题实验，不是完整 CRL 的替代品。

#### 阅读与实验

先选择只改变一种困难的两个数据集，再比较 HIQL 与平坦目标策略。把观察到的差异写成可检验机制假设，而不是直接归因于“层次更好”。

#### 原文与相关入口

- [论文](https://arxiv.org/abs/2410.20092)：ICLR 2025；环境、数据与基线定义。
- [作者基准库](https://github.com/seohongpark/ogbench)：数据获取、环境与统一算法实现。

#### 作者代码

[基准作者维护的官方实现。](https://github.com/seohongpark/ogbench)

离线目标环境、数据集与标准化基线。

### Understanding Plasticity in Neural Networks

Clare Lyle, Zeyu Zheng, Evgenii Nikishin, Bernardo Avila Pires, Razvan Pascanu, Will Dabney

ICML 2023 · 2023 · 支持方法与理论

#### 研究问题

学习变慢一定意味着网络已饱和或特征秩下降吗？

#### 关键机制

论文通过新目标拟合实验研究可塑性，并分析优化几何与曲率的影响。某些表示统计与学习能力下降会同时出现，却不是所有设置中的充分解释。评价对象从“网络看起来是否健康”转向“在受控更新预算内还能学会什么”。

#### 证据

受控探针与 RL 实验展示了不同机制之间的区别，并检验网络设计和优化过程的作用。它为可塑性研究提供诊断方式，而不是单一通用修复算法。

#### 条件与限制

探针目标、优化器和步数会改变测得的可塑性。相关性不等于所有控制任务中的因果机制；探针训练也不能写回被评价的在线智能体。

#### 阅读与实验

复制同一个检查点，在副本上拟合两类新目标。保持训练预算一致，并报告探针过程与真实环境回报之间的区别。

#### 原文与相关入口

- [ICML 2023 原文](https://proceedings.mlr.press/v202/lyle23b.html)：可塑性探针、优化几何与诊断边界。

### Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn

Hongyao Tang, Johan Obando-Ceron, Pablo Samuel Castro, Aaron Courville, Glen Berseth

ICML 2025 · 2025 · 直接研究持续学习

#### 研究问题

一次局部更新为什么会在其他输入上引发大幅预测变化，并损害后续学习？

#### 关键机制

C-CHAIN 抑制相对于近期参考网络的函数输出变化，降低一次更新在其他样本上造成的 churn。论文把该现象与经验神经切线核及学习动力学联系起来。正则化对象是函数变化，不是直接把所有参数锁在旧值附近。

#### 证据

作者在持续 Gym Control、ProcGen、DMC 和 MinAtar 序列中比较，并提供对应环境和算法代码。

#### 条件与限制

近期函数稳定性不等于长期任务知识保留；参考样本和参考网络也占资源。若环境突然发生真实变化，过强抑制输出变化可能延迟必要适应。

#### 阅读与实验

将 churn 按旧分布、新分布分别计算，并同时画适应速度。这样才能区分“减少无关干扰”和“阻止有用改变”。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/tang25g.html)：机制、理论分析与持续实验。
- [作者代码](https://github.com/bluecontra/C-CHAIN)：四类持续环境的基线和 C-CHAIN 对照实现。

#### 作者代码

[作者仓库，README 说明依赖的 TRAC、CleanRL 与 MinAtar 基础实现。](https://github.com/bluecontra/C-CHAIN)

持续控制环境与 C-CHAIN 对照实验。

### Revisiting Adam for Streaming Reinforcement Learning

Florin Gogianu, Luțu Adrian-Cătălin, Razvan Pascanu

RLC 2026 / RLJ 预会议版 · 2026 · 支持方法与理论

#### 研究问题

流式 RL 的不稳定来自 Adam 本身，还是目标导数、方差与超参数的组合？

#### 关键机制

论文重新分析自适应更新的信噪比，将 Adam 的稳定项与目标导数尺度联系起来，并研究有界导数的回报分布学习及多步更新。它改变的是目标与更新的配合，而非简单沿用批量训练时的默认配置。

#### 证据

作者在大规模 Atari 流式实验中展示了具有竞争力的结果，并重新比较早期流式方法。正式 RLJ 入口收录为 RLC 2026 预会议论文。

#### 条件与限制

主体实验采用经典回合式 Atari 的流式学习协议，不是任意非平稳终生适应的证据。这些结果也不否定归一化、资格迹或更新约束在其他任务中的价值。版本、调参预算和目标分布必须对齐。

#### 阅读与实验

建立二维对照：固定目标换优化器，固定优化器换目标。将调参种子与最终测试分开，再判断改进来自哪一个因素。

#### 原文与相关入口

- [RLC 2026 论文入口](https://rlj.cs.umass.edu/2026/papers/Paper131.html)：会议收录信息与论文。
- [作者预印本](https://arxiv.org/abs/2605.06764)：Adam 尺度分析、回报分布目标与实验协议。

### Discovering state-of-the-art reinforcement learning algorithms

Junhyuk Oh, Gregory Farquhar, Iurii Kemaev, Dan A. Calian, Matteo Hessel, Luisa Zintgraf, Satinder Singh, Hado van Hasselt, David Silver

Nature · 2025 · 支持方法与理论

#### 研究问题

除了学习策略，能否从大量学习过程里学出更有效的 RL 更新规则？

#### 关键机制

DiscoRL 用外层优化评价执行若干内层更新后的行为表现，学习价值、策略与辅助预测之间的更新方式。被训练的对象是学习算法本身，而不仅是某个任务的策略参数。内外两层有各自的数据、时间尺度与计算预算。

#### 证据

论文报告跨环境发现更新规则与迁移到未见环境的结果，并公开配套算法实现。它展示了自动算法发现的可能性，但依赖大规模外层训练。

#### 条件与限制

外层在大量环境和设备上的搜索属于设计者侧资源，不能记作测试智能体单次生命内的自主学习。公开规则的执行成本与发现该规则的成本应分别报告。

#### 阅读与实验

画出内层参数和外层参数的更新依赖，再列出测试时哪些量被冻结。与在线 IDBD 比较时，先区分跨任务算法发现和单流步长追踪。

#### 原文与相关入口

- [Nature 原文](https://www.nature.com/articles/s41586-025-09761-x)：算法发现过程、外层资源与泛化实验。
- [作者实现](https://github.com/google-deepmind/disco_rl)：配套代码与发现的更新规则。

#### 作者代码

[Google DeepMind 的论文配套仓库。](https://github.com/google-deepmind/disco_rl)

DiscoRL 配套实现与学习到的更新规则；具体训练资源以仓库说明为准。

### Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline

Hongjoon Ahn, Jinu Hyeon, Youngmin Oh, Bosun Hwang, Taesup Moon

ICLR 2025 · 2025 · 直接研究持续学习

#### 研究问题

一个网络还能拟合新目标，为什么先前训练仍可能让它在新任务上学得更慢？

#### 关键机制

论文把任务之间的负迁移与一般可塑性损失区分开。Reset & Distill 在新任务开始时重置在线 actor 和 critic，避免旧初始化阻碍学习；随后离线蒸馏当前策略与旧专家的动作分布以整合知识。适应和保留通过不同过程实现。

#### 证据

作者在控制与游戏任务中分析负迁移，并在长 MetaWorld 序列上检验该基线。原文直接提供实现地址。

#### 条件与限制

任务边界、在线网络重置、旧专家和离线蒸馏都需要资源。它不能直接当作无边界、不能重置、禁止回放的单次生命方案。

#### 阅读与实验

除了与连续微调比较，还要与同等预算的从头训练比较。若新任务表现低于从头训练，先检查负迁移，再判断是否属于单纯容量损失。

#### 原文与相关入口

- [ICLR 2025 原文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ba9e3d60610f3525717665966d86e0cd-Abstract-Conference.html)：负迁移诊断、Reset & Distill 机制与边界。
- [原文代码入口](https://github.com/hongjoon0805/Reset-Distill)：论文首页提供的作者实现。

#### 作者代码

[ICLR 正式论文首页明确链接的代码。](https://github.com/hongjoon0805/Reset-Distill)

Reset & Distill 以及任务序列实验。

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

### Position: Lifetime tuning is incompatible with continual reinforcement learning

Golnaz Mesbahi, Parham Mohammad Panahi, Olya Mastikhina, Steven Tang, Martha White, Adam White

ICML 2025 Position Paper · 2025 · 评价与实验协议

#### 研究问题

如果设计者用完整未来生命反复调参，实验还在测智能体面对未知变化的能力吗？

#### 关键机制

论文限制调参可访问的生命阶段，并比较这种选择方式与利用完整生命回报挑选配置的差异。外部设计者掌握未来变化信息，可能使一个并不自适应的固定算法显得适应良好。核心改变发生在评价协议，而不是 TD 更新公式。

#### 证据

作者用持续、非平稳设置中的深度 RL 实验说明超参数选择可以改变方法比较。该工作属于立场论文，论证与示例用于推动更符合问题目标的评价。

#### 条件与限制

允许多少开发阶段经验需要按应用规定，不存在由该论文推出的普适固定比例。仅限制时间前缀也不能替代独立测试种子和计算预算控制。

#### 阅读与实验

对同一配置集合分别按开发前缀和完整生命选择超参数，再在独立测试生命上比较。报告两种选择使用了哪些未来信息。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/mesbahi25a.html)：调参协议、论证与示例实验。

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

### Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning

Jiaheng Hu, Jay Shim, Chen Tang, Yoonchang Sung, Bo Liu, Peter Stone, Roberto Martín-Martín

RLC 2026 · 2026 · 直接研究持续学习

#### 研究问题

大规模预训练的视觉—语言—动作模型，是否仍需要复杂机制才能顺序学习控制任务？

#### 关键机制

论文研究对预训练 VLA 进行顺序强化学习，并以低秩适配等相对简单的训练流程检验持续学习。预训练表示、可更新参数子空间和 RL 目标共同决定迁移与遗忘，不能只把结果归因于单一保留正则项。

#### 证据

作者在多种 VLA 与长期任务基准上比较，并提供实验代码。RLJ 的 RLC 2026 论文页与论文脚注分别给出正式入口和作者仓库。

#### 条件与限制

预训练数据和算力属于外部资源，任务与重置协议也影响难度。该结果不意味着从零训练的网络不会遗忘，更不意味着任意无边界任务流只需微调。

#### 阅读与实验

固定预训练模型，分别改变可训练参数量和任务顺序。报告预训练资源、每任务在线数据、回放或重置条件，再与传统 CRL 方法比较。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2603.11653)：VLA 持续学习设置、机制与比较。
- [RLC 2026 / RLJ 论文页](https://rlj.cs.umass.edu/2026/papers/Paper84.html)：会议原文入口；PDF 脚注链接作者代码。
- [作者实现](https://github.com/UT-Austin-RobIn/continual-vla-rl)：持续 VLA 的训练与评价代码。

#### 作者代码

[UT Austin RobIn 实验室的原论文仓库。](https://github.com/UT-Austin-RobIn/continual-vla-rl)

预训练 VLA 的顺序 RL 训练和论文评价。

### Recurrent Reinforcement Learning with Memoroids

Steven Morad, Chris Lu, Ryan Kortvelesy, Stephan Liwicki, Jakob Foerster, Amanda Prorok

NeurIPS 2024 · 2024 · 支持方法与理论

#### 研究问题

当记忆网络能够保存信息时，训练序列的切分是否仍会阻止学习器给早期信息分配信用？

#### 关键机制

Memoroids 将一类线性递归模型写成结合运算，利用并行 scan 处理长序列；Tape-Based Batching 将多个完整回合接入同一条 tape，用显式边界处理状态重置，减少分段、补零和截断反传带来的问题。

#### 证据

论文在 POPGym 等部分可观测任务和循环价值学习中比较分段与 tape 训练，并研究观测敏感度、样本效率及运行时间。

#### 条件与限制

并行 scan 和长序列反传使用保存的序列与批处理资源，不属于严格逐步、每条经验只使用一次的 RTRL。结合结构也不使任意非线性 RNN 都能采用同样的 scan。

#### 阅读与实验

固定同一种记忆模型，对照截断长度、完整回合和流式在线导数；分别检查活动能记多久、梯度能传多久、持久内存与训练峰值内存。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://papers.nips.cc/paper_files/paper/2024/file/19f7f755908372efb25826d61959cdf9-Paper-Conference.pdf)：结合运算、inline reset、Tape-Based Batching 与实验。
- [作者公开版本](https://arxiv.org/html/2402.09900v3)：附录给出不同递归模型与回报的 memoroid 写法。

#### 作者代码

[论文附录原链接 memory-monoids 对应作者 Prorok Lab 的现有 memoroids 仓库；README 标明论文。](https://github.com/proroklab/memoroids)

memory 模型、buffer、losses 与 segment_dqn/tape_dqn 对照。

### Parseval Regularization for Continual Reinforcement Learning

Wesley Chung, Lynn Cherif, David Meger, Doina Precup

NeurIPS 2024 · 2024 · 直接研究持续学习

#### 研究问题

仅在初始化时保持良好的权重几何，是否足以让很晚出现的新任务仍容易学习？

#### 关键机制

在选定隐藏层加入 $\lambda\|WW^\top-sI\|_F^2$，持续约束行向量的范数与角度；输出层及额外尺度设计保留表达能力。它维护学习的几何条件，并不直接保存旧任务标签或预测。

#### 证据

作者在 Gridworld、CARL、MetaWorld 任务序列中检验，并拆分范数与角度约束。稳定秩、Jacobian 与熵属于诊断量，不单独构成可塑性或保留的因果证明。

#### 条件与限制

约束会限制函数类；输出行数大于输入维度时，全部行正交不可实现。非线性门控仍能切断梯度。有限任务序列的结果不保证无限生命内有效，也不是无任务信息的万能机制。

#### 阅读与实验

同预算比较仅初始化正交、持续范数约束、持续角度约束和完整正则；同时记录新目标拟合、真实回报、旧功能与额外计算。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/e6df4efa20adf8ef9acb80e94072a429-Paper-Conference.pdf)：目标函数、容量限制、角度/范数消融及持续任务协议。
- [作者版本记录](https://arxiv.org/abs/2412.07224)：正式会议年份为 2024。

#### 作者代码

[仓库明确标为 NeurIPS 2024 官方实现。](https://github.com/wechu/parseval_reg)

PPO、任务序列、正则化与网络结构消融。

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

### How Should We Meta-Learn Reinforcement Learning Algorithms?

Alexander David Goldie, Zilin Wang, Jaron Cohen, Jakob Foerster, Shimon Whiteson

RLC 2025 / RLJ · 2025 · 评价与实验协议

#### 研究问题

算法表示、发现算法的方法和测试智能体的学习成本，应该如何独立比较？

#### 关键机制

对 RL 流程的不同组件进行算法发现，比较黑盒学习、神经/符号蒸馏与 LLM 代码提案。学习器的表示形式和搜索过程分开定义，才能识别泛化、可解释性和成本之间的取舍。

#### 证据

论文直接比较元训练、元测试、样本成本、训练时间与可解释性，作者代码按发现方法和评价入口组织。

#### 条件与限制

跨环境发现规则主要发生在设计者侧，不等于运行智能体已能终生修改规则。论文的训练任务和预算范围不支持所有算法发现方法的普适排序。

#### 阅读与实验

固定被学习组件、输入权限、元训练数据和发现预算；封存规则后再检验未见环境、长生命与未通知漂移。

#### 原文与相关入口

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_218.pdf)：比较对象、元训练/测试与多维成本。
- [RLJ 论文记录](https://rlj.cs.umass.edu/2025/papers/Paper218.html)：作者与正式会议收录。

#### 作者代码

[论文提供、仓库标为官方的作者实现。](https://github.com/AlexGoldie/learn-rl-algorithms)

learning_algorithms 中各发现方法与独立 evaluation 流程。

### Does Zero-Shot Reinforcement Learning Exist?

Ahmed Touati, Jérémy Rapin, Yann Ollivier

ICLR 2023 · 2023 · 支持方法与理论

#### 研究问题

没有事先指定奖励时，怎样学一套预测表示，日后接收新奖励就能选行为？

#### 关键机制

Forward–Backward 表示联合学习行为条件的未来占用与奖励读出，而非先固定任意编码器再学习 successor features。新奖励被映射到任务向量，策略根据这个向量直接行动；该论文系统比较 FB 与多种 SF 基础特征。

#### 证据

原文在固定离线 replay buffers 上比较零样本任务迁移，借此把表示学习与探索数据的质量分开。不同特征与数据覆盖产生显著差异，不能仅靠“所有奖励”的理论目标预测实际效果。

#### 条件与限制

假定共享动力学与可用经验覆盖。无下游梯度更新不等于无预训练成本；有限秩、近似训练和奖励估计都有误差。新动力学、历史混叠和严格一次使用经验均须另测。

#### 阅读与实验

同一 buffer 对比随机特征、谱特征与联合 FB，再独立换 buffer。奖励读出误差、占用误差与新任务回报分别报告，避免把数据覆盖优势记成表示优势。

#### 原文与相关入口

- [作者原文](https://arxiv.org/abs/2209.14935)：2022 首稿，ICLR 2023；比较奖励表示、SF 与 FB。
- [作者研究平台](https://github.com/facebookresearch/controllable_agent)：README 直接关联两篇 FB 论文；该仓库已经归档。

#### 作者代码

[论文作者团队仓库；归档工程，依赖和旧环境需单独核验。](https://github.com/facebookresearch/controllable_agent)

FB 与 SF 的训练、固定数据实验及奖励查询示例。


<a id="chapter-code"></a>

## 下载与运行

标准库非平稳 bandit 的开发选择、独立测试、配对区间及失败诊断；不包含大型神经网络基准。

[下载 experiment_design_lab.py](https://yingwen.io/zh/continual-rl/download/experiment_design_lab.py)

```sh
python experiment_design_lab.py demo --tiny
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Patterson et al. · Empirical Design in Reinforcement Learning](https://jmlr.org/papers/v25/23-0183.html)：JMLR 2024。贯穿性能分布、区间、超参数选择、基线与实验者偏差；适合按研究流程精读。

- [Jordan et al. · Evaluating the Performance of Reinforcement Learning Algorithms](https://proceedings.mlr.press/v119/jordan20a.html)：ICML 2020。完整算法定义、可用性、选择成本与跨任务评价；其研究对象不同于充分调参后的峰值。

- [Jordan et al. 作者评价代码](https://github.com/emmaajordan/EvaluationOfRLAlgs)：原论文 Software 链接指向的作者仓库，包含评价方法及实验实现。

- [Agarwal et al. · Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)：NeurIPS 2021。任务内 bootstrap、IQM 与性能剖面；少量重复的覆盖误差与聚合条件需要一起阅读。

- [rliable · 作者统计工具与 notebook](https://github.com/google-research/rliable)：运行×任务矩阵、指标、区间及原始 benchmark 示例。仓库已归档，复现时固定版本与依赖。

- [Mesbahi et al. · Lifetime tuning is incompatible with continual RL](https://proceedings.mlr.press/v267/mesbahi25a.html)：ICML 2025 position paper。讨论限制寿命调参的动机与实验；前缀比例应按具体部署问题选择。

- [Adam White · Deep RL Course 讲义](https://deeprlcourse.github.io/assets/guests/adam_white.pdf)：从实验目的、运行变异、超参数与模型辅助选择建立问题地图。配合 Patterson 等长文使用。

- [Chan et al. · Measuring the Reliability of RL Algorithms](https://arxiv.org/abs/1912.05663)：ICLR 2020。区分训练过程与固定策略的波动、风险；用于设计均值之外的可靠性指标。

- [Tanner & White · RL-Glue](https://www.jmlr.org/papers/v10/tanner09a.html)：环境、学习器与实验控制的接口分离；思想可用于现代实验框架，不要求采用旧运行时。
