# 强化学习问题与大世界中的持续学习

先定义交互接口、未知量、信息权限、候选解和评价准则，再推导求解机制。强化学习的动作影响奖励、后续情境与能学到的经验；持续强化学习进一步研究有资源限制的智能体如何在长期交互中适应。MDP、目标函数、深度方法与持续学习分别改变不同条件，需要逐项说明关系。

## 1. 形式化问题：给定什么，求什么

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

Sutton 与 Barto 从目标导向的交互学习出发定义强化学习。智能体选择动作，环境返回信息和奖励。智能体需要从后果中学习，而不是在每一步获得正确动作的标签。强化学习既指这一类问题，也指研究它的方法；某个具体算法并不等于问题本身。

智能体与环境的边界由研究对象确定，不必与身体或设备的物理边界一致。环境也不等于未知部分：即使已知游戏规则，如何选择动作仍可能很难。本文先用观测描述一般接口，再把充分的 Markov 状态作为一个额外条件。

一个清楚的问题描述还必须规定目标、信息权限和资源条件。折扣回报、有限时域累计奖励与平均奖励是不同评价准则。不能只给一条奖励曲线，却不说明它评价的是训练后的固定策略，还是包含训练成本的完整学习过程。

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

以下总览沿问题规格推导方法的用途：第 2 节确定评价对象与内部实现，第 3 节说明环境假设提供的递推结构，第 4 节限制可执行智能体，第 5 节检查协议和持续性的含义。后面的表示、预测、控制、规划和适应机制，都必须回到这些给定条件上判断是否有帮助。

## 2. 评价对象与智能体的实现状态

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

| 符号 | 含义 | 需要保留的区别 |
| --- | --- | --- |
| $X_t$ | 环境的内部状态 | 未必可观测；在一般历史过程下，也未必存在一个已知、有限维的表示。 |
| $O_t$ | 接口提供的当前观测 | 不是充分状态的同义词。 |
| $S_t$ | 用于决策和预测的智能体状态 | 是历史的摘要；Markov 性需要论证，不能由命名获得。 |
| $\theta_t$ | 可学习参数 | 记忆状态的递推不等于参数学习。 |
| $Z_t$ | 完整实现状态 | 包括参数、记忆、迹、优化器统计、模型与允许保存的经验；不是环境真状态。 |

$$
\begin{aligned}A_t&\sim\pi_{\theta_t}(\cdot\mid S_t),\\ Z_{t+1}&=F(Z_t,A_t,R_{t+1},O_{t+1},\xi_{t+1}),\\ (S_{t+1},\theta_{t+1})&=\operatorname{readout}(Z_{t+1}).\end{aligned}
$$

$Z_t$ 已包含对当前观测的处理结果。$F$ 是实现的更新规则，$\xi_{t+1}$ 表示内部随机性。该形式描述有限智能体如何执行，不保证 $S_t$ 或 $Z_t$ 使环境响应满足 Markov 性。

Sutton 的 Common Model 将感知／状态构造、反应式策略、价值函数和转移模型作为相互联系的部件。状态构造从经验历史中提取当前有用的信息。价值与模型分别预测回报和后果。规划使用这些预测比较行动。这里采用这一功能划分，但不把它视为唯一架构，也不要求每个算法显式实现全部部件。

## 3. 问题之间的关系与可解结构

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

若某个状态包含预测下一步所需的全部信息，且响应规律不随时间改变，就可以使用平稳 Markov 决策过程。有限状态与动作集合进一步给出有限 MDP。此时能用逐状态的方程研究策略评价与改善。

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

$$
v_\pi(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)\bigl[r+\gamma v_\pi(s')\bigr]
$$

这是固定平稳策略下的折扣 Bellman 方程。此处假设有限 MDP、奖励有界且 $0\leq\gamma<1$。为简洁起见，对离散奖励求和；连续奖励应改用积分。若策略随学习状态改变，应把未来学习规律纳入评价，不能直接套用当前冻结策略的方程。

这里 $v_\pi(s)$ 是从状态 $s$ 起按固定策略 $\pi$ 行动的期望折扣回报，$a,s',r$ 是动作、下一状态和奖励的取值。它把一般历史递推压缩到状态空间，减少求解对象。有限精确折扣控制的最优 Bellman 算子是最大范数下的 $\gamma$ 收缩；表格动态规划利用此结构。近似价值、循环状态或网络策略则须重新检查充分性、表示能力和更新条件，不能直接继承精确问题的全部保证。

Bandit 省去动作对后续状态的控制，集中研究即时收益与探索。表格方法逐项保存估计。函数逼近在状态间共享参数。深度强化学习使用神经网络承担表示和近似。这些选择分属问题结构与方法设计两个层次，不组成相互替代的三类环境。

## 4. 从一般强化学习到大世界中的持续学习

Javed 与 Sutton 的大世界假设关注这样一类问题：环境的复杂性显著超过智能体可用于感知、表示和计算的资源。它不是关于所有任务的定理。研究重点由“保存整个问题的解”转向“在有限资源下维护当前有用的知识”。

即使环境规律固定，有限表示也可能无法同时精确表达所有相关情境。若相邻时刻的经验具有结构性关联，智能体可以跟踪当前相关的部分。保留什么、替换什么、何时规划，因而成为学习问题。这里的依据是容量限制与经验结构，而不是把平稳环境误称为非平稳环境。

$$
\max_{\mathcal L\in\mathfrak L(B)}\;\mathbb E_{\mathcal L,P}\!\left[\sum_{t=0}^{T-1}R_{t+1}\right]
$$

这是说明问题的一种有限生命期形式，不是 CRL 的唯一公认定义。$\mathcal L$ 是包含初始化、行动与更新规则的完整学习器；$\mathfrak L(B)$ 指满足资源预算 $B$ 的学习器集合。比较时固定环境协议与时域 $T$。平均奖励或其他准则需另行定义。

持续学习并不要求每一步都改变所有参数。它要求把获得新能力、维护已有能力和长期行为表现一起研究。资源预算应覆盖参数、经验存储、每步更新、模型规划及开发阶段的选择成本；不同预算定义会产生不同的比较问题。

$$
\mathfrak L(B_1)\subseteq\mathfrak L(B_2)\quad\Longrightarrow\quad
  \sup_{\mathcal L\in\mathfrak L(B_1)}J_P(\mathcal L)\le
  \sup_{\mathcal L\in\mathfrak L(B_2)}J_P(\mathcal L)
$$

$J_P$ 为在同一环境 $P$ 与同一准则下的性能，$B_1,B_2$ 是预算。预算放宽所带来的候选集合包含，给出最优性能的单调性。增加预算的某个具体算法仍可能表现更差；预算限制也不独自证明最优者必须永久学习。

这解释了持续学习各机制的用途：状态构造压缩可用历史，预测支持尚未掌握的后果判断，探索取得新信息，信用分配选择如何更新，知识管理在容量内决定保留什么，规划把计算花在行动后果上。每项机制都要用相同外部准则检验收益，并记录它新增的交互、存储和计算成本。

## 5. 非回合任务、持续学习与流式学习

| 术语 | 本文含义 | 不应据此推断 |
| --- | --- | --- |
| 非回合式任务（continuing task） | 交互没有问题定义中的自然终止。 | 不等于持续更新参数；固定策略也能在非回合任务中运行。 |
| 持续强化学习（continual RL） | 研究贯穿长期交互的学习与适应；具体形式化见不同研究观点。 | 不只等于任务序列，也不要求环境核一定变化。 |
| 非平稳环境（non-stationary environment） | 相对于所声明的状态，奖励或响应核随时间改变。 | 策略改变造成的数据分布变化，不自动意味着环境核变化；潜在固定核也可能产生观测上的漂移。 |
| 单次生命期（single-life） | 算法不能通过外部重置任意恢复到先前环境状态。 | 是否允许回放、预训练或救援仍需单独规定。 |
| 严格流式学习（strict streaming） | 按本教材协议，逐步处理经验，不存储并重放历史样本。 | 资格迹和优化器状态可以是允许的摘要；其内存与计算仍须计入。 |

这些条件可以交叉组合。例如，固定 MDP 中的非回合控制可以使用持续更新的深度网络，也可以使用冻结策略。反过来，带自然回合终止的任务序列仍可研究知识保留与可塑性。术语不能代替实验协议。

把时间并入状态 $\widetilde S_t=(S_t,t)$，可将时间不齐次的 Markov 核写成形式上齐次的核：从 $(s,t)$ 转移到 $(s',t+1)$。但新状态空间可能无限，未来核仍可能未知；若变化取决于隐藏因素，单加时钟也未必让观测充分。形式上的状态扩张不消除学习、泛化或资源难题。

Abel 等的定义固定环境、性能、候选智能体和能够生成它们的基底，再要求所有最优智能体永不到达基底。这里“到达”要求从某个可实现历史开始，所有可实现延续上的行为永远等于一个基底元素；不是仅观察参数在更新，也不是以非平稳标签代替证明。该定义与资源受限平均奖励、变化任务评测关注的对象有交集，但它们没有无条件等价关系。

## 共享经验的学习问题

### 表示与智能体状态

过去哪些信息必须保留，才能预测和行动？

观测编码、记忆递推和参数学习不同。学得状态未必满足 Markov 性。

### 预测与知识

在某种行为下，哪些信号将怎样累积？

预测准确不等于控制有用。GVF 的问题规格与学习它的算法分开定义。

### 控制与策略改善

现在做什么，才能改善整个学习过程的结果？

动作改变世界，也改变以后能学到的数据。冻结策略价值不包含完整适应过程。

### 模型与规划

不实际执行时，怎样估计行动后果并分配计算？

一步预测、想象训练和决策时搜索有不同接口。精确预测与有用规划不是同一指标。

### 目标与时间抽象

哪些子任务和多步行为值得构造、维护与复用？

总体目标、子任务奖励和技能发现目标必须分开。发现技能本身也有交互成本。

### 学习机制与资源分配

有限记忆和计算下，怎样长期学习而不失去能力？

信用分配、数据协议、元学习与可塑性处理不同对象；它们可以组合。

## 简化与约束

### Bandit

在标准独立拉臂特例中，动作只选择当次奖励分布；未知收益仍要求学习。

Contextual、非平稳与相关奖励是额外假设；不能用简单 bandit 检验受控状态的长程后果。

### 有限 MDP

有限充分状态、有限动作与 Markov 核带来逐状态递推；平稳性再消除时间下标。

MDP 不要求回合重置。已知模型可规划，未知模型需经验；折扣收缩仍需奖励有界和折扣小于一。

### POMDP / 一般历史过程

前者假设潜在 Markov 状态但不完全揭示；后者允许更一般的历史依赖。

POMDP 可嵌入历史过程。已知模型的信念可以充分，但不代表有限智能体能精确保存和求解。

### 函数逼近 / 深度 RL

用特征或神经网络共享参数，处理大输入与泛化。

这是表示与计算方法，不是与 MDP 并列的一种环境定义。

### 任务序列 / 外部非平稳

环境、目标或数据分布按协议改变。

是否给边界、任务标签和重置，会直接改变难度。不能等同于所有 CRL。

### Single-life / 严格 streaming

分别限制环境重置、经验存储与每步更新。

一次生命期不等于不许回放；逐步交互也不等于严格流式。

### 有界智能体 / 大世界

限制表示、记忆、计算与经验，研究持续适应。

固定世界也可产生持续学习需求。“世界很大”是研究假设，不是所有任务的定理。

## 不同研究观点

### Sutton、Javed 与 OaK：有限智能体怎样在大世界中构造知识

以环境复杂性超过智能体容量为出发点，研究跟踪、选择和替换知识。时间抽象与规划是其架构研究的重要组成。

Big World 是问题选择与研究假设；OaK 架构目标不是一个已经完成全部验证的通用智能体。

即使环境规则固定，有限容量也可能使持续适应有价值。不能因此推断所有基准都会出现同一瓶颈。

- [The Big World Hypothesis and its Ramifications for AI](https://oaklab.ai/posts/the-big-world-hypothesis)
- [OaK Lab 研究使命](https://www.oaklab.ai/mission)

### Abel、Barreto、Van Roy、Precup、van Hasselt 与 Singh：什么叫“持续”

用智能体的生成与到达关系讨论何时最优行为需要持续学习。问题中心是学习过程，而不只是环境是否发生变化。

A Definition of Continual Reinforcement Learning（NeurIPS 2023）给出一套形式化定义，不是唯一被全领域采用的定义。

需要明确基底、智能体集合及其限制。若改变允许的智能体类别，持续性的判断也可能改变。

- [A Definition of Continual Reinforcement Learning](https://arxiv.org/abs/2307.11046)

### Elelimy、Szepesvári、White 与 Bowling：如何评价持续适应的智能体

从智能体实际经历的情况出发，研究基于偏离行为的比较，而不是只对照一条未必可实现的全局最优轨迹。

Rethinking the Foundations for Continual Reinforcement Learning（RLC 2025）提出评价框架及实证估计方法。估计依赖可用数据与模型条件。

不能把“冻结策略的值不足”改写成“CRL 没有价值函数”。改变的是评价对象、条件信息与可行比较器。

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_243.pdf)

### Abel、Ho 与 Harutyunyan：重新检查环境、解与奖励的默认地位

Three Dogmas of Reinforcement Learning 把智能体本身、持续适应和奖励假设的细节重新放到问题中心。

RLC 2024 的概念性论证。对默认做法的批评不是统一算法，更不能当作所有旧结论都失效。

研究可以从寻找固定问题的解转向研究适应过程，但每个实验仍需明确其目标与假设。

- [Three Dogmas of Reinforcement Learning](https://david-abel.github.io/tdorl.pdf)

### Kumar、Van Roy 及合作者：把计算限制写进问题

Continual Learning as Computationally Constrained Reinforcement Learning 研究智能体计算和记忆约束下的持续学习。

形式化研究框架。具体结论取决于如何计量计算、允许哪些算法和环境。

不能只报告样本数。两个方法使用不同存储、规划和调参预算时，可能在解决不同的受限问题。

- [Computationally Constrained RL](https://arxiv.org/abs/2307.04345)

### Khetarpal、Riemer、Rish 与 Precup：变化条件下的适应

综述从非平稳性出发，组织持续强化学习中的变化、方法与评测。相关理论还用变化预算限制奖励和转移的漂移。

这是一条可操作的研究路线，不覆盖所有大世界或单次生命期问题。

外部变化、策略导致的数据变化、表征变化需要分开。适应速度与旧任务保留也可能冲突。

- [Towards Continual Reinforcement Learning: A Review and Perspectives](https://arxiv.org/abs/2012.13490)
- [Non-Stationary RL: The Blessing of (More) Optimism](https://proceedings.mlr.press/v119/cheung20a.html)

### Bowling、Abel 及奖励学习研究：目标从哪里来

把奖励的表达能力、设计者偏好和奖励推断分开。既研究何时奖励表示成立，也研究观察、反馈和有限资源下怎样得到可用奖励。

表示定理有公理条件。偏好学习与逆强化学习有行为模型与可辨识性限制。

优化给定奖励不等于实现设计者意图。加入人的反馈也不是消除了外部设计者。

- [Settling the Reward Hypothesis](https://proceedings.mlr.press/v202/bowling23a.html)
- [On the Expressivity of Markov Reward](https://arxiv.org/abs/2111.00876)
- [Cooperative Inverse Reinforcement Learning](https://arxiv.org/abs/1606.03137)

## 原始参考

- [Sutton & Barto · Reinforcement Learning: An Introduction（第 1、3、9 章）](http://incompleteideas.net/book/the-book-2nd.html)：交互学习、智能体—环境接口、回报与函数逼近。
- [Sutton · The Quest for a Common Model of the Intelligent Decision Maker](https://arxiv.org/html/2202.13252v1)：观测、主观状态、策略、价值与模型之间的功能关系。
- [Javed & Sutton · The Big World Hypothesis](https://oaklab.ai/posts/the-big-world-hypothesis)：有限智能体、跟踪和计算约束；文中明确说明它不是所有任务的共同性质。
- [Sutton, Koop & Silver · On the Role of Tracking in Stationary Environments](https://doi.org/10.1145/1273496.1273606)：环境平稳与跟踪学习并不矛盾；需结合表示限制和经验结构。
- [Kumar et al. · Continual Learning as Computationally Constrained Reinforcement Learning](https://arxiv.org/abs/2307.04345)：将智能体的计算限制纳入持续学习问题。
- [Abel et al. · A Definition of Continual Reinforcement Learning](https://arxiv.org/html/2307.11046v2)：第 2 节从历史行为、环境和性能定义一般问题；第 3–4 节给出相对于 agent basis 的生成、到达与 CRL。
- [Cassandra, Kaelbling & Littman · Acting Optimally in Partially Observable Stochastic Domains](https://cdn.aaai.org/AAAI/1994/AAAI94-157.pdf)：已知 POMDP 的状态信念与信息获取；表明隐藏状态问题可以有价值函数和控制目标。

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

# 从研究问题进入持续强化学习

这是一种教学组织，不是唯一分类。先明确目标、信息与干预权限，再选择机制。

## 一生的“学得好”到底指什么？

现象：终局测试很高，但适应时损失巨大，或者依赖免费重置。

先区分：目标、交互协议和变化来源是三条独立轴；不是看到 continual 就必须使用某一个算法。

- 优化折扣收益、平均奖励率，还是一段生命期的累计效用？
- 变化发生在奖励、动力学、观测、动作集合，还是学习器自己诱导的数据分布？
- 任务 ID、边界、reset、replay、预训练经验各允许什么？

### 算法线与条件

- **折扣 / 有限时域 / 平均奖励**：改变如何评价未来与时间代价。条件与限制：平均奖励需要相应的遍历/通信等假设，并不自动解决变化。
- **非平稳 MDP / 潜在情境 POMDP**：前者描述外部随时间改变；后者把未观测情境纳入状态。条件与限制：同一表象可以有不同模型，不要混用各自理论保证。
- **任务序列 / task-agnostic / single-life**：改变可用信息和干预权限。条件与限制：知道边界不等于知道任务身份；没有手动 reset 不等于永不终止。

诊断：同一策略分别计算整体收益、变化后损失和每真实时间单位收益；先看结论是否因评价方式翻转。

基准：CSuite 适合持续目标；Continual World 适合任务序列；Single-Life RL 适合一次试验中的恢复。三者不能直接混排。

- [Khetarpal et al. · Towards Continual RL](https://arxiv.org/abs/2012.13490)：按非平稳性的发生位置和驱动力组织问题；提醒我们先写设定，再比较方法。
- [Abel et al. · A Definition of Continual RL](https://arxiv.org/abs/2307.11046)：将持续性落在智能体是否需要一直学习，而不只落在外部任务切换。这里采用其问题视角，不把下面的教学分类说成该文定理。
- [Pan et al. · A Survey of Continual RL](https://arxiv.org/abs/2506.21872)：补充任务序列、保留、迁移、容量和评测的广泛方法线；其任务化视角与 Abel 的定义不是完全相同的设定。
- [Wan et al. · Learning and Planning in Average-Reward MDPs](https://proceedings.mlr.press/v139/wan21a.html)：Differential TD/Q 与平均奖励规划，区分奖励率和相对价值。

---

## 规律变了，怎样及时发现并改正？

现象：旧估计非常稳定，却迟迟跟不上新奖励或新动力学。

先区分：没采到变化、统计估计太慢、情境识别错误，是三种不同失败；不必先假定神经网络可塑性坏了。

- 变化是突变、渐变还是周期重现？
- 检测误报和检测延迟如何影响控制？
- 是忘掉旧统计量，还是保存旧模型并识别何时复用？

### 算法线与条件

- **常数步长 / 指数遗忘 / 滑动窗口**：使近期数据有更大权重。条件与限制：更快适应会增加估计方差；窗口按真实时间还是访问次数要写清。
- **变化点检测 / 多模型 / 上下文推断**：判断何时切换学习状态或选择旧专家。条件与限制：TD error 大也可能只是探索到了新状态，不能直接当变化证明。
- **SWUCRL2-CW / BORL；permanent–transient values**：分别从非平稳遗憾、快慢价值分解处理持续修正。条件与限制：理论设定、任务边界权限、保存状态和深度实现并不相同。

诊断：先用固定数据流比较估计器，再让动作影响数据；同时记录预测误差、检测延迟/误报和变化后损失。

基准：先分段平稳或渐变 bandit / 小 MDP；再进入 task-agnostic Meta-World 或 AgarCL，不从最高维平台猜失败来源。

- [Cheung et al. · Non-stationary MDPs](https://proceedings.mlr.press/v119/cheung20a.html)：SWUCRL2-CW 与 BORL 把滑窗、乐观探索和变化预算联系起来；不是只换一个常数学习率。
- [Anand & Precup · Prediction and Control in CRL](https://arxiv.org/abs/2312.11669)：用 permanent / transient 两个价值分量连接长期积累与短期修正；区分可见边界与持续设置。
- [Caccia et al. · Task-Agnostic CRL](https://proceedings.mlr.press/v232/caccia23a.html)：3RL 把循环状态与回放结合，用历史推断情境；任务未知不等于历史中没有可推断的信息。

---

## 当前画面不够，应该记住什么？

现象：相同观测需要不同动作；更大的前馈网络仍做错。

先区分：工作状态更新、持久知识学习和训练梯度记忆不是同一个东西。

- 哪些历史必须被区分，哪些可丢弃？
- 记忆的训练目标是奖励、观测预测还是一组未来测试？
- 长依赖的梯度怎样在每步预算内传播？

### 算法线与条件

- **Frame stack / RNN / Transformer**：从固定窗口到递归状态或上下文。条件与限制：更长上下文不自动等于更强长期权重学习。
- **Belief / PSR / GVFN**：以后验或预测赋予状态可解释语义。条件与限制：已知生成模型、核心测试和任意学习到的预测集合并不等价。
- **BPTT / TBPTT / RTRL / RTU**：为同一递归表示设计不同信用计算。条件与限制：detach 切断梯度，不一定抹掉前向记忆；成本必须随容量报告。

诊断：用同画面异动作的平衡线索任务，加上可观测状态参照、手工记忆参照，再比较学得记忆。

基准：从 delayed cue 到 POPGym/Forager；必须控制记忆容量、历史长度、每步计算与特权信息。

- [Caccia et al. · Task-Agnostic CRL](https://proceedings.mlr.press/v232/caccia23a.html)：3RL 把循环状态与回放结合，用历史推断情境；任务未知不等于历史中没有可推断的信息。
- [Schlegel et al. · General Value Function Networks](https://arxiv.org/abs/1807.06763)：让递归表示受到预测问题的约束，连接状态构造与预测学习。
- [Elelimy et al. · Real-Time Recurrent Learning using Trace Units](https://arxiv.org/abs/2409.01449)：结构化递归与前向敏感度降低在线信用分配成本；研究重点不只是网络能否记住。

---

## 新知识怎样不破坏仍有用的旧知识？

现象：学完 B 后，在相同协议下重新测 A，A 的表现下降。

先区分：保留准确的旧知识不等于保留已经失效的答案；保留与跟踪可能冲突。

- 该保留原始经验、输出行为、参数约束还是独立模块？
- buffer 应代表过去平均分布还是未来会重访的情境？
- 没有任务 ID 时，如何决定使用哪个知识模块？

### 算法线与条件

- **Replay / CLEAR / generative replay**：重新暴露旧训练信号，必要时增加行为蒸馏。条件与限制：旧 transition 可能已不符合当前动力学；生成模型本身也会忘。
- **EWC / distillation / policy consolidation**：约束重要参数或旧输入上的输出。条件与限制：保护强度太大可能阻止适应；重要性估计不等于未来用途。
- **Progress & Compress / 模块化 / PT values**：隔离快学与长存，或按情境重用。条件与限制：存储、路由、压缩和边界权限都要计入成本。

诊断：做 A→B→A 的保留矩阵，同时测 B 的学习速度；增加固定内存和未知任务 ID 的对照。

基准：Continual World / COOM / continual_rl；同时报告旧任务、当前任务和前向迁移，不能只给最终平均分。

- [Rolnick et al. · Experience Replay for Continual Learning](https://arxiv.org/abs/1811.11682)：CLEAR 结合旧经验上的离策略学习、当前经验学习与行为克隆；回放不只是存一个 buffer。
- [Schwarz et al. · Progress & Compress](https://arxiv.org/abs/1805.06370)：把快速学习与压缩到长期知识分开；检查蒸馏、保留约束和协议权限。
- [Anand & Precup · Prediction and Control in CRL](https://arxiv.org/abs/2312.11669)：用 permanent / transient 两个价值分量连接长期积累与短期修正；区分可见边界与持续设置。
- [Wołczyk et al. · Continual World](https://arxiv.org/abs/2105.10919)：机械臂任务序列同时考察保留与前向迁移；不是无重置单生命期环境。

---

## 训练久了，为什么新东西越来越难学？

现象：老网络获得相同新数据和预算，却比合适的 fresh 参照学得慢。

先区分：新任务误差高可能只是起点不同；神经元稀疏、梯度小等是诊断，不是可塑性结论本身。

- 是数据覆盖差、目标不稳定，还是表示和优化退化？
- 替换哪个单元，何时替换，损失多少旧功能？
- 收益是一次重启，还是长生命史持续保持的新学习能力？

### 算法线与条件

- **CReLU / normalization / regularization**：改变激活、尺度或参数轨迹。条件与限制：需和同预算学习率及初始化控制比较。
- **ReDo / Continual Backprop / shrink-and-perturb**：重新引入可训练特征或扰动。条件与限制：必须测输出扰动、optimizer 状态、替换频率和旧知识。
- **Primacy reset / replay-ratio 控制**：减弱早期经验固化或过拟合。条件与限制：保留 buffer 的网络重置不等于无先验从头学习。

诊断：固定新数据、模型容量和调参预算；比较 aged / fresh / random-reset，并分开初始误差与后续下降速度。

基准：先做受控新目标学习，再到 Atari 模式/任务变化与 AgarCL；可塑性方法在一种场景有效不保证所有 CRL 场景有效。

- [Dohare et al. · Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7)：长期新学习能力及 Continual Backprop；特征替换的效果不能仅由激活率判断。
- [Abbas et al. · Loss of Plasticity in Continual Deep RL](https://proceedings.mlr.press/v232/abbas23a.html)：在价值学习和不同非平稳条件下研究激活、梯度与后续学习，讨论 CReLU。
- [Nikishin et al. · The Primacy Bias in Deep RL](https://arxiv.org/abs/2205.07802)：区分早期经验固化与后续证据利用不足；参数重置可能保留 replay，不能视作全部重来。
- [Mohamed et al. · AgarCL](https://arxiv.org/abs/2505.18347)：非 episodic、高维、部分可观测且持续变化的研究平台；论文中若干可塑性方法提升有限，说明困难不止一条。

---

## 每步预算很小，怎样可靠地分配信用？

现象：实时更新发散、梯度延迟太长，或者一个环境步用了大量隐含回放。

先区分：数据协议、长期收益目标、表示结构和优化器是可组合的选择，不是一条算法世代替换线。

- 一个新误差应该影响哪些过去参与计算的量？
- off-policy 修正改变的是采样分布、目标还是更新方向？
- 尺度、步长、trace 与 actor/critic 怎样相互作用？

### 算法线与条件

- **TD(λ) / true-online / emphatic / gradient TD**：改变时间信用或离策略稳定机制。条件与限制：各式 eligibility 不通用；γ、ρ 的下标与支持集必须对齐。
- **Stream-X / 输入与更新归一化**：把深度每步更新做成可计算的训练流程。条件与限制：batch_size=1 不足以复现方法。
- **IDBD / TIDBD / meta-gradient**：让部分更新参数适应当前学习过程。条件与限制：监督误差、bootstrap 误差与策略目标的导数不能互换。

诊断：先验证 λ=0、终止、ρ=0 等退化情况，再记录每步延迟、内存、更新/样本比以及多 seed 回报。

基准：表格/线性解析任务 → 严格 streaming 控制；必须说明是否重复访问旧样本和隐藏的大批量计算。

- [Stream-X · 作者实现与论文入口](https://github.com/mohmdelsayed/streaming-drl)：研究逐条经验的深度 RL 更新；归一化、迹、更新幅度和 actor–critic 需要一起检查。
- [Elelimy et al. · Real-Time Recurrent Learning using Trace Units](https://arxiv.org/abs/2409.01449)：结构化递归与前向敏感度降低在线信用分配成本；研究重点不只是网络能否记住。
- [Xu et al. · Meta-Gradient Reinforcement Learning](https://arxiv.org/abs/1805.09801)：对学习更新求导以调整 return 等元参数；元目标不能随便跟训练题目一起改变。

---

## 经验该变成哪些预测，才能在未来派上用场？

现象：奖励之外的规律反复重学；辅助损失下降却没有改善决策。

先区分：定义一个预测问题、学准该预测、让下游使用它，是三个独立研究步骤。

- 预测信号、条件行为和终止尺度如何选择？
- 预测是否可由当前经验覆盖？
- 下游是状态构造、奖励迁移、技能发现还是规划？

### 算法线与条件

- **GVF / Horde / nexting**：多个 c、γ、π 定义多种未来知识。条件与限制：不是所有累计信号都具有事件概率语义。
- **Auxiliary tasks / predictive representation**：用额外学习信号塑造共享表示。条件与限制：辅助目标也可能争抢容量或产生干扰。
- **Successor features / GPI**：预测特征累计量，再按奖励权重重估价值。条件与限制：依赖可复用的动力学、策略与奖励特征。

诊断：分别测 held-out 预测误差、删除预测输入后的决策变化、分布变化后的再学习成本。

基准：先用多信号小 MDP 给出解析答案，再测试新奖励重估；预测数量多不能代替知识用途。

- [Sutton et al. · Horde](https://sites.ualberta.ca/~amw8/horde.pdf)：多个不同策略、信号和时间尺度的预测共享经验；学习问题和回答问题的算法分开。
- [Jaderberg et al. · Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397)：多个辅助控制/预测信号训练共享表示；辅助任务的用途需要用主任务表现检验。
- [Barreto et al. · Successor Features for Transfer](https://arxiv.org/abs/1606.05312)：在相同动力学、线性奖励族中重用后果预测，并用 GPI 组合策略。

---

## 怎样形成可复用的目标与长行为？

现象：每个任务都从原始动作重新探索，技能库却越积越多而用不上。

先区分：目标定义要学什么，option 定义怎样行动；技能有多样性不代表对主任务有用。

- 子目标由谁提出，怎样衡量用途？
- 技能该何时启动和停止？
- 新任务能否组合旧能力，而不只是重新训练一遍？

### 算法线与条件

- **UVFA / HER / goal generation**：目标条件化、重标记经验、选择训练目标。条件与限制：三者改变不同对象；HER 不是自动课程本身。
- **Option-Critic / eigenoptions / DIAYN / DADS**：任务驱动、图结构或信息量驱动技能学习。条件与限制：技能发现目标与主任务价值之间仍有距离。
- **Reward-respecting subtasks / SF-GPI / Option Keyboard**：按价值或奖励组合接口复用能力。条件与限制：GPI 不需要 commitment；option 的持续执行和终止是另一层。

诊断：对照 primitive-only、随机技能与同预算学得技能；同时计发现、建模、选择和执行的成本。

基准：先两步技能/导航连通性，再到连续技能；报告技能覆盖、真实主回报和学习总成本。

- [Schaul et al. · Universal Value Function Approximators](https://proceedings.mlr.press/v37/schaul15.html)：价值函数以目标为输入；给定目标后的泛化，不等于自动选择目标。
- [Sutton, Precup & Singh · Between MDPs and semi-MDPs](https://doi.org/10.1016/S0004-3702(99)00052-1)：用启动集合、内部策略和终止规则定义时间扩展行为，并导出相应备份。
- [Sutton et al. · Reward-Respecting Subtasks](https://arxiv.org/abs/2202.03466)：连接子任务、option、后果模型和规划；任务回报与抽象价值的关系是重点。
- [Barreto et al. · Successor Features for Transfer](https://arxiv.org/abs/1606.05312)：在相同动力学、线性奖励族中重用后果预测，并用 GPI 组合策略。

---

## 怎样用后果模型减少真实试错？

现象：模型内预测很好，现实行为却错误；规则变后越规划越自信。

先区分：动作模型、option model、价值函数和规划器分开；模型正确性与求解模型的精度也分开。

- 模型需预测完整分布，还是仅预测某个 backup 所需的统计量？
- 变化发生后模型怎样更新、忘记或重用？
- 把计算花在哪些起点、尺度和分支上？

### 算法线与条件

- **Dyna / prioritized sweeping**：用一步模型进行额外价值更新。条件与限制：模型学习与规划分配都影响结果。
- **Option / expectation / abstract models**：压缩时间或后果以降低规划成本。条件与限制：非线性下游一般不能只接一个均值。
- **MPC / MCTS / Dreamer / Continual-Dreamer**：在线搜索或想象中策略学习。条件与限制：世界模型、critic、actor 和 buffer 可能分别遗忘。

诊断：用已知模型确认 planner，再替换为学习模型；固定真实步数和总计算，画模型误差及真实回报随规划预算变化。

基准：blocking / shortcut maze → Minigrid / Minihack → 高维持续任务；不给某方法额外真值地图。

- [Sutton et al. · Reward-Respecting Subtasks](https://arxiv.org/abs/2202.03466)：连接子任务、option、后果模型和规划；任务回报与抽象价值的关系是重点。
- [Kessler et al. · World Models for Continual RL](https://proceedings.mlr.press/v232/kessler23a.html)：Continual-Dreamer 研究世界模型、经验选择与持续探索；模型与策略要分别测保留。
- [Cheung et al. · Non-stationary MDPs](https://proceedings.mlr.press/v119/cheung20a.html)：SWUCRL2-CW 与 BORL 把滑窗、乐观探索和变化预算联系起来；不是只换一个常数学习率。

---

## 接下来学什么、去哪探索，还能否安全回来？

现象：永远遇不到新证据，或者探索一次就失去后续交互与学习的机会。

先区分：新奇、信息增益、学习进展、主任务收益和可恢复性可能互相冲突。

- 如何区分未学会与不可约噪声？
- teacher 是否能挑任务、重置位置或改环境？
- 没有外部救援时怎样控制不可逆风险？

### 算法线与条件

- **Counts / RND / uncertainty**：按新奇或不确定性分配行为。条件与限制：误差大不保证可学，也不保证值得冒险。
- **Learning progress / ALP-GMM / 自动课程**：优先选择仍能提升能力的目标或环境。条件与限制：训练分布由 teacher 控制，需要明确权限。
- **Reset-free / recovery / Single-Life RL**：把回到可进行后续学习的状态纳入决策。条件与限制：给负奖励不等于具有安全保证；预训练经验也要计入。

诊断：除外在回报外记录覆盖、预测学习进展、人工干预、失能时间和恢复成功率；危险实验先在模拟器做。

基准：可控稀疏奖励任务、可参数化课程、Single-Life RL；不把有 teacher 的结果直接称为完全自主探索。

- [Burda et al. · Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)：固定随机目标与可训练预测器产生新奇信号；它不是可学习进展或安全性的直接度量。
- [Portelas et al. · Teacher Algorithms for Curriculum Learning](https://arxiv.org/abs/1910.07224)：以学习进展选择可参数化环境；需要说明 teacher 是否能重置和控制环境。
- [Chen et al. · You Only Live Once](https://arxiv.org/abs/2210.08863)：给定先前经验，在一次测试生命期中自主适应并恢复；不同于“训练完全没有先验”。

---

## 能不能让学习过程本身越用越合适？

现象：每次换环境都重新调步长，或者短期学得快却长期不稳定。

先区分：调整在线超参数、学习初始化、推断上下文、学习更新规则不是同一种“学会学习”。

- 元目标在评价什么未来，使用哪些额外数据？
- 对参数、隐藏状态还是完整 optimizer 状态求导？
- 元训练任务重置和部署持续性如何对齐？

### 算法线与条件

- **IDBD / TIDBD / meta-gradient RL**：在线改变步长或 return 参数。条件与限制：需要处理敏感度近似、尺度和训练/元评价目标。
- **MAML / context-based meta-RL / PEARL**：跨任务准备一个快速适应机制。条件与限制：元测试从头适应不证明长期累积。
- **Learned update rules / DiscoRL**：把学习规则参数化后优化其跨任务结果。条件与限制：发布规则的使用与重新进行元搜索是不同成本。

诊断：先有限差分验证导数，再匹配固定超参数调参预算；改变任务分布和学习时域，检查收益是否仍存在。

基准：标量敏感度 → 受控在线漂移 → 多任务元训练/测试；不同层次的结果分开报告。

- [Xu et al. · Meta-Gradient Reinforcement Learning](https://arxiv.org/abs/1805.09801)：对学习更新求导以调整 return 等元参数；元目标不能随便跟训练题目一起改变。
- [Finn et al. · MAML](https://proceedings.mlr.press/v70/finn17a.html)：跨任务训练易适应的初始化；重新从同一初始化适应和累积一段生命史是不同协议。
- [Abel et al. · A Definition of Continual RL](https://arxiv.org/abs/2307.11046)：将持续性落在智能体是否需要一直学习，而不只落在外部任务切换。这里采用其问题视角，不把下面的教学分类说成该文定理。

---

## 把模块接起来，怎样证明真的在持续积累？

现象：每个模块单测正常，整机却没有长期收益，结论只剩一条平均曲线。

先区分：内部活动、局部预测改善、持续参数更新和整体能力增长，需要不同证据。

- 状态变化会让哪些预测、技能和模型失效？
- 计算与记忆如何分给多个相互影响的学习器？
- 如何分开适应、保留、迁移、可塑性和安全？

### 算法线与条件

- **Life-long metrics / benchmark protocols**：测全过程，而不只测最终政策。条件与限制：eval reset、探针数据和冻结评测都会引入额外权限。
- **模块化消融 / 接口与版本管理**：固定一个对象，检查其真实下游用途。条件与限制：单模块更准不保证整个控制循环更好。
- **Alberta Plan / OaK / integrated agents**：把知识构建与长期更新接为研究路线。条件与限制：路线图不是已完成的端到端保证。

诊断：固定总预算，分别冻结状态、预测、模型或元更新；同时报告整段效用、恢复、旧知识、新学习、内存与延迟。

基准：由解析诊断到完整平台逐级增加困难。AgarCL、CSuite、Continual World 各覆盖部分问题，不存在一个分数概括整张地图。

- [Abel et al. · A Definition of Continual RL](https://arxiv.org/abs/2307.11046)：将持续性落在智能体是否需要一直学习，而不只落在外部任务切换。这里采用其问题视角，不把下面的教学分类说成该文定理。
- [Sutton et al. · The Alberta Plan](https://arxiv.org/abs/2208.11173)：将从经验持续构建知识组织为研究计划；研究路线与已验证的整机能力分开。
- [Oak Lab · The OaK Architecture](https://oaklab.ai/posts/the-oak-architecture)：用知识、抽象和元学习连接持续智能体模块；本指南把它作为架构路线，而非已完成的通用算法。
- [Mohamed et al. · AgarCL](https://arxiv.org/abs/2505.18347)：非 episodic、高维、部分可观测且持续变化的研究平台；论文中若干可塑性方法提升有限，说明困难不止一条。
- [DeepMind · CSuite 文档](https://rl-csuite.readthedocs.io/en/latest/)：持续交互的小型环境，可在复杂平台前隔离机制；continuing 不自动等于包含所有 continual 困难。


## 独立算法章节与实现

- [强化学习问题的形式化：交互、目标与持续学习](chapter-objectives.md) · [objectives_lab.py](objectives_lab.py)
- [奖励假设与奖励设计](chapter-reward-design.md) · [reward_design_lab.py](reward_design_lab.py)
- [平均奖励：奖励率、差分价值与持续控制](chapter-average.md) · [lifelong_algorithms_lab.py](lifelong_algorithms_lab.py)
- [Agent state：部分可观测性、递归记忆与在线信用分配](chapter-state.md) · [state_meta_lab.py](state_meta_lab.py)
- [价值预测与时间差分学习](chapter-value.md) · [foundations_detail_lab.py](foundations_detail_lab.py)
- [通用价值函数与预测知识](chapter-gvf.md) · [gvf_lab.py](gvf_lab.py)
- [持续控制：比较策略与学习智能体](chapter-control.md) · [continual_control_lab.py](continual_control_lab.py)
- [深度价值学习：DQN 与 Double DQN](chapter-deep-value.md) · [foundations_detail_lab.py](foundations_detail_lab.py)
- [策略梯度、Actor–Critic 与 PPO](chapter-policy.md) · [foundations_detail_lab.py](foundations_detail_lab.py)
- [最大熵控制与 Soft Actor–Critic](chapter-soft-control.md) · [foundations_detail_lab.py](foundations_detail_lab.py)
- [时间信用分配：从资格迹到深度梯度学习](chapter-credit.md) · [credit_assignment_lab.py](credit_assignment_lab.py)
- [流式强化学习：交互协议与更新稳定性](chapter-streaming.md) · [lifelong_algorithms_lab.py](lifelong_algorithms_lab.py)
- [学习规则的适应：在线元梯度与跨任务元学习](chapter-meta.md) · [state_meta_lab.py](state_meta_lab.py)
- [目标与子任务：条件控制、经验重用与技能设计](chapter-goals.md) · [knowledge_algorithms_lab.py](knowledge_algorithms_lab.py)
- [Options：多步决策、技能发现与可复用行为](chapter-options.md) · [knowledge_algorithms_lab.py](knowledge_algorithms_lab.py)
- [Dyna：模型学习与规划](chapter-dyna.md) · [foundations_detail_lab.py](foundations_detail_lab.py)
- [模型与后果预测：学什么，才能用于下一次决策？](chapter-models.md) · [knowledge_algorithms_lab.py](knowledge_algorithms_lab.py)
- [规划：把模型中的经验转成更好的决策](chapter-planning.md) · [knowledge_algorithms_lab.py](knowledge_algorithms_lab.py)
- [知识保留：经验重放、参数约束与模型记忆](chapter-retention.md) · [lifelong_algorithms_lab.py](lifelong_algorithms_lab.py)
- [可塑性：梯度通路、有效学习率与预测干扰](chapter-plasticity.md) · [lifelong_algorithms_lab.py](lifelong_algorithms_lab.py)
- [持续探索：新奇、不确定性、学习进展与恢复](chapter-exploration.md) · [lifelong_algorithms_lab.py](lifelong_algorithms_lab.py)
- [持续智能体架构：模块接口、更新调度与长期评价](chapter-architectures.md) · [lifelong_algorithms_lab.py](lifelong_algorithms_lab.py)
- [实验设计：从更新正确到持续学习证据](chapter-experiments.md) · [experiment_design_lab.py](experiment_design_lab.py)