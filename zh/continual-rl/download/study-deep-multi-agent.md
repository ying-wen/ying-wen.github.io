# 多智能体合作：结构化探索与信用分配

现代深度强化学习 · 并列研究分支

团队共享一个奖励时，怎样从联合经验中学习可执行的协作策略，并正确处理同伴更新？

## 本章内容

- 区分合作中的结构化探索与信用分配，以及竞争和开放合作中的评价与目标构建。
- 从反事实基线与可分散贪心条件推导 COMA、VDN、QMIX 和 QPLEX 的机制。
- 解释 MAVEN 与 Q-DPP 如何组织联合探索，以及额外的信息与表示条件。
- 区分 HAPPO 的顺序参数更新、A2PO 的评价修正与 MAT 的顺序动作生成。
- 核对理论中的精确优势与信赖域条件，不把 PPO 裁剪或网络结构当成回报保证。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [深度价值学习：DQN、Double DQN 与目标的时间顺序](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/)：掌握动作价值与自举。
- [策略更新的尺度：TRPO 与 PPO](https://yingwen.io/zh/continual-rl/foundations/deep/trust-region/)：理解联合更新前的采样策略与优势估计。


### MDP、POMDP 与策略评价

理解转移、回报和观察历史。团队共享目标，不代表每个人能看到相同信息；本章将从这些单智能体概念建立共同设定。

### DQN 与策略梯度

理解冻结 TD 目标、经验回放、score-function 梯度和与当前动作无关的 baseline。

### PPO 与异策略修正

理解新旧动作概率比、GAE、裁剪替代目标。样本上的 ratio 裁剪不是对全部状态的 KL 约束。

<a id="problem-definition"></a>

## 本章的问题定义

多个行动者在同一世界中获得共同奖励。每个人只有局部信息。训练者可能得到联合状态与动作，但执行权限必须另行声明。先研究固定成员、固定动力学和折扣目标，再讨论持续适应。

### 给定条件与符号

- 状态转移与共同奖励由环境产生，不假定已知其模型。
- 每个行动者的观察、动作集合、记忆和允许的通信。
- 训练可用的联合轨迹、集中信息、计算预算和重置协议。

### 需要求解的对象

满足执行信息约束、使团队收益较高的联合策略，以及能从有限联合经验中改善该策略的学习过程。

### 信息与数据权限

严格分散执行时，actor i 只能使用自身历史 Hᶦ。共享潜变量、联合观察和前序动作都是额外资源，必须写入执行协议；集中 critic 不能自动授予这些资源。

$$
J(\boldsymbol\pi)=\mathbb E_{\mu,P,\boldsymbol\pi}\!\left[\sum_{t=0}^{\infty}\gamma^t R_{t+1}\right],\qquad 0\le\gamma<1
$$

固定初始分布 μ 下的共同折扣回报。全章的策略改进首先指此目标，不是一般和均衡，也不是无重置生命期收益。

### 成立条件与解的含义

- 共同奖励、有界收益和声明的观察模型。
- 推导使用精确价值时单独说明；神经 critic、有限轨迹和截断均属于近似。
- Markov 策略定理需要充分状态；不能将局部单帧观察直接当作满足该条件。

判断准则：在相同执行权限、环境交互和调参预算下比较团队收益，并分别诊断表示误差、探索覆盖、critic 误差与策略漂移。

### 适用边界

- 不把 benchmark 胜率解释为未知搭档上的普遍协调能力。
- 不把所有合作方法称为无通信 CTDE。
- 不把可分散 argmax、优势恒等式与实际训练回报单调改善混为一谈。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：将控制目标扩展到共同回报的多个行动者；本章主体暂不处理零和和一般和均衡。

- 改变信息或数据协议 · [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)：信息不仅不完整，而且分散在不同参与者手中；局部历史是否充分决定可用的策略类。

- 组合不同学习问题 · [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)：变化的同伴会改变边缘过程；进一步加入生命期评价、有限记忆和重置成本才形成相应 CRL 问题。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

联合动作空间大；同一奖励混合了多人的影响；独立探索难以形成配合；同伴更新使旧数据与旧优势失配。

### 本章的核心思路

合作学习先组织能产生配合的联合经验，再将反馈变成各部分可用的更新。baseline 改梯度估计，mixer 改价值类，顺序更新与条件动作生成改联合改善的参照。竞争和开放合作还要在外层评价策略并构建下一轮目标。

1. [声明目标与信息](#lesson-setting)：先写执行者能看什么，再选择可部署的策略类。

2. [采到协作经验](#lesson-exploration)：引入共享行为模式或质量—多样性采样，并计入协调权限。

3. [从联合反馈学习](#lesson-credit)：对比反事实 baseline 与价值分解，不把局部 utility 当成个人奖励。

4. [改善联合策略](#lesson-sequential)：区分旧策略优势、部分更新后的优势和自回归执行。

结论与条件：保证依赖各自的问题条件。结构性质可以逐点检查；性能保证还需要正确评价、充分信息和受控更新。有限样本深网训练不自动继承理想策略迭代定理。

### 相关方法改变了什么

- 集中联合 actor：更容易表达动作相关性，但改变执行权限与规模成本。

- 独立学习：实现简单，但自身看到的边缘过程随同伴策略变化。


<a id="lesson-setting"></a>

## 1 · 两条研究主线，从不同的困难出发

多智能体学习可以沿两条问题主线理解。合作主线首先问：怎样探索到有效的联合行为，怎样把团队反馈转成各个策略的信用信号？竞争及开放合作主线首先问：怎样评价当前策略的不足，怎样构建下一轮对手、伙伴与学习目标，进而在明确条件下获得策略改善？两条主线并非互斥；它们强调学习循环中的不同困难。

| 本组三章 | 核心问题 | 关系 |
| --- | --- | --- |
| 本章：合作学习 | 结构化探索、团队信用与联合策略改善 | 从共同目标出发，组织经验并学习配合 |
| [自对弈与开放式多智能体学习](/zh/continual-rl/foundations/deep/multi-agent-populations/) | 怎样产生对手、评价策略并构建下一轮目标 | 从历史平均和搜索改进进入策略种群，再扩展到未见伙伴合作 |
| [对手建模与递归推理](/zh/continual-rl/foundations/deep/multi-agent-reasoning/) | 怎样利用他者的行为与响应模型学习 | 改善内层响应；不替代外层的能力评价与课程构建 |

Stochastic game 在状态、转移与奖励之外加入多个行动者。共同回报问题追求团队收益；双人零和问题中一方收益等于另一方损失；一般和问题允许利益部分一致、部分冲突。团队回报最大化、利用一个特定对手和逼近 Nash 均衡，不是同一种“最优”。

$$
\mathbf A_t=(A_t^1,\ldots,A_t^n),\quad S_{t+1}\sim P(\cdot\mid S_t,\mathbf A_t),\quad r_i(s,\mathbf a)=\mathbb E[R_{t+1}^i\mid S_t=s,\mathbf A_t=\mathbf a]
$$

联合动作决定下一状态和各人的奖励分布。设计者规定参与者、奖励、观察和通信。本章主体取共同奖励；AgA 一节再讨论个人与集体目标不一致的边界。

两辆车共用一条狭窄通道。它们都希望尽快通过，但同时前进可能堵塞。共享奖励只说明成功标准一致；它既不提供对方的观察，也不告诉双方该怎样打破对称。合作学习必须同时回答“追求什么”和“谁知道什么”。

这里的合作 MARL 指共同回报下的团队学习。它不等同于合作博弈论中的联盟形成与联盟收益分配。Shapley 值等概念可以启发信用分配，却不会自动给出正确的在线策略梯度。

<a id="lesson-notation"></a>

## 2 · Dec-POMDP：共同目标不等于共同信息

$$
S_{t+1}\sim P(\cdot\mid S_t,\mathbf A_t),\quad O_t^i\sim O_i(\cdot\mid S_t),\quad A_t^i\sim\pi_i(\cdot\mid H_t^i),\quad \boldsymbol\pi(\mathbf a\mid\mathbf h)=\prod_i\pi_i(a_i\mid h_i)
$$

这是没有额外共同随机信号、没有本步通信的分散策略类。环境可在同一步联合执行所有动作。局部历史包含已经收到的观察、奖励和自己的既往动作。

Dec-POMDP 描述共同回报、局部观察和分散行动。每人的历史未必足以重建共同 belief，也未必包含同伴看到了什么。把所有实体交给一个全状态中央控制器，是另一种信息条件。若观察还依赖先前动作或状态转移，应相应扩展观察核；上式采用当前状态生成观察的简化记法。

集中训练、分散执行（CTDE）允许训练器读取更多信息。局部 actor 的部署输入仍受限制。下文用 X 表示能够评价未来的集中上下文，例如环境状态与所需记忆。若定理只用 Markov 状态 s，必须先满足其充分性条件；把有记忆 actor 配上只看单帧的 critic，还会产生状态逼近误差。

执行信息约束也适用于归一化器、递归初始状态和共享随机信号。仅在测试时删除 critic，并不足以排除全局信息已经从其他路径进入 actor。

| 困难 | 它改变什么 | 对应思路 |
| --- | --- | --- |
| 协作探索 | 哪些联合轨迹会被采到 | MAVEN 的模式变量、Q-DPP 的联合采样 |
| 团队信用 | 给局部策略的梯度信号 | COMA 的反事实 baseline |
| 分散选动作 | 联合价值的表示与 argmax | VDN、QMIX、QPLEX |
| 同伴更新 | 更新时评价哪个联合策略 | HATRPO/HAPPO、A2PO |
| 相关动作生成 | 执行策略的条件依赖 | MAT 的自回归策略 |

<a id="lesson-derive"></a>

## 3 · 同伴学习，怎样改变我的有效过程

$$
P_t^i(s'\mid s,a_i)=\sum_{\mathbf a_{-i}}P(s'\mid s,a_i,\mathbf a_{-i})\boldsymbol\pi_{-i,t}(\mathbf a_{-i}\mid s)
$$

这里先假定全状态可见、其他人的策略为 Markov 策略。物理转移 P 可以固定，单个学习者看到的边缘转移仍随同伴策略变化；边缘期望奖励也要用同样的策略求平均。

$$
Q_i^{\boldsymbol\pi}(s,\mathbf a)=r_i(s,\mathbf a)+\gamma\mathbb E_{s'\sim P,\,\mathbf a'\sim\boldsymbol\pi}[Q_i^{\boldsymbol\pi}(s',\mathbf a')]
$$

先固定联合策略，才得到固定的价值预测问题。局部历史策略还需把必要记忆放进条件，不能把变化都归入一个平稳的单帧 MDP。

旧 replay 可能来自自己的旧策略，也来自不同的同伴或对手。集中 critic 把联合动作纳入条件，可以缓解一部分归因困难，却不会使未来的目标策略固定。需分别记录行为版本、目标版本和可用的联合信息。

<a id="lesson-games"></a>

## 4 · 策略好不好，先由解概念规定

$$
v^*=\max_{x\in\Delta}\min_{y\in\Delta}x^{\mathsf T}Ay,\qquad \operatorname{Gap}(x,y)=\max_{x'\in\Delta}x'^{\mathsf T}Ay-\min_{y'\in\Delta}x^{\mathsf T}Ay'
$$

A 是有限双人零和博弈的行玩家收益，x 与 y 是对应概率单纯形中的混合策略。gap 非负，在 saddle point 为零；战胜某个固定弱对手不能推出 gap 小。

$$
J_i(\pi_i^*,\boldsymbol\pi_{-i}^*)\ge J_i(\pi_i,\boldsymbol\pi_{-i}^*)\quad\text{对每个 }i,\pi_i
$$

Nash 条件禁止有利的单方偏离。它不保证团队最优或社会福利最大；共享回报的团队也可能停在次优协调均衡。

Matching Pennies 的收益矩阵为 $\bigl(\begin{smallmatrix}1&-1\\-1&1\end{smallmatrix}\bigr)$。若以概率 $p$ 选第一行，两列收益为 $2p-1$ 与 $1-2p$。最大化较小者得到 $p=1/2$，安全价值为零。某一局赢得一分，不等于找到了这种不可利用的策略。

自对弈是产生训练交互的一种方式，不是解概念，也不要求两个角色对称或共享参数。一般和、多均衡和陌生伙伴场景需要相应的响应与交叉评价。本章先固定共同收益，第二章再展开这些外部评价如何产生学习目标。

<a id="lesson-learning-objectives"></a>

## 5 · 内层更新与外层目标构建必须分开

$$
\mu_k=\mathcal M(\widehat U_k,\mathcal P_k),\qquad J_k(\theta)=\mathbb E_{\omega\sim\mu_k}[J(\pi_\theta;\omega)]
$$

外层用策略池或伙伴集合的回报估计 U，选择本轮交互条件分布 μ。ω 可以是对手、伙伴组合或任务条件。内层 RL 对本轮已经声明的目标做更新。

$$
\begin{aligned}J_{k+1}(\theta_{k+1})-J_k(\theta_k)={}&[J_k(\theta_{k+1})-J_k(\theta_k)]\\&+[J_{k+1}(\theta_{k+1})-J_k(\theta_{k+1})].\end{aligned}
$$

第一项在固定目标下改善策略；第二项固定新策略后改变评价分布。它们可以异号。更强的对手让训练收益下降，不等于策略能力退化。

有条件的单调提升必须明确比较什么。内层可在精确评价与受控更新等条件下讨论固定目标的改善；外层若讨论可利用度、安全价值或伙伴泛化，就要使用对应的评价量与证明。扩大策略池、加入多样性或更换训练目标，本身都不保证每轮指标单调变好。合作的探索与信用方法提供内层能力；对手模型可改善响应，开放式学习机制组织评价与目标构建。

<a id="lesson-counterexample"></a>

## 6 · 同样的团队奖励，可以暴露不同困难

$$
R=\begin{pmatrix}8&0\\0&6\end{pmatrix}
$$

两人各选动作 0 或 1；矩阵是双方共同的一步奖励。下列手算是机制例子，不是论文实验结果。

两人策略均匀时，团队期望为 $3.5$。若固定列动作 0，行玩家应选 0；固定列动作 1，则应选 1。最佳动作依赖同伴动作。另一方面，联合最优动作 $(0,0)$ 可以由两个固定的局部策略执行。因此，“整个价值表能否精确表示”和“最优动作能否分散执行”不是同一问题。

再把同伴在训练中的策略从均匀改为大多选 1。没有改变物理世界，自己面对的有效奖励已经改变。旧数据是否还能用，取决于保存了哪些联合条件、评价哪个策略，以及怎样处理采样分布差异。集中 critic 提供条件信息，不会让不断变化的目标价值自动保持不变。

<a id="lesson-exploration"></a>

## 7 · MAVEN：协作探索需要跨 agent、跨时间的相关性

假设一次探索必须让 $n$ 人连续 $T$ 步都执行指定的低概率动作，每人每步独立成功的概率是 $p$，那么整段概率为 $p^{nT}$。加大随机动作比例未必能形成一致的探索计划。若允许先共同抽取一个完整行为模式，则模式概率不需要再为每人每步相乘；代价是引入协调信号与模式表示。

$$
Z\sim p_\phi(\cdot\mid s_0),\qquad q_i=q_i(h_i,a_i,Z),\qquad I(Z;\Tau\mid S_0)\ge H(Z\mid S_0)+\mathbb E\log q_\psi(Z\mid\Tau,S_0)
$$

Z 在一段 episode 内保持不变。这里显式条件于初始状态；若固定 $s_0$，可省略条件。互信息下界用轨迹反推模式，避免不同 Z 都学成同一行为；Tau 表示联合轨迹或其可微表征。

MAVEN 用回报训练高层模式选择，用 TD 学习条件化 utility，并训练轨迹判别器。该结构改变采样行为与表示，不只是给 epsilon 换个调度。共享模式怎样产生、是否依赖初始全局状态、执行时怎样让各人获得模式，都属于协议。固定模式之后的分散动作不等于从来不需要协调。

作者的 2-corridors 实验在训练中关闭短通道，展示模式化探索与适应的联系；它仍是有回合和重置的特定实验，不能据此证明无重置持续学习。代码的 noise_q_learner.py 可看到条件 mixer、轨迹判别器及 TD 与互信息辅助损失。[原文](https://arxiv.org/abs/1910.07483) · [作者代码](https://github.com/AnujMahajanOxf/MAVEN)。

相同 Bernoulli(1/2) 动作边际下，比较独立采样和共同随机比特；共享信号属于显式协议，不是免费的 CTDE 能力。

```python
def exploration_probabilities(agents):
    """Reward is 1 iff all agents choose 1 in a one-step cooperative game.

    Both policies have Bernoulli(1/2) marginals. The second protocol permits
    a shared random bit Z visible to every actor before acting. This is an
    information/coordination assumption, not free extra power for CTDE.
    """
    if type(agents) is not int or agents < 1:
        raise ValueError("agents must be a positive integer")
    return {"independent": 2.0**(-agents), "shared_latent": .5}
```

<a id="lesson-qdpp"></a>

## 8 · Q-DPP：把联合动作看成兼顾质量与多样性的集合

MAVEN 先选一个团队模式。Q-DPP 则在每个 agent 的候选观察—动作对中各选一个，用行列式度量这组选择的质量与互补性。候选集按 agent 分区，约束每个分区恰好选择一项；普通 DPP 允许任意子集，不能直接替代这一约束。

$$
\Pr(Y\mid\mathbf o)={\det L_Y\over\sum_{Y'\in\mathcal C(\mathbf o)}\det L_{Y'}},\qquad Q(\mathbf o,\mathbf a)=\log\det L_Y
$$

集合 Y 对应一个联合动作，C 是每个分区恰选一项的合法集合族。分母不是无约束 DPP 的 det(I+L)。

$$
L_Y=D_YB_Y^{\mathsf T}B_YD_Y,\quad D_{jj}=e^{q_j/2},\qquad \log\det L_Y=\sum_{j\in Y}q_j+\log\det(B_Y^{\mathsf T}B_Y)
$$

本页把 diversity 向量写作 B 的列。第一项表示各候选质量；第二项是张成体积的对数。向量冗余会缩小体积；学习表征不同，不等于物理动作标签必须不同。

训练时按分区采样，并将剩余向量投影到已选向量的正交补，随后以联合 TD 误差更新核。此采样器是近似方法；原文的概率误差界需要分区奇异值的平衡条件。若所有候选的 diversity 向量均为单位向量，且逐分区选出的最大质量候选彼此正交，Hadamard 不等式给出分散最优的充分条件。不能假定训练任意时刻都已满足它。

Q-DPP 是连接价值表示与协调探索的重要工作。原文 §3.6 对完整深度版本保留了进一步工作的边界；作者仓库中的神经网络扩展、数值处理和具体任务配置应分别核对。不能由某个正交特例，直接推出所有 QMIX 参数化都被无条件包含。[原文](https://proceedings.mlr.press/v119/yang20i.html) · [作者工程](https://github.com/QDPP-GitHub/QDPP)。

<a id="lesson-credit"></a>

## 9 · COMA：保持同伴动作不变，只改变自己的动作

团队完成任务后，一个没有起作用的动作也收到了奖励。直接用共同回报乘每人的 score 仍可能是正确的梯度估计，但噪声很大。COMA 不把奖励拆成几份，而是在集中 critic 中问：同伴已经这样行动时，自己的实际动作比自己的策略平均动作好多少？

$$
b_i(X,\mathbf a_{-i})=\sum_{u}\pi_i(u\mid h_i)Q^{\boldsymbol\pi}(X,(u,\mathbf a_{-i})),\qquad A_i^{\rm cf}=Q^{\boldsymbol\pi}(X,\mathbf a)-b_i(X,\mathbf a_{-i})
$$

求和只替换本人的动作。它不是把同伴也重新采样，更不是对实际世界进行了一次因果干预。

$$
\mathbb E_{A_i\sim\pi_i}[b_i\nabla_{\theta_i}\log\pi_i(A_i\mid h_i)\mid X,\mathbf h,\mathbf a_{-i}]=b_i\sum_u\nabla_{\theta_i}\pi_i(u\mid h_i)=0
$$

在声明的条件独立采样下，固定各人历史，baseline 不依赖本次采到的 Aᶦ，因此其 score 期望消去。共享随机变量存在时，条件集合也需包含它。

条件中保留历史有实际意义。仅给环境状态而混合不同控制器记忆时，条件动作分布未必等于式中的局部策略。若各 actor 共享参数，联合 log-probability 的梯度是各人的 score 之和；baseline 仍须逐项停止梯度，但一次共享参数更新可能同时改变所有人的行为。

在共同收益矩阵 $(8,0;0,6)$ 中，同伴选 0、自己均匀采样时，baseline 为 4。选 0 的反事实优势为 4，选 1 为 −4。均值为零并不表示学习信号为零；优势与动作 score 的乘积仍有非零期望。不同 agent 的反事实优势不必相加为团队优势。

**算法：COMA 的核心更新；critic 的多步目标仍要正确处理终止与截断。**

1. 1. 用旧局部策略采样联合动作，保存动作概率和训练上下文。
1. 2. 由集中 critic 计算实际 Q 和逐 agent 的反事实 baseline。
1. 3. 冻结优势的梯度，用 −log πᵢ × Aᵢᶜᶠ 更新局部 actor。
1. 4. 用多步回报或 TD(λ) 训练 critic；按协议更新目标网络。
1. 5. 执行时只保留局部 actor，不读取集中 critic。

原文的收敛分析有正确或兼容 critic 等条件。baseline 的消去证明本身不保证一个学习中的深度 critic 准确，也不保证每个有限样本更新都降方差。原团队 PyMARL 在 coma_learner.py 中将 baseline 和优势 detach，避免误把 actor 更新变成穿过 critic 的另一种梯度。[原文](https://arxiv.org/abs/1705.08926) · [实现](https://github.com/oxwhirl/pymarl/blob/master/src/learners/coma_learner.py)。

另一项精确枚举：Q(a,b)=a+10b，比较三种 baseline 的同均值梯度与方差；此特殊加法任务的零方差不推广到一般 COMA。

```python
def credit_variances():
    """Enumerate four equally likely joint actions, Q(a,b)=a+10b.

    Agent one's Bernoulli logit is zero. Its score is a-1/2. Teammate
    b is sampled independently. E[g]=1/4 for all three baselines.
    A conditional COMA baseline averages ONLY our unobserved action:
    b_COMA(b)=sum_a pi(a)Q(a,b)=1/2+10b.
    This special additive game gives zero variance; that is not a general
    COMA guarantee, nor a causal interpretation of a learned critic.
    """
    result = {}
    for name in ("none", "state_value", "counterfactual"):
        samples = []
        for a in (0, 1):
            for b in (0, 1):
                baseline = {"none": 0., "state_value": 5.5,
                            "counterfactual": .5+10*b}[name]
                samples.append((a-.5)*(a+10*b-baseline))
        mean = sum(samples)/4
        result[name] = {"mean": mean,
                        "variance": sum((x-mean)**2 for x in samples)/4}
    return result
```

<a id="lesson-bicnet"></a>

## 10 · BiCNet：先让动作生成器获得协作信息

COMA 改变训练信号，不改变局部 actor 的信息。BiCNet 选择另一条路线：在采取动作前，让 actor 和 critic 的隐层沿 agent 序列双向传递信息。这里 RNN 展开的位置是不同 agent，不应直接理解为时间记忆。其共享观察、局部输入与通信通道共同决定了执行权限。

$$
\nabla_\theta J=\mathbb E_{d^\mu}\!\left[\sum_i\sum_j \nabla_\theta\mu_{\theta,j}(s)\,\nabla_{a_j}Q_i^\mu(s,\mathbf a)\big|_{\mathbf a=\boldsymbol\mu_\theta(s)}\right]
$$

这是团队目标为各人回报之和时的确定性策略梯度结构，需满足相应可微性与状态占据条件。一个动作 j 会影响多个 Qᵢ；双重求和保留这些交叉影响。

BiCNet 使用局部奖励设计与固定敌方策略等实验条件，因此不是本章统一奖励设定的逐字实例。它启发我们把“动作相互依赖”放入网络，但不能把共享参数推出任意队伍规模的可靠迁移。与后文 MAT 相比，BiCNet 在隐层交流；MAT 显式条件于已生成动作。[原文 §3 与附录](https://arxiv.org/abs/1703.10069)。

<a id="lesson-factorization"></a>

## 11 · VDN 与 QMIX：怎样不用枚举联合动作

价值方法先问一个执行问题：训练时能评价联合动作，执行时能否让各人独立取自己的最大值？VDN 用可加结构，QMIX 放宽为对局部 utility 单调的状态条件 mixer。这里的局部 utility 是学习到的协调变量，未必等于某个真实个人奖励的价值。

$$
Q_{\rm VDN}(X,\mathbf a)=\sum_i q_i(h_i,a_i),\qquad Q_{\rm QMIX}(X,\mathbf a)=f_s(q_1,\ldots,q_n),\quad {\partial f_s\over\partial q_i}\ge0
$$

QMIX 的局部网络可包含递归记忆；全局状态调节 mixer，而不必成为局部执行输入。单调激活与非负混合权重是充分的实现方式。

$$
q_i(h_i,a_i)\le q_i(h_i,a_i^*)\quad\forall i\quad\Longrightarrow\quad f_s(\mathbf q(\mathbf a))\le f_s(\mathbf q(\mathbf a^*))
$$

逐坐标替换为局部最大值不会减小联合估计，因此局部 greedy 的组合是该估计的联合 greedy。这个结论不涉及一次训练前后的真实收益。

可加矩阵必须满足 $Q_{00}+Q_{11}-Q_{01}-Q_{10}=0$，所以 VDN 无法精确表示共同收益矩阵 $(8,0;0,6)$。QMIX 同样不能精确表示其中的条件排序反转。它仍可能学到正确的最优动作；表示误差、数据分布与优化误差共同决定是否做到。

$$
y=r+\gamma(1-d)Q_{\bar\theta}(X',\mathbf a'),\quad \delta=y-Q_\theta(X,\mathbf a),\qquad \nabla_{\theta_i}(y-Q_\theta)^2=-2\delta\,{\partial f_s\over\partial q_i}\nabla_{\theta_i}q_i
$$

目标 y 停止梯度。局部 utility 通过团队 TD 误差和 mixer 导数收到更新，这与 COMA 的动作边际化 baseline 是不同的信用机制。

这里的后继动作 a′ 还需指定选择器。Double 型目标用当前局部 utility 各自选择合法的贪心动作，再交给目标局部网络与目标 mixer 评价；另一种目标直接由目标 utility 选择。二者都能使用单调结构避免枚举联合动作，但数值标签未必相同。若 utility 共享参数，上式是经第 i 条分支传来的梯度贡献，总更新还要加上其他分支。

训练依次执行联合采样、存储转移、重放 batch、构造冻结目标、更新局部网络与 mixer、更新目标网络。必须同时保存每人的终止/存活 mask、合法动作与记忆边界。PyMARL 的 qmix.py 使用绝对值生成非负权重；这不是约束所有网络参数为正。[VDN 原文](https://arxiv.org/abs/1706.05296) · [QMIX 原文](https://proceedings.mlr.press/v80/rashid18a.html) · [mixer](https://github.com/oxwhirl/pymarl/blob/master/src/modules/mixers/qmix.py)。

<a id="experiment-qmix_cooperative"></a>

### 实验：集中训练的价值，能否支持分散选动作

QMIX 的单调混合保证怎样的 argmax 关系？在简单合作任务中是否需要它的额外容量？

**环境与可用信息。** 两个 agent，每回合各收到一个随机比特且保持三步。每人只看自己的比特、时间与身份，动作为 0／1。每个动作匹配自己的比特得团队奖励 0.2，两者都匹配另加 0.6。三步真实终止；集中训练 mixer 可以看两个比特。

**设置。** 本图实际运行 1200 个预算单位，训练种子为 0–4。一个联合转移计一步。局部网络 7→16 ReLU→2，mixer 隐层 8；绝对值超网络权重保证非负。ε=0.2，折扣 0.9，回放 2048，batch 32，Adam 0.003，每 50 次训练更新复制目标参数。

**检验的机制。** 局部最大值经过单调 mixer 构成联合目标；TD 平方误差训练本地网络及 mixer。执行和评价只读取本地 Q，不能把集中状态泄漏进动作。VDN 对照直接求和。

**测量。** 评价枚举四种比特组合及三步局部贪心决策，精确算折扣回报；最优值是 1+0.9+0.81=2.71。评价不用训练 RNG。

```bash
python3 implementations/multiagent/qmix.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/qmix_cooperative/curves.svg)

横轴：environment_steps。纵轴：冻结分散贪心策略的折扣回报。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 600 与 1200 步两者都为 2.71，五个训练 seed 的末点完全一致。这个易合作问题没有展示 QMIX 相比 VDN 的性能优势。

**结论边界。** 额外 mixer 有额外参数和计算。本例没有循环网络、长程隐状态或变化队友；不是 SMAC 复现，也不证明任意非单调联合价值都可表示。

**继续实验。** 构造违反单调性的两动作联合收益表，区分“优化没找到”与“函数类无法表示”。保持执行时观察权限，切勿用集中状态选动作修补结果。

[源码](https://yingwen.io/crl-code/implementations/multiagent/qmix.py) · [逐种子记录](https://yingwen.io/crl-code/results/qmix_cooperative/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/qmix_cooperative/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/qmix_cooperative/curves.json)

<a id="lesson-qplex"></a>

## 12 · QPLEX：约束最优动作的一致性，而非所有动作的排序

QMIX 的充分条件比“最优动作能够分散实现”更强。QPLEX 把局部 utility 写成最大值与非正优势，并用依赖联合动作的正权重组合优势。这样可以改变非最优联合动作的相对值，同时保留最优动作的一致性。

$$
v_i(h_i)=\max_u q_i(h_i,u),\quad A_i^{\rm loc}(h_i,u)=q_i(h_i,u)-v_i(h_i)\le0,\qquad Q_{\rm tot}(X,\mathbf a)=V_{\rm tot}(X)+\sum_i w_i(X,\mathbf a)A_i^{\rm loc}(h_i,a_i),\quad w_i>0
$$

这是说明 IGM 机制的记法；完整 QPLEX 还包含正比例、加偏置的 transformation 与 duplex dueling 结构，不能删掉这些模块后仍声称具有完整表示定理。

所有局部 greedy 的优势均为零，联合值达到 V；任何严格次优局部动作引入负项。这解释了贪心一致性。正权重可以随联合动作变化，所以不要求整个价值表都维持同一个局部排序。

手算：令 $V_{\rm tot}=8$，两人的非最优动作优势均为 −1。对 $(0,1)$ 与 $(1,0)$，把唯一非零优势的权重设为 8；对 $(1,1)$，两个权重均设为 1。得到的表恰为 $(8,0;0,6)$。此例说明非最优动作的表达能力，不证明一般训练收敛。

原文 Proposition 2 的“完整 IGM 函数类”以足够的函数逼近能力为前提，不能读成有限网络表达所有 Dec-POMDP 或保证 SGD 找到最优解。作者实现的 advantage 分支还有 detach 和权重补偿，阅读时需连同 learner 检查。[原文](https://arxiv.org/abs/2008.01062) · [作者实现](https://github.com/wjh720/QPLEX/blob/master/pymarl-master/src/modules/mixers/dmaq_general.py)。

<a id="lesson-fql"></a>

## 13 · Factorized Q-Learning：保留成对交互，换取近似动作求解

还有一种取舍是不强制 IGM，而直接表示自己与同伴的交互。Zhou、Wen 等人的 Factorized Q-Learning（FQL）用低秩成对项近似高阶联合价值，并在同类 agent 间共享网络。它与 VDN 的“把所有局部 utility 相加”不同。

$$
Q_i(s,\mathbf a)\approx q(s^i,a^i)+\eta\,v(s^i,a^i)^{\mathsf T}\left({1\over n-1}\sum_{j\ne i}u(s^j,a^j)\right)
$$

n≥2。q 是独立项；v 和 u 是两个学得的嵌入，内积表示成对交互。η 是交互尺度。高于二阶的交互未被显式保留。

原方法固定同伴上一时刻的动作，近似求自己动作的最大值，并与 TD 更新交替。因此要声明训练和执行中的同伴状态、旧动作或邻域信息。它不是无通信局部 greedy 的精确联合 argmax；非凸价值与并行更新也不能直接继承凸坐标优化的收敛结论。[原文 §3.1–3.3](https://arxiv.org/abs/1809.03738)。

<a id="lesson-mappo"></a>

## 14 · MAPPO：集中评价仍需面对同伴同时变化

若使用策略梯度，就不必通过单调 mixer 求联合 argmax。MAPPO 把 PPO 用于合作学习，以训练期集中价值估计形成优势，执行 actor 仍可只看局部信息。参数共享是一种配置选择；有 agent ID、角色输入或独立参数时，策略类的对称性约束不同。

$$
\rho_{i,t}(\theta_i)={\pi_{\theta_i}(a_t^i\mid h_t^i)\over\pi_{i,\rm old}(a_t^i\mid h_t^i)},\qquad L_i=\mathbb E_{\rm old}\!\left[\min\!\left(\rho_{i,t}\hat A_t,\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_t\right)\right]
$$

优势通常由集中价值与 GAE 得到。这个式子是实际裁剪替代目标，不是逐点保证真实收益改善的约束问题。

考虑共同奖励表 $(0,2;2,-1)$，旧动作是 $(0,0)$。单独改变任一人的动作都把收益从 0 提到 2；两人同时改变却得到 −1。它说明“相对旧同伴的单方改善”不能直接相加。小步更新可以降低风险，但仅有 clip 不能排除此类误差。

集中价值不是永远不变的参照：策略变了，价值目标也会变。MAPPO 的经验效果还依赖 rollout 长度、更新轮数、归一化、存活 mask 与合法动作处理。应先把它作为匹配预算的强基线，而非把任何提升都归因于新网络。[原论文](https://arxiv.org/abs/2103.01955) · [官方 on-policy 工程](https://github.com/marlbenchmark/on-policy)。

<a id="lesson-advantage"></a>

## 15 · 优势分解是望远镜恒等式，不是最优性的捷径

先固定共同回报的 Markov 联合策略，并指定 agent 排列 $i_1,\ldots,i_n$。依次固定前缀动作，对剩余动作按旧策略求平均。每增加一个已固定动作，价值的变化就是一个条件优势。

$$
Q_m^\pi(s,a^{i_{1:m}})=\mathbb E_{a^{i_{m+1:n}}\sim\pi}[Q^\pi(s,\mathbf a)],\quad Q_0^\pi=V^\pi,\quad A_m^\pi=Q_m^\pi-Q_{m-1}^\pi
$$

式中的策略先按状态因子化。具有相关随机信号或自回归依赖时，应使用相应的条件分布定义边际值，不能随意换成独立乘积。

$$
\sum_{m=1}^{n}A_m^\pi(s,a^{i_{1:m}})=\sum_{m=1}^{n}(Q_m^\pi-Q_{m-1}^\pi)=Q^\pi(s,\mathbf a)-V^\pi(s)
$$

中间项逐个相消。这个恒等式不要求 QMIX 的单调性，也不要求联合 Q 可加。

在 $(8,0;0,6)$ 例子中，旧策略均匀，固定第一个动作 0 后价值为 4。第一项优势为 $4-3.5=0.5$；再固定第二个动作 0，第二项为 $8-4=4$；两项之和 $4.5$ 恰为联合优势。COMA 在同一联合动作给两人各 4；它是另一个 baseline 构造，不能与该恒等式互换。

恒等式并不说明依次最大化条件期望就找到全局联合最大值。例如奖励表 $(60,60;0,100)$，旧策略均匀时，第一行条件平均 60 大于第二行 50。先 greedy 选择第一行，随后最多得到 60，而联合最优是 100。逐步非负优势提供改善方向，不等于全局最优搜索。

枚举两人的前缀条件价值，核对局部增量之和等于联合优势。

```python
def advantage_decomposition(matrix, row, col, action):
    """Two-agent telescoping identity at a FIXED reference joint policy."""
    a, b = action
    baseline = value(matrix, row, col)
    prefix = sum(col[j]*matrix[a][j] for j in range(len(col)))
    increments = [prefix-baseline, matrix[a][b]-prefix]
    return increments, matrix[a][b]-baseline
```

<a id="lesson-sequential"></a>

## 16 · HATRPO 与 HAPPO：让后更新者看见前面的策略变化

顺序更新的对象是策略参数，不一定是环境中的动作执行。HATRPO/HAPPO 用同一批旧联合轨迹，依次更新各人的策略；后更新者的目标中计入已更新同伴的新旧概率比。部署时各个局部 actor 仍可同时行动。

$$
F_{m-1}(s,\mathbf a)=\prod_{j<m}{\bar\pi_{i_j}(a^{i_j}\mid s)\over\pi_{i_j}(a^{i_j}\mid s)},\qquad M_m=F_{m-1}A^{\boldsymbol\pi}(s,\mathbf a),\qquad L_m^{\rm clip}=\mathbb E_{\boldsymbol\pi}[\min(\rho_{i_m}M_m,\operatorname{clip}(\rho_{i_m},1-\epsilon,1+\epsilon)M_m)]
$$

F 修正给定状态下前序动作的分布。它不单独修正完整状态占据分布，也没有把旧价值自动变成部分更新后联合策略的价值。

$$
\mathbb E_{\mathbf a\sim\boldsymbol\pi}[F_{m-1}(\rho_{i_m}-1)A^{\boldsymbol\pi}]=\mathbb E_{a^{i_{1:m-1}}\sim\bar\pi,\,a^{i_m}\sim\pi_{\theta_{i_m}}}[A_m^{\boldsymbol\pi}(s,a^{i_{1:m}})]
$$

在固定状态、因子化旧策略及足够概率支持下成立。先积分掉未更新的后缀：$\rho$ 项成为候选当前动作下的 $Q_m-V$；减一项成为旧当前动作平均的 $Q_{m-1}-V$。两者相减正是前缀优势 $A_m$。因此无需为每个前缀另训练一个 critic，旧联合优势加概率比即可表达该期望。

当已更新的前序策略被冻结时，$\mathbb E[F_{m-1}A^{\boldsymbol\pi}]$ 不依赖当前参数。因此去掉减一项，最大化 $\mathbb E[\rho_{i_m}M_m]$ 的梯度不变；再将它裁剪才得到上面的 HAPPO 近似目标。这里的等价只指未裁剪目标的梯度，不能由此推出 clip 有精确改善保证。若共享参数连带改变前序策略，原来应为常数的项也会变化。

**算法：实际工程须保留旧 log-prob，不能随着每个 minibatch 覆盖行为策略版本。**

1. 1. 固定联合策略版本，采一批轨迹并估计旧策略优势。
1. 2. 抽取本轮 agent 更新顺序，初始化每条样本的 factor=1。
1. 3. 对当前 agent，以 factor × 旧优势建立更新目标。
1. 4. HATRPO 近似求解受 KL 限制的步长；HAPPO 优化裁剪目标。
1. 5. 在同一批动作上计算该 agent 更新前后概率比，并乘入 factor。
1. 6. 冻结已更新策略块，继续下一人；全轮后重新采样。

原文 Algorithm 1 的证明针对精确优势、max-KL 惩罚的理想策略迭代，并声明策略正概率等条件。HATRPO 的有限样本二阶近似、HAPPO 的 clip 都是实现层的近似。理论中一整轮联合策略改善，不等于每次 minibatch 或每个中间策略都必然改善。共享参数若会连带改变已冻结策略块，也需要新的分析。[原文](https://arxiv.org/abs/2109.11251) · [作者 runner](https://github.com/cyanrain7/TRPO-in-MARL/blob/master/runners/separated/base_runner.py)。

精确团队矩阵中的同时 / 顺序 best response；它隔离更新参照的差别，不把坐标最优化冒充 HAPPO 或 PPO。

```python
def team_update(matrix, action, sequential):
    """Exact coordinate best response; ties retain the incumbent action.

    Simultaneous updates evaluate both changes against the OLD teammate.
    Sequential updates evaluate the second against the NEW first agent.
    Fixed reward, exact values and independently controlled policies are
    essential here. PPO with shared parameters does not satisfy this test.
    """
    a, b = action
    best_a = max(range(len(matrix)), key=lambda i: (matrix[i][b], i == a))
    other_a = best_a if sequential else a
    best_b = max(range(len(matrix[0])),
                 key=lambda j: (matrix[other_a][j], j == b))
    return best_a, best_b


def update_paths(rounds=8):
    matrix = [[0., 1.], [1., 0.]]  # Team succeeds iff actions differ.
    paths = {}
    for name, sequential in [("simultaneous", False), ("sequential", True)]:
        action = (0, 0)
        rows = []
        for k in range(rounds+1):
            rows.append({"round": k, "actions": list(action),
                         "return": matrix[action[0]][action[1]]})
            action = team_update(matrix, action, sequential)
        paths[name] = rows
    return paths
```

<a id="lesson-a2po"></a>

## 17 · A2PO：顺序更新以后，优势本身也需要重新评价

HAPPO 的 factor 修改当前动作权重。A2PO 进一步追问：更新第 $m$ 人时，前面的人已经改变，那么未来轨迹价值还应以整轮最初的策略为准吗？记 $\boldsymbol\pi^{(m-1)}$ 为前 $m-1$ 人已更新的联合策略。当前更新的自然参照是 A 的这个策略版本，而不是始终使用最初的优势。

$$
\widehat A_t^{(m)}=\delta_t+\sum_{k\ge1}\gamma^k\!\left(\prod_{j=1}^{k}c_{t+j}^{(m)}\right)\delta_{t+k},\qquad c_u^{(m)}=\lambda\min\!\left(1,{\boldsymbol\pi^{(m-1)}(\mathbf a_u\mid s_u)\over\boldsymbol\pi^{(0)}(\mathbf a_u\mid s_u)}\right)
$$

PreOPC 将前序策略变化放入后续 TD 误差的迹系数。δ=r+γbV(s′)−V(s)，b 只在真终止时为零；上述和只在同一序列内部展开，不能接入 reset 后另一回合的残差。有限轨迹、价值逼近和权重截断仍有误差；不是为一批旧样本补一个权重就得到精确新策略优势。

$$
B_m\le {4\gamma\varepsilon_m\over(1-\gamma)^2}\,\alpha_m\sum_{j\le m}\alpha_j+{\xi_m\over1-\gamma}
$$

原文给出的替代目标误差上界形式：α 是逐策略最大总变差，ε 是参照优势绝对值上界，ξ 是优势评价误差上界。这里 m 按当前更新顺序编号。

若替代目标改善超过误差界，才足以推出对应真实改善。较小的步幅不能消除任意大的 ξ；较紧的上界也不是“实际样本效率最高”的证明。A2PO 在实际优化中加入前序比率与当前联合比率的两层裁剪，并研究半贪心更新次序。参数共享、有限轮评价与训练噪声都要另行检查。

对照实现时，先读 `shared_buffer.py` 的轨迹概率比与 weighted returns，再读 `r_mappo.py` 的 `sequential_train`、agent mask 和裁剪；不能只把循环顺序改一下就称为 A2PO。作者仓库说明其重构版本可能与论文表现不一致，因此应固定版本并重跑原协议。[原文 §3–4](https://arxiv.org/html/2302.06205v2) · [作者代码](https://github.com/xihuai18/A2PO-ICLR2023)。

<a id="lesson-mat"></a>

## 18 · MAT：顺序生成动作，不等于顺序更新参数

MAT 把本时刻不同 agent 的观测与动作组织为序列，而不是只把时间轨迹换成 Transformer。编码器处理联合观察，解码器依次生成本步动作；已有轨迹中的动作前缀则可用于并行计算训练概率。它与 HAPPO 的顺序参数更新是不同操作。

$$
\boldsymbol\pi_\theta(\mathbf a\mid\mathbf o)=\prod_{m=1}^{n}\pi_\theta(a^{i_m}\mid\mathbf o,a^{i_1},\ldots,a^{i_{m-1}})
$$

这是概率链式分解，允许相关行动。与 πᵢ(aᵢ|hᵢ) 的独立乘积相比，它需要额外的观察与前序动作信息。

**算法：训练可并行计算各位置；从未知前缀采样时仍需自回归。环境联合执行一次，不代表生成联合动作只需一次网络调用。**

1. 训练：固定行为版本，收集联合观察、联合动作、奖励与旧条件 log-prob。
1. 编码联合观察；将已记录动作右移一位，首位放开始符号。
1. 用因果 mask 一次计算各位置的条件概率，禁止读取当前/未来动作。
1. 以估计优势优化 PPO 型 clip 目标，另训练价值预测。
1. 执行：先编码当前联合观察，再采第一个动作。
1. 把已采动作加入前缀，依次采其余动作，最后交给世界执行。

必须检查执行资源：默认 MAT 的联合观察编码与前序动作条件，不满足“每人只拿自己的私有历史且完全不通信”的协议。集中控制或允许通信时可以使用；局部 actor 变体则需要重新说明失去或保留了哪些依赖。作者 ma_transformer.py 与 transformer_act.py 明确展示这两种数据路径。

优势分解说明理想的逐条件改善怎样累积。实际 MAT 使用估计的联合优势、共享网络和裁剪目标，不能只凭恒等式就保证每次 SGD 后真实收益单调增加。论文所说的加法动作搜索量，也不等于 attention 的完整计算复杂度为线性；还需计编码、重复解码、通信和记忆成本。[原文](https://arxiv.org/abs/2205.14953) · [作者工程](https://github.com/PKU-MARL/Multi-Agent-Transformer)。

<a id="lesson-pmat"></a>

## 19 · PMAT：谁先生成动作，也可以成为学习问题

MAT 规定怎样按给定顺序生成动作。PMAT 进一步学习顺序：关键角色先表明动作，后续角色再据此配合。这里排序的是同一步的动作生成，不是 A2PO 的参数更新顺序。无约束的精确概率链可以采用任意排列；次序的实际作用来自有限模型、学习过程和条件决策方式。

$$
P_\psi(\sigma\mid\mathbf o)=\prod_{m=1}^{n-1}{\exp z_{\sigma_m}\over\sum_{j=m}^{n}\exp z_{\sigma_j}},\qquad \nabla_\psi J_{\rm order}=\mathbb E_{\sigma}\!\left[\widehat A(\sigma)\nabla_\psi\log P_\psi(\sigma\mid\mathbf o)\right]
$$

AGPS 用 Plackett–Luce 分布逐个无放回选 agent；z 来自观察表示的评分器。排序梯度把当前动作网络和优势估计作为固定反馈，不是对离散 sort 直接求导。

PMAT 依次执行编码、评分、采样排列、重排表示、自回归采动作，再还原 agent 索引交给环境。评分器用次序相关的联合优势学习；作者工程的 ranking loss 还使用新旧排列概率比的 PPO 式裁剪。近似优势会带来噪声；该机制不保证找遍 n! 种排列或获得全局最优顺序。联合观察和前序动作的权限与 MAT 相同。[原文 §5](https://arxiv.org/html/2502.16496v1) · [作者 PMAT 工程](https://github.com/NUDT-BI-MARL/PMAT)。

<a id="lesson-relations"></a>

## 20 · 将这些方法放回同一张问题图

| 方法 | 改变的核心对象 | 不能由此直接得到 |
| --- | --- | --- |
| COMA | 动作相关的梯度方差；反事实 baseline | 奖励的唯一公平分配、任意 critic 下的正确梯度 |
| BiCNet | 隐层通信与跨 agent 动作梯度 | 无通信部署、任意队伍规模的迁移 |
| VDN / QMIX / QPLEX | 联合 Q 的可分散表示与 greedy 一致性 | 训练回报单调提高、无结构限制的最优协作 |
| Factorized Q-Learning | 低秩成对交互与近似动作选择 | 高阶交互的完整表达、精确联合 argmax |
| MAVEN / Q-DPP | 联合经验的模式与互补性 | 无额外协调权限的任意探索、单生命期最优性 |
| MAPPO | 局部 PPO 更新与集中价值评价 | 同伴同时更新时的自动兼容 |
| HATRPO / HAPPO | 依次更新参数并重加权前序动作 | 任意近似下的逐次真实改善 |
| A2PO | 部分更新后策略的优势评价 | 零评价误差、裁剪等价于严格信赖域 |
| MAT | 自回归联合策略与并行条件概率训练 | 无通信分散执行、线性总计算、自动全局最优 |
| PMAT | 依观察学习动作生成排列 | 最优排列保证、无额外执行信息 |

这些不是简单的替代关系。一个系统可以用 recurrent state、集中 critic 和结构化探索；也可以把协调策略放入层次控制中。但组合前必须统一行为分布、目标策略版本和执行信息。把多个模块的结论并列起来，不等于得到了组合系统的保证。

<a id="lesson-aga"></a>

## 21 · AgA：当个人目标与集体目标并不相同

前面的主体都先给定共同目标。若每人追求个人得分，而团队关心整体收益，合作还面临目标冲突。AgA 研究可微混合动机博弈：个人损失为 ℓᵢ，集体损失为 ℓc。直接跟随个人梯度可能损害团队；只优化集体损失也未必保护个人利益。

$$
\xi=(\nabla_{w_i}\ell_i)_i,\quad \xi_c=\nabla_w\ell_c,\quad H_c=\nabla_w^2\ell_c,\qquad \widetilde\xi=\xi_c+\lambda(\xi+H_c^{\mathsf T}\xi_c),\quad w\leftarrow w-\alpha\widetilde\xi
$$

个人梯度、集体梯度与曲率修正是三个不同量。Hessian–vector product 等于集体梯度平方范数的一半的梯度；实现不必显式存储整个 Hessian。λ 按原文的符号对齐条件选择，不是总取正值。

原文在集体目标固定点邻域、可微性与曲率条件下讨论方向对齐。它不是任意神经 MARL 的全局收敛定理，也不保证每个人每次更新都增益。此处的“对齐”是目标之间的优化关系，不是 COMA 的信用基线或 Q-DPP 的多样性。[原文 §4.2 与附录算法](https://arxiv.org/html/2402.12416v3)。

<a id="lesson-code"></a>

## 22 · 从可核查机制到作者工程

反事实优势、矩阵博弈与单调 greedy 的标准库机制核。先验证定义，再进入神经训练。

```python
def matrix_value(matrix, row_policy, column_policy):
    probabilities(row_policy); probabilities(column_policy)
    return sum(row_policy[i]*column_policy[j]*matrix[i][j]
               for i in range(len(row_policy)) for j in range(len(column_policy)))


def minimax_2x2(matrix):
    """Row maximizes, column minimizes. Optimize the lower envelope of two lines."""
    a, b = matrix[0]
    c, d = matrix[1]
    candidates = [0., 1.]
    denominator = a-c-b+d
    if denominator != 0:
        crossing = (d-c)/denominator
        if 0 <= crossing <= 1:
            candidates.append(crossing)
    lower = lambda p: min(p*a+(1-p)*c, p*b+(1-p)*d)
    p = max(candidates, key=lower)
    return [p, 1-p], lower(p)


def zero_sum_gap(matrix, row_policy, column_policy):
    row_best = max(dot(row, column_policy) for row in matrix)
    column_best = min(sum(row_policy[i]*matrix[i][j] for i in range(len(matrix)))
                      for j in range(len(matrix[0])))
    return row_best-column_best


def counterfactual_advantage(matrix, row_action, column_action, row_policy):
    probabilities(row_policy)
    baseline = sum(row_policy[i]*matrix[i][column_action] for i in range(len(row_policy)))
    return matrix[row_action][column_action]-baseline


def monotone_joint_greedy(local_values, weights):
    if len(local_values) != len(weights) or any(w < 0 for w in weights):
        raise ValueError('nonnegative mixing weights required')
    local_choice = tuple(max(range(len(q)), key=q.__getitem__) for q in local_values)
    joint = list(itertools.product(*(range(len(q)) for q in local_values)))
    value = lambda acts: sum(w*q[a] for w, q, a in zip(weights, local_values, acts))
    return local_choice, value(local_choice), max(map(value, joint))
```

下载本页两个脚本并置于同目录后执行；机制实验导出图表与原始数值，后一个命令检查既有基础核。

```bash
python3 marl_objectives_lab.py test
python3 marl_objectives_lab.py demo --out results/marl-objectives
python3 extended_foundations_lab.py test
```

本站 [COMA 教学实验](/zh/continual-rl/code/coma_contextual/) 包含一步局部观察、学习的联合 critic 与 actor 更新；[QMIX](/zh/continual-rl/code/qmix_cooperative/) 和 [VDN](/zh/continual-rl/code/vdn_cooperative/) 包含三步合作任务、真正的神经训练及已有学习曲线。前者不是深度 TD(λ) COMA，后两者不是循环 SMAC 全工程；小任务能检查数据流，不能替代复杂协调评估。

连续动作的集中 critic 与局部确定性 actor，可对照 [MADDPG 原论文](https://arxiv.org/abs/1706.02275) 与 [原工程](https://github.com/openai/maddpg)；它不同于 COMA 的离散动作边际化。要检验 minimax、响应与均衡，使用 [OpenSpiel](https://github.com/google-deepmind/open_spiel) 或 [MARL 教材配套代码](https://github.com/marl-book/codebase)。这些入口分别服务算法工程和解概念，不能互作完整复现证据。

| 读代码的顺序 | 要验证的问题 |
| --- | --- |
| 采样器与 controller | 动作使用了哪些观察、共同潜变量或前序动作？ |
| buffer 与目标构造 | 旧 log-prob、同伴版本、mask、terminal 与时间截断是否匹配？ |
| learner 与 mixer / critic | 哪些目标 detach？单调权重、反事实求和、重要性比率在哪一步出现？ |
| 评估与 ablation | 评估是否移除训练专属信息？换新搭档后是否仍有效？ |

原工程入口包括 PyMARL、QPLEX、MAVEN、Q-DPP、on-policy、TRPO-in-MARL、A2PO 和 MAT。依赖及环境版本会影响结果。作者仓库、已验证的机制、本站训练曲线和完整 benchmark 复现是四种不同证据，不能互相替代。

<a id="lesson-branches"></a>

## 23 · 与持续强化学习的衔接

合作 MARL 是现代深度 RL 的一个分支。它不自动等于持续 RL：许多研究固定成员、反复重置环境、在训练后冻结策略。转向长期运行，首先要明确是搭档在学习、成员在更换、世界动力学在改变，还是三者同时发生。

- 状态构造：私有历史还需保留哪些同伴行为证据？角色或意图估计是否比身份标签更可迁移？
- 信用与塑性：长期共享奖励是否使旧角色垄断表示？新成员加入后，既有 critic 和 trace 是否仍对应同一预测问题？
- 经验与模型：replay 中的队伍组成、通信协议和行为策略版本是否可追溯？旧样本在当前联合任务中代表什么？
- 控制评价：冻结策略 cross-play 衡量协调泛化；继续更新时的生命期总收益衡量适应过程。二者需要分别报告。

一个可辨别的研究协议是：先固定世界，只改变同伴策略；再固定同伴，只改变物理转移。让在线学习器与同一初始参数的冻结学习器面对相同观察和交互预算，同时记录更换后的恢复成本、遗忘与不可逆失败。若允许共享训练状态或外部 reset，单独报告其成本和权限。这是从本章提出的实验设计，不是上述算法已经完成的通用 CRL 结论。

每轮从头训练一个新 best response 的开放种群，不等于同一个受限资源智能体持续学习。还需声明历史策略是冻结保存还是继续更新、新伙伴到来能否重置网络，以及经验、记忆、计算和外部设计各花多少资源。[开放式学习与 CRL](/zh/continual-rl/foundations/deep/multi-agent-populations/#lesson-branches)进一步比较这些条件。

<a id="lesson-check"></a>

## 24 · 练习与判断依据

- 在奖励表 (8,0;0,6) 中证明 VDN 不能精确表示，并说明为何这不排除局部固定策略执行联合最优动作。答案：交叉差为 14，不满足可加条件；但双方固定选 0 即可。
- 同一动作上，COMA 两个反事实优势之和为什么不是联合优势？答案：它们固定的同伴条件与边际化顺序不同；只有逐前缀差分才逐项相消。
- HAPPO 在当前动作上乘前序策略比，是否已把未来 critic 变成部分更新后策略的精确价值？答案：没有；当前动作重加权与整段策略评价是不同问题，A2PO 正针对后者。
- 设计一个测试揭露 MAT 的信息泄漏：固定 agent 的局部历史，改变其本不应获得的同伴私有观察，比较动作分布。若分布改变，该实现不符合这次声明的无通信执行协议。
- 为何 Q-DPP 对数行列式需要数值与秩检查？答案：PSD 只保证行列式非负，不保证严格正；线性相关向量会令 log det 为负无穷。数值正则会改变所实现的函数。
- 只观察最终 SMAC 胜率，能判断结构化探索改善了无重置生命期收益吗？不能；需补充有预算、有后果、保留适应过程的协议。



<a id="chapter-code"></a>

## 下载与运行

标准库枚举的协调探索概率、反事实梯度方差与顺序更新机制实验；精确计算不是神经训练曲线，也不是完整论文复现。

[下载 marl_objectives_lab.py](https://yingwen.io/zh/continual-rl/download/marl_objectives_lab.py)

```sh
python3 marl_objectives_lab.py test
python3 marl_objectives_lab.py demo --out results/marl-objectives
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Bernstein et al. · The Complexity of Decentralized Control of MDPs](https://pubsonline.informs.org/doi/10.1287/moor.27.4.819.297)：分散控制与局部信息的原始理论研究；局部历史不等于所有参与者共享的充分状态。

- [Albrecht、Christianos、Schäfer · Multi-Agent Reinforcement Learning](https://marl-book.com/)：免费教材、课程、讲义与练习；用 games 和 solution concepts 区分共同收益、最佳响应与均衡。

- [MARL 教材 · 作者代码](https://github.com/marl-book/codebase)：教材配套 Python 算法；用于学习和机制核对，不应与某篇原始实验工程混同。

- [DeepMind · OpenSpiel](https://github.com/google-deepmind/open_spiel)：游戏、解概念与算法实验平台；需按目标选择环境和评价，不用单次胜率代替均衡差距。

- [Lowe et al. · Multi-Agent Actor-Critic](https://arxiv.org/abs/1706.02275)：MADDPG 的集中 critic、局部 actor 和混合合作—竞争设置；不是 COMA 反事实求和的连续版。

- [OpenAI · MADDPG](https://github.com/openai/maddpg)：原论文工程入口；比较训练时 critic 与部署时 actor 的输入权限。

- [Foerster et al. · Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)：§4 的反事实 baseline 与附录条件；不能把局部收敛论证外推为任意深度 critic 的每步改善。

- [Sunehag et al. · Value-Decomposition Networks](https://arxiv.org/abs/1706.05296)：从团队回报学习局部 utility 的加法分解；utility 不等于给定的个体奖励价值。

- [Rashid et al. · QMIX](https://proceedings.mlr.press/v80/rashid18a.html)：单调混合、状态条件 hypernetwork 与分散 argmax；单调性不是跨训练迭代的性能保证。

- [Wang et al. · QPLEX](https://arxiv.org/abs/2008.01062)：§3 与 Proposition 2；IGM 完整表达以足够函数逼近能力为条件。

- [Mahajan et al. · MAVEN](https://arxiv.org/abs/1910.07483)：共享 episode 潜变量、互信息辅助目标与协调探索；2-corridors 不是一般单生命期证明。

- [Yang、Wen et al. · Multi-Agent Determinantal Q-Learning](https://proceedings.mlr.press/v119/yang20i.html)：§3 的分区约束、质量—多样性表示、正交充分条件和近似采样边界；§3.6 区分完整深度工程与概念验证。

- [Yu et al. · The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games](https://arxiv.org/abs/2103.01955)：MAPPO 的经验评估与实现选择；参数共享和集中 critic 都不是无条件协调保证。

- [Kuba et al. · Trust Region Policy Optimisation in MARL](https://arxiv.org/abs/2109.11251)：Lemma 1、Algorithm 1、Theorem 2–3 与 practical algorithms；分开读理想 max-KL 更新和 HATRPO/HAPPO 近似。

- [Wang et al. · Order Matters: Agent-by-agent Policy Optimization](https://arxiv.org/html/2302.06205v2)：§3.3 PreOPC、误差项 ξ 与 §4 实际裁剪。较紧误差界不直接证明实际样本效率排序。

- [Wen et al. · Multi-Agent Reinforcement Learning is a Sequence Modeling Problem](https://arxiv.org/abs/2205.14953)：优势分解、联合观察编码与自回归执行；实际 PPO clip 不能单凭恒等式获得逐步性能保证。

- [Peng、Wen et al. · BiCNet](https://arxiv.org/abs/1703.10069)：§3 的双向隐层通信、团队确定性策略梯度与固定敌方设定；不是完全无通信执行，也不是平均奖励证明。

- [Zhou、Wen et al. · Factorized Q-Learning for Large-Scale Multi-Agent Systems](https://arxiv.org/abs/1809.03738)：§3.1–3.3 的低秩成对项、同类参数共享和旧同伴动作近似；坐标形式不是精确联合求解保证。

- [Hu、Wen et al. · PMAT](https://arxiv.org/html/2502.16496v1)：§5 的 AGPS、Plackett–Luce 排列概率与评分器梯度；动作生成次序不同于参数更新次序。

- [Li、Wen et al. · Aligning Individual and Collective Objectives in Multi-Agent Cooperation](https://arxiv.org/html/2402.12416v3)：AgA §4.2 与附录 Algorithm 1；混合动机、邻域方向与符号条件不能简写为任意深度博弈中的全局最优保证。

- [Oxford · PyMARL](https://github.com/oxwhirl/pymarl)：COMA、VDN、QMIX 的原团队工程入口；连读 controller、learner、mixer 和环境版本。

- [QPLEX · 作者实现](https://github.com/wjh720/QPLEX)：dmaq_general、dmaq_si_weight 与 dmaq_qatten_learner 对照完整结构，不只摘取一条公式。

- [MAVEN · 作者实现](https://github.com/AnujMahajanOxf/MAVEN)：noise_controller、noise_q_learner 与高层 bandit；区分默认配置和辅助损失变体。

- [Q-DPP · 作者实现](https://github.com/QDPP-GitHub/QDPP)：qdpp_controller、projection_selector、qdpp mixer；训练联合采样与测试局部策略的数据权限不同。

- [MAPPO · 官方 on-policy](https://github.com/marlbenchmark/on-policy)：集中价值、局部策略、价值归一化、GAE 与 rollout / active masks 的工程入口。

- [HATRPO / HAPPO · 作者实现](https://github.com/cyanrain7/TRPO-in-MARL)：base_runner 的随机顺序与 factor 累乘，happo_trainer / hatrpo_trainer 的不同更新。

- [A2PO · 作者实现](https://github.com/xihuai18/A2PO-ICLR2023)：`shared_buffer` 的轨迹修正和 `r_mappo` 的顺序训练；仓库 README 明示重构版本的复现边界。

- [MAT · 作者实现](https://github.com/PKU-MARL/Multi-Agent-Transformer)：ma_transformer、transformer_act 与 mat_trainer；检查联合输入、因果 mask、顺序采样和 PPO clip。

- [PMAT · 作者实现](https://github.com/NUDT-BI-MARL/PMAT)：pma_transformer、transformer_policy 和 mat_trainer；对照评分、排列采样、索引还原与动作网络更新。


<a id="marl-mechanism-experiments"></a>

## 小型实验：把更新规则与评价对象分开

全部结果来自完全列举的有限博弈，没有采样误差或置信区间。它们检验具体机制，不是大型神经算法复现。

### 联合探索

每人选择0或1，全部选择1得奖励1。每人边缘动作概率均为一半；独立采样的成功率为2的负人数次幂，共享随机比特为1/2。共享信号必须在执行协议中允许。六人时分别为1/64和1/2。这说明联合分布的区别，不证明MAVEN或Q-DPP的学习效率。

![联合探索概率](https://yingwen.io/crl-code/marl-objectives/exploration.svg)

### 反事实信用

精确价值Q(a,b)=a+10b；动作独立均匀。第一人的logit score为a−1/2。

| 基线 | 梯度期望 | 梯度方差 |
|---|---:|---:|
| none | 0.25 | 13.8125 |
| state_value | 0.25 | 6.25 |
| counterfactual | 0.25 | 0 |

条件基线为0.5+10b。零方差依赖此加性例子，不是一般COMA保证。

### 更新顺序

团队矩阵((0,1),(1,0))，初始动作(0,0)。同时最佳响应在(0,0)与(1,1)间循环；后人使用前人新动作的顺序响应在一轮后达到收益1。每次更新精确枚举，没有critic或采样误差。

![更新顺序](https://yingwen.io/crl-code/marl-objectives/updates.svg)

反例练习：将矩阵改为((3,0),(0,2))，从(1,1)开始。坐标更新不能逃离这个较差的局部最优。

### 运行与复核

```bash
python3 marl_objectives_lab.py test
python3 marl_objectives_lab.py demo --out results/marl-objectives
```

[源码](https://yingwen.io/zh/continual-rl/download/marl_objectives_lab.py) · [全部数值与源码哈希](https://yingwen.io/crl-code/marl-objectives/results.json)

仅依赖Python标准库，含15项机制检查。图不用于排序大型算法。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-multi-agent#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-multi-agent#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-deep-multi-agent)
<a id="study-connections"></a>

## 与教材主线的衔接

本章是可按问题选择的并列研究分支，不要求先读完其他深度研究分支。

### 一次策略更新，为什么能改善未来？

精确策略改善使用旧策略的真实价值；策略梯度使用与目标匹配的访问分布。值函数近似或梯度估计误差会破坏这些推理的前提。

函数逼近与深度方法：PPO 的动作概率比不等于新策略的状态访问比。连续动作 actor 还会追逐 critic 的误差。限制局部更新尺度与证明实际回报单调提高是不同要求。

持续学习中的研究问题：动作不仅改变世界，也改变未来数据与学习。比较冻结策略不等于比较持续更新的智能体；什么时候应付出当前回报去获得长期有用的经验？

[精确策略改善](https://yingwen.io/zh/continual-rl/foundations/tabular/dynamic-programming/) → [策略梯度定理](https://yingwen.io/zh/continual-rl/foundations/approximation/policy-gradient/) → [TRPO 与 PPO 的近似](https://yingwen.io/zh/continual-rl/foundations/deep/trust-region/) → [持续控制的比较器](https://yingwen.io/zh/continual-rl/algorithms/control/)


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)
- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)
- [元学习与学习规则的适应](https://yingwen.io/zh/continual-rl/algorithms/meta/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)
- [持续学习的智能体架构](https://yingwen.io/zh/continual-rl/construction/architectures/)

对应原始材料：动手学博弈论：第 19–21、23–28 章；Multi-Agent Reinforcement Learning：第 3–6、9–11 章；合作 MARL：Dec-POMDP 与 CTDE；MAVEN、Q-DPP 与团队信用；COMA、VDN、QMIX、QPLEX；HATRPO/HAPPO、A2PO、MAT 与 PMAT。本文为原创讲解，原书、论文与上游代码保留各自许可。
