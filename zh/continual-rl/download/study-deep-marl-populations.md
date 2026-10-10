# 自对弈与开放式多智能体学习：评估、目标与策略种群

现代深度强化学习 · 并列研究分支

自对弈怎样产生课程和训练标签，又怎样通过历史保留、交互评价与策略种群发现值得继续学习的问题？

## 本章内容

- 从自对弈出发，分清最新响应、历史平均与搜索策略标签。
- 用手算博弈树贯通 AlphaGo、AlphaGo Zero、AlphaZero 与 MuZero 的学习闭环。
- 建立评估、目标构建、响应学习与重新评估的循环，区分循环克制与可检验进步。
- 用 gamescape、DO/PSRO 与课程机制研究策略保留，再扩展到 COLE/HOLA 的合作组合。
- 区分可利用度、种群安全价值、陌生伙伴泛化与全生命期收益。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [多智能体合作：结构化探索与信用分配](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent/)：区分联合策略、局部信息与集中训练，再研究对手和伙伴分布。


### 联合策略与局部信息

各参与者按自己的观察历史行动。集中训练不等于执行时可获得所有信息。

### 信息集与完美回忆

信息集汇集当前不可区分的历史；完美回忆要求玩家不忘记曾获得的信息和自己的动作。

### 最佳响应

固定其他参与者后，使自身期望回报最大的策略。实际 RL 通常只能给出近似响应。

### Nash 均衡

没有参与者能通过单方偏离提高自身收益。合作中的均衡不必是最高团队回报。

<a id="problem-definition"></a>

## 本章的问题定义

多个参与者反复交互，自对弈产生新的对手与状态分布；历史平均或搜索将这些经验转成策略学习目标。再由交互评估决定保留哪些响应、下一轮训练谁，从竞争扩展到合作伙伴组合。主体采用固定奖励与动力学、可重置回合；开放的是学习问题的生成过程，不要求规则改变。

### 给定条件与符号

- 游戏规则或可学习模型的交互接口，合法动作与奖励来源。
- 各角色允许使用的观察、历史与通信。
- 任务奖励、回合长度、初始分布与交互预算。
- 初始策略池，以及明确的外部评价准则。

### 需要求解的对象

构造与信息权限一致的响应或搜索标签，训练并保留有用策略，再通过评价暴露当前能力缺口；选择可提交的策略或组合，在固定评价定义下检验逐轮改善。

### 信息与数据权限

外层评估与课程模块可读取训练策略之间的经验回报。执行策略不得因此获取对手私有状态、未来动作或测试伙伴身份。

$$
J_k(\pi)=\mathbb E_{\xi\sim\mu_k}[u(\pi,\xi)]+\lambda_k D_k(\pi),\qquad \mathcal E(\mathcal O_{k+1})\ge\mathcal E(\mathcal O_k)
$$

左式是可随轮次改变的训练目标，μ 是对手或伙伴分布，D 是可选的多样性项。右式是希望建立的单调改善性质，不是假定已成立的算法定理。O 是明确声明的输出，E 是各轮相同、越大越好的评价准则。

### 成立条件与解的含义

- 仅在两人零和分支使用极小极大性质。
- 精确响应、完整对手空间与无遗憾条件逐项核对。
- 课程生成和最终评估分开；双方消耗的资源均计入预算。

判断准则：说明改善对象和评价空间；给出相应条件下的保证，或用独立数据检验逐轮改善及退步。同步报告估计误差、测试覆盖和全部交互成本。

### 适用边界

- 不把种群增长直接视为有限资源单一生命的完整解。
- 不把零样本合作直接称为部署阶段的持续参数学习。
- 不将不同单调性结果合并成“每一代都更强”。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)：不只决定当前动作，也决定与谁交互以发现学习机会。

- 改变评价目标 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：参与者分布改变后，期望回报目标随之改变。

- 组合不同学习问题 · [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)：策略档案保存旧响应，删除或蒸馏会改变可用组合。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

参与者分布决定当前学习目标；策略更新又会改变下一轮可选的参与者。没有分清评价空间，就可能把池内成功误当作稳健性或泛化。

### 本章的核心思路

用评估发现能力缺口，把缺口转成下一轮学习目标，再训练响应并重新评估。目标构建连接当前不足与下一步学习；保留旧解和选择新输出为建立单调改善提供可能。

1. [由自对弈产生课程](#lesson-self-play)：区分最新对手、历史平均与信息集上的响应学习。

2. [由搜索构造训练标签](#lesson-search-policy)：同一博弈树上分开先验、访问计数与真实终局。

3. [定义进步并评估现状](#lesson-evaluation)：声明提交策略还是种群，固定 Gap、安全价值或测试伙伴回报的定义。

4. [把竞争弱点变成响应目标](#lesson-psro)：PSRO 用元策略构建训练对手；响应学习后扩池并重新评估。

5. [将兼容性缺口转为伙伴分布](#lesson-cole)：COLE 的求解器评价策略间关系，训练器优化给定伙伴课程。

6. [评价完整伙伴组合](#lesson-hola)：HOLA 用高阶组合表示两两关系无法表达的合作机会。

结论与条件：嵌套己方集合的最佳安全价值不下降；精确有限零和 DO 有相应终止证书。平均无遗憾界、局部偏好结果与实际神经训练分别有条件，均不自动给出全寿命或陌生伙伴回报单调改善。

### 相关方法改变了什么

- 冻结伙伴／对手分布：评价简单，但不能主动寻找新的弱点或兼容性缺口。

- 无差别增加种群规模：增加资源，不一定增加响应覆盖，也未定义哪些策略值得保留。

- 仅评价最后网络：忽略种群的组合能力，也忽略整个学习期间的成本。


<a id="lesson-setting"></a>

## 1 · 问题从哪里来：由交互产生学习目标

给定对手与奖励，强化学习回答怎样改进策略。对手也在学习时，还要回答：下一轮应当回应谁，针对什么弱点学习，怎样判断整体能力提高了？开放式多智能体学习把这些问题纳入学习系统，以交互评估持续产生新的学习目标。

Balduzzi 把前一个层次的问题称为“问题从哪里来”（the problem problem）：能优化一个给定目标，还没有说明下一轮应构造哪个有用目标。在两人竞争中，固定一个对手就得到一个目标函数；换对手，即使环境奖励完全不变，也改变了自己必须学会解决的问题。这给出从求解任务走向构造任务的一条具体路线。[Balduzzi 等，ICML 2019](https://arxiv.org/abs/1901.08106)

$$
\phi(v,w)=-\phi(w,v),\qquad J_w(v)=\phi(v,w),\qquad J_q(v)=\sum_{w\in\mathcal P}q(w)\phi(v,w).
$$

本节的竞争主例是两人对称零和游戏，双方策略来自同一集合。$\phi$ 为行方收益；$w$ 是固定对手，q 是对手混合。改变 q 就改变训练目标，外层仍须判断这个目标是否有助于既定评价。混合策略在回合开始抽取，参数平均不是这里的混合。

先从最小闭环开始：双方用当前策略下棋，依据结果更新；新策略又产生新的对局。随后有两个具体问题：怎样保留已经遇到的行为，避免只追逐最新对手？怎样把落子前搜索形成的决策分布变成可学习标签？本章先把这两条路径推导完整，再研究单个最新策略为什么可能绕圈，以及策略种群如何把评估、目标与响应组织起来。

合作也需要构建学习目标，但缺口从“被谁克制”变为“与谁不能配合”。温颖及合作者的 COLE 将这条思路扩展到合作兼容性：评估策略间关系，生成伙伴课程，再学习新的合作策略。HOLA 进一步处理多个伙伴的组合关系。[COLE](https://proceedings.mlr.press/v202/li23au.html) · [HOLA](https://arxiv.org/abs/2409.08767)

$$
u_i(\pi_i,\pi_{-i})=\mathbb E\left[\sum_{t=0}^{H-1}\gamma^tR^i_{t+1}\right]
$$

$u_i$ 是固定环境、起点、回合长度和各方策略后的期望回报。策略可以依赖局部历史，不要求物理状态完全可见。

$$
\begin{aligned}J_i(\pi_i;q)&=\mathbb E_{\pi_{-i}\sim q}u_i(\pi_i,\pi_{-i}),\\\operatorname{BR}_i(q)&\in\arg\max_{\pi_i}J_i(\pi_i;q).\end{aligned}
$$

q 在每局开始抽取完整对手策略，局内冻结；BR 是对此分布的最佳响应。自对弈改变 q，搜索则可在当次决策中增加计算。两者都须遵守各自的信息权限。

$$
\widehat U_k=\operatorname{Evaluate}(\mathcal P_k),\quad \mu_k=\operatorname{Target}(\widehat U_k,\mathcal P_k),\quad \pi_{k+1}=\operatorname{Learn}(J_k),\quad \mathcal P_{k+1}=\operatorname{Retain}(\mathcal P_k,\pi_{k+1})
$$

P 是策略档案；U 是交互评估；Target 将弱点或兼容性缺口转成训练分布。Learn 使用有限预算求响应，Retain 保留所需策略；随后重新评估。评估器、目标构建器和学习器具有不同职责。

例如，已有策略能战胜对手甲，却持续输给乙。评估先定位乙揭露的弱点；目标构建提高相关挑战的训练权重；响应学习形成新策略；重新评估检查它是否弥补弱点，也检查旧能力是否保留。合作中可以把乙换成难以配合的伙伴。关键不是盲目扩大种群，而是让每轮学习针对可识别的能力缺口。

| 问题 | 学习对象 | 最终检验 |
| --- | --- | --- |
| 固定团队合作 | 共同训练的参与者 | 遵守执行信息限制时的团队回报。 |
| 竞争种群学习 | 新响应与历史对手混合 | 对完整或明确测试对手空间的可利用度。 |
| 开放合作 | 学习者与训练伙伴组合 | 未共同训练伙伴上的收益与失败分布。 |
| 部署期持续学习 | 同一学习者长期遇到新旧参与者 | 全寿命收益、适应损失、记忆与恢复成本。 |

这里专指 open-ended multi-agent learning。开放的是由交互结果不断构建后续学习问题的过程，不要求奖励函数或物理世界不断改变。它是开放式人工智能的一条具体研究线，不是整个 open-ended AI 的定义。它也不等同于 CRL：种群可在可重置回合中训练新策略；CRL 还需声明同一学习者的生命期、资源与知识更新。

<a id="lesson-self-play"></a>

## 2 · 自对弈：让参与者产生自己的课程

$$
\pi_i^{k+1}\approx\operatorname{BR}_i(q_{-i}^{k}),\qquad q_{-i}^{k}=\delta_{\pi_{-i}^{k}}
$$

最新策略自对弈用对手当前版本定义本轮目标。实际可只做少量梯度步，而非求完整 best response。

每轮固定训练对手，采集对局，再更新策略。轮换与同时训练有不同数据分布。共享网络是对称任务中的一种实现，不是自对弈的定义；不对称游戏仍可为不同角色维护不同策略。

石头—剪刀—布中，只回应最新纯策略会在石头、布、剪刀之间反复循环。每次响应都正确，最新策略却始终容易被利用。“战胜上一轮”因此不等于全局进步。

$$
\operatorname{Gap}(x,y)=\max_a(Ay)_a-\min_b(x^\top A)_b
$$

A 是有限双人零和游戏的行玩家收益矩阵，x、y 分别为行、列玩家的混合策略。gap 等于双方最优单方偏离收益之和；零 gap 才说明该策略对是均衡。

收益回答“对这一分布表现如何”，gap 回答“还有哪些偏离能获利”。只报告相邻训练版本的胜率，会漏掉循环和未遇见的克制策略。

自对弈还在选择下一批学习经验。对手改变，会使原本足够的防守失效，也会使某些状态更常出现；学习者于是面对新的预测与控制问题。固定规则的棋类因此同样可以产生课程，开放式学习并不要求另造新规则。但这个过程可能循环、遗忘旧弱点，或只在狭窄策略族内相互适应，所以“课程会改变”尚未回答“能力会怎样扩大”。先看一种保留历史的办法。

<a id="lesson-fictitious"></a>

### 2.1 · 虚拟博弈：回应历史平均

$$
\beta_i^{k+1}\in\operatorname{BR}_i(\bar\pi_{-i}^{k}),\qquad \bar\pi_i^{k+1}=\frac{k\bar\pi_i^{k}+\beta_i^{k+1}}{k+1}
$$

有限正规形游戏中，平均完整策略的概率分布；k 是已纳入平均的响应数。

历史平均保留已经遇到的行为，避免只追逐最新对手。经典虚拟博弈在有限双人零和等游戏中有平均策略的收敛结论，不是一般和游戏或最新策略的普遍结论。

序列游戏不能简单地对每个信息集的动作概率做等权平均。几乎不到达某处的策略，不应与经常到达的策略在该处获得相同权重。Fictitious Self-Play 用实现概率处理这个问题。

$$
\bar\pi_i(a\mid I)=\frac{\sum_k w_k r_i^{\pi_i^k}(I)\pi_i^k(a\mid I)}{\sum_k w_k r_i^{\pi_i^k}(I)}
$$

I 是信息集，r 是玩家自身动作对到达 I 的概率贡献。完美回忆下，它与先按 w 抽取完整策略的混合实现等价；分母为零处可任意定义。网络参数平均不能替代此式。

两步手算：完整策略甲始终左，乙始终右，回合开始各半抽取，则路径 LL、RR 各有一半，LR、RL 不发生。若每一步都等权平均两种动作，四条路径各有四分之一。完美回忆的第二步信息集记住自己先前选了左还是右：到达左分支时甲的自身实现概率为一、乙为零，上式便在该分支只取甲的左动作；右分支反之。由此恢复完整策略混合的路径分布，而不是独立重抽。

FSP 接着把“求响应”和“保存平均”拆成两个可学习问题：用强化学习从对局中近似求响应，用监督学习从响应者的行为流拟合平均策略。监督数据的产生方式承担了上述实现权重的工作，不是先平均网络参数。原文先给出 extensive-form fictitious play 的精确关系，再引入采样和函数近似；后一层不能继承前一层的所有保证。

这里还有采样条件：为无偏地拟合等权历史响应，原文要求各响应提供相同数量的对局，并面对同一个固定且有覆盖的采样对手。实际 FSP 改用逐渐变化的平均对手以聚焦相关状态，这会引入额外偏差。Reservoir 可以保留行为流，却不能自动消除行为流本身的分布偏移。

<a id="lesson-nfsp"></a>

### 2.2 · NFSP：响应学习与平均策略学习

NFSP 用 Q 网络近似响应，用另一网络拟合自己的历史响应行为。前者学习怎样获胜，后者保留已经采用过什么。平均策略网络不是用来拟合对手动作的。

$$
\sigma_i=(1-\eta)\bar\pi_i+\eta\beta_i,\qquad 0<\eta<1
$$

完整策略层面的 anticipatory mixture；原算法在每局开始选择平均或响应模式，不是每一步重新抽模式。

$$
L_{\rm SL}(\vartheta)=\mathbb E_{(I,a)\sim\mathcal M_{\rm SL}}[-\log\bar\pi_\vartheta(a\mid I)]
$$

只将执行响应模式时的自身行为写入监督记忆。Reservoir sampling 使有限记忆近似保留整段行为流，不保证精确保存全部策略。

**算法：两种记忆的对象、采样方式和用途不同。**

1. 每局开始，以概率 η 执行近似响应 β，否则执行平均策略。
1. 所有转移写入 RL replay。
1. 仅将响应模式下的自身 (信息集, 动作) 写入 SL reservoir。
1. 用 Q-learning 更新响应；用交叉熵更新平均策略。
1. 冻结平均策略，独立计算或估计可利用性。

改用最近窗口会改变所拟合的平均对象。非线性逼近、有限记忆与不精确响应也会改变理论条件。NFSP 是 FSP 的可扩展近似，不是“给 DQN 加一个网络就保证 Nash”。

NFSP 的实际对手也在按上述混合行动，因此 Q 学到的是对这一 anticipatory 行为分布的近似响应，而非一个始终冻结的精确历史平均。小比例响应行为让数据包含平均策略正在改变的方向。η 调整这两类经验的占比，不是自动保证稳定的学习率。评价时要说明用的是平均网络、响应网络还是混合行为；三者不同。

<a id="lesson-cfr"></a>

### 2.3 · CFR：从反事实遗憾组织自对弈

历史平均还可以由另一种更新过程产生。CFR 问：在自己能够决策的信息集，若换一个动作，累计会少后悔多少？它据此组织下一轮策略，而非像 FSP 那样显式训练一个完整最佳响应。COMA 的反事实 baseline 服务于策略梯度的方差缩减；二者不是同一种更新。

$$
v_i^\sigma(I,a)=\sum_{h\in I}\rho_{-i}^\sigma(h)\sum_{z\succeq ha}\rho^\sigma(ha,z)u_i(z)
$$

h 是信息集内历史，z 是终局。第一个权重含对手与 chance 到达 h 的贡献，排除自身到达概率；之后强制选 a，再按 σ 继续。此值不是归一化的条件期望。

$$
\begin{aligned}r_i^k(I,a)&=v_i^{\sigma^k}(I,a)-\sum_b\sigma_i^k(b\mid I)v_i^{\sigma^k}(I,b),\\R_i^K(I,a)&=\sum_{k=1}^K r_i^k(I,a),\\\sigma_i^{K+1}(a\mid I)&=\frac{[R_i^K(I,a)]_+}{\sum_b[R_i^K(I,b)]_+}.\end{aligned}
$$

分母为零时取均匀分布。这是原始累计遗憾的 regret matching；CFR+ 的逐轮截断是另一更新。

有限、完美回忆的双人零和游戏中，局部反事实遗憾控制整体外部遗憾，再由双方平均遗憾界控制平均策略的均衡误差。采样、神经近似和游戏抽象各有额外条件；局部指标小不能独立证明任意深度策略达到均衡。

<a id="lesson-search-policy"></a>

## 3 · 搜索响应：一个状态，两种标签

历史平均和遗憾更新利用多轮对局积累策略。另一条自对弈路线则在每次落子前花计算量改善当前决策。先限定问题：两位玩家轮流行动，完整局面可见，规则和合法动作已知，终局收益互为相反数。玩家 1 在根状态选 L 或 R，玩家 2 看见这一步后选 a 或 b；终局给玩家 1 的收益分别为 L→a：+1、L→b：−1、R→a：0、R→b：+1。

若能看完这棵小树，玩家 2 会在每个分支最小化玩家 1 的收益，故 L 的保底值为 −1，R 为 0，根状态应选 R。直接把四个终局取最大值会把对手当成帮助自己的人。大树无法穷尽时，网络给出动作先验与叶节点价值，搜索用有限预算逐渐修正局部决策。

$$
\begin{aligned}(p_\theta(s),v_\theta(s))&=f_\theta(s),\\ \pi(a\mid s)&=\frac{N(s,a)^{1/\tau}}{\sum_bN(s,b)^{1/\tau}}.\end{aligned}
$$

p 是网络给搜索的先验，v 估计当前行动玩家的后续终局收益，N 是本次搜索的根边访问次数。τ>0 控制按计数行动的温度；这里 π 是搜索输出，不是网络输出。

![同一两层零和博弈树产生搜索动作分布与实际终局标签。先验偏向 L，32 次给定搜索使访问计数偏向 R；给定实际路径 L→b 的玩家标签为 −1 与 +1。](https://yingwen.io/crl-figures/concept-selfplay-search-labels.svg)

原创精确机制算例。圆中 1、2 标行动玩家，终局方格统一写玩家 1 收益；蓝实线仅标给定实际对局，紫虚线为规则模拟。概率柱来自 [确定性计算](/crl-code/figures/selfplay-search-labels.mjs)：无训练、无随机采样、无 Dirichlet 噪声，模拟均走到终局，不是完整 AlphaZero 实现。桌面与手机使用相同数据。

为使每一步可核对，配套程序取根先验 p=(0.8,0.2)，两处对手先验均为 (0.5,0.5)，每条未访问边的 Q 初始化为零。每次沿下面的教学评分选最大项，同分取第一项，一直走到真实终局。回传时，根边累计玩家 1 收益，对手边累计其相反数。这省去了扩展叶的网络评价，只隔离访问计数与标签的机制。

$$
\begin{aligned}\operatorname{score}(s,a)&=Q(s,a)+U(s,a),\\U(s,a)&=p(a\mid s)\frac{\sqrt{1+\sum_bN(s,b)}}{1+N(s,a)}.\end{aligned}
$$

PUCT 型教学评分，探索系数固定为一。Q 始终采用该节点行动玩家的视角。加一的初始计数约定、无噪声和终局模拟均已给定，不声称逐行复现论文搜索。

第一次选 L→a，根获得 +1；对手从该动作看到 −1，所以第二次改试 L→b，根得到 −1。前三次根访问 L，第四次转向 R。32 次后根计数为 (6,26)，τ=1 给 π=(3/16,13/16)。根边的样本均值却为 (−2/3,1/13)，既不是 π，也不等于精确 minimax 值 (−1,0)。若只给一次模拟，搜索反而输出纯 L；有限预算没有逐状态改善保证。

现在固定一次实际对局为 L→b，它仍是按上述 π 行动时可能发生的路径。根状态的策略标签是整向量 π，不只是实际选中的 L；等对局结束，根状态价值标签 z=−1，而玩家 2 所见后继状态的 z=+1。胜负来自真实完成的对局，不是把搜索中最乐观的模拟结果当成事实。

$$
\begin{aligned}\ell(\theta)&=(z-v_\theta(s))^2\\&\quad-\sum_a\pi(a\mid s)\log p_\theta(a\mid s)\\&\quad+c\lVert\theta\rVert_2^2.\end{aligned}
$$

AlphaGo Zero / AlphaZero 的联合训练形式。一次拟合中 π 和 z 是给定标签；通过网络求梯度，不反传穿过访问计数或真实对局。c 为权重正则系数。

若 p 是 softmax，策略交叉熵对 logits 的梯度就是 p−π。算例根状态为 (49/80,−49/80)，梯度下降因而降低 L、提高 R，即使这局实际下了 L。取根价值预测为零，价值平方误差对 v 的导数为 2；在玩家 2 的状态则为 −2。把所有状态都贴同一个符号的终局标签，会把一方的成功训练成另一方的成功。

闭环至此才完整：网络提供先验和评价，搜索分配计算得到 π，双方据此产生新局面与终局 z，再用这些标签改网络。网络改变后，搜索访问和自对弈状态分布也改变。π 是当次搜索的改进目标，NFSP 的平均策略则是在保留历史响应，两者虽都用监督损失，却在学习不同对象。

<a id="lesson-alphago-lineage"></a>

### 3.1 · AlphaGo 到 AlphaZero：改变的是学习闭环

AlphaGo（2016）并非从零开始的上述单网络闭环。它先用人类棋谱训练监督策略，再以该策略初始化强化学习策略，用自对弈胜负做策略梯度；训练对手从较早策略版本中抽取。另一个价值网络回归强化学习策略对局的结果。最终搜索结合监督策略先验、价值网络和快速 rollout；人类动作、强化学习胜负与搜索分别承担不同角色，不能把它追述成“用搜索计数直接训练所有策略”。

AlphaGo Zero（2017）取消人类棋谱和快速 rollout，把策略与价值放到同一网络。规则仍由程序准确提供；搜索以网络评价叶节点，根访问分布成为策略标签，真实自对弈结果成为价值标签。它把当次昂贵搜索得到的决策信息压缩进网络，再用改后的网络支持未来搜索。原训练流程会评价新版本，达到对当前最好版本 55% 胜率的门槛才替换自对弈者；这是一项版本管理规则，不是对所有对手的均衡证明。

AlphaZero（2018）把同类闭环用于围棋、国际象棋和将棋，适应平局与不同规则、动作编码；不再依赖围棋旋转反射对称的数据增强，也不使用上述最好版本替换门槛，自对弈持续使用最新网络。论文分别训练三个实例，而非一个共享网络连续学会三种棋并保留旧知识。因此“同一算法跨游戏”与“单一智能体持续积累能力”必须区分。

| 方法 | 策略训练信号 | 搜索读取什么 | 仍然给定什么 |
| --- | --- | --- | --- |
| AlphaGo 2016 | 先人类动作监督；再对局胜负的策略梯度 | 监督策略先验、独立价值网络、rollout | 棋谱、围棋规则、合法动作与终局判定 |
| AlphaGo Zero 2017 | 搜索访问分布；价值学终局结果 | 单网络的 policy / value | 围棋规则与状态、动作编码 |
| AlphaZero 2018 | 搜索访问分布；价值学含平局的结果 | 同类单网络、已知规则搜索 | 每个游戏的规则与表示；分别训练 |
| MuZero 2020 | 搜索分布、价值目标与实际奖励 | 学得的隐状态转移及其预测 | 真实交互与奖励、动作接口；根合法动作 |

这些方法的共同点，是由对手与搜索生成局面课程，而任务规则、输赢和动作接口已经给定。非零和奖励、同时行动或私有信息会改变问题本身；把隐藏牌当作搜索可见状态，或给对手也最大化自己的价值，都不是合法迁移。下一节的学习模型也不会消除这些信息条件。

<a id="lesson-muzero"></a>

### 3.2 · MuZero：搜索需要预测什么

前面的树搜索知道走一步会得到哪个真实棋盘。MuZero 改为学习一个供规划使用的隐状态模型：先从真实观察历史得到根表示，再在隐空间按候选动作展开。预测器只需为控制提供奖励、价值与动作分布，不要求隐状态等于真实棋盘，也不以重建每个像素作为原算法的必需目标。

$$
\begin{aligned}h_t^0&=h_\theta(o_{\le t}),\\(\widehat r_t^{k+1},h_t^{k+1})&=g_\theta(h_t^k,a_{t+k}),\\(p_t^k,v_t^k)&=f_\theta(h_t^k).\end{aligned}
$$

h 是表示函数，g 是学得的动力学，f 是预测函数；上标 k 数隐空间展开步，不是新的真实观察。r̂ 预测走该动作产生的奖励。这里用 h 表示隐状态以免与真实状态混淆。

沿真实数据记录的动作序列展开模型，每一层预测与相应时间的实际奖励、价值目标和搜索策略标签对齐。棋类价值目标可用最终胜负；Atari 使用带 bootstrap 的多步回报，不能把两个实验都写成只学终局 ±1。搜索在学得模型内试算候选行动，真正落子后仍须从环境接收新的观察与奖励。

回到两层树：若模型把 L→b 的 −1 错预测为 +1，增加搜索预算只会更深入利用这个错误模型，不能靠“想得更多”把错误奖励变正确。真实 L→b 经验能为模型提供纠错目标；但尚未访问分支的误差与搜索选择又相互影响。这使覆盖、模型失配和控制相关预测成为核心问题，而不是把 model-free 网络前面加一棵树。

“不使用已知转移规则规划”不等于没有环境接口。原文 Appendix A 明确：MuZero 在搜索根部仍由环境提供合法动作掩码，内部模拟节点不做同样的规则合法性过滤；终局通过吸收式预测学习处理。作者公开伪代码也把 Environment.step、Game.legal_actions 和网络函数留作待实现接口。读者不能把这份伪代码称为可直接运行的完整复现。

至此已看到怎样从当次交互、历史行为或搜索产生训练目标。仍有一个共同缺口：局部响应学得更好，不等于最新策略对所有对手更强。下面回到完整策略之间的收益关系，用[循环与 gamescape](#lesson-gamescape)说明为什么应保留多个回应方向，再定义可比较的评价和下一轮目标。MuZero 的棋类与 Atari 结果不提供一般不完全信息博弈的均衡保证，也不保证课程永不停止。

<a id="lesson-gamescape"></a>

## 4 · 循环与 gamescape：保留哪些回应方向

先看何时固定对手就够用。若 $\phi(v,w)=f(v)-f(w)$，对手只贡献与 v 无关的常数，因此所有对手都要求提高同一个评分 f。稍广的情形是 $\phi(v,w)=\sigma(f(v)-f(w))$，其中 $\sigma$ 为奇函数且单调递增。排序仍由 f 决定，但遇到太弱或太强的对手时，饱和响应可能使梯度很小，自对弈可以提供较合适的挑战。这是对一种博弈结构的分析，不是断言棋类只含单一技能轴。

第二节的剪刀石头布循环正好违反这个前提。现在不只观察“石头→布→剪刀→石头”的版本序列，而要记录每个版本对所有参考对手的收益。最后策略仍能被评价：规定对手分布后，其期望回报很明确；缺少的是一个脱离对手仍适用的“普遍更强”排序。保留旧策略，则能形成最后一个网络不具有的混合与回应选择。

$$
A=\begin{pmatrix}0&-1&1\\1&0&-1\\-1&1&0\end{pmatrix},\qquad a(v)=\bigl(u(v,R),u(v,P),u(v,S)\bigr).
$$

行列次序均为石头 R、布 P、剪刀 S，行方胜得 1、负得 −1。固定布作对手，就只优化第二列；固定剪刀则优化第三列，两个目标偏好的策略不同。

把一个策略对各参考对手的收益排成向量，就看见它能应对什么。每回合以权重 x 抽取策略时，收益向量是这些行向量的加权平均。所有可得平均组成凸包：经验 gamescape（经验博弈空间）。它保留的是对一组对手的响应关系，不是神经参数之间的距离。

$$
\mathcal G(\mathcal P;\mathcal R)=\operatorname{conv}\!\left\{\bigl(u(\pi,r)\bigr)_{r\in\mathcal R}:\pi\in\mathcal P\right\},\qquad a(x)=\sum_{\pi\in\mathcal P}x(\pi)a(\pi).
$$

$\mathcal P$ 是可选策略池，$\mathcal R$ 是固定参考对手集合。原文 EGS 取 $\mathcal R=\mathcal P$；为比较不同轮的几何图，这里先固定参考坐标。若参考对手也变了，要补测旧策略在新列上的收益后再比较。

先预测：复制十个石头策略，会扩大这个集合吗？加入一个此前没有的剪刀，为什么可能扩大它？图中只画对石头与布的两个坐标；这个特定游戏的第三个坐标始终等于前两者之和取负，因此没有丢失信息。一般博弈的二维投影则可能遮住新的战略方向。

![上方是剪刀石头布的收益矩阵与只回应前任产生的循环。下方在对石头、对布的共同收益坐标中，石头是一个点，石头加布是线段，加入剪刀形成三角形。复制石头仍在同一点，均匀混合在原点。](https://yingwen.io/crl-figures/marl-population-gamescape.svg)

由正文给定矩阵精确计算的原创图。蓝色点、紫色线段、青色三角形对应逐渐扩大的可选策略池；原点是三个纯策略的均匀混合。坐标是期望收益，不是 Elo、参数距离或训练轨迹。灵感与术语见 [Balduzzi 等（2019）§2–3](https://arxiv.org/abs/1901.08106)。[计算代码](/crl-code/figures/marl-population-geometry.mjs)。

功能 gamescape 则把有限参考坐标换成整个策略空间上的收益函数，描述所有潜在对手与目标。这说明经验图的缺口：在当前池里看似多余的策略，可能能抵挡一个尚未发现的对手；在当前池里占优的策略，也可能只是循环的一部分。判断冗余必须说明相对于哪些对手。完整函数上的等价比有限矩阵行相同强得多。

传递与循环还可在有限矩阵中明确拆开。对 n 个策略采用均匀参考测度，令 $r_i=\frac1n\sum_j A_{ij}$，则 $A^{\rm trans}_{ij}=r_i-r_j$，$A^{\rm cyc}=A-A^{\rm trans}$，且 $A^{\rm cyc}\mathbf1=0$。剪刀石头布的 r 全为零，所以全部留在循环项中。这个分解依赖参考测度；传递子类的行凸包是一条线段，退化时为一点。非线性单调变换保留共同技能排序，却不保证任意经验矩阵仍为一维。

<a id="lesson-evaluation"></a>

### 4.1 · 评价策略，与评价一个种群

单调提升是本章希望建立的性质。先把第 $k$ 轮提交的对象记为 $\mathcal O_k$，并固定越大越好的评价函数 $\mathcal E$。对象可以是一个策略、一个混合，或一个允许重新选择混合的种群。逐轮改善要求 $\mathcal E(\mathcal O_{k+1})\ge\mathcal E(\mathcal O_k)$；它比仅要求最终一轮更好更强。

| 学习问题 | 可选择的评价准则 | 怎样理解进步 |
| --- | --- | --- |
| 竞争中的提交策略对 | 完整游戏的负 Gap | 更少的单方偏离获利空间。 |
| 竞争中的策略档案 | 面对固定完整对手空间的最佳安全价值 | 可从档案组合出的最强防守不退步。 |
| 开放合作中的提交策略 | 固定伙伴分布上的期望团队回报 | 在相同配合要求下，合作收益提高。 |

训练分布可以改变，比较进步的准则仍需可比。若同时更换考题和策略，分数变化混合了两个原因。保留旧策略可使旧解仍然可选；是否能找到并提交更好的解，则取决于评估和优化精度。以下先给出确实成立的集合层面结论，再说明它与最新策略改善的区别。

先取两人零和游戏，行方最大化 $u$，列方最小化 $u$。$\Pi_1,\Pi_2$ 是完整策略集合；$\mathcal P_1,\mathcal P_2$ 是已发现的有限池；$x,y$ 是完整策略上的混合。每回合抽一个策略，与每一步重抽策略，不是同一个执行过程。

$$
\begin{aligned}\operatorname{Gap}(x,y)&=\max_{p\in\Delta(\Pi_1)}u(p,y)-\min_{q\in\Delta(\Pi_2)}u(x,q),\\v(\mathcal P_1)&=\max_{x\in\Delta(\mathcal P_1)}\min_{q\in\Delta(\Pi_2)}u(x,q).\end{aligned}
$$

Gap 衡量提交策略对的单方获利空间；部分文献将其一半称为 exploitability。种群安全价值则允许重新选择池内混合，但仍面对完整对手空间。

$$
\mathcal P_1\subseteq\mathcal P'_1\quad\Longrightarrow\quad v(\mathcal P'_1)\ge v(\mathcal P_1)
$$

旧混合仍可行，因此扩张己方集合不能降低最佳安全价值。这是集合包含的结论，不是神经网络训练定理。

它没有保证实际求解器找到这个最优混合，也没有保证最新单策略更强。若同时扩大对手池，受限游戏的数值可能下降，因为新对手暴露了旧漏洞。只在自己的档案中评价，会把尚未发现的反例当作不存在。

普通剪刀石头布中，双方池里只有石头时，池内 Gap 为零；完整游戏中双方都出石头的 Gap 为二。另一个加权游戏甚至允许“加入精确响应后，受限均衡的完整 Gap 变大”，下一节给出手算。

Balduzzi 等还定义了直接比较两个种群的相对表现。两边可以大小不同：先计算一个 m×n 的交叉收益矩阵，再允许双方各自在自己的池中选混合，取这个有限零和游戏的值。它衡量两池交手的可保证收益；不用先选出各自一个所谓冠军。

$$
v(\mathcal P,\mathcal Q)=\max_{x\in\Delta(\mathcal P)}\min_{y\in\Delta(\mathcal Q)}x^\top A_{\mathcal P,\mathcal Q}y.
$$

有限零和游戏的所有 Nash 均衡给出同一个游戏值，因此均衡可能不唯一并不使这个值含糊。它不是双方任取一个池内 Nash 再交手；这里求解的是两池之间的交叉游戏。

例如，小池 $\mathcal P=\{R,P\}$ 与大池 $\mathcal Q=\{S,S,S\}$ 的交叉矩阵是 $\left(\begin{smallmatrix}1&1&1\\-1&-1&-1\end{smallmatrix}\right)$。小池选 R 可保证 1，故 $v(\mathcal P,\mathcal Q)=1$。大池的三份剪刀没有增加选择能力。把参考对手改成完整的 $\{R,P,S\}$，小池最佳混合却是 $(1/3,2/3)$，最坏收益为 −1/3；加入 S 后可以用均匀混合把安全值提高到 0。

$$
\mathcal G(\mathcal P;\mathcal R)\subseteq\mathcal G(\mathcal P';\mathcal R)\quad\Longrightarrow\quad v(\mathcal P,\mathcal Q)\le v(\mathcal P',\mathcal Q),\quad\mathcal Q\subseteq\mathcal R.
$$

同一参考收益坐标下，旧混合的收益向量仍可行，因此面对固定对手池 Q 的最优安全值弱单调。对完整策略空间作此断言，需要函数层面的包含或直接保留全部旧策略。

凸包严格变大，也可能只是不下降：上例加入剪刀后，对三份剪刀的游戏值仍为 1。这个性质沿包含链成立，不把任意种群都排成传递的优劣次序；只含一个纯策略的三个种群仍然剪刀石头布相克。相对种群表现、完整可利用度与种群能利用哪些特定弱点，是三个需要分别报告的量。

<a id="lesson-psro"></a>

## 5 · DO 与 PSRO：评价、生成反例、扩张集合

竞争中的目标构建有一个直接依据：当前策略仍能被什么行为利用？用这样的对手组织响应学习，再把学到的行为纳入种群，可以逐步补齐战略覆盖。DO 与 PSRO 将这个直觉变成可执行的评估—响应循环。

Double Oracle 从受限集合开始。先解受限游戏，再到完整策略空间寻找有利偏离。PSRO 将完整行为策略作为元博弈的动作，用 RL 近似求响应。原框架允许不同元求解器；这里只推导两人零和的 Nash 版本。[PSRO 原文](https://arxiv.org/abs/1711.00832)

$$
\begin{gathered}A^k_{ab}=u(\pi_1^a,\pi_2^b),\qquad(x_k,y_k)\in\operatorname{NE}(A^k),\\\pi_1^{k+1}\approx\arg\max_{\pi_1}u(\pi_1,y_k),\qquad\pi_2^{k+1}\approx\arg\min_{\pi_2}u(x_k,\pi_2).\end{gathered}
$$

矩阵通常由多次交互估计。固定对方元策略并从中抽完整对手，响应训练就形成一个可用单智能体 RL 求解的子问题。

**算法：响应训练与响应评价使用不同随机数据，避免把训练噪声当成新的弱点。**

1. 保留双方策略池，以及每个策略版本的身份。
1. 评估新增策略对，记录 payoff、样本数与估计误差。
1. 在受限游戏中求元策略 x、y。
1. 固定 y 训练行方响应；固定 x 训练列方响应。
1. 独立评价新响应的获利幅度，再加入档案。
1. 重新求元策略，并在完整或独立对手集合上评价。

$$
A=\begin{pmatrix}0&-1&1\\1&0&-10\\-1&10&0\end{pmatrix}
$$

行方最大化。初始双方池均为 {0}，唯一池内均衡是动作零，完整 Gap 为 1−(−1)=2。动作一是对动作零的精确最佳响应。

双方加入动作一后，受限矩阵为 $\left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)$，唯一均衡双方都选一。面对完整游戏，它的 Gap 为 $10-(-10)=20$。没有采样误差，也没有神经优化失败；变化来自受限均衡没有防守尚未纳入的动作二。

有限两人零和游戏中，若受限均衡与完整最佳响应都精确，双方均无有利偏离就是终止证书。近似 RL 没找到新策略，只说明本次搜索失败；不能证明不存在更好的响应。

$$
\hat B-\hat C\le\operatorname{Gap}(x,y)\le\hat B-\hat C+\epsilon_1+\epsilon_2
$$

令 $\hat B=u(\hat p,y)$、$\hat C=u(x,\hat q)$ 是找到的响应的真实期望收益。若已知行方响应距完整最大值至多 $\epsilon_1$，列方响应距完整最小值至多 $\epsilon_2$，便得到上下界。未知 oracle 误差时只有左边，找不到获利偏离不构成上界证书；若 $\hat B,\hat C$ 也来自有限评价，还需计入其误差。

<a id="lesson-rectified-nash"></a>

### 5.1 · 不怕被利用以后，为什么继续学习？

回应受限 Nash 的意义可以从几何中证明。对称零和池的均衡值为零，若一个新策略对该池的 Nash 混合获得严格正收益，它的旧对手收益向量就不可能在旧凸包内：旧池中的任何混合都不能对这个均衡获利。因此，正收益证据给出了严格扩大经验 gamescape 的充分条件。前提是收益和均衡已按所用精度核对；搜索没有找到正收益则不构成反证。

但是，拥有一种安全的混合，还没有描述能怎样利用不同对手。考虑一个连续的循环博弈：策略是单位圆盘里的二维向量，收益为两个向量的有向面积。三点等距放在半径 0.5 的圆上，均匀混合的均值为原点；它对任意对手都得零。此时回应这个混合的目标对所有候选策略都相同，无法指出向哪个方向继续学习。三点却仍各有能击败和会输给的对手。

$$
\phi(v,w)=v_1w_2-v_2w_1,\quad \|v\|_2,\|w\|_2\le1,\qquad \mathbb E_{w\sim q}\phi(v,w)=\phi(v,\mathbb E_qw)=0\ \text{当 }\mathbb E_qw=0.
$$

这是受原文 disc game 启发的给定连续博弈。原点/零均值混合为完整游戏的均衡；每个边缘点有正负收益的对手，所以安全性与对特定对手的利用能力分开。

Rectified Nash response 先求当前 Nash 混合，再从其中权重为正的各个成员出发创建候选。固定这个混合后，各成员优化的是同一个正部收益目标；起点不同，当前能战胜的对手便不同，因而局部响应方向不同。它试图从各自已有专长出发扩展能力，再把候选加入原池。原文称之为 game-theoretic niching。这里不是为每个成员任意定义一个不同的函数，也不是为整个种群挑同一个最难对手。

$$
L(v)=\sum_j q_j[\phi(v,w_j)]_+,\qquad [z]_+=\max(z,0),\qquad \nabla_vL\big|_{v=w_i}=\sum_{j:\phi(w_i,w_j)>0}q_j\nabla_v\phi(w_i,w_j).
$$

从每个 $q_i>0$ 的成员 $w_i$ 出发优化；式中零收益处采用零次梯度，仅为本页算例约定。梯度因此由当前可战胜的对手贡献。原文算法写的是正部收益目标；若工程实现先固定筛选对手并重新归一化，再训练很多步，需要说明它与边训练边改变正部区域的区别。

先预测：对零均值 Nash 混合做一步梯度上升，会移动圆盘中的点吗？改用正部目标以后，每个点朝哪个方向移动？下图只计算一次给定步长的更新，不调用 RL 训练。

![左图三点在半径0.5圆上，均匀Nash目标处处为零，三个梯度均为零。右图对正部收益目标作一步更新，三个新点伸出旧三角形。两组三点的均匀混合仍是原点，因此相对种群游戏值仍为零。](https://yingwen.io/crl-figures/marl-population-rectified.svg)

原创确定性计算：初始角度为 0、120、240 度，Nash 权重各 1/3，步长 0.6，零收益处取零次梯度。旧三角形用灰线、新候选用紫线，箭头为解析梯度更新，非人类数据或论文性能曲线。保留旧点后的并集扩大了可选收益函数；图中相对值仍为 0，显示几何扩大不保证每个评价严格提高。[Balduzzi 等（2019）§4、图 3](https://arxiv.org/abs/1901.08106) · [计算代码](/crl-code/figures/marl-population-geometry.mjs)。

原文还用 $d(\mathcal P,q)=q^\top[A_{\mathcal P}]_+q$ 描述 Nash 支持成员之间相互利用的强度，称为有效多样性。普通剪刀石头布的均匀 q 给出 1/3，而收益全零的池给出 0；两者的对称零和游戏值都是 0。该指标以所选 Nash 权重为条件，衡量的对象不同于策略数、任意参数距离或完整可利用度。

正部目标也可能追逐不能泛化的局部克制，生成大量狭窄专家。原文在 Blotto 与可微 Lotto 上比较了这类方法，计算预算主要按响应 oracle 调用计数，未计入随种群增长的交叉评估成本。其观察支持这两类资源分配游戏中的方法比较；用于更大种群或持续系统时，应把评估、存储、选择混合和删除旧成员的成本一并计入。保留旧解保证的是候选集合不缩小，有限优化器是否找到并提交更有用的回应仍要检验。

<a id="lesson-epsro"></a>

### 5.2 · EPSRO：响应与对手混合共同更新

标准 PSRO 分开进行元游戏评估和冻结混合上的响应训练。EPSRO 使用 unrestricted–restricted game，简称 URR：一方搜索完整策略类，另一方只在旧池中混合，两者共同更新。URR 是问题设定，不是另一篇独立算法论文。[EPSRO](https://arxiv.org/abs/2202.00633)

$$
\max_{p\in\Delta(\Pi_1)}\min_{y\in\Delta(\mathcal P_2^k)}u(p,y)
$$

行方不只回应冻结权重；列方也调整旧策略混合，继续暴露行方弱点。反向的 URR 可用于另一方。

实践中，一边用 RL 改进 response，一边以在线无遗憾方法更新元策略；旧信息可用于 warm start，响应可以流水并行。“减少完整元博弈评估”不等于无需交互：训练和收益估计仍消耗数据。

$$
\operatorname{Gap}_{\rm URR}(\bar p_T,\bar y_T)\le\frac{R_1(T)+R_2(T)}{T}
$$

有限双线性零和 URR 中，Gap 的行方最大化遍历完整集合，列方最小化仅遍历该轮受限池。若双方在这些集合上的累积外部遗憾为 R，时间平均策略满足此界。它不是对双方完整策略空间的无条件 Gap 界。

$$
\begin{aligned}R_1&=\max_p\sum_{t=1}^T u(p,y_t)-\sum_{t=1}^T u(p_t,y_t),\\R_2&=\sum_{t=1}^T u(p_t,y_t)-\min_y\sum_{t=1}^T u(p_t,y),\\\frac{R_1+R_2}{T}&=\max_pu(p,\bar y_T)-\min_yu(\bar p_T,y).\end{aligned}
$$

先在同一轮固定的双方可行集合中定义累积外部遗憾；相加时实际对局收益相消，双线性把时间平均移入 u。精确遗憾给出等式，遗憾上界给出上面的不等式。扩张集合、近似收益或非线性参数平均不能不经分析直接代入。

平均策略界不保证最后一次参数更新。EPSRO 的精确 URR 均衡、嵌套集合及遗憾分析也不能替换成任意深度 RL oracle。声称单调改善时，必须说明评价对象、策略空间、求解精度，以及输出是平均策略还是最新网络。

<a id="lesson-diversity"></a>

### 5.3 · 多样性：去过哪里，与能应对谁

两个策略可以走不同路线，却输给同一类对手；也可以只有一个关键动作不同，却克制不同对手。参数距离、占据分布和响应向量不是同一种多样性。Diverse-PSRO 研究 payoff 表征上的 DPP 指标；BD/RD 区分行为与响应两个层面。[Diverse-PSRO](https://proceedings.mlr.press/v139/perez-nieves21a.html) · [BD/RD](https://arxiv.org/abs/2106.04958)

$$
a(\pi)_j=u(\pi,\pi_2^j),\qquad D_R(\pi)=\min_{w\in\Delta(\mathcal P_1^k)}\|a(\pi)-(A^k)^\top w\|_2^2
$$

响应多样性衡量新 payoff 向量到旧向量凸包的距离。它依赖作为坐标的参考对手，不能直接代表所有未知对手。

$$
J_k(\pi)=u(\pi,y_k)+\lambda_B D_f(\rho_{\pi,y_k}\Vert\rho_{x_k,y_k})+\lambda_R D_R(\pi)
$$

这里用共同教学记号，ρ 是归一化折扣占据分布。若原文采用占据测度下平均奖励，权重需按回报尺度调整。加正则后不再是纯任务收益最佳响应。

凸包外也可能有很差的策略：新向量每个坐标都更低，仍可能离凸包很远。多样性要服务于发现有效响应，而不能自己充当最终成绩。复制策略能增加人数，却不增加可表达的混合。

合作的结构化探索与此相关，但尺度不同。Q-DPP 在联合行动层组织探索；BD/RD 在完整策略种群层组织探索。前者不自动生成稳健课程，后者也不自动解决轨迹内的团队信用分配。[Q-DPP](https://proceedings.mlr.press/v119/yang20i.html)

<a id="lesson-curricula"></a>

### 5.4 · NAC：学习怎样安排下一轮课程

如果响应训练只有有限预算，最适合学习的对手混合未必是当前受限 Nash。NAC 参数化元求解器，再根据若干轮后的可利用度训练它。[Neural Auto-Curricula](https://arxiv.org/abs/2106.02745)

$$
\mu_k=f_\psi(A^k),\quad\theta_{k+1}=\operatorname{BRLearn}(\theta_k,\mu_k),\quad\min_\psi\mathbb E_{G\sim\mathcal D}\operatorname{Gap}_G(x_T(\psi),y_T(\psi))
$$

元参数决定课程；课程影响响应；响应又改变未来矩阵。外层评价必须考虑这条依赖，不是只对当前对手概率做一次动作层策略梯度。

原文研究展开优化、元梯度和进化策略等途径。内层在一个游戏中增加策略，外层跨训练游戏更新课程生成器。跨游戏泛化与一个游戏中的逐轮改善不同。

固定外部评价仍然存在。自动学习的是为它服务的课程规则，不是凭空创造最终好坏标准。展开长度、训练游戏分布和响应质量限定了结论。[作者代码](https://github.com/waterhorse1/NAC)

构造目标还可以改变什么？NAC 改的是与谁交互；温颖参与的 Learning to Design Games 将可改变对象推进到环境转移。以迷宫为例，玩家努力缩短抵达终点的路程，设计者在允许的地图集合内安排墙，让最优玩家也要走得更远。固定玩家时，原问题的状态—动作对可作为新状态，下一状态作为新动作；于是原转移分布成为一个待优化的策略。[IJCAI 2018 原文 §3、图 1–2](https://www.ijcai.org/Proceedings/2018/0426.pdf)

$$
\min_{\theta\in\Theta_{\rm allowed}}\max_\varphi\mathbb E[G\mid\pi_\varphi,M_\theta],\qquad s^{E}=(s,a),\quad a^{E}=s',\quad\pi^{E}_\theta(s'\mid(s,a))=P_\theta(s'\mid s,a).
$$

左式为论文的对抗环境设计目标，显式写出允许的环境集合；右式是固定玩家策略时的 dual MDP 对应。它说明怎样优化转移，不保证交替近似训练找到全局鞍点。

连续可调的转移概率可用转移梯度；离散墙体则由另一个生成过程逐步产生合法地图，不能直接对墙是否存在求同样的梯度。迷宫可达性等约束必须在可行集合中声明。更低的当前回报不自动等于更多未来学习进展：若目标是课程，应另测学习后的收益、可解性及生成和求响应的总成本。运行复现前，还需确认作者实现与依赖；通用 RL 库不能代替这个环境生成过程。

<a id="lesson-composition"></a>

### 5.5 · 策略生成：继承、融合与局内组合

| 方法 | 改变哪个环节 | 必须保留的边界 |
| --- | --- | --- |
| Fusion-PSRO | 按元策略权重融合历史参数，再训练 response | 参数平均不等于策略概率混合；网络的局部对齐条件重要。 |
| Conflux-PSRO | 状态级路由调用不同子策略，再蒸馏 | 路由和子策略同时更新，不是固定 MDP 的表格算法。 |
| XDO / NXDO | 在信息状态级组织受限游戏与组合 | 相邻团队工作；神经近似不自动继承表格 XDO 的界。 |

$$
\theta_{\rm init}=\sum_jx_k(j)\theta_j,\qquad\pi_{\theta_{\rm init}}\ne\sum_jx_k(j)\pi_{\theta_j}\quad\text{一般情况下}
$$

Fusion-PSRO 改的是 response 的起点。即使权重来自均衡，也不能据此宣称融合网络仍是均衡或拥有所需状态覆盖。

$$
\pi_{\rm route}(a\mid h)=\sum_j\mu(j\mid h)\pi_j(a\mid h)
$$

逐信息历史路由与回合开始一次抽取不同，后者保留跨时间相关性。若路由读取部署者不允许获得的全局状态，则改变了原问题。

XDO/NXDO 由 McAleer 等提出，不属于温颖共同作者系列。它在信息状态上组织组合，与 EPSRO 联动元策略和 response 的做法不同。组合扩大候选类，但近似优化仍可能失败；应给路由、蒸馏和评估同等预算。[Fusion-PSRO](https://doi.org/10.3233/FAIA251106) · [Conflux-PSRO](https://arxiv.org/abs/2410.22776) · [XDO/NXDO](https://arxiv.org/abs/2103.06426)

<a id="lesson-cole"></a>

## 6 · COLE：把合作不兼容转成伙伴课程

前面的竞争主线通过评估寻找克制关系，再据此构建训练目标。转向合作后，核心循环保留，评价关系改变：需要发现哪些约定或伙伴组合尚不能兼容，并使后续策略学会配合。这是 COLE 扩展开放式多智能体学习的出发点。

共同回报矩阵 $C=\left(\begin{smallmatrix}3&0\\0&2\end{smallmatrix}\right)$ 有两种成功约定。总选左和总选右在各自自博弈中表现良好，彼此组队却得零。提高 self-play 分数不等于解决 cross-play。

COLE 用图记录策略间合作收益。节点是策略，边权是互动评价；不是物理连接。求解器识别兼容性缺口，生成伙伴分布，再由 RL 训练新策略。新的互动结果重新进入图，使评价、目标构建与策略改善连接起来。原始论文是 ICML 2023，JAIR 2024 为扩展。[COLE](https://proceedings.mlr.press/v202/li23au.html)

$$
M^k_{ij}=u(\pi_i,\pi_j),\quad q_k=\operatorname{Solver}(M^k),\quad J_k(\pi)=\mathbb E_{j\sim q_k}u(\pi,\pi_j)+\alpha u(\pi,\pi)
$$

第一项学习与难配合伙伴合作，第二项保留自我配合目标。外部任务奖励可以不变，训练伙伴分布持续改变。

COLE-SV 的 graphic Shapley value 评价策略在合作图中的作用，以构建课程；不是将每步团队奖励分摊给动作。COMA baseline、团队成员 Shapley 信用与种群课程评价是三个不同对象。

局部偏好收敛分析要求 response oracle 满足规定质量，不保证任意 PPO 调用都产生合格响应，更不保证所有陌生人类回报单调提高。当前池内兼容性、冻结测试伙伴表现和人机评价应分开。[JAIR 扩展](https://doi.org/10.1613/jair.1.15884)

还需审查原证明的一个具体推步。ICML 版本 Appendix B 写出偏好中心性 $\eta_T=\eta_0\prod_{t=0}^{T-1}(1-\alpha_t)$，其中 $\alpha_t$ 是当轮相对改善量，不是 self-play 的混合超参。从这个乘积推到零，这一步还需要累计改善条件，例如 $\sum_t\alpha_t=\infty$（或统一正下界）。仅有每轮 $0<\alpha_t<1$ 不够：取 $\alpha_t=1/(t+2)^2$，乘积等于 $(T+2)/(2(T+1))$，极限为一半。正文 Theorem 4.4 未给出这样的统一下界；每代有限、随后扩张的策略池也不自动提供它。因此这里保留 COLE 构建课程的机制与 Overcooked 实验观察，不把该推步或 PPO 近似视为无条件收敛证书。[原文 §4.1–4.2、Appendix B](https://proceedings.mlr.press/v202/li23au/li23au.pdf)

<a id="lesson-hola"></a>

### 6.1 · HOLA：评价完整的伙伴组合

COLE 说明怎样从两方合作关系构建课程。多个伙伴同时参与时，目标构建还需知道“哪些组合”值得学习；不能只把所有两两评价分别提高。HOLA 将评估单元扩展为整个团队组合，再用评估结果形成下一轮课程。

一个任务同时需要侦察者、搬运者和协调者。任意两两组合都可能失败，三者一起才能成功。此时“某个伙伴好不好”不能独立回答；学习机会属于整个组合。

$$
W_k(i_1,\ldots,i_m)=u(\pi_{i_1},\ldots,\pi_{i_m}),\qquad J_k(\pi)=\mathbb E_{\boldsymbol z\sim q_k}[u(\pi,\boldsymbol z)]
$$

超边对应一个策略组合，权重为共同回报。课程分布选择伙伴元组，而非必然独立抽样的单个伙伴。

HOLA-Drone 用超图及偏好关系组织评估，生成下一轮伙伴课程，研究多无人机与未见伙伴的配合。超图是博弈关系的表示，不等于策略必须使用超图神经网络。[HOLA-Drone](https://arxiv.org/abs/2409.08767)

组合数随种群和团队规模增长。未评价的超边不能当作零收益，共享成员的组合也不是独立样本。评价采样和不确定性应计入算法成本。

后续 Multi-Robot Open Adaptive Teaming 考察环境、伙伴及团队规模变化。无微调迁移表明已训练策略可以适应；它不等于同一实体在部署中持续更新参数。[后续原文](https://arxiv.org/abs/2607.04972)

<a id="lesson-test-distribution"></a>

### 6.2 · 独立评价：课程不能同时充当考卷

训练分布由方法选择，测试分布应由研究问题确定。若每种方法挑自己的容易伙伴，平均回报不能比较。即便共用伙伴，不同伙伴能实现的最佳团队收益也可能不同。

$$
\operatorname{Regret}(\pi,z)=\max_{\pi'}u(\pi',z)-u(\pi,z)
$$

伙伴条件 regret 比较同一个伙伴下的收益差。复杂环境的最大值通常未知，只能用独立训练的近似响应作参考。

ZSC-Eval 通过行为偏好奖励生成候选伙伴，按所需最佳响应的差异筛选集合，再用 BR-Prox 等衡量适应差距。它使“哪些配合能力还没有学会”成为可研究的评价问题。[ZSC-Eval](https://arxiv.org/abs/2310.05208)

参考响应并不等于真实最优。BR-Prox 分母近零、收益可正可负或参考训练不足时，不能机械套用比值。必须同时报告原始回报、分伙伴结果、区间和参考训练预算。测试生成过程也只是部署分布的代理。

AT-Drone 将独立伙伴评价落实到无人机追逃：任务中有 N 个可学习追踪者、M 个不受学习者控制的陌生伙伴，以及逃逸者与障碍物。四组配置改变逃逸者和障碍物数量；训练与评价须声明哪些策略能更新、是否显式建模伙伴，以及测试伙伴来自哪里。§3.3 的三个测试池分别为 Greedy 伙伴、两种水平的 IPPO 自博弈伙伴及二者的混合；文中还讨论 VICSEK 伙伴，但不能把候选生成方式与这三个具体测试池混写。[AT-Drone，CoRL 2025](https://proceedings.mlr.press/v305/li25a.html) · [方法版本 §3.1–3.3](https://arxiv.org/html/2502.09762v2)

它将成功率 SUC、碰撞率 COL、成功回合的平均步数 AST、平均奖励 REW 分开记录。前两者分别回答是否完成追逐、是否发生碰撞，AST 只回答成功以后花了多久。因此，失败增加也可能伴随更短的 AST；这是条件平均的数学边界，不是该论文报告的方法结论。追踪系统支持的有限场地验证也不是任意部署环境的安全保证。[作者项目页](https://sites.google.com/view/at-drone)提供任务说明与演示，但尚未列出公开训练代码；使用时需另行确认复现入口。

人类会主动迁就 AI。首次与重复配合、角色和沟通权限、主观负担都应记录，以区分 AI 适应、人类适应和共同适应。短期人机高分不直接证明长期协作稳定。

<a id="lesson-series"></a>

## 7 · 扩展问题：数据、团队结构与一般和目标

开放式多智能体学习不只面临“再增加一个策略”的问题。无法继续采样时，响应必须受已有数据约束；双方各有一支团队时，需要声明能否相关地选择联合策略；利益不完全对立时，则要改变解概念。以下作品分别扩展评估、目标构建或响应学习的条件。

| 问题 | 温颖及合作者的工作 | 核心思想与边界 |
| --- | --- | --- |
| 无法继续与对手交互 | [Offline Fictitious Self-Play](https://arxiv.org/abs/2403.00841) | 重加权固定数据来近似不同对手下的经验，再用离线 RL 学响应；重要性采样不能创造数据未覆盖的动作。 |
| 对手能整队协调偏离 | [Leveraging Team Correlation](https://arxiv.org/abs/2403.00255) | 区分个人偏离与团队相关偏离，以受限团队解和顺序相关机制组织搜索；安全性只相对于声明的偏离空间。 |
| 队友角色不同 | [Heterogeneous-PSRO](https://arxiv.org/abs/2410.01575) | 扩展异质团队策略的表达与顺序响应；表示限制、理想改进界与有限训练误差需要分别分析。 |
| 既非共同奖励也非零和 | [Stochastic Games with Potentials](https://proceedings.mlr.press/v139/mguni21a.html) | 利用特殊势结构关联个人改进和共同势函数；不是任意一般和游戏都能化为单目标优化。 |
| 采样、训练、评估彼此嵌套 | [MALib](https://jmlr.org/papers/v24/22-0169.html) | 以任务调度和 Actor–Evaluator–Learner 支持种群课程；并行吞吐是系统指标，不是学习质量保证。 |

$$
\sigma_{\rm team}\in\Delta\!\left(\prod_i\Pi_i\right)\quad\text{与}\quad(\sigma_i)_i\in\prod_i\Delta(\Pi_i)
$$

左侧允许对联合策略进行相关随机选择；右侧仅独立随机化。两个集合对应不同的协作与对手能力假设。事前协商不等于允许执行时共享私有观察。

若对手被允许整队协调改变策略，只测试单个成员偏离不够。反之，不能把己方有特权通信、对方只能独立随机化时的优势称为公平的均衡比较。异质性也不等于必须完全独立参数：共享网络若有充分角色条件，可能表达不同角色；应检验具体函数类。

Off-FSP 属于数据受限的竞争学习，不能因为使用历史经验就称为在线 CRL。势博弈是结构性分支，不能将其收敛结论外推给所有开放合作。上述研究分别改变数据协议、策略可行集合、评价或计算组织，因而需要不同实验。

<a id="lesson-branches"></a>

### 7.1 · 与 CRL 相连：评价整个学习过程

开放式多智能体学习研究怎样由交互生成新的学习目标，并沿明确准则改善能力。把它放入 CRL 的生命期设定后，还要把形成能力的过程本身纳入评价。最终档案更强与同一智能体一生获得更高收益，是相关但不同的目标。

| 现有机制 | 新增困难 | 所需对照 |
| --- | --- | --- |
| 策略池不断增加 | 参数、检索和评估开销不能无限增长 | 固定档案容量，比保留、删除、蒸馏后的旧弱点。 |
| 每回合抽取伙伴 | 伙伴会返回、漂移或在途中离开 | 控制变化可观测性，测恢复时间与历史利用。 |
| 冻结策略零样本适应 | 状态、记忆和参数是不同更新载体 | 分别冻结各载体，不把活动变化当作无适应。 |
| 只测最终 checkpoint | 探索、更新和错误在生命中有成本 | 全寿命回报、适应损失、失败与恢复代价。 |

合作结构化探索决定一次交互发现什么；种群课程决定下一轮遇到什么；持续学习还要决定保留哪些知识、何时更新和付出多少代价。三层相连，但一层的成功不能代替其他层的证据。

可以先固定物理规则，只让伙伴约定变化；再固定伙伴，改变动力学。最后组合二者，研究有限记忆下的状态、预测和控制学习，避免把所有困难笼统归为非平稳。

<a id="lesson-code"></a>

## 8 · 算例与原始工程：从标签到种群评价

先运行 [selfplay_search_labels.py](/crl-code/tutorials/selfplay_search_labels.py)：python3 selfplay_search_labels.py 打印 32 次确定性搜索的逐步轨迹、计数、π、价值标签和交叉熵梯度；加 test 运行五项断言。只依赖标准库，没有网络训练、随机自对弈或棋类工程。把预算改为 1，可以立刻检查“搜索必然改善”的反例。

matrix_value、minimax_2x2 与 zero_sum_gap 检验矩阵评价；没有训练 PR2、GR2 或 NFSP 网络。

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

| 自对弈实现入口 | 阅读位置 | 实现边界 |
| --- | --- | --- |
| OpenSpiel NFSP / CFR | python/pytorch/nfsp.py；python/algorithms/cfr.py | 平台实现，不标作原论文当年实验快照 |
| OpenSpiel AlphaZero | alpha_zero.py 的 _play_game 与 collect_trajectories | 对照根访问计数、温度、对局回报和训练标签；平台实现，不是作者实验发布包 |
| MuZero 作者伪代码 | play_game、store_search_statistics、make_target、update_weights | 对照真实交互、搜索标签、展开目标和梯度；环境与网络接口未实现 |

实现细节必须读到损失处。所核对的 OpenSpiel NFSP 版本在响应模式的 reservoir 中存动作概率向量，再做交叉熵；这与论文 Algorithm 1 写出的采样动作记录不是逐字段相同。OpenSpiel AlphaZero 的当前实现还采用固定玩家 1 的训练价值视角，并在搜索 evaluator 中转换双方收益；本章手算采用当前行动玩家视角。两种约定都需全链条一致，不能只复制一个负号。

精确小博弈：fictitious play 的平均混合与最新响应、完整 gap、伙伴分布变化，以及精确最佳响应扩池后的反例。不是神经 PSRO 复现。

```python
def saddle_gap(matrix, row, col):
    """Full-game zero-sum gap, not win rate against one training opponent."""
    distribution(row)
    distribution(col)
    upper = max(sum(a*b for a, b in zip(r, col)) for r in matrix)
    lower = min(sum(row[i]*matrix[i][j] for i in range(len(row)))
                for j in range(len(col)))
    return upper-lower


def fictitious_play(rounds=300):
    """Symmetric RPS: BR to opponent's empirical average, not latest action.

    Start with one rock observation. Both roles have the same population
    by game symmetry; a best response is computed by full enumeration.
    Tie-breaking selects the lowest action index. Report BOTH the latest
    pure policy and the empirical mixture, which are different objects.
    This is exact matrix-game fictitious play, not NFSP or neural PSRO.
    """
    counts = [1, 0, 0]
    rows = []
    for k in range(rounds+1):
        mixture = [c/sum(counts) for c in counts]
        scores = [sum(a*b for a, b in zip(r, mixture)) for r in RPS]
        action = max(range(3), key=lambda i: scores[i])
        pure = [float(i == action) for i in range(3)]
        rows.append({"round": k, "mean_gap": saddle_gap(RPS, mixture, mixture),
                     "latest_gap": saddle_gap(RPS, pure, pure),
                     "mixture": mixture, "best_response": action})
        counts[action] += 1
    return rows


def partner_shift():
    """Coordination reward I[a=b]. Fixed and changing objectives differ."""
    matrix = [[1., 0.], [0., 1.]]
    policies = {"train_specialist": [1., 0.], "balanced": [.5, .5]}
    partners = {"training": [.9, .1], "held_out": [0., 1.]}
    return {name: {group: value(matrix, pi, mu) for group, mu in partners.items()}
            for name, pi in policies.items()}


def expanding_pool_counterexample():
    """Exact double-oracle expansion can increase full-game exploitability.

    Both pools initially contain action 0 only. Action 1 is each player's
    exact best response. The expanded 2x2 restricted equilibrium is (1,1).
    But the absent action 2 exploits it with magnitude 10. Pool inclusion
    alone does not make the CURRENT restricted equilibrium's gap monotonic.
    """
    matrix = [[0., -1., 1.], [1., 0., -10.], [-1., 10., 0.]]
    return {"matrix": matrix, "initial_gap": saddle_gap(matrix, [1., 0., 0.], [1., 0., 0.]),
            "expanded_gap": saddle_gap(matrix, [0., 1., 0.], [0., 1., 0.])}
```

标准库运行。枚举期望和精确最佳响应，不含采样置信区间；图用于检验机制而非论文性能。

```bash
python3 marl_objectives_lab.py test
python3 marl_objectives_lab.py demo --out results/marl-objectives
```

| 原始入口 | 重点代码接口 | 边界 |
| --- | --- | --- |
| [OpenSpiel PSRO](https://github.com/google-deepmind/open_spiel/tree/master/open_spiel/python/algorithms/psro_v2) | 元游戏、求解器、response oracle、evaluation | 通用 PSRO 不是 EPSRO 作者实现。 |
| [NAC](https://github.com/waterhorse1/NAC) | 课程网络、展开内循环、外层评价 | 元梯度不提供最后策略的普遍单调保证。 |
| [NXDO](https://github.com/indylab/nxdo) | 信息状态 meta-action 与受限求解 | 神经版本不自动满足表格定理。 |
| [COLE-Platform](https://github.com/liyang619/COLE-Platform) | cole_training 与 baseline_training 分支 | main 分支不能直接当作 COLE-SV 训练器。 |
| [ZSC-Eval](https://github.com/sjtu-marl/ZSC-Eval) | 伙伴生成、筛选、参考响应、分伙伴评价 | 有限伙伴池不代表所有部署人群。 |
| [MALib](https://github.com/sjtu-marl/malib) | 种群与经验管理基础设施 | 清单中未勾选的算法不是实现证据。 |

EPSRO、BD/RD、Fusion-PSRO、Conflux-PSRO 与 HOLA 的原文见参考。这里不以背景仓库替代尚未确认的完整作者实现，也不把小矩阵核的正确性当作大规模训练性能证据。

<a id="lesson-check"></a>

### 8.1 · 练习：同一结果为什么会有不同解读

- 问：自对弈必须共享网络吗？答：不需要。各角色的奖励和信息可以不同。
- 练习：两个策略各自连续两步总选左或总选右。比较每局抽一次策略和每步混合动作可产生的轨迹。

- 练习：只做一次两层树模拟，比较先验与计数策略的保底收益，说明有限搜索为什么没有逐状态改善保证。
- 练习：实际对局 L→b 后，写出根状态的 π 和 z；解释为什么策略梯度仍可提高未执行的 R。
- 练习：若把对手节点也按玩家 1 收益最大化，指出哪条分支和哪项任务假设被改变。
- 问：AlphaZero 的三棋结果证明持续学习与不遗忘吗？答：没有；原文分别训练三个实例。
- 问：MuZero 的搜索更深就能修复错误模型吗？答：不能；需真实反馈提供纠错依据，并检验未覆盖分支。

- 问：扩池后仍提交旧混合，实际表现一定提高吗？答：不一定。最佳安全价值不下降，实际输出可以完全不变。
- 问：新 payoff 向量离旧凸包很远，一定有用吗？答：不一定，它可能在所有参考对手上更差。
- 问：COLE 的 Shapley 思想和 COMA baseline 相同吗？答：不同。前者评价策略课程，后者构造动作梯度 baseline。
- 问：部署权重固定、recurrent state 改变，是否完全没有适应？答：仍可能适应，只是载体不同。
- 实验：固定协调矩阵，改变训练伙伴分布，再用同一冻结测试分布重算回报，检查课程分数上升是否代表泛化改善。
- 实验：复算加权剪刀石头布的 Gap 2→20；分别记录池内 Gap、完整 Gap、最新策略收益与种群安全价值，解释排序差异。



<a id="chapter-code"></a>

## 下载与运行

精确有限游戏的评价与学习目标诊断；不是神经 PSRO、COLE 或 HOLA 的论文性能复现。

[下载 marl_objectives_lab.py](https://yingwen.io/zh/continual-rl/download/marl_objectives_lab.py)

```sh
python3 marl_objectives_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Heinrich、Lanctot、Silver · Fictitious Self-Play（ICML 2015）](https://proceedings.mlr.press/v37/heinrich15.html)：阅读实现等价的行为平均与 XFP/FSP，注意到达概率权重。

- [Heinrich、Silver · Neural Fictitious Self-Play（2016）](https://arxiv.org/abs/1603.01121)：Algorithm 1 的按局模式抽样、两种记忆与平均策略评价。

- [Silver 等 · AlphaGo（Nature 2016）](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)：Figure 1、强化学习策略与价值训练、MCTS；监督先验与 rollout 不可追述成 Zero 的单网络闭环。

- [Silver 等 · AlphaGo Zero（Nature 2017）](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf)：第 1 节、公式 1：网络、搜索计数、当前玩家终局标签与联合损失。

- [Silver 等 · AlphaZero（Science 2018，作者开放稿）](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf)：正文公式 1、与 Zero 的区别及三个分别训练实例；区分 2017 预印本与最终评估。

- [Schrittwieser 等 · MuZero（Nature 2020）](https://arxiv.org/abs/1911.08265)：第 3 节及 Appendix A：表示、动力学、预测；根合法动作与内部隐模型的界限。

- [MuZero 作者公开伪代码](https://arxiv.org/src/1911.08265v1/anc/pseudocode.py)：记录搜索分布、构建展开目标与模型训练；带未实现接口，不是完整软件包。

- [OpenSpiel · AlphaZero 数据生成与学习](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/alpha_zero/alpha_zero.py)：_play_game 存访问计数分布；collect_trajectories 接终局回报。平台版本可能演化。

- [OpenSpiel · AlphaZero 价值视角转换](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/alpha_zero/evaluator.py)：evaluate 将固定玩家价值转成双方收益；与训练标签约定一起阅读。

- [Zinkevich 等 · Regret Minimization in Games with Incomplete Information（2007）](https://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf)：反事实价值、局部到整体遗憾界与完美回忆条件。

- [OpenSpiel · NFSP](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/pytorch/nfsp.py)：公开平台实现；阅读按局策略抽样、reservoir 和监督损失。

- [OpenSpiel · CFR](https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/cfr.py)：对照 counterfactual reach、regret matching 和 average_policy。

- [David Balduzzi · 作者主页与公开讲义入口](https://sites.google.com/site/dbalduzzi/)：主页列出 Open-ended learning in games（2019）与 Designing learning dynamics（NeurIPS tutorial，2020）。讲义用于理解问题脉络；下面的 ICML 2019 论文提供正式定义、算法和证明。本页原创图不转载整套讲义。

- [Balduzzi et al. · Open-ended Learning in Symmetric Zero-sum Games](https://arxiv.org/abs/1901.08106)：ICML 2019。按 §2 的对手→目标、§3 的 FGS/EGS 与相对种群值、§4 的 Nash/rectified Nash、Appendix E 的包含与严格扩张证明阅读。弱单调沿集合包含链成立，不给任意种群全序；§5 的预算未计交叉评估。

- [Albrecht、Christianos、Schäfer · Multi-Agent Reinforcement Learning](https://www.marl-book.com/)：先读博弈与解概念，再读深度算法和实现；官网提供教材、讲义与代码。

- [Lanctot et al. · A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://arxiv.org/abs/1711.00832)：PSRO 元博弈、元策略与学习响应接口；框架和具体解概念分开。

- [Zhou et al. · Efficient Policy Space Response Oracles](https://arxiv.org/abs/2202.00633)：URR、response 与元策略联合学习、warm start 和并行；保留均衡与遗憾条件。

- [Perez-Nieves et al. · Modelling Behavioural Diversity for Learning in Open-Ended Games](https://proceedings.mlr.press/v139/perez-nieves21a.html)：DPP 期望集合大小和 payoff 几何，与动作层 Q-DPP 不同。

- [Liu et al. · Towards Unifying Behavioral and Response Diversity](https://arxiv.org/abs/2106.04958)：占据分布、响应向量、gamescape 与 population effectivity。

- [Feng et al. · Neural Auto-Curricula](https://arxiv.org/abs/2106.02745)：依据后续可利用度训练课程生成器，连接种群学习与元学习。

- [Zhang et al. · Learning to Design Games: Strategic Environments in Reinforcement Learning](https://www.ijcai.org/Proceedings/2018/0426.pdf)：IJCAI 2018，§3.1–3.3、式 2、Definition 1 与图 1–2：对抗环境目标、dual MDP、连续转移梯度与离散生成。难度或轨迹等价不是持续学习与全局优化保证；实现入口须另行确认。

- [Lian et al. · Fusion-PSRO](https://doi.org/10.3233/FAIA251106)：Nash 加权参数融合用于 response 初始化，不等于概率混合。

- [Huang et al. · Conflux-PSRO](https://arxiv.org/abs/2410.22776)：状态级路由复用子策略，再蒸馏；机制说明不等于无条件性能排序。

- [McAleer et al. · XDO: A Double Oracle Algorithm for Extensive-Form Games](https://arxiv.org/abs/2103.06426)：信息状态级组合及 NXDO；相邻团队研究，表格界与神经近似分开。

- [Li et al. · Cooperative Open-ended Learning Framework for Zero-shot Coordination](https://proceedings.mlr.press/v202/li23au.html)：ICML 2023 原始 COLE：图评价、兼容性课程与局部偏好目标。

- [Li et al. · AT-Drone: Benchmarking Adaptive Teaming in Multi-Drone Pursuit](https://proceedings.mlr.press/v305/li25a.html)：CoRL 2025 / PMLR 305。方法详见 arXiv v2 §3.1–3.3：多学习者与陌生伙伴、三个具体测试池、SUC/COL/AST/REW。项目页未提供公开训练仓库；不外推有限物理场地结果。

- [Li et al. · Tackling Cooperative Incompatibility for Zero-Shot Human-AI Coordination](https://doi.org/10.1613/jair.1.15884)：JAIR 2024 扩展，含 COLE-SV、COLE-R 与人机研究。

- [Li et al. · HOLA-Drone](https://arxiv.org/abs/2409.08767)：超图合作关系、多伙伴课程与实体无人机零样本配合，部署策略固定。

- [Li et al. · Multi-Robot Open Adaptive Teaming Across Unseen Environments, Partners, and Scales](https://arxiv.org/abs/2607.04972)：后续环境、伙伴与规模变化研究；无微调不等于部署阶段参数学习。

- [Wang et al. · ZSC-Eval](https://arxiv.org/abs/2310.05208)：测试伙伴生成、BR-Div 与 BR-Prox，评价未见伙伴泛化。

- [Yang et al. · Multi-Agent Determinantal Q-Learning](https://proceedings.mlr.press/v119/yang20i.html)：联合行动探索结构，连接但不代替种群课程。

- [Chen et al. · Offline Fictitious Self-Play for Competitive Games](https://arxiv.org/abs/2403.00841)：固定数据上的对手重加权和离线响应；数据支持缺口不能由权重消除。

- [Liu et al. · Leveraging Team Correlation for Approximating Equilibrium in Two-Team Zero-Sum Games](https://arxiv.org/abs/2403.00255)：团队相关、偏离空间与受限团队解，连接合作内层与竞争外层。

- [Liu et al. · Computing Ex Ante Equilibrium in Heterogeneous Zero-Sum Team Games](https://arxiv.org/abs/2410.01575)：H-PSRO 的异质团队表达与顺序更新；区分局部训练界和总体均衡近似。

- [Mguni et al. · Learning in Nonzero-Sum Stochastic Games with Potentials](https://proceedings.mlr.press/v139/mguni21a.html)：特殊势结构下的一般和学习，不覆盖任意利益关系。

- [Zhou et al. · MALib](https://jmlr.org/papers/v24/22-0169.html)：JMLR 2023 种群训练系统，组织采样、评价与训练的嵌套工作负载。


<a id="marl-mechanism-experiments"></a>

## 小型实验：把更新规则与评价对象分开

全部结果来自完全列举的有限博弈，没有采样误差或置信区间。它们检验具体机制，不是大型神经算法复现。

### 末次策略与历史混合

标准石头剪刀布；从一次石头观测开始，双方精确响应历史平均策略，平局选最小索引。300轮后，末次纯策略的完整博弈gap仍为2，历史混合为0.0532。总体下降不意味着逐轮单调。

![虚拟对弈](https://yingwen.io/crl-code/marl-objectives/population.svg)

### 扩张种群的反例

收益矩阵((0,−1,1),(1,0,−10),(−1,10,0))。双方初始种群只有动作0。加入精确最佳响应动作1后，受限均衡变成双方都选1，完整博弈gap由2增至20。有限博弈DO的终止结论，不保证每轮受限均衡的full-game gap下降。

### 训练伙伴与未见伙伴

同动作得1，异动作得0。训练伙伴选0的概率为0.9。始终选0的训练回报为0.9，但面对始终选1的测试伙伴回报为0；均匀策略在两个分布下均为0.5。应固定评价分布，单独报告分布外泛化与适应成本。

### 运行与复核

```bash
python3 marl_objectives_lab.py test
python3 marl_objectives_lab.py demo --out results/marl-objectives
```

[源码](https://yingwen.io/zh/continual-rl/download/marl_objectives_lab.py) · [全部数值与源码哈希](https://yingwen.io/crl-code/marl-objectives/results.json)

仅依赖Python标准库，含15项机制检查。图不用于排序大型算法。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-marl-populations#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-marl-populations#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-deep-marl-populations)
<a id="study-connections"></a>

## 与教材主线的衔接

本章是可按问题选择的并列研究分支，不要求先读完其他深度研究分支。

### 一次策略更新，为什么能改善未来？

精确策略改善使用旧策略的真实价值；策略梯度使用与目标匹配的访问分布。值函数近似或梯度估计误差会破坏这些推理的前提。

函数逼近与深度方法：PPO 的动作概率比不等于新策略的状态访问比。连续动作 actor 还会追逐 critic 的误差。限制局部更新尺度与证明实际回报单调提高是不同要求。

持续学习中的研究问题：动作不仅改变世界，也改变未来数据与学习。比较冻结策略不等于比较持续更新的智能体；什么时候应付出当前回报去获得长期有用的经验？

[精确策略改善](https://yingwen.io/zh/continual-rl/foundations/tabular/dynamic-programming/) → [策略梯度定理](https://yingwen.io/zh/continual-rl/foundations/approximation/policy-gradient/) → [TRPO 与 PPO 的近似](https://yingwen.io/zh/continual-rl/foundations/deep/trust-region/) → [持续控制的比较器](https://yingwen.io/zh/continual-rl/algorithms/control/)


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)
- [目标条件化与子任务构造](https://yingwen.io/zh/continual-rl/construction/goals/)
- [元学习与学习规则的适应](https://yingwen.io/zh/continual-rl/algorithms/meta/)
- [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)
- [持续学习的智能体架构](https://yingwen.io/zh/continual-rl/construction/architectures/)

对应原始材料：Fictitious Play、FSP、NFSP 与 CFR；AlphaGo、AlphaGo Zero、AlphaZero 与 MuZero；MARL book 第 3–6 章：博弈、解概念与博弈学习；MARL book 第 9–11 章：深度算法、实现与环境；PSRO、EPSRO、COLE 与 HOLA 原文。本文为原创讲解，原书、论文与上游代码保留各自许可。
