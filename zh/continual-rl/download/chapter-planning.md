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
