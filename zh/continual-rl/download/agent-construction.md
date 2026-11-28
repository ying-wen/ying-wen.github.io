[统一学习地图](https://yingwen.io/zh/continual-rl/download/curriculum.md) · 本文件是知识构建的补充导读。正文阅读顺序见教材目录。

# 从价值学习到持续智能体构建

这是一条与遗忘、可塑性和实时更新并行的主线：预测知识 → 状态 → 子任务 → options → 后果模型 → 规划。经验与控制效用又反馈到知识的生成、检验和淘汰。箭头表示学习对象之间的接口，不是所有论文的继承关系。

## 读法

只有基础 RL：先读 01 的手算，再读 04、05、06，最后补 02、03、07。有 RL 研究背景：先看 07 的接口，再沿自己的缺口进入原始论文。

## 完整系统比较

### Dyna

学习对象：价值 + 模型；模拟更新改进控制

给定条件：状态、动作与主奖励通常给定

接口：模型支持 learning-time planning

边界：不是自动生成状态、目标与技能的完整系统

### Horde

学习对象：共享经验流上的多个预测器

给定条件：预测问题与表示通常由设计者给定

接口：同一 transition 回答多个问题

边界：多预测器不等于自动选择有用问题

### UNREAL

学习对象：主任务 actor–critic + 辅助学习

给定条件：辅助任务类型由设计者规定

接口：共享表示与辅助控制/预测

边界：辅助学习不自动构成长期自主知识发现

### Dreamer / Director

学习对象：潜在模型、行为；Director 加入层次目标

给定条件：模型结构、训练协议与高层接口有设计约束

接口：想象学习；manager/worker 目标传递

边界：有运行实现；不等于全部 OaK 接口或 strict streaming

### ROD / skills-to-symbols

学习对象：分别强调表示—技能反馈与技能支持的符号抽象

给定条件：候选结构、输入信息与实现设置各有假设

接口：抽象如何影响下一轮经验或高层规划

边界：两条相关路线，不是同一个架构的前后版本

### STOMP

学习对象：SubTask → Option → Model → Planning

给定条件：给定特征；子任务与停止奖励的设计

接口：将学到的技能后果放进规划

边界：组件与特定实验不等于自主状态/目标发现全部完成

### Alberta Plan / OaK

学习对象：持续表示、控制、抽象知识与规划的研究路线

给定条件：需要把各个假设逐步变成可学习接口

接口：期待形成知识构建与控制效用的闭环

边界：路线图、公开机制与整体系统结果要区分

## 配套作者代码与公式检查

[GVF、Options、Average Reward：19 个代码入口](https://yingwen.io/zh/continual-rl/construction/implementations/)：按问题、源码位置、运行条件与思考题阅读。包含八组公式对照；[独立公式测试](https://yingwen.io/zh/continual-rl/download/formula_checks.py)只需 Python 标准库。

## 01 · 预测知识：从一个价值函数到一组可检验的问题

除了“能得多少奖励”，智能体还应该预测什么？

先修：TD(0)、条件期望；off-policy 部分可第二遍再读。

输入：经验流、一个行为策略，以及指定的预测问题。

输出：不同策略、时间尺度与信号下的预测；供控制、状态构造或模型使用。

你已经会估计 Vπ：按策略 π 行动，未来能得到多少折扣奖励。第一步不是换一种网络，而是把“奖励”换成任何可从交互中观察的累积信号，把固定折扣换成描述时间尺度或终止的函数。于是“按这条路线走，会耗多少电？”也成为价值预测问题。

GVF（General Value Function，通用价值函数）描述一个问题；TD、GTD 或其他更新规则负责学习答案。Horde 将许多这样的预测器放在同一经验流上学习。拥有很多答案以后，还要单独解决：哪些问题值得提出、答案是否有用，以及问题应在何时淘汰。

### 算法分支

#### 奖励价值 → 广义预测

改变被累积的信号、目标策略和延续函数；不是只把输出头变多。

代表工作：TD → GVF；Horde（Sutton、Modayil、Delp、Degris、Pilarski、White、Precup，2011）

继续问：问题由谁给定？新的 cumulant、策略和时间尺度怎样从经验中产生？

#### 一条行为轨迹 → 多条反事实预测

行为 μ 收集数据，目标 π 定义要预测的行为；不同问题可共享一次 transition。

代表工作：importance sampling → GTD / TDC → emphatic TD

继续问：数据覆盖、方差与计算预算：共享经验不等于免费获得未尝试动作的后果。

#### 答案 → 控制可用的知识

把预测送入状态表示、决策或后果模型；检验下游用途，而非只看预测数量。

代表工作：predictive knowledge → TD networks / PSR → option models

继续问：预测误差很小但控制没有改善，是问题无用、表示不足，还是控制器没用上？

### 把熟悉的 Bellman 方程换一个预测对象

$$
G_t^q=c_{t+1}^q+\gamma_{t+1}^qG_{t+1}^q,\qquad v_q(s)=\mathbb E_{\pi_q}[G_t^q\mid S_t=s]
$$

q 是问题编号；c 是 cumulant（被累积的可观察信号），不必是主任务奖励；γ∈[0,1] 是延续系数；πq 是目标策略。这里先用 Markov 状态，并假定回报存在。

1. 展开递归得到 $c_{t+1}+γ_{t+1}c_{t+2}+γ_{t+1}γ_{t+2}c_{t+3}+\cdots$。普通折扣价值是 c=R、γ固定的特例。

2. 令“到充电站时 γ=0”，即可让问题在那里停止；环境本身不需要 reset。预测的终止与环境终止不是一回事。

3. 取 c=每步耗电量，预测累计耗电；取 c=到达事件指示并在到达时终止，在适当的吸收条件下预测到达概率。不能只改名字而不定义事件。

### 同一条经验，怎样回答另一条策略的问题？

$$
\delta_t^q=c_{t+1}^q+\gamma_{t+1}^q\hat v_q(S_{t+1})-\hat v_q(S_t),\quad \rho_t^q=\frac{\pi_q(A_t\mid S_t)}{\mu(A_t\mid S_t)}
$$

δ 是预测误差；μ 是真正执行动作的策略。目标策略可能选择的动作，行为策略必须有正概率选择。

1. 表格、一步 importance-sampled TD 的示意更新为 $V_q(S_t)←V_q(S_t)+αρ_qδ_q$。我们在教学代码中使用充分覆盖的数据与有限状态。

2. 不能把这个示意公式直接当作深度 off-policy 稳定算法。函数逼近、自举、off-policy 组合有发散风险；GTD 与 ETD 针对相应假设给出不同处理。

3. GVF 不等于 UVFA：前者规定“问什么预测问题”；后者强调一个函数如何在多个状态与目标之间泛化。两者可以结合。

### 手算：仓库机器人：同一个路口，三个问题

机器人以 μ 探索。问题 A：按右行策略，折扣后的充电到达信号是多少？问题 B：同一策略预计耗电多少？问题 C：按另一条路线，碰撞事件的累积量是多少？

一次向右的 transition 可以更新多个问题，但不同问题有不同 c、γ 和 ρ。没有碰撞，不意味着某个从未执行的转弯一定安全。

教学脚本把 A、B 化成两步链。γ=0.9 时，起点到达信号的真值为 0.9，累计耗电的真值为 1.9；用这两个数排查下标。

若 γ=0.9，为什么“两步后到达”的累计事件信号是 0.9，而不是 0.81？

参考答案：我们定义到达事件为第二次 transition 的 cumulant：$G_0=c_1+γc_2=0+0.9×1$。第 05 章的终点价值位于执行两步之后，权重才是 $γ^2$。两者预测对象不同。

### 按问题读原始材料

#### [Horde：共享经验流上的预测知识](https://sites.ualberta.ca/~amw8/horde.pdf)

Sutton 等 · 2011 · 核心论文

先给每个预测器写出 c、γ、π，再看数据共享与学习更新。

思考：Horde 学会回答的问题中，哪些是设计者事先指定的？

#### [Emphatic TD：为什么 off-policy 不能只写一个 TD error](https://www.jmlr.org/beta/papers/v17/14-488.html)

Sutton、Mahmood、Martha White · 2016 · 进阶理论

只在需要函数逼近稳定性时读；重点是更新权重、interest 与理论条件。

思考：若覆盖不足或环境持续变化，哪部分保证不再可直接使用？

#### [预测知识与 options 课程笔记](https://www.cs.mcgill.ca/~dprecup/courses/Winter2017/RL/lectures.html)

Doina Precup · COMP-767 · 课程

沿 Predictive Knowledge 及 temporal abstraction 阅读，把 GVF 接到模型和状态。

思考：哪些预测可以被环境经验直接验证？

### 最小研究练习

运行 knowledge_lab.py gvf。检查两个问题在三种随机种子下的表格 TD 估计与解析真值。

对照：保持相同轨迹，比较正确重要性比、漏掉比率以及改变行为策略；一次只改一个因素。

指标：分问题 RMSE、访问次数、重要性比的最大值；另加独立控制实验才可主张控制收益。

容易误判：某动作从未采样时，不能通过增加预测头修补覆盖；低 RMSE 也不是自主知识发现的证据。

## 02 · 构造 agent state：如何压缩历史以支持预测与控制？

环境没有直接给出完整状态时，智能体如何构造自己的学习输入？

先修：理解“同一张图像可能对应不同历史”；知道函数逼近即可。

输入：历史中的动作、观测、奖励，以及已有的记忆与预测。

输出：有限维 agent state；它要支持后续预测和决策，但未必是严格的 Markov 状态。

仓库两条走廊外观相同，只有进入前的一盏灯告诉机器人哪边正在检修。只把当前图像送给 Q 网络，可能把两个不同的问题混成一个。更大的网络不一定补得回已经丢掉的信息。

状态构造有两类相邻路线：用 RNN 等压缩历史；用“采取某些动作后会看到什么”的预测来表示历史。预测表示赋予分量可检验的含义，但不是任意挑几个预测就能保证充分性。长期运行还多一层困难：表示自身不断改变，所有依赖它的价值、目标和模型也在跟着变。

### 算法分支

#### 历史 → 递归压缩

学习一个逐步更新的内部记忆；序列梯度、截断长度与实时成本成为问题的一部分。

代表工作：recurrent RL / R2D2 → RTU / streaming RTRL

继续问：能处理多长信用？参数更新与 hidden-state 演化怎样分开评估？

#### 历史 → 行动条件预测

以未来动作—观测测试的预测刻画状态，追求能递归更新的表示。

代表工作：Littman、Sutton、Singh：PSR；Sutton、Tanner：TD networks

继续问：选择哪些测试才能支持控制？在函数逼近与部分覆盖下会丢失什么信息？

#### 固定特征 → 持续生成与筛选

候选特征被经验训练，再按学习或决策用途保留、替换。

代表工作：generate-and-test / Continual Backprop；Alberta Plan 的持续表示问题

继续问：替换有用特征会让旧模型失效；局部活跃度不是长期知识效用的同义词。

### 先区分真实状态、历史与内部状态

$$
H_t=(O_0,A_0,R_1,\ldots,O_t),\qquad X_t=f_\theta(X_{t-1},A_{t-1},O_t,R_t)
$$

H 是可见历史，O 是观测，X 是 agent state；环境的真实状态 S 通常不能直接访问。θ 可在整个生命期更新。

1. X 是有限资源下的历史摘要。Q(X,a) 的输入可持续变化，不应把 X 自动视为环境给定的 Markov 状态。

2. 如果两种历史被压成同一 X，但最优动作不同，那么后面的控制器再准确也无法同时选对。

3. 如果 θ 改变，过去学到的模型“输入坐标”也可能改变。固定表示与联合更新表示的结果必须分开。

### 预测表示提供的是一种状态构造原则

$$
X_t^i=\Pr(o^i_{1:k}\mid H_t,\operatorname{do}(a^i_{1:k}))
$$

一个测试 i 指定未来动作序列与观测序列；do 表示执行这些动作，而不是把策略选择造成的相关性当成干预。该式是 PSR 直觉式，不是通用估计器。

1. 相同的当前画面，因历史线索不同，可以有不同的“走到门口是否畅通”预测。

2. 一组足够的核心测试在相应系统条件下能构成预测状态；实际挑选的有限 GVF 集合通常只能近似。

3. 评测应同时检查预测误差与动作选择；预测照明亮度很准确，可能对是否绕路毫无帮助。

### 手算：相同走廊，不同的门

设起点灯色 L∈{红,蓝}；进入走廊后灯不可见。红灯意味着左门通、蓝灯意味着右门通。

只用当前画面，平衡数据下任一固定选门规则最多成功一半；保存一位线索的 oracle 可以全对。

先用 oracle 检验环境与下游 Q 更新；再用可学习的递归状态或预测状态替代 oracle。后一步才在测试状态学习。

一个 RNN 训练成功，是否证明它持续构建了新的预测知识？

参考答案：没有。它说明在该训练与测试协议下，递归状态支持了行为。还要检查是否在线更新、是否依靠预训练、面对新依赖能否形成新能力，以及预测分量是否真的被构造。

### 按问题读原始材料

#### [Predictive Representations of State](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf)

Littman、Sutton、Singh · 2001 · 核心论文

先画出“历史—测试—预测”的对应关系，再看表示条件。

思考：哪一组测试能区分两段外观相同但后果不同的历史？

#### [Temporal-Difference Networks](https://arxiv.org/abs/1504.05539)

Sutton、Tanner · 进阶论文

看预测之间怎样组成网络及其训练目标，别只把它理解为多个独立辅助头。

思考：如果某个预测依赖另一个尚不准确的预测，误差如何传播？

#### [Streaming RTRL 与部分可观测 RL](https://arxiv.org/abs/2605.24709)

Farr、Reddi、D’Eramo、Peters · 2026 · 近期方法

对照结构限制、每步信用计算和所用部分可观测任务。

思考：增加记忆容量时，计算与梯度精确性付出什么代价？

高效计算依赖所选结构，不是任意 RNN 的通用线性复杂度保证。

#### [Generate-and-test methods for Continual Learning](https://www.youtube.com/watch?v=NaJxvGV8MRg)

Richard Sutton · 讲座

带着“候选是什么、测试信号是什么、何时删除”三个问题观看。

思考：如何避免删除暂时无用但以后重要的特征？

### 最小研究练习

从已有 memory 教学实验出发，先复现无记忆与 oracle 的差距，再加入小型可学习递归表示。

对照：相同观测与训练预算；无记忆、oracle、冻结表示、持续更新表示；报告 reset 和序列截断。

指标：按线索和延迟分组的成功率、后续新规律学习、模型失效程度、状态与梯度内存。

容易误判：把额外可见信息带来的收益说成学习算法更好；或把 oracle 记忆当作已训练出的 RNN。

## 03 · 目标与子任务：不仅学习怎么做，还要决定学什么

“目标已给定”“选择一个目标”“构造一个新子任务”有什么不同？

先修：奖励、终止和策略的基本概念；线性价值函数的点积。

输入：已有状态与特征、当前主任务价值、候选结果及有限的学习预算。

输出：一个可优化、可检验的子任务定义；它尚不是执行该子任务的技能。

用户说“到充电站”，已经给出了目标。让网络以充电站坐标 g 为输入，是目标条件控制。让系统挑下一个要练的充电站，是课程选择。连“什么结果值得成为一个目标”也从经验中形成，才涉及目标或子任务构建。这三层不能混写。

子任务是要解的问题，option 是一种行为解；同一个子任务可以有不同的解。同样，“去门口”不能只按最短路径定义：如果路中有损伤，忽略主任务奖励的子任务可能让规划得到一种看似方便、实际很昂贵的技能。

### 算法分支

#### 给定目标空间 → 跨目标泛化

V(s,g) 或 Q(s,a,g) 用一个函数分享多个目标的经验。

代表工作：Schaul 等：UVFA；Andrychowicz 等：HER

继续问：HER 重标记已达到的目标，但目标的表示和成功判据从哪里来？

#### 给定候选集 → 自动课程

选择当前值得练习的目标，或生成适合学习难度的候选。

代表工作：Florensa 等：GoalGAN；intrinsically motivated goal exploration

继续问：学得快与长期有用可能冲突；不可约噪声不应被当作持续学习进步。

#### 已有特征 → 尊重主奖励的子任务

保留环境奖励，再为特定终止结果设置偏好，而非只奖励更快到达。

代表工作：Sutton、Machado 等：Reward-Respecting Subtasks / STOMP

继续问：候选特征、终止奖励强度与资源分配仍需设计；论文没有自动解决全部状态与目标发现。

#### 技能多样性 → 目标空间的候选来源

从可区分行为或可预测后果产生行为集合，再问能否服务下游任务。

代表工作：DIAYN / DADS；ROD；Director 的潜在目标

继续问：彼此不同不等于可控、有用或值得长期保存。

### 目标条件化不是自动目标发现

$$
Q(s,a,g)=\mathbb E\!\left[\sum_{k\ge0}\gamma^k r_g(S_{t+k},A_{t+k},S_{t+k+1})\mid s,a,g\right]
$$

g 是给定的目标描述；rg 是对应的奖励判据。此处略去后续策略的下标，策略同样可依赖 g。

1. UVFA 的核心是跨状态与目标共享近似；不能仅由 Q 有 g 输入，就断言目标空间会自我增长。

2. HER 将历史结果当作另一目标，重新计算相应奖励，主要服务 off-policy 学习。它依赖经验重用，不是 strict streaming 的直接方案。

3. 一个目标生成器还需处理有效性、可达性和价值；这些并不由 TD target 自动作答。

### 从“达到某结果”到 reward-respecting subtask

$$
z^i(s)=w^\top x(s)+(\bar w_i-w_i)x_i(s)
$$

主任务价值近似为 $w^Tx$；i 指定已有特征；把该特征的权重临时替换成偏乐观的 w̄i，得到终止价值 zi。

1. 子任务继续使用真实环境奖励 R，并在停止时加入 zi；这样“去门口”仍需权衡沿途代价。

2. 这里改变的是学习子任务的目标，不是永久篡改环境主奖励；主任务规划使用真实奖励后果。

3. 这是给定特征后的子任务构建机制。特征是否有意义、乐观幅度是否合适，仍需验证。

### 读 STOMP 时留意终止奖励的下标约定

$$
\delta=R_{t+1}+\beta(S_{t+1})z(S_{t+1})+\gamma[1-\beta(S_{t+1})]v(S_{t+1})-v(S_t)
$$

β 是到达下一状态后停止的概率。此处遵循 Reward-Respecting Subtasks 的子任务约定：第一步奖励与同一步停止奖励均不折扣。

1. 终止时 target=R+z；继续时 target=R+γv′。因此其回报为 $\sum_{j=1}^K γ^{j-1}R_j+γ^{K-1}z(S_K)$。

2. 不要把它直接替换为标准 option model 中的 γᴷV(Sᴷ)。后者表示执行 K 步以后接续主任务的价值，第 05 章会单独推导。

### 手算：去门口，为什么不一定越快越好？

这是自编有限回合例子，取 γ=1。捷径需 2 步但总损伤代价 −6；绕行需 4 步、总奖励 −1；两条路线到达同一门口。

若子任务只用每步 −1，捷径得 −2、绕行得 −4，选捷径。若保留真实奖励并加同一终止奖励 +5，则得 −1 与 +4，选绕行。

这说明目标定义能改变技能，不证明某一种奖励设计普遍最好。若奖励非负的循环可以无限刷分，γ=1 的回报还可能不存在；本例限定有限路径。

把终止奖励设得极大，是否就能同时保证到达和尊重沿途代价？

参考答案：不保证。在有折扣、可变终止状态或其他竞争结果时，巨大终止奖励可能压过沿途代价，重新偏向快速到达。需要扫描奖励尺度，并检查主任务规划效用。

### 按问题读原始材料

#### [Universal Value Function Approximators](https://proceedings.mlr.press/v37/schaul15.html)

Schaul、Horgan、Gregor、Silver · 2015 · 核心起点

先看 state 与 goal 两个泛化维度，确定哪些对象被固定。

思考：一个从未出现过的目标为什么能被表示，凭什么能泛化？

#### [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)

Andrychowicz 等 · 2017 · 方法

画出一次重标记前后的 transition 和奖励，确认目标达成判据。

思考：环境动力学依赖目标时，可以原样重用 transition 吗？

#### [Automatic Goal Generation for RL Agents](https://arxiv.org/abs/1705.06366)

Florensa、Held、Geng、Abbeel · GoalGAN · 自动课程

关注训练目标的难度选择，而不是只看生成网络。

思考：已学会目标的遗忘会怎样影响课程？

#### [Reward-Respecting Subtasks for Model-Based RL](https://arxiv.org/abs/2202.03466)

Sutton、Machado、Holland、D. Szepesvári、Timbers、Tanner、A. White · 进阶主线

依次读 subtask 定义、option 学习、model 和 planning，核对每个接口。

思考：论文的特征和候选子任务哪些已给定，哪些真正学习了？

#### [Deep Hierarchical Planning from Pixels · Director](https://danijar.com/project/director/)

Hafner、Lee、Fischer、Abbeel · 论文 / 视频 / 代码入口

看 latent goal 怎样被 manager 选择，再由 worker 实现；项目页有视频和消融。

思考：世界模型认为合理的目标是否一定能被 worker 达到？

### 最小研究练习

运行 knowledge_lab.py subtask，先重现捷径/绕行算例；再在两房间环境实现不同子任务奖励。

对照：固定候选门口、样本与 option 数；对比最短路径、reward-respecting、随机子目标；扫描终止奖励。

指标：真实奖励损失、到达率、技能耗时与后续规划收益；同时报告技能学习成本。

容易误判：用更容易的目标提高成功率，却声称形成了更有用的知识；或把事先给定的特征称为自动发现。

## 04 · 时间抽象：学习、发现与组合 options

怎样把多步行为变成新的决策单位，又怎样判断哪些技能值得学？

先修：Q-learning；了解策略梯度有助于读 Option-Critic。

输入：主任务或子任务目标、状态表示和经验。

输出：可启动、可执行、可终止的行为单元；还没有自动获得其后果模型。

原始动作是“向前一步”；option 可以是“穿过门口”。它不是保存一串不变的动作，而是内部仍根据状态选择动作的策略。高层减少了做选择的频率，低层仍在和环境逐步交互。

这里至少有四个问题：如何执行和更新一个已给定的 option；如何学习其内部策略与终止；如何发现一组有用 options；如何组合和淘汰已有 options。不同工作回答的层次不同，不能画成一条替代链。

### 算法分支

#### 原始动作 → 时间扩展动作

把启动集、内部策略和终止条件包装成 option，用半 MDP 处理随机持续时间。

代表工作：Sutton、Precup、Singh：options；SMDP / intra-option learning

继续问：终止、切换与中断是否允许？不同执行语义影响学习和评测。

#### 给定 option 结构 → 学策略与终止

以控制回报训练 option 内策略、终止函数和高层选择。

代表工作：Bacon、Harb、Precup：Option-Critic

继续问：技能坍缩、频繁终止、缺少可复用性；端到端控制收益不等于发现了好的规划抽象。

#### 表示与拓扑 → 探索技能

从状态空间结构产生内在方向或子目标，再学习实现它们的行为。

代表工作：Machado 等：eigenoptions / ROD；Jinnai、Park、Abel、Konidaris：cover time

继续问：表示受行为数据影响，技能又影响下次数据，构成反馈；拓扑方向不自动最适于任务规划。

#### 无监督目标 → 多样或可预测技能

优化技能可区分性，或学习不同技能造成的可预测状态变化。

代表工作：Eysenbach 等：DIAYN；Sharma 等：DADS

继续问：预训练阶段的多样性需要通过下游迁移验证；不是一运行就变成终身知识积累。

#### 未来特征 → 策略与行为的组合

SF 把未来特征与奖励权重分离，GPI 从策略集合构造改进控制；keyboard 探索组合接口。

代表工作：Barreto 等：SF / GPI / Option Keyboard；Chandrasekar、Machado：Laplacian Keyboard

继续问：奖励迁移的假设不能外推为任意动力学改变；GPI 本身不要求时间抽象。

### option 不是动作序列：三个接口缺一不可

$$
o=(I_o,\pi_o,\beta_o),\qquad Q(s,o)\leftarrow Q(s,o)+\alpha\!\left[\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}+\gamma^\tau\max_{o'}Q(S_{t+\tau},o')-Q(s,o)\right]
$$

I 是允许启动的状态集；π 是内部策略；β 是终止概率；τ≥1 为实际执行步数。这个 SMDP 更新在 option 终止后进行。

1. 把执行期间的奖励按原始时间逐步折扣，再把后续价值乘 $γ^τ$。不能因为高层只选了一次，就只折扣一次。

2. Intra-option 方法利用执行中的单步经验来更新满足条件的 option；并非必须等一段行为结束才可学习。

3. 这里的 max 假定标准调用后执行到终止的控制语义。允许中断时需要另行说明。

### SF / GPI 接在价值线，不自动等于时间抽象

$$
r_w=\phi^\top w,\quad \psi^\pi(s,a)=\mathbb E_\pi\!\left[\sum_{k\ge0}\gamma^k\phi_{t+k+1}\mid s,a\right],\quad Q_w^\pi=\psi^{\pi\top}w,\quad \pi_{\rm GPI}(s)\in\arg\max_a\max_j Q_w^{\pi_j}(s,a)
$$

φ 是 transition 特征，w 是当前奖励权重，ψ 是策略 π 下的未来特征累积；j 索引已有策略。

1. 奖励换了但动力学、特征与策略未变时，可重用 ψ，改变 w 来重估价值。

2. GPI 比较多个策略的动作价值，并不只是在 episode 开始选一条整段固定策略。

3. 若门被封住，ψ 对未来占用的预测可能过时；若用近似价值，策略改进结论要包含误差条件。

### 手算：“穿门”持续三步，应该如何更新？

取 γ=0.9，三步奖励依次为 −1、−1、0，终点最佳高层价值为 10。

正确 $target=-1-0.9+0+0.9^3×10=5.39$。只把后续价值折扣一次会得到 7.1，系统性高估耗时行为。

一个 option 能成功穿门，并不说明知道会耗几步、在哪里结束、会不会损伤。下一章才学习这些后果。

把技能数从 8 增加到 80，探索一定更快吗？

参考答案：不一定。冗余、耗时、难以终止的行为可能拖慢探索，也增加选择与训练成本。必须明确是在最小化 cover time、planning backups，还是提高主任务回报。

### 按问题读原始材料

#### [Options 原论文与公式导读](https://www.cs.mcgill.ca/~dprecup/courses/Winter2017/RL/lectures.html)

Sutton、Precup、Singh；Precup 课程讲义 · 核心起点

课程标出 1999 原文的 SMDP 更新、intra-option 方程与 model 位置，适合对照读。

思考：一步动作如何作为 option 的特例？

#### [The Option-Critic Architecture](https://arxiv.org/abs/1609.05140)

Bacon、Harb、Precup · 策略梯度分支

分开高层选择、内部策略梯度与终止梯度。

思考：训练学到的终止概率怎样影响技能时长与复用？

#### [A Laplacian Framework for Option Discovery](https://proceedings.mlr.press/v70/machado17a.html)

Machado、Bellemare、Bowling · 2017 · 表示分支

看状态拓扑如何变成内在奖励方向，再变成 options。

思考：拓扑变化以后，特征和技能谁先过时？

#### [Temporal Abstraction in RL with the Successor Representation](https://jmlr.org/papers/v24/21-1213.html)

Machado、Barreto、Precup、Bowling · JMLR 2023 · 系统阅读

围绕 representation-driven option discovery cycle 画一张反馈图，而非只记 eigenoptions 名称。

思考：新增技能改变经验分布后，旧表示应如何更新？

#### [Successor Features for Transfer in RL](https://arxiv.org/abs/1606.05312)

Barreto 等 · 迁移分支

推导一次 $Q=ψ^Tw$，再比较 GPI 与单纯挑选最佳旧策略。

思考：改变动力学时，哪个等式形式仍成立、哪组已学参数不再可信？

#### [DIAYN / Diversity Is All You Need](https://arxiv.org/abs/1802.06070)

Eysenbach、Gupta、Ibarz、Levine · 无监督技能

读技能辨别目标与下游评测，明确奖励来自哪里。

思考：如果技能只在无关视觉背景上不同，判别器分数说明了什么？

#### [DADS / Dynamics-Aware Unsupervised Discovery of Skills](https://arxiv.org/abs/1907.01657)

Sharma 等 · 技能后果

把技能辨别与可预测的状态变化对照，留意模型怎样用于控制。

思考：可预测的技能是否也覆盖了完成目标所需的行为？

#### [Marlos Machado 讲座与 slides](https://deeprlcourse.github.io/guests/marlos_machado/)

Deep RL Course · 讲座 / slides

先看图示建立表示—行为反馈直觉，再读 ROD 与原始方法。

思考：把一张讲座架构图拆成哪些已实现接口和待解决接口？

#### [Representation-driven Option Discovery · slides PDF](https://deeprlcourse.github.io/assets/guests/marlos_machado.pdf)

Marlos Machado · Sharif 2025 · 直接课件入口

先对照 subtask 与 option 的图，再看 ROD cycle 与 successor representation 的连接。

思考：为什么 SR 的不同分量可以看成一组共享策略和折扣的 GVFs？

#### [mcmachado/options](https://github.com/mcmachado/options)

Marlos Machado · 作者代码

从 main.py 和实验入口追到 option 的生成与执行；先跑小网格而非直接扩大任务。

思考：替换表示或内在奖励时，哪些实验配置必须保持一致？

历史实现应使用独立环境；这里提供读码路线，不宣称已复现其论文结果。

#### [Laplacian Keyboard](https://arxiv.org/abs/2602.07730)

Chandrasekar、Machado · 2026 · 近期组合方法

对照 SF / GPI 再读行为基与组合接口。

思考：线性奖励组合与任意策略拼接的表达能力差在哪里？

### 最小研究练习

从作者 options 代码的小环境开始。固定 primitive policy 数据，先比较随机 options、eigenoptions 与手工门口 option。

对照：相同 option 数、数据与总更新预算；分别打开技能探索与模型规划，不让两者混在一个消融里。

指标：首次到达时间、状态覆盖、主任务回报、技能持续时间、规划调用与训练开销。

容易误判：比较高层决策数却忽略 primitive steps；或用探索收益证明规划模型有用。

## 05 · 后果模型：预测执行一个动作或 option 会发生什么

一个技能会执行了，为什么还不能直接拿来规划？

先修：Bellman backup、期望的线性性；先看第 04 章的三步算例。

输入：已定义的 option、当前状态表示与环境经验。

输出：累计奖励、终点后果及时间信息；模型必须对应具体的行为与表示版本。

控制器回答“下一步怎么走”，后果模型回答“照这个控制器走完，会得到什么、到哪里、多久以后到”。这是两个不同的学习对象。即使外部环境不变，option 的策略或终止条件正在学习，也会让它的后果模型变成一个移动目标。

模型不必完整生成未来图像。要支持某种规划，必须保留那种规划所需的量：线性价值下的期望特征可能足够；非线性价值、风险或多峰后果则可能需要更多分布信息。模型的形式要由使用方式决定。

### 算法分支

#### 一步模型 → 多时间尺度模型

从 P(s′|s,a)、r(s,a) 扩展到 option 的累计奖励与按时长折扣的终点分布。

代表工作：options / multi-time models；TD model learning

继续问：内部策略、β 与环境都改变时，怎样让模型跟上？

#### 完整分布 → 用途匹配的预测

sample model 生成样本；distribution model 表示分布；expectation model 只给某些后果的期望。

代表工作：Wan 等：Planning with Expectation Models；STOMP 的特征后果模型

继续问：均值在哪些价值函数下够用？风险、非线性与多峰分布会丢掉什么？

#### 技能后果 → 符号与抽象状态

用技能的可执行条件与效果构造高层规划表示。

代表工作：Konidaris、Kaelbling、Lozano-Pérez：From Skills to Symbols

继续问：抽象是否保持可达性与规划正确性？技能变化以后，符号边界也可能要变化。

#### 重建世界 → 支持价值与行为

潜在模型可以服务 imagined learning 或搜索，不必都还原像素。

代表工作：Dreamer / MuZero；Director 的潜在世界模型

继续问：一层原始步动力学模型，不等于已经学到可组合的 option-level transition model。

### 将一个 option 压缩成 Bellman backup 所需的两项

$$
\begin{aligned}r_o(s)&=\mathbb E_o\!\left[\sum_{k=0}^{\tau-1}\gamma^k R_{t+k+1}\mid S_t=s\right]\\p_o^\gamma(s'\mid s)&=\mathbb E_o[\gamma^\tau\mathbf1\{S_{t+\tau}=s'\}\mid S_t=s]\end{aligned}
$$

τ 是 option 的随机持续时间；poγ 已包含折扣，其总和为 E[$γ^τ$]，通常小于 1，因此不是普通归一化转移概率。

1. 原始动作是 τ=1 的特例，此时 $p_a^γ=γP_a$。高层与低层模型可放进同一个 backup。

2. 终点位置与时间相关时，不能只保存平均时长，再单独乘普通终点概率。

3. 学习 model 时必须标明针对哪个 πo、βo；“同一个 option 名称”不能代替同一个行为定义。

### 为什么线性价值能使用期望特征模型？

$$
n_o(x)=\mathbb E_o[\gamma^\tau X_{t+\tau}\mid X_t=x],\qquad \mathbb E_o[\gamma^\tau w^\top X_{t+\tau}]=w^\top n_o(x)
$$

X 是特征向量，v(x)=$w^Tx$；no 是按时间折扣的终点特征期望。等式只是在同一分布与固定线性价值下移动期望。

1. 先把 $γ^τ$X 作为一个整体取期望，再与 w 点积。这样无需完整终点分布即可算这一类期望价值。

2. 非线性神经网络一般有 E[v(X)]≠v(E[X])。预测一幅“平均图像”再送给任意 critic，没有此等式保证。

3. 即使等式成立，表示混淆、模型估计误差和数据分布偏移仍会影响规划；“均值足够”不是“学习无误差”。

### 终点预测也能逐步自举

$$
n_o(x_t)\approx\mathbb E_{a\sim\pi_o}\!\left[\gamma\{\beta_o(S_{t+1})x_{t+1}+[1-\beta_o(S_{t+1})]n_o(x_{t+1})\}\right]
$$

采用本章标准 $γ^τ$ 终点模型约定。若下一步终止，预测 γx′；否则继续预测，并再乘一步 γ。

1. 奖励模型同理以 $R+γ(1-β)r_o(x^{\prime})$ 为 target；终点特征各分量都是一个预测任务。

2. 因此 GVF、option 与 model 不是互不相干的名单：通用预测学习提供了学习后果模型的一套接口。

3. 此式暂假定状态与 option 固定。off-policy 估计仍需校正与覆盖，非 Markov 的 agent state 还引入近似。

### 手算：“平均耗时两步”不能替代随机耗时

某技能总回报为 0、总到达同一终点，终点价值为 10。它一半概率用 1 步，一半概率用 3 步；γ=0.9。

正确后续价值是 $10×(0.9+0.9^3)/2=8.145$。用平均时长 2 代入，得到 $10×0.9^2=8.1$。

再看一个非线性反例：终点特征 X=−1 或 +1，各一半，$v(X)=X^2$。E[v(X)]=1，而 v(E[X])=0。误差不来自样本不足，而来自模型丢失的信息。

只预测“能否到达充电站”足以决定采用哪项技能吗？

参考答案：通常不够。两项技能可能成功率相同，但奖励损伤、执行时长和失败后位置不同。所需的后果量由折扣、平均奖励或风险敏感规划目标决定。

### 按问题读原始材料

#### [Planning with Expectation Models](https://www.ijcai.org/proceedings/2019/506)

Wan 等 · IJCAI 2019 · 核心论文

把每个等式中的价值函数形式标出来，找出线性性究竟用在哪里。

思考：若换成非线性 critic，哪一步不再成立？

#### [Planning with Expectation Models for Control](https://arxiv.org/abs/2104.08543)

Kudashkina、Wan、Naik、Sutton · 进阶理论

区分对状态价值的 backup 与对 action-value 的不当均值代入。

思考：为什么“能精确预测平均后果”仍不自动支持任意控制更新？

#### [From Skills to Symbols](https://www.jair.org/index.php/jair/article/view/11175)

Konidaris、Kaelbling、Lozano-Pérez · 2018 · 抽象规划

把 initiation set、effects 与高层符号联系起来，理解行为能力如何约束抽象。

思考：若底层技能改变，高层符号系统需要重学哪部分？

#### [Resolving the Sensorimotor Dilemma](https://www.youtube.com/watch?v=Af5UFE7CdKs)

George Konidaris · 讲座

带着“符号由什么感知运动能力支持”观看，再回读 Skills to Symbols。

思考：高层计划中的一个谓词，如何在真实交互中验证？

#### [Reward-Respecting Subtasks · model 部分](https://arxiv.org/abs/2202.03466)

Sutton、Machado 等 · 接口实例

重点核对奖励模型、终点特征模型与后续规划的接口，而不只看技能成功率。

思考：option 改进以后，旧模型是否仍描述它的真实后果？

### 最小研究练习

运行 knowledge_lab.py model，核对随机时长与非线性均值两个反例。再加入一个会改变 π 或 β 的 option。

对照：精确当前模型、冻结旧模型、持续更新模型；把表示固定，先隔离 option policy drift。

指标：奖励/终点/时长误差、Bellman target 误差、控制收益与模型更新预算。

容易误判：像素 MSE 下降不等于规划更好；只比较回报无法定位是模型、策略还是表示引入的误差。

## 06 · 规划：让已经学到的知识改变下一次决策

模型究竟用于更新价值、当前动作搜索，还是构造长时间尺度计划？

先修：Bellman optimality、Dyna 基本循环；前一章的奖励与终点模型。

输入：动作或 option 后果模型、当前价值、可用的计算预算。

输出：改进的价值或策略，或者当前决策；要计算模型误差与额外计算的代价。

规划的共同点是使用模型改善决策，但有不同位置：Dyna 用模拟经验更新学习器；MPC 或树搜索在决策时计算；层次规划使用跨多步的模型缩短推理深度。把它们统称“多想几步”会丢掉关键算法区别。

CRL 增加了知识寿命问题。昨天学会的穿门技能、旧终点模型与今天改变的货架布局可能不一致。更多规划可能更快传播错误；规划预算应花在哪里，以及何时回到真实交互校准模型，都变成研究问题。

### 算法分支

#### 真实样本更新 → 模型样本更新

Dyna 把模型学习、直接 RL 与 planning backups 放进同一循环；prioritization 决定先更新哪里。

代表工作：Sutton：Dyna；prioritized sweeping

继续问：真实经验有限时，哪些模拟更新仍有用，哪些在放大旧误差？

#### 原始步搜索 → 多步后果 backup

把 actions 与 options 放进同一规划操作，减少某些任务的价值传播步数。

代表工作：Silver、Ciosek：option models；STOMP；Jinnai 等：minimize planning time

继续问：减少 backup 数不是减少全部运行成本；发现、训练和维护抽象也花资源。

#### 当前奖励任务 → 新奖励下复用

模型、SF 或结构表示允许在一定条件下改变奖励并重新决策。

代表工作：SF / GPI；goal-conditioned planning；ALPS

继续问：固定数据学习的规划器能否在长期在线变化中维护有效知识，是额外问题。

#### 折扣回报 → 每单位时间的收益

平均奖励规划显式处理行为耗时的机会成本；并非取 γ=1 就结束。

代表工作：Wan、Naik、Sutton：Average-Reward Learning and Planning with Options

继续问：平均奖励目标本身不解决非平稳性、可塑性或状态漂移；深度近似需另查假设。

### 同一个 Bellman 操作，放入不同时间尺度的行为

$$
(\mathcal T V)(s)=\max_{o\in\mathcal O(s)}\left[r_o(s)+\sum_{s'}p_o^\gamma(s'\mid s)V(s')\right]
$$

O(s) 可以同时包含一步动作与较长 options。p 已包含 $γ^τ$；此处不能再多乘一个 γ。

1. 先固定 V：每个 option 提供一个“途中回报 + 终点接续价值”。取最大值形成一次 backup。

2. 若模型只给线性特征期望，将求和写为 $w^Tn_o(x)$，这是第 05 章等式的直接用途。

3. 已知模型时可算规划误差；学习模型时还需检查该模型对应的经验、策略与表示版本。

### 持续任务不必靠人为 episode：平均奖励视角

$$
h(s)=\max_o\mathbb E_o\!\left[\sum_{k=0}^{\tau-1}R_{t+k+1}-g\tau+h(S_{t+\tau})\mid S_t=s\right]
$$

g 是长期每个原始时间步的平均奖励率；h 是差分价值。该最优性方程需合适的 SMDP、可达性与有限时长等条件，不是任意非平稳系统的保证。

1. −gτ 表示执行这个 option 期间占用时间的机会成本。不能只按“每次技能执行得多少奖励”排序。

2. h 通常只确定到加性常数，还需要相应的规范化或参考值方法；这不同于简单把折扣 TD 的 γ 改为 1。

3. 若环境或行为持续漂移，估计的 g 与模型都在跟踪变化。应分别检查奖励率估计、模型误差与策略改进。

### 从 Bellman 方程到算法：期望时长不等于样本时长

$$
\delta_n=\widehat R_n-\bar R_nL_n(s,o)+\max_{o'}Q_n(s',o')-Q_n(s,o),\qquad\Delta Q=\alpha_n\delta_n/L_n(s,o)
$$

这里是 Wan、Naik、Sutton 2021 inter-option Differential Q 的核心，不是上一式的任意随机梯度。R̂ 是未折扣的 option 总奖励；L>0 是执行前的期望长度估计。

1. 同一个 δ/L 也用于更新奖励率：ΔR̄=ηαδ/L。长度另更新为 L←L+β(τ−L)，各式右侧使用旧 L。

2. 为什么不直接除这次的 τ？设每个周期奖励为 1，长度等概率为 1 或 3：真实每步奖励率是 1/Eτ=1/2，而 E[1/τ]=2/3。改变分母可能改变更新的零点。

3. 完整式 (6)–(9)、原 PDF 入口和可运行数值反例见“作者代码与公式对照”F6。本站独立测试不冒充作者的 options 实验代码。

### 手算：慢技能可能成功率更高，却降低长期收益

考虑两个都返回相同起点的循环技能：A 用 2 步得 3 分；B 用 5 步得 5 分。忽略探索与学习成本时，重复 A 的奖励率为 1.5，B 为 1。

若只比较每次调用的总奖励会选 B；按每个原始步的收益选 A。在 g=1.5 时，A 的净项为 3−1.5×2=0，B 为 5−1.5×5=−2.5。

这个简单例子由 knowledge_lab.py planning 校验。它解释时间单位，不是完整 average-reward 控制算法的复现。

加入 option 后，VI 收敛迭代数下降，是否证明真实系统更高效？

参考答案：还没有。要加上 option 学习、模型维护、一次 backup 的成本、内存与真实交互成本；也应检查 model 已知与 model 学习两种协议。

### 按问题读原始材料

#### [Sutton & Barto · 第 8 章与第 17 章](http://incompleteideas.net/book/the-book-2nd.html)

Reinforcement Learning: An Introduction · 教材入口

第 8 章看 Dyna 的真实/模拟更新；第 17 章将时间抽象接到动作模型。

思考：代码中一次 model call 的输出怎样进入更新？

#### [Finding Options that Minimize Planning Time](https://proceedings.mlr.press/v97/jinnai19a.html)

Jinnai、Abel、Hershkowitz、Littman、Konidaris · 2019 · 规划效用

把优化指标写成 VI 迭代数，再找其模型与初始化假设。

思考：同一批 options 是否也能提高未知奖励下的探索效率？

#### [Optimal-Options-ICML-2019](https://github.com/jinnaiyuu/Optimal-Options-ICML-2019)

Jinnai 等 · 论文页链接的代码 · 作者代码

先定位小环境、options 生成与规划评价入口，对照论文计量的 planning time。

思考：代码中一个规划操作的成本是否随 option 数量变化？

历史代码需单独固定依赖；不把链接可访问等同复现成功。

#### [Discovering Options for Exploration by Minimizing Cover Time](https://proceedings.mlr.press/v97/jinnai19b.html)

Jinnai、Park、Abel、Konidaris · 2019 · 对照阅读

与上一篇对读：技能发现的好坏取决于下游究竟要探索还是规划。

思考：更短 cover time 为何不等于更少 planning backups？

#### [Average-Reward Learning and Planning with Options](https://arxiv.org/abs/2110.13855)

Wan、Naik、Sutton · 2021 · 平均奖励主线

重点看 duration 怎样进入 inter-option / intra-option 学习与模型规划。

思考：如果用高层决策步数当时间单位，优化目标变成了什么？

#### [ALPS：Laplacian 表示与决策时规划](https://arxiv.org/abs/2602.05031)

Shehmar、Schlegel、Taylor、Machado · 2026 · 近期方法

接在表示—子目标—规划接口上读，并记录离线数据来源与测试时计算。

思考：若转为持续在线系统，表示、模型和数据覆盖哪些部分还要更新？

离线数据上的规划评测不能直接作为完整在线 CRL 的证据。

### 最小研究练习

固定两房间模型，先比较原始动作 VI 与带 options 的 VI；然后加入一次布局变化，比较冻结模型与在线重估。

对照：相同模型、初始化与 backup 数；再按总模型调用/墙钟时间重做比较；单独加入真实交互和模型学习成本。

指标：Bellman residual、规划误差、primitive-step 回报、错误传播和恢复时间；平均奖励实验按原始时间计数。

容易误判：把多做计算获得的收益全归给更好抽象；或者用错误旧模型做更多规划，再把失败归为策略遗忘。

## 07 · 完整智能体架构：让知识构建成为持续闭环

怎样把预测、状态、目标、技能、模型与规划放进同一生命期？

先修：先能区分 GVF、subtask、option 与 option model；不需要先精通全部论文。

输入：流式经验、有限资源，以及已经存在但可能过时的知识。

输出：持续改变的行为与知识系统；需要逐接口评测，而不只是一个漂亮的方框图。

架构不只是选 DQN 还是 PPO。它规定哪些量会被保存、什么模块产生训练目标、哪些模块共享经验、控制如何使用模型，以及哪些知识值得继续维护。只有这些接口明确，才能谈“越活越会学”。

把所有模块同时训练，是比单个组件更难的问题：状态变了，目标语义可能变；option 改了，模型目标会变；规划改变行为，数据覆盖又随之改变。需要研究的不只是每块是否可学，还包括耦合后是否有益、稳定、可持续。下面是阅读与实验的综合框架，不把所有工作视为同一种已完成的 CRL 系统。

### 算法分支

#### 学习与规划闭环

真实交互同时更新价值和模型，模型反过来产生学习信号。

代表工作：Dyna → learned world models / Dreamer

继续问：多数实现仍需明确回放、批量优化、预训练与任务边界；闭环不自动等于 strict streaming。

#### 共享经验的知识与辅助学习

不同预测/控制目标从一条经验流学习；可改善表示或提供预测知识。

代表工作：Horde；Jaderberg 等：UNREAL

继续问：辅助任务如何产生、怎样衡量用途？固定辅助目标不能代替自主目标发现。

#### 状态—子任务—技能—模型—规划

把抽象行为及其可预测后果作为知识单位，明确上下游接口。

代表工作：STOMP；ROD；Konidaris 的 skills-to-symbols；Director

继续问：不同工作完成不同连接。给定状态/目标、预训练技能和自主持续构建必须逐项区别。

#### 长期研究路线与构建式智能体

把有限计算、持续表示学习、知识发现与规划纳入同一研究议程。

代表工作：Sutton、Bowling、Pilarski：Alberta Plan；Sutton / Oak Lab：OaK

继续问：公开组件、概念演示与整体目标分开看；尚不能把路线图写成完整可复现系统。

### 把一个学习器看成带状态的动力系统

$$
L_{t+1}=F(L_t,O_{t+1},R_{t+1},A_t),\qquad A_t\sim\mu(\cdot\mid L_t)
$$

L 代表全部学习器状态：表示参数、预测器、目标/技能库、模型、价值、优化器及被允许的经验存储。F 是一次完整更新，不是一篇论文专有的公式。

1. 只保存神经网络参数，可能遗漏优化器、资格迹、模型和技能库；重置这些对象会改变持续学习问题。

2. 同一个 L 决定动作，动作决定以后的数据，数据又改变 L。因此离线固定数据上的改进不一定在闭环成立。

3. 观察某模块的局部误差下降，只是第一层证据。最终还要测后续学习和控制是否受益。

### 知识的价值，要放回同一预算与后续任务中比较

$$
\Delta J_B(k)=J_B(\text{agent with }k)-J_B(\text{matched agent without }k)
$$

这是本指南的实验比较量，不是现成的无偏在线 utility 算法。k 可以是预测、特征、option 或模型；B 是对齐的数据/计算/内存预算。

1. 删除 k 同时改变计算和数据分布时，要安排资源匹配与信息匹配的对照；否则 ΔJ 混合了多种原因。

2. 使用频率、TD error、学习进步、规划中被选次数都是可能的代理指标，但不等于这个反事实效用。

3. 长期有用的知识可能暂时不被调用；有限资源下生成、试用、保留和淘汰之间的信用分配仍是开放问题。

### 手算：仓库货架搬动一次，会牵动哪些模块？

状态：旧路口特征还能区分新布局吗？预测：旧策略的耗电/到达预测是否失效？目标：门口还值得成为子任务吗？

技能：原来的穿门策略要改；模型：必须预测修改后的技能，而不是继续描述旧动作；规划：旧估计可能让机器人反复选错。

分别固定表示、技能或模型做受控消融，最后再运行全部联动版本。前者定位问题，后者检验真正闭环。

“每个模块各自通过测试”足以说明整个智能体能持续积累吗？

参考答案：不够。各模块可能都在跟踪会被其他模块改变的目标；还需在没有隐式重置、明确预算的同一生命期里评估，检查耦合失效和后续任务收益。

### 按问题读原始材料

#### [The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173)

Sutton、Bowling、Pilarski · 总体路线

把它读成研究问题的依赖结构：表示、预测、控制、规划与抽象如何接起来。

思考：你的项目解决其中哪个接口，哪些先决条件仍由实验者提供？

是一份研究规划，不是单个可直接运行的算法。

#### [The OaK Architecture](https://oaklab.ai/posts/the-oak-architecture)

Richard Sutton · Oak Lab · 架构讲座

与前六章对照，标出目标/技能/模型的生成与用途；页面提供讲座入口。

思考：系统怎样从下游规划和控制中获得知识保留信号？

#### [Oak Lab Mission](https://oaklab.ai/mission)

Oak Lab · 研究目标

了解从经验中构建知识、时间抽象与有限资源的总体定位。

思考：哪些部分已有论文或演示，哪些仍是未来目标？

官网能效与规模愿景不作为已测得能力；完整架构需独立的系统级证据。

#### [DreamerV3](https://danijar.com/project/dreamerv3/)

Danijar Hafner 等 · 世界模型代码

追踪观测编码、动力学、想象轨迹和 actor/critic 的耦合。

思考：更长训练和强基准成绩能否证明自主形成新子任务或长期知识库？

#### [Director：层次行为、模型与目标](https://danijar.com/project/director/)

Hafner、Lee、Fischer、Abbeel · 架构实例

从项目页进入论文、演讲和作者代码；比较 manager/worker 与 option-model 接口。

思考：latent goals 的解码可解释性，与可用于规划的因果效果有何区别？

#### [danijar/director](https://github.com/danijar/director)

Director 作者实现 · 代码

从项目 README 建立独立环境，沿 world model、manager 与 worker 的调用关系读；先固定一个小任务。

思考：同一 imagined trajectory 分别给三个模块提供了什么学习目标？

深度训练实现与本站标准库小例子用途不同，计算预算需单独核算。

#### [Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397)

Jaderberg 等 · UNREAL · 辅助任务分支

看固定的辅助控制/预测任务怎样帮助表示，不要把任务数量当知识规模。

思考：如果辅助目标与后来任务无关，会怎样影响表示？

#### [David Abel · 研究与讲座](https://david-abel.github.io/)

David Abel · 概念与理论入口

沿 abstraction、agent 与 continual learning 理解“究竟哪里在学习”。

思考：部署后只更新 context、只更新参数、持续生成技能，各自改变了什么？

### 最小研究练习

先做“一个候选子目标 + 一项 option + 一个后果模型 + 固定预算规划”的最小闭环；再逐项释放手工假设。

对照：固定表示/技能/模型与全部联动；手工子目标、随机子目标、学习子目标；统一环境信息、数据和计算预算。

指标：分模块误差、技能复用、后续学习速度、整段收益、峰值内存、每步延迟和知识淘汰后的恢复。

容易误判：没有控制知识生成和维护成本；重置了学习器却称为 single-life；把模块叠加造成的收益称作已找到通用智能体。

## 可运行的接口检查

下载 knowledge_lab.py 后运行：

```sh
python3 knowledge_lab.py all
python3 knowledge_lab.py gvf --seed 7
```

脚本只使用 Python 标准库，向终端输出 JSON，不写入文件。包括表格 GVF 估计、子任务奖励算例、随机时长/均值模型反例及规划 backup。不是 Horde、STOMP 或 OaK 的完整复现。

原创教程 CC BY 4.0；原创教学脚本 MIT。所链接外部材料遵循各自许可。
