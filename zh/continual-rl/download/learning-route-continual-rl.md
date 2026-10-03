# CRL：长期交互中的学习、适应与知识积累

运行时，智能体与世界形成交互闭环；外部设计者选择奖励、初始化、数据权限、调参与预算。经典方法、神经表示和持续学习描述不同维度，可以共同用于同一智能体。下面按教学先修组织，不把三册视为互斥问题类，也不要求读完全部分支才开始研究。

[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：比较完整学习器](https://yingwen.io/zh/continual-rl/algorithms/control/)

基础目录包含表格方法、函数逼近与深度核心算法。深度拓展中的部分可观测、探索、回报分布、离线数据、模型、约束和多智能体是并列研究分支，可按问题选择。

持续强化学习研究受资源限制的完整智能体怎样在长期交互中适应环境、选择行动并积累知识。经典或深度方法都可以成为其部件。先区分运行时的智能体—世界闭环与外部设计者的奖励、初始化、调参和重置权限，再定义表现与可实现的比较者，最后选择机制。

学习目标：

- 能把“持续”写成环境、信息、reset、内存、计算与评价协议。
- 能分开状态、预测、信用分配、流式更新和元学习，并说明它们怎样组合。
- 能从一个最小反例出发，设计具有对照、独立测试和明确估计对象的研究实验。

## 1 · 完整学习器、目标与长期交互协议

比较的是当前策略，还是包含内部状态与更新规则的完整智能体？

Continuing 指任务没有自然终点；online 指数据到来后学习；严格 streaming 通常进一步限制经验重放与逐步预算；continual 关注学习能否长期持续。这些维度互不等价。固定环境中也可能需要不断学习，分任务的基准则可能允许任意 reset。

比较持续学习方案时，写清六项：哪些规律或任务改变，智能体能否看到变化标记，能否重置环境，哪些内部学习状态跨阶段保留，内存与计算如何计量，以及最终估计什么表现。设计者在完整测试生命期上挑超参数，与智能体根据已到达经验更新参数，是两种不同的信息权限。

当前动作分布相同，不意味着未来学习能力相同。一个学习率为零的贪心智能体和一个持续更新的智能体可以从相同 Q 值出发，却在奖励改变后获得不同收益。价值仍可在历史和完整算法条件下定义；把历史写进状态，并不自动解决表示、覆盖与有限计算问题。

长期在线效用要把学习期间的奖励也计入。平均奖励率是 continuing 问题的一种目标，折扣目标也是另一种合法选择；二者可能偏好不同策略。平均奖励的稳态存在、对初始状态的依赖和差分价值定义需要链结构条件。CRL 不要求所有任务一律改用 average reward。

$$
\begin{aligned}\bar J_T(\mathcal A,e)&=\mathbb E_{\mathcal A,e}\!\left[\frac1T\sum_{t=0}^{T-1}R_{t+1}\right],\\g(\mathcal A,e)&=\lim_{T\to\infty}\bar J_T(\mathcal A,e)\quad\text{若极限存在}.\end{aligned}
$$

A 包含行动规则、更新规则、内部状态与初始化；e 指定世界条件。有限生命期平均表现不要求稳态存在。一般非平稳学习过程的极限可能不存在；冻结策略的价值又是另一评价对象。

### 动手与核对

[下载 continual_control_lab.py](https://yingwen.io/zh/continual-rl/download/continual_control_lab.py)

在保存该文件的目录运行：

```sh
python3 continual_control_lab.py demo
python3 continual_control_lab.py test
```

预期检查：相同初始贪心策略，学习率 0、0.1、1 的十步回报分别为 0、3、9；不可逆反例中从初始世界的回报差为 96，沿实际历史的局部平均差仅 .02。

实验范围：已知小世界用于检验完整学习器与不同比较标准；没有声称真实单生命期中可同时观察两个算法的反事实。

自测：两个智能体现在的动作分布相同，能否只用一个冻结策略价值判断未来谁更好？

解答：不能。未来结果还依赖其更新规则、内部记忆、资源及世界条件。可以定义包含后续学习的条件价值，但不能把当前冻结策略的价值与之混为一谈。

完整推导与问题衔接：

- [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)
- [奖励假设与奖励设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/)
- [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)
- [平均奖励与差分价值](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

原文、课程与实现：

- [Abel 等 · A Definition of Continual Reinforcement Learning](https://arxiv.org/abs/2307.11046)：相对于 agent basis 定义持续学习；不把 CRL 限定为外部任务切换。
- [Elelimy 等 · Rethinking the Foundations for Continual Reinforcement Learning](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_243.pdf)：学习规则、历史依赖世界与偏离比较；与从初始世界比较总回报的含义分开。
- [Mesbahi 等 · Lifetime tuning is incompatible with continual reinforcement learning](https://proceedings.mlr.press/v267/mesbahi25a.html)：外部设计者用完整部署寿命调参与在线因果更新的信息权限不同。
- [Sutton & Barto · Reinforcement Learning, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：第 10.3 节：平均奖励和差分价值。

## 2 · 状态与预测知识的分工

长期经验应当形成什么可供后续决策使用的知识？

状态构建回答“当前历史中哪些信息需要保留”。预测知识回答“在指定行为条件下，未来某个信号会怎样”。例如机器人可以记住最近是否见过充电站，也可以预测沿墙策略下到达充电站的能耗。前者是信息表征，后者是带明确语义的未来问题。

GVF 用目标策略 $\pi$、累积信号 $C$ 和延续规则 $\gamma$ 规定一个预测。把任务奖励换成温度、接触次数或能耗，就能定义不同问题。GVF 是问题规格，不是 TD 的竞争算法；TD、GTD、Emphatic TD 等是在不同采样和稳定性条件下学习它的办法。

同一行为流可训练多个 GVF，但行为必须覆盖问题所需动作。若目标策略与行为不同，需要说明 off-policy 校正和函数逼近稳定性。把这些预测作为 state feature 或 model component 还需要验证它们对控制的充分性与效用；准确预测一些信号，并不自动形成充分状态。

$$
G_t^{\rm GVF}=\sum_{k=0}^{\infty}\left(\prod_{j=1}^{k}\gamma_{t+j}\right)C_{t+k+1},\qquad v(s)=\mathbb E_\pi[G_t^{\rm GVF}\mid S_t=s]
$$

空乘积为 1；延续权重可依赖转移。需保证回报与期望存在，例如有界信号配合统一小于 1 的延续上界。某个 GVF 的延续变零，只停止它所问的未来，不必重置真实环境。

### 动手与核对

[下载 gvf_lab.py](https://yingwen.io/zh/continual-rl/download/gvf_lab.py)

在保存该文件的目录运行：

```sh
python3 gvf_lab.py compare
python3 gvf_lab.py test
```

预期检查：对照多个预测头的解析答案和各学习器误差；改变目标策略时同时检查采样行为是否仍有覆盖。

实验范围：固定小环境用于核对预测算法，不检验这些预测构成控制充分状态。

自测：行为在一个状态下均匀选两动作，信号分别为 0、1，γ=.5。目标策略总选信号 1 的动作。预测答案是 1 还是 2？

解答：目标策略答案为 1/(1-.5)=2。若不做适当 off-policy 处理而学到行为策略的问题，答案会是 .5/(1-.5)=1。

完整推导与问题衔接：

- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)
- [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)

原文、课程与实现：

- [Sutton 等 · Horde](https://people.bordeaux.inria.fr/degris/papers/Sutton_Horde_aamas_11.pdf)：作者提供的论文：共享传感运动经验学习多个预测问题。
- [Sutton、Bowling、Pilarski · The Alberta Plan](https://arxiv.org/abs/2208.11173)：把状态、预测、控制、子任务和模型作为相互衔接的研究问题。

## 3 · 信用分配与流式更新是两个问题

反馈怎样传到过去，与每步允许做多少计算，有什么区别？

时间信用分配决定当前反馈应当影响哪些过去预测、动作或参数路径。流式协议决定数据何时使用、能保存什么、是否可重放以及每步算多久。资格迹可以用于流式学习，也可以与回放或批处理共存；这两个概念不应合并成一个算法类别。

在没有 replay 和 target network 的严格协议下，观测尺度、奖励尺度、迹长度和 actor–critic 耦合会影响更新稳定性。Stream-X 一类工作研究这些机制的组合。按当前输入或梯度缩放步长，也不必是元学习：关键是规则有没有通过后续学习效果被训练。

有效步长可以先从局部输出变化理解。线性预测器沿迹更新后，当前预测改变量是 $\alpha\delta_t x_t^\top e_t$。相同数值步长在特征尺度或迹范数改变后，可能导致很不同的输出变化。对这种尺度进行控制不能替代对长时信用分配或 off-policy 偏差的分析。

$$
\Delta w_t=\alpha\delta_t e_t,\qquad \Delta\hat v(S_t)=x_t^\top\Delta w_t=\alpha\delta_t x_t^\top e_t
$$

等式对固定线性特征准确；神经网络中以梯度代替 x 只是局部一阶近似，参数更新还会改变梯度与表示。

### 动手与核对

[下载 lifelong_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)

在保存该文件的目录运行：

```sh
python3 lifelong_algorithms_lab.py streaming
python3 lifelong_algorithms_lab.py test
```

预期检查：两步迹更新为 (.072,.1)，在线均值和方差为 (2,1)；还包含大梯度下步长启发式失效的反例。

实验范围：只隔离 traces、在线统计与尺度机制，不是完整 Stream-X 复现，也不是稳定性定理。

自测：一个算法每条新经验只更新一次，但保留资格迹和归一化统计量。它是否违反“无经验回放”？

解答：不违反。压缩学习状态不是保存样本后重新训练；仍需报告这些状态的内存和每步计算。若保存整段轨迹反复更新，则是另一种协议。

完整推导与问题衔接：

- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)
- [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

原文、课程与实现：

- [van Seijen & Sutton · True Online TD](https://proceedings.mlr.press/v32/seijen14.html)：时间信用分配的线性在线等价性。
- [Elsayed 等 · Streaming Deep RL, v3](https://arxiv.org/abs/2410.14606v3)：流式网络训练的完整机制与预算设置；与早期版本区分。
- [Stream-X 作者源码](https://github.com/mohmdelsayed/streaming-drl/tree/9326fc3e23a401f28087ae2e41b635888740586b)：固定版本入口；按 optimizer、归一化和 agent 主循环阅读。

## 4 · 学习规则的适应与跨任务元学习

什么时候需要学习“怎样学习”，而不仅是更新策略？

在线元梯度把步长、训练目标或更新规则参数记为 $\eta$，研究它们怎样影响后续预测误差或回报。内层权重持续更新；外层通过这些更新的影响调整学习规则。它可以发生在同一条经验流上，不要求一组可重置任务。

跨任务 meta-RL 则先规定训练任务分布和任务内适应过程。MAML 学习便于少量梯度适应的初始化；RL² 或上下文方法可把任务适应放在循环活动或隐变量中。测试时是否还更新网络权重、是否清空上下文、能否得到任务边界，都必须明确。

二者共享对学习过程的关注，但不共享所有评价条件。某个规则在新任务上适应快，不代表它在一次没有参数重置的长寿命中始终有效。固定数据下求内层更新的导数，也不自动包含策略改变导致的采样分布导数。

$$
w_{t+1}=F_\eta(w_t,\xi_t),\qquad H_{t+1}=\frac{\partial F_\eta}{\partial w_t}H_t+\frac{\partial F_\eta}{\partial\eta},\qquad H_t=\frac{\partial w_t}{\partial\eta}
$$

这里先固定经验 ξ 和元参数，H 传播过去学习对 η 的敏感度。对 RL 期望目标求完整梯度时，还要讨论数据采样路径与近似。

### 动手与核对

[下载 state_meta_lab.py](https://yingwen.io/zh/continual-rl/download/state_meta_lab.py)

在保存该文件的目录运行：

```sh
python3 state_meta_lab.py meta
python3 state_meta_lab.py test
```

预期检查：两步内层更新的 (w,H) 为 (.2,.2)、(.38,.36)，后续评价对 log-step-size 的导数为 -.9432；比较解析导数与有限差分。

实验范围：固定数据的标量敏感度和上下文演示，不报告 meta-RL 基准性能。

自测：根据当前梯度范数确定一个较小步长，是否已经是元学习？

解答：不一定。这可能是预设的尺度控制。若规则或其参数根据过去更新对后续学习效果的影响被训练，才明确涉及本节的学习规则适应。

完整推导与问题衔接：

- [元学习与学习规则的适应](https://yingwen.io/zh/continual-rl/algorithms/meta/)

原文、课程与实现：

- [Xu 等 · Meta-Gradient RL](https://arxiv.org/abs/1805.09801)：在线学习训练目标参数与后续评价。
- [Finn 等 · MAML](https://proceedings.mlr.press/v70/finn17a.html)：任务内适应与任务外目标的设定。
- [Duan 等 · RL²](https://arxiv.org/abs/1611.02779)：用循环活动承载任务内适应的不同路线。

## 5 · 知识保留与可塑性

一个学了很久的网络变差，是忘记了，还是学不进了？

遗忘要比较同一旧任务在新学习前后的表现。可塑性损失要比较有学习历史的网络与合适对照在同一新任务上的学习过程。两种失败可能同时发生，但测量对象不同。只报告旧任务回报，不能推断学习新任务的能力；只报告激活率，也不能替代学习曲线。

回放通过数据分布保留经验，参数正则限制部分权重改变，输出蒸馏约束已有功能。它们都可能在保护旧知识时限制新学习。ReDo 与 continual backpropagation 等方法研究失效或低效用特征的更新与替换，必须同时检查新学习与旧能力。

一个最小诊断可以固定新任务、数据顺序、网络规模和更新预算，比较 aged、fresh 与随机替换对照。fresh 的初始化、优化器状态和超参数选择也要对齐。性能改善可能来自步长、尺度或重置优化器，不能仅凭替换神经元就归因于一种可塑性机制。

$$
F_A=J_A(\theta_{\rm before})-J_A(\theta_{\rm after}),\qquad P_C(K)=L_C(\theta_{\rm aged}^{(K)})-L_C(\theta_{\rm fresh}^{(K)})
$$

$F_a$ 表示旧任务收益下降；$P_c$ 在匹配 K 次新任务学习后比较损失。这里用较小损失为好，正值表示 aged 学得较差；二者不能互相替代。

### 动手与核对

[下载 lifelong_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)

在保存该文件的目录运行：

```sh
python3 lifelong_algorithms_lab.py plasticity
python3 lifelong_algorithms_lab.py test
```

预期检查：机制例子中 aged 的新任务损失为 .5，fresh 约为 .019865；检查单元替换前后输出和清零的优化器状态。

实验范围：这是构造的 dead-ReLU 反例，不是长期训练导致可塑性损失的实证复现。

自测：重置网络后新任务学习更快，是否足以说明新方法实现了持续知识积累？

解答：不够。还要检验旧知识损失、后续多任务表现，以及相同预算下的 fresh、随机替换和优化器重置对照。持续积累要求保留与新学习共同受益或明确权衡。

完整推导与问题衔接：

- [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)
- [可塑性与特征更新](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)

原文、课程与实现：

- [Rolnick 等 · CLEAR](https://papers.nips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html)：回放、off-policy 校正与行为克隆的组合。
- [Sokar 等 · Dormant Neurons / ReDo](https://proceedings.mlr.press/v202/sokar23a.html)：深度 RL 中的休眠单元与回收机制。
- [Dohare 等 · Loss of Plasticity](https://www.nature.com/articles/s41586-024-07711-7)：长期训练与 continual backpropagation。
- [CBP 作者实现](https://github.com/shibhansh/loss-of-plasticity)：效用计算、成熟期、替换预算及优化器状态处理。

## 6 · 从预测到子任务、技能、模型与规划

长期经验怎样形成可重复使用的行为，而不只是一组新参数？

先有总体任务奖励，再问哪些中间目标值得学习。目标条件化策略把目标作为输入，用同一个函数表示一族控制问题；HER 改写已发生轨迹的目标与相应奖励，改善稀疏目标学习。目标构造决定学习什么，而不是仅给策略输入一个额外向量。

Option 把可复用行为定义为启动集合、内部策略和终止规则。高层选择 option 后，低层会执行随机时长 $\tau$。因此高层 target 必须累计这段真实奖励，并用 $\gamma^\tau$ 折扣终点价值。技能发现决定哪些 option 值得加入集合，内部策略优化则决定已有 option 如何执行。

可执行技能还不足以规划。模型需要预测它的累计奖励和带时长折扣的终点分布；规划再比较不同技能的长期后果。STOMP 将子任务、option、模型和规划串成这种因果关系。OaK 和 Alberta Plan 提出更广的持续构建框架；研究架构时仍要逐项检验接口和收益，不能把纲领当成完整算法。

$$
Q(s,o)\leftarrow Q(s,o)+\alpha\left[\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}+\gamma^\tau\max_{o'\in\mathcal O(S_{t+\tau})}Q(S_{t+\tau},o')-Q(s,o)\right]
$$

这是 option 结束时的 SMDP Q-learning target。O(s′) 是终点可启动的合法 option 集合；环境真实终止时余项为零。随机时长与终点可能相关。

### 动手与核对

[下载 knowledge_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/knowledge_algorithms_lab.py)

在保存该文件的目录运行：

```sh
python3 knowledge_algorithms_lab.py all
python3 knowledge_algorithms_lab.py test
```

预期检查：奖励 1、2，γ=.9、两步后终点价值 10，SMDP 与 option 模型 backup 均为 10.9；检查 HER、终止规则和合法 option 集合测试。

实验范围：每个机制使用可手算环境；脚本不是 STOMP 或 OaK 全系统复现。

自测：一个技能有时一步结束、有时三步结束。只保存平均时长，再用 γ 的平均时长次方乘终点价值，是否一般正确？

解答：不正确。需要期望中的 $γ^τ$ 与终点价值乘积；指数非线性，且时长可能与终点相关。正确模型保存相应的折扣终点权重。

完整推导与问题衔接：

- [目标条件化与子任务构造](https://yingwen.io/zh/continual-rl/construction/goals/)
- [Options 与技能发现](https://yingwen.io/zh/continual-rl/construction/options/)
- [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)
- [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)
- [持续学习的智能体架构](https://yingwen.io/zh/continual-rl/construction/architectures/)

原文、课程与实现：

- [Sutton 等 · Reward-Respecting Subtasks / STOMP](https://arxiv.org/abs/2202.03466)：子任务定义如何连接 option 学习、模型和规划。
- [Sutton、Precup、Singh · Options](https://doi.org/10.1016/S0004-3702(99)00052-1)：启动、内部策略、终止与 semi-Markov backup。
- [Oak Lab · The OaK Architecture](https://oaklab.ai/posts/the-oak-architecture)：架构研究纲领与讲座入口；与单个已实现算法区分。

## 7 · 持续探索、学习进展与恢复

长期智能体应该把有限经验花在哪些问题上？

学习器只会从实际获得的数据中学习。环境变化后，旧行为可能不再访问变好的区域；更新规则即使能快速适应，也没有收到新信息。持续探索因此不是训练早期的一次附加技巧，而是长期经验分配的问题。

新奇、预测误差、不确定性和学习进展并不相同。RND 用固定随机目标与可训练预测器的误差衡量不熟悉程度；但噪声也会造成误差。学习进展比较一段训练前后的改变量，试图区分难而可学的区域与不可预测噪声。探索奖励还会改变数据分布及 critic 的目标。

single-life 条件增加恢复和安全约束。探索到一个无法返回的状态，会影响后续全部学习机会。能够由外部生成任务、重置机器人或恢复仿真的课程算法，不能直接视为无这些权限的问题解法。应把可行动作、恢复策略和重置成本写进协议。

$$
b_t(s)=\|f_{\rm predictor}(s)-f_{\rm fixed}(s)\|^2,\qquad \ell_t(s)=E_{\rm before}(s)-E_{\rm after}(s)
$$

第一式是 RND 型误差奖励，第二式是简化学习进展指标。它们依赖模型和测量窗口，不等于真实信息增益或安全保证。

### 动手与核对

[下载 lifelong_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)

在保存该文件的目录运行：

```sh
python3 lifelong_algorithms_lab.py exploration
python3 lifelong_algorithms_lab.py test
```

预期检查：RND 误差从 4 降到 2.25、1.265625、.711914；若重置预测器，同一输入又显示误差 4。

实验范围：标量机制与动作过滤示例不构成真实机器人安全控制器。

自测：一个熟悉场景在重置预测器后获得很高内在奖励，能否据此判断环境发生了新变化？

解答：不能。误差也可能来自学习器状态丢失。要区分环境变化、表示变化、统计量重置和预测器遗忘。

完整推导与问题衔接：

- [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)
- [目标条件化与子任务构造](https://yingwen.io/zh/continual-rl/construction/goals/)

原文、课程与实现：

- [Burda 等 · RND](https://arxiv.org/abs/1810.12894)：冻结随机目标和可训练预测器的原始探索方法。
- [RND 作者代码](https://github.com/openai/random-network-distillation)：完整 Atari 实现及观测、内在奖励归一化。
- [Eysenbach 等 · Leave no Trace](https://arxiv.org/abs/1711.06782)：前向学习与恢复行为的共同设计。

## 8 · 从学习材料到可检验的研究实验

什么样的实验能够支持“更适合持续学习”的结论？

先选择一个具体失败：奖励反转后跟踪太慢、延迟线索无法学到、长期训练后新任务学习变慢，或旧模型阻碍新规划。固定基础 agent 和预算，提出只针对这个失败的机制假设。先用有解析答案或明确反例的小环境检查实现，再进入更复杂 benchmark。

把开发任务和种子用于调参与选择，保留独立测试用于最终估计。时间步不是独立重复实验；一条生命中的回报具有相关性。若使用共同随机数配对，需说明两算法共享的是哪些潜在随机事件，而不仅是把整数 seed 设为相同。

同时报告整个寿命收益、变化附近的短期代价、最终或分阶段冻结评价，以及资源预算。冻结诊断应运行副本，不能把测试产生的更新写回训练 agent。恢复时间在观察结束时仍未恢复，就属于删失数据，不能只平均成功恢复的运行。

复杂基准用于检验机制能否推广，而不是替代机制解释。例如 AgarCL 将探索、记忆和长期学习放到持续环境中；不同动作集、重生规则和算法预算仍需对齐。小实验通过后，再使用原环境与对应基线，逐项保留实验条件。

$$
\Delta=\mathbb E[U_A-U_B],\qquad \hat\Delta=\frac1n\sum_{i=1}^{n}(U_{A,i}-U_{B,i})
$$

U 是事先指定的每条生命统计量，例如固定寿命平均回报。配对只在第 i 对具有明确共同随机结构时成立；bootstrap 应重采样独立生命或实验块，而非任意时间步。

### 动手与核对

[下载 experiment_design_lab.py](https://yingwen.io/zh/continual-rl/download/experiment_design_lab.py)

在保存该文件的目录运行：

```sh
python3 experiment_design_lab.py demo --tiny
python3 experiment_design_lab.py test
```

预期检查：开发集选参数后，在 8 条独立测试生命上比较；tiny 示例平均差约 .03333，95% bootstrap 区间跨零，不能宣称稳定收益。

实验范围：小型 bandit 的输出用于验证统计流程和恢复删失处理，不是新方法性能证据。

自测：八条运行里只有两条在预算内恢复，能否把这两条的平均恢复时间当作算法的恢复能力？

解答：不能。这会漏掉更慢的六条。应同时报告恢复比例、观察上限和删失信息；需要总体时间摘要时采用适当的删失处理。

完整推导与问题衔接：

- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)
- [持续学习的智能体架构](https://yingwen.io/zh/continual-rl/construction/architectures/)

原文、课程与实现：

- [Patterson 等 · Empirical Design in RL](https://jmlr.org/papers/v25/23-0183.html)：目标量、算法比较、超参数选择与不确定性的系统说明。
- [AgarCL · The Cell Must Go On](https://arxiv.org/abs/2505.18347)：持续环境与分解问题的基准设定。
- [AgarCL 作者环境](https://github.com/machado-research/AgarCL)：环境、配置和安装入口。
- [AgarCL 作者基线](https://github.com/machado-research/AgarCL-benchmark)：完整游戏的不同基础 agent；需要匹配观测、动作和训练预算。

下一步：完成这一册后，可按问题进入对应教材章，而不必一次读完全部方向。一个可执行的起点是：固定基础 agent 和预算，只改变一个长期学习条件，提出能被对照实验否定的机制假设。
