# 持续强化学习：研究问题、方法与实验

教材解释更新规则；研究地图比较这些规则能够解决的问题和仍受限制的条件。以下实验是可检验的研究设计，不是已得到的结果。

## Ⅰ · 目标与设定

评价的是最终策略，还是整个学习生命期？

### 奖励表示与奖励学习：目标怎样进入智能体？

问题设定：设计者拥有偏好，智能体只能接收有限观测和奖励。奖励可能手写，也可能由人的比较或示范推断。必须分开目标变化、奖励估计变化与环境变化。

已有认识：期望效用与局部加性奖励需要不同条件。势函数塑形在边界条件满足时保持策略排序；偏好模型可以从片段比较中学习。

尚缺什么：新行为会改变奖励模型的数据分布。有限反馈下，怎样保持奖励语义并防止代理失效仍是开放问题。

#### 方法与条件

- **表示条件与奖励状态**：用反例检查原状态是否丢失时间结构；必要时显式增加奖励自动机或记忆。
  条件与代价：更大状态不保证任意偏好都可由有限计算表示。
- **偏好学习与重标记**：比较损失更新奖励模型；控制器使用更新后的信号并产生新经验。
  条件与代价：反馈噪声、片段选择和奖励模型版本都属于协议。

#### 相互竞争的解释

- 收益可能来自更接近真实目标，也可能只是更稠密的反馈。
- 奖励下降可能来自目标变化、模型外推错误或控制器失去可塑性。

#### 可执行实验

假设：相同反馈预算下，版本一致的经验重标记可减少奖励模型更新后的控制滞后。

设计：固定环境与真实评价，仅改变奖励模型；再独立改变真实目标。设置训练分布内与新策略片段两种偏好测试。

对照：真奖励诊断、固定模型、不重标记、全部重标记；匹配查询与计算预算。

测量：原任务收益、偏好校准、奖励投机行为、适应成本与反馈数。

什么结果会反驳该解释：若优势只来自更高反馈量，或训练奖励提高但真实评价下降，则不支持目标学习改进。

#### 教材与研究者

[奖励假设与奖励设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/) · [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/) · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/) · [目标条件化与子任务构造](https://yingwen.io/zh/continual-rl/construction/goals/) · [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

[Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/) · [David Abel](https://yingwen.io/zh/continual-rl/resource/S08/) · [Michael Bowling](https://yingwen.io/zh/continual-rl/resource/S17/) · [Satinder Singh](https://yingwen.io/zh/continual-rl/resource/S41/) · [Will Dabney](https://yingwen.io/zh/continual-rl/resource/S65/)

#### 实验平台

- [解析小 MDP 与短序列](https://yingwen.io/zh/continual-rl/labs/)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。

#### 原文与代码

- [Bowling et al. · Settling the Reward Hypothesis](https://proceedings.mlr.press/v202/bowling23a.html)（ICML 2023）：区分期望效用表示与逐步 Markov 奖励表示；时间一致性是额外条件。
  undefined
- [Abel et al. · On the Expressivity of Markov Reward](https://arxiv.org/abs/2111.00876)（NeurIPS 2021）：给定状态和奖励输入后，并非所有任务偏好都能由固定 Markov 奖励表达。
  undefined
- [Lee et al. · PEBBLE](https://arxiv.org/abs/2106.05091)（2021）：用预训练与奖励重标记提高偏好反馈的利用率。反馈模型与策略的数据分布共同变化。
  [B-Pref / PEBBLE 作者代码](https://github.com/rll-research/BPref)：奖励模型、查询策略、回放重标记与训练配置。

---

### 01 · 什么目标能够评价一个始终在学习的智能体？

问题设定：智能体只有有限存储和每步计算时间。学习改变其后续行为，环境可能持续而不终止，也可能存在成本不为零的重置。比较前需规定奖励、时间单位、评价时域和可用干预。平稳环境仍可能对有限资源智能体提出长期学习需求。

已有认识：折扣回报、有限生命期累计奖励和长期平均奖励并非同一个目标。平均奖励消除人为终止对目标的部分影响，但忽略有限前缀的长期极限也可能掩盖高昂适应成本。Abel 的持续性定义与 history-process 研究进一步把学习过程本身纳入讨论。

尚缺什么：仍需要连接渐近算法理论、有限寿命在线效用与真实时间成本。把折扣设为接近一，不能自动得到差分价值算法；平均奖励表现好，也不能单独证明智能体一直在累积有用能力。

#### 方法与条件

- **Differential TD/Q → RVI-SAC**：差分 TD 用 $R_{t+1}-\bar r_t$ 替代未中心化奖励，联合更新奖励率与相对价值。RVI-SAC 进一步把平均奖励目标接到熵正则 actor–critic，并处理 reset cost。
  条件与代价：表格收敛定理不直接覆盖非线性深网或任意漂移。RVI-SAC 的 replay 和目标网络也不属于严格 streaming 协议。
- **Reward centering 与协议比较**：从折扣 critic 中分离共同奖励偏移，减少无关基准对数值尺度的影响；同时显式比较三种 reset 协议。改变估计方法不必然改变任务目标。
  条件与代价：奖励中心化、平均奖励优化和延长 episode 是不同干预，实验不能将三者同时改变后只归因于平均奖励。
- **History process / deviation regret**：用交互历史刻画问题，并通过允许的行为偏离比较实际学习过程。评价对象不再只是训练完成后的一个策略。
  条件与代价：必须规定偏离集合、可观测信息和估计条件；这是一条形式化路线，而非已统一整个领域的标准。

#### 相互竞争的解释

- 方法收益可能来自更合适的任务目标，也可能只是奖励中心化改善了数值条件。
- 高终局成绩可能伴随更长探索期；高在线平均分也可能来自初始先验，而非更强持续学习。

#### 可执行实验

假设：算法排序的一部分差异由目标与 reset 权限造成，而不是网络表达能力造成。

设计：从教材循环奖励例子开始，解析比较折扣与平均奖励的策略排序；随后固定同一控制环境，分别改变折扣、奖励常数偏移和 reset 成本，每次只改一项。

对照：同网络、数据预算、种子与调参次数；记录 reset 是否由 agent 决策。设置不学习、固定步长学习和对应平均奖励方法。

测量：整段累计奖励、窗口奖励率、冻结策略诊断、恢复时间、真实交互秒数和重置次数。

什么结果会反驳该解释：在匹配目标、中心化与重置条件后排序差异消失，就不支持“目标形式本身带来普遍收益”的解释。

#### 教材与研究者

[交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/) · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/) · [平均奖励与差分价值](https://yingwen.io/zh/continual-rl/algorithms/average-reward/) · [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

[Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/) · [A. Rupam Mahmood](https://yingwen.io/zh/continual-rl/resource/S05/) · [David Abel](https://yingwen.io/zh/continual-rl/resource/S08/) · [Michael Bowling](https://yingwen.io/zh/continual-rl/resource/S17/)

#### 实验平台

- [解析小 MDP 与短序列](https://yingwen.io/zh/continual-rl/labs/)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [DeepRL-continuing-tasks](https://github.com/facebookresearch/DeepRL-continuing-tasks)：无重置、预设重置、智能体控制重置的连续控制。 奖励偏移、回报目标、重置成本对方法排序的影响。 边界：continuing 不自动包含任务漂移或无界学习需要。
- [CSuite](https://rl-csuite.readthedocs.io/en/latest/)：持续交互的小型诊断环境。 低成本测试适应与学习机制，再接复杂平台。 边界：每个任务测量的困难有限；不可用单一分数代表整个 CRL。

#### 原文与代码

- [Wan, Naik & Sutton · Learning and Planning in Average-Reward MDPs](https://proceedings.mlr.press/v139/wan21a.html)（ICML 2021）：用 TD 误差更新平均奖励率，建立无参考状态的差分预测、控制与规划更新。收敛条件针对论文规定的表格问题。
  [作者实验代码](https://github.com/abhisheknaik96/average-reward-methods)：从 prediction_agents.py、control_agents.py 对照奖励率和价值的联合更新。
- [Hisaki & Ono · RVI-SAC](https://proceedings.mlr.press/v235/hisaki24a.html)（ICML 2024）：把相对价值迭代、平均奖励 soft policy improvement 和 reset cost 接到深度连续控制。
  [作者 RVI-SAC 实现](https://github.com/yhisaki/average-reward-drl)：检查 rvi_sac.py 的 critic 基准、actor 目标和重置成本，不仅修改 discount。
- [Wan et al. · An Empirical Study of Deep RL in Continuing Tasks](https://arxiv.org/abs/2501.06937)（2025 论文）：比较无重置、预设重置和智能体控制重置的任务，分析 reward centering 对深度方法的影响。
  [作者环境与实验](https://github.com/facebookresearch/DeepRL-continuing-tasks)：协议、奖励中心化和各基线配置同时构成实验对象。
- [Elelimy et al. · Rethinking the Foundations for Continual RL](https://rlj.cs.umass.edu/2025/papers/Paper243.html)（RLC 2025）：以 history process 和 deviation regret 研究学习过程的评价。比较对象与可实现偏离集合仍需明确，不能只把指标名称换掉。
  论文给出形式化与实验；此处没有可确认的作者原始代码入口。
- [Abel et al. · A Definition of Continual Reinforcement Learning](https://arxiv.org/abs/2307.11046)（NeurIPS 2023）：用智能体对行为基础的持续搜索定义持续性。它与“环境显式切换若干任务”的定义不相同。
  理论定义论文；没有需要复现的单一控制算法。

## Ⅱ · 状态与预测

历史中的哪些信息应当成为可用知识？

### 02 · 有限的内部状态应当保留哪些历史信息？

问题设定：当前观测不能唯一决定后续结果。智能体从历史构造内部状态，但内部状态维度、计算和信用传播长度受限。环境的隐藏状态、网络活动、可学习参数以及 replay buffer 是不同对象。

已有认识：状态更新 $h_{t+1}=f_\theta(h_t,A_t,O_{t+1})$ 只规定如何压缩历史，不保证压缩后具有 Markov 性。控制充分性允许丢掉与最优决策无关的信息；完整预测充分性通常更强。递归结构和用于训练它的目标需要分别选择。

尚缺什么：仍缺少在相同信息、容量和延迟条件下，对“保留什么”与“怎样学会保留”作出明确归因的证据。更长记忆也可能存下无关历史；表示不断改变还会让依赖它的 critic 和模型失效。

#### 方法与条件

- **历史堆叠 / GRU / LSTM**：历史堆叠显式保留固定窗口；门控递归通过活动延续信息。BPTT/TBPTT 则决定参数能够接收到多远的训练信用，不能由 hidden state 不清零推断信用也没有截断。
  条件与代价：要匹配观测历史权限和实际训练窗口。完整回放序列、缓存 hidden state 与纯逐步处理具有不同成本。
- **GVFN：用预测约束状态**：将递归坐标与一组可由经验检验的预测联系起来，以预测误差训练状态。它回答“内部坐标应携带什么”，而不仅增加网络容量。
  条件与代价：选择的问题必须能被数据覆盖，并对下游有用；有限预测集合不会自动成为充分状态。
- **RTU → 部分可观测 streaming RTRL**：结构化递归使参数对状态的敏感度可以逐步维护。近期工作把该敏感度与控制资格迹结合，同时研究在线权重变化造成的敏感度陈旧。
  条件与代价：高效精确计算依赖递归结构；不是任意 RNN 都能在线性成本下完成 RTRL。2026 工作仍是预印本。

#### 相互竞争的解释

- 提高回报可能来自额外历史信息，而不是更好的表示学习目标。
- RNN 比前馈网络好，可能只是在隐式推断任务标签；也可能真正记住了任务内关键事件。

#### 可执行实验

假设：预测式内部状态在相同容量下更容易保留决策所需的稀疏历史事件。

设计：先做延迟线索记忆链，分别改变线索延迟、无关观测和任务切换；再在 Forager 中只改变一个观测模式。对状态做冻结线性探针，但不把探针数据反馈给 agent。

对照：同参数量、动作信息与每步延迟预算；前馈、固定窗口、GRU/RTU、加预测损失各自单独比较。

测量：在线回报、延迟事件预测、记忆消融后的行为变化、状态漂移、敏感度误差与峰值内存。

什么结果会反驳该解释：匹配历史窗口后收益消失，或能预测关键事件却不改善决策，就不能归因于更好的控制状态。

#### 教材与研究者

[智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/) · [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/) · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

[Martha White](https://yingwen.io/zh/continual-rl/resource/S03/) · [Adam White](https://yingwen.io/zh/continual-rl/resource/S04/) · [Esraa Elelimy](https://yingwen.io/zh/continual-rl/resource/S34/) · [Matthew Schlegel](https://yingwen.io/zh/continual-rl/resource/S35/) · [Jan Peters](https://yingwen.io/zh/continual-rl/resource/S72/)

#### 实验平台

- [解析小 MDP 与短序列](https://yingwen.io/zh/continual-rl/labs/)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [POPGym](https://github.com/proroklab/popgym)：可控记忆难度的部分可观测任务集合。 记忆长度、噪声与序列信用。 边界：通常有 episode；记忆基准成绩不能直接证明单生命期适应。
- [Forager](https://github.com/andnp/forager)：部分可观测的持续资源获取。 信息历史、状态容量、更新成本与长期行为。 边界：感知模式和地图差异会改变问题；不只比较网络名字。

#### 原文与代码

- [Elelimy et al. · Real-Time Recurrent Learning using Trace Units](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e616bde0438cb10cb6adf076ae7d336-Paper-Conference.pdf)（NeurIPS 2024）：通过结构化递归单元降低前向敏感度维护成本，使逐步递归学习可以直接与控制任务结合。
  [作者 RTU 实现](https://github.com/esraaelelimy/rtus)：linear_rtus.py 的状态与敏感度 carry；复杂度优势依赖单元结构。
- [Schlegel et al. · General Value Function Networks](https://arxiv.org/abs/1807.06763)（GVFN 原论文）：让递归状态的坐标对应预测问题，以预测约束帮助状态学习。问题设计和递归求导是两个独立部分。
  [作者 Julia 实验](https://github.com/mkschleg/GVFN)：分别阅读预测问题定义、递归状态和训练更新；不是通用任意 GVF 集合的充分性证明。
- [Farr et al. · Streaming RL under Partial Observability with RTRL](https://arxiv.org/abs/2605.24709)（2026 预印本）：将实时递归学习用于部分可观测流式控制。网络记忆、信用传播与流式优化需要共同评价。
  此处未提供已确认的该论文作者代码；RTU 仓库对应上面的另一篇论文。
- [Forager: a Lightweight Testbed for Continual Learning with Partial Observability in RL](https://arxiv.org/abs/2605.01131)（2026 预印本）：在长期交互中隔离观测与状态构造问题，研究局部感知下的资源获取。
  [作者 Forager 环境](https://github.com/andnp/forager)：配置 observation_mode 和地图；start()/step() 接口不同于 Gymnasium 五元组。

---

### 03 · 哪些预测值得学习，谁来使用这些预测？

问题设定：同一数据流可以支持许多不同未来问题。计算和容量有限，行为策略也未必覆盖每个目标策略。核心研究对象不仅是一个预测器的误差，还包括问题的选择、淘汰与下游用途。

已有认识：一个 GVF 由累积信号 $c$、延续条件 $\gamma$ 和目标策略 $\pi$ 指定。TD、GTD 与 emphatic 方法是回答该问题的算法。GVF 的定义不要求其参与状态，也不要求其信号就是任务奖励。

尚缺什么：预测学得准，并不表示它能改善主任务。辅助目标可能利用额外数据，也可能与主目标争夺表示容量。发现有用问题还要平衡可学习性、未来复用和当前计算成本。

#### 方法与条件

- **Horde / off-policy prediction**：为多个问题维护各自的目标与权重，把一条 transition 共享给多个学习器。重要性比率修正行为与目标策略差异；梯度 TD 或 emphatic weighting 处理特定离策略稳定性问题。
  条件与代价：行为策略必须覆盖目标策略，方差和表示误差仍可能很大。线性稳定性不等于深度预测任意稳定。
- **辅助预测与 GVFN**：用观测重构、潜在自预测或指定未来信号提供训练目标。GVFN 进一步把预测纳入递归状态。两者都需要独立检验预测是否被决策使用。
  条件与代价：RLC 2024 的分析表明目标用途依赖干扰和与 TD 的组合，不能把一个辅助 loss 的下降当作通用知识增益。
- **学习预测语义与更新目标**：DiscoRL 不预先给全部预测指定 GVF 语义，而是让元网络产生学习目标，再通过 agent 回报选择规则。这把“预测什么”与“怎样更新”一并纳入搜索。
  条件与代价：语义自由度增加时，可解释性、元训练成本及未见分布的可检验性也成为研究问题。

#### 相互竞争的解释

- 收益来自预测所携带的知识，还是来自额外梯度产生的正则化？
- 辅助任务有效，是因为其预测目标有因果用途，还是因为它更容易优化且共享了更多参数？

#### 可执行实验

假设：与未来控制相关的预测，比同规模无关辅助目标带来更低的奖励变化后再学习成本。

设计：在已知小 MDP 中先给每个 GVF 求解析答案；固定行为数据学习两组问题，再仅改变奖励或下游目标。通过删除、打乱或冻结预测输入检查用途。

对照：无辅助目标、随机目标、等计算的额外主任务更新；相同数据覆盖和表示维度。

测量：预测校准、目标策略覆盖、主任务在线回报、奖励变化后的再学习样本与下游消融差值。

什么结果会反驳该解释：随机辅助目标同样有效，或删除预测输入不改变行为，就不能把收益解释为所学预测知识被利用。

#### 教材与研究者

[价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/) · [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/) · [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/) · [元学习与学习规则的适应](https://yingwen.io/zh/continual-rl/algorithms/meta/)

[Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/) · [Martha White](https://yingwen.io/zh/continual-rl/resource/S03/) · [Adam White](https://yingwen.io/zh/continual-rl/resource/S04/) · [Patrick M. Pilarski](https://yingwen.io/zh/continual-rl/resource/S18/) · [Joseph Modayil](https://yingwen.io/zh/continual-rl/resource/S32/) · [Matthew Schlegel](https://yingwen.io/zh/continual-rl/resource/S35/)

#### 实验平台

- [解析小 MDP 与短序列](https://yingwen.io/zh/continual-rl/labs/)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [Forager](https://github.com/andnp/forager)：部分可观测的持续资源获取。 信息历史、状态容量、更新成本与长期行为。 边界：感知模式和地图差异会改变问题；不只比较网络名字。

#### 原文与代码

- [Sutton et al. · Horde: A Scalable Real-time Architecture for Learning Knowledge from Unsupervised Sensorimotor Interaction](https://sites.ualberta.ca/~amw8/horde.pdf)（AAMAS 2011）：同一经验服务多个由累积量、延续条件和目标策略定义的预测；学习问题与回答问题的更新器分离。
  [原研究系统 RLPark](https://github.com/rlpark/rlpark)：Horde.java 调度多个 demon，GTDLambda.java 实现离策略预测；旧 Java/机器人栈。
- [Voelcker et al. · When does Self-Prediction Help?](https://rlj.cs.umass.edu/2024/papers/Paper197.html)（RLC 2024）：比较自预测目标及其与价值学习的组合，说明辅助预测的用途依赖表示、数据和主任务。
  [作者仓库（仅论文说明）](https://github.com/adaptive-agents-lab/understanding_auxiliary_tasks)：公开 main 分支只有 README 与 .gitignore，没有可运行的实验源码。
- [Schlegel et al. · General Value Function Networks](https://arxiv.org/abs/1807.06763)（GVFN 原论文）：让递归状态的坐标对应预测问题，以预测约束帮助状态学习。问题设计和递归求导是两个独立部分。
  [作者 Julia 实验](https://github.com/mkschleg/GVFN)：分别阅读预测问题定义、递归状态和训练更新；不是通用任意 GVF 集合的充分性证明。
- [Oh et al. · Discovering State-of-the-Art RL Algorithms](https://www.nature.com/articles/s41586-025-09761-x)（Nature 2025）：通过元网络产生策略与预测的学习目标，从多智能体经验中搜索更新规则；学习出的预测未必具有预先指定的 GVF 语义。
  [作者 DiscoRL 工程](https://github.com/google-deepmind/disco_rl)：区分已发布规则的使用、训练 agent 与重新发现规则；三者需要不同预算。

## Ⅲ · 控制与学习机制

行为、信用与学习规则怎样持续更新？

### 04 · 旧价值什么时候应该复用，什么时候应该快速改写？

问题设定：策略改善不断改变数据分布。奖励或动力学还可能发生外部变化。一个旧价值函数既可能提供知识，也可能阻碍适应；仅靠最终价值误差不能判断控制失败的原因。

已有认识：Q-learning、actor–critic 和最大熵控制规定不同的策略改善步骤。CRL 并不取代这些控制骨干，而是要求其在持续变化的数据和有限经验预算下继续有效。环境变化速度与估计速度的关系比算法名称更重要。

尚缺什么：“快忘记”和“保留长期规律”存在冲突。显式任务边界下的重置或压缩不一定能用于边界未知的情形；环境不变时，由策略改善产生的数据漂移也会触发类似现象。

#### 方法与条件

- **Permanent–Transient 价值分解**：用 $V=V^{\mathrm{permanent}}+V^{\mathrm{transient}}$ 表示长期知识与短期残差，分别安排更新和知识整合。变化出现时不必把全部价值重新学一遍。
  条件与代价：原文区分有边界和持续实验。两个时间尺度增加超参数，也可能把过时知识长期保留下来。
- **滑窗估计 / 乐观探索 / 变化预算**：滑窗减少过时转移的影响；置信集加宽与乐观规划避免窗口过短导致探索停止。理论将变化量与可达到的动态 regret 联系起来。
  条件与代价：需要明确有限状态、可观察信息和变化预算；不是对任意非平稳深度 actor–critic 的保证。
- **重置在线学习器并蒸馏**：Reset & Distill 将新任务的快速学习与过去行为保存放到不同网络和训练阶段，直接处理旧初始化带来的负迁移。
  条件与代价：依赖任务化协议、重置和额外蒸馏成本；不能直接移植为单一无边界 agent 而保持原结论。

#### 相互竞争的解释

- 快速适应可能来自减少陈旧价值，也可能来自探索策略重新增加覆盖。
- 从头训练优于接着训练，可能是表示塑性下降，也可能是旧策略把 agent 困在错误数据分布中。

#### 可执行实验

假设：在反复出现的奖励模式中，双时间尺度价值能够兼顾首次适应和回访复用。

设计：设置 A→B→A 奖励变化与独立的动力学变化两条序列。先固定 replay 数据比较估计，再放开在线行为比较控制；逐渐移除任务边界提示。

对照：单价值固定步长、步长网格、边界重置、随机探索重启；参数量和总更新次数匹配。

测量：首次适应、回访收益、旧价值误差、状态覆盖、在线累计奖励及负迁移。

什么结果会反驳该解释：固定数据后差异消失且只在重新探索时有效，就应将解释归于数据获取而非价值记忆机制。

#### 教材与研究者

[持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/) · [深度价值学习](https://yingwen.io/zh/continual-rl/algorithms/deep-value/) · [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/) · [最大熵控制](https://yingwen.io/zh/continual-rl/algorithms/soft-control/) · [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)

[Doina Precup](https://yingwen.io/zh/continual-rl/resource/S07/) · [Csaba Szepesvári](https://yingwen.io/zh/continual-rl/resource/S15/) · [Hado van Hasselt](https://yingwen.io/zh/continual-rl/resource/S61/) · [Wang Chi Cheung](https://yingwen.io/zh/continual-rl/resource/S98/)

#### 实验平台

- [解析小 MDP 与短序列](https://yingwen.io/zh/continual-rl/labs/)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [MinAtar](https://github.com/kenjyoung/MinAtar)：低成本、简化视觉的 Atari 风格环境。 深度控制、函数变化和多 seed 消融。 边界：任务切换须另行规定；切换版本与普通 MinAtar 不是同一协议。
- [Continual World](https://github.com/awarelab/continual_world)：基于 Meta-World 的机器人任务序列。 遗忘、前向迁移及回访表现。 边界：任务边界、重置和任务标识应明确；不是严格无重置协议。

#### 原文与代码

- [Anand & Precup · Prediction and Control in Continual RL](https://arxiv.org/abs/2312.11669)（NeurIPS 2023）：把价值拆为长期 permanent 与快速 transient 分量，研究长期积累和短期适应的兼容性。
  [作者预测与控制实验](https://github.com/NishanthVAnand/prediction-and-control-in-continual-reinforcement-learning)：prediction_semi_crl 与 control 分开；有边界和持续协议要分别运行。
- [Cheung et al. · Reinforcement Learning for Non-Stationary MDPs](https://proceedings.mlr.press/v119/cheung20a.html)（ICML 2020）：用滑动窗口、置信集加宽及变化预算处理探索与历史失效；理论对象是规定条件下的非平稳 MDP。
  以论文算法和定理条件为实现依据；此处没有可确认的作者代码入口。
- [Ahn et al. · Prevalence of Negative Transfer in Continual RL](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ba9e3d60610f3525717665966d86e0cd-Abstract-Conference.html)（ICLR 2025）：Reset & Distill 在新任务中重置在线学习器，并离线蒸馏在线策略与过去 expert，研究负迁移和知识保留。
  [作者 Reset & Distill](https://github.com/hongjoon0805/Reset-Distill)：任务边界、网络重置、离线蒸馏与额外 expert 存储均应纳入实验协议。

---

### 05 · 远处的反馈应该怎样更新早先的决策与内部计算？

问题设定：关键线索或动作很早出现，奖励很晚到达。时间信用涉及不同时间步；结构信用涉及同一计算图中的参数与中间量。是否只能在线更新，是另一个资源条件。

已有认识：冻结参数时，$\lambda$-return 的总增量可改写为 TD 误差乘资格迹的后向更新。传统在线 TD($\lambda$) 在每步改变参数后不再精确保持该恒等式。True-online TD 重新定义在线前向目标并加入荷兰迹与修正。

尚缺什么：深度控制需要同时处理非线性表示、离策略数据、递归状态和权重变化。把一条 trace 加到 optimizer 中不足以证明它解决了长时域信用，也不能将 RTRL 敏感度直接当作价值资格迹。

#### 方法与条件

- **n-step / λ-return / true-online TD**：多步目标在不同 bootstrap 长度之间混合。后向资格迹压缩过去预测的梯度方向；true-online 的额外预测修正使指定线性算法对应逐前缀前向参考。
  条件与代价：精确等价针对定义好的线性、固定特征、步长等条件。更大 λ 会延长信用，也会增加误差与方差传播。
- **控制中的资格迹**：SARSA 的 bootstrap 使用实际选择的下一动作；Watkins Q(λ) 使用 max target，并在下一行为非贪心时剪断后续迹。剪迹发生在当前反馈完成信用分配之后。
  条件与代价：两种方法目标不同；不能把任意 off-policy actor–critic 都用同一个剪迹规则替换。
- **RTRL 敏感度 × 价值/策略时间信用**：RTRL 追踪参数对当前内部状态的影响；外层 TD 或策略更新还需要把延迟反馈与预测/动作联系起来。结构化递归研究使这两个层次可以在有限预算下组合。
  条件与代价：在线参数变化使全历史敏感度产生陈旧误差；截断和低秩近似又带来不同偏差或方差。

#### 相互竞争的解释

- 提高 λ 后变好，可能只是有效步幅变大，而不是利用了更久以前的信息。
- 更长反传窗口有效，可能来自更多重复训练，而非更准确的历史因果路径。

#### 可执行实验

假设：在匹配参数更新尺度后，正确的历史信用仍能改善稀疏延迟反馈学习。

设计：先运行教材独立在线前向 oracle；再控制线索到奖励的延迟和干扰数量。分别改变 λ、BPTT 窗口与状态递归，保持其余部分固定。

对照：TD(0)、传统 TD(λ)、true-online 线性参考；递归实验设置 detach 但不清空活动的对照。

测量：逐前缀权重误差、有限差分梯度误差、回报随延迟曲线、实际更新范数及每步成本。

什么结果会反驳该解释：收益在匹配更新范数或训练次数后消失，就不能将其归因于更准确的长时间信用。

#### 教材与研究者

[时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/) · [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/) · [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/) · [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)

[Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/) · [A. Rupam Mahmood](https://yingwen.io/zh/continual-rl/resource/S05/) · [Khurram Javed](https://yingwen.io/zh/continual-rl/resource/S33/) · [Esraa Elelimy](https://yingwen.io/zh/continual-rl/resource/S34/) · [Anna Harutyunyan](https://yingwen.io/zh/continual-rl/resource/S66/)

#### 实验平台

- [解析小 MDP 与短序列](https://yingwen.io/zh/continual-rl/labs/)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [POPGym](https://github.com/proroklab/popgym)：可控记忆难度的部分可观测任务集合。 记忆长度、噪声与序列信用。 边界：通常有 episode；记忆基准成绩不能直接证明单生命期适应。

#### 原文与代码

- [van Seijen et al. · True Online Temporal-Difference Learning](https://www.jmlr.org/papers/v17/15-599.html)（JMLR 2016）：荷兰迹与预测修正实现线性在线前向视图；这不同于冻结参数下的传统前后向总增量恒等式。
  [作者原始 true-online 实现](https://github.com/armahmood/totd-rndmdp-experiments/blob/4b6d459890a0e9da898f6cf8b849ba7870ea044a/pysrc/algorithms/tdprediction/onpolicy/totd.py)：此源码将步长包含在 trace 中；与教材未缩放 trace 对照时需换算。
- [Sutton & Barto · Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)（第二版，2018）：第 7、11、12 章连接多步目标、离策略学习和资格迹；用于核对预测与控制算法的不同条件。
  作者教材和伪代码入口；本网站实验为独立教学实现。
- [Elelimy et al. · Real-Time Recurrent Learning using Trace Units](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e616bde0438cb10cb6adf076ae7d336-Paper-Conference.pdf)（NeurIPS 2024）：通过结构化递归单元降低前向敏感度维护成本，使逐步递归学习可以直接与控制任务结合。
  [作者 RTU 实现](https://github.com/esraaelelimy/rtus)：linear_rtus.py 的状态与敏感度 carry；复杂度优势依赖单元结构。
- [Farr et al. · Streaming RL under Partial Observability with RTRL](https://arxiv.org/abs/2605.24709)（2026 预印本）：将实时递归学习用于部分可观测流式控制。网络记忆、信用传播与流式优化需要共同评价。
  此处未提供已确认的该论文作者代码；RTU 仓库对应上面的另一篇论文。

---

### 06 · 不存 replay、每步只处理新经验时，怎样避免更新失稳？

问题设定：数据到达后立即用于学习；过去 transition 不被存储再训练。资格迹、递归活动和 optimizer 状态仍可存在，但须计入内存。数值稳定、样本利用率和动作 deadline 都是评价对象。

已有认识：逐步更新时，样本间的相关性和瞬时尺度不会被 batch 平均掉。归一化、网络尺度、资格迹与 actor–critic 时序会共同影响稳定性。“batch size=1”只是数据粒度，不是一套完整算法。

尚缺什么：稳定参数步幅不等于稳定输出，更不等于稳定长期控制。不同网络、奖励尺度和罕见大误差下的适用条件仍需检验；单步更新便宜也可能因为需要更多真实经验而整体昂贵。

#### 方法与条件

- **Stream-X：共同更新配方**：信号归一化与表示稳定化配合受控参数更新。2024 ObGD 以全局尺度限制步幅；2026 实现维护 $v_i=\max(\beta v_i,|\delta e_i|)$，并按坐标缩放更新。两者不是同一个 optimizer。
  条件与代价：逐坐标上界限制参数变化，不保证 TD 误差绝不越过零；版本、网络和归一化都需匹配。
- **Intentional TD / Policy Gradient**：在局部近似下先指定希望减少多少 TD 误差，或允许多大策略 KL 变化，再求更新尺度。eligibility 与对角缩放改变方向，输出约束决定其长度。
  条件与代价：近似依赖局部函数几何，真实输出改变可能偏离目标；须记录实际 TD-error 与 KL，而不能只看参数范数。

#### 相互竞争的解释

- 方法提高回报，是归一化减少了数值异常，还是输出尺度控制更有效？
- 吞吐量变高，是更新更便宜，还是省略了基线需要完成的学习工作？

#### 可执行实验

假设：相较固定参数步长，输出变化控制能在奖励缩放后维持更接近的学习行为。

设计：在同一 Stream-AC 骨干上改变奖励比例、干扰特征和少量异常奖励；一次替换一个尺度机制，再测试组合。

对照：匹配架构、trace、初始化、归一化、调参预算和每秒环境步；旧版 ObGD 与新版 StreamingOptimizer 分列。

测量：实际输出变化、TD 误差越过零比例、策略 KL、崩溃率、峰值内存、动作 deadline 违约率和在线回报。

什么结果会反驳该解释：在匹配归一化与有效步幅后不再改善，就不支持输出约束是主要收益来源的解释。

#### 教材与研究者

[流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/) · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/) · [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/) · [最大熵控制](https://yingwen.io/zh/continual-rl/algorithms/soft-control/)

[A. Rupam Mahmood](https://yingwen.io/zh/continual-rl/resource/S05/) · [Mohamed Elsayed](https://yingwen.io/zh/continual-rl/resource/S107/) · [Gautham Vasan](https://yingwen.io/zh/continual-rl/resource/S108/) · [Khurram Javed](https://yingwen.io/zh/continual-rl/resource/S33/)

#### 实验平台

- [解析小 MDP 与短序列](https://yingwen.io/zh/continual-rl/labs/)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [MinAtar](https://github.com/kenjyoung/MinAtar)：低成本、简化视觉的 Atari 风格环境。 深度控制、函数变化和多 seed 消融。 边界：任务切换须另行规定；切换版本与普通 MinAtar 不是同一协议。
- [DeepRL-continuing-tasks](https://github.com/facebookresearch/DeepRL-continuing-tasks)：无重置、预设重置、智能体控制重置的连续控制。 奖励偏移、回报目标、重置成本对方法排序的影响。 边界：continuing 不自动包含任务漂移或无界学习需要。

#### 原文与代码

- [Elsayed et al. · Streaming Deep Reinforcement Learning Finally Works](https://arxiv.org/abs/2410.14606)（2024 初稿；2026 修订）：将信号归一化、表示稳定化和受控参数更新组成 Stream-X。2024 与 2026 的优化器不同。
  [作者 Stream-X 实现](https://github.com/mohmdelsayed/streaming-drl)：2024 分支为 ObGD；2026 分支使用逐坐标 StreamingOptimizer，并包含 Stream-RAC。必须与所用论文版本对应。
- [Sharifnassab et al. · Intentional Updates for Streaming RL](https://proceedings.mlr.press/v306/sharifnassab26a.html)（ICML 2026）：先规定 TD 误差或策略输出的期望变化，再近似求合适步幅；输出约束不同于参数范数约束。
  [作者 Intentional RL](https://github.com/sharifnassab/Intentional_RL)：optimizer.py 中的方向缩放、输出变化近似、迹与截断共同决定更新。

---

### 07 · 哪些学习参数应当适应，怎样评价学出来的更新规则？

问题设定：权重在内层学习，步长、return 参数、初始化或整个更新器在外层适应。在线元学习使用一段持续经验；跨任务元 RL 使用任务分布及重置。两者的元目标和数据权限不同。

已有认识：IDBD 用步长敏感度选择特征的学习尺度；TIDBD 把该思路带到 bootstrap 预测。Metatrace 将元敏感度与 actor–critic 的时间信用结合。Meta-gradient RL 更一般地对学习过程求导，学习 return 等参数。

尚缺什么：短展开元梯度可能优化短期可见效果，而不是长期适应。学出的规则可能依赖元训练时域、任务分布和网络尺度；有限任务上的泛化不能保证无限生命期收益。

#### 方法与条件

- **逐特征步长 → Metatrace**：维护 $\partial w_t/\partial\beta$ 等敏感度，用后续误差判断过去步长是否有益。Metatrace 再累积元资格，使延迟反馈能够影响步长参数。
  条件与代价：监督误差、TD semi-gradient 和完整 TD-error 导数不是同一目标。归一化与截断也改变算法。
- **Meta-gradient RL**：内层更新生成新权重，外层用固定评价问题衡量这些权重，并沿更新过程反传到元参数。内层 discount 可学习，不代表外层评价目标也应随它任意改变。
  条件与代价：完整敏感度含跨步与跨参数耦合；截断、停止梯度和随机轨迹导数均需说明。
- **规则搜索与 DiscoRL**：将损失、目标或更新规则参数化，通过元训练评价其学习效果。RLC 2025 比较搜索途径与成本；DiscoRL 提供表达更丰富的目标生成规则及公开元训练/元评价入口。
  条件与代价：发现规则的总算力与应用已发布规则的成本分开；发布实验不等于每个研究者都能同预算复现原发现规模。

#### 相互竞争的解释

- “学会学习”的收益可能只是更大的超参数搜索预算，或更适合某个评价时域的手工偏置。
- 元网络依赖的是任务结构，还是 reward scale、episode 长度等容易泄漏的任务标记？

#### 可执行实验

假设：学习尺度或更新规则能够在未见变化速度上迁移，而非只记住开发任务的时域。

设计：先用标量与向量问题做有限差分；随后在信号相关性、噪声和漂移速度的网格中划分元训练/测试区域，测试更长生命期。

对照：相同搜索总预算的固定步长、Adam/RMSProp、简单尺度归一化；冻结元更新、重置元状态两项消融。

测量：在线误差与回报、元梯度误差、元训练总成本、长时域失稳、未见分布性能和部署更新成本。

什么结果会反驳该解释：增益仅存在于元训练时域内，或基线获得同等调参预算后差距消失，就不足以支持通用学习规则改进。

#### 教材与研究者

[元学习与学习规则的适应](https://yingwen.io/zh/continual-rl/algorithms/meta/) · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/) · [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/) · [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

[Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/) · [Matthew E. Taylor](https://yingwen.io/zh/continual-rl/resource/S19/) · [Jakob Foerster](https://yingwen.io/zh/continual-rl/resource/S58/) · [Shimon Whiteson](https://yingwen.io/zh/continual-rl/resource/S59/) · [Hado van Hasselt](https://yingwen.io/zh/continual-rl/resource/S61/) · [Luisa Zintgraf](https://yingwen.io/zh/continual-rl/resource/S69/)

#### 实验平台

- [解析小 MDP 与短序列](https://yingwen.io/zh/continual-rl/labs/)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [MinAtar](https://github.com/kenjyoung/MinAtar)：低成本、简化视觉的 Atari 风格环境。 深度控制、函数变化和多 seed 消融。 边界：任务切换须另行规定；切换版本与普通 MinAtar 不是同一协议。

#### 原文与代码

- [Sutton · Adapting Bias by Gradient Descent](https://cdn.aaai.org/AAAI/1992/AAAI92-027.pdf)（AAAI 1992）：IDBD 对每个特征的对数步长维护敏感度，使学习尺度由后续预测效果调整。
  原文包含算法；这里没有 1992 年作者原始代码发布入口。
- [Kearney et al. · TIDBD: Adapting Step-sizes Through Stochastic Meta-descent for Temporal-Difference Learning](https://arxiv.org/abs/1804.03334)（2018 论文）：将逐特征步长适应从监督预测扩展到 TD，并处理 bootstrap 误差与资格迹；不同梯度和归一化版本须逐式区分。
  原文 Algorithm 1 与教材线性实现可以对应；此处没有可确认的完整作者原始实验仓库。
- [Young, Wang & Taylor · Metatrace Actor-Critic](https://www.ijcai.org/proceedings/2019/0581.pdf)（IJCAI 2019）：把步长敏感度和时间信用结合；scalar、vector、mixed 形式具有不同的适应粒度和归一化。
  未找到可确认的作者原始代码；元学习教材提供独立线性特例，不含原文全部控制实验。
- [Xu, van Hasselt & Silver · Meta-Gradient Reinforcement Learning](https://arxiv.org/abs/1805.09801)（NeurIPS 2018）：对内层学习更新求导，以后续评价调整 return 等学习参数；元目标和内层目标必须分开。
  以论文伪代码和教材敏感度实验为入口；此处没有可确认的作者原始训练工程。
- [Goldie et al. · How Should We Meta-Learn RL Algorithms?](https://rlj.cs.umass.edu/2025/papers/Paper218.html)（RLC 2025）：在不同算法对象上比较元优化途径，并比较泛化、可解释性、样本开销与训练成本。
  [作者算法搜索实验](https://github.com/AlexGoldie/learn-rl-algorithms)：按元训练任务与元测试任务恢复配置，不能只计发现后规则的部署成本。
- [Oh et al. · Discovering State-of-the-Art RL Algorithms](https://www.nature.com/articles/s41586-025-09761-x)（Nature 2025）：通过元网络产生策略与预测的学习目标，从多智能体经验中搜索更新规则；学习出的预测未必具有预先指定的 GVF 语义。
  [作者 DiscoRL 工程](https://github.com/google-deepmind/disco_rl)：区分已发布规则的使用、训练 agent 与重新发现规则；三者需要不同预算。

## Ⅳ · 子任务与规划

怎样构造可复用行为，并用其后果做决策？

### 08 · 什么样的时间抽象值得加入技能库？

问题设定：Option 具有启动条件、内部策略与终止规则。技能发现还要决定哪些 option 值得占用容量。探索覆盖、执行效率、信用压缩和规划价值是不同用途，不能仅凭技能数量或轨迹图判断。

已有认识：Machado 的谱发现路线把环境连通结构变成内在奖励，再变成长行为。现代技能方法可从可区分性、后果模型或时间距离组织行为。它们改变的目标不同，产生的技能也未必适合相同任务。

尚缺什么：技能发现的计算和交互成本往往在下游成功率之外。环境或表示改变后，旧技能的启动、终止与后果模型都可能失效。持续技能库还缺少可靠的保留、合并、替换和重估准则。

#### 方法与条件

- **Eigenoptions → ALLO**：谱方向描述由经验转移诱导的连通结构；沿特征坐标变化构造奖励，学习长距离移动。ALLO 更准确地区分谱方向与特征值，给技能和后续规划提供结构化坐标。
  条件与代价：图取决于采样行为与状态表示。慢变化方向不必然是当前奖励最有用的方向。
- **METRA 的度量感知技能**：使沿技能方向的表示变化成为内在奖励，并限制相邻状态的表示距离。约束使表示更接近时间可达性结构，而非任意放大尺度获得高奖励。
  条件与代价：原目标和源码中的按维度缩放、slack 与 dual 更新需对应。状态覆盖好仍不是外部任务有效性的充分条件。
- **奖励感知谱结构**：Default Representation 把默认动力学与奖励结构共同纳入后果表示，进一步影响谱特征、塑形和 option 发现。它提供与纯动力学 SR 不同的技能偏置。
  条件与代价：奖励改变会改变表示本身的有效性；奖励感知不等于对所有未来任务更可迁移。

#### 相互竞争的解释

- 技能收益来自更好的探索覆盖，还是来自更少的高层决策与更有效的规划？
- 谱技能看似更好，可能只是执行时长或可到达范围比随机技能更大。

#### 可执行实验

假设：结构化技能在计入发现成本后，仍能降低跨房间目标变化的适应成本。

设计：在同一网格世界采样固定经验，比较随机、谱和奖励感知技能。先冻结低层技能只改变高层目标，再改变通道位置检验技能失效。

对照：primitive-only、时长与终点覆盖匹配的随机技能；同技能数、总数据、决策预算；有无模型规划分别报告。

测量：技能发现总成本、启动成功率、时长分布、主任务累计奖励、模型误差和通道改变后的恢复。

什么结果会反驳该解释：等时长随机技能或等覆盖原始探索得到相同收益，则不足以证明谱结构本身带来可复用抽象。

#### 教材与研究者

[Options 与技能发现](https://yingwen.io/zh/continual-rl/construction/options/) · [目标条件化与子任务构造](https://yingwen.io/zh/continual-rl/construction/goals/) · [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/) · [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)

[Marlos C. Machado](https://yingwen.io/zh/continual-rl/resource/S06/) · [Doina Precup](https://yingwen.io/zh/continual-rl/resource/S07/) · [George Konidaris](https://yingwen.io/zh/continual-rl/resource/S09/) · [Pierre-Luc Bacon](https://yingwen.io/zh/continual-rl/resource/S37/) · [André Barreto](https://yingwen.io/zh/continual-rl/resource/S40/) · [Benjamin Eysenbach](https://yingwen.io/zh/continual-rl/resource/S53/)

#### 实验平台

- [解析小 MDP 与短序列](https://yingwen.io/zh/continual-rl/labs/)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [OGBench](https://github.com/seohongpark/ogbench)：离线目标条件强化学习的环境、数据与基线。 长时域、轨迹拼接、观测与数据覆盖。 边界：离线成功率不能直接说明在线技能发现或持续控制；规划时间另计。

#### 原文与代码

- [Machado et al. · A Laplacian Framework for Option Discovery in RL](https://arxiv.org/abs/1703.00956)（ICML 2017）：用状态转移图的谱方向构造内在奖励，并学习沿这些方向移动的时间扩展行为。
  [Machado 作者原始 options 工程](https://github.com/mcmachado/options)：main.py、谱表示和 option 学习相连；环境封装和旧依赖属于原实验条件。
- [Gomez, Bowling & Machado · Proper Laplacian Representation Learning](https://arxiv.org/abs/2310.10833)（ICLR 2024）：ALLO 的约束与对偶动态恢复有序谱方向及相应特征值，减少只学到任意旋转子空间的问题。
  [作者 ALLO 实现](https://github.com/tarod13/laplacian_dual_dynamics)：同时读原始变量、对偶变量和正交约束；只保留相邻状态平滑项不是完整算法。
- [Park, Rybkin & Levine · METRA](https://arxiv.org/abs/2310.08887)（ICLR 2024）：在转移上的表示距离约束下，最大化技能方向与状态变化的对齐，再以该信号学习技能策略。
  [作者 METRA 实现](https://github.com/seohongpark/METRA)：iod/metra.py 的内在奖励、距离约束、对偶更新和 SAC 技能训练需一并对照。
- [Tse, Chandrasekar & Machado · Reward-Aware Proto-Representations in RL](https://arxiv.org/abs/2505.16217)（NeurIPS 2025）：分析 Default Representation 的动态规划、TD 学习与特征形式，把奖励结构引入后果表示。
  [作者 DR 实验](https://github.com/httse9/Reward-Aware-Proto-Representations)：rep_utils.py 区分 SR/DR；ROD_DR.py 连接采样学习、谱分解与发现。

---

### 09 · 子目标从哪里来，为什么学习它能帮助总体任务？

问题设定：总体任务奖励已经给定。智能体还可以选择中间状态、预测目标、技能描述或训练课程。定义目标、学习目标条件策略、选择当前目标是三层计算，不应都称为“目标发现”。

已有认识：UVFA 提供目标条件价值接口；HER 从失败轨迹重标记可达目标；HIQL 将长期任务分解为高层目标和低层动作。它们主要解决已给目标下的学习和组合，不能单独回答哪些新目标值得创造。

尚缺什么：子目标既要可学习，又要对总体任务有用。过易目标浪费经验，过难目标没有信号；语言先验能提供候选目标，却也引入外部知识和设计权限。持续生成目标还需要检测冗余、陈旧和奖励投机。

#### 方法与条件

- **HER：增加已达到目标的学习信号**：从同一轨迹选取 achieved goal，重算目标相关奖励，使原本失败的轨迹成为另一个控制问题的训练样本。它修改 replay 样本的目标，不改变真实执行结果。
  条件与代价：需要可重算目标奖励和适合离策略学习的数据；目标相关终止与随机性可能影响重标记语义。
- **HIQL：分层提取目标条件策略**：先学习目标价值，再分别利用一步和跨步优势拟合低层动作与高层中间目标。优势作为停止梯度的权重，避免策略更新反向扭曲其评价标尺。
  条件与代价：离线数据必须支持相关状态与目标；更短有效时域不能解决完全没有覆盖的行为。
- **Reward-respecting 与 AI-feedback 技能设计**：STOMP 将子任务停止条件与原奖励联系起来；MaestroMotif 用语言描述、偏好奖励和规则组织技能。前者强调任务回报接口，后者引入可扩展的外部先验。
  条件与代价：子任务奖励与环境奖励应分开记录；人工或语言模型的目标、反馈、程序生成成本属于方法。

#### 相互竞争的解释

- 目标生成有效，可能是更准确地选择学习难度，也可能只是引入了额外任务信息。
- 层级方法提升，可能源于降低价值估计噪声，而非发现了可迁移语义。

#### 可执行实验

假设：按下游用途选择的子目标，比只按成功率或可达性选择的目标更节省总经验。

设计：固定低层目标条件学习器，分别使用均匀目标、适中难度目标和基于主任务改进的目标；在 OGBench 先分析数据支持，再进入在线小迷宫。

对照：相同目标数、低层数据和高层更新次数；人工/语言先验单独设有无对照。

测量：目标成功率、总体回报、目标覆盖、目标生成成本、失败尝试和未见总体任务表现。

什么结果会反驳该解释：目标完成得更多却没有改善总体回报，或收益完全来自额外先验，则不支持自主目标构建的解释。

#### 教材与研究者

[目标条件化与子任务构造](https://yingwen.io/zh/continual-rl/construction/goals/) · [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/) · [Options 与技能发现](https://yingwen.io/zh/continual-rl/construction/options/)

[Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/) · [Marlos C. Machado](https://yingwen.io/zh/continual-rl/resource/S06/) · [Peter Stone](https://yingwen.io/zh/continual-rl/resource/S10/) · [Sergey Levine](https://yingwen.io/zh/continual-rl/resource/S49/) · [Benjamin Eysenbach](https://yingwen.io/zh/continual-rl/resource/S53/) · [Tom Schaul](https://yingwen.io/zh/continual-rl/resource/S63/)

#### 实验平台

- [解析小 MDP 与短序列](https://yingwen.io/zh/continual-rl/labs/)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [OGBench](https://github.com/seohongpark/ogbench)：离线目标条件强化学习的环境、数据与基线。 长时域、轨迹拼接、观测与数据覆盖。 边界：离线成功率不能直接说明在线技能发现或持续控制；规划时间另计。
- [TeachMyAgent](https://github.com/flowersteam/TeachMyAgent)：teacher 可以选择环境参数的自动课程实验。 学习进展估计和经验分配。 边界：必须计算 teacher 权限、失败尝试和环境重置，不能视作完全自主单生命期。

#### 原文与代码

- [Andrychowicz et al. · Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)（NeurIPS 2017）：把轨迹实际达到的目标重标记为训练目标，以稀疏反馈学习目标条件控制。
  [作者团队 HER 实现](https://github.com/openai/baselines/tree/master/baselines/her)：her_sampler.py 中的目标替换与 reward_fun；结束标志与目标依赖需按任务处理。
- [Park et al. · HIQL: Offline Goal-Conditioned RL with Latent States as Actions](https://arxiv.org/abs/2307.11949)（NeurIPS 2023）：用目标价值分别提取高层子目标策略和低层动作策略，缩短每层面对的有效时域。
  [作者 HIQL 实现](https://github.com/seohongpark/HIQL)：hiql.py 的 expectile 价值学习、停止梯度优势权重和 high/low actor 更新。
- [Sutton et al. · Reward-Respecting Subtasks](https://arxiv.org/abs/2202.03466)（Artificial Intelligence 2023）：STOMP 连接子任务、option、真实后果模型与规划；子任务停止奖励和环境真实回报属于不同接口。
  原文提供更新规则和实验；此处没有可确认的完整作者 STOMP 工程。
- [Klissarov et al. · MaestroMotif: Skill Design from AI Feedback](https://arxiv.org/abs/2412.08542)（ICLR 2025）：结合技能描述、偏好奖励和程序化组合规则，训练并组织 NetHack 技能。语言提供了额外的先验和任务设计信息。
  [作者 MaestroMotif 实现](https://github.com/mklissa/maestromotif)：preference、code_generation、rl_baseline 分别对应奖励、组合与技能学习。

---

### 10 · 模型需要预测什么，才能在变化后继续支持决策？

问题设定：模型用于预测动作或 option 的后果，而不是仅重建观测。状态表示、策略和环境可以分别改变。模型误差必须相对于它将被使用的 backup 或决策定义。

已有认识：对于线性下游价值，适当的期望后果有时足以完成 backup；非线性价值、max 运算或风险目标一般不能只保留一个均值。世界模型、successor features 和 option model 的条件行为与输出语义不同。

尚缺什么：持续学习会产生接口失配：旧模型预测旧表示，规划却读取新表示；option 政策更新后，其终点与时长分布也会改变。高平均预测精度可能掩盖极少发生但决定策略排序的错误。

#### 方法与条件

- **Option 的奖励、终点与时长模型**：折扣模型需预测执行期间奖励及 $\gamma^\tau$ 加权的终点后果，再将终点价值接入备份。平均奖励模型还需计入持续时间成本，不能直接复用同一个折扣模型公式。
  条件与代价：必须说明模型针对哪个内部策略和终止规则；子任务 stopping bonus 不能混入真实环境奖励模型。
- **SR / SF 与 Default Representation**：Successor 类表示预测指定策略下的未来占用或特征累积，在奖励线性族中支持快速重估。DR 将奖励纳入默认后果表示，使表示本身更偏向特定奖励结构。
  条件与代价：动力学、默认行为或奖励条件改变时，可复用范围不同。不能把奖励变化、策略变化和转移变化视作同一迁移问题。
- **Dreamer 的潜在世界模型**：模型在真实序列上学习潜在动力学和预测头，随后在想象序列上训练行为。它直接服务 actor/critic，而不是要求所有像素都同样准确。
  条件与代价：模型训练数据、rollout 长度和想象分布决定偏差；跨域表现并不自动证明长期模型保留。

#### 相互竞争的解释

- 模型改善回报，可能因预测更准确，也可能因生成了更多训练目标。
- 更低 reconstruction loss 可能只来自拟合背景，而非保留了决策相关后果。

#### 可执行实验

假设：模型版本与当前技能策略一致，比单纯增加模型容量更能减少变化后的规划错误。

设计：在小 MDP 中分别改变 option 策略、奖励和环境转移。保存旧模型，比较立即更新、延迟更新、版本匹配及模型冻结；再做目标相关与无关视觉变化。

对照：真实模型 oracle、无模型控制、相同数据但更大模型；固定规划器与规划次数。

测量：奖励/时长/终点误差、backup 误差、动作排序错误、真实回报、模型重学成本。

什么结果会反驳该解释：模型预测改善但 backup 与动作排序不变，或收益仅由额外更新次数解释，则不能归因于更好的决策模型。

#### 教材与研究者

[转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/) · [Dyna 与模型学习](https://yingwen.io/zh/continual-rl/algorithms/dyna/) · [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/) · [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/) · [平均奖励与差分价值](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)

[Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/) · [Marlos C. Machado](https://yingwen.io/zh/continual-rl/resource/S06/) · [Matthew Schlegel](https://yingwen.io/zh/continual-rl/resource/S35/) · [André Barreto](https://yingwen.io/zh/continual-rl/resource/S40/) · [Danijar Hafner](https://yingwen.io/zh/continual-rl/resource/S104/)

#### 实验平台

- [解析小 MDP 与短序列](https://yingwen.io/zh/continual-rl/labs/)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [OGBench](https://github.com/seohongpark/ogbench)：离线目标条件强化学习的环境、数据与基线。 长时域、轨迹拼接、观测与数据覆盖。 边界：离线成功率不能直接说明在线技能发现或持续控制；规划时间另计。
- [Forager](https://github.com/andnp/forager)：部分可观测的持续资源获取。 信息历史、状态容量、更新成本与长期行为。 边界：感知模式和地图差异会改变问题；不只比较网络名字。

#### 原文与代码

- [Sutton et al. · Reward-Respecting Subtasks](https://arxiv.org/abs/2202.03466)（Artificial Intelligence 2023）：STOMP 连接子任务、option、真实后果模型与规划；子任务停止奖励和环境真实回报属于不同接口。
  原文提供更新规则和实验；此处没有可确认的完整作者 STOMP 工程。
- [Wan, Naik & Sutton · Learning and Planning in Average-Reward MDPs](https://proceedings.mlr.press/v139/wan21a.html)（ICML 2021）：用 TD 误差更新平均奖励率，建立无参考状态的差分预测、控制与规划更新。收敛条件针对论文规定的表格问题。
  [作者实验代码](https://github.com/abhisheknaik96/average-reward-methods)：从 prediction_agents.py、control_agents.py 对照奖励率和价值的联合更新。
- [Barreto et al. · Successor Features for Transfer in Reinforcement Learning](https://arxiv.org/abs/1606.05312)（NeurIPS 2017）：在相同动力学与线性奖励族下分离后果特征和奖励权重，通过 generalized policy improvement 复用多个策略。
  [后续作者 Option Keyboard 工程](https://github.com/google-deepmind/deepmind-research/tree/f5de0ede8430809180254ee957abf36ed62579ef/option_keyboard)：展示 SF 与策略组合的接口；对应后续 Option Keyboard，不是该 2017 论文的原始实验代码。
- [Tse, Chandrasekar & Machado · Reward-Aware Proto-Representations in RL](https://arxiv.org/abs/2505.16217)（NeurIPS 2025）：分析 Default Representation 的动态规划、TD 学习与特征形式，把奖励结构引入后果表示。
  [作者 DR 实验](https://github.com/httse9/Reward-Aware-Proto-Representations)：rep_utils.py 区分 SR/DR；ROD_DR.py 连接采样学习、谱分解与发现。
- [Hafner et al. · Mastering Diverse Domains through World Models](https://www.nature.com/articles/s41586-025-08744-2)（Nature 2025）：DreamerV3 通过真实序列学习潜在世界模型，再用想象轨迹训练策略与价值；行动时不必执行树搜索。
  [作者维护的公开重实现](https://github.com/danijar/dreamerv3)：README 将其称为 reimplementation；不是原内部训练工程。模型、想象学习和真实数据比例均需匹配。
- [Fu et al. · Knowledge Retention in Continual Model-Based RL](https://proceedings.mlr.press/v267/fu25f.html)（ICML 2025）：DRAGO 用生成经验保持动力学，再用内在奖励引导回访旧状态；任务共享状态空间与动力学。
  [作者 DRAGO 实现](https://github.com/YixiangSun/drago)：合成回放和真实回访各自消融；“不存旧任务数据”不等于没有当前任务 replay。

---

### 11 · 什么时候值得规划，应该把计算花在哪里？

问题设定：规划用已学模型进行额外计算。真实交互、模型更新、搜索和想象训练共享有限预算。问题不是简单地增加规划步数，而是选择何时、何处、以什么时间尺度使用模型。

已有认识：Dyna 在模型中产生价值更新；prioritized sweeping 根据前驱与变化传播分配备份；MPC 在行动时优化短序列；Dreamer 在想象中训练策略。它们把计算放在不同阶段。

尚缺什么：模型偏差可能随规划深度放大；规划太少又无法利用好已有知识。持续情形中，模型和状态同时变，旧搜索结构会失效。离线规划收益也不能直接代表在线发现、建模和规划的总收益。

#### 方法与条件

- **Dyna / STOMP：由模型支持价值改善**：Dyna 交替真实学习与模型备份；STOMP 进一步学习子任务、option 及其后果模型，将更长时间的行为作为可规划对象。
  条件与代价：抽象模型需与当前技能一致；计入 option 发现、模型训练及规划全部成本。
- **TD-MPC2：短模型路径加终点价值**：用短时域潜在预测和终点 critic 避免完全依赖长 rollout，策略先验帮助生成候选动作序列。
  条件与代价：模型误差和终点价值误差可以互相补偿，也可以共同误导；短时域不是无偏性保证。
- **ALPS：谱结构支持决策时分层规划**：先从离线数据学谱表示和模型，以区域图提出中间目标，再通过先验引导的短时域优化行动。它把全局连通结构和局部模型搜索分开。
  条件与代价：ICML 2026 的主要证据来自离线目标条件任务。它没有同时解决完整在线 STOMP 链条，二者不是新旧替代关系。

#### 相互竞争的解释

- 更好表现来自时间抽象，还是仅仅多用了测试时计算？
- 规划带来的收益来自更好的状态距离，还是更好的候选行为先验？

#### 可执行实验

假设：在固定总计算下，可靠短时域模型与抽象子目标比一味延长 rollout 更有效。

设计：先使用真模型比较规划器，再替换为相同数据学出的模型。扫描规划深度、候选数量和抽象层数；改变环境后观察恢复。

对照：无规划、同计算的额外策略更新、真模型 oracle；固定离线数据与动作 deadline。

测量：真实回报对总计算的曲线、模型误差、决策延迟、搜索失败率和环境变化后的恢复。

什么结果会反驳该解释：匹配测试时计算后优势消失，或真模型下仍无法获得收益，应分别否定效率解释或规划器设计。

#### 教材与研究者

[Dyna 与模型学习](https://yingwen.io/zh/continual-rl/algorithms/dyna/) · [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/) · [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/) · [Options 与技能发现](https://yingwen.io/zh/continual-rl/construction/options/)

[Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/) · [Marlos C. Machado](https://yingwen.io/zh/continual-rl/resource/S06/) · [George Konidaris](https://yingwen.io/zh/continual-rl/resource/S09/) · [Matthew E. Taylor](https://yingwen.io/zh/continual-rl/resource/S19/) · [Matthew Schlegel](https://yingwen.io/zh/continual-rl/resource/S35/) · [David Silver](https://yingwen.io/zh/continual-rl/resource/S62/) · [Danijar Hafner](https://yingwen.io/zh/continual-rl/resource/S104/)

#### 实验平台

- [解析小 MDP 与短序列](https://yingwen.io/zh/continual-rl/labs/)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [OGBench](https://github.com/seohongpark/ogbench)：离线目标条件强化学习的环境、数据与基线。 长时域、轨迹拼接、观测与数据覆盖。 边界：离线成功率不能直接说明在线技能发现或持续控制；规划时间另计。

#### 原文与代码

- [Sutton et al. · Reward-Respecting Subtasks](https://arxiv.org/abs/2202.03466)（Artificial Intelligence 2023）：STOMP 连接子任务、option、真实后果模型与规划；子任务停止奖励和环境真实回报属于不同接口。
  原文提供更新规则和实验；此处没有可确认的完整作者 STOMP 工程。
- [Wan, Naik & Sutton · Learning and Planning in Average-Reward MDPs](https://proceedings.mlr.press/v139/wan21a.html)（ICML 2021）：用 TD 误差更新平均奖励率，建立无参考状态的差分预测、控制与规划更新。收敛条件针对论文规定的表格问题。
  [作者实验代码](https://github.com/abhisheknaik96/average-reward-methods)：从 prediction_agents.py、control_agents.py 对照奖励率和价值的联合更新。
- [Hansen, Su & Wang · TD-MPC2](https://arxiv.org/abs/2310.16828)（ICLR 2024）：结合短时域潜在模型、终点价值与策略先验做决策时优化，降低长模型 rollout 的需求。
  [作者 TD-MPC2 实现](https://github.com/nicklashansen/tdmpc2)：模型损失、规划函数和策略先验共同构成方法；预训练迁移不等于持续在线积累。
- [Shehmar et al. · Laplacian Representations for Decision-Time Planning](https://proceedings.mlr.press/v306/shehmar26a.html)（ICML 2026）：ALPS 用谱表示组织区域子目标，并结合已学模型和短时域优化完成离线目标条件规划。
  [作者 ALPS 实现](https://github.com/machado-research/ALPS)：hierarchical.py 的区域路径与 optimizer.py 的 CEM 分开；训练数据是离线数据。
- [Gomez, Bowling & Machado · Proper Laplacian Representation Learning](https://arxiv.org/abs/2310.10833)（ICLR 2024）：ALLO 的约束与对偶动态恢复有序谱方向及相应特征值，减少只学到任意旋转子空间的问题。
  [作者 ALLO 实现](https://github.com/tarod13/laplacian_dual_dynamics)：同时读原始变量、对偶变量和正交约束；只保留相邻状态平滑项不是完整算法。

## Ⅴ · 长期适应与探索

怎样保留有用能力，并继续获得可学习的经验？

### 12 · 怎样保留旧能力，而不把过时知识强加给新任务？

问题设定：学习分布变化后，旧任务能力可能下降。保存旧轨迹、生成旧经验、保持旧策略或分配独立模块都可以减少遗忘，但也可能减慢适应。任务边界和回访权限决定哪些方法可用。

已有认识：Replay 保留数据，蒸馏保留函数输出，参数正则保留局部参数几何，生成模型保留经验分布。它们保存的对象不同。低遗忘不等于高前向迁移，快速新学习也不等于旧知识保存良好。

尚缺什么：多数任务序列允许明确边界与评估重置，持续无边界条件更难分清“旧知识”与“已失效的规律”。固定容量下应保留哪些知识，仍取决于未来用途和重学代价。

#### 方法与条件

- **CLEAR：经验与行为共同保留**：混合当前和历史经验进行学习，同时在旧经验上约束行为与价值输出。单纯增加 reservoir 容量不等于实现完整 CLEAR。
  条件与代价：允许存储和再访问旧 transition；历史行为概率及离策略修正是接口的一部分。
- **DRAGO：模型的生成回放与真实回访**：合成旧经验用于维护动力学模型，内在奖励引导重新访问熟悉区域，使保留机制同时作用于训练数据和行为。
  条件与代价：论文任务共享动力学与状态空间。模型生成误差、当前任务 replay 和任务边界不能从成本中省略。
- **Reset & Distill：分开新任务学习与保留**：新任务在线学习器重新开始，expert 蒸馏保留过去行为。它明确面对旧参数初始化造成的负迁移，而非只压制遗忘。
  条件与代价：网络重置、离线训练和多个信息来源有额外权限与资源。

#### 相互竞争的解释

- 回访更好来自知识仍在，也可能来自更快重新学习；需要零更新诊断区分。
- 保留正则有效可能只是降低总体学习速度，新任务变化慢时恰好受益。

#### 可执行实验

假设：保存决策相关行为或后果，比同容量随机旧数据更有效地兼顾回访与新任务学习。

设计：使用 A→B→A 与 A→B→C 两条序列；在回访前先冻结诊断，再恢复更新。分别改变奖励与动力学，不合并解释。

对照：从头训练、连续微调、等容量 reservoir、等计算蒸馏；有任务边界与无边界分别比较。

测量：零更新保留、再学习速度、前向迁移、在线总效用、存储字节和额外计算。

什么结果会反驳该解释：只在恢复训练后才出现回访收益，就不能称为零更新知识保留；新任务成本抵消收益时也不支持总效用提升。

#### 教材与研究者

[知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/) · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/) · [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/) · [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

[Doina Precup](https://yingwen.io/zh/continual-rl/resource/S07/) · [George Konidaris](https://yingwen.io/zh/continual-rl/resource/S09/) · [Sarath Chandar](https://yingwen.io/zh/continual-rl/resource/S13/) · [Razvan Pascanu](https://yingwen.io/zh/continual-rl/resource/S14/) · [Michael L. Littman](https://yingwen.io/zh/continual-rl/resource/S42/) · [Samuel Kessler](https://yingwen.io/zh/continual-rl/resource/S100/) · [Piotr Miłoś](https://yingwen.io/zh/continual-rl/resource/S103/)

#### 实验平台

- [解析小 MDP 与短序列](https://yingwen.io/zh/continual-rl/labs/)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [Continual World](https://github.com/awarelab/continual_world)：基于 Meta-World 的机器人任务序列。 遗忘、前向迁移及回访表现。 边界：任务边界、重置和任务标识应明确；不是严格无重置协议。
- [MinAtar](https://github.com/kenjyoung/MinAtar)：低成本、简化视觉的 Atari 风格环境。 深度控制、函数变化和多 seed 消融。 边界：任务切换须另行规定；切换版本与普通 MinAtar 不是同一协议。

#### 原文与代码

- [Rolnick et al. · Experience Replay for Continual Learning](https://papers.nips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html)（NeurIPS 2019）：CLEAR 将新经验学习、旧经验离策略学习与行为/价值克隆组合，以维持已有能力。
  原始论文为方法依据。AGI-Labs/continual_rl 提供后续比较框架中的复现，不是 CLEAR 原论文作者代码。
- [Fu et al. · Knowledge Retention in Continual Model-Based RL](https://proceedings.mlr.press/v267/fu25f.html)（ICML 2025）：DRAGO 用生成经验保持动力学，再用内在奖励引导回访旧状态；任务共享状态空间与动力学。
  [作者 DRAGO 实现](https://github.com/YixiangSun/drago)：合成回放和真实回访各自消融；“不存旧任务数据”不等于没有当前任务 replay。
- [Ahn et al. · Prevalence of Negative Transfer in Continual RL](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ba9e3d60610f3525717665966d86e0cd-Abstract-Conference.html)（ICLR 2025）：Reset & Distill 在新任务中重置在线学习器，并离线蒸馏在线策略与过去 expert，研究负迁移和知识保留。
  [作者 Reset & Distill](https://github.com/hongjoon0805/Reset-Distill)：任务边界、网络重置、离线蒸馏与额外 expert 存储均应纳入实验协议。

---

### 13 · 为什么训练越久，学习新东西反而越慢？

问题设定：长期训练后的网络面对新的可学习目标，下降速度比新初始化更慢。需在相同数据和目标上比较 aged 与 fresh 网络，才能把优化能力下降和探索失败区分开。

已有认识：休眠单元、参数尺度、有效学习率、损失几何和预测函数剧烈变化都可能参与可塑性损失。单一活动度或参数范数是诊断变量，不是可塑性的完整定义。

尚缺什么：一个机制在监督漂移或特定控制环境有效，不代表它解决了长期交互。替换单元改善新学习时，可能损坏旧功能；约束函数变化又可能拖慢必要的适应。

#### 方法与条件

- **ReDo / Continual Backprop**：ReDo 以低活动度发现可回收单元；CBP 结合效用、成熟期和替换预算引入新特征。替换涉及输入、输出连接与 optimizer 状态。
  条件与代价：单位替换量、初始扰动和实际参数预算必须匹配；激活率提高不证明新任务学得更快。
- **Normalize-and-Project**：归一化网络中，权重范数增长会改变有效学习率。NaP 将归一化与权重投影结合，使这一隐含调度可被显式研究。
  条件与代价：保持有效学习率不必在所有任务最好；原文也分析隐含衰减何时有帮助。
- **C-CHAIN：减少函数 churn**：在参考状态上限制近期更新造成的全动作 Q 变化，以减轻一个数据片段对其他预测的干扰。它干预的是函数变化，而不是直接重置单元。
  条件与代价：需要参考数据与近期网络；正则过强可能妨碍合理策略变化，且不满足无 replay 的最严格协议。

#### 相互竞争的解释

- 新任务学得慢是优化退化，还是新数据没有覆盖需要学习的区域？
- 重置有效是因为产生了新特征，还是因为重置了学习率、动量或价值尺度？

#### 可执行实验

假设：在固定训练数据上控制有效学习率后，特征回收仍能独立改善长期新学习能力。

设计：保存不同训练年龄的网络，在同一新目标上做短期学习探针；再将同一干预放回在线控制。设置 feature reset 与 optimizer reset 的交叉消融。

对照：fresh、aged、随机部分重置、只重置 optimizer、NaP/步长调整；每组相同参数和调参预算。

测量：相同初始误差附近的下降速度、旧功能扰动、有效学习率、函数 churn、在线回报与长期失稳。

什么结果会反驳该解释：调整有效学习率即可消除 aged/fresh 差异，或回收只改善活动指标却不改善学习曲线，就应缩小特征退化解释。

#### 教材与研究者

[可塑性与特征更新](https://yingwen.io/zh/continual-rl/algorithms/plasticity/) · [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/) · [深度价值学习](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)

[Clare Lyle](https://yingwen.io/zh/continual-rl/resource/S11/) · [Shibhansh Dohare](https://yingwen.io/zh/continual-rl/resource/S12/) · [Razvan Pascanu](https://yingwen.io/zh/continual-rl/resource/S14/) · [Will Dabney](https://yingwen.io/zh/continual-rl/resource/S65/) · [Evgenii Nikishin](https://yingwen.io/zh/continual-rl/resource/S105/) · [Ghada Sokar](https://yingwen.io/zh/continual-rl/resource/S106/)

#### 实验平台

- [解析小 MDP 与短序列](https://yingwen.io/zh/continual-rl/labs/)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [MinAtar](https://github.com/kenjyoung/MinAtar)：低成本、简化视觉的 Atari 风格环境。 深度控制、函数变化和多 seed 消融。 边界：任务切换须另行规定；切换版本与普通 MinAtar 不是同一协议。
- [AgarCL 与作者基线](https://github.com/machado-research/AgarCL-benchmark)：完整持续世界与分解小游戏。 探索、记忆、信用及可塑性的共同作用。 边界：死亡后局部重生不等于整套学习器清空；混合动作与预算需保持一致。

#### 原文与代码

- [Sokar et al. · The Dormant Neuron Phenomenon in Deep RL](https://proceedings.mlr.press/v202/sokar23a.html)（ICML 2023）：ReDo 根据相对活动度回收休眠单元，重置输入权重并控制输出影响。
  [作者 Dopamine/ReDo](https://github.com/google/dopamine/tree/master/dopamine/labs/redo)：weight_recyclers.py 的活动度阈值、替换频率与权重处理。
- [Dohare et al. · Loss of Plasticity in Deep Continual Learning](https://www.nature.com/articles/s41586-024-07711-7)（Nature 2024）：Continual Backprop 以效用、成熟期和替换预算控制特征更新，研究长期新学习能力。
  [作者 CBP 实现](https://github.com/shibhansh/loss-of-plasticity)：gnt.py 和 AdamGnT.py 处理单元替换及 optimizer 状态；不同效用变体要分开。
- [Lyle et al. · Normalization and Effective Learning Rates in RL](https://papers.nips.cc/paper_files/paper/2024/hash/c04d37be05ba74419d2d5705972a9d64-Abstract-Conference.html)（NeurIPS 2024）：研究归一化、权重尺度和有效学习率之间的关系，为可塑性下降提供区别于死神经元的解释。
  本条以原文算法与分析为入口；此处没有可确认的作者原始实现。
- [Tang et al. · Mitigating Plasticity Loss in Continual RL by Reducing Churn](https://arxiv.org/abs/2506.00592)（ICML 2025）：C-CHAIN 对参考状态上的函数变化施加约束，降低当前更新对其他预测的扰动。
  [作者 C-CHAIN 实现](https://github.com/bluecontra/C-CHAIN)：crl_minatar/agents/double_dqn_c_chain.py 包含参考 batch、网络队列和全动作 Q 正则。
- [Mohamed et al. · The Cell Must Go On: Agar.io for Continual RL](https://arxiv.org/abs/2505.18347)（2025 初稿；2026 修订）：用完整世界与分解小游戏共同研究探索、记忆、信用与可塑性；一个局部机制难以解释全部困难。
  [作者环境](https://github.com/machado-research/AgarCL)：配套训练在 machado-research/AgarCL-benchmark；动作、重生和观察配置须与实验配对。

---

### 14 · 应当探索什么、练习什么，以及如何保留未来交互与学习的机会？

问题设定：智能体的行为决定未来数据。新奇、可学习性、学习进展和任务收益并不一致。课程 teacher 能否选目标、改环境或重置，也会从根本上改变问题。

已有认识：预测误差可以鼓励访问未熟悉状态，学习进展可以优先分配可改善任务，恢复策略可以减少陷入失能状态的风险。但噪声环境可能永远产生大误差，短期安全也不等于长期可恢复。

尚缺什么：持续经验选择需要区分暂时不会、根本不可预测和对未来没用。没有外部救援时，还要把不可逆风险纳入评价。技能覆盖或新奇奖励不能代替任务回报与恢复能力。

#### 方法与条件

- **RND：用预测误差生成新奇奖励**：固定随机 target 网络，训练 predictor；未熟悉输入产生较大误差。归一化和内外在价值估计决定该信号怎样改变行为。
  条件与代价：误差受输入噪声与表示尺度影响，不自动等价于信息增益或真正的学习进步。
- **ALP-GMM：按学习进展分配课程**：根据环境参数附近表现变化估计绝对学习进展，用混合模型选择下一批训练条件。它调整的是经验分布，而非低层 RL 更新。
  条件与代价：teacher 需要选择训练条件和反复试验的权限；变化的成功率也可能来自噪声。
- **Single-Life RL / QWALE**：以先前经验指导一次部署中的恢复与适应。QWALE 用价值加权的分布匹配信号，帮助 agent 从未见状态回到有用经验支持区域。
  条件与代价：允许先验数据与预训练，但部署中不能随意重置；返回旧分布不等于任意环境中的安全保证。

#### 相互竞争的解释

- 课程收益来自学习进展估计，还是来自 teacher 排除了太难或危险的任务？
- 恢复成功来自在线适应，还是来自预训练已经覆盖了测试变化？

#### 可执行实验

假设：区分可学习变化与不可约噪声，能够减少无效探索而不损失任务发现。

设计：在小环境同时设置稀疏有用线索、纯噪声区域和可恢复/不可恢复陷阱。比较新奇和学习进展；若有 teacher，另开同权限实验。

对照：随机探索、固定内在权重、oracle 可学区域仅作诊断上界；匹配先验数据和外部救援次数。

测量：有用覆盖、噪声区域停留、首次任务发现、恢复成功、失能时间、外在累计奖励和人工干预。

什么结果会反驳该解释：去掉 teacher 的环境筛选权限后优势消失，或探索仅增加覆盖而未改善后续学习能力，则需更换机制解释。

#### 教材与研究者

[探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/) · [目标条件化与子任务构造](https://yingwen.io/zh/continual-rl/construction/goals/) · [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)

[Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/) · [Peter Stone](https://yingwen.io/zh/continual-rl/resource/S10/) · [Sergey Levine](https://yingwen.io/zh/continual-rl/resource/S49/) · [Chelsea Finn](https://yingwen.io/zh/continual-rl/resource/S50/) · [Tim Rocktäschel](https://yingwen.io/zh/continual-rl/resource/S60/) · [Ian Osband](https://yingwen.io/zh/continual-rl/resource/S64/) · [Pierre-Yves Oudeyer](https://yingwen.io/zh/continual-rl/resource/S71/)

#### 实验平台

- [解析小 MDP 与短序列](https://yingwen.io/zh/continual-rl/labs/)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [TeachMyAgent](https://github.com/flowersteam/TeachMyAgent)：teacher 可以选择环境参数的自动课程实验。 学习进展估计和经验分配。 边界：必须计算 teacher 权限、失败尝试和环境重置，不能视作完全自主单生命期。
- [Forager](https://github.com/andnp/forager)：部分可观测的持续资源获取。 信息历史、状态容量、更新成本与长期行为。 边界：感知模式和地图差异会改变问题；不只比较网络名字。
- [AgarCL 与作者基线](https://github.com/machado-research/AgarCL-benchmark)：完整持续世界与分解小游戏。 探索、记忆、信用及可塑性的共同作用。 边界：死亡后局部重生不等于整套学习器清空；混合动作与预算需保持一致。

#### 原文与代码

- [Burda et al. · Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)（ICLR 2019）：以固定随机网络的预测误差构造新奇奖励，减少显式密度建模需求，但误差不是学习进展本身。
  [作者 RND 实现](https://github.com/openai/random-network-distillation)：观察内在奖励归一化、内外价值估计及 PPO 训练，不仅复制预测器损失。
- [Portelas et al. · Teacher Algorithms for Curriculum Learning of Deep RL](https://arxiv.org/abs/1910.07224)（CoRL 2019 / PMLR 2020）：ALP-GMM 用目标/环境参数附近的绝对学习进展分配训练分布。teacher 可以选择环境条件。
  [作者团队课程实验平台](https://github.com/flowersteam/TeachMyAgent)：支持 ALP-GMM 等方法的后续比较框架；不是无 teacher 的单生命期协议。
- [Chen et al. · You Only Live Once: Single-Life Reinforcement Learning](https://arxiv.org/abs/2210.08863)（NeurIPS 2022）：研究一次部署生命期内利用先前经验完成适应与恢复，突出无法随意外部重置的限制。
  [作者 Single-Life RL / QWALE](https://github.com/anniesch/single-life-rl)：envs/、data/ 和 train.py 分别提供环境、先验数据及部署学习；比较时应计算预训练和先验数据。

## Ⅵ · 架构与证据

模块之间的改进能否变成长生命期收益？

### 15 · 状态、知识、技能与规划怎样共同构成持续智能体？

问题设定：完整 agent 同时更新状态、预测、控制、技能与模型。每个局部学习器看到的输入和目标都可能被其他模块改变。整体表现还受内存、计算分配和数据获取限制。

已有认识：Dyna 给出直接学习、模型学习和规划的基本接口；Alberta Plan 与 OaK 把长期知识构建、时间抽象及规划纳入更广路线。它们提出研究组织方式，但模块相连本身不能保证收益相加。

尚缺什么：需要可检验的接口契约与端到端预算。状态变化会使预测和模型陈旧；技能变化会使后果模型失效；规划又改变经验。完整系统的增益必须与其组件、额外容量和额外先验分别比较。

#### 方法与条件

- **STOMP 的模块链**：子任务定义所学行为，option 实现行为，真实后果模型描述执行结果，规划使用模型改善任务价值。模块关系可以逐项测量，而不只画架构图。
  条件与代价：子任务奖励、真实奖励、停止条件和模型时间尺度须一致；每个接口都可能成为误差源。
- **Alberta Plan / OaK 的研究路线**：把有限资源下的实时学习与经验构造知识连接起来，强调可由经验检验、可支持规划的时间抽象。研究焦点是模块如何在生命期中产生并改进。
  条件与代价：研究纲领和讲座不等于完整通用 agent 的可复现性能报告；能力与效率需要对应实现和测试。
- **Oak 的选择性信用与事件驱动方向**：公开研究文章用稀疏相关信号与大量干扰说明选择性步长的动机，并展示 NetworkIDBD 的预测演示。事件驱动方向涉及计算分配，但对应页面尚无技术正文。
  条件与代价：不能从 NoisyMNIST 图示推出持续控制收益；没有完整更新与运行包时，不能复现或比较其能耗主张。

#### 相互竞争的解释

- 整机提高回报可能只是参数和计算更多，或某一个强模块掩盖了其他模块无效。
- 模块之间看似协同，可能来自共同访问了额外任务标签或真实环境状态。

#### 可执行实验

假设：模型与技能的版本一致性，是持续抽象系统在环境变化后保持收益的必要因素之一。

设计：搭建只含一个技能、一个后果模型与固定预算 planner 的小系统；逐步允许状态、技能和模型更新。通过延迟一个模块制造可控失配，再进行完全组合。

对照：相同总容量与预算的平坦控制器；分别冻结状态、预测、技能、模型与元更新。环境真状态只供诊断，不输入 agent。

测量：整段效用、接口预测误差、模型版本失配、每模块成本、技能使用率与变化后恢复。

什么结果会反驳该解释：删除某模块不影响行为，或收益只随总计算增长而不随接口一致性变化，就不支持该模块协同解释。

#### 教材与研究者

[持续学习的智能体架构](https://yingwen.io/zh/continual-rl/construction/architectures/) · [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/) · [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/) · [目标条件化与子任务构造](https://yingwen.io/zh/continual-rl/construction/goals/) · [Options 与技能发现](https://yingwen.io/zh/continual-rl/construction/options/) · [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/) · [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/) · [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

[Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/) · [Marlos C. Machado](https://yingwen.io/zh/continual-rl/resource/S06/) · [Michael Bowling](https://yingwen.io/zh/continual-rl/resource/S17/) · [Patrick M. Pilarski](https://yingwen.io/zh/continual-rl/resource/S18/) · [Joseph Modayil](https://yingwen.io/zh/continual-rl/resource/S32/) · [Khurram Javed](https://yingwen.io/zh/continual-rl/resource/S33/)

#### 实验平台

- [解析小 MDP 与短序列](https://yingwen.io/zh/continual-rl/labs/)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [Forager](https://github.com/andnp/forager)：部分可观测的持续资源获取。 信息历史、状态容量、更新成本与长期行为。 边界：感知模式和地图差异会改变问题；不只比较网络名字。
- [AgarCL 与作者基线](https://github.com/machado-research/AgarCL-benchmark)：完整持续世界与分解小游戏。 探索、记忆、信用及可塑性的共同作用。 边界：死亡后局部重生不等于整套学习器清空；混合动作与预算需保持一致。

#### 原文与代码

- [Sutton et al. · Reward-Respecting Subtasks](https://arxiv.org/abs/2202.03466)（Artificial Intelligence 2023）：STOMP 连接子任务、option、真实后果模型与规划；子任务停止奖励和环境真实回报属于不同接口。
  原文提供更新规则和实验；此处没有可确认的完整作者 STOMP 工程。
- [Sutton, Bowling & Pilarski · The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173)（2022 研究纲领）：从实时学习、状态与预测走向子任务、模型和规划的组合研究路线；各步骤仍有独立科学问题。
  研究纲领，不是一套完整 agent 发布包。
- [Sutton · The OaK Architecture](https://oaklab.ai/posts/the-oak-architecture)（RLC 2025 讲座入口）：以经验中的时间抽象、可检验知识和规划构造智能体；适合分析模块之间的信息与资源接口。
  公开入口为架构讲座；没有在此页面发布完整 OaK agent 代码。
- [Oak Lab · Learning from Experience instead of Curated Datasets](https://oaklab.ai/posts/learning-from-experience-instead-of-curated-datasets)（2026 官方研究文章）：用信号、干扰特征和噪声目标的受控预测实验讨论选择性信用，展示 IDBD 与 NetworkIDBD。
  文章未给出完整 NetworkIDBD 更新规则与公开复现包。NoisyMNIST 演示不等于长期控制实验。
- [Oak Lab · Event-Driven Neural Networks with Batch-Size One Learning Algorithms](https://oaklab.ai/posts/event-driven-computation-with-batch-size-one)（官方主题入口）：页面仍为 Coming soon；仅说明研究主题，不能由标题推导算法、能耗或性能。
  无公开技术正文或实现入口。

---

### 16 · 什么实验能区分“仍在更新”与“仍在有效学习”？

问题设定：一条长曲线包含适应、探索、保留、环境变化和随机波动。训练中评估可能重置环境或暂停更新，这些操作会改变单生命期。研究需要先确定估计量与独立随机单位。

已有认识：在线奖励评价实际学习过程；冻结策略诊断回答某个时间点会做什么；回访与新目标探针分别测试保留与可塑性。一个指标不能代替另外几个。跨种子区间与多任务稳健聚合有助于避免只看最好结果。

尚缺什么：有限实验不能证明永远持续改善，但可以排除明确失败解释。领域仍需要跨协议可比较的报告：真实时间、内存、全部训练成本、重置权限、变化来源及模型版本，而不仅是最终归一化均分。

#### 方法与条件

- **生命期指标与诊断分离**：把实际交互累计效用作为主要结果；冻结、回访和新任务探针在独立拷贝中进行，避免诊断数据改变主生命期。
  条件与代价：拷贝只用于离线诊断且不把结果回流到部署 agent；无法复制真实系统时，需要另行设计安全评估。
- **rliable 与分层统计**：以独立训练运行和任务为重采样层级，报告区间、分布和稳健聚合，避免少数异常高分支配平均数。
  条件与代价：相邻窗口不是独立 seed；只挑显著任务或事后选择最优时间点会使区间失去原解释。
- **机制诊断 → 简化平台 → 复杂世界**：解析任务验证公式，Forager 检查信息与状态，AgarCL 的小游戏及完整环境检查组合难点。不同平台提供互补证据，不按难度大小简单替代。
  条件与代价：某一平台上的失败可限制具体主张，但不能直接宣布一种机制普遍无用。

#### 相互竞争的解释

- 后期变好可能因为任务自然变容易，或评价状态更接近训练分布。
- 极小均分提升可能来自种子方差、任务权重或选择性报告，而不是算法机制。

#### 可执行实验

假设：一个被称作“持续学习改进”的方法，应在共同预算下至少改善预先指定的生命期指标，而不是仅改善代理诊断。

设计：预先固定任务顺序、变化时点、主指标和停止条件。开发集调参；测试集运行完整生命期。使用配对任务种子，保留失败和超时运行。

对照：固定策略、不学习、从头训练、持续基础算法与单机制消融；匹配所有环境干预。

测量：在线效用及区间、适应/保留/可塑性诊断、峰值内存、时延分位数、崩溃和干预频次；显示每任务结果。

什么结果会反驳该解释：主指标无可靠改善，或改善仅存在于获得额外 reset/搜索预算的条件，就不支持原定端到端主张；负结果仍可支持更窄机制结论。

#### 教材与研究者

[实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/) · [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/) · [持续学习的智能体架构](https://yingwen.io/zh/continual-rl/construction/architectures/) · [可塑性与特征更新](https://yingwen.io/zh/continual-rl/algorithms/plasticity/) · [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)

[Martha White](https://yingwen.io/zh/continual-rl/resource/S03/) · [Adam White](https://yingwen.io/zh/continual-rl/resource/S04/) · [David Abel](https://yingwen.io/zh/continual-rl/resource/S08/) · [Csaba Szepesvári](https://yingwen.io/zh/continual-rl/resource/S15/) · [Michael Bowling](https://yingwen.io/zh/continual-rl/resource/S17/) · [Rishabh Agarwal](https://yingwen.io/zh/continual-rl/resource/S109/)

#### 实验平台

- [解析小 MDP 与短序列](https://yingwen.io/zh/continual-rl/labs/)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [CSuite](https://rl-csuite.readthedocs.io/en/latest/)：持续交互的小型诊断环境。 低成本测试适应与学习机制，再接复杂平台。 边界：每个任务测量的困难有限；不可用单一分数代表整个 CRL。
- [Forager](https://github.com/andnp/forager)：部分可观测的持续资源获取。 信息历史、状态容量、更新成本与长期行为。 边界：感知模式和地图差异会改变问题；不只比较网络名字。
- [Continual World](https://github.com/awarelab/continual_world)：基于 Meta-World 的机器人任务序列。 遗忘、前向迁移及回访表现。 边界：任务边界、重置和任务标识应明确；不是严格无重置协议。
- [AgarCL 与作者基线](https://github.com/machado-research/AgarCL-benchmark)：完整持续世界与分解小游戏。 探索、记忆、信用及可塑性的共同作用。 边界：死亡后局部重生不等于整套学习器清空；混合动作与预算需保持一致。

#### 原文与代码

- [Abel et al. · A Definition of Continual Reinforcement Learning](https://arxiv.org/abs/2307.11046)（NeurIPS 2023）：用智能体对行为基础的持续搜索定义持续性。它与“环境显式切换若干任务”的定义不相同。
  理论定义论文；没有需要复现的单一控制算法。
- [Elelimy et al. · Rethinking the Foundations for Continual RL](https://rlj.cs.umass.edu/2025/papers/Paper243.html)（RLC 2025）：以 history process 和 deviation regret 研究学习过程的评价。比较对象与可实现偏离集合仍需明确，不能只把指标名称换掉。
  论文给出形式化与实验；此处没有可确认的作者原始代码入口。
- [Agarwal et al. · Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)（NeurIPS 2021）：用区间估计、分层 bootstrap 和稳健聚合处理多任务、多随机种子的评价不确定性。
  [作者 rliable](https://github.com/google-research/rliable)：独立随机单位是 run/任务而非同一曲线的相邻时间点；统计工具不能修复不公平协议。
- [Forager: a Lightweight Testbed for Continual Learning with Partial Observability in RL](https://arxiv.org/abs/2605.01131)（2026 预印本）：在长期交互中隔离观测与状态构造问题，研究局部感知下的资源获取。
  [作者 Forager 环境](https://github.com/andnp/forager)：配置 observation_mode 和地图；start()/step() 接口不同于 Gymnasium 五元组。
- [Mohamed et al. · The Cell Must Go On: Agar.io for Continual RL](https://arxiv.org/abs/2505.18347)（2025 初稿；2026 修订）：用完整世界与分解小游戏共同研究探索、记忆、信用与可塑性；一个局部机制难以解释全部困难。
  [作者环境](https://github.com/machado-research/AgarCL)：配套训练在 machado-research/AgarCL-benchmark；动作、重生和观察配置须与实验配对。