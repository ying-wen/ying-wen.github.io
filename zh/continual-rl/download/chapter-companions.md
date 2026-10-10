# 章节、实验、资源与学者

## 交互、奖励与优化目标

折扣与平均奖励是目标选择。持续学习不能仅靠把回合接长来定义。

先用可解析的循环 MDP 检查策略排序，再讨论更复杂环境。

[教材](https://yingwen.io/zh/continual-rl/foundations/objectives/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-objectives)

- Richard S. Sutton｜从时间信用分配到持续经验型智能体：奖励、回报与持续交互的统一问题表述。
- Andrew G. Barto｜自适应控制、内在动机与层级学习：自适应控制与强化学习问题的形成。
- David Abel｜从表示抽象到“学习究竟在哪里”：持续学习定义与智能体内部学习过程。
- Michael Bowling｜博弈、评测与 agency：比较固定策略与完整学习过程，明确奖励表示、研究假说和形式化提案的不同角色。
- Martha White｜可靠 off-policy 学习到长期控制：将环境、任务、关注分布与学习目标分别定义。
- Esraa Elelimy｜高效在线递归学习：区分冻结策略评价与持续学习过程评价。
- Khurram Javed｜有限算力下的持续构造与信用分配：把大世界立场转化为可被反驳的研究问题。
- Patrick M. Pilarski｜预测知识、人机共适应与身体：智能增强的系统目标

## 奖励假设与奖励设计

偏好、奖励机制、回报与辅助信号不是同一对象。保持最优策略、加快学习与符合设计者意图需要不同证据。

先检验顺序目标、塑形边界和偏好梯度；再用 B-Pref 与安全诊断环境检验反馈成本和代理失效。

[教材](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-reward-design)

- Richard S. Sutton｜从时间信用分配到持续经验型智能体：奖励假设与从经验学习的总体问题。
- David Abel｜从表示抽象到“学习究竟在哪里”：奖励表达能力与持续学习的概念基础。
- Michael Bowling｜博弈、评测与 agency：奖励表示的公理条件与设计者目标。
- Will Dabney｜价值分布与目标／agent 基础：奖励表达与偏好表示。
- Satinder Singh｜内在动机、时间抽象与 agency：奖励、内在动机与智能能力的关系。
- John D. Martin｜奖励表达、规划计算与环境中的记忆：检验目标能被奖励表达的条件。
- Matthew E. Taylor｜迁移、教学与人类输入：人类输入的语义

## 平均奖励与差分价值

平均奖励按原始时间计收益。随机时长 option 要使用半马尔可夫时间口径。

已知平稳分布的循环 MDP；再用持续控制任务检查暂态、长期奖励率与漂移。

[教材](https://yingwen.io/zh/continual-rl/algorithms/average-reward/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-average)

- Martha White｜可靠 off-policy 学习到长期控制：平均奖励、可靠 off-policy 学习与长期控制。
- Adam White｜从实时预测知识到可信实验：平均奖励学习与经验研究。
- Richard S. Sutton｜从时间信用分配到持续经验型智能体：差分价值与 continuing control。
- Kris De Asis｜多步价值学习与面向真实时间的机器人：从真实时间理解持续任务的回报。

## 智能体状态与递归学习

深度网络不能自动消除部分可观测性。必须区分观测编码、历史状态递推与参数学习。

POPGym 隔离记忆需求；Forager 检查长期交互。对照完整状态 oracle 时标明额外信息权限。

[教材](https://yingwen.io/zh/continual-rl/construction/state/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-state)

- Matthew Schlegel｜预测式状态与表示支持规划：预测状态与可规划表示。
- Esraa Elelimy｜高效在线递归学习：递归网络的在线训练。
- Leslie Pack Kaelbling｜部分可观测与层级机器人规划：部分可观测决策与信念状态。
- Amy Zhang｜可泛化与分层决策所需的表示：面向决策的状态表示。
- Joseph Modayil｜从感知结构到可验证预测知识：从观测关系构造内部状态。
- Randy Goebel｜知识表示、推理与可检验的解释：分开状态的预测充分性与人类可理解性。
- John D. Martin｜奖励表达、规划计算与环境中的记忆：研究内部状态与外部痕迹的边界。
- Dale Schuurmans｜表示、优化与可规划状态：表示要支持预测与决策；部分可观测的理论条件应和实现一起阅读。
- Martha White｜可靠 off-policy 学习到长期控制：比较预测语义、表示可更新性与循环计算成本。
- Adam White｜从实时预测知识到可信实验：同时评价部分可观测性、记忆语义和更新代价。
- Khurram Javed｜有限算力下的持续构造与信用分配：从计算依赖解释可在线训练的循环架构。

## 价值预测与资格迹

表格 TD 的局部更新推广为共享特征上的参数更新。GVF 再改变预测问题，而非只改变网络。

Random Walk 与小型马尔可夫奖励过程：有解析真值，便于分开估计误差和程序错误。

[教材](https://yingwen.io/zh/continual-rl/algorithms/value/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-value)

- Richard S. Sutton｜从时间信用分配到持续经验型智能体：TD 学习与资格迹。
- Andrew G. Barto｜自适应控制、内在动机与层级学习：价值学习与演员—评论家框架。
- Peter Dayan｜预测表征与计算神经科学：Successor representation 连接预测与表征。
- Arsalan Sharifnassab｜更新几何、步长适应与流式学习：比较 Bellman 目标与更新几何。
- Csaba Szepesvári｜统计效率和算法边界：把表格 TD、函数逼近与随机逼近保证的条件逐一列出。
- Martin Müller｜长期价值与临时搜索记忆：TD 不只可用于真实轨迹学习，也可作为模拟搜索中的局部价值更新。

## 通用价值函数与预测知识

标量奖励价值是 GVF 的一种特例。定义多个问题不等于已经学出有用状态或控制策略。

可解析 off-policy MRP 检查预测真值；机器人信号预测需独立测决策效用。

[教材](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-gvf)

- Joseph Modayil｜从感知结构到可验证预测知识：以通用价值函数组织预测知识。
- Patrick M. Pilarski｜预测知识、人机共适应与身体：机器人预测知识与人机共适应。
- Adam White｜从实时预测知识到可信实验：并行离策略预测与实时经验学习。
- Martha White｜可靠 off-policy 学习到长期控制：稳定的离策略预测方法。
- Matthew Schlegel｜预测式状态与表示支持规划：预测问题及其发现

## 持续控制与学习智能体比较

固定策略、记忆递推和完整学习器是不同评价对象。CRL 可定义历史条件价值，但不能不加条件地沿用固定 MDP 的策略排序。

解析例子检验时域排序、相同策略下不同学习规则、不可逆后果与偏离遗憾；再扩展到单次生命期评价。

[教材](https://yingwen.io/zh/continual-rl/algorithms/control/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-control)

- Michael Bowling｜博弈、评测与 agency：持续学习器的评价与可行偏离比较。
- Martha White｜可靠 off-policy 学习到长期控制：适应过程、学习目标与评测。
- David Abel｜从表示抽象到“学习究竟在哪里”：历史过程与持续学习定义。
- Richard S. Sutton｜从时间信用分配到持续经验型智能体：GPI 为局部策略改善提供基础。
- Csaba Szepesvári｜统计效率和算法边界：固定问题中的策略优化率不能代替整个持续学习器的生命期评价。
- Dieter Büchler｜高动态机器人中的在线适应：动态运动与物理时间

## 深度价值学习

DQN 保留 TD 目标。网络、回放与目标网络增加新的时间尺度，长期训练时也可能引入陈旧数据与可塑性问题。

CartPole 做管线检查；MinAtar 或 ALE 做视觉控制。不能把 CartPole 成功当作 Atari 复现。

[教材](https://yingwen.io/zh/continual-rl/algorithms/deep-value/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-deep-value)

- Hado van Hasselt｜稳定深度价值学习与持续 RL 基础：Double Q-learning 与深度价值估计偏差。
- Marc G. Bellemare｜表示、分布式价值与可靠环境：分布式价值学习与 Atari 评测。
- Will Dabney｜价值分布与目标／agent 基础：分位数价值分布。

## 策略梯度与 actor–critic

从 REINFORCE 到 actor–critic，评论家提供低方差学习信号。PPO 的批量多轮更新与严格流式协议不同。

离散 bandit 检查梯度；CartPole 检查完整采样更新；连续控制再检查动作概率密度。

[教材](https://yingwen.io/zh/continual-rl/algorithms/policy/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-policy)

- Sergey Levine｜从现实机器人经验到可泛化控制：策略优化、机器人学习与深度 RL 教学。
- Pieter Abbeel｜技能学习、元学习与通用机器人：策略搜索、机器人控制与元学习。
- Gautham Vasan｜增量策略梯度与在线连续控制：增量策略梯度和连续控制。
- Dale Schuurmans｜表示、优化与可规划状态：把真梯度优化理论与含估计误差的 actor–critic 更新区分。
- Martha White｜可靠 off-policy 学习到长期控制：从明确目标推导 actor 更新，再分析近似与分布错配。
- Shibhansh Dohare｜持续更新特征，而非只保护参数：区分表征可训练性和策略分布坍塌。

## 最大熵控制

连续动作中，策略承担动作搜索。熵、双评论家与回放各有作用，不能合并为一个“稳定化技巧”。

Pendulum 做单位与动作范围检查；MuJoCo / DMC 控制需固定版本、动作重复与观测处理。

[教材](https://yingwen.io/zh/continual-rl/algorithms/soft-control/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-soft-control)

- Sergey Levine｜从现实机器人经验到可泛化控制：最大熵控制与现实经验学习。
- Hado van Hasselt｜稳定深度价值学习与持续 RL 基础：价值估计误差与双估计器方法的基础。
- Dale Schuurmans｜表示、优化与可规划状态：用 soft consistency 理解策略、价值与熵正则化目标的共同结构。

## 时间信用分配与资格迹

先分开前向目标、后向计算与离策略校正，再比较 Expected Traces 和梯度迹的估计对象。RTRL 传播递归敏感度；元学习传播更新规则的敏感度。

固定轨迹的代数与导数检查 → Markov/混叠路径反例 → 延迟反馈和部分可观测控制；MinAtar、MuJoCo 与严格流式协议分别比较。

[教材](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-credit)

- Richard S. Sutton｜从时间信用分配到持续经验型智能体：多步回报、资格迹与在线信用分配。
- A. Rupam Mahmood｜把学习放回物理时间：True-online TD 与在线更新。
- Khurram Javed｜有限算力下的持续构造与信用分配：资源受限的递归信用分配。
- Esraa Elelimy｜高效在线递归学习：高效在线递归学习。
- Kris De Asis｜多步价值学习与面向真实时间的机器人：连接抽样、期望与多步备份。

## 流式更新与稳定性

在线交互不等于严格 streaming。必须分别说明回放、批量、每步计算与持久存储。

固定每步算力的连续控制；同时记录延迟尾部、学习状态大小、输出变化与回报。

[教材](https://yingwen.io/zh/continual-rl/algorithms/streaming/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-streaming)

- A. Rupam Mahmood｜把学习放回物理时间：严格增量深度 RL 与在线稳定化。
- Mohamed Elsayed｜严格增量的深度强化学习：Streaming deep RL 的实现与评测。
- Gautham Vasan｜增量策略梯度与在线连续控制：增量 actor–critic。
- Joseph Modayil｜从感知结构到可验证预测知识：检查并行预测的每步计算。
- Kris De Asis｜多步价值学习与面向真实时间的机器人：逐步更新必须满足运行时预算。
- Arsalan Sharifnassab｜更新几何、步长适应与流式学习：从预期函数变化理解单样本稳定化。
- Sorina Lupu｜自适应控制与直接从经验学习的机器人：真实机器人中的延迟与逐步计算。
- Khurram Javed｜有限算力下的持续构造与信用分配：按每步成本而不只按参数量比较方法。

## 元学习与学习规则的适应

IDBD 适应步长，MAML 学初始化，context-based meta-RL 推断任务；内外层目标与数据权限不同。

漂移预测测试步长适应；跨任务元 RL 必须分训练任务、开发任务与未见测试任务。

[教材](https://yingwen.io/zh/continual-rl/algorithms/meta/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-meta)

- Richard S. Sutton｜从时间信用分配到持续经验型智能体：增量步长适应与元学习。
- Martha White｜可靠 off-policy 学习到长期控制：学习规则适应和稳定性。
- Chelsea Finn｜快速适应与机器人元学习：MAML 与快速适应。
- Luisa Zintgraf｜用潜变量推断进行快速适应：基于隐变量的元强化学习。
- Arsalan Sharifnassab｜更新几何、步长适应与流式学习：追踪学习规则对未来误差的影响。
- John D. Martin｜奖励表达、规划计算与环境中的记忆：通过后续学习效果适应规划采样。
- Sorina Lupu｜自适应控制与直接从经验学习的机器人：区分离线元学习与运行时适应。
- A. Rupam Mahmood｜把学习放回物理时间：跟踪逐特征学习速度，同时追问元参数和非平稳性。
- Khurram Javed｜有限算力下的持续构造与信用分配：区分学习表示的外层过程与在线更新率的自适应。

## 目标条件化与子任务构造

目标定义偏好，状态保留信息。学会达到给定目标，不等于学会构建有用目标。

先给定目标集合，再比较随机、人工与学得目标。计入发现和技能训练的成本。

[教材](https://yingwen.io/zh/continual-rl/construction/goals/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-goals)

- Satinder Singh｜内在动机、时间抽象与 agency：内在动机、子任务与智能体自主性。
- George Konidaris｜从技能出发构造符号世界：技能与可规划抽象。
- Khimya Khetarpal｜affordance、抽象与 CRL 框架：Affordance 与任务相关抽象。
- Benjamin Eysenbach｜从未来状态预测理解自监督控制：目标条件、自监督控制与未来状态。

## Options 与技能发现

单步动作推广为可变持续时间的策略。发现、学习、选择和终止 option 是不同子问题。

Four Rooms 检查覆盖与打断条件；迁移目标时同时报告预训练成本和主任务交互。

[教材](https://yingwen.io/zh/continual-rl/construction/options/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-options)

- Marlos C. Machado｜表示—技能—经验的循环：SR、eigenoptions 与表示驱动的技能发现。
- Doina Precup｜从 options 到可持续的抽象智能体：Options、时间抽象与层级控制。
- Pierre-Luc Bacon｜时间抽象与优化视角：Option-Critic 的端到端优化。
- André Barreto｜可组合的价值知识与技能：可组合价值知识与技能。
- Richard S. Sutton｜从时间信用分配到持续经验型智能体：理解随机时长行为的接口。
- Levi H. S. Lelis｜让学到的知识指导搜索：比较程序组件、可执行技能与有明确启动／终止语义的 options。
- Dieter Büchler｜高动态机器人中的在线适应：带时间的子目标

## Dyna 与模型学习

一次真实经验既更新模型，也支持额外规划。持续环境中应同时测模型陈旧程度和规划收益。

Blocking Maze / Shortcut Maze：固定真实交互，另报规划备份次数与墙钟。

[教材](https://yingwen.io/zh/continual-rl/algorithms/dyna/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-dyna)

- Richard S. Sutton｜从时间信用分配到持续经验型智能体：Dyna：直接学习、模型学习与规划的统一。
- David Silver｜规划、强化学习与经验规模：搜索、价值学习与模型支持的决策。
- Martin Müller｜长期价值与临时搜索记忆：比较真实经验、模型模拟和双记忆的不同更新来源。

## 转移模型与后果模型

模型不必重建全部观测。应预测规划真正需要的量，并测试模型误差如何改变决策。

解析模型作 oracle；学得模型与无模型同骨干对照。区分重建、价值和闭环误差。

[教材](https://yingwen.io/zh/continual-rl/construction/models/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-models)

- Dale Schuurmans｜表示、优化与可规划状态：表示、优化与模型的可规划性。
- Danijar Hafner｜潜在世界模型与行为想象：潜在世界模型与想象学习。
- Samuel Kessler｜task-agnostic 世界模型的持续适应：任务无关的持续世界模型。
- Matthew Schlegel｜预测式状态与表示支持规划：预测表示与规划接口。
- Randy Goebel｜知识表示、推理与可检验的解释：讨论学得模型表达了什么以及何时失效。
- Sorina Lupu｜自适应控制与直接从经验学习的机器人：辨认模型误差与闭环控制需求。

## 时间抽象与规划

训练时想象与决策时搜索不是同一算法接口。比较时要同时限定模型调用和环境交互。

目标条件任务比较不同搜索深度；在固定计算下测模型误差与计划执行偏差。

[教材](https://yingwen.io/zh/continual-rl/construction/planning/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-planning)

- David Silver｜规划、强化学习与经验规模：搜索与价值学习。
- George Konidaris｜从技能出发构造符号世界：抽象技能的符号规划。
- Levi H. S. Lelis｜让学到的知识指导搜索：学习引导搜索。
- Martin Müller｜长期价值与临时搜索记忆：搜索中的临时记忆与长期价值。
- John D. Martin｜奖励表达、规划计算与环境中的记忆：分配有限的模型查询与备份。
- Michael Bowling｜博弈、评测与 agency：将 DeepStack 的决策时重求解与跨经验的参数学习分账。
- Csaba Szepesvári｜统计效率和算法边界：区分可查询模型中的模拟效率、模型误差和真实交互效率。
- Dale Schuurmans｜表示、优化与可规划状态：联合评价学得的表示与可执行的规划，而不是只考察编码质量。
- Marlos C. Machado｜表示—技能—经验的循环：表示如何支持决策时规划
- Matthew Schlegel｜预测式状态与表示支持规划：表示支持多尺度规划

## 知识保留与再适应

记住旧任务和快速学习新任务可能冲突。冻结诊断与保持更新的再适应实验分别回答不同问题。

Continual World / COOM：逐任务学习矩阵、任务身份权限、首次习得和再学习速度。

[教材](https://yingwen.io/zh/continual-rl/algorithms/retention/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-retention)

- Sarath Chandar｜稳定适应与持续学习的方法共同体：稳定适应与持续学习。
- Eric Eaton｜可组合知识的 lifelong RL：终身策略学习与知识组合。
- Jorge A. Mendez｜模块化与知识组合：模块化持续学习。
- Alessandro Lazaric｜知识迁移的条件与负迁移：迁移条件与负迁移。
- Shibhansh Dohare｜持续更新特征，而非只保护参数：同时观察旧知识保留与新知识学习，避免只报告其中一个。
- Matthew E. Taylor｜迁移、教学与人类输入：迁移与负迁移

## 可塑性与特征更新

不遗忘不意味着还能学习。应在相同新数据预算下测试老网络、新网络与局部重置。

稳定世界中的训练年龄实验和外部变化实验分开；控制梯度尺度、优化器状态与特征容量。

[教材](https://yingwen.io/zh/continual-rl/algorithms/plasticity/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-plasticity)

- Shibhansh Dohare｜持续更新特征，而非只保护参数：持续反向传播与特征替换。
- Clare Lyle｜持续可训练性的机制解释：可塑性衰退的机制。
- Evgenii Nikishin｜早期经验偏置与部分重置：早期经验偏置与网络重置。
- Ghada Sokar｜休眠神经元与 ReDo：休眠神经元与 ReDo。
- Rishabh Agarwal｜统计可靠的 RL 比较与可塑性诊断：可靠比较与可塑性诊断。
- Richard S. Sutton｜从时间信用分配到持续经验型智能体：检验训练年龄对继续学习的影响。
- A. Rupam Mahmood｜把学习放回物理时间：把短期数值稳定与长期学习能力分成两个指标。

## 探索与经验选择

经验获取决定之后能学习什么。预测误差大可能只是噪声，未必代表学习进展或控制价值。

Procgen / MiniGrid 分离训练关卡和测试关卡；加入不可学习噪声目标检验课程选择。

[教材](https://yingwen.io/zh/continual-rl/algorithms/exploration/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-exploration)

- Peter Stone｜技能迁移、协作与自主课程：自动课程与技能迁移。
- Tim Rocktäschel｜开放式学习与环境生成：无监督环境设计与开放式学习。
- Ian Osband｜深度探索与认识不确定性：不确定性与深度探索。
- Pierre-Yves Oudeyer｜学习进步驱动的自主发展：学习进展驱动的自主发展。
- Jeff Clune｜从质量多样性到开放式能力增长：质量多样性与开放式搜索。
- Csaba Szepesvári｜统计效率和算法边界：用 bandit 上下界理解反馈不足和比较器选择，再进入状态会随动作改变的 RL。
- Marlos C. Machado｜表示—技能—经验的循环：技能如何改变探索
- Dieter Büchler｜高动态机器人中的在线适应：身体决定探索空间

## 持续学习的智能体架构

连接模块后才出现的数据分布反馈，需要单独验证。MARL 还要控制其他智能体变化与训练信息权限。

从两模块因子消融到完整持续环境。多智能体以独立训练团队为单位，补充 cross-play。

[教材](https://yingwen.io/zh/continual-rl/construction/architectures/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-architectures)

- Richard S. Sutton｜从时间信用分配到持续经验型智能体：从经验构造知识的整体架构。
- David Abel｜从表示抽象到“学习究竟在哪里”：智能体结构与持续学习定义。
- Marlos C. Machado｜表示—技能—经验的循环：表征、技能与经验生成的耦合。
- 温颖 Ying Wen｜相互影响、递归推理与合作决策：递归推理与多智能体相互影响。
- 汪军 Jun Wang｜博弈论、多智能体交互与规模化决策：多智能体交互与博弈。
- 俞扬 Yang Yu｜经验复用到持续多智能体协调：经验复用与多智能体协调。
- 章宗长 Zongzhang Zhang｜不确定环境中的策略迁移：部分可观测决策与策略迁移。
- 郝建业 Jianye Hao｜多智能体适应与可迁移深度 RL：多智能体适应与可迁移控制。
- 高阳 Yang Gao｜机器人学习与知识积累：机器人学习与知识积累。
- 朱军 Jun Zhu｜Bayesian 持续学习与主动遗忘：Bayesian 持续学习与遗忘机制。
- 张伟楠 Weinan Zhang｜可读 RL 教学、决策学习与 agent：深度 RL 教学与决策学习。
- 杨耀东 Yaodong Yang｜大规模合作、博弈与适应：合作、博弈与多智能体适应。
- Joseph Modayil｜从感知结构到可验证预测知识：让预测知识服务后续决策。
- Randy Goebel｜知识表示、推理与可检验的解释：连接学习模块、推理模块与解释接口。
- John D. Martin｜奖励表达、规划计算与环境中的记忆：把身体、环境和计算纳入系统边界。
- Levi H. S. Lelis｜让学到的知识指导搜索：把慢速表示学习、快速程序搜索和执行过程定义为不同接口。
- Martin Müller｜长期价值与临时搜索记忆：检验局部搜索与长期学习的接口，保留模块组合失败的负证据。
- Patrick M. Pilarski｜预测知识、人机共适应与身体：人机双方共同适应
- Matthew E. Taylor｜迁移、教学与人类输入：交互中人的适应

## 实验设计、统计与算法测试

从精确测试到受控学习曲线，再到多运行基准。每一层支持不同强度的结论。

预先定义随机单位、主指标与失败规则。持续学习的窗口不是独立训练样本。

[教材](https://yingwen.io/zh/continual-rl/experiments/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-experiments)

- Adam White｜从实时预测知识到可信实验：实验设计、性能变异与调参。
- Martha White｜可靠 off-policy 学习到长期控制：算法比较与基准方法论。
- Rishabh Agarwal｜统计可靠的 RL 比较与可塑性诊断：统计可靠的深度 RL 评价。
- Joseph Modayil｜从感知结构到可验证预测知识：分别测预测误差与控制效用。
- Randy Goebel｜知识表示、推理与可检验的解释：为解释的忠实性设计干预和对照。
- Kris De Asis｜多步价值学习与面向真实时间的机器人：比较控制频率、延迟与恢复成本。
- Arsalan Sharifnassab｜更新几何、步长适应与流式学习：拆分预条件、尺度控制与元学习的作用。
- Sorina Lupu｜自适应控制与直接从经验学习的机器人：同时报告学习、恢复和平台成本。
- Michael Bowling｜博弈、评测与 agency：从 ALE 与扑克评测学习平台设计、统计方差和可比较条件。
- Csaba Szepesvári｜统计效率和算法边界：让统计效率结论与实际实验的数据访问、资源预算保持一致。
- Levi H. S. Lelis｜让学到的知识指导搜索：测试组合泛化与跨任务摊销，同时保留预训练、建库和模拟成本。
- Martin Müller｜长期价值与临时搜索记忆：精确求解证据、有限样本胜率和持续适应曲线各自回答不同问题。
- A. Rupam Mahmood｜把学习放回物理时间：将机器人设置与生命期成本纳入比较。
- Esraa Elelimy｜高效在线递归学习：联合操纵可观测性与变化机制，而非只延长训练。
- Shibhansh Dohare｜持续更新特征，而非只保护参数：用同难度长期序列与替换对照识别机制。
- Marlos C. Machado｜表示—技能—经验的循环：固定技能与在线发现的对照
- Patrick M. Pilarski｜预测知识、人机共适应与身体：交互价值与使用者负担
- Matthew E. Taylor｜迁移、教学与人类输入：计入教师与源任务成本
- Dieter Büchler｜高动态机器人中的在线适应：恢复、损耗和外部干预

## 经典与深度强化学习：按正文查阅

### 经典强化学习

#### [多臂老虎机：估计、探索与直接策略学习](https://yingwen.io/zh/continual-rl/foundations/tabular/bandits/)

没有状态转移时，仍需一边估计动作收益，一边决定下一次尝试什么。这个最小问题把估计误差、探索代价和策略更新分开。

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-tabular-bandits) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-tabular-bandits) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-tabular-bandits)

#### [MDP、回报与价值：序列决策的数学对象](https://yingwen.io/zh/continual-rl/foundations/tabular/mdps/)

动作会改变后续状态时，需要评价整个未来。本章从随机交互过程推导价值与 Bellman 方程，明确后续算法共同使用的数学对象。

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-tabular-mdps) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-tabular-mdps) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-tabular-mdps)

#### [动态规划：评价、改善与最优递推](https://yingwen.io/zh/continual-rl/foundations/tabular/dynamic-programming/)

已知环境模型时，怎样通过局部计算得到长期价值和策略？本章将 Bellman 方程转化为迭代，并证明评价与改善之间的联系。

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-tabular-dynamic-programming) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-tabular-dynamic-programming) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-tabular-dynamic-programming)

#### [Monte Carlo：完整回报、探索控制与离策略评价](https://yingwen.io/zh/continual-rl/foundations/tabular/monte-carlo/)

不知道模型时，可以将完整回报作为样本。估计还会改变下一次行动：需要看清回报来自哪一版策略、探索怎样影响收益，以及长回合怎样消耗有效覆盖。

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-tabular-monte-carlo) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-tabular-monte-carlo) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-tabular-monte-carlo)

#### [TD 预测与控制：SARSA、Expected SARSA、Q-learning 和 Double Q](https://yingwen.io/zh/continual-rl/foundations/tabular/temporal-difference/)

如何在完整回报尚不可用时学习？TD 用下一预测补足未来；控制算法再根据不同的下一动作处理方式，形成不同的学习目标。

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-tabular-temporal-difference) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-tabular-temporal-difference) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-tabular-temporal-difference)

#### [多步学习：n-step、Tree Backup 与 Q(σ)](https://yingwen.io/zh/continual-rl/foundations/tabular/multistep/)

学习目标可以在一步 bootstrap 与完整回报之间选择，也可以在动作采样与动作期望之间选择。这是两条不同的设计维度。

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-tabular-multistep) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-tabular-multistep) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-tabular-multistep)

#### [学习与规划：Dyna、优先扫描和执行时搜索](https://yingwen.io/zh/continual-rl/foundations/tabular/planning/)

真实经验既能直接改进价值，也能训练后果模型。规划使用这个模型继续计算，关键是模型语义、backup 的成本以及计算应分配到哪里。

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-tabular-planning) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-tabular-planning) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-tabular-planning)

#### [函数逼近预测：从回归到 TD 固定点](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/)

共享少量参数以后，MC、TD 和最小二乘方法究竟在求解什么？

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-approximation-prediction) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-approximation-prediction) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-approximation-prediction)

#### [特征、泛化与半梯度控制](https://yingwen.io/zh/continual-rl/foundations/approximation/features-control/)

特征怎样改变学习行为，Sarsa 又怎样在共享参数下改善策略？

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-approximation-features-control) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-approximation-features-control) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-approximation-features-control)

#### [持续控制与平均奖励](https://yingwen.io/zh/continual-rl/foundations/approximation/average-control/)

智能体没有自然回合终点时，怎样定义和学习长期控制目标？

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-approximation-average-control) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-approximation-average-control) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-approximation-average-control)

#### [离策略函数逼近：覆盖、发散与稳定更新](https://yingwen.io/zh/continual-rl/foundations/approximation/off-policy/)

行为数据足够覆盖目标策略，为什么 TD 仍可能发散，又能怎样修复？

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-approximation-off-policy) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-approximation-off-policy) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-approximation-off-policy)

#### [多步回报、资格迹与 True-online TD](https://yingwen.io/zh/continual-rl/foundations/approximation/traces/)

当前到来的奖励怎样更新过去的预测，同时保留正确的在线更新语义？

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-approximation-traces) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-approximation-traces) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-approximation-traces)

#### [策略梯度、基线与 Actor–Critic](https://yingwen.io/zh/continual-rl/foundations/approximation/policy-gradient/)

直接学习策略时，哪一个目标的梯度能由经验估计，近似从哪里进入？

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-approximation-policy-gradient) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-approximation-policy-gradient) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-approximation-policy-gradient)

### 深度强化学习

#### [深度价值学习：DQN、Double DQN 与目标的时间顺序](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/)

把表格 Q-learning 换成网络后，损失、数据与目标为什么都需要重新组织？

- [Hado van Hasselt](https://yingwen.io/zh/continual-rl/resource/S61/)：Double Q-learning 与 Double DQN：区分动作选择和动作评价。
- [David Silver](https://yingwen.io/zh/continual-rl/resource/S62/)：深度价值学习与强化学习课程；对照数据、目标与策略的时间顺序。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-deep-value) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-deep-value) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-deep-value)

#### [策略梯度：从轨迹概率到 GAE 与 actor–critic](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/)

延迟奖励怎样改变动作概率？有限 rollout、critic 与停止梯度分别改变哪一项估计？

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：策略梯度定理与 actor–critic 的基础。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：演员—评论家的基本结构及其教材论述。
- [Pieter Abbeel](https://yingwen.io/zh/continual-rl/resource/S51/)：策略优化与深度强化学习课程。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-policy-gradient) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-policy-gradient) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-policy-gradient)

#### [策略更新的尺度：TRPO 与 PPO](https://yingwen.io/zh/continual-rl/foundations/deep/trust-region/)

旧策略的数据能支持多远的策略更新？怎样从局部代理走到采样、更新与独立评价？

- [Sergey Levine](https://yingwen.io/zh/continual-rl/resource/S49/)：TRPO 的共同作者；从代理目标与状态分布近似理解更新。
- [Pieter Abbeel](https://yingwen.io/zh/continual-rl/resource/S51/)：TRPO 的共同作者；区分理论约束、数值求解和 PPO 近似。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-trust-region) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-trust-region) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-trust-region)

#### [连续动作的价值优化：DDPG 与 TD3](https://yingwen.io/zh/continual-rl/foundations/deep/deterministic-control/)

不能枚举连续动作时，如何用 critic 的梯度改进 actor？

- [David Silver](https://yingwen.io/zh/continual-rl/resource/S62/)：确定性策略梯度；连续动作的价值导数。
- [Hado van Hasselt](https://yingwen.io/zh/continual-rl/resource/S61/)：双估计器与价值估计偏差，为比较 actor 所依赖的 critic 提供背景。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-deterministic-control) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-deterministic-control) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-deterministic-control)

#### [最大熵连续控制：SAC 的价值、密度与温度](https://yingwen.io/zh/continual-rl/foundations/deep/entropy-control/)

随机 actor 不只是加噪声：熵如何进入 Bellman 方程与自动微分？

- [Sergey Levine](https://yingwen.io/zh/continual-rl/resource/S49/)：SAC 的共同作者；最大熵目标、随机 actor 与离策略更新。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-entropy-control) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-entropy-control) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-entropy-control)

#### [深度 RL 的机制接口：模型、记忆、离线数据与实验](https://yingwen.io/zh/continual-rl/foundations/deep/practice/)

改变数据来源或 agent state 后，哪些推导和实现条件必须重新检查？

- [Rishabh Agarwal](https://yingwen.io/zh/continual-rl/resource/S109/)：深度 RL 的统计可靠性；从运行与采样单位理解结果比较。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-practice) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-practice) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-practice)

#### [大规模训练：算法与系统怎样共同设计](https://yingwen.io/zh/continual-rl/foundations/deep/systems/)

环境、推理和学习并行以后，怎样把更多计算变成更快的策略改善？

- [David Silver](https://yingwen.io/zh/continual-rl/resource/S62/)：A3C 的共同作者；并行采样与异步更新的算法接口。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-systems) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-systems) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-systems)

#### [不完全可观测：信念状态、信息行动与递归记忆](https://yingwen.io/zh/continual-rl/foundations/deep/partial-observability/)

当前观察不能决定未来时，智能体应记住什么，信息又如何影响行动？

- [Leslie Pack Kaelbling](https://yingwen.io/zh/continual-rl/resource/S43/)：POMDP 中的状态不确定性、信念更新与规划。
- [Michael L. Littman](https://yingwen.io/zh/continual-rl/resource/S42/)：POMDP 的求解与学习；对照历史、信念和行动条件。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-partial-observability) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-partial-observability) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-partial-observability)

#### [探索与不确定性：后验、乐观估计和时间一致行动](https://yingwen.io/zh/continual-rl/foundations/deep/exploration/)

为什么每一步都随机，并不等于有效获取长期有用的信息？

- [Ian Osband](https://yingwen.io/zh/continual-rl/resource/S64/)：后验采样与深度探索；时间一致的探索行为。
- [Benjamin Van Roy](https://yingwen.io/zh/continual-rl/resource/S46/)：信息、后验采样与不确定决策。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-exploration) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-exploration) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-exploration)

#### [分布强化学习：Bellman 分布、分位数与风险目标](https://yingwen.io/zh/continual-rl/foundations/deep/distributional/)

学习完整回报分布，与学习均值、评估风险和估计知识不确定性分别有什么关系？

- [Marc G. Bellemare](https://yingwen.io/zh/continual-rl/resource/S39/)：分布强化学习；回报分布与 Bellman 更新。
- [Will Dabney](https://yingwen.io/zh/continual-rl/resource/S65/)：分位数回归及价值分布的表示。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-distributional) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-distributional) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-distributional)

#### [离线强化学习：数据支持、策略评估与保守改进](https://yingwen.io/zh/continual-rl/foundations/deep/offline/)

不能补采数据时，怎样判断策略好坏，怎样避免利用没有证据的高价值动作？

- [Sergey Levine](https://yingwen.io/zh/continual-rl/resource/S49/)：CQL、IQL 等离线 RL 工作；数据支持与策略改进。
- [Nan Jiang](https://yingwen.io/zh/continual-rl/resource/S55/)：离策略评价及其信息和统计条件。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-offline) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-offline) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-offline)

#### [模型学习与规划：MPC、短模型 rollout 和潜在想象](https://yingwen.io/zh/continual-rl/foundations/deep/model-based/)

模型在哪里进入决策，预测误差又怎样变成控制误差？

- [Danijar Hafner](https://yingwen.io/zh/continual-rl/resource/S104/)：Dreamer 系列；潜在动力学中的行为学习与想象。
- [Sergey Levine](https://yingwen.io/zh/continual-rl/resource/S49/)：PETS、MBPO 等模型控制工作；模型误差与数据使用。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-model-based) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-model-based) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-model-based)

#### [约束强化学习：占据测度、拉格朗日与可行策略](https://yingwen.io/zh/continual-rl/foundations/deep/constraints/)

“回报高且代价不超过预算”与“每一步都安全”之间差了哪些条件？

- [Pieter Abbeel](https://yingwen.io/zh/continual-rl/resource/S51/)：Constrained Policy Optimization 的共同作者；约束、近似和实际策略更新。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-constraints) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-constraints) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-constraints)

#### [多智能体合作：结构化探索与信用分配](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent/)

团队共享一个奖励时，怎样从联合经验中学习可执行的协作策略，并正确处理同伴更新？

- [温颖 Ying Wen](https://yingwen.io/zh/continual-rl/resource/S110/)：合作决策与 MAT 等工作；联合动作结构与学习目标。
- [汪军 Jun Wang](https://yingwen.io/zh/continual-rl/resource/S111/)：多智能体合作学习与联合决策。
- [Jakob Foerster](https://yingwen.io/zh/continual-rl/resource/S58/)：反事实基线与合作信用分配。
- [Shimon Whiteson](https://yingwen.io/zh/continual-rl/resource/S59/)：价值分解和合作多智能体学习。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-multi-agent) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-multi-agent) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-multi-agent)

#### [自对弈与开放式多智能体学习：评估、目标与策略种群](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent-populations/)

自对弈怎样产生课程和训练标签，又怎样通过历史保留、交互评价与策略种群发现值得继续学习的问题？

- [温颖 Ying Wen](https://yingwen.io/zh/continual-rl/resource/S110/)：开放式多智能体学习及合作、竞争场景。
- [汪军 Jun Wang](https://yingwen.io/zh/continual-rl/resource/S111/)：种群学习与博弈中的评估和目标构建。
- [Michael Bowling](https://yingwen.io/zh/continual-rl/resource/S17/)：博弈评估与不完全信息决策。
- [David Silver](https://yingwen.io/zh/continual-rl/resource/S62/)：AlphaGo 系列：自对弈、搜索和策略价值学习。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-marl-populations) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-marl-populations) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-marl-populations)

#### [对手建模与递归推理：预测谁，回应什么？](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent-reasoning/)

给定参与者和评价目标，怎样利用行为预测、条件响应与有限递归改善决策，并检验模型是否可信？

- [温颖 Ying Wen](https://yingwen.io/zh/continual-rl/resource/S110/)：PR2、GR2 等递归策略推理工作。
- [汪军 Jun Wang](https://yingwen.io/zh/continual-rl/resource/S111/)：相互影响与递归博弈建模。
- [Stefano V. Albrecht](https://yingwen.io/zh/continual-rl/resource/S68/)：其他智能体的行为模型与适应。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-marl-reasoning) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-marl-reasoning) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-marl-reasoning)
