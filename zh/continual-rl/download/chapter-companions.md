# 章节、实验、资源与学者

## 交互、奖励与优化目标

折扣与平均奖励是目标选择。持续学习不能仅靠把回合接长来定义。

先用可解析的循环 MDP 检查策略排序，再讨论更复杂环境。

[教材](https://yingwen.io/zh/continual-rl/foundations/objectives/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-objectives)

- Richard S. Sutton｜持续经验型智能体的总纲：奖励、回报与持续交互的统一问题表述。
- Andrew G. Barto｜自适应控制、内在动机与层级学习：自适应控制与强化学习问题的形成。
- David Abel｜从表示抽象到“学习究竟在哪里”：持续学习定义与智能体内部学习过程。

## 平均奖励与差分价值

平均奖励按原始时间计收益。随机时长 option 要使用半马尔可夫时间口径。

已知平稳分布的循环 MDP；再用持续控制任务检查暂态、长期奖励率与漂移。

[教材](https://yingwen.io/zh/continual-rl/algorithms/average-reward/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-average)

- Martha White｜可靠 off-policy 学习到长期控制：平均奖励、可靠 off-policy 学习与长期控制。
- Adam White｜从实时预测知识到可信实验：平均奖励学习与经验研究。
- Richard S. Sutton｜持续经验型智能体的总纲：差分价值与 continuing control。

## 智能体状态与递归学习

深度网络不能自动消除部分可观测性。必须区分观测编码、历史状态递推与参数学习。

POPGym 隔离记忆需求；Forager 检查长期交互。对照完整状态 oracle 时标明额外信息权限。

[教材](https://yingwen.io/zh/continual-rl/construction/state/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-state)

- Matthew Schlegel｜预测式状态与表示支持规划：预测状态与可规划表示。
- Esraa Elelimy｜高效在线递归学习：递归网络的在线训练。
- Leslie Pack Kaelbling｜部分可观测与层级机器人规划：部分可观测决策与信念状态。
- Amy Zhang｜可泛化与分层决策所需的表示：面向决策的状态表示。

## 价值预测与资格迹

表格 TD 的局部更新推广为共享特征上的参数更新。GVF 再改变预测问题，而非只改变网络。

Random Walk 与小型马尔可夫奖励过程：有解析真值，便于分开估计误差和程序错误。

[教材](https://yingwen.io/zh/continual-rl/algorithms/value/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-value)

- Richard S. Sutton｜持续经验型智能体的总纲：TD 学习与资格迹。
- Andrew G. Barto｜自适应控制、内在动机与层级学习：价值学习与演员—评论家框架。
- Peter Dayan｜预测表征与计算神经科学：Successor representation 连接预测与表征。

## 通用价值函数与预测知识

标量奖励价值是 GVF 的一种特例。定义多个问题不等于已经学出有用状态或控制策略。

可解析 off-policy MRP 检查预测真值；机器人信号预测需独立测决策效用。

[教材](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-gvf)

- Joseph Modayil｜由可验证预测构成知识：以通用价值函数组织预测知识。
- Patrick M. Pilarski｜预测知识、人机共适应与身体：机器人预测知识与人机共适应。
- Adam White｜从实时预测知识到可信实验：并行离策略预测与实时经验学习。
- Martha White｜可靠 off-policy 学习到长期控制：稳定的离策略预测方法。

## 控制问题与广义策略迭代

评价固定策略与改进策略不是同一问题。状态、预测或技能的改进必须最终接受行为收益检验。

Cliff Walking / 小型 Gridworld：记录访问覆盖、策略和状态价值，而不只记录总分。

[教材](https://yingwen.io/zh/continual-rl/algorithms/control/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-control)

- Chris Watkins｜Q-learning 与动物学习的反思：Q-learning：从行为数据学习最优动作价值。
- Richard S. Sutton｜持续经验型智能体的总纲：广义策略迭代与控制。
- Csaba Szepesvári｜统计效率和算法边界：强化学习的统计效率与理论边界。

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

## 最大熵控制

连续动作中，策略承担动作搜索。熵、双评论家与回放各有作用，不能合并为一个“稳定化技巧”。

Pendulum 做单位与动作范围检查；MuJoCo / DMC 控制需固定版本、动作重复与观测处理。

[教材](https://yingwen.io/zh/continual-rl/algorithms/soft-control/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-soft-control)

- Sergey Levine｜从现实机器人经验到可泛化控制：最大熵控制与现实经验学习。
- Hado van Hasselt｜稳定深度价值学习与持续 RL 基础：价值估计误差与双估计器方法的基础。

## 时间信用分配与资格迹

资格迹将当前误差分配给过去的预测。元学习则学习如何更新参数。二者可组合，但估计对象不同。

固定短轨迹做前后向一致性检查；延迟反馈任务再测有限资源下的学习速度。

[教材](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-credit)

- Richard S. Sutton｜持续经验型智能体的总纲：多步回报、资格迹与在线信用分配。
- A. Rupam Mahmood｜把学习放回物理时间：True-online TD 与在线更新。
- Khurram Javed｜有限算力下的持续构造与信用分配：资源受限的递归信用分配。
- Esraa Elelimy｜高效在线递归学习：高效在线递归学习。

## 流式更新与稳定性

在线交互不等于严格 streaming。必须分别说明回放、批量、每步计算与持久存储。

固定每步算力的连续控制；同时记录延迟尾部、学习状态大小、输出变化与回报。

[教材](https://yingwen.io/zh/continual-rl/algorithms/streaming/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-streaming)

- A. Rupam Mahmood｜把学习放回物理时间：严格增量深度 RL 与在线稳定化。
- Mohamed Elsayed｜严格增量的深度强化学习：Streaming deep RL 的实现与评测。
- Gautham Vasan｜增量策略梯度与在线连续控制：增量 actor–critic。

## 元学习与学习规则的适应

IDBD 适应步长，MAML 学初始化，context-based meta-RL 推断任务；内外层目标与数据权限不同。

漂移预测测试步长适应；跨任务元 RL 必须分训练任务、开发任务与未见测试任务。

[教材](https://yingwen.io/zh/continual-rl/algorithms/meta/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-meta)

- Richard S. Sutton｜持续经验型智能体的总纲：增量步长适应与元学习。
- Martha White｜可靠 off-policy 学习到长期控制：学习规则适应和稳定性。
- Chelsea Finn｜快速适应与机器人元学习：MAML 与快速适应。
- Luisa Zintgraf｜用潜变量推断进行快速适应：基于隐变量的元强化学习。

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

## Dyna 与模型学习

一次真实经验既更新模型，也支持额外规划。持续环境中应同时测模型陈旧程度和规划收益。

Blocking Maze / Shortcut Maze：固定真实交互，另报规划备份次数与墙钟。

[教材](https://yingwen.io/zh/continual-rl/algorithms/dyna/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-dyna)

- Richard S. Sutton｜持续经验型智能体的总纲：Dyna：直接学习、模型学习与规划的统一。
- David Silver｜规划、强化学习与经验规模：搜索、价值学习与模型支持的决策。

## 转移模型与后果模型

模型不必重建全部观测。应预测规划真正需要的量，并测试模型误差如何改变决策。

解析模型作 oracle；学得模型与无模型同骨干对照。区分重建、价值和闭环误差。

[教材](https://yingwen.io/zh/continual-rl/construction/models/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-models)

- Dale Schuurmans｜表示、优化与可规划状态：表示、优化与模型的可规划性。
- Danijar Hafner｜潜在世界模型与行为想象：潜在世界模型与想象学习。
- Samuel Kessler｜task-agnostic 世界模型的持续适应：任务无关的持续世界模型。
- Matthew Schlegel｜预测式状态与表示支持规划：预测表示与规划接口。

## 时间抽象与规划

训练时想象与决策时搜索不是同一算法接口。比较时要同时限定模型调用和环境交互。

目标条件任务比较不同搜索深度；在固定计算下测模型误差与计划执行偏差。

[教材](https://yingwen.io/zh/continual-rl/construction/planning/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-planning)

- David Silver｜规划、强化学习与经验规模：搜索与价值学习。
- George Konidaris｜从技能出发构造符号世界：抽象技能的符号规划。
- Levi H. S. Lelis｜让学到的知识指导搜索：学习引导搜索。
- Martin Müller｜长期价值与临时搜索记忆：搜索中的临时记忆与长期价值。

## 知识保留与再适应

记住旧任务和快速学习新任务可能冲突。冻结诊断与继续学习的再适应实验分别回答不同问题。

Continual World / COOM：逐任务学习矩阵、任务身份权限、首次习得和再学习速度。

[教材](https://yingwen.io/zh/continual-rl/algorithms/retention/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-retention)

- Sarath Chandar｜稳定适应与持续学习的方法共同体：稳定适应与持续学习。
- Eric Eaton｜可组合知识的 lifelong RL：终身策略学习与知识组合。
- Jorge A. Mendez｜模块化与知识组合：模块化持续学习。
- Alessandro Lazaric｜知识迁移的条件与负迁移：迁移条件与负迁移。

## 可塑性与特征更新

不遗忘不意味着还能学习。应在相同新数据预算下测试老网络、新网络与局部重置。

稳定世界中的训练年龄实验和外部变化实验分开；控制梯度尺度、优化器状态与特征容量。

[教材](https://yingwen.io/zh/continual-rl/algorithms/plasticity/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-plasticity)

- Shibhansh Dohare｜持续更新特征，而非只保护参数：持续反向传播与特征替换。
- Clare Lyle｜持续可训练性的机制解释：可塑性衰退的机制。
- Evgenii Nikishin｜早期经验偏置与部分重置：早期经验偏置与网络重置。
- Ghada Sokar｜休眠神经元与 ReDo：休眠神经元与 ReDo。
- Rishabh Agarwal｜统计可靠的 RL 比较与可塑性诊断：可靠比较与可塑性诊断。

## 探索与经验选择

经验获取决定之后能学习什么。预测误差大可能只是噪声，未必代表学习进展或控制价值。

Procgen / MiniGrid 分离训练关卡和测试关卡；加入不可学习噪声目标检验课程选择。

[教材](https://yingwen.io/zh/continual-rl/algorithms/exploration/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-exploration)

- Peter Stone｜技能迁移、协作与自主课程：自动课程与技能迁移。
- Tim Rocktäschel｜开放式学习与环境生成：无监督环境设计与开放式学习。
- Ian Osband｜深度探索与认识不确定性：不确定性与深度探索。
- Pierre-Yves Oudeyer｜学习进步驱动的自主发展：学习进展驱动的自主发展。
- Jeff Clune｜从质量多样性到开放式能力增长：质量多样性与开放式搜索。

## 持续学习的智能体架构

连接模块后才出现的数据分布反馈，需要单独验证。MARL 还要控制其他智能体变化与训练信息权限。

从两模块因子消融到完整持续环境。多智能体以独立训练团队为单位，补充 cross-play。

[教材](https://yingwen.io/zh/continual-rl/construction/architectures/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-architectures)

- Richard S. Sutton｜持续经验型智能体的总纲：从经验构造知识的整体架构。
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

## 实验设计、统计与算法测试

从精确测试到受控学习曲线，再到多运行基准。每一层支持不同强度的结论。

预先定义随机单位、主指标与失败规则。持续学习的窗口不是独立训练样本。

[教材](https://yingwen.io/zh/continual-rl/experiments/) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-experiments)

- Adam White｜从实时预测知识到可信实验：实验设计、性能变异与调参。
- Martha White｜可靠 off-policy 学习到长期控制：算法比较与基准方法论。
- Rishabh Agarwal｜统计可靠的 RL 比较与可塑性诊断：统计可靠的深度 RL 评价。