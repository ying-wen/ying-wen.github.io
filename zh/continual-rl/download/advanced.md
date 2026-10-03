# CRL 分方向进阶阅读与学习材料

更新：2026-09-30。起点是[入门主线](/zh/continual-rl/reading/intro/)中的五份共同材料。

**使用方法：只选一条路线，按顺序读前两项，再决定是否深入。** 每条路线中的“进一步”都是选修。这里按研究问题划分，不是互斥、完备或全社区统一的分类。

注意材料的三种角色：**直接研究 CRL 的工作、可被 CRL 使用的方法基础、启发研究问题的交叉材料**。三者都值得学，但不能混作已经解决 CRL 的证据。

## 先看方向之间的关系

```text
长期交互中的智能体
├─ 持续学得动：遗忘与可塑性（1）、流式更新与在线优化（2）
├─ 知道该记什么：状态构造与预测知识（3）
├─ 形成可复用能力：时间抽象与技能（4）、世界模型与规划（5）
├─ 决定下一步学什么：探索、课程与开放式学习（6）
├─ 面对真实约束：single-life / reset-free（7）
└─ 拓展问题与方法：理论（8）、元学习（9）、多智能体（10）

所有路线共用：清楚的实验协议、长期评价、计算与存储预算。
```

<a id="track-1"></a>
## 1. 遗忘与可塑性：旧知识和新学习能力如何兼得？

**先修**：基本深度 RL、梯度下降。**脉络**：保住已学知识 → 发现“学不动”是另一个问题 → 研究表示更新与知识保留之间的冲突。

1. **先读保留机制：[CLEAR / Experience Replay for Continual Learning](https://arxiv.org/abs/1811.11682)，Rolnick 等，NeurIPS 2019。** 回放旧经验，结合新经验学习与行为克隆来减轻遗忘。重点看为什么只有 RL 更新和加入行为保持约束会不同，以及 replay 容量会付出什么代价。
2. **再读学习能力：[Loss of Plasticity](https://www.nature.com/articles/s41586-024-07711-7)，Dohare 等，Nature 2024。** 在入门阅读基础上细看 Continual Backprop 的单元效用、替换与实验对照。把“新任务更难”和“网络经历更长后更难学”分开。
3. **再比较一种干预：[The Dormant Neuron Phenomenon in Deep Reinforcement Learning](https://proceedings.mlr.press/v202/sokar23a.html)，Sokar 等，ICML 2023。** ReDo 回收低活动单元。比较它与 Continual Backprop 的筛选依据和操作细节；二者不能简单视为同一个算法。

**进一步二选一**：想研究机制，读 [Understanding Plasticity in Neural Networks](https://proceedings.mlr.press/v202/lyle23b.html)（Lyle 等，ICML 2023），从优化和学习动态理解可塑性；想研究知识保护，读 [EWC](https://arxiv.org/abs/1612.00796)（Kirkpatrick 等，PNAS 2017），理解按参数重要性限制更新的思路。EWC 是基础对照，不代表当前最佳方案。

**代码 / 实验**：[Continual Backprop 官方代码](https://github.com/shibhansh/loss-of-plasticity)；[CORA](https://github.com/AGI-Labs/continual_rl)含 CLEAR 基线与任务序列实验。先做固定网络容量下的对照，同时测旧任务保留和新任务学习速度。

**读后追问**：神经元替换改善了新学习，却损害旧知识，怎样确定净收益？回报改善究竟来自恢复可塑性、正则化，还是额外计算？

**关注作者线索**：Dohare、Mahmood、Sutton；Lyle、Sokar、Castro、Evci；Rolnick、Pascanu。沿机制和共同作者追踪，不必先读他们的全部论文。

<a id="track-2"></a>
## 2. 流式学习与在线优化：每来一步经验，怎样可靠地更新？

**先修**：Sutton 第 9、11、12 章的相关内容；actor–critic。**脉络**：表格增量更新 → 函数逼近稳定性 → 深度网络中的实时更新、尺度控制和自适应步长。

1. **深度 RL 入口：[Streaming Deep Reinforcement Learning Finally Works](https://arxiv.org/abs/2410.14606)，Elsayed 等，2024 起，2026 修订。** 看直接把常见算法改为流式更新会出现什么问题，以及 Stream-X 如何组合归一化、表示稳定与受控更新。重点不是背组件，而是逐项解释它修复哪个失败模式。
2. **连续控制路线：[Deep Policy Gradient Methods Without Batch Updates, Target Networks, or Replay Buffers](https://arxiv.org/abs/2411.15370)，Vasan 等，NeurIPS 2024。** 学习 AVG 的增量策略梯度设计，比较它与熟悉的深度 actor–critic 的数据使用方式。论文使用了无 replay 的更新，并不等于整个系统已经实现无限期自主学习。
3. **在线优化基础：[SwiftTD](https://rlj.cs.umass.edu/2024/papers/Paper111.html)，Javed、Sharifnassab、Sutton，RLJ 2024。** 将自适应步长与学习速率约束放在 TD 学习中理解。先在线性函数逼近里看清更新，再考虑如何移入深度网络；不能把线性预测结果直接推广为深度控制保证。

**代码 / 实验**：[Stream-X](https://github.com/mohmdelsayed/streaming-drl)、[SwiftTD](https://github.com/kjaved0/swifttd)。先用一个小环境，记录回报、更新幅度和单步计算耗时，再做单组件消融。

**版本提醒**：Stream-X 的 arXiv v3 为 2026-09-21 修订，官方默认代码已标为 2026 实现；2024 实现在 `2024` 分支。复现时必须记录论文版本与代码 commit，不能混用。

**读后追问**：在相同环境步数、实际计算预算和调参权限下，稳定性还成立吗？算法在平稳环境学得好，是否足以说明它能适应长期变化？

**关注作者线索**：Mahmood、Elsayed、Vasan、Javed、Sutton，以及 Martha White / Adam White 的在线学习研究。

<a id="track-3"></a>
## 3. 状态构造与预测知识：智能体到底应该记住什么？

**先修**：TD、off-policy、RNN；知道观测不一定是 Markov 状态。**脉络**：学多个未来预测 → 用预测约束内部表示 → 以可负担的实时信用分配学习记忆。

1. **预测知识入口：[Horde](https://josephmodayil.com/papers/horde-final.pdf)，Sutton、Modayil 等，AAMAS 2011。** 把多个关于未来的预测或控制问题写成广义价值函数，利用同一条经验流并行学习。要能自己写出一个问题的策略、累积信号与终止条件，而不只是说“增加 auxiliary loss”。
2. **预测与状态结合：[General Value Function Networks](https://arxiv.org/abs/1807.06763)，Schlegel 等，JAIR 2021。** 让内部状态的组成部分对应未来预测，研究这种约束能否改善递归网络训练。重点辨析“额外预测头”和“用预测构造内部状态”的不同。
3. **实时实现路线：[Real-Time Recurrent Learning using Trace Units in Reinforcement Learning](https://arxiv.org/abs/2409.01449)，Elelimy、Adam White、Bowling、Martha White，NeurIPS 2024。** RTU 从结构上降低实时递归学习的成本。比较 RTRL 与截断 BPTT 在梯度、缓存、延迟和计算上的差异。

**进一步**：[Auxiliary Task Discovery through Generate-and-Test](https://proceedings.mlr.press/v232/rafiee23a.html)，Rafiee 等，CoLLAs 2023。前面的预测问题大多需要设计；这篇开始问：辅助任务能否自己生成、评价和替换？

**代码 / 实验**：[RTU 官方实现](https://github.com/esraaelelimy/rtus)、[POPGym](https://github.com/proroklab/popgym)。先选一个需要记忆的小任务，再比较固定参数下的状态适应与持续参数学习。POPGym 本身是部分可观测记忆测试，不自动构成非平稳 CRL 协议。

**读后追问**：预测更准一定意味着控制更好吗？当表示本身持续变化时，旧的价值估计和知识如何继续有用？

**关注作者线索**：Sutton、Modayil、Pilarski、Schlegel、Elelimy、Martha White、Adam White。

<a id="track-4"></a>
## 4. 时间抽象与技能迁移：怎样让过去学到的东西继续有用？

**先修**：MDP、Bellman 方程、基本线性代数。**脉络**：定义技能 → 表示可复用的未来行为后果 → 自动发现并组合技能。

1. **基础概念：[Between MDPs and Semi-MDPs](https://doi.org/10.1016/S0004-3702(99)00052-1)，Sutton、Precup、Singh，AIJ 1999。** 理解 option 的启动集合、内部策略和终止条件，以及为什么多步技能改变了决策时间尺度。这是方法基础，不是 CRL 完整方案。
2. **迁移机制：[Successor Features for Transfer in Reinforcement Learning](https://arxiv.org/abs/1606.05312)，Barreto 等，NeurIPS 2017。** 将未来特征占用与奖励权重分开，并通过 generalized policy improvement 复用已有策略。先弄清动力学保持不变、奖励结构变化等条件，再讨论泛化范围。
3. **研究主线：[Temporal Abstraction in Reinforcement Learning with the Successor Representation](https://www.jmlr.org/papers/v24/21-1213.html)，Machado、Barreto、Precup、Bowling，JMLR 2023。** 连接表示、option discovery、探索和技能组合，理解 ROD 中表示与技能相互改进的思路。长文第一遍读框架与关键实验，不必逐个推导全读。

**讲座 / 代码**：先看 [Machado 的课程录像与 slides](https://deeprlcourse.github.io/guests/marlos_machado/)，再读长文；[eigenoptions 代码](https://github.com/mcmachado/options)可配合 [Laplacian option discovery 论文](https://proceedings.mlr.press/v70/machado17a.html)读。旧研究代码适合理解机制，移植和运行兼容性需另行检查。

**读后追问**：有用技能是由当前奖励、覆盖空间、未来任务还是规划效率定义的？环境动力学变化后，旧 successor features 与旧 option 还能保留多少价值？

**关注作者线索**：Machado、Precup、Barreto、Konidaris、Abel、Bacon。技能抽象与符号规划的连接，可再沿 Konidaris 的工作展开。

<a id="track-5"></a>
## 5. 世界模型与规划：经验有限、模型不准，还值得规划吗？

**先修**：Part I 第 8 章 Dyna；深度世界模型还需要潜变量与序列建模。**脉络**：用模型重用经验 → 控制模型误差 → 持续更新模型、数据记忆与策略。

1. **从已知基础过渡：[Selective Dyna-Style Planning Under Limited Model Capacity](https://proceedings.mlr.press/v119/abbas20a.html)，Abbas、Sokota、Talvitie、Martha White，ICML 2020。** 不完美的模型可能帮助，也可能伤害学习。研究何时使用模型、何时少规划，比单纯追求更低预测误差更接近控制问题。
2. **深度模型参考：[DreamerV3 / Mastering Diverse Control Tasks through World Models](https://danijar.com/project/dreamerv3/)，Hafner 等，Nature 2025。** 看潜在状态、想象轨迹与策略学习如何组成闭环。它是强大的 model-based RL 方法基础；跨任务使用同一套超参数，不等于同一智能体在一生中不断积累所有任务。
3. **进入 CRL：[The Effectiveness of World Models for Continual Reinforcement Learning](https://arxiv.org/abs/2211.15944)，Kessler 等，CoLLAs 2023。** Continual-Dreamer 研究回放选择、遗忘、迁移和探索。与上一项是问题上的连接，不要误认为它必然基于后来发表的 DreamerV3 实现。

**进一步**：[Goal-Space Planning with Subgoal Models](https://www.jmlr.org/papers/v25/24-0040.html)，Lo 等，JMLR 2024。把规划放在子目标及局部模型上，连接时间抽象与模型误差控制；适合接路线 4。

**代码 / 实验**：[DreamerV3 官方代码](https://github.com/danijar/dreamerv3)。初学本方向也可以只用小型 Dyna：在改变一处转移规则后，比较不同规划量如何帮助或放大过时模型的错误，不必首先跑大型视觉模型。

**读后追问**：环境变化后，应该更新世界模型、改变回放样本、减少规划，还是重学策略？怎样通过对照分离这些影响？

**关注作者线索**：Martha White、Talvitie、Hafner、Kessler、Parker-Holder、Miłoś，以及做抽象模型与规划的 Machado、Konidaris。

<a id="track-6"></a>
## 6. 探索、自动课程与开放式学习：下一步应该学什么？

**先修**：探索—利用权衡；最好熟悉一种 policy gradient。**脉络**：改变动作探索 → 改变训练任务分布 → 不断生成新挑战与能力。

1. **先建立地图：[Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey](https://www.jmlr.org/papers/v21/20-212.html)，Narvekar、Peng、Leonetti、Sinapov、Taylor、Stone，JMLR 2020。** 分清谁生成任务、谁安排顺序、什么知识在任务间迁移。第一遍只读分类、评价与开放问题。
2. **再看一个可理解的系统：[Evolving Curricula with Regret-Based Environment Design（ACCEL）](https://arxiv.org/abs/2203.01302)，Parker-Holder 等，2022。** 从[项目演示](https://accelagent.github.io/)理解如何在能力边界附近改造和选择环境。重点看课程生成、选择与学生更新之间的反馈，而不只看最终难关。
3. **再走向开放式研究：[Enhanced POET](https://proceedings.mlr.press/v119/wang20l.html)，Rui Wang、Lehman、Clune、Stanley 等，ICML 2020。** 让环境与求解者共同演化，并通过跨任务迁移突破单一路径。它涉及多个求解者，不能直接等同于一个固定资源智能体的单次生命学习。

**探索支线**：[Random Network Distillation](https://arxiv.org/abs/1810.12894)，Burda 等，2018 预印本入口。学习用预测误差构造探索奖励，再思考在持续变化中“新颖”与“值得学”是否相同。

**讲座 / 工具**：[Peter Stone：Continual RL and Automatic Curriculum Learning](https://www.youtube.com/watch?v=5D5rJJp5jnw)；[Syllabus](https://github.com/RyanNavillus/Syllabus)提供课程接口与示例，可从 README 链接的 demo 开始，不必自己先写完整课程框架。

**读后追问**：方法是在训练阶段找到更好的课程，还是在部署的一生中持续决定学什么？课程成本、被丢弃环境和额外训练的代价算进去了吗？

**关注作者线索**：Stone、Narvekar、Taylor、Clune、Stanley、Parker-Holder、Dennis。

<a id="track-7"></a>
## 7. Single-life 与 reset-free：没有免费重来的机会怎么办？

**先修**：actor–critic、连续控制；必要时补基本反馈控制。**脉络**：看见人工重置的隐藏成本 → 学会自主恢复 → 面对长期、安全和不可逆的交互后果。

1. **讲座入口：[Rupam Mahmood：Challenges of Single-Life Continual RL](https://www.youtube.com/watch?v=G-LgYNEIwBE)。** 带着一个问题看：哪些常见实验便利，在真实系统的一生里不再免费？这是一份问题导向材料，不是具体算法的替代说明书。
2. **恢复策略：[Leave No Trace](https://arxiv.org/abs/1711.06782)，Eysenbach、Gu、Ibarz、Levine，2017 预印本入口。** 同时学习前进策略和 reset 策略，减少人工干预，并利用恢复能力判断何时中止。这让“如何继续得到训练经验”成为学习问题本身。
3. **任务互相创造条件：[Reset-Free Reinforcement Learning via Multi-Task Learning](https://arxiv.org/abs/2104.11203)，Gupta 等，ICRA 2021。** 利用一项任务结束后的状态为另一项任务提供起点。重点理解任务如何组织，而不是把 episode 边界简单删除。

**实验入口**：先在可控仿真中显式记录恢复操作、失败和人工干预的代价。论文页中的演示适合理解设置；不建议把真实机器人试错作为第一项入门复现。

**读后追问**：学会重置仍然消耗时间和动作，这些成本计入总回报了吗？reset-free 解决了经验采集自主性，是否同时解决遗忘、可塑性和长期目标变化？

**边界**：single-life、reset-free、安全 RL、机器人 lifelong learning 有交集，但不是同义词；“不调用环境 reset”本身并不足以建立完整的 single-life 评价。

**关注作者线索**：Mahmood、Gupta、Levine、Eysenbach；机器人长期知识保留方向可衔接高阳的相关工作。

<a id="track-8"></a>
## 8. 理论与长期目标：什么才算一生中表现得好？

**先修**：MDP、概率、基本证明能力；regret 路线建议先补 bandit 理论。这里有三条平行支线，不必全部精读。

1. **长期目标：[Learning and Planning in Average-Reward Markov Decision Processes](https://proceedings.mlr.press/v139/wan21a.html)，Wan、Naik、Sutton，ICML 2021。** 接 Sutton 第 10.3–10.5 节，学习 differential value、平均奖励估计与学习/规划算法。平均回报是长期目标的一种选择；平稳平均回报 MDP 本身不自动等于 CRL。
2. **外部变化：[Reinforcement Learning for Non-Stationary MDPs: The Blessing of (More) Optimism](https://proceedings.mlr.press/v119/cheung20a.html)，Cheung、Simchi-Levi、Zhu，ICML 2020。** 看 variation budget 如何限制环境变化，以及滑动窗口和 optimism 如何服务于 dynamic regret。读定理时先圈出比较对象、变化预算与 MDP 假设。
3. **重审形式化：[Rethinking the Foundations for Continual Reinforcement Learning](https://rlj.cs.umass.edu/2025/papers/Paper243.html)，Elelimy、David Szepesvari、Martha White、Bowling，RLJ 2025。** 在 Abel 定义论文之后读，了解 history process 与 deviation regret 的动机。它提出另一种研究框架，不表示 MDP 理论已被普遍否定或取代。

**代码 / 实验**：[平均回报方法官方代码](https://github.com/abhisheknaik96/average-reward-methods)。先复现小型访问控制或排队任务，解释收益率与差分价值；再讨论学习过程中的适应，而不是直接移到复杂机器人。

**读后追问**：你的比较对象是最优固定策略、随时间变化的策略，还是允许改变学习过程的智能体？如果比较对象掌握未来信息，结论应怎样解释？

**关注作者线索**：Abel、Van Roy、Precup；Wan、Naik、Sutton；Cheung、Simchi-Levi；Elelimy、Bowling、Martha White。

<a id="track-9"></a>
## 9. 元学习与快速适应：能否学会“怎样学习”？

**先修**：策略梯度、RNN 或梯度链式法则。**脉络**：利用多个任务训练适应机制 → 区分参数更新与内部状态更新 → 研究适应机制在长期变化中的有效性。

1. **状态中的适应：[RL²: Fast Reinforcement Learning via Slow Reinforcement Learning](https://arxiv.org/abs/1611.02779)，Duan 等，2016。** 外层训练 RNN，内层通过其状态积累经验并适应任务。要说清哪些参数在测试时固定、哪些状态在变化，以及状态何时重置。
2. **参数中的适应：[Model-Agnostic Meta-Learning](https://proceedings.mlr.press/v70/finn17a.html)，Finn、Abbeel、Levine，ICML 2017。** 学习便于少量梯度更新的初始化。重点理解内外层目标与任务分布；快速适应预先定义的任务分布，不自动保证无边界、长时间的新任务流适应。
3. **更贴近持续设置：[Task-Agnostic Continual Reinforcement Learning: Gaining Insights and Overcoming Challenges](https://proceedings.mlr.press/v232/caccia23a.html)，Caccia 等，CoLLAs 2023。** 将快速适应、递归状态和回放放到 task-agnostic CRL 中考察。这是一项递归 CRL 研究，不应误称为 MAML 的直接后继算法。

**实验入口**：先做训练/测试任务分布可控的 bandit 或小 MDP；比较固定权重的 RNN 状态适应与持续梯度更新。自动步长与在线 meta-gradient 则可从路线 2 的 SwiftTD 接入。

**读后追问**：所谓“持续学习”发生在权重、内部状态，还是外部记忆中？新分布超出元训练范围后，还能继续改善吗？

**关注作者线索**：Finn、Abbeel、Duan；在线学习规则方向看 Sutton、Mahmood、Javed。不要把所有 meta-RL 都归类为 CRL。

<a id="track-10"></a>
## 10. 持续多智能体学习：队友和对手也在变化怎么办？

**先修**：先了解 Markov game、合作/竞争、集中训练分散执行（CTDE）。**脉络**：理解其他学习者造成的变化 → 建模他者 → 在任务、伙伴变化中保留和适应协作知识。

1. **基础入口：[Christopher Amato 的合作 MARL 课程页与 slides](https://deeprlcourse.github.io/guests/christopher_amato/)。** 用来补信息结构、合作决策和部分可观测问题，不要一开始就读大量新算法。
2. **他者建模：[Probabilistic Recursive Reasoning for Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1901.09207)，温颖、杨耀东、Rui Luo、汪军、Wei Pan，ICLR 2019。** PR2 考虑他者如何响应自己的行为。它提供多智能体适应的方法视角，但不是持续任务序列基准上的 CRL 结果。
3. **直接的持续协作：[Multi-agent Continual Coordination via Progressive Task Contextualization](https://arxiv.org/abs/2305.13937)，Lei Yuan 等、俞扬，2023 预印本入口。** MACPro 通过任务上下文和逐步扩展的策略头应对任务流。重点分析共享与专用部分如何分配，以及新任务到来后资源是否增长。

**实验入口**：从小型矩阵博弈或网格协作开始，分别改变任务和伙伴策略。保持自身算法不变，只换其中一个变化来源；先能解释失败，再升级复杂 benchmark。

**读后追问**：性能下降来自环境变化、伙伴变化，还是自身遗忘？训练时见过伙伴身份吗？扩展模块所增加的参数和数据是否纳入比较？

**关注作者线索**：Amato、Stone、Foerster；国内及英国线索沿俞扬、温颖、杨耀东、汪军展开，并与既有资料中的郝建业、张伟楠、章宗长等相关研究对照。这里是追踪入口，不表示这些学者的全部工作都属于 CRL。

## 11. 代码和 benchmark 怎么选：只选能回答你问题的

先明确研究主张，再选环境。环境复杂、画面漂亮、任务数量多，都不能替代合适的对照。

| 入口 | 适合研究什么 | 使用时最需要警惕什么 |
|---|---|---|
| [MiniGrid](https://github.com/Farama-Foundation/Minigrid) | 低成本的状态、技能、探索与模型诊断 | 普通任务不是自动的 CRL；你需明确定义变化和评价协议 |
| [POPGym](https://github.com/proroklab/popgym) | 记忆、部分可观测、递归表示 | 记忆能力不等于持续适应能力 |
| [NS-Gym](https://github.com/scope-lab-vu/ns_gym) | 控制动力学变化的类型与速度 | 明确变化参数、通知权限、种子与是否可重置 |
| [CORA](https://github.com/AGI-Labs/continual_rl) | CRL 任务序列、基线、遗忘与迁移 | 检查 replay、任务边界与资源约束是否和你的问题一致 |
| [Continual World](https://github.com/awarelab/continual_world) | 操作任务序列上的知识保留与迁移 | 任务序列不等于无重置机器人一生；旧依赖需要单独处理 |
| [Syllabus](https://github.com/RyanNavillus/Syllabus) | 将课程算法接入训练管线 | 它是工具库，不是统一的 CRL 评价协议 |

第一次复现优先顺序：**小型可解释环境 → 一个主张清楚的基准 → 更复杂设置**。不建议同时安装以上所有工具。

### 所有分支都值得补读的一篇评价文章

[Position: Lifetime Tuning Is Incompatible with Continual Reinforcement Learning](https://proceedings.mlr.press/v267/mesbahi25a.html)，Mesbahi 等，ICML 2025。它讨论用完整的一生反复调参，再在同样长度上报告结果的问题。阅读重点是：评估过程中是否提前用到了部署未来的信息？这是立场论文及其配套证据，不是一条禁止所有超参数选择的定理。

开始实验前写下以下协议：

- **信息**：任务 ID、边界、变化通知和奖励函数是否对智能体可见？
- **生命周期**：什么会重置——环境、网络、优化器、递归状态、replay，还是都不重置？
- **资源**：环境步数、更新次数、单步耗时、峰值存储、模型规模各是多少？
- **调参**：在哪些环境和哪一段生命周期上选参数？测试种子及后期数据是否被用于选择？
- **指标**：全程收益、适应延迟、保留/遗忘、迁移、失败与恢复成本，哪些才对应你的主张？
- **统计**：多个种子和任务顺序；展示不确定性，而不是只挑最佳曲线。

没有自然任务边界时，不要生硬套用任务级 forgetting 矩阵；应解释如何定义评价窗口、探测任务或适应事件。如果额外评估需要重置或额外交互，要说明这是离线诊断，还是实际在线协议的一部分。

## 12. 讲座怎么用：带问题看，不把全部视频当必修

| 材料 | 最适合搭配 | 观看时抓住的问题 |
|---|---|---|
| [Sutton：Generate-and-test Methods for Continual Learning](https://www.youtube.com/watch?v=NaJxvGV8MRg) | 路线 1、3 | 系统如何提出候选、评价效用、保留或替换？ |
| [Abel：Where Is Learning?](https://www.youtube.com/watch?v=vVZuoEtxjzA) | 入门材料③、路线 8、9 | 你把学习归因于智能体的哪个组成部分？ |
| [Machado：课程录像与 slides](https://deeprlcourse.github.io/guests/marlos_machado/) | 路线 4 | 表示怎样帮助发现技能，技能怎样反过来改变经验与表示？ |
| [Mahmood：Challenges of Single-Life Continual RL](https://www.youtube.com/watch?v=G-LgYNEIwBE) | 路线 2、7 | 一生中的计算、恢复和适应成本应该如何计入？ |
| [Stone：Continual RL and Automatic Curriculum Learning](https://www.youtube.com/watch?v=5D5rJJp5jnw) | 路线 6 | 谁决定学什么，任务选择是否也能学习？ |
| [Adam White：课程录像与 slides](https://deeprlcourse.github.io/guests/adam_white/) | 所有实验路线 | 什么证据足以支持你的研究主张？ |

这是推荐观看问题，不是逐段转录或对视频结论的逐条认证。[Continual RL Workshop 官网](https://sites.google.com/view/continual-learning)提供社区与录像集合入口，可在选定方向后继续追踪。

## 13. 跨学科选修：先补三扇窗，不另开一份巨大书单

这些材料帮助提出问题，不是进入 CRL 的先修要求。

1. **认知科学 / 心理学：[Anne Collins 的课程录像与 slides](https://deeprlcourse.github.io/guests/anne_collins/)。** 关注工作记忆、慢速学习和快速灵活行为的区别。可连接路线 3、9：更快的行为适应究竟来自学会了新参数，还是利用了内部状态和已有结构？
2. **神经科学：[Wolfram Schultz 的课程录像与 slides](https://deeprlcourse.github.io/guests/wolfram_schultz/)。** 从奖赏预测误差与神经信号理解 RL 的生物学联系。可连接 TD 与学习信号设计，但不要由现象对应直接推断算法与大脑机制相同。
3. **控制论：[Åström 与 Murray，Feedback Systems 第二版](https://www.cds.caltech.edu/~murray/FBS/Second_Edition.html)。** 先读反馈原理、系统建模；做状态构造时选读观测与状态估计，做真实系统时补稳定性。作者旧页面提供章节并指向新站；可连接路线 3、5、7，但估计器稳定性不自动等于学习系统稳定性。

读完一项，尝试提出一个可以在 RL 实验中区分的预测。没有可检验连接的类比，暂时保留为灵感即可。

## 14. 从读书到选题：一个最小研究提案模板

```text
我要研究的不是“提高 CRL 性能”，而是：________________。
在环境的 ______ 变化、智能体只能使用 ______ 的条件下，
我怀疑现有方法的 ______ 机制会导致 ______。

我先用 ______ 小环境排查这个原因；
与 ______ 基线比较，并控制数据、计算和调参预算；
用 ______ 指标检验，而不是只看最终回报。

如果移除 ______ 后现象消失，说明 ______；
如果现象仍在，但新方法无效，下一步检查 ______。
```

第一份提案能把这些空填清楚，比再收藏二十篇论文更接近开展研究。

---

### 延伸阅读

通过[学者索引](/zh/continual-rl/people/)查找代表性工作与学习材料，再沿选定问题深入阅读。
