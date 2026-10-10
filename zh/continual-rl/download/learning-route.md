# 三册教材与阅读起点

## 第 I 册 · 经典强化学习

怎样从经验预测后果、改善行动，并用模型进行规划？

### 表格方法

- [第 1 章 · 多臂老虎机：估计、探索与直接策略学习](https://yingwen.io/zh/continual-rl/foundations/tabular/bandits/)：没有状态转移时，仍需一边估计动作收益，一边决定下一次尝试什么。这个最小问题把估计误差、探索代价和策略更新分开。
- [第 2 章 · MDP、回报与价值：序列决策的数学对象](https://yingwen.io/zh/continual-rl/foundations/tabular/mdps/)：动作会改变后续状态时，需要评价整个未来。本章从随机交互过程推导价值与 Bellman 方程，明确后续算法共同使用的数学对象。
- [第 3 章 · 动态规划：评价、改善与最优递推](https://yingwen.io/zh/continual-rl/foundations/tabular/dynamic-programming/)：已知环境模型时，怎样通过局部计算得到长期价值和策略？本章将 Bellman 方程转化为迭代，并证明评价与改善之间的联系。
- [第 4 章 · Monte Carlo：完整回报、探索控制与离策略评价](https://yingwen.io/zh/continual-rl/foundations/tabular/monte-carlo/)：不知道模型时，可以将完整回报作为样本。估计还会改变下一次行动：需要看清回报来自哪一版策略、探索怎样影响收益，以及长回合怎样消耗有效覆盖。
- [第 5 章 · TD 预测与控制：SARSA、Expected SARSA、Q-learning 和 Double Q](https://yingwen.io/zh/continual-rl/foundations/tabular/temporal-difference/)：如何在完整回报尚不可用时学习？TD 用下一预测补足未来；控制算法再根据不同的下一动作处理方式，形成不同的学习目标。
- [第 6 章 · 多步学习：n-step、Tree Backup 与 Q(σ)](https://yingwen.io/zh/continual-rl/foundations/tabular/multistep/)：学习目标可以在一步 bootstrap 与完整回报之间选择，也可以在动作采样与动作期望之间选择。这是两条不同的设计维度。
- [第 7 章 · 学习与规划：Dyna、优先扫描和执行时搜索](https://yingwen.io/zh/continual-rl/foundations/tabular/planning/)：真实经验既能直接改进价值，也能训练后果模型。规划使用这个模型继续计算，关键是模型语义、backup 的成本以及计算应分配到哪里。

### 函数近似与经典进阶

- [第 1 章 · 函数逼近预测：从回归到 TD 固定点](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/)：共享少量参数以后，MC、TD 和最小二乘方法究竟在求解什么？
- [第 2 章 · 特征、泛化与半梯度控制](https://yingwen.io/zh/continual-rl/foundations/approximation/features-control/)：特征怎样改变学习行为，Sarsa 又怎样在共享参数下改善策略？
- [第 3 章 · 持续控制与平均奖励](https://yingwen.io/zh/continual-rl/foundations/approximation/average-control/)：智能体没有自然回合终点时，怎样定义和学习长期控制目标？
- [第 4 章 · 离策略函数逼近：覆盖、发散与稳定更新](https://yingwen.io/zh/continual-rl/foundations/approximation/off-policy/)：行为数据足够覆盖目标策略，为什么 TD 仍可能发散，又能怎样修复？
- [第 5 章 · 多步回报、资格迹与 True-online TD](https://yingwen.io/zh/continual-rl/foundations/approximation/traces/)：当前到来的奖励怎样更新过去的预测，同时保留正确的在线更新语义？
- [第 6 章 · 策略梯度、基线与 Actor–Critic](https://yingwen.io/zh/continual-rl/foundations/approximation/policy-gradient/)：直接学习策略时，哪一个目标的梯度能由经验估计，近似从哪里进入？

## 第 II 册 · 深度强化学习

表示、目标和数据都在变化时，怎样组织一个可靠的训练循环？

### 核心算法与训练循环

- [第 1 章 · 深度价值学习：DQN、Double DQN 与目标的时间顺序](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/)：把表格 Q-learning 换成网络后，损失、数据与目标为什么都需要重新组织？
- [第 2 章 · 策略梯度：从轨迹概率到 GAE 与 actor–critic](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/)：延迟奖励怎样改变动作概率？有限 rollout、critic 与停止梯度分别改变哪一项估计？
- [第 3 章 · 策略更新的尺度：TRPO 与 PPO](https://yingwen.io/zh/continual-rl/foundations/deep/trust-region/)：旧策略的数据能支持多远的策略更新？怎样从局部代理走到采样、更新与独立评价？
- [第 4 章 · 连续动作的价值优化：DDPG 与 TD3](https://yingwen.io/zh/continual-rl/foundations/deep/deterministic-control/)：不能枚举连续动作时，如何用 critic 的梯度改进 actor？
- [第 5 章 · 最大熵连续控制：SAC 的价值、密度与温度](https://yingwen.io/zh/continual-rl/foundations/deep/entropy-control/)：随机 actor 不只是加噪声：熵如何进入 Bellman 方程与自动微分？
- [第 6 章 · 深度 RL 的机制接口：模型、记忆、离线数据与实验](https://yingwen.io/zh/continual-rl/foundations/deep/practice/)：改变数据来源或 agent state 后，哪些推导和实现条件必须重新检查？
- [第 7 章 · 大规模训练：算法与系统怎样共同设计](https://yingwen.io/zh/continual-rl/foundations/deep/systems/)：环境、推理和学习并行以后，怎样把更多计算变成更快的策略改善？

### 按问题选择的研究分支

- [不完全可观测：信念状态、信息行动与递归记忆](https://yingwen.io/zh/continual-rl/foundations/deep/partial-observability/)：当前观察不能决定未来时，智能体应记住什么，信息又如何影响行动？
- [探索与不确定性：后验、乐观估计和时间一致行动](https://yingwen.io/zh/continual-rl/foundations/deep/exploration/)：为什么每一步都随机，并不等于有效获取长期有用的信息？
- [分布强化学习：Bellman 分布、分位数与风险目标](https://yingwen.io/zh/continual-rl/foundations/deep/distributional/)：学习完整回报分布，与学习均值、评估风险和估计知识不确定性分别有什么关系？
- [离线强化学习：数据支持、策略评估与保守改进](https://yingwen.io/zh/continual-rl/foundations/deep/offline/)：不能补采数据时，怎样判断策略好坏，怎样避免利用没有证据的高价值动作？
- [模型学习与规划：MPC、短模型 rollout 和潜在想象](https://yingwen.io/zh/continual-rl/foundations/deep/model-based/)：模型在哪里进入决策，预测误差又怎样变成控制误差？
- [约束强化学习：占据测度、拉格朗日与可行策略](https://yingwen.io/zh/continual-rl/foundations/deep/constraints/)：“回报高且代价不超过预算”与“每一步都安全”之间差了哪些条件？
- [多智能体合作：结构化探索与信用分配](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent/)：团队共享一个奖励时，怎样从联合经验中学习可执行的协作策略，并正确处理同伴更新？
- [自对弈与开放式多智能体学习：评估、目标与策略种群](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent-populations/)：自对弈怎样产生课程和训练标签，又怎样通过历史保留、交互评价与策略种群发现值得继续学习的问题？
- [对手建模与递归推理：预测谁，回应什么？](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent-reasoning/)：给定参与者和评价目标，怎样利用行为预测、条件响应与有限递归改善决策，并检验模型是否可信？

## 第 III 册 · 持续强化学习

智能体持续改变自身时，怎样评价学习，并积累可用于未来行动的知识？

### 目标、评价与持续控制

先规定长期学习的目标和评价对象，再讨论奖励怎样表达目标，以及怎样比较会继续学习的控制器。平均奖励随后展开一种特定的长期准则；它不是所有持续学习问题都必须采用的目标。

- [第 1 章 · 强化学习问题的形式化：交互、目标与持续学习](https://yingwen.io/zh/continual-rl/foundations/objectives/)：一个长期运行的智能体应当优化什么？这个选择怎样影响状态、价值函数、学习算法和评价？
- [第 2 章 · 奖励假设与奖励设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/)：什么样的目标可以表示为奖励？智能体学会最大化奖励，是否就实现了设计者的意图？
- [第 3 章 · 持续控制：比较策略与学习智能体](https://yingwen.io/zh/continual-rl/algorithms/control/)：一个智能体当前做得好，不代表它以后仍能学得好。持续控制要评价完整的行动—学习过程：行动改变世界和数据，学习改变后续行动，有限记忆与计算又限制了这个过程。本章从这些依赖出发，定义可以比较的对象，并用可解析反例检验不同评价标准。
- [第 4 章 · 平均奖励：奖励率、差分价值与持续控制](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)：为什么长期奖励率与折扣回报可能选择不同策略？去掉目标中的折扣后，预测、控制与规划各需要多解决什么问题？

[从这里即可查阅：实验设计与算法比较](https://yingwen.io/zh/continual-rl/experiments/)

### 智能体状态与预测知识

智能体状态保留决策所需的历史信息；预测知识回答在指定行为条件下，未来某个信号会怎样。先区分这两个学习对象，再讨论如何用预测构造状态，以及状态表示怎样影响预测。

- [第 1 章 · Agent state：部分可观测性、递归记忆与在线信用分配](https://yingwen.io/zh/continual-rl/construction/state/)：任务目标给定以后，智能体应当保留哪些历史信息，才能预测未来并选择动作？
- [第 2 章 · 通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)：同一张地图既能问“向左会成功吗”，也能问“还要走几步”。怎样规定这些问题，从一条经验流学出答案，再让决策用上它们？

### 信用分配、流式学习与元学习

信用分配研究反馈怎样影响过去的预测、动作与参数路径；流式学习研究数据、内存和逐步计算受限时怎样更新；元学习进一步根据学习效果调整更新规则。三章沿这些问题展开，同一个算法可以同时使用它们。

- [第 1 章 · 时间信用分配：从资格迹到深度梯度学习](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)：结果到来时，怎样更新过去的预测、动作和记忆参数？策略变化、表示变化和部分可观测性会怎样改变信用与等价条件？
- [第 2 章 · 流式强化学习：交互协议与更新稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)：机器人刚获得一条经验，下一次行动已经快到了：学习器能保存什么，还能算几次？从相同数据流、两步 TD 与中途恢复的算例出发，理解原始经验、活动状态、权重、资格迹和尺度统计怎样影响下一次更新。
- [第 3 章 · 学习规则的适应：在线元梯度与跨任务元学习](https://yingwen.io/zh/continual-rl/algorithms/meta/)：一次更新减小了当前误差，但它是否让下一次学习更容易？元学习把这个问题变成可计算的评价：先按某条规则学习，再用后续表现改进这条规则。

### 知识保留与学习可塑性

旧知识是否丢失，与新知识是否仍能学会，是长期学习的两项并列问题。先分析知识保留，再分析学习可塑性，并用旧能力表现和新任务学习过程检查二者的取舍。

- [第 1 章 · 知识保留：经验重放、参数约束与模型记忆](https://yingwen.io/zh/continual-rl/algorithms/retention/)：新经验要求适应，旧技能又可能重新有用；如何在固定预算内管理它们的冲突？
- [第 2 章 · 可塑性：梯度通路、有效学习率与预测干扰](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)：在相同新数据与更新预算下，学习能力为什么可能下降，又该怎样诊断与恢复？

### 子任务、技能、模型与规划

先定义值得学习的子任务，再学习完成它的技能，预测执行技能的后果，最后用这些预测进行规划。子任务规定要解决什么，技能规定怎样行动，模型描述行动后果，规划据此改善决策。

- [第 1 章 · 目标与子任务：条件控制、经验重用与技能设计](https://yingwen.io/zh/continual-rl/construction/goals/)：怎样让同一套控制器应对不同目标、从未成功的尝试中学习，并选择对未来控制有用的子任务？
- [第 2 章 · Options：多步决策、技能发现与可复用行为](https://yingwen.io/zh/continual-rl/construction/options/)：怎样把连续多步的行为当成可复用的决策单位，同时仍能在每个原始时间步学习？
- [第 3 章 · 模型与后果预测：学什么，才能用于下一次决策？](https://yingwen.io/zh/continual-rl/construction/models/)：执行一个动作或技能以后，会积累多少奖励、何时到哪里；这些预测怎样支持规划与任务变化后的迁移？
- [第 4 章 · 规划：把模型中的经验转成更好的决策](https://yingwen.io/zh/continual-rl/construction/planning/)：真实交互很贵、计算预算有限时，怎样决定想象什么、更新什么，以及何时应该不再相信模型？

[需要时回顾：Dyna 的模型学习与规划循环](https://yingwen.io/zh/continual-rl/algorithms/dyna/)

### 探索与经验获取

上述学习都依赖实际获得的经验。探索研究应该采取哪些行动来获得有用信息，并在单生命期中保留后续学习的机会；它贯穿前面的过程，可在读完持续控制后提前阅读。

- [第 1 章 · 持续探索：新奇、不确定性、学习进展与恢复](https://yingwen.io/zh/continual-rl/algorithms/exploration/)：外部奖励稀疏、世界持续改变时，怎样获得有用的新经验，而不是追逐永远无法学会的噪声？

### 完整智能体与研究实验

把状态、预测、学习规则、技能和模型放回同一个智能体，考察它们共同变化时如何协作。架构章讨论这些关系，实验章说明怎样区分机制作用、资源代价与长期收益；实验方法也应伴随前面各章使用。

- [第 1 章 · 持续智能体架构：模块接口、更新调度与长期评价](https://yingwen.io/zh/continual-rl/construction/architectures/)：各模块单独能学，不代表接在一起就能持续改善；它们究竟交换什么、何时更新、如何共享有限计算？
- [第 2 章 · 实验设计：从更新正确到持续学习证据](https://yingwen.io/zh/continual-rl/experiments/)：一个算法通过测试、曲线更高，分别能说明什么？怎样用有限预算得到可以重复检验的结论？

### 基础工具的专题回顾

这些章节回顾价值学习、深度价值学习、策略梯度、最大熵控制与 Dyna，供阅读前文时补充先修。它们是可以用于持续学习的工具，不必在本册末尾再顺序读一遍。

- [价值预测与时间差分学习](https://yingwen.io/zh/continual-rl/algorithms/value/)：持续学习仍要预测后果。沿用 MC、TD 与资格迹时，需要固定哪些对象，才能分清估计在更新、预测问题在变化，以及学习器未来会改变行为这三件事？
- [深度价值学习：DQN 与 Double DQN](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)：把 DQN 放进长期运行的学习器后，哪些量只是为一次更新而固定，哪些旧经验、目标和特征会继续影响未来行动？
- [策略梯度、Actor–Critic 与 PPO](https://yingwen.io/zh/continual-rl/algorithms/policy/)：策略梯度和 PPO 的一次更新以什么行为分布、价值版本与评价目标为参照？把这些更新连成持续学习过程后，哪些结论还需要重新检验？
- [最大熵控制与 Soft Actor–Critic](https://yingwen.io/zh/continual-rl/algorithms/soft-control/)：持续运行 SAC 时，温度改变的是哪个优化问题？怎样区分 soft 价值、外部任务收益，以及长期学习中保留随机性的实际作用？
- [Dyna：模型学习与规划](https://yingwen.io/zh/continual-rl/algorithms/dyna/)：直接学习从真实经验更新价值或策略；模型学习估计行动后果；规划用这些估计进行额外计算。Dyna 将三者连接，使已有经验可以通过模型继续影响决策。

# 强化学习：共同框架与学习路线

运行时，智能体与世界形成交互闭环；外部设计者选择奖励、初始化、数据权限、调参与预算。经典方法、神经表示和持续学习描述不同维度，可以共同用于同一智能体。下面按教学先修组织，不把三册视为互斥问题类，也不要求读完全部分支才开始研究。

[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：比较完整学习器](https://yingwen.io/zh/continual-rl/algorithms/control/)

基础目录包含表格方法、函数逼近与深度核心算法。深度拓展中的部分可观测、探索、回报分布、离线数据、模型、约束和多智能体是并列研究分支，可按问题选择。



<a id="foundation-continuity"></a>

## 沿同一个问题，读通基础方法与持续学习

同一种更新式可能用于不同目标；不同算法也可能求解同一个问题。以下从预测或优化对象出发，逐项考察表示、数据和计算条件改变后的结论。每条线均可独立阅读；箭头表示论证的衔接，不表示方法之间的优劣。

### 预测的量是什么，误差又是什么？

基础：先固定策略。价值是回报的条件期望。MC 使用完整回报；TD 用下一时刻的预测替代未观察的余项。两者不能仅按同一批样本上的 TD error 排序。

条件改变：共享参数限制了可表示的函数。采样权重决定在哪些状态上拟合。最小价值误差、最小 Bellman 残差和 TD 固定点一般不同；神经网络又使可表示的局部方向随参数改变。

研究问题：多个 GVF 共用表示时，哪些预测值得占用容量？应分别检查问题定义是否改变、数据是否覆盖，以及回答该问题的误差是否降低。

[MC 与 TD](https://yingwen.io/zh/continual-rl/foundations/tabular/temporal-difference/) → [投影与半梯度](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/) → [神经价值更新](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/) → [GVF 的问题与答案](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)

### 当前输入保留了哪些历史信息？

基础：Bellman 方程先假定有足够的状态。表格为不同状态分别存值；它不负责从相同观察中恢复被遗漏的历史。

条件改变：特征共享是在已知信息上泛化；递归状态则保留过去信息。增加网络宽度不等于补回历史，低训练误差也不证明输入满足 Markov 性。

研究问题：策略改变以后，原来的状态压缩是否仍能预测行动后果？构造状态的网络、运行时记忆、资格迹与优化器状态如何共同更新？

[MDP 的状态条件](https://yingwen.io/zh/continual-rl/foundations/tabular/mdps/) → [表示与泛化](https://yingwen.io/zh/continual-rl/foundations/approximation/features-control/) → [不完全可观测](https://yingwen.io/zh/continual-rl/foundations/deep/partial-observability/) → [智能体状态](https://yingwen.io/zh/continual-rl/construction/state/)

### 估计哪种策略的价值，又按什么状态分布加权？

基础：行为策略决定怎样获得经验；目标策略决定要预测哪种行为。重要性比可以校正给定状态下的动作分布，但不会自动把状态出现的频率改成目标策略的频率。

条件改变：Replay 还引入缓冲区的时间组成与抽样规则。神经更新受到数据分布、共享梯度和移动目标共同影响。重复旧数据与逐条使用新数据有不同的资源和适应代价。

研究问题：单一行为流怎样支持许多预测和技能？在固定内存下，怎样权衡覆盖、样本年龄、更新方差与适应速度，而不把离策略修正当作完整稳定性保证？

[离策略稳定性](https://yingwen.io/zh/continual-rl/foundations/approximation/off-policy/) → [数据与训练接口](https://yingwen.io/zh/continual-rl/foundations/deep/practice/) → [大规模系统与策略滞后](https://yingwen.io/zh/continual-rl/foundations/deep/systems/) → [离线数据的覆盖](https://yingwen.io/zh/continual-rl/foundations/deep/offline/) → [流式更新](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

### 怎样把较晚的反馈归给较早的计算？

基础：多步回报定义用多远的未来构造目标。资格迹压缩过去的特征或梯度方向。前向与后向等价必须说明参数是在整个轨迹内固定，还是每一步改变。

条件改变：神经网络改变后，旧梯度不再等于用当前参数重算的梯度。递归状态还带来参数经过历史状态影响当前输出的路径，不能用一条普通 TD trace 代替。

研究问题：在每步计算有界的条件下，保留多少过去影响才有用？替换特征时，怎样处理与旧特征绑定的资格迹、优化器动量和元梯度？

[多步回报](https://yingwen.io/zh/continual-rl/foundations/tabular/multistep/) → [资格迹与等价条件](https://yingwen.io/zh/continual-rl/foundations/approximation/traces/) → [GAE 与 actor–critic](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/) → [在线信用分配](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

### 优化一段折扣回报，还是长期单位时间收益？

基础：回报定义决定策略排序。折扣奖励、有限时域和平均奖励是不同目标；把折扣取大，只是在某些条件下接近相应极限。

条件改变：持续任务的相对价值需要奖励率和定标条件。训练中更新策略时，奖励率也在变化。动作时长不同，还要区分每次决策与单位物理时间的收益。

研究问题：长期收益率忽略有限的启动损失；单生命期不能忽略。怎样同时报告生命期收益、适应成本和后期表现，并让预测、控制与模型使用一致的时间单位？

[平均奖励控制基础](https://yingwen.io/zh/continual-rl/foundations/approximation/average-control/) → [熵如何改变目标](https://yingwen.io/zh/continual-rl/foundations/deep/entropy-control/) → [平均奖励的预测、控制与规划](https://yingwen.io/zh/continual-rl/algorithms/average-reward/) → [完整学习器的评价](https://yingwen.io/zh/continual-rl/algorithms/control/)

### 一次策略更新，为什么能改善未来？

基础：精确策略改善使用旧策略的真实价值；策略梯度使用与目标匹配的访问分布。值函数近似或梯度估计误差会破坏这些推理的前提。

条件改变：PPO 的动作概率比不等于新策略的状态访问比。连续动作 actor 还会追逐 critic 的误差。限制局部更新尺度与证明实际回报单调提高是不同要求。

研究问题：动作不仅改变世界，也改变未来数据与学习。比较冻结策略不等于比较持续更新的智能体；什么时候应付出当前回报去获得长期有用的经验？

[精确策略改善](https://yingwen.io/zh/continual-rl/foundations/tabular/dynamic-programming/) → [策略梯度定理](https://yingwen.io/zh/continual-rl/foundations/approximation/policy-gradient/) → [TRPO 与 PPO 的近似](https://yingwen.io/zh/continual-rl/foundations/deep/trust-region/) → [持续控制的比较器](https://yingwen.io/zh/continual-rl/algorithms/control/)

### 模型需要预测什么，内部计算应花在哪里？

基础：规划根据模型更新价值或选择动作。Dyna 把真实经验更新、模型拟合和规划查询分开。模型更新次数更多，不代表新增了真实证据。

条件改变：线性价值可使用期望特征模型；非线性价值通常不能把后果分布替换成均值。Option 模型还要保留随机持续时间、真实奖励和终点的关系。

研究问题：当表示、技能或环境改变时，哪些旧模型仍可复用？有限计算应优先用于收集真实经验、改进模型，还是在已有模型中规划？

[Dyna 与搜索控制](https://yingwen.io/zh/continual-rl/foundations/tabular/planning/) → [神经模型与规划](https://yingwen.io/zh/continual-rl/foundations/deep/model-based/) → [Option 后果模型](https://yingwen.io/zh/continual-rl/construction/models/) → [规划与计算分配](https://yingwen.io/zh/continual-rl/construction/planning/)

## 经典强化学习：表格方法和函数近似

强化学习的基本问题，是智能体怎样通过行动及其后果，学会获得更好的长期结果。我们先研究能把困难分开的简单情形：bandit 隔离估计与探索；有限 MDP 加入动作对未来处境的影响；函数逼近再引入经验共享。这样的顺序便于发现每种方法解决了什么，也便于看清哪些困难尚未解决。

学习目标：

- 能从奖励序列写出回报，区分真实终止与采样截断。
- 能推导固定策略的 Bellman 方程和最优递推，并实现评价—改善循环。
- 能解释 MC、TD、控制、资格迹和函数逼近之间的依赖关系。

### 1 · 交互、状态与回报

一次行动的好坏，为什么不能只由眼前的奖励判断？

先考虑每次从同样条件出发的选择。两个动作的奖励均值未知，选择一个动作就获得它的一次奖励样本。这是 bandit 问题。样本既带来收益，也提供信息；总选当前估计最好的动作，可能永远发现不了另一个动作更好。bandit 暂时省去动作改变下一处境的困难，让我们先集中研究估计与探索。

现在允许动作改变后续处境。在离散时间中，智能体观察状态 $S_t$，按策略 $\pi(a\mid s)$ 选择动作 $A_t$，环境返回奖励 $R_{t+1}$ 和下一状态 $S_{t+1}$。MDP 假定给定当前状态与动作后，下一步的分布不再依赖更早的历史。这里先假定有限状态、有限动作、固定转移规律和有界奖励。观测不满足这个条件时，需要另外构造状态，而不是把观测改名为状态。

智能体与环境的边界取决于正在研究的决策者。策略能改变身体的位置，并不表示身体的动力学由策略直接决定；这部分后果仍由环境产生。奖励也在此接口上传入。它可以来自物理传感器、任务规则或另一套评价程序，不必由人逐次给分。规定哪种奖励值得追求，与学习怎样得到这种奖励，是两个问题。

策略需要比较动作带来的整个未来，因此先规定回报。折扣回报让距离当前更远的奖励获得权重 $\gamma^k$；当 $0\leq\gamma<1$ 且奖励有界时，无限和存在。$\gamma$ 是目标的一部分，不只是训练参数。若一个动作眼前收益高但让后续选择变差，只优化即时奖励就会选错。

例如，在 A 可以领取 1 并结束，也可以不领奖励而走到 B；在 B 下一步领取 2 并结束。取 $\gamma=0.9$，两种行动的回报分别是 1 和 1.8。奖励是每步收到的数；回报是这一串数的汇总；价值则是在给定条件和策略下对回报取期望。价值估计还可能不准确，但不能因此改变这些量的定义。

终止与停止采样不是一回事。到达定义中的终点后没有未来回报；达到一次训练调用的步数上限，则环境可能仍会继续。后一种情况的价值目标通常需要 bootstrap。若截止时间是任务本身的一部分，一般还需把剩余时间纳入决策状态，或明确使用随时间变化的价值和策略；有无折扣都不能代替这一信息。

$$
G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}=R_{t+1}+\gamma G_{t+1},\qquad v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s]
$$

第一式把未来分成第一步和剩余部分。第二式对策略引起的所有未来取条件期望，定义价值；它还不是学习算法。

#### 动手与核对

[下载 objectives_lab.py](https://yingwen.io/zh/continual-rl/download/objectives_lab.py)

在保存该文件的目录运行：

```sh
python3 objectives_lab.py stopping
python3 objectives_lab.py test
```

预期检查：输出 discounted 与 independent_stopping 均为 2.75；同一条长度为一的前缀，保留继续价值时为 10，误当终止时为 1。

实验范围：这是目标与边界条件的确定性检验，不是学习性能比较。

自测：奖励依次为 1、2、3，随后真正终止；折扣为 0.5。第一个状态的回报是多少？

解答：1 + 0.5 × 2 + 0.25 × 3 = 2.75。若第三步只是采样截断，应再加 $0.5^3$ 乘下一状态的价值估计。

完整推导与问题衔接：

- [Bandit：估计、探索与跟踪](https://yingwen.io/zh/continual-rl/foundations/tabular/bandits/)
- [MDP、回报与价值的完整推导](https://yingwen.io/zh/continual-rl/foundations/tabular/mdps/)
- [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)
- [奖励假设与奖励设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/)

原文、课程与实现：

- [Sutton & Barto · Reinforcement Learning, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：第 1—3 章：强化学习问题、bandit、agent–environment interface、回报和 MDP。
- [Spinning Up · 关键概念](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)：对照状态、观测、动作、策略、轨迹和回报的区别。

### 2 · 已知模型：策略评价与动态规划

若已知每个动作的后果，怎样把局部计算变成长期判断？

模型 $p(s',r\mid s,a)$ 给出下一状态和奖励的联合分布。固定策略后，对回报递推取期望，就得到 Bellman 期望方程。更新某个状态时不必枚举所有未来轨迹，只要枚举一步后果，再接上对未来的当前估计。这种一步递推称为 backup。

评价固定 $\pi$ 时，下一动作按 $\pi$ 求平均；寻找最优策略时，下一动作取最大值。前者的固定点是 $v_\pi$，后者是 $v_*$。在本节的折扣有限 MDP 中，两种 Bellman 算子都是最大范数下的 $\gamma$ 收缩，因此反复完整更新会收敛到唯一价值固定点。

策略迭代先评价当前策略，再选择该价值下的一步贪心动作。策略改善成立，是因为用新动作并继续旧策略已不差，再在后续每一步重复这种改善不会降低价值。价值迭代则不等评价完全结束，每一轮直接做最优 backup。评价与改善相互作用的共同思想称为广义策略迭代（generalized policy iteration，GPI）。这里不是迁移学习中同缩写的 generalized policy improvement。

$$
\begin{aligned}(T_\pi v)(s)&=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)[r+\gamma v(s')],\\(T_*v)(s)&=\max_a\sum_{s',r}p(s',r\mid s,a)[r+\gamma v(s')].\end{aligned}
$$

策略迭代求解 $v=T_\pi v$ 后改善策略；价值迭代使用 $v_{k+1}=T_*v_k$。真实终止状态的余项取零。

#### 动手与核对

[下载 control_problem_lab.py](https://yingwen.io/zh/continual-rl/download/control_problem_lab.py)

在保存该文件的目录运行：

```sh
python3 control_problem_lab.py dp
python3 control_problem_lab.py test
```

预期检查：策略依次为 (0,0)、(1,0)、(1,1)，最后价值约为 (2.368421, 2.631579)；策略迭代与价值迭代一致。

实验范围：模型已知，计算量按 backup 计数；它不消耗真实环境交互来估计模型。

自测：A 可拿 1 后结束，也可零奖励转到 B；B 可拿 2 后结束，也可拿 0.5 后回 A。γ = 0.9。为什么两处都选循环动作反而更好？

解答：循环策略满足 v(A)=0.9v(B)、v(B)=0.5+0.9v(A)，解为 45/19 和 50/19，均高于对应退出奖励。只比较即时奖励会漏掉循环中的长期收入。

完整推导与问题衔接：

- [动态规划：评价、改善与最优递推](https://yingwen.io/zh/continual-rl/foundations/tabular/dynamic-programming/)
- [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)
- [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

原文、课程与实现：

- [Sutton & Barto · Reinforcement Learning, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：第 4 章：策略评价、策略改善、策略迭代、价值迭代与 generalized policy iteration。
- [David Silver · UCL 强化学习课程](https://davidstarsilver.wordpress.com/teaching/)：课程页提供讲义和视频；按 MDP、动态规划、无模型预测与控制、函数逼近的顺序选读。

### 3 · 未知模型：Monte Carlo 与 TD 预测

不知道转移概率时，一条经验能够教会价值函数什么？

现在模型未知，但暂时仍固定策略。Monte Carlo 等一个回合结束后，把观测到的完整回报作为训练目标。TD 不等完整结果，而用一步奖励加下一状态的价值估计作为目标。这就是 bootstrap：用已有预测帮助学习另一个预测。

在适当采样条件下，MC 的完整回报是当前策略价值的无偏样本，但方差可能较大。TD 的单步目标受当前估计误差影响，却能较早更新。表格 on-policy TD 的收敛需要访问覆盖、合适步长和固定环境等条件。单次 TD 误差衡量的是样本目标与当前预测之差，真实价值误差还要用策略的期望回报来判断。

MC 与 TD 都可以逐样本实现，也都可以使用表格或神经网络。区别在于学习目标：MC 等待完整回报，TD 接上已有价值估计。先在一条两步轨迹上手算，再观察多次重复后两种方法如何接近相同答案。

$$
\begin{aligned}V(S_t)&\leftarrow V(S_t)+\alpha[G_t-V(S_t)]&&\text{MC},\\\delta_t&=R_{t+1}+\gamma V(S_{t+1})-V(S_t),\\V(S_t)&\leftarrow V(S_t)+\alpha\delta_t&&\text{TD(0)}.\end{aligned}
$$

右端各价值都在本次更新前读取。真正终止时下一价值为零。α 控制每次新证据改变预测的幅度。

#### 动手与核对

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

在保存该文件的目录运行：

```sh
python3 foundations_detail_lab.py value
python3 foundations_detail_lab.py test
```

预期检查：反复观察 A→B→终点，奖励为 0、1；MC 和 TD 最终都接近 (0.9, 1.0)。

实验范围：这是固定策略的预测实验，没有策略改善或探索性能结论。

自测：初值全零、α = 0.1、γ = 0.9。第一个 A→B→终点回合后，MC 与按时间顺序执行的 TD 各是什么值？

解答：MC 得到 A=0.09、B=0.1。在线 TD 在访问 A 时尚不知道 B 的价值，因此 A=0、B=0.1。后续经验会把 B 的信息传播到 A。

完整推导与问题衔接：

- [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)

原文、课程与实现：

- [Sutton & Barto · Reinforcement Learning, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：第 5、6 章：MC 与 TD prediction；先读预测部分，再读对应控制扩展。
- [Sutton · Learning to Predict by the Methods of Temporal Differences](https://doi.org/10.1007/BF00115009)：TD 学习的原始论文；与本节的表格例子对应阅读。

### 4 · 从预测到控制：数据与策略共同变化

学到价值以后，怎样让行动真的变好？

控制的目标是改善策略，不只是降低预测误差。动作价值 $q_\pi(s,a)$ 比状态价值多固定了第一步动作，因此可以直接比较动作。一个完整学习循环包括按行为策略采样、更新动作价值、再让行为依赖新价值。只在固定数据上比较 target，不足以构成这个闭环。

SARSA 把实际选择的下一动作价值接进目标，评价并改善带探索的当前行为。Q-learning 对下一动作取最大值，目标指向贪心控制，即使收集数据时仍在探索。Expected SARSA 则对指定下一策略的动作分布取期望。三者不同在余项，不同的余项决定它们在学习哪个策略。

行为必须提供目标所需的信息。表格 Q-learning 的经典收敛结论要求每个状态动作对持续得到访问、逐对步长满足随机逼近条件等，并非“使用 max 就能收敛”。SARSA 常用探索逐渐消失但访问不停止的条件。有限寿命或环境变化时，保持常数步长和持续探索可能有实际意义，但目标转为跟踪，而非收敛到固定表。

$$
\begin{aligned}y_t^{\rm Sarsa}&=R_{t+1}+\gamma Q(S_{t+1},A_{t+1}),\\y_t^{\rm Q}&=R_{t+1}+\gamma\max_aQ(S_{t+1},a),\\Q(S_t,A_t)&\leftarrow Q(S_t,A_t)+\alpha[y_t-Q(S_t,A_t)].\end{aligned}
$$

下一动作要按与算法一致的时序取得。二者均需处理终止；Q-learning 的行为策略不必等于 target 中的贪心策略。

#### 动手与核对

[下载 control_problem_lab.py](https://yingwen.io/zh/continual-rl/download/control_problem_lab.py)

在保存该文件的目录运行：

```sh
python3 control_problem_lab.py control
python3 control_problem_lab.py test
```

预期检查：查看真实采样后的 greedy policy 为 (1,1)，Q(A)≈(1,2.368421)、Q(B)≈(2,2.631579)，并核对访问次数。

实验范围：这个确定性两状态例子用常数步长。成功运行不构成随机环境或深度 Q-learning 的收敛证明。

自测：下一状态 Q 值为 (2,0)，行为以 (0.9,0.1) 选动作，本次实际选了第二个动作；r=0、γ=0.9。三个 target 是多少？

解答：SARSA 为 0；Expected SARSA 为 0.9×1.8=1.62；Q-learning 为 1.8。只有先明确要评价的策略，才能判断哪个 target 合适。

完整推导与问题衔接：

- [TD 控制：SARSA、Expected SARSA 与 Q-learning](https://yingwen.io/zh/continual-rl/foundations/tabular/temporal-difference/)
- [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

原文、课程与实现：

- [Sutton & Barto · Reinforcement Learning, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：第 6 章及第 4.6 节：sample-based control 与广义策略迭代。
- [Watkins & Dayan · Q-learning](https://doi.org/10.1007/BF00992698)：表格控制及其收敛假设；不能直接外推到神经网络。

### 5 · 多步目标与时间信用分配

延迟到来的反馈，怎样影响更早的预测和决策？

一步 TD 每次只跨一个时间步传播信息。n-step 方法先累计 n 步实际奖励，再接上价值；$\lambda$-return 则按几何权重组合不同长度的目标。更长回报改变 bootstrap 误差、采样方差和反馈延迟，不意味着总是更好。

资格迹把最近访问过的特征方向压缩为一个向量 $e_t$。当前 TD error 到来时，不只更新当前状态，也按迹的大小更新过去相关参数。在线计算的迹不需要存整条轨迹，但它并没有给策略增加关于环境历史的信息。看不到过去线索的策略，不会仅因 critic 使用资格迹就获得记忆。

冻结价值参数后，可以推导前向回报与后向误差累积的恒等关系。参数每步变化时，普通 accumulating traces 并不在有限步长下逐次等于在线前向视图；true-online TD 针对线性近似修正了这个差异。递归网络的参数还通过内部状态影响未来输出，需要另外处理这条导数路径。

$$
e_t=\gamma\lambda e_{t-1}+x_t,\qquad \delta_t=R_{t+1}+\gamma w_t^\top x_{t+1}-w_t^\top x_t,\qquad w_{t+1}=w_t+\alpha\delta_t e_t
$$

这是固定 γ 的 accumulating linear TD(λ)。x 是特征，one-hot 情况对应表格。真实 episode 边界重置资格迹，但保留已学权重。

#### 动手与核对

[下载 credit_assignment_lab.py](https://yingwen.io/zh/continual-rl/download/credit_assignment_lab.py)

在保存该文件的目录运行：

```sh
python3 credit_assignment_lab.py online
python3 credit_assignment_lab.py test
```

预期检查：同一例子的普通 TD 最后为 2.0，true-online TD 与在线前向视图均为 1.75；不是浮点误差造成的不同。

实验范围：演示线性在线等价性；不提供非线性神经网络上的一般等价定理。

自测：两步轨迹奖励为 0、1，初值零、γ=.9、λ=.8、α=.1，特征分别为 (1,0)、(0,1)。最后更新是什么？

解答：第一步误差为 0。第二步迹为 (.72,1)，误差为 1，权重变为 (.072,.1)。λ=0 时第一项仍是 0。

完整推导与问题衔接：

- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)
- [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)

原文、课程与实现：

- [Sutton & Barto · Reinforcement Learning, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：第 7、12 章：n-step、前向视图与资格迹。
- [van Seijen & Sutton · True Online TD Learning](https://proceedings.mlr.press/v32/seijen14.html)：理解普通在线迹与 true-online 修正的区别。

### 6 · 函数逼近、采样分布与稳定性条件

不能为每个状态保存一行表格时，怎样共享经验？

线性近似用 $\hat v_w(s)=w^\top x(s)$ 把许多状态映射到共享参数。学习一个状态可能同时改变其他状态。对一个暂时固定的目标 $y$，损失为 $\tfrac12[y-w^\top x(s)]^2$，其负梯度是 $(y-w^\top x(s))x(s)$。若 $y$ 含下一状态的当前预测，仍在求导时把 $y$ 固定，就得到 semi-gradient TD。

共享带来两种不同后果。若两个状态都用特征 $x=1$，它们只能共享一个预测；真实价值分别为 0 和 2 时，就不可能同时预测准确。在两状态等权的平方误差下，最好的共享预测是 1。若允许不同特征，拟合能力可以提高，但一次更新仍会影响特征相似的其他状态。这种影响既可能是有益泛化，也可能是干扰。

共享参数意味着不一定能同时精确拟合所有状态。误差按哪些状态更常出现加权，因此数据分布本身成为算法的一部分。on-policy 线性 TD 在固定策略、适当遍历与满秩等条件下有稳定性结果；它通常求的是投影 Bellman 固定点，并非直接最小化真实价值均方误差。步长还要符合相应定理。

当函数逼近、bootstrap 和 off-policy 数据组合时，误差反馈可能发散，这就是 deadly triad 所指的组合风险。它在线性小例子中已会出现，并不是网络太深才产生。特征缩放也会改变同一个数值步长的实际作用。进入深度 RL 前，应先能指出一次更新的目标、梯度和采样分布。

$$
\begin{aligned}\delta_t&=R_{t+1}+\gamma w_t^\top x_{t+1}-w_t^\top x_t,\qquad w_{t+1}=w_t+\alpha\delta_t x_t,\\h(w)&=\mathbb E_d\!\left[x_t\bigl(R_{t+1}+\gamma w^\top x_{t+1}-w^\top x_t\bigr)\right]=b-Aw,\\A&=\mathbb E_d[x_t(x_t-\gamma x_{t+1})^\top],\qquad b=\mathbb E_d[R_{t+1}x_t].\end{aligned}
$$

第二行先固定任意参数 $w$，再按固定策略的平稳转移分布 $d(s)\pi(a\mid s)p(s',r\mid s,a)$ 取期望，定义平均更新方向 $h(w)$。它不是自适应轨迹上 $\mathbb E[\delta_t x_t]=b-Aw_t$ 的无条件恒等式：在线参数与后来的状态可能相关。矩阵 $A$ 的稳定性提供分析起点；样本轨迹的收敛还需要遍历性与步长等条件。

#### 动手与核对

[下载 gvf_lab.py](https://yingwen.io/zh/continual-rl/download/gvf_lab.py)

在保存该文件的目录运行：

```sh
python3 gvf_lab.py counterexample
python3 gvf_lab.py test
```

预期检查：比较 expected TD 的增长与相应梯度 TD 方法的行为；阅读脚本给出的明确线性反例和固定点，而不只观察训练回报。

实验范围：这是期望更新层面的反例。它不表示所有 off-policy TD 都发散，也不表示某个稳定修正自动解决非线性控制。

自测：把所有特征乘 10，同时将权重除 10，初始预测不变。保持相同 α，预测的单次改变量会不变吗？

解答：不会。TD 权重更新与特征成正比，而预测又乘一次特征；对同一误差，局部预测变化放大 100 倍。表示尺度与步长必须一起考虑。

完整推导与问题衔接：

- [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)
- [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)
- [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

原文、课程与实现：

- [Sutton & Barto · Reinforcement Learning, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：第 9 章线性预测；第 11 章离策略学习与 deadly triad。
- [Spinning Up · 算法分类](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)：按学什么、怎样获取数据和是否使用模型理解方法，不把分类树当成完整历史。

### 7 · 学习与规划的接口

得到一条经验后，除了直接更新价值，还能学习什么？

经验还可以训练后果模型：采取某动作会得到什么奖励，接下来到哪里。Dyna 用同一条真实转移做直接价值学习和模型学习，再选择若干状态动作，通过模型产生模拟 backup。这样把“获得新事实”和“用已有事实继续计算”分开。

模型规划并不免费。一次额外 backup 消耗计算，错误模型也可能反复强化错误价值。需要同时报告真实交互数和模拟更新数。优先扫描根据可能改变价值的大小安排 backup，比均匀分配计算更有针对性；其意义是调度，不是改变控制目标。

从这里可以理解后来的世界模型方法：模型可以是表格、神经网络或潜变量系统，规划可以是价值迭代、轨迹搜索或通过模拟经验训练策略。这些选择有共同接口，却不能只凭“用了模型”就视为同一算法。进入下一册时，应继续区分模型误差、价值误差和优化误差。

$$
\hat y=\hat r(s,a)+\gamma\sum_{s'}\hat p(s'\mid s,a)V(s'),\qquad Q(s,a)\leftarrow Q(s,a)+\alpha[\hat y-Q(s,a)]
$$

在贪心控制中可令 $V(s^{\prime})=\max_{a^{\prime}}Q(s^{\prime},a^{\prime})$。模型生成一次随机后果时用样本替代上式求和；随机模型不应仅保留最后一次观测。

#### 动手与核对

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

在保存该文件的目录运行：

```sh
python3 foundations_detail_lab.py dyna
python3 foundations_detail_lab.py test
```

预期检查：1000 次真实交互对应 5000 次规划更新；向右动作价值接近 (0.6561,0.729,0.81,0.9,1)。

实验范围：环境是小型确定性链，模型不需要处理复杂随机性；这不是世界模型基准。

自测：增加十倍模型 backup 后，用更少交互达到同一回报，是否已经说明算法更高效？

解答：只说明对应交互预算下的采样效率。还要报告模拟计算、模型训练和运行时间；如果模型错误，更多规划甚至可能更差。

完整推导与问题衔接：

- [Dyna 与模型学习](https://yingwen.io/zh/continual-rl/algorithms/dyna/)
- [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)
- [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)

原文、课程与实现：

- [Sutton & Barto · Reinforcement Learning, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：第 8 章：Dyna、模型误差与优先扫描。
- [Sutton · Dyna, an integrated architecture for learning, planning, and reacting](https://doi.org/10.1145/122344.122377)：直接学习、模型学习和规划的共同架构。

下一步：表格逐项保存预测，共享参数让一次更新影响多个状态。状态很多时，可继续补函数逼近与深度方法；研究持续预测、平均奖励或单生命期控制时，也可以直接使用经典工具。持续控制章随后把评价对象扩展为完整学习器，重新明确目标与比较条件。

## 现代 DRL：可学习表示与稳定控制

深度强化学习用神经网络表示价值、策略或模型。它与持续学习可以同时成立，也可以研究固定任务或离线数据。本册先解释 DQN、actor–critic、PPO 与连续控制的核心接口，再按部分可观测性、探索、分布、离线数据、模型、约束和多智能体等条件选择并列研究分支。

学习目标：

- 能区分 target、prediction、stop-gradient 和实际更新参数。
- 能解释 replay、target network、Double、GAE、PPO clipping 各改变哪个环节。
- 能把一个完整训练循环与机制小实验区分开，并定位原始实现。

### 1 · 神经网络改变了哪些条件

把表格换成网络，为什么不是只换一种存储方式？

神经网络 $Q_\theta(s,a)$ 同时学习表示与输出。一次更新可以改变许多未访问状态的预测，后续数据也可能改变已学特征。控制又会改变行为和数据分布，因此网络看到的输入、监督目标和自身表示都可能移动。监督学习中固定标签、独立采样的直觉只能部分沿用。

计算 TD 平方误差时，先用当前约定构造目标 $y$，再只对预测求导，是一个明确算法选择。若让梯度穿过下一状态的 bootstrap，会变成另一条更新。即使目标暂时冻结，优化器只是在拟合这批目标；损失下降也不等于真实策略回报上升。

deadly triad 描述函数逼近、bootstrap、off-policy 的交互风险。replay 改变采样，target network 减缓标签变化，梯度限制控制更新幅度；这些机制保留了共享参数带来的耦合，仍需检查其稳定性。调试时先检查 terminal mask、target 梯度、张量维度和数据分布，再考虑网络规模。

$$
L(\theta)=\tfrac12\mathbb E_{(s,a,r,s')\sim\mathcal D}\bigl[Q_\theta(s,a)-\operatorname{sg}(y)\bigr]^2,\qquad \nabla_\theta L=\mathbb E[(Q_\theta-y)\nabla_\theta Q_\theta]
$$

sg 表示停止梯度；D 是训练采样分布。此梯度是对冻结目标的回归梯度，不是对完整 Bellman 误差所有依赖路径的梯度。

#### 动手与核对

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

在保存该文件的目录运行：

```sh
python3 foundations_detail_lab.py deep-value
python3 foundations_detail_lab.py test
```

预期检查：两层 ReLU 小网络的最终 Q 接近 [[.9,.1],[1,-1]]；检查解析参照和 max_error，而不只看 loss。

实验范围：这是有限 MDP 上带 replay、target network 和 Double target 的教学网络，不是 Atari DQN 复现。

自测：固定一批样本后，TD loss 降到零，是否保证目标策略最优？

解答：不保证。目标本身可能错误，样本可能不覆盖重要状态动作，网络还可能在未见区域误泛化。零训练误差仅说明拟合了这批目标。

完整推导与问题衔接：

- [深度价值学习](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)
- [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)

原文、课程与实现：

- [Sutton & Barto · Reinforcement Learning, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：第 9、11 章：函数逼近、semi-gradient 与不稳定性。
- [Spinning Up · 算法分类](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)：先区分学习价值、策略和模型，再理解它们如何组合。

### 2 · DQN：经验重放、目标网络与 Double target

如何避免价值网络用迅速变化的自身预测反复训练自己？

DQN 将新转移放入 replay buffer，从中抽取小批量样本训练。这样可以重用经验并改变相邻训练样本的相关性，但缓冲区分布不等于当前策略分布。目标网络 $\bar\theta$ 在若干更新内保持不变，使回归标签变化更慢；随后再同步或缓慢跟踪在线参数。

普通 DQN 用目标网络同时选择最大动作和评价它。当估计噪声与最大值选择耦合时，会出现过估计倾向。Double DQN 用在线网络选动作、目标网络评价该动作，分开两个角色来减轻这种选择偏差。两个网络仍有相关性，其他估计误差也仍可能存在。

完整循环还包括探索策略、初始采样期、每步更新次数、target 同步周期、真实终止处理和评价协议。比较算法时这些环节不能无意中改变。原始 DQN 代码有 Atari 图像预处理和旧 Torch 依赖；学习核心逻辑可以从 NeuralQLearner 开始，但不能把核心文件独立运行当成完整实验。

$$
\begin{aligned}y_{\rm DQN}&=r+\gamma(1-d)\max_aQ_{\bar\theta}(s',a),\\a^*&=\arg\max_aQ_\theta(s',a),\\y_{\rm Double}&=r+\gamma(1-d)Q_{\bar\theta}(s',a^*).\end{aligned}
$$

d 只表示该转移真实终止。两种 target 都停止梯度；预算截断不应不加区分地设 d=1。

#### 动手与核对

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

在保存该文件的目录运行：

```sh
python3 foundations_detail_lab.py deep-value
python3 foundations_detail_lab.py test
```

预期检查：先读 target 的动作选择与网络求值位置；再运行 test，检查 Double target 与网络梯度的单元测试。

实验范围：同一个小实验只承担目标与梯度核对，不提供图像编码、Atari 协议或论文性能结果。

自测：在线网络在 s′ 给出 (5,4)，目标网络给出 (1,3)，r=0、γ=.9、非终止。两种 target 分别是多少？

解答：DQN 为 .9×3=2.7。Double 由在线网络选第一个动作，再由目标网络评价，因此为 .9×1=.9。差异来自选择与评价分工，不是折扣不同。

完整推导与问题衔接：

- [深度价值学习](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)

原文、课程与实现：

- [DeepMind · DQN 原始实现](https://github.com/google-deepmind/dqn/blob/master/dqn/NeuralQLearner.lua)：核心学习器：经验采样、目标网络与 TD 更新；原仓库使用 Lua/Torch。
- [van Hasselt、Guez、Silver · Double DQN](https://arxiv.org/abs/1509.06461)：选择与评价分离的原论文；对照普通 DQN target。

### 3 · 策略梯度与 actor–critic

不通过枚举最大 Q，能否直接改善动作分布？

参数化策略 $\pi_\theta(a\mid s)$ 直接输出动作分布。对轨迹概率求导，可把回报的梯度写成动作 log-probability 的梯度乘回报。这里环境规律不直接依赖策略参数，因此不需要对未知转移模型求导。REINFORCE 用完整回报估计该权重。固定状态后，$\sum_a\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s)=\nabla_\theta\sum_a\pi_\theta(a\mid s)=0$；所以减去与本次动作无关的 baseline 不改变期望梯度，却可能降低方差。

advantage 定义为 $A_\pi(s,a)=q_\pi(s,a)-v_\pi(s)$，表示这个动作比当前策略在该状态的平均选择好多少。actor–critic 用 critic 的价值预测构造更早取得、通常方差较低的更新权重，例如 TD error 或 advantage 估计。actor 改变策略，critic 评价随之变化的策略，二者形成耦合闭环。critic 偏差仍会影响 actor。

策略梯度的采样分布依赖当前策略。对同一批旧数据无限优化 log-probability surrogate，不能保证继续沿正确的回报梯度移动。实现时还要区分用于策略梯度的回报权重和用于报告性能的 episode return。

$$
\nabla_\theta J=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t\nabla_\theta\log\pi_\theta(A_t\mid S_t)\bigl(G_t-b(S_t)\bigr)\right]
$$

这里 $J=\mathbb E[G_0]$，初始分布与有限时域固定，$G_t$ 从时刻 $t$ 重新计折扣，因此外面还有 $\gamma^t$。baseline 在 actor 更新中停止梯度；其随机拟合过程还须满足给定状态后与本次动作无关。仅有 $b(S_t)$ 的函数签名不足以保证这一点，例如用同一条样本先拟合 baseline 再原地扣除，可能引入依赖。

#### 动手与核对

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

在保存该文件的目录运行：

```sh
python3 foundations_detail_lab.py policy
python3 foundations_detail_lab.py test
```

预期检查：单步 bandit 中好动作概率约为 0.9704；结合 test 中的梯度检验，区分概率改善与长时信用分配。

实验范围：这里实现的是简化 PPO bandit，不包含长轨迹 critic、GAE rollout 或连续动作。

自测：二动作 softmax 的初始概率各为 .5，选到动作 1，其 advantage 为 2。对两个 logits 的 ascent 方向是什么？

解答：若 logits 按动作 0、1 排列，方向为 2×(-.5,.5)=(-1,1)。它提高被选且优于 baseline 的动作概率，而不是直接把概率加 2。

完整推导与问题衔接：

- [策略梯度与优势估计的完整实现](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/)
- [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)
- [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

原文、课程与实现：

- [Sutton & Barto · Reinforcement Learning, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：第 13 章：策略梯度定理、REINFORCE、baseline 与 actor–critic。
- [Spinning Up · 策略优化入门](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)：从 log-derivative 逐步理解梯度；特别注意 surrogate loss 不是性能指标。

### 4 · GAE 与 PPO：估计优势和限制批内策略变化

怎样利用一批轨迹多次更新，而不忽略数据来自旧策略？

GAE 将一连串 TD errors 按 $(\gamma\lambda)^k$ 加权，得到 advantage 估计。它在一步 critic bootstrap 与较长回报之间调节依赖。rollout 因时间限制结束时可以 bootstrap 最后价值，但不能把 reset 后另一条轨迹的 TD error 接入同一条 GAE 递推。

PPO 保存采样策略的 log-probability，定义新旧动作概率比 $r_t(\theta)$，并用 clipped surrogate 做多轮小批量更新。对正优势，过度增加动作概率不再得到额外目标收益；对负优势，过度降低概率也不再得到额外收益。这不是对所有状态施加硬 KL 约束。

GAE 是优势估计器，PPO 是策略更新规则，可以分别讨论。完整实现需要冻结采样时的旧 log-probability 和本轮优势、正确处理边界、训练 critic，并监测 KL 与 clip fraction。Spinning Up 的 PPO 还提供 KL 超阈值时提前停止策略更新的实现。

$$
\begin{aligned}\hat A_t&=\sum_{k=0}^{K-1}(\gamma\lambda)^k\delta_{t+k},\qquad r_t(\theta)=\frac{\pi_\theta(A_t\mid S_t)}{\pi_{\rm old}(A_t\mid S_t)},\\L^{\rm clip}(\theta)&=\mathbb E[\min(r_t\hat A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)].\end{aligned}
$$

GAE 求和只沿同一轨迹到可用边界；terminal 与 truncation 决定最后 TD error 的 bootstrap。PPO 对上述 L 做梯度上升。

#### 动手与核对

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

在保存该文件的目录运行：

```sh
python3 foundations_detail_lab.py test
```

预期检查：测试覆盖 PPO clipping 的符号分支与 GAE 截断处理；随后对照公开 PPO 的 buffer 和 update 函数。

实验范围：本地测试只验证目标和边界；完整 rollout、并行采样和环境依赖请以公开实现及其配置为准。

自测：优势为 -2，概率比为 .5，ε=.2，clipped surrogate 是多少？

解答：min(.5×-2,.8×-2)=min(-1,-1.6)=-1.6。负优势乘法会反转大小关系；不能仅把正优势的直觉照搬。

完整推导与问题衔接：

- [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)
- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

原文、课程与实现：

- [Schulman 等 · Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)：优势估计的偏差、方差与 λ 权重。
- [Schulman 等 · PPO](https://arxiv.org/abs/1707.06347)：PPO-Clip 与 PPO-Penalty 是不同变体。
- [Spinning Up · PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)：配合伪代码理解采样、优势估计和批内更新。
- [Spinning Up · PyTorch PPO 核心代码](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py)：PPOBuffer.finish_path、compute_loss_pi 与 update；这是公开教学实现，不是本地 bandit 的依赖。

### 5 · 连续控制：TD3 与最大熵 SAC

动作是连续向量时，怎样完成价值改善中的最大化？

连续动作无法逐项枚举。确定性 actor 可以近似输出高 Q 动作，通过 critic 对动作的梯度训练。TD3 使用两个 critic 的较小目标值、延迟 actor 更新，并在 target action 附近加入裁剪噪声。这些机制分别处理价值高估、critic 尚不准确时的策略跟随，以及对狭窄误差峰的利用。

SAC 不只是给 TD3 加随机动作。它把策略熵纳入目标，在回报和保持动作分布之间权衡，学习随机 actor。critic target 包含下一动作的负 log-density；actor 则比较熵成本与 critic 估值。熵系数与奖励尺度共同决定偏好，不能把含熵目标和原始回报当作同一量。

实践中，连续策略常由高斯变量经 tanh 变换得到。计算 log-probability 必须包含变换的 Jacobian 修正；直接截断动作再沿用高斯密度不是同一分布。TD3 的 target smoothing 噪声与 SAC 的策略随机性也不是同一个参数。

$$
\begin{aligned}y_{\rm TD3}&=r+\gamma(1-d)\min_iQ_{\bar\phi_i}(s',\tilde a'),\\y_{\rm SAC}&=r+\gamma(1-d)\left[\min_iQ_{\bar\phi_i}(s',a')-\alpha_{\rm ent}\log\pi_\theta(a'\mid s')\right],\quad a'\sim\pi_\theta,\\L_\pi^{\rm SAC}&=\mathbb E_{s\sim\mathcal D,a\sim\pi_\theta}[\alpha_{\rm ent}\log\pi_\theta(a\mid s)-\min_iQ_{\phi_i}(s,a)].\end{aligned}
$$

TD3 的 ã′ 是目标 actor 输出加裁剪噪声再裁剪到动作范围；SAC 的 a′ 从当前随机 actor 采样。critic target 停止梯度，SAC actor loss 对重参数化动作路径求导。

#### 动手与核对

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

在保存该文件的目录运行：

```sh
python3 foundations_detail_lab.py soft-control
python3 foundations_detail_lab.py test
```

预期检查：二动作 soft bandit 的策略接近 (.119203,.880797)，soft value 约 1.063464；用解析 softmax 校验。

实验范围：它隔离最大熵目标，不是连续 SAC 或 TD3 实现。完整连续控制代码在上方公开实现中。

自测：奖励为 (0,1)，熵系数为 .5。最优 soft bandit 策略会对好动作给概率 1 吗？

解答：不会。概率正比于 exp(r/.5)，好动作概率为 exp(2)/(1+exp(2))≈.880797。这里的随机性来自优化目标，不是估计尚未收敛。

完整推导与问题衔接：

- [最大熵控制](https://yingwen.io/zh/continual-rl/algorithms/soft-control/)
- [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)

原文、课程与实现：

- [Fujimoto 等 · TD3](https://proceedings.mlr.press/v80/fujimoto18a.html)：三个稳定化机制及其消融。
- [Haarnoja 等 · SAC](https://proceedings.mlr.press/v80/haarnoja18b.html)：最大熵 actor–critic 的原始论文；早期版本含独立 V 网络。
- [Spinning Up · TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html)：目标噪声、双 critic、延迟策略更新的完整循环。
- [Spinning Up · SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)：不含独立 V 网络的版本，以及熵项约定。
- [Spinning Up · PyTorch TD3](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/td3/td3.py)：compute_loss_q 与延迟 actor/target 更新。
- [Spinning Up · PyTorch SAC](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py)：critic target、重参数化 actor loss 与 target 更新。

### 6 · 世界模型与规划：预测误差怎样进入决策

学会预测环境，为什么还不等于学会最优行动？

世界模型把观测历史编码为内部状态，并预测下一内部状态、奖励或观测。它可以支持模型预测控制：每步从当前状态比较候选动作序列，只执行首个动作，再用真实观测重规划。也可以在模型中产生轨迹，以此训练价值与策略。

两种路径都需要明确模型在哪个数据分布上可靠。策略优化会主动寻找高预测奖励的区域，也可能找到模型最不准确的区域。多步想象还会把早期预测误差带入后续输入，因此应同时检查一步模型损失、长时轨迹误差和真实环境中的控制收益。

Dyna、MPC、MCTS 与 Dreamer 因此应按接口区分：Dyna 强调真实学习与模拟 backup 的结合；MPC 强调执行一小段再重规划；MCTS 把搜索预算分配到树上；Dreamer 类方法从潜在模型想象训练行为。它们可能共享部件，但不是可互换的名称。

$$
(a_0^*,\ldots,a_{H-1}^*)\in\arg\max_{a_{0:H-1}}\mathbb E_{\hat p}\!\left[\sum_{k=0}^{H-1}\gamma^k\hat r_k+\gamma^H\hat V(z_H)\right]
$$

这是有限时域 MPC 的一个形式。z 是模型状态，H 是规划长度，末端价值弥补规划截断。实际只执行 $a_0$，再重新估计状态并规划。

#### 动手与核对

[下载 knowledge_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/knowledge_algorithms_lab.py)

在保存该文件的目录运行：

```sh
python3 knowledge_algorithms_lab.py planning
python3 knowledge_algorithms_lab.py test
```

预期检查：MPC 演示首个动作是 1，候选序列为 (1,1,0)，得分 -1.02；再观察 prioritized sweeping 的反向传播次序。

实验范围：模型和动作集合是教学规模；不包含潜变量学习、MCTS 或 Dreamer 训练。

自测：模型只预测下一特征的期望，能否直接用 V(E[X]) 替代 E[V(X)]？

解答：只有特定条件下可以，例如 V 对该特征线性。若 X 等概率为 ±1，$V(x)=x^2$，则 V(E[X])=0，而 E[V(X)]=1。

完整推导与问题衔接：

- [Dyna 与模型学习](https://yingwen.io/zh/continual-rl/algorithms/dyna/)
- [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)
- [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)

原文、课程与实现：

- [Sutton & Barto · Reinforcement Learning, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：第 8 章提供学习与规划的共同基础。
- [Hafner 等 · DreamerV3](https://arxiv.org/abs/2301.04104)：潜在世界模型、想象行为学习及跨任务实验。
- [DreamerV3 作者代码](https://github.com/danijar/dreamerv3)：完整模型与行为学习系统；对照配置阅读，不把小型 MPC 当作该算法复现。

### 7 · 部分可观测性与循环状态

当前图像一样，最优动作却不同，网络应当学什么？

环境状态 $S_t$ 与智能体收到的观测 $O_t$ 不一定相同。若过去线索影响现在的动作，必须从历史构造内部状态 $h_t$。已知模型时可以维护对隐藏状态的 belief；模型未知时，可用循环网络学习一个压缩历史。这个状态是行动所需信息的表示，不是价值函数的别名。

循环活动和参数有不同生命周期。活动记录当前交互中的情境，参数记录如何根据经验更新活动及行动。仅让 hidden state 持续变化，并不能说明算法在长期积累知识；反过来，频繁清空活动也可能破坏任务所需的记忆。

BPTT 沿展开的计算图传播参数对后续状态的影响。截断 BPTT 节省内存，却会删去截断点之前的梯度路径；前向记忆仍可保留，因此“记住了信息”和“能学会记住信息”必须分开。RTRL 用前向敏感度维护这些导数，但一般计算代价很高。

$$
h_t=f_\theta(h_{t-1},O_t,A_{t-1}),\qquad \frac{\partial h_t}{\partial\theta}=\left.\frac{\partial f_\theta}{\partial\theta}\right|_{h_{t-1}}+\frac{\partial f_\theta}{\partial h_{t-1}}\frac{\partial h_{t-1}}{\partial\theta}
$$

这是固定参数运行序列时的链式法则。第一项是本步直接影响，第二项是过去参数经递归活动的影响。在线每步改变参数后，历史活动并未按新参数重算，必须另外说明导数近似。

#### 动手与核对

[下载 state_meta_lab.py](https://yingwen.io/zh/continual-rl/download/state_meta_lab.py)

在保存该文件的目录运行：

```sh
python3 state_meta_lab.py state
python3 state_meta_lab.py test
```

预期检查：RTRL 与完整 BPTT 的三个梯度分量一致；TBPTT-2 的早期线索输入权重梯度为零；训练后的线索预测约为 ±.8。

实验范围：包括已知模型过滤、固定参数导数和监督序列实验；不代表完成 recurrent PPO 或 POMDP 控制基准。

自测：在一个截断点 detach hidden state，但数值不清零，会改变之后的前向输出吗？

解答：不一定。相同数值可以产生相同输出，但梯度不再经过截断点返回更早的运算。需要同时检查信息保留与学习信用。

完整推导与问题衔接：

- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)
- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

原文、课程与实现：

- [Williams & Zipser · A Learning Algorithm for Continually Running Fully Recurrent Neural Networks](https://doi.org/10.1162/neco.1989.1.2.270)：RTRL 的原始前向敏感度方法。
- [Hausknecht & Stone · Deep Recurrent Q-Learning](https://arxiv.org/abs/1507.06527)：把循环状态用于部分可观测深度 Q-learning。

### 8 · 大规模训练：数据、算法与硬件共同决定效率

增加环境与 GPU 后，更多计算怎样成为更快的策略改善？

先区分环境推进、动作推理、轨迹存储与参数更新。A3C 让 worker 提交本地梯度，IMPALA 让 actor 提交轨迹，再在 learner 用当前参数求梯度；两者分别面临过期梯度与行为策略失配。

V-trace 把本步误差校正与后续误差传播分开。集中推理则把许多环境的请求组成批次。OpenAI Five 和 SEED RL 展示了采样、推理与优化器分工，GEAR 进一步处理大模型训练中的经验选择与搬运。

样本年龄、样本复用、推理延迟与吞吐不是同一个量。先用固定数据核验估计器，再测数据管线，最后按墙钟、交互与硬件成本比较策略学习。

$$
r_{\mathrm{reuse}}=\frac{B\,U}{F},\qquad \Delta\tau=\tau_{\mathrm{use}}-\tau_{\mathrm{action}}
$$

F 为每秒新增转移数，U 为每秒 learner 更新数，B 为每次更新参与 loss 的转移数。复用率与数据年龄要分开记录。

#### 动手与核对

[下载 distributed_systems_lab.py](https://yingwen.io/zh/continual-rl/download/distributed_systems_lab.py)

在保存该文件的目录运行：

```sh
python3 distributed_systems_lab.py demo
python3 distributed_systems_lab.py test
```

预期检查：两步 V-trace 标签为 1.6 与 2；有界队列减小数据年龄但丢弃部分经验。

实验范围：标准库确定性核验，不是集群训练吞吐测量。

自测：为何学习 GPU 利用率升高，策略改善反而变慢？

解答：可能重复消费更多过期经验、训练分布改变、裁剪比例升高，或只是在处理更多 padding。必须把有效数据、算法更新和端到端策略表现分别测量。

完整推导与问题衔接：

- [大规模训练系统：A3C、IMPALA、OpenAI Five、SEED RL 与 GEAR](https://yingwen.io/zh/continual-rl/foundations/deep/systems/)
- [深度价值学习](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)
- [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)
- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)
- [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

原文、课程与实现：

- [IMPALA](https://proceedings.mlr.press/v80/espeholt18a.html)：actor–learner 解耦与 V-trace。
- [GEAR](https://proceedings.mlr.press/v202/wang23aj.html)：轨迹分片、索引一致性与 GPU-centric 数据收集。

### 9 · 多智能体：合作的探索与信用，开放式学习的评估与目标

合作如何发现并学会配合？竞争和开放式合作又怎样确定每一轮该学什么，以形成可检验的策略改善？

合作多智能体强化学习的主线是结构化探索和信用分配：前者发现有效的联合行为，后者把共同反馈转成各个策略的学习信号。Q-DPP、COMA、价值分解与联合策略优化分别处理这些环节。

顺序优化与顺序行动是不同操作。HATRPO/HAPPO、A2PO研究更新时如何考虑前序策略的变化；MAT用条件序列生成联合动作。PR2/GR2则研究对手会如何响应，以及怎样建模不同推理层次。每种方法都必须声明训练和执行可见的信息。

自对弈至少有两条不同的学习逻辑。FSP/NFSP 回应并保留历史策略；AlphaGo Zero / AlphaZero 用当次搜索的访问分布训练策略，用实际对局结果训练价值。MuZero 再把已知规则搜索中的转移替换为学得的模型。先用一棵两层博弈树区分搜索、真实行动和训练标签，才能理解这些方法怎样构成闭环。

竞争与开放式合作的核心是评估并构建每轮学习目标。希望获得怎样的改善，必须先由评价对象规定。固定参照下扩张自身可行策略集合，可以保留旧解的最优值；这不等于每轮训练出的策略都更好。这里的开放式学习特指开放式多智能体学习：代表性脉络从竞争自对弈、PSRO 与竞争多样性发展，温颖及合作者以 COLE、HOLA 拓展合作伙伴课程。

Balduzzi 的 gamescape 给出了理解这一外层问题的几何语言：一个策略的坐标是它对不同对手的收益，种群的混合形成这些坐标的凸包。剪刀石头布说明击败前任仍可绕回原处；增加不同回应方向则可能扩大可选能力。几何扩大、固定参照下的种群价值和完整游戏的可利用性需要分别评价。

$$
J_k(\pi)=\mathbb E_{\xi\sim\mu_k}u(\pi,\xi),\qquad \mu_{k+1}=\mathcal M(\mathcal P_{k+1},\widehat U_{k+1})
$$

μ 是当前对手或伙伴分布；u 是明确规定的交互收益。外层根据策略档案 P 与评价 U 构造下一轮目标，内层学习对当前目标作响应。不同轮的目标可能不同。

#### 动手与核对

[下载 marl_objectives_lab.py](https://yingwen.io/zh/continual-rl/download/marl_objectives_lab.py)

在保存该文件的目录运行：

```sh
python3 marl_objectives_lab.py demo
python3 marl_objectives_lab.py test
```

预期检查：精确枚举检查：COMA型基线保留梯度期望；同时最佳响应可以循环；扩张受限种群的full-game gap可由2增至20。

实验范围：这是小型博弈机制和反例，不是PR2、MAT、PSRO或COLE的完整神经系统复现。

自测：种群加入新策略后，为什么不能仅凭“策略更多了”就断言当前均衡更不容易被利用？

解答：当前元均衡本身可能改变，遗漏的对手仍可能更强。集合包含关系只保证在固定评价准则下，保留旧解的最优可行值不变差；不保证任意新元均衡的完整博弈可利用性逐轮下降。

完整推导与问题衔接：

- [合作主线与共同设定：结构化探索和信用分配](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent/)
- [自对弈与开放式多智能体学习：评估、目标与策略种群](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent-populations/)
- [对手建模与递归推理：怎样学习响应](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent-reasoning/)
- [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)
- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)
- [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)
- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)
- [元学习与学习规则的适应](https://yingwen.io/zh/continual-rl/algorithms/meta/)

原文、课程与实现：

- [Albrecht、Christianos、Schäfer · MARL Book](https://www.marl-book.com/)：第3–6章的交互模型与解概念，第9–11章的深度算法、实践与环境。
- [MAT · Multi-Agent Reinforcement Learning is a Sequence Modeling Problem](https://arxiv.org/abs/2205.14953)：固定策略的优势分解与条件动作生成。
- [Lanctot et al. · A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://arxiv.org/abs/1711.00832)：经验元博弈、元策略求解与近似最佳响应。

下一步：进入 CRL 时，可按所选问题补充深度研究分支：先确定是否需要神经表示，再规定信息与数据权限，以及学习状态怎样长期保留。评价对象随之扩展为这些模块共同组成的完整学习器，包含训练期间的行动和更新。

## CRL：长期交互中的学习、适应与知识积累

持续强化学习研究受资源限制的完整智能体怎样在长期交互中适应环境、选择行动并积累知识。可以先从一个反复经过的岔路口开始：路线奖励变了，当前动作规则相同的两个实现，会因是否继续更新而获得不同收益。经典或深度方法都可成为其部件。先规定奖励、信息、初始化、调参与重置权限，再定义表现与可实现的比较者，最后选择机制。

学习目标：

- 能把“持续”写成环境、信息、reset、内存、计算与评价协议。
- 能分开状态、预测、信用分配、流式更新和元学习，并说明它们怎样组合。
- 能从一个最小反例出发，设计具有对照、独立测试和明确估计对象的研究实验。

### 1 · 完整学习器、目标与长期交互协议

比较的是当前策略，还是包含内部状态与更新规则的完整智能体？

Continuing 指任务没有自然终点；online 指数据到来后学习；严格 streaming 通常进一步限制经验重放与逐步预算；continual 关注学习能否长期持续。这些维度互不等价。固定环境中也可能需要不断学习，分任务的基准则可能允许任意 reset。

比较持续学习方案时，写清六项：哪些规律或任务改变，智能体能否看到变化标记，能否重置环境，哪些内部学习状态跨阶段保留，内存与计算如何计量，以及最终估计什么表现。设计者在完整测试生命期上挑超参数，与智能体根据已到达经验更新参数，是两种不同的信息权限。

当前动作分布相同，不意味着未来学习能力相同。一个学习率为零的贪心智能体和一个持续更新的智能体可以从相同 Q 值出发，却在奖励改变后获得不同收益。价值仍可在历史和完整算法条件下定义；把历史写进状态，并不自动解决表示、覆盖与有限计算问题。

长期在线效用要把学习期间的奖励也计入。平均奖励率是 continuing 问题的一种目标，折扣目标也是另一种合法选择；二者可能偏好不同策略。平均奖励的稳态存在、对初始状态的依赖和差分价值定义需要链结构条件。CRL 不要求所有任务一律改用 average reward。

这组正文依次明确交互与优化准则、奖励如何表达目标、怎样比较完整学习器，再展开平均奖励这一具体目标。实验设计提供贯穿这些问题的评价方法，从开始设计比较时就可以查阅，并在各机制章中使用。

$$
\begin{aligned}\bar J_T(\mathcal A,e)&=\mathbb E_{\mathcal A,e}\!\left[\frac1T\sum_{t=0}^{T-1}R_{t+1}\right],\\g(\mathcal A,e)&=\lim_{T\to\infty}\bar J_T(\mathcal A,e)\quad\text{若极限存在}.\end{aligned}
$$

A 包含行动规则、更新规则、内部状态与初始化；e 指定世界条件。有限生命期平均表现不要求稳态存在。一般非平稳学习过程的极限可能不存在；冻结策略的价值又是另一评价对象。

#### 动手与核对

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

- [岔路迷宫：累计收益、冻结诊断、保持与迁移的逐步图解](https://yingwen.io/zh/continual-rl/foundations/objectives/#lesson-maze-evaluation)
- [从一开始查阅：实验设计与评价方法](https://yingwen.io/zh/continual-rl/experiments/)
- [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)
- [奖励假设与奖励设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/)
- [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)
- [平均奖励与差分价值](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)

原文、课程与实现：

- [Abel 等 · A Definition of Continual Reinforcement Learning](https://arxiv.org/abs/2307.11046)：相对于 agent basis 定义持续学习；不把 CRL 限定为外部任务切换。
- [Elelimy 等 · Rethinking the Foundations for Continual Reinforcement Learning](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_243.pdf)：学习规则、历史依赖世界与偏离比较；与从初始世界比较总回报的含义分开。
- [Mesbahi 等 · Lifetime tuning is incompatible with continual reinforcement learning](https://proceedings.mlr.press/v267/mesbahi25a.html)：外部设计者用完整部署寿命调参与在线因果更新的信息权限不同。
- [Sutton & Barto · Reinforcement Learning, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：第 10.3 节：平均奖励和差分价值。

### 2 · 智能体状态与预测知识的分工

长期经验应当形成什么可供后续决策使用的知识？

状态构建回答“当前历史中哪些信息需要保留”。预测知识回答“在指定行为条件下，未来某个信号会怎样”。例如，走廊入口的提示灯决定岔路应该向哪边走；保存提示信息是状态构造。在此基础上，估计沿某条路线到出口还要走几步，是一个预测问题。可以拥有足够记忆而预测不准，也可以准确预测走廊颜色而仍不知道该向哪边走。

GVF 用目标策略 $\pi$、累积信号 $C$ 和延续规则 $\gamma$ 规定一个预测。把任务奖励换成温度、接触次数或能耗，就能定义不同问题。GVF 是问题规格，不是 TD 的竞争算法；TD、GTD、Emphatic TD 等是在不同采样和稳定性条件下学习它的办法。

“按指定路线到出口还要几步”可以具体规定为：$C_{t+1}=1$；到出口的转移令 $\gamma_{t+1}=0$，其他转移令其为 1；$\pi$ 是所问的路线策略。若三步到出口，回报就是 $1+1+1=3$。这些规格先由问题提出者给定，GVF 学习器只从经验估计答案，不会自动发明问题。期望到达步数有限时，这个预测才有有限的期望值。

主任务奖励可以仍是到出口得到 10，其他时候为 0。到达步数预测并不把这份奖励改成每步 1。它可以帮助构造决策特征或比较路线，但是否提高主任务回报仍需检验。若决定把辅助信号加入行为奖励，就另外改变了控制目标；“增加知识”和“增加奖励项”不是同一个操作。

同一行为流可训练多个 GVF，但行为必须覆盖问题所需动作。若目标策略与行为不同，需要说明 off-policy 校正和函数逼近稳定性。把这些预测作为 state feature 或 model component 还需要验证它们对控制的充分性与效用；准确预测一些信号，并不自动形成充分状态。

$$
G_t^{\rm GVF}=\sum_{k=0}^{\infty}\left(\prod_{j=1}^{k}\gamma_{t+j}\right)C_{t+k+1},\qquad v(s)=\mathbb E_\pi[G_t^{\rm GVF}\mid S_t=s]
$$

空乘积为 1；延续权重可依赖转移。需保证回报与期望存在，例如有界信号配合统一小于 1 的延续上界。某个 GVF 的延续变零，只停止它所问的未来，不必重置真实环境。

#### 动手与核对

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

### 3 · 信用分配与流式学习：反馈路径和逐步预算

反馈怎样传到过去，与每步允许做多少计算，有什么区别？

时间信用分配决定当前反馈应当影响哪些过去预测、动作或参数路径。流式协议决定数据何时使用、能保存什么、是否可重放以及每步算多久。资格迹可以用于流式学习，也可以与回放或批处理共存；这两个概念不应合并成一个算法类别。

本节和下一节共同对应“信用分配、流式学习与元学习”一组：先说明反馈更新谁，再规定更新使用的数据与计算预算，最后讨论更新规则自身能否从经验中改进。这三个维度可以共同描述一个学习器。

在没有 replay 和 target network 的严格协议下，观测尺度、奖励尺度、迹长度和 actor–critic 耦合会影响更新稳定性。Stream-X 一类工作研究这些机制的组合。按当前输入或梯度缩放步长，也不必是元学习：关键是规则有没有通过后续学习效果被训练。

有效步长可以先从局部输出变化理解。线性预测器沿迹更新后，当前预测改变量是 $\alpha\delta_t x_t^\top e_t$。相同数值步长在特征尺度或迹范数改变后，可能导致很不同的输出变化。对这种尺度进行控制不能替代对长时信用分配或 off-policy 偏差的分析。

$$
\Delta w_t=\alpha\delta_t e_t,\qquad \Delta\hat v(S_t)=x_t^\top\Delta w_t=\alpha\delta_t x_t^\top e_t
$$

等式对固定线性特征准确；神经网络中以梯度代替 x 只是局部一阶近似，参数更新还会改变梯度与表示。

#### 动手与核对

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

### 4 · 学习规则的适应与跨任务元学习

什么时候需要学习“怎样学习”，而不仅是更新策略？

明确了信用路径与更新预算以后，还可以让步长、更新目标等学习规则参数根据后续学习效果改变。这是同一组中的第三个维度；元学习方法仍需说明自身的信用分配和数据协议。

在线元梯度把步长、训练目标或更新规则参数记为 $\eta$，研究它们怎样影响后续预测误差或回报。内层权重持续更新；外层通过这些更新的影响调整学习规则。它可以发生在同一条经验流上，不要求一组可重置任务。

跨任务 meta-RL 则先规定训练任务分布和任务内适应过程。MAML 学习便于少量梯度适应的初始化；RL² 或上下文方法可把任务适应放在循环活动或隐变量中。测试时是否还更新网络权重、是否清空上下文、能否得到任务边界，都必须明确。

二者共享对学习过程的关注，但不共享所有评价条件。某个规则在新任务上适应快，不代表它在一次没有参数重置的长寿命中始终有效。固定数据下求内层更新的导数，也不自动包含策略改变导致的采样分布导数。

$$
w_{t+1}=F_\eta(w_t,\xi_t),\qquad H_{t+1}=\frac{\partial F_\eta}{\partial w_t}H_t+\frac{\partial F_\eta}{\partial\eta},\qquad H_t=\frac{\partial w_t}{\partial\eta}
$$

这里先固定经验 ξ 和元参数，H 传播过去学习对 η 的敏感度。对 RL 期望目标求完整梯度时，还要讨论数据采样路径与近似。

#### 动手与核对

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

### 5 · 知识保留与学习可塑性

一个学了很久的网络变差，是忘记了，还是学不进了？

遗忘要比较同一旧任务在新学习前后的表现。可塑性损失要比较有学习历史的网络与合适对照在同一新任务上的学习过程。两种失败可能同时发生，因此需要旧任务回报与新任务学习曲线这两组测量。激活率等内部指标可帮助定位原因，其意义要结合这些行为结果判断。

回放通过数据分布保留经验，参数正则限制部分权重改变，输出蒸馏约束已有功能。它们都可能在保护旧知识时限制新学习。ReDo 与 continual backpropagation 等方法研究失效或低效用特征的更新与替换，必须同时检查新学习与旧能力。

一个最小诊断可以固定新任务、数据顺序、网络规模和更新预算，比较 aged、fresh 与随机替换对照。fresh 的初始化、优化器状态和超参数选择也要对齐。性能改善可能来自步长、尺度或重置优化器，不能仅凭替换神经元就归因于一种可塑性机制。

$$
F_A=J_A(\theta_{\rm before})-J_A(\theta_{\rm after}),\qquad P_C(K)=L_C(\theta_{\rm aged}^{(K)})-L_C(\theta_{\rm fresh}^{(K)})
$$

$F_A$ 表示旧任务收益下降；$P_C$ 在匹配 K 次新任务学习后比较损失。这里用较小损失为好，正值表示 aged 学得较差；二者不能互相替代。

#### 动手与核对

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

### 6 · 子任务、技能、模型与规划的衔接

长期经验怎样形成可重复使用的行为，而不只是一组新参数？

先有总体任务奖励，再问哪些中间目标值得学习。目标条件化策略把目标作为输入，用同一个函数表示一族控制问题；HER 改写已发生轨迹的目标与相应奖励，改善稀疏目标学习。目标构造决定学习什么，而不是仅给策略输入一个额外向量。

Option 把可复用行为定义为启动集合、内部策略和终止规则。高层选择 option 后，低层会执行随机时长 $\tau$。因此高层 target 必须累计这段真实奖励，并用 $\gamma^\tau$ 折扣终点价值。技能发现决定哪些 option 值得加入集合，内部策略优化则决定已有 option 如何执行。

以“走到门口”为例，启动集合可以先规定为门附近可到达的状态，内部策略从尝试到门口的经验中学习，到达门口时终止。这样只学习了已有技能怎样执行。若连门口是否值得作为目标、何时可以启动、何时应该停止也要在线决定，就分别增加了子任务选择、启动条件学习和终止学习。给一个行为起名字，并没有解决这些问题。

可执行技能还不足以规划。模型需要预测它的累计奖励和带时长折扣的终点分布；规划再比较不同技能的长期后果。STOMP 将子任务、option、模型和规划串成这种因果关系。OaK 和 Alberta Plan 提出更广的持续构建框架；研究架构时仍要逐项检验接口和收益，不能把纲领当成完整算法。

$$
Q(s,o)\leftarrow Q(s,o)+\alpha\left[\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}+\gamma^\tau\max_{o'\in\mathcal O(S_{t+\tau})}Q(S_{t+\tau},o')-Q(s,o)\right]
$$

这是 option 结束时的 SMDP Q-learning target。O(s′) 是终点可启动的合法 option 集合；环境真实终止时余项为零。随机时长与终点可能相关。

#### 动手与核对

[下载 knowledge_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/knowledge_algorithms_lab.py)

在保存该文件的目录运行：

```sh
python3 knowledge_algorithms_lab.py all
python3 knowledge_algorithms_lab.py test
```

预期检查：奖励 1、2，γ=.9、两步后终点价值 10，SMDP 与 option 模型 backup 均为 10.9；检查 HER、终止规则和合法 option 集合测试。

实验范围：每个机制使用可手算环境；脚本不是 STOMP 或 OaK 全系统复现。

自测：一个技能有时一步结束、有时三步结束。只保存平均时长，再用 γ 的平均时长次方乘终点价值，是否一般正确？

解答：不正确。需要期望中的 $\gamma^\tau$ 与终点价值乘积；指数非线性，且时长可能与终点相关。正确模型保存相应的折扣终点权重。

完整推导与问题衔接：

- [需要补先修时：Dyna 中的模型学习与规划](https://yingwen.io/zh/continual-rl/algorithms/dyna/)
- [目标条件化与子任务构造](https://yingwen.io/zh/continual-rl/construction/goals/)
- [Options 与技能发现](https://yingwen.io/zh/continual-rl/construction/options/)
- [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)
- [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)

原文、课程与实现：

- [Sutton 等 · Reward-Respecting Subtasks / STOMP](https://arxiv.org/abs/2202.03466)：子任务定义如何连接 option 学习、模型和规划。
- [Sutton、Precup、Singh · Options](https://doi.org/10.1016/S0004-3702(99)00052-1)：启动、内部策略、终止与 semi-Markov backup。
- [Oak Lab · The OaK Architecture](https://oaklab.ai/posts/the-oak-architecture)：架构研究纲领与讲座入口；与单个已实现算法区分。

### 7 · 持续探索、学习进展与恢复

长期智能体应该把有限经验花在哪些问题上？

探索与经验获取贯穿前面的学习过程。这里单独讨论真实交互怎样提供新信息；研究状态、预测或流式学习时，也可以直接进入本节。子任务和技能进一步改变可选择的探索行为，规划则利用已经学到的后果。

学习器只会从实际获得的数据中学习。环境变化后，旧行为可能不再访问变好的区域；更新规则即使能快速适应，也没有收到新信息。持续探索因此不是训练早期的一次附加技巧，而是长期经验分配的问题。

新奇、预测误差、不确定性和学习进展并不相同。RND 用固定随机目标与可训练预测器的误差衡量不熟悉程度；但噪声也会造成误差。学习进展比较一段训练前后的改变量，试图区分难而可学的区域与不可预测噪声。探索奖励还会改变数据分布及 critic 的目标。

single-life 条件增加恢复和安全约束。探索到一个无法返回的状态，会影响后续全部学习机会。能够由外部生成任务、重置机器人或恢复仿真的课程算法，不能直接视为无这些权限的问题解法。应把可行动作、恢复策略和重置成本写进协议。

$$
b_t(s)=\|f_{\rm predictor}(s)-f_{\rm fixed}(s)\|^2,\qquad \ell_t(s)=E_{\rm before}(s)-E_{\rm after}(s)
$$

第一式是 RND 型误差奖励，第二式是简化学习进展指标。它们依赖模型和测量窗口，不等于真实信息增益或安全保证。

#### 动手与核对

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

- [相关问题：子任务选择怎样影响经验获取](https://yingwen.io/zh/continual-rl/construction/goals/)
- [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)

原文、课程与实现：

- [Burda 等 · RND](https://arxiv.org/abs/1810.12894)：冻结随机目标和可训练预测器的原始探索方法。
- [RND 作者代码](https://github.com/openai/random-network-distillation)：完整 Atari 实现及观测、内在奖励归一化。
- [Eysenbach 等 · Leave no Trace](https://arxiv.org/abs/1711.06782)：前向学习与恢复行为的共同设计。

### 8 · 从学习材料到可检验的研究实验

什么样的实验能够支持“更适合持续学习”的结论？

先选择一个具体失败：奖励反转后跟踪太慢、延迟线索无法学到、长期训练后新任务学习变慢，或旧模型阻碍新规划。固定基础 agent 和预算，提出只针对这个失败的机制假设。先用有解析答案或明确反例的小环境检查实现，再进入更复杂 benchmark。

把开发任务和种子用于调参与选择，保留独立测试用于最终估计。时间步不是独立重复实验；一条生命中的回报具有相关性。若使用共同随机数配对，需说明两算法共享的是哪些潜在随机事件，而不仅是把整数 seed 设为相同。

同时报告整个寿命收益、变化附近的短期代价、最终或分阶段冻结评价，以及资源预算。冻结诊断应运行副本，不能把测试产生的更新写回训练 agent。恢复时间在观察结束时仍未恢复，就属于删失数据，不能只平均成功恢复的运行。

复杂基准用于检验机制能否推广，而不是替代机制解释。例如 AgarCL 将探索、记忆和长期学习放到持续环境中；不同动作集、重生规则和算法预算仍需对齐。小实验通过后，再使用原环境与对应基线，逐项保留实验条件。

$$
\Delta=\mathbb E[U_A-U_B],\qquad \hat\Delta=\frac1n\sum_{i=1}^{n}(U_{A,i}-U_{B,i})
$$

U 是事先指定的每条生命统计量，例如固定寿命平均回报。配对只在第 i 对具有明确共同随机结构时成立；bootstrap 应重采样独立生命或实验块，而非任意时间步。

#### 动手与核对

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

- [持续学习的智能体架构](https://yingwen.io/zh/continual-rl/construction/architectures/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

原文、课程与实现：

- [Patterson 等 · Empirical Design in RL](https://jmlr.org/papers/v25/23-0183.html)：目标量、算法比较、超参数选择与不确定性的系统说明。
- [AgarCL · The Cell Must Go On](https://arxiv.org/abs/2505.18347)：持续环境与分解问题的基准设定。
- [AgarCL 作者环境](https://github.com/machado-research/AgarCL)：环境、配置和安装入口。
- [AgarCL 作者基线](https://github.com/machado-research/AgarCL-benchmark)：完整游戏的不同基础 agent；需要匹配观测、动作和训练预算。

下一步：本册目录组织完整正文，概念导读与自测帮助检查章节之间的联系；可按当前问题选择阅读。进入研究时，先固定基础 agent 和预算，只改变一个长期学习条件，提出能被对照实验否定的机制假设。
