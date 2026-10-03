# 持续强化学习：统一学习地图

目录与算法谱系使用同一套章节结构。优化目标、状态、预测知识和控制分别讨论。跨章节关系表示概念依赖或模块组合。

## 只有基础 RL：先完成一个小闭环

不必先补齐全部深度 RL。按下面五步，先理解持续任务、追踪变化、完成实验，再认识预测与长行为。

1. [分清持续任务与学习协议](/zh/continual-rl/start/01-lifetime/)：能说明什么变化、什么重置。
2. [常数步长怎样跟踪变化](/zh/continual-rl/start/02-tracking/)：能手算一次更新并解释新旧经验权重。
3. [跑完第一份实验](/zh/continual-rl/first-project/)：保存多种子结果，解释恢复速度与波动。
4. [把 TD 推广成 GVF](/zh/continual-rl/construction/predictive-knowledge/)：能写出一个非奖励预测问题。
5. [把单步动作推广成 option](/zh/continual-rl/construction/options/)：能区分行为、终止与行为后果模型。

再读平均奖励、模型与规划。根据实验需要补充状态、子任务、保留与可塑性。

## 已有 DQN / PPO 背景：先选问题，再补接口

先定义目标和交互协议。再选择一种学习对象或机制。

1. [确定持续交互的目标](/zh/continual-rl/algorithms/average-reward/)：说明为何使用折扣或平均奖励。
2. [明确评测与预算](/zh/continual-rl/experiments/)：把 reset、replay、task ID 写进协议。
3. [选一个学习对象或失败模式](/zh/continual-rl/algorithms/#textbook-prediction)：只改变一个接口或一种更新机制。
4. [把公式对应到原始实现](/zh/continual-rl/construction/implementations/)：追踪 target、梯度、mask 与更新顺序。
5. [设计研究对照](/zh/continual-rl/research/)：先排除更多数据、计算和调参的解释。

架构章讨论模块组合。研究可塑性或流式更新时，可先阅读相应章节。

## I · 强化学习问题与目标

什么结果算更好？

完成标志：目标规定要改善什么。状态规定决策时可以使用什么信息。两者需要分别定义。

### [B0 · 交互、奖励与优化目标](/zh/continual-rl/foundations/objectives/)

什么结果算更好？

- 先修：基础 RL 符号
- 学会什么：区分奖励、回报、价值、策略和在线表现。
- 动手：[目标选择与奖励塑形实验](/zh/continual-rl/foundations/objectives/#lesson-code)

### [B1 · 生命期、变化与交互协议](/zh/continual-rl/start/01-lifetime/)

continuing、continual、streaming、single-life 差在哪？

- 先修：[A2 控制问题与广义策略迭代](/zh/continual-rl/algorithms/control/)
- 学会什么：明确环境与学习器各在何时重置。
- 动手：[从变化 bandit 完成第一份实验](/zh/continual-rl/first-project/)

### [B2 · 平均奖励与 differential learning](/zh/continual-rl/algorithms/average-reward/)

没有自然终点时，如何学每单位时间的收益？

- 先修：[A1 价值预测：MC → TD → traces](/zh/continual-rl/algorithms/value/)；[A2 控制问题与广义策略迭代](/zh/continual-rl/algorithms/control/)
- 学会什么：区分折扣价值、奖励率、差分价值与时长。
- 动手：[Differential TD/Q 原代码与公式检查](/zh/continual-rl/construction/implementations/#impl-differential)

## II · 状态构造与表征

智能体需要从历史中保留什么信息？

完成标志：给定状态后，可以定义关于未来的预测。状态是否充分，取决于它需要支持哪些预测与决策。

### [C2 · 状态构造：把历史变成决策依据](/zh/continual-rl/construction/state/)

哪些不同历史必须区分？

- 先修：[B0 交互、奖励与优化目标](/zh/continual-rl/foundations/objectives/)
- 学会什么：区分 hidden state、预测状态与学习参数。
- 动手：[GVFN 的问题、网络与递归更新](/zh/continual-rl/construction/implementations/#impl-gvfn)

## III · 预测与预测知识

在指定行为下，未来会发生什么？

完成标志：控制需要使用预测来改变策略。策略改变后，数据分布和需要预测的未来也会改变。

### [A1 · 价值预测：MC → TD → traces](/zh/continual-rl/algorithms/value/)

没有完整回报时怎样逐步学预测？

- 先修：基础 RL 符号
- 学会什么：手算 TD target，解释偏差、方差与 λ。
- 动手：[固定策略预测实验](/zh/continual-rl/algorithms/#algorithm-code)

### [C1 · GVF：从一个回报到一组预测](/zh/continual-rl/construction/predictive-knowledge/)

问什么、在哪个策略下问、问多远？

- 先修：[A1 价值预测：MC → TD → traces](/zh/continual-rl/algorithms/value/)
- 学会什么：写出 c、γ、π，并追踪 off-policy 更新。
- 动手：[两步 GVF → RLPark / Horde](/zh/continual-rl/construction/implementations/#impl-rlpark)

## IV · 控制与策略改善

智能体怎样选择动作并改善行为？

完成标志：这些控制方法都需要分配信用并选择更新尺度。持续交互要求学习过程也能适应。

### [A2 · 控制问题与广义策略迭代](/zh/continual-rl/algorithms/control/)

怎样比较策略，并用预测改善行为？

- 先修：[A1 价值预测：MC → TD → traces](/zh/continual-rl/algorithms/value/)
- 学会什么：从 Bellman 最优方程推到策略迭代、价值迭代与在线控制。
- 动手：[动态规划与在线控制对照](/zh/continual-rl/algorithms/control/#lesson-code)

### [A3 · 深度价值：DQN 与扩展](/zh/continual-rl/algorithms/deep-value/)

函数逼近、bootstrap 与复用数据怎样相互影响？

- 先修：[A2 控制问题与广义策略迭代](/zh/continual-rl/algorithms/control/)
- 学会什么：定位 replay、target network 和 Double DQN。
- 动手：[从单文件 DQN 追一次更新](/zh/continual-rl/algorithms/deep-value/#algorithm-sources)

### [A4 · 策略梯度 → actor–critic → PPO](/zh/continual-rl/algorithms/policy/)

怎样直接改动作概率，又控制更新幅度？

- 先修：[A1 价值预测：MC → TD → traces](/zh/continual-rl/algorithms/value/)
- 学会什么：对应 log-prob、advantage 和 PPO surrogate。
- 动手：[REINFORCE 与 actor–critic 小实验](/zh/continual-rl/algorithms/#algorithm-code)

### [A5 · 连续控制：DDPG / TD3 / SAC](/zh/continual-rl/algorithms/soft-control/)

连续动作与熵正则如何改变 actor 和 critic？

- 先修：[A2 控制问题与广义策略迭代](/zh/continual-rl/algorithms/control/)；[A4 策略梯度 → actor–critic → PPO](/zh/continual-rl/algorithms/policy/)
- 学会什么：分开 off-policy 数据、双 Q 与熵目标。
- 动手：[阅读 SAC 数据与 target 路径](/zh/continual-rl/algorithms/soft-control/#algorithm-sources)

## V · 时间信用分配

当前反馈应当更新过去哪些预测与决策？

完成标志：信用分配规定反馈如何作用于过去。流式协议进一步限制数据保存与每步计算。

### [D0 · 时间信用分配与资格迹](/zh/continual-rl/algorithms/credit-assignment/)

当前反馈应怎样更新过去的预测与决策？

- 先修：[A1 价值预测：MC → TD → traces](/zh/continual-rl/algorithms/value/)
- 学会什么：区分前向目标、后向资格迹、递归状态导数和元敏感度。
- 动手：[前后向等价与在线资格迹实验](/zh/continual-rl/algorithms/credit-assignment/#lesson-code)

## VI · 流式学习与更新稳定性

在有限内存和逐步计算预算下，怎样执行学习？

完成标志：固定的更新规则可以依据当前数据调整尺度。元学习进一步用经验改善学习规则本身。

### [D3 · 流式学习：Stream-X 与更新尺度控制](/zh/continual-rl/algorithms/streaming/)

在有限内存和逐步预算下，怎样稳定地更新？

- 先修：[D0 时间信用分配与资格迹](/zh/continual-rl/algorithms/credit-assignment/)；[B1 生命期、变化与交互协议](/zh/continual-rl/start/01-lifetime/)
- 学会什么：区分交互协议、资格迹、尺度控制与元梯度学习。
- 动手：[逐行追踪流式更新](/zh/continual-rl/algorithms/streaming/#algorithm-sources)

## VII · 元学习与学习规则的适应

怎样根据学习效果调整学习过程？

完成标志：有了学习机制，还需要规定可复用行为的目标、启动条件与终止规则。

### [D5 · Meta-learning：步长、初始化与学习规则](/zh/continual-rl/algorithms/meta/)

一次学习怎样影响后续学习，怎样对这种影响求导？

- 先修：[A4 策略梯度 → actor–critic → PPO](/zh/continual-rl/algorithms/policy/)；[B1 生命期、变化与交互协议](/zh/continual-rl/start/01-lifetime/)
- 学会什么：区分在线元梯度、跨任务适应、上下文推断与学得的更新规则。
- 动手：[两步学习的元梯度与数值校验](/zh/continual-rl/algorithms/meta/#lesson-code)

## VIII · 子任务与时间抽象

哪些可复用行为值得学习？

完成标志：能够执行一个行为，不等于知道它的后果。规划还需要这个行为的模型。

### [C3 · 目标构造：决定值得学什么](/zh/continual-rl/construction/goals/)

目标、内在奖励和子任务怎样定义用途？

- 先修：[A2 控制问题与广义策略迭代](/zh/continual-rl/algorithms/control/)；[C1 GVF：从一个回报到一组预测](/zh/continual-rl/construction/predictive-knowledge/)
- 学会什么：区分 goal conditioning、目标生成与价值保留。
- 动手：[最短路与 reward-respecting 子任务](/zh/continual-rl/construction/#construction-code)

### [C4 · Options：行为的时间抽象](/zh/continual-rl/construction/options/)

怎样发现、执行、终止和复用长行为？

- 先修：[A2 控制问题与广义策略迭代](/zh/continual-rl/algorithms/control/)
- 学会什么：定义 I、π、β；区分技能发现与 SF/GPI。
- 动手：[Option-Critic → eigenoptions](/zh/continual-rl/construction/implementations/#impl-option-critic)

## IX · 模型学习与规划

怎样用学得的后果模型改善决策？

完成标志：模型、状态、技能与策略都会变化。接下来需要研究这些变化怎样影响保留、适应与探索。

### [A6 · Dyna：学习模型，再做规划](/zh/continual-rl/algorithms/dyna/)

真实经验和模型生成的 backup 如何配合？

- 先修：[A2 控制问题与广义策略迭代](/zh/continual-rl/algorithms/control/)
- 学会什么：分开真实步数、模型误差和规划计算。
- 动手：[Q-learning 与 Dyna-Q 对照](/zh/continual-rl/algorithms/#algorithm-code)

### [C5 · 转移与后果模型](/zh/continual-rl/construction/models/)

一个长行为会带来什么奖励、时长和终点？

- 先修：[C4 Options：行为的时间抽象](/zh/continual-rl/construction/options/)；[A6 Dyna：学习模型，再做规划](/zh/continual-rl/algorithms/dyna/)
- 学会什么：区分 option policy、option value 和 option model。
- 动手：[随机时长与期望特征反例](/zh/continual-rl/construction/models/#construction-example)

### [C6 · 抽象模型与规划](/zh/continual-rl/construction/planning/)

怎样用行为模型改善当前决策？

- 先修：[C5 转移与后果模型](/zh/continual-rl/construction/models/)；[B2 平均奖励与 differential learning](/zh/continual-rl/algorithms/average-reward/)
- 学会什么：写出折扣与平均奖励下不同的 option backup。
- 动手：[planning 小实验与时长公式检查](/zh/continual-rl/construction/implementations/#audit-duration)

## X · 长期适应、保留与探索

学习能否长期保持有效？

完成标志：在完整智能体中，这些机制共享经验与计算资源。它们的相互作用需要单独研究。

### [D1 · 知识保留：replay / EWC / 蒸馏](/zh/continual-rl/algorithms/retention/)

新知识会怎样干扰过去能力？

- 先修：[B1 生命期、变化与交互协议](/zh/continual-rl/start/01-lifetime/)；[A3 深度价值：DQN 与扩展](/zh/continual-rl/algorithms/deep-value/)
- 学会什么：分开知识保留和重新适应的成本。
- 动手：[保留与适应的确定性反例](/zh/continual-rl/algorithms/#algorithm-code)

### [D2 · 可塑性：ReDo / Continual Backprop](/zh/continual-rl/algorithms/plasticity/)

旧任务没忘，为何新任务仍越来越学不动？

- 先修：[B1 生命期、变化与交互协议](/zh/continual-rl/start/01-lifetime/)；[A3 深度价值：DQN 与扩展](/zh/continual-rl/algorithms/deep-value/)
- 学会什么：设计 fresh-network 与替换策略对照。
- 动手：[定位 CBP 的替换与优化器状态](/zh/continual-rl/algorithms/plasticity/#algorithm-sources)

### [D4 · 探索：新奇与学习进展](/zh/continual-rl/algorithms/exploration/)

什么经验值得主动获取？

- 先修：[A2 控制问题与广义策略迭代](/zh/continual-rl/algorithms/control/)；[B1 生命期、变化与交互协议](/zh/continual-rl/start/01-lifetime/)
- 学会什么：区分预测误差、新奇、学习进步与最终用途。
- 动手：[给不可约噪声加一个对照](/zh/continual-rl/algorithms/exploration/#algorithm-practice)

## XI · 持续学习的智能体架构

各学习过程怎样共同改善长期行为？

完成标志：架构提出机制假设。实验检验实现、机制与长期收益。

### [E1 · 系统架构：Dyna → STOMP / OaK](/zh/continual-rl/construction/architectures/)

哪些部件学习，哪些仍由人提供？

- 先修：[C6 抽象模型与规划](/zh/continual-rl/construction/planning/)；[D3 流式学习：Stream-X 与更新尺度控制](/zh/continual-rl/algorithms/streaming/)
- 学会什么：画出数据、目标、参数和行为之间的接口。
- 动手：[比较模块已实现的连接与开放接口](/zh/continual-rl/construction/#architecture-comparison)

## XII · 强化学习实验方法

哪些数据能够支持所提出的结论？

完成标志：实验设计贯穿全部章节。每学完一个更新规则，就可以运行小问题、提出反例并检验假设。

### [E3 · 实验设计、统计与算法测试](/zh/continual-rl/experiments/)

哪些证据能够检验机制与性能？

- 先修：[B0 交互、奖励与优化目标](/zh/continual-rl/foundations/objectives/)
- 学会什么：分离开发与测试，按独立运行估计不确定性。
- 动手：[完整开发、选择与独立测试实验](/zh/continual-rl/experiments/#lesson-code)

### [B3 · 持续学习的评测与基准](/zh/continual-rl/start/07-evaluate/)

高平均回报掩盖了哪些失败？

- 先修：[B1 生命期、变化与交互协议](/zh/continual-rl/start/01-lifetime/)
- 学会什么：设计恢复、保留、新学习和资源指标。
- 动手：[写一份公平实验协议](/zh/continual-rl/download/protocol.md)

### [E2 · 从机制假设走到研究实验](/zh/continual-rl/research/)

在同等资源下，哪个改变真正带来长期收益？

- 先修：[B3 持续学习的评测与基准](/zh/continual-rl/start/07-evaluate/)
- 学会什么：写出最小基线、干预、消融与结论边界。
- 动手：[选择一个可检验的起步项目](/zh/continual-rl/construction/#construction-projects)

## 按用途查材料

- [算法历史与家族关系](/zh/continual-rl/download/algorithm-map.md)
- [作者代码与公式对照](/zh/continual-rl/construction/implementations/)
- [资源库：论文、课程、讲座、代码与基准](/zh/continual-rl/library/)
