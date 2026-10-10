# 强化学习基础：按工具查阅

表格方法与函数近似属于第一册，深度方法属于第二册。完整阅读顺序见[三册教材](https://yingwen.io/zh/continual-rl/start/)。这里按工具组织索引，不把经典 RL、深度 RL 与 CRL 当作互斥的问题类。表格与函数逼近描述表示和更新方式；深度方法使用神经网络；持续学习关注完整学习器在长期交互中的表现。它们可以同时用于同一智能体。

运行时，智能体与世界交互；设计者在外层选择奖励、初始化、数据权限和预算。先用[领域总览](https://yingwen.io/zh/continual-rl/overview/)定位问题，再按需补基础。目标问题见[奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/)，算法比较见[持续控制](https://yingwen.io/zh/continual-rl/algorithms/control/)。

基础共 29 章。表格方法与 Part II 提供预测、控制、函数逼近、信用分配和策略梯度工具。深度分册先给核心算法，再并列展开研究分支；无需学完全部分支才进入 CRL 研究。

## 表格强化学习

没有共享参数时，怎样从经验评价策略、改善行为并进行规划？

对应：Sutton & Barto · Part I，第 2–8 章

### 建议的基础阅读顺序

按先修关系组织。已有基础的读者可用各章练习定位缺口，不必逐章重读。

#### [第 1 章 · 多臂老虎机：估计、探索与直接策略学习](https://yingwen.io/zh/continual-rl/foundations/tabular/bandits/)

没有状态转移时，仍需一边估计动作收益，一边决定下一次尝试什么。这个最小问题把估计误差、探索代价和策略更新分开。

- 推导样本平均和常数步长，解释历史奖励的权重。
- 区分 ε-greedy、乐观初值、UCB 与 gradient bandit 改变的量。
- 运行逐步反馈与随机奖励采样，区分机制检查与有限样本性能。

#### [第 2 章 · MDP、回报与价值：序列决策的数学对象](https://yingwen.io/zh/continual-rl/foundations/tabular/mdps/)

动作会改变后续状态时，需要评价整个未来。本章从随机交互过程推导价值与 Bellman 方程，明确后续算法共同使用的数学对象。

- 区分状态、观测、策略和环境模型。
- 从回报推导状态价值、动作价值与 Bellman 方程。
- 解释 Markov、平稳、终止和截断条件。

#### [第 3 章 · 动态规划：评价、改善与最优递推](https://yingwen.io/zh/continual-rl/foundations/tabular/dynamic-programming/)

已知环境模型时，怎样通过局部计算得到长期价值和策略？本章将 Bellman 方程转化为迭代，并证明评价与改善之间的联系。

- 推导迭代评价、收缩与残差误差界。
- 证明策略改善，区分策略迭代、价值迭代和 GPI。
- 对照解析求解，理解计算预算与模型误差。

#### [第 4 章 · Monte Carlo：完整回报、探索控制与离策略评价](https://yingwen.io/zh/continual-rl/foundations/tabular/monte-carlo/)

不知道模型时，可以将完整回报作为样本。估计还会改变下一次行动：需要看清回报来自哪一版策略、探索怎样影响收益，以及长回合怎样消耗有效覆盖。

- 实现 first-visit、every-visit、ε-soft 与加权离策略 MC 控制。
- 推导轨迹与逐决策重要性采样，区分 ordinary IS 与 weighted IS。
- 识别终止、覆盖、方差和策略变化的边界。

#### [第 5 章 · TD 预测与控制：SARSA、Expected SARSA、Q-learning 和 Double Q](https://yingwen.io/zh/continual-rl/foundations/tabular/temporal-difference/)

如何在完整回报尚不可用时学习？TD 用下一预测补足未来；控制算法再根据不同的下一动作处理方式，形成不同的学习目标。

- 从 Bellman 样本推导 TD(0)。
- 区分四种控制 target、行为策略和更新顺序。
- 理解 maximization bias、Double 选择评价分离及表格收敛条件。

#### [第 6 章 · 多步学习：n-step、Tree Backup 与 Q(σ)](https://yingwen.io/zh/continual-rl/foundations/tabular/multistep/)

学习目标可以在一步 bootstrap 与完整回报之间选择，也可以在动作采样与动作期望之间选择。这是两条不同的设计维度。

- 推导 n-step 回报及延迟更新时序。
- 理解 Tree Backup 的已采样分支和未采样分支。
- 从两端推导 on-policy Q(σ)，检查终止与退化条件。

#### [第 7 章 · 学习与规划：Dyna、优先扫描和执行时搜索](https://yingwen.io/zh/continual-rl/foundations/tabular/planning/)

真实经验既能直接改进价值，也能训练后果模型。规划使用这个模型继续计算，关键是模型语义、backup 的成本以及计算应分配到哪里。

- 实现 Dyna 的真实交互、模型更新与模拟更新循环。
- 推导优先扫描的前驱传播，区别 sampling 与 expectation backup。
- 理解 rollout、MCTS 与持续环境中模型错误的边界。

## 函数逼近与经典进阶方法

状态不能逐项保存时，哪些更新仍成立，哪些保证需要重新检查？

对应：Sutton & Barto · Part II，第 9–13 章

### 建议的基础阅读顺序

按先修关系组织。已有基础的读者可用各章练习定位缺口，不必逐章重读。

#### [第 1 章 · 函数逼近预测：从回归到 TD 固定点](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/)

共享少量参数以后，MC、TD 和最小二乘方法究竟在求解什么？

- 从平方价值误差推导梯度 MC，说明采样分布的作用。
- 推导线性 TD 的平均更新、投影 Bellman 方程与 LSTD。
- 用同一个二状态例子计算不同解，并检查半梯度的含义。

#### [第 2 章 · 特征、泛化与半梯度控制](https://yingwen.io/zh/continual-rl/foundations/approximation/features-control/)

特征怎样改变学习行为，Sarsa 又怎样在共享参数下改善策略？

- 在同一走廊比较状态聚合、局部 tile coding 与分离特征，并理解全局 Fourier 特征的泛化。
- 推导动作分块表示和半梯度 Sarsa，沿一次共享更新追踪动作与下一条真实经验。
- 手算特征缩放的步长效应，并辨别预测保证与控制保证。

#### [第 3 章 · 持续控制与平均奖励](https://yingwen.io/zh/continual-rl/foundations/approximation/average-control/)

智能体没有自然回合终点时，怎样定义和学习长期控制目标？

- 区分折扣价值、平均奖励率与差分价值。
- 从 Poisson 方程得到差分 TD 和 Sarsa 的两组同步更新。
- 在同一服务站中手算真实时钟下的奖励率、启动 bias 和一次时长归一化更新。
- 说明 unichain、communicating 与函数逼近保证的边界。

#### [第 4 章 · 离策略函数逼近：覆盖、发散与稳定更新](https://yingwen.io/zh/continual-rl/foundations/approximation/off-policy/)

行为数据足够覆盖目标策略，为什么 TD 仍可能发散，又能怎样修复？

- 用同一二状态例子区分动作校正、当前状态权重和目标占据。
- 从投影几何推导 MSPBE，手算 GTD2、TDC 的样本方向与辅助量。
- 算出 emphatic 强调质量，并区分稳定更新、投影固定点与真实价值。

#### [第 5 章 · 多步回报、资格迹与 True-online TD](https://yingwen.io/zh/continual-rl/foundations/approximation/traces/)

当前到来的奖励怎样更新过去的预测，同时保留正确的在线更新语义？

- 把 n-step 与 lambda-return 的前向目标推导成后向信用传播。
- 解释普通累积迹与 true-online 的差别，推导 Dutch trace。
- 用重复状态和独立前向实现检查每个轨迹前缀。

#### [第 6 章 · 策略梯度、基线与 Actor–Critic](https://yingwen.io/zh/continual-rl/foundations/approximation/policy-gradient/)

直接学习策略时，哪一个目标的梯度能由经验估计，近似从哪里进入？

- 从轨迹概率推导带完整折扣权重的 REINFORCE。
- 证明动作无关基线不改变梯度，并定位 critic 引入的偏差。
- 区分普通梯度、自然梯度以及平均奖励目标的采样权重。

## 现代深度强化学习

怎样组织神经智能体的核心更新，再按信息、目标与数据条件选择研究分支？

对应：Spinning Up · 核心方法与原始研究扩展

### 核心算法与训练循环

先掌握价值学习、策略梯度、批内策略更新、连续控制和训练检查。它们是工具基础，不要求在所有研究问题中同时使用。

#### [第 1 章 · 深度价值学习：DQN、Double DQN 与目标的时间顺序](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/)

把表格 Q-learning 换成网络后，损失、数据与目标为什么都需要重新组织？

- 在同一个共享特征算例中算出预测、动作与旧经验标签的变化。
- 写出 DQN 与 Double DQN 的不同目标，区分冻结副本与停止梯度。
- 区分环境终止、采样截断和目标网络更新。
- 把标准库手算对应到完整 PyTorch 交互与训练循环。

#### [第 2 章 · 策略梯度：从轨迹概率到 GAE 与 actor–critic](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/)

延迟奖励怎样改变动作概率？有限 rollout、critic 与停止梯度分别改变哪一项估计？

- 从完整轨迹 score 推到 reward-to-go 与 baseline。
- 分清回合目标、有限 rollout 权重和实现 surrogate。
- 解释 critic 误差怎样进入有限 GAE，并设置两类 mask。
- 追踪 actor/critic 的固定量与求导路径，再进入 PPO。

#### [第 3 章 · 策略更新的尺度：TRPO 与 PPO](https://yingwen.io/zh/continual-rl/foundations/deep/trust-region/)

旧策略的数据能支持多远的策略更新？怎样从局部代理走到采样、更新与独立评价？

- 由性能差分恒等式定位旧状态分布替代的近似。
- 求解局部 KL 约束，理解 Fisher、共轭梯度与回溯。
- 按优势符号解释 PPO，并固定旧策略、优势与 critic target。
- 区分批次、优化步和交互步，评价冻结策略与持续学习器。

#### [第 4 章 · 连续动作的价值优化：DDPG 与 TD3](https://yingwen.io/zh/continual-rl/foundations/deep/deterministic-control/)

不能枚举连续动作时，如何用 critic 的梯度改进 actor？

- 区分行为噪声、目标动作噪声与目标网络。
- 推导确定性 actor 的链式梯度。
- 写出 TD3 双 critic、延迟更新与平滑目标的完整次序。

#### [第 5 章 · 最大熵连续控制：SAC 的价值、密度与温度](https://yingwen.io/zh/continual-rl/foundations/deep/entropy-control/)

随机 actor 不只是加噪声：熵如何进入 Bellman 方程与自动微分？

- 推导 soft Bellman 与 actor 的 KL 投影。
- 计算 tanh 与动作缩放后的概率密度。
- 区分 actor、critic、温度三类梯度与停止梯度位置。

#### [第 6 章 · 深度 RL 的机制接口：模型、记忆、离线数据与实验](https://yingwen.io/zh/continual-rl/foundations/deep/practice/)

改变数据来源或 agent state 后，哪些推导和实现条件必须重新检查？

- 由 Bellman 算子解释模型误差的放大。
- 区分 recurrent hidden state 与跨时间参数梯度。
- 识别离线外推问题，并设计可解释的 CRL 对照。

#### [第 7 章 · 大规模训练：算法与系统怎样共同设计](https://yingwen.io/zh/continual-rl/foundations/deep/systems/)

环境、推理和学习并行以后，怎样把更多计算变成更快的策略改善？

- 沿一段经验追踪行为策略、传输、排队和梯度更新。
- 区分 A3C 的过期梯度、IMPALA 的离策略目标与 PPO 的重复优化。
- 解释 OpenAI Five、SEED RL 和 GEAR 怎样分配计算与移动数据。
- 用数据年龄、采样概率、资源预算和达到目标性能的时间共同评价系统。

### 并列研究分支

根据研究问题选择：这些方向改变信息、数据、模型或目标条件，可以交叉组合，不是必须依次完成的后续关卡。

#### [不完全可观测：信念状态、信息行动与递归记忆](https://yingwen.io/zh/continual-rl/foundations/deep/partial-observability/)

当前观察不能决定未来时，智能体应记住什么，信息又如何影响行动？

- 从历史条件分布推导 Bayes filter。
- 把信息获取的价值纳入 Bellman 决策。
- 区分精确信念、学习的 recurrent state 和训练时的隐状态权限。
- 逐版本计算 recurrent replay 的状态、TD 标签与截断梯度。

#### [探索与不确定性：后验、乐观估计和时间一致行动](https://yingwen.io/zh/continual-rl/foundations/deep/exploration/)

为什么每一步都随机，并不等于有效获取长期有用的信息？

- 区分环境随机性与知识不足。
- 推导 Bernoulli 后验与 UCB 的置信思路。
- 理解 posterior sampling、Bootstrapped DQN 与内在奖励各自改变什么。

#### [分布强化学习：Bellman 分布、分位数与风险目标](https://yingwen.io/zh/continual-rl/foundations/deep/distributional/)

学习完整回报分布，与学习均值、评估风险和估计知识不确定性分别有什么关系？

- 写出分布 Bellman 递推及其固定策略条件。
- 推导 categorical 投影和 quantile loss。
- 区分回报分布、参数后验和风险偏好。

#### [离线强化学习：数据支持、策略评估与保守改进](https://yingwen.io/zh/continual-rl/foundations/deep/offline/)

不能补采数据时，怎样判断策略好坏，怎样避免利用没有证据的高价值动作？

- 把数据支持假设写入 IS 与 doubly robust 评估。
- 推导 CQL 正则和 IQL expectile 的不同作用。
- 分开学习策略、选择超参数与独立评估。

#### [模型学习与规划：MPC、短模型 rollout 和潜在想象](https://yingwen.io/zh/continual-rl/foundations/deep/model-based/)

模型在哪里进入决策，预测误差又怎样变成控制误差？

- 由监督建模到有限时域 MPC。
- 推导多步误差传播与短 rollout 的动机。
- 区分 MPC、MBPO 和 Dreamer 的模型使用位置。

#### [约束强化学习：占据测度、拉格朗日与可行策略](https://yingwen.io/zh/continual-rl/foundations/deep/constraints/)

“回报高且代价不超过预算”与“每一步都安全”之间差了哪些条件？

- 从折扣访问频率推导 CMDP 的占据流约束。
- 解释随机策略为何可能必要及拉格朗日乘子的方向。
- 区分平均代价、概率约束、硬约束和训练期安全。

#### [多智能体合作：结构化探索与信用分配](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent/)

团队共享一个奖励时，怎样从联合经验中学习可执行的协作策略，并正确处理同伴更新？

- 区分合作中的结构化探索与信用分配，以及竞争和开放合作中的评价与目标构建。
- 从反事实基线与可分散贪心条件推导 COMA、VDN、QMIX 和 QPLEX 的机制。
- 解释 MAVEN 与 Q-DPP 如何组织联合探索，以及额外的信息与表示条件。
- 区分 HAPPO 的顺序参数更新、A2PO 的评价修正与 MAT 的顺序动作生成。
- 核对理论中的精确优势与信赖域条件，不把 PPO 裁剪或网络结构当成回报保证。

#### [自对弈与开放式多智能体学习：评估、目标与策略种群](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent-populations/)

自对弈怎样产生课程和训练标签，又怎样通过历史保留、交互评价与策略种群发现值得继续学习的问题？

- 从自对弈出发，分清最新响应、历史平均与搜索策略标签。
- 用手算博弈树贯通 AlphaGo、AlphaGo Zero、AlphaZero 与 MuZero 的学习闭环。
- 建立评估、目标构建、响应学习与重新评估的循环，区分循环克制与可检验进步。
- 用 gamescape、DO/PSRO 与课程机制研究策略保留，再扩展到 COLE/HOLA 的合作组合。
- 区分可利用度、种群安全价值、陌生伙伴泛化与全生命期收益。

#### [对手建模与递归推理：预测谁，回应什么？](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent-reasoning/)

给定参与者和评价目标，怎样利用行为预测、条件响应与有限递归改善决策，并检验模型是否可信？

- 区分真实信息、行为相关与模型假设的响应。
- 推导 PR2 的软响应、普通期望与软价值的不同梯度。
- 理解 GR2 的有限递归、ROMMEO 的经验约束与 GSCU 的模型使用选择。
- 分别检验预测、控制收益与模型失配，保留理论及实现条件。

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


# 持续强化学习：按问题索引

顺序学习请使用[三册教材目录](https://yingwen.io/zh/continual-rl/start/)。以下按概念联系查阅；合订讲义保留各专题的完整正文。

## I · 强化学习问题与目标

从智能体与世界的持续交互出发，区分外部设计者的偏好、实际奖励和评价准则。奖励是假设与设计的对象，不是结果好坏的天然真值。折扣、时域和平均奖励进一步规定跨时间的比较。

- [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)
- [奖励假设与奖励设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/)
- [平均奖励与差分价值](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)

目标规定要改善什么。奖励机制传递学习信号。状态规定决策时可以使用什么信息。三者需要分别定义。

## II · 状态构造与表征

环境状态通常不可直接获得。智能体从观测、动作和奖励历史构造内部状态。状态更新、特征学习和参数学习是不同的计算过程。

- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)

给定状态后，可以定义关于未来的预测。状态是否充分，取决于它需要支持哪些预测与决策。

## III · 预测与预测知识

先固定策略，再学习回报预测。MC、TD 和资格迹提供不同的估计方法。GVF 进一步指定累积信号、延续条件和目标策略。预测不自动产生更好的行为。

- [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)
- [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)

控制需要使用预测来改变策略。策略改变后，数据分布和需要预测的未来也会改变。

## IV · 控制与策略改善

先区分固定策略的改善与完整学习器的生命期表现。持续控制中，动作同时改变世界、未来数据和后续学习。经典 GPI、动作价值和策略梯度提供局部工具；历史条件、可行比较器与资源限制决定这些工具能支持什么结论。

- [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)
- [深度价值学习](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)
- [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)
- [最大熵控制](https://yingwen.io/zh/continual-rl/algorithms/soft-control/)

这些控制方法都需要分配信用并选择更新尺度。持续交互要求学习过程也能适应。

## V · 时间信用分配

多步回报从未来反馈构造学习目标。资格迹在反馈到来时更新过去相关的参数方向。递归网络还需要传播参数对内部状态的影响。信用分配可以用于预测，也可以用于控制。

- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

信用分配规定反馈如何作用于过去。流式协议进一步限制数据保存与每步计算。

## VI · 流式学习与更新稳定性

流式学习规定经验何时使用、能否重放。归一化和更新尺度控制处理数值稳定性。这些约束与机制不等于信用分配，也不要求使用元学习。

- [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

固定的更新规则可以依据当前数据调整尺度。元学习进一步用经验改善学习规则本身。

## VII · 元学习与学习规则的适应

在线元梯度估计学习参数对后续表现的影响。跨任务元强化学习则利用任务分布，学习初始化、上下文推断或更新规则。两者的训练协议与评价对象需要分别定义。

- [元学习与学习规则的适应](https://yingwen.io/zh/continual-rl/algorithms/meta/)

有了学习机制，还需要规定可复用行为的目标、启动条件与终止规则。

## VIII · 子任务与时间抽象

任务的总体目标与子任务目标需要区分。目标条件化策略描述一族控制问题。Option 则规定行为的启动条件、内部策略和终止规则。技能发现决定哪些行为进入可用集合。

- [目标条件化与子任务构造](https://yingwen.io/zh/continual-rl/construction/goals/)
- [Options 与技能发现](https://yingwen.io/zh/continual-rl/construction/options/)

能够执行一个行为，不等于知道它的后果。规划还需要这个行为的模型。

## IX · 模型学习与规划

Dyna 用真实经验学习模型，再用模型更新价值。Option 模型预测随机时长行为的奖励和终点。规划决定在哪些状态、对哪些行为、使用多少次模型。

- [Dyna 与模型学习](https://yingwen.io/zh/continual-rl/algorithms/dyna/)
- [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)
- [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)

模型、状态、技能与策略都会变化。接下来需要研究这些变化怎样影响保留、适应与探索。

## X · 长期适应、保留与探索

遗忘指已有能力下降。可塑性损失指学习新能力的速度下降。探索改变后续经验。这三类问题需要不同的对照实验，不能用同一个训练误差代替。

- [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)
- [可塑性与特征更新](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)
- [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)

在完整智能体中，这些机制共享经验与计算资源。它们的相互作用需要单独研究。

## XI · 持续学习的智能体架构

状态为预测、策略和模型提供输入。子任务产生可学习的行为。模型支持规划。架构需要定义各模块的更新接口，也需要确定有限资源下的保留与替换规则。

- [持续学习的智能体架构](https://yingwen.io/zh/continual-rl/construction/architectures/)

架构提出机制假设。实验检验实现、机制与长期收益。

## XII · 强化学习实验方法

先定义要估计的量，再选择任务、基线与随机单位。开发数据用于选择配置。独立测试用于估计表现。持续学习还要区分在线收益、冻结诊断和恢复过程。

- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

实验设计贯穿全部章节。每学完一个更新规则，就可以运行小问题、提出反例并检验假设。

# 从研究问题进入持续强化学习

这是一种教学组织，不是唯一分类。先明确目标、信息与干预权限，再选择机制。

## 一生的“学得好”到底指什么？

现象：终局测试很高，但适应时损失巨大，或者依赖免费重置。

先区分：目标、交互协议和变化来源是三条独立轴；不是看到 continual 就必须使用某一个算法。

- 优化折扣收益、平均奖励率，还是一段生命期的累计效用？
- 变化发生在奖励、动力学、观测、动作集合，还是学习器自己诱导的数据分布？
- 任务 ID、边界、reset、replay、预训练经验各允许什么？

### 算法线与条件

- **折扣 / 有限时域 / 平均奖励**：改变如何评价未来与时间代价。条件与限制：平均奖励需要相应的遍历/通信等假设，并不自动解决变化。
- **非平稳 MDP / 潜在情境 POMDP**：前者描述外部随时间改变；后者把未观测情境纳入状态。条件与限制：同一表象可以有不同模型，不要混用各自理论保证。
- **任务序列 / task-agnostic / single-life**：改变可用信息和干预权限。条件与限制：知道边界不等于知道任务身份；没有手动 reset 不等于永不终止。

诊断：同一策略分别计算整体收益、变化后损失和每真实时间单位收益；先看结论是否因评价方式翻转。

基准：CSuite 适合持续目标；Continual World 适合任务序列；Single-Life RL 适合一次试验中的恢复。三者不能直接混排。

- [Khetarpal et al. · Towards Continual RL](https://arxiv.org/abs/2012.13490)：按非平稳性的发生位置和驱动力组织问题；提醒我们先写设定，再比较方法。
- [Abel et al. · A Definition of Continual RL](https://arxiv.org/abs/2307.11046)：将持续性落在智能体是否需要一直学习，而不只落在外部任务切换。这里采用其问题视角，不把下面的教学分类说成该文定理。
- [Pan et al. · A Survey of Continual RL](https://arxiv.org/abs/2506.21872)：补充任务序列、保留、迁移、容量和评测的广泛方法线；其任务化视角与 Abel 的定义不是完全相同的设定。
- [Wan et al. · Learning and Planning in Average-Reward MDPs](https://proceedings.mlr.press/v139/wan21a.html)：Differential TD/Q 与平均奖励规划，区分奖励率和相对价值。

---

## 规律变了，怎样及时发现并改正？

现象：旧估计非常稳定，却迟迟跟不上新奖励或新动力学。

先区分：没采到变化、统计估计太慢、情境识别错误，是三种不同失败；不必先假定神经网络可塑性坏了。

- 变化是突变、渐变还是周期重现？
- 检测误报和检测延迟如何影响控制？
- 是忘掉旧统计量，还是保存旧模型并识别何时复用？

### 算法线与条件

- **常数步长 / 指数遗忘 / 滑动窗口**：使近期数据有更大权重。条件与限制：更快适应会增加估计方差；窗口按真实时间还是访问次数要写清。
- **变化点检测 / 多模型 / 上下文推断**：判断何时切换学习状态或选择旧专家。条件与限制：TD error 大也可能只是探索到了新状态，不能直接当变化证明。
- **SWUCRL2-CW / BORL；permanent–transient values**：分别从非平稳遗憾、快慢价值分解处理持续修正。条件与限制：理论设定、任务边界权限、保存状态和深度实现并不相同。

诊断：先用固定数据流比较估计器，再让动作影响数据；同时记录预测误差、检测延迟/误报和变化后损失。

基准：先分段平稳或渐变 bandit / 小 MDP；再进入 task-agnostic Meta-World 或 AgarCL，不从最高维平台猜失败来源。

- [Cheung et al. · Non-stationary MDPs](https://proceedings.mlr.press/v119/cheung20a.html)：SWUCRL2-CW 与 BORL 把滑窗、乐观探索和变化预算联系起来；不是只换一个常数学习率。
- [Anand & Precup · Prediction and Control in CRL](https://arxiv.org/abs/2312.11669)：用 permanent / transient 两个价值分量连接长期积累与短期修正；区分可见边界与持续设置。
- [Caccia et al. · Task-Agnostic CRL](https://proceedings.mlr.press/v232/caccia23a.html)：3RL 把循环状态与回放结合，用历史推断情境；任务未知不等于历史中没有可推断的信息。

---

## 当前画面不够，应该记住什么？

现象：相同观测需要不同动作；更大的前馈网络仍做错。

先区分：工作状态更新、持久知识学习和训练梯度记忆不是同一个东西。

- 哪些历史必须被区分，哪些可丢弃？
- 记忆的训练目标是奖励、观测预测还是一组未来测试？
- 长依赖的梯度怎样在每步预算内传播？

### 算法线与条件

- **Frame stack / RNN / Transformer**：从固定窗口到递归状态或上下文。条件与限制：更长上下文不自动等于更强长期权重学习。
- **Belief / PSR / GVFN**：以后验或预测赋予状态可解释语义。条件与限制：已知生成模型、核心测试和任意学习到的预测集合并不等价。
- **BPTT / TBPTT / RTRL / RTU**：为同一递归表示设计不同信用计算。条件与限制：detach 切断梯度，不一定抹掉前向记忆；成本必须随容量报告。

诊断：用同画面异动作的平衡线索任务，加上可观测状态参照、手工记忆参照，再比较学得记忆。

基准：从 delayed cue 到 POPGym/Forager；必须控制记忆容量、历史长度、每步计算与特权信息。

- [Caccia et al. · Task-Agnostic CRL](https://proceedings.mlr.press/v232/caccia23a.html)：3RL 把循环状态与回放结合，用历史推断情境；任务未知不等于历史中没有可推断的信息。
- [Schlegel et al. · General Value Function Networks](https://arxiv.org/abs/1807.06763)：让递归表示受到预测问题的约束，连接状态构造与预测学习。
- [Elelimy et al. · Real-Time Recurrent Learning using Trace Units](https://arxiv.org/abs/2409.01449)：结构化递归与前向敏感度降低在线信用分配成本；研究重点不只是网络能否记住。

---

## 新知识怎样不破坏仍有用的旧知识？

现象：学完 B 后，在相同协议下重新测 A，A 的表现下降。

先区分：保留准确的旧知识不等于保留已经失效的答案；保留与跟踪可能冲突。

- 该保留原始经验、输出行为、参数约束还是独立模块？
- buffer 应代表过去平均分布还是未来会重访的情境？
- 没有任务 ID 时，如何决定使用哪个知识模块？

### 算法线与条件

- **Replay / CLEAR / generative replay**：重新暴露旧训练信号，必要时增加行为蒸馏。条件与限制：旧 transition 可能已不符合当前动力学；生成模型本身也会忘。
- **EWC / distillation / policy consolidation**：约束重要参数或旧输入上的输出。条件与限制：保护强度太大可能阻止适应；重要性估计不等于未来用途。
- **Progress & Compress / 模块化 / PT values**：隔离快学与长存，或按情境重用。条件与限制：存储、路由、压缩和边界权限都要计入成本。

诊断：做 A→B→A 的保留矩阵，同时测 B 的学习速度；增加固定内存和未知任务 ID 的对照。

基准：Continual World / COOM / continual_rl；同时报告旧任务、当前任务和前向迁移，不能只给最终平均分。

- [Rolnick et al. · Experience Replay for Continual Learning](https://arxiv.org/abs/1811.11682)：CLEAR 结合旧经验上的离策略学习、当前经验学习与行为克隆；回放不只是存一个 buffer。
- [Schwarz et al. · Progress & Compress](https://arxiv.org/abs/1805.06370)：把快速学习与压缩到长期知识分开；检查蒸馏、保留约束和协议权限。
- [Anand & Precup · Prediction and Control in CRL](https://arxiv.org/abs/2312.11669)：用 permanent / transient 两个价值分量连接长期积累与短期修正；区分可见边界与持续设置。
- [Wołczyk et al. · Continual World](https://arxiv.org/abs/2105.10919)：机械臂任务序列同时考察保留与前向迁移；不是无重置单生命期环境。

---

## 训练久了，为什么新东西越来越难学？

现象：老网络获得相同新数据和预算，却比合适的 fresh 参照学得慢。

先区分：新任务误差高可能只是起点不同；神经元稀疏、梯度小等是诊断，不是可塑性结论本身。

- 是数据覆盖差、目标不稳定，还是表示和优化退化？
- 替换哪个单元，何时替换，损失多少旧功能？
- 收益是一次重启，还是长生命史持续保持的新学习能力？

### 算法线与条件

- **CReLU / normalization / regularization**：改变激活、尺度或参数轨迹。条件与限制：需和同预算学习率及初始化控制比较。
- **ReDo / Continual Backprop / shrink-and-perturb**：重新引入可训练特征或扰动。条件与限制：必须测输出扰动、optimizer 状态、替换频率和旧知识。
- **Primacy reset / replay-ratio 控制**：减弱早期经验固化或过拟合。条件与限制：保留 buffer 的网络重置不等于无先验从头学习。

诊断：固定新数据、模型容量和调参预算；比较 aged / fresh / random-reset，并分开初始误差与后续下降速度。

基准：先做受控新目标学习，再到 Atari 模式/任务变化与 AgarCL；可塑性方法在一种场景有效不保证所有 CRL 场景有效。

- [Dohare et al. · Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7)：长期新学习能力及 Continual Backprop；特征替换的效果不能仅由激活率判断。
- [Abbas et al. · Loss of Plasticity in Continual Deep RL](https://proceedings.mlr.press/v232/abbas23a.html)：在价值学习和不同非平稳条件下研究激活、梯度与后续学习，讨论 CReLU。
- [Nikishin et al. · The Primacy Bias in Deep RL](https://arxiv.org/abs/2205.07802)：区分早期经验固化与后续证据利用不足；参数重置可能保留 replay，不能视作全部重来。
- [Mohamed et al. · AgarCL](https://arxiv.org/abs/2505.18347)：非 episodic、高维、部分可观测且持续变化的研究平台；论文中若干可塑性方法提升有限，说明困难不止一条。

---

## 每步预算很小，怎样可靠地分配信用？

现象：实时更新发散、梯度延迟太长，或者一个环境步用了大量隐含回放。

先区分：数据协议、长期收益目标、表示结构和优化器是可组合的选择，不是一条算法世代替换线。

- 一个新误差应该影响哪些过去参与计算的量？
- off-policy 修正改变的是采样分布、目标还是更新方向？
- 尺度、步长、trace 与 actor/critic 怎样相互作用？

### 算法线与条件

- **TD(λ) / true-online / emphatic / gradient TD**：改变时间信用或离策略稳定机制。条件与限制：各式 eligibility 不通用；γ、ρ 的下标与支持集必须对齐。
- **Stream-X / 输入与更新归一化**：把深度每步更新做成可计算的训练流程。条件与限制：batch_size=1 不足以复现方法。
- **IDBD / TIDBD / meta-gradient**：让部分更新参数适应当前学习过程。条件与限制：监督误差、bootstrap 误差与策略目标的导数不能互换。

诊断：先验证 λ=0、终止、ρ=0 等退化情况，再记录每步延迟、内存、更新/样本比以及多 seed 回报。

基准：表格/线性解析任务 → 严格 streaming 控制；必须说明是否重复访问旧样本和隐藏的大批量计算。

- [Stream-X · 作者实现与论文入口](https://github.com/mohmdelsayed/streaming-drl)：研究逐条经验的深度 RL 更新；归一化、迹、更新幅度和 actor–critic 需要一起检查。
- [Elelimy et al. · Real-Time Recurrent Learning using Trace Units](https://arxiv.org/abs/2409.01449)：结构化递归与前向敏感度降低在线信用分配成本；研究重点不只是网络能否记住。
- [Xu et al. · Meta-Gradient Reinforcement Learning](https://arxiv.org/abs/1805.09801)：对学习更新求导以调整 return 等元参数；元目标不能随便跟训练题目一起改变。

---

## 经验该变成哪些预测，才能在未来派上用场？

现象：奖励之外的规律反复重学；辅助损失下降却没有改善决策。

先区分：定义一个预测问题、学准该预测、让下游使用它，是三个独立研究步骤。

- 预测信号、条件行为和终止尺度如何选择？
- 预测是否可由当前经验覆盖？
- 下游是状态构造、奖励迁移、技能发现还是规划？

### 算法线与条件

- **GVF / Horde / nexting**：多个 c、γ、π 定义多种未来知识。条件与限制：不是所有累计信号都具有事件概率语义。
- **Auxiliary tasks / predictive representation**：用额外学习信号塑造共享表示。条件与限制：辅助目标也可能争抢容量或产生干扰。
- **Successor features / GPI**：预测特征累计量，再按奖励权重重估价值。条件与限制：依赖可复用的动力学、策略与奖励特征。

诊断：分别测 held-out 预测误差、删除预测输入后的决策变化、分布变化后的再学习成本。

基准：先用多信号小 MDP 给出解析答案，再测试新奖励重估；预测数量多不能代替知识用途。

- [Sutton et al. · Horde](https://sites.ualberta.ca/~amw8/horde.pdf)：多个不同策略、信号和时间尺度的预测共享经验；学习问题和回答问题的算法分开。
- [Jaderberg et al. · Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397)：多个辅助控制/预测信号训练共享表示；辅助任务的用途需要用主任务表现检验。
- [Barreto et al. · Successor Features for Transfer](https://arxiv.org/abs/1606.05312)：在相同动力学、线性奖励族中重用后果预测，并用 GPI 组合策略。

---

## 怎样形成可复用的目标与长行为？

现象：每个任务都从原始动作重新探索，技能库却越积越多而用不上。

先区分：目标定义要学什么，option 定义怎样行动；技能有多样性不代表对主任务有用。

- 子目标由谁提出，怎样衡量用途？
- 技能该何时启动和停止？
- 新任务能否组合旧能力，而不只是重新训练一遍？

### 算法线与条件

- **UVFA / HER / goal generation**：目标条件化、重标记经验、选择训练目标。条件与限制：三者改变不同对象；HER 不是自动课程本身。
- **Option-Critic / eigenoptions / DIAYN / DADS**：任务驱动、图结构或信息量驱动技能学习。条件与限制：技能发现目标与主任务价值之间仍有距离。
- **Reward-respecting subtasks / SF-GPI / Option Keyboard**：按价值或奖励组合接口复用能力。条件与限制：GPI 不需要 commitment；option 的持续执行和终止是另一层。

诊断：对照 primitive-only、随机技能与同预算学得技能；同时计发现、建模、选择和执行的成本。

基准：先两步技能/导航连通性，再到连续技能；报告技能覆盖、真实主回报和学习总成本。

- [Schaul et al. · Universal Value Function Approximators](https://proceedings.mlr.press/v37/schaul15.html)：价值函数以目标为输入；给定目标后的泛化，不等于自动选择目标。
- [Sutton, Precup & Singh · Between MDPs and semi-MDPs](https://doi.org/10.1016/S0004-3702(99)00052-1)：用启动集合、内部策略和终止规则定义时间扩展行为，并导出相应备份。
- [Sutton et al. · Reward-Respecting Subtasks](https://arxiv.org/abs/2202.03466)：连接子任务、option、后果模型和规划；任务回报与抽象价值的关系是重点。
- [Barreto et al. · Successor Features for Transfer](https://arxiv.org/abs/1606.05312)：在相同动力学、线性奖励族中重用后果预测，并用 GPI 组合策略。

---

## 怎样用后果模型减少真实试错？

现象：模型内预测很好，现实行为却错误；规则变后越规划越自信。

先区分：动作模型、option model、价值函数和规划器分开；模型正确性与求解模型的精度也分开。

- 模型需预测完整分布，还是仅预测某个 backup 所需的统计量？
- 变化发生后模型怎样更新、忘记或重用？
- 把计算花在哪些起点、尺度和分支上？

### 算法线与条件

- **Dyna / prioritized sweeping**：用一步模型进行额外价值更新。条件与限制：模型学习与规划分配都影响结果。
- **Option / expectation / abstract models**：压缩时间或后果以降低规划成本。条件与限制：非线性下游一般不能只接一个均值。
- **MPC / MCTS / Dreamer / Continual-Dreamer**：在线搜索或想象中策略学习。条件与限制：世界模型、critic、actor 和 buffer 可能分别遗忘。

诊断：用已知模型确认 planner，再替换为学习模型；固定真实步数和总计算，画模型误差及真实回报随规划预算变化。

基准：blocking / shortcut maze → Minigrid / Minihack → 高维持续任务；不给某方法额外真值地图。

- [Sutton et al. · Reward-Respecting Subtasks](https://arxiv.org/abs/2202.03466)：连接子任务、option、后果模型和规划；任务回报与抽象价值的关系是重点。
- [Kessler et al. · World Models for Continual RL](https://proceedings.mlr.press/v232/kessler23a.html)：Continual-Dreamer 研究世界模型、经验选择与持续探索；模型与策略要分别测保留。
- [Cheung et al. · Non-stationary MDPs](https://proceedings.mlr.press/v119/cheung20a.html)：SWUCRL2-CW 与 BORL 把滑窗、乐观探索和变化预算联系起来；不是只换一个常数学习率。

---

## 接下来学什么、去哪探索，还能否安全回来？

现象：永远遇不到新证据，或者探索一次就失去后续交互与学习的机会。

先区分：新奇、信息增益、学习进展、主任务收益和可恢复性可能互相冲突。

- 如何区分未学会与不可约噪声？
- teacher 是否能挑任务、重置位置或改环境？
- 没有外部救援时怎样控制不可逆风险？

### 算法线与条件

- **Counts / RND / uncertainty**：按新奇或不确定性分配行为。条件与限制：误差大不保证可学，也不保证值得冒险。
- **Learning progress / ALP-GMM / 自动课程**：优先选择仍能提升能力的目标或环境。条件与限制：训练分布由 teacher 控制，需要明确权限。
- **Reset-free / recovery / Single-Life RL**：把回到可进行后续学习的状态纳入决策。条件与限制：给负奖励不等于具有安全保证；预训练经验也要计入。

诊断：除外在回报外记录覆盖、预测学习进展、人工干预、失能时间和恢复成功率；危险实验先在模拟器做。

基准：可控稀疏奖励任务、可参数化课程、Single-Life RL；不把有 teacher 的结果直接称为完全自主探索。

- [Burda et al. · Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)：固定随机目标与可训练预测器产生新奇信号；它不是可学习进展或安全性的直接度量。
- [Portelas et al. · Teacher Algorithms for Curriculum Learning](https://arxiv.org/abs/1910.07224)：以学习进展选择可参数化环境；需要说明 teacher 是否能重置和控制环境。
- [Chen et al. · You Only Live Once](https://arxiv.org/abs/2210.08863)：给定先前经验，在一次测试生命期中自主适应并恢复；不同于“训练完全没有先验”。

---

## 能不能让学习过程本身越用越合适？

现象：每次换环境都重新调步长，或者短期学得快却长期不稳定。

先区分：调整在线超参数、学习初始化、推断上下文、学习更新规则不是同一种“学会学习”。

- 元目标在评价什么未来，使用哪些额外数据？
- 对参数、隐藏状态还是完整 optimizer 状态求导？
- 元训练任务重置和部署持续性如何对齐？

### 算法线与条件

- **IDBD / TIDBD / meta-gradient RL**：在线改变步长或 return 参数。条件与限制：需要处理敏感度近似、尺度和训练/元评价目标。
- **MAML / context-based meta-RL / PEARL**：跨任务准备一个快速适应机制。条件与限制：元测试从头适应不证明长期累积。
- **Learned update rules / DiscoRL**：把学习规则参数化后优化其跨任务结果。条件与限制：发布规则的使用与重新进行元搜索是不同成本。

诊断：先有限差分验证导数，再匹配固定超参数调参预算；改变任务分布和学习时域，检查收益是否仍存在。

基准：标量敏感度 → 受控在线漂移 → 多任务元训练/测试；不同层次的结果分开报告。

- [Xu et al. · Meta-Gradient Reinforcement Learning](https://arxiv.org/abs/1805.09801)：对学习更新求导以调整 return 等元参数；元目标不能随便跟训练题目一起改变。
- [Finn et al. · MAML](https://proceedings.mlr.press/v70/finn17a.html)：跨任务训练易适应的初始化；重新从同一初始化适应和累积一段生命史是不同协议。
- [Abel et al. · A Definition of Continual RL](https://arxiv.org/abs/2307.11046)：将持续性落在智能体是否需要一直学习，而不只落在外部任务切换。这里采用其问题视角，不把下面的教学分类说成该文定理。

---

## 把模块接起来，怎样证明真的在持续积累？

现象：每个模块单测正常，整机却没有长期收益，结论只剩一条平均曲线。

先区分：内部活动、局部预测改善、持续参数更新和整体能力增长，需要不同证据。

- 状态变化会让哪些预测、技能和模型失效？
- 计算与记忆如何分给多个相互影响的学习器？
- 如何分开适应、保留、迁移、可塑性和安全？

### 算法线与条件

- **Life-long metrics / benchmark protocols**：测全过程，而不只测最终政策。条件与限制：eval reset、探针数据和冻结评测都会引入额外权限。
- **模块化消融 / 接口与版本管理**：固定一个对象，检查其真实下游用途。条件与限制：单模块更准不保证整个控制循环更好。
- **Alberta Plan / OaK / integrated agents**：把知识构建与长期更新接为研究路线。条件与限制：路线图不是已完成的端到端保证。

诊断：固定总预算，分别冻结状态、预测、模型或元更新；同时报告整段效用、恢复、旧知识、新学习、内存与延迟。

基准：由解析诊断到完整平台逐级增加困难。AgarCL、CSuite、Continual World 各覆盖部分问题，不存在一个分数概括整张地图。

- [Abel et al. · A Definition of Continual RL](https://arxiv.org/abs/2307.11046)：将持续性落在智能体是否需要一直学习，而不只落在外部任务切换。这里采用其问题视角，不把下面的教学分类说成该文定理。
- [Sutton et al. · The Alberta Plan](https://arxiv.org/abs/2208.11173)：将从经验持续构建知识组织为研究计划；研究路线与已验证的整机能力分开。
- [Oak Lab · The OaK Architecture](https://oaklab.ai/posts/the-oak-architecture)：用知识、抽象和元学习连接持续智能体模块；本指南把它作为架构路线，而非已完成的通用算法。
- [Mohamed et al. · AgarCL](https://arxiv.org/abs/2505.18347)：非 episodic、高维、部分可观测且持续变化的研究平台；论文中若干可塑性方法提升有限，说明困难不止一条。
- [DeepMind · CSuite 文档](https://rl-csuite.readthedocs.io/en/latest/)：持续交互的小型环境，可在复杂平台前隔离机制；continuing 不自动等于包含所有 continual 困难。


# 持续强化学习：统一学习地图

目录与算法谱系使用同一套章节结构。优化目标、状态、预测知识和控制分别讨论。跨章节关系表示概念依赖或模块组合。

## 只有基础 RL：先完成一个小闭环

不必先补齐全部深度 RL。按下面五步，先理解持续任务、追踪变化、完成实验，再认识预测与长行为。

1. [分清持续任务与学习协议](https://yingwen.io/zh/continual-rl/start/01-lifetime/)：能说明什么变化、什么重置。
2. [常数步长怎样跟踪变化](https://yingwen.io/zh/continual-rl/start/02-tracking/)：能手算一次更新并解释新旧经验权重。
3. [跑完第一份实验](https://yingwen.io/zh/continual-rl/first-project/)：保存多种子结果，解释恢复速度与波动。
4. [把 TD 推广成 GVF](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)：能写出一个非奖励预测问题。
5. [把单步动作推广成 option](https://yingwen.io/zh/continual-rl/construction/options/)：能区分行为、终止与行为后果模型。

再读平均奖励、模型与规划。根据实验需要补充状态、子任务、保留与可塑性。

## 已有 DQN / PPO 背景：先选问题，再补接口

先定义目标和交互协议。再选择一种学习对象或机制。

1. [确定持续交互的目标](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)：说明为何使用折扣或平均奖励。
2. [明确评测与预算](https://yingwen.io/zh/continual-rl/experiments/)：把 reset、replay、task ID 写进协议。
3. [选一个学习对象或失败模式](https://yingwen.io/zh/continual-rl/algorithms/#textbook-prediction)：只改变一个接口或一种更新机制。
4. [把公式对应到原始实现](https://yingwen.io/zh/continual-rl/construction/implementations/)：追踪 target、梯度、mask 与更新顺序。
5. [设计研究对照](https://yingwen.io/zh/continual-rl/research/)：先排除更多数据、计算和调参的解释。

架构章讨论模块组合。研究可塑性或流式更新时，可先阅读相应章节。

## I · 强化学习问题与目标

什么结果算更好？

完成标志：目标规定要改善什么。奖励机制传递学习信号。状态规定决策时可以使用什么信息。三者需要分别定义。

### [B4 · 奖励假设与奖励设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/)

奖励能表达哪些偏好，怎样让学习信号与设计意图一致？

- 先修：[B0 交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)
- 学会什么：推导塑形、偏好梯度和 MaxEnt，并检验时间结构、代理失效与约束。
- 动手：[奖励语义、边界项与梯度反例](https://yingwen.io/zh/continual-rl/foundations/reward-design/#lesson-code)

### [B0 · 交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)

什么结果算更好？

- 先修：基础 RL 符号
- 学会什么：区分奖励、回报、价值、策略和在线表现。
- 动手：[目标选择与奖励塑形实验](https://yingwen.io/zh/continual-rl/foundations/objectives/#lesson-code)

### [B1 · 生命期、变化与交互协议](https://yingwen.io/zh/continual-rl/start/01-lifetime/)

continuing、continual、streaming、single-life 差在哪？

- 先修：[A2 持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)
- 学会什么：明确环境与学习器各在何时重置。
- 动手：[从变化 bandit 完成第一份实验](https://yingwen.io/zh/continual-rl/first-project/)

### [B2 · 平均奖励与 differential learning](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)

没有自然终点时，如何学每单位时间的收益？

- 先修：[A1 价值预测：MC → TD → traces](https://yingwen.io/zh/continual-rl/algorithms/value/)；[A2 持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)
- 学会什么：区分折扣价值、奖励率、差分价值与时长。
- 动手：[Differential TD/Q 原代码与公式检查](https://yingwen.io/zh/continual-rl/construction/implementations/#impl-differential)

## II · 状态构造与表征

智能体需要从历史中保留什么信息？

完成标志：给定状态后，可以定义关于未来的预测。状态是否充分，取决于它需要支持哪些预测与决策。

### [C2 · 状态构造：把历史变成决策依据](https://yingwen.io/zh/continual-rl/construction/state/)

哪些不同历史必须区分？

- 先修：[B0 交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)
- 学会什么：区分 hidden state、预测状态与学习参数。
- 动手：[GVFN 的问题、网络与递归更新](https://yingwen.io/zh/continual-rl/construction/implementations/#impl-gvfn)

## III · 预测与预测知识

在指定行为下，未来会发生什么？

完成标志：控制需要使用预测来改变策略。策略改变后，数据分布和需要预测的未来也会改变。

### [A1 · 价值预测：MC → TD → traces](https://yingwen.io/zh/continual-rl/algorithms/value/)

没有完整回报时怎样逐步学预测？

- 先修：基础 RL 符号
- 学会什么：手算 TD target，解释偏差、方差与 λ。
- 动手：[固定策略预测实验](https://yingwen.io/zh/continual-rl/algorithms/#algorithm-code)

### [C1 · GVF：从一个回报到一组预测](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)

问什么、在哪个策略下问、问多远？

- 先修：[A1 价值预测：MC → TD → traces](https://yingwen.io/zh/continual-rl/algorithms/value/)
- 学会什么：写出 c、γ、π，并追踪 off-policy 更新。
- 动手：[两步 GVF → RLPark / Horde](https://yingwen.io/zh/continual-rl/construction/implementations/#impl-rlpark)

## IV · 控制与策略改善

智能体怎样选择动作并改善行为？

完成标志：这些控制方法都需要分配信用并选择更新尺度。持续交互要求学习过程也能适应。

### [A2 · 持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

从固定策略到持续适应的智能体，评价对象怎样变化？

- 先修：[A1 价值预测：MC → TD → traces](https://yingwen.io/zh/continual-rl/algorithms/value/)；[B0 交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)
- 学会什么：定义历史条件价值、生命期指标与可行比较器；区分信息收益和不可逆后果。
- 动手：[完整学习器与策略排序反例](https://yingwen.io/zh/continual-rl/algorithms/control/#lesson-code)

### [A3 · 深度价值：DQN 与扩展](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)

函数逼近、bootstrap 与复用数据怎样相互影响？

- 先修：[A2 持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)
- 学会什么：定位 replay、target network 和 Double DQN。
- 动手：[从单文件 DQN 追一次更新](https://yingwen.io/zh/continual-rl/algorithms/deep-value/#algorithm-sources)

### [A4 · 策略梯度 → actor–critic → PPO](https://yingwen.io/zh/continual-rl/algorithms/policy/)

怎样直接改动作概率，又控制更新幅度？

- 先修：[A1 价值预测：MC → TD → traces](https://yingwen.io/zh/continual-rl/algorithms/value/)
- 学会什么：对应 log-prob、advantage 和 PPO surrogate。
- 动手：[REINFORCE 与 actor–critic 小实验](https://yingwen.io/zh/continual-rl/algorithms/#algorithm-code)

### [A5 · 连续控制：DDPG / TD3 / SAC](https://yingwen.io/zh/continual-rl/algorithms/soft-control/)

连续动作与熵正则如何改变 actor 和 critic？

- 先修：[A2 持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)；[A4 策略梯度 → actor–critic → PPO](https://yingwen.io/zh/continual-rl/algorithms/policy/)
- 学会什么：分开 off-policy 数据、双 Q 与熵目标。
- 动手：[阅读 SAC 数据与 target 路径](https://yingwen.io/zh/continual-rl/algorithms/soft-control/#algorithm-sources)

## V · 时间信用分配

当前反馈应当更新过去哪些预测与决策？

完成标志：信用分配规定反馈如何作用于过去。流式协议进一步限制数据保存与每步计算。

### [D0 · 时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

当前反馈应怎样更新过去的预测与决策？

- 先修：[A1 价值预测：MC → TD → traces](https://yingwen.io/zh/continual-rl/algorithms/value/)
- 学会什么：区分前向目标、后向资格迹、递归状态导数和元敏感度。
- 动手：[前后向等价与在线资格迹实验](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/#lesson-code)

## VI · 流式学习与更新稳定性

在有限内存和逐步计算预算下，怎样执行学习？

完成标志：固定的更新规则可以依据当前数据调整尺度。元学习进一步用经验改善学习规则本身。

### [D3 · 流式学习：Stream-X 与更新尺度控制](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

在有限内存和逐步预算下，怎样稳定地更新？

- 先修：[D0 时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)；[B1 生命期、变化与交互协议](https://yingwen.io/zh/continual-rl/start/01-lifetime/)
- 学会什么：区分交互协议、资格迹、尺度控制与元梯度学习。
- 动手：[逐行追踪流式更新](https://yingwen.io/zh/continual-rl/algorithms/streaming/#algorithm-sources)

## VII · 元学习与学习规则的适应

怎样根据学习效果调整学习过程？

完成标志：有了学习机制，还需要规定可复用行为的目标、启动条件与终止规则。

### [D5 · Meta-learning：步长、初始化与学习规则](https://yingwen.io/zh/continual-rl/algorithms/meta/)

一次学习怎样影响后续学习，怎样对这种影响求导？

- 先修：[A4 策略梯度 → actor–critic → PPO](https://yingwen.io/zh/continual-rl/algorithms/policy/)；[B1 生命期、变化与交互协议](https://yingwen.io/zh/continual-rl/start/01-lifetime/)
- 学会什么：区分在线元梯度、跨任务适应、上下文推断与学得的更新规则。
- 动手：[两步学习的元梯度与数值校验](https://yingwen.io/zh/continual-rl/algorithms/meta/#lesson-code)

## VIII · 子任务与时间抽象

哪些可复用行为值得学习？

完成标志：能够执行一个行为，不等于知道它的后果。规划还需要这个行为的模型。

### [C3 · 目标构造：决定值得学什么](https://yingwen.io/zh/continual-rl/construction/goals/)

目标、内在奖励和子任务怎样定义用途？

- 先修：[A2 持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)；[C1 GVF：从一个回报到一组预测](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)
- 学会什么：区分 goal conditioning、目标生成与价值保留。
- 动手：[最短路与 reward-respecting 子任务](https://yingwen.io/zh/continual-rl/construction/#construction-code)

### [C4 · Options：行为的时间抽象](https://yingwen.io/zh/continual-rl/construction/options/)

怎样发现、执行、终止和复用长行为？

- 先修：[A2 持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)
- 学会什么：定义 I、π、β；区分技能发现与 SF/GPI。
- 动手：[Option-Critic → eigenoptions](https://yingwen.io/zh/continual-rl/construction/implementations/#impl-option-critic)

## IX · 模型学习与规划

怎样用学得的后果模型改善决策？

完成标志：模型、状态、技能与策略都会变化。接下来需要研究这些变化怎样影响保留、适应与探索。

### [A6 · Dyna：学习模型，再做规划](https://yingwen.io/zh/continual-rl/algorithms/dyna/)

真实经验和模型生成的 backup 如何配合？

- 先修：[A2 持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)
- 学会什么：分开真实步数、模型误差和规划计算。
- 动手：[Q-learning 与 Dyna-Q 对照](https://yingwen.io/zh/continual-rl/algorithms/#algorithm-code)

### [C5 · 转移与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)

一个长行为会带来什么奖励、时长和终点？

- 先修：[C4 Options：行为的时间抽象](https://yingwen.io/zh/continual-rl/construction/options/)；[A6 Dyna：学习模型，再做规划](https://yingwen.io/zh/continual-rl/algorithms/dyna/)
- 学会什么：区分 option policy、option value 和 option model。
- 动手：[随机时长与期望特征反例](https://yingwen.io/zh/continual-rl/construction/models/#construction-example)

### [C6 · 抽象模型与规划](https://yingwen.io/zh/continual-rl/construction/planning/)

怎样用行为模型改善当前决策？

- 先修：[C5 转移与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)；[B2 平均奖励与 differential learning](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)
- 学会什么：写出折扣与平均奖励下不同的 option backup。
- 动手：[planning 小实验与时长公式检查](https://yingwen.io/zh/continual-rl/construction/implementations/#audit-duration)

## X · 长期适应、保留与探索

学习能否长期保持有效？

完成标志：在完整智能体中，这些机制共享经验与计算资源。它们的相互作用需要单独研究。

### [D1 · 知识保留：replay / EWC / 蒸馏](https://yingwen.io/zh/continual-rl/algorithms/retention/)

新知识会怎样干扰过去能力？

- 先修：[B1 生命期、变化与交互协议](https://yingwen.io/zh/continual-rl/start/01-lifetime/)；[A3 深度价值：DQN 与扩展](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)
- 学会什么：分开知识保留和重新适应的成本。
- 动手：[保留与适应的确定性反例](https://yingwen.io/zh/continual-rl/algorithms/#algorithm-code)

### [D2 · 可塑性：ReDo / Continual Backprop](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)

旧任务没忘，为何新任务仍越来越学不动？

- 先修：[B1 生命期、变化与交互协议](https://yingwen.io/zh/continual-rl/start/01-lifetime/)；[A3 深度价值：DQN 与扩展](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)
- 学会什么：设计 fresh-network 与替换策略对照。
- 动手：[定位 CBP 的替换与优化器状态](https://yingwen.io/zh/continual-rl/algorithms/plasticity/#algorithm-sources)

### [D4 · 探索：新奇与学习进展](https://yingwen.io/zh/continual-rl/algorithms/exploration/)

什么经验值得主动获取？

- 先修：[A2 持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)；[B1 生命期、变化与交互协议](https://yingwen.io/zh/continual-rl/start/01-lifetime/)
- 学会什么：区分预测误差、新奇、学习进步与最终用途。
- 动手：[给不可约噪声加一个对照](https://yingwen.io/zh/continual-rl/algorithms/exploration/#algorithm-practice)

## XI · 持续学习的智能体架构

各学习过程怎样共同改善长期行为？

完成标志：架构提出机制假设。实验检验实现、机制与长期收益。

### [E1 · 系统架构：Dyna → STOMP / OaK](https://yingwen.io/zh/continual-rl/construction/architectures/)

哪些部件学习，哪些仍由人提供？

- 先修：[C6 抽象模型与规划](https://yingwen.io/zh/continual-rl/construction/planning/)；[D3 流式学习：Stream-X 与更新尺度控制](https://yingwen.io/zh/continual-rl/algorithms/streaming/)
- 学会什么：画出数据、目标、参数和行为之间的接口。
- 动手：[比较模块已实现的连接与开放接口](https://yingwen.io/zh/continual-rl/construction/#architecture-comparison)

## XII · 强化学习实验方法

哪些数据能够支持所提出的结论？

完成标志：实验设计贯穿全部章节。每学完一个更新规则，就可以运行小问题、提出反例并检验假设。

### [E3 · 实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

哪些证据能够检验机制与性能？

- 先修：[B0 交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)
- 学会什么：分离开发与测试，按独立运行估计不确定性。
- 动手：[完整开发、选择与独立测试实验](https://yingwen.io/zh/continual-rl/experiments/#lesson-code)

### [B3 · 持续学习的评测与基准](https://yingwen.io/zh/continual-rl/start/07-evaluate/)

高平均回报掩盖了哪些失败？

- 先修：[B1 生命期、变化与交互协议](https://yingwen.io/zh/continual-rl/start/01-lifetime/)
- 学会什么：设计恢复、保留、新学习和资源指标。
- 动手：[写一份公平实验协议](https://yingwen.io/zh/continual-rl/download/protocol.md)

### [E2 · 从机制假设走到研究实验](https://yingwen.io/zh/continual-rl/research/)

在同等资源下，哪个改变真正带来长期收益？

- 先修：[B3 持续学习的评测与基准](https://yingwen.io/zh/continual-rl/start/07-evaluate/)
- 学会什么：写出最小基线、干预、消融与结论边界。
- 动手：[选择一个可检验的起步项目](https://yingwen.io/zh/continual-rl/construction/#construction-projects)

## 按用途查材料

- [算法历史与家族关系](https://yingwen.io/zh/continual-rl/download/algorithm-map.md)
- [作者代码与公式对照](https://yingwen.io/zh/continual-rl/construction/implementations/)
- [资源库：论文、课程、讲座、代码与基准](https://yingwen.io/zh/continual-rl/library/)


## 分册与可运行材料

这些文件按统一地图分工，不是几套独立的必修课。
- [表格、函数逼近与深度 RL 基础目录](https://yingwen.io/zh/continual-rl/download/foundations.md)
- [Part II：函数逼近与经典进阶](https://yingwen.io/zh/continual-rl/foundations/approximation/)
- [基础更新和长期学习机制](https://yingwen.io/zh/continual-rl/download/algorithms.md)
- [算法精读：问题设定、逐步推导、手算与实现](https://yingwen.io/zh/continual-rl/download/algorithm-tutorials.md)
- [研究问题地图](https://yingwen.io/zh/continual-rl/download/research-problems.md)
- [强化学习问题的形式化：交互、目标与持续学习](https://yingwen.io/zh/continual-rl/download/chapter-objectives.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/objectives_lab.py)
- [奖励假设与奖励设计](https://yingwen.io/zh/continual-rl/download/chapter-reward-design.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/reward_design_lab.py)
- [平均奖励：奖励率、差分价值与持续控制](https://yingwen.io/zh/continual-rl/download/chapter-average.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)
- [Agent state：部分可观测性、递归记忆与在线信用分配](https://yingwen.io/zh/continual-rl/download/chapter-state.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/state_meta_lab.py)
- [价值预测与时间差分学习](https://yingwen.io/zh/continual-rl/download/chapter-value.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)
- [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/download/chapter-gvf.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/gvf_lab.py)
- [持续控制：比较策略与学习智能体](https://yingwen.io/zh/continual-rl/download/chapter-control.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/continual_control_lab.py)
- [深度价值学习：DQN 与 Double DQN](https://yingwen.io/zh/continual-rl/download/chapter-deep-value.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)
- [策略梯度、Actor–Critic 与 PPO](https://yingwen.io/zh/continual-rl/download/chapter-policy.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)
- [最大熵控制与 Soft Actor–Critic](https://yingwen.io/zh/continual-rl/download/chapter-soft-control.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)
- [时间信用分配：从资格迹到深度梯度学习](https://yingwen.io/zh/continual-rl/download/chapter-credit.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/credit_assignment_lab.py)
- [流式强化学习：交互协议与更新稳定性](https://yingwen.io/zh/continual-rl/download/chapter-streaming.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)
- [学习规则的适应：在线元梯度与跨任务元学习](https://yingwen.io/zh/continual-rl/download/chapter-meta.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/state_meta_lab.py)
- [目标与子任务：条件控制、经验重用与技能设计](https://yingwen.io/zh/continual-rl/download/chapter-goals.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/knowledge_algorithms_lab.py)
- [Options：多步决策、技能发现与可复用行为](https://yingwen.io/zh/continual-rl/download/chapter-options.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/knowledge_algorithms_lab.py)
- [Dyna：模型学习与规划](https://yingwen.io/zh/continual-rl/download/chapter-dyna.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)
- [模型与后果预测：学什么，才能用于下一次决策？](https://yingwen.io/zh/continual-rl/download/chapter-models.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/knowledge_algorithms_lab.py)
- [规划：把模型中的经验转成更好的决策](https://yingwen.io/zh/continual-rl/download/chapter-planning.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/knowledge_algorithms_lab.py)
- [知识保留：经验重放、参数约束与模型记忆](https://yingwen.io/zh/continual-rl/download/chapter-retention.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)
- [可塑性：梯度通路、有效学习率与预测干扰](https://yingwen.io/zh/continual-rl/download/chapter-plasticity.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)
- [持续探索：新奇、不确定性、学习进展与恢复](https://yingwen.io/zh/continual-rl/download/chapter-exploration.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)
- [持续智能体架构：模块接口、更新调度与长期评价](https://yingwen.io/zh/continual-rl/download/chapter-architectures.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)
- [实验设计：从更新正确到持续学习证据](https://yingwen.io/zh/continual-rl/download/chapter-experiments.md) · [配套代码](https://yingwen.io/zh/continual-rl/download/experiment_design_lab.py)
- [补充机制片段与检查](https://yingwen.io/zh/continual-rl/download/mechanisms.py)
- [B2：平均奖励](https://yingwen.io/zh/continual-rl/download/average-reward.md)
- [C 与 E1：知识构建及架构](https://yingwen.io/zh/continual-rl/download/agent-construction.md)
- [直觉练习：八课教程](https://yingwen.io/zh/continual-rl/download/tutorial.md)
- [谱系参考：算法家族与交叉关系](https://yingwen.io/zh/continual-rl/download/algorithm-map.md)
- [作者实现导读与公式核对](https://yingwen.io/zh/continual-rl/download/implementations.md)
- [基础算法实验](https://yingwen.io/zh/continual-rl/download/rl_foundations.py)
- [知识构建接口实验](https://yingwen.io/zh/continual-rl/download/knowledge_lab.py)
- [独立公式检查](https://yingwen.io/zh/continual-rl/download/formula_checks.py)
- [实验协议](https://yingwen.io/zh/continual-rl/download/protocol.md)
- [GitHub 教程与代码](https://github.com/ying-wen/continual-rl-tutorial)
- [公开研究 Workbench](https://github.com/ying-wen/rl-research-workbench)

教学实验用于理解与诊断机制，不等同于论文完整复现。代码 MIT，原创教程 CC BY 4.0；外部材料按各自许可。
