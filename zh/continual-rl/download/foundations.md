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
