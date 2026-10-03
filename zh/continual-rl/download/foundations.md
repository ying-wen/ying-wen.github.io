# 强化学习基础分册

这些分册按教学先修关系组织，不把经典 RL、深度 RL 与 CRL 当作互斥的问题类。表格与函数逼近描述表示和更新方式；深度方法使用神经网络；持续学习关注完整学习器在长期交互中的表现。它们可以同时用于同一智能体。

运行时，智能体与世界交互；设计者在外层选择奖励、初始化、数据权限和预算。先用[领域总览](https://yingwen.io/zh/continual-rl/overview/)定位问题，再按需补基础。目标问题见[奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/)，算法比较见[持续控制](https://yingwen.io/zh/continual-rl/algorithms/control/)。

基础共 26 章。表格方法与 Part II 提供预测、控制、函数逼近、信用分配和策略梯度工具。深度分册先给核心算法，再并列展开研究分支；无需学完全部分支才进入 CRL 研究。

## 表格强化学习

没有共享参数时，怎样从经验评价策略、改善行为并进行规划？

对应：Sutton & Barto · Part I，第 2–8 章

### 建议的基础阅读顺序

按先修关系组织。已有基础的读者可用各章练习定位缺口，不必逐章重读。

#### [多臂老虎机：估计、探索与直接策略学习](https://yingwen.io/zh/continual-rl/foundations/tabular/bandits/)

没有状态转移时，仍需一边估计动作收益，一边决定下一次尝试什么。这个最小问题把估计误差、探索代价和策略更新分开。

- 推导样本平均和常数步长，解释历史奖励的权重。
- 理解 ε-greedy、UCB 和 gradient bandit 的不同学习对象。
- 运行随机奖励采样，区分实现正确性与有限样本性能。

#### [MDP、回报与价值：序列决策的数学对象](https://yingwen.io/zh/continual-rl/foundations/tabular/mdps/)

动作会改变后续状态时，需要评价整个未来。本章从随机交互过程推导价值与 Bellman 方程，明确后续算法共同使用的数学对象。

- 区分状态、观测、策略和环境模型。
- 从回报推导状态价值、动作价值与 Bellman 方程。
- 解释 Markov、平稳、终止和截断条件。

#### [动态规划：评价、改善与最优递推](https://yingwen.io/zh/continual-rl/foundations/tabular/dynamic-programming/)

已知环境模型时，怎样通过局部计算得到长期价值和策略？本章将 Bellman 方程转化为迭代，并证明评价与改善之间的联系。

- 推导迭代评价、收缩与残差误差界。
- 证明策略改善，区分策略迭代、价值迭代和 GPI。
- 对照解析求解，理解计算预算与模型误差。

#### [Monte Carlo：完整经历、重复访问与离策略评价](https://yingwen.io/zh/continual-rl/foundations/tabular/monte-carlo/)

不知道模型时，可以将完整回报作为样本。需要规定重复访问如何计数、数据由谁生成，以及目标策略能否被行为覆盖。

- 实现 first-visit、every-visit 和 ε-soft MC 控制。
- 推导轨迹重要性比率、ordinary IS 与 weighted IS。
- 识别终止、覆盖、方差和策略变化的边界。

#### [TD 预测与控制：SARSA、Expected SARSA、Q-learning 和 Double Q](https://yingwen.io/zh/continual-rl/foundations/tabular/temporal-difference/)

如何在完整回报尚不可用时学习？TD 用下一预测补足未来；控制算法再根据不同的下一动作处理方式，形成不同的学习目标。

- 从 Bellman 样本推导 TD(0)。
- 区分四种控制 target、行为策略和更新顺序。
- 理解 maximization bias、Double 选择评价分离及表格收敛条件。

#### [多步学习：n-step、Tree Backup 与 Q(σ)](https://yingwen.io/zh/continual-rl/foundations/tabular/multistep/)

学习目标可以在一步 bootstrap 与完整回报之间选择，也可以在动作采样与动作期望之间选择。这是两条不同的设计维度。

- 推导 n-step 回报及延迟更新时序。
- 理解 Tree Backup 的已采样分支和未采样分支。
- 从两端推导 on-policy Q(σ)，检查终止与退化条件。

#### [学习与规划：Dyna、优先扫描和执行时搜索](https://yingwen.io/zh/continual-rl/foundations/tabular/planning/)

真实经验既能直接改进价值，也能训练后果模型。规划使用这个模型继续计算，关键是模型语义、backup 的成本以及计算应分配到哪里。

- 实现 Dyna 的真实交互、模型更新与模拟更新循环。
- 推导优先扫描的前驱传播，区别 sampling 与 expectation backup。
- 理解 rollout、MCTS 与持续环境中模型错误的边界。

## 函数逼近与经典进阶方法

状态不能逐项保存时，哪些更新仍成立，哪些保证需要重新检查？

对应：Sutton & Barto · Part II，第 9–13 章

### 建议的基础阅读顺序

按先修关系组织。已有基础的读者可用各章练习定位缺口，不必逐章重读。

#### [函数逼近预测：从回归到 TD 固定点](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/)

共享少量参数以后，MC、TD 和最小二乘方法究竟在求解什么？

- 从平方价值误差推导梯度 MC，说明采样分布的作用。
- 推导线性 TD 的平均更新、投影 Bellman 方程与 LSTD。
- 用同一个二状态例子计算不同解，并检查半梯度的含义。

#### [特征、泛化与半梯度控制](https://yingwen.io/zh/continual-rl/foundations/approximation/features-control/)

特征怎样改变学习行为，Sarsa 又怎样在共享参数下改善策略？

- 比较局部 tile coding 与全局 Fourier 特征的泛化。
- 推导动作分块表示和半梯度 Sarsa，说明动作采样与更新顺序。
- 手算特征缩放的步长效应，并辨别预测保证与控制保证。

#### [持续控制与平均奖励](https://yingwen.io/zh/continual-rl/foundations/approximation/average-control/)

智能体没有自然回合终点时，怎样定义和学习长期控制目标？

- 区分折扣价值、平均奖励率与差分价值。
- 从 Poisson 方程得到差分 TD 和 Sarsa 的两组同步更新。
- 说明 unichain、communicating 与函数逼近保证的边界。

#### [离策略函数逼近：覆盖、发散与稳定更新](https://yingwen.io/zh/continual-rl/foundations/approximation/off-policy/)

行为数据足够覆盖目标策略，为什么 TD 仍可能发散，又能怎样修复？

- 区分动作重要性修正与状态加权带来的稳定性。
- 推导 MSPBE 及 GTD2、TDC 的辅助权重更新。
- 理解 emphatic weighting 的目的、时间索引和线性理论条件。

#### [多步回报、资格迹与 True-online TD](https://yingwen.io/zh/continual-rl/foundations/approximation/traces/)

当前到来的奖励怎样更新过去的预测，同时保留正确的在线更新语义？

- 把 n-step 与 lambda-return 的前向目标推导成后向信用传播。
- 解释普通累积迹与 true-online 的差别，推导 Dutch trace。
- 用重复状态和独立前向实现检查每个轨迹前缀。

#### [策略梯度、基线与 Actor–Critic](https://yingwen.io/zh/continual-rl/foundations/approximation/policy-gradient/)

直接学习策略时，哪一个目标的梯度能由经验估计，近似从哪里进入？

- 从轨迹概率推导带完整折扣权重的 REINFORCE。
- 证明动作无关基线不改变梯度，并定位 critic 引入的偏差。
- 区分普通梯度、自然梯度以及平均奖励目标的采样权重。

## 现代深度强化学习

怎样组织神经智能体的核心更新，再按信息、目标与数据条件选择研究分支？

对应：Spinning Up · 核心方法与原始研究扩展

### 核心算法与训练循环

先掌握价值学习、策略梯度、批内策略更新、连续控制和训练检查。它们是工具基础，不要求在所有研究问题中同时使用。

#### [深度价值学习：DQN、Double DQN 与目标的时间顺序](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/)

把表格 Q-learning 换成网络后，损失、数据与目标为什么都需要重新组织？

- 写出 DQN 与 Double DQN 的不同目标。
- 区分环境终止、采样截断和目标网络更新。
- 通过 autograd 检查停止梯度和张量形状。
- 运行含真实交互、replay、优化与评估的 CPU 小实验。

#### [策略梯度：从轨迹概率到 GAE 与 actor–critic](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/)

不对环境求导，怎样从采样动作计算策略梯度？价值网络和优势各自做什么？

- 推导 score-function 梯度与 baseline 消去。
- 区分真实目标、优势估计和实现 surrogate。
- 给 GAE 设置独立的 bootstrap 与跨序列 mask。
- 正确使用 log_prob、detach 和 actor/critic loss。

#### [策略更新的尺度：TRPO 与 PPO](https://yingwen.io/zh/continual-rl/foundations/deep/trust-region/)

同一批数据可以重复使用多少次？为什么 clipping 不是一个性能保证？

- 由性能差分恒等式解释 surrogate 的来源。
- 求解局部 KL 约束，理解 Fisher、共轭梯度与回溯。
- 按优势符号解释 PPO，并固定旧策略、优势与 critic target。

#### [连续动作的价值优化：DDPG 与 TD3](https://yingwen.io/zh/continual-rl/foundations/deep/deterministic-control/)

不能枚举连续动作时，如何用 critic 的梯度改进 actor？

- 区分行为噪声、目标动作噪声与目标网络。
- 推导确定性 actor 的链式梯度。
- 写出 TD3 双 critic、延迟更新与平滑目标的完整次序。

#### [最大熵连续控制：SAC 的价值、密度与温度](https://yingwen.io/zh/continual-rl/foundations/deep/entropy-control/)

随机 actor 不只是加噪声：熵如何进入 Bellman 方程与自动微分？

- 推导 soft Bellman 与 actor 的 KL 投影。
- 计算 tanh 与动作缩放后的概率密度。
- 区分 actor、critic、温度三类梯度与停止梯度位置。

#### [深度 RL 的机制接口：模型、记忆、离线数据与实验](https://yingwen.io/zh/continual-rl/foundations/deep/practice/)

改变数据来源或 agent state 后，哪些推导和实现条件必须重新检查？

- 由 Bellman 算子解释模型误差的放大。
- 区分 recurrent hidden state 与跨时间参数梯度。
- 识别离线外推问题，并设计可解释的 CRL 对照。

### 并列研究分支

根据研究问题选择：这些方向改变信息、数据、模型或目标条件，可以交叉组合，不是必须依次完成的后续关卡。

#### [不完全可观测：信念状态、信息行动与递归记忆](https://yingwen.io/zh/continual-rl/foundations/deep/partial-observability/)

当前观察不能决定未来时，智能体应记住什么，信息又如何影响行动？

- 从历史条件分布推导 Bayes filter。
- 把信息获取的价值纳入 Bellman 决策。
- 区分精确信念、学习的 recurrent state 和训练时的隐状态权限。

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

#### [多智能体：博弈、局部信息与集中训练](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent/)

其他参与者也在学习时，“最优策略”和“环境变化”应怎样定义？

- 区分共同回报、零和与一般和目标。
- 推导对手策略变化产生的有效转移变化。
- 解释 centralized critic、反事实 baseline 与单调价值分解。