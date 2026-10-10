# 按教材查找逐步算例

先在正文完成手算，再运行对应的独立脚本。输入、运行命令和图中数值的解释见各节；完整训练与基线另在实验入口中提供。

## 第 I 册 · 经典强化学习

### 多臂老虎机：估计、探索与直接策略学习

- [7 · 源码与实验观察](https://yingwen.io/zh/continual-rl/foundations/tabular/bandits/#lesson-code) · [bandit_tracking_walkthrough.py](https://yingwen.io/crl-code/tutorials/bandit_tracking_walkthrough.py)
- [7 · 源码与实验观察](https://yingwen.io/zh/continual-rl/foundations/tabular/bandits/#lesson-code) · [bandit_exploration_walkthrough.py](https://yingwen.io/crl-code/tutorials/bandit_exploration_walkthrough.py)

### MDP、回报与价值：序列决策的数学对象

- [7 · 模型表示与实验观察](https://yingwen.io/zh/continual-rl/foundations/tabular/mdps/#lesson-code) · [mdp_dp_walkthrough.py](https://yingwen.io/crl-code/tutorials/mdp_dp_walkthrough.py)

### Monte Carlo：完整回报、探索控制与离策略评价

- [3.1 · 估计改变行动，行动产生下一条回报](https://yingwen.io/zh/continual-rl/foundations/tabular/monte-carlo/#mc-control-loop) · [mc-control-walkthrough.py](https://yingwen.io/crl-code/tutorials/mc-control-walkthrough.py)
- [5.6 · 从一条真实路径走过 Q、C、W 和下一版行为](https://yingwen.io/zh/continual-rl/foundations/tabular/monte-carlo/#mc-offpolicy-loop) · [mc-offpolicy-control-walkthrough.py](https://yingwen.io/crl-code/tutorials/mc-offpolicy-control-walkthrough.py)
- [8 · 实验观察与估计诊断](https://yingwen.io/zh/continual-rl/foundations/tabular/monte-carlo/#lesson-code) · [monte-carlo-walkthrough.py](https://yingwen.io/crl-code/tutorials/monte-carlo-walkthrough.py)

### TD 预测与控制：SARSA、Expected SARSA、Q-learning 和 Double Q

- [3 · 控制的三种下一动作](https://yingwen.io/zh/continual-rl/foundations/tabular/temporal-difference/#td-control-targets) · [control_targets_walkthrough.py](https://yingwen.io/crl-code/tutorials/control_targets_walkthrough.py)

### 多步学习：n-step、Tree Backup 与 Q(σ)

- [6 · 完整 n-step 执行顺序](https://yingwen.io/zh/continual-rl/foundations/tabular/multistep/#multistep-algorithm) · [control_targets_walkthrough.py](https://yingwen.io/crl-code/tutorials/control_targets_walkthrough.py)

### 学习与规划：Dyna、优先扫描和执行时搜索

- [7 · 可运行模型循环](https://yingwen.io/zh/continual-rl/foundations/tabular/planning/#lesson-code) · [planning-budget-walkthrough.py](https://yingwen.io/crl-code/tutorials/planning-budget-walkthrough.py)
- [7 · 可运行模型循环](https://yingwen.io/zh/continual-rl/foundations/tabular/planning/#lesson-code) · [stochastic-model-walkthrough.py](https://yingwen.io/crl-code/tutorials/stochastic-model-walkthrough.py)
- [7 · 可运行模型循环](https://yingwen.io/zh/continual-rl/foundations/tabular/planning/#lesson-code) · [dyna-planning-walkthrough.py](https://yingwen.io/crl-code/tutorials/dyna-planning-walkthrough.py)

### 函数逼近预测：从回归到 TD 固定点

- [共享几何 · 一次更新究竟帮助了谁？](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/#prediction-sharing) · [shared_gradient_walkthrough.py](https://yingwen.io/crl-code/tutorials/shared_gradient_walkthrough.py)

### 特征、泛化与半梯度控制

- [4.1 · 在走廊里执行更新，再看它选出的经验](https://yingwen.io/zh/continual-rl/foundations/approximation/features-control/#features-sarsa-corridor) · [feature-control-walkthrough.py](https://yingwen.io/crl-code/tutorials/feature-control-walkthrough.py)

### 持续控制与平均奖励

- [2.1 · 同一个服务站：决策次数与真实秒数](https://yingwen.io/zh/continual-rl/foundations/approximation/average-control/#average-rate-cycles) · [average_rate_walkthrough.py](https://yingwen.io/crl-code/tutorials/average_rate_walkthrough.py)

### 离策略函数逼近：覆盖、发散与稳定更新

- [8 · 两种独立计算检查同一例子](https://yingwen.io/zh/continual-rl/foundations/approximation/off-policy/#lesson-code) · [offpolicy-geometry-walkthrough.py](https://yingwen.io/crl-code/tutorials/offpolicy-geometry-walkthrough.py)

### 多步回报、资格迹与 True-online TD

- [6 · 两个独立实现逐前缀对照](https://yingwen.io/zh/continual-rl/foundations/approximation/traces/#lesson-code) · [shared_gradient_walkthrough.py](https://yingwen.io/crl-code/tutorials/shared_gradient_walkthrough.py)

### 策略梯度、基线与 Actor–Critic

- [6.1 · 同一个例子：占用怎样收集全部 score](https://yingwen.io/zh/continual-rl/foundations/approximation/policy-gradient/#policy-occupancy-example) · [policy-gradient-walkthrough.py](https://yingwen.io/crl-code/tutorials/policy-gradient-walkthrough.py)

## 第 II 册 · 深度强化学习

### 深度价值学习：DQN、Double DQN 与目标的时间顺序

- [2.2 · 固定标签，与让标签参与求导](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/#lesson-gradient-paths) · [dqn_shared_feature_walkthrough.py](https://yingwen.io/crl-code/tutorials/dqn_shared_feature_walkthrough.py)

### 策略梯度：从轨迹概率到 GAE 与 actor–critic

- [6 · 从数值核验到实际 on-policy 训练](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/#lesson-code) · [gae_ppo_walkthrough.py](https://yingwen.io/crl-code/tutorials/gae_ppo_walkthrough.py)

### 策略更新的尺度：TRPO 与 PPO

- [6 · 数值检验：TRPO 的步长与 PPO 的轨迹](https://yingwen.io/zh/continual-rl/foundations/deep/trust-region/#lesson-example) · [gae_ppo_walkthrough.py](https://yingwen.io/crl-code/tutorials/gae_ppo_walkthrough.py)

### 连续动作的价值优化：DDPG 与 TD3

- [3.1 · 预测上升，为什么真实回报反而下降？](https://yingwen.io/zh/continual-rl/foundations/deep/deterministic-control/#lesson-critic-error) · [continuous_control_walkthrough.py](https://yingwen.io/crl-code/tutorials/continuous_control_walkthrough.py)

### 最大熵连续控制：SAC 的价值、密度与温度

- [6 · soft target 与密度手算](https://yingwen.io/zh/continual-rl/foundations/deep/entropy-control/#lesson-example) · [continuous_control_walkthrough.py](https://yingwen.io/crl-code/tutorials/continuous_control_walkthrough.py)

### 不完全可观测：信念状态、信息行动与递归记忆

- [7 · 可运行 filter 与信息价值](https://yingwen.io/zh/continual-rl/foundations/deep/partial-observability/#lesson-code) · [belief-planning-walkthrough.py](https://yingwen.io/crl-code/tutorials/belief-planning-walkthrough.py)
- [7 · 可运行 filter 与信息价值](https://yingwen.io/zh/continual-rl/foundations/deep/partial-observability/#lesson-code) · [recurrent-replay-walkthrough.py](https://yingwen.io/crl-code/tutorials/recurrent-replay-walkthrough.py)

### 探索与不确定性：后验、乐观估计和时间一致行动

- [6.1 · 把远端信息接到一个未知奖励模型](https://yingwen.io/zh/continual-rl/foundations/deep/exploration/#deep-exploration-task) · [deep-exploration-walkthrough.py](https://yingwen.io/crl-code/tutorials/deep-exploration-walkthrough.py)

### 分布强化学习：Bellman 分布、分位数与风险目标

- [1 · 同一均值可以来自不同的回报分布](https://yingwen.io/zh/continual-rl/foundations/deep/distributional/#lesson-setting) · [distributional-walkthrough.py](https://yingwen.io/crl-code/tutorials/distributional-walkthrough.py)
- [4 · 分位数回归与均值决策](https://yingwen.io/zh/continual-rl/foundations/deep/distributional/#lesson-quantiles) · [quantile-walkthrough.py](https://yingwen.io/crl-code/tutorials/quantile-walkthrough.py)
- [共享网络：一个输入的下降怎样影响另一个输入](https://yingwen.io/zh/continual-rl/foundations/deep/distributional/#lesson-neural-shared) · [neural-distribution-walkthrough.py](https://yingwen.io/crl-code/tutorials/neural-distribution-walkthrough.py)

### 离线强化学习：数据支持、策略评估与保守改进

- [8 · 估计器与目标函数的可执行核](https://yingwen.io/zh/continual-rl/foundations/deep/offline/#lesson-code) · [offline_support_walkthrough.py](https://yingwen.io/crl-code/tutorials/offline_support_walkthrough.py)

### 模型学习与规划：MPC、短模型 rollout 和潜在想象

- [8 · 学模型、枚举规划与想象回报核](https://yingwen.io/zh/continual-rl/foundations/deep/model-based/#lesson-code) · [belief-planning-walkthrough.py](https://yingwen.io/crl-code/tutorials/belief-planning-walkthrough.py)

### 约束强化学习：占据测度、拉格朗日与可行策略

- [8 · 占据流、解析混合与 primal–dual 核](https://yingwen.io/zh/continual-rl/foundations/deep/constraints/#lesson-code) · [constraint-control-walkthrough.py](https://yingwen.io/crl-code/tutorials/constraint-control-walkthrough.py)

### 自对弈与开放式多智能体学习：评估、目标与策略种群

- [8 · 算例与原始工程：从标签到种群评价](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent-populations/#lesson-code) · [selfplay_search_labels.py](https://yingwen.io/crl-code/tutorials/selfplay_search_labels.py)

## 第 III 册 · 持续强化学习

### 强化学习问题的形式化：交互、目标与持续学习

- [9.2 奖励带噪声时，哪一个学习器更好？](https://yingwen.io/zh/continual-rl/foundations/objectives/#lesson-noisy-lifetime) · [lifetime_evaluation_walkthrough.py](https://yingwen.io/crl-code/tutorials/lifetime_evaluation_walkthrough.py)

### 奖励假设与奖励设计

- [13. 运行、阅读原始代码与选择基准](https://yingwen.io/zh/continual-rl/foundations/reward-design/#lesson-code) · [reward-design-walkthrough.py](https://yingwen.io/crl-code/tutorials/reward-design-walkthrough.py)

### 平均奖励：奖励率、差分价值与持续控制

- [7.1 · 贯穿算例：让预测与控制共用一个时钟](https://yingwen.io/zh/continual-rl/algorithms/average-reward/#average-rate-task) · [average_rate_walkthrough.py](https://yingwen.io/crl-code/tutorials/average_rate_walkthrough.py)

### Agent state：部分可观测性、递归记忆与在线信用分配

- [5.1 四步延迟提示：让保留系数也能学习](https://yingwen.io/zh/continual-rl/construction/state/#lesson-delayed-recurrent) · [recurrent_state_walkthrough.py](https://yingwen.io/crl-code/tutorials/recurrent_state_walkthrough.py)
- [5.2 中途更新参数：活动、trace 与求导对象怎样改变](https://yingwen.io/zh/continual-rl/construction/state/#lesson-online-versions) · [state-online-walkthrough.py](https://yingwen.io/crl-code/tutorials/state-online-walkthrough.py)
- [5.3 学到的递归系数怎样改变行动与下一条经验](https://yingwen.io/zh/continual-rl/construction/state/#lesson-recurrent-control) · [recurrent-control-walkthrough.py](https://yingwen.io/crl-code/tutorials/recurrent-control-walkthrough.py)
- [5.4 门后继续走：无重置活动、在线变参与后续经验](https://yingwen.io/zh/continual-rl/construction/state/#lesson-continuing-control) · [continuing-state-walkthrough.py](https://yingwen.io/crl-code/tutorials/continuing-state-walkthrough.py)
- [5.5 新模式与长走廊：保存信息和保存梯度各花什么资源](https://yingwen.io/zh/continual-rl/construction/state/#lesson-delayed-memory-budget) · [delayed-state-walkthrough.py](https://yingwen.io/crl-code/tutorials/delayed-state-walkthrough.py)
- [5.6 让写入门学习：旧提示该留下多少](https://yingwen.io/zh/continual-rl/construction/state/#lesson-learned-memory) · [learned-memory-walkthrough.py](https://yingwen.io/crl-code/tutorials/learned-memory-walkthrough.py)

### 时间信用分配：从资格迹到深度梯度学习

- [从两步到三步：同一经验流上的预测版本](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/#lesson-prefix-walkthrough) · [credit_meta_walkthrough.py](https://yingwen.io/crl-code/tutorials/credit_meta_walkthrough.py)

### 流式强化学习：交互协议与更新稳定性

- [9 · 机制实验与作者实现](https://yingwen.io/zh/continual-rl/algorithms/streaming/#lesson-code) · [streaming_state_walkthrough.py](https://yingwen.io/crl-code/tutorials/streaming_state_walkthrough.py)

### 学习规则的适应：在线元梯度与跨任务元学习

- [沿同一条经验流，区分资格与步长敏感度](https://yingwen.io/zh/continual-rl/algorithms/meta/#lesson-trace-sensitivity) · [credit_meta_walkthrough.py](https://yingwen.io/crl-code/tutorials/credit_meta_walkthrough.py)

### 知识保留：经验重放、参数约束与模型记忆

- [同一反转问题：忘得更多，也可能学得更快](https://yingwen.io/zh/continual-rl/algorithms/retention/#lesson-shared-switch) · [retention_plasticity_walkthrough.py](https://yingwen.io/crl-code/tutorials/retention_plasticity_walkthrough.py)

### 可塑性：梯度通路、有效学习率与预测干扰

- [同一反转问题：对称轨道、等函数干预与重新学习](https://yingwen.io/zh/continual-rl/algorithms/plasticity/#lesson-paired-probe) · [retention_plasticity_walkthrough.py](https://yingwen.io/crl-code/tutorials/retention_plasticity_walkthrough.py)

### 目标与子任务：条件控制、经验重用与技能设计

- [1.1 同一条运输记录，换目标时究竟改什么](https://yingwen.io/zh/continual-rl/construction/goals/#lesson-shared-world) · [goal_model_planning_walkthrough.py](https://yingwen.io/crl-code/tutorials/goal_model_planning_walkthrough.py)
- [3.2 候选变成行为，再让新经验改变候选与规划](https://yingwen.io/zh/continual-rl/construction/goals/#lesson-discovery-loop) · [goal_discovery_walkthrough.py](https://yingwen.io/crl-code/tutorials/goal_discovery_walkthrough.py)

### Options：多步决策、技能发现与可复用行为

- [动手：从一个固定技能走到模型规划](https://yingwen.io/zh/continual-rl/construction/options/#lesson-option-practice) · [option_model_walkthrough.py](https://yingwen.io/crl-code/tutorials/option_model_walkthrough.py)

### 模型与后果预测：学什么，才能用于下一次决策？

- [1.1 从五条记录到模型，再看关闸后的错误](https://yingwen.io/zh/continual-rl/construction/models/#lesson-model-repair) · [goal_model_planning_walkthrough.py](https://yingwen.io/crl-code/tutorials/goal_model_planning_walkthrough.py)
- [1.4 把复查代价放回行动循环](https://yingwen.io/zh/continual-rl/construction/models/#lesson-model-recheck) · [model-change-walkthrough.py](https://yingwen.io/crl-code/tutorials/model-change-walkthrough.py)

### 规划：把模型中的经验转成更好的决策

- [1.1 模型已经修正，起点何时改选路线](https://yingwen.io/zh/continual-rl/construction/planning/#lesson-replanning) · [goal_model_planning_walkthrough.py](https://yingwen.io/crl-code/tutorials/goal_model_planning_walkthrough.py)

### 持续探索：新奇、不确定性、学习进展与恢复

- [9 · 奖励时序实现与探索诊断](https://yingwen.io/zh/continual-rl/algorithms/exploration/#lesson-code) · [exploration-information-walkthrough.py](https://yingwen.io/crl-code/tutorials/exploration-information-walkthrough.py)

### 持续智能体架构：模块接口、更新调度与长期评价

- [3 · 真实交互与规划的更新调度](https://yingwen.io/zh/continual-rl/construction/architectures/#lesson-schedule) · [architecture-walkthrough.py](https://yingwen.io/crl-code/tutorials/architecture-walkthrough.py)

## 心理与神经科学等视角：学习、心智与意识

### 有限理性与信息成本：学习、思考和行动如何共同取舍

- [3 · 描述偏好，不等于规定智能体应有的目标](https://yingwen.io/zh/continual-rl/supplements/economics/#reference-dependence) · [economic-choice-walkthrough.py](https://yingwen.io/crl-code/tutorials/economic-choice-walkthrough.py)

### 模型怎样解释学习：从实验到科学理论

- [4.3 恢复参数之后，还要比较更新规则](https://yingwen.io/zh/continual-rl/supplements/explanation/#explanation-identifiability-recovery) · [explanation-identifiability-walkthrough.py](https://yingwen.io/crl-code/tutorials/explanation-identifiability-walkthrough.py)
