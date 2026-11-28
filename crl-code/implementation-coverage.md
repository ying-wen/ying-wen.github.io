# 正文算法实现覆盖盘点

先选择教材问题，再确认实现范围。独立教学算法与独立子机制实验都提供单文件更新、运行入口、对照与记录。子机制实验只回答限定的问题，不能代替原论文完整系统。

机器清单：`integrations/algorithm_coverage.json`。本表不认证训练完成；实际结果须核对对应实验的全部种子记录、配置与源码哈希。

## 范围与阅读方式

- **独立教学算法**：在明确的小型问题中实现算法更新与学习循环。
- **独立子机制实验**：运行一种估计器、更新机制或受限特例，保留未实现的部分。
- **公式核**：只有综合实验中的公式函数，尚未形成独立实验。
- **作者工程**：保留作者源码入口；不表示已完成论文规模训练。
- **尚未接入**：教材已有讲解，但未提供可运行入口。

计数包含算法变体和基线，不是互斥的研究方法数。

## 独立教学算法与基线（小型问题，不是论文规模复现）（88 项）

### 固定步长 ε-greedy

- 源码：[implementations/classic/bandit_constant_step.py](../implementations/classic/bandit_constant_step.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/bandit_constant_step/)。
- 任务与对照：`stationary_bernoulli_bandit`；`bandit_sample_average`。
- 指标与预算：累计平均奖励；`environment_steps`。
- 设定：三臂平稳 Bernoulli 赌博机，均值为 0.2、0.5、0.8；记录实际采样奖励。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/exploration](https://yingwen.io/zh/continual-rl/algorithms/exploration/) · [algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### 梯度赌博机

- 源码：[implementations/classic/bandit_gradient.py](../implementations/classic/bandit_gradient.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/bandit_gradient/)。
- 任务与对照：`stationary_bernoulli_bandit`；`bandit_sample_average`。
- 指标与预算：累计平均奖励；`environment_steps`。
- 设定：三臂平稳 Bernoulli 赌博机，均值为 0.2、0.5、0.8；记录实际采样奖励。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/exploration](https://yingwen.io/zh/continual-rl/algorithms/exploration/) · [algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### 样本均值 ε-greedy

- 源码：[implementations/classic/bandit_sample_average.py](../implementations/classic/bandit_sample_average.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/bandit_sample_average/)。
- 任务与对照：`stationary_bernoulli_bandit`；`bandit_constant_step`。
- 指标与预算：累计平均奖励；`environment_steps`。
- 设定：三臂平稳 Bernoulli 赌博机，均值为 0.2、0.5、0.8；记录实际采样奖励。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/exploration](https://yingwen.io/zh/continual-rl/algorithms/exploration/) · [algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### 置信上界 UCB

- 源码：[implementations/classic/bandit_ucb.py](../implementations/classic/bandit_ucb.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/bandit_ucb/)。
- 任务与对照：`stationary_bernoulli_bandit`；`bandit_sample_average`。
- 指标与预算：累计平均奖励；`environment_steps`。
- 设定：三臂平稳 Bernoulli 赌博机，均值为 0.2、0.5、0.8；记录实际采样奖励。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/exploration](https://yingwen.io/zh/continual-rl/algorithms/exploration/) · [algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### Double Q-learning

- 源码：[implementations/classic/double_q.py](../implementations/classic/double_q.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/double_q/)。
- 任务与对照：`chain_control`；`q_learning`。
- 指标与预算：确定性贪心策略折扣回报；`environment_steps`。
- 设定：六格链在线控制：从0开始，右端终止奖励1，其余奖励-0.01，γ=0.95，ε=0.1。训练与测量分开，测量是冻结当前Q的贪心策略回报。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### Dyna-Q

- 源码：[implementations/classic/dyna_q.py](../implementations/classic/dyna_q.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/dyna_q/)。
- 任务与对照：`chain_control`；`q_learning`。
- 指标与预算：确定性贪心策略折扣回报；`environment_steps`。
- 设定：六格链在线控制：起点0，终点奖励1，其余-0.01，γ=0.95。环境预算与Q-learning一致，额外模型备份单独记录；不能据此声称相同计算成本。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/dyna](https://yingwen.io/zh/continual-rl/algorithms/dyna/) · [construction/planning](https://yingwen.io/zh/continual-rl/construction/planning/) · [construction/models](https://yingwen.io/zh/continual-rl/construction/models/)

### Emphatic TD(0)

- 源码：[implementations/classic/emphatic_td.py](../implementations/classic/emphatic_td.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/emphatic_td/)。
- 任务与对照：`random_walk_linear_prediction`；`linear_td`。
- 指标与预算：五状态价值 RMSE；`environment_steps`。
- 设定：在线随机游走预测，从状态3开始；γ=1，特征[1,s/6]，冻结测量五状态RMSE。全部使用同策略数据ρ=1，因此此实验不检验离策略稳定性。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [algorithms/streaming](https://yingwen.io/zh/continual-rl/algorithms/streaming/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### Expected SARSA

- 源码：[implementations/classic/expected_sarsa.py](../implementations/classic/expected_sarsa.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/expected_sarsa/)。
- 任务与对照：`chain_control`；`q_learning`。
- 指标与预算：确定性贪心策略折扣回报；`environment_steps`。
- 设定：六格链在线控制：从0开始，右端终止奖励1，其余奖励-0.01，γ=0.95，ε=0.1。训练与测量分开，测量是冻结当前Q的贪心策略回报。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### GTD2

- 源码：[implementations/classic/gtd2.py](../implementations/classic/gtd2.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/gtd2/)。
- 任务与对照：`random_walk_linear_prediction`；`linear_td`。
- 指标与预算：五状态价值 RMSE；`environment_steps`。
- 设定：在线随机游走预测，从状态3开始；γ=1，特征[1,s/6]，冻结测量五状态RMSE。全部使用同策略数据ρ=1，因此此实验不检验离策略稳定性。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [algorithms/streaming](https://yingwen.io/zh/continual-rl/algorithms/streaming/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### 线性半梯度 TD

- 源码：[implementations/classic/linear_td.py](../implementations/classic/linear_td.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/linear_td/)。
- 任务与对照：`random_walk_linear_prediction`；`gtd2`。
- 指标与预算：五状态价值 RMSE；`environment_steps`。
- 设定：在线随机游走预测，从状态3开始；γ=1，特征[1,s/6]，冻结测量五状态RMSE。全部使用同策略数据ρ=1，因此此实验不检验离策略稳定性。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [algorithms/streaming](https://yingwen.io/zh/continual-rl/algorithms/streaming/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### 首次访问蒙特卡洛预测

- 源码：[implementations/classic/mc_prediction.py](../implementations/classic/mc_prediction.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/mc_prediction/)。
- 任务与对照：`random_walk_prediction`；`td0`。
- 指标与预算：五状态价值 RMSE；`environment_steps`。
- 设定：五个非终止状态的无偏随机游走；左端奖励 0、右端奖励 1，γ=1。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### 三步 SARSA

- 源码：[implementations/classic/nstep_sarsa.py](../implementations/classic/nstep_sarsa.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/nstep_sarsa/)。
- 任务与对照：`chain_control`；`q_learning`。
- 指标与预算：确定性贪心策略折扣回报；`environment_steps`。
- 设定：六格链在线控制：从0开始，右端终止奖励1，其余奖励-0.01，γ=0.95，ε=0.1。训练与测量分开，测量是冻结当前Q的贪心策略回报。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### 三步 TD 预测

- 源码：[implementations/classic/nstep_td.py](../implementations/classic/nstep_td.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/nstep_td/)。
- 任务与对照：`random_walk_prediction`；`td0`。
- 指标与预算：五状态价值 RMSE；`environment_steps`。
- 设定：五个非终止状态的无偏随机游走；左端奖励 0、右端奖励 1，γ=1。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### 策略迭代

- 源码：[implementations/classic/policy_iteration.py](../implementations/classic/policy_iteration.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/policy_iteration/)。
- 任务与对照：`chain_exact_planning`；`value_iteration`。
- 指标与预算：最优价值最大绝对误差；`model_sweeps`。
- 设定：已知六格链模型，γ=0.95；横轴是同步扫描次数，绝非环境交互。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [construction/planning](https://yingwen.io/zh/continual-rl/construction/planning/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### 优先扫描

- 源码：[implementations/classic/prioritized_sweeping.py](../implementations/classic/prioritized_sweeping.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/prioritized_sweeping/)。
- 任务与对照：`chain_control`；`q_learning`。
- 指标与预算：确定性贪心策略折扣回报；`environment_steps`。
- 设定：六格链在线控制：起点0，终点奖励1，其余-0.01，γ=0.95。环境预算与Q-learning一致，额外模型备份单独记录；不能据此声称相同计算成本。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/dyna](https://yingwen.io/zh/continual-rl/algorithms/dyna/) · [construction/planning](https://yingwen.io/zh/continual-rl/construction/planning/) · [construction/models](https://yingwen.io/zh/continual-rl/construction/models/)

### Q-learning

- 源码：[implementations/classic/q_learning.py](../implementations/classic/q_learning.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/q_learning/)。
- 任务与对照：`chain_control`；`sarsa`。
- 指标与预算：确定性贪心策略折扣回报；`environment_steps`。
- 设定：六格链在线控制：从0开始，右端终止奖励1，其余奖励-0.01，γ=0.95，ε=0.1。训练与测量分开，测量是冻结当前Q的贪心策略回报。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### SARSA

- 源码：[implementations/classic/sarsa.py](../implementations/classic/sarsa.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/sarsa/)。
- 任务与对照：`chain_control`；`q_learning`。
- 指标与预算：确定性贪心策略折扣回报；`environment_steps`。
- 设定：六格链在线控制：从0开始，右端终止奖励1，其余奖励-0.01，γ=0.95，ε=0.1。训练与测量分开，测量是冻结当前Q的贪心策略回报。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### SARSA(λ)

- 源码：[implementations/classic/sarsa_lambda.py](../implementations/classic/sarsa_lambda.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/sarsa_lambda/)。
- 任务与对照：`chain_control`；`q_learning`。
- 指标与预算：确定性贪心策略折扣回报；`environment_steps`。
- 设定：六格链在线控制：从0开始，右端终止奖励1，其余奖励-0.01，γ=0.95，ε=0.1。训练与测量分开，测量是冻结当前Q的贪心策略回报。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### TD(0)

- 源码：[implementations/classic/td0.py](../implementations/classic/td0.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/td0/)。
- 任务与对照：`random_walk_prediction`；`mc_prediction`。
- 指标与预算：五状态价值 RMSE；`environment_steps`。
- 设定：五个非终止状态的无偏随机游走；左端奖励 0、右端奖励 1，γ=1。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### 累积迹 TD(λ)

- 源码：[implementations/classic/td_lambda.py](../implementations/classic/td_lambda.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/td_lambda/)。
- 任务与对照：`random_walk_prediction`；`td0`。
- 指标与预算：五状态价值 RMSE；`environment_steps`。
- 设定：五个非终止状态的无偏随机游走；左端奖励 0、右端奖励 1，γ=1。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### TDC

- 源码：[implementations/classic/tdc.py](../implementations/classic/tdc.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/tdc/)。
- 任务与对照：`random_walk_linear_prediction`；`linear_td`。
- 指标与预算：五状态价值 RMSE；`environment_steps`。
- 设定：在线随机游走预测，从状态3开始；γ=1，特征[1,s/6]，冻结测量五状态RMSE。全部使用同策略数据ρ=1，因此此实验不检验离策略稳定性。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [algorithms/streaming](https://yingwen.io/zh/continual-rl/algorithms/streaming/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### True-online TD(λ)

- 源码：[implementations/classic/true_online_td.py](../implementations/classic/true_online_td.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/true_online_td/)。
- 任务与对照：`random_walk_prediction`；`td0`。
- 指标与预算：五状态价值 RMSE；`environment_steps`。
- 设定：五个非终止状态的无偏随机游走；左端奖励 0、右端奖励 1，γ=1。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### 价值迭代

- 源码：[implementations/classic/value_iteration.py](../implementations/classic/value_iteration.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/value_iteration/)。
- 任务与对照：`chain_exact_planning`；`policy_iteration`。
- 指标与预算：最优价值最大绝对误差；`model_sweeps`。
- 设定：已知六格链模型，γ=0.95；横轴是同步扫描次数，绝非环境交互。
- 范围：小型教学任务；曲线不构成一般性能或持续学习优势的证据。
- 教材：[algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [construction/planning](https://yingwen.io/zh/continual-rl/construction/planning/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### 链任务 Q-learning

- 源码：[implementations/continual/chain_q.py](../implementations/continual/chain_q.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/chain_q/)。
- 任务与对照：`potential_chain_control`；`potential_shaping`。
- 指标与预算：原始奖励贪心策略折扣回报；`environment_steps`。
- 设定：六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1，Q=0。势函数固定、在线更新后冻结贪心策略，以原始环境奖励评价。任务全程平稳，无中点切换；塑形只改变训练奖励。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[foundations/reward-design](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### 固定步长 LMS

- 源码：[implementations/continual/constant_step_lms.py](../implementations/continual/constant_step_lms.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/constant_step_lms/)。
- 任务与对照：`adaptive_regression`；`idbd`。
- 指标与预算：当前任务无噪声预测 MSE；`environment_steps`。
- 设定：流式两维线性预测：x=[1,U(-1,1)]，中点真权重从[0.2,0.7]变为[-0.2,-0.7]，观测噪声σ=0.03，初始w=0。每次更新后冻结测量当前真任务解析MSE。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[algorithms/meta](https://yingwen.io/zh/continual-rl/algorithms/meta/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [algorithms/streaming](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

### 计数探索奖励

- 源码：[implementations/continual/count_bonus.py](../implementations/continual/count_bonus.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/count_bonus/)。
- 任务与对照：`count_exploration_chain`；`plain_q_exploration`。
- 指标与预算：原始奖励贪心策略折扣回报；`environment_steps`。
- 设定：平稳六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1，Q=0。状态动作计数从0开始，每次访问先加1，再计算0.2/√N探索奖励；对照系数0。冻结测量原始奖励贪心回报；计数新奇不等于信息增益。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[algorithms/exploration](https://yingwen.io/zh/continual-rl/algorithms/exploration/) · [foundations/reward-design](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/)

### 差分 Q-learning

- 源码：[implementations/continual/differential_q.py](../implementations/continual/differential_q.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/differential_q/)。
- 任务与对照：`continuing_bandit_control`；`differential_sarsa`。
- 指标与预算：贪心动作真实平均奖励；`environment_steps`。
- 设定：单状态无终点两动作继续任务，中点均值[0.2,0.8]交换；Bernoulli奖励。Q=0、奖励率=0，ε=0.1。冻结测量当前贪心动作真实平均奖励，训练不使用真均值。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/) · [algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### 差分 SARSA

- 源码：[implementations/continual/differential_sarsa.py](../implementations/continual/differential_sarsa.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/differential_sarsa/)。
- 任务与对照：`continuing_bandit_control`；`differential_q`。
- 指标与预算：贪心动作真实平均奖励；`environment_steps`。
- 设定：单状态无终点两动作继续任务，中点均值[0.2,0.8]交换；Bernoulli奖励。Q=0、奖励率=0，ε=0.1。冻结测量当前贪心动作真实平均奖励，训练不使用真均值。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/) · [algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### 对角 EWC

- 源码：[implementations/continual/ewc.py](../implementations/continual/ewc.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/ewc/)。
- 任务与对照：`retention_regression`；`online_sgd`。
- 指标与预算：当前任务无噪声预测 MSE；`environment_steps`。
- 设定：流式两维线性预测：x=[1,U(-1,1)]，中点真权重从[0.2,0.7]变为[-0.2,-0.7]，观测噪声σ=0.03，初始w=0。每次更新后冻结测量当前真任务解析MSE。同时记录第一任务误差，保留与适应可能冲突。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[algorithms/retention](https://yingwen.io/zh/continual-rl/algorithms/retention/) · [algorithms/plasticity](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)

### 向量 GVF GTD(λ)

- 源码：[implementations/continual/gvf_gtd_lambda.py](../implementations/continual/gvf_gtd_lambda.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/gvf_gtd_lambda/)。
- 任务与对照：`vector_gvf_prediction`；`gvf_td`。
- 指标与预算：两个 GVF 四状态分量 RMSE；`environment_steps`。
- 设定：继续两状态确定性交替流，从0开始，无终点。两个cumulant分别为到达1和到达0；折扣0.8/0.5，中点cumulant幅度从1变0.5。one-hot权重全0，在线更新后冻结测量解析GVF RMSE；ρ=1，不检验离策略稳定性。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[construction/predictive-knowledge](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/)

### 向量 GVF TD

- 源码：[implementations/continual/gvf_td.py](../implementations/continual/gvf_td.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/gvf_td/)。
- 任务与对照：`vector_gvf_prediction`；`gvf_gtd_lambda`。
- 指标与预算：两个 GVF 四状态分量 RMSE；`environment_steps`。
- 设定：继续两状态确定性交替流，从0开始，无终点。两个cumulant分别为到达1和到达0；折扣0.8/0.5，中点cumulant幅度从1变0.5。one-hot权重全0，在线更新后冻结测量解析GVF RMSE；ρ=1，不检验离策略稳定性。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[construction/predictive-knowledge](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/)

### IDBD

- 源码：[implementations/continual/idbd.py](../implementations/continual/idbd.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/idbd/)。
- 任务与对照：`adaptive_regression`；`constant_step_lms`。
- 指标与预算：当前任务无噪声预测 MSE；`environment_steps`。
- 设定：流式两维线性预测：x=[1,U(-1,1)]，中点真权重从[0.2,0.7]变为[-0.2,-0.7]，观测噪声σ=0.03，初始w=0。每次更新后冻结测量当前真任务解析MSE。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[algorithms/meta](https://yingwen.io/zh/continual-rl/algorithms/meta/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [algorithms/streaming](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

### 学习模型滚动规划

- 源码：[implementations/continual/learned_model_mpc.py](../implementations/continual/learned_model_mpc.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/learned_model_mpc/)。
- 任务与对照：`learned_chain_model_control`；`one_step_model`。
- 指标与预算：原始平均奖励冻结策略折扣回报；`environment_steps`。
- 设定：未知六格链从0开始：非终点期望奖励-0.01，终点奖励中点从1变0.5，观测Gaussian噪声σ=0.02。经验均值奖励与频率转移模型初始空，未访问乐观奖励0.05；γ=0.95，ε=0.1。滚动有限时域5步对照1步，两者每转移只执行首动作然后重新规划，额外模型计算不同。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[construction/models](https://yingwen.io/zh/continual-rl/construction/models/) · [construction/planning](https://yingwen.io/zh/continual-rl/construction/planning/) · [algorithms/dyna](https://yingwen.io/zh/continual-rl/algorithms/dyna/)

### 学习模型一步贪心

- 源码：[implementations/continual/one_step_model.py](../implementations/continual/one_step_model.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/one_step_model/)。
- 任务与对照：`learned_chain_model_control`；`learned_model_mpc`。
- 指标与预算：原始平均奖励冻结策略折扣回报；`environment_steps`。
- 设定：未知六格链从0开始：非终点期望奖励-0.01，终点奖励中点从1变0.5，观测Gaussian噪声σ=0.02。经验均值奖励与频率转移模型初始空，未访问乐观奖励0.05；γ=0.95，ε=0.1。滚动有限时域5步对照1步，两者每转移只执行首动作然后重新规划，额外模型计算不同。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[construction/models](https://yingwen.io/zh/continual-rl/construction/models/) · [construction/planning](https://yingwen.io/zh/continual-rl/construction/planning/) · [algorithms/dyna](https://yingwen.io/zh/continual-rl/algorithms/dyna/)

### 在线 SGD

- 源码：[implementations/continual/online_sgd.py](../implementations/continual/online_sgd.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/online_sgd/)。
- 任务与对照：`retention_regression`；`ewc`。
- 指标与预算：当前任务无噪声预测 MSE；`environment_steps`。
- 设定：流式两维线性预测：x=[1,U(-1,1)]，中点真权重从[0.2,0.7]变为[-0.2,-0.7]，观测噪声σ=0.03，初始w=0。每次更新后冻结测量当前真任务解析MSE。同时记录第一任务误差，保留与适应可能冲突。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[algorithms/retention](https://yingwen.io/zh/continual-rl/algorithms/retention/) · [algorithms/plasticity](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)

### SMDP 选项 Q-learning

- 源码：[implementations/continual/option_smdp_q.py](../implementations/continual/option_smdp_q.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/option_smdp_q/)。
- 任务与对照：`option_chain_control`；`primitive_q`。
- 指标与预算：原始奖励贪心策略折扣回报；`environment_steps`。
- 设定：六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1，Q=0。选项0向左一步，选项1向右最多两步；原始动作对照仅一步。横轴严格累计真实原始转移，任务平稳。未完成且被预算截断的选项不更新，真终点可提前结束选项。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[construction/options](https://yingwen.io/zh/continual-rl/construction/options/) · [algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/) · [algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

### 无探索奖励 Q-learning

- 源码：[implementations/continual/plain_q_exploration.py](../implementations/continual/plain_q_exploration.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/plain_q_exploration/)。
- 任务与对照：`count_exploration_chain`；`count_bonus`。
- 指标与预算：原始奖励贪心策略折扣回报；`environment_steps`。
- 设定：平稳六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1，Q=0。状态动作计数从0开始，每次访问先加1，再计算0.2/√N探索奖励；对照系数0。冻结测量原始奖励贪心回报；计数新奇不等于信息增益。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[algorithms/exploration](https://yingwen.io/zh/continual-rl/algorithms/exploration/) · [foundations/reward-design](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/)

### 势函数奖励塑形

- 源码：[implementations/continual/potential_shaping.py](../implementations/continual/potential_shaping.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/potential_shaping/)。
- 任务与对照：`potential_chain_control`；`chain_q`。
- 指标与预算：原始奖励贪心策略折扣回报；`environment_steps`。
- 设定：六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1，Q=0。势函数固定、在线更新后冻结贪心策略，以原始环境奖励评价。任务全程平稳，无中点切换；塑形只改变训练奖励。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[foundations/reward-design](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/)

### 原始动作 Q-learning

- 源码：[implementations/continual/primitive_q.py](../implementations/continual/primitive_q.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/primitive_q/)。
- 任务与对照：`option_chain_control`；`option_smdp_q`。
- 指标与预算：原始奖励贪心策略折扣回报；`environment_steps`。
- 设定：六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1，Q=0。选项0向左一步，选项1向右最多两步；原始动作对照仅一步。横轴严格累计真实原始转移，任务平稳。未完成且被预算截断的选项不更新，真终点可提前结束选项。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[construction/options](https://yingwen.io/zh/continual-rl/construction/options/) · [algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/) · [algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

### 蓄水池经验回放

- 源码：[implementations/continual/reservoir_replay.py](../implementations/continual/reservoir_replay.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/reservoir_replay/)。
- 任务与对照：`retention_regression`；`online_sgd`。
- 指标与预算：当前任务无噪声预测 MSE；`environment_steps`。
- 设定：流式两维线性预测：x=[1,U(-1,1)]，中点真权重从[0.2,0.7]变为[-0.2,-0.7]，观测噪声σ=0.03，初始w=0。每次更新后冻结测量当前真任务解析MSE。同时记录第一任务误差，保留与适应可能冲突。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[algorithms/retention](https://yingwen.io/zh/continual-rl/algorithms/retention/) · [algorithms/plasticity](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)

### RTRL

- 源码：[implementations/continual/rtrl.py](../implementations/continual/rtrl.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/rtrl/)。
- 任务与对照：`recurrent_sequence_prediction`；`tbptt`。
- 指标与预算：更新前序列预测平方误差 EMA；`sequences`。
- 设定：每步8个U(-1,1)输入组成独立序列；tanh老师a=0.8，b中点从0.4变-0.4。学生初始[0.3,0.1,0]，段内参数冻结、段间SGD；曲线为更新前误差平方EMA(0.95)，每段隐藏状态清零。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[construction/state](https://yingwen.io/zh/continual-rl/construction/state/) · [algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

### 后继特征 GPI

- 源码：[implementations/continual/successor_features_gpi.py](../implementations/continual/successor_features_gpi.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/successor_features_gpi/)。
- 任务与对照：`successor_feature_transfer`；`successor_features_policy`。
- 指标与预算：冻结选择策略的解析折扣价值；`environment_steps`。
- 设定：单状态两动作继续任务，γ=0.8，行为动作均匀采样；特征为动作one-hot，SF全0。基础策略库总动作0/总动作1。中点奖励权重[0.2,1]变[1,0.2]并直接告知算法；此实验展示已知线性奖励的迁移，不包含奖励权重学习。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[construction/models](https://yingwen.io/zh/continual-rl/construction/models/) · [construction/goals](https://yingwen.io/zh/continual-rl/construction/goals/) · [construction/predictive-knowledge](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)

### 后继特征固定基础策略

- 源码：[implementations/continual/successor_features_policy.py](../implementations/continual/successor_features_policy.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/successor_features_policy/)。
- 任务与对照：`successor_feature_transfer`；`successor_features_gpi`。
- 指标与预算：冻结选择策略的解析折扣价值；`environment_steps`。
- 设定：单状态两动作继续任务，γ=0.8，行为动作均匀采样；特征为动作one-hot，SF全0。基础策略库总动作0/总动作1。中点奖励权重[0.2,1]变[1,0.2]并直接告知算法；此实验展示已知线性奖励的迁移，不包含奖励权重学习。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[construction/models](https://yingwen.io/zh/continual-rl/construction/models/) · [construction/goals](https://yingwen.io/zh/continual-rl/construction/goals/) · [construction/predictive-knowledge](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)

### 截断 BPTT

- 源码：[implementations/continual/tbptt.py](../implementations/continual/tbptt.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/tbptt/)。
- 任务与对照：`recurrent_sequence_prediction`；`rtrl`。
- 指标与预算：更新前序列预测平方误差 EMA；`sequences`。
- 设定：每步8个U(-1,1)输入组成独立序列；tanh老师a=0.8，b中点从0.4变-0.4。学生初始[0.3,0.1,0]，段内参数冻结、段间SGD；曲线为更新前误差平方EMA(0.95)，每段隐藏状态清零。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[construction/state](https://yingwen.io/zh/continual-rl/construction/state/) · [algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

### TIDBD(λ)

- 源码：[implementations/continual/tidbd.py](../implementations/continual/tidbd.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/tidbd/)。
- 任务与对照：`adaptive_regression`；`constant_step_lms`。
- 指标与预算：当前任务无噪声预测 MSE；`environment_steps`。
- 设定：流式两维线性预测：x=[1,U(-1,1)]，中点真权重从[0.2,0.7]变为[-0.2,-0.7]，观测噪声σ=0.03，初始w=0。每次更新后冻结测量当前真任务解析MSE。此γ=0退化教学仅验证即时预测步长自适应，不检验带bootstrap的TIDBD。
- 范围：小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。
- 教材：[algorithms/meta](https://yingwen.io/zh/continual-rl/algorithms/meta/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/) · [algorithms/streaming](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

### A2C

- 源码：[implementations/deep/a2c.py](../implementations/deep/a2c.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/deep-a2c/)。
- 任务与对照：`deadline-chain`；`deep-vpg`。
- 指标与预算：frozen_evaluation_return；`environment_steps`。
- 设定：A2C：显式网络更新的 CPU 教学控制实验
- 范围：小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。
- 教材：[foundations/deep/policy-gradient](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/) · [algorithms/policy](https://yingwen.io/zh/continual-rl/algorithms/policy/)

### C51

- 源码：[implementations/deep/c51.py](../implementations/deep/c51.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/deep-c51/)。
- 任务与对照：`deadline-chain`；`deep-dqn`。
- 指标与预算：frozen_evaluation_return；`environment_steps`。
- 设定：C51：显式网络更新的 CPU 教学控制实验
- 范围：小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。
- 教材：[foundations/deep/distributional](https://yingwen.io/zh/continual-rl/foundations/deep/distributional/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/)

### CQL

- 源码：[implementations/deep/cql.py](../implementations/deep/cql.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/deep-cql/)。
- 任务与对照：`offline-deadline-chain-fixed512`；`deep-offline_q_learning`。
- 指标与预算：frozen_evaluation_return；`training_batches`。
- 设定：CQL：显式网络更新的 CPU 教学控制实验
- 范围：小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。
- 教材：[foundations/deep/offline](https://yingwen.io/zh/continual-rl/foundations/deep/offline/)

### DDPG

- 源码：[implementations/deep/ddpg.py](../implementations/deep/ddpg.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/deep-ddpg/)。
- 任务与对照：`bounded-lq`；`deep-td3`。
- 指标与预算：frozen_evaluation_return；`environment_steps`。
- 设定：DDPG：显式网络更新的 CPU 教学控制实验
- 范围：小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。
- 教材：[foundations/deep/deterministic-control](https://yingwen.io/zh/continual-rl/foundations/deep/deterministic-control/) · [algorithms/policy](https://yingwen.io/zh/continual-rl/algorithms/policy/)

### Double DQN

- 源码：[implementations/deep/double_dqn.py](../implementations/deep/double_dqn.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/deep-double_dqn/)。
- 任务与对照：`deadline-chain`；`deep-dqn`。
- 指标与预算：frozen_evaluation_return；`environment_steps`。
- 设定：Double DQN：显式网络更新的 CPU 教学控制实验
- 范围：小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。
- 教材：[foundations/deep/deep-value](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/)

### DQN

- 源码：[implementations/deep/dqn.py](../implementations/deep/dqn.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/deep-dqn/)。
- 任务与对照：`deadline-chain`；`deep-double_dqn`。
- 指标与预算：frozen_evaluation_return；`environment_steps`。
- 设定：DQN：显式网络更新的 CPU 教学控制实验
- 范围：小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。
- 教材：[foundations/deep/deep-value](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/)

### IQL

- 源码：[implementations/deep/iql.py](../implementations/deep/iql.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/deep-iql/)。
- 任务与对照：`offline-deadline-chain-fixed512`；`deep-offline_q_learning`。
- 指标与预算：frozen_evaluation_return；`training_batches`。
- 设定：IQL：显式网络更新的 CPU 教学控制实验
- 范围：小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。
- 教材：[foundations/deep/offline](https://yingwen.io/zh/continual-rl/foundations/deep/offline/)

### Offline Q-learning

- 源码：[implementations/deep/offline_q_learning.py](../implementations/deep/offline_q_learning.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/deep-offline_q_learning/)。
- 任务与对照：`offline-deadline-chain-fixed512`；`deep-cql`。
- 指标与预算：frozen_evaluation_return；`training_batches`。
- 设定：Offline Q-learning：显式网络更新的 CPU 教学控制实验
- 范围：小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。
- 教材：[foundations/deep/offline](https://yingwen.io/zh/continual-rl/foundations/deep/offline/)

### PPO

- 源码：[implementations/deep/ppo.py](../implementations/deep/ppo.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/deep-ppo/)。
- 任务与对照：`deadline-chain`；`deep-vpg`。
- 指标与预算：frozen_evaluation_return；`environment_steps`。
- 设定：PPO：显式网络更新的 CPU 教学控制实验
- 范围：小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。
- 教材：[foundations/deep/trust-region](https://yingwen.io/zh/continual-rl/foundations/deep/trust-region/) · [algorithms/policy](https://yingwen.io/zh/continual-rl/algorithms/policy/)

### QR-DQN

- 源码：[implementations/deep/qr_dqn.py](../implementations/deep/qr_dqn.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/deep-qr_dqn/)。
- 任务与对照：`deadline-chain`；`deep-dqn`。
- 指标与预算：frozen_evaluation_return；`environment_steps`。
- 设定：QR-DQN：显式网络更新的 CPU 教学控制实验
- 范围：小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。
- 教材：[foundations/deep/distributional](https://yingwen.io/zh/continual-rl/foundations/deep/distributional/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/)

### SAC

- 源码：[implementations/deep/sac.py](../implementations/deep/sac.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/deep-sac/)。
- 任务与对照：`bounded-lq`；`deep-ddpg`。
- 指标与预算：frozen_evaluation_return；`environment_steps`。
- 设定：SAC：显式网络更新的 CPU 教学控制实验
- 范围：小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。
- 教材：[foundations/deep/entropy-control](https://yingwen.io/zh/continual-rl/foundations/deep/entropy-control/) · [algorithms/soft-control](https://yingwen.io/zh/continual-rl/algorithms/soft-control/)

### TD3

- 源码：[implementations/deep/td3.py](../implementations/deep/td3.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/deep-td3/)。
- 任务与对照：`bounded-lq`；`deep-ddpg`。
- 指标与预算：frozen_evaluation_return；`environment_steps`。
- 设定：TD3：显式网络更新的 CPU 教学控制实验
- 范围：小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。
- 教材：[foundations/deep/deterministic-control](https://yingwen.io/zh/continual-rl/foundations/deep/deterministic-control/) · [algorithms/policy](https://yingwen.io/zh/continual-rl/algorithms/policy/)

### TRPO

- 源码：[implementations/deep/trpo.py](../implementations/deep/trpo.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/deep-trpo/)。
- 任务与对照：`deadline-chain`；`deep-vpg`。
- 指标与预算：frozen_evaluation_return；`environment_steps`。
- 设定：共轭梯度与 KL 回溯约束的实际 TRPO 教学训练
- 范围：小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。
- 教材：[foundations/deep/trust-region](https://yingwen.io/zh/continual-rl/foundations/deep/trust-region/) · [algorithms/policy](https://yingwen.io/zh/continual-rl/algorithms/policy/)

### VPG

- 源码：[implementations/deep/vpg.py](../implementations/deep/vpg.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/deep-vpg/)。
- 任务与对照：`deadline-chain`；`deep-a2c`。
- 指标与预算：frozen_evaluation_return；`environment_steps`。
- 设定：VPG：显式网络更新的 CPU 教学控制实验
- 范围：小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。
- 教材：[foundations/deep/policy-gradient](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/) · [algorithms/policy](https://yingwen.io/zh/continual-rl/algorithms/policy/)

### 迭代策略评价

- 源码：[implementations/extended_classic/iterative_policy_evaluation.py](../implementations/extended_classic/iterative_policy_evaluation.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/iterative_policy_evaluation/)。
- 任务与对照：`extended_fixed_chain_planning`；`policy_evaluation_inplace`。
- 指标与预算：固定策略最大价值误差；`model_sweeps`。
- 设定：已知六格链，从状态0开始，π=(0.2,0.8)，γ=0.95；V初始0。每次预算是5个状态备份的一轮扫描；误差按精确线性方程解测量。
- 范围：有限教学任务，不构成一般性能或原论文实验复现。
- 教材：[foundations/tabular/dynamic-programming](https://yingwen.io/zh/continual-rl/foundations/tabular/dynamic-programming/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/)

### Every-visit MC

- 源码：[implementations/extended_classic/every_visit_mc.py](../implementations/extended_classic/every_visit_mc.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/every_visit_mc/)。
- 任务与对照：`extended_walk_mc`；`first_visit_mc_reference`。
- 指标与预算：五状态价值 RMSE；`environment_steps`。
- 设定：五状态无偏随机游走，从3开始，终点左右奖励0/1，γ=1，价值初始0。仅完整回合产生MC更新；预算尾未完成回合不伪终止。
- 范围：有限教学任务，不构成一般性能或原论文实验复现。
- 教材：[foundations/tabular/monte-carlo](https://yingwen.io/zh/continual-rl/foundations/tabular/monte-carlo/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/)

### MC control

- 源码：[implementations/extended_classic/mc_control.py](../implementations/extended_classic/mc_control.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/mc_control/)。
- 任务与对照：`extended_chain_control`；`watkins_q_lambda`。
- 指标与预算：贪心策略精确折扣回报；`environment_steps`。
- 设定：平稳六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1。初始估值0，每预算单位是真实环境转移；冻结贪心策略用终点/循环精确原奖励回报评价。MC每回合固定当前ε-soft策略，只在完整回合首次状态动作访问更新。
- 范围：有限教学任务，不构成一般性能或原论文实验复现。
- 教材：[foundations/tabular/monte-carlo](https://yingwen.io/zh/continual-rl/foundations/tabular/monte-carlo/) · [algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/)

### Tree Backup

- 源码：[implementations/extended_classic/tree_backup.py](../implementations/extended_classic/tree_backup.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/tree_backup/)。
- 任务与对照：`extended_fixed_q_prediction`；`q_sigma`。
- 指标与预算：固定策略十状态动作价值 RMSE；`environment_steps`。
- 设定：六格链固定随机π=μ=(0.2,0.8)，γ=0.95，Q初始0，从0开始。保存最长三步前缀，成熟后更新最早Q，终点依次冲洗短目标；预算未成熟尾不伪终止。固定策略评价，不是控制训练；Retrace本实验仅on-policy。
- 范围：有限教学任务，不构成一般性能或原论文实验复现。
- 教材：[foundations/tabular/multistep](https://yingwen.io/zh/continual-rl/foundations/tabular/multistep/) · [algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

### Q(σ)

- 源码：[implementations/extended_classic/q_sigma.py](../implementations/extended_classic/q_sigma.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/q_sigma/)。
- 任务与对照：`extended_fixed_q_prediction`；`tree_backup`。
- 指标与预算：固定策略十状态动作价值 RMSE；`environment_steps`。
- 设定：六格链固定随机π=μ=(0.2,0.8)，γ=0.95，Q初始0，从0开始。保存最长三步前缀，成熟后更新最早Q，终点依次冲洗短目标；预算未成熟尾不伪终止。固定策略评价，不是控制训练；Retrace本实验仅on-policy。
- 范围：有限教学任务，不构成一般性能或原论文实验复现。
- 教材：[foundations/tabular/multistep](https://yingwen.io/zh/continual-rl/foundations/tabular/multistep/) · [algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

### Gradient MC

- 源码：[implementations/extended_classic/gradient_mc.py](../implementations/extended_classic/gradient_mc.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/gradient_mc/)。
- 任务与对照：`extended_linear_walk`；`lstd`。
- 指标与预算：五状态线性价值 RMSE；`environment_steps`。
- 设定：无偏五状态随机游走，从3开始，γ=1，特征[1,s/6]、w=0。冻结当前线性预测测量五状态RMSE；LSTD显式1e-5对角正则，MC只更新完整回合。
- 范围：有限教学任务，不构成一般性能或原论文实验复现。
- 教材：[foundations/approximation/prediction](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/)

### LSTD

- 源码：[implementations/extended_classic/lstd.py](../implementations/extended_classic/lstd.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/lstd/)。
- 任务与对照：`extended_linear_walk`；`gradient_mc`。
- 指标与预算：五状态线性价值 RMSE；`environment_steps`。
- 设定：无偏五状态随机游走，从3开始，γ=1，特征[1,s/6]、w=0。冻结当前线性预测测量五状态RMSE；LSTD显式1e-5对角正则，MC只更新完整回合。
- 范围：有限教学任务，不构成一般性能或原论文实验复现。
- 教材：[foundations/approximation/prediction](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/)

### 半梯度 SARSA

- 源码：[implementations/extended_classic/semi_gradient_sarsa.py](../implementations/extended_classic/semi_gradient_sarsa.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/semi_gradient_sarsa/)。
- 任务与对照：`extended_chain_control`；`mc_control`。
- 指标与预算：贪心策略精确折扣回报；`environment_steps`。
- 设定：平稳六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1。初始估值0，每预算单位是真实环境转移；冻结贪心策略用终点/循环精确原奖励回报评价。每动作三个共享多项式特征[1,s/5,(s/5)^2]，共6权重；与表格MC容量不同，不能推论同表示优势。
- 范围：有限教学任务，不构成一般性能或原论文实验复现。
- 教材：[foundations/approximation/features-control](https://yingwen.io/zh/continual-rl/foundations/approximation/features-control/) · [algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/)

### Differential TD

- 源码：[implementations/extended_classic/differential_td_prediction.py](../implementations/extended_classic/differential_td_prediction.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/differential_td_prediction/)。
- 任务与对照：`extended_average_prediction`；`average_smdp_option`。
- 指标与预算：目标奖励率绝对误差；`environment_steps`。
- 设定：两状态无终点交替流，从0开始，Bernoulli均值0.2/0.8，目标长期奖励率0.5、差分偏差差0.3。价值及率从0开始；原始转移预算。选项持续1或3步，学旧平均时长并归一化更新；只含固定策略预测子机制，非完整平均奖励选项控制。
- 范围：有限教学任务，不构成一般性能或原论文实验复现。
- 教材：[algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/) · [construction/options](https://yingwen.io/zh/continual-rl/construction/options/)

### Relative Value Iteration

- 源码：[implementations/extended_classic/relative_value_iteration.py](../implementations/extended_classic/relative_value_iteration.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/relative_value_iteration/)。
- 任务与对照：`extended_average_known_model`；`unnormalized_value_iteration`。
- 指标与预算：相对偏差差绝对误差；`model_sweeps`。
- 设定：已知不可约非周期两状态模型，奖励0.2/0.8，P(自环)=0.8，其余0.2。初值0，每扫描2个模型备份；测量h1-h0相对于1.5的误差，未归一化对照保留公共线性增长项。
- 范围：有限教学任务，不构成一般性能或原论文实验复现。
- 教材：[algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)

### Watkins Q(λ)

- 源码：[implementations/extended_classic/watkins_q_lambda.py](../implementations/extended_classic/watkins_q_lambda.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/watkins_q_lambda/)。
- 任务与对照：`extended_chain_control`；`mc_control`。
- 指标与预算：贪心策略精确折扣回报；`environment_steps`。
- 设定：平稳六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1。初始估值0，每预算单位是真实环境转移；冻结贪心策略用终点/循环精确原奖励回报评价。累积迹λ=0.8；非贪心下一动作先按旧Q判定，本次误差信用更新后才清除未来资格。
- 范围：有限教学任务，不构成一般性能或原论文实验复现。
- 教材：[algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/) · [algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/)

### Retrace

- 源码：[implementations/extended_classic/retrace.py](../implementations/extended_classic/retrace.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/retrace/)。
- 任务与对照：`extended_fixed_q_prediction`；`tree_backup`。
- 指标与预算：固定策略十状态动作价值 RMSE；`environment_steps`。
- 设定：六格链固定随机π=μ=(0.2,0.8)，γ=0.95，Q初始0，从0开始。保存最长三步前缀，成熟后更新最早Q，终点依次冲洗短目标；预算未成熟尾不伪终止。固定策略评价，不是控制训练；Retrace本实验仅on-policy。
- 范围：有限教学任务，不构成一般性能或原论文实验复现。
- 教材：[algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/) · [foundations/tabular/multistep](https://yingwen.io/zh/continual-rl/foundations/tabular/multistep/)

### REINFORCE

- 源码：[implementations/extended_adaptation/reinforce.py](../implementations/extended_adaptation/reinforce.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-reinforce/)。
- 任务与对照：`extended-chain`；`extended-policy_td`。
- 指标与预算：frozen_greedy_return；`environment_steps`。
- 设定：REINFORCE的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：完整回合 Monte Carlo REINFORCE 教学实例；价值网络仅作独立回报拟合诊断，不作为 actor baseline；未完成尾部回合不更新。冻结贪心回报不等同随机策略目标。
- 教材：[foundations/approximation/policy-gradient](https://yingwen.io/zh/continual-rl/foundations/approximation/policy-gradient/) · [foundations/deep/policy-gradient](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/)

### Thompson Sampling

- 源码：[implementations/extended_classic/thompson_sampling.py](../implementations/extended_classic/thompson_sampling.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/thompson_sampling/)。
- 任务与对照：`extended_stationary_bandit`；`ucb_extended_baseline`。
- 指标与预算：累计采样平均奖励；`environment_steps`。
- 设定：三臂平稳Bernoulli均值0.2/0.5/0.8，初始无观测，Thompson使用独立Beta(1,1)先验。真实选择和奖励逐步更新，报告实际累计平均奖励，不使用真均值选动作。
- 范围：有限教学任务，不构成一般性能或原论文实验复现。
- 教材：[foundations/deep/exploration](https://yingwen.io/zh/continual-rl/foundations/deep/exploration/) · [algorithms/exploration](https://yingwen.io/zh/continual-rl/algorithms/exploration/)

### 完整 BPTT

- 源码：[implementations/extended_adaptation/bptt.py](../implementations/extended_adaptation/bptt.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-bptt/)。
- 任务与对照：`delayed-cue-sequences`；`extended-truncated_bptt`。
- 指标与预算：frozen_sequence_mse；`sequences`。
- 设定：完整BPTT RNN训练的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：8输入末端监督任务完整BPTT，序列内不截断计算图；不代表无限流在线控制。
- 教材：[construction/state](https://yingwen.io/zh/continual-rl/construction/state/)

### 无回放 Actor-Critic

- 源码：[implementations/extended_adaptation/fresh_ac.py](../implementations/extended_adaptation/fresh_ac.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-fresh_ac/)。
- 任务与对照：`context-bandit`；`extended-clear`。
- 指标与预算：冻结当前上下文策略期望奖励；`environment_steps`。
- 设定：无回放 Actor-Critic的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：小型教学任务，不是原论文性能复现；组件实验不计作完整原方法。冻结评价与训练更新分离，预算不代表相同计算成本。
- 教材：[algorithms/meta](https://yingwen.io/zh/continual-rl/algorithms/meta/)

### 不替换 MLP SGD

- 源码：[implementations/extended_adaptation/plain_mlp.py](../implementations/extended_adaptation/plain_mlp.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-plain_mlp/)。
- 任务与对照：`switch-regression`；`extended-redo`。
- 指标与预算：冻结当前函数均方误差；`training_samples`。
- 设定：不替换 MLP SGD的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：小型教学任务，不是原论文性能复现；组件实验不计作完整原方法。冻结评价与训练更新分离，预算不代表相同计算成本。
- 教材：[algorithms/meta](https://yingwen.io/zh/continual-rl/algorithms/meta/)

### 一步Actor-Critic对照

- 源码：[implementations/extended_adaptation/policy_td.py](../implementations/extended_adaptation/policy_td.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-policy_td/)。
- 任务与对照：`extended-chain`；`extended-reinforce`。
- 指标与预算：frozen_greedy_return；`environment_steps`。
- 设定：一步Actor-Critic对照的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：小型教学任务，不是原论文性能复现；组件实验不计作完整原方法。冻结评价与训练更新分离，预算不代表相同计算成本。
- 教材：[foundations/deep/policy-gradient](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/)

### 固定步长非线性TD对照

- 源码：[implementations/extended_adaptation/streaming_td.py](../implementations/extended_adaptation/streaming_td.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-streaming_td/)。
- 任务与对照：`nonlinear-alternating-prediction`；`extended-obgd2024`。
- 指标与预算：frozen_value_mse；`environment_steps`。
- 设定：固定步长非线性TD对照的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：小型教学任务，不是原论文性能复现；组件实验不计作完整原方法。冻结评价与训练更新分离，预算不代表相同计算成本。
- 教材：[algorithms/streaming](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

### 固定步长AC(lambda)对照

- 源码：[implementations/extended_adaptation/trace_ac.py](../implementations/extended_adaptation/trace_ac.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-trace_ac/)。
- 任务与对照：`linear-trace-chain`；`extended-metatrace`。
- 指标与预算：frozen_greedy_return；`environment_steps`。
- 设定：固定步长AC(lambda)对照的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：小型教学任务，不是原论文性能复现；组件实验不计作完整原方法。冻结评价与训练更新分离，预算不代表相同计算成本。
- 教材：[algorithms/meta](https://yingwen.io/zh/continual-rl/algorithms/meta/)

### TBPTT三步对照

- 源码：[implementations/extended_adaptation/truncated_bptt.py](../implementations/extended_adaptation/truncated_bptt.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-truncated_bptt/)。
- 任务与对照：`delayed-cue-sequences`；`extended-bptt`。
- 指标与预算：frozen_sequence_mse；`sequences`。
- 设定：TBPTT三步对照的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：小型教学任务，不是原论文性能复现；组件实验不计作完整原方法。冻结评价与训练更新分离，预算不代表相同计算成本。
- 教材：[construction/state](https://yingwen.io/zh/continual-rl/construction/state/)

### 首次访问 MC 对照

- 源码：[implementations/extended_classic/first_visit_mc_reference.py](../implementations/extended_classic/first_visit_mc_reference.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/first_visit_mc_reference/)。
- 任务与对照：`extended_walk_mc`；`every_visit_mc`。
- 指标与预算：五状态价值 RMSE；`environment_steps`。
- 设定：五状态无偏随机游走，从3开始，终点左右奖励0/1，γ=1，价值初始0。仅完整回合产生MC更新；预算尾未完成回合不伪终止。
- 范围：有限教学任务，不构成一般性能或原论文实验复现。
- 教材：[foundations/tabular/monte-carlo](https://yingwen.io/zh/continual-rl/foundations/tabular/monte-carlo/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/)

### 原地策略评价对照

- 源码：[implementations/extended_classic/policy_evaluation_inplace.py](../implementations/extended_classic/policy_evaluation_inplace.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/policy_evaluation_inplace/)。
- 任务与对照：`extended_fixed_chain_planning`；`iterative_policy_evaluation`。
- 指标与预算：固定策略最大价值误差；`model_sweeps`。
- 设定：已知六格链，从状态0开始，π=(0.2,0.8)，γ=0.95；V初始0。每次预算是5个状态备份的一轮扫描；误差按精确线性方程解测量。
- 范围：有限教学任务，不构成一般性能或原论文实验复现。
- 教材：[foundations/tabular/dynamic-programming](https://yingwen.io/zh/continual-rl/foundations/tabular/dynamic-programming/) · [algorithms/value](https://yingwen.io/zh/continual-rl/algorithms/value/)

### UCB 对照

- 源码：[implementations/extended_classic/ucb_extended_baseline.py](../implementations/extended_classic/ucb_extended_baseline.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/ucb_extended_baseline/)。
- 任务与对照：`extended_stationary_bandit`；`thompson_sampling`。
- 指标与预算：累计采样平均奖励；`environment_steps`。
- 设定：三臂平稳Bernoulli均值0.2/0.5/0.8，初始无观测，Thompson使用独立Beta(1,1)先验。真实选择和奖励逐步更新，报告实际累计平均奖励，不使用真均值选动作。
- 范围：有限教学任务，不构成一般性能或原论文实验复现。
- 教材：[foundations/deep/exploration](https://yingwen.io/zh/continual-rl/foundations/deep/exploration/) · [algorithms/exploration](https://yingwen.io/zh/continual-rl/algorithms/exploration/)

### 未归一化平均奖励价值迭代对照

- 源码：[implementations/extended_classic/unnormalized_value_iteration.py](../implementations/extended_classic/unnormalized_value_iteration.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/unnormalized_value_iteration/)。
- 任务与对照：`extended_average_known_model`；`relative_value_iteration`。
- 指标与预算：相对偏差差绝对误差；`model_sweeps`。
- 设定：已知不可约非周期两状态模型，奖励0.2/0.8，P(自环)=0.8，其余0.2。初值0，每扫描2个模型备份；测量h1-h0相对于1.5的误差，未归一化对照保留公共线性增长项。
- 范围：有限教学任务，不构成一般性能或原论文实验复现。
- 教材：[algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)

### Differential Q · multistate control

- 源码：[implementations/average_systems/differential_q_multistate.py](../implementations/average_systems/differential_q_multistate.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/differential_q_multistate/)。
- 任务与对照：`average_control_learned_model`；`differential_dyna`。
- 指标与预算：冻结贪心策略的精确平均奖励；`environment_steps`。
- 设定：三状态双动作遍历MDP，epsilon=0.25持续探索；同时学最优奖励率与动作差分价值。评估冻结贪心策略的真实奖励率，不把探索轨迹均值当目标。
- 范围：平稳有限表格MDP与常数步长教学实验；不声称在1200步收敛，也不外推神经网络保证。
- 教材：[algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/) · [construction/planning](https://yingwen.io/zh/continual-rl/construction/planning/)

### Off-policy Differential TD

- 源码：[implementations/average_systems/differential_td_offpolicy.py](../implementations/average_systems/differential_td_offpolicy.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/differential_td_offpolicy/)。
- 任务与对照：`average_prediction_offpolicy`；`wrong_behavior_mean_td`。
- 指标与预算：参考状态对齐后的差分价值RMSE；`environment_steps`。
- 设定：行为策略各动作一半，目标策略因状态而异。动作重要性比同时乘bias与g更新，目标仍是pi的Poisson方程；无重放、无外部重置。
- 范围：平稳遍历有限MDP、表格表示、常数步长、1200步教学预算；不是神经网络稳定性或非平稳收敛证明。
- 教材：[algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)

### On-policy Differential TD

- 源码：[implementations/average_systems/differential_td_onpolicy.py](../implementations/average_systems/differential_td_onpolicy.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/differential_td_onpolicy/)。
- 任务与对照：`average_prediction_onpolicy`；`sample_mean_td`。
- 指标与预算：参考状态对齐后的差分价值RMSE；`environment_steps`。
- 设定：三状态双动作持续链，固定目标策略实际采样。旧TD误差共同更新bias和奖励率；每个环境步只更新一次。固定步长用于有限预算演示，不宣称渐近收敛。
- 范围：平稳遍历有限MDP、表格表示、常数步长、1200步教学预算；不是神经网络稳定性或非平稳收敛证明。
- 教材：[algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)

### RVI · known three-state model

- 源码：[implementations/average_systems/rvi_multistate.py](../implementations/average_systems/rvi_multistate.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/rvi_multistate/)。
- 任务与对照：`average_known_model_planning`；`unnormalized_vi_multistate`。
- 指标与预算：最优偏差参考对齐RMSE；`model_sweeps`。
- 设定：已知三状态双动作模型，同步Bellman最优备份后减去参考状态0值；与同模型不减常数的VI对照。每扫描6个状态动作备份。
- 范围：已知小模型、精确期望备份、遍历非周期任务。种子不影响确定性结果，不把模型扫描当真实交互。普通RVI在其他周期任务中可能振荡。
- 教材：[algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/) · [construction/planning](https://yingwen.io/zh/continual-rl/construction/planning/)

## 独立子机制实验（有学习循环、对照与明确适用范围）（93 项）

### Ordinary IS

- 源码：[implementations/extended_classic/ordinary_is.py](../implementations/extended_classic/ordinary_is.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/ordinary_is/)。
- 任务与对照：`extended_ope_horizon3`；`per_decision_is`。
- 指标与预算：目标策略回报估计绝对误差；`episodes`。
- 设定：固定三步回合，μ=(0.5,0.5)、π=(0.2,0.8)，动作Bernoulli均值0.1/0.9，γ=0.9。每预算单位是一条真实三转移回合；目标策略固定、比率精确给定，累计估计初始0；不训练控制策略。
- 范围：仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。
- 教材：[foundations/tabular/monte-carlo](https://yingwen.io/zh/continual-rl/foundations/tabular/monte-carlo/) · [foundations/deep/offline](https://yingwen.io/zh/continual-rl/foundations/deep/offline/)

### Weighted IS

- 源码：[implementations/extended_classic/weighted_is.py](../implementations/extended_classic/weighted_is.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/weighted_is/)。
- 任务与对照：`extended_ope_horizon3`；`ordinary_is`。
- 指标与预算：目标策略回报估计绝对误差；`episodes`。
- 设定：固定三步回合，μ=(0.5,0.5)、π=(0.2,0.8)，动作Bernoulli均值0.1/0.9，γ=0.9。每预算单位是一条真实三转移回合；目标策略固定、比率精确给定，累计估计初始0；不训练控制策略。
- 范围：仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。
- 教材：[foundations/tabular/monte-carlo](https://yingwen.io/zh/continual-rl/foundations/tabular/monte-carlo/) · [foundations/deep/offline](https://yingwen.io/zh/continual-rl/foundations/deep/offline/)

### 平均奖励 SMDP option 更新

- 源码：[implementations/extended_classic/average_smdp_option.py](../implementations/extended_classic/average_smdp_option.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/average_smdp_option/)。
- 任务与对照：`extended_average_prediction`；`differential_td_prediction`。
- 指标与预算：目标奖励率绝对误差；`environment_steps`。
- 设定：两状态无终点交替流，从0开始，Bernoulli均值0.2/0.8，目标长期奖励率0.5、差分偏差差0.3。价值及率从0开始；原始转移预算。选项持续1或3步，学旧平均时长并归一化更新；只含固定策略预测子机制，非完整平均奖励选项控制。
- 范围：仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。
- 教材：[algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/) · [construction/options](https://yingwen.io/zh/continual-rl/construction/options/)

### V-trace

- 源码：[implementations/extended_adaptation/vtrace.py](../implementations/extended_adaptation/vtrace.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-vtrace/)。
- 任务与对照：`offpolicy-one-state`；`extended-is_td`。
- 指标与预算：frozen_target_value_mse；`environment_steps`。
- 设定：V-trace价值学习组件的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：非同策略一状态价值预测、多步V-trace目标组件；策略固定、只有价值优化，非完整IMPALA控制系统。
- 教材：[algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

### Greedy adaptive λ

- 源码：[implementations/extended_classic/greedy_adaptive_lambda.py](../implementations/extended_classic/greedy_adaptive_lambda.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/greedy_adaptive_lambda/)。
- 任务与对照：`extended_local_lambda_mixture`；`fixed_lambda_reference`。
- 指标与预算：独立验证样本累计混合 MSE；`sample_pairs`。
- 设定：单状态局部混合：固定bootstrap预测0，未来回报N(1,0.25)。每单位含一个统计训练样本和一个独立验证样本；用旧均值/方差选择λ，再看新样本。仅学局部回报矩与λ，不实现整条TD链、状态相关网络和原方法的全部递归估计。
- 范围：仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。
- 教材：[algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

### Expected Eligibility Traces

- 源码：[implementations/extended_classic/expected_eligibility_traces.py](../implementations/extended_classic/expected_eligibility_traces.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/expected_eligibility_traces/)。
- 任务与对照：`extended_markov_merge_credit`；`sampled_eligibility_reference`。
- 指标与预算：累计平均信用方向 RMSE；`episodes`。
- 设定：两个等概率历史汇入完整Markov状态，历史资格为[0.72,0,1]或[0,0.72,1]；后继奖励独立取0或2。主预测参数冻结0，条件期望迹由历史样本均值学习；比较累计信用方向与真实均值[0.36,0.36,1]。不训练完整value网络或递归ET混合。
- 范围：仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。
- 教材：[algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

### Gradient Eligibility Traces

- 源码：[implementations/extended_classic/gradient_eligibility_traces.py](../implementations/extended_classic/gradient_eligibility_traces.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/gradient_eligibility_traces/)。
- 任务与对照：`extended_frozen_gtd2_episodes`；`gradient_trace_forward_reference`。
- 指标与预算：两状态非线性价值 RMSE；`episodes`。
- 设定：每episode固定两转移0→1→真终点，末奖励Bernoulli(0.5)，γ=0.9，λ=0.8。V=tanh(w·x)，H=tanh(h·x)，one-hot特征，w=h=0；两网络段内冻结、段间更新，测量真值[0.45,0.5]。只覆盖GTD2冻结参数资格迹方向，不覆盖TDC/TDRC和深度控制。
- 范围：仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。
- 教材：[algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

### GAE

- 源码：[implementations/extended_adaptation/gae.py](../implementations/extended_adaptation/gae.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-gae/)。
- 任务与对照：`extended-chain`；`extended-policy_td`。
- 指标与预算：frozen_greedy_return；`environment_steps`。
- 设定：GAE actor-critic组件实验的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：仅 GAE(lambda) 配合普通 actor-critic，不含 PPO/TRPO；有限deadline是真终止，批截止则bootstrap。冻结贪心回报不等同随机策略目标。
- 教材：[foundations/deep/policy-gradient](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/)

### Per-decision IS

- 源码：[implementations/extended_classic/per_decision_is.py](../implementations/extended_classic/per_decision_is.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/per_decision_is/)。
- 任务与对照：`extended_ope_horizon3`；`ordinary_is`。
- 指标与预算：目标策略回报估计绝对误差；`episodes`。
- 设定：固定三步回合，μ=(0.5,0.5)、π=(0.2,0.8)，动作Bernoulli均值0.1/0.9，γ=0.9。每预算单位是一条真实三转移回合；目标策略固定、比率精确给定，累计估计初始0；不训练控制策略。
- 范围：仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。
- 教材：[foundations/deep/offline](https://yingwen.io/zh/continual-rl/foundations/deep/offline/) · [foundations/tabular/monte-carlo](https://yingwen.io/zh/continual-rl/foundations/tabular/monte-carlo/)

### Sequential Doubly Robust

- 源码：[implementations/extended_classic/sequential_dr.py](../implementations/extended_classic/sequential_dr.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/sequential_dr/)。
- 任务与对照：`extended_ope_horizon3`；`ordinary_is`。
- 指标与预算：目标策略回报估计绝对误差；`episodes`。
- 设定：固定三步回合，μ=(0.5,0.5)、π=(0.2,0.8)，动作Bernoulli均值0.1/0.9，γ=0.9。每预算单位是一条真实三转移回合；目标策略固定、比率精确给定，累计估计初始0；不训练控制策略。
- 范围：仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。
- 教材：[foundations/deep/offline](https://yingwen.io/zh/continual-rl/foundations/deep/offline/) · [foundations/tabular/monte-carlo](https://yingwen.io/zh/continual-rl/foundations/tabular/monte-carlo/)

### CMDP primal-dual

- 源码：[implementations/extended_classic/cmdp_primal_dual.py](../implementations/extended_classic/cmdp_primal_dual.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/cmdp_primal_dual/)。
- 任务与对照：`extended_constrained_bandit`；`unconstrained_policy_reference`。
- 指标与预算：最优可行动作概率绝对误差；`environment_steps`。
- 设定：单状态两动作reward Bernoulli均值0.2/0.9，cost=动作1指示，预算0.4，独立真实样本。π1=sigmoid(logit)，初值logit=0、乘子0；解析可行最优π1=0.4只用于测量。对照忽略约束，不是等可行解竞争；不实现一般多状态CMDP占据测度求解。
- 范围：仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。
- 教材：[foundations/deep/constraints](https://yingwen.io/zh/continual-rl/foundations/deep/constraints/)

### 2×2 minimax

- 源码：[implementations/extended_classic/minimax_2x2.py](../implementations/extended_classic/minimax_2x2.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/minimax_2x2/)。
- 任务与对照：`extended_empirical_zero_sum_game`；`fictitious_play_reference`。
- 指标与预算：真实矩阵 Nash 间隙；`payoff_samples`。
- 设定：固定零和矩阵[[0.8,-0.2],[-0.4,0.6]]，每步均匀采一个矩阵元素、观测Gaussian噪声σ=0.2，经验均值初值0。解析minimax或虚拟博弈只用已学习矩阵；真实矩阵仅测量Nash间隙，不实现一般Markov博弈。
- 范围：仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。
- 教材：[foundations/deep/multi-agent](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent/)

### COMA counterfactual advantage

- 源码：[implementations/multiagent/coma.py](../implementations/multiagent/coma.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/coma_contextual/)。
- 任务与对照：`contextual_cooperative_actor_critic`；`joint_policy_gradient`。
- 指标与预算：冻结随机联合策略的精确期望奖励；`environment_steps`。
- 设定：两智能体一步合作任务；局部策略、学习的联合critic与反事实基线。
- 范围：COMA的一步表格特例；无循环actor、多步critic或原论文任务。两方法共享critic学习规则；比较不是等梯度方差保证。
- 教材：[foundations/deep/multi-agent](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent/)

### QMIX monotone mixing

- 源码：[implementations/multiagent/qmix.py](../implementations/multiagent/qmix.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/qmix_cooperative/)。
- 任务与对照：`three_step_cooperative_value_learning`；`vdn_cooperative`。
- 指标与预算：冻结分散贪心策略的折扣回报；`environment_steps`。
- 设定：两agent三步Dec-POMDP，集中状态条件mixer、局部Q网络、经验回放与冻结目标。
- 范围：可运行的小型前馈QMIX实例；未复现循环SMAC系统，不证明持续多智能体泛化或相同计算成本。
- 教材：[foundations/deep/multi-agent](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent/)

### Bayes filter

- 源码：[implementations/extended_adaptation/bayes_filter.py](../implementations/extended_adaptation/bayes_filter.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-bayes_filter/)。
- 任务与对照：`two-state-hmm`；`extended-memoryless_filter`。
- 指标与预算：prequential_logloss_ema；`observations`。
- 设定：Bayes filter状态实验的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：已知二状态 HMM 的状态估计组件，不学习未知模型；value 是在线隐藏状态负对数损失的 EMA，不是冻结策略回报。隐藏状态仅用于诊断，不能用于滤波更新。
- 教材：[construction/state](https://yingwen.io/zh/continual-rl/construction/state/)

### RTU

- 源码：[implementations/extended_adaptation/rtu.py](../implementations/extended_adaptation/rtu.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-rtu/)。
- 任务与对照：`delayed-cue-sequences`；`extended-bptt`。
- 指标与预算：frozen_sequence_mse；`sequences`。
- 设定：RTU可训练旋转块实验的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：一对非线性旋转递归单元的精确序列内RTRL；参数在序列内固定、末端更新，不是跨更新无限流精确梯度，也非作者完整PPO系统。
- 教材：[construction/state](https://yingwen.io/zh/continual-rl/construction/state/)

### Metatrace

- 源码：[implementations/extended_adaptation/metatrace.py](../implementations/extended_adaptation/metatrace.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-metatrace/)。
- 任务与对照：`linear-trace-chain`；`extended-trace_ac`。
- 指标与预算：frozen_greedy_return；`environment_steps`。
- 设定：归一化Scalar Metatrace AC组件的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：normalized scalar Metatrace AC(lambda) 特例；忽略trace参数Jacobian的局部敏感度近似，不是per-parameter变体。冻结贪心回报不等同随机策略目标。
- 教材：[algorithms/meta](https://yingwen.io/zh/continual-rl/algorithms/meta/)

### Meta-gradient RL

- 源码：[implementations/extended_adaptation/meta_gradient.py](../implementations/extended_adaptation/meta_gradient.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-meta_gradient/)。
- 任务与对照：`meta-random-walk`；`extended-fixed_lambda`。
- 指标与预算：frozen_value_mse；`training_batches`。
- 设定：Meta-gradient lambda-return学习组件的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：预测random-walk上的lambda-return截断一批元梯度；没有actor/meta-RL完整控制系统，也不保留跨批无限图。
- 教材：[algorithms/meta](https://yingwen.io/zh/continual-rl/algorithms/meta/)

### MAML

- 源码：[implementations/extended_adaptation/maml.py](../implementations/extended_adaptation/maml.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-maml/)。
- 任务与对照：`linear-task-meta-learning`；`extended-joint_training`。
- 指标与预算：heldout_adapted_query_mse；`meta_training_batches`。
- 设定：多任务MAML监督训练的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：线性监督任务的二阶MAML训练，冻结测试含一次支持集适应；不是MAML-RL。
- 教材：[algorithms/meta](https://yingwen.io/zh/continual-rl/algorithms/meta/)

### ObGD 2024

- 源码：[implementations/extended_adaptation/obgd2024.py](../implementations/extended_adaptation/obgd2024.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-obgd2024/)。
- 任务与对照：`nonlinear-alternating-prediction`；`extended-streaming_td`。
- 指标与预算：frozen_value_mse；`environment_steps`。
- 设定：ObGD2024非线性TD组件的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：2024 ObGD优化器配合非线性TD(lambda)，不包含完整Stream-X网络/归一化系统，不是后续ObGD变体。
- 教材：[algorithms/streaming](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

### Intentional Updates

- 源码：[implementations/extended_adaptation/intentional.py](../implementations/extended_adaptation/intentional.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-intentional/)。
- 任务与对照：`linear-trace-chain`；`extended-trace_ac`。
- 指标与预算：frozen_greedy_return；`environment_steps`。
- 设定：Intentional AC流式组件的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：保留RMSProp/trace/sigma/剪裁与归一化核心的线性离散actor-critic；不复现连续控制/Atari系统。冻结贪心回报不等同随机策略目标。
- 教材：[algorithms/streaming](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

### CLEAR

- 源码：[implementations/extended_adaptation/clear.py](../implementations/extended_adaptation/clear.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-clear/)。
- 任务与对照：`context-bandit`；`extended-fresh_ac`。
- 指标与预算：冻结当前上下文策略期望奖励；`environment_steps`。
- 设定：CLEAR 单步回放组件的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：终止上下文bandit的单步V-trace/策略与价值克隆组件，有蓄水池；无多步分布式CLEAR系统。current return是随机策略期望，old_return为旧上下文诊断。
- 教材：[algorithms/retention](https://yingwen.io/zh/continual-rl/algorithms/retention/) · [algorithms/meta](https://yingwen.io/zh/continual-rl/algorithms/meta/)

### ReDo

- 源码：[implementations/extended_adaptation/redo.py](../implementations/extended_adaptation/redo.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-redo/)。
- 任务与对照：`switch-regression`；`extended-plain_mlp`。
- 指标与预算：冻结当前函数均方误差；`training_samples`。
- 设定：ReDo 激活回收组件的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：单隐层回归中的激活回收机制；SGD无动量、入边重置出边零；不包含Atari训练系统。新旧冲突函数MSE不能解读为可兼得的保持目标。
- 教材：[algorithms/plasticity](https://yingwen.io/zh/continual-rl/algorithms/plasticity/) · [algorithms/meta](https://yingwen.io/zh/continual-rl/algorithms/meta/)

### Continual Backpropagation

- 源码：[implementations/extended_adaptation/continual_backprop.py](../implementations/extended_adaptation/continual_backprop.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-continual_backprop/)。
- 任务与对照：`switch-regression`；`extended-plain_mlp`。
- 指标与预算：冻结当前函数均方误差；`training_samples`。
- 设定：Continual Backpropagation 贡献效用组件的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：单隐层回归的contribution效用GnT组件，有EMA偏差校正/成熟期/累计替换信用；无centered/adaptable效用或均值偏置补偿，不是完整持续RL系统。
- 教材：[algorithms/plasticity](https://yingwen.io/zh/continual-rl/algorithms/plasticity/) · [algorithms/meta](https://yingwen.io/zh/continual-rl/algorithms/meta/)

### Random Network Distillation

- 源码：[implementations/extended_knowledge/rnd_exploration.py](../implementations/extended_knowledge/rnd_exploration.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/rnd_exploration/)。
- 任务与对照：`ek_rnd_chain`；`rnd_no_bonus`。
- 指标与预算：在线已访问状态比例；`environment_steps`。
- 设定：12格反射链起点0；固定随机MLP12→12→3与独立预测MLP，MSE SGD .03；Q α=.2 γ=.99 ε=.15，内奖为更新前误差，预测器每交互更新；无外奖，访问覆盖为指标。
- 范围：实测RND神经预测/内奖驱动探索组件；无PPO、Atari、双critic或reward/observation归一化，非完整原工程。
- 教材：[algorithms/exploration](https://yingwen.io/zh/continual-rl/algorithms/exploration/)

### Learning progress curriculum

- 源码：[implementations/extended_knowledge/learning_progress.py](../implementations/extended_knowledge/learning_progress.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/learning_progress/)。
- 任务与对照：`ek_progress_regression`；`uniform_curriculum`。
- 指标与预算：冻结三任务期望预测MSE；`training_examples`。
- 设定：三个标量回归任务斜率1/-2/0，第三有σ=1不可约噪声；各任务SGD .05，20误差队列两个10窗均值差绝对值，ε=.2最高progress调度；固定真实斜率只用于环境/冻结MSE评价。
- 范围：学习进展调度组件，非原IAC区域分裂机器人系统；噪声可能欺骗progress，不能保证课程优于均匀。
- 教材：[algorithms/exploration](https://yingwen.io/zh/continual-rl/algorithms/exploration/)

### Recovery filter

- 源码：[implementations/extended_knowledge/recovery_filter.py](../implementations/extended_knowledge/recovery_filter.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/recovery_filter/)。
- 任务与对照：`ek_reset_filter`；`recovery_unfiltered`。
- 指标与预算：在线净奖励均值（含恢复失败成本）；`reset_trials`。
- 设定：三种前进行为，收益0/.4/1，真实恢复概率.98/.6/.05，仅环境知道；失败成本2。前30trial轮流收集恢复证据，之后Beta(1,1)后验均值过滤阈值.75、无合格选最高估计；奖励Q均值ε=.2；统计全部trial。
- 范围：从实际恢复结果学习的过滤组件；不是完整Leave No Trace双策略/神经reset critic，不提供统计安全证书，初始化和误估可能失败。
- 教材：[algorithms/exploration](https://yingwen.io/zh/continual-rl/algorithms/exploration/)

### UVFA 目标条件价值

- 源码：[implementations/extended_knowledge/uvfa_shared.py](../implementations/extended_knowledge/uvfa_shared.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/uvfa_shared/)。
- 任务与对照：`ek_goal_navigation`；`uvfa_tabular`。
- 指标与预算：七目标冻结贪心到达率；`environment_steps`。
- 设定：7格链，随机起点/目标，负一步代价；输入(s,g,s-g,|s-g|)共享MLP4→16→2，SGD .03，γ=.95，ε=.25；最长12步仅截断。每20步冻结遍历42对起点目标。
- 范围：仅目标条件TD共享逼近；不复现原论文低秩分解预训练或未见目标泛化结果。
- 教材：[construction/goals](https://yingwen.io/zh/continual-rl/construction/goals/)

### Hindsight Experience Replay

- 源码：[implementations/extended_knowledge/her_replay.py](../implementations/extended_knowledge/her_replay.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/her_replay/)。
- 任务与对照：`ek_her_navigation`；`goal_replay_plain`。
- 指标与预算：七目标冻结贪心到达率；`environment_steps`。
- 设定：7格确定性目标独立链，每回合至多12步；表格Q .2、γ=.95、ε=.4；完成轨迹随机future目标重新算reward/done，每交互另做2次回放；评价42起点目标。
- 范围：表格HER教学训练，非机器人DDPG/神经HER；未来目标只取已记录后继状态，假设动力学与goal无关；回放备份额外诊断。
- 教材：[construction/goals](https://yingwen.io/zh/continual-rl/construction/goals/)

### Subtask stopping

- 源码：[implementations/extended_knowledge/subtask_stopping.py](../implementations/extended_knowledge/subtask_stopping.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/subtask_stopping/)。
- 任务与对照：`ek_stopping_subtask`；`subtask_no_bonus`。
- 指标与预算：起点冻结子任务目标回报；`environment_steps`。
- 设定：0..6链，环境一步-.04，6终止得1；状态3到达停止bonus .8。均匀行为采样，α=.2 γ=.9；停止β由z(s')>=maxQ(s')，目标r+βz(s')+γ(1-β)V。冻结起点0评价最多30步。
- 范围：仅STOMP子任务策略/停止组件；bonus不写入主任务环境模型；不含完整STOMP生成/模型/规划闭环。
- 教材：[construction/goals](https://yingwen.io/zh/continual-rl/construction/goals/)

### Intra-option Q-learning

- 源码：[implementations/extended_knowledge/intra_option_q.py](../implementations/extended_knowledge/intra_option_q.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/intra_option_q/)。
- 任务与对照：`ek_intra_option_chain`；`option_on_policy_only`。
- 指标与预算：冻结call-and-return起点折扣回报；`environment_steps`。
- 设定：6格链起点0终点5，奖励1否则-.02；两固定随机option右动作概率.2/.8，β=.25，α=.1 γ=.9；行为执行active option，终止后ε=.2重新选择。每转移对全部option重要性加权备份，冻结精确call-and-return评价。
- 范围：固定option的片段价值学习组件；不学习内部策略/终止；环境终止不bootstrap，β=.25不是environment done；重要性比需正行为支持。
- 教材：[construction/options](https://yingwen.io/zh/continual-rl/construction/options/)

### Option-Critic

- 源码：[implementations/extended_knowledge/option_critic.py](../implementations/extended_knowledge/option_critic.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/option_critic/)。
- 任务与对照：`ek_option_critic_chain`；`option_critic_fixed_beta`。
- 指标与预算：冻结call-and-return起点折扣回报；`environment_steps`。
- 设定：6格链，r=-.02/终点1，γ=.9；两option softmax内部策略，sigmoid β，每primitive Q_U/QΩ critic α=.15、actor .03、termination .03；ε=.2选择option，termination梯度在arrival state应用，终点跳过。精确冻结策略评价。
- 范围：完整小表格actor/critic/termination训练组件；非原深网Atari方法，选项可能退化，未加deliberation cost/entropy正则。
- 教材：[construction/options](https://yingwen.io/zh/continual-rl/construction/options/)

### DIAYN skill information reward

- 源码：[implementations/extended_knowledge/diayn_tabular.py](../implementations/extended_knowledge/diayn_tabular.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/diayn_tabular/)。
- 任务与对照：`ek_diayn_skills`；`diayn_no_discriminator_reward`。
- 指标与预算：冻结技能与状态互信息；`environment_steps`。
- 设定：5格链两skill等先验，长度8 skill固定episode；状态包含remaining-time。表格soft Q α=.2 γ=.95，最大熵actor α=.05 温度.1，discriminator CE α=.15；内奖logq(z|s')-log.5。冻结精确全部8时刻占用，计算真实joint MI。
- 范围：联合discriminator/skill policy最大熵训练组件，非神经SAC DIAYN或下游迁移实验；观测含position/time，技能可能塌缩；MI不用训练分类器准确率代替。
- 教材：[construction/options](https://yingwen.io/zh/continual-rl/construction/options/)

### Option reward/endpoint model

- 源码：[implementations/extended_knowledge/option_model.py](../implementations/extended_knowledge/option_model.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/option_model/)。
- 任务与对照：`ek_option_model`；`option_model_monte_carlo`。
- 指标与预算：冻结reward/discounted-endpoint联合RMSE；`environment_steps`。
- 设定：7格链；固定option向右概率.8，β=.3（终点6强制止），γ=.9，r=-.02/终点1；α=.15逐primitive TD学R与Pγ。每次option终止随机起点0..5，与精确固定点RMSE比较。
- 范围：固定option模型组件，R仅真实环境reward、不混入子任务bonus；P行和Eγ^τ不归一；预算末未结束option不伪作终止，MC基线不更新该前缀。
- 教材：[construction/models](https://yingwen.io/zh/continual-rl/construction/models/)

### Option Value Iteration

- 源码：[implementations/extended_knowledge/option_value_iteration.py](../implementations/extended_knowledge/option_value_iteration.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/option_value_iteration/)。
- 任务与对照：`ek_option_planning`；`primitive_value_iteration`。
- 指标与预算：冻结精确最优价值最大误差；`model_backups`。
- 设定：7格链γ=.9，两个primitive及固定随机option（右概率.8，β=.3）；已知R/Pγ有限模型，随机状态原子最大化backup，每step一个state backup。真V由primitive最优右移闭式计算。
- 范围：独立有限option VI规划组件；不算环境训练，不把Pγ归一化或只乘一次γ；模型精确求解是预处理，模型估计非此实验内容。
- 教材：[construction/planning](https://yingwen.io/zh/continual-rl/construction/planning/)

### Bradley–Terry preference reward

- 源码：[implementations/extended_knowledge/preference_reward.py](../implementations/extended_knowledge/preference_reward.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/preference_reward/)。
- 任务与对照：`ek_preference_reward`；`preference_single_feature`。
- 指标与预算：冻结偏好期望交叉熵；`preference_labels`。
- 设定：长度3 binary-action轨迹，features=(右动作数,转向数)，teacher w=(1,-.7)仅生成偏好；每step一个随机轨迹对Bernoulli label，BT SGD .08；所有64轨迹对以teacher分布精确冻结期望交叉熵。
- 范围：独立线性片段偏好奖励拟合组件；不含人类交互收集/主动查询/策略优化闭环，reward平移不可辨识；精确teacher仅评价不用训练梯度。
- 教材：[foundations/reward-design](https://yingwen.io/zh/continual-rl/foundations/reward-design/)

### MaxEnt IRL

- 源码：[implementations/extended_knowledge/maxent_irl.py](../implementations/extended_knowledge/maxent_irl.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/maxent_irl/)。
- 任务与对照：`ek_maxent_paths`；`maxent_single_feature`。
- 指标与预算：冻结专家分布轨迹交叉熵；`gradient_updates`。
- 设定：4层binary-action确定性DAG，枚举16合法完整轨迹，features=(右边数,转向数)，固定64专家轨迹由θ=(.8,-.5)全局分布采样。每step全批NLL SGD .05，logZ=logsumexp轨迹回报；冻结真实专家分布交叉熵。
- 范围：有限确定性全轨迹MaxEnt IRL奖励学习实例；不是随机动力学causal entropy、无深网/大图采样；64演示数据生成成本另记，可能有采样偏差和奖励不可辨识。
- 教材：[foundations/reward-design](https://yingwen.io/zh/continual-rl/foundations/reward-design/)

### Intrinsic reward meta-gradient

- 源码：[implementations/extended_knowledge/intrinsic_meta_gradient.py](../implementations/extended_knowledge/intrinsic_meta_gradient.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/intrinsic_meta_gradient/)。
- 任务与对照：`ek_intrinsic_meta`；`intrinsic_fixed_reward`。
- 指标与预算：冻结更新后外在奖励期望；`meta_updates`。
- 设定：二动作bandit外奖(0,1)，内奖η·1(a=1)。logit θ用精确策略梯度 α=.1 内层一步，外层η沿更新后外奖精确链式梯度 β=.2；θ在线推进，η初值0。每step一个内外更新，额外梯度评价计数。
- 范围：可微内奖reward-design组件；精确有限bandit而非原A2C/PPO/延迟信用系统；不含多步unroll Hessian/state分布项，不能推出复杂任务优势。
- 教材：[foundations/reward-design](https://yingwen.io/zh/continual-rl/foundations/reward-design/)

### Reward Centering

- 源码：[implementations/extended_knowledge/reward_centering.py](../implementations/extended_knowledge/reward_centering.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/reward_centering/)。
- 任务与对照：`ek_centered_prediction`；`reward_uncentered`。
- 指标与预算：冻结重构原折扣价值RMSE；`environment_steps`。
- 设定：三状态确定性continuing环，奖励(9,10,11)，γ=.99 α=.1，on-policy V TD；参照c+=ηαδ (η=.05)，目标r-c+γV(s')；原值重构V+c/(1-γ)，解析真值从环几何级数；没有done或episodes。
- 范围：原2024论文value-based reward centering的表格prediction组件，仍γ<1非Differential TD/平均奖励控制；c的固定点与初始化耦合，不能当行为奖励均值或无偏目标reward-rate。
- 教材：[algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)

### 固定lambda-return对照

- 源码：[implementations/extended_adaptation/fixed_lambda.py](../implementations/extended_adaptation/fixed_lambda.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-fixed_lambda/)。
- 任务与对照：`meta-random-walk`；`extended-meta_gradient`。
- 指标与预算：frozen_value_mse；`training_batches`。
- 设定：固定lambda-return对照的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：小型教学任务，不是原论文性能复现；组件实验不计作完整原方法。冻结评价与训练更新分离，预算不代表相同计算成本。
- 教材：[algorithms/meta](https://yingwen.io/zh/continual-rl/algorithms/meta/)

### 一步IS TD对照

- 源码：[implementations/extended_adaptation/is_td.py](../implementations/extended_adaptation/is_td.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-is_td/)。
- 任务与对照：`offpolicy-one-state`；`extended-vtrace`。
- 指标与预算：frozen_target_value_mse；`environment_steps`。
- 设定：一步IS TD对照的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：小型教学任务，不是原论文性能复现；组件实验不计作完整原方法。冻结评价与训练更新分离，预算不代表相同计算成本。
- 教材：[algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

### 联合训练初始化对照

- 源码：[implementations/extended_adaptation/joint_training.py](../implementations/extended_adaptation/joint_training.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-joint_training/)。
- 任务与对照：`linear-task-meta-learning`；`extended-maml`。
- 指标与预算：heldout_adapted_query_mse；`meta_training_batches`。
- 设定：联合训练初始化对照的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：小型教学任务，不是原论文性能复现；组件实验不计作完整原方法。冻结评价与训练更新分离，预算不代表相同计算成本。
- 教材：[algorithms/meta](https://yingwen.io/zh/continual-rl/algorithms/meta/)

### 无记忆状态推断对照

- 源码：[implementations/extended_adaptation/memoryless_filter.py](../implementations/extended_adaptation/memoryless_filter.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/extended-memoryless_filter/)。
- 任务与对照：`two-state-hmm`；`extended-bayes_filter`。
- 指标与预算：prequential_logloss_ema；`observations`。
- 设定：无记忆状态推断对照的实际CPU教学实验；设置、公式和边界详见本文件。
- 范围：无记忆状态估计基线；value 是在线隐藏状态负对数损失的 EMA，不是冻结策略回报。隐藏状态仅用于诊断。
- 教材：[construction/state](https://yingwen.io/zh/continual-rl/construction/state/)

### 经验矩阵虚拟博弈对照

- 源码：[implementations/extended_classic/fictitious_play_reference.py](../implementations/extended_classic/fictitious_play_reference.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/fictitious_play_reference/)。
- 任务与对照：`extended_empirical_zero_sum_game`；`minimax_2x2`。
- 指标与预算：真实矩阵 Nash 间隙；`payoff_samples`。
- 设定：固定零和矩阵[[0.8,-0.2],[-0.4,0.6]]，每步均匀采一个矩阵元素、观测Gaussian噪声σ=0.2，经验均值初值0。解析minimax或虚拟博弈只用已学习矩阵；真实矩阵仅测量Nash间隙，不实现一般Markov博弈。
- 范围：仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。
- 教材：[foundations/deep/multi-agent](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent/)

### 固定λ局部混合对照

- 源码：[implementations/extended_classic/fixed_lambda_reference.py](../implementations/extended_classic/fixed_lambda_reference.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/fixed_lambda_reference/)。
- 任务与对照：`extended_local_lambda_mixture`；`greedy_adaptive_lambda`。
- 指标与预算：独立验证样本累计混合 MSE；`sample_pairs`。
- 设定：单状态局部混合：固定bootstrap预测0，未来回报N(1,0.25)。每单位含一个统计训练样本和一个独立验证样本；用旧均值/方差选择λ，再看新样本。仅学局部回报矩与λ，不实现整条TD链、状态相关网络和原方法的全部递归估计。
- 范围：仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。
- 教材：[algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

### 冻结参数前向 GTD2 对照

- 源码：[implementations/extended_classic/gradient_trace_forward_reference.py](../implementations/extended_classic/gradient_trace_forward_reference.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/gradient_trace_forward_reference/)。
- 任务与对照：`extended_frozen_gtd2_episodes`；`gradient_eligibility_traces`。
- 指标与预算：两状态非线性价值 RMSE；`episodes`。
- 设定：每episode固定两转移0→1→真终点，末奖励Bernoulli(0.5)，γ=0.9，λ=0.8。V=tanh(w·x)，H=tanh(h·x)，one-hot特征，w=h=0；两网络段内冻结、段间更新，测量真值[0.45,0.5]。只覆盖GTD2冻结参数资格迹方向，不覆盖TDC/TDRC和深度控制。
- 范围：仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。
- 教材：[algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

### 普通采样资格迹对照

- 源码：[implementations/extended_classic/sampled_eligibility_reference.py](../implementations/extended_classic/sampled_eligibility_reference.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/sampled_eligibility_reference/)。
- 任务与对照：`extended_markov_merge_credit`；`expected_eligibility_traces`。
- 指标与预算：累计平均信用方向 RMSE；`episodes`。
- 设定：两个等概率历史汇入完整Markov状态，历史资格为[0.72,0,1]或[0,0.72,1]；后继奖励独立取0或2。主预测参数冻结0，条件期望迹由历史样本均值学习；比较累计信用方向与真实均值[0.36,0.36,1]。不训练完整value网络或递归ET混合。
- 范围：仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。
- 教材：[algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

### 不使用代价约束的策略梯度对照

- 源码：[implementations/extended_classic/unconstrained_policy_reference.py](../implementations/extended_classic/unconstrained_policy_reference.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/unconstrained_policy_reference/)。
- 任务与对照：`extended_constrained_bandit`；`cmdp_primal_dual`。
- 指标与预算：最优可行动作概率绝对误差；`environment_steps`。
- 设定：单状态两动作reward Bernoulli均值0.2/0.9，cost=动作1指示，预算0.4，独立真实样本。π1=sigmoid(logit)，初值logit=0、乘子0；解析可行最优π1=0.4只用于测量。对照忽略约束，不是等可行解竞争；不实现一般多状态CMDP占据测度求解。
- 范围：仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。
- 教材：[foundations/deep/constraints](https://yingwen.io/zh/continual-rl/foundations/deep/constraints/)

### 仅动作熵无技能内奖基线

- 源码：[implementations/extended_knowledge/diayn_no_discriminator_reward.py](../implementations/extended_knowledge/diayn_no_discriminator_reward.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/diayn_no_discriminator_reward/)。
- 任务与对照：`ek_diayn_skills`；`diayn_tabular`。
- 指标与预算：冻结技能与状态互信息；`environment_steps`。
- 设定：透明消融/参照：调用 diayn_tabular.py 的训练流程，仅设置 intrinsic=False。5格链两skill等先验，长度8 skill固定episode；状态包含remaining-time。表格soft Q α=.2 γ=.95，最大熵actor α=.05 温度.1，discriminator CE α=.15；内奖logq(z|s')-log.5。冻结精确全部8时刻占用，计算真实joint MI。
- 范围：baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。
- 教材：[construction/options](https://yingwen.io/zh/continual-rl/construction/options/)

### 不重标目标的相同回放基线

- 源码：[implementations/extended_knowledge/goal_replay_plain.py](../implementations/extended_knowledge/goal_replay_plain.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/goal_replay_plain/)。
- 任务与对照：`ek_her_navigation`；`her_replay`。
- 指标与预算：七目标冻结贪心到达率；`environment_steps`。
- 设定：透明消融/参照：调用 her_replay.py 的训练流程，仅设置 hindsight=False。7格确定性目标独立链，每回合至多12步；表格Q .2、γ=.95、ε=.4；完成轨迹随机future目标重新算reward/done，每交互另做2次回放；评价42起点目标。
- 范围：baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。
- 教材：[construction/goals](https://yingwen.io/zh/continual-rl/construction/goals/)

### 仅外奖η=0基线

- 源码：[implementations/extended_knowledge/intrinsic_fixed_reward.py](../implementations/extended_knowledge/intrinsic_fixed_reward.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/intrinsic_fixed_reward/)。
- 任务与对照：`ek_intrinsic_meta`；`intrinsic_meta_gradient`。
- 指标与预算：冻结更新后外在奖励期望；`meta_updates`。
- 设定：透明消融/参照：调用 intrinsic_meta_gradient.py 的训练流程，仅设置 learn_reward=False。二动作bandit外奖(0,1)，内奖η=0即只用外奖。logit θ用精确策略梯度 α=.1 内层一步，外层梯度虽计算但不更新η；θ在线推进。每step一个内层更新，额外梯度评价计数。
- 范围：baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。
- 教材：[foundations/reward-design](https://yingwen.io/zh/continual-rl/foundations/reward-design/)

### 仅右动作特征MaxEnt拟合基线

- 源码：[implementations/extended_knowledge/maxent_single_feature.py](../implementations/extended_knowledge/maxent_single_feature.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/maxent_single_feature/)。
- 任务与对照：`ek_maxent_paths`；`maxent_irl`。
- 指标与预算：冻结专家分布轨迹交叉熵；`gradient_updates`。
- 设定：透明消融/参照：调用 maxent_irl.py 的训练流程，仅设置 full_features=False。4层binary-action确定性DAG，枚举16合法完整轨迹，features=(右边数,转向数)，固定64专家轨迹由θ=(.8,-.5)全局分布采样。每step全批NLL SGD .05，logZ=logsumexp轨迹回报；冻结真实专家分布交叉熵。
- 范围：baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。
- 教材：[foundations/reward-design](https://yingwen.io/zh/continual-rl/foundations/reward-design/)

### Option-Critic固定β=.5消融

- 源码：[implementations/extended_knowledge/option_critic_fixed_beta.py](../implementations/extended_knowledge/option_critic_fixed_beta.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/option_critic_fixed_beta/)。
- 任务与对照：`ek_option_critic_chain`；`option_critic`。
- 指标与预算：冻结call-and-return起点折扣回报；`environment_steps`。
- 设定：透明消融/参照：调用 option_critic.py 的训练流程，仅设置 learn_beta=False。6格链，r=-.02/终点1，γ=.9；两option softmax内部策略，sigmoid β，每primitive Q_U/QΩ critic α=.15、actor .03、termination .03；ε=.2选择option，termination梯度在arrival state应用，终点跳过。精确冻结策略评价。
- 范围：baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。
- 教材：[construction/options](https://yingwen.io/zh/continual-rl/construction/options/)

### 完整option蒙特卡洛模型基线

- 源码：[implementations/extended_knowledge/option_model_monte_carlo.py](../implementations/extended_knowledge/option_model_monte_carlo.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/option_model_monte_carlo/)。
- 任务与对照：`ek_option_model`；`option_model`。
- 指标与预算：冻结reward/discounted-endpoint联合RMSE；`environment_steps`。
- 设定：透明消融/参照：调用 option_model.py 的训练流程，仅设置 td=False。7格链；固定option向右概率.8，β=.3（终点6强制止），γ=.9，r=-.02/终点1；α=.15逐primitive TD学R与Pγ。每次option终止随机起点0..5，与精确固定点RMSE比较。
- 范围：baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。
- 教材：[construction/models](https://yingwen.io/zh/continual-rl/construction/models/)

### 仅执行option的on-policy更新基线

- 源码：[implementations/extended_knowledge/option_on_policy_only.py](../implementations/extended_knowledge/option_on_policy_only.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/option_on_policy_only/)。
- 任务与对照：`ek_intra_option_chain`；`intra_option_q`。
- 指标与预算：冻结call-and-return起点折扣回报；`environment_steps`。
- 设定：透明消融/参照：调用 intra_option_q.py 的训练流程，仅设置 all_options=False。6格链起点0终点5，奖励1否则-.02；两固定随机option右动作概率.2/.8，β=.25，α=.1 γ=.9；行为执行active option，终止后ε=.2重新选择。每转移对全部option重要性加权备份，冻结精确call-and-return评价。
- 范围：baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。
- 教材：[construction/options](https://yingwen.io/zh/continual-rl/construction/options/)

### 仅右动作特征偏好拟合基线

- 源码：[implementations/extended_knowledge/preference_single_feature.py](../implementations/extended_knowledge/preference_single_feature.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/preference_single_feature/)。
- 任务与对照：`ek_preference_reward`；`preference_reward`。
- 指标与预算：冻结偏好期望交叉熵；`preference_labels`。
- 设定：透明消融/参照：调用 preference_reward.py 的训练流程，仅设置 full_features=False。长度3 binary-action轨迹，features=(右动作数,转向数)，teacher w=(1,-.7)仅生成偏好；每step一个随机轨迹对Bernoulli label，BT SGD .08；所有64轨迹对以teacher分布精确冻结期望交叉熵。
- 范围：baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。
- 教材：[foundations/reward-design](https://yingwen.io/zh/continual-rl/foundations/reward-design/)

### 仅primitive的相同模型备份基线

- 源码：[implementations/extended_knowledge/primitive_value_iteration.py](../implementations/extended_knowledge/primitive_value_iteration.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/primitive_value_iteration/)。
- 任务与对照：`ek_option_planning`；`option_value_iteration`。
- 指标与预算：冻结精确最优价值最大误差；`model_backups`。
- 设定：透明消融/参照：调用 option_value_iteration.py 的训练流程，仅设置 include_options=False。7格链γ=.9，两个primitive及固定随机option（右概率.8，β=.3）；已知R/Pγ有限模型，随机状态原子最大化backup，每step一个state backup。真V由primitive最优右移闭式计算。
- 范围：baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。
- 教材：[construction/planning](https://yingwen.io/zh/continual-rl/construction/planning/)

### 不经过恢复过滤的基线

- 源码：[implementations/extended_knowledge/recovery_unfiltered.py](../implementations/extended_knowledge/recovery_unfiltered.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/recovery_unfiltered/)。
- 任务与对照：`ek_reset_filter`；`recovery_filter`。
- 指标与预算：在线净奖励均值（含恢复失败成本）；`reset_trials`。
- 设定：透明消融/参照：调用 recovery_filter.py 的训练流程，仅设置 filtered=False。三种前进行为，收益0/.4/1，真实恢复概率.98/.6/.05，仅环境知道；失败成本2。前30trial轮流收集恢复证据，之后Beta(1,1)后验均值过滤阈值.75、无合格选最高估计；奖励Q均值ε=.2；统计全部trial。
- 范围：baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。
- 教材：[algorithms/exploration](https://yingwen.io/zh/continual-rl/algorithms/exploration/)

### 不中心化的相同折扣TD基线

- 源码：[implementations/extended_knowledge/reward_uncentered.py](../implementations/extended_knowledge/reward_uncentered.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/reward_uncentered/)。
- 任务与对照：`ek_centered_prediction`；`reward_centering`。
- 指标与预算：冻结重构原折扣价值RMSE；`environment_steps`。
- 设定：透明消融/参照：调用 reward_centering.py 的训练流程，仅设置 centered=False。三状态确定性continuing环，奖励(9,10,11)，γ=.99 α=.1，on-policy V TD；参照c+=ηαδ (η=.05)，目标r-c+γV(s')；原值重构V+c/(1-γ)，解析真值从环几何级数；没有done或episodes。
- 范围：baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。
- 教材：[algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)

### 同预测器无内奖探索基线

- 源码：[implementations/extended_knowledge/rnd_no_bonus.py](../implementations/extended_knowledge/rnd_no_bonus.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/rnd_no_bonus/)。
- 任务与对照：`ek_rnd_chain`；`rnd_exploration`。
- 指标与预算：在线已访问状态比例；`environment_steps`。
- 设定：透明消融/参照：调用 rnd_exploration.py 的训练流程，仅设置 use_bonus=False。12格反射链起点0；固定随机MLP12→12→3与独立预测MLP，MSE SGD .03；Q α=.2 γ=.99 ε=.15，内奖为更新前误差，预测器每交互更新；无外奖，访问覆盖为指标。
- 范围：baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。
- 教材：[algorithms/exploration](https://yingwen.io/zh/continual-rl/algorithms/exploration/)

### 无停止bonus训练基线

- 源码：[implementations/extended_knowledge/subtask_no_bonus.py](../implementations/extended_knowledge/subtask_no_bonus.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/subtask_no_bonus/)。
- 任务与对照：`ek_stopping_subtask`；`subtask_stopping`。
- 指标与预算：起点冻结子任务目标回报；`environment_steps`。
- 设定：透明消融/参照：调用 subtask_stopping.py 的训练流程，仅设置 use_bonus=False。0..6链，环境一步-.04，6终止得1；状态3到达停止bonus .8。均匀行为采样，α=.2 γ=.9；停止β由z(s')>=maxQ(s')，目标r+βz(s')+γ(1-β)V。冻结起点0评价最多30步。
- 范围：baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。
- 教材：[construction/goals](https://yingwen.io/zh/continual-rl/construction/goals/)

### 均匀任务课程基线

- 源码：[implementations/extended_knowledge/uniform_curriculum.py](../implementations/extended_knowledge/uniform_curriculum.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/uniform_curriculum/)。
- 任务与对照：`ek_progress_regression`；`learning_progress`。
- 指标与预算：冻结三任务期望预测MSE；`training_examples`。
- 设定：透明消融/参照：调用 learning_progress.py 的训练流程，仅设置 adaptive=False。三个标量回归任务斜率1/-2/0，第三有σ=1不可约噪声；各任务SGD .05，20误差队列两个10窗均值差绝对值，ε=.2最高progress调度；固定真实斜率只用于环境/冻结MSE评价。
- 范围：baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。
- 教材：[algorithms/exploration](https://yingwen.io/zh/continual-rl/algorithms/exploration/)

### 逐目标表格Q基线

- 源码：[implementations/extended_knowledge/uvfa_tabular.py](../implementations/extended_knowledge/uvfa_tabular.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/uvfa_tabular/)。
- 任务与对照：`ek_goal_navigation`；`uvfa_shared`。
- 指标与预算：七目标冻结贪心到达率；`environment_steps`。
- 设定：透明消融/参照：调用 uvfa_shared.py 的训练流程，仅设置 tabular=True。7格链，随机起点/目标，负一步代价；输入(s,g,s-g,|s-g|)共享MLP4→16→2，SGD .03，γ=.95，ε=.25；最长12步仅截断。每20步冻结遍历42对起点目标。
- 范围：baseline仅禁用上述组件或限制特征，不是新的完整算法；与candidate使用同任务/指标/预算，额外计算差异见诊断。
- 教材：[construction/goals](https://yingwen.io/zh/continual-rl/construction/goals/)

### 联合critic策略梯度：无反事实基线

- 源码：[implementations/multiagent/joint_policy_gradient.py](../implementations/multiagent/joint_policy_gradient.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/joint_policy_gradient/)。
- 任务与对照：`contextual_cooperative_actor_critic`；`coma_contextual`。
- 指标与预算：冻结随机联合策略的精确期望奖励；`environment_steps`。
- 设定：与COMA教学实验相同的联合critic、局部actor和一步合作任务，基线为零。
- 范围：一步表格对照；并非对所有多智能体actor–critic的性能结论。
- 教材：[foundations/deep/multi-agent](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent/)

### VDN：加性价值分解对照

- 源码：[implementations/multiagent/vdn.py](../implementations/multiagent/vdn.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/vdn_cooperative/)。
- 任务与对照：`three_step_cooperative_value_learning`；`qmix_cooperative`。
- 指标与预算：冻结分散贪心策略的折扣回报；`environment_steps`。
- 设定：QMIX三步合作任务的加性混合对照；局部网络、目标同步与回放批次相同。
- 范围：前馈小型对照，不是完整SMAC工程。两模型结构容量与计算不同。
- 教材：[foundations/deep/multi-agent](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent/)

### Differential Dyna · learned empirical model

- 源码：[implementations/average_systems/differential_dyna.py](../implementations/average_systems/differential_dyna.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/differential_dyna/)。
- 任务与对照：`average_control_learned_model`；`differential_q_multistate`。
- 指标与预算：冻结贪心策略的精确平均奖励；`environment_steps`。
- 设定：每个真实转移更新Q与g并学习奖励/转移频数，再做5次经验模型期望备份。模型只含已访问状态动作。环境步和额外规划计算分别记录。
- 范围：经验模型、常数步长和规划调度的有限任务实验；不是原论文完整复现，不宣称非平稳或非线性收敛。相同环境步不代表相同计算量。
- 教材：[algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/) · [construction/planning](https://yingwen.io/zh/continual-rl/construction/planning/)

### On-policy reward-mean TD

- 源码：[implementations/average_systems/sample_mean_td.py](../implementations/average_systems/sample_mean_td.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/sample_mean_td/)。
- 任务与对照：`average_prediction_onpolicy`；`differential_td_onpolicy`。
- 指标与预算：参考状态对齐后的差分价值RMSE；`environment_steps`。
- 设定：同策略的真实奖励均值估计g，表格TD估计bias。它是正确的on-policy对照，不应被误称为off-policy方法。逐样本均值不适合直接追踪突然漂移。
- 范围：平稳遍历有限MDP、表格表示、常数步长、1200步教学预算；不是神经网络稳定性或非平稳收敛证明。
- 教材：[algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)

### Unanchored VI · offset comparison

- 源码：[implementations/average_systems/unnormalized_vi_multistate.py](../implementations/average_systems/unnormalized_vi_multistate.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/unnormalized_vi_multistate/)。
- 任务与对照：`average_known_model_planning`；`rvi_multistate`。
- 指标与预算：最优偏差参考对齐RMSE；`model_sweeps`。
- 设定：同已知模型Bellman备份，不去除公共增长。相对值仍可正确，绝对值线性长大；展示规范选择，不假装其必然控制失败。
- 范围：已知小模型、精确期望备份、遍历非周期任务。种子不影响确定性结果，不把模型扫描当真实交互。普通RVI在其他周期任务中可能振荡。
- 教材：[algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/) · [construction/planning](https://yingwen.io/zh/continual-rl/construction/planning/)

### Wrong behavior-rate TD · negative control

- 源码：[implementations/average_systems/wrong_behavior_mean_td.py](../implementations/average_systems/wrong_behavior_mean_td.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/wrong_behavior_mean_td/)。
- 任务与对照：`average_prediction_offpolicy`；`differential_td_offpolicy`。
- 指标与预算：参考状态对齐后的差分价值RMSE；`environment_steps`。
- 设定：有意错误的负对照：价值更新乘pi/b，但g直接平均b产生的奖励。展示动作校正不能补救错误的目标奖励率。不得作为推荐算法。
- 范围：平稳遍历有限MDP、表格表示、常数步长、1200步教学预算；不是神经网络稳定性或非平稳收敛证明。
- 教材：[algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)

### 闭环对照：累计模型

- 源码：[implementations/integrated_agents/integrated_cumulative_model.py](../implementations/integrated_agents/integrated_cumulative_model.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/integrated_cumulative_model/)。
- 任务与对照：`integrated_continuing_ring`；`integrated_recent_model`。
- 指标与预算：最近100个真实步的平均外部奖励；`environment_steps`。
- 设定：七状态持续随机环；外部奖励位置半程改变，agent 不知道变化时刻。仅把模型固定步长替换为累计样本均值。原子控制 ε=.2；技能内 ε=.1，最长6步，给定目标1/4。高层α=.1、奖励率增量=.002δ；每次宏动作结束做所设次数规划，并记录模型备份数。
- 范围：原创集成教学原型，非 STOMP/OaK 原论文复现。状态和子目标手工给定，学子任务策略但不学终止规则。常步长非平稳 SMDP 组合无收敛保证。模型预测误差只在已存在模型的完整宏动作上记录，不是无偏全空间误差。
- 教材：[construction/architectures](https://yingwen.io/zh/continual-rl/construction/architectures/) · [algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/) · [construction/options](https://yingwen.io/zh/continual-rl/construction/options/) · [construction/predictive-knowledge](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)

### 闭环对照：仅真实更新

- 源码：[implementations/integrated_agents/integrated_no_planning.py](../implementations/integrated_agents/integrated_no_planning.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/integrated_no_planning/)。
- 任务与对照：`integrated_continuing_ring`；`integrated_recent_model`。
- 指标与预算：最近100个真实步的平均外部奖励；`environment_steps`。
- 设定：七状态持续随机环；外部奖励位置半程改变，agent 不知道变化时刻。仅移除模型规划；仍学习模型并记录预测误差。原子控制 ε=.2；技能内 ε=.1，最长6步，给定目标1/4。高层α=.1、奖励率增量=.002δ；每次宏动作结束做所设次数规划，并记录模型备份数。
- 范围：原创集成教学原型，非 STOMP/OaK 原论文复现。状态和子目标手工给定，学子任务策略但不学终止规则。常步长非平稳 SMDP 组合无收敛保证。模型预测误差只在已存在模型的完整宏动作上记录，不是无偏全空间误差。
- 教材：[construction/architectures](https://yingwen.io/zh/continual-rl/construction/architectures/) · [algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/) · [construction/options](https://yingwen.io/zh/continual-rl/construction/options/) · [construction/predictive-knowledge](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)

### 闭环对照：仅原子动作

- 源码：[implementations/integrated_agents/integrated_primitive_only.py](../implementations/integrated_agents/integrated_primitive_only.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/integrated_primitive_only/)。
- 任务与对照：`integrated_continuing_ring`；`integrated_recent_model`。
- 指标与预算：最近100个真实步的平均外部奖励；`environment_steps`。
- 设定：七状态持续随机环；外部奖励位置半程改变，agent 不知道变化时刻。移除子任务与 options，保留同样平均奖励控制和模型规划；单步计算量也改变。原子控制 ε=.2；技能内 ε=.1，最长6步，给定目标1/4。高层α=.1、奖励率增量=.002δ；每次宏动作结束做所设次数规划，并记录模型备份数。
- 范围：原创集成教学原型，非 STOMP/OaK 原论文复现。状态和子目标手工给定，学子任务策略但不学终止规则。常步长非平稳 SMDP 组合无收敛保证。模型预测误差只在已存在模型的完整宏动作上记录，不是无偏全空间误差。
- 教材：[construction/architectures](https://yingwen.io/zh/continual-rl/construction/architectures/) · [algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/) · [construction/options](https://yingwen.io/zh/continual-rl/construction/options/) · [construction/predictive-knowledge](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)

### 近期模型的持续闭环

- 源码：[implementations/integrated_agents/integrated_recent_model.py](../implementations/integrated_agents/integrated_recent_model.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/integrated_recent_model/)。
- 任务与对照：`integrated_continuing_ring`；`integrated_no_planning`。
- 指标与预算：最近100个真实步的平均外部奖励；`environment_steps`。
- 设定：七状态持续随机环；外部奖励位置半程改变，agent 不知道变化时刻。固定步长模型 + 策略版本失效处理。原子控制 ε=.2；技能内 ε=.1，最长6步，给定目标1/4。高层α=.1、奖励率增量=.002δ；每次宏动作结束做所设次数规划，并记录模型备份数。
- 范围：原创集成教学原型，非 STOMP/OaK 原论文复现。状态和子目标手工给定，学子任务策略但不学终止规则。常步长非平稳 SMDP 组合无收敛保证。模型预测误差只在已存在模型的完整宏动作上记录，不是无偏全空间误差。
- 教材：[construction/architectures](https://yingwen.io/zh/continual-rl/construction/architectures/) · [algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/) · [construction/options](https://yingwen.io/zh/continual-rl/construction/options/) · [construction/predictive-knowledge](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)

### 诊断反例：沿用旧技能模型

- 源码：[implementations/integrated_agents/integrated_stale_options.py](../implementations/integrated_agents/integrated_stale_options.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/integrated_stale_options/)。
- 任务与对照：`integrated_continuing_ring`；`integrated_recent_model`。
- 指标与预算：最近100个真实步的平均外部奖励；`environment_steps`。
- 设定：七状态持续随机环；外部奖励位置半程改变，agent 不知道变化时刻。故意不在 option 策略变化后失效旧模型及高层 Q；不是推荐算法。原子控制 ε=.2；技能内 ε=.1，最长6步，给定目标1/4。高层α=.1、奖励率增量=.002δ；每次宏动作结束做所设次数规划，并记录模型备份数。
- 范围：原创集成教学原型，非 STOMP/OaK 原论文复现。状态和子目标手工给定，学子任务策略但不学终止规则。常步长非平稳 SMDP 组合无收敛保证。模型预测误差只在已存在模型的完整宏动作上记录，不是无偏全空间误差。
- 教材：[construction/architectures](https://yingwen.io/zh/continual-rl/construction/architectures/) · [algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/) · [construction/options](https://yingwen.io/zh/continual-rl/construction/options/) · [construction/predictive-knowledge](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)

### Baird：期望半梯度 TD

- 源码：[implementations/nonlinear_diagnostics/baird_expected_td.py](../implementations/nonlinear_diagnostics/baird_expected_td.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/baird_expected_td/)。
- 任务与对照：`baird_expected_diagnostic`；`baird_residual_gradient`。
- 指标与预算：log10(1 + 七状态价值 RMSE)；`expected_sweeps`。
- 设定：七状态八维固定线性特征；行为状态均匀、目标策略总到下状态。每步枚举一次全部状态的期望更新。
- 范围：已知模型的期望更新，不是仅一次真实交互的流式算法。特征冗余，参数不唯一。横轴一次 sweep 包含七个状态项；误差未裁剪。
- 教材：[foundations/learning-dynamics](https://yingwen.io/zh/continual-rl/foundations/learning-dynamics/) · [foundations/approximation/prediction](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/) · [foundations/approximation/off-policy](https://yingwen.io/zh/continual-rl/foundations/approximation/off-policy/)

### Baird：精确模型残差梯度

- 源码：[implementations/nonlinear_diagnostics/baird_residual_gradient.py](../implementations/nonlinear_diagnostics/baird_residual_gradient.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/baird_residual_gradient/)。
- 任务与对照：`baird_expected_diagnostic`；`baird_expected_td`。
- 指标与预算：log10(1 + 七状态价值 RMSE)；`expected_sweeps`。
- 设定：同一 Baird 星形问题，对精确 Bellman 残差平方求梯度；每步枚举七状态。
- 范围：确定性目标转移使此例无需 double sampling。下降 Bellman 残差不等于每步下降价值 RMSE，也不意味着对一般随机模型可直接使用单后继残差梯度。
- 教材：[foundations/learning-dynamics](https://yingwen.io/zh/continual-rl/foundations/learning-dynamics/) · [foundations/approximation/prediction](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/) · [foundations/approximation/off-policy](https://yingwen.io/zh/continual-rl/foundations/approximation/off-policy/)

### 独立双后继残差梯度

- 源码：[implementations/nonlinear_diagnostics/double_sample_residual.py](../implementations/nonlinear_diagnostics/double_sample_residual.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/double_sample_residual/)。
- 任务与对照：`double_sampling_diagnostic`；`single_sample_residual`。
- 指标与预算：真实 MSBE；`paired_successor_queries`。
- 设定：与单后继法使用相同两个生成模型查询；交叉相乘残差和独立后继梯度，得到 MSBE 无偏梯度估计。
- 范围：每步需要同一起点的条件独立后继。严格单轨迹通常不能直接满足；这不是可无条件移植到流式 RL 的方法。有限时间曲线仍含采样方差。
- 教材：[foundations/learning-dynamics](https://yingwen.io/zh/continual-rl/foundations/learning-dynamics/) · [foundations/approximation/prediction](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/) · [foundations/approximation/off-policy](https://yingwen.io/zh/continual-rl/foundations/approximation/off-policy/)

### 隔离非线性表示：零交叉更新对照

- 源码：[implementations/nonlinear_diagnostics/isolated_nonlinear.py](../implementations/nonlinear_diagnostics/isolated_nonlinear.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/isolated_nonlinear/)。
- 任务与对照：`nonlinear_interference_diagnostic`；`shared_nonlinear`。
- 指标与预算：两个固定条件目标的均匀 RMSE；`stream_observations`。
- 设定：相同 A/B/A 观测序列，每个上下文有独立四参数 tanh 网络。只更新当前上下文，另一个保持不动。
- 范围：需要预先知道上下文路由，参数量八而非四。它不展示知识迁移，不是共享模型普遍更差的证据；也不是持续智能体的完整方案。
- 教材：[foundations/learning-dynamics](https://yingwen.io/zh/continual-rl/foundations/learning-dynamics/) · [foundations/approximation/prediction](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/) · [algorithms/plasticity](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)

### 非线性 TD：冻结 50 步目标副本

- 源码：[implementations/nonlinear_diagnostics/lagged_nonlinear_td.py](../implementations/nonlinear_diagnostics/lagged_nonlinear_td.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/lagged_nonlinear_td/)。
- 任务与对照：`nonlinear_target_diagnostic`；`online_nonlinear_td`。
- 指标与预算：二状态真实价值 RMSE；`environment_steps`。
- 设定：相同二状态流；bootstrap 用独立参数副本，每 50 次更新后复制在线参数。当前预测仍逐步训练。
- 范围：目标副本多保存四参数。没有 replay 或控制；不等于 DQN 实现。冻结期间的局部回归稳定不推出跨复制时刻全局收敛。
- 教材：[foundations/learning-dynamics](https://yingwen.io/zh/continual-rl/foundations/learning-dynamics/) · [foundations/approximation/prediction](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/) · [algorithms/plasticity](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)

### 非线性 TD：即时自举目标

- 源码：[implementations/nonlinear_diagnostics/online_nonlinear_td.py](../implementations/nonlinear_diagnostics/online_nonlinear_td.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/online_nonlinear_td/)。
- 任务与对照：`nonlinear_target_diagnostic`；`lagged_nonlinear_td`。
- 指标与预算：二状态真实价值 RMSE；`environment_steps`。
- 设定：A/B 确定性交替，奖励 0/1，gamma=0.8。四参数 tanh 价值函数，单步半梯度 TD；不回放。
- 范围：小型固定策略预测，不含控制、不含分布变化。例子用于检查时序和目标漂移，不承诺目标网络更优或半梯度必发散。
- 教材：[foundations/learning-dynamics](https://yingwen.io/zh/continual-rl/foundations/learning-dynamics/) · [foundations/approximation/prediction](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/) · [algorithms/plasticity](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)

### 共享非线性表示：干扰诊断

- 源码：[implementations/nonlinear_diagnostics/shared_nonlinear.py](../implementations/nonlinear_diagnostics/shared_nonlinear.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/shared_nonlinear/)。
- 任务与对照：`nonlinear_interference_diagnostic`；`isolated_nonlinear`。
- 指标与预算：两个固定条件目标的均匀 RMSE；`stream_observations`。
- 设定：两个固定回归目标 +0.5/-0.5，观测依次只来自 A、B、A。一个四参数 tanh 网络共享表示，逐观测 SGD，无回放。
- 范围：构造性监督回归诊断，不是 RL benchmark，也不是长期可塑性丧失的证据。隔离对照用八参数和已知上下文路由，不是同参数量优劣比较。
- 教材：[foundations/learning-dynamics](https://yingwen.io/zh/continual-rl/foundations/learning-dynamics/) · [foundations/approximation/prediction](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/) · [algorithms/plasticity](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)

### 单后继残差梯度：有偏目标诊断

- 源码：[implementations/nonlinear_diagnostics/single_sample_residual.py](../implementations/nonlinear_diagnostics/single_sample_residual.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/single_sample_residual/)。
- 任务与对照：`double_sampling_diagnostic`；`double_sample_residual`。
- 指标与预算：真实 MSBE；`paired_successor_queries`。
- 设定：生成模型返回两个条件独立后继，特征各为 0 或 2，概率各 1/2。单样本法平均两个各自的残差梯度。
- 范围：限定评价起点 A 的采样目标；后继价值使用同一参数特征，并未声称是完整 MDP 的所有状态价值回归。两个方法每步同为两次模型查询。
- 教材：[foundations/learning-dynamics](https://yingwen.io/zh/continual-rl/foundations/learning-dynamics/) · [foundations/approximation/prediction](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/) · [foundations/approximation/off-policy](https://yingwen.io/zh/continual-rl/foundations/approximation/off-policy/)

### 冻结表示 GVF + 资格迹

- 源码：[implementations/streaming_composition/gvf_frozen_trace.py](../implementations/streaming_composition/gvf_frozen_trace.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/gvf_frozen_trace/)。
- 任务与对照：`sc_neural_gvf_interference`；`gvf_shared_trace`。
- 指标与预算：固定主问题的全状态预测 RMSE；`environment_steps`。
- 设定：四状态环上两个不同 π/c/γ 的问题；行为随机一次真实转移，各头普通 IS-TD(0.6) 更新一次。共享 2–6–2 tanh 网络，冻结随机 trunk，仅学习两线性输出，α=.01。中点仅辅助 cumulant 变号，主问题不变。
- 范围：不是 Horde 原作者实现或神经 GTD；普通 IS 半梯度无一般稳定保证。冻结随机特征受初始表达能力限制。 λ 改变信用也可能改变逼近固定点。辅助变号后旧迹不清除；它是环境信号变化，非预测终止。
- 教材：[construction/predictive-knowledge](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/) · [algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

### 分离表示 GVF + 资格迹

- 源码：[implementations/streaming_composition/gvf_separate_trace.py](../implementations/streaming_composition/gvf_separate_trace.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/gvf_separate_trace/)。
- 任务与对照：`sc_neural_gvf_interference`；`gvf_shared_trace`。
- 指标与预算：固定主问题的全状态预测 RMSE；`environment_steps`。
- 设定：四状态环上两个不同 π/c/γ 的问题；行为随机一次真实转移，各头普通 IS-TD(0.6) 更新一次。两个独立 2–6–1 tanh 网络，训练全部参数，α=.01。中点仅辅助 cumulant 变号，主问题不变。
- 范围：不是 Horde 原作者实现或神经 GTD；普通 IS 半梯度无一般稳定保证。分离网络参数更多，因此不是等容量对照。 λ 改变信用也可能改变逼近固定点。辅助变号后旧迹不清除；它是环境信号变化，非预测终止。
- 教材：[construction/predictive-knowledge](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/) · [algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

### 共享神经 GVF + TD(0)

- 源码：[implementations/streaming_composition/gvf_shared_td0.py](../implementations/streaming_composition/gvf_shared_td0.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/gvf_shared_td0/)。
- 任务与对照：`sc_neural_gvf_interference`；`gvf_shared_trace`。
- 指标与预算：固定主问题的全状态预测 RMSE；`environment_steps`。
- 设定：四状态环上两个不同 π/c/γ 的问题；行为随机一次真实转移，各头普通 IS-TD(0) 更新一次。共享 2–6–2 tanh 网络，训练全部参数，α=.01。中点仅辅助 cumulant 变号，主问题不变。
- 范围：不是 Horde 原作者实现或神经 GTD；普通 IS 半梯度无一般稳定保证。共享更新可改变另一问题的预测。 λ 改变信用也可能改变逼近固定点。辅助变号后旧迹不清除；它是环境信号变化，非预测终止。
- 教材：[construction/predictive-knowledge](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/) · [algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

### 共享神经 GVF + 资格迹

- 源码：[implementations/streaming_composition/gvf_shared_trace.py](../implementations/streaming_composition/gvf_shared_trace.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/gvf_shared_trace/)。
- 任务与对照：`sc_neural_gvf_interference`；`gvf_frozen_trace`。
- 指标与预算：固定主问题的全状态预测 RMSE；`environment_steps`。
- 设定：四状态环上两个不同 π/c/γ 的问题；行为随机一次真实转移，各头普通 IS-TD(0.6) 更新一次。共享 2–6–2 tanh 网络，训练全部参数，α=.01。中点仅辅助 cumulant 变号，主问题不变。
- 范围：不是 Horde 原作者实现或神经 GTD；普通 IS 半梯度无一般稳定保证。共享更新可改变另一问题的预测。 λ 改变信用也可能改变逼近固定点。辅助变号后旧迹不清除；它是环境信号变化，非预测终止。
- 教材：[construction/predictive-knowledge](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/) · [algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)

### 非线性完整回报 Gradient MC

- 源码：[implementations/streaming_composition/neural_gradient_mc.py](../implementations/streaming_composition/neural_gradient_mc.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/neural_gradient_mc/)。
- 任务与对照：`sc_neural_delayed_prediction`；`neural_td_trace`。
- 指标与预算：全状态预测 RMSE；`environment_steps`。
- 设定：与 TD(λ) 同一六步链及 2–6–1 tanh 网络；α=.03。保存当前 episode 状态，结束后构造完整回报，按时间正序逐样本更新网络。真实终点无 bootstrap。
- 范围：完整回报对照使用最多六条转移并等待终点；不满足严格无轨迹缓冲的流式约束。预算末尾未完成 episode 不更新、不伪装成终止。固定六步上界才使此例缓冲有界。
- 教材：[algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/) · [foundations/approximation](https://yingwen.io/zh/continual-rl/foundations/approximation/)

### 神经 SMDP：按真实时长折扣

- 源码：[implementations/streaming_composition/neural_option_smdp.py](../implementations/streaming_composition/neural_option_smdp.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/neural_option_smdp/)。
- 任务与对照：`sc_neural_option_duration`；`neural_option_wrong_duration`。
- 指标与预算：相对真实 option Bellman 最优值的 RMSE；`environment_steps`。
- 设定：四状态环，两给定闭环 options，持续时间 1–2 primitive 步；高层均匀探索，2–6–2 tanh 网络进行 option Q-learning，α=.025，无 replay 或 target network。累计每步奖励并以 γ^τ bootstrap。固定真实 primitive 步预算，绘图与精确小模型 Q* 比较。
- 范围：正确 SMDP 目标不保证非线性 Q-learning 全局收敛。给定 option、Markov 状态、固定表示结构；没有实现 option 发现或神经 Option-Critic。未完成 option 不在预算末尾伪终止。
- 教材：[construction/options](https://yingwen.io/zh/continual-rl/construction/options/)

### 错误诊断：忽略 option 时长

- 源码：[implementations/streaming_composition/neural_option_wrong_duration.py](../implementations/streaming_composition/neural_option_wrong_duration.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/neural_option_wrong_duration/)。
- 任务与对照：`sc_neural_option_duration`；`neural_option_smdp`。
- 指标与预算：相对真实 option Bellman 最优值的 RMSE；`environment_steps`。
- 设定：四状态环，两给定闭环 options，持续时间 1–2 primitive 步；高层均匀探索，2–6–2 tanh 网络进行 option Q-learning，α=.025，无 replay 或 target network。故意将 continuation γ^τ 错写成 γ。固定真实 primitive 步预算，绘图与精确小模型 Q* 比较。
- 范围：这是故意错误的负对照，不是可推荐研究算法。给定 option、Markov 状态、固定表示结构；没有实现 option 发现或神经 Option-Critic。未完成 option 不在预算末尾伪终止。
- 教材：[construction/options](https://yingwen.io/zh/continual-rl/construction/options/)

### 非线性半梯度 TD(0)

- 源码：[implementations/streaming_composition/neural_td0.py](../implementations/streaming_composition/neural_td0.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/neural_td0/)。
- 任务与对照：`sc_neural_delayed_prediction`；`neural_td_trace`。
- 指标与预算：全状态预测 RMSE；`environment_steps`。
- 设定：固定策略六步链，末步 Bernoulli(.7) 奖励，γ=.9；2–6–1 tanh 网络，α=.03，λ=0，逐条转移更新，不重放。记录资格范数与同一旧输入的梯度变化。
- 范围：不是 True-online 非线性扩展，也不是神经 GTD；旧梯度迹与当前网络重算的迹不同。固定 Markov 输入、平稳目标、小型同策略预测。与 MC 的交互预算相同，但梯度计算和更新时刻不同。
- 教材：[algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/) · [construction/predictive-knowledge](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)

### 非线性半梯度 TD(λ)

- 源码：[implementations/streaming_composition/neural_td_trace.py](../implementations/streaming_composition/neural_td_trace.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/neural_td_trace/)。
- 任务与对照：`sc_neural_delayed_prediction`；`neural_td0`。
- 指标与预算：全状态预测 RMSE；`environment_steps`。
- 设定：固定策略六步链，末步 Bernoulli(.7) 奖励，γ=.9；2–6–1 tanh 网络，α=.03，λ=0.8，逐条转移更新，不重放。记录资格范数与同一旧输入的梯度变化。
- 范围：不是 True-online 非线性扩展，也不是神经 GTD；旧梯度迹与当前网络重算的迹不同。固定 Markov 输入、平稳目标、小型同策略预测。与 MC 的交互预算相同，但梯度计算和更新时刻不同。
- 教材：[algorithms/credit-assignment](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/) · [construction/predictive-knowledge](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)

### 持续走廊：400步后冻结参数，保留记忆

- 源码：[implementations/learner_control/frozen_parameters.py](../implementations/learner_control/frozen_parameters.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/learner_control_frozen/)。
- 任务与对照：`continuing_information_recovery_corridor`；`learner_control_online`。
- 指标与预算：完整生命期每原始步的实际奖励；`environment_steps`。
- 设定：前400步与在线方法完全相同；之后冻结该检查点的Q与gain。世界不重置，探测得到的新线索仍进入记忆，真实探测/恢复代价仍计入收益。
- 范围：冻结既有参数的检查点匹配诊断，不是所有固定控制器的最优基线；活动记忆仍更新，因此也不是冻结完整内部状态。
- 教材：[algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/) · [algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)

### 持续走廊：在线差分 Q 与线索记忆

- 源码：[implementations/learner_control/online_differential_q.py](../implementations/learner_control/online_differential_q.py)。
- 页面：[问题、源码与真实对照](https://yingwen.io/zh/continual-rl/code/learner_control_online/)。
- 任务与对照：`continuing_information_recovery_corridor`；`learner_control_frozen`。
- 指标与预算：完整生命期每原始步的实际奖励；`environment_steps`。
- 设定：持续走廊含有成本的信息动作与四步失败恢复；600步路况改变。每个原始步更新差分Q和gain，线索记忆独立递推。
- 范围：有限教学任务，不是一般偏离遗憾估计或非平稳收敛证明；对照在400步冻结参数而保留记忆及世界；无不可逆陷阱。
- 教材：[algorithms/control](https://yingwen.io/zh/continual-rl/algorithms/control/) · [foundations/objectives](https://yingwen.io/zh/continual-rl/foundations/objectives/) · [algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)

## 仅综合 lab 公式核（尚无独立实验）（0 项）

## 作者工程固定源码入口（未完整训练）（3 项）

### DreamerV3

https://github.com/danijar/dreamerv3/tree/e01491fad6434b2245a3b8ca201dd7faedcc458c

源码：[integrations/author_projects.json](../integrations/author_projects.json)。
范围：作者维护reimplementation；固定源码与入口已核验，未安装训练
- 教材：[construction/planning](https://yingwen.io/zh/continual-rl/construction/planning/)

### TD-MPC2

https://github.com/nicklashansen/tdmpc2/tree/e9f59321933cbc8e11a002b842adc7d4ffae8ff1

源码：[integrations/author_projects.json](../integrations/author_projects.json)。
范围：官方项目固定源码已核验，未训练
- 教材：[construction/planning](https://yingwen.io/zh/continual-rl/construction/planning/)

### DiscoRL

https://github.com/google-deepmind/disco_rl/tree/9059a29f7121d60948f25ef165e08e050e9399c8

源码：[integrations/author_projects.json](../integrations/author_projects.json)。
范围：minimal JAX harness meta-eval/meta-train；不包含原始全部搜索系统，未训练
- 教材：[algorithms/meta](https://yingwen.io/zh/continual-rl/algorithms/meta/)

## 正文涉及，尚未接入（27 项）

### UCT / MCTS

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[construction/planning](https://yingwen.io/zh/continual-rl/construction/planning/)

### RVI Q-learning

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[algorithms/average-reward](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)

### Stream-X / Streaming actor–critic

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[algorithms/streaming](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

### StreamingOptimizer 2026

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[algorithms/streaming](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

### GVFN

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[construction/state](https://yingwen.io/zh/continual-rl/construction/state/)

### GRU / LSTM 通用状态网络

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：已有RTRL/TBPTT标量RNN教学；不等同通用GRU/LSTM控制训练。
- 教材：[construction/state](https://yingwen.io/zh/continual-rl/construction/state/)

### UORO

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[construction/state](https://yingwen.io/zh/continual-rl/construction/state/)

### LRU

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[construction/state](https://yingwen.io/zh/continual-rl/construction/state/)

### RL²

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[algorithms/meta](https://yingwen.io/zh/continual-rl/algorithms/meta/)

### PEARL

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[algorithms/meta](https://yingwen.io/zh/continual-rl/algorithms/meta/)

### Learned Policy Gradient

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[algorithms/meta](https://yingwen.io/zh/continual-rl/algorithms/meta/)

### DRAGO

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[algorithms/retention](https://yingwen.io/zh/continual-rl/algorithms/retention/)

### NaP

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[algorithms/plasticity](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)

### C-CHAIN

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[algorithms/plasticity](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)

### Parseval regularization

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[algorithms/plasticity](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)

### LSAC

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[algorithms/exploration](https://yingwen.io/zh/continual-rl/algorithms/exploration/)

### HIQL

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[construction/goals](https://yingwen.io/zh/continual-rl/construction/goals/)

### MaestroMotif

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[construction/goals](https://yingwen.io/zh/continual-rl/construction/goals/)

### Eigenoptions

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[construction/options](https://yingwen.io/zh/continual-rl/construction/options/)

### ROD

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[construction/options](https://yingwen.io/zh/continual-rl/construction/options/)

### DCEO

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[construction/options](https://yingwen.io/zh/continual-rl/construction/options/)

### ALLO

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[construction/options](https://yingwen.io/zh/continual-rl/construction/options/)

### METRA

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[construction/options](https://yingwen.io/zh/continual-rl/construction/options/)

### Laplacian Keyboard

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[construction/options](https://yingwen.io/zh/continual-rl/construction/options/)

### Default Representation

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[construction/models](https://yingwen.io/zh/continual-rl/construction/models/)

### ALPS

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[construction/planning](https://yingwen.io/zh/continual-rl/construction/planning/)

### CPO

教材编号正文；尚无独立完整实现或本次固定作者工程接入

范围：未接入；禁止生成本次实测曲线。
- 教材：[foundations/deep/constraints](https://yingwen.io/zh/continual-rl/foundations/deep/constraints/)
