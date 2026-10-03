# 现代 DRL：可学习表示与稳定控制

运行时，智能体与世界形成交互闭环；外部设计者选择奖励、初始化、数据权限、调参与预算。经典方法、神经表示和持续学习描述不同维度，可以共同用于同一智能体。下面按教学先修组织，不把三册视为互斥问题类，也不要求读完全部分支才开始研究。

[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：比较完整学习器](https://yingwen.io/zh/continual-rl/algorithms/control/)

基础目录包含表格方法、函数逼近与深度核心算法。深度拓展中的部分可观测、探索、回报分布、离线数据、模型、约束和多智能体是并列研究分支，可按问题选择。

深度强化学习用神经网络表示价值、策略或模型。它与持续学习可以同时成立，也可以研究固定任务或离线数据。本册先解释 DQN、actor–critic、PPO 与连续控制的核心接口，再按部分可观测性、探索、分布、离线数据、模型、约束和多智能体等条件选择并列研究分支。

学习目标：

- 能区分 target、prediction、stop-gradient 和实际更新参数。
- 能解释 replay、target network、Double、GAE、PPO clipping 各改变哪个环节。
- 能把一个完整训练循环与机制小实验区分开，并定位原始实现。

## 1 · 神经网络改变了哪些条件

把表格换成网络，为什么不是只换一种存储方式？

神经网络 $Q_\theta(s,a)$ 同时学习表示与输出。一次更新可以改变许多未访问状态的预测，后续数据也可能改变已学特征。控制又会改变行为和数据分布，因此网络看到的输入、监督目标和自身表示都可能移动。监督学习中固定标签、独立采样的直觉只能部分沿用。

计算 TD 平方误差时，先用当前约定构造目标 $y$，再只对预测求导，是一个明确算法选择。若让梯度穿过下一状态的 bootstrap，会变成另一条更新。即使目标暂时冻结，优化器只是在拟合这批目标；损失下降也不等于真实策略回报上升。

deadly triad 描述函数逼近、bootstrap、off-policy 的交互风险。replay、target network 与梯度限制是具体稳定化机制，不是把网络恢复成表格的证明。调试时先检查 terminal mask、target 梯度、张量维度和数据分布，再谈更大网络。

$$
L(\theta)=\tfrac12\mathbb E_{(s,a,r,s')\sim\mathcal D}\bigl[Q_\theta(s,a)-\operatorname{sg}(y)\bigr]^2,\qquad \nabla_\theta L=\mathbb E[(Q_\theta-y)\nabla_\theta Q_\theta]
$$

sg 表示停止梯度；D 是训练采样分布。此梯度是对冻结目标的回归梯度，不是对完整 Bellman 误差所有依赖路径的梯度。

### 动手与核对

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

## 2 · DQN：经验重放、目标网络与 Double target

如何避免价值网络用迅速变化的自身预测反复训练自己？

DQN 将新转移放入 replay buffer，从中抽取小批量样本训练。这样可以重用经验并改变相邻训练样本的相关性，但缓冲区分布不等于当前策略分布。目标网络 $\bar\theta$ 在若干更新内保持不变，使回归标签变化更慢；随后再同步或缓慢跟踪在线参数。

普通 DQN 用目标网络同时选择最大动作和评价它。当估计噪声与最大值选择耦合时，会出现过估计倾向。Double DQN 用在线网络选动作、目标网络评价该动作，分开两个角色。它不要求两个网络完全独立，也不保证所有误差都无偏。

完整循环还包括探索策略、初始采样期、每步更新次数、target 同步周期、真实终止处理和评价协议。比较算法时这些环节不能无意中改变。原始 DQN 代码有 Atari 图像预处理和旧 Torch 依赖；学习核心逻辑可以从 NeuralQLearner 开始，但不能把核心文件独立运行当成完整实验。

$$
\begin{aligned}y_{\rm DQN}&=r+\gamma(1-d)\max_aQ_{\bar\theta}(s',a),\\a^*&=\arg\max_aQ_\theta(s',a),\\y_{\rm Double}&=r+\gamma(1-d)Q_{\bar\theta}(s',a^*).\end{aligned}
$$

d 只表示该转移真实终止。两种 target 都停止梯度；预算截断不应不加区分地设 d=1。

### 动手与核对

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

## 3 · 策略梯度与 actor–critic

不通过枚举最大 Q，能否直接改善动作分布？

参数化策略 $\pi_\theta(a\mid s)$ 直接输出动作分布。对轨迹概率求导，可把回报的梯度写成动作 log-probability 的梯度乘回报。环境的未知转移概率不必可微。REINFORCE 用完整回报估计该权重；减去只依赖状态的 baseline，不改变相应期望梯度，却可能降低方差。

advantage 定义为 $A_\pi(s,a)=q_\pi(s,a)-v_\pi(s)$，表示这个动作比当前策略在该状态的平均选择好多少。actor–critic 用 critic 的价值预测构造更早取得、通常方差较低的更新权重，例如 TD error 或 advantage 估计。actor 改变策略，critic 评价随之变化的策略，二者形成耦合闭环。critic 偏差仍会影响 actor。

策略梯度的采样分布依赖当前策略。对同一批旧数据无限优化 log-probability surrogate，不能保证继续沿正确的回报梯度移动。实现时还要区分用于策略梯度的回报权重和用于报告性能的 episode return。

$$
\nabla_\theta J=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t\nabla_\theta\log\pi_\theta(A_t\mid S_t)\bigl(G_t-b(S_t)\bigr)\right]
$$

这里 J 是从固定初始分布出发的有限轨迹折扣回报，$G_t$ 从时刻 t 重新计折扣。baseline 在 actor 更新中按固定数值使用；连续动作要使用联合概率密度。

### 动手与核对

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

## 4 · GAE 与 PPO：估计优势和限制批内策略变化

怎样利用一批轨迹多次更新，而不忽略数据来自旧策略？

GAE 将一连串 TD errors 按 $(\gamma\lambda)^k$ 加权，得到 advantage 估计。它在一步 critic bootstrap 与较长回报之间调节依赖。rollout 因时间限制结束时可以 bootstrap 最后价值，但不能把 reset 后另一条轨迹的 TD error 接入同一条 GAE 递推。

PPO 保存采样策略的 log-probability，定义新旧动作概率比 $r_t(\theta)$，并用 clipped surrogate 做多轮小批量更新。对正优势，过度增加动作概率不再得到额外目标收益；对负优势，过度降低概率也不再得到额外收益。这不是对所有状态施加硬 KL 约束。

GAE 是优势估计器，PPO 是策略更新规则，可以分别讨论。完整实现需要冻结采样时的旧 log-probability 和本轮优势、正确处理边界、训练 critic，并监测 KL 与 clip fraction。Spinning Up 的 PPO 还提供 KL 超阈值时提前停止策略更新的实现。

$$
\begin{aligned}\hat A_t&=\sum_{k=0}^{K-1}(\gamma\lambda)^k\delta_{t+k},\qquad r_t(\theta)=\frac{\pi_\theta(A_t\mid S_t)}{\pi_{\rm old}(A_t\mid S_t)},\\L^{\rm clip}(\theta)&=\mathbb E[\min(r_t\hat A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)].\end{aligned}
$$

GAE 求和只沿同一轨迹到可用边界；terminal 与 truncation 决定最后 TD error 的 bootstrap。PPO 对上述 L 做梯度上升。

### 动手与核对

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

## 5 · 连续控制：TD3 与最大熵 SAC

动作是连续向量时，怎样完成价值改善中的最大化？

连续动作无法逐项枚举。确定性 actor 可以近似输出高 Q 动作，通过 critic 对动作的梯度训练。TD3 使用两个 critic 的较小目标值、延迟 actor 更新，并在 target action 附近加入裁剪噪声。这些机制分别处理价值高估、critic 尚不准确时的策略跟随，以及对狭窄误差峰的利用。

SAC 不只是给 TD3 加随机动作。它把策略熵纳入目标，在回报和保持动作分布之间权衡，学习随机 actor。critic target 包含下一动作的负 log-density；actor 则比较熵成本与 critic 估值。熵系数与奖励尺度共同决定偏好，不能把含熵目标和原始回报当作同一量。

实践中，连续策略常由高斯变量经 tanh 变换得到。计算 log-probability 必须包含变换的 Jacobian 修正；直接截断动作再沿用高斯密度不是同一分布。TD3 的 target smoothing 噪声与 SAC 的策略随机性也不是同一个参数。

$$
\begin{aligned}y_{\rm TD3}&=r+\gamma(1-d)\min_iQ_{\bar\phi_i}(s',\tilde a'),\\y_{\rm SAC}&=r+\gamma(1-d)\left[\min_iQ_{\bar\phi_i}(s',a')-\alpha_{\rm ent}\log\pi_\theta(a'\mid s')\right],\quad a'\sim\pi_\theta,\\L_\pi^{\rm SAC}&=\mathbb E_{s\sim\mathcal D,a\sim\pi_\theta}[\alpha_{\rm ent}\log\pi_\theta(a\mid s)-\min_iQ_{\phi_i}(s,a)].\end{aligned}
$$

TD3 的 ã′ 是目标 actor 输出加裁剪噪声再裁剪到动作范围；SAC 的 a′ 从当前随机 actor 采样。critic target 停止梯度，SAC actor loss 对重参数化动作路径求导。

### 动手与核对

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

## 6 · 世界模型与规划：预测误差怎样进入决策

学会预测环境，为什么还不等于学会最优行动？

世界模型把观测历史编码为内部状态，并预测下一内部状态、奖励或观测。它可以支持模型预测控制：每步从当前状态比较候选动作序列，只执行首个动作，再用真实观测重规划。也可以在模型中产生轨迹，以此训练价值与策略。

两种路径都需要明确模型在哪个数据分布上可靠。策略优化会主动寻找高预测奖励的区域，也可能找到模型最不准确的区域。一步预测误差小，并不保证长时想象轨迹正确；模型损失也不能代替真实环境中的控制评价。

Dyna、MPC、MCTS 与 Dreamer 因此应按接口区分：Dyna 强调真实学习与模拟 backup 的结合；MPC 强调执行一小段再重规划；MCTS 把搜索预算分配到树上；Dreamer 类方法从潜在模型想象训练行为。它们可能共享部件，但不是可互换的名称。

$$
(a_0^*,\ldots,a_{H-1}^*)\in\arg\max_{a_{0:H-1}}\mathbb E_{\hat p}\!\left[\sum_{k=0}^{H-1}\gamma^k\hat r_k+\gamma^H\hat V(z_H)\right]
$$

这是有限时域 MPC 的一个形式。z 是模型状态，H 是规划长度，末端价值弥补规划截断。实际只执行 $a_0$，再重新估计状态并规划。

### 动手与核对

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

## 7 · 部分可观测性与循环状态

当前图像一样，最优动作却不同，网络应当学什么？

环境状态 $S_t$ 与智能体收到的观测 $O_t$ 不一定相同。若过去线索影响现在的动作，必须从历史构造内部状态 $h_t$。已知模型时可以维护对隐藏状态的 belief；模型未知时，可用循环网络学习一个压缩历史。这个状态是行动所需信息的表示，不是价值函数的别名。

循环活动和参数有不同生命周期。活动记录当前经历中的情境，参数记录如何根据经历更新活动及行动。仅让 hidden state 持续变化，并不能说明算法在长期积累知识；反过来，频繁清空活动也可能破坏任务所需的记忆。

BPTT 沿展开的计算图传播参数对后续状态的影响。截断 BPTT 节省内存，却会删去截断点之前的梯度路径；前向记忆仍可保留，因此“记住了信息”和“能学会记住信息”必须分开。RTRL 用前向敏感度维护这些导数，但一般计算代价很高。

$$
h_t=f_\theta(h_{t-1},O_t,A_{t-1}),\qquad \frac{\partial h_t}{\partial\theta}=\left.\frac{\partial f_\theta}{\partial\theta}\right|_{h_{t-1}}+\frac{\partial f_\theta}{\partial h_{t-1}}\frac{\partial h_{t-1}}{\partial\theta}
$$

这是固定参数运行序列时的链式法则。第一项是本步直接影响，第二项是过去参数经递归活动的影响。在线每步改变参数后，历史活动并未按新参数重算，必须另外说明导数近似。

### 动手与核对

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

下一步：不必先读完所有深度研究分支才研究 CRL。选定问题后，再确定神经表示是否必要、信息与数据权限是什么，以及学习状态怎样长期保留。持续学习评价的是这些模块共同组成的完整学习器，而不只评价训练结束时的一组网络权重。
