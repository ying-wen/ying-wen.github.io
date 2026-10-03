# 深度 RL 的机制接口：模型、记忆、离线数据与实验

改变数据来源或 agent state 后，哪些推导和实现条件必须重新检查？

## 本章内容

- 由 Bellman 算子解释模型误差的放大。
- 区分 recurrent hidden state 与跨时间参数梯度。
- 识别离线外推问题，并设计可解释的 CRL 对照。

<a id="problem-definition"></a>

## 本章的问题定义

将数学更新映射到有数据缓存、模型状态、终止标志和优化器的可执行学习器。

### 给定条件与符号

- 完整实现状态 $z_t$，包括参数、优化器、记忆、缓存和随机数状态。
- 转移输入 $e_t$ 与更新算子 $F$；预算、初始化和评测协议固定。

### 需要求解的对象

实现与所声明更新语义一致的学习过程，并定位模型、记忆和离线数据带来的额外误差。

### 信息与数据权限

训练数据、评估数据和目标网络有明确读写时刻；测试环境信息不得泄漏回训练选择。

$$
z_{t+1}=F(z_t,e_t),\qquad F^{(T)}(z_0,e_{0:T-1})=F^{(T-k)}(z_k,e_{k:T-1})
$$

右式表示完整状态恢复后应与不间断运行一致，在同样输入和随机性条件下比较。它是实现正确性要求，不是算法效能结论。

### 成立条件与解的含义

- 数值容差或逐位一致标准须预先选择。
- 数据协议改变时，需要重新检查目标估计，而不只是让张量维度正确。

判断准则：通过小问题、梯度、边界和状态恢复测试；算法收益另用独立实验评价。

### 适用边界

- 用测试通过替代论文效能复现或算法比较。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)：把估计量、实验随机单位与实现检查接在一起。

- 改变信息或数据协议 · [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)：递归记忆的读写和截断决定实际可用历史与梯度。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

同一个公式可能因版本、终止、缓存或梯度路径不同而实现成不同算法。

### 本章的核心思路

把每个更新视为显式状态转移，分别验证输入、时序、边界和输出。

1. [量化模型误差如何影响备份](#lesson-derive)：模型预测误差需要通过规划目标解释，不能只看预测 loss。

2. [分开记忆递推和参数信用](#lesson-memory)：前向保留历史与反向传播到历史是两件事。

3. [分层验证完整训练过程](#lesson-algorithm)：先机制测试，再短运行，再冻结协议做独立评价。

结论与条件：测试只对已覆盖输入和属性提供证据；没有测试到的分布变化仍可能失败。

### 相关方法改变了什么

- 单元测试 / 学习实验：分别检验算子语义与随机交互中的表现。

- 参数 checkpoint / 完整状态恢复：后者还需优化器、记忆、缓存、计数器与随机状态。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### Bellman 递推

当前价值等于即时奖励加折扣后的后继价值；预测目标与最优控制目标不同。

### 函数与梯度

神经网络把参数和输入映射为输出。链式法则必须指明哪些量参与求导，哪些量作为固定目标。

### 经验分布

on-policy 数据由当前策略产生；off-policy 数据可来自旧策略，但能否正确复用取决于算法目标和数据覆盖。

<a id="lesson-setting"></a>

## 1 · 算法首先是一份数据与时间约定

DQN、PPO、TD3 与 SAC 的损失不能脱离数据采集过程理解。每条样本需要知道：动作来自哪个策略，下一观测是否属于同一个 episode，终止是否属于任务定义，参数何时更新。引入模型、recurrent 网络或离线数据时，改变的正是这些约定。

| 机制 | 新增加的信息 | 必须维持的关系 |
| --- | --- | --- |
| 模型学习 | 预测奖励、下一状态或潜在状态 | 规划使用的模型应与任务目标和动作语义一致 |
| recurrent state | 历史顺序、初始 hidden state、边界 | 网络状态重建与参数梯度截断分别定义 |
| offline RL | 固定数据集与行为覆盖 | 不能依赖未授权的新环境交互纠正外推 |
| CRL | 持续时间线、变化机制、资源与 reset 权限 | 学习期间的损失和适应成本不能只由最终冻结评估代替 |

数据记录也包括观测归一化器与奖励变换的状态。若训练过程中更新归一化统计，旧 replay 的原始观测与已归一化特征会产生不同语义；必须选择保存哪一种，并保持实验一致。

<a id="lesson-derive"></a>

## 2 · 模型误差为什么会沿自举放大

固定策略，真实 Bellman 算子为 $TV=r+\gamma PV$，模型算子为 $\hat TV=\hat r+\gamma\hat PV$。若二者都是 sup 范数下的 $\gamma$ 压缩映射，固定点分别为 $V,\hat V$。假设模型在真实价值处的单步 Bellman 误差不超过 $\epsilon$。

$$
\begin{aligned}\|\hat V-V\|_\infty&=\|\hat T\hat V-TV\|_\infty\\&\le\|\hat T\hat V-\hat TV\|_\infty+\|\hat TV-TV\|_\infty\\&\le\gamma\|\hat V-V\|_\infty+\epsilon\end{aligned}
$$

第一步插入并减去同一项，第二步使用模型算子的压缩性与误差假设。

$$
\|\hat V-V\|_\infty\le\frac{\epsilon}{1-\gamma}
$$

这是固定策略、给定算子误差条件下的界，不是像素预测误差对所有控制算法的保证。

例如 $\epsilon=0.02$ 时，$\gamma=0.9$ 给出 0.2 的上界，$\gamma=0.99$ 则为二。降低单步重建误差不一定同比降低 Bellman 误差，因为与奖励和决策有关的细小差异可能被重建指标掩盖。模型是否足够好，要看它用于什么预测与规划。

Dyna 将模型生成的转移用于价值更新。Dreamer 在学得的潜在动力学中生成轨迹，用它们训练 actor–critic。TD-MPC2 则结合潜在模型、价值估计和在线动作序列规划。这些方法共享模型学习，但使用模型的计算位置不同：训练时想象与决策时规划不能混为一类 update。

<a id="lesson-memory"></a>

## 3 · recurrent state 与时间梯度是两层对象

$$
\begin{gathered}h_t=f_\theta(h_{t-1},o_t,a_{t-1})\\E_t=\frac{d h_t}{d\theta}=\frac{\partial f_\theta}{\partial h_{t-1}}E_{t-1}+\frac{\partial f_\theta}{\partial\theta}\end{gathered}
$$

$h$ 是当下的历史摘要，$E$ 是摘要对参数的敏感度。保留 $h$ 的数值并不表示保留了 $E$。

完整 BPTT 保存一段前向计算图并向过去回传；RTRL 在线递推敏感度，但一般稠密网络的成本很高。TBPTT 在片段边界 detach hidden state，相当于截断更早参数影响，不是清空 hidden state 数值。两种 reset 不能互换。

配套最小例子固定递归系数 $a$，学习输入系数 $\theta$：$h_t=ah_{t-1}+\theta x_t$，初始 $h_0=0$。于是 $E_t=aE_{t-1}+x_t$。取 $a=0.5,\theta=0.2$，输入 $(1,0,0)$，得到状态 $(0.2,0.1,0.05)$ 与敏感度 $(1,0.5,0.25)$。对输入系数做中心差分即可检验最后的 0.25；若改为学习递归系数，直接导数项应变为 $h_{t-1}$。

从 replay 抽取 recurrent 序列还需要初始状态。用一段 burn-in 重建 hidden state 能缓解直接用零状态的失配，但使用当前参数重建的状态不一定等于采集时的状态。跨任务的变化和长时记忆会放大这一差别。模型记忆、记忆重建和梯度窗口应分别记录。

<a id="lesson-offline"></a>

## 4 · 为什么 off-policy 不自动等于 offline 可用

Off-policy 指学习目标策略与数据行为策略可以不同；offline 还要求数据固定，不再通过新行动纠正估计。Q-learning 或 SAC 在未见动作上高估时，在线环境可能提供反证，离线数据却可能永远没有该动作的结果。因而旧经验复用能力不等于离线控制的可靠性。

$$
\begin{gathered}L_{\rm conservative}(Q)=\log\sum_a e^{Q(s,a)}-\mathbb E_{a\sim\mathcal D(\cdot\mid s)}Q(s,a)\\\frac{\partial L_{\rm conservative}}{\partial Q(s,a)}=\operatorname{softmax}(Q(s,\cdot))_a-p_{\mathcal D}(a\mid s)\end{gathered}
$$

这是离散动作 CQL 型正则的核心形式。完整 CQL 还包含 Bellman loss、权重选择和相应理论条件；连续动作通常需要采样近似。

若数据只含动作零，而 $Q(s,\cdot)=(0,10)$，正则约为 $10.000045$。它对动作一的导数接近一，对动作零的导数接近负一。因此梯度下降压低未支持的高值，并相对提高数据动作。正则本身只约束相对偏好，不能替代奖励监督。

保守性可能减少外推，也可能压制真正有用但数据很少的动作。对任意未覆盖结果都保持正确是不可能由损失形式单独保证的；数据覆盖和任务假设仍然重要。CRL 中的历史 replay 常兼有旧分布、策略覆盖不足和动力学变化，三种误差要分别诊断。

<a id="lesson-algorithm"></a>

## 5 · 一个可审计的训练与评估过程

**算法：评估策略可以冻结，但主训练过程是否继续适应必须独立说明。不能把冻结策略测试误写为单生命期学习表现。**

1. 定义目标、动作单位、终止条件及允许 reset 的对象。
1. 固定训练预算、独立运行种子、调参数据与报告指标。
1. 记录每次真实转移与对应网络、归一化器、行为概率信息。
1. 按算法约定更新；单独计数真实步、模型步和梯度步。
1. 用独立环境评估；评估数据不回流到训练或调参。
1. 保存完整时间曲线与资源成本；报告失败运行和区间。
1. 若研究 CRL，再加入变化后的恢复、遗忘与长期在线收益。

独立种子是不同学习过程。相邻时间窗口来自同一个不断变化的 agent，不能简单当作独立重复试验。跨任务汇总还需要可比的分数定义，并对不同任务和种子的变异分别处理。rliable 提供 bootstrap 区间、性能轮廓与稳健汇总入口，不替代实验协议。

<a id="lesson-example"></a>

## 6 · 小环境的终止定义也是模型的一部分

DeadlineChain 将剩余时间写入观察，十二步截止属于任务本身的真终止。

```python
class DeadlineChain:
    """Five positions; right reaches reward 1. The observed deadline is 12.

    Goal or deadline is genuine task termination. A rollout boundary in PPO
    is only a sampling cutoff and does not reset this environment.
    """
    def reset(self):
        self.position, self.remaining = 0, 12
        return self.observe()

    def observe(self):
        return torch.tensor([float(i == self.position) for i in range(5)]
                            + [self.remaining/12.0], dtype=torch.float32)

    def step(self, action):
        self.position = min(4, max(0, self.position + (1 if action else -1)))
        self.remaining -= 1
        reached = self.position == 4
        return self.observe(), (1.0 if reached else -0.02), reached or self.remaining == 0


def evaluate(actor, episodes=10):
    scores = []
    with torch.no_grad():
        for _ in range(episodes):
            env, score, done = DeadlineChain(), 0.0, False
            observation = env.reset()
            while not done:
                action = int(actor(observation).argmax())
                observation, reward, done = env.step(action)
                score += reward
            scores.append(score)
    return sum(scores)/len(scores)
```

内置任务有五个位置，动作是左或右，位置四为目标。到达目标奖励一，其余步骤奖励 −0.02。十二步是任务期限，因此必须在状态中保留剩余时间；不这样做，同一个位置的最优价值会因未观察的剩余步数不同而变化。最短四步到达的无折扣回报是 0.94。

PPO 的 rollout 长度只是优化批次长度。batch 在 episode 中间结束时，代码保留环境状态并 bootstrap；它不凭空制造任务终止。与之相反，真实期限耗尽时 continuation value 为零。两类边界使用相同的 done 变量会导致错误标签。

DQN 示例使用折扣 0.99，PPO 示例对这个有限时域任务使用一。二者用于各自流程教学，不构成控制变量的算法性能对比。若要比较，必须先统一优化目标、网络预算、采样预算与评估规则，再使用多个独立种子。

<a id="lesson-code"></a>

## 7 · 从可复查算子到真实训练

recurrent 敏感度、CQL 型正则和模型 Bellman 误差界的数值核。

```python
def recurrent_derivative(a, b, inputs):
    state, sensitivity = 0.0, 0.0
    for observation in inputs:
        state = a*state + b*observation
        sensitivity = a*sensitivity + observation
    return state, sensitivity


def conservative_penalty(q_values, data_action):
    maximum = max(q_values)
    return maximum + math.log(sum(math.exp(q-maximum) for q in q_values))-q_values[data_action]


def bootstrap_bias_bound(one_step_error, gamma):
    if one_step_error < 0 or not 0 <= gamma < 1:
        raise ValueError('nonnegative error and discount below one required')
    return one_step_error/(1-gamma)
```

标准库运行：python3 deep_textbook_lab.py demo 与 python3 deep_textbook_lab.py test。PyTorch 运行：python3 deep_textbook_train.py test；python3 deep_textbook_train.py dqn --steps 1600 --seed 0；python3 deep_textbook_train.py ppo --epochs 16 --seed 0。训练脚本和标准库脚本必须同目录，依赖由 deep_requirements.txt 固定。

这套小实验包含实际 DQN/Double DQN 与 PPO 训练，VPG、DDPG、TD3、SAC 的可微更新核，TRPO 的数值 CG/回溯和 categorical Fisher 核。它不包含 Dreamer、TD-MPC2、完整 CQL 或 recurrent 深度训练流水线。后几类方法的原论文和作者代码用于继续研究，不能由单步核测试替代基准复现。

<a id="lesson-branches"></a>

## 8 · 向持续强化学习迁移时的具体问题

| 观察到的问题 | 需要分开的解释 | 最小区分实验 |
| --- | --- | --- |
| 变化后回报下降 | 旧 replay 失配、critic 标签漂移、策略覆盖不足 | 固定网络容量，对照新数据与混合数据，并记录关键动作覆盖 |
| 训练越久越难学 | 表示退化、优化器状态、目标尺度改变 | 重置输出层、优化器或部分单元，分别测同一新目标的学习曲线 |
| 长延迟奖励学不到 | 历史状态不足、信用窗口太短、探索不到 | 给充分状态、延长梯度窗口、提供探索轨迹，分别作 oracle 对照 |
| 模型规划反而变差 | 模型误差、规划分布偏离数据、计算预算变化 | 真实模型与学习模型对照，并固定总计算预算 |

每个对照只改变一个机制更容易解释，但 oracle 也需要标出信息权限。例如给真实隐状态可以定位表示瓶颈，却不能作为同等信息下的部署方法；允许重置环境的实验也不能直接声称解决不可重置单生命期控制。

<a id="lesson-check"></a>

## 9 · 练习与答案

- 问：重建误差下降是否保证模型控制改善？答：不保证。Bellman 相关误差、规划覆盖和奖励敏感维度可能仍有偏差。
- 问：detach hidden state 是否清除 agent 的记忆？答：不清除数值记忆，但截断其对早先参数计算的梯度依赖。
- 问：CQL 正则能从零数据推断未见动作奖励吗？答：不能。它限制乐观外推，不产生缺失的因果信息。
- 实验：把 DeadlineChain 的剩余时间从观察删除，在不同剩余步数的同一位置比较价值目标；答案是出现状态混叠，不能简单归因于 DQN 优化失败。



<a id="chapter-code"></a>

## 下载与运行

标准库数值核验；完整小任务训练另需 deep_textbook_train.py 与 PyTorch。

[下载 deep_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/deep_textbook_lab.py)

```sh
python3 deep_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Hafner et al. · Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)：DreamerV3 的世界模型与想象 actor–critic。

- [Hafner · dreamerv3](https://github.com/danijar/dreamerv3)：作者维护的重实现；仓库明确说明基于 DreamerV2，不等同于原始内部实验代码。

- [Hansen et al. · TD-MPC2](https://arxiv.org/abs/2310.16828)：潜在模型与连续控制规划。

- [Hansen · tdmpc2](https://github.com/nicklashansen/tdmpc2)：作者代码与训练配置；与 Dreamer 的模型使用位置对照。

- [Kumar et al. · Conservative Q-Learning](https://arxiv.org/abs/2006.04779)：offline 分布偏移、保守价值目标及其条件。

- [Kumar · CQL](https://github.com/aviralkumar2907/CQL)：作者实现；包含离散与连续控制实验入口。

- [Agarwal et al. · Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)：有限种子下的可靠评价、区间与汇总。

- [rliable](https://github.com/google-research/rliable)：论文配套评价代码，不是训练算法。

- [Williams & Zipser · A Learning Algorithm for Continually Running Fully Recurrent Neural Networks](https://doi.org/10.1162/neco.1989.1.2.270)：RTRL 原始论文；敏感度在线递推的来源。

<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)
- [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)
- [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)
- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)
- [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)
- [可塑性与特征更新](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

对应原始材料：Sutton & Barto §8、§11、§17；CQL §3–4；Deep RL at the Edge of the Statistical Precipice。本文为原创讲解，原书、论文与上游代码保留各自许可。
