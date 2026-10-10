# 深度 RL 的机制接口：模型、记忆、离线数据与实验

现代深度强化学习 · 第 6 章

改变数据来源或 agent state 后，哪些推导和实现条件必须重新检查？

## 本章内容

- 由 Bellman 算子解释模型误差的放大。
- 区分 recurrent hidden state 与跨时间参数梯度。
- 识别离线外推问题，并设计可解释的 CRL 对照。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [深度价值学习：DQN、Double DQN 与目标的时间顺序](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/)：掌握采样、回放、标签和参数更新的基本循环。
- [策略梯度：从轨迹概率到 GAE 与 actor–critic](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/)：掌握rollout、优势和策略更新之间的数据关系。


### Bellman 递推

当前价值等于即时奖励加折扣后的后继价值；预测目标与最优控制目标不同。

### 函数与梯度

神经网络把参数和输入映射为输出。链式法则必须指明哪些量参与求导，哪些量作为固定目标。

### 经验分布

on-policy 数据由当前策略产生；off-policy 数据可来自旧策略，但能否正确复用取决于算法目标和数据覆盖。

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


<a id="lesson-setting"></a>

## 1 · 算法首先是一份数据与时间约定

前几章的深度方法把输入变成可学习的表示，再用这些表示预测和行动。在固定任务中，可以反复采样、重放和优化，直到找到较好的策略。若同一个学习器要长期运行，问题进一步改变：今天有用的表示明天可能需要扩充，旧预测可能对应已经改变的世界，而改进策略又会使原来常见的经验不再出现。学习能力本身必须在这些变化中保留下来。

DQN、PPO、TD3 与 SAC 的损失不能脱离数据采集过程理解。每条样本需要知道：动作来自哪个策略，下一观测是否属于同一个 episode，终止是否属于任务定义，参数何时更新。引入模型、recurrent 网络或离线数据时，改变的正是这些约定。

| 机制 | 新增加的信息 | 必须维持的关系 |
| --- | --- | --- |
| 模型学习 | 预测奖励、下一状态或潜在状态 | 规划使用的模型应与任务目标和动作语义一致 |
| recurrent state | 历史顺序、初始 hidden state、边界 | 网络状态重建与参数梯度截断分别定义 |
| offline RL | 固定数据集与行为覆盖 | 不能依赖未授权的新环境交互纠正外推 |
| CRL | 持续时间线、变化机制、资源与 reset 权限 | 学习期间的损失和适应成本不能只由最终冻结评估代替 |

数据记录也包括观测归一化器与奖励变换的状态。若训练过程中更新归一化统计，旧 replay 的原始观测与已归一化特征会产生不同语义；必须选择保存哪一种，并保持实验一致。

当环境、动作推理与梯度更新在不同机器上并行，这份约定还必须描述策略版本和经验等待时间。接续本章的 [大规模训练：算法与系统怎样共同设计](/zh/continual-rl/foundations/deep/systems/) 从 A3C、IMPALA 与 V-trace，讲到 OpenAI Five 的采样与推理分工、SEED RL 的集中推理，以及 GEAR 的经验存取；逐一解释吞吐、延迟、离策略与资源利用之间的关系。

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

$$
\|\hat TV-TV\|_\infty\le\epsilon_r+\gamma\epsilon_P\|V\|_\infty,\quad \epsilon_P=\sup_s\sum_{s'}|\hat P(s'\mid s)-P(s'\mid s)|
$$

有限状态中，令奖励均值的最大误差为 $\epsilon_r$，先用三角不等式，再对转移行的绝对误差求和，即得到这个单步条件。若 $|r|\le R_{\max}$，则 $\|V\|_\infty\le R_{\max}/(1-\gamma)$。平均训练误差小并不提供这里所有状态上的统一误差界。

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

![完整梯度、截断梯度与清空记忆下，前向 hidden state 和参数敏感度的不同演化](https://yingwen.io/crl-figures/concept-depth-classic-memory-gradients.svg)

在第一步之后施加边界操作，其他设置相同。蓝色是前向记忆；橙色圆面积表示对输入系数 $\theta$ 的敏感度。detach 将已得到的 $h_1$ 当作常数，后两步的记忆数值仍为 0.1、0.05；清空记忆才使它们变成 0。参数在三步内保持不变，图中未做优化。[精确递推代码](/crl-code/figures/classic-visual-depth.mjs)。

从 replay 抽取一段中途开始的序列，还需要解释其初始状态从哪里来。保存的状态由采集时的参数形成；当前网络若从历史起点重新展开，可能得到另一个状态。预热（burn-in）先读一段前缀，再在后缀计算损失，用来减轻这项起点失配。前缀如果没有包含必要线索，从零开始预热仍可能丢失决定动作的信息。

损失从哪里开始与梯度在哪里截断，是两个独立决定。可以让后缀损失穿过整个预热段求导，也可以将预热末端状态保留为数值、截断其梯度。两者在这次前向计算中输出相同，却可能得到不同的参数更新。[递归回放算例](/zh/continual-rl/foundations/deep/partial-observability/#recurrent-replay-task)逐项比较保存状态、零状态和当前参数重建，并把它们接到实际动作与 TD 标签。

<a id="course-optimizer-memory"></a>

## 3.1 · 学习器也有记忆：参数之外哪些状态会影响下一步

| 状态 | 保存的信息 | 是否属于 agent 的输入记忆 | 改变后必须复查什么 |
| --- | --- | --- | --- |
| recurrent hidden state | 已发生的观察与行动的压缩 | 通常是 | 能否区分影响行动的历史 |
| 价值、策略与 encoder 参数 | 跨经验积累的预测和行为规律 | 是慢速学习状态 | 旧知识、泛化与新任务学习速度 |
| 资格迹 | 过去参数或特征对当前误差的信用 | 不等同于 hidden state | 历史梯度在当前表示下的意义 |
| 优化器一阶、二阶矩 | 历史梯度方向及平方尺度 | 不等同于世界记忆 | 新梯度会被放大、压低还是反向覆盖 |
| 归一化统计与目标网络 | 历史输入尺度与延迟的预测 | 各有独立时间常数 | 新旧输入、标签是否仍在相同单位 |

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\qquad
\Delta\theta_t=-\alpha\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}
$$

这是逐坐标 Adam 的基本形式；帽子表示初始化偏差校正。这里 g 是所声明损失的梯度，不是 TD error 本身。

手算：旧梯度一直为 $+1$，所以 $m_{t-1}\approx1,v_{t-1}\approx1$。目标变化后梯度变为 $-1$。若 $\beta_1=0.9$，第一次新梯度到达后 $m_t=0.8$，更新仍沿负参数方向，和当前梯度下降方向相反。连续 n 次反向梯度后，$m=2(0.9)^n-1$，第七步才改变符号。

这不证明 Adam 一般有害。动量可以抑制噪声、加快固定目标优化；这里显示的是历史信息与适应速度的取舍。二阶矩的长记忆还可能使同一绝对梯度在不同时间得到不同有效步长。因此“学习率没有改变”不等于更新尺度没有改变。

若重新初始化部分神经元，需要一并决定相关优化器矩、资格迹、归一化统计和目标网络参数怎样处理。旧矩若附着到全新的参数功能上，会让重置后的第一步继承旧方向。重置全部统计也会丢掉有用尺度信息。两者都是算法选择，不应藏在实现细节中。

最小诊断是在相同的新数据流上复制学习器，分别只重置优化器、只重置输出头、只更换低效特征，并保留“不重置”对照。若只在闭环任务中看回报，数据覆盖差异可能掩盖真正的优化问题。后续流式与元学习章节分别研究如何计算一次更新，以及如何根据经验调整更新规则。

<a id="lesson-offline"></a>

## 4 · 为什么 off-policy 不自动等于 offline 可用

Off-policy 指学习目标策略与数据行为策略可以不同；offline 还要求数据固定，不再通过新行动纠正估计。Q-learning 或 SAC 在未见动作上高估时，在线环境可能提供反证，离线数据却可能永远没有该动作的结果。因而旧经验复用能力不等于离线控制的可靠性。

$$
\begin{gathered}L_{\rm conservative}(Q)=\log\sum_a e^{Q(s,a)}-\mathbb E_{a\sim\mathcal D(\cdot\mid s)}Q(s,a)\\\frac{\partial L_{\rm conservative}}{\partial Q(s,a)}=\operatorname{softmax}(Q(s,\cdot))_a-p_{\mathcal D}(a\mid s)\end{gathered}
$$

这是离散动作 CQL 型正则的核心形式。完整 CQL 还包含 Bellman loss、权重选择和相应理论条件；连续动作通常需要采样近似。

若数据只含动作零，而 $Q(s,\cdot)=(0,10)$，正则约为 $10.000045$。它对动作一的导数接近一，对动作零的导数接近负一。因此梯度下降压低未支持的高值，并相对提高数据动作。正则本身只约束相对偏好，不能替代奖励监督。

保守性可能减少外推，也可能压制真正有用但数据很少的动作。对任意未覆盖结果都保持正确是不可能由损失形式单独保证的；数据覆盖和任务假设仍然重要。CRL 中的历史 replay 常兼有旧分布、策略覆盖不足和动力学变化，三种误差要分别诊断。

<a id="experiment-deep-cql"></a>

### 实验：固定数据中的保守项：限制外推，不创造缺失证据

训练不再采集新经验时，降低未见动作的估值能否改变策略？

**环境与可用信息。** 位置 0–4 的 DeadlineChain，动作是左／右；观测是位置 one-hot 和剩余时间比例。到位置 4 得 1 并终止，其余每步 −0.02；12 步期限也是问题的真实终止。每回合从 0 开始。训练数据预先用独立 seed 2026 采集 512 条转移，行为以 0.65 概率向右。训练期不与环境交互。

**设置。** 本图实际运行 1200 个预算单位，训练种子为 0–4。这里的预算单位是训练 batch，不是环境步。每批从同一固定数据重采样 32 条；32 隐单元，Adam 0.003，折扣 0.99，Polyak 0.02。每批 CQL 与普通离线 Q-learning 均有一次优化器调用。

**检验的机制。** CQL 在半平方 TD 损失之外，加权重为 1 的 logsumexp 动作值减数据动作值。离散动作可精确求和；没有连续动作采样和自适应 Lagrange 权重。

**测量。** 冻结贪心策略在独立环境中的回报用于评价，不把评价经验加入固定数据。五个 seed 改变网络初始化和数据重采样，而不是产生五套离线数据。

```bash
python3 implementations/deep/cql.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/deep-cql/curves.svg)

横轴：training_batches。纵轴：冻结策略的外部回报。每种方法 1200 training_batches；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 600 个 batch 两者均约 0.94；1200 个 batch 分别为 0.94 与 0.932。当前数据已经足以学到近最优路线，不能用此例证明保守正则对严重覆盖不足普遍有效。

**结论边界。** 它没有扫描数据质量、行为分布和缺失动作，无法证明 D4RL 性能或无覆盖区域的可靠估计。保守偏置还可能压低需要但少见的动作。

**继续实验。** 在训练前固定不同的行为策略与数据量；冻结所有数据再比较普通 Q、CQL 和 IQL。把数据 seed 与训练 seed 分开，不能看过测试回报后挑最好数据集。

[源码](https://yingwen.io/crl-code/implementations/deep/cql.py) · [逐种子记录](https://yingwen.io/crl-code/results/deep-cql/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/deep-cql/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/deep-cql/curves.json)

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

<a id="rlss-plasticity-protocols"></a>

## 可塑性证据怎样读：五类实验、三种秩与必要的对照

“训练很久以后收益下降”是观测，不是机制诊断。新目标可能更难，数据可能变化，旧策略也可能访问不到有用区域。可塑性实验要问一个更窄的问题：在匹配的新目标、数据和学习预算下，已经训练过的学习器是否比合适的新初始化更难学习？训练误差、泛化误差和旧知识保持须分别观察。

| 原研究协议 | 主要隔离的问题 | 不能据此推出什么 |
| --- | --- | --- |
| Continual ImageNet：顺序二分类任务，每任务训练多轮，并重置输出层 | 重复学习新分类问题时的学习能力；隐藏表示跨任务保留 | 已知任务边界与头重置不是无边界严格流式协议 |
| 类增量 CIFAR-100：类别逐步增加，旧数据仍可训练 | 在数据仍可获得时，旧网络与从头训练网络的差别 | 类别变多本身会变难；准确率下降不单独证明遗忘 |
| Online Permuted MNIST：连续输入置换、逐样本更新 | 长序列新映射上的适应，较少跨任务语义共享 | 独立置换上的可塑性不等于有结构任务的快速迁移 |
| Slowly-Changing Regression：固定目标网络，输入的一部分缓慢翻转 | 没有 TD 或控制闭环时，表示与优化怎样适应分布变化 | 小网络监督结果不是深度 CRL 控制结果 |
| 长时 PPO：固定动力学或改变 Ant 摩擦系数 | 外部变化和策略自身诱发的数据变化下的学习 | PPO 有 rollout buffer 与多轮更新，不是单样本即弃 |

Nature 原研究的 PPO 配置用 2048 步 buffer、10 个 epoch、128 大小 minibatch；策略与价值网络各有两层 256 单元。长期运行中也比较了标准 Adam 与调节后的两个动量系数。因此复现“PPO 加 CBP”时，不能遗漏基础优化器、L2、数据复用和总训练时长，随后把所有差异都归因于神经元替换。

$$
r_{99}(H)=\min\left\{k:\frac{\sum_{i=1}^k\sigma_i}{\sum_i\sigma_i}>0.99\right\},\quad
r_{\rm ent}(H)=\exp\!\left(-\sum_i p_i\log p_i\right),\quad p_i=\frac{\sigma_i}{\sum_j\sigma_j}.
$$

H 是同一探测输入上得到的表示矩阵，奇异值按降序排列。r99 是这项工作使用的阈值秩定义之一，r_ent 是奇异值分布的熵有效秩。零矩阵需另外定义约定；不能直接除以零。

$$
r_{\rm stable}(H)=\frac{\|H\|_F^2}{\|H\|_2^2}=\frac{\sum_i\sigma_i^2}{\sigma_1^2}.
$$

常见的稳定秩是另一个量。若奇异值为 3 和 1，则 r99=2，熵有效秩约 1.755，稳定秩为 10/9。三个数字不能都简称“rank”并横向比较。

原论文的部分图使用 stable rank 这个名称，但相应 Methods 给出的是上面的 99% 奇异值质量阈值。读取结果时应按计算定义对应，而不是按标签猜测它一定是 Frobenius 范数与谱范数的比值。

高秩只说明在这些输入上存在多个方向，不说明这些方向对应未来任务。非零梯度、较少休眠单元和较小权重，也都不是充分的可塑性证书。诊断曲线与性能共同变化提供线索；要归因，还需分别干预尺度、激活、表示方向或优化器，并保持其他因素和预算一致。

一个清楚的试验可以先保存同一时刻的已有网络，复制到多组。让已有网络、同容量新网络、仅重置优化器的网络在完全相同的新数据流上学习。记录即时误差、达到指定误差所需样本、旧目标误差及计算成本。新网络是诊断基线，不是必须部署的方案；不可重置世界也未必允许重新获得同一批经验。

实验长度也是问题设定的一部分。一个方法可能在前几次任务变化时受益于迁移，经过更长时间才显现可塑性下降。短 SCR 实验适合检查更新和统计量，不能代替数百万步的长期检验。改变网络、输入变化速度或初始化后，没有出现退化也是有价值的结果：它帮助确定现象的发生条件，而不是应当删除的异常曲线。

一个更严格的 CRL 问题是同时维持当前表现、旧知识和未来学习能力。CBP 的当前贡献效用主要针对前两者中的当前贡献与可塑性，不会自动保存很久未用的技能。多 GVF、多 option 共享表示时，谁的损失决定一个单元可以被删除，仍须明确。

<a id="course-plasticity-diagnosis"></a>

## 8.1 · 保留旧知识与学会新知识是两个检验

遗忘检验问：训练新问题以后，旧问题的表现是否变差？可塑性检验问：给定相同的新问题与学习预算，当前学习器是否比适当对照更难学会？一个网络可以保留旧预测，却因步长极小而几乎学不会新目标；也可以快速学会新目标，却覆盖旧预测。

| 现象 | 必要的区分 | 可执行的诊断 |
| --- | --- | --- |
| 新任务学习变慢 | 任务更难，还是学习器变差 | 相同新目标、输入流、更新预算下比较已训练与新初始化网络 |
| 旧任务回报下降 | 遗忘，还是测试分布改变 | 固定旧任务测试集或固定策略评估；单独记录新分布 |
| 在线回报下降 | 数据缺失，还是有数据也学不会 | 固定行为/数据的预测探针，再与闭环控制对照 |
| 表示秩或活跃比例下降 | 诊断相关量，还是因果机制 | 单独干预对应机制，再测学习曲线而非只测秩 |

$$
h(x)=\max(0,w^\top x+b),\qquad
\frac{\partial h}{\partial w}=\mathbf1[w^\top x+b>0]x
$$

一个具体困难：若所有随后输入的预激活都为负，则这些样本上的局部梯度为零。仅把误差放大，不会让这个 ReLU 单元通过自身梯度重新活跃。输入分布或上游表示变化仍可能使它重新活跃。

Continual Backprop 在梯度学习之外保留少量特征生成与检验。其贡献型效用可用隐藏单元的活跃量与出连接幅度的乘积估计，再作指数平均。成熟期保护刚生成的单元，防止它在出连接尚未学到时立刻被淘汰。效用是当前数据下的代理量，不是该单元未来所有用途的真价值。

$$
u_i\leftarrow\eta u_i+(1-\eta)|h_i|\sum_j|w_{ji}^{\mathrm{out}}|
$$

这是原论文贡献型效用的基本形式。特征生成规则、每层替换率、成熟期和效用归一化应按具体实现声明。

若当前输出含 $w_i^{\rm out}h_i$，把这条出连接置零会先移除旧贡献。新单元的出连接为零，只保证“新生成的信号”暂时不额外改变输出，并不保证删除旧单元毫无影响。例如旧贡献为 0.02，输出就会先少 0.02；当前贡献很小也不代表罕见状态中的贡献很小。

原论文在持续监督学习与长时 PPO 任务中报告了可塑性退化及缓解结果。它不是所有网络必然退化、或某一种重置在所有 CRL 任务上收敛的定理。把神经元进一步看作具有局部目标的组件，是值得研究的架构设想；从局部效用到全局长期回报的信用分配仍需论证。

这一节给出的桥梁是：深度 RL 不只要在一个固定 benchmark 上找到好策略，还要让同一有限学习器持续产生可用的新表示。后续可塑性章节比较特征替换、正则化与优化尺度；知识构建章节再讨论这些表示被哪些预测、目标和技能使用。

<a id="lesson-check"></a>

## 9 · 练习与答案

- 问：重建误差下降是否保证模型控制改善？答：不保证。Bellman 相关误差、规划覆盖和奖励敏感维度可能仍有偏差。
- 问：detach hidden state 是否清除 agent 的记忆？答：不清除数值记忆，但截断其对早先参数计算的梯度依赖。
- 问：CQL 正则能从零数据推断未见动作奖励吗？答：不能。它限制乐观外推，不产生缺失的因果信息。
- 实验：把 DeadlineChain 的剩余时间从观察删除，在不同剩余步数的同一位置比较价值目标；答案是出现状态混叠，不能简单归因于 DQN 优化失败。

## 从本章进入实践

[策略梯度与控制](https://yingwen.io/zh/continual-rl/code/#practice-policy-control)：优化器确实降低了损失，为什么行动仍可能变差？



<a id="chapter-code"></a>

## 下载与运行

本文件用标准库核验数值；deep_textbook_train.py 提供 DQN/PPO 小任务训练与连续控制更新核，连续控制完整教学训练见 implementations/deep/ 的独立实现。

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

- [Sutton & Barto · Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)：§9–11：函数逼近与离策略；§12：资格迹；§13.1 的短走廊与 §13.2–13.5 的策略梯度。对照各结论采用的策略类、采样分布与函数表示。

- [Dohare et al. · Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7)：区分旧知识遗忘与新知识学习能力；Methods 给出 contribution utility、成熟期及单元重置。文中的长时实验不是任意任务上的稳定性证明。

- [Dohare et al. · Loss of plasticity 作者代码](https://github.com/shibhansh/loss-of-plasticity)：对照原论文的持续监督学习、PPO 与 generate-and-test 实现；复制实验前核对网络、优化器和重置配置。

- [Elsayed et al. · Streaming Deep Reinforcement Learning Finally Works](https://arxiv.org/abs/2410.14606)：Stream-X 将神经预测与控制放回逐样本更新；应逐项检查归一化、初始化、资格迹和步长控制，不把移除 replay 当作完整算法。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-practice#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-practice#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-deep-practice)
<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 估计哪种策略的价值，又按什么状态分布加权？

行为策略决定怎样获得经验；目标策略决定要预测哪种行为。重要性比可以校正给定状态下的动作分布，但不会自动把状态出现的频率改成目标策略的频率。

函数逼近与深度方法：Replay 还引入缓冲区的时间组成与抽样规则。神经更新受到数据分布、共享梯度和移动目标共同影响。重复旧数据与逐条使用新数据有不同的资源和适应代价。

持续学习中的研究问题：单一行为流怎样支持许多预测和技能？在固定内存下，怎样权衡覆盖、样本年龄、更新方差与适应速度，而不把离策略修正当作完整稳定性保证？

[离策略稳定性](https://yingwen.io/zh/continual-rl/foundations/approximation/off-policy/) → [数据与训练接口](https://yingwen.io/zh/continual-rl/foundations/deep/practice/) → [大规模系统与策略滞后](https://yingwen.io/zh/continual-rl/foundations/deep/systems/) → [离线数据的覆盖](https://yingwen.io/zh/continual-rl/foundations/deep/offline/) → [流式更新](https://yingwen.io/zh/continual-rl/algorithms/streaming/)


### 可进一步检验的问题

- [02 · 有限的内部状态应当保留哪些历史信息？](https://yingwen.io/zh/continual-rl/research/#research-agent-state)：将 recurrent 活动状态与跨时间参数梯度分开，才能区分线索没有被保存和保存机制没有学会。
- [13 · 为什么训练越久，学习新东西反而越慢？](https://yingwen.io/zh/continual-rl/research/#research-plasticity)：训练与评价协议中的参数、优化器和数据权限必须明确，才能比较老网络、新初始化及局部重置后是否还能学会新目标。
- [16 · 什么实验能区分“仍在更新”与“仍在有效学习”？](https://yingwen.io/zh/continual-rl/research/#research-measurement)：分开训练过程、冻结评价、随机重复和计算预算，才能判断一个机制改善了哪一段学习过程。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)
- [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)
- [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)
- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)
- [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)
- [可塑性与特征更新](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

对应原始材料：Sutton & Barto §8、§11、§17；CQL §3–4；Deep RL at the Edge of the Statistical Precipice。本文为原创讲解，原书、论文与上游代码保留各自许可。
