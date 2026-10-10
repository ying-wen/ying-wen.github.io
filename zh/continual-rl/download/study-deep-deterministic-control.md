# 连续动作的价值优化：DDPG 与 TD3

现代深度强化学习 · 第 4 章

不能枚举连续动作时，如何用 critic 的梯度改进 actor？

## 本章内容

- 区分行为噪声、目标动作噪声与目标网络。
- 推导确定性 actor 的链式梯度。
- 写出 TD3 双 critic、延迟更新与平滑目标的完整次序。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [深度价值学习：DQN、Double DQN 与目标的时间顺序](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/)：掌握回放、critic标签和目标网络的更新时间。
- [策略梯度、基线与 Actor–Critic](https://yingwen.io/zh/continual-rl/foundations/approximation/policy-gradient/)：理解策略参数怎样通过动作影响目标。


### Bellman 递推

当前价值等于即时奖励加折扣后的后继价值；预测目标与最优控制目标不同。

### 函数与梯度

神经网络把参数和输入映射为输出。链式法则必须指明哪些量参与求导，哪些量作为固定目标。

### 经验分布

on-policy 数据由当前策略产生；off-policy 数据可来自旧策略，但能否正确复用取决于算法目标和数据覆盖。

<a id="problem-definition"></a>

## 本章的问题定义

动作是连续向量，无法像离散 Q-learning 那样枚举所有动作来最大化价值。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 确定性 actor $a=\mu_\theta(s)$，critic $Q_\phi(s,a)$，replay 状态分布 $d_{\mathcal D}$。

### 需要求解的对象

同时学习动作价值和近似最大化它的连续动作函数。

### 信息与数据权限

训练数据由带探索噪声的行为产生；actor 的导数使用 critic，不使用真实环境导数。

$$
L_{\rm actor}(\theta)=-\mathbb E_{s\sim d_{\mathcal D}}[Q_\phi(s,\mu_\theta(s))]
$$

这是固定 critic 与 replay 分布下的实际 actor surrogate。它与完整 on-policy 收益梯度的关系需要分布及 critic 正确性条件，不能省略这一步。

### 成立条件与解的含义

- 动作和 critic 对动作可微，动作边界处理明确。
- 行为需提供足够局部动作覆盖；函数近似与 bootstrap 可能放大误差。

判断准则：验证动作梯度、target noise、更新频率，再检查真实收益与过估计，而不是仅看 actor loss。

### 适用边界

- 保证 actor 找到 critic 的全局动作最大值，或 critic 准确描述未覆盖动作。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 限制表示或采用近似 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：本章限制为确定性 actor，并用可微近似 critic 搜索连续动作；这是控制的表示与求解近似，不是离散 DQN 的特例。

- 改变评价目标 · [最大熵控制](https://yingwen.io/zh/continual-rl/algorithms/soft-control/)：本章优化外部奖励而不含策略熵；相对最大熵控制，去掉熵项改变目标，不只是改变行为噪声。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

actor 会主动寻找 critic 估计高的位置，因此可能利用 critic 的误差。

### 本章的核心思路

通过可微 critic 给 actor 方向，再限制目标过估计和过快策略反馈。

1. [明确链式梯度的对象](#lesson-derive)：梯度经动作进入 critic，不穿过未知环境。

2. [稳定 critic 的回归参照](#lesson-target)：DDPG 使用缓慢目标网络，区分当前优化与目标构造。

3. [分别限制三种反馈问题](#lesson-td3)：TD3 的双 critic、平滑目标和延迟 actor 更新有不同作用。

结论与条件：确定性策略梯度定理不等于任意 replay 和近似 critic 下的无偏实现。

### 相关方法改变了什么

- DDPG / TD3：TD3 改变目标估计、目标动作扰动和 actor 更新频率。

- 行为探索噪声 / target smoothing：前者改变真实采样，后者改变 critic 的训练目标。


<a id="lesson-setting"></a>

## 1 · 用 actor 近似连续动作的最大化

一个定位器要在两步内靠近位置一。状态是 $s=(x,h)$：$x$ 为位置，$h\in\{2,1,0\}$ 为剩余动作次数。动作是位移 $a\in[-2,2]$，环境执行 $x'=x+a$，给奖励 $r=-(x'-1)^2-0.1a^2$，并令 $h'=h-1$；$h'=0$ 时真正终止，折扣率为 $0.9$。误差平方鼓励靠近目标，动作平方惩罚大幅移动。剩余次数属于状态，因此相同位置在第一步与末步是不同决策问题。

固定一条经验：$(0,2)$ 执行动作 $0.25$，到达 $(0.25,1)$，奖励为 $-0.75^2-0.1(0.25)^2=-0.56875$。还有一步时，动作之后没有未来奖励，所以真实价值可直接算成 $q(a)=Q((0.25,1),a)=-(a-0.75)^2-0.1a^2$。本章先在这一状态检查动作梯度和 critic 误差，再把同一条经验送进 TD3 标签；下一章沿用这个任务计算 SAC。

离散 DQN 可以对所有动作求最大值。连续动作空间中，$\max_a Q(s,a)$ 通常需要一次数值优化。DDPG 引入确定性 actor $a=\mu_\theta(s)$，使一次网络前向直接给出动作。critic 学习当前策略的价值，actor 沿 critic 对动作的梯度移动。两者不是独立的监督任务：critic 的误差会影响 actor，actor 改变后又会影响后续目标。

本章采用有界连续动作、折扣奖励与经验 replay。行为动作为 $a=\operatorname{clip}(\mu_\theta(s)+\epsilon,a_{\min},a_{\max})$，以增加探索。评估可使用无噪声 actor。行为噪声改变收集到的数据，不等于 TD3 在训练 target 中加入的平滑噪声。

<a id="lesson-derive"></a>

## 2 · 确定性策略梯度与实际 replay 目标

$$
\begin{gathered}\nabla_\theta J=\frac1{1-\gamma}\mathbb E_{s\sim d_\mu}\left[D_\theta\mu_\theta(s)^\top\nabla_aQ^\mu(s,a)\big|_{a=\mu_\theta(s)}\right]\\d_\mu(B)=(1-\gamma)\sum_{t\ge0}\gamma^t\Pr_\mu(S_t\in B)\end{gathered}
$$

这里 J 是固定初始分布的折扣回报；用状态集合 B 定义占据分布，可同时处理连续状态和概率原子。归一化后的常数为 1/(1−γ)。动作 Jacobian 的形状为动作维度×参数维度，因此转置后得到参数梯度。精确定理仍要求可微性、可积性等条件。

为什么不需要显式对状态分布求导？对策略评价的 Bellman 方程求导，得到 $\nabla_\theta V^\mu=g+\gamma P_\mu\nabla_\theta V^\mu$，其中 $g(s)=D_\theta\mu(s)^\top\nabla_aQ^\mu(s,\mu(s))$。反复代入得到 $\sum_{t\ge0}\gamma^tP_\mu^t g$；对初始状态取期望就是上面的占据分布。后续状态的影响没有被忽略，而是由这个递推收集起来。

实践中既不知道真实 $Q^\mu$，也不直接采样精确的 $d_\mu$。DDPG 用 replay 状态和近似 critic 构造 $L_\mu=-\mathbb E_{s\sim\mathcal D}Q_\phi(s,\mu_\theta(s))$。这是实际优化的 surrogate，不应不加条件地称为原始起点回报的无偏梯度。

这里有两处不同的近似：把真实价值换成 $Q_\phi$，把当前策略的占据权重换成 replay 的状态权重。即使 critic 恰好准确，第二处也不会自行消失。DPG 原论文 §4.2 的离策略式含近似号；停止 critic 参数梯度只规定一次更新的计算路径，不把 replay 变成当前策略分布。本页末步的单状态算例能隔离第一处误差；它没有测量完整两步起点目标的策略梯度。

$$
\nabla_\theta L_\mu=-\mathbb E_{\mathcal D}\left[D_\theta\mu_\theta(s)^\top\nabla_a Q_\phi(s,a)\big|_{a=\mu_\theta(s)}\right]
$$

沿用上面的列向量梯度与动作 Jacobian 约定。actor 更新时固定 critic 参数，但保留 Q 对动作的导数。对整个 Q 前向使用 no_grad 会把所需的梯度一起删除。

例如 $Q(a)=-(a-0.8)^2$，$\mu_\theta=\theta$。在 $\theta=0$ 处，价值的上升方向为 $-2(0-0.8)=1.6$。这个方向来自 critic 的动作斜率。若 critic 在未见动作上虚构一座高峰，actor 也会向那座高峰移动。

![连续动作价值曲线的actor梯度，以及真实探索噪声和TD3目标平滑噪声的两条路径](https://yingwen.io/crl-figures/concept-deep3-actor-gradient-noise.svg)

上图固定状态和 critic，只计算一次 actor 步骤：$\theta=0$，步长 $0.1$，新动作为 $0.16$。橙色切线给出动作梯度；曲线上移表示这个给定 critic 的预测增加，不是已经测量到环境收益增加。下两行区分真实行为噪声和 replay 标签中的目标噪声，使用本章 §6 的 TD3 算例。理论来源为 DPG §3.1–3.2；目标平滑见 TD3 §5.3、Algorithm 1。

图中有三种不同的改变。actor 梯度改变以后会选择的动作。行为噪声改变这一次实际执行的动作。目标平滑噪声只改变 critic 回归时查询的后继动作。后两者即使都用 Gaussian，也不属于同一条交互路径；下一节先构造 DDPG 标签，再说明 TD3 怎样修改它。

<a id="course-critic-slope"></a>

## 2.1 · 价值拟合得准，为什么动作梯度仍然可能错误

确定性 actor 使用的是 $\nabla_aQ_\phi(s,a)$，而 critic 的回归损失主要约束采样动作上的数值。函数值接近，不自动意味着导数接近。要看清这个区别，暂时去掉采样与自举，只研究一个已知的一维价值函数。

$$
Q(a)=-a^2,\qquad \widehat Q(a)=-a^2+\varepsilon\sin(\omega a),
\qquad |\widehat Q(a)-Q(a)|\le\varepsilon
$$

近似价值在所有动作上与真值相差至多 ε。这个误差界本身不限制振荡频率。

$$
Q'(0)=0,\qquad \widehat Q'(0)=\varepsilon\omega
$$

取 ε=0.01、ω=100，最大价值误差只有 0.01，动作零处的梯度误差却为一。真实动作零已经最优，critic 仍可能把 actor 推离它。

TD3 的目标动作平滑可以降低对狭窄价值峰的敏感性，但它也改变了所用目标，效果取决于平滑尺度与真实价值几何。双 critic 的最小值约束目标数值，不是一个动作导数正确性的证书。两网络若共享相似数据与误差，错误方向仍可一致。

实际诊断应在 actor 常访问与准备访问的动作邻域检查局部价值排序。可在可重置的小任务中，用额外受控试验估计邻近动作的实际收益，并单独计入诊断预算；不可重置的 CRL 世界不一定允许这种反事实检查。动作平滑、保守更新与主动探索因此都有信息与资源代价。

持续任务还会同时改变 critic 的表示、访问分布和目标策略。仅让 critic 比 actor 多更新几次，没有保证它已经跟上变化。可固定 replay 测估计器，再固定估计器测 actor 更新，最后测闭环耦合；三步可以区分“没有学准”与“学准了当前 surrogate 但它不是当前任务”。

<a id="lesson-target"></a>

## 3 · DDPG 的 critic 与缓慢目标

$$
y=r+\gamma(1-d)Q_{\bar\phi}(s',\mu_{\bar\theta}(s')),\qquad L_Q=\mathbb E_{\mathcal D}(Q_\phi(s,a)-\operatorname{sg}(y))^2
$$

d 只标识真正终止。target actor 与 target critic 都用于生成停止梯度的标签。

$$
\bar\phi\leftarrow(1-\tau)\bar\phi+\tau\phi,\qquad\bar\theta\leftarrow(1-\tau)\bar\theta+\tau\theta
$$

本章 $\tau$ 是新参数的比例。某些实现用 $\rho$ 表示旧参数比例，因而 $\rho=1-\tau$。把两个约定混用会把慢更新变成近乎直接复制。

target network 减缓自举标签随在线网络移动，并不将错误标签变成正确标签。Replay 提供状态动作覆盖和数据复用，但 actor 仍可能选到数据稀少的动作。网络容量、更新次数与数据覆盖共同决定误差。

<a id="lesson-critic-error"></a>

## 3.1 · 预测上升，为什么真实回报反而下降？

回到定位器的末步。真实斜率为 $q'(a)=1.5-2.2a$；在 $a=1.2$ 处是 $-1.14$，应该减小动作。现在人为给 critic 一项局部误差：$Q_1(a)=q(a)+0.8\exp[-(a-1.4)^2/(2\times0.12^2)]$。这是给定的函数快照，方便逐项检查误差怎样进入控制。该 critic 在 $1.2$ 处的斜率约为 $1.63058$，方向已反转。

$$
\begin{gathered}\mu_\theta=2\tanh\theta,\quad\theta=\operatorname{atanh}(0.6)\approx0.693147\\\frac{\partial\mu_\theta}{\partial\theta}=2(1-0.6^2)=1.28\\\frac{\partial Q_1(\mu_\theta)}{\partial\theta}=1.63058\times1.28\approx2.08714\end{gathered}
$$

critic 参数保持固定，动作对 actor 参数的路径保留。若把整个 Q 前向停止梯度，就连这个动作导数也无法得到。

只计算一次步长为 $0.05$ 的上升更新，得到 $\theta^+\approx0.797504$，动作从 $1.2$ 增为 $1.325278$。critic 预测从 $-0.147018$ 升为 $0.152434$；真实末步回报却从 $-0.3465$ 降为 $-0.506582$。由于这已经是最后一步，真实评价直接由环境奖励给出，无须再相信另一个 learned critic。这个反例让“critic 越乐观，actor 越偏向错误动作”的反馈具体可见。

![两步定位器的末步真实价值与带窄峰误差的 critic；一次解析 actor 更新使预测上升、真实奖励下降。](https://yingwen.io/crl-figures/continuous-control-critic-error.svg)

先看位置尺上的真实经验，再在同一动作轴上比较两个价值函数。圆点是更新前，方点是沿给定 critic 做一次解析梯度步后；曲线放大显示 $a\in[0,1.8]$，完整动作范围仍为 $[-2,2]$。原创构造计算，无随机 rollout 或网络训练；[数值](/crl-figures/continuous-control-data.json)和[独立标准库脚本](/crl-code/tutorials/continuous_control_walkthrough.py)包含有限差分与真实奖励复核。

<a id="lesson-td3"></a>

## 4 · TD3 的三个改动分别限制什么

$$
\begin{gathered}\tilde a'=\operatorname{clip}\left(\mu_{\bar\theta}(s')+\operatorname{clip}(\epsilon,-c,c),a_{\min},a_{\max}\right)\\\epsilon\sim\mathcal N(0,\sigma^2 I)\end{gathered}
$$

target policy smoothing 先裁剪噪声，再裁剪动作；行为探索噪声有另一套参数和用途。

$$
\begin{gathered}y=r+\gamma(1-d)\min_{i=1,2}Q_{\bar\phi_i}(s',\tilde a')\\L_{Q_i}=\mathbb E(Q_{\phi_i}(s,a)-\operatorname{sg}(y))^2\\L_\mu=-\mathbb E Q_{\phi_1}(s,\mu_\theta(s))\end{gathered}
$$

两个 critic 都拟合同一个较小目标；标准 TD3 actor 使用第一个 critic，不在 actor loss 中再取两个值的最小。

第一，双 critic 取最小值抑制部分过估计，也可能引入低估；两个网络相关时，它不是统计置信下界。第二，actor 延迟更新，让 critic 在两次策略变化之间多做学习。第三，目标动作附近的平滑减少对狭窄价值尖峰的依赖。这三处分别作用于目标值、更新时间和目标动作，不是一个统一的学习率技巧。

仍用奖励 $-0.56875$ 的那条经验。给 target actor 动作 $1.4$，冻结 $\bar Q_1=q+0.8b$ 与 $\bar Q_2=q+0.3b$，其中 $b(a)=\exp[-(a-1.4)^2/(2\times0.12^2)]$。在中心点，两值为 $0.1815,-0.3185$；单 critic 标签为 $-0.4054$，只改为双 critic 最小值后是 $-0.8554$。两个 critic 仍共享同一位置的正误差，取最小值并未恢复真实值。

再令 $\epsilon\sim\mathcal N(0,0.15^2)$，先把噪声裁至 $[-0.2,0.2]$，再把动作裁至 $[-2,2]$。这里查询动作落在 $[1.2,1.6]$，噪声平均后的较小价值约为 $-0.441031$，对应平均标签约 $-0.965678$。裁剪 Gaussian 会把两端尾概率各约 $0.091211$ 堆在边界，不能按“截断后重新归一化的 Gaussian”积分。配套脚本用确定性积分计算这个平均；标准 TD3 通常对每个 batch 元素抽一个噪声来估计它。

![同一后继状态的两个固定 target critic 及裁剪 Gaussian 查询范围；比较单 critic、双 critic 和目标平滑后的回归标签。](https://yingwen.io/crl-figures/continuous-control-target-smoothing.svg)

阴影表示加噪后的动作范围，蓝横线是双 critic 最小值对裁剪噪声的期望。底部圆点共享标签数值轴，不表示算法得分。沿用同一定位器，2000 段 Simpson 积分加两端概率原子；1000/2000 段结果独立检查一致。[计算代码](/crl-code/figures/continuous-control-walkthrough.mjs)。目标平滑与延迟次序依据 [TD3 §5 / Algorithm 1](https://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf#page=5)。

平滑改变的是后继目标的价值查询；它没有直接改写在线 actor loss。标准 TD3 在 actor 更新日仍沿第一个在线 critic 的动作斜率更新。要让未来 actor 少受误差峰影响，还要通过 critic 拟合、数据覆盖和后续更新来传递这一作用。

<a id="experiment-deep-td3"></a>

### 实验：双 critic 与延迟更新，在小型控制中也可能更差

TD3 改变了目标噪声、critic 数量和 actor 时钟。这三个修改是否必然改善一次短训练？

**环境与可用信息。** 有界一维 LQ：位置从 [−1,1] 均匀初始化；动作截断到 [−1,1]，下一位置为 0.92 倍当前位置加 0.3 倍动作，再截断到 [−3,3]。奖励为负的位置平方减 0.05 倍动作平方；40 步真实终止。位置与剩余时间均可观测。

**设置。** 本图实际运行 1200 个预算单位，训练种子为 0–4。横轴只计训练转移，训练折扣为 0.99。actor 和 critic 使用 32 个 tanh 隐单元。actor Adam 步长 0.001，critic 为 0.002；前 64 步均匀动作，此后探索噪声标准差 0.15。回放容量 4000，batch 32；第 32 步起每步更新 critic，均匀探索期也学习。TD3 每两次 critic 批更新才更新 actor 和目标副本，Polyak 新参数占比 0.02。

**检验的机制。** 目标动作噪声标准差 0.2，截断到 ±0.5，再把动作截断到合法范围；以两目标 critic 的较小值构造目标。actor 通过 Q1 对动作的导数更新，而不同时改写 Q1 参数。

**测量。** 每个记录点冻结确定性 actor，在独立环境 seed 991 产生的同一组 12 个初态上各跑 40 步；纵轴是未折扣外部回报。包含初始化在内共 21 次评估，每个训练 seed 另用 10080 个评估环境步。这些经验不进入 replay，横轴也不计它们。五个训练 seed 才是独立重复，12 个固定初态不是另外 12 个训练种子。

```bash
python3 implementations/deep/td3.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-figures/result-deep-td3.svg)

横轴：训练环境步（独立评估交互另计）。纵轴：冻结策略的外部回报。每种方法 1200 训练环境步；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 冻结当前 actor，在种子991生成的同一组12个初态上各跑40步，平均未折扣外部回报。SAC 此处执行 tanh(mean)，没有抽样动作，也不把熵奖励加入评估。

**step：怎样计时。** step 只数训练环境转移；评估交互另计。critic、actor 和 target 的更新数保存在独立字段，相同 step 不保证相同计算量。

**怎样汇总。** 取该记录时刻的值；不先对曲线上的时间点求平均。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** CSV 可重算冻结评估曲线并读取更新计数；没有逐步训练奖励，不能重建行为策略的生命期收益。12个初态的平均先在单次运行内完成。

计算位置：[deep/ddpg.py](https://yingwen.io/crl-code/implementations/deep/ddpg.py) · [deep/td3.py](https://yingwen.io/crl-code/implementations/deep/td3.py) · [deep/sac.py](https://yingwen.io/crl-code/implementations/deep/sac.py) · [deep/_common.py](https://yingwen.io/crl-code/implementations/deep/_common.py)

</details>

**结果分析。** 600 步时 TD3 均值约 −30.95，DDPG 约 −4.47；1200 步时分别约 −4.27 与 −2.35。TD3 的末点样本标准差约 6.81，当前配置没有显示优势，早期还出现明显退化。

**结论边界。** 1200 步内两者各有 1169 次 critic 批更新；DDPG 每批更新一个 Q，TD3 更新两个 Q。actor/target 的更新次数分别为 1169 与 584，因而不属于等计算比较。结果限于这组网络、参数和短预算任务；训练期间的行为奖励没有记录在这条冻结评估曲线中。

**继续实验。** 固定同一骨干，依次关闭目标噪声、双 critic、延迟 actor；分别报告真实步与优化器调用。先预测哪个诊断量会变化，再运行独立 seed。

[源码](https://yingwen.io/crl-code/implementations/deep/td3.py) · [逐种子记录](https://yingwen.io/crl-code/results/deep-td3/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/deep-td3/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/deep-td3/curves.json)

<a id="lesson-algorithm"></a>

## 5 · 先更新 critic，再按计数器更新 actor

**算法：配套代码以 delay=2 为默认值。计数器统计 critic 更新次数，不应在未知 update-to-data ratio 下直接当作环境步数。**

1. 采样 replay 的 $(s,a,r,s',d)$ minibatch。
1. 无梯度计算平滑目标动作和双 target critic 的最小值。
1. 计算一个固定 $y$，分别反传并更新两个在线 critic。
1. 若 critic 更新计数是 delay 的整数倍：
  1. 暂时冻结 critic 参数，但保留动作导数。
  1. 对 $-Q_{\phi_1}(s,\mu_\theta(s))$ 更新 actor。
  1. 恢复 critic 参数的可训练标志。
  1. 对 actor 和两个 critic 的 target 做 Polyak 更新。
1. 继续收集真实数据，并单独处理终止与 reset。

若一条环境交互触发多个梯度步骤，目标更新频率与 actor 更新频率会随之改变。复现实验必须同时记录环境步数、critic 步数、actor 步数与 replay 大小，不能只给一个总 steps。

<a id="lesson-example"></a>

## 6 · 目标动作、目标值与梯度手算

以下是独立归一化输入的算子检查，用来单独触发裁剪边界。目标 actor 输出 $0.9$，抽样噪声为 $3$，噪声上限 $c=0.2$，动作范围为 $[-1,1]$。先得 $0.9+0.2=1.1$，再裁为 $1$。若两 target critic 在动作一上的值为 $3,2$，奖励为一、$\gamma=0.9$，则 target 为 $1+0.9\times2=2.8$。真终止时 target 只有一，噪声与 critic 均不影响它。

旧 target 参数为十，在线参数为二，$\tau=0.005$。新 target 为 $0.995\times10+0.005\times2=9.96$。若误把 0.995 当成新参数比例，会得到 2.04，更新时间尺度完全不同。

连续控制数值核包含 TD3 target、Polyak、SAC target 与 squash 密度；本节使用其中前两项。

```python
def td3_target(reward, gamma, terminated, action, noise, noise_clip,
               low, high, q1, q2):
    perturbation = max(-noise_clip, min(noise_clip, noise))
    smoothed = max(low, min(high, action + perturbation))
    target = reward + gamma * (not terminated) * min(q1(smoothed), q2(smoothed))
    return target, smoothed


def polyak(old, online, tau):
    if not 0 <= tau <= 1:
        raise ValueError('tau must lie in [0, 1]')
    return [(1-tau)*x + tau*y for x, y in zip(old, online)]


def tanh_log_prob(u, mean, log_std, scale=1.0):
    if scale <= 0:
        raise ValueError('positive affine action scale required')
    log_normal = -0.5*((u-mean)/math.exp(log_std))**2-log_std-0.5*math.log(2*math.pi)
    log_jacobian = 2*(math.log(2)-u-softplus(-2*u))
    return log_normal - log_jacobian - math.log(scale)


def sac_target(reward, gamma, terminated, q1, q2, log_prob, alpha):
    return reward + gamma*(not terminated)*(min(q1, q2)-alpha*log_prob)


def temperature_log_gradient(log_alpha, log_prob, target_entropy):
    # Exact gradient of -exp(log_alpha) * stopgrad(log_prob + target_entropy).
    return -math.exp(log_alpha)*(log_prob+target_entropy)
```

<a id="lesson-code"></a>

## 7 · 实际 autograd 更新核

先运行本页的独立算例：下载 [continuous_control_walkthrough.py](/crl-code/tutorials/continuous_control_walkthrough.py)，执行 python3 continuous_control_walkthrough.py 查看数值，执行 python3 continuous_control_walkthrough.py --test 检查动作链式导数、真实奖励、裁剪噪声积分及下一章 SAC 的梯度。它只使用 Python 标准库，给定数据与函数即可复算。

DDPG 的单 critic 目标、确定性 actor 梯度及每次优化后的 target 平滑更新。

```python
def ddpg_update(actor, critic, target_actor, target_critic,
                actor_opt, critic_opt, batch, gamma=.99, tau=.005):
    x, action, reward, xp, terminal = batch
    with torch.no_grad():
        y = reward+gamma*(1-terminal)*target_critic(xp, target_actor(xp))
    prediction = critic(x, action)
    assert prediction.shape == y.shape == reward.shape
    loss_q = ((prediction-y)**2).mean()
    critic_opt.zero_grad(); loss_q.backward(); critic_opt.step()
    freeze([critic], True)
    loss_actor = -critic(x, actor(x)).mean()
    actor_opt.zero_grad(); loss_actor.backward(); actor_opt.step()
    freeze([critic], False)
    update_targets([actor, critic], [target_actor, target_critic], tau)
    return float(loss_q.detach()), float(loss_actor.detach())
```

两个 critic 的标签停止梯度；延迟步骤才更新 actor 与全部 target。

```python
def td3_update(actor, q1, q2, target_actor, target_q1, target_q2,
               actor_opt, critic_opt, batch, update_index, delay=2, gamma=.99, tau=.005):
    x, action, reward, xp, terminal = batch
    with torch.no_grad():
        next_action = target_actor(xp)
        noise = (.2*torch.randn_like(next_action)).clamp(-.5, .5)
        next_action = (next_action+noise).clamp(-1, 1)
        y = reward+gamma*(1-terminal)*torch.minimum(target_q1(xp, next_action), target_q2(xp, next_action))
    loss_q = ((q1(x, action)-y)**2+(q2(x, action)-y)**2).mean()
    critic_opt.zero_grad(); loss_q.backward(); critic_opt.step()
    policy_loss = None
    if update_index % delay == 0:
        freeze([q1, q2], True)
        # Freeze critic parameters, not the path from action to critic output.
        loss_actor = -q1(x, actor(x)).mean()
        actor_opt.zero_grad(); loss_actor.backward(); actor_opt.step()
        freeze([q1, q2], False)
        update_targets([actor, q1, q2], [target_actor, target_q1, target_q2], tau)
        policy_loss = float(loss_actor.detach())
    return float(loss_q.detach()), policy_loss
```

执行 python3 deep_textbook_train.py test 会检查非 actor 更新步的参数保持不变，并执行真实 TD3 critic/actor 梯度更新。这一文件的 DDPG/TD3 部分是单步更新核。另有可从环境交互一直运行到独立评价的 [DDPG 完整教学实现](/zh/continual-rl/code/deep-ddpg/)与 [TD3 完整教学实现](/zh/continual-rl/code/deep-td3/)，它们使用一维 BoundedLQ，和本页两步定位器的解析算例分别服务于训练过程与机制检查。MuJoCo 复现则可继续读作者 TD3.py 与 main.py。

完整 BoundedLQ 教学实现的目标混合比例为 τ=.02；本节旧更新核默认 τ=.005。完整实现中，前 64 步采用均匀动作探索，replay 满 32 条即开始更新。探索时段、学习起点和 target 更新时间是三个不同设置。

本核默认动作已经归一化到 $[-1,1]$。环境的物理扭矩或速度范围需要外部可逆缩放。若直接在物理单位加入同一数值的噪声，探索强度和 target smoothing 强度会随单位改变。

<a id="lesson-branches"></a>

## 8 · 失败条件与 CRL 中的变化

- 梯度断开：冻结 critic 参数可以，detach actor 动作或对整个 actor loss 使用 no_grad 不可以。
- 旧数据失配：动力学改变后，同一个 (s,a) 可能对应新的奖励和转移；混合 replay 不再来自一个固定 Bellman 算子。
- 外推峰值：actor 可以优化到 critic 数据不足的动作区域；双网络并不消除这个问题。
- 动作饱和：tanh 接近边界时导数小，actor 可能很难离开边界；动作尺度和初始化都需要记录。

持续控制还需区分控制器记忆与学习器记忆。确定性 actor 可以是 recurrent 的，replay 也可以保存序列；但这会新增 hidden-state 重建与跨时间梯度问题。它不是在当前 feed-forward batch 上简单多加一列特征。

将定位器扩展成持续学习实验时，可以让目标位置或执行器增益在学习器生命期中变化，并继续累计外部奖励、适应代价与计算预算。一个生命期可以包含多个 episode；是否允许环境重置、是否保留 replay 和优化器状态，需要另外规定。先固定这些条件，才能比较失效来自 critic 旧标签、动作覆盖还是记忆不足。

<a id="lesson-check"></a>

## 9 · 练习与答案

- 问：DDPG critic loss 中也更新 actor 会怎样？答：actor 会沿着降低拟合误差而不是提高 Q 的方向改变 target，失去指定更新的含义。
- 问：TD3 的两个 critic 一致就表示可信么？答：不表示。相同数据和相似函数类可以产生相关错误。
- 问：探索噪声与 target smoothing 可以共享同一随机样本吗？答：标准机制不要求共享；前者发生在真实行为，后者发生在 replay 标签构造，必须分清数据时间。
- 实验：对标量 $Q(a)=-(a-0.8)^2$ 做中心差分，比较 actor 的解析梯度 1.6；再 detach 动作，检查自动微分为何无法更新 actor。

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

- [Spinning Up · DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)：用作算法与实现接口的对照；本章解释和配套小实验独立编写。

- [Spinning Up · TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html)：用作算法与实现接口的对照；本章解释和配套小实验独立编写。

- [Silver et al. · Deterministic Policy Gradient Algorithms](https://proceedings.mlr.press/v32/silver14.html)：确定性策略梯度定理及其占据分布条件。

- [Lillicrap et al. · Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)：DDPG：确定性 actor、replay 与 target network。

- [Fujimoto et al. · Addressing Function Approximation Error in Actor-Critic Methods](https://proceedings.mlr.press/v80/fujimoto18a.html)：TD3 原论文，ICML 2018。

- [Fujimoto · TD3.py](https://github.com/sfujim/TD3/blob/master/TD3.py)：作者实现；检查 policy_noise、noise_clip、policy_freq 与 target 更新时机。

- [Spinning Up · ddpg.py](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py)：官方教学实现，包含完整采样与 replay 循环。

- [Sutton & Barto · Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)：§9–11：函数逼近与离策略；§12：资格迹；§13.1 的短走廊与 §13.2–13.5 的策略梯度。对照各结论采用的策略类、采样分布与函数表示。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-deterministic-control#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-deterministic-control#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-deep-deterministic-control)
<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 一次策略更新，为什么能改善未来？

精确策略改善使用旧策略的真实价值；策略梯度使用与目标匹配的访问分布。值函数近似或梯度估计误差会破坏这些推理的前提。

函数逼近与深度方法：PPO 的动作概率比不等于新策略的状态访问比。连续动作 actor 还会追逐 critic 的误差。限制局部更新尺度与证明实际回报单调提高是不同要求。

持续学习中的研究问题：动作不仅改变世界，也改变未来数据与学习。比较冻结策略不等于比较持续更新的智能体；什么时候应付出当前回报去获得长期有用的经验？

[精确策略改善](https://yingwen.io/zh/continual-rl/foundations/tabular/dynamic-programming/) → [策略梯度定理](https://yingwen.io/zh/continual-rl/foundations/approximation/policy-gradient/) → [TRPO 与 PPO 的近似](https://yingwen.io/zh/continual-rl/foundations/deep/trust-region/) → [持续控制的比较器](https://yingwen.io/zh/continual-rl/algorithms/control/)


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)
- [最大熵控制](https://yingwen.io/zh/continual-rl/algorithms/soft-control/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

对应原始材料：Sutton & Barto §13.7；Deterministic Policy Gradient §3；TD3 §4–5。本文为原创讲解，原书、论文与上游代码保留各自许可。
