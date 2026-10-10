# 策略梯度、基线与 Actor–Critic

函数逼近与经典进阶方法 · 第 6 章

直接学习策略时，哪一个目标的梯度能由经验估计，近似从哪里进入？

## 本章内容

- 从轨迹概率推导带完整折扣权重的 REINFORCE。
- 证明动作无关基线不改变梯度，并定位 critic 引入的偏差。
- 区分普通梯度、自然梯度以及平均奖励目标的采样权重。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [MDP、回报与价值：序列决策的数学对象](https://yingwen.io/zh/continual-rl/foundations/tabular/mdps/)：明确策略决定轨迹分布及要优化的回报。
- [函数逼近预测：从回归到 TD 固定点](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/)：理解critic提供的是近似预测，不是已知真值。


### 可微随机策略

策略参数决定动作概率，而环境转移不直接依赖这些参数。需要策略具有合适支持，并满足交换求导与期望所需的正则条件。

$$
\pi_\theta(a\mid s),\qquad \psi_t=\nabla_\theta\log\pi_\theta(A_t\mid S_t)
$$

### 起点分布

本章先固定回合起点分布，优化从该分布出发的折扣回报。状态访问频率不是任意可替换的采样分布。

<a id="problem-definition"></a>

## 本章的问题定义

用可微随机策略直接参数化动作分布，不通过对动作价值取最大值间接改变行为。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 策略 $\pi_\theta(a\mid s)$、参数 $\theta$；完整轨迹或 on-policy 多步样本。

### 需要求解的对象

构造 $\nabla_\theta J(\pi_\theta)$ 的可用估计，并说明基线和 critic 如何改变估计性质。

### 信息与数据权限

无需知道或微分环境核；需要策略概率及其对参数的导数。

$$
\max_\theta J(\pi_\theta),\qquad \nabla_\theta J=\mathbb E\!\left[\sum_{t\ge0}\gamma^t\nabla_\theta\log\pi_\theta(A_t\mid S_t)\,G_t\right]
$$

$G_t$ 是从时刻 t 起的折扣回报。式中外层折扣对应初始状态分布下的目标；不能随意删掉再声称仍是同一精确梯度。

### 成立条件与解的含义

- 允许交换微分与积分；策略支持与光滑性满足推导条件。
- 基线须满足给定状态后的 score 消去条件；用独立数据拟合或在本次动作采样前固定可满足通常条件。同一样本拟合可能引入统计依赖，停止梯度不能消除它；有偏 critic 也不因此无偏。

判断准则：在可枚举轨迹上将估计器期望与有限差分梯度比较，再测量方差。

### 适用边界

- 随机梯度上升必然找到全局最优策略。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)：本章聚焦基础 score-function 梯度与完整 on-policy 回报；相关章还组合 actor–critic 与 PPO。那些后续近似不是本章精确梯度恒等式的前提。

- 改变评价目标 · [平均奖励与差分价值](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)：本章采用初始分布下的折扣目标；相对平均奖励梯度，其占用权重和价值对象不同。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

动作改变未来分布，而环境通常不可微；完整回报又可能噪声很大。

### 本章的核心思路

对轨迹概率求导，用 score-function 把不可微环境留在采样过程中，再用基线和 critic 控制方差。

1. [从概率乘积导出梯度](#lesson-derive)：先得到完整表达，再用因果性去掉动作无法影响的过去奖励。

2. [分开方差缩减与预测近似](#policy-baseline)：基线消去恒等式有明确条件，TD critic 的偏差需要另行分析。

3. [实现时固定采样与更新关系](#policy-algorithm)：采样策略、回报计算和参数更新时间必须与估计器声明一致。

结论与条件：无偏梯度估计的条件不等于非凸优化的全局收敛保证。

### 相关方法改变了什么

- REINFORCE / actor–critic：前者使用采样未来，后者用价值预测替代部分未来。

- 普通 / 自然梯度：改变参数空间的步进几何，不改变期望回报定义。


<a id="lesson-setting"></a>

## 1 · 直接优化策略的目标

此前的控制方法先学习动作价值，再依据这些价值选择动作。两个估计接近时，很小的数值变化就可能切换贪心动作；连续动作下，寻找最大动作还可能是一项昂贵的优化。另一条路线是直接表示动作分布，并让参数的小变化平滑地改变行为。策略梯度研究怎样从实际经验找到提高所规定目标的方向，而价值学习仍可帮助这项估计。

$$
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^tR_{t+1}\right].
$$

$T$ 可以是有限随机终点。策略通过动作改变未来状态分布，因此不能只把当前状态视为固定样本、对即时奖励求导。先写清目标，才能判断实现中的折扣与采样是否正确。

直接学习动作概率还有另一种用途。不同真实处境若在当前表示中不可区分，同一个确定动作可能在一处正确、在另一处错误。学习恰当的随机比例，有时比固定 ε 的探索规则更好。这里的随机性不仅是为了收集信息，也可能是受限策略类中的最好选择；本章后面的两步例子会把这一区别算清。

$$
p_\theta(\tau)=\mu(S_0)\prod_{t=0}^{T-1}\pi_\theta(A_t\mid S_t)p(S_{t+1},R_{t+1}\mid S_t,A_t).
$$

起点分布与环境转移不依赖 $\theta$，所以对轨迹对数概率求导时，只留下每一步的策略 score。环境无需可微。

<a id="lesson-derive"></a>

## 2 · 从轨迹导数到 REINFORCE

$$
\nabla J=\mathbb E\left[\left(\sum_t\psi_t\right)\left(\sum_k\gamma^kR_{k+1}\right)\right].
$$

使用对数导数恒等式，将未知的环境依赖包含在采样轨迹中。此时一个动作的 score 似乎乘上了全部奖励。

发生在动作之前的奖励，与当前动作的随机选择无关。给定动作前历史，策略 score 的期望为零，因此这些过去奖励对期望梯度的贡献为零。去掉它们减少无用方差，得到 reward-to-go。

$$
\begin{aligned}G_t&=\sum_{k=t}^{T-1}\gamma^{k-t}R_{k+1},\\ \nabla J&=\mathbb E\sum_{t=0}^{T-1}\gamma^tG_t\psi_t.\end{aligned}
$$

外层 $\gamma^t$ 与回报内部的相对折扣不是同一个因子。本页目标是从起点计算的折扣和；只保留 $G_t$ 而漏掉外层因子，一般不再是它的梯度。

$$
\begin{aligned}d_\gamma^\pi(s)&=\sum_{t\ge0}\gamma^t\Pr_\pi(S_t=s,\ t<T),\\ \nabla J&=\sum_s d_\gamma^\pi(s)\sum_a\nabla\pi_\theta(a\mid s)q_\pi(s,a).\end{aligned}
$$

这是策略梯度定理的占用权重表达。$s$ 只遍历非终止状态，$t<T$ 表示这一决策确实发生。$d_\gamma$ 是期望折扣访问次数，不是概率分布。

$$
\begin{aligned}Z_\gamma^\pi&=\sum_s d_\gamma^\pi(s)=\mathbb E_\pi\sum_{t=0}^{T-1}\gamma^t,\qquad \bar d_\gamma^\pi(s)=d_\gamma^\pi(s)/Z_\gamma^\pi,\\ \nabla J&=Z_\gamma^\pi\,\mathbb E_{S\sim\bar d_\gamma^\pi,\ A\sim\pi}[q_\pi(S,A)\nabla\log\pi_\theta(A\mid S)].\end{aligned}
$$

要求 $0<Z_\gamma^\pi<\infty$。有限终点且 $\gamma<1$ 时，$Z_\gamma^\pi=(1-\mathbb E[\gamma^T])/(1-\gamma)$；$\gamma=1$ 时为 $\mathbb E[T]$。持续交互、$T=\infty$ 且 $\gamma<1$ 时才直接得到 $1/(1-\gamma)$。

例如固定两步回合、$\gamma=.5$，两个决策时刻的占用质量分别为1和.5，总量1.5。只乘 $1-\gamma$ 会得到总量.75，尚未归一化；真正的归一化权重是 $2/3$ 和 $1/3$。也可以把终止后零奖励吸收状态的全部时间计入，这时总质量为 $1/(1-\gamma)$，但吸收状态不贡献策略梯度。两种约定的梯度一致，采样分布的定义却不能混用。

策略梯度定理并不是忽略了状态分布对参数的影响。递归展开价值导数后，这部分影响已被吸收到占用权重和动作价值中。随意用 replay 中的状态频率替换它，会引入另一个离策略估计问题。

下面用有限状态矩阵写出这个展开。回合任务的 $P_\pi$ 只保留非终止状态之间的转移，所以各行和至多为1；真正终止后的价值与导数取零。$\gamma<1$ 时，$I-\gamma P_\pi$ 可逆；$\gamma=1$ 时，还要求策略对矩阵中列入的每个状态都适当终止（proper），即从各状态出发的期望终止时间有限。仅在起点分布 $\mu$ 下有 $\mathbb E_\mu[T]<\infty$，不足以保证全矩阵可逆。求导还沿用前述正则条件。

$$
\begin{aligned}u(s)&=\sum_a\nabla_\theta\pi_\theta(a\mid s)q_\pi(s,a),\\\nabla_\theta v_\pi&=u+\gamma P_\pi\nabla_\theta v_\pi=(I-\gamma P_\pi)^{-1}u,\\\nabla_\theta J&=\mu^\top\sum_{t\ge0}\gamma^tP_\pi^t u.\end{aligned}
$$

对 vπ(s)=Σaπ(a|s)qπ(s,a) 求导。第一部分是策略概率的直接变化 u；第二部分沿下一状态继续传播价值导数。反复代入产生全部未来访问权重。μ 是起点分布，不是任意 minibatch 分布。

例如 A 一步终止，另有从 A 不可达的 B 永远自循环。若总从 A 开始，$\mathbb E_\mu[T]=1$，但完整非终止矩阵是 $P_\pi=\operatorname{diag}(0,1)$，$I-P_\pi$ 仍奇异。从起点产生的有限轨迹没有排除未访问闭类；不能据此前者直接写出全状态的无折扣逆矩阵。

例如一个动作先把智能体送到新区域，若只对当前动作奖励求导，就漏掉该区域里后续选择的后果。占用权重展开恰好把这些后果沿时间计入。策略梯度不是绕过了序列决策，只是把未知动力学的导数转成可以通过轨迹估计的统计量。

<a id="policy-baseline"></a>

## 3 · 基线为何无偏，critic 何时有偏

$$
\sum_a\pi_\theta(a\mid s)b(s)\nabla\log\pi_\theta(a\mid s)=b(s)\nabla\sum_a\pi_\theta(a\mid s)=0.
$$

只要基线在给定状态或动作前历史后不依赖当前采样动作，减去它就不改变期望 score 梯度。策略更新时对基线和回报停止梯度，不能把它们当作额外可微目标。

动作无关是一项统计条件，不能只看网络输入中有没有动作。如果先用当前样本的回报把基线拟合好，再对同一样本计算策略更新，拟合后的参数可能已泄露当前动作的结果。单步例子中若把基线直接设为本次奖励，每个优势都为零，而两个动作奖励不同时真实梯度并不为零。用动作发生前已确定的基线，或用适当独立数据拟合，可保留上面的条件无偏解释；仅停止自动微分不会消除这种数据依赖。

取基线为 $v_\pi(s)$ 得到优势 $A_\pi(s,a)=q_\pi(s,a)-v_\pi(s)$。状态价值是自然的选择，但不总是使所有参数方向总方差最小的基线；最优方差基线还可能依赖 score 的平方范数。

$$
\begin{aligned}\delta_t&=R_{t+1}+\gamma V_w(S_{t+1})-V_w(S_t),\\ \Delta\theta&=\alpha_\theta\gamma^t\delta_t\psi_t.\end{aligned}
$$

当 $V_w=v_\pi$ 且状态是合适的 Markov 状态时，条件期望 $\mathbb E[\delta_t\mid S_t,A_t]=A_\pi(S_t,A_t)$。近似 critic 不精确时，一步 bootstrap 通常带来策略梯度偏差。

MC 基线只减去动作无关项，仍可保持策略梯度无偏；把完整回报换成近似 TD 目标则是另一项近似。两者常一起出现，但作用不同。训练 critic 的损失也不能简单加到策略损失后任其通过所有共享路径反传。共享表示需要明确各项梯度如何作用。

$$
\begin{aligned}V_w(s)&=v_\pi(s)+\varepsilon(s),\\\mathbb E[\delta_t\mid s,a]&=A_\pi(s,a)+\gamma\mathbb E[\varepsilon(S_{t+1})\mid s,a]-\varepsilon(s).\end{aligned}
$$

将近似 critic 分成真实价值与误差，代入 TD 残差。当前状态的 −ε(s) 是动作无关基线，其 score 期望为0；但后继误差一般依赖当前动作，因此会改变策略的期望更新方向。

一个可手算的失败情形：起点两个动作等概率，真实即时及未来奖励全为0。critic 却把动作1所到后继状态估为10，把动作0的后继估为0；γ=.9，起点预测为0。两动作 TD 误差分别为9和0。对动作1概率的 Bernoulli logit，期望更新系数为 .5×9×.5=2.25，尽管真实策略梯度为0。Actor 会追逐 critic 的错误，这不是 MC baseline 的无偏性所能排除的。

$$
b^*(s)=\frac{\mathbb E[G_t\|\psi_t\|^2\mid S_t=s]}{\mathbb E[\|\psi_t\|^2\mid S_t=s]}.
$$

固定状态下，最小化 $\mathbb E[(G-b)^2\|\psi\|^2]$ 并对标量 $b$ 求导，得到使 score 估计总二阶矩最小的基线；分母须非零。只有相应加权关系成立时，它才等于 $v_\pi(s)$。所以“动作无关则无偏”与“取价值函数最小方差”不是同一条结论。

从这里进入深度 actor–critic，应分别追问：critic 在哪个状态分布下准确；其后继误差是否诱导错误行动；策略改变后旧 critic 还能否使用。在持续流中，策略与表示会同时漂移。训练多做几轮可能减小当前拟合误差，也可能扩大与当前行为的数据失配，不能只把它当作免费的优化改进。

<a id="policy-algorithm"></a>

## 4 · 数据、梯度与参数更新的顺序

**算法：回合批量 REINFORCE**

1. 用当前固定策略采样一个完整回合，保存动作和采样时的策略信息
1. 从末尾递推 $G_t=R_{t+1}+\gamma G_{t+1}$，真正终止处尾项为零
1. 在采样参数下计算 score 和动作无关基线
1. 累加 $g=\sum_t\gamma^t(G_t-b_t)\nabla\log\pi_\theta(A_t\mid S_t)$
1. 执行一次策略更新；基线可用回报作为监督目标单独更新

若在回合结束后对同一批数据做多次大幅策略更新，后续参数已偏离采样策略。原始在策略梯度等式不再自动适用，需要重要性修正、合适的近似目标或重新采样。本页实现返回梯度而不在枚举过程中改变策略，便于与解析导数比较。

一步 actor–critic 则可以每步更新。它以更早反馈和较低方差换取 critic 近似误差，通常还有两个步长或时间尺度。优势标准化、熵奖励、梯度裁剪和目标网络都属于额外机制，应在这个基本估计器之外逐项解释。

<a id="experiment-deep-vpg"></a>

### 实验：实验 · 完整回报与一步 critic 提供不同的策略学习信号

策略梯度的目标相同，等待回报与借助 critic 会怎样改变有限预算学习？

**环境与可用信息。** DeadlineChain：位置 0–4、左右动作、边界截断。观测为五维位置 one-hot 加剩余时间比例。到位置 4 得 1 并终止；其他步得 −0.02。12 步截止也是任务真实终止，且剩余时间可观测。每回合重新从位置 0 开始，网络跨回合保留。

**设置。** 1200 个真实步；actor 为 6→32 tanh→2，critic 为 6→32 tanh→1；PyTorch 默认随机初始化，Adam 的 actor 步长 0.003、critic 步长 0.01。VPG 每个完整回合更新一次；A2C 每 60 步用收集到的一步 TD 目标更新一次。γ=1。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** vpg.py 用完整 reward-to-go 减旧 critic 得到优势；a2c.py 用一步自举 target 减旧 critic。两者策略更新都停止优势的梯度。VPG 未完成的末尾回合不参与更新。

**测量。** 用隔离环境的贪心策略回报评价，而不是训练中的随机策略回报。还需读取 updated_batches：相同环境步数并没有固定 optimizer 更新次数。

```bash
python3 implementations/deep/vpg.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/deep-vpg/curves.svg)

横轴：训练环境步（独立评估交互另计）。纵轴：冻结策略的外部回报。每种方法 1200 训练环境步；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 冻结当前 actor，在独立 DeadlineChain 中按 argmax 动作重复12次完整回合，取未折扣外部回报的平均。每次都从位置0、剩余12步开始；到位置4奖励1，其余步奖励−.02。评价环境和动作选择都确定，同一网络的12次回合相同，不能当成12个训练种子。

**step：怎样计时。** step 只数训练环境转移，包含未完成回合的 pending_samples；评价交互另计。到位置4或用尽任务的12步时，VPG用该完整回合做一次actor梯度和一次critic梯度，updated_batches因此是已更新回合数。预算末尾未完成回合不作Monte Carlo更新，不能把预算截断当成终止。A2C对照每60步更新一批，1200步有20次actor和20次critic梯度，相同环境预算没有对齐更新次数。

**怎样汇总。** 取该记录时刻的值；不先对曲线上的时间点求平均。 第0步评价尚未更新的网络；后续每60步记录当前网络，可能仍在等待下一完整回合。VPG用γ=1的reward-to-go减去采样时的价值基线，没有优势归一化；A2C用一步自举。图中argmax回报与随机行为策略的期望回报分别定义，两方法还改变了等待长度与优化时钟。这是单环境教学实现，不是论文基准复现。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** CSV 可重算冻结回报均值和样本标准差；updated_batches给出VPG的actor、critic各自梯度次数，samples−pending_samples给出已用于完整回合更新的转移数。已知确定性任务下，成功时回合长度L=1+(1−value)/.02，失败时L=12；12×L给本检查点额外评价转移数。未保存训练奖励、回合回报、基线、优势或策略概率，不能恢复训练累计收益、随机行为策略回报或完整梯度。policy_loss和value_loss来自最近完成回合（A2C为最近批次），不是检查点间的平均。

计算位置：[deep/vpg.py](https://yingwen.io/crl-code/implementations/deep/vpg.py) · [deep/a2c.py](https://yingwen.io/crl-code/implementations/deep/a2c.py) · [deep/_common.py](https://yingwen.io/crl-code/implementations/deep/_common.py)

</details>

**结果分析。** 第 600 步 VPG 均值为 0.94，A2C 为 0.704，后者 seed 标准差约 0.528；第 1200 步两者都为 0.94。本配置说明 critic 引入后不必更早学好，但两者最终都达到最短路径行为。

**结论边界。** 更新频率和监督目标同时不同，不能把全部差异只归因于 bootstrap。小环境上的终点打平也不意味着估计器方差、随机策略和计算开销相同。

**继续实验。** 冻结一组完整轨迹，先比较两种优势与梯度，再做新的独立交互比较。固定样本导数正确与重新采样后表现更好，是两个命题。

[源码](https://yingwen.io/crl-code/implementations/deep/vpg.py) · [逐种子记录](https://yingwen.io/crl-code/results/deep-vpg/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/deep-vpg/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/deep-vpg/curves.json)

<a id="policy-continuous"></a>

## 5 · 连续动作：高斯策略的两个 score

连续动作可以使用高斯策略 $A\sim\mathcal N(\mu_\theta(s),\sigma_\theta(s)^2)$。均值控制偏好的动作，标准差控制随机程度。用对数标准差 $\ell=\log\sigma$ 参数化，可保证标准差为正。

$$
\begin{aligned}\partial_\mu\log\pi(a\mid s)&=\frac{a-\mu}{\sigma^2},\\ \partial_\ell\log\pi(a\mid s)&=\frac{(a-\mu)^2}{\sigma^2}-1.\end{aligned}
$$

把这两个局部导数通过均值与对数标准差网络继续链式求导，就得到策略 score。它们来自概率密度的导数，不要求奖励对动作可微。

真实执行动作若经过裁剪或可逆变换，必须说明策略密度究竟定义在哪个动作变量上。可逆 tanh 变换需要密度的 Jacobian 修正；直接把裁剪后的动作代入未裁剪高斯密度，通常不是执行策略的正确似然。本页枚举实验使用离散策略，不实现连续控制训练。

<a id="rlss-policy-mixture"></a>

## 策略梯度的适用范围由 score 决定，不由网络名称决定

策略梯度并不要求策略是 softmax 神经网络。它要求明确定义目标、轨迹分布和可计算的动作概率导数；还需要允许交换求导与积分、充分的矩条件及相应支持条件。相同 score 可以接收完整回报、差分回报或近似 TD 信号，但这不会使这些信号的偏差自动相同。

考虑由固定行为组件 $\pi_1,\ldots,\pi_K$ 构成的混合策略。只学习混合权重 $m_j=\operatorname{softmax}(\eta)_j$，令 $\pi(a|s)=\sum_jm_j(s)\pi_j(a|s)$。固定组件可以代表不同探索尺度或不同已有技能；此处不对其内部参数求导。

$$
\frac{\partial\log\pi(a|s)}{\partial\eta_j}
=\underbrace{\frac{m_j(s)\pi_j(a|s)}{\sum_km_k(s)\pi_k(a|s)}}_{\Pr(j\mid s,a)}-m_j(s).
$$

先对混合概率求导，再除以该概率。后验责任减去先验权重，就是这个层次的 score。它不同于直接把“被抽中的组件”的一热编码当作已观察动作的后验责任。

两种实现都可以严谨。若保存了实际抽中的组件编号，可对“先抽组件、再抽动作”的联合分布使用组件 score；若只保存最终动作，则上式是边缘动作分布的 score。它们在条件满足时估计同一目标梯度，但方差和需要记录的变量不同。混合离散动作点质量与连续密度时，还必须使用一致的测度，不能把数值概率与密度直接相加。

这条路线连接到 options，但尚不是完整时间抽象。每步重新混合组件只是一个平坦的随机策略；承诺执行某个组件多步，会改变轨迹分布和信用分配。若组件本身持续学习，混合权重、组件策略与 critic 又形成三个相互变化的对象。应分别记录“选谁”“选出的策略如何变化”“如何估值”，而非把它们都称为 actor 更新。

<a id="lesson-example"></a>

## 6 · 四条轨迹的精确梯度检查

两步决策共用一个 Bernoulli 参数 $p=\sigma(\theta)$，每一步独立抽样。策略不读取阶段或先前动作，所以两步都以同一个概率 p 选择动作1，以 1−p 选择动作0。第一步奖励为0；只有动作序列 (1,0) 在第二步得到奖励1，其余三条轨迹奖励为0。

$$
\begin{aligned}J(\theta)&=\gamma p(1-p),\\ \frac{dJ}{d\theta}&=\gamma p(1-p)(1-2p),\\ \nabla\log\pi(A_t)&=A_t-p.\end{aligned}
$$

先直接写出四条轨迹的概率，只有一条贡献收益。对 sigmoid 求导即可得到解析答案。

取 $\theta=0.7,\gamma=0.5$，则 $p\approx0.668188,J\approx0.110856$，解析梯度约为 −0.0372894。对获奖轨迹，$G_0=0.5,G_1=1$，两次 score 都应乘上总权重 0.5。

![四条两步轨迹的概率树，唯一获奖轨迹与解析目标曲线上的一次精确期望梯度更新。](https://yingwen.io/crl-figures/concept-classic-policy-gradient.svg)

原创精确枚举图。分支宽度表示条件动作概率，末端数字是完整轨迹概率；沿用 $\theta=0.7,\gamma=0.5$。另给一步教学步长 $\alpha=5$，期望梯度使 $\theta$ 变为约 0.513553、动作 1 概率从 0.668 变为 0.626。下方是解析 $J(\theta)$，不是采样训练结果；此步收益上升可直接代入核对，不是任意步长保证。

代码分别用轨迹枚举、解析公式和目标函数有限差分求梯度，三者相等。再减去常数基线 3，单条轨迹的梯度变化，但四条轨迹的概率加权期望不变。这个例子同时检查外层折扣、reward-to-go 和基线。

它也说明为什么有时要学习随机策略。取 $\gamma>0$，在这个共用概率的无记忆策略类中，两个确定性极限 $p\to0$ 或 $p\to1$ 的收益都趋于零；$p(1-p)=1/4-(p-1/2)^2$ 表明最优选择是 $p=1/2$，收益为 $\gamma/4$。这里即使已知全部奖励，也应该保留随机性，它不是尚未探索完的表现。

若让策略知道当前是第一步还是第二步，就能确定地依次选择1、0，获得 $\gamma$。因此“受限表示下随机策略更好”没有推翻有限 MDP 中存在最优确定策略的结论：原先的策略类根本表达不了这个按阶段区分的策略。学习动作概率改善的是现有表示下的行为；构造更有信息的 agent state 改变的是可选择的策略类。

<a id="policy-occupancy-example"></a>

### 6.1 · 同一个例子：占用怎样收集全部 score

继续使用上面的四条轨迹，只把环境已经保存的信息写出来。起点为 $s_0$；第一步动作 $0$、$1$ 分别到达 $s_1^{(0)}$、$s_1^{(1)}$，第二步后终止。仅在 $s_1^{(1)}$ 选择动作 $0$ 得奖励 $1$。这三个状态使环境满足 Markov 性；actor 仍在三个状态共用同一个 logit，不读取阶段或先前动作。后面的精确 critic 与基线可以使用记录中的完整状态，actor 的策略类没有因此扩大。

现在先求固定策略的条件价值，再乘访问质量。第一步选 $1$ 尚未获奖，但到达后以概率 $1-p$ 选择 $0$，所以 $q_\pi(s_0,1)=\gamma(1-p)$。在 $s_1^{(1)}$ 则有 $q_\pi(s_1^{(1)},0)=1$。三个状态的 score 仍都是 $a-p$。

| 环境状态 | 折扣占用 $d_\gamma$ | 动作价值 $(q(0),q(1))$ | 占用加权梯度贡献 |
| --- | --- | --- | --- |
| $s_0$ | $1$ | $(0,\gamma(1-p))$ | $\gamma p(1-p)^2$ |
| $s_1^{(0)}$ | $\gamma(1-p)$ | $(0,0)$ | $0$ |
| $s_1^{(1)}$ | $\gamma p$ | $(1,0)$ | $-\gamma p^2(1-p)$ |

$$
\nabla J=\underbrace{\gamma p(1-p)^2}_{s_0}+\underbrace{0}_{s_1^{(0)}}-\underbrace{\gamma p^2(1-p)}_{s_1^{(1)}}=\gamma p(1-p)(1-2p).
$$

起点的贡献倾向增加动作 $1$，后续获奖可达状态的贡献倾向增加动作 $0$。共享一个参数时，这两个要求在同一坐标中相加。$\theta=0.7$ 下，贡献约为 $+0.036784$ 与 $-0.074073$，合计为 $-0.037289$。

看图前先预测：如果只保留第一步 score，参数会向哪里移动？第一步奖励虽为零，它仍有正的梯度贡献，因为动作价值包含未来奖励。漏掉第二步 score 则会把当前参数推向相反方向。占用总质量是 $Z=1+\gamma=1.5$；若改用归一化状态分布，需先除以 $1.5$，最后再乘回 $1.5$ 才得到同一个梯度。

![相同共享策略下的三个环境状态、四条轨迹概率，以及起点和后续状态方向相反的梯度贡献。](https://yingwen.io/crl-figures/policy-gradient-walkthrough-occupancy.svg)

上图沿实线读动作与终止奖励，线宽表示条件动作概率；节点下方是折扣占用，末端依次为奖励、动作序列与完整轨迹概率。下图用同一有符号轴画未乘步长的参数贡献。原创精确枚举，$\theta=0.7,\gamma=0.5$；[原始数值](/crl-figures/policy-gradient-walkthrough-data.json)、[计算核](/crl-code/figures/policy-gradient-walkthrough.mjs)与[单文件教程](/crl-code/tutorials/policy-gradient-walkthrough.py)可逐项复算。

<a id="policy-variance-example"></a>

### 6.2 · 无偏、局部最小方差与整回合方差

基线无偏性已经证明，接着问应该选多大。固定一个决策状态，令 $Y_b=(G_t-b)\psi_t$。其均值不随 $b$ 改变，因此最小化方差等价于最小化二阶矩。对 $b$ 求导时，回报和 score 都来自固定的当前策略，得到前面的 score 平方加权公式。外层 $\gamma^t$ 在这个状态固定，不改变最优 $b$。

$$
\frac{d}{db}\mathbb E[Y_b^2\mid s]=-2\mathbb E[(G_t-b)\psi_t^2\mid s]=0.
$$

这次只有一个策略参数，$\psi^2$ 就是平方范数。分母为 $p(1-p)>0$。在获奖可达状态，获奖动作 $0$ 的 score 为 $-p$，未获奖动作 $1$ 的 score 为 $1-p$，两者平方通常不同。

| 状态 | 价值基线 $v_\pi$ | 该时刻的最优基线 $b^*$ |
| --- | --- | --- |
| $s_0$ | $\gamma p(1-p)$ | $\gamma(1-p)^2$ |
| $s_1^{(0)}$ | $0$ | $0$ |
| $s_1^{(1)}$ | $1-p$ | $p$ |

在 $s_1^{(1)}$ 使用 $b^*=p$ 时，动作 $0$ 给出 $(1-p)(-p)$，动作 $1$ 给出 $(-p)(1-p)$。两个样本的梯度相同，该状态的条件方差为零。价值基线为 $1-p\approx0.331812$，局部最优基线为 $p\approx0.668188$；二者只有在 $p=1/2$ 时重合。起点的最优基线则约为 $0.055050$，也不同于价值 $0.110856$。

但一次回合更新会把两个时刻的项相加。令 $g_0=(G_0-b(s_0))\psi_0$、$g_1=\gamma(G_1-b(S_1))\psi_1$。即使分别把它们的方差降低，整回合的方差还包含协方差：

$$
\operatorname{Var}(g_0+g_1)=\operatorname{Var}(g_0)+\operatorname{Var}(g_1)+2\operatorname{Cov}(g_0,g_1).
$$

两个 score 由独立动作抽样得到，但回报与下一状态把两个梯度项连接起来，因此两项不独立。单时刻最优公式没有优化这一协方差。

| 每个状态采用的基线 | 整回合梯度期望 | 整回合梯度方差 | 两项协方差 |
| --- | --- | --- | --- |
| 零基线 | $-0.037289$ | $0.004881$ | $-0.009564$ |
| 真实价值 | $-0.037289$ | $0.006159$ | $-0.002762$ |
| 各状态的局部 $b^*$ | $-0.037289$ | $0.006802$ | $0$ |

这里逐状态局部最优基线减小了每个状态的条件方差，却破坏了零基线下较强的负协方差，使整回合方差增大。价值基线也没有在这个策略参数处降低整回合方差。它们仍然无偏；这个精确反例说明“基线不改变期望”“局部最优”“整回合最优”是三个需要各自计算的判断。控制变量观点可继续读 [Greensmith 等 §5](https://www.jmlr.org/papers/v5/greensmith04a.html)，原论文分析的平均奖励 GPOMDP 与时序估计器不同，本页数值由当前有限回合独立推导。

<a id="policy-critic-example"></a>

### 6.3 · 同一个 critic 误差何时只是基线，何时反转行动方向

保持策略冻结，把三个状态的 critic 写成 $V=v_\pi+\varepsilon$，终止价值仍为零。若仅用它减去完整回报，任意固定的三个误差都作为动作前基线进入，期望梯度不变。若用它构造一步 TD，起点动作则决定读到哪个后继预测，这部分误差会进入动作比较。

$$
\begin{aligned}e&=\varepsilon(s_1^{(1)})-\varepsilon(s_1^{(0)}),\\ \mathbb E[g_{\mathrm{TD}}]-\nabla J&=\gamma p(1-p)e,\\ \mathbb E[g_{\mathrm{TD}}]&=\gamma p(1-p)[1-2p+e].\end{aligned}
$$

第一步的当前状态误差是基线而抵消；两个后继误差的差乘上起点 score 留下来。第二步已真正终止，误差只位于当前状态项，仍可抵消。这里每一步使用同一份冻结 critic 和采样参数，并保留外层折扣。

因此起点误差设为 $7$，或两个后继同加 $0.3$，都不会改变期望梯度。只把 $s_1^{(1)}$ 高估 $0.5$ 时，偏差约为 $+0.055428$，TD actor 的期望变为 $+0.018139$，而真实梯度仍为 $-0.037289$。向右更新会增加 $p$，恰好远离该策略类的最优概率 $1/2$。critic 不必每个状态都精确才可能无偏；本例要求的是两个后继误差相同。

图中先比较局部与整回合方差，再把横轴换成后继误差差值。预测一下：在 $e=2p-1\approx0.336376$ 处，TD actor 的期望应落在哪里？代入后为零；超过它就改变方向。完整回报的 MC 曲线保持不动，因为同一个 critic 在那里只承担基线的作用。

![单时刻基线方差曲线、三个整回合方差和后继critic误差引起的策略梯度反号。](https://yingwen.io/crl-figures/policy-gradient-walkthrough-variance-bias.svg)

左上曲线是在 $s_1^{(1)}$ 条件下的单项方差，圆标区分真实价值与局部最优基线；另一个面板画完整回合两项之和的方差，三个期望相同。下图蓝虚线为完整回报减固定 MC 基线，橙实线为一步 TD；后继误差差 $e=0.5$ 使后者从负变正。原创确定性枚举；无采样训练，坐标和全部协方差由[单文件教程](/crl-code/tutorials/policy-gradient-walkthrough.py)独立计算。

这里的偏差来自 actor 使用了有误差的后继条件价值，不取决于 critic 当初用 MC 还是 TD 拟合。增加 critic 拟合精度是否有帮助，要检查误差在哪个动作的后继上出现；一个整体均方误差数值尚不能说明这个误差差值。

<a id="policy-natural"></a>

## 7 · 自然梯度与平均奖励的不同几何

$$
\begin{gathered}D_{\mathrm{KL}}(\pi_\theta\|\pi_{\theta+\Delta\theta})\approx\tfrac12\Delta\theta^\top F\Delta\theta,\\ F=\mathbb E[\psi\psi^\top].\end{gathered}
$$

Fisher 矩阵衡量参数变化引起的局部策略分布变化。期望用哪个状态分布，必须与所研究的局部约束说明一致。

$$
\begin{gathered}\max_{\Delta\theta}g^\top\Delta\theta\quad\text{s.t.}\quad\tfrac12\Delta\theta^\top F\Delta\theta\le\varepsilon\\ \Longrightarrow\quad\Delta\theta\propto F^{-1}g.\end{gathered}
$$

自然梯度来自这个局部线性目标与二次 KL 约束。$F$ 可能奇异，实际常用伪逆或阻尼；有限步长、估计误差和近似求解都影响真实性能。

在单步 Bernoulli bandit 中，$F=p(1-p)$，若两个动作奖励差为 1，普通梯度为 $p(1-p)$，自然梯度为 1。它改变坐标尺度，但并不让任意有限更新都安全，也不保证全局最优。

兼容函数逼近进一步回答：critic 不完全准确时，能否仍给出准确的期望策略梯度？固定当前策略，令 $\psi(s,a)=\nabla_\theta\log\pi_\theta(a\mid s)$，选择 $f_w(s,a)=w^\top\psi(s,a)$，并按同一策略梯度所用的占用权重拟合动作价值。关键要求是拟合误差在这些 score 方向上正交。

$$
\begin{aligned}0&=\sum_s d_\gamma^\pi(s)\sum_a\pi(a\mid s)\psi(s,a)[q_\pi(s,a)-f_w(s,a)],\\ F_d&=\sum_s d_\gamma^\pi(s)\sum_a\pi(a\mid s)\psi(s,a)\psi(s,a)^\top,\\ \nabla J&=\sum_s d_\gamma^\pi(s)\sum_a\pi(a\mid s)\psi(s,a)f_w(s,a)=F_dw.\end{aligned}
$$

第一式是占用加权平方拟合在最优点的正规方程；将它从策略梯度定理中减去，便得到最后一行。因此需要精确的是相关误差矩，不必让每个动作价值都被精确表示。这是 Sutton 等 1999 年兼容逼近结果的线性 score 形式。

由于每个状态下 $\mathbb E_\pi[\psi\mid s]=0$，$f_w$ 的动作均值为零，更适合解释为优势近似；加入任意状态基线不改变上述误差矩。若上一式的 Fisher 使用归一化占用，即 $F=F_d/Z_\gamma^\pi$，且矩阵可逆，则 $F^{-1}\nabla J=Z_\gamma^\pi w$。省略归一化常数只能声称方向相同。有限数据、尚未拟合完成的 critic、错误的状态权重，或任意神经网络结构，都不会自动满足这项兼容条件。

平均奖励策略梯度使用平稳状态权重与差分动作价值。在适当遍历条件下，目标是长期奖励率，不是从一个起点累计的折扣和。因此不能把去掉外层折扣的回合更新直接叫作平均奖励算法；奖励率、差分 critic 与采样协议都需要重新定义。

<a id="lesson-code"></a>

## 8 · 完整枚举与数值梯度代码

高斯 score、Bernoulli REINFORCE、四条轨迹枚举与有限差分

```python
def gaussian_scores(action, mean, log_std):
    """Scores with respect to mean and log standard deviation, before transforms."""
    inverse_variance = math.exp(-2 * log_std)
    error = action - mean
    return error * inverse_variance, error * error * inverse_variance - 1


def sigmoid(theta):
    if theta >= 0:
        return 1 / (1 + math.exp(-theta))
    value = math.exp(theta)
    return value / (1 + value)


def reinforce_gradient(actions, rewards, probability, gamma, baseline=0.0):
    """Bernoulli policy with shared scalar logit. Return gradient, not parameter update."""
    if len(actions) != len(rewards) or not 0 < probability < 1:
        raise ValueError("trajectory sizes or policy support invalid")
    result, return_to_go = 0.0, 0.0
    for t in reversed(range(len(rewards))):
        return_to_go = rewards[t] + gamma * return_to_go
        score = actions[t] - probability
        result += gamma ** t * (return_to_go - baseline) * score
    return result


def exact_policy_gradient(theta, gamma=0.5, baseline=0.0):
    # Two decisions, reward 1 iff first action=1 and second action=0.
    # J(theta)=gamma*p*(1-p); enumerate all four trajectories exactly.
    probability, expectation = sigmoid(theta), 0.0
    for actions in itertools.product((0, 1), repeat=2):
        mass = math.prod(probability if a else 1 - probability for a in actions)
        rewards = [0.0, float(actions == (1, 0))]
        expectation += mass * reinforce_gradient(actions, rewards, probability, gamma, baseline)
    return expectation


def policy_demo():
    theta, gamma, eps = 0.7, 0.5, 1e-5
    objective = lambda z: gamma * sigmoid(z) * (1 - sigmoid(z))
    p = sigmoid(theta)
    return {"probability": p, "J": objective(theta),
            "enumerated_gradient": exact_policy_gradient(theta, gamma),
            "analytic_gradient": gamma * p * (1 - p) * (1 - 2 * p),
            "finite_difference": (objective(theta + eps) - objective(theta - eps)) / (2 * eps),
            "with_baseline_3": exact_policy_gradient(theta, gamma, baseline=3.0),
            "bandit_Fisher": p * (1 - p), "bandit_natural_gradient_reward_gap_1": 1.0}
```

sigmoid 采用分支避免大幅正负输入时溢出。梯度函数要求动作支持在数值上非退化。这个标准库例子没有自动微分，也没有环境模拟误差，所以三种梯度之间的差异能直接定位公式或代码问题。它不是通用连续控制 runner。

运行策略梯度的三重核对

```sh
python3 approximation_textbook_lab.py policy-gradient
python3 approximation_textbook_lab.py test
```

进一步下载[独立标准库教程](/crl-code/tutorials/policy-gradient-walkthrough.py)，将这个单文件保存为 policy-gradient-walkthrough.py，在文件所在目录运行。它用 50 位 Decimal 枚举四条轨迹，逐项打印占用、score 贡献、两项方差与协方差、critic 误差及两个错误实现。无外部文件或第三方库。

同一个例子的条件量与完整回合量

```bash
python3 policy-gradient-walkthrough.py
python3 policy-gradient-walkthrough.py --json
```

运行前手算三个预测：减去常数 $3$ 后均值是否改变；漏掉外层折扣后的梯度是多少；先把当前样本回报直接设为该样本基线会怎样。答案分别是均值不变、约 $-0.111362$、全部样本梯度归零。最后一项已经依赖动作后的结果，无法使用基线抵消证明。JS 计算核与独立 Decimal 实现逐路径交叉核对，有限差分则直接检查所规定的 $J$。

<a id="lesson-branches"></a>

## 9 · 从经典 actor–critic 到深度与持续学习

现代 PPO、SAC 等方法在目标、数据复用和策略约束上作了不同选择。理解它们之前，应能指出一个实现中的采样策略、被优化目标、critic 目标、终止处理以及停止梯度位置。算法名字无法替代这些定义。

持续学习进一步让策略、状态表示和环境同时变化。旧优势估计和旧状态特征可能失效；在线元学习还对学习更新本身求导。此时要分清策略的环境梯度估计与元参数通过学习过程产生的梯度，避免把截断、近似和忽略的依赖隐藏在自动微分图中。

最小研究起点是固定两步环境，通过枚举确认目标和梯度一致，再加入 critic 近似、状态别名、环境变化和计算预算。每次增加一种机制，都保留能够反驳错误实现的解析测试。大环境里的回报提高，不能代替这个正确性检查。

<a id="lesson-check"></a>

## 10 · 练习与答案

- 为什么 $G_t$ 内已折扣，前面还需要 $\gamma^t$？答案：$G_t$ 从当前时刻计时，而目标 J 从回合起点计时，两个时间原点相差 t。
- 减去动作相关基线仍自动无偏吗？答案：不。动作求和不再能提出基线，需要额外校正。
- critic 只用于 MC 基线，与用 TD 替换回报有何不同？答案：前者可只改变方差，后者通常还引入 bootstrap 近似偏差。
- 自然梯度方向为何不保证每一步收益上升？答案：推导使用局部线性目标与局部二次 KL，有限步长和估计误差可能使近似失效。



<a id="chapter-code"></a>

## 下载与运行

标准库教学实现。命令运行本章的可解析小问题和全文件的公式测试；不依赖深度学习库，不等同于论文中的大规模实验。

[下载 approximation_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/approximation_textbook_lab.py)

```sh
python3 approximation_textbook_lab.py policy-gradient
python3 approximation_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：Part II 第 9–13 章。原书建立函数逼近预测、控制、离策略稳定性、资格迹与策略梯度的共同框架。

- [Sutton et al. · Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)：策略梯度与兼容函数逼近的原始论文。区分起点折扣权重和平均奖励平稳分布。

- [Greensmith, Bartlett & Baxter · Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning](https://www.jmlr.org/papers/v5/greensmith04a.html)：从控制变量角度分析基线与 actor–critic。原论文的 GPOMDP 估计器和时序方差分析比本章单状态 score 基线算例更一般；不可将局部最小二阶矩公式直接当作整个轨迹的最小方差保证。

- [Kakade · A Natural Policy Gradient](https://proceedings.neurips.cc/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html)：自然策略梯度的原始工作，连接策略空间的局部度量与兼容逼近。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-approximation-policy-gradient#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-approximation-policy-gradient#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-approximation-policy-gradient)
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
- [平均奖励与差分价值](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)
- [元学习与学习规则的适应](https://yingwen.io/zh/continual-rl/algorithms/meta/)

对应原始材料：第 13 章：Policy Gradient Methods。本文为原创讲解，原书、论文与上游代码保留各自许可。
