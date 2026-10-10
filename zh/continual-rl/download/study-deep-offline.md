# 离线强化学习：数据支持、策略评估与保守改进

现代深度强化学习 · 并列研究分支

不能补采数据时，怎样判断策略好坏，怎样避免利用没有证据的高价值动作？

## 本章内容

- 把数据支持假设写入 IS 与 doubly robust 评估。
- 推导 CQL 正则和 IQL expectile 的不同作用。
- 分开学习策略、选择超参数与独立评估。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [离策略函数逼近：覆盖、发散与稳定更新](https://yingwen.io/zh/continual-rl/foundations/approximation/off-policy/)：理解支持、采样分布与离策略估计。
- [深度价值学习：DQN、Double DQN 与目标的时间顺序](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/)：理解函数逼近怎样给未见动作产生估计。


### 条件概率与期望

区分随机变量、一次样本与条件分布；可以使用 Bayes 法则和全期望公式。

### MDP 与价值

理解状态、动作、转移、奖励、策略和 Bellman 递推；学习数据可以来自不同策略。

### 优化与梯度

知道固定目标、参数梯度、约束和期望目标的含义；神经网络只是函数表示的一种选择。

<a id="problem-definition"></a>

## 本章的问题定义

只能使用固定数据集，学习期间不能向环境补采反例。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 数据集 $\mathcal D$、其采集协议及可用行为概率；候选策略类 $\Pi$。

### 需要求解的对象

从数据中选出可支持的高价值策略，并估计它的真实性能。

### 信息与数据权限

未被数据支持的行为没有自动可识别的价值；行为概率未知时，某些 IS 估计器不能直接使用。

$$
\hat\pi=\mathcal A(\mathcal D)\in\Pi,\qquad J(\hat\pi)\ \text{under the real environment}
$$

外部控制目标没有变，改变的是数据权限。CQL 和 IQL 的损失是不同保守近似机制；它们不能为任意无覆盖区域创造可辨识性。

### 成立条件与解的含义

- 评价和改进结论需覆盖、函数可实现性或模型误差等条件。
- 策略选择与最终评价最好使用分离数据或适当选择校正。

判断准则：在允许的独立评估下报告真实收益及估计不确定性，并检查对数据外动作的依赖。

### 适用边界

- 仅凭固定数据确定任意未访问动作的真实效果。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 改变信息或数据协议 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：禁止继续交互，无法通过在线试验纠正夸大的动作价值。

- 组合不同学习问题 · [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)：历史数据可以帮助保留，但数据可见性不等于新情境覆盖。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

策略改进会偏好估计高但缺乏证据的动作，且无法在线纠正。

### 本章的核心思路

分开离线策略评价和保守策略改进，明确各自使用的信息假设。

1. [校正可评价行为的估计](#lesson-derive)：Doubly robust 结合预测与残差，但仍需其条件。

2. [限制缺乏证据的高值](#lesson-cql)：CQL 在 critic 中加入保守项，不等于已经知道数据外真值。

3. [在数据动作内组织改进](#lesson-iql)：IQL 避免直接对任意未见动作最大化，并通过优势加权提取策略。

结论与条件：保守性结论只在相应假设和优化条件下成立；无支持数据下存在根本不可识别性。

### 相关方法改变了什么

- OPE / offline control：前者评价给定策略，后者还根据数据选择策略。

- CQL / IQL：分别限制 critic 的数据外高值与在数据支持内近似策略改进。


<a id="lesson-setting"></a>

## 1 · 固定数据改变了纠错方式

在线价值学习可能作出错误行动，但新结果有机会纠正旧估计。若只有既存日志，这条反馈通路就被切断了：优化器可以不断提高一个未见动作的预测值，数据却永远不给出它的后果。更强的函数逼近既能带来有用泛化，也能让没有证据的乐观外推更容易被策略利用。离线学习因此必须同时讨论数据支持、策略改善和独立评价。

Offline RL 用已经收集的数据学习策略，训练过程中不能任意向环境补采样。Off-policy 只说明行为策略与目标策略不同；在线 SAC 也可以 off-policy，因此两个概念不等价。设计者决定数据集、保留字段、覆盖和评估权限；算法不能从没有观测的行为结果中产生证据。

$$
\mathcal D=\{\tau_i\}_{i=1}^n,\quad\tau_i\sim\mu,\qquad J(\pi)=\mathbb E_\pi\sum_{t=0}^{T-1}\gamma^tR_{t+1}
$$

先考虑完整有限轨迹、相同环境、已记录行为概率与固定目标策略。数据缺失或策略依赖隐藏信息会改变可识别性。

$$
\pi(a\mid h)>0\ \Longrightarrow\ \mu(a\mid h)>0
$$

支持条件必须沿目标可能访问的历史成立。只在初始状态有共同动作支持还不够。

<a id="lesson-support-walkthrough"></a>

### 从入口走到一个没有记录后果的动作

固定一个两步任务。起点只有 go，以奖励零到达状态 s；s 有三个动作，全部在收到奖励后真正终止。我们给读者的任务规格是 $r(s,a_0)=0$、$r(s,a_1)=2$、$r(s,a_2)=-2$，折扣 $\gamma=0.9$。学习器只拿到四条完整日志：两条末步选 $a_0$，两条末步选 $a_1$；每条都先走 go。行为者在 s 按 $(0.5,0.5,0)$ 选择，动作概率已记录。$a_2$ 的奖励是读者检查用的真值，不作为学习输入。

| 状态动作 | 日志次数 | 可见奖励与后继 | 学习器能用的证据 |
| --- | --- | --- | --- |
| 起点，go | 4 | 0，s，未终止 | 到达 s 的转移 |
| s，$a_0$ | 2 | 0，真正终止 | 末步标签 0 |
| s，$a_1$ | 2 | 2，真正终止 | 末步标签 2 |
| s，$a_2$ | 0 | 日志未提供 | 没有动作后果样本 |

给定一个 critic 快照 $\hat Q(s,\cdot)=(0,2,4)$。两个已记录末步动作都拟合准确，第三个值却只是外推。若下一步以最大 Q 作自举，一条已经反复看过的 go 经验也会得到标签 $0+0.9\max(0,2,4)=3.6$。贪心策略随之在 s 选择 $a_2$，这个任务中的真实起点回报是 $0+0.9(-2)=-1.8$。终端拟合误差为零仍未约束那个最大值。神经网络的参数共享会让外推随其他动作的更新改变；这里用给定表格快照隔离信息缺口。

![两步任务的实际日志覆盖，未见动作问号，以及未知高值进入起点目标的共轴比较](https://yingwen.io/crl-figures/offline-support-coverage.svg)

上部沿蓝线读四条日志；灰色虚线是没有后果记录的 $a_2$，终端方框中的问号不泄漏奖励。下部共用 Q/标签数值轴：给定 Q 的四先进入旧 go 经验的 3.6 标签。先预测：若获得一条真正终止且奖励 −2 的新 $a_2$ 反馈，步长一的表格更新会把下次起点标签改成多少？蓝点给出答案 1.8。原创固定算例及反事实计算；附带代码没有补采环境或训练网络。

若获准在线继续行动，在这个确定任务中取得一条 $(s,a_2,-2,\mathrm{terminal})$，表格步长一会把 $\hat Q(s,a_2)$ 改为 −2；下一次 max 标签便是 $0.9\max(0,2,-2)=1.8$。离线训练没有取得这条新反馈的权限。更多次回放只重用原来的零与二。接下来可以限制高估的利用，也可以把改进计算放到数据动作内；这两条路线分别进入 CQL 和 IQL。

<a id="lesson-notation"></a>

## 2 · 重要性采样如何变换轨迹分布

$$
\rho_t=\frac{\pi(A_t\mid H_t)}{\mu(A_t\mid H_t)},\qquad w_{0:t}=\prod_{k=0}^{t}\rho_k
$$

在环境与初始分布相同的前提下，轨迹概率比中的环境项抵消，只保留动作概率比。

$$
\widehat J_{\rm PDIS}=\frac1n\sum_{i=1}^n\sum_{t=0}^{T-1}\gamma^t w^{(i)}_{0:t}R^{(i)}_{t+1}
$$

奖励只需用产生它之前的动作前缀校正。该形式在正确概率、支持和可积条件下无偏，但长乘积可能造成巨大方差。

加权归一化、截断比率可以改善有限样本稳定性，却通常引入偏差。有效样本量可以提示权重集中，但不能证明未覆盖区域没有风险。若目标策略和评价器是在同一数据上选出的，还存在自适应选择偏差，不能直接援引固定策略无偏结论。

<a id="lesson-derive"></a>

## 3 · Doubly robust：模型预测加残差校正

下面的 s 表示足以预测的状态；在不完全可观测任务中，可改为完整可用历史 h 并保留时间索引。若先前的目标策略依赖历史，却在这里换成仅按当前观察评价的 Q，递推就不再回答同一策略问题。

$$
\hat V_t(s)=\sum_a\pi(a\mid s)\hat Q_t(s,a),\quad\hat V_T=0
$$

有限时域价值可以随时间变化。Q 与 V 必须按同一目标策略对应，不能任取互不一致的两个网络。

$$
D_T=0,\qquad D_t=\hat V_t(S_t)+\rho_t\left[R_{t+1}+\gamma D_{t+1}-\hat Q_t(S_t,A_t)\right]
$$

从轨迹末端反向递推；最终使用 $D_0$。模型提供低方差基准，概率比校正模型残差。

正确重要性比使残差校正的期望补回模型偏差。展开递推后，模型项在条件期望中相消，从而得到目标回报。模型准确时，残差的条件均值接近零，方差往往降低；但并非每组样本中都比 IS 好。独立训练或交叉拟合 nuisance 模型时，应按独立 episode 或完整轨迹划分训练与评价折；把同一轨迹的转移随机分折，仍会共享动作后果，不提供所需独立性。单条相关长流需要另行分析其依赖结构。

$$
\mathbb E_\mu[D_t\mid S_t=s]=\hat V_t(s)+\sum_a\pi(a\mid s)\bigl[Q_t^\pi(s,a)-\hat Q_t(s,a)\bigr]=V_t^\pi(s)
$$

由末端 $D_T=0$ 向前作条件期望归纳：若下一步估计的条件均值为真实 V，则奖励加折扣后续的条件均值为真实 Q；正确比率把 $\mu$ 的动作权重变成 $\pi$，最后用 $\hat V=\sum_a\pi(a\mid s)\hat Q(s,a)$ 消去模型项。模型须先于独立评价数据固定，概率支持也须足够。历史依赖策略时，对完整历史作同一归纳。

$$
\mathbb E_\mu[D]-V^\pi=\sum_a\bigl(\pi(a)-\mu(a)\hat\rho(a)\bigr)\bigl(\hat Q(a)-Q^\pi(a)\bigr)
$$

先看固定状态的一步任务；模型和比率均固定于评价样本之前。这是把 D=ΣπQ̂+ρ̂(R−Q̂) 展开的精确偏差恒等式。

右侧有两条归零途径：比率完全正确，或每个动作的条件期望模型完全正确。序列情形要求相应条件沿各个时间和历史成立，并使用与 Q 一致的 V。两个模型都从有限数据拟合时，正确的函数类加上一致估计提供渐近论证，不能直接变成有限样本精确无偏。本页代码采用已知行为概率与独立固定模型。

即使 Q 完全正确，随机奖励和随机转移的单样本残差仍可能不为零。DR 在确定奖励的一步例子中可以零方差，不代表一般 MDP 中也能消除环境噪声。它也不能替代对行为覆盖和评价数据独立性的检查。

支持缺口还影响能否评价。回到两步任务，固定候选策略 $\pi(s,\cdot)=(0.1,0.8,0.1)$。构造两个世界，只把未记录的 $a_2$ 奖励从 −2 换成四，其他规则完全相同；它们给行为者产生的日志分布相同，这四条具体日志也逐项相同，候选策略的真实起点回报却分别为 $0.9(0.8\times2+0.1\times(-2))=1.26$ 与 $0.9(0.8\times2+0.1\times4)=1.8$。任何只读这份日志的估计器都无法辨认是哪一个世界。

如果只对日志里出现的动作算比率，PDIS 会返回 $0.9\times0.8\times2=1.44$，漏掉 $a_2$ 的贡献。若模型把 $a_2$ 预测为四，DR 的基准项则会返回 1.8，已见动作残差又恰为零：缺失后果没有残差可校正。正式评估应在读样本前检查完整目标支持并拒绝这个候选；用外部正确模型回答未知后果是新增假设。下文的数据内提取给第三动作零概率时，才可继续用这个任务核对 PDIS/DR 的算式。

<a id="lesson-cql"></a>

## 4 · CQL：限制没有数据支持的高值

$$
L_Q=L_{\rm Bellman}+\alpha\left(\mathbb E_{s\sim\mathcal D}\log\sum_a e^{Q(s,a)}-\mathbb E_{(s,a)\sim\mathcal D}Q(s,a)\right)
$$

这是离散动作 CQL 型核心正则，两项采用同一数据状态边缘分布。Bellman 回归标签停止梯度，正则中的当前 Q 则参与求导；完整算法还需指定目标策略、目标网络与权重。

$$
\begin{aligned}\ell_{\rm reg}(s)&=\log\sum_a e^{Q(s,a)}-\sum_a p_{\mathcal D}(a\mid s)Q(s,a),\\\frac{\partial\ell_{\rm reg}(s)}{\partial Q(s,a)}&=\operatorname{softmax}(Q(s,\cdot))_a-p_{\mathcal D}(a\mid s).\end{aligned}
$$

这是固定状态、尚未加权的正则项。对全局目标中的正则部分求表格坐标 $Q(s,a)$ 的导数，还要乘经验状态频率 $d_{\mathcal D}(s)$ 与 $\alpha$；共享网络参数则汇总各状态的链式梯度。正则压低高值但缺乏支持的动作，Bellman 项仍负责将价值连接到奖励。

相对保守不等于每个状态动作都是真实价值的逐点下界。原论文的界有具体分布、采样和权重条件。连续动作的 log-integral 通常用采样近似；不能直接枚举所有动作。本核只检验有限动作正则及其梯度。

在同一两步任务中，按八条 transition 均匀平均，经验状态频率 $d_{\mathcal D}(s)=4/8=0.5$；s 内动作频率为 $p=(0.5,0.5,0)$。为固定步长的含义，取 $L_{\rm Bellman}=\tfrac12\mathbb E_{\mathcal D}(Q-y)^2$，$\alpha=1$，本次所有 y 停止梯度。两个末步标签是零和二，所以给定快照处的 Bellman 残差为零。第三动作没有 Bellman 回归项，却仍进入 log-sum-exp 正则。

$$
\begin{aligned}\operatorname{softmax}(0,2,4)&\simeq(0.015876,0.117310,0.866813),\\g_a=\frac{\partial L_Q}{\partial Q(s,a)}&=0.5\bigl[p_a(Q(s,a)-y_a)+\operatorname{softmax}(Q)_a-p_a\bigr]\\&\simeq(-0.242062,-0.191345,0.433407),\\Q^+(s,\cdot)=Q(s,\cdot)-g&\simeq(0.242062,2.191345,3.566593).\end{aligned}
$$

步长为一的一次解析梯度下降，同时更新三个表格坐标。$p_2=0$ 时第一项直接为零，不使用未见动作的真实奖励作回归标签；只有正则为它提供下降梯度。其余状态项不依赖这些表格坐标。

这个更新压低了四，但尚未把它降到两以下；下次 max 自举标签仍约 3.2099。两个已见动作则被向上拉，其中 $Q^+(s,a_0)>r(s,a_0)=0$。这具体显示正则怎样改变相对值，也显示一步下降不等于准确评价，更没有给每个动作提供逐点下界。原文 §3.1 的策略期望界与 §3.2 的完整策略更新另有相应条件；不能从这一次计算套用。

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

<a id="lesson-iql"></a>

## 5 · IQL：在数据动作内近似策略改进

$$
\begin{gathered}L_V=\mathbb E_{\mathcal D}\left[|\tau-\mathbf1[u<0]|u^2\right]\\u=\operatorname{sg}(Q_{\bar\theta}(s,a))-V_\psi(s)\\\tfrac12<\tau<1\end{gathered}
$$

对偏高 Q 的残差赋更大权重，拟合数据动作价值的上 expectile。这里是非对称平方损失，不是上一章的分位数 pinball loss；τ=1 不能当作仍具有相同唯一解的普通端点。

$$
\mathbb E\!\left[|\tau-\mathbf1[Q-v<0]|(Q-v)\right]=0
$$

对 v 求导并令其为零得到 expectile 的平衡方程。它随残差大小变化，不只看有多少样本在两侧。

$$
L_Q=\mathbb E_{\mathcal D}\left[\left(Q_\theta(s,a)-\operatorname{sg}\!\left(r+\gamma(1-d)V_\psi(s')\right)\right)^2\right]
$$

Q 的目标使用后继 V，不需要在未见动作上查询当前策略的 Q。V 与 Q 的更新顺序和目标副本应在实现中明确。

$$
L_\pi=-\mathbb E_{\mathcal D}\left[\operatorname{sg}\!\left(e^{\beta(Q_{\bar\theta}(s,a)-V_\psi(s))}\right)\log\pi_\phi(a\mid s)\right]
$$

actor 做优势加权行为克隆。本章 β 是逆温度；工程里常对权重裁剪以稳定训练，裁剪会改变精确加权目标。

IQL 避免训练目标中显式最大化未见动作，但函数逼近得到的策略仍可能泛化到数据外。不能把它理解成无条件的支持保证，也不能认为上 expectile 等于任意高质量未知动作的真实价值。

仍取 $\hat Q(s,\cdot)=(0,2,4)$。V 损失只从已记录的 $a_0,a_1$ 各取一半权重，$a_2$ 的四不进入拟合。对 $0\le v\le2$，$L_V(v)=\tfrac12(0.25v^2+0.75(2-v)^2)$，一阶条件给出 $v=1.5$。因此 go 的 IQL 标签是 $0+0.9\times1.5=1.35$。计算这条标签只读后继 V；没有采样当前 actor 的动作再查询那个动作的 Q。三个末步都真正终止，已见末步 Q 的标签始终保留奖励零或二，尾值为零。

$$
\pi^*(a\mid s)=\frac{p_{\mathcal D}(a\mid s)\exp\!\left[\beta(\hat Q(s,a)-V(s))\right]}{\sum_b p_{\mathcal D}(b\mid s)\exp\!\left[\beta(\hat Q(s,b)-V(s))\right]}
$$

对固定权重的分类 likelihood，在完整概率单纯形上求精确最优：以归一化约束的乘子对各概率求导，得到概率与“数据频率 × 权重”成正比。零数据频率允许零概率；有限 softmax logits 通常只能逼近这个边界解。

取 $\beta=1$，两个数据动作的权重为 $e^{-1.5}$ 和 $e^{0.5}$，提取结果约为 $(0.119203,0.880797,0)$。共同的 $e^{-V(s)}$ 在归一化中消去，所以对这个固定 Q 的精确分类解，改变 τ 不直接改变动作比率；τ 仍通过 V 改变多步 Q 标签。数据内 expectile 的极限 $\tau\uparrow1$ 趋向二，未知动作的四不参与；$\beta=0$ 恢复数据频率，$\beta\to\infty$ 则在有数据的高 Q 动作上集中。$\tau=1$ 本身会使所有 $v\ge2$ 的损失为零，不再是同一个唯一解。

![CQL 同轴 Q 更新与有符号梯度，IQL 数据内 expectile 曲线及加权动作概率](https://yingwen.io/crl-figures/offline-support-objectives.svg)

上部圆点是给定 Q，方点是 $Q-g$；横条长度为完整数据均值的有符号梯度，下降更新方向与梯度符号相反。下部 V 损失仅用零和二，τ=0.75 的最小值在 1.5；概率条由数据频率乘以优势权重得到。先预测：把未见动作 Q 从四改成一千，哪条曲线或提取概率会改变？本例 IQL 的两者均保持不变，CQL 正则则会改变。原创解析计算；图中的概率是精确分类解，不是神经策略训练结果。

![固定数据的缺失动作与 IQL 的 Q、V、actor 更新路径](https://yingwen.io/crl-figures/concept-classic-offline-support.svg)

上半图是只记录动作零的表格例：未见动作的高估值没有后果证据，图中 actor 概率表示完全拟合数据后的解。下半图是另一份同频两动作数据，Q=(0,2)、τ=0.75、β=1；V=1.5，优势加权克隆概率为 (0.1192,0.8808)。沿箭头读三条更新，sg 表示目标或权重停止梯度；Q 的标签使用后继 V，actor 只拟合数据动作。原创结构图与精确计算，不是训练曲线，也不将表格支持性质推广为神经策略保证。

<a id="lesson-algorithm"></a>

## 6 · 学习与评估分成两条过程

**算法：离线模拟器评估若可用，是额外权限；它不是所有真实离线任务都具备的条件。**

1. 评估：固定待评策略及独立数据，检查行为概率与支持。
  1. 从末端计算 DR，或前向累乘 PDIS 权重。
  1. 汇总独立轨迹，报告区间、权重集中和数据限制。
1. IQL 学习：在 dataset batch 上先拟合 expectile V。
  1. 用固定后继 V 拟合 Q，再更新 Q 的 target 副本。
  1. 用 detached advantage 权重更新数据动作的 log likelihood。
1. CQL 学习：在 Bellman 回归上加入声明的保守正则。
1. 策略选择与最终评价使用分离的数据或明确的验证协议。

<a id="lesson-example"></a>

## 7 · 同一小数据解释三个公式

先把两步主例的“标签”与“策略价值”分开。IQL 的起点 Q 标签是 1.35；提取策略的真实起点回报则为 $0.9(0.119203\times0+0.880797\times2)\simeq1.585435$。V 的 expectile 与 actor 的指数权重定义不同，有限 τ、β 时并不要求这两个数相等。

对固定提取策略，在这四条构造日志上，go 的比率均为一，末步比率约为 $(0.238406,1.761594)$。两条零奖励轨迹的 PDIS 项为零，两条奖励二的项各为 $0.9\times1.761594\times2\simeq3.170869$；平均约 1.585435。若使用已见动作准确的 Q 和与该策略一致的 V，DR 每条轨迹都返回 1.585435；零 Q 模型则逐条退化为 PDIS。该等式核对用的是给定平衡日志。实际根据训练集选出策略后，最终评估仍要使用独立完整轨迹或相应交叉拟合协议。

再把评估过程压缩成一步 bandit，以单独检查“模型残差为零时 DR 返回什么”。这个退化检查使用下面另一组奖励和策略概率；没有改变前面两步任务的数值。

bandit 两动作行为概率均为 $0.5$，目标概率 $(0.8,0.2)$，确定奖励 $(1,3)$，目标收益为 $1.4$。IS 在动作零样本上给 $1.6$，在动作一样本上给 $1.2$，平均为 $1.4$。若模型完全准确，DR 的残差为零，每个样本都返回 $1.4$；模型全零时 DR 退化为 IS。

对同频 Q 数据 $(0,2)$，$\tau=0.75$，expectile 方程为 $0.75(2-v)=0.25v$，解为 $v=1.5$。取 $\beta=1$，分类 actor 的加权克隆概率与 $(e^{-1.5},e^{0.5})$ 成正比，因此高值动作概率约 $0.8808$。这不是直接执行确定 argmax。

若数据只含动作零，即使网络把动作一的 Q 写成十，本页精确表格加权克隆仍给动作一零概率；神经泛化时不能直接援引这条表格性质。

<a id="lesson-code"></a>

## 8 · 估计器与目标函数的可执行核

贯穿主例的独立入口是 [offline_support_walkthrough.py](/crl-code/tutorials/offline_support_walkthrough.py)。只依赖 Python 标准库；四条日志、Q 快照、行为概率及两种候选世界均写在文件里。直接运行输出 JSON；--test 核对 CQL 完整损失有限差分、τ/β 极限、终止字段、两世界同日志和 OPE 支持拒绝。图的全部数量另由独立 [JavaScript 计算](/crl-code/figures/offline-support-walkthrough.mjs) 产生，供逐项比较。

第一、二行在保存教程文件的目录运行；第三个命令在仓库根目录运行。计算给定目标与解析位移，不采样或训练。

```bash
python3 offline_support_walkthrough.py --test
python3 offline_support_walkthrough.py
# 在网站仓库根目录核对公开图中的数据：
python3 public/crl-code/tutorials/offline_support_walkthrough.py \
  --compare public/crl-figures/offline-support-data.json
```

PDIS、顺序 DR、expectile 求根、表格加权克隆与 CQL 梯度。

```python
def importance_ratio(target, behavior):
    if not 0 <= target <= 1 or not 0 < behavior <= 1:
        raise ValueError('positive recorded behavior probability required')
    return target/behavior


def per_decision_is(rewards, ratios, gamma=1.):
    if len(rewards) != len(ratios):
        raise ValueError('one ratio per reward required')
    product, estimate = 1., 0.
    for t, (reward, ratio) in enumerate(zip(rewards, ratios)):
        product *= ratio
        estimate += gamma**t*product*reward
    return estimate


def sequential_dr(rewards, ratios, q_hat, v_hat, gamma=1.):
    """v_hat contains estimates at each state and a terminal zero."""
    n = len(rewards)
    if len(ratios) != n or len(q_hat) != n or len(v_hat) != n+1 or v_hat[-1] != 0:
        raise ValueError('complete finite trajectory and zero terminal value required')
    estimate = 0.
    for t in reversed(range(n)):
        estimate = v_hat[t]+ratios[t]*(rewards[t]+gamma*estimate-q_hat[t])
    return estimate


def expectile(values, tau, weights=None):
    if not values or not 0 < tau < 1:
        raise ValueError('nonempty values and interior expectile required')
    weights = [1.]*len(values) if weights is None else weights
    if len(weights) != len(values) or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError('nonnegative observation weights required')
    low, high = min(values), max(values)
    for _ in range(100):
        middle = (low+high)/2
        derivative_sign = sum(w*(tau if q >= middle else 1-tau)*(q-middle)
                              for q, w in zip(values, weights))
        if derivative_sign > 0:
            low = middle
        else:
            high = middle
    return (low+high)/2


def iql_weighted_actor(q_values, counts, tau=.75, inverse_temperature=1.):
    """Exact weighted behavioral-cloning solution for a one-state categorical actor."""
    value = expectile(q_values, tau, counts)
    logits = [inverse_temperature*(q-value) for q in q_values]
    offset = max(z for n, z in zip(counts, logits) if n > 0)
    weights = [n*math.exp(z-offset) if n > 0 else 0. for n, z in zip(counts, logits)]
    return value, [w/sum(weights) for w in weights]


def cql_penalty_gradient(q_values, data_probabilities):
    probabilities(data_probabilities)
    if len(q_values) != len(data_probabilities):
        raise ValueError('matching action dimensions required')
    maximum = max(q_values)
    exponentials = [math.exp(q-maximum) for q in q_values]
    normalizer = sum(exponentials)
    loss = maximum+math.log(normalizer)-dot(q_values, data_probabilities)
    return loss, [x/normalizer-p for x, p in zip(exponentials, data_probabilities)]
```

运行 test 检查 DR 与零模型 IS 的一致、准确模型的基准值、零行为概率拒绝、expectile 与 quantile 的差别以及 CQL 有限差分。源码不含完整 CQL/IQL 神经训练，也不对任意数据产生置信保证。IQL 作者仓库是 JAX 实现；其中 actor.py、critic.py 与 value_net.py 分别对应上述三类更新。

<a id="lesson-branches"></a>

## 9 · 数据限制与 CRL 的历史经验

- 隐藏混杂：行为者使用了数据中未记录的信息时，仅从观察字段估计行为概率可能不足。
- 未知终止：人工日志截断与真正终止混用会改变 Q 和 DR 的尾部。
- 持续变化：不同年份或阶段的数据可能来自不同动力学，普通 stationary OPE 的环境抵消不再成立。

CRL 中历史 replay 可以提供旧技能证据，但其时间戳、策略概率、传感器版本与任务条件同样重要。可分别比较覆盖不足和环境变化两种干预，避免把离线外推、知识遗忘与世界非平稳合并成一个指标。

<a id="lesson-check"></a>

## 10 · 练习与答案

- 预测：将未见 $a_2$ 的 Q 从四改成一千，IQL 的 V 与精确分类提取会怎样？答：保持 1.5 与 (0.119203,0.880797,0)，因为两个目标的求和只覆盖数据动作；CQL 的 softmax 正则会改变。
- 手算：CQL 中若只平均 s 的末步样本而不平均全部八条 transition，正则梯度会变多少？答：状态频率从 0.5 改为一，三个坐标的梯度都加倍。步长也必须按新的目标约定解释。
- 判断：把日志复制一千遍能分辨 $a_2$ 奖励为 −2 还是四吗？答：不能；两世界在行为支持上的分布相同。扩大同一支持内的样本量与新增动作覆盖是不同干预。

- 问：已知目标动作的概率，是否就能做 IS？答：还需对应行为概率、支持和相同环境等条件。
- 问：DR 中用任意 V 与 Q 都能保持同样理论性质吗？答：不能；需要其目标策略关系及所用估计条件。
- 问：expectile 0.75 等于样本 75% 分位数吗？答：不等于，例子中分别为 1.5 和二。
- 实验：固定模型为零，给两步奖励 (1,2)、比率 (2,0.5)、折扣 0.9。PDIS 和 DR 均应得到 3.8；这验证代数，不证明数据覆盖。



<a id="chapter-code"></a>

## 下载与运行

标准库解析机制实验；不包含完整神经网络训练或大型 benchmark。

[下载 extended_foundations_lab.py](https://yingwen.io/zh/continual-rl/download/extended_foundations_lab.py)

```sh
python3 extended_foundations_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Jiang、Li · Doubly Robust Off-policy Value Evaluation](https://proceedings.mlr.press/v48/jiang16.html)：顺序 doubly robust 估计器、条件和方差分析。

- [Kumar et al. · Conservative Q-Learning](https://arxiv.org/abs/2006.04779)：保守价值目标与理论结论的适用条件。

- [Kumar · CQL 原作者代码](https://github.com/aviralkumar2907/CQL)：离散与连续动作实验工程；需对照具体配置。

- [Kostrikov、Nair、Levine · Implicit Q-Learning](https://arxiv.org/abs/2110.06169)：expectile 价值、隐式改进与加权行为克隆。

- [Kostrikov · IQL 原作者代码](https://github.com/ikostrikov/implicit_q_learning)：JAX 实现，含离线训练和在线微调入口。

- [UC Berkeley · CS 285](https://rail.eecs.berkeley.edu/deeprlcourse/)：原课程的 exploration、model-based RL 与 offline RL 讲义和视频入口；按问题专题阅读，不必按网络规模划分领域。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-offline#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-offline#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-deep-offline)
<a id="study-connections"></a>

## 与教材主线的衔接

本章是可按问题选择的并列研究分支，不要求先读完其他深度研究分支。

### 估计哪种策略的价值，又按什么状态分布加权？

行为策略决定怎样获得经验；目标策略决定要预测哪种行为。重要性比可以校正给定状态下的动作分布，但不会自动把状态出现的频率改成目标策略的频率。

函数逼近与深度方法：Replay 还引入缓冲区的时间组成与抽样规则。神经更新受到数据分布、共享梯度和移动目标共同影响。重复旧数据与逐条使用新数据有不同的资源和适应代价。

持续学习中的研究问题：单一行为流怎样支持许多预测和技能？在固定内存下，怎样权衡覆盖、样本年龄、更新方差与适应速度，而不把离策略修正当作完整稳定性保证？

[离策略稳定性](https://yingwen.io/zh/continual-rl/foundations/approximation/off-policy/) → [数据与训练接口](https://yingwen.io/zh/continual-rl/foundations/deep/practice/) → [大规模系统与策略滞后](https://yingwen.io/zh/continual-rl/foundations/deep/systems/) → [离线数据的覆盖](https://yingwen.io/zh/continual-rl/foundations/deep/offline/) → [流式更新](https://yingwen.io/zh/continual-rl/algorithms/streaming/)


### 可进一步检验的问题

- [12 · 怎样保留旧能力，而不把过时知识强加给新任务？](https://yingwen.io/zh/continual-rl/research/#research-retention-transfer)：保留旧经验并不补齐新行为的数据支持，因而复用旧数据时还要检查价值外推和独立策略评价。
- [16 · 什么实验能区分“仍在更新”与“仍在有效学习”？](https://yingwen.io/zh/continual-rl/research/#research-measurement)：策略学习、超参数选择和独立离线评价使用不同数据角色，避免把用于选择的方法再次当作独立验证。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [深度价值学习](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)
- [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

对应原始材料：Jiang、Li：Doubly Robust OPE；Kumar et al.：CQL；Kostrikov、Nair、Levine：IQL。本文为原创讲解，原书、论文与上游代码保留各自许可。
