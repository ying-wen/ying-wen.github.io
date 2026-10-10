# 价值预测与时间差分学习

持续学习仍要预测后果。沿用 MC、TD 与资格迹时，需要固定哪些对象，才能分清估计在更新、预测问题在变化，以及学习器未来会改变行为这三件事？

## 本章内容

- 沿回报、Bellman 方程与采样目标，区分固定预测问题和不断变化的估计。
- 用保留的两步轨迹核对 MC、TD 与资格迹的更新时序。
- 将固定策略预测接到追踪、GVF 与完整学习器评价，明确线性 TD($\lambda$) 等价条件。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [TD 预测与控制：SARSA、Expected SARSA、Q-learning 和 Double Q](https://yingwen.io/zh/continual-rl/foundations/tabular/temporal-difference/)：完整的一步预测推导在第一册。
- [函数逼近预测：从回归到 TD 固定点](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/)：共享参数下的误差与半梯度在第一册。


### 一条经验

在 $S_{t}$ 做 $A_{t}$ 后得到 $R_{t+1}$、$S_{t+1}$。奖励属于这次转移，不是到达状态之前另一轮的奖励。

### 折扣与终止

$\gamma$ 控制未来相对权重；真实终止后的价值设为 0。本章记 $\gamma_{t+1}$=$\gamma$(1−terminated)。训练脚本的时间截断不一定是任务终止。

### 表格与函数逼近

表格每个状态或状态动作一组独立参数；函数逼近让不同输入共享参数。更新一个输入可能改变其他输入的预测。

<a id="problem-definition"></a>

## 本章的问题定义

环境与目标策略固定，询问按该策略行动的期望回报；本章基础数据也由该策略产生。

### 给定条件与符号

- Markov状态、固定目标策略、奖励与延续/终止定义。
- 经验流、价值表示类及更新预算；已知模型是DP的额外权限。

### 需要求解的对象

指定策略的价值函数及在给定表示下的估计；策略本身不是本章待学对象。

### 信息与数据权限

在 $S_t$ 按 $\pi$ 选择 $A_t$，得到 $R_{t+1},S_{t+1}$；估计 $V_t$ 只能使用已经收到的经验。

$$
v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s],\qquad G_t=R_{t+1}+\gamma_{t+1}G_{t+1}
$$

$\pi$ 是给定策略，$G_t$ 是随机回报，$\gamma_{t+1}$ 是当前转移后的延续因子；真实终止为0。$V_t$ 是估计而不是答案本身；TD平方误差与真实价值误差不相等。

### 成立条件与解的含义

- 基础设定为有限、固定MDP、有界奖励、非终止处固定折扣小于1；无折扣终止问题需另给可积终止条件。
- 表格收敛还需访问与步长条件；共享非线性表示、离策略及不断漂移不自动继承该结论。

判断准则：两步链上预测趋近起点0.9、后继1；一般任务以独立回报或解析解测价值误差，训练样本TD误差无需逐条为零。

### 适用边界

- 不以对动作取最大值替代给定策略的动作平均。
- 不以训练TD损失下降宣称策略收益改善。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)：本章将累计信号限定为任务奖励，并采用普通折扣/终止规则，因此是更一般GVF预测规格的特例。

- 组合不同学习问题 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：预测可以评价一个候选策略；控制另需策略改善和行为数据更新。

- 组合不同学习问题 · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)：多步和资格迹决定同一预测问题的反馈如何作用于过去，不改变给定策略的题目。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

完整回报尚未到来；一步自举可立即更新，却依赖当前后续价值估计。

### 本章的核心思路

从同一回报递推式选择模型期望、完整样本或一步自举；比较的是估计方法而非三个任务。

1. [从回报推到条件期望](#lesson-derive)：因为预测对象是指定行为的未来，Bellman方程按目标策略平均动作而不取最大值。

2. [选择等待长度与信用路径](#value-traces)：因为短目标依赖估计、长目标等待更多数据，多步混合与资格迹在同一对象下改变传播。

3. [以解析值检查完整循环](#lesson-code)：因为一次正确梯度不保证整个时序正确，真实终止尾值、旧参数和访问更新一并用两步链验证。

结论与条件：正确有限折扣模型的Bellman算子收缩；表格MC/TD还需各自采样与步长条件。普通在线迹不与冻结前向视图无条件精确等价。

### 相关方法改变了什么

- DP：已知后果模型时计算条件期望，需要模型权限。

- MC：用完整回报采样，减少自举依赖但等待结果。

- TD与多步：用后继估计补足未来，改变目标偏差、方差与反馈延迟。


<a id="lesson-setting"></a>

## 1 · 先固定预测问题，再问什么发生了变化

第一册的 [TD 预测与控制](/zh/continual-rl/foundations/tabular/temporal-difference/) 给出采样更新的基础，[预测函数逼近](/zh/continual-rl/foundations/approximation/prediction/) 再说明共享参数、评价分布与投影固定点。本章保留它们的最小关系，供阅读持续预测与信用分配时核对：收到新经验后，究竟只是估计变了，还是要预测的后果也变了？

先固定策略 $\pi(a\mid s)$、有限状态动作空间、转移规律与有界奖励，取 $0\le\gamma<1$，数据由目标策略产生。在这些条件下，真实价值 $v_\pi$ 固定，估计 $V_t$ 仍可每步更新。这里“策略固定”没有要求预测参数固定；后文推导前后向等价时，才会另外冻结预测参数。若观测不足以条件化未来，还须使用历史或合适的状态表示。

$$
G_t=R_{t+1}+\gamma_{t+1}G_{t+1},\qquad v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s]
$$

$G_t$ 是一次实现的随机回报；$v_\pi(s)$ 是同一起点、同一策略下回报的条件期望。真实终止令后继延续因子为零，只删掉未来项，不删掉进入终点时的奖励。

预测以策略和回报规格为输入，以条件期望为答案。控制还要决定怎样据此行动。算法的 TD error 比较两种当前估计，价值误差则比较估计与真实答案；解析模型或独立回报样本可帮助检查后者。若问题已经改变，评估还应说明这些样本对应哪一个世界、策略和时间窗口。

<a id="lesson-derive"></a>

## 2 · 从回报拆分到 Bellman，再到三种更新

$$
v_\pi(s)=\sum_a\pi(a\mid s)\sum_{s\prime,r}p(s\prime,r\mid s,a)[r+\gamma(s,a,s\prime)v_\pi(s\prime)]
$$

第一步把 G 拆成当前奖励和未来；第二步对动作与转移取条件期望；Markov 性允许用下一状态而非完整历史描述后半段。这里尚未提出任何学习算法。

$$
V_{k+1}(s)=(T_\pi V_k)(s),\qquad \|T_\pi u-T_\pi v\|_\infty\le\gamma\|u-v\|_\infty
$$

已知模型时，可以对所有后果求和。这是动态规划的期望备份；$\gamma$<1 带来压缩性。在正确表格模型下重复备份趋向唯一固定点。$\gamma$=1 的终止任务需要另一套适当终止条件。

$$
V(S_t)\leftarrow V(S_t)+\alpha[G_t-V(S_t)]
$$

MC 用完整实际结果作为监督目标。需等到回报可计算；对固定策略、可积回报，它直接对正确条件期望采样。环境中途不断变化、轨迹没有结束或非常长时，等待完整回报代价很大。

$$
\delta_t=R_{t+1}+\gamma_{t+1}V_t(S_{t+1})-V_t(S_t),\qquad V_{t+1}(S_t)=V_t(S_t)+\alpha_t\delta_t
$$

TD(0) 只等一步，把剩下的未来交给当前估计。单个 target 通常不是对真实 v 的无偏样本，因为 V 还不准；但正确条件下长期固定点可以正确。因此，需要分析其长期固定点，而不只分析单次目标的偏差。

| 方法 | 需要什么 | 一条转移能否立即更新 | 误差从哪里来 |
| --- | --- | --- | --- |
| DP | 转移与奖励模型 | 可以，但使用模型期望 | 模型误差、备份不足 |
| MC | 完整采样回报 | 通常不能 | 有限样本方差 |
| TD | 一步经验与旧价值 | 可以 | 自举估计、有限样本、表示限制 |

<a id="experiment-td0"></a>

### 实验：实验 · MC 与 TD 估计的是同一个解析价值

等待完整回报和逐步自举，在相同轨迹预算下有什么差别？

**环境与可用信息。** 五个非终止状态 1–5 的无偏随机游走。观测就是状态，左右各以 0.5 概率发生，策略不学习。到 0 终止且奖励 0；到 6 终止且奖励 1；其余奖励 0。每回合从 3 开始，折扣为 1，解析价值为状态编号除以 6。

**设置。** 1200 个真实转移；五个价值估计全零。TD 每步以 0.1 更新；MC 在真正终止后对每回合首次访问状态用累计访问计数求样本均值。未完成的预算尾回合不伪造 MC 回报。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** td0.py 使用即时奖励与旧后继值；mc_prediction.py 反向求完整回报，再按首次访问更新。两个实现具有相同的固定随机游走轨迹，但更新时刻和步长制度不同。

**测量。** 纵轴是五个非终止状态对解析价值的等权 RMSE，不是训练 TD 误差。横轴包括所有真实转移，也包括 MC 尚未等到终点的回合。

```bash
python3 implementations/classic/td0.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-figures/result-td0.svg)

横轴：environment_steps。纵轴：五状态价值 RMSE。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 当前价值估计与真实值 s/6 的误差。在五个非终止状态 s=1,…,5 上均匀平均平方误差后开方；权重不是训练访问频率。

**step：怎样计时。** step 包含全部真实转移，也包含尚未完成的回合。MC 在自然终止后，对该回合首次访问的每个状态写一次回报均值；同回合重复访问不增加该状态的样本数。TD(0) 每个真实步写入；资格迹可同时影响多个状态。相同环境步不表示相同参数写入次数，预算结束也不产生额外终止反馈。

**怎样汇总。** 取该记录时刻的值；不先对曲线上的时间点求平均。 各状态初值均为0。MC 用各状态已完成回合的首次访问计数决定步长；TD(0) 用常数步长0.1，比较不仅改变了是否自举。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** CSV 可重算各记录时刻的跨种子均值、样本标准差和末点误差；没有保存全部预测向量，不能仅凭 value 重新计算状态权重或逐状态误差。 日志未保存回合边界、首次访问计数或逐步价值，曲线平台不能单独区分等待终止、零更新和稀疏记录漏掉变化。

计算位置：[classic/_common.py](https://yingwen.io/crl-code/implementations/classic/_common.py) · [classic/td0.py](https://yingwen.io/crl-code/implementations/classic/td0.py) · [classic/td_lambda.py](https://yingwen.io/crl-code/implementations/classic/td_lambda.py) · [classic/true_online_td.py](https://yingwen.io/crl-code/implementations/classic/true_online_td.py) · [classic/mc_prediction.py](https://yingwen.io/crl-code/implementations/classic/mc_prediction.py)

</details>

**结果分析。** 第 615 步，TD 与 MC 的平均 RMSE 分别为 0.10689、0.03574；第 1200 步为 0.04650、0.02674。本配置中 MC 误差更小，不能先验把逐步更新等同于有限预算更准确。

**结论边界。** 这不是等步长、等更新次数的单因素比较。MC 采用样本均值，TD 为固定率；环境平稳、回合短且真值可解析。结果不能推广为 MC 普遍优于 TD。

**继续实验。** 先手算第一次回合中两个方法的更新时间。再固定步长制度或改变回合长度，区分等待代价、bootstrap 误差与估计方差。

[源码](https://yingwen.io/crl-code/implementations/classic/td0.py) · [逐种子记录](https://yingwen.io/crl-code/results/td0/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/td0/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/td0/curves.json)

<a id="value-traces"></a>

## 3 · 多步与 $\lambda$：奖励应该传回多远？

$$
G_t^{(n)}=\sum_{k=1}^n\gamma^{k-1}R_{t+k}+\gamma^n V(S_{t+n})
$$

终止时截到终点。n 小更依赖价值估计，n 大使用更多实际结果，也等待更久、通常有更大方差。具体的误差权衡取决于奖励噪声、价值估计和轨迹长度。

![同一条轨迹上的前向多步目标与向过去传播的 TD 误差。](https://yingwen.io/crl-figures/concept-credit-forward.svg)

先固定整段轨迹上的价值参数。上方从一个起点向未来看，下方沿同一经验把误差分配给过去；资格迹让后者能逐步计算。图中轨迹用于解释传播方向，下节两步链另给第一次更新的数值。

$$
G_t^\lambda=(1-\lambda)\sum_{n\ge1}\lambda^{n-1}G_t^{(n)},\quad 0\le\lambda<1,\qquad G_t^\lambda-V(S_t)=\sum_{k\ge0}(\gamma\lambda)^k\delta_{t+k}
$$

在无限折扣轨迹、奖励和冻结 V 有界时，展开 n-step，按相同奖励与 V 项收集，价值项逐项抵消，就得到右边的 TD error 加权和。不能直接在左侧无限几何式中代入 λ=1。

$$
G_t^\lambda=(1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1}G_t^{(n)}+\lambda^{T-t-1}G_t,\quad 0\le\lambda\le1.
$$

有限终止轨迹的最后一项吸收全部剩余权重；T 是真正终点。λ=1 得完整回报，λ=0 得一步目标（最后一步的空和为零）。无限折扣情形的 λ=1 则按 λ↑1 的极限定义。

$$
e_t=\gamma\lambda e_{t-1}+x_t,\qquad w_{t+1}=w_t+\alpha\delta_t e_t
$$

线性 $V=w^Tx$ 时，把未来误差回传改写为记录过去特征。e 是“哪些参数对近期预测有影响”的信用痕迹，不是能在动作选择时回忆往事的隐状态。表格取 one-hot 特征。

这一步前向/后向等价的推导固定了参数。普通在线 TD($\lambda$) 每步都改参数，有限步长下不能声称严格等价于该固定参数目标；true-online TD($\lambda$) 用 Dutch trace 和额外修正实现相应在线前向视图。离策略时还要处理目标策略与行为策略的差别，相应修正将在通用价值函数一章中推导。

<a id="experiment-nstep_td"></a>

### 实验：实验 · 三步目标的传播更远，不保证最终误差更低

把一步目标改成三步目标，收益是否贯穿整个训练过程？

**环境与可用信息。** 五个非终止状态 1–5 的无偏随机游走。观测就是状态，左右各以 0.5 概率发生，策略不学习。到 0 终止且奖励 0；到 6 终止且奖励 1；其余奖励 0。每回合从 3 开始，折扣为 1，解析价值为状态编号除以 6。

**设置。** 1200 个真实转移，价值全零；三步 TD 与 TD(0) 都用步长 0.1。三步方法等待目标成熟，真终点时处理剩余项；预算末尾不把未成熟目标当终止。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** nstep_td.py 保存最多三步的待更新轨迹，将实际奖励和三步后价值组合；比较对象只用下一步价值。两者的目标等待时间不同。

**测量。** 仍对同一解析价值测 RMSE。观察整条曲线，不把某一个早期交叉点或最后一个点解释成稳定优势。

```bash
python3 implementations/classic/nstep_td.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/nstep_td/curves.svg)

横轴：environment_steps。纵轴：五状态价值 RMSE。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 当前三步 TD 价值估计与真实值 s/6 的误差。在五个非终止状态 s=1,…,5 上均匀平均平方误差后开方；权重不是训练访问频率。每回合从状态3开始，无偏左右游走，左端奖励0、右端奖励1，γ=1。

**step：怎样计时。** step 是实际随机游走转移数，包含尚未成熟的三步目标。队列积满三条后，每来一条真实转移，就用三步奖励和当前尾状态的价值更新最早的状态；自然终止时再更新余下的一步、两步尾项，不增加环境步。预算结束时至多两条未成熟转移保留为未更新。TD(0)对照每个真实步更新一次，相同 step 不保证相同写入时序。

**怎样汇总。** 取该记录时刻的值；不先对曲线上的时间点求平均。 两方法都从价值0开始、使用常数步长0.1；三步 TD 改变了等待长度和自举位置。每15步记录一次，第一点是第15步，不包含初值或全部中间更新。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** CSV 可重算各记录时刻的跨种子均值、样本标准差和末点误差；没有保存全部预测向量，不能仅凭 value 重新计算状态权重或逐状态误差。 CSV 仅有 step、value、phase，没有回合边界、pending 队列或写入计数，不能由误差平台恢复实际更新次数或判断哪些尾项仍在等待。该固定预测任务没有控制行动或变化后的适应阶段。

计算位置：[classic/nstep_td.py](https://yingwen.io/crl-code/implementations/classic/nstep_td.py) · [classic/td0.py](https://yingwen.io/crl-code/implementations/classic/td0.py) · [classic/_common.py](https://yingwen.io/crl-code/implementations/classic/_common.py)

</details>

**结果分析。** 第 615 步，三步 TD 的均值 0.08700 低于一步的 0.10689；第 1200 步三步为 0.07134，反高于一步的 0.04650。传播长度改变了有限样本误差，而不是单调提高准确性。

**结论边界。** 5 个 seed、单一步长和一个短链不足以建立最优 n。三步方法在预算末有未成熟目标，实际完成的更新次数也可能不同。

**继续实验。** 保持同一经验序列，分别检查 n=1、3 和完整回报的边界处理。预测改变步长后哪个比较可能反转，并用独立种子验证。

[源码](https://yingwen.io/crl-code/implementations/classic/nstep_td.py) · [逐种子记录](https://yingwen.io/crl-code/results/nstep_td/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/nstep_td/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/nstep_td/curves.json)

<a id="lesson-example"></a>

## 4 · 同一条两步轨迹：奖励何时影响起点？

A --奖励 0→ B --奖励 1→终点。$\gamma$=.9，初始 V(A)=V(B)=0，$\alpha$=.1。真实值为 $v_\pi(A)=0.9$、$v_\pi(B)=1$。

| 第一个回合结束后 | A 的 target / 更新后值 | B 的 target / 更新后值 |
| --- | --- | --- |
| MC | 完整回报 .9 → .09 | 完整回报 1 → .1 |
| TD(0) | 0+.9×0=0 → 0 | 1+0−0 → .1 |
| TD($\lambda$=.8) | 第二步迹 .9×.8=.72 → .072 | 第二步当前特征迹 1 → .1 |

TD 在第一步就能更新 A，此时目标恰为零；MC 等到终点后才用完整回报更新 A。第二回合 TD 在 A 的 target 已是 .9×.1=.09，因此 V(A)=.009。表中比较的是第一个回合结束后的传播范围；能更早执行更新，与奖励能在几个回合内传回起点，是两种不同的时间问题。

<a id="lesson-code"></a>

## 5 · 核心实现与一次完整训练循环

**算法：线性、on-policy、回合制的 accumulating TD(λ)**

1. 初始化 $w$；给定步长 $\alpha$、迹参数 $\lambda$ 和特征函数 $x(s)$。
1. 每个回合开始时令 $e=0$，观察起始状态 $S$。
1. 每次转移：
  1. 按目标策略执行动作，观察 $R,S'$。
  1. 若真实终止，令 $\gamma'=0$；否则令 $\gamma'=\gamma$。
  1. $\delta\leftarrow R+\gamma'w^\top x(S')-w^\top x(S)$
  1. $e\leftarrow\gamma\lambda e+x(S)$
  1. $w\leftarrow w+\alpha\delta e$
  1. $S\leftarrow S'$；真实终止时结束本回合。

相同环境与步长，分别用整段结果和一步自举

```python
def prediction(method="td", episodes=200, alpha=0.1):
    """A --0--> B --1--> terminal; gamma=.9, exact values [.9, 1]."""
    values = [0.0, 0.0, 0.0]
    trajectory = [(0, 0.0, 1, 0.9), (1, 1.0, 2, 0.0)]
    for _ in range(episodes):
        if method == "mc":
            ret = 0.0
            for state, reward, _, discount in reversed(trajectory):
                ret = reward + discount * ret
                values[state] += alpha * (ret - values[state])
        elif method == "td":
            for state, reward, nxt, discount in trajectory:
                delta = reward + discount * values[nxt] - values[state]
                values[state] += alpha * delta
        else:
            raise ValueError("Choose mc or td")
    return values[:2]
```

- MC 从终点向前计算 return；本例每个状态每回合仅访问一次，因此 first-visit 与 every-visit 没有差别。一般轨迹中须明确选择。
- TD 在收到下一状态后立刻更新，后继值使用更新前可用的估计。终点值为 0。
- 运行 value 后两者应趋近 [.9, 1]。将 `episodes` 设为 1，可以比较首次更新与本节的数值计算。

<a id="lesson-branches"></a>

## 6 · 持续预测需要说明哪一个对象在变

| 分支 | 实际改变 | 需要继续检查 |
| --- | --- | --- |
| 常数步长跟踪 | 让新数据持续改变估计 | 方差、变化速度、访问频率 |
| GVF | 把任务奖励改为任意累积信号，并明确策略与延续 | 预测题目是否定义正确 |
| GTD / emphatic TD | 改变离策略函数逼近的更新几何或加权 | 覆盖、方差、线性理论条件 |
| 平均奖励 TD | 去掉长期增长的奖励率，学习差分价值 | 奖励率与价值的联合估计 |
| 神经网络 TD | 用共享非线性表示 | 自举、离策略、函数逼近的耦合 |

常数步长保留对新数据的响应，同时留下采样噪声。标量均值更新 Q←Q+α(R−Q) 对历史奖励给出几何衰减权重；TD 还含自举与状态访问。例如单状态每步奖励 r、回到自己，TD 为 V⁺=[1−α(1−γ)]V+αr。α=.1、γ=.9 时旧 V 的系数是 .99，不是 .9；共享参数下还会产生矩阵耦合。因此，均值估计的遗忘速度不能直接作为 TD 的适应速度。

如果只让估计 V 更新，本章仍在求同一个固定策略价值。如果真实奖励或转移规律改变，要说明预测的是冻结当前世界后会得到的价值，还是包含未来变化规律的条件回报；用旧经验估计它们会有不同含义。如果策略也随经验更新，冻结当前策略的价值又与继续学习的未来收益分开。变化检测、滑动窗口和旧情境复用都必须围绕其中一个明确问题设计。

后续阅读按这个变化选择：[GVF](/zh/continual-rl/construction/predictive-knowledge/)改变预测规格；[第一册资格迹](/zh/continual-rl/foundations/approximation/traces/)完整推导线性在线前向参照；[时间信用分配](/zh/continual-rl/algorithms/credit-assignment/)继续检查表示、策略和更新时序改变后的信用；[持续控制](/zh/continual-rl/algorithms/control/)评价这些机制继续运行所得的行为。它们分别撤掉不同的固定条件，不能只用“非平稳”合并解释。

<a id="rlss-pid-dynamics"></a>

## 目标不变，求解动力学能否改变？

固定时域预测改变所问的未来。PID 加速路线则尝试保留原 Bellman 固定点，改变接近它的动态过程。先看已知模型的固定策略评价：令 B(v)=Tπv−v。普通价值迭代就是 v 加上这个残差。

$$
z_{k+1}=\beta z_k+a_I B(v_k),\qquad v_{k+1}=v_k+k_PB(v_k)+k_Iz_{k+1}+k_D(v_k-v_{k-1}).
$$

这是 PID value iteration 的结构。P 使用当前 Bellman 残差，I 保存衰减的残差积累，D 使用价值迭代差。k 是内部迭代次数；z 不是资格迹，不负责把真实后果分配给过去特征。

取 $k_P=1,k_I=k_D=0$ 恢复普通价值迭代。即使目标相同，参数也可能使递推失稳。最小例子是一个自循环状态，奖励一、折扣 0.9；正确价值为十。只有比例项时，误差递推为 $e_{k+1}=(1-0.1k_P)e_k$，因此需要 $0<k_P<20$ 才会收敛。目标正确不保证任意增益正确。

PID Accelerated TD 进一步处理只能获得样本的情形，包括随机逼近和增益适应。它不是在任意深网 TD error 上随意加一项积分与差分。原论文的 D 项使用价值差；对噪声很大的连续 TD error 直接作差，已经是另一规则。这里给出机制与稳定性入口，完整随机理论与论文实验仍属进阶阅读。

<a id="lesson-check"></a>

## 7 · 习题与讨论

问题：学习 TD 时把 target 中的下一状态价值也求导，会更“完整”吗？不一定。它改变了算法：TD 的半梯度更新是固定 target 后的回归方向；完整样本残差梯度优化另一目标。对期望 Bellman 残差平方求无偏梯度还涉及双采样。先写目标，再判断哪个梯度正确。

问题：为什么训练 TD error 不为零仍可能已经学对？随机奖励和随机转移会带来不可约的单样本误差；正确价值满足条件期望误差为零，不要求每条样本误差都零。用已知解析值或独立 rollout 估值，不能只看训练 loss。

## 本章的实验设计

在小型马尔可夫奖励过程中计算解析价值。再检查样本更新和参数误差。TD 误差不能代替价值误差。

设定：两状态奖励过程：A 无奖励到 B，B 获得 1 后回 A，γ=0.5。固定策略的价值为 v(A)=2/3、v(B)=4/3。

- 解析解满足两个 Bellman 方程。
- 零步长不写入，γ=0 只拟合下一奖励。
- 终端样本不读取无效后继值。

对照：MC、TD(0) 与指定版本 TD(λ)；相同轨迹与独立交互两种面板；固定表征及共同步长搜索预算

记录：对解析价值的均方误差；Bellman 残差与样本 TD 误差分别记录；固定数据量下的误差曲线

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-classic)

## 学习与研究衔接

表格 TD 的局部更新推广为共享特征上的参数更新。GVF 再改变预测问题，而非只改变网络。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-value) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=value) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=value)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

给定状态后可以估计价值；观测不足时，还要学习保留哪些历史。GVF 规定预测什么，RTRL 计算递归敏感度，资格迹组织时间信用。应分别检验信息是否进入状态、反馈能否教会这种保留，以及有限预测预算怎样分配，而不是把三者当作替代算法。

- [When does Self-Prediction help? Understanding Auxiliary Tasks in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-self-prediction-auxiliary-tasks)

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)
- [Safe and Efficient Off-Policy Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-retrace-safe-offpolicy)
- [Convergent Tree Backup and Retrace with Function Approximation](https://yingwen.io/zh/continual-rl/research/#recent-convergent-tree-retrace)
- [Multi-Step Reinforcement Learning: A Unifying Algorithm](https://yingwen.io/zh/continual-rl/research/#recent-q-sigma-backups)
- [A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning](https://yingwen.io/zh/continual-rl/research/#recent-lambda-greedy)
- [Per-decision Multi-step Temporal Difference Learning with Control Variates](https://yingwen.io/zh/continual-rl/research/#recent-openmind-control-variates)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [Reward Centering](https://yingwen.io/zh/continual-rl/research/#recent-reward-centering-discounted)
- [Extending Differential Temporal Difference Methods for Episodic Problems](https://yingwen.io/zh/continual-rl/research/#recent-openmind-episodic-differential)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning](https://yingwen.io/zh/continual-rl/research/#recent-lambda-greedy)

### When does Self-Prediction help? Understanding Auxiliary Tasks in Reinforcement Learning

Claas A. Voelcker, Tyler Kastner, Igor Gilitschenski, Amir-massoud Farahmand

RLC 2024 / RLJ · 2024 · 支持方法与理论

#### 研究问题

预测下一潜在状态、重建观测和学习价值，为什么会产生不同的表示？

#### 关键机制

论文在含干扰因素的线性问题中分析辅助目标的学习动力学。潜在状态自预测与价值学习共同作用时可能保留决策相关结构，但单独训练同一目标未必得到最有用的特征。目标的作用取决于它和 TD 目标怎样共享表示。

#### 证据

线性分析给出可检查的条件，并用神经网络实验检验部分预测。结果不支持“任何自监督预测都能改善 RL”这种无条件判断。

#### 条件与限制

线性分析中的观测映射、优化过程与神经网络控制并不完全等价。项目仓库入口不等于已经提供完整可复现实验实现，因此这里不列为可运行代码。

#### 阅读与实验

固定编码器容量，分别比较仅 TD、仅辅助任务和联合训练。记录价值误差与任务收益，不要只用辅助损失下降评价表示。

#### 原文与相关入口

- [RLC 2024 论文入口](https://rlj.cs.umass.edu/2024/papers/Paper197.html)：原文、线性假设与神经网络实验。
- [作者预印本](https://arxiv.org/abs/2406.17718)：便于追踪论文版本。

### Deep Reinforcement Learning with Gradient Eligibility Traces

Esraa Elelimy, Brett Daley, Andrew Patterson, Marlos C. Machado, Adam White, Martha White

RLC 2025 / RLJ · 2025 · 支持方法与理论

#### 研究问题

资格迹怎样与明确的梯度目标结合，而不是直接把线性半梯度规则搬到深度网络？

#### 关键机制

论文从广义投影 Bellman 误差出发构造多步目标，推导带资格迹的梯度学习方法。前向视角连接多步回报与经验重放，后向视角通过递推迹分配信用。目标函数、辅助估计器和迹的更新共同决定算法，不只是选择一个较大的 λ。

#### 证据

作者给出多种算法并在 MuJoCo、MinAtar 等任务中比较。代码同时提供相关梯度算法与实验设置，可以把推导中的量映射到实际更新。

#### 条件与限制

线性 GTD 的收敛条件不能自动赋予非线性实现全局收敛保证。重放版本与流式版本的数据使用预算也不能混为一谈。

#### 阅读与实验

从一段短轨迹分别计算前向多步目标和后向迹。随后对照原代码检查辅助网络、目标与主网络参数使用的是更新前还是更新后的值。

#### 原文与相关入口

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_302.pdf)：目标、算法推导与实验。
- [作者算法库](https://github.com/esraaelelimy/gtd_algos)：论文提供的梯度 TD 与资格迹实现。

#### 作者代码

[原论文链接的作者仓库。](https://github.com/esraaelelimy/gtd_algos)

论文梯度算法、资格迹和实验配置。

### Reward Centering

Abhishek Naik, Yi Wan, Manan Tomar, Richard S. Sutton

RLC 2024 / RLJ · 2024 · 支持方法与理论

#### 研究问题

接近一的折扣为何使共同价值偏移很大，中心化能改善什么、又不能改变什么？

#### 关键机制

从折扣价值的共同偏移与相对价值分解出发，移除奖励参照量；on-policy 可估计行为奖励均值，off-policy 提出 TD 驱动的参照更新。保留小于一的折扣时，中心化没有消除折扣对策略排序的影响。

#### 证据

原文给出理论动机与表格、线性、非线性控制实验，检验折扣及奖励常数平移。深度 continuing-task 后续研究扩大了算法与环境范围。

#### 条件与限制

TD 中心化中的标量在有限折扣下不必精确等于真实奖励率。训练期的联合参照/价值更新与固定常数下的平移恒等式需分别分析；真实终止改变平移条件。

#### 阅读与实验

用单状态常奖励问题解出联合更新固定点，再用多动作问题检查策略排序；同时记录参照量与直接观测的外部奖励率。

#### 原文与相关入口

- [RLC 2024 原文](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_261.pdf)：中心化分解、on/off-policy 区别及收敛讨论。
- [RLJ 论文记录](https://rlj.cs.umass.edu/2024/papers/Paper261.html)：正式题名、作者与会议年份。

### Safe and Efficient Off-Policy Reinforcement Learning

Rémi Munos, Tom Stepleton, Anna Harutyunyan, Marc G. Bellemare

NeurIPS 2016 · 2016 · 支持方法与理论

#### 研究问题

目标与行为策略不一致时，如何保留多步信用而避免重要性比率乘积爆炸？

#### 关键机制

统一多步目标为目标策略TD误差的加权和，Retrace采用λmin(1,π/μ)传播系数。近同策略时保留长迹，目标概率较低的动作则减少传播；一步误差仍使用目标动作期望。

#### 证据

论文分析表格算子的收缩性质，给出条件下的评价与控制收敛，并报告Atari实验。信用章独立检查传播系数和有限轨迹恒等式。

#### 条件与限制

表格安全性不是任意线性或神经逼近的稳定性保证。行为覆盖、变化策略与投影条件仍需检查；代码小实验不复现Atari。

#### 阅读与实验

在同样轨迹与表示上，分别改变策略差异和动作随机性，比较Tree-backup与Retrace的信用长度、方差和预测误差。

#### 原文与相关入口

- [原论文](https://arxiv.org/html/1606.02647)：统一算子、传播系数及理论条件。

### Convergent Tree Backup and Retrace with Function Approximation

Ahmed Touati, Pierre-Luc Bacon, Doina Precup, Pascal Vincent

ICML 2018 · 2018 · 支持方法与理论

#### 研究问题

传播系数已经截断，为什么函数逼近下的Tree-backup和Retrace仍可能发散？

#### 关键机制

分析函数逼近与off-policy多步bootstrap的学习算子，展示线性反例，再把相应目标写成二次凸凹鞍点问题，构造梯度版本。

#### 证据

原文给出线性不稳定例子、梯度方法收敛保证与有限样本界。它直接限定了从Retrace表格结论外推到逼近算法的范围。

#### 条件与限制

凸凹线性问题的保证不能自动覆盖学习表示的深度网络。稳定目标、更新速度与控制性能还需分别验证。

#### 阅读与实验

先检查固定表示下的期望更新矩阵，再将半梯度和梯度版本按相同样本、步数与计算预算比较。

#### 原文与相关入口

- [ICML原文](https://proceedings.mlr.press/v80/touati18a.html)：理论反例、鞍点方法和保证条件。

### Multi-Step Reinforcement Learning: A Unifying Algorithm

Kristopher De Asis, J. Fernando Hernandez-Garcia, G. Zacharias Holland, Richard S. Sutton

AAAI 2018 · 2018 · 支持方法与理论

#### 研究问题

多步动作价值目标必须始终采样下一动作，或始终对动作取期望吗？

#### 关键机制

Q(σ)逐处混合Sarsa的采样动作与Expected Sarsa的动作期望，并同步改变后续误差传播。σ控制采样程度，与控制回报长度的λ不同。

#### 证据

原文给出统一n-step表达、off-policy修正和实验比较。信用章小程序核验冻结on-policy几何λ混合的两个端点。

#### 条件与限制

原文n-step和本章λ混合参考具有不同实现范围。只改一步误差却不改多步传播或策略修正，不能称为完整Q(σ)。

#### 阅读与实验

把采样噪声、目标长度和策略差异分开改变，避免把σ与λ的作用归到同一“更长信用”解释。

#### 原文与相关入口

- [原文](https://arxiv.org/html/1703.01327)：式13–15：混合误差、传播及off-policy修正。

### A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning

Martha White, Adam White

arXiv预印本 · 2016 · 支持方法与理论

#### 研究问题

不同状态的预测可靠性不同，固定λ是否浪费了多步信用？

#### 关键机制

将下一处bootstrap选择写成局部偏差平方与回报方差的折中，得到$λ=b^2/(b^2+\operatorname{Var}(G))$。完整λ-greedy还用在线预测器估计回报均值和二阶矩。

#### 证据

原文给出状态相关λ的目标、增量算法和多个预测设置的实验。信用章仅核对已知统计量下的局部最优与变量λ恒等式。

#### 条件与限制

局部贪心目标不是整条轨迹的联合最优。逼近误差、统计滞后和非平稳性会影响λ估计；辅助资源需要计入比较。

#### 阅读与实验

先让噪声方差变化，再让bootstrap可靠性变化。比较固定λ、已知统计参照和在线估计，分别观察目标偏差与适应速度。

#### 原文与相关入口

- [作者原文](https://arxiv.org/html/1607.00446)：局部目标、状态λ、均值／二阶矩预测与完整算法。

### Extending Differential Temporal Difference Methods for Episodic Problems

Kris De Asis, Mohamed Elsayed, J. He

RLC 2026 / RLJ · 2026 · 支持方法与理论

#### 研究问题

中心化怎样加速回合任务的学习，又不因为回合长度不同而改变目标？

#### 关键机制

处理终止边界的中心化尾项，再用共享价值偏置重参数化。区分奖励单位的中心与价值单位的偏置，使最终终止的无折扣回合也能使用适当形式。

#### 证据

论文分别研究固定中心化的策略不变性、在线偏置学习的线性 TD 分析，以及流式深度实验。教材给出遗漏终止补偿导致排序翻转的两动作反例。

#### 条件与限制

这里沿用原回合目标，不是把目标改成长期平均奖励。线性预测条件不保证非线性控制全局收敛；采样截断也不是自然终止。

#### 阅读与实验

计算立即终止和延迟终止两条路径的原始、错误中心化、正确补偿回报。再辨认代码里的偏置、终止分支和更新前 TD 误差。

#### 原文与相关入口

- [RLC 正式记录](https://rlj.cs.umass.edu/2026/papers/Paper33.html)：问题、保证与实验分别阅读。
- [原文](https://arxiv.org/html/2605.04368v1)：终止补偿、共享偏置与回合式扩展。

### Per-decision Multi-step Temporal Difference Learning with Control Variates

Kristopher De Asis, Richard S. Sutton

UAI 2018 · 2018 · 支持方法与理论

#### 研究问题

怎样保留长路径中的新奖励信息，同时减去已经可预测的采样波动？

#### 关键机制

在逐决策重要性采样回报中加入条件均值为零的控制变量。期望动作价值承担可预测部分，重要性比率仍作用于真实回报相对当前预测的残差。

#### 证据

原文统一讨论动作和状态价值的多步目标，并连接 Expected Sarsa、Tree-backup 与 Retrace。教材枚举一个两动作例子的期望和方差。

#### 条件与限制

无新增偏差不代表没有 bootstrap 误差，也不保证任意差预测都降低方差。此为研究者加入 Openmind 前的工作。

#### 阅读与实验

保持采样策略、价值函数与路径长度不变，分别计算控制变量前后的均值和方差；随后再讨论神经网络参数变化。

#### 原文与相关入口

- [原论文](https://arxiv.org/html/1807.01830v1)：重点读动作价值回报、条件均值与 λ-return 的关系。


<a id="chapter-code"></a>

## 下载与运行

Python 3.10+，仅标准库。实现表格 MC/TD 完整小实验；资格迹的前后向推导与实现见时间信用分配章。

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

```sh
python3 foundations_detail_lab.py value
python3 foundations_detail_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：第 3–8 章建立 MDP、动态规划、MC、TD 与规划；第 9–13 章把这些更新扩展到函数逼近、资格迹和策略梯度。按本章的问题找相应章节，不必从头重读。

- [Sutton · Learning to Predict by the Methods of Temporal Differences](https://doi.org/10.1007/BF00115009)：TD 的原始问题动机与多步预测。读“如何利用尚未结束的经验”，并理解其更新规则。

- [van Seijen & Sutton · True Online TD($\lambda$)](https://proceedings.mlr.press/v32/seijen14.html)：检查在线前向视图、Dutch trace 与修正项；它解决的不是简单加大 $\lambda$。

- [Bedaywi、Rakhsha、Farahmand · PID Accelerated Temporal Difference Algorithms](https://arxiv.org/abs/2407.08803)：§2.1 PID value iteration 与 §3 采样扩展；区分残差积分、价值差和资格迹。
