# 策略梯度、Actor–Critic 与 PPO

策略梯度和 PPO 的一次更新以什么行为分布、价值版本与评价目标为参照？把这些更新连成持续学习过程后，哪些结论还需要重新检验？

## 本章内容

- 沿轨迹概率与 baseline 恒等式，区分当前策略目标和完整学习器目标。
- 核对 GAE 的旧价值、bootstrap 与跨序列边界。
- 用保留的 PPO 算例检查批内冻结量，再说明 rollout 与更新预算怎样进入生命期比较。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [策略梯度、基线与 Actor–Critic](https://yingwen.io/zh/continual-rl/foundations/approximation/policy-gradient/)：从轨迹概率到策略梯度的推导在第一册。
- [策略更新的尺度：TRPO 与 PPO](https://yingwen.io/zh/continual-rl/foundations/deep/trust-region/)：批内代理目标与策略更新尺度在第二册。


### 一条经验

在 $S_{t}$ 做 $A_{t}$ 后得到 $R_{t+1}$、$S_{t+1}$。奖励属于这次转移，不是到达状态之前另一轮的奖励。

### 折扣与终止

$\gamma$ 控制未来相对权重；真实终止后的价值设为 0。本章记 $\gamma_{t+1}$=$\gamma$(1−terminated)。训练脚本的时间截断不一定是任务终止。

### 表格与函数逼近

表格每个状态或状态动作一组独立参数；函数逼近让不同输入共享参数。更新一个输入可能改变其他输入的预测。

### 随机策略

$\pi_\theta$(a|s) 是动作概率。二动作可用 p=$\sigma$($\theta$)，softmax 推广到多动作；连续动作通常是参数化概率密度。

### 梯度上升与损失下降

要最大化回报 J，就沿 ∇J 上升；框架常定义 loss=−代理目标再下降，符号必须一致。

<a id="problem-definition"></a>

## 本章的问题定义

直接改善参数化随机策略；策略变化也改变后续数据分布。先以固定有限时域推导，再区分自举优势和旧数据代理。

### 给定条件与符号

- 初始分布、环境接口、有限时域与回报准则。
- 可微策略类、采样长度、优势估计器、优化次数和数据权限。

### 需要求解的对象

策略参数及指定回报目标的梯度估计；critic为估计辅助量，PPO裁剪目标为局部代理。

### 信息与数据权限

$\tau$ 是由策略 $\pi_\theta$ 产生的轨迹。PPO保存采样时旧动作概率，不能在每轮优化中重算分母；时间截断与真实终止分别处理。

$$
\max_\theta J(\theta),\qquad J(\theta)=\mathbb E_{\tau\sim p_\theta}\!\left[\sum_{t=0}^{T-1}R_{t+1}\right]
$$

$\theta$ 为策略参数，$p_\theta$ 为策略诱导的轨迹分布，$T$ 为固定有限时域。本章先用不折扣目标；若改为从起点严格折扣，梯度需相应时间/占用权重。PPO旧数据裁剪代理最大化不等于精确最大化此 $J$。

### 成立条件与解的含义

- 环境与初始分布不依赖策略参数；score求导需可交换期望与求导等正则条件。
- baseline不依赖当前动作且actor将其视为固定权重；近似critic、GAE和数据重用引入的误差分别声明。

判断准则：小bandit上梯度方向与精确期望/有限差分一致，正负优势裁剪分支及GAE边界正确；收益以新交互评估，不由actor loss替代。

### 适用边界

- PPO裁剪不提供所有状态上的硬KL信赖域。
- 局部梯度方向不保证有限大步后回报单调增加。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 限制表示或采用近似 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：参数化策略梯度提供局部策略改善，不穷举完整有限资源学习器。

- 组合不同学习问题 · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)：GAE和多步优势为动作梯度分配时间信用，不是独立控制目标。

- 改变评价目标 · [最大熵控制](https://yingwen.io/zh/continual-rl/algorithms/soft-control/)：相对最大熵控制，本章基本目标只累计外部奖励；若另外加入熵项就改变该基本目标。PPO与SAC的数据协议差异还需另行说明。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

环境奖励不可直接沿动作反传，完整回报的梯度估计又有较大方差；重复优化会离开采样策略。

### 本章的核心思路

先对轨迹概率求导，再用因果性和baseline减少无关噪声，最后明确控制旧数据代理的偏移。

1. [沿概率而非环境奖励求导](#lesson-derive)：因为动作改变轨迹分布，用log-derivative将真实回报转为采样score权重。

2. [用critic与多步优势降低等待](#policy-gae)：因为完整回报长且噪声大，TD与GAE用估计补尾；critic误差和混合长度带来相应偏差。

3. [限制旧数据的局部优化激励](#policy-ppo)：因为新策略会偏离旧采样分布，保存旧概率并按优势符号裁剪PPO代理，再以新交互验证。

结论与条件：精确score和合格baseline保持期望梯度；近似critic/均匀rollout/裁剪代理需各自解释，PPO本章实现没有全局最优或硬信赖域保证。

### 相关方法改变了什么

- REINFORCE：完整采样回报提供梯度权重，等待和方差较大。

- Actor–critic/GAE：以自举价值与多步优势替代完整回报，依赖critic质量。

- PPO/TRPO：分别通过裁剪代理与约束近似管理策略变化，求解成本和保证不同。


<a id="lesson-setting"></a>

## 1 · 先明确求导的是哪一种未来行为

第一册的[策略梯度](/zh/continual-rl/foundations/approximation/policy-gradient/)建立占用分布与梯度定理，第二册的[actor–critic 与 GAE](/zh/continual-rl/foundations/deep/policy-gradient/)和 [TRPO/PPO](/zh/continual-rl/foundations/deep/trust-region/)完成估计器与训练循环。本章回顾其中的固定量，连接到持续控制：一次局部更新的方向，与长期执行这些更新所得的收益，需要分别定义。

设轨迹 $\tau$=($S_{0}$,$A_{0}$,$R_{1}$,…,$S_T$)，环境动力学与初始分布不依赖策略参数 $\theta$。先取有限 $T$、不折扣总回报，并在生成一条轨迹时使用同一套 $\theta$。即使参数固定，动作仍会改变后续状态分布；下一节的 score-function 推导已经计入这条作用路径。

$$
J(\theta)=\mathbb E_{\tau\sim p_\theta}[\sum_{t=0}^{T-1}R_{t+1}],\qquad p_\theta(\tau)=p_0(s_0)\prod_t\pi_\theta(a_t\mid s_t)p(s_{t+1},r_{t+1}\mid s_t,a_t)
$$

这里求导的是这一固定参数策略诱导的回报期望。若在轨迹中按 $\theta_{t+1}=U(\theta_t,\ldots)$ 继续学习，未来动作还受更新规则影响；不能把上式中的同一个 $\theta$ 悄悄换成每步变化的参数，再声称已经求出了对初始化或学习规则的生命期梯度。

<a id="lesson-derive"></a>

## 2 · 对数导数、因果性与 baseline

$$
\nabla J=\sum_\tau p_\theta(\tau)R(\tau)\nabla\log p_\theta(\tau)=\mathbb E[\sum_t\nabla\log\pi_\theta(A_t\mid S_t)R(\tau)]
$$

使用 ∇p=p∇log p；环境项与 $\theta$ 无关，只有策略项留下。连续空间将求和换积分，还需要可交换求导与积分等正则条件。

![两步共享 Bernoulli 策略的四条轨迹概率，以及解析目标曲线上的一次梯度更新。](https://yingwen.io/crl-figures/concept-classic-policy-gradient.svg)

分支宽度是动作概率，末端数字是完整轨迹概率。这个独立例子不给策略阶段信息，仅动作 1 后接动作 0 的序列获奖；折扣为 0.5。参数从 0.7 沿精确期望梯度更新到约 0.513553，使动作 1 概率从 0.668 降到 0.626。曲线是解析目标，不是采样训练表现。

$$
\nabla J=\mathbb E[\sum_t\nabla\log\pi_\theta(A_t\mid S_t)G_t],\qquad G_t=\sum_{k=t}^{T-1}R_{k+1}
$$

当前动作不能影响之前已经发生的奖励。条件在历史上，策略 score 的期望为零，因此可从权重中删掉过去奖励而保持同一期望梯度。

$$
\sum_a\pi_\theta(a\mid s)\nabla\log\pi_\theta(a\mid s)b(s)=b(s)\nabla\sum_a\pi_\theta(a\mid s)=0
$$

给定动作前历史，baseline 须在本次采样前确定或满足相应条件独立性，才有上面的零期望。取真实 $b=V_\pi$，权重成为 advantage；共享参数时 actor 更新须将 baseline 当作固定权重。stop-gradient 只控制求导路径，不会消除先用同一样本拟合 baseline 所造成的统计依赖。

$$
A_\pi(s,a)=Q_\pi(s,a)-V_\pi(s),\qquad \delta_t=R_{t+1}+\gamma_{t+1}V(S_{t+1})-V(S_t)
$$

真实 V 与 on-policy 数据下，TD residual 的条件期望给 advantage。使用近似 V 会引入估计误差；actor–critic 就是在用较低方差、可能有偏的 critic 信号替代完整回报。

若 J 是从固定起点的严格折扣目标，轨迹求和形式中需要相应 $\gamma$^t 权重，或等价地在折扣占用分布下取样。很多实现直接对 rollout 的每个时刻均匀平均，再使用 $\gamma$ 折扣的 critic；这是常见代理方式，不能不说明就称与任意 J 完全同一梯度。

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

<a id="policy-gae"></a>

## 3 · GAE 把多步 advantage 写成反向递推

$$
\hat A_t=\delta_t+\gamma_{t+1}\lambda c_t\hat A_{t+1},\qquad \hat R_t=\hat A_t+V_{\mathrm{old}}(S_t)
$$

$c_t$ 表示后继样本是否仍在本次可用的同一条轨迹内；真实终止、reset 或采样块边界时置 0。沿用第二册的 c 作为跨序列 mask；其 bootstrap mask $b_t$ 在这里并入 $\gamma_{t+1}=\gamma b_t$。所有 TD residual 使用同一旧 critic。从末尾向前算即可，$\lambda$=0 只用一步 TD；value target 是原始优势加旧 value。

三个边界必须拆开：真实终止：bootstrap=0 且 trace 停；time-limit 截断：若原任务继续，bootstrap 使用最终观测 value，但 trace 不跨进 reset 的新轨迹；rollout 采样块结束：bootstrap 使用块尾状态 value，未采到的 GAE tail 截断为 0。这些不是一个 done 布尔变量能无歧义表达的。

第 1 节先用不折扣目标说明 score 推导；这里保留通用折扣 critic 的 GAE 写法。若要估计起点折扣目标的梯度，actor 仍需上一节说明的时间权重或相应采样分布。训练 actor 前固定原始 advantage、旧 log-prob 和 critic target；标准化若只用于 actor，不能再把它加回旧 value 作为 critic 标签。

这个反向递推在一批已经收到的数据上生成固定权重。[时间信用分配章](/zh/continual-rl/algorithms/credit-assignment/)改为让误差到来时立即影响过去的参数用途，须重新规定参数版本和在线等价条件。仅因两种计算都含 λ，就把批量 GAE 与逐条 TD(λ) 当作同一更新，会遗漏这些差别。

<a id="policy-ppo"></a>

## 4 · 从重要性比率到 PPO 的单样本分段函数

$$
r_t(\theta)=\exp[\log\pi_\theta(A_t\mid S_t)-\log\pi_{\mathrm{old}}(A_t\mid S_t)],\qquad L^{\mathrm{clip}}=\mathbb E_t[\min(r_t\hat A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)]
$$

比率修正旧状态上的动作分布；它不是完整轨迹重要性采样，也没修正当前策略诱导的新状态分布。PPO 因此是局部代理优化，不能把每次 loss 降低解释成严格策略改善。

![正优势和负优势对应的 PPO 裁剪函数，两者的平坦区位于不同侧。](https://yingwen.io/crl-figures/learning-classic-ppo-clip.svg)

横轴是新旧动作概率比。正优势停止奖励过大的概率增幅，负优势停止奖励过大的概率降幅；错误方向的梯度仍保留。图中的分段函数与本节 min 公式对应。

| 优势符号 | 停止继续奖励的区间 | 仍然保留梯度的情形 |
| --- | --- | --- |
| A>0 | r>1+$\epsilon$：已经增加够多 | r 太低时继续鼓励恢复 |
| A<0 | r<1−$\epsilon$：已经减少够多 | r 太高时继续惩罚错误增加 |

注意 min 的方向：负 advantage 乘以较大的 ratio 更负。因此不能先把 ratio 双向裁剪再简单乘 A；那会在应该纠错的方向也抹掉梯度。裁剪也不是硬 KL 约束，整个网络共享参数仍可能改变其他动作，通常还要监测 KL 并限制优化轮数。

**算法：PPO 完整循环；`old_logprob` 在 K 轮里始终不变**

1. 给定策略参数 $\theta$、价值参数 $\phi$、裁剪范围 $\epsilon$ 与优化轮数 $K$。
1. 每轮采样：
  1. 固定行为策略 $\pi_{\rm old}$，采集轨迹，保存动作对数概率和价值预测。
  1. 根据真实终止与时间截断计算 GAE，固定 $\widehat A_t$ 与价值目标 $\widehat V_t$。
  1. 重复 $K$ 轮，在轨迹中抽取小批量样本：
    1. $r_t\leftarrow\exp[\log\pi_\theta(A_t\mid S_t)-\log\pi_{\rm old}(A_t\mid S_t)]$
    1. $L_\pi\leftarrow-\operatorname{mean}\{\min[r_t\widehat A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\widehat A_t]\}$
    1. $L_V\leftarrow\operatorname{mean}[(V_\phi(S_t)-\widehat V_t)^2]$
    1. 按给定权重优化策略、价值和熵正则项。
    1. 根据预先设定的 KL 阈值决定是否提前停止本轮优化。
  1. 用更新后的策略采集下一轮轨迹。

<a id="experiment-deep-ppo"></a>

### 实验：实验 · PPO 的旧数据、多轮更新与裁剪

PPO 的完整更新流程能否在同一小任务上带来可见收益？

**环境与可用信息。** DeadlineChain：位置 0–4、左右动作、边界截断。观测为五维位置 one-hot 加剩余时间比例。到位置 4 得 1 并终止；其他步得 −0.02。12 步截止也是任务真实终止，且剩余时间可观测。每回合重新从位置 0 开始，网络跨回合保留。

**设置。** 1200 个真实步、同样的 6→32 tanh actor/critic 和初始化；actor Adam 0.003、critic Adam 0.01。每 60 步收集一批，γ=1、GAE λ=0.95，优势标准化，clip=0.2，最多 4 次 actor 更新，估计 KL 超过 0.03 时提前停；critic 做 4 次更新。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** ppo.py 保存采样时的旧 log-prob 和旧价值。裁剪目标对正负优势分别起作用；多轮优化不重算旧策略概率。该实验的 VPG 对照每个完整回合只进行一次策略更新。

**测量。** 纵轴是冻结 argmax 行为的原环境回报。应把数据采集步数、重复梯度更新与 rollout 边界分开；图不是 clip 单组件消融。

```bash
python3 implementations/deep/ppo.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/deep-ppo/curves.svg)

横轴：训练环境步（独立评估交互另计）。纵轴：冻结策略的外部回报。每种方法 1200 训练环境步；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 冻结当前 actor，在独立 DeadlineChain 中按 argmax 动作重复12次完整回合，取未折扣外部回报的平均。每次都从位置0、剩余12步开始；到位置4奖励1，其余步奖励−.02。评价环境和动作选择都确定，同一网络的12次回合相同，不能当成12个训练种子。

**step：怎样计时。** step 只数训练环境转移，评价交互另计。旧记录每60步用新批次更新，1200步有20批；每批最多4次全批策略梯度和固定4次 critic 梯度。策略更新前的样本近似 KL 大于.03时停止余下策略更新，实际策略梯度次数没有写入CSV；updated_batches 计批次，不能直接当成梯度次数。采样截止不重置环境，真实终止才关闭自举。

**怎样汇总。** 取该记录时刻的值；不先对曲线上的时间点求平均。 第0步评价尚未更新的网络；其余点在本批更新后评价。训练用γ=1、λ=.95的GAE和批内归一化优势，旧 log-prob、优势与回报在多轮优化中冻结；图中 argmax 回报与随机采样策略的期望回报、PPO代理目标分别定义。这是单环境、全批次教学实现，不是论文基准复现。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** CSV 可重算冻结回报均值和样本标准差，samples 核对真实预算，updated_batches×4给出critic梯度数（末点80）。已知确定性任务下，成功时回合长度L=1+(1−value)/.02，失败时L=12；12×L给本检查点额外评价转移数。日志没有逐步训练奖励、旧概率、优势、KL、clip fraction或actor更新计数；不能恢复训练累计收益、随机策略回报、实际信赖域变化或策略梯度总数。policy_loss和value_loss是最近批次中最后一次优化所用损失，不能跨方法直接比较其数值。

计算位置：[deep/ppo.py](https://yingwen.io/crl-code/implementations/deep/ppo.py) · [deep/vpg.py](https://yingwen.io/crl-code/implementations/deep/vpg.py) · [deep/_common.py](https://yingwen.io/crl-code/implementations/deep/_common.py)

</details>

**结果分析。** 第 600 步和第 1200 步，PPO 与 VPG 的平均冻结回报均为 0.94，五个 seed 在这些检查点也相同。这里没有实测终点优势；小链可能无法区分两套训练程序。

**结论边界。** PPO 与 VPG 同时在采样分段、GAE、优势标准化和更新次数上不同。不能把这张图称为“裁剪提升”的因果证据。argmax 行为相同不代表两者动作概率相同。

**继续实验。** 保持同一 rollout、同一更新次数与同一优势，仅开关裁剪。记录概率比、KL 和 clip fraction，再观察重新采样后的收益。

[源码](https://yingwen.io/crl-code/implementations/deep/ppo.py) · [逐种子记录](https://yingwen.io/crl-code/results/deep-ppo/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/deep-ppo/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/deep-ppo/curves.json)

<a id="lesson-example"></a>

## 5 · PPO 裁剪与 GAE 的数值例子

旧策略选择动作 1 的概率 .5，新概率 .75，ratio=1.5，$\epsilon$=.2。若 A=2，两个候选值为 3 和 2.4，取 2.4，该样本正向改善梯度已为 0；若 A=−2，则取 min(−3,−2.4)=−3，仍推动降低该动作概率。

两步奖励 [0,1]，旧 V=[.2,.5]，真实终止后的 value=0，$\gamma$=.9、$\lambda$=.8。$\delta_{1}$=1−.5=.5，$\delta_{0}$=0+.9×.5−.2=.25；$A_{1}$=.5，$A_{0}$=.25+.72×.5=.61。critic targets 为 [.81,1]。这和把 value target 简单设成 reward 不同。

<a id="lesson-code"></a>

## 6 · 核心实现：裁剪、边界递推与采样训练

二动作 Bernoulli PPO 与独立 GAE 函数

```python
def sigmoid(theta):
    if theta >= 0:
        return 1.0 / (1.0 + math.exp(-theta))
    exp_theta = math.exp(theta)
    return exp_theta / (1.0 + exp_theta)


def ppo_sample(theta, old_probability, action, advantage, epsilon=0.2):
    p = sigmoid(theta)
    probability = p if action == 1 else 1.0 - p
    ratio = probability / old_probability
    clipped = min(1.0 + epsilon, max(1.0 - epsilon, ratio))
    objective = min(ratio * advantage, clipped * advantage)
    inactive = (advantage > 0 and ratio > 1.0 + epsilon) or (advantage < 0 and ratio < 1.0 - epsilon)
    derivative = 0.0 if inactive else advantage * ratio * (action - p)
    return objective, derivative


def gae(rewards, values, next_values, discounts, trace_continues, lam=0.95):
    # At time-limit truncation, bootstrap may survive but the trace must not
    # jump into the next reset episode. next_values uses FINAL observation.
    out, tail = [0.0] * len(rewards), 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + discounts[t] * next_values[t] - values[t]
        tail = delta + discounts[t] * lam * trace_continues[t] * tail
        out[t] = tail
    return out


def train_ppo_bandit(rounds=100, seed=7):
    rng, theta = random.Random(seed), 0.0
    for _ in range(rounds):
        old_p = sigmoid(theta)
        batch = []
        for _ in range(64):
            action = int(rng.random() < old_p)
            # Exact old-policy baseline for this reward function.
            batch.append((old_p if action else 1.0 - old_p, action, action - old_p))
        for _ in range(4):
            grad = sum(ppo_sample(theta, *sample)[1] for sample in batch) / len(batch)
            theta += 0.1 * grad
    return {"probability_good_action": sigmoid(theta)}
```

最小 bandit 动作 1 得 1、动作 0 得 0，真实 $V_{\rm old}=p_{\rm old}$，因此 baseline 不需要另训练一个 critic。它隔离 PPO 的采样、旧概率冻结、多个 epoch 与裁剪机制；多步 actor–critic 的 GAE 是独立函数，并测试 time-limit 是否错误串到新回合。

运行后好动作概率应从 .5 上升。测试用中心有限差分核对裁剪之外及平坦区的导数，另检查正负 advantage 的不对称。这个 bandit 实验隔离策略裁剪的作用；多步控制还需要价值回归、轨迹边界处理和环境交互。

<a id="lesson-branches"></a>

## 7 · 将局部更新放回持续运行的学习器

| 方法 | 主要改变 | 关键边界 |
| --- | --- | --- |
| REINFORCE | 完整 return 的 Monte Carlo 梯度 | 方差、等待长度 |
| actor–critic / A2C / A3C | 用 critic 与自举优势 | critic 误差、同步方式 |
| GAE | 优势估计器的多步组合 | 不是独立控制算法 |
| TRPO | 约束或近似约束策略改变 | 二阶近似、求解成本 |
| PPO | 裁剪或惩罚局部代理目标 | 不提供任意更新的硬信赖域 |
| streaming actor–critic | 每条新经验立即更新 | 尺度、迹、近似梯度时序 |
| SAC | 离策略熵正则控制 | 目标与数据协议都改变 |

这些方法分别改变估计器、更新幅度或数据协议，可以成为同一个持续学习器的内部步骤。PPO 在一轮优化中固定旧概率与优势，下一轮仍会收集新经验、重算 critic 与更新参数。因此“批内固定”与“部署时冻结学习”有不同含义；前者并没有排除长期适应。

持续比较时，rollout 长度决定等多久才能使用新反馈，epoch 数与每步延迟决定付出多少计算。增加 epochs 能更充分拟合旧批次，却不保证及时响应后来发生的变化。[流式学习](/zh/continual-rl/algorithms/streaming/)研究撤掉批量等待后的状态与计算限制；[元学习](/zh/continual-rl/algorithms/meta/)进一步对更新规则的影响求导。这些问题需要新的时序与梯度路径，不能由一次 PPO surrogate 改善直接回答。

最后按[持续控制的比较对象](/zh/continual-rl/algorithms/control/#lesson-branches)评价完整闭环：算法设计和预算保持一致，内部参数继续学习，行动可以生成不同经验。若只冻结一份 rollout 比较优化器，得到的是更新机制诊断；允许更新改变下一批行动后，才测到它对实际交互收益的作用。

<a id="lesson-check"></a>

## 8 · 习题与讨论

为何 `actor_loss` 变低不代表回报更高？这是固定旧数据上的局部代理；策略更新后数据分布改变，过多优化可能损害未见状态。必须实际评测交互收益和变化后表现。

为何必须保存 `old_logprob`？若每个 epoch 都用新网络重算分母，ratio 会不断回到 1，裁剪失去对原采样策略的参照。只保存动作、但不保存对应旧分布，是常见的隐蔽错误。

## 本章的实验设计

梯度检查使用固定轨迹。性能评价则必须重新采样。两种实验不能互相替代。

设定：固定短 rollout，分别给正负优势并跨越 PPO 裁剪边界；然后在可枚举小策略上对期望目标核导数。

- ratio=1.4、A=2、clip=0.2 的最大化目标为 2.4。
- ratio=1.4、A=−2 时目标为 −2.8，不能统一先裁剪 ratio。
- 外部截断可以 bootstrap，但不得把新 episode 的 GAE 接回旧轨迹。

对照：REINFORCE 与带基线估计器；固定样本和真实重采样分开；相同旧策略数据与明确 epoch 制度

记录：概率比、clip fraction、KL 与熵；固定样本梯度误差；独立交互收益及 rollout/更新成本

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-deep)

## 学习与研究衔接

从 REINFORCE 到 actor–critic，评论家提供低方差学习信号。PPO 的批量多轮更新与严格流式协议不同。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-policy) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=policy) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=policy)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://yingwen.io/zh/continual-rl/research/#recent-vtrace-impala)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Intentional Updates for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-intentional-updates)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Intentional Updates for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-intentional-updates)

### Intentional Updates for Streaming Reinforcement Learning

Arsalan Sharifnassab, Mohamed Elsayed, Kris De Asis, A. Rupam Mahmood, Richard S. Sutton

ICML 2026 · 2026 · 支持方法与理论

#### 研究问题

能否先规定本次更新应产生多大作用，再反推合适的参数更新尺度？

#### 关键机制

Intentional 方法以局部线性近似连接参数变化和预测变化。critic 以减少一定比例的 TD 误差为目标，actor 控制策略输出变化的局部代理量；再结合资格迹和逐坐标尺度，求出这一次更新的强度。这是有目标的局部更新控制，不是对长期表现求导的元梯度。

#### 证据

论文给出推导和流式控制比较，ICML 2026 正式论文入口与作者实现均可用。实现将优化器与 actor–critic 交互区分开，便于检查更新时序。

#### 条件与限制

Taylor 近似在大更新时可能失准。采样动作上的对数概率变化不等于精确的全分布 KL 上界；熵项与 TD 误差符号也必须按原算法处理。

#### 阅读与实验

在一次更新前后直接测量预测变化，并与线性估计比较。分别测试正、负 TD 误差和很小梯度的情形，不要只检查参数是否有限。

#### 原文与相关入口

- [ICML 2026 原文](https://proceedings.mlr.press/v306/sharifnassab26a.html)：正式会议版本与更新意图的定义。
- [作者实现](https://github.com/sharifnassab/Intentional_RL)：重点对照 optimizer.py 与 intentional_ac.py。

#### 作者代码

[原论文作者提供的实现。](https://github.com/sharifnassab/Intentional_RL)

Intentional 更新与流式 actor–critic。

### IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures

Lasse Espeholt, Hubert Soyer, Rémi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, Koray Kavukcuoglu

ICML 2018 · 2018 · 支持方法与理论

#### 研究问题

actor采样策略落后于learner时，如何校正状态价值与策略更新？

#### 关键机制

V-trace用截断ρ校正当前TD误差，用独立截断c控制后续误差传播，再用下一状态V-trace目标构造actor优势。ρ上限还决定表格固定点对应的截断策略。

#### 证据

原文分析固定点并检验分布式多任务训练。固定版本作者代码明确区分clipped_rhos、cs、反向scan与pg_advantages。

#### 条件与限制

IMPALA保存短轨迹并批量训练，不属于严格单样本流式协议。截断后价值可能对应不同于原目标的策略；信用章bandit示例显示0.8变为0.5。

#### 阅读与实验

独立改变策略滞后、ρ上限和c上限。记录目标策略变化与传播长度，不把两种截断都只解释为方差控制。

#### 原文与相关入口

- [ICML原文](https://proceedings.mlr.press/v80/espeholt18a.html)：V-trace固定点与分布式实验。
- [作者固定实现](https://github.com/google-deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/vtrace.py)：from_importance_weights与下一状态actor目标。

#### 作者代码

[原作者团队仓库的固定版本。](https://github.com/google-deepmind/scalable_agent/tree/6c0c8a701990fab9053fb338ede9c915c18fa2b1)

IMPALA原始TensorFlow实现与V-trace；运行需要原项目环境。


<a id="chapter-code"></a>

## 下载与运行

Python 3.10+，仅标准库。执行 PPO bandit 完整采样/优化循环，独立实现多步 GAE；不声称包含网络 PPO 的通用训练工程。

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

```sh
python3 foundations_detail_lab.py policy
python3 foundations_detail_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Williams · Simple statistical gradient-following algorithms](https://doi.org/10.1007/BF00992696)：REINFORCE 的原始梯度估计思想。把 score function 与数据分布的关系读清楚。

- [Schulman et al. · Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)：区分 $\gamma$ 带来的目标/估计变化和 $\lambda$ 引入的优势估计权衡。

- [Schulman et al. · Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)：逐项对照 surrogate、clipping、训练轮数与采样协议。

- [Spinning Up · Policy optimization 推导与代码](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)：以轨迹概率、log derivative 和 baseline 连起数学与 PyTorch。

- [OpenAI Spinning Up · PPO PyTorch 作者教学实现](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py)：追 `buffer.finish_path`、`compute_loss_pi`、update 三部分；工程的旧环境 API 需单独适配。
