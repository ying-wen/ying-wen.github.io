# 分布强化学习：Bellman 分布、分位数与风险目标

现代深度强化学习 · 并列研究分支

学习完整回报分布，与学习均值、评估风险和估计知识不确定性分别有什么关系？

## 本章内容

- 写出分布 Bellman 递推及其固定策略条件。
- 推导 categorical 投影和 quantile loss。
- 区分回报分布、参数后验和风险偏好。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [深度价值学习：DQN、Double DQN 与目标的时间顺序](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/)：先掌握期望价值的Bellman目标，再学习回报分布。


### 条件概率与期望

区分随机变量、一次样本与条件分布；可以使用 Bayes 法则和全期望公式。

### MDP 与价值

理解状态、动作、转移、奖励、策略和 Bellman 递推；学习数据可以来自不同策略。

### 优化与梯度

知道固定目标、参数梯度、约束和期望目标的含义；神经网络只是函数表示的一种选择。

<a id="problem-definition"></a>

## 本章的问题定义

固定策略下，不仅预测回报均值，还预测整个回报分布，再明确控制时如何使用它。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 随机回报 $Z^\pi(s,a)$、有限支撑点或分位数表示；$\mathcal L(Z)$ 表示分布。

### 需要求解的对象

逼近回报分布；是否改用风险目标是另一个选择。

### 信息与数据权限

样本同时包含环境与策略随机性；不能仅由回报分散程度推断知识不足。

$$
Z^\pi(s,a)\overset D=R+\gamma Z^\pi(S^{\prime},A^{\prime}),\qquad A^{\prime}\sim\pi(\cdot\mid S^{\prime})
$$

等号表示分布相等；右侧按环境核和未来随机性联合生成。均值控制仍按 $\mathbb E[Z]$ 选动作，CVaR 等风险准则会改变问题。

### 成立条件与解的含义

- 分布的矩、固定支撑投影和距离需明确。
- 固定策略分布算子的收缩不能未经证明外推为所有分布控制算子的收缩。

判断准则：分别报告分布拟合、均值策略收益及所声明风险指标；不能相互替代。

### 适用边界

- 把 return variance 当作后验 epistemic uncertainty。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 推广：放宽条件 · [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)：本章把固定策略预测对象由期望扩为分布，均值是其统计量；控制准则仍可保持期望。

- 组合不同学习问题 · [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)：本章的分布预测需与另选的评价准则组合。默认均值控制并未改变目标；只有额外采用尾部风险准则时才改变排序，并需检查时间一致性。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

Bellman 备份后的分布未必仍落在有限参数表示中。

### 本章的核心思路

保留分布递推，再明确如何投影或用分位回归近似它。

1. [定义分布版备份](#lesson-notation)：先固定策略，区分随机变量递推与其均值递推。

2. [在离散支撑上重新分配概率质量](#lesson-derive)：Categorical 投影的边界与质量守恒是核心检查。

3. [改用分位数定位分布](#lesson-quantiles)：分位回归估计位置，不使用同一套概率质量参数。

结论与条件：固定策略下的理论、有限表示误差和控制改进是三层不同结论。

### 相关方法改变了什么

- C51 / quantile methods：分别固定值支撑并学习质量，或固定分位水平并学习位置。

- 均值 / 风险控制：前者改变预测工具，后者额外改变策略评价。


<a id="lesson-setting"></a>

## 1 · 同一均值可以来自不同的回报分布

上一章讨论为了获得有用信息而探索。本章先把模型完全给定，问一个更简单的问题：已经知道两种行动的平均回报相同，还可能关心它们的什么差别？接单者在起点 s 选择固定结算或浮动结算。两个动作都先收到首款 0.5，分别到达 f 或 v；下一步只能结算，随后真正终止。f 的末款恒为 2；v 的末款是 0 或 4，各占一半。折扣率为 0.5，收益单位沿两步相同。

$$
G_{\rm fixed}=0.5+0.5(2)=1.5,\qquad G_{\rm floating}=\begin{cases}0.5,&\text{概率 }1/2,\\2.5,&\text{概率 }1/2.\end{cases}
$$

两者均值都是 1.5；固定结算的回报方差是零，浮动结算的方差是一。这里概率是任务的给定量。

![两步接单任务的固定与浮动路径，真实回报柱高不同但均值都为1.5](https://yingwen.io/crl-figures/distributional-walkthrough-task.svg)

上方沿动作箭头读两次奖励；双圆 T 表示真正终止，之后回报为零。下方柱高表示真实概率质量，橙虚线标出同一均值。先只看均值，能否判断哪种接单方式更容易出现低回报？本图及后续两图是原创有限任务精算；[计算核](/crl-code/figures/distributional-walkthrough.mjs)与[独立教程](/crl-code/tutorials/distributional-walkthrough.py)可重算全部数字。

$$
Z^\pi(s,a)\overset{D}{=}\sum_{t=0}^\infty\gamma^tR_{t+1}\mid S_0=s,A_0=a,\qquad Q^\pi(s,a)=\mathbb E Z^\pi(s,a)
$$

分布来自环境、策略及初始条件下的随机轨迹。等分布符号不表示两个实际采样回报逐点相同。

回报分布回答“沿这条行为规则，可能得到哪些总收益、各有多大概率”。价值 Q 只保留其中的均值。若控制器仍按 Q 选动作，接单例的两个动作并列；本页约定并列时选固定结算。预测目标更丰富，并没有替控制器选定风险偏好。本章沿 categorical 表示展开一个完整更新，再把分位数作为另一种表示来比较。

环境已完全已知时回报仍可随机，因此回报分布的宽度不等于模型的认识不确定性。外部设计者选择分布表示、支持范围以及决策采用均值还是风险泛函；运行时 agent 学习的是这一约定下的预测。

<a id="lesson-notation"></a>

## 2 · 条件分布的 Bellman 递推

回报恒等式 $G_t=R_{t+1}+\gamma G_{t+1}$ 先给出一条轨迹上的分解。在接单例中，结算状态 f 的回报分布是 $\delta_2$，v 的分布是 $\tfrac12\delta_0+\tfrac12\delta_4$。把末款乘以 0.5、再加首款 0.5，便得到起点的两条分布。这里 $\delta_x$ 表示在 x 上的全部概率质量，终止后的分布是 $\delta_0$。

$$
(\mathcal T^\pi Z)(s,a)\overset{D}{=}R+\gamma Z(S',A'),\quad (R,S')\sim P(\cdot,\cdot\mid s,a),\quad A'\sim\pi(\cdot\mid S')
$$

先采样即时奖励与后继状态，再按策略采样后继动作，最后按相应后继分布采样回报。奖励与后继状态的相关性需要保留。

这条递推对随机后继状态先混合相应分布，再作缩放与平移。它保留奖励和后继状态的联合规律；若高奖励只在特定后继状态出现，分别抽两个独立的边缘分布就改变了任务。在接单例中首款和所到结算状态都是动作确定的，随机性只发生在 v 的末款，因此可直接对两种末款枚举。时间与终止约定承接 Sutton–Barto §3.3–3.5，分布递推见[作者教材第 2 章](https://www.distributional-rl.org/contents/chapter2)。

对两边取期望即可恢复通常的 Bellman 方程。这是均值方法嵌入分布方法的方式。分布递推还保留高阶结构；把每个状态动作只存成一个均值，会在这一步丢失结构。

$$
\bar W_p(\mathcal T^\pi Z_1,\mathcal T^\pi Z_2)\le\gamma\bar W_p(Z_1,Z_2),\qquad\bar W_p=\sup_{s,a}W_p
$$

当 0≤γ<1、相应最大 p 阶矩有限且保持同一固定策略时，分布算子在最大 Wasserstein 距离中压缩。控制时策略随估计变化，不能直接照搬这个结论。

压缩系数从哪里来？让两个备份使用同一组 $(R,S',A')$，并在每个后继状态动作上把两条回报分布尽可能靠近地配对，记配对值为 $X_1,X_2$。即时奖励相消，距离只剩 $\gamma|X_1-X_2|$。对后继随机性取 $p$ 阶平均再开根号，不超过 $\gamma\sup_{s',a'}W_p(Z_1,Z_2)$。Wasserstein 距离对所有配对取下确界，因此此配对给出上界。若策略分别由两个估计 greedy 选出，便无法保证共用同一个 $A'$，这个证明的关键步骤就失效了。

标准 C51 或 QR-DQN 通常用预测分布的均值选择贪心动作，再备份被选动作的整条分布。最优期望价值可以唯一，但最优策略的回报分布不必唯一；这正是控制分析与固定策略预测不同的原因之一。

<a id="lesson-derive"></a>

## 3 · Categorical 表示为什么需要投影

为了沿同一算例看清投影，当前与后继的分布都用支持 (0,1,2,3,4)，共五个原子。后继 f 的快照为 (0,0,1,0,0)，后继 v 为 (0.5,0,0,0,0.5)。这两份快照在此处给定且准确；实际网络需要从末步经验学习它们。五原子只缩小表示规模，投影规则沿用 C51；C51 的名称来自原实现使用 51 个支持点。

$$
Z_\theta(s,a)=\sum_{i=0}^{N-1}p_{\theta,i}(s,a)\,\delta_{z_i},\quad z_i=v_{\min}+i\Delta z,\quad\sum_i p_{\theta,i}=1
$$

位置固定、概率可学习；这里的 Dirac 是点质量，不是 TD error。网络可用 softmax 保证归一化。

$$
\tilde z_j=\operatorname{clip}(r+\gamma(1-d)z_j,v_{\min},v_{\max}),\quad b_j=\frac{\tilde z_j-v_{\min}}{\Delta z}
$$

这里 $d=1$ 表示本次转移后真正终止，$d=0$ 表示非终止；网格间距为 $\Delta z=(v_{\max}-v_{\min})/(N-1)$。自举后的点通常不落在原网格上，因此要把概率质量投影回固定支持。

令 $\ell_j=\lfloor b_j\rfloor,u_j=\lceil b_j\rceil$。若二者不同，向下格点加 $p_j(u_j-b_j)$，向上格点加 $p_j(b_j-\ell_j)$。若相同，全部质量都加到该格点；直接用两个插值权重会把整数点的质量错误清零。

浮动动作的本次转移是 $(s,\text{浮动},0.5,v,d=0)$。后继 0 的 0.5 质量移到 0.5，再向格点 0 和 1 各分 0.25；后继 4 的 0.5 质量移到 2.5，再向 2 和 3 各分 0.25。因此 $m_{\rm floating}=(0.25,0.25,0.25,0.25,0)$。固定动作的后继 2 移到 1.5，所以 $m_{\rm fixed}=(0,0.5,0.5,0,0)$。

$$
z_\ell(u-b)+z_u(b-\ell)=\tilde z\big[(u-b)+(b-\ell)\big]=\tilde z
$$

相邻格点距离为 Δz 且 u−ℓ=1 时，插值的权重和为一，加权位置仍是裁剪后的点。整数点则直接保留全部质量。逐点求和得到投影均值等于裁剪后的目标均值。

![后继点质量经Bellman变换到半整数，再分配给邻居格点；支持溢出时5被裁为4](https://yingwen.io/crl-figures/distributional-walkthrough-projection.svg)

上部追踪后继 v 的两份 0.5 质量，投影结果为四个 0.25，均值 1.5 保持。下部只把同一任务的首款改为 3：原目标 3/5 各半，5 的质量全部裁到 4，均值由 4 降到 3.5，总质量仍为一。横轴额外显示 5，是为了看见支持外的目标。先预测：归一化正确能否排除这个偏差？投影规则依据 [C51 §4.2、式 (7)](https://proceedings.mlr.press/v70/bellemare17a/bellemare17a.pdf#page=6)。

![Categorical Bellman 投影中，三个原子的平移缩放与概率质量向邻居格点分配](https://yingwen.io/crl-figures/concept-depth-classic-distribution-projection.svg)

先改变位置，再分配质量。颜色跟踪同一份概率，柱高表示概率大小；中间的 0.5 质量分成 0.25 与 0.25，最后得到 (0,0.45,0.55)。本例没有支持裁剪，均值保持 0.55。这里用三个原子展开 C51 所用投影，不是 51 原子的训练结果。原创精算，依据 [Bellemare 等 §4.2、式 (7)](https://proceedings.mlr.press/v70/bellemare17a/bellemare17a.pdf#page=6)；[计算代码](/crl-code/figures/classic-visual-depth.mjs)。

$$
L_{\rm cat}(\theta)=-\sum_i m_i\log p_{\theta,i}(s,a)
$$

投影概率 m 在当前反传中固定。该 cross-entropy 对预测概率求导，不是对采样奖励或后继 argmax 求导。

支持外的回报被裁剪，可能改变均值。即便投影步骤数值正确，过窄支持也会造成系统误差；长生命期的奖励尺度变化尤其需要检查这一点。

三个边界需要分别处理。原子恰在整数格点，例如上述首款为 3 的低结果，全部质量留在该点；真终止时令 d=1，仍保留本次奖励，所有目标原子合为奖励所在位置；终止奖励为 2 时目标正是 $\delta_2$。若终止奖励为 1.5，五原子表示仍需把这一个真实点投影到 1 和 2。这里表示产生的展宽与环境本来的随机性也不同。

<a id="experiment-deep-c51"></a>

### 实验：学习回报分布，与按均值选动作是两件事

C51 的类别分布进入了目标和损失，但最终贪心控制是否因此更好？

**环境与可用信息。** 位置 0–4 的 DeadlineChain，动作是左／右；观测是位置 one-hot 和剩余时间比例。到位置 4 得 1 并终止，其余每步 −0.02；12 步期限也是问题的真实终止。每回合从 0 开始。

**设置。** 本图实际运行 1200 个预算单位，训练种子为 0–4。预算为真实转移。使用 32 隐单元网络、51 个均匀原子，支持集 [−1,1]；折扣 0.99，Adam 0.003，回放 4000，batch 32；每两步优化一次，每 100 个真实步复制目标网络。探索率从 1 线性降到 0.05。

**检验的机制。** 把目标原子经 Bellman 变换后投影回固定支持；正好落在原子上的概率质量完整保留。训练用交叉熵，动作仍按分布期望选择。

**测量。** 主图是冻结贪心策略的原始回报，不是分布校准误差、分位数误差或风险度量。支持集、质量守恒与终止投影还需数值测试。

```bash
python3 implementations/deep/c51.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-figures/result-deep-c51.svg)

横轴：训练环境步（独立评估交互另计）。纵轴：冻结策略的外部回报。每种方法 1200 训练环境步；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 冻结当前网络，按预测分布的均值选贪心动作，在独立 DeadlineChain 中评价12次，平均未折扣外部回报。每次都从位置0、剩余12步开始；到位置4奖励1，其余步奖励−.02。相同网络的12次确定性评价不是12个训练种子。这里没有按分位数或尾部风险选动作。

**step：怎样计时。** step 只数真实训练转移。第32步首次更新，此后每两个环境步从回放中无放回取32条经验，做一次梯度更新；每100步复制目标网络。1200步对应585次梯度更新。两算法与 DQN 的更新时钟相同，但网络输出数、损失计算及梯度裁剪设置不同，相同交互预算不等于相同计算预算。

**怎样汇总。** 取该记录时刻的值；不先对曲线上的时间点求平均。 每个运行有独立初始化和探索随机性；先在运行内完成12次冻结评价，再跨训练种子汇总。训练目标的 γ=.99，图中回报不折扣；12步期限是任务终止，不是训练预算截断。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** CSV 保存冻结回报、样本数和梯度更新数；loss 是最近一次批次损失，不能在 C51、QR-DQN 与 DQN 间直接比较数值。日志没有保存完整预测分布或逐步行为奖励，所以不能由这张图重算分位数校准、Wasserstein 误差、风险表现或训练累计收益。两步订单的解析算例与本神经控制任务不同。

计算位置：[deep/c51.py](https://yingwen.io/crl-code/implementations/deep/c51.py) · [deep/qr_dqn.py](https://yingwen.io/crl-code/implementations/deep/qr_dqn.py) · [deep/dqn.py](https://yingwen.io/crl-code/implementations/deep/dqn.py) · [deep/_common.py](https://yingwen.io/crl-code/implementations/deep/_common.py)

</details>

**结果分析。** 600 步两者都达到约 0.94；1200 步 C51 为 0.94，DQN 为 0.936。任务接近饱和，这个微小末点差异不建立分布式方法的性能优势。

**结论边界。** 环境大部分后果是确定性的，且评价只看期望控制结果。本图没有检验风险敏感决策，也没有复现 Atari 规模的 C51。

**继续实验。** 增加已知随机奖励，先验证预测分布能否校准；然后保持风险准则不变比较控制。若把 argmax 期望改成低分位数，必须另行声明控制目标。

[源码](https://yingwen.io/crl-code/implementations/deep/c51.py) · [逐种子记录](https://yingwen.io/crl-code/results/deep-c51/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/deep-c51/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/deep-c51/curves.json)

<a id="lesson-categorical-sample"></a>

### 从一条转移到一次概率更新

现在只更新起点浮动动作的预测。冻结后继 v 的概率快照，当前五个 logits 全为零，softmax 给出每点 0.2。本次非终止转移的目标是上一节的 m；它已经通过后继预测包含两种未来末款，并非本次观察到了两个未来结果。若更新的是结算状态 v，一条真正终止的经验只提供本次末款 0 或 4，对应点目标 $\delta_0$ 或 $\delta_4$；两类经验的总体平均才是 v 的真实二点分布。

$$
p_i=\frac{e^{h_i}}{\sum_k e^{h_k}},\qquad \frac{\partial L_{\rm cat}}{\partial h_i}=p_i-m_i,\qquad h_i^+=h_i-\alpha(p_i-m_i)
$$

h 是当前动作的 logits；m 和后继快照在本次求导中固定。这里把 logits 当作独立表参数，以隔离分类更新；神经实现还需乘网络 Jacobian。

取步长 $\alpha=1$，梯度为 $(-0.05,-0.05,-0.05,-0.05,0.2)$，新 logits 为 $(0.05,0.05,0.05,0.05,-0.2)$。再次 softmax 得到约 $(0.209,0.209,0.209,0.209,0.163)$：前四点质量上升，错误的端点 4 质量下降。交叉熵由 1.609 降到约 1.564；一次下降还没有得到四个 0.25 的目标。后继值和目标 m 都没有被这次反传改写。

图中的上半部显示这一步，后面的行动比较使用已知任务的真实分布。两种计算承担不同角色：这一步解释样本如何写入概率；真实分布则给读者检查动作准则的参照。

<a id="lesson-quantiles"></a>

## 4 · 分位数回归与均值决策

固定支持需要搬运质量；另一种做法是固定每个点的质量 $1/N$，让位置随数据移动。仍看接单任务：固定回报只有 1.5，浮动回报在 0.5 与 2.5 各半。设 $F_Z(z)=\Pr(Z\le z)$，它给定一个收益门槛，回答不超过门槛的概率；分位函数反过来给定概率水平，回答对应的收益位置。

$$
F_Z^{-1}(\tau)=\inf\{z:F_Z(z)\ge\tau\},\quad 0<\tau\le1,\qquad \tau_i=\frac{i+\tfrac12}{N},\quad i=0,\ldots,N-1
$$

每个等宽概率区间用中点 midpoint 作分位目标；τ 是概率坐标，θ 是收益坐标，二者单位不同。

$$
Z_\theta(s,a)=\frac1N\sum_{i=0}^{N-1}\delta_{\theta_i(s,a)},\qquad W_1(Z,Z_\theta)=\sum_{i=0}^{N-1}\int_{i/N}^{(i+1)/N}|F_Z^{-1}(u)-\theta_i|\,du
$$

右式按位置非降序排列后成立。每一段对 θ 的最小绝对偏差解是该概率区间的中位位置，因此可取真实分位函数在区间中点的值；有跳跃时最优解可能是区间。

取 $N=4$，四个水平为 $(1/8,3/8,5/8,7/8)$。固定动作的真实分位位置都是 1.5；浮动动作是 $(0.5,0.5,2.5,2.5)$。两个位置重合就合并质量，因此浮动的四个等质量点恰好表示原来的二点分布。$F^{-1}(0.5)=0.5$ 来自上面的下确界约定；分位水平为 0.5 的 pinball 最优位置则可以是整个 $[0.5,2.5]$，不能把一个约定值当成唯一最优解。这里不需要分位函数在零点的端点值。

![接单回报的CDF与逆CDF台阶，四个等宽概率区间中点映到两个真实回报位置](https://yingwen.io/crl-figures/quantile-walkthrough-cdf.svg)

沿蓝色台阶将收益门槛与概率水平交换：CDF 在 0.5 和 2.5 跳跃，逆 CDF 在概率 0.5 两侧换位置；实心与空心点区分包含的端点。紫点是四个 midpoint；每点质量 1/4，重合后各为 1/2。青虚线是固定动作的 1.5。图显示已知任务的真实分位参照；后图才更新参数。原创精算，[JS 核](/crl-code/figures/quantile-walkthrough.mjs)、[独立 Python](/crl-code/tutorials/quantile-walkthrough.py)重算本节全部数值；等质量表示与 midpoint 依据 [QR 原文式 (7)、Lemma 2](https://arxiv.org/pdf/1710.10044#page=4)。

学习器不知道真实分位位置，只有目标样本 Y。令残差 $u=Y-\theta$，pinball 对低估使用权重 τ、对高估使用权重 $1-\tau$。若 $Y>\theta$，损失是 $\tau(Y-\theta)$，对 θ 的导数为 $-\tau$；若 $Y<\theta$，损失是 $(1-\tau)(\theta-Y)$，导数为 $1-\tau$。高水平的分位点因而受到更大的向上拉力。

$$
\rho_\tau(u)=u\bigl(\tau-\mathbf1[u<0]\bigr),\qquad\frac{d}{d\theta}\mathbb E\rho_\tau(Y-\theta)=F_Y(\theta)-\tau
$$

此导数式用于分布在 θ 处无原子的情况。有原子时次梯度区间为 [F(θ⁻)−τ,F(θ)−τ]；区间包含零才是最优条件，最优分位位置也可能不唯一。

这也给出为何样本更新有效的理由：除相等点外，样本次梯度 $\mathbf1[Y<\theta]-\tau$ 的期望是相应 CDF 与 τ 的差，不需要先把整个样本分布构造出来再求 Wasserstein 梯度。在浮动真分布的低点 0.5、τ=1/8 处，次梯度区间是 $[-1/8,3/8]$，包含零。到中间 1.5 时没有原子，导数为 $1/2-1/8=3/8$，所以应向下移动。

$$
L_{\rm QR}=\frac1{N^2}\sum_{i,j}\rho_{\tau_i}(r+\gamma(1-d)\theta_j^-(s',a^*)-\theta_i(s,a))
$$

每个当前分位数与所有目标分位数配对。本页数值核使用 pinball loss；QR-DQN 原文实践使用平滑的 quantile Huber 变体。

上标 − 是本次冻结的后继预测，$d=1$ 才去掉未来自举。先选择一个后继动作，再取这个动作的整条预测分布；不能对每个 j 分别挑最高的位置，那样会把不同动作拼成一个不存在的分布。QR 原文式 (13) 以同一后继分布的均值取 greedy，算法 1 没有区分网络副本上标。为写清计算顺序，本节采用由冻结副本的均值选择 $a^*=\arg\max_b N^{-1}\sum_j\theta_j^-(s',b)$、并由该副本供目标的约定。若改用在线网选动作、冻结网供目标，应明确标为 Double DQN 的选择/评价分工。

一条环境转移确定 r、s′、d，所有目标位置由该 s′ 的冻结预测给出；配对的两个索引分别表示当前分位水平和目标回报位置。它们没有要求再向环境索取同一个转移的第二个独立后继。本页接单任务的结算状态只有一个动作，因而后继 argmax 没有竞争；起点的两动作参数彼此独立，以下只写入浮动动作。

<a id="course-distribution-gradient"></a>

## 4.1 · 回报表示也改变梯度尺度，而不只是预测内容

设 categorical 网络输出 logits $\ell_i$，概率 $p_i=\operatorname{softmax}(\ell)_i$，目标分布为固定的 $m_i$。交叉熵 $L=-\sum_i m_i\log p_i$ 的梯度可以逐项展开。

$$
\frac{\partial L}{\partial\ell_i}=p_i-m_i\in[-1,1],\qquad
\nabla_\theta L=\sum_i(p_i-m_i)\nabla_\theta\ell_i
$$

目标总质量为一时，输出 logits 上的梯度有界。参数梯度还乘网络 Jacobian，所以不能据此声称参数更新或 TD 自举整体有界。

对照标量平方误差：预测为零、目标为一百时，标量输出梯度为 −100。两原子分类预测为 (0.9,0.1)，目标全部落在第二个原子时，logit 梯度是 (0.9,−0.9)。这里改变了误差的参数化与尺度，并不能把两个数字直接当成谁学习得更好的证据。支持裁剪还可能把一百误当成上限十。

因此，比较标量 TD、Huber TD 与分布 TD 时，要同时控制回报支持、输出参数化和优化器。均值预测更准确、单步梯度不易被异常值主导、以及长期学习更稳定，是三个需要分别测量的命题。

流式学习每次只收到一个新样本，这些尺度差异会更直接地影响更新。一个值得检验的问题是：收益变化后，固定支持造成的偏差与优化器历史尺度造成的滞后分别有多大？记录落在支持端点的目标质量，并与适当支持范围、标量 Huber 和相同数据流的对照比较。

<a id="lesson-quantile-sample"></a>

### 从冻结后继到一次位置更新

冻结 v 的四点快照为 $(0,0,4,4)$，每点质量 1/4；这份准确快照仍是给定量。起点浮动动作的当前表参数取 $\theta=(1,1,2,2)$。同一非终止转移 $(s,\text{浮动},0.5,v,d=0)$ 把后继变成 $Y=(0.5,0.5,2.5,2.5)$。十六个配对只是对冻结后继分布的有限和，一次环境转移没有观察到四个未来末款。若只从这份后继预测抽到低结果，目标仅为 0.5，四个位置的一步会不同；两类目标按冻结后继分布的概率取总体平均，才得到这里的平均更新。本例准确快照的概率恰与真实任务一致。

$$
g_i=\frac1N\sum_j\bigl(\mathbf1[Y_j<\theta_i]-\tau_i\bigr),\quad \frac{\partial L_{\rm QR}}{\partial\theta_i}=\frac{g_i}{N},\quad\theta_i^+=\theta_i-\alpha\frac{g_i}{N}
$$

本次没有零残差。零残差的单样本次梯度可取 [−τ,1−τ] 中任意值；可执行核明确取零。

四个当前位置都在两个目标值之间，各有一半目标低于它，所以 $g=(3/8,1/8,-1/8,-3/8)$。取 $\alpha=4$，正好消掉本页损失对四个当前头取平均的因子，得到 $\theta^+=(0.625,0.875,2.125,2.375)$。损失从 0.375 降到 0.296875。原论文对当前头求和、对目标头求平均，本页额外除以 N；这个步长只说明独立表参数下的尺度换算，神经网络还要乘共享 Jacobian，不能据此照抄优化器步长。

![十六个当前与目标残差配对产生四个平均次梯度，更新前后四个等质量原子位置沿同一回报轴移动](https://yingwen.io/crl-figures/quantile-walkthrough-update.svg)

矩阵的行标为 τ，列标为目标 Y；红色负残差表示当前位置高于目标，蓝色正残差表示低于目标。下方箭头跟踪同一参数，竖直错开只显示重合点，每个标记质量仍为 1/4。一步后均值还是 1.5，下尾均值已由 1 降到 0.75；目标和后继快照都冻结。先预测：均值准确是否足以说明四点位置也准确？配对更新依据 [QR 式 (12) 后的讨论与算法 1](https://arxiv.org/pdf/1710.10044#page=5)，数字由本页精算核生成。

保持相同准确目标，指定第 k 次表参数步长 $\alpha_k=4/\sqrt{k}$，每次计算精确总体期望，执行到 32 次。第 4 次位置约为 $(0.495,0.652,2.348,2.505)$，第 32 次约为 $(0.479,0.496,2.504,2.521)$。后者仍在真位置两侧，$W_1$ 约为 0.0123，不能把有限更新当作已经学到真分布或收敛证明。这里没有随机训练；准确后继、独立表参数与总体平均隔离了写入机制，实际环境还需估计后继并受到样本噪声影响。

<a id="lesson-quantile-huber"></a>

### 平滑残差会改变哪些位置

Pinball 在零残差处有折角，QR-DQN 的实际网络损失在附近换成二次曲线。采用 IQN 原文的归一化形式，对 $\kappa>0$ 定义 Huber 函数和分位损失：

$$
H_\kappa(u)=\begin{cases}\tfrac12u^2,&|u|\le\kappa,\\\kappa(|u|-\tfrac12\kappa),&|u|>\kappa,\end{cases}\qquad \rho_\tau^\kappa(u)=|\tau-\mathbf1[u<0]|\frac{H_\kappa(u)}\kappa
$$

QR 的 arXiv v1 式 (10) 没有除以 κ，IQN §2.3 的定义包含这一因子；固定正 κ 时二者相差一个损失尺度，κ=1 时相同。

$$
\frac{\partial\rho_\tau^\kappa(Y-\theta)}{\partial\theta}=-|\tau-\mathbf1[Y<\theta]|\operatorname{clip}\!\left(\frac{Y-\theta}\kappa,-1,1\right)
$$

零点导数为零，小残差的拉力随距离减弱。κ趋于零时，归一化形式在非零残差处趋向 pinball。

同一初值、同一目标，取 κ=1、α=4，Huber 一步得到 $(0.84375,1.03125,1.96875,2.15625)$。其中 τ=3/8 的位置反而由 1 升到 1.03125：低目标距离 0.5 的拉力被缩小，高目标距离 1.5 的拉力已经饱和。平滑没有改变目标样本，却改变了每个样本如何拉动参数。

甚至准确的低分位位置 0.5 在 Huber 下也未必最优。τ=1/8、θ=0.5 时，低目标残差为零，高目标提供平均导数 $-\tfrac12(1/8)=-1/16$，会向上移动。在 $0.5<\theta<1.5$ 中令导数为零，得 $\tfrac12(1-\tau)(\theta-0.5)-\tfrac12\tau=0$，故该最优位置为 $0.5+\tau/(1-\tau)$。本例四个 Huber 最优位置约为 $(0.643,1.1,1.9,2.357)$，均值仍为 1.5，下尾均值约 0.871；它们不等于真实 midpoint 分位数。这是固定阈值与离散原子的具体偏移，不能将 pinball 的精确分位解释直接赋给平滑损失。

<a id="lesson-quantile-iqn"></a>

### 让概率水平进入网络

QR 的四个头固定服务于四个 midpoint。IQN 改为网络 $f_\omega(s,a,\tau)$ 接收分位水平，用同一函数回答不同 τ。训练抽一组当前水平 $\tau_i$、一组独立目标水平 $\tau'_j$；动作选择还用独立的 $\tilde\tau_k$。三组都从均匀分布抽取时，当前损失、目标回报和均值行动各有自己的采样近似。

$$
Y_j=\operatorname{stopgrad}\!\left[r+\gamma(1-d)f_{\omega^-}(s',a^*,\tau'_j)\right],\qquad L_{\rm IQN}=\frac1{MM'}\sum_{i=1}^{M}\sum_{j=1}^{M'}\rho_{\tau_i}^{\kappa}\!\left(Y_j-f_\omega(s,a,\tau_i)\right)
$$

M、M′ 分别是当前与目标水平的样本数。IQN 原文式 (3) 只对目标组取平均，本页再对当前组取平均；仍是尺度约定。所有配对共享这条环境转移和冻结目标，因此配对项不是 MM′ 条相互独立的环境经验。

为了手算，本页指定合法水平：当前 $(0.15,0.7)$，目标 $(0.2,0.4,0.9)$。准确后继 v 给目标回报 $(0.5,0.5,2.5)$，六个配对将前两个目标拉向下部、第三个拉向上部。指定这组水平没有执行随机抽样；它也不等于把高结果概率改成 1/3。均匀 τ 的总体规律仍给每个真实结果各半；有限目标组有采样误差。

$$
\widehat Q(s,a)=\frac1K\sum_{k=1}^{K}f_\omega(s,a,\tilde\tau_k),\qquad \widehat Q_{\rm lower}(s,a)=\frac1K\sum_{k=1}^{K}f_\omega(s,a,\eta\tilde\tau_k)
$$

对均匀水平，准确分位函数的样本均值期望等于均值；将水平缩到 [0,η]，才读取收益下尾均值。有限 K 仍有误差，学习函数自身也可能不准确。

动作选择先对所有候选后继动作计算同一准则的读数，再取一个 $a^*$；随后目标 $Y_j$ 全部来自这个动作。若沿用本节的冻结网选择/评价约定，将上式的 $f_\omega$ 换为后继冻结副本 $f_{\omega^-}$ 再选动作；若使用在线网选择，需单独声明。训练的 $\tau_i,\tau'_j$ 仍覆盖均匀概率轴，以预测所选行为的整条回报分布；把行动读出的水平改为 $\eta\tilde\tau_k$ 并不把目标分布也截成下尾。它通过后继选择改变未来行为，与改变预测对象的权重是两项操作。

QR-DQN 的平均分位点近似均值，用于常规贪心选择。IQN 把分位水平也作为网络输入，可以采样不同水平并改变分位权重。改变权重为风险敏感控制时，已经改变决策准则，不能把这种变化只归因为预测更准确。三组独立水平及训练/行动公式见 [IQN §3、式 (2)–(4)](https://proceedings.mlr.press/v80/dabney18a/dabney18a.pdf#page=5)。

如果计算预算只容许少数点，还可让概率区间的边界一起学习。FQF 用累计 softmax 构造 $0=b_0<b_1<\cdots<b_N=1$，区间中点 $\hat b_i=(b_i+b_{i+1})/2$，位置由分位网络给出，质量是 $b_{i+1}-b_i$。因此均值读出必须是 $\sum_i(b_{i+1}-b_i)f(\hat b_i)$，不再是四点简单平均。原文对连续、非降分位函数给边界导数 $2f(b_i)-f(\hat b_{i-1})-f(\hat b_i)$；实际用学习函数估计并分别更新边界提议网与位置网。接单例有跳跃和原子，不能直接把这个连续条件的公式当作这里的光滑真值导数。参见 [FQF §3.1–3.4、Proposition 1 与算法 1](https://proceedings.neurips.cc/paper/2019/file/f471223d1a1614b58a7dc45c9d01df19-Paper.pdf#page=4)。

<a id="lesson-neural-shared"></a>

### 共享网络：一个输入的下降怎样影响另一个输入

前面的表格计算让每个分位位置独立移动。换成神经网络后，位置仍是预测对象，存储位置的参数却会同时服务于多个输入。保持接单任务不变：固定动作真回报是 1.5，浮动动作真回报是 0.5/2.5 各半。将两个起点动作编码为 $x_f=-1,x_v=+1$，用一个 tanh 隐单元和四个线性输出头预测四个 midpoint：

$$
h_\omega(x)=\tanh(ax+c),\qquad q_{\omega,i}(x)=b_i+w_i h_\omega(x),\quad i=0,1,2,3
$$

参数 $\omega=(a,c,w_0,\ldots,w_3,b_0,\ldots,b_3)$ 共十个。输入同时指明状态动作；这两点例只比较起点的两动作。四个头仍各代表 1/4 的质量，没有用 softmax 改质量。

取 $a=c=0.5$、全部 $b_i=1.5$，以及 $w=(-0.5,-0.5,0.5,0.5)/\tanh(1)$。于是 $h(x_f)=0,h(x_v)=\tanh(1)$：固定预测恰为 $(1.5,1.5,1.5,1.5)$，浮动预测是 $(1,1,2,2)$。若只把 w 改为 $(-1,-1,1,1)/\tanh(1)$，同一网络即可同时精确表示两条真分布。因此接下来的固定动作误差不能归因于这两点上的容量不足。

![一个tanh隐单元和四个输出头共享十个参数，两动作的激活不同，初始固定分布准确而浮动分布尚窄](https://yingwen.io/crl-figures/neural-distribution-walkthrough-network.svg)

上方箭头表示前向计算，曲线给出同一个隐单元在两个输入上的值；下方点均为 1/4 质量，青虚线为真位置。竖直错开仅显示重合点。这个两输入网络容量足以表示两真分布。[JS 计算](/crl-code/figures/neural-distribution-walkthrough.mjs)与[独立标准库 Python](/crl-code/tutorials/neural-distribution-walkthrough.py)生成本节原创数值。共享参数的损失反传承接 [QR 式 (12) 与算法 1](https://arxiv.org/pdf/1710.10044#page=5)；此图使用小型 tanh 网络来隔离参数共享。

先把目标固定为真实四点 $Y=(0.5,0.5,2.5,2.5)$，只给浮动输入写入。这是读者能精确检查的受控监督诊断：正常 TD 只取得实际转移并读取后继估计，并不收到环境的真分布标签。这里只暂时去掉后继误差和抽样噪声，检查共享网络怎样传递一项更新。损失仍为 $L_v(\omega)=16^{-1}\sum_{i,j}\rho_{\tau_i}(Y_j-q_{\omega,i}(x_v))$。

$$
d_i=\frac{\partial L_v}{\partial q_i},\quad \frac{\partial L_v}{\partial b_i}=d_i,\quad\frac{\partial L_v}{\partial w_i}=d_i h_v,\quad \frac{\partial L_v}{\partial c}=(1-h_v^2)\sum_i d_iw_i,\quad\frac{\partial L_v}{\partial a}=x_v\frac{\partial L_v}{\partial c}
$$

链式法则将位置梯度变成参数梯度。本次反传把 Y 固定；不能把表参数的步长直接移给网络。四次算例采用 α=0.5，是为了跟踪可读的有限写入。

初始位置梯度是 $d=(3/32,1/32,-1/32,-3/32)$，隐层的两个梯度均约为 $-0.06893$。所有参数同时做 $\omega^+=\omega-0.5\nabla L_v$，得到 $a^+=c^+\approx0.534465$。因为二者仍相同，固定输入的隐激活保持零，但共享偏置已经变成 $(1.453125,1.484375,1.515625,1.546875)$；这就是它更新后的四点预测。固定动作没有新经验，原先准确的分布也已展开。浮动预测则变为约 $(0.907,0.957,2.043,2.093)$，训练损失从 0.375 降到约 0.355。

$$
\Delta q_\omega(x_f)=-\alpha J_fJ_v^{\mathsf T}d+O(\alpha^2),\qquad J_x=\frac{\partial q_\omega(x)}{\partial\omega}
$$

这是有限步长的局部展开，$J_f$ 与 $J_v$ 的内积描述另一输入的预测敏感度，符号和大小由当前网络决定。本例初始 $J_fJ_v^{\mathsf T}$ 恰为四阶单位矩阵：偏置路径贡献单位阵，a 与 c 的交叉路径相消，固定输入的 w 路径为零。

继续对同一个冻结目标写入，共四次，浮动预测约为 $(0.628,0.838,2.162,2.372)$，$W_1$ 从 0.5 降到约 0.233，损失降到约 0.295。固定预测却成了 $(1.3125,1.4375,1.5625,1.6875)$，$W_1$ 从零升到 0.125。两动作的均值一直等于 1.5，因此按均值选动作始终并列；均值正确没有揭示分布的改写。固定动作的下尾均值由 1.5 变成 1.375，也说明这种改写会影响其他读出。

容量足够也不等于有限次优化已经到达解。本例浮动真位置的总体 pinball 损失是 0.25，而四次后的损失约为 0.295：仍有可改善的固定目标误差。真位置处的损失也不为零，因为所有当前头都要与两类目标配对；配对损失的数值和分布距离不是同一个量。

![四次浮动输入写入使浮动四头向真端点展开，也使固定四头从准确一点展开；同轴W1一升一降而均值不变](https://yingwen.io/crl-figures/neural-distribution-walkthrough-updates.svg)

上部逐头画出 k=0,…,4 的实际计算，实线与虚线区分头，青水平线标真位置；下部使用同一个 $W_1$ 纵轴比较两动作。连接点帮助追踪离散写入，不是补出的训练曲线。真目标、输入和 α 均固定；四次总体梯度计算解释本网络的干扰，不给出随机训练性能或普遍下降保证。

容量限制可以另外构造：若把两个动作都编码为同一个输入，则任何确定网络只能输出同一分布 $\widehat Z$。两真分布的 $W_1$ 距离是一，由三角不等式，$W_1(\widehat Z,Z_f)+W_1(\widehat Z,Z_v)\ge1$；至少一个误差不小于 0.5。增加同一输入后的网络宽度仍不能分辨动作。这项输入合并造成的不可表示，与容量足够时一次写入干扰另一预测，是两个困难。

<a id="lesson-neural-targets"></a>

### 固定目标的优化，与目标本身的变化

上例只优化一个固定的 $L_v$。在真正的分布 TD 中，结算状态 v 的分布也需学习，父节点的目标随它改变。继续相同任务，另建一个同结构的后继网络：输入 −1 表示结算 f、+1 表示结算 v，初始 f 四头都是 2，v 为 $(1,1,3,3)$；此时参数为 $a=c=0.5,b_i=2,w=(-1,-1,1,1)/\tanh(1)$。v 的真实终止奖励是 0 或 4，目标不带未来尾值。这里只对这两种终止标签按各半概率求一次总体梯度，不调用随机训练。

$$
Y_j^{(k)}=\operatorname{stopgrad}\!\left[0.5+0.5q_{\nu^{(k)},j}(v)\right],\qquad L^{(k)}(\omega)=\frac1{16}\sum_{i,j}\rho_{\tau_i}\!\left(Y_j^{(k)}-q_{\omega,i}(x_v)\right)
$$

ν 是结算网络参数，ω 是父预测参数；本次父求导固定 ν 的版本。读取旧 target 副本时 Y 保持旧值；同步副本后才得到新 Y。停止梯度规定本次求导路径，没有规定下一次目标必须相同。

后继网络对 $(0,0,4,4)$ 做一次 α=0.5 的写入，v 预测变成约 $(0.856,0.906,3.094,3.144)$。父目标因而从 $(1,1,2,2)$ 移到约 $(0.928,0.953,2.047,2.072)$。现在将父预测始终保持 $(1,1,2,2)$：目标离真实父分布的 $W_1$ 从 0.5 降到约 0.440，但父对新标签的原始 pinball 损失反而由 0.125 升到约 0.155。父参数没有退步；标签分布及配对损失的基线已经改变。比较跨版本训练损失时，需要同时记录目标和固定真值参照。

![结算网络一次更新改变父目标CDF；父预测冻结时，新目标更接近真值但原始配对损失上升](https://yingwen.io/crl-figures/neural-distribution-walkthrough-bootstrap.svg)

青短虚线是父真实分布，蓝长虚线是旧目标，橙实线是后继一次写入后同步得到的新目标，均在同一回报坐标。父四点保持不动。图中损失对两份不同标签计算；冻结旧副本时仍使用旧目标。分布标签和均值动作的版本约定承接 [C51 §4.2、§5](https://proceedings.mlr.press/v70/bellemare17a/bellemare17a.pdf#page=6) 与 [IQN §3](https://proceedings.mlr.press/v80/dabney18a/dabney18a.pdf#page=5)。

理论上的逐状态精确分位投影与有限次网络优化也需区分。[QR Proposition 2](https://arxiv.org/pdf/1710.10044#page=5)分析固定策略下“精确投影复合 Bellman”的最大 $W_\infty$ 压缩；[Rowland 等（2018）Proposition 2](https://arxiv.org/pdf/1802.08163#page=6)分析 categorical 精确投影在最大 Cramér 距离中的压缩，其后另给有限表格混合更新的条件。共享 tanh 网络的一次梯度步既未完成逐状态精确投影，也会同时改变未训练输入，因而不能把这些算子结论直接当成它的收敛证明。它们是理解近似的基础理论；神经实现还要面对固定目标优化、参数干扰和自举标签变化。

<a id="lesson-neural-calibration"></a>

### 有限样本：一次经验 CDF 与总体规律

最后恢复“末款只能观察到一次”的条件。已知真值供检查，给学习器的四份指定末款是 $(0,0,0,4)$，对应父完整回报 $(0.5,0.5,0.5,2.5)$。在门槛 z=1，经验 CDF 是 3/4，真实 CDF 是 1/2。即使网络完全拟合这四个回报的经验分布，总体分布误差仍在；若回报来自后继预测，还需另计该预测的误差。

$$
K\sim\operatorname{Binomial}(4,1/2),\qquad \widehat F_4(1)=K/4,\quad \mathbb E\widehat F_4(1)=1/2,\quad \mathbb E|\widehat F_4(1)-1/2|=3/16,\quad\mathbb E[\widehat F_4(1)-1/2]^2=1/16
$$

枚举全部十六个等概率四结果序列即可复算：低结果数 K=0,…,4 的序列数为 1,4,6,4,1。经验 CDF 的期望正确，某份数据仍有偏差。此式检查固定门槛的有限样本波动。

分布校准需要规定按什么输入、概率水平和独立评价数据比较预测与结果。这里只计算一个已知两点任务的 CDF 诊断，并未给出完整神经校准实验。有限样本、有限分位点的表示、共享参数干扰、固定标签未优化到位、自举目标漂移应分别检查；只报告均值收益或一次训练损失无法分辨它们。下一节再保持这些对象明确，讨论控制器怎样读取尾部。

<a id="lesson-risk"></a>

## 5 · 尾部风险不是认识不确定性

$$
\operatorname{CVaR}_{\eta}^{\rm lower}(Z)=\frac1\eta\int_0^\eta F_Z^{-1}(u)\,du,\qquad0<\eta\le1
$$

本章将收益的下尾平均称为 lower CVaR；有些文献用损失上尾，符号与最优化方向需要转换。

回到前面独立表参数的 32 次更新，让控制器读取那组四点估计；上一节共享网络的四次写入是另一组计算。表参数均值是 $\widehat Q=N^{-1}\sum_i\theta_i$；本例前两个点始终最低，最差一半的均值是 $(\theta_0+\theta_1)/2$。初值、一步、第 4 步和第 32 步的均值均为 1.5，下尾读数却分别为 1、0.75、约 0.573、约 0.488。真实浮动下尾是 0.5。即便均值动作一直与固定动作并列，分布拟合的变化仍会改变风险读数。若分位头发生交叉，要按收益位置排序后才能求所表示离散分布的尾部；它们各自的分位标签不会因排序自动变得准确。

![指定总体更新的四个位置快照均值不变而下尾读数变化；浮动首款增加后均值与下尾选择不同动作；有限四点表示产生均值和风险误差](https://yingwen.io/crl-figures/quantile-walkthrough-readout.svg)

上部横轴始终是回报，点的竖直错开只显示重合，标记均为 1/4 质量；32 次总体更新仍有约 0.0123 的 $W_1$ 误差。下部承接同一接单任务，只将浮动首款改为 0.6，并重新读取真实 midpoint 位置：均值选浮动，下尾选固定。这一动作比较使用已知分布参照，不读取尚有误差的 32 步估计。最后一项只改变末款概率，说明有限点数本身也能改变读数。所有快照与读数是原创确定计算。

四个 midpoint 的简单平均在一般任务中只是均值的数值近似，没有 categorical 未裁剪插值的保均值性质。仍用首款 0.5、末款 0/4，现只把高末款概率改成 0.8。真回报在 0.5 的质量为 0.2，在 2.5 为 0.8，真均值为 2.1。四个真实 midpoint 变成 $(0.5,2.5,2.5,2.5)$，其等质量表示将低结果质量变成 0.25，均值为 2。真实下尾 $\operatorname{CVaR}_{0.5}=(0.2\cdot0.5+0.3\cdot2.5)/0.5=1.7$，四点表示却是 1.5；即使这些位置已经准确，有限 N 仍留下表示误差。η 不恰好等于整点质量时，应积分到 η、只取边界原子的所需部分，不能四舍五入成若干完整头。

收益二的确定策略，下尾 CVaR 仍是二；零或四各半的策略，在最差一半的 CVaR 为零。均值相同，风险偏好却给出不同选择。对于多步风险目标，还要区分初始时刻的全回报风险与递归风险度量；逐状态贪心最大化一个 CVaR 预测不自动解决时间一致性。

回到接单任务，先用真实分布：固定回报 1.5 的下尾 CVaR 仍为 1.5；浮动回报 0.5/2.5 各半，在最差一半的 CVaR 为 0.5。均值并列时两动作都符合期望准则，本页用固定结算打破并列。为了让两项准则产生严格不同的动作，现只把浮动动作首款由 0.5 改为 0.6。其真实回报变成 0.6/2.6 各半，均值 1.6，$\operatorname{CVaR}_{0.5}$ 为 0.6。于是最大化期望选浮动，最大化下尾 CVaR 选固定，下一步进入不同结算状态并取得不同末款。

![上部五原子分类概率的一次固定目标更新，下部已知真实分布下期望与下尾CVaR选择不同结算动作](https://yingwen.io/crl-figures/distributional-walkthrough-update-risk.svg)

上半部只做一次步长一的 logits 更新，柱高为更新前、固定目标、更新后的概率。下半部单独将浮动首款改成 0.6，并使用真实回报分布作动作比较：橙箭头按期望走向 v，青箭头按下尾均值走向 f；紫色标出浮动最差一半的质量。先预测：若仍使用同一个均值准则，只提高分布拟合精度，是否必然变得更厌恶风险？常规 C51 按期望选择动作，见[原文 §5](https://proceedings.mlr.press/v70/bellemare17a/bellemare17a.pdf#page=6)；收益下尾准则与多步目标见[作者教材 §7.8](https://www.distributional-rl.org/contents/chapter7)。

风险计算还会受到表示误差影响。在首款同为 0.5 的原例中，固定回报 $\delta_{1.5}$ 投影为 $\tfrac12\delta_1+\tfrac12\delta_2$，均值仍为 1.5，下尾 $\operatorname{CVaR}_{0.5}$ 却由 1.5 变成 1。浮动的四点投影在最差一半取 0 与 1，$\operatorname{CVaR}_{0.5}$ 为 0.5。由此可见，投影保均值并不保尾风险。若控制器读取学得的分布，必须用该近似分布重算风险，并检验它与真实尾部的差距。本例只在起点作一次选择、结算状态没有后续选择，所以不存在多次风险重规划；更一般的全回报 CVaR 控制需另处理状态与时间一致性。

最后撤掉“浮动概率已知”这一条件。令末款为 4 的概率是未知参数 q，给它 Beta 后验。$\operatorname{Beta}(1,1)$ 和 $\operatorname{Beta}(50,50)$ 都有均值 0.5，因此下一次浮动回报的预测分布仍是 0.5/2.5 各半，预测方差仍为一；但 q 的后验方差从 $1/12$ 降到 $1/404$。一个分布描述下一次结果，另一个描述我们知道多少概率规律。可以对 q 更确定而面对同样的随机收益。

$$
\operatorname{Var}(G\mid\mathcal D)=\mathbb E\!\left[4q(1-q)\mid\mathcal D\right]+\operatorname{Var}(0.5+2q\mid\mathcal D)
$$

全方差公式将预测方差分成给定 q 后的结果随机性，以及条件均值随未知 q 变化的部分。认识不确定性下降时，前一项也随后验改变；只看预测分布的宽度无法把两项分开。

<a id="rlss-risk-atoms"></a>

## 离散回报的尾部风险：为什么不能直接取阈值以下的均值

许多 RL 回报分布有点质量：大量轨迹得到完全相同的分数。此时，按“所有不大于 5% 分位点的样本取平均”计算下尾风险，可能包含远超过 5% 的概率。必须只取所需的尾部概率质量，包括分位点处的部分质量。

$$
\Pr(Z=-99)=0.01,\qquad \Pr(Z=199/99)=0.99,\qquad\mathbb EZ=1.
$$

另一动作确定得到一，均值也为一。二者具有不同的后果分布，但这本身不是标量奖励无法表达目标的反例。

$$
\operatorname{CVaR}_{0.05}^{\rm lower}(Z)=\frac{0.01(-99)+0.04(199/99)}{0.05}\approx-18.1919.
$$

最差 5% 包含全部 1% 灾难质量与正常结果中的 4%。5% 分位点为 199/99；若把所有不超过此点的结果都选上，就包含了整个分布，错误地得到均值一。

在代码中，将结果按收益从小到大排序，逐项取 min(剩余尾部质量, 当前原子的概率)。累积加权收益后除以规定尾部质量。若有有限样本，最后一个样本也可能只占部分权重。配套 distributional 实现已经处理部分原子；应保留这项测试。

均值相同仅说明期望回报准则对这两个行动无差别。偏好更低风险可以通过约束、效用或风险函数表达，但改变准则后须重新推导多步决策的递推与时间一致性。分布预测准确、尾部样本足够与风险控制正确，是三个不同要求。

<a id="lesson-algorithm"></a>

## 6 · 两条分布更新的精确步骤

**算法：预测对象、训练损失和行动准则应分别记录。**

1. 从真实或 replay 数据得到 $(s,a,r,s',d)$。
1. 按声明的准则选后继动作；标准版本使用分布均值。
1. Categorical：平移并缩放 target atoms，裁剪支持，分配质量。
  1. 固定投影目标，对当前动作的概率做交叉熵更新。
1. Quantile：计算所有当前/目标分位数对的 TD residual。
  1. 固定目标，用相应分位水平的非对称损失更新位置。
1. 同步 target；独立报告均值收益、分布校准与风险指标。

<a id="lesson-example"></a>

## 7 · 投影、分位梯度和风险手算

支持 $(-1,0,1)$，概率 $(0.2,0.5,0.3)$，奖励 $0.5$、折扣 $0.5$。目标点为 $(0,0.5,1)$。中间点的一半质量分到零、一，所以投影概率是 $(0,0.45,0.55)$，总质量一，均值 $0.55$。若真终止且奖励零，全部质量应落在零。

数据为 $(0,2)$，$\tau=0.75,\theta=0.3$。两个 pinball 梯度平均为 $(1-0.75-0.75)/2=-0.25$，梯度下降提高位置。这里上分位数为二；下一章同一数据的 0.75 expectile 却为 1.5，两种不对称回归不能混用。

<a id="lesson-code"></a>

## 8 · 可运行分布算子

接单任务提供独立的[标准库 Python 单文件教程](/crl-code/tutorials/distributional-walkthrough.py)。保存到任意空目录即可运行：默认打印真实回报、投影、一次 logits 更新及两种动作选择；--json 给出图中数列；--test 用 Fraction 检查投影，并以 CVaR 的变分式和交叉熵有限差分作独立核对。

Python 标准库；已知两步任务与一次确定更新，随机 rollout 和神经训练步数均为零。

```bash
python3 distributional-walkthrough.py
python3 distributional-walkthrough.py --test
python3 distributional-walkthrough.py --json
```

同一任务的[分位数标准库单文件](/crl-code/tutorials/quantile-walkthrough.py)补充真实 midpoint、16 对残差、一步与 32 次指定总体更新、Huber 最优位置和有限点数读出。JS 以逆 CDF 分段积分算 $W_1$、尾部；Python 独立以回报轴上两条 CDF 间面积算 $W_1$、以尾部变分式算 CVaR，另用 Fraction 精算一次写入、用有限差分核对两种损失，并通过零导数搜索核对 Huber 偏移。

在任意空目录运行，Python 标准库，无随机采样或神经训练。JSON 为本页原创图的可重算数据。

```bash
python3 quantile-walkthrough.py
python3 quantile-walkthrough.py --test
python3 quantile-walkthrough.py --json
```

[共享非线性网络单文件](/crl-code/tutorials/neural-distribution-walkthrough.py)把本页十参数网络、四次冻结标签写入、一次独立后继写入及十六个四样本序列放在一起。--test 在没有零残差的两份参数处逐一有限差分十个参数，以独立完整轨迹枚举核对真均值，并以回报轴 CDF 面积核对 JS 的逆 CDF $W_1$。Python 只用标准库，从空目录运行；确定的五次网络写入不启动随机训练。

读 k=0,…,4 的两动作预测与 W1，再比较父目标前后版本。--json 给出三组原创图的可复算数据。

```bash
python3 neural-distribution-walkthrough.py
python3 neural-distribution-walkthrough.py --test
python3 neural-distribution-walkthrough.py --json
```

接着可读已有 [C51 小任务代码与运行结果](/zh/continual-rl/code/deep-c51/)及 [QR-DQN 小任务代码与运行结果](/zh/continual-rl/code/deep-qr_dqn/)。它们使用带期限的 DeadlineChain，环境与本页解析接单任务不同；页面奖励读数检查其特定行为表现，不能由回报曲线推断预测分布已经校准或尾部准确。

分类投影、pinball 次梯度和离散分布下尾 CVaR，包含整数点与部分质量的处理。

```python
def categorical_projection(reward, gamma, terminal, support, masses):
    probabilities(masses)
    n = len(support)
    if n < 2 or len(masses) != n or not 0 <= gamma <= 1:
        raise ValueError('two or more atoms and valid discount required')
    spacing = (support[-1]-support[0])/(n-1)
    if spacing <= 0 or any(not math.isclose(z, support[0]+i*spacing) for i, z in enumerate(support)):
        raise ValueError('equally spaced ascending support required')
    output = [0.]*n
    for atom, mass in zip(support, masses):
        target = reward+(0. if terminal else gamma*atom)
        target = min(support[-1], max(support[0], target))
        position = min(n-1, max(0., (target-support[0])/spacing))
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            output[lower] += mass
        else:
            output[lower] += mass*(upper-position)
            output[upper] += mass*(position-lower)
    return output


def quantile_loss_gradient(theta, samples, tau):
    if not samples or not 0 < tau < 1:
        raise ValueError('nonempty samples and interior quantile required')
    loss = sum((tau-float(y-theta < 0))*(y-theta) for y in samples)/len(samples)
    # At equality this selects one valid subgradient.
    gradient = sum(float(y < theta)-tau for y in samples)/len(samples)
    return loss, gradient


def lower_cvar(values, masses, fraction):
    probabilities(masses)
    if len(values) != len(masses) or not 0 < fraction <= 1:
        raise ValueError('valid mass fraction required')
    remaining, total = fraction, 0.
    for value, mass in sorted(zip(values, masses)):
        used = min(mass, remaining)
        total += used*value
        remaining -= used
        if remaining <= 1e-14:
            break
    return total/fraction
```

运行 test 检查质量守恒、支持裁剪、terminal 点质量、分位梯度的有限差分和 CVaR 部分原子。它不训练 C51 或 QR-DQN 网络。Dopamine 是 Google 研究团队提供的分布 RL 实现框架；阅读具体 agent 与配置，不能把框架默认值当作所有原论文的同一实验协议。

<a id="lesson-branches"></a>

## 9 · 适用边界与 CRL 衔接

- 预测分布变宽，既可能是环境风险上升，也可能是策略覆盖改变，不能直接解释成“不知道”。
- 固定支持的溢出、分位数交叉与低概率尾部样本不足，是不同的数值与统计问题。
- 依照期望行动的 distributional agent，并没有因此满足约束安全或最坏情况保证。

持续任务中可以为 GVF 预测整个未来信号分布，也可以研究奖励分布变化后的校准与尾部恢复。设计对照时固定行动准则，才能先测分布表示本身的影响；若同时更换风险目标，应把目标变化单独报告。

<a id="lesson-check"></a>

## 10 · 练习与答案

- 预测：初始固定四头完全准确，仅用浮动输入更新，共享偏置会怎样影响它？答：一步后固定输入仍有 h=0，但偏置变为 (1.453125,1.484375,1.515625,1.546875)，均值仍为1.5，分布 $W_1$ 已为1/32。
- 定位条件：两个动作都编码为+1，能否增加宽度使同一个确定输出同时精确表示两真分布？答：两输入完全相同，只能给同一分布；两项 $W_1$ 之和至少为1。
- 定位变化：父预测不变，后继写入后父的 pinball 损失上升，是否证明父网络优化失败？答：先检查目标版本；本例父参数不变而标签改变，目标 $W_1$ 下降、原始配对损失上升。
- 手算：四份浮动末款有三个0、一个4，z=1的完整回报经验CDF是多少？答：3/4；总体真CDF为1/2，完全拟合这批数据仍不能消去总体误差。

- 手算：浮动目标中只抽到 0.5，仍以 τ=(1/8,3/8,5/8,7/8)、θ=(1,1,2,2) 和每头步长一更新，得到什么？答：样本次梯度为 (7/8,5/8,3/8,1/8)，新位置为 (0.125,0.375,1.625,1.875)。抽到高目标时会不同；枚举全部冻结目标的平均不是一次环境抽样。
- 定位条件：τ=0.5、浮动真目标在0.5/2.5各半，最优 θ 是否唯一？答：pinball 最优区间为 [0.5,2.5]，广义逆 CDF 的约定值是 0.5。
- 预测：Huber κ=1 时把 θ=0.5、τ=1/8 作为准确位置停止更新，是否合理？答：总体导数为 −1/16，会向上移动，平滑最优位置约0.643。
- 读取尾部：四点位置 (0,1,2,3) 每点质量1/4，η=3/8，下尾均值为多少？答：取0的全部1/4和1的1/8，除以3/8，得到1/3，不能平均前两个完整点。

- 预测：首款同为0.5、γ=0时，两动作的回报分布怎样变化？答：都变成$\delta_{0.5}$，末款不再进入目标；投到本例整数支持后都为0/1各半。
- 手算：首款改为3时，固定与浮动的真实均值都为4。五原子表示给出的均值是否仍并列？答：固定目标恰为4，浮动的5被裁到4，投影均值仅3.5；归一化检查仍然通过。
- 定位错误：将真正终止的奖励2写成目标零，会丢失什么？答：丢失了本次奖励；正确目标是$\delta_2$，只去掉未来自举项。
- 设计对照：如何分辨收益随机性和概率知识不足？答：固定真实q与行动规则，增加概率估计的数据，分别报告q后验宽度、真实回报分布和分布预测误差；不把尾风险的宽度用作知识宽度。

- 问：C51 的 51 表示 51 个不同策略吗？答：不是，表示固定支持上的概率原子数。
- 问：原子恰落在格点时两个插值权重都为零，怎么办？答：直接把全部质量放入该格点。
- 问：回报方差大就应探索更多吗？答：不一定。已知的环境随机性不等于可以由数据减少的知识不确定性。
- 实验：将奖励从 0.5 改为五而保持支持 [-1,1]。投影全部落在一，质量仍守恒但均值被截断；这说明“测试归一化通过”不足以确认建模正确。

## 从本章进入实践

[预测与控制](https://yingwen.io/zh/continual-rl/code/#practice-prediction)：学会预测更多事情，什么时候会改变行动？



<a id="chapter-code"></a>

## 下载与运行

标准库解析机制实验；不包含完整神经网络训练或大型 benchmark。

[下载 extended_foundations_lab.py](https://yingwen.io/zh/continual-rl/download/extended_foundations_lab.py)

```sh
python3 extended_foundations_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)：§3.1 的奖励/后继联合核、§3.3–3.6 的回报递推、终止与期望价值，作为本章分布扩展的起点。

- [Bellemare、Dabney、Munos · A Distributional Perspective on Reinforcement Learning](https://proceedings.mlr.press/v70/bellemare17a.html)：固定策略分布算子、控制的差别与 categorical 算法。

- [Dabney et al. · Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)：式 (7)–(13)、Lemma 2 与算法 1：等质量 midpoint 投影、pinball、Huber 和均值动作；精确投影与有限网络更新不同。

- [Dabney et al. · Implicit Quantile Networks](https://proceedings.mlr.press/v80/dabney18a.html)：§2.3、§3、式 (1)–(4)：归一化 Huber、当前/目标/动作三组独立水平与风险读出。

- [Yang et al. · Fully Parameterized Quantile Function](https://proceedings.neurips.cc/paper/2019/hash/f471223d1a1614b58a7dc45c9d01df19-Abstract.html)：§3.1–3.4、Proposition 1 与算法 1：可学习概率边界、区间质量和加权均值；边界导数的连续非降条件。

- [Google · Dopamine](https://github.com/google/dopamine)：原研究团队框架，提供分布算法与 Atari 工程；非本页标准库核。

- [Bellemare、Dabney、Rowland · Distributional Reinforcement Learning](https://direct.mit.edu/books/oa-monograph/5590/Distributional-Reinforcement-Learning)：原作者体系教材，可按 Bellman 分布、投影、控制与 statistical functionals 深入。

- [Elsayed et al. · Streaming Deep Reinforcement Learning Finally Works](https://arxiv.org/abs/2410.14606)：Stream-X 将神经预测与控制放回逐样本更新；应逐项检查归一化、初始化、资格迹和步长控制，不把移除 replay 当作完整算法。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-distributional#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-distributional#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-deep-distributional)
<a id="study-connections"></a>

## 与教材主线的衔接

本章是可按问题选择的并列研究分支，不要求先读完其他深度研究分支。

### 预测的量是什么，误差又是什么？

先固定策略。价值是回报的条件期望。MC 使用完整回报；TD 用下一时刻的预测替代未观察的余项。两者不能仅按同一批样本上的 TD error 排序。

函数逼近与深度方法：共享参数限制了可表示的函数。采样权重决定在哪些状态上拟合。最小价值误差、最小 Bellman 残差和 TD 固定点一般不同；神经网络又使可表示的局部方向随参数改变。

持续学习中的研究问题：多个 GVF 共用表示时，哪些预测值得占用容量？应分别检查问题定义是否改变、数据是否覆盖，以及回答该问题的误差是否降低。

[MC 与 TD](https://yingwen.io/zh/continual-rl/foundations/tabular/temporal-difference/) → [投影与半梯度](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/) → [神经价值更新](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/) → [GVF 的问题与答案](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)


### 可进一步检验的问题

- [10 · 模型需要预测什么，才能在变化后继续支持决策？](https://yingwen.io/zh/continual-rl/research/#research-reusable-models)：相同回报均值可以对应不同的分布和风险排序，说明只预测平均后果何时不足以支持下游决策。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [深度价值学习](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)
- [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)
- [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

对应原始材料：Bellemare、Dabney、Munos：Distributional Perspective；Dabney et al.：Quantile Regression；Distributional Reinforcement Learning。本文为原创讲解，原书、论文与上游代码保留各自许可。
