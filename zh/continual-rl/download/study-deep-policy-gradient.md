# 策略梯度：从轨迹概率到 GAE 与 actor–critic

现代深度强化学习 · 第 2 章

延迟奖励怎样改变动作概率？有限 rollout、critic 与停止梯度分别改变哪一项估计？

## 本章内容

- 从完整轨迹 score 推到 reward-to-go 与 baseline。
- 分清回合目标、有限 rollout 权重和实现 surrogate。
- 解释 critic 误差怎样进入有限 GAE，并设置两类 mask。
- 追踪 actor/critic 的固定量与求导路径，再进入 PPO。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [策略梯度、基线与 Actor–Critic](https://yingwen.io/zh/continual-rl/foundations/approximation/policy-gradient/)：掌握轨迹概率、score与基线的基本推导。
- [多步回报、资格迹与 True-online TD](https://yingwen.io/zh/continual-rl/foundations/approximation/traces/)：理解多步误差加权，再解释GAE中的等待长度。


### Bellman 递推

当前价值等于即时奖励加折扣后的后继价值；预测目标与最优控制目标不同。

### 函数与梯度

神经网络把参数和输入映射为输出。链式法则必须指明哪些量参与求导，哪些量作为固定目标。

### 经验分布

on-policy 数据由当前策略产生；off-policy 数据可来自旧策略，但能否正确复用取决于算法目标和数据覆盖。

<a id="problem-definition"></a>

## 本章的问题定义

从当前随机策略产生的轨迹估计收益梯度，并用价值网络减少对完整回报的依赖。

### 给定条件与符号

- 初始分布 $\rho$、真正终止时刻 $T$、$0<\gamma\le1$；目标 $J(\theta)=\mathbb E_{\pi_\theta}\sum_{t<T}\gamma^tR_{t+1}$。
- 有限期限任务将剩余时间计入状态；回报可积，轨迹求导与期望可以交换。
- 可微策略 $\pi_\theta$、价值近似 $V_\phi$、优势估计 $\hat A_t$；采样前固定本批策略与 critic 参数。

### 需要求解的对象

构造与声明收益目标对应的 actor 更新，同时学习用于该估计器的 critic。

### 信息与数据权限

标准 on-policy 更新使用采样时策略的动作概率；使用旧策略数据需额外校正。

$$
\nabla_\theta J=\mathbb E\!\left[\sum_{t<T}\gamma^t\nabla_\theta\log\pi_\theta(A_t\mid S_t)A^{\pi_\theta}(S_t,A_t)\right]
$$

$A^\pi=q^\pi-v^\pi$ 是真实优势。有限 rollout 的均匀样本损失还需核对访问权重；近似 critic 可改变优势估计。真 critic 与正确边界下，有限截断本身不必增加偏差。

### 成立条件与解的含义

- 轨迹微分与期望交换合法；策略参数影响环境仅通过所执行动作。
- GAE 的有限 rollout 边界需正确区分真实终止和采样截断。

判断准则：先验证 score、优势与参数冻结时序，再将梯度估计的偏差方差与收益变化区分。

### 适用边界

- 由 GAE 降方差推断估计必然无偏或策略必然改善。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 限制表示或采用近似 · [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)：神经网络近似策略和价值，不改变 score-function 的概率学起点。

- 组合不同学习问题 · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)：GAE 在 critic 残差上分配时间权重，不等于完整递归网络参数信用。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

真实优势未知，长轨迹回报噪声大，截断窗口又留下未观察未来。

### 本章的核心思路

先核对目标和样本权重，再用具有明确边界的多步残差估计未知优势。

1. [先求轨迹梯度](#lesson-derive)：对概率求导，再用条件期望删除过去奖励和满足条件的基线。

2. [从目标到实际采样权重](#lesson-weighting)：完整回合、随机行数归一化与固定长度 rollout 不能不加区分地取均值。

3. [构造时间上的优势估计](#lesson-advantage)：GAE 混合 TD 残差，并按终止和截断语义处理尾值。

4. [隔离 actor 与 critic 的梯度职责](#lesson-algorithm)：actor 中固定优势；critic 使用独立的回归标签和损失。

结论与条件：真实优势给出梯度恒等式；实际 actor–critic 的性质还取决于 critic 误差和数据协议。

### 相关方法改变了什么

- REINFORCE / GAE：分别依赖完整采样回报和带自举的多步残差。

- VPG / A2C：共享策略梯度基础，但采样长度、critic 和同步方式可不同。


<a id="lesson-setting"></a>

## 1 · 随机策略和有限轨迹

第一册已解释直接优化动作概率的动机，本章沿用 Sutton 与 Barto 第13章的基本对象，继续处理神经策略的采样与优化。考虑三步通关：只有连续前进才在最后得到奖励，早期动作的即时奖励全为零。对当前奖励求导给不出早期动作的信用，而对未知环境的整条路径求导也不可行。我们需要从实际动作的概率和后来收到的奖励构造更新。

策略 $\pi_\theta(a\mid s)$ 输出动作分布，离散动作可用 softmax，连续动作可用 Gaussian。先规定目标 $J(\theta)=\mathbb E_{\pi_\theta}[G_0]$，其中 $G_t=\sum_{k=t}^{T-1}\gamma^{k-t}R_{k+1}$，$0<\gamma\le1$，$T$ 是任务真正终止的时刻。所有策略共用初始分布 $\rho$，奖励与环境转移不直接依赖 $\theta$。有限期限任务将剩余时间计入状态；下文价值指这个充分状态下的价值。

本轮用参数 $\theta_0$ 的策略 $\pi_0$ 收集经验，critic 参数为 $\phi_0$；先固定两者，再计算这轮估计。真实量为 $v^{\pi_0}(s)=\mathbb E_{\pi_0}[G_t\mid S_t=s]$、$q^{\pi_0}(s,a)=\mathbb E_{\pi_0}[G_t\mid S_t=s,A_t=a]$ 和优势 $A^{\pi_0}=q^{\pi_0}-v^{\pi_0}$。网络 $V_{\phi_0}$ 只是估计，不能在等式中直接替代真值。

Actor 改变动作分布。价值估计既可作为完整回报的比较基线（baseline），也可预测尚未观察到的尾部回报，即自举（bootstrap）。前一种用法仍是 REINFORCE 加 baseline；后一种让 critic 参与动作评价，构成这里的 actor–critic。A2C 同步收集与更新，A3C 使用异步工作者；区分它们时，还要看参数版本与执行次序。

<a id="course-policy-class"></a>

## 1.1 · 先定义策略类：为何有时必须学习随机策略

有限折扣 MDP 在常规条件下存在确定性的平稳最优策略。这不表示观察受限、记忆受限或参数共享之后，受限策略类的最优解仍然确定。策略梯度的一个动机，是直接优化可表达的动作概率，而非只在固定 ε-greedy 规则下改变动作排序。

考虑教材中的短走廊。非终止位置为零、一、二，终点在三。动作“右”在位置零、二向右，在位置一反而向左；“左”反向，位置零向左时原地不动。每步奖励 −1，本例使用无折扣的回合总回报（$\gamma=1$）。观察或参数化使三个位置使用相同概率 p 选择“右”。这里暂不允许 recurrent 记忆。

$$
\begin{aligned}
T_0&=1+(1-p)T_0+pT_1,\\
T_1&=1+pT_0+(1-p)T_2,\\
T_2&=1+(1-p)T_1.
\end{aligned}
$$

T 是从对应位置到达终点的期望步数。每式中的一，是刚刚执行的那一步。只有 0<p<1 时，起点的期望到达时间有限。

$$
J(p)=-T_0=-\frac{2(2-p)}{p(1-p)},\qquad
\frac{dT_0}{dp}=\frac{2(-2+4p-p^2)}{p^2(1-p)^2},\qquad
p^*=2-\sqrt2
$$

在可行区间中令导数为零，得到 p≈0.5858，期望回报约 −11.6569。p 为零会永远停在起点；p 为一会在前两个位置间循环。

复查策略类内的最优值；标准库即可执行。

```python
from math import sqrt
p = 2 - sqrt(2)
steps = 2 * (2 - p) / (p * (1 - p))
assert abs(steps - (6 + 4 * sqrt(2))) < 1e-10
print(p, -steps)  # 解析式核验，不是训练结果
```

若把真正的位置提供给策略，确定地依次选择右、左、右，只需三步。因此随机最优不是这个世界不可避免的性质，而是信息和策略类约束的结果。增加记忆、改善 agent state 和直接学习随机策略，是三个不同改动，实验应分开。

思考：如果动作概率已经接近零，softmax 梯度会很小，相关动作也很少被采到。新的任务要求反转这个偏好时，慢适应可能同时来自参数饱和与数据缺失。不能只增加 critic 容量，就断言问题已经解决。

<a id="lesson-derive"></a>

## 2 · 对轨迹概率求导，不对环境求导

$$
\begin{gathered}p_\theta(\tau)=\rho(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t\mid s_t)P(s_{t+1},r_{t+1}\mid s_t,a_t),\\\psi_t=\left.\nabla_\theta\log\pi_\theta(A_t\mid S_t)\right|_{\theta_0},\\\left.\nabla J\right|_{\theta_0}=\mathbb E_{\pi_0}\!\left[G_0\sum_{t=0}^{T-1}\psi_t\right].\end{gathered}
$$

先用 ∇p=p∇log p 对完整轨迹概率求导。环境核不含策略参数，故对数导数只保留动作项；策略改变到达后续状态的概率，已包含在这条轨迹的概率中。假定策略光滑、相关动作具有正概率，且回报可积并允许交换求导与期望。

完整轨迹公式把同一个 $G_0$ 乘给每个动作，包含了该动作发生前的奖励。记 $\mathcal H_t$ 为选择 $A_t$ 前的历史（含 $S_t$），$C_t=\sum_{k<t}\gamma^kR_{k+1}$ 为已收到的折扣奖励。条件于这段历史，$C_t$ 已确定，而动作 score 的平均为零。

$$
\begin{gathered}\mathbb E_{\pi_0}[\psi_t\mid\mathcal H_t]=\sum_a\pi_0(a\mid S_t)\left.\nabla\log\pi_\theta(a\mid S_t)\right|_{\theta_0}=0,\\G_0=C_t+\gamma^tG_t,\qquad\mathbb E[\psi_tC_t]=0,\\\left.\nabla J\right|_{\theta_0}=\mathbb E_{\pi_0}\!\left[\sum_{t<T}\gamma^t\psi_tG_t\right].\end{gathered}
$$

由条件期望删除过去奖励，得到从当前动作起的回报（reward-to-go）。删除的是期望为零的项；一条具体轨迹上，两个估计的数值可以不同。

$$
\mathbb E_{\pi_0}[\psi_tB_t\mid\mathcal H_t]=B_t\sum_a\pi_0(a\mid S_t)\left.\nabla\log\pi_\theta(a\mid S_t)\right|_{\theta_0}=0
$$

一个充分条件是 $B_t$ 在当前动作前已经确定，例如用预先固定的 $V_{\phi_0}(S_t)$。于是 $G_t$ 可换成 $G_t-B_t$，期望梯度不变。基线能改变方差；真实状态价值是有用的选择，但不保证使整个回合梯度方差最小。

这项条件涉及样本怎样产生。网络只输入 state，并不保证用当前回报拟合后的输出与当前动作无关。若先拟合本样本再作基线，统计抵消可能失效；detach 只切断求导路径。可先保存旧价值，再训练 critic，或按独立完整回合交叉拟合。把同一轨迹的转移随机拆开不能保证独立，因为后续状态仍可透露早期动作的后果。

actor 的求导路径是固定权重乘 score：$\operatorname{sg}(G_t-B_t)\nabla\log\pi_\theta$。若让梯度穿过 $B_t$，会额外出现基线导数；那是另一种更新。采样参数固定保证经验来自声明的 $\pi_0$，baseline 的统计条件保证消去成立，停止梯度规定优化器这次对谁求导。三项各有作用，同一轨迹内的转移仍然相互依赖。

<a id="lesson-weighting"></a>

## 2.1 · 从完整回合到有限 rollout：怎样给样本加权

$$
L_{\rm ep}(\theta)=-\frac1n\sum_{i=1}^{n}\sum_{t=0}^{T_i-1}\gamma^t\log\pi_\theta(a_{i,t}\mid s_{i,t})\operatorname{sg}(\hat A_{i,t})
$$

$n$ 是预先固定的独立完整回合数，每回合从 $\rho$ 开始并使用同一 $\pi_0$。在 $\theta_0$ 处，若优势满足相应 score 期望条件，负损失梯度估计 $\nabla J$；离开 $\theta_0$ 后，它只是固定数据的优化目标。

内层 $G_t$ 中的 $\gamma^{k-t}$ 规定从当前动作向后看多久；外层 $\gamma^t$ 规定该决策距回合起点多远。前者不能代替后者。例如固定两步、两处各有独立 Bernoulli logit，只有两次动作均为一才在第二步获奖。两处概率都是 $.5$、$\gamma=.5$ 时，$J=.5\times.5^2=.125$，两个 logit 的真实梯度均为 $.0625$；省去外层折扣得到 $(.0625,.125)$，改变了相对更新方向。

归一化也要看随机量。记完整回合的梯度和为 $\hat g$，总行数为 $M=\sum_iT_i$，一般有 $\mathbb E[\hat g/M]\ne\mathbb E[\hat g]/\mathbb E[M]$。最小例子取一个回合、$\gamma=1$：起点等概率选择立即结束并得一，或再等一步得一；第二步的行为与该 logit 无关。$J=1$，真实梯度为零，两类回合的梯度和是 $+.5,-.5$。按各自行数一、二取均值后，期望却为 $.5(.5)+.5(-.25)=.125$。

$$
L_B(\theta)=-\frac1B\sum_{j=1}^{B}\log\pi_\theta(a_j\mid s_j)\operatorname{sg}(\hat A_j)
$$

实际固定 B 行 rollout 常采用这个均匀样本目标。B 是固定数，没有上一例的随机分母；但这 B 行可以包含多个回合和未完成片段，它们的状态权重仍需单独判断。

配套训练取 $B=128,\gamma=1$，所以没有外层折扣的缺项；它仍使用有限 GAE、批内优势标准化与部分回合。普通 rollout 尾部继续保留环境活动，下一批的起始状态可能由上一版策略带到。当前动作由 $\pi_0$ 采样，并不自动令整个批次具有完整回合的访问权重。这里采用工程的局部样本 surrogate，不能仅凭参数冻结就称它为前一公式的精确估计；去掉外层折扣也没有定义平均奖励目标。

要复现起点折扣梯度，需要相应的回合起点与时间权重，或明确按折扣占用分布采样并保留归一化系数。要理解现有训练，则按它实际的 rollout 权重读损失。下一步在这些样本上估计回报：完整结果尚未到来时，critic 能提供怎样的尾部预测？

<a id="lesson-advantage"></a>

## 3 · TD 残差、多步优势与 GAE

用完整 $G_t-V_{\phi_0}(S_t)$ 时，价值误差只进入动作前的基线。缩短等待后，以 $R_{t+1}+\gamma V_{\phi_0}(S_{t+1})$ 替代未知回报，后继状态的预测也进入动作评价。若 $V_{\phi_0}=v^{\pi_0}$，其条件均值是 $q^{\pi_0}(s,a)$；近似值则可能通过动作依赖的后继状态改变优势方向。GAE 将不同等待长度的这些估计混合。

$$
\begin{gathered}\delta_t=R_{t+1}+\gamma b_tV_{\phi_0}(S_{t+1})-V_{\phi_0}(S_t)\\\hat A_t^{\mathrm{GAE}}=\delta_t+\gamma\lambda c_t\hat A_{t+1}^{\mathrm{GAE}}\end{gathered}
$$

$b_t$ 是能否 bootstrap；$c_t$ 是能否把下一行样本的优势继续接上。真正终止使二者为零。环境被人工重置的 timeout 通常 $b_t=1$、$c_t=0$。普通 batch 尾部用最后观测的价值，递推 carry 初始化为零。

![GAE 中价值自举与下一行优势的两条接续路径，在三种边界下分别开关](https://yingwen.io/crl-figures/concept-depth-classic-gae-masks.svg)

三栏固定奖励、旧价值与折扣参数，只比较不同边界语义。蓝箭头从最后观测取 V′；橙箭头接下一行的优势，只有它属于同一轨迹时才可接入。人工重置后不能使用重置观察代替最后观测。数值是 [GAE §3](https://arxiv.org/pdf/1506.02438#page=4) 递推的原创算例；时间截断的自举语义见 [Pardo 等 §3](https://proceedings.mlr.press/v80/pardo18a/pardo18a.pdf#page=5)。[计算代码](/crl-code/figures/classic-visual-depth.mjs)。

两个 mask 回答不同问题：尾部回报在任务中是否还存在，以及下一行残差是否仍属于这条轨迹。若任务规定截止时刻就是结局，该期限是终止，$b_t=c_t=0$；若时间限制只用于收集经验，尾部尚存在，应自举。自动 reset 的环境必须保存 reset 前的最后观测来计算 next value。

若同一片段从 $t$ 起还有 $K$ 个残差，展开为 $\hat A_t=\sum_{l=0}^{K-1}(\gamma\lambda)^l\delta_{t+l}$。$\lambda=0$ 只留一步误差；$\lambda=1$ 时，望远镜相消得到片段回报加尾值，再减起点预测。只有真终止才令尾值为零，并成为完整 $G_t-V_{\phi_0}(S_t)$。

$$
\begin{gathered}A_t^{(k)}=\sum_{l=0}^{k-1}\gamma^l\delta_{t+l}=\sum_{l=0}^{k-1}\gamma^lR_{t+l+1}+\gamma^kV_{\phi_0}(S_{t+k})-V_{\phi_0}(S_t)\\\hat A_t^{\rm GAE}=(1-\lambda)\sum_{k=1}^{K-1}\lambda^{k-1}A_t^{(k)}+\lambda^{K-1}A_t^{(K)}\end{gathered}
$$

在没有跨序列边界的 K 步片段上，先由 TD 残差望远镜相消得到 k 步优势，再混合不同长度。最后一项保留全部剩余权重，故权重和为一；真终止的尾值取零。K=1 时只有一步，λ=1 时只有 K 步估计，片段截断时仍含尾值。

第 $l$ 个残差出现在所有 $k\ge l+1$ 的项中。它的混合权重为 $(1-\lambda)\sum_{k=l+1}^{K-1}\lambda^{k-1}+\lambda^{K-1}=\lambda^l$，再乘 $\gamma^l$，就得到 GAE 的 $(\gamma\lambda)^l$。这解释了递推系数的来源，也避免把有限片段末端的余重丢掉。

若 critic 在采样动作前固定，且恰为 $v^{\pi_0}$、状态充分、边界正确，一步残差的条件均值就是 $A^{\pi_0}(s,a)$；后续残差在 $\pi_0$ 的动作上均值为零，有限 GAE 对任意 $\lambda$ 都不因自举额外产生偏差。近似 critic 的中间与尾部误差怎样进入，见紧接的误差展开。误差来自用于评价动作的预测，不由训练 critic 时使用 MC 还是 TD 的名称决定。

原 GAE 论文先以无折扣总回报为目标，再定义 $g^\gamma=\mathbb E\sum_t\psi_tA^{\pi,\gamma}(S_t,A_t)$；其 $\gamma$-just 条件保持这个表达的 score 期望，没有外层 $\gamma^t$。本章 $J$ 将折扣写进任务目标，故对应梯度还有外层时间权重。准确估计一个优势，与按正确访问权重汇总它，是两步不同的判断。

本教程先用旧 critic 计算全部优势，并选择有限 $\lambda$-return 作为回归目标：$\hat R_t=\hat A_t^{\rm GAE}+V_{\phi_{\rm old}}(s_t)$，然后才更新网络。这是 critic 标签的一种选择；GAE 本身并不规定所有实现必须如此。Spinning Up 的 PPO 则使用带尾值的完整片段 rewards-to-go，通常在 $\lambda<1$ 时与此不同。优势标准化只能用于 actor 的权重；若把标准化后的优势加回 value 当作 critic target，就改变了价值的奖励单位。

<a id="course-gae-error"></a>

## 3.1 · GAE 的偏差到底从哪一项进入

固定采样策略 $\pi$；critic 在采集这段数据前已经确定，并在片段内保持不变。记 $\epsilon_t=V_\phi(S_t)-v^\pi(S_t)$。用真价值形成的 TD 残差记为 $\delta_t^*$。先在长度 K、没有中间 reset 的同一段数据上比较两个估计，而不是把网络误差与轨迹噪声混在一起。

$$
\delta_t-\delta_t^*=\gamma\epsilon_{t+1}-\epsilon_t
$$

这是逐轨迹的代数恒等式，不需要平均。它说明一个价值误差会同时进入当前减项和前一步的自举项。

$$
\begin{aligned}
\hat A_t-\hat A_t^*&=\sum_{l=0}^{K-1}(\gamma\lambda)^l(\gamma\epsilon_{t+l+1}-\epsilon_{t+l})\\
&=-\epsilon_t+\gamma(1-\lambda)\sum_{l=1}^{K-1}(\gamma\lambda)^{l-1}\epsilon_{t+l}
+\gamma(\gamma\lambda)^{K-1}\epsilon_{t+K}.
\end{aligned}
$$

将相邻误差的系数合并。三项依次是起点 baseline 误差、中间 bootstrap 误差、有限片段的尾值误差。有限片段的优势递推从尾部的零开始；这不意味着尾部状态的价值为零。

条件于动作前的历史和已固定的 critic，$-\epsilon_t$ 不依赖当前动作，可在 actor 的 score 条件期望里作为 baseline 消去。若先用同批动作与奖励拟合 critic，再计算这些样本的 baseline，即使代码停止了梯度，也要另查统计依赖。中间与尾部状态依赖此前动作，不能同样删除。$\lambda=1$ 消掉中间项；只有真正终止且终止价值固定为零时，尾项才消失。无终止的采样窗口不会获得这个额外条件。

数值核验：$K=2,\gamma=0.9$，起点和终点误差为零，中间误差为二。残差误差分别为 $1.8,-2$。$\lambda=0$ 的优势误差为 1.8；$\lambda=0.8$ 时为 $1.8-0.72\times2=0.36$；$\lambda=1$ 时为零。这是 critic 误差传播的解析示例，不是“λ 越大越好”的实验结论。

即使 critic 恰是真价值，采样优势仍然有方差。更长的回报引入更多随机奖励和动作。真实任务中还要考虑策略是否在片段内变化、截断是否正确、价值目标是否共享参数，以及批次优势归一化对有限样本更新的影响。

$$
\mathbb E\!\left[\frac1B\sum_{i=1}^{B}\nabla\log\pi(A_i)(R_i-\bar R)\right]=\left(1-\frac1B\right)\nabla J,\qquad\bar R=\frac1B\sum_iR_i
$$

一个可精确检查的统计依赖：B 个独立同分布的单步 bandit 样本，均减去含自身回报的样本均值。交叉样本项的 score 期望为零，自身项则留下 1/B 的缩减。

B=1 时，这种中心化把学习信号完全消掉。用其余样本的均值作 leave-one-out baseline，可以在这个独立 bandit 例子中消除缩减。整段轨迹中的时间步并不独立，除以样本标准差又增加了随机缩放，因此不能把这条简单修正式直接用于所有优势标准化。

若进一步要求逐样本即时更新，就无法等待整批优势反向计算。资格迹可以把部分时间权重变成前向递推的记忆，但网络在递推期间也会改变。固定权重下的前向—后向等价，不自动等于非线性、逐步更新时的同一算法。后续在线信用分配章正是继续处理这个差别。

<a id="lesson-algorithm"></a>

## 4 · VPG / A2C 的更新次序与 autograd

**算法：这是同步批更新的均匀 rollout 规约，配套 vpg_update 执行一次 actor 和一次 critic 更新。若改用完整回合的起点折扣目标，actor 样本权重按 $L_{\rm ep}$ 处理；平均 loss 的写法本身没有完成转换。**

1. 输入：固定 rollout 行数 B、gamma、lambda、actor 与 critic 优化器。
1. 本批开始保存 theta_old、phi_old；采样期间固定参数。
1. 收集 B 行新经验，保存 state、实际 action、reward、old logp、旧 value。
1. 在 reset 前保存最后观测与 next value，分别记录终止和序列边界。
1. 每个片段的优势 carry 从零开始，按 b、c 后向递推原始 GAE。
1. 先固定 critic target = 原始 GAE + 旧 value；需要时另建标准化 actor 权重。
1. 用当前策略计算 loss_actor = -mean(log_prob(实际 action) * stopgrad(actor 权重))。
1. 清空 actor 梯度，反传 loss_actor，执行一次 actor 更新。
1. 用当前 critic 计算 loss_critic = mean((V - stopgrad(target)) ** 2)，更新 critic。
1. 丢弃本批优化数据，保留未终止的环境活动，开始下一批采样。

离散动作必须对实际采样动作计算 `log_prob`，不能直接对最大 logit 求导。`torch.distributions.Categorical(logits=...)` 会进行稳定归一化。连续多维独立 Gaussian 的 `log_prob` 通常先返回各坐标，必须对动作维度求和，而不是把动作维度误当 batch。

对分离网络，固定优势后，actor loss 只对策略参数求导；critic 的固定标签平方损失只对当前预测求导。若共享 encoder，actor 与 critic 两个损失都会训练共享参数，价值回归也会改变动作分布。必须规定损失系数、求导路径和更新次序；actor 优势停止梯度后，共享表示仍会收到单独的价值损失梯度。

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

<a id="lesson-example"></a>

## 5 · 一条三步轨迹贯穿 GAE 与 PPO

回到三个状态的通关任务。每处选择前进才进入下一状态；选择另一动作立即以零奖励终止；从最后一处前进得到 $+1$ 并终止。旧策略在三处前进概率均为 $.5$。给定一条成功轨迹，奖励 $(0,0,1)$，旧 critic 估计 $(.2,.8,.4,0)$。这些是刻意不准确的给定估计；旧策略的真实价值其实为 $(.125,.25,.5,0)$。取 $\gamma=1,\lambda=.5$；终止后的价值和优势 carry 都为零。

$$
\begin{aligned}\delta&=(0+.8-.2,\ 0+.4-.8,\ 1-.4)=(.6,-.4,.6),\\\hat A_2&=.6,\quad\hat A_1=-.4+.5(.6)=-.1,\\\hat A_0&=.6+.5(-.1)=.55,\\\hat R^{\lambda}&=V_{\rm old}+\hat A=(.75,.7,1).\end{aligned}
$$

第一步的优势要等后两步残差反向传回；中间动作的估计优势为负，即使整条轨迹最后获得了正奖励。

若选择完整 rewards-to-go 标签，本例三个标签均为 $1$，不是 $(.75,.7,1)$。后者混合不同等待长度，仍含不准确的旧价值。将 $\lambda$ 改为 $1$，优势变成 $(.8,.2,.6)$，加回旧值后才与三个完整回报一致。若最后一行只是人工截断，则应使用最后观测的旧价值自举，但不把重置后下一行的优势接回来。

一次 rollout 的准备阶段到此结束：保存三个旧 $\log\pi(a\mid s)=\log(.5)$、原始优势 $(.55,-.1,.6)$ 和 critic 标签 $(.75,.7,1)$。更新时三者都停止梯度；当前 $\log\pi_\theta$ 与当前 $V_\phi$ 仍参与各自损失的求导。若需要标准化优势，另建 actor 权重副本，不能覆盖用于构造价值标签的原始量。下一章[直接用这三个样本计算 PPO 裁剪与独立评价](/zh/continual-rl/foundations/deep/trust-region/#lesson-example)。

$$
\frac{\partial\log\pi(a\mid s)}{\partial z_j}=\mathbf1[j=a]-\pi(j\mid s)
$$

softmax 的对数导数包含所有动作，不只是被选动作。若概率为 $(0.25,0.75)$，动作零的优势为 $2$，梯度上升方向为 $(1.5,-1.5)$，归一化会共同改变两个概率。

固定旧 value 和 next value，分别处理两类 mask，并输出原始优势与 critic target。

```python
def gae(rewards, values, next_values, terminated, boundaries,
        gamma=0.99, lam=0.95):
    n = len(rewards)
    if not all(len(x) == n for x in
               (values, next_values, terminated, boundaries)):
        raise ValueError('one next value and two masks per transition')
    advantages, carry = [0.0] * n, 0.0
    for t in reversed(range(n)):
        bootstrap, trace = masks(terminated[t], boundaries[t])
        delta = rewards[t] + gamma * bootstrap * next_values[t] - values[t]
        carry = delta + gamma * lam * trace * carry
        advantages[t] = carry
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def score_gradient(probabilities, action, advantage):
    # Derivative of A * log softmax(logits)[action] with fixed A.
    return [advantage * (float(i == action) - p)
            for i, p in enumerate(probabilities)]
```

<a id="lesson-code"></a>

## 6 · 从数值核验到实际 on-policy 训练

下载 [gae_ppo_walkthrough.py](/crl-code/tutorials/gae_ppo_walkthrough.py)，执行 `python3 gae_ppo_walkthrough.py --test`，可在不安装 PyTorch 的情况下复算本节全部分数。脚本的不可变 batch 分开保存旧 logp、原始 GAE 与固定 critic 标签；有限差分检查对照下一章正、负优势的四个 clipping 分支。

VPG 的一次 actor–critic 更新。优势和回归目标停止梯度；actor 更新后必须重新采样。

```python
def vpg_update(actor, critic, actor_optimizer, critic_optimizer, x, actions,
               advantages, returns):
    # One update with fresh on-policy data; gamma=1 episodic convention.
    distribution = Categorical(logits=actor(x))
    loss_actor = -(distribution.log_prob(actions)*advantages.detach()).mean()
    actor_optimizer.zero_grad()
    loss_actor.backward()
    actor_optimizer.step()
    prediction = critic(x).squeeze(-1)
    assert prediction.shape == returns.shape
    loss_critic = ((prediction-returns.detach())**2).mean()
    critic_optimizer.zero_grad()
    loss_critic.backward()
    critic_optimizer.step()
    return float(loss_actor.detach()), float(loss_critic.detach())
```

PPO 使用同一套 actor–critic 采样和 GAE；区别在下一章的策略 surrogate。

```python
def train_ppo(epochs=20, seed=0, batch_steps=128):
    torch.manual_seed(seed)
    actor, critic = mlp(6, 2), mlp(6, 1)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=.003)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=.01)
    env, initial_score = DeadlineChain(), evaluate(actor)
    observation = env.reset()
    log = {}
    for _ in range(epochs):
        xs, actions, logps, rewards, values, next_values, terminals = [], [], [], [], [], [], []
        for _ in range(batch_steps):
            with torch.no_grad():
                distribution = Categorical(logits=actor(observation))
                action = distribution.sample()
                logp, value = distribution.log_prob(action), float(critic(observation).item())
            xp, reward, terminal = env.step(int(action))
            with torch.no_grad():
                next_value = float(critic(xp).item())
            xs.append(observation); actions.append(action); logps.append(logp)
            rewards.append(reward); values.append(value); next_values.append(next_value); terminals.append(terminal)
            observation = env.reset() if terminal else xp
        # gamma=1 matches this finite-horizon, undiscounted task objective.
        advantages, returns = gae(rewards, values, next_values, terminals, terminals, gamma=1., lam=.95)
        log = ppo_update(actor, critic, actor_opt, critic_opt, torch.stack(xs),
                         torch.stack(actions), torch.stack(logps),
                         torch.tensor(advantages), torch.tensor(returns))
    return {'algorithm': 'PPO-Clip', 'seed': seed, 'environment_steps': epochs*batch_steps,
            'initial_greedy_return': initial_score, 'final_greedy_return': evaluate(actor),
            **log, 'scope': 'DeadlineChain only; full-batch updates'}
```

执行 python3 deep_textbook_lab.py demo 查看 GAE 与 score-gradient 的手算值；执行 python3 deep_textbook_train.py ppo --epochs 20 --seed 0 查看真实 on-policy 采样。代码在 rollout 内固定参数，并保存旧 log-probability。环境已经真终止后才 reset；若 batch 恰在 episode 中间结束，环境活动继续保留。

VPG 更新核用固定优势乘当前 log_prob，进行一次 actor 更新，再要求重新采样。上面的完整采样循环运行 PPO；官方 VPG 文件提供独立、完整的工程入口，配套 VPG 核用于逐项检查梯度与 detach。两份官方 buffer 都用 GAE 作 actor 权重、带尾值的 rewards-to-go 作 critic 标签，而本地 gae 返回有限 λ-return 标签，比较实现时要追踪实际计算。

$$
r_\theta=\exp(\log\pi_\theta(a\mid s)-\log\pi_0(a\mid s)),\qquad\left.\nabla_\theta r_\theta\right|_{\theta_0}=\left.\nabla_\theta\log\pi_\theta(a\mid s)\right|_{\theta_0}
$$

旧参数处概率比为一，故固定优势下，概率比目标和 log-probability 目标的第一步梯度相同。

更新后继续复用同批数据时，PPO 用固定旧概率作分母，并裁剪部分更新激励。下一章进一步区分动作概率换分布的等式、状态权重替代的近似和策略改变尺度；本章得到的优势与旧概率正是它读取的固定输入。

<a id="lesson-branches"></a>

## 7 · 估计误差与持续学习接口

- 优势误差：近似 critic 的中间和尾值误差随 λ 与片段长度进入更新；真价值与正确边界时，截断本身不必增加偏差。
- 概率错误：离散采样后重新计算另一动作的 log_prob，或连续动作裁剪后仍使用裁剪前 Gaussian 密度，都改变了估计器。
- 策略陈旧：旧轨迹反复用于裸 log-probability loss，不再是当前在策略梯度。
- 任务变化：critic、状态和策略可以不同速度适应，优势的符号可能暂时错误。

在 CRL 中，策略熵变小可能让有用数据不再出现；critic 误差又会影响行动更新。诊断时应分别测覆盖、优势误差和网络学习能力，而不是看到回报下降就统一归因于遗忘。

<a id="lesson-check"></a>

## 8 · 练习与答案

- 问：可以从回报中减去任意 baseline 吗？答：一个充分条件是它在当前动作前已经确定，并在 actor 求导时固定；一般须核对 baseline 与 score 的条件期望是否为零。
- 问：critic 的 loss 下降是否保证策略变好？答：不保证。它可能只改善高频无关状态，或在关键动作上的优势排序仍错误。
- 问：把 (.55,−.1,.6) 标准化后加回旧值，还能得到本例价值标签吗？答：不能。actor 的尺度处理不应改变 critic 的奖励单位。
- 练习：两步例子省去外层 γ 后，能否靠同一个全局步长同时修正两处梯度？答：不能；两处缩放不同。
- 练习：$\lambda=1$、片段未终止，critic 尾值有误差 $e$。答：相对于真值估计，仍留下 $\gamma^K e$ 的尾项；起点 baseline 误差可在所述条件下消去。
- 实验：固定三步轨迹，比较 λ=0、.5、1 的目标，再将最后一行改为 timeout 并给定尾值 .3。分别检查哪一条 mask 控制自举、哪一条阻止连接重置后样本。

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

- [Spinning Up · VPG](https://spinningup.openai.com/en/latest/algorithms/vpg.html)：用作算法与实现接口的对照；本章解释和配套小实验独立编写。

- [Sutton et al. · Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)：§1 与附录的起点折扣梯度：状态权重包含策略改变访问频率的后果。兼容逼近在第一册展开。

- [Schulman et al. · Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)：§2 Eq.6–8 的 gγ 与 γ-just，§3 Eq.11–18 的多步与 GAE。先核对论文目标和时间权重，再用其估计性质。

- [Pardo et al. · Time Limits in Reinforcement Learning](https://proceedings.mlr.press/v80/pardo18a.html)：§2–3：任务期限与训练截断的不同价值语义；前者需状态含剩余时间，后者保留尾值。

- [Spinning Up · vpg.py](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/vpg.py)：VPGBuffer.finish_path/get 与 compute_loss_pi/compute_loss_v：GAE、回归标签、标准化和均匀样本均值。

- [Spinning Up · ppo.py](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py)：PPOBuffer.finish_path：actor 优势与完整片段 rewards-to-go 标签独立计算。

- [Sutton & Barto · Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)：§9–11：函数逼近与离策略；§12：资格迹；§13.1 的短走廊与 §13.2–13.5 的策略梯度。对照各结论采用的策略类、采样分布与函数表示。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-policy-gradient#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-policy-gradient#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-deep-policy-gradient)
<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 怎样把较晚的反馈归给较早的计算？

多步回报定义用多远的未来构造目标。资格迹压缩过去的特征或梯度方向。前向与后向等价必须说明参数是在整个轨迹内固定，还是每一步改变。

函数逼近与深度方法：神经网络改变后，旧梯度不再等于用当前参数重算的梯度。递归状态还带来参数经过历史状态影响当前输出的路径，不能用一条普通 TD trace 代替。

持续学习中的研究问题：在每步计算有界的条件下，保留多少过去影响才有用？替换特征时，怎样处理与旧特征绑定的资格迹、优化器动量和元梯度？

[多步回报](https://yingwen.io/zh/continual-rl/foundations/tabular/multistep/) → [资格迹与等价条件](https://yingwen.io/zh/continual-rl/foundations/approximation/traces/) → [GAE 与 actor–critic](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/) → [在线信用分配](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)


### 可进一步检验的问题

- [05 · 远处的反馈应该怎样更新早先的决策与内部计算？](https://yingwen.io/zh/continual-rl/research/#research-temporal-credit)：GAE 将价值误差组成动作优势，其 bootstrap 与跨序列边界决定远处反馈怎样进入策略更新。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)
- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)
- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)

对应原始材料：Sutton & Barto §13.1–13.5；§12：资格迹与多步估计。本文为原创讲解，原书、论文与上游代码保留各自许可。
