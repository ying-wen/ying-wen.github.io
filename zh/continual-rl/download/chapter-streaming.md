# 流式强化学习：交互协议与更新稳定性

流式学习首先约束数据和计算：新经验到来后及时更新，不重放历史转移，并限制每步时间与持久内存。在这些约束下，本章研究怎样组织资格迹、尺度统计和更新规则，使逐步学习可实现、可检查。

## 本章内容

- 明确流式数据权限、单步计算与持久内存预算，区分流式协议、持续任务和不可重置生命期。
- 在流式协议中实现 TD(λ)，检查痕迹与终止时序；将时间信用分配与更新稳定性分开分析。
- 推导并比较 ObGD、StreamingOptimizer 与 Intentional Updates 的尺度机制、保证范围和失败条件。

<a id="problem-definition"></a>

## 本章的问题定义

每次真实经验到来后及时学习，不重放历史转移，持久内存和每步计算有明确预算；该协议独立于是否允许环境重置。

### 给定条件与符号

- 基础预测或控制目标、新经验因果流、持久字节预算和单步时间预算。
- 网络、资格迹、尺度统计与更新控制器；真实终止清理规则。

### 需要求解的对象

满足协议且可持续执行的学习更新；更新尺度局部可检查，长期预测或控制质量另行评价。

### 信息与数据权限

允许保存参数、优化器状态、资格迹和在线统计，不保存供再次训练的历史转移。当前奖励统计、目标计算与更新顺序都需明确。

$$
M_t=U(M_{t-1},\xi_t),\qquad \operatorname{bytes}(M_t)\le B,\qquad C_t\le C_{\max}
$$

$\xi_t$ 是刚收到的经验，$M_t$ 是全部持久学习状态，$U$ 是更新映射，$B$ 为内存预算；$C_t$ 是包括动作选择、学习与规划的本步总计算，$C_{\max}$ 为上限。历史转移不得被重取训练。基础任务目标沿用预测或控制规格，这些资源约束不是新的奖励函数。

### 成立条件与解的含义

- 预算包含网络、优化器、迹和统计；batch size、无GPU或常数缓存大小不单独证明协议。
- 局部输出线性化、范数代理和逐坐标位移界的前提分别声明；终止不自动清除长期尺度统计。

判断准则：日志证明每条转移使用权限与延迟，持久内存不随寿命增长；手算更新/尺度边界正确；长期实验报告非有限更新、恢复及外部收益，参数位移界不替代收益。

### 适用边界

- 小参数更新不保证TD误差下降、策略改善或单生命期安全。
- 不把观测归一化和自适应统计当作无成本且不影响函数的预处理。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 改变信息或数据协议 · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)：流式预算允许迹这种统计机制，却限制长窗口、重放和完整反传。

- 组合不同学习问题 · [元学习与学习规则的适应](https://yingwen.io/zh/continual-rl/algorithms/meta/)：预设尺度控制可逐步执行；依据后续表现学习规则还需额外敏感度和资源。

- 改变信息或数据协议 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：完整控制比较需加入流式资源约束；无重置与无重放是两项独立权限。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

没有回放可反复纠正当前过冲；奖励、特征和迹尺度变化会放大单步更新。

### 本章的核心思路

用可递推统计保留信用并按声明的局部尺度限制更新，分别检验协议和任务表现。

1. [以迹替代保存训练历史](#lesson-derive)：因为预算禁止重取转移，资格迹保存过去方向统计并在新误差到来时更新。

2. [定位尺度与过冲](#lesson-scale)：因为同一名义步长随特征尺度产生不同输出变化，先在线性例上推有效步长，再辨认ObGD的范数代理条件。

3. [选择可检查的尺度控制器](#lesson-current-streamx)：因为整体代理与逐坐标边界处理的量不同，对照2026控制器的参数位移界与Intentional Updates的输出线性化。

4. [按输出变化解释学习尺度](#lesson-output-steps)：因为参数步长未直接指定预测改变，Intentional方法利用梯度内积决定局部尺度，非线性与统计近似仍需测量。

结论与条件：有限增量和正稳定项下可检验逐坐标位移界；线性固定标签的输出关系只约束本次更新。Stream-X/Intentional深度组合不提供无条件长期收敛。

### 相关方法改变了什么

- ObGD：以TD误差与迹范数做廉价整体缩放，依赖代理尺度解释。

- 逐坐标StreamingOptimizer：用衰减最大增量提供坐标位移边界，不保证输出收缩。

- Intentional Updates：按局部输出变化选尺度，含预条件、迹和在线平均近似。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 价值函数

$v(s)$ 预测从状态 $s$ 出发按固定策略累积的折扣奖励。学习器根据不断到来的经验修正这个预测。

### 半梯度 TD

误差 $\delta=r+\gamma v(s')-v(s)$ 中的两个预测使用同一份旧参数；更新方向只对当前预测求导。

### 资格迹

与参数同维的向量，积累过去预测对参数的敏感性。它保存信用分配所需的统计量，而不保存原始样本。

<a id="lesson-setting"></a>

## 1 · 流式学习的数据与计算约束

机器人每 20 毫秒拿到一条观测并必须作出下一步决定。若学习需要等一批数据、回放百万帧或者反复重算整段历史，就不符合这个实时任务的预算。这里采用严格的一步流式协议：每条新 transition 到来后立即更新，不保存原始 transition 供以后重复采样；允许维护固定大小的参数、痕迹、运行统计与 agent state。

这并不等于“算法没有记忆”：资格迹、RNN 隐状态与优化器二阶矩都占内存，必须记账。也不等于“环境永不变化”或“环境没有终止”。本章先研究固定策略预测，随后区分 Sarsa / Q / actor–critic。网络是否深、目标是否平均奖励、世界是否单生命期，是另外三个独立选择。

本章讨论的是学习协议及其数值实现。时间信用分配决定当前反馈影响哪些过去预测或决策；学习规则的适应则用经验改进步长、目标或更新方式。资格迹是可在流式预算内使用的一种信用机制，在线元梯度也可能采用流式实现，但两者都不构成 streaming 的定义。

| 约束或对象 | 保存内容或度量 | 边界 |
| --- | --- | --- |
| 严格 streaming 数据 | 当前样本；固定维参数和统计 | 不重放旧转移，不等待未来样本组成训练 batch |
| 资格迹 | 每个参数一个历史敏感性统计 | 不能无损恢复历史样本 |
| 实时计算预算 | 一次或固定少数次前后向 | 限制每步计算与历史回溯 |
| 学习评价 | 实际每步延迟、内存、累计奖励 | GPU 吞吐量不能单独代表实时控制能力 |

把 DQN 的 replay capacity 改成 1 是可用的诊断基线，却未必保留 DQN 的稳定性。原方法依靠的样本混合、target network 和批平均同时改变了。研究问题应是“哪些机制能在这个协议下工作”，而不是只换一个 buffer 参数。

<a id="lesson-derive"></a>

## 2 · 流式协议中的资格迹

时间信用分配研究延迟误差应当影响哪些过去预测、动作或参数。它既可以使用在线资格迹，也可以借助保存的轨迹与反向传播，因此不等于流式学习。本节只保留实现流式 TD 所需的前向—后向关系与边界；本书《时间信用分配》章独立讨论更完整的算法条线。

$$
G_t^{(n)}=\sum_{k=1}^{n}\gamma^{k-1}R_{t+k}+\gamma^n v(S_{t+n})
$$

n-step target 比一步 TD 看得更远，却要等 n 步才能构造。有限回合末尾的 bootstrap 为零。

先固定价值参数。将 $\delta_t,\gamma\delta_{t+1},\ldots$ 相加时，相邻两项中的状态价值相消，只留下首项 $-v(S_t)$ 和末尾的 bootstrap。这一望远镜求和说明了多步回报与 TD 误差序列的关系。

$$
G_t^{(n)}-v(S_t)=\sum_{k=0}^{n-1}\gamma^k\delta_{t+k}
$$

等式要求右侧所有 δ 使用与左侧同一套固定价值。在线每步变参数时，普通 accumulating trace 与这个冻结参数前向视角只有近似对应；不能把有限步长下的等价说成精确。

$$
\begin{aligned}G_t^\lambda&=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)},\\G_t^\lambda-v(S_t)&=\sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k}.\end{aligned}
$$

无限混合式先取 $0\le\lambda<1$，并要求回报和右侧级数收敛；$\lambda=1$ 用相应极限或完整回报定义，不直接代入第一行的零乘无限和。有限回合让最后一个 Monte Carlo return 承接剩余权重。$\lambda=0$ 退化为一步 TD，接近 1 则使较远的 TD error 参与信用分配。

前向视角考虑未来哪些 $\delta$ 影响当前预测；后向视角在误差到来时，给过去相关的预测分配信用。交换求和顺序，就需要保存过去梯度按 $\gamma\lambda$ 衰减后的总和。对于线性 $v=w^\top x$，梯度就是特征 $x$；对于网络则使用各时刻参数下的输出梯度。

$$
\begin{aligned}z_t&=\gamma_t\lambda z_{t-1}+\nabla_w v(S_t;w_t),\\\delta_t&=R_{t+1}+\gamma_{t+1}v(S_{t+1};w_t)-v(S_t;w_t),\\w_{t+1}&=w_t+\alpha_t\delta_t z_t.\end{aligned}
$$

γt 属于到达 St 的转移，用来衰减旧痕迹；γt+1 属于当前刚完成的转移，用来构造 target。常数 γ 时区别被隐藏，终止和 GVF 的状态依赖 discount 会把错误暴露出来。

若 $S_{t+1}$ 真正终止，$\gamma_{t+1}=0$，不再 bootstrap；但本次奖励仍通过 $z_t$ 分给本回合的过去状态。痕迹在完成终点更新后、新回合开始前清空。若只是时间限制截断而真实过程仍会继续，价值目标保留 bootstrap；若随后环境另行重置，资格迹则不能跨越这次非连续转移。

**算法：线性 accumulating TD(λ)；终点奖励在清迹之前更新**

1. 初始化参数 $w$、$z=0$，选择 $\lambda$ 与 $\alpha$
1. 每步接收 $(x,r,x')$ 和 $\gamma_{\rm current},\gamma_{\rm next}$：
  1. 保存旧参数 $w_{\rm old}$
  1. 计算 $v=x^\top w_{\rm old}$、$v'=x'^\top w_{\rm old}$
  1. 计算 $\delta=r+\gamma_{\rm next}v'-v$
  1. 更新 $z\leftarrow\gamma_{\rm current}\lambda z+x$
  1. 更新 $w\leftarrow w_{\rm old}+\alpha\delta z$
1. 新回合开始时令 $z=0$

| 方法 | 保存的痕迹 | 关键限制 |
| --- | --- | --- |
| TD(λ) | 过去 ∇v 的衰减和 | 普通在线 accumulating 版本不是任意步长下精确前向等价 |
| Momentum SGD | 过去 loss gradient 的衰减和 | 每步历史误差已乘在各自梯度上；不同于当前 δ 乘全部痕迹 |
| True online TD(λ) | Dutch trace 和预测差修正 | 对指定线性在线 λ-return 有精确等价；不能直接宣称任意深网同样成立 |

<a id="lesson-scale"></a>

## 3 · 数值尺度与更新过冲

考虑一维固定标签回归：预测为 $wx$，误差为 $e=y-wx$，更新为 $w_+=w+\alpha ex$。代入后得到 $e_+=e(1-\alpha x^2)$。即使 $\alpha=0.1$，只要 $x=10$，有效步长也达到 10，误差变号且放大 9 倍。步长的作用取决于特征尺度。

$$
\xi=\frac{e-e_+}{e}=\alpha x^2,\qquad 0<\xi\leq1\ \Longrightarrow\ \text{本次固定标签误差不越过零点}
$$

这是当前样本的代数性质，不等于新样本泛化、长期控制稳定性或单生命期安全。e=0 时用更新为零处理，不做 0/0。

TD 更新会同时改变当前预测和下一状态 bootstrap。线性情况下直接展开，得到 $\xi=\alpha z^\top(x-\gamma x')$，而非 $\alpha x^\top x$。它可以为负，即同一样本的 TD 误差变大；这体现半梯度 TD 与固定标签回归的区别。

$$
\delta_+=\delta-\alpha\delta\,z^\top(x-\gamma x'),\qquad \xi=\alpha z^\top(x-\gamma x')
$$

式中 z 在这一步更新中视为固定。深网使用局部线性化后只能得到近似；若再用更便宜的范数代理，还需要额外梯度尺度假设。

$$
\begin{aligned}M&=\alpha\kappa\max(|\delta|,1)\|z\|_1,\\\alpha_{\rm used}&=\frac{\alpha}{\max(1,M)},\\w_+&=w+\alpha_{\rm used}\delta z.\end{aligned}
$$

这是 ObGD 的代数更新形式，κ 是安全缩放系数。它用廉价统计抑制大更新；对任意网络、任意输入并不存在无条件不越界保证。零痕迹时分母至少 1，不会除零。

一个尺度反例是 $y=1$、$x=10$、$w=0$、$\alpha=0.1$、$\kappa=2$。此时 $z=10$、$M=2$，所以 $\alpha_{\rm used}=0.05$，更新后 $w_+=0.5$，预测为 5，仍然超过目标 1。这里梯度尺度不满足范数代理所需条件，说明该更新的实际性质依赖输入与网络尺度。

在线输入统计采用 Welford 递推：先用旧均值计算差，再更新新均值，最后用新旧两侧差的乘积更新 M2。所有统计只能使用已经到来的观测。全数据预先归一化会泄露未来；累积统计在漂移中会越来越迟钝，指数滑动统计则是另一种估计器，需要独立比较。

$$
\begin{aligned}n&\leftarrow n+1,\quad d=x-\mu_{\rm old},\\\mu&\leftarrow\mu_{\rm old}+d/n,\\M_2&\leftarrow M_2+d(x-\mu),\quad \sigma^2=M_2/(n-1).\end{aligned}
$$

n=1 时必须定义方差初始化和 ε，避免第一步除零。归一化统计的改变也在改变价值函数输入坐标，因此不是与学习完全无关的预处理。

<a id="lesson-streamx"></a>

## 4 · Stream-X：流式深度控制的组件组合

Stream-X 是将同一组稳定化机制应用到不同基础学习器的方法。2024 版使用 ObGD、资格迹、在线观测与奖励尺度统计、无可学习仿射参数的 LayerNorm，以及稀疏初始化。2026 修订版保留这一组织方式，但更换了更新控制器并修改初始化；两版的优化器需要分别理解。

$$
\bar o_{t,i}=\frac{o_{t,i}-\mu_{t,i}}{\sqrt{\sigma_{t,i}^2+\varepsilon}},\qquad U_t=\gamma U_{t-1}+r_t,\qquad \bar r_t=\frac{r_t}{\sqrt{\operatorname{Var}_t(U)+\varepsilon}}
$$

观测按各坐标中心化并缩放；奖励只缩放，不减均值。奖励迹使用已到来的奖励，是未来 return 尺度的因果代理。2026 实现先更新统计并缩放当前奖励，再在 episode 边界清零奖励迹；累计均值和方差保留。

LayerNorm 在单个观测的一层预激活上计算均值和方差，因而不需要 batch。稀疏初始化则改变参数更新对不同输入的共享程度。二者不相同：前者控制层内尺度，后者改变初始连接。2024 的逐层固定稀疏率在低输入维度下可能将一层全部置零；2026 的 layer-aware 初始化分配全网非零连接预算，复现时应连同 sparse_init.py 一起使用。

| 分支 | 当前误差或方向 | 必须保留的时序 |
| --- | --- | --- |
| Stream TD(λ) | 固定策略 δ=r+γv′−v | 价值输出梯度累积后再 ObGD |
| Stream Sarsa(λ) | δ=r+γq(s′,a′)−q(s,a) | 下一动作来自实际目标策略 |
| Stream Q(λ) | max 下一动作目标；Watkins 痕迹语义 | 执行非贪心动作时截断此前不再匹配贪心延续的痕迹 |
| Stream AC(λ) | critic δ 配合 ∇logπ 的 actor trace | 行为动作按更新前策略采样；actor/critic 采用同一旧 δ，熵项另核对 |

Stream-X 的控制实验允许环境终止与重置；2026 修订还研究了变化摩擦条件、机器人与资源受限设备。其结果需要按照各自的数据与环境协议解释。一次交互更新、没有经验重放，并不意味着所有实验都采用不可重置的单生命期。

作者 2024 实现先对 $-v$ 和负 log-probability 求梯度，再由 optim.py 做减法；本文采用正输出梯度加更新，两种符号约定一致。应避免先对平方 TD 损失求导，再把 $\delta$ 额外乘入优化器。原 actor 的熵项还含 $\operatorname{sign}(\delta)$，经本步乘误差后，其即时熵方向与 $|\delta|$ 成比例；这与直接加一个固定权重熵梯度不同。

<a id="lesson-current-streamx"></a>

## 5 · 2026 更新控制器：从整体缩放到逐坐标边界

ObGD 用全网 $\|z\|_1$ 形成一个共同步长。若许多坐标同时具有非零痕迹，网络越大，每个坐标获得的更新可能越小。修订版 StreamingOptimizer 转而对完整增量 $u_t=\delta_t z_t$ 的每个坐标维护衰减最大值：

$$
v_{t,i}=\max\{\beta v_{t-1,i},|u_{t,i}|\},\qquad w_{t+1,i}=w_{t,i}+\alpha\frac{u_{t,i}}{v_{t,i}+\varepsilon}
$$

初始化最大值为零，保留比例满足 $0\leq\beta<1$。这里的 $v_{t,i}$ 是优化器统计，不是状态价值；代码中名为 max_v。更新后若 episode 结束，只清资格迹，保留这个尺度统计。

$$
|\Delta w_{t,i}|=\alpha\frac{|u_{t,i}|}{v_{t,i}+\varepsilon}\leq\alpha\frac{|u_{t,i}|}{|u_{t,i}|+\varepsilon}<\alpha
$$

有限增量、$\alpha>0$ 与 $\varepsilon>0$ 下，单坐标位移严格小于名义步长。这个结论不需要前一节的 TD 梯度尺度假设，但它只约束参数位移，并不保证 TD 误差减小、策略改善或无长期发散。

**算法：StreamingOptimizer 的 TD 形式，对应作者 2026 optimizer.py**

1. 初始化资格迹 $z=0$ 和逐坐标最大值 $v=0$
1. 每条转移：
  1. 用旧参数计算 $\delta$ 和输出梯度 $g$
  1. 更新 $z\leftarrow\gamma\lambda z+g$
  1. 形成完整增量 $u\leftarrow\delta z$
  1. 更新 $v\leftarrow\max(\beta v,|u|)$（逐坐标）
  1. 更新 $w\leftarrow w+\alpha u/(v+\varepsilon)$
  1. 若将开始新 episode，清零 $z$，保留 $v$

手算一个尖峰：上一最大值为 $2$，$\beta=0.9$，新增量为 $100$，则新最大值立即变成 $100$，位移约为 $\alpha$。下一步增量降为 $1$ 时，最大值仍为 $90$，位移约为 $\alpha/90$。所以它立即限制尖峰，而后逐步恢复更新尺度。

2026 的 Stream-RAC 还采用重参数化 actor：通过动作对 soft Q 目标求导，而不是把 TD 误差乘 log-policy 梯度。作者实现每步清 actor trace，critic 则使用 soft next-action target 与资格迹。它仍没有 replay 和 target network，但不能当作 Stream-AC 只换一个函数名。

<a id="lesson-output-steps"></a>

## 6 · Intentional Updates：按输出变化选择步长

Intentional-AC 属于 ICML 2026 的 Intentional Updates for Streaming Reinforcement Learning。其起点是输出变化：已选定参数方向 $d$，希望标量输出 $y(w)$ 改变 $\Delta$。一阶展开给出 $y(w+\alpha d)-y(w)\approx\alpha\nabla y^\top d$，因此可以反解步长。这里不追踪参数对过去步长的敏感度，也不建立元参数的资格迹；“步长依数据变化”与“元梯度学习步长”是不同机制。

$$
\alpha\approx\frac{\Delta}{\nabla y(w)^\top d},\qquad d=\delta\nabla V_w(s),\qquad\Delta=\eta\delta\quad\Longrightarrow\quad\alpha=\frac{\eta}{\|\nabla V_w(s)\|^2}
$$

令 $\delta=r+\gamma V_w(s')-V_w(s)$，这里目标 $r+\gamma V_w(s')$ 在本次更新中固定。在线性 TD(0)、梯度非零且不加裁剪时，当前预测向这个固定目标移动恰好 $\eta\delta$。对非线性网络，它只是局部近似；对重新计算过 bootstrap 的新 TD 误差，也没有相同收缩等式。

例如 $w=0.5$、当前特征 $x=2$、下一特征 $x'=1$、奖励 $r=1.5$、$\gamma=0.9$、$\eta=0.4$。旧目标为 $1.95$，误差为 $0.95$；步长为 $0.4/4=0.1$，更新后 $w^+=0.69$。到旧目标的误差是 $1.95-1.38=0.57=0.6\times0.95$，但重新计算 bootstrap 得到 $1.5+0.9\times0.69-1.38=0.741$。区分两个误差是理解其保证范围的关键。

资格迹使更新方向包含过去梯度，逐坐标缩放又改变了几何。记当前输出梯度为 $g_t$，逐坐标缩放矩阵为 $D_t=\operatorname{diag}(\rho_t)$，资格迹为 $z_t=qz_{t-1}+g_t$，且 $q=\gamma\lambda<1$。对过去预测变化的加权平方和应用 Cauchy–Schwarz，可得到涉及 $z_t^\top D_tz_t$ 与历史梯度范数的尺度。作者实现用偏差校正的指数平均跟踪后者，而不是储存所有过去梯度。

$$
\begin{aligned}\nu_t&=\beta_2\nu_{t-1}+(1-\beta_2)g_t^2,\qquad \rho_t=\left(\sqrt{\nu_t/(1-\beta_2^t)}+\varepsilon\right)^{-1},\\\sigma_t&=g_t^\top D_tg_t,\qquad s_t=qs_{t-1}+(1-q)\sigma_t,\qquad\bar\sigma_t=\frac{s_t}{1-q^t},\\\alpha_t&=\frac{\eta}{\max\{\sqrt{\bar\sigma_t\,z_t^\top D_tz_t},\varepsilon\}},\qquad w^+=w+\alpha_t\widetilde\delta_tD_tz_t.\end{aligned}
$$

平方、开方和倒数均逐坐标进行。$\bar\sigma$ 是平均而非未归一化的历史和，不能将推导中的历史求和与实现直接等同；$D_t$ 还随时间改变，因此历史统计量也是在线代理。分母只确定局部更新尺度，不给任意长期训练轨迹提供收敛保证。

完整实现先用原始 $\delta^2$ 的偏差校正指数平均给出裁剪幅度，得到 $\widetilde\delta$。critic 使用它更新价值；actor 再除以裁剪后误差绝对值的运行平均，以限制奖励尺度的影响。对于无迹 actor，$g=\nabla\log\pi(a\mid s)$ 时，目标是让已采样动作的对数概率变化约为 $\eta\widetilde A/\overline{|A|}$。这既不是精确的全分布 KL trust region，也不保证每个动作概率都只改变固定百分比；动作相关步长还会重加权期望策略梯度。

$$
\begin{aligned}c_t&=\beta_c c_{t-1}+(1-\beta_c)\delta_t^2,\qquad B_t=C\sqrt{c_t/(1-\beta_c^t)},\\\widetilde\delta_t&=\operatorname{clip}(\delta_t,-B_t,B_t),\\a_t&=\beta_a a_{t-1}+(1-\beta_a)|\widetilde\delta_t|,\qquad \widehat A_t=\frac{\widetilde\delta_t}{\max\{a_t/(1-\beta_a^t),10^{-12}\}}.\end{aligned}
$$

critic 使用 $\widetilde\delta_t$，actor 使用 $\widehat A_t$。作者默认 $C=20$、$\beta_c=\beta_a=0.9998$。统计量包含当前误差，并采用裁剪后的误差更新 actor 尺度；交换顺序会得到不同算法。

$$
g_t^{\rm actor}=\nabla_\theta\left[\log\pi_\theta(a_t\mid s_t)+\xi\,\mathcal H(\pi_\theta(\cdot\mid s_t))\,\operatorname{sg}\!\left(\operatorname{sign}(\delta_t)\right)\right]
$$

这是作者 actor 实际送入优化器的梯度。TD 误差由更新前的 critic 计算，符号作为常数，不沿 critic 反传；裁剪与正尺度归一化保持符号，因此也可用处理后优势的符号。该梯度一起进入 RMSProp 统计与资格迹，随后再乘归一化优势。无迹时，符号修正使熵贡献乘上优势的绝对值；有迹时还混合了过去的熵梯度，不能据此断言每次更新都增加当前状态熵。它不同于在最终更新中另加一项不经过资格迹的熵梯度。

**算法：Intentional optimizer 的运算顺序；网络与行为策略由 actor–critic 主循环提供**

1. 用旧 critic 计算 $\delta$，固定它的符号。
1. critic 使用 $g=\nabla V(s)$；actor 使用 $g=\nabla[\log\pi+\xi\mathcal H\,\operatorname{sign}(\delta)]$。
1. 二者有独立优化器状态：
  1. 更新梯度二阶矩，得到 $D$。
  1. 更新资格迹 $z$、当前梯度范数 $\sigma$ 和偏差校正平均 $\bar\sigma$。
  1. 根据 $\sqrt{\bar\sigma\,z^\top Dz}$ 计算步长。
  1. 更新误差裁剪统计，得到 $\widetilde\delta$。
  1. actor 额外更新 $|\widetilde\delta|$ 的尺度统计并归一化。
  1. 执行参数更新；若是真正 episode 边界，在本次更新后清空资格迹。

作者默认归一化顺序的标准库实现，以及能精确分析的未归一化 TD(0) 特例。

```python
@dataclass
class IntentionalStep:
    """Vector optimizer matching the full author's default normalization order.

    The caller supplies +grad V for the critic. For the author's actor use
    +grad_theta[log pi(a|s) + xi*H(pi(.|s))*sign(delta)], with the sign of the
    pre-update critic's delta held constant. Entropy enters the trace and the
    RMSProp statistics with the score gradient; it is not added after the step.
    Clipping and positive-scale normalization preserve delta's sign. Omitting
    this sign reverses the immediate entropy contribution for negative delta.
    With nonzero traces, past entropy terms may have different signs; this is
    not a guarantee that each update increases the current state's entropy.
    'policy' additionally normalizes the clipped TD error by its running L1 scale.
    No environment, neural network, or author benchmark is reproduced here.
    """
    size: int
    eta: float = 0.5
    q: float = 0.72  # gamma*lambda, must be in [0, 1)
    beta2: float = 0.999
    beta_clip: float = 0.9998
    beta_norm: float = 0.9998
    clip_mult: float = 20.0
    policy: bool = False
    t: int = 0
    sigma: float = 0.0
    delta_sq: float = 0.0
    delta_abs: float = 0.0
    v: list = field(default_factory=list)
    e: list = field(default_factory=list)

    def __post_init__(self):
        if self.size < 1 or not 0 <= self.q < 1:
            raise ValueError('positive size and 0 <= q < 1 required')
        if any(not 0 <= b < 1 for b in (self.beta2, self.beta_clip, self.beta_norm)):
            raise ValueError('EMA decays must lie in [0,1)')
        self.v, self.e = [0.0] * self.size, [0.0] * self.size

    def step(self, g, delta, reset=False):
        if len(g) != self.size or not all(math.isfinite(x) for x in [*g, delta]):
            raise ValueError('finite gradient with declared dimension required')
        self.t += 1
        self.v = [self.beta2 * v + (1 - self.beta2) * gi * gi
                  for v, gi in zip(self.v, g)]
        rho = [1 / (math.sqrt(v / (1 - self.beta2 ** self.t)) + 1e-8)
               for v in self.v]
        sigma_now = sum(gi * gi * ri for gi, ri in zip(g, rho))
        self.e = [self.q * e + gi for e, gi in zip(self.e, g)]
        self.sigma = self.q * self.sigma + (1 - self.q) * sigma_now
        sigma_bc = self.sigma / (1 - self.q ** self.t)
        trace_norm = sum(e * e * ri for e, ri in zip(self.e, rho))
        alpha = self.eta / max(math.sqrt(sigma_bc * trace_norm), 1e-8)
        self.delta_sq = self.beta_clip * self.delta_sq + (1 - self.beta_clip) * delta**2
        cap = self.clip_mult * math.sqrt(self.delta_sq / (1 - self.beta_clip ** self.t))
        safe = math.copysign(min(abs(delta), cap), delta)
        if self.policy:
            self.delta_abs = self.beta_norm * self.delta_abs + (1 - self.beta_norm) * abs(safe)
            norm = self.delta_abs / (1 - self.beta_norm ** self.t)
            safe /= max(norm, 1e-12)
        update = [alpha * safe * ri * e for ri, e in zip(rho, self.e)]
        if reset:
            self.e = [0.0] * self.size
        return dict(update=update, alpha=alpha, safe_delta=safe, sigma=sigma_bc)


def intentional_td0(w, x, reward, next_x, gamma, eta):
    """Unpreconditioned, unclipped TD(0): freeze the old bootstrap target."""
    target = reward + gamma * dot(w, next_x)
    delta = target - dot(w, x)
    norm_sq = dot(x, x)
    if norm_sq == 0:
        return list(w), target, delta
    new_w = [wi + eta * delta * xi / norm_sq for wi, xi in zip(w, x)]
    return new_w, target, delta
```

偏差校正中的 $t$ 从优化器第一次更新开始计数，初始二阶矩、误差统计和梯度范数统计均为零。episode 结束时，在完成本次更新之后清空 $z$；其余幅度统计与更新计数保留。actor 与 critic 分别维护这些状态，不能共用一个尺度估计。

比较这些方法时，应保持网络、资格迹、奖励处理和交互预算相同：固定步长提供基线；Metatrace 增加对学习过程的敏感度与元信用分配；Intentional updates 改变每步输出尺度；Stream-X 则是一套为 streaming 深度控制设计的组件组合。任何一条的正结果都不足以单独判定其他组件无关。

<a id="lesson-example"></a>

## 7 · 两步链的流式执行

状态特征为 $x_0=(1,0)$、$x_1=(0,1)$；转移 $s_0\to s_1$ 的奖励为 0，$s_1\to\mathrm{terminal}$ 的奖励为 1。初始化 $w=z=(0,0)$，取 $\gamma=0.9$、$\lambda=0.8$、$\alpha=0.1$。第一步 $\delta_0=0$、$z_0=(1,0)$，权重不变。第二步 $\gamma_{\rm next}=0$，但 $\gamma_{\rm current}=0.9$，因此 $z_1=(0.72,1)$、$\delta_1=1$；最终 $w=(0.072,0.1)$。

若 $\lambda=0$，同一趟只更新第二个状态，得到 $w=(0,0.1)$；起点要等下次访问才利用新的后继预测。资格迹使用同一个 $\delta$ 乘保存的历史敏感性，将一次误差信号分配给过去，无需重放终点样本。

两种常见错误可以直接手算定位：在 terminal 更新前清零 z，会丢掉起点的 0.072；把当前 terminal 的 γnext=0 用来衰减过去痕迹，也会得到相同错误。测试把这两个 discount 分为独立参数，避免常数 γ 隐藏索引问题。

<a id="lesson-code"></a>

## 8 · 机制实验与作者实现

线性 TD(λ)、Welford 与 2024 ObGD 的可执行机制实验。

```python
@dataclass
class Welford:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def observe(self, x):
        self.count += 1
        old_delta = x - self.mean
        self.mean += old_delta / self.count
        self.m2 += old_delta * (x - self.mean)
        return self.mean, self.variance

    @property
    def variance(self):
        return self.m2 / (self.count - 1) if self.count > 1 else 1.0


def td_lambda_step(w, trace, x, xp, reward, gamma_current,
                   gamma_next, lam=0.8, alpha=0.1, kappa=None):
    """Linear TD: gamma_current decays past trace; gamma_next bootstraps.
    kappa enables the ObGD algebra, NOT a general nonlinear safety theorem.
    """
    if not (len(w) == len(trace) == len(x) == len(xp)):
        raise ValueError("dimension mismatch")
    v = sum(a*b for a, b in zip(w, x))
    vp = sum(a*b for a, b in zip(w, xp))
    error = reward + gamma_next * vp - v
    z = [gamma_current * lam * old + feature
         for old, feature in zip(trace, x)]
    step = alpha
    if kappa is not None:
        mass = alpha * kappa * max(abs(error), 1.0) * sum(map(abs, z))
        step = alpha / max(1.0, mass)  # defined also when z == 0
    return [a + step * error * b for a, b in zip(w, z)], z, error, step


def streaming_demo():
    outcomes = {}
    for lam in (0.0, 0.8):
        w, z, _, _ = td_lambda_step([0., 0.], [0., 0.], [1., 0.],
                                    [0., 1.], 0, 0, 0.9, lam)
        w, z, _, _ = td_lambda_step(w, z, [0., 1.], [0., 0.],
                                    1, 0.9, 0, lam)
        outcomes[str(lam)] = [round(v, 6) for v in w]
    stats = Welford()
    for x in (1., 2., 3.):
        stats.observe(x)
    bounded = td_lambda_step([0.], [0.], [10.], [0.], 1, 0, 0,
                             alpha=0.1, kappa=2)
    print("streaming", {"two_step_credit": outcomes,
          "mean_variance": (stats.mean, stats.variance),
          "obgd_illustration_weight": bounded[0],
          "warning": "gradient 10 violates the <=1 heuristic bound"})
```

仅标准库；包含 ObGD 尺度条件不满足时的过冲反例

```sh
python lifelong_algorithms_lab.py streaming
python lifelong_algorithms_lab.py test
```

输出中 $\lambda=0$ 为 $[0,0.1]$，$\lambda=0.8$ 为 $[0.072,0.1]$；输入 $1,2,3$ 的均值为 $2$、样本方差为 $1$。尺度反例的更新后权重为 $0.5$。下载脚本覆盖这些机制；深度控制实验使用下方作者实现。2024 分支对应 ObGD，2026 分支对应 StreamingOptimizer，二者的配置不能混用。

Intentional 输出尺度实验及回归测试；仅标准库

```sh
python3 meta_frontier_lab.py intentional
python3 meta_frontier_lab.py test
```

IntentionalStep 实现梯度缩放、资格迹、误差裁剪和 actor 误差归一化。它接收网络梯度与旧 TD 误差，不包含神经网络和环境；无迹线性特例用于检查输出变化，带迹与熵项的测试用于检查更新顺序。完整控制结果需使用下方作者训练入口。

- 诊断量一：每步 |δ|、trace 的 L1/L2 norm、实际参数更新 norm 和 αused/α，分别看是目标、历史累积还是步长造成尖峰。
- 诊断量二：原始及归一化观测/奖励的分位数，LayerNorm 前后的激活统计；不要只记录平均值。
- 诊断量三：峰值内存、单步 p50/p95/p99 延迟；用严格相同计算预算比较是否允许 replay 的方法。
- 消融顺序：线性 TD0→线性 traces→网络 TD0→network traces→逐个稳定化组件；每次只增加一个新的故障来源。

<a id="lesson-branches"></a>

## 9 · 从局部更新到持续学习问题

| 实际问题 | 首个可解释基线 | 不应跳过的下一步 |
| --- | --- | --- |
| 短期信用难 | TD(λ) / Sarsa(λ) | λ 与 trace 截断；固定参数前向/后向核对 |
| Off-policy 预测发散 | 重要性比 TD 与梯度 TD 对照 | 覆盖、trace 方差与投影目标；不能只减步长 |
| 输入数值尺度改变 | 在线统计和线性尺度实验 | 统计漂移是否改变目标，是否有未来泄露 |
| 深网每步不稳定 | 完整 Stream-X 组件消融 | 初始化、网络、优化器、归一化作为整体记录 |
| 平均收益目标 | differential error 与独立 rate learner | γ=1 的 trace 管理、平均奖励理论不能照搬折扣保证 |

研究“仅一次真实交互”时，还要把动作延迟带来的环境状态变化纳入协议。若一次复杂更新占用了十个控制周期，却仍只按一条 transition 计时，它与固定时钟世界不是同一问题。

<a id="lesson-check"></a>

## 10 · 习题与答案线索

- 为什么不把资格迹等同于 Adam 的一阶矩？答：痕迹积累的是输出梯度；当前 δ 到来时才分配误差。Adam 的矩积累的是各个时刻已经乘各自误差的 loss gradient。
- 为什么 terminal 的 γnext=0 不应清掉本步全部历史信用？答：terminal 奖励属于刚结束的这一串决策，仍应沿到达当前状态的历史痕迹传播。
- 为何一个方法没有 buffer，却可能不满足固定预算？答：RTRL 的敏感性矩阵、长历史重算或无界模型增长可能消耗随规模增长的内存和时间。
- 能否用归一化证明长期稳定？答：归一化只控制部分尺度，目标漂移、off-policy 投影、非线性与策略反馈仍然存在。

动手题：在两步链加入第三个延迟状态，手算起点更新为 $α(γλ)^2$；再将特征整体乘 10，比较固定 α、按平方缩放 α 与 ObGD。将同一样本误差改变量和长期预测误差分开画，观察它们何时不一致。

## 本章的实验设计

同时记录交互次数、每步延迟和持久内存。均值延迟可能掩盖超时步骤。

设定：一条连续数据流逐步到达，限定每步计算和持久内存。改变未来后缀、特征尺度和控制周期，检查学习前缀及超时行为。

- 未来奖励与标签改变不影响此前已完成的更新。
- 归一化只使用协议允许的当前和过去信息。
- 声明的数据重访次数、状态大小及 fallback 与日志一致。

对照：调好的固定步长与简单尺度控制；Stream-X/Intentional 组件的同基座消融；同交互与同控制时钟分别比较

记录：每步延迟分布和超时率；持久状态大小、样本读取与梯度次数；完整生命期收益和变化后恢复

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-streaming)

## 学习与研究衔接

在线交互不等于严格 streaming。必须分别说明回放、批量、每步计算与持久存储。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-streaming) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=streaming) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=streaming)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

状态是支持后续计算的内部信息；GVF 指定一个预测问题；RTRL 和资格迹规定信用如何传播。三者可以组合，但不是相互替代的算法名称。先理解给定策略的预测，再讨论预测怎样改善控制。

- [Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks](https://yingwen.io/zh/continual-rl/research/#recent-columnar-constructive-networks)
- [Real-Time Recurrent Learning using Trace Units in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-real-time-trace-units)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Streaming Deep Reinforcement Learning Finally Works](https://yingwen.io/zh/continual-rl/research/#recent-stream-x)
- [Intentional Updates for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-intentional-updates)
- [Revisiting Adam for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-revisiting-streaming-adam)
- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)
- [Real-Time Recurrent Learning using Trace Units in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-real-time-trace-units)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)
- [Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-upgd-utility-protection)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-upgd-utility-protection)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Step-size Optimization for Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-step-size-optimization)
- [Learning from experience instead of curated datasets](https://yingwen.io/zh/continual-rl/research/#recent-oak-network-idbd)
- [Intentional Updates for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-intentional-updates)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [Revisiting Adam for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-revisiting-streaming-adam)

### Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks

Khurram Javed, Haseeb Shah, Richard S. Sutton, Martha White

JMLR 24 · 2023 · 支持方法与理论

#### 研究问题

如果每次观测只处理一次，如何学习包含历史信息的状态，而不保存一段序列做反向传播？

#### 关键机制

一般递归网络的实时递归学习需要维护庞大的参数—状态敏感度。CCN 限制列之间的递归依赖，并逐步构造新特征，使敏感度可以局部计算。它通过改变网络结构和构造过程降低求导成本，而不是把任意稠密 RNN 的完整导数免费变小。

#### 证据

论文分析受限结构的计算性质，并在动物学习启发的预测问题和 Atari 策略评价中检验预测效率。这里的 Atari 结果主要是预测已有策略的回报，不等于从头训练完整控制智能体。

#### 条件与限制

结构约束、构造顺序和被冻结的旧特征共同限制函数类。监督预测和策略评价上的优势，还需要在会主动改变数据分布的控制闭环中检验。

#### 阅读与实验

先写出递归状态对参数的敏感度递推，再检查哪些跨列项被结构消除。比较 CCN、截断 BPTT 与 RTU 时，同时计入状态、梯度缓存和每步计算。

#### 原文与相关入口

- [JMLR 原文与论文入口](https://www.jmlr.org/papers/v24/23-0367.html)：从网络结构、敏感度传播与预测实验三部分阅读。

### Real-Time Recurrent Learning using Trace Units in Reinforcement Learning

Esraa Elelimy, Adam White, Michael Bowling, Martha White

NeurIPS 2024 · 2024 · 支持方法与理论

#### 研究问题

递归状态既要保存长时信息，又要在在线强化学习中以可控成本更新，怎样设计其递归结构？

#### 关键机制

RTU 使用有结构的递归连接，并维护状态关于参数的在线敏感度。复杂的递归动力学可以用实值运算实现。其关键是让状态更新与梯度迹具有相容的计算结构，减少一般 RTRL 的高阶成本；这与仅给 TD 误差加一条资格迹不同。

#### 证据

论文在部分可观测任务中与常见递归网络比较预测与控制表现。作者代码包含 RTU、其他递归基线、实时 actor–critic 以及部分可观测环境配置。

#### 条件与限制

计算优势依赖特定递归参数化，不能外推为任意记忆问题上的表达能力优势。PPO 版本和严格逐步更新版本的经验协议不同，应分别比较。

#### 阅读与实验

在同一部分可观测任务中固定隐状态维度，再比较完整运行内存和每步更新时间。检查 actor、critic 与递归状态的参数更新是否共享同一条敏感度。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1e616bde0438cb10cb6adf076ae7d336-Abstract-Conference.html)：结构、在线导数与实验协议。
- [作者代码](https://github.com/esraaelelimy/rtus)：从 src/nets、src/agents 和实验配置追踪递归状态到控制更新。

#### 作者代码

[论文作者维护的实现。](https://github.com/esraaelelimy/rtus)

RTU 网络、实时学习器与论文实验配置。

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

### Streaming Deep Reinforcement Learning Finally Works

Mohamed Elsayed, Elena Sorina Lupu, Gautham Vasan, A. Rupam Mahmood

arXiv（2024 首稿；2026 v3） · 2026 · 直接研究持续学习

#### 研究问题

不保存经验重放、不使用目标网络或训练批次时，深度 RL 能否逐步稳定学习？

#### 关键机制

Stream-X 把信号归一化、表示初始化、资格迹和受控更新尺度组织为一组流式学习方法。各组件处理的是不同问题：奖励尺度、激活与梯度传播、延迟信用，以及一次更新造成的输出变化。去掉重放并不意味着这些问题会自动消失。

#### 证据

2026 年第三版扩展到 Atari、控制与机器人等实验，并包含持续变化设置。论文和代码经历过版本变化，比较结果时需要同时标明论文版本和算法实现。

#### 条件与限制

广泛任务上的流式可行性不等于所有非平稳问题都已解决。不能把旧版较弱 Adam 基线推广成对所有流式 Adam 方法的否定；后续研究专门检验了这一点。代码许可证也应独立于本教材许可证处理。

#### 阅读与实验

按归一化、资格迹、更新控制分别做消融，并保持每步算力一致。先验证严格一次使用经验，再研究长期变化，而不是仅把小批量大小改成一。

#### 原文与相关入口

- [2026 年第三版论文](https://arxiv.org/abs/2410.14606v3)：作者名单、任务范围与算法版本以该版为准。
- [作者代码版本](https://github.com/mohmdelsayed/streaming-drl/tree/9326fc3e23a401f28087ae2e41b635888740586b)：固定实现版本，避免把不同年份的更新规则混在一起。

#### 作者代码

[原作者仓库的固定版本。](https://github.com/mohmdelsayed/streaming-drl/tree/9326fc3e23a401f28087ae2e41b635888740586b)

Stream-X 算法、变换、优化器及实验；使用前阅读仓库许可证。

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

### Revisiting Adam for Streaming Reinforcement Learning

Florin Gogianu, Luțu Adrian-Cătălin, Razvan Pascanu

RLC 2026 / RLJ 预会议版 · 2026 · 支持方法与理论

#### 研究问题

流式 RL 的不稳定来自 Adam 本身，还是目标导数、方差与超参数的组合？

#### 关键机制

论文重新分析自适应更新的信噪比，将 Adam 的稳定项与目标导数尺度联系起来，并研究有界导数的回报分布学习及多步更新。它改变的是目标与更新的配合，而非简单沿用批量训练时的默认配置。

#### 证据

作者在大规模 Atari 流式实验中展示了具有竞争力的结果，并重新比较早期流式方法。正式 RLJ 入口收录为 RLC 2026 预会议论文。

#### 条件与限制

主体实验采用经典回合式 Atari 的流式学习协议，不是任意非平稳终生适应的证据。这些结果也不否定归一化、资格迹或更新约束在其他任务中的价值。版本、调参预算和目标分布必须对齐。

#### 阅读与实验

建立二维对照：固定目标换优化器，固定优化器换目标。将调参种子与最终测试分开，再判断改进来自哪一个因素。

#### 原文与相关入口

- [RLC 2026 论文入口](https://rlj.cs.umass.edu/2026/papers/Paper131.html)：会议收录信息与论文。
- [作者预印本](https://arxiv.org/abs/2605.06764)：Adam 尺度分析、回报分布目标与实验协议。

### Step-size Optimization for Continual Learning

Thomas Degris, Khurram Javed, Arsalan Sharifnassab, Yuxin Liu, Richard S. Sutton

arXiv 预印本 · 2024 · 支持方法与理论

#### 研究问题

误差变大时，应该减小步长过滤噪声，还是增大步长追踪真实变化？

#### 关键机制

论文区分梯度归一化与步长优化。IDBD 类方法以 $\alpha_i=\exp(\beta_i)$ 保证步长为正，并用权重对过去步长的敏感度估计改变 $\beta_i$ 是否有利。持续学习中，静止的无关方向适合很小步长，而持续变化的有用方向需要保留追踪能力。

#### 证据

作者用权重翻转和带噪追踪等线性学习问题比较机制，显示相似的误差幅度可以要求相反的步长反应。

#### 条件与限制

这些可分析任务不是深度控制上的普适优越性证据。元步长、近似敏感度与输入尺度仍会影响结果；步长自适应并没有消除全部外部设计参数。

#### 阅读与实验

分别增加观测噪声和目标漂移速度，检查步长是否采取不同反应。若只记录平均误差，就看不到噪声过滤与追踪之间的区别。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2401.17401)：步长优化与归一化的对照实验。

### Learning from experience instead of curated datasets

Oak Lab

Oak Lab 技术博文 · 2026 · 支持方法与理论

#### 研究问题

有用信号稀疏且大量输入是噪声时，在线学习规则如何分配不同方向的更新能力？

#### 关键机制

博文从含稀有有效特征的线性预测问题出发，对比统一步长与 IDBD 的逐权重适应，再展示 NetworkIDBD 在非线性带噪观测中的例子。核心主张是让长期学习效果影响信用和步长分配，而不只依据当前梯度幅度归一化。

#### 证据

公开页面提供受控噪声特征任务和 NoisyMNIST 示例。它们是机制演示，便于理解有效信号密度与输入规模的关系。

#### 条件与限制

该页面不是完整 CRL 控制论文，也未给出可直接复现所有图表的完整代码和算法推导。监督噪声任务的结果不能证明一般 SGD 或所有深度 RL 都无法从经验学习。

#### 阅读与实验

先复现线性噪声特征问题，分开改变有效特征稀疏度与噪声维数。进入控制前，再加入策略改变数据分布这一因素。

#### 原文与相关入口

- [Oak Lab 原始博文](https://oaklab.ai/posts/learning-from-experience-instead-of-curated-datasets)：2026 年 7 月 13 日；受控实验、NetworkIDBD 示例与研究动机。

### Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning

Mohamed Elsayed, A. Rupam Mahmood

ICLR 2024 · 2024 · 支持方法与理论

#### 研究问题

同一网络里，哪些方向应当保护，哪些方向应当获得更强的新学习与扰动？

#### 关键机制

UPGD 用移除权重或特征的反事实损失变化定义效用，并以 Taylor 近似在线估计。平滑、缩放后的效用同时调制梯度与随机扰动，让近期高效用方向变化较小、低效用方向更活跃。

#### 证据

主体证据包括未知边界的非平稳流式监督任务；另外包含长时间 PPO 实验。两类证据应分别理解，不能把监督任务数量写成 RL 任务覆盖。

#### 条件与限制

近期分布上的效用不保证稀有旧知识的重要性；一阶和二阶近似、权重级和特征级版本不同。PPO 仍使用 rollout 与重复更新，不因 optimizer 在线就成为严格流式 RL。

#### 阅读与实验

用可精确消融的小网络检查效用估计，再拆开保护梯度、保护噪声和 weight decay 三种作用；独立报告新学习与旧功能。

#### 原文与相关入口

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/file/8e5f0591943d8dae5702af12dcdcd2f6-Paper-Conference.pdf)：效用定义、近似、不同 UPGD 变体与 PPO 实验。
- [作者预印本](https://arxiv.org/abs/2404.00781)：流式监督协议与 RL 证据范围。

#### 作者代码

[论文首页明确链接的作者仓库；README 的短实现是一个指定变体。](https://github.com/mohmdelsayed/upgd)

权重/特征效用实验、流式任务及 PPO 实现。

### Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning

Noah Farr, Aryaman Reddi, Carlo D’Eramo, Jan Peters

arXiv预印本（2026-07-07 v2） · 2026 · 支持方法与理论

#### 研究问题

严格逐步更新的智能体怎样同时学习递归记忆、分配延迟信用并控制计算？

#### 关键机制

将RTU结构的RTRL敏感度接入QRC与流式actor–critic。敏感度给出当前输出对记忆参数的导数，资格迹再组合过去输出的回报信用；两条递推保持分工。

#### 证据

v2在MemoryChain、五项POPGym和masked MuJoCo上报告5-seed结果，另用KMemoryChain比较在线敏感度与当前参数重算参考，并检验Taylor修正。

#### 条件与限制

masked MuJoCo仍落后批量PPO。固定参数精确RTRL不代表在线变参敏感度始终等于当前参数重算；诊断保存整个episode，须计为额外评价资源。尚未确认作者公开代码。

#### 阅读与实验

在相同递归容量下独立改变记忆跨度、回报λ与参数步幅，同时测敏感度误差和回报。诊断改善不能单独当作控制改进证据。

#### 原文与相关入口

- [2026年v2原文](https://arxiv.org/html/2605.24709v2)：方法、5-seed实验、masked MuJoCo负边界与staleness诊断。


<a id="chapter-code"></a>

## 下载与运行

流式 TD、在线统计与 ObGD 的机制检查；meta_frontier_lab.py 提供 Intentional 输出尺度实验。不包含完整神经网络控制训练。

[下载 lifelong_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)

```sh
python lifelong_algorithms_lab.py streaming
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [配套章节：时间信用分配](/zh/continual-rl/algorithms/credit-assignment/)：系统讨论多步回报、资格迹和跨时间信用；流式学习只是其中部分方法可以满足的数据协议。

- [配套章节：学习规则的适应](/zh/continual-rl/algorithms/meta/)：在线元梯度使用学习过程的敏感度调整规则；与本章预设的更新尺度控制区分。

- [Sharifnassab 等：Intentional Updates（ICML 2026）](https://proceedings.mlr.press/v306/sharifnassab26a.html)：Intentional-TD、Intentional-Q 与 Intentional-AC；按预期输出变化选择局部步长。

- [Intentional-AC 作者优化器源码](https://github.com/sharifnassab/Intentional_RL/blob/e86e26fd8613ac212e9a52c3fed8a01d0a31f685/optimizer.py)：二阶矩、sigma 偏差校正、误差裁剪、policy 归一化及更新后 trace reset 的具体顺序。

- [Intentional-AC 作者 actor–critic 主循环](https://github.com/sharifnassab/Intentional_RL/blob/e86e26fd8613ac212e9a52c3fed8a01d0a31f685/intentional_ac.py#L79)：update_params 用旧 critic 构造 TD 误差；actor 熵项乘该误差的符号，随后将合成梯度送入独立的 policy 优化器。

- [Sutton & Barto · Reinforcement Learning, Chapter 12](http://incompleteideas.net/book/the-book-2nd.html)：前向 λ-return、资格迹与控制中的 trace 语义；本页已给出所需核心推导。

- [van Seijen & Sutton · True Online TD Learning](https://proceedings.mlr.press/v32/seijen14.html)：线性在线前向/后向精确等价的适用范围，区别于普通 accumulating traces。

- [Elsayed, Vasan & Mahmood · Stream-X，2024 版](https://arxiv.org/abs/2410.14606v2)：ObGD 的有效步长动机、资格迹与网络稳定化组合；与修订版区分。

- [Stream-X · 2024 作者实现](https://github.com/mohmdelsayed/streaming-drl/tree/407dca7a8b584c1c20bc649053557f66e270b1e6)：optim.py 使用 ObGD；从 stream_ac_continuous.py 核对输出梯度、熵项和误差乘法。

- [Elsayed, Lupu, Vasan & Mahmood · Stream-X，2026 修订版](https://arxiv.org/abs/2410.14606v3)：逐坐标 StreamingOptimizer、layer-aware 初始化及 Stream-RAC；附录 B.2 解释相对于早期方法的变化。

- [Stream-X · 2026 作者实现](https://github.com/mohmdelsayed/streaming-drl/tree/9326fc3e23a401f28087ae2e41b635888740586b)：依次阅读 obs_reward_transforms.py、sparse_init.py、optimizer.py 和相应 agent；代码采用 CC BY-NC 4.0 许可证。

- [Welford · Note on a Method for Calculating Corrected Sums of Squares and Products](https://doi.org/10.1080/00401706.1962.10490022)：在线均值与方差递推的原始统计方法。
