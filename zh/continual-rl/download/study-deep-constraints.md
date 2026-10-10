# 约束强化学习：占据测度、拉格朗日与可行策略

现代深度强化学习 · 并列研究分支

“回报高且代价不超过预算”与“每一步都安全”之间差了哪些条件？

## 本章内容

- 从折扣访问频率推导 CMDP 的占据流约束。
- 解释随机策略为何可能必要及拉格朗日乘子的方向。
- 区分平均代价、概率约束、硬约束和训练期安全。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [MDP、回报与价值：序列决策的数学对象](https://yingwen.io/zh/continual-rl/foundations/tabular/mdps/)：明确回报与访问分布。
- [策略梯度：从轨迹概率到 GAE 与 actor–critic](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/)：理解策略参数改变回报的梯度，再引入成本和乘子。


### 条件概率与期望

区分随机变量、一次样本与条件分布；可以使用 Bayes 法则和全期望公式。

### MDP 与价值

理解状态、动作、转移、奖励、策略和 Bellman 递推；学习数据可以来自不同策略。

### 优化与梯度

知道固定目标、参数梯度、约束和期望目标的含义；神经网络只是函数表示的一种选择。

<a id="problem-definition"></a>

## 本章的问题定义

除了收益，还规定累计成本预算；允许的策略必须同时满足目标和约束。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 成本 $C_{t+1}$、预算 $b$，$J_C(\pi)=\mathbb E_\pi[\sum_{t\ge0}\gamma^tC_{t+1}]$。

### 需要求解的对象

在可行策略集合中最大化收益，必要时采用随机化策略。

### 信息与数据权限

成本是额外观测信号；执行时安全约束若要求每步成立，需要另给可验证信息。

$$
\max_\pi J(\pi)\quad\text{s.t.}\quad J_C(\pi)\le b
$$

这是期望累计成本约束。它不等价于每条轨迹、每个时刻都安全，也不等价于预设一个固定惩罚系数。

### 成立条件与解的含义

- 可行集非空；成本尺度与时间聚合明确。
- 强对偶及线性规划解释依赖相应有限凸表示条件。

判断准则：同时报告收益、约束违反量、置信区间和可行运行比例。

### 适用边界

- 从平均约束满足推出零事故或逐步安全保证。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)：本章从一般问题规格中选取有期望累计成本预算的 CMDP；逐步安全等其他约束不由这个特例涵盖。

- 改变评价目标 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：相对仅按收益排序的控制，本章增加成本可行性条件，再在可行集合内最大化收益；固定惩罚权重不是同一个问题。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

最高收益策略可能不可行；固定惩罚难以事先对应指定预算。

### 本章的核心思路

用占据测度定义可行策略，再将约束违反反馈给乘子和策略更新。

1. [把策略转成长期访问量](#lesson-notation)：占据流约束连接局部转移与整个策略表现。

2. [在可行集合里求最优](#lesson-derive)：线性例子展示为何随机化可以成为必要条件。

3. [把预算误差反馈给惩罚强度](#lesson-dual)：乘子方向来自对偶问题，不能随意将负成本当成另一种奖励。

结论与条件：期望 CMDP 保证、CPO 局部近似与执行时安全是不同强度的结论。

### 相关方法改变了什么

- 固定 penalty / dual learning：前者指定权衡，后者根据违反预算调节权衡。

- CMDP / safety shield：前者约束长期统计，后者在执行时限制具体动作。


<a id="lesson-setting"></a>

## 1 · 约束不是另一个奖励名称

前面的策略优化比较期望回报，并用信任域控制一次更新的幅度。配送机器人还可能受到另一项任务规格约束：进入风险区的次数不能超过预算。两条路线交付得几乎一样快，却可能只有一条策略满足这项规格。奖励差小，并不能代替成本核算。

$$
\max_\pi J_r(\pi),\qquad J_c(\pi)=\mathbb E_\pi\sum_{t=0}^\infty\gamma^t C_{t+1}\le d
$$

CMDP 在回报之外声明代价预算。代价可以表示能耗、碰撞次数或其他事件，但其定义由任务提供。

给奖励减去固定惩罚系数，是优化一个标量化目标；满足给定预算是另一个问题。除非有额外分析，不能从“惩罚很大”推出预算合格。设计者提供成本传感器、约束阈值、初始分布和事故定义；这些量并不是算法从奖励标量自动推断出的。

这里先考虑有限状态动作、平稳转移和折扣期望代价。期望累计代价小，仍允许某些轨迹发生严重事故。若目标是每次执行都不越界、事故概率低于阈值或不可逆状态永不进入，需要不同约束与模型条件。

用同一个已知配送模型贯穿本章。起点分布为 $\nu(s)=1$，$\gamma=0.9$。在路口 $s$ 选捷径，立即收到 $R_1=9.2,C_1=1$ 并到终点 $T$；选绕行，收到 $R_1=C_1=0$ 并到 $l$，随后唯一动作 go 带来 $R_2=10,C_2=0$ 并到 $T$。$T$ 吸收，之后奖励和成本全为零。成本一表示任务规定的一次风险区进入，不是从物理事故数据估出来的概率。设计者给定这个模型、成本事件和未归一化预算 $d=0.3$。

令 $p=\pi(\text{捷径}\mid s)$。先算单条路线：捷径的折扣回报为 $9.2$，绕行是 $0+0.9\times10=9$。再按起点动作求期望，得到 $J_r(p)=9+0.2p$、$J_c(p)=p$。预测一下：将 $p$ 从 $0.6$ 改为 $0.3$，只少了多少奖励，却改变了哪一项可行性判断？

![配送路口的捷径和两步绕行，以及同一回报成本坐标上p等于0.3和0.6的可行性比较](https://yingwen.io/crl-figures/constraint-control-walkthrough-routes.svg)

先沿路线读奖励发生在哪一步，再看下方的成本—回报线段。$p=0.6$ 得 $J_r=9.12,J_c=0.6$，超过预算；$p=0.3$ 得 $9.06,0.3$，恰好可行。绿色是 $J_c\le0.3$，回报纵轴从 8.98 开始以放大 0.06 的差异。图和数值来自附带已知模型的原创精确计算。

<a id="lesson-notation"></a>

## 2 · 用占据测度表示整个策略

$$
x_\pi(s,a)=(1-\gamma)\mathbb E_\pi\sum_{t=0}^\infty\gamma^t\mathbf1[S_t=s,A_t=a],\qquad x_\pi\ge0
$$

规范化后的折扣占据测度总和为一。它记录状态动作被使用多少，而不只记录每个状态下的动作概率。

$$
\sum_ax(s,a)=(1-\gamma)\nu(s)+\gamma\sum_{\tilde s,a}P(s\mid\tilde s,a)x(\tilde s,a)
$$

把时间零与随后时刻的访问拆开。后续访问等于来自所有前驱的折扣流入，得到线性流守恒。

将流守恒式对所有状态求和，设 $X=\sum_{s,a}x(s,a)$。转移行的总概率为一，所以 $X=(1-\gamma)+\gamma X$，从而 $X=1$。若任务会终止，应包含零奖励、零代价的吸收终止状态及其占据，才能保持这一归一化；若只统计终止前访问，则总质量可能小于一。这样增加吸收状态不会改变原回报与成本。

$$
\pi(a\mid s)=\frac{x(s,a)}{\sum_{a'}x(s,a')}
$$

状态占据非零时可以恢复平稳随机策略；零占据状态的动作可另作定义，不能由零除法确定。

$$
\begin{aligned}x(s,\text{绕行})&=0.1(1-p),&x(s,\text{捷径})&=0.1p,\\x(l,\text{go})&=0.09(1-p),&x(T,\text{wait})&=0.9p+0.81(1-p).\end{aligned}
$$

路口只在 t=0 访问，绕行格只可能在 t=1 访问。捷径从 t=1 起留在 T，绕行从 t=2 起留在 T；吸收尾部的几何级数产生最后一项。

例如 $p=0.3$ 的四项占据依次是 $(0.07,0.03,0.063,0.837)$，总和一。$T$ 的流是 $x(T)=0.9[x(s,\text{捷径})+x(l,\text{go})+x(T)]$，代入这些数恰好成立。若删去终止状态，只剩 $0.163$ 的质量；若又把它重新除以 $0.163$，得到的是另一个统计口径，不能继续套用这里的回报换算。

<a id="lesson-derive"></a>

## 3 · CMDP 的线性规划与随机化

$$
(1-\gamma)J_r=\sum_{s,a}x(s,a)r(s,a),\qquad(1-\gamma)J_c=\sum_{s,a}x(s,a)c(s,a)
$$

归一化占据对应归一化收益与代价，因此预算也必须乘相同因子。

$$
\max_{x\ge0}\sum_{s,a}x(s,a)r(s,a)\quad\text{s.t. flow},\qquad\sum_{s,a}x(s,a)c(s,a)\le(1-\gamma)d
$$

已知有限模型下，目标与约束对占据测度是线性的。这个精确规划形式不等于神经策略优化已经解决可行性。

没有约束时，有限折扣 MDP 可以找到最优确定策略；加入成本预算后，最优可行解可能需要随机化。随机化是为了在回报和代价之间满足预算，不只是训练时探索。若所有策略都超出预算，问题本身不可行，优化器不能创造一条不存在的安全策略。

配送例的占据成本是 $x(s,\text{捷径})=0.1p$，规范化预算必须同时变为 $(1-\gamma)d=0.03$。规范化回报为 $9.2(0.1p)+10[0.09(1-p)]=0.9+0.02p$。因此可行区间是 $0\le p\le0.3$；回报在这个区间递增，最优解 $p^*=0.3$。纯绕行可行但少得 $0.06$，纯捷径不可行。误将成本 $0.06$ 与未归一化预算 $0.3$ 比较，会把 $p=0.6$ 错判为可行。

在这个只访问一次路口的任务里，起点以概率 $0.3$ 选“全程捷径策略”，与路口按概率 $0.3$ 选捷径，产生相同轨迹分布。一般有限折扣 CMDP 中，占据流的凸组合 $x_{\mathrm{mix}}=(1-\alpha)x_A+\alpha x_B$ 仍满足线性流约束；恢复策略要按各状态占据除法，不能直接假设每个状态都以同一个 $\alpha$ 混合动作。具体是 $\pi_{\mathrm{mix}}(a\mid s)=[(1-\alpha)x_A(s,a)+\alpha x_B(s,a)]/[(1-\alpha)x_A(s)+\alpha x_B(s)]$，其中 $x_i(s)=\sum_a x_i(s,a)$，分母须非零。这保证对应的折扣占据；一般情况下不声称整条轨迹分布也相同。

可行混合仍有 $30\%$ 的轨迹走捷径，轨迹成本为一；另外 $70\%$ 的轨迹成本为零，所以期望成本是 $0.3$。这里“至少一次成本事件”的概率恰好也等于 $p$，因为事件至多发生一次且发生在 $t=0$。把事件移到更晚时刻或允许重复进入后，这个概率便不能由折扣成本直接读出。若规格要求每条轨迹成本不超过 $0.3$，本例只能选 $p=0$；若要求进入概率不超过 $0.05$，则是另一条约束 $p\le0.05$。

<a id="lesson-dual"></a>

## 4 · 乘子怎样把违反预算反馈给策略

$$
\mathcal L(\pi,\lambda)=J_r(\pi)-\lambda(J_c(\pi)-d),\qquad\lambda\ge0
$$

策略对拉格朗日函数做最大化，乘子对其做最小化。固定乘子相当于使用奖励减成本的目标。

$$
\begin{gathered}\theta\leftarrow\theta+\eta_\theta(\nabla J_r-\lambda\nabla J_c)\\\lambda\leftarrow[\lambda+\eta_\lambda(\widehat J_c-d)]_+\end{gathered}
$$

代价超预算时乘子上升，使下一轮策略更重视代价。正部投影保证乘子非负。

$$
\lambda^*(J_c(\pi^*)-d)=0
$$

互补松弛描述理想最优解：未激活的约束可有零乘子；正乘子对应预算恰好激活。它不是每个学习迭代都应满足的恒等式。

有限已知 CMDP 的占据线性规划可以分析对偶与可行性；神经参数化、采样噪声和同时更新可能破坏这些简洁性质。乘子振荡、成本 critic 滞后和策略表示不足都会影响实际学习，不能只看最后一次 multiplier。

回到配送例，$\mathcal L(p,\lambda)=9+0.2p-\lambda(p-0.3)$。策略上升方向为 $\partial_p\mathcal L=0.2-\lambda$；乘子做下降，而 $\partial_\lambda\mathcal L=-(p-0.3)$，因此更新中是加上预算违反量。按同一个旧版本计算两方向：

$$
p_{k+1}=\operatorname{clip}_{[0,1]}[p_k+0.5(0.2-\lambda_k)],\qquad\lambda_{k+1}=\max[0,\lambda_k+2(\widehat J_{c,k}-0.3)]
$$

$p$ 直接作为概率坐标；两项投影分别保护概率区间与非负乘子。这里用已知模型的精确成本 $p_k$ 作成本估计，演示更新时序。

从 $(p_0,\lambda_0)=(0.6,0)$ 出发，得到 $(p_1,\lambda_1)=(0.7,0.6)$，随后 $(0.5,1.4)$，再到 $(0,1.8)$。第一步成本违反 $0.3$ 使乘子变大，但策略仍读旧乘子零，所以成本反而先由 $0.6$ 上升到 $0.7$。第三步未经投影的概率为 $-0.1$，才被截到零。乘子投影同样有作用：例如旧 $p=0.1,\lambda=0.1$ 时，未经投影的新乘子是 $-0.3$，应变为零。理想活跃约束的解是 $p^*=0.3,\lambda^*=0.2$；这三次选定步长的更新并不宣称收敛到它。

![旧概率与旧乘子同时计算的四个快照，固定动作分位点导致捷径改为绕行，以及KL区间与成本区间的交集](https://yingwen.io/crl-figures/constraint-control-walkthrough-updates.svg)

上半部条长表示捷径与绕行概率，虚线是同一个给定动作分位点 $u=0.55$：$u<p$ 走捷径，否则绕行。因此 $p=0.6,0.7$ 对应单步 $(R_1,C_1)=(9.2,1)$；$p=0.5,0$ 则先到绕行格，再收到 $R_2=10$。这是配对的模型机制计算，固定 $u$ 不用于估计概率或训练性能。下半部从另一个明确的旧策略 $p=0.2$ 出发，比较只限制局部 KL 和同时限制成本的提案；绿色是两种限制的交集。

<a id="experiment-cmdp_primal_dual"></a>

### 实验：奖励提高，不代表满足成本约束

同一个策略梯度中加入对偶乘子，能否追踪平均成本限制？

**环境与可用信息。** 单状态两动作，奖励是均值 0.2／0.9 的 Bernoulli 随机变量；动作 1 的成本为 1，动作 0 为 0。要求期望成本不超过 0.4。没有转移、回合与延迟后果。

**设置。** 本图实际运行 1200 个预算单位，训练种子为 0–4。预算为独立拉臂；策略是 sigmoid(logit)，logit 和乘子从 0 开始。actor 与乘子均使用 0.15 除以时间平方根的步长，先读取旧乘子，再分别计算新值；乘子投影到非负。

**检验的机制。** actor 对奖励减乘子乘成本做 score-function 更新；乘子因实际成本超过预算而增大。对照忽略约束，因此不是同一可行集合内的竞争方法。

**测量。** 主图测动作 1 概率到解析可行最优 0.4 的距离。日志另记录期望收益、正成本违反和乘子。概率距离不等于已经满足约束。

```bash
python3 implementations/extended_classic/cmdp_primal_dual.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/cmdp_primal_dual/curves.svg)

横轴：environment_steps。纵轴：最优可行动作概率绝对误差。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 1200 步时概率绝对误差均值约 0.0465，对照约 0.4083。约束更新使概率更接近目标，但有限步随机迭代不保证每一步都可行。

**结论边界。** 这是静态平均约束的单状态机制，不是安全探索、多状态 CMDP 求解或逐轨迹零违反保证。忽略约束的对照可以有更高奖励，但解决了不同问题。

**继续实验。** 把目标从平均成本改为任何时候都不得执行危险动作：原始对偶算法是否仍满足要求？分别画成本违反、收益和乘子，不只画距离。

[源码](https://yingwen.io/crl-code/implementations/extended_classic/cmdp_primal_dual.py) · [逐种子记录](https://yingwen.io/crl-code/results/cmdp_primal_dual/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/cmdp_primal_dual/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/cmdp_primal_dual/curves.json)

<a id="lesson-safety"></a>

## 5 · CPO 与执行时安全的区别

在 [TRPO/PPO 章](/zh/continual-rl/foundations/deep/trust-region/) 中，KL 与 clipping 管的是策略变化幅度。CPO 进一步把成本变化写进候选策略的约束。[原论文 §5.3–6](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf)先给带成本上界的理论更新，再用式 (10) 的信任域近似；其 Proposition 2 允许一个随 KL 半径而变的期望成本违反界。实际算法还将目标与成本线性化、KL 二阶展开，用采样估计、共轭梯度与回溯；这些近似不构成逐条轨迹的无事故证书。

$$
\widetilde J_{c,k}(\pi)=J_c(\pi_k)+\frac{1}{1-\gamma}\mathbb E_{s\sim d_{\pi_k},a\sim\pi}[A_c^{\pi_k}(s,a)],\qquad\bar D_{\mathrm{KL}}(\pi\Vert\pi_k)=\mathbb E_{s\sim d_{\pi_k}}D_{\mathrm{KL}}(\pi(\cdot\mid s)\Vert\pi_k(\cdot\mid s))
$$

$d_{\pi_k}$ 是归一化旧状态占据，$A_c^{\pi_k}$ 是按成本定义的旧策略 advantage。这里沿用 CPO 式 (10) 的 KL 方向；成本代理的一阶匹配不意味着在任意候选策略上等于真实成本。

$$
\max_\Delta g_r^\top\Delta\quad\text{s.t.}\quad \hat J_c+g_c^\top\Delta\le d,\qquad\tfrac12\Delta^\top F\Delta\le\delta
$$

这是理解局部 constrained update 的近似优化问题。若当前策略已不可行，需恢复步或其他处理，不能默认约束一定有满足的上升方向。

配送例可直接算这个局部问题。另取可行旧策略 $p_k=0.2$，坐标仍是 $p$，令 $\Delta=p-p_k$，$\delta=0.005$。旧状态占据只有 $s$ 上的动作分布变化，$d_{\pi_k}(s)=0.1$，所以平均 KL 的二阶系数 $F=0.1/[0.2(1-0.2)]=0.625$。奖励方向按未归一化 $J_r$ 记为 $g_r=0.2$（正比例缩放不改变这里的最优提案），成本方向 $g_c=1$。KL 允许 $|\Delta|\le\sqrt{2\delta/F}\approx0.12649$，成本却要求 $0.2+\Delta\le0.3$。两者交集的奖励最大点是 $p=0.3$；只保留 KL 的奖励最大点约为 $0.32649$，期望成本超预算。

两候选的精确平均 KL 也可枚举两动作求得：$0.1[p\log(p/0.2)+(1-p)\log((1-p)/0.8)]$。在 $p=0.3$ 时约 $0.002817$；在 $0.32649$ 时约 $0.004409$，都低于 $0.005$。因此本例中的成本越界并非来自 KL 超标。由于真实回报与成本恰好对 $p$ 线性，这里能直接核对；神经策略通常仍需检验代理、曲率和采样误差。

图中计算的是单参数局部提案。CPO 原论文 §6.2 在近似可行域为空时提出降低成本的恢复方向，§7 则区分两种乘子：前节 primal–dual 保存乘子并跨轮更新，CPO 每轮从当前局部对偶问题重新求解。把前节的 λ 更新贴上 CPO 名称，会遗漏这项机制差异。

执行时 shield 或 barrier 方法限制具体动作的可行集合，常需要可靠动力学、状态估计与不变集条件。平均约束、鲁棒控制与这些执行过滤机制可以组合，但它们保护的对象不同。训练时违反多少约束，也应与最终冻结策略的代价分开报告。

<a id="lesson-algorithm"></a>

## 6 · 同时记录收益、代价和可行性

**算法：本页数值核采用旧策略与旧乘子同时计算两方向，避免时间顺序含糊。**

1. 明确代价事件、预算、折扣与初始分布。
1. 检查是否存在已知可行基线；没有时不能假设问题可行。
1. 采样策略数据，分别估计 reward return 与 cost return。
1. 固定旧策略估计，计算策略与乘子的方向。
1. 投影乘子到非负区间，按声明的顺序更新策略。
1. 记录每轮预算违反、训练期累计代价与独立评估。
1. 需要硬执行保证时另行验证安全模型和动作过滤器。

若改为采样学习，在旧策略采到的轨迹里应同时保存奖励、成本、终止标志和策略版本，分别估计 reward advantage 与 cost advantage。优化器读这些固定的旧批估计，更新后的策略再决定下一批实际路线。上图改变动作后，进入学习器的数据会从捷径的单步成本事件变成绕行的两步无成本经验；新策略的成本应重新评价，不能拿旧批的低成本记录替它作可行性判决。

<a id="lesson-example"></a>

## 7 · 一个必须混合的最优策略

单状态两动作，奖励为 $(1,3)$，代价为 $(0,2)$。使用规范化代价预算 $0.6$，选择动作一的概率为 $p$。奖励为 $1+2p$，代价为 $2p$，所以最优可行概率 $p=0.3$，收益为 $1.6$。确定选高奖励动作不可行，确定选低成本动作可行但次优。

若 $\gamma=0.9$，上述规范化预算 0.6 对应未规范化预算六，收益 1.6 对应折扣总收益十六。把 0.6 与未规范化成本二十比较，是单位错误。单状态占据 $(0.7,0.3)$ 满足流守恒。

固定惩罚 $\lambda<1$ 时仍倾向完全选择高成本动作；$\lambda>1$ 时倾向低成本动作；$\lambda=1$ 时两动作标量价值相同。正确乘子不独自指定混合比例，还需可行性和策略解。

<a id="lesson-code"></a>

## 8 · 占据流、解析混合与 primal–dual 核

贯穿配送例的单文件入口是 [constraint-control-walkthrough.py](/crl-code/tutorials/constraint-control-walkthrough.py)。下载到空目录即可运行，只用 Python 标准库。默认输出回报、成本、四项占据、三步更新与局部提案；--test 用 Fraction 精确枚举完整轨迹，再用独立线性方程求占据流，并枚举局部可行概率核对最优点。图的数字由独立 [JavaScript 计算](/crl-code/figures/constraint-control-walkthrough.mjs) 提供。

默认 JSON 中 random_rollouts 与 neural_training_steps 均为零；四个策略快照是确定性更新计算。教程不含完整 CPO 的神经 critic、共轭梯度、回溯或执行安全层。

```bash
# 把单文件教程保存到空目录，在该目录运行：
python3 constraint-control-walkthrough.py
python3 constraint-control-walkthrough.py --test
```

规范化流残差、两动作精确可行解以及同时计算的策略/乘子一步更新。

```python
def occupancy_residual(occupancy, transition, initial, gamma):
    """Normalized discounted occupancy x[s][a] and P[s][a][next_state]."""
    probabilities(initial)
    if not 0 <= gamma < 1:
        raise ValueError('discount below one required')
    n = len(initial)
    return [sum(occupancy[s])-(1-gamma)*initial[s]
            -gamma*sum(occupancy[sp][a]*transition[sp][a][s]
                       for sp in range(n) for a in range(len(occupancy[sp])))
            for s in range(n)]


def constrained_two_action(rewards, costs, budget):
    """One-state normalized occupancy LP; costs[1] > costs[0]."""
    if costs[1] <= costs[0]:
        raise ValueError('ordered distinct costs required')
    if budget < costs[0]:
        raise ValueError('infeasible budget')
    largest = min(1., (budget-costs[0])/(costs[1]-costs[0]))
    p = largest if rewards[1] > rewards[0] else 0.
    return p, (1-p)*rewards[0]+p*rewards[1], (1-p)*costs[0]+p*costs[1]


def primal_dual_step(logit, multiplier, reward_gap, cost0, cost_gap, budget,
                     actor_step=.1, dual_step=.1):
    probability = 1/(1+math.exp(-logit))
    gradient = probability*(1-probability)*(reward_gap-multiplier*cost_gap)
    cost = cost0+cost_gap*probability
    # Both directions use the same old policy and multiplier.
    return logit+actor_step*gradient, max(0., multiplier+dual_step*(cost-budget))
```

运行 test 检查流守恒、最优混合、不可行预算拒绝和 primal 梯度有限差分。代码没有实现 CPO 的共轭梯度、线搜索或安全执行层。jachiam/cpo 提供原作者研究代码；Safety Gym 是代价环境，不是安全性证明工具。

<a id="lesson-branches"></a>

## 9 · 约束在持续任务中也会变化

- 风险漏记：成本传感器未覆盖的事故，不会被成本 critic 自动补充。
- 尾部不足：低频严重事故可能在平均样本里消失；要与尾部风险和置信边界区分。
- 非平稳阈值或动力学：旧可行策略可能变得不可行，历史成本回放也可能不再代表当前风险。

CRL 的单生命期要求将探索成本、适应阶段事故和恢复能力计入评价，而非只测训练后的策略。状态预测、模型规划与约束优化相互影响：不能观察到危险状态的 agent，即使优化器正确，也可能无法执行所需安全控制。

<a id="lesson-check"></a>

## 10 · 练习与答案

- 预测：配送例把预算从 0.3 改为 0.05，最优策略怎样变？答：p=0.05，回报 9.01；仍有 5% 的轨迹计一次成本事件。
- 手算：p=0.6 的归一化成本和预算是多少？答：0.06 与 0.03，应判不可行。
- 定位错误：第一轮若以新 p=0.7 更新 λ，会得到什么？答：0.8；这与本页同用旧版本得到的 0.6 不同，必须另行声明更新时序。
- 改变条件：只将折扣改为 0.95，绕行回报是多少，最优 p 还在预算边界吗？答：绕行 9.5 已超过捷径 9.2，最优为 p=0，成本约束不激活。
- 设计对照：先冻结成本估计，分别比较旧/新乘子时序，再单独改变成本估计误差。记录动作概率、预算违反及训练成本；动作已改变并不自动说明估计可靠。

- 问：代价期望小于预算，是否保证每条轨迹安全？答：不是；这是平均约束。
- 问：提高固定惩罚一定能找到最优可行策略吗？答：不保证；混合策略、可行性及优化误差仍需处理。
- 问：所有动作成本至少一而预算 0.6，会发生什么？答：此单状态问题不可行，程序应明确拒绝。
- 实验：将预算从 0.6 改为一，解析最优 p 应从 0.3 变为 0.5。再检查规范化与未规范化指标是否同步改变。



<a id="chapter-code"></a>

## 下载与运行

标准库解析机制实验；不包含完整神经网络训练或大型 benchmark。

[下载 extended_foundations_lab.py](https://yingwen.io/zh/continual-rl/download/extended_foundations_lab.py)

```sh
python3 extended_foundations_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Altman · Constrained Markov Decision Processes](https://www.routledge.com/Constrained-Markov-Decision-Processes/Altman/p/book/9781315140223)：原作者专著；占据测度、线性规划与拉格朗日方法的系统来源。

- [Achiam et al. · Constrained Policy Optimization](https://proceedings.mlr.press/v70/achiam17a.html)：§4 给定起点的折扣 CMDP；§5.3 式 (10)、Proposition 2 的成本代理与违反界；§6 线性/二阶局部近似、共轭梯度、回溯和恢复；§7 与跨轮保存乘子的区别。

- [Schulman et al. · Trust Region Policy Optimization](https://proceedings.mlr.press/v37/schulman15.html)：§4–5 从策略性能界到平均 KL 与旧策略样本代理；KL 变化限制和任务成本约束是不同对象。

- [Achiam · CPO 原作者代码](https://github.com/jachiam/cpo)：原研究工程；算法实现与环境依赖应分别阅读。

- [OpenAI · Safety Gym](https://github.com/openai/safety-gym)：原团队的带成本环境，用于约束学习评估；不提供普遍安全保证。

- [Kochenderfer、Wheeler、Wray · Algorithms for Decision Making](https://algorithmsbook.com/)：作者书站提供决策、信念状态、模型与规划教材及配套 Julia 代码入口。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-constraints#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-constraints#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-deep-constraints)
<a id="study-connections"></a>

## 与教材主线的衔接

本章是可按问题选择的并列研究分支，不要求先读完其他深度研究分支。

### 一次策略更新，为什么能改善未来？

精确策略改善使用旧策略的真实价值；策略梯度使用与目标匹配的访问分布。值函数近似或梯度估计误差会破坏这些推理的前提。

函数逼近与深度方法：PPO 的动作概率比不等于新策略的状态访问比。连续动作 actor 还会追逐 critic 的误差。限制局部更新尺度与证明实际回报单调提高是不同要求。

持续学习中的研究问题：动作不仅改变世界，也改变未来数据与学习。比较冻结策略不等于比较持续更新的智能体；什么时候应付出当前回报去获得长期有用的经验？

[精确策略改善](https://yingwen.io/zh/continual-rl/foundations/tabular/dynamic-programming/) → [策略梯度定理](https://yingwen.io/zh/continual-rl/foundations/approximation/policy-gradient/) → [TRPO 与 PPO 的近似](https://yingwen.io/zh/continual-rl/foundations/deep/trust-region/) → [持续控制的比较器](https://yingwen.io/zh/continual-rl/algorithms/control/)


### 可进一步检验的问题

- [奖励表示与奖励学习：目标怎样进入智能体？](https://yingwen.io/zh/continual-rl/research/#research-reward-design)：将回报目标与成本可行性分开，才能判断改变奖励是在表达偏好，还是只用惩罚近似另一项约束。
- [14 · 应当探索什么、练习什么，以及如何保留未来交互与学习的机会？](https://yingwen.io/zh/continual-rl/research/#research-experience-selection)：平均成本约束不等于逐步安全或长期可恢复性，选择探索经验时需要分别规定这些未来交互条件。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)
- [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

对应原始材料：Constrained Markov Decision Processes；Achiam et al.：Constrained Policy Optimization。本文为原创讲解，原书、论文与上游代码保留各自许可。
