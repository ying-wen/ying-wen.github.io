# 策略更新的尺度：TRPO 与 PPO

现代深度强化学习 · 第 3 章

旧策略的数据能支持多远的策略更新？怎样从局部代理走到采样、更新与独立评价？

## 本章内容

- 由性能差分恒等式定位旧状态分布替代的近似。
- 求解局部 KL 约束，理解 Fisher、共轭梯度与回溯。
- 按优势符号解释 PPO，并固定旧策略、优势与 critic target。
- 区分批次、优化步和交互步，评价冻结策略与持续学习器。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [策略梯度：从轨迹概率到 GAE 与 actor–critic](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/)：区分采样策略、优势估计与要更新的策略。


### Bellman 递推

当前价值等于即时奖励加折扣后的后继价值；预测目标与最优控制目标不同。

### 函数与梯度

神经网络把参数和输入映射为输出。链式法则必须指明哪些量参与求导，哪些量作为固定目标。

### 经验分布

on-policy 数据由当前策略产生；off-policy 数据可来自旧策略，但能否正确复用取决于算法目标和数据覆盖。

<a id="problem-definition"></a>

## 本章的问题定义

已有旧策略生成的一批轨迹，希望在有限数据下改变策略，又不让分布变化破坏局部近似。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 旧策略 $\pi_{\rm old}$、其折扣占用分布 $d_{\rm old}$、优势估计 $\hat A$、KL 预算 $\delta>0$。

### 需要求解的对象

选择局部策略更新；真实外部目标是 J，局部 surrogate 只是可计算替代。

### 信息与数据权限

更新时不能立即知道新策略的完整占用分布；重复使用同批数据会逐渐偏离采样策略。

$$
\max_\theta\mathbb E_{s\sim d_{\rm old},a\sim\pi_{\rm old}}\!\left[\frac{\pi_\theta(a\mid s)}{\pi_{\rm old}(a\mid s)}\hat A(s,a)\right]\quad\text{s.t.}\quad\mathbb E_{d_{\rm old}}[D_{\rm KL}(\pi_{\rm old}\|\pi_\theta)]\le\delta
$$

这是常用平均 KL 信赖域近似。严谨性能界中的最坏状态分布偏差、优势误差与实践平均约束不能混为一谈。

### 成立条件与解的含义

- 旧策略支持需覆盖所评价动作；批内旧 log-prob 和优势版本固定。
- Fisher 与局部二次近似要求适当数值条件。

判断准则：同时记录实际 KL、clip fraction、收益和 value 误差；clipping 生效不是改善证明。

### 适用边界

- 声称 PPO clipping 严格限制所有状态的 KL 或保证单调提高收益。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 限制表示或采用近似 · [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)：用旧策略分布下的局部代理目标近似真实收益变化。

- 改变信息或数据协议 · [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)：多轮批内优化不同于每步只消费一次新经验。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

改变动作概率还会改变未来状态分布，旧数据只直接描述旧策略。

### 本章的核心思路

限制更新尺度以维持局部比较的可信度，并明确理论界与实际近似之间的差距。

1. [从真实性能差分找到近似位置](#lesson-derive)：用旧占用分布替代新占用分布是关键近似。

2. [用局部几何求受限步长](#lesson-local)：TRPO 借助 Fisher、共轭梯度和回溯近似求解信赖域问题。

3. [用裁剪代理简化优化](#lesson-ppo)：PPO 对有利的比率变化设置分支，但没有把实际约束精确解出来。

结论与条件：理想信赖域理论有优势和分布条件；实际平均 KL 与裁剪目标不自动继承全部保证。

### 相关方法改变了什么

- TRPO / PPO：分别近似求约束更新与优化裁剪代理目标。

- Clip / KL penalty：前者截取目标分支，后者在目标中惩罚偏离，二者都需观测实际分布变化。


<a id="lesson-setting"></a>

## 1 · 旧策略的数据与新策略的状态分布

上一章从固定采样策略的轨迹得到一个梯度估计。第一次更新后，动作概率已经改变，同一批轨迹却仍来自旧策略。若继续优化，既要计算这些旧动作在候选策略下的概率，也要判断旧经验还代表哪些未来状态。策略梯度只给出当前位置的上升方向；本章继续决定沿这个方向走多远，以及何时重新采样。

固定旧参数 $\theta_0$，记 $\pi_0=\pi_{\theta_0}$，候选策略为 $\pi_\theta$。分析先使用同一个奖励有界、$0<\gamma<1$ 的无限时域折扣 MDP，所有策略共用同一初始分布。$V^{\pi_0},Q^{\pi_0}$ 和 $A^{\pi_0}=Q^{\pi_0}-V^{\pi_0}$ 是旧策略的真实价值；实际可获得的是上一章用旧 critic 和轨迹算出的 $\hat A$。本页三步算例和配套训练取有限 episode、$\gamma=1$，后文会分别解释它们的采样目标。

$$
d_\pi(s)=(1-\gamma)\sum_{t=0}^{\infty}\gamma^t\Pr_\pi(S_t=s),\qquad r_\theta(s,a)=\frac{\pi_\theta(a\mid s)}{\pi_0(a\mid s)}
$$

$d_\pi$ 是归一化折扣状态分布，所有策略共用同一初始分布，$J(\pi)=\mathbb E_\pi\sum_t\gamma^tR_{t+1}$。概率比只修正给定状态中的动作分布，不会自动把旧状态分布变成新状态分布。

本章使用覆盖候选动作的旧策略。若旧策略对某个动作的概率为零，普通重要性比没有定义；小批量估计也无法凭空得到该动作的优势。

<a id="lesson-derive"></a>

## 2 · 从性能差分到局部 surrogate

$$
J(\pi_\theta)-J(\pi_0)=\frac{1}{1-\gamma}\mathbb E_{s\sim d_{\pi_\theta},a\sim\pi_\theta}[A^{\pi_0}(s,a)]
$$

将 $A=Q-V$ 展开为一步奖励与两个价值项，再沿新策略轨迹求折扣和；中间价值项望远镜相消。

$$
\mathbb E_{\pi_\theta}\sum_{t=0}^{N-1}\gamma^t A^{\pi_0}(S_t,A_t)=\mathbb E_{\pi_\theta}\!\left[\sum_{t=0}^{N-1}\gamma^tR_{t+1}+\gamma^N V^{\pi_0}(S_N)-V^{\pi_0}(S_0)\right]
$$

先在有限 N 上展开，才能看见尚未消去的尾值。奖励有界且 γ<1 使尾值趋零；共同初始分布下，最后的初始价值期望等于旧策略回报。再按 d 的定义重写折扣和，才得到上式。这一步不需要候选策略的 critic。

恒等式使用新策略的访问分布 $d_{\pi_\theta}$，但手中数据来自旧策略。先将状态权重固定为 $d_{\pi_0}$，再用动作概率比改写给定状态内的期望，得到局部代理目标（surrogate）。这两步中，动作换分布是支持条件下的等式，状态换分布才是近似。

$$
\begin{gathered}\bar A_\theta(s)=\sum_a\pi_\theta(a\mid s)A^{\pi_0}(s,a),\\L_{\pi_0}(\pi_\theta)=J(\pi_0)+\frac{1}{1-\gamma}\mathbb E_{d_{\pi_0},\pi_0}[r_\theta A^{\pi_0}],\\J(\pi_\theta)-L_{\pi_0}(\pi_\theta)=\frac{1}{1-\gamma}\sum_s[d_{\pi_\theta}(s)-d_{\pi_0}(s)]\bar A_\theta(s).\end{gathered}
$$

最后一式将真实性能差与代理的差逐项相减，直接指出代理遗漏了什么。连续状态、动作时将求和换成积分。

在 $\theta=\theta_0$，状态分布差为零，而且 $\bar A_{\theta_0}(s)=\sum_a\pi_0(a\mid s)[Q^{\pi_0}(s,a)-V^{\pi_0}(s)]=0$。对最后一式求导时，乘积的两个一阶项因而都为零。在策略和访问分布可微、可交换求导与求和的条件下，$L$ 与 $J$ 在旧参数处不仅值相同，梯度也相同。状态分布的作用被收集进旧策略的占用权重，而非从性能问题中消失。

$$
\left.\nabla_\theta L_{\pi_0}(\pi_\theta)\right|_{\theta_0}=\frac{1}{1-\gamma}\mathbb E_{d_{\pi_0},\pi_0}\!\left[\left.\nabla_\theta\log\pi_\theta(a\mid s)\right|_{\theta_0}A^{\pi_0}(s,a)\right]=\left.\nabla_\theta J(\pi_\theta)\right|_{\theta_0}
$$

由 $\nabla r_\theta=r_\theta\nabla\log\pi_\theta$，在旧参数处 $r=1$，接回上一章的 score 梯度。这里先对候选参数求导，再在旧参数处取值；远处候选的代理梯度不再由这个等式认证。

$$
J(\pi_\theta)\ge L_{\pi_0}(\pi_\theta)-C\alpha_{\rm TV}^{\,2},\qquad C=\frac{4\gamma\epsilon_A}{(1-\gamma)^2}
$$

TRPO §3 的界使用 $\epsilon_A=\max_{s,a}|A^{\pi_0}(s,a)|$ 与 $\alpha_{\rm TV}=\max_s D_{\rm TV}(\pi_0(\cdot\mid s),\pi_\theta(\cdot\mid s))$。它约束所有状态中最大的策略变化，仍使用真实优势。

用 $D_{\rm TV}^2\le D_{\rm KL}$ 可得更宽松的下界 $M(\theta)=L_{\pi_0}(\pi_\theta)-C\max_sD_{\rm KL}(\pi_0\Vert\pi_\theta)$。旧参数处 $M(\theta_0)=J(\pi_0)$，所以将这个精确下界提高到旧回报以上，才可推出真实性能不下降。只要求 KL 小、却未使下界提高，不足以完成这个推论。下一节先检查状态权重的差别，再将这一理论动机转成实际可计算的约束。

<a id="course-state-ratio"></a>

## 2.1 · 旧状态权重怎样限制这次比较

$$
\mathbb E_{s\sim d_{\pi_0},\ a\sim\pi_0}
\left[\frac{\pi(a\mid s)}{\pi_0(a\mid s)}A^{\pi_0}(s,a)\right]
=\mathbb E_{s\sim d_{\pi_0},\ a\sim\pi}A^{\pi_0}(s,a)
$$

在动作支持条件下，一步概率比把动作分布换成 π；等式右边的状态分布仍然是旧的 d。性能差分恒等式要求的是新策略的状态访问分布。

两步例子：起点以概率 p 进入支路，否则直接结束。旧策略 p=0.1，新策略 p=0.2。支路中的动作分布完全不变，所以该状态所有动作的概率比都是一；但支路实际出现的概率翻了一倍。只在支路样本上乘当地动作比，无法制造缺少的访问次数。完整轨迹或前缀比可以校正相应分布，但通常承受更大的方差。

这个两步例子隔离了状态权重变化；支路的动作未变，所以若使用真实旧优势，支路的动作平均优势仍为零。实际性能失配取决于上一节公式中两个因子的乘积：状态出现频率改变了多少，以及候选动作在这些状态下的平均旧优势是多少。不能仅因支路出现次数翻倍，就推断起点回报必然下降。

若旧策略的优势估计符号已经错误，还会出现另一种误差。限制更新幅度可以缩小沿错误方向的移动，却不能修正优势排序。后面的三步通关例将保留同一批轨迹，分别算样本代理和真实成功率；此处先假定真实优势可得，继续看怎样限制由策略变化引起的状态失配。

因此下一步要决定可接受的策略变化，而不是把每个样本动作的概率比当成全部访问频率的修正。TRPO 以策略距离限制这个变化，再用局部曲率求方向和尺度。参数很小的变化不一定产生小的动作分布变化，直接在动作分布上定义 KL 才给出了所需的比较对象。

<a id="lesson-local"></a>

## 3 · 从最大 KL 到局部 Fisher 与回溯

理论下界的惩罚系数可能使步子很小；逐个约束所有状态的最大 KL 又难以采样和优化。TRPO §4 先改为优化代理、约束 KL，再将最大状态 KL 换成旧访问分布下的平均 KL。这是实际算法的近似选择。平均变化小仍可能包含某些状态的较大变化，因而不能沿用刚才精确下界的逐步保证。

$$
\begin{gathered}S_B(\theta)=\frac1B\sum_{i=1}^{B}r_\theta(s_i,a_i)\hat A_i,\\K_B(\theta)=\frac1B\sum_{i=1}^{B}D_{\rm KL}(\pi_0(\cdot\mid s_i)\Vert\pi_\theta(\cdot\mid s_i)),\\\max_\theta S_B(\theta)\quad\text{s.t.}\quad K_B(\theta)\le\delta.\end{gathered}
$$

$B$ 是旧策略采到的状态动作行数，旧概率和 $\hat A_i$ 在整个优化中固定。这里直接写工程的均匀批均值。若要估计前文精确的折扣目标，采样或加权还须对应 $d_{\pi_0}$；均匀时间行不会自动具有这个分布。

即使能精确计算每个已采状态的动作分布 KL，上式也只在有限状态样本上限制策略。先在旧参数附近展开：$S_B(\theta_0+\Delta)\approx S_B(\theta_0)+g^\top\Delta$；$K_B(\theta_0)=0$，其梯度也为零，所以 KL 的首个非零项是二次项。

$$
\begin{gathered}\max_\Delta g^\top\Delta\quad\text{s.t.}\quad\tfrac12\Delta^\top F\Delta\le\delta,\\g=\nabla S_B(\theta_0),\qquad F=\left.\nabla_\theta^2K_B(\theta)\right|_{\theta=\theta_0}.\end{gathered}
$$

$g$ 是样本代理的梯度，$F$ 是同一旧参数处 KL 的局部曲率；两者采用同一批次权重。固定旧策略后，曲率描述的是候选策略改变多少。

$$
F=\frac1B\sum_i\mathbb E_{a\sim\pi_0(\cdot\mid s_i)}[u_i(a)u_i(a)^\top],\qquad u_i(a)=\left.\nabla_\theta\log\pi_\theta(a\mid s_i)\right|_{\theta_0}
$$

对归一化、光滑且支持不随参数改变的动作分布，$\mathbb E_{\pi_0}u_i=0$；对归一化式再求一次导数得到 $-\mathbb E\nabla^2\log\pi_\theta=\mathbb E uu^\top$。这就是 Fisher。TRPO §6 在每个状态上对动作分布解析求 KL 曲率；仅将实际采到的 score 外积平均，是另一种有限样本估计。

$$
x=F^{-1}g,\qquad\Delta_* =\sqrt{\frac{2\delta}{g^\top x}}\,x
$$

此式先假定 $F$ 正定且 $g\ne0$。拉格朗日条件给出 $g=\eta F\Delta$，再把二次约束取等号得到缩放。若梯度为零，则不需要更新；奇异的 $F$ 不能直接求逆。

神经网络不显式存储整个 $F$。给定向量 $v$，二次自动微分计算 $Fv=\nabla_\theta[(\nabla_\theta K_B)^\top v]$，在 $\theta_0$ 处取值。共轭梯度（CG）只需要这个乘法接口，就能近似解 $Fx=g$。正半定的 Fisher 可能因参数冗余而奇异；加入 $\kappa I$ 后求解的是 $(F+\kappa I)x=g$。阻尼改变了方向和所用的局部度量。

若 CG 只给出近似方向 $x$，原二次 KL 模型上的归一化使用 $\sqrt{2\delta/(x^\top Fx)}\,x$，要求 $x^\top Fx>0$。只有精确解 $Fx=g$ 时，分母才等于 $g^\top x$。若实现用阻尼曲率缩放，也要将其记录为另一个近似，再由真正的样本 KL 决定接受与否。

局部二次近似不保证候选点的真实 KL 合格。回溯依次尝试全步、半步、四分之一步，重新计算样本 surrogate 与实际样本 KL；未满足接受条件则恢复旧参数。回溯使用的仍是当前数据，不是对真实环境性能的证明。

<a id="lesson-ppo"></a>

## 4 · PPO 的符号分支与固定量

TRPO 每批需要 CG 和回溯。PPO 改用一阶优化器，在同一批数据上做多轮更新，并修改代理以减少过度更新的激励。PPO 原论文还给出自适应 KL 惩罚版本（§4）；这里展开的是 clipped 版本，取 $0<\varepsilon<1$。它承接 $S_B$ 的动作概率比，不是对上一节约束问题的精确求解。

$$
L^{\rm clip}(\theta)=\frac1B\sum_i\min\left(r_\theta(s_i,a_i)\hat A_i,\operatorname{clip}(r_\theta(s_i,a_i),1-\varepsilon,1+\varepsilon)\hat A_i\right)
$$

最大化这个固定数据上的目标；代码通常对其负数做梯度下降。

每一项都不大于未裁剪项，所以 $L^{\rm clip}\le S_B$。旧参数处 $r=1$ 位于区间内部，二者的值与梯度相同。这是对未裁剪样本代理的下界；$S_B$ 自身含状态分布替代、优势估计和有限采样误差，故这个逐项不等式尚未给出 $J$ 的下界。

$$
\ell(r,A)=\begin{cases}A\min(r,1+\varepsilon),&A\ge0,\\A\max(r,1-\varepsilon),&A<0.\end{cases}
$$

正优势动作的过度概率上升、负优势动作的过度概率下降，不再继续得到相同的 surrogate 奖励。

![直接计算ε=.2时正负优势的PPO单样本目标，显示正优势的右侧平台、负优势的左侧平台，与未裁剪rA比较。](https://yingwen.io/crl-figures/learning-classic-ppo-clip.svg)

原创精确函数图：ε=.2，固定A=±1，旧动作概率.25，r从.4到1.6（当前概率.1到.4），61个函数点；采样/训练更新均为0、种子不适用。不是PPO训练曲线；虚线是未裁剪目标。

先看正优势 A=1。r=.6 时，原目标为.6，裁剪项为.8，取小者仍为.6；r=1.4 时，原目标1.4超过裁剪项1.2，所以目标在右侧变平。再看负优势 A=−1：r=.6 时，两个候选是−.6与−.8，取小者为−.8，在左侧变平；r=1.4 时仍取−1.4，惩罚这个有害方向，而不是也在右侧清除梯度。

$$
\frac{\partial\ell(r,1)}{\partial r}=\begin{cases}1,&r<1+\varepsilon,\\0,&r>1+\varepsilon,\end{cases}\qquad \frac{\partial\ell(r,-1)}{\partial r}=\begin{cases}0,&r<1-\varepsilon,\\-1,&r>1-\varepsilon.\end{cases}
$$

这里只列拐点之外的导数。对策略参数求导还要乘 ∇θr；拐点是不可微点，由实现选择相应次梯度。图中的水平段是不再奖励这一单项继续改善，不是给概率设置硬边界。

这也解释数据冻结的作用。一次 rollout 先保存旧动作概率，再以旧 critic 计算优势；批内多轮优化仅让当前动作概率随 θ 改变，不能顺手把分母或优势重算成新值。停止对旧概率、优势和 critic target 求导，隔离的是本次优化中的责任；它不保证优势准确，也不补回新策略的状态分布。下一批 on-policy 数据到来后，才重新定义这一轮的旧策略。

本图按函数公式直接计算，不会因选择随机种子而改变。[数据 JSON](/crl-figures/learning-classic-data.json) 与 `scripts/generate-crl-learning-classic.mjs` 保存每个函数点。实际学到的策略与回报应看本章后面的可复现实验，而不是把横轴 r 看作训练步数；两张单项图也不能代替整批共享参数下的更新分析。

clipping 不是把网络参数投影回某个约束集，也不强制所有概率比留在区间内。共享网络会联动改变其他状态和动作，样本不覆盖的地方更没有直接约束。多轮更新时，旧 log-probability、原始优势和 critic target 全部固定；只有当前策略、当前 critic 和优化器状态变化。

这种差别在完整覆盖的单状态例子中也存在。旧概率各半，真实旧优势为 $(1,-1)$，$\varepsilon=0.2$，候选概率为 $(p,1-p)$。只要 $p\ge0.6$，两个动作按旧概率加权的 clipped 目标合计为 0.2；从 p=0.6 到接近一，目标完全平坦，但旧到新 KL 为 $-\tfrac12\log(4p(1-p))$，可任意大。平台停止提供额外奖励，却没有将策略限制在一个 KL 信赖域中。

配套代码记录 $\widehat{\mathrm{KL}}=\operatorname{mean}(r-1-\log r)$。在旧策略采样、归一化策略与适当覆盖下，其期望对应旧到新 KL。单批估计仍有误差。代码在更新前检查这个量并提前停止，最后一次更新仍可能越过阈值，所以它不是硬约束。

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

<a id="lesson-algorithm"></a>

## 5 · 两类算法的精确更新次序

**算法：这里列分离 actor/critic 的更新规约：PPO 有配套全批采样实现，TRPO 提供数值求解核。每个全批 actor 轮写入一次参数，B 行不被再次计作交互；TRPO 的回溯检查是非线性的整批检查。**

1. 第 k 批开始：保存本批旧策略参数和旧 critic 快照。
1. 固定采样参数，收集 B 行新经验，保存实际动作的 old logp、旧 value、next value 和两类边界。
1. 由旧 value 后向计算原始 GAE；本教学代码固定 target = 原始 GAE + 旧 value。
1. 另建 actor 优势副本并标准化；保留 critic 标签的奖励单位。
1. 选择一种 actor 更新：
  1. TRPO：在旧参数处计算 g 与曲率接口，用 CG 求方向并缩放。
    1. 候选参数始终从旧参数加回溯步得到；整批检查样本 KL 和代理改善。
    1. 无候选通过则恢复旧参数。
  1. PPO：重复最多 K 轮，每次读取固定 old logp 和 actor 优势副本。
    1. 在当前参数处计算 ratio、clipped loss 和样本 KL。
    1. 若 KL 已超过阈值则退出；否则反传并更新 actor。
1. 用固定 target 更新 critic；检查实际 actor/critic 优化步数及更新后 KL。
1. 保留未终止的环境活动，丢弃本批优化数据，以更新后的策略收集下一批。

PPO 原论文 §5 的 Algorithm 1 每批由 $N$ 个 actor 各采 $T$ 步，再对 $B=NT$ 行做 $K$ 个 epoch、小批量大小 $M\le B$ 的优化。一轮 epoch 通常包含多次参数写入；本地代码取 $M=B$，最多四次 actor 更新。批数、epoch 数、实际优化步数和环境交互步数因而不能互换。没有一个仅由 $\varepsilon$ 决定的通用复用次数；步长、优势尺度、共享参数、优化器状态和数据覆盖共同影响变化，KL 监测用于观测实际偏离。

共享 encoder 时，critic 回归也会改变动作分布。PPO 原论文 §5 可将策略代理、价值损失和熵奖励合并训练；此时即使某个策略项已经平坦，其他梯度仍能改变该概率。TRPO 若先接受 actor 步、随后再让共享 critic 参数改变策略，先前的 KL 接受条件就不再描述最终策略。必须把共享更新后的分布纳入检查，或明确采用本页的分离网络。

对 logits 张量做两次自动微分，检查 categorical 分布的 KL 曲率；这段代码尚未对整个 actor 参数求导。

```python
def fisher_vector_product(logits, vector, damping=0.):
    old_prob = logits.detach().softmax(-1)
    old_logp = logits.detach().log_softmax(-1)
    kl = (old_prob*(old_logp-logits.log_softmax(-1))).sum(-1).mean()
    gradient = torch.autograd.grad(kl, logits, create_graph=True)[0]
    product = torch.autograd.grad((gradient*vector).sum(), logits)[0]
    return product+damping*vector
```

代码中的自变量是 logits。对状态 $s$ 的 categorical 分布，其曲率为 $F_z=\operatorname{diag}(p)-pp^\top$。若 logits 来自网络 $z_\theta(s)$，令 $J_s=D_\theta z_\theta(s)$，参数空间的单状态 Fisher 才是 $J_s^\top F_zJ_s$；再按状态样本求平均。实现完整网络 TRPO 时，可直接对参数计算 KL 的 Hessian-vector product，或按这个 Jacobian 路径计算乘法。输出空间核的检查只覆盖这条路径中的一部分。

<a id="lesson-example"></a>

## 6 · 数值检验：TRPO 的步长与 PPO 的轨迹

单状态两个动作，$\pi_\theta(1)=\sigma(\theta)$，旧 $\theta=0$，优势分别为 $1,-1$。这里对归一化占用下的 $\mathbb E[r_\theta A]$ 求导，故 $g=0.5$，$F=0.25$；保留性能 surrogate 的 $1/(1-\gamma)$ 时，$g$ 也乘此正数，但它在约束全步的归一化中消去。令 KL 半径 $\delta=0.01$，局部全步为 $\Delta=\sqrt{0.08}\approx0.28284$。新动作一概率约为 $0.57024$，实际旧到新 KL 为 $\log\cosh(\Delta/2)\approx0.009967$，因此全步满足这一例的约束。

现在沿用策略梯度章的三步通关任务：在 $s_0,s_1,s_2$ 选择前进才能继续，其他动作立即以零奖励终止；从 $s_2$ 前进得到 $+1$ 并终止。旧策略在三处前进概率均为 $.5$。给定一条成功轨迹，奖励为 $(0,0,1)$，旧 critic 为 $(.2,.8,.4,0)$；$\gamma=1,\lambda=.5$ 得到原始 GAE $(.55,-.1,.6)$、固定 critic 标签 $(.75,.7,1)$。本算例不标准化 actor 优势，以便沿用同一组手算数值。

设候选策略的前进概率为 $(.7,.3,.55)$，而分母仍为旧概率 $.5$。于是 $r=(1.4,.6,1.1)$。取 $\varepsilon=.2$，三项 clipped 目标依次为 $(.66,-.08,.66)$：第一项正优势进入右平台，第二项负优势进入左平台，第三项仍有斜率。对各自当前 $\log\pi_\theta(a_t\mid s_t)$ 的偏导为 $(0,0,.66)$；变成参数梯度时，还要乘各自的 score gradient，并按 batch 求平均。

![同一条三步成功轨迹的奖励和旧价值，生成TD残差、GAE和固定价值标签，再比较三个样本动作的新旧概率与clipped目标，最后独立计算通关概率。](https://yingwen.io/crl-figures/gae-ppo-walkthrough.svg)

从上往下跟随同一批三个样本。概率条长度表示实际动作概率，不是训练时间；旧概率、原始优势与价值标签在本批优化中固定。$\gamma=1,\lambda=.5,\varepsilon=.2$；候选概率人为给定，随机 rollout 与优化步数均为 0。下方按完整小任务精确计算成功概率，因而不复用训练 loss 当评价。[数值](/crl-figures/gae-ppo-walkthrough-data.json)与[标准库计算](/crl-code/tutorials/gae_ppo_walkthrough.py)可复算；GAE 依据原论文 §3，clipping 依据 PPO §3、§5。

单项平坦仍允许其他样本推动同一个参数。令三个二动作策略都有 $p_i(w)=\sigma(z_i+w)$，选取 $z_i$ 使 $w=0$ 时概率恰为上述候选值。整批目标对共享 $w$ 的导数为 $.66(1-.55)/3=.099$。若只作一次步长 $.1$ 的解析梯度步，$w$ 增至 $.0099$，第一项的概率也继续增加，$r_0$ 超过原来的 $1.4$，虽然它自己的 clipped 项没有梯度。critic 的共享参数梯度、熵项和优化器动量也可能改变这些概率。

样本 ratio 也不能控制未采到的动作。另取一个三动作状态，旧分布为 $(.5,.25,.25)$，新分布为 $(.5,.5-\eta,\eta)$，$0<\eta<.5$。若只采到第一个动作，ratio 始终为一；但完整 $D_{\rm KL}(\pi_0\Vert\pi_\theta)=.25\log\frac{.25}{.5-\eta}+.25\log\frac{.25}{\eta}$ 在 $\eta\to0$ 时无界增大。clip 区间因此既不是所有动作概率比的约束，也不是完整 KL 的上界。

最后回到通关任务独立评价。旧策略成功概率为 $.5^3=.125$，候选策略为 $.7\times.3\times.55=.1155$。本批 clipped 均值却从 $(.55-.1+.6)/3=.35$ 上升到 $(.66-.08+.66)/3\approx.4133$。真实旧价值为 $(.125,.25,.5)$，所以中间状态选择前进的真实优势为 $.5-.25=.25>0$，与估计的 $-.1$ 相反。给定 critic 的错误加上一条轨迹的有限样本，足以让 surrogate 改善与起点回报方向不同。这里评价来自已知小 MDP 的精确枚举；真实任务应另取未参与更新的评估回合，并分别记录交互预算、更新次数和评估分布。

共轭梯度、单状态精确 KL 回溯和 PPO 符号分支。这个 TRPO 数值核不是完整神经网络训练器。

```python
def ppo_term(ratio, advantage, clip=0.2):
    clipped = min(1 + clip, max(1 - clip, ratio))
    return min(ratio * advantage, clipped * advantage)


def conjugate_gradient(matvec, b, iterations=20, tolerance=1e-12):
    x, residual = [0.0] * len(b), list(b)
    direction = list(residual)
    rr = dot(residual, residual)
    for _ in range(iterations):
        if rr <= tolerance * tolerance:
            break
        product = matvec(direction)
        curvature = dot(direction, product)
        if curvature <= 0:
            raise ValueError('positive curvature is required')
        step = rr / curvature
        x = [v + step * d for v, d in zip(x, direction)]
        residual = [r - step * p for r, p in zip(residual, product)]
        new_rr = dot(residual, residual)
        direction = [r + new_rr / rr * d
                     for r, d in zip(residual, direction)]
        rr = new_rr
    return x


def categorical_kl(old, new):
    return sum(p * math.log(p / q) for p, q in zip(old, new) if p > 0)


def trpo_binary(theta, advantage_one, advantage_zero, delta=0.01):
    """Exact scalar Fisher + actual KL/line search for one-state policy."""
    prob = lambda z: 1.0 / (1.0 + math.exp(-z))
    old_p = prob(theta)
    surrogate = lambda z: prob(z) * advantage_one + (1-prob(z)) * advantage_zero
    gradient = old_p * (1-old_p) * (advantage_one-advantage_zero)
    if abs(gradient) < 1e-14:
        return theta, 0.0, 0.0
    fisher = old_p * (1-old_p)
    natural = gradient / fisher
    full_step = math.sqrt(2*delta/(natural*fisher*natural)) * natural
    for power in range(20):
        step = 0.5**power * full_step
        candidate = theta + step
        kl = categorical_kl([old_p, 1-old_p], [prob(candidate), 1-prob(candidate)])
        gain = surrogate(candidate)-surrogate(theta)
        if kl <= delta and gain >= 0.1 * gradient * step:
            return candidate, kl, gain
    return theta, 0.0, 0.0
```

<a id="lesson-code"></a>

## 7 · 可运行 PPO 与实现边界

先运行 [gae_ppo_walkthrough.py](/crl-code/tutorials/gae_ppo_walkthrough.py)：`python3 gae_ppo_walkthrough.py` 输出轨迹、残差、优势、固定标签、clipped 项与独立评价；加 `--test` 运行精确分数、边界、有限差分和冻结量检查。它只用标准库，不进行随机训练。输出的 shifted_probability 另行计算给定步长的一次假设解析更新，用于检查共享参数联动；它不是图中候选概率的训练来源。元数据将这一次解析计算与零次训练优化器调用分开记录。

旧概率和优势来自固定 rollout。critic 使用未标准化的回归目标。

```python
def ppo_update(actor, critic, actor_optimizer, critic_optimizer, x, actions,
               old_logp, advantages, returns, epochs=4, clip=.2, target_kl=.03):
    # Old log-probabilities, advantages and targets stay fixed for all epochs.
    old_logp, returns = old_logp.detach(), returns.detach()
    advantages = advantages.detach()
    advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    updates, final_kl = 0, 0.0
    for _ in range(epochs):
        distribution = Categorical(logits=actor(x))
        logp = distribution.log_prob(actions)
        ratio = (logp-old_logp).exp()
        kl = ((ratio-1)-(logp-old_logp)).mean()
        if float(kl.detach()) > target_kl:
            break
        surrogate = torch.minimum(ratio*advantages,
                                  ratio.clamp(1-clip, 1+clip)*advantages)
        loss_actor = -surrogate.mean()
        actor_optimizer.zero_grad()
        loss_actor.backward()
        actor_optimizer.step()
        updates += 1
    for _ in range(epochs):
        prediction = critic(x).squeeze(-1)
        assert prediction.shape == returns.shape
        loss_critic = ((prediction-returns)**2).mean()
        critic_optimizer.zero_grad()
        loss_critic.backward()
        critic_optimizer.step()
    with torch.no_grad():
        difference = Categorical(logits=actor(x)).log_prob(actions)-old_logp
        final_kl = float((difference.exp()-1-difference).mean())
    return {'actor_updates': updates, 'sample_kl': final_kl,
            'value_loss': float(loss_critic.detach())}
```

将 deep_textbook_train.py 与 deep_textbook_lab.py 放在同一目录。安装 deep_requirements.txt 后执行 python3 deep_textbook_train.py ppo --epochs 16 --seed 0。该命令在内置小 MDP 中真实采样、计算 GAE、做 PPO 更新，再于独立环境评估；没有 Gym 或 GPU 依赖。命令中的 epochs 是外层采样批数，每批默认128个交互步；ppo_update 内部最多做四个全批 actor 步，KL 提前停止后实际步数可能更少。返回的 actor_updates 只来自最后一批，不能乘外层批数恢复整个训练的更新总数。

这段代码的 $\gamma=1$ 对应有限期限任务；批内 actor 使用均匀行、标准化优势。它是明确的教学优化循环，不能把每个批均值直接当成前文折扣占用恒等式中的精确期望。代码的 initial_greedy_return 与 final_greedy_return 又使用确定性 argmax 策略，因此评价对象与随机采样策略不同。三步手算独立计算的是随机策略通关概率；比较两种读数前要先选定同一个策略和回报准则。

完整 TRPO 工程入口是 Spinning Up 的 TensorFlow 1 实现；该项目没有对应的官方 PyTorch TRPO 目录。配套 PyTorch 代码只提供 Fisher-vector product，标准库代码提供 CG 与一维回溯。完整网络 TRPO 还需参数向量化、分布接口和整批 line search。

对照 Spinning Up 的 PyTorch PPO，`finish_path` 用 TD 残差计算 GAE，却用带尾值的 rewards-to-go 训练 critic；`get` 只标准化 actor 优势。`compute_loss_pi` 读取固定 old logp 和优势，`update` 重复更新策略，再回归固定 ret。它用 mean(old_logp−logp) 估计 KL，并在下一次策略步之前检查 1.5×target_kl；这与前文配套代码的非负样本估计形式不同，二者都不是硬约束。原工程的旧 Gym 循环会在 epoch 切片末重置环境，不能从 buffer 的自举公式推断它保留了不中断的环境轨迹。

<a id="lesson-branches"></a>

## 8 · 从批内优化到持续控制的评价

- 旧策略被覆盖：若每轮内重新保存当前 log_prob 作为 old，概率比总接近一，算法不再控制相对于采样策略的变化。
- critic target 漂移：每轮重新用已更新 critic 生成优势，会把多种变化混在同一目标中。
- 小 batch 的 KL 噪声：阈值过小可能几乎不更新，阈值过大又不能有效限制分布迁移。
- 变化环境：旧策略数据可以在收集完时就不再代表当前动力学。KL 小只表示策略分布接近，不表示环境未变。

PPO 的参数复用和 replay 不是一回事。它复用的是最近采样策略的一批数据，并保存其动作概率。把任意陈旧任务轨迹放入同一循环，不能只保留 clipped loss 就称为正确的 off-policy 算法。

还要区分旧行为与旧世界。在同一平稳 MDP 中，动作比只将给定状态内的行为概率替换为候选概率，状态访问仍需另行处理；奖励或转移已经改变时，过去经验所反映的条件后果也变了。固定旧 logp 能保留采样出处，却不能校正这种世界变化。状态表示或归一化器在采样中更新时，也应将其版本视为采样策略的一部分，才能解释同一输入上的概率比较。

评价先回答两个不同问题。若要问一次更新是否改善声明的策略回报，可冻结候选参数及归一化统计，在相同初始分布与回报准则下另采评估轨迹；随机策略与 argmax 策略分开报告。若要问持续学习器是否适应变化，应在不额外重置世界的主经验流上保留学习与行动，记录变化后累计收益、恢复时间和行动覆盖，并计入每次交互间的计算。前者隔离策略性能，后者包含采样、更新和后续经验的反馈。

在三步任务里，中间前进动作的估计优势为负，导致候选前进概率减小。KL 或 clip 可以减少这次错误更新的幅度；下一批还需实际到达中间状态、获得新反馈，critic 才有机会修正排序。持续任务中的旧偏好同样会改变获取反证的机会。比较更新限制时，应固定表示、critic 估计方式和交互预算，再记录实际更新次数、策略变化与独立收益；只见 clipping fraction 下降，还无法判断适应是否加快。

<a id="lesson-check"></a>

## 9 · 练习与答案

- 问：KL 小是否证明回报提高？答：不证明。真实优势、覆盖与状态分布误差仍有影响。
- 问：把所有 ratio 直接 clip 后再乘优势，等价于 PPO 吗？答：不等价。PPO 是两个目标取最小，某些变坏方向仍必须保留梯度。
- 问：为何使用 CG 而不直接求逆？答：网络参数很多，存储曲率矩阵需要平方级空间；Hessian-vector product 不需要显式矩阵。
- 实验：把一维例子的 δ 增大，比较二次预测 KL 与实际 KL，并观察回溯是否缩步；再将优势都设为零，确认参数保持不变。

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

- [Schulman et al. · Trust Region Policy Optimization, arXiv v5](https://arxiv.org/abs/1502.05477v5)：本章采用此版奖励最大化记法。§2 Eqs.(1)–(4)：性能差分与一阶对齐；§3 Eqs.(8)–(10)：最大状态差异与精确下界；§4–6：平均 KL、有限样本和实际近似；Appendix C：CG、方向缩放与非线性回溯。

- [Schulman et al. · TRPO, PMLR 2015 发表版](https://proceedings.mlr.press/v37/schulman15.html)：保留正式发表入口。§2 使用成本最小化，§3 的界和编号也与 arXiv v5 不同；核对公式时先确认版本。

- [Schulman et al. · Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)：§2–3 Eqs.(3)–(7)：平均 KL 与 clipped 样本代理；§4：自适应 KL 惩罚；§5 Eq.(9) / Algorithm 1：共享损失、多轮小批量优化和下一批采样。

- [Sutton & Barto · Reinforcement Learning, second edition](http://incompleteideas.net/book/the-book-2nd.html)：§13.2 pp.324–326：策略改变行动和状态访问；§13.3–13.5 pp.327–332：采样估计、baseline 与 critic 的评价作用。

- [Spinning Up · TRPO](https://spinningup.openai.com/en/latest/algorithms/trpo.html)：用作算法与实现接口的对照；本章解释和配套小实验独立编写。

- [Spinning Up · PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)：用作算法与实现接口的对照；本章解释和配套小实验独立编写。

- [Spinning Up · trpo.py](https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/trpo/trpo.py)：工程辅助：官方 TensorFlow 1 TRPO，实现 CG、Hessian-vector product 和整批回溯。

- [Spinning Up · ppo.py](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py)：工程辅助：官方 PyTorch PPO 与 KL 提前停止；其 critic 标签和采样边界需按代码核对。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-trust-region#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-trust-region#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-deep-trust-region)
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
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

对应原始材料：Sutton & Barto §13.2–13.5；TRPO §2–6、Appendix C；PPO §2–5。本文为原创讲解，原书、论文与上游代码保留各自许可。
