# 通用价值函数与预测知识

通用价值函数把一个预测问题写成目标策略、累积信号与延续规则。以“沿墙行走至充电点的能耗”为例，本章从问题定义推导 Bellman 方程，再逐步构造 on-policy 与 off-policy 学习算法。

## 本章内容

- 从一个自然语言问题写出 cumulant、continuation、目标策略和状态条件，分清问题与学习器。
- 独立推导 Bellman 方程、线性 TD、资格迹、GTD2/TDC、GTD($\lambda$) 与 Emphatic TD 的更新。
- 逐行运行多问题共享经验的学习循环，检查解析解、off-policy 发散反例与实现时序。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 价值函数与条件期望

$v_\pi(s)$ 是在状态 s 出发、以后按 $\pi$ 行动时，一个指定未来累计量的条件期望。改变后续行为或累计信号，就改变了问题，即使物理状态相同。

### Bootstrap 与半梯度

用当前预测的下一状态值构造训练目标叫 bootstrap。更新当前预测时，把这个目标暂时当常数叫半梯度；不是把整条 Bellman 残差对全部参数求导。

### 固定特征与线性逼近

x(s) 是给定的 d 维特征，w 是学习的 d 维参数。one-hot 特征让每个状态有自己的参数，退化为表格；一般特征会让不同状态的更新相互影响。

$$
\hat v_w(s)=w^\top x(s)
$$

### 行为策略与目标策略

b 实际选动作并产生数据；$\pi$ 是问题中假想后续采用的策略。预测“若一直向右”的后果，不要求真实机器人永远向右，但需要相应经验覆盖。

<a id="lesson-setting"></a>

## 1 · 通用价值函数的定义

General Value Function（通用价值函数）把通常只问“未来有多少奖励”的价值预测，扩展成“在某种行为条件下，未来某种信号会累计成什么”。GVF 定义预测对象，学习算法估计这个对象。二者可以分别选择：一个网络可以表示多个 GVF，同一个 GVF 则可以由表格 TD、线性 GTD 或其他预测方法学习。

本章属于预测，而不是策略改善。普通价值预测先固定任务奖励与策略；GVF 进一步允许指定其他累积信号和延续规则。两者都在回答给定行为条件下会发生什么。控制则还要规定行为的优化目标，并据预测改变动作选择。行为策略可以在实际运行中不断变化，但每个 GVF 的目标策略仍须单独定义。

| 概念 | 描述的对象 | 与 GVF 的关系 |
| --- | --- | --- |
| 预测问题 | 要估计的条件期望 | GVF 用目标策略、信号与延续给出这个问题的规格 |
| 学习算法 | 经验如何更新估计 | TD、GTD、ETD 等是求解这些预测问题的算法 |
| Agent state | 决策与预测使用的历史摘要 | 多个预测可作为其特征，但预测数量多不保证状态满足 Markov 性 |
| 环境模型 | 动作或 option 导致的奖励与后继信息 | 适当设计的一组 GVF 可表达模型的部分输出；任意一个 GVF 不等于完整转移模型 |
| 控制规则 | 根据后果选择何种行为 | 需要另行定义奖励目标、选择规则与探索机制 |

| 组成 | 要决定什么 | 充电点例子 |
| --- | --- | --- |
| 状态条件 s | 从什么信息出发提问 | 机器人当前区域和足以决策的历史摘要 |
| 目标策略 $\pi(a\mid s)$ | 后续假设采取什么行为 | 以 0.8 的概率向前，否则等待 |
| 累计信号 $C_{t+1}$ | 每次转移累计什么量 | 这一步的耗电量 |
| 延续因子 $\gamma_{t+1}\in[0,1]$ | 到达后是否继续累计，或者如何折扣 | 到充电点为 0，其他地方为 1 |
| 答案 v(s) | 给定上述规格，未来累计量的期望 | 到下次充电点的期望总耗电 |

本章先假定状态 s 是 Markov 的，环境、目标策略和特征在分析期间固定，且累计量存在。Markov 指给定当前状态和动作后，预测下一步不再需要完整历史。若只输入不充分的摄像头画面，不能直接沿用后面的精确 Bellman 方程或线性收敛结论。我们最后再讨论表示和预测问题本身不断变化的 CRL 情形。

数据接口是一条条 $(S_t,A_t,S_{t+1})$、传感器信号和行为概率 $b(A_t\mid S_t)$。一个 question 函数把每条真实经验变成自己的 $(C_{t+1},\gamma_{t+1},\pi(A_t\mid S_t))$。真实奖励可以是传感器信号之一，但不必是这个 GVF 的 cumulant。

<a id="gvf-semantics"></a>

## 2 · 累积信号、延续与预测语义

| 要问的内容 | $C_{t+1}$ | $\gamma_{t+1}$ | 答案的单位和含义 |
| --- | --- | --- | --- |
| 到充电点的步数 | 每一步为 1，包括到达那一步 | 到达为 0，否则 1 | 步；须保证目标策略下到达时间有有限期望 |
| 到充电点的能耗 | 该步耗电 | 到达为 0，否则 1 | 能量；不自动等于步数 |
| 最终是否会到达 | 到达指示为 1，否则 0 | 到达为 0，否则 1 | 首次到达概率；事件后停止计数，不能反复加 1 |
| 较近的到达有多大可能 | 到达指示 | 到达为 0，否则 0.9 | 折扣到达量 E[0.9^(T−1)1{到达}]，一般不是最终到达概率 |

$$
G_t=C_{t+1}+\gamma_{t+1}C_{t+2}+\gamma_{t+1}\gamma_{t+2}C_{t+3}+\cdots
$$

第一笔信号不折扣。$\gamma_{t+1}$ 作用于这次转移之后的余项，不作用于已经收到的 $C_{t+1}$。因此在到达事件上把 $\gamma$ 设为 0，仍会保留到达那一步的信号。

$$
G_t=\sum_{k=0}^{\infty}\left(\prod_{j=1}^{k}\gamma_{t+j}\right)C_{t+k+1},\qquad v_{\pi,c,\gamma}(s)=\mathbb E_\pi[G_t\mid S_t=s]
$$

k=0 的空乘积为 1。$\gamma$ 可以依赖转移；本页程序按到达状态计算。固定 $\gamma$<1 是熟悉的指数折扣；事件终止可以让某些地方 $\gamma$=1，也可以完全不重置真实环境。

若 C 是传感器温度而 $\gamma$=0.9，预测的是折扣温度总和，量级约等于温度的十倍，不是“十步以后的温度”。若想问第十步温度，需要重新设计预测结构/时间状态；若想近似未来平均温度，常数折扣下可考虑 $(1-\gamma)v$，但仍不是固定十步窗口平均。

旧 Horde/RLPark 接口还显式提供普通信号 r 和停止值 z，组合为 $c=r+(1-\gamma)z$。读这类源码时应先转换为统一 cumulant，再比较更新式。例如到达时 $\gamma$=0，才把 z 完整加入。不能既把 z 加进 c，又在 TD target 里重复加一次。

<a id="lesson-derive"></a>

## 3 · Bellman 方程

$$
G_t=C_{t+1}+\gamma_{t+1}G_{t+1}
$$

把无限和拆成第一项与剩下的同一类问题。这一步只用代数，还没有学习算法。

$$
v(s)=\sum_a\pi(a\mid s)\,\mathbb E[C_{t+1}+\gamma_{t+1}v(S_{t+1})\mid S_t=s,A_t=a]
$$

在状态 s 上取条件期望；Markov 条件允许用 $v(S_{t+1})$ 代替未来余项的条件期望。这里是按 $\pi$ 加权，不是对动作取 max：我们在评估一个给定行为问题，还没有优化行为。

$$
\begin{gathered}r_c(s)=\mathbb E_\pi[C_{t+1}\mid s],\quad M(s,s')=\mathbb E_\pi[\gamma_{t+1}\mathbf1\{S_{t+1}=s'\}\mid s]\\v=r_c+Mv,\qquad v=(I-M)^{-1}r_c\end{gathered}
$$

M 是已经含延续权重的转移矩阵，不一定每行和为 1。若谱半径小于 1，逆矩阵存在。统一 $\gamma\le\gamma_{\max}<1$ 是一个充分条件；事件终止时可通过转移结构满足该条件，不要求每个 $\gamma$ 都小于 1。

“解方程”适用于已知模型。本章用它给小实验提供独立正确答案。机器人通常不知道 M 和 $r_c$，因此需要从经验估计固定点。TD 的意义正是每收到一条转移就改进一次答案，而不等模型和完整回报都准备好。

$$
q(s,a)=\mathbb E[C_{t+1}+\gamma_{t+1}\sum_{a'}\pi(a'\mid S_{t+1})q(S_{t+1},a')\mid s,a]
$$

如果问题还条件于“第一步选 a”，就得到 action-value 形式；下一步仍对指定 $\pi$ 求期望。把这个期望换成 max，会把固定行为的预测问题变成控制问题。

<a id="gvf-td"></a>

## 4 · 线性 TD(0)

$$
\begin{gathered}y_t=C_{t+1}+\gamma_{t+1}w_t^\top x_{t+1},\quad\delta_t=y_t-w_t^\top x_t\\\widetilde L_t(w)=\tfrac12(y_t-w^\top x_t)^2,\qquad w_{t+1}=w_t+\alpha\delta_t x_t\end{gathered}
$$

先用旧参数得到 target，随后只对当前预测求导。$\delta$>0 说明这次 target 比预测高，要提高相关特征的权重。one-hot 时只改变当前状态的一个表项。

- 初始化 w（例如全 0）；观察当前状态并计算 $x_{t}$。
- 按行为策略选一次动作；保存动作概率，执行后收到传感器和下一状态。
- 用 question 算 c、$\gamma_{t+1}$、目标动作概率。
- 用旧 w 算当前预测、下一预测和 $\delta$；一次性更新 w。
- 转到下一状态继续；GVF 停止只影响它的余项，不自动重置机器人。

对 y 中的 w 也求导会得到 residual-gradient 型更新，方向含 $x_t-\gamma_{t+1}x_{t+1}$。这并不是同一算法。进一步要把单次转移的平方残差与期望 Bellman 残差区分开：对后者构造无偏梯度涉及独立下一状态样本等问题。因此，相同的单样本 TD error 可以对应不同的优化目标与更新方向。

<a id="gvf-traces"></a>

## 5 · 多步预测与资格迹

$$
G_t^{(n)}=\sum_{k=0}^{n-1}\left(\prod_{j=1}^{k}\gamma_{t+j}\right)C_{t+k+1}+\left(\prod_{j=1}^{n}\gamma_{t+j}\right)\hat v(S_{t+n})
$$

n-step target 累积 n 步真实信号后才接预测。n 大时通常减少对 bootstrap 的依赖，但需要更多未来数据，方差和等待时间也会改变。

$$
\begin{gathered}G_t^\lambda=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}\\G_t^\lambda-\hat v(S_t)=\sum_{k=0}^{\infty}\left(\prod_{j=1}^{k}\gamma_{t+j}\right)\lambda^k\delta_{t+k}\end{gathered}
$$

这里先取固定预测参数、on-policy 和收敛的级数；第二式由 bootstrap 项相消得到。$\lambda$=0 只剩当前 TD error。有限终止轨迹的 $\lambda$=1 对应完整回报极限，不能把任意无限无折扣问题也照搬。

$$
e_t=\gamma_t\lambda e_{t-1}+x_t,\qquad w_{t+1}=w_t+\alpha\delta_t e_t
$$

把同一个未来误差应对过去特征产生的影响，压缩到资格迹 e。迹延续到当前使用 $\gamma_{t}$；target 离开当前使用 $\gamma_{t+1}$。固定参数的前后向恒等式不意味着普通在线 accumulating TD($\lambda$) 在有限步长下逐次等于所有 forward-view 更新；true-online 方法正是处理这种在线等价性的一条路线。

| 两步轨迹，$\alpha$=.1、$\lambda$=.8 | e | $\delta$ | w |
| --- | --- | --- | --- |
| 初始化；s0→s1，c=0，$\gamma_{t+1}$=.9 | (1,0) | 0 | (0,0) |
| s1→终点，c=1，$\gamma_{t+1}$=0 | (.72,1) | 1 | (.072,.1) |
| 下一次问题开始，$\gamma_t$=0 | 旧迹不再延续 | 重新由当前样本决定 | 已学 w 不清零 |

$\lambda$ 不是“记住更久的状态”。它影响学习信用；当前输入仍可能不包含历史。一个只看当前画面的函数加上资格迹，不会因此自动变成能辨别所有历史的状态表示。

<a id="gvf-offpolicy"></a>

## 6 · 离策略预测与重要性采样

$$
\rho_t=\frac{\pi(A_t\mid S_t)}{b(A_t\mid S_t)},\qquad \mathbb E_{A\sim b}[\rho f(A)\mid s]=\sum_a\pi(a\mid s)f(a)
$$

条件是 $\pi(a\mid s)$>0 时 b(a|s)>0。若问题问一个真实行为永不尝试的动作，仅乘比率不能制造缺失信息。比率使用采样时的行为概率；更新策略后重新计算这个概率会改变含义。

$$
e_t=\rho_t(\gamma_t\lambda e_{t-1}+x_t),\qquad w_{t+1}=w_t+\alpha\delta_t e_t
$$

这是本章 ordinary importance-sampled TD($\lambda$) 的约定。$\lambda$=0 退化为 $\alpha\rho\delta x$ 更新。对旧迹继续乘当前 $\rho$，会形成历史比率乘积，可能产生很大方差。不同 off-policy return / trace 设计不是任意可互换的。

这个换测度只把给定状态下的动作分布 b 改成 $\pi$；状态仍由真实行为的长期访问分布 $d_b$ 提供。在表格覆盖充分的小问题上可能工作得很好，但函数逼近、bootstrap 和 off-policy 组合后，不保证稳定。先把“回答正确策略的问题”和“数值学习能稳定”分成两件事。

例：一个状态、两动作，动作 1 的 cumulant 为 1，动作 0 为 0，$\gamma$=.5；$\pi$ 永远选 1，b 均匀。问题答案是 2，不用 IS 则通常学到行为策略的答案 1。这个例子揭示换题错误，却还不是发散反例。

<a id="gvf-projection"></a>

## 7 · 线性函数逼近下的发散

$$
\begin{gathered}A=\mathbb E_b[\rho x(x-\gamma'x')^\top],\quad b_c=\mathbb E_b[\rho C'x],\quad C_x=\mathbb E_b[xx^\top]\\\mathbb E[\Delta w\mid w]=\alpha(b_c-Aw)\end{gathered}
$$

$b_c$ 是向量，不是行为策略 b；$C_x$ 是特征二阶矩，不是累计信号。表格可能精确满足 Bellman 方程，但固定低维表示通常不能，于是期望 TD 更新寻找的是特定采样权重下的投影固定点。矩阵 A 的稳定性决定这条动力学会不会远离解。

构造一个无需复杂神经网络的反例：两状态 s0、s1，特征分别为 1、2；动作 a 直接把下一状态设为 $s_a$。行为以 .9/.1 选择动作 0/1，因此 $d_b$=(.9,.1)；目标永远选动作 1，$\gamma$=.9，所有 cumulant 为 0。真实答案处处为 0，w=0 能精确表示。

$$
\begin{gathered}A=.9\cdot1(1-.9\cdot2)+.1\cdot2(2-.9\cdot2)=-.68\\\mathbb E[\Delta w]=+.68\alpha w\end{gathered}
$$

非零 w 的绝对值在期望更新下越来越大，尽管真实答案可表示、比率算对了、世界完全平稳。配套 `counterexample` 实验执行这一期望递推，分离更新方向与采样噪声的作用。

<a id="gvf-gtd"></a>

## 8 · 投影 Bellman 误差与 GTD2

$$
\begin{gathered}\Pi=\Phi(\Phi^\top D_b\Phi)^{-1}\Phi^\top D_b,\quad J(w)=\tfrac12\|\Pi(T\Phi w-\Phi w)\|_{D_b}^2\\J(w)=\tfrac12(b_c-Aw)^\top C_x^{-1}(b_c-Aw)\end{gathered}
$$

$\Phi$ 的每一行是一个状态特征；$D_b$ 用行为访问权重给误差加权；$\Pi$ 投影回特征能表示的函数空间。代入线性 Bellman 残差得到第二式，前提是 $C_x$ 可逆。这里最小化投影 Bellman 误差，不是直接最小化到真实 v 的平方误差。

$$
-\nabla_wJ=A^\top C_x^{-1}(b_c-Aw),\qquad h^*(w)=C_x^{-1}(b_c-Aw)
$$

把昂贵的矩阵求逆改为辅助预测：对固定 w，让 h 求解 $C_xh=b_c-Aw$。h 的作用是估计纠正方向，不是另一套预测目标网络。

$$
\begin{gathered}h_{t+1}=h_t+\beta\,[\rho_t\delta_t-x_t^\top h_t]x_t\\w_{t+1}=w_t+\alpha\rho_t(x_t-\gamma_{t+1}x_{t+1})(x_t^\top h_t)\end{gathered}
$$

第一行的期望是 $\beta(b_c-Aw-C_xh)$，第二行的期望是 $\alpha A^\top h$。辅助量跟踪得足够好时主更新逼近下降方向。两行都用旧 w、旧 h；源代码先缓存，不先改 h 再计算主方向。

在前面的两状态例子，$C_x$=1.3，理想下降方向为 −(.68²/1.3)w，与普通 TD 的 +.68w 方向相反。步长缩放不改变方向；GTD2 改变的是期望更新本身。随机收敛理论还需固定策略/特征、覆盖、矩阵条件和合适步长；不能把这个期望例子推广成任意神经网络的收敛保证。

<a id="gvf-tdc"></a>

## 9 · TDC 与 GTD($\lambda$)

$$
A^\top h=C_xh-\mathbb E_b[\rho\gamma'x'x^\top]h\ \approx\ b_c-Aw-\mathbb E_b[\rho\gamma'x'x^\top]h
$$

由于 $\mathbb E[\rho\mid s]=1$，Aᵀ 的第一项可写为 $C_x$。只有当 h 接近自己的固定点时，$C_xh$ 才可用 $b_c-Aw$ 替代。这样获得 TDC 的期望方向，不意味着它与 GTD2 每一步相同。

$$
\begin{gathered}w_{t+1}=w_t+\alpha\rho_t[\delta_t x_t-\gamma_{t+1}x_{t+1}(x_t^\top h_t)]\\h_{t+1}=h_t+\beta[\rho_t\delta_t-x_t^\top h_t]x_t\end{gathered}
$$

这是 TDC(0)：主项保留 TD 更新，再加入梯度校正；h 与 GTD2(0) 的更新相同。第一步 h=0 时 GTD2 主参数不动，TDC 一般会动——一个很简单的实现区分测试。

$$
\begin{gathered}e_t=\rho_t(\gamma_t\lambda e_{t-1}+x_t)\\w_{t+1}=w_t+\alpha[\delta_t e_t-\gamma_{t+1}(1-\lambda)x_{t+1}(e_t^\top h_t)]\\h_{t+1}=h_t+\beta[\delta_t e_t-x_t(x_t^\top h_t)]\end{gathered}
$$

这里称 GTD($\lambda$) 的是 RLPark 使用的 TDC-style 形式。$\lambda$=0 时 $e=\rho x$，正好退化为前面的 TDC，而不是 GTD2。辅助更新的投影项仍是 x(xᵀh)，不能把其中每个 x 都替换成 e。

- 保存旧 w、h 和上一时刻的 $\gamma$、迹。
- 用当前 c 和 $\gamma_{t+1}$ 计算 $\delta$。
- 用当前 $\rho$、进入当前的 $\gamma$ 更新 e。
- 从旧 h 计算 eᵀh 与 xᵀh，再生成两个增量。
- 应用增量；最后保存 $\gamma_{t+1}$ 供下一个转移使用。

本章 $\lambda$ 为常数。若使用状态相关 $\lambda$，下一状态 continuation 中的 $\lambda$ 下标也要按该算法原定义重新核对，不同下标约定对应不同的前向回报与资格迹。GTD 系列牺牲了额外向量与步长调节成本，以处理特定离策略函数逼近问题；它并不总是在有限样本上最快。

<a id="gvf-etd"></a>

## 10 · Emphatic TD 与状态加权

另一条路线不显式估计 MSPBE 梯度，而是调整更新的有效状态权重。interest $i_{t}$≥0 表示在哪些状态更关心预测；它不是奖励，也不改变 GVF 的 cumulant。Follow-on trace F 追踪目标行为从这些关心的状态出发会到达哪里；emphasis M 再结合 bootstrap 程度分配更新。

$$
\begin{gathered}F_t=i_t+\gamma_t\rho_{t-1}F_{t-1}\\M_t=\lambda i_t+(1-\lambda)F_t\\e_t=\rho_t(\gamma_t\lambda e_{t-1}+M_t x_t),\qquad w_{t+1}=w_t+\alpha\delta_t e_t\end{gathered}
$$

F 用上一动作的 $\rho_{t-1}$；e 用当前动作的 $\rho_{t}$。第一次没有前驱，置 $\gamma_t$=0、F=0；i=1 是常用起点。$\lambda$=1 时 M=i，i=1 下退化为相应 IS-TD(1) 迹；$\lambda$=0 时 M=F。

为什么强调有效？在固定策略下，$\lambda$=0 的期望 follow-on 权重满足 $f=D_bi+M^\top f$，其中此处的 M 是第 3 节折扣转移矩阵，不是标量 emphasis $M_{t}$。它把起点关注度沿目标动力学传播，使线性系统获得与普通 $d_b$ 投影不同的权重。更换权重也会更换逼近误差在状态之间的取舍。

代价是方差：连续的大重要性比可以让 F 和 e 很大。理论中的稳定期望方向、随机算法的收敛条件和短期数值表现是不同命题。随意裁剪 F/$\rho$ 会改变算法；可以作为实验变体，但不能继续直接引用未裁剪版本的结论。

<a id="lesson-example"></a>

## 11 · 共享经验的三个预测问题

实验环境是三个区域的环：动作 0 留在原地，动作 1 前进到下一区域；到区域 2 视为到达充电点。环境本身持续运行。行为以 .5 概率选各动作。energy 问目标 $\pi$(前进)=.8 下的折扣累计步成本；arrival 问相同行为下的折扣到达；slow-arrival 问 $\pi$(前进)=.2 下的折扣到达。到达时 $\gamma_{t+1}$=0，其余 .9。

| 真实转移 | energy 的 (c,$\gamma_{t+1}$,$\rho$) | arrival 的 (c,$\gamma_{t+1}$,$\rho$) | slow-arrival 的 (c,$\gamma_{t+1}$,$\rho$) |
| --- | --- | --- | --- |
| s0 —前进→ s1 | (1,.9,1.6) | (0,.9,1.6) | (0,.9,.4) |
| s1 —前进→ s2 | (1,0,1.6) | (1,0,1.6) | (1,0,.4) |
| s2 —前进→ s0 | (1,.9,1.6) | (0,.9,1.6) | (0,.9,.4) |

先按 TD(0)、$\alpha$=.1、各 w=0 手算 energy。第一步 $\delta$=1，w(s0)=.16。第二步 $\gamma_{t+1}$=0，因此不管 s2 预测多少，$\delta$=1，w(s1)=.16。第三步 $\delta$=1+.9×.16−0=1.144，w(s2)=.18304。第二步结束的是一个预测累计段，不是清空整个经验流；第三步仍可以从充电点出发问下次到达。

若换 GTD2，初始 h=0 导致第一步 w 不动，但 h(s0)=$\beta$×1.6；下一次相关特征出现时，辅助信息才影响 w。若换 GTD($\lambda$)，第二步的误差还能沿 e 回传给 s0。它们回答的是同一 question，瞬时学习路径不同。

$$
v_{\rm energy}(s_1)=1+.2\cdot.9v_{\rm energy}(s_1),\qquad v_{\rm energy}(s_0)=1+.9[.2v(s_0)+.8v(s_1)]
$$

到 s2 的余项为 0。因此 v(s1)=1/.82；再解出 v(s0)。程序直接构造 I−M 并用独立消元法求所有状态、所有 question 的参考答案，而不是拿某个学习器当真值。

<a id="lesson-code"></a>

## 12 · 算法实现

**算法：固定特征下的 Horde 式并行预测**

1. 为每个问题 $j$ 指定累积信号 $c^j$、延续因子 $\gamma^j$、目标策略 $\pi^j$ 和学习器。
1. 初始化各学习器的权重、辅助向量及资格迹。
1. 每个真实环境步：
  1. 按行为策略 $b(\cdot\mid S_t)$ 采样一次动作，保存 $b(A_t\mid S_t)$。
  1. 执行动作，观察 $S_{t+1}$ 和各传感器信号。
  1. 使用同一份更新前特征，对每个问题 $j$：
    1. 计算 $C^j_{t+1}$、$\gamma^j_{t+1}$、$\rho^j_t=\pi^j(A_t\mid S_t)/b(A_t\mid S_t)$。
    1. 由旧权重计算 $\delta^j_t=C^j_{t+1}+\gamma^j_{t+1}v^j(S_{t+1})-v^j(S_t)$。
    1. 按指定的 TD、GTD 或 ETD 规则，更新该问题独立的参数和迹。
    1. 保存该问题的延续系数及所需历史量。
  1. 继续真实交互；预测问题的终止不触发环境重置。

下面三段来自同一个独立脚本：question 接口、学习器、共享数据的完整 Horde-style 循环。Python 列表只是为了看清每个量，复杂度为每个 head 每步 O(d)；多 head 版本为 O(md)。这些向量不随生命期增长。

① Question：把真实 transition 转成三个预测问题各自的信号、延续与策略概率

```python
def question(name, state, action, next_state):
    """Return cumulant, gamma_next, target probability of the OBSERVED action.

    'arrival' predicts a discounted next-arrival signal, not an undiscounted
    eventual-arrival probability. 'slow-arrival' asks about a different policy.
    Arrival is a transition into cell 2, including staying there for one step.
    """
    forward = .2 if name == 'slow-arrival' else .8
    pi_observed = forward if action == 1 else 1-forward
    arrived = next_state == 2
    cumulant = 1. if name == 'energy' else float(arrived)
    gamma_next = 0. if arrived else .9
    return cumulant, gamma_next, pi_observed
```

② 六种固定特征学习器：delta 与 correction 使用旧参数；各 head 独立保存迹

```python
class LinearGVF:
    """Fixed-feature prediction. Right-hand sides always use OLD w and h.

    gamma_current belongs to the transition entering x; gamma_next to x -> xp.
    GTD(lambda) here is the TDC-style form used in RLPark, not GTD2(lambda).
    ETD uses a follow-on trace and interest, not a second learned value vector.
    """
    methods = ('td0','tdlambda','gtd2','tdc','gtdlambda','etd')

    def __init__(self, dimension, method='td0', alpha=.01, beta=.05, lam=.6):
        if method not in self.methods or dimension < 1:
            raise ValueError('Invalid method or dimension')
        if not all(math.isfinite(v) for v in (alpha,beta,lam)) or alpha<=0 or beta<=0 or not 0<=lam<=1:
            raise ValueError('Positive finite step sizes and lambda in [0,1] required')
        self.method,self.alpha,self.beta = method,alpha,beta
        self.lam = 0. if method in ('td0','gtd2','tdc') else lam
        self.w,self.h,self.e = [[0.]*dimension for _ in range(3)]
        self.gamma_current,self.rho_previous,self.followon = 0.,0.,0.

    def step(self, x, xp, cumulant, gamma_next, rho, interest=1.):
        if len(x)!=len(self.w) or len(xp)!=len(x):
            raise ValueError('Feature dimension mismatch')
        if not all(math.isfinite(v) for v in [*x,*xp,cumulant,gamma_next,rho,interest]):
            raise ValueError('Nonfinite input')
        if not 0<=gamma_next<=1 or rho<0 or interest<0:
            raise ValueError('Invalid discount, ratio or interest')
        old_w,old_h = self.w[:],self.h[:]
        delta = cumulant + gamma_next*dot(old_w,xp) - dot(old_w,x)
        if self.method == 'etd':
            self.followon = interest + self.gamma_current*self.rho_previous*self.followon
            emphasis = self.lam*interest + (1-self.lam)*self.followon
        else:
            emphasis = 1.
        self.e = [rho*(self.gamma_current*self.lam*ei + emphasis*xi)
                  for ei,xi in zip(self.e,x)]
        if self.method == 'gtd2':
            xh = dot(x,old_h)
            dw = [rho*(xi-gamma_next*xpi)*xh for xi,xpi in zip(x,xp)]
        elif self.method in ('tdc','gtdlambda'):
            eh = dot(self.e,old_h)
            dw = [delta*ei-gamma_next*(1-self.lam)*xpi*eh for ei,xpi in zip(self.e,xp)]
        else:
            dw = [delta*ei for ei in self.e]
        self.w = [wi+self.alpha*di for wi,di in zip(old_w,dw)]
        if self.method in ('gtd2','tdc','gtdlambda'):
            xh = dot(x,old_h)
            self.h = [hi+self.beta*(delta*ei-xh*xi) for hi,ei,xi in zip(old_h,self.e,x)]
        self.gamma_current,self.rho_previous = gamma_next,rho
        return delta

    def external_reset(self):
        """Use only when the data protocol actually breaks the trajectory.

        A GVF gamma_next=0 already terminates its own trace on the next step;
        it does not require resetting the physical environment or weights.
        """
        self.e = [0.]*len(self.w)
        self.gamma_current,self.rho_previous,self.followon = 0.,0.,0.
```

③ 完整训练与评估：环境只前进一步，每个预测问题各更新一次

```python
def run(method='gtdlambda', steps=40000, seed=7, alpha=.01, beta=.05, lam=.6, use_is=True):
    """A complete multi-question loop, sharing data but NOT traces or weights."""
    rng=random.Random(seed)
    names=('energy','arrival','slow-arrival')
    heads={name:LinearGVF(3,method,alpha,beta,lam) for name in names}
    features=[[float(i==j) for j in range(3)] for i in range(3)]
    state=0
    for _ in range(steps):
        # The real behavior is sampled exactly ONCE for all questions.
        action=int(rng.random()<.5); behavior_probability=.5
        next_state=(state+action)%3
        for name,head in heads.items():
            c,g,pi_observed=question(name,state,action,next_state)
            rho=pi_observed/behavior_probability if use_is else 1.
            head.step(features[state],features[next_state],c,g,rho)
        state=next_state  # no env.reset() when a question ends
    result={}
    for name,head in heads.items():
        truth=exact_values(name)
        result[name]={'prediction':head.w,'reference':truth,
                      'rmse':math.sqrt(sum((w-v)**2 for w,v in zip(head.w,truth))/3)}
    return result
```

下载本页脚本后，在文件所在目录运行；只需 Python 3.10+ 标准库

```sh
python3 gvf_lab.py compare --steps 40000 --seed 7
python3 gvf_lab.py gtdlambda --lam 0.6
python3 gvf_lab.py td0 --without-is
python3 gvf_lab.py counterexample
python3 gvf_lab.py test
```

compare 输出每个学习器的三个预测向量、解析 reference 和均匀状态加权 RMSE。RMSE 是对这个已知小环境的价值误差，不是 TD loss，也不是机器人控制回报。有限样本和常数步长让不同 seed 的结果不同；本实验不用于排出六种算法的普遍优劣。

counterexample 用完整期望更新复现第 7 节反例：TD 权重远离 0，GTD2 逼近 0。compare 的 one-hot 表示则用于接口和解析答案检查；表格例子的稳定表现不能取消反例。--without-is 会让不同目标策略的到达问题错误地趋向同一个行为策略答案。

<a id="gvf-diagnostics"></a>

## 13 · 数值性质与实验设计

| 检查 | 应该看到什么 | 否则先查什么 |
| --- | --- | --- |
| $\gamma_{t+1}$=0 | 当前 cumulant 保留，next value 消失 | 把当前奖励也乘了 $\gamma$，或 terminal mask 用错 |
| $\lambda$=0 | TD($\lambda$)→TD(0)，本章 GTD($\lambda$)→TDC | 错误地把 GTD($\lambda$) 对齐到 GTD2 |
| $\rho$=0 | 普通 IS-TD 主更新为 0；GTD 辅助量仍可能有投影衰减 | 把两条辅助更新都整体乘 $\rho$ |
| 问题到达终止事件 | 本次误差仍给旧迹信用；下一步旧迹不再延续 | 在终点更新之前就提前清掉 e |
| 所有 head 交换顺序 | 固定共享特征时训练结果不变 | 不小心共享了可变 w、e、F 或重复前进环境 |
| C 同时乘 10 | 真值相应乘 10，但固定步长数值表现未必不变 | 把尺度敏感性误认为问题语义变化 |
| 新 target policy 无行为覆盖 | 无法可靠回答该问题 | 给分母加数值常数不能弥补缺失的行为覆盖 |

实验可逐步放宽假设：先固定表示与问题，再分别改变策略覆盖、噪声、预测时域或特征混叠，最后考虑表示与控制的联合学习。预测精度衡量回答问题的能力；后续控制任务的样本需求衡量这些预测的用途。两类指标应分别报告。

<a id="lesson-branches"></a>

## 14 · 预测方法的关系

| 路线 | 实际改变的对象 | 关键区别 |
| --- | --- | --- |
| MC / n-step / TD($\lambda$) / true-online | 回报估计、信用时域与在线等价性 | 不改变给定 c、$\gamma$、$\pi$ 所定义的真值；有限表示下 fixed point/逼近可能随算法而变 |
| Ordinary IS-TD / GTD2 / TDC / GTD($\lambda$) / ETD | 离策略估计、梯度方向或有效状态权重 | 不是同一套迹加不同算法名 |
| LSTD / LSPE 等最小二乘预测 | 累计线性系统并求解 | 通常需 O(d²) 存储/计算及求解开销；遗忘因子和非平稳跟踪另需设计 |
| Horde / nexting | 问题集合和共享经验的调度 | Horde 是组织方式，nexting 强调多个近未来尺度；都不是新的单个 loss |
| TD networks / GVFN | 预测之间的依赖及递归表示 | 如果 cumulant 或输入依赖其他可学习预测，目标会移动，还需要跨预测/时间的信用 |
| Successor features | 把 cumulant 扩成特征向量，并按奖励权重复用 | 每个坐标可视为预测；适用的奖励族和动力学条件要成立 |
| 深度 GVF / auxiliary tasks | 非线性共享表示与学习信号 | stop-gradient、target network、replay 可成为工程选择，但不能直接继承固定线性理论 |

要使用预测进行控制，可以把多头输出作为决策输入，或用它们构造子目标、评估风险、近似模型。这个接口必须显式写出：下游究竟用了哪些预测，删去它们有什么变化？“预测到风险”也不等于保证安全，预测错误本身可能在新情境中最大。

这些分支分别放宽固定问题、固定表示或固定数据分布的假设。下一节讨论预测问题本身的选择，以及预测学习与行为学习的相互影响。

<a id="gvf-discovery"></a>

## 15 · 从学习预测到选择预测

前面假定预测问题和行为策略由外部给定。持续智能体还需要决定学哪些问题，以及采取什么行为才能改善这些预测。这两种选择处在不同位置：问题生成改变预测的语义，行为学习改变可获得的数据。

Veeriah 等的 Discovery of Useful Questions as Auxiliary Tasks（NeurIPS 2019）用后续主任务的学习效果评价预测问题。问题参数控制累积信号等内容；辅助预测更新共享表示；外层梯度通过这些更新反传，选择对主任务有用的问题。其关键区别是评价“学习该预测带来的效果”，而非仅选择容易预测的信号。

$$
\theta^+(\eta)=\theta-\alpha\nabla_\theta L_{\mathrm{aux}}(\theta;\eta),\qquad
\nabla_\eta L_{\mathrm{task}}(\theta^+)
=\left(\frac{\partial\theta^+}{\partial\eta}\right)^\top
\nabla_{\theta^+}L_{\mathrm{task}}
$$

这是问题发现的单步链式法则示意。问题参数影响辅助更新，辅助更新影响后续任务损失。多步展开还需累积中间更新的参数依赖；具体预测结构与梯度截断属于方法设定。

McLeod 等的 Continual Auxiliary Task Learning（NeurIPS 2021）保留一组辅助预测，进一步学习行为策略以收集有助于这些预测的数据。预测器不断改进会改变学习进展信号；行为变化又改变预测器的训练分布。论文使用 successor features 将动力学相关预测与变化的奖励权重分开，研究这种耦合系统的跟踪能力。

Modayil 与 Abbas 的 Nibbler（2023）研究另一种结构选择：从无结构的观测特征中选择与奖励相关的累积信号，为不同预测器选择局部输入，再把预测器生成的非线性特征用于主任务价值学习。其多组合环境用于检验观测维数增长时的计算与样本需求；结论依赖该问题族和特征选择机制。

Voelcker 等的 When does Self-Prediction Help?（RLC 2024）解释了为什么“预测准确”不足以决定表示是否适合控制。观测重构、潜在状态自预测与 TD 共同学习时，对特征的要求不同；奖励无关的干扰变量尤其会改变这种关系。其线性分析与 MinAtar 实验把“独立训练表示”和“辅助主任务训练”区分开来。

$$
z_t=\phi_\theta(o_t),\qquad
L_{\mathrm{latent}}=\|f_\psi(z_t,a_t)-\operatorname{sg}[\phi_\theta(o_{t+1})]\|^2,\qquad
L_{\mathrm{joint}}=L_{\mathrm{TD}}+\xi L_{\mathrm{latent}}
$$

上式给出常见潜在自预测的结构：预测目标是下一观测的表示，并在目标端停止梯度。这个一步辅助目标可以帮助解释表示学习，但它并不自动定义任意时域、任意目标策略下的 GVF。

因此，研究预测知识至少包含三个可分别检验的问题：预测对象能否表达所需信息，学习器能否在当前经验分布下准确跟踪，以及这些预测是否改善决策。它们对应问题设计、预测算法和下游使用三类实验。

<a id="lesson-check"></a>

## 16 · 习题与讨论

| 问题 | 推理与答案 |
| --- | --- |
| 预测“未来十步内碰撞概率”，直接用碰撞指示和 $\gamma$=.9 对吗？ | 不对。它通常是折扣碰撞累计量。要先定义首次事件、有限窗口与终止规则；固定折扣不能精确代表十步截断。 |
| 每个 GVF 的 $\gamma$=0 都要 env.reset() 吗？ | 不需要。它只终止自己的累计问题和后续迹延续；真实交互仍可继续，其他 GVF 也可以不停。 |
| 既然用了正确 $\rho$，为什么 TD 仍会发散？ | 动作条件期望被修正，函数逼近下的状态加权和 bootstrap 几何仍可能不稳定；第 7 节给出 A<0 的例子。 |
| 所有 GVF 都预测很准，能断言状态足够好了吗？ | 不能。若没有一个问题区分影响动作的历史，预测完全准确仍可能丢失控制所需信息；必须测表示充分性与下游用途。 |

练习：为一个具体的传感器事件定义累积信号、延续因子和目标策略。给出两条转移上的更新，并构造一个具有解析答案的小环境，比较价值误差与 TD error 的变化。

## 本章的实验设计

固定目标策略和累积信号。用解析预测或独立轨迹检验问题的真值。再改变行为策略检查覆盖与重要性采样。

设定：固定一个二动作预测问题，目标概率为 (0.8,0.2)，行为概率为 (0.4,0.6)。分别预测一次事件和多步累积量，再连接固定控制器。

- 动作 0 的重要性比为 2；加权期望与目标策略期望一致。
- 目标正概率而行为零概率时，程序拒绝声称可从现有数据识别。
- γ=0 预测下一 cumulant；目标策略或信号变化有独立版本。

对照：相同维度随机特征与固定 GVF；在线学习 GVF 与冻结已学 GVF；固定下游容量、数据权限与计算

记录：解析或独立目标策略下的预测误差；各问题尺度、覆盖、重要性比尾部；使用预测前后的原任务收益

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-modules)

## 学习与研究衔接

标量奖励价值是 GVF 的一种特例。定义多个问题不等于已经学出有用状态或控制策略。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-gvf) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=gvf) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=gvf)

<a id="chapter-code"></a>

## 下载与运行

六个线性学习器、三个共享经验的预测问题、解析 Bellman 参照与一个 off-policy 期望发散反例。

[下载 gvf_lab.py](https://yingwen.io/zh/continual-rl/download/gvf_lab.py)

```sh
python3 gvf_lab.py compare
python3 gvf_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Veeriah et al. · Discovery of Useful Questions as Auxiliary Tasks（NeurIPS 2019）](https://proceedings.neurips.cc/paper/2019/file/10ff0b5e85e5b85cc3095d431d8c08b4-Paper.pdf)：用多步元梯度学习 GVF 问题；区分问题参数、辅助预测更新和主任务评价。

- [McLeod et al. · Continual Auxiliary Task Learning（NeurIPS 2021）](https://papers.nips.cc/paper/2021/hash/68331ff0427b551b68e911eebe35233b-Abstract.html)：研究行为策略与辅助预测共同学习时的非平稳性，以及 successor features 的作用。

- [Modayil & Abbas · Towards model-free RL algorithms that scale well with unstructured data（2023）](https://arxiv.org/html/2311.02215v1)：Nibbler 的问题选择、局部输入选择与特征复用；算法 1–5 给出完整更新次序。

- [Voelcker et al. · When does Self-Prediction Help?（RLC 2024）](https://openreview.net/forum?id=izAJ8sHF5q)：比较观测重构与潜在自预测在独立表示学习和辅助 TD 学习中的不同作用。

- [Understanding Auxiliary Tasks · 作者项目说明](https://github.com/adaptive-agents-lab/understanding_auxiliary_tasks)：对应 RLC 2024 论文；仓库目前只有项目说明，未提供可运行实验源码。论文中的线性分析和神经网络实验不能视为已在此仓库公开实现。

- [Sutton et al. · Horde (2011)](https://sites.ualberta.ca/~amw8/horde.pdf)：原始 question/answer 分工、共享经验架构与停止信号约定。

- [Sutton et al. · Fast gradient-descent methods for TD learning (2009)](https://doi.org/10.1145/1553374.1553501)：GTD2 与 TDC 的目标和更新；注意两者不是逐次相同。

- [Sutton, Mahmood & White · An Emphatic Approach (2016)](https://www.jmlr.org/papers/v17/14-488.html)：interest、follow-on、emphasis 与线性稳定性；随机收敛需相应附加条件。

- [Sutton & Barto · 第 9、11、12 章](http://incompleteideas.net/book/the-book-2nd.html)：函数逼近、离策略预测、资格迹的完整教材背景。

- [RLPark · GTDLambda.java 固定版本](https://github.com/rlpark/rlpark/blob/2baf7389c75a31831e5b1a68539771c1947b7f09/rlpark.plugin.rltoys/jvsrc/rlpark/plugin/rltoys/algorithms/predictions/td/GTDLambda.java)：对照 update 中 $\gamma$t / $\gamma$t+1、$\rho$、correction 与旧辅助参数。Java 使用 v 为主权重、w 为辅助权重；本章使用 w、h。

- [RLPark · Horde.java](https://github.com/rlpark/rlpark/blob/2baf7389c75a31831e5b1a68539771c1947b7f09/rlpark.plugin.rltoys/jvsrc/rlpark/plugin/rltoys/horde/Horde.java)：看调度器怎样把同一 transition 分发给不同 demon；该实现面向原机器人系统。

- [GVFN · 作者代码](https://github.com/mkschleg/GVFN)：问题参与递归状态之后，需要额外处理表示与预测依赖。

- [GVFHordes.jl · 作者问题库](https://github.com/mkschleg/GVFHordes.jl)：按 cumulant / discount / policy 组织问题接口；可与本页 question 函数逐项对应。
