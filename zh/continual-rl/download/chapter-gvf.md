# 通用价值函数与预测知识

通用价值函数把一个预测问题写成目标策略、累积信号与延续规则。以“沿墙行走至充电点的能耗”为例，本章从问题定义推导 Bellman 方程，再逐步构造 on-policy 与 off-policy 学习算法。

## 本章内容

- 从一个自然语言问题写出 cumulant、continuation、目标策略和状态条件，分清问题与学习器。
- 独立推导 Bellman 方程、线性 TD、资格迹、GTD2/TDC、GTD($\lambda$) 与 Emphatic TD 的更新。
- 逐行运行多问题共享经验的学习循环，检查解析解、off-policy 发散反例与实现时序。

<a id="problem-definition"></a>

## 本章的问题定义

在指定行为条件下预测某种信号的累计量，例如到充电点前的能耗；信号不必是任务奖励。

### 给定条件与符号

- 每个问题的目标策略、cumulant、延续规则和条件状态。
- 真实行为数据及行为概率；固定特征/网络类和更新预算。

### 需要求解的对象

各个给定预测题目的条件期望；稳定估计和预测发现是另外需要声明的子问题。

### 信息与数据权限

$b$ 生成动作，$\pi$ 定义假想未来行为；由已到达转移构造 $C_{t+1}$ 与 $\gamma_{t+1}$。行为支持目标动作时才可计算 $\rho_t=\pi(A_t\mid S_t)/b(A_t\mid S_t)$。

$$
v_{\pi,c,\gamma}(s)=\mathbb E_\pi\!\left[\sum_{k=0}^{\infty}\left(\prod_{j=1}^{k}\gamma_{t+j}\right)C_{t+k+1}\,\middle|\,S_t=s\right]
$$

$C$ 是由信号规则 $c$ 生成的累计信号，$\gamma$ 为转移延续因子，空乘积为1。事件上停止仍保留该步信号。线性GTD的投影Bellman目标是求解代理，未必等于最小真实预测误差。

### 成立条件与解的含义

- 分析期间环境、目标策略与表示固定，状态Markov且累计量存在；延续矩阵谱半径小于1提供唯一解条件。
- 离策略需要覆盖；GTD/ETD稳定性须满足相应线性、遍历和步长条件，不推广为任意深网定理。

判断准则：有限题目直接解线性Bellman系统核对预测与单位；在离策略反例上分开测价值误差、发散和重要性比方差。

### 适用边界

- 离策略预测稳定不等于得到全局最优控制。
- 事件终止预测不要求重置真实环境。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 推广：放宽条件 · [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)：本章将任务奖励推广为指定累计信号，并允许转移依赖的延续规则；目标策略仍须单独给定。

- 组合不同学习问题 · [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)：预测可作状态坐标，但覆盖少量问题不证明所有相关历史已被保留。

- 组合不同学习问题 · [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)：设计奖励与折扣终点信号可以预测模型输出；普通单个GVF不等于完整后果模型。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

自然语言题目容易混淆累计信号与终止计时；实际行为又可能不同于假想行为。

### 本章的核心思路

先明确问题三元组，再分开处理Bellman递推、行为纠偏与逼近稳定性。

1. [固定题目语义和计时](#gvf-semantics)：因为能耗、到达概率和折扣到达量不同，先由cumulant与延续写出累计量和Bellman方程。

2. [由真实行为估计目标行为](#gvf-offpolicy)：因为样本动作由行为策略产生，用记录概率的比率纠偏并检查支持，缺失覆盖不能靠加小常数修复。

3. [以辅助量估计投影目标方向](#gvf-gtd)：因为重要性比不保证共享线性参数稳定，GTD用辅助向量估计投影Bellman目标所需的条件量，并保留所写主/辅助更新。

4. [用强调权重处理另一种逼近](#gvf-etd)：因为状态分布也影响稳定性与逼近解，ETD递推follow-on与强调权重；它不只是替换GTD的迹，需分别检验加权固定点和方差。

结论与条件：可积且延续矩阵满足条件时题目有确定解；线性GTD/ETD的理论依赖其假设，神经递归GVF不因此自动稳定。

### 相关方法改变了什么

- 普通TD：低成本一步自举，离策略共享逼近下可能发散。

- GTD2/TDC：借助辅助量优化投影Bellman相关目标，需区分目标与更新式。

- Emphatic TD：调整历史和状态强调权重，目标权重及方差与GTD不同。


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

在前面的两状态例子，$C_x$=1.3，理想下降方向为 $-(.68^2/1.3)w$，与普通 TD 的 +.68w 方向相反。步长缩放不改变方向；GTD2 改变的是期望更新本身。随机收敛理论还需固定策略/特征、覆盖、矩阵条件和合适步长；不能把这个期望例子推广成任意神经网络的收敛保证。

<a id="gvf-tdc"></a>

## 9 · TDC 与 GTD($\lambda$)

$$
A^\top h=C_xh-\mathbb E_b[\rho\gamma'x'x^\top]h\ \approx\ b_c-Aw-\mathbb E_b[\rho\gamma'x'x^\top]h
$$

由于 $\mathbb E[\rho\mid s]=1$，$A^T$ 的第一项可写为 $C_x$。只有当 h 接近自己的固定点时，$C_xh$ 才可用 $b_c-Aw$ 替代。这样获得 TDC 的期望方向，不意味着它与 GTD2 每一步相同。

$$
\begin{gathered}w_{t+1}=w_t+\alpha\rho_t[\delta_t x_t-\gamma_{t+1}x_{t+1}(x_t^\top h_t)]\\h_{t+1}=h_t+\beta[\rho_t\delta_t-x_t^\top h_t]x_t\end{gathered}
$$

这是 TDC(0)：主项保留 TD 更新，再加入梯度校正；h 与 GTD2(0) 的更新相同。第一步 h=0 时 GTD2 主参数不动，TDC 一般会动——一个很简单的实现区分测试。

$$
\begin{gathered}e_t=\rho_t(\gamma_t\lambda e_{t-1}+x_t)\\w_{t+1}=w_t+\alpha[\delta_t e_t-\gamma_{t+1}(1-\lambda)x_{t+1}(e_t^\top h_t)]\\h_{t+1}=h_t+\beta[\delta_t e_t-x_t(x_t^\top h_t)]\end{gathered}
$$

这里称 GTD($\lambda$) 的是 RLPark 使用的 TDC-style 形式。$\lambda$=0 时 $e=\rho x$，正好退化为前面的 TDC，而不是 GTD2。辅助更新的投影项仍是 $x(x^Th)$，不能把其中每个 x 都替换成 e。

- 保存旧 w、h 和上一时刻的 $\gamma$、迹。
- 用当前 c 和 $\gamma_{t+1}$ 计算 $\delta$。
- 用当前 $\rho$、进入当前的 $\gamma$ 更新 e。
- 从旧 h 计算 $e^Th$ 与 $x^Th$，再生成两个增量。
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
| LSTD / LSPE 等最小二乘预测 | 累计线性系统并求解 | 通常需 $O(d^2)$ 存储/计算及求解开销；遗忘因子和非平稳跟踪另需设计 |
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

<a id="research-gvf-measures-and-readouts"></a>

## 研究专题 A · 从有限预测向量到可查询的未来占用

GVF 的基本单位是一个明确的问题；successor features 将有限个信号在同一策略下的未来累计组成向量。当未来奖励尚未知时，新的问题是：有限信号族遗漏了什么，能否学一个可被更多信号查询的未来占用？FB 和 SF² 分别给出低秩占用与生成式 successor measure 两条路线。它们扩展预测对象，而不取消目标策略、时域和覆盖要求。

$$
\mu_\gamma^\pi(B\mid s,a)=(1-\gamma)\sum_{k=0}^{\infty}\gamma^k\Pr_\pi(S_{t+k+1}\in B\mid S_t=s,A_t=a),\qquad Q_c^\pi(s,a)=\frac{1}{1-\gamma}\int c(x)\,\mu_\gamma^\pi(dx\mid s,a)
$$

此处明确采用从下一状态开始、归一化的占用约定，且 cumulant 为到达状态函数 c(x)、固定 0≤γ<1。一般转移 cumulant 或动作相关奖励须扩展所占用的对象；状态相关 continuation 也不能机械使用这条固定折扣归一化。

若每步信号是“到达充电区”，占用积分给折扣访问累计；它可以反复计数，不是首次到达概率。若希望在首次事件停止，必须把 stopping 规则写进问题，或扩展状态为尚未到达/已到达。给 γ=0.9 时，归一化占用对充电区的质量为 0.2，累计访问期望便为 2；把质量 0.2 当成累计值会少掉因子 10。

$$
\mu_\gamma^\pi(\cdot\mid s,a)=(1-\gamma)P(\cdot\mid s,a)+\gamma\,\mathbb E_{S'\sim P,\,A'\sim\pi}[\mu_\gamma^\pi(\cdot\mid S',A')]
$$

measure Bellman 方程是分布的混合：部分目标来自真实一步后果，部分来自下一状态策略条件的长期预测。生成式 bootstrap 仍会传播估计误差；无需显式长 rollout 不代表没有长期误差。

$$
\psi^\pi(s,a)=\frac{1}{1-\gamma}\int\phi(x)\,\mu_\gamma^\pi(dx\mid s,a),\qquad r_w(x)=\phi(x)^\top w\Rightarrow Q_w^\pi(s,a)=\psi^\pi(s,a)^\top w
$$

SF 是占用分布在有限特征上的投影。奖励张成空间、固定策略与不变动力学一起决定复用边界；这条线性价值恒等式不能直接推广到任意 learned embedding。

Does Zero-Shot RL Exist?（ICLR 2023）将 FB 与不同 SF 基础特征放在同一固定数据上比较，说明联合学习可读出的占用结构与任意自监督特征并不等价。其最优性目标和有限神经训练结果需分开看：良好的 buffer 覆盖是迁移结果的一部分，不是凭空从零样本查询产生的经验。

实验可从固定策略、固定动力学开始：学习占用或 SF 后冻结表示，公布一组未参与表示训练的 cumulants；分别测查询读出误差、行为覆盖和下游控制。再改变策略、动力学或 continuation，每次只放宽一个复用前提。研究空缺是有限预算下怎样选择需保留的查询、检测新查询超出表示范围，并用未来经验补齐。

<a id="research-gvf-flow-feature-boundaries"></a>

## 研究专题 B · SF² 的线性结构究竟在哪一层

SF²（ICLR 2026）将未来占用作为生成式预测问题，再让控制器使用压缩特征。这条链值得逐层核对：它不是将所有 GVF 统一为一个线性 TD 网络，也不是自动学习完整 agent state。原文的结构约束加在生成向量场上，而不是最终的 reward readout 或 critic 上。

$$
u_\theta(x,k,s,a)=\zeta_\theta(x,k)^\top\psi_\theta(s,a),\qquad \frac{dx_k}{dk}=u_\theta(x_k,k,s,a),\qquad Q_\omega(s,a)=g_\omega(\psi_\theta(s,a))
$$

x 是生成的未来状态位置，k∈[0,1] 是噪声到目标分布的生成时间，ψ 是当前状态动作的条件特征，ζ 是随 x 与 k 改变的矩阵投影。g 可以非线性；k 不等于环境原始步数，也不是 GVF 的 discount。

$$
\mathcal L_{\rm flow}=(1-\gamma)\mathcal L_{\rm one\ step}+\gamma\mathcal L_{\rm bootstrap}
$$

两项分别拟合真实下一状态的 flow target 与下一个状态动作条件的目标向量场，具体采样路径、目标网络及停止梯度按原文算法实现。这里表示混合结构，不声称两项可由普通标量 TD error 直接替代。

即使 u 对 ψ 线性，求解 ODE 时 x 随 ψ 改变，而 ζ 又依赖 x；因此最终样本与其统计量一般可非线性依赖 ψ。由“生成向量场线性”跳到“任意新奖励的 Q 都可线性读出”，少了一个实质性证明或实验。原文在小生成时间下给出的 SR-like 更新联系也被明确限定为近似解释。

| 待证明的命题 | 原文提供什么 | 可增加什么检验 |
| --- | --- | --- |
| 能预测多步未来占用 | flow 与 successor mixture 的训练结构 | 独立未来轨迹上的分布距离和多模态覆盖 |
| ψ 适合原任务控制 | 与 TD3/SAC 联合的控制实验 | 同 backbone、相同 update ratio 与模型调用预算的对照 |
| ψ 支持线性新查询 | 不由向量场分解自动推出 | 冻结 ψ，以线性 head 预测新 cumulants；对照非线性 head |
| ψ 能作为递归 agent state | 不是该工作直接研究的对象 | 同观测不同历史、递归更新和梯度信用实验 |
| 适合 strict streaming | 作者算法含 replay、批次与目标网络 | 单次经验协议需要另行设计并报告 |

**算法：不同证据层逐次扩展，保留作者协议与新 CRL 协议的区别**

1. 读作者实现时（不在教材中声称已复现）：
  1. 追踪 networks 中 ψ、ζ 与非线性 Q 的维度
  1. 追踪 losses 中 one-step、bootstrap、critic loss 的梯度路径
  1. 核对 target ψ/ζ 更新时序、γ 与 ODE 积分步数
  1. 固定 replay 与 update ratio，先复现原任务对照
  1. 冻结 ψ 后做新 cumulant 的线性/非线性读出
  1. 最后才改变动力学或改为流式经验协议

对持续预测的启发是将“预测内容丰富”与“下游可便宜查询”分开优化。生成器可以保留多模态未来，而有限特征有助于低成本控制；两者之间是否形成可持续维护、可迁移且资源有界的知识接口，仍是可检验的研究问题。

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

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

状态是支持后续计算的内部信息；GVF 指定一个预测问题；RTRL 和资格迹规定信用如何传播。三者可以组合，但不是相互替代的算法名称。先理解给定策略的预测，再讨论预测怎样改善控制。

- [Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks](https://yingwen.io/zh/continual-rl/research/#recent-columnar-constructive-networks)
- [Towards model-free RL algorithms that scale well with unstructured data](https://yingwen.io/zh/continual-rl/research/#recent-nibbler-predictive-features)
- [When does Self-Prediction help? Understanding Auxiliary Tasks in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-self-prediction-auxiliary-tasks)
- [Does Zero-Shot Reinforcement Learning Exist?](https://yingwen.io/zh/continual-rl/research/#recent-zero-shot-forward-backward)
- [Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations](https://yingwen.io/zh/continual-rl/research/#recent-successor-flow-features)

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)
- [Expected Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-expected-eligibility-traces)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

模型可提取性的理论说明某类能力需要什么知识，不指定唯一网络。Dreamer 研究潜在想象控制，STOMP 研究随机时长行为模型，DRAGO 研究旧模型知识保留。模型误差、查询策略和规划收益之间仍需实验连接。

- [The Value Equivalence Principle for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-value-equivalence-models)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文提供定义和条件，架构讲座提出模块组织，算法论文检验特定机制。完整系统还要明确智能体、外部设计者和世界各自承担的工作；组件成立不自动意味着组合后的长期收益成立。

- [The OaK Architecture: A Vision of SuperIntelligence from Experience](https://yingwen.io/zh/continual-rl/research/#recent-oak-architecture)
- [Towards model-free RL algorithms that scale well with unstructured data](https://yingwen.io/zh/continual-rl/research/#recent-nibbler-predictive-features)
- [The Value Equivalence Principle for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-value-equivalence-models)
- [Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations](https://yingwen.io/zh/continual-rl/research/#recent-successor-flow-features)

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

### Towards model-free RL algorithms that scale well with unstructured data

Joseph Modayil, Zaheer Abbas

arXiv 预印本 · 2023 · 支持方法与理论

#### 研究问题

大量原始观测中只有少数局部组合与奖励有关，智能体能否逐步构造有用的预测特征？

#### 关键机制

Nibbler 将预测问题的构造和预测结果的复用结合起来：选择局部输入、学习与奖励相关的通用价值预测，再将预测作为后续学习的特征。GVF 在这里不是一个新的优化器，而是描述“预测什么、在什么行为下预测”的问题接口。

#### 证据

作者在组合式合成环境中增加观测规模，报告了利用任务结构的样本效率。环境可以具有指数增长的状态组合，但学习器不必显式枚举全部状态。

#### 条件与限制

这不是对任意高维观测的线性样本复杂度保证。局部可分解结构、候选问题与特征构造规则仍是关键条件；从该实验族迁移到视觉控制需要额外验证。

#### 阅读与实验

把一条预测完整写成累积量、延续条件、目标策略和输入特征四项，再指出它如何进入主任务的价值函数。区分问题生成带来的收益与增加参数量带来的收益。

#### 原文与相关入口

- [作者预印本](https://arxiv.org/abs/2311.02215)：问题族、Nibbler 构造过程与扩展性实验。

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

### The OaK Architecture: A Vision of SuperIntelligence from Experience

Richard S. Sutton

RLC 2025 讲座 / Oak Lab · 2025 · 定义与架构观点

#### 研究问题

持续学习是否只是在一个现成 actor–critic 上加入抗遗忘机制，还是需要重新安排知识构造与使用？

#### 关键机制

OaK 提出从经验持续形成状态、预测知识、子任务、时间抽象与模型，并让这些知识服务规划的架构方向。这里的重点是模块之间怎样产生可复用知识，而不只是保留某个固定策略网络的参数。

#### 证据

官方页面提供 Richard Sutton 的架构讲座与相关研究入口。STOMP、预测学习和在线特征学习等论文可以检验其中具体组件，但不能自动验证整体架构。

#### 条件与限制

这是研究愿景与架构讲解，不是一套已公布完整训练配方、统一基准结果和可复现端到端代码的系统。资源分配、问题生成、知识替换与模块相互干扰仍需明确算法。

#### 阅读与实验

为每个模块写出输入、输出、更新频率和资源上限。再选择一个双模块接口做可证伪实验，例如技能模型改善是否真的减少规划误差。

#### 原文与相关入口

- [Oak Lab 官方讲座页面](https://oaklab.ai/posts/the-oak-architecture)：讲座入口与架构研究方向。
- [Oak Lab 研究主页](https://oaklab.ai/)：区分已发表研究、技术文章和仍在预告中的项目。

### Expected Eligibility Traces

Hado van Hasselt, Sephora Madjiheurem, Matteo Hessel, David Silver, André Barreto, Diana Borsa

AAAI 2021（2020预印本） · 2021 · 支持方法与理论

#### 研究问题

当前误差能否同时更新本次未走过、但也可能到达当前状态的过去路径？

#### 关键机制

学习给定当前状态的资格迹条件均值，再用当前TD误差更新该均值所指向的过去预测。递归混合在实际轨迹迹与预测的期望迹之间插值；预测对象是过去资格，而非未来奖励。

#### 证据

原文在Markov状态与相应条件下证明更新均值相同、逐分量方差不增，并在路径汇合问题检验预测效率。信用章精确枚举一个正例和一个状态混叠反例。

#### 条件与限制

不完整观察、参数漂移和近似迹预测器会破坏无偏条件。全参数期望迹预测还有输出维度和计算成本；小实验不复现作者的神经实验。

#### 阅读与实验

保持奖励边际分布一致，仅改变奖励是否依赖隐藏的过去路径。先测信用均值与方差，再研究agent state能否恢复条件独立。

#### 原文与相关入口

- [作者原文](https://arxiv.org/html/2007.01839)：Lemma 1、Proposition 1及ET(λ,η)递归混合。
- [AAAI发表版本](https://ojs.aaai.org/index.php/AAAI/article/view/17200)：正式会议年份为2021。

### Does Zero-Shot Reinforcement Learning Exist?

Ahmed Touati, Jérémy Rapin, Yann Ollivier

ICLR 2023 · 2023 · 支持方法与理论

#### 研究问题

没有事先指定奖励时，怎样学一套预测表示，日后接收新奖励就能选行为？

#### 关键机制

Forward–Backward 表示联合学习行为条件的未来占用与奖励读出，而非先固定任意编码器再学习 successor features。新奖励被映射到任务向量，策略根据这个向量直接行动；该论文系统比较 FB 与多种 SF 基础特征。

#### 证据

原文在固定离线 replay buffers 上比较零样本任务迁移，借此把表示学习与探索数据的质量分开。不同特征与数据覆盖产生显著差异，不能仅靠“所有奖励”的理论目标预测实际效果。

#### 条件与限制

假定共享动力学与可用经验覆盖。无下游梯度更新不等于无预训练成本；有限秩、近似训练和奖励估计都有误差。新动力学、历史混叠和严格一次使用经验均须另测。

#### 阅读与实验

同一 buffer 对比随机特征、谱特征与联合 FB，再独立换 buffer。奖励读出误差、占用误差与新任务回报分别报告，避免把数据覆盖优势记成表示优势。

#### 原文与相关入口

- [作者原文](https://arxiv.org/abs/2209.14935)：2022 首稿，ICLR 2023；比较奖励表示、SF 与 FB。
- [作者研究平台](https://github.com/facebookresearch/controllable_agent)：README 直接关联两篇 FB 论文；该仓库已经归档。

#### 作者代码

[论文作者团队仓库；归档工程，依赖和旧环境需单独核验。](https://github.com/facebookresearch/controllable_agent)

FB 与 SF 的训练、固定数据实验及奖励查询示例。

### Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations

Haosen Shi, Jianda Chen, Sinno Jialin Pan

ICLR 2026 · 2026 · 支持方法与理论

#### 研究问题

能否直接学习多步未来状态的分布，并把它压缩为适合控制学习的特征？

#### 关键机制

SF² 以 flow matching 估计 successor measure，将条件向量场分解为未来位置及生成时间的投影与当前状态动作特征的乘积。特征进入 TD3/SAC 的 critic；线性的是向量场对条件特征的分解，critic 本身可以非线性。

#### 证据

正式原文给出 mixture Bellman 结构、生成式 bootstrap 与控制实验，并提供作者 JAX/Brax 仓库。实验研究在线收集数据下的 off-policy 控制，并使用 replay、批次与目标网络。

#### 条件与限制

“online policy learning”不代表 strict streaming。生成时间不是环境时间；向量场线性不保证任意奖励价值线性。文中与 SR 的小生成时间联系是近似动机，未证明递归 agent state 或任意持续变化下的充分性。

#### 阅读与实验

对齐模型调用与梯度预算，拆分直接预测、bootstrap、critic 联合训练。冻结特征后比较线性与非线性读出，再测新奖励和动力学变化，才能检验预测知识的可复用程度。

#### 原文与相关入口

- [ICLR 2026 正式原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48acf4b231771e693f42305b4c9b4c9f-Abstract-Conference.html)：第 2–3 节和算法附录；区分 flow 时间、环境时间与近似 SR 联系。
- [原文链接的作者实现](https://github.com/Shiien/successor-flow-representation-implementation)：SAC/TD3、flow 特征、对照和 sweep 配置。

#### 作者代码

[正式论文摘要直接链接的作者代码。](https://github.com/Shiien/successor-flow-representation-implementation)

基于 JAX/Brax 的 SF² 控制实验；不包含自动 GVF 问题发现或完整持续架构。

### The Value Equivalence Principle for Model-Based Reinforcement Learning

Christopher Grimm, André Barreto, Satinder Singh, David Silver

NeurIPS 2020 · 2020 · 支持方法与理论

#### 研究问题

模型容量有限时，必须预测全部状态细节，还是只须保持规划会查询的量？

#### 关键机制

Value equivalence 以策略集合和函数集合定义模型规格：模型对这些函数进行这些策略的 Bellman backup，应与真实环境相同。扩大查询族会缩小可接受模型族；它把“决策相关”从口号变成有条件的等价关系。

#### 证据

论文给出等价模型类的性质及有限实验，并解释若干隐式模型方法。后续 Proper Value Equivalence（NeurIPS 2021）研究策略价值固定点等价及规划充分性。

#### 条件与限制

少数当前 critic 的 backup 相同，不说明所有新奖励、新策略或风险目标都相同。精确算子等价与神经损失在样本上较小不同；奖励或查询族变化后须重新验证。

#### 阅读与实验

保存独立的 planner 查询集，直接测 target 误差和动作排序。用未参与模型拟合的价值函数检验迁移，并与像素误差对照，找出模型实际保留的信息。

#### 原文与相关入口

- [NeurIPS 2020 原文](https://papers.nips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)：VE 依赖策略与函数集合。
- [Proper Value Equivalence · NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/400e5e6a7ce0c754f281525fae75a873-Abstract.html)：多步算子、固定点与规划充分性；不是任意潜在网络的保证。


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

- [作者原文](https://arxiv.org/abs/2209.14935)：2022 首稿，ICLR 2023；比较奖励表示、SF 与 FB。

- [作者研究平台](https://github.com/facebookresearch/controllable_agent)：README 直接关联两篇 FB 论文；该仓库已经归档。

- [ICLR 2026 正式原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48acf4b231771e693f42305b4c9b4c9f-Abstract-Conference.html)：第 2–3 节和算法附录；区分 flow 时间、环境时间与近似 SR 联系。

- [原文链接的作者实现](https://github.com/Shiien/successor-flow-representation-implementation)：SAC/TD3、flow 特征、对照和 sweep 配置。

- [NeurIPS 2020 原文](https://papers.nips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)：VE 依赖策略与函数集合。

- [Proper Value Equivalence · NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/400e5e6a7ce0c754f281525fae75a873-Abstract.html)：多步算子、固定点与规划充分性；不是任意潜在网络的保证。

- [Barreto et al. · Successor Features for Transfer in Reinforcement Learning](https://arxiv.org/abs/1606.05312)：有限奖励特征、固定策略 SF 与 GPI 的经典机制桥梁。

- [Touati & Ollivier · Learning One Representation to Optimize All Rewards](https://arxiv.org/abs/2103.07945)：FB 表示的理论出发点；探索/经验覆盖、近似误差及奖励查询约定。

- [Sutton, Bowling & Pilarski · The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173)：状态、预测、时间抽象与规划的研究纲领，不是完成全闭环的报告。
