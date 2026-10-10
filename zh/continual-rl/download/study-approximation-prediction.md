# 函数逼近预测：从回归到 TD 固定点

函数逼近与经典进阶方法 · 第 1 章

共享少量参数以后，MC、TD 和最小二乘方法究竟在求解什么？

## 本章内容

- 从平方价值误差推导梯度 MC，说明采样分布的作用。
- 推导线性 TD 的平均更新、投影 Bellman 方程与 LSTD。
- 用同一个二状态例子计算不同解，并检查半梯度的含义。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [TD 预测与控制：SARSA、Expected SARSA、Q-learning 和 Double Q](https://yingwen.io/zh/continual-rl/foundations/tabular/temporal-difference/)：先固定预测对象，理解MC回报与TD目标，再研究共享参数。


### 固定策略的预测问题

策略不在本章学习过程中改变。它与环境共同给出转移矩阵 $P$ 和期望一步奖励向量 $r$。目标是估计执行此策略后的折扣回报。

$$
v_\pi=r_\pi+\gamma P_\pi v_\pi
$$

### 线性函数逼近

每个状态映射为一个 $d$ 维特征向量 $x$。$d$ 个共享权重产生全部状态的预测；线性指预测对权重线性，不要求特征对状态线性。

$$
\hat v(s,w)=x(s)^\top w,\qquad \Phi_{s,:}=x(s)^\top
$$

<a id="problem-definition"></a>

## 本章的问题定义

固定策略价值不能逐状态存储，只能用共享参数函数表示。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 给定目标策略 $\pi$，$v_\pi(s)=\mathbb E_\pi[\sum_{k\ge0}\gamma^kR_{t+k+1}\mid S_t=s]$。
- 误差权重 $d(s)\ge0$ 且 $\sum_sd(s)=1$；权重指定在哪些状态上评价估计。
- 函数族 $\{v_w:w\in\mathbb R^n\}$；$n$ 为参数维度，线性情形为 $v_w(s)=x(s)^\top w$。
- 状态采样权重与评价权重须区分：普通固定策略流在平稳极限按 $d_\pi$ 更新；任意指定的 $d$ 需要匹配的采样或显式重加权。

### 需要求解的对象

选择参数以逼近价值，并说明所用算法究竟逼近哪个投影或误差目标。

### 信息与数据权限

先使用固定目标策略的样本流，或按指定 $d$ 抽取状态再采样该策略的一步后果。使用行为策略 $b$ 的流时，动作概率校正本身不会把状态分布 $d_b$ 变成 $d$。

$$
\min_w\|v_w-v_\pi\|_d^2,\qquad v_w=\Pi_dT_\pi v_w
$$

左式是按评价权重 $d$ 的价值回归；右式只有在期望更新也按 $d$ 加权时才是相应线性 TD 的投影固定点。$\Pi_d$ 为固定线性函数空间的加权正交投影。普通 on-policy 流对应 $d=d_\pi$；一般两类解不相等。

### 成立条件与解的含义

- 投影解释限定为固定线性特征，且加权特征 Gram 矩阵可逆。普通 on-policy 推导假定策略链有适用的平稳访问分布 $d_\pi$。
- 另外指定 $d$ 时，需保证状态采样或重加权实现该期望；采样支持必须覆盖 $d$ 的支持。任意加权固定点的代数定义不自动继承 on-policy TD 的稳定性。

判断准则：对同一小问题分别求回归解和 TD 固定点，确认算法到达自己声称的对象。

### 适用边界

- 认为共享参数后仍可独立改善每个状态。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 限制表示或采用近似 · [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)：限制价值表示会产生逼近误差，并改变不同估计器的解。

- 改变信息或数据协议 · [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)：逐步更新限制计算形式；LSTD 的矩阵成本可能不满足同一预算。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

更新一个状态会改变其他状态；自举目标也随参数变化。

### 本章的核心思路

先区分回归问题和固定点问题，再为各自目标选择随机梯度、半梯度或线性方程求解器。

1. [从明确的回归误差得到梯度 MC](#lesson-derive)：真实回报作为监督信号时，可以直接推导所优化的均方目标。

2. [识别 TD 是半梯度与固定点更新](#prediction-td)：停止对尾值求导改变了更新方向，不能当作完整梯度的简写。

3. [比较逐步求解与矩阵求解](#prediction-algorithm)：TD 和 LSTD 可以联系到同一线性方程，但资源和数值条件不同。

结论与条件：线性 on-policy 的稳定性不覆盖任意深网或离策略数据。

### 相关方法改变了什么

- MC / TD：同一特征下通常得到不同投影结果。

- TD / LSTD：逐步随机逼近与批量矩阵求解在存储、条件数和跟踪能力上不同。


<a id="lesson-setting"></a>

## 1 · 表格无法逐项保存时的预测问题

表格预测为每个状态分配独立参数。连续观测或大量组合状态既可能超出存储预算，也可能使完全相同的状态很少再次出现。智能体因而需要把在一处学到的东西用于其他处境。函数逼近用共享参数建立这种联系：一次更新同时改变多个状态的预测。它带来泛化，也让不同状态争用同一组可调整的数值。

为什么先研究固定线性特征，再进入深度网络？固定特征已经足以产生近似误差、状态间干扰与 TD 的投影固定点，却允许我们把这些量独立算清。深度网络进一步从经验学习特征，扩大可表达的函数范围，同时使泛化关系本身随更新改变。持续强化学习还要求这套学习过程在有限资源下长时间保持可用；线性分析提供可检查的起点，无法替代后续关于表示变化与持续适应的论证。

先固定策略与环境，明确要预测什么。从状态 $s$ 出发，沿策略 $\pi$ 行动的折扣回报均值是真实价值；$P_\pi(s,s')$ 是该策略下的一步转移概率，$r_\pi(s)$ 是一步奖励的条件均值：

$$
v_\pi(s)=\mathbb E_\pi\!\left[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}\mid S_t=s\right],\qquad v_\pi=r_\pi+\gamma P_\pi v_\pi.
$$

本章的持续任务分析使用有限状态、$0\le\gamma<1$ 和有界奖励。后面的走廊另采用一定终止的两步回合、$\gamma=1$，以有限完整回报定义价值；它不依赖折扣持续任务的收缩条件。

共享参数通常无法同时消除所有状态的误差，因而还要指定各处有多重要。用 $d(s)\ge0$ 表示评价权重，$\sum_s d(s)=1$，令 $D=\operatorname{diag}(d)$。持续 on-policy 流的长期访问分布记为 $d_\pi$；在适用的平稳链与大数条件下，不经状态重加权的更新使用 $d=d_\pi$。另外指定评价分布后，还需获得匹配的样本或权重。

$$
J_{\mathrm{VE}}(w)=\frac12\sum_s d(s)\big[v_\pi(s)-\hat v(s,w)\big]^2.
$$

平方价值误差把真实价值当作目标。若一个状态的 $d(s)$ 为零，这个目标不会直接要求该状态准确；共享参数仍可能间接改变它。

回合数据的访问权重还取决于起点如何抽取。若每个回合都在每次非终止访问上执行一次同权更新，则长期样本份额为 $d_{\rm visit}(s)=\mathbb E_\pi[\sum_{t=0}^{T-1}\mathbf1\{S_t=s\}]/\mathbb E_\pi[T]$，这里假定回合长度有有限正均值。只更新首次访问、按回合等权，或再乘时间折扣，都会改变实际加权。因此回合访问权重与持续任务的平稳分布应分别计算。

<a id="lesson-derive"></a>

## 2 · 梯度 MC：从误差函数到样本更新

$$
\nabla_wJ_{\mathrm{VE}}=-\mathbb E_d[(v_\pi(S)-\hat v(S,w))\nabla_w\hat v(S,w)].
$$

先把 $w$ 视为固定，按 $d$ 取状态、再按 $\pi$ 取完整回报。对平方误差求导得到“误差乘预测梯度”；由 $\mathbb E[G\mid S=s]=v_\pi(s)$，可用回报样本替代未知价值。

$$
w_{t+1}=w_t+\alpha_t[G_t-\hat v(S_t,w_t)]\nabla_w\hat v(S_t,w_t).
$$

完整回报可用时，按“回报减当前预测”的误差回归。在线性情形，梯度是 $x_t$；固定策略与环境下，回报作为标签不依赖预测权重，所以不通过 $G_t$ 求导。

这里的总体期望关系需要与在线数据顺序分开。独立地按评价分布取状态和回报，会在每个固定参数处给出无偏梯度样本；实际轨迹中的状态彼此相关，当前权重又含有过去经验。回合内顺序使用同一条轨迹时，后一个更新还依赖前一个更新。不能仅凭回报的条件均值正确，就把每次在线更新都当作独立无偏梯度；其长期分析还需要相应的采样、噪声与步长条件。

回合式任务可以等到终止再计算回报。长回合使反馈延迟，随机未来使目标方差增大。持续交互任务也有定义良好的折扣价值，但无法在有限时间观察无限回报；截断后仍需说明尾项如何处理。因此 MC 的无偏目标条件和现实可用估计需要区分。

$$
w_{\mathrm{MC}}=(\Phi^\top D\Phi)^{-1}\Phi^\top Dv_\pi.
$$

固定线性特征、$C=\Phi^\top D\Phi$ 可逆时，令梯度为零得到加权最小二乘解。它是 $v_\pi$ 在特征张成空间上的正交投影。

$$
\mathbb E_d[(G-\hat v(S,w))^2]=\mathbb E_d[\operatorname{Var}(G\mid S)]+\sum_s d(s)[v_\pi(s)-\hat v(s,w)]^2.
$$

在固定参数与条件回报均值正确的前提下，先按状态展开平方，再取条件期望。MC 样本损失比平方价值误差多一个回报方差项；固定策略和环境时，这一项不依赖预测权重，因此不改变最优回归解。非零样本损失仍可能对应最佳可达预测。

先不要把拟合误差、采样噪声与算法误差混在一起。真实价值由环境、策略和回报定义决定。特征和评价权重决定可表示空间中的最佳近似。有限经验与计算预算决定算法距离这个近似还有多远。改变访问分布会改变近似的取舍，但不会改变同一策略在同一真实状态下的条件价值。

最小的状态聚合例子：两个真实状态的价值为0和10，却共享同一个常数预测 w。若它们的访问权重为.9和.1，最小二乘解为 w=1；若权重交换，则解为9。前一个解对少见状态错得很大，但这是当前表示与目标加权下的最优解。只降低学习率不会解除这个冲突；要么增加区分两个状态的特征，要么明确改变希望优先准确的分布。

先在没有自举与随机性的任务中单独看参数共享。每个回合从 A 或 B 开始，执行唯一动作后分别得 +1 或 −1 并终止，$\gamma=0$；因此完整回报就是真实值。取固定特征 $x(A)=(1,0)$、$x(B)=(0.8,0.6)$，初始 $w=0$，$\alpha=0.5$。依次观察一次 A 和一次 B；即使 B 尚未被观察，A 的更新也会改变它的预测。

![固定线性特征在观察 A、再观察 B 后的真实参数与两个预测；目标位置和同经验的独立表格基线显示有益泛化与冲突。](https://yingwen.io/crl-figures/learning-classic-feature-interference.svg)

原创实际回归更新：两个单步终止任务，固定目标 (+1,−1)、$\gamma=0$、$\alpha=0.5$、$w_0=0$；主图展示前 2 次访问，JSON 另保存 24 次交替访问。基线用独立表格参数、同经验与步长；无随机、种子不适用。

$$
\Delta\hat v(B)=x(B)^\top\Delta w=\alpha[G-\hat v(A)]x(B)^\top x(A).
$$

对 A 执行线性梯度 MC 更新，再左乘 B 的特征，就得到未访问状态的预测变化。变化由当次误差和特征内积共同决定，不是 B 获得了一份新奖励。

第一次在 A 得 1，误差为 1，$w$ 变为 $(0.5,0)$，所以 A 预测 0.5，B 预测 0.4。第二次在 B 得 −1，误差为 −1.4，参数增量是 $0.5\times(-1.4)\times(0.8,0.6)=(-0.56,-0.42)$。新的预测是 A=−0.06、B=−0.3：B 离自己的目标更近，A 却从误差 0.5 变成误差 1.06。图中黑竖线表示真实目标，圆点表示算法当前预测。

同样两份经验的表格基线只改变被访问的状态，得到 $(0.5,-0.5)$。这个对照隔离了共享造成的联动，尚不意味着表格总更好：若 B 的目标也为 +1，A 更新把 B 从 0 推向 0.4 就是有益泛化。两目标本来可以由 $w_*=(1,-3)$ 同时表示，所以这里的暂时干扰也不是不可消除的表示误差。后面的交替链将把表示能力不足与 TD 自举放到一起。

逐步数据与其他经典图一起保存在[图解数据 JSON](/crl-figures/learning-classic-data.json)，由网站源码仓库的 `scripts/generate-crl-learning-classic.mjs` 重算。这个固定标签例子给下一节提供参照：加入 TD 的自举后，更新还会改变以后用作标签的预测，必须另分析固定点，不能把本图的回归目标直接搬过去。

上面的聚合例子没有足够的参数区分真值，重叠特征例子则能够表示真值，却在学习途中产生干扰。这两种困难要分别检查。下一节把两个状态接成一条走廊，保留同样的完整回报与特征，把每次参数写入如何影响另一预测算清，再让目标本身使用预测。

<a id="prediction-sharing"></a>

## 共享几何 · 一次更新究竟帮助了谁？

把刚才的两个预测接到一条可以沿时间行走的走廊上：A 到 B 得 +2，B 到真正终点 $\bot$ 得 −1，唯一动作固定，$\gamma=1$。这改变了原先“各自一步终止”的任务与即时奖励；两步回合一定终止，所以完整回报仍为 $G(A)=1,G(B)=-1$。保留同一组特征、零初值与 $\alpha=0.5$，等两份奖励到齐后按 A、B 顺序做 MC，便复现刚才的参数更新。以后讨论资格迹时，我们会在这条走廊上逐步更新，不再等待完整回报。

![两步走廊的奖励为 +2 和 −1；单位特征夹角决定共享内积 0.8，预测平面中的两次 MC 更新同时移动 A、B。](https://yingwen.io/crl-figures/concept-shared-gradient-geometry.svg)

左图画特征方向，虚线投影给出内积 0.8；右图的两个坐标就是 A、B 的预测，0、1、2 是顺序 MC 更新后的点，方框是真值。回报完整后才回归；$\gamma=1,\alpha=0.5$，初始权重为零。原创精确计算；[数值与逐步状态](/crl-figures/shared-gradient-data.json)，[标准库脚本](/crl-code/tutorials/shared_gradient_walkthrough.py)。

设当前在状态 $s$ 用误差 $\varepsilon_s=G_s-\hat v(s,w)$ 更新，关心的另一状态是 $u$。线性预测对参数没有曲率，所以它的变化可以精确写为 $h_u=\alpha\varepsilon_s x(u)^\top x(s)$。把所有特征内积排成矩阵 $K=\Phi\Phi^\top$，本例得到 $K=\left[\begin{smallmatrix}1&0.8\\0.8&1\end{smallmatrix}\right]$：一列就是在相应状态更新时，所有预测共同移动的方向。表格的矩阵为单位阵，只有当前预测移动。

$$
\ell_u=\tfrac12\varepsilon_u^2,\qquad
\ell_u^+-\ell_u=\tfrac12(\varepsilon_u-h_u)^2-\tfrac12\varepsilon_u^2
=-\varepsilon_u h_u+\tfrac12h_u^2.
$$

这里 $\varepsilon_u=v_\pi(u)-\hat v(u,w)$ 用固定真实价值定义。一次连带更新使 u 更准确，当且仅当 $2\varepsilon_u h_u>h_u^2$。内积只决定预测移动方向；还需要 u 的误差方向与移动幅度，才能判断是否有益。

第一次在 A 回归时，B 的误差是 −1，而它的预测增加 $h_B=0.4$，于是 B 的半平方误差增加 $-(-1)(0.4)+0.4^2/2=0.48$。A 的半平方误差减少 0.375。按两状态各一半加权，总目标反而从 0.5 变为 0.5525。第二次回归后总目标为 0.4034，低于初值，但 A 比第一次更新后更不准确。一次样本更新、某个旧预测和整体评价可以给出不同判断。

在当前状态本身，$h_s=\alpha\varepsilon_s\|x(s)\|^2$；对固定、无噪声目标，若 $0<\alpha\|x(s)\|^2<2$ 且误差非零，该状态的平方误差会下降。这是单次回归的步长条件，尚未保证其他状态或 TD 自举系统稳定。将所有特征扩大 10 倍会使这次预测移动扩大 100 倍；特征与控制章会继续检查缩放怎样与步长、动作选择相互作用。

这一几何也说明 MC 的“期望梯度”为什么要规定采样分布。若回归状态确实按评价权重抽取且回报满足条件无偏性，样本更新才对应整体目标的期望梯度；固定走廊的某一个时刻却指定了 A 或 B，不是重新按该分布抽样。上面的单步上升与梯度 MC 的期望分析并不冲突。接下来加入自举：同一更新还会移动将来用作目标的预测，这时必须求 TD 自己的平均更新方程。

<a id="prediction-td"></a>

## 3 · 半梯度 TD：目标变化但只对当前预测求导

先令 $(S_t,R_{t+1},S_{t+1})$ 的采样条件为：当前状态按 $d$ 加权，下一步后果按固定目标策略的核生成。普通 on-policy 流在平稳极限满足这一条件且 $d=d_\pi$；若要使用任意指定的 $d$，可按该分布抽取起点再采样一步，或对覆盖它的状态采样分布 $q$ 使用权重 $d(s)/q(s)$。后者还需适当矩条件。行为流上的动作概率校正只改变条件动作分布，仍保留行为状态分布 $d_b$，不会自动实现任意 $d$。

$$
\begin{aligned}\delta_t&=R_{t+1}+\gamma x_{t+1}^\top w_t-x_t^\top w_t,\\ w_{t+1}&=w_t+\alpha_t\delta_t x_t.\end{aligned}
$$

一步 TD 用下一状态的当前预测补上未观察到的尾部。目标中也含 $w_t$，更新却只使用当前预测的梯度 $x_t$，因此称为半梯度。

在一次更新内部，把已经计算出的目标保存为固定标签，对当前预测求导就是一次正常的回归。半梯度这个名称指出的是整个自举程序的关系：下一轮标签又由改变后的预测产生，而更新没有沿这条目标依赖求导。因此“本次固定标签的梯度正确”和“整个过程最小化一个固定监督损失”是两项不同判断。

若把样本平方 TD 误差也对目标求导，更新方向会变成 $\delta_t(x_t-\gamma x_{t+1})$。那是另一种方法，不是“更完整的 TD”。它对应的目标、所需采样和所得近似解都可能不同。自举不自动等于优化样本 TD 平方误差。

$$
\begin{aligned}T_\pi z&=r_\pi+\gamma P_\pi z,\\ \mathbb E_d[\delta x]&=\Phi^\top D[T_\pi(\Phi w)-\Phi w]=b-Aw,\\ A&=\Phi^\top D(I-\gamma P_\pi)\Phi,\qquad b=\Phi^\top Dr_\pi.\end{aligned}
$$

Bellman 算子 $T_\pi$ 把任意试探值向量 $z$ 变为期望一步奖励加尾值。这里固定 $w$，再对按 $d$ 取状态、按目标策略取后果的总体求期望；它定义平均更新场，不等于把相关在线流按随机 $w_t$ 条件化后的每步平均。平均更新为零要求 $Aw=b$，通常不是价值误差的正规方程。

$$
\begin{aligned}\Phi w&=\Pi_D T_\pi(\Phi w),\\ \Pi_D&=\Phi(\Phi^\top D\Phi)^{-1}\Phi^\top D.\end{aligned}
$$

$C=\Phi^\top D\Phi$ 可逆时，把 $Aw=b$ 改写成 $\Phi^\top D[T_\pi\Phi w-\Phi w]=0$，再左乘 $\Phi C^{-1}$，得到投影 Bellman 固定点。这里的 $D$ 必须与期望更新权重一致。任意 $D$ 下可写出该方程，不代表普通 on-policy TD 会到达它，也不自动具有 on-policy 的收敛保证。

| 要回答的问题 | 相应数学量 | 不能据此推出什么 |
| --- | --- | --- |
| 是否准确预测真实回报 | 价值误差：真实价值与近似价值之差 | 单次回报噪声可以很大 |
| 是否满足环境的期望一步关系 | Bellman 残差：Tπv̂−v̂ | 投影后残差为0不代表原残差为0 |
| 是否已无平均参数更新 | 期望半梯度：E[δx] | 不同状态的误差可能沿共享特征抵消 |
| 单次预测目标有多大波动 | 样本 TD 平方误差：$\mathbb E[\delta^2]$ | 降低它不必等于降低真实价值误差 |

为什么投影会出现？一次 TD 更新只能沿特征方向改变预测。如果 Bellman 目标的一部分差异根本不在这些方向张成的空间内，学习器无法逐状态消除它。固定点要求的是剩余差异与所有可更新方向正交。这是逼近限制，不是 Bellman 方程本身被环境改变了。

因此，低 TD loss 不能单独证明学到了正确知识。即使预期更新为0，也应在可解析小环境中直接计算真实价值；在复杂环境中则可用独立回报、额外传感器事件或目标特定的校准检验。GVF 同样遵守这一点：增加许多预测头并不自动保证每个预测都准确。

<a id="prediction-algorithm"></a>

## 4 · 逐步 TD 与 LSTD 的计算顺序

**算法：线性半梯度 TD(0)**

1. 初始化权重 $w$，指定特征、策略、步长与终止语义
1. 观察 $S_t$，按固定策略产生动作和转移
1. 保存旧预测 $v=x_t^\top w$ 与 $v'=x_{t+1}^\top w$
1. 若真正终止，令 $v'=0$；计算 $\delta=R_{t+1}+\gamma v'-v$
1. 执行 $w\leftarrow w+\alpha\delta x_t$，然后推进状态

$$
\begin{aligned}\hat A&=\sum_t x_t(x_t-\gamma_{t+1}x_{t+1})^\top,\\ \hat b&=\sum_t x_tR_{t+1},\\ \hat A\hat w&=\hat b.\end{aligned}
$$

这是未额外加权的 on-policy LSTD；在适用的平稳极限中对应 $D=\operatorname{diag}(d_\pi)$。若改用另外指定的 $d$，两项样本和也须采用相匹配的状态权重。$\gamma_{t+1}$ 在真正终止时取零。相同数据和特征不保证有限样本 TD 迭代与 LSTD 已给出相同参数。

TD 每步需要 $O(d)$ 计算与内存。直接 LSTD 需要 $O(d^2)$ 存储，批量求解通常需要 $O(d^3)$ 计算。样本不足或特征相关可使矩阵奇异；加正则项会改变估计问题，不能把正则后的解称为原方程的精确解。

LSTD 的名字容易引起另一个误解：它并非通常意义上对每条样本的 TD 残差做最小二乘。令 $z=x-\gamma x'$。最小化样本残差平方会得到 $\sum zz^\top w=\sum zR$；LSTD 则求 $\sum xz^\top w=\sum xR$。两边使用的左乘向量不同。前者把 bootstrap 的随机变化也当作可降低的损失，后者让 TD 的平均特征方向更新为0。

$$
\begin{aligned}\hat A_t&=\kappa\hat A_{t-1}+x_t(x_t-\gamma_{t+1}x_{t+1})^\top,\\\hat b_t&=\kappa\hat b_{t-1}+x_tR_{t+1},\qquad 0<\kappa\le1.\end{aligned}
$$

这是为近期数据增加权重的一种统计设计，不是标准 LSTD 的额外收敛结论。κ=1 累计全部历史；κ<1 逐渐忘记旧样本，其标量几何权重总量约为1/(1−κ)。

这个设计揭示流式处理不等于持续适应：即使只存 A 和 b、不存原始经验，κ=1 的统计量仍让越来越长的旧历史支配结果。遗忘可以追踪变化，却降低有效样本量并可能导致病态矩阵。若特征本身改变，旧的 A、b 还对应旧坐标；仅加遗忘因子不足以获得与新表示下一致的最小二乘问题。

<a id="experiment-gradient_mc"></a>

### 实验：实验 · 两个共享参数，怎样从回报或 Bellman 方程得到？

同样的线性表示，逐样本梯度与累计正规方程究竟用了哪些不同的信息？

**环境与可用信息。** 五状态等概率随机游走，从中间开始；左终点奖励 0、右终点奖励 1，γ=1。每状态特征为 [1,s/6]，仅有两个共享权重。真值恰好可由权重 [0,1] 表示，因此没有不可约表示误差。

**设置。** 五种子各 1200 个真实转移，权重从零开始，输入轨迹匹配。Gradient MC 等完整回合结束后，从后向前用每次访问的完整回报做 SGD，α=0.03。LSTD 每步累计二乘二 A 矩阵和二维 b，再求解加 0.00001 对角正则的线性方程；没有 SGD 步长。

**检验的机制。** Gradient MC 对固定回报标签的平方误差求梯度。LSTD 累计当前特征与特征差的外积，求经验 TD 正规方程，不是把 MC 的同一个损失换成更好的优化器。前者保留本回合，后者保留全部历史的矩统计。

**测量。** 图中用当前参数计算五状态真实价值 RMSE。相同环境步数不等于相同存储或计算：LSTD 每步解一个小方程组；MC 只在终止时进行一串更新。

```bash
python3 implementations/extended_classic/gradient_mc.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/gradient_mc/curves.svg)

横轴：environment_steps。纵轴：五状态线性价值 RMSE。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 第 15 步，MC 的平均 RMSE 为 0.420，LSTD 为 0.504；第 300 步变为 0.210 对 0.0311；末尾为 0.1314 对 0.0278。LSTD 在后段更准确，但早期仍受有限数据和矩阵条件影响，MC 也持续受固定步长的轨迹噪声影响。

**结论边界。** 真值可表示，不能用本图展示 MC 投影与 TD 固定点的结构性差异。二乘二求解很便宜，不代表高维方法成本。固定正则会改变有限前缀解；没有测量最小奇异值，不能仅凭曲线确认某个尖峰的成因。

**继续实验。** 先记录矩阵条件数并核对终点特征为零。再去掉能表达斜率的特征，枚举访问分布下的 MC 投影与 TD 固定点，分开研究“表示不够”与“估计还不准”。

[源码](https://yingwen.io/crl-code/implementations/extended_classic/gradient_mc.py) · [逐种子记录](https://yingwen.io/crl-code/results/gradient_mc/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/gradient_mc/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/gradient_mc/curves.json)

<a id="lesson-example"></a>

## 5 · 手算：价值回归解与 TD 固定点

有两个状态 A、B，确定性地交替。A 到 B 的奖励为 0，B 到 A 的奖励为 1，折扣 $\gamma=1/2$。由 $v_A=v_B/2$ 和 $v_B=1+v_A/2$，得到 $v_A=2/3,v_B=4/3$。长期访问频率各为一半。

只允许一个权重，特征为 $x_A=1,x_B=3$。真实价值的比例为 1:2，表示强制比例为 1:3，所以不可能在两个状态同时精确。现在比较不同误差准则的解。

$$
C=\tfrac12(1^2+3^2)=5,\quad \Phi^\top Dv=\tfrac73,\quad w_{\mathrm{MC}}=\tfrac7{15}.
$$

价值回归先让真实价值误差对特征正交。

$$
\begin{aligned}A&=\tfrac12(1)(1-\tfrac12\cdot3)+\tfrac12(3)(3-\tfrac12\cdot1)=\tfrac72,\\ b&=\tfrac32,\qquad w_{\mathrm{TD}}=\tfrac37.\end{aligned}
$$

TD 则让期望 TD 更新为零。两个答案不同，不能据此断言某个实现错误。

| 方法 | 权重 | A 的预测 | B 的预测 |
| --- | --- | --- | --- |
| 真实价值 | 不可用单个权重表示 | 2/3 | 4/3 |
| 梯度 MC 的目标解 | 7/15 | 7/15 | 7/5 |
| TD / LSTD 固定点 | 3/7 | 3/7 | 9/7 |

把两个状态的预测当作平面坐标，参数共享就变成一条直线。MC 从真值投到这条线；TD 则要求先做 Bellman 备份、再投回直线后不再移动。图中的两个落点回答不同的问题。

![两状态值平面中的可表示直线、真实价值、MC 正交投影，以及 Bellman 备份再投影通向 TD 固定点的几何。](https://yingwen.io/crl-figures/concept-classic-projected-value.svg)

原创精确几何图，沿用上面的交替链、$\gamma=1/2$、特征 $(1,3)$ 与均匀状态权重。实线是可表示空间，绿色虚线是正交投影，紫色箭头是 $T$ 与 $\Pi_D$。从 $w_{\mathrm{MC}}=7/15$ 做一次投影备份得 $11/25$，继续迭代趋向 $w_{\mathrm{TD}}=3/7$；下方数轴放大显示两解的间距。所有坐标由模型计算，没有训练曲线。

再检验“写下评价权重”是否已经改变算法。若另指定 $d(A)=0.9,d(B)=0.1$，相应加权 TD 方程给出 $A=0.9(-1/2)+0.1(15/2)=0.3$、$b=0.1\times3=0.3$，所以该方程的解为 $w=1$。但原交替轨迹的访问频率仍各为一半；未经重加权的 TD 期望更新仍对应 $w=3/7$。要求解新方程，必须实际改变样本权重或起点采样，而非只在说明中改写 $d$。

<a id="rlss-prediction-occupancy"></a>

## 一条学习曲线之前：先求真值、访问分布与最佳近似

两状态例子已经区分了价值回归解与 TD 固定点。原书的 1000 状态随机游走把同一检查扩展到更大的回合任务：先由环境求真值，再由起点和转移求访问权重，最后求特征空间中的最佳近似。这样才能判断学习曲线离开的究竟是哪一个参照。

规定非终止状态为 1 到 1000，每回合从 500 开始。每步等概率选择左或右，再均匀选择 1 到 100 的距离。越过左边界获得 −1 并终止，越过右边界获得 +1 并终止，其余奖励为 0。令 γ=1。P 只保存非终止状态之间的转移，所以它的某些行和小于 1；r 保存一步期望奖励。

$$
v=r+Pv,\qquad v=(I-P)^{-1}r,\qquad \eta=e_{500}+P^\top\eta,\qquad \mu(s)=\frac{\eta(s)}{\sum_u\eta(u)}.
$$

假定该吸收链的期望回合长度有限。η(s) 是一个回合内访问 s 的期望次数；μ 是连接许多回合后按时间步计数的频率。它不是在吸收终点上的平稳分布，也不是把每个回合先归一化再平均。

μ 在 500 有尖峰，因为每个回合至少从这里访问一次。在 500 左右约 100 步的范围内，首次转移又额外增加占用。边缘状态较少被访问。真实价值在边缘不必是一条直线：越界后只给固定的 ±1，而不是按越界距离线性外推奖励。改变边界奖励就是改变问题，不能为获得好看的直线而偷偷修改它。

$$
\widehat v(s)=w_j\ (s\in G_j)\quad\Longrightarrow\quad w_j^*=\frac{\sum_{s\in G_j}\mu(s)v(s)}{\sum_{s\in G_j}\mu(s)}.
$$

状态聚合中，每组只有一个权重。对加权平方价值误差求导即可得到组内加权平均。边缘组内的 μ 不均匀，所以最优台阶也未必位于真值曲线的几何中点。

| 观察到的现象 | 先做的检查 | 仍不能推出的结论 |
| --- | --- | --- |
| MC 曲线尚未贴近组内最优值 | 比较解析 w*、当前权重及采样误差 | 不能先归咎于表示容量 |
| TD 的极限不同于 MC | 独立计算 Aw=b 和 MSVE 最优解 | 不表示某一实现必定写错 |
| n-step 的早期误差更低 | 固定评价分布、交互数及调参预算 | 不表示其渐近 MSVE 必定更小 |
| 终点附近误差较大 | 检查组内占用频率和边界奖励 | 不表示边界不重要或可以删掉失败样本 |

作图至少保留真值、预测、访问分布与误差随经验量变化四项。均匀 RMS 与 μ 加权 RMS 回答不同问题，不能只沿用同一个“误差”纵轴标签。TD 的折扣误差界还要求相应的平稳、固定特征和 γ<1 条件；不能直接把 1/(1−γ) 代入这个 γ=1 的回合例子。

这个回合例子与前面的持续交替链使用不同的边界和访问权重，检验时应各自保留。下一节的代码分别核对可解析固定点与走廊中的逐次写入；之后再讨论真实价值、访问分布或特征改变时，怎样判断原来的参照已经过时。

<a id="lesson-code"></a>

## 6 · 可运行实现与固定点检查

走廊练习从同一份固定轨迹输出特征内积、两次回归的分状态误差、普通迹、true-online 与逐前缀前向参照。下载[独立标准库脚本](/crl-code/tutorials/shared_gradient_walkthrough.py)后，可先遮住输出，手算第一步为什么使整体价值误差上升。

走廊中的共享更新与梯度记忆

```sh
python3 shared_gradient_walkthrough.py
python3 shared_gradient_walkthrough.py --json
```

MC、半梯度 TD、LSTD 与二状态解析例子

```python
def gradient_mc(weights, features, target, alpha):
    error = target - dot(weights, features)
    return [w + alpha * error * x for w, x in zip(weights, features)]


def semi_gradient_td(weights, features, reward, next_features, gamma, alpha):
    # Both predictions use the pre-update weights.
    delta = reward + gamma * dot(weights, next_features) - dot(weights, features)
    return [w + alpha * delta * x for w, x in zip(weights, features)], delta


def lstd(samples, dimension):
    """samples = (mass, x, r, gamma_next, x_next); mass may be 1 or probability."""
    a = [[0.0] * dimension for _ in range(dimension)]
    b = [0.0] * dimension
    for mass, x, reward, gamma, xp in samples:
        for i in range(dimension):
            b[i] += mass * reward * x[i]
            for j in range(dimension):
                a[i][j] += mass * x[i] * (x[j] - gamma * xp[j])
    return solve(a, b), a, b


def prediction_demo():
    # A -> B (r=0), B -> A (r=1), gamma=.5, stationary masses .5/.5.
    samples = [(0.5, [1.0], 0.0, 0.5, [3.0]),
               (0.5, [3.0], 1.0, 0.5, [1.0])]
    exact = solve([[1.0, -0.5], [-0.5, 1.0]], [0.0, 1.0])
    td, a, b = lstd(samples, 1)
    # Changing an evaluation weight alone does not change the original stream.
    # To solve the new projected equation, change the actual sample masses.
    weighted, weighted_a, weighted_b = lstd([
        (0.9, [1.0], 0.0, 0.5, [3.0]),
        (0.1, [3.0], 1.0, 0.5, [1.0])], 1)
    mc = (0.5 * exact[0] + 1.5 * exact[1]) / 5.0
    return {"true_values": exact, "A": a, "b": b, "TD_weight": td[0],
            "MC_weight": mc, "TD_values": [td[0], 3 * td[0]],
            "MC_values": [mc, 3 * mc],
            "reweighted_TD_weight": weighted[0],
            "reweighted_A": weighted_a, "reweighted_b": weighted_b}
```

代码中的样本权重既可以是观测次数，也可以是已知概率。本例用概率计算期望矩阵，因此没有蒙特卡洛误差。输出同时给出原访问权重下的 $3/7$ 与重加权后的 $1$，测试还检查原解不满足新方程。它适合检查推导，不能用于估计随机数据下的样本效率。测试另外对 MC 做有限差分，对 TD 检查更新前后参数的时序。

运行本章与全部公式测试

```sh
python3 approximation_textbook_lab.py prediction
python3 approximation_textbook_lab.py test
```

<a id="lesson-branches"></a>

## 7 · 收敛条件与持续学习的连接

一组常用的充分条件是：固定策略产生有限、不可约、非周期链，$0\le\gamma<1$，奖励具有有界二阶矩，固定特征在其平稳分布下线性独立，并采用合适的递减步长。此时线性在策略 TD 可以收敛到唯一投影固定点。控制中策略变化、离策略采样或神经特征共同训练，会改变这些条件。

平稳访问分布为什么出现在条件中？令 $D=\operatorname{diag}(d_\pi)$，$z$ 是任意状态值向量。下一状态值的条件均值先平方，不超过平方后求条件均值；再用 $d_\pi^\top P_\pi=d_\pi^\top$，便得到下面的范数关系。它把“在策略”接到了更新矩阵的稳定性。

$$
\begin{aligned}\|P_\pi z\|_D^2&\le\sum_s d_\pi(s)\sum_{s'}P_\pi(s,s')z(s')^2=\|z\|_D^2,\\ u^\top Au&=z^\top D(z-\gamma P_\pi z)\ge(1-\gamma)\|z\|_D^2,\qquad z=\Phi u.\end{aligned}
$$

第二行再用 Cauchy–Schwarz 不等式。固定特征在 D 下独立且 γ<1 时，非零 u 给出正值；A 不必对称，其对称部分仍为正定，因而冻结分布的平均线性更新在足够小步长下稳定。随机逐步收敛还需要上述采样、噪声与步长条件。

若下一步按目标策略转移，当前状态却按另一行为分布加权，第一行的平稳等式通常失效。于是覆盖和正确动作比率也未必保住第二行的正值。后面的离策略二状态反例正是改变了这个组合；把步长调小不能修复一个符号已经反转的平均反馈。

$$
\alpha_t>0,\qquad\sum_t\alpha_t=\infty,\qquad\sum_t\alpha_t^2<\infty.
$$

这是经典随机逼近步长条件：总更新量不耗尽，累计噪声的平方权重却可控。它们须和采样、矩与表示条件一起使用。常数步长不满足第二个级数条件，不能据此宣称几乎处处收敛。

持续环境中，奖励、转移或表示变化使固定点也变化。递减到接近零的步长不利于追踪，常数步长则留下稳态噪声。研究问题从“是否达到一个固定点”变成“变化后多快重新准确、长期误差多大、旧预测是否仍可用”。

神经预测器又增加了一项变化：真实环境与策略可以完全不动，参数共享的方向和自举标签仍随学习改变。新特征可能减小最佳可达误差，也会改变旧预测。若环境后果真的改变，真实价值才随之改变；若只更换采样权重，改变的则是近似取舍。持续学习应分别检查这三条路径，记录分状态误差、访问分布和表示变化。

接下来的特征与控制章让预测参与动作选择，使更新通过行为改变经验；多步回报与资格迹章则保留固定预测问题，改变反馈等待和信用传播。GVF 把奖励扩展为指定的累积信号，agent state 决定特征包含什么历史。这些扩展分别改变目标规格、输入信息或更新程序，理解本章的投影与采样条件后，才能逐项判断哪一条结论仍适用。

<a id="lesson-check"></a>

## 8 · 练习与答案

- 若 $\gamma=0$，TD 变成什么？答案：目标只剩一步奖励，线性更新就是以该奖励为标签的随机梯度回归。
- 为什么这里不能说 TD 最小化价值误差？答案：它的平衡条件是 $Aw=b$，而回归是 $Cw=\Phi^\top Dv_\pi$，两者通常不同。
- 把二状态特征改为两个 one-hot 分量，会怎样？答案：表示空间包含全部价值函数，MC 投影和 TD 固定点都等于真实价值。
- 只记录平均 TD 误差接近零，足以说明预测准确吗？答案：不够。标量误差可相互抵消；即使向量期望更新为零，也可能有函数逼近误差。

## 从本章进入实践

[预测与控制](https://yingwen.io/zh/continual-rl/code/#practice-prediction)：学会预测更多事情，什么时候会改变行动？



<a id="chapter-code"></a>

## 下载与运行

标准库教学实现。命令运行本章的可解析小问题和全文件的公式测试；不依赖深度学习库，不等同于论文中的大规模实验。

[下载 approximation_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/approximation_textbook_lab.py)

```sh
python3 approximation_textbook_lab.py prediction
python3 approximation_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：本章主论证对应 §9.1–9.4：预测对象、访问分布、梯度 MC 与线性 TD 固定点；LSTD 对应 §9.8。§9.7 用于核对非线性函数逼近的扩展边界。

- [Bradtke & Barto · Linear Least-Squares Algorithms for Temporal Difference Learning](https://link.springer.com/article/10.1007/BF00114723)：LSTD 的原始论文。适合在理解期望 TD 正规方程之后阅读。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-approximation-prediction#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-approximation-prediction#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-approximation-prediction)
<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 预测的量是什么，误差又是什么？

先固定策略。价值是回报的条件期望。MC 使用完整回报；TD 用下一时刻的预测替代未观察的余项。两者不能仅按同一批样本上的 TD error 排序。

函数逼近与深度方法：共享参数限制了可表示的函数。采样权重决定在哪些状态上拟合。最小价值误差、最小 Bellman 残差和 TD 固定点一般不同；神经网络又使可表示的局部方向随参数改变。

持续学习中的研究问题：多个 GVF 共用表示时，哪些预测值得占用容量？应分别检查问题定义是否改变、数据是否覆盖，以及回答该问题的误差是否降低。

[MC 与 TD](https://yingwen.io/zh/continual-rl/foundations/tabular/temporal-difference/) → [投影与半梯度](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/) → [神经价值更新](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/) → [GVF 的问题与答案](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)


### 可进一步检验的问题

- [03 · 哪些预测值得学习，谁来使用这些预测？](https://yingwen.io/zh/continual-rl/research/#research-predictive-knowledge)：先理解固定预测问题下的误差、投影与半梯度答案，再研究有限预算应该用来维持哪些预测问题。
- [06 · 不存 replay、每步只处理新经验时，怎样避免更新失稳？](https://yingwen.io/zh/continual-rl/research/#research-streaming-stability)：共享参数的几何表明相同参数步长可能产生不同输出变化，为流式更新的尺度控制提供直接检查对象。
- [13 · 为什么训练越久，学习新东西反而越慢？](https://yingwen.io/zh/continual-rl/research/#research-plasticity)：固定特征与固定新目标提供简单的学习参照，帮助将表示不足、共享干扰与更新规则的困难分开。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)
- [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)
- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)

对应原始材料：第 9 章：On-policy Prediction with Approximation。本文为原创讲解，原书、论文与上游代码保留各自许可。
