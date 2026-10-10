# 多步回报、资格迹与 True-online TD

函数逼近与经典进阶方法 · 第 5 章

当前到来的奖励怎样更新过去的预测，同时保留正确的在线更新语义？

## 本章内容

- 把 n-step 与 lambda-return 的前向目标推导成后向信用传播。
- 解释普通累积迹与 true-online 的差别，推导 Dutch trace。
- 用重复状态和独立前向实现检查每个轨迹前缀。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [多步学习：n-step、Tree Backup 与 Q(σ)](https://yingwen.io/zh/continual-rl/foundations/tabular/multistep/)：知道多步回报由哪些奖励和尾值组成。
- [函数逼近预测：从回归到 TD 固定点](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/)：知道梯度对应哪个参数版本，哪些状态共享参数。


### 前向与后向视图

前向视图从某个状态向未来构造回报目标；后向视图在反馈到来时，根据过去特征留下的资格更新权重。两者描述的时间方向不同。

### 固定特征与线性预测

本章 true-online 的精确等价以线性预测、固定特征及给定步长约定为基础。神经网络梯度迹是另一种近似，不自动具有同一等价。

<a id="problem-definition"></a>

## 本章的问题定义

给定固定策略的预测任务，要求在线把后续误差分配给此前参数方向。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 给定目标策略 $\pi$，$v_\pi(s)=\mathbb E_\pi[\sum_{k\ge0}\gamma^kR_{t+k+1}\mid S_t=s]$。
- 误差权重 $d(s)\ge0$ 且 $\sum_sd(s)=1$；权重指定在哪些状态上评价估计。
- 线性特征 $x_t$、参数 $w_t$、迹系数 $\lambda\in[0,1]$、步长 $\alpha$。

### 需要求解的对象

以递推摘要实现规定的前向多步更新；先验证更新语义，再比较统计效率。

### 信息与数据权限

只能按时间到达顺序处理数据；旧特征的影响可由资格迹保存。

$$
w_t^{\rm backward}=w_t^{\rm online\ forward}\quad\text{for every observed prefix }t
$$

这是实现等价性要求，不是外部控制目标。在线前向视图规定每个前缀如何重构目标；传统冻结参数 λ-return 是不同参照。

### 成立条件与解的含义

- True-online TD 的精确逐前缀等价先在线性函数逼近及匹配的步长定义下成立。
- 终止、截断和参数版本必须在两个实现中一致。

判断准则：对同一短轨迹比较每个前缀的参数，而非仅比较最后的平均回报。

### 适用边界

- 把线性 true-online 等价直接推广到任意深度网络。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)：本章把信用分配限定为固定线性预测及指定在线前向视图；相关章还包含离策略、非线性与递归状态信用。

- 组合不同学习问题 · [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)：本章的资格迹可作为流式学习器的持久摘要，组合时须计入其参数维度、计算和内存；它不是样本回放。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

参数在数据到来时不断改变，冻结权重下的交换求和不再精确描述实际更新。

### 本章的核心思路

先明确在线前向参照，再加入 Dutch trace 和预测变化修正以匹配它。

1. [看清多步目标的混合](#lesson-derive)：λ 控制不同未来长度的权重，先在冻结参数下建立代数关系。

2. [把参照改为在线前缀](#traces-online)：指定每一步重新计算什么，才能判断后向实现是否正确。

3. [修正重复特征与预测变化](#traces-dutch)：两项修正共同实现等价，不能只换一种迹递推。

结论与条件：等价保证的是算法语义，不是任意任务上的收益优势。

### 相关方法改变了什么

- Accumulating / replacing traces：两者处理重复激活不同，不可无条件互换。

- 传统 TD(λ) / true-online TD：后者精确匹配指定在线前向视图，而非只在小步长下近似。


<a id="lesson-setting"></a>

## 1 · 一步 TD 的信用传播瓶颈

在一条长轨迹中，奖励可能很晚才出现。表格一步 TD 只更新当前状态的条目，过去状态要等到再次访问后才逐渐得到影响。多步回报直接包含更多实际反馈，资格迹则让刚到来的误差同时影响近期相关特征。它们解决时间信用分配，不负责决定保存哪些旧任务能力。

函数逼近还带来另一条传播路径。函数逼近预测章的走廊 A 到 B 到终点使用重叠特征，更新 B 本来就会连带改变 A；资格迹额外保留的是过去出现过的梯度方向。我们将沿用这条奖励为 +2、−1 的轨迹，分清“当前特征已经共享的影响”与“随时间保存下来的信用”，再检查第一步参数变化怎样影响第二步。

先区分目标与执行时序。设回合在 $T$ 真正终止，特征 $x_t=x(S_t)$ 固定，$x_T=0$；冻结一份参数 $\bar w$，记 $\bar v_t=\bar w^\top x_t$。为先推导回报关系，下面的所有尾值都用这份参数。报告窗口结束而环境未终止时仍保留尾值；迹的重置也应对应实际历史边界。

$$
G_t^{(n)}(\bar w)=\sum_{j=0}^{n-1}\gamma^jR_{t+j+1}+\gamma^n\bar v_{t+n},\qquad 1\le n\le T-t.
$$

n 步目标纳入前 n 个实际奖励并预测剩余尾部；$n=T-t$ 时尾特征为零，得到完整回报 $G_t$。有限回合允许 $\gamma=1$，仍需所用回报具有适当矩。等待 n 步决定何时目标可用，尚未规定何时、按什么参数执行回归。

<a id="lesson-derive"></a>

## 2 · Lambda-return 与 TD 误差的展开

$$
\begin{gathered}G_t^\lambda(\bar w)=(1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1}G_t^{(n)}(\bar w)+\lambda^{T-t-1}G_t,\\ 0\le\lambda\le1.\end{gathered}
$$

有限回合的最后一项保留全部剩余权重，所以各项系数和为 1。$\lambda=0$ 给一步目标，$\lambda=1$ 给完整回报；只剩最后一步时，两者本来就是同一目标，按端点的连续约定处理 $0^0$。

$$
\begin{aligned}G_t^\lambda(\bar w)-\bar v_t&=\sum_{k=t}^{T-1}(\gamma\lambda)^{k-t}\bar\delta_k,\\ \bar\delta_k&=R_{k+1}+\gamma\bar v_{k+1}-\bar v_k.\end{aligned}
$$

展开右侧并消去相邻价值项，就得到左侧。所有残差使用同一份冻结参数；这一步给出从前向混合回报到后向误差传播的代数关系。

$$
\begin{gathered}\sum_{t=0}^{T-1}\alpha[G_t^\lambda(\bar w)-\bar v_t]x_t=\alpha\sum_{k=0}^{T-1}\bar\delta_k e_k,\\ e_{-1}=0,\qquad e_k=\sum_{t=0}^k(\gamma\lambda)^{k-t}x_t=\gamma\lambda e_{k-1}+x_k.\end{gathered}
$$

交换两重求和次序，得到累积资格迹。左侧各增量都在冻结参数处算出，最后相加；它尚未执行每次回归之后立即改变参数的顺序算法。

普通在线 TD(lambda) 保留这条迹递推，但在每步计算当前参数下的 TD 误差并立即写入权重。后续预测因而可能改变。它实现了即时信用传播；与冻结参数下增量总和的精确等式，则是另一个命题。

$$
\delta_t=R_{t+1}+\gamma w_t^\top x_{t+1}-w_t^\top x_t,\qquad e_t=\gamma\lambda e_{t-1}+x_t,\qquad w_{t+1}=w_t+\alpha\delta_t e_t.
$$

普通累积迹的执行顺序：先用 $w_t$ 保存当前预测、尾预测与残差，用旧迹形成 $e_t$，再更新权重；真正终止时仍使用最后奖励，尾值为零。相同状态重复或特征重叠时，早期写入更容易影响后续目标。

![确定性五步链首回合的多步目标与普通累积资格迹更新，末步资格为 (γλ) 的不同次幂。](https://yingwen.io/crl-figures/learning-classic-credit.svg)

原创实际更新：$\gamma=0.9$、$\alpha=0.25$、$\lambda=0.8$，零初值，1 回合/5 转移。每状态只出现一次且此前误差全为 0，因此本例可独立与冻结前向式核对；不是一般在线等价证明。

在图中的五步链，只有最后一步的 $\delta_4=1$；此前所有奖励和预测都为 0。因此起点的有限 $\lambda$-return 只剩最长回报的尾权重：$G_0^\lambda=\lambda^4\gamma^4$。前向增量 $\alpha G_0^\lambda=0.25\times0.8^4\times0.9^4=0.06718464$，恰等于后向累积迹 $\alpha\delta_4e_4(S_0)$。每个较晚状态按少一次的幂得到相应值，读者可以逐格核对表中的 $\lambda$ 行。

这一次相等有特别原因：反馈出现前没有发生任何参数变化，而且状态特征互不重叠。若中途已有非零误差，或同一参数参与多个位置，在线预测就可能与冻结推导不同。下一节把前向参照改为每个已见前缀，后面的重复状态算例则展示为什么还需要 Dutch trace 和预测变化修正。

<a id="traces-online"></a>

## 3 · 在线前向视图究竟指什么

要讨论在线等价，先把前向算法也定义为只使用当前可见经验的过程。时刻 $h$ 已看到 $S_0,R_1,S_1,\ldots,R_h,S_h$；对每个 $0\le k<h$，只能组合已经可见的 n 步目标，最长项吸收剩余权重。每次抵达一个状态时，保存用更新前参数算出的尾预测，以后扩展前缀时沿用这个数：

$$
\begin{aligned}G_k^{(n)}&=\sum_{j=0}^{n-1}\gamma^jR_{k+j+1}+\gamma^n w_{k+n-1}^\top x_{k+n},\quad 1\le n\le h-k,\\ G_k^{\lambda\mid h}&=(1-\lambda)\sum_{n=1}^{h-k-1}\lambda^{n-1}G_k^{(n)}+\lambda^{h-k-1}G_k^{(h-k)}.\end{aligned}
$$

这里 $w_j$ 是此前在线算法在前缀 j 结束后的实际参数，$w_{k+n-1}^\top x_{k+n}$ 在第 k+n 步更新之前保存。它与上一节使用冻结 $\bar w$ 的 n 步目标不同；真正终点的特征仍为零。

现在定义重算。用 $w_k^{[h]}$ 表示在前缀 h 的这一次回归扫描中，处理第 k 个访问位置之前的参数。每个新前缀都从同一个回合初始参数 $w_0$ 开始，目标已由上面的实际在线预测确定；扫描内部的当前预测则随这次回归逐项变化：

$$
\begin{aligned}w_0^{[h]}&=w_0,\\ w_{k+1}^{[h]}&=w_k^{[h]}+\alpha[G_k^{\lambda\mid h}-(w_k^{[h]})^\top x_k]x_k,\quad 0\le k<h,\\ w_h&=w_h^{[h]}.\end{aligned}
$$

两种下标分别回答“正在重算哪一个前缀”和“已经回归到哪个访问位置”。第 h 次扫描的末态才是实际在线参数 $w_h$；扫描中的中间权重用于当前预测，不替换先前保存的 bootstrap。

这便是在线 lambda-return 的前向参照：每次多看到一步，就修订目标并重做整个已见前缀。它没有使用未来经验，但重算开销随已见轨迹增长。True-online 在线性固定特征下只维护固定大小的递归状态，逐前缀得到同一个参数序列；代码用慢速前向扫描核对这项算法等价。

<a id="traces-dutch"></a>

## 4 · Dutch trace 与预测变化修正

在线前向扫描为什么不能用普通相加的迹压缩？先沿原书 §12.6 和课堂的同一简化问题推导：依次观察 $x_0,\ldots,x_{T-1}$，最后得到这些预测共同要预测的标量结果 $G$。等 $G$ 可用后按原顺序做线性回归，一次写入为 $w^+=(I-\alpha xx^\top)w+\alpha xG$。后一个回归会继续变换前一个增量；这正是冻结增量求和丢掉的关系。

$$
\begin{aligned}a_{-1}&=w_0,\quad z_{-1}=0,\\a_t&=a_{t-1}-\alpha x_t(x_t^\top a_{t-1}),\\z_t&=z_{t-1}+\alpha[1-x_t^\top z_{t-1}]x_t,\\w_T&=a_{T-1}+z_{T-1}G.\end{aligned}
$$

对“当前权重=与 G 无关的部分+G 的系数”作归纳即可得到递推。这里只讨论共同的最终结果 G，不是任意奖励序列的通用 MC 公式。z 是已把步长吸收进去的 Dutch 型迹。

每次只需两个向量和若干内积，内存与每步计算都是 $O(d)$，不随等待 $G$ 的时间 $T$ 增长。不要显式构造 $d\times d$ 矩阵：先算 $x^\top a$ 或 $x^\top z$，再乘 $x$。最终获得 $G$ 时也只做 $O(d)$ 合并。重复特征 $x=1$、$\alpha=.5$、$w_0=0$ 时，$z$ 依次为.5、.75；$G=1$ 后最终参数为.75，与逐次回归相同。

这里直到最终结果 G 到来才合成新参数，但等待期间已经均匀地完成了必要计算。span-independent 指每步资源需求不随延迟跨度增长。这个回归推导给出了资格迹作为计算机制的一个精确等价，无须先引入 TD。

回到在线 TD。新前缀会修订前向目标，同时还要沿原顺序重新回归；把这两种变化一起递归压缩，在线性固定特征与常数步长下得到 Dutch trace：

$$
e_t=\gamma\lambda e_{t-1}+[1-\alpha\gamma\lambda e_{t-1}^\top x_t]x_t.
$$

等价写法 $e_t=\gamma\lambda(I-\alpha x_tx_t^\top)e_{t-1}+x_t$ 保留了刚才回归中的变换，再加入时间衰减。步长出现在迹里，是因为迹压缩的是实际回归更新的作用，而非仅记录访问次数。

$$
\begin{aligned}v_t&=w_t^\top x_t,\qquad \delta_t=R_{t+1}+\gamma w_t^\top x_{t+1}-v_t,\\ w_{t+1}&=w_t+\alpha\delta_t e_t+\alpha(v_t-v_{\rm old})(e_t-x_t).\end{aligned}
$$

对 $t\ge1$，$v_{\rm old}=w_{t-1}^\top x_t$ 是上一步在更新前保存的当前状态旧预测；回合起点设为零，此时 $e_0=x_0$，修正项仍为零。$v_t-v_{\rm old}$ 衡量刚才参数写入造成的预测变化。Dutch 迹与这项权重修正共同构成完整 true-online 算法。

**算法：常数步长 True-online TD(lambda)**

1. 回合开始：初始化 $e=0,v_{\rm old}=0$；保留待学习权重
1. 用旧权重计算 $v=w^\top x,v'=w^\top x'$，再计算 $\delta$
1. 用旧 trace 的内积计算 Dutch 修正，然后写入新 trace
1. 执行 $w\leftarrow w+\alpha\delta e+\alpha(v-v_{\rm old})(e-x)$
1. 保存更新前已算出的 $v_{\rm old}\leftarrow v'$；推进到 $x'$
1. 真正终止保留末步更新，再清除回合 trace

本页精确等价使用常数步长。逐时间改变步长时，需使用相匹配的缩放迹和前向定义。它证明的是给定经验序列上每个前缀的参数一致；随机收敛、预测误差或控制回报仍各自需要条件与证据。

另一种旧方法是替换迹（replacing trace）。它为表格或二值特征定义：当前激活分量设为 1，其余分量按 $\gamma\lambda$ 衰减。累积迹则在激活处再加 1；Dutch 迹依据实际步长与已有迹修正当前方向。因此重复访问时，三种迹保存的量不同。

$$
e_{i,t}^{\rm rep}=\begin{cases}1,&x_{i,t}=1,\\\gamma\lambda e_{i,t-1}^{\rm rep},&x_{i,t}=0.\end{cases}
$$

这个逐分量规则要求二值输入。走廊中的特征含 0.8、0.6，不能直接使用此二分定义；对任意连续特征截断或重设数值，要另说明算法。替换迹也不承担上面 true-online 与在线前向扫描的精确等价。

<a id="experiment-true_online_td"></a>

### 实验：实验 · True-online 的“精确”，不表示误差始终最小

Dutch trace 与预测变化修正为什么是等价性要求，却不是性能保证？

**环境与可用信息。** 五个非终止状态的等概率随机游走，左终点奖励 0、右终点奖励 1，γ=1。每回合从中间状态开始，价值从零开始。代码用五维 one-hot 线性表示，表格只是该线性情形的特例。

**设置。** 种子 0–4，各 1200 个真实转移，α=0.1。True-online TD 使用 λ=0.8，初始 Dutch 迹与旧预测均为零；对照为相同表格、相同轨迹的 TD(0)。终点奖励先更新所有有资格的参数，再清迹和旧预测。

**检验的机制。** Dutch 迹校正有限步长下重复访问的影响；另一个参数修正处理相邻时刻预测已经变化的事实。两项共同对应在线前向视图。它没有取消 λ 带来的偏差、方差和传播速度折中。

**测量。** 纵轴是五状态相对于解析真值的 RMSE。该误差曲线检验学习过程，不检验前后向等价本身；等价必须通过逐前缀参数对照或严格推导验证。

```bash
python3 implementations/classic/true_online_td.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-figures/result-true_online_td.svg)

横轴：environment_steps。纵轴：五状态价值 RMSE。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 当前价值估计与真实值 s/6 的误差。在五个非终止状态 s=1,…,5 上均匀平均平方误差后开方；权重不是训练访问频率。

**step：怎样计时。** step 包含全部真实转移，也包含尚未完成的回合。MC 在自然终止后，对该回合首次访问的每个状态写一次回报均值；同回合重复访问不增加该状态的样本数。TD(0) 每个真实步写入；资格迹可同时影响多个状态。相同环境步不表示相同参数写入次数，预算结束也不产生额外终止反馈。

**怎样汇总。** 取该记录时刻的值；不先对曲线上的时间点求平均。 各状态初值均为0。MC 用各状态已完成回合的首次访问计数决定步长；TD(0) 用常数步长0.1，比较不仅改变了是否自举。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** CSV 可重算各记录时刻的跨种子均值、样本标准差和末点误差；没有保存全部预测向量，不能仅凭 value 重新计算状态权重或逐状态误差。 日志未保存回合边界、首次访问计数或逐步价值，曲线平台不能单独区分等待终止、零更新和稀疏记录漏掉变化。

计算位置：[classic/_common.py](https://yingwen.io/crl-code/implementations/classic/_common.py) · [classic/td0.py](https://yingwen.io/crl-code/implementations/classic/td0.py) · [classic/td_lambda.py](https://yingwen.io/crl-code/implementations/classic/td_lambda.py) · [classic/true_online_td.py](https://yingwen.io/crl-code/implementations/classic/true_online_td.py) · [classic/mc_prediction.py](https://yingwen.io/crl-code/implementations/classic/mc_prediction.py)

</details>

**结果分析。** 第 300 步，True-online TD 的平均 RMSE 为 0.0791，TD(0) 为 0.2368，长程传播在早期有利；第 1200 步却为 0.0772 对 0.0465。早期较快不意味着固定 λ=0.8 在整个预算上都更准确。

**结论边界。** 这里只用 one-hot 特征，尚未通过此实验展示重叠线性特征的优势，也没有与普通累积 TD(λ) 作同 λ 对照。更不能把线性 true-online 等价性直接移植到非线性网络。

**继续实验。** 先令 λ=0，逐条核对它退化为 TD(0)。再选择含重复状态的短轨迹，以在线前向视图逐前缀核对 Dutch 更新；最后在相同 λ 下对照普通累积迹，避免把 λ 的效果误认为修正项的效果。

[源码](https://yingwen.io/crl-code/implementations/classic/true_online_td.py) · [逐种子记录](https://yingwen.io/crl-code/results/true_online_td/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/true_online_td/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/true_online_td/curves.json)

<a id="traces-corridor"></a>

## 走廊算例 · 共享方向怎样跨步保留

回到 A 到 B 到终点的同一固定策略任务：$\gamma=1,\lambda=0.5,\alpha=0.5$，$x_A=(1,0),x_B=(0.8,0.6)$，初始参数和迹为零。第一步得到奖励 2；更新前两个预测都是 0，所以 $\delta_0=2,e_0=x_A,w_1=(1,0)$。A 的新预测为 1，B 已被共享更新推到 0.8。下一步尚未发生，算法已经保存下来的下一状态旧预测却仍是 0。

$$
\begin{gathered}e_1^{\rm acc}=0.5x_A+x_B=(1.3,0.6),\qquad
\delta_1=-1-\hat v(B,w_1)=-1.8,\\
w_2^{\rm acc}=(1,0)+0.5(-1.8)(1.3,0.6)=(-0.17,-0.54).\end{gathered}
$$

第二步当前预测与终点目标均用更新前的参数计算。真正终止保留奖励 −1，尾值取零。

这条迹的两个坐标对应两个参数，不是分别给 A、B 存一份资格。用当前误差更新后，A 的预测变化为 $\alpha\delta_1 x_A^\top e_1^{\rm acc}$，其中内积为 1.3：来自保留 A 的 0.5，也来自当前 B 特征共享的 0.8。B 的内积为 1.4：来自历史 A 的 $0.5\times0.8=0.4$，加上自身的 1。时间衰减与状态间共享通过同一个内积相遇。

![同一坐标尺度下，衰减的 A 特征与 B 特征首尾相接，分别构成累积迹 (1.3,0.6) 和 Dutch 迹 (1.14,0.48)。](https://yingwen.io/crl-figures/concept-shared-gradient-memory.svg)

灰线是从上一时刻留下的 $0.5x_A$，青线是当前特征的贡献，虚线是合成向量。Dutch 图将 B 的系数改为 $1-\alpha\gamma\lambda x_A^\top x_B=0.8$。虚线表示向量合成关系。原创精确计算，沿用本节轨迹与参数；[逐步数值](/crl-figures/shared-gradient-data.json)。

现在独立执行在线前向定义。只见第一步时，A 的目标是 $2+\hat v(B,w_0)=2$，所以前缀结果也是 $(1,0)$。见到第二步后，A 的 interim λ-return 变成 $0.5\times2+0.5\times(2-1)=1.5$；B 的目标为 −1。从原始 $w_0=0$ 重做两次回归：先得到 $(0.75,0)$，再用 B 当前的预测 0.6 得到 $(0.11,-0.48)$。累积迹与这个明确定义的在线过程已经不同。

$$
\begin{aligned}
e_1^{\rm Dutch}&=0.5(1,0)+(1-0.5\times0.5\times0.8)(0.8,0.6)=(1.14,0.48),\\
\alpha\delta_1 e_1^{\rm Dutch}&=(-1.026,-0.432),\\
\alpha(v_1-v_{\rm old})(e_1^{\rm Dutch}-x_B)
&=0.5(0.8-0)(0.34,-0.12)=(0.136,-0.048),\\
w_2^{\rm true}&=(1,0)+(-1.026,-0.432)+(0.136,-0.048)=(0.11,-0.48).
\end{aligned}
$$

Dutch 迹修正方向，最后一项补偿 B 在第一步之后已经发生的预测变化。保存的旧预测必须仍为 0；若把它提前覆盖成 0.8，这一项就会错误消失。

![同一首步参数出发，普通累积迹、仅替换 Dutch 迹、加入完整预测变化修正的三个参数终点。](https://yingwen.io/crl-figures/concept-shared-gradient-online.svg)

蓝线是共同的第一步；红线使用累积迹，紫线只用 Dutch 迹的 TD 增量，橙线加入预测变化修正。坐标是两个参数，箭头是实际更新，不是环境动作。完整修正的终点等于独立在线前向重算；它与另两个终点的距离用于检查更新语义，不表示性能排名。原创精确计算。

本例连一个状态都没有重复访问，两个时刻仍因重叠特征而耦合。下面再把表示压缩为一个重复特征，可以用一个标量看清相同问题；它不是资格迹只对重复状态才必要的理由。

<a id="lesson-example"></a>

## 5 · 重复状态的两步手算

只有一个非终止状态，特征为 1。轨迹是“该状态、该状态、终止”，奖励依次为 0、1。取 $w_0=0,\gamma=\lambda=1,\alpha=0.5$。第一步没有误差，权重仍为 0，trace 为 1。

第二步，普通累积 trace 变为 2，误差为 1，因此权重变为 1。Dutch trace 则是 $1+[1-0.5\times1]\times1=1.5$；本例预测变化修正为零，得到权重 0.75。

$$
\begin{aligned}w^{\rm forward}_1&=0+0.5(1-0)=0.5,\\ w^{\rm forward}_2&=0.5+0.5(1-0.5)=0.75.\end{aligned}
$$

完整前缀的前向过程对两个出现位置依次作目标为 1 的监督更新，因此得到 0.75，与 true-online 相同。普通累积迹把两次误差都按旧预测累加，得到不同答案。

这不是说普通 TD 在这个小问题一定无法学习。例子只区分有限步长下的更新语义。把步长减小，两个结果更接近；把状态换成互不重叠的独立特征，也可能降低差异。

上例中的预测变化修正恰好为零。现在仍让同一特征出现两次，但将奖励改为 1、2，取 $\lambda=0.5$，其余保持 $\alpha=0.5,\gamma=1,w_0=0$。第一步就改变了预测，于是能分别看到 Dutch trace 与权重修正的作用。

![同一特征两次出现；首步已更新权重，第二步Dutch迹为1.25，TD增量0.9375之外还需0.0625的预测变化修正，最终权重1.5。](https://yingwen.io/crl-figures/concept-credit3-dutch-correction.svg)

灰段是保留下来的量，蓝段是 TD 增量，橙段是预测变化修正。底部从回合初始权重 0 重算同一已见前缀，两个更新目标都是 2，得到相同终点 1.5。两个方法此时都只使用已发生的两步经验。原创精确计算；公式对应 Sutton 与 Barto §12.5 及 van Seijen 等 Algorithm 2。

第一步后 $w_1=0.5,e_0=1$，保存的下一状态旧预测仍为 0，因为它在权重更新前计算。第二步的当前预测为 0.5，所以 $v_1-v_{\rm old}=0.5$。Dutch 迹为 $e_1=0.5\times1+[1-0.5\times0.5\times1]=1.25$，TD 误差为 $2-0.5=1.5$。

$$
w_2=0.5+\underbrace{0.5\times1.5\times1.25}_{0.9375}+\underbrace{0.5\times(0.5-0)\times(1.25-1)}_{0.0625}=1.5.
$$

最后一项是 $\alpha(v_t-v_{\rm old})(e_t-x_t)$。只改 Dutch 迹却删去这项，得到 1.4375；普通累积迹为 1.5，对应参数结果 1.625。三个数的区别是算法语义，不是三个方法的性能排序。

独立检查使用在线前向定义。第二步到达后，第一个出现位置的 interim λ-return 为 $0.5\times1+0.5\times(1+2)=2$，第二个位置的目标也为 2。从原始 $w_0=0$ 顺序回归，两次参数更新是 $0\to1\to1.5$。附带测试还用不同的非二值特征、步长与 λ，逐前缀核对两种计算，而不只比较最后一步。

<a id="rlss-trace-error-bound"></a>

## λ 改变什么：预测误差、信用跨度与表示不足

前面的走廊与重复特征比较了每个前缀怎样写入参数。现在回到函数逼近预测章的另一问题：学习稳定后会得到哪个近似？以下先分析普通累积迹 TD(λ) 的冻结参数平均更新，再看已有的采样诊断；逐前缀算法等价与长期固定点在这里分开。固定策略、固定特征和常数折扣 $0\leq\gamma<1$，令 $T_\pi v=r_\pi+\gamma P_\pi v$。对 $0\leq\lambda<1$，把各个 n 步目标按几何权重混合。

$$
T_\lambda v=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}T_\pi^n v,\qquad
\|T_\lambda v-T_\lambda u\|_\infty\leq \frac{\gamma(1-\lambda)}{1-\gamma\lambda}\|v-u\|_\infty.
$$

把每项的 γⁿ 误差界提出，再求几何级数，得到这个系数。λ 越接近一，期望目标对旧预测的依赖越弱；回报样本的方差则需另算。

$$
\begin{aligned}T_\lambda v&=(I-\gamma\lambda P_\pi)^{-1}[r_\pi+\gamma(1-\lambda)P_\pi v],\\ \Phi w_\lambda&=\Pi_{D_\pi}T_\lambda(\Phi w_\lambda).\end{aligned}
$$

第一式把混合目标的几何级数求和；$\gamma<1$ 时也可用它定义 $\lambda=1$ 的端点。第二式沿用预测章的投影：$\Pi_{D_\pi}=\Phi(\Phi^\top D_\pi\Phi)^{-1}\Phi^\top D_\pi$，假定特征 Gram 矩阵可逆。$\lambda=0$ 恢复一步投影 Bellman 方程；$\lambda=1$ 时 $T_1v=v_\pi$，回到价值回归投影。下一节从稳态累积迹独立推导这个固定点。

$$
\|v_{w_\lambda}-v_\pi\|_{D_\pi}^{2}\leq
\frac{1-\gamma\lambda}{1-\gamma}\min_w\|v_w-v_\pi\|_{D_\pi}^{2}.
$$

这是固定线性表示、在策略稳态加权及相应存在与收敛条件下的常用渐近误差界。不是深网、离策略或有限常数步长轨迹的保证。λ=1 的极限给 MC 的投影目标；γ=1 的回合任务需另一组 properness 条件，不能代入本式的分母。

这个界把两件事分开。表示无法表达真值时，最佳可达误差本来就非零。TD 的自举固定点还可能偏离这个最佳投影。增大 λ 可以减弱后一种差距，但不会创造缺失的状态信息。若两个历史被压成同一特征，长 trace 不会自动让策略在这两个历史中选择不同动作。

$$
q=\gamma\lambda,\qquad H_{1/2}=\frac{\log(1/2)}{\log q}\quad(0<q<1),\qquad
\sum_{j=0}^{\infty}q^j=\frac1{1-q}.
$$

第一式给 trace 在无新激活时衰减到一半所需的步数。第二式给单位特征每步重复激活时累积迹的极限。γ 规定预测问题的未来权重；λ 是学习算法的自举/信用选择，虽然它们在 trace 中相乘，含义仍不同。

例如 $\gamma=1,\lambda=.9$ 时，半衰期约 6.58 步，但反复激活的累积迹可接近 10。把当前梯度归一化到单位长度，也不会把累计 trace 的长度限制到一。相同步长在不同 λ、激活频率和特征重叠下产生的实际预测改变量不同。对稀疏表示，还应记录 trace 非零项数量，不能仅根据当前特征的稀疏度估计开销。

| 观察到的问题 | 不能立即得出的结论 | 下一项诊断 |
| --- | --- | --- |
| λ 增大后误差变小 | 长迹在所有任务上更好 | 联合扫描步长；拆开偏差和方差 |
| λ 增大后参数爆炸 | 长时信用本身不可用 | 记录实际更新、trace 范数与重复激活 |
| MC 仍有大误差 | 还需要更长 trace | 检查表示别名和数据覆盖 |
| 运行很久后学习变慢 | 必定是可塑性丧失 | 先冻结表示与目标，区分追踪时标和梯度退化 |

<a id="rlss-trace-lab-fixedpoint"></a>

## 诊断实验一：λ 能改变固定点，不能补全状态

问题：若两个实际状态具有相同特征，增加 λ 能否修复预测？下面只研究固定策略的预测。环境是三个状态的持续 Markov 奖励过程。没有动作，没有终止状态，也没有回合重置。每步先按当前状态发出奖励，再按转移矩阵采样下一状态。γ 固定为 0.9；所有 λ 预测同一个折扣回报。

$$
P=\begin{bmatrix}.8&.2&0\\0&.5&.5\\.3&0&.7\end{bmatrix},\quad
r=\begin{bmatrix}1\\0\\-1\end{bmatrix},\quad
\Phi=\begin{bmatrix}1&0\\1&0\\0&1\end{bmatrix},\quad
d=\frac1{31}\begin{bmatrix}15\\6\\10\end{bmatrix}.
$$

P 的每一行给出下一状态的分布，r(s) 是 Rₜ₊₁ 在当前状态 s 的值。Φ 的行是学习器可见特征。状态 0 与 1 共用一个预测权重。d 是环境稳态分布，满足 dᵀP=dᵀ。

信息权限必须分开。在线 TD 只接收当前特征、奖励和下一特征。解析诊断器可以读取 P、r 和隐藏状态，并计算真值。它们不能进入 TD 更新。奖励可能帮助一个有记忆的学习器区分状态；本实验故意固定无记忆特征。因此“表示不足”是这里选定学习器的限制，不是整个任务不可能解决。

$$
v=(I-\gamma P)^{-1}r,\qquad
w_{\rm proj}=(\Phi^\top D\Phi)^{-1}\Phi^\top Dv,\qquad D=\operatorname{diag}(d).
$$

第一式由 v=r+γPv 解出环境真值。第二式最小化稳态加权平方预测误差。它是此特征空间的最佳预测，不是实际 TD 自动获得的答案。

本例的 $v\approx(3.4890,-0.1282,-0.1567)^\top$。最佳可表示预测为 $(2.4555,2.4555,-0.1567)^\top$，RMS 误差仍为 1.34494。前两个状态的真值不同，而预测必须相同；只延长 trace 不会产生新的特征。

再求 TD 的目标。先将参数冻结为任意 $w$，令稳态 trace 包含无限过去。展开 $e_t=\sum_{k\geq0}(\gamma\lambda)^k\phi_{t-k}$，并逐项求 $\mathbb E[e_t\delta_t]$。每一项包含从过去状态到当前状态的 $P^k$。几何求和给出下式。

$$
\begin{gathered}\mathbb E[e_t\delta_t\mid w\text{ fixed}]=b_\lambda-A_\lambda w,\\
A_\lambda=\Phi^\top D(I-\gamma\lambda P)^{-1}(I-\gamma P)\Phi,\qquad
b_\lambda=\Phi^\top D(I-\gamma\lambda P)^{-1}r,\qquad
w_\lambda=A_\lambda^{-1}b_\lambda.\end{gathered}
$$

这里的期望是对固定参数、稳态状态与迹分布求的。实际在线 wₜ 与经验相关，不能把该式直接当成 E[wₜ] 的精确递推。λ=1 时两个环境矩阵抵消，得到 w₁=w_proj；λ<1 一般得到不同的投影 Bellman 固定点。

$$
\|\Phi w_\lambda-v\|_D^2
=\underbrace{\|\Phi w_{\rm proj}-v\|_D^2}_{\text{表示误差}}
+\underbrace{\|\Phi w_\lambda-\Phi w_{\rm proj}\|_D^2}_{\text{固定点偏离最佳投影}}.
$$

最佳投影残差与特征空间正交，因此交叉项为零。这是误差分解，不是“TD 噪声”的分解；此时尚未进行采样学习。

![TD 固定点与最佳可表示预测的稳态加权距离随 λ 变化](https://yingwen.io/crl-code/diagnostics/rlss-traces/fixed-point-gap.svg)

图一只画误差分解的第二项开平方。λ=1 时它为零；总预测误差不为零。曲线来自明确给出的线性系统，不是训练曲线。

| λ | 解析 TD 价值 RMS 误差 | 同一个表示下限 |
| --- | --- | --- |
| 0 | 1.39930 | 1.34494 |
| 0.5 | 1.36390 | 1.34494 |
| 0.9 | 1.34597 | 1.34494 |
| 1 | 1.34494 | 1.34494 |

解析固定点、期望迭代与稳定性：这些函数需要环境模型，只用于诊断。完整文件包含小型矩阵求解器。

```python
def projected_system(lam):
    """Stationary, frozen-parameter mean TD drift = b_lambda - A_lambda w.

    D weights hidden states by their stationary probabilities. The inverse is
    a geometric sum of transition powers. No trajectory is sampled here.
    """
    resolvent = i_minus(GAMMA * lam)
    a = mm(PHI_T_D, solve_columns(resolvent, mm(i_minus(GAMMA), PHI)))
    b = mv(PHI_T_D, solve(resolvent, R))
    return a, b, solve(a, b)


def expected_iteration(a, b, alpha, steps):
    """Model-based deterministic drift iteration, NOT a sampled TD run."""
    w = [0.0, 0.0]
    for _ in range(steps):
        aw = mv(a, w)
        w = [x + alpha * (target - current) for x, target, current in zip(w, b, aw)]
    return w


def spectral_radius(a, alpha):
    """rho(I-alpha*A), including complex eigenvalues of a real 2x2 matrix."""
    trace = a[0][0] + a[1][1]
    determinant = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    root = cmath.sqrt(trace * trace - 4 * determinant)
    eigenvalues = [(trace + root) / 2, (trace - root) / 2]
    return max(abs(1 - alpha * eig) for eig in eigenvalues)
```

读图问题：图一趋于零，为什么不意味着预测已经精确？如果把 Φ 换成三个状态的一热编码，哪一项先消失？固定点的解释参见 [Tsitsiklis 与 Van Roy 的原论文，第三节](https://web.mit.edu/jnt/www/Papers/J063-97-bvr-td.pdf)；资格迹与线性函数近似见 [Sutton 与 Barto，第 12 章](http://incompleteideas.net/book/the-book-2nd.html)。

<a id="rlss-trace-lab-scale"></a>

## 诊断实验二：更长的信用跨度，也改变更新尺度

固定预测器时，一次过去特征激活对 $k$ 步后 TD 误差的系数为 $(\gamma\lambda)^k$。这是信用传播。取本例 γ=0.9：λ=0 时只更新当前激活；λ=0.5、0.9、1 时，十步前激活的系数分别约为 0.00034、0.12158、0.34868。此处比较的是误差的分配系数，不是最终预测误差。

另一件事是反复激活。本例每一步都有一个非负单位特征。故资格迹的 L1 范数不依赖具体经过哪一个状态。即使两个坐标轮流激活，旧迹与新迹仍然累加。

$$
e_{-1}=0,\quad e_t=\gamma\lambda e_{t-1}+\phi_t,\qquad
\|e_t\|_1=\frac{1-(\gamma\lambda)^{t+1}}{1-\gamma\lambda},\qquad
\|\Delta w_t\|_1=\alpha|\delta_t|\,\|e_t\|_1.
$$

范数恒等式依赖这里的非负一热特征；有符号或一般连续特征不能直接套用。后一个等式来自 Δw=αδe，对任意 trace 都成立。

![四种 λ 下累积迹 L1 范数从零增长到不同极限](https://yingwen.io/crl-code/diagnostics/rlss-traces/trace-mass.svg)

图二：横轴为已观察转移数，初始 trace 为零。γ=0.9 时，λ=0、0.5、0.9、1 的极限分别为 1、1.818、5.263、10。相同 α 和相同 δ 不代表相同大小的参数更新。

尺度也会改变确定性迭代的稳定范围。令解析迭代为 $w_{k+1}=w_k+\alpha(b_\lambda-A_\lambda w_k)$，则 $w_{k+1}-w_\lambda=(I-\alpha A_\lambda)(w_k-w_\lambda)$。对每个初值都收敛的充要条件是该矩阵的谱半径严格小于 1。α=0 位于边界，但完全不学习。

![完整 α 扫描中各 λ 的确定性期望迭代谱半径](https://yingwen.io/crl-code/diagnostics/rlss-traces/expected-stability.svg)

图三：所有 λ 都扫描 α=0,0.1,…,10，没有分别挑选有利步长。曲线低于 1 的部分是确定性迭代稳定区。一次这样的迭代使用已知模型做矩阵计算，不是一次流式样本更新。

例如 α=3 时，λ=0 的谱半径约为 0.855，λ=1 约为 1.032；前者的期望迭代收敛，后者一般发散。这不意味着在线 TD 可安全使用 α=3。随机更新、状态相关性和常数步长偏差不由这个二维确定性稳定条件完全刻画。下一实验使用远小得多的采样步长。

读图问题：若把 α 乘以 1−γλ，累计迹的极限尺度得到补偿，是否连学习噪声、收敛速度和每个方向的更新也相同？答案是否定的。Aλ 的几何、δ 的分布和样本相关性仍会改变。尺度补偿是诊断对照，不是最优步长公式。

<a id="rlss-trace-lab-sampling"></a>

## 诊断实验三：有限经验流与解析答案相差多少

现在去掉学习器对模型的访问。每个条件独立运行 16 个种子，每次 20,000 步，从稳态分布采样初始状态；权重和 trace 均从零开始。不同 λ 和步长规则使用配对的同一状态流。保留 λ=0、0.5、0.9、1 的全部结果。规则一固定 α=0.01；规则二用 α=0.01(1−γλ) 补偿累计迹的极限尺度。没有经验回放、目标网络、投影截断或隐藏的梯度裁剪。

真正的采样 TD(λ)：先用旧权重计算 δ，再更新 trace 与权重；诊断记录不进入学习器。

```python
def td_step(w, e, features, reward, next_features, lam, alpha):
    """Accumulating on-policy semi-gradient TD(lambda), fixed linear features.

    Compute both predictions before changing any weight. There is no replay,
    model, hidden state, gradient through the target, or trace reset.
    """
    delta = reward + GAMMA * dot(w, next_features) - dot(w, features)
    trace = [GAMMA * lam * old + feature for old, feature in zip(e, features)]
    next_w = [weight + alpha * delta * z for weight, z in zip(w, trace)]
    return next_w, trace


def run_stream(seed, lam, alpha, steps=STEPS, checkpoints=CHECKPOINTS):
    w, e = [0.0, 0.0], [0.0, 0.0]
    _, _, target = projected_system(lam)
    target_values = mv(PHI, target)
    records = []

    def record(step):
        fitted = mv(PHI, w)
        records.append({"step": step, "weights": list(w),
                        "value_rmse": value_distance(fitted, VALUE),
                        "fixed_point_distance": value_distance(fitted, target_values),
                        "trace_l1": sum(abs(x) for x in e)})

    record(0)
    wanted = set(checkpoints)
    for t, (features, reward, next_features) in enumerate(transitions(seed, steps), 1):
        w, e = td_step(w, e, features, reward, next_features, lam, alpha)
        if not all(math.isfinite(x) for x in w):
            # Never omit a failed run or serialize Infinity as a curve point.
            raise ArithmeticError(f"Nonfinite weights: seed={seed}, lambda={lam}, alpha={alpha}, step={t}")
        if t in wanted:
            record(t)
    return {"seed": seed, "lambda": lam, "alpha": alpha, "checkpoints": records}
```

![两种预定步长规则下的有限轨迹终点误差、运行范围与解析参照](https://yingwen.io/crl-code/diagnostics/rlss-traces/sampled-error.svg)

图四：每个采样点为 16 个独立种子的第 20,000 步 RMS 误差均值；范围带为这些运行的最小值至最大值，不是置信区间。解析曲线是不含采样噪声的 TD 固定点；水平线是表示下限。全部运行均保留。

| λ | 固定 α：均值 ± 标准差 | 尺度补偿 α：均值 ± 标准差 |
| --- | --- | --- |
| 0 | 1.4203 ± 0.0366 | 1.4203 ± 0.0366 |
| 0.5 | 1.4056 ± 0.0567 | 1.3900 ± 0.0322 |
| 0.9 | 1.4355 ± 0.1125 | 1.3709 ± 0.0293 |
| 1 | 1.4996 ± 0.1869 | 1.3671 ± 0.0303 |

结果把三个问题分开了。解析固定点随 λ 接近最佳投影，但固定 α 的有限样本误差没有随之单调下降。λ=1 的解析误差最低，固定 α 时的这组终点均值却最高。尺度补偿缩小了这一差距，但仍没有消除表示下限。这组数字只说明机制可以相互抵消，不建立算法排名，也不能证明这个补偿规则适合其他任务。标准差描述运行间离散程度，不是均值的置信区间。

计算预算也应明确。实验共 128 条采样运行，合计 2,560,000 次 TD 更新；每步维护两个权重和两个迹坐标。模型求解和确定性期望迭代另计，不冒充环境样本。JSON 还保留每次运行的中间检查点、终点、全部 α 稳定性扫描以及相同迭代次数的确定性迭代。相同迭代次数不代表相同经验或计算预算。常数步长末端误差不等于渐近收敛定理。

仅需 Python 3.10 及标准库；run 会写入指定 JSON 文件。

```bash
python3 rlss_trace_diagnostics.py test
python3 rlss_trace_diagnostics.py run --output results.json
```

[下载独立实验代码](/zh/continual-rl/download/rlss_trace_diagnostics.py) · [查看全部结果与运行协议](/crl-code/diagnostics/rlss-traces/results.json)。test 核对稳态、Bellman 真值、λ=1 投影、矩阵逆与 400 项几何展开的一致性，以及逐步 trace 和更新范数恒等式；它们是实现检查，不是学习性能证据。

这些已有结果使用固定表示、固定策略与普通累积迹。它们把最佳可达误差、固定点偏差和有限经验误差分开，却没有测量 true-online 在该任务上的效果。下一节用两个独立实现逐前缀检查 true-online 的更新语义；之后再撤掉固定特征条件，观察旧梯度怎样失配。若表示或环境改变，需相应记录目标漂移、梯度方向与真实更新，才知道哪条固定点分析仍可使用。

<a id="lesson-code"></a>

## 6 · 两个独立实现逐前缀对照

先运行[走廊标准库脚本](/crl-code/tutorials/shared_gradient_walkthrough.py)：它打印每步旧预测、迹、TD 增量与预测变化修正，并另写一个完全不使用资格迹的前向重算。比较每个前缀，再尝试 λ=0 的退化情形，能够定位时序错误。

两步固定轨迹；无训练或外部依赖

```sh
python3 shared_gradient_walkthrough.py
python3 shared_gradient_walkthrough.py --json
```

True-online 更新与慢速在线前向参照

```python
def true_online_td(features, rewards, alpha=0.1, gamma=0.9, lam=0.8, initial=None):
    """One episode/prefix. features has len(rewards)+1; terminal features are zero."""
    if len(features) != len(rewards) + 1 or not 0 <= lam <= 1:
        raise ValueError("invalid trajectory or lambda")
    weights = list(initial) if initial is not None else [0.0] * len(features[0])
    trace, old_value, history = [0.0] * len(weights), 0.0, [weights[:]]
    for x, reward, xp in zip(features, rewards, features[1:]):
        value, next_value = dot(weights, x), dot(weights, xp)
        delta = reward + gamma * next_value - value
        correction = 1 - alpha * gamma * lam * dot(trace, x)
        trace = [gamma * lam * e + correction * v for e, v in zip(trace, x)]
        weights = [w + alpha * (delta + value - old_value) * e
                   - alpha * (value - old_value) * v
                   for w, e, v in zip(weights, trace, x)]
        old_value = next_value  # Save the PRE-update prediction at xp.
        history.append(weights[:])
    return history


def online_forward_view(features, rewards, alpha=0.1, gamma=0.9, lam=0.8, initial=None):
    """Slow independent reference: recompute all interim lambda-return updates."""
    initial = list(initial) if initial is not None else [0.0] * len(features[0])
    history = [initial[:]]
    for horizon in range(1, len(rewards) + 1):
        weights = initial[:]
        for start in range(horizon):
            target = 0.0
            for n in range(1, horizon - start + 1):
                end = start + n
                nreturn = sum(gamma ** j * rewards[start + j] for j in range(n))
                nreturn += gamma ** n * dot(history[end - 1], features[end])
                mixture = lam ** (n - 1)
                if end < horizon:
                    mixture *= 1 - lam
                target += mixture * nreturn
            weights = gradient_mc(weights, features[start], target, alpha)
        history.append(weights)
    return history


def traces_demo():
    features, rewards = [[1.0], [1.0], [0.0]], [0.0, 1.0]
    online = true_online_td(features, rewards, alpha=0.5, gamma=1.0, lam=1.0)
    forward = online_forward_view(features, rewards, alpha=0.5, gamma=1.0, lam=1.0)
    return {"true_online_history": online, "forward_history": forward,
            "accumulating_final": 1.0, "true_online_final": online[-1][0]}
```

测试使用多维、重叠且非二值的特征，同时改变步长和 lambda；对每个可见前缀核对参数。只检查最后一个权重可能漏掉中途时序错误。还检查 lambda 为零退化成 TD(0)、零步长不更新以及空前缀。

运行重复状态例子与前后向一致性检查

```sh
python3 approximation_textbook_lab.py traces
python3 approximation_textbook_lab.py test
```

<a id="lesson-branches"></a>

## 7 · 动作价值、流式内存与内部状态

将特征改成状态—动作特征，并按 Sarsa 的实际下一动作计算 bootstrap，可以构造动作价值的 true-online 版本。离策略时还涉及重要性比、trace 截断或其他校正；不能把在策略公式直接用于任意旧数据。

n-step 在线更新通常保留最近 n 步的奖励、状态或特征。线性 trace 使用 $O(d)$ 递归内存，不随生命期长度增长；但稀疏特征的 trace 可能逐渐变稠密。截断小 trace 可以节约计算，也改变精确更新。

资格迹保存“过去哪些参数方向应对当前误差负责”。Agent state 保存“为当前预测或决策应记住什么历史”。两者并不相同。递归神经状态的参数变化还需要通过状态转移传播导数；把线性特征 trace 叫作完整 RTRL 或 BPTT，会混淆两种信用路径。

$$
e_t=\gamma\lambda e_{t-1}+\nabla_w\hat v(S_t,w_t)=\sum_{k=0}^{t}(\gamma\lambda)^{t-k}\nabla_w\hat v(S_k,w_k).
$$

这是普通非线性梯度迹实际保存的对象：每个历史状态在当时参数下的梯度。它通常不等于把所有历史状态在当前参数 $w_t$ 下重新求导后再相加。

仍用走廊的奖励与时间顺序，只换成一个非线性预测器：$\hat v(A,\theta)=\theta^2,\hat v(B,\theta)=\theta$，初值 $\theta_0=0.5$。第一步 $\delta_0=2+0.5-0.25=2.25$，A 的梯度为 1，步长 0.5 使 $\theta_1=1.625$。到 B 时，普通梯度迹是 $0.5\times1+1=1.5$；若把历史 A 用当前参数重新求导，会得到 $0.5\times(2\times1.625)+1=2.625$。环境没有变化，迹的差别已由参数变化产生。这次替换也改变了函数类，所以该数字只检查旧梯度与当前梯度的含义。

在线性走廊中，特征不随权重改变，两个向量和一个旧预测就能精确压缩前向重算。非线性例子要重新获得 A 的当前梯度，通常需要再次取得 A 的输入并计算导数；长轨迹下重算全部历史会失去固定资源的优点。持续流中的问题因而更具体了：给定每步时间与内存，多久以前的梯度仍适合承担当前信用？可先固定轨迹与奖励，分别记录旧梯度和当前梯度的夹角、预测变化及每步代价，再加入环境变化，避免把表示变化与任务变化混为一谈。

当表示变化缓慢时，旧梯度方向可能仍有用；变化快速时，当前误差沿旧方向更新可能已不能改变当初那项预测，甚至改变相反。较长迹同时扩大信用范围与梯度陈旧程度。缩短 λ、限制更新量或重新计算历史梯度处理的是不同折中；后者又会引入存储和计算成本。不能仅在代码中把线性 x 换成自动微分梯度，就宣称保留 true-online 的精确前向等价。

持续变化时，较长 trace 可以更快传递延迟反馈，也可能跨越动力学变化而把新误差作用于旧情境。实验需要改变奖励延迟和环境变化频率，记录恢复速度与 trace 范数，不能只在固定短回合中选一个 lambda。

<a id="lesson-check"></a>

## 8 · 练习与答案

- 为什么有限回合 lambda-return 的最后项没有乘 $1-\lambda$？答案：它必须吸收所有更长回报的剩余几何权重，否则权重和不足 1。
- 能否把 $v_{\rm old}$ 存成更新后对下一状态的预测？答案：不能；true-online 修正需要更新前计算的下一预测。
- 普通累积迹与冻结权重前向更新相等，为何在线仍不同？答案：冻结权重推导中所有价值和梯度不变，在线更新破坏了这一前提。
- $\lambda=0$ 时 Dutch trace 和修正如何退化？答案：$e_t=x_t$，两个预测变化修正相抵，剩下普通 TD(0)。

## 从本章进入实践

[预测与控制](https://yingwen.io/zh/continual-rl/code/#practice-prediction)：学会预测更多事情，什么时候会改变行动？



<a id="chapter-code"></a>

## 下载与运行

标准库教学实现。命令运行本章的可解析小问题和全文件的公式测试；不依赖深度学习库，不等同于论文中的大规模实验。

[下载 approximation_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/approximation_textbook_lab.py)

```sh
python3 approximation_textbook_lab.py traces
python3 approximation_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：本章前后向定义对应 §§12.1–12.4；完整 true-online、Dutch 与二值 replacing trace 对应 §12.5；共同末端目标的 MC 递归压缩对应 §12.6。在策略动作价值扩展见 §12.7。

- [van Seijen et al. · True Online Temporal-Difference Learning](https://jmlr.org/papers/v17/15-599.html)：在线前向定义、Dutch trace 和精确等价的原始期刊论文；Algorithm 2 对应本文的常数步长更新。

- [Mahmood · True-online TD random MDP experiments](https://github.com/armahmood/totd-rndmdp-experiments)：论文作者的随机 MDP 实验代码，包含 accumulating、replacing 与 true-online 对照。先读 pysrc 和 pysrctest，再查看完整实验脚本。本文没有重跑这些大批量实验。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-approximation-traces#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-approximation-traces#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-approximation-traces)
<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 怎样把较晚的反馈归给较早的计算？

多步回报定义用多远的未来构造目标。资格迹压缩过去的特征或梯度方向。前向与后向等价必须说明参数是在整个轨迹内固定，还是每一步改变。

函数逼近与深度方法：神经网络改变后，旧梯度不再等于用当前参数重算的梯度。递归状态还带来参数经过历史状态影响当前输出的路径，不能用一条普通 TD trace 代替。

持续学习中的研究问题：在每步计算有界的条件下，保留多少过去影响才有用？替换特征时，怎样处理与旧特征绑定的资格迹、优化器动量和元梯度？

[多步回报](https://yingwen.io/zh/continual-rl/foundations/tabular/multistep/) → [资格迹与等价条件](https://yingwen.io/zh/continual-rl/foundations/approximation/traces/) → [GAE 与 actor–critic](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/) → [在线信用分配](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)


### 可进一步检验的问题

- [05 · 远处的反馈应该怎样更新早先的决策与内部计算？](https://yingwen.io/zh/continual-rl/research/#research-temporal-credit)：在线前向参考、荷兰迹和预测差修正说明精确等价依赖哪些版本条件，随后才能研究非线性与变动表示。
- [07 · 哪些学习参数应当适应，怎样评价学出来的更新规则？](https://yingwen.io/zh/continual-rl/research/#research-learning-rules)：资格迹规定本次误差怎样改变权重；学习步长或迹长度则要继续追踪这些更新怎样影响后续评价。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)
- [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)
- [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)
- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)

对应原始材料：第 12 章：Eligibility Traces。本文为原创讲解，原书、论文与上游代码保留各自许可。
