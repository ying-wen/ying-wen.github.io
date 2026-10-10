# 多步学习：n-step、Tree Backup 与 Q(σ)

表格强化学习 · 第 6 章

学习目标可以在一步 bootstrap 与完整回报之间选择，也可以在动作采样与动作期望之间选择。这是两条不同的设计维度。

## 本章内容

- 推导 n-step 回报及延迟更新时序。
- 理解 Tree Backup 的已采样分支和未采样分支。
- 从两端推导 on-policy Q(σ)，检查终止与退化条件。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [TD 预测与控制：SARSA、Expected SARSA、Q-learning 和 Double Q](https://yingwen.io/zh/continual-rl/foundations/tabular/temporal-difference/)：掌握一步TD、Sarsa与目标策略，再改变等待长度和分支期望。


### 概率与期望

概率非负且总和为一；期望是按概率加权的平均，不是单次结果。条件期望只比较满足已知条件的情形。

### 递推与赋值

递推通过已有估计计算新估计。下标表示时间；更新符号表示程序中的赋值，不是数学恒等式。

<a id="problem-definition"></a>

## 本章的问题定义

固定预测问题后，决定使用多少个真实奖励，以及如何处理后继动作的不确定性。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 目标策略 $\pi$、行为策略 $b$、备份深度 $n$ 与当前估计 $V$；终止时间记为 $T$。

### 需要求解的对象

构造具有明确期望对象的多步学习目标；控制版本还需指定目标策略如何更新。

### 信息与数据权限

可以等待并保存一个有限窗口；离策略概率比需要行为概率和覆盖。

$$
G_{t:t+n}=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^nV(S_{t+n}),\quad t+n<T
$$

该目标估计固定策略价值；在真实终止处截断奖励和并去掉尾部自举。增加 n 只改变估计器，不能自动保证更高回报。

### 成立条件与解的含义

- 前向恒等式须规定备份中使用的参数版本；在线变化参数时不能直接套冻结参数推导。
- 离策略采样校正、策略期望和混合策略各有覆盖与有界性条件。

判断准则：验证一步极限、终止边界、确定性例子及离策略概率比的起止索引。

### 适用边界

- 把更长备份等同于更无偏、更低方差或更好的控制。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)：本章的多步目标可接入资格迹的后向实现；前向与后向更新是否等价取决于参数版本等条件。

- 改变信息或数据协议 · [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)：本章允许等待 n 步并保存窗口，相对严格流式增加延迟与存储。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

短备份依赖不准确的尾值；长备份增加等待、方差及离策略概率乘积。

### 本章的核心思路

把时间深度与动作采样方式分成两个独立的选择维度。

1. [展开奖励并保留一个尾值](#lesson-derive)：明确哪些后果是真实观察，哪些仍来自预测。

2. [未采样的动作分支用期望保留](#tree-backup)：Tree Backup 以策略概率传播后续误差，不等于简单乘整条轨迹比率。

3. [连续调节采样与期望](#q-sigma)：逐层的 σ 选择改变估计方式；两端退化情况可作为实现测试。

结论与条件：冻结估计下可精确核对目标代数关系；控制学习收敛仍需额外假设。

### 相关方法改变了什么

- n-step / λ-return：前者选择固定深度，后者混合多个深度。

- 采样 IS / Tree Backup / Q(σ)：分别使用概率校正、动作期望和逐层混合来处理行为差异。


<a id="lesson-setting"></a>

## 1 · 两条相互独立的选择轴

一步 TD 很快得到更新，却可能把尚不准确的后继预测传给更早状态；MC 看到了完整后果，却等待更久、承受更多随机性。这两个极端提示一个自然问题：能否先观察若干步，再使用预测补尾？多步方法让我们明确选择经验与已有预测各承担多大一部分。

一步 TD 使用一个奖励再接预测；MC 等完整回合；n-step 使用若干实际奖励后才 bootstrap。另一方面，SARSA 采样下一动作，Expected SARSA 对下一动作求和。回报跨多长时间，与每层对动作采样还是求期望，是两个不同问题。

先考虑固定策略、表格预测、真实终点 T 与有限可积回报。记 $G_{t:h}$ 是从 t 开始、利用经验到 h 的目标，$h=\min(t+n,T)$。下面区分更新前的当前表与为推导恒等式而冻结的快照。

控制主例沿用上一章的A/B/C任务，并在本页写全所需数据：A的go得1到B、exit得0到终点；B的fast得2到C、detour得4到终点；C的finish得3、quit得−1后均终止。旧Q依次为A(2,0)、B(1,5)、C(4,0)，冻结行为与目标概率依次为(.75,.25)、(.25,.75)、(.75,.25)，γ=.9。实际给定轨迹选go、fast、finish，奖励依次1、2、3。目标构造先使用冻结快照，在线执行顺序在后面单独追踪。

<a id="lesson-derive"></a>

## 2 · n-step 回报与等待

$$
G_{t:t+n}=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^nV_{t+n-1}(S_{t+n})
$$

重复展开回报递推 n 次，将仍未观察的余项替换为第 n 次转移到来前的价值估计。若先到终点，则只累积到 T，不加终点价值。

$$
V_{t+n}(S_t)=V_{t+n-1}(S_t)+\alpha[G_{t:t+n}-V_{t+n-1}(S_t)]
$$

第 t 时刻尚无未来 n 个奖励，实际最早在 t+n 才能执行这次更新。下标不是装饰，它规定了可用信息与更新时序。

开始时前 n−1 次转移没有可完成的更新。到终点后也不能立即清空缓冲区：最后 n−1 个未完成目标还要用已知终点逐一补齐。保存 n+1 个状态和 n 个奖励的环形缓冲区就足够，不需要永久保存整条生命。

n 较大通常减少对当前价值误差的依赖，但增加随机奖励、等待和行为失配的影响。不能仅凭“更接近 MC”断言更正确；不同 n 的有限样本误差由噪声、当前估计和访问分布共同决定。

$$
\begin{aligned}\mathbb E_\pi[G_{t:t+n}\mid S_t=s]-v_\pi(s)&=\gamma^n\mathbb E_\pi[V(S_{t+n})-v_\pi(S_{t+n})\mid S_t=s],\\\max_s\left|\mathbb E_\pi[G_{t:t+n}\mid S_t=s]-v_\pi(s)\right|&\le\gamma^n\|V-v_\pi\|_\infty.\end{aligned}
$$

误差缩减性质在固定策略、冻结 bootstrap 预测 V 及正确的 Markov 状态下成立。对真实终止后的状态令两种价值都为0。实际奖励部分的期望与真实回报相消，剩下的只有第 n 步尾项误差。

例如 γ=.9，当前最大价值误差为10。一、五、二十步目标的最大条件期望偏差分别不超过9、5.9049、1.2158。这里约束的是“目标的条件期望”，不是单个目标与答案的距离。二十步中若有高方差奖励，实际平方误差仍可能更大。

$$
\mathbb E[(G^{(n)}-v_\pi(s))^2\mid s]=\operatorname{Var}(G^{(n)}\mid s)+\big(\mathbb E[G^{(n)}\mid s]-v_\pi(s)\big)^2
$$

偏差平方与方差的分解说明为何更长 n 不是无条件更优。各时刻奖励可能相关，所以方差也不应简单写成独立奖励方差之和。

函数逼近以后，还要投影回有限表示空间，不能把目标的误差缩减直接读成参数误差收缩。严格流式学习还关心等待 n 步期间错过了多少决策改善，以及缓冲区是否可接受。资格迹在后一问题上提供了递归实现，但并不会让尚未发生的奖励提前可见。

<a id="rlss-multistep-design"></a>

## 两条设计轴：看多远，在哪一层求期望

多步学习有两个独立选择。第一个是备份深度：使用多少步真实反馈再自举。第二个是备份宽度：对哪些随机变量采样，对哪些随机变量求期望。把下一动作求期望，不等于知道下一状态的分布；Tree Backup 仍沿实际轨迹的状态转移向前展开。

| 方法 | 中间动作 | 末端动作 | 额外资源 |
| --- | --- | --- | --- |
| n-step Sarsa | 采样 | 采样 | 保存最近 n 步并延迟更新 |
| n-step Expected Sarsa | 采样 | 求期望 | 末端枚举合法动作 |
| Tree Backup | 未走动作用当前 Q；已走动作继续展开 | 求期望 | 每层枚举动作，不枚举未知环境后果 |
| Q(σ) | 由 σ 混合采样与期望 | 按所选定义混合 | 必须同时规定行为校正与时间索引 |

离策略代码还要检查返回值的具体定义。带控制变量的递推在 σ=1 时得到逐决策校正的采样回报，并不逐样本等于“整个 n 步回报乘一个概率比乘积”的普通实现。两者可以针对同一个期望量，样本值和方差却不同。固定 π、只更新 Q 的试验是策略评价；只有同时改善 π，才是在检验控制算法。

$$
\mathbb E[Z]=0\quad\Longrightarrow\quad \mathbb E[Y-cZ]=\mathbb E[Y],\qquad
\operatorname{Var}(Y-cZ)=\operatorname{Var}(Y)+c^2\operatorname{Var}(Z)-2c\operatorname{Cov}(Y,Z).
$$

控制变量保持期望，不保证降低方差。正相关还不够：系数与尺度也必须合适；若 Var(Z)>0，理想系数为 Cov(Y,Z)/Var(Z)。在线估计基线时还要保留其条件独立或可预测性要求。

先选目标，再写循环。回合提前结束时，最后 n−1 个待更新位置仍需补齐；它们不再自举。若是人为停止记录，世界却继续运行，应保留相应尾部预测。为终止补齐目标和因计算预算中断执行，是两种不同的边界。

19 状态随机游走是一个有解析答案的诊断任务，不是算法排行榜。其非终止位置为 1 至 19，从 10 开始，等概率左右移动；左端终止奖励 −1，右端终止奖励 +1，其他奖励零，折扣为 1。真实价值为 v(s)=s/10−1。比较前十回合后的误差，实际上同时考查传播速度、采样方差和有限步长。应联合扫描 n 与步长，不固定一种方法的最优步长去比较所有方法。

对持续智能体，等待更长不免费。除了缓存和反馈延迟，等待期间策略、表示和奖励规律也可能变化。先在冻结预测器上核对目标公式，再在同一固定策略经验流上比较学习器，最后才允许控制器改变经验流。三层实验分别检验代数、估计和闭环行为。

<a id="rlss-fixed-horizon"></a>

## 备份长度与预测时域不是同一个量

n-step TD 改变一次更新使用多少步真实奖励，通常仍估计同一个无限折扣价值。固定时域预测则改变要回答的问题：只问未来恰好 h 步的奖励。两个算法即使都出现“十步”，含义也可能完全不同。

$$
v_\pi^{[h]}(s)=\mathbb E_\pi\!\left[\sum_{k=0}^{h-1}\gamma^kR_{t+k+1}\mid S_t=s\right],\quad v_\pi^{[0]}(s)=0,\quad v_\pi^{[h]}(s)=\mathbb E_\pi[R_{t+1}+\gamma v_\pi^{[h-1]}(S_{t+1})\mid S_t=s].
$$

h 是所预测的时域，γ 是这一有限和中的奖励权重；有限时域也可以取 γ=1。时域 h 的头只从更短的 h−1 头自举。

$$
\delta_t^{[h]}=R_{t+1}+\gamma\widehat v_t^{[h-1]}(S_{t+1})-\widehat v_t^{[h]}(S_t),\qquad \theta_{t+1}^{[h]}=\theta_t^{[h]}+\alpha\delta_t^{[h]}\nabla\widehat v_t^{[h]}(S_t).
$$

各头采用同一个更新前快照可得到同步算法。原地循环从 h=1 向上写入，会让长时域头使用本次刚更新的短时域值，成为另一种时序。

常规 TD 的自举依赖可能回到自身；时域分层切断了这条目标依赖环。原论文分析了相应稳定性条件。但如果多个头共享可变的神经 encoder，更新一个头仍会改变其他头。目标图无环不等于参数更新图无环；有限数据、非凸优化与表示漂移仍需检查。

三次同步更新依次得到 [1,1,1]、[1,2,2]、[1,2,3]；这是依赖时序核验。

```python
# 单状态：每步奖励 1，gamma=1。一次样本同步更新三个时域头。
values = [0.0, 0.0, 0.0, 0.0]  # 第 0 头固定为 0
for _ in range(3):
    old = values.copy()
    for h in range(1, 4):
        values[h] = 1.0 + old[h - 1]  # alpha=1，使用旧快照
    print(values[1:])
assert values == [0.0, 1.0, 2.0, 3.0]
```

| 量 | 改变什么 | 不等同于 |
| --- | --- | --- |
| γ | 问题中的时间权重，或明确说明的算法参数 | 实际采了几步未来数据 |
| h | 有限预测范围 | 同一价值函数的备份长度 |
| n | 形成 target 前收集的实际步数 | 有限时域的终点 |
| λ | 混合不同备份长度或分配迹权重 | 环境终止概率的一般替代品 |

控制时也要保留剩余时域：最大化 h 步收益的第一步后，应使用剩余 h−1 步的策略。每一步都重新贪心优化 h 步目标，是滚动时域控制，不是同一个有限生命最优策略。持续任务里，两者都可以研究，但必须先说明评价目标。

<a id="multistep-control"></a>

## 3 · 动作价值与离策略校正

$$
G_{t:t+n}^{\rm Sarsa}=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^nQ(S_{t+n},A_{t+n})
$$

非终止边界还需要一个已选择的动作。若把最后一项改为目标策略加权平均，得到末端 Expected SARSA；这仍不同于在每一层做 Tree Backup。

在主例中，到达C后已有两个奖励1、2，且已选择finish。两步Sarsa目标为1+.9×2+.81×4=6.04。只把末端动作改为期望，C的尾值为.75×4+.25×0=3，目标变为5.23；B的fast仍是一个已经采样的行动，这个目标没有再平均B的detour。等到真正终止，普通三步Sarsa的完整样本回报也为1+.9×2+.81×3=5.23。这里两项数值恰好相等，是旧C期望等于这次终止奖励3造成的巧合，信息来源和等待时间仍不同。

$$
\rho_{t+1:\min(t+n,T-1)}=\prod_{k=t+1}^{\min(t+n,T-1)}\frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)}
$$

这是普通 n-step off-policy SARSA 对当前动作价值更新的一种比率约定。当前动作已被条件化，所以从 t+1 开始；非终止时采样 bootstrap 动作也需校正。

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\rho_{t+1:\min(t+n,T-1)}[G_{t:t+n}^{\rm Sarsa}-Q(S_t,A_t)]
$$

这里整个误差乘比率，不是仅让 G 乘比率。目标动作须被采样时的行为策略覆盖，并保存当时的行为概率；空乘积为 1。冻结目标策略和 Q 快照才能直接使用前面的期望解释。

手算：旧 Q=2，n-step target=5，α=.1，后续动作比乘积为 0，则这次更新仍为 2。错写成 α(ρG−Q) 会把它降到 1.8；该写法是另一种估计器，不能冒充这里的普通 SARSA 更新。

若末端直接对目标动作求期望，该末端动作没有被采样，不应再给它乘采样比率。完整轨迹权重、逐决策权重和带控制变量的目标还可以产生不同估计器。写代码前必须明确到底校正了哪一段随机选择。

<a id="tree-backup"></a>

## 4 · Tree Backup：保留未采样分支的预测

一条轨迹只真的走过动作 $A_{t+1}$。未走过的动作没有更深的实际奖励，仍使用当前 Q；走过的动作则用更深目标替代它原来的 Q。两类分支都按目标策略概率加权，就得到树备份。

![同一A到B到C轨迹，三种两步展开分别在C读取已选动作、在C求期望、在B与C逐层保留动作分支；目标为6.04、5.23与5.4325。](https://yingwen.io/crl-figures/control-targets-multistep.svg)

先预测：只在C求期望以后，B未执行的detour是否影响目标？三个面板使用同一旧表、奖励和冻结π=b。蓝实线表示已收奖励1、2的A→B→C前缀；Sarsa面板的C→finish蓝线仅表示已选动作，此时尚未收到第三奖励。绿色实线为C的末端期望分支，紫色虚线为B未执行detour的旧预测读取，都不表示额外执行动作。C叶端的Q=4、Q=0是旧动作价值，不是finish或quit的即时奖励。Tree Backup在B按.75保留detour的5、按.25继续fast的深目标4.7。末行说明收到finish奖励3并真正终止以后，普通Sarsa与树备份仍可能不同。原创精确计算，依据 Sutton 与 Barto §7.2、§7.5；[数据与逐项目标](/crl-figures/control-targets-data.json)。

从C向回算，末端期望为3，所以B的已采fast分支得到深目标 $2+.9\times3=4.7$。在B，未采detour仍读旧Q=5；A的两步树备份目标因而是 $1+.9[.75\times5+.25\times4.7]=5.4325$。若把B的detour旧Q改成50，两步普通Sarsa仍为6.04，树备份目标却增加；这直接检查了未走分支是否进入更新。

再收到finish的奖励3并终止，树备份在C把已采finish旧值4换成3，未采quit仍保留旧值0；B的深目标成为 $2+.9[.75\times3+.25\times0]=4.025$，A的三步树备份目标为 $1+.9[.75\times5+.25\times4.025]=5.280625$。一条轨迹到达终点只消除了这条实际路径的终点尾值；树备份中未走动作的自举仍在，因此展开到终止不一般等于MC回报。

用一个更短的终止情形检查相同替换规则：

![一条实际分支获得终止奖励6，另一分支没有新经验，仍读取旧Q值4；按目标动作概率加权后，树备份目标为5.05。](https://yingwen.io/crl-figures/concept-credit3-tree-backup.svg)

蓝色实线是实际轨迹，紫色虚线只表示读取未采样动作的旧预测，不代表查询了其后果。冻结 Q 与目标策略，γ=0.9。动作 0 的旧值 2 被更深回报 6 替换，动作 1 的旧值 4 保留。原创计算；[数值算例](/crl-code/figures/credit-knowledge-round3.mjs)。

先收到奖励 1，再到达有两个动作的状态。目标策略以 0.25、0.75 的概率选它们，旧预测为 2、4。这次实际选动作 0，得到奖励 6 并真正终止。因此树备份目标为 $1+0.9[0.25\times6+0.75\times4]=5.05$。完全沿本次轨迹计算的两步 Sarsa 目标则为 $1+0.9\times6=6.4$。它们使用同一经验，区别在于目标如何处理下一动作的选择。

也可以从旧期望 $0.25\times2+0.75\times4=3.5$ 出发，只补上已采样分支的变化 $0.25(6-2)=1$。这样得到 4.5，再乘折扣并加首步奖励，仍是 5.05。将动作 0 的旧 Q 改成任何数，这次替换后的目标都不变；未采样动作的旧 Q 却仍影响答案。

$$
G_{t:h}^{\rm TB}=R_{t+1}+\gamma\left[\sum_{a\ne A_{t+1}}\pi(a\mid S_{t+1})Q(S_{t+1},a)+\pi(A_{t+1}\mid S_{t+1})G_{t+1:h}^{\rm TB}\right]
$$

非终止边界设 $G_{h:h}=Q(S_h,A_h)$，最后一层因此退化为 Expected SARSA。到达真实终点时 G=最后奖励。

$$
G_{t:h}^{\rm TB}=R_{t+1}+\gamma\left[\bar V(S_{t+1})+\pi(A_{t+1}\mid S_{t+1})(G_{t+1:h}^{\rm TB}-Q(S_{t+1},A_{t+1}))\right]
$$

先把所有动作的旧预测相加，再只对实际分支补上“深目标减旧预测”。这里 V̄=ΣπQ；不需要枚举未走分支的环境后果。

Tree Backup 不依赖普通轨迹比率，因此可以处理不同于目标的行为分支，但这不等于没有数据覆盖要求。目标概率为零的采样分支不会把更深信息传回；很小概率也会衰减深层信用。

<a id="experiment-retrace"></a>

### 实验：实验 · 后继误差传播多远，取决于什么系数？

都保留三步、都使用目标策略的完整动作期望，Retrace 与 Tree Backup 的误差传播为何仍不同？

**环境与可用信息。** 六格终止链，末端奖励 1，其余 −0.01，γ=0.95。从 0 开始。目标与行为策略完全相同，每个状态向左 0.2、向右 0.8。十个状态动作估值全零；本实验只做固定策略评价。

**设置。** 五种子各 1200 个环境转移，步长 0.1。最多保存三条转移；三步成熟后更新最早状态动作，真实终止时依次处理余下短目标。预算用尽但未成熟的尾段不强制终止。每个目标的递推使用当时冻结的 Q。

**检验的机制。** 两者的 TD 残差都包含完整目标动作期望。Retrace 以 λ min(1,ρ) 传播后续残差；这里 λ=0.8 且 ρ=1，因此系数为 0.8。Tree Backup 的对应系数是采样动作的目标概率：向右 0.8，向左 0.2。两者并非仅仅换了名字。

**测量。** 纵轴是五个非终止状态、两个动作的均方根价值误差，真值由固定策略的精确模型解产生。不是贪心回报，也不是对最优 Q 的误差。

```bash
python3 implementations/extended_classic/retrace.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/retrace/curves.svg)

横轴：environment_steps。纵轴：固定策略十状态动作价值 RMSE。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 第 600 步，Retrace 的平均 RMSE 为 0.1413，Tree Backup 为 0.1932；末尾为 0.0299 对 0.0579。该同策略设置下，更长保留的左动作后续误差与较快误差下降相伴，但没有单独改变一个系数的因果消融。

**结论边界。** 当前比率恒为 1，未实际检验离策略比率裁截的主要价值。三步有限缓冲也不等于无限前向回报或无缓冲流式迹；不能将结果写成 Retrace 离策略稳定性实验。

**继续实验。** 选一段先左后右的三步轨迹，逐项手算两种目标。若改用不同的行为策略，必须在采样和概率比两处同步修改，并保持评价目标不变，随后再研究比率方差。

[源码](https://yingwen.io/crl-code/implementations/extended_classic/retrace.py) · [逐种子记录](https://yingwen.io/crl-code/results/retrace/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/retrace/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/retrace/curves.json)

<a id="q-sigma"></a>

## 5 · Q(σ)：逐层混合采样与期望

先固定 Q 快照并假定 on-policy。本节与配套源码采用 De Asis 等 AAAI 2018 原论文式 (13)–(14) 的混合误差版本：每一层用 σ 权重沿采样分支完全继续，用 1−σ 权重执行树备份。σ=1 逐样本等于普通 n-step SARSA，σ=0 等于 Tree Backup；中间值在每一层混合，而不只是把最终两个目标平均。

$$
G_{t:h}^{\sigma}=R_{t+1}+\gamma\left\{\sigma G_{t+1:h}^{\sigma}+(1-\sigma)\left[\bar V(S_{t+1})+\pi(A_{t+1}\mid S_{t+1})(G_{t+1:h}^{\sigma}-Q(S_{t+1},A_{t+1}))\right]\right\}
$$

边界与树备份相同。把括号按 G−Q 收集，系数变成 σ+(1−σ)π，这正是源码的 branch。

$$
\delta_t^\sigma=R_{t+1}+\gamma[\sigma Q(S_{t+1},A_{t+1})+(1-\sigma)\bar V(S_{t+1})]-Q(S_t,A_t)
$$

这是单层混合误差。在冻结 Q 下，深层误差还乘逐层 γ[σ+(1−σ)π] 的乘积；σ与 n 分别控制分支混合和展开长度。

版本核对：2020 版 Sutton 与 Barto §7.6 式 (7.17) 改用带控制变量的递归。下面只列它的 on-policy 形式（ρ=1）；一般离策略形式把 σ 改为 σρ。两者不能共用“σ=1 的单条轨迹 target 完全相同”这一测试。

$$
G_{t:h}^{\sigma,\mathrm{CV}}=R_{t+1}+\gamma\left\{\bar V(S_{t+1})+[\sigma+(1-\sigma)\pi(A_{t+1}\mid S_{t+1})]\,[G_{t+1:h}^{\sigma,\mathrm{CV}}-Q(S_{t+1},A_{t+1})]\right\}
$$

边界仍为 $G_{h:h}=Q(S_h,A_h)$，真实终止保留最后奖励。$\sigma=0$ 两版本相同；$\sigma=1$ 本式是带控制变量的 SARSA 回报，不是普通采样回报。

只比较一层且给定同一个深目标 g 时，两式相差 $\gamma\sigma[\bar V-Q(S_{t+1},A_{t+1})]$。给定动作前信息，on-policy、冻结 Q 下该项条件期望为零；这说明控制变量的作用，不说明两种样本值或方差相同。例：γ=.9，下一状态 Q=(2,4)、π=(.25,.75)，采到动作 0，g=3，当前奖励 1。σ=1 的普通回报为 3.7，书中控制变量回报为 5.05。

论文还给出 off-policy Q(σ)，需要相应的重要性修正。这里的统一源码只实现 on-policy、冻结快照的 target，不包括一般 off-policy 训练循环。动态 σ、在线变参和资格迹版本需要分别定义时序。

回到A/B/C主例，令两层σ都为.5。论文混合误差版本先在C读采样尾值4与期望尾值3的平均，B的深目标为5.15；再在B混合完整采样分支和树分支，A的目标为5.584375。书中控制变量版本的目标则为6.68125。配套Python以冻结TD误差展开计算、JS以递归计算，分别核对两个数及σ两端；版本之间的样本差异不被当成实现错误。主例π=b、σ固定，未实现一般离策略或会在线变化的策略。

<a id="multistep-algorithm"></a>

## 6 · 完整 n-step 执行顺序

**算法：算法伪代码**

1. 设 n≥1；初始化价值表与长度 n+1 的缓存。
1. 每次 episode 记录初始状态，暂令终点时间未知。
1. 每次时钟 t：
  1. 尚未终止时，执行动作并保存下一状态和奖励。
  1. 若到真实终点，记录 T；以后不再调用环境。
  1. 令待更新时刻 τ=t−n+1。
  1. 若 τ≥0，累积 τ 到 min(τ+n,T) 的奖励。
    1. 若 τ+n<T，接上当前表的边界预测；否则无 bootstrap。
    1. 更新 τ 对应的状态。
  1. 直到 τ=T−1，才完成本 episode。

实验为便于检查传入完整终止轨迹，但更新仍按上述延迟和补齐顺序执行。它不读取尚未到当前时钟的奖励。运行时若改为环形缓存，索引取模必须同时用于状态与奖励；真实终止后不能再选择一个不存在的动作。

![两步Sarsa沿三次实际转移等待、更新A、更新B，终止后再补齐C；标出旧Q版本和截断C时必须保留的尾值。](https://yingwen.io/crl-figures/control-targets-timing.svg)

先预测：到达T后是否还有未处理表项？蓝色到达钟计真实转移，最后一列只做计算、不调用环境。α=.25；A在钟2读旧C值4，目标6.04并更新到3.01；B在钟3更新到1.925；C在钟4补齐到3.75。下部比较非终止C的记录截断、真正终点T，以及晚读取C新值造成的目标差异。原创确定性时序，依据 Sutton 与 Barto §7.1–7.2；[独立执行脚本](/crl-code/tutorials/control_targets_walkthrough.py)。

按 $n=2,\alpha=.25$ 在线处理主例：到达B时还少一个奖励，只保存数据；到达C时先选择并保存finish，尚未用终止奖励更新C，因此读取的C值仍为4，A从2更新到 $2+.25(6.04-2)=3.01$。下一转移在T真正终止，B的剩余目标为 $2+.9\times3=4.7$，B的fast从1变为1.925。终止后继续补齐C，目标为3，C的finish从4变为3.75。这次计算不再执行环境，也不选择终点动作。

图中$Q^{(0)}$、$Q^{(1)}$、$Q^{(2)}$按已完成的表项更新计版本，分别用于A、B、C的目标。若等到回合结束后先更新C，再回头用3.75构造A的原两步目标，会得到1+.9×2+.81×3.75=5.8375，改变了此前6.04的目标与更新时序。主例状态不重复，先前A的更新不会改变后来B、C所读的表项；循环轨迹或共享参数则更需要逐项保存版本，不能直接沿用冻结恒等式。

若采样预算在C暂停，任务还没结束。两步Sarsa依然需要边界动作finish的旧估计4，得到6.04；误把暂停记成终止会只留下1+.9×2=2.8，少了3.24。若暂停接口没有选择边界动作，可以按声明选择一次目标动作，或采用末端期望尾值3，但不能无声地读取不存在的动作索引。真正到T时尾值才为零。这里比较的是边界契约，不把预算暂停改写为另一个吸收状态。

<a id="lesson-example"></a>

## 7 · 手算与可运行源码

![五步链首回合实际运行 n=1、3、5 的多步更新与 λ=0.8 的累积资格迹，列出各状态更新后的值并推算末步的信用。](https://yingwen.io/crl-figures/learning-classic-credit.svg)

原创实际更新：与 TD/MC 图相同的链，$\gamma=0.9$、$\alpha=0.25$、零初值；各方法各运行 1 回合/5 次转移，n-step 在终点补齐待更新目标。无随机、无需种子。$\lambda$ 行使用普通表格累积迹，不冒充一般 true-online 等价。

读表时先看起点 $S_0$，而不是先挑最大的数字。$n=3$ 的目标只含前三个零奖励和当时仍为 0 的 $S_3$ 尾值，因此起点仍是 0；从 $S_2$ 开始却能在三步内看到最终奖励，更新为 $0.25\times0.9^2=0.2025$。$n=5$ 在终点已看完整回报，所以起点更新为 0.164025。等待长度决定这一次目标中实际包含哪些后果。

表中 $\lambda$ 行从另一方向分配信用：每次访问在当前状态加一份资格，旧资格乘 $\gamma\lambda=0.72$；最后误差为 1 时，起点还留下 $0.72^4$ 份，所以立即更新 $0.25\times0.72^4=0.06718464$。资格迹没有提前知道奖励，它让已经到来的奖励找到过去参与预测的状态。完整推导和重复状态反例见函数逼近分册的资格迹章。

两步轨迹奖励为 1、3，γ=.9；下一状态 Q=(2,4)，目标概率 (.25,.75)，采样第一动作后终止。SARSA 的完整目标是 1+.9×3=3.7。Tree Backup 保留未采样动作：1+.9×(.75×4+.25×3)=4.375。σ=.5 得 4.0375。

随机游走在 1…5 五个非终止状态间等概率左右移动，从 3 开始；到 0 得零并结束，到 6 得一并结束，γ=1。真实价值 v(s)=s/6，由线性差分方程与边界值得到。默认 5000 回合、α=.02，n=1、3、8 的最终 RMSE 约 .01805、.02188、.02688；单次排序不是方法优劣结论。

<a id="lesson-code"></a>

## 8 · 边界与退化条件检查

主例的独立标准库脚本为[control_targets_walkthrough.py](/crl-code/tutorials/control_targets_walkthrough.py)。运行 `python3 control_targets_walkthrough.py --test` 后，不带参数运行可读取一步目标、两步/三步分支、n=2与n=3的延迟更新、截断差额及参数版本。n超过回合长度的检查仍要求每个已访问表项恰好完成一次更新，且只消费已提供的三条转移记录，不取得新环境数据。网站源码中的 `node scripts/generate-crl-control-targets-walkthrough.mjs --check` 只核对这组三组桌面/手机SVG与数据，没有启动训练。

延迟 n-step、终点补齐、冻结 Q(σ) target 与五状态随机游走的实际学习。

```python
def nstep_episode(states, rewards, values, n, alpha=0.1, gamma=1.0):
    """Terminal trajectory, online update order and terminal flush.
    Full trajectory is supplied for tests; live code needs only an n+1 ring buffer.
    """
    if n<1 or len(states)!=len(rewards)+1 or states[-1] is not None:
        raise ValueError("need n>=1 and true-terminal trajectory")
    terminal, values = len(rewards), values.copy()
    for t in range(terminal+n-1):
        tau = t-n+1
        if tau<0:
            continue
        end = min(tau+n, terminal)
        target = sum(gamma**(k-tau)*rewards[k] for k in range(tau, end))
        if tau+n<terminal:
            target += gamma**n*values[states[tau+n]]
        s = states[tau]
        values[s] += alpha*(target-values[s])
    return values

def qsigma_target(states, actions, rewards, q, policy, sigma, gamma=0.9):
    """Frozen-Q, ON-POLICY finite-horizon Q(sigma) target.
    A nonterminal endpoint requires its sampled action. No general off-policy
    importance correction is implemented. sigma=0: tree backup; sigma=1: Sarsa.
    """
    if not 0<=sigma<=1 or len(states)!=len(rewards)+1:
        raise ValueError("invalid sigma or trajectory")
    horizon = len(rewards)
    g = 0. if states[-1] is None else q[states[-1]][actions[-1]]
    for k in reversed(range(horizon)):
        sp = states[k+1]
        if sp is None:
            if k!=horizon-1:
                raise ValueError("terminal must be last")
            g = rewards[k]
        else:
            ap = actions[k+1]
            expected = sum(p*v for p, v in zip(policy[sp], q[sp]))
            branch = sigma+(1.-sigma)*policy[sp][ap]
            g = rewards[k]+gamma*(branch*(g-q[sp][ap])
                +sigma*q[sp][ap]+(1.-sigma)*expected)
    return g

def random_walk(n, episodes=5000, seed=7):
    rng = random.Random(seed)
    values = {s: .5 for s in range(1,6)}
    for _ in range(episodes):
        state, states, rewards = 3, [3], []
        while state not in (0,6):
            sp = state+rng.choice([-1,1])
            rewards.append(float(sp==6))
            states.append(None if sp in (0,6) else sp)
            state = sp
        values = nstep_episode(states, rewards, values, n, alpha=.02)
    return {"values": values, "reference": {s:s/6 for s in values},
            "rmse": math.sqrt(sum((values[s]-s/6)**2 for s in values)/5)}
```

源码把两类实验分开：随机游走会实际采样经验并执行 n-step 学习；Q(σ) 函数则接收冻结表与一条指定轨迹，只计算 target。对后者应先检查 σ=0、σ=1、单步、确定性策略与真实终止五种边界，再考虑接入会变化的在线价值表。通过 target 检验并不等于完整训练器已经实现。

改变随机游走的起点，会改变各状态的访问频率，却不改变固定左右策略的真实价值。分别输出逐状态误差和均方根误差，可以看到总指标掩盖的覆盖差异。若改变左右动作概率，则策略本身改变，原来的 s/6 不再是正确参照，应先重新解 Bellman 方程，不能继续使用旧解析答案。

<a id="lesson-branches"></a>

## 9 · 与资格迹、流式协议和 CRL 的关系

n-step 固定一个反馈长度；λ-return 组合多个长度；资格迹把过去方向压缩为递推状态。资格迹不是过去观测的记忆，因此不自动解决部分可观测问题。普通在线迹与冻结快照下的误差展开也不应混同。

持续流没有终点时，n-step 可以在缓存满后持续更新，但最末暂停位置只是截断。长期变化使等待 n 步期间策略或环境改变，较长目标可能收集更多实际结果，也可能包含更多旧情境。应同时研究传播速度、偏差、方差与每步存储。

<a id="lesson-check"></a>

## 10 · 带答案练习

问：n 大于 episode 长度，最前几个状态会永远不更新吗？答：不会，终止后要继续补齐，所有目标都退化为对应完整回报。立即清空缓存是错误。

问：Tree Backup 是否生成了未访问动作的新奖励？答：没有。未走分支使用已有 Q，只有实际分支使用更深的经验；“树”是目标展开而非实际并行采样。

问：目标策略在采样分支概率为一时，σ两端还不同吗？答：该层树备份只剩采样分支，与 SARSA 相同。代码测试这个退化条件。

问：两步末端Expected与三步普通Sarsa在主例都等于5.23，是否已经使用同一信息？答：前者到C时用旧表期望3代替未来，后者到T后才读到实际奖励3；相等来自这个算例的数值巧合。把C的finish旧值改为8，前者改变，完整终止回报仍为5.23。



<a id="chapter-code"></a>

## 下载与运行

Python 3，仅标准库。下载完整文件后在其目录运行；页面展示核心区域。小环境和解析检验用于理解机制，不是论文基准复现。

[下载 tabular_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/tabular_textbook_lab.py)

```sh
python3 tabular_textbook_lab.py multistep
python3 tabular_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：对应第 2–8 章。以下推导、例子和标准库实现独立编写；原书用于核对定义、算法关系与适用条件。

- [David Silver · UCL 强化学习课程](https://davidstarsilver.wordpress.com/teaching/)：原课程页面提供 MDP、动态规划、预测、控制、探索和规划的讲义及视频。

- [De Asis 等 · Multi-Step Reinforcement Learning: A Unifying Algorithm](https://arxiv.org/abs/1703.01327)：原始 Q(σ) 论文；区分采样程度、展开长度与 on/off-policy 修正。

- [De Asis et al. · Fixed-Horizon Temporal Difference Methods for Stable Reinforcement Learning](https://ojs.aaai.org/index.php/AAAI/article/view/5784)：AAAI 2020；时域分层的价值预测及其稳定性。本文同步三头算例独立编写，不是论文实验复现。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-tabular-multistep#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-tabular-multistep#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-tabular-multistep)
<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 怎样把较晚的反馈归给较早的计算？

多步回报定义用多远的未来构造目标。资格迹压缩过去的特征或梯度方向。前向与后向等价必须说明参数是在整个轨迹内固定，还是每一步改变。

函数逼近与深度方法：神经网络改变后，旧梯度不再等于用当前参数重算的梯度。递归状态还带来参数经过历史状态影响当前输出的路径，不能用一条普通 TD trace 代替。

持续学习中的研究问题：在每步计算有界的条件下，保留多少过去影响才有用？替换特征时，怎样处理与旧特征绑定的资格迹、优化器动量和元梯度？

[多步回报](https://yingwen.io/zh/continual-rl/foundations/tabular/multistep/) → [资格迹与等价条件](https://yingwen.io/zh/continual-rl/foundations/approximation/traces/) → [GAE 与 actor–critic](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/) → [在线信用分配](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)


### 可进一步检验的问题

- [05 · 远处的反馈应该怎样更新早先的决策与内部计算？](https://yingwen.io/zh/continual-rl/research/#research-temporal-credit)：多步目标明确等待多少真实奖励、从哪里自举，为研究有限等待和计算下的延迟信用建立参照。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)
- [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)
- [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)

对应原始材料：7.1–7.6。本文为原创讲解，原书、论文与上游代码保留各自许可。
