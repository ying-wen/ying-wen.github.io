# 多臂老虎机：估计、探索与直接策略学习

表格强化学习 · 第 1 章

没有状态转移时，仍需一边估计动作收益，一边决定下一次尝试什么。这个最小问题把估计误差、探索代价和策略更新分开。

## 本章内容

- 推导样本平均和常数步长，解释历史奖励的权重。
- 区分 ε-greedy、乐观初值、UCB 与 gradient bandit 改变的量。
- 运行逐步反馈与随机奖励采样，区分机制检查与有限样本性能。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 概率与期望

概率非负且总和为一；期望是按概率加权的平均，不是单次结果。条件期望只比较满足已知条件的情形。

### 递推与赋值

递推通过已有估计计算新估计。下标表示时间；更新符号表示程序中的赋值，不是数学恒等式。

<a id="problem-definition"></a>

## 本章的问题定义

学习器反复从有限动作中选择一个动作，只观察该动作的奖励。动作不改变下一轮的外部状态。

### 给定条件与符号

- 动作集合 $\mathcal A=\{1,\ldots,K\}$、交互预算 $T$。
- 每个动作有未知奖励分布 $\nu_a$，均值 $q_*(a)$；$q_*^{\max}=\max_aq_*(a)$。

### 需要求解的对象

因果选择规则：第 $t$ 轮只能使用此前的动作与奖励，决定 $A_t$。

### 信息与数据权限

没有全部动作奖励反馈，也不知道哪个动作最优。观测一个动作的奖励不等于获知其他动作的均值。

$$
\operatorname{Regret}_T=\mathbb E\!\left[\sum_{t=0}^{T-1}(q_*^{\max}-q_*(A_t))\right]
$$

在固定环境中减少累计伪遗憾，等价于增加同一预算下的期望累计奖励。动作均值的估计只是实现这一目标的中间问题。

### 成立条件与解的含义

- 标准随机 bandit 假设每个动作的奖励按固定分布独立采样，或至少满足给定历史及所选动作后的条件均值为该动作均值；仅有平稳边际分布不足。置信界另需有界或适当尾部条件。
- 常数步长跟踪漂移时，均值随时间变化；此时需要重新定义比较器。

判断准则：报告累计收益或遗憾及探索次数；不能仅用均值估计误差评价选择规则。

### 适用边界

- 保证在未知分布下每一步选择都最优。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：本章去掉动作对未来状态的影响，仍保留学习与探索的权衡。

- 特例：增加条件 · [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)：本章将探索限定为无上下文、独立拉臂；不包含序列探索中到达信息状态的成本。这是过程条件的限制，不只是反馈协议变化。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

只利用当前估计可能错过好动作；试验其他动作又会消耗当前收益。

### 本章的核心思路

将均值估计与动作选择分开，再用随机探索、置信上界或直接策略梯度分配试验。

1. [先得到可增量维护的估计](#lesson-derive)：样本平均把历史奖励压缩为计数和均值；常数步长改变了旧经验的权重。

2. [用不确定性决定是否试验](#bandit-exploration)：比较 ε-greedy 的无差别探索与 UCB 的访问次数校正。

3. [不估动作值，也可以改策略](#bandit-gradient)：偏好参数通过奖励与基线形成 score-function 更新；这与置信界选择是不同机制。

结论与条件：样本平均的一致性依赖足够采样；UCB 遗憾界还依赖具体分布与探索系数，不能直接用于任意漂移。

### 相关方法改变了什么

- 样本平均 / 常数步长：前者累计全部数据，后者遗忘旧数据。改变的是估计器，不是自动解决探索。

- UCB / Gradient bandit：前者以不确定性修正动作分数，后者直接更新随机策略。


<a id="lesson-setting"></a>

## 1 · 问题与符号

先考虑一个反复选择的问题：机器人每次从三种抓取方式中选一种，观察是否成功。实验暂把物体和位置恢复到相同条件，让这次选择不改变下一次面临的外部情形。即使如此，机器人仍不知道哪种方式最好，而且只看得到自己实际选择的结果。开始于这个简化问题，是为了单独研究一个困难：学习需要尝试，尝试却可能牺牲当前收益。

有 $k$ 个动作。第 $t$ 次选择 $A_t$ 后，只观察它的奖励 $R_t$，看不到未选动作本次会得到什么。先假定每个动作的分布固定，各次奖励在给定动作后独立且均值有限。真实动作价值为 $q_*(a)=\mathbb E[R_t\mid A_t=a]$，算法保存估计 $Q_t(a)$。

![三个动作 A、B、C，本次只选择 B 并观察奖励 1。B 的计数从 4 变为 5、样本均值从 0.5 变为 0.6；A 和 C 的结果仍为问号。](https://yingwen.io/crl-figures/concept-bandit-observation.svg)

沿用三种动作，把已经观察到的 B 臂奖励固定为 1、0、1、0。图中第五次又选择 B 并得到 1。蓝色圆是收到的结果，问号是没有观察到的结果；A、C 的零估计仅是初始化。数值为本章独立编写的确定性算例。

在揭示第五次结果之前，先问自己：如果这次 B 成功，A 和 C 的成功率估计也应当增加吗？表格型 bandit 为每个动作单独保存证据，因此只更新 B。后面采用共享特征时，一个动作的更新可能同时改变其他动作的估计；届时必须说明这种推广利用了什么结构。

目标是在交互预算内获得奖励，不只是最后估准全部动作。总选当前估计最大的动作称为利用；为了获得可能改变排序的信息而尝试其他动作称为探索。低奖励可能意味着动作较差，也可能只是噪声，所以估计与决策相互影响。

$$
a_*\in\arg\max_aq_*(a),\qquad \mathcal R_T=\sum_{t=1}^{T}[q_*(a_*)-q_*(A_t)]
$$

伪遗憾累计实际所选动作与最佳固定动作的均值差。当前动作不改变明天有哪些选择，这是 bandit 与一般序列决策的关键区别。

<a id="lesson-derive"></a>

## 2 · 样本平均的增量形式

某动作已观察 $n-1$ 个奖励，均值为 $Q_n$。得到第 $n$ 个奖励后，把旧奖励之和写成 $(n-1)Q_n$，便可不保存历史而求新均值。这里 n 是这个动作的访问次数，不是总交互步数。

$$
Q_{n+1}=\frac{(n-1)Q_n+R_n}{n}=Q_n+\frac1n(R_n-Q_n)
$$

误差是新观测减旧估计。步长 1/n 给每个历史样本相同权重；第一次观测时初值被消除。

$$
Q_{n+1}=Q_n+\alpha(R_n-Q_n)=(1-\alpha)^nQ_1+\sum_{i=1}^{n}\alpha(1-\alpha)^{n-i}R_i
$$

取 $0<\alpha\le1$，这些非负权重之和为1。$\alpha<1$ 时旧奖励的权重指数衰减；$\alpha=1$ 时只保留最新奖励。较大步长更快追踪变化，也更容易跟随噪声。

无论采用哪种估计，只有被选动作获得新信息。一个从不访问的动作不会因为其他动作被估得更准而自动改善。持续学习中的适应慢，可能来自步长过小，也可能来自没有访问已经改变的选择。

样本平均的收敛讨论针对固定奖励分布和不断增加的访问次数。常数步长通常不消除随机波动，而是在误差与跟踪速度之间折中。若只在总时间上减小步长，一个很晚才开始被尝试的动作可能第一份数据就几乎不起作用；因此必须区分全局时钟与该动作自己的计数。

<a id="experiment-bandit_constant_step"></a>

### 实验：实验 · 固定步长与样本平均，怎样利用同一条奖励流？

更重视新样本的估计器，是否在平稳问题中也一定获得更多奖励？

**环境与可用信息。** 三个独立 Bernoulli 臂，成功概率为 0.2、0.5、0.8。每步只看到所选臂的 0 或 1 奖励。均值始终不变，没有状态转移，也没有延迟回报。

**设置。** 种子 0–4，各交互 1200 步。估值和访问计数从零开始，两者都用 ε=0.1 的 ε-greedy，最大值并列时均匀选取。实验固定步长为 0.1；对照对该臂使用访问次数的倒数。每次选择后只更新所选臂。

**检验的机制。** 固定步长给旧奖励指数衰减的权重，因此始终能追踪变化，也始终保留采样波动。样本平均让单个新奖励的作用随访问次数减小。当前任务没有漂移，因此它并未提供固定步长的主要用武之地。

**测量。** 纵轴是从第一步到当前为止的实际累计平均奖励，不是当前贪心动作的价值，也不是估值 MSE。动作序列会因学习方法不同而分岔；相同 seed 不表示两者看到了相同的动作奖励对。

```bash
python3 implementations/classic/bandit_constant_step.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-figures/result-bandit_constant_step.svg)

横轴：environment_steps。纵轴：累计平均奖励。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 行为策略实际收到的外部奖励。value 是从第1步至当前 step 的奖励总和除以 step，已经包含这段交互的探索代价。

**step：怎样计时。** step 是实际拉臂次数；每次选择只给所选臂增加一个样本。UCB 的上界项只参与动作选择，不加入环境奖励。梯度 bandit 每次用旧策略概率和收到本次奖励前的基线更新全部偏好，再更新奖励基线。一次环境步不等于只改一个参数。

**怎样汇总。** 每条记录已是累计均值。末点给全程平均奖励，不能再把所有记录点的 value 平均当成全程平均。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** 末点 value × step 可恢复该运行的累计奖励；两个检查点的累计量相减，可恢复这段区间的奖励总和。稀疏检查点不能恢复区间内每一步奖励、访问计数、动作价值或策略概率。各臂真实均值固定为0.2、0.5、0.8；这些曲线不测量变化后的适应。

计算位置：[classic/bandit_sample_average.py](https://yingwen.io/crl-code/implementations/classic/bandit_sample_average.py) · [classic/bandit_constant_step.py](https://yingwen.io/crl-code/implementations/classic/bandit_constant_step.py) · [classic/bandit_ucb.py](https://yingwen.io/crl-code/implementations/classic/bandit_ucb.py) · [classic/bandit_gradient.py](https://yingwen.io/crl-code/implementations/classic/bandit_gradient.py) · [classic/_common.py](https://yingwen.io/crl-code/implementations/classic/_common.py)

</details>

**结果分析。** 第 1200 步，固定步长的五种子平均累计奖励为 0.7392，样本平均为 0.7628。两者都明显偏向高收益臂，但这五次运行没有显示固定步长优势。累计指标仍包含早期探索的成本。

**结论边界。** 这里只比较一个 ε 和一个固定步长。五种子并非算法排序的充分证据。平稳三臂任务不能支持非平稳追踪结论；日志也未记录每个臂的选择频率。

**继续实验。** 先补记臂选择频率和各臂估值，再在预定时刻交换最优臂。保留原来的平稳实验。分别报告全程奖励与变化后的窗口奖励，检查累计平均是否掩盖了适应延迟。

[源码](https://yingwen.io/crl-code/implementations/classic/bandit_constant_step.py) · [逐种子记录](https://yingwen.io/crl-code/results/bandit_constant_step/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/bandit_constant_step/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/bandit_constant_step/curves.json)

<a id="bandit-tracking"></a>

## 同一动作变化后，旧经验应该占多少分量？

继续看动作 B。先固定访问日程，让两个估计器读取完全相同的八次奖励：1、0、1、0、1、1、1、1。前四次是均值为 0.5 的 Bernoulli 奖励可能产生的一段样本；第五次访问前，B 的均值变成 1，所以后四次必得 1。A、C 暂不访问。这样可以先隔离“收到信息以后怎样更新”，随后再研究“何时选择能带来新信息的动作”。

两个估计器都从 $Q_1=0$ 开始。前四次之后，样本平均为 $2/4=0.5$；$\alpha=0.5$ 的常数步长依次得到 $0.5,0.25,0.625,0.3125$。收到第五次奖励 1 时，前者变为 $0.5+(1-0.5)/5=0.6$；后者变为 $0.3125+0.5(1-0.3125)=0.65625$。它们读到了同一证据，却给旧证据不同的分量。

![八次给定奖励的样本均值与常数步长历史权重，以及随 B 访问次数变化的估计曲线；真实均值在第五次访问前由 0.5 变为 1。](https://yingwen.io/crl-figures/concept-bandit-tracking.svg)

上方柱高表示第八次更新后各奖励及初值的权重；下方使用相同输入逐次计算。$\alpha=0.5$，初值为 0；最后的两项估计为 0.75 与 0.95703125。灰虚线给读者显示均值变化，算法只读取奖励。横轴为 B 的访问数；未运行行为策略或随机训练。

八次之后，样本平均为 $6/8=0.75$，最早和最新奖励都占 $1/8$。常数步长给最新奖励 $1/2$，前一次 $1/4$，再前一次 $1/8$；初值还占 $1/256$。因此最后四个奖励 1 占总权重 $15/16$，得到 $0.3125/16+15/16=0.95703125$。初值的权重虽然乘着零，仍须计算在总权重中。

这个差异并不局限于八步。假设一个臂先收到了 $m$ 个零，之后奖励恒为 1。变化后再访问 $j$ 次，样本平均是 $j/(m+j)$；初值和变化前估计都为零、常数步长为 $\alpha$ 时，估计是 $1-(1-\alpha)^j$。取 $m=100,\alpha=0.1$，达到 0.9 分别需要 900 次与 22 次新访问。第一种更新仍在回答全部历史的平均，第二种更新更强调近期。

$$
\operatorname{Var}(Q_{n+1})=(1-\alpha)^2\operatorname{Var}(Q_n)+\alpha^2\sigma^2\quad\Longrightarrow\quad \lim_{n\to\infty}\operatorname{Var}(Q_n)=\frac{\alpha}{2-\alpha}\sigma^2.
$$

若单臂奖励独立同分布、方差为 $\sigma^2$，且初值确定，则新奖励与旧估计独立，方差递推没有交叉协方差。取 $0<\alpha\le1$，极限由固定点求得。固定访问日程下，样本均值的方差是 $\sigma^2/n$；常数步长通常保留波动。这里的 $n$ 是固定的访问次数。

更大的步长能够缩短对变化的反应，也让随机波动更直接地进入估计。因此不能只给一个变化点就宣布某个步长最好。可以先固定奖励噪声改变变化速率，再固定变化过程改变噪声。若算法很少再访问已经改变的臂，任何步长都无法利用尚未到来的证据。

<a id="bandit-exploration"></a>

## 3 · 从估计到探索分布

前面的八次更新规定了何时访问 B。现在把选择权交回算法：尚未访问的 C 可能比 B 更好，学习器却没有证据。保留同一个三臂抓取接口，暂时撤掉前例的均值变化，让 A、B、C 的奖励成为独立 Bernoulli 结果，固定成功率为 (0.2,0.5,0.8)。这些均值供读者核对；选择器只能使用实际收到的奖励。先保持估计更新不变，问不同选择规则怎样获得下一份证据。

$$
\pi_t(a)=\frac{\epsilon}{k}+(1-\epsilon)\mathbf1\{a=g_t\},\qquad g_t\in\arg\max_bQ_t(b)
$$

先用固定规则选择一个贪心动作。探索分支在所有动作中均匀采样，因此贪心动作也可以被探索分支选到。

假设三臂的当前估计排错了序：估计最高的臂，真实均值反而最低。探索分支怎样给其他动作留下机会？沿图中的两条分支相加，就能算出最终动作概率。

![三臂动作机器与两条选择分支：利用分支指向动作零，均匀探索仍可选择全部三臂；一次第三臂样本只更新该臂。](https://yingwen.io/crl-figures/concept-classic-bandit-exploration.svg)

原创精确概率图。给定估计 $(0.6,0.5,0.4)$、真实均值 $(0.2,0.5,0.8)$、$\epsilon=0.1$，得到概率 $(14/15,1/30,1/30)$；真实均值供读者核对，算法不读取它。下方是一种可能的奖励样本，$\alpha=0.1$ 使 $Q(2)$ 从 0.4 变为 0.46。不是本章 4000 步实验的中途快照。

固定 ε 保留长期探索，但稳定环境中仍会付出选择较差动作的代价。让 ε 随时间减小可以减少代价，却不能快到某些动作仅访问有限次。最终贪心与充分访问是两个要分别检查的条件。

<a id="bandit-optimism"></a>

## 3.1 · 乐观初值怎样让贪心改选动作？

保留完全贪心的选择，也可以先给尚不了解的动作较高估计。取 $Q_1(a)=1.5$，超过本任务奖励上界 1；取常数步长 $\alpha=0.5$。并列时选字母在前的动作。第一次选 A 得到 0，$Q(A)$ 降到 0.75，B、C 仍为 1.5，于是下一次改选 B。若 B 得到 1，它的估计变为 1.25，下一次便选还保持 1.5 的 C。下一动作确实由上一份证据改变。

![同一奖励前缀下，零初值的贪心连续六次选 A；乐观初值的贪心依次选 A、B、C、B、C、C。下方用同一尺度画六次之后的估计。](https://yingwen.io/crl-figures/bandit-exploration-cold-start.svg)

原创精确反馈算例。两者均用 $\alpha=0.5$ 与字母先后的并列规则。零初值轨迹收到六个 0；乐观轨迹收到 0、1、1、0、1、0。脚本按各臂预先给定的访问前缀返回结果，算法只看到被选臂的结果；这是本任务中可能出现的有限经验，用于追踪初始化怎样改变行动。

零初值也采用同样更新，但前六次 A 的奖励都为 0，三个估计始终并列，固定并列规则使它留在 A。乐观轨迹六次后的估计为 $(0.75,0.625,0.5625)$，下一次又会选 A。这里探索来自估计中的初值分量：某臂访问 n 次后，它仍占 $(1-\alpha)^nQ_1(a)$。它会衰减，且会与噪声共同影响排序；一次成功也不保证算法就认出了最佳臂。

样本平均在某臂第一次访问后完全消除该臂的初值；常数步长逐次衰减初值的影响。两者都可以在开始时诱导探索，但任务后来变化时不会自动重新生成乐观初值。因此乐观初始化主要是平稳任务的启动手段；要持续获得新证据，还需持续的访问机制。ε-greedy 则在每一步重新提供探索机会，且不要求把 Q 改成乐观数值。

<a id="bandit-ucb"></a>

## 3.2 · UCB 把估计与探索项分开

ε-greedy 的探索分支把机会均分给全部动作。UCB 希望把较多机会给“估计尚不精确、仍可能最好”的动作。它保留样本平均 $Q_t(a)$，另外计算选择分数。令 $N_t(a)$ 为时刻 t 之前的访问数，因此 $\sum_aN_t(a)=t-1$。

$$
U_t(a)=Q_t(a)+c\sqrt{\frac{\log t}{N_t(a)}},\qquad A_t\in\arg\max_aU_t(a)
$$

探索项只用于选择，不写回 Q，也不作为环境奖励。对 N=0 的动作先行尝试；本页按字母先后处理多个未访问动作，避免除零。全部访问后才使用此公式。

本任务奖励在 $[0,1]$ 内、各臂分布固定。对某臂预先固定的 n 个独立样本，Hoeffding 界给出 $\Pr(q_*(a)-\widehat q_n(a)\ge u)\le e^{-2nu^2}$。取 $u=\sqrt{2\log t/n}$，右侧为 $t^{-4}$，得到本例 $c=\sqrt2$ 的探索项尺度。实际访问数由行为决定；对可能的 $n=1,\ldots,t-1$ 取并集界，单臂失效概率至多 $(t-1)t^{-4}$，不能把固定 n 的概率原样当作自适应时刻的精确置信度。这一构造对应 Auer、Cesa-Bianchi 与 Fischer 的 UCB1，参见本页原论文来源。

![相同已观察历史下，样本平均为 (0.6,0.5,0)，UCB 给仅访问一次的 C 更大的探索项；C 得到 1 后，UCB 下一次按并列规则选 B。ε-greedy 两个快照都给 A 最高概率。](https://yingwen.io/crl-figures/bandit-exploration-scores.svg)

原创精确算例。A 的 10 次奖励中有 6 个 1，B 的 2 次中有 1 个 1，C 的 1 次为 0；下一次是 $t=14$。两幅分数图共用纵轴，青柱是 Q，紫斜纹是 UCB 探索项，橙圈标所选动作；下方蓝柱为 $\epsilon=0.1$ 的条件概率。算法不读取真实均值。

这份历史给出 $Q=(0.6,0.5,0)$、$N=(10,2,1)$。在 $t=14$，UCB 分数约为 $(1.327,2.125,2.297)$，于是选 C；ε-greedy 给三个动作 $(14/15,1/30,1/30)$ 的概率。若本次 C 得到 1，则 $Q(C)$ 从 0 变为 0.5、计数从 1 变为 2。到 $t=15$，B 与 C 的分数都约为 2.146，固定并列规则使 UCB 选 B；ε-greedy 的贪心动作仍是 A。

这一步还显示两个时钟的作用：没有访问的 A，其均值和计数保持不变，但总时间增加使它的 UCB 探索项略增；C 的访问数增加则使自身探索项减小。一个动作很久未选时，时间项可以使它再次具有竞争力。乐观初值没有这项随时间重新增长的分数。

上述概率界依赖固定均值、独立样本、有界奖励和样本平均。对不同奖励尺度应调整探索项；重尾奖励或任意神经网络预测需要另外论证。均值变化时，旧的大计数与旧样本平均可能一起阻碍适应，窗口、折扣计数和变化检测需另行设计。本图比较给定证据下的条件决策，累计表现还取决于后续实际访问与奖励。

<a id="bandit-gradient"></a>

## 4 · Gradient bandit 的推导

另一条路线保存偏好 $H(a)$，直接用 softmax 决定行为。偏好不是收益估计：给全部 H 加相同常数不改变概率。对 log-softmax 求导，被选动作项贡献一，归一化项贡献对应概率。

$$
\pi_H(a)=\frac{e^{H(a)}}{\sum_be^{H(b)}},\qquad \frac{\partial\log\pi_H(A)}{\partial H(a)}=\mathbf1\{a=A\}-\pi_H(a)
$$

由 log π(A)=H(A)−log Σ exp H 得到。

固定当前偏好 H，所优化的一步期望奖励是 $J(H)=\sum_x\pi_H(x)q_*(x)$。对 H(a) 求导，得到 $\partial J/\partial H(a)=\pi_H(a)[q_*(a)-J(H)]$。这只是分析用关系，因为算法不知道真实均值。把它改写为可采样形式，才得到实际更新。

$$
\frac{\partial J}{\partial H(a)}=\mathbb E\!\left[(R_t-B_t)\bigl(\mathbf1\{A_t=a\}-\pi_t(a)\bigr)\mid\mathcal F_t\right],\qquad H_{t+1}(a)=H_t(a)+\alpha(R_t-B_t)\bigl(\mathbf1\{A_t=a\}-\pi_t(a)\bigr)
$$

\(\mathcal F_t\) 表示本次选择前的历史；\(H_t\)、\(\pi_t\) 与 \(B_t\) 在条件期望中已确定。给定动作后的奖励均值为 \(q_*\)，且 \(\sum_x\pi_t(x)[\mathbf1\{x=a\}-\pi_t(a)]=0\)，因此历史基线不改变期望方向。更新所有偏好时都使用同一份旧概率。

![三臂从零偏好开始，先选 C 得到 1，C 概率提高；下一次仍选 C 却得到 0，奖励低于旧基线 1，C 概率回落。每一步三个偏好共同改变。](https://yingwen.io/crl-figures/bandit-exploration-preferences.svg)

原创精确两步反馈。$\alpha=0.3$，初始基线为 0。第一次偏好增量为 $(-0.1,-0.1,0.2)$，C 概率从 $1/3$ 变为约 0.403；第二次使用旧基线 1，C 概率降到约 0.340。紫色带符号长度是 H，蓝柱是 softmax 概率，各快照共用尺度；底部是更新后供下一次使用的奖励基线。

第一次成功时，$R-B=1$，三个旧概率均为 $1/3$，所以偏好从零变为 $(-0.1,-0.1,0.2)$。第二次失败时，旧基线已为 1，$R-B=-1$；用旧概率 $(0.299,0.299,0.403)$ 计算，未选 A、B 的偏好上升，C 的偏好下降。它们没有获得新的收益观测，这些变化来自概率归一化。三项增量之和始终为零，H 的绝对零点不承担收益含义。

可以独立检查期望方向：零偏好时三个概率均为 $1/3$，本任务 $J=0.5$，解析梯度为 $(-0.1,0,0.1)$。把三种动作与奖励 0、1 的六种结果按真实概率全部枚举，使用任何事先确定的标量基线，都得到这个期望方向。乘上 $\alpha=0.3$ 后，期望偏好增量为 $(-0.03,0,0.03)$；本次成功 C 的实际增量为 $(-0.1,-0.1,0.2)$。奖励噪声使实际增量围绕其期望变化。附带脚本还在非均匀策略下用有限差分核对导数。

本页实现先保存旧概率和旧平均奖励，采样动作与奖励，再计算全部偏好增量，最后更新奖励平均。基线可以由历史估计，但不得在本次样本中依赖当前动作；仅对基线停止求导还不足以保证这个统计条件。计算 softmax 时先减去最大偏好，可以避免指数溢出。

原书 §2.8 的推导使用不含当前奖励的平均；其图 2.5 的实验脚注明确采用包含当前奖励的平均。两种约定应分别写出：若旧平均为 $B_t$，先加入当前奖励得到 $B_t^+=((t-1)B_t+R_t)/t$，则 $R_t-B_t^+=(1-1/t)(R_t-B_t)$。对 $t>1$，期望方向相当于乘上这个时间系数；第一次若平均直接设为 $R_1$，偏好增量为零。它是另一种更新时序，不能直接沿用上面“历史基线不改变期望梯度”的等式。本例选定初始 $B_1=0$ 并最后更新基线，使第一份反馈也进入偏好。

| 方法 | 选择依据 | 一份新奖励写入哪里 |
| --- | --- | --- |
| ε-greedy | Q 决定利用；ε/k 给所有动作探索机会 | 被选动作的 Q 与计数 |
| 乐观初值 + greedy | 同一个 Q，起点设得较高 | 被选 Q 的更新逐次消耗初值影响 |
| UCB | 样本均值 Q 加时间与计数探索项 | 被选 Q 与计数；下一步重算所有选择分数 |
| Gradient bandit | H 的 softmax 概率 | 使用旧概率更新全部 H，再更新奖励基线 |

前三种仍通过价值估计决定行为，gradient bandit 则直接学习相对偏好。softmax 在有限偏好下给各动作正概率，但概率可能变得极小；这本身不提供一个固定的长期访问下界。它也没有把不确定性单独变成 UCB 的探索项。判断方法的探索作用，需要追踪它维护的量如何改变下一次真实行动。

<a id="bandit-algorithm"></a>

## 5 · 完整交互步骤

**算法：算法伪代码**

1. 价值路线：初始化 Q、计数 N；乐观方法只改变 Q 的初值。
1. 每次由旧 Q 构造 ε-greedy，或由 Q、N、t 构造 UCB。
  1. 未访问的 UCB 动作先行尝试；其他并列按固定规则处理。
  1. 选择动作并观察其奖励；只增加该动作的 N 并更新该动作的 Q。
1. 偏好路线：初始化 H、奖励基线 B、总交互计数。
  1. 保存旧 softmax 概率与旧 B；抽样动作并观察其奖励。
  1. 用旧概率、R−B 同时计算所有 H 的增量。
  1. 最后更新奖励基线；记录奖励、动作与新策略。

比较要同时记录总奖励和各动作次数。相同整数 seed 不保证不同算法收到逐时对齐的潜在奖励，因为它们可能以不同顺序消耗随机数。本例只使各自运行可复现，不作共同随机数配对的结论。

<a id="lesson-example"></a>

## 6 · 手算与真实采样

| 情形 | 计算 | 结果 |
| --- | --- | --- |
| 第3次访问，旧均值1，新奖励4 | 1+(4−1)/3 | 新均值2 |
| 三动作，ε=.1 | 1−.1+.1/3 | 贪心概率 .933333 |
| 二动作偏好全零，选动作1，R−B=2，α=.1 | ΔH=.1×2×(−.5,.5) | 新偏好 (−.1,.1)，动作1概率约 .549834 |

三臂成功率为 (.2,.5,.8)，每种方法运行 4000 步。seed=7 时 ε-greedy 的次数为 (152,139,3709)，平均奖励 .76025。这说明代码执行了实际动作选择与奖励采样，不证明其优于另两种方法。改变种子和奖励差距后，应观察全过程。

<a id="lesson-code"></a>

## 7 · 源码与实验观察

先运行本页的逐步算例，再进入随机采样。[bandit_tracking_walkthrough.py](/crl-code/tutorials/bandit_tracking_walkthrough.py) 是可独立下载的标准库脚本；update_selected 只读写被选动作，history_weights 展开历史权重，fixed_reward_trace 输出每一步的旧值、步长、误差和新值。compare_estimates_and_actions 将估计误差与决策收益分开计算。

下载脚本后运行：先核对七项确定性检查，再查看八次更新的 JSON；不采样环境

```sh
python3 bandit_tracking_walkthrough.py --test
python3 bandit_tracking_walkthrough.py
```

逐行检查第五条记录：visit=5，sample_average.before=0.5，step_size=0.2，error=0.5，after=0.6。constant_step.after=0.65625。接着把未选择的 A、C 改成不同初值，确认这两项仍不随 B 的经验改变。若把 B 的访问日程从每步改成每十步，按访问次数画的图应保持相同，按环境时间画的图会拉长。

[bandit_exploration_walkthrough.py](/crl-code/tutorials/bandit_exploration_walkthrough.py) 延续同一三臂任务，输出乐观初始化的六步闭环、同一观察历史下的 ε-greedy/UCB 决策，以及偏好的两步反馈。它只用 Python 3 标准库，不需要前一脚本；下载后在任意目录运行下面命令。

六项确定性检查；JSON 保存旧值、奖励、计数、选择分数、概率与偏好增量。有限枚举核对期望，未采样训练曲线。

```sh
python3 bandit_exploration_walkthrough.py --test
python3 bandit_exploration_walkthrough.py --json
```

先读 selection.before：t=14、action=2 即 C；after.action=1 即 B。确认 C 的奖励只写入 C 的均值，但总时间变化会影响各臂 UCB 探索项。再读 gradient_trace：第二条 baseline_before=1，所有 delta 都由 policy_before 计算。最后看 gradient_check：六项结果概率之和为 1，prior_baseline.expectation 对上解析梯度；current_baseline.expectation 为前者的 4/5，对应先有四份历史奖励再纳入当前奖励。

稳定 softmax、探索分布、UCB 与 gradient bandit 的完整采样循环。

```python
def softmax(values):
    top = max(values)
    weights = [math.exp(v - top) for v in values]
    return [v / sum(weights) for v in weights]

def choose(probabilities, rng):
    u, total = rng.random(), 0.0
    for i, p in enumerate(probabilities):
        total += p
        if u < total:
            return i
    return len(probabilities) - 1

def epsilon_probs(values, epsilon):
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0,1]")
    best = max(range(len(values)), key=lambda i: values[i])
    probabilities = [epsilon / len(values)] * len(values)
    probabilities[best] += 1.0 - epsilon
    return probabilities

def ucb_action(values, counts, time):
    unseen = [a for a in range(len(values)) if counts[a] == 0]
    if unseen:
        return unseen[0]
    return max(range(len(values)), key=lambda a:
               values[a] + math.sqrt(2 * math.log(time) / counts[a]))

def bandit(method, steps=4000, seed=7):
    """Stationary Bernoulli arms. Gradient baseline uses the OLD mean."""
    if steps < 1:
        raise ValueError("steps must be positive")
    rng = random.Random(seed)
    means = [0.2, 0.5, 0.8]
    q, counts, preferences = [0.0]*3, [0]*3, [0.0]*3
    baseline, total = 0.0, 0.0
    for t in range(1, steps + 1):
        if method == "epsilon":
            action = choose(epsilon_probs(q, 0.1), rng)
        elif method == "ucb":
            action = ucb_action(q, counts, t)
        elif method == "gradient":
            probabilities = softmax(preferences)
            action = choose(probabilities, rng)
        else:
            raise ValueError(method)
        reward = float(rng.random() < means[action])
        total += reward
        counts[action] += 1
        q[action] += (reward - q[action]) / counts[action]
        if method == "gradient":
            for a in range(3):
                preferences[a] += 0.1 * (reward - baseline) * (
                    float(a == action) - probabilities[a])
        baseline += (reward - baseline) / t
    if method == "ucb":
        action = ucb_action(q, counts, steps+1)
        next_policy = [float(a == action) for a in range(3)]
    else:
        next_policy = (softmax(preferences) if method == "gradient"
                       else epsilon_probs(q, 0.1))
    return {"counts": counts, "mean_reward": total / steps, "q": q,
            "next_action_probabilities": next_policy}
```

需要运行完整行为循环时，可分别打开[样本均值 ε-greedy](/zh/continual-rl/code/bandit_sample_average/)与[常数步长 ε-greedy](/zh/continual-rl/code/bandit_constant_step/)的独立实现、命令和已有结果图。两者当前默认都在平稳三臂任务中运行；本页的变化算例不能当作这两个默认实验已经测过的结果。

进一步运行[UCB](/zh/continual-rl/code/bandit_ucb/)或[gradient bandit](/zh/continual-rl/code/bandit_gradient/)的完整采样循环，观察这些选择机制在相同平稳任务里累积了什么经验。完整算法比较应固定交互预算、任务分布与独立重复，并同时记录奖励和访问；本页的短轨迹与一时刻决策不能给四种方法排出通用名次。

tabular_textbook_lab.py 的 bandits 命令输出最终每个动作的访问次数、样本均值、全程平均奖励和下一次选择分布；独立实现的结果 CSV 则只保存 step、value、phase，不能从奖励曲线恢复动作计数、Q 或 π。本页新的 JSON 逐步保留这些机制诊断。ε-greedy 与 gradient bandit 给出随机分布；固定并列规则的 UCB 根据当前均值、计数和时间确定下一动作，因此其条件选择分布是 one-hot。

先把预算降到三步，检查 UCB 是否恰好尝试每个动作一次；再把最优两臂均值改得更接近，观察识别排序所需的信息增加。若引入中途均值变化，同时保留样本平均和常数步长对照。比较变化后的窗口奖励，而不是只看被变化前历史占主导的累计平均。

<a id="bandit-evaluation"></a>

## 估计得更准，是否一定选得更好？

回到固定均值 $(0.2,0.5,0.8)$ 的三臂任务。均方误差 $\frac13\sum_a[Q(a)-q_*(a)]^2$ 给三个动作等权，衡量估计与真值的距离；greedy 行为只取当前估计最大的动作，取决于排序。下表给出两个可以直接检查的估计快照。

| 估计快照 | 三个估计值 | 均方误差 | 贪心动作与期望奖励 |
| --- | --- | --- | --- |
| X | (0.2, 0.79, 0.78) | 约 0.02817 | B；0.5 |
| Y | (0, 0, 0.55) | 0.1175 | C；0.8 |

X 的整体估计更准确，却把 B、C 的顺序排反了。Y 的数值误差较大，却正确选择 C。若两者都以 ε=0.1 均匀探索，下一步期望奖励分别为 0.9×0.5+0.1×0.5=0.5 与 0.9×0.8+0.1×0.5=0.77。一次实际奖励仍然只能是 0 或 1；表中比较的是给定当前行为的期望。

评价因此需要保留两个层次。估计诊断看各臂误差、访问次数和变化后的反应。控制评价看实际获得的累计奖励、近期奖励与动作选择。在模拟器中可以读取真实均值计算误差和伪遗憾；真实系统通常只提供被选动作的奖励，不能把未知最优臂的收益直接写进日志。

均值随时间变成 $q_t(a)$ 时，可以另定义动态比较量 $\sum_{t=1}^T[\max_a q_t(a)-q_t(A_t)]$。这里的参照者每步知道当前最好动作，比固定最佳臂拥有更多信息。要讨论其可达性，需要约束变化幅度、变化频率或可用线索；任意且不可预测的变化没有可学习的共同规律。

实践按三个对照逐步推进：先给两个估计器完全相同的奖励流；再固定估计更新，只改变探索；最后允许估计和行为共同变化。前两项区分证据利用与证据获取，第三项才回答完整学习算法在同等交互预算下取得多少奖励。它也为后面的持续强化学习评价留下同一个问题：测的是一个最终策略，还是这个策略被学出来的整个过程？

<a id="lesson-branches"></a>

## 8 · 限制与持续学习

乐观初值通过高估尚未试过的选择诱导探索，但这种驱动力可能在所有动作访问过后消失。随机覆盖、置信上界和直接策略梯度从不同机制分配经验，不能仅根据一个最终动作概率判断其作用。

沿用抓取例子。假设一种方式的奖励从恒为0变为恒为1，变化时它的估计仍为0。以后每访问它一次就以 $\alpha=.1$ 更新。经过 n 次这样的访问，估计是 $1-.9^n$，至少要22次才达到.9。若每个环境步都尝试它，只需22步；若固定每100步才尝试一次，则要2200步。这里没有奖励噪声，慢适应完全可以由信息来得稀少造成。

因此要同时观察两个时钟：真实交互经过多少步，这个动作实际获得多少次新证据。提高步长改变收到证据后的反应；增加探索改变何时收到证据。持续学习实验可以先固定访问日程比较更新，再固定更新规则比较访问；最后让二者共同决定行为。这样才能区分“没有学会”与“没有再看见”。

若动作影响未来状态，眼前奖励均值不足以评价决策。若均值随时间改变，最佳固定动作也可能不是合适比较对象。先规定变化、预算与目标，再选择累计收益、窗口回报或跟踪误差。

<a id="lesson-check"></a>

## 9 · 带答案练习

问：Q=.8，α=.1，随后三个奖励均为零，估计是多少？答：.72、.648、.5832。若改用极小的样本平均步长，跟踪会更慢；这个算例尚未检验是否探索到了变化。

问：UCB 能否忽略从未访问、Q=0 的动作？答：不能。零是初始化而非低收益证据；程序应优先尝试未访问动作。

问：乐观例子把 α 改成 1，某臂第一次访问后还保留多少初值？答：零。若三个动作都访问完、估计均为零，纯贪心仍可能按并列规则只选 A；初始化没有提供每一步新的随机访问机会。

问：为什么 gradient bandit 更新未选动作？答：概率受到归一化约束。一次高于 baseline 的奖励提高被选动作的相对偏好，同时降低其他动作；偏好增量之和为零。

问：已经有四份历史奖励，先把当前奖励写进平均再更新偏好，方向会怎样变化？答：R−B 变成原来的 4/5，因此在相同旧策略下每项偏好增量也乘以 4/5；这个版本与使用历史基线的无偏梯度估计应分开说明。

## 从本章进入实践

[估计与适应](https://yingwen.io/zh/continual-rl/code/#practice-tracking)：旧经验越来越多时，怎样仍对新变化作出反应？



<a id="chapter-code"></a>

## 下载与运行

Python 3，仅标准库。下载完整文件后在其目录运行；页面展示核心区域。小环境和解析检验用于理解机制，不是论文基准复现。

[下载 tabular_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/tabular_textbook_lab.py)

```sh
python3 tabular_textbook_lab.py bandits
python3 tabular_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：对应第 2–8 章。以下推导、例子和标准库实现独立编写；原书用于核对定义、算法关系与适用条件。

- [David Silver · UCL 强化学习课程](https://davidstarsilver.wordpress.com/teaching/)：原课程页面提供 MDP、动态规划、预测、控制、探索和规划的讲义及视频。

- [Auer, Cesa-Bianchi & Fischer · Finite-time Analysis of the Multiarmed Bandit Problem (2002)](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf)：原论文 §2、UCB1 图 1 与证明中的 Chernoff–Hoeffding 界。适用条件为各臂独立同分布、奖励支持集 [0,1]；本页只借助上界构造解释选择分数。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-tabular-bandits#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-tabular-bandits#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-tabular-bandits)
<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 一次策略更新，为什么能改善未来？

精确策略改善使用旧策略的真实价值；策略梯度使用与目标匹配的访问分布。值函数近似或梯度估计误差会破坏这些推理的前提。

函数逼近与深度方法：PPO 的动作概率比不等于新策略的状态访问比。连续动作 actor 还会追逐 critic 的误差。限制局部更新尺度与证明实际回报单调提高是不同要求。

持续学习中的研究问题：动作不仅改变世界，也改变未来数据与学习。比较冻结策略不等于比较持续更新的智能体；什么时候应付出当前回报去获得长期有用的经验？

[精确策略改善](https://yingwen.io/zh/continual-rl/foundations/tabular/dynamic-programming/) → [策略梯度定理](https://yingwen.io/zh/continual-rl/foundations/approximation/policy-gradient/) → [TRPO 与 PPO 的近似](https://yingwen.io/zh/continual-rl/foundations/deep/trust-region/) → [持续控制的比较器](https://yingwen.io/zh/continual-rl/algorithms/control/)


### 可进一步检验的问题

- [07 · 哪些学习参数应当适应，怎样评价学出来的更新规则？](https://yingwen.io/zh/continual-rl/research/#research-learning-rules)：样本平均与常数步长在同一变化奖励流中的差别，先展示跟踪速度与噪声的取舍，再引出怎样从经验选择步长。
- [14 · 应当探索什么、练习什么，以及如何保留未来交互与学习的机会？](https://yingwen.io/zh/continual-rl/research/#research-experience-selection)：探索在当期收益与信息之间取舍；估计更准确却未必选择更好，说明取得的信息还需要任务用途。
- [16 · 什么实验能区分“仍在更新”与“仍在有效学习”？](https://yingwen.io/zh/continual-rl/research/#research-measurement)：同一奖励流中估计误差与选取动作的收益可以给出不同结论，要求评价指标先对应明确对象。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)
- [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

对应原始材料：2.1–2.8。本文为原创讲解，原书、论文与上游代码保留各自许可。
