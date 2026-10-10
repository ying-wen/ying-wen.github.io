# 学习与规划：Dyna、优先扫描和执行时搜索

表格强化学习 · 第 7 章

真实经验既能直接改进价值，也能训练后果模型。规划使用这个模型继续计算，关键是模型语义、backup 的成本以及计算应分配到哪里。

## 本章内容

- 实现 Dyna 的真实交互、模型更新与模拟更新循环。
- 推导优先扫描的前驱传播，区别 sampling 与 expectation backup。
- 理解 rollout、MCTS 与持续环境中模型错误的边界。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [动态规划：评价、改善与最优递推](https://yingwen.io/zh/continual-rl/foundations/tabular/dynamic-programming/)：理解模型备份与价值传播。
- [TD 预测与控制：SARSA、Expected SARSA、Q-learning 和 Double Q](https://yingwen.io/zh/continual-rl/foundations/tabular/temporal-difference/)：区分真实转移产生的样本更新和模型产生的更新。


### 概率与期望

概率非负且总和为一；期望是按概率加权的平均，不是单次结果。条件期望只比较满足已知条件的情形。

### 递推与赋值

递推通过已有估计计算新估计。下标表示时间；更新符号表示程序中的赋值，不是数学恒等式。

<a id="problem-definition"></a>

## 本章的问题定义

世界模型可查询或可由经验估计。真实交互有限，额外模型计算用于改进价值和决策。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 模型 $\hat p$ 与每次真实交互之间允许的备份预算 $B$。

### 需要求解的对象

在预算内安排模型备份，并从改进后的价值或搜索结果选择动作。

### 信息与数据权限

真实样本与模型生成样本来源不同；模型计算不会增加真实环境的信息量。

$$
(\hat T_*Q)(s,a)=\mathbb E_{\hat p}[R+\gamma\max_{a^{\prime}}Q(S^{\prime},a^{\prime})\mid s,a],\qquad N_{\rm backup}\le B
$$

模型备份逼近在模型中的最优性递推；最终仍按真实环境收益评价。模型内残差降低不代表真实控制误差降低。

### 成立条件与解的含义

- 模型语义、终止和奖励必须与外部过程一致。
- 精确模型下的 DP 结论不能直接用于有偏学习模型。

判断准则：分别计数真实步、模型查询和耗时，并区分采样误差、模型误差与规划不足。

### 适用边界

- 把模型 rollout 当作无成本的真实经验。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [Dyna 与模型学习](https://yingwen.io/zh/continual-rl/algorithms/dyna/)：本章把模型备份与真实更新、模型估计组合成 Dyna 循环，分别检查三者误差。

- 特例：增加条件 · [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)：本章限定为原子动作的表格备份；时间抽象规划还需技能持续时间和终点模型，是相关章更广的接口。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

计算预算不足以遍历全部状态；模型本身又可能过时或错误。

### 本章的核心思路

区分模型如何学习、一次备份如何执行，以及备份应分配到哪里。

1. [决定求期望还是采样](#lesson-derive)：两种备份在模型正确时指向同一对象，但方差与计算成本不同。

2. [把模型接入真实学习循环](#planning-dyna)：每次真实经验既更新价值也更新模型，再调用预算内的模型更新。

3. [把有限计算放到误差会传播的位置](#planning-priority)：前驱和优先队列加速传播，但队列本身不保证模型正确。

结论与条件：精确备份和充分覆盖可连接 DP 理论；有限预算、有偏模型需单独测量控制后果。

### 相关方法改变了什么

- Dyna / MPC：Dyna 将模型计算写入持久价值；MPC 为当前动作重新规划。

- 均匀采样 / 优先扫描：改变备份分配，不改变模型真实性条件。


<a id="lesson-setting"></a>

## 1 · 什么是模型与规划

从经验更新一次价值以后，这条经验还告诉了我们动作可能造成什么后果。若把这种关系学成模型，之后即使没有新的真实交互，也能重新考虑它与其他已知后果的联系。这样，动态规划中的计算与 MC、TD 中的经验学习开始结合。规划能进一步利用已有信息；发现模型不知道的后果，仍需要新的经验或其他信息来源。

分布模型给出所有可能后果及其概率；采样模型每次返回一个随机后果。它们都不是价值函数：模型描述动作之后会发生什么，价值结合奖励目标与继续行为评价这些后果。规划通过模型而非真实执行取得更新信息。

本章先使用有限平稳 MDP、折扣 $\gamma<1$。真实转移为 $(s,a,r,s')$，终止单独标记。若模型恰好等于环境，规划收敛指向环境中的相应价值；若模型错误，规划求解的是模型所定义的问题。

先让道路保持确定，再只改变一条出口。S 有 go、exit 两个动作：go 到 A，奖励 0；A 只有 cross，旧时到 G 得 1；exit 从 S 直接到 B 得 0.6。G、B、D 都是终止状态，其他奖励为 0，γ=0.9。学习器知道状态名与可用动作，后果表只从实际经验填写。接下来把 A 的出口由 G 改成 D，奖励变为 0；这个变化由读者看见，学习器必须执行 cross 才能观察到。

旧阶段先实际走过 S→A→G，再重置并探索 exit，走过 S→B，共 3 个环境步、1 次 episode 重置。Q 初值为 0、α=1：第一条记录还只把 Q(S,go) 更新为 0；第二条使 Q(A,cross)=1，随后对模型中的 S:go 做一次备份得到 0+0.9×1=0.9；第三条使 Q(S,exit)=0.6。模型由这三条记录建立，起点因 0.9>0.6 选择 go。这里的旧经验日程是给定的小例子，尚未比较探索算法。

<a id="lesson-derive"></a>

## 2 · 期望 backup 与采样 backup

$$
y_{\rm exp}(s,a)=\hat r(s,a)+\gamma\sum_{s'}\hat p(s'\mid s,a)\max_bQ(s',b)
$$

模型期望枚举所有后果。若奖励与后继相关，也可以直接对联合模型求和；r̂ 是相应即时奖励期望。

$$
(\tilde r,\tilde s')\sim\hat p(\cdot,\cdot\mid s,a),\qquad y_{\rm sample}=\tilde r+\gamma\max_bQ(\tilde s',b)
$$

后继真正终止时尾值为零。固定 Q，按同一个学得模型采样，样本 target 的条件期望等于该模型的期望 target；这不要求学得模型已经等于真实环境。

期望 backup 的成本随分支数增长；采样 backup 每次便宜，但需要重复才能降低随机性。比较应匹配计算成本或模型调用次数，而不是把“一次更新”当作跨算法相同单位。对确定性环境，两者在后果计算上可以相同。

在确定性旧模型中，cross 只给出 G、奖励 1、终止，期望与采样目标都为 1。随机道路还要回答另一个问题：概率从哪里来？下面沿用这些地点和动作，先让真实记录形成模型，再比较用这个模型计算的两种目标。随后回到原来的确定性关闸例，展开前驱调度。

<a id="rlss-planning-computation"></a>

## 一次期望更新不等于一次采样更新的计算量

固定一个状态动作及当前后继价值，令一次模型采样得到的 backup target 为随机变量 $Y$，其均值为 $m$、方差为 $\sigma^2$。若模型提供独立同分布样本，使用样本均值 $\bar Y_k$，便有下式。

$$
\mathbb E[(\bar Y_k-m)^2]=\frac{\sigma^2}{k},\qquad
\operatorname{RMSE}(\bar Y_k)=\frac{\sigma}{\sqrt{k}}.
$$

这是固定目标的采样误差，不是任意 Dyna 学习曲线。后继价值也在变化、样本相关或使用常数步长时，不能直接套用。

若精确求和要访问一千个后继，一千次模型后果计算可以用于一个精确备份，也可以分散到许多状态的粗备份。后者往往能更早改变行为，但不是必然。高方差、稀有高后果事件和相关采样可能使粗备份漏掉重要差异。精确期望只消除了当前模型内部的采样误差，不消除模型偏差。

| 设计轴 | 可选择的机制 | 须独立记录 |
| --- | --- | --- |
| 后果计算 | 完整求和、单次样本、多个样本、线性期望模型 | 模型调用数、方差、偏差 |
| 起点选择 | 均匀扫、从当前状态按策略模拟、优先队列 | 覆盖、起点性能、队列维护成本 |
| 更新保留 | 后台更新持久价值、只保留当前搜索树 | 记忆、复用、响应时间 |
| 模型维护 | 持续拟合、遗忘旧证据、重新探索 | 真实交互、模型变化和校准 |

Trajectory sampling 让规划跟随当前策略，从而把早期计算集中到当前可到达的区域；均匀扫描则可能更早处理暂时不用但未来有用的分支。两者不能仅凭最终起点价值决定胜负。在持续控制中，还应在路径突然堵塞或新捷径出现后检查模型能否得到新证据。计算分配只能重组已有信息，不能凭空发现尚未观察的改变。

<a id="stochastic-model-task"></a>

### 从真实后果学习随机模型

只撤掉 cross 后果确定这一项条件：在新的平稳道路中，它以 3/4 概率到 G 得 1，以 1/4 概率到 D 得 0；两种后果都真正终止。S:go 仍到 A 得 0，S:exit 仍到 B 得 0.6，γ=0.9。真实概率是读者核算的参照，学习器从空模型开始，只知道地点与合法动作；不把前面确定性旧阶段的记录混入这个随机变体。

先实际尝试一次 exit，再规定四个探索 episode，每次执行 go、cross。收到的 cross 后果依次为 D、G、G、G。这是随机道路可能生成的一段记录；探索日程给学习器四次观察 cross 的机会，后果并非由学习器指定。每条真实记录先做 α=1 的直接 Q 更新，再写模型；每次 cross 之后，依次对 A:cross 和 S:go 做期望备份。直接更新与模型估计使用不同规则：直接 cross 更新先读刚到来的奖励，模型则保留全部后果的计数。

$$
\hat p_n(s',r,d\mid s,a)=\frac{N_n(s,a,s',r,d)}{N_n(s,a)},\qquad \hat r_n(s,a)=\sum_{s',r,d}\hat p_n(s',r,d\mid s,a)\,r
$$

$N_n(s,a)$ 是这个状态动作得到的真实记录数，分子计联合后果；$d$ 是终止标记。仅在分母大于零时查询模型。奖励和后继可能相关，不能分别抽取两个边缘分布，再把本例没有出现过的“D、奖励1”组合出来。

![同一道路的随机cross分支与四个实际后果D、G、G、G；每个前缀显示联合计数归一化后的G和D概率及奖励均值。](https://yingwen.io/crl-figures/stochastic-model-walkthrough-observations.svg)

上方紫虚线画模型可能预测的两种后果；下方圆点是实际cross记录，青段与红段的长度分别为G、奖励1与D、奖励0的经验概率。N只计cross真实访问，初次exit及每次到A的go另计。四个前缀的奖励均值为0、1/2、2/3、3/4；原创给定记录的精确计算，依据 Sutton 与 Barto §8.1、§8.5。

| cross真实访问 | 新后果 | G计数 / N | 奖励均值与规划后cross | 规划后go | 贪心选择 |
| --- | --- | --- | --- | --- | --- |
| 1 | D，奖励0 | 0 / 1 | 0 | 0 | exit |
| 2 | G，奖励1 | 1 / 2 | 0.5 | 0.45 | exit |
| 3 | G，奖励1 | 2 / 3 | 2/3 | 0.6 | exit：平局 |
| 4 | G，奖励1 | 3 / 4 | 0.75 | 0.675 | go |

例如第二次 cross 之后，直接更新先把 cross 写成 1，模型计数却给出 $\hat p(G,1)=\hat p(D,0)=1/2$。期望备份把 cross 写成 0.5，再沿已学到的 S→A 模型把 go 写成 $0+0.9\times0.5=0.45$。它低于出口值 0.6，因此当前贪心行为仍选 exit。下一次真实探索使均值变为 2/3；go 恰为 0.6，按固定平局规则仍选 exit。只有经验成功比例严格超过 2/3 才会选 go。

第一份 D 记录使经验模型给 G 零质量，这是目前的频率估计，尚不足以认定 G 不可能。四次 cross 也只给四份这种后果信息。算上第一次 exit 和四次 go，获取阶段共 9 个环境步、4 次 episode 重置、9 次模型写入、9 次直接备份。随后查询模型形成的模拟后果不会增加这些真实计数。

<a id="stochastic-model-backup"></a>

### 冻结同一模型和 Q，比较两种备份

停在 D、G 这两份 cross 记录之后：模型给两个联合后果各 1/2，Q(cross)=0.5、Q(go)=0.45、Q(exit)=0.6。先冻结这份 Q 和模型。一次模型采样的 cross 目标只能是 0 或 1，期望目标是 0.5；两个采样目标按模型概率加权，正好得到期望目标。这里消除的是从当前模型抽样的波动。真实 cross 期望仍为 0.75，完整枚举当前模型也会保留 −0.25 的模型估计误差。

$$
y_{\rm exp}^{\hat p,Q}(s,a)=\sum_{s',r,d}\hat p(s',r,d\mid s,a)\left[r+\gamma(1-d)\max_bQ(s',b)\right],\qquad \mathbb E_{\hat p}[y_{\rm sample}\mid\hat p,Q]=y_{\rm exp}^{\hat p,Q}(s,a)
$$

终止项只取奖励，程序不读取终点动作值。条件期望固定的是当前模型和当前Q；它没有把学得概率替换成环境概率。

在这份模型和冻结 Q 下，采样目标方差为 $1/2\times1/2=0.25$，独立抽取 m 次并平均后的方差为 $0.25/m$。若改为读者知道的真实分布，则均值为 0.75、方差为 0.1875；它们属于另一分布。重复模型抽样可以更准确地算出 0.5，却不会把两份真实记录变成更多独立证据。后继 Q 或模型在调用间变化时，也不能直接沿用这条固定均值的方差公式。

接着从同一快照分出三条计算分支。各以 α=1 写入自己的 cross，再查询已学到的 S:go 后果，把新 cross 值传回 S。第一项目标比较使用共同冻结 Q；传播这一步读取各分支更新后的 Q，所以之后的表已经不同。

![同一冻结Q与经验模型下，抽到D、抽到G和枚举全部后果分别把cross写成0、1、0.5，再传播到go并产生下一条实际道路。](https://yingwen.io/crl-figures/stochastic-model-walkthrough-backups.svg)

从紫色模型后果沿虚线读到cross目标，再看橙色go柱长。三个cross目标都来自同一快照；随后的go目标分别读本分支更新后的cross。下方橙路径是下一真实episode：选择go的分支这次实际到D，选择exit的分支到B。底部只核算行动前规划，新真实样本为0；实际episode另加1或2个环境步。原创精确分支算例，依据 Sutton 与 Barto §8.5；不是算法表现排序。

| 行动前计算 | cross目标 → go目标 | 接口查询 | 后果展开 | 模型价值写入 | 下一真实episode |
| --- | --- | --- | --- | --- | --- |
| 采样抽到D，再备份go | 0 → 0 | 2 | 2 | 2 | exit：S→B；1步，回报0.6 |
| 采样抽到G，再备份go | 1 → 0.9 | 2 | 2 | 2 | go：本次S→A→D；2步，回报0 |
| 期望备份cross，再备份go | 0.5 → 0.45 | 2 | 3 | 2 | exit：S→B；1步，回报0.6 |

这里的一次接口查询，对分布模型返回全部已观察联合后果，对采样模型返回一个。cross 的期望查询展开两条后果，采样查询处理一条；S:go 只有一条后果，所以总展开量分别为 3 与 2，查询和价值写入却都为 2。只有两分支且全都终止的小例子还没有比较大模型上的墙钟成本。原书 §8.5 按后继价值计算讨论分支因子；接口查询、处理后果和写价值不是同一个计量单位。

三条分支的模型在行动前完全相同。价值写入改变了起点选择，真实执行才提供不同记录：exit 分支只新观察到 S→B；go 分支又得到一次 cross 失败，模型成功比例变为 1/3。按相同的两项期望规划日程，后者的 go 随后变为 0.3，前者回到 0.45。真实道路中 go 的下一完整 episode 期望回报仍为 0.9×0.75=0.675，高于 exit 的 0.6；图里 go 的一次失败不能用于判断哪种备份整体更好。

<a id="stochastic-model-feedback"></a>

### 估计怎样改变下一次信息获取

四次探索中的后两个 G 不是纯贪心行为自己保证取得的。从 D、G 前缀出发，若继续采用期望备份和纯贪心行为，S 会选 exit，下一段实际经验只有 S→B。cross 的真实计数仍为 2；沿当前贪心策略模拟十个模型 episode，也只会查询 S:exit，不会增加 cross 证据。要再次估计这条随机道路，必须有实际访问机会，例如保留探索。规定探索日程在此把“信息是否取得”与“取得以后如何计算”分开。

这段 D、G、G、G 记录恰好在第四次恢复了正确排序，还需要检查其余可能记录。保持相同探索日程，四次 cross 有 16 种 G/D 序列。按真实成功率 3/4 给每条序列赋概率，若其中有 k 个 G，经验 go 值为 0.9k/4。至少 3 个 G 才会选择 go。

![D、G、G、G前缀的go值跨过exit阈值后自主执行S→A→D；第五个真实失败使go降为0.54；下方枚举四次cross的全部16种后果概率。](https://yingwen.io/crl-figures/stochastic-model-walkthrough-feedback.svg)

上方横轴只计cross真实访问数；前四点来自规定探索，红色第五点来自第四次规划后自主选择go取得的新D记录。橙虚线是出口值0.6，平局选exit。下方柱高为全部16序列按G个数合并后的精确概率；青柱选go，红柱错误选exit。每个四次获取日程都是9个真实步，下一自主episode分别消耗1或2步。原创计数、真实执行日志和解析枚举，没有随机训练曲线。

| 四次cross中的G数 k | 精确概率 | 经验go值 | 下一贪心选择 |
| --- | --- | --- | --- |
| 0 | 1/256 | 0 | exit |
| 1 | 12/256 | 0.225 | exit |
| 2 | 54/256 | 0.45 | exit |
| 3 | 108/256 | 0.675 | go |
| 4 | 81/256 | 0.9 | go |

因此四份 cross 记录后的错误选择概率为 $(1+12+54)/256=67/256$，正确选择 go 的概率为 $189/256$。在这些记录之后冻结各自贪心策略，按下一完整 episode 的真实折扣回报比较，平均价值为 $\frac{189}{256}\times0.675+\frac{67}{256}\times0.6\approx0.655371$。真概率只用于这个解析评价，学习器仍只读取计数。获取阶段每条序列都用 9 个真实步；下一 episode 的真实长度却为 2 或 1，平均 $1+189/256\approx1.738281$ 步。固定 cross 样本数、固定完整episode数和固定总环境步预算是不同对照。

回到具体的 D、G、G、G 分支。第四次规划后 go=0.675，学习器自主选择 go；实际到 A 后 cross 又失败，收到 D、奖励0。这次新记录把成功比例由 3/4 改成 3/5。再备份 cross 与 go，得到 go=0.54，于是下一次重新选 exit。估计、行动和数据之间形成了反馈：少量随机记录可以让行动来回变化，后续访问机会又受到这些行动影响。

原书 §8.6 的轨迹采样讨论把规划计算投向哪里：沿当前策略在模型里生成轨迹，再更新经过的状态动作。它与“每次目标只抽一个后果”是两个选择；书中那组调度实验沿模型轨迹选位置，仍在各位置做期望备份。本例的当前模型轨迹选 exit 时，计算也集中于 S:exit。模型查询改变计算位置，真实行动才改变模型能得到的证据；两条路径的访问计数应分别保存。

<a id="planning-budget-task"></a>

### 分支后面还有动作：一次备份要读多少项

前面的 cross 直接终止，每个后果只需读奖励。为了看清分支计算与价值传播的联系，沿用 S、A、go、exit 这条道路，把 cross 的终点向后延一层：cross 奖励0，以相同概率1/4到 C1、C2、C3、C4。它们都未终止，各有 collect 和 wait；collect 到 G 的奖励依次为2、2、0、0，wait 到 D 的奖励均为0。G、D、B 真正终止，S:go 仍到 A 得0，S:exit 到 B 得0.6，折扣仍为0.9。

这一次给定完整且精确的模型，暂时把“后果学得准不准”固定下来，单独研究计算。它是道路的另一个变体，初值除 Q(S,exit)=0.6 外全为0；不沿用前面 D、G、G、G 的经验模型。wait 的初值恰好正确，四个 collect 的值则等待模型备份。模型知道后果并不使 Q 自动知道回报，起点必须通过一连串更新才能使用远处奖励。

![S的go经A随机到四个非终止C状态，每个C有collect与wait两项动作，collect奖励2、2、0、0；exit直接得0.6。](https://yingwen.io/crl-figures/planning-budget-walkthrough-task.svg)

沿紫虚线读模型后果，cross四个后继各占1/4；C的两个出口解释为什么每个非终止后果还要读取两项Q。重复画出的G、D分别表示同一终点，双圈为真正终止。初值只有exit=0.6。原创已知模型算例，计算问题依据 Sutton 与 Barto §8.5、§8.9。

先沿完整道路算一把参照尺。四个 C 的最优价值为 $2,2,0,0$，因此 $q_*(A,\mathrm{cross})=0.9(2+2+0+0)/4=0.9$，$q_*(S,\mathrm{go})=0.9\times0.9=0.81$。也可以直接沿三次转移求回报：奖励为 $(0,0,r_i)$，故 $\sum_i\frac14\gamma^2r_i=0.81$。这个路径求和只供读者检验；调度器按固定地点编号选更新位置，不用真值决定先更新哪片区域。

$$
y_{\rm exp}(A,\mathrm{cross})=\frac{\gamma}{4}\sum_{i=1}^{4}\max\{Q(C_i,\mathrm{collect}),Q(C_i,\mathrm{wait})\},\qquad y_{\rm sample}=\gamma\max\{Q(C_I,\mathrm{collect}),Q(C_I,\mathrm{wait})\},\quad I\sim\operatorname{Unif}\{1,2,3,4\}
$$

一期望目标展开四个联合后果，并读取八个后继动作值；一样本目标展开一个后果，读取两个动作值。两种目标都只写一次cross。它们读的是执行当时的Q，终点目标不会读取任何后继动作值。

规定本例的基本工作量为 $W=N_{\rm query}+N_{\rm outcome}+N_{\rm Qread}+N_{\rm Qwrite}$：一次模型接口请求、一项联合后果处理、一次后继动作值读取、一次价值表写入各记一单位。这个相加约定使预算可重复核算，不把它当作秒数。模型接口返回全部后果和返回一条后果可以各算一次请求，真正遍历的条数仍须另计。若不同后继有不同动作数，期望目标的读取数是 $\sum_i|\mathcal A(C_i)|$，不能只报分支因子。

| 被更新的项目 | 接口查询 | 后果展开 | 后继动作Q读取 | Q写入 | 基本工作量 W |
| --- | --- | --- | --- | --- | --- |
| 一个 collect | 1 | 1 | 0 | 1 | 3 |
| cross：采样 | 1 | 1 | 2 | 1 | 5 |
| cross：全期望 | 1 | 4 | 8 | 1 | 14 |
| go | 1 | 1 | 1 | 1 | 4 |

基本预算另有明确边界：模型拓扑的分支数与动作数已经可读取，备份前只检查这些成本元数据。构造静态列表的追加次数、每次执行前的预算检查、模型样本选择都另列；本例没有优先队列，入队和出队均为0。算术、内存分配、概率采样器实现与墙钟时间没有换算成 W。换一种接口、缓存后继最大值或维护别名采样表，会改变实际成本，比较时应重新计费。

<a id="planning-budget-propagation"></a>

### 同样六次期望更新，奖励能否传回当前选择

先固定备份类型，只改变次序。前向列表是 go、cross、C1:collect、C2:collect、C3:collect、C4:collect；后继优先列表把四个 collect 放在前面，再做 cross、go。两者都写同样六个项目，接口查询6次、后果展开9项、后继动作值读取9次、Q写入6次，W=30。列表构造追加6次，执行前预算检查6次，均在基本预算之外另计。后继优先来自这个无环模型的层次关系，与哪片区域奖励大无关。

前向列表的 go 首先读到 cross=0，cross 接着读到四个 C 的0。后来 collect 已变为2、2、0、0，先前写过的 go 与 cross 却不会自动重算。花完30单位后起点仍读 go=0，因此选 exit。后继优先列表先用12单位写四个 collect，再用14单位把 cross 写成0.9，最后用4单位把 go 写成0.81，于是起点改选 go。

![同样六次全期望备份和W30，前向更新后go仍为0，后继优先更新在W26写cross为0.9、W30写go为0.81，从而改变真实首动作。](https://yingwen.io/crl-figures/planning-budget-walkthrough-propagation.svg)

上下两图的坐标与计费相同。青点线是四个C当前最大动作值的均值；紫虚线为cross，橙实线为go，灰线是exit=0.6。阶梯只在写Q时改变；两个日程的六次写入均计费，包括写回0。图下橙箭头为停止计算后的真实首动作。原创精确异步更新，依据 Sutton 与 Barto §8.4、§8.9。

这个比较说明当前决策需要哪条传播链，并未测出一种调度器的普遍速度。两份列表是本例的已给条件，代码没有先免费搜索整张图来发现更新顺序。若要在一般图上自动生成后继顺序，就需计图遍历、排序与环的处理；用优先扫描发现顺序，则需计前驱索引、残差查询与队列操作。前面关闸例已经展示过，计算优先级本身也会读取模型。

预算不足时，完整期望备份的粒度还会产生等待。这里每项备份不可拆分：先看所需 W，余量不足就停止，不写半个目标。B=26时四个 collect 与 cross 都已完成，go 尚未得到4单位，起点仍选 exit；B=29也相同；B=30才会改选 go。剩余单位和未执行项目都保留在输出中。若把宽备份拆成按概率加权的小备份，便改变了可中断的计算机制，需要保存部分和或旧贡献，不能继续套用本表的原子成本。

现在固定后继优先位置，再比较目标类型。先完成四个 collect，然后采样一次 cross，最后做一次 go，W=12+5+4=21。若抽到 C1 或 C2，cross=1.8、go=1.62，起点选 go；若抽到 C3 或 C4，两值为0，起点选 exit。全期望用同样位置顺序，到30单位得到确定的 go=0.81。采样能较早完成一条传播链，同时把模型抽样的波动传给了当前决策。

选择随后真正执行：采样正分支决定 go 后，环境仍可能让 cross 到 C3；这一次实际奖励为0、0、0，三步回报为0。选择 exit 则实际走 S→B，用一步得到0.6。已知精确模型中，选择 go 的下一 episode 期望仍为0.81；一条零回报路线只是这个分布的可能后果。规划期间的真实转移与模型写入都为0，下一段实际执行另计1或3个环境步。

<a id="planning-budget-decision"></a>

### 固定目标的误差下降，怎样变成动作选择

一次模型抽样还不足以可靠比较两个动作。保持四个 C 的 Q 不动，独立查询 cross 共 n 次，每次用步长1/k更新，它就保存前 k 项目标的算术平均；最后只备份一次 go。这样固定了位置顺序、模型与后继值，只改变 cross 的抽样次数。重复 cross 期间 go 未更新，最后那次写入才让起点使用均值。

$$
Y_k\in\{0,1.8\},\quad\Pr(Y_k=1.8)=\tfrac12,\qquad Q_n(A,\mathrm{cross})=\frac1n\sum_{k=1}^{n}Y_k,\qquad Q_n(S,\mathrm{go})=\frac{1.62K}{n},\quad K\sim\operatorname{Binomial}(n,\tfrac12)
$$

$K$ 是 n 次模型抽样中落到 C1、C2 的次数，不是真实环境访问数。冻结目标时均值为0.9、目标方差为0.81；cross均值的方差为 $0.81/n$，go估计的均方误差为 $0.6561/n$。这些量只评价固定模型、固定后继值下的估计。

$$
\Pr(\mathrm{choose\ go})=\sum_{k=0}^{n}\binom nk2^{-n}\mathbf1\!\left\{\frac{1.62k}{n}>0.6\right\},\qquad W_n=4\times3+n\times5+4=16+5n
$$

起点严格比较go与exit，平局选择exit。全部 $4^n$ 条后继序列也可直接枚举，按K合并后得到这个二项分布；独立脚本用组合数检验枚举结果。

| cross抽样数 n | 接口查询 / 后果展开 | 后继Q读取 / Q写入 | W | 列表追加 / 预算检查 | 选择go概率 |
| --- | --- | --- | --- | --- | --- |
| 1 | 6 / 6 | 3 / 6 | 21 | 6 / 6 | 1/2 |
| 2 | 7 / 7 | 5 / 7 | 26 | 7 / 7 | 3/4 |
| 3 | 8 / 8 | 7 / 8 | 31 | 8 / 8 | 1/2 |
| 4 | 9 / 9 | 9 / 9 | 36 | 9 / 9 | 11/16 |

在同一个可用基本预算 B 下，采样列表取能容纳最后一次 go 写入的最大 n，全期望列表仍只做一轮。B=21时采样已能选 go，全期望只完成四个 collect，使用12单位、剩9单位；B=26时采样的 go 概率为3/4，全期望已写 cross但还不能写 go；B=30时全期望完成并必选 go，采样完成 n=2、用26单位且余4单位。少量计算时和足够计算时，比较结果可以不同。

![可用基本预算与真正选择go的概率，分解全期望和一次采样传播成本，并在exit为0.82的近平局任务中显示采样误选的实际期望回报。](https://yingwen.io/crl-figures/planning-budget-walkthrough-decision.svg)

上图给定同一后继优先日程，采样cross用1/k平均，最后备份go；蓝圆实线为 $4^n$ 条序列的精确动作概率，橙方虚线为一轮全期望。B=30时采样留4单位，期望完成。中部条长分别编码查询、展开、后继动作Q读取、Q写入；列表与检查另计，样本选择也另计。下图仅把exit改0.82，纵轴范围0.810–0.821，红线是采样分支下一实际回报的精确平均。全部为原创有限枚举，依据 Sutton 与 Barto §8.5的成本问题；没有随机训练曲线。

为什么把采样数从2加到3，go概率反而从3/4降到1/2？两次抽样中只要一次正目标，go=0.81，已超过0.6；三次抽样中一次正目标只给go=0.54，需要至少两次才过阈值。均方误差从0.32805降到0.2187，而动作阈值改变了估计分布的归类。当前动作概率、价值误差与下一段真实收益是三个不同读数，不能用其中一个的单调变化替代另两个。

在原任务中，计算停止后冻结首动作，下一完整 episode 的期望回报为 $0.6+0.21\Pr(\mathrm{go})$；因此 n=2 给0.7575，n=3给0.705，完整期望规划给0.81。下一实际交互长度也不同，期望为 $1+2\Pr(\mathrm{go})$。这些有限序列平均没有包含后续模型学习或下一轮规划。

再只把出口奖励改成0.82，go的真实期望仍是0.81，正确选择变为 exit。全期望完成后仍选 exit；两次采样只有两个都为正目标时才误选 go，概率1/4，下一实际期望回报为0.8175，低于0.82。若出口恰为0.81，两动作真实价值相同，即使采样换了首动作也没有期望回报增益。模型准确、采样均值无偏，都没有消除接近阈值时的选择误差。

冻结后继值这一条件也决定了平均的含义。把两次采样改成“先采样cross一次，再写四个collect，再采样cross一次，最后写go”，基本工作量仍为26。第一次目标恒为0，第二次才是0或1.8；步长1/2把它们平均，go只可能为0或0.81，均值为0.405，选择go概率为1/2。两个目标的条件均值从0变为0.9，已经不是同一固定随机量的独立重复，不能再用0.81/n解释它对最终cross的误差。

这把原书 §8.5 的局部成本问题接到了 §8.4、§8.6、§8.9 的计算位置问题：先决定在哪些状态动作上更新，再决定每项目标处理多少后果，最后检查信息是否在截止前传到正在选择的动作。本例采用给定无环模型与静态列表，优先扫描、轨迹采样和执行时搜索各自还要解决计算位置怎样产生以及为它付出多少开销。

<a id="planning-dyna"></a>

## 3 · Dyna 的三条更新路径

真实经验可以直接训练 Q，也可以训练模型。模型学得的后果再生成模拟转移，继续使用同一种价值更新。Dyna 不是先学完模型再独立规划，而是把行动、直接学习、模型学习和规划交错执行。

**算法：Dyna-Q 的完整循环**

1. 初始化 Q 与空模型。
1. 每次真实交互：
  1. 按 Q 的探索策略选动作，执行后得到奖励、后继与终止。
  1. 对这条真实经验做一次 Q-learning 更新。
  1. 把后果加入模型。
  1. 重复 `n_plan` 次：
    1. 从已建模的状态动作中选择一个。
    1. 查询模型取得后果，再做一次价值更新。
  1. 继续真实交互；真正终止时按环境协议开始下一 episode。

下面的关闸道路保持确定，模型保存最后一次后果即可。随机环境中，只保存最后样本的模型快照把分布压成单一后果；固定此快照反复查询，无法再现真实随机性。可以保留经验后果并抽样、学习计数分布或条件生成模型。模型随着新样本更新的样本式 Dyna 与固定这份快照反复查询，是不同流程；本章随机扩展选择显式拟合条件分布。模型仍须保存终止标记，真正终点的价值备份只保留到达奖励。

关闸后重置到 S，先按旧值选 go，实际得到 S→A、奖励 0。这一次直接备份仍读取 Q(A,cross)=1，所以 Q(S,go) 保持 0.9；接着 cross 实际到 D、奖励 0，并终止。现在同一条 (A,cross,0,D,done=true) 进入 Dyna 的三条路径：直接备份把 Q(A,cross) 从 1 改为 0；模型学习把 A:cross 的后果由 G 覆盖为 D；规划之后再从更新后的模型生成后果，继续做价值备份。前两条写入完成时，起点的旧估计仍没有自动改变；第三条还要决定查询哪一项已知后果。下一节的前驱调度就是其中一种安排。

$$
y=r+\gamma(1-d)\max_b Q(s',b),\qquad Q(s,a)\leftarrow Q(s,a)+\alpha[y-Q(s,a)]
$$

$d=1$ 表示环境真正终止，这时目标只保留 $r$，程序不读取终点动作值。本例 $\alpha=1$，真实与模拟更新都读取执行这一行时的最新 Q；这是逐项异步赋值，不是冻结整轮 Q 的同步迭代。

![同一S、A、G、B、D道路关闸前后的实际轨迹，以及go、exit、cross三项动作值；cross由1降至0，go仍为0.9。](https://yingwen.io/crl-figures/dyna-planning-walkthrough-sample.svg)

橙线显示这两段实际经过的道路；红叉表示A到G关闭。柱长是当前Q，终点双圈。三条旧记录和一次模型备份得到旧Q；新两步记录使cross降为0并更新模型，S的go仍等待规划。数值为本章原创确定性机制计算。

此刻若按固定合法顺序 S:exit、A:cross、S:go 查询模型并备份，前两次目标已经等于旧值，第三次才使 Q(S,go)=0+0.9×0=0。Dyna-Q 的随机调度可能抽到这个顺序，也可能首先抽到 S:go；原书先抽已见状态、再抽该状态下尝试过的动作，一般不等于在所有状态动作对上均匀抽样。本例固定顺序用来查时序，完整随机训练仍在后面的原实现中。

<a id="experiment-dyna_q"></a>

### 实验：同一批真实经验，增加五次模型备份

Dyna怎样在没有再访问上游状态时传播新奖励？这五次更新能否算成五条额外真实经验？

**环境与可用信息。** 六格确定性链，状态0—5完全可观测；从0开始，左移不越过0，右移到5得到1并终止，其他转移奖励−0.01。终止后从0重开。这张实测图用γ=0.95，与正文手算两步例子的参数分开。

**设置。** 5个种子，各1200个真实转移。Q零初始化、模型初始为空，Q步长0.2，ε=0.1。每步先做真实Q-learning，记录确定性后果，再从已见状态—动作中均匀抽样做5次备份。对照相同任务的Q-learning，无模型备份。

**检验的机制。** 模型记忆已观察的奖励、后继与真正终止；规划使用当前Q重新评价旧后果。因此新的下游价值可传到尚未再次真实访问的上游。想象不会增加模型对未知世界的观测。

**测量。** 图为从0执行冻结确定性贪心策略的精确折扣回报；循环用几何级数计入，而非事后强制结束。另记模型备份数，避免将计算量藏在环境步横轴之后。

```bash
python3 implementations/classic/dyna_q.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-figures/result-dyna_q.svg)

横轴：environment_steps。纵轴：确定性贪心策略折扣回报。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 冻结当前 Q，从六格链的起点0按确定性贪心动作计算折扣回报（γ=.95）。并列时评价选动作0，训练则在并列最优动作间均匀分配利用概率。循环路径的尾项用几何级数计入。

**step：怎样计时。** step 是真实转移次数，包含到达终点的那一步。Dyna 每步直接更新一次 Q，再做5次模型更新；优先扫描先写模型和队列，Q 只在出队时更新，每步至多5次。model_backups 不含排队、查找前驱和计算优先级的成本。

**怎样汇总。** 取该记录时刻的值；不先对曲线上的时间点求平均。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** CSV 可重算冻结贪心回报曲线及模型更新次数。Dyna 的 Q 更新总数为 step + model_backups；本优先扫描实现的 Q 更新总数就是 model_backups。日志没有逐步行为奖励或 Q 表，不能恢复训练累计收益或每个检查点的策略。

计算位置：[classic/dyna_q.py](https://yingwen.io/crl-code/implementations/classic/dyna_q.py) · [classic/prioritized_sweeping.py](https://yingwen.io/crl-code/implementations/classic/prioritized_sweeping.py) · [classic/q_learning.py](https://yingwen.io/crl-code/implementations/classic/q_learning.py) · [classic/_common.py](https://yingwen.io/crl-code/implementations/classic/_common.py)

</details>

**结果分析。** 首个15步检查点，Dyna平均回报0.5819，纯Q-learning为−0.2；到615步二者均为0.7774075。差别主要出现在样本有限的早期，不是更高的最终最优值。

**结论边界。** Dyna额外做6000次模型备份，没有等计算预算或变化检测。只记最后一次后果的模型在此确定性环境成立，不能不加修改地用于随机转移。

**继续实验。** 先将规划次数设0，逐步核对与相同Q-learning实现的目标。随后比较相同真实步与相同总备份两张图。若加入随机滑动，先修改模型表示再运行。

[源码](https://yingwen.io/crl-code/implementations/classic/dyna_q.py) · [逐种子记录](https://yingwen.io/crl-code/results/dyna_q/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/dyna_q/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/dyna_q/curves.json)

<a id="planning-priority"></a>

## 4 · 优先扫描与前驱传播

一个终点奖励首次被发现后，最值得更新的往往是能到达它的前驱，而不是所有已知状态均匀抽样。优先扫描记录模型的前驱关系，用预期 backup 改变量作为优先级，弹出高优先级项更新，再检查它的前驱。

$$
P(s,a)=\left|\hat r(s,a)+\gamma\sum_{s'}\hat p(s'\mid s,a)\max_bQ(s',b)-Q(s,a)\right|
$$

这里继续使用本章 Q-control 的贪心后继价值；终点值为零。优先级衡量当前模型 target 与旧估计的差，不是动作真实价值或样本不确定性的估计。

**算法：原书§8.4流程；额外重算用于处理过时项目**

1. 原书的确定性优先扫描：
  1. 收到真实 (s,a,r,next,done)，先更新模型及前驱索引。
  1. 用真实后果和当前 Q 计算 |target−Q(s,a)|，必要时入队。
  1. 此处不额外做一次直接 Q 更新。
1. 在最多 n 次弹出的预算内：
  1. 弹出最高优先项，读取当前模型后果并重算误差。
  1. 误差已很小则跳过；否则对它做 Q backup。
  1. 读取能到达该项目起始状态的前驱后果。
  1. 重新计算前驱优先级，必要时入队或提高优先级。

模型改变时前驱图也要更新，例如移除不再成立的确定性边。队列里重复或过时的优先级必须处理；队列算法正确不意味着模型正确。随机转移时，后继变化对前驱的影响还受转移概率加权。

先按原书的流程处理同一关闸样本。模型覆盖后，A:cross 的残差为 |0−1|=1，入队但 Q 尚未改变。第一次弹出 A:cross 后把它改为 0；读取前驱 S:go→A，算出 |0+0.9×0−0.9|=0.9 并入队。第二次弹出把 Q(S,go) 改为 0，队列为空。这里是 2 次模型价值备份、2 次队列弹出和 3 次模型后果读取：其中一次读取只计算前驱优先级。初始优先级直接使用刚收到的真实样本，不收取模型调用。

![原书优先扫描的三个时刻：cross残差1入队，弹出后go残差0.9入队，再弹出go后队列为空，S转选exit。](https://yingwen.io/crl-figures/dyna-planning-walkthrough-queue.svg)

从左到右或手机从上到下，条长表示队列优先级，图中数值显示当时Q。紫虚线为模型的前驱关系，橙箭头为起点最终选择。所有模型读取计费，包含前驱优先级；原书此流程没有先做直接备份。原创精确算例。

若保留 Dyna 的先直接更新流程，再使用优先队列，就需要改调度入口：直接备份已经把 cross 改成 0，这时只检查它自身会得到零残差，无法触发传播。本例混合实现显式检查 A 的前驱，读取 S:go 的模型得到优先级 0.9；再弹出、读取并备份 S:go，合计 2 次模型读取、1 次模型备份。原书流程与这个混合流程最终值相同，但直接备份数和模型读取数不同，不能把二者都记成同一个“每步规划次数”。

前驱索引保存的是哪些状态动作可以到达某状态。关闸后应从 G 的前驱中移除 A:cross，并把它加到 D 的前驱；S:go 仍是 A 的前驱，所以这条传播路线保留。维护索引、队列查找和堆操作也需要时间；下面单列后果读取，尚未把这些操作折算成相同墙钟成本。

<a id="planning-search"></a>

## 5 · 从后台规划到执行时搜索

后台 Dyna 更新长期保留的价值表，执行时可快速查表。执行时规划则围绕当前状态花计算：rollout 比较候选首动作后，沿某个默认策略模拟；MPC 规划有限动作序列，执行首动作后用真实观测重新规划。

MCTS 维护当前搜索树，通常循环选择、扩展、模拟或叶值评价、回传统计四步。UCT 用访问统计权衡树上探索与利用，它借鉴 bandit，但树节点的回报分布会随搜索策略和子树增长变化，不能直接把独立平稳老虎机假设照搬。

$$
a\in\arg\max_b\left[\widehat Q(s,b)+c\sqrt{\frac{\log N(s)}{N(s,b)}}\right]
$$

这是常见 UCT 选择形式；未访问边单独优先处理。最终执行哪个根动作、是否保留搜索树、叶值来自 rollout 还是网络，仍是算法定义的一部分。

搜索必须拥有可查询的模型或仿真器。能任意恢复仿真状态，不等于真实机器人拥有这种 reset 权限。策略价值网络、模型学习与搜索可组合，但搜索本身不能验证模型外推到未见区域是否正确。

<a id="lesson-example"></a>

## 6 · 手算：奖励的反向传播

现在回到关闸后的 S。episode 重置是任务给定操作，不作为模型中的一步转移。直接学习而不规划时 Q(S,go)=0.9>0.6，仍走 S→A→D，实际得到奖励 0、0；到 A 的这次新直接更新才将 go 降为 0。已经把变化传到 S 的规划分支会改选 exit，实际走 S→B 得 0.6，并把这条新经验继续用于直接学习和模型学习。估计改变行动，行动又改变接下来的样本。

![模型读取次数与S的go值的离散变化，以及零规划和规划两条分支随后实际执行的不同轨迹。](https://yingwen.io/crl-figures/dyna-planning-walkthrough-loop.svg)

上方保持关闸后的两步真实记录相同，阶梯只在执行备份时改变。固定顺序在第三次读取改go；先直接更新再检查前驱的混合实现用两次读取改go。下方是episode重置之后真正执行的路线：零规划两步得0、0，规划一步得0.6。γ=0.9、α=1；这是确定性机制闭环，不是算法性能实验。

| 读完相同两步新经验后 | 直接备份 | 模型备份 | 模型读取（含优先级） | 下一实际episode |
| --- | --- | --- | --- | --- |
| 零规划 | 2 | 0 | 0 | go；2步，回报0 |
| Dyna固定顺序exit,cross,go | 2 | 3 | 3 | exit；1步，回报0.6 |
| 直接更新后检查前驱 | 2 | 1 | 2 | exit；1步，回报0.6 |
| 原书优先扫描 | 0 | 2 | 3 | Q选择exit；本行只核对规划 |

表中真实预算都是关闸发现的 2 步、2 次模型写入和 1 次重置，旧阶段的 3 步、1 次备份与 1 次重置另记。前三行下一episode再增加一次重置，但实际交互长度不同：零规划的变化后累计为 4 步，规划分支为 3 步。相同真实样本能比较信息怎样传播；相同备份数能比较调度；相同总计算预算还要纳入优先级读取与队列维护。此表只核算给定道路和日程，没有独立随机重复或算法速度比较。

把两出口压成这个小图，是为了直接看清更新次序。下面保留较长链上的反向传播与已有随机Dyna实验，检验传播距离增加以后还有哪些工作。

四个非终止状态 0→1→2→3→终点，只有最后一步奖励 1，γ=.9，初值全零。已知确定性模型下，最初只有状态 3 的误差为 1。更新它后，状态 2 的误差变 .9，然后状态 1 为 .81，状态 0 为 .729。

| 第几次 backup | 更新状态 | 新值 |
| --- | --- | --- |
| 1 | 3 | 1 |
| 2 | 2 | .9 |
| 3 | 1 | .81 |
| 4 | 0 | .729 |

真实 Dyna 实验允许左右动作，从状态 0 出发，左边界停在 0。探索 ε=.5，4000 次真实交互；每步规划 5 次时增加 20000 次模型更新。向右价值接近 (.729,.81,.9,1)。这个最终一致性不是“规划比无规划更快”的充分证据，还需要整个学习曲线与计算对照。

<a id="lesson-code"></a>

## 7 · 可运行模型循环

下载独立单文件 [planning-budget-walkthrough.py](/crl-code/tutorials/planning-budget-walkthrough.py)，只需 Python 3 标准库，可从任意目录运行。`--json` 输出两个更新次序、逐步Q快照、每项计费、未花完的预算，以及固定目标与移动目标的有限枚举；`--test` 另用完整奖励路径求和和二项组合数核对，包含近似平局、精确平局、零折扣与预算不足。代码计算本章原创分支机制，没有启动随机训练。

先检验预算与值传播，再读动作分布

```sh
python3 planning-budget-walkthrough.py --test
python3 planning-budget-walkthrough.py --json
```

先看 `order.forward` 和 `order.backward`：两者 `counts.work=30`，但 `q["S:go"]` 分别为0与0.81。`frozen[1]` 是两次cross抽样，工作量26、选择go概率0.75；`moving` 同样工作量26，均值却变为0.405。`nearTie[1].meanTrueReturn=0.8175` 保留出口0.82时的误选损失。各行 `counts.environmentTransitions=0` 与 `modelWrites=0` 说明这些是计算调用；下一真实episode的1步或3步另在正文核算。网页图片的数据由 [planning-budget-walkthrough.mjs](/crl-code/figures/planning-budget-walkthrough.mjs) 枚举，并与独立 Fraction 脚本交叉核对。

随机扩展的独立单文件为 [stochastic-model-walkthrough.py](/crl-code/tutorials/stochastic-model-walkthrough.py)。只用 Python 3 标准库，从任意独立目录运行；`--json` 输出逐条真实记录形成的模型前缀、共同快照的三条备份分支、全部16种四次后果，以及行动之后的新经验。`--test` 用精确分数检查平局，并独立按真实奖励路径求和核对价值。

先运行随机模型的精确机制与枚举，再比较完整训练

```sh
python3 stochastic-model-walkthrough.py --test
python3 stochastic-model-walkthrough.py --json
```

先看 `prefixes[1].distribution` 的两条联合后果各占1/2，再看 `frozen.branches.expected.counts` 的 `modelQueries=2`、`outcomeEvaluations=3`、`planningBackups=2`、`environmentSteps=0`。`afterFour.counts.environmentSteps=9` 与 `afterFour.n=4` 是不同计数。`closure.after.rewardMean=0.6`、`closure.after.q` 中 `S:go=0.54` 显示下一真实失败如何再次改变行动；`enumeration.wrongChoiceProbability=67/256` 保留小数据错误选择。网页图与 JavaScript 核逐项对照独立 Fraction Python；脚本未启动随机训练。

下载单文件 [dyna-planning-walkthrough.py](/crl-code/tutorials/dyna-planning-walkthrough.py)，放入任意独立目录运行。Python 3，仅标准库；--json 输出旧Q、各调度的计数、下一段实际经验与最终Q，--test 用分数运算、独立奖励路径求和及退化条件核对。它实现本节确定性Dyna/优先扫描机制，原有随机训练入口继续保留。

独立目录中的确定性主例

```sh
python3 dyna-planning-walkthrough.py --json
python3 dyna-planning-walkthrough.py --test
```

先比较 scenarios.hybrid.counts 的 modelCalls=2、planningBackups=1；再看 nextExperience.firstAction=exit、return=0.6，以及 countsAfterNext.environmentSteps=3。--test 还检查模型前驱替换、过时队列不重复备份、未建模查询拒绝和 γ=0。网页图由独立 JavaScript 计算同例，并与标准库 Fraction 输出逐项交叉核对。

Dyna 的真实交互与模拟更新计数，以及含过时优先级检查的反向扫描。

```python
def dyna(planning_steps=5, real_steps=4000, seed=7):
    """Deterministic chain: a last-outcome model is exact after observation."""
    rng = random.Random(seed)
    q = {s:[0.,0.] for s in range(4)}
    model, state, simulated, terminals = {}, 0, 0, 0
    def update(s, a, reward, sp):
        target = reward+(.9*max(q[sp]) if sp is not None else 0.)
        q[s][a] += .2*(target-q[s][a])
    for _ in range(real_steps):
        action = choose(epsilon_probs(q[state], .5), rng)
        position = max(0, state+(1 if action else -1))
        reward, sp = float(position==4), None if position==4 else position
        update(state, action, reward, sp)
        model[state,action] = reward,sp
        for _ in range(planning_steps):
            s,a = rng.choice(list(model))
            r,sn = model[s,a]
            update(s,a,r,sn)
            simulated += 1
        terminals += int(sp is None)
        state = 0 if sp is None else sp
    return {"right_values":[q[s][1] for s in q], "reference":[.729,.81,.9,1],
            "real_steps":real_steps,"model_updates":simulated,"episodes":terminals}

def prioritized_chain():
    """Exact one-action model: changed terminal reward launches a reverse wave."""
    model = {s:(float(s==3),None if s==3 else s+1) for s in range(4)}
    predecessors = {s:[] for s in model}
    for s,(_,sp) in model.items():
        if sp is not None:
            predecessors[sp].append(s)
    values,heap,sequence = {s:0. for s in model},[],[]
    def target(s):
        reward,sp = model[s]
        return reward+(.9*values[sp] if sp is not None else 0)
    for s in model:
        error = abs(target(s)-values[s])
        if error>1e-12:
            heapq.heappush(heap,(-error,s))
    while heap:
        _,s = heapq.heappop(heap)
        if abs(target(s)-values[s])<=1e-12:
            continue  # stale priority
        values[s] = target(s)
        sequence.append(s)
        for predecessor in predecessors[s]:
            error = abs(target(predecessor)-values[predecessor])
            if error>1e-12:
                heapq.heappush(heap,(-error,predecessor))
    return {"backup_order":sequence,"values":values}
```

planning 模式分别运行零规划与每步五次规划，再输出优先扫描顺序 [3,2,1,0]。源码范围是表格 Dyna 和小链上的优先扫描；MCTS、世界模型与机器人控制还需要相应的搜索结构、模型接口和环境约束。

规划和行为当前使用同一随机数发生器，因此两个规划预算消耗随机数的顺序不同；它们不是严格共同随机数的配对实验。比较统计性能时，应先决定共享哪些环境随机事件、哪些采样过程独立，并相应构造随机流。

<a id="lesson-branches"></a>

## 8 · 模型错误与持续变化

未访问的后果不会因增加规划次数被自动发现。若路径突然堵塞，旧模型可能反复建议已失效的路线；如果新通道出现但从不尝试，也不会更新模型。Dyna-Q+ 在规划奖励中加入久未实际尝试的探索奖励，是主动重新检验模型的一种思路。

$$
\tilde r=\hat r+\kappa\sqrt{\tau(s,a)}
$$

τ 表示距离上次真实尝试的时间，不能每次模拟调用就清零。这是探索启发式，改变用于规划的奖励，不是对真实环境奖励的无偏估计；本节代码未实现该扩展。

Dyna-Q+ 还允许在已见状态上规划尚未实际尝试的动作，通常暂把它们建模为零奖励、自循环。若规划表只含已尝试的状态动作，给表中项目增加久未尝试奖励也无法直接重估表外的新选择。这个默认模型是待真实经验检验的假设；时间奖励促使行为重新尝试，真实后果才提供新证据。

持续规划要平衡新信息获取与已有信息利用，另需限制模型大小、旧知识失效和每步计算。更长模拟并不总更好，因为错误可以积累并被控制器利用。模型准确性、规划计算误差与真实控制收益应分开评估。

本例也能隔离错误来自哪里。只直接把 cross 改为 0，却保留旧模型 A→G、奖励 1，下一次模型备份会把 cross 恢复为 1，再把 go 恢复为 0.9；新经验与旧模型相互拉扯。已经改对模型却不备份 S，则 go 仍为 0.9，是计算尚未传播。若新捷径从未尝试，则模型中没有相应后果，是事实尚未获得。分别冻结模型、冻结价值传播、规定尝试日程，才能辨认这三种原因。

<a id="lesson-check"></a>

## 9 · 带答案练习

问：一个随机模型以 .75 概率 target=1、.25 概率 target=5，期望 backup 是多少？答：2。单次采样只能看到 1 或 5，但固定 Q 下其期望为 2。

问：把模拟步数算进环境样本数会怎样？答：混淆信息来源和计算成本。应单列真实交互、模型训练、模型调用与 backup 次数。

问：未发现终点奖励时，给零奖励旧模型做百万次规划能学会正确路线吗？答：一般不能。计算无法替代缺失事实；需要实际探索或额外先验。

问：关闸样本已直接把 cross 改为0，只把当前pair的非零残差入队，会发生什么？答：它自身残差为0，队列空，S仍选go。应在直接更新造成后继价值变化后检查前驱，或完整采用原书先模型、入队、再更新的流程。

问：对 S:go 做一次模型备份为何能改变下一条真实经验？答：目标读取当时已为0的cross，把go从0.9改为0，起点改选0.6的exit，于是下一episode实际观察S→B，替代S→A→D。若α=0或γ=0，分别没有价值写入或没有延迟价值传播；代码检查这两个退化情形。

问：cross只观察到D、G以后，做百万次模型期望备份，G计数会增加吗？答：仍为1、总真实计数仍为2。计算可以精确得到当前模型的0.5，却不提供新后果证据；新真实访问才更新计数。

问：四次cross后错误选择exit的概率为何不是1/4？答：1/4是一次真实失败的概率；错误选择取决于四次计数中G不超过2，其概率为67/256。两个事件属于不同对象。

## 从本章进入实践

[技能与规划](https://yingwen.io/zh/continual-rl/code/#practice-skills)：一个多步行为怎样成为可学习、可预测、可规划的动作？



<a id="chapter-code"></a>

## 下载与运行

Python 3，仅标准库。下载完整文件后在其目录运行；页面展示核心区域。小环境和解析检验用于理解机制，不是论文基准复现。

[下载 tabular_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/tabular_textbook_lab.py)

```sh
python3 tabular_textbook_lab.py planning
python3 tabular_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：对应第 2–8 章。以下推导、例子和标准库实现独立编写；原书用于核对定义、算法关系与适用条件。

- [David Silver · UCL 强化学习课程](https://davidstarsilver.wordpress.com/teaching/)：原课程页面提供 MDP、动态规划、预测、控制、探索和规划的讲义及视频。

- [Sutton · Dyna: An Integrated Architecture for Learning, Planning, and Reacting](https://doi.org/10.1145/122344.122377)：真实学习、模型学习和规划的原始架构。

- [Sutton (1990) · Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming](https://mlanthology.org/icml/1990/sutton1990icml-integrated/)：§1–2给出交互/模型共享更新的架构，§4–5区分blocking与shortcut并展开Dyna-Q；本章采用原书§8的确定性表格顺序。

- [Moore & Atkeson (1993) · Prioritized Sweeping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa15/optreadings/MooreAtkeson-1993-prioritized-sweeping.pdf)：§2与图3解释有限计算间隔、前驱与队列；原论文优先级形式和原书确定性Q版不同。本章队列主例核对的是原书§8.4。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-tabular-planning#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-tabular-planning#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-tabular-planning)
<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 模型需要预测什么，内部计算应花在哪里？

规划根据模型更新价值或选择动作。Dyna 把真实经验更新、模型拟合和规划查询分开。模型更新次数更多，不代表新增了真实证据。

函数逼近与深度方法：线性价值可使用期望特征模型；非线性价值通常不能把后果分布替换成均值。Option 模型还要保留随机持续时间、真实奖励和终点的关系。

持续学习中的研究问题：当表示、技能或环境改变时，哪些旧模型仍可复用？有限计算应优先用于收集真实经验、改进模型，还是在已有模型中规划？

[Dyna 与搜索控制](https://yingwen.io/zh/continual-rl/foundations/tabular/planning/) → [神经模型与规划](https://yingwen.io/zh/continual-rl/foundations/deep/model-based/) → [Option 后果模型](https://yingwen.io/zh/continual-rl/construction/models/) → [规划与计算分配](https://yingwen.io/zh/continual-rl/construction/planning/)


### 可进一步检验的问题

- [08 · 什么样的时间抽象值得加入技能库？](https://yingwen.io/zh/continual-rl/research/#research-useful-options)：区分实际执行、模型预测与规划 backup 的用途，才能评价一个长行为节省的经验或计算是否抵偿其维护成本。
- [11 · 什么时候值得规划，应该把计算花在哪里？](https://yingwen.io/zh/continual-rl/research/#research-planning-budget)：前驱传播、优先扫描与执行时搜索把计算用于不同位置，由此可问下一次模型查询的净收益。
- [15 · 状态、知识、技能与规划怎样共同构成持续智能体？](https://yingwen.io/zh/continual-rl/research/#research-integrated-architecture)：Dyna 中同一经验的直接学习、模型学习和规划三种用途，构成检查模块接口与相互影响的最小闭环。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [Dyna 与模型学习](https://yingwen.io/zh/continual-rl/algorithms/dyna/)
- [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)
- [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)
- [持续学习的智能体架构](https://yingwen.io/zh/continual-rl/construction/architectures/)

对应原始材料：8.1–8.11。本文为原创讲解，原书、论文与上游代码保留各自许可。
