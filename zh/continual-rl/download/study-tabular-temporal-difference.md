# TD 预测与控制：SARSA、Expected SARSA、Q-learning 和 Double Q

表格强化学习 · 第 5 章

如何在完整回报尚不可用时学习？TD 用下一预测补足未来；控制算法再根据不同的下一动作处理方式，形成不同的学习目标。

## 本章内容

- 从 Bellman 样本推导 TD(0)。
- 区分四种控制 target、行为策略和更新顺序。
- 理解 maximization bias、Double 选择评价分离及表格收敛条件。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [Monte Carlo：完整回报、探索控制与离策略评价](https://yingwen.io/zh/continual-rl/foundations/tabular/monte-carlo/)：知道完整回报提供什么信息，才能比较等待终止与立即自举。
- [动态规划：评价、改善与最优递推](https://yingwen.io/zh/continual-rl/foundations/tabular/dynamic-programming/)：理解Bellman递推如何用后继价值表达当前价值。


### 概率与期望

概率非负且总和为一；期望是按概率加权的平均，不是单次结果。条件期望只比较满足已知条件的情形。

### 递推与赋值

递推通过已有估计计算新估计。下标表示时间；更新符号表示程序中的赋值，不是数学恒等式。

<a id="problem-definition"></a>

## 本章的问题定义

模型未知，只收到逐步转移。既要定义预测目标，也要说明控制更新采用哪一种未来行为。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 转移 $(S_t,A_t,R_{t+1},S_{t+1})$；策略评价时固定 $\pi$，控制时可更新行为。

### 需要求解的对象

预测时估计 $v_\pi$；控制时学习动作价值并生成更好的行为。

### 信息与数据权限

每次更新只需当前转移和已有估计；没有真实的全部未来回报。

$$
v_\pi=T_\pi v_\pi,\qquad q_*=T_*q_*,\qquad \delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)
$$

前两式分别定义固定策略与最优控制的目标；TD 残差是利用一步样本构造的更新信号。残差不是外部目标的定义。

### 成立条件与解的含义

- 经典收敛结论要求平稳表格问题、充分访问以及合适步长。
- SARSA 的控制收敛还需规定探索随时间的变化；固定探索策略评价不是最优贪心控制。

判断准则：在有解析解的小 MDP 检查目标固定点，再在控制中报告实际行为收益和覆盖。

### 适用边界

- 由一次 TD loss 降低推出策略收益提高。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：本章的控制部分限定为平稳充分状态和独立表格参数；一般历史依赖学习器不必满足这些条件。共享神经表示不是表格方法的前提。

- 组合不同学习问题 · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)：本章的一步预测与控制可接入多步或资格迹的反馈传播机制，而不重新定义外部控制目标。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

完整未来尚不可用；控制还可能用带噪声的估计同时选择和评价动作。

### 本章的核心思路

用后继预测自举，并明确区分下一动作采样、策略期望与最大化。

1. [用一步样本估计 Bellman 备份](#lesson-derive)：自举允许立即更新，但目标也依赖正在学习的估计。

2. [由未来行为确定控制目标](#td-control-targets)：SARSA、Expected SARSA 与 Q-learning 的差别在后继动作的处理，不只是代码写法。

3. [拆开动作选择与评价](#td-double)：Double Q 用不同估计器减弱选择噪声造成的最大化偏差。

结论与条件：经典表格保证有访问和步长条件；Double Q 缓解一种偏差，不保证所有有限样本误差更小。

### 相关方法改变了什么

- SARSA / Q-learning：分别跟踪实际下一动作与贪心最优备份。

- Expected SARSA / Double Q：前者积分掉动作采样噪声，后者分离选择和评价噪声。


<a id="lesson-setting"></a>

## 1 · 一条转移能提供什么

如果一项任务要很久才结束，等完整回报就会推迟每次改进。另一方面，智能体可能早已多次到过眼前的后继状态，对它的未来有一定认识。TD 利用这份已有预测补足尚未发生的结果：新经验只提供一步，剩余部分由当前知识提供。于是学习可以逐步发生，但旧预测的误差也进入了新目标。

模型未知，但可观察 $(S_t,A_t,R_{t+1},S_{t+1})$。先固定目标策略 π，环境有限平稳、奖励有界，使用折扣回报或适当终止任务。真实价值满足 $v_\pi(s)=\mathbb E_\pi[R_{t+1}+\gamma v_\pi(S_{t+1})\mid s]$。

若当前后继预测等于真实价值，那么一步奖励加后继预测的条件期望就是当前真实价值。实际后继预测并不准确，TD 仍把它当作目前可用的未来估计。这种 bootstrap 让更新不必等到 episode 结束。

<a id="lesson-derive"></a>

## 2 · TD(0) 的目标与误差

$$
y_t=R_{t+1}+\gamma V_t(S_{t+1}),\quad\delta_t=y_t-V_t(S_t),\quad V_{t+1}(S_t)=V_t(S_t)+\alpha_t\delta_t
$$

先用旧表读取当前和后继预测，再只更新当前表项。真正终止时后继项为零。δ 是单步自举误差，不是未知的真实价值误差。

这与 MC 的增量均值具有相同外形，但监督目标不同。MC 使用完整随机回报；TD 使用实际一步加已有预测。正确价值只要求条件期望中的 TD error 为零，不要求每个随机样本的误差都为零。

$$
G_t-V(S_t)=\sum_{k=t}^{T-1}\gamma^{k-t}\delta_k
$$

这条轨迹恒等式先固定 V，且终点价值为零。展开右侧后，相邻后继价值与当前价值相消；在线每步变化 V 时，需要另外保留误差项，不能原样声称严格等价。

先沿同一条两步轨迹逐次观察，而不只比较回合结束后的曲线。轨迹为 A→B→终点，奖励依次为 0、1，γ=.9，所有初值为零，两种方法都用常数步长 α=.1。每回合各状态只出现一次，因此这里 first-visit 与 every-visit MC 相同。

![同一条 A→B→终点轨迹的三次观察快照，对照 TD 当前更新的状态与 MC 等待完整回报的状态。](https://yingwen.io/crl-figures/concept-core-update-clocks-mc-td.svg)

从上向下读观察顺序，每一行的两种方法接收相同转移。圆下数字是本次处理后的估计，青色圈标出更新对象；橙箭头表示构造目标的方向，不是环境倒行。依 Sutton 与 Barto §6.1 式 (6.1)–(6.2) 原创计算；[逐步计算代码](/crl-code/figures/core-update-clocks.mjs)，完全确定，无随机种子。

第一步只观察到 $A\to B$ 与奖励零。TD 读到此刻 $V(B)=0$，所以目标为 $0+0.9\times0=0$，更新 A 的增量也为零；它已经执行了一次更新，只是数值没变。MC 还不知道本回合的完整回报，因此暂存经验，不更新 A 或 B。

第二步收到奖励一并真正终止。TD 只把刚离开的 B 从零改为 $0+0.1(1-0)=0.1$；先前的 A 不会因此自动再更新。MC 此时得到 $G(B)=1$、$G(A)=0+0.9\times1=0.9$，才分别把 B 改为 0.1、A 改为 0.09。MC 的这次 A 更新用的是完整奖励序列，不是刚刚更新过的 B。

第二回合再次走到 B，TD 才读到上回合保存的 $V(B)=0.1$，使 $V(A)=0+0.1(0+0.9\times0.1-0)=0.009$。这一刻 MC 仍保留上回合的 0.09 和 0.1，等待新的终点。下面将链延长，并改用步长 0.25，观察同一种更新顺序怎样累积成整条学习曲线。

![同一确定性五步链、同一回合序列上真正运行 MC 与 TD(0)，绘出起点估计随 24 回合变化的曲线和真实值虚线。](https://yingwen.io/crl-figures/learning-classic-mc-td.svg)

原创实际更新数据：5 个非终止状态依次前进，奖励 0,0,0,0,1；$\gamma=0.9$，$\alpha=0.25$，$V_0=0$，24 回合/120 次转移，每回合从 $S_0$ 重启。MC 与 TD 使用相同经验，基线为精确值；过程完全确定，种子不适用。

把奖励再推迟到第五步，可看清更新顺序。该链只有一个动作，状态 $S_i$ 的真实价值为 $0.9^{4-i}$，所以起点值是 0.6561。第一回合 MC 等到终止，再把起点目标 0.6561 乘步长 0.25，得到 0.164025；TD 访问起点时，后继估计仍为 0，所以起点不变，直到末步才把 $S_4$ 改为 0.25。

第二回合访问 $S_3$ 时，TD 第一次读到非零的 $S_4$，得到更新 $0.25\times0.9\times0.25=0.05625$；这时 $S_2$ 已经在本回合更早被访问过，仍来不及使用它。因此非零值每回合至多向前传播一个状态，第 5 回合才影响起点。图中每个点都是执行这些更新得到的结果，不是按预期趋势手画的线。

24 回合后，起点 MC 约 0.6554，TD 约 0.4943；它们都在估计同一个 0.6561。这里没有奖励噪声、没有动作选择、每回合还允许回到同一起点，所以差距隔离的是零初始化与自举传播的有限预算效应。随机长回合中 MC 的方差、TD 已有尾值的作用及控制收益仍须另做实验。

这组图的逐步数据可[下载为 JSON](/crl-figures/learning-classic-data.json)。在网站源码仓库运行 `node scripts/generate-crl-learning-classic.mjs` 可重新计算并生成图；加 `--check` 只核对现有图与数据是否一致。

但“MC 第一遍传播更远”不等于 MC 在给定数据上总是更好。先分开三个问题：真实环境的价值是多少；有限数据支持哪个估计；采用什么更新顺序计算这个估计。即使使用同一批数据，MC 和 TD 也会因利用序列结构的方式不同而收敛到不同答案。

| 固定数据，γ=1，终点价值为0 | 观察到的回合轨迹 | 出现次数 |
| --- | --- | --- |
| 轨迹甲 | A → B → 终点；奖励依次为0、0 | 1 |
| 轨迹乙 | B → 终点；奖励为1 | 3 |

重复使用这四条回合轨迹。批量 MC 对每个状态的已观察完整回报做最小二乘：A 只出现一次，回报为0；B 的四次回报为0、1、1、1。因此 MC 得到 V(A)=0、V(B)=3/4。注意：这在评价有限样本回报拟合，不是已经知道真实环境中 A 的价值为0。

$$
\begin{aligned}0&=V(B)-V(A),\\0&=(0-V(B))+3(1-V(B)),\\V_{\rm TD}(A)&=V_{\rm TD}(B)=3/4.\end{aligned}
$$

批量 TD 每轮先用同一张旧表计算全部 TD 增量，再累加更新。足够小步长下，其极限让每个状态的总 TD 增量为零。A 总到 B，TD 因而利用了在其他回合中获得的 B 的信息。

还可以把数据变成经验模型：A 必到 B，B 必终止，B 的平均奖励为3/4。这个经验模型的 Bellman 解恰是上述 TD 解。这称为对经验 Markov 模型的确定性等价估计。它没有证明经验模型就是真实模型；若 A 后的 B 与直接从 B 开始的回合有不同隐藏情境，合并它们反而可能有偏。

同一有限数据集上的批量 MC 与 TD；Python 3，可独立运行

```python
returns_A = [0.0]
returns_B = [0.0, 1.0, 1.0, 1.0]
mc_A = sum(returns_A) / len(returns_A)
mc_B = sum(returns_B) / len(returns_B)
td_A = td_B = 0.0
for _ in range(2000):
    delta_A = td_B - td_A
    delta_B = sum(r - td_B for r in returns_B)
    td_A, td_B = td_A + 0.1 * delta_A, td_B + 0.1 * delta_B
assert abs(td_A - 0.75) < 1e-10
assert abs(td_B - 0.75) < 1e-10
print("MC:", mc_A, mc_B, "batch TD:", td_A, td_B)
```

这也给出通向函数逼近的关键问题。表格只在相同状态间共享后续经验；函数逼近还会把不同状态绑定到同一参数。MC 的最小二乘目标于是变成投影真实价值，TD 的平衡条件变成投影 Bellman 固定点。进入预测章时，需要比较这两个方程，而不能把所有“误差乘步长”的更新都当作同一个回归算法。

<a id="experiment-td0"></a>

### 实验：实验 · 能立即更新，不等于有限预算内必然更准确

TD 每条转移都学习，MC 必须等到结束；为什么这组 MC 曲线反而更低？

**环境与可用信息。** 五个非终止状态的等概率左右随机游走。每回合从状态 3 开始，到左端奖励 0、右端奖励 1，γ=1。状态真值依次为 1/6 到 5/6，表格估值全零。

**设置。** 五种子各运行 1200 个真实转移，固定策略使两方法可收到相同轨迹。TD(0) 每步以 α=0.1 更新当前状态。对照保存完整回合，以每状态的首次访问回报做样本平均；未结束的预算尾段不产生 MC 更新。

**检验的机制。** TD 用后继的当前估值自举，能及时更新，但零初值下正奖励需要多次访问逐步传回。MC 一旦看到终点，就能给该回合多个首次访问状态提供完整目标。这里还同时改变了固定步长与递减步长。

**测量。** 主图是五个状态相对于解析真值的等权 RMSE。横轴是真实转移，不是完整回合数，也不是总参数更新次数。MC 保存轨迹、等待终点；TD 不需要这段回合缓冲。

```bash
python3 implementations/classic/td0.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-figures/result-td0.svg)

横轴：environment_steps。纵轴：五状态价值 RMSE。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 当前价值估计与真实值 s/6 的误差。在五个非终止状态 s=1,…,5 上均匀平均平方误差后开方；权重不是训练访问频率。

**step：怎样计时。** step 包含全部真实转移，也包含尚未完成的回合。MC 在自然终止后，对该回合首次访问的每个状态写一次回报均值；同回合重复访问不增加该状态的样本数。TD(0) 每个真实步写入；资格迹可同时影响多个状态。相同环境步不表示相同参数写入次数，预算结束也不产生额外终止反馈。

**怎样汇总。** 取该记录时刻的值；不先对曲线上的时间点求平均。 各状态初值均为0。MC 用各状态已完成回合的首次访问计数决定步长；TD(0) 用常数步长0.1，比较不仅改变了是否自举。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** CSV 可重算各记录时刻的跨种子均值、样本标准差和末点误差；没有保存全部预测向量，不能仅凭 value 重新计算状态权重或逐状态误差。 日志未保存回合边界、首次访问计数或逐步价值，曲线平台不能单独区分等待终止、零更新和稀疏记录漏掉变化。

计算位置：[classic/_common.py](https://yingwen.io/crl-code/implementations/classic/_common.py) · [classic/td0.py](https://yingwen.io/crl-code/implementations/classic/td0.py) · [classic/td_lambda.py](https://yingwen.io/crl-code/implementations/classic/td_lambda.py) · [classic/true_online_td.py](https://yingwen.io/crl-code/implementations/classic/true_online_td.py) · [classic/mc_prediction.py](https://yingwen.io/crl-code/implementations/classic/mc_prediction.py)

</details>

**结果分析。** 第 300 步，TD 的平均 RMSE 为 0.2368，MC 为 0.0527；末尾为 0.0465 对 0.0267。结果说明立即更新并不自动带来更低误差，不能把更新时机与有限预算效果视为同一个概念。

**结论边界。** 不同步长规则、零初始化和短随机游走都会影响比较。没有扫描步长，也没有相同 MC 固定步长对照。这些曲线不是 Sutton 与 Barto 原书图的复现。

**继续实验。** 保留相同轨迹，给 TD 和 MC 分别扫描固定步长，再把初值改为 0.5。另记每状态访问数和回合长度，检验早期差异究竟来自等待、自举传播还是初始偏差。

[源码](https://yingwen.io/crl-code/implementations/classic/td0.py) · [逐种子记录](https://yingwen.io/crl-code/results/td0/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/td0/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/td0/curves.json)

<a id="td-control-targets"></a>

## 3 · 控制的三种下一动作

控制保存动作价值 Q，用它改善行为。SARSA 的五个量是状态、动作、奖励、下一状态、下一动作。必须先按行为选择下一动作，再使用它的 Q 更新当前动作，之后真的执行这个已选动作。

$$
y_{\rm Sarsa}=r+\gamma Q(s',a'),\qquad y_{\rm Expected}=r+\gamma\sum_a\pi(a\mid s')Q(s',a),\qquad y_{\rm Q}=r+\gamma\max_aQ(s',a)
$$

SARSA 采样下一动作；Expected SARSA 对指定目标策略求平均；Q-learning 是目标取贪心策略的特例。

让三种目标读取同一份信息。任务有 A、B、C 三个非终止状态和终点 T，位置完全可观察，以下转移确定。奖励在到达后收到，γ=.9；本次从 A 选择 go。先把价值表与行为策略冻结，只比较这一条经验会怎样更新 Q(A,go)，随后再讨论行为随学习改变的控制循环。

| 状态与动作 | 实际后果 | 更新前的 Q | 冻结行为概率 b |
| --- | --- | --- | --- |
| A：go / exit | (B,1) / (T,0) | 2 / 0 | .75 / .25 |
| B：fast / detour | (C,2) / (T,4) | 1 / 5 | .25 / .75 |
| C：finish / quit | (T,3) / (T,−1) | 4 / 0 | .75 / .25 |

这些行为概率由旧 Q 的 ε=.5、两动作均匀探索得到。B 的旧表偏爱 detour，但本次已经选择了探索动作 fast；这个动作仍会执行。要预测下一行为的延续，就读取 fast 的 1；要平均冻结行为，就用 .25×1+.75×5=4；要形成贪心目标，就读取最大值 5。它们都使用 A→B 的奖励1和同一旧表。

![同一 A到B转移，实际选择fast、旧表贪心detour；四种目标及Q(A,go)的一次更新，Double沿相同动作索引从另一张表读取评价。](https://yingwen.io/crl-figures/control-targets-one-step.svg)

先预测：实际探索动作与旧表贪心动作不同时，哪些目标会随本次下一动作改变？蓝圆标实际 fast，橙方标旧表贪心 detour；下方相同价值尺度的空圆与实圆是更新前后。α=.25，三个单表目标为1.9、4.6、5.5；Double 的额外表及角色见下一节。原创冻结数据精确算例，依据 Sutton 与 Barto §§6.4–6.7；[计算数据](/crl-figures/control-targets-data.json)与[独立 Python 核验](/crl-code/tutorials/control_targets_walkthrough.py)，没有随机训练。

| 算法 | 这次的延续及目标 | α=.25 更新旧值2 |
| --- | --- | --- |
| Sarsa | 已选 fast：1+.9×1=1.9 | 1.975 |
| Expected Sarsa，π=b | 行为平均：1+.9×4=4.6 | 2.65 |
| Q-learning | 旧表贪心：1+.9×5=5.5 | 2.875 |

$$
\mathbb E_{A'\sim b(\cdot\mid B)}[y_{\rm Sarsa}\mid A,\mathrm{go},R_1=1,S_1=B,Q]=.25\times1.9+.75\times5.5=4.6=y_{\rm Expected}.
$$

条件中固定了转移、旧 Q 和行为概率，只对下一动作平均。Expected Sarsa 消除了这一个动作抽样产生的方差；后继状态或奖励随机时，环境抽样的方差仍在。

这个目标均值还不是未知真实价值。按任务规则精确计算，冻结行为在 C 的真实价值是 $.75\times3+.25\times(-1)=2$，于是 B 的 fast 真值为 $2+.9\times2=3.8$，B 的行为价值为 $.25\times3.8+.75\times4=3.95$；A 的 go 真值为 $1+.9\times3.95=4.555$。当前 Expected target 的4.6仍包含旧估计的误差。最优延续则在 B 选择 fast、C 选择 finish，A 的 go 真值为5.23；当前 Q-learning target 的5.5也还未等于它。

Expected SARSA 可以 on-policy，也可以行为与目标不同。一步更新已经条件于当前状态动作，环境转移来自相同模型，因此直接对下一目标策略求和即可；不需要给当前样本乘一个当前动作比率来改变这个条件期望。仍然需要相关状态动作有访问覆盖。

固定 ε 的 SARSA 评价含探索的行为，Q-learning 则可一边探索一边学习贪心目标。两种 Q 数值不同不一定意味着某个实现有误。应先比较它们要评价的策略，再比较误差。

Q-learning 为什么能从探索动作学习贪心目标？在给定当前状态动作后，下一环境后果仍服从同一个转移分布。对样本 target 求条件期望，得到对后果求和、在每个后继状态选择最大 Q 的 Bellman 最优 backup。探索负责提供被更新表项的数据，而下一状态的最大值负责定义目标；二者作用不同。

<a id="td-double"></a>

## 4 · 最大化偏差与 Double Q

$$
\mathbb E[\max_a\widehat q(a)]\ge\max_a\mathbb E[\widehat q(a)]
$$

即使各动作估计无偏，最大值会偏向偶然高估的动作。两个真实值均零、估计各独立取 ±1 时，最大值的均值是 .5。

$$
a^*=\arg\max_aQ_1(s',a),\qquad y=r+\gamma Q_2(s',a^*),\qquad Q_1(s,a)\leftarrow Q_1(s,a)+\alpha[y-Q_1(s,a)]
$$

随机选择更新 Q1 或 Q2；当前表负责选动作，另一表评价。行为可依据两表之和进行 ε-greedy。

继续固定上节经验，把旧单表作为 $Q_1$，另外给定 $Q_2(B,\mathrm{fast/detour})=(4,2)$，两表在 A 的 go 都为2。若这次更新 $Q_1$，$Q_1$ 选择 detour，$Q_2$ 在同一 detour 上评价为2，所以目标为1+.9×2=2.8，$Q_1(A,\mathrm{go})$改为2.2；$Q_2$保持不动。若交换角色，$Q_2$选择 fast，$Q_1$评价为1，目标为1.9，只把$Q_2(A,\mathrm{go})$改为1.975。公平硬币选角色的条件目标均值是2.35。不能把“另一表评价”写成 $\max_aQ_2(B,a)$，否则第一种角色会错误读取fast的4。

硬币均值2.35描述被选更新分支的目标。被写入那张表的新值均值为2.0875；若将两表平均作为价值读数，两种角色的更新后平均分别为(2.2+2)/2=2.1与(2+1.975)/2=1.9875，再对硬币平均得到2.04375。旧读数为2，实际平均位移只有单表期望位移的一半。这些均值都条件于给定两张旧表，没有对估计误差或未来训练数据平均。

$$
\mathbb E[Q_2(A^*)\mid Q_1]=q(A^*),\qquad \mathbb E[Q_2(A^*)]=\mathbb E[q(A^*)]\le\max_aq(a),\quad A^* = \arg\max_a Q_1(a).
$$

这里额外假定评价表 $Q_2$ 对每个动作条件无偏、且其误差独立于选取表 $Q_1$。第一式评价的是被选动作；第二式再平均选择错误，不等于声称对最大真实价值整体无偏。

可枚举一个两动作估计问题检查这一步：真实值为(0,1)，每项估计独立加±1等概率噪声，两张表也独立，并列时选第一个动作。单表最大值平均为1.25；第一张表有1/4概率选错，因此被选动作的真实值平均为.75；另一张表对它的评价平均也恰为.75。条件评价无偏与相对最优值低估可以同时成立。配套脚本枚举全部16对表，不抽样。上节给定的两张表仅是数值快照，并未满足或验证这些统计假设。

完全独立的选取误差和评价误差提供直观解释，但实际 RL 中两表共享环境与行为数据，不会保持简单独立。Double Q 缓解选择评价耦合，不保证每次无偏，也可能出现低估。它与使用延迟 target network 的 Double DQN 相联系，但不是同一个实现。

<a id="rlss-afterstate"></a>

## 动作价值之外：何时可以学习 afterstate？

动作价值把当前状态与动作作为输入。有些任务中，动作先造成一个已知的确定后果，然后世界才给出未知的响应。例如，落子后的棋盘已知，接下来对手或随机事件的响应未知。此时可以把已知部分从学习问题中分离，学习动作后状态（afterstate）的价值。它不是把任意下一观察换一个名字。

$$
\widetilde S_t=f(S_t,A_t),\qquad (R_{t+1},S_{t+1})\sim P(\cdot,\cdot\mid\widetilde S_t),\qquad q_\pi(s,a)=u_\pi(f(s,a)).
$$

本节将一次完整的动作—世界响应计作一步，奖励全部在响应时记账。已知映射 f 必须保留决定后续分布的信息；若奖励还含只依赖原状态动作的已知代价，应另外加上，不能丢失。

$$
u_\pi(\widetilde s)=\mathbb E\!\left[R_{t+1}+\gamma\sum_a\pi(a\mid S_{t+1})u_\pi(f(S_{t+1},a))\mid\widetilde S_t=\widetilde s\right].
$$

对世界响应求期望，再对下一动作求期望。这是一次完整决策周期的 Bellman 方程；已知映射 f 本身不额外消耗一步折扣。

如果两对状态动作导向同一个充分的 afterstate，它们共享同一个待学习答案。一般的特征共享是一种近似；这里的合并在所列条件下是精确的。比如两个落子顺序产生同一局面、同一行动方且具有同样的规则状态，后续经验可复用。重复局面规则、已消耗资源或隐藏信息不同，则不能仅凭可见棋盘相同就合并。

**算法：Afterstate TD 的时序；已知动作后果与未知环境响应分开**

1. 先枚举合法动作，计算每个动作的已知 afterstate。
1. 按 afterstate 价值选择动作，并保存实际选择的 afterstate。
1. 执行动作，读取完整世界响应、奖励和下一个决策状态。
1. 若真实终止，target = reward。
1. 否则，按目标策略求下一 afterstate 价值的期望，或在控制中取最大值。
1. 只更新本次选择的 afterstate 价值。
1. 在后续步骤中继续探索，并独立检查充分性假设。

手算：四个动作的已知后果依次为 $(x,x,y,y)$。从 x 出发一步结束，奖励为零或二、各半；从 y 出发总获 0.6。于是 $u(x)=1,u(y)=0.6$。Q 表有四个条目，afterstate 表只需两个。观察第一个动作的后果能更新第二个动作的估值，因为两者对应同一个预测问题。

| 表示 | 共享依据 | 需要检验的条件 |
| --- | --- | --- |
| Q(s,a) | 逐状态动作保存 | 状态足以描述条件后果 |
| Afterstate u(f(s,a)) | 已知动作后果完全等价 | 给定 afterstate 后，原状态动作不再补充后果信息 |
| 神经特征上的 Q | 学习器决定的近似相似性 | 不同样本共享梯度，但目标未必相同 |

连接持续学习：afterstate 可以节省样本，却把一部分结构交给了设计者。若动作效果逐渐改变，已知 f 的假设可能失效；若 afterstate 本身要从经验学得，就同时引入状态构造和模型误差。不能把这种情况下的数据合并仍当作无损压缩。

<a id="td-loop"></a>

## 5 · 完整控制循环与终止处理

**算法：算法伪代码**

1. 初始化 Q（Double 时初始化两张表），并选择行为探索参数。
1. 每次 episode：
  1. 观察初始状态，按当前行为选择动作。
  1. 执行动作，获得奖励和下一状态。
  1. 若真实终止：target 仅为奖励，不查询下一动作。
  1. 否则：
    1. Sarsa 先按旧表选择并保存下一动作，再以该动作构造 target。
    1. Expected / Q / Double 读取规定的期望或选择评价余项。
  1. 更新当前状态动作；Double 随机只更新一张表。
  1. 转到下一状态；Sarsa 执行已保存动作。
  1. 其余算法按声明的行为时序选择下一动作，再继续交互。

不能在形成 SARSA target 后又重新采样实际下一动作，否则记录中的 target 与真实行为序列不再按所述方式对应。预算截断也不能不加区分地当成真实终止。如果训练后想评价纯贪心策略，应在独立评测中明确关闭探索。

上面的同数据对照固定了动作，所以能定位目标构造的差异。完整控制中，更新会改变下一次行为概率；不同动作又带来不同数据。此时应给各算法独立执行其行为循环，并按同样的真实交互预算比较实际奖励、访问次数和探索设置。只看本次哪个 Q 增加更多，无法判断哪个学习器今后获得更多奖励。

<a id="lesson-example"></a>

## 6 · 手算和可运行实现

| 下一状态 Q=(2,0)，r=0，γ=.9 | 规则 | target |
| --- | --- | --- |
| 实际选择第二动作 | SARSA | 0 |
| 目标概率 (.9,.1) | Expected SARSA | 1.62 |
| 贪心目标 | Q-learning | 1.8 |
| Q1=(5,4)，Q2=(1,3) | Q1选第一个，Q2评价 | 0.9；直接 max Q2 则为2.7 |

环境 A 退出得 .2，或零奖励到 B；B 得 1 或 −1 后终止。ε=.1、γ=.9，Expected SARSA 的 A继续值约 .81，Q-learning 和 Double 约 .9；固定步长 SARSA 在目标附近波动，默认末值约 .868319。这个数不是其精确期望，不能要求它逐种子等于 .81。

<a id="lesson-code"></a>

## 7 · 源码检查与改动实验

先下载[control_targets_walkthrough.py](/crl-code/tutorials/control_targets_walkthrough.py)，运行 `python3 control_targets_walkthrough.py --test` 核对10组确定性检查，再运行不带参数的命令读取目标、更新和时序JSON。Python使用精确有理数与冻结TD误差展开，JS使用分支递归，两种计算独立交叉核对。脚本输出的是固定经验诊断；已有完整采样环境入口仍在下面。

真实控制环境、已选下一动作的保留、Expected target，以及随机交换角色的 Double Q。

```python
def td_prediction(episodes=1000, alpha=0.1):
    values = [0., 0.]  # 0 --r0--> 1 --r1--> terminal
    for _ in range(episodes):
        values[0] += alpha*(.9*values[1]-values[0])
        values[1] += alpha*(1.-values[1])
    return values

def control_target(kind, reward, next_values, probabilities=None,
                   next_action=None, gamma=0.9):
    if next_values is None:
        return reward
    if kind == "sarsa":
        tail = next_values[next_action]
    elif kind == "expected":
        tail = sum(p*v for p, v in zip(probabilities, next_values))
    elif kind == "q":
        tail = max(next_values)
    else:
        raise ValueError(kind)
    return reward+gamma*tail

def td_control(kind, episodes=6000, seed=7, alpha=0.1, epsilon=0.1):
    rng = random.Random(seed)
    q1 = {s: [0.]*len(CONTROL_MODEL[s]) for s in CONTROL_MODEL}
    q2 = {s: [0.]*len(q1[s]) for s in q1}
    for _ in range(episodes):
        state = 0
        action = choose(epsilon_probs([x+y for x, y in zip(q1[state], q2[state])],
                                      epsilon), rng)
        while state is not None:
            reward, sp = step(CONTROL_MODEL, state, action, rng)
            next_action, probabilities = None, None
            if sp is not None:
                probabilities = epsilon_probs([x+y for x, y in zip(q1[sp], q2[sp])],
                                              epsilon)
                next_action = choose(probabilities, rng)
            if kind == "double":
                update, other = (q1, q2) if rng.random()<.5 else (q2, q1)
                target = reward
                if sp is not None:
                    selected = max(range(len(update[sp])), key=lambda a: update[sp][a])
                    target += .9*other[sp][selected]
                update[state][action] += alpha*(target-update[state][action])
            else:
                target = control_target(kind, reward, None if sp is None else q1[sp],
                                        probabilities, next_action)
                q1[state][action] += alpha*(target-q1[state][action])
            # Keep SARSA's already-selected next action.
            state, action = sp, next_action
    return {s: [(x+y)/2 if kind=="double" else x for x, y in zip(q1[s], q2[s])]
            for s in q1}
```

td_prediction 实现固定两步策略的预测，td_control 实现选动作、执行环境、构造 target、更新和继续行动的完整控制循环。前者只验证价值传播，不拿它的误差曲线代替控制收益。所有控制方法使用同一个奖励定义；Double 返回两表平均，便于和单表的价值尺度比较，行为选择则使用两表之和。

把步长从 .1 降到 .01，观察随机 SARSA 的波动与传播速度同时变化；再令 ε=0，检查是否因为初始并列选择而长期不去 B。后一个失败是覆盖问题，不能仅靠更小 TD error 诊断。若要研究 Double 的最大化偏差，需要额外设置多动作随机后果并跨独立运行统计；本章两步确定性环境主要检查更新角色和目标。

<a id="lesson-branches"></a>

## 8 · 收敛条件与 CRL 衔接

$$
\sum_n\alpha_n(s,a)=\infty,\qquad\sum_n\alpha_n(s,a)^2<\infty
$$

经典表格随机逼近结果使用逐状态动作的步长条件，还需要固定环境、相应访问与噪声条件。不能只检查一个全局步长序列。

SARSA 向最优策略收敛的典型条件还包括贪心极限且持续探索。固定 ε 则保留行为探索；固定步长在随机环境中通常持续波动。持续任务可能正需要固定步长追踪变化，但那是另一种估计目标，不能把固定点收敛定理直接搬过来。

函数逼近、bootstrap 和 off-policy 的组合可能不稳定，表格结论不自动扩展到共享网络。Afterstate 则是另一种表示选择：若动作先确定一个中间状态，之后才有环境随机性，可以评价这个中间状态；它不等于去掉探索或改变长期目标。

<a id="lesson-check"></a>

## 9 · 带答案练习

问：训练 TD loss 不为零是否一定没学对？答：不一定。随机后果可以产生不可约的单样本误差，应检查条件期望、解析价值或独立评估。

问：Expected SARSA 是否天然只能 on-policy？答：不是。它对指定目标策略求和，行为可以不同；必须保持覆盖，并明确目标随训练怎样变化。

问：Double Q 为何不总是更新两张表？答：随机分别更新有助于分离选择与评价。如果两表初始化相同并每次进行完全相同更新，它们会相等，不能形成预期的分离。

问：同一 A→B 经验中，Expected Sarsa为何比这一次Sarsa的目标大？答：本次采到旧表低估的fast值1；期望还按.75概率计入detour值5。对所有可能下一动作平均，Sarsa目标也为4.6；不能据这一条样本宣布Expected算法的控制收益更高。

## 从本章进入实践

[预测与控制](https://yingwen.io/zh/continual-rl/code/#practice-prediction)：学会预测更多事情，什么时候会改变行动？



<a id="chapter-code"></a>

## 下载与运行

Python 3，仅标准库。下载完整文件后在其目录运行；页面展示核心区域。小环境和解析检验用于理解机制，不是论文基准复现。

[下载 tabular_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/tabular_textbook_lab.py)

```sh
python3 tabular_textbook_lab.py temporal-difference
python3 tabular_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：对应第 2–8 章。以下推导、例子和标准库实现独立编写；原书用于核对定义、算法关系与适用条件。

- [David Silver · UCL 强化学习课程](https://davidstarsilver.wordpress.com/teaching/)：原课程页面提供 MDP、动态规划、预测、控制、探索和规划的讲义及视频。

- [Watkins & Dayan · Q-learning](https://doi.org/10.1007/BF00992698)：表格 Q-learning 与收敛条件。

- [van Hasselt · Double Q-learning](https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html)：最大化偏差、双估计器及原始算法。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-tabular-temporal-difference#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-tabular-temporal-difference#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-tabular-temporal-difference)
<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 预测的量是什么，误差又是什么？

先固定策略。价值是回报的条件期望。MC 使用完整回报；TD 用下一时刻的预测替代未观察的余项。两者不能仅按同一批样本上的 TD error 排序。

函数逼近与深度方法：共享参数限制了可表示的函数。采样权重决定在哪些状态上拟合。最小价值误差、最小 Bellman 残差和 TD 固定点一般不同；神经网络又使可表示的局部方向随参数改变。

持续学习中的研究问题：多个 GVF 共用表示时，哪些预测值得占用容量？应分别检查问题定义是否改变、数据是否覆盖，以及回答该问题的误差是否降低。

[MC 与 TD](https://yingwen.io/zh/continual-rl/foundations/tabular/temporal-difference/) → [投影与半梯度](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/) → [神经价值更新](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/) → [GVF 的问题与答案](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)


### 可进一步检验的问题

- [04 · 旧价值什么时候应该复用，什么时候应该快速改写？](https://yingwen.io/zh/continual-rl/research/#research-continual-control)：区分 SARSA、Expected SARSA 与 Q-learning 的目标和实际行为，才能把变化后的估计滞后与探索不足分别检查。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)
- [深度价值学习](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)

对应原始材料：6.1–6.8。本文为原创讲解，原书、论文与上游代码保留各自许可。
