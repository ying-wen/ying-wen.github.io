# 目标与子任务：条件控制、经验重用与技能设计

机器人学会到达一个充电站以后，怎样学习一族目标、利用未成功的尝试，并提出对未来控制有用的子任务？

## 本章内容

- 区分目标表示、目标条件控制、重标记经验、课程选择与子任务发现五个问题。
- 独立写出 goal-conditioned Q-learning 与 HER 的数据循环，知道何时必须重算 reward 和 terminal。
- 推导带 stopping value 的子任务 target，区分 STOMP 原文约定与普通 option 后续价值。
- 用真实任务收益与规划收益评价子任务，而非只看目标到达率。

<a id="problem-definition"></a>

## 本章的问题定义

同一动力学下需要应对一族结果目标，并决定经验如何共享、下一目标如何选择、哪些子任务有长期用途。

### 给定条件与符号

- 目标空间、目标奖励和成功/终止语义，目标分布。
- 目标无关动力学、真实转移及可重算标签的信息，数据和重置预算。

### 需要求解的对象

给定目标的策略/价值；HER改变训练标签，课程改变练习分布，子任务构造另选择有用的行为规格。

### 信息与数据权限

$g$ 是当前指定目标；保存状态、动作、后继、实际达到结果与物理终止。事后目标 $g'$ 只能重算评价标签，不能改写已发生转移。

$$
\max_\pi\ \mathbb E_{g\sim p_G}\mathbb E_{\pi(\cdot\mid s,g)}\!\left[\sum_{t=0}^{T_g-1}\gamma^t r_g(S_t,A_t,S_{t+1})\right]
$$

$p_G$ 是声明的目标评价分布，$r_g$ 为该目标奖励，$T_g$ 为按目标语义定义的停止时刻，$\gamma$ 为折扣。UVFA学习一族价值；重标记分布不等于 $p_G$。STOMP另以沿途cumulant与停止价值定义子任务，非普通到达目标。

### 成立条件与解的含义

- 重标记要求动力学不因目标改变；必要标签可由保留信息重算。
- 随机环境中按实际结果选目标可能条件化噪声；标准HER不普遍无偏。子任务停止价值计时遵从明确的return约定。

判断准则：在原先指定且未用于选择的目标上测成功率、步数和收益；重标记检验奖励与终止重算；子任务还需测其技能/模型对主任务规划的用途。

### 适用边界

- hindsight成功标签不等于原真实目标已经成功。
- 子任务到达率高不证明有主任务价值。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：相对完整持续控制，本章限定目标参数索引的任务族和可共享动力学；目标输入、重标记权限与目标分布是这类控制问题的具体结构。

- 组合不同学习问题 · [Options 与技能发现](https://yingwen.io/zh/continual-rl/construction/options/)：子任务是行为评价规格，option是包含执行与停止的可复用解。

- 组合不同学习问题 · [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)：课程选择改变真实经验分配，不能由目标条件控制或HER自动完成。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

稀疏目标使失败轨迹缺少反馈；共享目标可能引入分布偏差，局部子目标还可能伤害主任务。

### 本章的核心思路

把目标规格、标签操作、课程与子任务停止收益分别定义，再检查各自连接外部收益的路径。

1. [从每个目标推导共享控制](#lesson-derive)：因为目标只改变评价，先写每个目标的Bellman方程，再以UVFA共享参数。

2. [重算事实的评价而非事实](#lesson-derive)：因为失败轨迹可能完成别的目标，HER保留转移并重算奖励/目标终止，同时检查随机选择偏差。

3. [用沿途收益约束子任务](#lesson-subtasks)：因为最短到达可能忽略危险代价，reward-respecting子任务保留沿途奖励并以停止价值表达可复用后果。

结论与条件：固定目标恢复普通控制定义；HER有效性取决于动力学和重标记条件，STOMP停止规则及折扣计时不能凭普通option公式替换。

### 相关方法改变了什么

- UVFA：共享一族目标价值，不决定练习目标。

- HER：重用已发生经验的目标标签，不改变实际交互成功。

- 课程/STOMP：课程选择经验分配；STOMP提出有沿途收益与停止价值的技能子任务。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 状态、动作、目标

$s$ 是此刻用于决策的信息，$a$ 是可执行动作，$g$ 是希望达到的结果规格。目标可以用坐标、特征、图像或奖励参数表达；表示之外，还需要明确成功条件或奖励函数。

### Q-learning 的含义

$Q(s,a,g)$ 估计先做动作 $a$、随后尽力完成目标 $g$ 的累计回报。一次真实转移提供当前奖励和下一状态；最大值对应后续选择价值最高的动作。

$$
Q(s,a,g)\leftarrow Q(s,a,g)+\alpha\left[r_g+\gamma(1-d_g)\max_{a'}Q(s',a',g)-Q(s,a,g)\right]
$$

### Bootstrap 与半梯度

用自己对未来的估计构造当前标签叫 bootstrap。神经网络训练时通常对 target 停止梯度，只对当前 Q 的输出求导；target network 可减缓标签与拟合器同时变化。

### Option 与子任务不是同一物

子任务规定评价某种行为的标准；option 是一种可执行的解，由内部策略 π 与停止概率 β（以及允许启动的位置）组成。一个子任务可能有多个解，也可能始终解不好。

<a id="lesson-setting"></a>

## 1. 目标如何进入一个强化学习问题

想象一辆仓库小车，能够左右移动，当前位置是 s，目的货架是 g。它在一次尝试中没有到达货架 4，却经过了位置 2。最直接的问题是“以后给定任意 g，怎样行动”；另一个问题是“这次失败能否教会到达 2”；再一个问题是“下一次应该练习哪个目标”。三者分别改变价值函数的输入、训练样本的标签、真实交互的分配，不能用一个 HER 名称包办。

| 研究对象 | 输入与输出 | 没有被顺带解决的事 |
| --- | --- | --- |
| 目标表达 | 由 g 定义奖励 $r_g$、成功条件与可选的终止条件 | 并不保证目标可达或可观察 |
| 目标条件控制 / UVFA | 给 s,a,g，预测 Q；给 s,g，输出动作 | 不决定下一次练哪个 g |
| HER / 重标记 | 给已发生的轨迹，构造另一 g 下的训练转移 | 不把失败事实改成真实外部成功 |
| 目标生成 / 课程 | 给学习进度与经验，选择下一批 g | 不等于已学到可靠技能或它的模型 |
| 子任务 / STOMP | 给沿途 cumulant 与停止价值，学习行为及停止 | 学到行为以后还要学习后果模型并检查规划用途 |

先考虑有限、可观察、动力学不依赖目标的 MDP：环境转移 $P(s'|s,a)$ 不因事后改写 $g$ 而改变。真实数据包含 $(s,a,r,s')$；为了重算目标奖励，还要保存 achieved goal（实际达到的结果）、物理终止与必要的附加信息。下面允许经验重放和环境重置。扩展到 CRL 时，还需明确存储和重置限制，以及目标分布、动力学和目标语义分别怎样变化。

判断是否需要这一条线：若问题只是一个固定目标的回报优化，先把普通控制做好；若同一物理世界要反复应对不同结果，goal-conditioned 控制可以共享经验；若目标奖励过稀疏，考虑 HER；若经验总用于过易或根本不可达的目标，再考虑课程；若希望长期积累可组合的行为与模型，才进一步构造子任务。

<a id="lesson-derive"></a>

## 2. 从一族控制问题推到 UVFA 与 HER

对每个目标 g 先固定奖励 $r_g$ 和终止 $d_g$。最常见的到达任务令成功奖励为 0、未成功为 −1；成功立即结束时，回报衡量到达前的折扣步数。也可以成功后继续交互，只在奖励里反映成功。这两种任务的 Bellman 方程不同，不能把 success、环境 terminated 和时间上限 truncated 混成一个 done。

$$
Q_g^*(s,a)=\mathbb E\!\left[r_g(s,a,S')+\gamma(1-d_g)\max_{a'}Q_g^*(S',a')\mid s,a\right]
$$

这是对固定 g 的普通最优 Bellman 方程。期望来自真实动力学；g 告诉我们怎样评价结果，不负责改变已发生的转移。

UVFA 让不同目标 $g$ 共享一个函数近似器。最简单实现把状态与目标特征拼接后输入网络，也可以分别编码再组合；每个目标仍遵循自己的 Bellman 方程。共享使相近目标互相提供训练信号，也可能产生干扰。泛化能力取决于数据覆盖和表示，需在未用于训练的目标上单独评价。

$$
Q_\theta(s,a,g),\qquad y=r_g+\gamma(1-d_g)\max_{a'}Q_{\bar\theta}(s',a',g)
$$

θ 是在线网络，θ̄ 是慢更新的目标网络；表格实现不需要两个网络。这里展示离散动作版本；连续动作可使用目标 actor 提供下一动作。

$$
\begin{aligned}L(\theta)&=\tfrac12\left(Q_\theta(s,a,g)-\operatorname{stopgrad}(y)\right)^2\\ \theta^+&=\theta+\alpha\left(y-Q_\theta(s,a,g)\right)\nabla_\theta Q_\theta(s,a,g)\end{aligned}
$$

对平方误差求导就得到更新。target 被视作本次拟合的固定标签，因此这是半梯度；并不是对完整 Bellman 残差两侧一起求导。

HER 再增加一个数据操作：先真实执行原目标 g，保留轨迹；抽到其中一步时，选一个后续实际达到的结果 g′；状态、动作、下一状态保持不变，只按 g′ 重算奖励和目标相关的终止。于是“没完成 g”仍能为“怎样完成 g′”提供监督。HER 要配合能利用其他行为策略数据的 off-policy 学习器；它不是直接把 PPO 的 on-policy 轨迹标签改掉就仍保留原保证。

**算法：算法伪代码**

1. 收集一段 ($s_t,a_t,s_{t+1}$，$\mathrm{achieved}_{t+1}$，`physical_done`，`info`)
1. 对抽到的时刻 t：
  1. 以一定概率保留原目标 g；否则从未来 $\mathrm{achieved}_{t+1:T}$ 采样 g′
  1. 不改 $s_t,a_t,s_{t+1}$
  1. 重算 $r_{g^{\prime}}(s_t,a_t,s_{t+1})$
  1. 按所定义任务重算 $d_{g^{\prime}}$，物理终止仍保留
  1. 构造目标 y；更新 Q（以及连续动作时的 actor）
1. 独立评估：只用预先规定的真实目标，不用 hindsight 标签

在确定性且目标无关的环境中，原转移仍是新目标下同一个状态—动作对的有效结果。随机环境里，按后来实际发生的结果选择新目标，可能条件化转移噪声：例如偶然成功的随机结果，被过度表示为可稳定达成的目标。此时标准 HER 不普遍给出无偏 Bellman 样本。课程分布还会改变训练目标的权重，因此重标记后的训练误差与原目标分布上的成功率需要分别评价。

<a id="lesson-subtasks"></a>

## 3. 从到达目标到有停止规则的子任务

外部给定目标只要求解决一个任务；子任务构造还要决定什么行为值得长期保留。考虑通向门口的两条路：一条两步但经过危险区，每步真实代价 −5；另一条四步但无危险。仅奖励“最短到门口”可能学到对主任务有害的技能。reward-respecting 的出发点是保留沿途真实奖励，同时用停止价值表达“这个结果有时值得准备”。

$$
G^{c,z}=\sum_{k=1}^{K}\gamma^{k-1}C_{t+k}+\gamma^{K-1}z(S_{t+K})
$$

K≥1 是本次子任务停止时刻，C 是累积信号，z 是停止价值。这采用 STOMP 原文式(2)的计时：停止价值与最后一次 cumulant 具有同一折扣指数，不是再过一个原始时间步。

将第一步单独拿出。到 $s'$ 后，以 $\beta(s')$ 的概率停止并获得 $z(s')$；否则从 $s'$ 的下一步继续积累，其全部未来相对当前多乘一个 $\gamma$。对这两个分支取期望，得到以下递推。停止的是子任务回报的累积，物理环境仍可继续运行。

$$
\begin{aligned}v(s)&=\mathbb E_\pi\!\left[C+\beta(s')z(s')+\gamma(1-\beta(s'))v(s')\mid s\right]\\ \delta&=C+\beta(s')z(s')+\gamma(1-\beta(s'))v(s')-v(s)\end{aligned}
$$

β=1 时 target=C+z；β=0 时 target=C+γv(s′)。真实环境终止使用 β=1 且 z=0。两个端点是移植实现时最值得先写的测试。

$$
z^i(s)=w^\top x(s)-w_i x_i(s)+\bar w^i x_i(s)
$$

STOMP 的 feature-attainment 子任务把某一特征的价值权重替换为一个乐观 bonus weight。它不是任意给 feature 一个大奖励：其他特征当前的价值仍保留，沿途 cumulant 也保持真实奖励。

固定目标策略与停止函数时，可以直接做 TD；行为由另一策略 $b$ 产生时，一步半梯度更新可乘 $\rho=\pi(a|s)/b(a|s)$。分母使用产生动作时记录的概率，且目标动作需要行为支持。这个比率修正当前动作分布；函数逼近与 off-policy bootstrap 的稳定性仍需另行分析。

$$
\begin{aligned}v(s)&\leftarrow v(s)+\alpha\rho\delta\\ \theta&\leftarrow\theta+\alpha_\pi\rho\delta\nabla_\theta\log\pi_\theta(a|s)\end{aligned}
$$

这里取 $\lambda=0$，同一 TD 误差 $\delta$ 在 critic 更新前缓存。重要性比修正当前动作分布；actor 所见的状态分布仍由采样过程决定，完整目标的梯度解释还取决于相应占用权重。

停止收益的计时还会影响停止规则。STOMP 式(9)采用 $\beta(s)=\mathbf1[z(s)\geq v(s)]$；若保持当前 $v$ 不变、单独最大化前面的单步 target，比较量却是 $z(s)$ 与 $\gamma v(s)$。例如 $z=9.5$、$v=10$、$\gamma=0.9$ 时，前一规则继续，后一规则停止。因此，复现该论文应保留其规则；设计另一种最优停止问题则应明确 return 与边界条件，再重新推导，不能仅凭符号相似互换。

**算法：算法伪代码**

1. 初始化每个子任务的价值 $v_i$、策略 $π_i$、停止价值规格 $z_i$
1. 每个真实转移：
  1. 记录旧价值、旧策略概率、行为概率 b(a|s)
  1. 由选定规则计算 $β_i(s^{\prime})$，环境终止强制 $β_i=1,z_i=0$
  1. 对每个 i 计算 $δ_i = r + β_i z_i + γ(1-β_i)v_i(s^{\prime}) - v_i(s)$
  1. 更新 critic；用已缓存 $δ_i$ 更新 actor
  1. 另行学习此 option 的真实 reward/end-state 模型
  1. 评价该模型用于主任务规划的改善，不能只看子任务成功率

<a id="lesson-curriculum"></a>

## 4. 哪个目标值得现在学：显式课程到持续目标生成

先做一个透明的有限目标课程：每个目标维护尝试次数、成功次数和近期成功率，优先采样既非几乎必成、也非长期完全失败的目标。这是“适中难度”的可运行基线，不是 Goal GAN 的完整实现。Goal GAN 再学习一个生成器来覆盖适中难度目标区域；分类器用当前策略对目标的成功概率构造标签，生成器提出下一批练习目标。它增加的是目标分布学习，而不改变低层控制的 Bellman 方程。

$$
\hat p_t(g)=\frac{\text{近期成功次数}}{\text{近期尝试次数}},\qquad \mathcal G_{\rm learnable}=\{g:\ell\leq\hat p_t(g)\leq u\}
$$

近期窗口避免历史累计均值完全淹没近期变化；阈值与置信度、窗口长度共同决定课程。探索保底应保证暂时失败的目标并非永远被排除。

学习进度是另一种信号：比较同一目标在两个相邻窗口的成功率或预测误差改善。纯粹选进步最快的目标可能遗忘旧目标，也可能把噪声造成的波动当作进步；所以应保留复习预算、未探索目标预算和难度不确定性。CRL 中目标语义与环境可达性一起变化时，还需要区分“技能退化”与“世界变了”。这应成为实验中的独立变量，而非让课程默默选择更容易的任务。

| 分支 | 实际改变的对象 | 必要对照 |
| --- | --- | --- |
| 固定目标集 + 均匀采样 | 只控制训练目标分布 | 先确认控制器能解单目标 |
| 适中难度 / Goal GAN | 采样或生成可学习目标 | 相同交互预算下与均匀目标比较 |
| 学习进度驱动 | 按能力变化分配练习 | 区分噪声、遗忘和真正进步 |
| 奖励相关子任务 | 改变行为学习的 cumulant / stopping value | 与最短路径、随机技能比较规划收益 |
| 持续目标发现 | 候选产生、评估、保留与淘汰共同在线变化 | 计入候选学习与模型维护的算力/内存成本 |

<a id="lesson-example"></a>

## 5. 一次经验如何产生两个不同但合法的更新

小车从 $1$ 向右到 $2$，原目标为 $4$，$\gamma=0.9$、$\alpha=0.5$。当前动作价值为 $-2$，下一状态对目标 $4$ 的最大动作价值为 $-1$。任务成功即终止，成功奖励为 $0$，其余为 $-1$。原目标未达成，故 target 为 $-1+0.9\times(-1)=-1.9$，新值为 $-2+0.5\times0.1=-1.95$。

把同一转移重标为目标 $2$：物理过程不变，但目标已达成，$r'=0$、$d'=1$，不再 bootstrap。若当前动作对目标 $2$ 的价值原为 $-2$，更新后为 $-2+0.5\times(0-(-2))=-1$。两个值回答不同的条件问题。若任务规定成功后继续，$d'$ 则不自动变为 $1$，应保留相应后续价值。

| 本次计算 | 原目标 4 | 重标记目标 2 |
| --- | --- | --- |
| reward | −1 | 0 |
| 目标终止 | 否 | 是 |
| target | −1.9 | 0 |
| 更新前 Q | −2 | −2 |
| 更新后 Q | −1.95 | −1 |

子任务例子中，$v(s)=2$、$v(s')=6$、$r=1$、$z=4$、$\beta=0.25$。停止分支贡献 $0.25\times4=1$，继续分支贡献 $0.9\times0.75\times6=4.05$，target 合计 $6.05$。取 $\alpha=0.1$、$\rho=1$，新价值为 $2.405$。这里的 $z$ 是子任务停止价值，区别于主任务中“换一个 option 继续”的高层价值。

<a id="lesson-code"></a>

## 6. 实现与实验：目标标签和终止条件

目标重标记、表格 goal-Q、future HER、子任务 TD 与有限目标课程。

```python
def relabel_goal(next_position: int, goal: int, physical_terminal: bool = False):
    """Our goal task ENDS on success; another task may not use this convention."""
    success = next_position == goal
    reward = 0.0 if success else -1.0
    terminal = physical_terminal or success
    return reward, terminal


def goal_q_update(q, state, action, next_state, goal, alpha=0.5, gamma=0.9,
                  physical_terminal=False):
    reward, terminal = relabel_goal(next_state, goal, physical_terminal)
    next_value = 0.0 if terminal else max(q.get((next_state, a, goal), 0.0)
                                          for a in (-1, 1))
    key = (state, action, goal)
    old = q.get(key, 0.0)
    target = reward + gamma * next_value
    q[key] = old + alpha * (target - old)
    return reward, terminal, target, q[key]


def future_her(trajectory, original_goal, rng):
    """trajectory: (s,a,s_next,physical_terminal) in one recorded episode.

    Outputs (s,a,s_next,goal,physical_terminal), preserving physical termination
    for both original and hindsight goals. Goal success must never erase it.
    Dynamics are deterministic and goal independent in this teaching example.
    """
    replay = []
    for t, (s, a, sn, physical_terminal) in enumerate(trajectory):
        replay.append((s, a, sn, original_goal, physical_terminal))
        future_index = rng.randrange(t, len(trajectory))
        hindsight_goal = trajectory[future_index][2]
        replay.append((s, a, sn, hindsight_goal, physical_terminal))
    return replay


def subtask_td(value, state, next_state, reward, stopping_value, beta,
               alpha=0.1, gamma=0.9, rho=1.0):
    """STOMP Eq.(5) timing; beta is supplied, not optimized by this function."""
    old_here, old_next = value[state], value[next_state]
    target = reward + beta * stopping_value + gamma * (1.0 - beta) * old_next
    value[state] = old_here + alpha * rho * (target - old_here)
    return target


def stop_rules(stopping_value, continuation_value, gamma=0.9):
    paper_rule = stopping_value >= continuation_value  # STOMP Eq.(9)
    greedy_one_step = stopping_value >= gamma * continuation_value
    return paper_rule, greedy_one_step


def intermediate_goals(success_counts, attempt_counts, low=0.2, high=0.8):
    """Transparent finite curriculum; not the Goal GAN training algorithm."""
    return [g for g in success_counts if attempt_counts[g] > 0
            and low <= success_counts[g] / attempt_counts[g] <= high]
```

下载脚本后运行；仅需 Python 3.10+ 标准库

```sh
python3 knowledge_algorithms_lab.py goals
python3 knowledge_algorithms_lab.py test
```

运行结果中，原目标的新值为 $-1.95$，HER 新值为 $-1$，子任务 target 为 $6.05$；两种停止比较在 $z=9.5$ 时分别返回继续与停止。physical_terminal 随原样本和重标记样本一起传递。把物理终止后状态的 $Q$ 人为设成 $100$，未达成目标的转移 target 仍为 $-1$，而非错误 bootstrap 得到的 $89$。这个例子同时检查了“目标是否成功”和“物理过程是否结束”两个独立条件。

- 实验 A：把“成功即终止”改为“成功后仍继续”。同时修改 relabel_goal 和对应测试，观察若只改 reward、不改 bootstrap 会怎样。
- 实验 B：为相同 (s,a) 加随机分支，观察 future relabel 后的成功结果是否过度代表真实概率；分别在原目标评估与重放 loss 上作图。
- 实验 C：固定两条通向同一目标的路径，一条短但有负奖励。比较最短路径子任务与真实奖励加 stopping value 的排序，同时记录到达率和沿途代价。

OpenAI Baselines 的 her_sampler.py 展示了“选未来 achieved goal → 替换 desired goal → 调用 reward_fun”的数据流程；随后可读 ddpg.py 的 target 和 replay buffer。原工程采用自己的定长 episode 与成功处理约定，移植到成功即终止任务时，需要同步调整 bootstrap。Goal GAN 的 rllab-curriculum 则按目标采样、成功率标签、策略更新三个环节阅读。两套历史工程适合在独立依赖环境中运行。

<a id="lesson-branches"></a>

## 7. 如何组合目标学习的几个组成部分

目标相关方法可以组合，因为它们修改不同接口：UVFA 表示一族价值函数，HER 重用经验，课程分配交互，子任务定义评价，option 实现行为。一个系统可以同时包含它们，也可能只需要其中一个。组合并不自动有效：HER 的训练目标分布可能与课程提出的目标失配；新生成目标可能无法由现有表示判断成功；有用行为如果没有准确后果模型，仍不能用于可靠规划。

| 你观察到的失败 | 优先检查的量 | 可比较的方法线 |
| --- | --- | --- |
| 目标一换就从零学 | Q 是否以目标为条件，是否共享可迁移表示 | UVFA / goal-conditioned actor-critic |
| 几乎没有成功样本 | 奖励可否从记录数据重算，动力学是否目标无关 | HER / relabeling |
| 只会练容易任务 | 课程覆盖、成功率估计、探索保底 | 均匀采样 / 适中难度 / 学习进度 |
| 到达子目标却损害主任务 | 沿途真实 reward 是否被删掉 | shortest-path / reward-respecting |
| 技能很多而规划没收益 | 模型精度、技能适用性、维护预算 | STOMP 的行为—模型—规划联合评价 |

<a id="lesson-offline-goals"></a>

## 8. 目标很远时：从平坦价值函数到 HIQL

前面的 HER 假设能继续交互，错误的价值还有机会被新经验纠正。离线目标条件 RL 只给定一个已有轨迹集，学习者不能随时尝试一个看起来很好的动作。如果目标很远，好坏动作的价值差很小，估值噪声就可能盖过真正的控制信号。分层的一个作用是把“一步选哪个动作”改成“先到哪个较近的中间状态”。

HIQL（NeurIPS 2023）使用一个目标条件状态价值 $V(s,g)$，然后提取两个策略：高层 $\pi_h(z|s,g)$ 预测中间状态的潜在表示，低层 $\pi_\ell(a|s,z)$ 产生到达这个中间目标的动作。这里的高层动作不是环境按钮，而是数据中未来状态的表示；低层才承担可执行控制。

$$
\begin{aligned}\omega_j&=\operatorname{sg}\!\left[\min\{\exp(\eta_j A_j),100\}\right],\qquad j\in\{\ell,h\},\\ z_{\rm target}&=\operatorname{sg}\!\left[\phi(s_{t+k})\right],\\ J_\ell&=\mathbb E_{\mathcal D}\left[\omega_\ell\log\pi_\ell(a_t\mid s_t,\phi(g_\ell))\right],\\ J_h&=\mathbb E_{\mathcal D}\left[\omega_h\log\pi_h(z_{\rm target}\mid s_t,g)\right].\end{aligned}
$$

两层 actor 做优势加权回归；$\eta_\ell$ 与 $\eta_h$ 分别控制权重尖锐程度。这里写出作者实现的上限 100。$\operatorname{sg}$ 保持数值、停止该通路的梯度：actor 拟合固定权重和固定高层标签，不通过改变它们来降低回归损失。

价值为两层策略提供评价，作者实现可以在同一批更新中联合计算价值、低层 actor 与高层 actor 的损失。回归权重使用价值差：低层 $A_\ell=V(s_{t+1},g_\ell)-V(s_t,g_\ell)$，高层 $A_h=V(s_{t+k},g)-V(s_t,g)$。计算这些权重及高层标签时使用更新前参数，并相对于本次 actor 求导保持固定；价值仍由自己的损失学习。目标表示还可依赖当前位置，前式用 $\phi(g)$ 简写该接口。低层策略输入中的目标编码是否接受 actor 梯度由作者代码的 policy_train_rep 配置决定，不能与停止高层回归标签梯度混为一谈。

状态序列可训练价值和高层，但低层动作策略仍需要带动作的数据。这里利用无动作数据扩大的是对“哪些状态连接到哪里”的知识；执行动作仍须通过动作数据学习。把数据来源分开，是评价这类方法与纯视频学习、完全离线控制之间关系的必要条件。

HIQL 与 HER 的关系是互补的：HER 处理目标标签和经验使用，HIQL 处理长期控制中的策略结构与离线提取。它与可变时长 Option-Critic 也不同：中间目标和训练间隔给出了分层结构，并不自动学习同样的终止梯度。用于持续 RL 时，还需研究新经验怎样更新旧价值和目标表示，而不是仅换一批测试目标。

OGBench（ICLR 2025）适合检验这类区别。它把长时域到达、轨迹拼接、像素输入和随机性等挑战放进统一的离线目标学习基准，并提供多种基线实现。起步时选一个状态输入任务，固定数据与目标抽样，比较平坦目标策略和 HIQL；再依次改变数据覆盖、子目标间隔和观测形式。先解释哪一种困难导致差距，再扩大任务规模。轨迹边界标记与 bootstrap mask 在数据接口中具有不同含义，时间截断不应自动当作真实任务终止。

<a id="lesson-language-skills"></a>

## 9. 人类如何指定技能：MaestroMotif 的语言到控制过程

有些子任务来自图结构，有些来自人的语义要求。例如“寻找食物”和“回避危险”比一个坐标更适合描述 NetHack 中的长期行为，却难以直接手写一个可靠的数值奖励。MaestroMotif（ICLR 2025）将语言模型放在技能设计环节：把语言要求转换成可学习奖励、启动与终止规则、以及训练时的技能调度，然后仍由 RL 学习执行技能的低层策略。

第一步，给定技能描述 $x_i$，让语言模型比较两段候选观察中哪个更符合要求，得到偏好标签 $y$；随后训练奖励模型 $r_i$。一个基本的成对偏好模型是令奖励差决定偏好概率。

$$
\begin{aligned}p_i(s_1\succ s_2)&=\sigma(r_i(s_1)-r_i(s_2))\\ \mathcal L_i&=-\mathbb E\left[y\log p_i+(1-y)\log(1-p_i)\right]\end{aligned}
$$

这将语言反馈蒸馏成低层训练时可反复查询的标量信号。损失只约束相对偏好，奖励平移本身不可辨识；其尺度、偏好一致性及分布外可靠性仍影响控制。

第二步，依据技能规格生成 $I_i$、$\beta_i$ 的程序和训练时的高层调度程序。第三步，在调度所产生的状态分布中，以学得的 $r_i$ 训练技能条件低层策略。第四步，面对下游任务描述，生成组合这些既有技能的程序。学习奖励、写执行规则、学原始动作和组合技能是四种不同操作。

**算法：MaestroMotif 的流程；语言反馈规定行为语义，强化学习承担行为执行**

1. 人类给出技能描述与技能组合要求
1. 语言模型比较观察对 → 训练每个技能的奖励模型
1. 语言模型生成启动、终止与训练调度程序
1. 在调度产生的交互中，用 RL 训练技能条件低层策略
1. 下游任务描述 → 组合既有技能的程序 → 执行并评估外部任务

这种方法与 STOMP 的差别在奖励来源：STOMP 保留环境沿途奖励并改变停止价值，MaestroMotif 则依据语言偏好学习技能奖励，还借助人类规格和预训练语言知识。其下游组合无需重新训练某个任务的低层策略，不意味着技能本身没有经过 RL 训练，也不意味着技能规格已经由智能体自主发现。

作者仓库可按三个目录阅读：preference 对应偏好与奖励学习，code_generation 对应规则及组合程序，rl_baseline 与 sample_factory 对应技能执行训练。一个有辨识力的实验是固定技能策略，分别改变奖励模型、训练调度和下游组合程序：技能失败究竟来自语义评价、访问不到适合学习的状态，还是高层组合不当？这比只比较最终任务得分更能定位改进方向。

<a id="lesson-check"></a>

## 10. 失败诊断与自测答案

- Q 总是接近零而真实成功率不升：检查重标记比例过高、成功标签泄漏，以及评估是否错误采用事后目标。
- HER 后回报上升但原目标失败：检查训练目标覆盖；重标记只证明某些结果可学习，不保证原目标可达。
- 子任务全部立刻停止：检查 stopping value 尺度、初始化、是否错误地把 β 当作环境 done，以及是否允许零步启动就停止。
- CRL 性能忽高忽低：分别记录目标分布、环境变化和模型版本，并按这些条件分析成功率。

自测 1：为何 HER 不能随意改 s′？答：状态转移是事实证据；改变目标是在同一事实下提出新问题，改变结果则是在伪造没有经历过的转移。自测 2：把 stopping bonus 写进主任务模型会怎样？答：规划会把学习辅助奖励当真实收益，重复调用技能可能获得虚假的“奖励”。自测 3：目标编码共享的失败就是灾难性遗忘吗？答：不一定，也可能是目标本来不可区分、训练分布变了或价值估计尚未收敛，应通过固定目标保留集和表示可辨识性实验区分。

## 本章的实验设计

子任务成功率与主任务收益需要分别测量。设计固定目标、随机目标和学得目标三个对照。

设定：在小型地图固定低层技能和高层控制器，比较固定目标、随机目标与学习目标。研究者图结构只用于可达性诊断。

- 目标成功判定只用当时可得信号。
- 后见目标标签用于训练，不作为过去行动时已经知道的输入。
- 不可达目标有时限、失败与退出语义。

对照：数量和尺度匹配的随机目标；固定手工目标的特权诊断；固定技能只改目标；再固定目标只改技能

记录：目标到达率、到达时间与失败率；新任务收益和目标使用频率；全部目标生成和训练成本

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-modules)

## 学习与研究衔接

目标定义偏好，状态保留信息。学会达到给定目标，不等于学会构建有用目标。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-goals) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=goals) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=goals)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

状态是支持后续计算的内部信息；GVF 指定一个预测问题；RTRL 和资格迹规定信用如何传播。三者可以组合，但不是相互替代的算法名称。先理解给定策略的预测，再讨论预测怎样改善控制。

- [Does Zero-Shot Reinforcement Learning Exist?](https://yingwen.io/zh/continual-rl/research/#recent-zero-shot-forward-backward)

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

Laplacian 描述行为图结构，奖励感知表示加入路径价值，METRA 学习有区别的行为，HIQL 利用离线目标轨迹，MaestroMotif 引入语言先验。它们承担不同的设计工作；生成技能、选择技能与组合技能需要分别评价。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [Laplacian Keyboard: Beyond the Linear Span](https://yingwen.io/zh/continual-rl/research/#recent-laplacian-keyboard)
- [HIQL: Offline Goal-Conditioned RL with Latent States as Actions](https://yingwen.io/zh/continual-rl/research/#recent-hiql-hierarchical-goals)
- [MaestroMotif: Skill Design from Artificial Intelligence Feedback](https://yingwen.io/zh/continual-rl/research/#recent-maestromotif-semantic-skills)
- [Foundation Policies with Hilbert Representations](https://yingwen.io/zh/continual-rl/research/#recent-hilbert-foundation-policies)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

模型可提取性的理论说明某类能力需要什么知识，不指定唯一网络。Dreamer 研究潜在想象控制，STOMP 研究随机时长行为模型，DRAGO 研究旧模型知识保留。模型误差、查询策略和规划收益之间仍需实验连接。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [General Agents Contain World Models](https://yingwen.io/zh/continual-rl/research/#recent-general-agents-world-models)
- [Laplacian Keyboard: Beyond the Linear Span](https://yingwen.io/zh/continual-rl/research/#recent-laplacian-keyboard)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [OGBench: Benchmarking Offline Goal-Conditioned RL](https://yingwen.io/zh/continual-rl/research/#recent-ogbench-goal-evaluation)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文提供定义和条件，架构讲座提出模块组织，算法论文检验特定机制。完整系统还要明确智能体、外部设计者和世界各自承担的工作；组件成立不自动意味着组合后的长期收益成立。

- [General Agents Contain World Models](https://yingwen.io/zh/continual-rl/research/#recent-general-agents-world-models)
- [The OaK Architecture: A Vision of SuperIntelligence from Experience](https://yingwen.io/zh/continual-rl/research/#recent-oak-architecture)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)

### Reward-Respecting Subtasks for Model-Based Reinforcement Learning

Richard S. Sutton, Marlos C. Machado, G. Zacharias Holland, David Szepesvari, Finbarr Timbers, Brian Tanner, Adam White

Artificial Intelligence · 2023 · 支持方法与理论

#### 研究问题

学到一个能到达子目标的技能之后，为什么它仍可能不适合主任务规划？

#### 关键机制

STOMP 把子任务、option、模型和规划连起来。子任务保留原任务的路径奖励，并用带有特征偏好的终止价值表达目标；学习得到策略和终止规则后，再预测该行为的累计奖励与折扣终点。这样，技能不会因为只追求到达子目标而忽略途中代价。

#### 证据

论文用明确的小问题展示奖励感知子任务如何产生更有用的行为和规划模型。它提供的是可分析的构造链，而非只比较一个技能执行成功率。

#### 条件与限制

终止收益的约定是子任务定义的一部分，不能随意换成固定终点奖励。特征和子任务候选的选择尚不等于完整自主发现机制；实验也不构成整个 OaK 架构的验证。

#### 阅读与实验

在同一个绕路环境中比较“最短到达目标”和“保留路径奖励”的子任务。分别计算 option 的奖励模型、折扣终点模型与一次规划备份。

#### 原文与相关入口

- [期刊论文](https://doi.org/10.1016/j.artint.2023.104001)：STOMP 与奖励感知子任务的正式论文。
- [作者预印本](https://arxiv.org/abs/2202.03466)：最初预印本早于期刊年份；阅读停止收益的精确定义。

### Laplacian Keyboard: Beyond the Linear Span

Siddarth Chandrasekar, Marlos C. Machado

arXiv 预印本 · 2026 · 支持方法与理论

#### 研究问题

从一组谱技能出发，能否解决超出原特征线性奖励空间的新任务？

#### 关键机制

Laplacian 特征先定义行为基，并借助后继特征预测各行为的后果。固定任务权重的价值组合受特征张成空间限制；论文进一步使用随状态变化的元策略，在不同位置组合已有行为。关键变化是组合规则从一组全局固定权重变为状态相关的行为选择。

#### 证据

论文对行为基与任务组合给出理论分析，并报告有限环境中的组合实验。它延续 eigenoptions 与 successor features 的路线，同时解释了为什么单纯线性读出会遇到表达边界。

#### 条件与限制

理论结论依赖具体的行为基、近似误差和任务条件。技能集合的长期生成、淘汰与非平稳模型维护仍是另外的问题；此处按预印本收录，不指定未经确认的会议。

#### 阅读与实验

构造一个必须在中途切换方向的奖励任务。分别比较固定权重的技能选择与状态相关切换，并解释性能差异来自哪里。

#### 原文与相关入口

- [作者预印本](https://arxiv.org/abs/2602.07730)：阅读线性张成空间的限制及状态相关组合机制。

### HIQL: Offline Goal-Conditioned RL with Latent States as Actions

Seohong Park, Dibya Ghosh, Benjamin Eysenbach, Sergey Levine

NeurIPS 2023 · 2023 · 支持方法与理论

#### 研究问题

只拿到已有轨迹时，长距离目标为什么适合拆成高层子目标和低层动作？

#### 关键机制

HIQL 学习目标条件价值，并以潜在状态作为高层动作。高层提出中间目标，低层输出环境动作；两层利用优势加权回归学习。时间分解让低层面对较短的控制距离，而不是要求一个策略直接消化所有远距离价值误差。

#### 证据

论文在离线长时域目标任务中检验层次结构，并提供原始实现。作者后来在 OGBench 中提供更统一的实现，二者适合不同用途：原实验复现和统一基线比较。

#### 条件与限制

数据覆盖和行为分布约束仍然存在。目标采样、层级时间间隔与离线轨迹由外部流程提供，不能把效果解释为在线自主目标生成已经解决。

#### 阅读与实验

对一段轨迹明确标记最终目标、中间目标和当前动作。逐一检查价值目标、优势权重和高层标签的停止梯度边界。

#### 原文与相关入口

- [NeurIPS 2023 原文](https://papers.nips.cc/paper_files/paper/2023/file/6d7c4a0727e089ed6cdd3151cbe8d8ba-Paper-Conference.pdf)：离线目标学习和两层回归目标。
- [HIQL 原始实现](https://github.com/seohongpark/HIQL)：README 区分原始实验与 OGBench 中的新实现。

#### 作者代码

[作者仓库；更新的统一基线另见 OGBench。](https://github.com/seohongpark/HIQL)

HIQL 原论文的离线训练与评价。

### OGBench: Benchmarking Offline Goal-Conditioned RL

Seohong Park, Kevin Frans, Benjamin Eysenbach, Sergey Levine

ICLR 2025 · 2025 · 评价与实验协议

#### 研究问题

一个目标条件算法表现不好，是长时域、轨迹拼接、视觉表示还是随机性造成的？

#### 关键机制

OGBench 用不同环境类型与数据集分别施加这些困难，并提供统一的目标条件基线。固定离线数据让算法面对相同经验，从而将学习机制的差异与在线探索能力的差异暂时分离。

#### 证据

论文提供八类环境、八十五个数据集和六类算法实现。价值在于可复用的实验接口与困难分解，而不只是汇总一个排行榜。

#### 条件与限制

固定数据不检验智能体如何主动获得未来经验，也不直接检验单次生命的灾难性变化、恢复或长期资源管理。它适合 CRL 子问题实验，不是完整 CRL 的替代品。

#### 阅读与实验

先选择只改变一种困难的两个数据集，再比较 HIQL 与平坦目标策略。把观察到的差异写成可检验机制假设，而不是直接归因于“层次更好”。

#### 原文与相关入口

- [论文](https://arxiv.org/abs/2410.20092)：ICLR 2025；环境、数据与基线定义。
- [作者基准库](https://github.com/seohongpark/ogbench)：数据获取、环境与统一算法实现。

#### 作者代码

[基准作者维护的官方实现。](https://github.com/seohongpark/ogbench)

离线目标环境、数据集与标准化基线。

### MaestroMotif: Skill Design from Artificial Intelligence Feedback

Martin Klissarov, Mikael Henaff, Roberta Raileanu, Shagun Sodhani, Pascal Vincent, Amy Zhang, Pierre-Luc Bacon, Doina Precup, Marlos C. Machado, Pierluca D’Oro

ICLR 2025 · 2025 · 支持方法与理论

#### 研究问题

语言描述如何变成可训练的技能奖励，并进一步组织成一个层次策略？

#### 关键机制

设计者先给出技能描述。语言模型的偏好反馈被用于训练奖励模型，再用生成的代码规定技能启动、终止和组合方式；强化学习负责学习实际执行行为。这把语义先验、奖励学习和时间抽象串成了具体训练流程。

#### 证据

论文在 NetHack 学习环境中检验复杂技能与任务组合。作者仓库同时包含偏好、代码生成和 RL 训练模块，可以追踪自然语言到环境动作的完整依赖。

#### 条件与限制

语义知识、技能描述和语言模型来自外部设计过程。该证据并不说明智能体仅凭自身交互就能产生同样的技能体系；偏好模型也可能与真实目标不一致。

#### 阅读与实验

选择一项技能，分别列出描述、偏好标签、训练奖励、终止条件和下游用途。移除语义描述或改变奖励模型时，要单独计量额外查询与人工成本。

#### 原文与相关入口

- [ICLR 2025 原文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2dc5a0faac8102fd47363795f71126ee-Abstract-Conference.html)：技能设计、奖励学习与组合实验。
- [作者实现](https://github.com/mklissa/maestromotif)：偏好学习、代码生成和执行策略的不同模块。

#### 作者代码

[原论文作者仓库。](https://github.com/mklissa/maestromotif)

MaestroMotif 的偏好处理、技能组织与 RL 实验。

### General Agents Contain World Models

Jonathan Richens, David Abel, Alexis Bellot, Tom Everitt

ICML 2025 · 2025 · 支持方法与理论

#### 研究问题

能完成足够丰富的目标集合，是否意味着智能体内部已经包含可提取的环境预测知识？

#### 关键机制

论文在形式化条件下，将广泛多步目标上的行为能力与环境模型的可提取性联系起来。通过查询智能体对不同目标的行为，可以恢复关于环境后果的信息；目标集合和性能要求越强，所要求的预测知识也越强。

#### 证据

主要证据是给定假设下的理论结果，而不是某个世界模型架构在所有任务上击败无模型算法的实验。

#### 条件与限制

可提取模型不等于智能体显式保存一个 RSSM，也不意味着所有实用任务都需要重建全部环境。必要知识的结论不能代替如何高效学到它的算法。

#### 阅读与实验

列出定理要求的目标丰富性和查询能力，再尝试构造一个只会单一任务的反例。由此区分任务专门知识与支持广泛目标的预测模型。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2506.01622)：形式化设定、模型可提取性与证明。
- [David Abel 论文目录](https://david-abel.github.io/papers.html)：作者提供的 ICML 2025 发表信息及相关研究。

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

### Reset-free Reinforcement Learning with World Models

Zhao Yang, Thomas M. Moerland, Mike Preuss, Aske Plaat, Edward S. Hu

TMLR 2025 · 2025 · 支持方法与理论

#### 研究问题

不能靠外部重置回到起点时，怎样兼顾探索新状态与持续获得对任务有用的经验？

#### 关键机制

MoReFree 在 goal-conditioned world-model 系统中交替练习评测目标、返回初始分布与探索目标；模型内的策略训练也偏向任务相关目标。返回行为通过真实动作实现，调度块结束不会将物理世界 reset。

#### 证据

作者在八个 reset-free 任务中与模型自由及模型式基线比较；公开环境、探索调度和 imagination training 实现。

#### 条件与限制

训练无 reset，但主要评价仍使用可重置的 episodic 测试。已给定初始与目标状态分布、世界模型和 replay 都是资源；这不是任意非平稳 CRL 或真实安全的完整保证。

#### 阅读与实验

把返回成本计入总步数，分别消融数据获取目标与模型内训练目标；检查外部 reward-free 是否仍依赖设计者提供目标示例。

#### 原文与相关入口

- [作者论文 v3](https://arxiv.org/html/2408.09807v3)：训练与评价协议、back-and-forth exploration 与目标分布。
- [TMLR 作者项目页](https://yangzhao-666.github.io/morefree/)：正式发表状态与作者代码链接。

#### 作者代码

[TMLR 作者项目页明确链接的官方实现。](https://github.com/yangzhao-666/MoReFree)

resetfree/env.py、goal_picker_wrapper.py、Dreamer/PEG 与目标条件实验。

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

### Foundation Policies with Hilbert Representations

Seohong Park, Tobias Kreiman, Sergey Levine

ICML 2024 · 2024 · 支持方法与理论

#### 研究问题

如何从无任务标签的离线轨迹形成既能按方向调用、又能用于目标任务的策略接口？

#### 关键机制

HILP 先学习近似保存时间距离的 Hilbert 表示，再以潜在位移与方向的内积训练方向条件策略。新任务通过奖励回归、目标方向或分层调用选择策略条件，结构表示也支持测试时规划。

#### 证据

ICML 原文与作者项目包含零样本 RL、离线目标条件 RL 及规划实验；官方仓库将 zero-shot 与 goal-conditioned 两套实现分开。

#### 条件与限制

精确时间距离不总能无损嵌入有限维对称欧氏距离，尤其有向不可逆行为；理论充分条件与近似神经实验需区分。方向条件策略没有自动获得任意停止条件或完整技能后果模型。

#### 阅读与实验

固定离线数据分别测距离误差、方向执行误差、奖励可表达误差与高层收益。让同一视觉观测对应不同历史，检查仅观测编码是否足够，之后再讨论 CRL 状态维护。

#### 原文与相关入口

- [ICML 2024 原文](https://proceedings.mlr.press/v235/park24g.html)：Hilbert 距离、策略提示和定理前提；不是 ICLR 论文。
- [作者项目与公式](https://seohong.me/projects/hilp/)：时间距离与方向奖励接口。
- [官方实现](https://github.com/seohongpark/HILP)：hilp_zsrl 与 hilp_gcrl 对应不同实验。

#### 作者代码

[作者项目直接链接并标为 official implementation。](https://github.com/seohongpark/HILP)

离线预训练、零样本奖励适配及目标条件实验。

### Constructing an Optimal Behavior Basis for the Option Keyboard

Lucas N. Alegre, Ana L. C. Bazzan, André Barreto, Bruno C. da Silva

NeurIPS 2025 · 2025 · 支持方法与理论

#### 研究问题

状态相关的技能组合足够强时，是否仍须为每个新奖励保存一条完整策略？

#### 关键机制

OKB 联合扩充基础策略与 Option Keyboard 的元策略。线性支持方法选择需要补齐的任务权重，先训练现有基础上的组合，再检查无法表达的动作并新增基础，移除冗余项。优化的是可组合的行为基，而非仅增加技能数量。

#### 证据

正式原文分析基础数量与 convex coverage set 的关系，并在多任务领域检验规模与表现。附录提供元策略、新基础训练和角点枚举等实现细节；论文声明实验代码在 Supplemental Material。

#### 条件与限制

保证假定 NewPolicy 返回最优策略，且 TrainOK 能达到可表达的最优组合。近似 critic、有限训练、变化动力学不直接继承保证；非线性任务结论仅覆盖最优行为可由相应子策略组合的类别。

#### 阅读与实验

分别比较“增加基础”“只训练组合”和“去除冗余”。保留一组未用于基构造的奖励权重，记录基础规模、元策略成本、SF 误差与迁移回报。持续淘汰仍需未来任务效用检验。

#### 原文与相关入口

- [NeurIPS 2025 原文与补充材料入口](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0ab48777def88e73b50746a6011be0b0-Abstract-Conference.html)：算法 1–3、附录 A.3 的两个最优子程序假设及代码声明；未在本教材运行补充代码。
- [Option Keyboard 的经典桥梁](https://proceedings.neurips.cc/paper/2019/file/251c5ffd6b62cc21c446c963c76cf214-Paper.pdf)：cumulant 组合、GPE/GPI 与技能接口。


<a id="chapter-code"></a>

## 下载与运行

目标重标记、子任务 TD 与有限目标课程的表格实验；深度方法另附原论文和作者工程。

[下载 knowledge_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/knowledge_algorithms_lab.py)

```sh
python3 knowledge_algorithms_lab.py goals
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Schaul et al. — Universal Value Function Approximators](https://proceedings.mlr.press/v37/schaul15.html)：目标作为价值函数输入，状态与目标通过共享表示实现泛化；比较原文表示分解与直接拼接输入的实现。

- [Andrychowicz et al. — Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)：原文。对照 future 策略、off-policy 学习器与稀疏奖励实验；重标记不等于改变真实成功率。

- [OpenAI Baselines — her_sampler.py](https://github.com/openai/baselines/blob/master/baselines/her/her_sampler.py)：作者团队原工程入口。逐项追未来 achieved goal、目标替换和 reward_fun；依赖与 done 约定需同时检查。

- [Florensa et al. — Automatic Goal Generation for RL Agents](https://proceedings.mlr.press/v80/florensa18a.html)：原文与项目入口。Goal GAN 学的是适中难度目标分布；它不是 HER，也不是低层控制器。

- [Goal generation — 作者 rllab-curriculum](https://github.com/florensacc/rllab-curriculum)：课程层与策略层的实现：目标生成、成功率标签、策略训练各自改变什么。旧依赖适合放在独立环境。

- [Sutton et al. — Reward-Respecting Subtasks](https://arxiv.org/html/2202.03466v3)：式(2)、(5)、(9)分别给出 return、TD target 和停止规则；式(12)之后另定义真实 option model。

- [Park et al. — HIQL · NeurIPS 2023](https://arxiv.org/abs/2307.11949)：从同一个目标价值提取高层子目标与低层动作，分析长期价值噪声对平坦和分层策略的影响。

- [HIQL 作者实现](https://github.com/seohongpark/HIQL)：比较目标采样、价值学习、低层优势和高层跨步优势；way_steps 对应中间目标间隔。

- [HIQL 核心文件 — src/agents/hiql.py](https://github.com/seohongpark/HIQL/blob/master/src/agents/hiql.py)：compute_actor_loss 与 compute_high_actor_loss 分别构造一步和跨步价值差；compute_value_loss 做 expectile 更新；pretrain_update 组合三项损失。

- [OGBench · ICLR 2025](https://arxiv.org/abs/2410.20092)：离线目标条件 RL 的任务和数据设计；把长时域、拼接、观测和随机性等困难分开。

- [OGBench 作者基准与基线](https://github.com/seohongpark/ogbench)：环境、数据接口与算法实现；尤其注意 masks 与 terminals 分别表示什么。

- [Klissarov et al. — MaestroMotif · ICLR 2025](https://arxiv.org/abs/2412.08542)：技能描述、偏好奖励、程序化规则及低层 RL 的完整分工；实验重点是 NetHack 中技能的训练与组合。

- [MaestroMotif 作者实现](https://github.com/mklissa/maestromotif)：preference、code_generation、rl_baseline 和 sample_factory 对应奖励、组合规则与执行策略学习。
