# 强化学习实验与算法测试

从一个可判定的问题出发，把目标、估计器、程序事件和实验结果连接起来。本手册给出可手算的测试、受控比较和完整研究流程；算法是否更好，仍需在独立运行中检验。

<a id="handbook-question"></a>

## 1 · 问题、竞争解释与证据层级

一个更高的分数，究竟支持哪一个解释？

研究问题应包含对象、条件与可观测差异。例如“延迟奖励增加后，方法 A 学得更慢”是现象；“历史梯度丢失造成变慢”才是待检验的解释。另一种解释是探索没有到达奖励。只增加资格迹并看到回报上升，还不足以排除探索和有效步长的变化。

$$
J_T(\mathcal A)=\mathbb E\!\left[\frac1T\sum_{t=0}^{T-1}R_{t+1}\right].
$$

这里 $\mathcal A$ 包含初始化、动作选择、学习、记忆维护与计算规则；$T$ 是预定原始交互预算。有限寿命均值、无限期奖励率与最后策略的冻结回报是不同对象。

| 主张 | 最小可判定实验 | 仍不能推出 |
| --- | --- | --- |
| 实现符合公式 | 固定输入的手算和边界 | 长程稳定或更高回报 |
| 某机制发挥作用 | 可干预的正例与反例 | 任意基准上的优势 |
| 锁定方法表现更好 | 独立训练运行的差值及区间 | 所有任务和预算都更好 |
| 过去经验有持续价值 | 新未来条件下的复用与获取 | 无限时间或无限环境中的保证 |

同一实验可以得到分层结论：资格迹正确传播了延迟误差，但控制收益未改善；控制收益改善，但幅度匹配基线同样有效。两种结果都能改变下一步研究。把每项证据对应到它实际测量的主张，比只保留一个总分更有解释力。

### 实验步骤

1. 写出外部目标、数据权限、总预算和最小有意义差异。
2. 列出至少两个能产生同一症状的机制解释。
3. 为每个解释写一项能使其不成立的观测。

### 验收

- 改变机制时出现预先预测的差异，并报告区间和替代解释。
- 实现、机制和效能结果分别命名。

### 常见错误

- 只说“不稳定”而不区分数值发散、跨 run 波动与变化后恢复。
- 把代理损失下降直接当作原始任务收益改善。

原始来源：[Patterson et al. · Empirical Design in Reinforcement Learning](https://jmlr.org/papers/v25/23-0183.html)

---

<a id="handbook-estimator"></a>

## 2 · 原始公式、可得估计与事件顺序

论文中的一个梯度，如何变成某一时刻可执行的写入？

从理想目标到代码至少经过四层：目标及假设、精确表达、可计算估计、实际写入。未知转移、未来反馈、完整状态和轨迹分布导数可能出现在前两层。进入实现时，要说明它们被采样、截断、停止梯度还是替换；不能让这些近似在伪代码中消失。

$$
\delta_t=R_{t+1}+\gamma_{t+1}v_{w_t}(X_{t+1})-v_{w_t}(X_t),\qquad w_{t+1}=w_t+\alpha_t\delta_t\nabla_wv_{w_t}(X_t).
$$

这是 TD 半梯度写入。计算当前预测、后继预测和当前输出梯度时使用同一旧参数。若先更新一个共享网络再计算另一个目标，已得到不同的实际算法。

| 事件 | 可读内容 | 产生的量 |
| --- | --- | --- |
| 行动前 | 合法历史、当前参数、过去统计 | 当前预测与动作 |
| 转移后 | 实际奖励与后继观测 | 终止语义、待学习样本 |
| 目标构造 | 旧参数或指定目标快照 | TD 误差、优势、重要性比 |
| 参数写入 | 已计算目标与梯度 | 新参数、优化器和迹 |
| 下一动作 | 按协议更新后的状态 | 新的行为分布 |

有限差分必须对应同一个函数。验证 TD 半梯度时，先计算并固定 bootstrap 标签，再扰动当前预测的参数；若每次扰动都重算标签，检查的是残差梯度。策略梯度检查还要固定采样路径；固定数据导数正确，并不自动证明随机轨迹估计无偏。

### 实验步骤

1. 为每个目标标出参数版本与标签可得时间。
2. 手画两次交互，列出旧值、新值和重置事件。
3. 固定样本核有限差分，再恢复真实采样做闭环实验。

### 验收

- 独立参考式与程序写入在约定精度内一致。
- 能指出被忽略的导数、数据分布变化与截断偏差。

### 常见错误

- 测试代码逐字复制被测实现，重复同一个错误。
- 把更新后的预测用于本应依赖旧参数的误差。

原始来源：[van Seijen et al. · True Online Temporal-Difference Learning](https://jmlr.org/papers/v17/15-599.html)；[Schulman et al. · Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

---

<a id="handbook-classic"></a>

## 3 · 经典 RL：可解系统与退化条件

在引入网络之前，哪些错误能够用两三个状态排除？

固定策略、有限状态和已知转移能提供真值。以两个状态为例：A 无奖励地转到 B；B 获得 1 后回到 A；折扣为 0.5。Bellman 方程是 v(A)=0.5v(B)，v(B)=1+0.5v(A)，解为 2/3 与 4/3。TD 误差在某个样本上很小，不等于估计已达到这两个值。

$$
v_\pi=r_\pi+\gamma P_\pi v_\pi,\qquad v_\pi=(I-\gamma P_\pi)^{-1}r_\pi\quad(0\leq\gamma<1).
$$

该逆式针对有限折扣问题。平均奖励问题需处理奖励率和差分价值的常数不唯一性，不能将 γ 直接设为 1 后沿用逆矩阵公式。

| 算法对象 | 区分错误版本的测试 | 观察量 |
| --- | --- | --- |
| Sarsa 与 Q-learning | 后继采样动作不是贪心动作 | 两种目标应不同 |
| 资格迹 | 零 λ、重复状态、延迟奖励、终止 | 迹与权重逐步值 |
| 平均奖励 | 奖励统一加常数 | 奖励率应相应平移；差分价值解释不变 |
| 离策略预测 | 目标与行为概率不同；再破坏支持 | 加权恒等式与不可识别反例 |

$$
\rho(a)=\frac{\pi(a)}{\mu(a)},\qquad \sum_a\mu(a)\rho(a)f(a)=\sum_a\pi(a)f(a).
$$

要求 $\pi(a)>0\Rightarrow\mu(a)>0$。取 $\pi=(0.8,0.2)$、$\mu=(0.4,0.6)$、$f=(3,-2)$，两侧均为 2。若行为从不选择目标所需动作，截断比率不能补回缺失信息。

边界测试还包括零步长、零奖励、零折扣与关闭新增模块。有限步长的普通 accumulating TD(λ) 并不等于任意在线前向实现；检验等价前，要固定权重是否在 episode 内变化，并区分 Dutch trace 与普通累积迹。

### 实验步骤

1. 先求解析值，再固定一条轨迹逐步比较。
2. 给每个新增状态量安排一个能触发它的输入。
3. 主动构造支持不足或假设不成立的反例。

### 验收

- 误差符号、动作选择和终端值均能独立手算。
- 边界退化得到对应基础算法，而非仅得到有限数。

### 常见错误

- 把奖励率当普通折扣价值。
- 用能跑完一个 episode 代替目标正确性。

原始来源：[van Seijen et al. · True Online Temporal-Difference Learning](https://jmlr.org/papers/v17/15-599.html)；[Sutton et al. · Horde](https://josephmodayil.com/papers/horde-final.pdf)

---

<a id="handbook-deep"></a>

## 4 · 深度 RL：目标、掩码与梯度

为什么同一个 done 位不能控制所有递推？

考虑外部时间限制结束了一段原本可以继续的轨迹。价值目标通常仍从最终有效观测 bootstrap；但数据的下一行若已是新 episode，GAE 不能把它的优势接回旧轨迹。因此需要分别记录“目标是否延续”和“下一行是否属于同一轨迹”。若有限时域本身属于任务定义，终点与剩余时间应按该任务处理。

$$
\delta_t=r_{t+1}+\gamma b_tV(x_{t+1}^{\rm final})-V(x_t),\qquad \widehat A_t=\delta_t+\gamma\lambda c_t\widehat A_{t+1}.
$$

$b_t$ 表示该目标允许 bootstrap；$c_t$ 表示后续优势是否与当前轨迹相连。外部截断并重置时可有 $b_t=1,c_t=0$。代码在真实 terminal 分支直接返回奖励，避免 $0\times\mathrm{NaN}$。

手算：r=1，当前 V=2，最终 V=3，γ=0.9，得到 δ=1.7。下一行是新 episode，优势为 100；本次优势仍是 1.7。若错误地跨 reset 传播，λ=0.8 时会变成 73.7。自动重置返回的新初态也不能替代最终有效观测。

TD bootstrap 与 GAE 轨迹连接分别编码。

```python
def td_target(reward, discount, next_value, terminated=False):
    finite(reward)
    probability(discount)
    # Branch before reading a terminal value: 0 * NaN is still NaN.
    if terminated or discount == 0:
        return reward
    finite(next_value)
    return reward + discount * next_value


@dataclass(frozen=True)
class GaeRow:
    reward: float
    value: float
    next_value: float | None
    terminated: bool = False
    # Does the NEXT ROW belong to this same trajectory segment?
    # False at a reset, an unconnected rollout cut, or the end of available data.
    trace_continues: bool = False


def gae(rows, discount=0.9, trace_decay=0.8, tail_advantage=0.0):
    probability(discount)
    probability(trace_decay)
    finite(tail_advantage)
    future, result = tail_advantage, []
    for row in reversed(rows):
        finite(row.value)
        delta = td_target(row.reward, discount, row.next_value,
                          row.terminated) - row.value
        connected = row.trace_continues and not row.terminated
        advantage = delta + discount * trace_decay * future if connected else delta
        result.append(advantage)
        future = advantage
    return list(reversed(result))
```

$$
L^{\rm clip}=\min\!\left(\rho A,\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A\right).
$$

这是最大化目标。$A=2,\rho=1.4,\epsilon=0.2$ 时为 2.4；$A=-2,\rho=0.6$ 时为 −1.6；$A=-2,\rho=1.4$ 时为 −2.8。先裁剪所有 ratio 再乘优势，会把第三种情况算错。

| 家族 | 关键检查 | 一个有判别力的例子 |
| --- | --- | --- |
| Double DQN | 在线网络选动作、目标网络评估 | 在线 Q=(3,2)，目标 Q=(4,20)，必须评估 4 |
| PPO | old log-prob 固定、目标符号、优势的梯度边界 | 正负优势分别穿过上下裁剪边界 |
| SAC/TD3 | 终止、双 critic、动作尺度、目标更新时钟 | 终止样本不读取无效后继网络输出 |
| SAC | tanh 密度修正与温度目标 | 固定噪声核重参数化梯度；外部收益另报 |
| 循环网络 | 序列、burn-in、padding 和身份 | 改变填充内容不应改变有效时间步损失 |

离策略支持、PPO 标量目标与 Double Q 的最小核；不是完整训练器。

```python
def distribution(values):
    if not values:
        raise ValueError("empty policy")
    for value in values:
        probability(value)
    if not math.isclose(sum(values), 1.0, abs_tol=1e-12, rel_tol=0):
        raise ValueError("policy probabilities must sum to one")


def importance_ratio(target, behavior, action):
    distribution(target)
    distribution(behavior)
    if len(target) != len(behavior):
        raise ValueError("action spaces differ")
    if any(p > 0 and mu == 0 for p, mu in zip(target, behavior)):
        raise ValueError("target policy lacks behavior support")
    if not 0 <= action < len(target) or behavior[action] == 0:
        raise ValueError("recorded action is impossible under behavior policy")
    return target[action] / behavior[action]


def ppo_surrogate(ratio, advantage, clip=0.2):
    finite(ratio, advantage, clip)
    if ratio < 0 or not 0 <= clip < 1:
        raise ValueError("invalid PPO ratio or clip")
    clipped = max(1 - clip, min(1 + clip, ratio))
    # Objective to MAXIMIZE; a minimization loss negates this quantity.
    return min(ratio * advantage, clipped * advantage)


def double_q_target(reward, discount, online_next, target_next, terminated=False):
    if terminated or discount == 0:
        return td_target(reward, discount, None, terminated)
    if not online_next or len(online_next) != len(target_next):
        raise ValueError("invalid Q vectors")
    finite(*online_next, *target_next)
    selected = max(range(len(online_next)), key=online_next.__getitem__)
    return td_target(reward, discount, target_next[selected])
```

### 实验步骤

1. 在固定小批次上检查 gather、mask 和所有 stop-gradient。
2. 用极端但合法数值区分符号、版本和边界错误。
3. 数值核通过后再测试真实网络和数据管线。

### 验收

- 所有终止/截断/autoreset 情况都有明确预期。
- 目标网络、行为概率与 advantage 的版本可以定位。

### 常见错误

- 用一个 done 同时清理 bootstrap、GAE、循环记忆与 optimizer。
- 把一个小公式通过测试称作完整 PPO/SAC 复现。

原始来源：[Gymnasium · Handling Time Limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)；[Farama · Vector Environment Autoreset Modes](https://farama.org/Vector-Autoreset-Mode)；[Schulman et al. · Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)；[Schulman et al. · Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)；[Haarnoja et al. · Soft Actor-Critic](https://proceedings.mlr.press/v80/haarnoja18b.html)

---

<a id="handbook-control"></a>

## 5 · 对照、消融与公平预算

收益来自新机制，还是更多数据、参数和计算？

完整方法比较与共同基座实验回答不同问题。前者比较可实际使用的算法，各自保留关键网络、预处理、优化器与合理调参。后者固定基础系统，只改变一个机制，以理解原因。把作者方法移植到统一框架后，应说明哪些组成已经变化。

$$
I=J_{XY}-J_X-J_Y+J_0.
$$

两个模块 X、Y 的四格比较：基座、仅 X、仅 Y、X+Y。I 是选定收益尺度上的交互量。其正样本均值不自动证明可靠协同，仍需独立运行与不确定性分析。

| 要排除的解释 | 对照 | 仍存在的限制 |
| --- | --- | --- |
| 更多表示能力 | 匹配可训练容量的普通模型 | 相同参数量未必相同计算或归纳偏置 |
| 更新更小 | 调好的小步长、幅度匹配 | 相同范数仍可能不同方向和策略 KL |
| 信息更丰富 | 相同合法输入、固定/错位信号 | 打乱也可能改变频谱或边际分布 |
| 更多经验复用 | 匹配真实数据与更新次数 | 规划与反向传播的单次成本不同 |

以环境交互为主预算时，同时记录梯度步、模型调用、规划节点、内存和延迟。以墙钟为主预算时，固定硬件和计时范围。删除已训练模块的试验测量当前依赖；从头训练一个无该模块的系统，才测量缺少该模块时可达到的表现。

### 实验步骤

1. 每个对照只承担一项明确解释任务。
2. 将资源写为多维账本，指定主比较约束。
3. 对关键两模块做四格设计，再考虑更大组合。

### 验收

- 主方法与对手均有明确的信息和资源权限。
- 效应不能仅由预设的廉价对照解释。

### 常见错误

- 候选充分调参，对手只用来自另一个环境的默认值。
- 增加十倍规划后只宣称相同样本量更好，却不报告时延。

原始来源：[Patterson et al. · Empirical Design in Reinforcement Learning](https://jmlr.org/papers/v25/23-0183.html)；[Jordan et al. · Evaluating the Performance of Reinforcement Learning Algorithms](https://proceedings.mlr.press/v119/jordan20a.html)

---

<a id="handbook-selection"></a>

## 6 · 开发、调参与封存测试

测试的对象是一个配置，还是产生配置的程序？

开发数据用于发现错误、形成假设和选择候选。确认数据用于评价已锁定的方法。锁定的内容不仅是学习率，还包括预处理、指标、失败规则、预算、checkpoint 选择和提前停止规则。测试后修改其中任何一项，都需要承认已经使用了这批信息。

$$
\widehat h=\arg\max_{h\in\mathcal H}\overline J_{\rm dev}(h),\qquad \widehat J_{\rm test}=\frac1n\sum_iJ(\widehat h,\xi_i^{\rm test}).
$$

独立测试估计已选配置的性能。若主张某种搜索方法在预算 B 下更可靠，则外层要重复整个搜索和测试流程，不能仅对一次 sweep 的赢家画很窄的区间。

持续任务有两种合法但不同的协议：在独立开发生命期上完整搜索；或只在当前生命期的指定前缀选择，再在后续阶段部署。前者不等于偷看同一测试生命期的未来，后者也没有统一最佳前缀比例。在线自适应步长可以在测试中继续工作，但其规则和元参数必须事先确定。

| 记录 | 具体内容 | 验收问题 |
| --- | --- | --- |
| 搜索空间 | 全部候选、范围、采样分布 | 边界最优是否提示开发范围不足？ |
| 选择规则 | 均值、风险或约束的优先级 | 并列如何处理？ |
| 独立性 | 任务、种子、生命期和预训练来源 | 哪些随机层已共享？ |
| 成本 | 失败候选、评测、元训练与搜索 | 是否只统计最后赢家？ |

### 实验步骤

1. 开发时确定合理候选与预算，再固定选择程序。
2. 保存全部候选、失败和选择理由。
3. 使用未参与选择的样本运行确认，不依结果临时加跑赢家。

### 验收

- 能追溯每个测试配置如何被选择。
- 配置性能与搜索可靠性的主张分别报告。

### 常见错误

- 短预算筛选未经验证就假定长期排序不变。
- 公开测试种子反复使用后仍称其为未知确认集。

原始来源：[Jordan et al. · Evaluating the Performance of Reinforcement Learning Algorithms](https://proceedings.mlr.press/v119/jordan20a.html)；[Patterson et al. · Empirical Design in Reinforcement Learning](https://jmlr.org/papers/v25/23-0183.html)；[Mesbahi et al. · Lifetime Tuning Is Incompatible with Continual RL](https://proceedings.mlr.press/v267/mesbahi25a.html)

---

<a id="handbook-statistics"></a>

## 7 · 随机单位、失败与不确定性

一个种子、一个窗口和一场评测比赛各代表什么？

一次完整训练或生命期通常是算法比较的独立单位。同一曲线的窗口共享参数和历史；同一训练策略的很多评测回合主要降低执行噪声，不会增加训练重复数。只有一条生命期时，可以描述时间变化，但训练间置信区间不可由切窗制造。

$$
D_i=X_{A,i}-X_{B,i},\qquad \widehat\Delta=\frac1n\sum_iD_i,\qquad {\rm SE}(\widehat\Delta)=s_D/\sqrt n.
$$

配对以预先定义的外生情景为依据。相同整数 seed 不保证随机数调用或环境扰动对齐。若不存在合理配对，使用独立样本分析，而非事后配成较窄的区间。

配对 bootstrap 每次重采样完整配对索引。曲线区间则随索引带走整条轨迹。跨任务聚合先确定单位、归一化和任务权重；固定任务清单与从任务总体抽样是不同层级。逐时间点 95% 区间也不是整条曲线的 95% 同时覆盖。

| 终态 | 应保留的内容 | 分析规则 |
| --- | --- | --- |
| 算法数值失败 | 原因、发生时刻、已观测回报 | 预定失败效用或失败率与敏感性分析 |
| 基础设施中断 | 原 attempt 和重跑原因 | 按事先一致规则重试，保留身份 |
| 观察期结束未恢复 | 观察长度与恢复阈值 | 记录未恢复或相应右删失 |
| 尚未完成 | 当前前缀与真实状态 | 不能把缺失当已完成零分或静默删除 |

置信区间反映期望估计的不确定性，运行分位数反映风险，两者都重要。区间跨零不证明等价；极小正差即使统计显著，也未必有实际价值。运行数量应依据试点方差、最小有意义差异和所需精度决定，而不是统一规定三个或五个 seed。

### 实验步骤

1. 确定独立随机单位和预定主指标。
2. 保留计划、完成、失败与未完成的完整人口。
3. 画差值区间，同时展示所有 run、关键任务与尾部。

### 验收

- 每个统计样本对应真正独立的训练单位。
- 失败处理、任务权重和停止规则在看结果前确定。

### 常见错误

- 把同一生命期的阶段当作新的独立 seed。
- 只报告成功恢复者的平均恢复时间。

原始来源：[Agarwal et al. · Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)；[rliable · 作者统计实现](https://github.com/google-research/rliable)；[Patterson et al. · Empirical Design in Reinforcement Learning](https://jmlr.org/papers/v25/23-0183.html)

---

<a id="handbook-checkpoint"></a>

## 8 · 完整状态、精确恢复与诊断隔离

只保存网络参数，能否继续同一条学习轨迹？

算法的未来不只由权重决定。优化器、目标网络、资格迹、循环活动、归一化、replay、学习率日程、元状态和随机数都会改变下一步。继续同一模拟生命期还需要世界状态和环境随机数。无法恢复真实世界时，只能声明恢复了 learner，不能声称整条生命期精确重放。

恢复测试建立两个分支。分支 A 连续运行 n+m 步；分支 B 在 n 步序列化完整状态，重新创建对象后运行 m 步。比较后续观测、动作、奖励、写入和最终状态。确定性 CPU 小核可逐位一致；硬件非确定算子则需事先定义容差及不能精确重放的范围。

在线预测小过程：参数、动量、迹、归一化、环境状态和两个 RNG 一同恢复。

```python
class TinyProcess:
    """A synthetic online predictor with explicitly owned persistent state.

    Its purpose is checkpoint equivalence, not control performance. The trace,
    momentum, normalization and environment all affect subsequent updates.
    """
    def __init__(self, seed=7):
        self.weight = 0.1
        self.velocity = 0.0
        self.trace = 0.0
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.world = 1.0
        self.time = 0
        self.environment_rng = random.Random(seed)
        self.probe_rng = random.Random(seed + 1000)

    def step(self):
        # Only past normalization statistics are read for this prediction.
        scale = math.sqrt(self.m2 / self.count + 1) if self.count else 1.0
        feature = (self.world - self.mean) / scale
        feature *= -1 if self.probe_rng.random() < 0.5 else 1
        prediction = self.weight * feature
        target = 0.5 * feature + self.environment_rng.uniform(-0.1, 0.1)
        error = target - prediction
        self.trace = 0.4 * self.trace + feature
        self.velocity = 0.6 * self.velocity + error * self.trace
        self.weight += 0.01 * self.velocity
        self.count += 1
        delta = self.world - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (self.world - self.mean)
        self.world = 0.7 * self.world + self.environment_rng.uniform(-1, 1)
        self.time += 1
        return {"time": self.time, "prediction": prediction,
                "target": target, "weight": self.weight}

    def snapshot(self):
        fields = ("weight", "velocity", "trace", "count", "mean", "m2",
                  "world", "time")
        return {"version": 1, "state": {key: getattr(self, key) for key in fields},
                "environment_rng": self.environment_rng.getstate(),
                "probe_rng": self.probe_rng.getstate()}

    @classmethod
    def restore(cls, snapshot):
        required = {"weight", "velocity", "trace", "count", "mean", "m2",
                    "world", "time"}
        if snapshot["version"] != 1 or set(snapshot["state"]) != required:
            raise ValueError("incomplete or incompatible checkpoint")
        finite(*snapshot["state"].values())
        instance = cls(0)
        for key, value in snapshot["state"].items():
            setattr(instance, key, value)
        # JSON turns tuples into lists. random.setstate expects tuples.
        def tuples(value):
            return tuple(tuples(x) for x in value) if isinstance(value, (list, tuple)) else value
        instance.environment_rng.setstate(tuples(snapshot["environment_rng"]))
        instance.probe_rng.setstate(tuples(snapshot["probe_rng"]))
        return instance
```

脚本的 JSON 往返恢复后，后续 20 个事件完全相同。删除动量字段会被拒绝；只替换环境 RNG 会改变未来。它展示的是恢复合同测试，不是生产训练调度器。冻结评价还要创建隔离副本；多做一次诊断不应改变主运行的随机状态或统计量。

### 实验步骤

1. 列出所有能影响未来的持久状态及所有者。
2. 测试连续运行与保存恢复的一致性。
3. 增加评价频率，检查主训练轨迹是否被诊断污染。

### 验收

- 缺失状态字段被明确拒绝或标明不支持精确恢复。
- 评价副本不回写主参数、缓存、归一化和 RNG。

### 常见错误

- 只存 actor 权重，却继续声称完全恢复训练。
- 冻结权重但悄悄更新共享统计或消耗训练 RNG。

原始来源：[Patterson et al. · Empirical Design in Reinforcement Learning](https://jmlr.org/papers/v25/23-0183.html)；[Farama · Vector Environment Autoreset Modes](https://farama.org/Vector-Autoreset-Mode)

---

<a id="handbook-modules"></a>

## 9 · GVF、option 与模型的接口合同

一个模块正确，为什么还不足以说明整个 agent 更好？

GVF 问题由目标策略、累积信号和延续规则定义；option 由启动条件、内部策略和终止定义；模型则预测行为后果。每个模块还须声明输入何时可得、哪个参数版本产生输出、谁消费输出和持久状态何时重置。更换目标策略或 cumulant 会改变预测问题，而不只是更新同一个参数。

$$
G_t=\sum_{k\geq0}\left(\prod_{j=1}^{k}\gamma_{t+j}\right)c_{t+k+1},\qquad Y_o=\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}+\gamma^\tau V(X_{t+\tau}).
$$

第一式空乘积为 1，并要求相应回报及其期望存在；有界信号与统一小于 1 的折扣是一个充分条件。第二式对持续 $\tau$ 个原始转移的 option 使用常数折扣；一般 continuation 应改成逐转移乘积。option 终止通常只交回高层控制，不等于任务 terminal。

两步 option 的奖励为 1、2，γ=0.9，终点 V=3，目标为 $1+0.9×2+0.9^2×3=5.23$。若任务真正终止，目标为 2.8。若 option 持续时间与终点相关，不能先平均时长再折扣：一半 τ=1、V=2，一半 τ=3、V=6，真实后继项为 3.087，而 $γ^2×4$ 为 3.24。

逐原始时间累计回报；奖励率按总奖励除以总时长。

```python
def option_backup(rewards, continuations, next_value, task_terminal=False):
    if not rewards or len(rewards) != len(continuations):
        raise ValueError("one continuation is required per primitive transition")
    total, product = 0.0, 1.0
    for reward, continuation in zip(rewards, continuations):
        finite(reward)
        probability(continuation)
        total += product * reward
        product *= continuation
    # Option termination alone does not end the task.
    if not task_terminal and product != 0:
        finite(next_value)
        total += product * next_value
    return total


def reward_rate(reward_totals, durations):
    if not durations or len(reward_totals) != len(durations):
        raise ValueError("unaligned option outcomes")
    finite(*reward_totals, *durations)
    if any(duration <= 0 for duration in durations):
        raise ValueError("duration must be positive")
    return sum(reward_totals) / sum(durations)
```

| 模块层 | 独立测试 | 下游验收 |
| --- | --- | --- |
| GVF | 可解真值、γ=0、策略支持与目标版本 | 预测是否改变有用决策 |
| 子目标/option | 可达率、终止、持续时间和失败退出 | 新任务收益与发现成本 |
| 抽象/模型 | 同一抽象下奖励和后果混淆反例 | 组合后的动作排序和实际收益 |

### 实验步骤

1. 固定问题规格，再核单模块真值和时间边界。
2. 保持下游容量，比较不使用、固定和在线学习的模块。
3. 最后允许共同适应，记录目标与表示版本漂移。

### 验收

- 准确度、技能可达性和原任务收益分别测量。
- 发现、训练、执行与删除失败技能的成本都计入。

### 常见错误

- 把短期预测误差低当作状态充分性。
- 以 option 调用次数代替实际经过的环境时间。

原始来源：[Sutton et al. · Horde](https://josephmodayil.com/papers/horde-final.pdf)；[Sutton, Precup & Singh · Between MDPs and Semi-MDPs](https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/sutton%2Bal-1999.pdf)

---

<a id="handbook-planning"></a>

## 10 · 规划与架构：检查知识是否被使用

模型更准或搜索更深，为什么可能让策略更差？

规划主动选择高预测价值的分支，可能集中访问模型最乐观的错误区域。随机测试集上一小幅均方误差改善，不保证搜索后的行为更好。因此要同时测模型使用分布、奖励与终止错误、动作排序，以及真实执行结果。模型内高回报只是模型的预测。

| 模型 | 无额外规划 | 有额外规划 |
| --- | --- | --- |
| 固定模型 | 保持相同表示与基础策略 | 隔离规划计算的作用 |
| 在线模型 | 检验模型学习及辅助梯度 | 检验学习与规划交互 |
| 准确模型诊断 | 核基础决策机会 | 定位有限搜索或价值估计限制 |

准确模型加有限搜索也未必得到最优策略，不能自动称为理论上界。另设同真实交互的 model-free 基线和计算预算接近的额外学习对照。模型训练步、模拟步、搜索节点、候选数、动作延迟与真实交互分别计数。超时后的 fallback 行为也是算法的一部分。

组合架构时，按事件列出状态→预测→技能选择→模型→规划→行为的消费者关系。一次搜索使用固定模型快照，或明示并发更新的版本语义。共享表示改变后，旧技能和旧模型未必仍有相同含义；需记录版本并用同一批真实轨迹检查接口失配。

### 实验步骤

1. 先用准确小模型确认规划器确实能改变决策。
2. 替换成固定学习模型，测试模型误差被搜索放大的反例。
3. 最后开放在线模型学习，并比较固定/在线×有/无规划。

### 验收

- 增加模型质量后，指定消费者实际使用了改进。
- 真实收益与数据、计算、时延共同报告。

### 常见错误

- 同样环境步数下隐藏巨大模拟计算差异。
- 关闭一个已训练模块后失效，就断言该模块不可替代。

原始来源：[Sutton, Precup & Singh · Between MDPs and Semi-MDPs](https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/sutton%2Bal-1999.pdf)；[Patterson et al. · Empirical Design in Reinforcement Learning](https://jmlr.org/papers/v25/23-0183.html)

---

<a id="handbook-continual"></a>

## 11 · 持续学习：保留、获取与年龄效应

少遗忘是否只是从未学会？旧任务快恢复是否等于能学新任务？

A→B 检查变化后适应；A→B→A 检查重遇旧情境；加入从未出现的 C 检查新知识获取。日程、驻留时长和变化幅度是实验设计的一部分。研究者可以用变化标签做统计，但除非协议允许，不能把标签送给 agent 或在切换点专门重置它。

保留测试使用隔离的冻结读出；再适应测试允许学习，并记录新增交互。可塑性测试比较相同新任务、相同新增数据下的 aged 与 fresh learner，再分离重置表示、重置优化器和随机替换等机制。fresh 的较好表现可能来自不同表示或数据历史，不能用一个对照解释所有年龄差异。

| 现象 | 必要的并列指标 | 有判别力的对照 |
| --- | --- | --- |
| 旧 A 成绩低 | 之前是否达到习得门槛 | 冻结回访与允许重学 |
| 新 C 学得慢 | 获取曲线、失败和新数据量 | fresh、年龄匹配与经验匹配 |
| 内部状态适应快 | 冻结权重后的行为变化 | 保留记忆与清理记忆 |
| 长期回报高 | 所有阶段与资源增长 | 固定容量、同延迟与新日程 |

恢复阈值和连续达标窗口须预定。变化后奖励尺度不同，旧阶段回报可能不再是可达阈值。行政观察结束与算法崩溃也不是同一种恢复状态。完整生命期主指标应保留早期探索、低谷和失败，不只展示最终恢复后的片段。

### 实验步骤

1. 将动力学、奖励、观测和伙伴变化分开构造。
2. 在回访之外加入新任务，测保留和获取两条曲线。
3. 使用独立新日程确认，并检查存储和计算随年龄的增长。

### 验收

- 遗忘结论建立在先前确实学会的基础上。
- 在线主生命期与冻结诊断互不污染。

### 常见错误

- 在隐变化时刻给候选额外重置。
- 用有效秩或激活稀疏度代替真实新学习能力。

原始来源：[Mesbahi et al. · Lifetime Tuning Is Incompatible with Continual RL](https://proceedings.mlr.press/v267/mesbahi25a.html)；[Dohare et al. · Loss of Plasticity in Deep Continual Learning](https://www.nature.com/articles/s41586-024-07711-7)

---

<a id="handbook-streaming"></a>

## 12 · 流式学习：合法信息与每步预算

没有 replay 是否就意味着真正有界的在线学习？

逐步接收数据、只用最新样本和固定资源是不同约束。一个没有 replay 的算法仍可能保存随网络规模平方增长的敏感度，或重算越来越长的历史。因此协议须给出每个真实控制周期允许的计算、延迟、持久状态和数据重访次数。

$$
Z_{t+1}=F(Z_t,O_{t+1},A_t,R_{t+1}),\qquad A_{t+1}\sim\pi(\cdot\mid Z_{t+1}).
$$

Z 可以包含学习参数、资格迹、归一化与记忆。这里仅表示信息到达顺序，不保证 Z 是 Markov，也不保证计算量有界。未来轨迹统计不能作为当前归一化输入。

在线统计有两种可声明约定：用旧统计变换当前观测，再更新统计；或收到当前观测后先纳入它，再计算当前特征。两者都可因果，但结果不同。实现与原文必须保持同一顺序。用全生命期均值方差提前标准化，则使用了未来数据。

| 预算 | 记录内容 | 压力测试 |
| --- | --- | --- |
| 存储 | 参数、优化器、trace、模型、历史缓存 | 延长寿命后是否持续增长 |
| 时延 | 每步和高分位延迟、超时次数 | 复杂观察或替换事件是否错过控制周期 |
| 数据访问 | 每条样本被读几次，来源何处 | 移除重复读取后是否仍运行 |
| 信息 | 观察、诊断标签、未来日程的权限 | 改变隔离标签不应改变部署动作 |

### 实验步骤

1. 给每个状态量标明大小、更新时钟和重置规则。
2. 修改未来数据，检查已完成前缀的行为与更新不变。
3. 在固定控制时钟下记录超时和 fallback，而不仅是平均训练速度。

### 验收

- 信息因果性与资源上限各有独立检查。
- 元梯度和敏感度的额外状态、计算均计入。

### 常见错误

- 把 GPU 并行吞吐直接当单个物理 agent 的响应时间。
- 因为使用在线算法名称就假定它从不读取历史。

原始来源：[Patterson et al. · Empirical Design in Reinforcement Learning](https://jmlr.org/papers/v25/23-0183.html)；[Sutton et al. · Horde](https://josephmodayil.com/papers/horde-final.pdf)

---

<a id="handbook-marl"></a>

## 13 · 多智能体：权限、身份与训练人口

同一队中的十个 agent，能算十次独立算法实验吗？

多智能体实验要先定义合作或竞争目标，以及世界联合步、实际 agent 动作和学习更新三种计数。团队 reward 若已经是一次联合得分，不能再按人数重复求和。轮流行动 API 的一次调用也不必等于世界推进一次。

$$
T_{\rm agent}=\sum_{t=1}^{T_{\rm joint}}N_t.
$$

$N_t$ 是该联合转移中实际行动的个体数。独立统计单位通常是完整训练团队或人口；队员、共享训练得到的多个 checkpoint 和重复比赛不是新的独立训练样本。

集中训练可以合法地给 critic 全局信息，分散执行的 actor 则只能读取本地历史与允许消息。固定 actor 的合法输入和随机状态，改变中央诊断数据，不应改变部署动作。通过中央 critic 学到的参数仍属于合法训练机制；测试的是执行时信息通路，而不是禁止中央训练梯度。

| 测试对象 | 最小实验 | 常见错误 |
| --- | --- | --- |
| 身份与记忆 | 交换输入字典顺序、个体离开后重入 | 按遍历位置交换 hidden state |
| 奖励与步数 | 两个 agent 一次联合奖励 | 累计 reward 与即时 reward 重复相加 |
| 活跃与终止 | 一个个体离开，团队仍继续 | 把死亡、loss mask、bootstrap 合成一位 |
| 伙伴泛化 | 独立训练团队的 cross-play 矩阵 | 同队 self-play 胜率等同陌生伙伴协作 |

先固定队友，核接口和可学性；再隐藏更换冻结队友；最后允许所有成员共同学习。比较方法时使用相同的外部评价伙伴池，记录其来源和生成成本。若多组模型共享同一训练伙伴池，交叉矩阵的行列具有依赖，不能把所有比赛单元直接当 IID 样本。

### 实验步骤

1. 为每个张量登记 actor、critic 或仅诊断权限。
2. 用两 agent 微型游戏核奖励、身份、缺席与终止。
3. 按独立团队或人口组织重复，分开 self-play 与 cross-play。

### 验收

- 所有方法执行权限相同，通信和对手生成成本明确。
- 伙伴变化与环境任务变化能够单独检验。

### 常见错误

- 把参数共享读成中央执行，或把 independent 读成不共享参数。
- 注意力或值分解系数直接解释为真实因果贡献。

原始来源：[Rashid et al. · QMIX](https://proceedings.mlr.press/v80/rashid18a.html)；[Hu et al. · Other-Play for Zero-Shot Coordination](https://proceedings.mlr.press/v119/hu20a.html)；[PettingZoo · AEC API](https://pettingzoo.farama.org/api/aec/)；[MAPPO · 作者实现](https://github.com/marlbenchmark/on-policy)

---

<a id="handbook-report"></a>

## 14 · 从可运行测试到可复核研究报告

另一位读者需要什么，才能复核你的结论？

交付应能重建问题和比较人口，而不只是启动一条训练命令。保留完整配置、依赖、源版本、各候选和全部 attempt 的终态。原始奖励与后处理统计分开；平滑曲线、归一化和剔除规则可从分析入口复算。哈希能检查文件是否改变，但不自动证明科学设计正确。

| 层次 | 应交付的材料 | 通过条件 |
| --- | --- | --- |
| 规格 | 目标、信息、预算、版本和竞争解释 | 他人能说明究竟比较什么 |
| 算法测试 | 固定样本、边界、梯度与恢复测试 | 预期值独立于被测实现 |
| 机制 | 控制矩阵、正反例与中间量 | 解释能够被反证并接受负结果 |
| 效能 | 完整独立运行、失败、差值与区间 | 结果与预定比较一致 |
| 决定 | 保留、修订、确认或停止的理由 | 下一步针对缺失证据，不只是增加模块 |

算法接口教学测试：仅标准库，不创建或覆盖文件

```sh
python3 algorithm_testing_lab.py test
python3 algorithm_testing_lab.py demo
```

配套脚本检查标量更新核、时钟和恢复语义。它没有环境包装器、神经网络优化器或完整训练 runner，因此不能代替原作者深度算法的复现。完成这些低成本检查后，再对原实现增加同类断言，运行小型闭环，最后进入锁定配置的独立性能比较。

与实验设计章的非平稳 bandit 配合使用：先在本页定位算法正确性与数据边界，再运行开发选择和独立测试。若确认结果不理想，保留旧版本和负结果，重新提出一个可被区分的解释，而不是根据测试结果继续改指标。

### 实验步骤

1. 运行教学脚本，并把对应测试迁移到实际算法。
2. 生成包含全体运行的分析表和可复算图。
3. 写出最强反证、适用范围和下一项有预算的决定。

### 验收

- 代码、结果、图表和主张可以逐项对应。
- 小核测试、原方法复现和新算法效能状态明确区分。

### 常见错误

- 只上传最好 checkpoint 或完成的幸存 run。
- 把安装成功、测试数量或源代码行数当成科研进展。

原始来源：[Patterson et al. · Empirical Design in Reinforcement Learning](https://jmlr.org/papers/v25/23-0183.html)；[Jordan et al. · Evaluating the Performance of Reinforcement Learning Algorithms](https://proceedings.mlr.press/v119/jordan20a.html)；[rliable · 作者统计实现](https://github.com/google-research/rliable)

## 下载

[算法测试脚本](https://yingwen.io/zh/continual-rl/download/algorithm_testing_lab.py)

[完整开发选择与独立测试示例](https://yingwen.io/zh/continual-rl/experiments/)
