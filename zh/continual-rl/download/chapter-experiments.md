# 实验设计：从更新正确到持续学习证据

一个算法通过测试、曲线更高，分别能说明什么？怎样用有限预算得到可以重复检验的结论？

## 本章内容

- 明确估计对象、独立随机单位、信息权限与资源预算。
- 完成开发选择、独立测试、整次运行配对 bootstrap 的可执行实验。
- 分别解释在线收益、冻结诊断、恢复、失败和跨任务推广。

<a id="problem-definition"></a>

## 本章的问题定义

确定有限数据能够支持哪种算法结论；实现一致、机制解释、独立比较和适用范围分别需要证据。

### 给定条件与符号

- 完整算法、封存配置、预定寿命与世界/任务分布。
- 开发与测试拆分、随机单位、基线、指标、资源和特权信息预算。

### 需要求解的对象

声明评价量的估计、差值与不确定性，以及与该估计范围匹配的结论。

### 信息与数据权限

学习器只见所选动作的已到达后果；实验控制器可管理种子与诊断，但变化时间、潜在未选奖励和测试排序不得回流到封存算法选择。

$$
\Delta_T=\mathbb E_\xi[X(A,h_A,\xi;T)-X(B,h_B,\xi;T)],\qquad X(A,h,\xi;T)=\frac1T\sum_{t=0}^{T-1}R_{t+1}
$$

$A,B$ 为完整算法，$h_A,h_B$ 为开发后封存的配置，$\xi$ 为运行随机性，$T$ 为寿命。目标是有限寿命在线平均差；无限奖励率、最佳调参器性能和最后冻结策略分数是其他量。整次运行是本章的独立单位。

### 成立条件与解的含义

- 独立测试不参与设计选择；配对需有合理共同外生随机性并保留各自闭环轨迹。
- 区间依赖独立单位和重采样协议；时间点不作为独立重复，缺失失败和非有限运行有预定处理。

判断准则：手算/有限差分/边界测试与所写更新一致；开发测试无重叠；整次运行差值及区间可重算，保留负差、失败和全部预算，结论限定于覆盖条件。

### 适用边界

- 测试通过不是算法长期有效或回报优势的证据。
- 开发集最高分和tiny有符号差不能直接称为已证优势。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 改变信息或数据协议 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：完整生命期控制比较要求独立世界、封存设计和可实现比较器。

- 组合不同学习问题 · [可塑性与特征更新](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)：可塑性假设需要匹配probe与干预，在线奖励曲线不足以定位原因。

- 组合不同学习问题 · [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)：冻结诊断、恢复和在线表现估计不同对象，副本评价不回流训练。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

研究者选择、时间依赖与信息不公平会使高曲线看似有优势，即使差异来自预算或噪声。

### 本章的核心思路

先固定估计对象与独立单位，再分离选择和测试；实现与机制证据单独连接到回报比较。

1. [隔离配置选择与性能估计](#lesson-derive)：因为开发赢家带有选择噪声，用独立测试评价封存配置；若要评价搜索器，需重复整个搜索。

2. [以完整生命期处理依赖](#lesson-randomness)：因为同曲线时间点共享参数和探索历史，整次运行作为独立单位，配对按允许的共同随机性定义。

3. [报告差值和范围而非赢家标签](#lesson-intervals)：因为有限样本有不确定性，对整次运行重采样并保留负差、失败与选择规则，再限制结论范围。

结论与条件：独立测试可避免复用选择噪声；bootstrap区间依赖数据和采样假设，不证明未知任务普适优势或机制因果性。

### 相关方法改变了什么

- 实现测试：核对更新与边界，不能替代环境收益实验。

- 受控机制实验：用干预与匹配资源检验具体解释，范围通常较窄。

- 独立寿命比较：估计封存算法的在线差值及不确定性，推广需追加未见条件。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 随机变量与样本均值

相同算法多次运行会产生不同收益。单次结果为 $X_i$，总体期望为 $\mu=\mathbb E[X]$，样本均值为 $\bar X=n^{-1}\sum_iX_i$。

### 在线学习

动作依据当前可得历史产生。环境反馈发生后，学习器才更新参数。测试一个持续学习算法时，学习本身通常仍然开启。

### 实验控制器

它管理随机分配、预算、记录与评价。环境提供反馈，学习器只接收协议允许的信息。统计分析可以读取真实变化时间，但不能将其回传给学习器。

<a id="lesson-setting"></a>

## 1 · 问题、估计对象与四级证据

考虑一个需要长期选择服务策略的系统。方法 A 启动慢，但后期表现好；方法 B 较早学会，却在环境变化后退化。最后一段收益、全程收益和冻结策略得分可能给出不同排序。实验需要先决定哪个问题重要，再决定记录与聚合方式。

$$
X(A,h,\xi;T)=\frac1T\sum_{t=0}^{T-1}R_{t+1},\qquad \mu(A,h;T)=\mathbb E_\xi[X(A,h,\xi;T)].
$$

$A$ 是完整学习算法，$h$ 是固定配置，$\xi$ 包含环境和学习器的随机性，$T$ 是预定寿命。这里估计有限寿命的在线平均奖励，不是已经证明存在的无限时域奖励率。

Patterson 等的 Empirical Design in Reinforcement Learning 将评价量、变异、超参数和研究者选择放在同一实验流程中。可以据此把证据分成四级。它们支持不同强度的主张，不能互相替代。

| 证据层级 | 核心问题 | 具体实验 | 能够支持的结论 |
| --- | --- | --- | --- |
| 更新与实现 | 计算是否符合所写算法？ | 手算、有限差分、边界、恢复运行 | 指定输入与条件下实现一致 |
| 机制 | 改动是否按假设改变学习过程？ | 控制变量、反例、等容量与等计算对照 | 在受控条件下支持特定解释 |
| 独立比较 | 收益差是否能在新运行重现？ | 锁定配置、独立重复、差值区间 | 指定分布和预算下的差异 |
| 适用范围 | 改变任务与时间尺度后是否仍成立？ | 未见任务、长寿命、资源压力测试 | 覆盖到的条件及已知失效边界 |

较小的 TD 误差是机制量。较高的行为奖励是控制结果。一个数值恒等式是实现证据。三者之间需要实验连接。例如减小步长可以减少更新尖峰，也可能让变化后的恢复更慢。仅展示更新范数下降，无法判断完整学习器是否更好。

<a id="lesson-derive"></a>

## 2 · 配置性能与选择过程的性能

固定配置的性能，与搜索后所得方法的性能，是不同统计对象。设开发数据用于从候选配置中选择一个配置。测试必须在选择完成后独立进行。否则同一批幸运噪声既帮某配置胜出，又被拿来报告其优势。

$$
\widehat h=\arg\max_{h\in\mathcal H}\overline X_{\rm dev}(h),\qquad \widehat\mu_{\rm test}=\frac1n\sum_{i=1}^n X(A,\widehat h,\xi_i^{\rm test};T).
$$

测试随机性必须独立于选择过程。该均值估计的是已选配置 $\widehat h$ 的性能。重复整个搜索流程的期望则为 $\Psi(B)=\mathbb E_{D_{\rm dev},\xi_{\rm search}}[\mu(A,\widehat h_B;T)]$，需要把搜索作为外层重复单位。

$$
\mathbb E\!\left[\max_h\overline X_{\rm dev}(h)\right]\geq\max_h\mathbb E[\overline X_{\rm dev}(h)].
$$

最大值运算保留了向上的噪声。这个不等式说明为何开发集上的最高均值通常过于乐观；独立测试消除重复使用这批选择噪声的问题，但不会证明选到了未知的全局最优配置。

| 数据层 | 允许的用途 | 禁止的信息流 |
| --- | --- | --- |
| 训练交互 | 每个 run 内选择动作和更新 | 读取未发生的奖励或未选择动作的反馈 |
| 开发与选择 | 查错、设计候选、选择参数 | 将反复查看的数据仍称为独立测试 |
| 独立测试 | 评价锁定配置的完整学习过程 | 依据测试排序再换参数、换指标或只加跑赢家 |

RL 的“测试”不必意味着冻结参数。若评价的是持续学习算法，测试 run 从初始化开始，按锁定的更新规则持续学习；研究者不再根据这批结果修改规则。若评价的是固定部署策略，才冻结策略参数。两种协议对应不同对象。

Jordan 等将参数设置或自适应机制纳入完整算法定义，关注新任务上的可用性与调参成本。Mesbahi 等进一步讨论 lifetime tuning：在整个未来寿命上反复搜索，可能掩盖长期适应困难。限制开发可见前缀是一种具体研究协议，前缀比例并没有统一最佳值。

本章实验只用开发生命期的早期前缀选择参数，再运行独立测试生命期的全部三个阶段。变化时间不提供给学习器。这个设计检验的是早期选择后能否适应后续变化；它不等价于已经评价了整个调参器的可靠性。

<a id="lesson-implementation"></a>

## 3 · 更新测试与梯度测试

实现测试应从能手算的对象开始。固定目标的标量回归足以检查误差符号、步长与更新顺序。随后再加入 bootstrap、重要性比、资格迹、目标网络和终止语义。每增加一种依赖，就增加一项能区分错误版本的测试。

$$
L(q)=\tfrac12(q-r)^2,\qquad \nabla_qL=q-r,\qquad q^+=q+\alpha(r-q).
$$

取 $q=0.5$、$r=1$、$\alpha=0.2$，应得到 $q^+=0.6$。本章 bandit 的更新就是这个式子，因此可以把手算、代码和梯度检查逐项对齐。

$$
g_j^{\rm FD}=\frac{L(\theta+\varepsilon e_j)-L(\theta-\varepsilon e_j)}{2\varepsilon}.
$$

中心差分应固定样本与随机噪声，并尝试多个 $\varepsilon$。过大有截断误差，过小有舍入误差。若检查 TD 半梯度，bootstrap 标签在两侧扰动中也必须固定；重新计算标签检查的是另一个梯度。

| 测试 | 本章实例 | 扩展到深度 RL |
| --- | --- | --- |
| 单步精确值 | 常数步长后为 0.6；递减式在 decay_scale=1、第二次奖励为 0 时得到 0.54 | 检查全部辅助变量的旧值与新值 |
| 退化条件 | 零步长不改变价值 | 零 trace、零重要性比、关闭正则 |
| 边界 | 运行结束、无恢复、失败后停止 | terminated、truncated、批末与自动重置 |
| 因果性 | 修改未来奖励不改变过去前缀 | 归一化、目标构造及元梯度不使用未来信息 |
| 统计接口 | 重复种子、拆分重叠、非有限值被拒绝 | 计划 run 与实际产物一一对应 |

测试通过说明实现满足这些检查。算法长期稳定、回报优势和迁移能力仍属于后续证据。反例也应保留：如果一种步长控制只在有限特征尺度下成立，超出条件后的失败能帮助读者理解其适用范围。

<a id="lesson-budget"></a>

## 4 · 信息权限与公平预算

相同环境步数回答样本效率问题。相同墙钟时间回答特定硬件上的计算效率问题。相同梯度步数只匹配部分优化成本。三种约束通常不能同时满足。主比较选定一个约束，再报告其他资源坐标。

| 资源或权限 | 需要声明的内容 | 典型混杂 |
| --- | --- | --- |
| 交互 | 原始环境步、帧跳过、恢复、额外评测 | 把一个 option 决策与一个原始动作视为等量数据 |
| 计算 | 网络前后向、规划、模型滚动、生成样本 | 更高更新比带来的收益归给新公式 |
| 内存 | 参数、优化器、trace、replay、模型、历史副本 | 只统计网络参数 |
| 调参 | 候选数、每个候选的 run 数、可见前缀 | 只报告胜出配置的训练成本 |
| 特权信息 | 任务 ID、真实状态、变化时间、真模型 | 把信息优势归给学习能力 |

本实验的两个方法都保持两个动作价值和两个计数，每步只使用一个实际反馈。候选集合、探索概率、初始化、开发种子及选择目标相同。差别是步长是否随某个动作的访问次数递减。环境控制器可以生成所有动作的潜在奖励，但只把所选动作的奖励交给学习器。

标准运行每个方法使用三个候选、八个开发 run、每个前缀 240 步，共 5760 次开发交互。独立测试为 24 个 run，每个 900 步，共 21600 次测试交互。环境配对不减少实际部署成本；它只是模拟实验中的方差控制。日志与奖励表属于实验控制器，不能据此声称整个研究程序只有常数内存。

<a id="lesson-randomness"></a>

## 5 · 完整生命期作为随机单位

同一学习曲线中的相邻奖励相互依赖。它们共享参数、探索历史与先前经验。因此 900 个时间点不等于 900 个独立算法样本。这里一次从初始化到结束的完整生命期是一个样本。开发 run 和测试 run 使用互不重叠的种子集合。

仅给两个算法相同的整数 seed，不能保证它们面对相同随机事件。不同控制流可能消耗不同数量的随机数。本实验按时间和动作预先生成外生 Bernoulli 奖励表；按时间再生成固定三个动作随机数。不同方法在同一时间选择同一动作，会得到同一潜在反馈。方法间仍可选择不同动作和形成不同历史。

$$
R_{t+1}(a)=\mathbf1[U_{t,a}<p_t(a)],\qquad U_{t,a}\overset{\rm iid}{\sim}\operatorname{Uniform}(0,1).
$$

一个配对共享同一张 $U$ 表及同一组动作随机数。不同配对独立生成。共享外生情景保持各方法自己的边际分布；它不使两个方法的轨迹相同，也不保证差值方差一定降低。

环境的潜在反馈与学习器分开；学习器只接收所选动作的奖励。

```python
@dataclass(frozen=True)
class Scenario:
    seed: int
    # Controller-only potential outcomes; the learner never receives this table.
    rewards: tuple[tuple[float, float], ...]
    action_uniforms: tuple[tuple[float, float, float], ...]


def make_scenario(seed: int, horizon: int) -> Scenario:
    if horizon < 3 or horizon % 3:
        raise ValueError("horizon must be a positive multiple of three")
    env_rng, action_rng = rng_for(seed, "environment"), rng_for(seed, "actions")
    rewards, uniforms = [], []
    phase_length = horizon // 3
    for t in range(horizon):
        probabilities = (0.2, 0.8) if t // phase_length == 1 else (0.8, 0.2)
        rewards.append(tuple(float(env_rng.random() < p) for p in probabilities))
        # Exactly three action uniforms per time index, regardless of branching.
        uniforms.append(tuple(action_rng.random() for _ in range(3)))
    return Scenario(seed, tuple(rewards), tuple(uniforms))


class Learner:
    def __init__(self, config: Config):
        self.config = config
        self.q = [0.5, 0.5]
        self.counts = [0, 0]

    def choose(self, uniforms):
        explore, arm, tie = uniforms
        if explore < self.config.epsilon:
            return min(1, int(2 * arm))
        best = max(self.q)
        choices = [a for a, value in enumerate(self.q) if value == best]
        return choices[min(len(choices) - 1, int(tie * len(choices)))]

    def update(self, action, reward):
        self.counts[action] += 1
        step = self.config.alpha
        if self.config.method == "decay":
            step /= 1 + (self.counts[action] - 1) / self.config.decay_scale
        self.q[action] += step * (reward - self.q[action])
        if not all(math.isfinite(value) for value in self.q):
            raise FloatingPointError("non-finite value estimate")
```

$$
D_i=X_{A,i}-X_{B,i},\quad \widehat\Delta=\frac1n\sum_iD_i,\quad \operatorname{Var}(D)=\operatorname{Var}(X_A)+\operatorname{Var}(X_B)-2\operatorname{Cov}(X_A,X_B).
$$

配对的价值取决于两个方法结果的协方差。正相关可能降低差值方差。配对规则须在看结果前固定；不能事后挑选看起来最有利的匹配。

<a id="lesson-intervals"></a>

## 6 · 配对 bootstrap 与区间含义

把每个完整 run 汇总成预定主指标，得到成对分数。一次 bootstrap 从这些配对索引中有放回抽取与原样本相同数量的索引，再计算差值均值。重复这个过程，用重采样分布的分位数构造 percentile 区间。

$$
I_1^{(b)},\ldots,I_n^{(b)}\overset{\rm iid}{\sim}\operatorname{Uniform}\{1,\ldots,n\},\qquad\widehat\Delta^{(b)}=\frac1n\sum_{j=1}^nD_{I_j^{(b)}}.
$$

95% percentile 区间取这些均值的 2.5% 与 97.5% 分位数。重采样的是整对 run；不独立重采样同一曲线中的时间点，也不分别打散两个方法的配对。

运行级配对重采样，返回差值均值、区间与配对差的标准误。

```python
def paired_bootstrap(first, second, reps=2000, seed=917, confidence=0.95):
    """Percentile CI for a fixed pair of configurations, resampling whole runs."""
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("at least two aligned run pairs are required")
    if reps < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    finite(first + second)
    differences = [a - b for a, b in zip(first, second)]
    rng = random.Random(seed)
    n = len(differences)
    samples = [statistics.fmean(differences[rng.randrange(n)] for _ in range(n))
               for _ in range(reps)]
    tail = (1 - confidence) / 2
    return {"mean_difference": statistics.fmean(differences),
            "ci": [quantile(samples, tail), quantile(samples, 1 - tail)],
            "standard_error": statistics.stdev(differences) / math.sqrt(n),
            "pairs": n, "bootstrap_replicates": reps,
            "interval": "run-paired percentile; pointwise, conditional on selection"}
```

置信区间描述估计程序的不确定性，不是下一次运行的分数范围。标准差与经验分位数描述运行差异。很窄的均值区间可以与很不稳定的单次表现同时存在。Bootstrap 次数只控制重采样的数值精度；它不会增加真实 run 数，也无法创造尚未观察到的稀有失败。

本章只有一个构造环境。少量 run 的区间是教学性的近似，不保证标称覆盖率精确成立。Agarwal 等的 rliable 面向多任务基准，使用任务内运行重采样和 IQM 等汇总。固定任务集通常保持任务不变；要推广到新任务总体，还需定义任务抽样层级。运行数不同的任务也不能无说明直接摊平。

| 输出 | 回答的问题 | 不直接回答 |
| --- | --- | --- |
| 均值差及区间 | 锁定配置的平均收益差 | 哪个方法在每次运行都更好 |
| 全部 run 分数 | 波动、多峰与尾部表现 | 未知总体的完整失败模式 |
| IQM 或性能剖面 | 指定任务混合上的稳健表现 | 任意未见任务上的优势 |
| 逐时间点区间 | 特定时刻均值的不确定性 | 整条曲线同时有 95% 覆盖 |

<a id="lesson-example"></a>

## 7 · 可复算的非平稳 bandit

环境分三个等长阶段。两个动作的成功概率依次是 (0.8,0.2)、(0.2,0.8)、(0.8,0.2)。奖励为 0 或 1。智能体在阶段切换时不重置，也不接收阶段编号。两个价值都初始化为 0.5，采用探索概率 0.1 的 epsilon-greedy；并列最大值随机选择。

$$
Q_n(a)=Q_{n-1}(a)+\alpha_n[R_n-Q_{n-1}(a)],\qquad \alpha_n^{\rm constant}=\alpha_0,\quad\alpha_n^{\rm decay}=\frac{\alpha_0}{1+(n-1)/20}.
$$

$n$ 是所选动作含本次在内的访问次数，$R_n$ 是该动作第 $n$ 次被选择时的奖励。每个方法均从 $\alpha_0\in\{0.05,0.15,0.4\}$ 选择。递减式不是样本均值公式；它是一个有明确速度参数的对照更新。

常数步长将旧奖励的权重按几何速度衰减。若某个动作持续被采样，旧估计的贡献在后续若干次更新后越来越小。递减步长则使后期新反馈影响更弱。这给出“适应较慢”的机制假设，但控制结果还取决于探索是否重新访问已变好的动作。

$$
Q_n=(1-\alpha)^nQ_0+\alpha\sum_{j=1}^n(1-\alpha)^{n-j}R_j.
$$

这是同一动作按常数步长更新的精确展开。它描述访问次数下的记忆长度；若策略很少再选该动作，真实时间中的恢复仍可能很慢。

手算一次配对分析：三个独立生命期得到 A=[0.7,0.5,0.6]、B=[0.6,0.6,0.4]。差值为 [0.1,−0.1,0.2]，均值为 1/15。一次索引重采样 [3,3,1] 得到均值 1/6；[2,2,1] 得到 −1/30。重采样变动的是完整运行的组成，而非重新运行算法。

| tiny 命令的输出 | constant | decay |
| --- | --- | --- |
| 开发选择的初始步长 | 0.05 | 0.05 |
| 独立测试全程平均收益 | 0.595833 | 0.562500 |
| 第一阶段平均收益 | 0.700000 | 0.712500 |
| 反转阶段平均收益 | 0.504167 | 0.300000 |
| 恢复原概率阶段平均收益 | 0.583333 | 0.675000 |

tiny 模式使用四个开发 run、八个独立测试 run，每次 90 步。选择只看开发的前 30 步。固定种子得到平均差 0.033333，500 次配对 bootstrap 的区间约为 [−0.002118,0.060451]。区间包含零，因此这个小例子没有确定平均差的符号。第三阶段又出现递减方法较高的结果，显示全程均值之外还存在阶段取舍。

这些数字是可执行示例的输出。它们不支持深度网络或任意 CRL 环境的结论。公开固定种子有利于学习和复现；读者若据此反复修改算法，这批种子便成为开发数据，新的确认实验需要另外预定的运行。

<a id="lesson-code"></a>

## 8 · 选择、测试与结果保存

选择函数不接收测试数据；两种配置确定后才开始测试，所有候选和逐 run 指标均保留。

```python
def select_on_development(method, candidates, dev_seeds, horizon, prefix):
    """No test seeds, test outcomes or test score are accepted by this function."""
    unique_seeds(dev_seeds)
    if not 1 <= prefix <= horizon or not candidates:
        raise ValueError("invalid development budget")
    records = []
    for alpha in candidates:
        config = Config(method, alpha)
        runs = [run_lifetime(make_scenario(seed, horizon), config, limit=prefix)
                for seed in dev_seeds]
        records.append({"alpha": alpha, "score": statistics.fmean(map(mean_reward, runs)),
                        "failures": sum(run.status == "failed" for run in runs)})
    # Stable candidate order is the predeclared tie-break; no test-set tie-break.
    chosen = max(range(len(records)), key=lambda i: records[i]["score"])
    return Config(method, candidates[chosen]), records


def experiment(dev_seeds, test_seeds, horizon=900, prefix=240,
               candidates=(0.05, 0.15, 0.4), window=30, reps=2000, raw=False):
    unique_seeds(dev_seeds)
    unique_seeds(test_seeds)
    if set(dev_seeds) & set(test_seeds):
        raise ValueError("development and test seeds must be disjoint")
    if (len(test_seeds) < 2 or horizon % 3 or not 1 <= window <= horizon // 3
            or (horizon // 3) % window):
        raise ValueError("invalid test design")
    selected, development = {}, {}
    for method in ("constant", "decay"):
        selected[method], development[method] = select_on_development(
            method, candidates, dev_seeds, horizon, prefix)
    # Lock both configurations before any test scenario is evaluated.
    test_runs = {method: [] for method in selected}
    for seed in test_seeds:
        scenario = make_scenario(seed, horizon)
        for method, config in selected.items():
            test_runs[method].append(run_lifetime(scenario, config))
    records = {method: [run_metrics(run, window) for run in runs]
               for method, runs in test_runs.items()}
    comparison = paired_bootstrap(
        [mean_reward(run) for run in test_runs["constant"]],
        [mean_reward(run) for run in test_runs["decay"]], reps=reps)
    summary = {method: {
        "mean_lifetime_rate": statistics.fmean(item["lifetime_rate"] for item in rows),
        "mean_phase_rates": [statistics.fmean(item["phase_rates"][p] for item in rows)
                             for p in range(3)],
        "failures": sum(item["status"] == "failed" for item in rows),
        "recovery_counts": [{status: sum(item["recovery"][p]["status"] == status
                                         for item in rows)
                             for status in ("recovered", "right_censored", "failed")}
                            for p in range(2)]} for method, rows in records.items()}
    report = {
        "protocol": {"horizon": horizon, "development_prefix": prefix,
                     "dev_seeds": dev_seeds, "test_seeds": test_seeds,
                     "candidates": list(candidates), "epsilon": 0.1,
                     "decay_scale": 20, "window": window,
                     "pairing": "shared time-indexed potential rewards and action uniforms",
                     "estimand": "selected-configuration lifetime utility; not HPO reliability",
                     "failure_utility": "observed reward followed by zero after stopped service",
                     "development_interactions_per_family": len(candidates) * len(dev_seeds) * prefix,
                     "test_interactions_per_family": len(test_seeds) * horizon},
        "selected": {method: asdict(config) for method, config in selected.items()},
        "development_scores": development, "test_summary": summary,
        "constant_minus_decay": comparison, "per_run": records}
    if raw:
        report["raw_runs"] = {method: [asdict(run) for run in runs]
                              for method, runs in test_runs.items()}
    return report
```

行为收益在更新前记录。原始观测与失败后的部署效用分开保存。

```python
def run_lifetime(scenario: Scenario, config: Config, limit=None,
                 freeze_at=None, fault_at=None) -> Run:
    horizon = len(scenario.rewards) if limit is None else limit
    if not 1 <= horizon <= len(scenario.rewards):
        raise ValueError("invalid observation budget")
    for boundary in (freeze_at, fault_at):
        if boundary is not None and not 0 <= boundary < horizon:
            raise ValueError("boundary outside run")
    learner = Learner(config)
    observed, actions, failure = [], [], None
    for t in range(horizon):
        action = learner.choose(scenario.action_uniforms[t])
        reward = scenario.rewards[t][action]
        # Score behavior before learning. Unchosen rewards stay controller-only.
        observed.append(reward)
        actions.append(action)
        try:
            if t == fault_at:
                raise FloatingPointError("injected diagnostic fault")
            if freeze_at is None or t < freeze_at:
                learner.update(action, reward)
        except FloatingPointError as error:
            failure = {"after_step": t + 1, "reason": str(error),
                       "injected": fault_at is not None}
            break
    # Predeclared deployment utility: stopped service earns 0 for remaining time.
    # This tail is NOT observed reward; retain both arrays and the failure event.
    utility = tuple(observed + [0.0] * (horizon - len(observed)))
    # JSON has no non-finite numbers. Preserve the failure event and use null for
    # invalid final parameters; actual finite rewards and scores remain intact.
    final_q = tuple(value if math.isfinite(value) else None for value in learner.q)
    return Run(scenario.seed, config, horizon, tuple(observed), tuple(actions),
               utility, final_q, tuple(learner.counts),
               "failed" if failure else "complete", failure)
```

Python 3.10+，仅标准库。所有输出写到标准输出，不创建或覆盖文件。

```sh
python experiment_design_lab.py test
python experiment_design_lab.py demo --tiny
python experiment_design_lab.py demo
python experiment_design_lab.py demo --raw
```

test 检查手算、有限差分、零步长、未来数据隔离、拆分重叠、相同算法配对、恢复删失、失败记录及冻结副本。demo 的标准配置是 900 步、八个开发 run 和 24 个测试 run。--raw 额外输出每步实际奖励、动作、效用与最终学习状态，便于重新计算曲线。

报告同时包含开发候选得分、选择结果、测试种子、每个 run 的分段与窗口收益，以及主差值区间。失败示范通过单独的故障注入演示停机处理，明确标为 diagnostic_only_not_in_primary_comparison，不参与正常两方法的性能比较。

<a id="lesson-curves"></a>

## 9 · 曲线、失败与恢复

全程收益包含初始探索、变化后的低谷和重新学习。阶段均值帮助定位变化。最低窗口与原始曲线暴露平均值掩盖的崩溃。窗口宽度及步长应在查看结果前确定，平滑仅用于显示，不能据此改变主统计量。本章程序要求窗口长度整除每个阶段的长度，使窗口等长且不跨越变化边界；不将较短的尾窗与完整窗口混合取最低值。

$$
\tau_{\rm confirm}=\min\{kW:k\geq L,\ \bar R_j\geq c\ \text{for all }j\in\{k-L+1,\ldots,k\}\}.
$$

从变化时刻开始按长度 $W$ 的不重叠窗口计数。连续 $L$ 个窗口达到阈值 $c$ 后，在第 $k$ 个窗口末确认恢复。本实验用 $c=0.7$、$L=2$。这是观测收益的操作性指标，受奖励噪声影响。

下一次变化或观察期结束前未达标，记录 right_censored，并报告截至该时刻的恢复比例。不能只平均成功恢复者的时间。运行中崩溃或资源耗尽则是失败事件，不能自动当作与性能无关的删失。tiny 模式中，反转阶段两方法分别只有 1/8 和 0/8 达到确认条件，这些未恢复 run 全部保留。

若更新恰好在阶段最后一步失败，且此前未确认恢复，仍记为失败，而不是因为奖励数组长度刚好填满而记为删失。确认依据的是更新前已记录的奖励；已发生的首次确认不会被之后的故障抹去，但运行级失败状态始终单独保留。

本 bandit 的奖励非负，并把停机后不再提供服务的收益定义为零。程序分别保存已观测奖励和补齐预算后的效用，记录失败时刻与原因。对任意真实任务，填零不一定合理；若停机有成本或奖励无自然下界，应预先定义失败代价，或提供多个合理代价下的敏感性分析。

单个运行的日志窗口不增加独立重复数。需要整条曲线的区间时，以 run 为单位重采样整条轨迹。大量时刻逐点查看显著性会增加偶然发现；主指标与主要比较在实验前确定，其他曲线承担描述与诊断作用。

<a id="lesson-freeze"></a>

## 10 · 在线评价与冻结诊断

在线主评价问：这个学习过程在真实经历中获得了多少回报？冻结诊断问：从同一个历史起点出发，接下来继续更新是否有帮助？在可复制模拟器中，可以复制学习器与环境状态，再分别保持更新和冻结参数。诊断交互不回灌主生命期。

| 需要隔离的对象 | 正确做法 | 改变它会引入的混杂 |
| --- | --- | --- |
| 网络参数与优化器 | 声明哪些更新被停止 | 冻结权重但继续改变归一化仍有适应 |
| 循环记忆 | 通常允许其随观测递推 | 清空记忆同时删除了历史信息 |
| 环境与随机数流 | 复制环境或使用匹配独立生命期 | 副本评测消耗主 RNG 会改变主轨迹 |
| replay 与统计 | 副本独立维护或明确冻结 | 评测样本进入主 buffer 增加训练数据 |

本章 bandit 没有循环状态和归一化。run_lifetime 的 freeze_at 仅停止价值和计数更新，动作仍按当前价值与相同的探索规则产生。测试比较冻结前缀和独立重建的同一前缀，并检查主运行计数未受影响。这是较简单的冻结语义；深度 agent 需要列出更多持久状态。

冻结后变差，支持该分支时段保持更新有益。冻结后不变差，可能因为策略已足够好、环境未再挑战它，或学习器本来就没有学会。旧任务回访也只是诊断：若真实世界不能重新访问旧情境，完整遗忘矩阵无法仅凭单条在线轨迹识别。

<a id="lesson-branches"></a>

## 11 · 从小实验进入研究

| 研究问题 | 保持不变 | 有针对性的扩展 |
| --- | --- | --- |
| 追踪能力 | 数据权限与总交互数 | 改变反转频率、幅度和奖励噪声 |
| 探索与记忆的分工 | 价值更新规则 | 固定行为流与闭环行为对照 |
| 部分可观测性 | 环境动态和奖励 | 隐去线索、延迟线索，比较记忆模型 |
| 元学习的作用 | 元训练成本及开发权限 | 固定步长、在线自适应、额外计算对照 |
| 深度可塑性 | 容量、梯度预算和探针数据 | aged/fresh、随机替换、只重置优化器 |
| 可用性与调参 | 总搜索预算 | 重复整个选择过程，报告所得独立性能 |

进入多任务 benchmark 前，写出任务权重与归一化规则。一个任务的高分单位不能任意支配总均值。rliable 的矩阵接口按运行×任务组织数据；IQM 去掉上下各 25% 的分数质量，故尾部失败仍需单列。库提供统计实现，实验对象与重采样层级仍由研究问题决定。

一份可复核交付至少包括：问题与指标、环境版本和信息权限、完整算法与依赖、候选与选择规则、所有计划 run 的终态、原始曲线、分析入口、主差值与区间，以及不支持优势的条件。结论应落在这组证据覆盖的范围内。

<a id="lesson-check"></a>

## 12 · 习题与检查

- 把开发前缀从 240 步改为全寿命，需要改变哪句结论？选择器已看到后期变化，估计对象从前缀选择后的持续适应变为完整寿命调参后的性能。
- 同一策略测试 100 个回合，可以称作 100 次独立算法运行吗？它们主要描述同一训练产物的执行噪声，不能替代独立训练。
- 为什么 tiny 区间跨零不能证明两方法等价？区间也包含有实际意义的正差异；等价需要预定容忍范围和相应分析。
- 把八次生命期切成 24 个阶段后做 bootstrap 会怎样？共同参数和历史造成相关，独立样本数被夸大。
- 在测试后给失败方法加大搜索范围，还能沿用原区间吗？新方法已利用原测试信息，需要另外确认。

动手题：保持测试集不变，仅用不同开发 seed 组重复选择，观察所选步长怎样变化。此时这些测试分数可用来探索选择器不稳定性，但新的主张仍需重新设计外层独立重复。另做固定行为随机动作对照，判断常数步长的变化优势来自更快估计，还是来自估计与探索的反馈。

## 学习与研究衔接

从精确测试到受控学习曲线，再到多运行基准。每一层支持不同强度的结论。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-experiments) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=experiments) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=experiments)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

状态是支持后续计算的内部信息；GVF 指定一个预测问题；RTRL 和资格迹规定信用如何传播。三者可以组合，但不是相互替代的算法名称。先理解给定策略的预测，再讨论预测怎样改善控制。

- [Recurrent Reinforcement Learning with Memoroids](https://yingwen.io/zh/continual-rl/research/#recent-memoroids-sequence-learning)
- [Does Zero-Shot Reinforcement Learning Exist?](https://yingwen.io/zh/continual-rl/research/#recent-zero-shot-forward-backward)

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [Recurrent Reinforcement Learning with Memoroids](https://yingwen.io/zh/continual-rl/research/#recent-memoroids-sequence-learning)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Revisiting Adam for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-revisiting-streaming-adam)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [Rethinking the Foundations for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-rethinking-crl-foundations)
- [An Empirical Study of Deep Reinforcement Learning in Continuing Tasks](https://yingwen.io/zh/continual-rl/research/#recent-continuing-task-deep-study)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [Understanding Plasticity in Neural Networks](https://yingwen.io/zh/continual-rl/research/#recent-understanding-plasticity)
- [Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn](https://yingwen.io/zh/continual-rl/research/#recent-c-chain-churn)
- [Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline](https://yingwen.io/zh/continual-rl/research/#recent-reset-and-distill)
- [Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-vla-simple-recipe)
- [Parseval Regularization for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-parseval-continual-geometry)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Discovering state-of-the-art reinforcement learning algorithms](https://yingwen.io/zh/continual-rl/research/#recent-disco-rl)
- [How Should We Meta-Learn Reinforcement Learning Algorithms?](https://yingwen.io/zh/continual-rl/research/#recent-meta-algorithm-search-comparison)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [Position: Lifetime tuning is incompatible with continual reinforcement learning](https://yingwen.io/zh/continual-rl/research/#recent-lifetime-tuning)
- [The Cell Must Go On: Agar.io for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-agarcl)
- [OGBench: Benchmarking Offline Goal-Conditioned RL](https://yingwen.io/zh/continual-rl/research/#recent-ogbench-goal-evaluation)
- [Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-vla-simple-recipe)
- [Revisiting Adam for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-revisiting-streaming-adam)
- [An Empirical Study of Deep Reinforcement Learning in Continuing Tasks](https://yingwen.io/zh/continual-rl/research/#recent-continuing-task-deep-study)
- [How Should We Meta-Learn Reinforcement Learning Algorithms?](https://yingwen.io/zh/continual-rl/research/#recent-meta-algorithm-search-comparison)
- [Recurrent Reinforcement Learning with Memoroids](https://yingwen.io/zh/continual-rl/research/#recent-memoroids-sequence-learning)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文提供定义和条件，架构讲座提出模块组织，算法论文检验特定机制。完整系统还要明确智能体、外部设计者和世界各自承担的工作；组件成立不自动意味着组合后的长期收益成立。

- [Rethinking the Foundations for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-rethinking-crl-foundations)

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

### Understanding Plasticity in Neural Networks

Clare Lyle, Zeyu Zheng, Evgenii Nikishin, Bernardo Avila Pires, Razvan Pascanu, Will Dabney

ICML 2023 · 2023 · 支持方法与理论

#### 研究问题

学习变慢一定意味着网络已饱和或特征秩下降吗？

#### 关键机制

论文通过新目标拟合实验研究可塑性，并分析优化几何与曲率的影响。某些表示统计与学习能力下降会同时出现，却不是所有设置中的充分解释。评价对象从“网络看起来是否健康”转向“在受控更新预算内还能学会什么”。

#### 证据

受控探针与 RL 实验展示了不同机制之间的区别，并检验网络设计和优化过程的作用。它为可塑性研究提供诊断方式，而不是单一通用修复算法。

#### 条件与限制

探针目标、优化器和步数会改变测得的可塑性。相关性不等于所有控制任务中的因果机制；探针训练也不能写回被评价的在线智能体。

#### 阅读与实验

复制同一个检查点，在副本上拟合两类新目标。保持训练预算一致，并报告探针过程与真实环境回报之间的区别。

#### 原文与相关入口

- [ICML 2023 原文](https://proceedings.mlr.press/v202/lyle23b.html)：可塑性探针、优化几何与诊断边界。

### Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn

Hongyao Tang, Johan Obando-Ceron, Pablo Samuel Castro, Aaron Courville, Glen Berseth

ICML 2025 · 2025 · 直接研究持续学习

#### 研究问题

一次局部更新为什么会在其他输入上引发大幅预测变化，并损害后续学习？

#### 关键机制

C-CHAIN 抑制相对于近期参考网络的函数输出变化，降低一次更新在其他样本上造成的 churn。论文把该现象与经验神经切线核及学习动力学联系起来。正则化对象是函数变化，不是直接把所有参数锁在旧值附近。

#### 证据

作者在持续 Gym Control、ProcGen、DMC 和 MinAtar 序列中比较，并提供对应环境和算法代码。

#### 条件与限制

近期函数稳定性不等于长期任务知识保留；参考样本和参考网络也占资源。若环境突然发生真实变化，过强抑制输出变化可能延迟必要适应。

#### 阅读与实验

将 churn 按旧分布、新分布分别计算，并同时画适应速度。这样才能区分“减少无关干扰”和“阻止有用改变”。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/tang25g.html)：机制、理论分析与持续实验。
- [作者代码](https://github.com/bluecontra/C-CHAIN)：四类持续环境的基线和 C-CHAIN 对照实现。

#### 作者代码

[作者仓库，README 说明依赖的 TRAC、CleanRL 与 MinAtar 基础实现。](https://github.com/bluecontra/C-CHAIN)

持续控制环境与 C-CHAIN 对照实验。

### Revisiting Adam for Streaming Reinforcement Learning

Florin Gogianu, Luțu Adrian-Cătălin, Razvan Pascanu

RLC 2026 / RLJ 预会议版 · 2026 · 支持方法与理论

#### 研究问题

流式 RL 的不稳定来自 Adam 本身，还是目标导数、方差与超参数的组合？

#### 关键机制

论文重新分析自适应更新的信噪比，将 Adam 的稳定项与目标导数尺度联系起来，并研究有界导数的回报分布学习及多步更新。它改变的是目标与更新的配合，而非简单沿用批量训练时的默认配置。

#### 证据

作者在大规模 Atari 流式实验中展示了具有竞争力的结果，并重新比较早期流式方法。正式 RLJ 入口收录为 RLC 2026 预会议论文。

#### 条件与限制

主体实验采用经典回合式 Atari 的流式学习协议，不是任意非平稳终生适应的证据。这些结果也不否定归一化、资格迹或更新约束在其他任务中的价值。版本、调参预算和目标分布必须对齐。

#### 阅读与实验

建立二维对照：固定目标换优化器，固定优化器换目标。将调参种子与最终测试分开，再判断改进来自哪一个因素。

#### 原文与相关入口

- [RLC 2026 论文入口](https://rlj.cs.umass.edu/2026/papers/Paper131.html)：会议收录信息与论文。
- [作者预印本](https://arxiv.org/abs/2605.06764)：Adam 尺度分析、回报分布目标与实验协议。

### Discovering state-of-the-art reinforcement learning algorithms

Junhyuk Oh, Gregory Farquhar, Iurii Kemaev, Dan A. Calian, Matteo Hessel, Luisa Zintgraf, Satinder Singh, Hado van Hasselt, David Silver

Nature · 2025 · 支持方法与理论

#### 研究问题

除了学习策略，能否从大量学习过程里学出更有效的 RL 更新规则？

#### 关键机制

DiscoRL 用外层优化评价执行若干内层更新后的行为表现，学习价值、策略与辅助预测之间的更新方式。被训练的对象是学习算法本身，而不仅是某个任务的策略参数。内外两层有各自的数据、时间尺度与计算预算。

#### 证据

论文报告跨环境发现更新规则与迁移到未见环境的结果，并公开配套算法实现。它展示了自动算法发现的可能性，但依赖大规模外层训练。

#### 条件与限制

外层在大量环境和设备上的搜索属于设计者侧资源，不能记作测试智能体单次生命内的自主学习。公开规则的执行成本与发现该规则的成本应分别报告。

#### 阅读与实验

画出内层参数和外层参数的更新依赖，再列出测试时哪些量被冻结。与在线 IDBD 比较时，先区分跨任务算法发现和单流步长追踪。

#### 原文与相关入口

- [Nature 原文](https://www.nature.com/articles/s41586-025-09761-x)：算法发现过程、外层资源与泛化实验。
- [作者实现](https://github.com/google-deepmind/disco_rl)：配套代码与发现的更新规则。

#### 作者代码

[Google DeepMind 的论文配套仓库。](https://github.com/google-deepmind/disco_rl)

DiscoRL 配套实现与学习到的更新规则；具体训练资源以仓库说明为准。

### Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline

Hongjoon Ahn, Jinu Hyeon, Youngmin Oh, Bosun Hwang, Taesup Moon

ICLR 2025 · 2025 · 直接研究持续学习

#### 研究问题

一个网络还能拟合新目标，为什么先前训练仍可能让它在新任务上学得更慢？

#### 关键机制

论文把任务之间的负迁移与一般可塑性损失区分开。Reset & Distill 在新任务开始时重置在线 actor 和 critic，避免旧初始化阻碍学习；随后离线蒸馏当前策略与旧专家的动作分布以整合知识。适应和保留通过不同过程实现。

#### 证据

作者在控制与游戏任务中分析负迁移，并在长 MetaWorld 序列上检验该基线。原文直接提供实现地址。

#### 条件与限制

任务边界、在线网络重置、旧专家和离线蒸馏都需要资源。它不能直接当作无边界、不能重置、禁止回放的单次生命方案。

#### 阅读与实验

除了与连续微调比较，还要与同等预算的从头训练比较。若新任务表现低于从头训练，先检查负迁移，再判断是否属于单纯容量损失。

#### 原文与相关入口

- [ICLR 2025 原文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ba9e3d60610f3525717665966d86e0cd-Abstract-Conference.html)：负迁移诊断、Reset & Distill 机制与边界。
- [原文代码入口](https://github.com/hongjoon0805/Reset-Distill)：论文首页提供的作者实现。

#### 作者代码

[ICLR 正式论文首页明确链接的代码。](https://github.com/hongjoon0805/Reset-Distill)

Reset & Distill 以及任务序列实验。

### Rethinking the Foundations for Continual Reinforcement Learning

Esraa Elelimy, David Szepesvari, Martha White, Michael Bowling

RLC 2025 / RLJ · 2025 · 定义与架构观点

#### 研究问题

如果智能体终生交互且世界不断变化，传统形式化和评价对象遗漏了什么？

#### 关键机制

论文重新审视状态、时间与累计奖励评价中的隐含假设，并讨论以交互历史和偏离遗憾等对象描述持续学习。研究重点从“在固定任务上最终收敛到什么”转向“在持续过程里，什么样的行为比较才有意义”。

#### 证据

这是形式化与研究基础的论证，提出可继续研究的定义和问题；不是一个已经完成全部工程验证的通用智能体。

#### 条件与限制

对常见形式化局限的讨论不意味着 MDP、折扣回报或平均奖励在各自条件下无效。评价框架还需要与具体可计算算法和实验协议连接。

#### 阅读与实验

选择一个有不可逆代价的环境，分别写出最终任务分数、终生在线收益和比较策略集合。检验它们是否会给同一行为排出不同顺序。

#### 原文与相关入口

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_243.pdf)：形式化动机、定义与论证。
- [RLJ 论文入口](https://rlj.cs.umass.edu/2025/papers/Paper243.html)：作者与正式收录信息。

### Position: Lifetime tuning is incompatible with continual reinforcement learning

Golnaz Mesbahi, Parham Mohammad Panahi, Olya Mastikhina, Steven Tang, Martha White, Adam White

ICML 2025 Position Paper · 2025 · 评价与实验协议

#### 研究问题

如果设计者用完整未来生命反复调参，实验还在测智能体面对未知变化的能力吗？

#### 关键机制

论文限制调参可访问的生命阶段，并比较这种选择方式与利用完整生命回报挑选配置的差异。外部设计者掌握未来变化信息，可能使一个并不自适应的固定算法显得适应良好。核心改变发生在评价协议，而不是 TD 更新公式。

#### 证据

作者用持续、非平稳设置中的深度 RL 实验说明超参数选择可以改变方法比较。该工作属于立场论文，论证与示例用于推动更符合问题目标的评价。

#### 条件与限制

允许多少开发阶段经验需要按应用规定，不存在由该论文推出的普适固定比例。仅限制时间前缀也不能替代独立测试种子和计算预算控制。

#### 阅读与实验

对同一配置集合分别按开发前缀和完整生命选择超参数，再在独立测试生命上比较。报告两种选择使用了哪些未来信息。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/mesbahi25a.html)：调参协议、论证与示例实验。

### The Cell Must Go On: Agar.io for Continual Reinforcement Learning

Mohamed A. Mohamed, Kateryna Nekhomiazh, Vedant Vyas, Marcos M. José, Andrew Patterson, Marlos C. Machado

arXiv 预印本 · 2025 · 评价与实验协议

#### 研究问题

如何在持续、动态的高维交互里，同时研究记忆、探索、信用分配与学习能力保持？

#### 关键机制

AgarCL 提供持续运行的游戏环境，并用分解的小任务暴露不同困难。完整环境把这些机制放回同一交互循环，小任务则便于定位失败原因。游戏中的复活事件与把整个世界和智能体都重新开始不是同一种重置。

#### 证据

论文提供环境、基线与可塑性方法比较；部分常见修复在其测试中改善有限。这说明保持可塑性并不能单独代替记忆、探索和长期信用分配。

#### 条件与限制

一个游戏不能代表全部真实持续问题。小任务与完整游戏的协议需要分别阅读；此处仅按可确认的预印本状态收录，不把投稿信息写成会议录用。

#### 阅读与实验

先在一个小任务中验证机制，再检验它在完整环境中的作用是否仍存在。将世界重置、角色复活、参数重置和数据清空分开记录。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2505.18347)：环境设计、分解任务与基线结果。
- [作者环境仓库](https://github.com/machado-research/AgarCL)：环境安装、接口和运行示例；算法基线与环境本体分开。

#### 作者代码

[Machado 研究团队的环境实现。](https://github.com/machado-research/AgarCL)

AgarCL 环境与示例，非所有算法结果的单一训练脚本。

### Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning

Jiaheng Hu, Jay Shim, Chen Tang, Yoonchang Sung, Bo Liu, Peter Stone, Roberto Martín-Martín

RLC 2026 · 2026 · 直接研究持续学习

#### 研究问题

大规模预训练的视觉—语言—动作模型，是否仍需要复杂机制才能顺序学习控制任务？

#### 关键机制

论文研究对预训练 VLA 进行顺序强化学习，并以低秩适配等相对简单的训练流程检验持续学习。预训练表示、可更新参数子空间和 RL 目标共同决定迁移与遗忘，不能只把结果归因于单一保留正则项。

#### 证据

作者在多种 VLA 与长期任务基准上比较，并提供实验代码。RLJ 的 RLC 2026 论文页与论文脚注分别给出正式入口和作者仓库。

#### 条件与限制

预训练数据和算力属于外部资源，任务与重置协议也影响难度。该结果不意味着从零训练的网络不会遗忘，更不意味着任意无边界任务流只需微调。

#### 阅读与实验

固定预训练模型，分别改变可训练参数量和任务顺序。报告预训练资源、每任务在线数据、回放或重置条件，再与传统 CRL 方法比较。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2603.11653)：VLA 持续学习设置、机制与比较。
- [RLC 2026 / RLJ 论文页](https://rlj.cs.umass.edu/2026/papers/Paper84.html)：会议原文入口；PDF 脚注链接作者代码。
- [作者实现](https://github.com/UT-Austin-RobIn/continual-vla-rl)：持续 VLA 的训练与评价代码。

#### 作者代码

[UT Austin RobIn 实验室的原论文仓库。](https://github.com/UT-Austin-RobIn/continual-vla-rl)

预训练 VLA 的顺序 RL 训练和论文评价。

### Recurrent Reinforcement Learning with Memoroids

Steven Morad, Chris Lu, Ryan Kortvelesy, Stephan Liwicki, Jakob Foerster, Amanda Prorok

NeurIPS 2024 · 2024 · 支持方法与理论

#### 研究问题

当记忆网络能够保存信息时，训练序列的切分是否仍会阻止学习器给早期信息分配信用？

#### 关键机制

Memoroids 将一类线性递归模型写成结合运算，利用并行 scan 处理长序列；Tape-Based Batching 将多个完整回合接入同一条 tape，用显式边界处理状态重置，减少分段、补零和截断反传带来的问题。

#### 证据

论文在 POPGym 等部分可观测任务和循环价值学习中比较分段与 tape 训练，并研究观测敏感度、样本效率及运行时间。

#### 条件与限制

并行 scan 和长序列反传使用保存的序列与批处理资源，不属于严格逐步、每条经验只使用一次的 RTRL。结合结构也不使任意非线性 RNN 都能采用同样的 scan。

#### 阅读与实验

固定同一种记忆模型，对照截断长度、完整回合和流式在线导数；分别检查活动能记多久、梯度能传多久、持久内存与训练峰值内存。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://papers.nips.cc/paper_files/paper/2024/file/19f7f755908372efb25826d61959cdf9-Paper-Conference.pdf)：结合运算、inline reset、Tape-Based Batching 与实验。
- [作者公开版本](https://arxiv.org/html/2402.09900v3)：附录给出不同递归模型与回报的 memoroid 写法。

#### 作者代码

[论文附录原链接 memory-monoids 对应作者 Prorok Lab 的现有 memoroids 仓库；README 标明论文。](https://github.com/proroklab/memoroids)

memory 模型、buffer、losses 与 segment_dqn/tape_dqn 对照。

### Parseval Regularization for Continual Reinforcement Learning

Wesley Chung, Lynn Cherif, David Meger, Doina Precup

NeurIPS 2024 · 2024 · 直接研究持续学习

#### 研究问题

仅在初始化时保持良好的权重几何，是否足以让很晚出现的新任务仍容易学习？

#### 关键机制

在选定隐藏层加入 $\lambda\|WW^\top-sI\|_F^2$，持续约束行向量的范数与角度；输出层及额外尺度设计保留表达能力。它维护学习的几何条件，并不直接保存旧任务标签或预测。

#### 证据

作者在 Gridworld、CARL、MetaWorld 任务序列中检验，并拆分范数与角度约束。稳定秩、Jacobian 与熵属于诊断量，不单独构成可塑性或保留的因果证明。

#### 条件与限制

约束会限制函数类；输出行数大于输入维度时，全部行正交不可实现。非线性门控仍能切断梯度。有限任务序列的结果不保证无限生命内有效，也不是无任务信息的万能机制。

#### 阅读与实验

同预算比较仅初始化正交、持续范数约束、持续角度约束和完整正则；同时记录新目标拟合、真实回报、旧功能与额外计算。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/e6df4efa20adf8ef9acb80e94072a429-Paper-Conference.pdf)：目标函数、容量限制、角度/范数消融及持续任务协议。
- [作者版本记录](https://arxiv.org/abs/2412.07224)：正式会议年份为 2024。

#### 作者代码

[仓库明确标为 NeurIPS 2024 官方实现。](https://github.com/wechu/parseval_reg)

PPO、任务序列、正则化与网络结构消融。

### An Empirical Study of Deep Reinforcement Learning in Continuing Tasks

Yi Wan, Dmytro Korenkevych, Zheqing Zhu

arXiv 预印本 · 2025 · 评价与实验协议

#### 研究问题

把环境作为持续的转移过程后，无重置、预设重置和智能体控制重置怎样改变学习难点？

#### 关键机制

构造三类 continuing 协议，将重置后的收益纳入同一条持续过程；对深度控制算法及不同 reward centering 方法进行比较。重置权限属于环境/接口设计，而不是一个可以隐藏的评测便利。

#### 证据

作者公开 MuJoCo 与 Atari testbeds、训练和评价配置。论文报告中心化在多种方法中的收益，同时保留大折扣及无重置恢复困难等限制。

#### 条件与限制

continuing 指非回合式持续交互，不自动意味着环境任意非平稳或无限容量学习。仓库 citation 中的 2024 草稿年与 arXiv 2025 发布年不同；此处按可核验预印本记录，不指定未确认的会议。

#### 阅读与实验

先固定重置转移、成本和时间，再比较目标与算法；分别评价全程学习收益、冻结策略奖励率及失败恢复。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2501.06937)：三类持续协议与深度中心化实验；2025 年 arXiv 首稿。

#### 作者代码

[论文对应 Meta 作者团队的研究仓库，README 明确区分三个 reset 协议。](https://github.com/facebookresearch/DeepRL-continuing-tasks)

testbeds、Pearl 算法、experiments 配置与评测/作图。

### How Should We Meta-Learn Reinforcement Learning Algorithms?

Alexander David Goldie, Zilin Wang, Jaron Cohen, Jakob Foerster, Shimon Whiteson

RLC 2025 / RLJ · 2025 · 评价与实验协议

#### 研究问题

算法表示、发现算法的方法和测试智能体的学习成本，应该如何独立比较？

#### 关键机制

对 RL 流程的不同组件进行算法发现，比较黑盒学习、神经/符号蒸馏与 LLM 代码提案。学习器的表示形式和搜索过程分开定义，才能识别泛化、可解释性和成本之间的取舍。

#### 证据

论文直接比较元训练、元测试、样本成本、训练时间与可解释性，作者代码按发现方法和评价入口组织。

#### 条件与限制

跨环境发现规则主要发生在设计者侧，不等于运行智能体已能终生修改规则。论文的训练任务和预算范围不支持所有算法发现方法的普适排序。

#### 阅读与实验

固定被学习组件、输入权限、元训练数据和发现预算；封存规则后再检验未见环境、长生命与未通知漂移。

#### 原文与相关入口

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_218.pdf)：比较对象、元训练/测试与多维成本。
- [RLJ 论文记录](https://rlj.cs.umass.edu/2025/papers/Paper218.html)：作者与正式会议收录。

#### 作者代码

[论文提供、仓库标为官方的作者实现。](https://github.com/AlexGoldie/learn-rl-algorithms)

learning_algorithms 中各发现方法与独立 evaluation 流程。

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


<a id="chapter-code"></a>

## 下载与运行

标准库非平稳 bandit 的开发选择、独立测试、配对区间及失败诊断；不包含大型神经网络基准。

[下载 experiment_design_lab.py](https://yingwen.io/zh/continual-rl/download/experiment_design_lab.py)

```sh
python experiment_design_lab.py demo --tiny
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Patterson et al. · Empirical Design in Reinforcement Learning](https://jmlr.org/papers/v25/23-0183.html)：JMLR 2024。贯穿性能分布、区间、超参数选择、基线与实验者偏差；适合按研究流程精读。

- [Jordan et al. · Evaluating the Performance of Reinforcement Learning Algorithms](https://proceedings.mlr.press/v119/jordan20a.html)：ICML 2020。完整算法定义、可用性、选择成本与跨任务评价；其研究对象不同于充分调参后的峰值。

- [Jordan et al. 作者评价代码](https://github.com/emmaajordan/EvaluationOfRLAlgs)：原论文 Software 链接指向的作者仓库，包含评价方法及实验实现。

- [Agarwal et al. · Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)：NeurIPS 2021。任务内 bootstrap、IQM 与性能剖面；少量重复的覆盖误差与聚合条件需要一起阅读。

- [rliable · 作者统计工具与 notebook](https://github.com/google-research/rliable)：运行×任务矩阵、指标、区间及原始 benchmark 示例。仓库已归档，复现时固定版本与依赖。

- [Mesbahi et al. · Lifetime tuning is incompatible with continual RL](https://proceedings.mlr.press/v267/mesbahi25a.html)：ICML 2025 position paper。讨论限制寿命调参的动机与实验；前缀比例应按具体部署问题选择。

- [Adam White · Deep RL Course 讲义](https://deeprlcourse.github.io/assets/guests/adam_white.pdf)：从实验目的、运行变异、超参数与模型辅助选择建立问题地图。配合 Patterson 等长文使用。

- [Chan et al. · Measuring the Reliability of RL Algorithms](https://arxiv.org/abs/1912.05663)：ICLR 2020。区分训练过程与固定策略的波动、风险；用于设计均值之外的可靠性指标。

- [Tanner & White · RL-Glue](https://www.jmlr.org/papers/v10/tanner09a.html)：环境、学习器与实验控制的接口分离；思想可用于现代实验框架，不要求采用旧运行时。
