# 任务与优化目标：奖励、回报和持续交互

一个长期运行的智能体应当优化什么？这个选择怎样影响状态、价值函数、学习算法和评价？

## 本章内容

- 定义智能体与环境的边界，区分任务目标、历史摘要和预测知识。
- 从奖励序列推导回报与价值函数，比较回合总奖励、折扣目标和平均奖励。
- 说明随机停止解释和势函数塑形的成立条件，计算条件失效时的反例。
- 在同一资源预算下比较整个学习过程与冻结策略，区分 continuing、continual 和 non-stationary。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 期望

同一策略可能产生多条轨迹。期望按这些轨迹的概率加权，而不是只选成功轨迹。

$$
\mathbb E[X]=\sum_x P(X=x)x
$$

### 条件概率

给定已经看到的历史和选定动作，描述下一条经验的分布。条件中不能包含尚未到达的信息。

$$
P(O_{t+1},R_{t+1}\mid H_t,A_t)
$$

### 几何级数

固定比例衰减的无限和有闭式解。折扣回报、随机停止和周期算例都用到它。

$$
\sum_{k=0}^{\infty}\gamma^k=\frac1{1-\gamma},\qquad0\le\gamma<1
$$

### 马尔可夫状态

若状态保留了预测下一步奖励和状态所需的历史信息，就能支持一步递推。当前图像不一定满足这个条件。

<a id="lesson-setting"></a>

## 1. 任务、信息和方法是三个不同问题

考虑一台长期运送物品的机器人。“提高每小时完成的订单数”规定任务。“记住是否载货”规定决策需要的信息。“使用循环网络和 TD 更新”规定求解方法。这些选择互相影响，但不能互相替代。换一个状态编码器，不会自动改变订单收益；改变失败惩罚，则可能改变最优行为。

智能体与环境的边界是建模边界。它把所研究的决策和学习过程，与提供观测、奖励及动作后果的其余过程分开。边界不必沿机器人的外壳划定。若研究高层导航，电机控制器可以属于环境；若研究电流控制，它就属于智能体。奖励生成逻辑相对这个学习器位于环境一侧，不能由当前策略任意改写评分规则。

$$
H_t=(O_0,A_0,R_1,O_1,\ldots,A_{t-1},R_t,O_t),\qquad A_t\sim\Lambda(\cdot\mid H_t)
$$

$H_t$ 是截至选择动作时的可用历史。$\Lambda$ 描述整个智能体产生动作的规则，包含学习对后续行为的影响。执行 $A_t$ 后，环境才返回 $R_{t+1},O_{t+1}$。这里尚未假设观测就是马尔可夫状态。

| 问题 | 数学对象 | 典型研究内容 |
| --- | --- | --- |
| 哪些结果更好？ | 奖励与轨迹评价 $J$ | 折扣、平均奖励、风险约束、生命期收益。 |
| 过去的哪些信息要保留？ | 历史摘要 $h_t$ | Agent state、部分可观测性、记忆。 |
| 未来会发生什么？ | 条件预测、价值函数和模型 | GVF、预测校准、转移模型。 |
| 怎样改变行为？ | 策略、学习规则和规划过程 | TD、actor–critic、元学习、模型规划。 |

Sutton 与 Barto 的 reward hypothesis 把目标和目的理解为期望累计标量奖励的最大化。这是研究与建模主张，不是无需条件便成立的数学定理。它没有指定奖励的具体数值，也没有声称任意给出的奖励都准确表达设计者的意图。Bowling 等在 ICML 2023 研究了更精确的表述：给定对经验结果的偏好关系，哪些条件使奖励表示成为可能。其结论依赖所列假设，不等于所有现实偏好都满足这些假设。

一个任务定义至少要写出动作、观测、奖励、时间单位、终止或重置规则，以及评价方式。安全约束、允许使用的信息和计算预算也需要明示。“是否完成任务”与“每秒完成量减去能耗”不是同一个目标。

<a id="lesson-derive"></a>

## 2. 从奖励到回报，再到价值函数

奖励 $R_{t+1}$ 是一次转移后收到的标量。回报 $G_t$ 是对未来奖励进行聚合的随机变量。策略 $\pi$ 规定动作分布。价值函数则是给定策略和当前信息后的期望回报。它们分别是信号、聚合规则、行为规则和预测量。

$$
G_t^\gamma=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1},\qquad0\le\gamma<1
$$

若 $|R_t|\le R_{\max}$，则 $|G_t^\gamma|\le R_{\max}/(1-\gamma)$。较远奖励权重更小是目标定义的一部分，并不只是数值优化技巧。

$$
\begin{aligned}G_t^\gamma&=R_{t+1}+\gamma\sum_{k=0}^{\infty}\gamma^kR_{t+k+2}=R_{t+1}+\gamma G_{t+1}^\gamma,\\v_\pi(s)&=\mathbb E_\pi[G_t^\gamma\mid S_t=s],\\q_\pi(s,a)&=\mathbb E_\pi[G_t^\gamma\mid S_t=s,A_t=a].\end{aligned}
$$

第一行只把第一项从级数中取出。后两行引入期望与条件。$q_\pi$ 表示当前执行指定动作后，未来继续使用 $\pi$；它不要求这个动作是 $\pi$ 最常选择的动作。

现在假设 $S_t$ 是充分的马尔可夫状态，环境核 $p(s',r\mid s,a)$ 不随时间改变，策略也固定。对第一步动作与结果分组求期望，就得到 Bellman 方程。若剩余期限影响决策，应把必要时间信息纳入状态，或使用时间索引的价值函数。

$$
\begin{aligned}v_\pi(s)&=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)[r+\gamma v_\pi(s')],\\q_\pi(s,a)&=\sum_{s',r}p(s',r\mid s,a)\left[r+\gamma\sum_{a'}\pi(a'\mid s')q_\pi(s',a')\right].\end{aligned}
$$

有限集合对应求和，连续量对应积分。方程描述价值应满足的关系；TD 和动态规划则是求解它的方法。

$$
J_\gamma(\pi;d_0)=\mathbb E_{S_0\sim d_0}[v_\pi(S_0)],\qquad\pi^*\in\operatorname*{arg\,max}_{\pi\in\Pi}J_\gamma(\pi;d_0)
$$

初始分布 $d_0$ 和可用策略集合 $\Pi$ 也是问题定义的一部分。真实值 $v_\pi$ 与网络估计 $v_w$ 不同。降低预测误差是在改进估计，不直接等于提高策略收益。

固定策略问题是一个基础对象。在线学习时，未来行为还受尚未发生的更新影响。评价整个学习器应把学习状态纳入过程。当前冻结策略的价值，不能完整描述这个学习器今后的实际收益。

<a id="lesson-boundaries"></a>

## 3. 回合、持续任务和采样截断

Episodic 任务给出真实结束时刻 $T$，例如一局棋的胜负终局。未折扣回报是到终止为止的奖励和。奖励有界且期望 episode 长度有限，是保证期望回报有限的一组充分条件。不能仅凭“最终会结束”就默认所有期望存在。

$$
G_t=\sum_{k=t}^{T-1}R_{k+1},\qquad v_\pi(S_T)=0
$$

终止后没有待计入的任务奖励，因此尾值为零。折扣版 episodic 目标还给各项乘 $\gamma^{k-t}$。终止规则与折扣系数是两个独立选择。

Continuing 任务没有自然的最终结束。机器人充电、服务器等待请求、失败后恢复，都可能是持续过程中的普通转移。是否重置位置、是否清空记忆、重置需要多少时间和代价，要分别规定。把这些过程删掉会改变所评价的系统。

采样截断只表示记录或计算到此为止。例如保存 $n$ 步 rollout，但任务还会继续。此时后续收益没有消失。对固定策略，用真实价值补齐尾部有下面的恒等式；实际算法使用估计值，所以还会有估计误差。

$$
v_\pi(s)=\mathbb E_\pi\!\left[\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^n v_\pi(S_{t+n})\mid S_t=s\right]
$$

真正终止时尾值为零；单纯 timeout 通常保留尾值。若时间上限本来就是任务定义，应把剩余时间纳入状态，并在期限处使用终止边界。

每步奖励恒为 1、$\gamma=0.9$ 时，持续价值为 10。只记录一步，应有 $1+0.9\times10=10$。把采样边界错当终止，只得到 1。差值来自目标边界错误，与网络容量无关。

<a id="lesson-stopping"></a>

## 4. 折扣何时等价于随机停止

先假想一条完整的潜在奖励轨迹，再独立抽取停止时间 $N\ge1$。第一条奖励一定计入。之后每收完一条奖励，以概率 $\gamma$ 继续，因此 $P(N>k)=\gamma^k$。这个独立性是推导的关键。

$$
\begin{aligned}\mathbb E\!\left[\sum_{k=0}^{N-1}R_{t+k+1}\right]&=\sum_{k=0}^{\infty}\mathbb E[\mathbf1\{N>k\}R_{t+k+1}]\\&=\sum_{k=0}^{\infty}P(N>k)\mathbb E[R_{t+k+1}]\\&=\mathbb E[G_t^\gamma].\end{aligned}
$$

有界奖励和 $\gamma<1$ 保证可以交换和与期望。第二行使用停止事件与潜在奖励轨迹独立。若失败概率依赖动作或状态，通常需要状态相关的 continuation，而非统一常数 $\gamma$。

$$
\mathbb E[N]=\sum_{k=0}^{\infty}P(N>k)=\frac1{1-\gamma}
$$

这是有效时间尺度，不是硬截断。$\gamma=0.99$ 时平均 100 步，不表示第 101 步之后的奖励权重为零。还要说明一步对应多少真实时间。

潜在奖励为 $(1,2,3)$，之后为零，$\gamma=0.5$。第一步后停止的概率为 0.5，累计奖励为 1；第二步后停止的概率为 0.25，累计奖励为 3；至少运行三步的概率为 0.25，累计奖励为 6。期望为 $0.5\times1+0.25\times3+0.25\times6=2.75$，等于折扣和 $1+0.5\times2+0.25\times3$。

独立性失败的例子：第一步奖励为零。公平硬币同时决定是否继续，以及下一条潜在奖励是 2 还是 0；只有奖励为 2 时才继续。实际第二步贡献的期望为 1。若用继续概率 0.5 乘无条件奖励均值 1，则错误地得到 0.5。动作相关的失败风险不能直接由全局折扣替代。

枚举停止位置，独立计算期望；不调用折扣求和函数。

```python
def stopped_return_exact(rewards, gamma):
    """Expected undiscounted sum with an independent geometric stopping time.

    Collect the first reward for sure, then survive after each reward with
    probability gamma. Rewards beyond the supplied list are zero. The last
    outcome combines all stopping times at or beyond the end of the list.
    """
    discount(gamma)
    rewards = finite(rewards, 'rewards')
    expectation = 0.0
    partial_sum = 0.0
    for index, reward in enumerate(rewards):
        partial_sum += reward
        survival = gamma ** index
        probability = survival if index == len(rewards) - 1 else (1 - gamma) * survival
        expectation += probability * partial_sum
    return expectation
```

<a id="lesson-average"></a>

## 5. 平均奖励与差分价值

若任务关注长期单位时间产出，可以使用平均奖励。它把有限长度的累计收益除以经过的时间，再取极限。它不是未折扣的无限奖励和，也不是在折扣回报公式中直接令折扣等于一。

$$
g_\pi(s)=\lim_{T\to\infty}\frac1T\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}R_{t+1}\mid S_0=s\right]
$$

极限需要存在。有限单常返类等条件可使长期奖励率与初始状态无关，这时简写为 $g_\pi$。多常返类可能具有不同奖励率。每步时长不等时，还需用总真实时长作分母。

差分价值 $h_\pi$ 描述相对长期平均水平的瞬态优势。对固定策略，把增长项 $Tg_\pi$ 从有限时域价值中分离，可推导 Poisson 方程。在有限不可约且非周期链中，适当常数规范下有下面的渐近展开。周期链不一定有这个逐点展开，但可以直接求最后的方程。

$$
\begin{aligned}V_T(s)&=\mathbb E_\pi[R_{t+1}+V_{T-1}(S_{t+1})\mid S_t=s],\\V_T(s)&=Tg_\pi+h_\pi(s)+o(1),\\g_\pi+h_\pi(s)&=\mathbb E_\pi[R_{t+1}+h_\pi(S_{t+1})\mid S_t=s].\end{aligned}
$$

最后一式只包含 $h$ 的差值。给所有状态的 $h$ 加同一常数，不改变方程。可以约定参考状态的差分价值为零。平均奖励算法章进一步推导 Differential TD/Q 与 RVI 如何估计这些量。

$$
\lim_{\gamma\uparrow1}(1-\gamma)v_{\pi,\gamma}(s)=g_\pi(s)
$$

对有限 MDP 中的固定平稳策略，此极限联系了折扣价值与平均奖励。它不是说任意接近一的折扣都会给出相同策略排名，也不能据此任意交换极限、策略优化和函数逼近。

一个策略可能支付很大的启动代价后获得更高稳态收益。平均奖励会忽略有限启动代价，但有限预算任务可能无法承担它。因此目标函数与瞬态指标都需要明示。最终奖励率不能概括整段学习过程。

<a id="lesson-example"></a>

## 6. 同一环境中的策略偏好反转

环境有两条路线。初始时选择 A 或 B，随后只能沿所选路线的环行走。A 每一步得到 1。B 的周期奖励为 0、0、4，并从第一条零奖励开始。奖励与转移均不随时间变化。两条路线没有额外未计入的入场步。此例比较对应的两条确定性策略。

$$
v_A(\gamma)=\frac1{1-\gamma},\qquad v_B(\gamma)=\frac{4\gamma^2}{1-\gamma^3},\qquad g_A=1,\quad g_B=\frac43
$$

B 在第 3、6、9 等步发放奖励，所以折扣和为周期首项乘周期比例的几何级数。平均奖励则直接用周期收益除以周期长度。

$$
\begin{aligned}v_B>v_A&\iff4\gamma^2>1+\gamma+\gamma^2\\&\iff3\gamma^2-\gamma-1>0\\&\iff\gamma>\frac{1+\sqrt{13}}6\approx0.76759.\end{aligned}
$$

两个分母都为正，所以可以相乘比较。平均奖励总是偏好 B，低折扣却可能偏好 A；差别来自评价标准，不是学习算法出错。

| 评价规则 | A | B | 更优路线 |
| --- | --- | --- | --- |
| 折扣 $\gamma=0.5$ | 2 | 1.142857 | A |
| 折扣 $\gamma=0.9$ | 10 | 11.955720 | B |
| 每步长期平均 | 1 | 1.333333 | B |
| 从规定相位开始的前 5 步总奖励 | 5 | 4 | A |

B 的奖励率为 $4/3$。令第一状态的差分价值为零，Poisson 方程给出 $h=(0,4/3,8/3)$。第一式为 $4/3+0=0+4/3$，最后一式为 $4/3+8/3=4+0$。较大差分值表示更接近下一笔收益。周期性不妨碍方程有解，但中心化奖励的有限部分和会振荡。

有限回报后向计算、周期折扣闭式解与平均奖励 Poisson 方程；终止和截断尾值由调用方明确传入。

```python
def discounted_return(rewards, gamma, bootstrap=0.0):
    """Sum gamma**t * rewards[t], plus gamma**T * bootstrap.

    bootstrap=0 describes a true terminal boundary. For a truncated continuing
    trajectory, the caller can supply its continuation value instead.
    """
    discount(gamma)
    rewards = finite(rewards, 'rewards')
    if not math.isfinite(bootstrap):
        raise ValueError('bootstrap must be finite')
    value = float(bootstrap)
    for reward in reversed(rewards):
        value = reward + gamma * value
    return value


def periodic_value(cycle, gamma, phase=0):
    """Exact value of an infinite deterministic reward cycle at a given phase."""
    discount(gamma, continuing=True)
    cycle = finite(cycle, 'cycle')
    if not cycle:
        raise ValueError('cycle must be non-empty')
    if not isinstance(phase, int):
        raise ValueError('phase must be an integer')
    rewards = cycle[phase % len(cycle):] + cycle[:phase % len(cycle)]
    return discounted_return(rewards, gamma) / (1 - gamma ** len(cycle))


def periodic_average_and_bias(cycle):
    """Solve g+h(i)=r(i)+h(i+1) with h(0)=0, including periodic chains."""
    cycle = finite(cycle, 'cycle')
    if not cycle:
        raise ValueError('cycle must be non-empty')
    rate = sum(cycle) / len(cycle)
    bias = [0.0]
    for reward in cycle[:-1]:
        bias.append(bias[-1] + rate - reward)
    return rate, bias


def prefix_rewards(cycle, steps):
    cycle = finite(cycle, 'cycle')
    if not cycle or not isinstance(steps, int) or steps < 0:
        raise ValueError('non-empty cycle and nonnegative integer steps required')
    return [cycle[t % len(cycle)] for t in range(steps)]
```

<a id="lesson-shaping"></a>

## 7. 奖励塑形的边界条件

辅助奖励可以让反馈更密集，但一般会改变任务。势函数塑形给出一种有条件的保序方式。选择固定势函数 $\Phi$，在每次转移上加 $F(s,s')=\gamma\Phi(s')-\Phi(s)$。其中 $\gamma$ 必须与回报的折扣一致。

$$
\begin{aligned}R'_{t+1}&=R_{t+1}+\gamma\Phi(S_{t+1})-\Phi(S_t),\\G'_{0:T}&=\sum_{t=0}^{T-1}\gamma^tR'_{t+1}\\&=G_{0:T}+\sum_{t=0}^{T-1}[\gamma^{t+1}\Phi(S_{t+1})-\gamma^t\Phi(S_t)]\\&=G_{0:T}-\Phi(S_0)+\gamma^T\Phi(S_T).\end{aligned}
$$

相邻势函数项逐项抵消，只留下端点。是否保留策略排序，取决于剩余边界项，而不是奖励是否看起来有帮助。

无限持续折扣任务中，若 $\Phi$ 有界且 $\gamma<1$，尾项趋于零。固定起点后，每条策略的价值都减去相同的 $\Phi(s)$。有限 episode 中，一个充分条件是所有真实终止状态的势为零。这样无论终止时间如何变化，都只有起点偏移。

$$
v'_\pi(s)=v_\pi(s)-\Phi(s),\qquad q'_\pi(s,a)=q_\pi(s,a)-\Phi(s)
$$

等式使用上述边界条件及相同策略、转移和折扣。它保留原问题的策略比较，不保证任意函数逼近算法具有相同的学习轨迹或收敛速度。

原奖励为 $(0,2)$，沿途势为 $(3,1,0)$，折扣为 0.9。塑形奖励为 $(-2.1,1)$，回报为 $-2.1+0.9=-1.2$。原回报为 1.8，两者恰好相差起点势 3。奖励可以变负，但排序仍可保持。

反例：两种动作均一步终止，A 的原奖励为 1，B 为 0。起点势为 0，A 终点势为 0，B 终点势为 2，$\gamma=0.9$。塑形后 A 得 1，B 得 1.8，排序颠倒。即使终点势都是同一个非零常数，不同终止长度仍可能通过 $\gamma^T$ 产生不同偏移。

采用吸收状态的无限延伸时，必须计入吸收后的塑形奖励；不能使用无限和的结论，却在代码中删掉这些项。若只是 rollout 截断，应保留相应的塑形尾值。零终点、非零终点和截断尾值在配套测试中分别处理。

$$
\frac1T\sum_{t=0}^{T-1}(R'_{t+1}-R_{t+1})=\frac{\Phi(S_T)-\Phi(S_0)}T\longrightarrow0
$$

平均奖励中取 $F=\Phi(s')-\Phi(s)$。若势有界，端点差除以时间后消失，长期奖励率不变。有限生命收益仍有端点差；这与折扣塑形是不同的评价推导。

常数平移也要检查边界。无限持续折扣任务中，$r'=ar+b$ 且 $a>0$ 给出 $v'=av+b/(1-\gamma)$，因此保留排序。长度可变的未折扣 episode 却增加 $bT$，可能改变终止偏好。奖励裁剪、时间变化的探索 bonus、任意“靠近目标就加分”，也不自动满足势函数条件。

保留全部端点势，比较逐步回报与望远镜恒等式。

```python
def shape_rewards(rewards, potentials, gamma):
    """F_t=gamma*Phi(s[t+1])-Phi(s[t]); retain the final potential explicitly."""
    discount(gamma)
    rewards = finite(rewards, 'rewards')
    potentials = finite(potentials, 'potentials')
    if len(potentials) != len(rewards) + 1:
        raise ValueError('one potential per state, including both endpoints, required')
    return [reward + gamma * potentials[t + 1] - potentials[t]
            for t, reward in enumerate(rewards)]


def shaping_comparison(rewards, potentials, gamma):
    shaped = shape_rewards(rewards, potentials, gamma)
    original = discounted_return(rewards, gamma)
    boundary = -potentials[0] + gamma ** len(rewards) * potentials[-1]
    return dict(original=original, shaped=discounted_return(shaped, gamma),
                boundary=boundary, expected=original + boundary)
```

<a id="lesson-subtasks"></a>

## 8. 总任务、目标条件子任务和预测知识

“目标”有两种常见用法。总任务目标规定什么结果对整个智能体更好。目标条件策略的 goal 则是输入，例如“到充电站”或“使某个特征变大”。后者可以帮助求解总任务，但不自动等于总任务。

$$
\pi(a\mid s,z),\qquad r_z(s,a,s'),\qquad J_{\rm ext}(\Lambda)
$$

$z$ 是子任务描述，$r_z$ 是训练它所用的奖励，$J_{\rm ext}$ 是外部评价。运货过程中选择“去充电”只是中间决策，不必改变整个系统的订单收益目标。

将“到达充电站”训练得可靠，不保证订单产出提高。高层选择器还需判断何时执行，以及耗时和后果。子任务选择、option 终止和模型学习因此各有问题。外部评价应计入训练与执行子任务的交互，而不只统计技能成功率。

预测问题又不同。GVF 用 cumulant $c$、continuation $\gamma$ 和策略 $\pi$ 定义待预测的经验量，例如“持续前进会遇到多少碰撞”。预测学习要求把该量估计准确；控制器是否应避免碰撞，要由任务奖励和约束决定。$c$ 不必是外部奖励。

预测知识可以供控制、规划和状态构造使用，但不必全部成为状态坐标。状态也不必全部由可解释预测构成。总任务、内部子任务和知识预测分别定义后，才能分析它们是帮助、冲突还是增加了无用计算。Alberta Plan 的基本智能体也区分主要策略、其他策略、价值函数和转移模型；它们通过构造的状态交换信息。

<a id="lesson-lifetime"></a>

## 9. 整个学习过程与冻结策略

固定策略评价分析一个明确的行为规则。持续学习还需要评价完整智能体，包括初始化、探索、参数更新、记忆管理和规划。它在变好之前已经消耗时间，也已经产生收益或损失。

$$
J_T(\Lambda)=\mathbb E_\Lambda\!\left[\sum_{t=0}^{T-1}R_{t+1}\right],\qquad\overline J_T(\Lambda)=J_T(\Lambda)/T
$$

所有学习期动作都在累计和中。对固定 $T$，累计收益与平均收益排序相同。期望涵盖环境、探索和学习器的随机性。

$$
J_{\rm frozen}(\Lambda,T)=\mathbb E[J(\pi_{w_T};d_{\rm eval})]
$$

这里在训练至 $T$ 后冻结参数，再从指定分布 $d_{\rm eval}$ 评价策略。外层期望来自训练随机性。这是有用的诊断，但通常不同于 $J_T$；额外评价环境的交互量也应单列。

在平稳确定性三臂 bandit 中，三个动作分别给 0、0.6 和 1。过程 A 十步里一直选第二臂，在线收益为 6，冻结末次动作后的每步收益为 0.6。过程 B 前八步选第一臂，最后两步选第三臂，在线收益只有 2，但冻结后收益为 1。这里是两个预设行为过程的评价反例，不是学习算法的性能比较。

相同环境中的在线收益和冻结末次动作收益，可以产生相反排序。

```python
def evaluate_schedule(arm_rewards, action_schedule):
    """Compare online reward with the value of freezing the final bandit action.

    This is an evaluator of prescribed action schedules, not a bandit learner.
    The stationary deterministic environment returns arm_rewards[action].
    """
    arm_rewards = finite(arm_rewards, 'arm_rewards')
    actions = tuple(action_schedule)
    if not arm_rewards or not actions:
        raise ValueError('non-empty arms and action schedule required')
    if any(not isinstance(a, int) or not 0 <= a < len(arm_rewards) for a in actions):
        raise ValueError('each action must index an arm')
    rewards = [arm_rewards[a] for a in actions]
    cumulative = sum(rewards)
    return dict(cumulative=cumulative, online_mean=cumulative / len(rewards),
                frozen_final_mean=arm_rewards[actions[-1]])
```

资源限制决定哪些智能体可以公平比较。需要规定持久内存、每个交互步的计算量、数据保存范围、重置权限和预训练预算。若规划会延迟真实行动，墙钟时间也进入任务。若统一用环境步计分，至少要另报算力和延迟。

Alberta Plan 强调长期交互、有限计算和时间一致性。学习与规划是运行过程的一部分，而非奖励不计分的特殊准备阶段。有限生命评价体现这种关注，但不是该计划唯一指定的评分公式。不同应用仍需说明自己的时间聚合方式。

| 术语 | 含义 | 不能据此推出的结论 |
| --- | --- | --- |
| Continuing task | 没有自然最终终止。 | 不强制平均奖励，也不要求非平稳环境。 |
| Stationary environment | 给定充分状态和动作后，动力学与奖励核不随时刻改变。 | 不表示观测不变化、数据独立或智能体已学会。 |
| 持续更新的学习器 | 参数、知识或其他学习状态在运行中适应经验。 | 非零更新不证明所有最优智能体都必须永久学习。 |
| Abel 等的 CRL 定义 | 相对于给定 agent basis，最优智能体持续隐含搜索，而不最终停在其中一个基础智能体。 | 不单凭任务切换次数或非平稳标签来分类。 |

未知但平稳的 bandit 仍需要学习。不过它有最优固定策略，因此不自动成为“最优智能体必须永久学习”的例子。长期世界即使有固定的底层规律，也可能不断给有限资源智能体带来未掌握的情况。是否需要持续适应，要一起分析信息、资源和智能体集合，不能只检查转移函数有没有时间下标。

<a id="lesson-code"></a>

## 10. 运行与检查

下载 objectives_lab.py 后运行；Python 3.10+，仅标准库。

```sh
python3 objectives_lab.py all
python3 objectives_lab.py preferences
python3 objectives_lab.py stopping
python3 objectives_lab.py shaping
python3 objectives_lab.py lifetime
python3 objectives_lab.py test
```

preferences 输出折扣价值、奖励率和前五步收益。stopping 独立计算停止位置期望，并展示误把截断当终止的差异。shaping 输出原回报、塑形回报和解析边界项。lifetime 比较两个评价方式的排名。每个实验都有明确输入和确定性结果。

**算法：任务与评价的计算流程，不是某个控制算法的训练步骤**

1. 给定环境、允许信息、资源上限和候选行为过程。
1. 选择回合、折扣、平均奖励或有限生命目标。
1. 分别写出真实终止、重置和采样截断规则。
1. 在同一初始条件下生成奖励序列，保留全部学习期奖励。
1. 若采用奖励塑形：
  1. 固定势函数，匹配折扣，检查起点与终点项。
  1. 分别记录原始奖励与训练奖励。
1. 报告在线收益，以及需要的冻结策略诊断。
1. 改变目标时，重新计算策略排序。

解析实验不依赖采样估计，因此数值一致性只检验定义和实现。进入随机环境后，还要报告独立运行数、不确定性和失败轨迹。单次真实生命中，则需说明哪些统计只从日志离线计算，哪些干预不允许执行。

<a id="lesson-branches"></a>

## 11. 后续研究问题

| 问题 | 要学习的机制 | 与本章的连接 |
| --- | --- | --- |
| 未知状态的长期价值如何估计？ | TD、资格迹、off-policy 预测。 | 固定策略、奖励、折扣与边界。 |
| 没有自然终点时如何提高收益？ | Differential TD/Q、RVI、平均奖励控制。 | 明确奖励率、链结构和时间单位。 |
| 当前观测不足以决策？ | 信念、RNN、预测状态、在线递归梯度。 | 保持目标，比较历史摘要丢失的信息。 |
| 哪些知识值得长期保存？ | GVF、问题发现、预测表示。 | 明确预测对象及其对外部目标的用途。 |
| 怎样分解长程行为？ | 目标条件策略、option、子任务发现。 | 子任务奖励与总评价分开，计入时长。 |
| 学习很久后如何继续适应？ | 步长适应、可塑性、元学习、资源管理。 | 评价完整学习过程，不只测最后参数。 |
| 应当计划哪些未来？ | 模型学习、规划和搜索控制。 | 按决策影响分析模型误差，计入规划成本。 |

清楚的研究可以只改变一个选择。保持环境和策略类不变，比较不同折扣与平均奖励；或者保持外部目标不变，比较有明确边界条件的奖励塑形。用可解析环境确定预期策略后，再进入函数逼近实验，才能区分任务变化、估计误差和学习机制的影响。

<a id="lesson-check"></a>

## 12. 习题与诊断

- 把 B 改成 (0,0,5)，重新求折扣阈值。答案：4γ²−γ−1>0，阈值为 (1+√17)/8；前五步 A 与 B 都得 5。
- B=(0,0,4)，但从发放 4 的相位开始，平均奖励是否变？答案：不变；折扣价值变为 4/(1−γ³)。
- 终点势都为 10，γ=0.9，A 一步奖励 1，B 两步奖励 (0,1.5)，是否保序？答案：原回报 1 与 1.35，塑形后 10 与 9.45，排序反转。
- 碰撞 GVF 预测准确，是否定义了避免碰撞的任务？答案：没有；预测内容与控制评价中的碰撞代价不同。
- 平稳 MDP 运行百万步且一直更新网络，是否满足 Abel 等的 CRL 定义？答案：不能据此判断；还需指定 agent basis，并分析最优智能体是否能停止搜索。

| 现象 | 优先检查 |
| --- | --- |
| 训练损失降低但收益不升 | 是否把预测目标、内部奖励或辅助损失当成外部评价。 |
| 增大折扣后偏好不同动作 | 目标本来可能变化；先解小模型检查策略排序。 |
| timeout 附近价值突降 | 是否把采样截断写成真实终止。 |
| 塑形后循环刷奖励 | 是否满足势差形式，终点和折扣是否匹配。 |
| 最终策略好，长期运行却损失大 | 是否遗漏学习初期、探索、恢复和重置成本。 |
| 收益接近但计算量差异很大 | 交互与计算预算是否一致，延迟是否影响真实动作。 |

## 本章的实验设计

先明确比较冻结策略，还是比较持续更新的学习器。两者具有不同的估计对象。

设定：同一可控任务分别评价完整在线生命期和最终冻结策略。保留原始奖励、动作时间与所有重置事件。

- 两种评价的收益口径和数据来源可以分别重算。
- 冻结副本不会改变主训练的参数、统计、记忆与 RNG。
- 奖励变换若改变优化目标，明确标注而非沿用原目标名称。

对照：相同初始化的在线更新与冻结参数；相同协议下的未变换奖励基线；部署允许与禁止重置的分别报告

记录：完整生命期累计奖励或每原始时间奖励率；冻结策略条件表现；重置、评价及在线更新成本

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-question)

## 学习与研究衔接

折扣与平均奖励是目标选择。持续学习不能仅靠把回合接长来定义。

[分册导读](https://yingwen.io/zh/continual-rl/start/classic-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-objectives) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=objectives) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=objectives)

<a id="chapter-code"></a>

## 下载与运行

标准库解析实验：周期策略偏好、随机停止、终止与截断、势函数塑形、在线与冻结评价。

[下载 objectives_lab.py](https://yingwen.io/zh/continual-rl/download/objectives_lab.py)

```sh
python3 objectives_lab.py all
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 2nd ed.](https://incompleteideas.net/book/the-book-2nd.html)：第 3 章的智能体边界、奖励、回报与价值；第 10 章的持续任务平均奖励。

- [MIT Press · Reinforcement Learning, second edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)：正式第二版出版信息与作者入口。

- [Bowling, Martin, Abel & Dabney · Settling the Reward Hypothesis · ICML 2023](https://proceedings.mlr.press/v202/bowling23a.html)：将经验偏好可由奖励表示的条件形式化，而非无条件肯定所有目标都能由任意标量奖励表达。

- [Sutton, Bowling & Pilarski · The Alberta Plan for AI Research](https://arxiv.org/html/2208.11173v3)：长期交互、时间一致性、有限计算与基本智能体；状态、主要策略、其他策略、价值函数和模型有不同作用。

- [Abel 等 · A Definition of Continual Reinforcement Learning · NeurIPS 2023](https://arxiv.org/html/2307.11046v2)：通过 agent basis、生成和到达关系定义持续学习，而非仅凭非平稳性分类。

- [Ng, Harada & Russell · Policy Invariance under Reward Transformations · ICML 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)：固定势函数塑形与策略不变性；有限轨迹推导需要保留边界项。

- [Wan, Naik & Sutton · Learning and Planning in Average-Reward MDPs · ICML 2021](https://proceedings.mlr.press/v139/wan21a.html)：从奖励率与差分价值进入可执行的学习与规划算法。

- [平均奖励作者代码 · average-reward-methods](https://github.com/abhisheknaik96/average-reward-methods)：原论文预测、控制和实验配置；周期计算用于理解目标，后续平均奖励章解释更新算法。
