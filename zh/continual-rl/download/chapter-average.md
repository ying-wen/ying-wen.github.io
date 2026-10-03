# 平均奖励：奖励率、差分价值与持续控制

没有自然终点时，怎样同时学习每单位时间的收益、状态的相对价值，以及最优行为？

## 本章内容

- 从时间平均目标推导 Poisson / Bellman 方程，解释为何值函数只确定到常数。
- 独立实现 Differential TD、Differential Q 与已知模型的 RVI，明确每条更新使用哪个旧值。
- 把原子动作推广到随机时长的 option，分清目标、数据协议与深度实现假设。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 状态与马尔可夫性

给定当前状态和动作，下一步奖励与状态的条件分布不再依赖更早历史；观测不充分时，需要先构造带记忆的 agent state。

### 策略评估与控制

策略评估求固定策略 $\pi$ 的价值；控制则寻找收益更高的策略。控制样本可由探索策略 $b$ 产生，因此样本奖励均值未必是最优策略的奖励率。

### TD 与半梯度

用当前估计的下一状态价值补齐未知未来，把它视作本步固定目标再更新当前估计；这不是对整个 Bellman residual 求全梯度。

<a id="lesson-setting"></a>

## 1 · 每单位时间的收益

考虑一个长期运行的排队服务器。每一步可以接收或拒绝请求，接收高价值请求有收益，但占用容量可能妨碍未来工作。不存在一个自然的“游戏结束”时刻。我们要比较单位真实时间的收益，而不是人为截断后的一回合总分。为先把目标讲清，本页从有限、平稳的 MDP 开始，假定所评估策略有唯一长期奖励率；学习期间环境改变的问题放在诊断一节单独讨论。

$$
g_\pi=\lim_{T\to\infty}\frac1T\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}R_{t+1}\right]
$$

$g_\pi$ 是每个原始环境步的平均收益。本章假设它与初始状态无关；例如有限不可约策略链满足这一点。多个互不连通的常返类可能具有不同奖励率，因而需要更一般的、依赖状态的表述。

未折扣的无限奖励和通常发散；平均奖励目标使用时间平均，不能仅由折扣回报中代入 $\gamma=1$ 得到。与此同时，没有自然终点的环境仍可采用折扣目标。目标函数、是否重放数据、是否允许外部重置，是三个需要分别指定的选择。

| 量 | 含义 | 在算法中何时改变 |
| --- | --- | --- |
| $g_\pi$ / $g_*$ | 固定策略 / 最优策略的真实奖励率 | 问题的性质，未知 |
| $\bar g$ | 奖励率估计 | 每条实际更新后改变 |
| $h(s)$ / $q(s,a)$ | 差分价值，又称 bias | 根据 TD 误差更新 |
| $\alpha$、$\eta$ | 价值步长、奖励率相对步长 | 本页固定；非平稳场景可研究自适应 |

奖励率衡量长期稳态表现，相对价值衡量从当前状态出发，在进入稳态之前比平均水平多赚或少赚多少。两者都需要：两条策略可能长期奖励率相同，但在前几百步的收益相差很大。对有限预算研究还要报告累计收益、变化后恢复时间，不能只报告最终 g。

<a id="lesson-derive"></a>

## 2 · 从时间平均到 Differential TD / Q

固定策略的有限时域价值记为 $V_T(s)$。先将第一步与剩下 $T-1$ 步分开。在有限、不可约且非周期的链上，采用适当的常数规范，可得到 $V_T(s)=Tg_\pi+h_\pi(s)+o(1)$。代入递推并消去线性增长项，便得到下面的方程。周期链不一定有这个逐点渐近形式，但仍可以直接研究相应的 Poisson 方程。

$$
\begin{aligned}V_T(s)&=\mathbb E_\pi[R_{t+1}+V_{T-1}(S_{t+1})\mid S_t=s],\\g_\pi+h_\pi(s)&=\mathbb E_\pi[R_{t+1}+h_\pi(S_{t+1})\mid S_t=s].\end{aligned}
$$

这是 Poisson 方程的状态形式。令 $P_\pi$ 为策略转移矩阵、$r_\pi$ 为期望即时奖励向量，矩阵形式为 $(I-P_\pi)h=r_\pi-g_\pi\mathbf1$。由于 $(I-P_\pi)\mathbf1=0$，$h$ 与 $h+c\mathbf1$ 满足同一方程。

差分价值 $h$ 的绝对数值依赖常数规范。可以指定某个参考状态的价值为零，也可以允许算法得到由初始化决定的偏移；跨运行比较关注差值、策略与 Bellman 残差。Differential TD 的基本版本不固定某个参考状态。

$$
\begin{aligned}\delta_t&=R_{t+1}-\bar g_t+h_t(S_{t+1})-h_t(S_t),\\h_{t+1}(S_t)&=h_t(S_t)+\alpha_t\rho_t\delta_t,\\\bar g_{t+1}&=\bar g_t+\eta\alpha_t\rho_t\delta_t,\qquad \rho_t=\frac{\pi(A_t\mid S_t)}{b(A_t\mid S_t)}.\end{aligned}
$$

On-policy 时 $\rho=1$。Off-policy 预测要求 $\pi$ 可能选择的动作在 $b$ 下也有正概率。两个更新共用旧参数下的 TD 误差，即计算一次 $\delta_t$ 后同时用于价值与奖励率。

奖励率也由 TD 误差更新，是因为在正确解处，对目标策略求条件期望后，奖励、奖励率与差分价值变化相抵消。只平均真实奖励则估计行为策略 $b$ 的奖励率；当 $\pi\ne b$ 时，它可能与价值头的目标不同。重要性比修正动作分布，但不提供任意非线性函数逼近与任意状态分布下的收敛保证。

$$
\begin{aligned}q_*(s,a)&=\mathbb E[R_{t+1}-g_*+\max_{a'}q_*(S_{t+1},a')\mid s,a],\\\delta_t&=R_{t+1}-\bar g_t+\max_{a'}Q_t(S_{t+1},a')-Q_t(S_t,A_t),\\Q_{t+1}(S_t,A_t)&=Q_t(S_t,A_t)+\alpha_t\delta_t,\\\bar g_{t+1}&=\bar g_t+\eta\alpha_t\delta_t.\end{aligned}
$$

Differential Q 用 max 定义控制目标；采到的 (s,a) 在自身转移分布下更新，因此表格单步控制式不再乘评估式的 π/b。仍需要充分访问状态动作与适当步长。

**算法：Differential Q-learning；统计窗口不改变学习器状态**

1. 初始化 $Q(s,a)=0$、$\bar g=0$，选择 $\alpha>0$、$\eta>0$
1. 在每个环境步：
  1. 按 $\varepsilon$-greedy 策略选择动作，观察 $(s,a,r,s')$
  1. 用更新前参数计算 $\delta=r-\bar g+\max_{a'}Q(s',a')-Q(s,a)$
  1. 更新 $Q(s,a)\leftarrow Q(s,a)+\alpha\delta$
  1. 更新 $\bar g\leftarrow\bar g+\eta\alpha\delta$
  1. 令 $s\leftarrow s'$，继续交互

在理论证明里常使用满足随机逼近条件的递减步长并要求覆盖；在持续漂移中通常保留常数步长追踪新目标，代价是稳态方差和不同的保证。将线性特征换成深网也不是无条件继承表格收敛性。

<a id="lesson-rvi"></a>

## 3 · 相对价值迭代与 RVI Q-learning

转移分布与奖励均值已知时，可以直接计算 Bellman 更新。它与折扣算子不同：给所有 $h$ 加同一常数，更新结果也整体平移，不会由 $\gamma<1$ 压缩这个方向。RVI 每轮减去参考状态更新后的值，保持一个约定的原点。

$$
\begin{aligned}(Th)(s)&=\max_a\sum_{s',r}p(s',r\mid s,a)[r+h(s')],\\h_{k+1}(s)&=(Th_k)(s)-(Th_k)(s_{\rm ref}).\end{aligned}
$$

初始化 h(sref)=0；在适当条件下稳定后，(Th)(sref) 给出 g*。不要把第一轮任意偏移下的参考值立即当作真实奖励率；周期性也可能导致普通同步迭代振荡。

$$
Q_{t+1}(S_t,A_t)=Q_t(S_t,A_t)+\alpha_t\big[R_{t+1}-f(Q_t)+\max_a Q_t(S_{t+1},a)-Q_t(S_t,A_t)\big]
$$

RVI Q-learning 是采样形式：用满足所选理论条件的参考函数 f(Q) 代替另学一个 ḡ。一个常见选项是参考状态动作的 Q 值。参考函数影响数值锚定与瞬态行为，不能随意换成任何神经网络统计量后仍引用原证明。

| 方法 | 奖励率 / 原点从何而来 | 样本或模型 |
| --- | --- | --- |
| Differential TD/Q | 由同一 TD error 更新 ḡ；h/Q 可留常数偏移 | 单步实际样本 |
| RVI | 每轮减去参考状态的 backup | 完整已知模型 |
| RVI Q | f(Q) 提供参照；理论限制 f 的性质 | 访问到的状态动作样本 |
| Differential planning | 模型提供真实转移分布的模拟样本或期望 backup | 需另外检查模型误差和更新调度 |

<a id="lesson-duration"></a>

## 4 · 随机持续时间与半马尔可夫更新

执行一次导航技能，可能持续 2 步，也可能持续 200 步。若每次 option 结束只平均奖励，慢技能可能因为单次收益高而被偏爱。单位原始步奖励应为总奖励除以总耗时。在独立重复执行、均值有限且平均时长为正的例子中，它是 $\mathbb E[R]/\mathbb E[\tau]$，一般不同于 $\mathbb E[R/\tau]$。

$$
q_*(s,o)=\mathbb E\!\left[R_{t:t+\tau}-g_*\tau+\max_{o'}q_*(S_{t+\tau},o')\mid s,o\right]
$$

$R_{t:t+\tau}$ 是 option 内未折扣的外部奖励总和，$\tau$ 是原始步数，最大值只对终点可用的 options 取。时间扣除项为 $g_*\tau$，反映技能执行期间的机会成本。

下面采用期望时长归一化变体，维护严格正的长度估计 $L(s,o)$。用旧 $L$ 计算奖励率扣除与步长归一化后，再更新 $L$；当 $L=\mathbb E[\tau\mid s,o]$ 时，期望 TD 误差与上述半马尔可夫方程一致。该更新假设 option 终止且期望长度有限。

$$
\begin{aligned}\delta&=R-\bar g L_{\rm old}+\max_{o'}Q(s',o')-Q(s,o),\\Q(s,o)&\leftarrow Q(s,o)+\alpha\delta/L_{\rm old},\\\bar g&\leftarrow\bar g+\eta\alpha\delta/L_{\rm old},\\L(s,o)&\leftarrow L_{\rm old}+\alpha_L(\tau-L_{\rm old}).\end{aligned}
$$

若改为采样 τ 直接扣除、同时又用随机 τ 作分母，噪声与期望可能改变；必须追踪相应算法及其假设，不能把 L 和 τ 任意互换。

<a id="lesson-example"></a>

## 5 · 手算：奖励率、控制目标与更新时序

例一，确定性两状态环：$0\to1$ 的奖励为 0，$1\to0$ 的奖励为 2。每两步收益为 2，因此 $g=1$。取 $h(0)=0$，由 $1+h(0)=h(1)$ 得 $h(1)=1$，另一个方程 $1+1=2+0$ 也成立。另取一条用于单步计算的样本：$h=[1,3]$、$\bar g=0.5$，转移 $0\to1$、奖励为 2，则 $\delta=2-0.5+3-1=3.5$。当 $\alpha=\eta=0.1$ 时，更新后 $h(0)=1.35$、$\bar g=0.535$。这条单步样本的奖励与前述固定环不同。

例二，单状态双动作：动作 0 给奖励 0，动作 1 给奖励 1，然后返回原状态。探索行为各做一半，实际奖励率为 $0.5$；最优奖励率为 $1$，最优方程要求 $Q(1)-Q(0)=1$。Differential Q 的 $\bar g$ 估计的是最优目标，因此应接近 1。这个例子把控制目标与行为奖励滑动平均明确区分开。

例三，当前 $Q=1$、$\bar g=0.5$、旧 $L=2$；option 累计奖励为 5，实际长度为 4，终点最大价值为 3。由 $\delta=5-0.5\times2+3-1=6$，取 $\alpha=\eta=0.1$ 得更新后 $Q=1.3$、$\bar g=0.53$；最后取 $\alpha_L=0.1$ 得 $L=2.2$。三个结果均按旧长度构造，更新顺序因此是算法定义的一部分。

<a id="lesson-code"></a>

## 6 · 完整更新、运行命令与输出诊断

可执行源码：函数返回新表格，避免原地修改使后续目标读到半更新参数。

```python
def differential_td(values, rate, state, reward, next_state,
                    alpha=0.1, eta=0.1, rho=1.0):
    """Tabular fixed-policy prediction. Compute BOTH updates from old values."""
    if alpha < 0 or eta < 0 or rho < 0:
        raise ValueError("nonnegative steps and importance ratio required")
    error = reward - rate + values[next_state] - values[state]
    result = list(values)
    result[state] += alpha * rho * error
    return result, rate + eta * alpha * rho * error, error


def differential_q(q, rate, state, action, reward, next_state,
                   alpha=0.1, eta=0.1):
    """Off-policy control: max target; no behavior-reward EMA."""
    error = reward - rate + max(q[next_state]) - q[state][action]
    result = [row[:] for row in q]
    result[state][action] += alpha * error
    return result, rate + eta * alpha * error, error


def rvi_sweep(h, transitions, reference=0):
    """Known model: transitions[s][a] contains (probability,reward,next_state)."""
    backed = [max(sum(prob * (reward + h[sp]) for prob, reward, sp in action)
                  for action in state) for state in transitions]
    reference_value = backed[reference]
    return [v - reference_value for v in backed], reference_value


def option_rate_step(q, rate, mean_duration, total_reward, duration,
                     next_best, alpha=0.1, eta=0.1, duration_alpha=0.1):
    """Duration-normalized expected-length variant; use OLD positive length."""
    if mean_duration <= 0 or duration <= 0:
        raise ValueError("option durations must be positive")
    error = total_reward - rate * mean_duration + next_best - q
    scaled = alpha * error / mean_duration
    return (q + scaled, rate + eta * scaled,
            mean_duration + duration_alpha * (duration - mean_duration))


def average_demo():
    values, rate = [0.0, 0.0], 0.0
    for t in range(20000):
        state = t % 2
        values, rate, _ = differential_td(
            values, rate, state, 2.0 * state, 1 - state, 0.02, 0.1)
    q, optimal_rate = [[0.0, 0.0]], 0.0
    for t in range(10000):
        action = t % 2  # behavior reward rate is exactly 0.5
        q, optimal_rate, _ = differential_q(q, optimal_rate, 0, action,
                                            float(action), 0, 0.02, 0.1)
    print("average", {"prediction_rate": round(rate, 6),
          "h1_minus_h0": round(values[1] - values[0], 6),
          "control_rate": round(optimal_rate, 6), "behavior_rate": 0.5,
          "option_step": option_rate_step(1, 0.5, 2, 5, 4, 3)})
```

下载本页实验文件后运行；Python 3.10+，仅标准库

```sh
python lifelong_algorithms_lab.py average
python lifelong_algorithms_lab.py test
```

预期：两状态环 prediction_rate≈1、h1_minus_h0≈1；单状态控制 control_rate≈1，而 behavior_rate=0.5；option_step=(1.3,0.53,2.2)。测试额外验证值函数整体平移不改变 TD error、ρ=0 时不更新、零时长被拒绝、随机时长的“比值均值”反例。

作者仓库 average-reward-methods 将预测与控制分别放在 prediction_agents.py 和 control_agents.py。阅读时先对应奖励率与价值更新，再检查参考函数、探索策略、步长与环境重置规则。本页脚本用于表格机制实验；作者的统计实验还需要运行相应环境、多个随机种子及完整训练配置。

<a id="lesson-centering"></a>

## 7 · 奖励中心化：连接折扣价值与平均奖励

为什么 $\gamma$ 接近 $1$ 时，折扣价值往往很大，而动作之间的差异相对很小？设每步奖励减去固定常数 $c$。由于几何级数之和为 $1/(1-\gamma)$，所有策略、状态和动作的折扣价值都平移同一个量：

$$
Q_{r-c,\gamma}^{\pi}(s,a)=Q_{r,\gamma}^{\pi}(s,a)-\frac{c}{1-\gamma}
$$

此式假设无限持续交互、固定折扣且每一步均减去 $c$。因此固定 $c$ 不改变动作排序。若只在长度可变的 episode 内减去常数，累计平移依赖终止时间，这个结论就不再直接成立。

Naik 等人的 Reward Centering（RLC 2024）据此将共同的奖励偏移从价值学习中分离。它与 Differential TD/Q 共用一个重要思想：同时学习价值差异与标量参照量。但保留 $\gamma<1$ 时，学习的仍是中心化的折扣价值，而不是自动改成平均奖励控制。

$$
\delta_t=R_{t+1}-\bar g_t+\gamma\max_{a'}Q_t(S_{t+1},a')-Q_t(S_t,A_t),\qquad \bar g_{t+1}=\bar g_t+\eta\alpha_t\delta_t
$$

这是 TD 驱动的控制中心化形式。$Q$ 同时按 $\alpha_t\delta_t$ 更新。固定参照量下的平移恒等式说明其动机；参照量与函数近似同时变化时，还需分析联合学习过程。

例如 $c=2$、$\gamma=0.99$ 时，共同价值偏移是 $200$；移除它可让网络更多容量用于区分状态和动作。Wan、Korenkevych 与 Zhu 的 continuing-task 研究（2025）进一步比较了无重置、预设重置和智能体控制重置的环境，并发现中心化不能完全消除大折扣带来的性能下降。它解决部分数值与估计问题，并不消除恢复困难或探索不足。

- 对照实验一：给所有真实奖励加同一个常数，比较中心化前后的学习曲线和价值尺度。
- 对照实验二：保留奖励但改变重置规则，区分数值问题与恢复能力。
- 对照实验三：分别采用行为奖励滑动平均、TD 驱动参照量和指定参考状态，检查 off-policy 目标是否一致。

<a id="lesson-branches"></a>

## 8 · 从差分预测到深度平均奖励控制

平均奖励 off-policy 预测加函数逼近后，数据分布与目标占用分布不一致会破坏简单 TD 的稳定性。Differential GQ 等方法引入辅助变量与投影目标；它们解决的是策略评估，不等于完成了未知目标策略的深度控制。本页先把表格目标和时序固定，避免将数学对象、求解器和网络结构混为一谈。

$$
\begin{aligned}\widetilde V_Q(s')&=\mathbb E_{a'\sim\pi_\theta}[Q(s',a')-\tau_{\!H}\log\pi_\theta(a'\mid s')],\\y&=r-f(Q)+\widetilde V_Q(s'),\qquad L_Q=\mathbb E[(Q(s,a)-\operatorname{stopgrad}(y))^2].\end{aligned}
$$

这是 soft average-reward critic 的概念骨架，不是 RVI-SAC 全部实现。熵温度 τH 与 option 时长 τ 不同；具体算法还需要目标网络、双 critic、参考项估计及 reset 成本等。

RVI-SAC 将参考项估计与最大熵控制结合。训练收益可包含熵和重置成本，评测的外部奖励率则可能不含这些项。若失败后的重置是持续过程中的真实转移，价值目标需要计入其后续收益；若它是问题定义中的吸收终止，边界条件便不同。重置的动作、代价与时间决定了目标的形式。

- 研究目标改变：比较平均奖励、折扣目标和 reward centering。保留 γ<1 的 centering 主要改变数值与 baseline，不自动变成平均奖励。
- 研究可追踪性：在不向 agent 提供切换时间的情况下改变奖励或转移，报告变化后累计损失与估计延迟。
- 研究函数逼近：固定特征、线性辅助变量、深网三层递进；先确定失效来自覆盖、投影还是表示漂移。
- 研究 SMDP：控制时长分布、奖励与时长相关性，并统一按原始环境步计算分母。

<a id="lesson-check"></a>

## 9 · 习题与诊断

| 问题 | 应当能给出的解释 |
| --- | --- |
| h 全体加 100，策略会变吗？ | max 的相对次序和 TD 差分不变；参考规范化会改变存储值。 |
| 为何控制实验中 ḡ≈1，但真实流只有 0.5？ | ḡ 对应最优 Bellman 目标；实际探索仍选择低奖励动作。二者必须分别报告。 |
| 只有一次不可重置生命，能否用 g 的渐近目标解释全部价值？ | 目标仍可用，但有限生命损失、不可逆风险与恢复成本不能由极限奖励率单独概括。 |
| 算法收敛证明能否覆盖突然改变的深网实验？ | 不直接覆盖；需要重新陈述平稳性、函数逼近、覆盖和步长假设。 |

动手题：把两状态环的一条奖励改为 4，保留所有学习器状态。分别比较递减步长与常数步长的追踪速度；预测新 g=2、h(1)−h(0)=2。再把日志分段，但绝不在分段边界清零参数。将“记录一个新窗口”和“重新开始学习”严格分开。

## 本章的实验设计

记录每单位原始时间的收益。技能调用次数不能代替经过的时间。分别报告暂态和长期表现。

设定：先比较两个持续时间不同的固定技能：累计奖励分别为 2、9，时长分别为 1、9；再接小型 continuing MDP 的 Differential TD/Q。

- 总奖励率为 11/10，而不是两个技能奖励率的平均数。
- 所有奖励加常数后，奖励率按相同常数平移。
- 差分价值的常数偏移不改变所比较的动作优势。

对照：固定策略的解析奖励率；匹配动作持续时间的 primitive 对照；单独列出的折扣目标基线

记录：总原始奖励除以总原始时长；奖励率估计误差与暂态表现；差分 Bellman 残差及各状态访问量

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-modules)

## 学习与研究衔接

平均奖励按原始时间计收益。随机时长 option 要使用半马尔可夫时间口径。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-average) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=average) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=average)

<a id="chapter-code"></a>

## 下载与运行

标准库表格机制实验与单步公式测试；不是深度算法或论文 benchmark 复现。

[下载 lifelong_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)

```sh
python lifelong_algorithms_lab.py average
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Wan, Naik & Sutton · Learning and Planning in Average-Reward MDPs](https://proceedings.mlr.press/v139/wan21a.html)：表格 Differential TD/Q 与规划的原论文；对应本文奖励率由 TD error 驱动的核心。

- [作者实验仓库 · average-reward-methods](https://github.com/abhisheknaik96/average-reward-methods)：从 agents/prediction_agents.py 和 control_agents.py 对照更新，再读各实验配置；非本页教学代码的性能保证。

- [Wan, Naik & Sutton · Average-Reward Learning and Planning with Options](https://arxiv.org/abs/2110.13855)：不同 option 学习/规划更新与时长归一化；本页展示的是明确标记的 expected-duration 变体。

- [Wan & Sutton · Weakly Communicating MDPs](https://arxiv.org/abs/2209.15141)：扩展理解平均奖励控制的链结构和收敛条件；不可外推为任意非平稳深网定理。

- [Hisaki & Ono · RVI-SAC](https://proceedings.mlr.press/v235/hisaki24a.html)：深度平均奖励、熵正则与 reset 成本的完整方法。

- [RVI-SAC 作者代码](https://github.com/yhisaki/average-reward-drl)：核心文件 average_reward_drl/algorithms/rvi_sac.py；完整实验需要相应 MuJoCo 环境与训练配置。

- [Naik et al. · Reward Centering · RLC 2024](https://rlj.cs.umass.edu/2024/papers/Paper261.html)：将共同奖励偏移与折扣价值差异分离；对照 on-policy 奖励均值与 off-policy TD 中心化。

- [Wan, Korenkevych & Zhu · Deep RL in Continuing Tasks · 2025](https://arxiv.org/abs/2501.06937)：无重置、预设重置、智能体控制重置三种协议，以及 TD 中心化的深度实验和局限。

- [DeepRL-continuing-tasks · 环境协议与 reward centering](https://github.com/facebookresearch/DeepRL-continuing-tasks)：核对无重置、预设重置与 agent-controlled reset 的协议差异。
