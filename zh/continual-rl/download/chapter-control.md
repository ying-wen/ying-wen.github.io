# 控制问题：策略改进与动态规划

控制的目标是选择行为，使长期回报更高。本章先定义策略之间的优劣，再推导策略改进、策略迭代与价值迭代，最后把模型期望换成真实交互中的采样更新。

## 本章内容

- 区分给定策略的预测问题与寻找更好策略的控制问题，并说明最优价值的含义。
- 从 Bellman 期望方程推导策略改进定理，独立实现策略迭代和价值迭代。
- 写出 SARSA、Q-learning 与 actor–critic 的交互闭环，辨认探索、访问分布与评价对象。
- 判断有限平稳 MDP 的结论在函数逼近、部分可观测和持续学习中缺少哪些条件。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 一条经验

在 $S_{t}$ 做 $A_{t}$ 后得到 $R_{t+1}$、$S_{t+1}$。奖励属于这次转移，不是到达状态之前另一轮的奖励。

### 折扣与终止

$\gamma$ 控制未来相对权重；真实终止后的价值设为 0。本章记 $\gamma_{t+1}$=$\gamma$(1−terminated)。训练脚本的时间截断不一定是任务终止。

### 表格与函数逼近

表格每个状态或状态动作一组独立参数；函数逼近让不同输入共享参数。更新一个输入可能改变其他输入的预测。

<a id="lesson-setting"></a>

## 1 · 控制问题与策略的优劣

先考虑有限、平稳的 Markov 决策过程（MDP）。状态集合为 $\mathcal S$，状态 $s$ 的合法动作为 $\mathcal A(s)$。模型 $p(s',r\mid s,a)$ 描述动作之后的状态与奖励分布。策略 $\pi(a\mid s)$ 决定行动概率。奖励有界，固定折扣满足 $0\le\gamma<1$。真实终止状态之后的价值为零，但进入终点时的奖励仍计入回报。

$$
G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1},\qquad v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s],\qquad q_\pi(s,a)=\mathbb E_\pi[G_t\mid S_t=s,A_t=a]
$$

状态价值评价从这里开始、所有动作都按该策略选择的结果。动作价值只把第一步动作固定，后续仍按该策略选择。预测问题给定策略并估计这些量；控制问题还要改变策略。

$$
\pi'\succeq\pi\ \Longleftrightarrow\ v_{\pi'}(s)\ge v_\pi(s)\quad\forall s;\qquad v_*(s)=\max_\pi v_\pi(s),\quad q_*(s,a)=\max_\pi q_\pi(s,a)
$$

策略排序要求每个状态都不差，这是偏序：两个策略可能分别擅长不同起点，无法互相支配。在有限折扣 MDP 中，存在同一个确定性平稳策略，在所有状态同时达到最优价值；不必为每个起点分别寻找互不相容的策略。

若只关心给定起始分布 $d_0$，也可优化 $J(\pi)=\sum_s d_0(s)v_\pi(s)$。这个标量目标与逐状态排序不同：改进平均值不意味着每个状态都改善。下面的精确动态规划先采用逐状态标准；参数化策略梯度则通常明确指定 $J$。

| 对象 | 固定什么 | 求什么 |
| --- | --- | --- |
| 预测 | 策略、回报信号、环境 | 该策略的价值 |
| 控制 | 奖励目标、环境 | 行为策略及其价值 |
| 规划 | 可查询的模型 | 利用模型计算较好的决策 |
| 交互学习 | 只能取得实际经验 | 同时估计后果、选择行为和收集后续数据 |

<a id="lesson-derive"></a>

## 2 · Bellman 最优方程

由回报递推 $G_t=R_{t+1}+\gamma G_{t+1}$，先对一次转移取期望，再处理后续策略。在给定策略的预测问题里，动作按 $\pi$ 加权。控制则在观察到当前状态后，从合法动作中选择期望结果最大的一个。

$$
(T_\pi V)(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)[r+\gamma V(s')],\qquad v_\pi=T_\pi v_\pi
$$

这是评价算子。V 是任意当前估计，不一定是某个策略的真实价值。算子把下一状态的估计转换为当前状态的一步期望目标。

$$
(TV)(s)=\max_{a\in\mathcal A(s)}\sum_{s',r}p(s',r\mid s,a)[r+\gamma V(s')],\qquad v_*=Tv_*
$$

最优算子先对动作后果求期望，再比较动作。不能改成对每个随机后果先取最大值：那相当于在行动之前就知道尚未发生的随机结果。最优方程成立还使用了最优继续行为能在后继状态实现这一事实。

$$
q_*(s,a)=\sum_{s',r}p(s',r\mid s,a)[r+\gamma\max_{a'\in\mathcal A(s')}q_*(s',a')],\qquad v_*(s)=\max_a q_*(s,a)
$$

动作价值最优式先固定本次动作，因此只在下一状态取最大值。终止后没有合法动作，约定后继项为零，不能对空动作集合直接调用程序中的 max。

$$
\|TU-TV\|_\infty\le\gamma\|U-V\|_\infty,\qquad \|T_\pi U-T_\pi V\|_\infty\le\gamma\|U-V\|_\infty
$$

证明只需两步：期望差不超过最大的输入差；两组动作值的最大值之差也不超过逐动作最大差。折扣使差缩小。因此这两个算子各有唯一固定点，反复精确备份能够逼近它。该结论还没有涉及采样误差或神经网络。

<a id="control-improvement"></a>

## 3 · 策略改进定理

假设已经准确评价策略 $\pi$。现在构造新策略 $\pi'$，使它在每个状态选择的动作，按旧策略的后续价值衡量时都不差于原先。确定性贪心选择 $\pi'(s)\in\arg\max_a q_\pi(s,a)$ 是一个充分选择；随机策略只需满足下式。

$$
(T_{\pi'}v_\pi)(s)=\sum_a\pi'(a\mid s)q_\pi(s,a)\ge v_\pi(s)\quad\forall s
$$

这是单步改进条件。它比较的是“第一步改用新策略，后面仍按旧策略”的结果。下一步需要证明：将新选择应用到整个未来，仍然不会变差。

$$
v_\pi\le T_{\pi'}v_\pi\le T_{\pi'}^2v_\pi\le\cdots\longrightarrow v_{\pi'}
$$

算子保持逐分量大小关系，因为转移概率和折扣非负；所以可以反复对不等式两边应用同一算子。压缩性保证极限是新策略的价值。由此得到全程使用新策略的价值不低于旧策略。

$$
\pi'\text{ 对 }v_\pi\text{ 贪心且 }v_{\pi'}=v_\pi\quad\Longrightarrow\quad Tv_\pi=T_{\pi'}v_\pi=v_\pi=v_*
$$

若贪心改进后价值不再变化，旧价值已满足最优 Bellman 方程，因固定点唯一而达到最优。精确的有限策略迭代据此停止。

这里用的是准确的 $v_\pi$。估计 $\widehat v_\pi$ 的误差可能把动作排错，函数逼近还可能在改善一个状态时损害另一个状态。因此，实际 actor–critic 或深度 Q-learning 的每次更新不自动满足这个逐状态改进定理。多个动作并列时，程序使用固定的排序，或保留原动作，避免无意义的策略切换。

<a id="control-dp"></a>

## 4 · 策略迭代、价值迭代与广义策略迭代

策略迭代（policy iteration）交替进行两个完整阶段：评价当前策略，再据此改善所有状态的动作。已知模型时，评价可解线性方程 $v_\pi=(I-\gamma P_\pi)^{-1}r_\pi$，也可重复应用 $T_\pi$。大型问题通常不显式求逆，而采用迭代评价。

**算法：策略迭代：定理使用精确评价，数值实现使用足够严格的误差阈值**

1. 初始化确定性策略 $\pi$；终止状态的价值固定为 $0$。
1. 重复：
  1. 策略评价：反复执行 $V\leftarrow T_\pi V$，直到评价精度满足要求。
  1. 保存旧策略 $\pi_{\rm old}\leftarrow\pi$。
  1. 对每个非终止状态 $s$：
    1. $\pi(s)\leftarrow\arg\max_a\sum_{s',r}p(s',r\mid s,a)[r+\gamma V(s')]$。
  1. 若策略没有变化，则返回 $\pi,V$。

价值迭代不等待评价某个策略完成。每轮对每个状态直接应用最优算子，再从最终价值导出贪心策略。它把评价和改善压缩进一次备份，但仍然使用环境模型。同步实现每轮只读旧值；原地更新属于另一种更新调度，也可在适当条件下收敛。

$$
V_{k+1}=TV_k,\qquad \varepsilon(V)=\|TV-V\|_\infty,\qquad \|V-v_*\|_\infty\le\frac{\varepsilon(V)}{1-\gamma}
$$

误差界来自三角不等式：当前值与最优值的差，不超过当前 Bellman 残差再加折扣后的同一差。将后一项移到左边即可。代码计算的是返回值本身的残差，而不是误将上一轮的变化量当成同一个数。

Sutton 与 Barto 的广义策略迭代（Generalized Policy Iteration，GPI）指评价过程与改善过程的相互作用。评价不必完全收敛，改善不必一次覆盖所有状态。策略变化会改变待评价的价值；价值变化又会改变策略。动态规划、MC 控制和 TD 控制都可放在这个框架中。这里的 GPI 不是另一种算法，也不单独提供非线性近似的收敛保证。

另一个同缩写术语是迁移学习中的 Generalized Policy Improvement（广义策略改进）：给定多个已有策略的动作价值，按 $\arg\max_a\max_i q_{\pi_i}(s,a)$ 组合它们。前者是“评价与改善反复交互”的框架；后者是“从多个策略的后果估计中改善行为”的算子。讨论 successor features 时应明确指后者。

<a id="lesson-example"></a>

## 5 · 同一个 MDP 的完整计算

环境有 A、B 两个非终止状态及终点。折扣 $\gamma=0.9$。A 的动作 0 获得 1 并终止，动作 1 获得 0 并到 B。B 的动作 0 获得 2 并终止，动作 1 获得 0.5 并回 A。这个例子同时包含短期结束与长期循环，两者必须用同一个折扣目标比较。

| 阶段 | A / B 的动作 | 评价结果 | 下一次改善的依据 |
| --- | --- | --- | --- |
| 初始策略 | 0 / 0 | V(A)=1，V(B)=2 | A 继续值为 0+.9×2=1.8；B 回 A 仅为 .5+.9×1=1.4 |
| 第一次改善 | 1 / 0 | V(A)=1.8，V(B)=2 | B 回 A 变成 .5+.9×1.8=2.12，高于立即结束的 2 |
| 第二次改善 | 1 / 1 | V(A)=45/19，V(B)=50/19 | 两处继续动作均优于终止；策略稳定 |

$$
V(A)=0.9V(B),\qquad V(B)=0.5+0.9V(A)\quad\Longrightarrow\quad V(A)=\frac{0.45}{1-0.81}=\frac{45}{19},\quad V(B)=\frac{50}{19}
$$

最优策略永不终止，但回报有限，因为折扣严格小于一。这个例子不是平均奖励问题，也不要求所有策略都终止。

| 价值迭代轮次 | V(A) | V(B) |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 1 | 2 |
| 2 | 1.8 | 2 |
| 3 | 1.8 | 2.12 |
| 4 | 1.908 | 2.12 |
| 极限 | 2.368421… | 2.631579… |

价值迭代每轮只用上一轮的值，因此长期循环的收益逐轮传播。策略迭代则在每次改善之间求当前策略的长期价值。两者经过的中间估计不同，最终解相同。

<a id="control-sampling"></a>

## 6 · 从模型期望到 MC 与 TD 控制

交互学习通常不能查询完整 $p(s',r\mid s,a)$，只能看到实际执行的动作及后果。一条经验可以替代期望备份中的一次采样，但算法只更新访问到的状态动作。MC 控制等待完整回报，以 $Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha[G_t-Q(S_t,A_t)]$ 评价当前策略，再改善策略；TD 控制用后继估计替代尚未观察到的余项。

$$
\begin{aligned}
Y_t^{\rm Sarsa}&=R_{t+1}+\gamma_{t+1}Q_t(S_{t+1},A_{t+1}),\quad A_{t+1}\sim b_t(\cdot\mid S_{t+1}),\\
Y_t^{\rm Expected}&=R_{t+1}+\gamma_{t+1}\sum_a\pi_t(a\mid S_{t+1})Q_t(S_{t+1},a),\\
Y_t^{Q}&=R_{t+1}+\gamma_{t+1}\max_aQ_t(S_{t+1},a),\\
Q_{t+1}(S_t,A_t)&=Q_t(S_t,A_t)+\alpha_t[Y_t-Q_t(S_t,A_t)].
\end{aligned}
$$

实际选动作的行为策略是 b；target 所评价的目标策略是 π。SARSA 使用实际要执行的下一动作。Expected SARSA 显式平均目标策略的下一动作。Q-learning 使用贪心目标，不要求行为也贪心。真实终止时令延续因子为零。

Q-learning 的单步表格更新不需额外乘动作重要性比率：它已经条件于实际访问的状态动作，只对这个动作的环境后果采样。需要修正的不是“这个动作有没有经常发生”，而是多步余项中的动作分布或函数逼近目标的加权；它们属于进一步的问题。

固定折扣 MDP 中，Q-learning 的经典表格收敛条件包括每个状态动作被无限访问，以及逐对步长满足 $\sum_n\alpha_n(s,a)=\infty$、$\sum_n\alpha_n(s,a)^2<\infty$。SARSA 的最优收敛还需要行为逐渐变得贪心而探索仍无限进行，即 GLIE。固定 $\epsilon>0$ 的 SARSA 评价包含探索代价的行为，不能直接声称收敛到无探索的 $q_*$。

<a id="control-loop"></a>

## 7 · 动作选择、更新时序与探索

$$
b_t(a\mid s)=\frac{\epsilon}{|\mathcal A(s)|}+(1-\epsilon)\frac{\mathbf1\{a\in\arg\max_{a'}Q_t(s,a')\}}{|\arg\max_{a'}Q_t(s,a')|}
$$

探索部分平均分给所有合法动作，贪心部分平均分给并列最优动作。随着 Q 变化，即使 ε 固定，行为策略也会变化。这正是本章新实验与固定行为预测对照的区别。

**算法：动作采样与价值更新共同形成控制闭环**

1. 初始化 $Q$；观察起始状态 $S$；由 $b_Q$ 选动作 $A$。
1. 每次转移：
  1. 执行 $A$，观察 $R,S'$ 和真实终止标记。
  1. 若非终止且使用 SARSA，先由当前 $b_Q$ 选择并保存 $A'$。
  1. 用旧 $Q$ 计算选定算法的目标 $Y$；终止时 $Y=R$。
  1. 更新当前表项 $Q(S,A)\leftarrow Q(S,A)+\alpha[Y-Q(S,A)]$。
  1. 若真实终止，按任务协议重新开始；否则令 $S\leftarrow S'$。
  1. SARSA 继续执行保存的 $A'$；Q-learning 可按更新后的 $b_Q$ 重新选择下一动作。

探索决定未来会获得哪些证据。有限训练中，$\epsilon$-greedy 也不保证有效覆盖稀有状态或长动作序列。完全贪心可能永远不尝试初始低估的动作；随机探索也可能频繁进入危险区域。探索奖励、乐观估计、技能和模型规划是在这个数据获取问题上增加结构，不是修改一个 TD 误差就自动解决覆盖。

表格中每个状态动作有独立参数。函数逼近中，行为访问分布 $d_b(s,a)$ 还决定哪些误差主导拟合；共享参数可能牺牲很少访问但重要的状态。离线数据没有新的交互来纠正策略偏好，因此还必须限制对数据支持之外动作的乐观估计。

<a id="control-actor-critic"></a>

## 8 · Actor–critic 的评价与改善闭环

当动作连续，或希望直接表示随机策略时，可以给策略单独分配参数 $\theta$。Actor 根据 $\pi_\theta(a\mid s)$ 行动，critic 用参数 $w$ 估计它的价值。它们仍分别承担改善与评价，但改善由可微的目标实现，不再逐个枚举所有动作取最大值。

$$
J(\theta)=\mathbb E_{\pi_\theta,S_0\sim d_0}[G_0],\qquad
\nabla J(\theta)=\mathbb E\!\left[\sum_{t\ge0}\gamma^t\nabla_\theta\log\pi_\theta(A_t\mid S_t)q_{\pi_\theta}(S_t,A_t)\right]
$$

固定起始分布、折扣回报下的策略梯度，用轨迹各时刻的折扣权重表达。不能随意去掉外部的 γ 的 t 次方，再声称仍是同一起始分布目标；也可通过明确的折扣占据分布来等价表达。

$$
\begin{aligned}
\delta_t&=R_{t+1}+\gamma_{t+1}V_{w_t}(S_{t+1})-V_{w_t}(S_t),\\
w_{t+1}&=w_t+\alpha_w\delta_t\nabla_wV_{w_t}(S_t),\\
\theta_{t+1}&=\theta_t+\alpha_\theta I_t\delta_t\nabla_\theta\log\pi_{\theta_t}(A_t\mid S_t),\quad I_t=\gamma^t .
\end{aligned}
$$

这是回合制折扣目标的一步、on-policy actor–critic 更新。两组梯度与 δ 使用更新前参数；I 在回合起点为 1，每次非终止转移乘 γ。真实终止重置 I。不能把终止标记用于抹去本次奖励。

若 critic 恰为 $v_{\pi_\theta}$，则给定 $(s,a)$ 的 TD 误差期望为 $q_{\pi_\theta}(s,a)-v_{\pi_\theta}(s)$，即优势。状态基线不改变精确策略梯度，因为 $\sum_a\pi_\theta(a\mid s)\nabla_\theta\log\pi_\theta(a\mid s)=0$。实际 critic 有估计误差，策略又同步变化，所以这个一步更新通常是近似改善，而不是前面策略改进定理的逐步保证。

一次闭环包含：actor 选动作，环境返回后果，critic 形成误差，两者更新，然后新 actor 生成下一条数据。PPO 限制一批数据上策略改变的幅度；SAC 在策略目标中加入熵并使用离策略 critic；确定性 actor–critic 用动作价值对动作的梯度改进连续动作。它们不能仅靠“actor 加 critic”四个字区分，需要继续检查优化目标、数据分布和更新调度。

<a id="lesson-code"></a>

## 9 · 可运行实现与验证

两状态 MDP、模型期望和实际环境转移接口

```python
GAMMA = 0.9
# Outcomes are (probability, reward, next_state). State 2 is terminal.
MDP = {
    0: {0: [(1.0, 1.0, 2)], 1: [(1.0, 0.0, 1)]},
    1: {0: [(1.0, 2.0, 2)], 1: [(1.0, 0.5, 0)]},
    2: {},
}


def action_value(state, action, values, gamma=GAMMA):
    return sum(p * (r + gamma * values[nxt])
               for p, r, nxt in MDP[state][action])


def greedy_policy(values, gamma=GAMMA):
    # Deterministic, consistent tie breaking for finite policy iteration.
    return {s: max(actions, key=lambda a: action_value(s, a, values, gamma))
            for s, actions in MDP.items() if actions}


def sample_transition(state, action, rng):
    draw, cumulative = rng.random(), 0.0
    for p, reward, nxt in MDP[state][action]:
        cumulative += p
        if draw < cumulative:
            return reward, nxt, not MDP[nxt]
    raise ValueError("Transition probabilities must sum to one")
```

策略评价、策略迭代、价值迭代与 Bellman 残差

```python
def evaluate_policy(policy, gamma=GAMMA, tolerance=1e-12):
    """Synchronous expectation backups; terminal value stays zero."""
    values = {s: 0.0 for s in MDP}
    for _ in range(100000):
        updated = {s: action_value(s, policy[s], values, gamma)
                   if actions else 0.0 for s, actions in MDP.items()}
        change = max(abs(updated[s] - values[s]) for s in MDP)
        values = updated
        if change < tolerance:
            return values
    raise RuntimeError("Policy evaluation did not converge")


def policy_iteration():
    policy, history = {0: 0, 1: 0}, []
    while True:
        values = evaluate_policy(policy)
        history.append((policy.copy(), values.copy()))
        improved = greedy_policy(values)
        if improved == policy:
            return policy, values, history
        policy = improved


def optimality_backup(values, gamma=GAMMA):
    return {s: max(action_value(s, a, values, gamma) for a in actions)
            if actions else 0.0 for s, actions in MDP.items()}


def value_iteration(tolerance=1e-10):
    values = {s: 0.0 for s in MDP}
    for sweep in range(1, 100000):
        values = optimality_backup(values)
        next_values = optimality_backup(values)
        residual = max(abs(next_values[s] - values[s]) for s in MDP)
        if residual <= tolerance:
            return greedy_policy(values), values, sweep, residual
    raise RuntimeError("Value iteration did not converge")
```

Q-learning 的完整交互、探索和真实终止处理

```python
def epsilon_probabilities(row, epsilon):
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between zero and one")
    best = max(row.values())
    ties = [a for a, value in row.items() if value == best]
    return {a: epsilon / len(row) +
            ((1.0 - epsilon) / len(ties) if a in ties else 0.0)
            for a in row}


def choose_action(row, epsilon, rng):
    probs = epsilon_probabilities(row, epsilon)
    draw, cumulative = rng.random(), 0.0
    for action, probability in probs.items():
        cumulative += probability
        if draw < cumulative:
            return action
    return next(reversed(probs))  # floating-point rounding only


def q_learning(steps=20000, epsilon=0.2, alpha=0.1, seed=7):
    rng = random.Random(seed)
    q = {s: {a: 0.0 for a in actions}
         for s, actions in MDP.items() if actions}
    visits = {s: {a: 0 for a in row} for s, row in q.items()}
    state, reward_sum, resets = 0, 0.0, 0
    for _ in range(steps):
        # This behavior policy changes whenever Q changes.
        action = choose_action(q[state], epsilon, rng)
        reward, nxt, terminated = sample_transition(state, action, rng)
        target = reward if terminated else reward + GAMMA * max(q[nxt].values())
        q[state][action] += alpha * (target - q[state][action])
        visits[state][action] += 1
        reward_sum += reward
        if terminated:
            resets += 1
            state = 0  # A new task episode, only after a true terminal state.
        else:
            state = nxt
    policy = {s: max(row, key=row.get) for s, row in q.items()}
    return q, policy, visits, reward_sum, resets
```

下载本章 Python 文件后运行；仅依赖标准库

```sh
python3 control_problem_lab.py dp
python3 control_problem_lab.py control
python3 control_problem_lab.py test
```

**算法：默认种子和参数的一次运行输出**

1. PI 0: policy={0: 0, 1: 0}, V(A)=1.000000, V(B)=2.000000
1. PI 1: policy={0: 1, 1: 0}, V(A)=1.800000, V(B)=2.000000
1. PI 2: policy={0: 1, 1: 1}, V(A)=2.368421, V(B)=2.631579
1. VI: policy={0: 1, 1: 1}, V(A)=2.368421, V(B)=2.631579, sweeps=201, residual=9.407e-11
1. Q-learning: Q={0: {0: 1.0, 1: 2.368421}, 1: {0: 2.0, 1: 2.631579}}
1. greedy policy={0: 1, 1: 1}, independently evaluated V(A)=2.368421

Q-learning 运行 20000 次真实环境转移，每次都由当前 Q 重新定义行为。训练仅看到采样后果，不读取 DP 求得的值。学习结束后才用模型独立评价贪心策略。默认记录 2250 次真实终止与重置；累计训练奖励为 7359.5。这个训练总和受到探索和重置次数影响，不等于从 A 出发的折扣最优价值。

本例转移确定，常数步长即可数值逼近已知解。这验证了更新与交互实现，不证明常数步长在随机任务中的几乎必然收敛。12 个测试覆盖概率归一化、策略改进单调性、解析最优解、残差界、终止奖励、并列动作、访问覆盖和交互结果。

补充对照：固定行为下的 SARSA、Expected SARSA 与 Q-learning 目标

```python
def control_target(reward, discount, next_q, method, next_action=None, probs=None):
    if discount == 0.0:
        return reward
    if method == "q":
        bootstrap = max(next_q)
    elif method == "sarsa":
        bootstrap = next_q[next_action]
    elif method == "expected":
        bootstrap = sum(p * q for p, q in zip(probs, next_q))
    else:
        raise ValueError(method)
    return reward + discount * bootstrap


def train_control(method, episodes=20000, seed=7):
    """Only one action at A; at B behavior chooses bad action with p=.1."""
    rng = random.Random(seed)
    q_a, q_b = 0.0, [0.0, 0.0]
    for _ in range(episodes):
        next_action = int(rng.random() < 0.1)
        target = control_target(0.0, 0.9, q_b, method, next_action, [0.9, 0.1])
        q_a += 0.01 * (target - q_a)
        reward = 1.0 if next_action == 0 else -1.0
        q_b[next_action] += 0.01 * (reward - q_b[next_action])
    return {"q_A": q_a, "q_B": q_b}
```

补充脚本把行为固定为在 B 以 0.9/0.1 选择奖励 +1/−1 的动作，A 到 B 的奖励为零，$\gamma=0.9$。Q-learning 在 A 的目标为 0.9；Expected SARSA 为 $0.9(0.9-0.1)=0.72$；SARSA 的样本目标在 ±0.9 间变化，期望也是 0.72。它解释预测对象为何不同，不是策略随学习改善的控制性能实验。

- 实验一：把折扣改为 0，先用数学预测最优动作，再同步修改 DP 与采样控制的折扣进行验证。
- 实验二：降低训练步数，同时报告各动作访问次数、Q 误差与贪心策略价值。它们未必同步改善。
- 实验三：给奖励加入零均值噪声，多次运行比较常数步长的跟踪方差与递减步长的稳定性。解析期望模型应保持相同。
- 实验四：训练中途交换奖励，比较持续探索与停止探索；分别检查环境变化是否被发现、发现后的更新是否够快。

<a id="lesson-branches"></a>

## 10 · 向深度强化学习与持续控制的推广

| 改变的条件 | 新增问题 | 相应方法及边界 |
| --- | --- | --- |
| 状态很多或连续 | 不能逐状态存表，更新相互干扰 | DQN 等函数逼近控制；不再直接继承表格压缩与随机逼近证明 |
| 动作连续 | 每次备份的全局 argmax 难以求解 | 参数化 actor、采样优化或 MPC；局部优化不保证全局最优 |
| 部分可观测 | 当前观测不足以条件化未来 | 历史、belief state 或 agent state；记忆长度与更新方式成为控制的一部分 |
| 奖励或转移变化 | 过去样本与当前问题可能不一致 | 常数步长、变化跟踪、上下文建模和回放管理；动作重要性比率不能修复不同动力学 |
| 时间尺度很长 | 一步动作难以探索与传播信用 | 多步更新、option 与规划；随机时长需要正确折扣 |
| 持续任务以长期效率为目标 | 折扣起点价值未必是关心的量 | 平均奖励控制；需奖励率与差分价值方程，不能只把 γ 设为 1 |

预测知识与控制应分开分析。GVF 可以评价多种信号及假定行为，为状态构造、风险判断或规划提供输入；但学到更多预测不会自动产生更好的动作。仍需要明确哪个奖励定义任务、决策规则怎样利用这些预测、探索怎样取得必要的数据，以及在变化后哪些量需要重新学习。

持续控制还要问当前策略造成的访问分布是否允许恢复。如果智能体进入不可逆状态，或不再尝试已经改善的动作，即使预测更新完全正确，也可能无法恢复表现。固定经验流适合隔离学习器的适应性；闭环实验则同时评估数据获取、学习和决策，两者不能相互替代。

<a id="lesson-check"></a>

## 11 · 习题与诊断

问题一：策略评价损失下降是否意味着策略更优？不意味着。它可能只说明更准确地评价了一个差策略。控制必须比较改变后的行为价值；若只在训练访问分布上看损失，还可能遗漏策略将要访问的状态。

问题二：为什么最优 Bellman 备份不能对随机后果先取 max 再平均？动作在后果发生前选择。先对后果分别选择动作会引入环境未提供的信息，通常给出过高价值。

问题三：何时“贪心改善不会变差”可直接使用？固定 MDP、精确旧策略价值、每个状态满足单步改进不等式，且评价极限存在时。用有限样本估计、改变奖励，或把贪心动作投影到受限网络中，都需要重新检查条件。

问题四：Q-learning 是否可以无条件复用任意旧数据？不能。Off-policy 允许行为不同，前提仍是经验对应目标环境的转移与奖励。环境已变化时，旧数据来自另一问题；没有访问到的动作也不会因这个名称自动变得可估。

问题五：本例最优策略不终止，为什么仍能计算价值？折扣小于一且奖励有界。若把 $\gamma$ 改成 1，循环累计正奖励发散。平均奖励控制应比较单位时间奖励，并学习差分价值，不是对发散的折扣方程继续迭代。

## 本章的实验设计

先检查已知小型 MDP 中的策略排序。再比较交互学习。把动作覆盖不足与更新公式错误分开。

设定：在可枚举小 MDP 中，让后继采样动作与最大 Q 动作不同。先固定数据，随后使用相同交互预算的闭环学习。

- Sarsa 使用实际下一动作；Q-learning 使用最大动作值。
- DP 中的策略评估和改进各自可与枚举结果比较。
- ε-greedy 的探索、并列选择与合法动作规则明确。

对照：已知模型 DP 作为诊断；固定行为流隔离更新目标；相同探索制度的 Sarsa/Q-learning

记录：策略与解析最优策略的差距；原始交互收益和动作覆盖；更新次数与实际数据量

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-classic)

## 学习与研究衔接

评价固定策略与改进策略不是同一问题。状态、预测或技能的改进必须最终接受行为收益检验。

[分册导读](https://yingwen.io/zh/continual-rl/start/classic-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-control) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=control) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=control)

<a id="chapter-code"></a>

## 下载与运行

Python 3.10+，仅标准库。包含策略迭代、价值迭代、完整在线 Q-learning 与 12 个测试。确定性两状态教学环境；不作为复杂任务的性能证据。

[下载 control_problem_lab.py](https://yingwen.io/zh/continual-rl/download/control_problem_lab.py)

```sh
python3 control_problem_lab.py all
python3 control_problem_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：第 3–8 章建立 MDP、动态规划、MC、TD 与规划；第 9–13 章把这些更新扩展到函数逼近、资格迹和策略梯度。按本章的问题找相应章节，不必从头重读。

- [Sutton & Barto · Chapter 4: Dynamic Programming（CMU 课程镜像）](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)：4.1–4.4 的评价、改进、策略迭代与价值迭代；4.6 的 Generalized Policy Iteration 是本章的组织主线。第 3 章给出有限 MDP 与最优策略，第 6 章给出采样 TD 控制，第 13 章讨论策略梯度与 actor–critic。

- [Watkins & Dayan · Q-learning](https://doi.org/10.1007/BF00992698)：表格动作价值学习与收敛条件。逐对无限访问和步长条件与实际常数步长演示要分别理解。

- [Sutton et al. · Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)：策略梯度定理以及利用动作价值或优势辅助估计策略梯度；函数逼近的保证有相应条件。

- [van Hasselt · Double Q-learning](https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html)：从 max 与估计噪声的耦合出发，理解选择、评估为什么要拆开。
