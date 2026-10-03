# 时间信用分配：多步回报、资格迹与在线等价

结果到来时，怎样更新过去的预测、动作和记忆参数？哪些计算可以逐步完成，哪些等价关系需要冻结参数？

## 本章内容

- 区分时间信用、网络结构信用、流式约束与元学习。
- 推导 n-step、λ-return 和冻结参数的前向／后向等价，定位在线参数变化造成的差异。
- 实现 true-online TD(λ)，逐前缀核验荷兰迹与预测差修正。
- 说明 SARSA、Watkins、actor–critic 和 RTRL 如何处理不同的信用路径。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 价值预测

固定策略的价值 $v_\pi(s)$ 是未来回报的条件期望。网络 $v_w$ 只是估计。以下先讨论固定策略的预测问题。

### TD 误差

一步奖励与下一状态预测形成目标，再减去当前预测。两次预测使用同一组更新前参数。

$$
\delta_t=R_{t+1}+\gamma v_w(S_{t+1})-v_w(S_t)
$$

### 线性逼近

固定特征 $x_t$ 与权重 $w$ 的内积构成预测。对权重的梯度是 $x_t$；学习特征的情况需要额外求导。

$$
v_w(S_t)=w^\top x_t,\qquad\nabla_wv_w(S_t)=x_t
$$

### 终止与截断

真正终止的尾值为零。记录窗口结束不一定是终止；若任务仍继续，尾部应保留 bootstrap。

<a id="lesson-setting"></a>

## 1. 信用分配要回答的问题

智能体先记住一条提示，再经过多个状态，最后获得奖励。此时至少有三个学习问题。哪些过去的价值预测需要修正？哪些动作概率应该改变？哪些负责保存提示的参数应当更新？只说“把奖励反传回去”没有说明所用的目标、路径和近似。

| 层次 | 问题 | 典型计算 |
| --- | --- | --- |
| 时间信用 | 较晚的结果应当怎样影响较早的预测或动作？ | 多步回报、资格迹、动作优势估计。 |
| 结构信用 | 当前输出对哪些参数和内部计算敏感？ | 反向传播、链式法则、递归敏感度。 |
| 计算与数据约束 | 能保存多少历史？每次交互可以计算多久？ | rollout、重放、截断反传、固定内存的流式更新。 |
| 学习规则的适应 | 步长、迹长度或目标应当怎样随经验改变？ | IDBD、Metatrace、meta-gradient RL。 |

这些层次可以组合。资格迹是一种时间信用机制。它可以用于逐步更新，也可以用于离线等价分析。流式协议限制数据和计算，不定义哪段历史应该得到信用。元学习调整学习规则；即使它也使用“迹”，所追踪的可能是步长对未来权重的影响，而不是过去状态对当前 TD 更新的资格。

先固定行为策略，观察一条 episode：$(S_0,R_1,S_1,\ldots,R_T,S_T)$。取常数 $\gamma,\lambda\in[0,1]$。终点特征为 $x_T=0$。推导前半部分冻结参数 $w$；之后才允许每步更新。这样可以把目标的代数恒等式与在线算法分开。

这里的信用是一个指定学习目标下的更新分配，不是对行动因果责任的通用解释。TD 迹先解决预测更新。策略梯度还需说明动作怎样影响回报分布。两者不能只因都乘了奖励就被视作同一方法。

<a id="lesson-derive"></a>

## 2. n-step 与 λ-return 的前向视角

一步 TD 只使用下一条真实奖励，其余部分由价值估计补齐。$n$ 步目标使用更多真实奖励，再从第 $n$ 个后继状态 bootstrap。增加 $n$ 减少了对近处估计的直接依赖，也延迟了目标形成；在有噪声时，它通常改变偏差与方差。

$$
G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^n v_w(S_{t+n}),\qquad1\le n\le T-t
$$

本节所有 bootstrap 都用同一固定 $w$。当 $n=T-t$ 时，终点价值为零，目标等于完整 episode return。对于非终止的采样窗口，最后一项不能自动删掉。

不必只选一个 $n$。$\lambda$-return 对各个长度的目标作几何混合。有限 episode 的最后一个目标接收尚未分配的全部权重。这个尾项保证权重之和为一。

$$
G_t^\lambda=(1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1}G_t^{(n)}+\lambda^{T-t-1}G_t^{(T-t)}
$$

$\lambda=0$ 给出一步 TD 目标；$\lambda=1$ 给出完整 return。最后一步只有一个目标，其权重为一。$\gamma$ 决定任务中未来奖励的权重，$\lambda$ 决定学习时怎样混合不同 bootstrap 长度；二者职责不同。

$$
G_t^\lambda=R_{t+1}+\gamma\left[(1-\lambda)v_w(S_{t+1})+\lambda G_{t+1}^\lambda\right],\qquad G_T^\lambda=0
$$

把每个多步目标的第一条奖励分离后，剩下的部分仍是同类混合，所以可以后向递推。非终止窗口的边界应取其 bootstrap 值，而不取零。

完整前向目标需要未来数据。它适合解释学习方向，也适合在收集好的短轨迹上计算监督目标。若不能等待完整 episode，就需要有限窗口目标或后向机制。这个计算限制不改变“信用应分配给什么”的问题本身。

<a id="lesson-equivalence"></a>

## 3. 冻结参数时，资格迹来自交换求和

记固定预测为 $v_t=v_w(S_t)$。将前向递推减去 $v_t$，把一项 $\gamma v_{t+1}$ 加入并减去，便能用 TD 误差替代整段奖励。反复展开直到边界，就得到误差和。

$$
\begin{aligned}G_t^\lambda-v_t&=\delta_t+\gamma\lambda(G_{t+1}^\lambda-v_{t+1}),\\\delta_t&=R_{t+1}+\gamma v_{t+1}-v_t,\\G_t^\lambda-v_t&=\sum_{k=t}^{T-1}(\gamma\lambda)^{k-t}\delta_k.\end{aligned}
$$

终点的回报与预测均为零，所以最后没有残余项。若边界不是终止，只要前向目标与 TD 分解使用同一个固定尾值，这个有限窗口恒等式仍成立。

令 $g_t=\nabla_wv_w(S_t)$。对所有时刻的半梯度增量求和，且在整个求和期间不改变 $w$。交换“过去预测”和“后来误差”的求和次序，就得到后向资格迹。

$$
\begin{aligned}\Delta w&=\alpha\sum_{t=0}^{T-1}(G_t^\lambda-v_t)g_t\\&=\alpha\sum_{k=0}^{T-1}\delta_k\underbrace{\sum_{t=0}^{k}(\gamma\lambda)^{k-t}g_t}_{z_k},\\z_k&=\gamma\lambda z_{k-1}+g_k,\qquad z_{-1}=0.\end{aligned}
$$

这是固定参数下的总增量恒等式。前向计算使用未来误差修正早期预测；后向计算在误差到达时，将它乘以过去预测梯度的衰减和。对线性预测 $g_t=x_t$。

资格迹积累的是输出梯度，不是过去已经完成的损失梯度。当前 $\delta_k$ 到来后才乘整条迹。Momentum 则积累各时刻已经乘上各自误差的更新，两者时间结构不同。迹也不是样本缓冲区：它压缩了某种加权和，不能从中恢复任意旧转移。

前向 λ-return 与后向误差迹各自独立计算总增量，始终使用冻结权重。

```python
def lambda_returns(rewards, values, gamma, lam):
    """Finite-horizon forward targets; values[-1] is zero only at true terminal."""
    if len(values) != len(rewards) + 1 or not values:
        raise ValueError('T+1 values required')
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError('gamma and lambda must lie in [0,1]')
    targets = [0.0] * len(rewards)
    tail = values[-1]
    for t in range(len(rewards) - 1, -1, -1):
        tail = rewards[t] + gamma * ((1 - lam) * values[t + 1] + lam * tail)
        targets[t] = tail
    return targets


def frozen_forward_backward(features, rewards, weights, gamma, lam, alpha):
    """Two independently accumulated episode increments using ONE fixed w."""
    validate(features, rewards, weights, gamma, lam, alpha)
    values = [dot(weights, x) for x in features]
    targets = lambda_returns(rewards, values, gamma, lam)
    forward, backward, trace = [[0.0] * len(weights) for _ in range(3)]
    for t, x in enumerate(features[:-1]):
        for j in range(len(weights)):
            forward[j] += alpha * (targets[t] - values[t]) * x[j]
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        trace = [gamma * lam * z + xj for z, xj in zip(trace, x)]
        backward = [b + alpha * delta * z for b, z in zip(backward, trace)]
    return dict(targets=targets, forward=forward, backward=backward)
```

<a id="lesson-online"></a>

## 4. 传统在线 TD(λ) 为什么不再精确等价

实际 TD(λ) 通常每收到一次转移就改变权重。于是下一步预测已经不是冻结推导中的预测；非线性网络的梯度也会变化。继续使用同样的递推仍然定义一个合理算法，但不能继续引用冻结参数的总增量恒等式来证明它与原前向目标完全一致。

$$
\begin{aligned}\delta_t&=R_{t+1}+\gamma w_t^\top x_{t+1}-w_t^\top x_t,\\z_t&=\gamma\lambda z_{t-1}+x_t,\\w_{t+1}&=w_t+\alpha\delta_tz_t.\end{aligned}
$$

这里是固定线性特征的 accumulating TD(λ)。每步更新立即生效。小步长时与某些冻结近似的差别可能较小，但不能将近似写成任意步长下的精确等价。

若使用状态相关 continuation，必须区分到达当前状态的 $\gamma_t$ 与离开当前状态的 $\gamma_{t+1}$。前者衰减旧迹，后者构造 bootstrap。真正终止的 $\gamma_{t+1}=0$ 不意味着应当在使用终点奖励之前删除过去资格。

$$
z_t=\gamma_t\lambda z_{t-1}+g_t,\qquad\delta_t=R_{t+1}+\gamma_{t+1}v_{w_t}(S_{t+1})-v_{w_t}(S_t)
$$

配套代码先采用常数折扣，并将终点特征设为零。终止奖励完成更新后，新 episode 才重置资格迹。参数通常保留。单纯日志分段不提供清迹权限。

Replacing trace 是另一种选择。例如二值特征被重新激活时，将其迹置一，而不是再加一。它避免重复访问使该分量持续累加，但改变了更新定义。它也不等于下面的 Dutch trace；“都是资格迹”不意味着前向解释相同。

<a id="lesson-true-online"></a>

## 5. True-online TD(λ)：定义新的在线前向目标

精确在线等价需要先定义一个只使用已到达数据的前向算法。设当前只观察到时刻 $h$。对每个旧时刻 $t<h$，构造截至 $h$ 的 interim $\lambda$-return。第 $n$ 步 bootstrap 使用当时尚未处理该转移前的真实权重 $w_{t+n-1}$。这些权重不是后来重算的临时权重。

$$
\begin{aligned}G_t^{(n),\mathrm{on}}&=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^n w_{t+n-1}^\top x_{t+n},\\G_t^{\lambda\mid h}&=(1-\lambda)\sum_{n=1}^{h-t-1}\lambda^{n-1}G_t^{(n),\mathrm{on}}+\lambda^{h-t-1}G_t^{(h-t),\mathrm{on}}.\end{aligned}
$$

只有到达时刻 $h$ 之前的信息参与目标。若 $h$ 不是真实终点，最后目标仍有 bootstrap。新数据到来后，旧时刻的 interim 目标也随之变化。

字面实现每来一步，就从本 episode 的初始参数重新顺序执行全部旧状态的更新。它只使用已经到达的数据，但保存与重算成本很高。这个算法用于定义精确参考，不是推荐的流式实现。

$$
w_0^{[h]}=w_{\rm init},\qquad w_{t+1}^{[h]}=w_t^{[h]}+\alpha\left(G_t^{\lambda\mid h}-(w_t^{[h]})^\top x_t\right)x_t,\qquad w_h=w_h^{[h]}
$$

上标 $[h]$ 指当前重算的临时序列。必须从同一个 $w_{\rm init}$ 开始；不能在上一轮临时末值上再重复加整段更新。算法在每个观察前缀末端只保留对角结果 $w_h$。

直接重算所有 interim 目标的参考程序；用来逐前缀验证，不满足固定历史存储预算。

```python
def online_forward_reference(features, rewards, initial, gamma, lam, alpha):
    """Literal interim lambda-return algorithm; deliberately expensive.

    At each horizon h, recompute all interim targets and replay their sequential
    updates from initial. The n-step bootstrap uses the actual weights at time
    t+n-1, NOT the temporary helper weights used in this replay. This reference
    uses only the prefix observed so far, but stores and recomputes old data.
    """
    validate(features, rewards, initial, gamma, lam, alpha)
    history = [list(initial)]
    for horizon in range(1, len(rewards) + 1):
        helper = list(initial)
        for t in range(horizon):
            target, reward_sum = 0.0, 0.0
            remaining = horizon - t
            for n in range(1, remaining + 1):
                reward_sum += gamma ** (n - 1) * rewards[t + n - 1]
                bootstrap = dot(history[t + n - 1], features[t + n])
                n_step = reward_sum + gamma ** n * bootstrap
                mixture = lam ** (n - 1)
                if n < remaining:
                    mixture *= 1 - lam
                target += mixture * n_step
            error = target - dot(helper, features[t])
            helper = [w + alpha * error * x for w, x in zip(helper, features[t])]
        history.append(helper)
    return history
```

线性特征使一次更新关于旧权重呈仿射形式，其线性部分为 $A_t=I-\alpha x_tx_t^\top$。新数据对旧目标产生的修正，在到达当前权重前会经过这些更新矩阵。因此旧资格需要先乘 $A_t$，不能只乘标量衰减。

$$
\begin{aligned}z_t&=x_t+\gamma\lambda A_tz_{t-1}\\&=\gamma\lambda z_{t-1}+\left(1-\alpha\gamma\lambda z_{t-1}^\top x_t\right)x_t.\end{aligned}
$$

这就是荷兰迹。内积使用旧迹。减去的项处理先前更新与当前特征重叠造成的影响；它不是任意选择的稳定性裁剪。

迹修正还不够。记 $V=w_t^\top x_t$、$V'=w_t^\top x_{t+1}$，并保留上一转移计算的 $V_{\rm old}=w_{t-1}^\top x_t$。它是对当前状态的旧预测。新目标的变化涉及 $R+\gamma V'-V_{\rm old}=\delta+(V-V_{\rm old})$。加上旧更新已造成的预测变化补偿，得到完整更新。

$$
\begin{aligned}\delta&=R+\gamma V'-V,\\w^+&=w+\alpha(\delta+V-V_{\rm old})z-\alpha(V-V_{\rm old})x,\\V_{\rm old}&\leftarrow V'.\end{aligned}
$$

最后保存的是更新前计算的下一状态预测，不能重新用新权重计算。第一步可初始化旧预测为零，因为此时迹等于当前特征，两项预测差修正恰好相消。

**算法：固定特征、常数步长的 true-online TD(λ)；终点特征为零。**

1. 初始化 $w=w_{\rm init}$，$z=0$，$V_{\rm old}=0$。
1. 每条转移 $(x,R,x')$：
  1. 用旧权重计算 $V=w^\top x$、$V'=w^\top x'$。
  1. 计算 $\delta=R+\gamma V'-V$。
  1. 保存旧迹内积 $b=z^\top x$。
  1. 更新 $z\leftarrow\gamma\lambda z+(1-\alpha\gamma\lambda b)x$。
  1. 更新 $w\leftarrow w+\alpha(\delta+V-V_{\rm old})z-\alpha(V-V_{\rm old})x$。
  1. 保存 $V_{\rm old}\leftarrow V'$。
1. 真正 episode 结束后：保留权重，清零资格迹和旧预测。

在这组线性条件下，true-online TD(λ) 在每个前缀都与上述在线前向算法相等。这不同于冻结前向目标。对于 $d$ 维稠密特征，它每步使用 $O(d)$ 时间和额外迹存储；字面参考在第 $h$ 步使用 $O(h^2d)$ 时间与 $O(hd)$ 历史存储。非线性网络、在线变化的特征和任意优化器不能直接继承这一精确结论。

同一函数分别实现传统 accumulating TD 与荷兰迹加完整修正；返回每条转移后的权重。

```python
def td_episode(features, rewards, initial, gamma, lam, alpha, true_online=True):
    """Online linear TD(lambda), returning weights after EVERY transition.

    Terminal features must be zero. A nonzero last feature denotes a truncated
    continuing segment with its current value used as bootstrap. No artificial
    parameter reset occurs inside the segment.
    """
    validate(features, rewards, initial, gamma, lam, alpha)
    weights = list(initial)
    trace = [0.0] * len(weights)
    old_value = 0.0
    history = [list(weights)]
    for t, reward in enumerate(rewards):
        x, next_x = features[t], features[t + 1]
        value, next_value = dot(weights, x), dot(weights, next_x)
        delta = reward + gamma * next_value - value
        if true_online:
            # The overlap is computed from the OLD trace.
            overlap = dot(trace, x)
            trace = [gamma * lam * z + (1 - alpha * gamma * lam * overlap) * xj
                     for z, xj in zip(trace, x)]
            correction = value - old_value
            weights = [w + alpha * (delta + correction) * z - alpha * correction * xj
                       for w, z, xj in zip(weights, trace, x)]
        else:
            trace = [gamma * lam * z + xj for z, xj in zip(trace, x)]
            weights = [w + alpha * delta * z for w, z in zip(weights, trace)]
        old_value = next_value  # PRE-update prediction, not the new prediction.
        history.append(list(weights))
    return history
```

<a id="lesson-example"></a>

## 6. 两个手算例子

先看冻结计算。两步轨迹特征为 $(1,0)$、$(0,1)$、终点 $(0,0)$。奖励为 0、1。固定权重为 $(0.2,0.4)$，取 $\gamma=0.9$、$\lambda=0.8$、$\alpha=0.1$。末步目标为 1；首步目标为 $0.9[(1-0.8)0.4+0.8\times1]=0.792$。前向总增量为 $(0.0592,0.06)$。

$$
\begin{aligned}\delta_0&=0+0.9\times0.4-0.2=0.16,\qquad\delta_1=1-0.4=0.6,\\z_0&=(1,0),\qquad z_1=(0.72,1),\\\Delta w&=0.1[0.16(1,0)+0.6(0.72,1)]=(0.0592,0.06).\end{aligned}
$$

两种求和次序得到同一个增量。这里尚未在第一步真正改变权重。

再看重复特征的在线例子。$x_0=x_1=1$，$x_2=0$，奖励为 1、2，初始 $w=0$，取 $\gamma=\lambda=1$、$\alpha=0.5$。第一步两种在线算法都得到 $w=0.5$。第二步传统迹为 2，TD 误差为 1.5，所以最终 $w=0.5+0.5\times1.5\times2=2$。

True-online 第二步的荷兰迹为 $1+(1-0.5\times1)=1.5$。当前预测为 0.5，保存的旧预测为 0。完整更新为 $0.5+0.5(1.5+0.5)1.5-0.5\times0.5=1.75$。在线前向参考先用完整目标 3 把初值更新到 1.5，再用第二个目标 2 更新到 1.75。

| 算法或参考 | 第一步后 | 第二步后／总增量 | 为何不同 |
| --- | --- | --- | --- |
| 冻结前向与冻结后向 | 暂不应用增量 | 2.5 | 所有预测保持为零，总增量为 0.5×(3+2)。 |
| 传统在线 TD(1) | 0.5 | 2.0 | 权重改变，但仍使用 accumulating trace。 |
| 在线前向参考 | 0.5 | 1.75 | 每个前缀更新目标，并从初值顺序重算。 |
| True-online TD(1) | 0.5 | 1.75 | 用荷兰迹与修正高效实现同一在线参考。 |

这些数字区分的是算法定义。它们不能单独说明哪个算法在所有任务中学得更好。若只验证最终权重，也可能遗漏错误抵消；配套测试同时比较每个前缀，并覆盖非零初始化、稠密特征和非终止窗口。

<a id="lesson-control"></a>

## 7. 从预测到 SARSA 与 Watkins 控制

SARSA 把预测对象换为实际策略的动作价值。下一动作 $A_{t+1}$ 来自将要执行的行为策略，目标使用对应 $Q(S_{t+1},A_{t+1})$。表格 accumulating 迹为每个状态动作对保存一项资格。

$$
\begin{aligned}\delta_t^{\rm Sarsa}&=R_{t+1}+\gamma Q_t(S_{t+1},A_{t+1})-Q_t(S_t,A_t),\\z_t(s,a)&=\gamma\lambda z_{t-1}(s,a)+\mathbf1\{s=S_t,a=A_t\},\\Q_{t+1}(s,a)&=Q_t(s,a)+\alpha\delta_t^{\rm Sarsa}z_t(s,a).\end{aligned}
$$

策略随 Q 改变时，固定策略预测的前向解释成为局部解释。实际下一动作需要在明确的参数时序下选择。真正终止不选择下一动作，bootstrap 直接为零。

Q-learning 的一步目标却使用贪心延续 $\max_aQ(S_{t+1},a)$。若后续实际执行非贪心动作，沿这条轨迹继续累计奖励就不再直接对应贪心策略。Watkins Q(λ) 用剪迹处理这种不匹配：保留当前一步 max 目标更新，再阻止后续非贪心延续的误差继续传给更早的状态动作。

**算法：Watkins accumulating Q(λ)；剪迹发生在当前更新之后。**

1. 资格表 $z$ 保存已由上一转移衰减或剪断的旧资格。
1. 执行当前动作，获得奖励与下一状态。
1. 若非终止，用更新前 $Q$ 选择下一动作 $a'$。
1. 用同一旧 $Q$ 判断 $a'$ 是否达到下一状态的最大值；并列最大也算贪心。
1. 计算 $\delta=R+\gamma\max_bQ(s',b)-Q(s,a)$。
1. 令 $z(s,a)\leftarrow z(s,a)+1$。
1. 对全部状态动作对更新 $Q\leftarrow Q+\alpha\delta z$。
1. 若真正终止或下一动作非贪心，令 $z=0$；否则令 $z\leftarrow\gamma\lambda z$。
1. 执行已选定的下一动作。

例如当前 Q 为 0，下一状态两动作价值为 2 和 1，实际选择价值为 1 的动作，奖励为 0，$\gamma=0.9$、$\alpha=0.1$。SARSA 将当前值更新为 0.09，并保留衰减迹；Watkins 更新为 0.18，然后剪断旧迹。若在当前更新前清零，会连合法的一步信用也删掉。随机探索恰好选到并列最大动作时，不应仅因它来自探索分支就剪迹。

SARSA 与 Watkins 单步更新共用表格接口；下一动作贪心性在更新前判断。

```python
def control_trace_step(q, trace, state, action, reward, next_state, next_action,
                       gamma=0.9, lam=0.8, alpha=0.1, watkins=False, terminal=False):
    """Tabular accumulating Sarsa(lambda) or Watkins Q(lambda), one transition.

    Input trace is already decayed/cut by the preceding step. next_action must
    be selected using PRE-update q. For Watkins, determine whether it ties for
    the maximum before changing q. Cutting affects FUTURE credit only.
    """
    key = (state, action)
    if key not in q:
        raise ValueError('current state-action pair missing from q')
    greedy_next = False
    if terminal:
        target = reward
    else:
        next_values = [v for (s, _), v in q.items() if s == next_state]
        if not next_values or (next_state, next_action) not in q:
            raise ValueError('next state-action values required')
        greedy_next = q[(next_state, next_action)] == max(next_values)
        bootstrap = max(next_values) if watkins else q[(next_state, next_action)]
        target = reward + gamma * bootstrap
    delta = target - q[key]
    active = {pair: trace.get(pair, 0.0) for pair in q}
    active[key] += 1.0
    updated = {pair: value + alpha * delta * active[pair] for pair, value in q.items()}
    keep = not terminal and (not watkins or greedy_next)
    carry = {pair: gamma * lam * value if keep else 0.0 for pair, value in active.items()}
    return dict(q=updated, trace=carry, delta=delta, greedy_next=greedy_next)
```

频繁探索会使 Watkins 的有效信用范围很短。其他 off-policy 多步方法使用重要性比、截断比率或期望分支来处理策略差异。它们有不同目标与稳定性条件。不能只把 SARSA 的下一动作值改成 max，同时无条件保留所有旧迹，就称为 Watkins Q(λ)。

<a id="lesson-actor-recurrent"></a>

## 8. Actor–critic 与递归状态的接口

critic 的迹累积价值梯度。actor 的迹累积动作对数概率梯度，二者维度和含义可能不同。为明确折扣约定，先取 episodic 目标 $J=\mathbb E[\sum_t\gamma^tR_{t+1}]$，并在一条轨迹内固定策略参数。令 $\psi_t=\nabla_\theta\log\pi_\theta(A_t\mid S_t)$。

$$
\nabla_\theta J=\mathbb E\!\left[\sum_t\gamma^t\bigl(G_t-b(S_t)\bigr)\psi_t\right]
$$

动作无关的 baseline 不改变期望梯度。这个表达包含策略产生数据的概率路径；不是对环境奖励数值直接求动作导数。

以固定 critic 的 TD 误差和来构造优势估计，再交换求和，可以得到 actor 的后向形式。以下明确保留初始时刻目标中的 $\gamma^t$ 权重。若改成平均奖励或不同状态采样目标，应重新推导权重，而不是静默删掉它。

$$
\begin{aligned}\widehat A_t^\lambda&=\sum_{k=t}^{T-1}(\gamma\lambda_\theta)^{k-t}\delta_k,\\z_t^\theta&=\gamma\lambda_\theta z_{t-1}^\theta+\gamma^t\psi_t,\\\sum_t\gamma^t\widehat A_t^\lambda\psi_t&=\sum_t\delta_tz_t^\theta,\qquad\theta^+=\theta+\alpha_\theta\delta_tz_t^\theta.\end{aligned}
$$

最后一行的总和恒等式使用固定轨迹内的预测和梯度。逐步改变 actor 与 critic 后是在线算法近似。有限 episode 中 λ=1 给出 Monte Carlo 优势；λ<1 时近似 critic 的 bootstrap 误差会进入优势估计。

若输入是递归状态 $h_t=f_\theta(h_{t-1},u_t)$，当前输出的梯度还要经过状态构造。RTRL 保存 $E_t=\partial h_t/\partial\theta$。它回答旧输入通过递归计算怎样影响当前输出；资格迹则进一步把不同输出时刻的梯度按回报信用组合。两种记忆不能互相替代。

$$
\begin{aligned}E_t&=\frac{\partial f_\theta}{\partial h_{t-1}}E_{t-1}+\left.\frac{\partial f_\theta}{\partial\theta}\right|_{h_{t-1}},\\g_t&=E_t^\top\nabla_hv_\theta(h_t)+\left.\nabla_\theta v_\theta(h_t)\right|_{h_t},\\z_t&=\gamma\lambda z_{t-1}+g_t.\end{aligned}
$$

这些导数以固定参数的计算图为精确参照。训练中参数不断改变时，过去活动与梯度均有其生成时刻；TBPTT 的 detach 又会截断部分路径。不能因保存了长资格迹，就认定被截断的递归参数梯度已经恢复。

最小例子：$h_t=0.5h_{t-1}+bu_t$，$h_0=0$，输入依次为 1、0、0。最终 $h=0.25b$，所以 $\partial h/\partial b=0.25$。取 $b=0.2$、目标 1，平方损失梯度为 $(0.05-1)0.25=-0.2375$。若在第一次输入后 detach，保留的活动仍是 0.05，但后两步输入为零，截断梯度变成零。

固定参数的最小递归敏感度及 detach 对照；测试用中心有限差分核验完整梯度。

```python
def recurrent_sensitivity(inputs, b, target=1.0, decay=0.5, truncate_after=None):
    """Fixed-parameter h_t=decay*h_(t-1)+b*u_t, and exact dh_t/db.

    truncate_after cuts the derivative before that zero-based input index.
    The numerical hidden state is retained, demonstrating detach vs reset.
    """
    h, sensitivity = 0.0, 0.0
    for t, value in enumerate(inputs):
        if t == truncate_after:
            sensitivity = 0.0
        h = decay * h + b * value
        sensitivity = decay * sensitivity + value
    loss = 0.5 * (h - target) ** 2
    return dict(h=h, sensitivity=sensitivity, loss=loss,
                gradient=(h - target) * sensitivity)
```

<a id="lesson-code"></a>

## 9. 实验入口与原始实现

Python 3.10+，仅标准库。

```sh
python3 credit_assignment_lab.py all
python3 credit_assignment_lab.py frozen
python3 credit_assignment_lab.py online
python3 credit_assignment_lab.py control
python3 credit_assignment_lab.py recurrent
python3 credit_assignment_lab.py test
```

frozen 比较独立前向与后向总增量。online 将 true-online 与昂贵的在线前向参考逐前缀比较，同时保留传统 TD 的差异。control 检查下一动作值、并列贪心和剪迹时序。recurrent 检查活动不变但梯度路径被切断的情况。测试使用确定性轨迹和固定种子的稠密特征，不据此评价原论文的统计性能。

作者实验仓库的 totd.py 把步长 $\alpha$ 折进迹变量，并用 predprev 保存旧预测。配套代码使用迹外步长；在这里的常数步长条件下，二者通过 $z_{\rm author}=\alpha z$ 对应。比较源码时应先统一这个定义，再比较修正项。实验入口、随机 MDP 配置与结果处理位于同一作者仓库；完整论文实验需要其依赖和配置。

<a id="lesson-branches"></a>

## 10. 与流式学习和元学习的分工

| 研究问题 | 直接改变的对象 | 不能混同的对象 |
| --- | --- | --- |
| 奖励很晚，更新只到最近一步 | 回报长度、λ 混合、资格迹。 | 不是单纯增加网络深度。 |
| 记忆存在，但最初写入参数没梯度 | RTRL、BPTT、截断边界。 | 加长 TD 迹不能恢复已删掉的结构梯度。 |
| 同一轨迹上前后向更新不一致 | 参数冻结约定、在线目标、荷兰迹与修正。 | 不能用数值接近替代精确等价。 |
| 没有样本缓冲仍超出实时预算 | 迹存储、梯度计算、递归结构与数值尺度。 | 流式约束不等于某一种信用算法。 |
| 固定步长或 λ 在变化后失效 | 步长适应、meta-gradient、Metatrace。 | 元迹追踪学习参数影响，和价值资格迹分开存。 |
| off-policy 长迹方差或偏差很大 | 目标策略修正、截断、梯度 TD 或 emphatic 方法。 | 一条未经修正的 on-policy 迹不提供稳定性保证。 |

同一算法可以同时属于几条研究线。Metatrace 就为步长建立元时间信用；流式 actor–critic 可以同时使用价值迹、策略迹和递归敏感度。教材按这些对象分别讲解，是为了明确接口，不是把算法划进互不相交的名词类别。

<a id="lesson-check"></a>

## 11. 失败边界与思考题

| 观察到的错误 | 检查方法 |
| --- | --- |
| 终点奖励只更新最后状态 | 是否在使用该奖励前清空了旧迹。 |
| True-online 在第二步才出错 | 保存的是否是更新前 V′；内积是否使用旧迹；是否遗漏预测差修正。 |
| 随机动作导致所有迹都被切断 | 实际动作是否仍为并列最大；不要仅检查探索分支标志。 |
| 长迹在重复特征上造成大步更新 | 积累次数、特征尺度和 α；精确前向等价不是稳定性定理。 |
| 参数不更新时测试通过，训练中却失配 | 在线变参、特征漂移和策略采样变化是否超出了证明条件。 |
| Trace 很长却学不会远处提示 | 状态是否保存提示，以及递归梯度是否被 detach。 |

- 把手算的冻结参数例子改成 λ=0。答案：首步目标变成 0.36，首权重增量为 0.016；第二权重增量仍为 0.06。
- True-online 的 λ=0 为什么退化为普通 TD(0)？答案：迹等于当前特征，两个 V−Vold 修正相消，只剩 αδx。
- 能否只使用荷兰迹而删掉预测差修正？答案：不能；它一般不再实现这里定义的在线前向算法。
- Watkins 的 λ=1 是否必然使用整回合 Monte Carlo return？答案：不是；非贪心动作仍会剪断延续并使用 bootstrap。
- RTRL 敏感度与 TD 资格迹是否都是“历史压缩”，因而可以共用一块变量？答案：它们压缩不同导数对象，通常维度、递推和重置条件都不同。

一个可开始的研究是固定状态表示与环境，分别增加奖励延迟和重复特征比例。比较传统迹、true-online 与短窗口目标，同时记录预测误差、迹范数、每步计算和内存。第二阶段再引入表示学习，观察原先的精确等价如何失效。两个阶段分开，才能定位变化来自时间目标还是结构梯度。

## 本章的实验设计

先在同一条固定轨迹上比较前向与后向更新。再独立比较信用时域改变后的学习速度。

设定：用短线性轨迹放置延迟奖励，固定权重比较离线前向目标与后向累积和。再测试有限步长在线参数变化。

- λ=0 退化为相应单步更新。
- 真实终端奖励先沿已有迹传播，再按协议清理。
- 普通累积迹与 true-online 的等价对象和假设分开。

对照：相同轨迹、表征与步长；固定权重与在线变化两种条件；相同数据覆盖下改变 λ 或截断长度

记录：每个时间点的 trace、TD 误差和参数写入；独立参考式的数值差；延迟长度与采样预算下的学习曲线

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-estimator)

## 学习与研究衔接

资格迹将当前误差分配给过去的预测。元学习则学习如何更新参数。二者可组合，但估计对象不同。

[分册导读](https://yingwen.io/zh/continual-rl/start/classic-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-credit) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=credit) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=credit)

<a id="chapter-code"></a>

## 下载与运行

标准库：冻结前后向等价、传统与 true-online TD、在线前向参考、SARSA／Watkins 时序、最小递归敏感度。

[下载 credit_assignment_lab.py](https://yingwen.io/zh/continual-rl/download/credit_assignment_lab.py)

```sh
python3 credit_assignment_lab.py all
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton · Learning to Predict by the Methods of Temporal Differences · 1988](https://jmvidal.cse.sc.edu/library/sutton88a.pdf)：TD 与资格迹的原始研究，使用预测变化构造增量学习信号。

- [Sutton & Barto · Reinforcement Learning, Chapter 12](https://incompleteideas.net/book/the-book-2nd.html)：前向／后向视角、控制资格迹与 true-online 的教材讨论。

- [van Seijen & Sutton · True Online TD(λ) · ICML 2014](https://proceedings.mlr.press/v32/seijen14.html)：定义逐前缀在线前向参考，并推导精确等价的后向算法。

- [van Seijen 等 · True Online Temporal-Difference Learning · JMLR 2016](https://www.jmlr.org/papers/v17/15-599.html)：Algorithm 2 的迹外步长形式、线性等价分析，以及 TD／Sarsa 实验。

- [作者随机 MDP 实验 · totd-rndmdp-experiments](https://github.com/armahmood/totd-rndmdp-experiments)：论文作者 Rupam Mahmood 提供的 accumulating、replacing、true-online 对照与随机 MDP 实验。

- [作者 true-online 更新 · 固定版本 totd.py](https://github.com/armahmood/totd-rndmdp-experiments/blob/4b6d459890a0e9da898f6cf8b849ba7870ea044a/pysrc/algorithms/tdprediction/onpolicy/totd.py)：step 将 α 折入迹；predprev 与预测差修正对应原文更新。

- [Watkins · Learning from Delayed Rewards · 1989](https://www.repository.cam.ac.uk/items/f1ce2aa9-2e01-4a08-92b7-1eec76b652ed)：Q-learning 与延迟奖励控制的原始博士论文。

- [Sutton 等 · Policy Gradient Methods with Function Approximation · NIPS 1999](https://papers.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)：策略梯度、价值辅助与 actor–critic 的理论基础；需区分所选目标的状态权重。

- [Williams & Zipser · A Learning Algorithm for Continually Running Fully Recurrent Neural Networks · 1989](https://doi.org/10.1162/neco.1989.1.2.270)：递归敏感度的原始方法；用于理解网络内的历史梯度路径，而非替代回报资格迹。
