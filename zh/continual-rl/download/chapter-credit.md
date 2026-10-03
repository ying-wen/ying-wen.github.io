# 时间信用分配：从资格迹到深度梯度学习

结果到来时，怎样更新过去的预测、动作和记忆参数？策略变化、表示变化和部分可观测性会怎样改变信用与等价条件？

## 本章内容

- 区分时间信用、网络结构信用、流式约束与元学习。
- 推导 n-step、λ-return 和冻结参数的前向／后向等价，定位在线参数变化造成的差异。
- 实现 true-online TD(λ)，逐前缀核验荷兰迹与预测差修正。
- 用同一误差传播式比较 Tree-backup、Retrace，并区分 V-trace 的误差截断与传播截断。
- 推导状态相关 λ、Q(σ) 与期望资格迹，检查各自的目标、条件和反例。
- 从广义投影 Bellman 误差推导 GTD2／TDC／TDRC 的三条迹，核对非线性冻结总和等价。
- 连接 actor–critic、GAE 和 RTRL／RTU，区分回报信用、状态敏感度与参数变化造成的过时导数。

<a id="problem-definition"></a>

## 本章的问题定义

延迟反馈到来时，计算它应怎样改变早先预测、动作和记忆参数。时间传播与结构求导是两个耦合但不同的对象。

### 给定条件与符号

- 预测/控制的基础目标、轨迹和终止/窗口边界。
- 目标与行为策略、表示、迹参数、导数路径及可用内存。

### 需要求解的对象

与声明的前向目标、投影目标或策略目标相符的更新估计器，并说明在线变化和近似造成的差异。

### 信息与数据权限

起点 $S_t$ 的完整前向目标要等后续奖励到来；资格迹 $e_k$ 在时刻 $k$ 保存过去更新方向的统计量。它不是供行动回忆历史的agent state。

$$
\Delta w_{\rm forward}=\alpha\sum_{t=0}^{T-1}\bigl(G_t^\lambda-v_w(S_t)\bigr)\nabla_wv_w(S_t)
$$

$w$ 是本段冻结的预测参数，$T$ 是真实终点，$G_t^\lambda$ 是本章定义的多步混合目标，$\lambda$ 为混合权重，$\alpha$ 为步长。这是用于检验后向信用的理想总更新；在线true-online视图、离策略修正、投影Bellman梯度和actor敏感度各自使用对应对象，不全是此式。

### 成立条件与解的含义

- 冻结前后向总和需同一参数和边界；在线精确等价需true-online的线性条件与相应在线前向定义。
- 重要性比需要支持；传播截断、误差截断、状态相关迹以及递归敏感度均需各自计时和目标说明。

判断准则：在有限轨迹逐项核对冻结总和或每个在线前缀；检查单步、全回报、零比率和终止端点；策略/递归梯度以对应固定计算图有限差分检验。

### 适用边界

- 更新信用不等于通用因果责任归属。
- 资格迹、RTRL和元敏感度虽都递推，追踪对象并不相同。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)：同一固定策略预测对象可采用一步、多步或迹；信用机制改变如何利用反馈。

- 组合不同学习问题 · [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)：记忆参数的结构敏感度连接过去活动与当前输出，不由回报迹自动给出。

- 改变信息或数据协议 · [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)：后向迹可满足有限内存；能否重放、等待和展开仍由协议限定。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

前向目标依赖尚未发生的反馈；参数和行为持续变化又破坏简单冻结等价。

### 本章的核心思路

先写反馈传播的前向对象，再推出后向统计，并逐项修正在线参数变化、行为差异与结构导数。

1. [展开并抵消未来反馈项](#lesson-equivalence)：因为多步目标共享奖励和bootstrap，固定参数下将混合目标改写为TD误差加权和。

2. [补偿在线预测的变化](#lesson-true-online)：因为普通迹用不断变化的参数，true-online以Dutch trace和预测差修正对齐线性在线前向视图。

3. [处理不同目标行为的传播](#lesson-off-policy)：因为行为轨迹不等于目标轨迹，Tree-backup用动作期望分支，Retrace截断传播比率；有界传播本身不保证函数逼近稳定。

4. [对明确的投影目标求梯度](#lesson-gradient-traces)：因为半梯度自举未必对应稳定目标，辅助误差预测与多条迹分别估计广义投影目标中的条件均值和传播方向。

5. [接入actor和记忆敏感度](#lesson-actor-recurrent)：因为动作梯度与递归参数不是价值输出本身，分别用优势和结构敏感度连接相应目标。

结论与条件：冻结求和是代数恒等式；true-online精确等价限于相应线性视图。GTD/TDRC与离策略控制各有额外条件，不推广到任意深网或变化世界。

### 相关方法改变了什么

- 多步前向目标：等待有限窗口再计算，数据和延迟成本明确。

- 普通迹/true-online：前者低成本在线近似；后者修正线性在线前向视图。

- Retrace/Tree-backup/V-trace：行为纠偏与传播系数不同，误差截断可改变有效目标策略。

- GTD2/TDC/TDRC：以辅助误差预测和梯度迹连接指定投影目标，目标、曲率与正则路径需各自检查。


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

先固定行为策略，观察一条 episode：$(S_0,R_1,S_1,\ldots,R_T,S_T)$。取常数 $\gamma,\lambda\in[0,1]$。终点特征为 $x_T=0$。推导前半部分冻结参数 $w$；之后才允许每步更新。随后分别放松同策略、固定迹长度、已知 Markov 状态和线性表示这些条件。每次放松只引入一个新的问题。

Continuing task 指没有内在终点的持续交互任务；continual learning 指在长期经验中继续学习与适应。一个固定环境的 continuing task 未必包含分布变化；一串有限 episode 也可以构成持续学习。本文用“持续交互任务”指前者。日志窗口结束、环境终止和研究协议的任务切换需要分别记录。

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

<a id="lesson-off-policy"></a>

## 8. Tree-backup 与 Retrace：改变误差的传播系数

设数据由行为策略 $\mu$ 产生，要学习目标策略 $\pi$ 的动作价值。先冻结 $Q$ 和两种策略。一步误差用目标策略的动作期望；延长目标时，还要决定后续真实动作对更早预测有多大影响。记 $Q_t=Q(S_t,A_t)$、$\rho_t=\pi(A_t\mid S_t)/\mu(A_t\mid S_t)$。行为策略必须覆盖所需的目标动作；没有采到的动作不能靠截断比率补出来。

$$
\begin{aligned}\bar Q(S_{t+1})&=\sum_a\pi(a\mid S_{t+1})Q(S_{t+1},a),\\\delta_t^\pi&=R_{t+1}+\gamma\bar Q(S_{t+1})-Q_t,\\G_t^c-Q_t&=\sum_{k=t}^{T-1}\gamma^{k-t}\left(\prod_{i=t+1}^{k}c_i\right)\delta_k^\pi.\end{aligned}
$$

空乘积为一，所以当前一步误差不乘后续动作比率。终止时动作期望为零；非终止窗口保留尾部期望。这里的c是信用传播系数，不是折扣，也不是奖励。

这不是把未来奖励简单相加。每个 $\delta_k^\pi$ 已经减去实际动作的当前Q，并加入目标动作期望。传播系数再决定沿行为轨迹采到的后续误差可以传多远。交换求和后，旧预测梯度在到达当前动作时乘 $\gamma c_t$。

$$
G_t^c-Q_t=\delta_t^\pi+\gamma c_{t+1}(G_{t+1}^c-Q_{t+1}),\qquad z_t=\gamma c_tz_{t-1}+\nabla Q_t
$$

前向使用进入下一动作的c；后向使用进入当前动作的c。两个下标不能互换。对于冻结Q，误差和与梯度迹给出同一个总增量。逐步改变参数后的等价性另需证明。

| 方法 | 传播系数c | 效果与条件 |
| --- | --- | --- |
| 逐决策重要性采样 | $\lambda\rho_t$ | 策略差异被显式校正，但比率乘积可能迅速增大。 |
| Tree-backup(λ) | $\lambda\pi(A_t\mid S_t)$ | 期望分支不需要除以行为概率；即使完全同策略，随机动作概率仍会衰减旧信用。 |
| Retrace(λ) | $\lambda\min(1,\rho_t)$ | 近同策略时保留长信用；目标概率小于行为概率时减少传播。 |

例如 $\lambda=0.8$、$\pi=\mu$，采到的动作概率为0.1。Tree-backup的系数为0.08；Retrace为0.8。若 $\pi=0.8$、$\mu=0.01$，重要性采样系数变成64，Retrace仍为0.8。截断传播减少长比率乘积，没有删掉每条误差中的目标动作期望。

Munos等的Retrace原文证明了相应表格算子的收缩性质与条件下的控制收敛。这个“安全”有具体数学含义。Touati等随后给出了Tree-backup和Retrace在线性函数逼近中仍可发散的例子，并构造梯度版本。把c限制在一以内不能独自解决函数逼近、bootstrap和off-policy的联合不稳定。

固定误差的独立求和、Tree-backup／Retrace系数、on-policy Q(σ)与V-trace的有限轨迹计算。

```python
def error_kernel_targets(values, deltas, incoming, gamma):
    """Q targets from fixed errors and incoming c_t; c_0 is never used.

    G_t - Q_t = delta_t + gamma*c_(t+1)*(G_(t+1)-Q_(t+1)).
    This is an independently written finite-path calculation, not a learner.
    """
    n = len(deltas)
    if len(values) != n or len(incoming) != n:
        raise ValueError('equal-length values, errors, and incoming coefficients required')
    if not 0 <= gamma <= 1 or any(c < 0 or not math.isfinite(c) for c in incoming):
        raise ValueError('gamma in [0,1] and finite nonnegative coefficients required')
    targets, correction = [0.0] * n, 0.0
    for t in range(n - 1, -1, -1):
        carry = gamma * incoming[t + 1] if t + 1 < n else 0.0
        correction = deltas[t] + carry * correction
        targets[t] = values[t] + correction
    return targets


def offpolicy_trace_coefficients(target_probs, behavior_probs, lam, method):
    if len(target_probs) != len(behavior_probs) or not 0 <= lam <= 1:
        raise ValueError('matching action probabilities and lambda in [0,1] required')
    coefficients = []
    for pi, mu in zip(target_probs, behavior_probs):
        if not 0 <= pi <= 1 or not 0 < mu <= 1:
            raise ValueError('sampled actions require positive behavior probability')
        ratio = pi / mu
        if method == 'is':
            coefficient = lam * ratio
        elif method == 'tree':
            coefficient = lam * pi
        elif method == 'retrace':
            coefficient = lam * min(1.0, ratio)
        else:
            raise ValueError('method must be is, tree, or retrace')
        coefficients.append(coefficient)
    return coefficients


def q_sigma_targets(q_taken, expected_next, rewards, next_probs, gamma, lam, sigma):
    """Frozen ON-POLICY Q(sigma) with geometric lambda mixing.

    sigma controls sampled/expected actions; lambda controls return length.
    No off-policy Q(sigma) correction is implemented here.
    """
    n = len(rewards)
    if (len(q_taken) != n + 1 or len(expected_next) != n
            or len(next_probs) != n or not 0 <= sigma <= 1 or not 0 <= lam <= 1):
        raise ValueError('T+1 sampled Q and T expectations/probabilities required')
    deltas = [r + gamma * (sigma * q_taken[t + 1] + (1 - sigma) * expected_next[t])
              - q_taken[t] for t, r in enumerate(rewards)]
    incoming = [0.0] + [lam * (sigma + (1 - sigma) * next_probs[t])
                         for t in range(n - 1)]
    return error_kernel_targets(q_taken[:-1], deltas, incoming, gamma)


def vtrace_targets(rewards, values, discounts, ratios, rho_cap=1.0, c_cap=1.0):
    """Finite V-trace target and actor advantage, with explicit tail bootstrap.

    The continuation coefficient c_t is indexed at the CURRENT action, unlike
    the incoming coefficient c_(t+1) in the action-value kernel above.
    rho_cap=None leaves the error correction ratio unclipped.
    """
    n = len(rewards)
    if len(values) != n + 1 or len(discounts) != n or len(ratios) != n:
        raise ValueError('T+1 values and T rewards/discounts/ratios required')
    if c_cap <= 0 or (rho_cap is not None and rho_cap < c_cap):
        raise ValueError('0 < c_cap <= rho_cap required')
    if any(not 0 <= d <= 1 for d in discounts) or any(r < 0 for r in ratios):
        raise ValueError('discounts in [0,1] and nonnegative ratios required')
    clipped = [min(rho_cap, r) if rho_cap is not None else r for r in ratios]
    vs, correction = [0.0] * n, 0.0
    for t in range(n - 1, -1, -1):
        delta = rewards[t] + discounts[t] * values[t + 1] - values[t]
        correction = clipped[t] * delta + discounts[t] * min(c_cap, ratios[t]) * correction
        vs[t] = values[t] + correction
    next_targets = vs[1:] + [values[-1]]
    advantages = [rho * (r + d * tail - value)
                  for rho, r, d, tail, value in zip(clipped, rewards, discounts,
                                                    next_targets, values[:-1])]
    return dict(targets=vs, advantages=advantages)


def clipped_policy(target, behavior, rho_cap):
    """Tabular V-trace fixed-point policy, not the original target policy."""
    if len(target) != len(behavior) or not target or rho_cap <= 0:
        raise ValueError('matching nonempty distributions and positive rho cap required')
    if (any(p < 0 for p in target) or any(p <= 0 for p in behavior)
            or abs(sum(target) - 1) > 1e-10 or abs(sum(behavior) - 1) > 1e-10):
        raise ValueError('normalized distributions with behavior support required')
    masses = [min(pi, rho_cap * mu) for pi, mu in zip(target, behavior)]
    return [mass / sum(masses) for mass in masses]
```

<a id="lesson-vtrace-sigma"></a>

## 9. 两种不同的选择：Q(σ)的动作期望与V-trace的策略校正

$\lambda$ 混合回报长度，$\sigma$ 混合后继动作的采样与期望。先取on-policy、冻结Q的情形。记 $\bar Q_{t+1}$ 为目标策略动作期望。Q(σ)的一步误差和多步传播同时改变，而不是只修改其中一个。

$$
\begin{aligned}\delta_t^\sigma&=R_{t+1}+\gamma\left[\sigma_{t+1}Q_{t+1}+(1-\sigma_{t+1})\bar Q_{t+1}\right]-Q_t,\\c_i^\sigma&=\lambda\left[\sigma_i+(1-\sigma_i)\pi(A_i\mid S_i)\right],\\G_t^{\sigma,\lambda}-Q_t&=\sum_{k=t}^{T-1}\gamma^{k-t}\left(\prod_{i=t+1}^{k}c_i^\sigma\right)\delta_k^\sigma.\end{aligned}
$$

这是在原n-step Q(σ)传播式上作几何λ混合的冻结on-policy表达。σ=1得到Sarsa的采样延续；σ=0得到Tree-backup的期望分支。λ=0只保留一步，但σ仍决定一步目标。

De Asis等原文还给出了off-policy Q(σ)的修正。上式和本章代码只实现on-policy端点；不能在行为与目标策略不同时直接使用它。做比较时至少固定回报长度、σ、λ、策略匹配程度四项。改变σ通常是在改变采样噪声，改变λ还会改变bootstrap跨度。

V-trace解决另一个问题。分布式actor用较旧策略收集轨迹，learner用新策略训练状态价值。每条TD误差需要当前动作的校正；误差继续向前面的状态传播时还需要另一个系数。两类截断分工不同。

$$
\begin{aligned}\bar\rho_t&=\min(\rho_{\max},\rho_t),\qquad c_t=\min(c_{\max},\rho_t),\\\delta_t^V&=\bar\rho_t\left[R_{t+1}+\gamma_{t+1}V_{t+1}-V_t\right],\\v_t^{\rm VT}-V_t&=\delta_t^V+\gamma_{t+1}c_t\left(v_{t+1}^{\rm VT}-V_{t+1}\right).\end{aligned}
$$

取0<cmax≤ρmax；窗口边界取vVT=V。这里的 $c_t$ 乘当前动作比率，与上一节动作价值形式中的 $c_{t+1}$ 不同。原IMPALA常取两个上限为一。

在表格评价条件下，$\rho_{\max}$ 会改变V-trace固定点对应的策略，而 $c_{\max}$ 主要影响传播长度与收敛速度。固定点策略为：

$$
\pi_{\rho_{\max}}(a\mid s)=\frac{\min\{\rho_{\max}\mu(a\mid s),\pi(a\mid s)\}}{\sum_b\min\{\rho_{\max}\mu(b\mid s),\pi(b\mid s)\}}
$$

ρmax足够大时恢复目标策略；截断较强时可能得到介于行为与目标之间的策略。不能把两个截断都称为“只降低方差而不改变目标”。

单步bandit就能看出目标变化。取 $\mu=(0.8,0.2)$、$\pi=(0.2,0.8)$，两动作奖励为0、1。$\rho_{\max}=1$ 时截断质量为 $(0.2,0.2)$，归一化策略为 $(0.5,0.5)$，价值固定点为0.5；原目标价值为0.8。因为没有后继步，改变c根本不能改变这个例子的固定点。

$$
\widehat A_t^{\rm VT}=\bar\rho_t\left[R_{t+1}+\gamma_{t+1}v_{t+1}^{\rm VT}-V_t\right]
$$

actor使用下一状态的V-trace目标构造动作回报。它不等于直接使用$v_t^{VT}-V_t$。实践中策略梯度比率上限还可以单独设置；本文小程序令它与ρmax一致。

IMPALA作者代码先计算行为与目标动作log-prob的差，再用反向scan形成窗口目标，最后stop_gradient。它保存短轨迹并进行批量训练。反向scan在数组上从后向前计算，不等于严格流式后向资格迹；前者依赖已保存的窗口，后者在新误差到达时更新一条压缩迹。

<a id="lesson-state-lambda"></a>

## 10. 状态相关与自适应λ：先写清目标，再决定长度

固定λ把每一处bootstrap都视为同样可靠。状态相关 $\lambda(S_t)$ 可以让可信状态提前截断，让预测偏差较大的状态依赖更多真实奖励。先冻结预测和λ函数，并采用到达状态的下标。

$$
\begin{aligned}G_t^\lambda&=R_{t+1}+\gamma_{t+1}\left[(1-\lambda_{t+1})V_{t+1}+\lambda_{t+1}G_{t+1}^\lambda\right],\\G_t^\lambda-V_t&=\sum_{k=t}^{T-1}\left(\prod_{i=t+1}^{k}\gamma_i\lambda_i\right)\delta_k,\\z_t&=\gamma_t\lambda_tz_{t-1}+\nabla V_t.\end{aligned}
$$

到达当前状态的γtλt衰减旧资格；离开当前状态的γt+1λt+1形成前向延续。终止折扣为零会删除尾值，但终止奖励仍要乘此前已经存在的资格。

在可精确表示的表格问题中，不同合法λ可以共享真实价值固定点。投影和函数逼近介入后，λ会影响逼近解，因而不能只把它解释为优化速度参数。在线适应λ还会使目标随学习改变；在旧状态分布上估计得很好的长度函数，在变化后的分布上可能已经过时。

White与White的λ-greedy从一个局部选择出发：下一状态的当前预测有平方偏差 $b^2$，完整未来回报方差为 $s_G^2$。把下一处bootstrap与未来Monte Carlo回报混合；在固定预测、相应条件独立约定下，需要最小化的λ相关项是：

$$
L(\lambda)=(1-\lambda)^2b^2+\lambda^2s_G^2,\qquad\frac{dL}{d\lambda}=-2(1-\lambda)b^2+2\lambda s_G^2,\qquad\lambda^*=\frac{b^2}{b^2+s_G^2}
$$

这是单处混合的局部最优，不是联合优化整条轨迹的证明。两项都为零时任何λ等价；配套程序选择零作为约定。

若 $b^2=4$、$s_G^2=1$，最优λ为0.8，局部目标为0.8；λ=0和λ=1的目标分别为4和1。实际算法没有真值，需要额外预测未来回报均值与二阶矩，再估计偏差和方差。本章小实验只核对已知统计量下的最小化，不声称复现完整λ-greedy。噪声、混叠或非平稳性会使这些统计估计本身成为学习问题。

Meta-gradient RL采用另一条路线：一次参数更新依赖λ，再用后来经验评价更新后的参数，并对这条学习路径求导。元目标、未来数据与导数截断必须明确。它与直接减少同一批次上的λ-return拟合误差不同；后者可能把λ调到更容易拟合但更偏的目标。

变量折扣／λ的前后向数值恒等式；局部bias–variance目标的已知统计量最优点。

```python
def variable_lambda_equivalence(features, rewards, weights, discounts, lambdas):
    """Frozen forward/backward identity with arrival-indexed gamma_t/lambda_t."""
    n = len(rewards)
    if len(discounts) != n + 1 or len(lambdas) != n + 1:
        raise ValueError('T+1 arrival-indexed discounts and lambdas required')
    validate(features, rewards, weights, 1.0, 1.0)
    if any(not 0 <= v <= 1 for v in [*discounts, *lambdas]):
        raise ValueError('discounts and lambdas must lie in [0,1]')
    values = [dot(weights, x) for x in features]
    targets, tail = [0.0] * n, values[-1]
    for t in range(n - 1, -1, -1):
        tail = rewards[t] + discounts[t + 1] * (
            (1 - lambdas[t + 1]) * values[t + 1] + lambdas[t + 1] * tail)
        targets[t] = tail
    forward, backward, trace = [[0.0] * len(weights) for _ in range(3)]
    for t in range(n):
        delta = rewards[t] + discounts[t + 1] * values[t + 1] - values[t]
        trace = [discounts[t] * lambdas[t] * z + x for z, x in zip(trace, features[t])]
        forward = [a + (targets[t] - values[t]) * x for a, x in zip(forward, features[t])]
        backward = [a + delta * z for a, z in zip(backward, trace)]
    return dict(targets=targets, forward=forward, backward=backward)


def greedy_lambda(bias_squared, return_variance):
    """Oracle optimum of the LOCAL bias/variance surrogate, not full lambda-greedy.

    White & White's algorithm must also learn return moments online. We supply
    those statistics as inputs here so the closed-form optimization is testable.
    """
    if any(not math.isfinite(v) or v < 0 for v in (bias_squared, return_variance)):
        raise ValueError('finite nonnegative squared bias and variance required')
    total = bias_squared + return_variance
    return bias_squared / total if total else 0.0
```

<a id="lesson-expected-traces"></a>

## 11. Expected Eligibility Traces：信用能否复用于另一条过去路径？

普通迹只包含本次走过的状态。设两条路径在同一个状态汇合，后面的随机奖励与此前走哪条路径无关。当前奖励可以同时更新那些可能到达汇合状态的路径。期望资格迹学习的是“到达此状态时，过去资格通常是什么”，不是预测未来奖励的successor feature。

$$
\bar z(s)=\mathbb E[z_t\mid S_t=s],\qquad\Delta w=\alpha\delta_t\bar z(S_t),\qquad\min_\eta\ \frac12\|z_t-\bar z_\eta(S_t)\|^2
$$

保留普通迹作为训练标签，再用状态条件预测器估计它。η在这一式中仅表示预测器参数；下面的混合系数另记ξ，以免两种用途混淆。

van Hasselt等的均值与方差分析依赖 Markov 状态、固定预测参数与固定 Markov 策略 $\pi(a\mid s)$。当前 TD 误差的条件分布不能再依赖未纳入状态的历史。此时，给定完整状态，当前转移产生的 $\delta_t$ 与到达它之前产生的 $z_t$ 条件独立。因此：

$$
\mathbb E[\delta_tz_t\mid s]=\mathbb E[\delta_t\mid s]\mathbb E[z_t\mid s]=\mathbb E[\delta_t\bar z(s)\mid s]
$$

替换为精确条件均值后，保留条件平均更新，并消去历史路径随机性的一部分。方差不增是逐分量陈述；任意近似神经预测器和漂移参数不能自动继承无偏性。

最小数值例子：两条等概率历史的资格为 $(0.72,0,1)$ 和 $(0,0.72,1)$，因此期望资格为 $(0.36,0.36,1)$。汇合后的TD误差独立地取0或2。两种更新均值都为 $(0.36,0.36,1)$；前两分量的方差由0.3888降为0.1296。最后分量只受奖励噪声影响，方差仍为1。

反例同样重要。若表面观察相同，但走第一条历史时奖励必为2、走第二条必为0，那么普通迹平均更新是 $(0.72,0,1)$，期望迹变成 $(0.36,0.36,1)$。隐藏历史决定未来奖励，条件独立不成立，期望迹把信用分给了错误的路径。改善agent state可以恢复条件，单纯加长或平均迹不能解决状态混叠。

$$
y_t=(1-\xi)\bar z_\eta(S_t)+\xi\left(\gamma_t\lambda y_{t-1}+\nabla V_t\right),\qquad\xi\in[0,1]
$$

原文ET(λ,η)用η表示这处混合系数。这里改记ξ：ξ=1恢复普通迹，ξ=0使用预测的期望迹。它是递归混合，不是将两条独立算好的完整迹一次凸组合。

实现时需要选择预测器的状态输入、资格标签、更新时序与统计遗忘速度。全参数迹预测器的输出维度和主网络参数数目相同；它可能非常昂贵。表示学习还会改变资格所在的参数坐标，旧迹标签可能失效。应先在固定特征、路径汇合的小问题验证条件，再研究低维近似和状态变化。

精确枚举四种路径／奖励组合，不依赖随机采样；另保留状态混叠导致均值改变的反例。

```python
def expected_trace_enumeration(hidden_history=False):
    """Exact finite enumeration at a merging state, including an aliasing failure.

    Two equiprobable histories have traces (.72,0,1) and (0,.72,1).
    Markov case: reward in {0,2} is independent of the incoming history.
    Aliased case: the hidden incoming history determines reward 2 vs 0.
    Predictions and features are frozen; this is not an ET training benchmark.
    """
    traces = [[0.72, 0.0, 1.0], [0.0, 0.72, 1.0]]
    expected_trace = [sum(z[j] for z in traces) / 2 for j in range(3)]
    cases = [(0.5, 2.0, traces[0]), (0.5, 0.0, traces[1])] if hidden_history else [
        (0.25, reward, trace) for trace in traces for reward in (0.0, 2.0)]
    result = {}
    for name, use_expected in [('instantaneous', False), ('expected', True)]:
        updates = [(prob, [reward * z for z in (expected_trace if use_expected else trace)])
                   for prob, reward, trace in cases]
        mean = [sum(prob * row[j] for prob, row in updates) for j in range(3)]
        variance = [sum(prob * (row[j] - mean[j]) ** 2 for prob, row in updates)
                    for j in range(3)]
        result[name] = dict(mean=mean, variance=variance)
    return result
```

<a id="lesson-gradient-traces"></a>

## 12. 深度梯度资格迹：目标、辅助预测与三条递推

半梯度TD把bootstrap目标当常数，用 $\delta_tz_t$ 更新预测。非线性、off-policy和bootstrap组合时，这个方向不一定来自一个稳定的整体目标。Elelimy等的2025年工作先指定广义投影Bellman误差，再推导多步梯度算法。下面先取固定策略、on-policy、固定 $\gamma,\lambda$ 的情形；这是能够逐项检验的起点。

记 $\epsilon_t^\lambda=G_t^\lambda-V_w(S_t)$，其条件均值为 $\bar\epsilon_w^\lambda(s)$。辅助函数 $H_\eta(s)$ 估计这个条件平均误差。它既不是第二个价值目标网络，也不是任意保存梯度的变量。平方有共轭形式 $u^2=\max_h(2uh-h^2)$，于是限制辅助函数类后得到：

$$
\mathcal E_\lambda(w)=\max_{H\in\mathcal H}\ \mathbb E_{s\sim d}\left[2\bar\epsilon_w^\lambda(s)H(s)-H(s)^2\right]
$$

辅助函数类如果包含所有状态函数，会恢复均方条件Bellman误差；受限函数类定义相应的广义投影目标。状态分布d、策略、λ和函数类都是目标的一部分。

为什么不直接最小化一次采样TD误差的平方？条件误差平方的梯度涉及两个条件期望的乘积。用同一随机下一状态替代两个独立样本，通常得到额外协方差项。辅助预测把一个条件期望变成可学习的函数，避免要求环境从同一状态再独立采一次。它仍带来辅助估计误差与更新速度的选择。

将目标按二分之一缩放，用 $H_t=H_\eta(S_t)$。GTD2的前向主参数方向为 $-H_t\nabla_w\epsilon_t^\lambda$，辅助参数方向为 $(\epsilon_t^\lambda-H_t)\nabla_\eta H_t$。前者对包含bootstrap的误差求导；后者让辅助预测拟合多步误差。

$$
\begin{aligned}\nabla_w\delta_t&=\gamma\nabla_wV_w(S_{t+1})-\nabla_wV_w(S_t),\\z_t^H&=\gamma\lambda z_{t-1}^H+H_t,\\z_t^\eta&=\gamma\lambda z_{t-1}^\eta+\nabla_\eta H_t,\\\Delta w_t^{\rm GTD2}&=-z_t^H\nabla_w\delta_t,\\\Delta\eta_t&=\delta_tz_t^\eta-H_t\nabla_\eta H_t.\end{aligned}
$$

两条迹承担不同角色：标量迹累积辅助误差预测，参数迹累积辅助网络的输出梯度。主网络方向仍对当前一步误差完整求导；不能对下一状态预测stop-gradient后宣称实现同一GTD2。

推导后向形式仍然靠交换求和。固定 $w,\eta$ 后，$\nabla\epsilon_t^\lambda=\sum_{k\ge t}(\gamma\lambda)^{k-t}\nabla\delta_k$。有限窗口在 $T$ 处使用同一尾值 $G_T=V_w(S_T)$，并对这个尾值求导，因此 $\epsilon_T=0$ 且 $\nabla_w\epsilon_T=0$；真实终止则将尾值及其导数设为零。所有旧时刻的 $H_t$ 在当前误差梯度之前合并成标量迹；辅助网络的梯度同理合并。TDC在此基础上加入 $(\epsilon_t^\lambda-H_t)\nabla V_t$ 修正，后向形式还需第三条价值梯度迹。

$$
\begin{aligned}z_t^w&=\gamma\lambda z_{t-1}^w+\nabla_wV_t,\\\Delta w_t^{\rm TDC}&=\delta_tz_t^w-H_t\nabla_wV_t-z_t^H\nabla_w\delta_t,\\\Delta\eta_t^{\rm TDRC}&=\delta_tz_t^\eta-H_t\nabla_\eta H_t-\beta\eta_t.\end{aligned}
$$

TDC 使用同一辅助更新；TDRC 进一步加入辅助参数正则。主网络不能只保留 $\delta_tz_t^w$ 而删掉后两项。TDC 的校正依赖辅助估计，非线性情况下没有自动的全局收敛保证。

**算法：两个网络的更新方向先共同算完，避免第二个方向意外使用第一个网络的新参数。**

1. 收到转移后，先用更新前的主网络与辅助网络：
  1. 计算 $V_t,V_{t+1},H_t,\delta_t$，以及 $\nabla_wV_t,\nabla_w\delta_t,\nabla_\eta H_t$。
  1. 更新标量迹 $z_t^H$、辅助梯度迹 $z_t^\eta$、主梯度迹 $z_t^w$。
  1. 从同一组旧参数计算两个参数方向。
  1. 选择 GTD2／TDC／TDRC 规则，再分别应用主、辅助步长。
  1. 若真实终止，先完成奖励信用，再按算法边界清迹。
1. 控制版本还必须指定目标动作、策略不一致处理和剪迹时序。

原文Theorem 6.1明确假设两个参数集合在episode内不变，证明的是总增量相同。它不等于线性true-online的每前缀等价。本文用tanh主预测和tanh辅助预测独立计算前向目标及其导数，再与三条迹比较。GTD2还用中心有限差分核对双网络鞍点方向。数值结果检验代数与实现，不检验深度控制性能。

作者QRC实现把主预测换成动作价值，完整求导max-bootstrap误差，并保存`h_trace`、`grad_h_trace`、`grad_q_trace`。配置中的`gradient_correction` 和 `reg_coeff`分别控制梯度校正与辅助正则。代码在完成当前更新后，对终止、环境截断或指定的非贪心事件清迹；环境截断仍可保留bootstrap。这是该实验实现的协议，不能推广为所有日志分段都需要清迹。

论文的MuJoCo前向方法使用保存的窗口与PPO框架；MinAtar后向方法逐步更新。两者不能直接当作相同存储协议。MinAtar的QRC与StreamQ对照还涉及JAX／PyTorch实现差异，作者明确说明每秒步数比较有框架混杂。研究时应分别检查学习曲线、数据复用、参数量、辅助计算和更新时间。

冻结非线性GTD2／TDC／TDRC总增量检查，及主／辅助参数有限差分；不是作者基准复现。

```python
def gradient_trace_equivalence(features, rewards, weights, auxiliary, gamma, lam,
                               correction=False, regularization=0.0):
    """Frozen nonlinear GTD2/TDC/TDRC total-increment identities, on-policy.

    V=tanh(w dot x), H=tanh(theta dot x). Independently differentiate the
    recursive lambda target for the forward view, then accumulate the backward
    scalar-H trace and parameter traces. No parameters change during this call.
    The saddle objective is half-scaled: sum(H*delta_lambda - H**2/2).
    This is a mathematical unit experiment, not the authors' deep RL benchmark.
    """
    validate(features, rewards, weights, gamma, lam)
    if len(auxiliary) != len(weights) or regularization < 0:
        raise ValueError('matching auxiliary dimension and nonnegative penalty required')
    n, d = len(rewards), len(weights)
    value_rows = [nonlinear_value_gradient(weights, x) for x in features]
    values, gradients = zip(*value_rows)
    h_rows = [nonlinear_value_gradient(auxiliary, x) for x in features[:-1]]
    targets, target_gradients = [0.0] * n, [[0.0] * d for _ in range(n)]
    tail, tail_gradient = values[-1], list(gradients[-1])
    for t in range(n - 1, -1, -1):
        tail = rewards[t] + gamma * ((1 - lam) * values[t + 1] + lam * tail)
        tail_gradient = [gamma * ((1 - lam) * g + lam * old)
                         for g, old in zip(gradients[t + 1], tail_gradient)]
        targets[t], target_gradients[t] = tail, list(tail_gradient)
    fw, fh, bw, bh, trace_v, trace_h = [[0.0] * d for _ in range(6)]
    scalar_trace, objective = 0.0, 0.0
    for t, (h, grad_h) in enumerate(h_rows):
        grad_v = gradients[t]
        error = targets[t] - values[t]
        grad_error = [g - v for g, v in zip(target_gradients[t], grad_v)]
        objective += h * error - 0.5 * h * h - 0.5 * regularization * dot(auxiliary, auxiliary)
        fw = [a - h * ge + ((error - h) * gv if correction else 0.0)
              for a, ge, gv in zip(fw, grad_error, grad_v)]
        fh = [a + (error - h) * gh - regularization * theta
              for a, gh, theta in zip(fh, grad_h, auxiliary)]
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        grad_delta = [gamma * gn - gv for gn, gv in zip(gradients[t + 1], grad_v)]
        scalar_trace = gamma * lam * scalar_trace + h
        trace_h = [gamma * lam * z + g for z, g in zip(trace_h, grad_h)]
        trace_v = [gamma * lam * z + g for z, g in zip(trace_v, grad_v)]
        bw = [a - scalar_trace * gd + (delta * zv - h * gv if correction else 0.0)
              for a, gd, zv, gv in zip(bw, grad_delta, trace_v, grad_v)]
        bh = [a + delta * zh - h * gh - regularization * theta
              for a, zh, gh, theta in zip(bh, trace_h, grad_h, auxiliary)]
    return dict(forward_w=fw, backward_w=bw, forward_h=fh, backward_h=bh,
                objective=objective)
```

<a id="lesson-actor-recurrent"></a>

## 13. 深度actor–critic、GAE与RTRL／RTU的接口

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

这组优势误差和就是GAE的基本计算。批量actor–critic通常先保存窗口，以冻结预测反向计算优势，再做多轮更新；流式actor–critic在误差到来时使用过去的动作梯度迹。相同误差和不意味着相同学习协议。窗口bootstrap、策略更新次数、概率比率与目标的状态权重都会改变实际方向。

固定score与critic的折扣初始状态目标：独立前向优势与后向actor迹总和。

```python
def actor_trace_equivalence(scores, rewards, values, gamma, lam):
    """Frozen discounted-start-state policy scores; actor is not updated here."""
    if len(scores) != len(rewards) or not scores or len(values) != len(rewards) + 1:
        raise ValueError('T score vectors/rewards and T+1 values required')
    d = len(scores[0])
    if not d or any(len(score) != d for score in scores):
        raise ValueError('score dimensions must match')
    advantages = [g - v for g, v in zip(lambda_returns(rewards, values, gamma, lam), values)]
    forward, backward, trace = [[0.0] * d for _ in range(3)]
    for t, score in enumerate(scores):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        trace = [gamma * lam * z + gamma ** t * s for z, s in zip(trace, score)]
        forward = [a + gamma ** t * advantages[t] * s for a, s in zip(forward, score)]
        backward = [a + delta * z for a, z in zip(backward, trace)]
    return dict(forward=forward, backward=backward, advantages=advantages)
```

深度实现每步先计算主网络当前输出的梯度，再将数值梯度加入独立的资格缓冲。普通半梯度迹不保留历史自动求导图；否则内存会随时间增长。共享actor／critic主干时，两个目标的方向要明确组合。对过去各条梯度先分别做Adam变换，再累加，与先累加资格方向后交给Adam通常不同；优化器选择属于算法定义。

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

RTU的“trace”指递归状态对参数的敏感度。两维实值旋转块实现复值对角递归，使特定结构下的RTRL更便宜。作者实现将状态活动与grad_memory共同递推，并用自定义求导接到当前输出。这个敏感度先给出价值或动作score的梯度，回报资格迹再把不同时刻的输出梯度组合。两块变量的维度、用途与重置条件都需分别规定。

2026年Farr等把RTU-RTRL接入QRC与流式actor–critic，并单独研究参数不断改变时的敏感度过时问题。固定参数下的精确RTRL不意味着沿更新中的参数轨迹仍等于“把全部历史重新用当前参数运行一次”的导数。原文用保存轨迹的重算参考做诊断；这项额外存储应计为评价成本。Masked MuJoCo结果仍低于批量PPO，因此它支持特定部分可观测设置下的流式可行性，尚不支持全面性能优势。

<a id="lesson-code"></a>

## 14. 实验入口与原始实现

Python 3.10+，仅标准库；每个入口对应正文中的一个可检查问题。

```sh
python3 credit_assignment_lab.py all
python3 credit_assignment_lab.py online
python3 credit_assignment_lab.py offpolicy
python3 credit_assignment_lab.py adaptive
python3 credit_assignment_lab.py expected
python3 credit_assignment_lab.py gradient
python3 credit_assignment_lab.py actor
python3 credit_assignment_lab.py recurrent
python3 credit_assignment_lab.py test
```

| 入口 | 核验对象 | 不据此声称 |
| --- | --- | --- |
| frozen / online | 冻结总和恒等式；true-online每前缀等价；传统TD差异。 | 神经网络的true-online等价。 |
| control / offpolicy | 合法一步信用与剪迹时序；传播系数；Q(σ)端点；V-trace固定点策略。 | 完整分布式IMPALA或Atari复现。 |
| adaptive | 变量折扣／λ恒等式；局部已知统计量的最优混合。 | 完整在线λ-greedy或meta-gradient复现。 |
| expected | Markov汇合状态的均值／逐分量方差；混叠反例。 | 任意近似期望迹都无偏。 |
| gradient | 非线性冻结双网络总增量与GTD2有限差分。 | 深度全局收敛或QRC性能优势。 |
| actor / recurrent | 固定score误差和；递归敏感度有限差分与detach反例。 | 在线变参梯度或RTU完整基准复现。 |

作者实验仓库的 totd.py 把步长 $\alpha$ 折进迹变量，并用 predprev 保存旧预测。配套代码使用迹外步长；在这里的常数步长条件下，二者通过 $z_{\rm author}=\alpha z$ 对应。比较源码时应先统一这个定义，再比较修正项。实验入口、随机 MDP 配置与结果处理位于同一作者仓库；完整论文实验需要其依赖和配置。

<a id="lesson-branches"></a>

## 15. 可区分机制的研究路线

| 研究问题 | 直接改变的对象 | 不能混同的对象 |
| --- | --- | --- |
| 奖励很晚，更新只到最近一步 | 回报长度、λ 混合、资格迹。 | 不是单纯增加网络深度。 |
| 记忆存在，但最初写入参数没梯度 | RTRL、BPTT、截断边界。 | 加长 TD 迹不能恢复已删掉的结构梯度。 |
| 同一轨迹上前后向更新不一致 | 参数冻结约定、在线目标、荷兰迹与修正。 | 不能用数值接近替代精确等价。 |
| 没有样本缓冲仍超出实时预算 | 迹存储、梯度计算、递归结构与数值尺度。 | 流式约束不等于某一种信用算法。 |
| 固定步长或 λ 在变化后失效 | 步长适应、meta-gradient、Metatrace。 | 元迹追踪学习参数影响，和价值资格迹分开存。 |
| off-policy 长迹方差或偏差很大 | 目标策略修正、截断、梯度 TD 或 emphatic 方法。 | 一条未经修正的 on-policy 迹不提供稳定性保证。 |

同一算法可以同时属于几条研究线。Metatrace 就为步长建立元时间信用；流式 actor–critic 可以同时使用价值迹、策略迹和递归敏感度。教材按这些对象分别讲解，是为了明确接口，不是把算法划进互不相交的名词类别。

| 可证伪问题 | 受控改变 | 判断与竞争解释 |
| --- | --- | --- |
| 长迹改善来自更快奖励传播，还是更大更新？ | 固定表示，增加奖励延迟；匹配实际更新尺度。 | 比较首状态预测误差和回报；仅迹范数增大不足以支持机制。 |
| Retrace保留长信用是否提高效率？ | 固定Q和轨迹覆盖，仅改变π与μ差异；比较Tree-backup。 | 记录信用衰减、方差、近同策略学习速度；加进逼近后重新检查稳定性。 |
| 期望迹减少路径噪声，还是把不同隐藏状态混到一起？ | Markov汇合与观察混叠两组环境；保持相同奖励边际分布。 | 分别测均值偏差和方差；均方误差降低不证明信用无偏。 |
| 辅助误差预测改善了目标方向吗？ | 同样λ与算力，比较半梯度、GTD2、TDC与TDRC。 | 同时记录辅助误差、主预测误差、实际参数步幅；额外网络容量是竞争解释。 |
| 远处提示学不到，是状态遗失还是梯度被截断？ | 固定总参数预算，分别改变记忆跨度、RTRL导数与回报λ。 | 记录提示可解码性、敏感度误差、回报；这些量不能互相替代。 |
| 自适应λ在环境改变后仍有效吗？ | 同一生命改变噪声与预测可靠性；计入辅助统计的适应速度。 | 比较固定λ、已知统计的局部上限参照和在线估计；未来真值只能用于评价。 |

先完成有限轨迹的恒等式与反例，再进入固定策略预测，随后才让策略和表示共同学习。后一阶段还需报告新能力形成、旧能力保留、恢复速度及每步资源。资格迹解决了哪一条信用路径，应由受控改变检验，而不是由算法名称或一条更高的奖励曲线推断。

<a id="lesson-check"></a>

## 16. 失败边界与思考题

| 观察到的错误 | 检查方法 |
| --- | --- |
| 终点奖励只更新最后状态 | 是否在使用该奖励前清空了旧迹。 |
| True-online 在第二步才出错 | 保存的是否是更新前 V′；内积是否使用旧迹；是否遗漏预测差修正。 |
| 随机动作导致所有迹都被切断 | 实际动作是否仍为并列最大；不要仅检查探索分支标志。 |
| 长迹在重复特征上造成大步更新 | 积累次数、特征尺度和 α；精确前向等价不是稳定性定理。 |
| 参数不更新时测试通过，训练中却失配 | 在线变参、特征漂移和策略采样变化是否超出了证明条件。 |
| Trace 很长却学不会远处提示 | 状态是否保存提示，以及递归梯度是否被 detach。 |
| 所有比率已截断，线性预测仍发散 | 表格算子结论是否被外推到投影与函数逼近；是否需要梯度目标。 |
| V-trace训练值稳定，却与原目标价值不同 | ρ上限诱导的固定点策略是否已改变。 |
| 期望迹均值变了 | 条件输入是否Markov；旧标签是否仍在同一参数坐标。 |
| 辅助网络学习很好，主网络却实现不同规则 | 是否把∇δ中的下一预测detach；是否遗漏H项、标量迹或更新时序。 |

- 把手算的冻结参数例子改成 λ=0。答案：首步目标变成 0.36，首权重增量为 0.016；第二权重增量仍为 0.06。
- True-online 的 λ=0 为什么退化为普通 TD(0)？答案：迹等于当前特征，两个 V−Vold 修正相消，只剩 αδx。
- 能否只使用荷兰迹而删掉预测差修正？答案：不能；它一般不再实现这里定义的在线前向算法。
- Watkins 的 λ=1 是否必然使用整回合 Monte Carlo return？答案：不是；非贪心动作仍会剪断延续并使用 bootstrap。
- RTRL 敏感度与 TD 资格迹是否都是“历史压缩”，因而可以共用一块变量？答案：它们压缩不同导数对象，通常维度、递推和重置条件都不同。
- π=μ时Tree-backup和Retrace是否完全相同？答案：随机策略下不相同；前者仍乘动作概率，后者的比率为一。
- 截断V-trace的ρ与c是否都只改变方差？答案：不是；ρ上限决定固定点策略，c主要决定后续误差的传播。
- Expected Traces的低方差是否允许忽略agent state？答案：不允许；隐藏历史决定未来转移时，信用均值也可能改变。
- 双网络非线性GTD迹的总和测试通过，是否意味着在线每一步等价？答案：不是；测试冻结参数，原定理也明确这一条件。

一个可开始的研究是固定状态表示与环境，分别增加奖励延迟和重复特征比例。比较传统迹、true-online 与短窗口目标，同时记录预测误差、迹范数、每步计算和内存。第二阶段再引入表示学习，观察原先的精确等价如何失效。两个阶段分开，才能定位变化来自时间目标还是结构梯度。

## 本章的实验设计

先验证固定参数的前后向恒等式、在线 true-online 参考和离策略比率。再检验条件期望迹及非线性梯度迹，最后做学习速度实验。

设定：固定短轨迹核对线性在线等价和离策略传播，再用路径汇合/观测混叠检查 Expected Traces，用双 tanh 网络检查冻结非线性梯度总和。

- λ=0 退化为相应单步更新；真实终端反馈先沿迹传播再清理。
- Retrace 与 V-trace 的传播下标和两类比率分别核对。
- 条件期望迹的均值/方差结论需要状态条件独立；混叠反例应破坏结论。
- GTD2 主、辅助方向分别通过有限差分；TDC/TDRC 对应各自修正。
- 冻结参数总和等价不冒充逐前缀 true-online 等价。

对照：固定参数与在线更新分开；同策略与离策略、Markov 与混叠成对；固定 λ、状态 λ 与截断长度；GTD2/TDC/TDRC 主辅助网络预算匹配

记录：每步迹、比率、误差、参数及网络版本；独立前向参考与有限差分误差；条件均值/方差、信用长度与计算量；匹配资源下的延迟任务学习曲线

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-estimator)

## 学习与研究衔接

先分开前向目标、后向计算与离策略校正，再比较 Expected Traces 和梯度迹的估计对象。RTRL 传播递归敏感度；元学习传播更新规则的敏感度。

[分册导读](https://yingwen.io/zh/continual-rl/start/classic-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-credit) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=credit) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=credit)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

状态是支持后续计算的内部信息；GVF 指定一个预测问题；RTRL 和资格迹规定信用如何传播。三者可以组合，但不是相互替代的算法名称。先理解给定策略的预测，再讨论预测怎样改善控制。

- [Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks](https://yingwen.io/zh/continual-rl/research/#recent-columnar-constructive-networks)
- [Real-Time Recurrent Learning using Trace Units in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-real-time-trace-units)
- [Recurrent Reinforcement Learning with Memoroids](https://yingwen.io/zh/continual-rl/research/#recent-memoroids-sequence-learning)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)
- [Expected Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-expected-eligibility-traces)
- [Safe and Efficient Off-Policy Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-retrace-safe-offpolicy)
- [Convergent Tree Backup and Retrace with Function Approximation](https://yingwen.io/zh/continual-rl/research/#recent-convergent-tree-retrace)
- [Multi-Step Reinforcement Learning: A Unifying Algorithm](https://yingwen.io/zh/continual-rl/research/#recent-q-sigma-backups)
- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://yingwen.io/zh/continual-rl/research/#recent-vtrace-impala)
- [A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning](https://yingwen.io/zh/continual-rl/research/#recent-lambda-greedy)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)
- [Recurrent Reinforcement Learning with Memoroids](https://yingwen.io/zh/continual-rl/research/#recent-memoroids-sequence-learning)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Streaming Deep Reinforcement Learning Finally Works](https://yingwen.io/zh/continual-rl/research/#recent-stream-x)
- [Intentional Updates for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-intentional-updates)
- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)
- [Real-Time Recurrent Learning using Trace Units in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-real-time-trace-units)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Intentional Updates for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-intentional-updates)
- [A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning](https://yingwen.io/zh/continual-rl/research/#recent-lambda-greedy)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [Recurrent Reinforcement Learning with Memoroids](https://yingwen.io/zh/continual-rl/research/#recent-memoroids-sequence-learning)

### Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks

Khurram Javed, Haseeb Shah, Richard S. Sutton, Martha White

JMLR 24 · 2023 · 支持方法与理论

#### 研究问题

如果每次观测只处理一次，如何学习包含历史信息的状态，而不保存一段序列做反向传播？

#### 关键机制

一般递归网络的实时递归学习需要维护庞大的参数—状态敏感度。CCN 限制列之间的递归依赖，并逐步构造新特征，使敏感度可以局部计算。它通过改变网络结构和构造过程降低求导成本，而不是把任意稠密 RNN 的完整导数免费变小。

#### 证据

论文分析受限结构的计算性质，并在动物学习启发的预测问题和 Atari 策略评价中检验预测效率。这里的 Atari 结果主要是预测已有策略的回报，不等于从头训练完整控制智能体。

#### 条件与限制

结构约束、构造顺序和被冻结的旧特征共同限制函数类。监督预测和策略评价上的优势，还需要在会主动改变数据分布的控制闭环中检验。

#### 阅读与实验

先写出递归状态对参数的敏感度递推，再检查哪些跨列项被结构消除。比较 CCN、截断 BPTT 与 RTU 时，同时计入状态、梯度缓存和每步计算。

#### 原文与相关入口

- [JMLR 原文与论文入口](https://www.jmlr.org/papers/v24/23-0367.html)：从网络结构、敏感度传播与预测实验三部分阅读。

### Real-Time Recurrent Learning using Trace Units in Reinforcement Learning

Esraa Elelimy, Adam White, Michael Bowling, Martha White

NeurIPS 2024 · 2024 · 支持方法与理论

#### 研究问题

递归状态既要保存长时信息，又要在在线强化学习中以可控成本更新，怎样设计其递归结构？

#### 关键机制

RTU 使用有结构的递归连接，并维护状态关于参数的在线敏感度。复杂的递归动力学可以用实值运算实现。其关键是让状态更新与梯度迹具有相容的计算结构，减少一般 RTRL 的高阶成本；这与仅给 TD 误差加一条资格迹不同。

#### 证据

论文在部分可观测任务中与常见递归网络比较预测与控制表现。作者代码包含 RTU、其他递归基线、实时 actor–critic 以及部分可观测环境配置。

#### 条件与限制

计算优势依赖特定递归参数化，不能外推为任意记忆问题上的表达能力优势。PPO 版本和严格逐步更新版本的经验协议不同，应分别比较。

#### 阅读与实验

在同一部分可观测任务中固定隐状态维度，再比较完整运行内存和每步更新时间。检查 actor、critic 与递归状态的参数更新是否共享同一条敏感度。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1e616bde0438cb10cb6adf076ae7d336-Abstract-Conference.html)：结构、在线导数与实验协议。
- [作者代码](https://github.com/esraaelelimy/rtus)：从 src/nets、src/agents 和实验配置追踪递归状态到控制更新。

#### 作者代码

[论文作者维护的实现。](https://github.com/esraaelelimy/rtus)

RTU 网络、实时学习器与论文实验配置。

### Deep Reinforcement Learning with Gradient Eligibility Traces

Esraa Elelimy, Brett Daley, Andrew Patterson, Marlos C. Machado, Adam White, Martha White

RLC 2025 / RLJ · 2025 · 支持方法与理论

#### 研究问题

资格迹怎样与明确的梯度目标结合，而不是直接把线性半梯度规则搬到深度网络？

#### 关键机制

论文从广义投影 Bellman 误差出发构造多步目标，推导带资格迹的梯度学习方法。前向视角连接多步回报与经验重放，后向视角通过递推迹分配信用。目标函数、辅助估计器和迹的更新共同决定算法，不只是选择一个较大的 λ。

#### 证据

作者给出多种算法并在 MuJoCo、MinAtar 等任务中比较。代码同时提供相关梯度算法与实验设置，可以把推导中的量映射到实际更新。

#### 条件与限制

线性 GTD 的收敛条件不能自动赋予非线性实现全局收敛保证。重放版本与流式版本的数据使用预算也不能混为一谈。

#### 阅读与实验

从一段短轨迹分别计算前向多步目标和后向迹。随后对照原代码检查辅助网络、目标与主网络参数使用的是更新前还是更新后的值。

#### 原文与相关入口

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_302.pdf)：目标、算法推导与实验。
- [作者算法库](https://github.com/esraaelelimy/gtd_algos)：论文提供的梯度 TD 与资格迹实现。

#### 作者代码

[原论文链接的作者仓库。](https://github.com/esraaelelimy/gtd_algos)

论文梯度算法、资格迹和实验配置。

### Streaming Deep Reinforcement Learning Finally Works

Mohamed Elsayed, Elena Sorina Lupu, Gautham Vasan, A. Rupam Mahmood

arXiv（2024 首稿；2026 v3） · 2026 · 直接研究持续学习

#### 研究问题

不保存经验重放、不使用目标网络或训练批次时，深度 RL 能否逐步稳定学习？

#### 关键机制

Stream-X 把信号归一化、表示初始化、资格迹和受控更新尺度组织为一组流式学习方法。各组件处理的是不同问题：奖励尺度、激活与梯度传播、延迟信用，以及一次更新造成的输出变化。去掉重放并不意味着这些问题会自动消失。

#### 证据

2026 年第三版扩展到 Atari、控制与机器人等实验，并包含持续变化设置。论文和代码经历过版本变化，比较结果时需要同时标明论文版本和算法实现。

#### 条件与限制

广泛任务上的流式可行性不等于所有非平稳问题都已解决。不能把旧版较弱 Adam 基线推广成对所有流式 Adam 方法的否定；后续研究专门检验了这一点。代码许可证也应独立于本教材许可证处理。

#### 阅读与实验

按归一化、资格迹、更新控制分别做消融，并保持每步算力一致。先验证严格一次使用经验，再研究长期变化，而不是仅把小批量大小改成一。

#### 原文与相关入口

- [2026 年第三版论文](https://arxiv.org/abs/2410.14606v3)：作者名单、任务范围与算法版本以该版为准。
- [作者代码版本](https://github.com/mohmdelsayed/streaming-drl/tree/9326fc3e23a401f28087ae2e41b635888740586b)：固定实现版本，避免把不同年份的更新规则混在一起。

#### 作者代码

[原作者仓库的固定版本。](https://github.com/mohmdelsayed/streaming-drl/tree/9326fc3e23a401f28087ae2e41b635888740586b)

Stream-X 算法、变换、优化器及实验；使用前阅读仓库许可证。

### Intentional Updates for Streaming Reinforcement Learning

Arsalan Sharifnassab, Mohamed Elsayed, Kris De Asis, A. Rupam Mahmood, Richard S. Sutton

ICML 2026 · 2026 · 支持方法与理论

#### 研究问题

能否先规定本次更新应产生多大作用，再反推合适的参数更新尺度？

#### 关键机制

Intentional 方法以局部线性近似连接参数变化和预测变化。critic 以减少一定比例的 TD 误差为目标，actor 控制策略输出变化的局部代理量；再结合资格迹和逐坐标尺度，求出这一次更新的强度。这是有目标的局部更新控制，不是对长期表现求导的元梯度。

#### 证据

论文给出推导和流式控制比较，ICML 2026 正式论文入口与作者实现均可用。实现将优化器与 actor–critic 交互区分开，便于检查更新时序。

#### 条件与限制

Taylor 近似在大更新时可能失准。采样动作上的对数概率变化不等于精确的全分布 KL 上界；熵项与 TD 误差符号也必须按原算法处理。

#### 阅读与实验

在一次更新前后直接测量预测变化，并与线性估计比较。分别测试正、负 TD 误差和很小梯度的情形，不要只检查参数是否有限。

#### 原文与相关入口

- [ICML 2026 原文](https://proceedings.mlr.press/v306/sharifnassab26a.html)：正式会议版本与更新意图的定义。
- [作者实现](https://github.com/sharifnassab/Intentional_RL)：重点对照 optimizer.py 与 intentional_ac.py。

#### 作者代码

[原论文作者提供的实现。](https://github.com/sharifnassab/Intentional_RL)

Intentional 更新与流式 actor–critic。

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

### Expected Eligibility Traces

Hado van Hasselt, Sephora Madjiheurem, Matteo Hessel, David Silver, André Barreto, Diana Borsa

AAAI 2021（2020预印本） · 2021 · 支持方法与理论

#### 研究问题

当前误差能否同时更新本次未走过、但也可能到达当前状态的过去路径？

#### 关键机制

学习给定当前状态的资格迹条件均值，再用当前TD误差更新该均值所指向的过去预测。递归混合在实际轨迹迹与预测的期望迹之间插值；预测对象是过去资格，而非未来奖励。

#### 证据

原文在Markov状态与相应条件下证明更新均值相同、逐分量方差不增，并在路径汇合问题检验预测效率。信用章精确枚举一个正例和一个状态混叠反例。

#### 条件与限制

不完整观察、参数漂移和近似迹预测器会破坏无偏条件。全参数期望迹预测还有输出维度和计算成本；小实验不复现作者的神经实验。

#### 阅读与实验

保持奖励边际分布一致，仅改变奖励是否依赖隐藏的过去路径。先测信用均值与方差，再研究agent state能否恢复条件独立。

#### 原文与相关入口

- [作者原文](https://arxiv.org/html/2007.01839)：Lemma 1、Proposition 1及ET(λ,η)递归混合。
- [AAAI发表版本](https://ojs.aaai.org/index.php/AAAI/article/view/17200)：正式会议年份为2021。

### Safe and Efficient Off-Policy Reinforcement Learning

Rémi Munos, Tom Stepleton, Anna Harutyunyan, Marc G. Bellemare

NeurIPS 2016 · 2016 · 支持方法与理论

#### 研究问题

目标与行为策略不一致时，如何保留多步信用而避免重要性比率乘积爆炸？

#### 关键机制

统一多步目标为目标策略TD误差的加权和，Retrace采用λmin(1,π/μ)传播系数。近同策略时保留长迹，目标概率较低的动作则减少传播；一步误差仍使用目标动作期望。

#### 证据

论文分析表格算子的收缩性质，给出条件下的评价与控制收敛，并报告Atari实验。信用章独立检查传播系数和有限轨迹恒等式。

#### 条件与限制

表格安全性不是任意线性或神经逼近的稳定性保证。行为覆盖、变化策略与投影条件仍需检查；代码小实验不复现Atari。

#### 阅读与实验

在同样轨迹与表示上，分别改变策略差异和动作随机性，比较Tree-backup与Retrace的信用长度、方差和预测误差。

#### 原文与相关入口

- [原论文](https://arxiv.org/html/1606.02647)：统一算子、传播系数及理论条件。

### Convergent Tree Backup and Retrace with Function Approximation

Ahmed Touati, Pierre-Luc Bacon, Doina Precup, Pascal Vincent

ICML 2018 · 2018 · 支持方法与理论

#### 研究问题

传播系数已经截断，为什么函数逼近下的Tree-backup和Retrace仍可能发散？

#### 关键机制

分析函数逼近与off-policy多步bootstrap的学习算子，展示线性反例，再把相应目标写成二次凸凹鞍点问题，构造梯度版本。

#### 证据

原文给出线性不稳定例子、梯度方法收敛保证与有限样本界。它直接限定了从Retrace表格结论外推到逼近算法的范围。

#### 条件与限制

凸凹线性问题的保证不能自动覆盖学习表示的深度网络。稳定目标、更新速度与控制性能还需分别验证。

#### 阅读与实验

先检查固定表示下的期望更新矩阵，再将半梯度和梯度版本按相同样本、步数与计算预算比较。

#### 原文与相关入口

- [ICML原文](https://proceedings.mlr.press/v80/touati18a.html)：理论反例、鞍点方法和保证条件。

### Multi-Step Reinforcement Learning: A Unifying Algorithm

Kristopher De Asis, J. Fernando Hernandez-Garcia, G. Zacharias Holland, Richard S. Sutton

AAAI 2018 · 2018 · 支持方法与理论

#### 研究问题

多步动作价值目标必须始终采样下一动作，或始终对动作取期望吗？

#### 关键机制

Q(σ)逐处混合Sarsa的采样动作与Expected Sarsa的动作期望，并同步改变后续误差传播。σ控制采样程度，与控制回报长度的λ不同。

#### 证据

原文给出统一n-step表达、off-policy修正和实验比较。信用章小程序核验冻结on-policy几何λ混合的两个端点。

#### 条件与限制

原文n-step和本章λ混合参考具有不同实现范围。只改一步误差却不改多步传播或策略修正，不能称为完整Q(σ)。

#### 阅读与实验

把采样噪声、目标长度和策略差异分开改变，避免把σ与λ的作用归到同一“更长信用”解释。

#### 原文与相关入口

- [原文](https://arxiv.org/html/1703.01327)：式13–15：混合误差、传播及off-policy修正。

### IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures

Lasse Espeholt, Hubert Soyer, Rémi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, Koray Kavukcuoglu

ICML 2018 · 2018 · 支持方法与理论

#### 研究问题

actor采样策略落后于learner时，如何校正状态价值与策略更新？

#### 关键机制

V-trace用截断ρ校正当前TD误差，用独立截断c控制后续误差传播，再用下一状态V-trace目标构造actor优势。ρ上限还决定表格固定点对应的截断策略。

#### 证据

原文分析固定点并检验分布式多任务训练。固定版本作者代码明确区分clipped_rhos、cs、反向scan与pg_advantages。

#### 条件与限制

IMPALA保存短轨迹并批量训练，不属于严格单样本流式协议。截断后价值可能对应不同于原目标的策略；信用章bandit示例显示0.8变为0.5。

#### 阅读与实验

独立改变策略滞后、ρ上限和c上限。记录目标策略变化与传播长度，不把两种截断都只解释为方差控制。

#### 原文与相关入口

- [ICML原文](https://proceedings.mlr.press/v80/espeholt18a.html)：V-trace固定点与分布式实验。
- [作者固定实现](https://github.com/google-deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/vtrace.py)：from_importance_weights与下一状态actor目标。

#### 作者代码

[原作者团队仓库的固定版本。](https://github.com/google-deepmind/scalable_agent/tree/6c0c8a701990fab9053fb338ede9c915c18fa2b1)

IMPALA原始TensorFlow实现与V-trace；运行需要原项目环境。

### A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning

Martha White, Adam White

arXiv预印本 · 2016 · 支持方法与理论

#### 研究问题

不同状态的预测可靠性不同，固定λ是否浪费了多步信用？

#### 关键机制

将下一处bootstrap选择写成局部偏差平方与回报方差的折中，得到$λ=b^2/(b^2+\operatorname{Var}(G))$。完整λ-greedy还用在线预测器估计回报均值和二阶矩。

#### 证据

原文给出状态相关λ的目标、增量算法和多个预测设置的实验。信用章仅核对已知统计量下的局部最优与变量λ恒等式。

#### 条件与限制

局部贪心目标不是整条轨迹的联合最优。逼近误差、统计滞后和非平稳性会影响λ估计；辅助资源需要计入比较。

#### 阅读与实验

先让噪声方差变化，再让bootstrap可靠性变化。比较固定λ、已知统计参照和在线估计，分别观察目标偏差与适应速度。

#### 原文与相关入口

- [作者原文](https://arxiv.org/html/1607.00446)：局部目标、状态λ、均值／二阶矩预测与完整算法。

### Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning

Noah Farr, Aryaman Reddi, Carlo D’Eramo, Jan Peters

arXiv预印本（2026-07-07 v2） · 2026 · 支持方法与理论

#### 研究问题

严格逐步更新的智能体怎样同时学习递归记忆、分配延迟信用并控制计算？

#### 关键机制

将RTU结构的RTRL敏感度接入QRC与流式actor–critic。敏感度给出当前输出对记忆参数的导数，资格迹再组合过去输出的回报信用；两条递推保持分工。

#### 证据

v2在MemoryChain、五项POPGym和masked MuJoCo上报告5-seed结果，另用KMemoryChain比较在线敏感度与当前参数重算参考，并检验Taylor修正。

#### 条件与限制

masked MuJoCo仍落后批量PPO。固定参数精确RTRL不代表在线变参敏感度始终等于当前参数重算；诊断保存整个episode，须计为额外评价资源。尚未确认作者公开代码。

#### 阅读与实验

在相同递归容量下独立改变记忆跨度、回报λ与参数步幅，同时测敏感度误差和回报。诊断改善不能单独当作控制改进证据。

#### 原文与相关入口

- [2026年v2原文](https://arxiv.org/html/2605.24709v2)：方法、5-seed实验、masked MuJoCo负边界与staleness诊断。


<a id="chapter-code"></a>

## 下载与运行

标准库：31项检查覆盖冻结／在线等价、策略校正、Q(σ)端点、V-trace目标变化、状态λ、期望迹混叠反例、非线性梯度迹有限差分及actor／递归敏感度；不是完整论文复现。

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

- [Precup, Sutton & Singh · Eligibility Traces for Off-Policy Policy Evaluation · ICML 2000](https://web.eecs.umich.edu/~baveja/Papers/OffPolicy.pdf)：作者站点原文；重要性采样、Tree-backup与目标／行为策略的分工。

- [Munos等 · Safe and Efficient Off-Policy Reinforcement Learning · NeurIPS 2016](https://arxiv.org/html/1606.02647)：统一误差传播系数；Retrace算子收缩及表格评价／控制的条件。

- [Touati等 · Convergent Tree Backup and Retrace with Function Approximation · ICML 2018](https://proceedings.mlr.press/v80/touati18a.html)：线性函数逼近中的不稳定反例，以及鞍点形式的梯度方法。

- [De Asis等 · Multi-Step Reinforcement Learning: A Unifying Algorithm · AAAI 2018](https://arxiv.org/html/1703.01327)：Q(σ)采样／期望混合；正文数值代码仅实现冻结on-policy几何λ混合。

- [Espeholt等 · IMPALA · ICML 2018](https://proceedings.mlr.press/v80/espeholt18a.html)：V-trace两类截断、固定点策略与actor目标；区分轨迹窗口和流式资格迹。

- [IMPALA作者V-trace实现 · 固定版本](https://github.com/google-deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/vtrace.py)：from_importance_weights：clipped_rhos、cs、反向scan、下一目标的actor优势和stop_gradient。

- [White & White · A Greedy Approach to Adapting the Trace Parameter · 2016](https://arxiv.org/html/1607.00446)：状态相关λ；局部bias–variance目标；回报均值和二阶矩的在线估计。

- [Xu, van Hasselt & Silver · Meta-Gradient Reinforcement Learning · NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/2715518c875999308842e3455eda2fe3-Abstract.html)：通过学习更新和后续评价适应回报定义；区别于同样本直接选择容易拟合的目标。

- [van Hasselt等 · Expected Eligibility Traces · AAAI 2021](https://arxiv.org/html/2007.01839)：条件期望迹、Markov均值／逐分量方差结论、递归混合与特征混叠边界；2020年预印本。

- [Elelimy等 · Deep Reinforcement Learning with Gradient Eligibility Traces · RLC 2025](https://arxiv.org/html/2507.09087v1)：广义投影目标、GTD2／TDC／TDRC三条迹；Theorem 6.1冻结双网络参数，Table 2与QRC公式中的标量H迹相对应。

- [GTD资格迹作者QRC代码 · 固定版本](https://github.com/esraaelelimy/gtd_algos/blob/76293dea9b2129d55e08bfb4178618a0a26c2dd8/gtd_algos/src/algorithms/qrc.py)：update_step先共同计算主／辅助方向，再更新参数；检查完整∇δ、三条迹、正则与更新后的清迹。

- [Schulman等 · High-Dimensional Continuous Control Using Generalized Advantage Estimation · ICLR 2016](https://arxiv.org/html/1506.02438)：TD误差和的优势解释、bias–variance及折扣目标约定；GAE不等于完整actor迹学习协议。

- [Elelimy等 · Real-Time Recurrent Learning using Trace Units in Reinforcement Learning · NeurIPS 2024](https://arxiv.org/html/2409.01449)：特定递归结构的在线敏感度、固定参数精确导数与变参时过时问题。

- [RTU作者递归实现](https://github.com/esraaelelimy/rtus/blob/main/src/nets/rtus/non_linear_rtus.py)：FwdRealTimeNonLinearRTUs推进活动与grad_memory；自定义求导使用已存敏感度。

- [Farr等 · Streaming RL under Partial Observability with RTRL · 2026 v2](https://arxiv.org/html/2605.24709v2)：2026-07-07预印本；RTU与QRC／流式AC组合、5-seed实验、masked MuJoCo负边界及需保存轨迹的敏感度诊断；未确认作者代码。
