# 学习规则的适应：在线元梯度与跨任务元学习

学习规则的适应研究经验怎样改变学习速度、训练目标或更新方式。在线元梯度在同一经验流中评价过去更新的影响；跨任务元学习则利用训练任务中的适应经历，改进新任务上的适应。本章分别定义这两种设定，再推导其算法。

## 本章内容

- 明确内层状态、元参数、外层目标与数据权限，区分在线元梯度、跨任务元学习和预设尺度控制。
- 从多步参数更新推导精确敏感度；说明 IDBD/TIDBD、截断元梯度和一阶 MAML 丢掉了哪些项。
- 完整理解 MAML、RL²、PEARL、meta-gradient RL 与 learned update rules 的训练/测试循环及 reset 边界。
- 用可运行的有限差分、手算与机制反例验证代码，并设计适用于持续交互而非仅任务重置的评测。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 两种时间尺度

内层变量随当前经验适应；外层变量决定内层怎样适应，并由后续表现训练。“外层”不必在另一台机器，也不必慢到离线。

### 梯度与 Hessian

梯度描述损失随参数的一阶变化；Hessian 描述梯度本身如何改变。对一次梯度下降再求导，自然会出现 Hessian，而不是额外加的技巧。

$$
\frac{\partial (w-\alpha\nabla L(w))}{\partial w}=I-\alpha\nabla^2L(w)
$$

### 固定数据条件下求导

可微优化循环通常先把采样轨迹当常量。RL 的策略变化还会改变轨迹分布；是否估计这条路径必须另行说明。

<a id="lesson-setting"></a>

## 1. 问题设定：哪些变量在适应，适应后的表现如何评价

环境变化可以要求不同形式的适应。例如，地面摩擦系数改变后，机器人可以更新控制参数、推断新的动力学情境，或改变参数学习的速度。三种方法分别改变模型参数、活动状态和学习规则，所依赖的数据与评价目标也不同。

本章有两种主要数据设定。在线元梯度可在一个持续运行的问题上工作：权重不断学习，步长或学习目标也根据后续误差调整，不要求出现一组可重置的任务。跨任务 meta-RL 则先规定任务分布与任务内适应过程，用训练任务中的经历训练初始化、推断器或更新规则，再检验新任务适应。两者都涉及学习过程，但评价单位与信息权限不同。

| 设定 | 在线元梯度 | 跨任务 meta-RL |
| --- | --- | --- |
| 数据组织 | 同一经验流中的先前更新与后续评价 | 训练任务中的适应数据和适应后评价数据 |
| 持久状态 | 权重与元敏感度通常跨更新保留；按真实边界处理资格迹 | 按任务或 trial 协议复制初值、清理 context 或循环活动 |
| 部署时变化 | 权重与元参数均可继续更新 | 外层参数常固定，任务内权重或活动状态继续适应 |
| 主要检验 | 长寿命中的累计误差、控制表现与变化后适应 | 未见任务上，在规定数据与 reset 预算内的适应 |
| 额外条件 | 元梯度展开、数据复用与每步预算 | 任务分布、元训练成本、适应次数与测试任务隔离 |

在线元梯度也可能缓存 rollout 或展开计算图，因此不自动满足严格流式协议。跨任务训练得到的规则也可以用于持续交互，但跨任务适应表现不能替代单次长生命中的持续学习证据。资格迹解决的是时间信用分配；它既可服务普通 TD，也可服务元目标，不以元学习为前提。

| 路线 | 内层适应什么 | 外层学什么 | 典型监督/边界 |
| --- | --- | --- | --- |
| 在线超参数学习：IDBD/TIDBD、meta-gradient RL | 预测器或策略权重 $w$ | 步长 $\alpha$、discount、$\lambda$、更新系数等 $\eta$ | 后续误差或回报；可在单一流上工作，不必有任务 ID。 |
| 优化型 Meta-RL：MAML | 从共同初值出发的任务内权重 $w_i$ | 适合少量梯度适应的初始化 $\theta$ | 任务分布、适应数据与适应后数据；通常有明确 trial/reset。 |
| 上下文型 Meta-RL：RL²、PEARL | 循环活动 $h$ 或任务潜变量 $z$ | 推断与控制网络参数 $\theta,\phi$ | 训练任务分布；测试时可仅更新上下文，不更新网络参数。 |
| 学习更新规则：LPG、DiscoRL 等 | 被训练的 agent 参数与学习器状态 | 产生目标/损失/更新信号的规则 $\eta$ | 跨环境、跨训练轨迹的外层表现；元训练预算必须单列。 |

统一记经验为 $\xi_t$，内层学习状态为 $w_t$，元参数为 $\eta$，更新映射为 $F_\eta$，外层损失为 $J$。这里的“状态”包括所有影响后续学习的量：除了网络权重，还可能有优化器动量、资格迹和归一化统计量。将它们纳入更新映射，才能定义需要求导的过程；把某些状态视为常数，则对应一个明确的梯度近似。

$$
w_{t+1}=F_\eta(w_t,\xi_t),\qquad \mathcal J(\eta)=\mathbb E\bigl[J(w_{t+K},\xi_{t+K:t+K+M};\eta_{\rm eval})\bigr]
$$

$K$ 是适应长度，$M$ 是后续评价长度；$\eta_{\rm eval}$ 表示固定的评价约定。训练代理目标可以学习，最终评价标准仍需要事先定义。

- 先说明数据协议：独立任务、同一任务内多 episode，还是一条未分段的生命流。
- 再说明评价目标：预测误差、适应后回报、训练全过程回报、变化后恢复速度，不能彼此代替。
- 最后说明测试权限：能否重置参数/上下文/环境，是否拿到任务边界，是否继续更新外层参数。

<a id="lesson-derive"></a>

## 2. 多步更新的敏感度与链式法则

先固定 $\eta$ 和采样数据，定义 $H_t=\partial w_t/\partial\eta$。对更新 $F$ 求导，要包含 $F$ 对 $\eta$ 的直接依赖，以及 $\eta$ 通过旧参数 $w_t$ 产生的间接依赖。若 $\eta$ 就是初始化，则 $H_0=I$；若 $\eta$ 是独立步长而初始 $w_0$ 与它无关，则 $H_0=0$。

$$
H_{t+1}=\frac{\partial F_\eta}{\partial w_t}H_t+\frac{\partial F_\eta}{\partial\eta}
$$

这是固定元参数、固定经验序列下的精确前向敏感度。若 $F=w+f$，传播矩阵为 $I+\partial f/\partial w$，旧敏感度通常会被当前更新的 Jacobian 改变。

$$
\nabla_\eta J=H_{t+K}^{\top}\nabla_w J+\left.\nabla_\eta J\right|_w
$$

若评价损失只通过适应后的 $w$ 依赖 $\eta$，右端第二项为零。若评价目标也含可学习 discount，第二项可能通过缩短评价时间尺度降低损失，而没有改善原定的长期目标。

$$
\begin{aligned}F_\beta(w)&=w-\alpha\nabla_w L(w),\qquad \alpha=e^\beta,\\H_{t+1}&=(I-\alpha\nabla_w^2L_t)H_t-\alpha\nabla_wL_t.\end{aligned}
$$

对数步长 $\beta$ 保证 $\alpha>0$。第一项传播已有敏感度，第二项表示步长对本次更新的直接影响。敏感度矩阵 $H$ 的行数是内层状态维度，列数是元参数维度；二者都很大时需要结构或近似。

只看一次内层更新，且 $H_t=0$ 时，外层对 $\beta$ 的导数为 $-\alpha g_{\rm train}^{\top}g_{\rm eval}$。两个梯度方向一致，适当增大步长会局部改善后续损失；方向相反，则倾向缩小步长。这是局部导数的解释，不是任意大步长下的单调性保证。样本噪声和梯度相关性还会影响该信号的可信度。

$$
\frac{\partial J(w-\alpha g_{\rm train})}{\partial\beta}=-\alpha\,g_{\rm eval}^{\top}g_{\rm train}
$$

$g_{\rm eval}$ 在更新后的参数上计算，$g_{\rm train}$ 在更新前计算。把两者都在旧参数上计算，得到的是另一种近似。

RL 还有采样路径：$\eta$ 改变策略，策略改变动作及后续数据。完整期望回报的导数一般包含轨迹分布的 score-function 项。固定数据的解析实验能够隔离优化器微分；采样路径是否无偏，则需要在随机过程的层面另外检验。

两步及多步学习的完整标量敏感度；旧 w 和旧 H 一致使用，可由有限差分独立核对。

```python
def fixed_beta_training(beta, train_targets, validation_target, initial=0.0):
    """Exact unrolled log-step-size derivative for FIXED beta and fixed data.

    w_next = w + exp(beta)*(target-w). H=dw/dbeta, H0=0.
    Includes every inner update Jacobian; it is not an online changing-beta
    total derivative and does not differentiate the data collection policy.
    """
    alpha = math.exp(beta)
    w, h = initial, 0.0
    trace = []
    for target in train_targets:
        error = target - w
        h = (1.0 - alpha) * h + alpha * error  # use old w and H
        w = w + alpha * error
        trace.append((w, h))
    loss = 0.5 * (w - validation_target) ** 2
    hypergradient = (w - validation_target) * h
    return loss, hypergradient, trace
```

<a id="lesson-idbd"></a>

## 3. IDBD：从多步链式法则到每个特征一个步长

IDBD 考虑线性监督预测：输入 $x_t$、标签 $y_t$、预测 $w_t^\top x_t$，误差为 $\delta_t=y_t-w_t^\top x_t$。每个特征 $i$ 有自己的步长 $\alpha_i=\exp(\beta_i)$。一个方向的更新持续同向，可能表示跟踪速度不足；频繁反向，则可能是更新过冲。IDBD 通过权重对过去步长的敏感度 $h_i$ 估计这种关系，而不只统计梯度幅度。

$$
\delta_t=y_t-w_t^\top x_t,\qquad w_{i,t+1}=w_{i,t}+\alpha_{i,t}\delta_t x_{i,t}
$$

这是内层 LMS/线性 SGD 更新。所有特征共享误差 $\delta$，但各坐标的 $\alpha_i$ 不同，因此改变的不只是总更新长度，还包括学习的几何方向。

精确敏感度是矩阵 $\partial w_j/\partial\beta_i$：改变一个权重会改变公共误差，再影响其他权重。IDBD 只保存 $h_i\approx\partial w_i/\partial\beta_i$，忽略 $i\ne j$ 的交叉影响。在固定 $\beta$ 的对角近似下，对权重更新求导得到保留因子 $1-\alpha_i x_i^2$，以及直接作用项 $\alpha_i\delta x_i$。

$$
\begin{aligned}\beta_i^+&=\beta_i+\mu\delta x_i h_i,\qquad \alpha_i=e^{\beta_i^+},\\w_i^+&=w_i+\alpha_i\delta x_i,\\h_i^+&=h_i\max(0,1-\alpha_i x_i^2)+\alpha_i\delta x_i.\end{aligned}
$$

$\delta$ 来自更新前的权重，$\beta$ 使用旧 $h$ 更新，之后才计算新 $\alpha$、$w$ 与 $h$。正截断避免保留因子为负，是稳定化操作而非精确链式法则。$\mu$ 是另需指定的元步长。

**算法：算法伪代码**

1. 初始化 $w=0$、$h=0$、$\beta_i=\log\alpha_{i,0}$。
1. 每个样本 $(x,y)$：
  1. $\delta\leftarrow y-w^\top x$              # 更新前的预测误差
  1. $\beta_i\leftarrow\beta_i+\mu\delta x_i h_i$ # 使用旧敏感度
  1. $\alpha_i\leftarrow\exp\beta_i$
  1. $w_i\leftarrow w_i+\alpha_i\delta x_i$
  1. $h_i\leftarrow h_i\max(0,1-\alpha_i x_i^2)+\alpha_i\delta x_i$
1. 用更新前误差评价当前样本，下一样本再评价学习后的预测。

初始 $h=0$，第一步元更新为零，因为尚无过去步长影响当前权重的历史。普通权重更新产生非零 $h$ 后，后续误差才能评价这种影响。特征尺度、强相关性、噪声与过大的元步长 $\mu$ 都可能使适应不稳定，因此仍需监测步长及敏感度。

线性监督预测的 IDBD：保存旧误差与旧敏感度，依次更新对数步长、权重和敏感度。

```python
def idbd_step(weights, beta, history, features, target, meta_rate=0.01):
    """Linear supervised IDBD, with diagonal sensitivity and positive clipping."""
    error = target - sum(w * x for w, x in zip(weights, features))
    new_beta = [b + meta_rate * error * x * h
                for b, x, h in zip(beta, features, history)]
    alpha = [math.exp(b) for b in new_beta]
    new_weights = [w + a * error * x
                   for w, a, x in zip(weights, alpha, features)]
    new_history = [h * max(0.0, 1.0 - a * x * x) + a * error * x
                   for h, a, x in zip(history, alpha, features)]
    return new_weights, new_beta, new_history, error
```

<a id="lesson-tidbd"></a>

## 4. TIDBD：bootstrap 与资格迹改变了什么？

TD 的目标不是外部固定标签 $y$，而是奖励加下一状态的当前预测。因此，对 TD 误差求导时必须指定是否对 bootstrap 目标求导。以下采用 2018 TIDBD Algorithm 1：线性、on-policy、常数 $\gamma$、累积资格迹的半梯度版本。后续 TIDBD 和 AutoTIDBD 的完整误差导数、步长归一化等扩展，需要分别识别。

$$
\begin{aligned}\delta_t&=R_{t+1}+\gamma w_t^\top x_{t+1}-w_t^\top x_t,\\z_t&=\gamma\lambda z_{t-1}+x_t,\\w_{i,t+1}&=w_{i,t}+\alpha_{i,t}\delta_t z_{i,t}.\end{aligned}
$$

$z$ 是内层资格迹，$h$ 是步长敏感度；$\gamma$ 定义预测时间尺度，$\lambda$ 控制信用回溯。资格迹在真实 episode 间如何清零由协议决定，未通知的分布变化不提供额外清零信号。

$$
\begin{aligned}\beta_i^+&=\beta_i+\mu\delta_t x_{i,t}h_{i,t},\qquad\alpha_i=e^{\beta_i^+},\\h_i^+&=h_i\max(0,1-\alpha_i x_{i,t}z_{i,t})+\alpha_i\delta_t z_{i,t}.\end{aligned}
$$

外层局部误差的半梯度使用当前 $x$，内层更新通过资格迹 $z$ 分配信用。因此 $\beta$ 更新中是 $x$，敏感度传播和权重更新中则同时出现 $z$。

$$
\left.\frac{\partial\delta_t}{\partial w_i}\right|_{\rm semi}=-x_{i,t},\qquad\left.\frac{\partial\delta_t}{\partial w_i}\right|_{\rm full}=\gamma x_{i,t+1}-x_{i,t}
$$

若改为完整 TD-error 导数，外层方向涉及 $x_t-\gamma x_{t+1}$，$h$ 的传播因子也相应变化。目标导数约定、敏感度递推与步长归一化应成套指定。

当 $\gamma=0$、旧资格迹不再贡献且 $z=x$ 时，更新退化为 IDBD。真正 terminal 的 bootstrap 为零，可以将下一状态特征置零，在完成本次更新后清空 $z$；采样预算造成的 timeout 不自动等于 terminal。自适应步长本身也不改变 off-policy TD 的稳定性条件。

TIDBD 的全部一转移更新。调用者负责真实 episode 的资格迹边界；脚本不自行猜任务变化。

```python
def tidbd_step(weights, beta, history, eligibility, features, next_features,
               reward, gamma=0.9, lam=0.8, meta_rate=0.01):
    """2018 TIDBD(lambda) semi-gradient version, on-policy, constant gamma.

    This is NOT the later full TD-error-gradient/normalization variants.
    When the protocol declares a genuine new episode, the CALLER clears z.
    """
    value = sum(w * x for w, x in zip(weights, features))
    next_value = sum(w * x for w, x in zip(weights, next_features))
    delta = reward + gamma * next_value - value
    new_beta = [b + meta_rate * delta * x * h
                for b, x, h in zip(beta, features, history)]
    alpha = [math.exp(b) for b in new_beta]
    new_z = [gamma * lam * z + x for z, x in zip(eligibility, features)]
    new_weights = [w + a * delta * z
                   for w, a, z in zip(weights, alpha, new_z)]
    new_history = [h * max(0.0, 1.0 - a * x * z) + a * delta * z
                   for h, a, x, z in zip(history, alpha, features, new_z)]
    return new_weights, new_beta, new_history, new_z, delta
```

<a id="lesson-metatrace"></a>

## 5. Metatrace：过去的哪些步长影响了今天的控制误差

TIDBD 主要从一步预测误差出发。带资格迹的 actor–critic 则把一个控制目标的信用分散到许多时刻：今天的 TD 误差会更新过去状态的价值与动作概率。Metatrace（Young、Wang、Taylor，IJCAI 2019）相应地为步长建立一条元资格迹。区别不只是把 TIDBD 接上 actor，而是外层目标包含各时刻实际使用的不同权重 $w_k$，因此要保留各自的敏感度 $h_k$。

$$
\mathcal J=\tfrac12\sum_k\left[\bigl(\operatorname{sg}(G_k^\lambda)-V_{w_k}(S_k)\bigr)^2-\log\pi_{w_k}(A_k\mid S_k)\operatorname{sg}\bigl(G_k^\lambda-V_{w_k}(S_k)\bigr)\right]
$$

这是用于推导的时变权重半梯度目标：critic 拟合多步 return，actor 使用固定的优势评价。在线 return 写成 $G_k^\lambda=V_{w_k}(S_k)+\sum_{t\ge k}(\gamma\lambda)^{t-k}\delta_t$；求导时按 $\operatorname{sg}$ 约定停止目标路径。权重梯度因此组合为下面的 $\nabla U$。

$$
\begin{aligned}U_w(s,a)&=V_w(s)+\tfrac12\log\pi_w(a\mid s),\\g_t&=\nabla_w U_{w_t}(S_t,A_t),\qquad q=\gamma\lambda,\\z_t&=qz_{t-1}+g_t,\qquad \delta_t=R_{t+1}+\gamma V_{w_t}(S_{t+1})-V_{w_t}(S_t),\\w_{t+1}&=w_t+\alpha_t\delta_tz_t.\end{aligned}
$$

$U$ 把原文选择的 critic 与 actor 权重合并；$1/2$ 来自其联合目标的系数，并非任意 actor–critic 都必须采用的常数。$z$ 是权重更新的资格迹。目标策略随权重变化，以下元导数仍在给定采样路径下计算。

设 $h_k\approx\partial w_k/\partial\beta$，$\alpha=\exp\beta$。对时变权重的多步半梯度目标应用链式法则，再交换时间求和，当前 TD 误差应乘过去各时刻 $g_k^\top h_k$ 的衰减和。这导出第二条迹 $z_{\beta,t}$。如果错误地使用 $z_t^\top h_t$，就把今天的敏感度替换到了过去所有时刻，改变了目标的时间结构。

$$
\begin{aligned}z_{\beta,t}&=\sum_{k=0}^{t}q^{t-k}g_k^\top h_k=qz_{\beta,t-1}+g_t^\top h_t,\\\widetilde\nabla_\beta\mathcal J&=-\sum_t\delta_t z_{\beta,t},\qquad \beta^+=\beta+\mu\delta_tz_{\beta,t},\\h_{t+1}&\approx h_t+e^{\beta^+}z_t\left[\delta_t+\left(\gamma\nabla V_{w_t}(S_{t+1})-\nabla V_{w_t}(S_t)\right)^\top h_t\right].\end{aligned}
$$

外层目标对 return 采取半梯度约定；敏感度递推却使用 $\delta$ 对权重的完整导数，因为它要追踪实际更新怎样依赖步长。递推省略了 $\partial z_t/\partial w$，也未计算动作分布改变的采样路径。即使 critic 线性，策略对数概率一般仍非线性，因而整体不是精确超梯度。

**算法：无熵项、无归一化的标量步长 Metatrace 核心**

1. 初始化 $w$、$\beta$；$z=0$，$h=0$，$z_\beta=0$。
1. 每个转移，用旧 $w$ 计算 $\delta$、$g=\nabla U$ 和 $d=\gamma\nabla V'-\nabla V$。
  1. $z\leftarrow qz+g$
  1. $z_\beta\leftarrow qz_\beta+g^\top h$        # 此时 h 仍是旧敏感度
  1. $\beta\leftarrow\beta+\mu\delta z_\beta$
  1. $\alpha\leftarrow\exp(\beta)$
  1. $h_{\rm new}\leftarrow h+\alpha z(\delta+d^\top h)$
  1. $w\leftarrow w+\alpha\delta z$；$h\leftarrow h_{\rm new}$
1. 真正 episode 结束后清空 $z,z_\beta$；保留 $w,\beta,h$。

手算两步：令 $q=0.5$、$g_0=g_1=1$、$h_0=0$、$h_1=0.1$。第二步 $z_1=1.5$，但 $z_{\beta,1}=0.5\times0+1\times0.1=0.1$，不是 $z_1h_1=0.15$。差别在于第一步发生时敏感度仍为零。延长资格迹不会创造过去并不存在的步长影响。

原文还给出逐参数版本和混合版本：逐参数版本保存对角敏感度，混合版本以 $\alpha_i=\exp(\widehat\beta+\beta_i)$ 结合全局与局部适应。实用算法先把元增量除以运行幅度估计，再限制有效步长；若有熵正则 $\psi\mathcal H(\pi)$，元增量中还包括 $\psi\nabla\mathcal H^\top h$。这些状态必须与权重资格迹分开保存。

$$
\begin{aligned}\Delta_\beta&=\delta z_\beta+\psi\nabla\mathcal H^\top h,\\v&\leftarrow\max\{|\Delta_\beta|,\ v+\mu(|\Delta_\beta|-v)\},\\\beta&\leftarrow\beta+\mu\Delta_\beta/\widetilde v,\qquad \widetilde v=\begin{cases}v&v>0\\1&v=0,\end{cases}\\u&\leftarrow\max\{e^\beta\|g\|^2,\ u+(1-q)(e^\beta\|g\|^2-u)\},\\\beta&\leftarrow\beta-\log\max(u,1).\end{aligned}
$$

这是原文标量归一化的结构。末行只有在有效步长估计超过 1 时才缩小 $\alpha$，不是对任意非线性网络误差的全局收缩定理。加入熵正则时，权重和敏感度更新也需要对应熵项。

$$
\begin{aligned}w^+&=w+\alpha(z\delta+\psi\nabla\mathcal H),\\h^+&\approx h+\alpha\left[z(\delta+\nabla\delta^\top h)+\psi\nabla\mathcal H\right].\end{aligned}
$$

这里 $\alpha$ 使用归一化后的新对数步长。敏感度保留熵更新对步长的直接依赖，但与资格迹的高阶导数一样，省略熵梯度对权重的 Hessian 项。

各状态的生命周期不同。完整标量算法在训练开始时初始化 $h=0$、$v=0$ 和给定的 $\beta_0$；每个 episode 开始时将 $z,z_\beta,u$ 清零，保留权重 $w$、步长参数 $\beta$、敏感度 $h$ 和元增量幅度估计 $v$。其中 $u$ 约束当前资格迹涉及的状态，episode 结束后这些状态不再进入新轨迹；$h$ 则追踪仍被保留的权重如何依赖步长，不应随资格迹一起清空。下面的无归一化代码没有 $u,v$，但保持相同的 $h$ 生命周期。

线性 critic 特例的完整核心；固定步长时敏感度可与有限差分逐项比较，在线更新步长时使用局部近似。

```python
@dataclass
class LinearMetatrace:
    """Scalar, critic-only, unnormalized Metatrace core (no entropy term).

    U(w,x)=V(w,x)=w*x. For mu=0 and fixed data, h is exact dw/d(log alpha).
    With online beta updates, h is the local sensitivity approximation.
    """
    w: float = 0.0
    beta: float = math.log(0.1)
    mu: float = 0.01
    gamma: float = 0.9
    lam: float = 0.8
    z: float = 0.0
    h: float = 0.0
    z_beta: float = 0.0

    def step(self, x, reward, next_x, terminal=False):
        gamma = 0.0 if terminal else self.gamma
        delta = reward + gamma * self.w * next_x - self.w * x
        # Old h belongs to the weights that generated the current prediction.
        old_h = self.h
        self.z = self.gamma * self.lam * self.z + x
        self.z_beta = self.gamma * self.lam * self.z_beta + x * old_h
        self.beta += self.mu * delta * self.z_beta
        alpha = math.exp(self.beta)
        delta_derivative = gamma * next_x - x
        self.h = old_h + alpha * self.z * (delta + delta_derivative * old_h)
        self.w += alpha * delta * self.z
        result = dict(w=self.w, h=self.h, alpha=alpha, delta=delta,
                      z=self.z, z_beta=self.z_beta)
        if terminal:
            # Genuine episodic boundary; parameters and step-size persist.
            self.z = self.z_beta = 0.0
        return result
```

<a id="lesson-meta-rl"></a>

## 6. Meta-gradient RL：根据后续表现选择学习目标

discount 和 $\lambda$ 影响学习目标的时间尺度与偏差—方差权衡。Meta-gradient RL 将这些选择作为内层学习的元参数 $\eta$：先用一段经验更新 agent，再用后续经验和固定评价约定计算元损失。外层并不知道一个监督标签意义上的“正确 $\gamma$”，而是评价由该 $\gamma$ 产生的参数更新。

$$
G_t^\lambda=R_{t+1}+\gamma\left[(1-\lambda)v_w(S_{t+1})+\lambda G_{t+1}^\lambda\right]
$$

在有限 rollout 内从末端向前计算，末端 $G$ 由 bootstrap 值或真正 terminal 的零值初始化。固定 $w$、轨迹与末端值，可以单独推导 return 对超参数的直接导数。

$$
\begin{aligned}\frac{\partial G_t^\lambda}{\partial\gamma}&=(1-\lambda)v_{t+1}+\lambda G_{t+1}^\lambda+\gamma\lambda\frac{\partial G_{t+1}^\lambda}{\partial\gamma},\\\frac{\partial G_t^\lambda}{\partial\lambda}&=\gamma\left(G_{t+1}^\lambda-v_{t+1}+\lambda\frac{\partial G_{t+1}^\lambda}{\partial\lambda}\right).\end{aligned}
$$

改变 $\gamma$ 或 $\lambda$，既影响当前组合，也影响后续 return。若用 sigmoid 将它们限制在 $(0,1)$，链式法则还需乘 sigmoid 导数。若末端值依赖 $\eta$，边界导数也必须保留。

将 return 导数代入 actor/critic 的更新函数 $f$，即可得到直接导数 $\partial f/\partial\eta$，再经敏感度递推影响后续参数。原 Meta-gradient RL 使用衰减 trace 降低成本：把完整的 Jacobian 传播替换为一个标量衰减。

$$
\begin{aligned}H^+&=\left(I+\frac{\partial f}{\partial w}\right)H+\frac{\partial f}{\partial\eta},\\z^+&=\mu z+\frac{\partial f}{\partial\eta},\qquad\eta^+=\eta-\beta\,z^{+\top}\nabla_{w^+}J_{\rm eval}.\end{aligned}
$$

第二行以标量衰减近似完整 Jacobian 传播；$\mu=0$ 只考虑当前更新。这一节的 $z$ 是元敏感度近似，不是前文的权重资格迹。在线变化的 $\eta$ 又引入局部近似，因此它不等于固定元参数下的全历史总导数。

**算法：算法伪代码**

1. 保存 agent 参数 $w$、元参数 $\eta$ 和元敏感度 $z$。
1. 内层 rollout：
  1. 用 $\eta$ 构造 return，计算更新 $f$。
  1. 计算 $\partial f/\partial\eta$，推进完整 $H$ 或近似 $z$。
  1. $w^+\leftarrow w+f$
1. 后续评价 rollout：
  1. 用固定 $\eta_{\rm eval}$ 计算 $J_{\rm eval}$。
  1. 沿 $w^+$ 对 $\eta$ 的影响更新元参数。
1. 继续交互；元梯度截断与状态重置按预先声明的协议执行。

相邻 rollout 通常不是真正独立样本，验证经验也受策略变化影响。研究时需要报告数据复用、是否用新策略采样、元梯度截断长度，以及是否将策略采样路径计入估计。训练误差降低本身不足以说明学习规则更适合长期控制。

$\lambda$-return 对 $\gamma$ 和 $\lambda$ 的直接导数：固定边界值，执行后向递推，并由有限差分检验。

```python
def lambda_return_sensitivity(rewards, next_values, bootstrap, gamma=0.9, lam=0.8):
    """Lambda return and exact direct derivatives for fixed values/boundary."""
    if len(rewards) != len(next_values):
        raise ValueError("one next-state value is required per reward")
    g, dg, dl = bootstrap, 0.0, 0.0
    for reward, value in reversed(list(zip(rewards, next_values))):
        mixture = (1.0 - lam) * value + lam * g
        new_dg = mixture + gamma * lam * dg
        new_dl = gamma * (g - value + lam * dl)
        g = reward + gamma * mixture
        dg, dl = new_dg, new_dl
    return g, dg, dl
```

<a id="lesson-output-steps"></a>

## 7. 在线元梯度与预设更新规则的边界

步长随数据变化，不足以判断一个方法是否在学习更新规则。IDBD、TIDBD 和 Metatrace 保存参数对过去步长的敏感度，用后续误差评价这种影响，再改变步长参数。Stream-X 与 Intentional Updates 则按预先规定的归一化或输出变化公式计算本次更新，不估计这条元敏感度。它们属于流式更新稳定化方法，可以作为元学习的对照或与之组合。

| 机制 | 保留的历史 | 本次更新回答的问题 |
| --- | --- | --- |
| 资格迹 | 过去输出梯度的衰减统计 | 当前误差应分给哪些过去预测或决策？ |
| Stream-X / Intentional 的尺度控制 | 梯度、误差或完整增量的幅度统计 | 在给定方向上，本次参数或输出允许改变多少？ |
| 在线元梯度 | 内层参数对学习规则参数的敏感度 | 过去的规则选择怎样影响了后续误差，应如何调整？ |

流式学习章完整推导 ObGD、StreamingOptimizer 与 Intentional-AC，并保留实现、手算及边界检查。时间信用分配章讨论误差在时间上的分配。这里继续研究学习规则本身的适应；这三个问题可以在同一个 agent 中共同出现，但不是同一种算法分类。

<a id="lesson-maml"></a>

## 8. MAML：学习一个适合少量更新的初始化

MAML 换了问题设定：从任务分布 $p(T)$ 采样任务，每个任务允许收集少量适应经验，再评价适应后策略。外层变量是共享初始化 $\theta$。每个任务从同一初值复制出 $w_i$，执行若干梯度更新；外层评价这些更新后的表现。直接平均各任务未适应的损失，则是不同的多任务学习目标。

$$
\begin{aligned}w_i&=\theta-\alpha\nabla_\theta L_i^{\rm train}(\theta),\\\mathcal J(\theta)&=\mathbb E_{T_i\sim p(T)}\left[L_i^{\rm test}(w_i)\right],\\\nabla_\theta\mathcal J_i&=\left(I-\alpha\nabla^2_\theta L_i^{\rm train}\right)^\top\nabla_{w_i}L_i^{\rm test}.\end{aligned}
$$

train/test 在这里指同一任务内的适应与评价数据，不是把最终测试任务反复拿来调外层参数。两阶段数据区分能避免只优化对同一批样本的过拟合。

多步版本重复 $w^{k+1}=w^k-\alpha\nabla L^k$，元梯度经过每一步 Jacobian 的乘积。FOMAML 将曲率修正近似为单位矩阵，仍然使用内层适应后的评价梯度。Reptile 则令 $\theta\leftarrow\theta+\varepsilon(w_i-\theta)$，将初始化向任务内训练后的参数移动；两者都避免显式二阶导数，但并非同一更新规则。

**算法：算法伪代码**

1. 每轮外层更新：
  1. 从训练任务分布采样任务 $T_i$。
  1. 对每个任务复制初始化：$w_i\leftarrow\theta$。
    1. 收集适应数据 $D_{\rm train}$，估计策略梯度损失。
    1. 执行 $K$ 次内层更新，保留计算图。
    1. 用适应后的策略收集 $D_{\rm test}$，计算外层损失。
  1. 合并外层梯度，沿内层更新反传到 $\theta$。
1. 测试新任务：从 $\theta$ 出发，按相同适应预算更新 $w$，不更新 $\theta$。

在 RL 中，对策略梯度 surrogate 再做自动微分，并不自动得到期望元回报的正确高阶导数。适应轨迹的分布依赖 $\theta$，评价轨迹的分布依赖 $w_i$；数值相同的 surrogate 可能具有不同的高阶导数。因而需要明确 likelihood-ratio、baseline 和 stop-gradient 的位置。二次函数实验可以隔离优化器链式法则，下面的随机例子再分析采样路径。

解析二次任务：完整 MAML 与 FO 近似的数值差别可直接检查。

```python
def scalar_maml(initialization, train_target, test_target, curvature=2.0, alpha=0.1):
    """One task, deterministic quadratic losses, exact MAML and FO approximation.

    L_train = curvature/2*(w-train_target)^2; L_test = (w-test_target)^2/2.
    This illustrates optimizer differentiation, not policy-gradient sampling.
    """
    adapted = initialization - alpha * curvature * (initialization - train_target)
    outer_gradient = adapted - test_target
    exact = (1.0 - alpha * curvature) * outer_gradient
    first_order = outer_gradient
    return 0.5 * outer_gradient ** 2, exact, first_order, adapted
```

$$
\nabla_\theta\mathbb E_{D\sim p_\theta}[J(U_\theta(D))]=\mathbb E_D\left[\left.\nabla_\theta J(U_\theta(D))\right|_D+J(U_\theta(D))\nabla_\theta\log p_\theta(D)\right]
$$

第一项在给定适应数据下求导，第二项反映初值改变了适应数据的分布。若 $J$ 还通过新策略采样估计，它自身也需要正确的策略梯度；sampled reward 的数值通常没有可直接反传的动作导数。

下面穷举一个 Bernoulli 动作，比较只保留更新路径与同时保留采样 score 项。由于动作只有两种，期望可以精确求和，有限差分不含 Monte Carlo 噪声。DiCE 等工作进一步处理随机计算图的高阶梯度；它们解决的是估计器问题，而不是替代 MAML 的内外层目标。

固定数据微分遗漏了什么：用一个能精确穷举的随机适应过程检查两条梯度路径。

```python
def stochastic_adaptation(theta, alpha=0.2, target=1.0):
    """Enumerate a Bernoulli sample's TWO gradient paths exactly.

    a~Bernoulli(sigmoid(theta)); w=(1-alpha)*theta+alpha*a.
    The toy outer loss is (w-target)^2/2. No Monte Carlo noise is present.
    """
    probability = 1.0 / (1.0 + math.exp(-theta))
    expected_loss = path_gradient = score_gradient = 0.0
    for action, mass in [(0.0, 1.0 - probability), (1.0, probability)]:
        adapted = (1.0 - alpha) * theta + alpha * action
        loss = 0.5 * (adapted - target) ** 2
        expected_loss += mass * loss
        path_gradient += mass * (adapted - target) * (1.0 - alpha)
        score_gradient += mass * loss * (action - probability)
    return expected_loss, path_gradient, score_gradient
```

<a id="lesson-context"></a>

## 9. RL² 与 PEARL：在活动状态中进行任务推断

另一种办法是把“学到了哪个任务”放进活动状态或潜变量，而把推断规则留在固定网络权重里。RL² 将前一动作、奖励和 episode 结束信号与当前观测输入 RNN；内层适应发生在 hidden state 的变化，外层用 RL 训练整条 trial 的累计表现。若一个 trial 包含同一任务的多个 episode，hidden state 在 episode 间保留，在新的任务 trial 开始才重置；这是任务协议的一部分。

$$
h_{t+1}=f_\theta(h_t,O_{t+1},A_t,R_{t+1},d_{t+1}),\qquad A_{t+1}\sim\pi_\theta(\cdot\mid h_{t+1})
$$

测试时可以冻结 $\theta$，而 $h$ 仍随经验改变。活动变化支持探索、记忆和适应，也可以与参数学习组合；两种变化应分别记录。

**算法：算法伪代码**

1. RL² 元训练：
  1. 采样训练任务，初始化 trial 活动状态 $h$。
  1. 在同一任务运行多个 episode；环境重置时保留跨 episode 的 $h$。
  1. 将已到达的动作、奖励和 done 输入 RNN，累计 trial 回报。
  1. 用序列梯度更新共享参数 $\theta$。
1. 测试：冻结 $\theta$；新任务 trial 才初始化 $h$，适应协议与训练一致。

PEARL 引入任务潜变量 $z$，从转移集合 $c$ 推断 $q_\phi(z\mid c)$，策略和价值函数均以 $z$ 为条件。其 context encoder 将各条转移编码成高斯因子后相乘，因此不依赖 context 内转移的排列。这适合任务在 context 范围内固定的设定；混合变化前后的证据，可能推断出并不存在的折中任务。

$$
q_\phi(z\mid c)\propto\prod_{\xi\in c}\Psi_\phi(z\mid\xi),\qquad \sigma^{-2}=\sum_j\sigma_j^{-2},\qquad \mu=\sigma^2\sum_j\frac{\mu_j}{\sigma_j^2}
$$

右式是单坐标高斯因子乘积，其他坐标可逐项计算；作者实现还通过 KL 正则将编码分布约束到先验附近。不要在已有因子定义之外又无说明地重复乘同一个先验。

$$
\begin{aligned}L_Q&=\mathbb E\left[\left(Q_\omega(s,a,z)-\operatorname{sg}(y)\right)^2\right],\\y&=r+\gamma\,\bar V(s',z),\\L_{\rm enc}&=L_Q+\beta_{\rm KL}\,D_{\rm KL}\bigl(q_\phi(z\mid c)\Vert p(z)\bigr).\end{aligned}
$$

原实现使用 SAC 的 $V$ 网络版本：慢更新网络提供目标价值，terminal 时停止 bootstrap。编码器由 critic 误差与 KL 训练；原作者代码在 policy 通路对 $z$ detach，使该通路不直接更新 encoder。

**算法：算法伪代码**

1. PEARL 元训练：
  1. 为训练任务维护 replay；从 context 推断 $q_\phi(z\mid c)$ 并采样 $z$。
  1. 用 $\pi(a\mid s,z)$ 交互，将经验加入该任务 replay/context。
  1. 采样 critic 训练转移和 context，生成可重参数化的 $z$。
  1. 更新 $Q$ 与 encoder（包含 KL）；更新条件策略和价值/target 网络。
1. 元测试：冻结网络参数，从先验采样 $z$，交互收集 context。
  1. 重新推断 posterior 并采样 $z$；不输入真实任务参数。

解析例子采用 $y=z+\epsilon$，先验 $z\sim\mathcal N(0,1)$，独立噪声方差为 1。没有证据时后验仍为先验；一条 $y=1$ 后为 $\mathcal N(0.5,0.5)$；三条相同观测后为 $\mathcal N(0.75,0.25)$。即使模型参数不变，条件推断仍改变预测。这说明 context adaptation 的性质；PEARL 另外需要学习推断网络及控制策略。

已知高斯任务的可运行 context 推断；明确展示先验与观测因子的不同角色。

```python
def gaussian_context(observations, noise_variance=1.0, prior_mean=0.0,
                     prior_variance=1.0):
    """Exact Bayesian latent-task inference for y_i=z+Gaussian noise.

    It demonstrates context-driven adaptation with NO parameter gradient.
    It is not PEARL's learned encoder or SAC training loop.
    """
    if noise_variance <= 0 or prior_variance <= 0:
        raise ValueError("variances must be positive")
    precision = 1.0 / prior_variance + len(observations) / noise_variance
    natural_mean = prior_mean / prior_variance + sum(observations) / noise_variance
    return natural_mean / precision, 1.0 / precision
```

<a id="lesson-learned-rules"></a>

## 10. 学习完整更新规则：LPG、DiscoRL 与外层优化

单一标量步长在给定方向下缩放更新，逐参数步长还会改变更新的几何方向。更一般的 learned update rule 让元网络读取奖励、动作、agent 输出、预测误差或训练统计量，产生目标、损失系数或更新信号。agent 参数 $w$ 按内层流程变化，元网络 $\eta$ 根据训练轨迹的外层表现优化。状态构造、预测目标与学习规则可以联合学习，但分别具有自己的状态和梯度路径。

$$
\begin{aligned}(y_t^{\rm target},m_{t+1})&=U_\eta(\xi_{t:t+k},p_{w_t},m_t),\\w_{t+1}&=w_t-\alpha\nabla_w\ell\bigl(p_w,y_t^{\rm target}\bigr),\\\eta^+&=\eta-\beta\nabla_\eta\,\mathcal J(w_{t+K}).\end{aligned}
$$

这是一个通用接口，具体方法的损失仍有差异。$m$ 为元学习器活动状态，$p_w$ 为 agent 的策略或辅助预测。target 对内层 $w$ 和外层 $\eta$ 的导数可以采用不同约定；整块 detach 会同时删除外层希望学习的通路。

**算法：算法伪代码**

1. 元训练：
  1. 采样训练环境与 agent 初值，初始化 learner state。
  1. 重复内层更新：
    1. 采集交互数据，运行 $U_\eta$，得到目标或损失。
    1. 更新 $w$，推进 optimizer 与 meta state。
  1. 用外部回报目标评价训练过程，按指定估计器更新 $\eta$。
1. 部署固定规则：冻结 $\eta$，新 agent 仍按规则持续更新 $w$。
1. 若部署时也更新 $\eta$，另行定义其数据、目标和预算。

DiscoRL（Nature 2025）延续 LPG 的可学习更新规则路线，并扩大算法发现的训练环境集合。其规则不仅选择步长，还利用 agent 的预测输出、TD/advantage 统计量和元网络状态产生训练信号。内层 agent 不断学习，外层改进产生这些信号的规则。评价时需要分别讨论规则在未见环境上的迁移，以及单个 agent 在长时间、非平稳交互中的持续学习；这是两个不同的泛化轴。

这类方法的关键风险是外层训练分布与预算：规则可能记住特定奖励尺度、时间范围、终止模式或架构。除了新环境回报，还应测试更长生命、未通知变化、不同观测尺度、优化器状态持续保留等场景。一个规则在训练 1000 步后很好，不代表继续运行 100 万步还稳定。

<a id="lesson-frontier"></a>

## 11. 前沿问题：元目标短视、规则表示与发现预算

第一条问题是元目标的时间范围。只沿 $K$ 步内层更新求导，容易偏好立即降低误差的更新，而忽视更久以后的表示学习或探索。Bootstrapped Meta-Learning（ICLR 2022）在可微展开得到 $w_K$ 后，继续学习以产生未来目标 $\widehat w$，再让 $w_K$ 在选定的距离下接近这个目标。计算图仍只穿过前 $K$ 步，目标分支停止梯度；额外未来学习需要计算与数据，但不必保存同等长度的反传图。

$$
w_K=F_\eta^{(K)}(w_0),\qquad \widehat w=\operatorname{sg}\bigl(T^{(L)}(w_K)\bigr),\qquad J_{\rm BMG}=D\bigl(\widehat w,w_K\bigr)
$$

$T^{(L)}$ 表示构造目标的额外更新，可与 $F_\eta$ 不同；$D$ 可以是参数距离，也可以是策略分布距离。这样改变的是外层目标的几何与监督范围；梯度仍只穿过前 $K$ 步更新，不穿过未来目标分支。目标如果自身很差，匹配它也未必有益。

第二条问题是外层应该怎样搜索规则。RLC 2025 的 How Should We Meta-Learn Reinforcement Learning Algorithms? 比较黑盒进化学习、神经蒸馏、符号蒸馏和 LLM 代码发现等方法，涉及更新规则、优化器和策略损失等不同对象。其意义在于把“规则的表示形式”与“搜索规则的方法”拆开：可解释公式不一定必须由人工写出，神经更新规则也不一定必须靠元梯度训练。

| 研究条线 | 最小实验设计 | 应回答的机制问题 |
| --- | --- | --- |
| 在线元梯度：IDBD → TIDBD → Metatrace | 固定特征/小网络，比较完整低维敏感度、对角近似和元资格迹；再引入未通知漂移。 | 收益来自步长幅度、跨时间信用，还是额外归一化？同等内存下能否保留？ |
| 短视与目标几何：meta-gradient → BMG | 固定内层算法与总交互，分别增加反传窗口和停止梯度的目标生成窗口。 | 更多未来信息与更多反传计算哪一个更重要？目标滞后造成什么偏差？ |
| 规则发现：LPG → DiscoRL；RLC 2025 方法比较 | 把元训练环境、规则参数量、种子和外层预算独立控制，保留未见任务与长生命测试。 | 学到的是可迁移更新机制，还是某组训练环境/奖励尺度/训练长度的特化？ |

这些实验不应只比较最后一个 checkpoint 的分数。需要同时记录全过程回报、漂移后恢复曲线、参数与步长范数、资格迹统计、每步延迟、持久内存和外层总预算。对算法发现尤其要区分：发现一次规则的成本、训练一个新 agent 的成本，以及部署期间是否继续改规则。原始代码能帮助恢复这些训练协议，不能用单个优化器函数替代整个实验定义。

<a id="lesson-example"></a>

## 12. 两个可手算的元学习过程

算例 A：$w_0=0$，两个训练标签都是 2，$\alpha=0.1$，$\beta=\log(0.1)$，外层标签为 3。内层得到 $w_1=0.2$、$w_2=0.38$。由 $H_0=0$，先有 $H_1=0.2$，再有 $H_2=0.9\times0.2+0.1\times1.8=0.36$。外层损失为 $(0.38-3)^2/2=3.4322$，元梯度为 $(0.38-3)\times0.36=-0.9432$。梯度下降因而增大 $\beta$，使后续学习更快地向正目标移动。

| 两步超梯度方案 | 第二步敏感度 $H_2$ | 外层对 $\beta$ 的梯度 | 改变了什么 |
| --- | --- | --- | --- |
| 精确固定 $\beta$ 多步递推 | 0.36 | −0.9432 | 传播了第一次更新经过第二次更新的影响。 |
| 只保留当前更新，H 从零开始 | 0.18 | −0.4716 | 完全删掉过去影响。 |
| 忽略曲率、直接累计 | 0.38 | −0.9956 | 保留过去，但漏掉第二次更新的 0.9 收缩。 |

算例 B：MAML 初值 $\theta=0$，内层损失 $(w-2)^2$，步长 0.1，外层损失 $(w-3)^2/2$。内层曲率为 2，适应后 $w=0.4$。完整元梯度为 $(1-0.1\times2)(0.4-3)=-2.08$；FOMAML 为 $-2.6$。二者在这个例子中同向但不等；曲率更大时连符号也可能不同。有限差分应扰动初值 $\theta$，再重新执行内层更新。

若评价目标自身可以通过缩短时间尺度改变，元学习器有可能把 $\gamma$ 推向零，以回避长时误差。这说明评价目标应先于训练代理目标定义；降低一个可变目标的数值，不直接意味着改善了原定任务。

<a id="lesson-code"></a>

## 13. 可运行代码与控制变量实验

脚本只依赖 Python 3.10+ 标准库；直接运行即可打印全部手算对应值。

```sh
python3 state_meta_lab.py meta
python3 state_meta_lab.py test
```

基础实验依次给出两步超梯度、IDBD/TIDBD、MAML 与 FO、随机采样的额外梯度项、$\lambda$-return 导数和 context 后验。固定数据的实验用有限差分检验导数；随机例子穷举全部动作以检验期望的两条求导路径。TD 退化关系与边界测试检查更新时序。这里采用低维解析问题，便于观察每一个内部量。

Metatrace 机制实验与配套回归测试；标准库即可运行

```sh
python3 meta_frontier_lab.py metatrace
python3 meta_frontier_lab.py test
```

进阶脚本中的 Metatrace 是无熵正则、无归一化的线性 critic 特例，用来检查元信用的时间索引和敏感度传播。它不包含完整神经网络与环境。复现原论文回报，需要使用作者训练入口、网络、环境版本与统计协议。配套脚本中的 IntentionalStep 属于流式章的输出尺度实验，不作为本章的元学习方法。

完整机制实验驱动代码：每一种“适应”都有独立可观察的对象。

```python
def meta_demo():
    loss, grad, trace = fixed_beta_training(math.log(0.1), [2.0, 2.0], 3.0)
    print("two inner steps (w,H):", trace)
    print("validation loss, d/dbeta:", loss, grad)
    w, b, h = [0.0], [math.log(0.1)], [0.0]
    for target in [2.0, 2.0, -2.0]:
        w, b, h, delta = idbd_step(w, b, h, [1.0], target)
        print("IDBD: target,w,alpha,H:", target, w[0], math.exp(b[0]), h[0])
    result = tidbd_step([0.0, 0.0], [math.log(0.1)] * 2, [0.0] * 2,
                       [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], 1.0)
    print("TIDBD: weights, traces:", result[0], result[3])
    print("MAML: loss, exact, FO, adapted:", scalar_maml(0.0, 2.0, 3.0))
    print("stochastic adaptation loss,path,score:", stochastic_adaptation(0.3))
    print("lambda return, d/dgamma, d/dlambda:",
          lambda_return_sensitivity([1, 2], [0.3, 0.4], 0.4))
    for context in [[], [1.0], [1.0, 1.0, 1.0]]:
        print("context posterior (mean,var):", context, gaussian_context(context))
    print("scope: analytic mechanism tests, not meta-RL benchmark performance")
```

- 修改一：把两个训练标签改为 (2,−2)，保留外层标签 3；记录精确、多步截断与直接累计的符号和幅度，检查“连续梯度同向”的直觉何时失效。
- 修改二：改变 MAML 的内层曲率与步长，让 α×曲率大于 1；解释完整元梯度的符号为什么可能与 FO 相反。
- 修改三：将 context 后半部分的目标从 +1 改为 −1；累积全部证据与只使用近期证据的估计为何不同？这需要推断协议的改变，不是简单调大 actor 步长。
- 修改四：先用完全相同的数据核验导数，再开放策略影响采样。若性能改变，分别检查估计器偏差、探索差异与计算预算，不将它们混为元梯度精度。

<a id="lesson-branches"></a>

## 14. 算法条线与 CRL 的适用条件

| 路线 | 可继续推进的 CRL 问题 | 不能默认继承的假设 |
| --- | --- | --- |
| IDBD/TIDBD | 不同特征、预测问题和环境阶段需要不同适应速度。 | 线性、近似对角敏感度、采样与半梯度稳定条件；仍有元步长。 |
| Metatrace | 带资格迹的在线控制，显式追踪历史步长影响。 | 半梯度外层目标、忽略资格迹高阶导数和策略采样路径；标量/向量/混合变体不同。 |
| Meta-gradient RL | 在线适应回报尺度、λ、学习目标或更新系数。 | 外层评价可信、元梯度时序正确、不会靠改变目标投机。 |
| MAML/FOMAML | 过去任务是否能提供更易适应的参数起点？ | 独立任务采样、可回到共同初值、有清晰适应预算。 |
| RL²/PEARL | 上下文推断能否在固定参数下识别变化？ | 训练任务分布、context 内任务稳定、task/trial reset 权限。 |
| Learned update rules | 能否学到跨环境、跨生命长度的通用学习机制？ | 外层预算、分布外泛化、时间尺度与持久状态稳定性。 |
| 持续元学习 | 外层规则本身也要一生更新，且不能遗忘旧适应能力。 | 不能用无限保留过去所有训练轨迹作为默认实现。 |

这些方法处理不同的适应对象，没有单一的替代关系。上下文推断器可以使用在线自适应步长训练，预测网络的 GVF 问题参数可以接受元梯度，结构化 RNN 也可以成为 learned update rule 的记忆。组合方法时，需要重新定义内层与外层状态、数据来源和导数截断边界，以明确各模块之间传递哪些信息和梯度。

原始实现中，MAML-RL 的 maml_vpg.py 连接适应前后 surrogate 与内层计算图；PEARL 的 agent.py 实现 context、posterior 与潜变量采样，sac.py 组织 critic/encoder 更新；DiscoRL 的 disco.py 定义更新规则接口与持久元状态。检查这些接口后，再恢复配置中的环境、采样量和外层预算，才能比较完整算法。

<a id="lesson-check"></a>

## 15. 诊断、自测与研究起点

| 看到的症状 | 要检查的具体量 |
| --- | --- |
| 元参数不动 | H 是否被清零/截断；target 是否把 η 路径一起 detach；第一步 H=0 是否被误认为 bug。 |
| 步长突然爆炸 | β、α、特征尺度、元梯度范数、clip 触发频率；不能只画平滑后的 return。 |
| 测试适应特别快 | 是否拿到 task ID/边界、重置先验、测试轨迹参与外层更新；与基线权限是否相同。 |
| context 很自信却一直错 | 环境已变化、经验高度相关、后验模型失配；小方差不等于正确推断。 |
| 外层损失下降但控制无改善 | 评价目标是否与实际回报错位；是否只改善同批样本或代理目标。 |
| 精确梯度测试过了但 RL 效果差 | 采样路径、方差、截断偏差、外层过拟合、额外算力，逐项隔离。 |

- 问：Adam 是不是就等于本页的元学习？答：Adam 的统计量改变预设更新规则，但没有在这里所定义的外层后续目标上学习这些规则参数；“有自适应状态”不是充分定义。
- 问：测试时冻结权重就不可能 Meta-RL 适应吗？答：RL² 的活动和 PEARL 的 posterior 仍能变；必须指出变化发生在何处。
- 问：把 IDBD 的 y 换成 TD target 就已经完成 TIDBD 推导吗？答：没有，还要处理 target 导数约定与资格迹，特别是 β 用 x、内层更新用 z 的区别。
- 问：有限差分能证明 RL 元梯度无偏吗？答：只能验证给定可微程序的导数；轨迹分布的导数是否被正确估计，是另一个统计问题。

可开始的研究题：固定预测问题与 feature，先比较常数步长、IDBD/TIDBD 及精确低维敏感度在未通知变化下的终生预测误差，再扩展到 actor-critic。预算应同时限制环境交互和每步计算；变化检测只用于离线统计，不给智能体任务标签。记录变化前误差、恢复积分误差、长期稳定性、步长与 trace 轨迹，并保留元学习无益或有害的情形。这样得到的是可解释的机制研究，而不只是多一个可调超参数的学习曲线。

## 本章的实验设计

外层更新、超参数搜索和基线调参都消耗预算。先锁定开发程序，再用独立运行比较表现。

设定：先用两步标量学习核元敏感度，再选择在线步长学习或跨任务适应协议。两种设定分别声明重置、任务分布与测试权限。

- 完整元梯度与固定随机路径的有限差分一致。
- 第一步零历史敏感度不被误判为实现故障。
- 测试经验不得通过未声明外层更新泄漏到方法选择。

对照：调好的固定率、简单日程与元率；完整导数、截断和停止梯度；相同任务分布及额外计算对照

记录：元目标、敏感度、步长及实际参数写入；适应前后收益和所需数据；元训练、搜索与部署开销

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-selection)

## 学习与研究衔接

IDBD 适应步长，MAML 学初始化，context-based meta-RL 推断任务；内外层目标与数据权限不同。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-meta) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=meta) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=meta)

<a id="chapter-code"></a>

## 下载与运行

多步超梯度、IDBD/TIDBD、标量 MAML 与解析 context；meta_frontier_lab.py 提供 Metatrace 线性特例。

[下载 state_meta_lab.py](https://yingwen.io/zh/continual-rl/download/state_meta_lab.py)

```sh
python3 state_meta_lab.py meta
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [配套章节：时间信用分配](/zh/continual-rl/algorithms/credit-assignment/)：多步回报、资格迹与序列信用；区分权重更新的信用与元参数影响的信用。

- [配套章节：流式学习](/zh/continual-rl/algorithms/streaming/#lesson-output-steps)：数据与计算协议、Stream-X 和 Intentional 的完整尺度推导、实现与对照。

- [Sutton：Adapting Bias by Gradient Descent（IDBD）](https://cdn.aaai.org/AAAI/1992/AAAI92-027.pdf)：原始 IDBD：对数步长、逐特征适应与敏感度历史。

- [Kearney 等：TIDBD 2018 版本](https://arxiv.org/abs/1804.03334)：本页代码具体对应 Algorithm 1 的 semi-gradient、累积迹版本。

- [Kearney 等：Learning Feature Relevance Through Step Size Adaptation](https://arxiv.org/abs/1903.03252)：后续 TIDBD 文献；比较时必须逐式区分版本与归一化/梯度选择。

- [Vivek Veeriah 的 IDBD 实验实现](https://github.com/vivekveeriah/IncrementalDeltaBarDelta)：TIDBD 合作者提供的 IDBD 实验代码；不是 Sutton 1992 官方发布，也不是 TIDBD 完整复现仓库。

- [Young、Wang、Taylor：Metatrace Actor-Critic（IJCAI 2019）](https://www.ijcai.org/proceedings/2019/0581.pdf)：时变权重的元目标、scalar/vector/mixed Metatrace；重点对照元资格迹与敏感度中的完整 TD-error 导数。

- [Metatrace 扩展原文](https://arxiv.org/abs/1805.04514)：含归一化与熵项细节；配套脚本为独立实现的线性特例。

- [Xu、van Hasselt、Silver：Meta-Gradient Reinforcement Learning](https://arxiv.org/abs/1805.09801)：内层更新、后续评价、精确敏感度与衰减 trace 近似的原文。

- [Finn、Abbeel、Levine：MAML](https://proceedings.mlr.press/v70/finn17a.html)：明确任务分布、适应前后目标与可微内层更新。

- [Foerster 等：DiCE](https://proceedings.mlr.press/v80/foerster18a.html)：随机计算图的高阶梯度：普通一阶 surrogate 反复微分可能遗漏或产生错误项。

- [MAML-RL 作者源码](https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/algos/maml_vpg.py)：固定版本的策略梯度适应实现；旧 TensorFlow/rllab 栈，不能假设现代依赖直接兼容。

- [Duan 等：RL²](https://arxiv.org/abs/1611.02779)：循环活动承载快速学习，外层 RL 学习整个 trial 内的适应行为。

- [Rakelly 等：PEARL](https://proceedings.mlr.press/v97/rakelly19a.html)：概率 context、off-policy 数据与任务推断；注意任务稳定范围与 reset 权限。

- [PEARL 作者源码：agent.py](https://github.com/katerakelly/oyster/blob/44e20fddf181d8ca3852bdf9b6927d6b8c6f48fc/rlkit/torch/sac/agent.py)：高斯因子乘积、reparameterized sample、clear_z 与 policy 输入的 detach 实现。

- [PEARL 作者源码：sac.py](https://github.com/katerakelly/oyster/blob/44e20fddf181d8ca3852bdf9b6927d6b8c6f48fc/rlkit/torch/sac/sac.py)：原实现的 Q、V、策略及 encoder/KL 更新时序。

- [Oh 等：Discovering Reinforcement Learning Algorithms](https://arxiv.org/abs/2007.08794)：将学习目标与更新机制本身作为可学习对象的原始工作。

- [Flennerhag 等：Bootstrapped Meta-Learning（ICLR 2022）](https://arxiv.org/abs/2109.04504)：用未来学习产生停止梯度的目标，区分元目标的有效时间范围与反传展开长度。

- [Goldie 等：How Should We Meta-Learn RL Algorithms?（RLC 2025）](https://rlj.cs.umass.edu/2025/papers/Paper218.html)：比较算法发现方法，并同时分析泛化、可解释性、采样成本与训练时间。

- [RLC 2025 算法发现比较：作者代码](https://github.com/AlexGoldie/learn-rl-algorithms)：learning_algorithms 下按 black-box learning、LLM discovery 和两类 distillation 分目录；评价入口独立于发现过程。

- [Discovering state-of-the-art reinforcement learning algorithms（Nature 2025）](https://www.nature.com/articles/s41586-025-09761-x)：DiscoRL 的原论文；区分环境间规则迁移和单一 agent 的持续学习。

- [DiscoRL 作者仓库](https://github.com/google-deepmind/disco_rl)：元训练、评价入口与配置；区分规则发现预算和新 agent 的训练预算。

- [DiscoRL 固定版本更新规则源码](https://github.com/google-deepmind/disco_rl/blob/9059a29f7121d60948f25ef165e08e050e9399c8/disco_rl/update_rules/disco.py)：重点读 agent 输出契约、规则输入、辅助预测目标与持久 meta-state。
