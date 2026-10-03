# Agent state：部分可观测性、递归记忆与在线信用分配

任务目标给定以后，智能体应当保留哪些历史信息，才能预测未来并选择动作？

## 本章内容

- 区分真实环境状态、观测、历史、智能体状态、信念和可学习参数，识别观测混叠。
- 从条件概率推导 belief 更新，从链式法则推导 RTRL、BPTT 和 TBPTT，写清楚各自保留与丢弃的梯度。
- 读懂 RNN、GRU、LSTM、预测状态、GVFN、RTU 分别改变了什么，运行能逐项检查的记忆实验。
- 区分状态更新与参数学习，设计无任务边界、固定资源预算的 CRL 状态构造实验。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 条件概率

对未观察到的变量保留分布；新证据到来后按似然重新加权并归一化。不是把观测当成真实状态。

$$
P(x\mid o)=\frac{P(o\mid x)P(x)}{\sum_{x'}P(o\mid x')P(x')}
$$

### 链式法则

当前状态依赖旧状态，旧状态也依赖参数；求导必须同时包含当前直接影响和历史间接影响。

$$
\frac{d f_\theta(h_\theta)}{d\theta}=\frac{\partial f}{\partial\theta}+\frac{\partial f}{\partial h}\frac{dh_\theta}{d\theta}
$$

### TD 半梯度

把 bootstrap target 暂时视为常数，沿当前预测的梯度更新；这不等于完整地对平方 Bellman 残差求导。

$$
\theta^+=\theta+\alpha\bigl(r+\gamma v_\theta(h')-v_\theta(h)\bigr)\nabla_\theta v_\theta(h)
$$

<a id="lesson-setting"></a>

## 1. 问题设定：当前观测为什么不足以决定动作

任务目标规定哪些结果更好。状态构造规定决策时保留哪些信息。本章固定任务目标，研究第二个问题。智能体状态是可递归更新的历史摘要。它不等于当前观测，也不等于控制器要追求的目标。

设想走廊入口闪一次红灯或蓝灯，随后四个时刻看到的都是同一面灰墙。走到岔路，红灯要求向左，蓝灯要求向右。只接收当前灰墙图像的策略没有任何变量能区分这两段经历；在两种提示等概率、奖励只取决于最终方向时，再大的前馈网络也不能把平均正确率提高到 50% 以上。增加训练步数不能恢复已经被输入接口丢掉的信息。

| 对象 | 记号 | 定义与作用 |
| --- | --- | --- |
| 环境状态 | $X_t$ | 使环境动力学具有 Markov 性的变量，通常不可见；仿真器内部状态不属于默认的策略输入。 |
| 观测 | $O_t$ | 传感器当时给出的信息；同一观测可以来自不同环境状态。 |
| 历史 | $H_t=(O_0,A_0,R_1,\ldots,O_t)$ | 原则上可区分已见过的全部经历，但存储量随生命长度增长。 |
| 智能体状态 | 固定构造器时 $h_t=F_\theta(H_t)$ | 供预测与控制使用的有限历史摘要；可以是向量、分布、预测集或检索记忆。 |
| 学习参数 | $\theta_t$ | 决定状态更新规则的持久参数；与当时的活动状态不是同一种记忆。 |
| 梯度记忆 | $E_t$ | 保存历史如何影响参数梯度；它不同于活动状态，也不同于价值学习的资格迹。 |

$$
h_{t+1}=f_{\theta_t}(h_t,A_t,R_{t+1},O_{t+1}),\qquad A_t\sim\pi_{w_t}(\cdot\mid h_t)
$$

动作由 $h_t$ 产生，执行动作后得到奖励与新观测，再构造 $h_{t+1}$。讨论神经网络时，将动作、奖励和观测编码后合为输入 $u_{t+1}$。选择 $A_t$ 时尚未得到 $R_{t+1}$，时序必须保持一致。

$h_t$ 是当前活动，$\theta_t$ 是活动更新规则的参数。固定 $\theta$ 时，活动仍可随经验变化；这提供记忆，但不是参数学习。在线改变 $\theta_t$ 时，当前活动由过去各时刻的参数共同产生，通常不等于用最新参数重新处理全部历史的结果。分析整个学习器的运行状态时，还需包含参数、优化器统计量和梯度记忆。本章用 $h_t$ 专指感知与决策接口中的历史摘要。

状态有两类不同要求。预测充分性要求摘要保留某类未来预测需要的信息。控制充分性只要求摘要足以实现给定任务的最优行为。全面预测未来通常比完成一个特定任务要求更强。资源有限时，需要明确压缩要保留哪种能力。

$$
P(O_{t+1},R_{t+1}\mid H_t,A_t)=P(O_{t+1},R_{t+1}\mid h_t,A_t)
$$

若条件对所有相关历史和动作成立，且摘要能由自身与新经验按固定规则递归更新，则可递推得到多步动作条件的观测与奖励预测。这里要求所有动作条件，而不只是当前行为策略常见的动作。

$$
\exists\,\bar\pi\quad\text{s.t.}\quad\pi^*(a\mid H)=\bar\pi(a\mid f(H))\quad\text{for all relevant }H,a
$$

这是给定任务的一种控制充分性定义：至少存在一个最优策略 $\pi^*$ 能通过摘要 $f(H)$ 实现。它不要求重建全部未来观测。定义中的目标、可用动作和历史范围都必须固定；任务改变后，原摘要未必仍然充分。

例如未来屏幕背景色可由历史精确预测，但背景色不影响奖励、动作后果或决策。控制摘要可以丢掉它，仍实现最优行为，却不再足以预测完整图像。反过来，在当前策略下把某个价值预测准确，不保证能比较尚未尝试的动作，更不保证支持新的奖励任务。

精确信念提供预测充分状态的参照。RNN 指定可学习的摘要函数族。RTRL 和 BPTT 决定怎样训练参数。预测状态给摘要坐标规定经验语义。这些方法改变不同对象，可以组合，但每种组合仍需检查信息损失和计算成本。

- 允许信息：过去与当前的观测、自己的动作、已经到达的奖励，以及事先声明的模型。
- 不默认允许：环境隐藏状态、任务 ID、变化时刻、随意 reset、将未来奖励输入过去状态。
- 优化目标：最终仍是长期控制表现；预测误差、记忆任务准确率和状态重构误差是诊断，不是同一个目标。

<a id="lesson-derive"></a>

## 2. 有已知模型时，状态构造就是 Bayes 过滤

在有限 POMDP 中，$X_t$ 是隐藏状态，转移矩阵 $P_a(i,j)$ 给出执行动作 $a$ 后从 $i$ 到 $j$ 的概率，似然 $L_a(o\mid j)$ 给出到达 $j$ 后看到观测 $o$ 的概率。信念 $b_t(i)=P(X_t=i\mid H_t)$ 保留完整的状态不确定性，而不是只保留最可能的状态标签。两种信念即使具有相同的最大概率状态，也可能需要不同的辨识动作。

$$
\bar b_{t+1}(j)=\sum_i P_{A_t}(i,j)b_t(i)
$$

第一步预测：尚未用新观测，先将旧分布通过动作条件动力学向前传播。这里求和是对旧隐藏状态边缘化。

$$
b_{t+1}(j)=\frac{L_{A_t}(O_{t+1}\mid j)\bar b_{t+1}(j)}{\sum_k L_{A_t}(O_{t+1}\mid k)\bar b_{t+1}(k)}
$$

第二步校正：以新观测似然重加权预测分布，再除以证据概率。若奖励也包含隐藏状态信息，需要一并条件化；依赖旧状态和新状态的奖励由后面的联合核公式处理。

**算法：观测只依赖到达状态、奖励不另外提供信息时的 Bayes 过滤**

1. 初始化 $b$ 为已知初始分布。
1. 每次转移：
  1. 根据 $b$ 选动作 $a$；执行后获得 $o$。
  1. $\bar b(j)\leftarrow\sum_i b(i)P_a(i,j)$
  1. $u(j)\leftarrow\bar b(j)L_a(o\mid j)$
  1. 若 $\sum_j u(j)=0$，报告模型下不可能的观测。
  1. $b(j)\leftarrow u(j)/\sum_k u(k)$
  1. 用新信念做下一步预测与控制。

其依据是 Markov 条件：历史对下一步的影响先通过 $X_t$，再通过转移传给 $X_{t+1}$，因而知道 $b_t$ 就可以对未知旧状态积分。如果奖励还取决于旧状态，不能简单把它并入只依赖新状态的似然；应直接使用联合核 $P(j,o,r\mid i,a)$。下面的通式也涵盖奖励与观测相关的情况。

$$
b_{t+1}(j)=\frac{\sum_i b_t(i)P(j,O_{t+1},R_{t+1}\mid i,A_t)}{\sum_{k,i}b_t(i)P(k,O_{t+1},R_{t+1}\mid i,A_t)}
$$

已知、正确的有限状态模型使 belief 成为递归充分状态。模型未知、隐状态空间巨大或动力学变化时，仍需解决模型学习、近似推断与跟踪问题。

转移预测、似然校正与不可能观测检查；对应前面的简化观测模型。

```python
def belief_step(prior, transition, likelihood):
    """P[i][j] = P(next=j | current=i, selected_action)."""
    n = len(prior)
    if len(transition) != n or len(likelihood) != n:
        raise ValueError("inconsistent state dimensions")
    if any(len(row) != n for row in transition):
        raise ValueError("transition must be square")
    if any(v < 0 for v in prior + likelihood):
        raise ValueError("negative probability")
    if not math.isclose(sum(prior), 1.0):
        raise ValueError("prior must sum to one")
    if any(any(v < 0 for v in row) or not math.isclose(sum(row), 1.0)
           for row in transition):
        raise ValueError("each transition row must be a probability distribution")
    predicted = [sum(prior[i] * transition[i][j] for i in range(n))
                 for j in range(n)]
    unnormalized = [predicted[j] * likelihood[j] for j in range(n)]
    evidence = sum(unnormalized)
    if evidence <= 0:
        raise ValueError("observation has zero probability under this model")
    return [v / evidence for v in unnormalized]
```

<a id="lesson-recurrence"></a>

## 3. 学习递归状态：RNN、GRU 与 LSTM

不建立完整 POMDP 模型，也可以学习从历史到有用预测的递归压缩。最简单的 RNN 将旧状态与新输入送入非线性函数。设 $h\in\mathbb R^n$、$u\in\mathbb R^d$，$W\in\mathbb R^{n\times n}$、$U\in\mathbb R^{n\times d}$，每步活动更新的运算量约为 $O(n^2+nd)$。读出头既可以预测奖励或未来传感器，也可以输出价值与策略。

$$
z_t=Wh_{t-1}+Uu_t+b,\qquad h_t=\tanh(z_t),\qquad \hat y_t=g_\psi(h_t)
$$

状态构造器 $\theta=(W,U,b)$ 决定保留什么，读出头 $\psi$ 决定如何使用状态。只训练读出头而冻结 $\theta$，是在学习如何使用固定记忆特征。

GRU 和 LSTM 提供可调节的保留、擦除与写入路径。以下 GRU 采用 $z_t$ 越大越保留旧状态的约定；另一些实现使用互补门，或先做隐藏状态线性变换再乘重置门，参数不能直接逐项互换。门函数 $\sigma$ 是 sigmoid，$\odot$ 表示逐元素相乘。

$$
\begin{aligned}z_t&=\sigma(W_z h_{t-1}+U_z u_t+b_z),\\r_t&=\sigma(W_r h_{t-1}+U_r u_t+b_r),\\\tilde h_t&=\tanh(W_h(r_t\odot h_{t-1})+U_hu_t+b_h),\\h_t&=z_t\odot h_{t-1}+(1-z_t)\odot\tilde h_t.\end{aligned}
$$

保留门 $z$ 在旧记忆与候选状态间插值；重置门 $r$ 控制生成候选状态时参考多少旧记忆。门的参数也需要由训练信号学习。

$$
\begin{aligned}i_t,f_t,o_t&=\sigma(W_{i,f,o}h_{t-1}+U_{i,f,o}u_t+b_{i,f,o}),\\\tilde c_t&=\tanh(W_c h_{t-1}+U_c u_t+b_c),\\c_t&=f_t\odot c_{t-1}+i_t\odot\tilde c_t,\\h_t&=o_t\odot\tanh(c_t).\end{aligned}
$$

LSTM 区分 cell 记忆 $c$ 与读出活动 $h$。固定门时，直接保留路径的导数是 $\partial c_t/\partial c_{t-1}=\operatorname{diag}(f_t)$；完整 Jacobian 还包括门通过旧活动依赖旧 cell 的路径。

若 $f_t$ 接近 1，LSTM 的直接记忆通道衰减较慢；若 $z_t$ 接近 0，GRU 可以快速重写状态。但表达长记忆的能力不等于学会长记忆：在长度 8 的片段上训练，仍可能学不会相隔 200 步的因果关系。网络可表示什么、当前活动保留了什么、梯度能否追溯到记忆写入，必须分别分析。

<a id="lesson-rtrl"></a>

## 4. 从链式法则完整推导 RTRL

先固定整段序列使用同一个参数 $\theta$，并把输入序列视为给定。设 $h_t=f_\theta(h_{t-1},u_t)$，初始 $h_0$ 与 $\theta$ 无关。定义 $E_t=\partial h_t/\partial\theta$、$J_t=\partial f/\partial h_{t-1}$、$B_t=\partial f/\partial\theta$；求 $B_t$ 时保持旧状态不变。链式法则将导数分成当前直接作用 $B_t$ 与历史传播 $J_tE_{t-1}$。

$$
E_t=J_t E_{t-1}+B_t,\qquad E_0=0
$$

$E\in\mathbb R^{n\times p}$，$p$ 为参数数目。RTRL 在活动向前推进时同步维护敏感度，目标到达后不需要为求导重放全部旧输入。

$$
\nabla_\theta\ell_t=E_t^\top\nabla_{h_t}\ell_t+\left.\nabla_\theta\ell_t\right|_{h_t}
$$

最后一项只在损失直接使用 $\theta$ 时出现，例如读出与递归共享参数。若递归参数仅通过 $h_t$ 影响损失，这项为零。bootstrap 目标是否 stop-gradient 则决定这里求导的损失。

$$
\begin{aligned}D_t&=\operatorname{diag}(1-h_t\odot h_t),\qquad J_t=D_tW,\\\frac{\partial h_{t,k}}{\partial W_{ij}}&=(1-h_{t,k}^2)\left[\mathbf1_{k=i}h_{t-1,j}+\sum_l W_{kl}\frac{\partial h_{t-1,l}}{\partial W_{ij}}\right].\end{aligned}
$$

权重 $W_{ij}$ 直接连接旧节点 $j$ 与新节点 $i$，也通过旧活动继续传递历史影响。只保留括号内第一项是一步局部梯度。

**算法：RTRL 的前向敏感度计算；在线每步变参时保留递推，但改变了其精确导数解释**

1. 初始化 $h=0$、$E=0$。对固定参数 $\theta$ 的输入序列：
  1. 保存 $h_{\rm old}$、$E_{\rm old}$。
  1. $h\leftarrow f_\theta(h_{\rm old},u)$
  1. $J\leftarrow\partial f/\partial h_{\rm old}$
  1. $B\leftarrow\partial f/\partial\theta$，其中 $h_{\rm old}$ 固定。
  1. $E\leftarrow JE_{\rm old}+B$
  1. 目标到达时，用 $E^\top\nabla_h\ell$ 计算递归参数梯度。
  1. 当前片段结束后更新参数，以核验固定参数的完整梯度。

稠密 $n$ 单元 RNN 通常有 $p\approx n^2$ 个递归参数。敏感度的空间为 $O(np)=O(n^3)$，直接计算 $JE$ 的每步时间为 $O(n^2p)=O(n^4)$。这是普通稠密实现的成本，不是利用结构、稀疏性或随机低秩估计后的必然下界。

在线每步学习时，当前 $h_t$ 是历史上不同参数值产生的，而新参数不会重新计算全部过去活动。沿这条已实现轨迹维护的 RTRL trace，不等于“用当前 $\theta_t$ 重新运行整个历史”的导数。固定参数的有限差分检验可以发现微分实现错误；时变参数条件下，更新频率与状态滞后则是另外的研究问题。

标量 RNN 的三个参数（递归权重、输入权重、偏置）逐项维护敏感度。

```python
def rtrl(theta, inputs, initial=0.0):
    """Exact derivatives for one FIXED parameter vector and fixed initial state.

    h[t] = tanh(a*h[t-1] + b*x[t] + c), theta=(a,b,c).
    Parameters are not changed while processing the sequence.
    """
    a, b, c = theta
    h, eligibility = initial, [0.0, 0.0, 0.0]
    states, derivatives = [], []
    for x in inputs:
        old_h = h
        h = math.tanh(a * old_h + b * x + c)
        local = [old_h, x, 1.0]
        eligibility = [(1.0 - h * h) * (a * e + d)
                       for e, d in zip(eligibility, local)]
        states.append(h)
        derivatives.append(eligibility[:])
    return states, derivatives


def final_loss_grad(theta, inputs, target, initial=0.0):
    states, jacobians = rtrl(theta, inputs, initial)
    error = states[-1] - target
    return 0.5 * error * error, [error * e for e in jacobians[-1]]
```

<a id="lesson-bptt"></a>

## 5. BPTT 与 TBPTT：同一梯度，另一种计算顺序

RTRL 向前传播状态对参数的敏感度；BPTT 保存前向活动，再反向传播损失对状态的伴随量。设总损失为各步损失之和，$a_t$ 表示总损失对 $h_t$ 的导数。从最后时刻开始，逐步加上后续损失经状态递归传回的影响。

$$
\begin{aligned}a_t&=\nabla_{h_t}\ell_t+J_{t+1}^\top a_{t+1},\qquad a_{T+1}=0,\\\nabla_\theta L&=\sum_{t=1}^{T}B_t^\top a_t+\sum_{t=1}^{T}\left.\nabla_\theta\ell_t\right|_{h_t}.\end{aligned}
$$

在相同固定参数、相同目标、相同初始状态和完整序列上，BPTT 与 RTRL 给出同一梯度；区别是保存的中间对象和计算时序，不是优化了不同的数学目标。

完整 BPTT 对 $T$ 步序列的活动存储随 $T$ 增长。TBPTT 只在有限窗口内反传，并在片段边界 detach：状态数值继续传入下一片段，但跨边界的导数被切断。reset 则改变状态数值。二者对应不同实验：detach 限制训练信用范围，reset 还会删除当前已有的活动记忆。

$$
E_t^{(K)}=\sum_{k=0}^{K-1}\left(\prod_{j=0}^{k-1}J_{t-j}\right)B_{t-k}
$$

这是针对当前损失、最近 $K$ 步的截断敏感度；空乘积为单位矩阵，更早项置零。分块 TBPTT 中，各时刻距块边界不同，因此每个损失的有效窗口不一定都等于 $K$。

| 方法 | 保存什么 | 理想化成本 | 遗漏什么 |
| --- | --- | --- | --- |
| 完整 BPTT | T 步活动/计算图 | 时间 O(Tn²)，活动内存 O(Tn)，另加参数 | 固定参数且完整反传时不截断历史梯度。 |
| TBPTT | K 步活动和边界状态 | 每块 O(Kn²)，活动内存 O(Kn) | 跨截断边界的参数影响；数值记忆仍可继续。 |
| 稠密 RTRL | 当前状态与 n×p 敏感度 | 每步 O(n²p)，敏感度内存 O(np) | 固定参数时不截断；在线变参时有历史不一致。 |
| 结构化 RTRL | 受限结构的局部敏感度 | 可显著低于稠密成本 | 先限制递归连接，再获得便宜的精确结构内导数。 |

同一个 RNN 的反向推导；window=2 切断导数而不清空旧状态。

```python
def bptt_final(theta, inputs, target, window=None, initial=0.0):
    """Final-state loss; window=None is full BPTT.

    A finite window holds the earlier boundary state numerically intact but
    treats it as a constant during differentiation (detach, not state reset).
    """
    states, _ = rtrl(theta, inputs, initial)
    a = theta[0]
    left = 0 if window is None else max(0, len(inputs) - window)
    adjoint = states[-1] - target
    grad = [0.0, 0.0, 0.0]
    for t in range(len(inputs) - 1, left - 1, -1):
        previous = initial if t == 0 else states[t - 1]
        dz = adjoint * (1.0 - states[t] ** 2)
        for j, feature in enumerate([previous, inputs[t], 1.0]):
            grad[j] += dz * feature
        adjoint = a * dz
    return grad
```

<a id="lesson-predictive"></a>

## 6. 预测状态与 GVFN：给记忆坐标一个可检验的含义

预测知识与状态摘要是两个对象。知识回答指定的未来问题。状态保存下一个预测或决策所需的信息。知识可以被读出使用，也可以选一部分预测直接作为状态坐标。后一选择形成预测状态结构，但不是所有预测系统都必须这样构造。

普通隐藏向量的第 7 维通常没有独立语义。预测状态的思路是用“如果执行指定行动，之后会观察到什么”的答案表示历史。一个有限 test 指定动作序列和观测序列，其预测是该观测序列在干预动作序列下出现的概率。若一组 core tests 的预测能够推出所有相关 test 的预测，这组数就可作为状态；这是一项充分性条件，不是随便放几个预测头都会满足。

$$
q_i(H_t)=P(o^{(i)}_{1:k_i}\mid H_t,\operatorname{do}(a^{(i)}_{0:k_i-1}))
$$

动作序列在这里是条件干预；概率描述执行该动作序列后出现指定观测的可能性，而非行为策略选择该序列的概率。预测状态避免显式命名隐藏 X，但仍需要学会稳定地更新这些预测。

GVF 用 cumulant $c$、continuation $\gamma$ 和目标策略 $\pi$ 定义预测问题。$c$ 可以是接触、红色像素或另一个定义明确的信号；$\gamma$ 指定预测结束或衰减的方式；$\pi$ 指定假设采取什么行动。将多组 GVF 的预测值直接作为递归状态分量，就是 GVFN 的结构思路。

$$
\begin{aligned}G_t^{(i)}&=c_{t+1}^{(i)}+\gamma_{t+1}^{(i)}G_{t+1}^{(i)},\\v_i(H_t)&=\mathbb E_{\pi_i}[G_t^{(i)}\mid H_t],\\h_t&=f_\theta(h_{t-1},u_t)\approx(v_1(H_t),\ldots,v_n(H_t)).\end{aligned}
$$

例如第 1 个分量预测持续前进直到碰墙前是否见红，第 2 个分量预测随机转向下未来 20 步的碰撞累计量。它们的时间尺度、动作条件和可用监督都不同，不是同一个奖励值函数的复制。

一个直接的半梯度基线将每个分量作为 TD 预测：由同一组旧参数计算 $h_t$ 与 $h_{t+1}$，构造误差，把下一步 target 视为常数，再沿 $h_t$ 的递归梯度更新共享参数。它解释预测语义如何约束记忆。原 GVFN 进一步分析 Bellman 网络投影误差及梯度校正；这些算法针对的目标和稳定性问题比下式更广。

$$
\begin{aligned}\delta_t^{(i)}&=c_{t+1}^{(i)}+\gamma_{t+1}^{(i)}h_{t+1,i}-h_{t,i},\\\theta^+&=\theta+\alpha\sum_i\rho_t^{(i)}\delta_t^{(i)}\nabla_\theta h_{t,i},\qquad \rho_t^{(i)}=\frac{\pi_i(A_t\mid H_t)}{b(A_t\mid H_t)}.\end{aligned}
$$

目标策略与行为策略不同时，动作重要性比是必要的纠偏对象之一，但非线性递归、bootstrapping 与 off-policy 的组合并不因此自动稳定。若目标动作在行为策略下概率为零，问题无法仅靠比率修复。

**算法：算法伪代码**

1. 为每个预测分量明确 c_i、γ_i、π_i；初始化递归网络与所选梯度记忆。
1. 每次真实转移：
  1. 用行为策略 b 选动作，并记录真实选取概率。
  1. 用旧网络状态与新输入计算 h_next；同时推进 RTRL 或 TBPTT。
  1. 逐问题计算 c_i、γ_i、ρ_i，以及 δ_i。
  1. 固定 bootstrap target，组合各分量的梯度更新共享 θ。
  1. 更新控制头；将 h_next 作为后续活动状态。
  1. 定期检查每个预测的校准、方差，以及对控制的增益。

如果只是把 GVF 头挂在一个自由隐藏层后面，那么隐藏状态本身不必等于预测：这是辅助任务结构。两种结构都值得比较，但不能混称。设计研究时还应设置坏问题对照：大量准确却与决策无关的预测，可能耗费预算而不改善控制。预测充分性、易学性和控制实用性需要分别测量。

<a id="lesson-rtu"></a>

## 7. RTU：通过递归结构降低敏感度计算成本

在稠密 RTRL 中，一个参数可经所有递归单元影响其他单元，形成昂贵的敏感度传播。结构化递归让小块独立演化：线性 RTU 的一个块可表示为二维缩放旋转，多块并行运行，由输入投影和输出头混合信息。于是结构内的敏感度具有局部性，可以保留长期递推而降低成本。

$$
\begin{aligned}z_t&=rM(\varphi)z_{t-1}+\sqrt{1-r^2}\,Wu_t,\\M(\varphi)&=\begin{pmatrix}\cos\varphi&-\sin\varphi\\\sin\varphi&\cos\varphi\end{pmatrix},\qquad 0<r<1.\end{aligned}
$$

$z$ 的两个实数分量等价于一个复数递归单元。$r$ 控制衰减，$\varphi$ 控制相位；平方根项归一化输入尺度，与 RL discount $\gamma$ 无关。两个新状态分量都必须使用同一个旧 $z$。

$$
\begin{aligned}e_t^r&=rM e_{t-1}^r+M z_{t-1}-\frac{r}{\sqrt{1-r^2}}Wu_t,\\e_t^\varphi&=rM e_{t-1}^\varphi+rM'(\varphi)z_{t-1}.\end{aligned}
$$

第一式包含输入归一化因子对 $r$ 的导数。每个块维护其两个分量对本地参数的导数；固定输入维度时，这部分成本随块数线性增加。若参数跨块共享，或输入来自另一个可训练递归层，则需要重新分析敏感度结构。

RTU（NeurIPS 2024）沿用 LRU 的结构化递归思路，并针对在线 RL 设计 trace 计算。作者实现对半径与角度进一步参数化，使用自定义梯度接口将敏感度接入学习。非线性的位置很重要：放在读出头与放进递归方程，会产生不同的 Jacobian。配套实验单独实现线性块对 $r$、$\varphi$ 的敏感度，用于检验归一化导数与更新时序。

可检查的旋转块核心：保留归一化导数，并用中心有限差分核验两类敏感度。

```python
def rotation_trace(inputs, radius=0.9, angle=0.4, input_weights=(0.7, -0.2)):
    """One linear RTU-style rotation block and exact r/angle sensitivities.

    Radius is differentiated directly; a paper implementation usually learns
    log-log radius parameters instead. No actor, critic, optimizer, or benchmark
    is included. Both new components use OLD recurrent components.
    """
    if not 0.0 < radius < 1.0:
        raise ValueError("radius must be strictly inside (0,1)")
    cs, sn = math.cos(angle), math.sin(angle)
    norm = math.sqrt(1.0 - radius * radius)
    h, dr, dp = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]

    def rotate(v):
        return [cs * v[0] - sn * v[1], sn * v[0] + cs * v[1]]

    for x in inputs:
        rotated, er, ep = rotate(h), rotate(dr), rotate(dp)
        angle_direct = [-sn * h[0] - cs * h[1],
                        cs * h[0] - sn * h[1]]
        dr = [radius * er[j] + rotated[j]
              - radius / norm * input_weights[j] * x for j in range(2)]
        dp = [radius * ep[j] + radius * angle_direct[j] for j in range(2)]
        h = [radius * rotated[j] + norm * input_weights[j] * x for j in range(2)]
    return h, dr, dp
```

原始仓库把递归网络、actor–critic/PPO 和实验配置分开。linear_rtus.py 中的 carry 同时包含活动与梯度记忆；src/agents 与 src/algorithms 将它接到控制损失；configs 决定部分可观测包装、网络宽度和预算。因此，网络结构能够逐步更新，并不意味着所有使用它的实验都没有 rollout、环境 reset 或其他控制算法状态。

<a id="lesson-online-approximations"></a>

## 8. 降低在线梯度成本：近似导数，还是限制状态结构

降低 RTRL 成本有两类不同选择。其一保留一般递归函数，近似庞大的敏感度 $E$；其二限制递归连接，使该结构内的导数本来就便宜。UORO（ICLR 2018）属于前者，LRU 与 RTU 的局部敏感度计算属于后者。前者承担估计方差或偏差，后者承担表达结构的限制；二者也可以组合。

随机低秩估计的核心可用一个恒等式理解。给定需要累加的两个外积 $ab^\top+cd^\top$，令随机符号 $\epsilon\in\{-1,+1\}$ 等概率取值。下面构造只存两个向量，却在条件期望上等于原来的秩二矩阵，因为交叉项包含均值为零的 $\epsilon$。在采样符号前选定正比例系数 $\rho$，可以调节方差而不改变这一期望。

$$
\widehat E=(\rho a+\epsilon c)(\rho^{-1}b+\epsilon d)^\top,\qquad\mathbb E_\epsilon[\widehat E]=ab^\top+cd^\top
$$

展开后交叉项为 $\epsilon\rho ad^\top+\epsilon\rho^{-1}cb^\top$。这解释随机压缩为何可以保留条件期望，却不保证每一步估计接近真值。将它用于 $E_t=J_tE_{t-1}+B_t$ 还需要处理 $B_t$ 的投影与尺度选择；不能把这个恒等式当成完整 UORO 算法。

梯度估计无偏不保证优化效果良好。长序列中的随机敏感度可能具有很大方差；有限样本训练可能不稳定。相反，TBPTT 有明确的截断偏差，却可能因为方差小、实现高效而表现更好。比较时既要看小网络中的梯度误差与方差，也要看相同计算和内存预算下的控制表现。原始 UORO 论文附有实现材料，方差分析文献进一步解释了这种权衡。

| 状态/训练路线 | 保留的结构 | 主要代价或近似 | 适合怎样的对照 |
| --- | --- | --- | --- |
| GRU/LSTM + TBPTT | 非线性门控递归 | 窗口之外的时间梯度截断；存储窗口内活动。 | 固定前向状态，改变反传窗口，隔离信用分配误差。 |
| 一般递归 + UORO 类估计 | 不要求递归矩阵按小块独立 | 随机敏感度的方差；在线变参仍有轨迹解释问题。 | 小网络完整 RTRL 为参照，分别比较均值、方差与算力。 |
| LRU/RTU + 结构化 trace | 线性/局部块递归，适当输入与输出混合 | 表达族和跨块耦合方式受限。 | 与同结构 BPTT、不同结构 TBPTT 分别比较。 |
| GVFN + 递归梯度方法 | 状态坐标受预测问题约束 | 预测问题是否有用、off-policy 学习是否稳定。 | 固定训练方法，替换状态语义；固定语义，再替换梯度方法。 |

实验可以从短的提示—延迟—决策任务进入 POPGym，再进入屏蔽位置或速度的连续控制。RTU 作者仓库已经包含 POPGym 的 JAX 实现与部分可观测 Brax 包装，适合追踪从记忆诊断到控制的变化。但这些仍有各自的 episode 和观测协议；研究持续状态构造时，应另外设置无任务标签的变化，保持学习器状态连续，并报告资源上限。

<a id="lesson-example"></a>

## 9. 完整算例：证据累计与延迟信用

算例 A：隐藏颜色不变，初始红/蓝各半，看到红提示的概率分别为 0.8 与 0.2。第一次看到红后：未归一化权重为 0.4、0.1，因此红的后验为 0.8。第二次仍看到红：权重为 0.64、0.04，后验为 16/17≈0.941176。若只记最后一个提示，每次都会回到 0.8，无法累计证据；若第二次观测与第一次完全相关而仍当作独立，0.941176 则是错误的过度自信。

$$
\frac{P(X=\mathrm{red}\mid o,o)}{P(X=\mathrm{blue}\mid o,o)}=\frac{0.5}{0.5}\times\frac{0.8}{0.2}\times\frac{0.8}{0.2}=16
$$

这里隐藏状态恒定、两次提示给定颜色后条件独立，所以后验赔率可连续乘似然比。一般有状态转移时必须先执行预测步骤，不能无条件累乘。

算例 B：标量 RNN 为 $h_t=\tanh(0.8h_{t-1}+0.3u_t)$，$h_0=0$，输入为 $(1,0,0,0)$，最终目标为 0.8。前向活动保留了第一步的正信号；完整 RTRL 与 BPTT 都得到损失对输入权重的导数约 $-0.275922$。若只反传最后两步，这两步输入为零，输入权重的梯度也恰好为零。活动记忆仍在，训练信号却无法归因到最初写入记忆的权重。

| 计算 | 递归权重 a 的梯度 | 输入权重 b 的梯度 | 偏置的梯度 |
| --- | --- | --- | --- |
| RTRL / 完整 BPTT | −0.339975 | −0.275922 | −1.792498 |
| 只对最后两步 TBPTT | −0.230183 | 0 | −1.139690 |

用完整梯度训练两种等概率提示，可以得到接近 −0.8 与 +0.8 的预测。该重置式监督任务将表示能力、活动记忆和梯度可达性分开，因而适合诊断训练循环。进入控制问题后，还需要分析探索、bootstrap 目标和状态分布变化，监督任务的结果本身不能替代这些实验。

<a id="lesson-code"></a>

## 10. 从公式到运行循环

下载本页配套脚本后运行；仅需 Python 3.10+ 标准库。

```sh
python3 state_meta_lab.py state
python3 state_meta_lab.py test
```

state 子命令依次运行信念更新、RTRL/BPTT/TBPTT 对照、延迟提示训练和旋转块敏感度。中心有限差分采用 $[L(\theta+\varepsilon)-L(\theta-\varepsilon)]/(2\varepsilon)$，$\varepsilon=10^{-6}$。它检查给定程序的解析导数；预测泛化、在线变参稳定性与控制回报需要另外的评测。

真实训练与评估循环：权重在整个短序列之后更新，状态重置权限显式写在代码中。

```python
def delayed_cue(epochs=1400, seed=3):
    """Controlled, resettable SUPERVISED sequence experiment, not lifelong RL."""
    rng = random.Random(seed)
    theta = [0.8, 0.3, 0.0]
    for _ in range(epochs):
        cue = rng.choice([-1.0, 1.0])
        inputs, target = [cue, 0.0, 0.0, 0.0], 0.8 * cue
        _, grad = final_loss_grad(theta, inputs, target)
        # Sequence starts from h=0; weights change only AFTER its full gradient.
        theta = [w - 0.08 * g for w, g in zip(theta, grad)]
    predictions = [rtrl(theta, [cue, 0.0, 0.0, 0.0])[0][-1]
                   for cue in [-1.0, 1.0]]
    return theta, predictions


def state_demo():
    prior = [0.5, 0.5]
    identity = [[1.0, 0.0], [0.0, 1.0]]
    first = belief_step(prior, identity, [0.8, 0.2])
    second = belief_step(first, identity, [0.8, 0.2])
    print("belief, one/two red cues:", first, second)
    theta, inputs, target = [0.8, 0.3, 0.0], [1.0, 0.0, 0.0, 0.0], 0.8
    loss, forward = final_loss_grad(theta, inputs, target)
    backward = bptt_final(theta, inputs, target)
    truncated = bptt_final(theta, inputs, target, window=2)
    print("RTRL gradient:", forward)
    print("BPTT gradient:", backward)
    print("TBPTT-2 gradient:", truncated, "(cue input-weight gradient is zero)")
    learned, predictions = delayed_cue()
    print("trained cue predictions:", predictions)
    print("rotation block state, d/dr, d/dangle:", rotation_trace([1, 0, 0]))
    print("scope: known-model filtering + fixed-parameter derivatives + supervised sequences")
```

- 实验一：把四步延迟改成二十步；分别记录预测误差、梯度范数和运行时间，不只看最终准确率。
- 实验二：令 TBPTT 窗口从 1 增至序列长度，检查输入权重梯度何时非零。保持相同前向活动，才能隔离反传窗口的影响。
- 实验三：将半径改为 0.5、0.9、0.99；比较衰减时间尺度以及归一化导数的大小。靠近 1 时也要监测数值条件。
- 实验四：引入偶发错误提示，比较最后一次提示、精确 belief 与训练 RNN；不要在 policy 输入中放入生成提示的隐藏颜色。

<a id="lesson-branches"></a>

## 11. 状态构造的研究分支

| 研究问题 | 改变的对象与代表路线 | 关键对照与局限 |
| --- | --- | --- |
| 历史信息是否够用？ | belief、PSR、GVFN：分别用隐状态分布、未来测试概率、长期条件预测描述历史。 | 比较对未训练动作与奖励变化的预测；单一行为策略上的低误差不等于状态充分。 |
| 怎样保存长时信息？ | RNN、GRU、LSTM、结构化状态空间/RTU：改变递归动力学。 | 匹配参数数目、单步算力与记忆；门控结构仍需要可达的训练信号。 |
| 怎样在线分配历史信用？ | BPTT、TBPTT、RTRL、低秩/局部近似、结构化精确 trace。 | 固定参数的梯度正确性与在线时变参数近似需要分别检验；计算吞吐量与统计效率也应分别报告。 |
| 应当预测哪些东西？ | 辅助预测、GVF 问题发现、representation meta-learning。 | 问题选择本身消耗数据与算力；要保留无关预测、随机预测和固定问题的对照。 |
| 状态怎样接入控制？ | recurrent actor-critic、预测状态输入、belief-conditioned policy。 | 冻结控制头测试状态漂移；冻结状态测试控制学习，排查二者相互追逐。 |
| 知识会不会过时？ | 状态构造器持续更新、变化检测、记忆管理。 | 区分正常动力学、环境变化与策略诱导分布变化；未通知变化不提供记忆重置权限。 |

GVFN 作者仓库中的问题定义、递归单元与学习方法应分别阅读：改变 GVF 集合是在改变状态语义，改变递归结构是在改变表示族，改变更新方法是在改变训练信用或稳定性。这三者需要独立消融。将 RTU 与预测状态结合时也相同：低成本长期梯度并不会自动选出有用的预测问题。

<a id="lesson-check"></a>

## 12. 诊断、自测与研究问题

| 现象 | 优先检查 | 不能直接下的结论 |
| --- | --- | --- |
| 同一观测的动作标签冲突 | 是否遗漏历史、动作或奖励；构造对照历史 | “加深网络就能解决”。 |
| 训练预测很好，控制很差 | 预测是否与动作选择相关；是否覆盖反事实策略 | “预测状态没有价值”。 |
| 梯度为零 | 饱和、截断边界、提前 detach、无监督信号 | “环境没有长期依赖”。 |
| replay 后状态失配 | 旧参数存的 h 与当前 θ 是否兼容；是否做 burn-in | “replay 必然不能用于 RNN”。 |
| 换任务就崩溃 | 是否借用了 task reset；状态与参数哪个需要适应 | “模型容量不足”。 |

- 问：RTRL 是否允许无限记忆却不增加运行内存？答：固定维度网络和固定参数维度下，敏感度内存不随时间长度增长；但信息压缩和梯度稳定性仍有限，不能称为无损无限记忆。
- 问：网络权重不更新时，h 变化算不算适应？答：是基于经验的活动/上下文适应，但不是参数学习。比较 RL² 与在线梯度学习时尤其必须区分。
- 问：把边界状态 detach 是否等于智能体忘了过去？答：不等于；detach 只改变求导图。只有数值重置或动力学衰减才直接改变当时存的记忆。
- 问：有一百个准确 GVF，是否已经有充分状态？答：没有这种数量保证；关键是问题集合是否区分了控制相关历史，以及动作条件是否覆盖。

可开始的研究题：在相同单步算力和持久内存下，逐渐增加提示到决策的延迟，再加入未通知的提示规则变化。比较 GRU+TBPTT、结构化递归+trace、手工充分记忆与观测基线。记录终生回报、变化后恢复时间、预测校准、每步延迟与内存。手工记忆是可达上界诊断，不是可学习算法的公平替代；变化时刻只供评测统计，不传入智能体。

## 本章的实验设计

冻结参数后，内部状态仍可继续递推。用这一诊断区分记忆计算与参数适应。

设定：构造两个当前观测完全相同、早期线索不同且正确动作相反的历史。线索只在合法时间出现，研究者真状态用于分析。

- 无历史模型不能凭额外隐藏标签区分这两个历史。
- 固定权重时循环活动仍随观测变化。
- 改变隔离的诊断相位，不改变同一合法输入下的动作。

对照：当前观察、普通 GRU 与候选状态模型；相同持久状态容量的简单递推；冻结权重、清记忆与保留记忆

记录：延迟线索任务的分条件成功率；固定参数分支的行为与记忆读出；状态、敏感度和每步计算量

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-checkpoint)

## 学习与研究衔接

深度网络不能自动消除部分可观测性。必须区分观测编码、历史状态递推与参数学习。

[分册导读](https://yingwen.io/zh/continual-rl/start/deep-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-state) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=state) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=state)

<a id="chapter-code"></a>

## 下载与运行

Bayes 过滤、标量 RTRL/BPTT/TBPTT、重置式监督提示实验、单个线性 RTU 风格旋转块；不是原论文 RL 基准复现。

[下载 state_meta_lab.py](https://yingwen.io/zh/continual-rl/download/state_meta_lab.py)

```sh
python3 state_meta_lab.py state
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Williams & Zipser：RTRL 原始论文](https://doi.org/10.1162/neco.1989.1.2.270)：理解固定参数下的全历史敏感度；本页给出独立的标量推导与实现。

- [Williams & Zipser：递归网络梯度与复杂度](https://web.stanford.edu/class/psych209a/ReadingsByDate/02_25/Williams%20Zipser95RecNets.pdf)：补读精确求导与持续在线权重变化之间的区别。

- [Littman、Sutton、Singh：Predictive Representations of State](https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf)：原始 PSR 研究；关注 core tests 与条件更新，不把任意预测集合都称作充分状态。

- [Abel、Hershkowitz、Littman：Near Optimal Behavior via Approximate State Abstraction](https://proceedings.mlr.press/v48/abel16.html)：按保留的价值、模型和行为性质区分抽象，分析近似抽象造成的控制损失。

- [Schlegel 等：General Value Function Networks](https://arxiv.org/abs/1807.06763)：预测坐标约束、Bellman 网络目标、递归训练与截断敏感性。

- [GVFN 作者仓库](https://github.com/mkschleg/GVFN)：Julia 实验代码，按预测问题、递归网络、学习更新三部分恢复原实验。

- [Elelimy 等：Real-Time Recurrent Learning using Trace Units（NeurIPS 2024）](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e616bde0438cb10cb6adf076ae7d336-Paper-Conference.pdf)：线性/非线性 RTU 的结构、参数化与在线敏感度；结合作者配置辨识控制协议。

- [RTU 作者源码：linear_rtus.py](https://github.com/esraaelelimy/rtus/blob/be54e13b91edcd7988dd1764f8f2d412ca2db856/src/nets/rtus/linear_rtus.py)：固定版本入口：活动与梯度 carry、自定义 VJP、输入投影敏感度。

- [Hochreiter & Schmidhuber：Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735)：门控长期记忆的原始工作；本页采用常用现代门控写法，不把全部工程变体归于原式。

- [Cho 等：Learning Phrase Representations using RNN Encoder–Decoder](https://aclanthology.org/D14-1179/)：GRU 的原始来源之一，给出更新门与候选状态的定义。

- [Tallec & Ollivier：Unbiased Online Recurrent Optimization（ICLR 2018）](https://openreview.net/pdf?id=rJQDjk-0b)：随机低秩敏感度、在线无偏近似及其条件；原论文附有实现材料。

- [Cooijmans & Martens：On the Variance of UORO](https://arxiv.org/abs/1902.02405)：理解低秩无偏估计的方差；无偏性与有限预算训练效果需要分别讨论。

- [Orvieto 等：Resurrecting Recurrent Neural Networks for Long Sequences](https://arxiv.org/abs/2303.06349)：LRU 的线性/对角递归、初始化与尺度设计；长序列建模结果不是 streaming RL 的直接证据。

- [RTU 原始实验与配置](https://github.com/esraaelelimy/rtus)：网络、实时 actor–critic/PPO、POPGym 与部分可观测 Brax 的入口和配置。
