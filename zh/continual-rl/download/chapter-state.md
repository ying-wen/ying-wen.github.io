# Agent state：部分可观测性、递归记忆与在线信用分配

任务目标给定以后，智能体应当保留哪些历史信息，才能预测未来并选择动作？

## 本章内容

- 区分真实环境状态、观测、历史、智能体状态、信念和可学习参数，识别观测混叠。
- 从条件概率推导 belief 更新，从链式法则推导 RTRL、BPTT 和 TBPTT，写清楚各自保留与丢弃的梯度。
- 读懂 RNN、GRU、LSTM、预测状态、GVFN、RTU 分别改变了什么，运行能逐项检查的记忆实验。
- 区分状态更新与参数学习，设计无任务边界、固定资源预算的 CRL 状态构造实验。

<a id="problem-definition"></a>

## 本章的问题定义

当前观测不足以预测后果或选择动作，需要从历史构造有限、可更新的决策信息。

### 给定条件与符号

- 观测、动作、奖励的因果流，以及指定的预测问题或控制评价。
- 状态容量、每步计算预算、递归结构和允许的训练数据。

### 需要求解的对象

可递推的历史摘要及其参数，使声明的后果预测或决策所需信息得到保留；不是重建全部历史。

### 信息与数据权限

$H_t$ 是完整已到达历史；$z_t=f_\phi(z_{t-1},a_{t-1},o_t)$ 是实际保存的摘要，$\phi$ 为状态更新参数。隐藏环境状态不作为免费输入。

$$
\operatorname{Law}(Y\mid H_t=h,a)=\operatorname{Law}(Y\mid z_t=z(h),a)
$$

$Y$ 是本任务指定的未来后果，$a$ 是当前干预动作，$z(h)$ 是历史的状态编码。此式表达相对该后果族的理想充分性；实际网络用预测损失、Bellman目标或控制目标近似检验，不宣称有限状态总能满足它。

### 成立条件与解的含义

- 充分性必须相对后果、未来行为和时间尺度定义；任意多预测坐标不自动构成充分状态。
- RTRL固定参数全历史敏感度、在线参数变化和截断BPTT分别说明，不能共用精确梯度称谓。

判断准则：在同观测但不同历史的别名反例上保持不同预测/动作；用匹配历史探针测预测误差，敏感度对固定参数有限差分吻合，并报告内存与每步时间。

### 适用边界

- 资格迹不能代替行动时使用的记忆状态。
- 状态充分性或预测精度不直接证明控制最优。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)：GVF可提供预测坐标，但其题目集合是否保留决策信息仍需验证。

- 组合不同学习问题 · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)：状态构造处理存什么；递归敏感度处理后来的误差怎样更新早先记忆参数。

- 改变信息或数据协议 · [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)：严格流式预算限制保存历史及展开计算图，影响可选递归结构和导数近似。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

历史含有有用线索但持续增长；学习记忆又需要计算参数通过过去活动影响当前输出的路径。

### 本章的核心思路

先指定摘要必须保留的后果，再选择递归结构及其可负担的敏感度计算。

1. [从历史别名识别缺失信息](#lesson-derive)：因为同一观测可对应不同未来，先用历史条件分布和预测状态区分需要保留的线索。

2. [把活动与参数求导分开](#lesson-rtrl)：因为递归活动是运行状态而参数是学习对象，RTRL分别递推活动和全历史敏感度。

3. [按预算选择结构或导数近似](#lesson-online-approximations)：因为一般敏感度昂贵，RTU限制耦合结构，BPTT/UORO分别截断或压缩导数；误差和资源分别检验。

结论与条件：敏感度恒等式限定在所写固定参数/计算图；RTU成本依赖局部结构，近似导数的无偏性不保证低方差或回报增益。

### 相关方法改变了什么

- 信念状态或PSR：有相应模型/可识别性条件时定义充分信息，未必低成本可学。

- RTRL/RTU：递推敏感度；RTU通过结构降低成本而非通用精确RTRL的免费替代。

- 截断BPTT/UORO：分别丢弃长路径或随机压缩导数，产生不同偏差与方差。


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
| 完整 BPTT | T 步活动/计算图 | 时间 $O(Tn^2)$，活动内存 $O(Tn)$，另加参数 | 固定参数且完整反传时不截断历史梯度。 |
| TBPTT | K 步活动和边界状态 | 每块 $O(Kn^2)$，活动内存 $O(Kn)$ | 跨截断边界的参数影响；数值记忆仍可继续。 |
| 稠密 RTRL | 当前状态与 n×p 敏感度 | 每步 $O(n^2p)$，敏感度内存 $O(np)$ | 固定参数时不截断；在线变参时有历史不一致。 |
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

1. 为每个预测分量明确 $c_i,γ_i,π_i$；初始化递归网络与所选梯度记忆。
1. 每次真实转移：
  1. 用行为策略 b 选动作，并记录真实选取概率。
  1. 用旧网络状态与新输入计算 `h_next`；同时推进 RTRL 或 TBPTT。
  1. 逐问题计算 $c_i,γ_i,ρ_i$，以及 $δ_i$。
  1. 固定 bootstrap target，组合各分量的梯度更新共享 θ。
  1. 更新控制头；将 `h_next` 作为后续活动状态。
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

<a id="research-memory-training-interface"></a>

## 研究专题 A · Memoroids：记忆能保存多久，梯度能学习多久？

“隐状态能保留很久”和“参数能从早期事件学到什么”是两个问题。前者取决于递归动力学，后者还取决于训练中的梯度路径。即使活动保留了线索，在短块之间停止梯度，也可能无法教会网络哪些输入值得记住。Memoroids（NeurIPS 2024）提供一个重要对照：保持线性递归模型，改变长序列运算和训练批处理方式。

$$
h_t=A_t h_{t-1}+b_t,\qquad (A_1,b_1)\star(A_2,b_2)=(A_2A_1,A_2b_1+b_2)
$$

每次输入产生仿射变换，第二个变换作用在第一个之后。单位元为 (I,0)，一般不交换；对角 A 可降低合并成本。

$$
((A_1,b_1)\star(A_2,b_2))\star(A_3,b_3)=(A_3A_2A_1,A_3A_2b_1+A_3b_2+b_3)
$$

两种括号顺序结果相同，所以可以用树状 scan 计算前缀。固定大小合并的并行深度可为 O(log T)，总工作量仍为 O(T)；稠密矩阵乘法成本没有消失。

标量例子：$h_0=0$，三个变换为 $(0.5,1)$、$(0.5,2)$、$(0.5,3)$。逐步得到 $1,2.5,4.25$；组合为 $(0.125,4.25)$。若第三次输入来自新的独立回合，规定状态归零，则改成 $(0,3)$，前面活动被屏蔽，结果为 3。非零初始状态则将 $A_t h_{\rm init}$ 吸收入新偏置。

Tape-Based Batching 将多个完整回合存入一条 tape，以 begin/reset 信息阻断回合之间的状态传递，减少固定分段的补零和梯度截断。它仍保存序列并执行反传，既不属于严格逐步 RTRL，也不能让任意非线性门控模型采用同样的结合运算。

**算法：验证方案；不是论文实验已在本地运行的报告**

1. 固定一个早期线索、延迟决策的 POMDP和同一记忆模型
1. A：固定长度片段，保留活动但停止跨片段梯度
1. B：完整回合 tape，用 begin 标记阻断真实回合边界
1. C：允许相同数据协议时，对照结构化在线敏感度
1. 记录决策表现、早期观测敏感度、训练峰值内存与每步延迟

| 对象 | 作者实现入口 | 检查 |
| --- | --- | --- |
| 结合递归与边界 | proroklab/memoroids 的 memory、modules.py | 逐步、scan 与 begin 标记是否一致？ |
| 存储与切分 | buffer.py、segment_dqn.py、tape_dqn.py | 采样是否含完整回合？活动缓存来自哪版参数？ |
| 损失与回报 | losses.py、returns.py | target、padding 和终止定义是否匹配？ |

CRL 的进一步问题是参数变化时的状态一致性：实际活动由历次参数生成，从完整历史用当前参数重算则得到另一状态。缓存旧活动、burn-in、tape 重算和实时敏感度各有资源与近似边界。先固定参数验证代数等价，再开放参数更新测状态差异与控制表现；固定参数的恒等式不能证明不同学习时序的智能体等价。

<a id="research-state-query-sufficiency"></a>

## 研究专题 B · 状态充分性要相对于未来问题检验

把一张画面编码成漂亮的潜在向量，与从历史形成足够预测和行动的 agent state，是两个问题。DINO-WM 使用冻结视觉 patch 特征加观测历史做动作后果预测，V-JEPA 2-AC 也利用视频表示与动作条件预测器。它们提醒我们先检查表示接口中有哪些历史与运动信息，再讨论“模型理解了世界”。单帧自监督特征本身不保证隐藏速度、门锁状态或先前指令可被恢复。

$$
\begin{aligned}H_t&=(O_0,A_0,R_1,O_1,\ldots,A_{t-1},R_t,O_t),\quad Z_t=f_\theta(H_t),\\\mathcal Q&=\{(\pi_q,C_q,\gamma_q)\},\qquad v_q(h)=\mathbb E_{\pi_q}[G_q\mid H_t=h],\\Z(h)=Z(\tilde h)&\ \Rightarrow\ \begin{cases}\pi_q(\cdot\mid h)=\pi_q(\cdot\mid\tilde h),\\v_q(h)=v_q(\tilde h),\end{cases}\quad\forall q\in\mathcal Q.\end{aligned}
$$

这是目标策略可由该状态执行、且状态对查询族理想充分的联合规格，不是上述论文对任意历史的保证。每个 q 必须在两条历史上采用同一声明的策略规则；若规则要求已被状态丢弃的历史信息，先违反的是行为条件兼容性。查询族越丰富，允许丢弃的信息通常越少，但有限查询相容不等于任意控制的 Markov 充分性。

例子：两个历史都以“机器人站在门前”结束，但其中一个历史刚执行过解锁。若预测任务只问附近墙壁颜色，两种历史可以具有完全相同的正确答案；若增加“执行推门动作能否通过”的查询，二者必须区分。增加与门锁无关的预测数量无法补回这个信息。这个例子也说明，预测充分性必须包括行为条件，单纯预测自然视频的下一帧不足以验证反事实动作后果。

| 表示路线 | 怎样形成训练信号 | 须另外检验的条件 |
| --- | --- | --- |
| GVF / successor 表示 | 指定策略和时域下的未来量或占用 | 目标策略覆盖、查询族是否区分控制相关历史 |
| HILP 时间距离表示 | 离线目标价值约束潜在距离 | 有向距离、有限维嵌入和历史混叠 |
| DINO-WM 冻结视觉表示 | 动作条件的未来 patch 特征 | 视觉特征是否保留任务事件，历史窗口是否足够 |
| V-JEPA 视频与 2-AC | 视频潜在预测，再学机器人动作条件预测 | 动作坐标语义、预训练与部署视角、记忆跨度 |

**算法：从表示诊断推进到控制检验；不是将隐藏状态作为训练输入**

1. 受控状态实验（拟议）：
  1. 构造同观测、不同历史的成对起点；只把观测/动作历史交给 agent
  1. 固定未来行为策略，收集成对的后续轨迹
  1. 分别用单帧特征、固定历史窗口、可学习递归状态预测同一查询集
  1. 在冻结副本上测：隐藏条件可读性、未来预测误差、线性/非线性读出差
  1. 再用相同控制器和计算预算测试动作选择与全程回报
  1. 分开改变观测外观、隐藏动力学和奖励，保留失败结果

研究空缺在于：一个持续变化的查询族怎样反过来帮助构造状态，又怎样发现当前查询尚未区分的历史？可以提出新问题后只使用新发生的轨迹检验，但这仍需要合适的行为覆盖与在线信用分配。SF² 的状态动作特征、HILP 的离线距离、视频预训练都不能直接替代这一递归状态构造过程。

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

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

状态是支持后续计算的内部信息；GVF 指定一个预测问题；RTRL 和资格迹规定信用如何传播。三者可以组合，但不是相互替代的算法名称。先理解给定策略的预测，再讨论预测怎样改善控制。

- [Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks](https://yingwen.io/zh/continual-rl/research/#recent-columnar-constructive-networks)
- [Real-Time Recurrent Learning using Trace Units in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-real-time-trace-units)
- [Towards model-free RL algorithms that scale well with unstructured data](https://yingwen.io/zh/continual-rl/research/#recent-nibbler-predictive-features)
- [When does Self-Prediction help? Understanding Auxiliary Tasks in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-self-prediction-auxiliary-tasks)
- [Recurrent Reinforcement Learning with Memoroids](https://yingwen.io/zh/continual-rl/research/#recent-memoroids-sequence-learning)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)
- [Does Zero-Shot Reinforcement Learning Exist?](https://yingwen.io/zh/continual-rl/research/#recent-zero-shot-forward-backward)
- [Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations](https://yingwen.io/zh/continual-rl/research/#recent-successor-flow-features)

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [Expected Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-expected-eligibility-traces)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)
- [Recurrent Reinforcement Learning with Memoroids](https://yingwen.io/zh/continual-rl/research/#recent-memoroids-sequence-learning)

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

Laplacian 描述行为图结构，奖励感知表示加入路径价值，METRA 学习有区别的行为，HIQL 利用离线目标轨迹，MaestroMotif 引入语言先验。它们承担不同的设计工作；生成技能、选择技能与组合技能需要分别评价。

- [Proper Laplacian Representation Learning](https://yingwen.io/zh/continual-rl/research/#recent-proper-laplacian-representations)
- [METRA: Scalable Unsupervised RL with Metric-Aware Abstraction](https://yingwen.io/zh/continual-rl/research/#recent-metra-skills)
- [HIQL: Offline Goal-Conditioned RL with Latent States as Actions](https://yingwen.io/zh/continual-rl/research/#recent-hiql-hierarchical-goals)
- [Foundation Policies with Hilbert Representations](https://yingwen.io/zh/continual-rl/research/#recent-hilbert-foundation-policies)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

模型可提取性的理论说明某类能力需要什么知识，不指定唯一网络。Dreamer 研究潜在想象控制，STOMP 研究随机时长行为模型，DRAGO 研究旧模型知识保留。模型误差、查询策略和规划收益之间仍需实验连接。

- [Mastering diverse control tasks through world models](https://yingwen.io/zh/continual-rl/research/#recent-dreamerv3-world-models)
- [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://yingwen.io/zh/continual-rl/research/#recent-dino-wm-feature-planning)
- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://yingwen.io/zh/continual-rl/research/#recent-vjepa2-action-conditioned)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Real-Time Recurrent Learning using Trace Units in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-real-time-trace-units)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Learning from experience instead of curated datasets](https://yingwen.io/zh/continual-rl/research/#recent-oak-network-idbd)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [The Cell Must Go On: Agar.io for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-agarcl)
- [OGBench: Benchmarking Offline Goal-Conditioned RL](https://yingwen.io/zh/continual-rl/research/#recent-ogbench-goal-evaluation)
- [Recurrent Reinforcement Learning with Memoroids](https://yingwen.io/zh/continual-rl/research/#recent-memoroids-sequence-learning)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文提供定义和条件，架构讲座提出模块组织，算法论文检验特定机制。完整系统还要明确智能体、外部设计者和世界各自承担的工作；组件成立不自动意味着组合后的长期收益成立。

- [The OaK Architecture: A Vision of SuperIntelligence from Experience](https://yingwen.io/zh/continual-rl/research/#recent-oak-architecture)
- [Towards model-free RL algorithms that scale well with unstructured data](https://yingwen.io/zh/continual-rl/research/#recent-nibbler-predictive-features)
- [Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations](https://yingwen.io/zh/continual-rl/research/#recent-successor-flow-features)

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

### Towards model-free RL algorithms that scale well with unstructured data

Joseph Modayil, Zaheer Abbas

arXiv 预印本 · 2023 · 支持方法与理论

#### 研究问题

大量原始观测中只有少数局部组合与奖励有关，智能体能否逐步构造有用的预测特征？

#### 关键机制

Nibbler 将预测问题的构造和预测结果的复用结合起来：选择局部输入、学习与奖励相关的通用价值预测，再将预测作为后续学习的特征。GVF 在这里不是一个新的优化器，而是描述“预测什么、在什么行为下预测”的问题接口。

#### 证据

作者在组合式合成环境中增加观测规模，报告了利用任务结构的样本效率。环境可以具有指数增长的状态组合，但学习器不必显式枚举全部状态。

#### 条件与限制

这不是对任意高维观测的线性样本复杂度保证。局部可分解结构、候选问题与特征构造规则仍是关键条件；从该实验族迁移到视觉控制需要额外验证。

#### 阅读与实验

把一条预测完整写成累积量、延续条件、目标策略和输入特征四项，再指出它如何进入主任务的价值函数。区分问题生成带来的收益与增加参数量带来的收益。

#### 原文与相关入口

- [作者预印本](https://arxiv.org/abs/2311.02215)：问题族、Nibbler 构造过程与扩展性实验。

### When does Self-Prediction help? Understanding Auxiliary Tasks in Reinforcement Learning

Claas A. Voelcker, Tyler Kastner, Igor Gilitschenski, Amir-massoud Farahmand

RLC 2024 / RLJ · 2024 · 支持方法与理论

#### 研究问题

预测下一潜在状态、重建观测和学习价值，为什么会产生不同的表示？

#### 关键机制

论文在含干扰因素的线性问题中分析辅助目标的学习动力学。潜在状态自预测与价值学习共同作用时可能保留决策相关结构，但单独训练同一目标未必得到最有用的特征。目标的作用取决于它和 TD 目标怎样共享表示。

#### 证据

线性分析给出可检查的条件，并用神经网络实验检验部分预测。结果不支持“任何自监督预测都能改善 RL”这种无条件判断。

#### 条件与限制

线性分析中的观测映射、优化过程与神经网络控制并不完全等价。项目仓库入口不等于已经提供完整可复现实验实现，因此这里不列为可运行代码。

#### 阅读与实验

固定编码器容量，分别比较仅 TD、仅辅助任务和联合训练。记录价值误差与任务收益，不要只用辅助损失下降评价表示。

#### 原文与相关入口

- [RLC 2024 论文入口](https://rlj.cs.umass.edu/2024/papers/Paper197.html)：原文、线性假设与神经网络实验。
- [作者预印本](https://arxiv.org/abs/2406.17718)：便于追踪论文版本。

### Proper Laplacian Representation Learning

Diego Gomez, Michael Bowling, Marlos C. Machado

ICLR 2024 · 2024 · 支持方法与理论

#### 研究问题

技能发现需要一组确定的谱方向，为什么仅学到低频子空间还不够？

#### 关键机制

图上的平滑性目标倾向保留缓慢变化的特征，但旋转后的同一子空间未必给出可解释、排序明确的单个特征向量。ALLO 使用增广 Lagrangian、正交条件与对称性破除，同时恢复特征向量和特征值，从而为 eigenoption 的方向构造提供更明确的输入。

#### 证据

论文分析优化目标，并在多个环境中检验谱表示的恢复质量和下游使用。作者仓库包含表示学习训练程序。

#### 条件与限制

谱结构依赖采样行为诱导的图和覆盖程度，不是脱离数据分布的环境真值。低频方向也不自动等于有奖励价值的技能；这正是奖励感知表示要继续处理的问题。

#### 阅读与实验

先在小图上直接求特征分解，再比较学习特征的子空间误差和逐向量误差。两种指标不等价，后者才揭示任意旋转问题。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2310.10833)：ICLR 2024 论文的公开版本。
- [ALLO 作者代码](https://github.com/tarod13/laplacian_dual_dynamics)：增广 Lagrangian 的实际优化与实验入口。

#### 作者代码

[论文作者的 ALLO 实现。](https://github.com/tarod13/laplacian_dual_dynamics)

Laplacian 表示学习和论文实验。

### METRA: Scalable Unsupervised RL with Metric-Aware Abstraction

Seohong Park, Oleh Rybkin, Sergey Levine

ICLR 2024 · 2024 · 支持方法与理论

#### 研究问题

没有外部任务奖励时，怎样发现能产生长距离、有区别状态变化的技能？

#### 关键机制

METRA 学习反映时间距离的潜在表示，并让技能方向 $z$ 最大化内在奖励 $r_z=(\phi(s')-\phi(s))^\top z$。邻接状态间的距离约束阻止编码器靠任意放大数值提高奖励。表示学习和技能策略相互影响，因此它不同于先固定一个表示、再单独训练 option。

#### 证据

论文在视觉与状态输入的运动、操纵任务中研究无监督技能学习和下游使用。作者代码包括约束优化、技能策略和相应实验配置。

#### 条件与限制

预训练技能加下游任务不等于技能库在单次生命内持续维护。理论距离约束与源码中的均方尺度、松弛量截断需要分别对照，不能只照抄一个简化公式重现。

#### 阅读与实验

观察表示范数、约束残差和实际位移三条曲线。若内在回报上升而位移不变，应先检查尺度和约束，而不是直接解释为探索改善。

#### 原文与相关入口

- [ICLR 原文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/516593a423838642a2eb4e9c5b9c7f44-Abstract-Conference.html)：方法与技能评价。
- [作者代码](https://github.com/seohongpark/METRA)：核心方法在 iod/metra.py；同时检查约束的归一化与截断。

#### 作者代码

[作者提供的论文实现。](https://github.com/seohongpark/METRA)

METRA、技能训练与下游评价。

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

### Learning from experience instead of curated datasets

Oak Lab

Oak Lab 技术博文 · 2026 · 支持方法与理论

#### 研究问题

有用信号稀疏且大量输入是噪声时，在线学习规则如何分配不同方向的更新能力？

#### 关键机制

博文从含稀有有效特征的线性预测问题出发，对比统一步长与 IDBD 的逐权重适应，再展示 NetworkIDBD 在非线性带噪观测中的例子。核心主张是让长期学习效果影响信用和步长分配，而不只依据当前梯度幅度归一化。

#### 证据

公开页面提供受控噪声特征任务和 NoisyMNIST 示例。它们是机制演示，便于理解有效信号密度与输入规模的关系。

#### 条件与限制

该页面不是完整 CRL 控制论文，也未给出可直接复现所有图表的完整代码和算法推导。监督噪声任务的结果不能证明一般 SGD 或所有深度 RL 都无法从经验学习。

#### 阅读与实验

先复现线性噪声特征问题，分开改变有效特征稀疏度与噪声维数。进入控制前，再加入策略改变数据分布这一因素。

#### 原文与相关入口

- [Oak Lab 原始博文](https://oaklab.ai/posts/learning-from-experience-instead-of-curated-datasets)：2026 年 7 月 13 日；受控实验、NetworkIDBD 示例与研究动机。

### Mastering diverse control tasks through world models

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap

Nature · 2025 · 支持方法与理论

#### 研究问题

同一套世界模型训练与控制方法，能否减少跨任务重新设计损失和超参数的需求？

#### 关键机制

DreamerV3 从经验学习递归潜在状态、奖励和延续预测，再在潜在想象轨迹上学习 actor 和 critic。尺度稳健的表示与损失设计使同一配置可以适用于多种任务。模型是用于决策的学习接口，不必生成完整真实世界。

#### 证据

论文在大量视觉和状态控制任务上报告了广泛表现。关键含义是共享算法配置；这些结果主要来自分别训练的任务智能体，不是一个智能体按顺序学会全部任务。

#### 条件与限制

经验重放、批量训练和模型想象都有资源成本。模型偏差、表示遗忘与长期任务切换仍需要专门实验，不能由多任务覆盖范围自动推出持续学习能力。

#### 阅读与实验

把状态更新、模型训练、想象起点和策略更新四种分布分别写清。比较真实交互步数之外，还应记录想象步数和优化次数。

#### 原文与相关入口

- [Nature 原文](https://www.nature.com/articles/s41586-025-08744-2)：方法和任务协议；区分共享配置与单智能体持续学习。
- [作者维护的实现](https://github.com/danijar/dreamerv3)：公开实现的版本与论文实验环境应分别记录。

#### 作者代码

[作者发布的重实现；不把当前分支当作原论文实验的冻结快照。](https://github.com/danijar/dreamerv3)

DreamerV3 的作者维护公开实现及运行配置。

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

### Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations

Haosen Shi, Jianda Chen, Sinno Jialin Pan

ICLR 2026 · 2026 · 支持方法与理论

#### 研究问题

能否直接学习多步未来状态的分布，并把它压缩为适合控制学习的特征？

#### 关键机制

SF² 以 flow matching 估计 successor measure，将条件向量场分解为未来位置及生成时间的投影与当前状态动作特征的乘积。特征进入 TD3/SAC 的 critic；线性的是向量场对条件特征的分解，critic 本身可以非线性。

#### 证据

正式原文给出 mixture Bellman 结构、生成式 bootstrap 与控制实验，并提供作者 JAX/Brax 仓库。实验研究在线收集数据下的 off-policy 控制，并使用 replay、批次与目标网络。

#### 条件与限制

“online policy learning”不代表 strict streaming。生成时间不是环境时间；向量场线性不保证任意奖励价值线性。文中与 SR 的小生成时间联系是近似动机，未证明递归 agent state 或任意持续变化下的充分性。

#### 阅读与实验

对齐模型调用与梯度预算，拆分直接预测、bootstrap、critic 联合训练。冻结特征后比较线性与非线性读出，再测新奖励和动力学变化，才能检验预测知识的可复用程度。

#### 原文与相关入口

- [ICLR 2026 正式原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48acf4b231771e693f42305b4c9b4c9f-Abstract-Conference.html)：第 2–3 节和算法附录；区分 flow 时间、环境时间与近似 SR 联系。
- [原文链接的作者实现](https://github.com/Shiien/successor-flow-representation-implementation)：SAC/TD3、flow 特征、对照和 sweep 配置。

#### 作者代码

[正式论文摘要直接链接的作者代码。](https://github.com/Shiien/successor-flow-representation-implementation)

基于 JAX/Brax 的 SF² 控制实验；不包含自动 GVF 问题发现或完整持续架构。

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

### DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning

Gaoyue Zhou, Hengkai Pan, Yann LeCun, Lerrel Pinto

ICML 2025 · 2025 · 支持方法与理论

#### 研究问题

预训练视觉表示能否直接成为动作后果预测与目标规划的接口？

#### 关键机制

冻结 DINOv2 空间 patch 特征，用离线动作轨迹学习未来特征预测器；测试时优化动作序列，让预测特征接近目标图像特征。没有重建图像、奖励模型或逆模型，不表示没有动作条件的动力学训练。

#### 证据

ICML 原文在六类环境检验视觉目标规划，作者仓库公开数据、部分检查点、训练与 CEM 规划入口。零样本指给定已训练模型后解决目标，无额外任务策略训练。

#### 条件与限制

依赖视觉预训练与离线交互覆盖；patch 相近不总等于任务完成或风险相同。原实验不证明冻结视觉表示能适应长期新物体、新动作语义或隐藏状态。

#### 阅读与实验

分别改变背景、物体属性、控制动力学与目标分布。把冻结 encoder 和联合更新 encoder 分开，对照视觉距离、真实成功与模型误差，观察表示漂移的依赖成本。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/zhou25t.html)：正式发表入口；早期 ICLR 投稿页不能替代此状态。
- [作者项目代码](https://github.com/gaoyuezhou/dino_wm)：train.py、plan.py、数据与已公开模型检查点范围。

#### 作者代码

[原作者 Gaoyue Zhou 的论文配套仓库。](https://github.com/gaoyuezhou/dino_wm)

DINO 特征预测、离线环境数据与目标规划；README 公开部分环境检查点。

### V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, Nicolas Ballas

arXiv 预印本（此处采用 2025 首稿） · 2025 · 支持方法与理论

#### 研究问题

无动作标注的视频预训练，与能接受机器人动作的规划模型之间还缺哪一步？

#### 关键机制

V-JEPA 2 先学被遮蔽视频的潜在特征预测；V-JEPA 2-AC 冻结编码器，再用机器人轨迹训练动作条件预测器。控制以目标图像的特征差为代价进行 MPC；视频理解、动作条件预测和真实控制是三个独立证据层。

#### 证据

2025 首稿报告以大规模视频预训练，再用不到 62 小时 DROID 交互视频后训练，在两个实验室以图像目标做真实机器人规划。论文单独讨论相机位置、长程规划与图像目标的局限。

#### 条件与限制

无任务奖励并不等于无动作、无机器人状态或无外部数据。零样本部署未持续更新模型，也未发现和维护 options。官方仓库现含 V-JEPA 2.1，复现首稿须记录配置和模型版本。

#### 阅读与实验

按视觉编码、动作坐标、后果模型、目标代价逐项做迁移检验。若引入在线更新，记录模型更新使旧目标接口失效的程度，测未来交互收益，而非仅用 frozen probe 证明 CRL。

#### 原文与相关入口

- [2025 首稿](https://arxiv.org/abs/2506.09985v1)：action-free 预训练、2-AC 后训练、真实规划与第 4.3 节限制；此处不赋予未核实会议状态。
- [Meta FAIR 官方实现](https://github.com/facebookresearch/vjepa2)：包含 V-JEPA 2、2-AC 和较新的 2.1；版本不能混用。

#### 作者代码

[Meta FAIR 官方仓库；首稿模型与后续版本需按配置区分。](https://github.com/facebookresearch/vjepa2)

官方视频表征与动作条件模型；数据、机器人部署条件与检查点分别核验。


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

- [NeurIPS 2024 原文](https://papers.nips.cc/paper_files/paper/2024/file/19f7f755908372efb25826d61959cdf9-Paper-Conference.pdf)：结合运算、inline reset、Tape-Based Batching 与实验。

- [作者公开版本](https://arxiv.org/html/2402.09900v3)：附录给出不同递归模型与回报的 memoroid 写法。

- [Recurrent Reinforcement Learning with Memoroids · 作者实现](https://github.com/proroklab/memoroids)：memory 模型、buffer、losses 与 segment_dqn/tape_dqn 对照。 论文附录原链接 memory-monoids 对应作者 Prorok Lab 的现有 memoroids 仓库；README 标明论文。

- [作者原文](https://arxiv.org/abs/2209.14935)：2022 首稿，ICLR 2023；比较奖励表示、SF 与 FB。

- [作者研究平台](https://github.com/facebookresearch/controllable_agent)：README 直接关联两篇 FB 论文；该仓库已经归档。

- [ICLR 2026 正式原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48acf4b231771e693f42305b4c9b4c9f-Abstract-Conference.html)：第 2–3 节和算法附录；区分 flow 时间、环境时间与近似 SR 联系。

- [原文链接的作者实现](https://github.com/Shiien/successor-flow-representation-implementation)：SAC/TD3、flow 特征、对照和 sweep 配置。

- [ICML 2024 原文](https://proceedings.mlr.press/v235/park24g.html)：Hilbert 距离、策略提示和定理前提；不是 ICLR 论文。

- [作者项目与公式](https://seohong.me/projects/hilp/)：时间距离与方向奖励接口。

- [官方实现](https://github.com/seohongpark/HILP)：hilp_zsrl 与 hilp_gcrl 对应不同实验。

- [ICML 2025 原文](https://proceedings.mlr.press/v267/zhou25t.html)：正式发表入口；早期 ICLR 投稿页不能替代此状态。

- [作者项目代码](https://github.com/gaoyuezhou/dino_wm)：train.py、plan.py、数据与已公开模型检查点范围。

- [2025 首稿](https://arxiv.org/abs/2506.09985v1)：action-free 预训练、2-AC 后训练、真实规划与第 4.3 节限制；此处不赋予未核实会议状态。

- [Meta FAIR 官方实现](https://github.com/facebookresearch/vjepa2)：包含 V-JEPA 2、2-AC 和较新的 2.1；版本不能混用。
