# 对手建模与递归推理：预测谁，回应什么？

现代深度强化学习 · 并列研究分支

给定参与者和评价目标，怎样利用行为预测、条件响应与有限递归改善决策，并检验模型是否可信？

## 本章内容

- 区分真实信息、行为相关与模型假设的响应。
- 推导 PR2 的软响应、普通期望与软价值的不同梯度。
- 理解 GR2 的有限递归、ROMMEO 的经验约束与 GSCU 的模型使用选择。
- 分别检验预测、控制收益与模型失配，保留理论及实现条件。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [多智能体合作：结构化探索与信用分配](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent/)：明确他人的策略怎样进入自己的决策问题。
- [最大熵连续控制：SAC 的价值、密度与温度](https://yingwen.io/zh/continual-rl/foundations/deep/entropy-control/)：理解随机策略、熵和软响应。


### 随机博弈与信息集

联合动作决定转移；各参与者只能按自己的可用信息行动。

### 策略评价与改善

评价固定策略与求最优响应是不同问题。

### 概率、熵与 KL

区分真实分布与拟合模型；知道 KL 非负。

<a id="problem-definition"></a>

## 本章的问题定义

有限双人博弈，或具有已声明观察权限的折扣随机博弈。对手可以固定、从一个分布抽取，或随训练更新；三种协议分别讨论。

### 给定条件与符号

- 自身可用历史、动作集和奖励。
- 对手身份与已执行动作是否可见，是否有共同随机信号。
- 对手分布 q、模型族、更新预算和独立评价对手。

### 需要求解的对象

学习可检验的行为或响应模型，并求对指定对手分布的近似最优响应；均衡目标还需检查所有单方偏离。

### 信息与数据权限

同时行动时，不能把自身尚未执行的本步动作视为对手已收到的观察。训练期记录联合动作不等于执行前可见。

$$
J_i(\pi_i;q)=\mathbb E_{\pi_{-i}\sim q}\mathbb E_{\pi_i,\pi_{-i}}\!\left[\sum_{t=0}^{T-1}\gamma^tR^i_{t+1}\right],\quad \operatorname{BR}_i(q)\in\arg\max_{\pi_i}J_i(\pi_i;q)
$$

q 每局抽取完整对手策略，局内冻结。无限时域取 0≤γ<1、奖励有界。对固定 q 求响应不等于求 Nash 均衡。

### 成立条件与解的含义

- 策略遵守既定信息结构；历史摘要的充分性需独立论证。
- 矩阵算例使用已知收益；神经实验还存在采样和逼近误差。
- 软备份、局部动力学和有限递归保留各自的博弈与优化条件。

判断准则：分别检查预测误差、固定对手收益、交叉对战与偏离收益，不能彼此替代。

### 适用边界

- 不从条件相关推断因果影响。
- 不把内部递归层数当作真实对手的心理层次。
- 不声称深度 PR2、GR2 在任意游戏中收敛。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 推广：放宽条件 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：对手分布进入控制目标，单方策略改善与均衡评价需要分开。

- 组合不同学习问题 · [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)：对手类型不可见时，历史和信念进入行为预测与控制。

- 组合不同学习问题 · [元学习与学习规则的适应](https://yingwen.io/zh/continual-rl/algorithms/meta/)：选择响应对象与更新规则发生在基础策略学习之外，不能与局内推理混同。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

对手行为、训练分布与双方更新相互影响，短期胜利可能只是利用暂时弱点。

### 本章的核心思路

先固定评价对象，再分开数据生成、模型推断与策略响应。

1. [固定问题与信息权限](#lesson-setting)：先规定响应谁、能观测什么以及怎样评价。

2. [定义模型](#lesson-opponent-model)：区别行为拟合、假设响应与真实信息权限。

3. [构造响应](#lesson-pr2)：从变分目标到有限层递归。

4. [检验边界](#lesson-limits)：分别评价模型、收益、预算和偏离。

结论与条件：解析恒等式可精确验证；它们不推出神经网络训练的普遍收敛。

### 相关方法改变了什么

- 不显式建模的响应学习：直接从对局优化行为；少了模型失配，也少了对未执行候选的显式推断。

- 无条件行为预测：只按可用历史拟合动作，不能据此表达候选自身动作对应的模型响应。

- 固定保守策略：避免利用错误模型，但可能放弃针对当前参与者的可利用机会。


<a id="lesson-setting"></a>

## 1 · 给定参与者，先弄清模型应预测什么

给定本轮将遇到的参与者，学习者仍须判断对方会怎样行动，以及自己的候选动作会得到什么结果。只从回报优化行为是一种办法；显式预测对方、在模型内比较响应，是另一种办法。本章研究后一条路线。这里的“对手模型”也可用于伙伴；建模对象的名称不决定双方的奖励关系。

在[合作多智能体学习](/zh/continual-rl/foundations/deep/multi-agent/)中，核心问题包括怎样发现有效的联合行为，以及怎样把共同结果归因于各自行动。预测伙伴能帮助组织探索和控制，但不会自动解决联合探索或信用分配。在[开放式多智能体学习](/zh/continual-rl/foundations/deep/multi-agent-populations/)中，竞争与开放合作还需评价现有策略、构建下一轮对手或伙伴分布，再检查策略改善。更准确的模型可以提高内层响应质量，却不替代外层评价和目标构建，也不保证整体性能单调提高。

固定对手分布以后，模型也有三种不同职责：拟合已经看见的行为，预测未执行动作下的结果，或构造一个供优化使用的假设响应。第一种有监督标签；第二种需要覆盖或模型假设；第三种还要说明对手为何会遵循该响应。把三者混在一起，可能得到预测分数不错、实际决策却很差的策略。

| 对象 | 输入与输出 | 边界 |
| --- | --- | --- |
| 对手分布 q | 给完整策略或历史版本分配概率 | 近期对手不是未来所有对手 |
| 行为模型 ρ | 给定可用信息预测对手行为 | 预测准确不自动使自身稳健 |
| 响应算子 BR | 给定对手分布和自身奖励，求改善策略 | 一次响应不是双方均衡 |

对手固定、全状态可见且各方使用 Markov 策略时，可以积分掉对手动作，得到单智能体 MDP。若对手依赖私有历史，或每局抽取未知类型，当前观察通常仍非 Markov。冻结对手消除了参数更新，不会消除隐藏信息。

下面先辨认行为相关与真实响应，再用两动作价值推导 PR2 的软响应及其梯度。GR2 接着问模型中可以展开几层回应；ROMMEO 和 GSCU 则分别约束模型想象的行为，以及决定何时使用模型。每一步都要回到同一个检验：在允许的信息和相同评价对手下，它是否改善了真实控制。

<a id="lesson-opponent-model"></a>

## 2 · 行为预测不等于真实因果响应

$$
L_{\rm pred}(\phi)=-\mathbb E_{(H_i,A_{-i})\sim D}\log\rho_\phi(A_{-i}\mid H_i)
$$

用允许观测的历史预测对手已执行动作。动作不可观察时需额外推断，不能凭空得到监督标签。

条件模型还可输入候选自身动作，写成 $\rho(a_{-i}\mid s,a_i)$。它可以表示数据相关性，也可以表示内部假设的响应；二者都不意味着同时行动的对手已经看到了 $a_i$。

$$
\pi(a_i,a_{-i}\mid s)=\pi_i(a_i\mid s)\pi_{-i}(a_{-i}\mid s)
$$

给定同一个完整状态、双方独立随机行动时，真实联合分布满足此因子化。隐变量、共同信号或混合的训练版本会改变条件独立关系。

要把条件相关解释为“我改变动作就使对手改变动作”，需实际观察顺序、承诺机制或因果假设。先行动并被对手看见的 Stackelberg 游戏，与同时行动的 Nash 游戏是不同问题。

$$
\widetilde Q_i(s,a_i)=\sum_b\rho_\phi(b\mid s,a_i)Q_i(s,a_i,b)
$$

固定 φ 和 Q 后，这是模型内的期望评分。真实评价仍需由环境中的实际对手执行。

actor 可能选择数据未覆盖的动作，利用模型虚构的有利回应。应分别检查保留轨迹上的预测误差、动作覆盖、模型失配以及环境真实回报。

<a id="lesson-pr2"></a>

## 3 · PR2：从价值构造变分响应

PR2 使用条件对手模型与概率推断近似响应，再改进自身策略。重点不是把对手网络拼到输入，而是规定响应分布怎样从价值和正则项产生。下面在有限动作上推导其软响应核。

$$
F_i(s,a_i)=\alpha\log\sum_b e^{Q_i(s,a_i,b)/\alpha},\qquad \rho^*(b\mid s,a_i)=e^{[Q_i(s,a_i,b)-F_i(s,a_i)]/\alpha}
$$

α>0。F 是软聚合值，不是固定真实对手下的普通价值。连续动作还需声明积分的基准测度与可积性。

$$
\alpha D_{\rm KL}(\rho\Vert\rho^*)=F_i-\mathbb E_\rho Q_i-\alpha H(\rho)
$$

代入 log ρ*=(Q−F)/α 并使用概率和为一，即得恒等式。

$$
F_i=\max_\rho\{\mathbb E_\rho Q_i+\alpha H(\rho)\}
$$

KL 非负给出上界，ρ=ρ* 时达到。限制模型族后，最小化 KL 是变分近似；有限粒子又增加采样近似。

指数中使用的是玩家 i 的价值，不能因此说对手在最大化其自身奖励。这是推断构造中的响应目标，不是任意竞争对手的通用行为定律。未经检验地替代真实对手分布，可能产生乐观失配。

取 Q=(0,1)、α=1，ρ*≈(0.269,0.731)，普通期望≈0.731，熵≈0.582，F≈1.313。F 大于最大 Q=1 不代表获得额外环境奖励；差额来自正则化。若真实对手总选第一项，期望仍为零。

$$
(\mathcal T^{\pi_i}Q_i)(s,a_i,b)=r_i(s,a_i,b)+\gamma\mathbb E_{s',a_i'\sim\pi_i}[F_i(s',a_i')]
$$

与上述软响应配套的理想固定自身策略备份。神经 critic 用停止梯度的 target 版本计算右端，再回归当前 joint Q；若任务真实终止则不 bootstrap。PR2 的软值不能换成普通对手期望而仍称同一算子。

先只检查这个固定策略算子。若 $\|Q_1-Q_2\|_\infty\le\varepsilon$，逐项比较指数可得 $e^{-\varepsilon/\alpha}\sum_b e^{Q_2/\alpha}\le\sum_b e^{Q_1/\alpha}\le e^{\varepsilon/\alpha}\sum_b e^{Q_2/\alpha}$。取对数并乘温度，便有 $|F_1-F_2|\le\varepsilon$；相同固定转移与自身策略的期望再乘折扣，使备份差不超过 $\gamma\varepsilon$。有限动作、有界奖励、$\gamma<1$ 下，这是模型内部固定备份的压缩性；不证明真实对手等于推断响应，也不证明双方同时改变策略和模型时达到均衡。原文针对自对弈的更强结论还声明了额外博弈条件。

<a id="lesson-response-gradient"></a>

## 3.1 · 回应模型时，对什么求导？

$$
\nabla_a\widetilde Q(a)=\sum_b\rho_\phi(b\mid a)\nabla_aQ(a,b)+\sum_bQ(a,b)\nabla_a\rho_\phi(b\mid a)
$$

固定 φ，对候选动作 a 求导。第一项改变自身动作，第二项计入模型预测的对手分布变化。

stop-gradient 对手响应会删除第二项，因此改变更新算子，而不只是节省计算。保留第二项也只说明优化了模型评分，不能把它当作真实对手的因果导数。

$$
\nabla_a F(a)=\sum_b\rho^*(b\mid a)\nabla_aQ(a,b)
$$

直接微分 log-sum-exp 即得。它不与上式冲突：F 还包含熵，ρ* 是内层最优解。不能混用普通期望与软值的梯度。

**算法：PR2/GR2 型工程的阅读顺序；不是宣称不同论文采用相同的软聚合算子。**

1. 声明对手协议、可用信息、响应模型和正则化目标。
1. 收集自身奖励、后继观察以及允许记录的联合动作。
1. 固定 target 版本，用选定软 Bellman 目标更新 joint critic。
1. 用 KL / 粒子近似更新条件响应模型。
1. 更新 actor，明确哪些响应分支 stop-gradient；更新 target。
1. 在冻结且未参与拟合的对手上独立评价。

PR2-Q 与 PR2-Actor-Critic 分别使用价值控制与 actor–critic。原算法的数据元组包含对手已执行动作；分散训练不等于“不需要任何其他玩家行为信息”。

<a id="lesson-gr2"></a>

## 4 · GR2：有限递归与层次混合

模型还可以假设：对手正在回应一个关于我的模型。递归必须从 level-0 行为假设开始，并在有限深度停止。零层可以是均匀行为或学习到的基础策略，不是未经定义的“不会思考”。

$$
\pi_i^{(k)}=\operatorname{BR}_i(\pi_{-i}^{(k-1)}),\qquad k\ge1
$$

单状态、精确响应的教学抽象。GR2-L 以条件策略交替展开自身与对手的推理；实际更新不逐层求精确 BR。

$$
q_{-i}^{(<k)}=\sum_{\ell=0}^{k-1}w_\ell^{(k)}\pi_{-i}^{(\ell)},\quad w_\ell^{(k)}=\frac{\lambda^\ell/\ell!}{\sum_{j=0}^{k-1}\lambda^j/j!},\quad \pi_i^{(k)}=\operatorname{BR}_i(q_{-i}^{(<k)})
$$

λ>0。截断 Poisson 混合说明 cognitive hierarchy 与 GR2-M 的动机；不声称作者工程逐行实现这个教学形式。

用猜数说明响应的精确含义。n 人选数，目标为全体均值的 p 倍。已知其他 n−1 人均值 m，自己命中目标需满足 x=p[x+(n−1)m]/n，故 x=p(n−1)m/(n−p)。常见 x≈pm 忽略自身对均值的影响，只在 n 很大等近似条件下成立。

取 n=2、p=0.7、零层 m=50，精确响应≈26.923，不是 35。再假设对手处于该一层，二层响应≈14.497。这是固定对手数字、平方偏差最小化的教学任务；胜负奖励、随机对手与并列规则需另定义响应。

GR2 实践使用确定性内部展开、跨层参数共享与辅助层间改善目标控制成本。深层也重复使用有偏模型，未必更准。作者工程还截断部分对手分支梯度；递归展开不等于所有层完整反向传播。

<a id="lesson-rommeo"></a>

## 5 · ROMMEO：有利响应不能脱离经验依据

PR2 之后的一个自然问题是：模型偏向有利行为时，怎样防止它想象一个现实中不存在的合作伙伴？Tian、Wen 等的 ROMMEO 从合作决策的概率推断出发，用观测到的行为分布约束对手模型。它不是简单地提高行为预测准确率，而是在协调收益与经验依据之间建立明确目标。

$$
\mathcal J_s(\pi,\rho)=\mathbb E_{b\sim\rho,a\sim\pi(\cdot\mid s,b)}Q(s,a,b)+\alpha\mathbb E_{b\sim\rho}H(\pi(\cdot\mid s,b))-D_{\rm KL}(\rho(\cdot\mid s)\Vert P(\cdot\mid s))
$$

固定状态的改善子问题。P 是经验对手先验；这里 KL 系数为一，与原文该形式对应。π 的条件输入 b 在执行时来自内部模型，不等于提前偷看真实动作。

$$
F(s,b)=\alpha\log\sum_a e^{Q(s,a,b)/\alpha},\qquad \rho^*(b\mid s)=\frac{P(b\mid s)e^{F(s,b)}}{\sum_{b'}P(b'\mid s)e^{F(s,b')}}
$$

先对自身条件策略优化，再对 ρ 优化，分别得到软响应与经验先验的指数倾斜。与 PR2 中对对手动作做无先验软聚合不同，ROMMEO 明确保留 P 的约束。

若 P(b|s)=0，有限 KL 不允许模型给该行为正概率。这个限制能抑制虚构响应，也可能排除尚未观察到的有益协调方式。怎样平滑先验和收集覆盖需要另行定义。原文的理论设置及实验重点是合作游戏；不能把其乐观协调模型当作零和对抗的安全保证。

<a id="lesson-gscu"></a>

## 6 · GSCU：何时利用模型，何时保守行动？

另一个问题不在递归深度，而在于是否应相信模型。Fu、Tian、Wen 等的 GSCU 先离线学习对手策略的连续嵌入，并训练一个条件化该嵌入的近似响应。线上根据已经完成的对局更新嵌入后验，再在实时利用策略与固定保守策略之间做 bandit 选择。

该设置允许对手改变，但准备阶段已经提供训练对手集合与较强的保守策略。原文主要设定还允许读取历史对局中对手的观察—动作轨迹；当前局的未知动作并未提前提供。后验分布表达模型内的不确定性，不保证涵盖训练外对手。

原算法将嵌入的后验均值输入条件响应策略。EXP3 根据获得的回报更新两个候选的选择权重；它不是检查后验方差是否超过某个阈值，再机械切换到保守策略。

$$
\max_{e\in\{\mathrm{greedy},\mathrm{safe}\}}\sum_{j=1}^{T}g_j(e)-\mathbb E\sum_{j=1}^{T}g_j(E_j)=O(\Delta\sqrt T)
$$

这是两臂 EXP3 型选择器的外部遗憾尺度；每局反馈有界于跨度 Δ，且需遵守相应 bandit 协议。比较对象是两个候选过程，不是所有可能策略或每局事后最优选择。

若对手会因自己历史选择而改变，“相同实现反馈序列上的外部遗憾”和“如果一直采取另一策略，对手本来会怎样反应”也不同。不能把前者直接称为长期因果安全。GSCU 说明有价值的分工：后验推断管理模型不确定性，条件策略产生响应，外层选择器控制使用哪个候选。它不要求线上重新训练整个响应网络。

<a id="lesson-limits"></a>

## 7 · 理论边界：稳定的是哪个过程？

PR2 软价值迭代的结论具有对称性、特定均衡和价值算子条件，不能外推到任意一般和游戏及非线性 actor–critic。GR2 的均衡存在性涉及构造的推理博弈，不等于学习必然找到它。

$$
\frac{d}{dt}\begin{bmatrix}x\\y\end{bmatrix}=\begin{bmatrix}0&a\\b&0\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix},\qquad ab<0
$$

两动作单状态游戏内部均衡附近的一类连续时间动力学；特征值 ±i√(−ab) 表示绕行。这种中性稳定不应简单叫作发散。

$$
M_\zeta=\begin{bmatrix}\zeta ab&a\\b&\zeta ab\end{bmatrix},\qquad \lambda(M_\zeta)=\zeta ab\pm i\sqrt{-ab}
$$

在该局部模型中加入 ζ>0 的预期响应项，实部变负。它说明某种预期更新可阻尼旋转，不证明任意递归深度都更好。

$$
0<\eta<\frac{2\zeta}{1-\zeta^2ab}
$$

显式 Euler 更新还需此步长条件，使 |1+ηλ|<1。连续时间稳定不保证任意学习率下稳定；此模型不覆盖概率边界与神经随机梯度。

- 预测检验：新轨迹上的概率校准与未访问动作的覆盖。
- 控制检验：相同对手下比较无模型、无条件模型与条件模型。
- 递归检验：匹配环境步数和计算预算，分开深度与计算量。
- 稳健检验：保留错误层次、未知类型与变化的对手，同时报告偏离或交叉对战。

<a id="lesson-code"></a>

## 8 · 先验证响应核，再阅读神经工程

在完全给定的两动作任务上，比较条件响应与其边缘分布如何改变动作排序。没有拟合模型或运行 PR2 优化器。

```python
def conditional_response():
    """Isolate conditional vs marginal prediction, not the PR2 optimizer.

    rho(b|a) is a supplied hypothetical response model. It is not a claim
    that a simultaneous opponent observes a before choosing b. Fitting,
    variational inference and policy learning are deliberately not included.
    """
    q = [[4., 0.], [1., 2.]]
    rho = [[.1, .9], [.9, .1]]
    marginal = [.5, .5]  # rho averaged with a uniform prior over our action.
    return {"payoffs": q, "response": rho, "marginal": marginal,
            "marginal_values": [sum(x*y for x, y in zip(r, marginal)) for r in q],
            "conditional_values": [sum(x*y for x, y in zip(r, p))
                                   for r, p in zip(q, rho)]}
```

可复制为独立 Python 文件运行；检查软值恒等式和有限差分，不是学习实验。

```python
from math import exp, log

def soft_response(values, alpha=1.0):
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    peak = max(values)
    weights = [exp((q-peak)/alpha) for q in values]
    z = sum(weights)
    return [w/z for w in weights], peak + alpha*log(z)

q = [0.0, 1.0]
rho, free_value = soft_response(q)
expected = sum(p*v for p, v in zip(rho, q))
entropy = -sum(p*log(p) for p in rho if p > 0)
assert abs(free_value-expected-entropy) < 1e-12
# Differentiate F([a, 1-a]) at a = 0.3.
a, eps = 0.3, 1e-6
probs, _ = soft_response([a, 1-a])
finite = (soft_response([a+eps, 1-a-eps])[1]
          - soft_response([a-eps, 1-a+eps])[1]) / (2*eps)
assert abs(finite-(probs[0]-probs[1])) < 1e-8
print(rho, expected, entropy, free_value)
print("soft-value gradient:", finite)
```

| 原始入口 | 阅读位置 | 实现边界 |
| --- | --- | --- |
| ying-wen/gr2 | code/maci/get_agents.py；learners/mavb_ac.py | joint critic、SVGD 粒子、soft backup、replay 和 target；旧 TensorFlow 依赖需独立配置 |
| MultiLevelPolicy | policies/level_k_policy.py 的 actions_for | 交替自身/对手策略；部分对手分支 stop_gradient |
| GeneralizedMultiLevelPolicy | level_distribution 与 actions_for | 对 1…k 层确定动作加权；动作平均不同于先随机选一个策略 |

复现应记录提交、依赖、实际输入权限、层数、粒子数、replay 预算与更新顺序。先在小型游戏验证数据路径，再做高维任务。能导入旧工程并不说明已经复现论文结果。

<a id="lesson-branches"></a>

## 9 · 从内部模型到持续适应

对手模型可在局内依据历史推断，也可跨局积累参数或记忆。前者可能只是状态推断，后者涉及持久改变；应分别说明。固定网络增加内部推理深度，增加的是当次计算，不自动产生跨交互的持续学习。

长期面对新参与者，还需处理模型失配后的可塑性、身份未知时的状态构建、历史保留与资源预算。“对手总比自己少想一层”只是可检验假设。

策略种群解决哪些完整策略值得保留、怎样选择训练对手或伙伴、下一次应补哪个弱点。它可使用对手模型，却不要求每个成员递归推理。将内层响应放回评价、目标构建与改善的循环，见[开放式多智能体学习](/zh/continual-rl/foundations/deep/multi-agent-populations/)；联合探索与团队信用的困难则回到[合作多智能体学习](/zh/continual-rl/foundations/deep/multi-agent/)。

<a id="lesson-check"></a>

## 10 · 检查理解

- 问：$\rho(a_{-i}\mid s,a_i)$ 证明对手看到本步动作吗？答：不能。真实权限由环境协议决定。
- 问：F 大于最大 Q 违反回报上界吗？答：不违反。F 包含正则项，不是原始奖励期望。
- 问：level-3 必然战胜 level-2 吗？答：不必然；模型失配、响应近似与资源成本都影响结果。
- 练习：将算例 α 改为 0.1 和 10。分别计算普通期望与软值，解释二者差异。



## 相关主题：自对弈与开放式学习

<a id="lesson-self-play"></a>

[自对弈与历史策略](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent-populations/#lesson-self-play)

<a id="lesson-fictitious"></a>

[虚拟对弈与最佳响应平均](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent-populations/#lesson-fictitious)

<a id="lesson-nfsp"></a>

[NFSP 的两类记忆](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent-populations/#lesson-nfsp)

<a id="lesson-search-policy"></a>

[搜索怎样产生策略与价值标签](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent-populations/#lesson-search-policy)

<a id="lesson-alphago-lineage"></a>

[AlphaGo、AlphaGo Zero 与 AlphaZero](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent-populations/#lesson-alphago-lineage)

<a id="lesson-muzero"></a>

[MuZero 的学得模型](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent-populations/#lesson-muzero)

<a id="lesson-cfr"></a>

[不完全信息下的 CFR](https://yingwen.io/zh/continual-rl/foundations/deep/multi-agent-populations/#lesson-cfr)

<a id="chapter-code"></a>

## 下载与运行

下载本页配套脚本后运行精确条件评分检查；本页另附软响应梯度的独立代码。均不是 PR2/GR2 神经训练。

[下载 marl_objectives_lab.py](https://yingwen.io/zh/continual-rl/download/marl_objectives_lab.py)

```sh
python3 marl_objectives_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Wen、Yang、Luo、Wang、Pan · PR2（ICLR 2019）](https://arxiv.org/abs/1901.09207)：条件响应的变分推断、Theorem 1/2 和 PR2-Q/PR2-AC；保留定理假设。

- [Wen、Yang、Wang · GR2（IJCAI 2020）](https://www.ijcai.org/proceedings/2020/58)：第 4 节层次模型与第 5 节实现近似；有限理性模型不是人类心理机制的实证证明。

- [Tian、Wen 等 · ROMMEO（IJCAI 2019）](https://www.ijcai.org/proceedings/2019/85)：阅读经验对手先验、KL 正则与合作场景；它改变模型目标，不只是增加预测头。

- [ROMMEO 作者代码](https://github.com/rommeoijcai2019/rommeo)：论文指定的表格与 actor–critic 实现；不是任意对抗场景的稳健性保证。

- [Fu、Tian、Wen 等 · GSCU（ICML 2022）](https://proceedings.mlr.press/v162/fu22b.html)：阅读离线策略嵌入、线上后验与两策略选择；遗憾比较对象及信息权限须保留。

- [GSCU 作者代码](https://github.com/YeTianJHU/GSCU)：embedding_learning、conditional_RL 与 online_test 分别对应准备、响应训练和线上适应。

- [GR2 作者代码与理论附录](https://github.com/ying-wen/gr2)：原文指定工程，含 PR2 基线；附录 D 的低维动力学分析不是深度训练全局定理。

- [GR2 · 递归与层次混合实现](https://github.com/ying-wen/gr2/blob/master/code/maci/policies/level_k_policy.py)：检查 stop_gradient 与确定动作加权，勿称精确随机层次混合。

- [GR2 · 条件响应与 actor–critic](https://github.com/ying-wen/gr2/blob/master/code/maci/learners/mavb_ac.py)：检查 critic、SVGD、粒子软聚合与 replay 字段；仍需独立运行验证兼容性。

- [Lanctot 等 · Policy-Space Response Oracles（2017）](https://arxiv.org/abs/1711.00832)：将训练对手与新增响应组织为种群循环，和内部递归是不同维度。


<a id="marl-mechanism-experiments"></a>

## 小型实验：把更新规则与评价对象分开

全部结果来自完全列举的有限博弈，没有采样误差或置信区间。它们检验具体机制，不是大型神经算法复现。

### 条件响应与边缘化

价值表((4,0),(1,2))。对手边缘分布(1/2,1/2)给出我方动作值2与1.5；给定条件响应模型两行为(0.1,0.9)与(0.9,0.1)，动作值变成0.4与1.1，排序反转。这不证明响应模型正确或有因果解释。PR2还需要条件模型、critic与策略训练；GR2还需要明确递归深度和底层行为。

### 运行与复核

```bash
python3 marl_objectives_lab.py test
python3 marl_objectives_lab.py demo --out results/marl-objectives
```

[源码](https://yingwen.io/zh/continual-rl/download/marl_objectives_lab.py) · [全部数值与源码哈希](https://yingwen.io/crl-code/marl-objectives/results.json)

仅依赖Python标准库，含15项机制检查。图不用于排序大型算法。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-marl-reasoning#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-marl-reasoning#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-deep-marl-reasoning)
<a id="study-connections"></a>

## 与教材主线的衔接

本章是可按问题选择的并列研究分支，不要求先读完其他深度研究分支。

### 一次策略更新，为什么能改善未来？

精确策略改善使用旧策略的真实价值；策略梯度使用与目标匹配的访问分布。值函数近似或梯度估计误差会破坏这些推理的前提。

函数逼近与深度方法：PPO 的动作概率比不等于新策略的状态访问比。连续动作 actor 还会追逐 critic 的误差。限制局部更新尺度与证明实际回报单调提高是不同要求。

持续学习中的研究问题：动作不仅改变世界，也改变未来数据与学习。比较冻结策略不等于比较持续更新的智能体；什么时候应付出当前回报去获得长期有用的经验？

[精确策略改善](https://yingwen.io/zh/continual-rl/foundations/tabular/dynamic-programming/) → [策略梯度定理](https://yingwen.io/zh/continual-rl/foundations/approximation/policy-gradient/) → [TRPO 与 PPO 的近似](https://yingwen.io/zh/continual-rl/foundations/deep/trust-region/) → [持续控制的比较器](https://yingwen.io/zh/continual-rl/algorithms/control/)


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)
- [元学习与学习规则的适应](https://yingwen.io/zh/continual-rl/algorithms/meta/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)
- [持续学习的智能体架构](https://yingwen.io/zh/continual-rl/construction/architectures/)

对应原始材料：PR2（ICLR 2019）；GR2（IJCAI 2020）；ROMMEO（IJCAI 2019）；GSCU（ICML 2022）。本文为原创讲解，原书、论文与上游代码保留各自许可。
