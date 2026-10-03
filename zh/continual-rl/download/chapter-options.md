# Options：多步决策、技能发现与可复用行为

怎样把连续多步的行为当成可复用的决策单位，同时仍能在每个原始时间步学习？

## 本章内容

- 从原始回报拆出 SMDP target，而不是机械地把动作 a 换成 option o。
- 理解 SMDP Q-learning、intra-option learning 和 Option-Critic 分别更新什么。
- 推导 option 内策略与终止函数的两种梯度，正确处理停止、换技能和环境结束。
- 按覆盖、可区分性、可预测性、主任务价值区分发现技能的算法线。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### Primitive action 与 option

primitive action 是环境接收的一步指令。option $o=(I_o,\pi_o,\beta_o)$ 包含三个部分：$I_o$ 规定允许启动的状态；$\pi_o(a|s)$ 选择当前动作；$\beta_o(s')$ 给出到达下一状态后停止的概率。停止 option 不等于结束环境。

### Call-and-return 执行

高层策略 $\mu(o|s)$ 选择一个 option；低层反复按 $\pi_o$ 行动，直到按 $\beta_o$ 停止；随后高层重新选择。高层决策之间可能间隔一个或多个原始时间步。

### 价值与动作优势

$Q(s,o)$ 是从 $s$ 启动 $o$ 然后按高层策略继续的价值；$V(s)=\sum_o\mu(o|s)Q(s,o)$。给定高层策略的评价使用这个加权平均；高层最优控制才使用合法 options 上的最大值。

### Log-derivative

策略概率对参数的导数可写成概率乘 log 概率的导数，因此能用采样动作估计求和。优势 baseline 不依赖当前动作时，可减少方差而不改变局部期望。

$$
\nabla_\theta\pi_\theta(a|s)=\pi_\theta(a|s)\nabla_\theta\log\pi_\theta(a|s)
$$

<a id="lesson-setting"></a>

## 1. 从一步动作到闭环的多步行为

仓库导航中，“向右一次”只能跨一个格子；“走到门口”可能经过十个格子，而且途中每次观测后都调整动作。因此 option 不是预先固定、不可反馈的动作序列：它通常是闭环策略与停止规则。若门被堵住，同一 option 的执行路径和耗时都会改变。时间抽象把高层的搜索与信用分配单位变粗，但低层仍然每步感知和控制。

先考虑有限 Markov 状态、有界奖励和固定折扣 $0\leq\gamma<1$。每个 option 至少执行一步，每个非终止状态有可启动的 option；使用完整轨迹估计时，所执行的 option 还须几乎必然停止。数据包含每个原始步的 $(s,o,a,r,s')$，以及本次 option 的开始状态、累计折扣奖励和持续时间。后两项决定了高层决策的正确更新。

| 阶段 | 已知什么 | 学习什么 |
| --- | --- | --- |
| 给定技能的高层控制 | I、π、β 固定 | 在当前状态选哪个 option |
| Intra-option 估值 | 一个真实动作可与多个 option 相容 | 每步更新多个 option 的价值 |
| Option-Critic | 指定技能数及可微参数化 | 内部策略、停止函数与高层价值 |
| 技能发现 | 只给世界经验和发现准则 | 哪些不同的行为值得进入技能库 |

在 CRL 中，技能的复用取决于变化发生在哪一层。奖励改变可能只需重新估价；环境动力学改变可能要求重学技能；技能策略改变会使旧后果模型过期；表示改变可能同时影响低层策略与模型输入。因此，持续维护技能库还包括评价技能适用性、更新后果模型，以及替换失效或冗余的技能。

<a id="lesson-derive"></a>

## 2. SMDP：由原始回报推导跨多步更新

假设在 $t$ 启动 $o$，在 $t+\tau$ 停止。把回报沿这个边界拆开，前半段是执行 option 期间得到的奖励，后半段是从停止状态开始的后续回报。后半段隔了 $\tau$ 个原始步，因此乘 $\gamma^\tau$。持续时间可以随机；每次样本都使用其实际 $\tau$。

$$
\begin{aligned}G_t&=\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}+\gamma^\tau G_{t+\tau}\\ Q(s,o)&=\mathbb E\!\left[\widehat R_o+\gamma^\tau V(S_{t+\tau})\mid S_t=s,O_t=o\right]\end{aligned}
$$

第二行对第一行取条件期望。R̂_o 是本次 option 内的折扣奖励和，不是平均奖励，也不是子任务内在奖励，除非你明确改变了目标。

$$
Q(s,o)\leftarrow Q(s,o)+\alpha\left[\widehat R_o+\gamma^\tau\max_{o'\in I(s')}Q(s',o')-Q(s,o)\right]
$$

这是给定 options 时的 SMDP Q-learning。I(s′) 表示在 s′ 可启动的 options。真实环境终止时后项为 0；仅 option 结束时后项仍存在。

**算法：算法伪代码**

1. 初始化 Q；每次高层选择合法 o，保存 s_start
1. R_sum=0；discount=1；duration=0
1. 循环：
  1. 按 π_o(a|s) 行动，观察 r,s′,environment_done
  1. R_sum += discount*r；discount *= γ；duration += 1
  1. 若 environment_done，或在 s′ 按 β_o(s′) 抽样结束：
    1. continuation = 0（环境终止）否则 max_{合法 o′} Q(s′,o′)
    1. Q(s_start,o) += α*(R_sum + discount*continuation - Q(s_start,o))
    1. 若环境未结束，在 s′ 重新选 option
  1. 否则保留当前 option；s=s′

这种估值要等 option 结束才完成一次更新；长技能会延迟信用，几乎不停止的技能也可能妨碍高层探索。若每个 option 恰好执行一步，该算法退化为普通 Q-learning：内部奖励仅有一项，后续折扣恰为 $\gamma$。这个特例可以用来检查奖励和时间索引。

<a id="lesson-intra"></a>

## 3. Intra-option：不等执行完，立即学习多个技能

到达 $s'$ 后，以概率 $1-\beta_o(s')$ 继续当前 $o$，以概率 $\beta_o(s')$ 结束并交回高层。对两个分支求平均，得到 arrival value $U$。因此它的定义直接来自 option 的执行规则。

$$
\begin{aligned}U(s',o)&=(1-\beta_o(s'))Q(s',o)+\beta_o(s')V(s')\\ Q(s,o)&=\sum_a\pi_o(a|s)\mathbb E\!\left[R+\gamma U(S',o)\mid s,a\right]\end{aligned}
$$

只走一个原始步，故这里乘一个 γ。β=0 恢复继续同一 option；β=1 恢复马上让高层选择。环境终止则 U=0，无论 β 预测多少。

若真实行为的动作概率为 $b(a|s)$，用这个动作估计另一 option $\pi_o$ 的动作平均时，可乘 $\rho_o=\pi_o(a|s)/b(a|s)$。确定性 options 中，只有同样会选择刚才动作的候选有贡献，其他候选的比率为零。在表格、固定 options 的条件下，经验因而可在多个行为问题间共享。

$$
\begin{aligned}\delta_o&=r+\gamma U(s',o)-Q(s,o)\\ Q(s,o)&\leftarrow Q(s,o)+\alpha\rho_o\delta_o,\qquad \rho_o=\frac{\pi_o(a|s)}{b(a|s)}\end{aligned}
$$

所有 target 先使用同一份旧 Q 计算，特别是 s=s′ 自环时。ρ 修正动作分布；访问不到的状态不会凭空学到，函数逼近与 off-policy 的稳定性也需另行处理。

高层控制使用 $V(s')=\max_{o':s'\in I_{o'}}Q(s',o')$，最大值只遍历在 $s'$ 合法启动的 options。继续当前 option 的 $Q(s',o)$ 不受启动 mask 限制，因为它已在更早的合法位置开始执行；只有终止后重新选择的分支受限。若评价给定 $\mu$，则在合法集合上按 $\mu$ 求加权平均。Intra-option 更新的是固定技能的价值，内部策略是否有用、如何学习，还要由下一节解决。

<a id="lesson-critic"></a>

## 4. Option-Critic：内部动作与何时停止是两个梯度

为了学习内部动作，定义 $Q_U(s,o,a)$：当前处于 $o$ 时，先执行 $a$，再遵守 option 的继续、停止与高层选择规则的回报。它比 $Q(s,o)$ 多条件化了当前动作；$U$ 则描述到达下一状态后的混合价值。三者可以共享网络表示，但对应不同条件。

$$
\begin{aligned}Q_U(s,o,a)&=\mathbb E[R+\gamma U(S',o)\mid s,a]\\ Q(s,o)&=\sum_a\pi_{o,\theta}(a|s)Q_U(s,o,a)\end{aligned}
$$

第一式给 critic 的 TD target，第二式把内部策略的动作平均还原成 option 价值。实际工程也常用 r+γU 作为当前 Q_U 的样本估计，未必单独存一个完整三维表。

固定 critic 作为局部评价器，对第二式中的当前动作概率求导，再将以后状态的递归影响展开，得到沿状态-option 占用分布加权的策略梯度。我们不需要显式微分环境转移，但必须在正确的轨迹/占用分布上采样。以下写出精确目标所对应的结构，再写常用的单样本方向。

$$
\begin{aligned}\nabla_\theta J&=\sum_{s,o}d_\gamma(s,o)\sum_a\nabla_\theta\pi_{o,\theta}(a|s)\,Q_U(s,o,a)\\ \Delta\theta&=\alpha_\theta\nabla_\theta\log\pi_{o,\theta}(a|s)\,[\widehat Q_U(s,o,a)-b(s,o)]\end{aligned}
$$

$d_\gamma$ 是从指定起点出发的折扣占用权重；$b$ 不依赖当前动作。第一式对应精确目标，第二式给出单个样本的更新方向；其无偏解释需要相应采样权重，普通逐步在线实现常以实际访问分布近似。

终止梯度可直接从混合式推导。保持当前 $Q$ 与 $V$ 不动，对 $U=(1-\beta)Q+\beta V$ 求终止参数导数，得到 $\nabla\beta(V-Q)$。当 $Q<V$ 时，增大停止概率能提高局部价值；反之应鼓励延续。递归展开这些贡献后得到终止梯度定理，其占用权重对应到达状态。

$$
\begin{aligned}A_\Omega(s',o)&=Q(s',o)-V(s')\\ \nabla_\vartheta U(s',o)&=-\nabla_\vartheta\beta_{o,\vartheta}(s')\,A_\Omega(s',o)\\ \Delta\vartheta&=-\alpha_\beta\nabla_\vartheta\beta_{o,\vartheta}(s')\,\widehat A_\Omega(s',o)\end{aligned}
$$

若 $\beta=\sigma(h)$，则 $\partial\beta/\partial h=\beta(1-\beta)$。当前 option 优势为负时，更新增大 $h$，使它更容易停止；优势为正时则鼓励继续。真实环境终止后不再存在继续与停止的选择。

**算法：算法伪代码**

1. 初始化 π_o、β_o、Q 或 Q_U；选择当前 option o
1. 每一步：
  1. 按旧 π_o 行动，记录 s,o,a,r,s′ 与真实 terminal
  1. 缓存旧 Q(s′,o)、V(s′)、β_o(s′)，构造 U 和 y=r+γU
  1. 用 y 更新 critic（真实 terminal 时 y=r）
  1. 用缓存的动作优势更新 π_o 的 log-probability
  1. 若非 terminal，用 −∇β_o(s′)[Q(s′,o)−V(s′)] 更新停止参数
  1. 按明确规定的 β 版本抽样停止；停止才按高层 μ 重新选 o
  1. s=s′；持续记录技能长度、选择频率、动作熵与主任务回报

实现中可先缓存更新前的 $\beta$ 与 critic，利用同一次转移构造两个 actor 的更新，再按缓存的终止概率决定是否换技能。这样执行规则和训练标签具有一致的时间顺序。动作熵、终止正则和切换代价可以减少技能塌缩或频繁切换，但它们改变的是优化目标，须与上面的基本梯度分别记录。

<a id="lesson-example"></a>

## 5. 从原始轨迹到两个梯度：完整手算

跨步估值：某 option 连续得到奖励 $1,2$，两步后停止；$\gamma=0.9$，停止状态的高层价值为 $10$。内部奖励和为 $1+0.9\times2=2.8$，后续贡献为 $0.9^2\times10=8.1$，总 target 为 $10.9$。若环境也在此终止，总 target 才变为 $2.8$。

一步 intra-option：$Q(s',o_1)=2$、$V(s')=6$、$\beta_1=0.25$，故 $U_1=0.75\times2+0.25\times6=3$。奖励为 $1$ 时 target 为 $3.7$。若 $Q(s,o_1)=0$、$b(a|s)=0.5$、$\pi_1(a|s)=1$、$\alpha=0.1$，则 $\rho_1=2$，更新后价值为 $0.74$。第二个 option 若 $\beta_2=1$、$\pi_2(a|s)=0.5$，则 target 为 $6.4$、$\rho_2=1$，更新后价值为 $0.64$。同一真实动作产生了两个不同控制问题的样本更新。

内部策略梯度：两个动作 logits 都为 0，故概率各为 0.5；抽到动作 0，动作价值估计为 3、baseline 为 1。softmax 的 log 概率梯度为 (0.5,−0.5)，乘优势 2 与步长 0.1，得到新 logits (0.1,−0.1)，动作 0 的概率变为约 0.549834。

终止梯度：到达状态的 $Q=2$、$V=6$，option 优势为 $-4$。终止 logit $h=0$，$\beta=0.5$、导数为 $0.25$；步长 $0.1$ 给出 $\Delta h=-0.1\times0.25\times(-4)=0.1$，新停止概率约 $0.524979$。内部 actor 在 $s$ 更新动作选择，终止 actor 在 $s'$ 更新是否继续。

<a id="lesson-code"></a>

## 6. 实现与实验：检查三个时间尺度

跨步 target、arrival value、多个 option 的一步估值、两种 actor 更新与技能内在奖励的最小实现。

```python
def option_target(rewards, gamma, next_value, terminal=False):
    if not rewards:
        raise ValueError("An option must execute at least one primitive step")
    accumulated = sum(gamma**k * r for k, r in enumerate(rewards))
    return accumulated + (0.0 if terminal else gamma**len(rewards) * next_value)


def arrival_value(q_continue, v_select, beta, terminal=False):
    return 0.0 if terminal else (1.0 - beta) * q_continue + beta * v_select


def intra_option_update(q, state, next_state, reward, beta_next,
                        target_action_prob, behavior_action_prob,
                        alpha=0.1, gamma=0.9, terminal=False,
                        next_initiation_mask=None):
    """One observed action updates every fixed option with support.

    Dictionaries q[s] are vectors over options. q snapshots make targets
    simultaneous, including the case state == next_state. The initiation mask
    restricts NEW option selections at next_state, not continuation of an
    already active option. Omit it only if every option can initiate there.
    """
    if behavior_action_prob <= 0.0:
        raise ValueError("Observed action must have positive behavior probability")
    here, nxt = q[state][:], q[next_state][:]
    mask = [True] * len(nxt) if next_initiation_mask is None else next_initiation_mask
    if len(mask) != len(nxt):
        raise ValueError("Initiation mask must match the number of options")
    if not terminal and not any(mask):
        raise ValueError("A nonterminal selection state needs a legal option")
    value = 0.0 if terminal else max(v for v, legal in zip(nxt, mask) if legal)
    for o in range(len(here)):
        rho = target_action_prob[o] / behavior_action_prob
        continuation = arrival_value(nxt[o], value, beta_next[o], terminal)
        delta = reward + gamma * continuation - here[o]
        q[state][o] = here[o] + alpha * rho * delta
    return q[state]


def softmax(logits):
    maximum = max(logits)
    exps = [math.exp(x - maximum) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def option_critic_actor_step(logits, chosen_action, q_action, baseline,
                             termination_logit, q_continue, v_select,
                             alpha_actor=0.1, alpha_beta=0.1):
    """Local sampled ascent directions; caller supplies OLD critic estimates.

    At environment terminal the caller must skip the termination update.
    This is not a complete deep Option-Critic training system.
    """
    probs = softmax(logits)
    advantage_action = q_action - baseline
    new_logits = [h + alpha_actor * advantage_action *
                  ((1.0 if a == chosen_action else 0.0) - probs[a])
                  for a, h in enumerate(logits)]
    beta = sigmoid(termination_logit)
    advantage_option = q_continue - v_select
    new_termination = termination_logit - alpha_beta * beta * (1.0 - beta) * advantage_option
    return new_logits, new_termination


def skill_intrinsic_rewards(log_q_z_given_s, log_prior_z,
                            log_q_next_given_skill, log_mixture_next):
    """Mechanism only: no discriminator, SAC, density training or entropy term."""
    return (log_q_z_given_s - log_prior_z,
            log_q_next_given_skill - log_mixture_next)
```

标准库运行；测试包含动作与终止局部梯度的中心有限差分

```sh
python3 knowledge_algorithms_lab.py options
python3 knowledge_algorithms_lab.py test
```

运行结果对应前面的手算：跨步 target 为 $10.9$，arrival value 为 $3$，两个估值为 $[0.74,0.64]$，动作 logits 为 $[0.1,-0.1]$，终止 logit 为 $0.1$。还应检查启动集合：下一状态价值为 $[1,100]$、只有第一个 option 可启动时，重新选择的价值是 $1$；第二个 option 若早已开始且 $\beta=0$，继续执行的价值仍可为 $100$。代码中的 sigmoid 分支避免了极端 logit 的指数溢出。

这份表格实现适合逐步观察更新顺序。继续学习深度 Option-Critic 时，作者 Atari 仓库提供卷积网络、目标网络、策略与终止损失以及完整交互循环；可沿相同的三个接口阅读：跨步选技能、原始步选动作、到达状态判断停止。

- 改动 A：把所有 β 设成 1，并限制一个 option 对应一个 primitive action；验证 target 退化到 Q-learning。
- 改动 B：把当前 option 优势从 −4 改为 +4；停止 logit 应下降。再把其设为 0，更新应为零。
- 改动 C：构造 s=s′ 自环，依次更新两个 options；比较未缓存旧 Q 的实现与本页同时 target 的差异。
- 改动 D：增加 option 平均长度，固定真实步数和计算预算，比较估值误差与控制回报；不能只按“高层决策次数”比较样本效率。

<a id="lesson-branches"></a>

## 7. 技能发现首先要选择什么行为值得学习

Option-Critic 直接用当前任务回报优化技能；另一类发现方法在没有当前任务奖励时，先积累覆盖环境或具有可区分效果的行为。它们与 option 形式兼容，但并不都学习可变的终止条件，也不都提供规划所需的 reward/end-state 模型。比较之前先问：发现信号是什么，技能怎样被调用，何时结束，下游需要什么接口？

结构发现、互信息发现和任务回报优化回答的问题不同。图结构方法寻找能跨越环境大尺度区域的行为；DIAYN 寻找可被区分的行为；DADS 进一步要求行为后果可预测；Option-Critic 则直接改进当前任务回报。先看后两种无监督准则，再推导 Machado 一线如何从环境结构得到技能。

$$
r_{\rm DIAYN}(s,z)=\log q_\phi(z|s)-\log p(z)
$$

DIAYN 采样技能标识 z，让条件策略产生能被判别器区分的状态。qφ 猜“哪个技能产生了这个状态”，固定先验 p 保持技能使用多样性；策略训练还包含动作熵项。这里不是外部目标奖励，也不自动约束一个技能的终点。

其最小训练循环是：采样 z 并固定一段交互；保存 (s,a,s′,z)；训练判别器预测 z；用当前判别器构造内在奖励训练条件策略；在统一外部任务上评价技能复用。需警惕技能仅凭无关背景区分、所有技能起点泄漏标签、判别器与策略相互追逐。技能可分辨并不自动意味着有用。

$$
r_{\rm DADS}(s,z,s')=\log q_\phi(s'|s,z)-\log\!\left(\sum_{z'}p(z')q_\phi(s'|s,z')\right)
$$

DADS 比较特定技能下状态变化的可预测性与混合技能模型。离散求和也可用先验采样近似；连续技能需要相应积分/采样。它同时学习技能动力学，因而比只有技能判别器更直接提供模型控制接口。

| 准则 | 怎样学习 | 对 CRL 的价值与缺口 |
| --- | --- | --- |
| 主任务回报 / Option-Critic | critic + 内部策略梯度 + 终止梯度 | 能面向当前任务优化；可能缺少将来任务需要的多样性 |
| 图结构 / eigenoptions | 图特征 → 内在差分奖励 → 策略与停止 | 有利于结构覆盖；不自动保留真实沿途奖励 |
| 可区分性 / DIAYN | 技能条件策略 + 状态判别器 + 熵 | 获得不同技能；没有自动得到完整规划模型 |
| 可预测后果 / DADS | 技能条件动力学 + 内在奖励 + 下游模型控制 | 使行为更易预测；仍需检查模型覆盖与任务奖励接口 |
| 未来可能有用 / reward-respecting | 真实奖励 + feature stopping value → option → model | 显式连接规划用途；候选特征与维护预算仍是研究问题 |

<a id="lesson-eigenoptions"></a>

## 8. Eigenoptions：把环境的慢变化方向变成行为

在有多个房间的迷宫里，随机游走很容易反复访问同一房间，却很久才穿过狭窄门口。因此，探索需要的不只是不同动作，还包括改变所在大区域的多步行为。Machado、Bellemare 与 Bowling 的 Laplacian option discovery（ICML 2017）用转移图的低频方向描述这种大尺度变化，再让策略主动沿这些方向移动。

$$
L_{\rm sym}=I-D^{-1/2}AD^{-1/2},\qquad L_{\rm sym}u_i=\lambda_i u_i,\qquad 0=\lambda_0\leq\lambda_1\leq\cdots
$$

A 是无向转移图的邻接权重，D 的对角线是节点度数。常数对应的平凡方向需按所用归一化转换后去除。低频特征在相连状态间变化缓慢，跨瓶颈却能形成明显差异；它描述的是图上的连通结构，而非观测像素距离。

选定一个非平凡状态特征 $e_i(s)$ 后，把沿该特征上升定义为子任务。对正、负两个方向分别训练，可以得到向相反区域移动的技能。策略仍按 RL 学习，并不是直接对特征做梯度就能得到环境动作。

$$
r_i(s,a,s')=e_i(s')-e_i(s),\qquad Q_i(s,a)\leftarrow Q_i(s,a)+\alpha\left[r_i+\gamma_i\max_{b\in\mathcal A\cup\{\perp\}}Q_i(s',b)-Q_i(s,a)\right]
$$

终止动作 ⊥ 的后续收益规定为零；当所有继续动作的内在价值都不大于零时，可以选择停止。技能的启动区域是仍值得继续的状态。γ_i 是该发现问题自己的折扣，未必等于主任务折扣。

这里的差分奖励不是“保持原任务最优策略不变”的一般奖励塑形。策略不变塑形通常使用 $\gamma\Phi(s')-\Phi(s)$；eigenoption 则有意定义一个新的内在控制问题。它的价值在于生成长程行为，不在于保证该行为已经优化外部任务。

**算法：Eigenoptions 的表格流程；图特征、技能学习与下游评价是三个不同阶段**

1. 构造状态转移图，求低频非平凡特征 $e_i$
1. 对每个特征及其相反方向：
  1. 定义内在奖励 $r_i=e_i(s\prime)-e_i(s)$
  1. 加入收益为零的终止动作 $\perp$
  1. 用控制算法学习内部策略，并由继续价值决定停止区域
1. 将学到的 options 加入行为集合
1. 在相同真实交互预算下测量覆盖，再测下游任务回报

完整转移图在像素和连续状态中不可得。Machado 等人的 Deep Successor Representation（ICLR 2018）转向从经验学习多步后果表示。对固定策略，SR 的每一行记录未来折扣状态访问；它可以通过 TD 学习，而不必先收集所有边再完整求图。

$$
M^\pi=\sum_{k=0}^{\infty}\gamma^k(P^\pi)^k=(I-\gamma P^\pi)^{-1},\qquad M^\pi(s,:)\leftarrow M^\pi(s,:)+\alpha\left[\mathbf e_s+\gamma M^\pi(s',:)-M^\pi(s,:)\right]
$$

这里采用包含当前状态的访问约定，e_s 是当前状态的 one-hot 向量。神经表示用特征替换 one-hot，再学习其未来累计。SR 与转移算子共享适当的谱结构；与对称 Laplacian 的对应还需要可逆性或合适的对称化，不能对任意有向动力学直接当作同一个矩阵。

由此得到一条清楚的变化：2017 年先有图再发现技能，2018 年开始从行为数据学习发现技能所需的结构。代价是表示依赖采样策略；没有访问过的房间，不会因为使用深度网络就自动出现在可靠的结构表示中。

<a id="lesson-discovery-cycle"></a>

## 9. 从静态特征到在线发现：ROD、DCEO 与 ALLO

技能改变探索，探索又改变表示，因而“先学表示、以后永久冻结技能”的分阶段方案并非唯一选择。Machado、Barreto、Precup 与 Bowling 在 JMLR 2023 的 Temporal Abstraction with the Successor Representation 中把这条反馈链称为 representation-driven option discovery：经验形成表示，表示定义子任务，子任务产生 options，options 再改变后续经验。

| 反馈环节 | 需要观察的量 | 可能失效的原因 |
| --- | --- | --- |
| 经验 → 表示 | 转移覆盖、谱方向、表示误差 | 只看已访问区域，遗漏窄门另一侧 |
| 表示 → 子任务 | 内在奖励分布、方向差异 | 冗余特征产生重复行为 |
| 子任务 → option | 成功率、执行长度、停止位置 | 低层训练不足或终止不合理 |
| option → 新经验 | 新区域访问与外部任务收益 | 高层过早偏爱熟悉技能，进一步缩小覆盖 |

DCEO（Deep Covering Eigenoptions，ICML 2023）把结构表示与技能发现放入深度 RL 的交互循环：从收集的转移学习 Laplacian 表示，以其方向构造探索技能，同时利用这些技能继续收集数据。它所推进的是大状态空间中在线获得覆盖性技能的问题；获得探索收益后，仍需单独学习外部任务价值或技能模型，才能讨论规划和任务复用。作者 mklissa/dceo 仓库提供了与深度 RL 基线相连的实现。

但“学到正确的低频子空间”还不等于“学到每一条明确的特征方向”。设两个坐标 $u_1,u_2$ 张成正确子空间，对它们作旋转仍可能得到相同的子空间损失；若技能奖励分别由两个坐标构造，旋转却会改变每个技能的行为。需要特征值缩放距离时，仅得到一个任意旋转的空间也不够。

$$
\min_{u_1,\ldots,u_k}\sum_{i=1}^{k}\langle u_i,Lu_i\rangle\quad\text{subject to}\quad\langle u_i,u_j\rangle=\delta_{ij}
$$

谱学习的基本形式同时要求平滑与正交。只最小化这一无序子空间目标，不能指定每个输出神经元对应哪个有序特征向量；这正是从子空间估计走向特征对估计时的区别。

Proper Laplacian Representation Learning（ALLO，ICLR 2024）通过带次序的约束、增广拉格朗日原始—对偶更新和特定停止梯度处理，学习有序的特征向量及特征值。它不是另一套 option 策略梯度，而是修复上游表示学习，使下游每个方向的含义与尺度更明确。作者 laplacian_dual_dynamics 的训练入口为 train_laprepr.py；读代码时应同时看向量损失、正交约束乘子和特征值读取，不能只截取平滑损失。

继续把任务价值加入高层选择，会发生什么？Value-Aware Eigenoptions（RLC 2025 Inductive Biases workshop）发现，固定 eigenoptions 能帮助信用分配，但在线发现时按任务价值选择技能也可能削弱探索覆盖。其核心实验把“已有技能怎么使用”和“后续技能从哪些经验中发现”分开。这个负面结果说明，任务回报、结构覆盖和长期技能维护之间需要明确的采样分工，而不是把高层控制器换成贪心就自然完成持续发现。

<a id="lesson-skill-composition"></a>

## 10. 从单方向技能到可组合技能：METRA 与 Laplacian Keyboard

Eigenoptions 常把一个特征方向对应为一个技能。如果下游目标位于两个方向之间，或者需要先沿一条方向再沿另一条方向运动，应如何组合？可以离散切换已有 options，也可以把连续的方向向量作为低层策略输入。后者把技能库变成可查询的条件策略族，但还需要规定方向的意义与执行时长。

METRA（ICLR 2024）从“可区分的技能可能仍然只在原地摆动”出发，让技能在与时间距离相关的潜在空间中产生大幅位移。它不是对 Laplacian 特征直接求特征分解，而是联合学习表示与方向条件策略，并用相邻状态的距离约束防止表示任意放大。

$$
\max_{\phi,\pi}\ \mathbb E\left[(\phi(S')-\phi(S))^\top Z\right],\qquad \|\phi(s')-\phi(s)\|_2\leq1\ \text{on observed adjacent states}
$$

这是理解 METRA 的受约束结构；实际算法用对偶与松弛处理约束。固定方向 Z 时，内积给出一步内在奖励；约束让累计潜在位移与需要经过的步数相联系。方向在采集一段轨迹时保持固定。

$$
\begin{aligned}\Delta\phi&=\phi(s')-\phi(s),\qquad d=\dim\phi(s),\\ c(s,s')&=1-\frac{\|\Delta\phi\|_2^2}{d},\qquad \widetilde c=\min\{c,\varepsilon_{\rm slack}\},\\ \mathcal L_\phi&=-\mathbb E\left[(\Delta\phi)^\top Z+\operatorname{sg}(e^b)\widetilde c\right],\\ \mathcal L_b&=b\,\mathbb E\left[\operatorname{sg}(\widetilde c)\right].\end{aligned}
$$

这是作者 iod/metra.py 中连续技能、dual_dist='one' 分支的实际尺度：平方差用维度平均而非求和，因此零残差对应 $\|\Delta\phi\|_2=\sqrt d$。$b$ 是对偶乘子的对数，$\varepsilon_{\rm slack}$ 对应 dual_slack；clamp(max=dual_slack) 只截断过大的正残差，不截断违反约束时的负残差，也不是双侧裁剪。两项损失分别最小化；负残差会增大 $b$，加强表示约束。理论逐对约束在实现中由样本上的软惩罚近似，不能将上述残差与单位欧氏球尺度直接等同。

METRA 在状态与像素控制任务中研究无奖励预训练和后续复用；这与长期变化环境中同时维护旧技能和学习新技能的完整 CRL 协议不同。作者 Seohong Park 的 METRA 仓库包含表示更新和 SAC 技能训练；实验时可比较“技能可区分但位移小”与“覆盖扩大”这两种不同结果。

Laplacian Keyboard（2026）把 ALLO 提供的谱特征与 successor features 联系起来。先在无任务奖励的经验上学习特征，再对不同权重向量训练相应的低层策略及其未来特征预测；下游既可由奖励回归得到一个固定权重，也可训练高层策略按当前状态选择权重，分段调用低层行为。

$$
\begin{aligned}r_w(s,a,s')&=w^\top\phi(s')\\ \psi(s,a,w)&=\mathbb E_{\pi_w}\left[\sum_{k=0}^{\infty}\gamma^k\phi(S_{t+k+1})\mid s,a\right]\\ Q^{\pi_w}_w(s,a)&=w^\top\psi(s,a,w)\end{aligned}
$$

每个 w 同时规定一个奖励方向和对应的目标策略。预测 ψ 时必须沿同一个 π_w 递归，不能对各特征分量分别取最大后再组合。这里特征计在到达状态，避免与上一节包含当前状态的 SR 约定混淆。

**算法：Laplacian Keyboard 的分层接口；高层选择连续技能参数，低层实现原始动作**

1. 无奖励阶段：
  1. 从经验学习 Laplacian 特征 $\phi$
  1. 采样权重 $w$，训练奖励为 $w^\top\phi(s\prime)$ 的低层策略与 SF
1. 有奖励下游阶段：
  1. 高层在当前状态选 $w$；低层按 $\pi_w$ 执行一段时间
  1. 累加真实外部奖励，记录实际长度 $\tau$
  1. 段结束时用 $R_{\mathrm{sum}}+\gamma^\tau V(s\prime)$ 更新高层
  1. 再根据新状态选择下一个 $w$

为什么还要高层学习？若外部奖励不能由有限特征线性表示，固定 $w$ 的直接迁移会产生表达误差。即使拟合奖励的误差不超过 $\varepsilon$，对固定策略也只能得到价值误差至多 $\varepsilon/(1-\gamma)$：逐步奖励误差沿折扣级数累加。状态相关的分段组合扩大了行为表达能力，却不等于有限个谱特征已覆盖所有任务的最优策略。

与原始 Option Keyboard 的 GPI 组合相比，这里连续权重不仅用于给旧策略重新打分，还成为低层策略族的条件与高层的动作。与 Option-Critic 相比，其技能意义主要来自预训练的谱奖励，而不是完全由当前外部回报端到端塑造。论文附录 H 给出了低层训练和高层 SMDP 训练的伪代码，可重点追踪累计奖励、真实段长和环境终止三项。

至此，技能发现有了三条可分别检验的研究问题：表示是否保留环境长程结构；行为是否沿这些结构可靠执行；调用方式是否真正帮助新任务。接着还需要第四个问题：能否预测这些行为的后果，并据此规划？reward-respecting 子任务保留真实沿途奖励，option models 预测真实后果，STOMP 将两者接入规划；这些接口与纯覆盖性技能互补，而非简单的新旧替代。

<a id="lesson-check"></a>

## 11. 诊断与自测

- 技能全部长度为 1：检查终止梯度符号、critic 初始化、β 是否被当作环境 done，以及高层选择是否有过强的即时切换优势。
- 所有技能完全一样：检查各技能是否获得不同学习信号、初始化/探索能否破坏对称；增加技能数量并不自动增加有效能力。
- 长期 option 价值偏高：检查 γ^τ 是否误写为 γ，是否漏掉沿途负奖励，以及技能模型是否过期。
- 只在训练目标有效：分别评价固定技能后重新学习高层的速度、技能覆盖、维护开销与长期遗忘。

自测 1：$\beta=1$ 是否使 $V(s')=0$？答：只是重新选择 option，真实环境终止才清零后续价值。自测 2：为何 intra-option 能学习没被执行的 option？答：共享已观察的一步后果，用动作兼容性或概率比修正当前动作分布，以 bootstrap 表示后续行为。自测 3：SF 是否就是 option？答：SF 预测固定策略的未来特征，option 定义行为与停止；一个 option 可以有 SF 模型，两者的职责不同。

## 本章的实验设计

计入技能发现和执行的原始环境步。将技能覆盖、使用频率和主任务收益分别记录。

设定：一个 option 连续得到奖励 1、2，γ=0.9，终点价值 3。随后加入随机持续时间、真实终端与外部截断。

- 两步目标为 5.23；真实任务终端时为 2.8。
- τ=1 退化到对应单步 backup。
- option 终止只让高层重选，不自动清整个 learner。

对照：primitive-only 与等动作保持时长；固定技能、随机技能和学习技能；冻结旧库、在线维护与重新学习

记录：primitive steps、option 次数和时长分布；终止地点、覆盖、失败退出和使用集中度；原任务收益与技能发现成本

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-modules)

## 学习与研究衔接

单步动作推广为可变持续时间的策略。发现、学习、选择和终止 option 是不同子问题。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-options) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=options) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=options)

<a id="chapter-code"></a>

## 下载与运行

SMDP、intra-option、动作与终止梯度的表格实验；深度技能训练另附原论文和作者工程。

[下载 knowledge_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/knowledge_algorithms_lab.py)

```sh
python3 knowledge_algorithms_lab.py options
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton, Precup & Singh — Between MDPs and semi-MDPs](https://doi.org/10.1016/S0004-3702(99)00052-1)：原始 options 框架。重点是 SMDP 最优方程、intra-option learning 与 option models，而不只读三元组定义。

- [Bacon, Harb & Precup — The Option-Critic Architecture](https://arxiv.org/html/1609.05140)：原文式(1)–(3)、内部策略梯度及终止梯度。对照 arrival 状态与动作采样状态，区分理论占用权重和在线实现。

- [Jean Harb — Option-Critic 作者 Atari 实现](https://github.com/jeanharb/option_critic)：作者工程；沿 critic target、policy loss、termination loss 读，并独立检查旧框架环境依赖。

- [Machado et al. — A Laplacian Framework for Option Discovery](https://arxiv.org/abs/1703.00956)：原文。比较覆盖结构的发现目标与基于主任务回报的技能优化。

- [Marlos Machado — options](https://github.com/mcmachado/options)：作者源码；分别追踪表示、内在奖励和技能求解，不将此仓库当作所有 options 方法的统一实现。

- [Machado et al. — Eigenoption Discovery through the Deep Successor Representation · ICLR 2018](https://arxiv.org/abs/1710.11089)：从完整图上的谱发现转向从经验学习未来特征；重点比较 SR 的策略依赖和原始图结构的关系。

- [Machado et al. — Temporal Abstraction in RL with the Successor Representation · JMLR 2023](https://jmlr.org/papers/v24/21-1213.html)：系统解释表示—子任务—技能—经验的反馈，以及 SR 支持时间抽象的机制。

- [Klissarov & Machado — Deep Covering Options · ICML 2023](https://proceedings.mlr.press/v202/klissarov23a.html)：DCEO 原文：深度表示、覆盖性技能与在线探索如何连接。

- [DCEO 作者实现](https://github.com/mklissa/dceo)：深度探索技能实现；先定位表示学习、内在奖励和行为采样，再与普通深度 RL 基线对照。

- [Proper Laplacian Representation Learning · ICLR 2024](https://arxiv.org/abs/2310.10833)：ALLO 原文：从低频子空间到有序特征对，解释为何方向与特征值对后续技能和规划重要。

- [ALLO 作者实现](https://github.com/tarod13/laplacian_dual_dynamics)：train_laprepr.py 是训练入口；同时阅读原始变量、对偶乘子与正交约束，而非仅看平滑损失。

- [Value-Aware Eigenoptions · RLC 2025 workshop](https://arxiv.org/abs/2507.09127)：区分固定技能对信用分配的帮助与在线价值驱动发现可能损害探索的结果；该工作属于 workshop。

- [Park, Rybkin & Levine — METRA · ICLR 2024](https://arxiv.org/abs/2310.08887)：时间距离约束下的方向条件技能；与仅优化可区分性的目标比较。

- [METRA 作者实现](https://github.com/seohongpark/METRA)：包含潜在表示与技能策略的联合训练；阅读时分别追踪内在奖励、距离约束和 SAC 更新。

- [METRA 核心文件 — iod/metra.py](https://github.com/seohongpark/METRA/blob/master/iod/metra.py)：_update_rewards 构造方向内在奖励；_update_loss_te 与 _update_loss_dual_lam 处理表示和约束；_optimize_op 进入技能策略训练。

- [Laplacian Keyboard · 2026](https://arxiv.org/abs/2602.07730)：谱特征、连续奖励权重、SF 与高层分段控制；附录 H 提供作者算法伪代码。

- [Machado — Deep RL Course 讲座讲义](https://deeprlcourse.github.io/assets/guests/marlos_machado.pdf)：结合图示复习环境表示、技能发现和探索反馈；可在读完谱发现部分后使用。

- [Eysenbach et al. — Diversity Is All You Need](https://arxiv.org/abs/1802.06070)：原文。关注互信息代理、技能先验与动作熵；可区分性不是下游任务价值。

- [Ben Eysenbach — SAC / DIAYN 作者代码](https://github.com/ben-eysenbach/sac)：作者历史工程；DIAYN 在 SAC 上加入技能输入、判别器和内在奖励，依次阅读这三个改动。

- [Sharma et al. — Dynamics-Aware Unsupervised Discovery of Skills](https://arxiv.org/abs/1907.01657)：原文。理解技能条件后果模型如何既提供发现奖励又进入下游控制。

- [Google Research — DADS 作者代码](https://github.com/google-research/dads)：原工程含技能学习与技能空间 MPC；入口 unsupervised_skill_learning/dads_off.py，按其配置区分训练与评估。
