# 多步学习：n-step、Tree Backup 与 Q(σ)

学习目标可以在一步 bootstrap 与完整回报之间选择，也可以在动作采样与动作期望之间选择。这是两条不同的设计维度。

## 本章内容

- 推导 n-step 回报及延迟更新时序。
- 理解 Tree Backup 的已采样分支和未采样分支。
- 从两端推导 on-policy Q(σ)，检查终止与退化条件。

<a id="problem-definition"></a>

## 本章的问题定义

固定预测问题后，决定使用多少个真实奖励，以及如何处理后继动作的不确定性。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 目标策略 $\pi$、行为策略 $b$、备份深度 $n$ 与当前估计 $V$；终止时间记为 $T$。

### 需要求解的对象

构造具有明确期望对象的多步学习目标；控制版本还需指定目标策略如何更新。

### 信息与数据权限

可以等待并保存一个有限窗口；离策略概率比需要行为概率和覆盖。

$$
G_{t:t+n}=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^nV(S_{t+n}),\quad t+n<T
$$

该目标估计固定策略价值；在真实终止处截断奖励和并去掉尾部自举。增加 n 只改变估计器，不能自动保证更高回报。

### 成立条件与解的含义

- 前向恒等式须规定备份中使用的参数版本；在线变化参数时不能直接套冻结参数推导。
- 离策略采样校正、策略期望和混合策略各有覆盖与有界性条件。

判断准则：验证一步极限、终止边界、确定性例子及离策略概率比的起止索引。

### 适用边界

- 把更长备份等同于更无偏、更低方差或更好的控制。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)：本章的多步目标可接入资格迹的后向实现；前向与后向更新是否等价取决于参数版本等条件。

- 改变信息或数据协议 · [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)：本章允许等待 n 步并保存窗口，相对严格流式增加延迟与存储。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

短备份依赖不准确的尾值；长备份增加等待、方差及离策略概率乘积。

### 本章的核心思路

把时间深度与动作采样方式分成两个独立的选择维度。

1. [展开奖励并保留一个尾值](#lesson-derive)：明确哪些后果是真实观察，哪些仍来自预测。

2. [未采样的动作分支用期望保留](#tree-backup)：Tree Backup 以策略概率传播后续误差，不等于简单乘整条轨迹比率。

3. [连续调节采样与期望](#q-sigma)：逐层的 σ 选择改变估计方式；两端退化情况可作为实现测试。

结论与条件：冻结估计下可精确核对目标代数关系；控制学习收敛仍需额外假设。

### 相关方法改变了什么

- n-step / λ-return：前者选择固定深度，后者混合多个深度。

- 采样 IS / Tree Backup / Q(σ)：分别使用概率校正、动作期望和逐层混合来处理行为差异。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 概率与期望

概率非负且总和为一；期望是按概率加权的平均，不是单次结果。条件期望只比较满足已知条件的情形。

### 递推与赋值

递推通过已有估计计算新估计。下标表示时间；更新符号表示程序中的赋值，不是数学恒等式。

<a id="lesson-setting"></a>

## 1 · 两条相互独立的选择轴

一步 TD 使用一个奖励再接预测；MC 等完整经历；n-step 使用若干实际奖励后才 bootstrap。另一方面，SARSA 采样下一动作，Expected SARSA 对下一动作求和。回报跨多长时间，与每层对动作采样还是求期望，是两个不同问题。

先考虑固定策略、表格预测、真实终点 T 与有限可积回报。记 $G_{t:h}$ 是从 t 开始、利用经验到 h 的目标，$h=\min(t+n,T)$。下面区分更新前的当前表与为推导恒等式而冻结的快照。

<a id="lesson-derive"></a>

## 2 · n-step 回报与等待

$$
G_{t:t+n}=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^nV_{t+n-1}(S_{t+n})
$$

重复展开回报递推 n 次，将仍未观察的余项替换为第 n 次转移到来前的价值估计。若先到终点，则只累积到 T，不加终点价值。

$$
V_{t+n}(S_t)=V_{t+n-1}(S_t)+\alpha[G_{t:t+n}-V_{t+n-1}(S_t)]
$$

第 t 时刻尚无未来 n 个奖励，实际最早在 t+n 才能执行这次更新。下标不是装饰，它规定了可用信息与更新时序。

开始时前 n−1 次转移没有可完成的更新。到终点后也不能立即清空缓冲区：最后 n−1 个未完成目标还要用已知终点逐一补齐。保存 n+1 个状态和 n 个奖励的环形缓冲区就足够，不需要永久保存整条生命。

n 较大通常减少对当前价值误差的依赖，但增加随机奖励、等待和行为失配的影响。不能仅凭“更接近 MC”断言更正确；不同 n 的有限样本误差由噪声、当前估计和访问分布共同决定。

<a id="multistep-control"></a>

## 3 · 动作价值与离策略校正

$$
G_{t:t+n}^{\rm Sarsa}=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^nQ(S_{t+n},A_{t+n})
$$

非终止边界还需要一个已选择的动作。若把最后一项改为目标策略加权平均，得到末端 Expected SARSA；这仍不同于在每一层做 Tree Backup。

$$
\rho_{t+1:\min(t+n,T-1)}=\prod_{k=t+1}^{\min(t+n,T-1)}\frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)}
$$

这是普通 n-step off-policy SARSA 对当前动作价值更新的一种比率约定。当前动作已被条件化，所以从 t+1 开始；非终止时采样 bootstrap 动作也需校正。

若末端直接对目标动作求期望，该末端动作没有被采样，不应再给它乘采样比率。完整轨迹权重、逐决策权重和带控制变量的目标还可以产生不同估计器。写代码前必须明确到底校正了哪一段随机选择。

<a id="tree-backup"></a>

## 4 · Tree Backup：保留未采样分支的预测

一条轨迹只真的走过动作 $A_{t+1}$。未走过的动作没有更深的实际奖励，仍使用当前 Q；走过的动作则用更深目标替代它原来的 Q。两类分支都按目标策略概率加权，就得到树备份。

$$
G_{t:h}^{\rm TB}=R_{t+1}+\gamma\left[\sum_{a\ne A_{t+1}}\pi(a\mid S_{t+1})Q(S_{t+1},a)+\pi(A_{t+1}\mid S_{t+1})G_{t+1:h}^{\rm TB}\right]
$$

非终止边界设 $G_{h:h}=Q(S_h,A_h)$，最后一层因此退化为 Expected SARSA。到达真实终点时 G=最后奖励。

$$
G_{t:h}^{\rm TB}=R_{t+1}+\gamma\left[\bar V(S_{t+1})+\pi(A_{t+1}\mid S_{t+1})(G_{t+1:h}^{\rm TB}-Q(S_{t+1},A_{t+1}))\right]
$$

先把所有动作的旧预测相加，再只对实际分支补上“深目标减旧预测”。这里 V̄=ΣπQ；不需要枚举未走分支的环境后果。

Tree Backup 不依赖普通轨迹比率，因此可以处理不同于目标的行为分支，但这不等于没有数据覆盖要求。目标概率为零的采样分支不会把更深信息传回；很小概率也会衰减深层信用。

<a id="q-sigma"></a>

## 5 · Q(σ)：逐层混合采样与期望

先固定 Q 快照并假定 on-policy。每一层用 σ 权重沿采样分支完全继续，用 1−σ 权重执行树备份。σ=1 是 n-step SARSA，σ=0 是 Tree Backup；中间值在每一层混合，而不只是把最终两个目标平均。

$$
G_{t:h}^{\sigma}=R_{t+1}+\gamma\left\{\sigma G_{t+1:h}^{\sigma}+(1-\sigma)\left[\bar V(S_{t+1})+\pi(A_{t+1}\mid S_{t+1})(G_{t+1:h}^{\sigma}-Q(S_{t+1},A_{t+1}))\right]\right\}
$$

边界与树备份相同。把括号按 G−Q 收集，系数变成 σ+(1−σ)π，这正是源码的 branch。

$$
\delta_t^\sigma=R_{t+1}+\gamma[\sigma Q(S_{t+1},A_{t+1})+(1-\sigma)\bar V(S_{t+1})]-Q(S_t,A_t)
$$

这是单层混合误差。在冻结 Q 下，深层误差还乘逐层 γ[σ+(1−σ)π] 的乘积；σ与 n 分别控制分支混合和展开长度。

论文还给出 off-policy Q(σ)，需要相应的重要性修正。这里的统一源码只实现 on-policy、冻结快照的 target，不包括一般 off-policy 训练循环。动态 σ、在线变参和资格迹版本需要分别定义时序。

<a id="multistep-algorithm"></a>

## 6 · 完整 n-step 执行顺序

**算法：算法伪代码**

1. 设 n≥1；初始化价值表与长度 n+1 的缓存。
1. 每次 episode 记录初始状态，暂令终点时间未知。
1. 每次时钟 t：
  1. 尚未终止时，执行动作并保存下一状态和奖励。
  1. 若到真实终点，记录 T；以后不再调用环境。
  1. 令待更新时刻 τ=t−n+1。
  1. 若 τ≥0，累积 τ 到 min(τ+n,T) 的奖励。
    1. 若 τ+n<T，接上当前表的边界预测；否则无 bootstrap。
    1. 更新 τ 对应的状态。
  1. 直到 τ=T−1，才完成本 episode。

实验为便于检查传入完整终止轨迹，但更新仍按上述延迟和补齐顺序执行。它不读取尚未到当前时钟的奖励。运行时若改为环形缓存，索引取模必须同时用于状态与奖励；真实终止后不能再选择一个不存在的动作。

<a id="lesson-example"></a>

## 7 · 手算与可运行源码

两步轨迹奖励为 1、3，γ=.9；下一状态 Q=(2,4)，目标概率 (.25,.75)，采样第一动作后终止。SARSA 的完整目标是 1+.9×3=3.7。Tree Backup 保留未采样动作：1+.9×(.75×4+.25×3)=4.375。σ=.5 得 4.0375。

随机游走在 1…5 五个非终止状态间等概率左右移动，从 3 开始；到 0 得零并结束，到 6 得一并结束，γ=1。真实价值 v(s)=s/6，由线性差分方程与边界值得到。默认 5000 回合、α=.02，n=1、3、8 的最终 RMSE 约 .01805、.02188、.02688；单次排序不是方法优劣结论。

<a id="lesson-code"></a>

## 8 · 边界与退化条件检查

延迟 n-step、终点补齐、冻结 Q(σ) target 与五状态随机游走的实际学习。

```python
def nstep_episode(states, rewards, values, n, alpha=0.1, gamma=1.0):
    """Terminal trajectory, online update order and terminal flush.
    Full trajectory is supplied for tests; live code needs only an n+1 ring buffer.
    """
    if n<1 or len(states)!=len(rewards)+1 or states[-1] is not None:
        raise ValueError("need n>=1 and true-terminal trajectory")
    terminal, values = len(rewards), values.copy()
    for t in range(terminal+n-1):
        tau = t-n+1
        if tau<0:
            continue
        end = min(tau+n, terminal)
        target = sum(gamma**(k-tau)*rewards[k] for k in range(tau, end))
        if tau+n<terminal:
            target += gamma**n*values[states[tau+n]]
        s = states[tau]
        values[s] += alpha*(target-values[s])
    return values

def qsigma_target(states, actions, rewards, q, policy, sigma, gamma=0.9):
    """Frozen-Q, ON-POLICY finite-horizon Q(sigma) target.
    A nonterminal endpoint requires its sampled action. No general off-policy
    importance correction is implemented. sigma=0: tree backup; sigma=1: Sarsa.
    """
    if not 0<=sigma<=1 or len(states)!=len(rewards)+1:
        raise ValueError("invalid sigma or trajectory")
    horizon = len(rewards)
    g = 0. if states[-1] is None else q[states[-1]][actions[-1]]
    for k in reversed(range(horizon)):
        sp = states[k+1]
        if sp is None:
            if k!=horizon-1:
                raise ValueError("terminal must be last")
            g = rewards[k]
        else:
            ap = actions[k+1]
            expected = sum(p*v for p, v in zip(policy[sp], q[sp]))
            branch = sigma+(1.-sigma)*policy[sp][ap]
            g = rewards[k]+gamma*(branch*(g-q[sp][ap])
                +sigma*q[sp][ap]+(1.-sigma)*expected)
    return g

def random_walk(n, episodes=5000, seed=7):
    rng = random.Random(seed)
    values = {s: .5 for s in range(1,6)}
    for _ in range(episodes):
        state, states, rewards = 3, [3], []
        while state not in (0,6):
            sp = state+rng.choice([-1,1])
            rewards.append(float(sp==6))
            states.append(None if sp in (0,6) else sp)
            state = sp
        values = nstep_episode(states, rewards, values, n, alpha=.02)
    return {"values": values, "reference": {s:s/6 for s in values},
            "rmse": math.sqrt(sum((values[s]-s/6)**2 for s in values)/5)}
```

源码把两类实验分开：随机游走会实际采样经历并执行 n-step 学习；Q(σ) 函数则接收冻结表与一条指定轨迹，只计算 target。对后者应先检查 σ=0、σ=1、单步、确定性策略与真实终止五种边界，再考虑接入会变化的在线价值表。通过 target 检验并不等于完整训练器已经实现。

改变随机游走的起点，会改变各状态的访问频率，却不改变固定左右策略的真实价值。分别输出逐状态误差和均方根误差，可以看到总指标掩盖的覆盖差异。若改变左右动作概率，则策略本身改变，原来的 s/6 不再是正确参照，应先重新解 Bellman 方程，不能继续使用旧解析答案。

<a id="lesson-branches"></a>

## 9 · 与资格迹、流式协议和 CRL 的关系

n-step 固定一个反馈长度；λ-return 组合多个长度；资格迹把过去方向压缩为递推状态。资格迹不是过去观测的记忆，因此不自动解决部分可观测问题。普通在线迹与冻结快照下的误差展开也不应混同。

持续流没有终点时，n-step 可以在缓存满后持续更新，但最末暂停位置只是截断。长期变化使等待 n 步期间策略或环境改变，较长目标可能收集更多实际结果，也可能包含更多旧情境。应同时研究传播速度、偏差、方差与每步存储。

<a id="lesson-check"></a>

## 10 · 带答案练习

问：n 大于 episode 长度，最前几个状态会永远不更新吗？答：不会，终止后要继续补齐，所有目标都退化为对应完整回报。立即清空缓存是错误。

问：Tree Backup 是否生成了未访问动作的新奖励？答：没有。未走分支使用已有 Q，只有实际分支使用更深的经验；“树”是目标展开而非实际并行采样。

问：目标策略在采样分支概率为一时，σ两端还不同吗？答：该层树备份只剩采样分支，与 SARSA 相同。代码测试这个退化条件。



<a id="chapter-code"></a>

## 下载与运行

Python 3，仅标准库。下载完整文件后在其目录运行；页面展示核心区域。小环境和解析检验用于理解机制，不是论文基准复现。

[下载 tabular_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/tabular_textbook_lab.py)

```sh
python3 tabular_textbook_lab.py multistep
python3 tabular_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：对应第 2–8 章。以下推导、例子和标准库实现独立编写；原书用于核对定义、算法关系与适用条件。

- [David Silver · UCL 强化学习课程](https://davidstarsilver.wordpress.com/teaching/)：原课程页面提供 MDP、动态规划、预测、控制、探索和规划的讲义及视频。

- [De Asis 等 · Multi-Step Reinforcement Learning: A Unifying Algorithm](https://arxiv.org/abs/1703.01327)：原始 Q(σ) 论文；区分采样程度、展开长度与 on/off-policy 修正。

<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)
- [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)
- [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)

对应原始材料：7.1–7.6。本文为原创讲解，原书、论文与上游代码保留各自许可。
