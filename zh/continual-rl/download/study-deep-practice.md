# 深度 RL 的训练循环：经验、目标与更新

现代深度强化学习 · 第 6 章

一条真实经验怎样经过采样、保存、构造目标和求导，最后改变下一次行动？

## 本章内容

- 从回报目标推到 DQN 的样本标签与损失，定位停止梯度、终止处理和优化器写入。
- 沿实际 DQN、PPO、SAC 实现辨认行为参数、目标参数与批内固定量，解释下一次经验怎样改变。
- 分开环境步、梯度步、目标同步与冻结评价，并按原记录复算读数。
- 说明模型、递归记忆和离线数据各改变哪项接口，再选择必要的深入分支。

## 阅读这条论证

从控制目标和 Bellman 方程出发，沿一条真实经验推到样本标签、损失、梯度和下一次行动。DQN 给出完整循环；PPO 改变数据复用和策略更新，SAC 还改变优化目标。最后再看记忆、离线数据和模型各自改变了哪些条件。

[从收益到学习目标](https://yingwen.io/zh/continual-rl/foundations/deep/practice/#lesson-setting) → [任务与采样边界](https://yingwen.io/zh/continual-rl/foundations/deep/practice/#lesson-example) → [DQN 的完整循环](https://yingwen.io/zh/continual-rl/foundations/deep/practice/#lesson-algorithm) → [为何需要这些机制](https://yingwen.io/zh/continual-rl/foundations/deep/practice/#lesson-dqn-mechanisms) → [PPO 怎样复用一批轨迹](https://yingwen.io/zh/continual-rl/foundations/deep/practice/#lesson-ppo-loop) → [SAC 怎样使用旧经验](https://yingwen.io/zh/continual-rl/foundations/deep/practice/#lesson-sac-loop) → [对应实现与记录](https://yingwen.io/zh/continual-rl/foundations/deep/practice/#lesson-code)

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [深度价值学习：DQN、Double DQN 与目标的时间顺序](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/)：掌握采样、回放、标签和参数更新的基本循环。
- [策略梯度：从轨迹概率到 GAE 与 actor–critic](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/)：掌握rollout、优势和策略更新之间的数据关系。


### Bellman 递推

当前价值等于即时奖励加折扣后的后继价值；预测目标与最优控制目标不同。

### 函数与梯度

神经网络把参数和输入映射为输出。链式法则必须指明哪些量参与求导，哪些量作为固定目标。

### 经验分布

on-policy 数据由当前策略产生；off-policy 数据可来自旧策略，但能否正确复用取决于算法目标和数据覆盖。

<a id="problem-definition"></a>

## 本章的问题定义

在已有有限期限小世界中，让真实采样、数据保存、目标构造和参数更新共同产生下一次行动。

### 给定条件与符号

- 初始分布、奖励、真正终止时刻 $T$ 与训练折扣 $\gamma$；DeadlineChain 将剩余时间计入观察，任务期限是12步。
- 环境时钟 $t$、优化时钟 $k$、在线参数 $\theta_k$、目标参数 $\bar\theta$；一次环境步可对应零次或多次优化。
- 数据批 $\mathcal B_k$、采样时行为策略与按算法固定的标签；参数、优化器和缓存共同决定实际写入。

### 需要求解的对象

将收益目标落实为有明确定义的数据和更新循环，分清局部损失下降、当前策略评价和学习期间收益。

### 信息与数据权限

行为决定真实经验，replay只能复用已发生后果；标签使用声明的参数版本；冻结评价在独立环境中运行，不写入主训练流。

$$
\max_\theta J_\gamma(\pi_\theta)=\mathbb E_{\pi_\theta}\!\left[\sum_{t<T}\gamma^tR_{t+1}\right],\qquad g_k=\left.\nabla_\theta L_k(\theta;\mathcal B_k,\xi_k)\right|_{\theta=\theta_k}
$$

左侧规定控制所追求的收益，右侧在当前参数处计算本次批次损失的梯度；ξ包含此次固定的目标、旧概率或优势。DQN的回归损失、PPO的代理目标各由自己的推导与数据条件联系到收益，二者不直接等于收益函数。后面的SAC还将策略熵加入收益准则，须相应重定义价值和更新目标。

### 成立条件与解的含义

- 环境终止、采样截止和目标同步分别处理；停止梯度固定本次求导的对象，不保证目标估计准确。
- 目标、采样权限、更新次数与评价策略须明确；数据协议改变时重新检查估计条件。

判断准则：能从一条真实转移追到标签、梯度和下一参数版本，并从原始记录解释独立评价。边界、梯度与完整状态恢复用于检验实现，控制收益另作独立实验。

### 适用边界

- 损失下降或实现检查通过，不能替代原论文效能复现。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)：把更新语义、评价对象、独立学习过程与预算接在一起。

- 改变信息或数据协议 · [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)：递归状态与历史重建改变网络可使用的信息，梯度截断另行规定。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

控制目标规定希望得到的后果，却没有直接给出网络能计算的梯度。算法必须说明怎样用实际数据构造目标，以及哪条求导路径联系到要改进的预测或动作。

### 本章的核心思路

先从 Bellman 关系推到 DQN 的一次更新，再比较 PPO 与 SAC 改变了哪些数据、目标和梯度条件。

1. [从行为目标得到可学习的关系](https://yingwen.io/zh/continual-rl/foundations/deep/practice/#lesson-setting)：奖励回报规定控制问题；Bellman 关系连接当前动作与后继价值，再用真实转移形成样本目标。

2. [确认保存的是哪个后继与边界](https://yingwen.io/zh/continual-rl/foundations/deep/practice/#lesson-example)：任务期限在状态内；rollout 截止不产生真正终止，重置后的首观测不是上一转移的后继。

3. [沿经验推到下一次参数更新](https://yingwen.io/zh/continual-rl/foundations/deep/practice/#lesson-algorithm)：分开环境步与优化步；推导固定标签的梯度，再对照目标副本、裁剪和优化器的实际次序。

4. [说明旧轨迹怎样用于新策略](https://yingwen.io/zh/continual-rl/foundations/deep/practice/#lesson-ppo-loop)：旧概率、旧价值与优势各有来源；比率校正当前动作，不自动把旧轨迹变成新策略轨迹。

5. [区分两种动作及其梯度](https://yingwen.io/zh/continual-rl/foundations/deep/practice/#lesson-sac-loop)：旧转移用于 critic，当前策略另采样动作用于 actor；熵项改变了目标，actor 还需要沿动作经过 critic 求导。

6. [把推导对应到实现与记录](https://yingwen.io/zh/continual-rl/foundations/deep/practice/#lesson-code)：教材训练脚本与独立算法旧结果各用自己的参数、批次和评价定义。

结论与条件：实现核对验证已检查的更新和边界；当前损失改善不保证闭环收益改善或长期适应。

### 相关方法改变了什么

- 局部损失 / 策略收益：分别评价本次优化批与按策略产生的后果。

- 参数 checkpoint / 完整状态恢复：完整恢复还包括优化器、记忆、缓存、计数与随机状态，是实现核验手段。


<a id="lesson-setting"></a>

## 1 · 一条经验如何进入下一次行动

前几章已经给出价值网络与策略网络的更新。现在把它们放回实际交互：智能体先按当时的参数选动作，环境返回奖励和下一观测；学习器保存这条经验，用它构造目标并更新参数；新参数随后改变行动，进而改变以后收到的经验。损失函数只是这条循环中的一段。

先规定循环要改善什么。在一个 episode 内，令 $T$ 为真正终止的时刻，$R_{t+1}$ 为动作之后收到的奖励，$S_t$ 为能够预测后果的状态。初始状态分布固定，策略 $\pi$ 决定行动；环境的奖励和转移规律决定结果。DQN 通过学习动作价值来改善策略，PPO 直接优化策略参数，二者仍须先有可比较的回报目标。

$$
J_\gamma(\pi)=\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}\gamma^tR_{t+1}\right],\qquad Q^*(s,a)=\sup_\pi\mathbb E_\pi\!\left[\sum_{j=0}^{T-t-1}\gamma^jR_{t+j+1}\mid S_t=s,A_t=a\right]
$$

$J_\gamma$ 评价从共同起点出发的策略；$Q^*$ 先指定当前动作，再比较之后的策略。有限期限包含在状态中，真正终止后的回报为零。本章的 DQN 取 $\gamma=0.99$，PPO 取 $\gamma=1$；冻结评价另报告未折扣奖励和。

若状态满足马尔可夫条件、动作集合有限，最优动作价值满足下面的 Bellman 关系：先发生一次真实转移，再在后继状态选择价值最大的动作。学习器并不知道这个条件期望，也不知道 $Q^*$；网络输出 $Q_\theta$ 只是估计。训练循环要解释的，就是怎样用已经收到的奖励和后继观察去改写这个估计。

$$
Q^*(s,a)=\mathbb E\!\left[R_{t+1}+\gamma(1-D_{t+1})\max_{a'}Q^*(S_{t+1},a')\mid S_t=s,A_t=a\right]
$$

$D_{t+1}=1$ 表示这次转移使任务真正终止。最大值比较后继行动，条件期望平均环境可能产生的后果；二者承担不同的计算。将真实价值换成网络、将期望换成观测样本，才会得到后面的自举标签。

本章沿已有 DeadlineChain 的 DQN/Double DQN 教学实现追踪这两次替换，再比较同一任务中的 PPO。连续动作 SAC 的实际更新核用于辨认另一条 actor 梯度路径。模型、递归记忆和离线数据分别改变循环中的一个接口，后面按这些改变进入相应分支。

| 时钟或版本 | 在循环中何时改变 | 它决定什么 |
| --- | --- | --- |
| 环境步 $t$ | 每次 env.step 后加一 | 新增一条真实转移；从零起算的循环索引 t 对应第 t+1 条转移 |
| 梯度步 $k$ | 每次 optimizer.step 后加一 | 在线网络与优化器状态各发生一次写入 |
| 在线参数 $\theta_k$ | 梯度更新之后 | 下一次动作选择与当前 minibatch 的预测 |
| 目标参数 $\bar\theta$ | 明确的同步事件之后 | 后继价值标签使用哪一份延迟参数 |
| 采样时行为 $\mu_t$ | 动作发生时确定 | 由动作当时的网络版本与探索规则决定；更新时可能已经过时 |

在单线程 DQN 中，动作 $A_t$ 读取的是 env.step 之前已完成的最新版本 $\theta_{k(t)}$；该步收集经验之后才可能产生 $\theta_{k(t)+1}$。这些计数器不必一起前进。尚未攒够 minibatch 时，可以继续收集经验而没有梯度步；复用同一批数据时，可以有多个梯度步而没有新环境步。删除 replay、增加优化次数或提前同步目标，分别改变数据复用、写入预算和标签变化速度。

<a id="lesson-example"></a>

## 2 · 先辨认环境真正返回的边界

内置任务有五个位置，动作是左或右，位置四为目标。到达目标奖励一，其余步骤奖励 −0.02。令位置为 $p_t$、剩余步数为 $n_t$，则本任务状态为 $S_t=(p_t,n_t)$，观察用五维 one-hot 加 $n_t/12$ 完整编码它。十二步是任务期限：同在位置零，剩余四步可以连续向右到达目标，剩余一步只能得到 −0.02。省略剩余时间，会把不同价值的状态混成同一个输入。最短四步到达的未折扣回报是 $-0.02-0.02-0.02+1=0.94$；折扣0.99下为 $-0.02(1+0.99+0.99^2)+0.99^3=0.910897$。

这项十二步期限规定了回报在哪结束，耗尽时尾值为零。另一个常见情况是原任务仍会继续，但采样器因外部时间上限重置环境；那是截断（truncation），目标仍要读出截断前最后观测的价值。若把自动 reset 后的新起点保存成这条转移的 next_observation，就将两段不同经历错误连在了一起。Time Limits 原论文分别分析了这两种目标，不能只根据“timeout”这个接口名字判断。

| 边界 | bootstrap后继价值 | GAE连接下一行 | 随后采样从哪里开始 |
| --- | --- | --- | --- |
| 到达目标或本任务十二步期限 | 否；仍保留末步奖励 | 否 | reset后的新起点 |
| 外部时长上限使一个仍可继续的任务重置 | 是；使用重置前最后观测 | 否；不跨接新episode | reset后的新起点 |
| PPO rollout恰在episode中间收满 | 是；使用实际后继观测 | 本批末尾不再读取下一批残差 | 本任务保持当前状态继续 |

PPO 默认128步的 rollout 长度只是优化批次长度。它可以含多个完整 episode 和一个未完成的尾段；批次收满时不调用 reset。尾段最后的 TD 残差保留后继价值，优势递推则在本批末尾停止。代码用两个 mask 分开“任务还会不会产生回报”和“下一条残差是否属于本次连续片段”，后面的 PPO 节会逐项展开。

下面的 deep_textbook_train.py 中，DQN 使用折扣0.99，PPO使用1。冻结评价都累加未折扣环境奖励，因此“训练采用什么目标”和“读数评价什么”也要分别说明。若要比较算法，应先统一目标、预算和评价规则；本章用这两个入口检查执行顺序。

DeadlineChain 将剩余时间写入观察，十二步截止属于任务本身的真终止。

```python
class DeadlineChain:
    """Five positions; right reaches reward 1. The observed deadline is 12.

    Goal or deadline is genuine task termination. A rollout boundary in PPO
    is only a sampling cutoff and does not reset this environment.
    """
    def reset(self):
        self.position, self.remaining = 0, 12
        return self.observe()

    def observe(self):
        return torch.tensor([float(i == self.position) for i in range(5)]
                            + [self.remaining/12.0], dtype=torch.float32)

    def step(self, action):
        self.position = min(4, max(0, self.position + (1 if action else -1)))
        self.remaining -= 1
        reached = self.position == 4
        return self.observe(), (1.0 if reached else -0.02), reached or self.remaining == 0


def evaluate(actor, episodes=10):
    scores = []
    with torch.no_grad():
        for _ in range(episodes):
            env, score, done = DeadlineChain(), 0.0, False
            observation = env.reset()
            while not done:
                action = int(actor(observation).argmax())
                observation, reward, done = env.step(action)
                score += reward
            scores.append(score)
    return sum(scores)/len(scores)
```

<a id="lesson-algorithm"></a>

## 3 · 采样、保存、构造目标、求导与更新

从 train_dqn 中的一次 env.step 开始。动作由当前在线网络的 ε-greedy 规则产生。环境返回的 next_observation、reward、terminated 与动作前观测、实际动作组成五元组，先进入 replay；若真正终止，下一次采样才从 reset 后的观察继续。这条转移保存的是终止前的后继观测，而非新回合的起点。

**算法：deep_textbook_train.py 的 DQN/Double DQN 循环；一条新转移不保证马上被 replay 抽中。**

1. 按当前在线网络选择动作，执行一次 env.step
1. 保存 (observation, action, reward, next_observation, terminated)
1. 真正终止则准备新回合，否则保留后继观测
1. replay 至少有32条且零起算循环索引为偶数时，抽32条经验
1. 以当前在线参数计算已执行动作的预测；以目标参数构造尾值
1. 固定标签，反传当前预测的损失；裁剪梯度，调用 optimizer.step
1. 每100个真实环境步，把更新后的在线参数复制给目标网络
1. 随后用当前在线参数选择新的动作

先固定这一批的记录 $(s_i,a_i,r_i,s'_i,d_i)$。$s_i$ 在此是观察所完整编码的 DeadlineChain 状态；$a_i$ 是采样时已经执行的动作。当前预测取 $Q_\theta(s_i,a_i)$，不会把它替换成当前网络偏好的动作。普通 DQN 把前面的真实最优价值换成目标网络，再用这一条已观察转移替代环境条件期望；Double DQN 进一步分开后继动作的选择与评价。

$$
\begin{aligned}y_{k,i}^{\rm DQN}&=r_i+0.99(1-d_i)\max_aQ_{\bar\theta_k}(s'_i,a),\\a^*_{k,i}&=\arg\max_aQ_{\theta_k}(s'_i,a),\\y_{k,i}^{\rm Double}&=r_i+0.99(1-d_i)Q_{\bar\theta_k}(s'_i,a^*_{k,i}).\end{aligned}
$$

$\bar\theta_k$ 表示第 $k$ 次优化所读取的目标副本，未同步时相邻版本数值相同。两种标签都包含实际奖励与估计尾值；只有尾值估计恰当时，它们的条件平均才接近真正的 Bellman 目标。一次随机后果也不等于这个平均。

得到32个标签以后，将它们作为本次优化的固定数值。为了区分“在哪个参数处生成标签”和“对哪个参数求导”，以下把损失写成 $L_k(\theta)$，最后在 $\theta=\theta_k$ 处取梯度。脚本的 smooth_l1_loss 使用阈值一、批均值，等于下面的 Huber 损失。

$$
\begin{gathered}e_i(\theta)=Q_\theta(s_i,a_i)-\operatorname{sg}(y_{k,i}),\qquad L_k(\theta)=\frac1{32}\sum_{i=1}^{32}h(e_i(\theta)),\\h(e)=\begin{cases}\tfrac12e^2,&|e|\le1,\\|e|-\tfrac12,&|e|>1,\end{cases}\qquad h'(e)=\operatorname{clip}(e,-1,1),\\g_k=\left.\nabla_\theta L_k(\theta)\right|_{\theta_k}=\frac1{32}\sum_i\operatorname{clip}(e_i(\theta_k),-1,1)\nabla Q_{\theta_k}(s_i,a_i).\end{gathered}
$$

$\operatorname{sg}$ 保留标签数值、令它在这次求导中的导数为零。偏低的当前预测产生负残差，这条样本的负梯度贡献指向提高该预测的局部方向；合并其他样本并经过优化器后，实际变化还需另算。clip 在这里限制每条残差的导数，不等于裁剪奖励或限制整个网络的梯度范数。

no_grad 包围的是整个标签计算，连 Double DQN 的在线后继动作选择也不参与反传。冻结目标网络参数控制它跨优化步怎样变化；停止梯度控制本次计算图经过哪些路径。即使只有一个网络、仍将同网络尾值 detach，得到的也是只对当前预测求导的半梯度更新。固定标签时，这个 g 是本次回归损失的完整偏梯度；下一次重新生成标签后，整个训练过程并没有在下降同一个固定监督目标。

在本实现里，目标副本已requires_grad_(False)，离散argmax也不传递导数，因此仅删掉no_grad还不会使在线梯度沿目标返回。若改成由同一可更新网络提供可微尾值，并让它参与求导，才多出“改变后继预测以改变标签”的路径，优化对象也随之改变。对样本自举残差完整求导，还不自动得到平方期望Bellman残差的梯度：前者先平方再平均随机标签，后者先取条件平均再平方，两者相差标签的条件方差。

$$
\tilde g_k=g_k\min\!\left(1,\frac{10}{\|g_k\|_2+10^{-6}}\right),\qquad(\theta_{k+1},z_{k+1})=\operatorname{AdamStep}(\theta_k,z_k,\tilde g_k;\alpha=0.003)
$$

脚本先将合并后的梯度全局范数裁到10，再调用 Adam；$z_k$ 是一阶矩、二阶矩和已完成更新数等优化器状态。单纯 SGD 才写成 $\theta_{k+1}=\theta_k-\alpha\tilde g_k$。零梯度时上式的缩放为一，公式中的小量对应实际裁剪的数值保护。

这次写入会影响其他输入。对一个未在本批直接监督的 $(s,a)$，小更新下有 $\Delta Q(s,a)\approx\nabla Q_{\theta_k}(s,a)^\top\Delta\theta$；共享表示使它可能上升、下降或几乎不变。因而“一个动作的标签正确”还没有说明其他动作排序怎样变化。target network 暂时延迟这项变化进入尾值，却不阻止在线预测、贪心动作或后续数据一起变化。

再回到 train_dqn 的下一轮。若刚才终止，当前观察已经是位置零、剩余十二步；否则仍是刚收到的实际后继。ε-greedy 的贪心分支读取更新后的在线输出：在起点向右会到位置一并收到 −0.02，向左则留在位置零并收到 −0.02，后续可到达目标的时间因此不同。只有新输出改变动作排序且本次未选择探索分支时，参数写入才在这一步改变真实动作；学习率、replay损失或预测均值变化都不能单独断定动作已经改变。

目标同步发生在该轮可能的优化之后。每100个环境步复制一次在线参数；同步不新增经验，也不再调用优化器。下一次抽到旧转移时，它的奖励和后继仍相同，但预测与重算标签可能不同。至此，一条数据被保存、重新解释、用于写入，并通过下一次行动回到新的环境经验。

最后查看 evaluate：它新建评价环境，以当前网络的 argmax 动作累加未折扣外部奖励，并在 no_grad 下运行。它没有调用优化器，也不把转移放入 replay。得到的读数评价当前确定性策略；训练中的 ε-greedy 行为收益与学习成本需要由训练流另行计算。若扩展归一化器或递归网络，还须检查它们的统计量和活动状态是否在评价中改变。

这个 DeadlineChain 的初始状态和转移都是确定的；同一个已冻结网络的十次 argmax评价通常给出相同回报。十次 episode 并没有产生十份独立的训练结果。独立训练种子对应不同学习过程，同一过程里的相邻窗口也没有成为新的独立试验。先固定训练预算、调参数据、失败规则和指标，再报告全部计划运行。rliable 的区间与汇总服务于这个比较，不替代信息权限和采样单位的定义。

对照上面的标签与损失：no_grad 固定目标，Huber loss 的梯度只写入在线网络，梯度裁剪后调用优化器。

```python
def dqn_update(online, target, optimizer, batch, gamma=.99, double=True):
    observations, actions, rewards, next_observations, terminated = zip(*batch)
    x, xp = torch.stack(observations), torch.stack(next_observations)
    action = torch.tensor(actions, dtype=torch.long)
    reward, terminal = torch.tensor(rewards), torch.tensor(terminated, dtype=torch.float32)
    prediction = online(x).gather(1, action[:, None]).squeeze(1)
    with torch.no_grad():
        target_values = target(xp)
        if double:
            selected = online(xp).argmax(dim=1)
            tail = target_values.gather(1, selected[:, None]).squeeze(1)
        else:
            tail = target_values.max(dim=1).values
        y = reward + gamma*(1-terminal)*tail
    assert prediction.shape == y.shape == (len(batch),)
    loss = nn.functional.smooth_l1_loss(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    return float(loss.detach())
```

再把更新放回完整循环：保存真实转移、抽取 minibatch，并每100个环境步同步目标网络。

```python
def train_dqn(steps=2000, seed=0, double=True):
    random.seed(seed)
    torch.manual_seed(seed)
    online = mlp(6, 2)
    target = copy.deepcopy(online).requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=.003)
    replay = deque(maxlen=4000)
    env, initial_score = DeadlineChain(), evaluate(online)
    observation, losses = env.reset(), []
    for t in range(steps):
        epsilon = max(.05, 1-t/max(1, steps*.7))
        with torch.no_grad():
            action = random.randrange(2) if random.random() < epsilon else int(online(observation).argmax())
        next_observation, reward, terminated = env.step(action)
        replay.append((observation, action, reward, next_observation, terminated))
        observation = env.reset() if terminated else next_observation
        if len(replay) >= 32 and t % 2 == 0:
            batch = random.sample(list(replay), 32)
            losses.append(dqn_update(online, target, optimizer, batch, double=double))
        if (t+1) % 100 == 0:
            target.load_state_dict(online.state_dict())
    return {'algorithm': 'Double DQN' if double else 'DQN', 'seed': seed,
            'environment_steps': steps, 'updates': len(losses),
            'initial_greedy_return': initial_score, 'final_greedy_return': evaluate(online),
            'last_loss': losses[-1] if losses else None, 'scope': 'DeadlineChain only'}
```

<a id="lesson-dqn-mechanisms"></a>

## 3.1 · 为什么要分别控制数据、标签与求导路径

上面的循环用了 replay 和 target network，也能在 DQN 与 Double DQN 间切换。它们作用在不同位置。Replay 将一个已有转移再次提供给更新器；目标副本决定尾值何时接收新参数；Double DQN 决定用谁选择后继动作、用谁评价。只有先保持目标与预算一致，关闭一个机制后看到的差异才容易归因。

| 循环中的机制 | 它改变哪项依赖 | 仍然留下的问题 |
| --- | --- | --- |
| 均匀replay抽样 | 本批来自有限buffer，减少连续邻接经验在更新中的直接相关，并允许重复使用 | buffer覆盖和年龄决定拟合权重；同一buffer的各批仍共享数据，世界变化后旧转移可能失配 |
| 冻结目标副本 | 在线参数这一步的改变暂不进入目标网络尾值 | 标签可能错误或过时；参数共享与离策略自举的困难仍在 |
| 标签停止梯度 | 本次求导只经过当前预测，不经过尾值 | 没有使标签独立于历史数据，也没有使标签成为真实价值 |
| Double选择/评价 | 在线网络选动作，延迟网络评价该动作 | 两个网络相关且定期相等；没有普遍无偏或非线性收敛保证 |

为什么旧策略的转移还能供 Q-learning 使用？记录中的动作已固定，学习器要估计的是“执行这个动作之后”的条件后果。在同一 MDP 中，条件规律 $p(r,s',d\mid s,a)$ 不因采样时用哪个策略而改变；目标中的后继贪心选择也不要求下一步照旧行为行动。因此普通一步 Q-learning 标签不需要乘旧动作的重要性比。行为策略仍决定哪些 $(s,a)$ 能进入 buffer，以及它们出现多少次。

$$
\mathcal R_k(\theta)=\frac1{|\mathcal D_t|}\sum_{e\in\mathcal D_t}h\!\left(Q_\theta(s_e,a_e)-\operatorname{sg}(y_k(e))\right),\qquad\mathbb E[g_k\mid\mathcal D_t,\theta_k,\bar\theta_k]=\left.\nabla_\theta\mathcal R_k(\theta)\right|_{\theta_k}
$$

均匀、无放回抽取32条时，批均值梯度的条件期望等于当前有限buffer风险的偏梯度。这只认证这一次抽样对应什么平均；$\mathcal D_t$ 的经验权重并不自动等于当前策略的访问权重或起点目标 $J_\gamma$ 的梯度。

函数逼近用有限参数同时拟合多个状态动作；改变 buffer权重，就可能改变这些误差之间的取舍。若观察缺少必要历史，同一个输入下的后果混合还可能随行为改变，此时前面的条件规律也不再由该输入完整确定。若真实动力学或奖励变了，旧转移则来自另一规律。离策略算法允许行为与目标策略不同，并没有将缺失状态、覆盖不足和外部变化一起解决。

DQN 的最大值同时选择并评价高输出动作，估计误差可使偏高的输出更容易被选中。Double DQN用另一版本评价，缓解这项选择偏差；但在同步瞬间两个网络相同，两种标签也相同。同步之间，普通DQN对一条给定转移的标签保持固定，Double的在线argmax却仍可改变，带动标签跳到另一动作的目标输出。慢目标、固定标签和固定后继动作不是同一个条件。

采用 Huber 又改变了残差如何参与拟合。平方损失的总体最小值对应标签条件均值，Huber 的一般条件是平均截断残差为零；对偏斜、含大误差的随机标签，它可能对应不同位置。DeadlineChain本身的后果确定，但网络估计和buffer分布仍在变化。不能把Huber只解释成数值实现，也不能用loss变小代替真实回报改善。

原书§11.3把函数逼近、自举和离策略训练放在同一个稳定性问题中。Replay和目标副本在实际DQN中提供有用的时间尺度与采样选择，仍没有将这三项移除。判断某次失败，需要分别检查目标数值、行为覆盖、共享表示与更新尺度，而不只问是否“用了稳定化技巧”。

<a id="lesson-ppo-loop"></a>

## 3.2 · PPO：先完成一批经验，再固定其学习含义

同一DeadlineChain中的PPO改变了更新节奏。记第 $j$ 批开始时 actor 为 $\pi_{\eta_j}$、critic 为 $V_{\psi_j}$。在128次真实交互完成之前，两组参数都不更新；采样仍可能经历多次任务终止与reset。每行保存观察、实际动作、旧动作log-probability、奖励、旧价值、实际后继价值和终止标志。保存log-probability足以在之后重建已采动作的旧概率，无须再用已更新的actor回算它。

先用这批旧critic构造优势。令 $d_i$ 表示真终止，$b_i$ 表示连续片段边界：真终止、外部重置截断或本批末行都结束残差连接。bootstrap mask $1-d_i$ 决定是否读后继价值；trace mask $1-b_i$ 决定是否继续接下一行残差。这里先写一般 $\gamma$，实际PPO入口取1。

$$
\begin{aligned}\delta_i&=r_i+\gamma(1-d_i)V_{\psi_j}(s'_i)-V_{\psi_j}(s_i),\\\hat A_i&=\delta_i+\gamma\lambda(1-b_i)\hat A_{i+1},\qquad\lambda=0.95,\\\hat R_i&=V_{\psi_j}(s_i)+\hat A_i.\end{aligned}
$$

从批末往前计算，末尾以 $\hat A_B=0$ 起算。代码只把真终止传入 boundaries，本任务没有外部重置截断；初始化 carry=0 已使批末不跨接下一批。$\hat R_i$ 是给critic的固定回归标签，不会随着批内critic更新重算。

将递推展开，得到当前残差加上按 γλ衰减的后续残差，直到片段边界。λ=0只用一步残差，λ=1则将奖励展开到本片段末并保留必要尾值；一般λ混合不同等待长度。旧critic给了尚未收到的后果一个估计，因此真实回报与自举误差共同进入优势。detach只固定这次求导路径，无法去掉critic估计误差。

对照两个mask进入delta和carry的位置；returns在优势标准化之前构造。

```python
def gae(rewards, values, next_values, terminated, boundaries,
        gamma=0.99, lam=0.95):
    n = len(rewards)
    if not all(len(x) == n for x in
               (values, next_values, terminated, boundaries)):
        raise ValueError('one next value and two masks per transition')
    advantages, carry = [0.0] * n, 0.0
    for t in reversed(range(n)):
        bootstrap, trace = masks(terminated[t], boundaries[t])
        delta = rewards[t] + gamma * bootstrap * next_values[t] - values[t]
        carry = delta + gamma * lam * trace * carry
        advantages[t] = carry
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def score_gradient(probabilities, action, advantage):
    # Derivative of A * log softmax(logits)[action] with fixed A.
    return [advantage * (float(i == action) - p)
            for i, p in enumerate(probabilities)]
```

进入ppo_update之后，旧log-probability、$\hat A_i$ 和 $\hat R_i$ 固定。代码只为actor将优势按本批均值与总体标准差标准化为 $\tilde A_i=(\hat A_i-\overline A)/(\sigma_A+10^{-8})$；critic标签仍使用原始 $\hat A_i$。标准化改变有限批次的尺度与中心，不能把它和真实、未标准化优势的无偏梯度混为一个对象。

$$
\begin{gathered}\rho_i(\eta)=\exp\!\left(\log\pi_\eta(a_i\mid s_i)-\operatorname{sg}(\log\pi_{\eta_j}(a_i\mid s_i))\right),\\L_{\rm actor}(\eta)=-\frac1B\sum_i\min\!\left(\rho_i(\eta)\tilde A_i,\operatorname{clip}(\rho_i(\eta),0.8,1.2)\tilde A_i\right),\\L_{\rm critic}(\psi)=\frac1B\sum_i\left(V_\psi(s_i)-\operatorname{sg}(\hat R_i)\right)^2.\end{gathered}
$$

优化器执行梯度下降，所以actor对要提高的代理取负号。只有当前动作log-probability向 $\eta$ 求导；只有当前价值预测向 $\psi$ 求导。本代码actor与critic参数分开：更新critic不会通过共享encoder直接改变actor。

在批内第一次actor更新前，$\rho_i=1$。尚未进入clip平坦区时，梯度下降方向为 $\frac1B\sum_i\rho_i\tilde A_i\nabla_\eta\log\pi_\eta(a_i\mid s_i)$，接回策略梯度的score路径；后续epoch重新计算 $\rho_i$，而不是重新采动作、估优势或奖励。正优势在比率超过1.2后失去进一步增大的该项激励，负优势在比率低于0.8后失去进一步减小的该项激励。共享参数和其他样本仍会改变这个动作概率，所以clip没有强制所有比率留在区间内。

旧概率比修正给定已采状态上的动作权重；状态依然来自旧策略访问。批内优化把这个有限数据代理改好，尚没有测到新策略的真实回报。下一批开始时，新actor成为采样策略，新的状态、动作、奖励与critic估计再产生一批新优势。若每个epoch把旧log-probability改成当前值，比率便反复回到一，失去“与采样时策略比较”的含义。

实际核至多执行四次全批actor更新，每次写入前检查 $\frac1B\sum_i[\rho_i-1-\log\rho_i]$ 是否超过0.03；超出则提前停止。这个非负样本估计用旧动作数据近似旧到新KL，不是所有状态上的精确KL。最后一次被允许的更新仍可能越过阈值，随后返回的 sample_kl 才测到更新后结果。critic另执行四次更新；actor提前停止不会减少这里的critic次数。

先冷读采样阶段：两网络版本固定，128步结束后才算GAE并调用更新。

```python
def train_ppo(epochs=20, seed=0, batch_steps=128):
    torch.manual_seed(seed)
    actor, critic = mlp(6, 2), mlp(6, 1)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=.003)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=.01)
    env, initial_score = DeadlineChain(), evaluate(actor)
    observation = env.reset()
    log = {}
    for _ in range(epochs):
        xs, actions, logps, rewards, values, next_values, terminals = [], [], [], [], [], [], []
        for _ in range(batch_steps):
            with torch.no_grad():
                distribution = Categorical(logits=actor(observation))
                action = distribution.sample()
                logp, value = distribution.log_prob(action), float(critic(observation).item())
            xp, reward, terminal = env.step(int(action))
            with torch.no_grad():
                next_value = float(critic(xp).item())
            xs.append(observation); actions.append(action); logps.append(logp)
            rewards.append(reward); values.append(value); next_values.append(next_value); terminals.append(terminal)
            observation = env.reset() if terminal else xp
        # gamma=1 matches this finite-horizon, undiscounted task objective.
        advantages, returns = gae(rewards, values, next_values, terminals, terminals, gamma=1., lam=.95)
        log = ppo_update(actor, critic, actor_opt, critic_opt, torch.stack(xs),
                         torch.stack(actions), torch.stack(logps),
                         torch.tensor(advantages), torch.tensor(returns))
    return {'algorithm': 'PPO-Clip', 'seed': seed, 'environment_steps': epochs*batch_steps,
            'initial_greedy_return': initial_score, 'final_greedy_return': evaluate(actor),
            **log, 'scope': 'DeadlineChain only; full-batch updates'}
```

再读批内阶段：旧量detach，actor和critic分别写入，最后重新计算sample_kl。

```python
def ppo_update(actor, critic, actor_optimizer, critic_optimizer, x, actions,
               old_logp, advantages, returns, epochs=4, clip=.2, target_kl=.03):
    # Old log-probabilities, advantages and targets stay fixed for all epochs.
    old_logp, returns = old_logp.detach(), returns.detach()
    advantages = advantages.detach()
    advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    updates, final_kl = 0, 0.0
    for _ in range(epochs):
        distribution = Categorical(logits=actor(x))
        logp = distribution.log_prob(actions)
        ratio = (logp-old_logp).exp()
        kl = ((ratio-1)-(logp-old_logp)).mean()
        if float(kl.detach()) > target_kl:
            break
        surrogate = torch.minimum(ratio*advantages,
                                  ratio.clamp(1-clip, 1+clip)*advantages)
        loss_actor = -surrogate.mean()
        actor_optimizer.zero_grad()
        loss_actor.backward()
        actor_optimizer.step()
        updates += 1
    for _ in range(epochs):
        prediction = critic(x).squeeze(-1)
        assert prediction.shape == returns.shape
        loss_critic = ((prediction-returns)**2).mean()
        critic_optimizer.zero_grad()
        loss_critic.backward()
        critic_optimizer.step()
    with torch.no_grad():
        difference = Categorical(logits=actor(x)).log_prob(actions)-old_logp
        final_kl = float((difference.exp()-1-difference).mean())
    return {'actor_updates': updates, 'sample_kl': final_kl,
            'value_loss': float(loss_critic.detach())}
```

<a id="lesson-sac-loop"></a>

## 3.3 · SAC：旧转移训练critic，当前动作训练actor

SAC同时复用旧转移和学习随机连续策略，因此先改变价值含义。固定温度 $\alpha>0$ 时，比较准则在奖励之外加入策略熵：$\mathbb E\sum_{t=0}^{T-1}\gamma^t[R_{t+1}-\alpha\log\pi_\phi(A_t\mid S_t)]$。连续动作的 $\pi$ 是密度，熵与动作坐标有关。以下保持配套核的动作范围 $[-1,1]^2$；更完整的动作变换和温度约束见[熵正则控制章](https://yingwen.io/zh/continual-rl/foundations/deep/entropy-control/)。

本章使用deep_textbook_train.py的sac_update核，对应两个在线critic、两个目标critic、一个当前actor和log_alpha。它没有在DeadlineChain上运行SAC，也没有给出SAC的完整环境采样器；小任务连续控制的完整循环在独立implementations/deep/入口。这里检查的是：同一个replay行在critic步骤与actor步骤中究竟被怎样读取。

$$
\begin{aligned}a'_i&=f_{\phi_k}(\epsilon'_i;s'_i),\qquad\epsilon'_i\sim\mathcal N(0,I),\\y_i&=r_i+\gamma(1-d_i)\left[\min_{\ell=1,2}Q_{\bar\omega_{\ell,k}}(s'_i,a'_i)-\alpha_k\log\pi_{\phi_k}(a'_i\mid s'_i)\right],\\L_Q(\omega_1,\omega_2)&=\frac1B\sum_i\sum_{\ell=1,2}\left[Q_{\omega_\ell}(s_i,a_i)-\operatorname{sg}(y_i)\right]^2.\end{aligned}
$$

$\omega$ 是critic参数，$\phi$ 是actor参数；$k$在这里标记一次sac_update调用的入口版本。critic拟合的是数据已执行动作 $a_i$；尾值动作 $a'_i$ 则由本次当前actor重新采样。no_grad固定整个标签，$\gamma=0.99$，终止mask仍只作用于尾值。

replay提供奖励和真实后继，当前actor提供从后继继续的行动分布。SAC的soft Q把当前奖励和以后行动的熵回报合起来，当前动作的熵由actor目标另行计算。两个critic取较小值是控制乐观估计的一项选择，仍可能有估计偏差；目标critic的慢版本也不使尾值正确。

$$
a_i^{\rm new}=f_\phi(\epsilon_i;s_i),\qquad L_\pi(\phi)=\frac1B\sum_i\left[\alpha_k\log\pi_\phi(a_i^{\rm new}\mid s_i)-\min_{\ell=1,2}Q_{\omega_{\ell,k+1}}(s_i,a_i^{\rm new})\right]
$$

先完成critic写入，再以其更新后的参数指导actor。数据中的状态 $s_i$ 固定，actor重新生成动作，数据原动作 $a_i$ 不进入这个actor损失。代码将critic参数requires_grad置False，却保留函数对动作输入的导数。

这里的f由Normal.rsample生成高斯变量，再用tanh映射动作；固定基础噪声时，动作可随actor参数连续变化。梯度既经过log-density，也经过critic对动作的导数再回到actor。冻结critic参数表示“不写critic”，不表示“critic输出对动作是常数”。若将整个actor目标包进no_grad，或将新动作detach，就切断这条路径。PPO使用旧数据动作的score，SAC在旧数据状态上为当前动作求路径导数，两种复用方式需要不同的保存量。

温度步骤另用 $\zeta=\log\alpha$ 保证正值。该核下降 $L_\alpha(\zeta)=-\frac1B\sum_i e^\zeta\operatorname{sg}(\log\pi_{\phi_k}(a_i^{\rm new}\mid s_i)+\mathcal H_{\rm target})$，目标熵默认−2。这里的logp由本次actor写入之前的前向采样得到；actor写入后不重算它，只对温度求导。若该批估计熵低于目标，平均括号为正，下降会增大温度，提高下一次actor更新对熵的权重。

$$
\bar\omega_{\ell,k+1}=(1-\tau)\bar\omega_{\ell,k}+\tau\omega_{\ell,k+1},\qquad\tau=0.005
$$

该核最后对两个critic目标参数作Polyak写入。这个事件与三个优化器的写入都没有产生新环境步；完整采样器下一次调用当前actor行动时，更新才影响真实经验。

依次定位目标no_grad、critic写入、保留动作导数的critic冻结、actor写入、温度写入及目标同步。

```python
def sac_update(actor, q1, q2, target_q1, target_q2, log_alpha,
               actor_opt, critic_opt, alpha_opt, batch, target_entropy=-2., gamma=.99, tau=.005):
    x, action, reward, xp, terminal = batch
    alpha = log_alpha.exp().detach()
    with torch.no_grad():
        next_action, next_logp = actor(xp)
        soft_value = torch.minimum(target_q1(xp, next_action), target_q2(xp, next_action))-alpha*next_logp
        y = reward+gamma*(1-terminal)*soft_value
    loss_q = ((q1(x, action)-y)**2+(q2(x, action)-y)**2).mean()
    critic_opt.zero_grad(); loss_q.backward(); critic_opt.step()
    freeze([q1, q2], True)
    sampled_action, logp = actor(x)
    assert logp.shape == reward.shape
    loss_actor = (alpha*logp-torch.minimum(q1(x, sampled_action), q2(x, sampled_action))).mean()
    actor_opt.zero_grad(); loss_actor.backward(); actor_opt.step()
    freeze([q1, q2], False)
    # Exact log-alpha parameterization of the dual loss, not its common proxy.
    loss_alpha = -(log_alpha.exp()*(logp.detach()+target_entropy)).mean()
    alpha_opt.zero_grad(); loss_alpha.backward(); alpha_opt.step()
    update_targets([q1, q2], [target_q1, target_q2], tau)
    return {'critic_loss': float(loss_q.detach()), 'actor_loss': float(loss_actor.detach()),
            'alpha': float(log_alpha.exp().detach())}
```

DQN和SAC的旧转移都能训练条件后果，但replay状态权重与当前策略访问权重仍有差别。SAC在这些状态上提高soft策略代理，不等于每步取得起点熵回报的无偏梯度；精确表格soft policy iteration的结论也不能直接赋给交替训练的非线性网络。下一节回到实际命令和记录，检查这些差别在输出中留下了哪些证据。

<a id="lesson-code"></a>

## 4 · 把循环中的位置对应到代码与记录

先在文件所在目录运行 python3 deep_textbook_lab.py test；可选 PyTorch 环境中运行 python3 deep_textbook_train.py test，检查目标、梯度和边界。deep_textbook_train.py 与 deep_textbook_lab.py 放在同一目录，依赖按 deep_requirements.txt 安装。实现检查先回答公式是否被按声明执行，后续训练才回答随机学习过程表现怎样。

已有小任务训练入口；输出 JSON。DQN 的1600是环境步，PPO的16轮各采128步，共2048环境步。

```bash
python3 deep_textbook_train.py dqn --steps 1600 --seed 0
python3 deep_textbook_train.py ppo --epochs 16 --seed 0
```

| 入口 | 数据与训练目标 | 输出怎样解释 |
| --- | --- | --- |
| deep_textbook_train.py / train_dqn | γ=0.99；replay抽32条；每100环境步同步目标 | environment_steps 与 updates 分开；final_greedy_return 为同一冻结argmax策略的10次episode均值 |
| deep_textbook_train.py / train_ppo | γ=1、λ=0.95；默认128真实步形成一批；批内最多4次actor更新 | epochs 是采样轮数；返回的actor_updates仅为最后一批实际actor写入数 |
| 独立 implementations/deep/ppo.py 与已保存 deep-ppo 结果 | γ=1、λ=0.95；旧1200步在20个记录网格前更新，每批60步 | samples 是累计采样步；updated_batches 是累计采样批，value 为12次冻结argmax评价的均值 |

这两个 PPO 入口使用相同小世界，但批次长度、评价次数与输出格式不同。独立实现的旧 CSV 能复算五个训练种子的同一评价点，却没有逐次 actor 更新计数；不能把另一个入口的最后一批计数补到旧记录。阅读 [PPO完整实现与旧结果](https://yingwen.io/zh/continual-rl/code/deep-ppo/) 时，以包内当时源码和 config 为依据。

DQN的更新次数可以从循环规则独立复算：1600条环境转移对应零起算t=0至1599，第32条后buffer足够，首次满足偶数索引的是t=32；此后每隔二步写入一次，到t=1598共784次。目标同步16次。两项都不是1600，也不能由最后的loss反推出这一训练过程。PPO的16轮共2048环境步，actor每批实际写入数可变；脚本只返回最后一批计数，不能当成累计次数。

标准库文件给出可直接复算的算子，PyTorch 文件给出小任务的真实采样与优化。模型、离线和递归控制的完整作者工程各有自己的训练器；这里的模型误差、正则项与敏感度核，用来检查对应接口，不能替代那些系统的完整实验。

<a id="lesson-branches"></a>

## 5 · 沿循环的接口选择下一项问题

读通一次采样与更新后，再问哪个条件需要改变：当前观察缺信息，就研究记忆；实际交互昂贵，就研究学得模型怎样参与计算；只能使用既有数据，就研究覆盖与外推。下面三条深入阅读各保留自己的推导，读者可按这项缺口选取。

| 机制 | 新增加的信息 | 必须维持的关系 |
| --- | --- | --- |
| 模型学习 | 预测奖励、下一状态或潜在状态 | 规划使用的模型应与任务目标和动作语义一致 |
| recurrent state | 历史顺序、初始 hidden state、边界 | 网络状态重建与参数梯度截断分别定义 |
| offline RL | 固定数据集与行为覆盖 | 不能依赖未授权的新环境交互纠正外推 |
| CRL | 持续时间线、变化机制、资源与 reset 权限 | 学习期间的损失和适应成本不能只由最终冻结评估代替 |

数据记录也包括观测归一化器与奖励变换的状态。若训练过程中更新归一化统计，旧 replay 的原始观测与已归一化特征会产生不同语义；必须选择保存哪一种，并保持实验一致。

在分布式系统中，经验还可能等待多个参数版本。采样策略、推理参数和学习参数的延迟由 [大规模训练](https://yingwen.io/zh/continual-rl/foundations/deep/systems/) 进一步展开。持续学习则同时评价当下收益、知识保持和以后还能否学会新目标；下表从具体失败选择干预。

| 观察到的问题 | 需要分开的解释 | 最小区分实验 |
| --- | --- | --- |
| 变化后回报下降 | 旧 replay 失配、critic 标签漂移、策略覆盖不足 | 固定网络容量，对照新数据与混合数据，并记录关键动作覆盖 |
| 训练越久越难学 | 表示退化、优化器状态、目标尺度改变 | 重置输出层、优化器或部分单元，分别测同一新目标的学习曲线 |
| 长延迟奖励学不到 | 历史状态不足、信用窗口太短、探索不到 | 给充分状态、延长梯度窗口、提供探索轨迹，分别作 oracle 对照 |
| 模型规划反而变差 | 模型误差、规划分布偏离数据、计算预算变化 | 真实模型与学习模型对照，并固定总计算预算 |

每个对照只改变一个机制更容易解释，但 oracle 也需要标出信息权限。例如给真实隐状态可以定位表示瓶颈，却不能作为同等信息下的部署方法；允许重置环境的实验也不能直接声称解决不可重置单生命期控制。

<a id="rlss-plasticity-protocols"></a>

## 可塑性证据怎样读：五类实验、三种秩与必要的对照

深入阅读：比较可塑性研究的任务、测量与对照，作为进入持续学习的延伸阅读。

“训练很久以后收益下降”是观测，不是机制诊断。新目标可能更难，数据可能变化，旧策略也可能访问不到有用区域。可塑性实验要问一个更窄的问题：在匹配的新目标、数据和学习预算下，已经训练过的学习器是否比合适的新初始化更难学习？训练误差、泛化误差和旧知识保持须分别观察。

| 原研究协议 | 主要隔离的问题 | 不能据此推出什么 |
| --- | --- | --- |
| Continual ImageNet：顺序二分类任务，每任务训练多轮，并重置输出层 | 重复学习新分类问题时的学习能力；隐藏表示跨任务保留 | 已知任务边界与头重置不是无边界严格流式协议 |
| 类增量 CIFAR-100：类别逐步增加，旧数据仍可训练 | 在数据仍可获得时，旧网络与从头训练网络的差别 | 类别变多本身会变难；准确率下降不单独证明遗忘 |
| Online Permuted MNIST：连续输入置换、逐样本更新 | 长序列新映射上的适应，较少跨任务语义共享 | 独立置换上的可塑性不等于有结构任务的快速迁移 |
| Slowly-Changing Regression：固定目标网络，输入的一部分缓慢翻转 | 没有 TD 或控制闭环时，表示与优化怎样适应分布变化 | 小网络监督结果不是深度 CRL 控制结果 |
| 长时 PPO：固定动力学或改变 Ant 摩擦系数 | 外部变化和策略自身诱发的数据变化下的学习 | PPO 有 rollout buffer 与多轮更新，不是单样本即弃 |

Nature 原研究的 PPO 配置用 2048 步 buffer、10 个 epoch、128 大小 minibatch；策略与价值网络各有两层 256 单元。长期运行中也比较了标准 Adam 与调节后的两个动量系数。因此复现“PPO 加 CBP”时，不能遗漏基础优化器、L2、数据复用和总训练时长，随后把所有差异都归因于神经元替换。

$$
r_{99}(H)=\min\left\{k:\frac{\sum_{i=1}^k\sigma_i}{\sum_i\sigma_i}>0.99\right\},\quad
r_{\rm ent}(H)=\exp\!\left(-\sum_i p_i\log p_i\right),\quad p_i=\frac{\sigma_i}{\sum_j\sigma_j}.
$$

H 是同一探测输入上得到的表示矩阵，奇异值按降序排列。r99 是这项工作使用的阈值秩定义之一，r_ent 是奇异值分布的熵有效秩。零矩阵需另外定义约定；不能直接除以零。

$$
r_{\rm stable}(H)=\frac{\|H\|_F^2}{\|H\|_2^2}=\frac{\sum_i\sigma_i^2}{\sigma_1^2}.
$$

常见的稳定秩是另一个量。若奇异值为 3 和 1，则 r99=2，熵有效秩约 1.755，稳定秩为 10/9。三个数字不能都简称“rank”并横向比较。

原论文的部分图使用 stable rank 这个名称，但相应 Methods 给出的是上面的 99% 奇异值质量阈值。读取结果时应按计算定义对应，而不是按标签猜测它一定是 Frobenius 范数与谱范数的比值。

高秩只说明在这些输入上存在多个方向，不说明这些方向对应未来任务。非零梯度、较少休眠单元和较小权重，也都不是充分的可塑性证书。诊断曲线与性能共同变化提供线索；要归因，还需分别干预尺度、激活、表示方向或优化器，并保持其他因素和预算一致。

一个清楚的试验可以先保存同一时刻的已有网络，复制到多组。让已有网络、同容量新网络、仅重置优化器的网络在完全相同的新数据流上学习。记录即时误差、达到指定误差所需样本、旧目标误差及计算成本。新网络是诊断基线，不是必须部署的方案；不可重置世界也未必允许重新获得同一批经验。

实验长度也是问题设定的一部分。一个方法可能在前几次任务变化时受益于迁移，经过更长时间才显现可塑性下降。短 SCR 实验适合检查更新和统计量，不能代替数百万步的长期检验。改变网络、输入变化速度或初始化后，没有出现退化也是有价值的结果：它帮助确定现象的发生条件，而不是应当删除的异常曲线。

一个更严格的 CRL 问题是同时维持当前表现、旧知识和未来学习能力。CBP 的当前贡献效用主要针对前两者中的当前贡献与可塑性，不会自动保存很久未用的技能。多 GVF、多 option 共享表示时，谁的损失决定一个单元可以被删除，仍须明确。

<a id="course-plasticity-diagnosis"></a>

## 5.1 · 保留旧知识与学会新知识是两个检验

深入阅读：分别检验保留旧知识和学习新知识，避免把两种失败归为同一原因。

遗忘检验问：训练新问题以后，旧问题的表现是否变差？可塑性检验问：给定相同的新问题与学习预算，当前学习器是否比适当对照更难学会？一个网络可以保留旧预测，却因步长极小而几乎学不会新目标；也可以快速学会新目标，却覆盖旧预测。

| 现象 | 必要的区分 | 可执行的诊断 |
| --- | --- | --- |
| 新任务学习变慢 | 任务更难，还是学习器变差 | 相同新目标、输入流、更新预算下比较已训练与新初始化网络 |
| 旧任务回报下降 | 遗忘，还是测试分布改变 | 固定旧任务测试集或固定策略评估；单独记录新分布 |
| 在线回报下降 | 数据缺失，还是有数据也学不会 | 固定行为/数据的预测探针，再与闭环控制对照 |
| 表示秩或活跃比例下降 | 诊断相关量，还是因果机制 | 单独干预对应机制，再测学习曲线而非只测秩 |

$$
h(x)=\max(0,w^\top x+b),\qquad
\frac{\partial h}{\partial w}=\mathbf1[w^\top x+b>0]x
$$

一个具体困难：若所有随后输入的预激活都为负，则这些样本上的局部梯度为零。仅把误差放大，不会让这个 ReLU 单元通过自身梯度重新活跃。输入分布或上游表示变化仍可能使它重新活跃。

Continual Backprop 在梯度学习之外保留少量特征生成与检验。其贡献型效用可用隐藏单元的活跃量与出连接幅度的乘积估计，再作指数平均。成熟期保护刚生成的单元，防止它在出连接尚未学到时立刻被淘汰。效用是当前数据下的代理量，不是该单元未来所有用途的真价值。

$$
u_i\leftarrow\eta u_i+(1-\eta)|h_i|\sum_j|w_{ji}^{\mathrm{out}}|
$$

这是原论文贡献型效用的基本形式。特征生成规则、每层替换率、成熟期和效用归一化应按具体实现声明。

若当前输出含 $w_i^{\rm out}h_i$，把这条出连接置零会先移除旧贡献。新单元的出连接为零，只保证“新生成的信号”暂时不额外改变输出，并不保证删除旧单元毫无影响。例如旧贡献为 0.02，输出就会先少 0.02；当前贡献很小也不代表罕见状态中的贡献很小。

原论文在持续监督学习与长时 PPO 任务中报告了可塑性退化及缓解结果。它不是所有网络必然退化、或某一种重置在所有 CRL 任务上收敛的定理。把神经元进一步看作具有局部目标的组件，是值得研究的架构设想；从局部效用到全局长期回报的信用分配仍需论证。

这一节给出的桥梁是：深度 RL 不只要在一个固定 benchmark 上找到好策略，还要让同一有限学习器持续产生可用的新表示。后续可塑性章节比较特征替换、正则化与优化尺度；知识构建章节再讨论这些表示被哪些预测、目标和技能使用。

<a id="lesson-derive"></a>

## 6 · 模型分支：用预测后果构造更新

深入阅读：模型提供的后果替代了真实转移。沿 Bellman 方程计算单步误差怎样传播。

前面的目标读取真实经验中的后继。若改为读取模型预测的后继或生成一段想象经验，模型估计便进入标签和动作选择。先固定策略，才能单独分析这一项误差。

这里及后两项分支共用的标准库数值核：模型Bellman误差界、递归敏感度与CQL型正则。它们检查算子，不提供模型控制的采样闭环。

```python
def recurrent_derivative(a, b, inputs):
    state, sensitivity = 0.0, 0.0
    for observation in inputs:
        state = a*state + b*observation
        sensitivity = a*sensitivity + observation
    return state, sensitivity


def conservative_penalty(q_values, data_action):
    maximum = max(q_values)
    return maximum + math.log(sum(math.exp(q-maximum) for q in q_values))-q_values[data_action]


def bootstrap_bias_bound(one_step_error, gamma):
    if one_step_error < 0 or not 0 <= gamma < 1:
        raise ValueError('nonnegative error and discount below one required')
    return one_step_error/(1-gamma)
```

固定策略，真实 Bellman 算子为 $TV=r+\gamma PV$，模型算子为 $\hat TV=\hat r+\gamma\hat PV$。若二者都是 sup 范数下的 $\gamma$ 压缩映射，固定点分别为 $V,\hat V$。假设模型在真实价值处的单步 Bellman 误差不超过 $\epsilon$。

$$
\begin{aligned}\|\hat V-V\|_\infty&=\|\hat T\hat V-TV\|_\infty\\&\le\|\hat T\hat V-\hat TV\|_\infty+\|\hat TV-TV\|_\infty\\&\le\gamma\|\hat V-V\|_\infty+\epsilon\end{aligned}
$$

第一步插入并减去同一项，第二步使用模型算子的压缩性与误差假设。

$$
\|\hat V-V\|_\infty\le\frac{\epsilon}{1-\gamma}
$$

这是固定策略、给定算子误差条件下的界，不是像素预测误差对所有控制算法的保证。

$$
\|\hat TV-TV\|_\infty\le\epsilon_r+\gamma\epsilon_P\|V\|_\infty,\quad \epsilon_P=\sup_s\sum_{s'}|\hat P(s'\mid s)-P(s'\mid s)|
$$

有限状态中，令奖励均值的最大误差为 $\epsilon_r$，先用三角不等式，再对转移行的绝对误差求和，即得到这个单步条件。若 $|r|\le R_{\max}$，则 $\|V\|_\infty\le R_{\max}/(1-\gamma)$。平均训练误差小并不提供这里所有状态上的统一误差界。

例如 $\epsilon=0.02$ 时，$\gamma=0.9$ 给出 0.2 的上界，$\gamma=0.99$ 则为二。降低单步重建误差不一定同比降低 Bellman 误差，因为与奖励和决策有关的细小差异可能被重建指标掩盖。模型是否足够好，要看它用于什么预测与规划。

Dyna 将模型生成的转移用于价值更新。Dreamer 在学得的潜在动力学中生成轨迹，用它们训练 actor–critic。TD-MPC2 则结合潜在模型、价值估计和在线动作序列规划。这些方法共享模型学习，但使用模型的计算位置不同：训练时想象与决策时规划不能混为一类 update。

<a id="lesson-memory"></a>

## 7 · 记忆分支：重新构造样本所需的状态

深入阅读：递归状态保留信息；参数敏感度决定怎样学习这种保留。这里逐项展开两者。

前面的五元组用当前观察即可计算价值。若动作需要更早线索，一条孤立转移便缺少状态构造的输入。此时还要规定历史前缀、递归初态和片段边界。下面先让同一组参数生成整段递归状态，以分开信息保留与参数信用。

$$
\begin{gathered}h_t=f_\theta(h_{t-1},o_t,a_{t-1})\\E_t=\frac{d h_t}{d\theta}=\frac{\partial f_\theta}{\partial h_{t-1}}E_{t-1}+\frac{\partial f_\theta}{\partial\theta}\end{gathered}
$$

$h$ 是当下的历史摘要，$E$ 是摘要对参数的敏感度。保留 $h$ 的数值并不表示保留了 $E$。

完整 BPTT 保存一段前向计算图并向过去回传；RTRL 在线递推敏感度，但一般稠密网络的成本很高。TBPTT 在片段边界 detach hidden state，相当于截断更早参数影响，不是清空 hidden state 数值。两种 reset 不能互换。

配套最小例子固定递归系数 $a$，学习输入系数 $\theta$：$h_t=ah_{t-1}+\theta x_t$，初始 $h_0=0$。于是 $E_t=aE_{t-1}+x_t$。取 $a=0.5,\theta=0.2$，输入 $(1,0,0)$，得到状态 $(0.2,0.1,0.05)$ 与敏感度 $(1,0.5,0.25)$。对输入系数做中心差分即可检验最后的 0.25；若改为学习递归系数，直接导数项应变为 $h_{t-1}$。

![完整梯度、截断梯度与清空记忆下，前向 hidden state 和参数敏感度的不同演化](https://yingwen.io/crl-figures/concept-depth-classic-memory-gradients.svg)

在第一步之后施加边界操作，其他设置相同。蓝色是前向记忆；橙色圆面积表示对输入系数 $\theta$ 的敏感度。detach 将已得到的 $h_1$ 当作常数，后两步的记忆数值仍为 0.1、0.05；清空记忆才使它们变成 0。参数在三步内保持不变，图中未做优化。[精确递推代码](https://yingwen.io/crl-code/figures/classic-visual-depth.mjs)。

从 replay 抽取一段中途开始的序列，还需要解释其初始状态从哪里来。保存的状态由采集时的参数形成；当前网络若从历史起点重新展开，可能得到另一个状态。预热（burn-in）先读一段前缀，再在后缀计算损失，用来减轻这项起点失配。前缀如果没有包含必要线索，从零开始预热仍可能丢失决定动作的信息。

损失从哪里开始与梯度在哪里截断，是两个独立决定。可以让后缀损失穿过整个预热段求导，也可以将预热末端状态保留为数值、截断其梯度。两者在这次前向计算中输出相同，却可能得到不同的参数更新。[递归回放算例](https://yingwen.io/zh/continual-rl/foundations/deep/partial-observability/#recurrent-replay-task)逐项比较保存状态、零状态和当前参数重建，并把它们接到实际动作与 TD 标签。

<a id="course-optimizer-memory"></a>

## 7.1 · 学习器也有记忆：参数之外哪些状态会影响下一步

深入阅读：参数之外的统计量也影响下一次更新。进一步检查优化器和归一化器的记忆。

| 状态 | 保存的信息 | 是否属于 agent 的输入记忆 | 改变后必须复查什么 |
| --- | --- | --- | --- |
| recurrent hidden state | 已发生的观察与行动的压缩 | 通常是 | 能否区分影响行动的历史 |
| 价值、策略与 encoder 参数 | 跨经验积累的预测和行为规律 | 是慢速学习状态 | 旧知识、泛化与新任务学习速度 |
| 资格迹 | 过去参数或特征对当前误差的信用 | 不等同于 hidden state | 历史梯度在当前表示下的意义 |
| 优化器一阶、二阶矩 | 历史梯度方向及平方尺度 | 不等同于世界记忆 | 新梯度会被放大、压低还是反向覆盖 |
| 归一化统计与目标网络 | 历史输入尺度与延迟的预测 | 各有独立时间常数 | 新旧输入、标签是否仍在相同单位 |

$$
m_{k+1}=\beta_1m_k+(1-\beta_1)g_k,\qquad
v_{k+1}=\beta_2v_k+(1-\beta_2)g_k^2,\qquad
\theta_{k+1}-\theta_k=-\alpha\frac{\hat m_{k+1}}{\sqrt{\hat v_{k+1}}+\varepsilon}
$$

这是逐坐标 Adam 的基本形式。$k$ 计优化步骤，不计环境交互；$g_k$ 是本次损失在 $\theta_k$ 处的梯度，不是 TD error 本身。初始 $m_0=v_0=0$，帽子表示按已完成的 $k+1$ 次更新作初始化偏差校正。

手算：旧梯度一直为 $+1$，所以 $m_k\approx1,v_k\approx1$。目标变化后梯度变为 $-1$。若 $\beta_1=0.9$，第一次新梯度到达后 $m_{k+1}\approx0.8$，更新仍沿负参数方向，和当前梯度下降方向相反。把旧一阶矩取为 1，连续 $n$ 次反向梯度后，$m=2(0.9)^n-1$，第七次优化才改变符号。

这不证明 Adam 一般有害。动量可以抑制噪声、加快固定目标优化；这里显示的是历史信息与适应速度的取舍。二阶矩的长记忆还可能使同一绝对梯度在不同时间得到不同有效步长。因此“学习率没有改变”不等于更新尺度没有改变。

若重新初始化部分神经元，需要一并决定相关优化器矩、资格迹、归一化统计和目标网络参数怎样处理。旧矩若附着到全新的参数功能上，会让重置后的第一步继承旧方向。重置全部统计也会丢掉有用尺度信息。两者都是算法选择，不应藏在实现细节中。

最小诊断是在相同的新数据流上复制学习器，分别只重置优化器、只重置输出头、只更换低效特征，并保留“不重置”对照。若只在闭环任务中看回报，数据覆盖差异可能掩盖真正的优化问题。后续流式与元学习章节分别研究如何计算一次更新，以及如何根据经验调整更新规则。

<a id="lesson-offline"></a>

## 8 · 离线分支：停止取得新的真实经验

深入阅读：经验集合固定后，智能体不能用新行动取得反证。这里推导保守价值正则的作用。

前面的闭环仍能执行新动作，收到反证后再修正预测。固定数据集撤掉这条反馈路径：学习器可以多做优化，却不能主动获得数据没有包含的行动后果。

Off-policy 指学习目标策略与数据行为策略可以不同；offline 还要求数据固定，不再通过新行动纠正估计。Q-learning 或 SAC 在未见动作上高估时，在线环境可能提供反证，离线数据却可能永远没有该动作的结果。因而旧经验复用能力不等于离线控制的可靠性。

$$
\begin{gathered}L_{\rm conservative}(Q)=\log\sum_a e^{Q(s,a)}-\mathbb E_{a\sim\mathcal D(\cdot\mid s)}Q(s,a)\\\frac{\partial L_{\rm conservative}}{\partial Q(s,a)}=\operatorname{softmax}(Q(s,\cdot))_a-p_{\mathcal D}(a\mid s)\end{gathered}
$$

这是离散动作 CQL 型正则的核心形式。完整 CQL 还包含 Bellman loss、权重选择和相应理论条件；连续动作通常需要采样近似。

若数据只含动作零，而 $Q(s,\cdot)=(0,10)$，正则约为 $10.000045$。它对动作一的导数接近一，对动作零的导数接近负一。因此梯度下降压低未支持的高值，并相对提高数据动作。正则本身只约束相对偏好，不能替代奖励监督。

保守性可能减少外推，也可能压制真正有用但数据很少的动作。对任意未覆盖结果都保持正确是不可能由损失形式单独保证的；数据覆盖和任务假设仍然重要。CRL 中的历史 replay 常兼有旧分布、策略覆盖不足和动力学变化，三种误差要分别诊断。

<a id="experiment-deep-cql"></a>

### 实验：固定数据中的保守项：限制外推，不创造缺失证据

训练不再采集新经验时，降低未见动作的估值能否改变策略？

**环境与可用信息。** 位置 0–4 的 DeadlineChain，动作是左／右；观测是位置 one-hot 和剩余时间比例。到位置 4 得 1 并终止，其余每步 −0.02；12 步期限也是问题的真实终止。每回合从 0 开始。训练数据预先用独立 seed 2026 采集 512 条转移，行为以 0.65 概率向右。训练期不与环境交互。

**设置。** 本图实际运行 1200 个预算单位，训练种子为 0–4。这里的预算单位是训练 batch，不是环境步。每批从同一固定数据重采样 32 条；32 隐单元，Adam 0.003，折扣 0.99，Polyak 0.02。每批 CQL 与普通离线 Q-learning 均有一次优化器调用。

**检验的机制。** CQL 在半平方 TD 损失之外，加权重为 1 的 logsumexp 动作值减数据动作值。离散动作可精确求和；没有连续动作采样和自适应 Lagrange 权重。

**测量。** 冻结贪心策略在独立环境中的回报用于评价，不把评价经验加入固定数据。五个 seed 改变网络初始化和数据重采样，而不是产生五套离线数据。

```bash
python3 implementations/deep/cql.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/deep-cql/curves.svg)

横轴：training_batches。纵轴：冻结策略的外部回报。每种方法 1200 training_batches；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 600 个 batch 两者均约 0.94；1200 个 batch 分别为 0.94 与 0.932。当前数据已经足以学到近最优路线，不能用此例证明保守正则对严重覆盖不足普遍有效。

**结论边界。** 它没有扫描数据质量、行为分布和缺失动作，无法证明 D4RL 性能或无覆盖区域的可靠估计。保守偏置还可能压低需要但少见的动作。

**继续实验。** 在训练前固定不同的行为策略与数据量；冻结所有数据再比较普通 Q、CQL 和 IQL。把数据 seed 与训练 seed 分开，不能看过测试回报后挑最好数据集。

[源码](https://yingwen.io/crl-code/implementations/deep/cql.py) · [逐种子记录](https://yingwen.io/crl-code/results/deep-cql/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/deep-cql/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/deep-cql/curves.json)

<a id="lesson-check"></a>

## 9 · 练习与答案

- 问：旧replay转移被再用时，什么保持不变、什么重新计算？答：记录中的观察、实际动作、奖励、后继和终止标志不变；当前预测、后继选择及网络尾值按本次版本重算。
- 问：已用no_grad生成标签，为何还要目标副本？答：no_grad决定这次不沿标签求导；副本延迟在线参数变化进入后续标签。两者控制不同时间上的依赖。
- 问：PPO批次在episode中间结束，是否清零尾值？答：不清零；末行残差保留实际后继价值，GAE只停止跨入下一批。
- 问：SAC更新actor时冻结critic，梯度如何经过critic？答：critic参数不写入，但保留输出对新动作的导数，继续沿动作生成器回到actor参数。
- 练习：不运行训练，沿train_dqn复算1600环境步的写入与同步。答：从零索引t=32到1598的偶数索引共784次优化，目标网络按第100、200……1600条转移共同步16次。

- 问：重建误差下降是否保证模型控制改善？答：不保证。Bellman 相关误差、规划覆盖和奖励敏感维度可能仍有偏差。
- 问：detach hidden state 是否清除 agent 的记忆？答：不清除数值记忆，但截断其对早先参数计算的梯度依赖。
- 问：CQL 正则能从零数据推断未见动作奖励吗？答：不能。它限制乐观外推，不产生缺失的因果信息。
- 实验：把 DeadlineChain 的剩余时间从观察删除，在不同剩余步数的同一位置比较价值目标；答案是出现状态混叠，不能简单归因于 DQN 优化失败。

## 从本章进入实践

[策略梯度与控制](https://yingwen.io/zh/continual-rl/code/#practice-policy-control)：优化器确实降低了损失，为什么行动仍可能变差？



<a id="chapter-code"></a>

## 下载与运行

本文件用标准库核验数值；deep_textbook_train.py 提供 DQN/PPO 小任务训练与连续控制更新核，连续控制完整教学训练见 implementations/deep/ 的独立实现。

[下载 deep_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/deep_textbook_lab.py)

```sh
python3 deep_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Mnih et al. · Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)：主文标签与Methods训练算法：replay、冻结目标、误差导数裁剪的不同作用。教学循环采用自己的小任务、Adam与同步计时。

- [van Hasselt et al. · Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)：DQN、Double Q-learning与Double DQN段落：后继动作选择与评价分开，原实验与非线性保证分开。

- [Schulman et al. · Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)：§2–3、§5：有限批代理、比率clip与采样/优化交替；本章对应实际全批教学核。

- [Schulman et al. · High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)：§3：多步优势与残差加权；两mask及批末边界由本任务实现单独规定。

- [Haarnoja et al. · Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)：§4–5：双critic、当前actor的路径导数、温度对偶与Polyak目标；soft迭代理论与神经近似分开。

- [Pardo et al. · Time Limits in Reinforcement Learning](https://proceedings.mlr.press/v80/pardo18a.html)：§2–3：任务有限期限需要剩余时间；外部截断保留实际后继的bootstrap。

- [Hafner et al. · Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)：DreamerV3 的世界模型与想象 actor–critic。

- [Hafner · dreamerv3](https://github.com/danijar/dreamerv3)：作者维护的重实现；仓库明确说明基于 DreamerV2，不等同于原始内部实验代码。

- [Hansen et al. · TD-MPC2](https://arxiv.org/abs/2310.16828)：潜在模型与连续控制规划。

- [Hansen · tdmpc2](https://github.com/nicklashansen/tdmpc2)：作者代码与训练配置；与 Dreamer 的模型使用位置对照。

- [Kumar et al. · Conservative Q-Learning](https://arxiv.org/abs/2006.04779)：offline 分布偏移、保守价值目标及其条件。

- [Kumar · CQL](https://github.com/aviralkumar2907/CQL)：作者实现；包含离散与连续控制实验入口。

- [Agarwal et al. · Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)：有限种子下的可靠评价、区间与汇总。

- [rliable](https://github.com/google-research/rliable)：论文配套评价代码，不是训练算法。

- [Williams & Zipser · A Learning Algorithm for Continually Running Fully Recurrent Neural Networks](https://doi.org/10.1162/neco.1989.1.2.270)：RTRL 原始论文；敏感度在线递推的来源。

- [Sutton & Barto · Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)：§9–11：函数逼近与离策略；§12：资格迹；§13.1 的短走廊与 §13.2–13.5 的策略梯度。对照各结论采用的策略类、采样分布与函数表示。

- [Dohare et al. · Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7)：区分旧知识遗忘与新知识学习能力；Methods 给出 contribution utility、成熟期及单元重置。文中的长时实验不是任意任务上的稳定性证明。

- [Dohare et al. · Loss of plasticity 作者代码](https://github.com/shibhansh/loss-of-plasticity)：对照原论文的持续监督学习、PPO 与 generate-and-test 实现；复制实验前核对网络、优化器和重置配置。

- [Elsayed et al. · Streaming Deep Reinforcement Learning Finally Works](https://arxiv.org/abs/2410.14606)：Stream-X 将神经预测与控制放回逐样本更新；应逐项检查归一化、初始化、资格迹和步长控制，不把移除 replay 当作完整算法。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-practice#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-practice#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-deep-practice)
<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 估计哪种策略的价值，又按什么状态分布加权？

行为策略决定怎样获得经验；目标策略决定要预测哪种行为。重要性比可以校正给定状态下的动作分布，但不会自动把状态出现的频率改成目标策略的频率。

函数逼近与深度方法：Replay 还引入缓冲区的时间组成与抽样规则。神经更新受到数据分布、共享梯度和移动目标共同影响。重复旧数据与逐条使用新数据有不同的资源和适应代价。

持续学习中的研究问题：单一行为流怎样支持许多预测和技能？在固定内存下，怎样权衡覆盖、样本年龄、更新方差与适应速度，而不把离策略修正当作完整稳定性保证？

[离策略稳定性](https://yingwen.io/zh/continual-rl/foundations/approximation/off-policy/) → [数据与训练接口](https://yingwen.io/zh/continual-rl/foundations/deep/practice/) → [大规模系统与策略滞后](https://yingwen.io/zh/continual-rl/foundations/deep/systems/) → [离线数据的覆盖](https://yingwen.io/zh/continual-rl/foundations/deep/offline/) → [流式更新](https://yingwen.io/zh/continual-rl/algorithms/streaming/)


### 可进一步检验的问题

- [02 · 有限的内部状态应当保留哪些历史信息？](https://yingwen.io/zh/continual-rl/research/#research-agent-state)：将 recurrent 活动状态与跨时间参数梯度分开，才能区分线索没有被保存和保存机制没有学会。
- [13 · 为什么训练越久，学习新东西反而越慢？](https://yingwen.io/zh/continual-rl/research/#research-plasticity)：训练与评价协议中的参数、优化器和数据权限必须明确，才能比较老网络、新初始化及局部重置后是否还能学会新目标。
- [16 · 什么实验能区分“仍在更新”与“仍在有效学习”？](https://yingwen.io/zh/continual-rl/research/#research-measurement)：分开训练过程、冻结评价、随机重复和计算预算，才能判断一个机制改善了哪一段学习过程。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)
- [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)
- [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)
- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)
- [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)
- [可塑性与特征更新](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

对应原始材料：Sutton & Barto §6.5、§9.2–9.3、§11.3、§16.5；DQN Methods；Double DQN目标；PPO §2–5；SAC Algorithms and Applications §4–6；Time Limits §2–3；CQL §3–4。本文为原创讲解，原书、论文与上游代码保留各自许可。
