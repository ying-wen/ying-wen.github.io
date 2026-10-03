# 深度价值学习：DQN、Double DQN 与目标的时间顺序

把表格 Q-learning 换成网络后，损失、数据与目标为什么都需要重新组织？

## 本章内容

- 写出 DQN 与 Double DQN 的不同目标。
- 区分环境终止、采样截断和目标网络更新。
- 通过 autograd 检查停止梯度和张量形状。
- 运行含真实交互、replay、优化与评估的 CPU 小实验。

<a id="problem-definition"></a>

## 本章的问题定义

在离散动作控制中，用神经网络近似最优动作价值；环境模型未知，数据来自不断变化的行为。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 在线网络 $Q_\theta$、目标网络 $Q_{\bar\theta}$、经验分布 $\mathcal D$；$m=0$ 表示真实终止，否则 $m=1$。

### 需要求解的对象

学习能产生有效贪心动作的价值近似；最终目标仍是外部期望回报。

### 信息与数据权限

可重放已保存转移，但不能由 replay 自动获得未覆盖动作的真实结果。

$$
y=r+\gamma m\max_{a^{\prime}}Q_{\bar\theta}(s^{\prime},a^{\prime}),\qquad L(\theta)=\mathbb E_{\mathcal D}[(Q_\theta(s,a)-\operatorname{sg}(y))^2]
$$

这是在固定目标版本和数据分布下的回归损失，$\operatorname{sg}$ 表示停止梯度。它是逼近 Bellman 最优性更新的机制，不是等同于最大化 J 的恒等式。

### 成立条件与解的含义

- 有限可枚举动作；明确终止、目标网络同步和 replay 采样规则。
- 网络容量、覆盖和优化误差均可能限制解；一般非线性 DQN 没有表格式全局收敛保证。

判断准则：先在已知小问题检查 target 和动作选择；再报告实际收益、价值尺度及覆盖，不能只看训练 loss。

### 适用边界

- 把固定 target 下的回归下降解释为整个非平稳控制过程单调改善。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 限制表示或采用近似 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：本章用神经动作价值与指定贪心提取规则近似折扣控制；表示、覆盖和求解误差限制实际行为，回归损失不是完整学习器收益。

- 改变信息或数据协议 · [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)：本章允许 replay 和目标网络滞后；相对严格流式协议，它增加数据复用、持久内存及更新调度。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

自举目标、采样分布与网络参数同时变化，且最大化会选择正向估计噪声。

### 本章的核心思路

将采样、回归和目标更新放在不同时间尺度，并分离动作选择与评价。

1. [固定一个回归目标再求梯度](#lesson-derive)：停止对 target 反传，明确每次优化实际使用的目标版本。

2. [拆开选择和评价](#lesson-double)：Double DQN 由在线网络选动作，再由目标网络评值。

3. [规定三个时钟的执行顺序](#lesson-algorithm)：环境步、优化步和目标同步步不是同一计数器。

结论与条件：目标网络与 replay 是稳定化机制；它们不消除函数逼近、自举和离策略耦合的所有风险。

### 相关方法改变了什么

- DQN / Double DQN：改变后继目标中的动作选择器，不改变外部奖励目标。

- Replay / 严格在线更新：改变数据复用和相关性，也改变内存与每步成本。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### Bellman 递推

当前价值等于即时奖励加折扣后的后继价值；预测目标与最优控制目标不同。

### 函数与梯度

神经网络把参数和输入映射为输出。链式法则必须指明哪些量参与求导，哪些量作为固定目标。

### 经验分布

on-policy 数据由当前策略产生；off-policy 数据可来自旧策略，但能否正确复用取决于算法目标和数据覆盖。

<a id="lesson-setting"></a>

## 1 · 学习对象、数据和三个时间尺度

在离散动作 MDP 中，网络 $Q_\theta(s,a)$ 近似动作价值。它不是一个动作分类器：输出可以为负，也不需要和为一。行为策略通常采用 $\epsilon$-greedy，而学习目标使用贪心后继动作，因此行为策略与目标策略不同。

三种时间需要分开。环境时钟产生转移；优化时钟从 replay 抽样更新网络；目标网络时钟更新用于 bootstrap 的参数。每个环境步做多少次优化，和每隔多少次更新同步目标，都会改变算法。只报告学习率不足以复现实验。

| 符号 | 含义 |
| --- | --- |
| θ / θ⁻ | 当前 Q 网络 / 目标网络参数 |
| D | 保存 transition 的有限 replay buffer |
| d / b | 真正终止标志 / bootstrap mask |
| B | 一个 minibatch 的样本数，不是时间跨度 |

<a id="lesson-derive"></a>

## 2 · 从 Bellman 目标到停止梯度的回归

$$
Q^*(s,a)=\mathbb E[R_{t+1}+\gamma\max_{a'}Q^*(S_{t+1},a')\mid s,a]
$$

已知最优价值必须满足这个固定点关系。但采样更新只能使用一个后继和一个当前估计。

$$
\begin{gathered}y_i=r_i+\gamma(1-d_i)\max_{a'}Q_{\theta^-}(s_i',a')\\L(\theta)=\frac1B\sum_i\ell\!\left(Q_\theta(s_i,a_i)-\operatorname{sg}(y_i)\right)\end{gathered}
$$

sg 表示该次优化中停止梯度。目标会在未来改变，但在当前 backward 中被视为固定标签。

若 $\ell(e)=\tfrac12e^2$，梯度为 $B^{-1}\sum_i(Q_\theta-y_i)\nabla_\theta Q_\theta$。这叫半梯度更新。若同时对后继目标求导，就变成另一种残差优化；在随机转移下，单个样本的平方 TD 误差也不等于真实 Bellman 误差平方，存在双采样问题。

$$
\ell_{\mathrm{Huber}}(e)=\begin{cases}\tfrac12e^2,&|e|\le1,\\|e|-\tfrac12,&|e|>1.\end{cases}
$$

教学实现使用 Huber 损失：小误差保留二次梯度，大误差的导数幅度限制为一。它减少单个异常目标的影响，但不解决错误目标或数据覆盖不足。

网络表示共享使一个动作的更新也能改变其他状态动作的价值。表格中的局部更新性质消失了。非线性函数逼近、bootstrap 和离策略数据的组合不能直接继承表格 Q-learning 的收敛结论。

<a id="lesson-double"></a>

## 3 · Double DQN：把选择和评价分开

如果两个动作的真实价值相同，而估计误差有噪声，max 倾向挑中被高估的那个。DQN 在目标网络里同时选动作和估值。Double DQN 用当前网络选择动作，再让目标网络评价这个被选中的动作。

$$
a_i^*=\arg\max_a Q_\theta(s_i',a),\qquad y_i^{\mathrm{Double}}=r_i+\gamma(1-d_i)Q_{\theta^-}(s_i',a_i^*)
$$

不是对两个 Q 值取 min，也不是交替更新两个完全独立的网络。当前网与目标网有关联，因此它并不使过估计严格归零，也可能产生低估。

目标网络降低标签在连续梯度步之间的变化速度。Replay 将连续经验重新抽样，提高数据复用并减弱相邻样本的相关性。二者没有把数据变成来自真实分布的独立样本，也没有消除策略导致的覆盖偏差。

<a id="lesson-algorithm"></a>

## 4 · 一次交互与一次优化的精确次序

**算法：算法伪代码**

1. 初始化 $Q_\theta$，复制 $\theta^-=\theta$，清空 replay。
1. 每个环境步：
  1. 用当前 $Q_\theta$ 的 $\epsilon$-greedy 行为选动作。
  1. 获取真实的 $(s,a,r,s',d)$，再决定是否重置环境。
  1. 将原来的 $s'$ 写入 replay，不能写成 reset 后的状态。
  1. 样本足够时，抽取 batch，先计算所有停止梯度的目标。
  1. 用 gather 取出当前动作预测，反传 loss，只更新 $\theta$。
  1. 达到目标同步时钟后，复制或平滑更新 $\theta^-$。
  1. 用独立环境评估，不把评估转移放进训练 replay。

真正终止时 $b=1-d=0$。日志窗口或人工采样上限不是终止，通常仍需从截断前的最后观测 bootstrap。有限时域任务的最后一步可以是真终止，但若剩余时间会影响最优决策，它应作为状态输入。本页小环境有可观察的十二步期限，期限耗尽属于任务终止。

<a id="lesson-example"></a>

## 5 · 一个 batch 元素的手算

设奖励为 $1$，折扣为 $0.9$。后继当前网络输出为 $(5,4)$，目标网络输出为 $(2,6)$。DQN 目标是 $1+0.9\times6=6.4$。Double DQN 先选择动作零，所以目标是 $1+0.9\times2=2.8$。若该转移真正终止，两者目标都退化为 $1$。

若当前动作预测为 $2$，Double 目标下误差为 $-0.8$，Huber 梯度系数也是 $-0.8$，下降步骤会提高该预测。DQN 目标下误差为 $-4.4$，Huber 梯度系数截到 $-1$。这是损失梯度的限制，不是把目标值限制为某个范围。

独立计算 DQN/Double 目标与两类 mask。

```python
def dqn_target(reward, gamma, terminated, next_online, next_target,
               double=True):
    if not next_online or len(next_online) != len(next_target):
        raise ValueError('matching nonempty action vectors required')
    if terminated:
        return reward
    if double:
        selected = max(range(len(next_online)), key=next_online.__getitem__)
        continuation = next_target[selected]
    else:
        continuation = max(next_target)
    return reward + gamma * continuation


def masks(terminated, boundary):
    # boundary includes terminal, timeout, or the end of this rollout.
    return float(not terminated), float(not (terminated or boundary))
```

<a id="lesson-code"></a>

## 6 · 实际 PyTorch 更新与完整小任务训练

batch 维度始终为 B；目标处 no_grad；gather 后去掉单维。

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

完整交互、replay 抽样、优化、目标同步及独立 greedy 评估。

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

将 deep_textbook_lab.py 与 deep_textbook_train.py 放在同一目录。执行 python3 deep_textbook_train.py dqn --steps 2000 --seed 0；加 --ordinary-dqn 切换普通 DQN。内置 DeadlineChain 使用小 MLP，不下载数据，不使用 Gym 或 GPU。DQN 用折扣 0.99，显示的 greedy return 是便于读数的未折扣任务收益；它不是跨算法公平评测报告。

<a id="lesson-branches"></a>

## 7 · 失败条件与进入 CRL 的接口

- 形状错误：预测为 [B,1] 而目标为 [B] 会广播成 [B,B]；loss 仍能下降，却优化了错误配对。
- 目标泄漏：缺少 detach 会改变求导对象；把 reset 后观测存入旧 transition 会虚构动力学。
- 数据陈旧：大 replay 在平稳问题中有益，在奖励变化后可能反复训练失效标签。需明确奖励能否重新计算。
- 表示干扰：目标网络不能防止已有特征失去可塑性；增加更新/样本比也可能放大过拟合。

研究持续深度价值学习时，可先固定网络和数据，分别改变 replay 的年龄分布、目标更新速度与特征回收。若三个模块同时改变，回报差异无法说明是哪一种机制有效。

<a id="lesson-check"></a>

## 8 · 练习与答案

- 问：训练窗口在第 100 步结束，但任务没有结束，bootstrap mask 是多少？答：一。需用真正最后观测的价值；不能把采样边界误写为终止。
- 问：目标网络当前输出为 (2,6)，Double 一定取 6 吗？答：不一定。当前网络负责选择，例子中选择了目标值为 2 的动作。
- 问：replay 与 target 哪个解决长期遗忘？答：二者都不提供一般保证。Replay 保存部分数据，target 降低短期标签变化；保留能力还依赖覆盖、容量和更新规则。
- 实验：把目标同步间隔改为 1、100、1000，在相同交互预算记录 TD loss、目标变化和回报。解释过快与过慢同步各自可能带来的问题，而不是只选最好种子。



<a id="chapter-code"></a>

## 下载与运行

CPU DeadlineChain 完整训练；需要同目录 deep_textbook_lab.py 和 PyTorch。

[下载 deep_textbook_train.py](https://yingwen.io/zh/continual-rl/download/deep_textbook_train.py)

```sh
python3 deep_textbook_train.py dqn --steps 2000 --seed 0
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Mnih et al. · Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)：DQN 原文，检查 replay、目标网络和 Atari 协议。

- [DeepMind · 原始 DQN](https://github.com/google-deepmind/dqn)：原作者 Lua/Torch 工程；不是现代 PyTorch 代码。

- [van Hasselt et al. · Double DQN](https://arxiv.org/abs/1509.06461)：选择与评价分离的原文与过估计分析。

<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [深度价值学习](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)
- [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)
- [可塑性与特征更新](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)

对应原始材料：Sutton & Barto §6.5；§11.3：半梯度方法与离策略风险。本文为原创讲解，原书、论文与上游代码保留各自许可。
