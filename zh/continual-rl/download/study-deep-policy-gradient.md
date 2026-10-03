# 策略梯度：从轨迹概率到 GAE 与 actor–critic

不对环境求导，怎样从采样动作计算策略梯度？价值网络和优势各自做什么？

## 本章内容

- 推导 score-function 梯度与 baseline 消去。
- 区分真实目标、优势估计和实现 surrogate。
- 给 GAE 设置独立的 bootstrap 与跨序列 mask。
- 正确使用 log_prob、detach 和 actor/critic loss。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### Bellman 递推

当前价值等于即时奖励加折扣后的后继价值；预测目标与最优控制目标不同。

### 函数与梯度

神经网络把参数和输入映射为输出。链式法则必须指明哪些量参与求导，哪些量作为固定目标。

### 经验分布

on-policy 数据由当前策略产生；off-policy 数据可来自旧策略，但能否正确复用取决于算法目标和数据覆盖。

<a id="lesson-setting"></a>

## 1 · 随机策略和有限轨迹

策略 $\pi_\theta(a\mid s)$ 输出动作分布。离散动作通常使用 softmax logits，连续动作可使用 Gaussian。这里先考虑终止时间为 $T$ 的 episode，目标 $J(\theta)=\mathbb E_\theta[\sum_{t=0}^{T-1}\gamma^tR_{t+1}]$。环境转移对参数未知，但轨迹中动作的概率可以计算。

Actor 改变行为分布；critic 提供价值与优势估计。Critic 不负责决定策略梯度是否存在，而是降低估计方差或用 bootstrap 换取更及时的反馈。A2C 通常同步收集并更新，A3C 使用异步工作者；同步与异步是执行协议，不是另一条策略梯度定理。

<a id="lesson-derive"></a>

## 2 · 对轨迹概率求导，不对环境求导

$$
\begin{gathered}p_\theta(\tau)=p(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t\mid s_t)P(s_{t+1},r_{t+1}\mid s_t,a_t)\\\nabla_\theta\log p_\theta(\tau)=\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\end{gathered}
$$

环境概率不含策略参数，所以对数导数只保留动作概率项。需要可微策略及使交换期望与求导成立的常规条件。

$$
\nabla J=\mathbb E\!\left[\sum_t\gamma^t\nabla\log\pi_\theta(A_t\mid S_t)G_t\right],\quad G_t=\sum_{k=t}^{T-1}\gamma^{k-t}R_{k+1}
$$

从完整 return 的 score estimator 出发，动作无法改变它之前已经发生的奖励，条件期望使这些过去奖励项为零，于是得到 reward-to-go。

$$
\sum_a\pi_\theta(a\mid s)\nabla\log\pi_\theta(a\mid s)b(s)=b(s)\nabla\sum_a\pi_\theta(a\mid s)=0
$$

任何不依赖当前动作的 baseline 都可以在真实期望中消去。令 baseline 近似状态价值，便得到优势形式。baseline 自身参与训练，但 actor 的这次 score-function 求导将优势视为固定权重。

因此可以最小化 $L_\pi=-B^{-1}\sum_t\gamma^t\log\pi_\theta(a_t\mid s_t)\operatorname{sg}(\hat A_t)$。这里的 $\gamma^t$ 对应从初始状态出发的折扣目标。很多实现均匀采样时间步并省去这项，使用的是常见近似，或另一个状态加权目标，不能在推导中悄悄删掉。配套 PPO 小任务取 $\gamma=1$。

<a id="lesson-advantage"></a>

## 3 · TD 残差、多步优势与 GAE

$$
\begin{gathered}\delta_t=R_{t+1}+\gamma b_tV_\phi(S_{t+1})-V_\phi(S_t)\\\hat A_t^{\mathrm{GAE}}=\delta_t+\gamma\lambda c_t\hat A_{t+1}^{\mathrm{GAE}}\end{gathered}
$$

$b_t$ 是能否 bootstrap；$c_t$ 是能否把下一行样本的优势继续接上。真正终止使二者为零。环境被人工重置的 timeout 通常 $b_t=1$、$c_t=0$。普通 batch 尾部用最后观测的价值，递推 carry 初始化为零。

若序列内部没有边界，展开得到 $\hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}$。$\lambda=0$ 只保留一步误差；$\lambda=1$ 在真终止 episode 中望远镜消去为 $G_t-V_\phi(S_t)$。不准确的 critic 在较小 $\lambda$ 下通常引入更多 bootstrap 偏差，较长估计又通常承受更多采样噪声。

先用旧 critic 计算全部优势和回归目标 $\hat R_t=\hat A_t+V_{\phi_{\rm old}}(s_t)$，再更新网络。优势标准化只能用于 actor 的权重；若把标准化后的优势加回 value 当作 critic target，就改变了价值的奖励单位。

<a id="lesson-algorithm"></a>

## 4 · VPG / A2C 的更新次序与 autograd

**算法：这里写 γ=1 的有限时域形式；若采用初始状态折扣目标，actor 项还需对应时间权重。**

1. 固定本轮策略与 critic，收集新 trajectories。
1. 记录 observation、action、reward、terminal、value 与 next value。
1. 后向计算原始 GAE 与固定 critic target。
1. 构造 $L_\pi=-\operatorname{mean}(\log\pi_\theta(a\mid s)\operatorname{sg}(\hat A))$。
1. 构造 $L_V=\operatorname{mean}((V_\phi(s)-\operatorname{sg}(\hat R))^2)$。
1. 清空梯度，反传相应 loss，更新对应参数。
1. 丢弃本轮 on-policy 数据，开始下一轮采样。

离散动作必须对实际采样动作计算 `log_prob`，不能直接对最大 logit 求导。`torch.distributions.Categorical(logits=...)` 会进行稳定归一化。连续多维独立 Gaussian 的 `log_prob` 通常先返回各坐标，必须对动作维度求和，而不是把动作维度误当 batch。

如果共享 encoder，actor 和 critic 两个损失都会训练共享参数。必须明确损失系数与优化次序。把 critic 当 baseline 消去，并不意味着共享 critic 参数的梯度可以偷偷流过 actor 优势；这会增加与策略梯度不同的项。

<a id="lesson-example"></a>

## 5 · 两步 GAE 与 softmax 梯度

取奖励 $(1,2)$，旧价值 $(0.5,1)$，第二步真终止，$\gamma=0.9,\lambda=0.8$。两步残差分别为 $1+0.9\times1-0.5=1.4$ 与 $2-1=1$。故优势为 $(2.12,1)$，critic 目标为 $(2.62,2)$。若第一步其实是独立序列的截断，不能把第二条的残差接到它后面。

$$
\frac{\partial\log\pi(a\mid s)}{\partial z_j}=\mathbf1[j=a]-\pi(j\mid s)
$$

softmax 的对数导数包含所有动作，不只是被选动作。若概率为 $(0.25,0.75)$，动作零的优势为 $2$，梯度上升方向为 $(1.5,-1.5)$，归一化会共同改变两个概率。

固定旧 value 和 next value，分别处理两类 mask，并输出原始优势与 critic target。

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

<a id="lesson-code"></a>

## 6 · 从数值核验到实际 on-policy 训练

VPG 的一次 actor–critic 更新。优势和回归目标停止梯度；actor 更新后必须重新采样。

```python
def vpg_update(actor, critic, actor_optimizer, critic_optimizer, x, actions,
               advantages, returns):
    # One update with fresh on-policy data; gamma=1 episodic convention.
    distribution = Categorical(logits=actor(x))
    loss_actor = -(distribution.log_prob(actions)*advantages.detach()).mean()
    actor_optimizer.zero_grad()
    loss_actor.backward()
    actor_optimizer.step()
    prediction = critic(x).squeeze(-1)
    assert prediction.shape == returns.shape
    loss_critic = ((prediction-returns.detach())**2).mean()
    critic_optimizer.zero_grad()
    loss_critic.backward()
    critic_optimizer.step()
    return float(loss_actor.detach()), float(loss_critic.detach())
```

PPO 使用同一套 actor–critic 采样和 GAE；区别在下一章的策略 surrogate。

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

执行 python3 deep_textbook_lab.py demo 查看 GAE 与 score-gradient 的手算值；执行 python3 deep_textbook_train.py ppo --epochs 20 --seed 0 查看真实 on-policy 采样。代码在 rollout 内固定参数，并保存旧 log-probability。环境已经真终止后才 reset；若 batch 恰在 episode 中间结束，环境活动继续保留。

VPG 更新核用固定优势乘当前 log_prob，进行一次 actor 更新，再要求重新采样；PPO 则使用旧策略概率比并对同批数据做受限复用。上面的完整采样循环运行 PPO，不是独立 VPG 训练命令。官方 VPG 文件提供独立、完整的工程入口；配套 VPG 核用于逐项检查梯度与 detach。

<a id="lesson-branches"></a>

## 7 · 估计误差与持续学习接口

- 优势偏差：critic 有误差、λ 小、轨迹截断都会改变估计。不能把所有 GAE 都称作无偏真实优势。
- 概率错误：离散采样后重新计算另一动作的 log_prob，或连续动作裁剪后仍使用裁剪前 Gaussian 密度，都改变了 estimator。
- 策略陈旧：旧轨迹反复用于裸 log-probability loss，不再是当前 on-policy 梯度。
- 任务变化：critic、状态和策略可以不同速度适应，优势的符号可能暂时错误。

在 CRL 中，策略熵变小可能让有用数据不再出现；critic 误差又会影响行动更新。诊断时应分别测覆盖、优势误差和网络学习能力，而不是看到回报下降就统一归因于遗忘。

<a id="lesson-check"></a>

## 8 · 练习与答案

- 问：可以把即时奖励换成任意 baseline 吗？答：baseline 必须在条件期望中不依赖当前采样动作，且 actor 求导时固定它；否则消去推导不成立。
- 问：critic 的 loss 下降是否保证策略变好？答：不保证。它可能只改善高频无关状态，或在关键动作上的优势排序仍错误。
- 问：原始优势是 (2.12,1)，标准化后还能作为价值 target 吗？答：不能。actor 的尺度处理不应改变 critic 的奖励单位。
- 实验：固定一批完整轨迹，比较 λ=0、0.8、1 的目标，并单独制造一个 timeout。确认目标差异由哪一条 mask 和哪个尾值产生。

<a id="chapter-code"></a>

## 下载与运行

标准库数值核验；完整小任务训练另需 deep_textbook_train.py 与 PyTorch。

[下载 deep_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/deep_textbook_lab.py)

```sh
python3 deep_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Spinning Up · VPG](https://spinningup.openai.com/en/latest/algorithms/vpg.html)：用作算法与实现接口的对照；本章解释和配套小实验独立编写。

- [Sutton et al. · Policy Gradient Methods with Function Approximation](https://papers.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)：策略梯度定理与函数逼近的原始研究。

- [Schulman et al. · Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)：GAE 的偏差、方差与 γ-just 估计条件。

- [Spinning Up · vpg.py](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/vpg.py)：官方教学工程；追踪 buffer 的 finish_path 与 actor/critic loss。

<a id="study-connections"></a>

## 与教材主线的衔接

- [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)
- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)
- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)

对应原始材料：Sutton & Barto §13.1–13.5；§12：资格迹与多步估计。本文为原创讲解，原书、论文与上游代码保留各自许可。
