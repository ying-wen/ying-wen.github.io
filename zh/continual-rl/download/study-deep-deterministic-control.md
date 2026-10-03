# 连续动作的价值优化：DDPG 与 TD3

不能枚举连续动作时，如何用 critic 的梯度改进 actor？

## 本章内容

- 区分行为噪声、目标动作噪声与目标网络。
- 推导确定性 actor 的链式梯度。
- 写出 TD3 双 critic、延迟更新与平滑目标的完整次序。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### Bellman 递推

当前价值等于即时奖励加折扣后的后继价值；预测目标与最优控制目标不同。

### 函数与梯度

神经网络把参数和输入映射为输出。链式法则必须指明哪些量参与求导，哪些量作为固定目标。

### 经验分布

on-policy 数据由当前策略产生；off-policy 数据可来自旧策略，但能否正确复用取决于算法目标和数据覆盖。

<a id="lesson-setting"></a>

## 1 · 用 actor 近似连续动作的最大化

离散 DQN 可以对所有动作求最大值。连续动作空间中，$\max_a Q(s,a)$ 通常需要一次数值优化。DDPG 引入确定性 actor $a=\mu_\theta(s)$，使一次网络前向直接给出动作。critic 学习当前策略的价值，actor 沿 critic 对动作的梯度移动。两者不是独立的监督任务：critic 的误差会影响 actor，actor 改变后又会影响后续目标。

本章采用有界连续动作、折扣奖励与经验 replay。行为动作为 $a=\operatorname{clip}(\mu_\theta(s)+\epsilon,a_{\min},a_{\max})$，以增加探索。评估可使用无噪声 actor。行为噪声改变收集到的数据，不等于 TD3 在训练 target 中加入的平滑噪声。

<a id="lesson-derive"></a>

## 2 · 确定性策略梯度与实际 replay 目标

$$
\nabla_\theta J\ \propto\ \mathbb E_{s\sim d_\mu}\left[\nabla_\theta\mu_\theta(s)\,\nabla_a Q^\mu(s,a)\big|_{a=\mu_\theta(s)}\right]
$$

精确确定性策略梯度定理使用相应策略占据分布和真实 Q；正的归一化常数由占据分布定义决定。

实践中既不知道真实 $Q^\mu$，也不直接采样精确的 $d_\mu$。DDPG 用 replay 状态和近似 critic 构造 $L_\mu=-\mathbb E_{s\sim\mathcal D}Q_\phi(s,\mu_\theta(s))$。这是实际优化的 surrogate，不应不加条件地称为原始起点回报的无偏梯度。

$$
\nabla_\theta L_\mu=-\mathbb E_{\mathcal D}\left[\nabla_a Q_\phi(s,a)\big|_{a=\mu_\theta(s)}\nabla_\theta\mu_\theta(s)\right]
$$

actor 更新时固定 critic 参数，但保留 Q 对动作的导数。对整个 Q 前向使用 no_grad 会把所需的梯度一起删除。

例如 $Q(a)=-(a-0.8)^2$，$\mu_\theta=\theta$。在 $\theta=0$ 处，价值的上升方向为 $-2(0-0.8)=1.6$。这个方向来自 critic 的动作斜率。若 critic 在未见动作上虚构一座高峰，actor 也会向那座高峰移动。

<a id="lesson-target"></a>

## 3 · DDPG 的 critic 与缓慢目标

$$
y=r+\gamma(1-d)Q_{\bar\phi}(s',\mu_{\bar\theta}(s')),\qquad L_Q=\mathbb E_{\mathcal D}(Q_\phi(s,a)-\operatorname{sg}(y))^2
$$

d 只标识真正终止。target actor 与 target critic 都用于生成停止梯度的标签。

$$
\bar\phi\leftarrow(1-\tau)\bar\phi+\tau\phi,\qquad\bar\theta\leftarrow(1-\tau)\bar\theta+\tau\theta
$$

本章 $\tau$ 是新参数的比例。某些实现用 $\rho$ 表示旧参数比例，因而 $\rho=1-\tau$。把两个约定混用会把慢更新变成近乎直接复制。

target network 减缓自举标签随在线网络移动，并不将错误标签变成正确标签。Replay 提供状态动作覆盖和数据复用，但 actor 仍可能选到数据稀少的动作。网络容量、更新次数与数据覆盖共同决定误差。

<a id="lesson-td3"></a>

## 4 · TD3 的三个改动分别限制什么

$$
\begin{gathered}\tilde a'=\operatorname{clip}\left(\mu_{\bar\theta}(s')+\operatorname{clip}(\epsilon,-c,c),a_{\min},a_{\max}\right)\\\epsilon\sim\mathcal N(0,\sigma^2 I)\end{gathered}
$$

target policy smoothing 先裁剪噪声，再裁剪动作；行为探索噪声有另一套参数和用途。

$$
\begin{gathered}y=r+\gamma(1-d)\min_{i=1,2}Q_{\bar\phi_i}(s',\tilde a')\\L_{Q_i}=\mathbb E(Q_{\phi_i}(s,a)-\operatorname{sg}(y))^2\\L_\mu=-\mathbb E Q_{\phi_1}(s,\mu_\theta(s))\end{gathered}
$$

两个 critic 都拟合同一个较小目标；标准 TD3 actor 使用第一个 critic，不在 actor loss 中再取两个值的最小。

第一，双 critic 取最小值抑制部分过估计，也可能引入低估；两个网络相关时，它不是统计置信下界。第二，actor 延迟更新，让 critic 在两次策略变化之间多做学习。第三，目标动作附近的平滑减少对狭窄价值尖峰的依赖。这三处分别作用于目标值、更新时间和目标动作，不是一个统一的学习率技巧。

<a id="lesson-algorithm"></a>

## 5 · 先更新 critic，再按计数器更新 actor

**算法：配套代码以 delay=2 为默认值。计数器统计 critic 更新次数，不应在未知 update-to-data ratio 下直接当作环境步数。**

1. 采样 replay 的 $(s,a,r,s',d)$ minibatch。
1. 无梯度计算平滑目标动作和双 target critic 的最小值。
1. 计算一个固定 $y$，分别反传并更新两个在线 critic。
1. 若 critic 更新计数是 delay 的整数倍：
  1. 暂时冻结 critic 参数，但保留动作导数。
  1. 对 $-Q_{\phi_1}(s,\mu_\theta(s))$ 更新 actor。
  1. 恢复 critic 参数的可训练标志。
  1. 对 actor 和两个 critic 的 target 做 Polyak 更新。
1. 继续收集真实数据，并单独处理终止与 reset。

若一条环境交互触发多个梯度步骤，目标更新频率与 actor 更新频率会随之改变。复现实验必须同时记录环境步数、critic 步数、actor 步数与 replay 大小，不能只给一个总 steps。

<a id="lesson-example"></a>

## 6 · 目标动作、目标值与梯度手算

目标 actor 输出 $0.9$，抽样噪声为 $3$，噪声上限 $c=0.2$，动作范围为 $[-1,1]$。先得 $0.9+0.2=1.1$，再裁为 $1$。若两 target critic 在动作一上的值为 $3,2$，奖励为一、$\gamma=0.9$，则 target 为 $1+0.9\times2=2.8$。真终止时 target 只有一，噪声与 critic 均不影响它。

旧 target 参数为十，在线参数为二，$\tau=0.005$。新 target 为 $0.995\times10+0.005\times2=9.96$。若误把 0.995 当成新参数比例，会得到 2.04，更新时间尺度完全不同。

连续控制数值核包含 TD3 target、Polyak、SAC target 与 squash 密度；本节使用其中前两项。

```python
def td3_target(reward, gamma, terminated, action, noise, noise_clip,
               low, high, q1, q2):
    perturbation = max(-noise_clip, min(noise_clip, noise))
    smoothed = max(low, min(high, action + perturbation))
    target = reward + gamma * (not terminated) * min(q1(smoothed), q2(smoothed))
    return target, smoothed


def polyak(old, online, tau):
    if not 0 <= tau <= 1:
        raise ValueError('tau must lie in [0, 1]')
    return [(1-tau)*x + tau*y for x, y in zip(old, online)]


def tanh_log_prob(u, mean, log_std, scale=1.0):
    if scale <= 0:
        raise ValueError('positive affine action scale required')
    log_normal = -0.5*((u-mean)/math.exp(log_std))**2-log_std-0.5*math.log(2*math.pi)
    log_jacobian = 2*(math.log(2)-u-softplus(-2*u))
    return log_normal - log_jacobian - math.log(scale)


def sac_target(reward, gamma, terminated, q1, q2, log_prob, alpha):
    return reward + gamma*(not terminated)*(min(q1, q2)-alpha*log_prob)


def temperature_log_gradient(log_alpha, log_prob, target_entropy):
    # Exact gradient of -exp(log_alpha) * stopgrad(log_prob + target_entropy).
    return -math.exp(log_alpha)*(log_prob+target_entropy)
```

<a id="lesson-code"></a>

## 7 · 实际 autograd 更新核

DDPG 的单 critic 目标、确定性 actor 梯度及每次优化后的 target 平滑更新。

```python
def ddpg_update(actor, critic, target_actor, target_critic,
                actor_opt, critic_opt, batch, gamma=.99, tau=.005):
    x, action, reward, xp, terminal = batch
    with torch.no_grad():
        y = reward+gamma*(1-terminal)*target_critic(xp, target_actor(xp))
    prediction = critic(x, action)
    assert prediction.shape == y.shape == reward.shape
    loss_q = ((prediction-y)**2).mean()
    critic_opt.zero_grad(); loss_q.backward(); critic_opt.step()
    freeze([critic], True)
    loss_actor = -critic(x, actor(x)).mean()
    actor_opt.zero_grad(); loss_actor.backward(); actor_opt.step()
    freeze([critic], False)
    update_targets([actor, critic], [target_actor, target_critic], tau)
    return float(loss_q.detach()), float(loss_actor.detach())
```

两个 critic 的标签停止梯度；延迟步骤才更新 actor 与全部 target。

```python
def td3_update(actor, q1, q2, target_actor, target_q1, target_q2,
               actor_opt, critic_opt, batch, update_index, delay=2, gamma=.99, tau=.005):
    x, action, reward, xp, terminal = batch
    with torch.no_grad():
        next_action = target_actor(xp)
        noise = (.2*torch.randn_like(next_action)).clamp(-.5, .5)
        next_action = (next_action+noise).clamp(-1, 1)
        y = reward+gamma*(1-terminal)*torch.minimum(target_q1(xp, next_action), target_q2(xp, next_action))
    loss_q = ((q1(x, action)-y)**2+(q2(x, action)-y)**2).mean()
    critic_opt.zero_grad(); loss_q.backward(); critic_opt.step()
    policy_loss = None
    if update_index % delay == 0:
        freeze([q1, q2], True)
        # Freeze critic parameters, not the path from action to critic output.
        loss_actor = -q1(x, actor(x)).mean()
        actor_opt.zero_grad(); loss_actor.backward(); actor_opt.step()
        freeze([q1, q2], False)
        update_targets([actor, q1, q2], [target_actor, target_q1, target_q2], tau)
        policy_loss = float(loss_actor.detach())
    return float(loss_q.detach()), policy_loss
```

执行 python3 deep_textbook_train.py test 会检查非 actor 更新步的参数保持不变，并执行真实 TD3 critic/actor 梯度更新。此处没有 MuJoCo 采样器、完整 TD3 replay 训练循环或性能基准。需要完整实验时，以作者 TD3.py 和配套 main.py 为入口，对照动作缩放、噪声和训练步数。

本核默认动作已经归一化到 $[-1,1]$。环境的物理扭矩或速度范围需要外部可逆缩放。若直接在物理单位加入同一数值的噪声，探索强度和 target smoothing 强度会随单位改变。

<a id="lesson-branches"></a>

## 8 · 失败条件与 CRL 中的变化

- 梯度断开：冻结 critic 参数可以，detach actor 动作或对整个 actor loss 使用 no_grad 不可以。
- 旧数据失配：动力学改变后，同一个 (s,a) 可能对应新的奖励和转移；混合 replay 不再来自一个固定 Bellman 算子。
- 外推峰值：actor 可以优化到 critic 数据不足的动作区域；双网络并不消除这个问题。
- 动作饱和：tanh 接近边界时导数小，actor 可能很难离开边界；动作尺度和初始化都需要记录。

持续控制还需区分控制器记忆与学习器记忆。确定性 actor 可以是 recurrent 的，replay 也可以保存序列；但这会新增 hidden-state 重建与跨时间梯度问题。它不是在当前 feed-forward batch 上简单多加一列特征。

<a id="lesson-check"></a>

## 9 · 练习与答案

- 问：DDPG critic loss 中也更新 actor 会怎样？答：actor 会沿着降低拟合误差而不是提高 Q 的方向改变 target，失去指定更新的含义。
- 问：TD3 的两个 critic 一致就表示可信么？答：不表示。相同数据和相似函数类可以产生相关错误。
- 问：探索噪声与 target smoothing 可以共享同一随机样本吗？答：标准机制不要求共享；前者发生在真实行为，后者发生在 replay 标签构造，必须分清数据时间。
- 实验：对标量 Q(a)=−(a−0.8)² 做中心差分，比较 actor 的解析梯度 1.6；再 detach 动作，检查自动微分为何无法更新 actor。

<a id="chapter-code"></a>

## 下载与运行

标准库数值核验；完整小任务训练另需 deep_textbook_train.py 与 PyTorch。

[下载 deep_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/deep_textbook_lab.py)

```sh
python3 deep_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Spinning Up · DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)：用作算法与实现接口的对照；本章解释和配套小实验独立编写。

- [Spinning Up · TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html)：用作算法与实现接口的对照；本章解释和配套小实验独立编写。

- [Silver et al. · Deterministic Policy Gradient Algorithms](https://proceedings.mlr.press/v32/silver14.html)：确定性策略梯度定理及其占据分布条件。

- [Lillicrap et al. · Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)：DDPG：确定性 actor、replay 与 target network。

- [Fujimoto et al. · Addressing Function Approximation Error in Actor-Critic Methods](https://proceedings.mlr.press/v80/fujimoto18a.html)：TD3 原论文，ICML 2018。

- [Fujimoto · TD3.py](https://github.com/sfujim/TD3/blob/master/TD3.py)：作者实现；检查 policy_noise、noise_clip、policy_freq 与 target 更新时机。

- [Spinning Up · ddpg.py](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py)：官方教学实现，包含完整采样与 replay 循环。

<a id="study-connections"></a>

## 与教材主线的衔接

- [控制问题与广义策略迭代](https://yingwen.io/zh/continual-rl/algorithms/control/)
- [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)
- [最大熵控制](https://yingwen.io/zh/continual-rl/algorithms/soft-control/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

对应原始材料：Sutton & Barto §13.7；Deterministic Policy Gradient §3；TD3 §4–5。本文为原创讲解，原书、论文与上游代码保留各自许可。
