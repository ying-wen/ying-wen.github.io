# 最大熵连续控制：SAC 的价值、密度与温度

随机 actor 不只是加噪声：熵如何进入 Bellman 方程与自动微分？

## 本章内容

- 推导 soft Bellman 与 actor 的 KL 投影。
- 计算 tanh 与动作缩放后的概率密度。
- 区分 actor、critic、温度三类梯度与停止梯度位置。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### Bellman 递推

当前价值等于即时奖励加折扣后的后继价值；预测目标与最优控制目标不同。

### 函数与梯度

神经网络把参数和输入映射为输出。链式法则必须指明哪些量参与求导，哪些量作为固定目标。

### 经验分布

on-policy 数据由当前策略产生；off-policy 数据可来自旧策略，但能否正确复用取决于算法目标和数据覆盖。

<a id="lesson-setting"></a>

## 1 · 熵是目标的一部分

SAC 优化奖励和策略熵的加权和。$\alpha>0$ 称为温度。它控制奖励与随机性之间的权衡，不是 actor 的学习率。这里动作连续，熵指微分熵；它依赖坐标单位，也可以为负。确定性评估动作与训练时的随机策略需要分别记录。

$$
\begin{gathered}J_\alpha(\pi)=\mathbb E_\pi\sum_{t\ge0}\gamma^t\left[R_{t+1}+\alpha\mathcal H(\pi(\cdot\mid S_t))\right]\\\mathcal H(\pi)=-\mathbb E_{a\sim\pi}\log\pi(a\mid s)\end{gathered}
$$

奖励和熵均按时间累计。标准 SAC 的 replay 训练使用实际状态样本近似相应目标。

更高的单步动作熵不保证发现遥远目标。对多个状态都随机抖动，仍可能无法形成连贯的探索轨迹。最大熵目标、内在奖励和有记忆的探索策略是不同机制。

<a id="lesson-derive"></a>

## 2 · soft Bellman 与策略改进

$$
\begin{gathered}Q^\pi(s,a)=\mathbb E[R_{t+1}+\gamma V^\pi(S_{t+1})\mid s,a]\\V^\pi(s)=\mathbb E_{a\sim\pi}[Q^\pi(s,a)-\alpha\log\pi(a\mid s)]\end{gathered}
$$

此约定的 Q 不含当前已指定动作的熵项，V 才对当前动作分布加入熵；Q 的未来展开包含后续熵。

固定状态与 Q，选择策略最大化 $\int\pi(a)[Q(a)-\alpha\log\pi(a)]\,da$。用拉格朗日乘子满足密度积分为一，对 $\pi(a)$ 求变分导数，得到 $Q(a)-\alpha(1+\log\pi(a))+\eta=0$。归一化后即下面的 Boltzmann 密度。有限配分函数与适当可积性是该推导的条件。

$$
\begin{gathered}\pi^*(a\mid s)=\frac{\exp(Q(s,a)/\alpha)}{Z(s)}\\D_{\rm KL}\!\left(\pi_\theta\middle\Vert\frac{\exp(Q/\alpha)}{Z}\right)=\frac{1}{\alpha}\mathbb E_{\pi_\theta}[\alpha\log\pi_\theta-Q]+\log Z\end{gathered}
$$

Z 与待优化 actor 参数无关，因此最小化这个 KL 等价于最小化 actor loss。参数化 Gaussian 通常不能精确表示 Boltzmann 密度。

常用双 Q 版本没有单独的 value network：下一状态的 actor 给出动作与 log-density，target critic 给出价值。这里使用后续 SAC 版本的这一结构；不要把早期带 value network 的实现与当前更新式混在一起。

<a id="lesson-density"></a>

## 3 · 重参数化与 tanh 的 Jacobian

$$
\begin{gathered}\epsilon\sim\mathcal N(0,I)\\u=\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon\\a=\tanh u\\\log\pi_\theta(a\mid s)=\sum_j\left[\log\mathcal N(u_j;\mu_j,\sigma_j^2)-\log(1-\tanh^2u_j)\right]\end{gathered}
$$

固定基础噪声 $\epsilon$ 后，动作是参数的可微函数。actor 的 $Q$ 项通过动作回传梯度；不能使用不保留路径的普通 sample。

密度变换满足 $p_a(a)=p_u(u)/|\det(\partial a/\partial u)|$。tanh 的导数是 $1-\tanh^2u$，所以 log-density 必须减去 log-Jacobian。各动作维度独立的 Gaussian 输出逐维 log_prob，求和后才得到形状为 batch 的联合动作密度。

$$
\log(1-\tanh^2u)=2\left[\log2-u-\operatorname{softplus}(-2u)\right]
$$

这个等式避免直接计算 $1-\tanh^2u$ 时的浮点消减。饱和区仍可能有很大的密度修正，但不必先算出 $\log(0)$。

若物理动作为 $a^{\rm env}_j=b_j+c_j\tanh u_j$，还需再减去 $\sum_j\log|c_j|$。固定温度时这个常数不改变 actor 的参数梯度，但会改变熵数值和自动温度的目标关系。配套 PyTorch 核只实现 $[-1,1]$ 动作；标准库测试另外检查仿射缩放修正。

<a id="lesson-temperature"></a>

## 4 · 熵约束与温度更新的符号

若要求平均熵不低于目标 $\bar{\mathcal H}$，其约束为 $\mathbb E[-\log\pi]\ge\bar{\mathcal H}$。将约束加入奖励优化，可对非负乘子 $\alpha$ 做对偶下降。固定当前策略样本时，温度损失为 $L_\alpha=-\alpha\,\operatorname{sg}(\log\pi+\bar{\mathcal H})$。熵过低意味着括号为正，下降会增加温度。

$$
\begin{gathered}\alpha=e^\beta\\L_\beta=-e^\beta\operatorname{sg}(\log\pi+\bar{\mathcal H})\\\nabla_\beta L_\beta=-\alpha\,\operatorname{sg}(\log\pi+\bar{\mathcal H})\end{gathered}
$$

$\beta$ 保证温度为正。策略样本在温度 loss 中固定，actor 更新时则固定温度。

一些工程采用代理损失 $-\beta\,\operatorname{sg}(\log\pi+\bar{\mathcal H})$。其梯度没有乘 $\alpha$，因此不是上式的精确链式导数。正温度下零点相同，但更新尺度不同。本配套代码实现精确参数化；对照其他仓库时需要检查实际损失，不能只看变量名 log_alpha。

数值例：$\alpha=0.2,\log\pi=2,\bar{\mathcal H}=-1$。当前样本熵为 $-2$，低于目标 $-1$。于是梯度为 $-0.2$，梯度下降增大 $\beta$ 和 $\alpha$。连续密度可大于一，所以 log-density 为正与概率论并不冲突。

<a id="lesson-algorithm"></a>

## 5 · critic、actor 与温度的更新次序

$$
y=r+\gamma(1-d)\left[\min_iQ_{\bar\phi_i}(s',a')-\alpha\log\pi_\theta(a'\mid s')\right],\quad a'\sim\pi_\theta(\cdot\mid s')
$$

整个 y 停止梯度；SAC 使用当前 actor 生成下一动作，而非 TD3 式 target actor。

$$
\begin{gathered}L_{Q_i}=\mathbb E(Q_{\phi_i}(s,a)-\operatorname{sg}(y))^2\\L_\pi=\mathbb E_{\mathcal D,\epsilon}\left[\alpha\log\pi_\theta(a_\theta\mid s)-\min_iQ_{\phi_i}(s,a_\theta)\right]\end{gathered}
$$

actor loss 中 critic 参数固定，动作路径保持可微；温度作为常数。

**算法：这是配套核选定的次序。温度使用 actor 更新前算出的那批样本；重新采样也是一种实现选择，但必须明确。**

1. 采样 replay batch，读取真正终止标志。
1. 固定温度；无梯度计算下一动作、log-density 与 soft target。
1. 用固定 target 更新两个 critic。
1. 冻结 critic 参数；重参数化采样当前动作。
1. 更新 actor 的 entropy-minus-Q loss；恢复 critic 可训练标志。
1. 使用刚才样本的 detached log-density 更新 log temperature。
1. 仅对两个 target critic 做 Polyak 更新。

<a id="lesson-example"></a>

## 6 · soft target 与密度手算

取奖励一、$\gamma=0.9$、两个 target Q 的最小值二、$\log\pi=-0.5$、$\alpha=0.2$。soft value 为 $2-0.2(-0.5)=2.1$，target 为 $1+0.9\times2.1=2.89$。相比同一数值下 TD3 的 2.8，多出的不是环境奖励，而是目标内的熵项。

一维标准 Normal 在 $u=0$ 的 log-density 为 $-\frac12\log(2\pi)\approx-0.91894$。这里 tanh 导数为一，所以 squash 修正为零。若再映射到 $[-2,2]$，需要减去 $\log2$，得到约 $-1.61209$。这解释了为何温度目标不能脱离动作尺度。

Normal.rsample 保留动作路径；稳定 Jacobian 与动作维度求和组成完整的 log-density。

```python
class SquashedGaussian(nn.Module):
    def __init__(self, observation_dim=3, action_dim=2):
        super().__init__()
        self.net = mlp(observation_dim, 2*action_dim)

    def forward(self, observation):
        mean, log_std = self.net(observation).chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)
        normal = Normal(mean, log_std.exp())
        u = normal.rsample()
        action = u.tanh()
        log_jacobian = 2*(math.log(2)-u-nn.functional.softplus(-2*u))
        log_prob = (normal.log_prob(u)-log_jacobian).sum(dim=-1)
        return action, log_prob
```

<a id="lesson-code"></a>

## 7 · 可执行的 SAC 更新核

双 critic、重参数化 actor 与精确 log-temperature 对偶下降。

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

执行 python3 deep_textbook_lab.py test 检查 tanh 密度、动作缩放、soft target 与温度有限差分；执行 python3 deep_textbook_train.py test 检查动作与 log-probability shape、可微采样路径和实际 SAC 更新。此处只提供更新核，没有完整连续环境采样、replay 训练和 benchmark。

Spinning Up 的 SAC 教学实现使用固定 alpha，适合对照核心损失和停止梯度；自动温度见 SAC Algorithms and Applications。作者早期 sac 仓库与后续 softlearning 仓库所处算法版本不同，不能认为所有源码都应含有相同网络。

<a id="lesson-branches"></a>

## 8 · 失败条件：密度、尺度与数据覆盖

- 对 Gaussian 动作直接 clip，却仍使用原 Gaussian 密度：边界出现概率质量，原密度公式失效。
- 多维 log_prob 未求和：temperature loss 与 actor loss 的维度和尺度都可能错误。
- reward scale 改变而 alpha 不变：奖励与熵的相对权重随之改变，已经不是同一优化问题。
- 只检查两个 critic 的 loss：actor 仍可能访问 critic 外推区域，低训练误差不代表可靠控制。

在持续任务中，温度调节回答的是随机性目标是否满足，不回答表征是否足够、旧任务是否遗忘或参数是否失去可塑性。它是一种特定学习参数的自适应机制，不能代表所有元学习问题。

<a id="lesson-check"></a>

## 9 · 练习与答案

- 问：SAC 的温度等于 actor 学习率吗？答：不是。温度出现在目标里，学习率控制优化该目标时的参数步长。
- 问：为什么 actor 更新不能 detach sampled_action？答：Q 项需要通过动作对 actor 求导；detach 会删除这一项。
- 问：连续动作的目标熵为负是否错误？答：不错误，微分熵可为负并依赖动作单位。
- 实验：将动作尺度从一改为二，确认 log-density 减少 log2；保持目标熵不变时观察温度梯度如何变化，再讨论怎样平移目标以保持同等约束。

<a id="chapter-code"></a>

## 下载与运行

标准库数值核验；完整小任务训练另需 deep_textbook_train.py 与 PyTorch。

[下载 deep_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/deep_textbook_lab.py)

```sh
python3 deep_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Spinning Up · SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)：用作算法与实现接口的对照；本章解释和配套小实验独立编写。

- [Haarnoja et al. · Soft Actor-Critic](https://proceedings.mlr.press/v80/haarnoja18b.html)：ICML 2018 最大熵 actor–critic 原论文。

- [Haarnoja et al. · Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)：后续 SAC 结构与自动温度的约束推导。

- [Spinning Up · sac.py](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py)：官方 PyTorch 教学实现，温度 alpha 固定。

- [Haarnoja · sac](https://github.com/haarnoja/sac)：作者早期实现；包含的算法结构与后续双 Q、无独立 V 版本需分别对照。

- [RAIL · softlearning](https://github.com/rail-berkeley/softlearning)：作者团队后续连续控制框架；使用其配置时需记录版本与算法设置。

<a id="study-connections"></a>

## 与教材主线的衔接

- [最大熵控制](https://yingwen.io/zh/continual-rl/algorithms/soft-control/)
- [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)
- [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)
- [元学习与学习规则的适应](https://yingwen.io/zh/continual-rl/algorithms/meta/)

对应原始材料：SAC §4；SAC Algorithms and Applications §5；Spinning Up SAC key equations。本文为原创讲解，原书、论文与上游代码保留各自许可。
