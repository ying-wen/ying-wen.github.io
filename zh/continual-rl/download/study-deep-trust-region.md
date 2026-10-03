# 策略更新的尺度：TRPO 与 PPO

同一批数据可以重复使用多少次？为什么 clipping 不是一个性能保证？

## 本章内容

- 由性能差分恒等式解释 surrogate 的来源。
- 求解局部 KL 约束，理解 Fisher、共轭梯度与回溯。
- 按优势符号解释 PPO，并固定旧策略、优势与 critic target。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### Bellman 递推

当前价值等于即时奖励加折扣后的后继价值；预测目标与最优控制目标不同。

### 函数与梯度

神经网络把参数和输入映射为输出。链式法则必须指明哪些量参与求导，哪些量作为固定目标。

### 经验分布

on-policy 数据由当前策略产生；off-policy 数据可来自旧策略，但能否正确复用取决于算法目标和数据覆盖。

<a id="lesson-setting"></a>

## 1 · 旧策略的数据与新策略的状态分布

策略梯度给出局部上升方向，却没有给出安全步长。同一梯度方向走得太远，策略可能进入旧数据很少覆盖的状态。令旧策略为 $\pi_0$，候选策略为 $\pi_\theta$，优势为 $A^{\pi_0}$。本节的理论采用奖励有界、$0<\gamma<1$ 的无限时域折扣 MDP。训练代码则采用有限 episode、$\gamma=1$；它使用同类 surrogate，不直接满足下面折扣定理的所有条件。

$$
d_\pi(s)=(1-\gamma)\sum_{t=0}^{\infty}\gamma^t\Pr_\pi(S_t=s),\qquad r_\theta(s,a)=\frac{\pi_\theta(a\mid s)}{\pi_0(a\mid s)}
$$

$d_\pi$ 是归一化折扣状态分布。概率比只修正给定状态中的动作分布，不会自动把旧状态分布变成新状态分布。

本章使用覆盖候选动作的旧策略。若旧策略对某个动作的概率为零，普通重要性比没有定义；小批量估计也无法凭空得到该动作的优势。

<a id="lesson-derive"></a>

## 2 · 从性能差分到局部 surrogate

$$
J(\pi_\theta)-J(\pi_0)=\frac{1}{1-\gamma}\mathbb E_{s\sim d_{\pi_\theta},a\sim\pi_\theta}[A^{\pi_0}(s,a)]
$$

将 $A=Q-V$ 展开为一步奖励与两个价值项，再沿新策略轨迹求折扣和；中间价值项望远镜相消。

右侧仍需新策略的状态访问分布。可采样的近似是用 $d_{\pi_0}$ 替换它，再用动作概率比。新旧策略相同时，surrogate 的值与一阶导数都对齐，但远离旧策略后不再是精确性能。

$$
\begin{gathered}L_{\pi_0}(\pi_\theta)=J(\pi_0)+\frac{1}{1-\gamma}\mathbb E_{d_{\pi_0},\pi_0}[r_\theta A^{\pi_0}]\\J(\pi_\theta)\ge L_{\pi_0}(\pi_\theta)-\frac{4\gamma\epsilon_A}{(1-\gamma)^2}\alpha_{\rm TV}^{\,2}\end{gathered}
$$

这里 $\epsilon_A=\max_{s,a}|A^{\pi_0}(s,a)|$。策略差异为 $\alpha_{\rm TV}=\max_s D_{\rm TV}(\pi_0(\cdot\mid s),\pi_\theta(\cdot\mid s))$。这是 TRPO 使用的最坏状态差异形式之一。

这个界解释了为什么要限制策略变化。它也显示理论与实现的距离：算法估计有限样本平均 KL，不直接约束所有状态的最大距离；优势由 critic 估计；共轭梯度和回溯也有数值误差。实际 TRPO 不能因此宣称每一步真实回报必然增加，PPO 更没有继承该保证。

<a id="lesson-local"></a>

## 3 · Fisher、共轭梯度与回溯

$$
\begin{gathered}\max_\Delta g^\top\Delta\quad\text{s.t.}\quad\tfrac12\Delta^\top F\Delta\le\delta\\F=\left.\nabla_\theta^2\mathbb E_{s\sim d_{\pi_0}}D_{\rm KL}(\pi_0\Vert\pi_\theta)\right|_{\theta=\theta_0}\end{gathered}
$$

$g$ 是样本 surrogate 在旧参数处的梯度；$F$ 是 KL 的局部曲率。旧策略的概率必须固定，不参与求导。

$$
x=F^{-1}g,\qquad\Delta_* =\sqrt{\frac{2\delta}{g^\top x}}\,x
$$

拉格朗日条件给出 $g=\eta F\Delta$，再把二次约束取等号得到缩放。若梯度为零，则不需要更新。

神经网络不显式存储整个 $F$。给定向量 $v$，二次自动微分计算 $Fv=\nabla_\theta[(\nabla_\theta\bar D_{\rm KL})^\top v]$。共轭梯度只需要这个乘法接口，就能近似解线性系统。加入 $\kappa I$ 阻尼有助于处理奇异和数值噪声，但求解的已是阻尼后的方向。

局部二次近似不保证候选点的真实 KL 合格。回溯依次尝试全步、半步、四分之一步，重新计算样本 surrogate 与实际样本 KL；未满足接受条件则恢复旧参数。回溯使用的仍是当前数据，不是对真实环境性能的证明。

<a id="lesson-ppo"></a>

## 4 · PPO 的符号分支与固定量

$$
L^{\rm clip}(\theta)=\mathbb E\left[\min\left(r_\theta\hat A,\operatorname{clip}(r_\theta,1-\varepsilon,1+\varepsilon)\hat A\right)\right]
$$

最大化这个目标；代码通常对其负数做梯度下降。

$$
\ell(r,A)=\begin{cases}A\min(r,1+\varepsilon),&A\ge0,\\A\max(r,1-\varepsilon),&A<0.\end{cases}
$$

正优势动作的过度概率上升、负优势动作的过度概率下降，不再继续得到相同的 surrogate 奖励。

clipping 不是把网络参数投影回某个约束集，也不强制所有概率比留在区间内。共享网络会联动改变其他状态和动作，样本不覆盖的地方更没有直接约束。多轮更新时，旧 log-probability、原始优势和 critic target 全部固定；只有当前策略、当前 critic 和优化器状态变化。

配套代码记录 $\widehat{\mathrm{KL}}=\operatorname{mean}(r-1-\log r)$。在旧策略采样、归一化策略与适当覆盖下，其期望对应旧到新 KL。单批估计仍有误差。代码在更新前检查这个量并提前停止，最后一次更新仍可能越过阈值，所以它不是硬约束。

<a id="lesson-algorithm"></a>

## 5 · 两类算法的精确更新次序

**算法：本教学实现的 actor 与 critic 分开。共享 encoder 的实现必须另外说明共享参数在各阶段如何改变。**

1. 用旧 actor 和 critic 采样；固定旧 log probability。
1. 用旧 value 计算 GAE；固定 critic target；只标准化 actor 优势。
1. TRPO：计算 $g$，用 Hessian-vector product 与 CG 求方向。
  1. 缩放方向，回溯检查实际样本 KL 和 surrogate；失败则恢复参数。
1. PPO：在每轮梯度更新前计算当前 ratio、clipped loss 与样本 KL。
  1. KL 已超过阈值则退出；否则更新 actor。
1. 用固定回归目标训练 critic；重新收集 on-policy 数据。

Fisher-vector product 用二次自动微分求出，并与 categorical Fisher 的解析式对照。

```python
def fisher_vector_product(logits, vector, damping=0.):
    old_prob = logits.detach().softmax(-1)
    old_logp = logits.detach().log_softmax(-1)
    kl = (old_prob*(old_logp-logits.log_softmax(-1))).sum(-1).mean()
    gradient = torch.autograd.grad(kl, logits, create_graph=True)[0]
    product = torch.autograd.grad((gradient*vector).sum(), logits)[0]
    return product+damping*vector
```

<a id="lesson-example"></a>

## 6 · 一维 TRPO 与两个 clipping 例子

单状态两个动作，$\pi_\theta(1)=\sigma(\theta)$，旧 $\theta=0$，优势分别为 $1,-1$。此时 $g=0.5$，$F=0.25$。令 KL 半径 $\delta=0.01$，局部全步为 $\Delta=\sqrt{0.08}\approx0.28284$。新动作一概率约为 $0.57024$，实际旧到新 KL 为 $\log\cosh(\Delta/2)\approx0.009967$，因此全步满足这一例的约束。

PPO 取 $\varepsilon=0.2$。若 $A=2,r=1.4$，目标是 $\min(2.8,2.4)=2.4$。若 $A=-2,r=0.6$，目标是 $\min(-1.2,-1.6)=-1.6$。第二例很容易写反：负优势下概率减少过多时，截断取的是更负的一项。

共轭梯度、单状态精确 KL 回溯和 PPO 符号分支。这个 TRPO 数值核不是完整神经网络训练器。

```python
def ppo_term(ratio, advantage, clip=0.2):
    clipped = min(1 + clip, max(1 - clip, ratio))
    return min(ratio * advantage, clipped * advantage)


def conjugate_gradient(matvec, b, iterations=20, tolerance=1e-12):
    x, residual = [0.0] * len(b), list(b)
    direction = list(residual)
    rr = dot(residual, residual)
    for _ in range(iterations):
        if rr <= tolerance * tolerance:
            break
        product = matvec(direction)
        curvature = dot(direction, product)
        if curvature <= 0:
            raise ValueError('positive curvature is required')
        step = rr / curvature
        x = [v + step * d for v, d in zip(x, direction)]
        residual = [r - step * p for r, p in zip(residual, product)]
        new_rr = dot(residual, residual)
        direction = [r + new_rr / rr * d
                     for r, d in zip(residual, direction)]
        rr = new_rr
    return x


def categorical_kl(old, new):
    return sum(p * math.log(p / q) for p, q in zip(old, new) if p > 0)


def trpo_binary(theta, advantage_one, advantage_zero, delta=0.01):
    """Exact scalar Fisher + actual KL/line search for one-state policy."""
    prob = lambda z: 1.0 / (1.0 + math.exp(-z))
    old_p = prob(theta)
    surrogate = lambda z: prob(z) * advantage_one + (1-prob(z)) * advantage_zero
    gradient = old_p * (1-old_p) * (advantage_one-advantage_zero)
    if abs(gradient) < 1e-14:
        return theta, 0.0, 0.0
    fisher = old_p * (1-old_p)
    natural = gradient / fisher
    full_step = math.sqrt(2*delta/(natural*fisher*natural)) * natural
    for power in range(20):
        step = 0.5**power * full_step
        candidate = theta + step
        kl = categorical_kl([old_p, 1-old_p], [prob(candidate), 1-prob(candidate)])
        gain = surrogate(candidate)-surrogate(theta)
        if kl <= delta and gain >= 0.1 * gradient * step:
            return candidate, kl, gain
    return theta, 0.0, 0.0
```

<a id="lesson-code"></a>

## 7 · 可运行 PPO 与实现边界

旧概率和优势来自固定 rollout。critic 使用未标准化的回归目标。

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

将 deep_textbook_train.py 与 deep_textbook_lab.py 放在同一目录。安装 deep_requirements.txt 后执行 python3 deep_textbook_train.py ppo --epochs 16 --seed 0。该命令在内置小 MDP 中真实采样、计算 GAE、做 PPO 更新并独立评估；没有 Gym 或 GPU 依赖。

完整 TRPO 工程入口是 Spinning Up 的 TensorFlow 1 实现；该项目没有对应的官方 PyTorch TRPO 目录。配套 PyTorch 代码只提供 Fisher-vector product，标准库代码提供 CG 与一维回溯。完整网络 TRPO 还需参数向量化、分布接口和整批 line search。

<a id="lesson-branches"></a>

## 8 · 失效条件与持续适应

- 旧策略被覆盖：若每轮内重新保存当前 log_prob 作为 old，概率比总接近一，算法不再控制相对于采样策略的变化。
- critic target 漂移：每轮重新用已更新 critic 生成优势，会把多种变化混在同一目标中。
- 小 batch 的 KL 噪声：阈值过小可能几乎不更新，阈值过大又不能有效限制分布迁移。
- 变化环境：旧策略数据可以在收集完时就不再代表当前动力学。KL 小只表示策略分布接近，不表示环境未变。

PPO 的参数复用和 replay 不是一回事。它复用的是最近采样策略的一批数据，并保存其动作概率。把任意陈旧任务轨迹放入同一循环，不能只保留 clipped loss 就称为正确的 off-policy 算法。

<a id="lesson-check"></a>

## 9 · 练习与答案

- 问：KL 小是否证明回报提高？答：不证明。真实优势、覆盖与状态分布误差仍有影响。
- 问：把所有 ratio 直接 clip 后再乘优势，等价于 PPO 吗？答：不等价。PPO 是两个目标取最小，某些变坏方向仍必须保留梯度。
- 问：为何使用 CG 而不直接求逆？答：网络参数很多，存储曲率矩阵需要平方级空间；Hessian-vector product 不需要显式矩阵。
- 实验：把一维例子的 δ 增大，比较二次预测 KL 与实际 KL，并观察回溯是否缩步；再将优势都设为零，确认参数保持不变。

<a id="chapter-code"></a>

## 下载与运行

标准库数值核验；完整小任务训练另需 deep_textbook_train.py 与 PyTorch。

[下载 deep_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/deep_textbook_lab.py)

```sh
python3 deep_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Spinning Up · TRPO](https://spinningup.openai.com/en/latest/algorithms/trpo.html)：用作算法与实现接口的对照；本章解释和配套小实验独立编写。

- [Spinning Up · PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)：用作算法与实现接口的对照；本章解释和配套小实验独立编写。

- [Schulman et al. · Trust Region Policy Optimization](https://proceedings.mlr.press/v37/schulman15.html)：性能下界、局部近似与实际 TRPO 的区别。

- [Schulman et al. · Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)：clipped surrogate 与多轮小批量更新。

- [Spinning Up · trpo.py](https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/trpo/trpo.py)：官方 TensorFlow 1 TRPO，实现 CG、Hessian-vector product 和回溯。

- [Spinning Up · ppo.py](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py)：官方 PyTorch PPO 与 KL 提前停止。

<a id="study-connections"></a>

## 与教材主线的衔接

- [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)
- [控制问题与广义策略迭代](https://yingwen.io/zh/continual-rl/algorithms/control/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

对应原始材料：Sutton & Barto §13.2–13.5；TRPO §3–5；PPO §3。本文为原创讲解，原书、论文与上游代码保留各自许可。
