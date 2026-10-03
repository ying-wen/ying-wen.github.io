# 策略梯度、基线与 Actor–Critic

直接学习策略时，哪一个目标的梯度能由经验估计，近似从哪里进入？

## 本章内容

- 从轨迹概率推导带完整折扣权重的 REINFORCE。
- 证明动作无关基线不改变梯度，并定位 critic 引入的偏差。
- 区分普通梯度、自然梯度以及平均奖励目标的采样权重。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 可微随机策略

策略参数决定动作概率，而环境转移不直接依赖这些参数。需要策略具有合适支持，并满足交换求导与期望所需的正则条件。

$$
\pi_\theta(a\mid s),\qquad \psi_t=\nabla_\theta\log\pi_\theta(A_t\mid S_t)
$$

### 起点分布

本章先固定回合起点分布，优化从该分布出发的折扣回报。状态访问频率不是任意可替换的采样分布。

<a id="lesson-setting"></a>

## 1 · 直接优化策略的目标

$$
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^tR_{t+1}\right].
$$

$T$ 可以是有限随机终点。策略通过动作改变未来状态分布，因此不能只把当前状态视为固定样本、对即时奖励求导。先写清目标，才能判断实现中的折扣与采样是否正确。

价值控制通过改善动作价值间接改变策略。策略梯度直接参数化动作概率，适合表达随机行为，也方便处理连续动作。它不意味着不需要价值学习；价值估计常用来降低策略更新的方差。

$$
p_\theta(\tau)=\mu(S_0)\prod_{t=0}^{T-1}\pi_\theta(A_t\mid S_t)p(S_{t+1},R_{t+1}\mid S_t,A_t).
$$

起点分布与环境转移不依赖 $\theta$，所以对轨迹对数概率求导时，只留下每一步的策略 score。环境无需可微。

<a id="lesson-derive"></a>

## 2 · 从轨迹导数到 REINFORCE

$$
\nabla J=\mathbb E\left[\left(\sum_t\psi_t\right)\left(\sum_k\gamma^kR_{k+1}\right)\right].
$$

使用对数导数恒等式，将未知的环境依赖包含在采样轨迹中。此时一个动作的 score 似乎乘上了全部奖励。

发生在动作之前的奖励，与当前动作的随机选择无关。给定动作前历史，策略 score 的期望为零，因此这些过去奖励对期望梯度的贡献为零。去掉它们减少无用方差，得到 reward-to-go。

$$
\begin{aligned}G_t&=\sum_{k=t}^{T-1}\gamma^{k-t}R_{k+1},\\ \nabla J&=\mathbb E\sum_{t=0}^{T-1}\gamma^tG_t\psi_t.\end{aligned}
$$

外层 $\gamma^t$ 与回报内部的相对折扣不是同一个因子。本页目标是从起点计算的折扣和；只保留 $G_t$ 而漏掉外层因子，一般不再是它的梯度。

$$
\begin{aligned}d_\gamma^\pi(s)&=\sum_{t\ge0}\gamma^t\Pr_\pi(S_t=s),\\ \nabla J&=\sum_s d_\gamma^\pi(s)\sum_a\nabla\pi_\theta(a\mid s)q_\pi(s,a).\end{aligned}
$$

这是策略梯度定理的占用权重表达。这里的 $d_\gamma$ 没有归一化；若改用 $(1-\gamma)d_\gamma$，就要保留相应的 $1/(1-\gamma)$ 系数。

策略梯度定理并不是忽略了状态分布对参数的影响。递归展开价值导数后，这部分影响已被吸收到占用权重和动作价值中。随意用 replay 中的状态频率替换它，会引入另一个离策略估计问题。

<a id="policy-baseline"></a>

## 3 · 基线为何无偏，critic 何时有偏

$$
\sum_a\pi_\theta(a\mid s)b(s)\nabla\log\pi_\theta(a\mid s)=b(s)\nabla\sum_a\pi_\theta(a\mid s)=0.
$$

只要基线在给定状态或动作前历史后不依赖当前采样动作，减去它就不改变期望 score 梯度。策略更新时对基线和回报停止梯度，不能把它们当作额外可微目标。

取基线为 $v_\pi(s)$ 得到优势 $A_\pi(s,a)=q_\pi(s,a)-v_\pi(s)$。状态价值是自然的选择，但不总是使所有参数方向总方差最小的基线；最优方差基线还可能依赖 score 的平方范数。

$$
\begin{aligned}\delta_t&=R_{t+1}+\gamma V_w(S_{t+1})-V_w(S_t),\\ \Delta\theta&=\alpha_\theta\gamma^t\delta_t\psi_t.\end{aligned}
$$

当 $V_w=v_\pi$ 且状态是合适的 Markov 状态时，条件期望 $\mathbb E[\delta_t\mid S_t,A_t]=A_\pi(S_t,A_t)$。近似 critic 不精确时，一步 bootstrap 通常带来策略梯度偏差。

MC 基线只减去动作无关项，仍可保持策略梯度无偏；把完整回报换成近似 TD 目标则是另一项近似。两者常一起出现，但作用不同。训练 critic 的损失也不能简单加到策略损失后任其通过所有共享路径反传。共享表示需要明确各项梯度如何作用。

<a id="policy-algorithm"></a>

## 4 · 数据、梯度与参数更新的顺序

**算法：回合批量 REINFORCE**

1. 用当前固定策略采样一个完整回合，保存动作和采样时的策略信息
1. 从末尾递推 $G_t=R_{t+1}+\gamma G_{t+1}$，真正终止处尾项为零
1. 在采样参数下计算 score 和动作无关基线
1. 累加 $g=\sum_t\gamma^t(G_t-b_t)\nabla\log\pi_\theta(A_t\mid S_t)$
1. 执行一次策略更新；基线可用回报作为监督目标单独更新

若在回合结束后对同一批数据做多次大幅策略更新，后续参数已偏离采样策略。原始在策略梯度等式不再自动适用，需要重要性修正、合适的近似目标或重新采样。本页实现返回梯度而不在枚举过程中改变策略，便于与解析导数比较。

一步 actor–critic 则可以每步更新。它以更早反馈和较低方差换取 critic 近似误差，通常还有两个步长或时间尺度。优势标准化、熵奖励、梯度裁剪和目标网络都属于额外机制，应在这个基本估计器之外逐项解释。

<a id="policy-continuous"></a>

## 5 · 连续动作：高斯策略的两个 score

连续动作可以使用高斯策略 $A\sim\mathcal N(\mu_\theta(s),\sigma_\theta(s)^2)$。均值控制偏好的动作，标准差控制随机程度。用对数标准差 $\ell=\log\sigma$ 参数化，可保证标准差为正。

$$
\begin{aligned}\partial_\mu\log\pi(a\mid s)&=\frac{a-\mu}{\sigma^2},\\ \partial_\ell\log\pi(a\mid s)&=\frac{(a-\mu)^2}{\sigma^2}-1.\end{aligned}
$$

把这两个局部导数通过均值与对数标准差网络继续链式求导，就得到策略 score。它们来自概率密度的导数，不要求奖励对动作可微。

真实执行动作若经过裁剪或可逆变换，必须说明策略密度究竟定义在哪个动作变量上。可逆 tanh 变换需要密度的 Jacobian 修正；直接把裁剪后的动作代入未裁剪高斯密度，通常不是执行策略的正确似然。本页枚举实验使用离散策略，不实现连续控制训练。

<a id="lesson-example"></a>

## 6 · 四条轨迹的精确梯度检查

两步决策共用一个 Bernoulli 参数 $p=\sigma(\theta)$。动作 1 的概率为 p，动作 0 的概率为 1−p。第一步奖励为 0；只有动作序列 (1,0) 在第二步得到奖励 1，其余三条轨迹奖励为 0。

$$
\begin{aligned}J(\theta)&=\gamma p(1-p),\\ \frac{dJ}{d\theta}&=\gamma p(1-p)(1-2p),\\ \nabla\log\pi(A_t)&=A_t-p.\end{aligned}
$$

先直接写出四条轨迹的概率，只有一条贡献收益。对 sigmoid 求导即可得到解析答案。

取 $\theta=0.7,\gamma=0.5$，则 $p\approx0.668188,J\approx0.110856$，解析梯度约为 −0.0372894。对获奖轨迹，$G_0=0.5,G_1=1$，两次 score 都应乘上总权重 0.5。

代码分别用轨迹枚举、解析公式和目标函数有限差分求梯度，三者相等。再减去常数基线 3，单条轨迹的梯度变化，但四条轨迹的概率加权期望不变。这个例子同时检查外层折扣、reward-to-go 和基线。

<a id="policy-natural"></a>

## 7 · 自然梯度与平均奖励的不同几何

$$
\begin{gathered}D_{\mathrm{KL}}(\pi_\theta\|\pi_{\theta+\Delta\theta})\approx\tfrac12\Delta\theta^\top F\Delta\theta,\\ F=\mathbb E[\psi\psi^\top].\end{gathered}
$$

Fisher 矩阵衡量参数变化引起的局部策略分布变化。期望用哪个状态分布，必须与所研究的局部约束说明一致。

$$
\begin{gathered}\max_{\Delta\theta}g^\top\Delta\theta\quad\text{s.t.}\quad\tfrac12\Delta\theta^\top F\Delta\theta\le\varepsilon\\ \Longrightarrow\quad\Delta\theta\propto F^{-1}g.\end{gathered}
$$

自然梯度来自这个局部线性目标与二次 KL 约束。$F$ 可能奇异，实际常用伪逆或阻尼；有限步长、估计误差和近似求解都影响真实性能。

在单步 Bernoulli bandit 中，$F=p(1-p)$，若两个动作奖励差为 1，普通梯度为 $p(1-p)$，自然梯度为 1。它改变坐标尺度，但并不让任意有限更新都安全，也不保证全局最优。兼容函数逼近还要求 critic 的梯度特征与策略 score 匹配，并满足特定加权拟合条件；任意神经 critic 不自动兼容。

平均奖励策略梯度使用平稳状态权重与差分动作价值。在适当遍历条件下，目标是长期奖励率，不是从一个起点累计的折扣和。因此不能把去掉外层折扣的回合更新直接叫作平均奖励算法；奖励率、差分 critic 与采样协议都需要重新定义。

<a id="lesson-code"></a>

## 8 · 完整枚举与数值梯度代码

高斯 score、Bernoulli REINFORCE、四条轨迹枚举与有限差分

```python
def gaussian_scores(action, mean, log_std):
    """Scores with respect to mean and log standard deviation, before transforms."""
    inverse_variance = math.exp(-2 * log_std)
    error = action - mean
    return error * inverse_variance, error * error * inverse_variance - 1


def sigmoid(theta):
    if theta >= 0:
        return 1 / (1 + math.exp(-theta))
    value = math.exp(theta)
    return value / (1 + value)


def reinforce_gradient(actions, rewards, probability, gamma, baseline=0.0):
    """Bernoulli policy with shared scalar logit. Return gradient, not parameter update."""
    if len(actions) != len(rewards) or not 0 < probability < 1:
        raise ValueError("trajectory sizes or policy support invalid")
    result, return_to_go = 0.0, 0.0
    for t in reversed(range(len(rewards))):
        return_to_go = rewards[t] + gamma * return_to_go
        score = actions[t] - probability
        result += gamma ** t * (return_to_go - baseline) * score
    return result


def exact_policy_gradient(theta, gamma=0.5, baseline=0.0):
    # Two decisions, reward 1 iff first action=1 and second action=0.
    # J(theta)=gamma*p*(1-p); enumerate all four trajectories exactly.
    probability, expectation = sigmoid(theta), 0.0
    for actions in itertools.product((0, 1), repeat=2):
        mass = math.prod(probability if a else 1 - probability for a in actions)
        rewards = [0.0, float(actions == (1, 0))]
        expectation += mass * reinforce_gradient(actions, rewards, probability, gamma, baseline)
    return expectation


def policy_demo():
    theta, gamma, eps = 0.7, 0.5, 1e-5
    objective = lambda z: gamma * sigmoid(z) * (1 - sigmoid(z))
    p = sigmoid(theta)
    return {"probability": p, "J": objective(theta),
            "enumerated_gradient": exact_policy_gradient(theta, gamma),
            "analytic_gradient": gamma * p * (1 - p) * (1 - 2 * p),
            "finite_difference": (objective(theta + eps) - objective(theta - eps)) / (2 * eps),
            "with_baseline_3": exact_policy_gradient(theta, gamma, baseline=3.0),
            "bandit_Fisher": p * (1 - p), "bandit_natural_gradient_reward_gap_1": 1.0}
```

sigmoid 采用分支避免大幅正负输入时溢出。梯度函数要求动作支持在数值上非退化。这个标准库例子没有自动微分，也没有环境模拟误差，所以三种梯度之间的差异能直接定位公式或代码问题。它不是通用连续控制 runner。

运行策略梯度的三重核对

```sh
python3 approximation_textbook_lab.py policy-gradient
python3 approximation_textbook_lab.py test
```

<a id="lesson-branches"></a>

## 9 · 从经典 actor–critic 到深度与持续学习

现代 PPO、SAC 等方法在目标、数据复用和策略约束上作了不同选择。理解它们之前，应能指出一个实现中的采样策略、被优化目标、critic 目标、终止处理以及停止梯度位置。算法名字无法替代这些定义。

持续学习进一步让策略、状态表示和环境同时变化。旧优势估计和旧状态特征可能失效；在线元学习还对学习更新本身求导。此时要分清策略的环境梯度估计与元参数通过学习过程产生的梯度，避免把截断、近似和忽略的依赖隐藏在自动微分图中。

最小研究起点是固定两步环境，通过枚举确认目标和梯度一致，再加入 critic 近似、状态别名、环境变化和计算预算。每次增加一种机制，都保留能够反驳错误实现的解析测试。大环境里的回报提高，不能代替这个正确性检查。

<a id="lesson-check"></a>

## 10 · 练习与答案

- 为什么 $G_t$ 内已折扣，前面还需要 $\gamma^t$？答案：$G_t$ 从当前时刻计时，而目标 J 从回合起点计时，两个时间原点相差 t。
- 减去动作相关基线仍自动无偏吗？答案：不。动作求和不再能提出基线，需要额外校正。
- critic 只用于 MC 基线，与用 TD 替换回报有何不同？答案：前者可只改变方差，后者通常还引入 bootstrap 近似偏差。
- 自然梯度方向为何不保证每一步收益上升？答案：推导使用局部线性目标与局部二次 KL，有限步长和估计误差可能使近似失效。

<a id="chapter-code"></a>

## 下载与运行

标准库教学实现。命令运行本章的可解析小问题和全文件的公式测试；不依赖深度学习库，不等同于论文中的大规模实验。

[下载 approximation_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/approximation_textbook_lab.py)

```sh
python3 approximation_textbook_lab.py policy-gradient
python3 approximation_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：Part II 第 9–13 章。原书建立函数逼近预测、控制、离策略稳定性、资格迹与策略梯度的共同框架。

- [Sutton et al. · Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)：策略梯度与兼容函数逼近的原始论文。区分起点折扣权重和平均奖励平稳分布。

- [Kakade · A Natural Policy Gradient](https://proceedings.neurips.cc/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html)：自然策略梯度的原始工作，连接策略空间的局部度量与兼容逼近。

<a id="study-connections"></a>

## 与教材主线的衔接

- [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)
- [最大熵控制](https://yingwen.io/zh/continual-rl/algorithms/soft-control/)
- [平均奖励与差分价值](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)
- [元学习与学习规则的适应](https://yingwen.io/zh/continual-rl/algorithms/meta/)

对应原始材料：第 13 章：Policy Gradient Methods。本文为原创讲解，原书、论文与上游代码保留各自许可。
