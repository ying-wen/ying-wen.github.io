# 特征、泛化与半梯度控制

特征怎样改变学习行为，Sarsa 又怎样在共享参数下改善策略？

## 本章内容

- 比较局部 tile coding 与全局 Fourier 特征的泛化。
- 推导动作分块表示和半梯度 Sarsa，说明动作采样与更新顺序。
- 手算特征缩放的步长效应，并辨别预测保证与控制保证。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 动作价值

$q(s,a)$ 预测先执行指定动作，之后遵循当前策略的回报。用状态—动作特征替代预测章的状态特征。

$$
\hat q(s,a,w)=x(s,a)^\top w
$$

### 半梯度

更新只对当前预测求导。下一状态的 bootstrap 目标也依赖权重，但在这一步视为固定。

<a id="lesson-setting"></a>

## 1 · 控制中的表示和数据相互影响

在预测中，行为策略可以固定。在控制中，权重决定动作价值，动作价值影响策略，策略又决定下一个训练样本。特征不仅影响怎样拟合数据，还会通过行为改变未来看到的数据。某个表示在一个固定数据集上误差小，不足以保证它适合在线控制。

给定状态特征 $\phi(s)\in\mathbb R^d$ 和有限动作数 $K$，一种清楚的起点是动作分块：为每个动作保存独立的 $d$ 个权重。总参数数是 $Kd$。这避免不同动作仅因共享同一特征权重而被强制具有相同价值。

$$
x(s,a)=e_a\otimes\phi(s),\qquad \hat q(s,a)=w_a^\top\phi(s).
$$

$e_a$ 是动作 one-hot 向量。状态间仍共享参数，动作间则由不同参数块区分。神经网络可以选择共享主干，但那引入了另一种跨动作泛化。

<a id="features-local"></a>

## 2 · Tile coding：局部相似性如何进入更新

Tile coding 使用多套错开的网格。一个状态在每套网格激活一个 tile。每个 tile 对应一个二值特征，预测等于所有激活权重的和。邻近状态共享部分 tile，因此一次更新会改变局部邻居的预测；跨越边界时变化不是同时发生的。

$$
x_{k,j}(s)=\mathbf1\{j=\lfloor s/h+k/K\rfloor\},\quad k=0,\ldots,K-1.
$$

这是教学代码的一维版本。$h$ 是格宽，$K$ 是 tiling 数；特征键包含 tiling 编号，避免把不同网格中同编号的格子合并。

若没有哈希碰撞，每次恰有 $K$ 个激活特征，单次半梯度更新让当前预测改变 $\alpha K\delta$。因此常用 $\alpha=\eta/K$，让有效更新比例接近 $\eta$。这来自特征范数，不是一条适用于所有表示的步长定律。

多维观测可以联合划分或分组划分。联合网格表达变量交互，但组合数增长很快；分组特征省内存，却可能无法表示关键交互。哈希限制参数容量，但碰撞会把无关状态绑定在一起。记录容量、激活数和碰撞策略，才能解释性能差异。

<a id="features-global"></a>

## 3 · Fourier 特征、缩放与条件数

$$
\phi_c(s)=\cos(\pi c^\top\tilde s),\quad \tilde s\in[0,1]^n,\quad c_i\in\{0,\ldots,m\}.
$$

Fourier 基用不同频率表达平滑到快速变化的形状。最高阶 $m$、维数 $n$ 下，完整张量基有 $(m+1)^n$ 项；提高阶数增加表达能力，也增加计算与数据需求。

一维代码在状态 0.5、阶数 3 时得到近似向量 [1,0,-1,0]。数值中的极小非零项来自浮点余弦，不是新的特征。Fourier 特征通常全局非零，因此一次更新会影响较远状态。局部表示与全局表示没有统一优胜者，关键是环境中相似状态是否确实应有相似价值。

$$
\begin{gathered}x'=cx,\qquad w'=w/c,\\ \Delta\hat v=\alpha\delta\|x\|^2,\qquad \alpha'=\alpha/c^2.\end{gathered}
$$

统一放大全部特征并相应缩小权重，初始预测不变；若步长不变，预测更新会放大 $c^2$ 倍。线性 TD 在同时调整步长后才保持对应的预测轨迹。

不同坐标的尺度差异会使特征协方差病态。逐维标准化或逐维步长能改善数值尺度，但改变在线归一化统计量也会改变同一原始状态的特征。这种表示漂移在持续学习中尤其重要：旧权重可能不再表示原来的函数。

<a id="lesson-derive"></a>

## 4 · 从策略评价到半梯度 Sarsa

$$
\begin{aligned}\delta_t&=R_{t+1}+\gamma\hat q(S_{t+1},A_{t+1},w_t)-\hat q(S_t,A_t,w_t),\\ w_{t+1}&=w_t+\alpha\delta_t x(S_t,A_t).\end{aligned}
$$

下一动作 $A_{t+1}$ 来自实际行为策略。因此目标在评价该行为，包括它的探索动作。真正终止时 bootstrap 项为零。

常用行为是 $\epsilon$-greedy：以 $1-\epsilon$ 的概率在最大价值动作间均匀选择，以 $\epsilon$ 的概率在全部动作间均匀选择。最大值并列时如何处理也是算法定义的一部分。先按旧权重采样下一动作，再更新，并在下一次交互真正执行这个动作，就得到一致的 Sarsa 时序。

**算法：半梯度 Sarsa 的采样与更新**

1. 初始化动作价值权重，观察状态并按当前策略采样动作
1. 执行动作，观察奖励、下一状态与是否真正终止
1. 若未终止，按更新前的策略采样下一动作并保存其价值
1. 计算旧参数下的 TD 误差，更新当前状态—动作特征的权重
1. 把已采样的下一动作作为下一步实际动作；终止则开始新回合

Expected Sarsa 将下一动作样本替换为当前策略下的价值期望，降低这部分采样方差。将其替换为最大动作价值则得到 Q-learning 风格目标，目标策略与探索行为分离。共享参数时，这个改变引入离策略稳定性问题，不能仅视为低方差替换。

<a id="lesson-example"></a>

## 5 · 手算一个动作块的更新

两个动作，各有特征 [1,s]。当前状态 $s=0.5$，当前动作 0；下一状态 $s'=1$，已采样下一动作 1。权重为 $w=(0.2,0.4,0.1,0.2)$，奖励为 1，$\gamma=0.9,\alpha=0.1$。

$$
\begin{gathered}x=(1,0.5,0,0),\qquad x'=(0,0,1,1),\\ q=0.4,\qquad q'=0.3,\\ \delta=1+0.9(0.3)-0.4=0.87.\end{gathered}
$$

当前和下一价值都在更新前计算。

$$
w^+=(0.287,0.4435,0.1,0.2).
$$

只有动作 0 的参数块被更新，但该动作在其他状态的价值也会改变。动作 1 的本次预测不变。

如果这一步真正终止，目标应该是 1，误差变成 0.6。若只是为了日志而切开一段持续交互，环境仍会继续，则不应自动清除 bootstrap。算法边界来自任务语义，不来自数组或日志文件的边界。

<a id="lesson-code"></a>

## 6 · 表示与 Sarsa 的可运行代码

无哈希的一维 tile coding、Fourier 基、动作分块与 Sarsa

```python
def tiles_1d(state, width=0.25, tilings=4):
    """Unhashed sparse keys: one active interval in each offset tiling."""
    if width <= 0 or tilings < 1 or not math.isfinite(state):
        raise ValueError("finite state, positive width and tilings required")
    return tuple((k, math.floor(state / width + k / tilings))
                 for k in range(tilings))


def fourier_1d(state, order=3):
    if not 0 <= state <= 1 or order < 0:
        raise ValueError("normalize state to [0, 1], use nonnegative order")
    return [math.cos(math.pi * k * state) for k in range(order + 1)]


def action_features(features, action, actions):
    if not 0 <= action < actions:
        raise ValueError("action outside action set")
    return [v if a == action else 0.0 for a in range(actions) for v in features]


def semi_gradient_sarsa(weights, x, reward, xp, gamma, alpha, terminated=False):
    # x and xp are state-action features. xp belongs to the already sampled A'.
    if terminated:
        xp = [0.0] * len(weights)
    return semi_gradient_td(weights, x, reward, xp, gamma, alpha)


def features_demo():
    x = action_features([1.0, 0.5], 0, 2)
    xp = action_features([1.0, 1.0], 1, 2)
    weights, delta = semi_gradient_sarsa([0.2, 0.4, 0.1, 0.2], x, 1, xp, 0.9, 0.1)
    return {"tiles_at_0.2": tiles_1d(0.2), "tiles_at_0.21": tiles_1d(0.21),
            "Fourier_at_0.5": fourier_1d(0.5), "delta": delta,
            "updated_weights": weights}
```

教学版 tiles 返回稀疏键，不分配无限长向量。实际实现可以使用稀疏字典或固定容量哈希表。代码不替你选择传感器范围；Fourier 函数要求输入已在 [0,1]，越界时明确报错。学习器使用未来统计量做归一化会泄露信息，因此部署版应规定训练或在线统计的更新时序。

运行特征与控制的手算例子

```sh
python3 approximation_textbook_lab.py features-control
python3 approximation_textbook_lab.py test
```

<a id="lesson-branches"></a>

## 7 · 控制保证、表示实验与 CRL

线性在策略预测的收敛结论不能直接证明线性 Sarsa 控制全局收敛。原因是策略和状态分布随参数改变，近似误差还可能改变贪心动作。表格 GLIE 条件下的结论也不自动适用于共享参数。实际研究应报告行为稳定性，而不只检查预测损失。

比较特征时固定交互预算、探索策略和超参搜索预算。记录激活特征数、总参数数、每步计算量以及对未训练区域的预测变化。固定参数数量并不等于固定更新尺度；固定名义步长也不等于公平。

进入持续学习后，可以分别让奖励变化、访问区域变化和传感器尺度变化。这样才能判断失效来自旧表示不充分、归一化漂移，还是控制回路把智能体带到低覆盖区域。Agent state、流式归一化和可塑性方法分别处理其中不同部分。

<a id="lesson-check"></a>

## 8 · 练习与答案

- 8 个二值 tile 同时激活，想使一次更新的当前预测变化为 $0.2\delta$，步长取多少？答案：无碰撞时取 $0.2/8=0.025$。
- Fourier 阶数增加，能否保证控制收益提高？答案：不能。更高表达能力也增加估计难度，且价值误差与行为收益不同。
- 将所有特征乘 10，只把权重除 10，为什么仍可能发散？答案：同名义步长下，预测变化会放大 100 倍；还应将步长除以 100。
- Sarsa 的下一动作采样后又重新采样执行，会有什么问题？答案：更新评价的动作与实际后继动作不再是同一条 Sarsa 转移；必须明确改用其他估计器。

<a id="chapter-code"></a>

## 下载与运行

标准库教学实现。命令运行本章的可解析小问题和全文件的公式测试；不依赖深度学习库，不等同于论文中的大规模实验。

[下载 approximation_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/approximation_textbook_lab.py)

```sh
python3 approximation_textbook_lab.py features-control
python3 approximation_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：Part II 第 9–13 章。原书建立函数逼近预测、控制、离策略稳定性、资格迹与策略梯度的共同框架。

- [Sutton · Tile Coding Software](http://incompleteideas.net/tiles/tiles3.html)：原作者的 tile coding 使用说明和代码入口。实际使用时需记录哈希容量、tiling 数与缩放。

- [Konidaris, Osentoski & Thomas · Value Function Approximation in RL Using the Fourier Basis](https://people.cs.umass.edu/~pthomas/papers/Konidaris2011a.pdf)：Fourier 基的原始论文；解释频率表示与强化学习中的使用。

<a id="study-connections"></a>

## 与教材主线的衔接

- [控制问题与广义策略迭代](https://yingwen.io/zh/continual-rl/algorithms/control/)
- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)
- [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

对应原始材料：第 9 章：线性方法、特征构造与步长；第 10 章：On-policy Control with Approximation。本文为原创讲解，原书、论文与上游代码保留各自许可。
