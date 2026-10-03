# 深度价值学习：DQN 与 Double DQN

DQN 用神经网络近似动作价值。在共享参数下，一次更新会影响多个状态；自举目标又依赖当前估计。经验回放与目标网络分别调整数据使用方式和目标变化速度，Double DQN 则分开动作选择与动作评估。

## 本章内容

- 能从 Bellman 最优方程写出 DQN / Double DQN 的 target 与半梯度。
- 实现两层 ReLU Q 网络及其反向传播。
- 知道 buffer、更新比率、目标滞后和非平稳环境之间的冲突。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 一条经验

在 $S_{t}$ 做 $A_{t}$ 后得到 $R_{t+1}$、$S_{t+1}$。奖励属于这次转移，不是到达状态之前另一轮的奖励。

### 折扣与终止

$\gamma$ 控制未来相对权重；真实终止后的价值设为 0。本章记 $\gamma_{t+1}$=$\gamma$(1−terminated)。训练脚本的时间截断不一定是任务终止。

### 表格与函数逼近

表格每个状态或状态动作一组独立参数；函数逼近让不同输入共享参数。更新一个输入可能改变其他输入的预测。

### 链式法则与 stop-gradient

同一个数可以参与前向计算但不参与本次求导。DQN 把本次目标当作固定标签；这不是遗漏反向传播。

<a id="lesson-setting"></a>

## 1 · 学习对象仍然是最优动作价值

先讨论离散动作、固定 MDP、折扣回报。$Q_\theta(s,a)$ 输出每个动作的值。参数共享使一次更新能泛化到未访问输入，但也会改变过去已经正确的值。环境经验来自当前 $\epsilon$-greedy 策略，训练 minibatch 来自 replay 分布，目标又来自另一个滞后的参数快照 $\theta^-$。三种时间尺度同时存在。

| 持久对象 | 存什么 | 何时改 |
| --- | --- | --- |
| 在线网络 $\theta$ | 当前动作价值 | 每次梯度更新 |
| 目标网络 $\theta^-$ | 较慢移动的 target | 每 C 步硬复制，或明确的软更新 |
| replay D | 真实经验及终止语义 | 每步加入、按容量淘汰 |
| 优化器状态 | 动量、二阶矩等 | 每次参数更新；本章用 SGD 便于检查 |

本章不把 replay 称为 CRL 的完整记忆方案。均匀缓冲区回答的是“从现有数据分布抽样”，不是“哪些知识以后仍值得保留”。

<a id="lesson-derive"></a>

## 2 · Bellman target 到半梯度，再到 Double DQN

$$
Y=r+\gamma(1-d)\max_{a\prime}Q_{\theta^-}(s\prime,a\prime),\qquad L(\theta)=\frac1B\sum_{i=1}^B\frac12[Q_\theta(s_i,a_i)-\operatorname{sg}(Y_i)]^2
$$

d 仅表示真实任务终止。目标网络不参与梯度；经验分布也被当作这次优化给定的数据。网络训练的是拟合一批 Bellman target 的代理问题，不是直接最小化全局真实价值误差。

$$
\theta\leftarrow\theta+\frac{\alpha}{B}\sum_i[Y_i-Q_\theta(s_i,a_i)]\nabla_\theta Q_\theta(s_i,a_i)
$$

从平方损失求导即可得到。若使用 Huber loss，大误差区的导数会截到常数幅度；这改变鲁棒性与梯度尺度，不改变 target 定义。本教学实现用半平方误差，便于有限差分。

$$
a^*=\operatorname{argmax}_a Q_\theta(s\prime,a),\qquad Y^{\mathrm{Double}}=r+\gamma(1-d)Q_{\theta^-}(s\prime,a^*)
$$

DQN 同一目标网络既选动作又评估；Double DQN 用在线网络选择、目标网络评估。两网络仍相关，不能声称完全消除高估或保证性能更好。

为什么目标要慢一点？若预测和 target 同时追着彼此跑，当前网络误差会立刻成为新的标签。目标网络暂时固定这层反馈；replay 则减弱相邻经验相关性并重用样本。二者是实践稳定化手段，不是对任意非线性网络收敛的证明。

<a id="dqn-derivative"></a>

## 3 · 两层网络的梯度推导

$$
h=\operatorname{ReLU}(W_1x+b_1),\quad q=W_2h+b_2,\quad e=q_a-Y
$$

只对实际采取动作 a 的输出拟合 target；其余输出的直接梯度为零，但共享隐藏层更新后可能间接改变它们。

$$
\frac{\partial L}{\partial W_{2,a:}}=e h^\top,\quad \frac{\partial L}{\partial b_{2,a}}=e,\quad g_h=eW_{2,a:}^\top,\quad g_z=g_h\odot\mathbf1_{W_1x+b_1>0},\quad \frac{\partial L}{\partial W_1}=g_zx^\top
$$

所有右侧必须使用同一次前向传播的旧参数。若先改 $W_{2}$ 再用新 $W_{2}$ 算 $W_{1}$ 梯度，就不再是所写损失的梯度。ReLU 恰好为 0 不可微；测试避开拐点，用常规子梯度 0。

<a id="lesson-example"></a>

## 4 · 一条样本，三个容易错的 target

当前 Q(s,a)=2，r=1，$\gamma$=.9；下一状态在线输出 [3,2]，目标输出 [1,4]。DQN target=1+.9×4=4.6；Double DQN 在线选动作 0，目标评估为 1，因此 target=1.9。它们给出相反更新方向完全可能。

若此转移真实终止，两个 target 都是 1，且根本不必调用下一状态网络；若只是采样器时间上限，任务本身还能继续，则要从真实最后观测 bootstrap。不能误用自动 reset 后新回合的初始观测。

把 r 改成 10 后，TD error 增大并不自动意味着环境发生了变化：也可能只是首次访问高奖励状态。诊断变化需要固定条件或独立检测设计。

<a id="lesson-code"></a>

## 5 · 核心实现：网络、反向传播、回放与目标同步

**算法：DQN 与 Double DQN；每次运行选择一种目标**

1. 初始化在线网络 $Q_\theta$、目标网络 $Q_{\bar\theta}$ 与回放池 $\mathcal D$。
1. 每个真实步：
  1. 按 $\epsilon$-greedy 策略行动，将 $(s,a,r,s',d)$ 加入 $\mathcal D$。
  1. 抽取小批量转移，并使用更新前的网络计算固定目标：
    1. DQN：$Y=r+\gamma(1-d)\max_bQ_{\bar\theta}(s',b)$
    1. Double DQN：$a^*=\arg\max_bQ_\theta(s',b)$，$Y=r+\gamma(1-d)Q_{\bar\theta}(s',a^*)$
  1. 对 $\tfrac12\operatorname{mean}[(Q_\theta(s,a)-\operatorname{sg}(Y))^2]$ 做一次梯度下降。
  1. 每隔给定的真实步数，复制 $\bar\theta\leftarrow\theta$。
  1. 真实终止时按协议重置环境，保留网络和回放池。

无需深度框架也能运行的两层 ReLU DQN / Double DQN

```python
class TinyQ:
    """Two-layer ReLU Q-network: 2 state features -> 8 hidden -> 2 actions."""
    def __init__(self, rng, hidden=8):
        self.w1 = [[rng.uniform(-0.5, 0.5) for _ in range(2)] for _ in range(hidden)]
        self.b1 = [0.1] * hidden
        self.w2 = [[rng.uniform(-0.2, 0.2) for _ in range(hidden)] for _ in range(2)]
        self.b2 = [0.0, 0.0]

    def forward(self, state):
        # An integer state is a one-hot input; terminal state is never evaluated.
        hidden = [max(0.0, row[state] + b) for row, b in zip(self.w1, self.b1)]
        q = [sum(w * h for w, h in zip(row, hidden)) + b
             for row, b in zip(self.w2, self.b2)]
        return hidden, q

    def gradient(self, state, action, target):
        hidden, q = self.forward(state)
        error = q[action] - target  # gradient of 1/2 * squared error
        g = {"w1": [[0.0] * 2 for _ in hidden], "b1": [0.0] * len(hidden),
             "w2": [[0.0] * len(hidden) for _ in range(2)], "b2": [0.0, 0.0]}
        g["b2"][action] = error
        for j, h in enumerate(hidden):
            g["w2"][action][j] = error * h
            dh = error * self.w2[action][j] * (h > 0.0)
            g["w1"][j][state] = dh
            g["b1"][j] = dh
        return g

    def train_batch(self, frozen_samples, alpha):
        # All gradients use the SAME pre-update parameters.
        gradients = [self.gradient(s, a, y) for s, a, y in frozen_samples]
        for name in ("w1", "w2", "b1", "b2"):
            param = getattr(self, name)
            for i in range(len(param)):
                if isinstance(param[i], list):
                    for j in range(len(param[i])):
                        param[i][j] -= alpha * sum(g[name][i][j] for g in gradients) / len(gradients)
                else:
                    param[i] -= alpha * sum(g[name][i] for g in gradients) / len(gradients)


def dqn_target(online, target, reward, discount, next_state, double):
    if discount == 0.0:
        return reward  # do not even index a terminal observation
    _, target_q = target.forward(next_state)
    if double:
        _, online_q = online.forward(next_state)
        selected = max(range(2), key=lambda a: online_q[a])
        return reward + discount * target_q[selected]
    return reward + discount * max(target_q)


def train_dqn(steps=5000, seed=7, double=True):
    rng = random.Random(seed)
    online = TinyQ(rng)
    target = copy.deepcopy(online)
    replay, state = [], 0
    for t in range(steps):
        _, q = online.forward(state)
        action = rng.randrange(2) if rng.random() < 0.2 else max(range(2), key=lambda a: q[a])
        if state == 0 and action == 0:
            reward, discount, nxt = 0.0, 0.9, 1
        else:
            reward = 0.1 if state == 0 else (1.0 if action == 0 else -1.0)
            discount, nxt = 0.0, None
        replay.append((state, action, reward, discount, nxt))
        replay = replay[-256:]
        if len(replay) >= 16:
            batch = rng.sample(replay, 16)
            frozen = [(s, a, dqn_target(online, target, r, g, sp, double))
                      for s, a, r, g, sp in batch]
            online.train_batch(frozen, alpha=0.03)
        if (t + 1) % 25 == 0:
            target = copy.deepcopy(online)
        state = 0 if discount == 0.0 else nxt
    estimates = [online.forward(s)[1] for s in range(2)]
    reference = [[0.9, 0.1], [1.0, -1.0]]
    return {"Q": estimates, "reference": reference,
            "max_error": max(abs(estimates[s][a] - reference[s][a]) for s in range(2) for a in range(2))}
```

环境只有两个非终止状态。状态 0 动作 0 以奖励 0 到状态 1，动作 1 以 .1 终止；状态 1 两动作分别以 +1/−1 终止。最优 Q 为 [[.9,.1],[1,−1]]。每轮真实交互写入容量 256 的 buffer，抽 16 条经验，先冻结所有 target，再基于同一旧网络求 minibatch 梯度；每 25 个真实步同步目标。

输入用 one-hot 不代表表格算法：八个 ReLU 隐藏单元和两个输出共享参数。实验用中心有限差分检查网络梯度，以解析动作价值检查训练结果。环境只有两个非终止状态，适合观察目标网络、回放和梯度更新的作用。

- 读输出先检查 `max_error`，再改变 double、target 同步频率、buffer 容量。每次只改一项。
- 若加入奖励切换，记录 buffer 样本的年龄和切换前后比例；不要只报告容量。
- 比较更多梯度更新是否有利时，同时报告环境步数与梯度步数。

<a id="lesson-branches"></a>

## 6 · DQN 家族的不同改动在解决什么

| 分支 | 改什么 | 并不自动解决 |
| --- | --- | --- |
| Double DQN | 动作选择与评估耦合 | 持续漂移、可塑性 |
| Dueling network | V 与 advantage 的输出分解 | 任务身份、记忆 |
| Prioritized replay | 采样更高 TD error 的经验并校正权重 | 高误差是否只是噪声 |
| n-step / distributional RL | 目标传播距离 / 回报分布表示 | 任意环境变化 |
| ReDo / resets / continual backprop | 长时间训练后的特征与优化状态 | 旧知识保留与安全 |
| Recurrent DQN | 把历史编码成决策状态 | replay 中隐状态是否过时 |

Rainbow 将若干改动组合，在其任务与预算中评估。研究 CRL 时不应把“使用 Rainbow”当作所有机制已处理：对变化场景，旧缓冲区、目标网络、特征和状态都可能以不同速度过时。

<a id="lesson-check"></a>

## 7 · 怎样知道是实现错还是学习困难？

先检查网络能否过拟合固定 target 的一小批监督样本；再固定目标网络检查 Bellman target；再加入 replay 与环境循环。如果连固定数据回归都失败，就不应把结果解释为 CRL 可塑性损失。

为何 batch size=1 不等于严格 streaming？因为样本仍可能来自旧 replay，并可能对同一经验重复训练。数据权限、重用次数、目标快照和每步延迟都需单独描述。

## 本章的实验设计

分别检查终止处理、目标网络和梯度方向。通过这些测试后，再比较跨种子的回报。

设定：固定小批次：在线后继 Q=(3,2)，目标后继 Q=(4,20)，奖励 1，γ=0.9；另加入真实 terminal 和外部 truncation。

- Double Q 选择动作 0，并得到目标 4.6。
- 目标网络不从该损失收到非预期梯度。
- 自动重置时 bootstrap 使用最终有效观测而非新初态。

对照：普通 DQN 与 Double DQN 的同基座核；相同 replay、网络及更新比；各完整方法的独立合理调参

记录：逐样本 target、TD 残差和梯度；在线/目标参数版本与同步事件；回放访问、梯度步和独立 run 回报

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-deep)

## 学习与研究衔接

DQN 保留 TD 目标。网络、回放与目标网络增加新的时间尺度，长期训练时也可能引入陈旧数据与可塑性问题。

[分册导读](https://yingwen.io/zh/continual-rl/start/deep-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-deep-value) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=deep-value) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=deep-value)

<a id="chapter-code"></a>

## 下载与运行

Python 3.10+，仅标准库。包含两层网络及手写反向传播、replay 与目标网络。实验环境为可解析的小型 MDP。

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

```sh
python3 foundations_detail_lab.py deep-value
python3 foundations_detail_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Mnih et al. · Human-level control through deep RL](https://www.nature.com/articles/nature14236)：DQN 原论文，包含网络结构、输入预处理、经验回放、目标网络与 Atari 训练协议。

- [van Hasselt et al. · Deep RL with Double Q-learning](https://arxiv.org/abs/1509.06461)：追踪选择动作和评估动作分别来自哪套参数。

- [CleanRL · DQN 单文件实现与说明](https://docs.cleanrl.dev/rl-algorithms/dqn/)：可运行的现代复现工程，不是 DQN 原论文作者代码。可对照 replay、final observation、target 同步与训练频率。

- [DeepMind · 原始 DQN 工程](https://github.com/google-deepmind/dqn)：历史作者实现，Torch/Lua 与旧 Atari 依赖；用于核对原方法，运行需要相应历史依赖。
