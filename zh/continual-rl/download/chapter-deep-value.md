# 深度价值学习：DQN 与 Double DQN

把 DQN 放进长期运行的学习器后，哪些量只是为一次更新而固定，哪些旧经验、目标和特征会继续影响未来行动？

## 本章内容

- 核对 DQN / Double DQN 的标签、梯度与目标副本各在什么时刻固定。
- 沿两层网络和既有小任务，区分回归检查、闭环学习与变化后诊断。
- 说明 replay、目标滞后、特征干扰与生命期收益之间还缺哪些比较。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [深度价值学习：DQN、Double DQN 与目标的时间顺序](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/)：完整DQN训练循环在第二册；这里回顾其持续使用条件。


### 一条经验

在 $S_{t}$ 做 $A_{t}$ 后得到 $R_{t+1}$、$S_{t+1}$。奖励属于这次转移，不是到达状态之前另一轮的奖励。

### 折扣与终止

$\gamma$ 控制未来相对权重；真实终止后的价值设为 0。本章记 $\gamma_{t+1}$=$\gamma$(1−terminated)。训练脚本的时间截断不一定是任务终止。

### 表格与函数逼近

表格每个状态或状态动作一组独立参数；函数逼近让不同输入共享参数。更新一个输入可能改变其他输入的预测。

### 链式法则与 stop-gradient

同一个数可以参与前向计算但不参与本次求导。DQN 把本次目标当作固定标签；这不是遗漏反向传播。

<a id="problem-definition"></a>

## 本章的问题定义

离散动作折扣控制中，以共享神经网络估计动作价值；数据行为、回放分布和目标网络具有不同时间尺度。

### 给定条件与符号

- 固定Markov任务、奖励、真实终止语义和可选离散动作。
- 网络、回放容量、采样规则、目标同步、探索及更新预算。

### 需要求解的对象

可产生高回报动作的近似最优动作价值；每批训练只拟合给定Bellman标签。

### 信息与数据权限

行为策略收集 $(s,a,r,s',d)$，$d$ 只表示真实终止；回放分布 $D$ 决定本批样本，在线参数 $\theta$ 与目标参数 $\theta^-$ 的更新时间各自规定。

$$
Q^*(s,a)=\mathbb E\!\left[R+\gamma(1-d)\max_{a'}Q^*(S',a')\mid s,a\right]
$$

$R,S'$ 是真实条件后果，$0\le\gamma<1$。这是理想最优价值固定点；本批DQN损失是 $\tfrac12\mathbb E_D[(Q_\theta(s,a)-\operatorname{sg}(Y))^2]$，$Y$ 为旧目标网络构造的标签，$\operatorname{sg}$ 表示停止梯度。二者不是同一个优化问题。

### 成立条件与解的含义

- 基础题目固定且状态Markov；数据必须覆盖决策所需动作与状态。
- 任意非线性网络、回放与自举的组合没有本章给出的全局收敛保证。

判断准则：两状态解析问题上核对最优价值[[0.9,0.1],[1,−1]]、终止标签和网络梯度；复杂任务用独立行为收益并记录真实步、梯度步和回放年龄。

### 适用边界

- Double DQN不保证完全消除高估或总有更高回报。
- batch size为1不构成严格流式协议。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 限制表示或采用近似 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：DQN解决固定离散折扣任务的局部价值控制，并不完整评价持续学习过程。

- 组合不同学习问题 · [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)：回放重用经验，但是否保留未来需要的旧知识还需历史采样与回访评价。

- 组合不同学习问题 · [可塑性与特征更新](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)：共享网络长期可学习性是额外问题，较低Bellman标签损失不能诊断全部退化。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

共享估计同时改变多个输入，目标又依赖价值估计，产生相关样本与追逐标签的反馈。

### 本章的核心思路

分别控制数据重用、目标移动和选择—评估耦合，不把三个机制混成一项收敛保证。

1. [从最优固定点构造冻结标签](#lesson-derive)：因为网络不能直接枚举真实期望，用目标网络生成本批Bellman标签并停止其梯度。

2. [分开动作选择和评估](#lesson-derive)：因为最大值会偏爱估计偏高的动作，Double DQN用在线网络选、目标网络评估；两网络仍可能相关。

3. [验证共享梯度与三种时钟](#lesson-code)：因为一处参数更新影响多个输出，先检查固定标签梯度，再接回放、真实交互和目标同步循环。

结论与条件：精确有限折扣Bellman算子有唯一固定点；这不构成神经DQN训练的收敛证明，目标网络与回放是有限协议下的稳定化机制。

### 相关方法改变了什么

- 表格Q-learning：独立参数消除共享逼近干扰，但不能扩展到任意高维输入。

- DQN：目标网络同时选与评估下一动作。

- Double DQN：分开选择与评估来源，减少一类最大化偏差而不消除所有误差。


<a id="lesson-setting"></a>

## 1 · 从训练循环追踪学习器的持久状态

第二册的 [DQN 与目标的时间顺序](/zh/continual-rl/foundations/deep/deep-value/)完成 Bellman 目标、共享特征、Double DQN 与完整训练循环的推导。本章回顾其中会跨步保存的量，随后用它们定位持续学习中的问题：旧经验是否仍对应当前后果，目标是否滞后，网络是否还学得动，以及这些差异怎样改变下一次行动。

![真实交互、经验回放、在线 Q 网络和目标网络的分离，以及一次更新的梯度边界。](https://yingwen.io/crl-figures/concept-classic-dqn-dataflow.svg)

沿图检查采样、梯度更新、目标复制三种时钟。图中独立算例给定目标后继值 (2,6)、奖励 1、折扣 0.9，故 DQN 标签为 6.4；画面中的迷宫不是这些值的训练来源。后文另一组给定输出用于对比 Double DQN，不与本图数值混用。

先保留离散动作、固定 MDP 和折扣目标。$Q_\theta(s,a)$ 的参数持续更新，$Q^*$ 仍是同一个数学对象。经验来自当前 $\epsilon$-greedy 策略，训练 batch 来自 replay，bootstrap 使用滞后快照 $\theta^-$。行为分布、训练分布与标签在变化，本身不表示环境规律变了。

| 持久对象 | 存什么 | 何时改 |
| --- | --- | --- |
| 在线网络 $\theta$ | 当前动作价值 | 每次梯度更新 |
| 目标网络 $\theta^-$ | bootstrap 的价值副本 | 每 C 步硬复制，或明确的软更新；须声明采用哪种时钟 |
| replay D | 真实经验及终止语义 | 每步加入、按容量淘汰 |
| 优化器状态 | 动量、二阶矩等 | 每次参数更新；本章用 SGD 便于检查 |

这些量连同探索计数器决定学习器之后怎样行动和学习。只保存在线网络，可以重现某次前向预测，却未必能续上同一个训练过程。均匀 replay 规定怎样抽取已存数据；决定保留哪些历史，进一步取决于未来用途与内存预算。

<a id="lesson-derive"></a>

## 2 · 一次回归固定了什么，下一次又会改变什么

$$
Y=r+\gamma(1-d)\max_{a\prime}Q_{\theta^-}(s\prime,a\prime),\qquad L(\theta)=\frac1B\sum_{i=1}^B\frac12[Q_\theta(s_i,a_i)-\operatorname{sg}(Y_i)]^2
$$

d 仅表示真实任务终止。目标网络不参与梯度；经验分布也被当作这次优化给定的数据。网络训练的是拟合一批 Bellman target 的代理问题，不是直接最小化全局真实价值误差。

$$
\theta\leftarrow\theta+\frac{\alpha}{B}\sum_i[Y_i-Q_\theta(s_i,a_i)]\nabla_\theta Q_\theta(s_i,a_i)
$$

固定这批输入与标签后，这是回归损失对在线参数的完整梯度。放回整个自举过程看，它采用不沿后继估计求导的半梯度原则。Huber loss 保留 target 定义，但改变误差导数，可能连最优拟合值也改变；本教学实现用半平方误差，便于有限差分。

$$
a^*=\operatorname{argmax}_a Q_\theta(s\prime,a),\qquad Y^{\mathrm{Double}}=r+\gamma(1-d)Q_{\theta^-}(s\prime,a^*)
$$

DQN 同一目标网络既选动作又评估；Double DQN 用在线网络选择、目标网络评估。两网络仍相关，不能声称完全消除高估或保证性能更好。

对一条固定旧转移，若目标网络的输入处理与前向计算也固定，普通 DQN 的标签在两次目标复制之间不变。Double DQN 的在线 argmax 却可能在下一次更新后换动作，因而即使目标副本未动，重新计算的标签也会变化。停止梯度只固定本次求导责任；冻结副本只固定指定参数的演化。两者都没有冻结整个学习器。

Replay 重用经验并改变训练样本的混合，目标副本减慢部分自举反馈。第二册的[共享特征与旧经验重标记](/zh/continual-rl/foundations/deep/deep-value/#lesson-shared-feature)已经逐项算出这种联动。这里继续保留更新和网络导数，供检查持久状态的时序；这些稳定机制的存在仍不足以保证任意非线性网络收敛。

<a id="experiment-deep-double_dqn"></a>

### 实验：实验 · 分开选动作与评价动作的 Double DQN

同一网络、回放和目标同步下，更换 bootstrap 的动作选择方式会怎样？

**环境与可用信息。** DeadlineChain：位置 0–4、左右动作、边界截断。观测为五维位置 one-hot 加剩余时间比例。到位置 4 得 1 并终止；其他步得 −0.02。12 步截止也是任务真实终止，且剩余时间可观测。每回合重新从位置 0 开始，网络跨回合保留。

**设置。** 1200 个真实步。两者用 6→32 tanh→2 网络、相同 seed 的 PyTorch 默认随机初始化和复制的目标网络；Adam 0.003，γ=0.99，Huber 损失，梯度范数上限 10。replay 容量 4000、批量 32，每两步一次更新，每 100 步硬同步。ε 从 1 降到最低 0.05。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** double_dqn.py 用在线网络选择后继动作，用目标网络评价该动作；dqn.py 对目标网络直接取最大。两者都停止整个 critic target 的梯度。

**测量。** 每 60 步冻结网络，在隔离环境执行 12 个贪心回合。图中是未折扣环境回报；训练 target 使用 γ=0.99。此图没有直接测量 Q 高估偏差。

```bash
python3 implementations/deep/double_dqn.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-figures/result-deep-double_dqn.svg)

横轴：训练环境步（独立评估交互另计）。纵轴：冻结策略的外部回报。每种方法 1200 训练环境步；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 冻结当前网络，以 argmax 选择动作，在独立 DeadlineChain 中跑12次完整回合，取未折扣外部回报的平均。这里起点相同且评价过程确定，12次并非12个独立训练种子。

**step：怎样计时。** step 只数训练环境转移。评价交互另计；任务中的12步期限是真正终止条件。

**怎样汇总。** 取该记录时刻的值；不先对曲线上的时间点求平均。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** CSV 的 phase=evaluation。可以重算冻结评估曲线；未保存逐步训练奖励，不能从这些评估点反推训练全程收益。

计算位置：[deep/dqn.py](https://yingwen.io/crl-code/implementations/deep/dqn.py) · [deep/double_dqn.py](https://yingwen.io/crl-code/implementations/deep/double_dqn.py) · [deep/_common.py](https://yingwen.io/crl-code/implementations/deep/_common.py)

</details>

**结果分析。** 第 600 步两者平均冻结回报都是 0.94；第 1200 步 Double DQN 为 0.94，DQN 为 0.936。短链上两者几乎打平，这个小差异不能证明普遍优势或已解决高估。

**结论边界。** 任务是可观测有限时域小链，不是 Atari。网络是 tanh MLP，不是卷积网络；模型大小、回放次数及更新比固定。真正的 deadline 终止与外部时间截断不能混用。

**继续实验。** 先对同一小批次打印选中动作和两种 target，再在相同模型下测对解析 Q 的误差。只比较环境回报，能否判断差异来自高估？

[源码](https://yingwen.io/crl-code/implementations/deep/double_dqn.py) · [逐种子记录](https://yingwen.io/crl-code/results/deep-double_dqn/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/deep-double_dqn/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/deep-double_dqn/curves.json)

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

## 6 · 从 DQN 的内部状态定位持续学习问题

| 分支 | 改什么 | 并不自动解决 |
| --- | --- | --- |
| Double DQN | 动作选择与评估耦合 | 持续漂移、可塑性 |
| Dueling network | V 与 advantage 的输出分解 | 任务身份、记忆 |
| Prioritized replay | 采样更高 TD error 的经验并校正权重 | 高误差是否只是噪声 |
| n-step / distributional RL | 目标传播距离 / 回报分布表示 | 任意环境变化 |
| ReDo / resets / continual backprop | 长时间训练后的特征与优化状态 | 旧知识保留与安全 |
| Recurrent DQN | 把历史编码成决策状态 | replay 中隐状态是否过时 |

这些改动可以组合，Rainbow 就在其任务与预算中评估了其中若干项。持续运行时还要定位变化的来源：旧转移可能来自已经改变的世界，目标副本可能落后于当前估计，网络即使面对固定新标签也可能学得很慢。这三种情况分别涉及数据时效、目标同步与可训练能力，单看 TD loss 无法区分。

本章的小型固定 MDP 与梯度检查先提供可知答案。[知识保留](/zh/continual-rl/algorithms/retention/)进一步问保存哪些经验或功能，[可塑性](/zh/continual-rl/algorithms/plasticity/)问旧参数化还能否学习新目标。比较它们对控制的作用时，应从同一协议初始化完整学习器，继续采样与更新，把适应期低谷、旧能力诊断和总收益分别记录；仅比较最终 Q 网络遗漏了学习过程。

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

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-deep-value) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=deep-value) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=deep-value)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

给定状态后可以估计价值；观测不足时，还要学习保留哪些历史。GVF 规定预测什么，RTRL 计算递归敏感度，资格迹组织时间信用。应分别检验信息是否进入状态、反馈能否教会这种保留，以及有限预测预算怎样分配，而不是把三者当作替代算法。

- [Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations](https://yingwen.io/zh/continual-rl/research/#recent-successor-flow-features)

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [Safe and Efficient Off-Policy Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-retrace-safe-offpolicy)
- [Convergent Tree Backup and Retrace with Function Approximation](https://yingwen.io/zh/continual-rl/research/#recent-convergent-tree-retrace)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

给定模型可研究怎样规划；模型也在学习时，规划会选择性地查询误差，并改变以后的数据。Dreamer、STOMP 和 DRAGO 分别研究想象控制、随机时长行为模型和旧知识保留。新的比较应固定规划查询与总预算，检验哪些后果误差真正改变选择，哪些维护值得继续。

- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://yingwen.io/zh/continual-rl/research/#recent-tdmpc2-decision-time-model)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Revisiting Adam for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-revisiting-streaming-adam)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [The Dormant Neuron Phenomenon in Deep Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-redo-dormant-neurons)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [Revisiting Adam for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-revisiting-streaming-adam)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文规定对象与条件，架构路线提出组织方式，算法实验检验局部机制。撤掉阶段间冻结后，一个模块会改变另一个模块的学习问题；有限预算应优先维护哪条知识，成为新的决策。先检验两模块反馈和资源分配，再扩大整机，而不是由组件分别有效推断长期组合收益。

- [Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations](https://yingwen.io/zh/continual-rl/research/#recent-successor-flow-features)

### The Dormant Neuron Phenomenon in Deep Reinforcement Learning

Ghada Sokar, Rishabh Agarwal, Pablo Samuel Castro, Utku Evci

ICML 2023 · 2023 · 支持方法与理论

#### 研究问题

网络参数数量没有变，为什么越来越多隐藏单元不再对输出产生有效贡献？

#### 关键机制

ReDo 用相对激活量识别低活跃单元，重新初始化其输入连接，并处理输出连接，使被回收单元可以重新参与学习。它针对的是可用表示容量，而不是直接惩罚旧任务表现变化。

#### 证据

论文记录深度 RL 中的休眠单元现象，并比较回收机制对多个任务学习的影响。实现进入作者所在团队的 Dopamine 代码库。

#### 条件与限制

低激活只是可塑性问题的一种诊断，不能覆盖曲率变化、优化器状态和负迁移。回收也可能损坏低频但重要的旧知识，需要与保留指标共同评价。

#### 阅读与实验

同时记录休眠比例、新目标拟合速度与旧任务冻结表现。三者发生不同方向变化时，不要用单个表示指标替代整个持续学习结论。

#### 原文与相关入口

- [ICML 2023 原文](https://proceedings.mlr.press/v202/sokar23a.html)：休眠定义、回收规则与实验。
- [Dopamine ReDo 实现](https://github.com/google/dopamine/tree/master/dopamine/labs/redo)：作者团队公开代码中的 ReDo 模块。

#### 作者代码

[论文作者团队发布的实现，不是本教材的简化版本。](https://github.com/google/dopamine/tree/master/dopamine/labs/redo)

Dopamine 中的 ReDo 神经元回收与实验实现。

### Revisiting Adam for Streaming Reinforcement Learning

Florin Gogianu, Luțu Adrian-Cătălin, Razvan Pascanu

RLC 2026 / RLJ 预会议版 · 2026 · 支持方法与理论

#### 研究问题

流式 RL 的不稳定来自 Adam 本身，还是目标导数、方差与超参数的组合？

#### 关键机制

论文重新分析自适应更新的信噪比，将 Adam 的稳定项与目标导数尺度联系起来，并研究有界导数的回报分布学习及多步更新。它改变的是目标与更新的配合，而非简单沿用批量训练时的默认配置。

#### 证据

作者在大规模 Atari 流式实验中展示了具有竞争力的结果，并重新比较早期流式方法。正式 RLJ 入口收录为 RLC 2026 预会议论文。

#### 条件与限制

主体实验采用经典回合式 Atari 的流式学习协议，不是任意非平稳终生适应的证据。这些结果也不否定归一化、资格迹或更新约束在其他任务中的价值。版本、调参预算和目标分布必须对齐。

#### 阅读与实验

建立二维对照：固定目标换优化器，固定优化器换目标。将调参种子与最终测试分开，再判断改进来自哪一个因素。

#### 原文与相关入口

- [RLC 2026 论文入口](https://rlj.cs.umass.edu/2026/papers/Paper131.html)：会议收录信息与论文。
- [作者预印本](https://arxiv.org/abs/2605.06764)：Adam 尺度分析、回报分布目标与实验协议。

### Safe and Efficient Off-Policy Reinforcement Learning

Rémi Munos, Tom Stepleton, Anna Harutyunyan, Marc G. Bellemare

NeurIPS 2016 · 2016 · 支持方法与理论

#### 研究问题

目标与行为策略不一致时，如何保留多步信用而避免重要性比率乘积爆炸？

#### 关键机制

统一多步目标为目标策略TD误差的加权和，Retrace采用λmin(1,π/μ)传播系数。近同策略时保留长迹，目标概率较低的动作则减少传播；一步误差仍使用目标动作期望。

#### 证据

论文分析表格算子的收缩性质，给出条件下的评价与控制收敛，并报告Atari实验。信用章独立检查传播系数和有限轨迹恒等式。

#### 条件与限制

表格安全性不是任意线性或神经逼近的稳定性保证。行为覆盖、变化策略与投影条件仍需检查；代码小实验不复现Atari。

#### 阅读与实验

在同样轨迹与表示上，分别改变策略差异和动作随机性，比较Tree-backup与Retrace的信用长度、方差和预测误差。

#### 原文与相关入口

- [原论文](https://arxiv.org/html/1606.02647)：统一算子、传播系数及理论条件。

### Convergent Tree Backup and Retrace with Function Approximation

Ahmed Touati, Pierre-Luc Bacon, Doina Precup, Pascal Vincent

ICML 2018 · 2018 · 支持方法与理论

#### 研究问题

传播系数已经截断，为什么函数逼近下的Tree-backup和Retrace仍可能发散？

#### 关键机制

分析函数逼近与off-policy多步bootstrap的学习算子，展示线性反例，再把相应目标写成二次凸凹鞍点问题，构造梯度版本。

#### 证据

原文给出线性不稳定例子、梯度方法收敛保证与有限样本界。它直接限定了从Retrace表格结论外推到逼近算法的范围。

#### 条件与限制

凸凹线性问题的保证不能自动覆盖学习表示的深度网络。稳定目标、更新速度与控制性能还需分别验证。

#### 阅读与实验

先检查固定表示下的期望更新矩阵，再将半梯度和梯度版本按相同样本、步数与计算预算比较。

#### 原文与相关入口

- [ICML原文](https://proceedings.mlr.press/v80/touati18a.html)：理论反例、鞍点方法和保证条件。

### Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations

Haosen Shi, Jianda Chen, Sinno Jialin Pan

ICLR 2026 · 2026 · 支持方法与理论

#### 研究问题

能否直接学习多步未来状态的分布，并把它压缩为适合控制学习的特征？

#### 关键机制

SF² 以 flow matching 估计 successor measure，将条件向量场分解为未来位置及生成时间的投影与当前状态动作特征的乘积。特征进入 TD3/SAC 的 critic；线性的是向量场对条件特征的分解，critic 本身可以非线性。

#### 证据

正式原文给出 mixture Bellman 结构、生成式 bootstrap 与控制实验，并提供作者 JAX/Brax 仓库。实验研究在线收集数据下的 off-policy 控制，并使用 replay、批次与目标网络。

#### 条件与限制

“online policy learning”不代表 strict streaming。生成时间不是环境时间；向量场线性不保证任意奖励价值线性。文中与 SR 的小生成时间联系是近似动机，未证明递归 agent state 或任意持续变化下的充分性。

#### 阅读与实验

对齐模型调用与梯度预算，拆分直接预测、bootstrap、critic 联合训练。冻结特征后比较线性与非线性读出，再测新奖励和动力学变化，才能检验预测知识的可复用程度。

#### 原文与相关入口

- [ICLR 2026 正式原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48acf4b231771e693f42305b4c9b4c9f-Abstract-Conference.html)：第 2–3 节和算法附录；区分 flow 时间、环境时间与近似 SR 联系。
- [原文链接的作者实现](https://github.com/Shiien/successor-flow-representation-implementation)：SAC/TD3、flow 特征、对照和 sweep 配置。

#### 作者代码

[正式论文摘要直接链接的作者代码。](https://github.com/Shiien/successor-flow-representation-implementation)

基于 JAX/Brax 的 SF² 控制实验；不包含自动 GVF 问题发现或完整持续架构。

### TD-MPC2: Scalable, Robust World Models for Continuous Control

Nicklas Hansen, Hao Su, Xiaolong Wang

ICLR 2024 · 2024 · 支持方法与理论

#### 研究问题

如何让短期动力学与长期价值分工，并在动作选择时继续使用模型？

#### 关键机制

TD-MPC2 学习无需观测 decoder 的潜在动力学、奖励、价值与策略先验。决策时优化有限动作序列，用终点价值补上未展开的后果；执行第一步后，利用新观测重新规划。

#### 证据

正式会议原文报告 104 个在线任务和单一大型多任务智能体的实验。官方仓库包含模型训练与计划接口，适合与 Dreamer 的想象 actor 学习比较计算位置。

#### 条件与限制

跨任务共享超参数和多任务能力不是单条生命流中持续适应的证据。replay、任务条件、模型更新、决策延迟等成本需进入 CRL 协议；长程 critic 错误不能被短期模型精度自动修复。

#### 阅读与实验

固定模型，对比无终点价值、不同 horizon 和不同规划预算；再固定预算比较部署 actor 与决策时搜索。环境变化后同时记录模型校准、critic 误差和恢复收益。

#### 原文与相关入口

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html)：短期预测、终点价值、多任务协议。
- [作者实现](https://github.com/nicklashansen/tdmpc2)：训练、模型与 plan 函数分别阅读。

#### 作者代码

[作者维护的原论文代码。](https://github.com/nicklashansen/tdmpc2)

TD-MPC2 的单任务/多任务训练和决策时规划。


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
