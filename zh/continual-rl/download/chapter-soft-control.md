# 最大熵控制与 Soft Actor–Critic

持续运行 SAC 时，温度改变的是哪个优化问题？怎样区分 soft 价值、外部任务收益，以及长期学习中保留随机性的实际作用？

## 本章内容

- 从给定 Q 的 softmax 改善，区分固定温度目标与自动温度的约束目标。
- 核对 critic、actor、温度三类更新各自的固定量和梯度路径。
- 沿既有离散算例说明 soft value 的含义，并将 SAC 训练循环接到持续控制评价。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [最大熵连续控制：SAC 的价值、密度与温度](https://yingwen.io/zh/continual-rl/foundations/deep/entropy-control/)：完整soft目标、密度、温度与SAC循环在第二册。


### 一条经验

在 $S_{t}$ 做 $A_{t}$ 后得到 $R_{t+1}$、$S_{t+1}$。奖励属于这次转移，不是到达状态之前另一轮的奖励。

### 折扣与终止

$\gamma$ 控制未来相对权重；真实终止后的价值设为 0。本章记 $\gamma_{t+1}$=$\gamma$(1−terminated)。训练脚本的时间截断不一定是任务终止。

### 表格与函数逼近

表格每个状态或状态动作一组独立参数；函数逼近让不同输入共享参数。更新一个输入可能改变其他输入的预测。

### 熵与温度

H($\pi$)=−Σ$\pi$ log$\pi$ 衡量离散分布不确定性；$\tau$>0 是熵权重。本章用 $\tau$ 避免与学习率 $\alpha$ 混淆。连续密度的微分熵依赖坐标尺度，不与离散熵直接数值比较。

<a id="problem-definition"></a>

## 本章的问题定义

将动作分布熵作为明确的优化收益，控制目标随之改变；不是给普通控制算法附加一个无影响的探索技巧。

### 给定条件与符号

- 固定折扣任务、策略类、动作坐标与熵定义。
- 温度或目标熵、回放与双critic配置、计算预算。

### 需要求解的对象

熵正则策略和soft价值；SAC近似学习这些量并可另行适应温度。

### 信息与数据权限

真实经验生成回放；critic标签停止梯度。actor更新冻结critic参数，但保留其对动作输入的导数。连续动作密度必须包含变换Jacobian。

$$
J_\tau(\pi)=\mathbb E_\pi\!\left[\sum_{t=0}^{\infty}\gamma^t\{R_{t+1}+\tau\mathcal H(\pi(\cdot\mid S_t))\}\right]
$$

$\gamma<1$ 是折扣，$\tau>0$ 是熵温度，$\mathcal H$ 为离散熵或明确坐标下的微分熵。$\tau$ 固定时，这是区别于纯外部回报的目标；自动温度另有目标熵约定。SAC的critic与actor loss是估计和改善该目标的代理。

### 成立条件与解的含义

- 有限离散精确softmax推导要求各动作价值有限；连续积分、微分熵和重参数化需要相应可积/可微条件。
- 奖励尺度、动作尺度与温度共同决定目标；深网、双critic最小值与回放不自动保证收敛。

判断准则：离散Q=[0,1]、温度0.5时动作1概率约0.880797、soft value约1.063464；连续实现检查变换密度与梯度路径，外部收益和熵收益分别报告。

### 适用边界

- soft value不是纯外部回报的价值。
- 连续微分熵不与离散熵直接数值比较。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 改变评价目标 · [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)：熵进入回报，而actor优化和数据分布也按SAC协议改变。

- 组合不同学习问题 · [平均奖励与差分价值](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)：最大熵准则可以结合平均奖励，但需重新定义奖励率、差分critic与参照项。

- 组合不同学习问题 · [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)：熵鼓励分布多样性，但不等于访问新区域、信息增益或恢复能力。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

普通贪心选择忽略目标中的熵；连续策略还需可微采样和正确概率密度。

### 本章的核心思路

先由熵正则最优化推到softmax/soft Bellman，再将评价、改善和温度分别落实。

1. [从熵收益推导策略改善](#lesson-derive)：因为确定贪心不再最优，拉格朗日推导得到softmax与log-sum-exp，并明确温度尺度。

2. [构造soft评价与actor梯度](#sac-targets)：因为后续收益含熵，critic标签扣对数概率；actor通过重参数化动作保留动作价值梯度。

3. [安排各模块的冻结边界](#sac-loop)：因为同批数据上critic、actor和温度互相依赖，明确标签停止梯度、critic参数冻结及目标软更新次序。

结论与条件：有限离散精确局部熵优化有解析解；近似双critic/SAC训练不继承任意网络的全局最优保证，自动温度也需其目标熵可行。

### 相关方法改变了什么

- 普通贪心控制：只优化外部回报，不支付熵收益。

- 精确soft策略迭代：已知或精确价值下执行soft评价与改善。

- SAC：以回放、双critic和重参数化actor近似实现，含额外估计和工程误差。


<a id="lesson-setting"></a>

## 1 · 温度属于目标，学习率属于更新

第二册的 [SAC：价值、密度与温度](/zh/continual-rl/foundations/deep/entropy-control/)完整推导连续动作密度、重参数化与温度更新。本章保留精确离散改善和三类梯度的接口，进一步区分：熵正则规定希望采取怎样的行为，自动温度规定怎样适应约束，生命期评价则规定这种行为和学习付出的代价如何计分。

$$
J_\tau(\pi)=\mathbb E_\pi[\sum_{t\ge0}\gamma^t(R_{t+1}-\tau\log\pi(A_t\mid S_t))]
$$

先固定平稳 MDP、$0\le\gamma<1$ 与 $\tau>0$，并假设回报可积。对离散动作，期望下 $-\log\pi$ 就是熵。$\tau$ 调整奖励和熵的相对权重；参数学习率则决定靠近这个目标的速度。改变前者会改变最优策略问题。

给定当前 Q(s,a)，先解一个状态上的概率分布；再将这个改善步与学习 Q 交替。soft Q 包含当前外部奖励与未来熵，不含当前动作自身的 −$\tau$log$\pi$；soft V 在该状态对 Q−$\tau$log$\pi$ 求期望。固定温度的精确分析为后续近似更新提供参照；自动温度还需另行声明目标熵与约束。

<a id="lesson-derive"></a>

## 2 · 用拉格朗日乘子推导软策略改善

$$
\max_{p_a\ge0,\,\sum_a p_a=1}\sum_a p_a Q_a-\tau\sum_a p_a\log p_a
$$

这是期望价值加熵的优化。对正概率的内部解加乘子 $\eta$，令导数 $Q_a$−$\tau$(log $p_a$+1)+$\eta$=0。

![三个固定动作价值下，三种温度对应的精确 softmax 策略。](https://yingwen.io/crl-figures/concept-research-soft-policy.svg)

每行价值均为 (0,1,2)，仅温度改变。较高温度使动作概率更均匀；它增加目标中的熵权重，不意味着每个随机动作都有较高信息价值。计算脚本 research-mechanisms.mjs。

$$
p_a^*=\frac{\exp(Q_a/\tau)}{\sum_b\exp(Q_b/\tau)},\qquad V^*(s)=\tau\log\sum_a\exp(Q(s,a)/\tau)
$$

这里的星号只表示给定这一组 Q 时的最优状态内分布与目标值；Q 尚未必是真实最优价值。先解出 p∝exp(Q/$\tau$)，再归一化。把 logp=Q/$\tau$−logZ 代回目标，Q 项抵消，剩下 $\tau$logZ。$\tau$>0 下解严格正；计算时减掉最大 Q 防止溢出。

$$
Q^\pi(s,a)=\mathbb E[R_{t+1}+\gamma V^\pi(S_{t+1})],\quad V^\pi(s)=\mathbb E_{a\sim\pi}[Q^\pi(s,a)-\tau\log\pi(a\mid s)]
$$

这是策略评价；前面的 softmax 是给定 Q 的策略改善。两者交替构成 soft policy iteration 的基础。学习中的近似 Q 与有限网络容量会让真实训练偏离精确运算。

$$
L_\pi=\mathbb E_{s\sim D,a\sim\pi_\theta}[\tau\log\pi_\theta(a\mid s)-Q(s,a)]
$$

给定 critic，对这个损失下降等价于向 exp(Q/$\tau$) 的分布靠近（加上与 $\theta$ 无关的归一化常数，可写 KL）。离散时能枚举所有动作；连续时通常借可微重参数化采样。

<a id="sac-targets"></a>

## 3 · SAC 的 critic、actor 与温度不是同一个更新

$$
a\prime\sim\pi_\theta(\cdot\mid s\prime),\quad Y=r+\gamma(1-d)[\min_{j=1,2}Q_{\bar\phi_j}(s\prime,a\prime)-\tau\log\pi_\theta(a\prime\mid s\prime)]
$$

这是现代常用双 Q SAC 的 target；使用 target critics 和当前 actor 采样后继动作，整个 Y 在 critic 回归中停止梯度。min 减少某些高估影响，也可能引入低估；不是无误差真值。

$$
L_{Q_j}=\mathbb E_D[\tfrac12(Q_{\phi_j}(s,a)-\operatorname{sg}Y)^2],\quad L_\pi=\mathbb E[\tau\log\pi_\theta(a_\theta\mid s)-\min_jQ_{\phi_j}(s,a_\theta)]
$$

critic 更新不动 actor；actor 更新冻结 critic 参数，但必须保留 Q 对动作输入的梯度，才能把“哪个动作好”传回 actor。直接对 Q 的整个输出 detach 会把这条路径剪断。

$$
u=\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon,\quad\epsilon\sim\mathcal N(0,I),\quad a=\tanh u,\quad\log\pi(a\mid s)=\log\mathcal N(u;\mu,\sigma)-\sum_i\log(1-\tanh^2u_i)
$$

tanh 改变密度，必须扣 Jacobian。动作若进一步映射到环境区间，还要处理尺度常数。连续多维动作的 log-prob 对动作维求和；不是留一个 [batch, action_dim] 与 [batch] 错误广播。

$$
L_\tau(\log\tau)=\mathbb E[-\tau\operatorname{sg}(\log\pi(a\mid s)+\mathcal H_{\mathrm{target}})]
$$

这是常见温度目标的一种参数化。当实际熵低于目标时，梯度下降会增大 $\tau$。许多实现用 −log$\tau$ 乘同一停止梯度项，方向相同但步长尺度不同。固定 $\tau$ 与自动 $\tau$ 是不同实验配置；目标熵是超参数而不是环境真值。

<a id="sac-loop"></a>

## 4 · 完整训练顺序与一次更新的冻结边界

**算法：固定温度或自动温度二选一；更新次数与次序属于算法设定**

1. 初始化策略 $\pi_\theta$、两个 critic $Q_{\phi_1},Q_{\phi_2}$、目标 critic 与回放池。
1. 每个真实环境步：
  1. 按探索规则采样动作；保存 $(s,a,r,s',d)$，其中 $d$ 只表示真实终止。
  1. 从回放池抽取小批量样本。
  1. 在停止梯度的上下文中计算下一动作和 soft target $Y$。
  1. 分别最小化两个 critic 的平方误差。
  1. 固定 critic 参数，保留 critic 输出对动作的导数。
  1. 通过重参数化动作，最小化 $\tau\log\pi_\theta(a\mid s)-\min_jQ_{\phi_j}(s,a)$。
  1. 若使用自适应温度，以固定的策略对数概率更新温度参数。
  1. 更新目标参数：$\bar\phi_j\leftarrow(1-\eta)\bar\phi_j+\eta\phi_j$。
  1. 环境重置由任务协议决定；网络权重与回放池跨回合保留。

软更新系数 $\eta$ 的不同工程约定可能相反，有的写 polyak 接近 1 作为旧参数保留率。本章式子中 $\eta$ 是新参数占比；照抄变量名却不对式子是常见错误。

<a id="experiment-deep-sac"></a>

### 实验：实验 · 连续 SAC 的梯度路径与真实控制结果

把 tanh 随机策略、双 critic 和固定温度接成完整训练循环后，结果是什么？

**环境与可用信息。** BoundedLQ：观测为位置 x 和剩余时间比例；动作 u∈[−1,1]。位置按 0.92x+0.3u 更新并截到 [−3,3]；奖励为 $-(x^2+0.05u^2)$。初态均匀取自 [−1,1]，40 步是真实有限时域终止。

**设置。** 1200 个真实步，种子默认初始化 32 单元 tanh 隐层；SAC Gaussian actor、两个 critic，温度固定 0.1，不学习温度。actor Adam 0.001，critic Adam 0.002，γ=0.99，replay 4000、批量 32；第 32 步起每步更新，前 64 步均匀探索，目标软更新新参数占比 0.02。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** sac.py 的 critic target 使用当前随机策略和目标双 Q；actor 更新冻结 Q 参数但保留 Q 对动作的导数。_common.py 的 GaussianActor 在 tanh 后校正 log-prob。对照 DDPG 使用确定性 actor、单 critic 和标准差 0.15 的动作噪声。

**测量。** 初始化及之后每 60 个训练步，用独立环境 seed 991 产生的同一组 12 个初态各评价 40 步。SAC 执行 tanh(mean)，不是随机动作；纵轴为未折扣外部回报，不含熵。21 次评估使每个训练 seed 额外使用 10080 个环境步，不进入 replay，也不计入横轴。曲线不能还原训练中随机行为获得的奖励。

```bash
python3 implementations/deep/sac.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-figures/result-deep-sac.svg)

横轴：训练环境步（独立评估交互另计）。纵轴：冻结策略的外部回报。每种方法 1200 训练环境步；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 冻结当前 actor，在种子991生成的同一组12个初态上各跑40步，平均未折扣外部回报。SAC 此处执行 tanh(mean)，没有抽样动作，也不把熵奖励加入评估。

**step：怎样计时。** step 只数训练环境转移；评估交互另计。critic、actor 和 target 的更新数保存在独立字段，相同 step 不保证相同计算量。

**怎样汇总。** 取该记录时刻的值；不先对曲线上的时间点求平均。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** CSV 可重算冻结评估曲线并读取更新计数；没有逐步训练奖励，不能重建行为策略的生命期收益。12个初态的平均先在单次运行内完成。

计算位置：[deep/ddpg.py](https://yingwen.io/crl-code/implementations/deep/ddpg.py) · [deep/td3.py](https://yingwen.io/crl-code/implementations/deep/td3.py) · [deep/sac.py](https://yingwen.io/crl-code/implementations/deep/sac.py) · [deep/_common.py](https://yingwen.io/crl-code/implementations/deep/_common.py)

</details>

**结果分析。** SAC 的平均评价回报从初始化 −21.609 变为第 600 步 −21.157，再到第 1200 步 −0.690；DDPG 对应为 −5.967、−4.474、−2.353。初期 SAC 明显更差，后期均值更高；DDPG 末端 seed 标准差约 2.350，不能只报两个终点数字。

**结论边界。** actor 架构和初始行为分布不同，即使 seed 相同也不是完全同参数对照。1200 步内各有 1169 轮 critic 与 actor 更新，但 SAC 每轮更新两个 Q，DDPG 只更新一个。这是完整方法比较，不能单独归因于熵、双 Q 或随机策略，也未检验自动温度或整个学习过程的行为收益。

**继续实验。** 在同一网络、replay 和数据预算下比较固定温度的多个取值。分别报告随机行为外部收益、确定性评价和含熵目标，解释三者为何可能不同。

[源码](https://yingwen.io/crl-code/implementations/deep/sac.py) · [逐种子记录](https://yingwen.io/crl-code/results/deep-sac/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/deep-sac/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/deep-sac/curves.json)

<a id="lesson-example"></a>

## 5 · 两个动作的解析答案

Q=[0,1]，$\tau$=.5。最优动作 1 概率为 exp(2)/(1+exp(2))≈.880797，不是 1；soft value=.5 log(1+exp(2))≈1.063464，大于最大外部 Q=1，因为目标还包含熵收益。

若两动作 Q 都加 3，概率不变、soft value 加 3；若把奖励整体乘 10 而 $\tau$ 不变，则策略更接近贪心。奖励尺度与温度不能完全分开比较。目标 critic 的两个值也须先对同一动作取 min，不能先对各 critic 分别最大化。

<a id="lesson-code"></a>

## 6 · 核心实现：精确离散备份与可检查 actor 梯度

离散动作 soft value、双 critic target 与分类 actor 的完整梯度下降

```python
def softmax(logits):
    shifted = [math.exp(x - max(logits)) for x in logits]
    return [x / sum(shifted) for x in shifted]


def soft_value(q, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    largest = max(q)
    return largest + temperature * math.log(sum(math.exp((x - largest) / temperature) for x in q))


def discrete_sac_target(reward, discount, probabilities, q1, q2, temperature):
    if discount == 0.0:
        return reward
    expectation = sum(p * (min(a, b) - temperature * math.log(p))
                      for p, a, b in zip(probabilities, q1, q2) if p > 0)
    return reward + discount * expectation


def categorical_actor(logits, q, temperature):
    probabilities = softmax(logits)
    top = max(logits)
    log_z = math.log(sum(math.exp(z - top) for z in logits))
    log_probs = [z - top - log_z for z in logits]
    cost = [temperature * logp - val for logp, val in zip(log_probs, q)]
    objective = sum(p * c for p, c in zip(probabilities, cost))
    # Gradient of sum pi(a) [temperature log pi(a) - Q(a)].
    gradient = [p * (c - objective) for p, c in zip(probabilities, cost)]
    return objective, gradient


def train_soft_bandit(temperature=0.5, iterations=2000):
    q, logits = [0.0, 1.0], [0.0, 0.0]
    for _ in range(iterations):
        _, grad = categorical_actor(logits, q, temperature)
        logits = [z - 0.1 * g for z, g in zip(logits, grad)]
    return {"learned": softmax(logits), "exact": softmax([v / temperature for v in q]),
            "soft_value": soft_value(q, temperature)}
```

此实验先固定 Q=[0,1]，从均匀概率更新 actor，输出 learned 与解析 exact；测试每个 logit 的梯度，并验证 log-sum-exp 的平移性质。它实现 SAC 中可独立核查的 soft policy improvement 与 target，不包含连续环境的全部 SAC 工程。

扩展到连续网络实现时，按本章第 3–4 节逐项对齐张量、停止梯度与训练顺序；不能把离散 softmax 的枚举公式直接换成未归一化的连续概率。章节末的作者实现提供完整优化器与环境循环。

<a id="lesson-branches"></a>

## 7 · 从 soft 改善到持续控制评价

| 概念 | 和 SAC 的关系 | 必须分开的问题 |
| --- | --- | --- |
| 最大熵控制 | 定义 Q / V / actor 的目标 | 不是任意探索都等价于熵 |
| DDPG / TD3 | 相邻的离策略连续控制路线 | 确定性 actor、平滑 target、双 critic 的设计不同 |
| 离散 SAC | 动作期望可精确枚举 | 算法与连续重参数化实现不同 |
| 内在奖励 / RND | 在外部奖励之外增加学习信号 | 新奇不等于策略熵 |
| CRL replay / plasticity | 影响长期训练数据与可训练能力 | SAC 本身不保证长期适应与保留 |

两个动作的例子已经显示：soft value 可以超过最大的外部 Q，因为它还包含熵。若持续任务按外部奖励总和评价，训练时的熵项是一项算法选择，评价仍须单独累计真实奖励；若任务本身要求最小平均熵，就要共同声明该约束及其满足程度。Haarnoja 等的自动温度由后者的对偶问题导出，并非对任意变化任务自动找到最佳探索策略。

因此应先决定共同的评价问题，再比较固定温度与自动温度的完整学习器。Replay、critic、actor、温度和优化器都按各自规则继续更新，探索损失与计算预算进入同一生命期。若每种算法事后改用自己的熵权重给自己计分，soft return 的差异就混合了行为变化与评分变化。

回到变化后的世界，陈旧 critic 可能让 actor 追逐旧后果；熵约束满足也不会指出哪条预测错了。固定数据上的 critic 检查可以定位更新问题，闭环收益才能进一步判断动作带来的新经验是否有用。[探索章](/zh/continual-rl/algorithms/exploration/)继续区分动作随机性、信息获得与任务收益，[持续控制章](/zh/continual-rl/algorithms/control/)则规定比较对象和预算。

<a id="lesson-check"></a>

## 8 · 两个梯度检查题

actor 更新时冻结 critic，是不是要对 min Q detach？不是。冻结的是 critic 参数的梯度，但 Q 对 a 的导数仍需流向 actor。critic 更新时才对整个 target Y stop-gradient。

把 $\tau$ 降到 0 是不是仍需计算 log$\pi$ 的 tanh Jacobian？当实现中还训练温度或报告熵时当然需要；即使固定 $\tau$ 恰为 0，算法也退化成不同的极限目标，不能继续用“相同 SAC 配置”的名义比较。

## 本章的实验设计

区分外部奖励与带熵的优化目标。固定评价口径后，再比较温度、样本量和计算成本。

设定：固定连续动作样本与重参数化噪声，检查动作缩放、tanh 密度修正、双 critic 与温度。性能面板使用相同外部奖励定义。

- 真正终止时不读取后继策略和 critic 输出。
- tanh 后动作的 log-prob 包含相应 Jacobian 修正。
- actor、critic、温度及目标网络的更新时钟可独立追踪。

对照：固定温度与可学习温度；相同网络、动作范围及奖励尺度；完整 SAC/TD3 与共同基座消融分列

记录：原始奖励与熵正则目标分别记录；温度、动作饱和、critic 偏差与策略 KL；回放更新比、数据量和计算

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-deep)

## 学习与研究衔接

连续动作中，策略承担动作搜索。熵、双评论家与回放各有作用，不能合并为一个“稳定化技巧”。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-soft-control) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=soft-control) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=soft-control)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [RVI-SAC: Average Reward Off-Policy Deep Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-rvi-sac-average-control)

### RVI-SAC: Average Reward Off-Policy Deep Reinforcement Learning

Yukinari Hisaki, Isao Ono

ICML 2024 · 2024 · 支持方法与理论

#### 研究问题

深度连续控制若最终按单位时间收益评测，训练能否直接采用平均奖励而非有限折扣？

#### 关键机制

RVI-SAC 将相对价值参照项加入 soft critic，以平均奖励的 soft policy improvement 构造 actor，并用额外 reset critic 与可学习成本控制重置频率。完整实现包含双 critic、经验重放、目标网络和温度更新。

#### 证据

论文给出平均奖励最大熵控制推导，并在 MuJoCo 运动任务中比较；公开实现可核对重置转移是否继续 bootstrap。

#### 条件与限制

理论的表格或精确评价条件不自动覆盖所有神经网络训练。最大熵奖励率、外部奖励率与带 reset 成本的奖励率是三个量；不可将有限折扣 reward centering 当作同一算法。

#### 阅读与实验

逐项对应 critic 参照、actor 分布、reset 指示与 reset 后状态；评价保留外部原始奖励、实际时长、重置次数和训练修正目标。

#### 原文与相关入口

- [ICML 2024 正式论文](https://proceedings.mlr.press/v235/hisaki24a.html)：平均奖励 soft improvement、RVI 与自动 reset cost。

#### 作者代码

[作者仓库 README 标明 reference code 与同名原论文。](https://github.com/yhisaki/average-reward-drl)

average_reward_drl/algorithms/rvi_sac.py 及其参照项、固定 reset cost 变体。


<a id="chapter-code"></a>

## 下载与运行

Python 3.10+，仅标准库。运行精确离散 actor 优化和公式检查；连续 SAC 的完整训练顺序在正文，作者工程在章末。

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

```sh
python3 foundations_detail_lab.py soft-control
python3 foundations_detail_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Haarnoja et al. · Soft Actor-Critic](https://proceedings.mlr.press/v80/haarnoja18b.html)：原始最大熵 actor–critic；原版含 value 网络，与后来常见无独立 value 网络的实现需区分。

- [Haarnoja et al. · Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)：自动温度与实际算法版本，核对目标熵和温度参数化。

- [作者工程 · softlearning](https://github.com/rail-berkeley/softlearning)：原作者维护的最大熵 RL 工程；环境与框架依赖应在单独环境中固定版本。

- [Spinning Up · SAC PyTorch 实现](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py)：教学实现入口，重点追 `compute_loss_q`、`compute_loss_pi`、critic 参数冻结与 Polyak 更新。

- [Spinning Up · SAC 公式与算法说明](https://spinningup.openai.com/en/latest/algorithms/sac.html)：作为实现导览，区分固定温度示例和自动温度扩展。
