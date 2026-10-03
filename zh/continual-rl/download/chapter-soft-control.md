# 最大熵控制与 Soft Actor–Critic

把多样性纳入目标，会怎样改变 Bellman 方程、策略改善和实际训练？先用离散精确解推清楚，再给出连续 SAC 的完整接口。

## 本章内容

- 从带熵约束的最优化推导 softmax 与 log-sum-exp。
- 辨认 soft Q、策略熵、双 critic 和自动温度分别起什么作用。
- 实现离散熵正则 actor 更新，并能检查连续 SAC 的 log-prob 和梯度路径。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 一条经验

在 $S_{t}$ 做 $A_{t}$ 后得到 $R_{t+1}$、$S_{t+1}$。奖励属于这次转移，不是到达状态之前另一轮的奖励。

### 折扣与终止

$\gamma$ 控制未来相对权重；真实终止后的价值设为 0。本章记 $\gamma_{t+1}$=$\gamma$(1−terminated)。训练脚本的时间截断不一定是任务终止。

### 表格与函数逼近

表格每个状态或状态动作一组独立参数；函数逼近让不同输入共享参数。更新一个输入可能改变其他输入的预测。

### 熵与温度

H($\pi$)=−Σ$\pi$ log$\pi$ 衡量离散分布不确定性；$\tau$>0 是熵权重。本章用 $\tau$ 避免与学习率 $\alpha$ 混淆。连续密度的微分熵依赖坐标尺度，不与离散熵直接数值比较。

<a id="lesson-setting"></a>

## 1 · 熵正则化控制目标

$$
J_\tau(\pi)=\mathbb E_\pi[\sum_{t\ge0}\gamma^t(R_{t+1}-\tau\log\pi(A_t\mid S_t))]
$$

期望下 −log$\pi$ 就是熵。$\tau$ 大更愿意保留动作多样性；$\tau$→0 才回到普通奖励最大化的相应极限。用 entropy bonus 改变了优化问题，而不仅是修复梯度。

先假设离散动作、已知一组当前 Q(s,a)，求一个状态上最好的概率分布。之后再将这个策略改善步与从 replay 学 Q 交替。soft Q 的常见约定包含当前外部奖励与未来熵，但不包含当前动作自身的 −$\tau$log$\pi$；soft V 才在该状态对 Q−$\tau$log$\pi$ 求期望。

<a id="lesson-derive"></a>

## 2 · 用拉格朗日乘子推导软策略改善

$$
\max_{p_a\ge0,\,\sum_a p_a=1}\sum_a p_a Q_a-\tau\sum_a p_a\log p_a
$$

这是期望价值加熵的优化。对正概率的内部解加乘子 $\eta$，令导数 $Q_a$−$\tau$(log $p_a$+1)+$\eta$=0。

$$
p_a^*=\frac{\exp(Q_a/\tau)}{\sum_b\exp(Q_b/\tau)},\qquad V^*(s)=\tau\log\sum_a\exp(Q(s,a)/\tau)
$$

先解出 p∝exp(Q/$\tau$)，再归一化。把 logp=Q/$\tau$−logZ 代回目标，Q 项抵消，剩下 $\tau$logZ。$\tau$>0 下解严格正；计算时减掉最大 Q 防止溢出。

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

## 7 · 熵正则、探索与持续学习的关系

| 概念 | 和 SAC 的关系 | 必须分开的问题 |
| --- | --- | --- |
| 最大熵控制 | 定义 Q / V / actor 的目标 | 不是任意探索都等价于熵 |
| DDPG / TD3 | 相邻的离策略连续控制路线 | 确定性 actor、平滑 target、双 critic 的设计不同 |
| 离散 SAC | 动作期望可精确枚举 | 算法与连续重参数化实现不同 |
| 内在奖励 / RND | 在外部奖励之外增加学习信号 | 新奇不等于策略熵 |
| CRL replay / plasticity | 影响长期训练数据与可训练能力 | SAC 本身不保证长期适应与保留 |

在持续环境中，陈旧 critic 会让 actor 追逐过时值；自动温度只能调整随机性，不能识别哪部分世界已改变。先用固定数据测试价值跟踪，再分析策略诱导分布，能避免把所有失败归因于熵系数。

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

[分册导读](https://yingwen.io/zh/continual-rl/start/deep-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-soft-control) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=soft-control) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=soft-control)

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
