# 多臂老虎机：估计、探索与直接策略学习

没有状态转移时，仍需一边估计动作收益，一边决定下一次尝试什么。这个最小问题把估计误差、探索代价和策略更新分开。

## 本章内容

- 推导样本平均和常数步长，解释历史奖励的权重。
- 理解 ε-greedy、UCB 和 gradient bandit 的不同学习对象。
- 运行随机奖励采样，区分实现正确性与有限样本性能。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 概率与期望

概率非负且总和为一；期望是按概率加权的平均，不是单次结果。条件期望只比较满足已知条件的情形。

### 递推与赋值

递推通过已有估计计算新估计。下标表示时间；更新符号表示程序中的赋值，不是数学恒等式。

<a id="lesson-setting"></a>

## 1 · 问题与符号

有 $k$ 个动作。第 $t$ 次选择 $A_t$ 后，只观察它的奖励 $R_t$，看不到未选动作本次会得到什么。先假定每个动作的分布固定，各次奖励在给定动作后独立且均值有限。真实动作价值为 $q_*(a)=\mathbb E[R_t\mid A_t=a]$，算法保存估计 $Q_t(a)$。

目标是在交互预算内获得奖励，不只是最后估准全部动作。总选当前估计最大的动作称为利用；为了获得可能改变排序的信息而尝试其他动作称为探索。低奖励可能意味着动作较差，也可能只是噪声，所以估计与决策相互影响。

$$
a_*\in\arg\max_aq_*(a),\qquad \mathcal R_T=\sum_{t=1}^{T}[q_*(a_*)-q_*(A_t)]
$$

伪遗憾累计实际所选动作与最佳固定动作的均值差。当前动作不改变明天有哪些选择，这是 bandit 与一般序列决策的关键区别。

<a id="lesson-derive"></a>

## 2 · 样本平均的增量形式

某动作已观察 $n-1$ 个奖励，均值为 $Q_n$。得到第 $n$ 个奖励后，把旧奖励之和写成 $(n-1)Q_n$，便可不保存历史而求新均值。这里 n 是这个动作的访问次数，不是总交互步数。

$$
Q_{n+1}=\frac{(n-1)Q_n+R_n}{n}=Q_n+\frac1n(R_n-Q_n)
$$

误差是新观测减旧估计。步长 1/n 给每个历史样本相同权重；第一次观测时初值被消除。

$$
Q_{n+1}=Q_n+\alpha(R_n-Q_n)=(1-\alpha)^nQ_1+\sum_{i=1}^{n}\alpha(1-\alpha)^{n-i}R_i
$$

常数步长对历史进行指数衰减。较大 α 更快追踪变化，也更容易跟随噪声。

无论采用哪种估计，只有被选动作获得新信息。一个从不访问的动作不会因为其他动作被估得更准而自动改善。持续学习中的适应慢，可能来自步长过小，也可能来自没有访问已经改变的选择。

样本平均的收敛讨论针对固定奖励分布和不断增加的访问次数。常数步长通常不消除随机波动，而是在误差与跟踪速度之间折中。若只在总时间上减小步长，一个很晚才开始被尝试的动作可能第一份数据就几乎不起作用；因此必须区分全局时钟与该动作自己的计数。

<a id="bandit-exploration"></a>

## 3 · ε-greedy 与 UCB

$$
\pi_t(a)=\frac{\epsilon}{k}+(1-\epsilon)\mathbf1\{a=g_t\},\qquad g_t\in\arg\max_bQ_t(b)
$$

先用固定规则选择一个贪心动作。探索分支在所有动作中均匀采样，因此贪心动作也可以被探索分支选到。

固定 ε 保留长期探索，但稳定环境中仍会付出选择较差动作的代价。让 ε 随时间减小可以减少代价，却不能快到某些动作仅访问有限次。最终贪心与充分访问是两个要分别检查的条件。

$$
A_t\in\arg\max_a\left[Q_t(a)+c\sqrt{\frac{\log t}{N_t(a)}}\right]
$$

UCB 将估计收益与访问不足的奖励相加。尚未访问的动作先尝试，不能直接除以零。代码使用 c=√2。

这种上界项针对平稳、有界或适当尾界的奖励假设，不是任意神经网络预测的置信区间。均值变化后，旧的大计数可能让探索奖励过小；窗口、折扣计数与变化检测是进一步的选择。

<a id="bandit-gradient"></a>

## 4 · Gradient bandit 的推导

另一条路线保存偏好 $H(a)$，直接用 softmax 决定行为。偏好不是收益估计：给全部 H 加相同常数不改变概率。对 log-softmax 求导，被选动作项贡献一，归一化项贡献对应概率。

$$
\pi_H(a)=\frac{e^{H(a)}}{\sum_be^{H(b)}},\qquad \frac{\partial\log\pi_H(A)}{\partial H(a)}=\mathbf1\{a=A\}-\pi_H(a)
$$

由 log π(A)=H(A)−log Σ exp H 得到。

$$
H(a)\leftarrow H(a)+\alpha(R-B)[\mathbf1\{a=A\}-\pi_H(a)]
$$

baseline B 在选动作前由历史确定。因为 E[∇logπ(A)]=0，减去与本次动作无关的 B 不改变期望收益梯度。

实现先保存旧概率和旧平均奖励，再采样奖励，用它们计算所有偏好更新，最后更新平均奖励。如果先将当前奖励写入 baseline，就改变了上述动作独立性解释。计算 softmax 时先减去最大偏好，可以避免指数溢出。

<a id="bandit-algorithm"></a>

## 5 · 完整交互步骤

**算法：算法伪代码**

1. 初始化 Q、计数 N；或初始化偏好 H 与奖励均值。
1. 每次交互：
  1. 由旧估计构造 ε-greedy / UCB，或由旧偏好构造 softmax。
  1. 选择一个动作；环境只返回这个动作的奖励。
  1. 增加被选动作的计数并更新均值。
  1. 若为 gradient bandit，用旧概率和旧 baseline 更新所有偏好。
  1. 更新奖励平均；记录奖励与动作。

比较要同时记录总奖励和各动作次数。相同整数 seed 不保证不同算法收到逐时对齐的潜在奖励，因为它们可能以不同顺序消耗随机数。本例只使各自运行可复现，不作共同随机数配对的结论。

<a id="lesson-example"></a>

## 6 · 手算与真实采样

| 情形 | 计算 | 结果 |
| --- | --- | --- |
| 第3次访问，旧均值1，新奖励4 | 1+(4−1)/3 | 新均值2 |
| 三动作，ε=.1 | 1−.1+.1/3 | 贪心概率 .933333 |
| 二动作偏好全零，选动作1，R−B=2，α=.1 | ΔH=.1×2×(−.5,.5) | 新偏好 (−.1,.1)，动作1概率约 .549834 |

三臂成功率为 (.2,.5,.8)，每种方法运行 4000 步。seed=7 时 ε-greedy 的次数为 (152,139,3709)，平均奖励 .76025。这说明代码执行了实际动作选择与奖励采样，不证明其优于另两种方法。改变种子和奖励差距后，应观察全过程。

<a id="lesson-code"></a>

## 7 · 源码与实验观察

稳定 softmax、探索分布、UCB 与 gradient bandit 的完整采样循环。

```python
def softmax(values):
    top = max(values)
    weights = [math.exp(v - top) for v in values]
    return [v / sum(weights) for v in weights]

def choose(probabilities, rng):
    u, total = rng.random(), 0.0
    for i, p in enumerate(probabilities):
        total += p
        if u < total:
            return i
    return len(probabilities) - 1

def epsilon_probs(values, epsilon):
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0,1]")
    best = max(range(len(values)), key=lambda i: values[i])
    probabilities = [epsilon / len(values)] * len(values)
    probabilities[best] += 1.0 - epsilon
    return probabilities

def ucb_action(values, counts, time):
    unseen = [a for a in range(len(values)) if counts[a] == 0]
    if unseen:
        return unseen[0]
    return max(range(len(values)), key=lambda a:
               values[a] + math.sqrt(2 * math.log(time) / counts[a]))

def bandit(method, steps=4000, seed=7):
    """Stationary Bernoulli arms. Gradient baseline uses the OLD mean."""
    if steps < 1:
        raise ValueError("steps must be positive")
    rng = random.Random(seed)
    means = [0.2, 0.5, 0.8]
    q, counts, preferences = [0.0]*3, [0]*3, [0.0]*3
    baseline, total = 0.0, 0.0
    for t in range(1, steps + 1):
        if method == "epsilon":
            action = choose(epsilon_probs(q, 0.1), rng)
        elif method == "ucb":
            action = ucb_action(q, counts, t)
        elif method == "gradient":
            probabilities = softmax(preferences)
            action = choose(probabilities, rng)
        else:
            raise ValueError(method)
        reward = float(rng.random() < means[action])
        total += reward
        counts[action] += 1
        q[action] += (reward - q[action]) / counts[action]
        if method == "gradient":
            for a in range(3):
                preferences[a] += 0.1 * (reward - baseline) * (
                    float(a == action) - probabilities[a])
        baseline += (reward - baseline) / t
    if method == "ucb":
        action = ucb_action(q, counts, steps+1)
        next_policy = [float(a == action) for a in range(3)]
    else:
        next_policy = (softmax(preferences) if method == "gradient"
                       else epsilon_probs(q, 0.1))
    return {"counts": counts, "mean_reward": total / steps, "q": q,
            "next_action_probabilities": next_policy}
```

输出分开记录每个动作的访问次数、样本均值、累计平均奖励和下一次选择分布。ε-greedy 与 gradient bandit 给出随机分布；固定并列规则的 UCB 根据当前均值、计数和时间确定下一动作，因此其条件选择分布是 one-hot，不是 ε-greedy 分布。

先把预算降到三步，检查 UCB 是否恰好尝试每个动作一次；再把最优两臂均值改得更接近，观察识别排序所需的信息增加。若引入中途均值变化，同时保留样本平均和常数步长对照。比较变化后的窗口奖励，而不是只看被变化前历史占主导的累计平均。

<a id="lesson-branches"></a>

## 8 · 限制与持续学习

乐观初值通过高估尚未试过的选择诱导探索，但这种驱动力可能在所有动作访问过后消失。随机覆盖、置信上界和直接策略梯度从不同机制分配经验，不能仅根据一个最终动作概率判断其作用。

若动作影响未来状态，眼前奖励均值不足以评价决策。若均值随时间改变，最佳固定动作也可能不是合适比较对象。先规定变化、预算与目标，再选择累计收益、窗口回报或跟踪误差。

<a id="lesson-check"></a>

## 9 · 带答案练习

问：Q=.8，α=.1，随后三个奖励均为零，估计是多少？答：.72、.648、.5832。若改用极小的样本平均步长，跟踪会更慢；这个算例尚未检验是否探索到了变化。

问：UCB 能否忽略从未访问、Q=0 的动作？答：不能。零是初始化而非低收益证据；程序应优先尝试未访问动作。

问：为什么 gradient bandit 更新未选动作？答：概率受到归一化约束。一次高于 baseline 的奖励提高被选动作的相对偏好，同时降低其他动作；偏好增量之和为零。

<a id="chapter-code"></a>

## 下载与运行

Python 3，仅标准库。下载完整文件后在其目录运行；页面展示核心区域。小环境和解析检验用于理解机制，不是论文基准复现。

[下载 tabular_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/tabular_textbook_lab.py)

```sh
python3 tabular_textbook_lab.py bandits
python3 tabular_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：对应第 2–8 章。以下推导、例子和标准库实现独立编写；原书用于核对定义、算法关系与适用条件。

- [David Silver · UCL 强化学习课程](https://davidstarsilver.wordpress.com/teaching/)：原课程页面提供 MDP、动态规划、预测、控制、探索和规划的讲义及视频。

<a id="study-connections"></a>

## 与教材主线的衔接

- [控制问题与广义策略迭代](https://yingwen.io/zh/continual-rl/algorithms/control/)
- [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)
- [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

对应原始材料：2.1–2.8。本文为原创讲解，原书、论文与上游代码保留各自许可。
