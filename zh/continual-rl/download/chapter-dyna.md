# Dyna：模型学习与规划

直接学习只使用真实发生的一步；模型学会了后果以后，可以从旧状态重新思考。Dyna 的关键是把真实学习、模型学习、规划分开再连接。

## 本章内容

- 能区别 transition model、Q 函数与规划更新。
- 实现表格 Dyna-Q，并解释规划步数为何不是免费样本。
- 理解模型陈旧、分布选择与优先规划的失效方式。

<a id="problem-definition"></a>

## 本章的问题定义

真实交互昂贵时，从真实经验学后果模型，再用预算内的模型备份传播价值。

### 给定条件与符号

- 固定折扣控制任务、真实经验、模型表示和可规划状态动作。
- 每个真实步的规划次数或总计算预算；模型更新与真实终止规则。

### 需要求解的对象

共享价值上的真实学习与模型规划组合；目标是控制收益，模型拟合与模型内收敛是中间子问题。

### 信息与数据权限

真实经验更新模型 $\hat M=(\hat r,\hat P)$；模拟后果仅用于价值/策略计算。规划不得将自己的预测重新记为独立真实事实。

$$
(\hat TQ)(s,a)=\hat r(s,a)+\gamma\sum_{s'}\hat P(s'\mid s,a)\max_bQ(s',b)
$$

$\hat r$ 是奖励模型，$\hat P$ 是转移模型，$\gamma<1$ 是折扣，$Q$ 为动作价值。此式是当前模型的控制备份，不是新获得的真实经验；真实任务的最优值对应真实算子 $T$。规划预算有限且 $\hat M$ 可能有偏。

### 成立条件与解的含义

- 静态界要求真实与模型的有界折扣算子均满足相应收缩条件。
- 随机世界不能把最后一个后果当精确分布；非平稳世界必须检查模型年龄和真实验证。

判断准则：两步链在模型保存后的一次起点备份传播0.9；固定模型下检查备份残差及解析值；匹配真实步和总备份两种预算分别比较，并报告模型偏差。

### 适用边界

- 增加模拟备份不增加真实证据。
- 模型内收敛不证明真实最优或变化后已经适应。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)：模型学习提供后果接口，Dyna将它接到真实学习与模拟备份。

- 特例：增加条件 · [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)：Dyna是规划的一种学习期价值更新方式；行动时搜索和想象训练采用不同接口。

- 组合不同学习问题 · [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)：旧模型无法自行发现外界变化，Dyna-Q+类真实再验证机制需要探索。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

稀少真实奖励传播慢，但模型偏差可被反复规划放大。

### 本章的核心思路

真实数据决定模型事实，模型备份重用其后果计算；对真实性与传播速度分开检验。

1. [复用同一个价值备份](#lesson-derive)：因为真实与模型后果都能构造Bellman标签，直接学习和规划共享价值，但区别后果来源。

2. [落实真实—模型—规划顺序](#dyna-steps)：因为只有真实转移带来新证据，先真实备份和建模，再预算内选择已观察起点规划。

3. [把计算投向传播前驱](#dyna-priorities)：因为价值变化只影响相关前驱，用残差队列与前驱索引调度，而模型真实性仍单独验证。

结论与条件：固定精确有限折扣模型可按Bellman收缩求解；学得模型留下误差，更多备份只能减少当前模型内的求解误差。

### 相关方法改变了什么

- 无规划Q-learning：只用真实后果更新，可作为零规划退化对照。

- 均匀Dyna：在已建模状态动作上分配备份。

- Prioritized sweeping/Dyna-Q+：前者调度计算传播，后者鼓励真实再尝试陈旧行为，解决不同不足。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 一条经验

在 $S_{t}$ 做 $A_{t}$ 后得到 $R_{t+1}$、$S_{t+1}$。奖励属于这次转移，不是到达状态之前另一轮的奖励。

### 折扣与终止

$\gamma$ 控制未来相对权重；真实终止后的价值设为 0。本章记 $\gamma_{t+1}$=$\gamma$(1−terminated)。训练脚本的时间截断不一定是任务终止。

### 表格与函数逼近

表格每个状态或状态动作一组独立参数；函数逼近让不同输入共享参数。更新一个输入可能改变其他输入的预测。

<a id="lesson-setting"></a>

## 1 · 三个部件不是三个互斥算法

设离散状态动作、先从确定性固定环境开始，优化折扣回报。直接学习更新 Q；模型记下每个已访问 (s,a) 的奖励和后继；规划从模型选一个状态动作再进行备份。真实经验仍是模型知识的来源，规划不会凭空获得环境已经变化的证据。

| 部件 | 输入 | 输出 / 持久状态 |
| --- | --- | --- |
| 直接学习 | 真实 (s,a,r,s′) | 改善当前 Q |
| 模型学习 | 相同真实转移 | M(s,a)=(r,$\gamma_{t+1}$,s′) |
| 规划 | 模型转移及当前 Q | 再次改善 Q |
| 行为 | 当前状态与 Q | 实际动作，决定下一条真实经验 |

这个简单模型只适用于确定性环境。随机环境若只记最后一个后继，会把噪声当作确定规律；应保存条件分布、经验样本集合或其他概率模型。用神经网络并不会取消建模这个分布的责任。

<a id="lesson-derive"></a>

## 2 · 从真实 Q-learning 备份到模型备份

$$
Q(s,a)\leftarrow Q(s,a)+\alpha[r+\gamma_{t+1}\max_b Q(s\prime,b)-Q(s,a)]
$$

真实更新与 Q-learning 相同。模型预测一个后果后，可把预测的 r、s′代入相同备份。改变的是备份的数据来源。价值函数与转移模型仍是不同的学习对象。

$$
\hat TQ(s,a)=\hat r(s,a)+\gamma\sum_{s\prime}\hat P(s\prime\mid s,a)\max_b Q(s\prime,b)
$$

这是概率模型的期望备份；也可从模型采样一个后果进行 sample backup。期望备份计算更贵、单次方差更小；采样备份可用较低成本集中在更有用的状态动作。

$$
\|Q^*_{\hat M}-Q^*_M\|_\infty\le\frac{\|\hat TQ^*_M-TQ^*_M\|_\infty}{1-\gamma}
$$

两个模型的 Bellman 最优算子均是 $\gamma$ 压缩且有界时，由加减 T̂Q*M 和三角不等式得到。它说明即使规划完全收敛，错误模型仍会留下由 1/(1−$\gamma$) 放大的误差；更多规划不等于更多真实知识。

这条不等式是固定模型、固定 MDP 下的分析。CRL 中模型、奖励和表示同时改变，误差项随时间变，需要重新衡量旧模型与当前世界的偏差，不能拿静态收敛替代适应分析。

<a id="dyna-steps"></a>

## 3 · Dyna-Q 的完整更新过程

**算法：Dyna-Q 的三个学习步骤共享 Q，但模型只由真实数据更新**

1. 初始化动作价值 $Q$ 与空模型 $\mathcal M$；给定每步规划次数 $n$。
1. 每个真实环境步：
  1. 按 $\epsilon$-greedy 策略选择 $a$，观察 $r,s'$ 与终止标记。
  1. 使用真实转移进行一次 Q-learning 更新。
  1. 更新模型 $\mathcal M(s,a)\leftarrow(r,\gamma',s')$。
  1. 重复 $n$ 次：
    1. 从已观察的状态—动作对中选择 $(\tilde s,\tilde a)$。
    1. 由模型得到 $(\hat r,\hat\gamma,\hat s')$。
    1. $Q(\tilde s,\tilde a)\leftarrow Q(\tilde s,\tilde a)+\alpha[\hat r+\hat\gamma\max_bQ(\hat s',b)-Q(\tilde s,\tilde a)]$。
  1. 继续真实交互；回合边界由任务协议决定。

不能把规划生成的转移重新当成真实样本去更新同一个模型，否则会用自己的想象强化自己的错误；也不能把 n 次规划算作 n 个真实环境步。在报告中分别记录 real steps、model updates、planning backups 和实际运行时间。

<a id="lesson-example"></a>

## 4 · 两步链说明规划为什么能传播新奖励

A→B 的奖励为 0，B→终点奖励为 1，$\gamma$=.9，Q 初始 0，$\alpha$=1。第一次真实经过 A 时，Q(A)=0；经过 B 后，Q(B)=1。若模型已保存 A→B，在下一次真实从 A 经过之前，规划一次 A 就能得 Q(A)=.9。

现在世界把 B 的奖励改成 −1，但智能体尚未访问 B。模型仍保存 +1，规划一千次也只会把旧答案强化得更一致。只有获得变化证据、采用先验不确定性或主动再探索，才可能发现新事实。

若 n=0，本算法退回相同真实更新的 Q-learning。若模型是 oracle，比较的是规划调度；若模型也学习，结果同时混合了建模与规划质量。最好分别做这两个对照。

<a id="lesson-code"></a>

## 5 · 核心实现：五状态链的真实学习与规划

可独立运行的 Dyna-Q，记录真实与规划预算

```python
def q_backup(q, s, a, reward, discount, nxt, alpha):
    bootstrap = max(q[nxt]) if discount else 0.0
    q[s][a] += alpha * (reward + discount * bootstrap - q[s][a])


def dyna_chain(planning=5, steps=1000, seed=7):
    """Deterministic 5-state chain. Right at 4 gives 1 and truly terminates."""
    rng, q, model, state = random.Random(seed), [[0.0, 0.0] for _ in range(5)], {}, 0
    for _ in range(steps):
        greedy = [a for a in range(2) if q[state][a] == max(q[state])]
        action = rng.randrange(2) if rng.random() < 0.2 else rng.choice(greedy)
        nxt = max(0, state - 1) if action == 0 else state + 1
        reward, discount = (1.0, 0.0) if nxt == 5 else (0.0, 0.9)
        q_backup(q, state, action, reward, discount, nxt, 0.1)
        model[(state, action)] = (reward, discount, nxt)
        for _ in range(planning):
            simulated_s, simulated_a = rng.choice(list(model))
            r, g, sp = model[(simulated_s, simulated_a)]
            q_backup(q, simulated_s, simulated_a, r, g, sp, 0.1)
        state = 0 if not discount else nxt
    return {"Q": q, "right_action_reference": [0.9 ** (4 - s) for s in range(5)],
            "real_steps": steps, "planning_updates": planning * steps}
```

五个状态，右移到第 5 个边界得奖励 1 并终止，左移不越过 0。$\gamma$=.9，最优右移动作值为 $[.9^4,.9^3,.9^2,.9,1]$。测试对这个解析向量检查，以此区分数值误差与有限样本误差。

运行 dyna 后检查 `right_action_reference` 与各行的右移 Q。把 planning 从 5 改为 0，比同样真实步数下的传播速度；再比同样总 backup 次数，结论可能不同。若引入随机转移，先换成正确的概率模型，再谈 Dyna 是否有效。

<a id="dyna-priorities"></a>

## 6 · 有限规划预算应该花在哪里？

$$
p(s,a)=|\hat r+\hat\gamma\max_bQ(\hat s\prime,b)-Q(s,a)|
$$

prioritized sweeping 用模型预测的备份残差安排更新；后继值改变后，沿 predecessor 关系把影响传播到前驱。这里的 priority 是规划更新优先级，不等于 replay 的采样权重。

- 维护每个状态有哪些前驱 (s,a)，并在模型结构变化时更新这个反向索引。
- 真实观察后计算残差，大于阈值则入优先队列；每次弹出最大项做备份。
- 该状态 value 变化后，重新计算其前驱残差并入队；避免无限重复、陈旧队列值和重复预算统计。
- 如果规划误差很大只是因为模型本身错误，优先更新可能把错误放大；应同时监测模型真实性和价值残差。

Dyna-Q+ 给长期未尝试的行为加随时间增长的探索奖励，鼓励重新检查世界；它不是通过旧模型自行检测变化。奖励的时间单位、未尝试动作的初始化、访问时间更新必须一致，才能解释实验。

<a id="lesson-branches"></a>

## 7 · 从 Dyna 到世界模型和时间抽象规划

| 路线 | 复用的共同思想 | 新增难点 |
| --- | --- | --- |
| 表格 Dyna / prioritized sweeping | 真实经验学模型，模型改善价值 | 规划起点与预算 |
| MPC | 模型预测短期候选动作序列，每步重规划 | 模型误差、约束、在线搜索代价 |
| Dreamer / latent imagination | 在学得潜在模型中训练行为 | 表示与动力学的联合误差、想象分布 |
| MuZero | 为搜索学习决策相关模型 | 不要求重建完整观测；训练与搜索协议不同 |
| Option model + planning | 一个模型后果覆盖随机多步行为 | 奖励累计、时长折扣、终止分布 |
| 持续世界模型 | 在变化中更新模型并保存有用规律 | 旧数据、潜在状态漂移、模型与策略分别遗忘 |

这些方法不是只有网络大小不同。预测完整观测、预测奖励与转移、预测 option 后果、为搜索构造潜在状态，是不同模型接口；判断方法前应把输入、输出、使用者和训练信号写出来。

<a id="lesson-check"></a>

## 8 · 习题与讨论

更多规划后真实回报下降，是不是说明模型学习没用？不能。可能是模型不准、规划分布选错、价值逼近不稳，或计算延迟占用了真实行动机会。分别用 oracle 模型、固定 Q 目标、匹配预算隔离原因。

为什么不能把 model loss 下降直接等同于控制收益？模型可能在大量容易预测但与动作选择无关的观测上变准，却仍把关键奖励或终止事件预测错。应记录决策相关误差，并做固定策略或固定模型的消融。

## 本章的实验设计

同时匹配真实交互预算和规划计算预算。增加备份次数本身也可能提高回报。

设定：在小迷宫提供可准确维护的表格模型，比较每个真实步后执行 0、少量或较多模拟 backup；再引入模型错误。

- 真实转移与模型生成转移进入各自计数。
- 规划 backup 使用指定模型和价值版本。
- 关闭模型更新与关闭规划是不同开关。

对照：无规划 Q-learning；准确模型与学习模型；同计算的额外真实数据复用

记录：真实交互收益与模拟 backup 数；模型奖励/转移误差；墙钟、规划占比和决策延迟

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-planning)

## 学习与研究衔接

一次真实经验既更新模型，也支持额外规划。持续环境中应同时测模型陈旧程度和规划收益。

[分册导读](https://yingwen.io/zh/continual-rl/start/classic-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-dyna) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=dyna) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=dyna)



<a id="chapter-code"></a>

## 下载与运行

Python 3.10+，仅标准库。完整表格 Dyna-Q 与解析参照；概率模型、优先队列与神经世界模型需按正文接口进一步实现。

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

```sh
python3 foundations_detail_lab.py dyna
python3 foundations_detail_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：第 3–8 章建立 MDP、动态规划、MC、TD 与规划；第 9–13 章把这些更新扩展到函数逼近、资格迹和策略梯度。按本章的问题找相应章节，不必从头重读。

- [Sutton · Dyna, an integrated architecture](https://doi.org/10.1145/122344.122377)：把行动、模型学习、直接学习与规划整合的原始架构思想。

- [Moore & Atkeson · Prioritized Sweeping](https://doi.org/10.1007/BF00993104)：理解前驱索引与异步更新调度，而不只记“按误差排序”。

- [Sutton & Barto 教材复现代码](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)：社区维护的教材实现，不是 Dyna 原论文历史代码；chapter08 可比较 Dyna-Q 与 Dyna-Q+。
