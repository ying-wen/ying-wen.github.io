# 多步回报、资格迹与 True-online TD

当前到来的奖励怎样更新过去的预测，同时保留正确的在线更新语义？

## 本章内容

- 把 n-step 与 lambda-return 的前向目标推导成后向信用传播。
- 解释普通累积迹与 true-online 的差别，推导 Dutch trace。
- 用重复状态和独立前向实现检查每个轨迹前缀。

<a id="problem-definition"></a>

## 本章的问题定义

给定固定策略的预测任务，要求在线把后续误差分配给此前参数方向。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 给定目标策略 $\pi$，$v_\pi(s)=\mathbb E_\pi[\sum_{k\ge0}\gamma^kR_{t+k+1}\mid S_t=s]$。
- 误差权重 $d(s)\ge0$ 且 $\sum_sd(s)=1$；权重指定在哪些状态上评价估计。
- 线性特征 $x_t$、参数 $w_t$、迹系数 $\lambda\in[0,1]$、步长 $\alpha$。

### 需要求解的对象

以递推摘要实现规定的前向多步更新；先验证更新语义，再比较统计效率。

### 信息与数据权限

只能按时间到达顺序处理数据；旧特征的影响可由资格迹保存。

$$
w_t^{\rm backward}=w_t^{\rm online\ forward}\quad\text{for every observed prefix }t
$$

这是实现等价性要求，不是外部控制目标。在线前向视图规定每个前缀如何重构目标；传统冻结参数 λ-return 是不同参照。

### 成立条件与解的含义

- True-online TD 的精确逐前缀等价先在线性函数逼近及匹配的步长定义下成立。
- 终止、截断和参数版本必须在两个实现中一致。

判断准则：对同一短轨迹比较每个前缀的参数，而非仅比较最后的平均回报。

### 适用边界

- 把线性 true-online 等价直接推广到任意深度网络。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)：本章把信用分配限定为固定线性预测及指定在线前向视图；相关章还包含离策略、非线性与递归状态信用。

- 组合不同学习问题 · [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)：本章的资格迹可作为流式学习器的持久摘要，组合时须计入其参数维度、计算和内存；它不是样本回放。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

参数在数据到来时不断改变，冻结权重下的交换求和不再精确描述实际更新。

### 本章的核心思路

先明确在线前向参照，再加入 Dutch trace 和预测变化修正以匹配它。

1. [看清多步目标的混合](#lesson-derive)：λ 控制不同未来长度的权重，先在冻结参数下建立代数关系。

2. [把参照改为在线前缀](#traces-online)：指定每一步重新计算什么，才能判断后向实现是否正确。

3. [修正重复特征与预测变化](#traces-dutch)：两项修正共同实现等价，不能只换一种迹递推。

结论与条件：等价保证的是算法语义，不是任意任务上的收益优势。

### 相关方法改变了什么

- Accumulating / replacing traces：两者处理重复激活不同，不可无条件互换。

- 传统 TD(λ) / true-online TD：后者精确匹配指定在线前向视图，而非只在小步长下近似。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 前向与后向视图

前向视图从某个状态向未来构造回报目标；后向视图在反馈到来时，根据过去特征留下的资格更新权重。两者描述的时间方向不同。

### 固定特征与线性预测

本章 true-online 的精确等价以线性预测、固定特征及给定步长约定为基础。神经网络梯度迹是另一种近似，不自动具有同一等价。

<a id="lesson-setting"></a>

## 1 · 一步 TD 的信用传播瓶颈

在一条长轨迹中，奖励可能很晚才出现。一步 TD 只更新当前预测，过去状态要等到再次访问后才逐渐得到影响。多步回报直接包含更多实际反馈，资格迹则让刚到来的误差同时影响近期相关特征。它们解决时间信用分配，不负责决定保存哪些旧任务能力。

考虑回合终点 $T$、固定折扣 $\gamma$ 和线性预测 $v_t=w^\top x_t$。真正终止状态的特征取零。报告窗口结束但环境未终止时，不能自动将下一预测清零。资格迹重置的位置也必须与任务的历史边界一致。

$$
G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^n\hat v(S_{t+n}),
$$

n 步目标将前 n 个真实奖励与后续预测结合；若提前到达真正终止，则在终止处截断且没有 bootstrap。

<a id="lesson-derive"></a>

## 2 · Lambda-return 与 TD 误差的展开

$$
\begin{gathered}G_t^\lambda=(1-\lambda)\sum_{n=1}^{T-t-1}\lambda^{n-1}G_t^{(n)}+\lambda^{T-t-1}G_t,\\ 0\le\lambda\le1.\end{gathered}
$$

有限回合的最后一项保留全部剩余权重，所以各项系数和为 1。$\lambda=0$ 给一步目标，$\lambda=1$ 给完整回报。

$$
\begin{aligned}G_t^\lambda-v_t&=\sum_{k=t}^{T-1}(\gamma\lambda)^{k-t}\delta_k,\\ \delta_k&=R_{k+1}+\gamma v_{k+1}-v_k.\end{aligned}
$$

在全部预测使用同一组冻结权重时，展开右侧并消去相邻价值项，就得到左侧。这个代数恒等式解释了为什么后来的 TD 误差可以按衰减系数更新过去特征。

$$
\begin{gathered}\sum_t\alpha(G_t^\lambda-v_t)x_t=\alpha\sum_k\delta_k e_k,\\ e_k=\sum_{t=0}^k(\gamma\lambda)^{k-t}x_t=\gamma\lambda e_{k-1}+x_k.\end{gathered}
$$

交换两重求和的次序，得到累积资格迹。不需要保存全部过去特征，只保存它们的衰减和。

冻结权重的批量等价并不证明在线逐步更新时的精确等价。在线更新会改变后续出现的预测值；相同状态重复出现时，差异尤其明显。普通 TD(lambda) 是重要的有效方法，但不能忽略这个条件就声称它严格实现某个在线前向过程。

<a id="traces-online"></a>

## 3 · 在线前向视图究竟指什么

在只看到前缀 $0,\ldots,h$ 时，构造每个过去状态的 interim lambda-return：只能组合已经可见的 n 步目标，最长项保留剩余权重。随着 $h$ 增长，过去状态的目标不断修订。

$$
G_k^{\lambda\mid h}=(1-\lambda)\sum_{n=1}^{h-k-1}\lambda^{n-1}G_k^{(n)}+\lambda^{h-k-1}G_k^{(h-k)}.
$$

这里 n 步 bootstrap 在抵达 $S_{k+n}$ 之前的在线权重上计算，即使用 $w_{k+n-1}^\top x_{k+n}$。预测参数的时间索引是定义的一部分。

独立参考实现对每个新前缀都从回合初始权重开始，按时间顺序重做过去所有状态的监督更新，并使用上述已修订目标。这样定义清楚，但计算开销随轨迹增长，不适合长期流式使用。True-online 方法用固定大小的递归状态计算相同的最终权重。

这个前向算法是一个用来定义和核对更新含义的参照。它允许概念性地重算前缀，不意味着真实在线学习器获得了未来经验。代码仅在测试中运行这个慢版本，并逐前缀比较结果。

<a id="traces-dutch"></a>

## 4 · Dutch trace 与预测变化修正

$$
e_t=\gamma\lambda e_{t-1}+[1-\alpha\gamma\lambda e_{t-1}^\top x_t]x_t.
$$

把它写成 $e_t=\gamma\lambda(I-\alpha x_tx_t^\top)e_{t-1}+x_t$，可以看出当前监督更新会改变既有信用方向，不能只是继续相加。这是 Dutch trace 的线性修正。

$$
\begin{gathered}\delta_t=R_{t+1}+\gamma w_t^\top x_{t+1}-w_t^\top x_t,\\ w_{t+1}=w_t+\alpha(\delta_t+v_t-v_{\rm old})e_t-\alpha(v_t-v_{\rm old})x_t.\end{gathered}
$$

$v_{\rm old}$ 保存上一时刻在当前状态上作出的旧预测。后两项的组合补偿在线权重变化；只有换成 Dutch trace 而保留普通 TD 更新，并不构成完整 true-online 算法。

**算法：常数步长 True-online TD(lambda)**

1. 回合开始：初始化 $e=0,v_{\rm old}=0$；保留待学习权重
1. 用旧权重计算 $v=w^\top x,v'=w^\top x'$，再计算 $\delta$
1. 用旧 trace 的内积计算 Dutch 修正，然后写入新 trace
1. 执行完整的 true-online 权重更新
1. 保存 $v_{\rm old}\leftarrow v'$；真正终止后清除回合 trace

本页公式使用常数步长。逐时间改变步长时，原论文给出另外的缩放 trace 形式，不应在未核对的情况下随意替换。线性精确等价也是有限轨迹的算法等价，不等于在所有任务上都保证更高回报。

<a id="lesson-example"></a>

## 5 · 重复状态的两步手算

只有一个非终止状态，特征为 1。轨迹是“该状态、该状态、终止”，奖励依次为 0、1。取 $w_0=0,\gamma=\lambda=1,\alpha=0.5$。第一步没有误差，权重仍为 0，trace 为 1。

第二步，普通累积 trace 变为 2，误差为 1，因此权重变为 1。Dutch trace 则是 $1+[1-0.5\times1]\times1=1.5$；本例预测变化修正为零，得到权重 0.75。

$$
\begin{aligned}w^{\rm forward}_1&=0+0.5(1-0)=0.5,\\ w^{\rm forward}_2&=0.5+0.5(1-0.5)=0.75.\end{aligned}
$$

完整前缀的前向过程对两个出现位置依次作目标为 1 的监督更新，因此得到 0.75，与 true-online 相同。普通累积迹把两次误差都按旧预测累加，得到不同答案。

这不是说普通 TD 在这个小问题一定无法学习。例子只区分有限步长下的更新语义。把步长减小，两个结果更接近；把状态换成互不重叠的独立特征，也可能降低差异。

<a id="lesson-code"></a>

## 6 · 两个独立实现逐前缀对照

True-online 更新与慢速在线前向参照

```python
def true_online_td(features, rewards, alpha=0.1, gamma=0.9, lam=0.8, initial=None):
    """One episode/prefix. features has len(rewards)+1; terminal features are zero."""
    if len(features) != len(rewards) + 1 or not 0 <= lam <= 1:
        raise ValueError("invalid trajectory or lambda")
    weights = list(initial) if initial is not None else [0.0] * len(features[0])
    trace, old_value, history = [0.0] * len(weights), 0.0, [weights[:]]
    for x, reward, xp in zip(features, rewards, features[1:]):
        value, next_value = dot(weights, x), dot(weights, xp)
        delta = reward + gamma * next_value - value
        correction = 1 - alpha * gamma * lam * dot(trace, x)
        trace = [gamma * lam * e + correction * v for e, v in zip(trace, x)]
        weights = [w + alpha * (delta + value - old_value) * e
                   - alpha * (value - old_value) * v
                   for w, e, v in zip(weights, trace, x)]
        old_value = next_value  # Save the PRE-update prediction at xp.
        history.append(weights[:])
    return history


def online_forward_view(features, rewards, alpha=0.1, gamma=0.9, lam=0.8, initial=None):
    """Slow independent reference: recompute all interim lambda-return updates."""
    initial = list(initial) if initial is not None else [0.0] * len(features[0])
    history = [initial[:]]
    for horizon in range(1, len(rewards) + 1):
        weights = initial[:]
        for start in range(horizon):
            target = 0.0
            for n in range(1, horizon - start + 1):
                end = start + n
                nreturn = sum(gamma ** j * rewards[start + j] for j in range(n))
                nreturn += gamma ** n * dot(history[end - 1], features[end])
                mixture = lam ** (n - 1)
                if end < horizon:
                    mixture *= 1 - lam
                target += mixture * nreturn
            weights = gradient_mc(weights, features[start], target, alpha)
        history.append(weights)
    return history


def traces_demo():
    features, rewards = [[1.0], [1.0], [0.0]], [0.0, 1.0]
    online = true_online_td(features, rewards, alpha=0.5, gamma=1.0, lam=1.0)
    forward = online_forward_view(features, rewards, alpha=0.5, gamma=1.0, lam=1.0)
    return {"true_online_history": online, "forward_history": forward,
            "accumulating_final": 1.0, "true_online_final": online[-1][0]}
```

测试使用多维、重叠且非二值的特征，同时改变步长和 lambda；对每个可见前缀核对参数。只检查最后一个权重可能漏掉中途时序错误。还检查 lambda 为零退化成 TD(0)、零步长不更新以及空前缀。

运行重复状态例子与前后向一致性检查

```sh
python3 approximation_textbook_lab.py traces
python3 approximation_textbook_lab.py test
```

<a id="lesson-branches"></a>

## 7 · 动作价值、流式内存与内部状态

将特征改成状态—动作特征，并按 Sarsa 的实际下一动作计算 bootstrap，可以构造动作价值的 true-online 版本。离策略时还涉及重要性比、trace 截断或其他校正；不能把在策略公式直接用于任意旧数据。

n-step 在线更新通常保留最近 n 步的奖励、状态或特征。线性 trace 使用 $O(d)$ 递归内存，不随生命期长度增长；但稀疏特征的 trace 可能逐渐变稠密。截断小 trace 可以节约计算，也改变精确更新。

资格迹保存“过去哪些参数方向应对当前误差负责”。Agent state 保存“为当前预测或决策应记住什么历史”。两者并不相同。递归神经状态的参数变化还需要通过状态转移传播导数；把线性特征 trace 叫作完整 RTRL 或 BPTT，会混淆两种信用路径。

持续变化时，较长 trace 可以更快传递延迟反馈，也可能跨越动力学变化而把新误差作用于旧情境。实验需要改变奖励延迟和环境变化频率，记录恢复速度与 trace 范数，不能只在固定短回合中选一个 lambda。

<a id="lesson-check"></a>

## 8 · 练习与答案

- 为什么有限回合 lambda-return 的最后项没有乘 $1-\lambda$？答案：它必须吸收所有更长回报的剩余几何权重，否则权重和不足 1。
- 能否把 $v_{\rm old}$ 存成更新后对下一状态的预测？答案：不能；true-online 修正需要更新前计算的下一预测。
- 普通累积迹与冻结权重前向更新相等，为何在线仍不同？答案：冻结权重推导中所有价值和梯度不变，在线更新破坏了这一前提。
- $\lambda=0$ 时 Dutch trace 和修正如何退化？答案：$e_t=x_t$，两个预测变化修正相抵，剩下普通 TD(0)。



<a id="chapter-code"></a>

## 下载与运行

标准库教学实现。命令运行本章的可解析小问题和全文件的公式测试；不依赖深度学习库，不等同于论文中的大规模实验。

[下载 approximation_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/approximation_textbook_lab.py)

```sh
python3 approximation_textbook_lab.py traces
python3 approximation_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：Part II 第 9–13 章。原书建立函数逼近预测、控制、离策略稳定性、资格迹与策略梯度的共同框架。

- [van Seijen et al. · True Online Temporal-Difference Learning](https://jmlr.org/papers/v17/15-599.html)：在线前向定义、Dutch trace 和精确等价的原始期刊论文；Algorithm 2 对应本文的常数步长更新。

- [Mahmood · True-online TD random MDP experiments](https://github.com/armahmood/totd-rndmdp-experiments)：论文作者的随机 MDP 实验代码，包含 accumulating、replacing 与 true-online 对照。先读 pysrc 和 pysrctest，再查看完整实验脚本。本文没有重跑这些大批量实验。

<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)
- [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)
- [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)
- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)

对应原始材料：第 12 章：Eligibility Traces。本文为原创讲解，原书、论文与上游代码保留各自许可。
