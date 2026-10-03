# 离线强化学习：数据支持、策略评估与保守改进

不能补采数据时，怎样判断策略好坏，怎样避免利用没有证据的高价值动作？

## 本章内容

- 把数据支持假设写入 IS 与 doubly robust 评估。
- 推导 CQL 正则和 IQL expectile 的不同作用。
- 分开学习策略、选择超参数与独立评估。

<a id="problem-definition"></a>

## 本章的问题定义

只能使用固定数据集，学习期间不能向环境补采反例。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 数据集 $\mathcal D$、其采集协议及可用行为概率；候选策略类 $\Pi$。

### 需要求解的对象

从数据中选出可支持的高价值策略，并估计它的真实性能。

### 信息与数据权限

未被数据支持的行为没有自动可识别的价值；行为概率未知时，某些 IS 估计器不能直接使用。

$$
\hat\pi=\mathcal A(\mathcal D)\in\Pi,\qquad J(\hat\pi)\ \text{under the real environment}
$$

外部控制目标没有变，改变的是数据权限。CQL 和 IQL 的损失是不同保守近似机制；它们不能为任意无覆盖区域创造可辨识性。

### 成立条件与解的含义

- 评价和改进结论需覆盖、函数可实现性或模型误差等条件。
- 策略选择与最终评价最好使用分离数据或适当选择校正。

判断准则：在允许的独立评估下报告真实收益及估计不确定性，并检查对数据外动作的依赖。

### 适用边界

- 仅凭固定数据确定任意未访问动作的真实效果。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 改变信息或数据协议 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：禁止继续交互，无法通过在线试验纠正夸大的动作价值。

- 组合不同学习问题 · [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)：历史数据可以帮助保留，但数据可见性不等于新情境覆盖。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

策略改进会偏好估计高但缺乏证据的动作，且无法在线纠正。

### 本章的核心思路

分开离线策略评价和保守策略改进，明确各自使用的信息假设。

1. [校正可评价行为的估计](#lesson-derive)：Doubly robust 结合预测与残差，但仍需其条件。

2. [限制缺乏证据的高值](#lesson-cql)：CQL 在 critic 中加入保守项，不等于已经知道数据外真值。

3. [在数据动作内组织改进](#lesson-iql)：IQL 避免直接对任意未见动作最大化，并通过优势加权提取策略。

结论与条件：保守性结论只在相应假设和优化条件下成立；无支持数据下存在根本不可识别性。

### 相关方法改变了什么

- OPE / offline control：前者评价给定策略，后者还根据数据选择策略。

- CQL / IQL：分别限制 critic 的数据外高值与在数据支持内近似策略改进。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 条件概率与期望

区分随机变量、一次样本与条件分布；可以使用 Bayes 法则和全期望公式。

### MDP 与价值

理解状态、动作、转移、奖励、策略和 Bellman 递推；学习数据可以来自不同策略。

### 优化与梯度

知道固定目标、参数梯度、约束和期望目标的含义；神经网络只是函数表示的一种选择。

<a id="lesson-setting"></a>

## 1 · 固定数据改变了纠错方式

Offline RL 用已经收集的数据学习策略，训练过程中不能任意向环境补采样。Off-policy 只说明行为策略与目标策略不同；在线 SAC 也可以 off-policy，因此两个概念不等价。设计者决定数据集、保留字段、覆盖和评估权限；算法不能从没有观测的行为结果中产生证据。

$$
\mathcal D=\{\tau_i\}_{i=1}^n,\quad\tau_i\sim\mu,\qquad J(\pi)=\mathbb E_\pi\sum_{t=0}^{T-1}\gamma^tR_{t+1}
$$

先考虑完整有限轨迹、相同环境、已记录行为概率与固定目标策略。数据缺失或策略依赖隐藏信息会改变可识别性。

$$
\pi(a\mid h)>0\ \Longrightarrow\ \mu(a\mid h)>0
$$

支持条件必须沿目标可能访问的历史成立。只在初始状态有共同动作支持还不够。

<a id="lesson-notation"></a>

## 2 · 重要性采样如何变换轨迹分布

$$
\rho_t=\frac{\pi(A_t\mid H_t)}{\mu(A_t\mid H_t)},\qquad w_{0:t}=\prod_{k=0}^{t}\rho_k
$$

在环境与初始分布相同的前提下，轨迹概率比中的环境项抵消，只保留动作概率比。

$$
\widehat J_{\rm PDIS}=\frac1n\sum_{i=1}^n\sum_{t=0}^{T-1}\gamma^t w^{(i)}_{0:t}R^{(i)}_{t+1}
$$

奖励只需用产生它之前的动作前缀校正。该形式在正确概率、支持和可积条件下无偏，但长乘积可能造成巨大方差。

加权归一化、截断比率可以改善有限样本稳定性，却通常引入偏差。有效样本量可以提示权重集中，但不能证明未覆盖区域没有风险。若目标策略和评价器是在同一数据上选出的，还存在自适应选择偏差，不能直接援引固定策略无偏结论。

<a id="lesson-derive"></a>

## 3 · Doubly robust：模型预测加残差校正

$$
\hat V_t(s)=\sum_a\pi(a\mid s)\hat Q_t(s,a),\quad\hat V_T=0
$$

有限时域价值可以随时间变化。Q 与 V 必须按同一目标策略对应，不能任取互不一致的两个网络。

$$
D_T=0,\qquad D_t=\hat V_t(S_t)+\rho_t\left[R_{t+1}+\gamma D_{t+1}-\hat Q_t(S_t,A_t)\right]
$$

从轨迹末端反向递推；最终使用 $D_0$。模型提供低方差基准，概率比校正模型残差。

正确重要性比使残差校正的期望补回模型偏差。展开递推后，模型项在条件期望中相消，从而得到目标回报。模型准确时，残差的条件均值接近零，方差往往降低；但并非每组样本中都比 IS 好。独立训练或交叉拟合 nuisance 模型，有助于避免训练误差与评价样本相关。

“双稳健”不是两个近似有误也必然正确。需要明确哪一组条件成立、模型与策略如何拟合，以及覆盖是否足够。本页用已知行为策略与固定模型展示估计器，不用它宣称有限数据下策略安全。

<a id="lesson-cql"></a>

## 4 · CQL：限制没有数据支持的高值

$$
L_Q=L_{\rm Bellman}+\alpha\left(\mathbb E_s\log\sum_a e^{Q(s,a)}-\mathbb E_{(s,a)\sim\mathcal D}Q(s,a)\right)
$$

这是离散动作 CQL 型核心正则；完整算法还有目标网络、采样分布与正则权重等条件。

$$
\frac{\partial L_{\rm reg}}{\partial Q(s,a)}=\operatorname{softmax}(Q(s,\cdot))_a-p_{\mathcal D}(a\mid s)
$$

正则压低高值但缺乏数据支持的动作，并相对提高数据动作。Bellman 项仍负责把价值连接到实际奖励。

相对保守不等于每个状态动作都是真实价值的逐点下界。原论文的界有具体分布、采样和权重条件。连续动作的 log-integral 通常用采样近似；不能直接枚举所有动作。本核只检验有限动作正则及其梯度。

<a id="lesson-iql"></a>

## 5 · IQL：在数据动作内近似策略改进

$$
\begin{gathered}L_V=\mathbb E_{\mathcal D}\left[|\tau-\mathbf1[u<0]|u^2\right]\\u=\operatorname{sg}(Q_{\bar\theta}(s,a))-V_\psi(s)\\\tau>\tfrac12\end{gathered}
$$

对偏高 Q 的残差赋更大权重，拟合数据动作价值的上 expectile。这里是非对称平方损失，不是上一章的分位数 pinball loss。

$$
\mathbb E\!\left[|\tau-\mathbf1[Q-v<0]|(Q-v)\right]=0
$$

对 v 求导并令其为零得到 expectile 的平衡方程。它随残差大小变化，不只看有多少样本在两侧。

$$
L_Q=\mathbb E_{\mathcal D}\left[\left(Q_\theta(s,a)-\operatorname{sg}\!\left(r+\gamma(1-d)V_\psi(s')\right)\right)^2\right]
$$

Q 的目标使用后继 V，不需要在未见动作上查询当前策略的 Q。V 与 Q 的更新顺序和目标副本应在实现中明确。

$$
L_\pi=-\mathbb E_{\mathcal D}\left[\operatorname{sg}\!\left(e^{\beta(Q_{\bar\theta}(s,a)-V_\psi(s))}\right)\log\pi_\phi(a\mid s)\right]
$$

actor 做优势加权行为克隆。本章 β 是逆温度；工程里常对权重裁剪以稳定训练，裁剪会改变精确加权目标。

IQL 避免训练目标中显式最大化未见动作，但函数逼近得到的策略仍可能泛化到数据外。不能把它理解成无条件的支持保证，也不能认为上 expectile 等于任意高质量未知动作的真实价值。

<a id="lesson-algorithm"></a>

## 6 · 学习与评估分成两条过程

**算法：离线模拟器评估若可用，是额外权限；它不是所有真实离线任务都具备的条件。**

1. 评估：固定待评策略及独立数据，检查行为概率与支持。
  1. 从末端计算 DR，或前向累乘 PDIS 权重。
  1. 汇总独立轨迹，报告区间、权重集中和数据限制。
1. IQL 学习：在 dataset batch 上先拟合 expectile V。
  1. 用固定后继 V 拟合 Q，再更新 Q 的 target 副本。
  1. 用 detached advantage 权重更新数据动作的 log likelihood。
1. CQL 学习：在 Bellman 回归上加入声明的保守正则。
1. 策略选择与最终评价使用分离的数据或明确的验证协议。

<a id="lesson-example"></a>

## 7 · 同一小数据解释三个公式

bandit 两动作行为概率均为 $0.5$，目标概率 $(0.8,0.2)$，确定奖励 $(1,3)$，目标收益为 $1.4$。IS 在动作零样本上给 $1.6$，在动作一样本上给 $1.2$，平均为 $1.4$。若模型完全准确，DR 的残差为零，每个样本都返回 $1.4$；模型全零时 DR 退化为 IS。

对同频 Q 数据 $(0,2)$，$\tau=0.75$，expectile 方程为 $0.75(2-v)=0.25v$，解为 $v=1.5$。取 $\beta=1$，分类 actor 的加权克隆概率与 $(e^{-1.5},e^{0.5})$ 成正比，因此高值动作概率约 $0.8808$。这不是直接执行确定 argmax。

若数据只含动作零，即使网络把动作一的 Q 写成十，本页精确表格加权克隆仍给动作一零概率；神经泛化时不能直接援引这条表格性质。

<a id="lesson-code"></a>

## 8 · 估计器与目标函数的可执行核

PDIS、顺序 DR、expectile 求根、表格加权克隆与 CQL 梯度。

```python
def importance_ratio(target, behavior):
    if not 0 <= target <= 1 or not 0 < behavior <= 1:
        raise ValueError('positive recorded behavior probability required')
    return target/behavior


def per_decision_is(rewards, ratios, gamma=1.):
    if len(rewards) != len(ratios):
        raise ValueError('one ratio per reward required')
    product, estimate = 1., 0.
    for t, (reward, ratio) in enumerate(zip(rewards, ratios)):
        product *= ratio
        estimate += gamma**t*product*reward
    return estimate


def sequential_dr(rewards, ratios, q_hat, v_hat, gamma=1.):
    """v_hat contains estimates at each state and a terminal zero."""
    n = len(rewards)
    if len(ratios) != n or len(q_hat) != n or len(v_hat) != n+1 or v_hat[-1] != 0:
        raise ValueError('complete finite trajectory and zero terminal value required')
    estimate = 0.
    for t in reversed(range(n)):
        estimate = v_hat[t]+ratios[t]*(rewards[t]+gamma*estimate-q_hat[t])
    return estimate


def expectile(values, tau, weights=None):
    if not values or not 0 < tau < 1:
        raise ValueError('nonempty values and interior expectile required')
    weights = [1.]*len(values) if weights is None else weights
    if len(weights) != len(values) or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError('nonnegative observation weights required')
    low, high = min(values), max(values)
    for _ in range(100):
        middle = (low+high)/2
        derivative_sign = sum(w*(tau if q >= middle else 1-tau)*(q-middle)
                              for q, w in zip(values, weights))
        if derivative_sign > 0:
            low = middle
        else:
            high = middle
    return (low+high)/2


def iql_weighted_actor(q_values, counts, tau=.75, inverse_temperature=1.):
    """Exact weighted behavioral-cloning solution for a one-state categorical actor."""
    value = expectile(q_values, tau, counts)
    logits = [inverse_temperature*(q-value) for q in q_values]
    offset = max(z for n, z in zip(counts, logits) if n > 0)
    weights = [n*math.exp(z-offset) if n > 0 else 0. for n, z in zip(counts, logits)]
    return value, [w/sum(weights) for w in weights]


def cql_penalty_gradient(q_values, data_probabilities):
    probabilities(data_probabilities)
    if len(q_values) != len(data_probabilities):
        raise ValueError('matching action dimensions required')
    maximum = max(q_values)
    exponentials = [math.exp(q-maximum) for q in q_values]
    normalizer = sum(exponentials)
    loss = maximum+math.log(normalizer)-dot(q_values, data_probabilities)
    return loss, [x/normalizer-p for x, p in zip(exponentials, data_probabilities)]
```

运行 test 检查 DR 与零模型 IS 的一致、准确模型的基准值、零行为概率拒绝、expectile 与 quantile 的差别以及 CQL 有限差分。源码不含完整 CQL/IQL 神经训练，也不对任意数据产生置信保证。IQL 作者仓库是 JAX 实现；其中 actor.py、critic.py 与 value_net.py 分别对应上述三类更新。

<a id="lesson-branches"></a>

## 9 · 数据限制与 CRL 的历史经验

- 隐藏混杂：行为者使用了数据中未记录的信息时，仅从观察字段估计行为概率可能不足。
- 未知终止：人工日志截断与真正终止混用会改变 Q 和 DR 的尾部。
- 持续变化：不同年份或阶段的数据可能来自不同动力学，普通 stationary OPE 的环境抵消不再成立。

CRL 中历史 replay 可以提供旧技能证据，但其时间戳、策略概率、传感器版本与任务条件同样重要。可分别比较覆盖不足和环境变化两种干预，避免把离线外推、知识遗忘与世界非平稳合并成一个指标。

<a id="lesson-check"></a>

## 10 · 练习与答案

- 问：已知目标动作的概率，是否就能做 IS？答：还需对应行为概率、支持和相同环境等条件。
- 问：DR 中用任意 V 与 Q 都能保持同样理论性质吗？答：不能；需要其目标策略关系及所用估计条件。
- 问：expectile 0.75 等于样本 75% 分位数吗？答：不等于，例子中分别为 1.5 和二。
- 实验：固定模型为零，给两步奖励 (1,2)、比率 (2,0.5)、折扣 0.9。PDIS 和 DR 均应得到 3.8；这验证代数，不证明数据覆盖。



<a id="chapter-code"></a>

## 下载与运行

标准库解析机制实验；不包含完整神经网络训练或大型 benchmark。

[下载 extended_foundations_lab.py](https://yingwen.io/zh/continual-rl/download/extended_foundations_lab.py)

```sh
python3 extended_foundations_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Jiang、Li · Doubly Robust Off-policy Value Evaluation](https://proceedings.mlr.press/v48/jiang16.html)：顺序 doubly robust 估计器、条件和方差分析。

- [Kumar et al. · Conservative Q-Learning](https://arxiv.org/abs/2006.04779)：保守价值目标与理论结论的适用条件。

- [Kumar · CQL 原作者代码](https://github.com/aviralkumar2907/CQL)：离散与连续动作实验工程；需对照具体配置。

- [Kostrikov、Nair、Levine · Implicit Q-Learning](https://arxiv.org/abs/2110.06169)：expectile 价值、隐式改进与加权行为克隆。

- [Kostrikov · IQL 原作者代码](https://github.com/ikostrikov/implicit_q_learning)：JAX 实现，含离线训练和在线微调入口。

- [UC Berkeley · CS 285](https://rail.eecs.berkeley.edu/deeprlcourse/)：原课程的 exploration、model-based RL 与 offline RL 讲义和视频入口；按问题专题阅读，不必按网络规模划分领域。

<a id="study-connections"></a>

## 与教材主线的衔接

本章是可按问题选择的并列研究分支，不要求先读完其他深度研究分支。

[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [深度价值学习](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)
- [知识保留与再适应](https://yingwen.io/zh/continual-rl/algorithms/retention/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

对应原始材料：Jiang、Li：Doubly Robust OPE；Kumar et al.：CQL；Kostrikov、Nair、Levine：IQL。本文为原创讲解，原书、论文与上游代码保留各自许可。
