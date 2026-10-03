# 持续控制与平均奖励

智能体没有自然回合终点时，怎样定义和学习长期控制目标？

## 本章内容

- 区分折扣价值、平均奖励率与差分价值。
- 从 Poisson 方程得到差分 TD 和 Sarsa 的两组同步更新。
- 说明 unichain、communicating 与函数逼近保证的边界。

<a id="problem-definition"></a>

## 本章的问题定义

交互没有自然终点，评价单位原始时间内的长期收益，而非折扣累计量。

### 给定条件与符号

- 平稳 MDP、策略类 $\Pi$、每一步原始奖励 $R_{t+1}$。
- 平均奖励 $g_\pi$ 与差分价值 $h_\pi$；近似动作价值使用 $q_w(s,a)$。

### 需要求解的对象

在规定策略类中提高平均奖励，同时估计相对未来价值。

### 信息与数据权限

只有连续样本流；人为日志分段不产生新的问题终止。

$$
\sup_{\pi\in\Pi}g_\pi,\qquad g_\pi=\lim_{T\to\infty}\frac1T\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}R_{t+1}\right]
$$

本章先讨论极限存在且与起点无关的情形。差分价值刻画有限暂态与长期奖励率的偏差，不是发散的无折扣总回报。

### 成立条件与解的含义

- 策略诱导过程需具有保证平均奖励定义适用的链结构。
- 差分价值只确定到加法常数，需要参考规范。

判断准则：真实奖励率、学习器内部奖励率估计、差分 Bellman 残差分别报告。

### 适用边界

- 用渐近奖励率概括所有有限生命期损失或不可逆风险。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)：本章从一般准则中选取极限存在且与起点无关的平均奖励设定；它增加适用条件，而非推广所有准则。

- 限制表示或采用近似 · [平均奖励与差分价值](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)：本章用共享特征近似差分动作价值并采用差分 Sarsa；相关平均奖励章还讨论表格预测、离策略与规划。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

无折扣累计回报可能发散，价值的共同偏移也不可辨识。

### 本章的核心思路

每步减去奖励率，学习相对价值，再把奖励率估计与价值更新耦合。

1. [从长期速率推导差分方程](#lesson-derive)：先定义中心化的回报对象，避免对发散总和直接求值。

2. [确定两个估计的更新时序](#average-algorithm)：差分 Sarsa 同时更新价值和奖励率；两者不应被误认成独立真值。

3. [检验目标是否真正相同](#average-objective)：构造折扣与速率排序不同的例子，阻止把一种目标的实验结论外推到另一种。

结论与条件：链结构和函数逼近条件决定可用结论；表格定理不能自动保证近似控制。

### 相关方法改变了什么

- 平均奖励 / 折扣回报：前者强调长期速率，后者为相同时间距离设置几何权重。

- 差分更新 / 折扣奖励中心化：有限折扣下的中心化参照不一定等于真实奖励率。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 持续交互

任务本身不要求在某个时间终止。训练日志可以分段，但分段不等于环境终止，也不自动提供无成本重置。

### 差分价值

先扣除长期奖励率，再比较从不同状态出发的暂态优势。差分价值允许任意共同加法常数，需要参考状态或其他归一化约定。

<a id="lesson-setting"></a>

## 1 · 长期速率与折扣和是不同目标

$$
g_\pi(s)=\lim_{T\to\infty}\frac1T\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}R_{t+1}\mid S_0=s\right].
$$

平均奖励率衡量每个真实时间步的长期收益。存在此极限以及是否依赖初始状态，都是问题的条件，而不是符号定义自动提供的性质。

固定策略若具有一个常返类、其余状态至多是暂态，通常称为 unichain。在有限状态、适当奖励条件下，不同初始状态最终进入同一常返类，长期奖励率可以相同。若策略产生多个互不连通的常返类，奖励率可能依赖起点，不能用一个标量概括所有状态。

Communicating 描述 MDP 层面的可达性：不同状态之间存在某些策略可以建立到达关系。它不等于每一个策略都只有一个常返类。相关最优平均奖励结果还需要区分策略类、探索条件和算法假设。实践中先检查有没有不可逆动作和吸收陷阱。

$$
\begin{aligned}v_\gamma^\pi(s)&=\mathbb E_\pi\sum_{t\ge0}\gamma^tR_{t+1},\\ g_\pi&=\lim_{\gamma\uparrow1}(1-\gamma)v_\gamma^\pi(s).\end{aligned}
$$

极限关系在适当有限遍历条件下成立，但固定 $\gamma<1$ 仍定义另一个目标；把它设得很接近 1 也会放大价值尺度并延长有效预测时域。

<a id="lesson-derive"></a>

## 2 · 从长期速率到差分 Bellman 方程

平均奖励率只描述长期斜率，无法区分获得同一速率但暂态体验不同的状态。差分价值补充这种信息。为避免在周期链上把一个不收敛的普通无限和当作定义，本章直接用 Poisson 方程定义差分价值，并固定一个参考值。

$$
g_\pi\mathbf1+h_\pi=r_\pi+P_\pi h_\pi,\qquad h_\pi(s_{\rm ref})=0.
$$

在适当 unichain 条件下，$g_\pi$ 是公共奖励率，$h_\pi$ 在加法常数之外确定。式子表示：当前位置的暂态优势，等于当前奖励超出平均的部分，加上下个状态的优势。

$$
\delta_t=R_{t+1}-\bar r_t+x_{t+1}^\top w_t-x_t^\top w_t.
$$

用在线估计 $\bar r_t$ 替代未知 $g_\pi$，再用线性近似替代 $h_\pi$，得到样本残差。这里不是在折扣 TD 中简单令 $\gamma=1$；新增的奖励率项负责去掉随时间累积的趋势。

差分价值的数值可以为负。它不表示负的长期收益，只表示相对参考状态的暂态劣势。将所有差分价值加上同一常数不会改变 TD 残差，也不会改变按动作价值差选择动作的结果。原始参数均值漂移和价值差失真应分别诊断。

<a id="average-algorithm"></a>

## 3 · 差分半梯度 Sarsa 的更新

$$
\begin{aligned}\delta_t&=R_{t+1}-\bar r_t+\hat q(S_{t+1},A_{t+1},w_t)-\hat q(S_t,A_t,w_t),\\ w_{t+1}&=w_t+\alpha\delta_t\nabla_w\hat q(S_t,A_t,w_t),\\ \bar r_{t+1}&=\bar r_t+\beta\delta_t.\end{aligned}
$$

两组更新使用同一个旧参数下的误差。先更新奖励率再重新计算价值误差，会得到不同算法。

**算法：差分半梯度 Sarsa**

1. 初始化动作价值权重与奖励率估计，按探索策略选择动作
1. 执行动作，得到下一状态与奖励；按旧策略选择下一动作
1. 保存两个旧动作价值，计算 $\delta=R-\bar r+q'-q$
1. 用这一个 $\delta$ 更新权重和奖励率；推进到已选的下一动作
1. 持续记录真实奖励，不因报告窗口结束而重置价值或奖励率

在一般动态控制过程中，策略、状态分布和奖励率估计都在变化。上式是教材中的算法构造，不能把固定策略表格评价或特定 Differential Q-learning 的定理直接作为它对任意线性或神经表示的收敛保证。

若下一动作项改为 $\max_a q(S_{t+1},a)$，便进入差分 Q-learning 的控制路线。RVI 方法则通过参考函数减去一个标量来规范相对价值。它们都处理平均奖励问题，但规范方式和收敛条件并不相同，不能混用变量更新。

<a id="lesson-example"></a>

## 4 · 手算：周期链与奖励率同步更新

A 到 B 的奖励为 0，B 到 A 的奖励为 2。两个状态确定性交替，每两个时间步共得到 2，所以平均奖励率为 1。选择 $h(A)=0$，则 A 的方程给出 $h(B)=1$；B 的方程也成立。这个周期例子有平均奖励和 Poisson 解，却不应直接假设普通中心化无穷回报逐点收敛。

$$
1+0=0+1,\qquad1+1=2+0.
$$

这是两个状态的差分 Bellman 方程。

现在令估计 $w=(0,1)$、$\bar r=0.5$，观察 A 到 B。步长 $\alpha=0.1,\beta=0.2$。误差为 $0-0.5+1-0=0.5$。因此 $w_A^+=0.05,w_B^+=1,\bar r^+=0.6$。

若错误地先令奖励率变为 0.6，再给价值计算误差，就会把误差变为 0.4，得到 $w_A^+=0.04$。这就是一项能区分实现时序的单步测试；仅看长时间曲线可能不容易发现。

<a id="average-objective"></a>

## 5 · 折扣目标为什么可能改变动作排序

设初始决策可以进入两个不可逆分支。分支甲先支付 100，之后每步奖励为 2；分支乙无初始成本，之后每步奖励为 1。长期平均收益分别为 2 和 1，平均奖励偏好甲。有限折扣却可能偏好乙，因为初始成本不会被按时间平均稀释。

$$
\begin{aligned}v_\gamma(\text{甲})&=-100+\frac{2\gamma}{1-\gamma},\\ v_\gamma(\text{乙})&=\frac{\gamma}{1-\gamma},\\ v_\gamma(\text{甲})>v_\gamma(\text{乙})&\iff\gamma>\frac{100}{101}.\end{aligned}
$$

这是目标差异的反例，不是满足 communicating 假设的收敛示例。不可逆分支也提醒我们，最优长期率可能掩盖前期成本和安全代价。

平均奖励也有局限。有限生命期里付出巨额不可恢复成本可能不可接受，即使渐近奖励率更高。真正研究单生命期系统时，需要同时记录累计奖励、坏事件概率与恢复时间，不能只报告理论上的无限时域平均率。

如果动作或 option 的持续时间不同，应按真实时间归一化。一个持续十步的宏动作获得奖励 5，不能与持续一步获得奖励 2 按“每个决策点的奖励”直接比较。半马尔可夫残差中要扣除奖励率乘持续时间，而非只减一次奖励率。

<a id="lesson-code"></a>

## 6 · 可运行的差分更新与周期实验

差分 Sarsa 更新核与两状态固定策略评价

```python
def differential_sarsa(weights, rate, x, reward, xp, alpha, beta):
    delta = reward - rate + dot(weights, xp) - dot(weights, x)
    # One old delta drives BOTH updates. There is no terminal discount mask.
    next_weights = [w + alpha * delta * value for w, value in zip(weights, x)]
    next_rate = rate + beta * delta
    return next_weights, next_rate, delta


def average_demo():
    # A -> B gives 0; B -> A gives 2. Exact gain=1, h(B)-h(A)=1.
    weights, rate = [0.0, 0.0], 0.0
    for t in range(12000):
        s = t % 2
        x, xp = [0.0, 0.0], [0.0, 0.0]
        x[s], xp[1 - s] = 1.0, 1.0
        weights, rate, _ = differential_sarsa(weights, rate, x, 2.0 * s, xp, 0.05, 0.01)
    return {"gain": rate, "relative_values": [0.0, weights[1] - weights[0]],
            "exact_gain": 1.0, "exact_relative_values": [0.0, 1.0]}
```

示例每个状态只有一个动作，因此它验证的是差分评价，而不是已解决非平稳控制。运行后奖励率接近 1，相对价值接近 [0,1]。测试还检查给价值共同加常数后的残差不变，以及奖励率与价值是否确实共用旧误差。

运行差分评价及边界测试

```sh
python3 approximation_textbook_lab.py average-control
python3 approximation_textbook_lab.py test
```

<a id="lesson-branches"></a>

## 7 · 非平稳性、时间尺度与实验指标

固定奖励率是平稳问题中的对象。环境发生变化时，常数步长奖励率估计可以追踪局部变化，但它不再估计同一个全局常数。奖励率跟踪过慢会把均值变化误认为状态优势；过快则可能吸收本应由价值解释的差异。两种步长需要共同研究。

控制实验应保留完整交互轨迹，报告总体每步奖励、变化前后窗口收益、恢复时间和失败次数。把轨迹人为切成回合再给每回合相同权重，可能改变时间加权；删除崩溃段则会夸大奖励率。

与 CRL 主线的连接有两层：平均奖励规定长期控制目标，流式学习规定经验与计算的使用权限。使用平均奖励并不自动满足有限内存；严格流式也不要求采用平均奖励目标。研究时应把这两个维度分开。

<a id="lesson-check"></a>

## 8 · 练习与答案

- 只把折扣 TD 的折扣设为 1，会得到差分 TD 吗？答案：不会；还需要奖励率估计来消除线性增长趋势。
- 周期链能否具有平均奖励？答案：可以。时间平均与状态分布逐点收敛是不同性质，本章交替链就是例子。
- 奖励率已经正确，差分价值是否必然正确？答案：不必然。奖励率只有一个标量，无法描述不同状态的暂态优势。
- 持续 4 步、总奖励 7 的 option，在奖励率估计为 1.5 时，其未加终点价值的差分目标是多少？答案：$7-4\times1.5=1$。



<a id="chapter-code"></a>

## 下载与运行

标准库教学实现。命令运行本章的可解析小问题和全文件的公式测试；不依赖深度学习库，不等同于论文中的大规模实验。

[下载 approximation_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/approximation_textbook_lab.py)

```sh
python3 approximation_textbook_lab.py average-control
python3 approximation_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：Part II 第 9–13 章。原书建立函数逼近预测、控制、离策略稳定性、资格迹与策略梯度的共同框架。

- [Wan, Naik & Sutton · Learning and Planning in Average-Reward MDPs](https://proceedings.mlr.press/v139/wan21a.html)：Differential Q-learning 等算法的原始论文。其表格控制理论不应直接替代任意函数逼近的保证。

<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [平均奖励与差分价值](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)
- [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)
- [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

对应原始材料：第 10 章：平均奖励与 differential semi-gradient Sarsa。本文为原创讲解，原书、论文与上游代码保留各自许可。
