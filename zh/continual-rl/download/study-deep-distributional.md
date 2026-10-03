# 分布强化学习：Bellman 分布、分位数与风险目标

学习完整回报分布，与学习均值、评估风险和估计知识不确定性分别有什么关系？

## 本章内容

- 写出分布 Bellman 递推及其固定策略条件。
- 推导 categorical 投影和 quantile loss。
- 区分回报分布、参数后验和风险偏好。

<a id="problem-definition"></a>

## 本章的问题定义

固定策略下，不仅预测回报均值，还预测整个回报分布，再明确控制时如何使用它。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 随机回报 $Z^\pi(s,a)$、有限支撑点或分位数表示；$\mathcal L(Z)$ 表示分布。

### 需要求解的对象

逼近回报分布；是否改用风险目标是另一个选择。

### 信息与数据权限

样本同时包含环境与策略随机性；不能仅由回报分散程度推断知识不足。

$$
Z^\pi(s,a)\overset D=R+\gamma Z^\pi(S^{\prime},A^{\prime}),\qquad A^{\prime}\sim\pi(\cdot\mid S^{\prime})
$$

等号表示分布相等；右侧按环境核和未来随机性联合生成。均值控制仍按 $\mathbb E[Z]$ 选动作，CVaR 等风险准则会改变问题。

### 成立条件与解的含义

- 分布的矩、固定支撑投影和距离需明确。
- 固定策略分布算子的收缩不能未经证明外推为所有分布控制算子的收缩。

判断准则：分别报告分布拟合、均值策略收益及所声明风险指标；不能相互替代。

### 适用边界

- 把 return variance 当作后验 epistemic uncertainty。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 推广：放宽条件 · [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)：本章把固定策略预测对象由期望扩为分布，均值是其统计量；控制准则仍可保持期望。

- 组合不同学习问题 · [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)：本章的分布预测需与另选的评价准则组合。默认均值控制并未改变目标；只有额外采用尾部风险准则时才改变排序，并需检查时间一致性。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

Bellman 备份后的分布未必仍落在有限参数表示中。

### 本章的核心思路

保留分布递推，再明确如何投影或用分位回归近似它。

1. [定义分布版备份](#lesson-notation)：先固定策略，区分随机变量递推与其均值递推。

2. [在离散支撑上重新分配概率质量](#lesson-derive)：Categorical 投影的边界与质量守恒是核心检查。

3. [改用分位数定位分布](#lesson-quantiles)：分位回归估计位置，不使用同一套概率质量参数。

结论与条件：固定策略下的理论、有限表示误差和控制改进是三层不同结论。

### 相关方法改变了什么

- C51 / quantile methods：分别固定值支撑并学习质量，或固定分位水平并学习位置。

- 均值 / 风险控制：前者改变预测工具，后者额外改变策略评价。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 条件概率与期望

区分随机变量、一次样本与条件分布；可以使用 Bayes 法则和全期望公式。

### MDP 与价值

理解状态、动作、转移、奖励、策略和 Bellman 递推；学习数据可以来自不同策略。

### 优化与梯度

知道固定目标、参数梯度、约束和期望目标的含义；神经网络只是函数表示的一种选择。

<a id="lesson-setting"></a>

## 1 · 同一均值可以来自不同的回报分布

$$
Z^\pi(s,a)\overset{D}{=}\sum_{t=0}^\infty\gamma^tR_{t+1}\mid S_0=s,A_0=a,\qquad Q^\pi(s,a)=\mathbb E Z^\pi(s,a)
$$

分布来自环境、策略及初始条件下的随机轨迹。等分布符号不表示两个实际采样回报逐点相同。

确定得到二，与一半概率得到零、一半概率得到四，均值都是二。仅预测均值无法区分二者。分布学习可以提供更丰富的预测目标，但控制器若仍按均值选动作，优化目标依然是期望收益，不自动变成风险厌恶。

环境已完全已知时回报仍可随机，因此回报分布的宽度不等于模型的认识不确定性。外部设计者选择分布表示、支持范围以及决策采用均值还是风险泛函；运行时 agent 学习的是这一约定下的预测。

<a id="lesson-notation"></a>

## 2 · 条件分布的 Bellman 递推

$$
(\mathcal T^\pi Z)(s,a)\overset{D}{=}R+\gamma Z(S',A'),\quad (R,S')\sim P(\cdot,\cdot\mid s,a),\quad A'\sim\pi(\cdot\mid S')
$$

先采样即时奖励与后继状态，再按策略采样后继动作，最后按相应后继分布采样回报。奖励与后继状态的相关性需要保留。

对两边取期望即可恢复通常的 Bellman 方程。这是均值方法嵌入分布方法的方式。分布递推还保留高阶结构；把每个状态动作只存成一个均值，会在这一步丢失结构。

$$
\bar W_p(\mathcal T^\pi Z_1,\mathcal T^\pi Z_2)\le\gamma\bar W_p(Z_1,Z_2),\qquad\bar W_p=\sup_{s,a}W_p
$$

在有限 p 阶矩等条件下，固定策略分布算子在相应最大 Wasserstein 距离中压缩。控制时策略随估计变化，不能直接照搬这个结论。

标准 C51 或 QR-DQN 通常用预测分布的均值选择贪心动作，再备份被选动作的整条分布。最优期望价值可以唯一，但最优策略的回报分布不必唯一；这正是控制分析与固定策略预测不同的原因之一。

<a id="lesson-derive"></a>

## 3 · Categorical 表示为什么需要投影

$$
Z_\theta(s,a)=\sum_{i=0}^{N-1}p_{\theta,i}(s,a)\,\delta_{z_i},\quad z_i=v_{\min}+i\Delta z,\quad\sum_i p_{\theta,i}=1
$$

位置固定、概率可学习；这里的 Dirac 是点质量，不是 TD error。网络可用 softmax 保证归一化。

$$
\tilde z_j=\operatorname{clip}(r+\gamma(1-d)z_j,v_{\min},v_{\max}),\quad b_j=\frac{\tilde z_j-v_{\min}}{\Delta z}
$$

自举后的点通常不落在原网格上，因此要把概率质量投影回固定支持。

令 $\ell_j=\lfloor b_j\rfloor,u_j=\lceil b_j\rceil$。若二者不同，向下格点加 $p_j(u_j-b_j)$，向上格点加 $p_j(b_j-\ell_j)$。若相同，全部质量都加到该格点；直接用两个插值权重会把整数点的质量错误清零。

$$
L_{\rm cat}(\theta)=-\sum_i m_i\log p_{\theta,i}(s,a)
$$

投影概率 m 在当前反传中固定。该 cross-entropy 对预测概率求导，不是对采样奖励或后继 argmax 求导。

支持外的回报被裁剪，可能改变均值。即便投影步骤数值正确，过窄支持也会造成系统误差；长生命期的奖励尺度变化尤其需要检查这一点。

<a id="lesson-quantiles"></a>

## 4 · 分位数回归与均值决策

$$
F_Z^{-1}(\tau)=\inf\{z:F_Z(z)\ge\tau\},\qquad\tau_i=\frac{i+\tfrac12}{N}
$$

QR 表示将相同质量放在可学习的位置，位置对应不同分位水平，与固定位置、学习质量的 categorical 表示相反。

$$
\rho_\tau(u)=u\bigl(\tau-\mathbf1[u<0]\bigr),\qquad\partial_\theta\mathbb E\rho_\tau(Y-\theta)=F_Y(\theta)-\tau
$$

避开质量点时是普通导数；质量点处使用次梯度。正负误差权重不对称，使最优位置落在 τ 分位数。

$$
L_{\rm QR}=\frac1{N^2}\sum_{i,j}\rho_{\tau_i}(r+\gamma(1-d)\theta_j^-(s',a^*)-\theta_i(s,a))
$$

每个当前分位数与所有目标分位数配对。本页数值核使用 pinball loss；QR-DQN 原文实践使用平滑的 quantile Huber 变体。

QR-DQN 的平均分位点近似均值，用于常规贪心选择。IQN 把分位水平也作为网络输入，可以采样不同水平并改变分位权重。改变权重为风险敏感控制时，已经改变决策准则，不能把这种变化只归因为预测更准确。

<a id="lesson-risk"></a>

## 5 · 尾部风险不是认识不确定性

$$
\operatorname{CVaR}_{\eta}^{\rm lower}(Z)=\frac1\eta\int_0^\eta F_Z^{-1}(u)\,du,\qquad0<\eta\le1
$$

本章将收益的下尾平均称为 lower CVaR；有些文献用损失上尾，符号与最优化方向需要转换。

收益二的确定策略，下尾 CVaR 仍是二；零或四各半的策略，在最差一半的 CVaR 为零。均值相同，风险偏好却给出不同选择。对于多步风险目标，还要区分初始时刻的全回报风险与递归风险度量；逐状态贪心最大化一个 CVaR 预测不自动解决时间一致性。

<a id="lesson-algorithm"></a>

## 6 · 两条分布更新的精确步骤

**算法：预测对象、训练损失和行动准则应分别记录。**

1. 从真实或 replay 数据得到 $(s,a,r,s',d)$。
1. 按声明的准则选后继动作；标准版本使用分布均值。
1. Categorical：平移并缩放 target atoms，裁剪支持，分配质量。
  1. 固定投影目标，对当前动作的概率做交叉熵更新。
1. Quantile：计算所有当前/目标分位数对的 TD residual。
  1. 固定目标，用相应分位水平的非对称损失更新位置。
1. 同步 target；独立报告均值收益、分布校准与风险指标。

<a id="lesson-example"></a>

## 7 · 投影、分位梯度和风险手算

支持 $(-1,0,1)$，概率 $(0.2,0.5,0.3)$，奖励 $0.5$、折扣 $0.5$。目标点为 $(0,0.5,1)$。中间点的一半质量分到零、一，所以投影概率是 $(0,0.45,0.55)$，总质量一，均值 $0.55$。若真终止且奖励零，全部质量应落在零。

数据为 $(0,2)$，$\tau=0.75,\theta=0.3$。两个 pinball 梯度平均为 $(1-0.75-0.75)/2=-0.25$，梯度下降提高位置。这里上分位数为二；下一章同一数据的 0.75 expectile 却为 1.5，两种不对称回归不能混用。

<a id="lesson-code"></a>

## 8 · 可运行分布算子

分类投影、pinball 次梯度和离散分布下尾 CVaR，包含整数点与部分质量的处理。

```python
def categorical_projection(reward, gamma, terminal, support, masses):
    probabilities(masses)
    n = len(support)
    if n < 2 or len(masses) != n or not 0 <= gamma <= 1:
        raise ValueError('two or more atoms and valid discount required')
    spacing = (support[-1]-support[0])/(n-1)
    if spacing <= 0 or any(not math.isclose(z, support[0]+i*spacing) for i, z in enumerate(support)):
        raise ValueError('equally spaced ascending support required')
    output = [0.]*n
    for atom, mass in zip(support, masses):
        target = reward+(0. if terminal else gamma*atom)
        target = min(support[-1], max(support[0], target))
        position = min(n-1, max(0., (target-support[0])/spacing))
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            output[lower] += mass
        else:
            output[lower] += mass*(upper-position)
            output[upper] += mass*(position-lower)
    return output


def quantile_loss_gradient(theta, samples, tau):
    if not samples or not 0 < tau < 1:
        raise ValueError('nonempty samples and interior quantile required')
    loss = sum((tau-float(y-theta < 0))*(y-theta) for y in samples)/len(samples)
    # At equality this selects one valid subgradient.
    gradient = sum(float(y < theta)-tau for y in samples)/len(samples)
    return loss, gradient


def lower_cvar(values, masses, fraction):
    probabilities(masses)
    if len(values) != len(masses) or not 0 < fraction <= 1:
        raise ValueError('valid mass fraction required')
    remaining, total = fraction, 0.
    for value, mass in sorted(zip(values, masses)):
        used = min(mass, remaining)
        total += used*value
        remaining -= used
        if remaining <= 1e-14:
            break
    return total/fraction
```

运行 test 检查质量守恒、支持裁剪、terminal 点质量、分位梯度的有限差分和 CVaR 部分原子。它不训练 C51 或 QR-DQN 网络。Dopamine 是 Google 研究团队提供的分布 RL 实现框架；阅读具体 agent 与配置，不能把框架默认值当作所有原论文的同一实验协议。

<a id="lesson-branches"></a>

## 9 · 适用边界与 CRL 衔接

- 预测分布变宽，既可能是环境风险上升，也可能是策略覆盖改变，不能直接解释成“不知道”。
- 固定支持的溢出、分位数交叉与低概率尾部样本不足，是不同的数值与统计问题。
- 依照期望行动的 distributional agent，并没有因此满足约束安全或最坏情况保证。

持续任务中可以为 GVF 预测整个未来信号分布，也可以研究奖励分布变化后的校准与尾部恢复。设计对照时固定行动准则，才能先测分布表示本身的影响；若同时更换风险目标，应把目标变化单独报告。

<a id="lesson-check"></a>

## 10 · 练习与答案

- 问：C51 的 51 表示 51 个不同策略吗？答：不是，表示固定支持上的概率原子数。
- 问：原子恰落在格点时两个插值权重都为零，怎么办？答：直接把全部质量放入该格点。
- 问：回报方差大就应探索更多吗？答：不一定。已知的环境随机性不等于可以由数据减少的知识不确定性。
- 实验：将奖励从 0.5 改为五而保持支持 [-1,1]。投影全部落在一，质量仍守恒但均值被截断；这说明“测试归一化通过”不足以确认建模正确。



<a id="chapter-code"></a>

## 下载与运行

标准库解析机制实验；不包含完整神经网络训练或大型 benchmark。

[下载 extended_foundations_lab.py](https://yingwen.io/zh/continual-rl/download/extended_foundations_lab.py)

```sh
python3 extended_foundations_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Bellemare、Dabney、Munos · A Distributional Perspective on Reinforcement Learning](https://proceedings.mlr.press/v70/bellemare17a.html)：固定策略分布算子、控制的差别与 categorical 算法。

- [Dabney et al. · Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)：QR-DQN、分位数投影和 quantile Huber loss。

- [Dabney et al. · Implicit Quantile Networks](https://arxiv.org/abs/1806.06923)：以分位水平为输入的分布表示。

- [Google · Dopamine](https://github.com/google/dopamine)：原研究团队框架，提供分布算法与 Atari 工程；非本页标准库核。

- [Bellemare、Dabney、Rowland · Distributional Reinforcement Learning](https://direct.mit.edu/books/oa-monograph/5590/Distributional-Reinforcement-Learning)：原作者体系教材，可按 Bellman 分布、投影、控制与 statistical functionals 深入。

<a id="study-connections"></a>

## 与教材主线的衔接

本章是可按问题选择的并列研究分支，不要求先读完其他深度研究分支。

[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [深度价值学习](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)
- [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)
- [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

对应原始材料：Bellemare、Dabney、Munos：Distributional Perspective；Dabney et al.：Quantile Regression；Distributional Reinforcement Learning。本文为原创讲解，原书、论文与上游代码保留各自许可。
