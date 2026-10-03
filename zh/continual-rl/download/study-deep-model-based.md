# 模型学习与规划：MPC、短模型 rollout 和潜在想象

模型在哪里进入决策，预测误差又怎样变成控制误差？

## 本章内容

- 由监督建模到有限时域 MPC。
- 推导多步误差传播与短 rollout 的动机。
- 区分 MPC、MBPO 和 Dreamer 的模型使用位置。

<a id="problem-definition"></a>

## 本章的问题定义

从真实转移学习后果模型，再决定模型计算在训练或行动的哪个位置发挥作用。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 模型 $\hat p_\phi$、规划时域 $H$、模型计算预算 $B$、终点价值 $\hat V$。

### 需要求解的对象

用模型辅助真实控制；模型预测准确与控制表现分别衡量。

### 信息与数据权限

模型只在训练数据覆盖范围内获得直接约束；策略可能主动进入模型错误区域。

$$
\max_{a_{0:H-1}}\mathbb E_{\hat p_\phi}\!\left[\sum_{k=0}^{H-1}\gamma^kR_{k+1}+\gamma^H\hat V(S_H)\right]
$$

这是 MPC 的模型内开环序列目标，执行第一个动作后重规划。MBPO 与 Dreamer 使用模型的位置不同，不都求这个动作序列问题。

### 成立条件与解的含义

- 有限模型预算；模型奖励、终止及动作语义与真实系统一致。
- 模型误差界需说明在哪个状态动作分布上成立。

判断准则：分别控制模型质量、模型使用长度和额外计算，判断收益来自哪一部分。

### 适用边界

- 用一步预测误差低证明长程规划可靠。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)：模型学习是规划的输入问题；下游查询决定需要保留哪些后果。

- 组合不同学习问题 · [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)：模型存在之后仍需选择备份位置、深度和预算。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

规划越积极，越可能利用模型误差；长虚拟轨迹会累积分布偏移。

### 本章的核心思路

分别设计模型、使用方式与误差控制，不将所有模型方法视为同一个 rollout 算法。

1. [先确定模型学习对象](#lesson-notation)：预测状态、奖励和终止的目标必须与后续查询一致。

2. [用有限时域并反复重规划](#lesson-derive)：MPC 用真实新观测纠正计划，但仍依赖模型和尾值。

3. [限制模型展开的误差](#lesson-error)：短 rollout 用真实状态作起点；长度是误差与计算的折衷。

4. [把模型用于策略训练](#lesson-latent)：潜在想象通过学习策略传递模型收益，不等同于每次行动显式搜索。

结论与条件：精确模型的规划性质不覆盖有偏神经模型；控制错误需结合实际访问分布分析。

### 相关方法改变了什么

- MPC / Dyna：分别直接选择当前动作与更新可复用价值。

- MBPO / Dreamer：模型生成的数据用途、表示空间和策略优化路径不同。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 条件概率与期望

区分随机变量、一次样本与条件分布；可以使用 Bayes 法则和全期望公式。

### MDP 与价值

理解状态、动作、转移、奖励、策略和 Bellman 递推；学习数据可以来自不同策略。

### 优化与梯度

知道固定目标、参数梯度、约束和期望目标的含义；神经网络只是函数表示的一种选择。

<a id="lesson-setting"></a>

## 1 · 学会预测，不等于已经会规划

模型可以预测原状态、潜在状态、奖励、终止概率或某种技能的结果。Model-based 方法用这些预测来选择行动或训练价值与策略。仅增加一个辅助预测头，却不让它参与行动相关计算，不能据此判断已经获得模型规划的优势。神经网络、线性系统与表格模型都能承担这一角色。

$$
\hat P_\psi(s'\mid s,a),\quad \hat r_\psi(s,a),\quad \hat c_\psi(s,a)
$$

分别预测转移、奖励和继续概率。任务定义决定真正终止，日志截断不能自动作为零继续概率。

外部设计者选择观察表示、模型损失、规划深度、终端价值和计算预算。运行时 agent 可以从真实经验更新模型与策略。若使用设计者提供的精确模拟器，本页称它为规划 oracle，不将模型知识计作学习得到。

<a id="lesson-notation"></a>

## 2 · 从单步数据学习动力学

$$
L_{\rm model}(\psi)=-\mathbb E_{\mathcal D}\log\hat P_\psi(S'\mid S,A)
$$

最大似然训练概率模型。若使用固定方差 Gaussian，负 log-likelihood 的参数相关部分等价于均方误差。

$$
s_{t+1}=s_t+ba_t+\epsilon_t,\qquad\hat b=\frac{\sum_ta_t(s_{t+1}-s_t)}{\sum_ta_t^2}
$$

在零均值噪声、无未建模漂移且分母非零的标量例子中，最小二乘对 b 求导得到此解。没有动作激励时不能识别增益。

低训练误差只约束数据分布附近。优化器可能选择训练数据从未支持过的动作序列，专门利用模型的系统误差。留出数据要检查多步预测和规划实际访问的区域；多个相似网络的分歧小，也不能证明模型正确。

<a id="lesson-derive"></a>

## 3 · MPC 的有限时域目标与重新规划

$$
\hat s_{k+1}=\hat f_\psi(\hat s_k,a_k),\quad\hat s_0=s_t
$$

每轮规划以当前真实观察或估计状态为起点，内部向前生成候选轨迹。

$$
a_{0:H-1}^*=\arg\max_{a_{0:H-1}}\left[\sum_{k=0}^{H-1}\gamma^k\hat r_\psi(\hat s_k,a_k)+\gamma^H\hat V(\hat s_H)\right]
$$

有限时域优化中的终端价值近似更远的后果。它不是免费得到的真值，误差和训练来源都需要说明。

MPC 只执行最优序列的第一个动作。收到新观察后重新规划，而不是盲目执行整个序列。这样可纠正部分开放环误差，但不能自动保证稳定或安全。连续动作常用 CEM、梯度或采样优化；本核用有限候选枚举，便于精确核对。

如果规划本身每步消耗大量模型调用，真实交互样本少不等于总计算便宜。需要分别报告环境步、模型步、候选数、规划深度和墙钟时间。

<a id="lesson-error"></a>

## 4 · 为什么短模型 rollout 有时更可靠

$$
e_{k+1}\le L_f e_k+\epsilon_f,\quad e_0=0,\qquad e_H\le\epsilon_f\sum_{j=0}^{H-1}L_f^j
$$

在相同动作序列、统一单步误差界和 Lipschitz 条件下，插入真实动力学在预测状态处的值即可得到误差递推。

$$
|\hat G-G|\le\sum_{k=0}^{H-1}\gamma^k(\epsilon_r+L_r e_k)+\gamma^H L_Ve_H
$$

此式还假定奖励误差界、奖励和终端价值的 Lipschitz 条件，并使用相同终端价值函数；它不是任意像素模型的通用保证。

MBPO 从真实 replay 状态出发生成较短的模型轨迹，将这些合成样本用于 off-policy 策略学习。短分支缩短连续承受模型误差的距离，同时保留数据增广。起点覆盖不足、模型偏差和奖励模型错误仍然存在，因此 rollout 长度是一项需要验证的选择，不是越短越优的定理。

<a id="lesson-latent"></a>

## 5 · Dreamer 把模型计算用于想象中的策略学习

$$
h_t=f_\psi(h_{t-1},z_{t-1},a_{t-1}),\quad z_t\sim q_\psi(z_t\mid h_t,o_t),\quad z_{t+1}\sim p_\psi(z_{t+1}\mid h_{t+1})
$$

有真实观察时 posterior 编码状态；想象时使用 prior 预测下一潜在状态。确定性记忆与随机变量承担不同角色。

$$
\mathcal L_{\rm model}=-\mathbb E_q\sum_t\left[\log p_\psi(o_t\mid h_t,z_t)+\log p_\psi(r_t\mid h_t,z_t)-\beta D_{\rm KL}(q_\psi\Vert p_\psi)\right]
$$

这是解释重建、奖励与潜在一致性的简化变分目标，不是 DreamerV3 完整工程损失。实际版本还有继续预测、KL balancing、free bits 和尺度变换等设计。

$$
\begin{gathered}\hat G_t^\lambda=\hat r_t+\gamma\hat c_t\left[(1-\lambda)V(\hat s_{t+1})+\lambda\hat G_{t+1}^\lambda\right]\\\hat G_H^\lambda=V(\hat s_H)\end{gathered}
$$

想象轨迹中的多步目标，末端使用 critic。奖励与继续概率由模型产生，因此目标同时受模型和价值误差影响。

Dreamer 家族在学得的世界模型中训练 actor–critic，部署通常可直接执行 actor。MPC 则在当下选择动作时做序列优化。潜在模型不意味着完全无需观察重建，也不意味着任意 latent 就是控制充分状态；应查看具体版本训练哪些预测、哪些梯度流入模型。

<a id="lesson-algorithm"></a>

## 6 · 三个模型使用位置

**算法：本章把机制分别说明，不将三个名字当成同一算法的可互换实现。**

1. 收集真实转移，更新模型，并在留出数据上检查误差。
1. MPC：从当前状态生成候选序列，评价奖励和终端价值。
  1. 执行第一个动作，观察后重新规划。
1. MBPO：从真实 replay 状态分支出短模型轨迹。
  1. 混合真实和模型数据，按声明比例更新 off-policy agent。
1. Dreamer：由真实序列推断潜在状态，再在 prior 中想象。
  1. 在想象轨迹上训练 actor 和 critic。
1. 分别计数真实数据、模型生成、梯度更新与决策时间。

<a id="lesson-example"></a>

## 7 · 一维控制与错误模型

状态初值零，目标二，动作集合 $(-1,0,1)$，模型 $\hat s'=s+a$。两步成本为每个后继状态到目标的平方距离加 $0.1a^2$。序列 $(1,1)$ 得到状态 $(1,2)$，成本 $1+0.1+0+0.1=1.2$，优于 $(1,0)$ 的 2.1。MPC 此刻只执行第一个一。

若真实增益为 0.5，同一组三个动作一产生真实状态 $(0.5,1,1.5)$，模型却预测 $(1,2,3)$。误差是 $(0.5,1,1.5)$，恰好达到 $L_f=1,\epsilon_f=0.5$ 的累计界。重规划可以利用新状态，但若模型增益一直错误，偏差不会自动消失。

<a id="lesson-code"></a>

## 8 · 学模型、枚举规划与想象回报核

标量模型、精确枚举 MPC、误差递推与有限想象 λ-return；本核的想象区间内继续概率均为一，末端使用给定价值。

```python
def fit_scalar_gain(actions, increments):
    if len(actions) != len(increments) or not actions:
        raise ValueError('paired action and state-increment samples required')
    denominator = dot(actions, actions)
    if denominator == 0:
        raise ValueError('no excitation: gain is not identifiable')
    return dot(actions, increments)/denominator


def model_rollout(state, sequence, gain=1., drift=0.):
    states = []
    for action in sequence:
        state = state+gain*action+drift
        states.append(state)
    return states


def mpc_action(state, goal, actions=(-1., 0., 1.), horizon=2,
               gain=1., action_cost=.1, terminal_weight=0.):
    """Enumerate open-loop candidates; return only the first action to execute."""
    if horizon < 1 or not actions:
        raise ValueError('positive horizon and nonempty actions required')
    best = None
    for sequence in itertools.product(actions, repeat=horizon):
        states = model_rollout(state, sequence, gain)
        cost = sum((s-goal)**2+action_cost*a*a for s, a in zip(states, sequence))
        cost += terminal_weight*(states[-1]-goal)**2
        if best is None or cost < best[0]:
            best = cost, sequence, states
    return best[1][0], best


def rollout_error_bound(one_step_error, lipschitz, horizon):
    """Same actions, same initial state; error_{k+1} <= L error_k + epsilon."""
    if min(one_step_error, lipschitz, horizon) < 0:
        raise ValueError('nonnegative inputs required')
    error, errors = 0., []
    for _ in range(horizon):
        error = lipschitz*error+one_step_error
        errors.append(error)
    return errors


def imagined_lambda_return(rewards, values, gamma=.9, lam=.8):
    """values[k] = V(s_k); final value supplies the finite imagination tail."""
    if len(values) != len(rewards)+1:
        raise ValueError('one more state value than reward required')
    carry, returns = values[-1], []
    for k in reversed(range(len(rewards))):
        carry = rewards[k]+gamma*((1-lam)*values[k+1]+lam*carry)
        returns.append(carry)
    return list(reversed(returns))
```

运行 demo 得到两步最优候选与误差曲线，运行 test 检查最小二乘、规划序列及 λ 的两个极端。它不是完整 PETS、MBPO 或 Dreamer 复现。PETS 与 MBPO 作者仓库提供原论文工程；danijar/dreamerv3 明确是作者维护的重实现，不能冒充原内部实验快照。

<a id="lesson-branches"></a>

## 9 · 持续变化中的模型寿命

- 模型陈旧：动力学改变后，更多规划可能放大旧知识错误，而不是补偿它。
- 表示错误：低重建损失可能忽略控制相关变量；belief 与 latent 的充分性要单独检验。
- 未知继续概率：把任务截止和人为采样停止混为一类，会改变想象长度与价值。

CRL 可以研究模型何时失效、哪些可复用以及每步规划预算怎样分配。有效对照包括真实模型 oracle、冻结旧模型和在线更新模型，并保持总计算预算一致。模型能支持未来新目标，才与持续构建的可复用知识直接相关。

<a id="lesson-check"></a>

## 10 · 练习与答案

- 问：模型误差很小，长规划是否总是更好？答：不保证；误差累积、优化利用偏差和终端价值误差仍有影响。
- 问：MPC 必須执行完整最优动作序列吗？答：不必，标准 receding-horizon 只执行首动作并重新规划。
- 问：拟合标量增益时所有动作都是零，会得到什么？答：分母为零，数据无法识别增益，不能静默给出可信模型。
- 实验：保持候选动作不变，将模型增益改为 0.5，对照真实轨迹和原模型轨迹；区分模型学习带来的变化与增加规划候选数带来的变化。



<a id="chapter-code"></a>

## 下载与运行

标准库解析机制实验；不包含完整神经网络训练或大型 benchmark。

[下载 extended_foundations_lab.py](https://yingwen.io/zh/continual-rl/download/extended_foundations_lab.py)

```sh
python3 extended_foundations_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Chua et al. · Deep Reinforcement Learning in a Handful of Trials](https://arxiv.org/abs/1805.12114)：PETS 原文：概率模型与候选轨迹评估。

- [Chua et al. · PETS 作者代码](https://github.com/kchua/handful-of-trials)：概率动力学 ensemble 与 trajectory sampling 的原论文实现。

- [Janner et al. · When to Trust Your Model](https://arxiv.org/abs/1906.08253)：MBPO 的短分支 rollout、模型偏差与实践动机。

- [Janner · MBPO 原作者代码](https://github.com/jannerm/mbpo)：真实/模型 buffer、rollout schedule 和 SAC 工程。

- [Hafner et al. · Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)：DreamerV3 世界模型与想象控制。

- [Hafner · dreamerv3](https://github.com/danijar/dreamerv3)：作者维护的重实现；仓库公开说明与原内部实现的区别。

- [Kochenderfer、Wheeler、Wray · Algorithms for Decision Making](https://algorithmsbook.com/)：作者书站提供决策、信念状态、模型与规划教材及配套 Julia 代码入口。

- [UC Berkeley · CS 285](https://rail.eecs.berkeley.edu/deeprlcourse/)：原课程的 exploration、model-based RL 与 offline RL 讲义和视频入口；按问题专题阅读，不必按网络规模划分领域。

<a id="study-connections"></a>

## 与教材主线的衔接

本章是可按问题选择的并列研究分支，不要求先读完其他深度研究分支。

[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)
- [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)
- [Dyna 与模型学习](https://yingwen.io/zh/continual-rl/algorithms/dyna/)
- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)
- [持续学习的智能体架构](https://yingwen.io/zh/continual-rl/construction/architectures/)

对应原始材料：PETS；Janner et al.：MBPO；Hafner et al.：DreamerV3；Algorithms for Decision Making：planning。本文为原创讲解，原书、论文与上游代码保留各自许可。
