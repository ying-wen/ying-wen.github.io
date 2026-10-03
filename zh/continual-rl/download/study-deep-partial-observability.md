# 不完全可观测：信念状态、信息行动与递归记忆

当前观察不能决定未来时，智能体应记住什么，信息又如何影响行动？

## 本章内容

- 从历史条件分布推导 Bayes filter。
- 把信息获取的价值纳入 Bellman 决策。
- 区分精确信念、学习的 recurrent state 和训练时的隐状态权限。

<a id="problem-definition"></a>

## 本章的问题定义

当前观测不能确定未来。智能体必须利用历史形成足够的决策信息。

### 给定条件与符号

- 有限潜在状态集 $\mathcal X$、观测集 $\mathcal O$、动作集 $\mathcal A$、奖励域 $\mathcal R\subset\mathbb R$；$X_t,O_t,A_t,R_{t+1}$ 为相应变量。平稳联合核为 $K(x',B_o,B_r\mid x,a)=\Pr(X_{t+1}=x',O_{t+1}\in B_o,R_{t+1}\in B_r\mid X_t=x,A_t=a)$，其中 $B_o,B_r$ 为可测观测与奖励事件；离散情形可写为 $K(x',o,r\mid x,a)$。
- 历史 $H_t=(O_0,A_0,R_1,O_1,\ldots,A_{t-1},R_t,O_t)$；给定初始观测律 $\nu_0$ 与条件初始信念 $b_0(x\mid O_0)$，它们共同指定 $(X_0,O_0)$ 的联合分布。奖励有界，$0\le\gamma<1$。
- $\Pi_H$ 是非空的允许可测因果策略类，$\pi_t(\cdot\mid H_t)$ 只能使用已到达的信息。已知 $K,b_0$ 时可维护 $b_t(x)=\Pr(X_t=x\mid H_t)$；未知模型需另给模型族和信息条件，联合估计模型与状态，或学习近似摘要。

### 需要求解的对象

在 $\Pi_H$ 中求使折扣收益接近上确界的策略，并构造支持它的信息状态；不假定任意受限策略类的最优者必然存在。

### 信息与数据权限

执行时不可访问潜在真状态；精确 filter 的参照问题允许查询正确模型与初始律。未知模型版本不提供这项权限，须声明模型族、先验或识别数据。奖励如含状态信息，也必须纳入历史和后验。

$$
J(\pi)=\mathbb E_{\nu_0,b_0,K,\pi}\!\left[\sum_{t\ge0}\gamma^tR_{t+1}\right],\qquad J(\hat\pi)\ge\sup_{\pi\in\Pi_H}J(\pi)-\varepsilon,\quad\varepsilon>0
$$

这是指定初始化与策略类下的历史条件控制，$A_t\sim\pi_t(\cdot\mid H_t)$。正确已知模型下 belief 对未来预测充分；若策略类另有限制，其信息或资源约束也须保留，不能自动换成任意 belief 策略。

### 成立条件与解的含义

- 潜在 Markov 条件是给定 $X_t,A_t$ 后，下一潜在状态、观测和奖励的联合条件律不再依赖更早历史，并由同一 $K$ 给出。精确 filter 使用正确模型与初始律；离散观测的证据概率为正，连续情形使用相应密度或正规条件分布。
- 本章正文的分解式 filter 另假定奖励不提供额外状态信息；否则必须将奖励纳入联合似然。未知模型可在指定先验下维护状态与模型参数的联合后验，并非只能使用近似记忆。
- 有限记忆及计算限制应进入 $\Pi_H$ 或完整实现集合；近似摘要的控制损失需另检验。

判断准则：除状态预测误差，还检查不同历史被合并后是否仍能选择正确动作。

### 适用边界

- 以记忆长度或重构准确率直接证明控制充分性。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)：本章将隐藏 Markov 模型的 belief 构造接到历史控制；一般状态构造章已包含部分可观测过程，本章不是再次放宽它的假设。

- 组合不同学习问题 · [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)：本章的信息状态控制可与探索结合：动作可能通过改善信息而有价值，即使即时奖励较低。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

相同观测可能来自需要不同动作的情境，反应式策略无法区分。

### 本章的核心思路

先用已知模型展示准确后验如何递推，再定位学习记忆的近似位置。

1. [明确更新前后有什么信息](#lesson-notation)：动作先改变潜在状态，新观测和奖励再修正后验。

2. [把历史压缩为 belief](#lesson-derive)：Bayes 递推给出充分状态的参照，而非要求神经网络储存完整历史。

3. [用决策差异评价信息价值](#lesson-information)：更准确识别状态只有改变未来行动或预测时才产生对应用途。

结论与条件：belief 充分性有正确模型条件；近似 recurrent state 必须通过下游任务检验。

### 相关方法改变了什么

- Bayes filter / recurrent network：前者依据指定生成模型精确递推，后者从目标和数据学习摘要。

- 观测重构 / 控制充分性：优化的误差对象不同；重构很好仍可能遗漏决策关键变量。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 条件概率与期望

区分随机变量、一次样本与条件分布；可以使用 Bayes 法则和全期望公式。

### MDP 与价值

理解状态、动作、转移、奖励、策略和 Bellman 递推；学习数据可以来自不同策略。

### 优化与梯度

知道固定目标、参数梯度、约束和期望目标的含义；神经网络只是函数表示的一种选择。

<a id="lesson-setting"></a>

## 1 · 观察不等于状态

设环境隐状态为 $S_t$，智能体收到 $O_t$。同一幅图像可能来自不同速度、物体遮挡或任务阶段。即使环境隐状态满足 Markov 性，观察序列也未必满足。POMDP 是信息条件；表格 belief、线性滤波器和深度 recurrent 网络都可以处理它。它不是深度 RL 之后才出现的阶段。

$$
H_t=(O_0,A_0,R_1,\ldots,A_{t-1},R_t,O_t),\qquad b_t(s)=\Pr(S_t=s\mid H_t)
$$

历史是已经到达的信息；belief 是给定模型时对当前隐状态的后验分布，不是对网络参数的置信区间。

以下有限状态推导假定转移与观测模型已知，且奖励没有提供额外状态信息。如果奖励也携带信息，应将它并入观测似然或使用联合转移—奖励模型。设计者提供了状态假设、传感器接口和模型结构；运行时智能体更新 belief 并选择行动。

<a id="lesson-notation"></a>

## 2 · 动作之后，先预测再条件化

$$
T_a(s,s')=\Pr(S_{t+1}=s'\mid S_t=s,A_t=a),\qquad O_a(o\mid s')=\Pr(O_{t+1}=o\mid S_{t+1}=s',A_t=a)
$$

转移矩阵每一行和为一。观测似然按收到的观测取值，它不是在状态维度归一化的 posterior。

$$
\bar b_{t+1}(s')=\sum_sT_a(s,s')b_t(s)
$$

用全概率公式将旧状态不确定性传播到下一状态。这一步尚未使用新观察。

$$
\Pr(o\mid b_t,a)=\sum_{s'}O_a(o\mid s')\bar b_{t+1}(s')
$$

这是新观测的证据概率，随后用作归一化常数。

<a id="lesson-derive"></a>

## 3 · Bayes 递推与 belief MDP

$$
b_{t+1}(s')=\frac{O_a(o\mid s')\sum_sT_a(s,s')b_t(s)}{\sum_{\tilde s}O_a(o\mid\tilde s)\sum_sT_a(s,\tilde s)b_t(s)}=:B(b_t,a,o)(s')
$$

分子是下一状态与观测的联合条件概率，分母消去状态得到观测概率。分母为零时模型认为该观测不可能，不能静默归一化。

给定正确模型与初始分布，belief 对未来控制是充分的：已知当前 belief 和新动作，即可预测后继 belief 及奖励，不必再次读取整段历史。状态集合变成概率单纯形，通常连续，即使原隐状态只有有限个。充分性来自模型和条件分布，不来自把向量命名为 state。

$$
\bar r(b,a)=\sum_sb(s)r(s,a)
$$

即时奖励对隐藏状态求条件平均。

$$
V^*(b)=\max_a\left[\bar r(b,a)+\gamma\sum_o\Pr(o\mid b,a)V^*(B(b,a,o))\right]
$$

行动既影响外部状态，也影响未来能得到什么信息。这个递推自然包含感知行动的长期收益。

<a id="lesson-information"></a>

## 4 · 信息只有在改变决策时才有控制价值

$$
\operatorname{VOI}=\sum_o\Pr(o\mid b)\max_a\sum_sb(s\mid o)r(s,a)-\max_a\sum_sb(s)r(s,a)-c_{\rm sense}
$$

这是静态隐状态、先感知一次再作终端决策的价值差；不是所有序列任务的一般信息价值公式。

若不计感知成本，观察后仍可采用原动作，因此这里的信息价值非负。加入成本后可能为负。减少 belief 的熵不必改善控制：传感器可能准确识别与奖励无关的细节。控制充分的状态只需保留会影响最优选择的信息，不必重建全部原始观察。

对模型未知的任务，还需区分“当前状态在哪里”与“动力学是什么”两种不确定性。仅对状态做 Bayes filtering，并没有自动学习未知转移。将模型参数也作为隐变量会得到更大的信念空间，计算成本随之增加。

<a id="lesson-algorithm"></a>

## 5 · 精确 filter 与学习的 recurrent state

**算法：外部 reset 改变隐状态分布时，需要相应重置先验；普通训练 batch 边界不自动改变世界。**

1. 初始化已声明的先验 belief。
1. 根据当前 belief 计算或近似价值，选择动作。
1. 取得下一观察与奖励；仅使用已经到达的数据。
1. 用 transition 预测下一状态分布。
1. 乘观测似然，检查证据概率，再归一化。
1. 将后验保存为下一步 belief；重复行动。

$$
z_{t+1}=f_\theta(z_t,A_t,O_{t+1},R_{t+1}),\qquad \hat y_{t+1}=g_\theta(z_{t+1})
$$

RNN 用可训练的有限维摘要替代显式 belief；预测头或控制损失提供训练信号。

RNN 能携带历史，但不因此等于精确后验。预测下一像素、预测多种未来信号和优化控制回报会保留不同信息。TBPTT 截断参数梯度，hidden-state reset 则清除记忆内容；即使二者都发生在同一个代码边界，它们仍是不同操作。

<a id="lesson-example"></a>

## 6 · 两个可手算的例子

先验为 $(0.6,0.4)$，转移矩阵两行为 $(0.9,0.1)$ 与 $(0.2,0.8)$。预测分布是 $(0.62,0.38)$。收到观测的似然为 $(0.8,0.2)$，未归一化后验为 $(0.496,0.076)$，证据为 $0.572$，后验约为 $(0.86713,0.13287)$。直接将似然归一化为 $(0.8,0.2)$ 会丢掉先验和动态预测。

另一个任务只有左右两扇门，隐藏的正确门先验各半，选对得一、选错得负一。立即选择的期望收益为零。准确率 0.8 的传感器让观察后的最佳选择收益为 0.6；感知成本 0.1，净收益为 0.5。若传感器完全无信息，净增益为 −0.1。

<a id="lesson-code"></a>

## 7 · 可运行 filter 与信息价值

矩阵预测、Bayes 归一化与一次感知的期望决策价值；两个例子不依赖采样误差。

```python
def belief_update(prior, transition, likelihood):
    """Transition[s][s_next], likelihood[s_next] for the observation received."""
    probabilities(prior)
    n = len(prior)
    if len(transition) != n or len(likelihood) != n or any(len(row) != n for row in transition):
        raise ValueError('matching state dimensions required')
    for row in transition:
        probabilities(row)
    if any(not 0 <= p <= 1 for p in likelihood):
        raise ValueError('observation likelihood must be in [0,1]')
    predicted = [sum(prior[s]*transition[s][sp] for s in range(n)) for sp in range(n)]
    evidence = dot(predicted, likelihood)
    if evidence <= 0:
        raise ValueError('impossible observation under this model')
    return [p*l/evidence for p, l in zip(predicted, likelihood)], evidence


def information_value(prior, sensor, rewards, sensing_cost=0.):
    """One sensing step, then choose one terminal action.

    sensor[state][observation]; rewards[action][state]; hidden state is static.
    """
    probabilities(prior)
    for row in sensor:
        probabilities(row)
    before = max(dot(prior, r) for r in rewards)
    after = 0.
    for o in range(len(sensor[0])):
        joint = [p*row[o] for p, row in zip(prior, sensor)]
        evidence = sum(joint)
        if evidence:
            posterior = [x/evidence for x in joint]
            after += evidence*max(dot(posterior, r) for r in rewards)
    return before, after-sensing_cost, after-sensing_cost-before
```

执行 python3 extended_foundations_lab.py demo 查看 posterior、evidence 和信息收益；执行 test 检查归一化、零证据与无信息传感器。代码没有训练 recurrent 网络，也没有实现一般 POMDP 规划器。pomdp-solve 是 Cassandra 的经典求解软件；pomdp-py 是后续研究框架，二者的归属与实现范围不同。

<a id="lesson-branches"></a>

## 8 · 假设边界与持续学习接口

- 模型错误：精确计算错误模型下的 posterior，仍可能产生系统性误判。
- 信息泄漏：训练时的 simulator state 可以帮助 critic，但执行策略若依赖它，就不是相同观察条件。
- 序列失配：从 replay 抽单帧无法一般性重建 history-dependent state；burn-in 也需要明确参数版本。
- 变化环境：原观测或转移模型失效时，belief 可能越来越自信地错误，而不是自动适应。

与 CRL 的连接是状态持续构建和状态模型持续校准。可固定任务奖励，单独改变传感器可靠度，比较已知新模型的 Bayes oracle、旧模型 filter 和学习的 recurrent state。这样能区分信息不足、模型滞后与优化失败，不能仅凭回报下降判断遗忘。

<a id="lesson-check"></a>

## 9 · 练习与答案

- 问：观测看起来相同，是否必须采取相同动作？答：不必；不同历史可导致不同 belief。
- 问：belief 熵下降是否意味着决策价值增加？答：不保证。它可能只减少无关信息的不确定性，且感知有成本。
- 问：Bayes 更新的证据为零时可以加一个很小的数继续吗？答：数值平滑可以作为建模改动，但不能隐去模型与观测矛盾；需要报告所用平滑和支持假设。
- 实验：将传感器准确率从 0.8 改为 0.5。在门任务里后验保持先验，信息净收益应为 −0.1；此结论不依赖控制网络。



<a id="chapter-code"></a>

## 下载与运行

标准库解析机制实验；不包含完整神经网络训练或大型 benchmark。

[下载 extended_foundations_lab.py](https://yingwen.io/zh/continual-rl/download/extended_foundations_lab.py)

```sh
python3 extended_foundations_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Kaelbling、Littman、Cassandra · Planning and Acting in Partially Observable Stochastic Domains](https://www.cassandra.org/arc/papers/aij98.pdf)：作者托管原文；belief、决策与 POMDP 求解。

- [Cassandra · pomdp-solve](https://www.pomdp.org/code/index.html)：经典 POMDP 求解软件的作者入口，并非神经 recurrent agent。

- [h2r · pomdp-py](https://github.com/h2r/pomdp-py)：框架作者的模型、belief 与规划接口；是后续工具，不是 1998 原文代码。

- [Kochenderfer、Wheeler、Wray · Algorithms for Decision Making](https://algorithmsbook.com/)：作者书站提供决策、信念状态、模型与规划教材及配套 Julia 代码入口。

<a id="study-connections"></a>

## 与教材主线的衔接

本章是可按问题选择的并列研究分支，不要求先读完其他深度研究分支。

[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)
- [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)
- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)
- [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)

对应原始材料：Kaelbling、Littman、Cassandra：POMDP；Algorithms for Decision Making：state uncertainty。本文为原创讲解，原书、论文与上游代码保留各自许可。
