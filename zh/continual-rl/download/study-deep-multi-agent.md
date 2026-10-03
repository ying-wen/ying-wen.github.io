# 多智能体：博弈、局部信息与集中训练

其他参与者也在学习时，“最优策略”和“环境变化”应怎样定义？

## 本章内容

- 区分共同回报、零和与一般和目标。
- 推导对手策略变化产生的有效转移变化。
- 解释 centralized critic、反事实 baseline 与单调价值分解。

<a id="problem-definition"></a>

## 本章的问题定义

多个决策者同时影响环境。它们可能共享目标，也可能具有相互冲突的收益。

### 给定条件与符号

- 智能体集合 $i=1,\ldots,N$，联合策略 $\pi=(\pi_1,\ldots,\pi_N)$，每个参与者的目标 $J_i(\pi)$。
- 执行时局部历史 $H_t^i$；训练时额外可见量必须明确声明。

### 需要求解的对象

先选择解概念：团队最优、零和 minimax 或一般和均衡，再设计相应学习器。

### 信息与数据权限

集中训练信息不自动在分散执行时可得；其他学习者改变策略会改变个体所见过程。

$$
J_i(\pi_i^*,\pi_{-i}^*)\ge J_i(\pi_i,\pi_{-i}^*)\quad\text{for every }i,\pi_i
$$

这是 Nash 条件：其他策略固定时无人能单边获益。它不同于最大化总收益；团队和零和只是特殊收益结构。

### 成立条件与解的含义

- 信息结构与允许策略类固定；精确最优或均衡未必容易计算。
- 把其他智能体当环境时，必须处理其策略学习造成的非平稳性。

判断准则：按所声明解概念评价：团队收益、可利用性或单边偏离收益，而非只看训练均值。

### 适用边界

- 把一个训练稳定的联合策略自动称为 Nash 均衡。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 推广：放宽条件 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：多个优化者产生多个目标及战略反馈，不能直接使用单智能体最优策略概念。

- 组合不同学习问题 · [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)：局部信息需要记忆；集中 critic 不会替执行策略提供缺失观察。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

其他参与者也在改变行为，局部信用和局部信息又限制各自的学习。

### 本章的核心思路

先固定收益与信息结构，再选择均衡求解、集中 critic 或价值分解等不同机制。

1. [先确定什么叫更好的联合行为](#lesson-games)：零和与一般和采用不同解概念，团队最优也不等于任意均衡。

2. [用集中信息帮助训练信用](#lesson-credit)：反事实基线处理个体动作贡献，但不改变执行时信息权限。

3. [让集中价值支持分散选择](#lesson-factorization)：QMIX 的单调结构使局部 argmax 可组合，同时限制可表达联合价值。

结论与条件：特定博弈或分解结构的结果不等于一般多智能体持续学习收敛。

### 相关方法改变了什么

- 团队优化 / Nash：一个共同目标与多方单边偏离条件是不同问题。

- 集中 critic / 单调价值分解：前者辅助局部策略梯度，后者施加支持分散 argmax 的表示约束。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 条件概率与期望

区分随机变量、一次样本与条件分布；可以使用 Bayes 法则和全期望公式。

### MDP 与价值

理解状态、动作、转移、奖励、策略和 Bellman 递推；学习数据可以来自不同策略。

### 优化与梯度

知道固定目标、参数梯度、约束和期望目标的含义；神经网络只是函数表示的一种选择。

<a id="lesson-setting"></a>

## 1 · 多个决策者不意味着相同目标

Stochastic game 在状态、转移与奖励之外加入多个行动者。共同回报问题追求团队收益；零和问题中一方收益等于另一方损失；一般和问题允许利益部分一致、部分冲突。不能把团队回报最大化、利用某个对手和逼近 Nash 均衡都叫同一种“最优”。

$$
\mathbf A_t=(A_t^1,\ldots,A_t^n),\quad S_{t+1}\sim P(\cdot\mid S_t,\mathbf A_t),\quad R_{t+1}^i=r_i(S_t,\mathbf A_t)
$$

这里为简洁写确定的期望奖励函数；转移和实际奖励也可以随机。联合动作决定世界后续。

设计者规定参与者、奖励、通信与训练信息。运行时每个 actor 能使用的信息可能很少。将多个实体全部交给一个观察完整状态的中央控制器，与多个仅有局部历史的 agent，是不同的信息条件。

<a id="lesson-notation"></a>

## 2 · Dec-POMDP 与执行权限

$$
\begin{gathered}H_t^i=(O_0^i,A_0^i,\ldots,O_t^i)\\A_t^i\sim\pi_i(\cdot\mid H_t^i)\\\pi(\mathbf a\mid\mathbf h)=\prod_i\pi_i(a_i\mid h_i)\end{gathered}
$$

该因子化形式假定没有额外共享随机协调变量；允许通信或共同随机信号时，策略信息结构需要相应扩展。

Dec-POMDP 通常描述共享团队奖励、局部观察与分散行动。每个 agent 的局部 history 不一定足以重建共同 belief，也不知道同伴看到了什么。集中训练允许 critic 使用联合状态或其他 agent 的动作，但执行 actor 仍必须遵守局部信息限制。

这就是 CTDE 的核心：训练信息可以更丰富，部署时动作计算仍可分散。若 actor 的归一化器、输入特征或 recurrent 初始状态暗中依赖全局信息，就会破坏这一约定。

<a id="lesson-derive"></a>

## 3 · 别人学习，会改变我的有效环境

$$
P_t^i(s'\mid s,a_i)=\sum_{\mathbf a_{-i}}P(s'\mid s,a_i,\mathbf a_{-i})\pi_{-i,t}(\mathbf a_{-i}\mid s)
$$

此式为全状态可见、给定其他 agent Markov 策略的简化情形。底层世界转移可以固定，但对单个学习者的边缘转移随别人策略改变。

因此旧 replay 不只可能来自自己的旧策略，也可能来自不同的同伴或对手。集中 critic 把联合动作纳入条件，可缓解一部分归因问题；它不能自动消除对手更新、数据陈旧和未知策略带来的所有非平稳性。

$$
Q_i^\pi(s,\mathbf a)=r_i(s,\mathbf a)+\gamma\mathbb E_{s',\mathbf a'\sim\pi}Q_i^\pi(s',\mathbf a')
$$

此式先取 Markov 联合策略。历史依赖策略还需把相关记忆并入评价条件；固定这些策略后才是固定的价值预测问题。若各策略同时变化，目标也随之变化。

<a id="lesson-games"></a>

## 4 · 零和的 minimax 与一般和均衡

$$
v^*=\max_x\min_y x^\top Ay
$$

在有限双人零和矩阵博弈中，x 与 y 是概率单纯形中的混合策略。确定最佳动作通常无法防止被对手利用。

$$
J_i(\pi_i^*,\pi_{-i}^*)\ge J_i(\pi_i,\pi_{-i}^*)\quad\text{对每个 }i,\pi_i
$$

Nash 条件要求单方偏离不能提高自己的收益，不保证团队最优或社会福利最大。

$$
\operatorname{Gap}(x,y)=\max_{x'}x'^\top Ay-\min_{y'}x^\top Ay'
$$

零和 saddle-point gap 为非负，平衡时为零。只对一个固定弱对手取得高回报，不足以说明 gap 小。

一般和博弈可以有多个均衡，其收益与稳定性不同。自我博弈训练的高胜率可能反映对某组对手的专门适应；应使用独立对手集合、交叉对战和对应 solution concept，而非只报告自身最近策略间的胜负。

<a id="lesson-credit"></a>

## 5 · 集中 critic 与团队信用

对于历史依赖 actor，以下精确形式以 $X=(S,\mathbf H)$ 表示包含状态与所需历史的集中上下文。实际 critic 可以使用更小的输入，但若丢失预测未来所需的记忆，就还存在状态逼近误差。集中信息必须是训练协议允许获得的，而不是部署时凭空新增的观察。

$$
\nabla_{\theta_i}J_i\ \propto\ \mathbb E_\pi[\nabla_{\theta_i}\log\pi_i(A_i\mid H_i)\,Q_i^\pi(X,\mathbf A)]
$$

在相应策略梯度条件与占据加权下，集中 Q 可以为局部 actor 提供训练信号；执行不需要把 Q 的全部信息传给 actor。

$$
A_i(X,\mathbf A)=Q(X,\mathbf A)-\sum_{a_i'}\pi_i(a_i'\mid H_i)Q(X,(a_i',\mathbf A_{-i}))
$$

COMA 型 baseline 固定其他动作，仅对本 agent 的动作求策略平均。这是对联合 Q 的反事实比较，不等同于现实世界的无假设因果识别。

baseline 在给定其他变量后不依赖实际采样的当前动作，故它乘 score 的条件期望为零。这保留梯度方向的期望，同时尝试降低团队共同回报造成的方差。若 critic 错误或遗漏历史，优势仍可能误导行动。

MADDPG 对连续动作使用集中 critic 与局部确定性 actor，让 critic 条件化联合动作；COMA 使用离散随机策略与上述 baseline。两者都使用训练期额外信息，但梯度和适用动作类型并不相同。

<a id="lesson-factorization"></a>

## 6 · QMIX 的可分散 argmax 及表示限制

$$
Q_{\rm tot}=f_s(Q_1(H_1,a_1),\ldots,Q_n(H_n,a_n)),\qquad\frac{\partial f_s}{\partial Q_i}\ge0
$$

mixer 对各局部值单调，训练时还可由全局状态调节。非负权重是实现单调性的一种方式。

$$
(\arg\max_{a_1}Q_1,\ldots,\arg\max_{a_n}Q_n)\in\arg\max_{\mathbf a}Q_{\rm tot}
$$

因为增大任一局部值不会减小联合值，分别选择局部最大值就能达到该表示的联合最大值。平局时可有多个最优动作。

这条性质解决的是该函数类中的 greedy 执行，不保证任意团队价值都能被表示。若 agent 一偏好哪个动作取决于 agent 二的动作，单一局部排序就可能不足。价值分解的可扩展性来自结构约束，同时也承担结构偏差。

<a id="lesson-algorithm"></a>

## 7 · 声明信息边界后组织训练

**算法：不能用共同回报环境中的一次胜利证明对抗博弈中的稳健性。**

1. 定义共同、零和或一般和奖励及目标 solution concept。
1. 逐步收集每个 agent 的局部历史、联合动作与允许的训练状态。
1. 固定目标策略版本，用联合样本更新 centralized critic 或 mixer。
1. 更新局部 actor，或按局部 Q 做分散 greedy 行动。
1. 记录 replay 中的同伴/对手版本和 reset 权限。
1. 执行评估时移除训练专属输入。
1. 按目标报告团队收益、cross-play、best response 或 exploitability。

<a id="lesson-example"></a>

## 8 · 两个矩阵说明不同问题

Matching Pennies 的行玩家收益矩阵为 $\bigl(\begin{smallmatrix}1&-1\\-1&1\end{smallmatrix}\bigr)$。若以概率 $p$ 选第一行，对两列的收益分别为 $2p-1$ 与 $1-2p$。最大化二者的较小值在 $p=0.5$ 达到零。纯策略对纯策略的 saddle gap 可为二，即使某一场获得一。

团队矩阵为 $\bigl(\begin{smallmatrix}3&0\\0&2\end{smallmatrix}\bigr)$。若同伴以概率 0.2 选第一列，自己两动作价值是 $(0.6,1.6)$；若概率改为 0.8，则是 $(2.4,0.4)$。自己没有改变、底层矩阵没有改变，最佳动作却反转。

同一团队矩阵中，实际动作是两人均选零，本 agent 策略均匀。反事实 baseline 为 1.5，优势为 3−1.5=1.5。对自己两种动作按策略求平均，优势均值恰好为零，验证了 baseline 的条件消去性质。

<a id="lesson-code"></a>

## 9 · 博弈解与信用的可运行核

二维 minimax 的精确候选求解、零和 gap、反事实 baseline 与非负线性 mixer 的 greedy 一致性。

```python
def matrix_value(matrix, row_policy, column_policy):
    probabilities(row_policy); probabilities(column_policy)
    return sum(row_policy[i]*column_policy[j]*matrix[i][j]
               for i in range(len(row_policy)) for j in range(len(column_policy)))


def minimax_2x2(matrix):
    """Row maximizes, column minimizes. Optimize the lower envelope of two lines."""
    a, b = matrix[0]
    c, d = matrix[1]
    candidates = [0., 1.]
    denominator = a-c-b+d
    if denominator != 0:
        crossing = (d-c)/denominator
        if 0 <= crossing <= 1:
            candidates.append(crossing)
    lower = lambda p: min(p*a+(1-p)*c, p*b+(1-p)*d)
    p = max(candidates, key=lower)
    return [p, 1-p], lower(p)


def zero_sum_gap(matrix, row_policy, column_policy):
    row_best = max(dot(row, column_policy) for row in matrix)
    column_best = min(sum(row_policy[i]*matrix[i][j] for i in range(len(matrix)))
                      for j in range(len(matrix[0])))
    return row_best-column_best


def counterfactual_advantage(matrix, row_action, column_action, row_policy):
    probabilities(row_policy)
    baseline = sum(row_policy[i]*matrix[i][column_action] for i in range(len(row_policy)))
    return matrix[row_action][column_action]-baseline


def monotone_joint_greedy(local_values, weights):
    if len(local_values) != len(weights) or any(w < 0 for w in weights):
        raise ValueError('nonnegative mixing weights required')
    local_choice = tuple(max(range(len(q)), key=q.__getitem__) for q in local_values)
    joint = list(itertools.product(*(range(len(q)) for q in local_values)))
    value = lambda acts: sum(w*q[a] for w, q, a in zip(weights, local_values, acts))
    return local_choice, value(local_choice), max(map(value, joint))
```

运行 test 检查 matching pennies、被支配动作、优势条件均值和单调 mixer。这里没有训练 MADDPG、COMA 或 QMIX 网络。OpenAI maddpg 是原论文代码入口，PyMARL 包含原团队多种合作算法；OpenSpiel 用于更广泛游戏与 solution concept 实验；MARL 书站另有作者代码、练习与讲义。

<a id="lesson-branches"></a>

## 10 · 多智能体适应与 CRL 的交集

- 共同信息不足：共享奖励不能让局部 agent 自动知道同伴的私有观察。
- 策略分布变化：旧对手、旧搭档和当前参与者可能不一致，评估应指定群体分布。
- 训练状态泄漏：中央 critic 可以用的信息不一定可以进入执行策略。

长期协作和对抗都要求持续识别参与者、保留可迁移技能并适应变化。可以冻结世界物理动力学，只改变同伴策略，研究 agent state 与元适应；再单独改变物理过程。这样的实验能区分社会非平稳与外部世界变化，避免把所有效果笼统归为 CRL。

<a id="lesson-check"></a>

## 11 · 练习与答案

- 问：环境物理转移固定，独立学习者面对的过程就平稳吗？答：不一定，其他策略变化会改变其边缘转移和奖励。
- 问：CTDE 是否允许部署时使用全局状态？答：只有执行协议本来允许时才可以；否则违背信息边界。
- 问：QMIX 能表示任意合作价值矩阵吗？答：不能，单调局部排序限制表示能力。
- 实验：在团队矩阵中改变同伴第一列概率，求两动作等值点。方程 3p=2(1−p)，阈值为 0.4；这不是自身参数更新造成的变化。



<a id="chapter-code"></a>

## 下载与运行

标准库解析机制实验；不包含完整神经网络训练或大型 benchmark。

[下载 extended_foundations_lab.py](https://yingwen.io/zh/continual-rl/download/extended_foundations_lab.py)

```sh
python3 extended_foundations_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Bernstein et al. · The Complexity of Decentralized Control of MDPs](https://pubsonline.informs.org/doi/10.1287/moor.27.4.819.297)：分散控制与局部信息条件的原始理论研究。

- [Lowe et al. · Multi-Agent Actor-Critic](https://arxiv.org/abs/1706.02275)：MADDPG 与混合合作—竞争环境。

- [OpenAI · MADDPG](https://github.com/openai/maddpg)：原论文算法工程，阅读 centralized critic 与局部 actor 输入。

- [Foerster et al. · Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)：COMA baseline 与团队信用分配。

- [Rashid et al. · QMIX](https://proceedings.mlr.press/v80/rashid18a.html)：单调混合与分散 greedy 一致性。

- [Oxford · PyMARL](https://github.com/oxwhirl/pymarl)：原团队合作 MARL 框架，含 QMIX、COMA 等具体实现。

- [DeepMind · OpenSpiel](https://github.com/google-deepmind/open_spiel)：博弈环境与算法平台，适合检查不同 solution concept。

- [Albrecht、Christianos、Schäfer · MARL 教材](https://marl-book.com/)：作者免费教材、视频课程、讲义与练习入口；先读 games 与 solution concepts。

- [MARL 教材作者代码](https://github.com/marl-book/codebase)：教材配套自包含 Python 算法，与原论文工程用途不同。

<a id="study-connections"></a>

## 与教材主线的衔接

本章是可按问题选择的并列研究分支，不要求先读完其他深度研究分支。

[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)
- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)
- [元学习与学习规则的适应](https://yingwen.io/zh/continual-rl/algorithms/meta/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)
- [持续学习的智能体架构](https://yingwen.io/zh/continual-rl/construction/architectures/)

对应原始材料：Multi-Agent Reinforcement Learning：Games、Solution Concepts、Deep MARL；MADDPG；COMA；QMIX。本文为原创讲解，原书、论文与上游代码保留各自许可。
