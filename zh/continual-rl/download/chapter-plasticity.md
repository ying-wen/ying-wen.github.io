# 可塑性：梯度通路、有效学习率与预测干扰

网络不是不会做旧事，而是越来越难学会新事；怎样诊断、定位并恢复这种能力？

## 本章内容

- 用 matched aged/fresh probe 区分能力丧失、探索失败和普通遗忘。
- 从梯度通路推导 dormant-unit 机制，具体实现 ReDo 与 CBP 的选择、替换、成熟期和优化器状态处理。
- 理解 CReLU、正则化、网络重置、plasticity injection 等分支的不同作用及保留代价。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 梯度链式法则

网络输出对某个权重的梯度由下游权重、激活导数与输入相乘；任一环节接近零都可能使参数难以变化。

### ReLU

$\operatorname{ReLU}(z)=\max(0,z)$；$z<0$ 时导数为零。一个单元在当前数据上不激活，并不意味着它在所有未来状态上都无用。

### 优化器状态

Adam 保存一阶矩、二阶矩及时间计数；只重置参数而不处理这些状态，可能让新参数继承旧方向和尺度。

<a id="lesson-setting"></a>

## 1 · 可塑性的操作性定义

设两个同样结构的网络接收到完全相同的新目标和训练数据：一个已经训练很久，另一个刚初始化。若前者在相同学习预算下改进更慢，便有可塑性下降的证据。反之，仅看到在线 reward 下降，还不能排除探索不足、环境更难、价值估计错误或任务本身无解。

$$
\mathcal P_K(\theta;\mathcal D)=L_{\mathcal D}(\theta)-L_{\mathcal D}(U^K(\theta;\mathcal D))
$$

这是一种可操作的 $K$ 步改进量：$U$ 指定优化器、步长、训练序列和统计更新。评价采用独立但匹配的 probe 数据，并控制初始损失；仅比较不同起点的绝对改进量会混入任务难度差异。

完整 aged/fresh 对照至少固定网络容量、目标难度、数据顺序、更新数、优化器和归一化。为了拆解原因，还应增加“只重置优化器”“只重置部分权重”“保留 replay 但重置网络”等条件。若 fresh 用额外探索或外部 task reset，就不再是同一学习能力测试。

| 诊断量 | 能提示什么 | 不能单独证明什么 |
| --- | --- | --- |
| 新目标上的固定预算 loss reduction | 局部可学习性 | 在线探索是否成功 |
| Dormant 比例与梯度范数 | 激活或梯度通路变化 | 全部 plasticity loss 都由死 ReLU 导致 |
| 特征矩阵的有效秩 | 表征多样性退化 | 秩越大一定回报越高 |
| Hessian/曲率代理、权重范数 | 优化几何与尺度变差 | 一个相关曲线就是因果机制 |

CRL 中即使环境外部不变，策略、bootstrap target 与所访问状态也持续变化。因此可塑性问题不要求人为切任务；但人为目标切换能帮助在小实验里隔离机制。

<a id="lesson-derive"></a>

## 2 · 隐藏单元的梯度通路

$$
\begin{aligned}f(x)&=\sum_i v_i h_i(x),\qquad h_i(x)=\max(0,u_i^\top x),\\L&=\tfrac12(f(x)-y)^2,\\\nabla_{u_i}L&=(f-y)v_i\mathbf1[u_i^\top x>0]x,\\\partial_{v_i}L&=(f-y)h_i(x).\end{aligned}
$$

若当前数据上 $u_i^\top x<0$，这个单元的输入权重与输出权重梯度均为零。参数容量与能够通过梯度使用的容量因而不同。

这个例子刻意简单：真实可塑性下降也可能出现于没有饱和单元的网络，与目标尺度、曲率、优化器状态和特征相关性有关。因而 ReDo/CBP 是针对某类故障的机制，不是所有网络老化现象的定义。

$$
s_i=\frac{\mathbb E_{x\sim D}[|h_i(x)|]}{\frac1H\sum_{j=1}^H\mathbb E_{x\sim D}[|h_j(x)|]}
$$

ReDo 用层内相对活动度辨认 dormant 单元，H 为该层单元数，D 是检测用数据分布。若整层全零，分母为零，实现必须明确处理，而不是让 NaN 静默决定选择。

一个活动度低的单元可能在稀有但重要的状态上有用。D 的覆盖、检测频率和阈值决定了“无用”的含义。把检测 batch 换成当前一个观测，会极大改变噪声和选择；这不是不需要评估的实现小节。

<a id="lesson-redo"></a>

## 3 · ReDo：检测休眠，再恢复可用的梯度通路

**算法：算法伪代码**

1. 按既定周期收集检测 batch；前向记录各层隐藏激活
1. 对每层计算 mean(abs(h_i)) 和相对 score s_i
1. 选择 s_i≤阈值 的单元；阈值与检测频率是超参数
1. 对选中单元：
  1. 输入权重按原初始化分布重新采样，输入 bias 重新初始化
  1. 对应输出权重置零，避免新随机特征立即注入任意输出
  1. 清理相关优化器状态；核对 target network 与参数同步规则
1. 恢复正常 RL 更新；记录替换数量与替换前后的预测变化

“输出权重置零”只保证新随机特征在替换瞬间不产生新贡献，不保证旧输出完全保留。因为旧单元的贡献也被删掉了；只有旧贡献恰好为零时，替换瞬间才严格不变。低活动度通常减小扰动，但不是函数等价变换。

$$
f_{\rm after}(x)-f_{\rm before}(x)=-\sum_{i\in\mathcal R}v_i h_i(x)
$$

假定其余参数和 bias 不变且新输出权重设为零，R 是替换集合。这是精确的单隐藏层输出差，可用于检查重置实现有没有删除未选中的单元。

下一个样本上，若新激活 $h_{\rm new}\ne0$ 且误差非零，即使输出权重为零，输出权重仍有梯度。输出连接建立后，输入权重才重新获得下游梯度。新生单元需要学习时间，频繁再次替换会打断这一过程。

<a id="lesson-cbp"></a>

## 4 · CBP：特征的持续生成与检验

Continual Backpropagation 在常规反向传播之外，持续维护候选特征的效用与年龄，优先替换已经成熟却低效用的单元。效用不必等同于活动度：高活动但没有下游用途的特征，也可能值得替换。作者实现提供 contribution、zero-contribution、adaptable-contribution 等选项；必须写清实际使用哪一种。

$$
\begin{aligned}c_{i,t}&=\mathbb E_B[|h_i|]\,\operatorname{mean}_j|W_{{\rm out},ji}|,\\u_{i,t}&=\beta u_{i,t-1}+(1-\beta)c_{i,t},\\\widehat u_{i,t}&=u_{i,t}/(1-\beta^{a_i}).\end{aligned}
$$

这是 contribution 效用的一个明确版本；ai 是自出生以来的更新次数。它同时考虑激活和下游权重，并用年龄进行 EMA 偏差修正。不要把此式泛称为所有 CBP 实验的唯一评分。

$$
c_{i,t}^{\rm adapt}=\frac{\mathbb E_B[|h_i-\widehat\mu_i|]\,\operatorname{mean}_j|W_{{\rm out},ji}|}{\operatorname{mean}_k|W_{{\rm in},ik}|+\varepsilon}
$$

adaptable-contribution 风格还考虑非恒定贡献及输入权重尺度。本式显式加入 ε 以定义零分母；对应原代码/实验的均值、范数及 ε 选择应按实际配置复核。

**算法：算法伪代码**

1. 1. 用当前样本完成普通梯度更新；保留检测所需激活
1. 2. 单元年龄 ai += 1；更新激活均值与效用 EMA
1. 3. 只让 ai>成熟阈值 的单元进入候选集合 E
1. 4. replacement_credit += replacement_rate × |E|
1. 5. 本步替换 floor(replacement_credit) 个候选，扣掉整数部分
1. 6. 在候选内按 bias-corrected utility 从小到大选择
1. 7. 重采输入参数，输出连接归零；必要的 bias 补偿按指定变体处理
1. 8. 清零新生单元的 age、utility、activation statistics
1. 9. 清理相关 optimizer moments / coordinate step；继续反向传播

步骤 4–5 是累计预算变体，也可用概率方式实现小于一个单元的期望替换率。成熟期保护新单元的学习机会。替换率为零时，算法退化成不替换的原学习器。

Adam 的一阶矩、二阶矩以及时间计数都与参数年龄有关。某些作者实现使用可逐坐标清零的 AdamGnT；普通框架只给整个张量一个 step 时，局部重置无法等同于所有坐标的全新 Adam。复现时应保留原优化器语义，或把替代策略作为一个新的实验条件。

<a id="lesson-normalization"></a>

## 5 · NaP：归一化后的有效学习率

单元保持活动时，网络仍可能逐渐学得更慢。一个原因是归一化改变了参数尺度与函数变化的关系。考虑无偏置、忽略数值稳定项的理想尺度不变层：把进入归一化的权重乘以正数，输出保持不变。设损失也具有这种不变性，用链式法则可得：

$$
L(cw)=L(w),\quad \nabla L(cw)=c^{-1}\nabla L(w),\quad c>0.\qquad \eta_{\rm eff}^{\rm SGD}=\frac{\eta}{\|w\|^2}.
$$

令单位方向为 $w/\|w\|$，一次普通 SGD 的方向变化与权重范数的平方成反比。这里的有效学习率公式针对普通梯度步；归一化梯度的对应尺度为 $\eta/\|w\|$，不能直接把两者等同于所有 Adam 更新。

因此，同样的标称步长下，权重范数从 2 增长到 4，理想 SGD 的有效步长变为原来的四分之一。这个效应不会被“休眠单元比例仍然很低”排除。Lyle 等在 NeurIPS 2024 的 Normalize-and-Project（NaP）将非线性之前的归一化，与定期恢复每层初始权重范数结合，使学习率调度显式化。

$$
\widetilde W_\ell\leftarrow\operatorname{OptimizerStep}(W_\ell),\qquad W_\ell^+\leftarrow\frac{\|W_{\ell,0}\|_F}{\|\widetilde W_\ell\|_F}\widetilde W_\ell.
$$

式中假定投影前范数非零。投影保持权重方向，仅恢复范数。归一化的可学习缩放与偏置是另一组参数，需要按具体变体单独约束；它们不自动满足这里的尺度不变条件。

归一化还有第二个作用：均值和方差使单元之间的梯度耦合，位于 ReLU 前的归一化可以给部分不激活的预激活量传递其他单元的梯度。这与直接替换单元是不同机制。有效学习率保持恒定也并非总是最优：价值估计需要一定程度的收敛，论文中的部分 Rainbow 实验仍需要显式衰减。研究问题因而是适应速度与估计噪声的调度，而非无条件维持最大更新幅度。

<a id="lesson-churn"></a>

## 6 · C-CHAIN：控制更新在参考状态上的影响

共享参数让一次局部学习同时改变其他输入的预测。称这种改变为 prediction churn。设当前训练输入为 $x$，参考输入为 $\bar x$，标量网络为 $f_\theta$，一次梯度步为 $\Delta\theta=-\eta\nabla_\theta L_x$。对参考预测做一阶展开：

$$
\begin{aligned}\Delta f(\bar x)&\approx\nabla_\theta f_\theta(\bar x)^\top\Delta\theta\\&=-\eta\underbrace{\nabla_\theta f_\theta(\bar x)^\top\nabla_\theta f_\theta(x)}_{K_\theta(\bar x,x)}\frac{\partial L_x}{\partial f_\theta(x)}.\end{aligned}
$$

$K_\theta$ 是两个输入的梯度内积，即经验神经切线核的一个元素。它可以为正或负；在 $x$ 上降低损失并不意味着在 $\bar x$ 上也降低损失。该一阶式在更新较小时解释局部干扰，并非有限大步长下的精确等式。

ICML 2025 的 C-CHAIN 在参考状态上约束这种变化。其值函数形式可写成下面的函数空间正则：当前网络在参考状态的全部动作价值，靠近近期冻结网络的输出。它与 EWC 的参数距离不同，也与长期保存旧任务教师不同，主要控制训练过程中的预测变化。

$$
L(\theta)=L_{\rm TD}(\theta)+\lambda\,\mathbb E_{s\sim D_{\rm ref}}\|Q_\theta(s,\cdot)-\operatorname{sg}[Q_{\theta^-}(s,\cdot)]\|_2^2.
$$

$\theta^-$ 是近期参数快照，$\operatorname{sg}$ 表示停止梯度。作者 MinAtar 实现独立采样 TD batch 与参考 batch，因此样本可能重叠；它还维护有界的历史网络，并根据损失尺度调节正则系数。

**算法：算法伪代码**

1. 从经验库采样 TD batch，构造停止梯度的 bootstrap target
1. 独立采样参考状态，选择一个近期冻结网络 $Q_{\theta^-}$
1. 计算 TD 损失与参考状态上的全动作价值差
1. 对两项加权和反向传播并更新当前网络
1. 按配置保存近期网络快照、淘汰超出窗口的快照
1. 用运行中的损失统计调整正则强度，记录预测改变与新目标学习速度

实现不需要显式构造完整神经切线核；核只用于解释干扰机制。过强的约束会妨碍必要的价值修正，参考分布遗漏的状态也不受保护。参考 batch、历史网络和额外前向传播均有成本，所以该方法的经验重放条件与严格流式学习不同。一个直接实验是匹配更新数与内存预算，在固定 probe 上比较替换、范数投影和短期函数约束分别改善哪类故障。

<a id="lesson-example"></a>

## 7 · 手算：替换扰动与学习恢复

网络只有两个单元，x=1，输入权重 u=[−1,1]、输出权重 v=[0.5,2]。激活 h=[0,1]，输出 f=2；活动度的层均值为 0.5，scores=[0,2]，阈值 0.1 只选第一个。把它的输入改成 0.5、输出改成 0，输出仍为 2，因为旧贡献为零。

若误选第二个单元，它的旧贡献为 2，输出会从 2 降为 0。这个变化不是数值误差，而是更新定义直接造成的。对神经网络实验应记录预测扰动、策略 KL 和短期回报下降，不能只记录 dormant 数量减少。

再用单单元学习 y=1。aged 网络 u=−1、v=1，在 x=1 上所有相关梯度为零，平方损失一直为 0.5。fresh 网络 u=0.5、v=0，第一步 α=0.1 更新输出权重 v→0.05，输入 u 暂时不变；此后输出通路非零，输入也能学习。两者用相同 20 次样本训练，本页程序验证后者损失下降。这个人为构造只证明死 ReLU 机制，不代表所有 aged 网络都会失效。

成熟期例子：三个单元效用 [0,0.1,1]、年龄 [0,21,21]，阈值 20，rate=0.25。最年轻的单元虽效用最低也不进入候选；两个候选每步积累 0.5 替换名额，第二步才替换效用 0.1 的成熟单元。

<a id="lesson-code"></a>

## 8 · 替换器实现与诊断实验

单输入/单输出隐藏层的机制实现；moment 数组分别表示被替换输入/输出权重的状态。完整网络还需处理 bias、卷积轴和 target 参数。

```python
def redo_indices(mean_abs_activations, threshold=0.1):
    """Relative activity score; an all-zero layer is entirely dormant."""
    if not mean_abs_activations:
        return []
    denominator = sum(mean_abs_activations) / len(mean_abs_activations)
    if denominator == 0:
        return list(range(len(mean_abs_activations)))
    return [i for i, a in enumerate(mean_abs_activations)
            if a / denominator <= threshold]


def cbp_candidates(utilities, ages, maturity, rate, credit=0.0):
    """Accumulated-budget variant: protect young units, rank eligible utility."""
    if not 0 <= rate <= 1:
        raise ValueError("replacement rate outside [0,1]")
    eligible = [i for i, age in enumerate(ages) if age > maturity]
    credit += rate * len(eligible)
    count = min(len(eligible), int(credit))
    selected = sorted(eligible, key=lambda i: (utilities[i], i))[:count]
    return selected, credit - count


def cbp_contribution_utility(old_utility, mean_absolute_activation,
                             mean_absolute_outgoing, age, decay=0.99):
    """EMA plus age correction for the explicitly chosen contribution variant."""
    if age < 1 or not 0 <= decay < 1:
        raise ValueError("increment age before updating utility")
    instantaneous = mean_absolute_activation * mean_absolute_outgoing
    updated = decay * old_utility + (1. - decay) * instantaneous
    return updated, updated / (1. - decay**age)


def recycle_one_unit(incoming, outgoing, first_moment, second_moment,
                     index, new_incoming):
    """Scalar-input, scalar-output hidden unit; reset its incoming/outgoing state.
    first/second_moment[i] each contain [incoming_parameter, outgoing_parameter].
    """
    incoming[index] = new_incoming
    outgoing[index] = 0.0
    first_moment[index] = [0.0, 0.0]
    second_moment[index] = [0.0, 0.0]


def relu_prediction(incoming, outgoing, x=1.0):
    return sum(v * max(0., u*x) for u, v in zip(incoming, outgoing))


def train_relu_toy(incoming, outgoing, target=1., steps=20, alpha=0.1):
    incoming, outgoing = incoming[:], outgoing[:]
    for _ in range(steps):
        error = target - relu_prediction(incoming, outgoing)
        old_u, old_v = incoming[:], outgoing[:]
        for i in range(len(incoming)):
            incoming[i] += alpha * error * old_v[i] * (old_u[i] > 0)
            outgoing[i] += alpha * error * max(0., old_u[i])
    return 0.5 * (target - relu_prediction(incoming, outgoing)) ** 2


def plasticity_demo():
    u, v = [-1., 1.], [0.5, 2.]
    before = relu_prediction(u, v)
    moments1, moments2 = [[3., 4.], [3., 4.]], [[5., 6.], [5., 6.]]
    recycle_one_unit(u, v, moments1, moments2, 0, 0.5)
    print("plasticity", {"redo_selected": redo_indices([0., 2.]),
          "prediction_before_after": (before, relu_prediction(u, v)),
          "cleared_optimizer_state": moments1[0] + moments2[0],
          "matched_probe_loss_aged": train_relu_toy([-1.], [1.]),
          "matched_probe_loss_fresh": train_relu_toy([0.5], [0.]),
          "scope": "synthetic dead-ReLU mechanism, not an empirical LoP diagnosis"})
```

输出应包括替换前后 (2,2)、清零的 optimizer state，以及 aged/fresh probe loss

```sh
python lifelong_algorithms_lab.py plasticity
python lifelong_algorithms_lab.py test
```

示例覆盖全零层、相对活动度、成熟期、分数替换预算与替换扰动。fresh 网络是固定数据诊断的对照条件。作者 ReDo 代码中的 weight_recyclers.py、CBP 的 lop/algos/gnt.py 分别给出完整框架中的选择与生成流程；C-CHAIN 的 MinAtar agent 展示参考采样和快照队列。本页小实验实现替换机制，不包含这些论文的完整深度 RL 训练。

- 先做固定数据 probe，分离探索与学习能力；再返回在线 RL 看是否仍有实际收益。
- 匹配参数量与 FLOPs，特别是 CReLU 改变特征维度、模块扩容和 injection 增加容量的情况。
- 与随机等量替换、只清 optimizer state、全网络重置和只改激活函数对照，辨认选择规则是否真的有用。
- 同时报告新任务适应、旧技能保留、替换后短期损失和每步额外成本，避免只选择一种有利指标。

<a id="lesson-branches"></a>

## 9 · 干预位置与选择依据

$$
\operatorname{CReLU}(z)=[\max(0,z),\max(0,-z)]
$$

对 z≠0 至少一支对 z 有非零导数，但最终输出梯度仍取决于下游权重与损失。它不是整个网络梯度永远不为零的保证，并且会改变宽度/参数预算。

| 方法线 | 干预对象 | 应排除的替代解释 |
| --- | --- | --- |
| ReDo | 当前分布下低活动单元周期性回收 | 收益是否仅来自额外随机扰动？ |
| CBP / generate-and-test | 低效用成熟特征的持续替换 | 是否只需随机替换？成熟期与效用是否都重要？ |
| CReLU / 激活改造 | 激活及梯度通路 | 增加的特征维度是否解释提升？ |
| LayerNorm / 参数正则 | 尺度、曲率与优化几何 | 是否只是更好的超参数而非长期机制？ |
| Primacy-bias resets | 周期重置一部分网络，保留经验 | 收益可能依赖 replay 重建能力，不自动适于无 replay 流式学习 |
| Plasticity injection | 接入可训练新分支并控制初始输出 | 额外参数和 optimizer state 如何计入固定预算？ |

最重要的研究逻辑是先诊断，再选干预。若 aged 与 fresh 在同一数据上学得同样快，但在线 aged 不再到达新状态，应转向探索或 agent state；若输出尺度越来越大而 dormant 比例不变，优先检查优化几何；若旧技能特别珍贵，则回收规则要加入保留约束。

<a id="lesson-check"></a>

## 10 · 自测与研究练习

- 为什么 low-utility 不是“永远无用”？效用根据某个近期分布估计；稀有历史状态和未来任务可能改变结论。
- 为什么新输出权重置零后仍能成长？第一步输出权重有非零激活梯度，后续输入权重才得到非零下游信号。
- 清零全张量 Adam step 可以当作局部重置吗？不可以，它也改变未替换坐标的偏差修正；必须明确这项额外干预。
- ReDo 能否证明可塑性下降就是 dead neurons？不能。存在其他机制，也存在休眠是结果而非根因的可能。

动手题：在 toy 网络中让旧单元的小贡献从 0 缓慢变到 0.1，逐次核对替换前后输出差正好等于删掉的贡献。再把随机 reset 与 score-based reset 设置为相同替换次数，用 matched probe 比较改善，避免将替换率不同误判为选择规则更好。

## 本章的实验设计

使用新任务学习速度检验可塑性。保留原参数与重新初始化的对照必须具有相同的新增数据预算。

设定：让 learner 具有不同训练年龄，再在相同新增数据下学习新信号。用 fresh 模型、优化器重置与不同替换机制拆解原因。

- 新信号对普通强基线确实可学。
- 替换包含按算法约定处理入边、出边和 optimizer 状态。
- 诊断探针引入的任务变化与原主轨迹分开。

对照：完全 fresh、仅重置优化器；相同替换数量的随机替换；相同新增数据、容量与总计算

记录：新任务达到标准的样本量；aged/fresh 学习曲线与失败；激活、梯度、秩及替换成本

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-continual)

## 学习与研究衔接

不遗忘不意味着还能学习。应在相同新数据预算下测试老网络、新网络与局部重置。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-plasticity) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=plasticity) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=plasticity)

<a id="chapter-code"></a>

## 下载与运行

替换、optimizer 状态与合成 aged/fresh 机制诊断；不是 ReDo/CBP 全论文复现。

[下载 lifelong_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)

```sh
python lifelong_algorithms_lab.py plasticity
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sokar et al. · The Dormant Neuron Phenomenon in Deep RL](https://proceedings.mlr.press/v202/sokar23a.html)：ReDo 活动度定义、重置流程与实验。

- [ReDo 作者实现 · Google Dopamine](https://github.com/google/dopamine/tree/master/dopamine/labs/redo)：作者发布的 Dopamine 版本；核心文件为 weight_recyclers.py 及各 recycled agent。

- [Dohare et al. · Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7)：长期可塑性与 continual backpropagation 的研究。

- [CBP 作者代码 · loss-of-plasticity](https://github.com/shibhansh/loss-of-plasticity)：lop/algos/gnt.py 与 lop/utils/AdamGnT.py；效用变体、成熟期、预算和 optimizer 状态均须按配置核对。

- [Lyle et al. · Understanding Plasticity in Neural Networks](https://proceedings.mlr.press/v202/lyle23b.html)：强调曲率等机制，避免将全部 plasticity loss 简化为 dead ReLU。

- [Abbas et al. · Loss of Plasticity in Continual Deep RL](https://proceedings.mlr.press/v232/abbas23a.html)：循环 Atari 与 CReLU 的原始研究，配合预算匹配理解激活改造。

- [Nikishin et al. · The Primacy Bias in Deep RL](https://proceedings.mlr.press/v162/nikishin22a.html)：保留经验但周期重置部分网络的机制；与严格 streaming 条件不同。

- [Lyle et al. · Normalization and effective learning rates in reinforcement learning](https://papers.nips.cc/paper_files/paper/2024/hash/c04d37be05ba74419d2d5705972a9d64-Abstract-Conference.html)：NeurIPS 2024：NaP、归一化的梯度耦合，以及显式与隐式学习率调度。

- [Tang et al. · Mitigating Plasticity Loss in Continual RL by Reducing Churn](https://arxiv.org/abs/2506.00592)：ICML 2025：用参考状态的函数空间约束控制预测改变。

- [C-CHAIN 作者实现 · MinAtar Double DQN](https://github.com/bluecontra/C-CHAIN/blob/main/crl_minatar/agents/double_dqn_c_chain.py)：参考 batch、全动作 Q 正则、近期网络队列和损失尺度自适应；完整运行还需对应环境及训练配置。
