# 知识保留：经验重放、参数约束与模型记忆

新经验要求适应，旧技能又可能重新有用；如何在固定预算内管理它们的冲突？

## 本章内容

- 区分任务再次出现时的遗忘、分布改变后的必要适应，以及网络失去学习能力。
- 从目标函数推导 replay、EWC、策略蒸馏与 CLEAR，并把算法落实为完整存储/采样/更新流程。
- 用固定内存、相同数据与计算预算设计保留—适应对照，而不让“存得更多”掩盖算法贡献。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 训练分布与评价分布

当前数据分布由环境和行为策略共同产生；评价可以关注当前环境、历史环境混合或整个生命期收益，需要事先指定。

### KL 与策略蒸馏

$D_{\rm KL}(p\Vert q)=\sum_a p(a)\log[p(a)/q(a)]$。蒸馏在指定状态上保持教师分布 $p$，学生参数仍可改变。

### 二阶近似

在旧解附近，用曲率加权的二次函数近似旧损失；该近似描述局部敏感性。

<a id="lesson-setting"></a>

## 1 · 保留的对象与评价目标

一个导航智能体曾学会走左门。门的开关暂时改变，之后可能恢复原状。若新训练破坏了左门技能，回来时重新学习很慢，这是知识保留问题；若门永久封死，强迫新策略继续走左门反而有害。持续学习不是无条件保留一切，而是明确未来会如何使用历史知识。

用 A→B→A 的重复环境可以观察保留，但它只是一个研究协议：是否给任务 ID、是否外部 reset、每段多长、是否重新初始化优化器，都影响解释。真实 single-life 不能随时回 A 测试；可以用独立评测副本诊断，但副本不能向在线 agent 提供额外训练数据或重置其主状态。

| 现象 | 可检验问题 | 不能混称为 |
| --- | --- | --- |
| 遗忘 | A 的表现训练 B 后是否下降？ | 不一定是 B 学不会 |
| 负迁移 | 有历史的 agent 学 B 是否比 fresh 慢？ | 不一定是旧 A 被遗忘 |
| 可塑性丧失 | 相同新数据与优化预算下 aged 学习能力是否下降？ | 仅凭低回报不能确诊 |
| 必要适应 | 旧行为已失效时能否及时改变？ | 不是所有旧性能下降都应阻止 |

$$
L_t(\theta)=(1-\omega)L_{\rm now}(\theta)+\omega L_{\rm retained}(\theta),\qquad 0\leq\omega\leq1
$$

这是说明稳定—适应取舍的目标骨架。历史损失可以由真实 replay、参数近似或旧模型输出构造；不同构造保留的是不同信息。

先固定预算：参数、目标网络、教师网络、buffer 中的观测/动作/概率/价值、统计量都按字节计算；还要固定每个环境步允许的梯度更新数。一个方法多训练十次或保存整个历史，不能只与只用一条新样本的方法比较“算法更好”。

<a id="lesson-derive"></a>

## 2 · 经验重放与历史采样分布

Replay 使已经不再出现的状态继续影响梯度。FIFO buffer 强调近期数据；reservoir sampling 则在固定容量 $M$ 下，让前 $t$ 条样本具有相同的保留概率。两者代表的训练分布不同，应由未来评价需求决定。

**算法：Reservoir sampling；保存分布与训练时的采样分布分别定义**

1. 初始化样本计数 $t=0$，buffer 容量为 $M$
1. 每来一条 transition：
  1. 令 $t\leftarrow t+1$
  1. 若 buffer 未满，追加样本
  1. 否则均匀采样 $j\in\{0,\ldots,t-1\}$
    1. 若 $j<M$，以新样本替换 buffer 的第 $j$ 项
1. 训练时按指定的新旧样本比例采样并更新

当 $t>M$ 时，新样本被接收的概率为 $M/t$。任一旧样本此前被保留的概率为 $M/(t-1)$，本步被替换的条件概率为 $1/t$；于是存活概率为 $[M/(t-1)](1-1/t)=M/t$。均匀的是时间索引，并不意味着每个任务、状态或稀有事件等权。

$$
\mathbb E[\widehat g]= (1-\omega)\mathbb E_{d_t}[\nabla\ell]+\omega\mathbb E_{d_B}[\nabla\ell]
$$

dB 由保存与采样共同决定。损失权重、batch 中新旧比例和重用次数都能改变实际有效权重，不能只报告 buffer 容量。

强化学习还有一个监督学习没有的困难：旧 transition 由旧策略 μ 产生，当前要改进的是 π。旧奖励和转移样本在平稳环境中仍可有效，但策略梯度与多步 bootstrap 需要处理行为差异；环境动力学若也变了，动作重要性比并不能把旧世界样本变成新世界样本。

<a id="lesson-clear"></a>

## 3 · CLEAR：新经验学习、旧经验校正、旧输出克隆

CLEAR 结合三类更新：当前经验支持适应；历史经验通过 off-policy actor–critic 继续改善价值与行为；旧状态上的策略和价值克隆约束输出漂移。为此，存储项除转移外还包含旧策略分布与旧价值输出。该方法不要求任务边界，但保留效果取决于 replay 覆盖。

$$
\begin{aligned}\bar\rho_t&=\min(\rho_{\max},\pi(A_t\mid S_t)/\mu(A_t\mid S_t)),\\c_t&=\min(c_{\max},\pi(A_t\mid S_t)/\mu(A_t\mid S_t)),\\d_t&=\bar\rho_t[R_{t+1}+\gamma V(S_{t+1})-V(S_t)],\\v_s^{\rm trace}&=V(S_s)+\sum_{t=s}^{s+n-1}\gamma^{t-s}\!\left(\prod_{i=s}^{t-1}c_i\right)d_t.\end{aligned}
$$

这是常数折扣下长度为 $n$ 的 V-trace 目标，空乘积为 1。$\bar\rho_t$ 调整本步误差，其截断影响固定点对应的有效策略；$c_t$ 控制跨步传播与方差，在原理论条件下不改变该固定点。真实终止位置用零折扣截断，unroll 的非终止末端保留价值 bootstrap。

$$
\begin{aligned}L_V&=\tfrac12(V_\theta(s)-\operatorname{stopgrad}(v_s^{\rm trace}))^2,\\L_\pi&=-\log\pi_\theta(a_s\mid s)\operatorname{stopgrad}\!\left[\bar\rho_s(r+\gamma v_{s+1}^{\rm trace}-V_\theta(s))\right],\\L_{\rm clone}&=\beta_\pi D_{\rm KL}(\mu_{\rm stored}\Vert\pi_\theta)+\beta_V(V_\theta-V_{\rm stored})^2.\end{aligned}
$$

RL 项应用于新旧数据，cloning 项用于 replay；再按原配置加入熵项和各损失权重。旧 logits、旧值与 trace target 不通过梯度回流到历史模型。actor 的整个重要性加权 advantage 都要 stop-gradient；若把 ρ̄ 留在外面，会额外对概率比求导，改变政策梯度估计器。

**算法：算法伪代码**

1. 动作采样前保存 μ(a|s)、完整策略输出和 Vstored(s)
1. 环境返回 transition；将新 unroll 与历史 unroll 按规定比例组成训练数据
1. 用当前网络计算 π、V；用存储 μ 计算裁剪的重要性比
1. 自后向前计算 V-trace targets；将 targets stop-gradient
1. actor 使用整体 detach 的 weighted advantage（包括重要性比）；计算 critic、entropy；只对 replay 加策略/价值 cloning
1. 合并损失做一次优化；按容量策略存入新经验
1. 记账 buffer 字节数、unroll 长度、每步更新数与 replay 年龄

旧策略未必最优，因此克隆与 RL 更新有不同作用：前者限制偏离，后者利用奖励继续改进。强克隆可能保存旧错误，弱克隆则可能无法保留旧能力。比较方法时需要同时观察当前适应与历史回访表现。

<a id="lesson-ewc"></a>

## 4 · EWC：用参数曲率近似旧知识

无法保存旧数据时，可以保留旧解 $\theta_A$ 与每个参数的重要性。若旧损失在 $\theta_A$ 的梯度近似为零，二阶展开中的 Hessian 描述沿哪些方向移动代价大。EWC 用对角 Fisher 近似这种局部几何，从而让重要参数较难改变。

$$
\begin{aligned}L_A(\theta)&\approx L_A(\theta_A)+\tfrac12(\theta-\theta_A)^\top H_A(\theta-\theta_A),\\L_{\rm EWC}(\theta)&=L_B(\theta)+\frac{\kappa}{2}\sum_iF_i(\theta_i-\theta_{A,i})^2,\\\nabla_iL_{\rm EWC}&=\nabla_iL_B+\kappa F_i(\theta_i-\theta_{A,i}).\end{aligned}
$$

Fisher 来自指定概率模型的 score 外积期望：$F_i=\mathbb E[(\partial_{\theta_i}\log p_\theta(y\mid x))^2]$。真实 Fisher 在模型分布上取期望；经验 Fisher 改用观测标签。两者与实际 RL 损失的 Hessian 不普遍相等，因此样本、动作及分布都是估计器定义的一部分。

完整流程是：在约定的 consolidation 时刻保存 θA；在相应数据分布上估计并冻结 F；后续每步把二次惩罚加入当前损失。若没有任务边界，要定义 consolidation 的触发或连续累计规则。每个任务另存一个 F 与参数快照会随任务数增长；online EWC 将历史重要性折叠成固定规模统计，但其遗忘系数与近似误差需要评估。

EWC 保留的是参数附近的旧损失近似，不是完整旧数据，也不保证旧动作概率不变。网络可用不同参数实现同一功能；对角近似忽略参数之间的相关方向。因此功能空间的蒸馏与参数空间的正则可以产生明显不同的行为。

<a id="lesson-model-memory"></a>

## 5 · DRAGO：保留世界模型，而非只保留旧策略

如果任务改变的只是奖励与起点，不同任务会让智能体看到同一世界的不同区域。此时值得保留的是动力学知识：旧任务的最优策略未必适用于新任务，但旧房间的布局仍然有效。DRAGO（Fu 等，ICML 2025）研究这一设定，允许已知任务切换、当前任务 buffer，以及一个旧模型快照；不保存往期任务的原始数据。

先用生成器 $G$ 表示过去访问过的状态—动作联合分布。从中采样 $(\hat s,\hat a)$，由冻结的旧动力学模型 $T_{\rm old}$ 产生后继标签。当前模型同时拟合真实新经验与这些合成旧经验：

$$
\begin{aligned}(\hat s,\hat a)&\sim p_G,\qquad \hat s'=\operatorname{stopgrad}(T_{\rm old}(\hat s,\hat a)),\\L_{\rm dyn}(\psi)&=\mathbb E_{(s,a,s')\sim D_i}\|T_\psi(s,a)-s'\|^2+\lambda\mathbb E_{(\hat s,\hat a)\sim p_G}\|T_\psi(\hat s,\hat a)-\hat s'\|^2.\end{aligned}
$$

第一项校正当前可见区域，第二项保护过去区域的模型输出。与 CLEAR 的区别在于教师输出是后继状态预测，输入则由生成器产生。

生成器自身也会遗忘，因此用当前真实状态—动作对与上一生成器产生的样本共同训练新生成器。DRAGO 还用真实探索重新连接旧区域；这一部分补充了生成数据无法证明当前真实可达性的缺口。保留与重新获取知识是两种互补操作。

**算法：DRAGO 的知识保留流程；策略、价值、奖励头及探索器仍有各自训练目标**

1. 在已知任务边界保存冻结副本 $T_{\rm old}$ 与 $G_{\rm old}$
1. 每个训练阶段：
  1. 收集当前任务的真实转移，维护当前 buffer
  1. 从生成器采样旧状态—动作对，计算冻结模型的后继标签
  1. 以真实项与合成项联合更新当前动力学模型
  1. 以当前真实输入和 $G_{\rm old}$ 的样本训练当前生成器
  1. 用任务行为与记忆恢复行为收集新的真实经验
1. 下次任务切换时更新冻结副本

这个方法的边界来自问题设定：若真实动力学永久改变，旧模型标签可能已经错误，保持它会妨碍适应；若生成器遗漏了某个区域，蒸馏便没有对应输入。实验应分别测模型保持误差、跨区域规划与新任务适应，并计入生成器和冻结副本的存储、训练成本。作者 drago 仓库提供完整模型式实现；本页脚本仍聚焦 reservoir、EWC 和 CLEAR 的数值机制。

<a id="lesson-example"></a>

## 6 · 稳定性与适应性的二次例子

$$
L(w)=\tfrac12(w+1)^2+\tfrac\kappa2(w-1)^2,\qquad \frac{dL}{dw}=(w+1)+\kappa(w-1),\qquad w_*=\frac{\kappa-1}{\kappa+1}
$$

旧目标是 w=1，新目标是 w=−1，F=1。κ=0、1、5 分别得到 −1、0、2/3；这是精确二次例子，不是深网保证。

κ=0 完全适应新任务；κ=1 平均折中，但两个任务都不完美；κ=5 更接近旧解，却留下较大的新误差。这说明“新任务回报低”可能是强保留目标的结果，不一定是网络失去学习能力。

功能克隆的例子：旧策略 $\mu=(0.8,0.2)$，新策略 $\pi=(0.5,0.5)$，故 $D_{\rm KL}(\mu\Vert\pi)=0.8\log1.6+0.2\log0.4\approx0.192745$。旧值为 2、新值为 1 时，值克隆平方误差为 1。交换 KL 方向会改变数值与梯度，教师和学生的角色须固定。

两步 V-trace 样本的奖励为 $[0,1]$，价值全零、$\gamma=0.9$，所有比率与裁剪系数为 1，则第一状态目标为 $0.9$。若第一步目标策略概率为零，$\bar\rho_0=c_0=0$，第一状态本次没有轨迹修正，保持原值。这展示了当前误差与后续误差乘积的边界。

<a id="lesson-code"></a>

## 7 · 可运行实现与实验设计

固定容量 reservoir、EWC 精确小例子、克隆项与有限 unroll V-trace；不是完整 CLEAR 神经网络代理。

```python
def reservoir_insert(buffer, item, seen, capacity, rng):
    """seen is the number INCLUDING this item; O(capacity) retained data."""
    if capacity < 0 or seen < 1:
        raise ValueError("invalid budget or sample counter")
    if len(buffer) < capacity:
        buffer.append(item)
    elif capacity:
        index = rng.randrange(seen)
        if index < capacity:
            buffer[index] = item


def ewc_scalar_optimum(new_target, old_weight, importance, strength):
    if importance < 0 or strength < 0:
        raise ValueError("curvature and strength must be nonnegative")
    k = importance * strength
    return (new_target + k * old_weight) / (1.0 + k)


def ewc_update(weights, current_gradient, old_weights, importance,
               strength=1.0, alpha=0.1):
    """One SGD update; current_gradient excludes the consolidation penalty."""
    if not (len(weights) == len(current_gradient) == len(old_weights) == len(importance)):
        raise ValueError("EWC dimensions differ")
    if strength < 0 or any(f < 0 for f in importance):
        raise ValueError("EWC requires nonnegative importance")
    return [w - alpha * (g + strength*f*(w-old))
            for w, g, old, f in zip(weights, current_gradient, old_weights, importance)]


def clone_loss(teacher_prob, student_prob, teacher_value, student_value):
    """CLEAR-like replay cloning terms only, not the complete RL objective."""
    if len(teacher_prob) != len(student_prob):
        raise ValueError("policy dimensions differ")
    if any(p < 0 for p in teacher_prob) or any(q <= 0 for q in student_prob):
        raise ValueError("teacher >=0, student >0 required")
    if not math.isclose(sum(teacher_prob), 1.) or not math.isclose(sum(student_prob), 1.):
        raise ValueError("policies must sum to one")
    kl = sum(p * math.log(p / q) for p, q in zip(teacher_prob, student_prob) if p)
    return kl, (student_value - teacher_value) ** 2


def vtrace_target(rewards, values, rhos, gamma=0.9, rho_cap=1., c_cap=1.):
    """Finite unroll value target. values includes the final bootstrap value."""
    if len(values) != len(rewards) + 1 or len(rhos) != len(rewards):
        raise ValueError("unroll dimensions differ")
    if min(rho_cap, c_cap) < 0 or c_cap > rho_cap or any(rho < 0 for rho in rhos):
        raise ValueError("require 0 <= c_cap <= rho_cap and nonnegative ratios")
    correction, product = 0., 1.
    for t, reward in enumerate(rewards):
        delta = min(rho_cap, rhos[t]) * (reward + gamma * values[t+1] - values[t])
        correction += product * delta
        product *= gamma * min(c_cap, rhos[t])
    return values[0] + correction


def retention_demo():
    buffer, rng = [], random.Random(7)
    for seen in range(1, 101):
        reservoir_insert(buffer, seen, seen, 5, rng)
    weight = [1.]
    for _ in range(100):
        weight = ewc_update(weight, [weight[0]+1.], [1.], [1.], strength=1.)
    print("retention", {"fixed_budget_reservoir": buffer,
          "ewc_optima_k_0_1_5": [ewc_scalar_optimum(-1, 1, 1, k) for k in (0, 1, 5)],
          "ewc_sgd_k1": round(weight[0], 8),
          "clone_terms": clone_loss([0.8, 0.2], [0.5, 0.5], 2., 1.),
          "two_step_vtrace": vtrace_target([0, 1], [0, 0, 0], [1, 1])})
```

运行后先核对手算，再增加新旧分布和 buffer 预算

```sh
python lifelong_algorithms_lab.py retention
python lifelong_algorithms_lab.py test
```

脚本检查固定容量、零容量、二次目标的驻点、教师与学生相同时的零克隆损失，以及 V-trace 的目标策略零概率边界。完整 CLEAR 还包含序列收集、分布式 actor 与神经网络更新；这些应在机制函数正确后逐层加入。

- 最小研究矩阵：online-only、FIFO replay、reservoir replay、replay+clone、EWC；固定总字节数与每步梯度计算。
- 评价至少三项：当前适应曲线、旧任务保持/重学速度、全生命期累计收益；另记真实回访与诊断评测的区别。
- 当环境永久改变，增加“有意遗忘”的对照：统一历史 replay 是否持续施加过时约束？
- 把任务 ID、教师 checkpoint、离线旧数据访问列为显式资源，不与无需这些信息的方法直接混淆。

<a id="lesson-branches"></a>

## 8 · 数据、参数、功能、模块与快慢学习

| 条线 | 保存对象和更新机制 | 主要代价 |
| --- | --- | --- |
| Replay / CLEAR | 旧样本与旧输出；混合 RL 学习及功能约束 | 观测存储、旧分布偏差、多步行为校正 |
| EWC / online EWC | 旧参数与重要性；加二次惩罚 | 局部/对角近似、consolidation 调度 |
| Distillation | 旧策略或表示输出；在指定输入上对齐 | 需要有代表性的输入；教师错误也会保留 |
| Progressive / modular | 把新知识放在新或路由模块中，限制相互干扰 | 模块增长、任务识别与路由、知识复用成本 |
| Progress & Compress | 快的 active column 学习，再蒸馏进慢的 knowledge base | 双系统预算、压缩时间、压缩时旧知识保护 |

快慢系统的本质是时间尺度分离：快速部分吸收近期证据，慢速部分整合可复用信息。若压缩只在新状态上进行，旧功能可能没有监督；如果每次任务都增长模块，就必须把无限寿命下的容量问题列出来。认知科学的互补学习系统可提供设计动机，但动机并不能替代算法和实验的因果证据。

一个可检验的研究问题是：在无任务 ID 的循环环境中，总内存固定为 10 MB，将一部分观测存储预算改为保存旧 logits，是否改善全程收益与回访表现？相同原则也适用于 DRAGO：生成器、旧模型和少量真实 replay，应在同一存储预算下比较。

<a id="lesson-check"></a>

## 9 · 习题与下一步

- 旧任务分数下降、新任务学习很快，说明什么？可能是遗忘或必要适应，不足以判断可塑性丧失。
- Reservoir 是否让每个任务等权？否，它让各时刻样本等概率；较长任务会占更多份额。
- 已知 μ(a|s) 能否修正所有历史偏差？不能；动作比率不修正环境转移本身已经改变，也不恢复从未保存的状态。
- 为什么保存 KL 教师也要计内存？每条样本的 logits/value 与教师 checkpoint 都是真实资源。
- 如果 EWC 的 F 全零会怎样？二次约束消失，退化为当前任务学习；若 κ 极大，则会压制必要适应。

动手题：用同一组 A→B→A 样本分别做 FIFO 和 reservoir，比较 buffer 内各阶段比例；再把 B 的长度扩大十倍，检查所谓“平衡历史”是否仍成立。先用确定性样本查清保存分布，再把真实 RL 策略反馈加进去。

## 本章的实验设计

冻结副本测试保留能力。继续学习的副本测试再适应。副本中的反馈不能写回主运行。

设定：使用 A→B→A，并在回访前复制隔离的评测分支。一个分支冻结，另一个允许新增学习。加入未见 C 以防“少遗忘但学不动”。

- A 的初次习得达到预定非平凡标准。
- 冻结回访样本不进入主 buffer 和归一化。
- 保存、回放与模型增长计入固定预算。

对照：无保留机制、固定容量 replay；相同存储与更新预算的 EWC/蒸馏；独立任务顺序与相同信息权限

记录：冻结保留与允许重学后的恢复曲线；B/C 获取速度和完整生命期回报；缓存、参数、蒸馏与梯度成本

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-continual)

## 学习与研究衔接

记住旧任务和快速学习新任务可能冲突。冻结诊断与继续学习的再适应实验分别回答不同问题。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-retention) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=retention) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=retention)

<a id="chapter-code"></a>

## 下载与运行

存储策略、目标项和 trace 数值实验；完整 CLEAR 训练还需要神经网络、序列收集与环境配置。

[下载 lifelong_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)

```sh
python lifelong_algorithms_lab.py retention
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Rolnick et al. · Experience Replay for Continual Learning (CLEAR)](https://papers.nips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html)：新旧经验混合、off-policy 学习与行为/价值克隆的原论文。

- [Espeholt et al. · IMPALA](https://proceedings.mlr.press/v80/espeholt18a.html)：V-trace 原始推导及裁剪系数所控制的偏差、方差与目标策略。

- [Kirkpatrick et al. · Overcoming catastrophic forgetting in neural networks](https://doi.org/10.1073/pnas.1611835114)：EWC 的概率解释与 Fisher 近似；注意不同 Fisher 估计器的语义。

- [Schwarz et al. · Progress & Compress](https://proceedings.mlr.press/v80/schwarz18a.html)：固定架构大小的 active/knowledge 双系统与 online EWC 压缩。

- [Fu et al. · Knowledge Retention in Continual Model-Based RL · ICML 2025](https://proceedings.mlr.press/v267/fu25f.html)：DRAGO 的生成回放与真实记忆恢复；共享动力学、已知任务边界，当前任务数据仍可重放。

- [DRAGO 作者实现](https://github.com/YixiangSun/drago)：对应生成器、冻结旧动力学模型和任务/回访行为；与仅用 reservoir 的模型自由方法有不同预算。

- [AGI-Labs continual_rl](https://github.com/AGI-Labs/continual_rl)：可扩展 CRL 比较框架，包含统一接口下的方法复现；实现来源与原论文作者仓库分别标识。
