# 持续探索：新奇、不确定性、学习进展与恢复

外部奖励稀疏、世界持续改变时，怎样获得有用的新经验，而不是追逐永远无法学会的噪声？

## 本章内容

- 从计数不确定性和预测误差分别推导 count bonus 与 RND，明确 reward 在更新前还是更新后计算。
- 区分新奇、信息增益、学习进展、技能多样性和自动课程这些不同目标。
- 将探索放回无免费 reset 的世界：记录恢复成本、可达性与真实外部收益，而不是只累计 intrinsic reward。

<a id="problem-definition"></a>

## 本章的问题定义

行为既获得外部收益也决定未来能学到什么；在稀疏反馈、漂移或无免费重置条件下选择有用且可恢复的经验。

### 给定条件与符号

- 外部任务、可达动作、反馈及恢复/重置权限。
- 探索先验或内部信号类、可保存统计、训练与恢复预算。

### 需要求解的对象

服务声明外部目标的数据获取策略；新奇、不确定性、进展与技能多样性是不同候选代理。

### 信息与数据权限

$r_t^{\rm ext}$ 是外部反馈，$r_t^{\rm int}$ 由已到达经验和当前统计生成；训练信号在更新前或后计算的约定必须明确。

$$
J^{\rm ext}_T(L)=\mathbb E_L\!\left[\sum_{t=0}^{T-1}r_t^{\rm ext}\right],\qquad r_t^{\rm train}=r_t^{\rm ext}+\beta_t r_t^{\rm int}
$$

$L$ 是完整探索与学习过程，$T$ 为寿命，$\beta_t$ 为内部信号权重。第一式定义此处有限寿命外部评价，第二式只是常见训练代理；内部回报增加不能代替外部收益、可达性或信息增益。

### 成立条件与解的含义

- 认识不确定性与环境随机性分别讨论；预测误差包含模型限制和遗忘。
- 无重置问题明确先前知识、恢复权限和不可逆失败；可重置基准不能免费证明single-life效果。

判断准则：匹配预算记录真实覆盖、原始外部收益、恢复步数和不可逆失败；在随机噪声区/可学习区对照代理，并用固定probe区分新奇与预测器遗忘。

### 适用边界

- 随机参数或动作噪声不自动构成校准后验。
- 高RND误差不等于高信息增益；恢复数据也不能假装免费重置。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：探索改变后续世界和训练分布，完整控制需保留探索损失与长期收益。

- 组合不同学习问题 · [目标条件化与子任务构造](https://yingwen.io/zh/continual-rl/construction/goals/)：课程选择目标改变真实数据获取，底层目标条件策略另学如何达成。

- 组合不同学习问题 · [元学习与学习规则的适应](https://yingwen.io/zh/continual-rl/algorithms/meta/)：探索信号或选择规则可由外部寿命评价学习，但需计入元训练预算。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

外部奖励无法指出应访问哪里；高预测误差可能来自不能学会的噪声或内部遗忘。

### 本章的核心思路

为所需数据价值选择对应代理，再用覆盖、学习与恢复的独立量检查其用途。

1. [估计访问不足或熟悉程度](#lesson-rnd)：因为尚未覆盖的行为可能有价值，计数与RND分别用访问统计和固定目标拟合构造代理，计时先定义。

2. [区分误差与可学习进展](#lesson-progress)：因为永远噪声可保持高误差，用匹配probe的误差/成功率变化选择练习目标。

3. [将恢复与不可逆代价纳入](#lesson-recovery)：因为无法免费回到起点，探索还需检查返回安全集的能力，原始收益计入真实恢复过程。

结论与条件：计数递推/RND梯度可核验，但bonus与误差只是代理；后验或探索保证需额外模型与采样条件，single-life结果限于声明权限。

### 相关方法改变了什么

- 计数/RND：分别代理覆盖与随机函数熟悉程度，可能受表示漂移或遗忘影响。

- 后验/不确定性探索：面向相容价值假设，需区分认识与回报随机性。

- 进展课程/恢复策略：前者分配可学习目标，后者处理真实可达性和失败成本。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 外部奖励与内部奖励

外部奖励定义任务成功；内部奖励由学习器构造，引导获得数据。提高内部奖励不自动意味着外部目标更好。

### 认识不确定性与随机性

认识不确定性可通过更多数据减少；环境固有随机性可能不能减少。单次预测误差往往混合两者。

### 目标条件策略

$\pi(a\mid s,g)$ 把目标 $g$ 当作输入；课程决定练习的目标，控制器决定到达方法。课程调度与底层策略学习处于不同层次。

<a id="lesson-setting"></a>

## 1 · 探索的数据获取目标

在长走廊尽头才有奖励，ε-greedy 的零散随机动作可能始终无法形成连续前进。若环境还会改变，以前足够的经验也会过时。但用新奇奖励强迫 agent 去任何陌生地方，同样可能进入死胡同，或者一辈子盯着随机闪烁的屏幕。真正的问题是：哪些行动会带来未来仍有用、可学习、可恢复的经验？

$$
r_t^{\rm train}=r_t^{\rm ext}+\beta_t r_t^{\rm int}
$$

这是常见实现接口，而非所有探索算法的定义。若 β 随时间变，训练目标也在变；报告原始外部收益时不能把内部分数混进去。

| 要估计的对象 | 典型信号 | 容易混淆的失败 |
| --- | --- | --- |
| 访问不足 | 计数或密度模型更新幅度 | 表示每次变化产生假新奇 |
| 尚未学会 | 固定随机目标的预测误差 | 预测器遗忘使旧状态再次显得陌生 |
| 正在学会 | 固定 probe 上误差/成功率改进 | 噪声波动被当成进展 |
| 不同技能 | 状态对 latent skill 的可识别性 | 多样但不一定对任务有用 |
| 可恢复性 | 返回安全集的概率/成本 | 估计乐观却进入不可逆失败 |

计数估计覆盖程度，RND 估计随机函数的熟悉程度，课程选择练习目标，恢复机制限制可接受风险。这些方法作用于不同决策层次；组合时需分别定义其目标、更新数据和资源预算。

<a id="lesson-derive"></a>

## 2 · Count bonus：少见为什么值得再看？

在独立同分布、有限方差的样本均值估计中，标准误差按 $1/\sqrt N$ 缩小。表格探索由此把状态动作访问次数作为不确定性的粗代理，对少见选择增加奖励。完整的探索保证还需要奖励界、转移模型、置信水平与规划条件。

$$
b_t(s,a)=\frac{\beta}{\sqrt{N_t(s,a)+1}},\qquad N_{t+1}(s,a)=N_t(s,a)+\mathbf1[(S_t,A_t)=(s,a)]
$$

此约定先用访问前计数 Nt 计算本次奖励，再计数加一；+1 定义首次访问。若先加一，就应相应改写公式，避免 off-by-one。

像素状态几乎不重复，直接计数会把每帧当作首次访问。可在固定表示上哈希计数，或者用密度模型提供会泛化的伪计数。令 $p(x)$ 是学习当前 $x$ 之前的概率，$p'(x)$ 是更新一次之后的概率；假想二者分别等于 $\widehat N/\widehat n$ 与 $(\widehat N+1)/(\widehat n+1)$，解这两个方程得到：

$$
\widehat N(x)=\frac{p(x)(1-p'(x))}{p'(x)-p(x)}
$$

需要 p′≥p 的 learning-positive 条件使计数非负；p′=p 的极限对应无学习增量，不能直接除零。深度密度优化不总满足这个条件，实际实现必须处理。

如果表示也不停学习，同一个物理状态可能被重新映射到未访问位置。此时 bonus 反映的是编码漂移，不一定是真实知识增加；固定随机表示、冻结评测编码和原始状态覆盖可帮助分离这两者。

<a id="lesson-rnd"></a>

## 3 · RND：把熟悉程度变成一个可训练的预测任务

Random Network Distillation 初始化两个网络：目标网络 $f_\xi$ 随机生成后永久冻结，预测网络 $f_\theta$ 学习拟合它。对刚看到的观测，两个输出的差作为新奇信号；重复观察后预测误差逐渐下降。目标固定，是它与预测随机环境下一帧的关键区别。

$$
\begin{aligned}r_t^{\rm int}&=\|f_\theta(o_{t+1})-f_\xi(o_{t+1})\|_2^2,\qquad \xi\ \text{冻结},\\L_{\rm pred}&=\tfrac12\|f_\theta(o_{t+1})-f_\xi(o_{t+1})\|_2^2,\\\theta_+&=\theta-\alpha J_\theta(o_{t+1})^\top[f_\theta(o_{t+1})-f_\xi(o_{t+1})].\end{aligned}
$$

Jθ 为预测网络输出的 Jacobian。intrinsic reward 先由更新前 θ 计算并保存，再训练预测器；不能在反向传播后重算奖励却不说明目标时序。

**算法：算法伪代码**

1. 初始化冻结的随机 target ξ、可训练 predictor θ 和行为学习器
1. 循环收集经验：
  1. 行为策略给动作，环境返回 o′ 与外部奖励
  1. 按规定在线统计归一化 o′；target 始终 stop-gradient
  1. 用更新前 predictor 计算并保存 $r_{int}=\|\mathrm{predictor}(o^{\prime})-\mathrm{target}(o^{\prime})\|^2$
  1. 以 $r_{ext},r_{int}$ 及各自 discount 更新价值/策略目标
  1. 用指定数量样本和更新次数训练 predictor
  1. target 保持冻结；跨回合保留新奇统计

原始 RND 使用 actor–critic/PPO 系统，并区分外部与内部奖励的价值估计。完整实现还包含观测归一化、内部奖励缩放、预测器更新比例、两个折扣因子、终止语义与优势混合。本页表格示例只保留预测误差与更新次序，完整神经网络实验使用文末作者代码。

固定 target 降低了一类“预测环境随机下一帧”问题，却不消除全部噪声陷阱：随机像素可以产生不断变化的输入，预测器容量和覆盖仍有限。更重要的是，持续学习中 predictor 也会遗忘；旧状态预测误差重新升高可能只是内部退化。评价应同时检查覆盖和预测器保留，而不是把每一次高误差都叫信息增益。

<a id="lesson-uncertainty"></a>

## 4 · LSAC：通过价值函数的不确定性探索

新奇程度与决策不确定性不同。一个从未见过但显然不可达的画面可能有很高 RND 误差；两个动作的长期收益难以区分，即使画面熟悉，也可能值得进一步尝试。后验采样用一组与数据相容的价值函数表示这种认识不确定性，采样其中一个，再让策略根据它选择行动。

Langevin Monte Carlo 为这种思路提供一种参数空间采样方法。设固定数据上的能量函数为 $L(w)$，逆温度为 $\beta>0$。沿损失梯度下降并加入高斯噪声：

$$
w_{k+1}=w_k-\eta\nabla L(w_k)+\sqrt{2\eta/\beta}\,\epsilon_k,\qquad\epsilon_k\sim\mathcal N(0,I).
$$

连续时间扩散在适当正则条件下以 $p(w)\propto\exp[-\beta L(w)]$ 为不变分布。有限步长离散化存在偏差；只有能量确实对应负对数后验、且采样充分时，才具有相应的贝叶斯后验解释。RL 的目标还会改变，因此实际使用是近似。

以一维能量为例：$L(w)=(w-2)^2/2$，取 $w=0$、$\eta=0.1$、$\beta=10$。梯度项把参数移到 0.2，噪声标准差为 $\sqrt{0.02}$。噪声使多次独立运行保留不同假设；这与把同一个高斯噪声直接加到环境动作上不同。

ICLR 2025 的 Langevin Soft Actor-Critic（LSAC）把这一思路用于分布式回报 critic，再让 SAC actor 面对采样出的 critic 更新。完整方法还使用自适应 Langevin 漂移、多个参数链，以及扩散模型生成的样本与动作梯度修正。论文称其多链方案为简化 parallel tempering，但实际采用同温、不同初始化且不交换副本；它与经典不同温度的交换算法有所区别。

**算法：算法伪代码**

1. 维护真实经验库、生成样本库、多个 critic 及其目标网络
1. 从真实与生成经验中构造训练 batch
1. 对每个 critic 计算分布式 Bellman 损失，执行自适应 Langevin 参数更新
1. 从多个 critic 中采样一个，更新最大熵 actor 和温度参数
1. 按配置更新生成器、目标网络和样本库
1. 分别记录外部回报、critic 差异、计算量与实际访问覆盖

需要区分两类方差：一个状态下随机回报的方差可以是不可减少的环境随机性；不同可信参数模型之间的差异才可能反映认识不确定性。分布式 critic 不会自动完成这种分解。LSAC 的实验证据来自可重置的连续控制任务；经验库、生成器和多 critic 的资源开销，以及这些估计在长期漂移中的有效性，仍需在 CRL 条件下单独研究。

<a id="lesson-progress"></a>

## 5 · 学习进展与自动课程

$$
\operatorname{LP}_k=E_{k,\rm before}-E_{k,\rm after},\qquad \operatorname{ALP}_k=|E_{k,\rm before}-E_{k,\rm after}|
$$

E 应在匹配的任务/目标和可比较的 probe 上估计。正向 LP 强调改善；绝对进展也会把退步当作需要重新关注的信号。二者不是同一种调度规则。

想象两个区域：随机噪声区误差一直是 4，学习进展为 0；一个可学习区误差从 2 降到 1，进展为 1。若只追求误差，会偏向噪声区；若追求进展，会偏向可学习区。但有限样本中的噪声涨跌也会伪造进展，必须平滑、匹配评测数据或估计置信区间。

离散课程的最小实现是：维护每个目标的近期与较早成功率；按进展设置采样权重，并保留非零随机探索概率；学生完成一次练习后更新该目标统计，再选下一个。ALP-GMM 将这一思想扩展到连续环境参数空间：用附近历史参数的表现估计绝对进展，在参数—进展空间拟合混合模型，偏向有进展的区域采样，同时保留随机探索。

**算法：算法伪代码**

1. 维护目标集合或可采样的连续参数空间
1. 初始化各目标的历史表现、近期表现与非零探索权重
1. 每轮：按学习进展 + 探索机制选目标/环境参数 g
  1. 学生用相同固定预算练习 g，记录外部成功率或回报
  1. 与可比较的历史表现计算 LP / ALP
  1. 更新课程模型；下一轮再采样
1. 评测始终使用预先约定的目标分布，而不是只评教师偏爱的任务

在仿真中教师可以生成新地图并 reset 到起点；现实单生命期通常没有这种权限。将课程移植到 single-life，必须把“选目标”变成当前可达目标的选择，加入到达和恢复的真实成本。随机参数生成器提供的环境控制权是一项资源，不能隐去。

<a id="lesson-skills"></a>

## 6 · 技能发现与时间一致的探索

有时需要的不是更高的一步 bonus，而是能连续执行很久的行为。技能发现可以使不同 latent z 对应不同可辨认的访问分布。以 DIAYN 风格目标为例，固定先验 p(z)，训练判别器 qψ(z|s) 分辨当前状态来自哪个技能；策略获得 log qψ(z|s)−log p(z) 的内部奖励，并以最大熵控制鼓励技能内部的动作多样性。

$$
I(S;Z)=\mathbb E[\log p(z\mid s)-\log p(z)]\ \geq\ \mathbb E[\log q_\psi(z\mid s)-\log p(z)]
$$

下界来自条件 KL 非负。判别器通过带技能标签的状态训练，技能策略把该下界中的项作为奖励；这提供多样性目标，不保证技能对应人类语义或外部奖励。

技能本身可以提高 temporally extended exploration，但何时开始、何时终止、如何选择技能仍需控制器；固定 latent 一个回合与可学习 option termination 不是同一个机制。判断技能是否有用，应测外部任务学习加速和模型规划收益，而非只展示不同轨迹颜色。

<a id="lesson-recovery"></a>

## 7 · 单生命期探索与恢复约束

在模拟器中，跌进坑之后 reset() 即可；在现实中，卡住、断电或不可逆损坏会改变此后全部数据分布。探索可以维护一个恢复策略与恢复价值，估计从候选动作后能否回到安全或可维持后续交互的状态集合。若风险过大则切换到恢复动作，或停止冒险并请求外部干预。

$$
\mathcal A_{\rm admissible}(s)=\{a:\widehat P(\text{恢复成功}\mid s,a)\geq 1-\varepsilon\}
$$

这只是一个简化的决策接口。用学习估计 P̂ 过滤并不自动给真实安全保证；需要校准、保守不确定性界、分布外检测或外部安全机制。

“Leave no Trace”同时学习前向任务与 reset 行为，展示了恢复应成为学习系统的一部分。不同任务中的失败集与允许干预必须明确。研究报告至少记录恢复成功率、恢复耗时、不可逆失败和人工干预次数；不能只保留 forward policy 的成功回合。

<a id="lesson-example"></a>

## 8 · 手算：新奇奖励的衰减

把 RND 缩小为某个表格状态的一个标量输出：冻结 target=2，predictor 初值 0，半平方损失梯度步长 α=0.25。第一次看见，先得到 bonus=4，再更新预测为 0.5；第二次 $bonus=(2-0.5)^2=2.25$，更新为 0.875；之后两次 bonus 分别为 1.265625、0.711914。

若误把 predictor 每回合清零，相同旧状态又得到 4。这个高分不是环境提供了新知识，而是实现制造的遗忘。同样，count bonus 在访问前 N=0、3、8 时分别为 1、0.5、1/3；更新计数的时刻必须固定。

恢复例子：动作 0 的已知恢复概率是 0.99，动作 1 是 0.2，阈值 0.9。即便探索器提议动作 1，示例 filter 也只能返回动作 0；如果所有动作都不满足阈值，它返回 None，而不是随意宣称最不坏动作安全。这一保守接口应保留到真实系统。

<a id="lesson-code"></a>

## 9 · 奖励时序实现与探索诊断

计数、更新前 RND reward、正向进展和已知恢复概率的最小实现。

```python
def count_bonus(count, scale=1.):
    if count < 0:
        raise ValueError("count must be nonnegative")
    return scale / math.sqrt(count + 1.)


def rnd_step(prediction, fixed_target, alpha=0.25):
    """One table feature: novelty is measured BEFORE fitting this sample."""
    error = fixed_target - prediction
    return error * error, prediction + alpha * error


def progress_score(old_error, new_error, floor=0.):
    """Positive progress, not raw surprise; both errors use a matched probe."""
    return max(floor, old_error - new_error)


def recovery_filter(proposed_action, recovery_probabilities, threshold):
    """Toy known-probability filter; learned estimates give no safety guarantee."""
    if recovery_probabilities[proposed_action] >= threshold:
        return proposed_action
    feasible = [a for a, prob in enumerate(recovery_probabilities)
                if prob >= threshold]
    return max(feasible, key=lambda a: recovery_probabilities[a]) if feasible else None


def exploration_demo():
    predictor = 0.0
    novelty = []
    for _ in range(4):
        bonus, predictor = rnd_step(predictor, 2.0)
        novelty.append(round(bonus, 6))
    print("exploration", {"rnd_bonus_before_fit": novelty,
          "fake_novelty_if_predictor_reset": rnd_step(0., 2.)[0],
          "count_0_3_8": [count_bonus(n) for n in (0, 3, 8)],
          "noisy_vs_learnable_progress": [progress_score(4., 4.), progress_score(2., 1.)],
          "filtered_action": recovery_filter(1, [0.99, 0.2], 0.9)})
```

预期 RND bonus 为 4、2.25、1.265625、0.711914；代码不执行真实危险动作

```sh
python lifelong_algorithms_lab.py exploration
python lifelong_algorithms_lab.py test
```

本页示例实现奖励顺序、冻结目标、相同状态的学习进展和无可行动作的边界。完整 RND、ALP-GMM 与 LSAC 训练系统由文末作者仓库提供；这些系统的神经网络训练与性能结果不属于本页小实验的复现范围。

- 将 intrinsic reward、extrinsic reward、状态覆盖、预测误差和实际学习进展分别记录。
- 固定 predictor 训练次数和容量；否则更慢的预测器可能仅因“总是没学会”得到更高内部奖励。
- 在重复访问旧区域时评测 RND 是否发生假新奇；比较冻结表征与学习表征。
- 课程基线至少包含随机目标、均匀目标、固定难度，统一学生训练预算和最终评价目标分布。
- 单生命期报告全程数据：成功、恢复、失败和干预，不只挑可 reset 的局部片段。

<a id="lesson-branches"></a>

## 10 · 方法选择与实验条件

| 观察到的瓶颈 | 优先切入 | 关键对照 |
| --- | --- | --- |
| 同一少量状态循环 | count/RND 与覆盖度 | 内部奖励大小不等于真正覆盖 |
| 远处奖励到不了 | 时间一致的探索技能 / options | 同样原始步预算，不只比技能决策次数 |
| 困难任务总学不会 | learning-progress curriculum | 区分不可学与只是暂时困难 |
| 旧区域反复显得新奇 | 预测器保留/表征漂移 | 冻结 target 仍可能 predictor 遗忘 |
| 一次失败就无法继续 | 恢复、风险和可达目标 | 干预和 reset 费用计入整个生命期 |

<a id="research-cpsrl-resampling"></a>

## 研究专题 A · CPSRL：世界不重置，探索假设可重采样

每步换一个可信模型，行动可能相互抵消；永久坚持初始抽样，又可能长期错过新证据。CPSRL（RLC 2024）以随机时钟决定更换整条探索假设，时钟不会调用环境 reset。

$$
\Pr(L=\ell)=p(1-p)^{\ell-1},\quad\Pr(L>k)=(1-p)^k,\quad\mathbb E[\sum_{k=0}^{L-1}R_{t+k+1}]=\mathbb E[\sum_{k\ge0}(1-p)^kR_{t+k+1}]
$$

L 为至少一的几何持续时间。右侧奖励来自永久保持本次抽样策略的反事实轨迹，而非重采样之后的实际新策略；时钟独立于该轨迹。奖励有界等条件允许按存活概率交换求和。

所以 $\gamma=1-p$ 的规划对应随机长度试验的期望未折扣收益。$p=0.1$ 时平均承诺 10 步，第 5 个奖励纳入概率为 $0.9^4$。p 控制算法节奏，外部评价仍可使用整个真实流的未折扣收益。

**算法：探索时钟接口；规划目标与重采样时间需匹配**

1. 初始化先验与抽样模型
1. 按当前抽样模型规划并行动，真实后果更新 posterior
1. 独立时钟决定下一步是否重新抽样模型/策略
1. 不清零网络、经验或物理状态
1. ensemble 替代 posterior 时，另说明近似方式

$$
\gamma=1-\sqrt{SA/T},\qquad\mathbb E[\operatorname{Regret}(T)]=\widetilde O(\tau S\sqrt{AT})
$$

原文 Theorem 4.1 的已知时域选择；T≥SA 使 γ 非负。S、A 是表格规模。未知时域采用原文 Appendix B、C 的调度，不能让任意固定 p 承担这个渐近保证。

$$
\left|\mathbb E_{\pi^*_{\mathcal E}}\!\left[\sum_{t=0}^{T-1}R_{t+1}\mid\mathcal E,S_0=s\right]-Tg^*_{\mathcal E}\right|\le\tau\quad\text{for all }T,s
$$

τ 是先验支持环境中最优平均奖励策略的统一 reward-averaging bound（原文 Assumption 3.2）。它约束期望累计收益与长期线性收益之差；不是所有策略的 mixing time，也不是样本均值的置信收敛时间。

该理论依赖平稳有限、弱连通 MDP，正确后验与规划，以及上述统一有界条件。任意固定 p 的深网 ensemble 不直接继承表格界。Bayesian regret 也不等于任意漂移路径上的动态遗憾。

- 已知小 MDP 用精确后验/规划，比较逐步换、固定长度和几何长度；记录远奖励到达概率。
- 匹配模型更新次数、动作预算与规划精度，计入重规划成本。
- 漂移实验另定义 posterior 遗忘或变点机制；只改时钟不会删除过时证据。
- 未确认原作者完整仓库，此处仅提供原文及算法，不猜测代码链接。

<a id="research-morefree-data-and-goals"></a>

## 研究专题 B · MoReFree：真实探索与模型内目标分布

无 reset 世界中，最大化覆盖可能长期停留在任务无关区域。MoReFree（TMLR 2025）同时改变真实目标调度与 imagination training：任务目标、返回初始区域和探索目标彼此配合。返回由真实动作实现，日志块边界不会将物理世界复位。

$$
\rho_{\rm imag}={\alpha\over2}\rho_{g^*}+{\alpha\over2}\rho_0+(1-\alpha)\rho_{\rm replay}
$$

模型内 goal-conditioned policy 的采样分布：评测目标、初始状态与 replay 状态。它不是实际状态占用分布。

真实采样又不同：概率 α 执行前向目标与返回目标的一组 Go-Explore，概率 1−α 执行探索目标的 Go-Explore。相关 pair 耗时可为单个探索块的两倍，因此调度事件概率不等于原始时间占比。

$$
\operatorname{TimeShare}_{\rm relevant}\approx{\alpha\mathbb E[T_{\rm pair}]\over\alpha\mathbb E[T_{\rm pair}]+(1-\alpha)\mathbb E[T_{\rm explore}]}
$$

可积、稳定调度循环下的时间比例诊断。pair 耗时两倍、α=0.2 时份额约为 1/3；重尾或不稳定恢复应直接统计真实日志。

作者 resetfree/env.py 的分块 done 让 PEG 切换阶段，但保留物理环境；goal_picker_wrapper.py 决定调度。它是算法阶段边界，不能直接当作外部任务真实终止，也不应称为环境 reset。

| 消融 | 机制问题 |
| --- | --- |
| 只改真实调度 | 是否获得更多任务相关数据？ |
| 只改模型内目标 | 相同数据是否训练更有用的返回与前向策略？ |
| 共同改变 | 是否存在数据与训练目标的交互？ |
| 匹配原始步与模型计算 | 收益是否只是 pair 更长或虚拟训练更多？ |

**算法：区分论文能力评价与新增全生命评价**

1. 一次初始化后保留物理世界与学习器
1. 调度前向+返回 pair 或探索块，记录真实耗时
1. 保存数据并更新 world model，按指定分布训练条件策略
1. 评测副本：按论文的给定起点测试目标到达
1. 补充 CRL 评测：原生命收益、返回率、恢复时间与不可逆失败

已有证据支持 reset-free 训练与目标到达，但目标/初始分布由设计者给定、主要评测仍 episodic，world model 与 replay 均占资源。进一步 CRL 研究应检验未通知变化后如何更新返回区域与探索价值；真实机器人恢复保证还需独立验证。

<a id="lesson-check"></a>

## 11 · 自测与动手题

- RND error 就是 Bayesian 信息增益吗？不是，它是固定随机函数的拟合误差，受表示、容量、优化和遗忘影响。
- 环境噪声大就一定值得探索吗？不一定；固有随机性可能不提供可减少的不确定性。
- ALP 为何会关注退步区域？绝对变化将遗忘或能力退化也当作重新练习的理由；正向 LP 则不这样做。
- 技能可辨认就一定对任务有用吗？不一定，必须验证其外部效用和复用接口。
- 有恢复概率估计是否意味着安全？不意味着，模型可能失准或遇到分布外状态。

动手题：给标量 RND 增加两组状态 A、B，长时间只训练 B，再返回 A，检查预测器是否因共享参数发生干扰；比较高误差来自真正未见状态，还是来自遗忘。然后以相同训练预算比较“选最大误差”与“选最大误差下降”的行为。

## 本章的实验设计

探索阶段的交互也属于总成本。区分访问覆盖、新预测学得速度与实际奖励。

设定：构造可学习的稀疏反馈区域和不可预测噪声区域，保持外部任务奖励不变。比较探索信号与真实行为结果。

- 高预测误差不被直接解释成高学习进步。
- 内在奖励和外在奖励分别记录。
- 探索与课程生成交互计入总预算。

对照：随机探索与固定探索强基线；噪声关闭/开启的受控条件；固定数据检查信号，再做闭环行为

记录：访问覆盖、目标发现时间；预测误差下降或学习进步；完整任务收益、恢复与探索成本

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-control)

## 学习与研究衔接

经验获取决定之后能学习什么。预测误差大可能只是噪声，未必代表学习进展或控制价值。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-exploration) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=exploration) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=exploration)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

Laplacian 描述行为图结构，奖励感知表示加入路径价值，METRA 学习有区别的行为，HIQL 利用离线目标轨迹，MaestroMotif 引入语言先验。它们承担不同的设计工作；生成技能、选择技能与组合技能需要分别评价。

- [Proper Laplacian Representation Learning](https://yingwen.io/zh/continual-rl/research/#recent-proper-laplacian-representations)
- [Reward-Aware Proto-Representations in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-reward-aware-proto-representations)
- [METRA: Scalable Unsupervised RL with Metric-Aware Abstraction](https://yingwen.io/zh/continual-rl/research/#recent-metra-skills)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)
- [Posterior Sampling for Continuing Environments](https://yingwen.io/zh/continual-rl/research/#recent-cpsrl-continuing-exploration)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

模型可提取性的理论说明某类能力需要什么知识，不指定唯一网络。Dreamer 研究潜在想象控制，STOMP 研究随机时长行为模型，DRAGO 研究旧模型知识保留。模型误差、查询策略和规划收益之间仍需实验连接。

- [Knowledge Retention in Continual Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-drago-model-retention)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [Posterior Sampling for Continuing Environments](https://yingwen.io/zh/continual-rl/research/#recent-cpsrl-continuing-exploration)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [Knowledge Retention in Continual Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-drago-model-retention)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [The Cell Must Go On: Agar.io for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-agarcl)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文提供定义和条件，架构讲座提出模块组织，算法论文检验特定机制。完整系统还要明确智能体、外部设计者和世界各自承担的工作；组件成立不自动意味着组合后的长期收益成立。

- [Plasticity as the Mirror of Empowerment](https://yingwen.io/zh/continual-rl/research/#recent-plasticity-mirror-empowerment)

### Proper Laplacian Representation Learning

Diego Gomez, Michael Bowling, Marlos C. Machado

ICLR 2024 · 2024 · 支持方法与理论

#### 研究问题

技能发现需要一组确定的谱方向，为什么仅学到低频子空间还不够？

#### 关键机制

图上的平滑性目标倾向保留缓慢变化的特征，但旋转后的同一子空间未必给出可解释、排序明确的单个特征向量。ALLO 使用增广 Lagrangian、正交条件与对称性破除，同时恢复特征向量和特征值，从而为 eigenoption 的方向构造提供更明确的输入。

#### 证据

论文分析优化目标，并在多个环境中检验谱表示的恢复质量和下游使用。作者仓库包含表示学习训练程序。

#### 条件与限制

谱结构依赖采样行为诱导的图和覆盖程度，不是脱离数据分布的环境真值。低频方向也不自动等于有奖励价值的技能；这正是奖励感知表示要继续处理的问题。

#### 阅读与实验

先在小图上直接求特征分解，再比较学习特征的子空间误差和逐向量误差。两种指标不等价，后者才揭示任意旋转问题。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2310.10833)：ICLR 2024 论文的公开版本。
- [ALLO 作者代码](https://github.com/tarod13/laplacian_dual_dynamics)：增广 Lagrangian 的实际优化与实验入口。

#### 作者代码

[论文作者的 ALLO 实现。](https://github.com/tarod13/laplacian_dual_dynamics)

Laplacian 表示学习和论文实验。

### Reward-Aware Proto-Representations in Reinforcement Learning

Hon Tik Tse, Siddarth Chandrasekar, Marlos C. Machado

NeurIPS 2025 · 2025 · 支持方法与理论

#### 研究问题

仅编码可达关系的表示，怎样进一步反映奖励与行动成本？

#### 关键机制

论文研究 default representation，将奖励或成本纳入对未来状态关系的表示，并给出动态规划与 TD 学习方法。由此提取的谱特征可以参与技能发现、奖励塑形和迁移。它沿着 SR 的后果预测思路前进，但不再把奖励完全留到最后的线性读出阶段。

#### 证据

作者提供表格问题中的推导，并用表示、技能和迁移实验展示奖励信息如何改变学得的结构。代码包含 SR、DR 的计算和在线表示学习实验。

#### 条件与限制

把奖励纳入表示会改变迁移边界：奖励或内部成本变化后，原表示可能需要重学。论文结果不能解释为任意新奖励下都能免费零样本迁移。

#### 阅读与实验

固定转移图，只改变一处通行成本，比较 SR 与 DR 的谱方向。随后检查新的 eigenoption 是改变了可达性，还是改变了对路径代价的偏好。

#### 原文与相关入口

- [论文与版本记录](https://arxiv.org/abs/2505.16217)：NeurIPS 2025；后续版本修订不改变会议年份。
- [作者实现](https://github.com/httse9/Reward-Aware-Proto-Representations)：从 minigrid_basics/examples 的表示计算与技能实验开始。

#### 作者代码

[原论文作者仓库。](https://github.com/httse9/Reward-Aware-Proto-Representations)

奖励感知表示、谱特征与相关 MiniGrid 实验。

### METRA: Scalable Unsupervised RL with Metric-Aware Abstraction

Seohong Park, Oleh Rybkin, Sergey Levine

ICLR 2024 · 2024 · 支持方法与理论

#### 研究问题

没有外部任务奖励时，怎样发现能产生长距离、有区别状态变化的技能？

#### 关键机制

METRA 学习反映时间距离的潜在表示，并让技能方向 $z$ 最大化内在奖励 $r_z=(\phi(s')-\phi(s))^\top z$。邻接状态间的距离约束阻止编码器靠任意放大数值提高奖励。表示学习和技能策略相互影响，因此它不同于先固定一个表示、再单独训练 option。

#### 证据

论文在视觉与状态输入的运动、操纵任务中研究无监督技能学习和下游使用。作者代码包括约束优化、技能策略和相应实验配置。

#### 条件与限制

预训练技能加下游任务不等于技能库在单次生命内持续维护。理论距离约束与源码中的均方尺度、松弛量截断需要分别对照，不能只照抄一个简化公式重现。

#### 阅读与实验

观察表示范数、约束残差和实际位移三条曲线。若内在回报上升而位移不变，应先检查尺度和约束，而不是直接解释为探索改善。

#### 原文与相关入口

- [ICLR 原文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/516593a423838642a2eb4e9c5b9c7f44-Abstract-Conference.html)：方法与技能评价。
- [作者代码](https://github.com/seohongpark/METRA)：核心方法在 iod/metra.py；同时检查约束的归一化与截断。

#### 作者代码

[作者提供的论文实现。](https://github.com/seohongpark/METRA)

METRA、技能训练与下游评价。

### Knowledge Retention in Continual Model-Based Reinforcement Learning

Haotian Fu, Yixiang Sun, Michael Littman, George Konidaris

ICML 2025 · 2025 · 直接研究持续学习

#### 研究问题

当前任务不再访问旧区域时，世界模型怎样避免忘记那些区域的动力学？

#### 关键机制

DRAGO 将生成式旧经验、旧模型知识和探索结合起来。模型学习不只跟随当前奖励驱动的数据分布，还尝试维持对曾经学过区域的预测能力。它关注知识保留发生在模型里，而不只是保存旧策略输出。

#### 证据

论文在 MiniGrid 和连续控制任务中检验模型保留与任务表现，并给出原作者实现。关键设定包括共享状态与动力学、变化的任务奖励。

#### 条件与限制

已知任务切换、旧模型和数据资源是协议的一部分。共享动力学下的保留不能直接外推到动力学本身任意变化，也不是禁止重放的严格流式算法。

#### 阅读与实验

分别测量旧区域模型误差、旧任务回报与新任务学习速度。若模型误差改善而控制没有改善，再检查规划是否真正查询了被保留的知识。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/fu25f.html)：任务设定、模型保留和探索机制。
- [作者代码](https://github.com/YixiangSun/drago)：原文链接的 DRAGO 实验实现。

#### 作者代码

[作者 Yixiang Sun 的仓库；不使用同名第三方项目。](https://github.com/YixiangSun/drago)

DRAGO 的模型、重放、探索与实验。

### Plasticity as the Mirror of Empowerment

David Abel, Michael Bowling, Andre Barreto, Will Dabney, Shi Dong, Steven Hansen, Anna Harutyunyan, Khimya Khetarpal, Clare Lyle, Razvan Pascanu, Georgios Piliouras, Doina Precup, Jonathan Richens, Mark Rowland, Tom Schaul, Satinder P. Singh

NeurIPS 2025 · 2025 · 定义与架构观点

#### 研究问题

环境改变智能体的能力，与智能体改变环境的能力，能否放在统一的信息论框架中？

#### 关键机制

论文用广义有向信息描述两个方向：环境对智能体的影响对应一种可塑性，智能体对环境的影响对应赋能。统一表达使二者的关系和权衡可以被形式化，而不仅用神经元休眠或短期奖励间接描述。

#### 证据

贡献主要是概念定义和理论关系，提供研究长期交互的新坐标。它没有把信息量指标直接等同于某个具体神经网络算法的长期回报。

#### 条件与限制

信息论可塑性与“新目标拟合速度”不是相同估计量，也不等于参数变化越大越好。有限数据下怎样稳健估计这些信息量，需要额外方法。

#### 阅读与实验

分别举出高环境影响但低奖励、高赋能但不学习的过程。说明为什么两类能力与任务成功都需要独立评价。

#### 原文与相关入口

- [NeurIPS 2025 原文](https://papers.nips.cc/paper_files/paper/2025/hash/f04957cc30544d62386f402e1da0b001-Abstract-Conference.html)：统一定义、理论关系与解释。
- [作者预印本](https://arxiv.org/abs/2505.10361)：便于检索定义和证明。

### The Cell Must Go On: Agar.io for Continual Reinforcement Learning

Mohamed A. Mohamed, Kateryna Nekhomiazh, Vedant Vyas, Marcos M. José, Andrew Patterson, Marlos C. Machado

arXiv 预印本 · 2025 · 评价与实验协议

#### 研究问题

如何在持续、动态的高维交互里，同时研究记忆、探索、信用分配与学习能力保持？

#### 关键机制

AgarCL 提供持续运行的游戏环境，并用分解的小任务暴露不同困难。完整环境把这些机制放回同一交互循环，小任务则便于定位失败原因。游戏中的复活事件与把整个世界和智能体都重新开始不是同一种重置。

#### 证据

论文提供环境、基线与可塑性方法比较；部分常见修复在其测试中改善有限。这说明保持可塑性并不能单独代替记忆、探索和长期信用分配。

#### 条件与限制

一个游戏不能代表全部真实持续问题。小任务与完整游戏的协议需要分别阅读；此处仅按可确认的预印本状态收录，不把投稿信息写成会议录用。

#### 阅读与实验

先在一个小任务中验证机制，再检验它在完整环境中的作用是否仍存在。将世界重置、角色复活、参数重置和数据清空分开记录。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2505.18347)：环境设计、分解任务与基线结果。
- [作者环境仓库](https://github.com/machado-research/AgarCL)：环境安装、接口和运行示例；算法基线与环境本体分开。

#### 作者代码

[Machado 研究团队的环境实现。](https://github.com/machado-research/AgarCL)

AgarCL 环境与示例，非所有算法结果的单一训练脚本。

### Reset-free Reinforcement Learning with World Models

Zhao Yang, Thomas M. Moerland, Mike Preuss, Aske Plaat, Edward S. Hu

TMLR 2025 · 2025 · 支持方法与理论

#### 研究问题

不能靠外部重置回到起点时，怎样兼顾探索新状态与持续获得对任务有用的经验？

#### 关键机制

MoReFree 在 goal-conditioned world-model 系统中交替练习评测目标、返回初始分布与探索目标；模型内的策略训练也偏向任务相关目标。返回行为通过真实动作实现，调度块结束不会将物理世界 reset。

#### 证据

作者在八个 reset-free 任务中与模型自由及模型式基线比较；公开环境、探索调度和 imagination training 实现。

#### 条件与限制

训练无 reset，但主要评价仍使用可重置的 episodic 测试。已给定初始与目标状态分布、世界模型和 replay 都是资源；这不是任意非平稳 CRL 或真实安全的完整保证。

#### 阅读与实验

把返回成本计入总步数，分别消融数据获取目标与模型内训练目标；检查外部 reward-free 是否仍依赖设计者提供目标示例。

#### 原文与相关入口

- [作者论文 v3](https://arxiv.org/html/2408.09807v3)：训练与评价协议、back-and-forth exploration 与目标分布。
- [TMLR 作者项目页](https://yangzhao-666.github.io/morefree/)：正式发表状态与作者代码链接。

#### 作者代码

[TMLR 作者项目页明确链接的官方实现。](https://github.com/yangzhao-666/MoReFree)

resetfree/env.py、goal_picker_wrapper.py、Dreamer/PEG 与目标条件实验。

### Posterior Sampling for Continuing Environments

Wanqiao Xu, Shi Dong, Benjamin Van Roy

RLC 2024 / RLJ · 2024 · 支持方法与理论

#### 研究问题

没有自然回合边界，后验采样探索应在什么时候更换整条行动假设？

#### 关键机制

CPSRL 以独立随机时钟重采样模型并规划，而不等待真实 reset 或逐状态计数翻倍。几何持续时间把策略试验的未折扣收益与相应折扣规划目标联系起来；改变的是探索承诺的时间尺度。

#### 证据

论文在有限平稳 MDP 条件下分析 Bayesian regret，得到含奖励平均时间 $\tau$ 的 $\widetilde O(\tau S\sqrt{AT})$ 量级，并给出模拟。

#### 条件与限制

定理依赖正确后验、规划与平均时间条件；深网 ensemble 只是一种近似，不直接继承表格界。重采样不重置世界；平稳后验也不会自动遗忘已过时的动力学。

#### 阅读与实验

比较每步换假设、几何时钟与固定时钟，控制同一模型学习预算；在漂移实验中另外定义后验遗忘，避免误用平稳遗憾保证。

#### 原文与相关入口

- [RLC 2024 原文](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_277.pdf)：随机重采样、折扣联系与 Bayesian regret 假设。
- [RLJ 论文记录](https://rlj.cs.umass.edu/2024/papers/Paper277.html)：作者、会议与理论结果。


<a id="chapter-code"></a>

## 下载与运行

内部奖励/进展/恢复接口的确定性检查；非 RND、课程或安全算法完整性能复现。

[下载 lifelong_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)

```sh
python lifelong_algorithms_lab.py exploration
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Bellemare et al. · Unifying Count-Based Exploration and Intrinsic Motivation](https://proceedings.neurips.cc/paper_files/paper/2016/hash/afda332245e2af431fb7b672a68b659d-Abstract.html)：从密度更新推导 pseudo-count；需满足学习正性并处理退化分母。

- [Burda et al. · Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)：冻结随机目标、预测器及内部/外部价值设置。

- [RND 作者代码](https://github.com/openai/random-network-distillation)：完整 Atari 训练系统；重点阅读 ppo_agent.py、policies、归一化与 predictor 更新。

- [Portelas et al. · Teacher algorithms for curriculum learning](https://proceedings.mlr.press/v100/portelas20a.html)：ALP-GMM 的连续环境参数调度，不等于无环境生成权限的 single-life 问题。

- [ALP-GMM 作者实现 · teachDeepRL](https://github.com/flowersteam/teachDeepRL)：包含 toy teacher 环境和参数化 BipedalWalker；从 toy_env 开始核对课程机制。

- [Eysenbach et al. · Diversity is All You Need](https://arxiv.org/abs/1802.06070)：技能可辨认性与最大熵策略，不应把多样性视为外部效用保证。

- [Eysenbach et al. · Leave no Trace](https://arxiv.org/abs/1711.06782)：将前向学习与恢复策略共同建模，理解自主学习中的 reset 成本。

- [Ishfaq et al. · Langevin Soft Actor-Critic](https://proceedings.iclr.cc/paper_files/paper/2025/file/2420e12b7af4ac1e411fa6000576ffbd-Paper-Conference.pdf)：ICLR 2025：近似 critic 参数采样、分布式回报与生成样本；需区分认识不确定性与回报噪声。

- [LSAC 作者实现](https://github.com/hmishfaq/LSAC)：从 lsac.py 及 configs/gym_exp.json 对应 critic、生成器与训练预算；需要论文指定的连续控制依赖。

- [作者论文 v3](https://arxiv.org/html/2408.09807v3)：训练与评价协议、back-and-forth exploration 与目标分布。

- [TMLR 作者项目页](https://yangzhao-666.github.io/morefree/)：正式发表状态与作者代码链接。

- [Reset-free Reinforcement Learning with World Models · 作者实现](https://github.com/yangzhao-666/MoReFree)：resetfree/env.py、goal_picker_wrapper.py、Dreamer/PEG 与目标条件实验。 TMLR 作者项目页明确链接的官方实现。

- [RLC 2024 原文](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_277.pdf)：随机重采样、折扣联系与 Bayesian regret 假设。

- [RLJ 论文记录](https://rlj.cs.umass.edu/2024/papers/Paper277.html)：作者、会议与理论结果。
