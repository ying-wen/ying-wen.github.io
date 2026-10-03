# 持续探索：新奇、不确定性、学习进展与恢复

外部奖励稀疏、世界持续改变时，怎样获得有用的新经验，而不是追逐永远无法学会的噪声？

## 本章内容

- 从计数不确定性和预测误差分别推导 count bonus 与 RND，明确 reward 在更新前还是更新后计算。
- 区分新奇、信息增益、学习进展、技能多样性和自动课程这些不同目标。
- 将探索放回无免费 reset 的世界：记录恢复成本、可达性与真实外部收益，而不是只累计 intrinsic reward。

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
  1. 用更新前 predictor 计算并保存 rint=||predictor(o′)−target(o′)||²
  1. 以 r_ext、r_int 及各自 discount 更新价值/策略目标
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

在模拟器中，跌进坑之后 reset() 即可；在现实中，卡住、断电或不可逆损坏会改变此后全部数据分布。探索可以维护一个恢复策略与恢复价值，估计从候选动作后能否回到安全或可继续学习的集合。若风险过大则切换到恢复动作，或停止冒险并请求外部干预。

$$
\mathcal A_{\rm admissible}(s)=\{a:\widehat P(\text{恢复成功}\mid s,a)\geq 1-\varepsilon\}
$$

这只是一个简化的决策接口。用学习估计 P̂ 过滤并不自动给真实安全保证；需要校准、保守不确定性界、分布外检测或外部安全机制。

“Leave no Trace”同时学习前向任务与 reset 行为，展示了恢复应成为学习系统的一部分。不同任务中的失败集与允许干预必须明确。研究报告至少记录恢复成功率、恢复耗时、不可逆失败和人工干预次数；不能只保留 forward policy 的成功回合。

<a id="lesson-example"></a>

## 8 · 手算：新奇奖励的衰减

把 RND 缩小为某个表格状态的一个标量输出：冻结 target=2，predictor 初值 0，半平方损失梯度步长 α=0.25。第一次看见，先得到 bonus=4，再更新预测为 0.5；第二次 bonus=(2−0.5)²=2.25，更新为 0.875；之后两次 bonus 分别为 1.265625、0.711914。

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
