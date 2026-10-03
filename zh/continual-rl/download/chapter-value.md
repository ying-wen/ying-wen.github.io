# 价值预测与时间差分学习

价值预测估计给定策略的期望回报。回报的定义确定预测对象，MC、TD 和多步方法则提供不同的估计方式。本章从回报递推式推导这些方法，并讨论资格迹怎样分配时间上的信用。

## 本章内容

- 从 return 逐行推到 Bellman 方程，辨认模型期望和经验采样。
- 独立实现 MC 与 TD，解释为什么同一条轨迹给出的第一次更新不同。
- 理解 $\lambda$ 在传播信用中做什么，以及为什么普通在线 TD($\lambda$) 与固定参数前向视图不能无条件画等号。

<a id="problem-definition"></a>

## 本章的问题定义

环境与目标策略固定，询问按该策略行动的期望回报；本章基础数据也由该策略产生。

### 给定条件与符号

- Markov状态、固定目标策略、奖励与延续/终止定义。
- 经验流、价值表示类及更新预算；已知模型是DP的额外权限。

### 需要求解的对象

指定策略的价值函数及在给定表示下的估计；策略本身不是本章待学对象。

### 信息与数据权限

在 $S_t$ 按 $\pi$ 选择 $A_t$，得到 $R_{t+1},S_{t+1}$；估计 $V_t$ 只能使用已经收到的经验。

$$
v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s],\qquad G_t=R_{t+1}+\gamma_{t+1}G_{t+1}
$$

$\pi$ 是给定策略，$G_t$ 是随机回报，$\gamma_{t+1}$ 是当前转移后的延续因子；真实终止为0。$V_t$ 是估计而不是答案本身；TD平方误差与真实价值误差不相等。

### 成立条件与解的含义

- 基础设定为有限、固定MDP、有界奖励、非终止处固定折扣小于1；无折扣终止问题需另给可积终止条件。
- 表格收敛还需访问与步长条件；共享非线性表示、离策略及不断漂移不自动继承该结论。

判断准则：两步链上预测趋近起点0.9、后继1；一般任务以独立回报或解析解测价值误差，训练样本TD误差无需逐条为零。

### 适用边界

- 不以对动作取最大值替代给定策略的动作平均。
- 不以训练TD损失下降宣称策略收益改善。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)：本章将累计信号限定为任务奖励，并采用普通折扣/终止规则，因此是更一般GVF预测规格的特例。

- 组合不同学习问题 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：预测可以评价一个候选策略；控制另需策略改善和行为数据更新。

- 组合不同学习问题 · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)：多步和资格迹决定同一预测问题的反馈如何作用于过去，不改变给定策略的题目。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

完整回报尚未到来；一步自举可立即更新，却依赖当前后续价值估计。

### 本章的核心思路

从同一回报递推式选择模型期望、完整样本或一步自举；比较的是估计方法而非三个任务。

1. [从回报推到条件期望](#lesson-derive)：因为预测对象是指定行为的未来，Bellman方程按目标策略平均动作而不取最大值。

2. [选择等待长度与信用路径](#value-traces)：因为短目标依赖估计、长目标等待更多数据，多步混合与资格迹在同一对象下改变传播。

3. [以解析值检查完整循环](#lesson-code)：因为一次正确梯度不保证整个时序正确，真实终止尾值、旧参数和访问更新一并用两步链验证。

结论与条件：正确有限折扣模型的Bellman算子收缩；表格MC/TD还需各自采样与步长条件。普通在线迹不与冻结前向视图无条件精确等价。

### 相关方法改变了什么

- DP：已知后果模型时计算条件期望，需要模型权限。

- MC：用完整回报采样，减少自举依赖但等待结果。

- TD与多步：用后继估计补足未来，改变目标偏差、方差与反馈延迟。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 一条经验

在 $S_{t}$ 做 $A_{t}$ 后得到 $R_{t+1}$、$S_{t+1}$。奖励属于这次转移，不是到达状态之前另一轮的奖励。

### 折扣与终止

$\gamma$ 控制未来相对权重；真实终止后的价值设为 0。本章记 $\gamma_{t+1}$=$\gamma$(1−terminated)。训练脚本的时间截断不一定是任务终止。

### 表格与函数逼近

表格每个状态或状态动作一组独立参数；函数逼近让不同输入共享参数。更新一个输入可能改变其他输入的预测。

<a id="lesson-setting"></a>

## 1 · 给定策略的预测问题

本章固定策略 $\pi(a\mid s)$，估计按它持续行动的后果，不改变动作选择规则。假设状态与动作有限、转移规律固定、奖励有界且 $0\le\gamma<1$。数据由目标策略产生。若当前观测不足以条件化未来，需要先引入历史或适当的状态表示；不能直接将观测当作 Markov 状态。

价值预测与控制使用相同的回报定义，但求解的对象不同。预测以策略为输入，以价值为答案；控制根据价值或其他后果估计改善策略。先独立理解预测，可以分清一个控制算法的问题来自评价误差、决策规则，还是收集到的数据。通用价值函数则进一步扩展预测的信号和延续规则，仍不自动承担策略改善。

$$
G_t=R_{t+1}+\gamma_{t+1}G_{t+1},\qquad v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s]
$$

$G_t$ 是一次实现的随机回报；$v_\pi(s)$ 是同一起点、同一策略下回报的条件期望。真实终止令后继延续因子为零，只删掉未来项，不删掉进入终点时的奖励。

算法维护估计 $V$；真实价值通常未知。解析小 MDP 可以直接计算价值误差；一般任务可用独立采样回报进行评估。训练中的 TD error 与价值估计误差是不同的量。

<a id="lesson-derive"></a>

## 2 · 从回报拆分到 Bellman，再到三种更新

$$
v_\pi(s)=\sum_a\pi(a\mid s)\sum_{s\prime,r}p(s\prime,r\mid s,a)[r+\gamma(s,a,s\prime)v_\pi(s\prime)]
$$

第一步把 G 拆成当前奖励和未来；第二步对动作与转移取条件期望；Markov 性允许用下一状态而非完整历史描述后半段。这里尚未提出任何学习算法。

$$
V_{k+1}(s)=(T_\pi V_k)(s),\qquad \|T_\pi u-T_\pi v\|_\infty\le\gamma\|u-v\|_\infty
$$

已知模型时，可以对所有后果求和。这是动态规划的期望备份；$\gamma$<1 带来压缩性。在正确表格模型下重复备份趋向唯一固定点。$\gamma$=1 的终止任务需要另一套适当终止条件。

$$
V(S_t)\leftarrow V(S_t)+\alpha[G_t-V(S_t)]
$$

MC 用完整实际结果作为监督目标。需等到回报可计算；对固定策略、可积回报，它直接对正确条件期望采样。环境中途不断变化、轨迹没有结束或非常长时，等待完整回报代价很大。

$$
\delta_t=R_{t+1}+\gamma_{t+1}V_t(S_{t+1})-V_t(S_t),\qquad V_{t+1}(S_t)=V_t(S_t)+\alpha_t\delta_t
$$

TD(0) 只等一步，把剩下的未来交给当前估计。单个 target 通常不是对真实 v 的无偏样本，因为 V 还不准；但正确条件下长期固定点可以正确。因此，需要分析其长期固定点，而不只分析单次目标的偏差。

| 方法 | 需要什么 | 一条转移能否立即更新 | 误差从哪里来 |
| --- | --- | --- | --- |
| DP | 转移与奖励模型 | 可以，但使用模型期望 | 模型误差、备份不足 |
| MC | 完整采样回报 | 通常不能 | 有限样本方差 |
| TD | 一步经验与旧价值 | 可以 | 自举估计、有限样本、表示限制 |

<a id="value-traces"></a>

## 3 · 多步与 $\lambda$：奖励应该传回多远？

$$
G_t^{(n)}=\sum_{k=1}^n\gamma^{k-1}R_{t+k}+\gamma^n V(S_{t+n})
$$

终止时截到终点。n 小更依赖价值估计，n 大使用更多实际结果，也等待更久、通常有更大方差。具体的误差权衡取决于奖励噪声、价值估计和轨迹长度。

$$
G_t^\lambda=(1-\lambda)\sum_{n\ge1}\lambda^{n-1}G_t^{(n)},\qquad G_t^\lambda-V(S_t)=\sum_{k\ge0}(\gamma\lambda)^k\delta_{t+k}
$$

在无限折扣轨迹、固定 V 下，展开 n-step，按相同奖励与 V 项收集，价值项逐项抵消，就得到右边的 TD error 加权和。有限终止轨迹的最后一项吸收剩余权重。

$$
e_t=\gamma\lambda e_{t-1}+x_t,\qquad w_{t+1}=w_t+\alpha\delta_t e_t
$$

线性 $V=w^Tx$ 时，把未来误差回传改写为记录过去特征。e 是“哪些参数对近期预测有影响”的信用痕迹，不是能在动作选择时回忆往事的隐状态。表格取 one-hot 特征。

这一步前向/后向等价的推导固定了参数。普通在线 TD($\lambda$) 每步都改参数，有限步长下不能声称严格等价于该固定参数目标；true-online TD($\lambda$) 用 Dutch trace 和额外修正实现相应在线前向视图。离策略时还要处理目标策略与行为策略的差别，相应修正将在通用价值函数一章中推导。

<a id="lesson-example"></a>

## 4 · 两步轨迹：为什么 MC 比 TD 更早改动起点？

A --奖励 0→ B --奖励 1→终点。$\gamma$=.9，初始 V(A)=V(B)=0，$\alpha$=.1。真实值为 $v_\pi(A)=0.9$、$v_\pi(B)=1$。

| 第一次经历 | A 的 target / 更新后值 | B 的 target / 更新后值 |
| --- | --- | --- |
| MC | 完整回报 .9 → .09 | 完整回报 1 → .1 |
| TD(0) | 0+.9×0=0 → 0 | 1+0−0 → .1 |
| TD($\lambda$=.8) | 第二步迹 .9×.8=.72 → .072 | 第二步当前特征迹 1 → .1 |

第二回合 TD 在 A 的 target 已是 .9×.1=.09，因此 V(A)=.009。延迟奖励通过多轮自举逐渐传播到起点。因此，MC 与 TD 即使具有相同的极限值，有限数据下的学习过程也不同。

<a id="lesson-code"></a>

## 5 · 核心实现与一次完整训练循环

**算法：线性、on-policy、回合制的 accumulating TD(λ)**

1. 初始化 $w$；给定步长 $\alpha$、迹参数 $\lambda$ 和特征函数 $x(s)$。
1. 每个回合开始时令 $e=0$，观察起始状态 $S$。
1. 每次转移：
  1. 按目标策略执行动作，观察 $R,S'$。
  1. 若真实终止，令 $\gamma'=0$；否则令 $\gamma'=\gamma$。
  1. $\delta\leftarrow R+\gamma'w^\top x(S')-w^\top x(S)$
  1. $e\leftarrow\gamma\lambda e+x(S)$
  1. $w\leftarrow w+\alpha\delta e$
  1. $S\leftarrow S'$；真实终止时结束本回合。

相同环境与步长，分别用整段结果和一步自举

```python
def prediction(method="td", episodes=200, alpha=0.1):
    """A --0--> B --1--> terminal; gamma=.9, exact values [.9, 1]."""
    values = [0.0, 0.0, 0.0]
    trajectory = [(0, 0.0, 1, 0.9), (1, 1.0, 2, 0.0)]
    for _ in range(episodes):
        if method == "mc":
            ret = 0.0
            for state, reward, _, discount in reversed(trajectory):
                ret = reward + discount * ret
                values[state] += alpha * (ret - values[state])
        elif method == "td":
            for state, reward, nxt, discount in trajectory:
                delta = reward + discount * values[nxt] - values[state]
                values[state] += alpha * delta
        else:
            raise ValueError("Choose mc or td")
    return values[:2]
```

- MC 从终点向前计算 return；本例每个状态每回合仅访问一次，因此 first-visit 与 every-visit 没有差别。一般轨迹中须明确选择。
- TD 在收到下一状态后立刻更新，后继值使用更新前可用的估计。终点值为 0。
- 运行 value 后两者应趋近 [.9, 1]。将 `episodes` 设为 1，可以比较首次更新与本节的数值计算。

<a id="lesson-branches"></a>

## 6 · 向持续预测的推广

| 分支 | 实际改变 | 需要继续检查 |
| --- | --- | --- |
| 常数步长跟踪 | 让新数据持续改变估计 | 方差、变化速度、访问频率 |
| GVF | 把任务奖励改为任意累积信号，并明确策略与延续 | 预测题目是否定义正确 |
| GTD / emphatic TD | 改变离策略函数逼近的更新几何或加权 | 覆盖、方差、线性理论条件 |
| 平均奖励 TD | 去掉长期增长的奖励率，学习差分价值 | 奖励率与价值的联合估计 |
| 神经网络 TD | 用共享非线性表示 | 自举、离策略、函数逼近的耦合 |

常数步长不追求在静止问题里把噪声彻底平均掉，而是保留对新规律的响应。有效样本权重随年龄呈指数衰减；真正的变化检测、滑动窗口选择和历史情境复用则是进一步的问题，不是 TD 自动具备的功能。

<a id="lesson-check"></a>

## 7 · 习题与讨论

问题：学习 TD 时把 target 中的下一状态价值也求导，会更“完整”吗？不一定。它改变了算法：TD 的半梯度更新是固定 target 后的回归方向；完整样本残差梯度优化另一目标。对期望 Bellman 残差平方求无偏梯度还涉及双采样。先写目标，再判断哪个梯度正确。

问题：为什么训练 TD error 不为零仍可能已经学对？随机奖励和随机转移会带来不可约的单样本误差；正确价值满足条件期望误差为零，不要求每条样本误差都零。用已知解析值或独立 rollout 估值，不能只看训练 loss。

## 本章的实验设计

在小型马尔可夫奖励过程中计算解析价值。再检查样本更新和参数误差。TD 误差不能代替价值误差。

设定：两状态奖励过程：A 无奖励到 B，B 获得 1 后回 A，γ=0.5。固定策略的价值为 v(A)=2/3、v(B)=4/3。

- 解析解满足两个 Bellman 方程。
- 零步长不写入，γ=0 只拟合下一奖励。
- 终端样本不读取无效后继值。

对照：MC、TD(0) 与指定版本 TD(λ)；相同轨迹与独立交互两种面板；固定表征及共同步长搜索预算

记录：对解析价值的均方误差；Bellman 残差与样本 TD 误差分别记录；固定数据量下的误差曲线

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-classic)

## 学习与研究衔接

表格 TD 的局部更新推广为共享特征上的参数更新。GVF 再改变预测问题，而非只改变网络。

[分册导读](https://yingwen.io/zh/continual-rl/start/classic-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-value) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=value) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=value)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

状态是支持后续计算的内部信息；GVF 指定一个预测问题；RTRL 和资格迹规定信用如何传播。三者可以组合，但不是相互替代的算法名称。先理解给定策略的预测，再讨论预测怎样改善控制。

- [When does Self-Prediction help? Understanding Auxiliary Tasks in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-self-prediction-auxiliary-tasks)

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)
- [Safe and Efficient Off-Policy Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-retrace-safe-offpolicy)
- [Convergent Tree Backup and Retrace with Function Approximation](https://yingwen.io/zh/continual-rl/research/#recent-convergent-tree-retrace)
- [Multi-Step Reinforcement Learning: A Unifying Algorithm](https://yingwen.io/zh/continual-rl/research/#recent-q-sigma-backups)
- [A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning](https://yingwen.io/zh/continual-rl/research/#recent-lambda-greedy)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [Reward Centering](https://yingwen.io/zh/continual-rl/research/#recent-reward-centering-discounted)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning](https://yingwen.io/zh/continual-rl/research/#recent-lambda-greedy)

### When does Self-Prediction help? Understanding Auxiliary Tasks in Reinforcement Learning

Claas A. Voelcker, Tyler Kastner, Igor Gilitschenski, Amir-massoud Farahmand

RLC 2024 / RLJ · 2024 · 支持方法与理论

#### 研究问题

预测下一潜在状态、重建观测和学习价值，为什么会产生不同的表示？

#### 关键机制

论文在含干扰因素的线性问题中分析辅助目标的学习动力学。潜在状态自预测与价值学习共同作用时可能保留决策相关结构，但单独训练同一目标未必得到最有用的特征。目标的作用取决于它和 TD 目标怎样共享表示。

#### 证据

线性分析给出可检查的条件，并用神经网络实验检验部分预测。结果不支持“任何自监督预测都能改善 RL”这种无条件判断。

#### 条件与限制

线性分析中的观测映射、优化过程与神经网络控制并不完全等价。项目仓库入口不等于已经提供完整可复现实验实现，因此这里不列为可运行代码。

#### 阅读与实验

固定编码器容量，分别比较仅 TD、仅辅助任务和联合训练。记录价值误差与任务收益，不要只用辅助损失下降评价表示。

#### 原文与相关入口

- [RLC 2024 论文入口](https://rlj.cs.umass.edu/2024/papers/Paper197.html)：原文、线性假设与神经网络实验。
- [作者预印本](https://arxiv.org/abs/2406.17718)：便于追踪论文版本。

### Deep Reinforcement Learning with Gradient Eligibility Traces

Esraa Elelimy, Brett Daley, Andrew Patterson, Marlos C. Machado, Adam White, Martha White

RLC 2025 / RLJ · 2025 · 支持方法与理论

#### 研究问题

资格迹怎样与明确的梯度目标结合，而不是直接把线性半梯度规则搬到深度网络？

#### 关键机制

论文从广义投影 Bellman 误差出发构造多步目标，推导带资格迹的梯度学习方法。前向视角连接多步回报与经验重放，后向视角通过递推迹分配信用。目标函数、辅助估计器和迹的更新共同决定算法，不只是选择一个较大的 λ。

#### 证据

作者给出多种算法并在 MuJoCo、MinAtar 等任务中比较。代码同时提供相关梯度算法与实验设置，可以把推导中的量映射到实际更新。

#### 条件与限制

线性 GTD 的收敛条件不能自动赋予非线性实现全局收敛保证。重放版本与流式版本的数据使用预算也不能混为一谈。

#### 阅读与实验

从一段短轨迹分别计算前向多步目标和后向迹。随后对照原代码检查辅助网络、目标与主网络参数使用的是更新前还是更新后的值。

#### 原文与相关入口

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_302.pdf)：目标、算法推导与实验。
- [作者算法库](https://github.com/esraaelelimy/gtd_algos)：论文提供的梯度 TD 与资格迹实现。

#### 作者代码

[原论文链接的作者仓库。](https://github.com/esraaelelimy/gtd_algos)

论文梯度算法、资格迹和实验配置。

### Reward Centering

Abhishek Naik, Yi Wan, Manan Tomar, Richard S. Sutton

RLC 2024 / RLJ · 2024 · 支持方法与理论

#### 研究问题

接近一的折扣为何使共同价值偏移很大，中心化能改善什么、又不能改变什么？

#### 关键机制

从折扣价值的共同偏移与相对价值分解出发，移除奖励参照量；on-policy 可估计行为奖励均值，off-policy 提出 TD 驱动的参照更新。保留小于一的折扣时，中心化没有消除折扣对策略排序的影响。

#### 证据

原文给出理论动机与表格、线性、非线性控制实验，检验折扣及奖励常数平移。深度 continuing-task 后续研究扩大了算法与环境范围。

#### 条件与限制

TD 中心化中的标量在有限折扣下不必精确等于真实奖励率。训练期的联合参照/价值更新与固定常数下的平移恒等式需分别分析；真实终止改变平移条件。

#### 阅读与实验

用单状态常奖励问题解出联合更新固定点，再用多动作问题检查策略排序；同时记录参照量与直接观测的外部奖励率。

#### 原文与相关入口

- [RLC 2024 原文](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_261.pdf)：中心化分解、on/off-policy 区别及收敛讨论。
- [RLJ 论文记录](https://rlj.cs.umass.edu/2024/papers/Paper261.html)：正式题名、作者与会议年份。

### Safe and Efficient Off-Policy Reinforcement Learning

Rémi Munos, Tom Stepleton, Anna Harutyunyan, Marc G. Bellemare

NeurIPS 2016 · 2016 · 支持方法与理论

#### 研究问题

目标与行为策略不一致时，如何保留多步信用而避免重要性比率乘积爆炸？

#### 关键机制

统一多步目标为目标策略TD误差的加权和，Retrace采用λmin(1,π/μ)传播系数。近同策略时保留长迹，目标概率较低的动作则减少传播；一步误差仍使用目标动作期望。

#### 证据

论文分析表格算子的收缩性质，给出条件下的评价与控制收敛，并报告Atari实验。信用章独立检查传播系数和有限轨迹恒等式。

#### 条件与限制

表格安全性不是任意线性或神经逼近的稳定性保证。行为覆盖、变化策略与投影条件仍需检查；代码小实验不复现Atari。

#### 阅读与实验

在同样轨迹与表示上，分别改变策略差异和动作随机性，比较Tree-backup与Retrace的信用长度、方差和预测误差。

#### 原文与相关入口

- [原论文](https://arxiv.org/html/1606.02647)：统一算子、传播系数及理论条件。

### Convergent Tree Backup and Retrace with Function Approximation

Ahmed Touati, Pierre-Luc Bacon, Doina Precup, Pascal Vincent

ICML 2018 · 2018 · 支持方法与理论

#### 研究问题

传播系数已经截断，为什么函数逼近下的Tree-backup和Retrace仍可能发散？

#### 关键机制

分析函数逼近与off-policy多步bootstrap的学习算子，展示线性反例，再把相应目标写成二次凸凹鞍点问题，构造梯度版本。

#### 证据

原文给出线性不稳定例子、梯度方法收敛保证与有限样本界。它直接限定了从Retrace表格结论外推到逼近算法的范围。

#### 条件与限制

凸凹线性问题的保证不能自动覆盖学习表示的深度网络。稳定目标、更新速度与控制性能还需分别验证。

#### 阅读与实验

先检查固定表示下的期望更新矩阵，再将半梯度和梯度版本按相同样本、步数与计算预算比较。

#### 原文与相关入口

- [ICML原文](https://proceedings.mlr.press/v80/touati18a.html)：理论反例、鞍点方法和保证条件。

### Multi-Step Reinforcement Learning: A Unifying Algorithm

Kristopher De Asis, J. Fernando Hernandez-Garcia, G. Zacharias Holland, Richard S. Sutton

AAAI 2018 · 2018 · 支持方法与理论

#### 研究问题

多步动作价值目标必须始终采样下一动作，或始终对动作取期望吗？

#### 关键机制

Q(σ)逐处混合Sarsa的采样动作与Expected Sarsa的动作期望，并同步改变后续误差传播。σ控制采样程度，与控制回报长度的λ不同。

#### 证据

原文给出统一n-step表达、off-policy修正和实验比较。信用章小程序核验冻结on-policy几何λ混合的两个端点。

#### 条件与限制

原文n-step和本章λ混合参考具有不同实现范围。只改一步误差却不改多步传播或策略修正，不能称为完整Q(σ)。

#### 阅读与实验

把采样噪声、目标长度和策略差异分开改变，避免把σ与λ的作用归到同一“更长信用”解释。

#### 原文与相关入口

- [原文](https://arxiv.org/html/1703.01327)：式13–15：混合误差、传播及off-policy修正。

### A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning

Martha White, Adam White

arXiv预印本 · 2016 · 支持方法与理论

#### 研究问题

不同状态的预测可靠性不同，固定λ是否浪费了多步信用？

#### 关键机制

将下一处bootstrap选择写成局部偏差平方与回报方差的折中，得到$λ=b^2/(b^2+\operatorname{Var}(G))$。完整λ-greedy还用在线预测器估计回报均值和二阶矩。

#### 证据

原文给出状态相关λ的目标、增量算法和多个预测设置的实验。信用章仅核对已知统计量下的局部最优与变量λ恒等式。

#### 条件与限制

局部贪心目标不是整条轨迹的联合最优。逼近误差、统计滞后和非平稳性会影响λ估计；辅助资源需要计入比较。

#### 阅读与实验

先让噪声方差变化，再让bootstrap可靠性变化。比较固定λ、已知统计参照和在线估计，分别观察目标偏差与适应速度。

#### 原文与相关入口

- [作者原文](https://arxiv.org/html/1607.00446)：局部目标、状态λ、均值／二阶矩预测与完整算法。


<a id="chapter-code"></a>

## 下载与运行

Python 3.10+，仅标准库。实现表格 MC/TD 完整小实验；资格迹的前后向推导与实现见时间信用分配章。

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

```sh
python3 foundations_detail_lab.py value
python3 foundations_detail_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：第 3–8 章建立 MDP、动态规划、MC、TD 与规划；第 9–13 章把这些更新扩展到函数逼近、资格迹和策略梯度。按本章的问题找相应章节，不必从头重读。

- [Sutton · Learning to Predict by the Methods of Temporal Differences](https://doi.org/10.1007/BF00115009)：TD 的原始问题动机与多步预测。读“如何利用尚未结束的经验”，并理解其更新规则。

- [van Seijen & Sutton · True Online TD($\lambda$)](https://proceedings.mlr.press/v32/seijen14.html)：检查在线前向视图、Dutch trace 与修正项；它解决的不是简单加大 $\lambda$。
