# 探索与不确定性：后验、乐观估计和时间一致行动

现代深度强化学习 · 并列研究分支

为什么每一步都随机，并不等于有效获取长期有用的信息？

## 本章内容

- 区分环境随机性与知识不足。
- 推导 Bernoulli 后验与 UCB 的置信思路。
- 理解 posterior sampling、Bootstrapped DQN 与内在奖励各自改变什么。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [多臂老虎机：估计、探索与直接策略学习](https://yingwen.io/zh/continual-rl/foundations/tabular/bandits/)：区分价值估计、乐观选择与直接策略更新。
- [深度价值学习：DQN、Double DQN 与目标的时间顺序](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/)：理解价值网络的数据来源和自举目标。


### 条件概率与期望

区分随机变量、一次样本与条件分布；可以使用 Bayes 法则和全期望公式。

### MDP 与价值

理解状态、动作、转移、奖励、策略和 Bellman 递推；学习数据可以来自不同策略。

### 优化与梯度

知道固定目标、参数梯度、约束和期望目标的含义；神经网络只是函数表示的一种选择。

<a id="problem-definition"></a>

## 本章的问题定义

环境未知，动作既产生收益也改变以后可获得的信息。选择行为不能只最大化当前估计。

### 给定条件与符号

- 未知 MDP 族 $\mathcal M$、先验或置信集合、交互预算 $T$。
- $\mathcal L$ 表示因果学习规则，$J_T(\mathcal L,M)$ 为其在环境 M 中的期望累计奖励。

### 需要求解的对象

选择探索规则，使有限预算内的完整学习表现较好，而非只让状态访问熵最大。

### 信息与数据权限

只观察实际轨迹，不能免费查询未执行动作的反事实后果。

$$
\mathbb E_{M\sim P_0}\!\left[J_T^*(M)-J_T(\mathcal L,M)\right]
$$

这是给定先验下的一种 Bayesian regret；$J_T^*$ 指知道环境的同目标最优值。其他探索问题可选择不同比较器，不能混用其理论界。

### 成立条件与解的含义

- 后验采样的理论依赖模型族和后验正确性；乐观界依赖覆盖概率。
- 神经 ensemble 的分歧不自动等于校准后验不确定性。

判断准则：测量远端有用状态到达率、信息利用和原始奖励；把内在奖励与外部收益分开。

### 适用边界

- 保证任意新奇访问都能改善任务收益。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)：本章先固定未知 MDP 族与有限交互预算，研究识别和信息利用；更广的持续探索还要处理已有知识过时后的重新验证。

- 特例：增加条件 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：本章将完整学习器比较限定到未知 MDP 族、指定先验和有限预算的 Bayesian regret；相关控制章并不限于这一环境类与比较器。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

长程有用试验需要连续一致的动作，逐步随机可能反复取消自己的计划。

### 本章的核心思路

用后验假设或乐观模型支撑跨时间行动，并区分信息不确定性与环境噪声。

1. [先看可精确计算的后验](#lesson-notation)：小问题提供检验近似不确定性是否可信的参照。

2. [让探索假设维持足够长](#lesson-derive)：抽样整个模型或策略与每步随机动作具有不同到达概率。

3. [说明奖励奖励的是哪种未知](#lesson-bonus)：置信上界、新奇度和预测误差奖励不能相互冒充。

结论与条件：表格遗憾分析有特定假设；内在奖励或 ensemble 不直接继承相同保证。

### 相关方法改变了什么

- Posterior sampling / optimism：前者按可信模型抽样，后者在可信集合中选择有利模型。

- 新奇度 / epistemic uncertainty：访问少不等于信息有用，预测误差也可能来自不可减少的噪声。


<a id="lesson-setting"></a>

## 1 · 行动同时获得回报与数据

探索的困难在于当前行动影响未来知道什么。随机动作可能增加覆盖，也可能反复把智能体带离有价值的路径。设计者通常给定奖励、可访问状态、重置权限与探索预算；智能体需要在这些边界内分配数据采集。探索算法并不自动解决目标从何而来。

$$
\operatorname{Regret}(T)=T\mu^*-\mathbb E\sum_{t=1}^T R_t
$$

这是平稳 bandit 的期望遗憾定义。序列控制还涉及状态可达性、信息延迟和探索行动的不可逆后果，不能直接套用同一分析。

一枚已经知道成功率为 0.5 的硬币仍有结果随机性，但不一定值得继续研究。相反，成功率未知的硬币有认识不确定性，可以由数据减少。预测残差大、回报方差大和模型不确定性大不是同一件事。

<a id="lesson-notation"></a>

## 2 · 一个精确后验的起点

$$
R\mid p\sim\operatorname{Bernoulli}(p),\quad p\sim\operatorname{Beta}(\alpha_0,\beta_0)
$$

模型假设每次条件独立且成功率不变。Beta 先验是设计者选择的模型知识，不是观测结果。

$$
p\mid\mathcal D\sim\operatorname{Beta}(\alpha_0+n_1,\beta_0+n_0)
$$

Bernoulli 似然提供成功次数与失败次数的幂，乘上先验后仍是同一族；归一化得到后验。

$$
\begin{gathered}\mathbb E[p\mid\mathcal D]=\frac{\alpha}{\alpha+\beta}\\\operatorname{Var}(p\mid\mathcal D)=\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}\end{gathered}
$$

这是对未知成功率的方差，不是下一次二元结果的方差。固定后验均值时，增加证据量仍会降低参数不确定性。

<a id="lesson-derive"></a>

## 3 · 后验采样为何需要保持时间一致

$$
\tilde p_a\sim p(p_a\mid\mathcal D),\qquad A_t=\arg\max_a\tilde p_a
$$

Thompson sampling 对每个 bandit 的可能成功率采样，再按该可能世界作最优选择。

在已知先验模型族的 episodic MDP 中，PSRL 在 episode 开始采样一个 $\tilde M\sim p(M\mid\mathcal D)$，求解该模型的策略，并在本 episode 中保持它。完整模型的采样使多步行动围绕同一个假设保持一致；每一步重新抛独立动作硬币并不具有这项性质。

$$
\Pr(\text{长度 }H\text{ 的全右路径})=2^{-H}\quad\text{或}\quad\tfrac12
$$

左式是在每步公平地独立选左右；右式是 episode 开始公平选“始终右”或“始终左”两个完整策略。本比较只说明时间相关性，不是通用学习速度定理。

PSRL 的采样发生在模型上，求解则发生在这个采样模型里。采样模型会相信某条路线值得走，策略就应坚持到环境能检验这项假设的地方；沿途可以根据位置改变动作。它按采样模型优化本轮外部回报，并没有求解包含未来后验变化的 Bayes-adaptive 最优控制。下面让这两种对象在同一个走廊中分别可计算。

<a id="lesson-bonus"></a>

## 4 · 乐观估计与新奇奖励的不同角色

$$
U_t(a)=\hat\mu_t(a)+\sqrt{\frac{2\log t}{N_t(a)}}
$$

UCB1 形式以样本均值加置信宽度排序；未访问动作优先探索。其经典保证依赖平稳、有界且适当独立的 bandit 观测。

宽度随样本数减小，使已确定的低价值动作逐步退出。Hoeffding 型尾界控制经验均值偏差，由高概率上界产生乐观行动。序列 MDP 的 bonus 还需处理转移估计、规划误差与访问次数，不能仅把 bandit 式原样贴在神经 Q 输出上就获得 regret 保证。

$$
\Pr(\mu_a-\hat\mu_{a,n}\ge b)\le e^{-2nb^2},\qquad b=\sqrt{\frac{2\log t}{n}}\ \Longrightarrow\ e^{-2nb^2}=t^{-4}
$$

对 [0,1] 内独立同分布奖励，先固定某动作的采样次数 n，再用这个单侧界。行动使 $N_t$ 随机时，还需对可能的 n 做并集控制；不能把固定 n 的概率式直接当成自适应采样的无条件式。常数二由这里选用的尾概率给出，不是任意奖励尺度都通用。

$$
Q_k(s,a)=\hat r(s,a)+b_k(s,a)+\gamma\sum_{s'}\hat P(s'\mid s,a)\max_{a'}Q_{k-1}(s',a')
$$

示意性乐观模型备份：bonus 进入规划，沿未来状态传播。具体置信项必须由所用假设推导。

$$
r_t^{\rm int}=\|f_\theta(O_{t+1})-f_{\rm fixed}(O_{t+1})\|^2
$$

RND 的固定随机目标与学习预测器产生新奇信号。它不直接估计环境奖励或转移模型的 Bayes 方差。

RND 对确定随机映射做预测，避免将随机下一状态预测误差直接当作新奇的某些问题，但无关高维变化、表征泛化和遗忘仍能影响 bonus。熵正则则直接偏好较随机的动作分布；它不会自动区分有信息的状态与无信息的随机循环。

<a id="experiment-count_bonus"></a>

### 实验：实验 · 新奇奖励可以暂时偏离原任务

简单链已能用 ε-greedy 探索时，额外访问奖励是否仍会更快找到原任务的最优行为？

**环境与可用信息。** 六格链从状态 0 出发，左右移动，到第 5 格终止并奖励 1，其余转移奖励 −0.01。环境平稳，折扣 0.95。五个非终止状态各有两个 Q 值，初值全零。

**设置。** 五种子各 1200 个真实转移。Q-learning 步长 0.2，ε=0.1。每次访问先将状态动作计数加 1，再给训练目标加入 0.2/sqrt(N)；无 bonus 对照其余规则相同。终点之后重置到 0。

**检验的机制。** 少访问的状态动作会得到较大正奖励，策略可能因此停留或绕行。随着计数增加，人工奖励衰减。它改变训练目标，但评价始终使用原环境奖励。

**测量。** 纵轴是冻结贪心策略从 0 出发的原奖励折扣回报。代码对确定性策略中的循环求精确无限和；没有把卡住的轨迹按某个有利截止删掉。最短路径的值约为 0.7774。

```bash
python3 implementations/continual/count_bonus.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/count_bonus/curves.svg)

横轴：environment_steps。纵轴：原始奖励贪心策略折扣回报。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 第 615 步，有 bonus 的平均原任务回报为 0.1910，无 bonus 对照五条都已达到 0.7774。第 1200 步两者都达到 0.7774。此处额外新奇奖励延迟了原任务行为形成，末尾没有差异。

**结论边界。** 短小平稳链没有困难的稀疏探索瓶颈。这不是密度模型 pseudo-count 的验证，也不是对所有探索奖励的否定。计数新奇不等于信息增益或外部效用。

**继续实验。** 先分别记录访问覆盖与原任务回报，观察二者能否反向变化。再增加链长度、改变 bonus 系数，明确在哪个任务难度区间 bonus 的探索收益超过目标偏移的代价。

[源码](https://yingwen.io/crl-code/implementations/continual/count_bonus.py) · [逐种子记录](https://yingwen.io/crl-code/results/count_bonus/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/count_bonus/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/count_bonus/curves.json)

<a id="lesson-algorithm"></a>

## 5 · 三类方法的更新对象

**算法：这些是不同的信息处理机制，不应同时改变后再把全部差异归因于“探索”。**

1. 后验采样：根据真实观测更新模型后验。
  1. 在声明的 episode 或决策周期采样一个模型/价值假设。
  1. 由该假设规划行动，保持约定的时间一致性。
1. 乐观方法：更新统计量，计算假设支持的置信宽度。
  1. 用乐观目标或模型规划；实际数据更新统计量。
1. 内在奖励：用到达的观察计算 bonus，再训练预测器。
  1. 明确 bonus 在训练前还是训练后计算，并单独记录外部收益。

若 predictor 先在新观察上充分训练，再计算同一观察的新奇误差，bonus 会明显减小。若使用陈旧目标或回放重新计算 bonus，其含义也改变。时间顺序是机制的一部分，而不只是工程细节。

<a id="lesson-example"></a>

## 6 · 不确定性和路径概率的手算

先验 $\operatorname{Beta}(1,1)$，观察三次成功、一次失败，后验为 $\operatorname{Beta}(4,2)$。成功率均值为 $2/3$，参数方差为 $2/63\approx0.03175$。下一结果的预测方差却是 $(2/3)(1/3)=2/9$。两者不能互换。

把时间一致性放进一个具体环境。走廊位置为零到十，从零出发。向右移动一格，向左也移动一格；在零向左则原地停留。十次行动后结束，只有抵达十才能读取远端信息。因此任意一次向左都会使这次尝试来不及到达。

![十步走廊的时间位置网格，比较逐步独立随机与整段方向承诺的可达概率](https://yingwen.io/crl-figures/concept-deep3-exploration-corridor.svg)

相同走廊、相同十步预算。纵向是时间，横向是位置，网格深浅是精确占据概率。左侧每步独立选左右；右侧在起点等概率选“始终向左”或“始终向右”。下方以同一分母显示成功概率 $1/1024$ 和 $1/2$。线是例示路径，不是训练曲线。这个算例将 Bootstrapped DQN §3、§5 的时间一致探索思想单独取出；没有实现后验更新或神经网络训练。

十步链必须连续向右才看到远端信息。独立随机动作的概率为 1/1024；episode 级二策略采样的概率为 1/2。第二个过程并不意味着永远选右最好，它只展示了探索假设必须持续足够久才能被环境检验。这项几何诊断中两种规则都执行十次动作；下面增加可提前终止的安全动作时，实际交互成本也会随策略改变。

第二种过程不是每个状态都只会输出一个确定动作：开始时选择的方向本身仍是随机的。区别在于后续动作共享同一次选择。Bootstrapped DQN 保持的是一个价值头所导出的策略，而不是在一般任务中机械保持同一个动作；遇到弯曲路径时，一致的策略当然可以改变动作。

<a id="deep-exploration-task"></a>

### 6.1 · 把远端信息接到一个未知奖励模型

保留位置 $x\in\{0,\ldots,10\}$ 与左右确定性转移，给每个非终止位置增加一个安全结算动作 safe。safe 立即得到 $1/4$ 并真正终止。right 向右一格，left 向左一格且在零处反射；普通移动奖励为零。只有首次到达十时才抽一次二元奖励 $Y\in\{0,1\}$，到账后也真正终止。每个 episode 从零出发，$\gamma=1$，最多十次决策；若预算用尽仍未到端点，本轮结束且尾值为零。

未知量是固定的奖励模型 $\theta\in\{\mathrm{low},\mathrm{high}\}$。两种候选模型的端点成功率由设计者给定：$p_{\rm low}=1/5,p_{\rm high}=4/5$，初始 $q_0=\Pr(\theta=\mathrm{high})=1/2$。运行时不知道哪种模型为真。一共三轮 episode，每轮最多十次动作；首轮初始化，其余两轮允许重置位置与预算，真实 $\theta$ 和保存的后验不重置。给定 $\theta$，不同到达事件的端点奖励条件独立。没有中途换模型、额外端点查询或免费试走权限。

| 量 | 谁可以使用 | 何时提供 |
| --- | --- | --- |
| 位置、剩余动作数、左右转移、safe 奖励 | agent 与读者 | 任务开始时给定接口与规律 |
| 两种端点似然、初始 q | agent 与读者 | 候选模型与先验由设计者提供 |
| 真实 θ | 仅模拟器和独立参照 | actor 输入中始终遮挡 |
| 普通移动的 0、safe 的 1/4 | agent | 动作之后到账；两模型似然相同 |
| 端点 Y | agent | 首次抵达后才抽取并到账；此时才能更新 q |

从可观察的 $x$、剩余预算 $h$ 出发，在候选模型里作有限期动态规划。记 $V_0^\theta(x)=0$；非终止 $x<10$ 的动作值是下面三项。进入端点的奖励期望是 $p_\theta$，而不再读取一个端点自举值。safe 的 $1/4$ 也没有未来项。

$$
\begin{aligned}Q_h^\theta(x,\mathrm{safe})&=\tfrac14,\\Q_h^\theta(x,\mathrm{left})&=V_{h-1}^\theta(\max(0,x-1)),\\Q_h^\theta(x,\mathrm{right})&=\begin{cases}p_\theta,&x+1=10,\\V_{h-1}^\theta(x+1),&x+1<10,\end{cases}\\V_h^\theta(x)&=\max_aQ_h^\theta(x,a).\end{aligned}
$$

这里 h≥1；预算进入决策状态。并列时选 safe，随后按 left、right 的次序。候选模型允许内部计算，不能把模拟器的真实 θ 或未到达奖励当作观测。

沿仍可在预算内抵达的路径，high 模型的最优值为 $0.8$，策略连续向右；low 模型比较 $0.2$ 与 safe 的 $0.25$，选择结算。若因向左已失去可达性，即使 high 也会选 safe。这是同一任务的两条完整策略，由模型与有限预算共同决定；随机数只选择本轮读哪一个模型。

![走廊中普通零反馈与端点反馈的不同到达时机，比较固定模型和每步重抽的精确到达概率及实际动作成本](https://yingwen.io/crl-figures/deep-exploration-walkthrough-reach.svg)

上方问号表示 actor 尚未观察端点奖励，橙箭头为向右移动，绿色出口为每格可选的 safe。下方将走廊长度与最多动作数一同设为 H，比较 $q=1/2$ 时的端点反馈概率：episode 固定模型为 $q$，每步独立重抽为 $q^H$。前者平均实际动作 5.5，后者 $1023/512\approx1.998$；预算上限相同并不代表成本相同。原创有限任务精算，由[计算核](/crl-code/figures/deep-exploration-walkthrough.mjs)与[独立 Python](/crl-code/tutorials/deep-exploration-walkthrough.py)核对。

$$
\Pr(\mathrm{reach}\mid q,\mathrm{held})=q,\qquad\Pr(\mathrm{reach}\mid q,\mathrm{step})=q^H
$$

固定模型：起点抽到 high 就连续抵达。每步重抽：所有未终止步的后验仍为 q，抽样彼此独立，必须 H 次都抽到 high；任一次抽到 low 就选 safe 终止。这条 $q^H$ 推导依赖长度等于动作上限、上述候选策略及无中途信息。

$$
\mathbb E[N_{\rm held}]=1-q+Hq,\qquad\mathbb E[N_{\rm step}]=\sum_{j=0}^{H-1}q^j
$$

N 是实际原子动作数。第 j+1 次动作发生的概率在逐步重抽下为 $q^j$；固定模型则以 1−q 执行一次 safe，以 q 执行 H 次 right。几何诊断的公平左右和这里的模型重抽在 q=1/2 时数值相同，但后者的失败分支会收到 safe 奖励，不能把两套回报混用。

<a id="deep-exploration-feedback"></a>

### 6.2 · 一条新证据怎样改变下一条经验

给定已经收到的历史 $\mathcal D$，下一次端点奖励的预测成功率是 $\bar p(q)=0.2+0.6q$。普通移动和 safe 的后果在两种模型中完全相同，所以不会改变 q。端点 Y 到达后才使用 Bayes 法则。动作由保存的后验与自己的随机数决定；自己的模型选择本身不是来自环境的新证据。

$$
q^+=\frac{q\,p_{\rm high}^{Y}(1-p_{\rm high})^{1-Y}}{q\,p_{\rm high}^{Y}(1-p_{\rm high})^{1-Y}+(1-q)\,p_{\rm low}^{Y}(1-p_{\rm low})^{1-Y}}
$$

分母是已收到二元奖励的证据概率。先验各半时，Y=0 得到 q⁺=.2；Y=1 得到 q⁺=.8。一条失败降低 high 的权重，却未将它排除。

现在让消费者实际执行。模拟器固定 $\theta=\mathrm{low}$，actor 看不到这项参数。第一次模型抽样分位点取 $u=.4<q_0=.5$，因而抽到 high 并执行十次 right。端点奖励分位点取 .4，大于真实成功率 .2，这条合法路径得到 Y=0。后验降到 .2。重置位置后，第二次同样用 u=.4，它超过新 q，因而抽到 low，第一步 safe 结算 .25。没有再次到达端点，第二次奖励就没有补充模型证据。两次相同 u 用来显示后验对动作的作用，是指定路径的耦合，不是一批独立随机训练。

![实际到达记录把high后验从.5降到.2，下一episode的同一模型分位点转为safe，已知high也可能失败的对照](https://yingwen.io/crl-figures/deep-exploration-walkthrough-loop.svg)

沿箭头读“抽模型 → 十次 right → Y=0 → q=.2 → 下一轮 safe”。紫条是 high 后验质量，浅紫是 low。第二轮只花一次动作，后验保持 .2；若首次 Y=1，同一 u 会在下轮继续选择 high。底部已知 high 对照的失败概率仍为 .2，失败后 q=1。图中的每一步来自指定合法行动记录与准确似然，非学习性能曲线。

| 同一 low 模拟器中的轮次 | 模型抽样分位点 | 动作和实际步数 | 本轮奖励 | 本轮结束的 q |
| --- | --- | --- | --- | --- |
| 1 | .4 | 十次 right → 端点 | 0 | .2 |
| 2 | .4 | 一次 safe → 终止 | .25 | .2 |
| 3 | .1 | 十次 right → 端点 | 0 | 1/17≈.05882 |

第三轮的 u=.1 仍可能抽到 high，于是重新到达并收到另一条失败证据；两次失败的似然比使 q 降到 1/17。整条记录用 21 次真实动作得到总奖励 .25。若首次成功，q=.8 会提高以后抽到 high 的概率，从而改变未来端点证据的到达机会。代码也枚举了这些未在图中画出的成功与失败分支，不能把这一条记录当成期望。

$$
\begin{aligned}J_e(q;\theta,c)&=(1-\rho_c(q))\big[\tfrac14+J_{e-1}(q;\theta,c)\big]\\&\quad+\rho_c(q)\sum_{y=0}^1\Pr(y\mid\theta)\big[y+J_{e-1}(B(q,y);\theta,c)\big],\\J_0&=0,\qquad\rho_{\rm held}(q)=q,\quad\rho_{\rm step}(q)=q^{10}.\end{aligned}
$$

e 是剩余 episode 轮数，c 是模型抽样时钟；B 是上面的后验更新。这项递推评价给定抽样策略的真实环境期望，使用 θ 的部分属于独立参照，不进入 actor。它没有对 belief 状态中的动作取最大值，所以不声称求得 Bayes-adaptive 最优策略。

| 固定真实模型、三轮预算 | episode 固定模型：期望奖励 / 端点次数 / 实际动作 | 每步重抽：期望奖励 / 端点次数 / 实际动作 |
| --- | --- | --- |
| low | .68731 / 1.25382 / 14.28441 | .74985 / .00299 / 5.99380 |
| high | 1.71040 / 1.74618 / 18.71559 | .75176 / .00321 / 5.99966 |

这份精确表保留一个有用的失败对照：真实模型为 low 时，持续探路会放弃安全收益，本例三轮内的期望奖励反而低于逐步重抽；high 时额外端点信息与成功机会才带来更多收益。两种策略消耗的期望动作数也不同。时间一致性提高检验远端假设的机会，并不单独决定某个世界、某个预算下的收益排序。

$$
\operatorname{Var}(Y\mid\mathcal D)=\underbrace{0.16}_{\mathbb E[p_\theta(1-p_\theta)\mid\mathcal D]}+\underbrace{0.36q(1-q)}_{\operatorname{Var}(p_\theta\mid\mathcal D)}
$$

第一项是本模型族内每种已知硬币仍有的结果噪声，第二项是未知成功率的后验方差。已经知道 high、q=1 时，第二项为零，失败仍以 .2 的概率发生；实际失败后 q 仍是 1。预测残差和知识不足因此可以在同一环境中分开。

<a id="deep-exploration-heads"></a>

### 6.3 · 从采样模型到一个持续使用的价值头

如果模型太复杂，先抽整个 MDP 再精确规划可能难以实施。Bootstrapped DQN 改为维护 K 个随机化价值头：episode 开始均匀选择一个行为头，整轮按该头的贪心策略行动。走廊里的“本轮持续读取 high 假设”对应这种选择时钟；一般环境中固定头会随状态改变动作。每步重新选头可能在远端证据到达前中断行动。一个固定头只说明行动持续读同一条价值函数；在线训练若改变这条函数，其参数也仍会变化。

训练还要产生并保持头之间有意义的差异。data bootstrap 给经验 i 保存训练权重 $m_i^k$：该经验进入哪些头的样本或梯度。重放这条经验时读取保存的 mask，不把 mask 当作行为头选择。2016 论文附录 B 给逐条记录的形式，§6.1 的 Atari 讨论另采用 episode 共享的 Bernoulli flags，并报告 p=1 的共享数据版本；这些粒度应按具体实现声明。共享表征会在头间传递训练影响，mask=0 只屏蔽该头对这条经验的直接损失，不能宣称它的整个输出永远不变。

$$
\begin{aligned}a_i^{*,k}&=\arg\max_aQ_k(s_i',a;\vartheta),\\y_i^k&=r_i+\gamma(1-d_i)Q_k(s_i',a_i^{*,k};\vartheta^-),\\\mathcal L_k&=\sum_i m_i^k\,[\operatorname{stopgrad}(y_i^k)-Q_k(s_i,a_i;\vartheta)]^2.\end{aligned}
$$

这是头内 Double DQN 目标的常见记法：online 头选后继动作，对应的 target 头评价；目标副本在复制之间冻结，d=1 为真正终止，截断需另处理。共享参数由各头损失共同更新。data bootstrap 重抽/加权数据；TD bootstrap 用后继估计构造目标，两个 bootstrap 的对象不同。

每个头读自己的目标副本，使一条价值假设的后继答案可以向前传播。若把所有头的后继目标先平均再送给各头，便会改变这种对应。随机初始化是可训练参数的起点；它在后续训练中会被改变。固定随机 prior function 则是另一个始终不训练的加法函数，连同可训练残差共同参与预测和目标。

$$
Q_k(s,a)=f_{\vartheta_k}(s,a)+\beta p_k(s,a),\qquad \nabla_{\vartheta_k}p_k=0
$$

$p_k$ 先抽样再固定，$\beta$ 规定其尺度。残差 f 学习与固定 prior 的总和去拟合数据；在有充分证据的区域可以抵消 prior，在缺少数据处的行为还取决于表示、泛化和尺度。此式用于说明 2018 年 randomized prior 扩展，不把它写成 2016 年算法的必备定义。

![两头读取同一真实端点样本，通过保存的mask更新可训练残差而保持prior，另一个非终止探针比较各头自己的TD目标](https://yingwen.io/crl-figures/deep-exploration-walkthrough-heads.svg)

上半图读实际端点 Y=0、真终止 d=1，两个目标都为零。给定表格快照、学习率 .5、mask=(1,0)，头一总 Q 从 .8 到 .4，头二保持 .2；固定 prior 分别 .3、−.1 都不更新。下半图另给非终止快照，两个当前值均 .5，各自冻结目标 .8、.2 使更新结果为 .65、.35。箭头传递目标值，浅条为更新前、深条为更新后。这个小探针没有共享神经层、replay 训练或完整 Bootstrapped DQN。

这里的二元候选模型提供了一个精确后验参照，价值头探针则只执行表格快照更新；它们不是同一后验实现。多头的分歧还可能来自优化误差、数据覆盖与共享表征，头数和神经网络容量都不自动给出校准后验。Randomized Prior Functions §3 的精确采样检验依赖线性高斯模型、相应数据噪声和正则化；对非线性 DQN，应另外用覆盖、预测校准与声明预算内的行动结果检验，而不沿用该精确性结论。

<a id="lesson-code"></a>

## 7 · 后验、置信分数与深探索概率

精确 Beta 更新、带种子的 Thompson 采样、UCB 分数和链式概率；不需要训练神经网络。

```python
def beta_posterior(successes, failures, alpha=1., beta=1.):
    a, b = alpha+successes, beta+failures
    if min(a, b) <= 0 or min(successes, failures) < 0:
        raise ValueError('positive prior and nonnegative counts required')
    return a, b, a/(a+b), a*b/((a+b)**2*(a+b+1))


def thompson_action(counts, seed):
    rng = random.Random(seed)
    samples = [rng.betavariate(*beta_posterior(s, f)[:2]) for s, f in counts]
    return max(range(len(samples)), key=samples.__getitem__), samples


def ucb_scores(means, counts, time):
    if time < 1 or len(means) != len(counts) or any(n < 0 for n in counts):
        raise ValueError('valid counts and time required')
    return [float('inf') if n == 0 else q+math.sqrt(2*math.log(time)/n)
            for q, n in zip(means, counts)]


def coherent_chain_probability(length):
    """Only all-right succeeds; unbiased independent actions vs one episode head."""
    if length < 1:
        raise ValueError('positive chain length required')
    return 0.5**length, 0.5
```

运行 demo 查看 posterior 与链概率，运行 test 检查共轭更新和未访问动作处理。bsuite 提供 Deep Sea 等诊断环境及随机化价值基线；它是后续作者团队框架，不等同于 2016 论文的全部原始 Atari 训练。RND 作者仓库则提供该论文的神经训练代码。

贯穿例另有[单文件标准库 Python 教程](/crl-code/tutorials/deep-exploration-walkthrough.py)，下载到空目录即可运行。默认打印各次实际动作记录、所有三轮分支的精确期望与成本；--test 用 Fraction 有理数和完整方向轨迹枚举核对概率。JS 用位置分布 DP 与后验分支递推，Python 用批量似然和前向分支枚举，二者独立核对。

无需第三方库。JSON 的 `q_before` / `q_after` 是已到达证据的后验，`primitive_actions` 是实际交互成本；指定分位点生成的记录不等于 iid 训练实验。

```sh
python3 deep-exploration-walkthrough.py --test
python3 deep-exploration-walkthrough.py
```

<a id="lesson-branches"></a>

## 8 · 假设边界与 CRL 的持续探索

- 不可逆世界：探索可能进入无法返回的状态。episode 重置得出的结论不能直接迁移到单生命期。
- 变化后验：对非平稳成功率持续累积旧计数会过度自信；遗忘窗口或变化模型改变了统计假设。
- 奖励混合：内在奖励系数改变行为目标，评估应单独报告外部任务收益和探索成本。
- 随机性混淆：环境噪声不会因为多看几次完全消失；高误差不一定意味着可学习的信息。

CRL 中新任务、新预测和新 option 都可能需要新的数据。应比较固定预算下的信息获取与后续适应，而不仅看访问状态数量。能否识别旧知识何时失效，是持续探索与一次性探索的重要区别。

<a id="lesson-check"></a>

## 9 · 练习与答案

- 问：两个 posterior 均值相同，探索优先级是否一定相同？答：不一定；后验宽度和行动的信息后果可以不同。
- 问：每步随机选价值头等价于 episode 固定头吗？答：不等价；前者可能破坏连续行动的假设一致性。
- 问：RND bonus 下降是否意味着任务学会了？答：不意味着；它只说明所访问输入的随机特征更容易预测。
- 实验：保持十步链不变，只把随机决策从每步改为每个 episode；核对 1/1024 与 1/2，但不要将这个固定路径概率称为完整算法的学习曲线。
- 问：新走廊第二轮 safe 奖励是否增加了 θ 的知识？答：没有；两模型都预言 .25，q 保持 .2。若把途中九个零当作九次端点失败，便会凭未提供的信息错误收缩后验。
- 问：已知 high 仍失败，是否应该把它的 posterior 调低？答：本模型已知时 q=1 保持；.2 的失败概率属于模型本身。若失败来自超出声明模型的变化，才需要另设变化模型与检验。
- 问：固定模型的三轮收益是否必然更多？答：low 对照给出反例；还须读取实际步数、safe 机会成本及评价预算。

## 从本章进入实践

[行动与经验](https://yingwen.io/zh/continual-rl/code/#practice-action-evidence)：行动决定了能得到哪些证据。学习又怎样改变下一次行动？



<a id="chapter-code"></a>

## 下载与运行

标准库解析机制实验；不包含完整神经网络训练或大型 benchmark。

[下载 extended_foundations_lab.py](https://yingwen.io/zh/continual-rl/download/extended_foundations_lab.py)

```sh
python3 extended_foundations_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Auer、Cesa-Bianchi、Fischer · Finite-time Analysis of the Multiarmed Bandit Problem](https://link.springer.com/article/10.1023/A:1013689704352)：UCB1 的置信宽度、有限时间分析及平稳 bandit 条件。

- [Osband、Russo、Van Roy · (More) Efficient Reinforcement Learning via Posterior Sampling](https://arxiv.org/abs/1306.0940)：§§2–4：有限 episode、模型 posterior sampling、策略承诺与正确先验下的采样恒等式；PSRL 不等于 Bayes-adaptive 最优控制。

- [Osband et al. · Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)：§§2–5、§6.1 与附录 B：行为头时钟、各头目标、固定经验 mask 与共享数据版本。

- [Osband、Aslanides、Cassirer · Randomized Prior Functions for Deep Reinforcement Learning](https://arxiv.org/abs/1806.03335)：§3、§4：固定 prior 加可训练残差；线性高斯精确采样条件与非线性推广边界。

- [DeepMind · bsuite](https://github.com/google-deepmind/bsuite)：作者团队后续诊断平台，包含 Deep Sea；不是原文所有实验的原始快照。

- [Burda et al. · Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)：固定随机目标、预测误差 bonus 与外部/内部收益。

- [OpenAI · RND 原始代码](https://github.com/openai/random-network-distillation)：论文作者实现；查看 predictor、奖励归一化与 rollout 的顺序。

- [UC Berkeley · CS 285](https://rail.eecs.berkeley.edu/deeprlcourse/)：原课程的 exploration、model-based RL 与 offline RL 讲义和视频入口；按问题专题阅读，不必按网络规模划分领域。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-exploration#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-exploration#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-deep-exploration)
<a id="study-connections"></a>

## 与教材主线的衔接

本章是可按问题选择的并列研究分支，不要求先读完其他深度研究分支。

### 一次策略更新，为什么能改善未来？

精确策略改善使用旧策略的真实价值；策略梯度使用与目标匹配的访问分布。值函数近似或梯度估计误差会破坏这些推理的前提。

函数逼近与深度方法：PPO 的动作概率比不等于新策略的状态访问比。连续动作 actor 还会追逐 critic 的误差。限制局部更新尺度与证明实际回报单调提高是不同要求。

持续学习中的研究问题：动作不仅改变世界，也改变未来数据与学习。比较冻结策略不等于比较持续更新的智能体；什么时候应付出当前回报去获得长期有用的经验？

[精确策略改善](https://yingwen.io/zh/continual-rl/foundations/tabular/dynamic-programming/) → [策略梯度定理](https://yingwen.io/zh/continual-rl/foundations/approximation/policy-gradient/) → [TRPO 与 PPO 的近似](https://yingwen.io/zh/continual-rl/foundations/deep/trust-region/) → [持续控制的比较器](https://yingwen.io/zh/continual-rl/algorithms/control/)


### 可进一步检验的问题

- [08 · 什么样的时间抽象值得加入技能库？](https://yingwen.io/zh/continual-rl/research/#research-useful-options)：时间一致探索解释了长行为的一种用途，同时要求把覆盖改善与技能自身的执行成功率分别评价。
- [09 · 子目标从哪里来，为什么学习它能帮助总体任务？](https://yingwen.io/zh/continual-rl/research/#research-goal-construction)：内在奖励与学习进展改变智能体练习什么；研究目标构造时，还需检查这种练习是否改善后续外部任务。
- [14 · 应当探索什么、练习什么，以及如何保留未来交互与学习的机会？](https://yingwen.io/zh/continual-rl/research/#research-experience-selection)：环境噪声与知识不足会产生不同的信息价值，时间一致探索进一步检验哪些经验值得为之连续行动。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)
- [目标条件化与子任务构造](https://yingwen.io/zh/continual-rl/construction/goals/)
- [Options 与技能发现](https://yingwen.io/zh/continual-rl/construction/options/)
- [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

对应原始材料：Osband、Russo、Van Roy：Posterior Sampling；Bootstrapped DQN；RND。本文为原创讲解，原书、论文与上游代码保留各自许可。
