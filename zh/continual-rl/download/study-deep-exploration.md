# 探索与不确定性：后验、乐观估计和时间一致行动

为什么每一步都随机，并不等于有效获取长期有用的信息？

## 本章内容

- 区分环境随机性与知识不足。
- 推导 Bernoulli 后验与 UCB 的置信思路。
- 理解 posterior sampling、Bootstrapped DQN 与内在奖励各自改变什么。

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


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 条件概率与期望

区分随机变量、一次样本与条件分布；可以使用 Bayes 法则和全期望公式。

### MDP 与价值

理解状态、动作、转移、奖励、策略和 Bellman 递推；学习数据可以来自不同策略。

### 优化与梯度

知道固定目标、参数梯度、约束和期望目标的含义；神经网络只是函数表示的一种选择。

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

Bootstrapped DQN 用多个价值头和随机化训练近似多种可信价值函数，并在 episode 内保持一个头。共享表征与有限 bootstrap 不构成精确 Bayes posterior；实验里应测校准、覆盖和探索效率，而不是把所有 ensemble 都解释成同一后验。

<a id="lesson-bonus"></a>

## 4 · 乐观估计与新奇奖励的不同角色

$$
U_t(a)=\hat\mu_t(a)+\sqrt{\frac{2\log t}{N_t(a)}}
$$

UCB1 形式以样本均值加置信宽度排序；未访问动作优先探索。其经典保证依赖平稳、有界且适当独立的 bandit 观测。

宽度随样本数减小，使已确定的低价值动作逐步退出。Hoeffding 型尾界控制经验均值偏差，由高概率上界产生乐观行动。序列 MDP 的 bonus 还需处理转移估计、规划误差与访问次数，不能仅把 bandit 式原样贴在神经 Q 输出上就获得 regret 保证。

$$
Q_k(s,a)=\hat r(s,a)+b_k(s,a)+\gamma\sum_{s'}\hat P(s'\mid s,a)\max_{a'}Q_{k-1}(s',a')
$$

示意性乐观模型备份：bonus 进入规划，沿未来状态传播。具体置信项必须由所用假设推导。

$$
r_t^{\rm int}=\|f_\theta(O_{t+1})-f_{\rm fixed}(O_{t+1})\|^2
$$

RND 的固定随机目标与学习预测器产生新奇信号。它不直接估计环境奖励或转移模型的 Bayes 方差。

RND 对确定随机映射做预测，避免将随机下一状态预测误差直接当作新奇的某些问题，但无关高维变化、表征泛化和遗忘仍能影响 bonus。熵正则则直接偏好较随机的动作分布；它不会自动区分有信息的状态与无信息的随机循环。

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

十步链必须连续向右才看到远端信息。独立随机动作的概率为 1/1024；episode 级二策略采样的概率为 1/2。第二个过程并不意味着永远选右最好，它只展示了探索假设必须持续足够久才能被环境检验。

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

- [Osband、Russo、Van Roy · (More) Efficient Reinforcement Learning via Posterior Sampling](https://arxiv.org/abs/1306.0940)：模型 posterior sampling 与 episode 级策略承诺。

- [Osband et al. · Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621)：随机价值函数与多步一致探索。

- [DeepMind · bsuite](https://github.com/google-deepmind/bsuite)：作者团队后续诊断平台，包含 Deep Sea；不是原文所有实验的原始快照。

- [Burda et al. · Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)：固定随机目标、预测误差 bonus 与外部/内部收益。

- [OpenAI · RND 原始代码](https://github.com/openai/random-network-distillation)：论文作者实现；查看 predictor、奖励归一化与 rollout 的顺序。

- [UC Berkeley · CS 285](https://rail.eecs.berkeley.edu/deeprlcourse/)：原课程的 exploration、model-based RL 与 offline RL 讲义和视频入口；按问题专题阅读，不必按网络规模划分领域。

<a id="study-connections"></a>

## 与教材主线的衔接

本章是可按问题选择的并列研究分支，不要求先读完其他深度研究分支。

[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)
- [目标条件化与子任务构造](https://yingwen.io/zh/continual-rl/construction/goals/)
- [Options 与技能发现](https://yingwen.io/zh/continual-rl/construction/options/)
- [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

对应原始材料：Osband、Russo、Van Roy：Posterior Sampling；Bootstrapped DQN；RND。本文为原创讲解，原书、论文与上游代码保留各自许可。
