# MDP、回报与价值：序列决策的数学对象

表格强化学习 · 第 2 章

动作会改变后续状态时，需要评价整个未来。本章从随机交互过程推导价值与 Bellman 方程，明确后续算法共同使用的数学对象。

## 本章内容

- 区分状态、观测、策略和环境模型。
- 从回报推导状态价值、动作价值与 Bellman 方程。
- 解释 Markov、平稳、终止和截断条件。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 概率与期望

概率非负且总和为一；期望是按概率加权的平均，不是单次结果。条件期望只比较满足已知条件的情形。

### 递推与赋值

递推通过已有估计计算新估计。下标表示时间；更新符号表示程序中的赋值，不是数学恒等式。

<a id="problem-definition"></a>

## 本章的问题定义

在完全可观测的序列决策问题中，动作改变下一状态和未来奖励。先定义数学对象，再区分评价与控制。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。

### 需要求解的对象

评价问题求固定策略的价值 $v_\pi$；控制问题选择策略 $\pi$ 使 $J(\pi)$ 尽可能大。

### 信息与数据权限

本章定义模型，不预设算法已经知道模型。已知核时可计算期望；未知核时需另规定数据来源。

$$
v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s],\quad G_t=\sum_{k\ge0}\gamma^kR_{t+k+1},\qquad \sup_\pi J(\pi)
$$

左式定义预测对象；右式定义控制目标。两者不是同一个优化问题。

### 成立条件与解的含义

- 平稳 Markov 核与充分状态。
- 本章折扣式使用有界奖励；有限时域需把剩余时间纳入价值索引。

判断准则：能从同一交互分布推导回报、价值和 Bellman 方程，并说明哪一步用了 Markov 性和策略固定条件。

### 适用边界

- 把已写出 Bellman 方程等同于已有可计算、可收敛的学习算法。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)：本章对一般历史过程增加充分状态与平稳核假设，得到 MDP 方程。

- 改变评价目标 · [平均奖励与差分价值](https://yingwen.io/zh/continual-rl/algorithms/average-reward/)：本章采用折扣求和而非平均奖励；时间聚合不同，两类最优策略不能默认相同。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

当前奖励不能直接比较动作的长期后果；直接枚举全部未来轨迹不可行。

### 本章的核心思路

按第一步拆开回报，并利用充分状态把后续条件期望重新写成价值函数。

1. [先检查状态是否足够](#mdp-state)：只有丢弃历史不改变下一步条件分布时，才能在当前状态上递推。

2. [把轨迹期望变成局部方程](#mdp-bellman)：回报分解与全期望公式给出固定策略 Bellman 方程。

3. [从预测转到动作选择](#mdp-optimal)：将策略平均改为最优动作选择，得到另一组控制方程，而非同一方程换名字。

结论与条件：有限折扣 MDP 的 Bellman 算子有唯一相应固定点；这依赖模型设定，不是所有学习表示的性质。

### 相关方法改变了什么

- 回合终止 / 时间截断：自然终止改变未来回报；采样窗口结束通常只改变估计方式。

- 固定策略价值 / 最优价值：前者指定未来行为；后者还对允许的未来行为取上确界。


<a id="lesson-setting"></a>

## 1 · 交互过程与符号

现在取消上一章“每次回到相同条件”的简化。一次抓取可能移动物体、消耗电量，也可能为下一次操作腾出空间。当前成功率最高的动作未必带来最好的后续处境。要评价这种选择，必须同时描述智能体目前处于什么情形、动作怎样改变情形，以及未来奖励怎样进入当前目标。状态、转移和回报就是为这三个问题引入的。

智能体在时刻 t 观察 $S_t$、选择 $A_t$，环境返回 $R_{t+1}$ 与 $S_{t+1}$。奖励下标属于刚完成的转移。有限 MDP 用状态集合 $\mathcal S$、合法动作 $\mathcal A(s)$ 和联合模型 $p(s',r\mid s,a)$ 描述后果。

$$
p(s',r\mid s,a)=\Pr(S_{t+1}=s',R_{t+1}=r\mid S_t=s,A_t=a),\qquad \sum_{s',r}p(s',r\mid s,a)=1
$$

联合分布允许奖励和后继状态相关，也允许一个动作产生多种后果。

策略规定智能体怎样选动作，模型规定动作之后会怎样。知道一个不等于知道另一个。表格型方法给有限状态或状态动作分别保存数值，本章暂不引入共享神经网络。

例如同一动作以 .75 概率得到奖励一并回到原状态，以 .25 概率得到奖励五并终止。模型应保存这两个联合后果，而不只是“平均奖励二、平均下一状态”。状态可以是类别，类别编号的平均往往没有环境含义；价值计算需要的是后继价值的概率加权。

下面用一张地图把这些对象连接起来，并沿用它进入动态规划。位置坐标写作（列，行），左上角为（0,0）。所有位置都可观察；墙（1,1）固定，四个动作是上、右、下、左。进入 G 得 1 并真正终止；其他转移得 −0.04，撞墙或越界留在原格。状态是当前位置，模型给出动作的后果，策略另行规定选哪个动作。

![4×3 网格、墙和终点，给定路径从左下起点先上两步到记录边界 P，再由模型沿顶行走到 G；下方显示五个转移奖励与截断目标。](https://yingwen.io/crl-figures/mdp-dp-task.svg)

原创已知模型算例，$\gamma=0.9$。蓝实线是给定的前两步记录，紫虚线是该路径的模型后续；并未采样训练。记录在 P 停止时，环境还未终止。进入 G 的奖励是 1，G 本身的价值为 0；这些约定在下一章保持相同。

| 状态与动作 | 后继与奖励 | 联合模型中的非零项 |
| --- | --- | --- |
| （0,2），上 | 到（0,1），−0.04 | p((0,1),−0.04 \| (0,2),上)=1 |
| （0,1），右 | 撞墙，留在（0,1），−0.04 | p((0,1),−0.04 \| (0,1),右)=1 |
| （2,0），右 | 进入 G，1，真正终止 | p(G,1 \| (2,0),右)=1 |

<a id="mdp-state"></a>

## 2 · Markov 性与平稳性

$$
\Pr(S_{t+1},R_{t+1}\mid S_0,A_0,\ldots,S_t,A_t)=\Pr(S_{t+1},R_{t+1}\mid S_t,A_t)
$$

当前状态包含预测下一步所需的历史影响。它不表示系统没有历史，而表示相关历史已被状态概括。

如果走廊尽头应选的方向由十步前的灯色决定，当前画面一样却需要不同动作，那么画面不构成充分状态。可以加入灯色记忆，或维护隐藏情境的概率分布。价值更新更快并不能弥补缺失信息。

平稳性要求同一状态动作的后果规律不随绝对时间变化。Markov 与平稳是不同条件；可以把时间或情境加入状态得到更大系统，但新变量是否可观测、可估计，仍需解决。

在这张固定地图里，同一位置和动作总产生同一后果，当前位置足以概括历史对后续转移的影响。若墙只在拿到钥匙后开放，就要把是否持有钥匙也加入状态；相同位置下的两个历史会产生不同转移。记录还剩几步则只有在它本来限制任务时，才是影响价值的状态变量。

<a id="lesson-derive"></a>

## 3 · 奖励与回报

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1},\qquad G_t=R_{t+1}+\gamma G_{t+1}
$$

有限和在真实终点 T 停止。第二式把第一项取出，剩余项提取公共折扣。这是代数恒等式，尚未涉及学习。

对不自然终止的任务，可把求和上限改为无限。奖励有界且 $0\le\gamma<1$ 时，回报绝对有界；$\gamma=1$ 需要终止与可积性等额外条件。目标改变会改变策略偏好，因此 γ 不能只当作无关紧要的训练旋钮。

真实终止后可以引入奖励永为零的吸收状态，终点价值设为零，但进入终点的奖励仍需计入。达到一次采样预算只表示没有继续观察，不表示未来不存在。有限时域任务还可能要将剩余时间放入状态。

![确定性任务 A 到 B 再到终点，标明转移奖励、真实价值，以及终止与采样截断的不同目标。](https://yingwen.io/crl-figures/learning-classic-boundaries.svg)

原创精确计算：A→B 奖励 0，B→T 奖励 1，$\gamma=0.9$，唯一动作、无随机。上排是任务真值，下排的 0.36 使用明确给定的估计 $V(B)=0.4$，不是训练所得真值。

沿图逐项算一次。B 的下一次转移得 1 并终止，所以 $v(B)=1$；A 的第一步没有奖励，但后续仍有 B 的回报，所以 $v(A)=0+0.9\times1=0.9$。终点 T 的价值为 0，因为从 T 起不再取得奖励；它不抹掉 B→T 已经取得的 1。

现在只采到 A→B 就用完记录预算。若当时估计 $V(B)=0.4$，截断目标是 $0+0.9\times0.4=0.36$。它与真实值 0.9 的差来自尾值估计不准；若把记录结束误作任务终止，则目标变为 0，又引入了一个不同的错误。即使使用完全相同的已观察奖励，边界语义也会改变更新。

因此程序至少需要保存两种信息：环境是否真正终止，以及采样是否暂时停止。前者决定余项是否为零，后者决定能否继续取得经验。若“只允许走一步”本来就是任务规则，才应把剩余时间纳入状态并按这个有限期限重新定义价值。

把同一判断带回网格。给定路径的五次奖励为 $(-0.04,-0.04,-0.04,-0.04,1)$，所以完整回报是 $-0.04(1+0.9+0.9^2+0.9^3)+0.9^4=0.51854$。前两步后停在 P；若接着沿顶行走，其尾值为 $-0.04-0.9\times0.04+0.9^2=0.734$。保留这项尾值，截断目标为 $-0.04-0.9\times0.04+0.9^2\times0.734=0.51854$；误将记录结束当成任务终止则只有 $-0.076$。这里尾值按给定后续路径精确计算；换用一个估计尾值时，还要承担它的误差。

每步奖励加常数也可能改变任务：若策略影响 episode 长度，新奖励会鼓励或惩罚停留。应先问希望智能体实现什么，再设计信号，而不是训练后才用另一套未说明的目标解释结果。

<a id="mdp-bellman"></a>

## 4 · 从条件期望到 Bellman 方程

$$
v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s],\quad q_\pi(s,a)=\mathbb E_\pi[G_t\mid S_t=s,A_t=a],\quad v_\pi(s)=\sum_a\pi(a\mid s)q_\pi(s,a)
$$

状态价值平均第一步动作；动作价值固定第一步动作，之后仍按 π。最后一个式子由全期望公式得到。

$$
q_\pi(s,a)=\sum_{s',r}p(s',r\mid s,a)\left[r+\gamma\sum_{a'}\pi(a'\mid s')q_\pi(s',a')\right]
$$

先拆分回报，再按一步后果分组。Markov 性使后半段只需条件于下一状态。这里对策略求平均，而不是选择最大动作。

$$
v_\pi=r_\pi+\gamma P_\pi v_\pi,\qquad v_\pi=(I-\gamma P_\pi)^{-1}r_\pi
$$

把各状态的方程堆叠。Pπ 是策略诱导转移矩阵；只保留非终止状态时，行和可以小于一。有限折扣条件保证可逆，程序通常使用迭代或线性求解而非显式求逆。

现在固定网格策略 π：每个非终止格四方向各选 1/4。即使环境后果确定，行动仍使回报随机。左上 P 的上、左动作留在原地，右、下分别到（1,0）和（0,1）；四项转移奖励都为 −0.04。把这四个分支按策略求平均，就得到一个可以逐项检查的 Bellman 方程。

$$
v_\pi(0,0)=-0.04+\frac{0.9}{4}\left[2v_\pi(0,0)+v_\pi(1,0)+v_\pi(0,1)\right].
$$

两项原地转移仍属于模型，不能删去或重新归一化。列出其余九个非终止格的方程，便形成十元线性系统。

解这个系统，得到 $v_\pi(0,0)=-0.187077$、$v_\pi(1,0)=-0.022438$、$v_\pi(0,1)=-0.257083$，起点 $v_\pi(0,2)=-0.263571$。代入上式约得 $-0.04+0.225[2(-0.187077)-0.022438-0.257083]=-0.187077$。这些值评价均匀随机行为；它经常停留或绕路，近期步成本可以超过被折扣的到达奖励。它们与沿指定五步路径的回报 0.51854 回答不同问题。

<a id="experiment-iterative_policy_evaluation"></a>

### 实验：实验 · Bellman 方程求的是哪个策略的价值？

固定同一个转移模型和策略，仅改变读取旧值或新值的顺序，会不会改变最终评价对象？

**环境与可用信息。** 六格确定性链，0–4 为非终止状态，5 为终点。在最左端向左仍停在原地。进入终点奖励 1，其余转移奖励 −0.01，γ=0.95。固定策略每个状态向左概率 0.2、向右概率 0.8；此处不改善策略。

**设置。** 价值从零开始。每轮扫描五个非终止状态，枚举两个动作的已知模型。同步评价整轮只读旧价值；原地对照按状态 0 到 4 更新，可以读到本轮已更新值。保存的五个 seed 都运行 1200 轮，但算法无采样随机性。

**检验的机制。** 两种扫描都应用同一个固定策略的 Bellman 期望关系。同步和原地是求解器差别，不是两个不同的控制目标。原地方法可能更快传播信息，但传播效果也受扫描方向与转移结构影响。

**测量。** 纵轴是相对于独立线性方程解的最大绝对价值误差，不是相对于最优价值的误差。横轴一轮包含五次状态备份；它不表示五次真实环境交互。

```bash
python3 implementations/extended_classic/iterative_policy_evaluation.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/iterative_policy_evaluation/curves.svg)

横轴：model_sweeps。纵轴：固定策略最大价值误差。每种方法 1200 model_sweeps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 第 15 轮，同步误差为 0.01164，原地为 0.000406；第 60 轮为约 0.000000000388 与浮点精度。末尾两者均在浮点精度。五条完全重合的曲线反映确定性计算，不是统计置信度很高。

**结论边界。** 已知小模型、固定策略和折扣收缩使问题可精确求解。它没有学习模型，没有采样误差，也没有评价一个不断改变的学习器。

**继续实验。** 先用当前策略概率建立线性方程，再故意换成逐状态取最大动作，观察求解对象怎样从策略评价变成最优控制。将原地扫描顺序反转，验证“相同固定点”不意味着“相同中间过程”。

[源码](https://yingwen.io/crl-code/implementations/extended_classic/iterative_policy_evaluation.py) · [逐种子记录](https://yingwen.io/crl-code/results/iterative_policy_evaluation/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/iterative_policy_evaluation/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/iterative_policy_evaluation/curves.json)

<a id="mdp-optimal"></a>

## 5 · 最优价值与动作选择

若所有状态满足 $v_{\pi'}(s)\ge v_\pi(s)$，称新策略不差于旧策略。不同策略可能具有相同的价值向量；把同值策略视为等价后，这个逐状态比较构成偏序，而两个策略也可能分别擅长不同起点、无法互相排序。有限折扣 MDP 存在一个确定性平稳策略同时实现所有状态的最优值。

$$
v_*(s)=\max_\pi v_\pi(s)=\max_a\sum_{s',r}p(s',r\mid s,a)[r+\gamma v_*(s')]
$$

先确定动作，再对随机后果求平均。若把最大值移到未知随机后果之后，相当于允许智能体提前知道尚未发生的事件。

方程定义最优值，并不意味着数值已经可用。已知模型仍需要求固定点，未知模型则需要经验。动态规划、MC 和 TD 在取得后果信息及构造目标的方式上不同。

<a id="lesson-example"></a>

## 6 · 两状态完整算例

A 可以退出得 1 并终止，或继续得 0 到 B；B 可以退出得 2，或继续得 .5 回 A。γ=.9，策略在每个状态均匀选两动作。根据转移与策略概率得到两个线性方程。

$$
v(A)=.5+.45v(B),\qquad v(B)=1.25+.45v(A),\qquad v(A)=\frac{425}{319},\quad v(B)=\frac{590}{319}
$$

B 的即时期望为 .5×2+.5×.5=1.25。继续分支的 .45 是策略概率 .5 乘折扣 .9。解约为 (1.332288,1.849530)，不是最优值。

mdps 模式还输出奖励 (1,2,3)、γ=.5 的各起点回报 (2.75,3.5,3)。只观察奖励 1、γ=.9 并保留后继价值 10 时，目标为 10；误当终止则只有 1。

<a id="lesson-code"></a>

## 7 · 模型表示与实验观察

**算法：回报递推与模型期望的计算顺序**

1. 输入一段已观察奖励、折扣和边界余项。
1. 若已到真实终止，边界余项为零；否则明确它的估计来源。
1. 令 G 等于边界余项。
1. 从最后一个奖励反向遍历：
  1. 令 G = 当前奖励 + 折扣 × G。
  1. 将 G 保存为该时刻的回报目标。
1. 若求动作价值，则枚举联合后果，对奖励加后继价值做概率加权。

明确表示真实终止的模型、回报递推与随机后果采样。完整脚本调用下一章的公共策略评价函数。

```python
# model[s][a]: (probability, reward, next_state) outcomes.
# None means TRUE terminal, not a budget truncation.
MODEL = {0: [[(1., 1., None)], [(1., 0., 1)]],
         1: [[(1., 2., None)], [(1., .5, 0)]]}
CONTROL_MODEL = {0: [[(1., .2, None)], [(1., 0., 1)]],
                 1: [[(1., 1., None)], [(1., -1., None)]]}

def returns(rewards, gamma, tail=0.0):
    result, value = [0.0]*len(rewards), tail
    for t in reversed(range(len(rewards))):
        value = rewards[t] + gamma * value
        result[t] = value
    return result

def action_values(model, state, values, gamma):
    return [sum(p * (r + gamma * (0 if sp is None else values[sp]))
                for p, r, sp in outcomes) for outcomes in model[state]]

def step(model, state, action, rng):
    outcomes = model[state][action]
    _, reward, successor = outcomes[choose([x[0] for x in outcomes], rng)]
    return reward, successor
```

模型用状态索引、动作索引和后果列表表示。每个后果包含概率、进入该后果的奖励和后继状态；None 专门表示真实终止。回报函数的 tail 是未观测余项的边界估计，真正终止时为零，采样截断时不应自动归零。两者由调用者的任务定义决定，而不是由数组长度决定。

把 A 的继续动作改成两个随机后果后，先手算动作价值，再与 action_values 的求和结果核对。随后分别改变奖励和转移概率，观察价值为何都改变。若只是把状态索引重新编号，策略的物理含义和价值应保持对应不变；这可以检查是否误把索引当作数值状态特征。

贯穿两章的网格另有一个可独立下载的标准库脚本：[mdp_dp_walkthrough.py](/crl-code/tutorials/mdp_dp_walkthrough.py)。build_model 明确表示撞墙和进入终点，evaluate_policy 独立解策略线性方程，discounted_return 检查路径与边界。下载后运行下面两条命令；输出的是已知模型计算，不产生随机训练轨迹。

先核对手算、终止/截断与退化检查，再查看模型和价值 JSON

```bash
python3 mdp_dp_walkthrough.py --test
python3 mdp_dp_walkthrough.py
```

<a id="lesson-branches"></a>

## 8 · 假设与持续学习

有限平稳 MDP 提供可分解的参照，而不是规定所有现实环境必须如此。长期交互可能改变动力学、观测和奖励，也可能在固定大环境中持续发现新规律。需要逐项说明哪些状态变量、模型条件与评价时间范围发生变化。

没有自然终点不意味着一定使用无限无折扣总奖励；该总和常常发散。可以选择折扣回报、有限寿命效用，或在适当链结构下定义平均奖励率。数据协议与问题目标应当分别陈述。

<a id="lesson-check"></a>

## 9 · 带答案练习

问：奖励属于当前状态还是后继状态？答：一般属于一次状态动作转移。只依赖某个状态是特定环境的简化，不应据此随意移动奖励下标。

问：单次回报为 5，正确价值为 2，是否模型一定有错？答：不是。价值是条件期望，单个随机结果可以偏离均值。

问：γ=0 时优化什么？答：即时奖励的条件期望。环境仍可能转移，但优化目标不计后续影响，未必满足原先期待的长期行为。



<a id="chapter-code"></a>

## 下载与运行

Python 3，仅标准库。下载完整文件后在其目录运行；页面展示核心区域。小环境和解析检验用于理解机制，不是论文基准复现。

[下载 tabular_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/tabular_textbook_lab.py)

```sh
python3 tabular_textbook_lab.py mdps
python3 tabular_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：对应第 2–8 章。以下推导、例子和标准库实现独立编写；原书用于核对定义、算法关系与适用条件。

- [David Silver · UCL 强化学习课程](https://davidstarsilver.wordpress.com/teaching/)：原课程页面提供 MDP、动态规划、预测、控制、探索和规划的讲义及视频。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-tabular-mdps#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-tabular-mdps#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-tabular-mdps)
<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 当前输入保留了哪些历史信息？

Bellman 方程先假定有足够的状态。表格为不同状态分别存值；它不负责从相同观察中恢复被遗漏的历史。

函数逼近与深度方法：特征共享是在已知信息上泛化；递归状态则保留过去信息。增加网络宽度不等于补回历史，低训练误差也不证明输入满足 Markov 性。

持续学习中的研究问题：策略改变以后，原来的状态压缩是否仍能预测行动后果？构造状态的网络、运行时记忆、资格迹与优化器状态如何共同更新？

[MDP 的状态条件](https://yingwen.io/zh/continual-rl/foundations/tabular/mdps/) → [表示与泛化](https://yingwen.io/zh/continual-rl/foundations/approximation/features-control/) → [不完全可观测](https://yingwen.io/zh/continual-rl/foundations/deep/partial-observability/) → [智能体状态](https://yingwen.io/zh/continual-rl/construction/state/)


### 可进一步检验的问题

- [奖励表示与奖励学习：目标怎样进入智能体？](https://yingwen.io/zh/continual-rl/research/#research-reward-design)：先明确状态、奖励与回报的定义，再问设计者的偏好能否由这一状态上的奖励表达。
- [01 · 什么目标能够评价一个始终在学习的智能体？](https://yingwen.io/zh/continual-rl/research/#research-lifetime-objective)：折扣和与有限期回报已经给出不同的比较对象；评价持续学习器时，还需把初始学习和恢复代价放回这些时间范围。
- [02 · 有限的内部状态应当保留哪些历史信息？](https://yingwen.io/zh/continual-rl/research/#research-agent-state)：Markov 条件规定状态需要支持哪些预测；当前观察不满足这一条件时，才需要研究内部状态应保留哪些历史。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)
- [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)
- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)

对应原始材料：3.1–3.7。本文为原创讲解，原书、论文与上游代码保留各自许可。
