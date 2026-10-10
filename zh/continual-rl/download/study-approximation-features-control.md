# 特征、泛化与半梯度控制

函数逼近与经典进阶方法 · 第 2 章

特征怎样改变学习行为，Sarsa 又怎样在共享参数下改善策略？

## 本章内容

- 在同一走廊比较状态聚合、局部 tile coding 与分离特征，并理解全局 Fourier 特征的泛化。
- 推导动作分块表示和半梯度 Sarsa，沿一次共享更新追踪动作与下一条真实经验。
- 手算特征缩放的步长效应，并辨别预测保证与控制保证。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [函数逼近预测：从回归到 TD 固定点](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/)：理解一个参数写入怎样同时改变多个状态的预测。
- [TD 预测与控制：SARSA、Expected SARSA、Q-learning 和 Double Q](https://yingwen.io/zh/continual-rl/foundations/tabular/temporal-difference/)：掌握Sarsa与行为策略对后续经验的影响。


### 动作价值

$q(s,a)$ 预测先执行指定动作，之后遵循当前策略的回报。用状态—动作特征替代预测章的状态特征。

$$
\hat q(s,a,w)=x(s,a)^\top w
$$

### 半梯度

更新只对当前预测求导。下一状态的 bootstrap 目标也依赖权重，但在这一步视为固定。

<a id="problem-definition"></a>

## 本章的问题定义

用特征共享动作价值参数，并据此改变行为策略；表示和访问分布会相互影响。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 特征 $x(s,a)$、参数 $w$、动作价值 $q_w(s,a)=x(s,a)^\top w$；探索规则需明确。

### 需要求解的对象

在有限表示中学习可用于选择动作的价值，并获得较好的实际行为收益。

### 信息与数据权限

只观察行为访问的状态动作。未访问区域的估计来自特征共享，而非新增证据。

$$
\sup_{\pi\in\Pi_x}J(\pi),\qquad \Pi_x=\{\pi:\pi\text{由 }q_w\text{和指定选择规则生成}\}
$$

这是受表示与策略提取规则限制的控制目标。半梯度 Sarsa 是求解机制，通常不是这个收益目标的精确梯度。

### 成立条件与解的含义

- 特征尺度、激活数量与动作编码固定，并纳入步长解释。
- 探索策略和特征近似会限制可实现策略。

判断准则：同时检查泛化干扰、访问覆盖和收益；表示误差不能仅由训练 TD 残差判断。

### 适用边界

- 声称一般半梯度控制在任意共享表示下全局收敛。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 限制表示或采用近似 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：本章限制价值表示与策略提取规则；控制收益对象不变，但可实现行为受到限制。

- 组合不同学习问题 · [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)：本章的固定特征控制可接入状态构造模块；若进一步学习特征，表示也会变化，固定特征分析不再直接适用。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

特征相似性可能帮助泛化，也可能把互相冲突的动作结果混在一起。

### 本章的核心思路

先确定特征如何共享，再分析更新怎样沿这些共享方向传播。

1. [用局部特征控制干扰范围](#features-local)：Tile coding 的激活数量决定一次更新的总尺度。

2. [区分表达能力与数值条件](#features-global)：Fourier 特征能表达的函数与优化条件数是两个问题。

3. [将自举误差投到参数方向](#lesson-derive)：Sarsa 的下一动作来自实际策略，之后策略又受更新后的价值影响。

结论与条件：固定策略预测理论不能未经证明变成同时改变策略的控制保证。

### 相关方法改变了什么

- Tile coding / Fourier：局部与全局共享改变泛化和干扰结构。

- 线性特征 / 学习表示：后者还改变特征本身，必须记录表示漂移。


<a id="lesson-setting"></a>

## 1 · 控制中的表示和数据相互影响

预测章暂时固定行为，以便看清同一批经验能学到什么。进入控制以后，一次共享参数更新还可能改变动作排序，让智能体走向不同区域，并由此改变后续训练数据。特征因而同时影响预测和经验获取。我们需要先问哪些状态应当共享一次更新，再问如何依据这种近似价值改善策略；固定数据上的拟合误差无法独自回答第二个问题。

给定状态特征 $\phi(s)\in\mathbb R^d$ 和有限动作数 $K$，一种清楚的起点是动作分块：为每个动作保存独立的 $d$ 个权重。总参数数是 $Kd$。这避免不同动作仅因共享同一特征权重而被强制具有相同价值。

$$
x(s,a)=e_a\otimes\phi(s),\qquad \hat q(s,a)=w_a^\top\phi(s).
$$

$e_a$ 是动作 one-hot 向量。状态间仍共享参数，动作间则由不同参数块区分。神经网络可以选择共享主干，但那引入了另一种跨动作泛化。

<a id="features-corridor"></a>

## 走廊中的选择 · 预测联动怎样改变下一份经验

现在只给前章走廊的 A 增加一个“退出”动作：立即得 0 并终止；“继续”仍按 A 到 B 得 +2、B 到终点得 −1。取 $\gamma=1$，两步以内真正终止，继续的真实动作价值为 1，退出为 0。为继续动作使用前章那一组共享特征；退出动作采用独立参数块，初值也取 0。B 只有继续这一个可用动作。这就把状态间共享与动作间区分放进了同一个小任务。

读者知道转移表，可以枚举奖励序列 [2,−1] 得到继续的真值 1。学习器每步只收到当前位置、所选动作、奖励、后继状态和终止标志；它有固定特征图与自己维护的权重，没有真值标签，也不查询未执行动作的后果。退出估计从零开始只是初始化；在这条任务里，它恰好等于分析真值。

先用一条完整的继续轨迹，按 A、B 顺序回归继续动作的回报。A 更新后，其继续估计为 0.5；B 更新后，这个未再次访问的估计变成 −0.06。退出的估计仍为 0。因此下一回合再到 A 时，贪心选择会从继续变成退出，尽管继续的真实回报较高。这里使用 MC 是为了单独看清共享的影响；后面的 Sarsa 会进一步让目标随行为而变。

如果此后总按贪心退出，学习器便不再获得 B 的新经验；保留探索会改变这个后果。分动作参数块避免了不同动作被强制同值，却没有消除同一动作跨状态的干扰。由此，特征设计既决定一次更新扩散到哪里，也会通过下一次选择改变此后实际看到的数据。

可以先保持这条轨迹与步长不变，只把 A、B 换成互不重叠的单位特征。继续动作在 A 的估计就不会被 B 的回归改写。下面的 tile coding 和 Fourier 基，是在更大状态空间中选择这种共享关系的两种办法。

<a id="features-local"></a>

## 2 · Tile coding：局部相似性如何进入更新

Tile coding 使用多套错开的网格。一个状态在每套网格激活一个 tile。每个 tile 对应一个二值特征，预测等于所有激活权重的和。邻近状态共享部分 tile，因此一次更新会改变局部邻居的预测；跨越边界时变化不是同时发生的。

$$
x_{k,j}(s)=\mathbf1\{j=\lfloor s/h+k/K\rfloor\},\quad k=0,\ldots,K-1.
$$

这是教学代码的一维版本。$h$ 是格宽，$K$ 是 tiling 数；特征键包含 tiling 编号，避免把不同网格中同编号的格子合并。

若没有哈希碰撞，每次恰有 $K$ 个激活特征，单次半梯度更新让当前预测改变 $\alpha K\delta$。因此常用 $\alpha=\eta/K$，让有效更新比例接近 $\eta$。这来自特征范数，不是一条适用于所有表示的步长定律。

![两套错位网格中共享活动 tile 的数量，决定一次更新对邻居预测的影响](https://yingwen.io/crl-figures/concept-depth-classic-tiles.svg)

着色格是状态 0.25 激活的两个特征，每个权重增加 0.1。下方四个位置分别共享 1、2、1、0 个特征，因此预测增加 0.1、0.2、0.1、0。图中没有哈希碰撞。原创精确算例，依据 [Sutton 与 Barto §9.5.4](http://incompleteideas.net/book/the-book-2nd.html) 的 tile coding 定义；[计算代码](/crl-code/figures/classic-visual-depth.mjs)。

多维观测可以联合划分或分组划分。联合网格表达变量交互，但组合数增长很快；分组特征省内存，却可能无法表示关键交互。哈希限制参数容量，但碰撞会把无关状态绑定在一起。记录容量、激活数和碰撞策略，才能解释性能差异。

<a id="rlss-feature-geometry"></a>

## Tile 的宽度、组合方式与哈希分别改变什么

一个 tile 规定哪些输入共用一项参数。宽度规定单项特征覆盖多远；多套错开的 tilings 规定哪些细小差异仍可被区分。这是两个不同的设计自由度。宽 receptive field 可以先形成粗略预测，再由重叠特征的不同组合表达较细边界；它不保证任意目标都能精确表示。

$$
\Delta\widehat v(u)=\alpha\delta\,x(u)^\top x(s),\qquad \frac{\Delta\widehat v(u)}{\Delta\widehat v(s)}=\frac{\#\{\text{shared active tiles}\}}{K}.
$$

这里固定二值特征、无哈希碰撞，每个输入激活 K 个 tile，且当前预测变化非零。比值精确描述一次更新的相对泛化；它不是测试集准确率。

条带与联合 tile 有更实质的区别。仅分别对两个坐标做条带编码，线性读出具有 f(x,y)=a(x)+b(y) 的加法结构。它会把一次更新传播到整行和整列。联合 tile 则可以表示“这个 x 范围并且这个 y 范围”的交互作用。

$$
f(0,0)+f(1,1)=f(0,1)+f(1,0)\quad\text{for}\quad f(x,y)=a(x)+b(y).
$$

这是加法表示必有的恒等式。XOR 目标在对角取 0、非对角取 1，不满足该式。再多采样、再小步长都不能让纯条带表示精确拟合它；联合特征能解除这一限制。

| 输入处理或表示 | 改变的几何 | 需要明确的代价 |
| --- | --- | --- |
| 分别缩放坐标后送入单位 tile | 每个原始坐标上的分辨率 | 传感器单位与在线尺度变化 |
| 对正数取 log 后编码 | 等比例变化相似，而非等绝对差相似 | 零、负数及截断范围的处理 |
| 对 x−y 或 x+y 编码 | 沿斜向条带泛化 | 人为加入了特定不变性 |
| 增加错位 tilings | 更细的组合分辨率与重叠 | 每步处理 K 项，通常不是单 tiling 的同等成本 |
| 固定容量哈希 | 把未访问区域的预分配改为有限存储 | 无关 tile 可能意外共享权重 |

原作者 tiles3 的索引表先为新坐标分配独立槽位。表满之后，新的坐标才回退到哈希。表满访问次数不是不同碰撞对的计数。若同一输入的多个 tile 落入同一槽位，还必须选择一种明确语义：把它们累加为计数特征，或先去重为二值特征。预测、梯度与步长归一化必须采用同一种语义。

碰撞后的单样本尺度检查；这不是完整 tiles3，也不是收敛或性能实验。

```python
from collections import Counter

def counted_tile_update(weights, active_ids, target, fraction=0.1):
    # 计数语义：重复索引表示该槽位的特征值大于 1。
    counts = Counter(active_ids)
    prediction = sum(weights[j] * count for j, count in counts.items())
    norm2 = sum(count * count for count in counts.values())
    if norm2 == 0:
        return prediction
    error = target - prediction
    for j, count in counts.items():
        weights[j] += fraction * error * count / norm2
    return prediction  # 返回更新前预测，供 prequential 评价

w = [0.0, 0.0]
counted_tile_update(w, [0, 0, 1], target=1.0, fraction=0.5)
assert abs(2*w[0] + w[1] - 0.5) < 1e-12
```

此例的特征是 (2,1)，平方范数为 5，而不是原始 tile 个数 3。按 3 归一化会过度改变当前预测。NumPy 的重复高级索引更新还可能不累加重复项；需用明确的计数表示或 scatter-add。哈希节省存储，不会免费消除高维交互的学习难度，也不会保证原来的局部泛化结构。

<a id="rlss-receptive-fields"></a>

## 从错位网格到 RBF：平滑不等于更合适的泛化

二维网格若在两个坐标上采用完全相同的偏移，各套边界会保留相同的方向关系。偏移设计因此不只是实现细节。常见做法令第 k 套 tiling 沿各维按不同奇数倍偏移，并按格宽取模。例如 8 套 tilings 的二维偏移为 k(1,3)/8，而不是 k(1,1)/8。

$$
o_{k,j}=\operatorname{frac}\!\left(\frac{k(2j+1)}K\right),\qquad K(s,u)=\sum_{k=0}^{K-1}\mathbf1\{\operatorname{tile}_k(s)=\operatorname{tile}_k(u)\}.
$$

j 从 0 开始。这只给出一种偏移构造；最终应检查重叠核 K(s,u)，而不是仅看偏移点图。有限套轴对齐网格不保证旋转不变，也不保证任何维数下都是最优偏移。

如果希望特征响应平滑，可以把二值 receptive field 换成径向基函数。仍然保持对权重线性的读出。问题随之转为：中心放在哪里、尺度如何选、需要多远的响应，以及是否允许截断小响应来节省计算。

$$
x_j(s)=\exp\!\left[-\tfrac12(s-c_j)^\top\Sigma_j^{-1}(s-c_j)\right],\qquad \widehat v(s)=\sum_jw_jx_j(s).
$$

c 是中心，正定矩阵 Σ 指定各方向宽度与相关结构。它对输入可以高度非线性，参数梯度仍为 x(s)。如果中心与宽度也学习，便不再是固定特征下的线性学习问题。

$$
\Delta\widehat v(u)=\alpha\delta\sum_jx_j(u)x_j(s),\qquad \Delta\widehat v(s)=\alpha\delta\sum_jx_j(s)^2.
$$

RBF 的响应随位置变化，平方范数一般不恒定。高斯响应数学上在远处也非零；若逐项计算，每步成本通常随全部中心数增长。截断可获得局部计算，但它改变了特征定义。

一维例子取一个中心 0、宽度 σ。距离 σ 的输入响应为 exp(−1/2)≈0.6065。只在中心训练一次，那个输入的预测就改变中心改变量的约 60.7%。若真实目标在二者之间存在突变，平滑会造成干扰；若目标确实平滑，早期传播可能帮助学习。这是环境与表示的匹配，不是“平滑特征总比二值特征好”。

d 个坐标各放 b 个位置，完整联合中心或 tile 组合的数量按 bᵈ 增长。分组表示降低成本却限制交互，稀疏访问节省更新却不补足未见区域，哈希限制内存却引入混叠。比较方法应同时给出输入尺度、总容量、激活数、每步耗时与误差，而不只给出参数量。持续表示学习要自动解决的正是这些选择及其随经验变化的问题。

<a id="rlss-normalization-objective"></a>

## 每一步都除以特征范数，是否还是同一个学习目标

“一次消去固定比例的当前误差”和“最小化原来的平均平方误差”并不总是一回事。先考虑固定标签分布下的线性监督预测。把步长设为依赖输入的正数，会改变不同输入的相对更新权重。

$$
\Delta w=\alpha_0 a(X)[Y-w^\top X]X\quad\Longrightarrow\quad J_a(w)=\frac12\mathbb E\!\left[a(X)(Y-w^\top X)^2\right].
$$

a(X) 固定且不依赖权重时，期望更新是这个加权目标的负梯度。若 a(X)=1/‖X‖²，一步预测变化确实是 α₀ 倍样本误差，但目标一般被重新加权了。零特征须另外处理。

手算反例：两类样本等概率出现，分别为 (x,y)=(1,0) 和 (2,1)，只学习一个权重。普通均方误差最优权重为 E[xy]/E[x²]=1/2.5=0.4。逐样本除以 x² 后，最优权重变为 E[y/x]=0.25。两种方法都能稳定地学习，却在逼近冲突中作出了不同取舍。

$$
\alpha=\frac{\alpha_0}{\mathbb E\|X\|^2}\quad\text{versus}\quad \alpha_t=\frac{\alpha_0}{\|X_t\|^2}.
$$

左式使用冻结的总体尺度，只整体缩放期望梯度；右式通常改变各样本的权重。无碰撞 tile coding 的激活数恒为 K，两个分母相同，因而没有这项差别。

所以不能把逐样本归一化简单说成“不允许”。它可以是有意选择的 normalized LMS 或输出步长控制。必须写明它保留了什么、改变了什么。持续任务中估计尺度也在变化，目标还可能随时间移动。进入 TD 后还多了 bootstrap 耦合，原先的半梯度本来就不必对应一个固定标量损失。

为什么不一次消去全部误差？标签包含随机噪声；相关状态共用参数；自举目标也可能尚未正确。当前样本完全拟合，不等于下一样本误差更小。研究逐样本稳定机制时，应同时报告更新前预测误差、状态加权、预测变化量和长期收益，而不只报告没有发生数值爆炸。

<a id="features-map-control"></a>

### 2.1 · 只换特征图，保留同一走廊

状态聚合把同组状态编码为同一个 one-hot 分量，组内预测必然相同；一套 tiling 就是这种表示。两套错开的 tiling 则允许“共享一部分，但仍能区分”。给走廊位置一个固定特征坐标 $z(A)=0.30,z(B)=0.60$，格宽 $h=0.5$。用上面的键规则，A 激活 $(0,0),(1,1)$，B 激活 $(0,1),(1,1)$：只有后一键共享。坐标用于生成特征，走廊仍只有 A、B 和终点，并未增加中间的可访问状态。

为了只比较共享关系，把三种表示都放进三个坐标，补零坐标不增加有效信息；每个动作保存独立的三个权重。将两套 tile 的二值向量除以 $\sqrt2$，使它与聚合、分离特征同为单位范数。于是可以使用同一零初值与同一步长 $\alpha=0.6$，而不把激活数差异误当成共享关系的效果：

| 表示 | $\phi(A)$ | $\phi(B)$ | 共享内积 $g$ |
| --- | --- | --- | --- |
| 分离特征 | $(1,0,0)$ | $(0,1,0)$ | 0 |
| 两套 normalized tile | $(1/\sqrt2,0,1/\sqrt2)$ | $(0,1/\sqrt2,1/\sqrt2)$ | $1/2$ |
| 粗状态聚合 | $(1,0,0)$ | $(1,0,0)$ | 1 |

看图前先判断：B 的一个负误差会不会写到 A？三种特征的当前状态范数都相同，区别在跨状态的内积。退出在另一动作块，继续动作的写入不改它。

![相同走廊的转移奖励与解析真值；A和B在两套错位tiling中共享一个tile；三个单位范数特征图的活动分量和内积0、1/2、1。](https://yingwen.io/crl-figures/feature-control-walkthrough-features.svg)

上方是读者的完整任务，真值仅供分析；下方条带是特征的激活区域，方块是实际向量分量。两套 tile 的键含网格编号。原创精确算例，依据 Sutton 与 Barto §§9.3、9.5.3–4；[计算核](/crl-code/figures/feature-control-walkthrough.mjs)与[逐事件数据](/crl-figures/feature-control-walkthrough-data.json)给出相同定义。

原始二值 tile 仍可直接用于实现：不除 $\sqrt2$，改用 $\alpha=0.3$，便与归一化表示的 $\alpha=0.6$ 产生相同预测写入。这里的 $1/2$ 是两个归一化向量的内积，非零分量各为 $1/\sqrt2$。聚合无法同时表示 A 的 1 和 B 的 −1；另外两种表示能区分它们，但有限几步更新仍会有误差。

<a id="features-global"></a>

## 3 · Fourier 特征、缩放与条件数

$$
\phi_c(s)=\cos(\pi c^\top\tilde s),\quad \tilde s\in[0,1]^n,\quad c_i\in\{0,\ldots,m\}.
$$

Fourier 基用不同频率表达平滑到快速变化的形状。最高阶 $m$、维数 $n$ 下，完整张量基有 $(m+1)^n$ 项；提高阶数增加表达能力，也增加计算与数据需求。

一维代码在状态 0.5、阶数 3 时得到近似向量 [1,0,-1,0]。数值中的极小非零项来自浮点余弦，不是新的特征。Fourier 特征通常全局非零，因此一次更新会影响较远状态。局部表示与全局表示没有统一优胜者，关键是环境中相似状态是否确实应有相似价值。

$$
\begin{gathered}x'=cx,\qquad w'=w/c,\\ \Delta\hat v=\alpha\delta\|x\|^2,\qquad \alpha'=\alpha/c^2.\end{gathered}
$$

统一放大全部特征并相应缩小权重，初始预测不变；若步长不变，预测更新会放大 $c^2$ 倍。线性 TD 在同时调整步长后才保持对应的预测轨迹。

不同坐标的尺度差异会使特征协方差病态。逐维标准化或逐维步长能改善数值尺度，但改变在线归一化统计量也会改变同一原始状态的特征。这种表示漂移在持续学习中尤其重要：旧权重可能不再表示原来的函数。

$$
\begin{aligned}\Delta w&=\alpha\delta\nabla_w\hat v(s,w),\\\hat v(u,w+\Delta w)-\hat v(u,w)&=\alpha\delta\,\nabla_w\hat v(u,w)^\top\nabla_w\hat v(s,w)+O(\|\Delta w\|^2).\end{aligned}
$$

对另一个状态 u 的预测作一阶展开。两个预测梯度的内积决定这次更新怎样泛化：正值同向改变，负值反向改变，0表示一阶无影响。线性固定特征时余项为0。

例如 x(A)=(1,0)、x(B)=(−1,1)，在 A 得到正误差1并用步长.1更新，A 的预测增加.1，B 的预测却减少.1。没有神经网络、没有离策略，也能发生干扰。神经网络还会让这两个梯度方向随每次更新改变，因此“先把网络当作固定特征提取器”的分析只适用于被冻结的表示阶段。

这给出一个比“神经网络不稳定”更具体的研究问题：当前更新是否沿着会破坏重要旧预测的方向？固定一组探测状态，记录更新前后的预测及梯度内积，再逐项加入 bootstrap、旧数据和表示更新，就能区分静态泛化冲突与不断变化的更新几何。资格迹保存过去的梯度方向时，也会受到这种几何变化影响。

<a id="features-scale-control"></a>

### 3.1 · 同一特征图的尺度也会改变控制

把走廊的 normalized tile 向量全乘 $c=10$。若初始权重同时除以 10，所有初始预测相同；零初值也满足这个对应。内积和范数平方却都乘了 100。同一份经验产生相同的旧误差，未补偿的第一次预测写入因此放大 100 倍。对固定线性特征，采用 $\alpha'=\alpha/c^2$ 才能在每步保持权重 $w'_t=w_t/c$，从而保持预测、动作和后续经验的对应。

先预测第一步的变化：A 继续得到 +2，旧的 A、B 估计均为 0。同名义步长 0.6 下，原尺度的 A 估计变为 1.2，放大后的估计为 120。随后 B 真正终止，目标只有 −1；两种尺度又会收到不同大小的旧误差，第二次写入不能继续用一个固定的 100 倍比例猜测。

![同一tile特征图在原尺度、乘10未改步长、乘10且步长除100的三种条件下，第一步写入值为1.2、120、1.2；B终止更新后A值为0.72、−1710、0.72。](https://yingwen.io/crl-figures/feature-control-walkthrough-scale.svg)

条长按同一动作值轴画第一步写入，数字给出随后 B 的终止更新与再到 A 的选择。三种条件共用任务与零初值；原步长 0.6，补偿后 0.006。原创有限次实际更新，未补偿的巨大值来自特征范数与后续误差的联动；它解释一次控制改变，不构成渐近发散或训练性能的证据。

具体地，放大后第一步还令 B 估计为 60，所以 B 的误差为 $-1-60=-61$。A 与 B 的内积为 50，A 接着变为 $120+0.6\times(-61)\times50=-1710$，再到 A 就会退出。补偿后整个两回合事件序列与原尺度相同。这个标量补偿要求统一缩放、固定线性特征及同样的动作选择规则；各坐标放大不同倍数时，一般需要相应的逐坐标步长或预条件，不能只除一个公共常数。

<a id="lesson-derive"></a>

## 4 · 从策略评价到半梯度 Sarsa

$$
\begin{aligned}\delta_t&=R_{t+1}+\gamma\hat q(S_{t+1},A_{t+1},w_t)-\hat q(S_t,A_t,w_t),\\ w_{t+1}&=w_t+\alpha\delta_t x(S_t,A_t).\end{aligned}
$$

下一动作 $A_{t+1}$ 来自实际行为策略。因此目标在评价该行为，包括它的探索动作。真正终止时 bootstrap 项为零。

常用行为是 $\epsilon$-greedy：以 $1-\epsilon$ 的概率在最大价值动作间均匀选择，以 $\epsilon$ 的概率在全部动作间均匀选择。最大值并列时如何处理也是算法定义的一部分。先按旧权重采样下一动作，再更新，并在下一次交互真正执行这个动作，就得到一致的 Sarsa 时序。

**算法：半梯度 Sarsa 的采样与更新**

1. 初始化动作价值权重，观察状态并按当前策略采样动作
1. 执行动作，观察奖励、下一状态与是否真正终止
1. 若未终止，按更新前的策略采样下一动作并保存其价值
1. 计算旧参数下的 TD 误差，更新当前状态—动作特征的权重
1. 把已采样的下一动作作为下一步实际动作；终止则开始新回合

Expected Sarsa 将下一动作样本替换为当前策略下的价值期望，降低这部分采样方差。将其替换为最大动作价值则得到 Q-learning 风格目标，目标策略与探索行为分离。共享参数时，这个改变引入离策略稳定性问题，不能仅视为低方差替换。

<a id="experiment-semi_gradient_sarsa"></a>

### 实验：实验 · 共享特征怎样把一次 Sarsa 更新扩散到其他状态？

只有六个权重的动作价值近似器，能否在六格链上形成正确动作排序？

**环境与可用信息。** 六格链从 0 出发，进入终点奖励 1，其余 −0.01，γ=0.95。每个动作有三个共享多项式特征：常数、归一化位置和位置平方。两个动作使用互不重叠的参数块，共六个零初始化权重。

**设置。** 五种子各 1200 个环境转移。训练采用 ε=0.1 的行为策略，Sarsa 步长 0.05。先按旧参数选后继动作，再用该动作价值自举；真实终点用零后继特征。图中基线为十项表格 first-visit MC control，整回合后用样本平均更新。

**检验的机制。** Sarsa 只对当前动作的特征块求导，但该块在多个位置复用，因此一次更新会改变多个状态的动作价值。策略改善需要动作排序正确，不要求所有价值数值都精确。此处还同时改变了表示、步长和更新时间。

**测量。** 主图冻结当前参数，执行确定性贪心策略，精确计算原奖励折扣回报；遇到循环时求无限尾和。它不是训练期 ε-greedy 的回报，也不展示每个状态动作的估值误差。

```bash
python3 implementations/extended_classic/semi_gradient_sarsa.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/semi_gradient_sarsa/curves.svg)

横轴：environment_steps。纵轴：贪心策略精确折扣回报。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 第 60 步，五条 Sarsa 曲线均达到最短路径回报 0.7774，MC 对照均值为 0.1910；第 600 步两者都达到 0.7774。后续主图饱和，无法判断价值预测是否还在变化。

**结论边界。** 该比较不是同表示消融，不能把所有早期差异归功于自举或共享特征。平稳短链也没有检验函数逼近控制的稳定性，更不能代替持续非平稳控制评价。

**继续实验。** 增加表格 Sarsa 与相同多项式表示的 Gradient MC control，构成表示和更新目标的二维对照。另画十个 Q 估值及实际 ε-greedy 回报，检查贪心成功是否掩盖估计误差或探索成本。

[源码](https://yingwen.io/crl-code/implementations/extended_classic/semi_gradient_sarsa.py) · [逐种子记录](https://yingwen.io/crl-code/results/semi_gradient_sarsa/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/semi_gradient_sarsa/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/semi_gradient_sarsa/curves.json)

<a id="features-sarsa-corridor"></a>

### 4.1 · 在走廊里执行更新，再看它选出的经验

回到三种单位范数特征图。这一次不等待回报，直接在线运行半梯度 Sarsa。初始权重全零，$\gamma=1,\alpha=0.6$；为得到可逐项检查的有限过程，取 $\epsilon=0$，A 的并列估计固定选继续。B 只有继续，因此第一回合三者都会实际经过 A→B→终点。一般探索策略的作用稍后单独计算。

第一步在 A 执行继续，收到 +2 到达 B。先选定 B 的继续，保存其旧值 0，才计算 $\delta_0=2+0-0=2$。写入 $\Delta w_0=1.2\phi(A)$ 后，A 的继续值为 1.2，B 的估计为 $1.2g$，其中 $g=\phi(A)^\top\phi(B)$。B 的 −1 此时还没有收到，第一步目标也没有使用分析真值 1。

执行已选定的 B 继续后，收到 −1 并真正终止。无需选择终点动作，bootstrap 为零；用更新后的旧权重算 $\delta_1=-1-1.2g$。这一步发生在 B，但它改变了暂未再次访问的 A：

$$
\begin{aligned}\hat q^+(A,\text{继续})&=1.2+0.6(-1-1.2g)g,\\ \hat q^+(B,\text{继续})&=1.2g+0.6(-1-1.2g).\end{aligned}
$$

先在 A 收到一条转移并写入，再在 B 收到下一条转移并写入；这是两个不同参数版本。A 的连带变化来自内积，B 的本地变化来自单位范数。退出动作块仍为 0。

| 表示 | 首回合后 A 继续值 | 首回合后 B 值 | 下一次在 A 执行动作 | 随即实际收到 |
| --- | --- | --- | --- | --- |
| 分离特征 | 1.2 | −0.6 | 继续 | +2，到 B |
| 两套 tile | 0.72 | −0.36 | 继续 | +2，到 B |
| 粗状态聚合 | −0.12 | −0.12 | 退出 | 0，终止 |

看图前先判断：如果只给三者重放相同的第二条走廊轨迹，会丢掉哪一个差别？下一回合要先让各自的估计选动作，再调用环境取得后果，才能看到粗聚合已经不去 B 的事实。

![三个表示的A继续估计在B终止写入前后分别为1.2到1.2、0.72、−0.12；下一回合实际动作分成继续到B得2与退出终止得0。](https://yingwen.io/crl-figures/feature-control-walkthrough-control.svg)

空圆是 A 第一次写入后，实圆是 B 终止写入后；三行采用同一价值轴。右侧或下方箭头是下一回合实际执行的动作及真实奖励，非预定重放轨迹。原创确定性有限次 Sarsa，首回合两次更新与下一回合真实事件均由[计算核](/crl-code/figures/feature-control-walkthrough.mjs)产生；[标准库教程](/crl-code/tutorials/feature-control-walkthrough.py)以独立的精确 Gram 空间递推逐事件复算。

这个过程把控制中的信息次序接起来：特征图决定写入范围，B 的反馈改变 A 的未重访估计，A 的新排序决定动作，动作决定下一条经验。粗聚合的表示冲突与零探索共同让 B 暂时失去覆盖。这里的有限序列没有建立哪一种表示在一般控制任务中的性能排序。

若改为 $\epsilon=0.2$ 的行为，保持上述首回合后的估计，粗聚合在 A 选继续的概率为 0.1，其余两种为 0.9。因此下一步奖励的条件期望分别是 0.2 与 1.8；这是对下一次动作的精确平均，还没有对重新更新后的整段学习求期望。探索使退出者仍有机会访问 B，但不会让聚合特征突然能够区分两个状态。

<a id="lesson-example"></a>

## 5 · 手算一个动作块的更新

两个动作，各有特征 [1,s]。当前状态 $s=0.5$，当前动作 0；下一状态 $s'=1$，已采样下一动作 1。权重为 $w=(0.2,0.4,0.1,0.2)$，奖励为 1，$\gamma=0.9,\alpha=0.1$。

$$
\begin{gathered}x=(1,0.5,0,0),\qquad x'=(0,0,1,1),\\ q=0.4,\qquad q'=0.3,\\ \delta=1+0.9(0.3)-0.4=0.87.\end{gathered}
$$

当前和下一价值都在更新前计算。

$$
w^+=(0.287,0.4435,0.1,0.2).
$$

只有动作 0 的参数块被更新，但该动作在其他状态的价值也会改变。动作 1 的本次预测不变。

如果这一步真正终止，目标应该是 1，误差变成 0.6。若只是为了日志而切开一段持续交互，环境仍会继续，则不应自动清除 bootstrap。算法边界来自任务语义，不来自数组或日志文件的边界。

<a id="lesson-code"></a>

## 6 · 表示与 Sarsa 的可运行代码

无哈希的一维 tile coding、Fourier 基、动作分块与 Sarsa

```python
def tiles_1d(state, width=0.25, tilings=4):
    """Unhashed sparse keys: one active interval in each offset tiling."""
    if width <= 0 or tilings < 1 or not math.isfinite(state):
        raise ValueError("finite state, positive width and tilings required")
    return tuple((k, math.floor(state / width + k / tilings))
                 for k in range(tilings))


def fourier_1d(state, order=3):
    if not 0 <= state <= 1 or order < 0:
        raise ValueError("normalize state to [0, 1], use nonnegative order")
    return [math.cos(math.pi * k * state) for k in range(order + 1)]


def action_features(features, action, actions):
    if not 0 <= action < actions:
        raise ValueError("action outside action set")
    return [v if a == action else 0.0 for a in range(actions) for v in features]


def semi_gradient_sarsa(weights, x, reward, xp, gamma, alpha, terminated=False):
    # x and xp are state-action features. xp belongs to the already sampled A'.
    if terminated:
        xp = [0.0] * len(weights)
    return semi_gradient_td(weights, x, reward, xp, gamma, alpha)


def features_demo():
    x = action_features([1.0, 0.5], 0, 2)
    xp = action_features([1.0, 1.0], 1, 2)
    weights, delta = semi_gradient_sarsa([0.2, 0.4, 0.1, 0.2], x, 1, xp, 0.9, 0.1)
    return {"tiles_at_0.2": tiles_1d(0.2), "tiles_at_0.21": tiles_1d(0.21),
            "Fourier_at_0.5": fourier_1d(0.5), "delta": delta,
            "updated_weights": weights}
```

教学版 tiles 返回稀疏键，不分配无限长向量。实际实现可以使用稀疏字典或固定容量哈希表。代码不替你选择传感器范围；Fourier 函数要求输入已在 [0,1]，越界时明确报错。学习器使用未来统计量做归一化会泄露信息，因此部署版应规定训练或在线统计的更新时序。

运行特征与控制的手算例子

```sh
python3 approximation_textbook_lab.py features-control
python3 approximation_textbook_lab.py test
```

走廊的[独立单文件教程](/crl-code/tutorials/feature-control-walkthrough.py)仅需 Python 3 标准库。下载并保存后，在文件所在目录运行下列命令；无需网站源码、安装包或训练数据。正常输出是 JSON，按事件读 state、action、reward、next，以及更新前后的 A/B 估计，特别比较三个 next_real_transition。

两回合真实选择与十项独立精确检查

```bash
python3 feature-control-walkthrough.py
python3 feature-control-walkthrough.py --test
```

网站计算核直接写浮点特征向量与权重；Python 教程用 $\Delta\hat q(u,a)=\alpha\delta\,K_{us}$ 在预测空间更新，以活动键的交集除以 tiling 数精确得到 Gram 内积。它还枚举完整奖励路径核对真值，检查终止目标、尺度补偿和原始二值 tile 的等效步长。这两种计算方式核对机制语义，不用这十项检查评价控制性能。

<a id="lesson-branches"></a>

## 7 · 控制保证、表示实验与 CRL

线性在策略预测的收敛结论不能直接证明线性 Sarsa 控制全局收敛。原因是策略和状态分布随参数改变，近似误差还可能改变贪心动作。表格 GLIE 条件下的结论也不自动适用于共享参数。实际研究应报告行为稳定性，而不只检查预测损失。

比较特征时固定交互预算、探索策略和超参搜索预算。记录激活特征数、总参数数、每步计算量以及对未训练区域的预测变化。固定参数数量并不等于固定更新尺度；固定名义步长也不等于公平。

进入持续学习后，可以分别让奖励变化、访问区域变化和传感器尺度变化。这样才能判断失效来自旧表示不充分、归一化漂移，还是控制回路把智能体带到低覆盖区域。Agent state、流式归一化和可塑性方法分别处理其中不同部分。

<a id="rlss-online-representation-search"></a>

## 从固定特征到在线表示搜索：怎样生成、测试并替换一个特征

固定特征的线性学习器不能表达任意非线性关系，但“对参数线性”不要求“对原始输入线性”。先由非线性单元展开输入，再只学习线性读出，就能把表示构造和预测更新分开研究。Online Representation Search 的课堂例子采用在线监督回归；它不是已经解决神经 TD 稳定性的算法。

$$
z_i(x)=\mathbf 1\{v_i^\top x\geq\vartheta_i\},\quad
v_{ij}\in\{-1,+1\},\quad x_j\in\{0,1\},\quad \widehat y=w^\top z+b.
$$

每个线性阈值单元 LTU 检查一类输入模式。符号向量可随机生成，读出仍然线性；增加 LTU 是改变函数类，而不只是把原有优化器运行更久。

$$
S_i=\#\{j:v_{ij}=-1\},\qquad
\#\text{匹配}=v_i^\top x+S_i,\qquad
\vartheta_i=\beta d-S_i.
$$

共有 d 个二值输入。正权重对应输入一，负权重对应输入零。逐项计数得到匹配公式；阈值因此表示至少 β 比例的位匹配。是否包含 bias 位必须统一计入 d。

$$
\widetilde z=(1,z^\top)^\top,\qquad
\delta=y-\theta^\top\widetilde z,\qquad
\theta^+=\theta+\frac{\alpha\delta\widetilde z}{\varepsilon+\|\widetilde z\|^2}.
$$

归一化 LMS 把当前特征范数显式放入步长。当 ε 为零且范数非零时，当前样本的预测恰改变 αδ；这并不保证其他输入的误差下降。

| 过程 | 最小定义 | 须检查的问题 |
| --- | --- | --- |
| 生成 | 抽取输入权重与阈值，建立候选 LTU | 激活是否过稀或过密？输入尺度是否改变？ |
| 学习 | 逐样本更新读出；可另学逐特征步长 | 特征数量增加是否改变总更新幅度？ |
| 测试 | 在成熟候选中按既定效用排序 | 效用是当前贡献、学习潜力，还是历史重要性？ |
| 替换 | 删除低效用特征，重置新特征的年龄与读出 | 旧预测改变多少？相关迹和元统计是否仍有意义？ |

最简单的测试器按 $|w_i|$ 排序。这依赖固定的特征单位：令 $z_i'=c z_i,w_i'=w_i/c$，预测完全不变，而权重排序可能反转。二值 LTU 固定了幅度，却没有消除激活频率、冗余和未来任务的重要性。因此贡献统计、权重变化迹与元步长是不同的候选测试器，不能统称为已知的“真实效用”。

特征对当前表现有用，与它能加快未来学习，是两种性质。一个旧特征可能暂时没有输出贡献，却能在再次遇到熟悉情形时快速恢复；另一个特征可能解释眼前标签，却使后续目标的学习更困难。MBRL2 的开篇把这两种用途分开，并提出将有用性沿下游预测与子问题传播。它没有给出已经验证的通用效用函数。

| 当前表现 | 未来学习 | 保留或替换前需要的证据 |
| --- | --- | --- |
| 有用 | 有用 | 可作为核心，但仍需检查变化后的有效性 |
| 有用 | 无明显帮助 | 短期应保护；不能由此宣称可迁移 |
| 当前无用 | 有用 | 仅按当前贡献删除可能损失学习准备 |
| 当前无用 | 未见帮助 | 在限定候选集与测试时长内优先替换；不等于证明永远无用 |

成熟期也是资源约束。若每个候选至少保留 m 步，长期平均每步可替换的比例不可能任意大。实现需要同时记录请求替换率、实际替换数与合格候选数。每步把 $\rho N$ 直接取整，会在 $\rho N<1$ 时永不替换；应保留小数预算，或明确采用随机替换协议。

核对特征定义；运行结果不构成表示搜索性能证据。

```python
from itertools import product
v = (1, -1, 1, -1)
S = sum(a < 0 for a in v)
for x in product((0, 1), repeat=len(v)):
    matches = sum(bit == int(a > 0) for bit, a in zip(x, v))
    assert matches == sum(a*bit for a, bit in zip(v, x)) + S
print("LTU matching identity checked for every binary input")
```

与持续 RL 的衔接不只是把监督标签换成 TD target。新特征会改变价值、GVF、模型和策略的共享坐标；它的效用还受自举标签和行动分布影响。先冻结目标测试表示搜索，再逐项放开自举、控制和多预测共享，才能定位失败来自生成器、测试器还是学习回路。

<a id="lesson-check"></a>

## 8 · 练习与答案

- 8 个二值 tile 同时激活，想使一次更新的当前预测变化为 $0.2\delta$，步长取多少？答案：无碰撞时取 $0.2/8=0.025$。
- Fourier 阶数增加，能否保证控制收益提高？答案：不能。更高表达能力也增加估计难度，且价值误差与行为收益不同。
- 将所有特征乘 10，只把权重除 10，为什么仍可能发散？答案：同名义步长下，预测变化会放大 100 倍；还应将步长除以 100。
- Sarsa 的下一动作采样后又重新采样执行，会有什么问题？答案：更新评价的动作与实际后继动作不再是同一条 Sarsa 转移；必须明确改用其他估计器。
- 把走廊首回合后粗聚合的退出初值改成 −0.2，会怎样？答案：继续的 −0.12 更高，下一次会继续到 B 得 +2；动作改变需要比较两个估计，单看继续值为负并不够。
- 只看首回合后的 A/B 误差，能断言哪种表示终身收益最高吗？答案：不能。下一次选择已改变访问分布；需另定交互预算、探索与评价口径，真正运行闭环。

## 从本章进入实践

[预测与控制](https://yingwen.io/zh/continual-rl/code/#practice-prediction)：学会预测更多事情，什么时候会改变行动？



<a id="chapter-code"></a>

## 下载与运行

标准库教学实现。命令运行本章的可解析小问题和全文件的公式测试；不依赖深度学习库，不等同于论文中的大规模实验。

[下载 approximation_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/approximation_textbook_lab.py)

```sh
python3 approximation_textbook_lab.py features-control
python3 approximation_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：Part II 第 9–13 章。原书建立函数逼近预测、控制、离策略稳定性、资格迹与策略梯度的共同框架。

- [Sutton · Tile Coding Software](http://incompleteideas.net/tiles/tiles3.html)：原作者的 tile coding 使用说明和代码入口。实际使用时需记录哈希容量、tiling 数与缩放。

- [Konidaris, Osentoski & Thomas · Value Function Approximation in RL Using the Fourier Basis](https://people.cs.umass.edu/~pthomas/papers/Konidaris2011a.pdf)：Fourier 基的原始论文；解释频率表示与强化学习中的使用。

- [Mahmood & Sutton · Online Representation Search and Its Interactions with Unsupervised Learning](https://www.eng.uwaterloo.ca/~jbergstr/files/nips_dl_2012/Paper%2019.pdf)：在线监督表示搜索；生成器和测试器的原始设置，不等同于深度 TD 稳定性结论。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-approximation-features-control#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-approximation-features-control#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-approximation-features-control)
<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 当前输入保留了哪些历史信息？

Bellman 方程先假定有足够的状态。表格为不同状态分别存值；它不负责从相同观察中恢复被遗漏的历史。

函数逼近与深度方法：特征共享是在已知信息上泛化；递归状态则保留过去信息。增加网络宽度不等于补回历史，低训练误差也不证明输入满足 Markov 性。

持续学习中的研究问题：策略改变以后，原来的状态压缩是否仍能预测行动后果？构造状态的网络、运行时记忆、资格迹与优化器状态如何共同更新？

[MDP 的状态条件](https://yingwen.io/zh/continual-rl/foundations/tabular/mdps/) → [表示与泛化](https://yingwen.io/zh/continual-rl/foundations/approximation/features-control/) → [不完全可观测](https://yingwen.io/zh/continual-rl/foundations/deep/partial-observability/) → [智能体状态](https://yingwen.io/zh/continual-rl/construction/state/)


### 可进一步检验的问题

- [04 · 旧价值什么时候应该复用，什么时候应该快速改写？](https://yingwen.io/zh/continual-rl/research/#research-continual-control)：走廊中的共享特征让一次价值更新改变下一次选择，说明旧价值的复用还会改变随后得到的数据。
- [12 · 怎样保留旧能力，而不把过时知识强加给新任务？](https://yingwen.io/zh/continual-rl/research/#research-retention-transfer)：共享特征使新经验同时改变未访问状态的预测，提供了检查旧能力受干扰以及何时可复用的最小情形。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)
- [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

对应原始材料：第 9 章：线性方法、特征构造与步长；第 10 章：On-policy Control with Approximation。本文为原创讲解，原书、论文与上游代码保留各自许可。
