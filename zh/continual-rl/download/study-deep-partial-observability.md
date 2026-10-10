# 不完全可观测：信念状态、信息行动与递归记忆

现代深度强化学习 · 并列研究分支

当前观察不能决定未来时，智能体应记住什么，信息又如何影响行动？

## 本章内容

- 从历史条件分布推导 Bayes filter。
- 把信息获取的价值纳入 Bellman 决策。
- 区分精确信念、学习的 recurrent state 和训练时的隐状态权限。
- 逐版本计算 recurrent replay 的状态、TD 标签与截断梯度。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [MDP、回报与价值：序列决策的数学对象](https://yingwen.io/zh/continual-rl/foundations/tabular/mdps/)：区分状态与当前观测，理解Markov条件。
- [函数逼近预测：从回归到 TD 固定点](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/)：理解参数化预测及其误差，再引入递归活动。


### 条件概率与期望

区分随机变量、一次样本与条件分布；可以使用 Bayes 法则和全期望公式。

### MDP 与价值

理解状态、动作、转移、奖励、策略和 Bellman 递推；学习数据可以来自不同策略。

### 优化与梯度

知道固定目标、参数梯度、约束和期望目标的含义；神经网络只是函数表示的一种选择。

<a id="problem-definition"></a>

## 本章的问题定义

当前观测不能确定未来。智能体必须利用历史形成足够的决策信息。

### 给定条件与符号

- 有限潜在状态集 $\mathcal X$、观测集 $\mathcal O$、动作集 $\mathcal A$、奖励域 $\mathcal R\subset\mathbb R$；$X_t,O_t,A_t,R_{t+1}$ 为相应变量。平稳联合核为 $K(x',B_o,B_r\mid x,a)=\Pr(X_{t+1}=x',O_{t+1}\in B_o,R_{t+1}\in B_r\mid X_t=x,A_t=a)$，其中 $B_o,B_r$ 为可测观测与奖励事件；离散情形可写为 $K(x',o,r\mid x,a)$。
- 历史 $H_t=(O_0,A_0,R_1,O_1,\ldots,A_{t-1},R_t,O_t)$；给定初始观测律 $\nu_0$ 与条件初始信念 $b_0(x\mid O_0)$，它们共同指定 $(X_0,O_0)$ 的联合分布。奖励有界，$0\le\gamma<1$。
- $\Pi_H$ 是非空的允许可测因果策略类，$\pi_t(\cdot\mid H_t)$ 只能使用已到达的信息。已知 $K,b_0$ 时可维护 $b_t(x)=\Pr(X_t=x\mid H_t)$；未知模型需另给模型族和信息条件，联合估计模型与状态，或学习近似摘要。

### 需要求解的对象

在 $\Pi_H$ 中求使折扣收益接近上确界的策略，并构造支持它的信息状态；不假定任意受限策略类的最优者必然存在。

### 信息与数据权限

执行时不可访问潜在真状态；精确 filter 的参照问题允许查询正确模型与初始律。未知模型版本不提供这项权限，须声明模型族、先验或识别数据。奖励如含状态信息，也必须纳入历史和后验。

$$
J(\pi)=\mathbb E_{\nu_0,b_0,K,\pi}\!\left[\sum_{t\ge0}\gamma^tR_{t+1}\right],\qquad J(\hat\pi)\ge\sup_{\pi\in\Pi_H}J(\pi)-\varepsilon,\quad\varepsilon>0
$$

这是指定初始化与策略类下的历史条件控制，$A_t\sim\pi_t(\cdot\mid H_t)$。正确已知模型下 belief 对未来预测充分；若策略类另有限制，其信息或资源约束也须保留，不能自动换成任意 belief 策略。

### 成立条件与解的含义

- 潜在 Markov 条件是给定 $X_t,A_t$ 后，下一潜在状态、观测和奖励的联合条件律不再依赖更早历史，并由同一 $K$ 给出。精确 filter 使用正确模型与初始律；离散观测的证据概率为正，连续情形使用相应密度或正规条件分布。
- 本章正文的分解式 filter 另假定奖励不提供额外状态信息；否则必须将奖励纳入联合似然。未知模型可在指定先验下维护状态与模型参数的联合后验，并非只能使用近似记忆。
- 有限记忆及计算限制应进入 $\Pi_H$ 或完整实现集合；近似摘要的控制损失需另检验。

判断准则：除状态预测误差，还检查不同历史被合并后是否仍能选择正确动作。

### 适用边界

- 以记忆长度或重构准确率直接证明控制充分性。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)：本章将隐藏 Markov 模型的 belief 构造接到历史控制；一般状态构造章已包含部分可观测过程，本章不是再次放宽它的假设。

- 组合不同学习问题 · [探索与经验选择](https://yingwen.io/zh/continual-rl/algorithms/exploration/)：本章的信息状态控制可与探索结合：动作可能通过改善信息而有价值，即使即时奖励较低。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

相同观测可能来自需要不同动作的情境，反应式策略无法区分。

### 本章的核心思路

先用已知模型展示准确后验如何递推，再定位学习记忆的近似位置。

1. [明确更新前后有什么信息](#lesson-notation)：动作先改变潜在状态，新观测和奖励再修正后验。

2. [把历史压缩为 belief](#lesson-derive)：Bayes 递推给出充分状态的参照，而非要求神经网络储存完整历史。

3. [用决策差异评价信息价值](#lesson-information)：更准确识别状态只有改变未来行动或预测时才产生对应用途。

结论与条件：belief 充分性有正确模型条件；近似 recurrent state 必须通过下游任务检验。

### 相关方法改变了什么

- Bayes filter / recurrent network：前者依据指定生成模型精确递推，后者从目标和数据学习摘要。

- 观测重构 / 控制充分性：优化的误差对象不同；重构很好仍可能遗漏决策关键变量。


<a id="lesson-setting"></a>

## 1 · 观察不等于状态

前面的 Bellman 递推以状态为条件。神经网络接收一幅图像，却不等于已经获得这样的状态：同样的位置可以对应向左或向右的速度，眼前看不见的线索也可能决定下一步奖励。增加网络容量只扩大当前输入的函数类；要利用过去信息，还需说明历史怎样进入计算。这使我们回到控制问题的信息条件。

设环境隐状态为 $S_t$，智能体收到 $O_t$。同一幅图像可能来自不同速度、物体遮挡或任务阶段。即使环境隐状态满足 Markov 性，观察序列也未必满足。POMDP 是信息条件；表格 belief、线性滤波器和深度 recurrent 网络都可以处理它。它不是深度 RL 之后才出现的阶段。

$$
H_t=(O_0,A_0,R_1,\ldots,A_{t-1},R_t,O_t),\qquad b_t(s)=\Pr(S_t=s\mid H_t)
$$

历史是已经到达的信息；belief 是给定模型时对当前隐状态的后验分布，不是对网络参数的置信区间。

以下有限状态推导假定模型已知。先讨论奖励没有提供额外状态信息的情形，再用联合转移—奖励—观测模型纳入奖励信息。设计者提供了状态假设、传感器接口和模型结构；运行时智能体更新 belief 并选择行动。

<a id="lesson-notation"></a>

## 2 · 动作之后，先预测再条件化

$$
T_a(s,s')=\Pr(S_{t+1}=s'\mid S_t=s,A_t=a),\qquad O_a(o\mid s')=\Pr(O_{t+1}=o\mid S_{t+1}=s',A_t=a)
$$

转移矩阵每一行和为一。观测似然按收到的观测取值，它不是在状态维度归一化的 posterior。

$$
\bar b_{t+1}(s')=\sum_sT_a(s,s')b_t(s)
$$

用全概率公式将旧状态不确定性传播到下一状态。这一步尚未使用新观察。

$$
\Pr(o\mid b_t,a)=\sum_{s'}O_a(o\mid s')\bar b_{t+1}(s')
$$

这是新观测的证据概率，随后用作归一化常数。

<a id="lesson-derive"></a>

## 3 · Bayes 递推与 belief MDP

$$
b_{t+1}(s')=\frac{O_a(o\mid s')\sum_sT_a(s,s')b_t(s)}{\sum_{\tilde s}O_a(o\mid\tilde s)\sum_sT_a(s,\tilde s)b_t(s)}=:B(b_t,a,o)(s')
$$

分子是下一状态与观测的联合条件概率，分母消去状态得到观测概率。分母为零时模型认为该观测不可能，不能静默归一化。

$$
b_{t+1}(s')=\frac{\sum_s b_t(s)K(s',r,o\mid s,a)}{\sum_{\tilde s,s}b_t(s)K(\tilde s,r,o\mid s,a)}=:B_K(b_t,a,r,o)(s')
$$

奖励携带信息时，K 是后继状态、已收到奖励与观察的联合条件核。这里为便于求和取离散变量；连续变量需用相应密度。这一形式保留奖励、观察与转移之间的相关性。

例如隐藏状态在本步保持不变，普通观察完全相同，奖励却等于隐藏的二元状态。先验各半，收到奖励一后正确后验把全部质量放在状态一；忽略奖励的 filter 仍是各半。奖励既是目标信号，也可能是可用信息，二者不能只保留其一。

给定正确模型与初始分布，belief 对未来控制是充分的：已知当前 belief 和新动作，即可预测后继 belief 及奖励，不必再次读取整段历史。状态集合变成概率单纯形，通常连续，即使原隐状态只有有限个。充分性来自模型和条件分布，不来自把向量命名为 state。

$$
\Pr(S_{t+1}=s',R_{t+1}=r,O_{t+1}=o\mid H_t,a)=\sum_s b_t(s)K(s',r,o\mid s,a)
$$

先对当前隐藏状态用全概率公式求和。两个历史若给出相同 belief，对任何新动作便给出相同的后果分布；相同后果再经 Bayes 更新给出相同后验。逐步重复这两个事实，才说明 belief 保留了未来预测所需的历史信息。这里假设世界模型固定且正确。

$$
\Pr(r,o\mid b,a)=\sum_{s,s'}b(s)K(s',r,o\mid s,a),\qquad \bar r(b,a)=\sum_{r,o}r\Pr(r,o\mid b,a)
$$

联合后果概率既给出即时奖励的条件均值，也给出不同后验 belief 的发生概率。

$$
V^*(b)=\max_a\left[\bar r(b,a)+\gamma\sum_{r,o}\Pr(r,o\mid b,a)V^*(B_K(b,a,r,o))\right]
$$

行动既影响外部状态，也影响未来信息。奖励携带的信息必须进入后继 belief，不能只把奖励换成均值。若给定动作与观察后奖励没有额外状态信息，$B_K$ 不再依赖 r，才可把后续项化为仅对 o 求和的 $B(b,a,o)$ 形式。

<a id="lesson-information"></a>

## 4 · 信息只有在改变决策时才有控制价值

$$
\operatorname{VOI}=\sum_o\Pr(o\mid b)\max_a\sum_sb(s\mid o)r(s,a)-\max_a\sum_sb(s)r(s,a)-c_{\rm sense}
$$

这是静态隐状态、先感知一次再作终端决策的价值差；不是所有序列任务的一般信息价值公式。

若不计感知成本，观察后仍可采用原动作，因此这里的信息价值非负。加入成本后可能为负。减少 belief 的熵不必改善控制：传感器可能准确识别与奖励无关的细节。控制充分的状态只需保留会影响最优选择的信息，不必重建全部原始观察。

对模型未知的任务，还需区分“当前状态在哪里”与“动力学是什么”两种不确定性。仅对状态做 Bayes filtering，并没有自动学习未知转移。将模型参数也作为隐变量会得到更大的信念空间，计算成本随之增加。

<a id="experiment-extended-bayes_filter"></a>

### 实验：实验 · 一次错误观测，应当推翻过去的全部证据吗？

当隐藏状态通常保持不变、观测偶尔出错时，递归信念能否比只看当前观测更准确？

**环境与可用信息。** 两个隐藏状态以 0.9 的概率保持、0.1 的概率切换。观测以 0.8 的概率报告正确状态。学习器知道这两个概率，只看到观测；真实状态仅供评价使用。初始信念为 [0.5,0.5]。

**设置。** 种子 0–4，各处理 1200 条观测。Bayes filter 先传播旧信念，再用似然校正。无记忆对照每步都从均匀先验开始。两者接收相同种子的同一隐藏状态与观测流。没有梯度训练，也没有控制动作。

**检验的机制。** 变化只在于是否保留上一时刻的概率分布。连续一致的观测会积累证据；孤立的相反观测不必立即翻转判断。状态真的切换时，旧信念又会使反应产生滞后。

**测量。** 纵轴是观测后给真实隐藏状态分配的负对数概率，采用 0.95×旧值 + 0.05×新损失的滑动平均。越低越好。它评价概率质量，不是仅评价最可能状态是否猜对，也不是 RL 回报。

```bash
python3 implementations/extended_adaptation/bayes_filter.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/extended-bayes_filter/curves.svg)

横轴：observations。纵轴：prequential_logloss_ema。每种方法 1200 observations；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 第 1200 条观测后，五种子的平均损失为 0.353 nats，无记忆对照为 0.478 nats。记忆在该持久状态模型中有帮助，但曲线仍随错误观测与真实切换波动；滤波不是把不确定性消除为零。

**结论边界。** 这是已知正确模型的状态估计，不包含未知模型学习或策略改善。图中的末值是近期损失，不是全程平均。若实际切换率与假定模型不符，旧信念也可能有害。

**继续实验。** 先手算连续三次相同观测后的信念，再接入一次相反观测。然后分别只改变真实切换率与滤波器假定切换率，比较“需要更强记忆”和“模型失配”两种解释。

[源码](https://yingwen.io/crl-code/implementations/extended_adaptation/bayes_filter.py) · [逐种子记录](https://yingwen.io/crl-code/results/extended-bayes_filter/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/extended-bayes_filter/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/extended-bayes_filter/curves.json)

<a id="lesson-algorithm"></a>

## 5 · 精确 filter 与学习的 recurrent state

**算法：外部 reset 改变隐状态分布时，需要相应重置先验；普通训练 batch 边界不自动改变世界。**

1. 初始化已声明的先验 belief。
1. 根据当前 belief 计算或近似价值，选择动作。
1. 取得下一观察与奖励；仅使用已经到达的数据。
1. 用联合模型计算下一状态及实际奖励、观察的概率。
1. 按已收到的奖励与观察条件化，检查证据概率，再归一化。
1. 将后验保存为下一步 belief；重复行动。

$$
z_{t+1}=f_\theta(z_t,A_t,O_{t+1},R_{t+1}),\qquad \hat y_{t+1}=g_\theta(z_{t+1})
$$

RNN 用可训练的有限维摘要替代显式 belief；预测头或控制损失提供训练信号。

RNN 能携带历史，但不因此等于精确后验。预测下一像素、预测多种未来信号和优化控制回报会保留不同信息。TBPTT 截断参数梯度，hidden-state reset 则清除记忆内容；即使二者都发生在同一个代码边界，它们仍是不同操作。

训练时使用真实隐状态的 critic 也要声明评价对象。若 actor 带记忆，同一个环境状态搭配不同的 actor 记忆，未来行为可以不同；只给 critic 环境状态未必足以定义一个固定的 Bellman 递推。充分的集中上下文可包括隐状态与必要的控制器记忆，actor 执行权限仍保持原来的观察历史。

<a id="course-recurrent-credit"></a>

## 5.1 · 保存历史信息不等于学会怎样保存

给定递归状态 $h_t=f_\theta(h_{t-1},o_t,a_{t-1})$，其数值负责把过去信息带到当前。另一个对象是损失对过去计算的梯度。为了写清楚，先假设整段历史由同一组固定参数计算，再令 $P_t=\partial h_t/\partial\theta$。

$$
P_t=\frac{\partial f_\theta}{\partial\theta}
+\frac{\partial f_\theta}{\partial h_{t-1}}P_{t-1}
$$

第一项是当前计算对参数的直接依赖；第二项把过去依赖传播过来。BPTT 反向展开这张计算图；RTRL 前向递推敏感度。两者的时间顺序和存储代价不同。

例如 $h_t=0.99h_{t-1}+o_t$。一百步前的观察对当前数值的系数仍为 $0.99^{100}\approx0.366$。若只保留最近三十步的反向图，早于该边界的计算被当作常数，其参数梯度路径已经断开。hidden state 没有清零，不等于百步前的“如何写入记忆”仍能获得梯度。

把旧 hidden state 存入 replay，又出现另一个问题：这个向量由旧 encoder 生成。当前网络可能已经重新分配其坐标含义。用当前参数对一段旧观察做 burn-in 能缓解失配，但 burn-in 有限，未必恢复远期信息；从零状态开始则直接放弃此前历史。R2D2 专门研究了这种表示漂移与状态陈旧性。

参数逐步改变时，还需声明要对哪个计算过程求导：固定某组参数重新解释历史，还是对实际用多组参数执行过的学习轨迹求导。上面的共享参数敏感度递推对应前者的计算图，不能直接冒充“整个在线学习过程”的精确梯度。资格迹与近似在线梯度分别提供不同的折中。

与持续学习的连接：在一个长期运行、不方便重置的世界里，清空 hidden state、截断梯度、清空 replay 和重置参数是四个不同的操作。报告中必须写出实际发生了哪个操作，才能判断算法是否真的保留并利用长时经验。

<a id="recurrent-replay-task"></a>

### 5.1 · 仓库标签消失后，回放从哪份记忆开始

前面的 filter 从一个已声明先验开始，随后一直保存后验。学习的递归网络还会改变状态更新函数；若只从经验库抽取中间一段，就同时缺少早期观察和它们形成的活动。[状态构建章的长走廊](/zh/continual-rl/construction/state/#lesson-delayed-memory-budget)已经分开前向记忆和时间信用。本节再加入经验回放与参数副本，逐一检查初始活动、求导窗口和 TD 标签。

用一个只有左右货道的仓库取件任务隔离回放问题；第 6.1–6.2 节再加入中央货道与扫描行动。每回合包裹在 L 或 R，各半；入口 C 的准确标签 $x_0\in\{+1,-1\}$ 分别表示 L、R。随后必须前进四次，观察信号 $x_1=\cdots=x_4=0$；可见位置类型从 C、走廊 M 到路口 J，标签已经被遮挡。J 才能取 L 或 R，正确奖励 4、错误 −2，取件后真正终止；前进奖励为 0，折扣 $\gamma=0.9$。位置类型规定可用动作，但包裹位置不作为网络的额外输入。这里的准确标签与第 6.2 节带噪扫描是两个明示的任务条件。

$$
h_{-1}=0,\qquad h_t^{(v)}=\tanh\!\left(\rho^{(v)}h_{t-1}^{(v)}+w^{(v)}x_t\right),\quad\theta^{(v)}=(w^{(v)},\rho^{(v)})
$$

观察先进入状态，随后才选动作。回合内三套参数各自固定：采集版本 $\theta^b=(-0.8,0.8)$、当前版本 $\theta=(0.8,0.8)$、目标副本 $\theta^-= (0.6,0.7)$。它们是本例给定权重，不是训练所得。上标标记“用谁重算这份历史”，不是时间下标。

$$
Q_\theta(h,L)=1+3h,\quad Q_\theta(h,R)=1-3h,\quad Q_\theta(h,F)=1+3h^2
$$

F 是走廊唯一可用的前进动作，L、R 只在 J 可用。读出头手工固定，只有递归参数可写入；Q 是待校准的折扣回报估计。平局取 L。准确标签下 J 的真实动作值为正确门 4、错误门 −2；这里的网络估计尚未等于真值。

取一条正标签记录 $(1,0,0,0,0)$。旧参数得到 $h_4^b=-0.2264448$，在 J 取 R，收到 −2。当前参数从回合起点重算相同观察，得到 $h_4=0.2264448$，两个取件估计为 $(1.6793345,0.3206655)$，会选 L；在同一确定性包裹分支上另行重演，这次奖励为 4。经验库仍保存旧行动 R 与奖励 −2，回放训练使用这对样本；当前策略的重演结果不会替换记录。

![入口标签消失后的仓库轨迹与同一观察在旧、当前、目标三个参数版本下的递归活动](https://yingwen.io/crl-figures/recurrent-replay-walkthrough-task.svg)

上方只揭示已执行的右取件结果，未选左门保持问号。下方三条曲线从相同回合零状态开始，各自使用一套固定参数；横轴为观察时刻，曲线是精确活动展开。旧策略的右取件记录与当前策略的左取件重演分别报告。原创非线性算例，由独立 Python 与 JavaScript 计算交叉核验。

<a id="recurrent-replay-state"></a>

### 5.2 · 存储旧状态、零起点与 burn-in 分别恢复什么

设抽样片段从观察 $x_2$ 开始，前两项 $x_2,x_3$ 只用于预热（burn-in），学习段从 $x_4$ 开始。存储状态必须说明它位于哪次观察之前：本例保存的是旧采集器处理完 $x_1$ 后的 $h_1^b$，因此应先输入 $x_2$，不能再输入一次 $x_1$。旧状态、当前参数下的完整前缀重建和零初始化是三种不同起点。

$$
\tilde h_1\in\{h_1^b,0\},\qquad\tilde h_t=f_\theta(\tilde h_{t-1},x_t)\ (t=2,3),\qquad\tilde h_4=f_\theta(\tilde h_3,x_4)
$$

预热使用当前参数；目标副本也需要自己的状态展开。这里的完整前缀 $h_t^\theta$ 只是“同一观察、当前固定参数、真实回合零起点”的参照，不是精确 belief，也不必等于一个在线变化参数的控制器当时携带的活动。在线状态可能由 $\theta_0,\theta_1,\ldots$ 的混合版本形成。

| 如何开始学习段 | $x_4$ 前的活动 | $x_4$ 后的活动 | 当前读出的门 |
| --- | --- | --- | --- |
| 完整当前前缀：从 $x_0$ 重算 | 0.2880487 | 0.2264448 | L |
| 学习段直接清零 | 0 | 0 | L（平局） |
| 存储 $h_1^b=-0.4863203$，再预热 $x_2,x_3$ | −0.2880487 | −0.2264448 | R |
| 从零预热 $x_2,x_3$ | 0 | 0 | L（平局） |

本例的零预热没有恢复标签：两个输入都是 0，零活动仍为零。旧状态携带了标签的影响，却是在旧输入权重下写成负号；当前参数只处理后面的灰观察，也没有机会重写这个符号。若取得从真实回合起点开始的前缀 $x_0,\ldots,x_3$，从零预热便能精确恢复本例的当前参照。前缀是否覆盖必要线索，和它有多少步，是两个需要一起记录的条件。

DRQN 将单帧卷积特征送入 LSTM，再输出各动作 Q 值；其“Stable Recurrent Updates”比较从回合起点连续推进活动与在随机片段处清零。R2D2 §3 则比较零状态、存储状态和 burn-in，分析采集参数与 learner 参数不同导致的表示漂移及状态陈旧。这些是原论文中的具体训练选择；本页用标量 tanh 隔离其中的计算，并未实现整套 Atari agent。

$$
a_4^*=\arg\max_{a\in\{L,R\}}Q_\theta(h_4^\theta,a),\qquad y_3=\operatorname{stopgrad}\!\left[0+0.9Q_{\theta^-}(h_4^{\theta^-},a_4^*)\right]
$$

$M_3\to J_4$ 尚未终止。在线副本选门，目标副本评值，是一步 Double-Q 标签。本例 $h_4^{\theta^-}=0.1189176$，故 $y_3=0.9(1+3\times0.1189176)=1.2210776$；当前 F 估计为 $1+3h_3^2=1.2489161$。误把当前 $h_4$ 直接送入目标读出，会得到 1.5114010。目标副本包含递归单元时，同一观察不意味着同一活动坐标。

上式用完整前缀分别构造两个副本以便核验。实际只取得中段时，还需为两个副本分别声明初始化近似；相同旧存储值可以作为两个起点，但后续用 $\theta$ 与 $\theta^-$ 展开后通常不同。[DeepMind 官方 Acme JAX R2D2 learner](https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/r2d2/learning.py)明确执行这两次展开，再由在线输出选择、目标输出评值。冻结 target 的参数和标签，也不等于将 target 的活动复制成 online 活动。

<a id="recurrent-replay-gradient"></a>

### 5.3 · 不在前缀计损失，是否仍让末端误差穿过前缀

现在只训练 J 的旧记录 R／−2。终止标签 $y_4=-2$ 不含自举；在本次求导期间固定记录动作、奖励与参数版本，不对选门、环境结果或跨次优化器更新求导。先保留完整当前前缀，再比较“前缀没有直接损失”和“学习入口把前缀活动作为常数”。两个程序有相同的 $h_4$，却不是同一个可微函数。

$$
\ell_4(\theta)=\tfrac12\left(Q_\theta(h_4,R)-(-2)\right)^2,\qquad E_t=\begin{bmatrix}\partial h_t/\partial w\\\partial h_t/\partial\rho\end{bmatrix}=(1-h_t^2)\left[\rho E_{t-1}+\begin{bmatrix}x_t\\h_{t-1}\end{bmatrix}\right]
$$

完整当前前缀从 $E_{-1}=0$ 开始。因子 $1-h_t^2$ 是 tanh 的局部导数，括号保留过去的参数路径及当前直接影响。旧采集版本下保存的敏感度是关于旧参数的导数，不能当作当前参数的 $E$ 接上；存储旧活动作常数初始化时，当前 E 应从零开始。

$$
E_4^{\rm full}=(0.1312196,0.9596107)^\top,\qquad\tilde h_3=\operatorname{stopgrad}(h_3),\quad E_4^{\rm detach}=(1-h_4^2)(0,h_3)^\top=(0,0.2732783)^\top
$$

detach 保存 $h_3=0.2880487$ 的数值，但删除它对 $w,\rho$ 的历史路径。学习段唯一输入 $x_4=0$，故直接 w 导数为零；ρ 仍直接乘在保存的活动上，因此其局部导数保留。清空 $h_3$ 则会把 $h_4$ 也变为零，那是另一项操作。

$$
g_4=(Q_\theta(h_4,R)+2)(-3)E_4,\qquad\theta^+=\theta-0.01g_4
$$

记录 R 的读出导数是 −3。完整梯度约为 $(−0.9135506,−6.6808061)$，写入 $(w^+,\rho^+)=(0.8091355,0.8668081)$；入口 detach 给 $(0,−1.9025628)$，写入 $(0.8,0.8190256)$。本次写入之后，已经形成的 h 不自动重算；下次重演或未来观察才按新参数计算。

![完整当前前缀与入口detach的相同前向活动、不同反向路径和两个参数的梯度](https://yingwen.io/crl-figures/recurrent-replay-walkthrough-gradient.svg)

蓝色节点按观察推进活动；紫色线单独表示求导路径，叉号切断 $h_3$ 的导数。灰色前缀没有直接损失，完整 BPTT 仍可以接收末端误差。下方长度表示参数梯度的绝对值，同一参数内共用零点；w 与 ρ 分别使用标明的范围。独立嵌套展开和固定边界有限差分核验两种函数。

因此 burn-in 长度 B、学习损失窗口 U、反传可达窗口 K 要分开声明。只把预热输出从损失求和中排除，没有切断它通向后续损失的计算图。本例显示两种约定；官方 Acme JAX 实现将预热展开放在被求导的 loss 函数内，学习段剔除前缀的直接损失，但入口没有显式 stop-gradient。不能根据“burn-in”这个名称推断所有实现都采用本例的 detach 版本。

**算法：本页固定一次终止样本和一次非终止标签分别核验。采样权重、优先级、n-step 与值变换属于完整 R2D2 的其他机制，未包含在此标量程序中。**

1. 固定本次 online θ 与 target θ⁻；读出旧记录与真实终止标记。
1. 明确初始状态位于哪次观察之前：真回合起点、旧存储活动或零近似。
1. 用两个副本各自的参数推进 burn-in 前缀，不计该前缀的直接损失。
1. 若协议要求 TBPTT：在学习入口 detach 活动；数值 h 继续保留。
1. 推进学习段；online 在后继可用动作中选择，target 评值，标签停止梯度。
1. 真实终止只保留已收到奖励；普通片段边界保留所需后继自举。
1. 对记录动作计算损失和梯度，再更新 online 参数；不改写记录动作。
1. 按另行声明的周期复制 target 参数；下次重建活动时使用新版本。

<a id="recurrent-replay-boundaries"></a>

### 5.4 · 片段结束没有让仓库结束，预热也不能补出缺失标签

在 $M_3$ 处结束一次采样，机器人仍将到 $J_4$，故 continuation 保留；本例尾标签仍是 1.2210776。把“chunk 结束”写成 terminal 会把它误设为 0。若片段没有保存所需后继观察，应取得后缀或不训练这条尚无标签的转移。取件后的真实终止才让尾值为零；下一回合重新抽取包裹时，内部活动从零开始。仅在抽样入口 detach 会保留记忆，在抽样入口 reset 则会删除记忆。

$$
\|f_\theta(h,x)-f_\theta(h',x)\|\leq\kappa\|h-h'\|\ (\kappa<1)\quad\Longrightarrow\quad\|\tilde h_{s+B}-h_{s+B}\|\leq\kappa^B\|\tilde h_s-h_s\|
$$

这个初始化误差界要求两条计算采用同一固定 θ、同一 B 步输入，并在涉及的状态区域统一满足收缩条件。本例 tanh 的导数不超过一，故 |ρ|=0.8 可取 κ=0.8。它只比较初始化造成的活动差，不保证恢复必要标签、不保证 Q 校准，也不保证动作相同。

这里旧、当前活动符号相反；输入都是 0 时，$h\mapsto\tanh(0.8h)$ 保持符号。因此误差随灰观察缩小，旧状态重建仍一直选错门。若只把保持系数改为 $\rho=1.2$，正、负活动在八次灰观察之后仍相距约 1.31763；没有上述全局收缩条件，就不能沿用 $0.8^B$ 的界。这个例子给出一项失败，并不说明所有非收缩记忆都失败。

![收缩与非收缩递归的初始化误差，以及两个缺失入口标签的相同灰片段被零起点合并](https://yingwen.io/crl-figures/recurrent-replay-walkthrough-boundaries.svg)

上图横轴为同一灰前缀的长度 B，纵轴为当前参照与重建的活动差。ρ=0.8 的误差缩小但错误符号保留；ρ=1.2 的指定正负起点持续分离。下方两条入口标签均在采样片段外，零初始化看到相同灰观察，便都取左。曲线由固定 tanh 递推生成，不是训练性能曲线。

再同时检查两个历史：入口分别为 +1 与 −1，片段都只保存 $(0,0,0)$。零初始化后它们都给出 h=0，无论再读多少个零也无法区分，固定平局规则总取 L。两种包裹位置等概率时，期望奖励为 $(4-2)/2=1$；保存完整准确标签历史可得 4。这个缺口来自片段与初始化删除了信息。延长反传窗口只能改变可求导路径，无法创造未保存的标签。

持续学习时，还需分别记录环境变化、采集策略版本、活动初始化来源、在线与目标参数版本，以及预热和信用窗口。可以先固定观察记录，比较完整当前前缀、存储旧状态、零状态和不同前缀长度；再用同一信息条件下的新交互检验动作和回报。活动误差变小与闭环收益改善应各自测量。

<a id="lesson-example"></a>

## 6 · 两个可手算的例子

设物体藏在左室或右室。智能体不能看见它，只能等待一步，再读取一盏有噪声的指示灯。等待期间物体可能换房间；灯亮也不唯一对应某个位置。因此先用转移模型预测位置，再用实际灯光更新信念。下图始终不揭示物体的真实位置。

![两房间隐藏位置的先验、概率搬运、灯光似然和后验四步图](https://yingwen.io/crl-figures/concept-deep3-belief-update.svg)

给定模型下的一次精确滤波。圆面积表示位置概率；虚线表示模型搬运概率。灯光行的十个点表示似然比例，不是新采集的十次观测。预测概率分别乘以灯亮似然，再共同除以证据 $0.572$。图按 Kaelbling、Littman、Cassandra（1998）§3.3 的滤波规则绘制；两房间参数是本节算例。

先验为 $(0.6,0.4)$，转移矩阵两行为 $(0.9,0.1)$ 与 $(0.2,0.8)$。预测分布是 $(0.62,0.38)$。收到观测的似然为 $(0.8,0.2)$，未归一化后验为 $(0.496,0.076)$，证据为 $0.572$，后验约为 $(0.86713,0.13287)$。直接将似然归一化为 $(0.8,0.2)$ 会丢掉先验和动态预测。

为什么不能把灯亮时的左室概率直接写成 0.8？0.8 回答的是“已知物体在左室，灯有多大概率亮”。控制需要的却是“已经看到灯亮，物体有多大概率在左室”。前一个问题给出似然；后一个问题还取决于等待后各位置原本有多可能。

另一个任务只有左右两扇门，隐藏的正确门先验各半，选对得一、选错得负一。立即选择的期望收益为零。准确率 0.8 的传感器让观察后的最佳选择收益为 0.6；感知成本 0.1，净收益为 0.5。若传感器完全无信息，净增益为 −0.1。

<a id="belief-planning-task"></a>

### 6.1 · 同一个路口：保存分布还是只保存位置均值

继续用一个取件任务，把状态估计与行动接在一起。机器人站在仓库路口，左、中、右货道均被门遮住；包裹只在一个货道，隐藏位置为 $S\in\{L,C,R\}$，坐标编码是 $x(L)=-1,x(C)=0,x(R)=1$。部署观察总是同一个路口画面，机器人不能读取真实位置。已知的任务模型、初始 belief 与预算由设计者提供；不同的已到达历史可以给出不同 belief。

任务最多允许两个动作，使用 $\gamma=1$ 的有限期望总奖励。初始阶段可直接取左、中或右，也可扫描一次。扫描不移动包裹，立即扣除 $0.4$，随后只能取件，不能再次扫描。取件正确得 $4$、错误得 $-2$，奖励到账后任务真正终止，尾值为零。若直接取件，任务在第一个动作后便结束。阶段与剩余预算也是决策状态的一部分。

$$
b^{A}=(\tfrac12,0,\tfrac12),\quad b^{B}=(0,1,0),\qquad \mathbb E_{b^A}[x(S)]=\mathbb E_{b^B}[x(S)]=0
$$

belief 分量按 L、C、R 排列。两个 belief 都可在同一个路口出现；它们是本节给定的两份历史条件分布，不是把读者图中的真实位置透露给机器人。

$$
Q_{\rm collect}(b,j)=4b(j)-2[1-b(j)]=6b(j)-2,\quad j\in\{L,C,R\}
$$

对包裹位置取完整条件期望。历史 A 的三个动作值为 $(1,-2,1)$，历史 B 为 $(-2,4,-2)$，最佳取件货道不同。平局时教程选择最左动作。

![读者全知仓库示例、机器人实际遮挡观察，以及两份均值相同却动作不同的belief](https://yingwen.io/crl-figures/belief-planning-walkthrough-task.svg)

先读上方两种信息权限：读者示例画出左侧包裹，部署观察只画遮挡门。下方柱高表示两份给定 belief，二者位置均值都为零，却分别偏好侧边与中央。原创任务与精确动作期望，由本页教程生成；全知示例不作为 actor 输入。

若把平均位置零当作确定的中央位置，便会给历史 A 的取中动作预测 $4$，而它的真实期望为 $-2$。即使均值预测完全正确，这个非线性取件奖励仍需位置分布。给定正确模型时完整 belief 递归地保留预测与控制所需的信息；只保留一个均值没有这样的保证。能否进一步压缩，要看压缩后是否仍能区分后果与动作值。

<a id="belief-planning-scan"></a>

### 6.2 · 扫描之后，哪条观察真正改变取件动作

扫描给出一个标签 $O\in\{L,C,R\}$。用行表示包裹位置、列表示标签，完整观测模型如下；准确率 $0.8$ 的剩余概率均分给另两个标签。扫描奖励固定，所以此步奖励不额外揭示位置。

$$
\Pr(O=o\mid S=s,A=\mathrm{scan})=\begin{cases}0.8,&o=s,\\0.1,&o\ne s,\end{cases}\qquad O_{\rm scan}=\begin{bmatrix}.8&.1&.1\\.1&.8&.1\\.1&.1&.8\end{bmatrix}
$$

包裹在扫描期间保持原位，$T_{\rm scan}(s,s')=\mathbf1[s=s']$。这两项共同规定了全部扫描后果，而不是只给一个正确率。

$$
p_o=\sum_s b^A(s)O_{\rm scan}(s,o),\qquad b^A_o(s)=\frac{b^A(s)O_{\rm scan}(s,o)}{p_o}
$$

分母是该标签的发生概率。收到 L 时分子为 $(0.4,0,0.05)$，分母 $0.45$，所以后验为 $(8/9,0,1/9)$；收到标签不是直接取得真实位置。

| 收到标签 | 分支概率 | 归一化后验 | 取件动作 | 条件期望奖励 |
| --- | --- | --- | --- | --- |
| $L$ | $0.45$ | $(8/9,0,1/9)$ | 左 | $10/3$ |
| $C$ | $0.10$ | $(1/2,0,1/2)$ | 左（与右平局） | $1$ |
| $R$ | $0.45$ | $(1/9,0,8/9)$ | 右 | $10/3$ |

$$
Q_{\rm scan}(b^A)=-0.4+0.45\cdot\tfrac{10}{3}+0.10\cdot1+0.45\cdot\tfrac{10}{3}=2.7
$$

立即取左的期望为 $1$，所以扫描在这里增加 $1.7$。标签 C 没有区分左右；保留这一条分支及其成本，才得到完整期望。

![扫描计划树的三条信号分支、后验概率和不同取件动作](https://yingwen.io/crl-figures/belief-planning-walkthrough-plan.svg)

紫色虚线表示模型中的三种可能标签；机器人真实执行时只沿收到的一条分支行动。横条按左、中央、右后验概率着色。每条分支先条件化再选取件动作，最后按标签概率加权。固定 scan→左的净期望为 0.6；扫描后依标签取左/左/右的净期望为 2.7。原创精确枚举。

信息价值来自后续行为对标签的响应。若扫描后无论看到什么都取左，成功率仍为 $1/2$，净收益只剩 $1-0.4=0.6$。历史 B 已知包裹在中央，扫描后也始终取中；扫描只能把收益从 $4$ 降到 $3.6$。同一个传感器，面对不同 belief 会有不同控制价值。[模型与规划章](/zh/continual-rl/foundations/deep/model-based/#belief-planning-contingent)将用这一任务比较确定动作序列与条件计划。

本节滤波与条件计划分别对应 [Kaelbling、Littman、Cassandra（1998）§3.3–3.4 与 §4.1](https://www.cassandra.org/arc/papers/aij98.pdf)及 [Algorithms for Decision Making §19.2、§20.1–20.2](https://algorithmsbook.com/)。取件参数是本节原创算例，读者可以在已知模型下枚举所有计划。

<a id="lesson-code"></a>

## 7 · 可运行 filter 与信息价值

矩阵预测、Bayes 归一化与一次感知的期望决策价值；两个例子不依赖采样误差。

```python
def belief_update(prior, transition, likelihood):
    """Transition[s][s_next], likelihood[s_next] for the observation received."""
    probabilities(prior)
    n = len(prior)
    if len(transition) != n or len(likelihood) != n or any(len(row) != n for row in transition):
        raise ValueError('matching state dimensions required')
    for row in transition:
        probabilities(row)
    if any(not 0 <= p <= 1 for p in likelihood):
        raise ValueError('observation likelihood must be in [0,1]')
    predicted = [sum(prior[s]*transition[s][sp] for s in range(n)) for sp in range(n)]
    evidence = dot(predicted, likelihood)
    if evidence <= 0:
        raise ValueError('impossible observation under this model')
    return [p*l/evidence for p, l in zip(predicted, likelihood)], evidence


def information_value(prior, sensor, rewards, sensing_cost=0.):
    """One sensing step, then choose one terminal action.

    sensor[state][observation]; rewards[action][state]; hidden state is static.
    """
    probabilities(prior)
    for row in sensor:
        probabilities(row)
    before = max(dot(prior, r) for r in rewards)
    after = 0.
    for o in range(len(sensor[0])):
        joint = [p*row[o] for p, row in zip(prior, sensor)]
        evidence = sum(joint)
        if evidence:
            posterior = [x/evidence for x in joint]
            after += evidence*max(dot(posterior, r) for r in rewards)
    return before, after-sensing_cost, after-sensing_cost-before
```

执行 python3 extended_foundations_lab.py demo 查看 posterior、evidence 和信息收益；执行 test 检查归一化、零证据与无信息传感器。代码没有训练 recurrent 网络，也没有实现一般 POMDP 规划器。pomdp-solve 是 Cassandra 的经典求解软件；pomdp-py 是后续研究框架，二者的归属与实现范围不同。

取件任务另有一个可独立下载的 [标准库 Python 教程](/crl-code/tutorials/belief-planning-walkthrough.py)。保存到空目录后运行下面的命令，逐条打印信号概率、后验和动作；--test 用精确分数枚举 27 个条件计划，并在多组 belief 与传感器核上交叉检查。

单文件即可运行，无额外依赖；输出是已知有限任务的解析结果。

```bash
python3 belief-planning-walkthrough.py
python3 belief-planning-walkthrough.py --test
python3 belief-planning-walkthrough.py --json
```

递归回放另有 [独立 Python 标准库教程](/crl-code/tutorials/recurrent-replay-walkthrough.py) 和 [JavaScript 逐步计算](/crl-code/figures/recurrent-replay-walkthrough.mjs)。Python 对每个前缀重新嵌套 tanh，并展开局部 Jacobian 的乘积求导；JavaScript 推进敏感度递推。两者逐字段核对三个参数版本、两个隐藏历史、损失、标签和边界，有限差分分别固定或重算前缀以核验完整与 detach 函数。

下载单文件后可从空目录运行，只需 Python 3.10+ 标准库；默认打印状态、取件动作、标签与梯度，--json 输出同一确定性数据。

```bash
python3 recurrent-replay-walkthrough.py
python3 recurrent-replay-walkthrough.py --test
python3 recurrent-replay-walkthrough.py --json
```

<a id="lesson-branches"></a>

## 8 · 假设边界与持续学习接口

- 模型错误：精确计算错误模型下的 posterior，仍可能产生系统性误判。
- 信息泄漏：训练时的 simulator state 可以帮助 critic，但执行策略若依赖它，就不是相同观察条件。
- 序列失配：从 replay 抽单帧无法一般性重建 history-dependent state；burn-in 也需要明确参数版本。
- 变化环境：原观测或转移模型失效时，belief 可能越来越自信地错误，而不是自动适应。

与 CRL 的连接是状态持续构建和状态模型持续校准。可固定任务奖励，单独改变传感器可靠度，比较已知新模型的 Bayes oracle、旧模型 filter 和学习的 recurrent state。这样能区分信息不足、模型滞后与优化失败，不能仅凭回报下降判断遗忘。

<a id="rlss-hidden-state-bootstrap"></a>

## 不充分的状态下，Monte Carlo、TD 与资格迹分别能修复什么？

先固定策略，只考虑价值预测。设真实过程是有限 Markov 奖励过程，但学习器只看到其特征。不同真实状态可以产生同一个特征。此时有两个独立困难：表示无法区分不同未来；自举又把这种不准确的预测当成学习目标。更长的回报可以减轻第二项，却不能凭空恢复第一项丢失的信息。

$$
v=(I-\gamma P)^{-1}r,\qquad \widehat v=\Phi w,\qquad
\Pi_Dv=\Phi(\Phi^\top D\Phi)^{-1}\Phi^\top Dv.
$$

$P$ 是固定策略的转移矩阵，$r$ 是一步期望奖励，$0\leq\gamma<1$。$D$ 是访问权重的对角矩阵；假设所有相关状态有正权重，且 $\Phi$ 列满秩。右式是这个表示在加权平方误差下能达到的最佳预测，不是每个隐藏状态的真价值。

若两个等频出现的隐藏状态有价值 2 和 8，而特征完全相同，最佳单一预测是 5，均方误差仍为 9。更多样本能更准确地学到 5，却不能把两个情形分开。MC 学习的是给定现有信息的平均回报；它没有偷偷获得真实状态。

$$
\begin{aligned}
 A_\lambda&=\Phi^\top D(I-\gamma\lambda P)^{-1}(I-\gamma P)\Phi,\\
 b_\lambda&=\Phi^\top D(I-\gamma\lambda P)^{-1}r,\qquad A_\lambda w_\lambda=b_\lambda.
 \end{aligned}
$$

这是固定特征、固定策略、相应访问分布下线性 TD(λ) 的期望固定点方程。由多步 Bellman 算子的几何级数得到；它不是非线性网络、变化策略或任意数据分布的稳定性声明。

$$
\lambda=1:\quad A_1=\Phi^\top D\Phi,\qquad b_1=\Phi^\top Dv,\qquad \Phi w_1=\Pi_Dv.
$$

这里反矩阵相消。λ 趋向一时，期望固定点趋向直接回报回归的投影。实际采样仍有方差、回报延迟与有限步长误差。

对持续、平稳、on-policy 的链，取 $D$ 为其稳态分布。投影后的多步算子的一个压缩系数是 $\kappa_\lambda=\gamma(1-\lambda)/(1-\gamma\lambda)$。三角不等式给出下面这个保守的范数界。它解释趋势，不断言某个有限样本实验随 λ 单调改善。

$$
\|\Phi w_\lambda-v\|_D\leq
 \frac{1}{1-\kappa_\lambda}\|\Pi_Dv-v\|_D
 =\frac{1-\gamma\lambda}{1-\gamma}\|\Pi_Dv-v\|_D.
$$

把 v 与其投影插入固定点误差，并把压缩项移到左边即可得到。回合访问权重、离策略权重或随时间变化的特征不能不加检查地套用这组假设。

固定点诊断：区分表示误差与自举引入的偏离；不是训练曲线。

```python
import numpy as np
P = np.array([[.8, .2, 0.], [0., .5, .5], [.3, 0., .7]])
r = np.array([1., 0., -1.])
Phi = np.array([[1., 0.], [1., 0.], [0., 1.]])
gamma = .9
# 先求稳态分布；这个诊断拥有真实模型，在线学习器并没有。
d = np.linalg.lstsq(np.vstack([P.T-np.eye(3), np.ones(3)]),
                    np.r_[np.zeros(3), 1.], rcond=None)[0]
D = np.diag(d)
v = np.linalg.solve(np.eye(3)-gamma*P, r)
projected = Phi @ np.linalg.solve(Phi.T@D@Phi, Phi.T@D@v)
for lam in (0., .5, 1.):
    M = np.linalg.solve(np.eye(3)-gamma*lam*P, np.eye(3))
    w = np.linalg.solve(Phi.T@D@M@(np.eye(3)-gamma*P)@Phi,
                        Phi.T@D@M@r)
    print(lam, np.sqrt(d @ (Phi@w-v)**2))
    if lam == 1.: assert np.allclose(Phi@w, projected)
```

课程中的 bit-to-bit 练习则走向另一个问题：行动会影响获得的信息。增加历史特征、构造可递归更新的状态、选择能区分假设的行动，才可能减少信息混叠。长观察窗口也会增加稀疏度和估计难度。比较状态构造时，应给它们相同的数据与存储预算；不能把拥有真实隐藏状态的零表示误差与有限样本学习误差直接比较。

接入控制后还要重新检查策略类。某个固定策略下两个历史的回报相同，并不表示对所有未来行动都相同。价值预测器可以在当前策略上工作良好，但规划器仍需要区分这些历史。状态、预测问题与控制权限必须共同声明。

<a id="lesson-check"></a>

## 9 · 练习与答案

- 问：观测看起来相同，是否必须采取相同动作？答：不必；不同历史可导致不同 belief。
- 问：belief 熵下降是否意味着决策价值增加？答：不保证。它可能只减少无关信息的不确定性，且感知有成本。
- 问：Bayes 更新的证据为零时可以加一个很小的数继续吗？答：数值平滑可以作为建模改动，但不能隐去模型与观测矛盾；需要报告所用平滑和支持假设。
- 实验：将传感器准确率从 0.8 改为 0.5。在门任务里后验保持先验，信息净收益应为 −0.1；此结论不依赖控制网络。

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

- [Sutton & Barto · Reinforcement Learning, second edition](http://incompleteideas.net/book/the-book-2nd.html)：§17.3 pp.464–467：观察历史、递归状态更新与给定模型下的 belief。

- [Hausknecht & Stone · Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527)：2015 工作，已核对 v4 的 DRQN Architecture 与 Stable Recurrent Updates；随机片段清零与整回合递推的区别。

- [Kapturowski et al. · Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/forum?id=r1lyTjAqYX)：ICLR 2019 论文入口；本节已核对公开送审稿 §2.3、§3 的 stored state、burn-in 与状态陈旧诊断。

- [R2D2 · OpenReview 原始公开送审稿](https://openreview.net/references/pdf?id=Hy7PKCFCQ)：可读全文版本；§3 对初始化与 Q discrepancy 的机制说明，经验发现保持原实验范围。

- [DeepMind · Acme JAX R2D2 learner](https://github.com/google-deepmind/acme/blob/master/acme/agents/jax/r2d2/learning.py)：loss 中分别展开 online/target 活动，剔除前缀直接损失；预热处没有显式 stop_gradient。后续官方实现，不冒称论文原始训练工程。

- [Kaelbling、Littman、Cassandra · Planning and Acting in Partially Observable Stochastic Domains](https://www.cassandra.org/arc/papers/aij98.pdf)：作者托管原文；belief、决策与 POMDP 求解。

- [Cassandra · pomdp-solve](https://www.pomdp.org/code/index.html)：经典 POMDP 求解软件的作者入口，并非神经 recurrent agent。

- [h2r · pomdp-py](https://github.com/h2r/pomdp-py)：框架作者的模型、belief 与规划接口；是后续工具，不是 1998 原文代码。

- [Kochenderfer、Wheeler、Wray · Algorithms for Decision Making](https://algorithmsbook.com/)：作者书站提供决策、信念状态、模型与规划教材及配套 Julia 代码入口。

- [Baisero & Amato · Unbiased Asymmetric Reinforcement Learning under Partial Observability](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p44.pdf)：§4–5：state-only critic 的条件与 history-state value；额外训练信息不自动消除历史依赖。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-partial-observability#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-partial-observability#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-deep-partial-observability)
<a id="study-connections"></a>

## 与教材主线的衔接

本章是可按问题选择的并列研究分支，不要求先读完其他深度研究分支。

### 当前输入保留了哪些历史信息？

Bellman 方程先假定有足够的状态。表格为不同状态分别存值；它不负责从相同观察中恢复被遗漏的历史。

函数逼近与深度方法：特征共享是在已知信息上泛化；递归状态则保留过去信息。增加网络宽度不等于补回历史，低训练误差也不证明输入满足 Markov 性。

持续学习中的研究问题：策略改变以后，原来的状态压缩是否仍能预测行动后果？构造状态的网络、运行时记忆、资格迹与优化器状态如何共同更新？

[MDP 的状态条件](https://yingwen.io/zh/continual-rl/foundations/tabular/mdps/) → [表示与泛化](https://yingwen.io/zh/continual-rl/foundations/approximation/features-control/) → [不完全可观测](https://yingwen.io/zh/continual-rl/foundations/deep/partial-observability/) → [智能体状态](https://yingwen.io/zh/continual-rl/construction/state/)


### 可进一步检验的问题

- [02 · 有限的内部状态应当保留哪些历史信息？](https://yingwen.io/zh/continual-rl/research/#research-agent-state)：精确信念更新与信息行动提供可检查的参照，进而可以辨认有限递归状态丢失的是哪项决策信息。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)
- [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)
- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)
- [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)

对应原始材料：Sutton & Barto §17.3；Kaelbling、Littman、Cassandra：POMDP；DRQN：Stable Recurrent Updates；R2D2 §3：recurrent replay；Algorithms for Decision Making：state uncertainty。本文为原创讲解，原书、论文与上游代码保留各自许可。
