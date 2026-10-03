# 经典 RL：从交互到评价与控制

先建立评价与控制，再理解函数近似带来的训练问题，最后研究长期交互中的适应与知识积累。这是先修关系，不是后来的算法取代前面的算法。

先在状态和动作数量有限、转移规律固定的问题中理解 RL。这里的重点不是记住算法名称，而是区分：问题要求优化什么、一次更新估计什么、数据由谁产生。随后用函数逼近和模型学习说明这些基本关系如何扩展。

学习目标：

- 能从奖励序列写出回报，区分真实终止与采样截断。
- 能推导固定策略的 Bellman 方程和最优递推，并实现评价—改善循环。
- 能解释 MC、TD、控制、资格迹和函数逼近之间的依赖关系。

## 1 · 交互、状态与回报

一次行动的好坏，为什么不能只由眼前的奖励判断？

在离散时间中，智能体观察状态 $S_t$，按策略 $\pi(a\mid s)$ 选择动作 $A_t$，环境返回奖励 $R_{t+1}$ 和下一状态 $S_{t+1}$。MDP 假定给定当前状态与动作后，下一步的分布不再依赖更早的历史。这里先假定有限状态、有限动作、固定转移规律和有界奖励。

策略需要比较动作带来的整个未来，因此先规定回报。折扣回报让距离当前更远的奖励获得权重 $\gamma^k$；当 $0\leq\gamma<1$ 且奖励有界时，无限和存在。$\gamma$ 是目标的一部分，不只是训练参数。若一个动作眼前收益高但让后续选择变差，只优化即时奖励就会选错。

终止与停止采样不是一回事。到达定义中的终点后没有未来回报；达到一次训练调用的步数上限，则环境可能仍会继续。后一种情况的价值目标通常需要 bootstrap。无折扣有限时域问题还应把剩余时间纳入状态，否则同一位置在剩一步和剩十步时未必有相同最优动作。

$$
G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}=R_{t+1}+\gamma G_{t+1},\qquad v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s]
$$

第一式把未来分成第一步和剩余部分。第二式对策略引起的所有未来取条件期望，定义价值；它还不是学习算法。

### 动手与核对

[下载 objectives_lab.py](https://yingwen.io/zh/continual-rl/download/objectives_lab.py)

在保存该文件的目录运行：

```sh
python3 objectives_lab.py stopping
python3 objectives_lab.py test
```

预期检查：输出 discounted 与 independent_stopping 均为 2.75；同一条长度为一的前缀，保留继续价值时为 10，误当终止时为 1。

实验范围：这是目标与边界条件的确定性检验，不是学习性能比较。

自测：奖励依次为 1、2、3，随后真正终止；折扣为 0.5。第一个状态的回报是多少？

解答：1 + 0.5 × 2 + 0.25 × 3 = 2.75。若第三步只是采样截断，应再加 0.5³ 乘下一状态的价值估计。

完整教材：

- [交互、奖励与优化目标](https://yingwen.io/zh/continual-rl/foundations/objectives/)

原文、课程与实现：

- [Sutton & Barto · Reinforcement Learning, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：第 3 章：agent–environment interface、回报和 MDP。
- [Spinning Up · 关键概念](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)：对照状态、观测、动作、策略、轨迹和回报的区别。

## 2 · 已知模型：策略评价与动态规划

若已知每个动作的后果，怎样把局部计算变成长期判断？

模型 $p(s',r\mid s,a)$ 给出下一状态和奖励的联合分布。固定策略后，对回报递推取期望，就得到 Bellman 期望方程。更新某个状态时不必枚举所有未来轨迹，只要枚举一步后果，再接上对未来的当前估计。这种一步递推称为 backup。

评价固定 $\pi$ 时，下一动作按 $\pi$ 求平均；寻找最优策略时，下一动作取最大值。前者的固定点是 $v_\pi$，后者是 $v_*$。在本节的折扣有限 MDP 中，两种 Bellman 算子都是最大范数下的 $\gamma$ 收缩，因此反复完整更新会收敛到唯一价值固定点。

策略迭代先评价当前策略，再选择该价值下的一步贪心动作。策略改善成立，是因为用新动作并继续旧策略已不差，再在后续每一步重复这种改善不会降低价值。价值迭代则不等评价完全结束，每一轮直接做最优 backup。评价与改善相互作用的共同思想称为广义策略迭代（generalized policy iteration，GPI）。这里不是迁移学习中同缩写的 generalized policy improvement。

$$
\begin{aligned}(T_\pi v)(s)&=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)[r+\gamma v(s')],\\(T_*v)(s)&=\max_a\sum_{s',r}p(s',r\mid s,a)[r+\gamma v(s')].\end{aligned}
$$

策略迭代求解 v = Tπv 后改善策略；价值迭代使用 vₖ₊₁ = T*vₖ。真实终止状态的余项取零。

### 动手与核对

[下载 control_problem_lab.py](https://yingwen.io/zh/continual-rl/download/control_problem_lab.py)

在保存该文件的目录运行：

```sh
python3 control_problem_lab.py dp
python3 control_problem_lab.py test
```

预期检查：策略依次为 (0,0)、(1,0)、(1,1)，最后价值约为 (2.368421, 2.631579)；策略迭代与价值迭代一致。

实验范围：模型已知，计算量按 backup 计数；它不消耗真实环境交互来估计模型。

自测：A 可拿 1 后结束，也可零奖励转到 B；B 可拿 2 后结束，也可拿 0.5 后回 A。γ = 0.9。为什么两处都选循环动作反而更好？

解答：循环策略满足 v(A)=0.9v(B)、v(B)=0.5+0.9v(A)，解为 45/19 和 50/19，均高于对应退出奖励。只比较即时奖励会漏掉循环中的长期收入。

完整教材：

- [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)
- [控制问题与广义策略迭代](https://yingwen.io/zh/continual-rl/algorithms/control/)

原文、课程与实现：

- [Sutton & Barto · Reinforcement Learning, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：第 4 章：策略评价、策略改善、策略迭代、价值迭代与 generalized policy iteration。
- [David Silver · UCL 强化学习课程](https://davidstarsilver.wordpress.com/teaching/)：课程页提供讲义和视频；按 MDP、动态规划、无模型预测与控制、函数逼近的顺序选读。

## 3 · 未知模型：Monte Carlo 与 TD 预测

不知道转移概率时，一条经历能够教会价值函数什么？

现在模型未知，但暂时仍固定策略。Monte Carlo 等一次经历结束后，把观测到的完整回报作为训练目标。TD 不等完整结果，而用一步奖励加下一状态的价值估计作为目标。这就是 bootstrap：用已有预测帮助学习另一个预测。

二者不是“一个有偏、一个一定正确”的简单对立。在适当采样条件下，完整回报是当前策略价值的无偏样本，但方差可能较大。TD 的单步目标会受当前估计误差影响，却能较早更新。表格 on-policy TD 的收敛需要访问覆盖、合适步长和固定环境等条件；一次 TD 误差不等于真实价值误差。

MC 与 TD 都可以逐样本实现。两者的关键差别是学习目标是否接上估计值，不是代码有没有循环、是否使用深度网络。先在一条两步轨迹上手算，再观察多次重复后两种方法如何接近相同答案。

$$
\begin{aligned}V(S_t)&\leftarrow V(S_t)+\alpha[G_t-V(S_t)]&&\text{MC},\\\delta_t&=R_{t+1}+\gamma V(S_{t+1})-V(S_t),\\V(S_t)&\leftarrow V(S_t)+\alpha\delta_t&&\text{TD(0)}.\end{aligned}
$$

右端各价值都在本次更新前读取。真正终止时下一价值为零。α 控制每次新证据改变预测的幅度。

### 动手与核对

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

在保存该文件的目录运行：

```sh
python3 foundations_detail_lab.py value
python3 foundations_detail_lab.py test
```

预期检查：反复观察 A→B→终点，奖励为 0、1；MC 和 TD 最终都接近 (0.9, 1.0)。

实验范围：这是固定策略的预测实验，没有策略改善或探索性能结论。

自测：初值全零、α = 0.1、γ = 0.9。第一条 A→B→终点经历后，MC 与按时间顺序执行的 TD 各是什么值？

解答：MC 得到 A=0.09、B=0.1。在线 TD 在访问 A 时尚不知道 B 的价值，因此 A=0、B=0.1。后续经历会把 B 的信息传播到 A。

完整教材：

- [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)

原文、课程与实现：

- [Sutton & Barto · Reinforcement Learning, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：第 5、6 章：MC 与 TD prediction；先读预测部分，再读对应控制扩展。
- [Sutton · Learning to Predict by the Methods of Temporal Differences](https://doi.org/10.1007/BF00115009)：TD 学习的原始论文；与本节的表格例子对应阅读。

## 4 · 从预测到控制：数据与策略共同变化

学到价值以后，怎样让行动真的变好？

控制的目标是改善策略，不只是降低预测误差。动作价值 $q_\pi(s,a)$ 比状态价值多固定了第一步动作，因此可以直接比较动作。一个完整学习循环包括按行为策略采样、更新动作价值、再让行为依赖新价值。只在固定数据上比较 target，不足以构成这个闭环。

SARSA 把实际选择的下一动作价值接进目标，评价并改善带探索的当前行为。Q-learning 对下一动作取最大值，目标指向贪心控制，即使收集数据时仍在探索。Expected SARSA 则对指定下一策略的动作分布取期望。三者不同在余项，不同的余项决定它们在学习哪个策略。

行为必须提供目标所需的信息。表格 Q-learning 的经典收敛结论要求每个状态动作对持续得到访问、逐对步长满足随机逼近条件等，并非“使用 max 就能收敛”。SARSA 常用探索逐渐消失但访问不停止的条件。有限寿命或环境变化时，保持常数步长和持续探索可能有实际意义，但目标转为跟踪，而非收敛到固定表。

$$
\begin{aligned}y_t^{\rm Sarsa}&=R_{t+1}+\gamma Q(S_{t+1},A_{t+1}),\\y_t^{\rm Q}&=R_{t+1}+\gamma\max_aQ(S_{t+1},a),\\Q(S_t,A_t)&\leftarrow Q(S_t,A_t)+\alpha[y_t-Q(S_t,A_t)].\end{aligned}
$$

下一动作要按与算法一致的时序取得。二者均需处理终止；Q-learning 的行为策略不必等于 target 中的贪心策略。

### 动手与核对

[下载 control_problem_lab.py](https://yingwen.io/zh/continual-rl/download/control_problem_lab.py)

在保存该文件的目录运行：

```sh
python3 control_problem_lab.py control
python3 control_problem_lab.py test
```

预期检查：查看真实采样后的 greedy policy 为 (1,1)，Q(A)≈(1,2.368421)、Q(B)≈(2,2.631579)，并核对访问次数。

实验范围：这个确定性两状态例子用常数步长。成功运行不构成随机环境或深度 Q-learning 的收敛证明。

自测：下一状态 Q 值为 (2,0)，行为以 (0.9,0.1) 选动作，本次实际选了第二个动作；r=0、γ=0.9。三个 target 是多少？

解答：SARSA 为 0；Expected SARSA 为 0.9×1.8=1.62；Q-learning 为 1.8。只有先明确要评价的策略，才能判断哪个 target 合适。

完整教材：

- [控制问题与广义策略迭代](https://yingwen.io/zh/continual-rl/algorithms/control/)

原文、课程与实现：

- [Sutton & Barto · Reinforcement Learning, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：第 6 章及第 4.6 节：sample-based control 与广义策略迭代。
- [Watkins & Dayan · Q-learning](https://doi.org/10.1007/BF00992698)：表格控制及其收敛假设；不能直接外推到神经网络。

## 5 · 多步目标与时间信用分配

延迟到来的反馈，怎样影响更早的预测和决策？

一步 TD 每次只跨一个时间步传播信息。n-step 方法先累计 n 步实际奖励，再接上价值；$\lambda$-return 则按几何权重组合不同长度的目标。更长回报改变 bootstrap 误差、采样方差和反馈延迟，不意味着总是更好。

资格迹把最近访问过的特征方向压缩为一个向量 $e_t$。当前 TD error 到来时，不只更新当前状态，也按迹的大小更新过去相关参数。在线计算的迹不需要存整条轨迹，但它并没有给策略增加关于环境历史的信息。看不到过去线索的策略，不会仅因 critic 使用资格迹就获得记忆。

冻结价值参数后，可以推导前向回报与后向误差累积的恒等关系。参数每步变化时，普通 accumulating traces 并不在有限步长下逐次等于在线前向视图；true-online TD 针对线性近似修正了这个差异。递归网络的参数还通过内部状态影响未来输出，需要另外处理这条导数路径。

$$
e_t=\gamma\lambda e_{t-1}+x_t,\qquad \delta_t=R_{t+1}+\gamma w_t^\top x_{t+1}-w_t^\top x_t,\qquad w_{t+1}=w_t+\alpha\delta_t e_t
$$

这是固定 γ 的 accumulating linear TD(λ)。x 是特征，one-hot 情况对应表格。真实 episode 边界重置资格迹，但保留已学权重。

### 动手与核对

[下载 credit_assignment_lab.py](https://yingwen.io/zh/continual-rl/download/credit_assignment_lab.py)

在保存该文件的目录运行：

```sh
python3 credit_assignment_lab.py online
python3 credit_assignment_lab.py test
```

预期检查：同一例子的普通 TD 最后为 2.0，true-online TD 与在线前向视图均为 1.75；不是浮点误差造成的不同。

实验范围：演示线性在线等价性；不提供非线性神经网络上的一般等价定理。

自测：两步轨迹奖励为 0、1，初值零、γ=.9、λ=.8、α=.1，特征分别为 (1,0)、(0,1)。最后更新是什么？

解答：第一步误差为 0。第二步迹为 (.72,1)，误差为 1，权重变为 (.072,.1)。λ=0 时第一项仍是 0。

完整教材：

- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)
- [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)

原文、课程与实现：

- [Sutton & Barto · Reinforcement Learning, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：第 7、12 章：n-step、前向视图与资格迹。
- [van Seijen & Sutton · True Online TD Learning](https://proceedings.mlr.press/v32/seijen14.html)：理解普通在线迹与 true-online 修正的区别。

## 6 · 函数逼近、采样分布与稳定性条件

不能为每个状态保存一行表格时，怎样共享经验？

线性近似用 $\hat v_w(s)=w^\top x(s)$ 把许多状态映射到共享参数。学习一个状态可能同时改变其他状态。若训练目标是固定回报样本，平方误差的随机梯度为误差乘特征；若目标含下一状态的当前预测，但更新时不对目标求导，则得到 semi-gradient TD。

共享参数意味着不一定能同时精确拟合所有状态。误差按哪些状态更常出现加权，因此数据分布本身成为算法的一部分。on-policy 线性 TD 在固定策略、适当遍历与满秩等条件下有稳定性结果；它通常求的是投影 Bellman 固定点，并非直接最小化真实价值均方误差。步长还要符合相应定理。

当函数逼近、bootstrap 和 off-policy 数据组合时，误差反馈可能发散，这就是 deadly triad 所指的组合风险。它在线性小例子中已会出现，并不是网络太深才产生。特征缩放也会改变同一个数值步长的实际作用。进入深度 RL 前，应先能指出一次更新的目标、梯度和采样分布。

$$
\begin{aligned}w_{t+1}&=w_t+\alpha\delta_t x_t,\\\mathbb E[\delta_t x_t]&=b-Aw_t,\\A&=\mathbb E[x_t(x_t-\gamma x_{t+1})^\top],\qquad b=\mathbb E[R_{t+1}x_t].\end{aligned}
$$

这里的期望来自 on-policy 样本分布；off-policy 版本必须明确重要性比率和状态分布。矩阵 A 的性质决定期望更新的稳定性，不能仅把 TD 当成普通固定标签回归。

### 动手与核对

[下载 gvf_lab.py](https://yingwen.io/zh/continual-rl/download/gvf_lab.py)

在保存该文件的目录运行：

```sh
python3 gvf_lab.py counterexample
python3 gvf_lab.py test
```

预期检查：比较 expected TD 的增长与相应梯度 TD 方法的行为；阅读脚本给出的明确线性反例和固定点，而不只观察训练回报。

实验范围：这是期望更新层面的反例。它不表示所有 off-policy TD 都发散，也不表示某个稳定修正自动解决非线性控制。

自测：把所有特征乘 10，同时将权重除 10，初始预测不变。保持相同 α，预测的单次改变量会不变吗？

解答：不会。TD 权重更新与特征成正比，而预测又乘一次特征；对同一误差，局部预测变化放大 100 倍。表示尺度与步长必须一起考虑。

完整教材：

- [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)
- [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)
- [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

原文、课程与实现：

- [Sutton & Barto · Reinforcement Learning, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：第 9 章线性预测；第 11 章离策略学习与 deadly triad。
- [Spinning Up · 算法分类](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)：按学什么、怎样获取数据和是否使用模型理解方法，不把分类树当成完整历史。

## 7 · 学习与规划的接口

得到一条经验后，除了直接更新价值，还能学习什么？

经验还可以训练后果模型：采取某动作会得到什么奖励，接下来到哪里。Dyna 用同一条真实转移做直接价值学习和模型学习，再选择若干状态动作，通过模型产生模拟 backup。这样把“获得新事实”和“用已有事实继续计算”分开。

模型规划并不免费。一次额外 backup 消耗计算，错误模型也可能反复强化错误价值。需要同时报告真实交互数和模拟更新数。优先扫描根据可能改变价值的大小安排 backup，比均匀分配计算更有针对性；其意义是调度，不是改变控制目标。

从这里可以理解后来的世界模型方法：模型可以是表格、神经网络或潜变量系统，规划可以是价值迭代、轨迹搜索或通过模拟经验训练策略。这些选择有共同接口，却不能只凭“用了模型”就视为同一算法。进入下一册时，应继续区分模型误差、价值误差和优化误差。

$$
\hat y=\hat r(s,a)+\gamma\sum_{s'}\hat p(s'\mid s,a)V(s'),\qquad Q(s,a)\leftarrow Q(s,a)+\alpha[\hat y-Q(s,a)]
$$

在贪心控制中可令 V(s′)=maxₐ′Q(s′,a′)。模型生成一次随机后果时用样本替代上式求和；随机模型不应仅保留最后一次观测。

### 动手与核对

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

在保存该文件的目录运行：

```sh
python3 foundations_detail_lab.py dyna
python3 foundations_detail_lab.py test
```

预期检查：1000 次真实交互对应 5000 次规划更新；向右动作价值接近 (0.6561,0.729,0.81,0.9,1)。

实验范围：环境是小型确定性链，模型不需要处理复杂随机性；这不是世界模型基准。

自测：增加十倍模型 backup 后，用更少交互达到同一回报，是否已经说明算法更高效？

解答：只说明对应交互预算下的采样效率。还要报告模拟计算、模型训练和运行时间；如果模型错误，更多规划甚至可能更差。

完整教材：

- [Dyna 与模型学习](https://yingwen.io/zh/continual-rl/algorithms/dyna/)
- [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)
- [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)

原文、课程与实现：

- [Sutton & Barto · Reinforcement Learning, 第二版](http://incompleteideas.net/book/the-book-2nd.html)：第 8 章：Dyna、模型误差与优先扫描。
- [Sutton · Dyna, an integrated architecture for learning, planning, and reacting](https://doi.org/10.1145/122344.122377)：直接学习、模型学习和规划的共同架构。

下一步：表格算法给每个状态或动作一个独立参数。状态很多时，这种表示无法扩展。现代 DRL 保留评价和控制的目标，却用共享的可学习表示近似它们；这会改变误差传播和稳定性。
