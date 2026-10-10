# 大规模训练：算法与系统怎样共同设计

现代深度强化学习 · 第 7 章

环境、推理和学习并行以后，怎样把更多计算变成更快的策略改善？

## 本章内容

- 沿一段经验追踪行为策略、传输、排队和梯度更新。
- 区分 A3C 的过期梯度、IMPALA 的离策略目标与 PPO 的重复优化。
- 解释 OpenAI Five、SEED RL 和 GEAR 怎样分配计算与移动数据。
- 用数据年龄、采样概率、资源预算和达到目标性能的时间共同评价系统。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [深度 RL 的机制接口：模型、记忆、离线数据与实验](https://yingwen.io/zh/continual-rl/foundations/deep/practice/)：先分清环境采样、推理与学习的数据接口。
- [离策略函数逼近：覆盖、发散与稳定更新](https://yingwen.io/zh/continual-rl/foundations/approximation/off-policy/)：理解异步采样导致的行为策略与当前策略差异。


### actor 与 learner

actor 选择动作并产生经验。learner 用经验改变参数。推理服务可以从 actor 中拆出。这里的角色是系统进程，不必对应不同智能体。

### 策略梯度与 TD

actor 更新用优势加权的动作对数概率梯度。critic 用奖励与后继估计构造目标。先掌握本册策略梯度章即可。

### 三种时间

t 是一条轨迹中的环境步，k 是 learner 的参数更新次数，τ 是墙钟时间。多环境的 t 没有一个天然共享的顺序。

<a id="problem-definition"></a>

## 本章的问题定义

先固定一个有界奖励的折扣 MDP。多个环境副本采样，推理服务选择动作，学习进程更新一个共享策略。随后再讨论环境变化和单生命约束。

### 给定条件与符号

- 环境步、网络推理、反向传播、传输各自的成本。
- CPU/GPU、内存、带宽、墙钟与交互预算。
- 行为策略产生的轨迹，以及每次动作的概率和参数版本。

### 需要求解的对象

设计采样、推理、存储和更新协议，在相同任务与资源条件下更快达到给定策略性能。

### 信息与数据权限

行为时只能使用当时发布的参数。learner 可读入旧轨迹，但不能把旧动作的概率重新标成当前策略概率。

$$
J(\pi)=\mathbb E_\pi\!\left[\sum_{t\ge0}\gamma^tR_{t+1}\right],\qquad T_q=\inf\{\tau:J(\pi_{k(\tau)})\ge q\}
$$

J 定义策略好坏；Tq 定义达到预先规定性能阈值 q 的墙钟时间。实验用独立评估估计 J，并报告不确定性、总交互和硬件小时；未达到阈值的运行也保留。

### 成立条件与解的含义

- 本章 V-trace 推导先采用固定目标策略、固定行为策略及充分动作覆盖；训练中的网络和策略还会变化。
- 数据交换保留奖励、真终止与截断区别、动作掩码、行为概率、序列边界和版本。
- 并行副本属于训练协议；它不是单个智能体在不可回滚现实世界中的同一条生命。

判断准则：同时检查达到阈值的时间、交互数、资源成本和最终性能。吞吐量本身不表示策略改善。

### 适用边界

- 本章的标准库算例核对估计器与调度，不把它当作多机训练性能测量。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 改变信息或数据协议 · [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)：策略改善目标可保持不变，生成和消费经验的协议已改变。

- 组合不同学习问题 · [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)：多步目标还要处理策略滞后与序列截断。

- 改变信息或数据协议 · [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)：大量副本、回放与集中学习是额外权限；流式单生命条件未必允许。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

加快采样可能使队列更长；加快更新可能使行为策略更旧；增大批量可能减少通信，却延迟下一次行动。

### 本章的核心思路

为每段数据保留产生条件，并让资源调度与算法允许的数据分布相匹配。

1. [先确定瓶颈与时钟](#systems-clocks)：分别测量产生经验、动作推理、消费数据和发布参数的速率。

2. [处理策略差异，而不只搬运张量](#systems-vtrace)：A3C 在本地求梯度；IMPALA 在 learner 重新求梯度，并校正行为与目标动作分布。

3. [存储布局也有算法含义](#systems-gear)：数据必须完整提交；分片选择必须保持指定的全局采样规则。

结论与条件：V-trace 的固定策略表格结论与深度控制的训练表现分开。系统速度只在明确硬件、任务、预算和指标后比较。

### 相关方法改变了什么

- 同步采样与更新：更容易控制策略版本，但可能等待慢环境和空闲设备。

- 异步采样与集中学习：提高流水重叠；需要控制数据年龄、离策略偏差和样本复用。


<a id="lesson-setting"></a>

## 1 · 把一个训练循环拆开，会改变什么？

先考虑一台机器：策略选择动作，环境推进一步，经验进入缓冲区，学习器做一次更新。四个动作顺序执行时，代码很容易知道当前参数来自哪次更新。现在增加一百个环境。有的正在等待动作，有的刚结束一段轨迹，有的正在传输数据；与此同时，学习器可能已经更新了十次。先前正确的“当前策略”不再指同一个对象。

大规模 RL 的基本困难由这个时间差产生。我们希望 CPU 不等待 GPU，也希望 GPU 不等待 CPU；但越充分地重叠这些工作，越需要区分行为时的参数和学习时的参数。因此，分布式训练不只是把一个优化器复制到更多机器上。

![环境副本产生观测，推理设备返回动作，轨迹经队列送入学习设备；参数沿反方向发布。不同颜色的轨迹带有不同策略版本。](https://yingwen.io/crl-figures/systems-pipeline.svg)

原创系统示意。蓝色实线搬运观测、动作和经验；橙色虚线发布参数。矩形中的小方格表示批处理，不表示真实硬件数量。推理可以在 actor 本地，也可以成为集中服务。

本章接续 [DQN 的三个更新时钟](/zh/continual-rl/foundations/deep/deep-value/#lesson-setting) 与 [PPO 的旧策略数据](/zh/continual-rl/foundations/deep/trust-region/)。要追踪的不仅是 target network 何时复制，还包括动作究竟由哪个策略产生、经验在何处等待、重复使用了几次。

<a id="systems-clocks"></a>

## 2 · 吞吐量、数据年龄与复用率

$$
\begin{gathered}F=\text{每秒新产生的转移数},\quad U=\text{每秒 learner 更新数}\\D=B\,U,\qquad r_{\rm reuse}=D/F\end{gathered}
$$

B 是每次更新实际用于 loss 的转移数。D 的单位是“转移使用次数/秒”。burn-in、padding 与丢弃的转移不应算成新训练样本。rreuse 是长期速率比，不保证每个样本都恰好使用相同次数。

若每秒产生 2,000 条转移，每次更新消费 256 条、每秒更新 10 次，则平均复用率是 1.28。若只增加 learner 而不增加采样，复用率进一步升高。GPU 可以更忙，但重复优化旧数据与获得新经验不是一回事。反过来，生产速度超过消费速度时，无界队列会不断积压；有界队列则必须阻塞生产者或丢弃数据。

$$
\Delta\tau=\tau_{\rm use}-\tau_{\rm action},\qquad \Delta k=k_{\rm use}-k_{\rm behavior}
$$

$\Delta\tau$ 是秒数；$\Delta k$ 是参数更新差。若“发布版本”每 32 次更新才递增一次，就还需记录版本号到更新次数的映射。相同版本差不保证相同策略差异。

动作分布的变化还取决于步长、梯度和表示。可在收到的状态上记录 KL、动作概率比及裁剪比例；这些局部诊断也不能代替整个状态访问分布的比较。一个更新很小的旧版本，可能比刚刚发生剧烈改变的新版本更接近当前策略。

![每个时钟刻度到达两段轨迹，消费一段。无界 FIFO 的等待量与数据年龄增长；容量为二的队列丢弃最旧段，使被消费段更新，但损失一部分经验。](https://yingwen.io/crl-figures/systems-queue.svg)

确定性调度算例，每格是一段经验，每列是一个抽象时钟刻度。先入队，再按容量丢弃最旧项，最后消费一项。颜色代表产生刻度；没有环境训练，也没有测得的加速比。

队列只能吸收短时波动，不能消除长期供需不平衡。一个简化吞吐上界是环境总推进能力、总推理能力、传输能力与 learner 消费能力的最小值；真实流水还受共享带宽、同步和负载不均影响。优化不在瓶颈上的模块，常常只让下一个队列更长。

<a id="systems-a3c"></a>

## 3 · A3C：让多个采样与学习进程异步工作

[A3C](https://proceedings.mlr.press/v48/mniha16.html) 把环境交互、策略推理和梯度计算放在各个 worker。worker 先读取共享参数，在本地运行一段短轨迹，用本地参数计算多步 actor–critic 梯度，再将梯度应用到共享参数。不同环境提供不同轨迹，不必等待一个大的同步采样屏障。

$$
G_t^{(n)}=\sum_{i=0}^{n-1}\gamma^iR_{t+i+1}+\gamma^n V_{w^-}(S_{t+n}),\qquad \widehat A_t=G_t^{(n)}-V_{w^-}(S_t)
$$

这里 θ⁻、w⁻ 是该 worker 本段采样与求导使用的本地参数。遇到真终点时尾值为零；同一短轨迹中的 n 可以不同。优势在 actor loss 中作为固定量。

$$
g^- =\sum_t\!\left[\widehat A_t\nabla_{\theta^-}\log\pi_{\theta^-}(A_t\mid S_t)+\beta\nabla_{\theta^-}\mathcal H(\pi_{\theta^-}(\cdot\mid S_t))\right]
$$

局部 actor 梯度 $g^-$ 被提交时，共享参数可能已由别的 worker 修改。对本地行为而言接近 on-policy，并不意味着这个过期梯度就是共享当前参数的无偏梯度。

关键改变是提交梯度的时机。更多 worker 可以减少单条轨迹的相关性影响，却也增加参数滞后。共享优化器统计、梯度范数和同步周期都会影响结果。它们不是经验回放或重要性采样的另一种名字。A2C 则等待一组 worker 完成规定长度后统一更新，牺牲部分异步重叠以简化版本管理。

**算法：A3C 的事件次序；不展开并发锁与优化器实现**

1. 各 worker 读取共享参数，保存本地版本。
1. 用本地策略交互至 n 步或真终点。
1. 从最后状态的尾值向前计算多步目标。
1. 在本地参数处计算 actor、critic 和熵项梯度。
1. 将梯度提交到共享优化器；随后刷新本地参数。

<a id="systems-impala"></a>

## 4 · IMPALA：提交轨迹，而不是提交过期梯度

[IMPALA](https://proceedings.mlr.press/v80/espeholt18a.html) 将 actor 与 learner 分开。actor 用取得的参数产生一段轨迹，发送观测、动作、奖励和行为策略信息。learner 汇集多段轨迹，在加速器上用自己的当前网络计算目标和梯度。这样可以把小规模分散的反向传播变成较大的批处理。

梯度不再由旧 worker 计算，但经验仍由旧策略产生。若动作来自 μ，更新却要评价 π，就需要处理离策略问题。保存的行为概率必须对应实际采样动作时的分布，包括探索、温度和动作掩码。若推理服务在同一段内切换版本，最好逐步保存 log μ，而不是只给整段贴一个版本号。

$$
\mathbb E_{A\sim\mu(\cdot\mid s)}\!\left[\frac{\pi(A\mid s)}{\mu(A\mid s)}f(s,A)\right]=\mathbb E_{A\sim\pi(\cdot\mid s)}[f(s,A)]
$$

把左边展开为对动作求和，μ 与分母相消。条件是 π 的动作支持包含在 μ 的支持中。这个等式固定状态 s；它没有把行为轨迹中的状态分布同时变成目标策略的状态分布。

如果每一步都乘未截断的重要性比，长轨迹上的权重可能极大。IMPALA 的 V-trace 不直接用整个轨迹权重乘一个回报，而是分别控制当前 TD 误差的权重与误差向更早状态传播的权重。

<a id="systems-vtrace"></a>

## 5 · V-trace：从一步误差到多步校正

$$
\begin{gathered}q_t=\pi(A_t\mid S_t)/\mu(A_t\mid S_t),\quad \rho_t=\min(\bar\rho,q_t),\quad c_t=\min(\bar c,q_t)\\\delta_t^V=\rho_t\bigl(R_{t+1}+\gamma_t V(S_{t+1})-V(S_t)\bigr)\end{gathered}
$$

ρ 控制本步 TD 误差；c 控制更晚误差能够传播多远。这里 γt 表示当前转移的折扣，真终点为零；它对应其他章节常用的 γt+1。常用设置为两种上限均为 1。

$$
v_s=V(S_s)+\sum_{t=s}^{T-1}\left(\prod_{i=s}^{t-1}\gamma_i c_i\right)\delta_t^V
$$

空乘积等于一。读成：先保留当前估计，再累加后面各步经过衰减的修正。ρ 与 c 不承担同一种工作。

把求和第一项拆出，再把其余项提取 γs cs，就得到可以逆序实现的递推。轨迹末尾没有更多修正，令 vT=V(ST)。所以实现只需一个从右向左扫描的累加器，不必为每个起点重算整段乘积。

$$
v_s=V(S_s)+\delta_s^V+\gamma_s c_s\bigl(v_{s+1}-V(S_{s+1})\bigr)
$$

当 μ=π 且两种上限不小于一，ρ=c=1。相邻 V 项相消，所得就是尾部 bootstrap 的 n 步回报。若到达真终点，γs=0，后续片段不应传播回来。

$$
\pi_{\bar\rho}(a\mid s)=\frac{\min(\bar\rho\mu(a\mid s),\pi(a\mid s))}{\sum_b\min(\bar\rho\mu(b\mid s),\pi(b\mid s))}
$$

在论文的固定策略表格条件下，截断 ρ 所评价的是这个归一化策略。提高 ρ 上限可使它更接近 π，却增大权重波动；c 的截断改变误差传播，不以同样方式改变这个固定点。

这一固定点结论先固定环境与两种策略，取 $0\le\gamma<1$、$0<\bar c\le\bar\rho$，并要求每个状态下 $\mathbb E_\mu[\rho_t\mid S_t=s]\ge\beta>0$。在这些条件下，期望 V-trace 算子是收缩映射，见[原文补充材料 A.1](https://proceedings.mlr.press/v80/espeholt18a/espeholt18a-supp.pdf)。本章代码进一步限制 $\bar c\le1$。随机表格更新收敛还需访问覆盖与步长条件；非线性网络上最小化一批标签误差，不直接继承该收敛结论。

最小反例是一次行动后终止的二臂任务。μ=(.8,.2)，π=(.2,.8)，两臂奖励为 0 和 1。ρ 上限为 1 时，两臂的有效质量分别为 min(.8,.2)=.2、min(.2,.8)=.2，归一化后为 (.5,.5)。价值固定点是 .5，不是目标 π 的 .8。“裁剪后仍完全无偏”在这个最小任务上就不成立。

$$
\widehat A_s^{\rm VT}=\rho_s^{\rm PG}\bigl(R_{s+1}+\gamma_s v_{s+1}-V(S_s)\bigr),\qquad g_s=\operatorname{sg}(\widehat A_s^{\rm VT})\nabla_\theta\log\pi_\theta(A_s\mid S_s)
$$

actor 的概率比上限可以另设。critic 拟合 stop-gradient 的 vs；actor 使用下一步的修正值。这里的 actor 更新不是对 vs 的整条计算图求导，也不是深度控制收敛定理。

<a id="lesson-example"></a>

## 6 · 两次转移，把公式算到底

取轨迹 S0→S1→真终点，奖励依次为 1、2，折扣为 .9、0。当前网络的三个值为 .4、.8、0；记录动作的行为概率均为 .5，当前概率依次为 .25、.75。两种截断上限均为 1。

$$
\begin{gathered}q=(0.5,1.5),\quad\rho=c=(0.5,1)\\\delta_0^V=0.5(1+0.9\times0.8-0.4)=0.66\\\delta_1^V=1(2-0.8)=1.2\\v_1=2,\qquad v_0=0.4+0.66+0.9\times0.5\times1.2=1.6\end{gathered}
$$

不做校正的两步回报是 2.8。这里只对同一条给定轨迹比较标签，不能把 1.6 与 2.8 的大小直接解释为谁更准确。真正的准确性需要指定策略价值并对采样分布求期望。

![三个状态按时间排列，局部 TD 修正为.66和1.2；后一个修正以.9乘.5回传给第一个状态，得到目标1.6。](https://yingwen.io/crl-figures/systems-vtrace.svg)

精确算例。实线为环境转移，橙色反向箭头为数值修正传播。状态下方的旧值与新标签分别显示，标签还不是已经更新的网络输出。

actor 在第一步使用 .5×(1+.9×2−.4)=1.2。本例中 ρ、c 与 actor 权重相同，因此它在代数上也等于 v0−V(S0)。若改变这些截断上限，两者便未必相同。先分别定义 critic 的回归标签与 actor 的优势，再检查梯度路径。

<a id="systems-inference"></a>

## 7 · 集中推理：批量变大，单次动作会更快吗？

很多环境的推进适合 CPU，网络前向传播适合 GPU。若每个 actor 各自用 CPU 推理，网络越大，CPU 越可能主要忙于推理而不是环境。集中推理把不同环境的观测组成一个批次，完成前向传播后，再把各自动作发回。

$$
\text{推理吞吐}=\frac{b}{L_{\rm infer}(b)},\qquad L_{\rm action}=L_{\rm wait}+L_{\rm net}+L_{\rm infer}(b)+L_{\rm return}
$$

b 为推理批大小。吞吐按整批除以整批服务时间计算；单个环境等待动作的时间还包括凑批、网络与返回。提高吞吐与降低延迟是两个目标。

一个环境必须先获得动作才能产生下一观测，不能仅靠增大 batch 凭空增加并行请求。增加环境副本可以填满批次；设置最大等待时间可以限制凑批延迟。慢环境、长序列、不同动作空间和 recurrent state 都要求正确路由。把两个环境的隐藏状态交换，即使张量形状正确，也已改变策略。

[SEED RL §3](https://arxiv.org/html/1910.06591#S3) 把推理和轨迹积累移到 learner 侧，使用持续 RPC 连接和批处理减少每步远程调用开销，并提供 V-trace 与 R2D2 两条算法路径。它减少了向每个 actor 反复传输大模型的需求，但每个环境步仍依赖一次观测—动作往返。

集中推理可以缩短参数同步距离，不能保证 learner 消费轨迹时仍与行为策略一致。多步收集、排队和反向传播都需要时间。若部署的是实时控制器，还要检查动作截止时间；仿真能暂停等待，真实世界往往不能。

<a id="systems-five"></a>

## 8 · OpenAI Five：训练规模与数据新鲜度一起设计

[OpenAI Five](https://cdn.openai.com/dota-2.pdf) 将 Dota 2 自对弈拆成 CPU 游戏进程、GPU 推理池、GPU 优化器与参数控制器。优化器同步聚合梯度；游戏与数据上传异步进行。它采用 PPO 与 GAE，输入为结构化游戏观测，而不是把游戏画面直接输入网络。

这里有一个容易混淆的区别：优化器之间同步，不等于采样与优化之间同步。游戏不断推进，推理服务定期拿新参数，优化器从不断补充的缓冲区取数据。于是可以同时存在同步的数据并行 SGD 与离策略程度不为零的经验流。

论文 Appendix M 分别改变数据延迟和样本复用。加入发送队列会延缓学习；减少采样资源、提高复用也损害其设定下的表现。这说明只加优化器、反复使用同一批经验，不必然更快。改变 rollout 数还会改变并行游戏的多样性，所以复用消融并未完全隔离所有因素。

PPO 的比率分母应来自定义该批数据的参考策略。在异步数据中，如果行为策略与这个参考策略不同，不能仅靠一个“old log-prob”变量名抹去差别。即便比率使用真实行为概率，clipping 也不会校正所有状态分布偏差、过期优势或环境版本变化；限制延迟仍有价值。

自对弈还引入另一种非平稳性：对手分布在变。某段经验的新鲜度不仅由本方参数决定，还由当时的对手、游戏版本和奖励定义决定。将这些元数据保存下来，才能分清性能变化来自学习规则、系统调度还是学习目标的变化。

这是一项大规模训练成果。它的长期训练与人工进行的网络、环境接口调整，为知识迁移提供了经验；但多副本、自对弈重启和工程师调整仍是重要条件，不能直接等同于单生命持续学习已经得到解决。

<a id="systems-gear"></a>

## 9 · GEAR：大模型也可能在等待经验

当输入变成较长序列，模型采用多 GPU 数据并行或流水并行时，训练数据本身的移动会成为瓶颈。传统路径可能经过 CPU 采样、对象序列化、网络传输、主机缓存和设备复制；学习器有算力，却拿不到足够快的数据。

[GEAR](https://proceedings.mlr.press/v202/wang23aj.html) 在 GPU 服务器的主机内存组织按字段分列的轨迹分片。索引与数据载荷分开管理；先分配、写入，再提交为可选择。集中选择支持全局均匀或加权采样；分散选择可先局部筛选，再合成全局确定性的 FIFO 或 Top-K 结果。

其本地读取路径利用 GPU 对 pinned 主机内存的直接访问，经 PCIe 收集所需字段；论文的远程收集通过 NCCL 和 InfiniBand。GPU-centric 不表示所有经验一直驻留显存，也不表示没有物理传输。目标是减少不必要的复制、序列化和 CPU 调度，并与模型并行所需的数据位置匹配。

先考虑不依赖具体硬件的正确性问题。若两个分片分别有 1 和 3 条有效经验，先以相同概率选分片，再在分片内均匀选一条，四条经验的概率就是 .5、1/6、1/6、1/6。这不等于全局均匀采样。按有效条数 1:3 选择分片，才使每条概率都为 .25。

$$
P(i)=P(j)P(i\mid j),\qquad P(j)=\frac{W_j}{\sum_\ell W_\ell},\quad P(i\mid j)=\frac{w_i}{W_j}\ \Longrightarrow\ P(i)=\frac{w_i}{\sum_\ell W_\ell}
$$

j 是经验 i 所在分片，Wj 是该分片有效经验权重和。若先按相同概率选机器，只有在各分片总权重相同的特殊情况下，才仍得到所需的全局加权分布。

这又给出两种不同的校正：1/(N P(i)) 修正回放索引抽样相对于均匀经验集的偏差；π(a|s)/μ(a|s) 修正给定状态下行为动作与目标动作的差别。前者不能替代后者。Top-K 选择还会把一些经验的入选概率变为零，不能套一个有限的重要性权重就恢复被排除的总体。

模型并行还有一个独立约束：同一次训练样本的各字段、各流水阶段必须指向同一批轨迹与时间位置。若一个设备读到轨迹 i 的观测，另一个设备却用轨迹 j 的动作标签，得到的是错误训练样本，而不是普通的离策略数据。为未提交或已回收内存设置不可选择状态，解决的是这一类系统正确性。

<a id="systems-sequences"></a>

## 10 · 序列、终止与环境速度带来的隐藏偏差

增加资源时，先检查“一个样本”究竟是什么。recurrent 网络需要序列边界和初始 hidden state。保存的 hidden state 由旧参数计算；新参数重新展开同一序列，未必得到同一状态。R2D2 将这种表示漂移与 recurrent state staleness 单独研究，采用存储状态与 burn-in 等机制。预热只为后缀提供起点，是否对它反向求导还要看实现；在线网络与目标网络也需要按各自参数展开。[同一仓库轨迹的逐步比较](/zh/continual-rl/foundations/deep/partial-observability/#recurrent-replay-state)展示状态版本怎样改变动作和 bootstrap 标签。

- 真终止、时间上限和通信切块不同：切块仍需尾值。bootstrap 用截断前最后观测，不用 reset 后下一局的初始观测。
- padding 只为批处理对齐；对应 loss 和跨边界递推需要 mask。burn-in 用来重建状态，通常不作为同样的损失区间。
- 较快的环境可能贡献更多样本。若任务混合目标原本要求等权，这会改变训练任务分布；吞吐调度也在隐式选择课程。
- 动作概率必须包含实际推理时的动作掩码。某动作当时不合法，而现在合法，不能在旧轨迹上假装它已获得行为覆盖。

这些问题不会被 V-trace 自动修复。重要性比需要正确输入；它不修复串错的序列、错误终止标签、已经变化的转移规律或缺失的历史。先建立数据契约，再讨论哪种估计器适用。

<a id="lesson-code"></a>

## 11 · 从数值核验进入作者实现

标准库 V-trace：从固定 learner 快照计算目标；反向传播时将返回值视为常量

```python
def vtrace(rewards, discounts, values, behavior_probs, target_probs,
           rho_cap=1.0, c_cap=1.0, pg_cap=1.0):
    """One unpadded trajectory of T transitions; values has T+1 entries.

    Values and target probabilities use the same fixed learner snapshot. A true
    terminal transition has discount 0. A rollout cut keeps its discount and
    supplies the bootstrap value. Behavior probabilities are saved at action
    selection, not recomputed using current parameters. Returned targets are
    constants (stop-gradient) when used in an autodiff actor-critic loss.
    """
    T = len(rewards)
    if T < 1 or len(values) != T + 1 or any(
        len(a) != T for a in (discounts, behavior_probs, target_probs)
    ):
        raise ValueError("one probability/discount per transition; T+1 values")
    arrays = (rewards, discounts, values, behavior_probs, target_probs)
    if not all(math.isfinite(x) for a in arrays for x in a):
        raise ValueError("inputs must be finite")
    if not all(0 <= d <= 1 for d in discounts):
        raise ValueError("discounts must lie in [0,1]")
    if not all(0 < p <= 1 for p in behavior_probs):
        raise ValueError("a recorded action must have positive behavior support")
    if not all(0 <= p <= 1 for p in target_probs):
        raise ValueError("target action probabilities must lie in [0,1]")
    if not all(math.isfinite(c) and c > 0 for c in (rho_cap, c_cap, pg_cap)):
        raise ValueError("caps must be positive and finite")
    if c_cap > min(1.0, rho_cap):
        raise ValueError("this implementation uses c_cap <= min(1, rho_cap)")
    ratios = [p / mu for p, mu in zip(target_probs, behavior_probs)]
    rhos = [min(rho_cap, r) for r in ratios]
    cs = [min(c_cap, r) for r in ratios]
    deltas = [rhos[t] * (rewards[t] + discounts[t] * values[t+1] - values[t])
              for t in range(T)]
    targets = [0.0] * T
    correction = 0.0  # At the rollout boundary v_T = V_T.
    for t in reversed(range(T)):
        correction = deltas[t] + discounts[t] * cs[t] * correction
        targets[t] = values[t] + correction
    next_targets = targets[1:] + [values[-1]]
    advantages = [min(pg_cap, ratios[t]) * (
        rewards[t] + discounts[t] * next_targets[t] - values[t])
        for t in range(T)]
    return {"ratios": ratios, "deltas": deltas,
            "targets": targets, "pg_advantages": advantages}
```

运行八项核验，再打印逐步结果

```bash
python3 distributed_systems_lab.py test
python3 distributed_systems_lab.py demo
```

测试分别检查手算值、on-policy 望远镜相消、正向求和与逆序递推一致、真终点与切块的区别、ρ 裁剪固定点、队列守恒、分片概率，以及非法输入。代码不启动环境或 GPU 训练。要研究系统性能，需要下一节的测量协议。

| 研究问题 | 原始实现入口 | 沿什么路径读 |
| --- | --- | --- |
| 集中学习与校正 | [DeepMind scalable_agent](https://github.com/google-deepmind/scalable_agent) | 先看 experiment.py 中数据流，再看 vtrace.py 的 from_importance_weights；注意 value target 与 policy-gradient advantage 分开返回。 |
| 集中推理 | [Google Research SEED RL](https://github.com/google-research/seed_rl) | 跟踪 actor 请求、learner 推理、序列积累与 V-trace/R2D2 更新；它不是把 actor 进程简单搬到 GPU。 |
| 轨迹存储与收集 | [GEAR 作者仓库](https://github.com/bigrl-team/gear) | 从 examples 与 api.md 进入分配、提交、选择、收集；README 说明公开版本仍有功能范围限制，复现前固定版本与硬件。 |
| 长时自对弈训练 | [OpenAI Five 论文与附录](https://cdn.openai.com/dota-2.pdf) | Figure 2 看设备分工，Appendix M 看 staleness、reuse 与同步消融；本章没有将其完整训练平台标为已开放代码。 |

<a id="systems-evaluation"></a>

## 12 · 怎样知道系统改动真正有用？

先做固定数据的估计器核验，再做系统微基准，最后比较完整学习。三者分别回答“算对了吗”“搬运和计算有多快”“策略是否更快改善”。在这三个阶段都使用同一条吞吐指标，会掩盖不同层次的失败。

| 比较 | 固定什么 | 同时记录什么 |
| --- | --- | --- |
| 增加 actor 数 | 网络、任务、learner 配置 | 新转移/秒、独立轨迹数、队列长度与年龄分位数、行为/目标策略差异。 |
| 增大 inference batch | 环境总数与推理模型 | 推理吞吐、端到端动作延迟 p50/p95、环境等待比例、丢失截止时间比例。 |
| 增加 learner 或 replay 复用 | 数据来源与总交互预算 | 有效 loss 样本数、每条经验使用次数、裁剪比例、达到阈值的交互数与时间。 |
| 更换 replay 系统 | 抽样规则、序列长度、字段、精度 | 读取字节/秒、实际 batch 就绪延迟、主机/设备内存、网络流量、抽样分布。 |
| 完整系统 | 评价对手或任务集、评估预算 | 多种子学习曲线、失败运行、墙钟、CPU/GPU 小时与总环境成本。 |

硬件利用率低时，先判断是在等待环境、数据、通信、同步屏障，还是核函数过小。利用率升高后，还要检查有效学习工作有没有增加。多跑 padding、重复编码同一批过期序列或反复裁剪掉更新，都会让设备更忙而策略不变好。

完整对照可从单机同步 actor–critic 开始：固定算法，分别加入异步队列、策略校正、集中推理和存储优化。控制一次只改变的机制，也保留最佳整体系统对比；前者解释原因，后者回答实际是否值得采用。

<a id="lesson-branches"></a>

## 13 · 到持续强化学习，哪些条件要重新讨论？

大规模训练常用重启、平行副本和长期回放换取效率。若目标是一个不可回滚世界里的持续控制，就要重新分配这些权限。物理环境的每秒变化不能由训练器暂停；错过一次行动的代价，也不只是队列中少一个样本。

即使可以并行训练，旧经验是否有用仍是学习问题。如果变化只来自行为策略，动作概率校正可能适用；如果奖励定义或动力学变了，同一个 (s,a) 已有不同后果，就需要辨别变化、重加权或维护模型。继续加大重要性比不能恢复不存在的数据。

这也把系统问题接到持续学习的核心：在有限每步预算下，哪些经验值得保留，哪些预测值得更新，何时应把计算用于新交互、回放或规划？可以沿 [流式学习](/zh/continual-rl/algorithms/streaming/)、[在线信用分配](/zh/continual-rl/algorithms/credit-assignment/) 与 [完整智能体](/zh/continual-rl/construction/architectures/) 继续研究。

<a id="lesson-check"></a>

## 14 · 思考与检验

- 若 actor 数翻倍，学习速度却降低：先查新经验速率是否真的增加，再查队列年龄、参数差异和任务混合是否变化。只看 GPU 利用率无法区分这些原因。
- 若 μ=π，但回放来自旧环境版本：V-trace 比率全为一，仍无法修正动力学失配。请给一个奖励反转的二臂反例。
- 把算例中的 ρ 上限增大到 2、保持 c 上限为 1：重新计算两个 target。答案为 v1=2.6，v0=1.87；这只是同一条轨迹上的标签变化。
- 如果两分片大小为 1 和 3、权重各为 1，应如何选择分片？答案为 1/4 与 3/4；若先均匀选机器，得到的是另一个目标分布。
- 集中推理若将吞吐翻倍却把动作延迟提高十倍，何时仍值得采用？可暂停的仿真训练与有控制截止时间的现实部署需要不同判断。
- 研究题：给定总计算与真实交互预算，是否存在一个依赖变化速率的最优数据年龄或复用范围？分别操纵策略变化与环境变化，避免把两个失配合成一个“陈旧度”。

## 从本章进入实践

[预测与控制](https://yingwen.io/zh/continual-rl/code/#practice-prediction)：学会预测更多事情，什么时候会改变行动？



<a id="chapter-code"></a>

## 下载与运行

标准库精确核验：V-trace、队列年龄与分片概率。不启动训练，也不测量真实集群性能。

[下载 distributed_systems_lab.py](https://yingwen.io/zh/continual-rl/download/distributed_systems_lab.py)

```sh
python3 distributed_systems_lab.py test
python3 distributed_systems_lab.py demo
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Mnih et al. (2016) · Asynchronous Methods for Deep Reinforcement Learning](https://proceedings.mlr.press/v48/mniha16.html)：§4 与 Algorithm S2：本地短轨迹、本地梯度与共享异步更新。

- [Espeholt et al. (2018) · IMPALA](https://proceedings.mlr.press/v80/espeholt18a.html)：§3 的 actor–learner 架构；§4 的 V-trace、固定点及 actor 更新。

- [DeepMind · scalable_agent / vtrace.py](https://github.com/google-deepmind/scalable_agent/blob/master/vtrace.py)：原始 V-trace 实现；对照逆序修正、bootstrap 与独立的 PG 截断。

- [Berner et al. (2019) · Dota 2 with Large Scale Deep Reinforcement Learning](https://cdn.openai.com/dota-2.pdf)：§3.2、Figure 2、Appendix M：设备分工、数据新鲜度、复用与同步比较。

- [Espeholt et al. (2020) · SEED RL](https://arxiv.org/abs/1910.06591)：§3：将推理集中到 learner，处理每步 RPC 与批处理延迟。

- [Google Research · SEED RL](https://github.com/google-research/seed_rl)：作者系统实现，含 V-trace 与 R2D2。

- [Wang et al. (2023) · GEAR](https://proceedings.mlr.press/v202/wang23aj.html)：§3：分列轨迹、索引管理、分布式选择与 GPU-centric 收集。

- [GEAR · 作者实现](https://github.com/bigrl-team/gear)：论文系统的公开工程入口；检查版本功能、CUDA/NCCL 与硬件条件。

- [Kapturowski et al. (2019) · Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/forum?id=r1lyTjAqYX)：R2D2：分布式回放中的表示漂移、recurrent state staleness 与 burn-in。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-systems#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-systems#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-deep-systems)
<a id="study-connections"></a>

## 与教材主线的衔接

本章提供一组可复用的算法工具；基础阅读顺序不是问题类别的互斥划分。

### 估计哪种策略的价值，又按什么状态分布加权？

行为策略决定怎样获得经验；目标策略决定要预测哪种行为。重要性比可以校正给定状态下的动作分布，但不会自动把状态出现的频率改成目标策略的频率。

函数逼近与深度方法：Replay 还引入缓冲区的时间组成与抽样规则。神经更新受到数据分布、共享梯度和移动目标共同影响。重复旧数据与逐条使用新数据有不同的资源和适应代价。

持续学习中的研究问题：单一行为流怎样支持许多预测和技能？在固定内存下，怎样权衡覆盖、样本年龄、更新方差与适应速度，而不把离策略修正当作完整稳定性保证？

[离策略稳定性](https://yingwen.io/zh/continual-rl/foundations/approximation/off-policy/) → [数据与训练接口](https://yingwen.io/zh/continual-rl/foundations/deep/practice/) → [大规模系统与策略滞后](https://yingwen.io/zh/continual-rl/foundations/deep/systems/) → [离线数据的覆盖](https://yingwen.io/zh/continual-rl/foundations/deep/offline/) → [流式更新](https://yingwen.io/zh/continual-rl/algorithms/streaming/)


### 可进一步检验的问题

- [15 · 状态、知识、技能与规划怎样共同构成持续智能体？](https://yingwen.io/zh/continual-rl/research/#research-integrated-architecture)：行为、学习与参数版本各有时间顺序；检查这些接口后，才能研究多个持续更新的模块能否协同改善行为。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [深度价值学习](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)
- [策略梯度与 actor–critic](https://yingwen.io/zh/continual-rl/algorithms/policy/)
- [时间信用分配与资格迹](https://yingwen.io/zh/continual-rl/algorithms/credit-assignment/)
- [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)
- [实验设计、统计与算法测试](https://yingwen.io/zh/continual-rl/experiments/)

对应原始材料：Sutton & Barto §12.8、§13.5、§16.5；A3C §4 与 Algorithm S2；IMPALA §3–4；OpenAI Five §3.2 与 Appendix M；GEAR §3。本文为原创讲解，原书、论文与上游代码保留各自许可。
