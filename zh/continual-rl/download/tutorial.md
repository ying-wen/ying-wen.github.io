# 持续强化学习：从基础 RL 到研究起步

更新：2026-09-30。

按八个具体问题循序进入 CRL，从概念和实验出发，再深入代表论文。

## 01 从“学会一个任务”到“活着时一直学习”

先修：知道状态、动作、奖励与策略。建议时间：25 分钟。

学习产出：能用自己的话写出一个 CRL 问题，并说清允许的重置与记忆。

### 熟悉的 RL 没有错，只是还少了一份“生命史”

你熟悉的实验通常是：选一个环境，训练 DQN 或 PPO，画回报随训练步数的曲线。现在把横轴拉长：机器人已经工作一年，货架被挪动、轮子变旧、订单变了，但充电和安全规则仍要遵守。它不仅要把今天的任务学好，还要决定哪些旧经验值得保留、哪些该修正。

### 只增加一个变化

先不要同时改变图像、动力学和奖励。设机器人有两条配送路线：前半段 A 较快，后半段 B 较快。你可以仍用最熟悉的 bandit 或 Q-learning；算法不新，问题变成了“如何及时跟踪”。如果它到 B 后再也回不来，则又增加了 single-life 的约束。一次只增加一个困难，才能知道失败在哪里。

### 四个容易混用的词

Continuing：任务没有自然终点；Online：随数据到来更新，可以包含回放；Streaming：本教程实验指每条新数据做增量更新、无经验回放；Continual：研究长期学习、适应与积累。一个固定环境中的持续任务可以是平稳的，一个按任务切换的 CRL 基准可以允许重置。它们不是同义词。

### 先写协议，再找算法

拿纸写六项：什么发生变化、谁能观察到变化、能否重置、保留多少记忆、优化哪个长期目标、怎样测新学习。若任务 ID 直接告诉网络，就不要把结果描述成“自动发现变化”；若每个任务重新初始化优化器，也应明确记录。

### 具体例子

旧实验：固定 CartPole，训练后测试。最小持续改造：不中断参数学习，按预先指定时间改变杆长，智能体看不到变化时间。更严格改造：不提供任意状态重置，失败后也由智能体恢复。这些改造逐步增加变化、信息约束和生存约束，不应在同一张榜单上混为一谈。

### 术语小补丁

- **任务变化**：奖励规则、动作后果或可见信息随经历发生改变。
- **重置 reset**：把环境重新放回某个起点；不是把学习器参数清零。
- **学习器状态**：除模型权重外，还包括价值估计、优化器统计、记忆等会影响以后学习的量。

### 推导：把“环境变了”写成一个最小数学例子

只改奖励规律就够了，不需要先引入新的深度网络。

符号：t 是全局交互步；A、B 是两个动作；μₜ(a) 是动作 a 在时刻 t 的期望奖励。

前 1,000 步，A 更好；从第 1,001 步起，B 更好。

$$
(\mu_t(A),\mu_t(B))=\begin{cases}(0.8,0.2),&t\le1000\\(0.2,0.8),&t>1000\end{cases}
$$

以完整 2,000 步内的期望平均奖励为这个教学实验的目标。恢复期间的损失也算数。

$$
J_T=\frac{1}{T}\mathbb{E}\!\left[\sum_{t=1}^{T}R_t\right],\qquad T=2000
$$

例子：如果前后两段始终选 A，期望平均奖励是 (0.8+0.2)/2=0.5；知道切换时刻并立即换动作的参照是 0.8。但学习器并不知道这个时刻。

适用条件：这里只构造奖励变化的 bandit，不把它当作 CRL 的全部定义，也不把知道未来的参照当作可实现算法。

[配套来源：CRL 综述 · 第 4 节：问题设定](https://arxiv.org/html/2012.13490v2#S4)

### 理解与操作练习

问题：货架位置不变，但配送奖励改变了，需要把整个 agent 清零吗？

提示：区分仍然有用的路线知识与已经过时的收益估计。

参考答案：不必。地图或已有技能仍可能有用，奖励估计需要更新。保留什么、更新什么，正是持续学习要研究的选择。

问题：为两条配送路线填写：变什么、何时变、学习器知道吗、是否清零？

提示：先固定一个最小协议，别同时加入视觉和机器人故障。

参考答案：奖励概率互换；在第 1,001 步开始；学习器只看到自己动作的奖励；价值估计与动作计数都不清零。这个协议已经足够用于下一课。

### 思考题

写一张六项协议卡；然后回答：环境从不变化，是否就一定没有 CRL 问题？

参考答案：不一定。环境可以固定却包含远超过有限智能体能力的规律，经验分布也由行为塑造。Abel 等强调的是智能体是否需要不断学习，而不仅是外部参数是否变化。刚入门时先掌握具体协议，不必立即证明形式化条件。

### 配套资源

- [Towards Continual Reinforcement Learning: A Review and Perspectives — Khetarpal 等，JAIR 2022](https://arxiv.org/abs/2012.13490)
- [A Definition of Continual Reinforcement Learning — Abel 等，NeurIPS 2023](https://arxiv.org/abs/2307.11046)
- [David Abel — Where is Learning?](https://www.youtube.com/watch?v=vVZuoEtxjzA)

---

## 02 第一条曲线：旧经验什么时候成了负担？

先修：样本平均与 ε-greedy。建议时间：35 分钟 + 实验。

学习产出：解释常数步长的跟踪能力与噪声代价，并运行一个实验。

### 一条你认识的更新式

Q ← Q + α(R − Q)。括号里是新证据与旧估计的差。α = 1/N 是样本平均；随着经验增加，一条新证据的权重越来越小。它很适合估计固定均值，却可能对后来变化反应迟钝。常数 α 不会随生命长度自动降到零。

### 手算三步

当前 Q = 0.8，α = 0.1，随后连续得到奖励 0。更新后依次为 0.72、0.648、0.5832。若已访问该动作一万次、采用 1/N，第一次仅降到约 0.79992。这里没有深度网络、没有灾难性遗忘；它已经暴露了估计稳定性与跟踪速度的权衡。

### 不要把探索漏掉

即使 B 变好了，若永远不选 B 就观察不到它。ε-greedy 保留小概率探索；这不是只调步长能替代的。实验必须区分“没有采到新信息”和“采到了但更新得太慢”。最好记录动作频率、价值估计和回报，而不只记录平均奖励。

### 指标要围绕变化

看切换前回报、切换后短窗口损失、后期恢复和整段累计回报。一个方案可能恢复快却一直抖动。下面的小实验固定奖励分布的切换时间；它不是一般非平稳 RL 的解决方案，也不含深度可塑性。

### 具体例子

浏览器实验让两臂的奖励概率由 (0.8, 0.2) 交换为 (0.2, 0.8)。你可以改变步长、探索率和随机种子，重新计算两种估计器的真实采样曲线。请先预测结果再点运行。

### 术语小补丁

- **步长 α**：新观测对当前估计的影响大小。
- **探索率 ε**：每一步以 ε 的概率随机选动作；随机分支也可能选到当前最优动作。
- **随机种子 seed**：用于重现一次伪随机采样；一个 seed 是一次运行，不是一个新算法。

### 推导：从样本平均推到“有记忆长度”的更新

看清旧数据为什么拖慢跟踪，也理解大步长为何更吵。

符号：固定一个动作，Rₙ 是它第 n 次被选中时的奖励；Qₙ 是处理这 n 个奖励后的估计。n 不是环境总步数。

把 n 个奖励的和拆成前 n−1 个与最后一个，得到不用保存全部历史的平均数。

$$
Q_n=\frac{1}{n}\sum_{i=1}^nR_i=Q_{n-1}+\frac{1}{n}(R_n-Q_{n-1})
$$

将 1/n 换成固定 α，旧估计每次保留 1−α。

$$
Q_n=(1-\alpha)Q_{n-1}+\alpha R_n
$$

重复展开：每条奖励的权重随之后的更新次数指数衰减。

$$
Q_n=(1-\alpha)^nQ_0+\sum_{i=1}^{n}\alpha(1-\alpha)^{n-i}R_i
$$

在持续观测同一个固定均值的简化条件下，初始偏差缩小到一半所需的更新次数为：

$$
h=\frac{\ln(1/2)}{\ln(1-\alpha)},\qquad 0<\alpha<1
$$

若奖励独立同分布、方差为 σ²，稳态估计方差来自几何级数。

$$
\operatorname{Var}(Q_\infty)=\alpha^2\sigma^2\sum_{j=0}^{\infty}(1-\alpha)^{2j}=\frac{\alpha}{2-\alpha}\sigma^2
$$

例子：α=0.1 时偏差半衰期约为 6.58 次“选中该动作后的更新”；α=0.5 时为 1 次。更快适应，也对近期噪声更敏感。Q=0.8 连续遇到三次 0 奖励，α=0.1 时得到 0.72、0.648、0.5832。

适用条件：方差式假设固定奖励分布和独立样本，不直接描述自适应选择动作后的整段控制回报。半衰期是更新次数，不保证 7 个环境步就能恢复。

[配套来源：Sutton & Barto · 第二版，第 2、6、12 章](http://incompleteideas.net/book/the-book-2nd.html)

### 理解与操作练习

问题：已经选过 A 一万次，为什么一次坏奖励几乎改不动样本平均？

提示：把这次的步长 1/10001 与 0.1 比较。

参考答案：新观测只占约万分之一。Q=0.8、R=0 时，更新后约为 0.79992；固定步长 0.1 则变为 0.72。

问题：固定 ε=0.1、seed=7，只把 α 从 0.02 改到 0.5。请同时看回报、选 B 比例与 Q 值。

提示：先预测哪种设置更快放弃 A；再区分“估值已改变”和“动作已经改变”。

参考答案：观察切换后的 Q 排序和选 B 比例是否更快反转。大步长也可能让 Q 值更波动；不同 seed 未必同样排序。应记录现象，不能只根据一条线给出一般性结论。

### 思考题

让常数步长从 0.02 增到 0.5。为什么恢复速度可能更快，但后期回报未必更高？

参考答案：更大的 α 更快遗忘旧估计，也放大近期奖励噪声；价值排序可能反复翻转。单个种子不能断言最优步长。用 Python 版跑多个种子，分别报告变化后早期和稳定后指标。

### 配套资源

- [Sutton & Barto / Spinning Up / MARL Book：三种开放教材范式](http://incompleteideas.net/book/the-book-2nd.html)
- [CleanRL：从一个文件读懂深度 RL](https://docs.cleanrl.dev/)

---

## 03 分清三种失败：遗忘、学不进、没看懂

先修：了解神经网络训练与 train/test。建议时间：45 分钟。

学习产出：能设计 old / new / fresh 三组对照，不把 dormant neuron 数量当结论。

### 遗忘看过去，可塑性看未来

在 A 上学好，学 B 后 A 的表现下降，是遗忘的证据。训练过很多任务后，给一个新任务 C，老网络比相同结构的新初始化网络学得更慢，才是研究学习能力下降的一个入口。二者可以同时发生，也可以分别发生。

### 第三种：表示里根本没有答案

若当前图像无法区分两个需要不同动作的情境，换优化器或重置神经元也不能凭空获得信息。需要历史、记忆或更合适的预测状态。因此诊断顺序是：任务是否可解、信息是否足够、更新是否稳定，最后才讨论哪种可塑性机制。

### 三类修补各解决什么

回放把旧经验带回来；EWC 类方法限制重要参数改变；ReDo 与 Continual Backprop 类方法更新部分失效或低效用特征。保护太强会妨碍新学习，替换太多会破坏旧知识。它们不是互相排斥的标签，应在同一预算下比较具体机制。

### 怎样读第一篇深度论文

先读 Loss of Plasticity 的问题、实验流程和主要图，再看低效用单元如何被选择与替换。暂不追逐全部超参数。记下：任务分布怎样变化、fresh baseline 怎么初始化、更新次数是否相同。激活率、梯度范数、特征秩是诊断信号，不等于能力本身。

### 具体例子

评测矩阵的行是“训练到哪个阶段”，列是“测哪个任务”。A→B 后 A 的单元格下降说明保留变差；另画老网络与 fresh 网络在同一新 C 上的学习曲线，才能讨论学新东西的速度。仅有矩阵的最后一行还不够。

### 术语小补丁

- **遗忘 forgetting**：学习新内容后，在同一个旧任务上表现变差。
- **可塑性 plasticity**：继续学习新内容的能力；不能仅由当前分数或神经元活跃度判断。
- **fresh baseline**：相同结构、重新初始化的学习器，用于比较在同一新任务上的学习过程。

### 推导：把“忘了”和“学得慢”分成两个测量

它们可以同时出现，但不是同一个数字。

符号：Sᵢⱼ 是学完阶段 i 后在任务 j 的分数，越高越好；L(k) 是进入同一新任务后第 k 次更新的损失，越低越好。

先用一个允许回访旧任务的两阶段协议，测旧任务 A 的下降。

$$
F_A=S_{A,A}-S_{B,A}
$$

新任务的整段损失差可以作诊断，但还受初始难度与先验知识影响。

$$
\Delta_K=\frac{1}{K}\sum_{k=1}^{K}\left[L_{\mathrm{old}}(k)-L_{\mathrm{fresh}}(k)\right]
$$

用二次损失看一个容易误判的例子。目标是 y，梯度更新为：

$$
L_y(w)=\frac12(w-y)^2,\quad w_{k+1}=w_k-\alpha(w_k-y)
$$

移项后，每一步到目标的距离都缩小相同倍数；平方误差也按固定比例缩小。

$$
w_{k+1}-y=(1-\alpha)(w_k-y),\qquad \frac{E_{k+1}}{E_k}=(1-\alpha)^2
$$

为什么学 B 可能伤害 A？在固定两组数据的可微损失下，做一次小步更新，用一阶展开看梯度方向。

$$
L_A(\theta-\alpha g_B)-L_A(\theta)\approx-\alpha\, g_A^\top g_B,\quad g_j=\nabla L_j(\theta)
$$

例子：先学目标 +1，再换成 −1。旧参数约 +1，fresh 参数为 0；它们到新目标的初始平方误差是 4 和 1，但收缩率完全相同。老参数暂时误差更大，不足以证明学习能力受损。

适用条件：这些是教学诊断，不是可塑性的通用定义。一阶干扰式忽略二阶项，且 RL 的数据分布也会随策略改变；旧任务若不能回访，F_A 也不能直接测。

[配套来源：Dohare 等 · Loss of Plasticity](https://www.nature.com/articles/s41586-024-07711-7)

### 理解与操作练习

问题：学习 B 后，A 的分数从 80 降到 50。这说明了什么，还没说明什么？

提示：这里测到了旧任务，没有测新的 C。

参考答案：说明这个评测协议下 A 的表现下降了 30 分；还没有证明 agent 学 C 变慢，需要在同一 C 上比较学习过程。

问题：看本课的线性例子：老参数在新任务上误差更大，但归一化误差曲线重合。你会怎样表述？

提示：比较误差 E(k)/E(0)，而不只看绝对误差。

参考答案：旧目标发生遗忘；在这个二次损失例子里，两种初始化具有相同的误差收缩速度。它用于说明测量不能混淆，不能代替深度网络的可塑性实验。

### 思考题

如果重置一半网络后新任务更好，能否说“解决了可塑性损失”？

参考答案：还不能。需要匹配随机替换对照、学习率搜索和算力，并测旧任务、多个后续任务及长生命史。重置带来的短暂优化优势不自动意味着长期知识积累。

### 配套资源

- [Loss of Plasticity in Deep Continual Learning — Dohare 等，Nature 2024](https://www.nature.com/articles/s41586-024-07711-7)
- [Understanding Plasticity in Neural Networks — Lyle 等，ICML 2023](https://proceedings.mlr.press/v202/lyle23b.html)
- [The Dormant Neuron Phenomenon in Deep Reinforcement Learning — Sokar 等，ICML 2023（ReDo）](https://proceedings.mlr.press/v202/sokar23a.html)
- [The Primacy Bias in Deep Reinforcement Learning — Nikishin 等，ICML 2022](https://proceedings.mlr.press/v162/nikishin22a.html)
- [Continual Backprop / Loss of Plasticity](https://github.com/shibhansh/loss-of-plasticity)
- [Plasticine](https://arxiv.org/abs/2504.17490)
- [Richard S. Sutton — Generate-and-test Methods for Continual Learning](https://www.youtube.com/watch?v=NaJxvGV8MRg)
- [Clare Lyle — Why Do Neural Networks Lose Plasticity?（CoLLAs 2023）](https://www.youtube.com/watch?v=1EwKYesnKAA)

---

## 04 看不到，不等于学不会：状态与记忆

先修：MDP 中的状态概念；无需先学 POMDP 理论。建议时间：40 分钟 + 实验。

学习产出：区分环境状态、观测、工作记忆与长期知识。

### 当前观测不总是状态

红灯意味着最后选左，蓝灯意味着选右；中间十步看到相同走廊。最终图像相同，但正确动作不同。只把当前图像输入前馈网络，无法区分这两种历史。环境的真实状态可以包含灯色，智能体当前的观测却不包含。

### 一位记忆足够，但如何学到它不简单

手工把灯色记下来就能解决这个诊断任务。它说明信息瓶颈在哪里，却没有证明 RNN 能自动发现、保存并在正确时刻使用这位信息。RNN 内部状态随每一步变化；网络权重则跨多次经历积累如何更新状态的知识。

### 从“记住一切”到“预测有用的事”

GVF（general value function）把一个知识问题写成：在某种行为条件下，未来的某个信号会怎样累计？例如继续走这条路，多久会遇到充电点。Horde 研究从共享经验同时学习多个预测；预测状态能否足以支持控制，还需要另外验证。

### 通往前沿的桥

在小线索任务上确认缺少历史的基线确实失败，再进入 POPGym、Forager。比较同等状态容量的前馈、循环或预测式状态表示。若改进同时增大内存、历史长度和计算量，就不能只归因于新的状态学习方法。

### 具体例子

Python memory 实验刻意用 balanced 的随机灯色：不记忆的固定策略正确率是 1/2；保存灯色的手工策略是 1。它不是“学会记忆”的论文结果，而是检查任务信息结构是否符合你的假设。

### 术语小补丁

- **观测 observation**：agent 当前实际收到的输入，不一定包含决策所需的全部信息。
- **内部状态 z**：由历史压缩得到的工作记忆；它不同于环境真实状态。
- **参数 θ**：规定怎样更新内部状态或作出决策的可学习规则。

### 推导：为什么没有历史，网络再大也猜不出灯色？

先排除信息不足，再讨论优化方法。

符号：C 是等概率的红／蓝线索；O 是最终相同的走廊图像；p 是无记忆策略此时选择左的概率。

假设当前观测与过去线索独立，随机选动作也不能突破一半。

$$
\Pr(\text{correct})=\tfrac12p+\tfrac12(1-p)=\tfrac12
$$

保存过去线索的内部状态可以让两段相同观测对应不同决策。

$$
z_t=f_\theta(z_{t-1},O_t,A_{t-1}),\qquad A_t\sim\pi_\theta(\cdot\mid z_t)
$$

例子：红灯时 z=0、蓝灯时 z=1，走廊中保持 z，最后按 z 决定左右。它能正确决策，但“手工提供一位记忆”和“学会怎样记忆”是两件事。

适用条件：1/2 的结论依赖平衡线索、相同最终观测、没有旁路信息。若策略能访问历史、trial 编号或隐藏标签，就不再是这个无记忆条件。



### 理解与操作练习

问题：环境真实状态包含灯色，为什么当前图像仍然不足以决策？

提示：真实世界包含的信息，不一定都进入相机。

参考答案：两种灯色历史产生相同最终图像，但要求不同动作。映射“当前图像→动作”无法区分它们。

问题：把灯色保留在最终图像里，再比较无记忆与有记忆策略，你预计会怎样？

提示：你改变的是信息条件，而不只是算法。

参考答案：无记忆策略也可以按当前灯色作出正确动作。因此应先固定观测协议，才能比较记忆机制。

### 思考题

固定参数 RNN 的 hidden state 改变，是学习吗？

参考答案：本教程先把当次情境推断与跨经历的知识积累分开。更广的 agent 定义允许持久可更新状态承载学习；关键要说明保留什么、影响哪些未来行为。仅看到 hidden state 变化，不能直接证明长期能力增长。

### 配套资源

- [Horde — Sutton、Modayil、Delp、Degris、Pilarski、White、Precup，2011](https://josephmodayil.com/papers/horde-final.pdf)
- [Recurrent Trace Units — Elelimy、Adam White、Bowling、Martha White，NeurIPS 2024](https://arxiv.org/abs/2409.01449)
- [RTU](https://github.com/esraaelelimy/rtus)
- [Forager](https://github.com/andnp/forager)
- [POPGym](https://github.com/proroklab/popgym)
- [Ida Momennejad — Guest Lecture（页面未提供直接 slides）](https://deeprlcourse.github.io/guests/ida_momennejad/)
- [Andrej Karpathy：RNN 的直观解释](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)

---

## 05 只用最新一条经验，怎样稳定地学？

先修：TD 误差；可按需补 Sutton & Barto 第 9、12、13 章。建议时间：50 分钟。

学习产出：能画出更新顺序，区分缓冲区、trace 与模型状态。

### 先对齐更新协议

DQN 常用 replay 和 target network；严格 streaming 路线要研究没有这些设施时怎么稳定更新。不要只把 batch_size 改成 1 就称为复现。归一化、步长、actor/critic 耦合、梯度截断和初始化都可能影响稳定性。

### trace 是压缩的责任记录

Eligibility trace 可记成 e ← γλe + 当前特征，再用 TD 误差 δ 更新 θ ← θ + αδe。它让今天的误差影响先前参与预测的特征，不需要重放全部轨迹。这里是线性预测的直观入口；带函数逼近、策略梯度或 off-policy 时还需检查各自公式。

### 不同参数不一定该同速学习

线性 IDBD 为各特征自适应步长，用过去更新的方向信息调节未来更新幅度。先在有用特征加噪声特征的预测任务上理解它，再去看 Oak 的 selective credit 思路。线性 IDBD 不等于 NetworkIDBD，也不是从监督预测到深度控制的完整推导。

### 从教程进真实实现

先读 Stream-X 的一条 transition 如何更新，再固定官方分支 2026 与一个 commit 运行小任务。RTU / Streaming RTRL 则针对递归状态的实时信用分配；其效率来自结构限制，不能概括成所有 RNN 的精确 RTRL 都是线性的。

### 具体例子

把一步切成 act → environment → reward/observation → state update → critic/actor update。记录每段耗时。一个回报更高却经常超过动作 deadline 的算法，在真实时间协议下可能不可用。

### 术语小补丁

- **TD 误差 δ**：一步奖励加上下一个状态的估值，与当前估值之差。
- **资格迹 e**：压缩过去特征对当前更新的影响，不是保存原始样本。
- **半梯度 semi-gradient**：更新当前预测时，暂时把自举目标视为常数。

### 推导：把 TD(λ) 的一步更新拆开

先会手算线性预测，再阅读复杂的 streaming actor–critic。

符号：xₜ 是状态特征向量；w 是权重；γ 是折扣，λ 控制迹的衰减。以下为 on-policy 线性预测的 accumulating traces。

用线性函数预测价值，并构造一步自举目标。

$$
\hat v_w(S_t)=w^\top x_t,\qquad \delta_t=R_{t+1}+\gamma w_t^\top x_{t+1}-w_t^\top x_t
$$

若先不使用迹，对平方 TD 误差作半梯度更新，就得到当前特征的更新方向。

$$
w_{t+1}=w_t+\alpha\delta_t x_t
$$

让过去的特征也承担责任：先更新迹，再更新权重。

$$
e_t=\gamma\lambda e_{t-1}+x_t,\qquad w_{t+1}=w_t+\alpha\delta_t e_t
$$

将迹展开，看到被压缩的历史。

$$
e_t=\sum_{k=0}^{t}(\gamma\lambda)^{t-k}x_k\quad(e_{-1}=0)
$$

例子：α=0.1、δ=2、γλ=0.8，上一条迹 e=(1,0)，当前特征 x=(0,1)。新迹为 (0.8,1)，权重增加 (0.16,0.2)：先前特征也收到更新。

适用条件：这不是任意 off-policy 或非线性控制算法的稳定性保证。终止转移的后继价值为 0，episodic 协议还要按定义清空迹；continuing 协议不可偷偷按固定窗口清空。

[配套来源：Sutton & Barto · 第二版，第 2、6、12 章](http://incompleteideas.net/book/the-book-2nd.html)

### 理解与操作练习

问题：迹保存过去影响，为什么不等于 replay buffer？

提示：一个保存向量，另一个保存可再次训练的样本。

参考答案：迹是随每步衰减和累积的责任摘要，不能从它还原并重新训练某条历史 transition；replay 则明确重用保留下来的样本。

问题：在数值例子中令 λ=0，更新会怎样改变？

提示：此时历史迹不再保留。

参考答案：新迹等于当前特征 (0,1)，权重只增加 (0,0.2)，退回一步 TD 更新。

### 思考题

Eligibility traces、RNN hidden state、replay buffer 都算记忆，为什么要分别报告？

参考答案：它们保存的对象和使用方式不同：trace 压缩信用历史，hidden state 汇总观测，replay 保存可再次训练的样本。三者都占资源，但只有后者必然涉及样本重用；具体算法还可能同时使用它们。

### 配套资源

- [stream-x / Streaming Deep RL](https://github.com/mohmdelsayed/streaming-drl)
- [RTU](https://github.com/esraaelelimy/rtus)
- [SwiftTD / Average-Reward Methods](https://github.com/kjaved0/swifttd)
- [A. Rupam Mahmood — Streaming Deep Reinforcement Learning](https://www.youtube.com/watch?v=QOfkOl9QrZY)
- [Intentional Updates](https://arxiv.org/abs/2604.19033)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://arxiv.org/abs/2605.24709)
- [Oak Lab：从经验学习，而非从整理好的数据集学习](https://oaklab.ai/posts/learning-from-experience-instead-of-curated-datasets)

---

## 06 不只反应：技能、模型与规划如何积累？

先修：Sutton & Barto Part I 中的 Dyna；没学过可先读第 8 章。建议时间：45 分钟。

学习产出：解释技能、技能模型和规划三者的接口，并指出旧模型风险。

### 动作太短，计划会很长

逐步按键可以完成导航，但远距离计划若每次从原始动作开始，计算成本很高。Option 把一段有目的的行为封装为：何时可启动、内部如何行动、何时结束。它不是单纯把 action repeat 固定为十步。

### 技能会做，不等于规划会用

规划需要知道执行某项技能大致会到哪里、要多久、获得什么回报。因此技能策略和技能模型是不同对象。Dyna 用学得的模型产生规划更新；若模型过时，多规划反而会重复过时知识。

### 技能从哪里来

Machado 的 eigenoptions 用状态空间的谱结构构造探索方向；successor representation 将行为下的未来占用结构编码进表示。Konidaris 强调从底层技能通向可组合的高层表示。发现多样动作不是终点，要测它们是否改善探索、迁移或规划。

### Alberta 与 Oak 的位置

Alberta Plan 将在线预测、状态构造、子任务、技能模型和规划连成研究路线。Oak 延续经验驱动、有限资源下自主发现有用抽象的目标。你可以先验证其中一条接口：新技能出现后，模型预测和规划收益是否改善；无需先实现整个架构。

### 具体例子

仓库中“去充电区”是 option；“执行后 20 步左右抵达、耗电 3 单位”是 option model；在送货和充电之间选择是 planning。货架挪动后，策略、模型、规划中至少有一处要更新，且它们的时间尺度可能不同。

### 术语小补丁

- **Option**：带启动条件、内部策略和终止规则的一段行为。
- **技能模型**：预测执行技能后的状态、累计奖励或持续时间。

### 推导：为什么技能的 Bellman 目标要用 γ 的 τ 次方？

一项技能跨越多个基本时间步，不能一律当作一步动作。

符号：o 是 option；τ 是它执行到终止的步数；Sₜ₊τ 是终止状态。Q 表示给定技能集合上的最优技能价值。假设折扣与期望良好定义，技能会终止。

先把执行期间的奖励按原始时间尺度相加。

$$
G_t^{(o)}=\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}
$$

到技能终止后再作选择，后续价值已经过去了 τ 步。

$$
Q(s,o)=\mathbb{E}\!\left[G_t^{(o)}+\gamma^\tau\max_{o'\in\mathcal O(S_{t+\tau})}Q(S_{t+\tau},o')\mid S_t=s,o\right]
$$

例子：γ=0.9，技能执行 2 步，两步奖励均为 1，终止状态后续价值为 10。目标为 1+0.9+0.9²×10=10；若错误地只折扣一次，会得到 10.9。

适用条件：写出技能 Bellman 关系，不等于已经学会发现技能或准确预测技能后果。真实环境变化时，需要同时检查技能策略和模型。

[配套来源：Options 原论文入口](https://www.sciencedirect.com/science/article/pii/S0004370299000521)

### 理解与操作练习

问题：“走到门口”和“预测走到门口花多少步”是同一个模型吗？

提示：一个产生动作，一个预测行为后果。

参考答案：前者是技能策略，后者属于技能模型。规划需要模型信息，但会执行技能不意味着已经知道这些后果。

问题：技能耗时从 2 步变成 5 步，规划器只更新终点、不更新耗时，会遗漏什么？

提示：时间影响折扣，也可能影响运行成本。

参考答案：后续价值应以 γ⁵ 折扣，并重新计算执行期间的奖励或代价。相同终点不代表两个技能同样有价值。

### 思考题

一个新 option 提高了训练回报，如何判断它真的帮到了规划？

参考答案：固定数据与探索预算，对比有无技能模型、有无规划，并报告规划计算量、模型误差和未见布局表现。若收益只来自更好的探索，仍有价值，但不是相同机制。

### 配套资源

- [Machado options：从 main.py 读懂 eigenoptions](https://github.com/mcmachado/options)
- [Marlos C. Machado — Representation-driven Option Discovery in Reinforcement Learning](https://deeprlcourse.github.io/guests/marlos_machado/)
- [George Konidaris — Resolving the Sensorimotor Dilemma](https://www.youtube.com/watch?v=Af5UFE7CdKs)
- [The Alberta Plan for AI Research — Sutton、Bowling、Pilarski，2022](https://arxiv.org/abs/2208.11173v1)
- [ALPS：Laplacian 表示与决策时规划](https://arxiv.org/abs/2602.05031)
- [Laplacian Keyboard](https://arxiv.org/abs/2602.07730)
- [The OaK Architecture](https://oaklab.ai/posts/the-oak-architecture)

---

## 07 怎样证明“持续学习变好了”？

先修：均值、方差与训练/测试分离。建议时间：50 分钟。

学习产出：写出可审计的实验协议，选择与问题匹配的 benchmark。

### 先选问题，不先选排行榜

保留旧任务可选任务序列基准；状态记忆可选 POPGym / Forager；无自然终点与复杂交互可看 AgarCL；多智能体课程变化可看 MEAL。一个基准不会自动测出所有能力。OGBench 的离线任务成绩也不能直接当成在线 CRL 成绩。

### 记录整段生命史

至少记录变化前后曲线、旧任务保留、新任务学习速度、总体回报、峰值内存与实际运行时间。奖励量纲不同，跨任务聚合需要预先确定归一化。只有最后一段平均回报，可能把早期大规模失败和长期遗忘隐藏起来。

### 对照不要偷看未来

随机种子分为调参用和报告用；同一方法不得用测试任务选择最佳配置。变化时间是否已知、是否给 task ID、每阶段是否重置 optimizer 都应列入协议。可先用 5 个种子排错，正式数量由方差、资源与精度要求决定，而非固定神奇数字。

### 小规模复现的准确说法

成功 import 包是安装验证；跑 1000 步不崩是 smoke test；复现趋势是缩小实验；匹配论文配置与统计结果才接近论文复现。本站教学实验只验证实现和机制，不宣称复现了完整 CRL 论文。

### 具体例子

方法 A 最终 90 分但切换后损失 1000 步；方法 B 最终 85 分但 100 步恢复。哪个更好取决于任务目标、生命周期长度和安全代价。先定义要优化的量，再挑展示指标。

### 术语小补丁

- **学习曲线**：随交互或更新推进，持续记录的表现。
- **独立运行**：用不同 seed 重复完整学习过程；重叠时间窗不能替代独立运行。

### 推导：从一条曲线到可比较的数字

先明确每个点的含义，再谈平均和误差条。

符号：W 为窗口长度；μₜ(a) 是已知教学环境的真实期望奖励；m 是独立运行编号。

滑动或不重叠窗口都可以，但必须注明是哪一种。本站图采用不重叠窗口。

$$
\bar R_t=\frac{1}{W}\sum_{i=t-W+1}^{t}R_i
$$

在这个已知两臂均值的实验里，可以按实际动作计算相对当时最好动作的机会损失。

$$
\mathcal R_T=\sum_{t=1}^{T}\big[\max_a\mu_t(a)-\mu_t(A_t)\big]
$$

独立运行的均值与样本标准差是两种不同摘要。

$$
\bar J=\frac1M\sum_{m=1}^{M}J_m,\qquad s=\sqrt{\frac{\sum_{m=1}^{M}(J_m-\bar J)^2}{M-1}}
$$

例子：切换后选错动作一次，期望机会损失为 0.8−0.2=0.6；不是用随机的 0/1 奖励直接相减。20 个 seed 的标准差表示运行波动，不等于均值的置信区间。

适用条件：真实控制任务通常不知道最优策略或真实均值，不能照搬这个 bandit regret。单次运行中相邻窗口相关；不把它们冒充独立样本。



### 理解与操作练习

问题：最终分数相同，变化后恢复时间不同，两种方法是否等价？

提示：把恢复过程中损失的奖励也算进去。

参考答案：不等价。整段累计收益、短期损失和稳定后表现可能不同，要先选与应用目标一致的指标。

问题：先运行 1 个 seed，再运行 20 个 seed，报告中多出来的信息是什么？

提示：看每个 seed 的结果和跨 seed 波动，而不只是更平滑的线。

参考答案：可以检查趋势是否稳定、是否被少数运行主导。样本标准差反映运行差异；它不会自动排除调参泄漏或协议不公平。

### 思考题

三个种子的平均曲线高一点，能否称为 state of the art？

参考答案：不能据此下结论。先排除调参预算、环境版本、种子选择与协议差异，给出跨任务和不确定性分析。若证据不足，准确表述“在这个小规模设置下观察到差异”。

### 配套资源

- [Adam White — Reinforcement Learning Experiments that Matter!](https://deeprlcourse.github.io/guests/adam_white/)
- [Michael Littman — Assessing the Robustness of Deep RL Algorithms](https://deeprlcourse.github.io/guests/michael_littman/)
- [Forager](https://github.com/andnp/forager)
- [AgarCL](https://github.com/AgarCL/AgarCL)
- [Continual World（CW10／CW20）](https://arxiv.org/abs/2105.10919)
- [COOM（Continual DOOM）](https://github.com/TTomilin/COOM)
- [MEAL](https://github.com/TTomilin/MEAL)
- [rliable](https://github.com/google-research/rliable)

---

## 08 从读懂论文到提出一个可检验的问题

先修：前七课；不要求先读完全部论文。建议时间：40 分钟 + 项目规划。

学习产出：产出一页研究计划与可复现的实验目录。

### 先定位失败，再提出方法

例如“随着干扰特征增加，固定步长的在线预测变差”比“我要实现通用持续智能”更容易开始。写清观察、候选原因和能排除候选原因的实验。先有能稳定重现的失败，再引入新机制。

### 只选择一条主线

喜欢优化：可塑性或自动步长；喜欢序列：状态构造与时间信用；喜欢结构：技能发现与规划；喜欢系统：single-life 或 benchmark；喜欢交互：课程、元学习与多智能体。先沿一条支线深入，两条交叉线放到第二个项目。

### 一篇论文的三遍读法

第一遍写问题、假设、图与结论边界；第二遍还原更新式、伪代码和实验权限；第三遍追源码、配置和随机种子。每遍结束必须产出一个东西：一页解释、一张更新流程图、一条可运行命令。

### 六周里程碑是建议，不是保证

第 1 周完成教程和小实验；第 2 周读两篇核心论文并安装一个官方仓库；第 3 周复现一个最小趋势；第 4 周做失败诊断；第 5 周引入单个改动和强对照；第 6 周写结果、负结果与边界。安装困难或高方差都可能延长时间。

### 具体例子

候选项目：Forager 中，固定状态维度和每步时间预算，对比短历史堆叠与预测式内部状态。先问收益来自额外信息、额外参数还是学习目标；不要一开始同时换 encoder、optimizer 和 reward。

### 思考题

你的方法没有超过 baseline，项目是否就失败了？

参考答案：若实验排除了一个有吸引力的解释，或发现改进仅来自某项权限/预算差异，就是有用的研究证据。保存负结果、配置与完整曲线；不要只挑最好种子。

### 配套资源

- [Forager: a lightweight testbed for continual learning with partial observability in RL](https://arxiv.org/abs/2605.01131)
- [Plasticine：可塑性方法实现与指标](https://github.com/RLE-Foundation/Plasticine)
- [Oak Lab：从经验学习，而非从整理好的数据集学习](https://oaklab.ai/posts/learning-from-experience-instead-of-curated-datasets)
- [Adam White — Reinforcement Learning Experiments that Matter!](https://deeprlcourse.github.io/guests/adam_white/)