# 从研究问题进入持续强化学习

这是一种教学组织，不是唯一分类。先明确目标、信息与干预权限，再选择机制。

## 一生的“学得好”到底指什么？

现象：终局测试很高，但适应时损失巨大，或者依赖免费重置。

先区分：目标、交互协议和变化来源是三条独立轴；不是看到 continual 就必须使用某一个算法。

- 优化折扣收益、平均奖励率，还是一段生命期的累计效用？
- 变化发生在奖励、动力学、观测、动作集合，还是学习器自己诱导的数据分布？
- 任务 ID、边界、reset、replay、预训练经验各允许什么？

### 算法线与条件

- **折扣 / 有限时域 / 平均奖励**：改变如何评价未来与时间代价。条件与限制：平均奖励需要相应的遍历/通信等假设，并不自动解决变化。
- **非平稳 MDP / 潜在情境 POMDP**：前者描述外部随时间改变；后者把未观测情境纳入状态。条件与限制：同一表象可以有不同模型，不要混用各自理论保证。
- **任务序列 / task-agnostic / single-life**：改变可用信息和干预权限。条件与限制：知道边界不等于知道任务身份；没有手动 reset 不等于永不终止。

诊断：同一策略分别计算整体收益、变化后损失和每真实时间单位收益；先看结论是否因评价方式翻转。

基准：CSuite 适合持续目标；Continual World 适合任务序列；Single-Life RL 适合一次试验中的恢复。三者不能直接混排。

- [Khetarpal et al. · Towards Continual RL](https://arxiv.org/abs/2012.13490)：按非平稳性的发生位置和驱动力组织问题；提醒我们先写设定，再比较方法。
- [Abel et al. · A Definition of Continual RL](https://arxiv.org/abs/2307.11046)：将持续性落在智能体是否需要一直学习，而不只落在外部任务切换。这里采用其问题视角，不把下面的教学分类说成该文定理。
- [Pan et al. · A Survey of Continual RL](https://arxiv.org/abs/2506.21872)：补充任务序列、保留、迁移、容量和评测的广泛方法线；其任务化视角与 Abel 的定义不是完全相同的设定。
- [Wan et al. · Learning and Planning in Average-Reward MDPs](https://proceedings.mlr.press/v139/wan21a.html)：Differential TD/Q 与平均奖励规划，区分奖励率和相对价值。

---

## 规律变了，怎样及时发现并改正？

现象：旧估计非常稳定，却迟迟跟不上新奖励或新动力学。

先区分：没采到变化、统计估计太慢、情境识别错误，是三种不同失败；不必先假定神经网络可塑性坏了。

- 变化是突变、渐变还是周期重现？
- 检测误报和检测延迟如何影响控制？
- 是忘掉旧统计量，还是保存旧模型并识别何时复用？

### 算法线与条件

- **常数步长 / 指数遗忘 / 滑动窗口**：使近期数据有更大权重。条件与限制：更快适应会增加估计方差；窗口按真实时间还是访问次数要写清。
- **变化点检测 / 多模型 / 上下文推断**：判断何时切换学习状态或选择旧专家。条件与限制：TD error 大也可能只是探索到了新状态，不能直接当变化证明。
- **SWUCRL2-CW / BORL；permanent–transient values**：分别从非平稳遗憾、快慢价值分解处理持续修正。条件与限制：理论设定、任务边界权限、保存状态和深度实现并不相同。

诊断：先用固定数据流比较估计器，再让动作影响数据；同时记录预测误差、检测延迟/误报和变化后损失。

基准：先分段平稳或渐变 bandit / 小 MDP；再进入 task-agnostic Meta-World 或 AgarCL，不从最高维平台猜失败来源。

- [Cheung et al. · Non-stationary MDPs](https://proceedings.mlr.press/v119/cheung20a.html)：SWUCRL2-CW 与 BORL 把滑窗、乐观探索和变化预算联系起来；不是只换一个常数学习率。
- [Anand & Precup · Prediction and Control in CRL](https://arxiv.org/abs/2312.11669)：用 permanent / transient 两个价值分量连接长期积累与短期修正；区分可见边界与持续设置。
- [Caccia et al. · Task-Agnostic CRL](https://proceedings.mlr.press/v232/caccia23a.html)：3RL 把循环状态与回放结合，用历史推断情境；任务未知不等于历史中没有可推断的信息。

---

## 当前画面不够，应该记住什么？

现象：相同观测需要不同动作；更大的前馈网络仍做错。

先区分：工作状态更新、持久知识学习和训练梯度记忆不是同一个东西。

- 哪些历史必须被区分，哪些可丢弃？
- 记忆的训练目标是奖励、观测预测还是一组未来测试？
- 长依赖的梯度怎样在每步预算内传播？

### 算法线与条件

- **Frame stack / RNN / Transformer**：从固定窗口到递归状态或上下文。条件与限制：更长上下文不自动等于更强长期权重学习。
- **Belief / PSR / GVFN**：以后验或预测赋予状态可解释语义。条件与限制：已知生成模型、核心测试和任意学习到的预测集合并不等价。
- **BPTT / TBPTT / RTRL / RTU**：为同一递归表示设计不同信用计算。条件与限制：detach 切断梯度，不一定抹掉前向记忆；成本必须随容量报告。

诊断：用同画面异动作的平衡线索任务，加上可观测状态参照、手工记忆参照，再比较学得记忆。

基准：从 delayed cue 到 POPGym/Forager；必须控制记忆容量、历史长度、每步计算与特权信息。

- [Caccia et al. · Task-Agnostic CRL](https://proceedings.mlr.press/v232/caccia23a.html)：3RL 把循环状态与回放结合，用历史推断情境；任务未知不等于历史中没有可推断的信息。
- [Schlegel et al. · General Value Function Networks](https://arxiv.org/abs/1807.06763)：让递归表示受到预测问题的约束，连接状态构造与预测学习。
- [Elelimy et al. · Real-Time Recurrent Learning using Trace Units](https://arxiv.org/abs/2409.01449)：结构化递归与前向敏感度降低在线信用分配成本；研究重点不只是网络能否记住。

---

## 新知识怎样不破坏仍有用的旧知识？

现象：学完 B 后，在相同协议下重新测 A，A 的表现下降。

先区分：保留准确的旧知识不等于保留已经失效的答案；保留与跟踪可能冲突。

- 该保留原始经验、输出行为、参数约束还是独立模块？
- buffer 应代表过去平均分布还是未来会重访的情境？
- 没有任务 ID 时，如何决定使用哪个知识模块？

### 算法线与条件

- **Replay / CLEAR / generative replay**：重新暴露旧训练信号，必要时增加行为蒸馏。条件与限制：旧 transition 可能已不符合当前动力学；生成模型本身也会忘。
- **EWC / distillation / policy consolidation**：约束重要参数或旧输入上的输出。条件与限制：保护强度太大可能阻止适应；重要性估计不等于未来用途。
- **Progress & Compress / 模块化 / PT values**：隔离快学与长存，或按情境重用。条件与限制：存储、路由、压缩和边界权限都要计入成本。

诊断：做 A→B→A 的保留矩阵，同时测 B 的学习速度；增加固定内存和未知任务 ID 的对照。

基准：Continual World / COOM / continual_rl；同时报告旧任务、当前任务和前向迁移，不能只给最终平均分。

- [Rolnick et al. · Experience Replay for Continual Learning](https://arxiv.org/abs/1811.11682)：CLEAR 结合旧经验上的离策略学习、当前经验学习与行为克隆；回放不只是存一个 buffer。
- [Schwarz et al. · Progress & Compress](https://arxiv.org/abs/1805.06370)：把快速学习与压缩到长期知识分开；检查蒸馏、保留约束和协议权限。
- [Anand & Precup · Prediction and Control in CRL](https://arxiv.org/abs/2312.11669)：用 permanent / transient 两个价值分量连接长期积累与短期修正；区分可见边界与持续设置。
- [Wołczyk et al. · Continual World](https://arxiv.org/abs/2105.10919)：机械臂任务序列同时考察保留与前向迁移；不是无重置单生命期环境。

---

## 训练久了，为什么新东西越来越难学？

现象：老网络获得相同新数据和预算，却比合适的 fresh 参照学得慢。

先区分：新任务误差高可能只是起点不同；神经元稀疏、梯度小等是诊断，不是可塑性结论本身。

- 是数据覆盖差、目标不稳定，还是表示和优化退化？
- 替换哪个单元，何时替换，损失多少旧功能？
- 收益是一次重启，还是长生命史持续保持的新学习能力？

### 算法线与条件

- **CReLU / normalization / regularization**：改变激活、尺度或参数轨迹。条件与限制：需和同预算学习率及初始化控制比较。
- **ReDo / Continual Backprop / shrink-and-perturb**：重新引入可训练特征或扰动。条件与限制：必须测输出扰动、optimizer 状态、替换频率和旧知识。
- **Primacy reset / replay-ratio 控制**：减弱早期经验固化或过拟合。条件与限制：保留 buffer 的网络重置不等于无先验从头学习。

诊断：固定新数据、模型容量和调参预算；比较 aged / fresh / random-reset，并分开初始误差与后续下降速度。

基准：先做受控新目标学习，再到 Atari 模式/任务变化与 AgarCL；可塑性方法在一种场景有效不保证所有 CRL 场景有效。

- [Dohare et al. · Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7)：长期新学习能力及 Continual Backprop；特征替换的效果不能仅由激活率判断。
- [Abbas et al. · Loss of Plasticity in Continual Deep RL](https://proceedings.mlr.press/v232/abbas23a.html)：在价值学习和不同非平稳条件下研究激活、梯度与后续学习，讨论 CReLU。
- [Nikishin et al. · The Primacy Bias in Deep RL](https://arxiv.org/abs/2205.07802)：区分早期经验固化与后续证据利用不足；参数重置可能保留 replay，不能视作全部重来。
- [Mohamed et al. · AgarCL](https://arxiv.org/abs/2505.18347)：非 episodic、高维、部分可观测且持续变化的研究平台；论文中若干可塑性方法提升有限，说明困难不止一条。

---

## 每步预算很小，怎样可靠地分配信用？

现象：实时更新发散、梯度延迟太长，或者一个环境步用了大量隐含回放。

先区分：数据协议、长期收益目标、表示结构和优化器是可组合的选择，不是一条算法世代替换线。

- 一个新误差应该影响哪些过去参与计算的量？
- off-policy 修正改变的是采样分布、目标还是更新方向？
- 尺度、步长、trace 与 actor/critic 怎样相互作用？

### 算法线与条件

- **TD(λ) / true-online / emphatic / gradient TD**：改变时间信用或离策略稳定机制。条件与限制：各式 eligibility 不通用；γ、ρ 的下标与支持集必须对齐。
- **Stream-X / 输入与更新归一化**：把深度每步更新做成可计算的训练流程。条件与限制：batch_size=1 不足以复现方法。
- **IDBD / TIDBD / meta-gradient**：让部分更新参数适应当前学习过程。条件与限制：监督误差、bootstrap 误差与策略目标的导数不能互换。

诊断：先验证 λ=0、终止、ρ=0 等退化情况，再记录每步延迟、内存、更新/样本比以及多 seed 回报。

基准：表格/线性解析任务 → 严格 streaming 控制；必须说明是否重复访问旧样本和隐藏的大批量计算。

- [Stream-X · 作者实现与论文入口](https://github.com/mohmdelsayed/streaming-drl)：研究逐条经验的深度 RL 更新；归一化、迹、更新幅度和 actor–critic 需要一起检查。
- [Elelimy et al. · Real-Time Recurrent Learning using Trace Units](https://arxiv.org/abs/2409.01449)：结构化递归与前向敏感度降低在线信用分配成本；研究重点不只是网络能否记住。
- [Xu et al. · Meta-Gradient Reinforcement Learning](https://arxiv.org/abs/1805.09801)：对学习更新求导以调整 return 等元参数；元目标不能随便跟训练题目一起改变。

---

## 经验该变成哪些预测，才能在未来派上用场？

现象：奖励之外的规律反复重学；辅助损失下降却没有改善决策。

先区分：定义一个预测问题、学准该预测、让下游使用它，是三个独立研究步骤。

- 预测信号、条件行为和终止尺度如何选择？
- 预测是否可由当前经验覆盖？
- 下游是状态构造、奖励迁移、技能发现还是规划？

### 算法线与条件

- **GVF / Horde / nexting**：多个 c、γ、π 定义多种未来知识。条件与限制：不是所有累计信号都具有事件概率语义。
- **Auxiliary tasks / predictive representation**：用额外学习信号塑造共享表示。条件与限制：辅助目标也可能争抢容量或产生干扰。
- **Successor features / GPI**：预测特征累计量，再按奖励权重重估价值。条件与限制：依赖可复用的动力学、策略与奖励特征。

诊断：分别测 held-out 预测误差、删除预测输入后的决策变化、分布变化后的再学习成本。

基准：先用多信号小 MDP 给出解析答案，再测试新奖励重估；预测数量多不能代替知识用途。

- [Sutton et al. · Horde](https://sites.ualberta.ca/~amw8/horde.pdf)：多个不同策略、信号和时间尺度的预测共享经验；学习问题和回答问题的算法分开。
- [Jaderberg et al. · Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397)：多个辅助控制/预测信号训练共享表示；辅助任务的用途需要用主任务表现检验。
- [Barreto et al. · Successor Features for Transfer](https://arxiv.org/abs/1606.05312)：在相同动力学、线性奖励族中重用后果预测，并用 GPI 组合策略。

---

## 怎样形成可复用的目标与长行为？

现象：每个任务都从原始动作重新探索，技能库却越积越多而用不上。

先区分：目标定义要学什么，option 定义怎样行动；技能有多样性不代表对主任务有用。

- 子目标由谁提出，怎样衡量用途？
- 技能该何时启动和停止？
- 新任务能否组合旧能力，而不只是重新训练一遍？

### 算法线与条件

- **UVFA / HER / goal generation**：目标条件化、重标记经验、选择训练目标。条件与限制：三者改变不同对象；HER 不是自动课程本身。
- **Option-Critic / eigenoptions / DIAYN / DADS**：任务驱动、图结构或信息量驱动技能学习。条件与限制：技能发现目标与主任务价值之间仍有距离。
- **Reward-respecting subtasks / SF-GPI / Option Keyboard**：按价值或奖励组合接口复用能力。条件与限制：GPI 不需要 commitment；option 的持续执行和终止是另一层。

诊断：对照 primitive-only、随机技能与同预算学得技能；同时计发现、建模、选择和执行的成本。

基准：先两步技能/导航连通性，再到连续技能；报告技能覆盖、真实主回报和学习总成本。

- [Schaul et al. · Universal Value Function Approximators](https://proceedings.mlr.press/v37/schaul15.html)：价值函数以目标为输入；给定目标后的泛化，不等于自动选择目标。
- [Sutton, Precup & Singh · Between MDPs and semi-MDPs](https://doi.org/10.1016/S0004-3702(99)00052-1)：用启动集合、内部策略和终止规则定义时间扩展行为，并导出相应备份。
- [Sutton et al. · Reward-Respecting Subtasks](https://arxiv.org/abs/2202.03466)：连接子任务、option、后果模型和规划；任务回报与抽象价值的关系是重点。
- [Barreto et al. · Successor Features for Transfer](https://arxiv.org/abs/1606.05312)：在相同动力学、线性奖励族中重用后果预测，并用 GPI 组合策略。

---

## 怎样用后果模型减少真实试错？

现象：模型内预测很好，现实行为却错误；规则变后越规划越自信。

先区分：动作模型、option model、价值函数和规划器分开；模型正确性与求解模型的精度也分开。

- 模型需预测完整分布，还是仅预测某个 backup 所需的统计量？
- 变化发生后模型怎样更新、忘记或重用？
- 把计算花在哪些起点、尺度和分支上？

### 算法线与条件

- **Dyna / prioritized sweeping**：用一步模型进行额外价值更新。条件与限制：模型学习与规划分配都影响结果。
- **Option / expectation / abstract models**：压缩时间或后果以降低规划成本。条件与限制：非线性下游一般不能只接一个均值。
- **MPC / MCTS / Dreamer / Continual-Dreamer**：在线搜索或想象中策略学习。条件与限制：世界模型、critic、actor 和 buffer 可能分别遗忘。

诊断：用已知模型确认 planner，再替换为学习模型；固定真实步数和总计算，画模型误差及真实回报随规划预算变化。

基准：blocking / shortcut maze → Minigrid / Minihack → 高维持续任务；不给某方法额外真值地图。

- [Sutton et al. · Reward-Respecting Subtasks](https://arxiv.org/abs/2202.03466)：连接子任务、option、后果模型和规划；任务回报与抽象价值的关系是重点。
- [Kessler et al. · World Models for Continual RL](https://proceedings.mlr.press/v232/kessler23a.html)：Continual-Dreamer 研究世界模型、经验选择与持续探索；模型与策略要分别测保留。
- [Cheung et al. · Non-stationary MDPs](https://proceedings.mlr.press/v119/cheung20a.html)：SWUCRL2-CW 与 BORL 把滑窗、乐观探索和变化预算联系起来；不是只换一个常数学习率。

---

## 接下来学什么、去哪探索，还能否安全回来？

现象：永远遇不到新证据，或者探索一次就失去继续学习的机会。

先区分：新奇、信息增益、学习进展、主任务收益和可恢复性可能互相冲突。

- 如何区分未学会与不可约噪声？
- teacher 是否能挑任务、重置位置或改环境？
- 没有外部救援时怎样控制不可逆风险？

### 算法线与条件

- **Counts / RND / uncertainty**：按新奇或不确定性分配行为。条件与限制：误差大不保证可学，也不保证值得冒险。
- **Learning progress / ALP-GMM / 自动课程**：优先选择仍能提升能力的目标或环境。条件与限制：训练分布由 teacher 控制，需要明确权限。
- **Reset-free / recovery / Single-Life RL**：把回到可继续学习状态纳入决策。条件与限制：给负奖励不等于具有安全保证；预训练经验也要计入。

诊断：除外在回报外记录覆盖、预测学习进展、人工干预、失能时间和恢复成功率；危险实验先在模拟器做。

基准：可控稀疏奖励任务、可参数化课程、Single-Life RL；不把有 teacher 的结果直接称为完全自主探索。

- [Burda et al. · Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)：固定随机目标与可训练预测器产生新奇信号；它不是可学习进展或安全性的直接度量。
- [Portelas et al. · Teacher Algorithms for Curriculum Learning](https://arxiv.org/abs/1910.07224)：以学习进展选择可参数化环境；需要说明 teacher 是否能重置和控制环境。
- [Chen et al. · You Only Live Once](https://arxiv.org/abs/2210.08863)：给定先前经验，在一次测试生命期中自主适应并恢复；不同于“训练完全没有先验”。

---

## 能不能让学习过程本身越用越合适？

现象：每次换环境都重新调步长，或者短期学得快却长期不稳定。

先区分：调整在线超参数、学习初始化、推断上下文、学习更新规则不是同一种“学会学习”。

- 元目标在评价什么未来，使用哪些额外数据？
- 对参数、隐藏状态还是完整 optimizer 状态求导？
- 元训练任务重置和部署持续性如何对齐？

### 算法线与条件

- **IDBD / TIDBD / meta-gradient RL**：在线改变步长或 return 参数。条件与限制：需要处理敏感度近似、尺度和训练/元评价目标。
- **MAML / context-based meta-RL / PEARL**：跨任务准备一个快速适应机制。条件与限制：元测试从头适应不证明长期累积。
- **Learned update rules / DiscoRL**：把学习规则参数化后优化其跨任务结果。条件与限制：发布规则的使用与重新进行元搜索是不同成本。

诊断：先有限差分验证导数，再匹配固定超参数调参预算；改变任务分布和学习时域，检查收益是否仍存在。

基准：标量敏感度 → 受控在线漂移 → 多任务元训练/测试；不同层次的结果分开报告。

- [Xu et al. · Meta-Gradient Reinforcement Learning](https://arxiv.org/abs/1805.09801)：对学习更新求导以调整 return 等元参数；元目标不能随便跟训练题目一起改变。
- [Finn et al. · MAML](https://proceedings.mlr.press/v70/finn17a.html)：跨任务训练易适应的初始化；重新从同一初始化适应和累积一段生命史是不同协议。
- [Abel et al. · A Definition of Continual RL](https://arxiv.org/abs/2307.11046)：将持续性落在智能体是否需要一直学习，而不只落在外部任务切换。这里采用其问题视角，不把下面的教学分类说成该文定理。

---

## 把模块接起来，怎样证明真的在持续积累？

现象：每个模块单测正常，整机却没有长期收益，结论只剩一条平均曲线。

先区分：内部活动、局部预测改善、持续参数更新和整体能力增长，需要不同证据。

- 状态变化会让哪些预测、技能和模型失效？
- 计算与记忆如何分给多个相互影响的学习器？
- 如何分开适应、保留、迁移、可塑性和安全？

### 算法线与条件

- **Life-long metrics / benchmark protocols**：测全过程，而不只测最终政策。条件与限制：eval reset、探针数据和冻结评测都会引入额外权限。
- **模块化消融 / 接口与版本管理**：固定一个对象，检查其真实下游用途。条件与限制：单模块更准不保证整个控制循环更好。
- **Alberta Plan / OaK / integrated agents**：把知识构建与长期更新接为研究路线。条件与限制：路线图不是已完成的端到端保证。

诊断：固定总预算，分别冻结状态、预测、模型或元更新；同时报告整段效用、恢复、旧知识、新学习、内存与延迟。

基准：由解析诊断到完整平台逐级增加困难。AgarCL、CSuite、Continual World 各覆盖部分问题，不存在一个分数概括整张地图。

- [Abel et al. · A Definition of Continual RL](https://arxiv.org/abs/2307.11046)：将持续性落在智能体是否需要一直学习，而不只落在外部任务切换。这里采用其问题视角，不把下面的教学分类说成该文定理。
- [Sutton et al. · The Alberta Plan](https://arxiv.org/abs/2208.11173)：将从经验持续构建知识组织为研究计划；研究路线与已验证的整机能力分开。
- [Oak Lab · The OaK Architecture](https://oaklab.ai/posts/the-oak-architecture)：用知识、抽象和元学习连接持续智能体模块；本指南把它作为架构路线，而非已完成的通用算法。
- [Mohamed et al. · AgarCL](https://arxiv.org/abs/2505.18347)：非 episodic、高维、部分可观测且持续变化的研究平台；论文中若干可塑性方法提升有限，说明困难不止一条。
- [DeepMind · CSuite 文档](https://rl-csuite.readthedocs.io/en/latest/)：持续交互的小型环境，可在复杂平台前隔离机制；continuing 不自动等于包含所有 continual 困难。
