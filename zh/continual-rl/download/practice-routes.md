# 从手算到研究：实践路线

各条路线使用不同的小环境隔离不同困难。每次换环境都会说明变化；不同任务的分数不直接排名。

<a id="practice-tracking"></a>

## 估计与适应

旧经验越来越多时，怎样仍对新变化作出反应？

先修：样本均值与期望。基础 RL 从第1步起；熟悉 DQN/PPO 的读者算完第1步可先到第4步；研究适应机制的读者先对照第3、4步的评价对象，再回查估计与采样。

### 1 · 先固定经验，检查估计

对同一臂的 8 个给定观测分别更新样本均值与常数步长估计。画出每个观测保留下来的权重。下载脚本后，在文件所在目录执行 python3 bandit_tracking_walkthrough.py，对照 rows 的逐次写入和 weights 的历史权重。

初值为0、α=0.5；第8次访问后的两项估计应为0.75与245/256。按该臂被选择的次数计数。其他臂产生的经验不会自动更新它。

[逐步推导与权重图](https://yingwen.io/zh/continual-rl/foundations/tabular/bandits/#bandit-tracking) · [下载独立手算脚本](https://yingwen.io/crl-code/tutorials/bandit_tracking_walkthrough.py)

### 2 · 区分估计、选择分数和策略偏好

转入均值不变的三臂任务。先改变初值，看同样的贪心规则怎样产生不同经验；再给两种选择规则相同的13次观测，计算 ε-greedy 概率和 UCB 分数。最后让两次奖励直接改变 softmax 偏好。

均值0.2、0.5、0.8只供读者核算，动作选择不能读取。乐观初值留在估计中，UCB 上界项只用于选择；梯度 bandit 的偏好不是动作价值。给定短历史说明更新机制，不是算法的性能排名。

[初值怎样改变尝试](https://yingwen.io/zh/continual-rl/foundations/tabular/bandits/#bandit-optimism) · [相同证据上的选择](https://yingwen.io/zh/continual-rl/foundations/tabular/bandits/#bandit-ucb) · [偏好、概率与基线](https://yingwen.io/zh/continual-rl/foundations/tabular/bandits/#bandit-gradient) · [运行探索机制算例](https://yingwen.io/crl-code/tutorials/bandit_exploration_walkthrough.py)

### 3 · 再让估计影响动作

进入均值仍为0.2、0.5、0.8的三臂 Bernoulli 完整采样循环。先读源码中的动作选择、奖励和更新，再解压旧记录：从 bandit_sample_average/seed-0/metrics.csv 取 step=1200 的 value，乘 step 恢复累计奖励；取五个 seed 的同一末点重算均值与样本标准差，核对 curves.json。随后以相同方法查看另三种规则。

seed-0 应恢复922份奖励；value 是累计平均奖励，不能再次平均所有检查点。现有图的奖励概率不变。这些结果使用各自记录的运行，不是上一阶段给定短历史的延长；UCB 和梯度方法改变的不只是估计步长。

[样本均值：代码与旧运行](https://yingwen.io/zh/continual-rl/code/bandit_sample_average/) · [常数步长：代码与旧运行](https://yingwen.io/zh/continual-rl/code/bandit_constant_step/) · [UCB：选择与实际收益](https://yingwen.io/zh/continual-rl/code/bandit_ucb/) · [梯度方法：概率与实际收益](https://yingwen.io/zh/continual-rl/code/bandit_gradient/) · [下载样本均值与常数步长旧记录](https://yingwen.io/crl-code/results/bandit_sample_average/raw-runs.zip)

### 4 · 从跟踪进入步长自适应

在线性预测例中，真实权重在中点改变。输入分布保持不变，比较常数步长与 IDBD 逐特征调整步长的预测误差。沿各源码的 update 找到预测误差与步长写入；在旧 CSV 中对照 step=600 和615的 phase，再重算变化后的同一检查点。

这里从动作选择改为给定数据上的预测，value 也换成当前无噪声目标的 MSE，不能乘环境步得到奖励。旧 CSV 未保存完整特征、权重与步长轨迹；它能重算误差汇总，不能解释每个特征为什么改变步长。

[变化目标的线性预测](https://yingwen.io/zh/continual-rl/code/constant_step_lms/) · [IDBD 的更新与对照](https://yingwen.io/zh/continual-rl/code/idbd/) · [下载变化前后的预测记录](https://yingwen.io/crl-code/results/constant_step_lms/raw-runs.zip) · [数据与计算权限](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

### 练习与解释

**还没运行：连续几个新观测为 1 时，哪种估计更快接近 1？**

必须给出初值、已见样本数和常数步长。旧样本很多时，1/n 已很小；常数步长仍保留同样的新样本权重。少量样本时则未必如此。

**手算：Q=0.25，α=0.5，得到 R=1，下一次估计是多少？**

0.25+0.5×(1−0.25)=0.625。未被选择的臂保持原值。

**补全代码：样本均值中的 n 应在何时增加？**

本次所选臂获得奖励后，先把其访问计数加一，再用新的 n 计算步长。不能使用全局环境步代替该臂的访问数。

**定位故障：MSE 降了，为什么累计奖励反而降低？**

MSE 评价哪些臂以及按什么权重平均，需要先明确。控制还取决于最优臂排序、探索动作及访问分布；并非所有估计误差对行动都同等重要。

**设计对照：怎样分清新步长规则和不同采样造成的收益？**

先让两种更新器读取同一固定经验流，再让它们各自控制行动。前者比较估计机制，后者比较包含采样反馈的完整学习器。两种结果分别报告。

**下一问题：变化越快，是否就该一直增大步长？**

增大步长会提高对新信号的响应，也提高观测噪声的影响。分别改变变化速率和噪声，再看相同误差是否对应相同困难。

### 研究问题

先写出一个对照表：同一输入与噪声序列、相同预测次数，只改变更新器；再分别改变变化速率与观测噪声，报告变化后误差和恢复所需样本数。若转为 bandit 控制，还需计入各自探索收到的全部奖励，并保存动作、访问数和步长；这些新增诊断不能由旧误差曲线恢复。

<a id="practice-prediction"></a>

## 预测与控制

学会预测更多事情，什么时候会改变行动？

先修：一步 TD、条件期望和固定策略。基础 RL 从第1步起；熟悉 DQN/PPO 的读者先做第6步，再回第2、3步检查共享与问题规格；研究预测用途的读者先做第3、5步，再按所缺机制选读分布或神经表示。

### 1 · 固定策略，先检查一个答案

先手算 A→B→终点、奖励0与1的同一轨迹，两种方法都用 α=0.1、γ=0.9，检查谁何时写入。再转入五状态随机游走：读 mc_prediction 的 update 与 td0 的 update，检查重复访问与真实终止。解压旧记录，用五个 seed 在 step=1200 的 value 重算均值和样本标准差，核对 curves.json。

两步手算首回合得到 MC 的(A,B)=(0.09,0.1)、TD的(0,0.1)。旧随机游走改用γ=1，参照为 s/6；现有 MC 用回报样本均值，TD(0) 用常数步长0.1，曲线同时包含这项步长差异。value 是五状态均匀 RMSE；CSV 未保存价值向量，能重算跨运行汇总，不能重建逐状态误差。

[同轨迹上的 MC 与 TD](https://yingwen.io/zh/continual-rl/foundations/tabular/temporal-difference/#lesson-derive) · [MC：完整回报、首次访问与结果](https://yingwen.io/zh/continual-rl/code/mc_prediction/) · [TD：代码、解析参照与结果](https://yingwen.io/zh/continual-rl/code/td0/) · [下载 MC 与 TD 的旧记录](https://yingwen.io/crl-code/results/mc_prediction/raw-runs.zip)

### 2 · 让两个状态共享参数

改用 A→B→终点的给定走廊，奖励为 +2、−1。先画两个特征方向，再算更新 B 时 A 的预测怎样改变，最后比较 accumulating trace 与 true-online TD。下载后执行 python3 shared_gradient_walkthrough.py，把每个前缀的 true-online 权重与独立前向计算逐项对照。

此处环境与前一个随机游走不同。资格迹保留的是参数更新方向；每步参数变化后，普通迹不自动等价于在线前向回报。所链接 linear_td 旧实验又回到五状态随机游走，使用特征[1,s/6]和α=0.03；它不检验这条两步走廊的在线前向等价。

[特征夹角与交叉影响](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/#prediction-sharing) · [同轨迹上的资格迹](https://yingwen.io/zh/continual-rl/foundations/approximation/traces/#traces-corridor) · [共享梯度逐步算例](https://yingwen.io/crl-code/tutorials/shared_gradient_walkthrough.py) · [线性 TD 的实测对照](https://yingwen.io/zh/continual-rl/code/linear_td/)

### 3 · 在同一世界中改变问题

回到 T 形地图。保留收到的提示和物理转移，分别问任务回报、到左门成功、到右门成功和剩余步数。

逐项填写 cumulant、目标策略、continuation 与单位。到门后不再累计，但进门的最后一个信号仍算。

[四个问题的值图与定义](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/) · [精确计算模块](https://yingwen.io/crl-code/figures/gvf-grid-predictions.mjs)

### 4 · 让答案从经验中形成

在另一个更小的两状态交替任务中，同时学习两个 GVF；中点把信号幅度减半。先从源码写出两个折扣0.8、0.5对应的四个真实分量，再检查 update 如何读取旧权重。旧运行在第601步换幅度，比较第600与615步的误差参照。

T形到门预测使用到门停止；这个交替任务没有终点，改用常数折扣。value 对两个GVF×两个状态的四项误差均匀汇总。行为策略就是目标策略，重要性比率为1；当前结果不检验 GTD 的离策略稳定性。

[向量 GVF TD](https://yingwen.io/zh/continual-rl/code/gvf_td/) · [同任务的 GTD(λ) 对照](https://yingwen.io/zh/continual-rl/code/gvf_gtd_lambda/)

### 5 · 明确预测的使用者

回到T形地图，读取已保存JSON的 learning.history 末项。用 memory、observation 与 truth 重算向左预测的六条件RMSE，再在J比较左右成功预测，核对 consumer 中三种读取规则的期望奖励。之后才进入共享神经表示与资格迹的诊断例。

预测比较、固定向左和只读步数三种规则分别得到0.8、0.5、0.5；它们使用同一批答案。T形训练允许回合重置，末项为2000回合/6000转移。共享表示旧实验另用四状态环，中点只把辅助cumulant变号，value测主问题误差；该曲线没有记录使用GVF控制行动的收益。

[GVF 如何进入行动](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/#gvf-grid-online) · [下载已保存的参数、真值与消费者数据](https://yingwen.io/crl-figures/concept-gvf-state-data.json) · [共享与分离表示的对照](https://yingwen.io/zh/continual-rl/code/gvf_shared_trace/)

### 6 · 再检查深度控制的标签与梯度

在两状态、两动作共享 ReLU 网络中，只更新当前动作，观察后继贪心动作是否改变；分别计算 DQN 与 Double DQN 标签。

这是精确的一次更新，不是训练优势证据。接着阅读现有小任务的学习曲线，将标签机制与完整控制结果分开。

[一次更新怎样改变其他状态](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/#lesson-shared-feature) · [DQN 独立手算代码](https://yingwen.io/crl-code/tutorials/dqn_shared_feature_walkthrough.py) · [DQN 与 Double 的结果](https://yingwen.io/zh/continual-rl/code/deep-dqn/) · [从单机循环进入分布式训练](https://yingwen.io/zh/continual-rl/foundations/deep/systems/)

### 7 · 把均值扩成分布，再决定怎样使用

改用两步结算任务。先从真实回报画出 CDF 与逆 CDF，再把四个分位点位置逐一写出。沿一条给定转移计算16个残差，执行一次参数更新，最后分别按均值与下尾风险选结算方式。

一个后继的四个目标分位值是冻结预测分布的表示，不是四次新环境观测。分布表示、损失梯度、分布误差和所选策略收益需逐层检验。现有 C51/QR-DQN 神经实验使用另一个有限时域链任务，仍按均值选动作。

[从概率到分位值](https://yingwen.io/zh/continual-rl/foundations/deep/distributional/#lesson-quantiles) · [逐项残差与一次写入](https://yingwen.io/zh/continual-rl/foundations/deep/distributional/#lesson-quantile-sample) · [分布的读出与分辨率](https://yingwen.io/zh/continual-rl/foundations/deep/distributional/#lesson-risk) · [运行分位数逐步算例](https://yingwen.io/crl-code/tutorials/quantile-walkthrough.py) · [C51 的神经控制与结果](https://yingwen.io/zh/continual-rl/code/deep-c51/) · [QR-DQN 的神经控制与结果](https://yingwen.io/zh/continual-rl/code/deep-qr_dqn/)

### 8 · 最后撤掉各预测独立存储的条件

保持两步结算任务，改用共享 tanh 隐层和四个分位输出。先给浮动结算四次固定标签更新，检查未训练的固定结算预测；再单独更新后继网络，观察父目标怎样移动。

固定真标签暂时隔离共享参数干扰，不是正常 TD 能获得的额外信息。两个动作的均值都保持正确，也可能有不同的分布误差。自举目标版本和有限样本 CDF 另行检查；现有收益曲线没有保存这些分布诊断。

[共享参数的实际写入](https://yingwen.io/zh/continual-rl/foundations/deep/distributional/#lesson-neural-shared) · [目标改变与损失改变](https://yingwen.io/zh/continual-rl/foundations/deep/distributional/#lesson-neural-targets) · [有限样本与分布校准](https://yingwen.io/zh/continual-rl/foundations/deep/distributional/#lesson-neural-calibration) · [运行十参数网络算例](https://yingwen.io/crl-code/tutorials/neural-distribution-walkthrough.py)

### 练习与解释

**还没运行：把外部奖励换成“每步 1”，预测的量会怎样变？**

如果在首次到门时终止累计、此前延续为 1，就得到到门的期望步数。若不设停止且过程无限持续，这个无折扣总和可能发散。

**手算：到门的概率为 0.8，途中没有新信息，成功信号只在门口出现且延续为 1，前一个位置的预测是多少？**

在图中固定目标策略下仍为 0.8。每向前一步乘 0.9 得到的是另一种带折扣的量。

**补全代码：哪一个 continuation 应进入当前 target？**

到达下一状态后，由该预测问题决定的 continuation。若本次到达触发终止，尾值乘零，而本次 cumulant 保留。

**定位故障：预测误差很小，选门仍错，先查什么？**

查预测的是哪条目标策略，以及控制器如何使用它。一个总是向左的策略价值，并不同时回答向右会怎样。还要查信息是否足够区分两种历史。

**设计对照：两个动作均值始终正确，就排除共享表示干扰了吗？**

没有。两步结算例中，浮动动作四次写入使固定动作的 W₁ 从0升至0.125，均值仍为1.5。先冻结标签并检查另一个输入，再单独改变后继估计和采样。这样才能把参数干扰、目标漂移和有限样本误差分开。

**下一问题：哪个新预测值得长期保留？**

先明确谁使用它、替代哪项计算和维护成本。可比较移除预测、使用精确答案与使用学得答案，区分信息价值和估计困难。

### 研究问题

预测的价值取决于使用者。把“问题选得不好”“答案学得不好”和“控制器不会用”分别形成干预，才能判断应改问题发现、学习算法还是控制接口。

<a id="practice-skills"></a>

## 技能与规划

一个多步行为怎样成为可学习、可预测、可规划的动作？

先修：折扣回报、TD、价值迭代。基础 RL 先做第1、2步；熟悉 DQN/PPO 的读者从第3步辨认跨步模型；研究技能或规划的读者先列第6、8步的交互与计算预算，再回查模型怎样获得。

### 1 · 先学习一次动作的后果

在随机道路任务中，先取得一次安全退出和四次指定探索的真实经验。按联合后果计数学习模型，分别计算样本备份与期望备份，再执行模型所选的真实行动。

这不是四房间任务。随机道路的成功率未知；给定探索日程使智能体能取得过桥样本。期望备份消除的是给定模型下的抽样波动，不能消除模型本身的估计误差。模型查询也不会增加真实观测计数。

[经验怎样形成随机模型](https://yingwen.io/zh/continual-rl/foundations/tabular/planning/#stochastic-model-task) · [两种备份与计算成本](https://yingwen.io/zh/continual-rl/foundations/tabular/planning/#stochastic-model-backup) · [规划如何改变下一段经验](https://yingwen.io/zh/continual-rl/foundations/tabular/planning/#stochastic-model-feedback) · [运行随机模型与规划算例](https://yingwen.io/crl-code/tutorials/stochastic-model-walkthrough.py) · [确定性链上的 Dyna 对照](https://yingwen.io/zh/continual-rl/code/dyna_q/)

### 2 · 给规划计算设一个明确预算

改用模型已知的四分支道路。先比较同一种备份按起点到终点、终点到起点的两种次序，再固定次序比较抽一个后果与展开全部后果。计算完以后，检查起点究竟选择过桥还是退出。

已知模型暂时移除了估计误差，后继价值却仍须计算。一次备份的模型查询、后果展开、价值读取与写入分别计数；调度开销另外列出。估计均方误差下降不保证每个有限预算上的动作更好。

[四分支任务与工作量](https://yingwen.io/zh/continual-rl/foundations/tabular/planning/#planning-budget-task) · [计算顺序与价值传播](https://yingwen.io/zh/continual-rl/foundations/tabular/planning/#planning-budget-propagation) · [预算如何影响真实决策](https://yingwen.io/zh/continual-rl/foundations/tabular/planning/#planning-budget-decision) · [运行分支规划算例](https://yingwen.io/crl-code/tutorials/planning-budget-walkthrough.py)

### 3 · 执行一个给定技能

沿四房间地图走到门口。每个原始步记录位置、动作、真实奖励、到达后的终止决定和已经经过的步数。下载后执行 python3 option_model_walkthrough.py；先对照 execute 的四步轨迹，再读 outcome 的奖励和与终点核。

先固定启动集、内部策略、终止函数。门口是技能边界，不是环境终点。每步−1、γ=0.9的四步样本应得到奖励和−3.439、终点权重0.6561；代入门口价值10，target为3.122。

[启动、执行与交回控制](https://yingwen.io/zh/continual-rl/construction/options/#lesson-option-components) · [下载执行、模型与规划算例](https://yingwen.io/crl-code/tutorials/option_model_walkthrough.py)

### 4 · 把经验变成后果模型

继续读同一四房间脚本的 boundary_update、intra_pass 与 backup：比较整段写入和逐步 bootstrap。然后进入旧 option_model 结果的七格随机链，按源码的固定内部策略与停止概率重列模型规格，再从五个 seed 的末点 value 重算联合误差均值和样本标准差。

终点权重之和通常小于1，包含折扣和真实时长。四房间脚本最后规划另把奖励换成到Z得1、其他步0，并重建模型。旧随机链则向右概率0.8、β=0.3、γ=0.9，误差合并6个起点×8个分量；它既未延续四房间奖励，也未学习技能策略。

[同例的模型学习与规划](https://yingwen.io/zh/continual-rl/construction/options/#lesson-option-practice) · [返回四房间的模型与规划代码](https://yingwen.io/crl-code/tutorials/option_model_walkthrough.py) · [七格随机链：逐步与完整段模型](https://yingwen.io/zh/continual-rl/code/option_model/) · [下载七格链模型的旧记录](https://yingwen.io/crl-code/results/option_model/raw-runs.zip)

### 5 · 让子目标的候选也来自经验

转到六地点运输世界。用已观察道路和当前可用但未尝试的动作生成待探索地点，算出到达并尝试一次的成本。执行所选子目标，重算候选，最后让外部任务规划读取技能的奖励与终点模型。核对脚本 budgets 中5次初始采样、5次探索、3次部署。

地点表示、可用动作接口、首次到达停止规则和评分仍由设计者给定。13次真实转移之外允许4次回到起点，重置代价暂未计入。发现一条新道路可能得到零外部回报；图覆盖率与任务收益要分别计算。

[候选目标从哪些记录产生](https://yingwen.io/zh/continual-rl/construction/goals/#lesson-goal-discovery) · [两次探索与一次真实部署](https://yingwen.io/zh/continual-rl/construction/goals/#lesson-discovery-loop) · [选目标的标准与剩余问题](https://yingwen.io/zh/continual-rl/construction/goals/#lesson-discovery-criteria) · [运行目标发现算例](https://yingwen.io/crl-code/tutorials/goal_discovery_walkthrough.py)

### 6 · 区分控制收益和规划计算

先在原子动作与 option 的链式控制中读真实交互预算，再在七格链精确模型规划中读状态备份预算。解压后找到 option_value_iteration/seed-0/metrics.csv 的末行，核对 step、primitive_backups、option_backups 和 value_start；再对照原始动作规划的同一时刻。

两个现有实验环境不同，横轴也不同。step=1200表示1200次状态最大化备份：含技能的方法评估2400个原始候选与1200个技能候选，原始动作方法只评估2400个候选。模型预计算另计；控制图的value则是冻结贪心折扣回报，不能当作真实训练累计奖励。

[SMDP Q-learning 控制](https://yingwen.io/zh/continual-rl/code/option_smdp_q/) · [精确模型上的规划](https://yingwen.io/zh/continual-rl/code/option_value_iteration/) · [下载规划值与候选备份计数](https://yingwen.io/crl-code/results/option_value_iteration/raw-runs.zip)

### 7 · 世界改变，却不告诉智能体

保留六地点布局，改为到X得1、上路先关闭后重开。让累计与近期模型分别控制行动，再加入按访问年龄重查路线的规则。先列共同的28次校准交互，再比较其后96个真实时钟内的奖励、失败探路和绕路成本。

这一任务把回到起点也计作时钟，已经改变上一步的重置预算。模型只有实际经过道路后才能收到新证据；近期更新能修正已观察的变化，却未必重新访问弃用道路。重查规则的年龄阈值仍给定；在不变世界与永久关闭对照中也要计入成本。

[未被通知的变化](https://yingwen.io/zh/continual-rl/construction/models/#lesson-model-change) · [模型记忆与观测噪声](https://yingwen.io/zh/continual-rl/construction/models/#lesson-model-aging) · [重新采样及其成本](https://yingwen.io/zh/continual-rl/construction/models/#lesson-model-recheck) · [运行变化模型算例](https://yingwen.io/crl-code/tutorials/model-change-walkthrough.py)

### 8 · 再让技能改变

转入七状态持续随机环。学习两个给定子任务，并在技能策略改变后清除对应旧模型与高层值。先从旧记录的 seed-0 第1100与1200步 lifetime_reward 之差重算最近100步 value，再以第1200步累计量除以1200重算 average_reward；最后比较近期模型、累计模型与不规划的完整运行。

最近100步奖励、生命期累计奖励率和算法内部率估计是不同量。第1200步应分别得到0.43与约0.260833。每次宏动作完成才有高层更新；核对 model_backups=4×real_backups、subtask_backups=2×step。预算结束时未完成技能的奖励照常计入，但不能伪造完整模型样本。

[近期模型：收益与各类时钟](https://yingwen.io/zh/continual-rl/code/integrated_recent_model/) · [累计模型的同系统对照](https://yingwen.io/zh/continual-rl/code/integrated_cumulative_model/) · [下载环任务的收益与更新计数](https://yingwen.io/crl-code/results/integrated_recent_model/raw-runs.zip) · [模块间的学习与使用](https://yingwen.io/zh/continual-rl/construction/architectures/)

### 练习与解释

**还没运行：option 停止后，后续价值应为零吗？**

仅停止 option 时，高层还会选择下一项行为。只有真实环境终止，或所定义目标的 continuation 为零，才移除后续价值。

**手算：四步每步 −1，γ=0.9，门口后续价值为 10，target 是多少？**

−1−0.9−0.81−0.729+0.6561×10=3.122。用一次 γ 代替 γ 的四次方会高估远处后果。

**补全代码：高层记录的奖励应来自子目标还是外部任务？**

本路线比较原任务回报，高层记录真实外部奖励。内部技能可使用单独的子任务信号；不能覆盖同一转移上的真实奖励字段。

**定位故障：最近实际经过的道路都预测正确，为什么始终走较长路线？**

错误可能留在已被放弃的道路上。智能体不经过它，就没有新证据说明它重新开放。把近期模型与访问年龄重查分别开关，并保留永久关闭的反例，才能区分估计速度、数据获取和探索成本。

**设计对照：两种规划器都做了十次备份，计算成本就相同吗？**

先看备份数在数什么。七格规划的十次step分别评估30个或20个候选行为；四分支算例还把后果展开与价值读取逐项计数。先固定调度比较备份方式，再固定备份方式比较次序，另计模型构建、调度与真实交互。高层决策次数减少也不表示技能学习更省经验。

**下一问题：最省动作的探索目标，是否也最有利于主任务？**

运输例中先选 A，因为覆盖一个未知动作只需2步；这次探路的外部回报却为0。到 C 需要3步，新道路随后提供更高的任务回报。比较时同时记录发现成本、所获信息、后续使用次数和剩余任务期限；未知道路的真实收益不能提前给目标选择器。

### 研究问题

用同一环境与任务分别固定技能、替换精确模型、撤掉规划，定位收益来自执行、后果预测还是使用。比较时同时列真实步数、候选备份和模型构建成本；变化道路例还需比较近期更新与重查规则，并把失败探路计入收益。若要学习重查时机，先规定可读的访问年龄、模型误差和剩余预算，以及为哪项实际收益付出探路代价。

[研究作业：模型记忆与规划：完成一份持续控制研究作业](https://yingwen.io/zh/continual-rl/download/practice-model-memory-planning.md)

<a id="practice-policy-control"></a>

## 策略梯度与控制

优化器确实降低了损失，为什么行动仍可能变差？

先修：TD 目标、条件期望、链式求导。基础 RL 先补深度价值学习与策略梯度，再从第1步起；熟悉 DQN/PPO 的读者从第2步核对标签后进入连续动作；研究控制失败的读者先重算第5步，再回第1、3步区分目标与critic错误。

### 1 · 先检查正在改善哪个目标

在三步送货任务中枚举尽早送达、延迟送达和超时三条路径。先按送达奖励选择，再加入停留门口的奖金，检查规划器实际选择是否改变。随后推导势差塑形，并分别删去折扣项与终端归零。下载后执行 python3 reward-design-walkthrough.py，按路径核对原任务与各训练奖励的排序。

这里使用已知模型和精确规划，没有 critic 估计误差。位置奖金使超时的训练分数最高；改善优化器不能修复这个奖励规格。正确势差保持所给目标的排序，也不会自动修复一个错误目标。

[同一任务里的目标与代理](https://yingwen.io/zh/continual-rl/foundations/reward-design/#lesson-setting) · [势差与动作排序](https://yingwen.io/zh/continual-rl/foundations/reward-design/#lesson-derive) · [终止、截断与折扣](https://yingwen.io/zh/continual-rl/foundations/reward-design/#lesson-example) · [运行奖励设计算例](https://yingwen.io/crl-code/tutorials/reward-design-walkthrough.py)

### 2 · 固定一条轨迹，检查策略改进的依据

在三步通关任务中，先算 TD 残差与 GAE，再用同一组优势计算 PPO 裁剪目标。枚举真实成功概率，核对样本目标与真实收益。随后读已有 PPO 的 update 和旧记录：找到 step=1200 的 samples、updated_batches 与 value，再按五个训练种子重算冻结评估。

手算中旧策略、优势与critic标签固定，γ=1、λ=0.5。旧完整实现另用12步期限链、γ=0.99、λ=0.95，批内归一化优势；20批采样不等于20次策略梯度。图中value为确定性argmax策略的未折扣回报，不是三步任务成功率或PPO损失。

[GAE 的逐步计算](https://yingwen.io/zh/continual-rl/foundations/deep/policy-gradient/#lesson-example) · [同一轨迹上的 PPO](https://yingwen.io/zh/continual-rl/foundations/deep/trust-region/#lesson-example) · [运行 GAE/PPO 算例](https://yingwen.io/crl-code/tutorials/gae_ppo_walkthrough.py) · [PPO：完整采样、更新与旧结果](https://yingwen.io/zh/continual-rl/code/deep-ppo/) · [下载 PPO 的冻结评估与采样批次](https://yingwen.io/crl-code/results/deep-ppo/raw-runs.zip)

### 3 · 从动作概率转到连续动作

改用两步定位任务。先求已知后果下的真实动作价值，再让 actor 沿一个给定的错误 critic 更新。分别检查动作导数、critic 参数冻结和目标网络的更新时间。

离散通关任务已换为连续任务。DDPG/TD3 的梯度经 critic 的动作输入回到 actor；它不经过真实环境。双 critic 的最小值用于 TD3 标签，actor 仍优化 Q1。

[连续控制的问题与推导](https://yingwen.io/zh/continual-rl/foundations/deep/deterministic-control/) · [运行连续动作算例](https://yingwen.io/crl-code/tutorials/continuous_control_walkthrough.py)

### 4 · 再改变目标，而不只是添加噪声

继续使用 continuous_control_walkthrough.py 的 sac_sample 与 walkthrough 中的 sac 输出，计算重参数化动作、变换后密度和温度梯度。把动作单位放大两倍，检查密度与目标熵是否一起改变。

SAC 的价值还累计未来熵。普通 target 与 soft target 的数值不能直接排性能；外部奖励、熵和温度需分别解释。此处温度梯度核与下一步固定温度的完整训练采用不同设置。

[密度、熵与温度](https://yingwen.io/zh/continual-rl/foundations/deep/entropy-control/#lesson-density) · [自动温度的梯度](https://yingwen.io/zh/continual-rl/foundations/deep/entropy-control/#lesson-temperature) · [返回连续动作与温度算例](https://yingwen.io/crl-code/tutorials/continuous_control_walkthrough.py)

### 5 · 让估计与采样形成完整反馈

进入已有 BoundedLQ 的完整 replay 训练实现，先读环境、真实终止、探索阶段与更新计数。解压各旧记录，用五个 seed 在 step=1200、phase=evaluation 的 value 重算均值和样本标准差；对照同一行的 critic_updates、actor_updates 和 replay_size，解释不同方法所用的计算。

这是另一个40步有限时域任务，动作改为[−1,1]、训练γ=0.99。冻结评估先在同一组12个初态上取未折扣外部回报均值，再跨5个训练种子汇总。第1200步两种方法都有1169批critic更新，DDPG有1169次actor更新、TD3有584次；SAC温度固定0.1，执行tanh(mean)评估。

[DDPG：完整循环与对照](https://yingwen.io/zh/continual-rl/code/deep-ddpg/) · [TD3：延迟更新与对照](https://yingwen.io/zh/continual-rl/code/deep-td3/) · [SAC：固定温度与对照](https://yingwen.io/zh/continual-rl/code/deep-sac/) · [下载 DDPG 与 TD3 的旧记录](https://yingwen.io/crl-code/results/deep-ddpg/raw-runs.zip) · [下载 SAC 的旧记录](https://yingwen.io/crl-code/results/deep-sac/raw-runs.zip)

### 6 · 把评价对象扩展到完整学习过程

先声明 40 步期限与重置是否保留，并评价跨回合持续更新的完整学习器。一个生命期可以包含多个回合；若另行撤掉期限，再定义没有任务终止的控制目标。随后一次只改变一种条件，例如限制 replay 或改变环境规律。为每个真实交互记录更新前收到的奖励，再单独保留冻结诊断。

现有 BoundedLQ 日志只有周期性冻结评估，不能据此重算训练期间的生命期奖励。新实验需新增逐步奖励记录，并同时报告环境步、critic 批次、actor 更新和内存预算。

[同一迷宫的两种评价](https://yingwen.io/zh/continual-rl/foundations/objectives/#lesson-noisy-lifetime) · [逐项恢复学习器状态](https://yingwen.io/zh/continual-rl/algorithms/streaming/#lesson-learner-state) · [在线收益与冻结诊断](https://yingwen.io/zh/continual-rl/experiments/#lesson-freeze) · [控制目标与学习过程](https://yingwen.io/zh/continual-rl/algorithms/control/) · [平均奖励的目标选择](https://yingwen.io/zh/continual-rl/algorithms/average-reward/#lesson-discount-choice) · [流式学习的数据权限](https://yingwen.io/zh/continual-rl/algorithms/streaming/)

### 练习与解释

**还没运行：critic 在当前动作上估计很准，actor 的方向就可信吗？**

数值准确不保证局部斜率准确。可以在动作 a=0 处令两个函数值相同而导数符号相反；actor 的路径梯度使用的是斜率。还需检查周围动作的数据覆盖。

**手算：Q(a)=−(a−0.8)²，a=tanh θ，θ=0，Q 对 θ 的导数是多少？**

动作导数为 −2(a−0.8)=1.6，tanh 在零点的导数为 1，所以乘积为 1.6。最小化 −Q 时 loss 导数是 −1.6，梯度下降增大 θ。

**补全代码：actor 更新时，应该冻结哪个对象？**

冻结 critic 参数，但保留 critic 对 actor 动作的导数；不要 detach 动作，也不要把整个 actor loss 放进 no_grad。critic 标签则应整体停止梯度。

**定位故障：同样的归一化策略，动作单位放大两倍后温度方向变了，先查什么？**

一维物理动作密度需减去 log 2，微分熵相应增加 log 2。若要保持同一熵约束，目标熵也应增加 log 2。检查的是同一约束是否被正确换算，而非只看温度优化器。

**设计对照：TD3 与 DDPG 在相同环境步数下不同，能归因于双 critic 吗？**

两者还改变目标动作平滑和 actor/target 更新频率；critic 数量也改变计算量。完整算法比较回答如何选择方法，机制归因则需逐项干预，并另报或匹配相应更新预算。

**下一问题：末次冻结评估更好，就代表持续学习更好吗？**

不一定。训练期间的探索成本、失败和恢复都可能被末次评估遗漏。带噪声岔路迷宫给出一个可逐圈计算的反例：完整生命期胜留败换 18.48 > 始终向左 15.2，但末态冻结到 A 后再测八圈却是 6.56 < 7.2。先定义生命期准则，再用冻结诊断解释机制；不能从冻结回报曲线倒推出未记录的行为奖励。

### 研究问题

同一控制失败可能来自价值斜率错误、旧经验失配，或 actor 更新过快。先在固定经验上替换 critic 或冻结 actor，再让各方法独立采样；这样可以分开更新机制与采样反馈。将初始探索和变化后的损失计入同一预先规定的生命期目标，才进入持续控制的评价。

<a id="practice-action-evidence"></a>

## 行动与经验

行动决定了能得到哪些证据。学习又怎样改变下一次行动？

先修：完整回报、ε-greedy、条件概率；递归状态另用链式求导。基础 RL 从第1步起；熟悉 DQN/PPO 的读者先做第4、5步，再回第1步检查数据如何改变；研究记忆的读者先做第6步，辨认第7、8步新增的模式与写入条件。

### 1 · 让 MC 估计真正改变下一回合

从两步、两动作任务开始。先冻结均匀策略，按给定均匀数实际选动作；终止后更新回报均值，再改善策略。依次手算 AB、BA、AA 三条回合轨迹，下载后执行 python3 mc-control-walkthrough.py，对照 assigned_episodes 的写入与 adaptive_enumerations 的完整分支。

指定路径只是一条可能轨迹。第二回合的概率取决于第一回合产生的策略；16个分支并不等权，须用各自概率加权。两个回合后的当前策略价值期望为269/125；它与指定三回合末态的18/5评价不同。策略改善定理使用准确的 qπ，不保证每次有噪声的样本更新都提高收益。

[控制循环与三回合轨迹](https://yingwen.io/zh/continual-rl/foundations/tabular/monte-carlo/#mc-control-loop) · [旧回报与当前价值](https://yingwen.io/zh/continual-rl/foundations/tabular/monte-carlo/#mc-control-target) · [运行 MC 控制逐步算例](https://yingwen.io/crl-code/tutorials/mc-control-walkthrough.py)

### 2 · 把探索行为与要改善的策略分开

保留两步任务，行为策略继续探索，目标策略则对 Q 贪心。从回合末端倒着计算 G、权重 W 和累计权重 C。检查先更新当前状态动作、再改善目标策略、最后决定是否停止向前处理的顺序。另用重复访问的短回路区分首次访问、每次访问和权重和。

Q 已条件于当前动作，当前写入的权重校正后续动作；即使该动作在改善后不再贪心，这个状态动作已经完成本次写入。行为分母保留采样时版本；不断改善的目标不能直接套固定策略估计的无偏性结论。

[倒序控制与下一回合行为](https://yingwen.io/zh/continual-rl/foundations/tabular/monte-carlo/#mc-offpolicy-loop) · [重复访问与后缀权重](https://yingwen.io/zh/continual-rl/foundations/tabular/monte-carlo/#mc-offpolicy-repeat) · [运行加权离策略控制算例](https://yingwen.io/crl-code/tutorials/mc-offpolicy-control-walkthrough.py)

### 3 · 分开行为收益与冻结诊断

进入已有的六格链式实验，比较 MC 控制与 Watkins Q(λ)。先找到源码中 MC 等待真实终止的位置，再解压旧记录：从五个 seed 的 step=1200、value 重算均值和样本标准差，并用五步向右的折扣和独立核对末态贪心值。

这里已换任务：γ=0.95、ε=0.1，非终止奖励−0.01，入终点奖励1。四次−0.01再得1给出0.7774075，与旧末点一致。图中是当前确定性贪心策略的精确折扣价值，不是 ε-soft 训练行为的累计奖励。单条平线不能说明是否刚完成一次回合；日志没有保存回合边界。

[完整控制实现、基线与结果](https://yingwen.io/zh/continual-rl/code/mc_control/) · [下载 MC 与 Watkins 的旧记录](https://yingwen.io/crl-code/results/mc_control/raw-runs.zip)

### 4 · 给探索决策足够的持续时间

改用十步走廊：安全结算得 0.25，远端成功率未知。比较每步重抽模型与整轮保持模型；真正到达远端才更新后验，再让新后验决定下一轮是否继续探路。

普通移动的零和安全奖励在两个候选模型下完全相同，不是端点失败证据。整轮保持模型更常到达远端，也可能付出更多动作和机会成本；相同轮数不等于相同交互预算。

[任务、信息和可达概率](https://yingwen.io/zh/continual-rl/foundations/deep/exploration/#deep-exploration-task) · [后验如何改变后续行动](https://yingwen.io/zh/continual-rl/foundations/deep/exploration/#deep-exploration-feedback) · [价值头与随机先验](https://yingwen.io/zh/continual-rl/foundations/deep/exploration/#deep-exploration-heads) · [运行探索与后验算例](https://yingwen.io/crl-code/tutorials/deep-exploration-walkthrough.py)

### 5 · 重放经验时，先重建决策所需的记忆

在两门仓库中，入口标签决定哪一侧有货物，之后只看到灰色走廊。用旧参数、当前参数和目标网络各自展开同一记录，比较保存状态、零初始化与预热，再算终止与非终止 TD 标签。之后读 RTRL/TBPTT 旧源码的 gradient：从旧 CSV 的 step、element_steps 与 value 核对输入预算，再重算同一检查点的跨种子误差汇总。

前向处理了多少历史、哪些位置产生损失、梯度能穿过多远，是三项设置。预热段没有直接损失，不代表其梯度已截断。现有 RTRL/TBPTT 结果另用每段8输入、活动逐段清零的监督预测任务，1200段对应9600个输入；value是更新前平方误差的EMA，不是仓库控制收益，稀疏CSV也不能恢复每段原始误差。

[旧状态与当前重建](https://yingwen.io/zh/continual-rl/foundations/deep/partial-observability/#recurrent-replay-state) · [相同输出、不同梯度](https://yingwen.io/zh/continual-rl/foundations/deep/partial-observability/#recurrent-replay-gradient) · [运行递归回放算例](https://yingwen.io/crl-code/tutorials/recurrent-replay-walkthrough.py) · [RTRL：独立序列的预测结果](https://yingwen.io/zh/continual-rl/code/rtrl/) · [TBPTT：三步梯度对照](https://yingwen.io/zh/continual-rl/code/tbptt/) · [下载递归预测的旧记录](https://yingwen.io/crl-code/results/rtrl/raw-runs.zip)

### 6 · 撤掉回合重置，保留整段交互历史

进入带提示的选门任务。门后继续走，追踪在线参数更新怎样改变下一次动作与奖励。逐步核对五次真实转移、一次初始化和零次 reset，再比较保留活动、仅 detach 和清空活动的后果。

短窗口的结束不是环境终止。历史参数共同形成当前活动与敏感度；本例的导数不穿过过去的参数更新，也不是生命期奖励梯度。奖励本身携带信息，这里学习即时选门奖励的预测，不是求出最优长期价值。

[没有重置的递归控制](https://yingwen.io/zh/continual-rl/construction/state/#lesson-continuing-control) · [运行持续状态算例](https://yingwen.io/crl-code/tutorials/continuing-state-walkthrough.py)

### 7 · 把信息保留与信用路径分开

再改三个条件：每次回到入口重新抽取隐藏模式，提示与选门之间加入可变长度的灰色走廊，并用显式提示写入规则构造状态。比较最近观测窗口、保留活动但截断梯度、完整状态敏感度和衰减记忆。随后跟踪参数写入怎样改变下一次真实选门。

旧奖励只揭示上一隐藏模式。相同前向状态不保证相同学习梯度；有非零梯度也不保证一次更新足以改变动作。这个算例学习即时选门奖励的预测，固定输入求导，不把它当作完整生命期目标的梯度。

[长延迟下的信息、状态与梯度](https://yingwen.io/zh/continual-rl/construction/state/#lesson-delayed-memory-budget) · [运行延迟记忆与信用算例](https://yingwen.io/crl-code/tutorials/delayed-state-walkthrough.py)

### 8 · 让写入多少也从反馈中学习

保留同一选门环境，撤掉入口强制覆盖：旧活动与新提示按可学习的门混合。沿六次到达逐项计算门梯度，再只取消第三次参数更新，检查第六次选门为何不同。

学习后的门只作用于后续写入，不重算已经形成的活动。六次门反馈共43个真实步；完整4096路径的加权结果与指定分支分别报告。此时灰路保持率仍给定为1，门的预测学习不等于已经学会整个记忆架构或生命期策略梯度。

[混合、梯度与实际选门](https://yingwen.io/zh/continual-rl/construction/state/#lesson-learned-memory) · [运行可学习写入门算例](https://yingwen.io/crl-code/tutorials/learned-memory-walkthrough.py)

### 练习与解释

**还没运行：下一回合策略改善了，旧 MC 回报也会变成新策略的样本吗？**

不会。已经观察到的奖励不变。若旧回报包含后来被改变的决策，它仍由采样时的后续策略生成。两步例中 S₀ 的回报受此影响；S₁ 的动作后立即终止，其动作价值不依赖后续策略。

**手算：条件于 S₀,A，先从旧策略取一条回报，再从预先固定的新策略取一条，均值的期望是多少？**

本例旧策略在 S₁ 以 0.5 选 A，新策略以 0.9 选 A。两回报的期望为 2+2×0.5=3 和 2+2×0.9=3.8，因此等权均值的期望为 3.4，不是当前价值 3.8。这里两版策略预先固定，不能把此独立诊断直接当作自适应策略序列的分布。

**补全代码：倒序 MC 控制发现当前动作已不贪心，应当先 break 吗？**

先完成当前 Q 与 C 的写入，再根据新 Q 更新贪心动作，随后才检查 break。当前 Q 已条件于当前动作；不匹配使更早位置所需的后缀权重为零，却不取消已经完成的本次写入。继续处理时才用采样时的行为概率更新下一位置的 W。

**定位故障：远端失败后 q=0.2，下一轮安全结算得到 0.25，后验应该继续下降吗？**

不应该。两个候选模型都确定地产生安全奖励 0.25，似然相同，q 仍是 0.2。途中普通移动的零也不是端点试验。只有真正到达端点后抽取的 Y 才是本任务中的新证据。

**设计对照：整轮保持模型更常到达远端，就一定更节省经验吗？**

不一定。初始 q=0.5 时，保持模型的到达概率为 0.5，平均动作数为 5.5；每步重抽分别为 1/1024 和约 1.998。要比较同一经验预算下的价值，应另行规定总动作预算及耗尽时的处理，不能只比较相同轮数。

**下一问题：删去入口提示，旧奖励还能帮助下一次选门吗？**

短周期持续选门任务的 Z 不变，选门后可由 Z=dr 恢复模式，保留奖励的状态因而能帮助下一次选择。长延迟与写入门任务却在每次回到入口独立重抽 Z；旧奖励只揭示旧模式，不能替代当前提示。重放没有提示的灰色前缀，也无法从零重建已经缺失的信息。判断记忆是否有用，先检查世界中哪些关系仍然保持。

### 研究问题

依次改变策略更新时间、探索承诺时间和活动重置规则，并记录实际动作、已到达反馈与更新前后的内部状态。若收益变化，先判断它来自获得了不同数据，还是对相同数据作了不同更新；再让各学习器独立行动，比较预先规定预算内的完整收益。