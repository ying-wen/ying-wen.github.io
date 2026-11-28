# 表格、线性与信用机制的独立扩展

这组实验围绕四个问题组织：怎样估计价值；怎样传播多步信用；怎样处理无终点的奖励流；怎样在约束或竞争下选择策略。每种方法都有一个明确的小型任务，并与同任务的基线比较。公式检查、实验结果与原论文规模复现是不同层次的证据。

每个文件包含问题、公式、初值、更新顺序及限制。run(seed,steps,emit)实际采样或执行已知模型备份；通用文件共享环境、采样、测量和线性代数。先选择下面一个实验，再查看源码中的更新式与预算时钟：

    python3 implementations/extended_classic/ordinary_is.py --steps 1200 --seeds 0 1 2 --out results/OPE_NEW_RUN
    python3 implementations/extended_classic/gradient_eligibility_traces.py --steps 1200 --seeds 0 1 2 --out results/GTD2_NEW_RUN
    python3 -m unittest discover -s tests -p test_extended_classic.py -v

输出必须为新目录。比较同时绑定family、task、metric、unit和budget；计算成本、表示容量和信息差异仍须单独解释。

## 1. Bellman与MC

| 文件（均位于implementations/extended_classic） | 原覆盖编号 | 更新与对照 |
|---|---|---|
| iterative_policy_evaluation.py | core-policy-evaluation | 同步期望Bellman扫描；对照原地扫描policy_evaluation_inplace |
| every_visit_mc.py | core-every-visit-mc | 每次状态访问的完整回报样本均值；对照first_visit_mc_reference |
| mc_control.py | core-mc-control | 完整回合首次状态动作访问，回合结束后ε-soft改善 |

策略评价采用已知六格链、固定π=(0.2,0.8)、γ=0.95，初值0，每扫描5个状态备份。真值来自独立线性方程解。原地和同步使用不同轮内信息顺序；扫描不是环境交互。

MC预测采用从3开始的五状态无偏随机游走，终点左右奖励0/1，γ=1，真值s/6只用于测量。Every-visit接受同回合相关回报，不把它们称为独立样本。预算末未完成回合不更新，也不补终点。MC control固定ε=0.1，不满足一般GLIE最优收敛的全部条件。

所有控制链非终点奖励−0.01、终点1、γ=0.95、起点0。冻结贪心策略用确定性终点或无限循环解析评价；一直向左回报精确−0.2，没有100步截断遗漏。

## 2. IS与DR：评价组件

| 文件 | 原覆盖编号 | 核心表达式 |
|---|---|---|
| ordinary_is.py | core-ordinary-is | ΣWG/N，W=Πρ |
| weighted_is.py | core-weighted-is | ΣWG/ΣW |
| per_decision_is.py | core-pdis | Σγᵗ(Πᵢ≤ₜρᵢ)rₜ |
| sequential_dr.py | core-sequential-dr | DRₜ=Vhatₜ+ρₜ(rₜ+γDRₜ₊₁−Qhatₜ) |

四方法共用三步回合：μ=(0.5,0.5)、π=(0.2,0.8)，Bernoulli动作均值0.1/0.9，γ=0.9。每预算单位一条真实三转移回合，另记3倍原始步数。目标回报0.74(1+0.9+0.9²)只用于测量累计估计误差。

比率函数拒绝零行为概率及非法概率，允许零目标概率。Weighted IS有限样本通常有偏，总权重0时明确返回0。DR的经验奖励模型只看旧回合，Qhat与Vhat来自同一π；先估计当前DR，之后模型才吸收本回合。准确比率使有偏模型仍可保持期望正确，不等于任意同样本拟合都无偏。这些文件不训练控制策略，不测试未知行为概率或长轨迹方差。

## 3. 多步预测和控制迹

| 文件 | 原覆盖编号 | 范围 |
|---|---|---|
| tree_backup.py | core-tree-backup | 固定随机策略三步期望分支 |
| q_sigma.py | core-q-sigma | 固定策略三步σ=0.5混合 |
| retrace.py | core-retrace | λ=0.8截断误差传播，当前ρ=1 |
| watkins_q_lambda.py | core-watkins-q-lambda | 表格控制、累积迹、非贪心动作剪迹 |
| semi_gradient_sarsa.py | core-semi-gradient-sarsa | 每动作3个共享多项式特征的线性SARSA |

前三者π=μ=(0.2,0.8)，Q=0，测量固定目标策略十个状态动作RMSE。三步前缀成熟后更新最早Q，真终点冲洗短目标，未成熟预算尾不伪终止。Q(σ)的σ=0/1分别退化Tree Backup/SARSA；没有一般离策略Q(σ)校正。Retrace保留完整目标动作期望，c仅作用下一误差传播；on-policy运行不能展示离策略控制或函数逼近收敛。

Watkins先用旧Q判断下一动作是否贪心（并列也保留），当前资格加1，传播本次误差，然后才衰减或清迹。终点奖励先获过去资格信用再清迹。半梯度SARSA有6共享权重，表格方法10独立值；同任务比较也需保留容量差异，不证明同表示优势。

## 4. 线性与平均奖励

| 文件 | 原覆盖编号 | 目标 |
|---|---|---|
| gradient_mc.py | core-gradient-mc | 完整回报平方损失的真实梯度 |
| lstd.py | core-lstd | A=Σx(x−γx′)ᵀ、b=Σxr |
| differential_td_prediction.py | core-differential-td | 同一旧δ更新差分价值及奖励率 |
| relative_value_iteration.py | core-rvi | 参考状态归一化的已知模型迭代 |
| average_smdp_option.py | core-smdp-rate | 旧平均持续时间归一化的固定策略选项更新 |

Gradient MC/LSTD同用随机游走、[1,s/6]特征、w=0。终点特征零，预算截断不补终点。LSTD固定加10⁻⁵I避免有限前缀奇异；不是未正则原始LSTD。手算测试另外验证ridge=0可逆解和奇异报错。均匀状态RMSE与采样占用加权投影目标不能混称。

Differential TD/SMDP同用无终点两状态交替流，从0开始，Bernoulli均值0.2/0.8，率0.5、差分偏差差0.3。选项持续1或3原始步，收集未折扣总奖励；δ=R−gLold+v′−v，Δv=αδ/Lold、Δg=ηΔv，最后更新L。预算中未完成选项不更新。只实现固定策略expected-duration子机制，没有一般多选项控制、intra-option、模型学习或中断。

RVI另用已知非周期两状态模型：自环0.8、跨状态0.2、奖励0.2/0.8，偏差差真值1.5。每扫描2备份；unnormalized_value_iteration对照保留公共增长项。相对误差曲线可完全相同，这是坐标规范恒等性，不是效能优势。RVI不是未知环境样本学习。

## 5. 三个信用子机制

greedy_adaptive_lambda.py对应core-adaptive-lambda。固定bootstrap=0、未来回报N(1,0.25)，Welford学习回报矩，取λ=b²/(b²+variance)。旧统计选择λ，独立验证回报测量混合MSE；对照固定λ=0.5，每预算都消耗一训练/一验证样本。补齐真实矩学习，但不实现整条TD链、全状态λ网络或原论文全部递归矩估计。

expected_eligibility_traces.py对应core-expected-traces。完整Markov汇合状态的过去迹为[0.72,0,1]或[0,0.72,1]，末奖励独立取0/2。条件迹由历史标签均值学习，本次方向先用旧预测，再吸收标签；对照sampled_eligibility_reference使用本条真实历史。主价值参数冻结，仅比较累计方向均值与[0.36,0.36,1]。初始预测0会有有限前缀偏差。未实现完整ET价值学习、递归ET(λ,η)、神经预测器、表示漂移或非Markov混叠。

gradient_eligibility_traces.py对应core-gradient-traces。两步0→1→真终点，末奖励Bernoulli(0.5)，V=tanh(w·x)、H=tanh(h·x)，两网络参数回合内冻结、回合间更新。zH累积辅助预测，zη累积辅助输出梯度；主方向−zH∇δ保留下一状态完整导数。前向参考独立递推λ目标和目标梯度，理论预期总方向相同，不声称在线每前缀等价。本组仅含冻结GTD2，不含TDC第三条迹、TDRC正则、QRC控制或原深度基准。

## 6. 探索、约束和零和矩阵

thompson_sampling.py对应core-thompson：三臂平稳Bernoulli、独立Beta(1,1)先验，每步从各臂均值参数的后验分布抽样，再选抽样值最大的臂，用实际奖励更新成功/失败。UCB对照同任务同真实交互预算。后验正确性限定于独立平稳Bernoulli。

cmdp_primal_dual.py对应core-primal-dual：单状态两动作奖励均值0.2/0.9，cost=动作1，上限0.4。θ初值0，ν初值0；actor随机方向(r−νc)(a−π1)，dual投影max(0,ν+α(c−0.4))，两者均读取旧参数。不用真均值更新，真最优π1=0.4只测量。无约束策略梯度对照明确不保证可行。本实现是Lagrangian子机制，不复现所引CMDP论文的regularized policy iteration、多状态占据测度求解或安全保证；有限训练也不保证从第一步可行。

minimax_2x2.py对应core-minimax：矩阵[[0.8,−0.2],[−0.4,0.6]]，每步均匀采一个元素并观测σ=0.2 Gaussian噪声，经验均值初始0。学习矩阵后，行玩家解两仿射分支下包络，列玩家解负转置；对照同经验矩阵上的虚拟博弈。真矩阵仅测量Nash间隙。这是经验矩阵子机制，不是Markov game或深度多智能体学习。

## 7. 原文身份和检查

各文件META给出原文。下列重点来源已通过原始论文、作者站点或出版社页面核验身份；代码为独立教学实现，不移植作者成绩。

- [LSTD原文PDF](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/BradtkeBarto-lstd-1996)：正则是本实验额外约定。
- [Q(σ)](https://arxiv.org/abs/1703.01327)、[Retrace](https://arxiv.org/abs/1606.02647)、[序贯DR](https://arxiv.org/abs/1511.03722)：目标、传播和比率的角色分别核对。
- [平均奖励学习](https://arxiv.org/abs/2006.16318)、[平均奖励Options](https://arxiv.org/html/2110.13855)：本组明确expected-duration固定策略范围。
- [λ-greedy](https://arxiv.org/html/1607.00446)、[Expected Traces](https://arxiv.org/html/2007.01839)、[Gradient Traces](https://arxiv.org/html/2507.09087v1)：局部矩、条件迹和冻结双网络范围不同。
- [Thompson 1933](https://www.jstor.org/stable/2332286)、[von Neumann 1928](https://link.springer.com/article/10.1007/BF01448847)：原文书目信息确认，不宣称获取全部订阅全文。
- [Primal-Dual CMDP](https://arxiv.org/abs/2101.10895)：约束原始对偶背景，不将本子机制称为原文完整算法。
- [Sutton与Barto第二版](http://incompleteideas.net/book/the-book-2nd.html)与作者OffPolicy PDF作为基础依据；本次访问有超时，不宣称逐页重新核验全书。

测试覆盖32入口×三个seed的真实1200预算、23个coverage编号、同任务baseline契约、MC访问、IS穷举期望、DR偏模型/零模型、Q(σ)端点及终点/非终点尾值、LSTD方程和奇异边界、Watkins更新后剪迹及并列动作、RVI坐标移位、旧平均时长、Welford矩、双网络前后向及有限差分、dual投影和纯/混合/退化矩阵。

数学测试验证公式和时序；一般效能仍需另行登记任务、计算预算、完整seed人口和负结果。本组不发布该类效能结论。
