# 模型记忆与规划：完成一份持续控制研究作业

<a id="study-model-memory-planning"></a>

新奖励出现后，旧模型记忆与额外规划怎样共同影响整段交互收益？

沿“技能与规划”路线，使用七状态持续环的实现与已保存记录，重算收益，比较两种解释，再设计能够区分它们的对照、测量和预算。已保存记录支持哪些观察，哪些问题还需要新的实验，是这份作业的主线。

[下载完整研究作业（Markdown）](https://yingwen.io/zh/continual-rl/download/practice-model-memory-planning.md) · [下载研究设计模板（JSON）](https://yingwen.io/zh/continual-rl/download/practice-model-memory-planning.protocol.json) · [下载只读复算工具](https://yingwen.io/crl-code/tutorials/compare_continual_runs.py) · [下载原始记录与当时源码](https://yingwen.io/crl-code/results/integrated_recent_model/raw-runs.zip) · [下载四组对照短例（Python）](https://yingwen.io/crl-code/tutorials/continual_protocol.py)

<a id="study-model-memory-planning-setting"></a>

## 先固定同一个世界与信息权限

智能体从环上位置0出发，位置为0至6。每个原子动作向左或向右，概率0.1留在原地，否则移动一格并在环上回绕。第1至600步到达位置1得到0.98，其余到达得到−0.02；第601步起奖励位置改为4。世界没有终点，也没有在变化时重置。1200只是测量预算。

学习器看到当前位置、执行的动作、下一位置和外部奖励；它知道设计者给定的两个子目标1、4及技能上限6步。奖励位置、切换时刻和阶段标签只由环境与分析者使用，不传给动作选择或更新器。模型的奖励、时长和终点分布从实际经验估计。

该任务把物理位置交给学习器，隐藏奖励阶段。它不研究从图像构建状态或自动发现子目标。三个方法都在同一给定表示上持续学习子策略和模型；这里的“记忆”特指模型如何保留历史样本。

- 已保存记录固定T=1200、切换在600之后。生成这些记录的程序用steps//2设置变化点；改总步数也会改变化时刻，不能把改--steps说成单独延长生命期。
- 每个seed的环境随机流为Random(seed+10000)，每个真实步取一次。各方法动作不同，因而即使外生随机数配对，访问位置与经验仍可不同。

[查看生成记录的程序](https://yingwen.io/crl-code/implementations/integrated_agents/_system.py)

<a id="study-model-memory-planning-loop"></a>

## 从一次真实行动读到下一次行动

高层在左右原子动作和可启动的两个技能之间用ε=0.2选择；位于子目标位置时不能启动对应技能。技能内部以ε=0.1探索，否则使用启动时已经提交的子策略，到目标或6个原子步后结束。每个真实转移同时更新两个给定子任务的Q表：到各自目标的子任务奖励为1，其他为−0.01，折扣0.95、步长0.25。这些内部信号不计入外部收益。

记一个完成宏动作的累计外部奖励为R、时长为τ、起终位置为s、s′。高层先算δ=R−ρτ+maxₐQ(s′,a)−Q(s,o)，再做Q(s,o)←Q(s,o)+0.1δ和ρ←ρ+0.002δ。ρ是学习器内部的奖励率估计；它不同于日志中的累计平均奖励。这一直接使用样本时长、未经期望时长缩放的常步长组合是教学原型；下方原论文说明了它与收敛算法的差别。

接着先用旧模型计算刚完成宏动作的奖励预测误差，再写入该(s,o)的奖励均值、时长均值和终点概率。规划从已存在的单元中随机选取，使用模型目标R̂−ρτ̂+Σₓp̂(x)maxₐQ(x,a)，以步长0.1更新Q；这里读取的是刚更新后的ρ与Q。随后提交新的子策略：若技能的贪心策略发生变化，清除该技能的模型单元及高层Q，版本加一。最后才选择下一个宏动作。

累计模型的n是当前有效单元的样本数。策略变更删除单元后，n也从头开始；近期模型首次观察同样全量写入，此后才用0.2。两者都不是保存全部旧轨迹。总预算耗尽时若技能尚未结束，已得奖励计入C(T)，但不伪造技能终止或完整模型样本。

<a id="study-model-memory-planning-comparison"></a>

## 完整学习器比较与机制对照回答不同问题

先比较三个已经实现的完整闭环：在同样1200次真实交互内，各自收到了多少外部奖励，消耗了多少更新。它们共享状态、技能、真实控制更新与版本处理，只在表中两项配置上有区别。这回答的是选择哪一个完整配置；每个配置都要自己行动，不能把同一条固定轨迹的拟合差当成行为收益。

机制解释再缩小问题。近期与累计的比较只改变模型写入规则，但这个改变会经规划影响行动、数据和后续失效次数。近期与无规划的比较移除了模拟写入，却保留模型学习。它不是“所有模型成本都取消”的无模型基线。

| 完整配置 | 模型记忆 | 每次完整宏动作后的规划 | 本组回答什么 |
| --- | --- | --- | --- |
| 近期模型＋规划 | 新单元首次样本全量写入，此后步长0.2 | 4次 | 共同参照：同一学习器持续行动、学习子策略、学习模型并规划。 |
| 累计模型＋规划 | 每个仍有效单元内按1/n取样本均值 | 4次 | 与近期模型比较记忆更新规则；两者都做规划。 |
| 近期模型＋无规划 | 同近期模型，仍估计模型并记录误差 | 0次 | 与近期模型比较移除规划后的完整行为；保留模型学习成本。 |

- 这批记录缺少累计模型＋无规划，因此只能比较上述两项差异。要估计“记忆×规划”的完整2×2交互，还需要第四组；下文用四组短例解释这种对照。
- 生成记录的程序中，Agent.rng同时供行为和规划采样使用。移除规划会改变后续行为随机数的消耗，所以相同seed不是动作噪声逐次匹配。分离行为与规划随机流，可以去掉这项直接依赖；比较时还要说明动作选择如何消耗各自的随机流。
- 在同一固定经验上重估两种模型，能检查写入与遗忘，却不包含控制反馈；随后让方法独立行动，才能比较完整收益。两项试验分别记账。
- 本组没有模型年龄输入、复查决策或调度策略的学习。路线中给定的年龄复查规则也不因连接到闭环就变成学得的调度。

[近期模型＋规划：实现与记录](https://yingwen.io/zh/continual-rl/code/integrated_recent_model/) · [累计模型＋规划：实现与记录](https://yingwen.io/zh/continual-rl/code/integrated_cumulative_model/) · [近期模型＋无规划：实现与记录](https://yingwen.io/zh/continual-rl/code/integrated_no_planning/)

<a id="study-model-memory-planning-recompute"></a>

## 从已保存记录重算读数

只读工具compare_continual_runs.py只需Python 3标准库。它读取压缩包内的配置、manifest、当时源码和CSV，核对文件摘要与计数后输出JSON；这一步不运行学习器，也不产生新训练证据。

包内还有另外两个对照，作业的主要比较只使用这里列出的三个方法及seed 0至4。先检查每个运行都有step=600与1200、源与配置核验成功，再看逐seed结果及配对汇总。不要把缺失运行自动删掉后按剩余seed排名。

在文件所在目录运行下面命令。先将下载的raw-runs.zip放到这个工作目录；也可以在--archive后填写压缩包的绝对路径。结果写入该目录下的comparison.json。

```sh
python3 compare_continual_runs.py --archive raw-runs.zip > comparison.json
```

[下载只读复算工具](https://yingwen.io/crl-code/tutorials/compare_continual_runs.py) · [下载原始记录与当时源码](https://yingwen.io/crl-code/results/integrated_recent_model/raw-runs.zip)

<a id="study-model-memory-planning-metrics"></a>

## 先规定主读数，再解释曲线

用完整生命期率J作为主要读数，近期−累计作为主要比较，近期−无规划作为辅助比较。对每个相同seed先求差dᵢ，再报告五个差的均值和样本标准差；SD的分母为n−1，不是标准误或置信区间。也保留每个方法各自的逐seed读数。一个seed内的61个检查点相关，不能当成61次独立重复。

主读数来自收到的真实奖励。预测误差、内部ρ、末窗口和模型备份解释学习过程；它们不能替代主读数。各方法共享环境步上限，并不共享总计算量；尤其规划按完成宏动作计数，动作持续时间改变时，规划总次数也会改变。

| 读数 | 从哪里计算 | 单位 |
| --- | --- | --- |
| 主要：完整1200步平均奖励 | J = C(1200) / 1200；C(t)为lifetime_reward。初始探索、变化后的损失及末尾未完成技能的已得奖励全部计入。 | reward / environment step |
| 辅助：变化后600步平均奖励 | J后 = [C(1200) − C(600)] / 600；J前 = C(600) / 600。窗口预先固定，用来定位全程差异发生在哪一段。 | reward / environment step |
| 诊断：最后100步平均奖励 | 取step=1200的value，等于[C(1200) − C(1100)] / 100。它只描述末段；不能用各检查点value的平均代替J。 | reward / environment step |
| 预算：环境、真实更新、模型备份、子任务更新 | 同样1200个真实步，分别报告real_backups、model_backups、subtask_backups。前三组都有2400次子任务写入；有规划组每个完整宏动作执行4次模型备份。 | 分项计数；不合并为等价FLOPs |
| 诊断：已覆盖宏动作上的模型奖励误差 | 完整宏动作结束时，若更新前已有该单元，记录实际累计奖励与旧预测的绝对差；model_reward_mae取最近至多100个此类误差的均值。model_error_samples=0时记录的0表示尚无样本。 | reward / completed macro observation |

<a id="study-model-memory-planning-recorded-observations"></a>

## 已有记录给出的可检查现象

下表由2026-10-04保存的五个seed重算，按同一1200步协议描述这批记录。±为跨seed样本标准差，规划数为跨seed均值。它提供作业的核算参照；这批公开数据已经可见，不是后来设计的封存确认集。

近期−累计的全程率差为0.02250，配对差SD为0.04927；seed 0和1的差为负。近期−无规划的全程差为0.07417，配对差SD为0.02462。近期模型全程均值较高，累计模型末100步均值却较高。这足以说明两个评价对象会给出不同排序，尚不足以确定未见运行中的优势。

一个单独可核算的例子是近期模型seed 0：C(600)=126、C(1200)=313，所以J=313/1200，J后=187/600。末尾pending_duration=1，真实宏动作更新869次、模型备份3476次；累计净奖励313包含尚未完成技能的贡献。

| 配置 | 全程J，均值±SD | 变化后600步，均值 | 末100步，均值 | 模型备份，均值 |
| --- | --- | --- | --- | --- |
| 近期模型＋规划 | 0.26983 ± 0.02462 | 0.24233 | 0.32800 | 3580.0 |
| 累计模型＋规划 | 0.24733 ± 0.03542 | 0.20567 | 0.36400 | 3553.6 |
| 近期模型＋无规划 | 0.19567 ± 0.02533 | 0.08667 | 0.21800 | 0.0 |

<a id="study-model-memory-planning-alternatives"></a>

## 写出会削弱自己解释的结果

一个可检验的解释是：同一奖励位置改变后，近期模型更快减少旧奖励信息的影响，规划把新预测传播到行动价值，最终改善全程收益。这个解释包含“估计→规划→动作→新经验”四个环节；只观察到末端均值差，不能知道哪一环起了作用。

反证可以很具体。若新封存运行的全程差不超过预先填写的最小有意义差，或跨运行方向不稳定，就缩小优势主张；若相同数据上的模型追踪更快，但独立行动后的收益没有提高，就不能把估计改进称为控制收益；若优势只在额外计算下出现，就只保留交互效率主张。

| 替代解释 | 旧记录能看什么 | 需要怎样区分 |
| --- | --- | --- |
| 优势来自更多规划计算 | 可见real_backups与model_backups；规划组总次数本就不同 | 预先规定同环境预算比较与另一个计算预算比较，分别报曲线；不要事后按结果选择计费方式。 |
| 访问分布或技能版本变化造成差异 | 可见失效次数和误差滑窗，缺逐步动作与版本轨迹 | 固定经验诊断估计器，再用独立闭环及逐步日志检验动作和数据的变化。 |
| 模型MAE降低只是更多访问容易样本 | MAE只覆盖更新前已有模型的完整宏动作，且混合不同τ | 预先规定共同诊断输入与时长分层；诊断使用副本和独立RNG，不污染主生命期。 |
| 该配置适合公开的半程切换 | 全部旧运行都是1200/600，设计者已知此规律 | 新seed仅检查这个已知协议；未知变化条件需另设不可用于调参的封存环境，并记录新环境实现。 |

<a id="study-model-memory-planning-freeze"></a>

## 参数选择与独立测试

现有公开记录用于理解、查错和设计；任何看过其全程结果后作出的模型选择都属于开发历史。把旧seed改名为test，或事后截取一个未画出的窗口，不会消除这段历史。下载模板将开发记录与未来测试保持为不同对象，结果栏留空。

确认性研究需要预先写清主问题、比较方向、最小有意义差、总交互预算、计算与内存约束、候选配置数和选择规则。各方法使用相同开发权限，保留所有候选和失败。若采用受限前缀调参，填写允许访问的前缀长度与开发种子；选择配置后，在新运行上从初始化开始评完整生命期，学习器在测试中继续按已经冻结的算法学习。

封存的是实现、配置选择和分析规则，不是让测试中的学习器停止更新。记录源/协议/配置的哈希、保管测试种子与环境规格的角色、何时允许打开测试结果，并冻结主要指标与停止规则。若目标是未知变化，测试变化表不能交给调参者。将变化表单独交给环境，可以明确学习器的信息权限；正式封存还取决于谁能查看和修改测试材料。

预先列出计划运行的完整集合。数值错误、缺失日志和提前退出逐项保留，按已写明规则报告或判为不完整；看到结果后换seed、延长有利运行、删去失败或重新调参，都要另立开发轮次。只允许因预定预算耗尽或记录的技术故障停止，不按当前排名早停。

已保存的五个seed支持描述性复算。确认性研究需在看测试前填写运行数量、希望辨别的差异与不确定性方法；这里的SD描述样本间差异，不是显著性检验。若预算不足以分辨目标差异，应报告精度不足并保留原始结果。

[下载研究设计模板（JSON）](https://yingwen.io/zh/continual-rl/download/practice-model-memory-planning.protocol.json)

<a id="study-model-memory-planning-logs"></a>

## 用逐步记录解释学习闭环

已保存的CSV和events.jsonl在第1步、每20步及末步记录同类检查点。累计奖励足以重算这里固定的两段率；要重建逐步轨迹、精确首次恢复时间或每次模型写入，还需要更细的记录。下表据此提出日志规范，并列出已有记录能提供哪些信息。

若加入冻结诊断，先复制主学习器全部可变状态，包括Q、子任务Q、已提交策略、版本、模型单元和计数、ρ、正在执行的宏动作以及随机状态；诊断在独立环境和RNG上运行，结束后主流保持原样。诊断成本单列，主指标继续覆盖真实部署流中全部奖励。

| 记录时机 | 拟议字段 | 为何需要 | 已有与缺失 |
| --- | --- | --- | --- |
| 真实交互：逐原子步 | run_id, seed, t, state_before, primitive_action, successor, external_reward, cumulative_reward, active_option, option_version, macro_id | 从收到的原始奖励独立重算J，并判断模型变化是否先改变行动、再改变经验。 | 旧CSV只有稀疏检查点的累计奖励；没有逐步状态、动作与奖励。 |
| 宏动作边界：每次完成或预算截断 | macro_id, start_t, end_t, start_state, option_id, executed_policy_version, reward_sum, duration, successor, stop_reason, censored | 区分到目标、达到6步上限与整个运行的截断。未完成宏动作不得写入完整模型；已经收到的外部奖励仍计入J。 | 旧记录有累计完成数、上限停止数和pending_duration，不能恢复每次边界。 |
| 模型与控制：每次真实/模拟写入 | update_id, model_cell, model_count_before, model_version, prediction_before, observed_target, model_after, q_before, q_after, rate_before, rate_after, backup_source, policy_commit, invalidated_cells | 检查旧模型被谁读取、误差用哪个参数版本，以及规划是否跨过动作排序阈值；不能从一次残差下降推出后续收益。 | 旧记录只保存误差滑窗、模型单元数、失效次数与备份总数。 |
| 资源：逐步计数＋运行汇总 | environment_steps, real_backups, model_backups, subtask_backups, model_writes, model_cells, peak_memory_bytes, elapsed_seconds, process_cpu_seconds, hardware, dependency_versions | 把交互节省与计算/存储代价分别报告。时间和峰值内存随平台变化，须保存测量方法；单元数不是字节数。 | 旧日志可重算前三种备份与单元数；没有足以比较的墙钟、CPU或峰值内存记录。 |
| 随机性与来源：运行前与检查点 | source_hashes, protocol_hash, config_hash, environment_rng_id, behavior_rng_id, planning_rng_id, rng_states, learner_checkpoint_hash, diagnostic_rng_id | 让后续运行可检查环境配对、行为与规划随机流、学习器恢复及诊断隔离。主流和诊断流各自记账。 | 旧包包含源码哈希、配置和种子；行为与规划共用随机发生器，没有逐检查点完整状态。 |

<a id="study-model-memory-planning-protocol-v2"></a>

## 模型记忆与规划的四组对照

模型记忆的作用是否依赖规划？在同一个七状态环、两个给定子任务和宏动作更新循环中，把近期或累计模型分别配上4次或0次规划，就得到完整的2×2对照。两种记忆规则在有规划时的收益差，再减去它们在无规划时的收益差，构成交互差[近期＋规划−累计＋规划]−[近期＋无规划−累计＋无规划]。比较收益时，仍需分别报告各项计算预算。

四组共用子任务Q学习、完整宏动作的真实Q与ρ更新、模型误差计算、模型写入和策略版本失效规则。无规划两组也继续学模型。在这里，模型只通过规划影响控制，因此给定相同的环境与行为随机序列，两组无规划学习器应当产生相同行动和收益；模型估计却可以不同。这个可预期的结果帮助读者判断，模型记忆是否通过规定的路径影响行动。

为单独改变测量时长，奖励变化表使用start_step与rewarding_position：start_step表示新奖励位置从该原子步起生效，表从1开始严格递增，独立于environment_steps。把预算从1200延长到1800，就不会把第601步的变化移到第901步。环境单独读入变化表，Agent不接收环境对象、变化表或阶段标签。

为单独改变规划次数，环境、行为和规划各使用自己的随机流。环境每原子步取一次；行为按动作选择调用取样；规划只消耗规划流。关掉规划便不会直接推进行为随机流。不过，动作与技能时长仍会影响以后调用行为流的时刻，所以行为随机数按抽样调用配对，不保证按原子步配对。

continual_protocol.py提供这四组的固定序列短例。默认validate模式只检查参数与运行规则，不创建环境；fixture模式使用文件中给定的uniform序列，让每组各执行16个原子步，保留学习后选择下一行动的闭环。这些读数用于核对更新过程；四组尚无性能研究数据，脚本也没有随机性能试验的运行入口。前面的三组15次运行来自另一程序，随机选取方式不同，应继续独立解释，不能与短例合并。

读短例时，先从atomic_step找到收到的奖励，再沿事件编号查看宏动作如何结束、真实更新和模型写入如何发生、规划读取了什么，以及子策略改变后哪些模型单元失效。最后检查下一次行动和累计奖励。预算耗尽时若技能尚未结束，已得奖励仍应计入，但不应多出一个完整模型样本。下表给出各类事件与这些问题的对应关系。

在文件所在目录运行 python3 continual_protocol.py，只需 Python 3 标准库。下面的命令依次检查默认参数、生成两个配置文件、联合检查文件内容，再记录固定序列短例。contract.json规定学习与运行规则，environment.json给出公开的奖励变化表示例；研究问题、样本量与测试封存安排则填写在上面的研究设计JSON中。最后一条命令将完整事件日志写入当前目录下的fixture-events.jsonl。

| 事件 | 从记录中观察什么 |
| --- | --- |
| atomic_step / macro_start / macro_boundary | 逐原子步状态、动作、奖励、累计奖励和执行版本；宏动作启动、到目标、6步上限和预算截断。变化时保留正在执行的宏动作。 |
| subtask_update / real_update / model_write / planning_update | 每次实际写入前后值、目标、模型样本数/版本、规划读取的模型与后继价值、更新前后ρ。事件按执行次序编号。 |
| policy_commit / resource_counts / run_end | 记录新旧子策略、清除的模型单元与Q列；逐步报告各类更新计数。末尾未完成技能的奖励进入累计奖励，不写完整模型。 |
| run_start | 源码、短例实际采用的运行规则、配置、环境变化表和固定序列的哈希；三个随机流身份、抽样计数及状态摘要。 |

- 末尾参数快照与RNG状态摘要用于查阅，不能恢复运行；日志中的checkpoint_restorable为false。

```sh
python3 continual_protocol.py
python3 continual_protocol.py --mode template > contract.json
python3 continual_protocol.py --mode environment-example > environment.json
python3 continual_protocol.py --protocol contract.json --environment environment.json
python3 continual_protocol.py --mode fixture --events > fixture-events.jsonl
```

[下载四组对照短例（Python）](https://yingwen.io/crl-code/tutorials/continual_protocol.py) · [下载研究设计模板（JSON）](https://yingwen.io/zh/continual-rl/download/practice-model-memory-planning.protocol.json)

<a id="study-model-memory-planning-submit"></a>

## 交付一份可以复查的研究作业

把comparison.json、读数表、解释文字和填写后的研究协议放在一起。先用一段话回答“这批运行显示什么”，再提出一个现有日志还不能决定的问题，说明怎样的对照能够区分候选解释。复算让读者检查已有观察，研究设计则说明需要哪些新证据才能判断所提出的解释。

- 一张逐seed读数表及两项配对差：近期−累计、近期−无规划；同时保留全程率、前后两段率、末窗口和预算。
- 一段解释末窗口与全程排序为何不同的文字；明确列出两项旧数据无法判断的替代解释。
- 一份填写后的协议：问题、主要比较、最小有意义差、开发权限、封存测试范围、资源限制、失败处理和停止规则。保留尚无证据的字段为空。
- 一份新增日志规范和两项反证条件，使后来的人能区分模型更准、动作改变与生命期收益改善。

## 依据与进一步阅读

[Sutton & Barto，Reinforcement Learning: An Introduction，第2版，§§8.2–8.3](http://incompleteideas.net/book/the-book-2nd.html)。真实经验既能直接更新价值，也能改进模型后用于规划；错误模型与探索会相互影响。Dyna-Q+的年龄奖金是给定启发式，并非从反馈中学得复查调度。本作业使用自己的随机宏动作模型，不是原书Dyna-Q复现。

[Mesbahi et al.，Position: Lifetime tuning is incompatible with continual reinforcement learning，2025，§§3、5](https://arxiv.org/html/2404.02113v4)。作者指出反复查看完整生命期来设计和调参会泄漏未来规律，并提出限制调参前缀的评价办法。本作业借此限制设计者的权限；下述封存协议是针对这里的研究设计，不是复现论文实验。

[Abel et al.，A Definition of Continual Reinforcement Learning，NeurIPS 2023，§§4.1–4.3](https://papers.neurips.cc/paper_files/paper/2023/file/9d8cf1247786d6dfeefeeb53b8b5f6d7-Paper-Conference.pdf)。论文相对于给定agent basis定义持续学习，并要求最优智能体持续搜索。这里有限1200步、只变化一次的记录只能研究一段适应过程；无回合终止本身不足以验证这个无限时域定义。

[Wan, Naik & Sutton，Average-Reward Learning and Planning with Options，NeurIPS 2021，§§2–3，式(2–9)](https://papers.neurips.cc/paper_files/paper/2021/file/c058f544c737782deacefa532d9add4c-Paper.pdf)。SMDP目标按时长扣除奖励率。本文高层真实更新采用式(3–5)所讨论的直接扩展：实际时长乘ρ，更新不除以期望时长。论文指出长时长会放大奖励率误差；其正式算法式(6–9)改用期望时长估计并缩放更新，收敛结论还要求固定问题、合适递减步长与覆盖条件。这里持续改变子策略、使用常步长的集成原型不继承该结论。
