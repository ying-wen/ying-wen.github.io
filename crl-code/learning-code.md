# 从一个算法文件到可追查的实验

本目录面向学习者。先读一个真实更新，再跟踪它如何进入完整经验流。所有结果来自运行；没有预置“应当上升”的曲线。

## 文件怎么分

```text
implementations/
  classic/        表格方法与线性预测：每个算法一个文件
  deep/           神经控制与离线学习：每个算法一个文件
  continual/      持续预测、元步长、保留、状态、平均奖励与规划
  extended_classic/  重要性采样、多步信用、平均奖励与约束
  extended_adaptation/  状态推断、时间梯度、元学习与可塑性
  extended_knowledge/   探索、目标、选项模型与奖励学习
  multiagent/     反事实信用与单调价值分解
  average_systems/  多状态平均奖励预测、控制与规划
  nonlinear_diagnostics/  自举、共享特征与目标依赖的反例
  streaming_composition/  神经GVF、资格迹和技能时长
  integrated_agents/  子任务、技能、预测模型与持续闭环
  learner_control/  同一学习快照下的在线适应与冻结参数对照
  runtime.py      参数、日志、失败记录、CSV 与 SVG；不包含算法更新
```

多数基础组的 `_common.py` 共享环境、网络构造、采样和指标。`learner_control/_common.py` 还明确共享两组对照的交互与记忆更新循环；差分 Q 更新在 `online_differential_q.py` 中，冻结入口只改变参数更新权限。阅读时应追踪实际调用，而不能仅凭文件名判断算法边界。例如固定温度 SAC 不等于自动调温版本；单工作器同步 A2C 不等于分布式 A3C；已知奖励权重下的 GPI 不等于自动发现任务。

集成智能体另在 `integrated_agents/_system.py` 明确展示共享闭环。五个命名入口是同一系统的不同模块配置，不是五种新的研究算法。网页在每个入口下提供可展开的核心源码。

独立子机制实验也能完整运行，但只回答限定问题。透明消融入口会显式调用主方法并关闭一个机制，不计作新的完整算法。覆盖层次见[实现清单](implementation-coverage.md)。

## 从问题找到公式与实现

- [估计、信用分配与平均奖励](extended-classic.md)：IS/DR 为什么使用不同权重；多步误差怎样传播；无终点奖励率怎样进入更新。
- [状态、梯度与持续适应](extended-adaptation.md)：过滤、时间梯度、元梯度、参数回收分别改变什么；哪些实验是预测，哪些是控制。
- [探索、目标、选项与奖励](extended-knowledge.md)：怎样构造预测误差奖励、合法重标目标、选项模型与可学习奖励。
- [多智能体信用与价值分解](extended-multiagent.md)：反事实基线消去了什么；单调混合为什么允许分散贪心。
- [平均奖励预测、控制与规划](average-systems.md)：奖励率与 bias 分开；用真实多状态流和解析真值检查。
- [非线性学习困难](nonlinear-learning-depth.md)：自举反馈、double sampling、共享梯度与目标网络。
- [流式 GVF、资格迹与技能](streaming-composition.md)：问题规格、历史梯度、两种时钟与真实终止。
- [集成智能体](integrated-agents.md)：让学到的技能和后果预测真正进入控制与规划。
- [比较持续学习器](learner-control.md)：从同一历史和参数快照继续行动，区分冻结参数、保留记忆与外部重置。

每组给出任务、公式、更新顺序、基线和未覆盖部分。代码包包含对应测试：

```bash
python3 -m unittest discover -s tests -p 'test_extended_*.py' -v
python3 -m unittest discover -s tests -p 'test_average_systems.py' -v
python3 -m unittest discover -s tests -p 'test_nonlinear_diagnostics.py' -v
python3 -m unittest discover -s tests -p 'test_streaming_composition.py' -v
python3 -m unittest discover -s tests -p 'test_integrated_agents.py' -v
python3 -m unittest discover -s tests -p 'test_reading_derivations.py' -v
python3 -m unittest discover -s tests -p 'test_learner_control.py' -v
```

## 只会基本深度 RL 时，先做这四组实验

不必逐个运行全部目录。按下面顺序，每次先解释预期差异，再运行。

1. `differential_td_offpolicy`：价值误差不大，是否就说明预测正确？检查奖励率与共同偏移。
2. `double_sample_residual`：优化一个看似合理的 TD 平方误差，为何会得到错误固定点？先手算 1 和 2 两个解。
3. `gvf_shared_trace`：多个预测共享神经表示后，辅助目标改变怎样影响其他预测？固定骨干与问题规格读对照。
4. `integrated_recent_model`：技能学习、模型学习和规划如何共同作用？同时看近期收益、全程收益与模型更新成本。

```bash
python3 implementations/average_systems/differential_td_offpolicy.py --steps 1200 --seeds 0 1 2 3 4 --out results/first-average
python3 implementations/nonlinear_diagnostics/double_sample_residual.py --steps 1200 --seeds 0 1 2 3 4 --out results/first-target
python3 implementations/streaming_composition/gvf_shared_trace.py --steps 1200 --seeds 0 1 2 3 4 --out results/first-gvf
python3 implementations/integrated_agents/integrated_recent_model.py --steps 1200 --seeds 0 1 2 3 4 --out results/first-agent
```

这四组都只依赖标准库。最后一组不是完整 OaK。它的状态、目标和终止规则由设计者给定；技能内部策略、后果模型和控制价值从交互中学习。

## 阅读顺序

1. 读 `META`：任务、可见信息、初始化、指标和预算时钟。
2. 读 `update`、target 函数或明确标记的更新段。确定每个量取更新前还是更新后的值。
3. 读 `run`：它在哪里采样、构造目标、写入参数、处理终止、评价和记录？
4. 看对应测试。手算、有限差分和退化等价回答实现问题；训练曲线回答该任务中的学习问题。
5. 再运行默认 baseline。不要先根据结果调整多个机制。

## 一条命令，保存完整结果

所有命令从仓库根运行。Python 3.10+；标准库方法不要求 GPU。深度方法另装 `examples/deep_requirements.txt` 中的 PyTorch。

```bash
python3 implementations/runtime.py --list
python3 implementations/classic/td_lambda.py --steps 1200 --seeds 0 1 2 3 4 --out results/td-lambda-first
python3 implementations/deep/ppo.py --steps 1200 --seeds 0 1 2 3 4 --out results/ppo-first
```

`--baseline none` 仅运行当前方法；`--baseline ALGORITHM_ID` 更换对照。运行器拒绝不同任务、指标、方向或预算时钟的直接比较。默认超参数都在算法文件中，修改代码后应使用新的输出目录和新的实验记录。

终端显示 `[prepare]`，然后显示当前算法、seed、阶段、当前时钟/预算与指标。phase 是测量时的任务或学习阶段，具体含义见源码。更新次数、TD error、loss、目标网络更新或模型备份等诊断随算法记录。它们不全是优化目标，也不保证单调下降。

输出结构：

```text
results/td-lambda-first/
  index.html                  可离线打开的对照图
  learning-curves.svg          矢量图
  curves.json                 均值、样本标准差、样本数
  manifest.json               配置、源码摘要、完整实验人口与失败
  td_lambda/seed-0/
    events.jsonl              原始阶段记录
    metrics.csv               逐记录点指标与诊断
  td0/seed-0/                 baseline，其他 seeds 同结构
```

预算时钟不总是环境步数。DP 使用模型扫描，三个离线 RL 例子使用固定512条转移上的训练批次（training_batches），递归预测例子可能使用序列。离线 Q-learning 与 CQL 每批各执行一次 Q 优化器更新，IQL 每批对 V、Q、actor 各执行一次，共三次；日志同时保留 training_batches 与 optimizer_steps。相同批次数不代表相同优化器调用数、梯度计算或计算成本。不要把它们放在“同样1000步”的全局排行榜。模型规划和评价还会消耗额外计算，需另查诊断和代码。

## 怎样读图

图显示全部指定训练种子的均值与 ±1 个样本标准差。五个种子是教学展示配置，不是统计充分性保证。没有置信区间或显著性检验。任务、参数和种子均公开；这些任务很小，有些算法很快都达到相同回报，平线也应保留。

先看完整曲线及每个种子。再问：差异来自目标估计、策略选择、学习率、模型备份还是保留旧样本？要做因果解释，应固定其他因素设计消融。对角线、缓慢学习和负结果不能删除。运行失败保存失败回执，不在成功种子里重新挑最好结果。

## 接入 Workbench

教程负责算法实现；Workbench 负责研究协议与证据管理。使用Python 3.10+，在 Workbench 仓库根运行，替换 `TUTORIAL_CHECKOUT` 为教程 checkout 的绝对路径。依赖环境应包含所选算法的依赖，并在plan/freeze/run/import阶段使用同一解释器。

有Workbench checkout时直接进入该目录；源码根的 `python3 -m ...` 不要求安装包。没有基本仓库时可先获取：

```bash
git clone https://github.com/ying-wen/rl-research-workbench.git
cd rl-research-workbench
```

**版本前提：**使用包含 `rlworkbench/tutorial_adapter.py` 的 Workbench 版本。已有较早 checkout 的读者应先更新；运行 `python3 -m rlworkbench.tutorial_adapter --help` 确认接入命令可用。

若需要安装Workbench，推荐在该checkout建立隔离环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```

bandit示例无需第三方依赖；深度方法先在同一环境安装教程 `examples/deep_requirements.txt`，再建立协议和冻结版本。修改Python/包版本或算法源码后用新协议、新锁与新输出目录。完整接入说明见Workbench `docs/tutorial-adapter.md`。

```bash
python3 -m rlworkbench.tutorial_adapter plan \
  --tutorial-root TUTORIAL_CHECKOUT \
  --ids bandit_constant_step bandit_sample_average --baseline bandit_sample_average \
  --study-id bandit-teaching-v1 --seeds 11 23 --holdout-seeds 101 211 \
  --steps 1000 --failure-score -10 --out studies/bandit-teaching-v1.json
python3 -m rlworkbench freeze studies/bandit-teaching-v1.json --external --out studies/bandit-teaching-v1.lock.json
python3 -m rlworkbench.tutorial_adapter run studies/bandit-teaching-v1.lock.json \
  --tutorial-root TUTORIAL_CHECKOUT --out incoming/bandit-teaching-v1
python3 -m rlworkbench import-results studies/bandit-teaching-v1.lock.json \
  --input incoming/bandit-teaching-v1 --out runs/bandit-teaching-v1
python3 -m rlworkbench audit runs/bandit-teaching-v1
python3 -m rlworkbench report runs/bandit-teaching-v1
python3 -m rlworkbench compare runs/bandit-teaching-v1 --candidate bandit_constant_step --baseline bandit_sample_average
```

适配器冻结源码摘要并拒绝漂移；保存完整算法×种子人口。`failure-score` 是预先规定的复合分析分值，不是算法真实回报。对 RMSE 等越小越好的指标，应设定高而非低的失败分值。外层不重复评价；使用算法自己记录的指标，因此先读指标定义。

此流程不自动启动 HPO、分布式训练或检查点恢复。大型作者工程保留原训练器，参见[作者工程接入](author-projects.md)和 Workbench 的 `docs/external-adapters.md`。

## 明确近期与生命期指标

Workbench 适配器 v2 可用 `--metric-spec` 固定字段、聚合、时钟、窗口、单位和字段含义。运行 `python3 -m rlworkbench.tutorial_adapter plan --help` 确認当前 checkout 包含此选项。不指定时仍取最后一条 `value`；它的统计对象由算法文件定义。

集成智能体中，`value` 是最近 `min(t,100)` 个环境步的奖励均值；`average_reward` 是从第 1 步到 t 的实际奖励总和除以 t。前者回答近期表现，后者包含全部学习成本。要比较生命期收益率，取最后一条 `average_reward`，不要再次平均各检查点的累计均值。

例如四步奖励 A=`[4,4,0,0]`、B=`[0,0,1,1]`。最终两步均值为 A=0、B=1；生命期均值为 A=2、B=0.5。这是固定日志算例，用于检查指标含义，不是算法训练结果。

在 Workbench 根目录，先只生成和校验协议：

```bash
python3 -m rlworkbench.tutorial_adapter plan \
  --tutorial-root TUTORIAL_CHECKOUT \
  --ids integrated_recent_model integrated_no_planning \
  --baseline integrated_no_planning --track continual \
  --metric-spec examples/tutorial-metric-lifetime.json \
  --study-id integrated-lifetime-smoke-v2 \
  --seeds 11 23 --holdout-seeds 101 211 --steps 1200 \
  --failure-score -10 --out studies/integrated-lifetime-smoke-v2.json
python3 -m rlworkbench validate studies/integrated-lifetime-smoke-v2.json --external
```

`examples/tutorial-metric-recent.json` 则选择最终近期表现。不同问题使用新协议；旧锁和旧结果不改写。两种智能体的模型备份成本不同，固定环境步不等于固定计算预算。完整字段语义、无训练的检查例和后续执行步骤见 Workbench 的 `docs/tutorial-adapter.md`。

记录窗口只筛选日志，不改变字段内部已有的滚动窗口。`mean` 聚合要求窗口内逐步完整的记录；稀疏检查点不能直接当作时间平均。

## 维护网页结果

源码确认后运行一次固定小预算；同一算法在每个seed只训练一次，其原始记录用于不同baseline图。生成的是可重建的快照，不手工编辑曲线。

```bash
python3 scripts/build_learning_gallery.py --out results/gallery-v1 --steps 1200 --seeds 0 1 2 3 4
python3 scripts/build_learning_gallery.py --out results/gallery-v1 --export-only --site WEBSITE_CHECKOUT
```

导出核对源码摘要、数据完整性和可比条件。修改代码后需要新运行。原记录不覆盖。网站只复制明确列出的教学源码与结果，不复制作者 checkout、凭据或本机环境。
