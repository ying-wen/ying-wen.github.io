[统一学习地图](https://yingwen.io/zh/continual-rl/download/curriculum.md) · 代码按预测知识、时间抽象与平均奖励等学习对象组织。

# Agent state、Meta-learning、GVF、Options、Average Reward：作者代码与公式对照

先读一个核心实现，再扩展；公开源码不自动等于有开放许可证。所有 commit 链接指向固定版本。本站教学代码 MIT，原创解释 CC BY 4.0；上游代码各按其许可证。

## Agent state：递归表示与跨时间信用

先在有延迟线索的最小例子中区分活动状态与敏感度，再读 RTU 的结构化递归；GVFN 仍可在 GVF 分支找到。

### RTU · 递归状态与前向敏感度

当前误差如何更新产生旧记忆的参数？

先读 FwdRealTimeLinearRTUs 的 carry=(hidden state, grad_memory)，再读 RealTimeLinearRTUs 的 custom_vjp。模型状态与梯度记忆是两种对象。JAX/Flax 原实验；本站标量 RNN 不是 RTU 的替代复现。

- [仓库](https://github.com/esraaelelimy/rtus) · 论文作者 / 研究团队实现 · MIT
- [归属依据](https://arxiv.org/abs/2409.01449)
- commit: be54e13b91edcd7988dd1764f8f2d412ca2db856
- [核心算法源码](https://github.com/esraaelelimy/rtus/blob/be54e13b91edcd7988dd1764f8f2d412ca2db856/src/nets/rtus/linear_rtus.py)
- [运行与论文说明](https://github.com/esraaelelimy/rtus/blob/be54e13b91edcd7988dd1764f8f2d412ca2db856/README.md)

第一次做：先运行 agent state 章的标量 RTRL，并对照敏感度的张量维度。

运行条件：JAX / Flax / Brax / POPGym，使用独立环境和作者配置。

检查范围：确认作者归属、固定 commit 与核心源码接口；未运行完整原实验训练。本站机制单元测试独立列示。

思考：不改变参数时的精确敏感度，与每步参数漂移后的在线敏感度有什么区别？

## Meta-learning：初始化、上下文、学习规则

MAML、PEARL 与 DiscoRL 改变的是不同对象。分别追适应前后的参数、上下文后验和元学习器状态，不把所有快速适应都称作在线权重学习。

### MAML-RL · 穿过策略更新求导

适应后的表现如何改变适应前的参数？

沿 init_opt → updated_dist_info_sym 追内层梯度怎样产生新参数，再回到 batch_maml_polopt 看适应前后如何收集轨迹。历史 TensorFlow/rllab 依赖；cbfinn/maml 主仓库主要是监督实验，不能代替这个 RL 入口。

- [仓库](https://github.com/cbfinn/maml_rl) · 论文作者 / 研究团队实现 · MIT 文本；rllab 贡献者共享版权
- [归属依据](https://proceedings.mlr.press/v70/finn17a.html)
- commit: 9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95
- [核心算法源码](https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/algos/maml_vpg.py)
- [运行与论文说明](https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/README.md)

第一次做：先运行 meta-learning 章的标量双层导数，再追原工程的训练与评价数据流。

运行条件：历史 TensorFlow / rllab / MuJoCo；不要在现有 Python 环境覆盖安装。

检查范围：确认作者归属、固定 commit 与核心源码接口；未运行完整原实验训练。本站机制单元测试独立列示。

思考：部署期间到底哪些状态持续变化？测试任务的经验、重置和元训练预算是否公平？

### PEARL · 用经验推断任务上下文

新任务适应时改变的是参数，还是上下文后验？

先读 infer_posterior 中的 Gaussian product 与 sample_z，再读 sac.py 中的 KL、critic 和 policy loss。原实现依赖旧 PyTorch/MuJoCo 环境；上下文推断不等于部署时重训整个网络。

- [仓库](https://github.com/katerakelly/oyster) · 论文作者 / 研究团队实现 · MIT
- [归属依据](https://proceedings.mlr.press/v97/rakelly19a.html)
- commit: 44e20fddf181d8ca3852bdf9b6927d6b8c6f48fc
- [核心算法源码](https://github.com/katerakelly/oyster/blob/44e20fddf181d8ca3852bdf9b6927d6b8c6f48fc/rlkit/torch/sac/agent.py)
- [运行与论文说明](https://github.com/katerakelly/oyster/blob/44e20fddf181d8ca3852bdf9b6927d6b8c6f48fc/README.md)

第一次做：先运行 meta-learning 章的标量双层导数，再追原工程的训练与评价数据流。

运行条件：历史 PyTorch / MuJoCo，按 README 恢复独立依赖。

检查范围：确认作者归属、固定 commit 与核心源码接口；未运行完整原实验训练。本站机制单元测试独立列示。

思考：部署期间到底哪些状态持续变化？测试任务的经验、重置和元训练预算是否公平？

### DiscoRL · 学习更新规则本身

如何在多任务、多步学习后评价一套更新规则？

先看 update rule 的输入、输出和 learned target，再按 colabs/eval.ipynb 评估发布规则；meta_train.ipynb 是元训练入口。发布的 Disco103 权重与从头重跑论文级搜索是不同成本，不能混写为一条复现命令。

- [仓库](https://github.com/google-deepmind/disco_rl) · 论文作者 / 研究团队实现 · Apache-2.0
- [归属依据](https://doi.org/10.1038/s41586-025-09761-x)
- commit: 9059a29f7121d60948f25ef165e08e050e9399c8
- [核心算法源码](https://github.com/google-deepmind/disco_rl/blob/9059a29f7121d60948f25ef165e08e050e9399c8/disco_rl/update_rules/disco.py)
- [运行与论文说明](https://github.com/google-deepmind/disco_rl/blob/9059a29f7121d60948f25ef165e08e050e9399c8/README.md)

第一次做：先运行 meta-learning 章的标量双层导数，再追原工程的训练与评价数据流。

运行条件：JAX / Haiku / RLax；先按评估 notebook 运行已发布规则，再考虑元训练成本。

检查范围：确认作者归属、固定 commit 与核心源码接口；未运行完整原实验训练。本站机制单元测试独立列示。

思考：部署期间到底哪些状态持续变化？测试任务的经验、重置和元训练预算是否公平？

## GVF / Horde：先定义预测问题，再选学习器

TD → 多信号、多时间尺度预测 → off-policy Horde → 用预测构造递归状态。GVF 是问题的规格，GTD 是更新方法；一个仓库可能只提供其中一层。

### RLPark · Horde / GTD(λ) / nexting

怎样让同一条机器人经验同时训练许多预测器？

Horde 与 multi-timescale nexting 的项目页将 RLPark 列为实验软件。沿调度器、prediction demon、GTD(λ) 三层读，可以分清经验共享和每个预测器自己的学习状态。这里固定的是仓库可访问版本，不等同于恢复了 2011 年实验机器的全部环境。

- [仓库](https://github.com/rlpark/rlpark) · 原实验使用的软件体系 · EPL-1.0
- [归属依据](https://rlpark.github.io/publications.html)
- commit: 2baf7389c75a31831e5b1a68539771c1947b7f09
- [共享 transition 的组织](https://github.com/rlpark/rlpark/blob/2baf7389c75a31831e5b1a68539771c1947b7f09/rlpark.plugin.rltoys/jvsrc/rlpark/plugin/rltoys/horde/Horde.java)
- [连接 cumulant、策略与学习器](https://github.com/rlpark/rlpark/blob/2baf7389c75a31831e5b1a68539771c1947b7f09/rlpark.plugin.rltoys/jvsrc/rlpark/plugin/rltoys/horde/demons/PredictionOffPolicyDemon.java)
- [误差、迹和辅助权重](https://github.com/rlpark/rlpark/blob/2baf7389c75a31831e5b1a68539771c1947b7f09/rlpark.plugin.rltoys/jvsrc/rlpark/plugin/rltoys/algorithms/predictions/td/GTDLambda.java#L50)

第一次做：先给一个 demon 写出 c、γ、π；再对照下方 F1–F2，追一次 update。官方文档另有 Critterbot 的 Jython Horde 示例。

运行条件：Java / Eclipse 插件生态，依赖 Zephyr；机器人驱动在独立仓库。先读代码与自带 jvsrctests，不建议新手直接恢复整套机器人栈。

检查范围：静态核对 GTDLambda.update：包含终止信号 z；迹用 γt，bootstrap 用 γt+1；主权重与辅助权重均使用更新前的辅助向量。本地另有数值接口测试，未运行 Java 工程。

思考：两条预测的目标策略不同，但传感器信号相同：哪些量可以共享，哪些迹与参数必须分开？

### GVFN · General Value Function Networks

预测能否不仅是输出，还成为 agent state？

作者仓库承载 JAIR 2021 GVFN 的主要实验。相较于许多独立 GVF，它把递归状态分量约束成预测，并研究梯度截断的影响。先看问题对象，再看网络和更新，避免把 recurrent hidden state 与学习参数混成同一对象。

- [仓库](https://github.com/mkschleg/GVFN) · 论文作者实现 · 未见仓库级 LICENSE
- [归属依据](https://github.com/mkschleg/GVFN)
- commit: d8280b8256bdd7533b6b271d863f0a0f54d9f254
- [cumulant / discount / policy 的定义](https://github.com/mkschleg/GVFN/blob/d8280b8256bdd7533b6b271d863f0a0f54d9f254/src/GVF.jl)
- [递归状态](https://github.com/mkschleg/GVFN/blob/d8280b8256bdd7533b6b271d863f0a0f54d9f254/src/GVFRNN.jl)
- [递归梯度学习](https://github.com/mkschleg/GVFN/blob/d8280b8256bdd7533b6b271d863f0a0f54d9f254/src/RGTD.jl)
- [历史依赖边界](https://github.com/mkschleg/GVFN/blob/d8280b8256bdd7533b6b271d863f0a0f54d9f254/Project.toml)

第一次做：先选 RingWorld 或 CompassWorld；手工写出一组预测目标，比较不同截断长度，而非先运行 Critterbot 数据实验。

运行条件：Julia，Project.toml 指定 Julia 1.0 兼容与 Flux 0.9.0；现代 Flux 不能直接当作兼容替代。数据与计算需求依实验而异。

检查范围：核对问题定义、文件入口和依赖；未运行 Julia 训练。不能用本站两步表格 GVF 的成功代表 GVFN 复现成功。

思考：如果所有预测都很准，但仍区分不了需要不同动作的历史，应该增加哪些问题？

### GVFHordes.jl · 可组合的 GVF 问题库

如何把“问什么”从“怎样更新”中拆出来？

按 cumulant、discount、policy 分文件，适合学习预测问题的程序接口。它是作者生态中的工具，不应标成 Horde 2011 的完整原始实验包，也不应把包名当成学习稳定性保证。

- [仓库](https://github.com/mkschleg/GVFHordes.jl) · 研究作者的工具库 · MIT
- [归属依据](https://github.com/mkschleg/GVFHordes.jl)
- commit: 8dbaf54bc55d510193ae7e58d9828badffcacc75
- [信号规格](https://github.com/mkschleg/GVFHordes.jl/blob/8dbaf54bc55d510193ae7e58d9828badffcacc75/src/gvf/cumulant.jl)
- [延续与终止](https://github.com/mkschleg/GVFHordes.jl/blob/8dbaf54bc55d510193ae7e58d9828badffcacc75/src/gvf/discount.jl)
- [目标策略](https://github.com/mkschleg/GVFHordes.jl/blob/8dbaf54bc55d510193ae7e58d9828badffcacc75/src/gvf/policy.jl)

第一次做：先定义“到达门口之前的累计能耗”，确认信号来自哪次 transition；再选择独立学习器。

运行条件：Julia；按 Project.toml 建隔离环境。README 简短，需要结合源码读接口。

检查范围：接口与许可证检查；未运行包测试或训练。

思考：同一个到达事件能否同时定义成 cumulant 和 termination？这会怎样改变回报？

### ReHorde.jl · 大量预测器的吞吐实验

问题很多时，每步学习成本怎样增长？

该仓库探索 Julia 中并行 GVF 更新的运行时间。README 明确说其重点是学习部分的 runtime，尚无误差计算；所需 Critterbot 数据不是公开随仓库提供。因此它适合研究系统开销，不是预测准确性 benchmark。

- [仓库](https://github.com/mkschleg/ReHorde.jl) · 研究作者的工程实验 · MIT
- [归属依据](https://github.com/mkschleg/ReHorde.jl)
- commit: 1da1b3f32a89cd50972e91e7711a1772700f9edb
- [GVF 与 TDλ 更新](https://github.com/mkschleg/ReHorde.jl/blob/1da1b3f32a89cd50972e91e7711a1772700f9edb/src/ReHorde.jl)
- [数据实验接口](https://github.com/mkschleg/ReHorde.jl/blob/1da1b3f32a89cd50972e91e7711a1772700f9edb/src/Critterbot.jl)
- [数据与测量边界](https://github.com/mkschleg/ReHorde.jl/blob/1da1b3f32a89cd50972e91e7711a1772700f9edb/README.md)

第一次做：先用合成可解析轨迹增加误差检查，再测线程数、预测器数与每步延迟。

运行条件：Julia / Tullio / LoopVectorization；真实数据有获取限制。

检查范围：源码与 README 检查；未运行吞吐实验。其状态相关 γ 取值约定需单独核对，不把它当作 RLPark GTD 的等价移植。

思考：吞吐变快时，学习结果、数值精度及完整 agent 的 wall-clock 是否也改善？

## Options：学习行为、发现技能、组合技能、用于规划

先读 Option-Critic 的策略与终止；再比较 eigenoptions、cover-time / planning-time 的发现目标；最后读 Option Keyboard 与 DADS 的复用接口。无监督技能不自动包含可用的 option model。

### Option-Critic · 表格与深度原实现

怎样同时学 option 内策略与终止函数？

不要先从 Atari 训练脚本开始。fourrooms/transfer.py 将 option-value、intra-option action value、softmax policy、sigmoid termination 分成短类，最适合对着原论文读梯度。深度分支则需要历史 Theano / Lasagne 栈。

- [仓库](https://github.com/jeanharb/option_critic) · 论文作者实现 · 未见仓库级 LICENSE
- [归属依据](https://github.com/jeanharb/option_critic)
- commit: 5d6c81a650a8f452bc8ad3250f1f211d317fde8c
- [一步 critic 与终止梯度](https://github.com/jeanharb/option_critic/blob/5d6c81a650a8f452bc8ad3250f1f211d317fde8c/fourrooms/transfer.py#L69)
- [Four Rooms 环境](https://github.com/jeanharb/option_critic/blob/5d6c81a650a8f452bc8ad3250f1f211d317fde8c/fourrooms/fourrooms.py)
- [深度训练入口](https://github.com/jeanharb/option_critic/blob/5d6c81a650a8f452bc8ad3250f1f211d317fde8c/train_q.py)

第一次做：读 F3，手算继续与终止的混合目标，再检查 A<0 时终止概率应增大。用小网格看 option 持续时间分布。

运行条件：表格脚本依赖旧 Gym、SciPy、dill；scipy.misc.logsumexp 已不适用于现代 SciPy。深度版另需 Theano、Lasagne、ALE；不要直接全局安装旧依赖。

检查范围：抽取原文件的核心类做隔离数值测试：critic 混合目标、环境终止 mask、termination 梯度符号；不是完整 Four Rooms / Atari 训练复现。

思考：算法学出一个步步终止的 option 时，是实现错误、目标本身允许，还是缺少技能使用约束？

### A2OC · Deliberation Cost

如何抑制没有收益的频繁切换？

在 Option-Critic 后读：切换代价进入奖励与终止优势，使“是否终止”具有代价权衡。它不同于直接把 option 长度固定成 K，也不同于平均奖励中的时间机会成本 gτ。

- [仓库](https://github.com/jeanharb/a2oc_delib) · 论文作者实现 · 未见仓库级 LICENSE
- [归属依据](https://github.com/jeanharb/a2oc_delib)
- commit: 42e25ce74c4ccc8e9e2f0667cb511aca4af45066
- [终止优势和切换代价](https://github.com/jeanharb/a2oc_delib/blob/42e25ce74c4ccc8e9e2f0667cb511aca4af45066/OC_theano.py#L73)
- [并行训练入口](https://github.com/jeanharb/a2oc_delib/blob/42e25ce74c4ccc8e9e2f0667cb511aca4af45066/train.py)

第一次做：追 delib_cost 的两处用途：critic 回报和 termination gradient；检查是否重复扣费、环境终止是否扣费。

运行条件：Theano / Lasagne / 旧 Gym Atari，多线程训练；未迁移到现代 PyTorch。

检查范围：静态检查终止项及 reward sequence；未训练。

思考：长技能带来的回报变化，来自减少切换代价还是更有用的时间抽象？

### Machado options · Laplacian eigenoptions

没有下游奖励时，如何从状态空间结构发现探索技能？

网格分支从邻接图构造归一化 Laplacian，以特征向量变化作为内在奖励，再解策略并加入 terminate。应将“图与表示”“内在奖励”“技能策略”“下游使用”四步拆开；该老仓库不自动代表后来的 ROD 或 Laplacian Keyboard。

- [仓库](https://github.com/mcmachado/options) · 作者主页链接的研究代码 · 未见仓库级 LICENSE
- [归属依据](https://mcmachado.github.io/research.html)
- commit: 176916c984cd1637fe5d51c8e7b677a9ce41f917
- [Laplacian → eigenpurpose → option](https://github.com/mcmachado/options/blob/176916c984cd1637fe5d51c8e7b677a9ce41f917/main.py#L42)
- [内在奖励是终点势函数减起点](https://github.com/mcmachado/options/blob/176916c984cd1637fe5d51c8e7b677a9ce41f917/Environment.py#L186)
- [策略求解与 option 使用](https://github.com/mcmachado/options/blob/176916c984cd1637fe5d51c8e7b677a9ce41f917/Learning.py)
- [最小地图](https://github.com/mcmachado/options/blob/176916c984cd1637fe5d51c8e7b677a9ce41f917/mdps/4rooms.mdp)

第一次做：先手画 4rooms，追 e(s′)−e(s)；观察特征向量正负号对应两种方向，并检查真正处理特征向量的遍历顺序。

运行条件：包含 Python 2 的 print / xrange；README 很短。网格和 Atari/ALE 应分开恢复，不能承诺现代 Python 开箱即用。

检查范围：静态核对内在奖励符号、归一化 Laplacian、反向遍历排序与 terminate；未运行 Python 2 / ALE。

思考：探索覆盖改善，是否也意味着规划迭代更少？奖励改变或动力学改变时哪些部件要重学？

### Jinnai 等 · Cover Time 与 Planning Time

“好技能”是让探索快，还是让规划快？

同一代码包对应两种不同目标：降低覆盖时间，以及减少规划时间。读它有助于理解为什么只展示技能轨迹不够：必须写出构造技能时优化了哪种效率。包含作者改过的 simple_rl，不能直接换成 pip 最新版本。

- [仓库](https://github.com/jinnaiyuu/Optimal-Options-ICML-2019) · 两篇 ICML 论文作者实现 · 根目录未见 LICENSE；内嵌 simple_rl 有单独许可证
- [归属依据](https://github.com/jinnaiyuu/Optimal-Options-ICML-2019)
- commit: 4f5cd1776b47f9b16c1022d22b2cc91d6044775b
- [技能构造：MOMI](https://github.com/jinnaiyuu/Optimal-Options-ICML-2019/blob/4f5cd1776b47f9b16c1022d22b2cc91d6044775b/options/option_generation/MOMI.py)
- [规划实验入口](https://github.com/jinnaiyuu/Optimal-Options-ICML-2019/blob/4f5cd1776b47f9b16c1022d22b2cc91d6044775b/options/experiments/planning_experiments.py)
- [带 options 的价值迭代](https://github.com/jinnaiyuu/Optimal-Options-ICML-2019/blob/4f5cd1776b47f9b16c1022d22b2cc91d6044775b/simple_rl/planning/ValueIterationOptionClass.py)

第一次做：固定小图和 option 数量，比较规划 sweep 数、环境步数与建模成本。

运行条件：Python 3 / NumPy / SciPy / NetworkX；精确优化分支还需 OR-Tools。

检查范围：论文归属、目录与依赖检查；未运行优化求解器。

思考：在已知完整图上找到的最优 option，在线未知模型时还需要多少发现成本？

### DeepMind · Option Keyboard / GPE–GPI

能否通过组合 cumulant，立即组合既有技能？

预测不同策略的 successor features，再用新奖励权重计算价值并做 GPI。这里是 2019 Option Keyboard；2026 Laplacian Keyboard 还增加了不同的表示与高层组合问题，不能混为同一个代码版本。

- [仓库](https://github.com/google-deepmind/deepmind-research) · 研究团队发布的论文实现 · Apache-2.0（研究代码，不是受支持产品）
- [归属依据](https://github.com/google-deepmind/deepmind-research/tree/master/option_keyboard)
- commit: f5de0ede8430809180254ee957abf36ed62579ef
- [GPI 对策略维取 max](https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/option_keyboard/keyboard_agent.py#L212)
- [训练 keyboard 和高层 agent](https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/option_keyboard/run_ok.py)
- [资源收集任务](https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/option_keyboard/scavenger.py)

第一次做：先在 Scavenger 中理解 [policy, cumulant, action] 张量维度，检查到底在哪个维度做加权和、在哪个维度取最大。

运行条件：按子目录 requirements.txt 恢复 TensorFlow / Sonnet 生态；固定仓库 commit 与子目录，不直接运行无关研究代码。

检查范围：GPI 张量操作与训练入口静态检查；未运行 TensorFlow 训练。

思考：GPI 的改善保证依赖什么价值估计误差界？累计特征的目标策略变了会怎样？

### DIAYN · 作者 SAC 分支

不用任务奖励，怎样学彼此可区分的技能？

正确入口在作者 sac 仓库的 DIAYN.md，而不是名字恰好叫 diayn 的第三方仓库。判别器的 log q(z|s) 与技能先验 log p(z) 形成伪奖励，再用最大熵控制学习。可区分不等于对将来任务有用。

- [仓库](https://github.com/ben-eysenbach/sac) · 作者项目页链接的实现 · BSD-2-Clause 文本；保留贡献者版权
- [归属依据](https://sites.google.com/view/diayn/)
- commit: 2116fc394749ca745f093a36635a9b253da8170d
- [原实验说明](https://github.com/ben-eysenbach/sac/blob/2116fc394749ca745f093a36635a9b253da8170d/DIAYN.md)
- [判别器伪奖励](https://github.com/ben-eysenbach/sac/blob/2116fc394749ca745f093a36635a9b253da8170d/sac/algos/diayn.py#L161)
- [技能训练入口](https://github.com/ben-eysenbach/sac/blob/2116fc394749ca745f093a36635a9b253da8170d/examples/mujoco_all_diayn.py)

第一次做：先核对 add_p_z 与先验；再看技能何时重采样、episode 终止如何处理，最后才看视频。

运行条件：历史 TensorFlow / rllab / MuJoCo 栈。仓库提供的 pickle checkpoint 不宜从不可信来源加载。

检查范围：作者归属与伪奖励实现静态检查；未运行训练或载入 checkpoint。

思考：判别器只根据位置区分技能，会不会奖励“待在不同地方”而非有用行为？

### DADS / off-DADS · 带动力学的技能发现

能否把学到的技能连同后果预测交给规划器？

与只判别当前状态的技能目标比较：DADS 同时学习 skill-conditioned dynamics，随后可在技能空间做 MPC。仓库还包含 off-DADS；要根据配置区分数据复用方式，而不是只根据文件名判断算法版本。

- [仓库](https://github.com/google-research/dads) · 研究团队发布的论文实现 · Apache-2.0
- [归属依据](https://github.com/google-research/dads)
- commit: abc37f532c26658e41ae309b646e8963bd7a8676
- [技能训练与模型规划入口](https://github.com/google-research/dads/blob/abc37f532c26658e41ae309b646e8963bd7a8676/unsupervised_skill_learning/dads_off.py)
- [训练 / 评测与缓冲区参数](https://github.com/google-research/dads/blob/abc37f532c26658e41ae309b646e8963bd7a8676/configs/template_config.txt)
- [历史环境](https://github.com/google-research/dads/blob/abc37f532c26658e41ae309b646e8963bd7a8676/env.yml)

第一次做：先追 process_observation 与模型预测对象；检查模型预测的是哪个坐标、哪个时间尺度，再读 MPC。

运行条件：旧 MuJoCo / TensorFlow 环境；作者 README 的目标任务评估主要配置在 Ant。

检查范围：代码入口、配置与版本区别检查；未训练或复现机器人结果。

思考：一步技能条件模型怎样变成多步规划模型？组合新技能时误差是否积累？

## Average reward：预测 → 控制 → 规划 → 深度 continuing 实验

从 Differential TD/Q 的一条更新开始，再看函数逼近和深度方法。平均奖励目标、持续交互协议、非平稳适应是三个维度，不应互相代称。

### Differential TD / Q / RVI-Q · ICML 2021

不用 γ<1，如何同时学相对价值与奖励率？

这是平均奖励入门最值得先读的原仓库。TD error 同时驱动价值和平均奖励估计，而不是另取行为策略奖励的简单均值。预测、控制和 planning_update 分开提供；自带旧算法作为实验对照。

- [仓库](https://github.com/abhisheknaik96/average-reward-methods) · 论文作者原实验仓库 · 未见仓库级 LICENSE
- [归属依据](https://proceedings.mlr.press/v139/wan21a.html)
- commit: ecfef0a5325608ef312a629909cd8ae39796809b
- [Differential TD 核心与重要性比](https://github.com/abhisheknaik96/average-reward-methods/blob/ecfef0a5325608ef312a629909cd8ae39796809b/agents/prediction_agents.py#L211)
- [Differential Q / planning_update](https://github.com/abhisheknaik96/average-reward-methods/blob/ecfef0a5325608ef312a629909cd8ae39796809b/agents/control_agents.py#L250)
- [AccessControl 配置](https://github.com/abhisheknaik96/average-reward-methods/blob/ecfef0a5325608ef312a629909cd8ae39796809b/config_files/control_AC_diff-q.json)

第一次做：先跑一个确定性 transition 的数值检查，再用 AccessControl 的小配置。README 示例写 control_AccessControl，但此 commit 的实际文件名是 control_AC_diff-q.json。

运行条件：作者测试环境 Python 3.7.6 / NumPy 1.18.1 / tqdm 4.40.2；本站核心检查使用 NumPy 1.26.2，未复现全套 sweep。

检查范围：直接导入原核心模块，测试 TD/Q 单步更新、planning_update 和小型 continuing 学习。另发现 prediction 的 past_rho 对 TwoChoice 状态作了专门处理，不能不改就移植成任意 off-policy 预测器。

思考：奖励率为什么用 TD error 更新？如果直接用行为奖励均值，学习的是哪条策略的奖励率？

### Naik · 统一 continuing 实验库 / Reward Centering

平均奖励、discounted TD 与 reward centering 怎样接在同一份代码里？

覆盖 Differential TD(λ)、Q/Sarsa、reward centering 与深度变体。作者将多项论文整合进一个库，并明确没有用统一版本重跑全部历史实验。适合扩展实验；要复现 ICML 2021 图表则先用原仓库。

- [仓库](https://github.com/abhisheknaik96/continuing-rl-exps) · 作者后续整合实现 · 未见仓库级 LICENSE
- [归属依据](https://github.com/abhisheknaik96/continuing-rl-exps)
- commit: de89d496bd90a5e078bd96d69ec4971b8bfc8877
- [预测与 traces](https://github.com/abhisheknaik96/continuing-rl-exps/blob/de89d496bd90a5e078bd96d69ec4971b8bfc8877/agents/prediction_agents.py)
- [γ、η 与更新顺序](https://github.com/abhisheknaik96/continuing-rl-exps/blob/de89d496bd90a5e078bd96d69ec4971b8bfc8877/agents/control_agents.py#L35)
- [深度 Q 变体](https://github.com/abhisheknaik96/continuing-rl-exps/blob/de89d496bd90a5e078bd96d69ec4971b8bfc8877/agents/control_agents_deep.py)

第一次做：做三格对照：γ<1、η=0；γ<1、η>0；γ=1、η>0。固定其余设置，检查它们对应的目标与更新。

运行条件：NumPy / 深度分支 PyTorch；部分环境使用作者 csuite fork，须连同配置固定。

检查范围：核对统一参数入口与 centering 次序选项；未运行完整库。保持 γ<1 的 reward centering 不因此变成平均奖励控制。

思考：收益来自目标改变、数值条件改善，还是奖励率估计对初始化更稳健？

### Differential GQ · Off-policy 函数逼近

平均奖励 off-policy policy evaluation 如何处理函数逼近？

该分支对应 Zhang、Wan、Sutton、Whiteson 的 ICML 2021 论文，包含 Diff-GQ1/2、Diff-SGQ 与 GradientDICE 对照。它回答固定目标策略的评估问题，不是任意非平稳深度控制算法的收敛证明。

- [仓库](https://github.com/ShangtongZhang/DeepRL) · 论文作者冻结分支 DifferentialGQ · MIT
- [归属依据](https://sites.google.com/view/yi-wan)
- commit: 05e38af518972fa80f36cef69ac78428553f4671
- [线性 Boyan chain 实验](https://github.com/ShangtongZhang/DeepRL/blob/05e38af518972fa80f36cef69ac78428553f4671/deep_rl/agent/LinearOPEAgent.py)
- [神经网络 OPE](https://github.com/ShangtongZhang/DeepRL/blob/05e38af518972fa80f36cef69ac78428553f4671/deep_rl/agent/NeuralOPEAgent.py)
- [配置入口](https://github.com/ShangtongZhang/DeepRL/blob/05e38af518972fa80f36cef69ac78428553f4671/template_jobs.py)

第一次做：先只看 linear_ope_boyans_chain；把奖励率、值函数与辅助变量各自更新的目标写出来。

运行条件：使用 DifferentialGQ 分支的 Dockerfile / requirements.txt，不是 master。作者警告此分支未用到的其他算法可能失效。

检查范围：论文—分支—文件身份核对；未执行 PyTorch / MuJoCo 实验。

思考：数据分布与目标策略的占用分布不同，哪一项修正负责解决偏差？

### Differential Value Iteration · DVI / RVI / MDVI

有已知模型时，怎样做平均奖励规划而不是采样学习？

把 sampling noise 拿掉，比较同步与异步规划更新、奖励率与相对价值的规范化。README 将项目描述为尚未完成的研究进展，不应把它称为最终通用规划器。

- [仓库](https://github.com/abhisheknaik96/differential-value-iteration) · 作者团队的规划研究代码 · Apache-2.0
- [归属依据](https://github.com/abhisheknaik96/differential-value-iteration)
- commit: e0aef39770d9baea7dc46f897baa463042df4c24
- [Differential VI](https://github.com/abhisheknaik96/differential-value-iteration/blob/e0aef39770d9baea7dc46f897baa463042df4c24/src/differential_value_iteration/algorithms/dvi.py)
- [Relative VI](https://github.com/abhisheknaik96/differential-value-iteration/blob/e0aef39770d9baea7dc46f897baa463042df4c24/src/differential_value_iteration/algorithms/rvi.py)
- [对照测试](https://github.com/abhisheknaik96/differential-value-iteration/blob/e0aef39770d9baea7dc46f897baa463042df4c24/src/differential_value_iteration/algorithms/golden_test.py)

第一次做：从 Micro / Queueing MDP 的已知转移开始，同时记录 Bellman residual、策略回报和迭代开销。

运行条件：Python / JAX；按源码版本恢复依赖与测试，未安装执行。

检查范围：项目状态与算法 / 测试入口核对；未声称其研究候选都有收敛保证。

思考：不同 reference function 只改变值函数常数，还是也改变了瞬态更新和稳定性？

### RVI-SAC · Average Reward Deep RL

如何在深度连续控制中使用平均奖励目标？

Hisaki 与 Ono 的 ICML 2024 方法。代码的目标包含策略熵；主变体还学习 reset cost 与 reset-rate critic。不能把它压缩成“普通 SAC 减去奖励均值”，也不能因为叫 continuing 就默认不允许 reset。

- [仓库](https://github.com/yhisaki/average-reward-drl) · 论文作者参考实现 · 未见仓库级 LICENSE
- [归属依据](https://proceedings.mlr.press/v235/hisaki24a.html)
- commit: b5b0d31bcd7e1778f11254490538e8662b01c2ed
- [critic target、f(Q) 与 reset-rate](https://github.com/yhisaki/average-reward-drl/blob/b5b0d31bcd7e1778f11254490538e8662b01c2ed/average_reward_drl/algorithms/rvi_sac.py#L151)
- [固定 reset cost 对照](https://github.com/yhisaki/average-reward-drl/blob/b5b0d31bcd7e1778f11254490538e8662b01c2ed/average_reward_drl/algorithms/rvi_sac_with_fixed_reset_cost.py)
- [方法配置](https://github.com/yhisaki/average-reward-drl/blob/b5b0d31bcd7e1778f11254490538e8662b01c2ed/experiments/conf/algo/rvi_sac.yaml)

第一次做：读 F8：先确认 reward 扣了哪些 cost，再追 terminated 与 next_state 的语义；随后比较固定与自适应 reset cost。

运行条件：仓库提供 uv.lock 与 Hydra 配置；示例为 Ant-v5。需要 PyTorch / MuJoCo；未执行 GPU / 控制训练。

检查范围：静态核对无 γ 的 critic target、熵项、f(Q) 与 reset cost；不能把局部代数检查当作论文性能复现。

思考：报告的 reward rate 是否包含 reset 代价和熵？评测与训练目标哪里不同？

### ARO-DDPG · 作者公开实验代码

确定性 actor–critic 怎样学习平均奖励 baseline？

可作为 RVI-SAC 的机制对照：这里的 rho 是待优化参数，critic loss 将 Q+rho 对齐到奖励加下一状态价值。不要把变量同名 rho 与 off-policy importance ratio 混淆。公开目录集中在 cheetah_run，不宜说它附带了完整通用 benchmark 包。

- [仓库](https://github.com/namansaxena9/ARO-DDPG) · 作者仓库；RVI-SAC 指向的原实现 · 未见仓库级 LICENSE
- [归属依据](https://arxiv.org/abs/2305.12239)
- commit: 5803e81b3bc0d1ee0e45236bae688002d6f1685a
- [critic 与 rho 的损失](https://github.com/namansaxena9/ARO-DDPG/blob/5803e81b3bc0d1ee0e45236bae688002d6f1685a/cheetah_run/ddpg_model.py#L219)
- [实验参数](https://github.com/namansaxena9/ARO-DDPG/blob/5803e81b3bc0d1ee0e45236bae688002d6f1685a/cheetah_run/config.py)
- [训练入口](https://github.com/namansaxena9/ARO-DDPG/blob/5803e81b3bc0d1ee0e45236bae688002d6f1685a/cheetah_run/train.py)

第一次做：先在固定 batch 上画出 Q 与 rho 的计算图，确认 target 网络、detach 与优化频率。

运行条件：PyTorch / 连续控制环境；缺少根目录 README 和完整统一安装说明，恢复成本较高。

检查范围：静态检查 critic loss 与 rho optimizer；该版本 target 使用 (1−done)，与 RVI-SAC 的 continuing target 处理不同，需先核对环境 done 语义。未训练。

思考：Q 的常数偏移和 rho 的估计如何区分？两者耦合会怎样影响训练？

### DeepRL-continuing-tasks · 21 个 continuing testbeds

如何公平比较无重置、预设重置和 agent-controlled reset？

基于 Pearl，将 MuJoCo 与 Atari 改成不同 continuing 协议，并比较 reward-centering 变体。适合把算法放回可审查的环境语义；21 个 testbeds 不等于 21 个严格 single-life、无任何外部干预的世界。

- [仓库](https://github.com/facebookresearch/DeepRL-continuing-tasks) · 论文团队代码与评测协议 · MIT
- [归属依据](https://github.com/facebookresearch/DeepRL-continuing-tasks)
- commit: def090b876b1866a263834d88b174476b36b4216
- [reset 的环境语义](https://github.com/facebookresearch/DeepRL-continuing-tasks/blob/def090b876b1866a263834d88b174476b36b4216/pearl/user_envs/wrappers/reset_wrapper.py)
- [centering 类型](https://github.com/facebookresearch/DeepRL-continuing-tasks/blob/def090b876b1866a263834d88b174476b36b4216/pearl/utils/functional_utils/learning/reward_centering.py)
- [环境协议和实验分组](https://github.com/facebookresearch/DeepRL-continuing-tasks/blob/def090b876b1866a263834d88b174476b36b4216/README.md)

第一次做：只挑一个 reset 条件；确认奖励、失败、reset 代价、bootstrap 和评测分母，再添加算法。

运行条件：PyTorch / Gymnasium / MuJoCo 或 ALE / AlphaEx；不作为第一个零依赖实验。

检查范围：仓库归属、reset 分组、centering 入口检查；未重跑 21 个 testbeds。

思考：自动重置后返回初始观测，为什么 continuing critic 仍可能需要 bootstrap？

### CSuite · Continuing 诊断环境

能否先用小环境定位算法问题，而不是立即跑 MuJoCo？

环境、wrapper 与测试分开提供，适合检查长期交互和非 episodic 接口。它是环境套件，不是平均奖励求解器；原仓库与 Naik 实验中使用的 fork 也要区分。

- [仓库](https://github.com/google-deepmind/csuite) · 研究团队 benchmark · Apache-2.0
- [归属依据](https://github.com/google-deepmind/csuite)
- commit: b74faed74685dc852d43c2022150c4186c244929
- [AccessControl 环境定义](https://github.com/google-deepmind/csuite/blob/b74faed74685dc852d43c2022150c4186c244929/csuite/environments/access_control.py)
- [Gym 接口](https://github.com/google-deepmind/csuite/blob/b74faed74685dc852d43c2022150c4186c244929/csuite/utils/gym_wrapper.py)
- [环境测试](https://github.com/google-deepmind/csuite/blob/b74faed74685dc852d43c2022150c4186c244929/csuite/csuite_test.py)

第一次做：先跑随机策略，画奖励流并检查是否出现 termination / truncation；然后再接 Differential Q。

运行条件：Python，按 requirements.txt 安装；选择正确的 dm_env / Gym wrapper 版本。

检查范围：协议与测试入口检查；未执行环境套件。

思考：环境没有 done，是否足以说明 agent 的状态、目标和参数从不重置？

## 公式与实现

### F1 · GVF 回报与终止信号

$$
G_t=c_{t+1}+\gamma_{t+1}G_{t+1},\quad c_{t+1}=r_{t+1}+(1-\gamma_{t+1})z_{t+1}
$$

本站 c-only 写法把终止信号吸收到 cumulant；RLPark 把 r 和 z 分开传入。z=0 时退回普通写法。γ 是延续系数，不是环境 done 的同义词。

检查：两步到达：c=(0,1)、γ=(0.9,0)，回报是 0.9；同样两步之后的终点价值折扣是 $0.9^2$。先说明预测对象，再判断下标。

思考：如果把 z 和到达事件 cumulant 同时加进去，会不会重复计算？

### F2 · GTD(λ)：旧权重、当前迹与下一时刻折扣

$$
\begin{aligned}\delta_t&=c_{t+1}+\gamma_{t+1}v_t^\top x_{t+1}-v_t^\top x_t\\e_t&=\rho_t(\gamma_t\lambda e_{t-1}+x_t)\\v_{t+1}&=v_t+\alpha[\delta_te_t-\gamma_{t+1}(1-\lambda)(e_t^\top w_t)x_{t+1}]\\w_{t+1}&=w_t+\alpha_w[\delta_te_t-(w_t^\top x_t)x_t]\end{aligned}
$$

这里沿用 RLPark 命名：v 是预测权重，w 是辅助权重。先算 correction 再更新 v、w；e 中的 γt 与 bootstrap 中 γt+1 不可合并。此式对应所选线性 GTD 实现，不是任意神经网络的稳定更新保证。

检查：数值测试故意令 γt=0.5、γt+1=0.9，避免常数折扣掩盖下标错误；还检查 correction 使用更新前的 w。Java 运行时未验证。

思考：若把更新后的 w 代入 correction，你得到的还是同一个随机逼近算法吗？

### F3 · Option-Critic：终止 option 不等于终止环境

$$
\begin{aligned}U(s',o)&=(1-\beta_o(s'))Q(s',o)+\beta_o(s')V(s')\\y&=r+\gamma(1-d)U(s',o)\\\Delta\theta_\beta&=-\alpha_\beta\nabla\beta_o(s')\,[Q(s',o)-V(s')]\end{aligned}
$$

d 表示本实验的环境终止；β 表示 option 终止。四房间代码用 max Q 构造 V；理论中 V 可由 option 选择策略定义，不能无条件互换。终止梯度的负号意味着较差 option 更应该结束。

检查：β=0.25、Q当前=2、V=6 时 U=3；r=1、γ=0.9 时 y=3.7。d=1 时 y=1。另用有限差分检查 sigmoid 终止的梯度符号。

思考：环境继续运行、只是 option 结束时，为什么不应该把 bootstrap 清零？

### F4 · 多步 option 与折扣终点模型

$$
\begin{aligned}Q(s,o)&\leftarrow Q(s,o)+\alpha\left[\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}+\gamma^\tau V(S_{t+\tau})-Q(s,o)\right]\\p_o^\gamma(s'\mid s)&=\mathbb E[\gamma^\tau\mathbf1\{S_{t+\tau}=s'\}]\end{aligned}
$$

一旦转移模型已经包含 $γ^τ$，规划中的 $r_o+Σp_o^γV$ 就不能再乘一个 γ。随机持续时间的 $E[γ^τ]$ 不等于 $γ^{E[τ]}$；随机终点与时长相关时，还要保留联合加权。

检查：三步奖励 (−1,−1,0)、γ=0.9、终点价值 10：backup=5.39。随机耗时 1/3、等概率时，终点价值项为 8.145，不是 8.1。

思考：若只存平均终点特征，非线性 V 是否仍能精确做 expectation-model backup？

### F5 · Differential TD/Q：奖励率也由 TD error 更新

$$
\begin{aligned}\delta_t&=R_{t+1}-\bar R_t+\max_aQ_t(S_{t+1},a)-Q_t(S_t,A_t)\\Q_{t+1}(S_t,A_t)&=Q_t(S_t,A_t)+\alpha\delta_t\\\bar R_{t+1}&=\bar R_t+\eta\alpha\delta_t\end{aligned}
$$

这是表格 Differential Q 的核心；固定策略预测时把 max Q 换成 v，并在相应 off-policy 版本中用 ρ 修正两项更新。它不同于仅对即时奖励做 EMA。代码还有值函数中心化辅助量，不能漏看。

检查：原核心类单步、planning_update 与确定性小链测试。额外标出预测代码对 TwoChoice 的 past_rho 特例：移植到一般状态依赖策略时必须重新实现覆盖与概率比。

思考：如果平均奖励只按行为轨迹均值更新，而 Q 用 max bootstrap，两个目标是否一致？

### F6 · 平均奖励 options：必须区分 τ 与 L

$$
\begin{aligned}\delta_n&=\widehat R_n-\bar R_nL_n(s,o)+\max_{o'}Q_n(s',o')-Q_n(s,o)\\Q_{n+1}(s,o)&=Q_n(s,o)+\alpha_n\delta_n/L_n(s,o)\\\bar R_{n+1}&=\bar R_n+\eta\alpha_n\delta_n/L_n(s,o)\\L_{n+1}(s,o)&=L_n(s,o)+\beta_n(\tau_n-L_n(s,o))\end{aligned}
$$

Wan、Naik、Sutton（NeurIPS 2021）式 (6)–(9)：L 是期望长度估计，要求正值初始化；同一步右侧均用旧 L。这里 R̂ 是 option 内未折扣奖励和。Bellman 方程中的时间成本可写 E[R̂−gτ+V]，但这个算法不是简单用样本 τ 替换 L。

检查：原论文 PDF 第 4 页逐式核对；本站写独立教学测试。反例：每个周期奖励 1，τ 等概率为 1 或 3。真实奖励率 E[R̂]/E[τ]=1/2；E[R̂/τ]=2/3，按样本长度除会改变零点。未找到该论文可确认的完整作者实验代码。

思考：先更新 L，再计算 δ/L，会不会改变论文规定的更新次序？

### F7 · Reward centering 与平均奖励不是同义词

$$
\delta_t=R_{t+1}-b_t+\gamma Q(S_{t+1},A')-Q(S_t,A_t)
$$

若 γ<1 且 b 是固定常数，无限时域 continuing 价值平移 b/(1−γ)，不会因此变成 γ=1 的 differential value。学习中的 b 会改变更新动力学。有限 episode、策略依赖终止或额外代价下，要另行检查这种平移论证。

检查：参数三格对照和常数 baseline 的解析测试。作者统一库还允许改变 baseline 与权重的更新次序，实验必须固定这些开关。

思考：当 reward centering 提高性能时，你能分开数值稳定、初始化和目标变化三种解释吗？

### F8 · RVI-SAC：熵、reference 与 reset cost 都在目标中

$$
y_t=r_t-c_{\rm reset}d_t-f_t+\min_iQ_i^-(s_{t+1},a')-\alpha_{\rm ent}\log\pi(a'\mid s_{t+1})
$$

对应固定版本 rvi_sac.py 的 critic target：没有 γ，也没有 (1−d) 截断整个 next-value；d 在此用于 reset 成本。f 由 sampled next soft values 的统计量更新，不是单纯奖励均值。是否在 reset 后正确 bootstrap，还依赖采样器给出的 next_state。

检查：静态核对 update_critic 的 151–191 行和 update_reset_cost；另有局部 target 算术测试。未运行 PyTorch / MuJoCo 训练，因此不作收敛或性能认证。

思考：若套用 episodic SAC 的 done mask，你还在实现这份 continuing 算法吗？

## 尚未定位到作者原实现的连接

### [Average-Reward Learning and Planning with Options](https://arxiv.org/abs/2110.13855)

作者主页与论文入口可找到 paper、talk、slides；尚未定位到可以确认归属、可下载的完整作者实验仓库。这里提供论文式 (6)–(9) 的独立教学测试，不将 primitive-action 仓库冒充 options 实验。

### [STOMP · Reward-Respecting Subtasks](https://arxiv.org/abs/2202.03466)

未确认完整作者原实现。检索到的 ramos-ai/STOMP 自述为第三方 work in progress，且未匹配原论文结果，不列入作者代码目录。子任务终止奖励的 $γ^{K-1}$ 约定与 option model 终点价值的 $γ^K$ 必须分别处理。

### [Laplacian Keyboard · 2026](https://arxiv.org/html/2602.07730v2)

可读论文附录 H 的代码式算法说明；尚未定位到可确认的配套作者仓库。2019 Option Keyboard 是相关前作，不是它的实现。
