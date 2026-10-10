# 扩展适应机制：可运行训练与组件实验

这一组实验区分三个问题：怎样从历史形成状态，怎样把误差传给参数，怎样在数据变化时调整学习过程。
状态过滤、时间梯度、元梯度与参数回收并不解决同一个问题。下面先列出各方法的实际机制和测量方式，再说明它们之间的边界。
REINFORCE、完整 BPTT 在给定小型问题中给出完整更新；复杂系统则只实现明确说明的模块或特例，不代表原论文完整控制系统。

## 实验与读图

| 方法/文件 | 实际机制 | 同任务基线 | 横轴和主指标 |
|---|---|---|---|
| REINFORCE / reinforce | 完整回合 Monte Carlo score-function 策略梯度 | policy_td | 环境转移；冻结贪心 Chain 回报 |
| GAE / gae | 残差逆向递推、终止掩码、采样截止 bootstrap | policy_td | 环境转移；冻结贪心 Chain 回报 |
| V-trace / vtrace | 非同策略多步截断比率目标、价值 SGD | is_td | 环境转移；已知目标策略价值 MSE |
| Bayes filter / bayes_filter | 已知二状态 HMM 预测与归一化校正 | memoryless_filter | 观测数；在线真实隐藏状态负对数损失 EMA |
| 完整 BPTT / bptt | 8输入最终监督全时间反向传播 | truncated_bptt | 序列数；固定32序列 MSE |
| RTU / rtu | 一对旋转递归单元、非线性局部 RTRL 敏感度 | bptt | 序列数；同固定序列 MSE |
| Metatrace / metatrace | AC(lambda)、标量元步长与归一化敏感度递推 | trace_ac | 环境转移；冻结贪心 Chain 回报 |
| Meta-gradient RL / meta_gradient | 对 lambda-return 内层价值更新求元导数 | fixed_lambda | train/validation轨迹批数；已知 random-walk 价值 MSE |
| MAML / maml | 支持集内层适应与查询集二阶外层梯度 | joint_training | 4任务元批数；固定24未见任务适应后 MSE |
| ObGD2024 / obgd2024 | TD(lambda) 半梯度与原2024 L1过冲约束 | streaming_td | 环境转移；已知交替状态价值 MSE |
| Intentional Updates / intentional | RMSProp尺度、trace、sigma、功能步长、TD剪裁/策略归一化 | trace_ac | 环境转移；冻结贪心 Chain 回报 |
| CLEAR / clear | 终止bandit的单步V-trace回放及策略/价值克隆、蓄水池 | fresh_ac | 环境转移；当前上下文随机策略精确期望奖励 |
| ReDo / redo | 最近32样本相对平均激活检测、入边重置/出边置零 | plain_mlp | 训练样本数；固定64样本当前函数 MSE |
| Continual Backpropagation / continual_backprop | contribution效用EMA/偏差校正、成熟期和累计替换信用 | plain_mlp | 同训练样本；同冻结 MSE |

Chain 的有限12步截止是真终止，剩余时间在观测里；没有将真实终止当作截断。
完整 REINFORCE 不利用预算尾部未完成回合。GAE 的未终止批末则从已采样下一状态价值启动。
贪心冻结回报是统一可比控制指标，**不等于**随机策略优化目标的期望回报。

Bayes filter 是状态推断而非参数学习：隐藏状态只用于诊断，不能进入过滤更新。
它的损失是同一生成流上的 EMA，不是独立冻结评价；所有其他冻结测量不消耗训练随机流。
CLEAR的两上下文偏好固定，后半只访问第二上下文，`old_return` 才是旧任务保持诊断。
切换回归后新旧目标冲突，`old_mse` 是旧函数误差，不能把不可兼得的两个函数当作保持失败证明。

RTU 的参数在8输入序列内固定，末端一次更新，因此手写 RTRL 与该序列梯度精确相等；
并非跨更新无限流的完全导数，更没有复现作者 PPO 系统。
Metatrace 实现 scalar normalized AC(lambda) 的局部敏感度近似，忽略 trace 对参数的高阶导数；
没有冒充 per-parameter 变体。
MAML 是线性监督任务元学习，不是 MAML-RL。Meta-gradient 是预测任务的截断一批元导数，
不保留跨批无限计算图，不是完整 actor-critic meta-RL。
ObGD 使用2024公式，未混入后续不同算法；未包含 Stream-X 的全部网络/归一化技巧。
Intentional 保留原优化器核心顺序但仅线性离散 Chain，没有连续控制/Atari系统。
CLEAR 此处每拉杆真终止，V-trace退化成单步；无分布式多步actor。
ReDo/CBP 仅单隐藏层SGD且无动量，所以没有待重置的 optimizer moments。
CBP选作者代码支持的 contribution 效用，未包含 centered/adaptable变体和均值偏置补偿。

## 预算不是计算成本

所有同task基线在 `task/family/budget/metric/unit/higher_better` 完全一致。
同一主横轴不表示同一 FLOPs、参数量或优化器次数：
Chain策略/价值批方法通常各一个optimizer；Metatrace/Intentional是逐步向量更新；
CLEAR每步联合优化读新样本和旧样本；fresh_ac只读新样本。
元梯度每批含两条轨迹，实际 `environment_steps` 另记；MAML每元批64监督样本且二阶反传。
序列实验每步计一整序列，`element_steps=8*step`；回归每样本一次 SGD。
记录网格统一从0到总预算、至多21行。诊断列记录相应误差、步长、敏感度、回收数或旧知识。
短预算/少种子运行仅检查有限数值与更新链条，不作为优越性结论。

## 原文与作者实现核对

独立教学实现，不复制作者文件。公开原文和作者代码用于检查公式和顺序：

- [V-trace/IMPALA](https://arxiv.org/abs/1802.01561)，[RLax V-trace实现](https://github.com/google-deepmind/rlax/blob/main/rlax/_src/vtrace.py)。
- [REINFORCE/VPG教学公式](https://spinningup.openai.com/en/latest/algorithms/vpg.html)，[GAE原文](https://arxiv.org/abs/1506.02438)。
- [Bayes过滤教学原始课程](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/bayes-filters.pdf)；已知模型的精确二状态过滤。
- [BPTT原文](https://doi.org/10.1162/neco.1990.2.4.490)，[RTU原文](https://arxiv.org/html/2409.01449v1)，[RTU作者仓库](https://github.com/esraaelelimy/rtus)。
- [Metatrace算法1](https://arxiv.org/html/1805.04514v2)；未定位可信公开作者代码，核对原文而不宣称作者实现等价。
- [Meta-gradient RL](https://arxiv.org/abs/1805.09801)，[MAML原文](https://arxiv.org/abs/1703.03400)，[MAML作者代码](https://github.com/cbfinn/maml)。
- [ObGD2024作者优化器](https://github.com/mohmdelsayed/streaming-drl/blob/main/optim.py)。
- [Intentional Updates原文](https://arxiv.org/html/2604.19033v1)，[作者优化器](https://github.com/sharifnassab/Intentional_RL/blob/main/optimizer.py)：偏差校正、先统计后trace、sigma、剪裁/归一化、最后清终止trace。
- [CLEAR原文](https://arxiv.org/abs/1811.11682)：本文只验证终止bandit特例，没有定位/复制完整原作者分布式训练码。
- [ReDo原文](https://arxiv.org/abs/2302.12902)，[作者回收器](https://github.com/google/dopamine/blob/master/dopamine/labs/redo/weight_recyclers.py)。
- [Continual Backpropagation原文](https://arxiv.org/abs/2108.06325)，[作者GnT](https://github.com/shibhansh/loss-of-plasticity/blob/main/lop/algos/gnt.py)。

## 运行与验证

```sh
python3 implementations/extended_adaptation/rtu.py --steps 1200 --seeds 0 1 2 --out /tmp/rtu-new-results
python3 -m unittest discover -s tests -p test_extended_adaptation.py -v
```

测试含 GAE/V-trace 真终止与截止手算、Bayes归一化、BPTT与RTU全参数有限差分、
MAML二阶梯度、lambda元梯度有限差分、Metatrace旧敏感度时序、
Intentional首步功能改变量/终止清trace、蓄水池容量和回收入/出边，
以及全部24方法3种子的有限日志和baseline契约检查。
测试还逐个子进程执行24个文件的直接CLI（40步、seed0、新结果目录），验证网页命令路径。
验收运行另以1200预算、seed0/1/2运行全部24方法，主指标及所有数值诊断均有限；
这些检查不要求某方法改善，不隐去短预算负面结果，也不支持方法效果优越性的结论。
