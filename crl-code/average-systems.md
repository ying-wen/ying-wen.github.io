# 平均奖励：预测、控制与规划的完整教学实验

本组把单步公式放回真实的多状态经验流中。八个独立文件分别包含更新与运行循环。只有环境、精确评价器和日志辅助函数共用。代码仅依赖 Python 标准库。

## 1. 先确定求解对象

预测给定目标策略 π，求每原始时间步的奖励率 gπ 与相对价值 hπ。控制允许改变策略，目标是提高真实可执行策略的奖励率。规划改变求解器可用的信息，不改变奖励目标。已知模型规划可以直接查询 P 与 r；学习模型规划只能利用已经观察的数据。

在本章有限遍历链上：

\[
g_\pi=\lim_{T\to\infty}\frac1T E_\pi\!\left[\sum_{t=0}^{T-1}R_{t+1}\right]
     =d_\pi^\top r_\pi,\qquad d_\pi^\top P_\pi=d_\pi^\top.
\]

将 T 步价值递推中的线性增长项 Tgπ 消去，得到 Poisson 方程：

\[
g_\pi\mathbf1+h_\pi=r_\pi+P_\pi h_\pi.
\]

(I−Pπ) 消去常数向量。因此 h 只确定到一个常数。精确评价器指定 h(0)=0。误差计算也先做此对齐，不把算法自行选择的常数偏移误当成预测错误。同时保留未经对齐的偏移诊断，避免掩盖错误奖励率造成的漂移。

## 2. 同一个三状态双动作问题

状态 s∈{0,1,2}，动作 a∈{0,1}，没有终止和外部重置。每次执行一个原子动作耗时一步。

| 状态 | 动作 0 奖励 | 动作 1 奖励 | 动作 0 首选终点 | 动作 1 首选终点 |
|---|---:|---:|---:|---:|
| 0 | 0.10 | −0.05 | 1 | 2 |
| 1 | 0.50 | 0.05 | 0 | 2 |
| 2 | 0.20 | 1.20 | 0 | 1 |

转移以概率 0.8 到首选终点，以概率 0.2 均匀选择三个终点。因此每个状态动作到每个终点的概率至少为 1/15。所有平稳策略的链都不可约且非周期。

固定目标策略选择动作 1 的概率依次为 (0.85,0.75,0.8)。异策略数据行为在每个状态各选两个动作一半。精确值是：

- gπ = 0.4806503050，hπ = (0, 0.1232321131, 0.6497504160)。
- gb = 1/3。
- 最优确定性策略三个状态都选动作 1；g* = 0.5970370370。

评价器通过高斯消元求解三个 Poisson 方程加 h(0)=0，并通过枚举八条确定性策略求最优奖励率。测试另外求稳态分布验证 g=dᵀr。**真实 P 只用于评价，学习器和经验模型不能读取这些答案。** 已知模型 RVI/VI 是明确的例外。

## 3. 同策略预测

### Differential TD

~~~
differential_td_onpolicy.py
~~~

\[
\delta_t=r_{t+1}-\bar g_t+h_t(s')-h_t(s),\quad
h_{t+1}(s)=h_t(s)+\alpha\delta_t,\quad
\bar g_{t+1}=\bar g_t+\eta\alpha\delta_t.
\]

先用旧 h 与旧 g 算一次 δ，再分别更新。此处 α=0.08，η=0.2。它是常数步长有限预算实验，不宣称渐近精确收敛。

### 奖励均值 TD 对照

~~~
sample_mean_td.py
~~~

该对照使用同样的价值 TD 更新，但用在线样本均值更新奖励率。因为 b=π，这是合理的估计器，不是错误算法。它与 TD 驱动奖励率更新的瞬态和耦合方式不同。把逐样本均值直接用于非平稳追踪则是另一个问题：历史权重逐渐降低新数据影响。

## 4. 异策略预测与故意错误的负对照

~~~
differential_td_offpolicy.py
wrong_behavior_mean_td.py
~~~

令 ρ=π(a|s)/b(a|s)。正确的表格 Differential TD 在 h 与 g 两条更新中都使用 ρδ：

\[
E_b[\rho\delta\mid s]=r_\pi(s)-\bar g+(P_\pi h)(s)-h(s).
\]

表格情况下每个被覆盖的状态都能对应一个条件方程。重要性比需要动作覆盖；它不会创造未到访状态的信息。

负对照仅在 h 更新中乘 ρ，g 却直接平均行为奖励。即使 h 已经准确，其条件残差仍是 gπ−gb≈0.147317。它有意求错了问题，不是推荐算法。

**为何保留看起来可能更好的错误 bias 曲线？** 错误 g 的影响可能主要表现为共同偏移不断增长。对齐后的 bias 误差偶尔会小于正确方法。故同时记录 gain_abs_error、raw_bias_offset、真实行为均值和 bias RMSE。不能从一个归一化指标推断全部正确。

## 5. 多状态控制

~~~
differential_q_multistate.py
~~~

\[
\delta=r-\bar g+\max_{a'}Q(s',a')-Q(s,a).
\]

Q(s,a) 增加 αδ，g 增加 ηαδ；α=0.08，η=0.1。行为是 ε=0.25 的 ε-greedy，平局选动作 0。单步控制目标使用 max，而条件转移来自实际 (s,a)，不需要额外乘目标动作重要性比。

评价器冻结贪心策略，解其真实平均奖励。探索经验均值另行记录。它通常小于冻结贪心策略奖励率，因为行为仍会探索。学习器内部的 g 估计又是第三个量：它针对最优性方程，不是当前行为奖励均值。

## 6. 学习模型再规划

~~~
differential_dyna.py
~~~

每个真实转移做一次 Differential Q 更新，然后记录：

\[
\widehat r(s,a)=\frac{\text{奖励总和}}{N(s,a)},\qquad
\widehat P(s'|s,a)=\frac{N(s,a,s')}{N(s,a)}.
\]

从已观察的状态动作中均匀抽取一项，做一次期望模型备份：

\[
\delta^{model}=\widehat r(s,a)-\bar g+
 \sum_{s'}\widehat P(s'|s,a)\max_{a'}Q(s',a')-Q(s,a).
\]

Q 与 g 共用旧参数误差。每个真实步额外规划 K=5 次。规划随机数流独立于行为环境随机数流，防止额外规划消耗环境随机数制造隐蔽差异。

N=0 的条目不得规划。早期模型可能错误且欠覆盖。多次备份会同时加速有用信息传播与错误模型传播。这里没有外推固定正确模型规划理论到任意在线耦合系统。

预算分开：

- environment_updates = t。
- model_backups = 5t。
- total_backups = 6t。
- 一个模型备份求和三个终点，它也不等同于一次常数成本样本备份。

比较的横轴是相同环境步，不是相同计算预算。真实回报曲线不会假设 Dyna 必然优于基线。

## 7. 已知模型 RVI 与不锚定 VI

~~~
rvi_multistate.py
unnormalized_vi_multistate.py
~~~

\[
(Th)(s)=\max_a\{r(s,a)+P_{s,a}h\},\qquad
h_{k+1}=Th_k-(Th_k)(0)\mathbf1.
\]

另一条对照直接保留 Th，不减常数。它们在精确算术中的相对值完全相同。未锚定版本绝对值线性增长，不应编造为必然策略失败。图示 bias RMSE，日志另存共同偏移与奖励率估计。

横轴是 model_sweeps，每扫描六个状态动作期望备份。没有真实环境交互。该实验是确定性的；重复种子应得相同结果，零误差带不是五个独立随机环境支持的泛化证据。

## 8. 不要遗漏的困难

**周期性。** 确定性两状态环奖励为 0、2，g=1、h=(0,1) 存在。但同步 RVI 从零开始使 h(1) 在 0 与 2 之间振荡。解存在不意味着选定迭代器收敛。

**多常返类。** 两个互不连通的吸收状态分别每步奖励 0、2。单个标量 g 无法同时满足两个状态方程。求解器应报奇异/不适用，而不是静默给出唯一奖励率。

**慢混合。** 有限轨迹可能长时间只覆盖部分区域。估计误差、策略比较与参考量更新都会受影响。大 γ 不是平均奖励目标的替代定义。在有限遍历固定策略下 (1−γ)Vγ→g 是极限结论，不表示任意 γ=0.99 已足够。

**非线性逼近。** 表格 h 更新只改变一个坐标，因此保持 g−η∑h 不变。神经网络一次参数更新会改变许多状态预测；一般不保持此不变量。重要性比不修正所有投影与表示变化，目标网络不自动恢复严格收缩。线性保证与非线性经验结果需分别陈述。

**双时间尺度。** 本组 Differential TD/Q 使用固定比例 η。它不是依赖步长比趋零的 actor–critic 证明。把网络学习率调小不能代替检查具体定理条件。

**日志边界。** t=1200 只是实验停止收集数据的位置，不是吸收终止。最后一次真实转移仍需 bootstrap。模型、参数与状态不会在统计窗口处清零。

## 9. 运行、测试与阅读次序

在仓库根目录运行，输出目录必须不存在：

~~~
python -m implementations.average_systems.differential_td_offpolicy --steps 1200 --seeds 0 1 2 3 4 --out results/average-prediction
python -m implementations.average_systems.differential_dyna --steps 1200 --seeds 0 1 2 3 4 --out results/average-dyna
python -m implementations.average_systems.rvi_multistate --steps 1200 --out results/average-rvi
python -m unittest discover -s tests -p test_average_systems.py -v
~~~

建议次序：先读 _common.py 中任务定义与 evaluate；再读 on-policy TD；然后看 off-policy 负对照；最后比较多状态 Q、Dyna、RVI。不要先从绘图判断算法好坏。先验证每条曲线究竟在测量什么。

测试覆盖手算两状态 Poisson、独立稳态分布、旧参数时序、常数规范、重要性校正、行为覆盖、负对照、终止误用、经验模型计数与期望、额外规划预算、RVI相对值一致性、多常返类失败以及八个入口的重复性。

## 原文与作者实现

- [Wan, Naik & Sutton, ICML 2021](https://proceedings.mlr.press/v139/wan21a.html)：Differential TD/Q 与模型规划的核心来源。本组 Dyna 是显式经验模型教学组合，不是论文整套 benchmark 复现。
- [作者代码](https://github.com/abhisheknaik96/average-reward-methods)：agents/prediction_agents.py、agents/control_agents.py；先核对共享 TD error，再核对平均奖励与价值步长。
- [Weakly communicating MDPs](https://arxiv.org/abs/2209.15141)：更一般表格控制与选项控制结构；不等价于任意非平稳神经网络保证。
- [Average-reward off-policy evaluation with function approximation](https://proceedings.mlr.press/v139/zhang21u.html)：区分线性收敛分析和非线性实验。
- [RVI-SAC](https://proceedings.mlr.press/v235/hisaki24a.html)：深度控制、soft policy improvement 与 reset cost；本组八个入口未声称实现完整 RVI-SAC。

