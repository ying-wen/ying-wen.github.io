# 预测问题、资格迹与时间抽象怎样组合

这组实验适合已了解一步 TD、神经 critic 和策略梯度的读者。它不把“多加一个 head”“把动作重复几次”或“把过去梯度累加”当作充分解释。每项改动都要先说明问题，再说明更新及资源协议。

## 1. 三个不同的改动

| 机制 | 改变的对象 | 没有自动解决的困难 |
|---|---|---|
| GVF | 预测的 cumulant、continuation 和目标策略 | 状态充分性、覆盖、稳定更新、问题是否有用 |
| 资格迹 | 当前结果如何更新早期输出 | 观测记忆、非线性优化、旧策略覆盖 |
| Option | 决策单位、跨段 Bellman 备份、停止选择 | 内部策略学习、长期探索、模型维护 |

把三者组合，不是把三个名字接在一起。一个具体的智能体需要同时指定问题、表示、行为、更新时钟和存储预算。

## 2. 神经网络的梯度到底是什么

所有实验使用显式的单隐层 tanh 网络：

\[
h_j=\tanh(a_j^\top x+b_j),\qquad v_i(x)=\sum_j c_{ij}h_j+d_i.
\]

输出层梯度是 \(\partial v_i/\partial c_{ij}=h_j\)。输入层梯度是

\[
\frac{\partial v_i}{\partial a_j}=c_{ij}(1-h_j^2)x.
\]

输入层梯度依赖当前的输出权重和激活。更新之后，同一个旧输入的梯度会改变。固定线性特征的梯度不随权重变化，差异正发生在这里。

`_common.py` 的 `Network.value_gradient` 显式计算全部导数。没有自动求导黑箱，也不保留历史计算图。测试对两个输出、每个参数做中心有限差分。

### 半梯度不是完整 Bellman 残差梯度

先用旧参数计算 \(y=c+\gamma'v_\theta(s')\)。半梯度更新为

\[
\Delta\theta=\alpha[y-v_\theta(s)]\nabla_\theta v_\theta(s).
\]

本次更新把 y 当成常数。若同时对下一状态预测求导，方向变成与
\(\nabla v(s)-\gamma'\nabla v(s')\) 成比例。二者是不同算法。
即使对单次随机转移的平方 TD 误差完整求导，也不等于对条件期望 Bellman 误差的平方求无偏梯度；后者有 double-sampling 问题。

## 3. 同一世界中的两个 GVF

环境是四状态环。动作 0 原地不动，动作 1 向前一格。行为策略以 0.5 概率选各动作。两个问题各自指定：

| 问题 | 目标向前概率 | cumulant | continuation |
|---|---:|---|---|
| 主问题 | 0.8 | 到达状态 3 时为 1，否则为 0 | 到达状态 3 为 0，否则为 0.9 |
| 辅助问题 | 0.2 | 到达奇数状态为 1，偶数状态为 −1 | 0.8 |

主问题回答目标行为下的折扣到达量，不是窗口内到达概率。它在状态 3 终止累计，但真实世界继续运行。辅助问题也不随之停止。

实验中点将辅助 cumulant 乘以 −1。主问题和真实转移不变。这是控制变量干预，不是发现了自然变点。改变累计信号不等于 termination，所以程序不在中点偷偷清迹。

### 每条转移的计算顺序

1. 行为策略只采样一次动作；保存实际行为概率。
2. 同一转移分别产生各问题的 \(c^j,\gamma'^j,\rho^j\)。
3. 所有问题用同一组旧参数计算 \(\delta^j\) 和输出梯度。
4. 分别更新问题自己的迹。
5. 汇总共享参数的方向，然后只应用一次共享网络更新。

\[
\rho_t^j=\frac{\pi_j(A_t\mid S_t)}{b(A_t\mid S_t)},\quad
e_t^j=\rho_t^j[\gamma_t^j\lambda e_{t-1}^j+\nabla v_j(S_t)],\quad
\Delta\theta=\alpha\sum_j\delta_t^je_t^j.
\]

比率乘新梯度和旧迹两部分。迹用进入当前状态的 \(\gamma_t\)，target 用离开当前状态的 \(\gamma_{t+1}\)。当前终止信号必须先分配给过去资格；下一步才不延续旧迹。

### 四个独立入口

| 文件 | 改动 | 解释边界 |
|---|---|---|
| `gvf_shared_trace.py` | 共享 2–6–2 网络，λ=0.6 | 普通神经 IS 半梯度，不是有线性 GTD 保证的算法 |
| `gvf_frozen_trace.py` | 同一初始化，只训练输出层 | 不再学习表示；可能受固定表达能力限制 |
| `gvf_separate_trace.py` | 两个独立 2–6–1 网络 | 消除共享 trunk 干扰，参数量增加 |
| `gvf_shared_td0.py` | 同一共享网络，λ=0 | 移除长期资格，也改变多步逼近取舍 |

每个文件都有完整采样、target、迹和更新循环。共享文件只提供网络导数、小环境、解析参照与日志函数。

主图记录不变主问题的全状态 RMSE。辅助 RMSE 和共享更新内积在原始 CSV 中。内积为负只说明本次更新的局部方向存在冲突，不证明长期性能必然受损。不要因为某一条辅助误差下降，就断言学到的表示更适合控制。

真值由完整四状态模型的 Bellman 迭代计算，不来自任何被比较学习器。它用于小实验的独立检查；真实机器人通常没有这个真值。

## 4. 非线性资格迹：保存了什么，省略了什么

普通迹保存

\[
z_t=\sum_{k=0}^t(\gamma\lambda)^{t-k}\nabla v_{\theta_k}(S_k).
\]

它不是在当前参数下重新计算的

\[
\bar z_t(\theta_t)=\sum_{k=0}^t(\gamma\lambda)^{t-k}\nabla v_{\theta_t}(S_k).
\]

保存旧数值梯度是算法定义，并非一个程序错误。错误是据此宣称它满足使用固定当前网络推导的精确前后向等价。要重算右式必须保存旧输入，或使用另外一种近似；这改变了资源需求。

### 固定参数时的恒等式仍适用于非线性网络

冻结网络后，令 \(g_t=\nabla v_\theta(S_t)\)，所有梯度只求一次。
有限前向目标递推为

\[
G_t^\lambda=R_{t+1}+\gamma_{t+1}[(1-\lambda)V_{t+1}+\lambda G_{t+1}^\lambda].
\]

终点尾值为零。非终止窗口尾值是明确指定的预测，而不是零。
交换求和可得到

\[
\sum_t(G_t^\lambda-V_t)g_t=\sum_t\delta_t z_t.
\]

这个总增量恒等式不要求网络是线性的，但要求整次比较冻结参数。
True-online TD 的逐前缀等价是另一个更强的结论，其常见推导依赖固定线性特征。把 x 替换成当前神经梯度不能保留该证明。

### 三个真实训练循环

六步链只有最后一步获得 Bernoulli(0.7) 奖励。折扣 0.9，状态完整可观测。真值是 \(v(s)=0.7\,0.9^{5-s}\)。三个算法使用同一 2–6–1 网络、初始化、奖励序列和步长 0.03。

| 文件 | 每条真实转移后做什么 | 历史存储 |
|---|---|---|
| `neural_td_trace.py` | λ=0.8 的神经半梯度 TD | 参数大小的数值资格向量 |
| `neural_td0.py` | 一步神经 TD | 不保留多步资格 |
| `neural_gradient_mc.py` | 等六步结束后构造回报，正序逐样本更新 | 最多六条轨迹 |

这是平稳同策略预测。它有意不同时加入部分可观测性、策略优化和大重要性比率。先在此理解梯度和更新时序，再扩展任务。

MC 没有 bootstrap，但要等待目标并保存输入。它与严格流式 TD 的协议不同。
预算最后一步若不是 episode 终点，MC 不使用尚未完整的轨迹。日志切分不提供伪造终止的权限。

`gradient_drift` 只比较一个旧输入在相邻参数版本下的梯度。它不等于完整长历史资格的误差。为这个诊断额外保存一个输入和一个梯度，内存不随时间增长。

## 5. Option 改变的是 Bellman 时间尺度

若一个 option 执行 \(\tau\) 个原始步，则

\[
y=\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}+\gamma^\tau\max_{o'}Q(S_{t+\tau},o').
\]

停止 option 后仍可继续选择下一个 option。只有真实环境终止时尾值为零。
不同技能长度下仍应先固定 primitive 交互预算，否则长技能可能暗中获得更多数据。

### 闭环最小实验

四状态环上有两个给定 options。第一个执行一次向左。第二个根据当前状态移动到信标 3：在状态 0、3 向左，在状态 1、2 向右；到达 3 后停止。它在不同状态采取不同动作，持续一或两步，并非固定动作重复。

奖励为每步 −0.05，到达 3 再加 0.4，到达 0 再加 0.12。真实环境不终止。高层均匀随机选择 option，所以正负对照在同一 seed 下走完全相同的轨迹。

`neural_option_smdp.py` 使用真实持续时间。
`neural_option_wrong_duration.py` 故意把尾折扣写成一个 γ。后者是错误诊断，不是研究算法。
二者使用相同 2–6–2 网络、步长 0.025、不重放。主图是相对于正确小模型最优 Q 的 RMSE。

预算末尾尚未完成的 option 不强制结束。日志记录真实 primitive 步数与已完成的高层更新次数。

### 与 Option-Critic、模型学习的关系

这里固定内部策略与停止规则，尚未学习两者。神经 Option-Critic 还需内部动作梯度和到达状态的终止梯度：

\[
\nabla_\vartheta U_{\rm local}(s',o)=-\nabla_\vartheta\beta_\vartheta(s',o)[Q(s',o)-V(s')].
\]

这是固定 critic 的局部偏导；完整目标梯度需要正确的到达占用权重。
内部策略或 β 改变后，option 模型的终点和时间分布也改变。旧经验的一个动作比率不足以修正完整技能的变化路径。
加入共享表示、技能发现和规划后，需要分别衡量每一层的误差，不能将这两个小实验称为完整持续智能体实现。

## 6. 怎样运行与核对

在仓库根目录执行。输出目录必须尚不存在。

```bash
python3 implementations/streaming_composition/gvf_shared_trace.py --steps 1200 --seeds 0 1 2 3 4 --out results/gvf-shared-new
python3 implementations/streaming_composition/neural_td_trace.py --steps 1200 --seeds 0 1 2 3 4 --out results/neural-trace-new
python3 implementations/streaming_composition/neural_option_smdp.py --steps 1200 --seeds 0 1 2 3 4 --out results/option-duration-new
python3 -m unittest discover -s tests -p test_streaming_composition.py -v
```

直接运行任何一个文件都会默认运行其配对基线。曲线来自真实运行记录，均值与跨训练种子的一个样本标准差分别导出。标准差带不是置信区间。

测试覆盖：全部神经参数的有限差分；半梯度与残差梯度的区别；普通 IS 比率乘积位置；λ=0、γ=0、真正终止和窗口尾值；冻结非线性前后向恒等式；在线变参反例；问题伪终止；option 实际时长；解析 Bellman 残差；完整运行、种子重现和直接文件 CLI。

## 7. 原文与作者实现

- [Horde](https://sites.ualberta.ca/~amw8/horde.pdf)：先读 question 与 answer 分工，再读固定线性特征的离策略学习。本文小型神经实验不继承其线性保证。
- [Emphatic TD](https://www.jmlr.org/papers/v17/14-488.html)：有效状态权重与稳定性条件；重要性比率不是对所有不稳定的统一答案。
- [True Online TD](https://proceedings.mlr.press/v32/seijen14.html)：逐前缀在线前向参考是精确等价的对象，不是任意神经 TD(λ)。
- [Gradient Eligibility Traces](https://arxiv.org/abs/2507.09087) 与 [作者 QRC](https://github.com/esraaelelimy/gtd_algos/blob/76293dea9b2129d55e08bfb4178618a0a26c2dd8/gtd_algos/src/algorithms/qrc.py)：检查标量辅助迹、辅助梯度迹、价值梯度迹以及主辅助方向的同步计算。这里未重做作者的 MinAtar 或 MuJoCo 实验。
- [原始 options 框架](https://doi.org/10.1016/S0004-3702(99)00052-1)：半马尔可夫、intra-option 与 option model 的不同接口。
- [Option-Critic](https://arxiv.org/abs/1609.05140) 与 [作者代码](https://github.com/jeanharb/option_critic)：动作与终止两个导数的状态索引、critic 时序和实际神经工程。

## 8. 研究问题

1. 固定主问题，只改变辅助问题。性能变化来自共享梯度、容量还是训练分布？怎样设置等容量对照？
2. 在相同真实交互和额外计算预算下，资格迹、短窗口重算和 replay 各得到什么优势？
3. 改变 λ 后误差下降，是信用传播更快，还是逼近固定点改变？怎样加入精确表格参照？
4. 更新一个 option 的内部策略后，应该多快更新其模型？只检查一步误差能否检测持续时间分布的错误？
5. 共享表示更新时，预测、控制和模型各自的误差是否同步变化？一项损失下降是否足以说明整体智能体改善？

每个问题都需要先给定实验协议。增加网络、损失或缓冲区只是候选机制，不是问题已经解决的证据。
