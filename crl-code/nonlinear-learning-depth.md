# 函数逼近为什么改变强化学习：从误差定义到可运行反例

这份材料面向知道 TD target 和反向传播、但尚不熟悉函数逼近理论的读者。先区分四件事：想预测什么、能表示什么、能观察什么，以及如何更新。八个程序用小问题把这些区别变成可以检验的结果。它们都是机制诊断，不是大规模论文复现。

## 1. 一个明确的问题

先固定有限 MDP 和策略 π。设奖励有界，折扣 γ < 1。真实价值满足

$$v_\pi(s)=\mathbb E_\pi[\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}\mid S_t=s],\qquad v_\pi=r_\pi+\gamma P_\pi v_\pi.$$

网络选择函数族 $\{v_\theta\}$。评价指定状态分布 $d$。训练数据由行为策略 $b$ 或生成模型提供。更新算法将数据变成参数变化。后三者不会重新定义真实价值。

五类错误必须分开：

1. **信息不足**：状态压缩丢失了影响未来的历史。优化无法补回不存在的信息。
2. **表示不足**：函数族无法同时表达所有所需预测。
3. **统计不足**：状态覆盖不足，或后果样本太少。
4. **更新不稳定**：即使真值可表示、采样充分，期望更新仍会离开解。
5. **问题变化**：策略、环境、特征或预测问题改变，原来的固定点不再是目标。

## 2. 表格、固定线性与非线性

线性预测为 $v_w(s)=x(s)^Tw$。执行一次 TD 更新后，另一个状态的预测改变为

$$v_{w+\alpha\delta x(s)}(\bar s)-v_w(\bar s)=\alpha\delta x(\bar s)^Tx(s).$$

表格特征使不同状态正交。一般特征产生共享。共享可能迁移有用知识，也可能相互干扰。

平滑神经网络的小步更新满足

$$\Delta v(\bar s)\approx\nabla v_\theta(\bar s)^T\Delta\theta.$$

梯度内积决定局部交叉影响。但梯度本身也随 θ 变化。网络同时学习答案和决定泛化关系的表示。这比“损失非凸”更具体：更新会改变下一次学习所使用的局部几何。

## 3. 价值误差、Bellman 残差与样本残差

$$J_{VE}=\tfrac12\mathbb E_d[(v_\pi-v_\theta)^2],\quad J_{BE}=\tfrac12\mathbb E_d[(T_\pi v_\theta-v_\theta)^2].$$

令 $\delta=R+\gamma v_\theta(S')-v_\theta(S)$。样本平方误差是 $J_{sample}=\frac12\mathbb E[\delta^2]$。它不等于 $J_{BE}$，因为

$$J_{sample}=J_{BE}+\tfrac12\mathbb E_S[\operatorname{Var}(R+\gamma v_\theta(S')\mid S)].$$

方差项一般依赖参数。减小样本残差可能同时压低后继预测的变化，而不是只改善 Bellman 期望方程。

半梯度 TD 的更新为

$$\theta^+=\theta+\alpha\delta\nabla v_\theta(S).$$

如果对样本残差完整求导，则负梯度为

$$-\nabla(\delta^2/2)=\delta[\nabla v_\theta(S)-\gamma\nabla v_\theta(S')].$$

这是不同方法。不能把完整残差梯度称为“补全了缺项的 TD”。两者求解的对象不同。

## 4. 双采样反例：两个最优参数

在被评价的起点 A，预测为 $w$，奖励为 1，γ=1/2。后继特征 X 等概率取 0 或 2，后继预测为 Xw。这个局部目标只评价 A 的条件 Bellman 残差，不声称同时满足所有后继状态的价值方程。

$$\delta(X)=1+(X/2-1)w.$$

$$J_{BE}=\tfrac12(1-w/2)^2,\quad w_{BE}=2;\qquad J_{sample}=\tfrac14[(1-w)^2+1],\quad w_{sample}=1.$$

单样本残差梯度的期望方向为 $1/2-w/2$。真正 MSBE 的负梯度为 $1/2-w/4$。即使采样无穷多次，这个差异仍然存在。

若能从同一个起点条件独立采样 X₁、X₂，则

$$g=\tfrac12[\delta(X_1)(1-X_2/2)+\delta(X_2)(1-X_1/2)]$$

是 MSBE 负梯度的无偏估计。本例两种方法每步都查询两个后继。单样本方法平均两个各自梯度，双采样方法交叉相乘。预算相同，相关性不同。

- [单后继残差梯度](../implementations/nonlinear_diagnostics/single_sample_residual.py)
- [独立双后继残差梯度](../implementations/nonlinear_diagnostics/double_sample_residual.py)

生成模型提供了不可回退单轨迹通常没有的查询能力。从 replay 中随便抽两个转移不满足这个条件。相近起点也不等于相同条件分布。

## 5. Baird 星形诊断：没有神经网络也能失败

六个上状态、一个下状态。目标策略总到下状态。行为策略以 1/7 的概率到下状态，另以 6/7 的概率均匀到上状态，所以行为平稳状态分布均匀。奖励全零，真实价值全零。

八维单位范数特征为

$$x(i)=(2e_i+e_8)/\sqrt5\;(i=1,\ldots,6),\qquad x(7)=(e_7+2e_8)/\sqrt5.$$

期望 TD 每次枚举七个状态：

$$\Delta w=\tfrac\alpha7\sum_s[\gamma x(7)^Tw-x(s)^Tw]x(s).$$

这等于行为数据上的重要性加权期望更新。动作比校正目标动作，但状态权重仍是行为分布。实验没有随机更新噪声。

对照对精确模型的残差目标求梯度：

$$J(w)=\tfrac1{14}\sum_s[\gamma x(7)^Tw-x(s)^Tw]^2,\quad \Delta w=\tfrac\alpha7\sum_s\delta_s[x(s)-\gamma x(7)].$$

- [Baird 期望 TD](../implementations/nonlinear_diagnostics/baird_expected_td.py)
- [精确模型残差梯度](../implementations/nonlinear_diagnostics/baird_residual_gradient.py)

两者每步都是一次包含七状态项的 sweep，不是一次环境步。初始化为约 [1,1,1,1,1,1,10,1]。γ=.99，α=.05。记录原始 RMSE，同时以 log10(1+RMSE) 作图。没有梯度或参数裁剪。

TD 的增长必须保留。残差目标下降也不代表价值误差每步单调下降。γ 接近 1 时，小 Bellman 残差可以对应不小的价值误差。零价值可表示，不能据此断言所有更新都会找到它。

## 6. 非线性稳定性并非完全没有理论

令 $\phi_\theta=\nabla v_\theta$，$b_\theta=\mathbb E[\delta\phi_\theta]$，$C_\theta=\mathbb E[\phi_\theta\phi_\theta^T]$。固定采样条件下，若 C 可逆，可以定义切空间投影目标

$$J(\theta)=\tfrac12 b_\theta^TC_\theta^{-1}b_\theta.$$

令 $u=C_\theta^{-1}b_\theta$。其负梯度包含

$$\mathbb E[(\phi-\gamma\phi')\phi^Tu-(\delta-\phi^Tu)\nabla^2v_\theta(S)u].$$

最后一项来自变化中的梯度特征。线性情形的 Hessian 为零。非线性 GTD 研究使用辅助权重、Hessian-vector product、参数投影和双时间尺度等条件。不能只把线性 x 替换成网络梯度而删去该项，再继承同一保证。

这里的公式针对固定预测、平滑性和相应非退化条件。过参数化网络常有奇异梯度协方差；ReLU 也不是处处二阶光滑。持续控制中策略与数据分布还会改变。这些不是自动满足的技术条件。当前八个实现是诊断，不是完整非线性 GTD 复现。

## 7. 干扰不等于可塑性丧失

若 $g_i=\nabla L_i$，小步长下

$$L_j(\theta-\alpha g_i)-L_j(\theta)=-\alpha\nabla L_j^Tg_i+O(\alpha^2).$$

负梯度内积意味着局部冲突。它不证明共享一定有害。我们构造固定回归目标 A: x=1,y=.5；B: x=2,y=-.5。只改变观测顺序 A→B→A。

共享网络为 $a\tanh(bx+c)+d$，四参数。隔离对照为两个四参数网络，已知上下文负责路由。逐样本 SGD，α=.03，没有 replay。评价探针同时测 A/B，但探针标签不进入更新。

- [共享非线性表示](../implementations/nonlinear_diagnostics/shared_nonlinear.py)
- [隔离非线性表示](../implementations/nonlinear_diagnostics/isolated_nonlinear.py)

记录两个目标的单独误差、梯度内积、当前更新引起的未观测预测漂移。隔离对照多用一倍参数，也依赖预先给定的上下文。不能把它当作公平容量匹配的性能竞赛。

遗忘指旧能力变差。可塑性丧失指控制难度后，新能力越来越难学。表示漂移指同一输入的表示改变，不必然有害。两目标短实验没有验证长期可塑性丧失。

## 8. 停止梯度与目标网络

二状态 A/B 交替，奖励分别 0/1，γ=.8。真值 $v_A=20/9,v_B=25/9$。同一个四参数 tanh 网络做固定策略预测，单步半梯度更新，α=.03。

- [即时目标非线性 TD](../implementations/nonlinear_diagnostics/online_nonlinear_td.py)：每次用当前参数生成下一状态预测。
- [冻结目标非线性 TD](../implementations/nonlinear_diagnostics/lagged_nonlinear_td.py)：另存四个参数，每 50 次更新**之后**复制。第 50 次更新仍用旧目标。

停止梯度只决定当次求导。更新在线参数后，下次重新计算目标仍可能改变。冻结目标把这种变化推迟到复制时刻，但引入滞后与跳变。

代码记录真实价值误差、目标漂移、目标年龄和复制次数。这个小问题中即时目标可以更快。不要为了证明 target network 有用而删去这个结果。小型预测成功也不证明一般神经控制的稳定性。

## 9. 工程技巧改变什么

| 方法 | 直接作用 | 仍需说明 |
|---|---|---|
| Replay | 重用历史、改变采样相关性与权重 | 内存、更新比、旧策略/旧世界的数据偏差 |
| 输入归一化 | 调整数值尺度 | 在线统计使编码随时间改变 |
| 固定正奖励缩放 | 调整价值和梯度尺度 | 在适用目标下可保持策略排序；时变缩放需重分析 |
| 奖励裁剪 | 改变奖励数值关系 | 一般会改变控制目标 |
| 梯度范数裁剪 | 限制一次更新量 | 不保证方向无偏或不缓慢发散 |
| TD 残差裁剪 | 冻结目标下对应稳健损失 | 不是奖励裁剪，也不是梯度范数裁剪 |
| Adam / RMSProp | 按历史梯度预处理 | 额外状态；历史统计可能滞后；不把半梯度变成真梯度 |
| 特征替换 | 注入新的可学习方向 | 旧预测、优化器矩和资格迹如何处理 |

需要报告装置提供了什么能力，而不是把所有改善都归结为网络大或优化器强。

## 10. 资格迹、GVF、option 与持续架构的连接

普通梯度迹为

$$e_t=\gamma_t\lambda_te_{t-1}+\nabla v_{\theta_t}(S_t)=\sum_{k\le t}(\prod_{j=k+1}^t\gamma_j\lambda_j)\nabla v_{\theta_k}(S_k).$$

历史梯度是在历史参数上求得，不是所有历史样本在当前网络上的反向传播。对平滑网络，两者差别近似包含 Hessian 乘参数位移。普通迹有明确的历史信用含义，但不应冒充当前参数序列损失的精确梯度。

固定线性特征消除了这类梯度特征漂移。True-online TD 的特定线性等价性不能直接移植到不断学习的神经编码器。BPTT、RTRL、TD 梯度迹和元梯度也分别回答不同问题。

GVF 改变 cumulant、目标策略与延续条件。预测更新仍需选择。共享编码器会引入多预测目标干扰。不同预测的量纲与时间尺度也影响损失权重。

Option 改变决策持续时间。折扣 SMDP 目标使用累计折扣奖励及 $\gamma^\tau$ 终点价值。平均奖励目标扣除 $\bar r\tau$。改变内部策略、终止规则或编码器会改变旧 option 模型的含义。用旧模型规划会重复放大偏差。

整合智能体时，至少要记录：参数与模型版本、经验年龄、目标副本年龄、技能持续时间、梯度和规划预算，以及替换单元时如何处理关联状态。这些时序约定是算法的一部分，不是实现末尾的细节。

## 11. 运行与验证

在仓库根目录运行。每条命令会自动加入对应 baseline，输出目录必须是新目录。

```bash
python3 implementations/nonlinear_diagnostics/baird_expected_td.py --steps 1200 --seeds 0 1 2 3 4 --out results/baird-diagnostic
python3 implementations/nonlinear_diagnostics/double_sample_residual.py --steps 1200 --seeds 0 1 2 3 4 --out results/double-sampling-diagnostic
python3 implementations/nonlinear_diagnostics/shared_nonlinear.py --steps 1200 --seeds 0 1 2 3 4 --out results/interference-diagnostic
python3 implementations/nonlinear_diagnostics/lagged_nonlinear_td.py --steps 1200 --seeds 0 1 2 3 4 --out results/target-diagnostic
python3 -m unittest discover -s tests -p test_nonlinear_diagnostics.py
```

测试检查真梯度有限差分、重要性加权期望、相关与独立采样的偏差、非线性 Jacobian、干扰的一阶预测、停止梯度与完整导数的不同、目标副本不被原地修改、复制时钟以及发散未被裁剪。通过测试证明这些有限对象的一致性，不证明一般性能。

图的横轴单位必须按实验分别解释。四组任务之间不汇总成同一性能排名。保存所有种子，展示均值和种子间标准差，不能把标准差称为置信区间。

## 12. 原文与作者代码

- [Sutton & Barto，第二版，第 9–12 章](http://incompleteideas.net/book/the-book-2nd.html)：半梯度、函数逼近、离策略及资格迹的共同语言。
- [Baird，Residual Algorithms，ICML 1995](https://leemon.com/papers/1995b.pdf)：残差梯度与反例。本站特征归一化和步长已明确，机制与后续原文及解析测试交叉核对；不宣称逐图复刻该论文。
- [Maei 等，Convergent TD with Arbitrary Smooth Function Approximation，NeurIPS 2009](https://proceedings.neurips.cc/paper/2009/file/3a15c7d0bbe60300a39f76f8a5ba6896-Paper.pdf)：原文式 8–16 的切空间目标和二阶修正。定理条件与工程成功应分开。
- [van Hasselt 等，Deep RL and the Deadly Triad](https://arxiv.org/abs/1812.02648)：研究深度 Q 学习中的相互作用，不是每种深度算法的统一收敛保证。
- [DQN 原始训练器，固定源码版本](https://github.com/google-deepmind/dqn/blob/9d9b1d13a2b491d6ebd4d046740c511c662bbe0f/dqn/NeuralQLearner.lua)：`getQUpdate`、回放采样、目标副本和残差裁剪分开阅读。它是作者工程，本站小实验不是它的复现。
- [Yu 等，Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782)：梯度冲突路线。本站没有将隔离对照称为 PCGrad。
- [Dohare 等，Loss of Plasticity in Deep Continual Learning](https://www.nature.com/articles/s41586-024-07711-7)与[固定版本 Generate-and-Test 作者实现](https://github.com/shibhansh/loss-of-plasticity/blob/a6b79580d85f3025bdb601566d3627c5f489f13b/lop/algos/gnt.py)：效用、成熟期、替换预算与重新初始化。
- [Sokar 等，The Dormant Neuron Phenomenon in Deep RL](https://proceedings.mlr.press/v202/sokar23a.html)：ReDo 与休眠单元的操作性指标。

未在本组实现的内容：完整非线性 GTD/Hessian-vector 工程、完整可塑性长期基准、各种 trace 修正的理论与比较、动态表示下 option 模型版本管理。这些应当作为独立实验加入，而不是通过重命名八个诊断声称已覆盖。
