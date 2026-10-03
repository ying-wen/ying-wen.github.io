# 怎样证明“持续学习变好了”？

最后 100 步分数高，就代表一个生命期学得好吗？

先修：均值、方差与训练/测试分离

学习目标：写出可审计的实验协议，选择与问题匹配的 benchmark。

## 问题与例子

方法 A 最终 90 分但切换后损失 1000 步；方法 B 最终 85 分但 100 步恢复。哪个更好取决于任务目标、生命周期长度和安全代价。先定义要优化的量，再挑展示指标。

## 先选问题，不先选排行榜

保留旧任务可选任务序列基准；状态记忆可选 POPGym / Forager；无自然终点与复杂交互可看 AgarCL；多智能体课程变化可看 MEAL。一个基准不会自动测出所有能力。OGBench 的离线任务成绩也不能直接当成在线 CRL 成绩。

## 记录整段生命史

至少记录变化前后曲线、旧任务保留、新任务学习速度、总体回报、峰值内存与实际运行时间。奖励量纲不同，跨任务聚合需要预先确定归一化。只有最后一段平均回报，可能把早期大规模失败和长期遗忘隐藏起来。

## 对照不要偷看未来

随机种子分为调参用和报告用；同一方法不得用测试任务选择最佳配置。变化时间是否已知、是否给 task ID、每阶段是否重置 optimizer 都应列入协议。可先用 5 个种子排错，正式数量由方差、资源与精度要求决定，而非固定神奇数字。

## 小规模复现的准确说法

成功 import 包是安装验证；跑 1000 步不崩是 smoke test；复现趋势是缩小实验；匹配论文配置与统计结果才接近论文复现。本站教学实验只验证实现和机制，不宣称复现了完整 CRL 论文。

### 术语小补丁

- **学习曲线**：随交互或更新推进，持续记录的表现。
- **独立运行**：用不同 seed 重复完整学习过程；重叠时间窗不能替代独立运行。

### 推导：从一条曲线到可比较的数字

先明确每个点的含义，再谈平均和误差条。

符号：W 为窗口长度；μₜ(a) 是已知教学环境的真实期望奖励；m 是独立运行编号。

滑动或不重叠窗口都可以，但必须注明是哪一种。本站图采用不重叠窗口。

$$
\bar R_t=\frac{1}{W}\sum_{i=t-W+1}^{t}R_i
$$

在这个已知两臂均值的实验里，可以按实际动作计算相对当时最好动作的机会损失。

$$
\mathcal R_T=\sum_{t=1}^{T}\big[\max_a\mu_t(a)-\mu_t(A_t)\big]
$$

独立运行的均值与样本标准差是两种不同摘要。

$$
\bar J=\frac1M\sum_{m=1}^{M}J_m,\qquad s=\sqrt{\frac{\sum_{m=1}^{M}(J_m-\bar J)^2}{M-1}}
$$

例子：切换后选错动作一次，期望机会损失为 0.8−0.2=0.6；不是用随机的 0/1 奖励直接相减。20 个 seed 的标准差表示运行波动，不等于均值的置信区间。

适用条件：真实控制任务通常不知道最优策略或真实均值，不能照搬这个 bandit regret。单次运行中相邻窗口相关；不把它们冒充独立样本。



### 理解与操作练习

问题：最终分数相同，变化后恢复时间不同，两种方法是否等价？

提示：把恢复过程中损失的奖励也算进去。

参考答案：不等价。整段累计收益、短期损失和稳定后表现可能不同，要先选与应用目标一致的指标。

问题：先运行 1 个 seed，再运行 20 个 seed，报告中多出来的信息是什么？

提示：看每个 seed 的结果和跨 seed 波动，而不只是更平滑的线。

参考答案：可以检查趋势是否稳定、是否被少数运行主导。样本标准差反映运行差异；它不会自动排除调参泄漏或协议不公平。

## 进一步思考

三个种子的平均曲线高一点，能否称为 state of the art？

参考答案：不能据此下结论。先排除调参预算、环境版本、种子选择与协议差异，给出跨任务和不确定性分析。若证据不足，准确表述“在这个小规模设置下观察到差异”。

## 原始材料

- [Adam White — Reinforcement Learning Experiments that Matter!](https://deeprlcourse.github.io/guests/adam_white/)
- [Michael Littman — Assessing the Robustness of Deep RL Algorithms](https://deeprlcourse.github.io/guests/michael_littman/)
- [Forager](https://github.com/andnp/forager)
- [AgarCL](https://github.com/AgarCL/AgarCL)
- [Continual World（CW10／CW20）](https://arxiv.org/abs/2105.10919)
- [COOM（Continual DOOM）](https://github.com/TTomilin/COOM)
- [MEAL](https://github.com/TTomilin/MEAL)
- [rliable](https://github.com/google-research/rliable)
