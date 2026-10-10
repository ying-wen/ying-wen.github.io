# 不只反应：技能、模型与规划如何积累？

学会“走到门口”后，为什么还需要预测这项技能的结果？

先修：Sutton & Barto Part I 中的 Dyna；没学过可先读第 8 章

学习目标：解释技能、技能模型和规划三者的接口，并指出旧模型风险。

## 问题与例子

仓库中“去充电区”是 option；“执行后 20 步左右抵达、耗电 3 单位”是 option model；在送货和充电之间选择是 planning。货架挪动后，策略、模型、规划中至少有一处要更新，且它们的时间尺度可能不同。

## 动作太短，计划会很长

逐步按键可以完成导航，但远距离计划若每次从原始动作开始，计算成本很高。Option 把一段有目的的行为封装为：何时可启动、内部如何行动、何时结束。它不是单纯把 action repeat 固定为十步。

## 技能会做，不等于规划会用

规划需要知道执行某项技能大致会到哪里、要多久、获得什么回报。因此技能策略和技能模型是不同对象。Dyna 用学得的模型产生规划更新；若模型过时，多规划反而会重复过时知识。

## 技能从哪里来

Machado 的 eigenoptions 用状态空间的谱结构构造探索方向；successor representation 将行为下的未来占用结构编码进表示。Konidaris 强调从底层技能通向可组合的高层表示。发现多样动作不是终点，要测它们是否改善探索、迁移或规划。

## Alberta 与 Oak 的位置

Alberta Plan 将在线预测、状态构造、子任务、技能模型和规划连成研究路线。Oak 延续经验驱动、有限资源下自主发现有用抽象的目标。你可以先验证其中一条接口：新技能出现后，模型预测和规划收益是否改善；无需先实现整个架构。

### 术语小补丁

- **Option**：带启动条件、内部策略和终止规则的一段行为。
- **技能模型**：预测执行技能后的状态、累计奖励或持续时间。

### 推导：为什么技能的 Bellman 目标要用 γ 的 τ 次方？

一项技能跨越多个基本时间步，不能一律当作一步动作。

符号：o 是 option；τ 是它执行到终止的步数；$S_{t+τ}$ 是终止状态。Q 表示给定技能集合上的最优技能价值。假设折扣与期望良好定义，技能会终止。

先把执行期间的奖励按原始时间尺度相加。

$$
G_t^{(o)}=\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}
$$

到技能终止后再作选择，后续价值已经过去了 τ 步。

$$
Q(s,o)=\mathbb{E}\!\left[G_t^{(o)}+\gamma^\tau\max_{o'\in\mathcal O(S_{t+\tau})}Q(S_{t+\tau},o')\mid S_t=s,o\right]
$$

例子：γ=0.9，技能执行 2 步，两步奖励均为 1，终止状态后续价值为 10。目标为 $1+0.9+0.9^2×10=10$；若错误地只折扣一次，会得到 10.9。

适用条件：写出技能 Bellman 关系，不等于已经学会发现技能或准确预测技能后果。真实环境变化时，需要同时检查技能策略和模型。

[配套来源：Options 原论文入口](https://www.sciencedirect.com/science/article/pii/S0004370299000521)

### 理解与操作练习

问题：“走到门口”和“预测走到门口花多少步”是同一个模型吗？

提示：一个产生动作，一个预测行为后果。

参考答案：前者是技能策略，后者属于技能模型。规划需要模型信息，但会执行技能不意味着已经知道这些后果。

问题：技能耗时从 2 步变成 5 步，规划器只更新终点、不更新耗时，会遗漏什么？

提示：时间影响折扣，也可能影响运行成本。

参考答案：后续价值应以 $γ^5$ 折扣，并重新计算执行期间的奖励或代价。相同终点不代表两个技能同样有价值。

## 进一步思考

一个新 option 提高了训练回报，如何判断它真的帮到了规划？

参考答案：固定数据与探索预算，对比有无技能模型、有无规划，并报告规划计算量、模型误差和未见布局表现。若收益只来自更好的探索，仍有价值，但不是相同机制。

## 原始材料

- [Machado options：从 main.py 读懂 eigenoptions](https://github.com/mcmachado/options)
- [Marlos C. Machado — Representation-driven Option Discovery in Reinforcement Learning](https://deeprlcourse.github.io/guests/marlos_machado/)
- [George Konidaris — Resolving the Sensorimotor Dilemma](https://www.youtube.com/watch?v=Af5UFE7CdKs)
- [The Alberta Plan for AI Research — Sutton、Bowling、Pilarski，2022](https://arxiv.org/abs/2208.11173v1)
- [ALPS：Laplacian 表示与决策时规划](https://arxiv.org/abs/2602.05031)
- [Laplacian Keyboard](https://arxiv.org/abs/2602.07730)
- [The OaK Architecture](https://oaklab.ai/posts/the-oak-architecture)
