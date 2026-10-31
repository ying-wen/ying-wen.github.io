# 从读懂论文到提出一个可检验的问题

不做“大一统智能体”，我能从哪里开始前沿研究？

先修：前七课；不要求先读完全部论文

学习目标：产出一页研究计划与可复现的实验目录。

## 问题与例子

候选项目：Forager 中，固定状态维度和每步时间预算，对比短历史堆叠与预测式内部状态。先问收益来自额外信息、额外参数还是学习目标；不要一开始同时换 encoder、optimizer 和 reward。

## 先定位失败，再提出方法

例如“随着干扰特征增加，固定步长的在线预测变差”比“我要实现通用持续智能”更容易开始。写清观察、候选原因和能排除候选原因的实验。先有能稳定重现的失败，再引入新机制。

## 只选择一条主线

喜欢优化：可塑性或自动步长；喜欢序列：状态构造与时间信用；喜欢结构：技能发现与规划；喜欢系统：single-life 或 benchmark；喜欢交互：课程、元学习与多智能体。先沿一条支线深入，两条交叉线放到第二个项目。

## 一篇论文的三遍读法

第一遍写问题、假设、图与结论边界；第二遍还原更新式、伪代码和实验权限；第三遍追源码、配置和随机种子。每遍结束必须产出一个东西：一页解释、一张更新流程图、一条可运行命令。

## 把首个项目拆成六周计划

可先按六周安排：第 1 周完成教程和小实验；第 2 周读两篇核心论文并安装一个官方仓库；第 3 周复现一个最小趋势；第 4 周做失败诊断；第 5 周引入单个改动和强对照；第 6 周写结果、负结果与边界。再根据安装困难、实验方差和可用资源调整进度。

## 进一步思考

你的方法没有超过 baseline，项目是否就失败了？

参考答案：若实验排除了一个有吸引力的解释，或发现改进仅来自某项权限/预算差异，就是有用的研究证据。保存负结果、配置与完整曲线；不要只挑最好种子。

## 原始材料

- [Forager: a lightweight testbed for continual learning with partial observability in RL](https://arxiv.org/abs/2605.01131)
- [Plasticine：可塑性方法实现与指标](https://github.com/RLE-Foundation/Plasticine)
- [Oak Lab：从经验学习，而非从整理好的数据集学习](https://oaklab.ai/posts/learning-from-experience-instead-of-curated-datasets)
- [Adam White — Reinforcement Learning Experiments that Matter!](https://deeprlcourse.github.io/guests/adam_white/)
