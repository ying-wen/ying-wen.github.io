# 分清三种失败：遗忘、学不进、没看懂

一个训练很久的网络表现差，究竟哪里坏了？

先修：了解神经网络训练与 train/test

学习目标：能设计 old / new / fresh 三组对照，不把 dormant neuron 数量当结论。

## 问题与例子

评测矩阵的行是“训练到哪个阶段”，列是“测哪个任务”。A→B 后 A 的单元格下降说明保留变差；另画老网络与 fresh 网络在同一新 C 上的学习曲线，才能讨论学新东西的速度。仅有矩阵的最后一行还不够。

## 遗忘看过去，可塑性看未来

在 A 上学好，学 B 后 A 的表现下降，是遗忘的证据。训练过很多任务后，给一个新任务 C，老网络比相同结构的新初始化网络学得更慢，才是研究学习能力下降的一个入口。二者可以同时发生，也可以分别发生。

## 第三种：表示里根本没有答案

若当前图像无法区分两个需要不同动作的情境，换优化器或重置神经元也不能凭空获得信息。需要历史、记忆或更合适的预测状态。因此诊断顺序是：任务是否可解、信息是否足够、更新是否稳定，最后才讨论哪种可塑性机制。

## 三类修补各解决什么

回放把旧经验带回来；EWC 类方法限制重要参数改变；ReDo 与 Continual Backprop 类方法更新部分失效或低效用特征。保护太强会妨碍新学习，替换太多会破坏旧知识。它们不是互相排斥的标签，应在同一预算下比较具体机制。

## 怎样读第一篇深度论文

先读 Loss of Plasticity 的问题、实验流程和主要图，再看低效用单元如何被选择与替换。暂不追逐全部超参数。记下：任务分布怎样变化、fresh baseline 怎么初始化、更新次数是否相同。激活率、梯度范数、特征秩是诊断信号，不等于能力本身。

### 术语小补丁

- **遗忘 forgetting**：学习新内容后，在同一个旧任务上表现变差。
- **可塑性 plasticity**：学习新内容的能力；不能仅由当前分数或神经元活跃度判断。
- **fresh baseline**：相同结构、重新初始化的学习器，用于比较在同一新任务上的学习过程。

### 推导：把“忘了”和“学得慢”分成两个测量

它们可以同时出现，但不是同一个数字。

符号：Sᵢⱼ 是学完阶段 i 后在任务 j 的分数，越高越好；L(k) 是进入同一新任务后第 k 次更新的损失，越低越好。

先用一个允许回访旧任务的两阶段协议，测旧任务 A 的下降。

$$
F_A=S_{A,A}-S_{B,A}
$$

新任务的整段损失差可以作诊断，但还受初始难度与先验知识影响。

$$
\Delta_K=\frac{1}{K}\sum_{k=1}^{K}\left[L_{\mathrm{old}}(k)-L_{\mathrm{fresh}}(k)\right]
$$

用二次损失看一个容易误判的例子。目标是 y，梯度更新为：

$$
L_y(w)=\frac12(w-y)^2,\quad w_{k+1}=w_k-\alpha(w_k-y)
$$

移项后，每一步到目标的距离都缩小相同倍数；平方误差也按固定比例缩小。

$$
w_{k+1}-y=(1-\alpha)(w_k-y),\qquad \frac{E_{k+1}}{E_k}=(1-\alpha)^2
$$

为什么学 B 可能伤害 A？在固定两组数据的可微损失下，做一次小步更新，用一阶展开看梯度方向。

$$
L_A(\theta-\alpha g_B)-L_A(\theta)\approx-\alpha\, g_A^\top g_B,\quad g_j=\nabla L_j(\theta)
$$

例子：先学目标 +1，再换成 −1。旧参数约 +1，fresh 参数为 0；它们到新目标的初始平方误差是 4 和 1，但收缩率完全相同。老参数暂时误差更大，不足以证明学习能力受损。

适用条件：这些是教学诊断，不是可塑性的通用定义。一阶干扰式忽略二阶项，且 RL 的数据分布也会随策略改变；旧任务若不能回访，$F_A$ 也不能直接测。

[配套来源：Dohare 等 · Loss of Plasticity](https://www.nature.com/articles/s41586-024-07711-7)

### 理解与操作练习

问题：学习 B 后，A 的分数从 80 降到 50。这说明了什么，还没说明什么？

提示：这里测到了旧任务，没有测新的 C。

参考答案：说明这个评测协议下 A 的表现下降了 30 分；还没有证明 agent 学 C 变慢，需要在同一 C 上比较学习过程。

问题：看本课的线性例子：老参数在新任务上误差更大，但归一化误差曲线重合。你会怎样表述？

提示：比较误差 E(k)/E(0)，而不只看绝对误差。

参考答案：旧目标发生遗忘；在这个二次损失例子里，两种初始化具有相同的误差收缩速度。它用于说明测量不能混淆，不能代替深度网络的可塑性实验。

## 进一步思考

如果重置一半网络后新任务更好，能否说“解决了可塑性损失”？

参考答案：还不能。需要匹配随机替换对照、学习率搜索和算力，并测旧任务、多个后续任务及长生命史。重置带来的短暂优化优势不自动意味着长期知识积累。

## 原始材料

- [Loss of Plasticity in Deep Continual Learning — Dohare 等，Nature 2024](https://www.nature.com/articles/s41586-024-07711-7)
- [Understanding Plasticity in Neural Networks — Lyle 等，ICML 2023](https://proceedings.mlr.press/v202/lyle23b.html)
- [The Dormant Neuron Phenomenon in Deep Reinforcement Learning — Sokar 等，ICML 2023（ReDo）](https://proceedings.mlr.press/v202/sokar23a.html)
- [The Primacy Bias in Deep Reinforcement Learning — Nikishin 等，ICML 2022](https://proceedings.mlr.press/v162/nikishin22a.html)
- [Continual Backprop / Loss of Plasticity](https://github.com/shibhansh/loss-of-plasticity)
- [Plasticine](https://arxiv.org/abs/2504.17490)
- [Richard S. Sutton — Generate-and-test Methods for Continual Learning](https://www.youtube.com/watch?v=NaJxvGV8MRg)
- [Clare Lyle — Why Do Neural Networks Lose Plasticity?（CoLLAs 2023）](https://www.youtube.com/watch?v=1EwKYesnKAA)
