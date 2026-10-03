# 只用最新一条经验，怎样稳定地学？

机器人每 20 毫秒必须行动，还能等大 batch 训练吗？

先修：TD 误差；可按需补 Sutton & Barto 第 9、12、13 章

学习目标：能画出更新顺序，区分缓冲区、trace 与模型状态。

## 问题与例子

把一步切成 act → environment → reward/observation → state update → critic/actor update。记录每段耗时。一个回报更高却经常超过动作 deadline 的算法，在真实时间协议下可能不可用。

## 先对齐更新协议

DQN 常用 replay 和 target network；严格 streaming 路线要研究没有这些设施时怎么稳定更新。不要只把 batch_size 改成 1 就称为复现。归一化、步长、actor/critic 耦合、梯度截断和初始化都可能影响稳定性。

## trace 是压缩的责任记录

Eligibility trace 可记成 e ← γλe + 当前特征，再用 TD 误差 δ 更新 θ ← θ + αδe。它让今天的误差影响先前参与预测的特征，不需要重放全部轨迹。这里是线性预测的直观入口；带函数逼近、策略梯度或 off-policy 时还需检查各自公式。

## 不同参数不一定该同速学习

线性 IDBD 为各特征自适应步长，用过去更新的方向信息调节未来更新幅度。先在有用特征加噪声特征的预测任务上理解它，再去看 Oak 的 selective credit 思路。线性 IDBD 不等于 NetworkIDBD，也不是从监督预测到深度控制的完整推导。

## 从教程进真实实现

先读 Stream-X 的一条 transition 如何更新，再固定官方分支 2026 与一个 commit 运行小任务。RTU / Streaming RTRL 则针对递归状态的实时信用分配；其效率来自结构限制，不能概括成所有 RNN 的精确 RTRL 都是线性的。

### 术语小补丁

- **TD 误差 δ**：一步奖励加上下一个状态的估值，与当前估值之差。
- **资格迹 e**：压缩过去特征对当前更新的影响，不是保存原始样本。
- **半梯度 semi-gradient**：更新当前预测时，暂时把自举目标视为常数。

### 推导：把 TD(λ) 的一步更新拆开

先会手算线性预测，再阅读复杂的 streaming actor–critic。

符号：$x_t$ 是状态特征向量；w 是权重；γ 是折扣，λ 控制迹的衰减。以下为 on-policy 线性预测的 accumulating traces。

用线性函数预测价值，并构造一步自举目标。

$$
\hat v_w(S_t)=w^\top x_t,\qquad \delta_t=R_{t+1}+\gamma w_t^\top x_{t+1}-w_t^\top x_t
$$

若先不使用迹，对平方 TD 误差作半梯度更新，就得到当前特征的更新方向。

$$
w_{t+1}=w_t+\alpha\delta_t x_t
$$

让过去的特征也承担责任：先更新迹，再更新权重。

$$
e_t=\gamma\lambda e_{t-1}+x_t,\qquad w_{t+1}=w_t+\alpha\delta_t e_t
$$

将迹展开，看到被压缩的历史。

$$
e_t=\sum_{k=0}^{t}(\gamma\lambda)^{t-k}x_k\quad(e_{-1}=0)
$$

例子：α=0.1、δ=2、γλ=0.8，上一条迹 e=(1,0)，当前特征 x=(0,1)。新迹为 (0.8,1)，权重增加 (0.16,0.2)：先前特征也收到更新。

适用条件：这不是任意 off-policy 或非线性控制算法的稳定性保证。终止转移的后继价值为 0，episodic 协议还要按定义清空迹；continuing 协议不可偷偷按固定窗口清空。

[配套来源：Sutton & Barto · 第二版，第 2、6、12 章](http://incompleteideas.net/book/the-book-2nd.html)

### 理解与操作练习

问题：迹保存过去影响，为什么不等于 replay buffer？

提示：一个保存向量，另一个保存可再次训练的样本。

参考答案：迹是随每步衰减和累积的责任摘要，不能从它还原并重新训练某条历史 transition；replay 则明确重用保留下来的样本。

问题：在数值例子中令 λ=0，更新会怎样改变？

提示：此时历史迹不再保留。

参考答案：新迹等于当前特征 (0,1)，权重只增加 (0,0.2)，退回一步 TD 更新。

## 进一步思考

Eligibility traces、RNN hidden state、replay buffer 都算记忆，为什么要分别报告？

参考答案：它们保存的对象和使用方式不同：trace 压缩信用历史，hidden state 汇总观测，replay 保存可再次训练的样本。三者都占资源，但只有后者必然涉及样本重用；具体算法还可能同时使用它们。

## 原始材料

- [stream-x / Streaming Deep RL](https://github.com/mohmdelsayed/streaming-drl)
- [RTU](https://github.com/esraaelelimy/rtus)
- [SwiftTD / Average-Reward Methods](https://github.com/kjaved0/swifttd)
- [A. Rupam Mahmood — Streaming Deep Reinforcement Learning](https://www.youtube.com/watch?v=QOfkOl9QrZY)
- [Intentional Updates](https://arxiv.org/abs/2604.19033)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://arxiv.org/abs/2605.24709)
- [Oak Lab：从经验学习，而非从整理好的数据集学习](https://oaklab.ai/posts/learning-from-experience-instead-of-curated-datasets)
