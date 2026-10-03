# 策略梯度、Actor–Critic 与 PPO

策略梯度直接优化参数化策略的期望回报。REINFORCE 使用采样回报估计梯度；actor–critic 引入价值预测以分配信用；PPO 在旧策略采集的数据上优化一个局部代理目标。

## 本章内容

- 从轨迹概率推出策略梯度与 baseline 的零期望。
- 由 TD error 递推计算 GAE，正确处理终止、截断和 rollout 边界。
- 手算 PPO 的正负 advantage 裁剪，并运行有真实采样循环的最小实现。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 一条经验

在 $S_{t}$ 做 $A_{t}$ 后得到 $R_{t+1}$、$S_{t+1}$。奖励属于这次转移，不是到达状态之前另一轮的奖励。

### 折扣与终止

$\gamma$ 控制未来相对权重；真实终止后的价值设为 0。本章记 $\gamma_{t+1}$=$\gamma$(1−terminated)。训练脚本的时间截断不一定是任务终止。

### 表格与函数逼近

表格每个状态或状态动作一组独立参数；函数逼近让不同输入共享参数。更新一个输入可能改变其他输入的预测。

### 随机策略

$\pi_\theta$(a|s) 是动作概率。二动作可用 p=$\sigma$($\theta$)，softmax 推广到多动作；连续动作通常是参数化概率密度。

### 梯度上升与损失下降

要最大化回报 J，就沿 ∇J 上升；框架常定义 loss=−代理目标再下降，符号必须一致。

<a id="lesson-setting"></a>

## 1 · 先用有限时域把梯度推清楚

设轨迹 $\tau$=($S_{0}$,$A_{0}$,$R_{1}$,…,$S_T$)，环境动力学和初始状态分布不依赖策略参数 $\theta$。先取有限时域不折扣总回报。策略产生数据，因此 $\theta$ 改变的不只是预测，还会改变访问哪些状态与轨迹。网络 loss 的下降不是环境回报上升的直接证据。

$$
J(\theta)=\mathbb E_{\tau\sim p_\theta}[\sum_{t=0}^{T-1}R_{t+1}],\qquad p_\theta(\tau)=p_0(s_0)\prod_t\pi_\theta(a_t\mid s_t)p(s_{t+1},r_{t+1}\mid s_t,a_t)
$$

目标是对自己诱导的数据分布求期望。接下来对该分布求导，而不是把采到的奖励当作可对 $\theta$ 直接反向传播的环境函数。

<a id="lesson-derive"></a>

## 2 · 对数导数、因果性与 baseline

$$
\nabla J=\sum_\tau p_\theta(\tau)R(\tau)\nabla\log p_\theta(\tau)=\mathbb E[\sum_t\nabla\log\pi_\theta(A_t\mid S_t)R(\tau)]
$$

使用 ∇p=p∇log p；环境项与 $\theta$ 无关，只有策略项留下。连续空间将求和换积分，还需要可交换求导与积分等正则条件。

$$
\nabla J=\mathbb E[\sum_t\nabla\log\pi_\theta(A_t\mid S_t)G_t],\qquad G_t=\sum_{k=t}^{T-1}R_{k+1}
$$

当前动作不能影响之前已经发生的奖励。条件在历史上，策略 score 的期望为零，因此可从权重中删掉过去奖励而保持同一期望梯度。

$$
\sum_a\pi_\theta(a\mid s)\nabla\log\pi_\theta(a\mid s)b(s)=b(s)\nabla\sum_a\pi_\theta(a\mid s)=0
$$

任意不依赖当前动作的 baseline 都不改变期望。取 $b=V_\pi$，权重成为 advantage；若 baseline 共享参数，actor 更新中须将 baseline 当作固定权重，不额外沿它求导。

$$
A_\pi(s,a)=Q_\pi(s,a)-V_\pi(s),\qquad \delta_t=R_{t+1}+\gamma_{t+1}V(S_{t+1})-V(S_t)
$$

真实 V 与 on-policy 数据下，TD residual 的条件期望给 advantage。使用近似 V 会引入估计误差；actor–critic 就是在用较低方差、可能有偏的 critic 信号替代完整回报。

若 J 是从固定起点的严格折扣目标，轨迹求和形式中需要相应 $\gamma$^t 权重，或等价地在折扣占用分布下取样。很多实现直接对 rollout 的每个时刻均匀平均，再使用 $\gamma$ 折扣的 critic；这是常见代理方式，不能不说明就称与任意 J 完全同一梯度。

<a id="policy-gae"></a>

## 3 · GAE 把多步 advantage 写成反向递推

$$
\hat A_t=\delta_t+\gamma_{t+1}\lambda b_t\hat A_{t+1},\qquad \hat R_t=\hat A_t+V_{\mathrm{old}}(S_t)
$$

$b_{t}$ 表示后继样本是否仍在同一条轨迹内；真实终止、reset 或采样块边界时置 0，避免把新轨迹的误差接回来。它与控制 bootstrap 的 $\gamma_{t+1}$ 分开。从末尾向前算即可。$\lambda$=0 只用一步 TD，$\lambda$ 接近 1 更多依赖真实后续奖励；终止后 tail 为零。value target 是优势加旧 value，不要把更新中的新 value 混进去。

三个边界必须拆开：真实终止：bootstrap=0 且 trace 停；time-limit 截断：若原任务继续，bootstrap 使用最终观测 value，但 trace 不跨进 reset 的新轨迹；rollout 采样块结束：bootstrap 使用块尾状态 value，未采到的 GAE tail 截断为 0。这些不是一个 done 布尔变量能无歧义表达的。

训练 actor 前，advantage、旧 log-prob 和 critic target 都冻结。本章不强制标准化 advantage；若采用标准化，负正号与尺度可能改变，必须在实现说明中写清。

<a id="policy-ppo"></a>

## 4 · 从重要性比率到 PPO 的单样本分段函数

$$
r_t(\theta)=\exp[\log\pi_\theta(A_t\mid S_t)-\log\pi_{\mathrm{old}}(A_t\mid S_t)],\qquad L^{\mathrm{clip}}=\mathbb E_t[\min(r_t\hat A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t)]
$$

比率修正旧状态上的动作分布；它不是完整轨迹重要性采样，也没修正当前策略诱导的新状态分布。PPO 因此是局部代理优化，不能把每次 loss 降低解释成严格策略改善。

| 优势符号 | 停止继续奖励的区间 | 仍然保留梯度的情形 |
| --- | --- | --- |
| A>0 | r>1+$\epsilon$：已经增加够多 | r 太低时继续鼓励恢复 |
| A<0 | r<1−$\epsilon$：已经减少够多 | r 太高时继续惩罚错误增加 |

注意 min 的方向：负 advantage 乘以较大的 ratio 更负。因此不能先把 ratio 双向裁剪再简单乘 A；那会在应该纠错的方向也抹掉梯度。裁剪也不是硬 KL 约束，整个网络共享参数仍可能改变其他动作，通常还要监测 KL 并限制优化轮数。

**算法：PPO 完整循环；`old_logprob` 在 K 轮里始终不变**

1. 给定策略参数 $\theta$、价值参数 $\phi$、裁剪范围 $\epsilon$ 与优化轮数 $K$。
1. 每轮采样：
  1. 固定行为策略 $\pi_{\rm old}$，采集轨迹，保存动作对数概率和价值预测。
  1. 根据真实终止与时间截断计算 GAE，固定 $\widehat A_t$ 与价值目标 $\widehat V_t$。
  1. 重复 $K$ 轮，在轨迹中抽取小批量样本：
    1. $r_t\leftarrow\exp[\log\pi_\theta(A_t\mid S_t)-\log\pi_{\rm old}(A_t\mid S_t)]$
    1. $L_\pi\leftarrow-\operatorname{mean}\{\min[r_t\widehat A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\widehat A_t]\}$
    1. $L_V\leftarrow\operatorname{mean}[(V_\phi(S_t)-\widehat V_t)^2]$
    1. 按给定权重优化策略、价值和熵正则项。
    1. 根据预先设定的 KL 阈值决定是否提前停止本轮优化。
  1. 用更新后的策略采集下一轮轨迹。

<a id="lesson-example"></a>

## 5 · PPO 裁剪与 GAE 的数值例子

旧策略选择动作 1 的概率 .5，新概率 .75，ratio=1.5，$\epsilon$=.2。若 A=2，两个候选值为 3 和 2.4，取 2.4，该样本正向改善梯度已为 0；若 A=−2，则取 min(−3,−2.4)=−3，仍推动降低该动作概率。

两步奖励 [0,1]，旧 V=[.2,.5]，真实终止后的 value=0，$\gamma$=.9、$\lambda$=.8。$\delta_{1}$=1−.5=.5，$\delta_{0}$=0+.9×.5−.2=.25；$A_{1}$=.5，$A_{0}$=.25+.72×.5=.61。critic targets 为 [.81,1]。这和把 value target 简单设成 reward 不同。

<a id="lesson-code"></a>

## 6 · 核心实现：裁剪、边界递推与采样训练

二动作 Bernoulli PPO 与独立 GAE 函数

```python
def sigmoid(theta):
    if theta >= 0:
        return 1.0 / (1.0 + math.exp(-theta))
    exp_theta = math.exp(theta)
    return exp_theta / (1.0 + exp_theta)


def ppo_sample(theta, old_probability, action, advantage, epsilon=0.2):
    p = sigmoid(theta)
    probability = p if action == 1 else 1.0 - p
    ratio = probability / old_probability
    clipped = min(1.0 + epsilon, max(1.0 - epsilon, ratio))
    objective = min(ratio * advantage, clipped * advantage)
    inactive = (advantage > 0 and ratio > 1.0 + epsilon) or (advantage < 0 and ratio < 1.0 - epsilon)
    derivative = 0.0 if inactive else advantage * ratio * (action - p)
    return objective, derivative


def gae(rewards, values, next_values, discounts, trace_continues, lam=0.95):
    # At time-limit truncation, bootstrap may survive but the trace must not
    # jump into the next reset episode. next_values uses FINAL observation.
    out, tail = [0.0] * len(rewards), 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + discounts[t] * next_values[t] - values[t]
        tail = delta + discounts[t] * lam * trace_continues[t] * tail
        out[t] = tail
    return out


def train_ppo_bandit(rounds=100, seed=7):
    rng, theta = random.Random(seed), 0.0
    for _ in range(rounds):
        old_p = sigmoid(theta)
        batch = []
        for _ in range(64):
            action = int(rng.random() < old_p)
            # Exact old-policy baseline for this reward function.
            batch.append((old_p if action else 1.0 - old_p, action, action - old_p))
        for _ in range(4):
            grad = sum(ppo_sample(theta, *sample)[1] for sample in batch) / len(batch)
            theta += 0.1 * grad
    return {"probability_good_action": sigmoid(theta)}
```

最小 bandit 动作 1 得 1、动作 0 得 0，真实 $V_{\rm old}=p_{\rm old}$，因此 baseline 不需要另训练一个 critic。它隔离 PPO 的采样、旧概率冻结、多个 epoch 与裁剪机制；多步 actor–critic 的 GAE 是独立函数，并测试 time-limit 是否错误串到新回合。

运行后好动作概率应从 .5 上升。测试用中心有限差分核对裁剪之外及平坦区的导数，另检查正负 advantage 的不对称。这个 bandit 实验隔离策略裁剪的作用；多步控制还需要价值回归、轨迹边界处理和环境交互。

<a id="lesson-branches"></a>

## 7 · 为什么这些不是一条“后者替代前者”的序列

| 方法 | 主要改变 | 关键边界 |
| --- | --- | --- |
| REINFORCE | 完整 return 的 Monte Carlo 梯度 | 方差、等待长度 |
| actor–critic / A2C / A3C | 用 critic 与自举优势 | critic 误差、同步方式 |
| GAE | 优势估计器的多步组合 | 不是独立控制算法 |
| TRPO | 约束或近似约束策略改变 | 二阶近似、求解成本 |
| PPO | 裁剪或惩罚局部代理目标 | 不提供任意更新的硬信赖域 |
| streaming actor–critic | 每条新经验立即更新 | 尺度、迹、近似梯度时序 |
| SAC | 离策略熵正则控制 | 目标与数据协议都改变 |

CRL 中固定 rollout 收集长短、更新 epoch 数和每步延迟可能决定适应速度。增加 epochs 得到更低旧数据 loss，却可能更不适应变化后的分布；同时报告环境互动与优化计算。

<a id="lesson-check"></a>

## 8 · 习题与讨论

为何 `actor_loss` 变低不代表回报更高？这是固定旧数据上的局部代理；策略更新后数据分布改变，过多优化可能损害未见状态。必须实际评测交互收益和变化后表现。

为何必须保存 `old_logprob`？若每个 epoch 都用新网络重算分母，ratio 会不断回到 1，裁剪失去对原采样策略的参照。只保存动作、但不保存对应旧分布，是常见的隐蔽错误。

## 本章的实验设计

梯度检查使用固定轨迹。性能评价则必须重新采样。两种实验不能互相替代。

设定：固定短 rollout，分别给正负优势并跨越 PPO 裁剪边界；然后在可枚举小策略上对期望目标核导数。

- ratio=1.4、A=2、clip=0.2 的最大化目标为 2.4。
- ratio=1.4、A=−2 时目标为 −2.8，不能统一先裁剪 ratio。
- 外部截断可以 bootstrap，但不得把新 episode 的 GAE 接回旧轨迹。

对照：REINFORCE 与带基线估计器；固定样本和真实重采样分开；相同旧策略数据与明确 epoch 制度

记录：概率比、clip fraction、KL 与熵；固定样本梯度误差；独立交互收益及 rollout/更新成本

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-deep)

## 学习与研究衔接

从 REINFORCE 到 actor–critic，评论家提供低方差学习信号。PPO 的批量多轮更新与严格流式协议不同。

[分册导读](https://yingwen.io/zh/continual-rl/start/deep-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-policy) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=policy) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=policy)

<a id="chapter-code"></a>

## 下载与运行

Python 3.10+，仅标准库。执行 PPO bandit 完整采样/优化循环，独立实现多步 GAE；不声称包含网络 PPO 的通用训练工程。

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

```sh
python3 foundations_detail_lab.py policy
python3 foundations_detail_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Williams · Simple statistical gradient-following algorithms](https://doi.org/10.1007/BF00992696)：REINFORCE 的原始梯度估计思想。把 score function 与数据分布的关系读清楚。

- [Schulman et al. · Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)：区分 $\gamma$ 带来的目标/估计变化和 $\lambda$ 引入的优势估计权衡。

- [Schulman et al. · Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)：逐项对照 surrogate、clipping、训练轮数与采样协议。

- [Spinning Up · Policy optimization 推导与代码](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)：以轨迹概率、log derivative 和 baseline 连起数学与 PyTorch。

- [OpenAI Spinning Up · PPO PyTorch 作者教学实现](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py)：追 `buffer.finish_path`、`compute_loss_pi`、update 三部分；工程的旧环境 API 需单独适配。
