# 价值预测与时间差分学习

价值预测估计给定策略的期望回报。回报的定义确定预测对象，MC、TD 和多步方法则提供不同的估计方式。本章从回报递推式推导这些方法，并讨论资格迹怎样分配时间上的信用。

## 本章内容

- 从 return 逐行推到 Bellman 方程，辨认模型期望和经验采样。
- 独立实现 MC 与 TD，解释为什么同一条轨迹给出的第一次更新不同。
- 理解 $\lambda$ 在传播信用中做什么，以及为什么普通在线 TD($\lambda$) 与固定参数前向视图不能无条件画等号。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 一条经验

在 $S_{t}$ 做 $A_{t}$ 后得到 $R_{t+1}$、$S_{t+1}$。奖励属于这次转移，不是到达状态之前另一轮的奖励。

### 折扣与终止

$\gamma$ 控制未来相对权重；真实终止后的价值设为 0。本章记 $\gamma_{t+1}$=$\gamma$(1−terminated)。训练脚本的时间截断不一定是任务终止。

### 表格与函数逼近

表格每个状态或状态动作一组独立参数；函数逼近让不同输入共享参数。更新一个输入可能改变其他输入的预测。

<a id="lesson-setting"></a>

## 1 · 给定策略的预测问题

本章固定策略 $\pi(a\mid s)$，估计按它持续行动的后果，不改变动作选择规则。假设状态与动作有限、转移规律固定、奖励有界且 $0\le\gamma<1$。数据由目标策略产生。若当前观测不足以条件化未来，需要先引入历史或适当的状态表示；不能直接将观测当作 Markov 状态。

价值预测与控制使用相同的回报定义，但求解的对象不同。预测以策略为输入，以价值为答案；控制根据价值或其他后果估计改善策略。先独立理解预测，可以分清一个控制算法的问题来自评价误差、决策规则，还是收集到的数据。通用价值函数则进一步扩展预测的信号和延续规则，仍不自动承担策略改善。

$$
G_t=R_{t+1}+\gamma_{t+1}G_{t+1},\qquad v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s]
$$

$G_t$ 是一次实现的随机回报；$v_\pi(s)$ 是同一起点、同一策略下回报的条件期望。真实终止令后继延续因子为零，只删掉未来项，不删掉进入终点时的奖励。

算法维护估计 $V$；真实价值通常未知。解析小 MDP 可以直接计算价值误差；一般任务可用独立采样回报进行评估。训练中的 TD error 与价值估计误差是不同的量。

<a id="lesson-derive"></a>

## 2 · 从回报拆分到 Bellman，再到三种更新

$$
v_\pi(s)=\sum_a\pi(a\mid s)\sum_{s\prime,r}p(s\prime,r\mid s,a)[r+\gamma(s,a,s\prime)v_\pi(s\prime)]
$$

第一步把 G 拆成当前奖励和未来；第二步对动作与转移取条件期望；Markov 性允许用下一状态而非完整历史描述后半段。这里尚未提出任何学习算法。

$$
V_{k+1}(s)=(T_\pi V_k)(s),\qquad \|T_\pi u-T_\pi v\|_\infty\le\gamma\|u-v\|_\infty
$$

已知模型时，可以对所有后果求和。这是动态规划的期望备份；$\gamma$<1 带来压缩性。在正确表格模型下重复备份趋向唯一固定点。$\gamma$=1 的终止任务需要另一套适当终止条件。

$$
V(S_t)\leftarrow V(S_t)+\alpha[G_t-V(S_t)]
$$

MC 用完整实际结果作为监督目标。需等到回报可计算；对固定策略、可积回报，它直接对正确条件期望采样。环境中途不断变化、轨迹没有结束或非常长时，等待完整回报代价很大。

$$
\delta_t=R_{t+1}+\gamma_{t+1}V_t(S_{t+1})-V_t(S_t),\qquad V_{t+1}(S_t)=V_t(S_t)+\alpha_t\delta_t
$$

TD(0) 只等一步，把剩下的未来交给当前估计。单个 target 通常不是对真实 v 的无偏样本，因为 V 还不准；但正确条件下长期固定点可以正确。因此，需要分析其长期固定点，而不只分析单次目标的偏差。

| 方法 | 需要什么 | 一条转移能否立即更新 | 误差从哪里来 |
| --- | --- | --- | --- |
| DP | 转移与奖励模型 | 可以，但使用模型期望 | 模型误差、备份不足 |
| MC | 完整采样回报 | 通常不能 | 有限样本方差 |
| TD | 一步经验与旧价值 | 可以 | 自举估计、有限样本、表示限制 |

<a id="value-traces"></a>

## 3 · 多步与 $\lambda$：奖励应该传回多远？

$$
G_t^{(n)}=\sum_{k=1}^n\gamma^{k-1}R_{t+k}+\gamma^n V(S_{t+n})
$$

终止时截到终点。n 小更依赖价值估计，n 大使用更多实际结果，也等待更久、通常有更大方差。具体的误差权衡取决于奖励噪声、价值估计和轨迹长度。

$$
G_t^\lambda=(1-\lambda)\sum_{n\ge1}\lambda^{n-1}G_t^{(n)},\qquad G_t^\lambda-V(S_t)=\sum_{k\ge0}(\gamma\lambda)^k\delta_{t+k}
$$

在无限折扣轨迹、固定 V 下，展开 n-step，按相同奖励与 V 项收集，价值项逐项抵消，就得到右边的 TD error 加权和。有限终止轨迹的最后一项吸收剩余权重。

$$
e_t=\gamma\lambda e_{t-1}+x_t,\qquad w_{t+1}=w_t+\alpha\delta_t e_t
$$

线性 V=wᵀx 时，把未来误差回传改写为记录过去特征。e 是“哪些参数对近期预测有影响”的信用痕迹，不是能在动作选择时回忆往事的隐状态。表格取 one-hot 特征。

这一步前向/后向等价的推导固定了参数。普通在线 TD($\lambda$) 每步都改参数，有限步长下不能声称严格等价于该固定参数目标；true-online TD($\lambda$) 用 Dutch trace 和额外修正实现相应在线前向视图。离策略时还要处理目标策略与行为策略的差别，相应修正将在通用价值函数一章中推导。

<a id="lesson-example"></a>

## 4 · 两步轨迹：为什么 MC 比 TD 更早改动起点？

A --奖励 0→ B --奖励 1→终点。$\gamma$=.9，初始 V(A)=V(B)=0，$\alpha$=.1。真实值为 $v_\pi(A)=0.9$、$v_\pi(B)=1$。

| 第一次经历 | A 的 target / 更新后值 | B 的 target / 更新后值 |
| --- | --- | --- |
| MC | 完整回报 .9 → .09 | 完整回报 1 → .1 |
| TD(0) | 0+.9×0=0 → 0 | 1+0−0 → .1 |
| TD($\lambda$=.8) | 第二步迹 .9×.8=.72 → .072 | 第二步当前特征迹 1 → .1 |

第二回合 TD 在 A 的 target 已是 .9×.1=.09，因此 V(A)=.009。延迟奖励通过多轮自举逐渐传播到起点。因此，MC 与 TD 即使具有相同的极限值，有限数据下的学习过程也不同。

<a id="lesson-code"></a>

## 5 · 核心实现与一次完整训练循环

**算法：线性、on-policy、回合制的 accumulating TD(λ)**

1. 初始化 $w$；给定步长 $\alpha$、迹参数 $\lambda$ 和特征函数 $x(s)$。
1. 每个回合开始时令 $e=0$，观察起始状态 $S$。
1. 每次转移：
  1. 按目标策略执行动作，观察 $R,S'$。
  1. 若真实终止，令 $\gamma'=0$；否则令 $\gamma'=\gamma$。
  1. $\delta\leftarrow R+\gamma'w^\top x(S')-w^\top x(S)$
  1. $e\leftarrow\gamma\lambda e+x(S)$
  1. $w\leftarrow w+\alpha\delta e$
  1. $S\leftarrow S'$；真实终止时结束本回合。

相同环境与步长，分别用整段结果和一步自举

```python
def prediction(method="td", episodes=200, alpha=0.1):
    """A --0--> B --1--> terminal; gamma=.9, exact values [.9, 1]."""
    values = [0.0, 0.0, 0.0]
    trajectory = [(0, 0.0, 1, 0.9), (1, 1.0, 2, 0.0)]
    for _ in range(episodes):
        if method == "mc":
            ret = 0.0
            for state, reward, _, discount in reversed(trajectory):
                ret = reward + discount * ret
                values[state] += alpha * (ret - values[state])
        elif method == "td":
            for state, reward, nxt, discount in trajectory:
                delta = reward + discount * values[nxt] - values[state]
                values[state] += alpha * delta
        else:
            raise ValueError("Choose mc or td")
    return values[:2]
```

- MC 从终点向前计算 return；本例每个状态每回合仅访问一次，因此 first-visit 与 every-visit 没有差别。一般轨迹中须明确选择。
- TD 在收到下一状态后立刻更新，后继值使用更新前可用的估计。终点值为 0。
- 运行 value 后两者应趋近 [.9, 1]。将 `episodes` 设为 1，可以比较首次更新与本节的数值计算。

<a id="lesson-branches"></a>

## 6 · 向持续预测的推广

| 分支 | 实际改变 | 需要继续检查 |
| --- | --- | --- |
| 常数步长跟踪 | 让新数据持续改变估计 | 方差、变化速度、访问频率 |
| GVF | 把任务奖励改为任意累积信号，并明确策略与延续 | 预测题目是否定义正确 |
| GTD / emphatic TD | 改变离策略函数逼近的更新几何或加权 | 覆盖、方差、线性理论条件 |
| 平均奖励 TD | 去掉长期增长的奖励率，学习差分价值 | 奖励率与价值的联合估计 |
| 神经网络 TD | 用共享非线性表示 | 自举、离策略、函数逼近的耦合 |

常数步长不追求在静止问题里把噪声彻底平均掉，而是保留对新规律的响应。有效样本权重随年龄呈指数衰减；真正的变化检测、滑动窗口选择和历史情境复用则是进一步的问题，不是 TD 自动具备的功能。

<a id="lesson-check"></a>

## 7 · 习题与讨论

问题：学习 TD 时把 target 中的下一状态价值也求导，会更“完整”吗？不一定。它改变了算法：TD 的半梯度更新是固定 target 后的回归方向；完整样本残差梯度优化另一目标。对期望 Bellman 残差平方求无偏梯度还涉及双采样。先写目标，再判断哪个梯度正确。

问题：为什么训练 TD error 不为零仍可能已经学对？随机奖励和随机转移会带来不可约的单样本误差；正确价值满足条件期望误差为零，不要求每条样本误差都零。用已知解析值或独立 rollout 估值，不能只看训练 loss。

## 本章的实验设计

在小型马尔可夫奖励过程中计算解析价值。再检查样本更新和参数误差。TD 误差不能代替价值误差。

设定：两状态奖励过程：A 无奖励到 B，B 获得 1 后回 A，γ=0.5。固定策略的价值为 v(A)=2/3、v(B)=4/3。

- 解析解满足两个 Bellman 方程。
- 零步长不写入，γ=0 只拟合下一奖励。
- 终端样本不读取无效后继值。

对照：MC、TD(0) 与指定版本 TD(λ)；相同轨迹与独立交互两种面板；固定表征及共同步长搜索预算

记录：对解析价值的均方误差；Bellman 残差与样本 TD 误差分别记录；固定数据量下的误差曲线

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-classic)

## 学习与研究衔接

表格 TD 的局部更新推广为共享特征上的参数更新。GVF 再改变预测问题，而非只改变网络。

[分册导读](https://yingwen.io/zh/continual-rl/start/classic-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-value) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=value) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=value)

<a id="chapter-code"></a>

## 下载与运行

Python 3.10+，仅标准库。实现表格 MC/TD 完整小实验；资格迹的前后向推导与实现见时间信用分配章。

[下载 foundations_detail_lab.py](https://yingwen.io/zh/continual-rl/download/foundations_detail_lab.py)

```sh
python3 foundations_detail_lab.py value
python3 foundations_detail_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：第 3–8 章建立 MDP、动态规划、MC、TD 与规划；第 9–13 章把这些更新扩展到函数逼近、资格迹和策略梯度。按本章的问题找相应章节，不必从头重读。

- [Sutton · Learning to Predict by the Methods of Temporal Differences](https://doi.org/10.1007/BF00115009)：TD 的原始问题动机与多步预测。读“如何利用尚未结束的经验”，并理解其更新规则。

- [van Seijen & Sutton · True Online TD($\lambda$)](https://proceedings.mlr.press/v32/seijen14.html)：检查在线前向视图、Dutch trace 与修正项；它解决的不是简单加大 $\lambda$。
