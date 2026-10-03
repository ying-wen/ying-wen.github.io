# 函数逼近预测：从回归到 TD 固定点

共享少量参数以后，MC、TD 和最小二乘方法究竟在求解什么？

## 本章内容

- 从平方价值误差推导梯度 MC，说明采样分布的作用。
- 推导线性 TD 的平均更新、投影 Bellman 方程与 LSTD。
- 用同一个二状态例子计算不同解，并检查半梯度的含义。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 固定策略的预测问题

策略不在本章学习过程中改变。它与环境共同给出转移矩阵 $P$ 和期望一步奖励向量 $r$。目标是估计执行此策略后的折扣回报。

$$
v_\pi=r_\pi+\gamma P_\pi v_\pi
$$

### 线性函数逼近

每个状态映射为一个 $d$ 维特征向量 $x$。$d$ 个共享权重产生全部状态的预测；线性指预测对权重线性，不要求特征对状态线性。

$$
\hat v(s,w)=x(s)^\top w,\qquad \Phi_{s,:}=x(s)^\top
$$

<a id="lesson-setting"></a>

## 1 · 表格无法逐项保存时的预测问题

表格预测为每个状态分配独立参数。遇到连续观测或大量组合状态时，这个表示不可行。函数逼近把多个状态的预测绑定到同一组参数。于是一次更新不仅改变当前状态，也可能改变尚未再次访问的状态。泛化由表示决定，更新目标由学习算法决定。

考虑固定策略、有限状态和 $0\le\gamma<1$。用 $d(s)$ 表示评价状态的权重，令 $D=\operatorname{diag}(d)$。它可以是策略的平稳分布，也可以是另外指定的分布，但采样规则必须与推导一致。误差较小究竟意味着在哪些状态准确，需要在算法之前确定。

$$
J_{\mathrm{VE}}(w)=\frac12\sum_s d(s)\big[v_\pi(s)-\hat v(s,w)\big]^2.
$$

平方价值误差把真实价值当作目标。若一个状态的 $d(s)$ 为零，这个目标不会直接要求该状态准确；共享参数仍可能间接改变它。

本章先使用固定特征。可学习的神经表示会使特征和目标共同变化，不能直接继承下面的线性结论。这里把表示固定，是为了看清误差来源，而不是因为实际智能体只能使用固定表示。

<a id="lesson-derive"></a>

## 2 · 梯度 MC：从误差函数到样本更新

$$
\nabla_wJ_{\mathrm{VE}}=-\mathbb E_d[(v_\pi(S)-\hat v(S,w))\nabla_w\hat v(S,w)].
$$

对平方误差求导得到“误差乘预测梯度”。若完整回报 $G_t$ 满足 $\mathbb E[G_t\mid S_t=s]=v_\pi(s)$，便可以用一个回报样本替代未知价值。

$$
w_{t+1}=w_t+\alpha_t[G_t-\hat v(S_t,w_t)]\nabla_w\hat v(S_t,w_t).
$$

在线性情形，梯度就是 $x_t$。这里没有通过 $G_t$ 求导，因为预测问题中的策略和环境固定，完整回报不依赖预测权重。

回合式任务可以等到终止再计算回报。长回合使反馈延迟，随机未来使目标方差增大。继续型任务也有定义良好的折扣价值，但无法在有限时间观察无限回报；截断后仍需说明尾项如何处理。因此 MC 的无偏目标条件和现实可用估计需要区分。

$$
w_{\mathrm{MC}}=(\Phi^\top D\Phi)^{-1}\Phi^\top Dv_\pi.
$$

固定线性特征、$C=\Phi^\top D\Phi$ 可逆时，令梯度为零得到加权最小二乘解。它是 $v_\pi$ 在特征张成空间上的正交投影。

<a id="prediction-td"></a>

## 3 · 半梯度 TD：目标变化但只对当前预测求导

$$
\begin{aligned}\delta_t&=R_{t+1}+\gamma x_{t+1}^\top w_t-x_t^\top w_t,\\ w_{t+1}&=w_t+\alpha_t\delta_t x_t.\end{aligned}
$$

一步 TD 用下一状态的当前预测补上未观察到的尾部。目标中也含 $w_t$，更新却只使用当前预测的梯度 $x_t$，因此称为半梯度。

若把样本平方 TD 误差也对目标求导，更新方向会变成 $\delta_t(x_t-\gamma x_{t+1})$。那是另一种方法，不是“更完整的 TD”。它对应的目标、所需采样和所得近似解都可能不同。自举不自动等于优化样本 TD 平方误差。

$$
\begin{aligned}\mathbb E[\delta_t x_t]&=b-Aw,\\ A&=\Phi^\top D(I-\gamma P_\pi)\Phi,\\ b&=\Phi^\top Dr_\pi.\end{aligned}
$$

先将 $\delta_t$ 展开，再对奖励和下一状态取条件期望。TD 的平均更新为零时满足 $Aw=b$，通常不是价值误差的正规方程。

$$
\begin{aligned}\Phi w&=\Pi_D T_\pi(\Phi w),\\ \Pi_D&=\Phi(\Phi^\top D\Phi)^{-1}\Phi^\top D.\end{aligned}
$$

把 $Aw=b$ 改写成 $\Phi^\top D[T_\pi\Phi w-\Phi w]=0$，再左乘 $\Phi C^{-1}$，得到投影 Bellman 固定点。先作 Bellman 更新，再投影回可表示空间。

<a id="prediction-algorithm"></a>

## 4 · 逐步 TD 与 LSTD 的计算顺序

**算法：线性半梯度 TD(0)**

1. 初始化权重 $w$，指定特征、策略、步长与终止语义
1. 观察 $S_t$，按固定策略产生动作和转移
1. 保存旧预测 $v=x_t^\top w$ 与 $v'=x_{t+1}^\top w$
1. 若真正终止，令 $v'=0$；计算 $\delta=R_{t+1}+\gamma v'-v$
1. 执行 $w\leftarrow w+\alpha\delta x_t$，然后推进状态

$$
\begin{aligned}\hat A&=\sum_t x_t(x_t-\gamma_{t+1}x_{t+1})^\top,\\ \hat b&=\sum_t x_tR_{t+1},\\ \hat A\hat w&=\hat b.\end{aligned}
$$

LSTD 累积同一组固定点方程，再解线性系统。此处 $\gamma_{t+1}$ 在真正终止时取零。数据和特征相同并不意味着有限样本 TD 迭代与 LSTD 已经给出相同参数。

TD 每步需要 $O(d)$ 计算与内存。直接 LSTD 需要 $O(d^2)$ 存储，批量求解通常需要 $O(d^3)$ 计算。样本不足或特征相关可使矩阵奇异；加正则项会改变估计问题，不能把正则后的解称为原方程的精确解。

<a id="lesson-example"></a>

## 5 · 手算：同一表示下的两种正确答案

有两个状态 A、B，确定性地交替。A 到 B 的奖励为 0，B 到 A 的奖励为 1，折扣 $\gamma=1/2$。由 $v_A=v_B/2$ 和 $v_B=1+v_A/2$，得到 $v_A=2/3,v_B=4/3$。长期访问频率各为一半。

只允许一个权重，特征为 $x_A=1,x_B=3$。真实价值的比例为 1:2，表示强制比例为 1:3，所以不可能在两个状态同时精确。现在比较不同误差准则的解。

$$
C=\tfrac12(1^2+3^2)=5,\quad \Phi^\top Dv=\tfrac73,\quad w_{\mathrm{MC}}=\tfrac7{15}.
$$

价值回归先让真实价值误差对特征正交。

$$
\begin{aligned}A&=\tfrac12(1)(1-\tfrac12\cdot3)+\tfrac12(3)(3-\tfrac12\cdot1)=\tfrac72,\\ b&=\tfrac32,\qquad w_{\mathrm{TD}}=\tfrac37.\end{aligned}
$$

TD 则让期望 TD 更新为零。两个答案不同，不能据此断言某个实现错误。

| 方法 | 权重 | A 的预测 | B 的预测 |
| --- | --- | --- | --- |
| 真实价值 | 不可用单个权重表示 | 2/3 | 4/3 |
| 梯度 MC 的目标解 | 7/15 | 7/15 | 7/5 |
| TD / LSTD 固定点 | 3/7 | 3/7 | 9/7 |

<a id="lesson-code"></a>

## 6 · 可运行实现与固定点检查

MC、半梯度 TD、LSTD 与二状态解析例子

```python
def gradient_mc(weights, features, target, alpha):
    error = target - dot(weights, features)
    return [w + alpha * error * x for w, x in zip(weights, features)]


def semi_gradient_td(weights, features, reward, next_features, gamma, alpha):
    # Both predictions use the pre-update weights.
    delta = reward + gamma * dot(weights, next_features) - dot(weights, features)
    return [w + alpha * delta * x for w, x in zip(weights, features)], delta


def lstd(samples, dimension):
    """samples = (mass, x, r, gamma_next, x_next); mass may be 1 or probability."""
    a = [[0.0] * dimension for _ in range(dimension)]
    b = [0.0] * dimension
    for mass, x, reward, gamma, xp in samples:
        for i in range(dimension):
            b[i] += mass * reward * x[i]
            for j in range(dimension):
                a[i][j] += mass * x[i] * (x[j] - gamma * xp[j])
    return solve(a, b), a, b


def prediction_demo():
    # A -> B (r=0), B -> A (r=1), gamma=.5, stationary masses .5/.5.
    samples = [(0.5, [1.0], 0.0, 0.5, [3.0]),
               (0.5, [3.0], 1.0, 0.5, [1.0])]
    exact = solve([[1.0, -0.5], [-0.5, 1.0]], [0.0, 1.0])
    td, a, b = lstd(samples, 1)
    mc = (0.5 * exact[0] + 1.5 * exact[1]) / 5.0
    return {"true_values": exact, "A": a, "b": b, "TD_weight": td[0],
            "MC_weight": mc, "TD_values": [td[0], 3 * td[0]],
            "MC_values": [mc, 3 * mc]}
```

代码中的样本权重既可以是观测次数，也可以是已知概率。本例用概率计算期望矩阵，因此没有蒙特卡洛误差。它适合检查推导，不能用于估计某种学习器在随机数据下的样本效率。测试另外对 MC 做有限差分，对 TD 检查更新前后参数的时序。

运行本章与全部公式测试

```sh
python3 approximation_textbook_lab.py prediction
python3 approximation_textbook_lab.py test
```

<a id="lesson-branches"></a>

## 7 · 收敛条件与持续学习的连接

一组常用的充分条件是：固定策略产生有限、不可约、非周期链，$0\le\gamma<1$，奖励具有有界二阶矩，固定特征在其平稳分布下线性独立，并采用合适的递减步长。此时线性在策略 TD 可以收敛到唯一投影固定点。控制中策略变化、离策略采样或神经特征共同训练，会改变这些条件。

$$
\alpha_t>0,\qquad\sum_t\alpha_t=\infty,\qquad\sum_t\alpha_t^2<\infty.
$$

这是经典随机逼近步长条件：总更新量不耗尽，累计噪声的平方权重却可控。它们须和采样、矩与表示条件一起使用。常数步长不满足第二个级数条件，不能据此宣称几乎处处收敛。

持续环境中，奖励、转移或表示变化使固定点也变化。递减到接近零的步长不利于追踪，常数步长则留下稳态噪声。研究问题从“是否达到一个固定点”变成“变化后多快重新准确、长期误差多大、旧预测是否仍可用”。

GVF 将奖励替换为所关心的累积信号，并指定策略与延续条件。它仍需要这里的预测算法。Agent state 决定特征包含什么历史；预测目标和更新器不会自动修复缺失信息。由此可以分别实验表示充分性、更新稳定性和追踪能力。

<a id="lesson-check"></a>

## 8 · 练习与答案

- 若 $\gamma=0$，TD 变成什么？答案：目标只剩一步奖励，线性更新就是以该奖励为标签的随机梯度回归。
- 为什么这里不能说 TD 最小化价值误差？答案：它的平衡条件是 $Aw=b$，而回归是 $Cw=\Phi^\top Dv_\pi$，两者通常不同。
- 把二状态特征改为两个 one-hot 分量，会怎样？答案：表示空间包含全部价值函数，MC 投影和 TD 固定点都等于真实价值。
- 只记录平均 TD 误差接近零，足以说明预测准确吗？答案：不够。标量误差可相互抵消；即使向量期望更新为零，也可能有函数逼近误差。

<a id="chapter-code"></a>

## 下载与运行

标准库教学实现。命令运行本章的可解析小问题和全文件的公式测试；不依赖深度学习库，不等同于论文中的大规模实验。

[下载 approximation_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/approximation_textbook_lab.py)

```sh
python3 approximation_textbook_lab.py prediction
python3 approximation_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：Part II 第 9–13 章。原书建立函数逼近预测、控制、离策略稳定性、资格迹与策略梯度的共同框架。

- [Bradtke & Barto · Linear Least-Squares Algorithms for Temporal Difference Learning](https://link.springer.com/article/10.1007/BF00114723)：LSTD 的原始论文。适合在理解期望 TD 正规方程之后阅读。

<a id="study-connections"></a>

## 与教材主线的衔接

- [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)
- [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)
- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)

对应原始材料：第 9 章：On-policy Prediction with Approximation。本文为原创讲解，原书、论文与上游代码保留各自许可。
