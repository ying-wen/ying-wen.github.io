# 离策略函数逼近：覆盖、发散与稳定更新

行为数据足够覆盖目标策略，为什么 TD 仍可能发散，又能怎样修复？

## 本章内容

- 区分动作重要性修正与状态加权带来的稳定性。
- 推导 MSPBE 及 GTD2、TDC 的辅助权重更新。
- 理解 emphatic weighting 的目的、时间索引和线性理论条件。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 两种策略与覆盖

行为策略 $b$ 生成经验，目标策略 $\pi$ 指定待预测的未来。只要 $\pi$ 对某个状态—动作给正概率，$b$ 也必须给正概率，才可能使用有限的重要性比。

$$
\rho_t=\pi(A_t\mid S_t)/b(A_t\mid S_t)
$$

### 投影固定点

固定特征下，TD 求解期望更新为零的方程。这里的状态权重来自行为分布，而下一步目标来自目标策略。

<a id="lesson-setting"></a>

## 1 · 一个经验流支持多个预测

机器人只实际执行一个动作，却可能同时预测“继续前进会碰撞吗”和“转向后能否充电”。这些预测的目标策略与正在执行的策略不同。经验回放也会使用旧策略的数据。离策略学习因而不是一种特定控制算法，而是一种数据来源与目标行为分离的设定。

$$
\begin{aligned}&\mathbb E_b[\rho_t f(S_t,A_t,S_{t+1})\mid S_t=s]\\ &\qquad=\mathbb E_\pi[f(s,A_t,S_{t+1})\mid S_t=s].\end{aligned}
$$

重要性比修正给定当前状态时的动作分布。它没有把当前状态的访问权重 $d_b(s)$ 自动变成 $d_\pi(s)$，更没有改变特征共享。

覆盖保证需要的数据原则上能够被观察。概率极小时，权重可能很大，估计方差也很大。将比值截断或归一化可以改变方差，但通常也改变所估计的量。除了支持条件，还要检查状态分布、特征和更新算子的组合是否稳定。

<a id="offpolicy-counterexample"></a>

## 2 · 覆盖充分却发散的二状态反例

有 A、B 两状态和“去 A”“去 B”两个动作。行为在每个状态都以 0.1 概率去 B、0.9 概率去 A，所以平稳状态权重是 (0.9,0.1)。目标总去 B；它的动作被行为覆盖。令奖励全为 0，$\gamma=0.9$，一维特征 $x_A=1,x_B=2$。

$$
\begin{aligned}A&=\mathbb E_b[\rho x(x-\gamma x')]\\ &=0.9(1)(1-1.8)+0.1(2)(2-1.8)\\ &=-0.68.\end{aligned}
$$

目标的真实价值为零。但目标始终指向大特征状态，加上行为状态权重后，平均 TD 更新的反馈符号反转。

$$
\begin{aligned}\mathbb E[\Delta w\mid w]&=\alpha(0-Aw)=0.68\alpha w,\\ w^+&=(1+0.68\alpha)w.\end{aligned}
$$

任何正步长都会让非零权重在这个期望递推中增长。减小步长只会减慢增长，不能改变固定点的不稳定性。

这展示了“致命三元组”：自举、离策略和函数逼近同时存在时可能不稳定。它不是说每一个包含三者的算法都会发散，也不是说去掉任一项就无需其他条件。神经网络、目标网络与回放会改变动力学，需要另外检查。

<a id="lesson-derive"></a>

## 3 · 三种误差与 MSPBE 的推导

$$
\begin{aligned}\mathrm{MSBE}&=\|T_\pi\Phi w-\Phi w\|_D^2,\\ \mathrm{MSPBE}&=\|\Pi_D T_\pi\Phi w-\Phi w\|_D^2.\end{aligned}
$$

MSBE 衡量 Bellman 残差，MSPBE 先将残差投影到可表示空间。这里 $D=\operatorname{diag}(d_b)$。二者均不同于对真实价值的误差。

样本均方 TD 误差 $\mathbb E[\delta^2]$ 又是第三个量，因为“先平方再求期望”不同于“条件期望后平方”。其差值包含转移与奖励噪声的条件方差。直接对同一个样本的平方残差求梯度，通常不能得到 MSBE 的无偏梯度；条件独立的双采样是一个经典障碍。

$$
\begin{aligned}C&=\mathbb E_b[xx^\top],\\ A&=\mathbb E_b[\rho x(x-\gamma x')^\top],\\ b_v&=\mathbb E_b[\rho Rx],\\ J(w)&=\tfrac12(b_v-Aw)^\top C^{-1}(b_v-Aw).\end{aligned}
$$

用 $b_v$ 表示奖励向量，避免与行为策略 $b$ 混淆。将投影矩阵代入 MSPBE 并约去 $C$ 后，得到这个二次目标。要求特征在行为分布下独立，使 $C$ 可逆。

$$
\begin{aligned}-\nabla J(w)&=A^\top C^{-1}(b_v-Aw),\\ h&\approx C^{-1}(b_v-Aw).\end{aligned}
$$

辅助权重 $h$ 学习一组预条件化的期望 TD 更新。它不是第二个价值函数，而是为了用单个经验流跟踪梯度中的期望。

<a id="offpolicy-gtd"></a>

## 4 · GTD2 与 TDC 的两组更新

$$
\begin{aligned}h^+&=h+\beta[\rho\delta-x^\top h]x,\\ w^+_{\mathrm{GTD2}}&=w+\alpha\rho(x-\gamma x')(x^\top h).\end{aligned}
$$

GTD2 直接采样主参数的梯度方向。两个式子都使用旧 $h$ 和旧 $w$；不能用刚更新的辅助参数替代同一步公式中的 $h$。

$$
w^+_{\mathrm{TDC}}=w+\alpha\rho\big[\delta x-\gamma x'(x^\top h)\big].
$$

TDC 保留普通 TD 项，再加修正项。辅助参数仍按上一式更新。在 $h=C^{-1}(b_v-Aw)$ 时，TDC 的期望方向与 MSPBE 的负梯度一致；有限时刻的样本方向并不要求相同。

尤其注意：辅助更新的协方差项 $-(x^\top h)x$ 不整体乘 $\rho$。这组公式选择行为状态下的 $C$。当某个动作使 $\rho=0$ 时，主参数不更新，但辅助参数仍可以沿协方差项变化。代码为这个容易写错的情况设置了测试。

**算法：Gradient TD 的单步时序**

1. 初始化主权重 $w$、辅助权重 $h$，固定目标与行为策略
1. 读取转移及采样时的行为概率，检查支持并计算 $\rho$
1. 用旧参数计算 $\delta$ 和 $x^\top h$
1. 按 GTD2 或 TDC 计算完整主参数增量
1. 按辅助方程计算增量，最后同时写入两个新参数

经典理论针对固定线性特征、固定策略、适当矩阵非奇异性和矩条件，并要求与分析匹配的采样和递减步长。GTD2 的联合系统与 TDC 的两时间尺度分析需要区分；TDC 常令辅助参数比主参数更快，步长比趋于零。本页固定步长的小例子只检验公式和动力学，不是这些随机收敛定理的复现。

<a id="offpolicy-emphasis"></a>

## 5 · Emphatic TD：改变状态更新权重

另一条路线改变各状态的更新强调程度。非负 interest $i(s)$ 指定哪些预测重要；follow-on trace 追踪重要性沿目标策略的后继传播。设 $0\le\lambda_t\le1$，初始化 $F_{-1}=0,e_{-1}=0$。这种加权改变了投影几何，不是在原来 TD 更新外附加一个新的奖励。

$$
\begin{aligned}F_t&=i_t+\gamma_t\rho_{t-1}F_{t-1},\\ M_t&=\lambda_t i_t+(1-\lambda_t)F_t,\\ e_t&=\rho_t(\gamma_t\lambda_t e_{t-1}+M_t x_t),\\ w_{t+1}&=w_t+\alpha_t\delta_t e_t.\end{aligned}
$$

$\gamma_t$ 描述进入当前状态的延续，$\delta_t$ 的 bootstrap 使用 $\gamma_{t+1}$。$F_t$ 用上一动作比值，$e_t$ 用当前比值。interest 不是额外奖励。

有限状态 ETD 理论要求固定目标满足 $(I-P_\pi\Gamma)^{-1}$ 存在，行为链不可约且覆盖目标动作，强调为正的状态具有足够独立特征。奖励噪声方差有界；原论文还给出特定递减步长条件，例如适当的 $\alpha_t=a/(b+t)$，其中 $a,b>0$。表示学习、变化策略和常数步长追踪，不直接落在这一定理内。

强调提高稳定性不等于低方差。多个重要性比沿时间传播可能产生大幅 trace。记录 trace 范数和重尾更新很重要；若裁剪 trace，则应说明已改变算法。没有必要为了避免普通 TD 发散而隐藏修复方法自身的方差代价。

<a id="lesson-example"></a>

## 6 · 手算辅助变量与强调的时间索引

令 $w=1,h=0,x=1,x'=2,R=0,\gamma=0.9,\rho=2,\alpha=0.1,\beta=0.2$。误差 $\delta=0.8$。GTD2 第一步主权重仍为 1，因为旧辅助权重为零；辅助权重变为 0.32。TDC 第一步则得到主权重 1.16。

对于 ETD，取旧 $F=2$、上一比值 3、进入当前状态的折扣 0.5、interest 为 1。得到 $F_t=1+0.5\times3\times2=4$。若 $\lambda=0,\rho_t=4,x_t=1$，则新 trace 为 16。把当前比值错放进 follow-on，会得到不同答案。

二状态反例中 $C=1.3,b_v=0$，因此 $J(w)=0.68^2w^2/(2\times1.3)$。这是以零为最小点的凸二次函数。GTD2 的期望递推可以下降，而普通 TD 的期望递推增长；代码把它们在同一概率模型中比较。

<a id="lesson-code"></a>

## 7 · 可运行的稳定性与时序检查

GTD2、TDC、ETD 更新核与概率加权反例

```python
def gradient_td(weights, auxiliary, x, reward, xp, gamma, rho, alpha, beta, method):
    if method not in ("GTD2", "TDC") or rho < 0:
        raise ValueError("method must be GTD2/TDC and rho nonnegative")
    delta = reward + gamma * dot(weights, xp) - dot(weights, x)
    hx = dot(auxiliary, x)
    if method == "GTD2":
        direction = [rho * (v - gamma * vp) * hx for v, vp in zip(x, xp)]
    else:
        direction = [rho * (delta * v - gamma * vp * hx) for v, vp in zip(x, xp)]
    next_weights = [w + alpha * d for w, d in zip(weights, direction)]
    # rho multiplies delta, NOT the covariance term hx.
    next_auxiliary = [h + beta * (rho * delta - hx) * v for h, v in zip(auxiliary, x)]
    return next_weights, next_auxiliary


def emphatic_td(weights, trace, followon, previous_rho, x, reward, xp,
                gamma_current, gamma_next, rho, interest, lam, alpha):
    followon = interest + gamma_current * previous_rho * followon
    emphasis = lam * interest + (1 - lam) * followon
    trace = [rho * (gamma_current * lam * e + emphasis * v) for e, v in zip(trace, x)]
    delta = reward + gamma_next * dot(weights, xp) - dot(weights, x)
    weights = [w + alpha * delta * e for w, e in zip(weights, trace)]
    return weights, trace, followon, rho


def off_policy_demo():
    # b(go-to-B)=.1 in each state, pi(go-to-B)=1. d_b=(.9,.1).
    # Features x(A)=1, x(B)=2. All rewards zero. A=-.68, C=1.3.
    samples = [(0.81, [1.0], [1.0], 0.0), (0.09, [1.0], [2.0], 10.0),
               (0.09, [2.0], [1.0], 0.0), (0.01, [2.0], [2.0], 10.0)]
    w, h, td = [1.0], [0.0], 1.0
    for _ in range(2000):
        increments_w, increments_h = 0.0, 0.0
        for probability, x, xp, rho in samples:
            wn, hn = gradient_td(w, h, x, 0, xp, 0.9, rho, 0.01, 0.05, "GTD2")
            increments_w += probability * (wn[0] - w[0])
            increments_h += probability * (hn[0] - h[0])
        w, h = [w[0] + increments_w], [h[0] + increments_h]
        td += 0.01 * 0.68 * td
    return {"A": -0.68, "C": 1.3, "expected_TD_after_2000": td,
            "expected_GTD2_after_2000": w[0],
            "sampling": "exact probability-weighted updates, not a stochastic benchmark"}
```

反例使用全部四种状态—动作组合的精确概率加权，不把单次随机曲线当作期望。运行 2000 次期望更新后，普通 TD 权重约为 769864，GTD2 约为 0.000551。数值展示一个反例的差异；它不说明 GTD2 在所有任务上都更快或更好。

运行离策略反例与更新测试

```sh
python3 approximation_textbook_lab.py off-policy
python3 approximation_textbook_lab.py test
```

<a id="lesson-branches"></a>

## 8 · 从预测理论到 CRL 的边界

多个 GVF 可以共享一个行为流，但每个预测仍须定义自己的目标策略、信号、折扣和 interest。共享表示会耦合更新；线性固定特征下某个预测器稳定，不代表联合神经表示一定稳定。

环境或表示变化时，应分别监测覆盖、价值误差、重要性比、辅助参数范数和 trace 尾部。发散可能来自动态算子、数值尺度、非平稳目标或缺失信息。更换优化器未必解决投影几何，扩大 replay 也未必改善目标动作的覆盖。

<a id="lesson-check"></a>

## 9 · 练习与答案

- 为什么重要性比已经正确，TD 仍发散？答案：它只修正条件动作分布；行为状态加权与共享特征仍可能形成不稳定的投影更新。
- GTD2 第一步主权重不变，是否说明程序失效？答案：若辅助权重初始化为零，这是公式要求；后续辅助信息才驱动主更新。
- $\lambda=1$ 时，ETD 的 $M_t$ 是什么？答案：$i_t$，但 trace 仍含重要性比与历史递推。
- MSPBE 小是否保证控制策略好？答案：不保证。它是给定表示和权重下的预测准则，还要检验表示误差与决策收益。

<a id="chapter-code"></a>

## 下载与运行

标准库教学实现。命令运行本章的可解析小问题和全文件的公式测试；不依赖深度学习库，不等同于论文中的大规模实验。

[下载 approximation_textbook_lab.py](https://yingwen.io/zh/continual-rl/download/approximation_textbook_lab.py)

```sh
python3 approximation_textbook_lab.py off-policy
python3 approximation_textbook_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：Part II 第 9–13 章。原书建立函数逼近预测、控制、离策略稳定性、资格迹与策略梯度的共同框架。

- [Sutton et al. · Fast Gradient-Descent Methods for Temporal-Difference Learning with Linear Function Approximation](https://icml.cc/2009/papers/546.pdf)：GTD2、TDC 与投影误差推导的原始论文。本文代码额外显式写出目标与行为动作的重要性比。

- [Mahmood et al. · Emphatic Temporal-Difference Learning](https://arxiv.org/abs/1507.01569)：ETD 的 interest、follow-on 与线性收敛条件。阅读公式时区分当前与上一重要性比。

- [RLPark · Original reinforcement-learning implementations](https://github.com/rlpark/rlpark)：作者群体开发的线性与资格迹算法实现库。版本中的梯度 TD 命名和具体更新须逐项对应，不能只按类名互换。

<a id="study-connections"></a>

## 与教材主线的衔接

- [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)
- [价值预测与资格迹](https://yingwen.io/zh/continual-rl/algorithms/value/)
- [深度价值学习](https://yingwen.io/zh/continual-rl/algorithms/deep-value/)

对应原始材料：第 11 章：Off-policy Methods with Approximation。本文为原创讲解，原书、论文与上游代码保留各自许可。
