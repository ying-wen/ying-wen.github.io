# B2 · 平均奖励与 Differential Learning

[统一学习地图](/zh/continual-rl/algorithms/) · 先修 A1 TD、A2 Q-learning

## 先选目标：累计折扣收益，还是每一步的长期收益？

反复运行的机器人或服务系统可以没有自然终点。折扣目标按距离当前的远近给奖励加权；平均奖励目标则关心单位时间的长期收益。两种目标一般会产生不同偏好。改用平均奖励时，需要定义奖励率与相对价值，再修改更新规则，而非只把代码中的 γ 改成 1。

$$
J_\gamma^\pi(s)=\mathbb E_\pi\!\left[\sum_{t=0}^{\infty}\gamma^tR_{t+1}\mid S_0=s\right],\qquad g^\pi=\lim_{T\to\infty}\frac1T\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}R_{t+1}\right]
$$

这里先讨论有限、平稳 MDP 中的固定策略；假设长期奖励率存在且不依赖初始状态（例如该策略诱导的链是不可约的）。多 recurrent class、持续改变的策略或非平稳环境需要更细的定义与分析。γ∈[0,1) 是折扣因子，g 是每个原始环境步的奖励率。

## 为什么需要差分价值 h，而不是无穷奖励和？

长期每步收益为 g 时，直接相加的回报可能发散。差分价值衡量“从这个状态出发，相对长期奖励率的额外收益”。在上面的条件下，用 Poisson / Bellman 方程描述它：一步奖励减去 g，再加下一状态的相对价值。

$$
h^\pi(s)=\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)\big[r-g^\pi+h^\pi(s')\big]
$$

h 加上任意常数仍满足方程，因此需要参考或中心化来选定数值基准。奖励率 g 与 h 的常数偏移不是同一件事。这里用方程定义相对价值，避免在周期链上无条件假定未折扣无穷级数收敛。

## 从 Bellman 误差走到 Differential TD / Q

固定策略的 on-policy 预测，误差为 r−g+h(s′)−h(s)。控制时用 max Q bootstrap；Differential Q 用同一个 TD error 更新 Q 和奖励率估计，而不只是把行为轨迹上的即时奖励做滑动平均。

$$
\begin{aligned}\delta_t&=R_{t+1}-\bar R_t+\max_aQ_t(S_{t+1},a)-Q_t(S_t,A_t)\\Q_{t+1}(S_t,A_t)&=Q_t(S_t,A_t)+\alpha\delta_t\\\bar R_{t+1}&=\bar R_t+\eta\alpha\delta_t\end{aligned}
$$

α 是价值步长，η 调节奖励率估计的相对步长；右侧均使用更新前的值。论文的收敛结论有状态访问、步长和 MDP 结构等条件，不能从表格算法直接推出神经网络稳定。Off-policy 预测还需要目标策略／行为策略的概率比及覆盖条件。

## Options 的持续时间怎样进入价值？

一个 option 执行 τ 个原始步，得到未折扣总奖励 R̂。若长期每步可赚 g，执行它的机会成本是 gτ。于是半 MDP 的相对价值方程同时包含奖励、时长和终点价值。比较每步产出时，一个耗时很长的高奖励 option 仍可能不如短 option。

$$
Q(s,o)=\mathbb E\!\left[\widehat R-g\tau+\max_{o'}Q(S',o')\mid s,o\right]
$$

上述最优性方程规定要满足的关系，执行更新还需选择随机逼近算法。Wan 等的 inter-option Differential Q 用正的期望时长估计 L 缩放更新，并读取旧 L；样本 τ 用于更新 L，而非直接替换这个估计。推导和反例继续读 C6 及 F6。

## 与现代 DRL、reward centering、CRL 分别怎样连接？

Reward centering 在更新中减去奖励基线，也可用于 γ<1 的折扣学习；此时优化目标仍可保持为折扣回报。RVI-SAC 则把平均奖励相对价值更新与熵正则控制结合。确定长期目标以后，agent 还需解决知识构建、可塑性、探索和实时更新。

$$
r'_t=r_t-b,\quad 0\le\gamma<1\quad\Longrightarrow\quad V_{r'}^\pi(s)=V_r^\pi(s)-\frac{b}{1-\gamma}
$$

这个平移等式针对无限时域 continuing、固定且与策略无关的常数 b。若 b 在学习，或终止、reset 成本随策略变化，需要重新分析相应目标。变化环境中的跟踪还取决于步长、采样和表示，应结合适应曲线判断。

## 手算

δ = 3−1+4−2 = 4。α=0.1，η=0.5，更新后 Q=2.4、g=1.2。两个更新必须使用同一个旧值计算的 δ。

## 代码

```python
# Tabular Differential Q: a single transition, not a full learner.
# q is a mapping (state, action) -> value; alpha, eta > 0.
delta = reward - reward_rate + max(q[next_state, a] for a in actions) - q[state, action]
q[state, action] += alpha * delta
reward_rate += eta * alpha * delta
# Both updates use the SAME delta computed from the OLD q and reward_rate.
# Never replace next_state with an episodic terminal mask without specifying resets.
```

[作者实现与源码位置](/zh/continual-rl/construction/implementations/#family-average) · [公式小测试](/zh/continual-rl/download/formula_checks.py)

## 阅读与思考

- [Sutton & Barto · 第 10 章](http://incompleteideas.net/book/the-book-2nd.html)：读 continuing 任务、平均奖励与 differential return，再回来看上面为什么要减奖励率。
  - 思考：reward rate、差分价值和状态基准各在表达什么？
- [Wan、Naik、Sutton · Learning and Planning in Average-Reward MDPs](https://proceedings.mlr.press/v139/wan21a.html)：先对照 Differential TD/Q 伪代码，再读相应收敛假设与参考状态方法的比较。
  - 思考：为什么 max bootstrap 与行为轨迹奖励均值不能随意拼接？
- [Wan、Naik、Sutton · Average-Reward Learning and Planning with Options](https://proceedings.neurips.cc/paper/2021/file/c058f544c737782deacefa532d9add4c-Paper.pdf)：对照式 (6)–(9) 的 L、δ 和缩放次序；与 C6 的 option backup 一起读。
  - 思考：为什么 E[R̂]/E[τ] 与 E[R̂/τ] 不同？
- [Hisaki、Ono · RVI-SAC](https://proceedings.mlr.press/v235/hisaki24a.html)：沿 soft value、reference 统计量和 reset cost 追完整目标，而不是在普通 SAC 中简单删 γ。
  - 思考：训练中的 entropy、reset cost 与评测奖励率如何分别记录？
