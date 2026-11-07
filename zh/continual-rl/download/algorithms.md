[统一学习地图](https://yingwen.io/zh/continual-rl/download/curriculum.md) · 本文件为基础更新与长期学习机制的详细推导。预测、状态、子任务、时间抽象与规划各有独立章节。

# RL → CRL：基础算法脉络与代码

先读 01、02、04，再根据问题选择 CRL 机制。公式、例子与网页共用同一份内容。

## 01 价值从哪里来：Bellman、MC、TD 与资格迹

先修：状态、奖励、期望与加权平均

问题：还没走到终点，能否用刚发生的一步改善预测？

Bellman 递推 → MC：完整结果 → TD：一步自举 → 多步 / TD(λ)

先固定策略，只问“照这样走，未来能得到多少奖励”。动态规划用已知模型对所有可能后果取期望；Monte Carlo（MC）用实际完整轨迹；TD 用一步实际奖励接上对未来的估计。它们首先是不同的价值更新方式，不是三种不同的奖励目标。

符号：Sₜ、Aₜ、Rₜ₊₁：当前状态、动作、下一次奖励；γ：折扣；V：状态价值估计；α：学习步长。真实终止状态的后继价值为 0。

### 先定义要预测的量，再写 Bellman 关系

回报是未来奖励的折扣和。拆出第一项，就把长时间预测变成“即时奖励 + 下一状态价值”。Bellman 等式描述正确答案应满足的关系；它本身还没有告诉你怎样从有限数据学习。

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1},\qquad v^\pi(s)=\mathbb E_\pi[R_{t+1}+\gamma v^\pi(S_{t+1})\mid S_t=s]
$$

### 有模型时：策略迭代与价值迭代

若已知转移及奖励模型，策略迭代交替做“评估当前策略”和“对价值贪心改进”。价值迭代将二者合成最优 Bellman backup。下面对下一状态求精确期望；MC/TD 则从采样经验获得更新信号。实际复杂环境通常没有这样的完整模型。

$$
V_{k+1}(s)=\max_a\sum_{s'}p(s'|s,a)\left[r(s,a,s')+\gamma V_k(s')\right]
$$

这里 r 是给定转移的期望奖励。标准保证需要有限 MDP、折扣 γ<1 等条件；无折扣任务要另检查终止性。

### MC 与 TD 的区别在 target

二者都用“估计 ← 估计 + 步长 × 误差”。MC 等完整轨迹结束，用 Gₜ；TD(0) 每步即可用 R+γV(S′)。TD 的 target 包含自己的估计，称为自举：更及时，但也会传递估计误差。

$$
\begin{aligned}V(S_t)&\leftarrow V(S_t)+\alpha[G_t-V(S_t)]&&\text{MC}\\V(S_t)&\leftarrow V(S_t)+\alpha[R_{t+1}+\gamma V(S_{t+1})-V(S_t)]&&\text{TD(0)}\end{aligned}
$$

### 多步与资格迹：让奖励影响更早的状态

n-step target 用 n 个真实奖励再自举。资格迹则给过去参与过的状态或特征保留一份衰减责任记录。下面是表格、on-policy、accumulating trace；每个 episode 开始清零。

$$
G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^nV(S_{t+n}),\quad e_t=\gamma\lambda e_{t-1}+\mathbf1_{S_t},\quad V\leftarrow V+\alpha\delta_t e_t
$$

接近终点时按真实剩余奖励截断。λ=0 回到 TD(0)；在线 accumulating traces 的 λ=1 不应不加条件地说成与逐次 MC 完全相同。

### 手算例子

一条两步轨迹奖励为 0、1，γ=0.9，V(s₀)=0.2，V(s₁)=0.4。s₀ 的 MC target 为 0.9；第一步 TD target 为 0.36。α=0.1 时分别得到 0.27 和 0.216。目标都关心未来奖励，只是使用的证据不同。

### 对照代码

```python
# 一条 transition；terminal 是真正终止，而非任意日志窗口结束
target = reward if terminal else reward + gamma * values[next_state]
delta = target - values[state]
trace = [gamma * lam * e for e in trace]
trace[state] += 1
for s in nonterminal_states:
    values[s] += alpha * delta * trace[s]
```

下载 rl_foundations.py，先读 prediction()。它先生成固定策略的相同轨迹，再分别交给 every_visit_mc、td0 和 td_lambda，避免把策略不同误认为预测规则不同。

```sh
python3 rl_foundations.py prediction --seeds 5 --episodes 1000 --alpha 0.1 --out value-run
```

打开 value-run.html：纵轴是相对解析真值 v(i)=i/6 的 RMSE，不是训练回报。将 --lam 改为 0，td_lambda 与 td0 的输出应一致。

### 怎样接到 CRL

常数步长、即时更新和迹，是持续预测的基础。但平稳随机游走上能预测，并不证明面对长期变化也能跟踪；下一步改变目标时，应保留同一学习器而非重训。

适用条件：这里是有限状态、固定策略、自然终止任务。用神经网络、off-policy 数据或递归状态时，不能直接沿用表格收敛直觉。

### 思考题

γ=0.9，下一状态确实终止且奖励为 1；它的数组里残留 V=100。target 应为多少？

提示：终止之后没有这个任务的后续回报。

参考答案：为 1，而不是 91。把终止标记与时间上限截断混淆，可能系统性改变学习目标。

### 原始材料与代码

- [Sutton & Barto · 第 4–7、12 章](http://incompleteideas.net/book/the-book-2nd.html)：先比较 backup 目标，再看 traces；不用先学神经网络。

---

## 02 从预测到决策：SARSA 与 Q-learning

先修：上一章 TD；ε-greedy

问题：要估计自己接下来实际怎么走，还是估计下次都选最优动作会怎样？

TD 预测 → 动作价值 Q → SARSA / Q-learning → 探索与控制

只有 V(s) 时，你知道一个状态大概好不好，却不能直接比较不同动作。用 Q(s,a) 预测先做 a 的长期后果，就能由价值改善行为。SARSA 与 Q-learning 的关键分歧不是用不用神经网络，而是 target 里的下一动作。

符号：Q(s,a)：动作价值；A′：行为策略真正选出的下一动作；ε：随机探索概率。行为策略生成数据，目标策略决定你想学习谁的价值。

### SARSA：把未来探索也算进来

按当前 ε-greedy 策略选出 A′，再用它的估值更新。算法名称对应 (S,A,R,S′,A′)。若探索可能走入危险区，SARSA 的估值会反映这种未来行为风险。

$$
\delta_t^{\mathrm{SARSA}}=R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)
$$

### Q-learning：target 假设下一步贪心

真实行动仍可探索，但 target 用下一状态最大 Q。它学习的目标与实际生成动作的探索策略不同，因此是 off-policy。这里的 off-policy 不等于必须使用 replay。

$$
\delta_t^{Q}=R_{t+1}+\gamma\max_a Q(S_{t+1},a)-Q(S_t,A_t),\qquad Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\delta_t^Q
$$

### Expected SARSA 是第三种 target

如果知道下一动作的概率，可以对 Q 加权平均，代替采样一个 A′。这减少了下一动作采样这一项噪声，但不消除环境随机性。

$$
y_t=R_{t+1}+\gamma\sum_a\pi(a\mid S_{t+1})Q(S_{t+1},a)
$$

### 手算例子

R=0，γ=0.9，下一状态两个 Q 为 2、5。如果实际探索选了第一个动作，SARSA target 是 1.8，Q-learning 是 4.5。ε=0.1 的双动作 ε-greedy 下，Expected SARSA 为 0.9×(0.05×2+0.95×5)=4.365。

### 对照代码

```python
# q_step() 共用同一个更新；只改变 bootstrap target
boot = q[next_state][next_action]  # SARSA
# boot = max(q[next_state])       # Q-learning
target = reward if terminal else reward + gamma * boot
q[state][action] += alpha * (target - q[state][action])
```

读 q_step() 和 control()：SARSA 在更新前选择下一动作；Q-learning 用 max 构造 target。corridor() 决定奖励，切换标记不传入 q_step()。

```sh
python3 rl_foundations.py control --seeds 5 --episodes 800 --switch 400 --out control-run
```

生成回报、真实步数和更新次数三张图。第 400 个 episode 后左右终点收益交换；参数不清零。自然终止后位置重置，所以这是允许 episodic reset 的变化任务。

### 怎样接到 CRL

这组算法已经可以用于变化环境的基线。常数步长帮助更新旧估计，ε 帮助发现新收益；但是“知道变化时刻后手动清空 Q”是额外权限，不能偷偷加入。

适用条件：基础脚本不是 cliff-walking，也不保证展示 SARSA 与 Q-learning 的固定优劣排序。相同 episode 数下，轨迹长度可能不同，必须看真实交互量。

### 思考题

每步只用最新 transition 的 Q-learning，是 on-policy 还是 off-policy？

提示：看 target，而不是 buffer。

参考答案：仍然可以是 off-policy：行为含探索，target 却对下一动作取 max。数据是否回放与目标/行为策略是否一致，是两条不同的轴。

### 原始材料与代码

- [Sutton & Barto · 第 6 章](http://incompleteideas.net/book/the-book-2nd.html)：重点对照 SARSA、Q-learning 与 Expected SARSA 的备份目标。

---

## 03 当 Q 表放不下：DQN、Double DQN 与数据回放

先修：Q-learning；梯度下降与神经网络

问题：把表格换成网络，为什么还需要 target network 和 replay？

Q-learning → 共享参数 Qθ → DQN → Double DQN / 回报分布

表格更新只动一个格子；网络的一次更新可能同时改变许多状态动作的预测。再加上 target 依赖同一个网络、策略不断改变数据分布，训练不再像普通固定数据回归。DQN 的训练机制正是在处理这些耦合。

符号：θ：在线网络；θ⁻：滞后的 target 网络；d：真正终止标记；D：replay buffer；sg：停止梯度。

### 先看损失，再看两个稳定化设计

从 replay 采一批 transition，用滞后网络构造目标，再只更新在线网络。Replay 改变采样相关性并重复利用数据；target network 让目标短期内变化较慢。两者不是同一机制，也不构成一般稳定性证明。

$$
y=r+\gamma(1-d)\max_{a'}Q_{\theta^-}(s',a'),\qquad L(\theta)=\mathbb E_D\left[\tfrac12\big(Q_\theta(s,a)-\operatorname{sg}(y)\big)^2\right]
$$

### Double DQN：分开选动作与评动作

对带噪声的估计取 max，可能偏向恰好被高估的动作。Double DQN 用在线网络选动作，再用 target 网络评估该动作。它不是“两个 Q 取最小”，后者常见于 TD3/SAC。

$$
a^*=\arg\max_a Q_\theta(s',a),\qquad y_{\mathrm{Double}}=r+\gamma(1-d)Q_{\theta^-}(s',a^*)
$$

### 往外扩展时，分清改动对象

Prioritized replay 改采样分布，通常需要重要性权重控制偏差；dueling 改网络分解；C51 / QR-DQN 学回报分布。它们可以组合，但“预测回报分布”不等于已经识别环境变化或解决遗忘。

### 手算例子

下一状态在线网络 Q 为 (5,4)，target 网络为 (3,6)，r=0、γ=0.9。DQN target 是 5.4；Double DQN 先选在线网络偏好的第一个动作，再得到 2.7。两个更新公式的差异可以在一个数值例子里看到。

### 对照代码

```python
# 阅读伪代码；对应官方实现的 target / loss / optimizer 部分
with no_grad():
    next_action = online(next_obs).argmax(dim=1)
    next_q = target(next_obs).gather(1, next_action[:, None])
    y = reward + gamma * (1 - terminated) * next_q
loss = 0.5 * ((online(obs).gather(1, action) - y) ** 2).mean()
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

从 CleanRL 的 DQN 文档进入 dqn.py。先标记 replay 写入/采样、target 同步、终止处理，再看优化器。上面的片段是 Double DQN 的机制示意；不要把它当成 CleanRL dqn.py 的逐字摘录。

先在官方原任务跑通，再单独加入奖励变化。记录 buffer 中新旧数据比例；比较不同容量时要匹配梯度更新预算，而不仅是环境步数。

### 怎样接到 CRL

在 CRL 中，旧数据可能保护旧任务，也可能过时；replay 容量和采样策略因此成为研究对象。DQN 的回放不是为长期保留而自动设计好的。

适用条件：本章提供公式与官方深度实现入口；标准库教学包不实现 DQN。off-policy、函数逼近与自举的组合需要额外稳定性分析。

### 思考题

把 batch size 从 64 改成 1，就变成 streaming RL 了吗？

提示：样本来自新 transition 还是 replay？是否仍有 target network？

参考答案：不一定。若依旧从旧 buffer 采样，仍在回放。应报告数据重用、目标网络、更新频率与每步计算，而不是只报 batch size。

### 原始材料与代码

- [CleanRL DQN 文档与代码](https://docs.cleanrl.dev/rl-algorithms/dqn/)：先追一条 transition 的生命周期。
- [Double DQN 原论文](https://arxiv.org/abs/1509.06461)：对照动作选择与动作评估分离。
- [Spinning Up 算法分类](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)：定位价值学习与策略优化两条路线。

---

## 04 直接改进策略：REINFORCE、actor–critic、GAE 与 PPO

先修：TD；概率、对数与梯度

问题：不先对每个动作求最优 Q，能不能直接增加好动作出现的概率？

REINFORCE → baseline / actor–critic → GAE → TRPO → PPO

参数化策略 πθ 直接给出动作分布。核心问题变为：哪次行动让回报高于预期，就适当提高它在相似状态下的概率。先理解这个信用分配，再看 PPO 的裁剪；从复杂 loss 倒着背，往往会漏掉每个量从哪里来。

符号：τ：轨迹；J：期望回报；θ：策略参数；Vφ：critic；Â：估计优势；ρ：新旧动作概率比。这里先用有限 episode、γ=1 推导简单策略梯度。

### 为什么出现 log π

R(τ) 是整条轨迹回报，Gₜ 只保留从 t 起的奖励。轨迹概率由初始状态概率、策略概率与环境转移概率相乘。假设环境本身不依赖 θ，用 ∇p=p∇log p，将期望回报的梯度写成能用轨迹采样估计的形式。过去已发生的奖励不应给未来动作分信用，因此可以用 reward-to-go。

$$
\begin{aligned}\nabla_\theta J&=\sum_\tau p_\theta(\tau)R(\tau)\nabla_\theta\log p_\theta(\tau)\\\nabla_\theta\log p_\theta(\tau)&=\sum_t\nabla_\theta\log\pi_\theta(A_t\mid S_t)\\\nabla_\theta J&=\mathbb E\!\left[\sum_t\nabla_\theta\log\pi_\theta(A_t\mid S_t)G_t\right]\end{aligned}
$$

### 减去 baseline，再用 critic 近似

仅由状态决定、且在 actor 更新时停止梯度的 baseline，不改变期望策略梯度；它可减少方差。Actor–critic 再用价值函数自举，换来更及时的更新，也可能引入估计偏差。

$$
\mathbb E_{A\sim\pi}[\nabla_\theta\log\pi_\theta(A\mid s)b(s)]=b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)=0,\qquad \widehat A_t\approx\delta_t
$$

### GAE：把不同距离的 TD 误差组合起来

GAE 用衰减系数组织多个 TD residual。λ 控制偏差与方差的折中，不能凭“越长越好”选择。有限 rollout 末端还要处理 bootstrap 与终止。

$$
\delta_t=R_{t+1}+\gamma V_\phi(S_{t+1})-V_\phi(S_t),\qquad \widehat A_t^{\mathrm{GAE}}=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}
$$

### PPO：限制同一批数据上继续放大更新的激励

同一批 rollout 被多次优化后，策略已偏离采样策略。PPO-Clip 用概率比和裁剪后的保守目标控制更新；它不是严格的 KL 约束，也不保证每个动作概率都落在裁剪区间。

$$
\rho_t(\theta)=\frac{\pi_\theta(A_t\mid S_t)}{\pi_{\mathrm{old}}(A_t\mid S_t)},\quad L^{\mathrm{clip}}=\mathbb E\left[\min\big(\rho_t\widehat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\widehat A_t\big)\right]
$$

### 手算例子

两个动作概率均为 0.5，选中第二个且优势为 +1。softmax logits 的梯度是 (−0.5,+0.5)；步长 0.1 后 logits 从 (0,0) 变成 (−0.05,+0.05)，第二动作概率约 0.525。PPO 中若优势 +2、ρ=1.4、ε=0.2，该样本裁剪目标取 min(2.8,2.4)=2.4。

### 对照代码

```python
# 表格 softmax 的 log-prob 梯度；policy_step() 的实际更新
for j in range(len(logits)):
    logits[j] += alpha * advantage * ((j == action) - probabilities[j])
# REINFORCE: advantage = reward-to-go（此处不加 baseline）
# 一步 actor–critic: advantage = reward + V(next_state) - V(state)
```

标准库版 policy() 比较 REINFORCE 与一步 actor–critic，不实现 PPO。先定位采样时缓存的概率与更新时的梯度，再进入 CleanRL ppo.py，按 rollout → advantages → minibatches → losses → optimizer 的顺序读。

```sh
python3 rl_foundations.py policy --seeds 5 --episodes 800 --alpha 0.05 --switch 400 --out policy-run
```

看回报与起点向右的概率，而不只看 actor loss。环境变了以后，过于确定的策略可能很难再采到另一侧；这个现象不等同于深度网络可塑性损失。

### 怎样接到 CRL

PPO 是常用的 CRL 实验底座，但其批量 rollout 与多轮更新不自动满足严格 streaming。Critic、策略、归一化和优化器的长期状态，都可能影响后续适应。

适用条件：本章基础梯度推导使用 γ=1；折扣任务需统一状态访问权重与目标定义。用近似 critic、有限 GAE 或 PPO 裁剪后的优化，不再是同一个无偏 MC 梯度估计器。

### 思考题

优势为 −2、ρ=0.6、ε=0.2，PPO-Clip 为什么取 −1.6 而不是 −1.2？

提示：先计算两项，再取 min；不要把负优势按正优势理解。

参考答案：ρÂ=−1.2，clip(ρ)Â=0.8×(−2)=−1.6。目标不再鼓励继续把这个负优势动作的概率过度降低。

### 原始材料与代码

- [Spinning Up · 策略梯度推导与 PyTorch 小实现](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)：从轨迹概率开始，再对照代码。
- [GAE 原论文](https://arxiv.org/abs/1506.02438)：读多步 TD residual 的组合。
- [PPO 讲解](https://spinningup.openai.com/en/latest/algorithms/ppo.html)：比较正、负优势的裁剪。
- [CleanRL PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/)：追完整 rollout 与优化循环。

---

## 05 连续动作与数据复用：DDPG、TD3、SAC

先修：Q-learning、actor–critic、函数梯度

问题：动作是连续扭矩，无法枚举所有 a 取 max，怎么办？

连续动作的 max 难题 → DDPG → TD3 → SAC

可以学一个 actor 来近似挑出高 Q 的连续动作，同时用 Bellman target 训练 critic。这样把价值学习与策略优化接在一起。TD3 和 SAC 并非把 PPO 简单换一个损失，而是具有不同数据与更新协议的 off-policy actor–critic。

符号：μθ：确定性 actor；πθ：随机 actor；Qφ₁,Qφ₂：两个 critic；η：熵温度（避免与学习步长 α 混用）；D：回放数据。

### DDPG：让 actor 沿 critic 指出的方向走

固定 critic 时，actor 希望选出的动作有更高 Q。链式法则把 Q 对动作的梯度传回 actor。Critic 错误也可能被 actor 利用，因此 Q 的拟合误差不是单纯的预测误差。

$$
\nabla_\theta J\approx\mathbb E_{s\sim D}\!\left[\nabla_aQ_\phi(s,a)|_{a=\mu_\theta(s)}\;\nabla_\theta\mu_\theta(s)\right]
$$

### TD3：三处设计共同控制误差放大

两个 target critic 取较小值，降低某些过高估计；target action 加小扰动进行平滑；actor 相对 critic 延迟更新。这三项不能缩写成“多训练一个网络就够了”。

$$
y=r+\gamma(1-d)\min_{j=1,2}Q_{\phi_j^-}\big(s',\mu_{\theta^-}(s')+\text{clipped noise}\big)
$$

### SAC：把随机性纳入优化目标

SAC 在奖励之外加入策略熵，actor 既追求高 Q，也保留一定随机性。下面给出现代双 Q、无单独 V 网络的常见形式；a′ 从当前策略采样，critic target 停止梯度。

$$
\begin{aligned}y&=r+\gamma(1-d)\left[\min_jQ_{\phi_j^-}(s',a')-\eta\log\pi_\theta(a'|s')\right]\\L_\pi&=\mathbb E_{s\sim D,a\sim\pi_\theta}\left[\eta\log\pi_\theta(a|s)-\min_jQ_{\phi_j}(s,a)\right]\end{aligned}
$$

连续动作通常经过 tanh 压缩到动作范围，log probability 需要相应的变量变换修正。自动调温度是另一个优化问题，不是任意把熵系数设大。

### 手算例子

r=1、γ=0.9，两个 target Q 为 4、6，η=0.2，采样动作 logπ=−0.5，且未终止。SAC target 为 1+0.9×(4−0.2×(−0.5))=4.69。若真实终止，target 就是 1。

### 对照代码

```python
# 阅读伪代码：target 与 actor 的梯度路径不同
with no_grad():
    next_action, next_logp = actor.sample(next_obs)
    soft_q = min(target_q1(next_obs, next_action), target_q2(next_obs, next_action))
    y = reward + gamma * (1 - terminated) * (soft_q - temperature * next_logp)
# actor loss 中 action 必须保留重参数化梯度；不能整段放进 no_grad
action, logp = actor.sample(obs)
actor_loss = (temperature * logp - min(q1(obs, action), q2(obs, action))).mean()
```

从 CleanRL SAC 文档进入连续动作实现，依次查重参数化采样、tanh log-prob 修正、双 Q、target 更新与温度优化。上面是机制伪代码，张量逐元素 min 需用框架的运算。

先看终止处理与动作范围；再记录真实交互次数、梯度更新次数与 buffer 容量。改成任务序列时，明确 replay、温度和归一化是否跨任务保留。

### 怎样接到 CRL

Continual World 等任务序列研究常以 SAC 为底座。更大的旧数据池不一定更好：不同任务奖励或动力学冲突时，需要明确是否提供任务信息，以及旧数据如何被使用。

适用条件：熵鼓励随机性，不保证发现所有新奖励；双 Q 也不保证不存在偏差。此处给官方深度代码入口，未将 SAC 包装成已完成的 CRL 算法。

### 思考题

SAC 的两个 Q 取 min，与 Double DQN 的“Double”是同一个公式吗？

提示：分别写出谁选动作、谁评估、最终怎样聚合。

参考答案：不是。Double DQN 分开动作选择与评估；TD3/SAC 通常对两个 critic 的估计取较小值。两者都涉及多个估计器，但运算和偏差控制方式不同。

### 原始材料与代码

- [Spinning Up · TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html)：按三项设计逐一理解。
- [Spinning Up · SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)：先确定 soft value 的定义，再看损失。
- [CleanRL · SAC 实现](https://docs.cleanrl.dev/rl-algorithms/sac/)：从采样和 target 梯度边界开始读。
- [Continual World 官方代码](https://github.com/awarelab/continual_world)：观察 SAC 如何接到任务序列与不同 CL 方法。

---

## 06 从经验多学几次：模型、Dyna 与规划

先修：Q-learning；状态转移

问题：一条昂贵的真实经验，能否通过模型帮助更多价值更新？

真实 transition → 学习转移与奖励模型 → 模拟 transition → 规划更新

Model-free 不显式用环境模型做规划，不意味着没有任何预测。Dyna 的特别之处是同时学价值与模型：真实经验更新 Q，也更新模型；之后从模型生成虚拟经验，再交给同一个价值学习器。

符号：M(s,a)：学得的后继与奖励模型；k：每个真实步之后的规划更新数；θ：价值参数。下面的教学模型只适用于确定性环境。

### 把三个学习对象分开

行为策略决定采什么数据，价值函数决定当前偏好，模型预测动作后果。一个新奖励可以经真实 backup 改变局部 Q，也可以经模型上的多个 backup 传播到未实际重访的前序状态。

$$
(\hat s',\hat r,\hat d)\leftarrow M(s,a),\qquad Q(s,a)\leftarrow Q(s,a)+\alpha[\hat r+\gamma(1-\hat d)\max_bQ(\hat s',b)-Q(s,a)]
$$

### 模型更新与规划计算是两种成本

每个真实步之后增加 k 次模拟更新，可能提高样本效率，却用掉更多计算。报告中同时画真实环境步数与总更新量；只按 episode 比较容易把额外算力当成算法优势。

### 从 Dyna 接到更广的 model-based RL

Prioritized sweeping（优先更新） 选择最值得传播的 backup；MPC 每次用模型规划短期动作序列；学习潜在动力学的世界模型可用于想象训练。这些方法都涉及模型，但模型表示、如何使用、误差传播方式不同。不能把它们理解为一条自动升级的版本链。

### 手算例子

一个通道原来通向奖励 1，后来奖励变为 0。真实动作尚未再到终点时，模型仍可能保存奖励 1；做更多规划只会更自信地重复旧预测。问题可能是缺少新数据，而不是规划次数不够。

### 对照代码

```python
# control(..., experiment='dyna') 中的核心流程
q_step(q, s, a, r, ns, done, alpha, gamma)      # 真实更新
model[s, a] = (ns, r, done)                    # 学到的最近后果
for _ in range(planning):
    ms, ma = sample_seen_state_action(model)
    ns, r, done = model[ms, ma]
    q_step(q, ms, ma, r, ns, done, alpha, gamma)  # 模型更新
```

标准库包里的 control() 同时提供无规划 Q-learning 与 Dyna-Q，规划随机数单独管理；模型不会提前获知奖励切换。样例片段中 sample_seen_state_action 对应实际代码的 plan_rng.choice。

```sh
python3 rl_foundations.py dyna --seeds 5 --episodes 800 --switch 400 --planning 10 --out dyna-run
```

把 --planning 改为 0，dyna_q 应逐点等于 q_learning。再增加规划数，观察变化前与变化后的差异，并核对 update_count 的增长。

### 怎样接到 CRL

CRL 的模型必须跟踪世界，也可能需要跟踪自身不断改变的技能。研究问题可以先缩小到“旧模型怎样拖慢适应”，再扩展到抽象技能模型。

适用条件：脚本存最近一次确定性转移，不估计概率模型；允许 episode reset，不是 single-life。Dyna-Q+ 的探索奖励未在此实现。

### 思考题

把规划次数翻倍后样本效率提高，能否说计算效率也提高？

提示：分别统计真实交互、模型调用与价值更新。

参考答案：不能直接说。样本效率、计算量和真实时间是不同指标；在机器人动作 deadline 下还需测每步延迟。

### 原始材料与代码

- [Sutton & Barto · 第 8 章](http://incompleteideas.net/book/the-book-2nd.html)：先看 Dyna，再比较环境变化与 Dyna-Q+。
- [Spinning Up · model-based 分类](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)：区分 MPC、模型生成训练数据与规划嵌入策略。

---

## 07 积累可复用知识：Options、SR、Successor Features 与 GPI

先修：Bellman、Dyna、向量内积；先看入门第 6 课

问题：换了奖励以后，哪些行为结构可以不从头学习？

Options：行为跨多步 → 技能模型：预测后果 → SR / SF：未来占用 → GPI：复用多种策略

这里有两条相交但不同的线：options 把多步行为变成决策单位；SR/SF 把环境中的未来占用与奖励权重分开。Machado 的技能发现工作会利用表示来构造探索技能，但不能把 option、SR 和规划混为一个对象。

符号：o=(I,π,β)：启动集合、内部策略和终止函数；τ：技能持续时间；φ：转移特征；w：奖励权重；ψπ：策略 π 下的折扣特征累计。

### Option 的后续价值按真实持续时间折扣

执行一个技能会跨越 τ 个基本时间步。把这段内部奖励先累计，再在技能结束后选下一个可用技能。Q 在这里指给定技能集合上的最优技能价值。

$$
Q(s,o)=\mathbb E\left[\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}+\gamma^\tau\max_{o'}Q(S_{t+\tau},o')\mid S_t=s,o\right]
$$

### SR 与 SF：先预测会遇到什么，再赋予价值

SR 预测状态的折扣访问次数；SF 将它推广为特征的折扣累计。若即时奖励能写为 φᵀw，就可将价值分解为未来特征与奖励权重的内积。

$$
r(s,a,s')=\phi(s,a,s')^\top w,\quad \psi^\pi(s,a)=\mathbb E_\pi\left[\sum_{k\ge0}\gamma^k\phi_{t+k+1}\right],\quad Q_w^\pi(s,a)=\psi^\pi(s,a)^\top w
$$

### GPI：在已有策略的预测中选择动作

保存若干策略的 successor features 后，对新奖励 w 重新评估它们。GPI 对每个动作取已有策略预测中的最好值，再选动作。改进保证需要任务共享结构与估计精度等条件，不能用一个 max 省略这些假设。

$$
\pi_{\mathrm{GPI}}(s)\in\arg\max_a\max_i\left[\psi^{\pi_i}(s,a)^\top w\right]
$$

### 手算例子

未来特征的预测为 ψ=(2,1)。旧奖励权重 w=(1,0)，价值为 2；新权重 (0,3)，价值变为 3，不必为这个固定策略重新采完整回报。但若道路结构也改变，ψ 本身可能已经失准。

### 对照代码

```python
# SF 的 on-policy 一步更新：向量 TD
target = phi if terminal else phi + gamma * psi[next_state, next_action]
psi[state, action] += alpha * (target - psi[state, action])
q_for_new_reward = psi[state, action] @ new_reward_weights
# Options 还需要 initiation / intra-option policy / termination 与技能模型
```

先读 Machado options 仓库的 main.py 和 README，识别环境、option 构造与策略执行；再对照课程 slides 中的表示驱动技能发现。SF 片段是本页的机制示意，不声称该仓库实现了这里全部 SF/GPI 流程。

先在小网格中只改变奖励，再改变动力学。比较“只改 w”“更新 ψ”“重新学习”三种条件；否则不知道迁移失败在哪个分解假设。

### 怎样接到 CRL

这是从“适应变化”走向“知识积累”的重要一支：不仅防止旧能力丢失，还要让既有结构帮助未来任务。但技能数量增加不等于规划收益增加。

适用条件：经典 SF 的便捷重估依赖奖励表示与共享动力学。Option 的发现、执行、模型学习和规划应分别测量；未实现的环节不由表示图自动补齐。

### 思考题

世界动力学不变，但用于计算 ψ 的策略已经更新，旧 ψ 必然仍准确吗？

提示：ψ 上标是 π。

参考答案：不必然。它预测的是特定策略下的未来特征分布；策略变化也可能使它过时。可以保留旧策略及其 ψ，或跟踪更新新的 ψ。

### 原始材料与代码

- [SF 与 GPI 原论文](https://arxiv.org/abs/1606.05312)：抓住奖励分解及共享环境假设。
- [Machado 课程与 slides](https://deeprlcourse.github.io/guests/marlos_machado/)：先读 option 定义，再读表示如何用于发现技能。
- [Machado options 代码](https://github.com/mcmachado/options)：从 main.py 找到运行入口，按 README 建独立环境。

---

## 08 怎样保留旧知识：Replay、EWC、蒸馏与参数隔离

先修：梯度下降；区分旧任务评测与新任务学习

问题：为什么一个保护旧任务的方法，会让新任务更难学？

Fine-tuning 基线 → Replay：保留数据 → EWC：限制参数 → 蒸馏 / 参数隔离

先明确旧知识仍然值得保留。若只是同一任务的旧奖励规律已经失效，强行保留它可能妨碍适应。若未来还会回访旧任务，保留才是明确目标。不同方法保护的对象不同，需要对应的 memory、task ID 与计算权限。

符号：L_B：新任务损失；θ_A：学完旧任务的参数；Fᵢ：旧任务参数重要性的对角近似；κ：保护强度；β：旧数据损失权重。

### Replay：让旧样本继续参与优化

概念上可写成新旧数据损失的加权和。RL 里旧 transition 来自过去策略；如何构造当前 target、做 off-policy 校正或处理过时奖励，需要看具体底座，不能直接照搬监督混合数据。

$$
L(\theta)=(1-\beta)\mathbb E_{D_B}\ell(\theta)+\beta\mathbb E_{D_A}\ell(\theta)
$$

### EWC：对重要参数加弹簧

把旧任务解附近的损失/后验近似成二次形式，保留中心 θ_A 和重要性。常见 EWC 用对角 Fisher 近似；这是局部近似，不是所有参数移动都等价于遗忘。

$$
L(\theta)=L_B(\theta)+\frac\kappa2\sum_iF_i(\theta_i-\theta_{A,i})^2,\quad \nabla_iL=\nabla_iL_B+\kappa F_i(\theta_i-\theta_{A,i})
$$

### 蒸馏与参数隔离保护的是另外两件事

蒸馏要求新网络在选定输入上保留旧输出；没覆盖到的状态仍可能丢失。Progressive networks 或 PackNet 一类方法保留/划分参数，换来容量增长、掩码或任务路由等成本。Progress & Compress 则把快速适应与知识整合分开。它们不应只用一个最终分数比较。

### 手算例子

标量旧任务希望 w=+1，新任务希望 w=−1，F=1。EWC 目标为 ½(w+1)²+κ/2(w−1)²，最优解 w*=(κ−1)/(κ+1)。κ=0 得到 −1；κ=1 得到 0；κ 越大越靠近旧解，也越难满足新目标。

### 对照代码

```python
# consolidation() 中的已知曲率教学例子，不是神经网络 Fisher 估计
gradient = w + 1                           # 新目标 L_B 的梯度
gradient += strength * (w - 1)             # F=1，旧解为 +1
w -= 0.05 * gradient
# equal_rehearsal 则用 0.5*(w+1) + 0.5*(w-1)
```

先跑标量例子看 trade-off，再读 Continual World 的 CL 方法实现，或 AGI-Labs continual_rl 的策略接口、任务序列与评测。不要把“允许访问旧数据”和“不给旧数据”放在一张无协议说明的表里。

```sh
python3 rl_foundations.py consolidation --seeds 1 --strength 1 --out retention-run
```

分别画 old_loss 和 new_loss；把 --strength 改为 0、1、5，检查稳态是否接近解析解。它是确定性例子，不用多 seed 制造重复证据。

### 怎样接到 CRL

CRL 中不仅有遗忘，也有合理丢弃过时信息的需求。研究一个 retention 方法前，先决定旧任务何时会回来、是否可评估，以及 task ID 是否可用。

适用条件：这个标量实验是监督二次损失类比，不是 EWC 的完整 RL 复现。现实网络的 Fisher 估计、任务边界和多任务重要性累积都需要额外实现。

### 思考题

在 κ=1 的例子里，EWC 与等权 rehearsal 最优解相同，学习曲线也必然相同吗？

提示：比较两者的梯度幅度，而不只看最优点。

参考答案：不必然。EWC 梯度为 2w，等权 rehearsal 为 w。相同步长下更新速度不同；目标的整体缩放与学习率也会影响比较。

### 原始材料与代码

- [EWC 原论文](https://arxiv.org/abs/1612.00796)：读参数重要性和旧任务附近的近似。
- [Progress & Compress](https://arxiv.org/abs/1805.06370)：理解快速适应与慢速整合的分工。
- [Continual World](https://github.com/awarelab/continual_world)：查看 task sequence、CL 方法与 SAC 底座。
- [Continual RL 基线库](https://github.com/AGI-Labs/continual_rl)：追策略接口、实验定义与公共指标。

---

## 09 怎样一直学得进：ReDo、Continual Backprop 与特征更新

先修：梯度与神经网络；入门第 3 课的 old/fresh 诊断

问题：旧任务没忘，不代表还有能力学新内容；这种能力怎样维持？

先诊断学习能力 → ReDo：低活跃单元 → CBP：低效用特征 → 长期对照与知识积累

Retention 问“以前会的还会不会”，plasticity 问“接下来还学不学得进”。它们可能冲突，也可能同时受损。重置特征的一类方法尝试保留有用知识，同时为新学习留出可调整的表示。首先要分清机制信号与任务表现。

符号：hᵢ(x)：单元激活；H：该层单元数；sᵢ：归一化活跃度；uᵢ：效用统计；ρ：效用衰减系数。效用定义随算法版本而异。

### ReDo 先识别相对低活跃的单元

将一个单元平均绝对激活除以层平均，可以比较同一层中的相对活跃程度。达到低活跃阈值后，重置输入连接、控制输出连接，使新特征重新有机会参与。分母接近零时，实际代码还需要数值处理。

$$
s_i=\frac{\mathbb E_x|h_i(x)|}{H^{-1}\sum_{j=1}^H\mathbb E_x|h_j(x)|}
$$

### Continual Backprop：普通学习之外持续生成与筛选

除了 backprop 更新，还持续估计特征效用；只在达到一定年龄的特征中替换少量低效用单元，避免新单元还没学就被淘汰。效用可以涉及激活及输出贡献，不能把它直接等同于 ReDo 的低活跃度。

$$
u_{i,t}=\rho u_{i,t-1}+(1-\rho)c_{i,t}
$$

这只是指数平均的记账形式；cᵢ 的具体定义、偏差修正、年龄与替换预算必须回到对应版本论文/实现。不是用这一式就复现了 CBP。

### 重置一个神经元，也涉及优化器状态

需要一起检查输入权重、输出权重、偏置、Adam 等优化器统计与替换顺序。若只改权重而沿用不匹配的旧动量，后续更新可能立即改变新特征。还要测原任务是否被破坏，而不是只数有多少单元重新活跃。

### 手算例子

一个单元的平均绝对激活为 0.01，所在层平均为 0.5，相对活跃度为 0.02。这说明它在这批输入上低活跃，不说明它在所有稀有状态中无用。替换后的新任务收益和旧任务损失必须另外测量。

### 对照代码

```python
# 机制伪代码；不是任一论文全部实现
normal_gradient_update(batch)
update_feature_statistics(activations, outgoing_weights)
eligible = units_with_sufficient_age()
chosen = select_low_utility(eligible, replacement_budget)
reinitialize_incoming_weights(chosen)
reset_outgoing_weights_and_optimizer_state(chosen)
```

作者 loss-of-plasticity 仓库：lop/algos 看学习与特征更新，lop/nets 看网络，lop/slowly_changing_regression 看小任务，再进 lop/rl。先核对具体版本的效用定义。ReDo 从论文及其代码入口比较，不把两种选择规则混写。

最小对照包括不替换、匹配频率的随机替换、效用替换；另设 fresh reference。匹配网络、数据和学习率调参预算，测新任务曲线、旧任务保留及替换成本。

### 怎样接到 CRL

它与 streaming 可组合，但一次重置并不解决在线信用分配，也不自动形成知识积累。前沿问题是何时替换、替换什么，以及如何避免持续探索与保护旧知识互相干扰。

适用条件：低活跃、低秩、梯度变小都是诊断线索，不能代替可学习性指标。这里提供作者研究代码，不用一个表格/线性小实验声称复现深度可塑性。

### 思考题

新方法让 dormant neuron 比例下降，回报却不变，应怎样总结？

提示：测量量与科学主张是否对应？

参考答案：可以说改变了活跃度统计；还不能说改善了控制或长期学习能力。检查新任务学习、旧知识和不同任务分布，保留没有收益的结果。

### 原始材料与代码

- [ReDo · ICML 原论文与入口](https://proceedings.mlr.press/v202/sokar23a.html)：重点看低活跃度定义及替换过程。
- [Loss of Plasticity · Nature](https://www.nature.com/articles/s41586-024-07711-7)：从实验设置和新学习曲线理解现象。
- [Continual Backprop 作者代码](https://github.com/shibhansh/loss-of-plasticity)：按 algos → 小回归任务 → RL 阅读。

---

## 10 逐步学习：资格迹、信用分配与 Stream-X

先修：TD、actor–critic、向量梯度

问题：没有 replay 和大 batch，怎样在每个新样本后稳定地更新？

增量 TD / traces → 逐参数步长 → 深度 streaming 实现

Streaming 约束数据与更新方式；average reward 改变优化目标；single-life 约束环境重置。这三个概念互不等价。先用线性预测理解迹与步长，再看深度 streaming 代码如何处理输入、奖励尺度和更新幅度。

符号：xᵢ：特征；wᵢ：权重；δ：预测误差；βᵢ=log αᵢ；hᵢ：权重对 log 步长的近似敏感度；μ：meta 步长；r̄：平均奖励估计。

### 资格迹解决时间信用，不是自动步长

e 累积过去特征的责任；α 决定给定责任下实际更新多少。二者可以一起使用，但一个大的 trace 也可能放大更新，所以需要看特征尺度与控制方法。

$$
e_t=\gamma\lambda e_{t-1}+x_t,\qquad w_{t+1}=w_t+\alpha\delta_t e_t
$$

### IDBD：不同参数能否自动学得快慢不同？

在线性监督预测中，让每个参数有自己的 log 步长，利用误差与过去更新敏感度调节 β。下面是带非负因子的常见线性近似：先更新 β，再算 α、w 和 h，所有右侧误差来自更新前预测。

$$
\begin{aligned}\beta_i&\leftarrow\beta_i+\mu\delta x_i h_i,\quad\alpha_i=\exp(\beta_i)\\w_i&\leftarrow w_i+\alpha_i\delta x_i\\h_i&\leftarrow h_i\max(0,1-\alpha_i x_i^2)+\alpha_i\delta x_i\end{aligned}
$$

现有 credit 教学实验还对 β 做了数值裁剪。这不是 NetworkIDBD，也不能不改推导就用于所有 TD/actor–critic 目标。

### 长期运行也可以用平均奖励目标

折扣目标不是唯一选择。平稳、适当遍历条件下，可研究单位时间平均奖励及相对价值；其 TD 误差减去平均奖励估计。非平稳世界通常还需要有限窗口或动态目标，不能直接假定单一稳态平均存在。

$$
\bar r^\pi=\lim_{T\to\infty}\frac1T\mathbb E_\pi\sum_{t=1}^TR_t,\qquad\delta_t=R_{t+1}-\widehat{\bar r}_t+v(S_{t+1})-v(S_t)
$$

### 从线性更新进入 Stream-X

阅读官方实现时分开检查归一化、初始化/表示、梯度与参数更新控制，以及 actor–critic 顺序。省去 replay 与 target network 以后，这些组件会改变训练动力学；删除 buffer 本身不是完整方法。

### 手算例子

x=1、δ=2、旧 h=0.5、μ=0.01，则 β 增加 0.01，步长乘以 exp(0.01)≈1.01005。它是“小幅调整未来学习速度”，不是把当前权重直接加 0.01。

### 对照代码

```python
# 已有 crl_labs.py 的 credit()：先从线性预测理解
beta[i] += meta * delta * x[i] * h[i]
rate = exp(beta[i])
w[i] += rate * delta * x[i]
h[i] = h[i] * max(0, 1 - rate*x[i]*x[i]) + rate*delta*x[i]
# 深度 actor–critic 请回到完整实现，不直接粘贴此监督更新
```

先读 crl_labs.py 的 credit()；再进入 Stream-X 的 stream_ac_discrete.py、optimizer.py 和 obs_reward_transforms.py，逐段追 act → transition → update。平均奖励实现另看 Naik 等的仓库，不与 streaming 当作同一算法。

```sh
python3 crl_labs.py credit --seeds 5 --steps 4000 --out credit-study.csv
```

改变干扰特征数和噪声，观察信号误差与各类特征步长。SGD 与 IDBD 都要获得合理调参预算；固定一个默认步长的差异不能直接证明自动步长普遍更好。

### 怎样接到 CRL

这是通向 Oak 的选择性学习、在线 meta-gradient、Stream-X、实时递归学习的基础接口。先明确你改的是信用、尺度、参数步长还是更新预算。

适用条件：线性 IDBD、平均奖励 TD 和深度 Stream-X 的条件不同。本章把它们连成学习路线，不宣称共享同一个收敛定理。

### 思考题

一个算法没有 replay，但每次失败后 reset，能否称为 single-life？

提示：数据权限与环境权限分别检查。

参考答案：不能仅凭无 replay 判断。它可满足 streaming 更新协议，却仍然使用 episodic reset；single-life 需要另外说明失败与恢复机制。

### 原始材料与代码

- [IDBD 原论文](https://cdn.aaai.org/AAAI/1992/AAAI92-027.pdf)：读线性步长的敏感度近似。
- [Stream-X 官方代码](https://github.com/mohmdelsayed/streaming-drl)：从离散动作单文件和 optimizer 开始。
- [平均奖励方法代码](https://github.com/abhisheknaik96/average-reward-methods)：用小任务区分平均奖励与折扣价值。

---

## 11 学什么状态：GVF、Horde、RNN 与实时递归信用

先修：TD、内部状态与参数的区别

问题：相同当前图像需要不同动作时，如何让历史成为有用状态？

观测不够 → 递归状态 / 预测状态 → GVF 与 Horde → BPTT / RTRL / RTU

一个大网络不能从没有的信息里恢复答案。首先要允许智能体把历史带到当前。其次要决定如何训练这些历史摘要：可以直接为控制学习递归表示，也可以学习对未来信号的预测。两种路线有交集，但不是同义词。

符号：zₜ：递归内部状态；θ：产生状态的参数；C：cumulant（要预测的信号，不必是任务奖励）；γₜ：延续系数；π：预测对应的策略。

### RNN 更新状态，与参数学习是两个时间过程

每步用新观测、前一动作和旧内部状态产生 z。即使 θ 暂时不变，z 也会改变；而训练 θ 决定系统将来怎样记忆。必须说明状态在 episode、task 与生命期之间何时保留。

$$
z_t=f_\theta(z_{t-1},O_t,A_{t-1}),\qquad A_t\sim\pi_\theta(\cdot\mid z_t)
$$

### GVF：用价值学习表达“将来会怎样”

把普通奖励换成感兴趣的信号，并明确预测策略与停止/折扣条件，就得到更一般的预测问题。例如继续向前走，未来电量或碰撞信号怎样累计。Horde 将许多这样的预测学习器组织在同一经验流上。

$$
v(s)=\mathbb E_\pi\left[\sum_{k\ge0}\left(\prod_{j=1}^{k}\gamma_{t+j}\right)C_{t+k+1}\mid S_t=s\right]
$$

k=0 的空乘积为 1。若行为策略不是预测策略，不能直接假设普通 on-policy TD 安全；需要考虑 off-policy 学习算法与条件。

### 历史梯度怎样到达参数：BPTT 与 RTRL

BPTT 保存或重建一段计算图向后传播，截断会丢掉窗口外的梯度路径。RTRL 向前维护状态对参数的敏感度，但一般稠密递归网络成本很高。RTU 等利用结构约束降低成本，不是任意 RNN 的免费精确梯度。

$$
H_t=\frac{\partial z_t}{\partial\theta}=\frac{\partial f_\theta}{\partial z_{t-1}}H_{t-1}+\frac{\partial f_\theta}{\partial\theta}
$$

这里沿给定输入序列求递归导数；实际 RL 的数据生成与策略梯度还有额外问题。

### 手算例子

进门看到红灯，随后十步走廊完全相同，最后需要选左。手工保存灯色即可解决信息问题；如果 RNN 没学到这点，可能是训练目标或长程信用分配失败，而不是任务本身不可解。

### 对照代码

```python
# 状态与参数的生命周期要分别追踪
hidden = recurrent_cell(obs, previous_action, hidden)
prediction = prediction_head(hidden)        # 可预测 cumulant
action = policy_head(hidden).sample()
# detach(hidden) 截断梯度历史，不会自动把 hidden 的数值清零
# reset hidden / optimizer / replay 是三个不同操作
```

先运行现有 memory 诊断确认信息条件；再读 RTU 作者仓库中的递归单元和更新循环。Horde 原论文说明预测问题与学习器组织，不能用一段 RNN forward 代码替代它。

```sh
python3 crl_labs.py memory --seeds 1 --steps 1000 --out memory-study.csv
```

手工 oracle 应达到 1，无记忆固定策略在平衡线索中为 0.5。然后才研究“学习出的记忆”，并匹配历史、状态维度和每步计算；不能把 oracle 当成训练 RNN 的结果。

### 怎样接到 CRL

连接 Forager、POPGym、预测式 agent state 与实时递归学习。持续学习中，不仅世界变化，策略变化也会改变需要记住和预测的对象。

适用条件：多个预测准确，不保证它们足够支持控制；工作记忆丰富，也不自动代表长期知识持续增长。先为你的主张选择对应的测试。

### 思考题

对 hidden 做 detach 与重置为零，对行为和梯度有什么不同？

提示：区分数值和计算图。

参考答案：detach 保留当前数值、切断跨边界梯度；置零改变内部信息，可能立刻改变行为。两者都影响学习实验，但不是同一权限。

### 原始材料与代码

- [Horde 原论文](https://josephmodayil.com/papers/horde-final.pdf)：先看一般预测问题，再看多学习器架构。
- [RTU 原论文](https://arxiv.org/abs/2409.01449)：追实时梯度的结构假设。
- [RTU 作者代码](https://github.com/esraaelelimy/rtus)：定位递归状态与梯度计算。

---

## 12 Meta-learning：步长、初始化与学习规则

先修：链式法则、梯度下降；进阶部分需要策略梯度

问题：改变一次更新，怎样影响后续学习？

IDBD：在线步长适应 → MAML：可适应初始化 → Meta-gradient RL：更新超参数 → DiscoRL：学习更新规则

元学习改变学习过程，但不同方法更新的对象和可用经验并不相同。先从两步标量学习推导敏感度，再区分在线调整步长、跨任务学习初始化和离线学习更新规则。上下文适应、迁移与自动课程是相邻路线，需要分别说明协议。

符号：θ：共享初始化；Lᵢᴬ：任务 i 的适应损失；LᵢQ：适应后的评估损失；α：内部步长；p(i)：任务分布。RL 中常把负回报或策略代理目标写成损失。

### MAML：优化经过学习后的表现

先在任务 i 上用适应数据走一步，再用分离的评估数据衡量更新后的参数。外层不是只优化当前 θ 的分数，而是优化“从 θ 开始学一次”的结果。

$$
\theta_i'=\theta-\alpha\nabla_\theta L_i^A(\theta),\qquad\min_\theta\;\mathbb E_{i\sim p(i)}L_i^Q(\theta_i')
$$

### 为什么会出现二阶项

更新后的 θ′ 也依赖 θ，对外层目标应用链式法则，就有内部更新映射的导数。一阶 MAML 近似忽略这一 Hessian 项；它是近似，而不是完全相同的目标梯度。

$$
\nabla_\theta L_i^Q(\theta_i')=\left(I-\alpha\nabla_\theta^2L_i^A(\theta)\right)^\top\nabla_{\theta_i'}L_i^Q(\theta_i')
$$

### RL² 与自动课程分别改变什么

RL² 用跨交互保留的递归状态实现任务内适应，慢速外层 RL 训练其行为规则。自动课程则选择任务、目标或环境参数，让学习器获得更有用的经验。学习进展估计要区别于原始预测误差：不可约噪声很大，也可能几乎没有可学进展。

### 与终身 CRL 的接口

Meta-test 常从同一个初始化重启，CRL 则可能要求保留整段生命史；课程 teacher 也可能知道任务 ID、真实难度或成功标签。先列权限，再讨论迁移收益和长期知识积累。

### 手算例子

两个任务各自适应后都很好，但每次都从 θ 重启，这说明初始化便于适应；不能据此声称一位 agent 连续学习任务后越学越强。若每个任务都重置 hidden state，也要说明 RL² 允许记忆跨越哪些 episode。

### 对照代码

```python
# MAML 机制伪代码；support 与 query 数据必须分开
adapted = theta - inner_lr * grad(adaptation_loss(task, theta))
meta_loss = evaluation_loss(task, adapted)
meta_gradient = differentiate_through_update(meta_loss, theta)
# RL²: 重点检查 recurrent state 跨 episode 的保留与 task 边界
# Curriculum: 重点检查 teacher 选择任务时能看到什么指标
```

RL² 可用 garage 文档定位训练循环、任务采样与 recurrent policy；它是第三方参考实现。TeachMyAgent 提供课程算法与测试环境入口。分别读 teacher 和 student，不把环境生成器当成 agent。

划分训练任务与未见评测任务；匹配适应交互预算；增加无迁移初始化、随机课程等基线。另画不重启学习器的持续任务流，才能讨论它是否帮助 CRL。

### 怎样接到 CRL

与技能迁移、开放式课程、多智能体对手变化相接。下一步可沿研究地图的元学习/课程分支选一个协议，而不是把全部机制一次塞入同一 agent。

适用条件：MAML 的经典任务分布假设、RL² 的记忆协议和开放式环境生成不是同一个问题。这里的二阶公式先在可微损失下解释；完整 RL meta-gradient 还涉及采样分布。

### 思考题

课程 teacher 总挑预测误差最大的任务，为什么可能一直挑随机噪声？

提示：误差很大与误差正在下降是不同量。

参考答案：不可约噪声可以持续产生大误差，但训练未必带来进步。应衡量可学习的进展、任务多样性和覆盖，并与随机课程比较。

### 原始材料与代码

- [MAML 原论文](https://arxiv.org/abs/1703.03400)：读“适应之后”的外层目标。
- [RL² 原论文](https://arxiv.org/abs/1611.02779)：关注慢速参数学习与快速递归适应。
- [garage · RL² 实现说明](https://garage.readthedocs.io/en/latest/user/algo_rl2.html)：第三方实现，先按其版本文档建立环境。
- [Narvekar 等课程学习综述](https://www.jmlr.org/papers/v21/20-212.html)：区分任务生成、排序和知识迁移。
- [TeachMyAgent 官方代码](https://github.com/flowersteam/TeachMyAgent)：找到 teacher/student 边界与基线。

---

## 13 世界会变化，为什么还要探索：计数奖励、RND 与学习进展

先修：ε-greedy、价值学习、预测误差

问题：已经很会做当前任务的 agent，怎样发现以前没有价值的新机会？

ε-greedy：保留尝试 → 访问计数：奖励少见 → RND：特征预测新奇 → 学习进展 / 技能探索

探索决定未来会获得什么经验，因此影响所有后续学习。持续世界里，曾经熟悉的状态也可能产生新奖励；只奖励“从没见过”还不够。这里把基础随机探索、内在奖励和自动课程联系起来，同时区分它们实际测量的东西。

符号：Nₜ(s)：到 t 为止的访问次数；bₜ：内在奖励；β：探索奖励权重；f：固定随机目标网络；f̂θ：可学习预测网络；ℓ：预测损失。

### ε-greedy 很简单，却提供不可缺少的对照

每一步保留 ε 的随机动作概率，可以持续尝试非贪心动作。但一步随机不保证探索到长距离目标；若去新区域需要十步协调行动，局部噪声可能效率很低。SAC 的熵奖励也不能自动代替时间上连贯的探索。

### 计数奖励：少见的状态值得再看

一个常见教学形式是访问越少、额外奖励越大。深度观测中精确计数不实用，可以使用密度模型或表示近似新奇度；伪计数并非简单统计图像哈希次数。

$$
\widetilde R_{t+1}=R_{t+1}^{\mathrm{ext}}+\beta b_t(S_{t+1}),\qquad b_t(s)=\frac{1}{\sqrt{N_t(s)+1}}
$$

这是解释计数奖励的简化形式，不是所有算法共享的精确公式。必须同时报告外在回报，避免只优化了自己加入的奖励。

### RND：预测一个固定随机网络的输出

目标网络 f 初始化后固定，预测器 f̂θ 在访问过的观测上学习拟合它。预测误差作为新奇信号；它不直接预测随机的下一帧，所以要区别于动力学预测误差。不过随机观测、泛化和预测器遗忘仍可能使新奇信号失真。

$$
b_t(o)=\left\|\widehat f_\theta(o)-f(o)\right\|_2^2,\qquad\theta\leftarrow\theta-\alpha\nabla_\theta b_t(o)
$$

原始 RND 还涉及归一化、外在/内在价值估计及训练细节；一行平方误差不是完整算法。

### 持续探索：从“误差大”到“还有进步空间”

学习进展比较一段时间内表现或预测损失的变化。比较时应控制评测分布；否则误差下降也可能只是采到了容易的状态。自动课程选择任务，options 组织连贯行为，它们和内在奖励是可以组合但需要分别检验的组件。

### 手算例子

状态已经访问 99 次，计数奖励为 0.1；一个新状态的奖励为 1。现在旧状态的外在奖励突然提高，历史计数仍不会自动增加它的新奇度。若探索完全停止，agent 甚至没有数据发现这次改变。

### 对照代码

```python
# RND 机制伪代码；target 参数固定，只更新 predictor
with no_grad():
    target_features = target_network(obs)
error = ((predictor(obs) - target_features) ** 2).sum(dim=-1)
intrinsic_reward = error.detach()
loss = error.mean()
predictor_optimizer.zero_grad()
loss.backward()
predictor_optimizer.step()
# 交给控制算法时，外在回报与内在奖励分别记录
```

先用 control() 比较持续探索概率，理解没看到新奖励时再好的更新也无从下手。然后读 OpenAI RND 的历史参考实现：它已归档，需按 README 另建兼容环境，不与教学包混装。

```sh
python3 rl_foundations.py control --seeds 5 --episodes 800 --switch 400 --epsilon 0.3 --out exploration-run
```

与 --epsilon 0.1 的 control-run 比较变化前回报、变化后恢复及真实交互量。再去 RND 检查目标网络是否真的冻结、predictor 的训练输入、奖励归一化与各自的 value head。

### 怎样接到 CRL

连接 Machado 的技能发现、自动课程和开放式学习。可从“旧区域价值改变时，新奇驱动还能否发现它”提出一个小实验，而不是只比较首次探索率。

适用条件：标准库实验只实现 ε-greedy，不实现 RND 或计数奖励。新奇度与学习进展都是代理信号，不能直接等同于长期控制价值或知识增长。

### 思考题

RND predictor 因持续训练遗忘了旧区域，会发生什么？

提示：内在奖励来自当前预测误差。

参考答案：重访旧区域时误差可能重新升高，把遗忘误认为新奇。需要区分真实新信息、表示漂移和预测器遗忘；这正是探索与可塑性、知识保留相互作用的接口。

### 原始材料与代码

- [Count-Based Exploration 原论文](https://arxiv.org/abs/1606.01868)：看密度模型如何构造伪计数，不只看奖励形状。
- [RND 原论文](https://arxiv.org/abs/1810.12894)：区分固定随机目标与学习中的预测器。
- [OpenAI RND 历史代码](https://github.com/openai/random-network-distillation)：已归档；用于对照原算法的奖励与训练循环。
- [课程学习综述](https://www.jmlr.org/papers/v21/20-212.html)：比较新奇、学习进展与任务选择。