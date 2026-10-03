# 模型与后果预测：学什么，才能用于下一次决策？

执行一个动作或技能以后，会积累多少奖励、何时到哪里；这些预测怎样支持规划与任务变化后的迁移？

## 本章内容

- 区分样本模型、分布模型、期望模型、option model 与 successor features。
- 从随机持续时间的回报推导 reward/end-state 模型与一步 TD 学习。
- 知道期望模型为何在线性价值下足够，以及对非线性价值为什么会失败。
- 独立实现 SF 的向量 TD 与 GPI，并理解它们和 option/世界模型的边界。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 条件模型

模型回答“从这个状态，假如做这个动作/遵循这个技能，会发生什么”。它必须包含条件行为；把经验中的平均下一状态当作所有动作共同的预测，无法比较行动。

### Option

一个可执行技能包含内部策略 π_o 和停止概率 β_o。τ 是至少为 1 的原始步数。模型预测这个已指定行为的后果；改变其策略或停止函数，就是改变被建模的对象。

### 期望与采样

分布模型描述全部可能后果，样本模型随机生成一个后果，期望模型只输出某些统计量。平均值足够与否由下游计算决定，不由模型名字决定。

### 线性价值

φ(s) 是固定特征，w 是权重。线性指价值对特征线性；φ 本身可以是非线性编码。若编码也在学习，关于固定模型/表示的推导要重新检查。

$$
V_w(s)=w^\top\phi(s)
$$

<a id="lesson-setting"></a>

## 1. 模型要预测什么，取决于怎样使用它

“模型好不好”必须相对用途回答。预测下一张画面很准，可能仍错过决定动作的稀有碰撞；平均位移很准，可能把绕障碍的左右两条安全路径平均成穿墙。反过来，一个无法还原像素的模型，只要准确预测所需奖励和后续价值，也可能支持有效控制。先明确下游要计算哪个量，才能选择训练标签。

| 对象 | 模型的输出 | 直接用途 |
| --- | --- | --- |
| 一步分布模型 | P(s′,r\|s,a) | 对任意后续价值求期望，或采样模拟 |
| 一步样本模型 | 给 (s,a) 产生一次随机 (r,s′) | Dyna / rollout；需多样本处理随机性 |
| Option model | 内部折扣奖励 + 折扣终点后果 | 以技能为单位做 Bellman backup |
| 期望模型 | 下游所需特征的条件期望 | 线性价值的精确期望 backup |
| Successor features | 固定策略下未来特征的累计期望 | 奖励权重变化时重估该策略，再做 GPI |
| Latent/world model | 潜在状态、奖励、继续概率等 | 潜在轨迹上的控制学习或搜索 |

先假设环境是固定 MDP，状态或固定特征可观察，技能 $\pi_o,\beta_o$ 固定，且 $\gamma<1$。训练数据可以来自实际执行技能的轨迹，也可以来自支持其动作的其他行为策略。若同时改变环境、技能和表示，模型的预测目标也会漂移，除了静态收敛，还需要分析跟踪误差。

以 option 为例，模型接收 $(s,o)$，输出内部奖励 $r_o(s)$ 与折扣终点分布 $p_o^\gamma(\cdot|s)$；规划器用当前价值 $V$ 计算 $r_o+p_o^\gamma V$。这个分工允许价值改变时复用同一个后果模型。直接预测某个固定策略的完整回报也有用，但那是价值预测，不能替代面向不同后续价值的后果接口。

<a id="lesson-derive"></a>

## 2. Reward model 与折扣终点模型从哪里来

$$
Q(s,o)=\mathbb E_o\!\left[\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}+\gamma^\tau V(S_{t+\tau})\mid S_t=s\right]
$$

先执行指定 option，再按下游策略继续。其内奖励的和与终点的后续价值可以分别建模，因为期望具有线性。

$$
\begin{aligned}r_o(s)&=\mathbb E_o\!\left[\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}\mid s\right]\\ p_o^\gamma(j|s)&=\mathbb E_o\!\left[\gamma^\tau\mathbf1\{S_{t+\tau}=j\}\mid s\right]\\ Q(s,o)&=r_o(s)+\sum_jp_o^\gamma(j|s)V(j)\end{aligned}
$$

p_o^γ 已把时间与终点联合加权。它不是普通归一化转移概率，行和是 E[γ^τ]。规划时不能再乘一次 γ，也不能把行重新归一化到 1。

完整执行可直接提供监督：观察奖励 $r_1,\ldots,r_\tau$ 后计算折扣和，为终点构造 one-hot 向量并乘 $\gamma^\tau$，再分别拟合条件均值。这不需要 bootstrap，但须等技能停止；长技能的估计方差和更新延迟都可能增大。下面用一步递推来学习相同对象。

考虑刚到 $s'$：若 option 停止，后续内部奖励为零，终点就是 $s'$；若继续，剩余奖励由 $r_o(s')$ 描述，剩余折扣终点由 $p_o^\gamma(\cdot|s')$ 描述。两种分支都已多经过一个原始步，递归部分因此乘 $\gamma$。

$$
\begin{aligned}r_o(s)&=\mathbb E_{\pi_o}[R+\gamma(1-\beta_o(S'))r_o(S')\mid s]\\ p_o^\gamma(j|s)&=\mathbb E_{\pi_o}\!\left[\gamma\{\beta_o(S')\mathbf1_{S'=j}+(1-\beta_o(S'))p_o^\gamma(j|S')\}\mid s\right]\end{aligned}
$$

奖励模型的停止分支没有额外 bonus。终点模型的停止分支有 γ×one-hot，因为确实已经执行了一步。真实 terminal 可以作为 V=0 的显式终点，也可用零特征表示；两种约定都需与训练标签一致。

$$
\begin{aligned}\delta_r&=R+\gamma(1-\beta')\hat r_o(s')-\hat r_o(s)\\ \delta_{p,j}&=\gamma[\beta'\mathbf1_{s'=j}+(1-\beta')\hat p_o^\gamma(j|s')]-\hat p_o^\gamma(j|s)\\ \hat r_o(s)&\leftarrow\hat r_o(s)+\alpha_r\rho_o\delta_r\\ \hat p_o^\gamma(j|s)&\leftarrow\hat p_o^\gamma(j|s)+\alpha_p\rho_o\delta_{p,j}\end{aligned}
$$

状态值式模型对当前动作取 π_o 平均，off-policy 时 ρ_o=π_o(a|s)/b(a|s)；真实执行该 option 时 ρ=1。所有右侧使用旧参数；神经网络版本对 target 停止梯度，并对当前输出的梯度做半梯度更新。

**算法：算法伪代码**

1. 固定 option π_o、β_o 和表示版本；初始化 reward/end-state 模型
1. 每次看到原始 transition s,a,r,s′：
  1. 缓存旧 r_o(s)、r_o(s′)、p_o(s)、p_o(s′)
  1. 读取到达 s′ 后的 β；计算记录动作时的 π_o(a|s)/b(a|s)
  1. reward_target = r + γ*(1−β)*old_reward_next
  1. endpoint_target = γ*[β*one_hot(s′) + (1−β)*old_endpoint_next]
  1. 分别更新 reward 和 endpoint 输出
1. 规划调用时：r_o(s) + dot(p_o(s), 当前 V)
1. 若 option/表示改变，标记模型过期并重新采样或持续更新

<a id="lesson-expectation"></a>

## 3. 期望模型为什么有时够用，有时必错

状态很多时，显式存每个终点的概率昂贵。若下游价值对特征线性，就可以把求和移入特征期望，预测一个固定维度的向量 m_o(s)。这不是近似技巧，而是给定前提下的恒等式；但前提本身非常重要。

$$
\begin{aligned}m_o(s)&=\mathbb E_o[\gamma^\tau\phi(S_{t+\tau})\mid s]\\ \mathbb E_o[\gamma^\tau V_w(S_{t+\tau})\mid s]&=w^\top m_o(s)\\ y_m&=\gamma[\beta'\phi(s')+(1-\beta')\hat m_o(s')]\end{aligned}
$$

m 的 TD 递推只需把 one-hot 终点替换为特征。状态可以高维，但下游 V 必须对这套特征线性，才能把期望与价值计算交换。

反例：$X$ 以相同概率为 $-1$ 或 $+1$，$V(x)=x^2$。均值模型给 $\mathbb E[X]=0$，代入价值得到 $0$，而真正的 $\mathbb E[V(X)]=1$。只保存均值会丢掉方差信息。扩展特征为 $(x,x^2)$、学习完整分布或采样多个后果，分别提供了恢复所需信息的途径。

控制还包含最大值运算。即使每个动作的 $Q$ 对特征线性，$V(s)=\max_aQ(s,a)$ 通常仍非线性，所以 $\mathbb E[\max_aQ(S',a)]$ 不等于 $\max_aQ(\mathbb E[S'],a)$。采用线性状态价值 $V_w$、再在当前状态比较不同动作模型，是另一种规划接口；预测中的线性等价不能直接推广到任意 Q 控制。

随机时间与终点也可能相关。假设一半概率走一步到价值 $10$ 的 A，另一半概率走三步到价值 $0$ 的 B，$\gamma=0.9$。正确贡献是 $0.5\times0.9\times10=4.5$；若把 $\mathbb E[\gamma^\tau]=0.8145$ 与平均终点价值 $5$ 相乘，得到 $4.0725$。用平均时长 $2$ 得到的 $\gamma^2\times5=4.05$ 也不同。需要保存的是时间与终点的联合折扣后果。

<a id="lesson-successors"></a>

## 4. Successor features 与 GPI：不重学动力学地换奖励

另一个复用问题是：环境和行为策略不变，奖励的偏好改变。比如同一路线会产生“耗时、电量、采到的资源”三个信号；早上偏好快，晚上偏好省电。如果奖励是这些信号的线性组合，就能先预测整段未来信号，再乘新的偏好权重。这是 SF 的基本分解，不是预测“下一状态均值”。

$$
\begin{aligned}r_w(s,a,s')&=\phi(s,a,s')^\top w\\ \psi^\pi(s,a)&=\mathbb E_\pi\!\left[\sum_{k=0}^\infty\gamma^k\phi(S_{t+k},A_{t+k},S_{t+k+1})\mid S_t=s,A_t=a\right]\\ Q_w^\pi(s,a)&=\psi^\pi(s,a)^\top w\end{aligned}
$$

φ 是每步可观察的特征信号，ψ 是同一目标策略下的向量预测；每个分量都是一个价值预测问题。等式需要同样的动力学、策略、折扣与特征语义，奖励变化仅发生在 w。

将累计和拆成第一步与后续，就得到向量 Bellman 方程：标量奖励换为向量 $\phi$，标量价值换为向量 $\psi$。下一动作按同一个目标策略平均或采样。若对每个特征分量分别取最大，不同分量可能对应彼此冲突的行为，就不再是这个策略的未来特征预测。

$$
\begin{aligned}y_\psi&=\phi(s,a,s')+\gamma(1-d)\sum_{a'}\pi(a'|s')\hat\psi^\pi(s',a')\\ \hat\psi^\pi(s,a)&\leftarrow\hat\psi^\pi(s,a)+\alpha[y_\psi-\hat\psi^\pi(s,a)]\end{aligned}
$$

这是动作条件的表格 expected-TD：当前动作已作为条件给定，环境转移的样本不需要再乘当前动作概率比；后续动作明确按 π 求平均。如果改成状态式 ψ(s) 预测，当前动作平均和 off-policy 修正就必须相应改变。

只有一个旧策略时，重新加权得到的是该策略的新价值。为了改善控制，可保存多个 $\pi_i$ 的 SF，先计算每个候选在新奖励下的价值，再在每个状态选择候选中评价最高的动作，这就是 generalized policy improvement。它可以逐状态组合动作，而非只在 episode 开头选一个旧策略并始终照做。

$$
\pi_{\rm GPI}(s)\in\arg\max_a\max_i\left[\hat\psi^{\pi_i}(s,a)^\top w_{\rm new}\right]
$$

顺序是先沿特征维度与 w 做点积，再沿策略维度取 max，最后沿动作维度选 argmax。把两个 max 或特征求和维度弄错，会产生另一种算法。

**算法：算法伪代码**

1. 准备固定策略集合 π_1,…,π_n 与各自 SF 估计
1. 每个观察 transition：
  1. 对每个 i，用 π_i 的下一动作分布构造向量 TD target，更新 ψ_i
1. 若奖励权重变为 w_new：
  1. 不改 SF，先计算 Q_i(s,a)=dot(ψ_i(s,a),w_new)
  1. score(a)=max_i Q_i(s,a)
  1. 执行 argmax_a score(a)，并继续收集真实数据
1. 如果学习了新的专门策略，把它连同其 SF 加入集合；预算有限时须选择保留项

在精确 $Q$、相同动力学和折扣下，GPI 不劣于被比较的各个策略；近似保证取决于统一价值误差界。奖励不在当前 $\phi$ 的线性张成空间，或动力学改变导致旧 $\psi$ 失效时，需要重新估计相应误差。SF 预测策略产生的累计特征，option 定义执行与停止，option model 预测停止时后果；三者可以组合，但承担不同职责。

<a id="lesson-example"></a>

## 5. 两次原始转移怎样构造完整技能模型

技能从状态 $0$ 经过 $1$ 到 $2$，在 $2$ 停止，奖励分别为 $1,2$，$\gamma=0.9$。模型初值为零，步长为一。按从后往前的顺序更新，可以在两次 backup 中算出结果；在线从前往后更新时，还需再次访问或重放，才能把后方的新信息传到起点。

| 更新 | reward target | discounted endpoint target |
| --- | --- | --- |
| 1→2，β(2)=1 | 2 + 0 = 2 | 0.9 × one-hot(2) |
| 0→1，β(1)=0 | 1 + 0.9×2 = 2.8 | 0.9 × [0,0,0.9] = [0,0,0.81] |
| 给 V(2)=10 做一次规划 | 2.8 | 0.81×10 = 8.1，合计 10.9 |

这个 backup 与直接计算两步回报相同，无需重新模拟技能内的动作。若把终点向量错误归一化为 $[0,0,1]$，会得到 $12.8$；若额外再乘 $\gamma$，会得到 $10.09$。两个错误分别丢掉或重复计算了时间折扣。

SF 例子：两个动作分别产生特征 $(1,0)$ 和 $(0,1)$，随后进入动作编号对应的状态，$\gamma=0.9$。策略 $\pi_0$ 始终选动作 $0$，$\pi_1$ 始终选动作 $1$。先做动作 $1$ 再跟 $\pi_0$，SF 为 $(9,1)$；先做动作 $0$ 再跟 $\pi_1$，SF 为 $(1,9)$。当 $w=(1,2)$ 时，GPI 对两个动作的分数为 $19,20$，故选动作 $1$；改为 $w=(2,1)$ 后，无需重学 SF 就会改选动作 $0$。这里复用的是不变动力学和不变目标策略下的预测。

<a id="lesson-code"></a>

## 6. 实现与实验：期望、时间和策略条件

Monte Carlo/TD option model、随机时长联合模型、SF 向量更新及 GPI；数组和更新时序都可直接检查。

```python
def option_episode_target(rewards, gamma, endpoint, n_states):
    reward_target = sum(gamma**k * r for k, r in enumerate(rewards))
    endpoint_target = [0.0] * n_states
    endpoint_target[endpoint] = gamma**len(rewards)
    return reward_target, endpoint_target


def model_td_update(reward_model, endpoint_model, state, next_state, reward,
                    beta_next, alpha=0.1, gamma=0.9, rho=1.0):
    """Fixed option; n is a discounted endpoint distribution, not normalized.

    A terminal environment state can be represented explicitly with V=0 and
    beta=1. All right-hand sides use pre-update values (self-loops included).
    """
    old_reward, next_reward = reward_model[state], reward_model[next_state]
    old_row, next_row = endpoint_model[state][:], endpoint_model[next_state][:]
    r_target = reward + gamma * (1.0 - beta_next) * next_reward
    p_target = [gamma * (beta_next * float(j == next_state)
                        + (1.0 - beta_next) * next_row[j])
                for j in range(len(old_row))]
    reward_model[state] = old_reward + alpha * rho * (r_target - old_reward)
    endpoint_model[state] = [v + alpha * rho * (target - v)
                             for v, target in zip(old_row, p_target)]
    return r_target, p_target


def model_backup(reward, discounted_endpoints, values):
    return reward + sum(p * v for p, v in zip(discounted_endpoints, values))


def mixed_duration_model(outcomes, gamma, n_states):
    """outcomes: (probability, rewards, endpoint). Keep time/end correlation."""
    if not math.isclose(sum(p for p, _, _ in outcomes), 1.0):
        raise ValueError("Outcome probabilities must sum to one")
    r_bar, p_bar = 0.0, [0.0] * n_states
    for probability, rewards, endpoint in outcomes:
        r, p = option_episode_target(rewards, gamma, endpoint, n_states)
        r_bar += probability * r
        p_bar = [old + probability * new for old, new in zip(p_bar, p)]
    return r_bar, p_bar


def successor_feature_step(psi, state, action, features, next_state, next_action_probs,
                           alpha=0.1, gamma=0.9, terminal=False):
    """Action-conditioned SF for a fixed target policy, with known feature signal.

    The observed action conditions the prediction, so the next action is averaged
    under the target policy; no current-action importance ratio is needed here.
    psi[s][a] is a feature vector. Snapshot before mutation handles self-loops.
    """
    old = psi[state][action][:]
    next_vectors = [row[:] for row in psi[next_state]]
    next_features = [sum(pr * vec[j] for pr, vec in zip(next_action_probs, next_vectors))
                     for j in range(len(features))]
    target = [x + (0.0 if terminal else gamma * xn)
              for x, xn in zip(features, next_features)]
    psi[state][action] = [x + alpha * (y - x) for x, y in zip(old, target)]
    return target


def gpi_action(successor_feature_bank, state, reward_weights):
    """Greedy policy improvement over a bank of evaluated fixed policies."""
    action_scores = [max(sum(x*w for x,w in zip(psi[state][a], reward_weights))
                         for psi in successor_feature_bank)
                     for a in range(len(successor_feature_bank[0][state]))]
    return max(range(len(action_scores)), key=action_scores.__getitem__), action_scores


def learn_tiny_sf():
    bank = []
    for fixed_action in (0, 1):
        psi = [[[0.0, 0.0] for _ in range(2)] for _ in range(2)]
        probabilities = [float(a == fixed_action) for a in range(2)]
        for _ in range(400):
            for s in range(2):
                for a in range(2):
                    successor_feature_step(psi, s, a, [float(a==0), float(a==1)],
                                           a, probabilities, alpha=1.0)
        bank.append(psi)
    return bank
```

无第三方依赖；所有目标都可用上面的数字核算

```sh
python3 knowledge_algorithms_lab.py models
python3 knowledge_algorithms_lab.py test
```

运行得到 reward model 为 $2.8$，终点向量为 $[0,0,0.81]$，backup 为 $10.9$；随机时间与终点联合贡献为 $4.5$。SF/GPI 在 $w=(1,2)$ 时选择动作 $1$，两个动作的分数约为 $[19,20]$。逐项改变价值、持续时间或奖励权重，可以观察模型究竟保留了哪一种可复用信息。

- 改动 A：只改变 endpoint value，冻结 option 与模型，验证模型可被不同价值函数重复调用。
- 改动 B：令短路径终点与长路径终点交换，验证即使平均时长和未加权终点分布没变，正确价值仍可改变。
- 改动 C：保持线性 reward 但改变动力学，冻结 SF；观察立即迁移不再正确，再比较持续更新 ψ 的恢复速度。
- 改动 D：给 critic 加非线性，分别比较均值输入、特征扩展和多样本估计；报告 target 偏差而非只看训练 loss。

<a id="lesson-reward-aware"></a>

## 7. 当环境结构也需要考虑代价：Default Representation

SR 和 Laplacian 表示主要刻画在某种默认行为下哪些状态容易互相到达，但相同的图结构可以对应完全不同的实际代价。两条通道几何长度相同，一条却持续消耗大量资源：仅根据连通性构造的技能仍可能偏爱这条通道。Reward-Aware Proto-Representations（NeurIPS 2025）研究如何让用于技能与奖励塑形的结构表示反映沿途奖励。

理解 Default Representation（DR）要先限定控制问题。设非终止状态集合为 $N$、终止集合为 $T$，默认策略诱导转移 $P^{\pi_d}$，非终止奖励 $r(s)<0$。在线性可解控制中，控制器改变下一状态分布，同时为偏离默认分布付出 KL 代价，温度 $\lambda>0$ 控制偏离的价格。它不是任意标准动作 MDP 都自动具备的结构。

$$
v^*(s)=r(s)+\lambda\log\sum_{s'}P^{\pi_d}(s'|s)\exp\!\left(v^*(s')/\lambda\right)
$$

这个软 Bellman 方程来自最大化“期望后续价值减去对默认转移的 KL 代价”。默认分布没有支持的后果不能凭空控制出来；边界状态的价值由终止收益给定。

令 desirability 为 $z(s)=\exp(v^*(s)/\lambda)$，把两侧指数化。非线性的对数求和变成 $z$ 的线性递推，再把非终止状态和终止状态分开，就得到 DR。

$$
\begin{aligned}D_r&=\operatorname{diag}\!\left(\exp(r_N/\lambda)\right)\\ z_N&=D_r\left(P_{NN}^{\pi_d}z_N+P_{NT}^{\pi_d}z_T\right)\\ Z_{NN}&=\left[D_r^{-1}-P_{NN}^{\pi_d}\right]^{-1}\\ z_N&=Z_{NN}P_{NT}^{\pi_d}z_T,\qquad z_T=\exp(r_T/\lambda)\end{aligned}
$$

Z 把内部动力学与沿途代价编码在一起；固定这两项而改变终点收益时，可以重用 Z。负的非终止奖励使 D_r 的对角元素小于 1，有助于保证所需逆和递推收敛。终点收益并未混进内部模型。

DR 与 SR 的联系可直接算出。若所有非终止状态的奖励都等于 $\lambda\log\gamma$，则 $D_r=\gamma I$，因而 $Z_{NN}=\gamma(I-\gamma P_{NN}^{\pi_d})^{-1}$，即 SR 的常数倍。奖励不均匀时，每经过一个状态都受到不同的指数权重，表示便能区分低代价和高代价区域。

$$
Z=D_r+D_rPZ,\qquad Z(s,:)\leftarrow Z(s,:)+\alpha\left[e^{r(s)/\lambda}\left(\mathbf e_s+Z(s',:)\right)-Z(s,:)\right]
$$

这是在默认策略下采样、只考虑非终止行列时的递推；进入终止状态后，内部矩阵的后续行取零。若行为不同，需明确所估计的默认转移或使用相应分布修正。它近似 SR 的向量 TD，但当前状态奖励决定的缩放同时作用于 one-hot 与后续行，不能只替换 SR 的 $\gamma$。

手算一条通道：默认行为从状态 0 必然进入状态 1，再进入终点；令两个非终止状态的指数奖励权重均为 0.5。则内部矩阵第一行为 (0.5,0.25)，第二行为 (0,0.5)。若状态 0 的代价增大，使权重降为 0.25，第一行变成 (0.25,0.125)。到达状态 1 的默认路径没有变，但跨过代价区的权重下降了。这正是奖励感知结构与纯连通结构的差异。

论文将 DR 用于构造奖励感知的谱特征，再用于奖励塑形、技能发现与迁移实验。它提供的是另一种构造长期结构的准则，而非把所有模型都换成一个新矩阵。若环境奖励变化，DR 本身通常也要更新；相反，奖励线性权重变化而动力学不变时，固定策略 SF 可以保持不变。两者分别把奖励放在不同位置，因此具有不同的复用边界。

| 表示 | 固定哪些条件才能复用 | 直接改变什么 |
| --- | --- | --- |
| SR / SF | 动力学、目标策略、折扣、特征语义 | 用新奖励权重重新评价同一策略 |
| DR | 默认动力学、内部奖励、KL 控制约定 | 通过终点收益或奖励感知结构重新求解 |
| Option model | 技能策略与停止规则、动力学、奖励约定 | 用新的后续价值评价同一技能 |
| STOMP 子任务与模型 | 子任务可改变，后果模型须匹配当前技能 | 由奖励相关子任务产生行为，再预测真实后果 |

作者 Reward-Aware-Proto-Representations 仓库适合按“默认转移与奖励 → DR → 特征 → 技能/塑形 → 外部回报”阅读。实验中应分别改变内部代价和终点奖励：前者测试结构能否重新适应，后者测试已有结构能否复用。若两者同时变化，就难以解释改进来自哪一种能力。

<a id="lesson-branches"></a>

## 8. 从表格模型到潜在世界模型

潜在世界模型把观测历史压缩为 z，再学从 z 和动作预测下一潜在状态。一个常见训练结构包含：利用当前观测得到后验表示、仅凭过去表示和动作得到预测先验、重建或预测观测、预测 reward 与继续概率，并用一致性/KL 项约束先验后验。目标是让不接触未来真实观测的想象轨迹仍保持决策相关的信息，而不是在训练时用未来观测泄漏答案。

$$
\begin{aligned}z_t&\sim q_\theta(z_t\mid h_t,o_t),\qquad h_{t+1}=f_\theta(h_t,z_t,a_t)\\ \hat z_{t+1}&\sim p_\theta(z_{t+1}\mid h_{t+1})\\ \mathcal L_{\rm model}&=\mathcal L_{\rm obs}+\mathcal L_{\rm reward}+\mathcal L_{\rm continue}+\mathcal L_{\rm regularize}\end{aligned}
$$

这四项说明观测、奖励、继续概率和潜在先验—后验之间的分工；各项的实际权重、分布形式与梯度路径由具体算法规定。重建项使用真实观测，想象轨迹只能使用预测先验。

Dreamer 类方法在真实序列上学习模型，再从后验状态启动想象 rollout 来训练 actor/critic；部署可直接执行学到的 actor。MuZero 类方法学习对 reward、value、policy 有用的潜在递推，并在决策时做搜索，不要求重建全部像素。这两类模型“学来做什么”不同，不能以画面是否逼真作为统一分数。

持续学习增加三种独立失效：环境漂移使过去 P 不再正确；技能漂移使同名 o 的后果改变；表示漂移使旧 replay 中的潜在坐标与当前模型不兼容。最小研究协议应分别开关三者。模型带上技能/编码器版本、从原始经验重新编码、近期数据加权、保留校准集、缩短想象跨度，都是可比较的设计变量，但没有一种可在所有变化下保证模型可靠。

| 模型路线 | 特别擅长的复用 | 需要额外检查 |
| --- | --- | --- |
| Tabular option model | 技能后果被任意新 V 调用 | 持续时间、终止、技能版本 |
| Expectation model | 线性值函数改变时快速重估 | 特征充分性、非线性和 max 的交换 |
| SF / GPI | 同动力学下换奖励权重 | 目标策略固定、reward 特征可表达性 |
| Generative latent model | 多步想象与分布性后果 | 表示充分性、分布外误差、多步滚动 |
| Value-equivalent / decision-oriented model | 保留下游决策所需量 | 保证通常相对于某类策略/价值，不是全世界精确模型 |

<a id="lesson-check"></a>

## 9. 失败诊断与自测答案

- 模型 loss 很低，规划却变坏：检查训练分布与 planner 查询分布、是否漏掉罕见高代价事件、是否只预测均值而下游非线性。
- 长技能被过分偏爱：检查折扣终点模型是否错误归一化、是否用平均 τ 替换随机 τ、是否漏记沿途 reward。
- 换奖励后 SF 迁移失败：检查新奖励是否真在特征张成空间、旧策略 SF 是否准确、动力学是否同时变了。
- 同名技能更新后旧模型失效：不是随机异常，而是条件行为已改变；要么重估模型，要么让版本与参数成为显式条件。

自测 1：$p_o^\gamma$ 的行和小于一是否代表概率遗漏？答：它包含时间折扣，行和为 $\mathbb E[\gamma^\tau]$。自测 2：把每个奖励特征各自的最优累计值相加，能否得到新奖励的最优值？答：各分量的最优策略可能冲突，SF 必须条件化于同一策略。自测 3：为什么 stopping bonus 不进入 $r_o$？答：它是训练技能的辅助目标，而主任务模型预测执行技能真正得到的环境奖励。

## 本章的实验设计

模型预测误差与决策损失需要分别测量。对规划无关变量的准确预测，不一定改善策略。

设定：固定同一真实数据集训练后果模型，再在独立轨迹和规划实际访问的状态上评价。加入随机时长与终点相关的 option。

- 折扣后果模型保留持续时间与终点的相关性。
- 奖励、continuation 和后继特征目标分别核对。
- 模型生成的虚拟奖励不计作真实环境收益。

对照：准确模型诊断与学习模型；随机分布测试与规划访问分布测试；固定消费者只替换模型

记录：一步及多步预测误差；动作排序、价值误差和模型使用分布；真实执行表现及模型计算成本

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-planning)

## 学习与研究衔接

模型不必重建全部观测。应预测规划真正需要的量，并测试模型误差如何改变决策。

[分册导读](https://yingwen.io/zh/continual-rl/start/deep-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-models) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=models) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=models)

<a id="chapter-code"></a>

## 下载与运行

表格 option model、SF/GPI 与条件反例；不包含完整深度世界模型训练。

[下载 knowledge_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/knowledge_algorithms_lab.py)

```sh
python3 knowledge_algorithms_lab.py models
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton, Precup & Singh — Options 的 Bellman 模型](https://doi.org/10.1016/S0004-3702(99)00052-1)：原始框架。先比较完整轨迹定义与一步递推，再核对 discount 已包含在 transition model 中。

- [Sutton et al. — Reward-Respecting Subtasks，模型部分](https://arxiv.org/html/2202.03466v3)：原文第 4 节将 reward model 与 expectation model 分开；子任务 stopping bonus 不能混入环境 reward model。

- [Wan et al. — Planning with Expectation Models](https://arxiv.org/abs/1904.01191)：原文。理解线性状态价值下期望模型的充分性及函数逼近前提。

- [Planning with Expectation Models for Control](https://arxiv.org/abs/2104.08543)：控制与期望模型的接口；重点比较线性预测的等价条件与包含非线性 Q/max 的控制更新。

- [Barreto et al. — Successor Features for Transfer in RL](https://arxiv.org/abs/1606.05312)：原文。先查奖励分解，再查目标策略条件与 GPI 改善保证的精确/近似前提。

- [DeepMind — Option Keyboard 作者工程](https://github.com/google-deepmind/deepmind-research/tree/f5de0ede8430809180254ee957abf36ed62579ef/option_keyboard)：研究团队代码；keyboard_agent.py 的策略、cumulant、动作三个维度与本页 GPI 计算顺序对照。不是所有 SF 方法的统一实现。

- [Reward-Aware Proto-Representations in RL · NeurIPS 2025](https://arxiv.org/abs/2505.16217)：DR 的定义、线性可解控制推导与谱特征用途；注意内部奖励固定和终点收益改变是不同的迁移条件。

- [Reward-Aware Proto-Representations 作者实现](https://github.com/httse9/Reward-Aware-Proto-Representations)：从默认转移和奖励构造 DR，比较奖励感知特征、奖励塑形与技能发现；读取数值求解与谱处理设置。

- [DR 矩阵构造 — rep_utils.py](https://github.com/httse9/Reward-Aware-Proto-Representations/blob/master/minigrid_basics/examples/rep_utils.py)：compute_SR 与 compute_DR 并排给出两个矩阵的区别；get_representation 中的对称化决定了随后分解的谱对象。

- [DR 在线表示与发现 — ROD_DR.py](https://github.com/httse9/Reward-Aware-Proto-Representations/blob/master/minigrid_basics/examples/ROD_DR.py)：learn_representation 对应 DR 的采样递推；compute_eigenvector 从访问过的状态计算谱特征。结合数据采集循环阅读，辨认其经验转移实际对应的默认行为分布。

- [Hafner et al. — DreamerV3](https://arxiv.org/abs/2301.04104)：原文。区分真实序列模型学习与想象序列 actor/critic 学习，保留完整损失与超参数。

- [Danijar Hafner — DreamerV3 作者维护实现](https://github.com/danijar/dreamerv3)：作者维护的公开重实现；README 明确其为 reimplementation，并非原内部训练代码。先追 model/imagined rollout/actor 三个接口。

- [Schrittwieser et al. — MuZero](https://arxiv.org/abs/1911.08265)：原文。预测 reward、policy、value 的潜在模型与决策时树搜索；不以观测重建作为必要接口。
