# 模型与后果预测：学什么，才能用于下一次决策？

执行一个动作或技能以后，会积累多少奖励、何时到哪里；这些预测怎样支持规划与任务变化后的迁移？

## 本章内容

- 区分样本模型、分布模型、期望模型、option model 与 successor features。
- 从随机持续时间的回报推导 reward/end-state 模型与一步 TD 学习。
- 知道期望模型为何在线性价值下足够，以及对非线性价值为什么会失败。
- 独立实现 SF 的向量 TD 与 GPI，并理解它们和 option/世界模型的边界。

<a id="problem-definition"></a>

## 本章的问题定义

预测动作或技能后果，以支持可改变的下游价值和规划；输出应由用途确定，而非统一要求重建全部观测。

### 给定条件与符号

- 被建模的固定动作/技能、奖励与原始时间约定。
- 状态/特征、下游价值函数类、真实数据与模型容量预算。

### 需要求解的对象

足以计算指定后果备份的奖励及终点统计；完整分布、样本模型、期望特征模型和SF预测的是不同对象。

### 信息与数据权限

技能 $o$ 的策略与停止函数给定，$\tau$ 为原始时长。改变技能或表示即改变题目；离策略一步学习须记录行为概率并检查支持。

$$
\mathcal B_oV(s)=\mathbb E_o\!\left[\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}+\gamma^\tau V(S_{t+\tau})\mid s\right],\qquad \hat{\mathcal B}_oV\approx\mathcal B_oV\quad(V\in\mathcal V)
$$

$\mathcal V$ 为声明的下游价值类，$\gamma$ 为折扣。模型充分性以能否复算该类备份判断；奖励模型 $r_o$ 和折扣终点模型 $p_o^\gamma$ 是实现接口。若 $V_w(s)=w^\top\phi(s)$，折扣特征期望足够；一般非线性价值不满足此交换。

### 成立条件与解的含义

- 固定环境、表示与被建模行为；奖励有界，$0\le\gamma<1$。终点模型假设技能停止或真实终止几乎必然在有限时间发生；折扣时长与终点联合建模，真实终止的后续价值固定为零。
- SF重加权需相同动力学、策略、折扣和特征语义，奖励变化限于给定特征的线性张成。

判断准则：已知技能上检验奖励和折扣终点向量，并对多组未拟合下游价值比较备份误差；包含随机终点均值失败反例及时间—终点相关例。

### 适用边界

- 平均下一状态代入非线性价值通常不等于价值期望。
- 模型重建损失下降不直接证明动作选择改善。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [Options 与技能发现](https://yingwen.io/zh/continual-rl/construction/options/)：技能规定执行和停止，模型估计该行为的奖励、时长及终点后果。

- 组合不同学习问题 · [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)：规划使用模型统计计算候选价值，其需求决定模型必须保留什么。

- 组合不同学习问题 · [通用价值函数与预测知识](https://yingwen.io/zh/continual-rl/construction/predictive-knowledge/)：模型统计可化为特定预测题目；SF也预测固定策略的累计特征而非任意新行为。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

完整后果分布昂贵；均值压缩可能丢失下游最大值、非线性或随机时长所需的信息。

### 本章的核心思路

相对下游函数类定义足够统计，将内部奖励与联合折扣终点分开建模。

1. [从技能回报拆出后果接口](#lesson-derive)：因为未来价值可变，奖励和折扣终点分别建模，使同一模型可重用于不同尾值。

2. [检验压缩是否保留所需期望](#lesson-expectation)：因为非线性不能随意与期望交换，先在线性价值下推特征期望，再用均值反例界定适用范围。

3. [为奖励迁移预测累计特征](#lesson-successors)：因为只换奖励权重时可复用固定策略后果，SF保留累计特征，GPI再比较候选策略的重加权价值。

结论与条件：线性价值与固定特征下的期望备份等价是恒等式；SF/GPI保证需相同动力学等条件及价值误差控制，不覆盖任意奖励或技能变化。

### 相关方法改变了什么

- 完整分布/样本模型：保留多后果或提供样本，可用于一般尾值但成本与采样方差不同。

- 期望特征模型：对指定线性尾值充分，不能只凭均值支持任意非线性控制。

- Successor features：预测固定策略的累计特征，用于奖励重加权，不等于技能终点模型。


<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 条件模型

模型回答“从这个状态，假如做这个动作/遵循这个技能，会发生什么”。它必须包含条件行为；把经验中的平均下一状态当作所有动作共同的预测，无法比较行动。

### Option

一个可执行技能包含内部策略 $π_o$ 和停止概率 $β_o$。τ 是至少为 1 的原始步数。模型预测这个已指定行为的后果；改变其策略或停止函数，就是改变被建模的对象。

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

先假设环境是固定 MDP，状态或固定特征可观察，奖励有界，技能 $\pi_o,\beta_o$ 固定，且 $0\le\gamma<1$。本页的终点模型假设技能停止或真实终止几乎必然在有限时间发生，所以终点随机变量有定义。训练数据可以来自实际执行技能的轨迹，也可以来自支持其动作的其他行为策略。若同时改变环境、技能和表示，模型的预测目标也会漂移，除了静态收敛，还需要分析跟踪误差。

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

$p_o^γ$ 已把时间与终点联合加权。它不是普通归一化转移概率，行和是 $E[γ^τ]$。规划时不能再乘一次 γ，也不能把行重新归一化到 1。

完整执行可直接提供监督：观察奖励 $r_1,\ldots,r_\tau$ 后计算折扣和，为终点构造 one-hot 向量并乘 $\gamma^\tau$，再分别拟合条件均值。这不需要 bootstrap，但须等技能停止；长技能的估计方差和更新延迟都可能增大。下面用一步递推来学习相同对象。

考虑刚到 $s'$：若 option 停止，后续内部奖励为零，终点就是 $s'$；若继续，剩余奖励由 $r_o(s')$ 描述，剩余折扣终点由 $p_o^\gamma(\cdot|s')$ 描述。真实环境终止也必须停止技能。令 $D\in\{0,1\}$ 为真实终止标志，$\widetilde\beta=D+(1-D)\beta_o(S')$ 为合并后的停止概率。这里的 $\tau$ 取技能停止与真实终止中先发生的时间；采样窗口截断不算真实终止。

$$
\begin{aligned}r_o(s)&=\mathbb E_{\pi_o}[R+\gamma(1-\widetilde\beta)r_o(S')\mid s]\\ p_o^\gamma(j|s)&=\mathbb E_{\pi_o}\!\left[\gamma\{\widetilde\beta\mathbf1_{S'=j}+(1-\widetilde\beta)p_o^\gamma(j|S')\}\mid s\right]\end{aligned}
$$

两条分支都已经过一个原始步，所以终点项与继续项都乘折扣。奖励模型的停止分支没有额外奖励。这里保留显式终止状态，并固定其后续价值为零；终点模型仍记录到达它的折扣质量。

$$
\begin{aligned}\widetilde\beta'&=d+(1-d)\beta_o(s')\\ \delta_r&=R+\gamma(1-\widetilde\beta')\hat r_o(s')-\hat r_o(s)\\ \delta_{p,j}&=\gamma[\widetilde\beta'\mathbf1_{s'=j}+(1-\widetilde\beta')\hat p_o^\gamma(j|s')]-\hat p_o^\gamma(j|s)\\ \hat r_o(s)&\leftarrow\hat r_o(s)+\alpha_r\rho_o\delta_r\\ \hat p_o^\gamma(j|s)&\leftarrow\hat p_o^\gamma(j|s)+\alpha_p\rho_o\delta_{p,j}\end{aligned}
$$

d 是本次经验中的真实终止标志。状态值式模型对当前动作取 $π_o$ 平均，off-policy 时 $ρ_o=π_o(a|s)/b(a|s)$；真实执行该 option 时 ρ=1。所有右侧使用旧参数；神经网络版本对 target 停止梯度，并对当前输出的梯度做半梯度更新。

**算法：算法伪代码**

1. 固定 option $π_o$、$β_o$ 和表示版本；初始化 reward/end-state 模型
1. 每次看到原始 transition s,a,r,s′,d：
  1. 缓存旧 $r_o(s),r_o(s^{\prime}),p_o(s),p_o(s^{\prime})$
  1. 读取到达 s′ 后的 β；计算记录动作时的 $π_o(a|s)/b(a|s)$
  1. `stop = d + (1-d)*beta`
  1. `reward_target = r + gamma*(1-stop)*old_reward_next`
  1. `endpoint_target = gamma*(stop*one_hot(next_state) + (1-stop)*old_endpoint_next)`
  1. 分别更新 reward 和 endpoint 输出
1. 规划调用时：$r_o(s)$ + `dot(p_o(s), 当前 V)`；终止状态的 V 固定为零
1. 若 option/表示改变，标记模型过期并重新采样或持续更新

终止反例：本步奖励为 $1$、$\gamma=0.9$、技能本身的 $\beta_o(s')=0$，旧奖励模型在终止状态误估为 $5$。遗漏 $d$ 会得到目标 $1+0.9\times5=5.5$；正确目标为 $1$。即使控制价值在终止状态固定为零，也不能替代奖励模型自己的停止屏蔽。

<a id="lesson-expectation"></a>

## 3. 期望模型为什么有时够用，有时必错

状态很多时，显式存每个终点的概率昂贵。若下游价值对特征线性，就可以把求和移入特征期望，预测一个固定维度的向量 $m_o(s)$。这不是近似技巧，而是给定前提下的恒等式；但前提本身非常重要。

$$
\begin{aligned}m_o(s)&=\mathbb E_o[\gamma^\tau\phi(S_{t+\tau})\mid s]\\ \mathbb E_o[\gamma^\tau V_w(S_{t+\tau})\mid s]&=w^\top m_o(s)\\ y_m&=\gamma[\widetilde\beta'\phi(s')+(1-\widetilde\beta')\hat m_o(s')]\end{aligned}
$$

m 的 TD 递推只需把 one-hot 终点替换为特征，并沿用真实终止与技能停止的合并概率。状态可以高维，但下游 V 必须对这套特征线性，才能把期望与价值计算交换。

若采用终止状态零特征的约定，令 $\phi(s')=0$ 当 $d=1$。目标可等价写为 $y_m=\gamma(1-d)[\beta_o(s')\phi(s')+(1-\beta_o(s'))\hat m_o(s')]$。这时真实终止的目标向量为零；它与显式终止状态的 one-hot 约定输出不同，但在终止后价值为零的规划中一致。

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

1. 准备固定策略集合 $π_1,\ldots,π_n$ 与各自 SF 估计
1. 每个观察 transition：
  1. 对每个 i，用 $π_i$ 的下一动作分布构造向量 TD target，更新 $ψ_i$
1. 若奖励权重变为 $w_{new}$：
  1. 不改 SF，先计算 $Q_i(s,a)=ψ_i(s,a)^T w_{new}$
  1. $\mathrm{score}(a)=\max_i Q_i(s,a)$
  1. 执行 $\arg\max_a\mathrm{score}(a)$，并继续收集真实数据
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
                    beta_next, alpha=0.1, gamma=0.9, rho=1.0, terminal=False):
    """Fixed option; n is a discounted endpoint distribution, not normalized.

    True environment termination forces stopping, independently of beta.
    The explicit terminal endpoint has V=0 during planning; its discounted
    probability mass is retained. A rollout cutoff is not true termination.
    All right-hand sides use pre-update values (self-loops included).
    """
    old_reward, next_reward = reward_model[state], reward_model[next_state]
    old_row, next_row = endpoint_model[state][:], endpoint_model[next_state][:]
    stop = 1.0 if terminal else beta_next
    r_target = reward + gamma * (1.0 - stop) * next_reward
    p_target = [gamma * (stop * float(j == next_state)
                        + (1.0 - stop) * next_row[j])
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

Z 把内部动力学与沿途代价编码在一起；固定这两项而改变终点收益时，可以重用 Z。负的非终止奖励使 $D_r$ 的对角元素小于 1，有助于保证所需逆和递推收敛。终点收益并未混进内部模型。

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

<a id="research-model-query-equivalence"></a>

## 研究专题 A · 模型充分性由规划查询与风险目标共同决定

本章的期望模型已说明线性价值下哪些统计足够。Value Equivalence（NeurIPS 2020）进一步把模型规格写成查询集合：指定哪些策略与后续函数，要求模型在这些查询上生成正确 backup。模型可以舍弃不影响这些计算的细节；当规划器或奖励改变时，原先可忽略的细节也可能变成必要知识。

$$
(T_M^\pi v)(s)=\mathbb E_{M,\pi}[R+\gamma v(S')\mid s],\qquad \widehat M\equiv_{\Pi,\mathcal V}M\ \Longleftrightarrow\ T_{\widehat M}^\pi v=T_M^\pi v\quad\forall\pi\in\Pi,\ v\in\mathcal V
$$

等价针对指定策略与函数族。用一个当前 critic 拟合 targets，只检验一个有限且会变化的查询集合；VE 不是说任意小模型都足够，也不是把 reward-only 预测称为完整环境模型。

Proper Value Equivalence（NeurIPS 2021）考察多步算子与策略价值固定点，给出适当策略族下的规划充分性。Distributional Model Equivalence（NeurIPS 2023）揭示它的另一边界：保持期望值不足以保持风险敏感决策。后者需保留回报分布或与指定风险目标相容的统计摘要。

手算反例：动作 A 确定获得 1；动作 B 以各半概率获得 −9 或 11；两者期望均为 1。只保持均值的模型可以把它们当作同一动作，但最差一半的平均回报分别为 1 和 −9。若目标由最大期望改为最大 lower-tail CVaR，旧模型无法回答新问题。这个反例不需要非平稳环境，改变查询规格本身就能造成模型不足。

$$
\mathcal S(\nu)=(\mathbb E_\nu[G],\mathbb E_\nu[G^2],\ldots),\qquad \mathcal S(\mathcal T^\pi\eta)=\mathcal T_{\mathcal S}^\pi\mathcal S(\eta)
$$

第二式表达摘要的 Bellman 闭合要求：更新后的统计应能由已有统计正确计算。完整分布保留更多信息；有限矩、分位点与投影各有局限，均值加方差通常不能识别任意尾部风险。不要把任意摘要都默认为满足这条闭合式。

**算法：模型 loss、查询误差和决策结果是三类证据**

1. 模型用途实验（拟议）：
  1. 固定真实数据、编码器、模型容量与优化预算
  1. 分别训练状态预测模型、当前价值等价模型、指定分布摘要模型
  1. 在未用于拟合的策略/后续价值/奖励查询上测 target 误差
  1. 对相同候选动作同时报告均值、选定尾部指标与真实动作排序
  1. 只改变风险目标，再只改变动力学，区分规格不足与环境漂移

持续模型研究可以由此提出清楚的假设：固定容量模型应按未来规划查询而非仅按观测频率分配表示。怎样发现新查询、保留旧风险事件，并及时判断模型的等价规格失效，仍是开放问题。模型等价定理也不提供有限采样下的安全保证。

<a id="research-model-frozen-visual-dynamics"></a>

## 研究专题 B · 冻结视觉特征后的动力学：DINO-WM 与 V-JEPA 2-AC

两条近年的视觉模型路线都把“看到什么”与“执行动作后发生什么”分开，但外部经验来源不同。DINO-WM（ICML 2025）在 DINOv2 patch 特征上用离线行为轨迹拟合动力学；V-JEPA 2（2025 首稿）先以无动作标注的视频学潜在预测，再冻结编码器，用机器人交互轨迹训练 2-AC 动作条件预测器。没有像素 decoder，并不意味着没有动力学数据。

$$
z_t=e_{\rm frozen}(o_t),\qquad \hat z_{t+1}=f_\theta(z_{t-L+1:t},a_{t-L+1:t}),\qquad \mathcal L_{\rm pred}=\sum_{k=1}^{H}\ell(\hat z_{t+k},\operatorname{sg}[e_{\rm frozen}(o_{t+k})])
$$

这是两类方法的教学性共同接口，历史长度、动作/机器人状态输入、损失距离与预测 rollout 方式以各原文为准。冻结 e 稳定了模型输出坐标；训练看到真实未来观测，不意味着测试规划可以读取它。

例子：同一物体的视觉颜色改变，冻结 encoder 可能依然给出相近的任务特征，模型因而迁移；摩擦系数改变却会使相同动作产生不同位置，即使视觉编码完全稳定，f 也必须更新。若把 e 也在线更新，第三种问题出现：旧模型预测的坐标与当前目标图像的坐标可能不再相同。三类变化需要独立实验。

| 数据与接口 | DINO-WM | V-JEPA 2 / 2-AC |
| --- | --- | --- |
| 预训练来源 | DINOv2 图像特征 | 大规模图像/视频的潜在预测 |
| 动作条件阶段 | 离线行为轨迹上预测 patch 特征 | 机器人轨迹上后训练动作条件模型 |
| 目标 | 观测目标的特征距离 | 机器人图像目标的潜在距离 |
| 部署使用 | 优化动作序列并滚动重规划 | 2-AC 模型支持图像目标 MPC |
| 持续更新证据 | 原文主要检验离线学习后规划 | 首稿主要检验冻结模型零样本机器人部署 |

先在作者公开检查点和已支持环境上验证训练/规划接口，再用单变量扰动做预测误差与真实目标成功率的联合评价。图像目标很近时仍可能物理碰撞；视觉目标代价也可能漏掉执行中的负奖励。因此若研究奖励控制，还需要额外 reward/风险接口，不能把视觉相似度默认成环境目标。

两个工作给 CRL 的机会是稳定且可迁移的起始表示，但未来学习必须另行检验。可比较冻结 e 仅更新 f、联合更新 e/f、以及使用兼容约束的三种方案，固定每步训练预算，测新后果学习、旧查询保留与控制恢复。预训练资源与在线维护资源应各自记录；首稿结果不是自动问题发现、option 构造或终生世界模型维护的证明。

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

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

状态是支持后续计算的内部信息；GVF 指定一个预测问题；RTRL 和资格迹规定信用如何传播。三者可以组合，但不是相互替代的算法名称。先理解给定策略的预测，再讨论预测怎样改善控制。

- [Does Zero-Shot Reinforcement Learning Exist?](https://yingwen.io/zh/continual-rl/research/#recent-zero-shot-forward-backward)
- [Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations](https://yingwen.io/zh/continual-rl/research/#recent-successor-flow-features)

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

Laplacian 描述行为图结构，奖励感知表示加入路径价值，METRA 学习有区别的行为，HIQL 利用离线目标轨迹，MaestroMotif 引入语言先验。它们承担不同的设计工作；生成技能、选择技能与组合技能需要分别评价。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [Reward-Aware Proto-Representations in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-reward-aware-proto-representations)
- [Laplacian Keyboard: Beyond the Linear Span](https://yingwen.io/zh/continual-rl/research/#recent-laplacian-keyboard)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

模型可提取性的理论说明某类能力需要什么知识，不指定唯一网络。Dreamer 研究潜在想象控制，STOMP 研究随机时长行为模型，DRAGO 研究旧模型知识保留。模型误差、查询策略和规划收益之间仍需实验连接。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [Mastering diverse control tasks through world models](https://yingwen.io/zh/continual-rl/research/#recent-dreamerv3-world-models)
- [Knowledge Retention in Continual Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-drago-model-retention)
- [General Agents Contain World Models](https://yingwen.io/zh/continual-rl/research/#recent-general-agents-world-models)
- [Laplacian Keyboard: Beyond the Linear Span](https://yingwen.io/zh/continual-rl/research/#recent-laplacian-keyboard)
- [The Value Equivalence Principle for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-value-equivalence-models)
- [Distributional Model Equivalence for Risk-Sensitive Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-distributional-model-equivalence)
- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://yingwen.io/zh/continual-rl/research/#recent-tdmpc2-decision-time-model)
- [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://yingwen.io/zh/continual-rl/research/#recent-dino-wm-feature-planning)
- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://yingwen.io/zh/continual-rl/research/#recent-vjepa2-action-conditioned)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [Knowledge Retention in Continual Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-drago-model-retention)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文提供定义和条件，架构讲座提出模块组织，算法论文检验特定机制。完整系统还要明确智能体、外部设计者和世界各自承担的工作；组件成立不自动意味着组合后的长期收益成立。

- [General Agents Contain World Models](https://yingwen.io/zh/continual-rl/research/#recent-general-agents-world-models)
- [The OaK Architecture: A Vision of SuperIntelligence from Experience](https://yingwen.io/zh/continual-rl/research/#recent-oak-architecture)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)
- [The Value Equivalence Principle for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-value-equivalence-models)
- [Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations](https://yingwen.io/zh/continual-rl/research/#recent-successor-flow-features)

### Reward-Respecting Subtasks for Model-Based Reinforcement Learning

Richard S. Sutton, Marlos C. Machado, G. Zacharias Holland, David Szepesvari, Finbarr Timbers, Brian Tanner, Adam White

Artificial Intelligence · 2023 · 支持方法与理论

#### 研究问题

学到一个能到达子目标的技能之后，为什么它仍可能不适合主任务规划？

#### 关键机制

STOMP 把子任务、option、模型和规划连起来。子任务保留原任务的路径奖励，并用带有特征偏好的终止价值表达目标；学习得到策略和终止规则后，再预测该行为的累计奖励与折扣终点。这样，技能不会因为只追求到达子目标而忽略途中代价。

#### 证据

论文用明确的小问题展示奖励感知子任务如何产生更有用的行为和规划模型。它提供的是可分析的构造链，而非只比较一个技能执行成功率。

#### 条件与限制

终止收益的约定是子任务定义的一部分，不能随意换成固定终点奖励。特征和子任务候选的选择尚不等于完整自主发现机制；实验也不构成整个 OaK 架构的验证。

#### 阅读与实验

在同一个绕路环境中比较“最短到达目标”和“保留路径奖励”的子任务。分别计算 option 的奖励模型、折扣终点模型与一次规划备份。

#### 原文与相关入口

- [期刊论文](https://doi.org/10.1016/j.artint.2023.104001)：STOMP 与奖励感知子任务的正式论文。
- [作者预印本](https://arxiv.org/abs/2202.03466)：最初预印本早于期刊年份；阅读停止收益的精确定义。

### Reward-Aware Proto-Representations in Reinforcement Learning

Hon Tik Tse, Siddarth Chandrasekar, Marlos C. Machado

NeurIPS 2025 · 2025 · 支持方法与理论

#### 研究问题

仅编码可达关系的表示，怎样进一步反映奖励与行动成本？

#### 关键机制

论文研究 default representation，将奖励或成本纳入对未来状态关系的表示，并给出动态规划与 TD 学习方法。由此提取的谱特征可以参与技能发现、奖励塑形和迁移。它沿着 SR 的后果预测思路前进，但不再把奖励完全留到最后的线性读出阶段。

#### 证据

作者提供表格问题中的推导，并用表示、技能和迁移实验展示奖励信息如何改变学得的结构。代码包含 SR、DR 的计算和在线表示学习实验。

#### 条件与限制

把奖励纳入表示会改变迁移边界：奖励或内部成本变化后，原表示可能需要重学。论文结果不能解释为任意新奖励下都能免费零样本迁移。

#### 阅读与实验

固定转移图，只改变一处通行成本，比较 SR 与 DR 的谱方向。随后检查新的 eigenoption 是改变了可达性，还是改变了对路径代价的偏好。

#### 原文与相关入口

- [论文与版本记录](https://arxiv.org/abs/2505.16217)：NeurIPS 2025；后续版本修订不改变会议年份。
- [作者实现](https://github.com/httse9/Reward-Aware-Proto-Representations)：从 minigrid_basics/examples 的表示计算与技能实验开始。

#### 作者代码

[原论文作者仓库。](https://github.com/httse9/Reward-Aware-Proto-Representations)

奖励感知表示、谱特征与相关 MiniGrid 实验。

### Laplacian Keyboard: Beyond the Linear Span

Siddarth Chandrasekar, Marlos C. Machado

arXiv 预印本 · 2026 · 支持方法与理论

#### 研究问题

从一组谱技能出发，能否解决超出原特征线性奖励空间的新任务？

#### 关键机制

Laplacian 特征先定义行为基，并借助后继特征预测各行为的后果。固定任务权重的价值组合受特征张成空间限制；论文进一步使用随状态变化的元策略，在不同位置组合已有行为。关键变化是组合规则从一组全局固定权重变为状态相关的行为选择。

#### 证据

论文对行为基与任务组合给出理论分析，并报告有限环境中的组合实验。它延续 eigenoptions 与 successor features 的路线，同时解释了为什么单纯线性读出会遇到表达边界。

#### 条件与限制

理论结论依赖具体的行为基、近似误差和任务条件。技能集合的长期生成、淘汰与非平稳模型维护仍是另外的问题；此处按预印本收录，不指定未经确认的会议。

#### 阅读与实验

构造一个必须在中途切换方向的奖励任务。分别比较固定权重的技能选择与状态相关切换，并解释性能差异来自哪里。

#### 原文与相关入口

- [作者预印本](https://arxiv.org/abs/2602.07730)：阅读线性张成空间的限制及状态相关组合机制。

### Knowledge Retention in Continual Model-Based Reinforcement Learning

Haotian Fu, Yixiang Sun, Michael Littman, George Konidaris

ICML 2025 · 2025 · 直接研究持续学习

#### 研究问题

当前任务不再访问旧区域时，世界模型怎样避免忘记那些区域的动力学？

#### 关键机制

DRAGO 将生成式旧经验、旧模型知识和探索结合起来。模型学习不只跟随当前奖励驱动的数据分布，还尝试维持对曾经学过区域的预测能力。它关注知识保留发生在模型里，而不只是保存旧策略输出。

#### 证据

论文在 MiniGrid 和连续控制任务中检验模型保留与任务表现，并给出原作者实现。关键设定包括共享状态与动力学、变化的任务奖励。

#### 条件与限制

已知任务切换、旧模型和数据资源是协议的一部分。共享动力学下的保留不能直接外推到动力学本身任意变化，也不是禁止重放的严格流式算法。

#### 阅读与实验

分别测量旧区域模型误差、旧任务回报与新任务学习速度。若模型误差改善而控制没有改善，再检查规划是否真正查询了被保留的知识。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/fu25f.html)：任务设定、模型保留和探索机制。
- [作者代码](https://github.com/YixiangSun/drago)：原文链接的 DRAGO 实验实现。

#### 作者代码

[作者 Yixiang Sun 的仓库；不使用同名第三方项目。](https://github.com/YixiangSun/drago)

DRAGO 的模型、重放、探索与实验。

### Mastering diverse control tasks through world models

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap

Nature · 2025 · 支持方法与理论

#### 研究问题

同一套世界模型训练与控制方法，能否减少跨任务重新设计损失和超参数的需求？

#### 关键机制

DreamerV3 从经验学习递归潜在状态、奖励和延续预测，再在潜在想象轨迹上学习 actor 和 critic。尺度稳健的表示与损失设计使同一配置可以适用于多种任务。模型是用于决策的学习接口，不必生成完整真实世界。

#### 证据

论文在大量视觉和状态控制任务上报告了广泛表现。关键含义是共享算法配置；这些结果主要来自分别训练的任务智能体，不是一个智能体按顺序学会全部任务。

#### 条件与限制

经验重放、批量训练和模型想象都有资源成本。模型偏差、表示遗忘与长期任务切换仍需要专门实验，不能由多任务覆盖范围自动推出持续学习能力。

#### 阅读与实验

把状态更新、模型训练、想象起点和策略更新四种分布分别写清。比较真实交互步数之外，还应记录想象步数和优化次数。

#### 原文与相关入口

- [Nature 原文](https://www.nature.com/articles/s41586-025-08744-2)：方法和任务协议；区分共享配置与单智能体持续学习。
- [作者维护的实现](https://github.com/danijar/dreamerv3)：公开实现的版本与论文实验环境应分别记录。

#### 作者代码

[作者发布的重实现；不把当前分支当作原论文实验的冻结快照。](https://github.com/danijar/dreamerv3)

DreamerV3 的作者维护公开实现及运行配置。

### General Agents Contain World Models

Jonathan Richens, David Abel, Alexis Bellot, Tom Everitt

ICML 2025 · 2025 · 支持方法与理论

#### 研究问题

能完成足够丰富的目标集合，是否意味着智能体内部已经包含可提取的环境预测知识？

#### 关键机制

论文在形式化条件下，将广泛多步目标上的行为能力与环境模型的可提取性联系起来。通过查询智能体对不同目标的行为，可以恢复关于环境后果的信息；目标集合和性能要求越强，所要求的预测知识也越强。

#### 证据

主要证据是给定假设下的理论结果，而不是某个世界模型架构在所有任务上击败无模型算法的实验。

#### 条件与限制

可提取模型不等于智能体显式保存一个 RSSM，也不意味着所有实用任务都需要重建全部环境。必要知识的结论不能代替如何高效学到它的算法。

#### 阅读与实验

列出定理要求的目标丰富性和查询能力，再尝试构造一个只会单一任务的反例。由此区分任务专门知识与支持广泛目标的预测模型。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2506.01622)：形式化设定、模型可提取性与证明。
- [David Abel 论文目录](https://david-abel.github.io/papers.html)：作者提供的 ICML 2025 发表信息及相关研究。

### The OaK Architecture: A Vision of SuperIntelligence from Experience

Richard S. Sutton

RLC 2025 讲座 / Oak Lab · 2025 · 定义与架构观点

#### 研究问题

持续学习是否只是在一个现成 actor–critic 上加入抗遗忘机制，还是需要重新安排知识构造与使用？

#### 关键机制

OaK 提出从经验持续形成状态、预测知识、子任务、时间抽象与模型，并让这些知识服务规划的架构方向。这里的重点是模块之间怎样产生可复用知识，而不只是保留某个固定策略网络的参数。

#### 证据

官方页面提供 Richard Sutton 的架构讲座与相关研究入口。STOMP、预测学习和在线特征学习等论文可以检验其中具体组件，但不能自动验证整体架构。

#### 条件与限制

这是研究愿景与架构讲解，不是一套已公布完整训练配方、统一基准结果和可复现端到端代码的系统。资源分配、问题生成、知识替换与模块相互干扰仍需明确算法。

#### 阅读与实验

为每个模块写出输入、输出、更新频率和资源上限。再选择一个双模块接口做可证伪实验，例如技能模型改善是否真的减少规划误差。

#### 原文与相关入口

- [Oak Lab 官方讲座页面](https://oaklab.ai/posts/the-oak-architecture)：讲座入口与架构研究方向。
- [Oak Lab 研究主页](https://oaklab.ai/)：区分已发表研究、技术文章和仍在预告中的项目。

### Reset-free Reinforcement Learning with World Models

Zhao Yang, Thomas M. Moerland, Mike Preuss, Aske Plaat, Edward S. Hu

TMLR 2025 · 2025 · 支持方法与理论

#### 研究问题

不能靠外部重置回到起点时，怎样兼顾探索新状态与持续获得对任务有用的经验？

#### 关键机制

MoReFree 在 goal-conditioned world-model 系统中交替练习评测目标、返回初始分布与探索目标；模型内的策略训练也偏向任务相关目标。返回行为通过真实动作实现，调度块结束不会将物理世界 reset。

#### 证据

作者在八个 reset-free 任务中与模型自由及模型式基线比较；公开环境、探索调度和 imagination training 实现。

#### 条件与限制

训练无 reset，但主要评价仍使用可重置的 episodic 测试。已给定初始与目标状态分布、世界模型和 replay 都是资源；这不是任意非平稳 CRL 或真实安全的完整保证。

#### 阅读与实验

把返回成本计入总步数，分别消融数据获取目标与模型内训练目标；检查外部 reward-free 是否仍依赖设计者提供目标示例。

#### 原文与相关入口

- [作者论文 v3](https://arxiv.org/html/2408.09807v3)：训练与评价协议、back-and-forth exploration 与目标分布。
- [TMLR 作者项目页](https://yangzhao-666.github.io/morefree/)：正式发表状态与作者代码链接。

#### 作者代码

[TMLR 作者项目页明确链接的官方实现。](https://github.com/yangzhao-666/MoReFree)

resetfree/env.py、goal_picker_wrapper.py、Dreamer/PEG 与目标条件实验。

### Does Zero-Shot Reinforcement Learning Exist?

Ahmed Touati, Jérémy Rapin, Yann Ollivier

ICLR 2023 · 2023 · 支持方法与理论

#### 研究问题

没有事先指定奖励时，怎样学一套预测表示，日后接收新奖励就能选行为？

#### 关键机制

Forward–Backward 表示联合学习行为条件的未来占用与奖励读出，而非先固定任意编码器再学习 successor features。新奖励被映射到任务向量，策略根据这个向量直接行动；该论文系统比较 FB 与多种 SF 基础特征。

#### 证据

原文在固定离线 replay buffers 上比较零样本任务迁移，借此把表示学习与探索数据的质量分开。不同特征与数据覆盖产生显著差异，不能仅靠“所有奖励”的理论目标预测实际效果。

#### 条件与限制

假定共享动力学与可用经验覆盖。无下游梯度更新不等于无预训练成本；有限秩、近似训练和奖励估计都有误差。新动力学、历史混叠和严格一次使用经验均须另测。

#### 阅读与实验

同一 buffer 对比随机特征、谱特征与联合 FB，再独立换 buffer。奖励读出误差、占用误差与新任务回报分别报告，避免把数据覆盖优势记成表示优势。

#### 原文与相关入口

- [作者原文](https://arxiv.org/abs/2209.14935)：2022 首稿，ICLR 2023；比较奖励表示、SF 与 FB。
- [作者研究平台](https://github.com/facebookresearch/controllable_agent)：README 直接关联两篇 FB 论文；该仓库已经归档。

#### 作者代码

[论文作者团队仓库；归档工程，依赖和旧环境需单独核验。](https://github.com/facebookresearch/controllable_agent)

FB 与 SF 的训练、固定数据实验及奖励查询示例。

### Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations

Haosen Shi, Jianda Chen, Sinno Jialin Pan

ICLR 2026 · 2026 · 支持方法与理论

#### 研究问题

能否直接学习多步未来状态的分布，并把它压缩为适合控制学习的特征？

#### 关键机制

SF² 以 flow matching 估计 successor measure，将条件向量场分解为未来位置及生成时间的投影与当前状态动作特征的乘积。特征进入 TD3/SAC 的 critic；线性的是向量场对条件特征的分解，critic 本身可以非线性。

#### 证据

正式原文给出 mixture Bellman 结构、生成式 bootstrap 与控制实验，并提供作者 JAX/Brax 仓库。实验研究在线收集数据下的 off-policy 控制，并使用 replay、批次与目标网络。

#### 条件与限制

“online policy learning”不代表 strict streaming。生成时间不是环境时间；向量场线性不保证任意奖励价值线性。文中与 SR 的小生成时间联系是近似动机，未证明递归 agent state 或任意持续变化下的充分性。

#### 阅读与实验

对齐模型调用与梯度预算，拆分直接预测、bootstrap、critic 联合训练。冻结特征后比较线性与非线性读出，再测新奖励和动力学变化，才能检验预测知识的可复用程度。

#### 原文与相关入口

- [ICLR 2026 正式原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48acf4b231771e693f42305b4c9b4c9f-Abstract-Conference.html)：第 2–3 节和算法附录；区分 flow 时间、环境时间与近似 SR 联系。
- [原文链接的作者实现](https://github.com/Shiien/successor-flow-representation-implementation)：SAC/TD3、flow 特征、对照和 sweep 配置。

#### 作者代码

[正式论文摘要直接链接的作者代码。](https://github.com/Shiien/successor-flow-representation-implementation)

基于 JAX/Brax 的 SF² 控制实验；不包含自动 GVF 问题发现或完整持续架构。

### Constructing an Optimal Behavior Basis for the Option Keyboard

Lucas N. Alegre, Ana L. C. Bazzan, André Barreto, Bruno C. da Silva

NeurIPS 2025 · 2025 · 支持方法与理论

#### 研究问题

状态相关的技能组合足够强时，是否仍须为每个新奖励保存一条完整策略？

#### 关键机制

OKB 联合扩充基础策略与 Option Keyboard 的元策略。线性支持方法选择需要补齐的任务权重，先训练现有基础上的组合，再检查无法表达的动作并新增基础，移除冗余项。优化的是可组合的行为基，而非仅增加技能数量。

#### 证据

正式原文分析基础数量与 convex coverage set 的关系，并在多任务领域检验规模与表现。附录提供元策略、新基础训练和角点枚举等实现细节；论文声明实验代码在 Supplemental Material。

#### 条件与限制

保证假定 NewPolicy 返回最优策略，且 TrainOK 能达到可表达的最优组合。近似 critic、有限训练、变化动力学不直接继承保证；非线性任务结论仅覆盖最优行为可由相应子策略组合的类别。

#### 阅读与实验

分别比较“增加基础”“只训练组合”和“去除冗余”。保留一组未用于基构造的奖励权重，记录基础规模、元策略成本、SF 误差与迁移回报。持续淘汰仍需未来任务效用检验。

#### 原文与相关入口

- [NeurIPS 2025 原文与补充材料入口](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0ab48777def88e73b50746a6011be0b0-Abstract-Conference.html)：算法 1–3、附录 A.3 的两个最优子程序假设及代码声明；未在本教材运行补充代码。
- [Option Keyboard 的经典桥梁](https://proceedings.neurips.cc/paper/2019/file/251c5ffd6b62cc21c446c963c76cf214-Paper.pdf)：cumulant 组合、GPE/GPI 与技能接口。

### The Value Equivalence Principle for Model-Based Reinforcement Learning

Christopher Grimm, André Barreto, Satinder Singh, David Silver

NeurIPS 2020 · 2020 · 支持方法与理论

#### 研究问题

模型容量有限时，必须预测全部状态细节，还是只须保持规划会查询的量？

#### 关键机制

Value equivalence 以策略集合和函数集合定义模型规格：模型对这些函数进行这些策略的 Bellman backup，应与真实环境相同。扩大查询族会缩小可接受模型族；它把“决策相关”从口号变成有条件的等价关系。

#### 证据

论文给出等价模型类的性质及有限实验，并解释若干隐式模型方法。后续 Proper Value Equivalence（NeurIPS 2021）研究策略价值固定点等价及规划充分性。

#### 条件与限制

少数当前 critic 的 backup 相同，不说明所有新奖励、新策略或风险目标都相同。精确算子等价与神经损失在样本上较小不同；奖励或查询族变化后须重新验证。

#### 阅读与实验

保存独立的 planner 查询集，直接测 target 误差和动作排序。用未参与模型拟合的价值函数检验迁移，并与像素误差对照，找出模型实际保留的信息。

#### 原文与相关入口

- [NeurIPS 2020 原文](https://papers.nips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)：VE 依赖策略与函数集合。
- [Proper Value Equivalence · NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/400e5e6a7ce0c754f281525fae75a873-Abstract.html)：多步算子、固定点与规划充分性；不是任意潜在网络的保证。

### Distributional Model Equivalence for Risk-Sensitive Reinforcement Learning

Tyler Kastner, Murat A. Erdogdu, Amir-massoud Farahmand

NeurIPS 2023 · 2023 · 支持方法与理论

#### 研究问题

模型正确预测期望回报，能否同时支持避开低概率灾难的决策？

#### 关键机制

论文证明 proper value equivalence 对风险敏感规划不足，再以回报分布与统计摘要定义更强的模型等价。完整分布覆盖更多风险度量，有限摘要则限制可支持的风险目标；相应 Bellman 闭合性质决定摘要能否递推。

#### 证据

正式原文包含理论、表格反例与大规模实验，并直接给出 distribution-equivalence 作者仓库。它检验的是特定风险敏感目标下的模型学习与规划接口。

#### 条件与限制

正确均值和方差不自动保证尾部概率或 CVaR；有限 quantile 表示与投影也有近似误差。静态模型等价不保证新环境中的风险校准，更不等于安全约束保证。

#### 阅读与实验

构造均值相同、尾部不同的两动作，先验证期望控制无法区分，再用指定风险度量评价。训练分布、投影和风险目标必须匹配，不能在评估时随意换风险函数。

#### 原文与相关入口

- [NeurIPS 2023 原文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b0cd0e8027309ea050951e758b70d60e-Abstract-Conference.html)：proper VE 的不足、统计摘要与 Bellman 闭合。
- [作者实现](https://github.com/tylerkastner/distribution-equivalence)：原文第 7 节直接链接的实验代码。

#### 作者代码

[正式原文第 7 节提供的作者仓库。](https://github.com/tylerkastner/distribution-equivalence)

分布模型等价与风险敏感实验；不提供任意任务的安全证书。

### TD-MPC2: Scalable, Robust World Models for Continuous Control

Nicklas Hansen, Hao Su, Xiaolong Wang

ICLR 2024 · 2024 · 支持方法与理论

#### 研究问题

如何让短期动力学与长期价值分工，并在动作选择时继续使用模型？

#### 关键机制

TD-MPC2 学习无需观测 decoder 的潜在动力学、奖励、价值与策略先验。决策时优化有限动作序列，用终点价值补上未展开的后果；执行第一步后，利用新观测重新规划。

#### 证据

正式会议原文报告 104 个在线任务和单一大型多任务智能体的实验。官方仓库包含模型训练与计划接口，适合与 Dreamer 的想象 actor 学习比较计算位置。

#### 条件与限制

跨任务共享超参数和多任务能力不是单条生命流中持续适应的证据。replay、任务条件、模型更新、决策延迟等成本需进入 CRL 协议；长程 critic 错误不能被短期模型精度自动修复。

#### 阅读与实验

固定模型，对比无终点价值、不同 horizon 和不同规划预算；再固定预算比较部署 actor 与决策时搜索。环境变化后同时记录模型校准、critic 误差和恢复收益。

#### 原文与相关入口

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html)：短期预测、终点价值、多任务协议。
- [作者实现](https://github.com/nicklashansen/tdmpc2)：训练、模型与 plan 函数分别阅读。

#### 作者代码

[作者维护的原论文代码。](https://github.com/nicklashansen/tdmpc2)

TD-MPC2 的单任务/多任务训练和决策时规划。

### DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning

Gaoyue Zhou, Hengkai Pan, Yann LeCun, Lerrel Pinto

ICML 2025 · 2025 · 支持方法与理论

#### 研究问题

预训练视觉表示能否直接成为动作后果预测与目标规划的接口？

#### 关键机制

冻结 DINOv2 空间 patch 特征，用离线动作轨迹学习未来特征预测器；测试时优化动作序列，让预测特征接近目标图像特征。没有重建图像、奖励模型或逆模型，不表示没有动作条件的动力学训练。

#### 证据

ICML 原文在六类环境检验视觉目标规划，作者仓库公开数据、部分检查点、训练与 CEM 规划入口。零样本指给定已训练模型后解决目标，无额外任务策略训练。

#### 条件与限制

依赖视觉预训练与离线交互覆盖；patch 相近不总等于任务完成或风险相同。原实验不证明冻结视觉表示能适应长期新物体、新动作语义或隐藏状态。

#### 阅读与实验

分别改变背景、物体属性、控制动力学与目标分布。把冻结 encoder 和联合更新 encoder 分开，对照视觉距离、真实成功与模型误差，观察表示漂移的依赖成本。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/zhou25t.html)：正式发表入口；早期 ICLR 投稿页不能替代此状态。
- [作者项目代码](https://github.com/gaoyuezhou/dino_wm)：train.py、plan.py、数据与已公开模型检查点范围。

#### 作者代码

[原作者 Gaoyue Zhou 的论文配套仓库。](https://github.com/gaoyuezhou/dino_wm)

DINO 特征预测、离线环境数据与目标规划；README 公开部分环境检查点。

### V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, Nicolas Ballas

arXiv 预印本（此处采用 2025 首稿） · 2025 · 支持方法与理论

#### 研究问题

无动作标注的视频预训练，与能接受机器人动作的规划模型之间还缺哪一步？

#### 关键机制

V-JEPA 2 先学被遮蔽视频的潜在特征预测；V-JEPA 2-AC 冻结编码器，再用机器人轨迹训练动作条件预测器。控制以目标图像的特征差为代价进行 MPC；视频理解、动作条件预测和真实控制是三个独立证据层。

#### 证据

2025 首稿报告以大规模视频预训练，再用不到 62 小时 DROID 交互视频后训练，在两个实验室以图像目标做真实机器人规划。论文单独讨论相机位置、长程规划与图像目标的局限。

#### 条件与限制

无任务奖励并不等于无动作、无机器人状态或无外部数据。零样本部署未持续更新模型，也未发现和维护 options。官方仓库现含 V-JEPA 2.1，复现首稿须记录配置和模型版本。

#### 阅读与实验

按视觉编码、动作坐标、后果模型、目标代价逐项做迁移检验。若引入在线更新，记录模型更新使旧目标接口失效的程度，测未来交互收益，而非仅用 frozen probe 证明 CRL。

#### 原文与相关入口

- [2025 首稿](https://arxiv.org/abs/2506.09985v1)：action-free 预训练、2-AC 后训练、真实规划与第 4.3 节限制；此处不赋予未核实会议状态。
- [Meta FAIR 官方实现](https://github.com/facebookresearch/vjepa2)：包含 V-JEPA 2、2-AC 和较新的 2.1；版本不能混用。

#### 作者代码

[Meta FAIR 官方仓库；首稿模型与后续版本需按配置区分。](https://github.com/facebookresearch/vjepa2)

官方视频表征与动作条件模型；数据、机器人部署条件与检查点分别核验。


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

- [作者原文](https://arxiv.org/abs/2209.14935)：2022 首稿，ICLR 2023；比较奖励表示、SF 与 FB。

- [作者研究平台](https://github.com/facebookresearch/controllable_agent)：README 直接关联两篇 FB 论文；该仓库已经归档。

- [ICLR 2026 正式原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48acf4b231771e693f42305b4c9b4c9f-Abstract-Conference.html)：第 2–3 节和算法附录；区分 flow 时间、环境时间与近似 SR 联系。

- [原文链接的作者实现](https://github.com/Shiien/successor-flow-representation-implementation)：SAC/TD3、flow 特征、对照和 sweep 配置。

- [NeurIPS 2025 原文与补充材料入口](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0ab48777def88e73b50746a6011be0b0-Abstract-Conference.html)：算法 1–3、附录 A.3 的两个最优子程序假设及代码声明；未在本教材运行补充代码。

- [Option Keyboard 的经典桥梁](https://proceedings.neurips.cc/paper/2019/file/251c5ffd6b62cc21c446c963c76cf214-Paper.pdf)：cumulant 组合、GPE/GPI 与技能接口。

- [NeurIPS 2020 原文](https://papers.nips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)：VE 依赖策略与函数集合。

- [Proper Value Equivalence · NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/400e5e6a7ce0c754f281525fae75a873-Abstract.html)：多步算子、固定点与规划充分性；不是任意潜在网络的保证。

- [NeurIPS 2023 原文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b0cd0e8027309ea050951e758b70d60e-Abstract-Conference.html)：proper VE 的不足、统计摘要与 Bellman 闭合。

- [作者实现](https://github.com/tylerkastner/distribution-equivalence)：原文第 7 节直接链接的实验代码。

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html)：短期预测、终点价值、多任务协议。

- [作者实现](https://github.com/nicklashansen/tdmpc2)：训练、模型与 plan 函数分别阅读。

- [ICML 2025 原文](https://proceedings.mlr.press/v267/zhou25t.html)：正式发表入口；早期 ICLR 投稿页不能替代此状态。

- [作者项目代码](https://github.com/gaoyuezhou/dino_wm)：train.py、plan.py、数据与已公开模型检查点范围。

- [2025 首稿](https://arxiv.org/abs/2506.09985v1)：action-free 预训练、2-AC 后训练、真实规划与第 4.3 节限制；此处不赋予未核实会议状态。

- [Meta FAIR 官方实现](https://github.com/facebookresearch/vjepa2)：包含 V-JEPA 2、2-AC 和较新的 2.1；版本不能混用。
