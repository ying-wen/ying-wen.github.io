# 持续智能体架构：模块接口、更新调度与长期评价

各模块单独能学，不代表接在一起就能持续改善；它们究竟交换什么、何时更新、如何共享有限计算？

## 本章内容

- 用状态、参数、数据接口与更新调度描述一个持续学习系统。
- 追踪同一条 experience 如何服务控制、GVF 与模型学习，明确每个目标和概率的参数版本。
- 在固定预算下集成 agent，分析表示漂移、模型偏差与各模块的作用。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### Agent state

智能体从历史递推得到的决策输入 $z_t$；不必等于环境真实状态，但必须保留任务需要的信息。

### Prediction / control / model

预测回答指定策略下未来会怎样；控制选择外部收益高的行为；模型预测行为后果以便模拟计算。三者具有不同目标。

### Planning

使用模型产生的预测进行价值或策略计算，不直接增加真实环境经验；算力和模型错误须计入评价。

<a id="lesson-setting"></a>

## 1 · 持续学习模块的相互依赖

设一个机器人要持续感知、工作、学习新技能并适应磨损。它需要用历史判断当前情况，对多种未来结果形成预测，选择当前行为，形成可重复使用的技能，再用模型推演这些技能。若每个模块都依赖其他模块已经正确，整个系统就会在启动时陷入循环。架构工作的核心，是指定这些尚不准确的模块怎样共同成长。

考虑一个有限系统：数据来自一条交互流，内部包含六个手工记忆状态、两个原子动作、一个控制表、一个固定策略 GVF 和一个经验转移模型。它足以说明数据与参数的依赖关系。自动状态构造、技能发现以及跨模块元学习是在这些接口之上的进一步研究问题。

| 模块 | 输入 | 输出 / 持久状态 | 所需信息约束 |
| --- | --- | --- | --- |
| State constructor | 旧内部状态、动作、新观测 | 内部状态、表示参数、可选敏感性迹 | 不读取真实隐藏状态或任务切换 ID |
| Behavior / control | 内部状态、控制价值或 actor | 动作及采样时行为概率 | 概率必须对应实际采样策略 |
| Predictive knowledge | 转移、累积量、折扣、目标策略 | 一组 GVF 值与各自参数 | 每个问题有独立的预测定义 |
| Model learning | 状态动作与真实后果 | 奖励、终点、时长模型 | 区分已学模型与环境真模型 |
| Planning | 已学模型、价值、计算预算 | 更新后的价值或行为偏好 | 模拟计算不增加真实交互数 |
| Meta / resource allocation | 误差、后续验证信号、成本 | 步长、特征、问题及规划预算 | 后续数据只在发生后用于更新 |

说“同一条 experience 多种用途”不等于所有模块共享一个 loss。它们可以共享表示，但各自预测什么、控制什么、何时 detach 都必须明确。否则一个有用的辅助任务可能被误当成外部目标，或者目标策略概率被错误地从更新后的 actor 读取。

<a id="lesson-derive"></a>

## 2 · 一次交互的参数与目标

$$
z_t=f_{\phi_t}(z_{t-1},a_{t-1},o_t),\quad a_t\sim b_{\theta_t}(\cdot\mid z_t),\quad e_t=(z_t,a_t,r_{t+1},o_{t+1},b_t(a_t\mid z_t))
$$

状态形成后，智能体用此刻参数采样动作，并保存其行为概率。后续策略更新不改变这条经验的采样分布。

收到新观测后，在约定的旧表示版本下形成 $z_{t+1}$，计算本次真实数据的目标，再更新参数。本例控制采用平均奖励目标；GVF 的目标策略固定为 $\pi(a=1\mid z)=1$，折扣为 $0.8$。二者虽然使用相同奖励，预测对象仍然不同。

$$
\begin{aligned}\delta_t^{\rm control}&=r_{t+1}-\bar g_t+\max_a Q_t(z_{t+1},a)-Q_t(z_t,a_t),\\\delta_t^{\rm pred}&=r_{t+1}+0.8\,v_t(z_{t+1})-v_t(z_t),\\\rho_t&=\mathbf1[a_t=1]/b_t(a_t\mid z_t).\end{aligned}
$$

控制的 max 对应最优动作目标；GVF 的 ρ 对应固定 target policy。行为率、最优奖励率和 GVF 预测值必须分别命名与记录。

$$
\begin{aligned}Q(z_t,a_t)&\leftarrow Q_t(z_t,a_t)+\alpha\delta_t^{\rm control},\\\bar g&\leftarrow\bar g_t+\eta\alpha\delta_t^{\rm control},\\v(z_t)&\leftarrow v_t(z_t)+\alpha\rho_t\delta_t^{\rm pred}.\end{aligned}
$$

这三项使用同一更新前快照。将来若共享神经网络，需要明确定义 loss 合并、detach 和梯度冲突处理，不能依靠执行顺序偶然决定目标。

经验模型记录每个 (z,a) 后看到各 (r,z′) 的次数，形成经验条件分布。它不同于只记最后一次转移：后者会把真实随机性误当作不断变化的确定性结果。我们使用的有限表格模型容量有界，但累积计数会对外部变化迟钝，这正好暴露模型维护问题。

$$
\widehat p_t(r,z'\mid z,a)=\frac{N_t(z,a,r,z')}{\sum_{\tilde r,\tilde z}N_t(z,a,\tilde r,\tilde z)},\quad \delta^{\rm plan}=\sum_{r,z'}\widehat p_t(r,z'\mid z,a)[r-\bar g+\max_{a'}Q(z',a')]-Q(z,a)
$$

规划选择已观察的 (z,a) 做模型期望 backup。这里是教学性的表格 Dyna 组合；奖励率只从真实数据学习，规划只改 Q，这项选择写进接口并测试。

另一些 differential planning 方法也从模型更新奖励率，因此上述接口是一项具体设计选择。模型样本的分布与真实时钟不同；若扩展为规划也更新 $\bar g$，需要说明模型采样分布、时长和更新比例，不能将模拟次数直接当成实际经历的时间。

<a id="lesson-schedule"></a>

## 3 · 真实交互与规划的更新调度

**算法：算法伪代码**

1. 初始化 state memory、Q、ḡ、GVF、model 与固定规划预算 B
1. 每个真实环境步：
  1. 1. 用当前 agent state 计算行为分布；采样动作并保存 b(a|z)
  1. 2. 环境推进一次，返回真实 reward 和 observation
  1. 3. 用明确的表示版本计算 z′，保留本步必要中间量
  1. 4. 用更新前参数同时计算 control/GVF/model targets
  1. 5. 更新各学习器；模型只吸收此次真实 experience
  1. 6. 做恰好 B 次模型 backup；轮转选已知 state-action
  1. 7. 记录真实收益、预测误差、模型误差、更新成本
  1. 8. 若启用表示/技能重构，按版本规则处理依赖模块
  1. 9. 携带 memory 到下一步；不在记录窗口或隐藏变化时 reset

第 4 步的“同时”指语义上共同读取旧快照，不要求并行硬件执行。第 6 步的每次 planning 可以读取前一次 planning 更新后的 Q，这是异步迭代；要区别于第 4 步对真实经验目标的快照约定。代码显式写出这两个边界，避免一边修改 Q 一边意外改变本步 GVF 目标。

如果 GVF 数量 K、技能数 O 或网络规模持续增长，每步成本也可能增长。一个终生系统需要给新对象分配预算并淘汰低价值对象：例如只更新部分预测问题、限制模型缓存、分配固定 planning calls。自动增长不等于固定预算下的持续学习。

<a id="lesson-construction"></a>

## 4 · 从 Dyna 扩展到 subtask → option → model → planning

Dyna 的基础链路是“真实数据学价值和模型，模型供模拟价值更新”。时间抽象扩展还需要四个不同对象。Subtask 定义为何要学某种行为；option 定义具体如何执行以及何时停止；option model 描述实际执行产生的外部奖励、终点与时长；planner 用这些后果与当前主任务价值组合。学习技能用的人工奖励不能不加区分地写进主任务模型。

$$
\begin{aligned}r_o(z)&=\mathbb E\!\left[\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}\mid z,o\right],\\p_o(z,z')&=\mathbb E[\gamma^\tau\mathbf1(Z_{t+\tau}=z')\mid z,o],\\(T_oV)(z)&=r_o(z)+\sum_{z'}p_o(z,z')V(z').\end{aligned}
$$

折扣 option model 已把 γ^τ 包进终点权重，因此 planner 不能再乘一次 γ。τ 随机时也不能把 E[γ^τV] 简化成 γ^{Eτ}E[V]。平均奖励 option model 则保留奖励、时长和未折扣转移，使用 R−gτ。

例如学习“到门口”为了形成可复用 option，终止奖励可用于驱动这个 subtask；但主任务可能是递送物品，门口本身没有外部奖励。Planner 必须知道此技能真实消耗多少步、沿路得到什么外部收益、到达哪里，才能判断其价值。STOMP 的研究价值在于把这个链路作为可连接的学习问题，而不是在图中简单把 option 画成一条箭头。

Successor features 提供另一种可复用接口：若外部奖励近似 r=φ(s,a,s′)ᵀw，则策略的 successor feature 预测折扣特征累积，价值近似 ψπᵀw。它方便奖励改变时快速重算价值，但已知特征线性分解、固定策略及动力学变化都是重要边界；并非等价于任意世界模型。

<a id="lesson-drift"></a>

## 5 · 表示漂移与模块一致性

一个模型昨天学会“特征第 3 维高意味着门已开”，今天 state constructor 重构后第 3 维代表速度；即使 model 参数没变，语义也失效了。将 learned state 当作普通固定数组，会掩盖这类依赖。需要同时考虑表示误差、模型误差和控制目标，而不是分别验证每个模块 loss 下降。

| 设计选择 | 怎样保持接口 | 代价或未解决点 |
| --- | --- | --- |
| 固定状态编码 | 整个实验中保持坐标语义 | 无法研究自动 state construction |
| 保存原始经验重编码 | 新表示版本上重算状态和训练模型 | 额外存储/计算，严格 streaming 下可能不允许 |
| 慢表示 / 快预测器 | 限制表示变化，让依赖者追踪 | 时间尺度需要调节，仍无任意漂移保证 |
| 显式版本与兼容检查 | 模型记录表示版本，失配时作废或迁移 | 作废丢知识，迁移本身也是学习问题 |
| 联合一致性训练 | 共享或约束 representation/model/value | 损失冲突、梯度泄漏和新优化问题 |

处理网络单元回收也类似：如果一个隐藏特征被替换，依赖它的预测头、模型输入输出、eligibility trace 以及元梯度敏感性是否仍有旧语义？应明确哪些状态清零、哪些变换、哪些保留。把一个塑性机制接到架构里，影响不只发生在该层权重。

<a id="lesson-example"></a>

## 6 · 手算同一条样本的两个不同目标

设当前状态 z0 的 Q=[1,2]，下一状态 z1 的 Q=[3,4]，ḡ=0.5；GVF 值为 v(z0)=2、v(z1)=5。实际采动作 1 的行为概率为 0.25，收到奖励 1。控制 error=1−0.5+4−2=2.5；预测 error=1+0.8×5−2=3；目标策略固定选动作 1，故 ρ=4。

取 α=0.1、η=0.1：Q(z0,1) 从 2 变成 2.25，ḡ 从 0.5 变成 0.525，GVF 从 2 变成 3.2。三个量共用一次 experience，却遵循不同目标。若采的是动作 0，控制依然学习该动作后果，而该固定目标策略 GVF 的重要性比为 0，本步不更新。

若随后做一次模型规划，Q 可再次改变；本页约定 ḡ 与 GVF 不从这条 imagined update 改变。测试通过同时运行 B=0 与 B=5，验证额外规划仅改变允许改变的模块。这类模块隔离测试比只看最终 reward 曲线更容易发现隐藏耦合。

<a id="lesson-code"></a>

## 7 · 不间断交互的集成实验

明确的参数快照、保存行为概率、经验分布模型和固定预算 round-robin planning。

```python
class ModularAgent:
    """Hand-designed Markov state; tabular control/GVF/one-step model.
    The model stores empirical counts of (reward,next_state) per state/action.
    Planning changes q only. Reward-rate learning uses REAL data only here.
    """
    def __init__(self, states=6, alpha=0.05, eta=0.05, gamma=0.8):
        self.q = [[0., 0.] for _ in range(states)]
        self.gvf = [0.] * states
        self.rate = 0.
        self.model = {}
        self.planning_cursor = 0
        self.alpha, self.eta, self.gamma = alpha, eta, gamma

    def action_probabilities(self, state, epsilon=0.2):
        best = max(range(2), key=lambda a: self.q[state][a])
        prob = [epsilon / 2., epsilon / 2.]
        prob[best] += 1. - epsilon
        return prob

    def observe(self, state, action, reward, next_state, behavior_probability,
                planning_budget=0):
        if behavior_probability <= 0 or planning_budget < 0:
            raise ValueError("invalid recorded probability or planning budget")
        # All REAL targets use one parameter snapshot before any mutation.
        q_error = reward - self.rate + max(self.q[next_state]) - self.q[state][action]
        prediction_error = reward + self.gamma*self.gvf[next_state] - self.gvf[state]
        ratio = (1.0 if action == 1 else 0.0) / behavior_probability
        self.q[state][action] += self.alpha*q_error
        self.rate += self.eta*self.alpha*q_error
        self.gvf[state] += self.alpha*ratio*prediction_error
        outcomes = self.model.setdefault((state, action), {})
        outcomes[reward, next_state] = outcomes.get((reward, next_state), 0) + 1
        # A deterministic scheduler makes the simulated-update budget inspectable.
        keys = sorted(self.model)
        for _ in range(planning_budget):
            s, a = keys[self.planning_cursor % len(keys)]
            self.planning_cursor += 1
            outcomes = self.model[s, a]
            count = sum(outcomes.values())
            delta = sum(n * (r - self.rate + max(self.q[sp]) - self.q[s][a])
                        for (r, sp), n in outcomes.items()) / count
            self.q[s][a] += self.alpha*delta
        return {"control_error": q_error, "prediction_error": prediction_error,
                "ratio": ratio, "planning_updates": planning_budget}


def architecture_run(seed=7, planning_budget=0, steps=18000):
    rng, agent = random.Random(seed), ModularAgent()
    cue, phase = rng.randrange(2), 0
    rewards, trace = [], None
    for t in range(steps):
        # State is a hand-coded (phase, remembered cue), not a discovered representation.
        state = 2*phase + cue
        probabilities = agent.action_probabilities(state)
        action = 0 if rng.random() < probabilities[0] else 1
        mapping = 0 if t < steps // 2 else 1  # hidden change, never supplied to agent
        reward = float(action == (cue ^ mapping)) if phase == 1 else 0.0
        if phase == 2:
            phase, cue = 0, rng.randrange(2)
        else:
            phase += 1
        next_state = 2*phase + cue
        trace = agent.observe(state, action, reward, next_state,
                              probabilities[action], planning_budget)
        rewards.append(reward)
    block = steps // 3
    means = [sum(rewards[i*block:(i+1)*block]) / block for i in range(3)]
    return {"window_reward_rates": [round(v, 6) for v in means],
            "optimal_rate_estimate": round(agent.rate, 6),
            "model_entries": len(agent.model), "last_update": trace}


def architectures_demo():
    print("architectures", {"without_planning": architecture_run(),
          "one_simulated_update_per_step": architecture_run(planning_budget=1),
          "scope": "fixed-budget toy integration; not an OaK implementation"})
```

Python 3.10+，无外部环境、无 GPU、无网络

```sh
python lifelong_algorithms_lab.py architectures
python lifelong_algorithms_lab.py test
```

环境每三个原始步骤形成一个自然周期：阶段 0 显示二值 cue，阶段 1 必须依据记住的 cue 选动作并获得奖励，阶段 2 过渡到下一 cue。Agent state 是手工编码的 (phase,cue)，共六种。半程隐藏地翻转 cue→正确动作的映射；程序不把切换时间或映射传给 agent，也不清参数、记忆或模型。这个自然周期不是一个对 agent reset 的训练回合。

对比每步零次和一次模型更新。最优外部奖励率为 $1/3$；使用 $\epsilon=0.2$ 探索的实际率接近 $0.3$。固定随机种子 7、运行 18000 步，无规划的三个窗口实际率约为 $[0.299667,0.2895,0.300833]$；有规划时为 $[0.299667,0.202167,0.300833]$，变化附近反而更差。规划条件下最终 $\bar g\approx0.31847$，无规划约为 $1/3$。奖励率估计与行为实际率的对象不同，因此不能将这两个量互换。累积模型仍保留旧映射，可能解释适应迟缓；需通过改变模型记忆长度、多个种子和预算匹配进一步检验。

这个实验包含手工记忆、一个固定 GVF 和原子动作模型。它适合学习模块更新关系；自动记忆学习、预测问题发现、option 构造与元学习需要额外算法。研究时可逐项加入模块，测量它究竟改善了状态区分、预测准确性、规划效率还是控制回报。

<a id="lesson-evaluation"></a>

## 8 · 从固定策略评价到持续学习过程

传统实验常在训练结束后冻结策略，再评价若干回合。这个协议回答最终策略有多好，却不能完整描述一个始终学习的智能体。后者的对象包括参数更新规则、记忆、优化器状态与资源分配；相同当前策略可能因为内部学习状态不同而具有不同的未来表现。

$$
H_t=(O_0,A_0,R_1,O_1,\ldots,O_t),\qquad M_{t+1}=U(M_t,A_t,R_{t+1},O_{t+1}),\qquad A_t\sim b(\cdot\mid M_t).
$$

$H_t$ 是完整交互历史，$M_t$ 是智能体实际保存的有限信息，包含递归状态、参数及学习状态。环境可由历史条件分布描述；智能体必须用有限 $M_t$ 近似利用它，而不能存储无限历史。

Elelimy、Szepesvari、White 与 Bowling 在 RLC 2025 提出以 history process 和面向持续学习的 deviation regret 重新讨论这一评价对象。其启发是把学习中的变化和偏离行为的后果纳入形式化，而不只比较一个与时间无关的最优策略。这是一条研究立场，并不意味着有限 MDP 或平均奖励理论失效；关键是说明模型表达了环境什么信息，以及评价量是否覆盖学习的代价与收益。

| 评价条件 | 回答的问题 | 实现时的约束 |
| --- | --- | --- |
| 冻结参数的策略评测 | 已学行为在指定分布上有多好 | 不把这条曲线当作完整生命期表现 |
| 持续学习的全程收益 | 适应期间的损失能否被后续收益补偿 | 恢复、探索、失败期间同样计时 |
| 相同数据的学习 probe | 内部学习能力是否改变 | 固定数据顺序、容量与更新预算 |
| 相同实时计算预算 | 更多规划或预测是否值得其延迟 | 报告峰值内存、每步耗时及动作频率 |

AgarCL 将这些耦合问题放在同一环境中：细胞的质量改变移动速度与视觉尺度，部分可观测性要求记忆，吞噬与避敌要求长期控制。这里可以区分两种变化：固定规则下状态相关的动力学仍可构成平稳的完整状态 MDP，而智能体看到的分布与控制尺度会随行为持续改变。仅从观测变化不能推出环境转移核本身随时间变化。

平台的完整游戏没有统一回合重置，但被吞噬的细胞会局部重生，其余世界状态继续存在。动作包含连续方向和分裂等离散选择；DQN 基线将方向离散化，PPO/SAC 使用混合动作接口。论文测试的可塑性方法未在完整游戏中建立持续胜任能力，并用小游戏分离探索、信用分配等困难。因此，一个有用的复现路线是先核对小游戏中的模块机制，再进入完整游戏，保留失败、重生和学习成本。

<a id="lesson-branches"></a>

## 9 · 模块假设与研究路线

| 假设 | 只改哪一处 | 必需的机制检查 |
| --- | --- | --- |
| 更好的 state 减少混叠 | memory / representation | 同观测不同历史的预测与动作是否可区分 |
| GVF 作为知识改善控制 | 共享特征或决策输入 | 固定控制预算，比较真实预测与随机辅助任务 |
| 新技能节省规划深度 | option 和 model 接口 | 模型奖励/时长是否为真实外部后果 |
| 规划帮助适应 | 模型与 planning budget | 模型误差、旧数据权重、每步真实计算成本 |
| Meta-learning 改善未来学习 | 步长/目标/更新规则 | 是否穿过未来数据，是否多用元训练分布 |
| 替换机制维持长期能力 | 特征或技能的 generate-and-test | 被替换对象的全部依赖状态是否一致处理 |

Alberta Plan 与 OaK 将状态、预测、控制、规划、时间抽象和元学习组织为相互依赖的研究方向。它们是理解长期目标的研究纲领；具体实验仍需给出可执行的子系统、接口与预算。完整整合及统一评估属于持续推进的开放问题。

“架构 A 比 B 好”很容易混入更多计算、更多先验、更多模型查询或更丰富目标。更有解释力的问题是：“在相同真实步数、固定模型容量和固定每步更新预算下，自动 option model 比原子模型减少多少规划误差或适应延迟？”先给出可测量接口，才有可积累的结论。

<a id="lesson-check"></a>

## 10 · 自测与集成实验

- 为什么行为概率必须随经验保存？它描述采样当时分布；更新后策略不能倒过来解释历史动作。
- 同一个 reward 能否同时用于平均奖励控制和折扣 GVF？可以，但 discount、target policy 与意义不同，需要两个目标和估计器。
- 为什么模型有 loss 下降，planning 仍可能伤害行为？训练分布、规划查询分布和非平稳旧数据不同；小局部误差可以被反复 backup 放大。
- 在日志上切成多个窗口，能清空 recurrent state 吗？不应无声明清空，否则改变了任务协议。
- 只有多个模块接在一起就算 OaK 实现吗？不算；需逐项给出目标构造、学习器、接口、调度、预算以及开放环节。

动手题：把经验模型从终生累计计数改成带固定容量的近期统计，保持规划次数不变，研究隐藏变化后恢复速度与稳态方差。再把 cue memory 去掉，让两个不同历史映射到同一状态，观察加更多 planning 是否能修复信息丢失。预期不能：计算无法凭空恢复被状态表示丢弃的信息。

## 本章的实验设计

每次消融都对应一个接口假设。报告中间量和长期收益。完整系统的改进不能直接归因于其中一个模块。

设定：先固定技能和消费者检验预测模块，再比较预测开关×技能开关；在明确收益链后接模型或规划。

- 每个输出有明确消费者、参数版本与可用时刻。
- 改变诊断 oracle 不会通过中间模块进入部署输入。
- 完整快照包含各模块状态和事件队列；恢复能力按真实支持声明。

对照：四格交互设计与同容量基座；已训练系统断链与无模块重新训练分别报告；准确模块诊断及固定/在线版本

记录：预测准确、技能成功与真实回报三层指标；表示/目标/模型版本漂移；全部模块的计算、存储及延迟

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-planning)

## 学习与研究衔接

连接模块后才出现的数据分布反馈，需要单独验证。MARL 还要控制其他智能体变化与训练信息权限。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-architectures) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=architectures) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=architectures)

<a id="chapter-code"></a>

## 下载与运行

六状态、双动作、手工记忆的集成教学骨架；不是完整 OaK/STOMP 性能复现。

[下载 lifelong_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)

```sh
python lifelong_algorithms_lab.py architectures
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton · Dyna, an integrated architecture for learning, planning, and reacting](https://doi.org/10.1145/122344.122377)：模型学习、直接学习与模拟规划的经典接口；本页骨架受此启发，并明确另外加入平均奖励与 GVF。

- [Sutton, Bowling & Pilarski · The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173)：研究纲领及子问题组合，不是完整通用 agent 的实现报告。

- [Sutton et al. · Reward-Respecting Subtasks / STOMP](https://arxiv.org/abs/2202.03466)：Subtasks、Options、Models、Planning 的学习链路及 reward-respecting 构造。

- [Barreto et al. · Successor Features for Transfer in Reinforcement Learning](https://arxiv.org/abs/1606.05312)：奖励线性分解与预测特征累积的可迁移接口，注意策略和动力学假设。

- [Oak Lab · The OaK Architecture](https://oaklab.ai/posts/the-oak-architecture)：Richard Sutton 的架构研究纲领与讲座入口。

- [作者代码 · average-reward-methods](https://github.com/abhisheknaik96/average-reward-methods)：control_agents.py 中的直接学习与 planning_update，适合比较奖励率更新调度。

- [Elelimy et al. · Rethinking the Foundations for Continual RL](https://rlj.cs.umass.edu/2025/papers/Paper243.html)：RLC 2025：history process、deviation regret 与持续学习评价对象。

- [Mohamed et al. · The Cell Must Go On: Agar.io for Continual RL](https://arxiv.org/abs/2505.18347)：2025 年提出、后续修订的 AgarCL：完整环境与分解小游戏，讨论探索、记忆、信用分配和可塑性。

- [AgarCL 作者环境](https://github.com/machado-research/AgarCL)：C++ 仿真与 Python 接口；局部重生、观测和混合动作需按环境配置理解。

- [AgarCL 作者基线 · PPO 混合动作实现](https://github.com/machado-research/AgarCL-benchmark/blob/main/PPO_multi_heads_full_action.py)：同仓库包含 DQN_full_action_set.py、SAC_full_action_set.py 及 recurrent 版本；完整游戏实验需匹配动作、参数搜索与运行预算。
