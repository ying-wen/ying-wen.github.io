# 持续智能体架构：模块接口、更新调度与长期评价

各模块单独能学，不代表接在一起就能持续改善；它们究竟交换什么、何时更新、如何共享有限计算？

## 本章内容

- 用状态、参数、数据接口与更新调度描述一个持续学习系统。
- 追踪同一条 experience 如何服务控制、GVF 与模型学习，明确每个目标和概率的参数版本。
- 在固定预算下集成 agent，分析表示漂移、模型偏差与各模块的作用。

<a id="problem-definition"></a>

## 本章的问题定义

状态、预测、控制、技能、模型和规划共享有限资源；各模块单独可学并不保证组合后目标与时序一致。

### 给定条件与符号

- 各模块的输入/输出、目标、状态和更新规则。
- 真实交互接口、共享参数、总内存和计算预算、设计/测试权限。

### 需要求解的对象

完整可执行智能体与可反驳的模块协同假设，明确每条真实经验被哪些学习过程如何使用。

### 信息与数据权限

$M_t$ 是全部持久内部信息；行为 $b(\cdot\mid M_t)$ 产生动作，记录采样时概率。预测的目标策略、模型题目和控制目标各自固定或按声明规则变化。

$$
M_{t+1}=U(M_t,A_t,R_{t+1},O_{t+1}),\qquad J_T(U,b)=\mathbb E\!\left[\sum_{t=0}^{T-1}R_{t+1}\right]
$$

$U$ 是包含全部模块调度的更新，$b$ 是行为规则，$T$ 为预定寿命；评价还附带资源和干预成本。模块局部损失只支持相应子问题，不能简单相加就称等于此整体寿命目标。

### 成立条件与解的含义

- 各模块计时、目标参数版本、行为概率和表示版本明确，更新依赖已到达数据。
- 共享表示改变时需检查旧价值、模型和技能坐标；总成本包含规划、teacher与统计。

判断准则：同一条样本的控制与固定策略预测分别符合手算target；快照、模型折扣与资源记账一致；以等预算消融检验完整寿命收益、失败和模块漂移。

### 适用边界

- Alberta/OaK研究纲领不等于已验证的通用智能体。
- 模块loss各自下降不证明组合寿命收益上升。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)：状态给各模块提供决策输入，表示漂移会使多个接口同时变化。

- 组合不同学习问题 · [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)：技能与模型接口规定可重用后果，规划不得重复折扣或使用过期技能模型。

- 组合不同学习问题 · [持续控制与学习智能体比较](https://yingwen.io/zh/continual-rl/algorithms/control/)：架构必须作为完整行动与学习过程评价，而非只看最终冻结策略。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

不同目标共享经验和表示，更新顺序可隐式改变标签；有限计算还决定哪些模块能及时适应。

### 本章的核心思路

把目标和快照版本落实为显式接口，再把真实更新、模型更新与规划按预算调度。

1. [为同一经验标注不同题目](#lesson-derive)：因为控制最大化与GVF目标策略评价不同，先缓存采样概率和旧参数，分别构造误差。

2. [使调度成为算法的一部分](#lesson-schedule)：因为先改哪个模块会改变后续标签，明确真实学习、模型更新和规划次序，并分别记账。

3. [维护表示与技能模型一致性](#lesson-drift)：因为表示或技能改变会使旧后果坐标过期，增加版本和校准诊断，以等预算模块消融检验协同。

结论与条件：教学表格组合可验证接口与时序；不将各模块的局部理论相加成任意共享深网的收敛或通用智能保证。

### 相关方法改变了什么

- 表格Dyna组合：接口可解析，适合验证真实学习和规划分工。

- STOMP式链路：从子任务到技能、后果模型与规划，需完整计入发现与维护。

- 共享深度模块：减少部分重复表示，却引入梯度冲突、表示漂移和版本一致性问题。


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

经验模型记录每个 (z,a) 后看到各 (r,z′) 的次数，形成经验条件分布。它不同于只记最后一次转移：后者会把真实随机性误当作不断变化的确定性结果。本例的状态、动作与奖励取值有限，因此表格槽位数固定。但 Python 整数计数与规划游标的位宽仍可随运行时间增长；槽位固定不等于无限运行时固定字节预算。若奖励连续，以每个不同奖励作为字典键还会增加槽位数。

严格的资源限制需要另行规定有限精度计数、固定容量的近期统计或参数化模型，并明确溢出与淘汰规则。这些选择会改变模型估计。本例保留累计计数，以便观察旧经验如何延缓对环境变化的适应；它不是严格固定字节预算的终生实现。

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

折扣 option model 已把 $γ^τ$ 包进终点权重，因此 planner 不能再乘一次 γ。τ 随机时也不能把 $E[γ^τV]$ 简化成 $γ^{Eτ}E[V]$。平均奖励 option model 则保留奖励、时长和未折扣转移，使用 R−gτ。

例如学习“到门口”为了形成可复用 option，终止奖励可用于驱动这个 subtask；但主任务可能是递送物品，门口本身没有外部奖励。Planner 必须知道此技能真实消耗多少步、沿路得到什么外部收益、到达哪里，才能判断其价值。STOMP 的研究价值在于把这个链路作为可连接的学习问题，而不是在图中简单把 option 画成一条箭头。

Successor features 提供另一种可复用接口：若外部奖励近似 $r=φ(s,a,s^{\prime})^Tw$，则策略的 successor feature 预测折扣特征累积，价值近似 $ψ_π^Tw$。它方便奖励改变时快速重算价值，但已知特征线性分解、固定策略及动力学变化都是重要边界；并非等价于任意世界模型。

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
        # Finite toy outcome slots, not an unlimited-lifetime byte bound:
        # Python counts/cursor grow in bit width; new reward values add keys.
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

<a id="research-architecture-knowledge-contracts"></a>

## 研究专题 A · 用查询规格连接预测、技能与规划

完整架构中最容易遗漏的是知识的条件。相同的“ψ”可能指 SF 累计、flow 条件特征或 Laplacian 规划坐标；相同的“零样本”可能指无新奖励训练、无任务策略训练或无部署环境数据。模块名称不足以保证彼此兼容。可将每项知识记录为一份可检查的查询规格：它在什么状态版本、行为、时域与目标下预测什么，谁会使用它。

$$
K_j=(q_j,\,\mathcal D_j,\,\nu^{\rm state}_j,\,\nu^{\rm behavior}_j,\,\widehat y_j,\,\epsilon_j,\,c_j),\qquad q_j=(\pi_j,C_j,\gamma_j,\text{readout}_j)
$$

这是拟议的架构元数据规格：查询、验证域、状态与行为版本、预测器、所用校准误差及成本。ε 只有在明示验证协议下才有含义，不默认是全环境的数学上界；readout 指下游要从该知识算什么。

| 模块知识 | 明确依赖 | 允许的下游查询 | 不能据此宣称 |
| --- | --- | --- | --- |
| SF / FB | 动力学、策略族、特征与奖励读出 | 被表示覆盖的奖励价值或任务条件行为 | 任意动力学变化立即迁移 |
| SF² | 条件特征、flow 投影、目标策略及生成器 | 未来占用采样，或给非线性 critic 的特征 | 任意 GVF 可线性读出 |
| 方向技能 / OKB | 基础策略、SF、元策略及调用规则 | 组合执行；额外模型可支持规划 | 技能组合自动得到准确后果模型 |
| DINO-WM / 2-AC | 冻结视觉编码、动作语义、离线动力学数据 | 给定目标下的动作条件特征预测与搜索 | 在线自动状态/技能构造已完成 |
| VE / 分布等价模型 | 策略/价值或统计摘要查询族 | 该族范围内的 Bellman 或风险计算 | 模型还原全部世界，或保证任意新风险目标 |

例如编码器升级后，旧 DINO 特征终点与新目标坐标不能直接比较；低层停止规则改变后，旧 option model 的 τ 与终点不再对应当前执行；主任务改成风险规避后，旧 mean-value 等价模型的规格不够。失配来自明确依赖，而不是看到回报下降后笼统称“遗忘”。

**算法：这是一项可实现的集成研究设计，不等于现成 OaK 实现**

1. 接口维护骨架（拟议）：
  1. 真实 transition 只推进环境一次，保存采样时行为与状态版本
  1. 分发给各自明确的 prediction/model/control learner
  1. 问题或技能更新后标记依赖知识过期，安排真实后续验证
  1. planner 只调用查询条件与版本匹配的知识；失配时退回已有合法接口
  1. 记录旧知识重新校准、局部迁移与作废的成本
  1. 固定总容量与每步更新预算，比较接口检查对失败/恢复的作用

OaK 与 Alberta Plan 提出经验产生预测、子任务、options、模型与规划的长期研究链。上述近年工作分别强化其中若干接口，但外部数据、固定表示或已给定奖励族仍可能参与启动。研究价值在于明确这些模块怎样互相提供学习信号，并检验未见变化下的真实收益，而非将多个论文模块接在一起就宣称完成完整架构。

<a id="research-architecture-future-utility-and-budget"></a>

## 研究专题 B · 从预训练能力到有限预算下的持续知识维护

预训练模型降低在线学习的起点成本，持续架构还需要决定学什么、何时补齐、何时淘汰。OKB 的行为基补齐、GVF 问题发现、generate-and-test 的特征维护针对不同对象，可以共享“未来是否有用”的评价思想，但不能把它们直接视为同一个算法。特别是低预测误差可能仅说明问题太容易，高误差也可能来自不可约噪声。

$$
U_j(W)=\underbrace{J_W(\text{with }K_j)-J_W(\text{without }K_j)}_{\text{同协议下的未来收益差}}-\lambda_C\Delta C_j-\lambda_M\Delta M_j,\qquad \sum_jM_j\leq M_{\max},\quad \sum_jC_{j,t}\leq C_{\max}
$$

这是资源受限知识效用的拟议检验量，不是新引用定理。$J_W$ 必须明确是未来真实收益、查询误差减少或其他指标，不能混用；with/without 需要配对随机性、冻结副本或独立控制实验，不能读取同一生命的反事实结果。

例子：一个新预测头使充电风险估计更准确，却增加每步延迟，以致机器人错过控制频率；另一个技能提高覆盖，但长期不被规划器查询。这两者可能有预测或探索价值，却未必有净控制效用。只有在同样真实时间、容量与更新预算下测到未来收益，才能判断应否保留。

| 加入候选后的检验 | 回答的问题 | 应保留的边界 |
| --- | --- | --- |
| 未来固定行为预测 | 新知识是否减少未见后果误差 | 不把 replay 拟合当作未来检验 |
| 冻结副本读出 | 现有表示是否让查询可低成本恢复 | probe 不写回正在评价的 agent |
| 影子规划 | 知识是否改变正确的动作排序 | 不把模型里的收益当作真实环境收益 |
| 配对控制 | 实际调用是否改善未来互动 | 计入探索、失败和恢复成本 |
| 预算压力与淘汰 | 有限容量是否仍能持续补齐 | 记录旧能力损失与未来再学习成本 |

从 DINO-WM 或 V-JEPA 2-AC 启动的系统，可先冻结 encoder 只持续更新后果预测，之后才加入状态、问题或技能生成；从 SF² 启动则可先测试新查询可读出程度，再决定是否需扩容。每次新增机制都应有同预算对照，避免把更多数据、更多参数或更多规划混进架构结论。

尚未闭合的研究问题包括：任务和查询不断新增时如何分配信用；知识版本改变时怎样保留兼容接口；新动力学需要探索时怎样支付机会成本；被淘汰知识在未来再次出现时怎样恢复。这些是全局持续学习的问题，与某个静态基准上最终分数提高不同，应以一条完整经验流的收益与资源轨迹验证。

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

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

状态是支持后续计算的内部信息；GVF 指定一个预测问题；RTRL 和资格迹规定信用如何传播。三者可以组合，但不是相互替代的算法名称。先理解给定策略的预测，再讨论预测怎样改善控制。

- [Towards model-free RL algorithms that scale well with unstructured data](https://yingwen.io/zh/continual-rl/research/#recent-nibbler-predictive-features)

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://yingwen.io/zh/continual-rl/research/#recent-vtrace-impala)

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

Laplacian 描述行为图结构，奖励感知表示加入路径价值，METRA 学习有区别的行为，HIQL 利用离线目标轨迹，MaestroMotif 引入语言先验。它们承担不同的设计工作；生成技能、选择技能与组合技能需要分别评价。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [MaestroMotif: Skill Design from Artificial Intelligence Feedback](https://yingwen.io/zh/continual-rl/research/#recent-maestromotif-semantic-skills)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

模型可提取性的理论说明某类能力需要什么知识，不指定唯一网络。Dreamer 研究潜在想象控制，STOMP 研究随机时长行为模型，DRAGO 研究旧模型知识保留。模型误差、查询策略和规划收益之间仍需实验连接。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [Mastering diverse control tasks through world models](https://yingwen.io/zh/continual-rl/research/#recent-dreamerv3-world-models)
- [General Agents Contain World Models](https://yingwen.io/zh/continual-rl/research/#recent-general-agents-world-models)
- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://yingwen.io/zh/continual-rl/research/#recent-tdmpc2-decision-time-model)
- [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://yingwen.io/zh/continual-rl/research/#recent-dino-wm-feature-planning)
- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://yingwen.io/zh/continual-rl/research/#recent-vjepa2-action-conditioned)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [Rethinking the Foundations for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-rethinking-crl-foundations)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [Loss of plasticity in deep continual learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-backpropagation)
- [Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-vla-simple-recipe)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Learning from experience instead of curated datasets](https://yingwen.io/zh/continual-rl/research/#recent-oak-network-idbd)
- [Discovering state-of-the-art reinforcement learning algorithms](https://yingwen.io/zh/continual-rl/research/#recent-disco-rl)
- [How Should We Meta-Learn Reinforcement Learning Algorithms?](https://yingwen.io/zh/continual-rl/research/#recent-meta-algorithm-search-comparison)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [Position: Lifetime tuning is incompatible with continual reinforcement learning](https://yingwen.io/zh/continual-rl/research/#recent-lifetime-tuning)
- [The Cell Must Go On: Agar.io for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-agarcl)
- [Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-vla-simple-recipe)
- [How Should We Meta-Learn Reinforcement Learning Algorithms?](https://yingwen.io/zh/continual-rl/research/#recent-meta-algorithm-search-comparison)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文提供定义和条件，架构讲座提出模块组织，算法论文检验特定机制。完整系统还要明确智能体、外部设计者和世界各自承担的工作；组件成立不自动意味着组合后的长期收益成立。

- [Rethinking the Foundations for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-rethinking-crl-foundations)
- [Plasticity as the Mirror of Empowerment](https://yingwen.io/zh/continual-rl/research/#recent-plasticity-mirror-empowerment)
- [General Agents Contain World Models](https://yingwen.io/zh/continual-rl/research/#recent-general-agents-world-models)
- [The OaK Architecture: A Vision of SuperIntelligence from Experience](https://yingwen.io/zh/continual-rl/research/#recent-oak-architecture)
- [Towards model-free RL algorithms that scale well with unstructured data](https://yingwen.io/zh/continual-rl/research/#recent-nibbler-predictive-features)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)

### Towards model-free RL algorithms that scale well with unstructured data

Joseph Modayil, Zaheer Abbas

arXiv 预印本 · 2023 · 支持方法与理论

#### 研究问题

大量原始观测中只有少数局部组合与奖励有关，智能体能否逐步构造有用的预测特征？

#### 关键机制

Nibbler 将预测问题的构造和预测结果的复用结合起来：选择局部输入、学习与奖励相关的通用价值预测，再将预测作为后续学习的特征。GVF 在这里不是一个新的优化器，而是描述“预测什么、在什么行为下预测”的问题接口。

#### 证据

作者在组合式合成环境中增加观测规模，报告了利用任务结构的样本效率。环境可以具有指数增长的状态组合，但学习器不必显式枚举全部状态。

#### 条件与限制

这不是对任意高维观测的线性样本复杂度保证。局部可分解结构、候选问题与特征构造规则仍是关键条件；从该实验族迁移到视觉控制需要额外验证。

#### 阅读与实验

把一条预测完整写成累积量、延续条件、目标策略和输入特征四项，再指出它如何进入主任务的价值函数。区分问题生成带来的收益与增加参数量带来的收益。

#### 原文与相关入口

- [作者预印本](https://arxiv.org/abs/2311.02215)：问题族、Nibbler 构造过程与扩展性实验。

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

### MaestroMotif: Skill Design from Artificial Intelligence Feedback

Martin Klissarov, Mikael Henaff, Roberta Raileanu, Shagun Sodhani, Pascal Vincent, Amy Zhang, Pierre-Luc Bacon, Doina Precup, Marlos C. Machado, Pierluca D’Oro

ICLR 2025 · 2025 · 支持方法与理论

#### 研究问题

语言描述如何变成可训练的技能奖励，并进一步组织成一个层次策略？

#### 关键机制

设计者先给出技能描述。语言模型的偏好反馈被用于训练奖励模型，再用生成的代码规定技能启动、终止和组合方式；强化学习负责学习实际执行行为。这把语义先验、奖励学习和时间抽象串成了具体训练流程。

#### 证据

论文在 NetHack 学习环境中检验复杂技能与任务组合。作者仓库同时包含偏好、代码生成和 RL 训练模块，可以追踪自然语言到环境动作的完整依赖。

#### 条件与限制

语义知识、技能描述和语言模型来自外部设计过程。该证据并不说明智能体仅凭自身交互就能产生同样的技能体系；偏好模型也可能与真实目标不一致。

#### 阅读与实验

选择一项技能，分别列出描述、偏好标签、训练奖励、终止条件和下游用途。移除语义描述或改变奖励模型时，要单独计量额外查询与人工成本。

#### 原文与相关入口

- [ICLR 2025 原文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2dc5a0faac8102fd47363795f71126ee-Abstract-Conference.html)：技能设计、奖励学习与组合实验。
- [作者实现](https://github.com/mklissa/maestromotif)：偏好学习、代码生成和执行策略的不同模块。

#### 作者代码

[原论文作者仓库。](https://github.com/mklissa/maestromotif)

MaestroMotif 的偏好处理、技能组织与 RL 实验。

### Loss of plasticity in deep continual learning

Shibhansh Dohare, J. Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A. Rupam Mahmood, Richard S. Sutton

Nature · 2024 · 直接研究持续学习

#### 研究问题

一个长期训练的网络如何保留继续形成新特征的能力？

#### 关键机制

Continual Backpropagation 在梯度学习之外持续生成并测试特征。它估计单元的效用和成熟度，少量替换低效用的成熟单元，并协调新单元的输入、输出和相关状态。维护新的可学习方向是一个持续过程，而不是等到任务切换后整体重启。

#### 证据

论文在长序列监督学习与强化学习问题中展示可塑性损失，并检验特征替换的作用。作者仓库包含 generate-and-test 与优化器状态处理。

#### 条件与限制

有限序列上的学习保持不保证无限生命中的任意适应。替换率、效用定义与成熟度条件仍需选择；新任务学习速度和旧能力保留必须分开测量。

#### 阅读与实验

逐项消融“成熟度筛选”“效用筛选”“随机替换”。比较相同替换预算，检验收益究竟来自定向回收还是一般参数扰动。

#### 原文与相关入口

- [Nature 原文](https://doi.org/10.1038/s41586-024-07711-7)：长期可塑性实验与 continual backpropagation。
- [作者代码](https://github.com/shibhansh/loss-of-plasticity)：关注 lop/algos/gnt.py 及替换时的优化器状态。

#### 作者代码

[论文作者公开的实验实现。](https://github.com/shibhansh/loss-of-plasticity)

论文任务、持续反向传播与 generate-and-test。

### Discovering state-of-the-art reinforcement learning algorithms

Junhyuk Oh, Gregory Farquhar, Iurii Kemaev, Dan A. Calian, Matteo Hessel, Luisa Zintgraf, Satinder Singh, Hado van Hasselt, David Silver

Nature · 2025 · 支持方法与理论

#### 研究问题

除了学习策略，能否从大量学习过程里学出更有效的 RL 更新规则？

#### 关键机制

DiscoRL 用外层优化评价执行若干内层更新后的行为表现，学习价值、策略与辅助预测之间的更新方式。被训练的对象是学习算法本身，而不仅是某个任务的策略参数。内外两层有各自的数据、时间尺度与计算预算。

#### 证据

论文报告跨环境发现更新规则与迁移到未见环境的结果，并公开配套算法实现。它展示了自动算法发现的可能性，但依赖大规模外层训练。

#### 条件与限制

外层在大量环境和设备上的搜索属于设计者侧资源，不能记作测试智能体单次生命内的自主学习。公开规则的执行成本与发现该规则的成本应分别报告。

#### 阅读与实验

画出内层参数和外层参数的更新依赖，再列出测试时哪些量被冻结。与在线 IDBD 比较时，先区分跨任务算法发现和单流步长追踪。

#### 原文与相关入口

- [Nature 原文](https://www.nature.com/articles/s41586-025-09761-x)：算法发现过程、外层资源与泛化实验。
- [作者实现](https://github.com/google-deepmind/disco_rl)：配套代码与发现的更新规则。

#### 作者代码

[Google DeepMind 的论文配套仓库。](https://github.com/google-deepmind/disco_rl)

DiscoRL 配套实现与学习到的更新规则；具体训练资源以仓库说明为准。

### Learning from experience instead of curated datasets

Oak Lab

Oak Lab 技术博文 · 2026 · 支持方法与理论

#### 研究问题

有用信号稀疏且大量输入是噪声时，在线学习规则如何分配不同方向的更新能力？

#### 关键机制

博文从含稀有有效特征的线性预测问题出发，对比统一步长与 IDBD 的逐权重适应，再展示 NetworkIDBD 在非线性带噪观测中的例子。核心主张是让长期学习效果影响信用和步长分配，而不只依据当前梯度幅度归一化。

#### 证据

公开页面提供受控噪声特征任务和 NoisyMNIST 示例。它们是机制演示，便于理解有效信号密度与输入规模的关系。

#### 条件与限制

该页面不是完整 CRL 控制论文，也未给出可直接复现所有图表的完整代码和算法推导。监督噪声任务的结果不能证明一般 SGD 或所有深度 RL 都无法从经验学习。

#### 阅读与实验

先复现线性噪声特征问题，分开改变有效特征稀疏度与噪声维数。进入控制前，再加入策略改变数据分布这一因素。

#### 原文与相关入口

- [Oak Lab 原始博文](https://oaklab.ai/posts/learning-from-experience-instead-of-curated-datasets)：2026 年 7 月 13 日；受控实验、NetworkIDBD 示例与研究动机。

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

### Plasticity as the Mirror of Empowerment

David Abel, Michael Bowling, Andre Barreto, Will Dabney, Shi Dong, Steven Hansen, Anna Harutyunyan, Khimya Khetarpal, Clare Lyle, Razvan Pascanu, Georgios Piliouras, Doina Precup, Jonathan Richens, Mark Rowland, Tom Schaul, Satinder P. Singh

NeurIPS 2025 · 2025 · 定义与架构观点

#### 研究问题

环境改变智能体的能力，与智能体改变环境的能力，能否放在统一的信息论框架中？

#### 关键机制

论文用广义有向信息描述两个方向：环境对智能体的影响对应一种可塑性，智能体对环境的影响对应赋能。统一表达使二者的关系和权衡可以被形式化，而不仅用神经元休眠或短期奖励间接描述。

#### 证据

贡献主要是概念定义和理论关系，提供研究长期交互的新坐标。它没有把信息量指标直接等同于某个具体神经网络算法的长期回报。

#### 条件与限制

信息论可塑性与“新目标拟合速度”不是相同估计量，也不等于参数变化越大越好。有限数据下怎样稳健估计这些信息量，需要额外方法。

#### 阅读与实验

分别举出高环境影响但低奖励、高赋能但不学习的过程。说明为什么两类能力与任务成功都需要独立评价。

#### 原文与相关入口

- [NeurIPS 2025 原文](https://papers.nips.cc/paper_files/paper/2025/hash/f04957cc30544d62386f402e1da0b001-Abstract-Conference.html)：统一定义、理论关系与解释。
- [作者预印本](https://arxiv.org/abs/2505.10361)：便于检索定义和证明。

### Rethinking the Foundations for Continual Reinforcement Learning

Esraa Elelimy, David Szepesvari, Martha White, Michael Bowling

RLC 2025 / RLJ · 2025 · 定义与架构观点

#### 研究问题

如果智能体终生交互且世界不断变化，传统形式化和评价对象遗漏了什么？

#### 关键机制

论文重新审视状态、时间与累计奖励评价中的隐含假设，并讨论以交互历史和偏离遗憾等对象描述持续学习。研究重点从“在固定任务上最终收敛到什么”转向“在持续过程里，什么样的行为比较才有意义”。

#### 证据

这是形式化与研究基础的论证，提出可继续研究的定义和问题；不是一个已经完成全部工程验证的通用智能体。

#### 条件与限制

对常见形式化局限的讨论不意味着 MDP、折扣回报或平均奖励在各自条件下无效。评价框架还需要与具体可计算算法和实验协议连接。

#### 阅读与实验

选择一个有不可逆代价的环境，分别写出最终任务分数、终生在线收益和比较策略集合。检验它们是否会给同一行为排出不同顺序。

#### 原文与相关入口

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_243.pdf)：形式化动机、定义与论证。
- [RLJ 论文入口](https://rlj.cs.umass.edu/2025/papers/Paper243.html)：作者与正式收录信息。

### Position: Lifetime tuning is incompatible with continual reinforcement learning

Golnaz Mesbahi, Parham Mohammad Panahi, Olya Mastikhina, Steven Tang, Martha White, Adam White

ICML 2025 Position Paper · 2025 · 评价与实验协议

#### 研究问题

如果设计者用完整未来生命反复调参，实验还在测智能体面对未知变化的能力吗？

#### 关键机制

论文限制调参可访问的生命阶段，并比较这种选择方式与利用完整生命回报挑选配置的差异。外部设计者掌握未来变化信息，可能使一个并不自适应的固定算法显得适应良好。核心改变发生在评价协议，而不是 TD 更新公式。

#### 证据

作者用持续、非平稳设置中的深度 RL 实验说明超参数选择可以改变方法比较。该工作属于立场论文，论证与示例用于推动更符合问题目标的评价。

#### 条件与限制

允许多少开发阶段经验需要按应用规定，不存在由该论文推出的普适固定比例。仅限制时间前缀也不能替代独立测试种子和计算预算控制。

#### 阅读与实验

对同一配置集合分别按开发前缀和完整生命选择超参数，再在独立测试生命上比较。报告两种选择使用了哪些未来信息。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/mesbahi25a.html)：调参协议、论证与示例实验。

### The Cell Must Go On: Agar.io for Continual Reinforcement Learning

Mohamed A. Mohamed, Kateryna Nekhomiazh, Vedant Vyas, Marcos M. José, Andrew Patterson, Marlos C. Machado

arXiv 预印本 · 2025 · 评价与实验协议

#### 研究问题

如何在持续、动态的高维交互里，同时研究记忆、探索、信用分配与学习能力保持？

#### 关键机制

AgarCL 提供持续运行的游戏环境，并用分解的小任务暴露不同困难。完整环境把这些机制放回同一交互循环，小任务则便于定位失败原因。游戏中的复活事件与把整个世界和智能体都重新开始不是同一种重置。

#### 证据

论文提供环境、基线与可塑性方法比较；部分常见修复在其测试中改善有限。这说明保持可塑性并不能单独代替记忆、探索和长期信用分配。

#### 条件与限制

一个游戏不能代表全部真实持续问题。小任务与完整游戏的协议需要分别阅读；此处仅按可确认的预印本状态收录，不把投稿信息写成会议录用。

#### 阅读与实验

先在一个小任务中验证机制，再检验它在完整环境中的作用是否仍存在。将世界重置、角色复活、参数重置和数据清空分开记录。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2505.18347)：环境设计、分解任务与基线结果。
- [作者环境仓库](https://github.com/machado-research/AgarCL)：环境安装、接口和运行示例；算法基线与环境本体分开。

#### 作者代码

[Machado 研究团队的环境实现。](https://github.com/machado-research/AgarCL)

AgarCL 环境与示例，非所有算法结果的单一训练脚本。

### Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning

Jiaheng Hu, Jay Shim, Chen Tang, Yoonchang Sung, Bo Liu, Peter Stone, Roberto Martín-Martín

RLC 2026 · 2026 · 直接研究持续学习

#### 研究问题

大规模预训练的视觉—语言—动作模型，是否仍需要复杂机制才能顺序学习控制任务？

#### 关键机制

论文研究对预训练 VLA 进行顺序强化学习，并以低秩适配等相对简单的训练流程检验持续学习。预训练表示、可更新参数子空间和 RL 目标共同决定迁移与遗忘，不能只把结果归因于单一保留正则项。

#### 证据

作者在多种 VLA 与长期任务基准上比较，并提供实验代码。RLJ 的 RLC 2026 论文页与论文脚注分别给出正式入口和作者仓库。

#### 条件与限制

预训练数据和算力属于外部资源，任务与重置协议也影响难度。该结果不意味着从零训练的网络不会遗忘，更不意味着任意无边界任务流只需微调。

#### 阅读与实验

固定预训练模型，分别改变可训练参数量和任务顺序。报告预训练资源、每任务在线数据、回放或重置条件，再与传统 CRL 方法比较。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2603.11653)：VLA 持续学习设置、机制与比较。
- [RLC 2026 / RLJ 论文页](https://rlj.cs.umass.edu/2026/papers/Paper84.html)：会议原文入口；PDF 脚注链接作者代码。
- [作者实现](https://github.com/UT-Austin-RobIn/continual-vla-rl)：持续 VLA 的训练与评价代码。

#### 作者代码

[UT Austin RobIn 实验室的原论文仓库。](https://github.com/UT-Austin-RobIn/continual-vla-rl)

预训练 VLA 的顺序 RL 训练和论文评价。

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

### How Should We Meta-Learn Reinforcement Learning Algorithms?

Alexander David Goldie, Zilin Wang, Jaron Cohen, Jakob Foerster, Shimon Whiteson

RLC 2025 / RLJ · 2025 · 评价与实验协议

#### 研究问题

算法表示、发现算法的方法和测试智能体的学习成本，应该如何独立比较？

#### 关键机制

对 RL 流程的不同组件进行算法发现，比较黑盒学习、神经/符号蒸馏与 LLM 代码提案。学习器的表示形式和搜索过程分开定义，才能识别泛化、可解释性和成本之间的取舍。

#### 证据

论文直接比较元训练、元测试、样本成本、训练时间与可解释性，作者代码按发现方法和评价入口组织。

#### 条件与限制

跨环境发现规则主要发生在设计者侧，不等于运行智能体已能终生修改规则。论文的训练任务和预算范围不支持所有算法发现方法的普适排序。

#### 阅读与实验

固定被学习组件、输入权限、元训练数据和发现预算；封存规则后再检验未见环境、长生命与未通知漂移。

#### 原文与相关入口

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_218.pdf)：比较对象、元训练/测试与多维成本。
- [RLJ 论文记录](https://rlj.cs.umass.edu/2025/papers/Paper218.html)：作者与正式会议收录。

#### 作者代码

[论文提供、仓库标为官方的作者实现。](https://github.com/AlexGoldie/learn-rl-algorithms)

learning_algorithms 中各发现方法与独立 evaluation 流程。

### IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures

Lasse Espeholt, Hubert Soyer, Rémi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, Koray Kavukcuoglu

ICML 2018 · 2018 · 支持方法与理论

#### 研究问题

actor采样策略落后于learner时，如何校正状态价值与策略更新？

#### 关键机制

V-trace用截断ρ校正当前TD误差，用独立截断c控制后续误差传播，再用下一状态V-trace目标构造actor优势。ρ上限还决定表格固定点对应的截断策略。

#### 证据

原文分析固定点并检验分布式多任务训练。固定版本作者代码明确区分clipped_rhos、cs、反向scan与pg_advantages。

#### 条件与限制

IMPALA保存短轨迹并批量训练，不属于严格单样本流式协议。截断后价值可能对应不同于原目标的策略；信用章bandit示例显示0.8变为0.5。

#### 阅读与实验

独立改变策略滞后、ρ上限和c上限。记录目标策略变化与传播长度，不把两种截断都只解释为方差控制。

#### 原文与相关入口

- [ICML原文](https://proceedings.mlr.press/v80/espeholt18a.html)：V-trace固定点与分布式实验。
- [作者固定实现](https://github.com/google-deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/vtrace.py)：from_importance_weights与下一状态actor目标。

#### 作者代码

[原作者团队仓库的固定版本。](https://github.com/google-deepmind/scalable_agent/tree/6c0c8a701990fab9053fb338ede9c915c18fa2b1)

IMPALA原始TensorFlow实现与V-trace；运行需要原项目环境。

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

六状态、双动作、手工记忆的集成教学骨架；不是完整 OaK/STOMP 性能复现。

[下载 lifelong_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)

```sh
python lifelong_algorithms_lab.py architectures
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton · Dyna, an integrated architecture for learning, planning, and reacting](https://doi.org/10.1145/122344.122377)：模型学习、直接学习与模拟规划的经典接口；本页骨架受此启发，并明确另外加入平均奖励与 GVF。

- [Sutton, Bowling & Pilarski · The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173)：状态、预测、时间抽象与规划的研究纲领，不是完成全闭环的报告。

- [Sutton et al. · Reward-Respecting Subtasks / STOMP](https://arxiv.org/abs/2202.03466)：Subtasks、Options、Models、Planning 的学习链路及 reward-respecting 构造。

- [Barreto et al. · Successor Features for Transfer in Reinforcement Learning](https://arxiv.org/abs/1606.05312)：有限奖励特征、固定策略 SF 与 GPI 的经典机制桥梁。

- [Oak Lab · The OaK Architecture](https://oaklab.ai/posts/the-oak-architecture)：Richard Sutton 的架构研究纲领与讲座入口。

- [作者代码 · average-reward-methods](https://github.com/abhisheknaik96/average-reward-methods)：control_agents.py 中的直接学习与 planning_update，适合比较奖励率更新调度。

- [Elelimy et al. · Rethinking the Foundations for Continual RL](https://rlj.cs.umass.edu/2025/papers/Paper243.html)：RLC 2025：history process、deviation regret 与持续学习评价对象。

- [Mohamed et al. · The Cell Must Go On: Agar.io for Continual RL](https://arxiv.org/abs/2505.18347)：2025 年提出、后续修订的 AgarCL：完整环境与分解小游戏，讨论探索、记忆、信用分配和可塑性。

- [AgarCL 作者环境](https://github.com/machado-research/AgarCL)：C++ 仿真与 Python 接口；局部重生、观测和混合动作需按环境配置理解。

- [AgarCL 作者基线 · PPO 混合动作实现](https://github.com/machado-research/AgarCL-benchmark/blob/main/PPO_multi_heads_full_action.py)：同仓库包含 DQN_full_action_set.py、SAC_full_action_set.py 及 recurrent 版本；完整游戏实验需匹配动作、参数搜索与运行预算。

- [NeurIPS 2025 原文与补充材料入口](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0ab48777def88e73b50746a6011be0b0-Abstract-Conference.html)：算法 1–3、附录 A.3 的两个最优子程序假设及代码声明；未在本教材运行补充代码。

- [Option Keyboard 的经典桥梁](https://proceedings.neurips.cc/paper/2019/file/251c5ffd6b62cc21c446c963c76cf214-Paper.pdf)：cumulant 组合、GPE/GPI 与技能接口。

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html)：短期预测、终点价值、多任务协议。

- [作者实现](https://github.com/nicklashansen/tdmpc2)：训练、模型与 plan 函数分别阅读。

- [ICML 2025 原文](https://proceedings.mlr.press/v267/zhou25t.html)：正式发表入口；早期 ICLR 投稿页不能替代此状态。

- [作者项目代码](https://github.com/gaoyuezhou/dino_wm)：train.py、plan.py、数据与已公开模型检查点范围。

- [2025 首稿](https://arxiv.org/abs/2506.09985v1)：action-free 预训练、2-AC 后训练、真实规划与第 4.3 节限制；此处不赋予未核实会议状态。

- [Meta FAIR 官方实现](https://github.com/facebookresearch/vjepa2)：包含 V-JEPA 2、2-AC 和较新的 2.1；版本不能混用。

- [Touati & Ollivier · Learning One Representation to Optimize All Rewards](https://arxiv.org/abs/2103.07945)：FB 表示的理论出发点；探索/经验覆盖、近似误差及奖励查询约定。
