# 模型学习与规划：MPC、短模型 rollout 和潜在想象

现代深度强化学习 · 并列研究分支

模型在哪里进入决策，预测误差又怎样变成控制误差？

## 本章内容

- 由监督建模到有限时域 MPC。
- 推导多步误差传播与短 rollout 的动机。
- 区分 MPC、MBPO 和 Dreamer 的模型使用位置。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [学习与规划：Dyna、优先扫描和执行时搜索](https://yingwen.io/zh/continual-rl/foundations/tabular/planning/)：理解模型查询、备份与真实交互的区别。
- [深度价值学习：DQN、Double DQN 与目标的时间顺序](https://yingwen.io/zh/continual-rl/foundations/deep/deep-value/)：掌握可学习表示、价值尾项与目标版本。


### 条件概率与期望

区分随机变量、一次样本与条件分布；可以使用 Bayes 法则和全期望公式。

### MDP 与价值

理解状态、动作、转移、奖励、策略和 Bellman 递推；学习数据可以来自不同策略。

### 优化与梯度

知道固定目标、参数梯度、约束和期望目标的含义；神经网络只是函数表示的一种选择。

<a id="problem-definition"></a>

## 本章的问题定义

从真实转移学习后果模型，再决定模型计算在训练或行动的哪个位置发挥作用。

### 给定条件与符号

- 状态 $s\in\mathcal S$、动作 $a\in\mathcal A$；转移与奖励核 $p(s^{\prime},r\mid s,a)$。
- 初始分布 $d_0$、有界奖励 $R_{t+1}$、折扣 $0\le\gamma<1$；$J(\pi)=\mathbb E_{d_0,\pi,p}[\sum_{t\ge0}\gamma^tR_{t+1}]$。
- 模型 $\hat p_\phi$、规划时域 $H$、模型计算预算 $B$、终点价值 $\hat V$。

### 需要求解的对象

用模型辅助真实控制；模型预测准确与控制表现分别衡量。

### 信息与数据权限

模型只在训练数据覆盖范围内获得直接约束；策略可能主动进入模型错误区域。

$$
\max_{a_{0:H-1}}\mathbb E_{\hat p_\phi}\!\left[\sum_{k=0}^{H-1}\gamma^kR_{k+1}+\gamma^H\hat V(S_H)\right]
$$

这是 MPC 的模型内开环序列目标，执行第一个动作后重规划。MBPO 与 Dreamer 使用模型的位置不同，不都求这个动作序列问题。

### 成立条件与解的含义

- 有限模型预算；模型奖励、终止及动作语义与真实系统一致。
- 模型误差界需说明在哪个状态动作分布上成立。

判断准则：分别控制模型质量、模型使用长度和额外计算，判断收益来自哪一部分。

### 适用边界

- 用一步预测误差低证明长程规划可靠。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)：模型学习是规划的输入问题；下游查询决定需要保留哪些后果。

- 组合不同学习问题 · [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)：模型存在之后仍需选择备份位置、深度和预算。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

规划越积极，越可能利用模型误差；长虚拟轨迹会累积分布偏移。

### 本章的核心思路

分别设计模型、使用方式与误差控制，不将所有模型方法视为同一个 rollout 算法。

1. [先确定模型学习对象](#lesson-notation)：预测状态、奖励和终止的目标必须与后续查询一致。

2. [用有限时域并反复重规划](#lesson-derive)：MPC 用真实新观测纠正计划，但仍依赖模型和尾值。

3. [限制模型展开的误差](#lesson-error)：短 rollout 用真实状态作起点；长度是误差与计算的折衷。

4. [把模型用于策略训练](#lesson-latent)：潜在想象通过学习策略传递模型收益，不等同于每次行动显式搜索。

结论与条件：精确模型的规划性质不覆盖有偏神经模型；控制错误需结合实际访问分布分析。

### 相关方法改变了什么

- MPC / Dyna：分别直接选择当前动作与更新可复用价值。

- MBPO / Dreamer：模型生成的数据用途、表示空间和策略优化路径不同。


<a id="lesson-setting"></a>

## 1 · 学会预测，不等于已经会规划

无模型方法把经验直接用于价值或策略；有模型方法还学习“如果采取这个动作，接下来可能发生什么”。一个可复用的模型可以在实际行动前比较多个选择，也可以为价值学习生成额外经验。这样节省的是与真实世界交互的机会，代价是模型计算，以及把预测误差带入行动选择的风险。

模型可以预测原状态、潜在状态、奖励、终止概率或某种技能的结果。Model-based 方法用这些预测来选择行动或训练价值与策略。仅增加一个辅助预测头，却不让它参与行动相关计算，不能据此判断已经获得模型规划的优势。神经网络、线性系统与表格模型都能承担这一角色。

$$
\hat P_\psi(s'\mid s,a),\quad \hat r_\psi(s,a),\quad \hat c_\psi(s,a)
$$

分别预测转移、奖励和继续概率。任务定义决定真正终止，日志截断不能自动作为零继续概率。

外部设计者选择观察表示、模型损失、规划深度、终端价值和计算预算。运行时 agent 可以从真实经验更新模型与策略。若使用设计者提供的精确模拟器，本页称它为规划 oracle，不将模型知识计作学习得到。

<a id="lesson-notation"></a>

## 2 · 从单步数据学习动力学

$$
L_{\rm model}(\psi)=-\mathbb E_{\mathcal D}\log\hat P_\psi(S'\mid S,A)
$$

最大似然训练概率模型。若使用固定方差 Gaussian，负 log-likelihood 的参数相关部分等价于均方误差。

$$
s_{t+1}=s_t+ba_t+\epsilon_t,\qquad\hat b=\frac{\sum_ta_t(s_{t+1}-s_t)}{\sum_ta_t^2}
$$

对平方误差求导、令其为零，分母非零时得到此解。把它用于识别真实 b，还需无未建模漂移及噪声在给定当前状态和动作后均值为零等条件；只有无条件零均值并不充分。没有动作激励时不能识别增益。

低训练误差只约束数据分布附近。优化器可能选择训练数据从未支持过的动作序列，专门利用模型的系统误差。留出数据要检查多步预测和规划实际访问的区域；多个相似网络的分歧小，也不能证明模型正确。

<a id="lesson-derive"></a>

## 3 · MPC 的有限时域目标与重新规划

$$
\hat s_{k+1}=\hat f_\psi(\hat s_k,a_k),\quad\hat s_0=s_t
$$

每轮规划以当前真实观察或估计状态为起点，内部向前生成候选轨迹。

$$
a_{0:H-1}^*=\arg\max_{a_{0:H-1}}\left[\sum_{k=0}^{H-1}\gamma^k\hat r_\psi(\hat s_k,a_k)+\gamma^H\hat V(\hat s_H)\right]
$$

有限时域优化中的终端价值近似更远的后果。它不是免费得到的真值，误差和训练来源都需要说明。

上式采用确定性预测、H 步内不终止的简化任务。随机模型需要对候选回报取模型期望，不能一般地把下一状态均值代入非线性价值函数来替代它。允许真终止时，还需停止未来奖励与终端价值；模型的继续概率必须实际进入规划目标。

$$
w_0=1,\quad w_{k+1}=w_k\hat c_k,\qquad \hat G=\sum_{k=0}^{H-1}\gamma^kw_k\hat r_k+\gamma^Hw_H\hat V(\hat s_H)
$$

沿一条声明的模型路径，$\hat c_k$ 表示完成第 k 个转移后是否继续的条件概率；奖励 $\hat r_k$ 属于该次转移。$w_k$ 是到达该步的存活权重，故致终止转移本身的奖励仍保留。随机后继、奖励与终止相关时要用联合模型取期望，不能把相关量独立相乘。

MPC 只执行最优序列的第一个动作。收到新观察后重新规划，而不是盲目执行整个序列。这样可纠正部分开放环误差，但不能自动保证稳定或安全。连续动作常用 CEM、梯度或采样优化；本核用有限候选枚举，便于精确核对。

$$
G(\mathbf a^*)-G(\hat{\mathbf a}^*)\le2\epsilon_G,\quad \mathbf a^*\in\arg\max_{\mathcal A_H}G,\quad\hat{\mathbf a}^*\in\arg\max_{\mathcal A_H}\hat G
$$

若同一候选集合中每条序列均满足 |G−Ĝ|≤εG，插入 Ĝ(a*) 与 Ĝ(â*)；中间差不大于零，两端误差各不超过 εG。模型小误差可以改变排序，但统一误差界限制了这次规划的损失。若数值求解还差 εopt，则再加 εopt；仅有数据上的平均误差不足以应用此界，重新规划后的闭环性能也不是这里的序列目标。

如果规划本身每步消耗大量模型调用，真实交互样本少不等于总计算便宜。需要分别报告环境步、模型步、候选数、规划深度和墙钟时间。

<a id="belief-planning-contingent"></a>

### 3.1 · 从同一个 belief 出发，优化序列还是条件计划

[部分观测章的仓库任务](/zh/continual-rl/foundations/deep/partial-observability/#belief-planning-task)留下一个规划问题。包裹位置在左、中央、右；路口观察相同，当前 belief 为 $(1/2,0,1/2)$。最多两个动作，$\gamma=1$；扫描扣 $0.4$、保持位置，标签正确概率 $0.8$、两种错误各 $0.1$。扫描后只能取件，成功 $4$、失败 $-2$，然后终止。模型 rollout 应传播这些可能后果，而非把位置均值零当成中央包裹。

$$
\hat Q_{\rm fixed}(b,j)=-c+\sum_{s,o}b(s)\hat O(o\mid s)r(s,j)=-c+\sum_s b(s)r(s,j)
$$

确定序列「scan，然后固定取 j」没有让后续动作依标签变化。利用每行观测概率和为一，扫描项可求和消去；最好固定取左或右，净值 $0.6$，低于直接取件的 $1$。

$$
\hat Q_{\rm contingent}(b)=-c+\sum_o\hat p_o\max_j\sum_s\hat b_o(s)r(s,j)=\max_{\pi:\{L,C,R\}\to\{L,C,R\}}\left[-c+\sum_{s,o}b(s)\hat O(o\mid s)r(s,\pi(o))\right]
$$

条件计划为每种未来标签指定一个动作。这里只有 $3^3=27$ 种，模型可以精确枚举；把选择放到收到标签之后，得到取左、左、右的净值 $2.7$。没有读取真实隐藏位置。

![在模型内比较扫描的三条观察分支，再按实际标签执行一条取件分支](https://yingwen.io/crl-figures/belief-planning-walkthrough-plan.svg)

本图复用同一取件任务：扫描前只知道 belief，模型树按 0.45、0.10、0.45 分支；部署收到标签后更新后验再选动作。虚线树是模型计算，不是实际采集了三个标签。确定序列净值 0.6、条件计划净值 2.7，由独立 Python 枚举与 JavaScript belief backup 核对。

每次只执行首动作、收到观察后重新规划，可以让已发生的反馈影响下一动作。然而，若当前优化仍只评估固定动作序列，未来扫描信息没有进入现在的候选价值，它在本例就会选择立即取件，根本不会获得重规划机会。要在扫描前计入这项信息价值，需要条件计划，或能够表示扫描后决策价值的尾值；单写“会重规划”尚未改变这个目标。

PETS [§3、§5](https://arxiv.org/html/1805.12114)用概率动力学评估候选动作序列，执行首动作后再次优化；它的粒子传播处理预测分布，却不能仅因用了概率模型就认定序列目标已经枚举未来观察分支。此处的条件计划形式来自 [Algorithms for Decision Making §20.2、算法20.2](https://algorithmsbook.com/)，两种规划对象要分别说明。

<a id="lesson-error"></a>

## 4 · 为什么短模型 rollout 有时更可靠

$$
e_{k+1}\le L_f e_k+\epsilon_f,\quad e_0=0,\qquad e_H\le\epsilon_f\sum_{j=0}^{H-1}L_f^j
$$

在相同动作序列、统一单步误差界和 Lipschitz 条件下，插入真实动力学在预测状态处的值即可得到误差递推。

$$
|\hat G-G|\le\sum_{k=0}^{H-1}\gamma^k(\epsilon_r+L_r e_k)+\gamma^H L_Ve_H
$$

此式还假定奖励误差界、奖励和终端价值的 Lipschitz 条件，并使用相同终端价值函数；它不是任意像素模型的通用保证。

MBPO 从真实 replay 状态出发生成较短的模型轨迹，将这些合成样本用于 off-policy 策略学习。短分支缩短连续承受模型误差的距离，同时保留数据增广。起点覆盖不足、模型偏差和奖励模型错误仍然存在，因此 rollout 长度是一项需要验证的选择，不是越短越优的定理。

<a id="experiment-learned_model_mpc"></a>

### 实验：只执行计划的第一步：短视与有限时域规划

奖励模型已经学会每一步后果，为什么一步贪心仍可能无法完成任务？增加搜索深度会解决哪些问题，又留下哪些问题？

**环境与可用信息。** 六格确定性链，从0开始，左右动作在左边界截断。到5真正终止并回到0；其余期望奖励−0.01。终点奖励均值前600步为1，之后为0.5，所有奖励观测另加标准差0.02的高斯噪声。状态完全可观测。

**设置。** 5个种子，各1200个真实转移。经验模型从空开始，只从已发生转移更新奖励均值与转移频数；未访问动作的乐观值为0.05。γ=0.95、ε=0.1。候选每步递归规划5层，对照1层；都只执行首动作后重新规划。

**检验的机制。** 一步规划无法表达先付出若干步小代价再获得终点奖励。多层搜索组合模型后果，但不会自动消除奖励噪声、旧均值或错误模型。模型预测和真实执行是不同事件。

**测量。** 图为冻结当前规划策略在真实均值奖励下的解析折扣回报，不含评估采样噪声。同时看模型奖励RMSE；模型误差下降不保证动作排序立刻正确。

```bash
python3 implementations/continual/learned_model_mpc.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/learned_model_mpc/curves.svg)

横轴：environment_steps。纵轴：原始平均奖励冻结策略折扣回报。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 奖励降低后的最终检查点，五步规划回报为0.37015，一步版本为−0.2；后者对应持续支付−0.01而不抵达终点的循环。五种子这里得到同样冻结策略，并不意味着其训练轨迹相同。

**结论边界。** 有限小树可完整递归，计算开销未与一步版本匹配。它不是TD-MPC2或Dreamer的潜在模型工程，也没有模型不确定性或安全约束。

**继续实验。** 逐个增加搜索深度，找出首次能将终点收益传回起点的深度。然后冻结在变化前的奖励模型，重复搜索；解释为什么更多搜索无法自行发现第601步的新奖励。

[源码](https://yingwen.io/crl-code/implementations/continual/learned_model_mpc.py) · [逐种子记录](https://yingwen.io/crl-code/results/learned_model_mpc/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/learned_model_mpc/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/learned_model_mpc/curves.json)

<a id="belief-planning-model-error"></a>

### 4.1 · 一个观测模型误差，怎样变成错过扫描的损失

仍用同一个仓库，保持包裹转移与奖励完全不变，只改变规划器相信的扫描正确率 $\hat q$；另两个标签各占 $(1-\hat q)/2$。真实扫描器始终为 $q=0.8$。当 $\hat q\ge1/3$ 时，左、中央、右标签仍分别采用取左、左、右，扫描计划的值可以直接从联合后果概率算出。

$$
\begin{aligned}\hat p_{\rm correct}&=\hat q+\tfrac12\tfrac{1-\hat q}{2}=\tfrac14+\tfrac34\hat q,\\\hat Q_{\rm scan}&=-0.4+4\hat p_{\rm correct}-2(1-\hat p_{\rm correct})=-0.9+4.5\hat q.\end{aligned}
$$

正确标签总以概率 $\hat q$ 发生；中央误标签不区分左右，按左侧平局规则还能在左侧世界取对，贡献剩余项。此直线只用于已声明的 $\hat q\ge1/3$ 区间；教程在其他区间重新求分支动作。

$$
\hat Q_{\rm scan}>Q_{\rm direct}=1\ \Longleftrightarrow\ \hat q>\frac{19}{45}
$$

旧模型 $\hat q=0.4$ 计算扫描值 $0.9$，所以立即取左；在真实世界得到期望 $1$。正确模型 $q=0.8$ 会扫描并取得期望 $2.7$，错过扫描的机会损失为 $1.7$。

模型拟合可以从一个有限记录集开始。教程给定每个隐藏位置十条带位置标签的构造校准记录：八条正确、另两种误标签各一条。对称传感器族的 log-likelihood 为 $24\log q+6\log[(1-q)/2]$，求导为零得到 $\hat q=24/30=0.8$。这些是给定教学计数，位置标签只用于校准；运行中的取件策略仍只见路口与扫描标签。此处比较错误模型与校准模型，不把校准标签当成部署观察。

![扫描正确率的模型估计怎样越过动作排序阈值，改变首动作](https://yingwen.io/crl-figures/belief-planning-walkthrough-model.svg)

横轴是给定扫描模型的正确率，纵轴是该模型下的精确候选期望。紫线评估条件扫描计划，橙虚线为直接取左；红点旧模型 0.4 把扫描算成 0.9，蓝点校准模型 0.8 算成 2.7。真实传感器保持 0.8，所以旧模型动作的实际期望为 1，损失 1.7。原创解析曲线与给定校准计数；没有神经训练或采样性能曲线。

重复枚举同一个旧模型仍得到同一排序；缩短 rollout 也不会补回第一步观测核的错误。将模型失配、未来信息被序列目标遗漏和计算预算不足分别干预，才能知道哪个对象需要改。这里的模型误差影响行动选择；前面的长 rollout 递推则讨论误差怎样随深度传播，二者可以同时存在。

<a id="lesson-latent"></a>

## 5 · Dreamer 把模型计算用于想象中的策略学习

$$
h_t=f_\psi(h_{t-1},z_{t-1},a_{t-1}),\quad z_t\sim q_\psi(z_t\mid h_t,o_t),\quad z_{t+1}\sim p_\psi(z_{t+1}\mid h_{t+1})
$$

有真实观察时 posterior 编码状态；想象时使用 prior 预测下一潜在状态。确定性记忆与随机变量承担不同角色。

$$
\mathcal L_{\rm model}=-\mathbb E_q\sum_t\left[\log p_\psi(o_t\mid h_t,z_t)+\log p_\psi(r_t\mid h_t,z_t)-\beta D_{\rm KL}(q_\psi\Vert p_\psi)\right]
$$

这是解释重建、奖励与潜在一致性的简化变分目标，不是 DreamerV3 完整工程损失。实际版本还有继续预测、KL balancing、free bits 和尺度变换等设计。

$$
\begin{gathered}\hat G_t^\lambda=\hat r_t+\gamma\hat c_t\left[(1-\lambda)V(\hat s_{t+1})+\lambda\hat G_{t+1}^\lambda\right]\\\hat G_H^\lambda=V(\hat s_H)\end{gathered}
$$

想象轨迹中的多步目标，末端使用 critic。奖励与继续概率由模型产生，因此目标同时受模型和价值误差影响。

Dreamer 家族在学得的世界模型中训练 actor–critic，部署通常可直接执行 actor。MPC 则在当下选择动作时做序列优化。潜在模型不意味着完全无需观察重建，也不意味着任意 latent 就是控制充分状态；应查看具体版本训练哪些预测、哪些梯度流入模型。

回到仓库，在线规划器可以每次拿当前 belief 枚举计划；也可以把“收到左标签取左、收到右标签取右”的映射保存为策略，部署时直接查询。求出一份小型条件计划仍是已知模型下的规划结果，教程没有训练策略参数。MBPO [§5、算法2](https://arxiv.org/html/1906.08253)将短模型样本交给 SAC 更新；DreamerV3 [Critic learning 与 Actor learning](https://arxiv.org/html/2301.04104)用想象轨迹训练 actor–critic，并从 actor 采样环境动作而不作前瞻规划。它们把模型计算用于改善可直接执行的策略，与此刻重新优化候选的使用位置不同。

Dreamer 的模型状态包括确定性记忆和随机潜变量。一次潜变量样本、潜变量均值及完整 belief 是三种对象；网络训练得到潜在状态也不自动带来精确 belief 的充分性。仓库的均值反例说明，应检验表示能否保存不同未来行动的后果，不能只检查位置均值或重建损失。

<a id="course-latent-coordinates"></a>

## 5.1 · 世界没变，模型为什么也会过时

潜在模型学习的是 $z_t=e_\psi(h_t)$ 坐标中的转移，而不是直接拥有环境真实状态。若 encoder 改变，正确的潜在转移、奖励头和价值头都可能需要同时改变。即使真实世界保持不变，也会出现这种内部非平稳性。

$$
s_{t+1}=s_t+1,\quad z_t=s_t,\quad M(z)=z+1.
\qquad \tilde z_t=2s_t\ \Longrightarrow\ \tilde M(\tilde z)=\tilde z+2
$$

一维反例：只把表示尺度翻倍，旧模型的“加一”就变错了。若奖励 r=s，旧奖励头 r(z)=z 也应改成 r(tilde z)=tilde z/2。错误来自坐标变化，不是外部动力学变化。

若保存的是原始观察或可重建历史，可以用当前 encoder 重新编码，但需要额外计算，也可能受有限历史长度限制。若只保存旧 latent，重新解释它需要一个可靠的坐标变换；没有额外信息时，这个变换未必可识别。固定 encoder、慢速 target encoder 和联合训练是不同的时间尺度选择，并非普遍解。

规划还会主动寻找模型认为有利的动作。平均一步误差小，不代表被规划器偏好的一小块区域也准确。模型的评估分布、真实行为分布和想象规划分布应分开记录。必要时检查特定计划的多步奖励预测，而不是只看重建图像是否清晰。

在持续知识构建中，模型更新与状态构造必须协同：哪些预测可以保持旧含义，哪些需要重新学习，哪些需要重新收集经验？增加模型 rollout 数量只增加对现有模型的使用，不能替代新世界提供的反证。

<a id="lesson-algorithm"></a>

## 6 · 三个模型使用位置

**算法：本章把机制分别说明，不将三个名字当成同一算法的可互换实现。**

1. 收集真实转移，更新模型，并在留出数据上检查误差。
1. MPC：从当前状态生成候选序列，评价奖励和终端价值。
  1. 执行第一个动作，观察后重新规划。
1. MBPO：从真实 replay 状态分支出短模型轨迹。
  1. 混合真实和模型数据，按声明比例更新 off-policy agent。
1. Dreamer：由真实序列推断潜在状态，再在 prior 中想象。
  1. 在想象轨迹上训练 actor 和 critic。
1. 分别计数真实数据、模型生成、梯度更新与决策时间。

<a id="lesson-example"></a>

## 7 · 一维控制与错误模型

状态初值零，目标二，动作集合 $(-1,0,1)$，模型 $\hat s'=s+a$。两步成本为每个后继状态到目标的平方距离加 $0.1a^2$。序列 $(1,1)$ 得到状态 $(1,2)$，成本 $1+0.1+0+0.1=1.2$，优于 $(1,0)$ 的 2.1。MPC 此刻只执行第一个一。

若真实增益为 0.5，同一组三个动作一产生真实状态 $(0.5,1,1.5)$，模型却预测 $(1,2,3)$。误差是 $(0.5,1,1.5)$，恰好达到 $L_f=1,\epsilon_f=0.5$ 的累计界。重规划可以利用新状态，但若模型增益一直错误，偏差不会自动消失。

![MPC 枚举模型候选、只执行首动作，再从真实新观察重新规划](https://yingwen.io/crl-figures/concept-classic-model-rollout.svg)

每列横向表示位置、纵向表示两步时间；淡蓝线保留九条候选序列，粗蓝线是模型选中路径。初值零时预测序列 (1,1) 成本 1.2，只执行第一步；真实增益 0.5 产生新观察 0.5。模型增益仍为一，从新状态重规划选 (1,0)，预测成本 0.6。橙色是已发生的真实转移，模型路径不算真实数据。原创位置—时间重绘与精确枚举，不是模型训练的性能图。

<a id="lesson-code"></a>

## 8 · 学模型、枚举规划与想象回报核

标量模型、精确枚举 MPC、误差递推与有限想象 λ-return；本核的想象区间内继续概率均为一，末端使用给定价值。

```python
def fit_scalar_gain(actions, increments):
    if len(actions) != len(increments) or not actions:
        raise ValueError('paired action and state-increment samples required')
    denominator = dot(actions, actions)
    if denominator == 0:
        raise ValueError('no excitation: gain is not identifiable')
    return dot(actions, increments)/denominator


def model_rollout(state, sequence, gain=1., drift=0.):
    states = []
    for action in sequence:
        state = state+gain*action+drift
        states.append(state)
    return states


def mpc_action(state, goal, actions=(-1., 0., 1.), horizon=2,
               gain=1., action_cost=.1, terminal_weight=0.):
    """Enumerate open-loop candidates; return only the first action to execute."""
    if horizon < 1 or not actions:
        raise ValueError('positive horizon and nonempty actions required')
    best = None
    for sequence in itertools.product(actions, repeat=horizon):
        states = model_rollout(state, sequence, gain)
        cost = sum((s-goal)**2+action_cost*a*a for s, a in zip(states, sequence))
        cost += terminal_weight*(states[-1]-goal)**2
        if best is None or cost < best[0]:
            best = cost, sequence, states
    return best[1][0], best


def rollout_error_bound(one_step_error, lipschitz, horizon):
    """Same actions, same initial state; error_{k+1} <= L error_k + epsilon."""
    if min(one_step_error, lipschitz, horizon) < 0:
        raise ValueError('nonnegative inputs required')
    error, errors = 0., []
    for _ in range(horizon):
        error = lipschitz*error+one_step_error
        errors.append(error)
    return errors


def imagined_lambda_return(rewards, values, gamma=.9, lam=.8):
    """values[k] = V(s_k); final value supplies the finite imagination tail."""
    if len(values) != len(rewards)+1:
        raise ValueError('one more state value than reward required')
    carry, returns = values[-1], []
    for k in reversed(range(len(rewards))):
        carry = rewards[k]+gamma*((1-lam)*values[k+1]+lam*carry)
        returns.append(carry)
    return list(reversed(returns))
```

运行 demo 得到两步最优候选与误差曲线，运行 test 检查最小二乘、规划序列及 λ 的两个极端。它不是完整 PETS、MBPO 或 Dreamer 复现。PETS 与 MBPO 作者仓库提供原论文工程；danijar/dreamerv3 明确是作者维护的重实现，不能冒充原内部实验快照。

同一取件任务使用 [belief-planning-walkthrough.py](/crl-code/tutorials/belief-planning-walkthrough.py)：下载单文件，运行 python3 belief-planning-walkthrough.py 查看模型排序、三条观察分支与最优计划；加 --test 做精确分数和 27 种计划的检查。代码还对比固定序列、模型 belief backup 和给定计数的传感器拟合；它没有实现神经模型、PETS 采样规划、MBPO 的 SAC 更新或 Dreamer 的 actor–critic 训练。

<a id="lesson-branches"></a>

## 9 · 持续变化中的模型寿命

- 模型陈旧：动力学改变后，更多规划可能放大旧知识错误，而不是补偿它。
- 表示错误：低重建损失可能忽略控制相关变量；belief 与 latent 的充分性要单独检验。
- 未知继续概率：把任务截止和人为采样停止混为一类，会改变想象长度与价值。

CRL 可以研究模型何时失效、哪些可复用以及每步规划预算怎样分配。有效对照包括真实模型 oracle、冻结旧模型和在线更新模型，并保持总计算预算一致。模型能支持未来新目标，才与持续构建的可复用知识直接相关。

<a id="lesson-check"></a>

## 10 · 练习与答案

- 问：模型误差很小，长规划是否总是更好？答：不保证；误差累积、优化利用偏差和终端价值误差仍有影响。
- 问：MPC 必須执行完整最优动作序列吗？答：不必，标准 receding-horizon 只执行首动作并重新规划。
- 问：拟合标量增益时所有动作都是零，会得到什么？答：分母为零，数据无法识别增益，不能静默给出可信模型。
- 实验：保持候选动作不变，将模型增益改为 0.5，对照真实轨迹和原模型轨迹；区分模型学习带来的变化与增加规划候选数带来的变化。



<a id="chapter-code"></a>

## 下载与运行

标准库解析机制实验；不包含完整神经网络训练或大型 benchmark。

[下载 extended_foundations_lab.py](https://yingwen.io/zh/continual-rl/download/extended_foundations_lab.py)

```sh
python3 extended_foundations_lab.py test
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Chua et al. · Deep Reinforcement Learning in a Handful of Trials](https://arxiv.org/abs/1805.12114)：PETS 原文：概率模型与候选轨迹评估。

- [Chua et al. · PETS 作者代码](https://github.com/kchua/handful-of-trials)：概率动力学 ensemble 与 trajectory sampling 的原论文实现。

- [Janner et al. · When to Trust Your Model](https://arxiv.org/abs/1906.08253)：MBPO 的短分支 rollout、模型偏差与实践动机。

- [Janner · MBPO 原作者代码](https://github.com/jannerm/mbpo)：真实/模型 buffer、rollout schedule 和 SAC 工程。

- [Hafner et al. · Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)：DreamerV3 世界模型与想象控制。

- [Hafner · dreamerv3](https://github.com/danijar/dreamerv3)：作者维护的重实现；仓库公开说明与原内部实现的区别。

- [Kochenderfer、Wheeler、Wray · Algorithms for Decision Making](https://algorithmsbook.com/)：作者书站提供决策、信念状态、模型与规划教材及配套 Julia 代码入口。

- [UC Berkeley · CS 285](https://rail.eecs.berkeley.edu/deeprlcourse/)：原课程的 exploration、model-based RL 与 offline RL 讲义和视频入口；按问题专题阅读，不必按网络规模划分领域。

- [Kapturowski et al. · Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/forum?id=r1lyTjAqYX)：原论文分析 replay 中的参数延迟、表示漂移、recurrent state staleness，以及 stored state 和 burn-in 的取舍。


[本章配套阅读与原始材料](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-model-based#topic-directory) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-model-based#crl-catalog) · [人物与本章的关系](https://yingwen.io/zh/continual-rl/people/#people-study-deep-model-based)
<a id="study-connections"></a>

## 与教材主线的衔接

本章是可按问题选择的并列研究分支，不要求先读完其他深度研究分支。

### 模型需要预测什么，内部计算应花在哪里？

规划根据模型更新价值或选择动作。Dyna 把真实经验更新、模型拟合和规划查询分开。模型更新次数更多，不代表新增了真实证据。

函数逼近与深度方法：线性价值可使用期望特征模型；非线性价值通常不能把后果分布替换成均值。Option 模型还要保留随机持续时间、真实奖励和终点的关系。

持续学习中的研究问题：当表示、技能或环境改变时，哪些旧模型仍可复用？有限计算应优先用于收集真实经验、改进模型，还是在已有模型中规划？

[Dyna 与搜索控制](https://yingwen.io/zh/continual-rl/foundations/tabular/planning/) → [神经模型与规划](https://yingwen.io/zh/continual-rl/foundations/deep/model-based/) → [Option 后果模型](https://yingwen.io/zh/continual-rl/construction/models/) → [规划与计算分配](https://yingwen.io/zh/continual-rl/construction/planning/)


### 可进一步检验的问题

- [10 · 模型需要预测什么，才能在变化后继续支持决策？](https://yingwen.io/zh/continual-rl/research/#research-reusable-models)：模型误差经过 rollout 和末端价值进入动作选择，因而模型应保留什么信息取决于它的决策消费者。
- [11 · 什么时候值得规划，应该把计算花在哪里？](https://yingwen.io/zh/continual-rl/research/#research-planning-budget)：MPC、短模型 rollout 与潜在想象在不同阶段使用模型，比较规划预算时需同时记录模型误差与真实交互成本。


[领域总览与问题地图](https://yingwen.io/zh/continual-rl/overview/) · [奖励假设与设计](https://yingwen.io/zh/continual-rl/foundations/reward-design/) · [持续控制：完整学习器的比较](https://yingwen.io/zh/continual-rl/algorithms/control/)

- [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)
- [时间抽象与规划](https://yingwen.io/zh/continual-rl/construction/planning/)
- [Dyna 与模型学习](https://yingwen.io/zh/continual-rl/algorithms/dyna/)
- [智能体状态与递归学习](https://yingwen.io/zh/continual-rl/construction/state/)
- [持续学习的智能体架构](https://yingwen.io/zh/continual-rl/construction/architectures/)

对应原始材料：PETS；Janner et al.：MBPO；Hafner et al.：DreamerV3；Algorithms for Decision Making：planning。本文为原创讲解，原书、论文与上游代码保留各自许可。
