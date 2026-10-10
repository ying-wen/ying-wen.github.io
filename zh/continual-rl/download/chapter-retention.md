# 知识保留：经验重放、参数约束与模型记忆

新经验要求适应，旧技能又可能重新有用；如何在固定预算内管理它们的冲突？

## 本章内容

- 区分任务再次出现时的遗忘、分布改变后的必要适应，以及网络失去学习能力。
- 从目标函数推导 replay、EWC、策略蒸馏与 CLEAR，并把算法落实为完整存储/采样/更新流程。
- 用固定内存、相同数据与计算预算设计保留—适应对照，而不让“存得更多”掩盖算法贡献。

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [持续控制：比较策略与学习智能体](https://yingwen.io/zh/continual-rl/algorithms/control/)：区分旧任务诊断与生命期收益。
- [函数逼近预测：从回归到 TD 固定点](https://yingwen.io/zh/continual-rl/foundations/approximation/prediction/)：理解共享参数的误差与干扰。


### 训练分布与评价分布

当前数据分布由环境和行为策略共同产生；评价可以关注当前环境、历史环境混合或整个生命期收益，需要事先指定。

### KL 与策略蒸馏

$D_{\rm KL}(p\Vert q)=\sum_a p(a)\log[p(a)/q(a)]$。蒸馏在指定状态上保持教师分布 $p$，学生参数仍可改变。

### 二阶近似

在旧解附近，用曲率加权的二次函数近似旧损失；该近似描述局部敏感性。

<a id="problem-definition"></a>

## 本章的问题定义

新经验要求改变行为，而过去能力可能再有用途；在固定存储和更新预算内控制保留—适应冲突。

### 给定条件与符号

- 当前经验、历史能力的独立评价分布和未来回访协议。
- 固定内存、更新预算、可保存的经验/参数/教师输出及是否提供任务边界。

### 需要求解的对象

在当前适应与声明历史能力上满足可检验取舍的学习器；保留对象可为经验、参数近似或功能输出。

### 信息与数据权限

旧数据和教师输出来自当时行为；重放需处理行为差异，动力学变化不能仅由动作概率比修复。评测副本不得反馈数据或重置主智能体。

$$
\min L_{\rm now}(\theta)\quad\text{s.t.}\quad L_{\rm hist}(\theta)-L_{\rm hist}(\theta_{\rm ref})\le\varepsilon,\quad M\le B
$$

$\theta$ 为当前参数，$\theta_{\rm ref}$ 是历史参考，$L_{\rm now}$ 为当前声明损失，$L_{\rm hist}$ 是固定历史诊断分布的损失，$\varepsilon$ 为允许退化，$M$ 为持久内存，$B$ 为预算。此式定义一种可操作取舍；replay/EWC/CLEAR以不同代理近似它，RL最终还需在线外部收益。

### 成立条件与解的含义

- 历史诊断保持同一任务语义；永久失效的旧行为不应自动当作必须保持的能力。
- 固定预算包含buffer、教师、参数锚与统计；旧任务边界和真标签是额外权限。

判断准则：A→B→A或等价独立probe记录A的下降、B学习速度与A恢复；匹配容量、数据和更新数，报告全程收益，区分遗忘、负迁移和必要适应。

### 适用边界

- 保留不是无条件阻止参数变化或复制旧错误。
- 历史训练损失低不能替代旧能力在固定分布上的诊断。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [可塑性与特征更新](https://yingwen.io/zh/continual-rl/algorithms/plasticity/)：保留测旧能力下降，可塑性测新学习速度；两者可能同时出现但需不同对照。

- 组合不同学习问题 · [转移模型与后果模型](https://yingwen.io/zh/continual-rl/construction/models/)：模型/SF也可保留结构知识，但模型过期与重估成本须另测。

- 改变信息或数据协议 · [流式更新与稳定性](https://yingwen.io/zh/continual-rl/algorithms/streaming/)：原始经验replay超出严格不重放协议，参数统计和功能约束也需记入预算。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

当前梯度会覆盖旧行为，而保存全部经验或每任务完整模型不可持续扩展。

### 本章的核心思路

先确定未来需保留什么，再用预算内的经验、局部几何或教师输出近似其约束。

1. [选择代表的历史经验分布](#lesson-derive)：因为FIFO与reservoir保留不同年龄分布，明确保存与训练采样规则，匹配所需历史评价。

2. [纠偏历史学习并约束功能漂移](#lesson-clear)：因为旧行为与当前策略不同，CLEAR用V-trace处理旧经验并用旧输出克隆限制漂移，两项目的分开。

3. [以局部参数几何压缩历史](#lesson-ewc)：因为不能总保留数据，EWC保存锚点与对角Fisher近似；检查边界触发、容量和局部近似误差。

结论与条件：reservoir的时间索引保留概率可精确检验；EWC是局部曲率代理，CLEAR依赖覆盖和off-policy条件，均不提供任意变化任务的无遗忘保证。

### 相关方法改变了什么

- Replay：保留真实输入和经验，采样分布与行为纠偏决定用途。

- EWC：保留参数附近的局部损失几何，忽略部分相关方向。

- 蒸馏/CLEAR：保留指定状态上的功能输出，并可用RL项继续改进，强约束也会保存错误。


<a id="lesson-setting"></a>

## 1 · 保留的对象与评价目标

共享参数让新经验影响许多状态，也可能改变很久没有访问的状态上的行为。考虑一个曾学会走左门的导航智能体。门的开关暂时改变，之后又恢复；若新训练破坏了旧技能，回来时就要重新付出学习代价。知识保留关心怎样减少这种代价。但若门永久封死，保留旧动作偏好反而妨碍适应。因此首先要说明历史知识以后在哪些条件下仍有用，再选择保留的数据、参数约束或模型。

用 A→B→A 的重复环境可以观察保留，但它只是一个研究协议：是否给任务 ID、是否外部 reset、每段多长、是否重新初始化优化器，都影响解释。真实 single-life 不能随时回 A 测试；可以用独立评测副本诊断，但副本不能向在线 agent 提供额外训练数据或重置其主状态。

| 现象 | 可检验问题 | 不能混称为 |
| --- | --- | --- |
| 遗忘 | A 的表现训练 B 后是否下降？ | 不一定是 B 学不会 |
| 负迁移 | 有历史的 agent 学 B 是否比 fresh 慢？ | 不一定是旧 A 被遗忘 |
| 可塑性丧失 | 相同新数据与优化预算下 aged 学习能力是否下降？ | 仅凭低回报不能确诊 |
| 必要适应 | 旧行为已失效时能否及时改变？ | 不是所有旧性能下降都应阻止 |

$$
L_t(\theta)=(1-\omega)L_{\rm now}(\theta)+\omega L_{\rm retained}(\theta),\qquad 0\leq\omega\leq1
$$

这是说明稳定—适应取舍的目标骨架。历史损失可以由真实 replay、参数近似或旧模型输出构造；不同构造保留的是不同信息。

先固定预算：参数、目标网络、教师网络、buffer 中的观测/动作/概率/价值、统计量都按字节计算；还要固定每个环境步允许的梯度更新数。一个方法多训练十次或保存整个历史，不能只与只用一条新样本的方法比较“算法更好”。

<a id="lesson-shared-switch"></a>

### 同一反转问题：忘得更多，也可能学得更快

先把回报、探索和数据覆盖暂时拿开，只看一次目标反转。仪器给出左右线索 $x\in\{-1,+1\}$，两者等权；旧条件 A 的反馈为 $y=x$，新条件 B 为 $y=-x$。预测器是两层标量网络 $f_{u,v}(x)=v(ux)$，没有偏置或激活函数。它对输入仍是线性的，对两个待学习权重则是双线性的。本例给定已经准确预测 A 的检查点 $(u,v)=(1,1)$，不模拟其先前训练史。

$$
q=uv,\qquad L_A=\tfrac12\mathbb E_x[(qx-x)^2]=\tfrac12(q-1)^2,\qquad L_B=\tfrac12\mathbb E_x[(qx+x)^2]=\tfrac12(q+1)^2.
$$

每个期望都对完整的两个输入取等权平均。每次更新使用 B 的两个带标签输入；之后只读计算 A、B 的精确平均损失。没有抽样误差，也不以这个全支持评价估计未见数据上的泛化能力。

比较两个副本。第一份保留 $(1,1)$；第二份在学习 B 前做一次保持函数不变的变换 $(u,v)\mapsto(2u,v/2)$，得到 $(2,1/2)$。二者的每个初始预测都相同，旧损失都是 0、新损失都是 2，参数个数也相同。另列固定 fresh 参照 $(1,0)$：它的初始预测为 0，新旧损失均为 $1/2$，所以不能假定它与旧检查点有相同起点。

$$
r_k=u_kv_k+1,\qquad u_{k+1}=u_k-\alpha r_kv_k,\qquad v_{k+1}=v_k-\alpha r_ku_k,\qquad \alpha=\tfrac14.
$$

这是对 $L_B$ 的普通梯度下降。两个右端都使用第 $k$ 次更新前的权重；$\alpha$ 按一次完整梯度更新计，不按环境交互步计。本例没有动量、归一化统计或随机状态。

第一步就能看出分歧。$(1,1)$ 变为 $(1/2,1/2)$，因而 $q=1/4$、$L_B=25/32=0.78125$、$L_A=9/32=0.28125$。$(2,1/2)$ 则变为 $(7/4,-1/2)$，因而 $q=-7/8$、$L_B=1/128=0.0078125$、$L_A=225/128=1.7578125$。新目标学得更快的一份，此刻在旧目标上反而错得更多。

看图前先预测：如果只展示旧目标损失，哪份副本看起来较好？再加入新目标损失，你会怎样解释这个排序？

![相同左右线索的旧、新目标反号；两幅相同纵轴的损失轨迹分别显示新目标适应和旧目标误差。蓝实线对称旧解，橙虚线等预测重参数化，紫点线固定fresh参照。](https://yingwen.io/crl-figures/retention-plasticity-walkthrough-losses.svg)

原创确定性梯度算例。横轴是 B 上的更新次数，每步使用两个输入，步长 1/4；两个损失面板共用 0–2 的纵轴。蓝、橙两份初始预测相同，紫色 fresh 起始误差较小。橙线的新损失迅速下降，同时旧损失上升得更多；此图测量预测误差，不是深度 RL 训练回报。

| 副本 | B 学习前的 $L_B$ | B 学习八步后的 $L_B$ | 同一检查点只读 $L_A$ |
| --- | --- | --- | --- |
| 对称旧解 (1,1) | 2 | 0.503174 | 0.496836 |
| 等预测重参数化 (2,1/2) | 2 | 约 $1.08\times10^{-14}$ | 2.000000（约） |
| 固定 fresh (1,0) | 0.5 | 0.00008165 | 1.974523 |

旧损失由 0 上升，说明原有预测功能没有保持；新损失是否下降则回答另一问题。这里 A 与 B 要求相反预测，适应 B 本来就会改变 A 的输出。将这项改变称为需要防止的遗忘，还依赖 A 是否会再次有用。如果后来回到 A，还要测量重新学习的代价，不能从这个检查点的旧误差直接推断恢复速度。下一章沿用这两个副本解释它们为何走上不同轨迹。

[下载单文件标准库脚本](/crl-code/tutorials/retention_plasticity_walkthrough.py)，在保存文件的目录运行下方命令，可重算两章全部轨迹；[逐步数值](/crl-figures/retention-plasticity-walkthrough-data.json)和[独立 JavaScript 计算核](/crl-code/figures/retention-plasticity-walkthrough.mjs)给出图中对应值。

无需第三方库；输出 JSON，包含逐步参数、旧/新损失及副本上的重新学习轨迹

```bash
python3 retention_plasticity_walkthrough.py
```

<a id="lesson-derive"></a>

## 2 · 经验重放与历史采样分布

Replay 使已经不再出现的状态继续影响梯度。FIFO buffer 强调近期数据；reservoir sampling 则在固定容量 $M$ 下，让前 $t$ 条样本具有相同的保留概率。两者代表的训练分布不同，应由未来评价需求决定。

**算法：Reservoir sampling；保存分布与训练时的采样分布分别定义**

1. 初始化样本计数 $t=0$，buffer 容量为 $M$
1. 每来一条 transition：
  1. 令 $t\leftarrow t+1$
  1. 若 buffer 未满，追加样本
  1. 否则均匀采样 $j\in\{0,\ldots,t-1\}$
    1. 若 $j<M$，以新样本替换 buffer 的第 $j$ 项
1. 训练时按指定的新旧样本比例采样并更新

当 $t>M$ 时，新样本被接收的概率为 $M/t$。任一旧样本此前被保留的概率为 $M/(t-1)$，本步被替换的条件概率为 $1/t$；于是存活概率为 $[M/(t-1)](1-1/t)=M/t$。均匀的是时间索引，并不意味着每个任务、状态或稀有事件等权。固定的是样本槽位数；无限运行时精确计数 $t$ 的位数仍会增长。严格固定字节协议须给出最大寿命或有限精度与溢出规则，不能把这两个资源声明混为一谈。

$$
\mathbb E[\widehat g]= (1-\omega)\mathbb E_{d_t}[\nabla\ell]+\omega\mathbb E_{d_B}[\nabla\ell]
$$

dB 由保存与采样共同决定。损失权重、batch 中新旧比例和重用次数都能改变实际有效权重，不能只报告 buffer 容量。

强化学习还有一个监督学习没有的困难：旧 transition 由旧策略 μ 产生，当前要改进的是 π。旧奖励和转移样本在平稳环境中仍可有效，但策略梯度与多步 bootstrap 需要处理行为差异；环境动力学若也变了，动作重要性比并不能把旧世界样本变成新世界样本。

<a id="experiment-reservoir_replay"></a>

### 实验：实验 · 均匀保留历史，可能妨碍适应当前世界

同一输入的正确答案永久反转后，旧数据回放究竟是在保护知识，还是在拟合已经失效的答案？

**环境与可用信息。** 两维线性回归，输入 [1,U(−1,1)]。第 601 个样本起，真权重由 [0.2,0.7] 变为 [−0.2,−0.7]；噪声标准差 0.03。没有任务 ID，因此一个预测器不能同时为同一输入给出两个相反的条件均值。

**设置。** 五种子各 1200 个真实样本，权重从零开始，步长 0.03。回放容量 32；每个新样本先训练一次，再存入 reservoir 并额外均匀回放一次。在线 SGD 每个新样本只训练一次。真实数据流配对，回放单独使用随机流。

**检验的机制。** Reservoir 均匀覆盖时间索引，不偏向最新目标。第二阶段混合旧标签和新标签，相当于改变训练分布。额外计算并不能消除两个目标本身的冲突。

**测量。** 同时看当前任务 MSE 与旧任务保留误差图。前者低表示适应新函数；后者低表示接近原函数。横轴相同仅表示真实样本数相同；回放用了两倍参数更新。

```bash
python3 implementations/continual/reservoir_replay.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/reservoir_replay/curves.svg)

横轴：environment_steps。纵轴：当前任务无噪声预测 MSE。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 末尾当前 MSE：回放 0.0660、在线 SGD 0.0000327。旧任务 MSE：回放 0.4428、在线 SGD 0.8069。回放确实更接近旧函数，却明显拖慢了新函数适应。两个指标共同显示折中，不能只取旧误差讲“全面改善”。

**结论边界。** 这是冲突目标的监督预测，不是循环任务控制。实验没有再次返回旧任务，不能测量回访时重新学习的成本。若旧答案永久失效，保留本身未必是合适目标。

**继续实验。** 将序列改为 A→B→A，明确是否提供上下文，再与同容量 FIFO、等更新次数的当前样本重复训练比较。分别报告保留收益、恢复速度和额外计算。

[源码](https://yingwen.io/crl-code/implementations/continual/reservoir_replay.py) · [逐种子记录](https://yingwen.io/crl-code/results/reservoir_replay/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/reservoir_replay/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/reservoir_replay/curves.json)

<a id="lesson-clear"></a>

## 3 · CLEAR：新经验学习、旧经验校正、旧输出克隆

CLEAR 结合三类更新：当前经验支持适应；历史经验通过 off-policy actor–critic 继续改善价值与行为；旧状态上的策略和价值克隆约束输出漂移。为此，存储项除转移外还包含旧策略分布与旧价值输出。该方法不要求任务边界，但保留效果取决于 replay 覆盖。

$$
\begin{aligned}\bar\rho_t&=\min(\rho_{\max},\pi(A_t\mid S_t)/\mu(A_t\mid S_t)),\\c_t&=\min(c_{\max},\pi(A_t\mid S_t)/\mu(A_t\mid S_t)),\\d_t&=\bar\rho_t[R_{t+1}+\gamma V(S_{t+1})-V(S_t)],\\v_s^{\rm trace}&=V(S_s)+\sum_{t=s}^{s+n-1}\gamma^{t-s}\!\left(\prod_{i=s}^{t-1}c_i\right)d_t.\end{aligned}
$$

这是常数折扣下长度为 $n$ 的 V-trace 目标，空乘积为 1。$\bar\rho_t$ 调整本步误差，其截断影响固定点对应的有效策略；$c_t$ 控制跨步传播与方差，在原理论条件下不改变该固定点。真实终止位置用零折扣截断，unroll 的非终止末端保留价值 bootstrap。

$$
\begin{aligned}L_V&=\tfrac12(V_\theta(s)-\operatorname{stopgrad}(v_s^{\rm trace}))^2,\\L_\pi&=-\log\pi_\theta(a_s\mid s)\operatorname{stopgrad}\!\left[\bar\rho_s(r+\gamma v_{s+1}^{\rm trace}-V_\theta(s))\right],\\L_{\rm clone}&=\beta_\pi D_{\rm KL}(\mu_{\rm stored}\Vert\pi_\theta)+\beta_V(V_\theta-V_{\rm stored})^2.\end{aligned}
$$

RL 项应用于新旧数据，cloning 项用于 replay；再按原配置加入熵项和各损失权重。旧 logits、旧值与 trace target 不通过梯度回流到历史模型。actor 的整个重要性加权 advantage 都要 stop-gradient；若把 ρ̄ 留在外面，会额外对概率比求导，改变政策梯度估计器。

**算法：算法伪代码**

1. 动作采样前保存 μ(a|s)、完整策略输出和 Vstored(s)
1. 环境返回 transition；将新 unroll 与历史 unroll 按规定比例组成训练数据
1. 用当前网络计算 π、V；用存储 μ 计算裁剪的重要性比
1. 自后向前计算 V-trace targets；将 targets stop-gradient
1. actor 使用整体 detach 的 weighted advantage（包括重要性比）；计算 critic、entropy；只对 replay 加策略/价值 cloning
1. 合并损失做一次优化；按容量策略存入新经验
1. 记账 buffer 字节数、unroll 长度、每步更新数与 replay 年龄

旧策略未必最优，因此克隆与 RL 更新有不同作用：前者限制偏离，后者利用奖励继续改进。强克隆可能保存旧错误，弱克隆则可能无法保留旧能力。比较方法时需要同时观察当前适应与历史回访表现。

<a id="lesson-ewc"></a>

## 4 · EWC：用参数曲率近似旧知识

无法保存旧数据时，可以保留旧解 $\theta_A$ 与每个参数的重要性。若旧损失在 $\theta_A$ 的梯度近似为零，二阶展开中的 Hessian 描述沿哪些方向移动代价大。EWC 用对角 Fisher 近似这种局部几何，从而让重要参数较难改变。

$$
\begin{aligned}L_A(\theta)&\approx L_A(\theta_A)+\tfrac12(\theta-\theta_A)^\top H_A(\theta-\theta_A),\\L_{\rm EWC}(\theta)&=L_B(\theta)+\frac{\kappa}{2}\sum_iF_i(\theta_i-\theta_{A,i})^2,\\\nabla_iL_{\rm EWC}&=\nabla_iL_B+\kappa F_i(\theta_i-\theta_{A,i}).\end{aligned}
$$

Fisher 来自指定概率模型的 score 外积期望：$F_i=\mathbb E[(\partial_{\theta_i}\log p_\theta(y\mid x))^2]$。真实 Fisher 在模型分布上取期望；经验 Fisher 改用观测标签。两者与实际 RL 损失的 Hessian 不普遍相等，因此样本、动作及分布都是估计器定义的一部分。

完整流程是：在约定的 consolidation 时刻保存 θA；在相应数据分布上估计并冻结 F；后续每步把二次惩罚加入当前损失。若没有任务边界，要定义 consolidation 的触发或连续累计规则。每个任务另存一个 F 与参数快照会随任务数增长；online EWC 将历史重要性折叠成固定规模统计，但其遗忘系数与近似误差需要评估。

EWC 保留的是参数附近的旧损失近似，不是完整旧数据，也不保证旧动作概率不变。网络可用不同参数实现同一功能；对角近似忽略参数之间的相关方向。因此功能空间的蒸馏与参数空间的正则可以产生明显不同的行为。

<a id="experiment-ewc"></a>

### 实验：实验 · 二次参数约束为何留下非零适应误差？

新目标已经可表示且 SGD 能拟合时，EWC 仍有误差是否意味着网络失去可塑性？

**环境与可用信息。** 与回放实验相同的二特征权重反转任务：600 个旧任务样本，随后 600 个新任务样本。标签噪声标准差 0.03，参数从零开始。

**设置。** 五种子，步长 0.03，二次约束强度 0.5。第 601 步之前保存旧参数锚点，以前 600 个样本的特征平方均值作为对角 Fisher。该 Fisher 来自单位方差线性 Gaussian 工作模型，并非直接使用真实噪声标准差 0.03 的模型。

**检验的机制。** 新任务梯度把参数推向反向权重；旧锚点的二次罚项把它拉回。两者平衡后可以留下非零误差。这是被修改后的目标的折中，不需要用“学不动了”解释。

**测量。** 比较同一时刻的当前 MSE 与旧 MSE，检查约束强度改变后的两条曲线。EWC 被明确告知 consolidation 边界，在线 SGD 没有这项信息；本实验不是等信息比较。

```bash
python3 implementations/continual/ewc.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/ewc/curves.svg)

横轴：environment_steps。纵轴：当前任务无噪声预测 MSE。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 末尾当前 MSE 为 0.0978，在线 SGD 为 0.0000327；旧 MSE 为 0.3497，对照为 0.8069。约束保存了更多旧函数，却使新函数误差更大。这里没有证据说明基础线性学习器本身丧失了学习能力。

**结论边界。** 二特征对角近似只展示目标折中，不能代表大网络 Fisher 的质量。锚点和旧任务边界已知；未知边界下何时巩固仍未解决。

**继续实验。** 先令约束强度为零确认退化为 SGD，再逐步增大强度并画新旧误差的二维图。改变 Fisher 的工作方差时同时调整或报告正则系数，避免把量纲变化误当成新机制。

[源码](https://yingwen.io/crl-code/implementations/continual/ewc.py) · [逐种子记录](https://yingwen.io/crl-code/results/ewc/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/ewc/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/ewc/curves.json)

<a id="lesson-model-memory"></a>

## 5 · DRAGO：保留世界模型，而非只保留旧策略

如果任务改变的只是奖励与起点，不同任务会让智能体看到同一世界的不同区域。此时值得保留的是动力学知识：旧任务的最优策略未必适用于新任务，但旧房间的布局仍然有效。DRAGO（Fu 等，ICML 2025）研究这一设定，允许已知任务切换、当前任务 buffer，以及一个旧模型快照；不保存往期任务的原始数据。

先用生成器 $G$ 表示过去访问过的状态—动作联合分布。从中采样 $(\hat s,\hat a)$，由冻结的旧动力学模型 $T_{\rm old}$ 产生后继标签。当前模型同时拟合真实新经验与这些合成旧经验：

$$
\begin{aligned}(\hat s,\hat a)&\sim p_G,\qquad \hat s'=\operatorname{stopgrad}(T_{\rm old}(\hat s,\hat a)),\\L_{\rm dyn}(\psi)&=\mathbb E_{(s,a,s')\sim D_i}\|T_\psi(s,a)-s'\|^2+\lambda\mathbb E_{(\hat s,\hat a)\sim p_G}\|T_\psi(\hat s,\hat a)-\hat s'\|^2.\end{aligned}
$$

第一项校正当前可见区域，第二项保护过去区域的模型输出。与 CLEAR 的区别在于教师输出是后继状态预测，输入则由生成器产生。

生成器自身也会遗忘，因此用当前真实状态—动作对与上一生成器产生的样本共同训练新生成器。DRAGO 还用真实探索重新连接旧区域；这一部分补充了生成数据无法证明当前真实可达性的缺口。保留与重新获取知识是两种互补操作。

**算法：DRAGO 的知识保留流程；策略、价值、奖励头及探索器仍有各自训练目标**

1. 在已知任务边界保存冻结副本 $T_{\rm old}$ 与 $G_{\rm old}$
1. 每个训练阶段：
  1. 收集当前任务的真实转移，维护当前 buffer
  1. 从生成器采样旧状态—动作对，计算冻结模型的后继标签
  1. 以真实项与合成项联合更新当前动力学模型
  1. 以当前真实输入和 $G_{\rm old}$ 的样本训练当前生成器
  1. 用任务行为与记忆恢复行为收集新的真实经验
1. 下次任务切换时更新冻结副本

这个方法的边界来自问题设定：若真实动力学永久改变，旧模型标签可能已经错误，保持它会妨碍适应；若生成器遗漏了某个区域，蒸馏便没有对应输入。实验应分别测模型保持误差、跨区域规划与新任务适应，并计入生成器和冻结副本的存储、训练成本。作者 drago 仓库提供完整模型式实现；本页脚本仍聚焦 reservoir、EWC 和 CLEAR 的数值机制。

<a id="lesson-example"></a>

## 6 · 稳定性与适应性的二次例子

![新旧任务的两个抛物线损失，以及保留权重零、一、五对应的三个参数位置。](https://yingwen.io/crl-figures/concept-research-retention-conflict.svg)

上方两条曲线使用相同参数轴。下方每行标出一个加权联合目标的精确最小点；增加旧任务权重把参数拉向右侧。它改变的是优化目标，而不只是优化速度。计算脚本：research-mechanisms.mjs。

$$
L(w)=\tfrac12(w+1)^2+\tfrac\kappa2(w-1)^2,\qquad \frac{dL}{dw}=(w+1)+\kappa(w-1),\qquad w_*=\frac{\kappa-1}{\kappa+1}
$$

旧目标是 w=1，新目标是 w=−1，F=1。κ=0、1、5 分别得到 −1、0、2/3；这是精确二次例子，不是深网保证。

κ=0 完全适应新任务；κ=1 平均折中，但两个任务都不完美；κ=5 更接近旧解，却留下较大的新误差。这说明“新任务回报低”可能是强保留目标的结果，不一定是网络失去学习能力。

功能克隆的例子：旧策略 $\mu=(0.8,0.2)$，新策略 $\pi=(0.5,0.5)$，故 $D_{\rm KL}(\mu\Vert\pi)=0.8\log1.6+0.2\log0.4\approx0.192745$。旧值为 2、新值为 1 时，值克隆平方误差为 1。交换 KL 方向会改变数值与梯度，教师和学生的角色须固定。

两步 V-trace 样本的奖励为 $[0,1]$，价值全零、$\gamma=0.9$，所有比率与裁剪系数为 1，则第一状态目标为 $0.9$。若第一步目标策略概率为零，$\bar\rho_0=c_0=0$，第一状态本次没有轨迹修正，保持原值。这展示了当前误差与后续误差乘积的边界。

<a id="lesson-code"></a>

## 7 · 可运行实现与实验设计

固定容量 reservoir、EWC 精确小例子、克隆项与有限 unroll V-trace；不是完整 CLEAR 神经网络代理。

```python
def reservoir_insert(buffer, item, seen, capacity, rng):
    """seen is the number INCLUDING this item; O(capacity) retained data."""
    if capacity < 0 or seen < 1:
        raise ValueError("invalid budget or sample counter")
    if len(buffer) < capacity:
        buffer.append(item)
    elif capacity:
        index = rng.randrange(seen)
        if index < capacity:
            buffer[index] = item


def ewc_scalar_optimum(new_target, old_weight, importance, strength):
    if importance < 0 or strength < 0:
        raise ValueError("curvature and strength must be nonnegative")
    k = importance * strength
    return (new_target + k * old_weight) / (1.0 + k)


def ewc_update(weights, current_gradient, old_weights, importance,
               strength=1.0, alpha=0.1):
    """One SGD update; current_gradient excludes the consolidation penalty."""
    if not (len(weights) == len(current_gradient) == len(old_weights) == len(importance)):
        raise ValueError("EWC dimensions differ")
    if strength < 0 or any(f < 0 for f in importance):
        raise ValueError("EWC requires nonnegative importance")
    return [w - alpha * (g + strength*f*(w-old))
            for w, g, old, f in zip(weights, current_gradient, old_weights, importance)]


def clone_loss(teacher_prob, student_prob, teacher_value, student_value):
    """CLEAR-like replay cloning terms only, not the complete RL objective."""
    if len(teacher_prob) != len(student_prob):
        raise ValueError("policy dimensions differ")
    if any(p < 0 for p in teacher_prob) or any(q <= 0 for q in student_prob):
        raise ValueError("teacher >=0, student >0 required")
    if not math.isclose(sum(teacher_prob), 1.) or not math.isclose(sum(student_prob), 1.):
        raise ValueError("policies must sum to one")
    kl = sum(p * math.log(p / q) for p, q in zip(teacher_prob, student_prob) if p)
    return kl, (student_value - teacher_value) ** 2


def vtrace_target(rewards, values, rhos, gamma=0.9, rho_cap=1., c_cap=1.):
    """Finite unroll value target. values includes the final bootstrap value."""
    if len(values) != len(rewards) + 1 or len(rhos) != len(rewards):
        raise ValueError("unroll dimensions differ")
    if min(rho_cap, c_cap) < 0 or c_cap > rho_cap or any(rho < 0 for rho in rhos):
        raise ValueError("require 0 <= c_cap <= rho_cap and nonnegative ratios")
    correction, product = 0., 1.
    for t, reward in enumerate(rewards):
        delta = min(rho_cap, rhos[t]) * (reward + gamma * values[t+1] - values[t])
        correction += product * delta
        product *= gamma * min(c_cap, rhos[t])
    return values[0] + correction


def retention_demo():
    buffer, rng = [], random.Random(7)
    for seen in range(1, 101):
        reservoir_insert(buffer, seen, seen, 5, rng)
    weight = [1.]
    for _ in range(100):
        weight = ewc_update(weight, [weight[0]+1.], [1.], [1.], strength=1.)
    print("retention", {"fixed_budget_reservoir": buffer,
          "ewc_optima_k_0_1_5": [ewc_scalar_optimum(-1, 1, 1, k) for k in (0, 1, 5)],
          "ewc_sgd_k1": round(weight[0], 8),
          "clone_terms": clone_loss([0.8, 0.2], [0.5, 0.5], 2., 1.),
          "two_step_vtrace": vtrace_target([0, 1], [0, 0, 0], [1, 1])})
```

运行后先核对手算，再增加新旧分布和 buffer 预算

```sh
python lifelong_algorithms_lab.py retention
python lifelong_algorithms_lab.py test
```

脚本检查固定容量、零容量、二次目标的驻点、教师与学生相同时的零克隆损失，以及 V-trace 的目标策略零概率边界。完整 CLEAR 还包含序列收集、分布式 actor 与神经网络更新；这些应在机制函数正确后逐层加入。

- 最小研究矩阵：online-only、FIFO replay、reservoir replay、replay+clone、EWC；固定总字节数与每步梯度计算。
- 评价至少三项：当前适应曲线、旧任务保持/重学速度、全生命期累计收益；另记真实回访与诊断评测的区别。
- 当环境永久改变，增加“有意遗忘”的对照：统一历史 replay 是否持续施加过时约束？
- 把任务 ID、教师 checkpoint、离线旧数据访问列为显式资源，不与无需这些信息的方法直接混淆。

<a id="lesson-branches"></a>

## 8 · 数据、参数、功能、模块与快慢学习

| 条线 | 保存对象和更新机制 | 主要代价 |
| --- | --- | --- |
| Replay / CLEAR | 旧样本与旧输出；混合 RL 学习及功能约束 | 观测存储、旧分布偏差、多步行为校正 |
| EWC / online EWC | 旧参数与重要性；加二次惩罚 | 局部/对角近似、consolidation 调度 |
| Distillation | 旧策略或表示输出；在指定输入上对齐 | 需要有代表性的输入；教师错误也会保留 |
| Progressive / modular | 把新知识放在新或路由模块中，限制相互干扰 | 模块增长、任务识别与路由、知识复用成本 |
| Progress & Compress | 快的 active column 学习，再蒸馏进慢的 knowledge base | 双系统预算、压缩时间、压缩时旧知识保护 |

快慢系统的本质是时间尺度分离：快速部分吸收近期证据，慢速部分整合可复用信息。若压缩只在新状态上进行，旧功能可能没有监督；如果每次任务都增长模块，就必须把无限寿命下的容量问题列出来。认知科学的互补学习系统可提供设计动机，但动机并不能替代算法和实验的因果证据。

一个可检验的研究问题是：在无任务 ID 的循环环境中，总内存固定为 10 MB，将一部分观测存储预算改为保存旧 logits，是否改善全程收益与回访表现？相同原则也适用于 DRAGO：生成器、旧模型和少量真实 replay，应在同一存储预算下比较。

<a id="research-upgd-utility"></a>

## UPGD：保留与可塑性耦合在同一更新中

固定重要参数可以保护旧功能，却可能阻碍必要适应；无选择地扰动又可能破坏仍有用的功能。UPGD 根据近期效用，让一些方向少变化，另一些方向更容易改变。效用的干预方式与数据范围是定义的一部分，不能直接叫作永久知识重要性。

$$
U_i=L(w-w_ie_i)-L(w)\approx-w_i\frac{\partial L}{\partial w_i}+\frac12w_i^2\frac{\partial^2L}{\partial w_i^2}
$$

U 是将第 i 个权重置零造成的损失变化。右侧为局部二阶 Taylor 近似；大幅移除时可失准，特征级与权重级干预也不相同。

取 $L(w)=(w-1)^2/2$。在 $w=1$，精确移除效用为 0.5，一阶项却为零，二阶项补足 0.5。在 $w=-1$，精确效用为 −1.5，一阶项为 −2，二阶近似为 −1.5。负效用只说明这个分布上移除有益，不说明未来永远无用。

$$
\bar U_{t,i}=\beta\bar U_{t-1,i}+(1-\beta)U_{t,i},\qquad w_{t+1,i}=(1-\alpha\lambda)w_{t,i}-2\alpha(1-\widetilde U_{t,i})(g_{t,i}+\xi_{t,i})
$$

对应作者 README 的全局缩放、一阶、权重级短实现骨架。Ũ 经偏差修正、全局参照与 sigmoid 得到，ξ 是指定尺度扰动。论文另有效用/特征变体，不能混用。

同一保护系数同时乘在梯度和噪声上。因此高效用坐标不仅少受随机破坏，也可能更慢完成新任务更新；低效用坐标可能承担更大优化方差。weight decay 是第三条独立变化路径。只消融噪声不能解释全部效果。

| 对照 | 检验 |
| --- | --- |
| 统一乘子 vs 效用乘子 | 总体降低学习率还是方向选择？ |
| 只调梯度、只调噪声、同时调节 | 功能保护与参数多样性的不同作用。 |
| 精确效用、一阶、二阶 | 近似在漂移或饱和处何时失真？ |
| 近期分布、旧 probe、真实回访 | 近期有用与历史稀有重要知识是否相同？ |

**算法：从反事实效用到持续控制的验证路径**

1. 两坐标回归：精确计算逐坐标置零效用，固定相同噪声序列
1. A→B→A 数据流：不给切换标记，不清零效用统计
1. 报告当前误差、新目标速度、A 回访与整个过程损失
1. 再接入 PPO，明确 rollout 长度与重复更新预算

作者 mohmdelsayed/upgd 的 experiments/weight_utility.py 对应效用实验，core/run/rl 对应 PPO。需要核对缩放分母退化、梯度与参数是否来自同一步、统计是否跨任务保留。主体流式监督结果与 PPO 证据分别报告；换用在线 optimizer 不能使 PPO 自动满足严格一次使用经验协议。

<a id="lesson-check"></a>

## 9 · 习题与下一步

- 旧任务分数下降、新任务学习很快，说明什么？可能是遗忘或必要适应，不足以判断可塑性丧失。
- Reservoir 是否让每个任务等权？否，它让各时刻样本等概率；较长任务会占更多份额。
- 已知 μ(a|s) 能否修正所有历史偏差？不能；动作比率不修正环境转移本身已经改变，也不恢复从未保存的状态。
- 为什么保存 KL 教师也要计内存？每条样本的 logits/value 与教师 checkpoint 都是真实资源。
- 如果 EWC 的 F 全零会怎样？二次约束消失，退化为当前任务学习；若 κ 极大，则会压制必要适应。

动手题：用同一组 A→B→A 样本分别做 FIFO 和 reservoir，比较 buffer 内各阶段比例；再把 B 的长度扩大十倍，检查所谓“平衡历史”是否仍成立。先用确定性样本查清保存分布，再把真实 RL 策略反馈加进去。

## 本章的实验设计

冻结副本测试保留能力。保持更新的副本测试再适应。副本中的反馈不能写回主运行。

设定：使用 A→B→A，并在回访前复制隔离的评测分支。一个分支冻结，另一个允许新增学习。加入未见 C 以防“少遗忘但学不动”。

- A 的初次习得达到预定非平凡标准。
- 冻结回访样本不进入主 buffer 和归一化。
- 保存、回放与模型增长计入固定预算。

对照：无保留机制、固定容量 replay；相同存储与更新预算的 EWC/蒸馏；独立任务顺序与相同信息权限

记录：冻结保留与允许重学后的恢复曲线；B/C 获取速度和完整生命期回报；缓存、参数、蒸馏与梯度成本

[具体测试规程](https://yingwen.io/zh/continual-rl/experiments/handbook/#handbook-continual)

## 学习与研究衔接

记住旧任务和快速学习新任务可能冲突。冻结诊断与保持更新的再适应实验分别回答不同问题。

[分册导读](https://yingwen.io/zh/continual-rl/start/continual-rl/) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-retention) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=retention) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=retention)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

给定模型可研究怎样规划；模型也在学习时，规划会选择性地查询误差，并改变以后的数据。Dreamer、STOMP 和 DRAGO 分别研究想象控制、随机时长行为模型和旧知识保留。新的比较应固定规划查询与总预算，检验哪些后果误差真正改变选择，哪些维护值得继续。

- [Knowledge Retention in Continual Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-drago-model-retention)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-upgd-utility-protection)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [Loss of plasticity in deep continual learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-backpropagation)
- [Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn](https://yingwen.io/zh/continual-rl/research/#recent-c-chain-churn)
- [Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline](https://yingwen.io/zh/continual-rl/research/#recent-reset-and-distill)
- [Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-fame-fast-meta-learners)
- [Knowledge Retention in Continual Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-drago-model-retention)
- [Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-vla-simple-recipe)
- [Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-upgd-utility-protection)
- [Parseval Regularization for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-parseval-continual-geometry)
- [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-trac-online-regularization)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-fame-fast-meta-learners)
- [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-trac-online-regularization)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-vla-simple-recipe)

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

原文明确说明，其效用主要考虑当前数据，CBP 并不解决遗忘。对新数据持续学得动与旧功能仍被保留是不同结果；有限长序列也不保证无限生命中的任意适应。替换率、效用和成熟度仍需选择。

#### 阅读与实验

在同一替换预算下消融成熟度、效用与随机替换，先检验新目标学习。随后让旧情境返回，测被回收功能的损失；若再用旧功能代价约束回收，应作为新增机制检验，不能把收益归给原始 CBP。

#### 原文与相关入口

- [Nature 原文](https://doi.org/10.1038/s41586-024-07711-7)：长期可塑性实验与 continual backpropagation。
- [作者代码](https://github.com/shibhansh/loss-of-plasticity)：关注 lop/algos/gnt.py 及替换时的优化器状态。

#### 作者代码

[论文作者公开的实验实现。](https://github.com/shibhansh/loss-of-plasticity)

论文任务、持续反向传播与 generate-and-test。

### Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn

Hongyao Tang, Johan Obando-Ceron, Pablo Samuel Castro, Aaron Courville, Glen Berseth

ICML 2025 · 2025 · 直接研究持续学习

#### 研究问题

一次局部更新为什么会在其他输入上引发大幅预测变化，并损害后续学习？

#### 关键机制

C-CHAIN 抑制相对于近期参考网络的函数输出变化，降低一次更新在其他样本上造成的 churn。论文把该现象与经验神经切线核及学习动力学联系起来。正则化对象是函数变化，不是直接把所有参数锁在旧值附近。

#### 证据

作者在持续 Gym Control、ProcGen、DMC 和 MinAtar 序列中比较，并提供对应环境和算法代码。

#### 条件与限制

近期函数稳定性不等于长期任务知识保留；参考样本和参考网络也占资源。若环境突然发生真实变化，过强抑制输出变化可能延迟必要适应。

#### 阅读与实验

将 churn 按旧分布、新分布分别计算，并同时画适应速度。这样才能区分“减少无关干扰”和“阻止有用改变”。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/tang25g.html)：机制、理论分析与持续实验。
- [作者代码](https://github.com/bluecontra/C-CHAIN)：四类持续环境的基线和 C-CHAIN 对照实现。

#### 作者代码

[作者仓库，README 说明依赖的 TRAC、CleanRL 与 MinAtar 基础实现。](https://github.com/bluecontra/C-CHAIN)

持续控制环境与 C-CHAIN 对照实验。

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

### Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning

Ke Sun, Hongming Zhang, Jun Jin, Chao Gao, Xi Chen, Wulong Liu, Linglong Kong

ICLR 2026 · 2026 · 直接研究持续学习

#### 研究问题

快速学习新任务和整合旧知识，能否由不同学习器承担并以明确目标连接？

#### 关键机制

FAME 的快速学习器适应当前任务，元学习器整合此前知识。论文按旧策略的重要访问分布度量价值或策略变化，再据此构造减少遗忘的整合目标。自适应预热决定如何利用旧知识初始化或约束早期行为，以减少负迁移。

#### 证据

论文分析价值型和策略型版本，并在像素与连续控制任务序列中比较。作者提供官方实现，可追踪快速适应与知识整合两个阶段。

#### 条件与限制

设定要求相同状态与动作空间、已知任务边界以及额外整合计算。这里的 meta learner 主要是知识整合模块，不应因名称就当作通过长期回报反向求导的在线元梯度算法。脑机制类比也不是神经科学实验证据。

#### 阅读与实验

分别报告新任务前向迁移、旧任务保留和两个学习阶段的计算量。改变任务相似性，检验自适应预热是否确实避免有害旧知识。

#### 原文与相关入口

- [ICLR 2026 原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2230ffcd5da10015ce0c6ce588fc2936-Abstract-Conference.html)：任务边界假设、遗忘度量与快慢知识机制。
- [FAME 官方实现](https://github.com/datake/FAME)：论文链接的快速学习与知识整合代码。

#### 作者代码

[论文与仓库均注明为官方实现。](https://github.com/datake/FAME)

FAME 的价值型、策略型持续学习实验。

### Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline

Hongjoon Ahn, Jinu Hyeon, Youngmin Oh, Bosun Hwang, Taesup Moon

ICLR 2025 · 2025 · 直接研究持续学习

#### 研究问题

一个网络还能拟合新目标，为什么先前训练仍可能让它在新任务上学得更慢？

#### 关键机制

论文把任务之间的负迁移与一般可塑性损失区分开。Reset & Distill 在新任务开始时重置在线 actor 和 critic，避免旧初始化阻碍学习；随后离线蒸馏当前策略与旧专家的动作分布以整合知识。适应和保留通过不同过程实现。

#### 证据

作者在控制与游戏任务中分析负迁移，并在长 MetaWorld 序列上检验该基线。原文直接提供实现地址。

#### 条件与限制

任务边界、在线网络重置、旧专家和离线蒸馏都需要资源。它不能直接当作无边界、不能重置、禁止回放的单次生命方案。

#### 阅读与实验

除了与连续微调比较，还要与同等预算的从头训练比较。若新任务表现低于从头训练，先检查负迁移，再判断是否属于单纯容量损失。

#### 原文与相关入口

- [ICLR 2025 原文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ba9e3d60610f3525717665966d86e0cd-Abstract-Conference.html)：负迁移诊断、Reset & Distill 机制与边界。
- [原文代码入口](https://github.com/hongjoon0805/Reset-Distill)：论文首页提供的作者实现。

#### 作者代码

[ICLR 正式论文首页明确链接的代码。](https://github.com/hongjoon0805/Reset-Distill)

Reset & Distill 以及任务序列实验。

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

### Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning

Aneesh Muppidi, Zhiyu Zhang, Heng Yang

NeurIPS 2024 · 2024 · 直接研究持续学习

#### 研究问题

未知环境变化时间和速度时，怎样在线决定参数应离参考初始化多远？

#### 关键机制

TRAC 在基础优化器外维护一组具有不同遗忘时间尺度的一维 tuner，根据梯度与参考方向的内积调整参数位移尺度。它通过数据驱动的缩放联系到正则化，而不是对未来任务回报进行长窗口元梯度反传。

#### 证据

作者在 Procgen、Atari 与 Gym Control 变化序列中比较适应与可塑性，并分析在线凸优化对该设计的启发。

#### 条件与限制

凸在线优化中的遗憾理论不等于非凸、策略依赖采样的深度 RL 收敛定理。“parameter-free”不表示没有基础学习率、初始化、时间尺度网格、warm-start 或协议选择。

#### 阅读与实验

记录 tuner 尺度、距参考点的位移、旧分布干扰与变化后适应。用相同基础优化器比较固定尺度、单时间尺度和多时间尺度。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/5b76d77e7095c6480ed827b85f0c2878-Paper-Conference.pdf)：Algorithm 1–2、正则化联系与持续实验。
- [作者论文 v3](https://arxiv.org/html/2405.16642v3)：区分凸理论、RL 经验结果与初期表现限制。

#### 作者代码

[作者项目页与仓库均明确标为官方实现。](https://github.com/ComputationalRobotics/TRAC)

trac.py、PyTorch/JAX optimizer 包与控制/视觉实验。

### Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning

Mohamed Elsayed, A. Rupam Mahmood

ICLR 2024 · 2024 · 支持方法与理论

#### 研究问题

同一网络里，哪些方向应当保护，哪些方向应当获得更强的新学习与扰动？

#### 关键机制

UPGD 用移除权重或特征的反事实损失变化定义效用，并以 Taylor 近似在线估计。平滑、缩放后的效用同时调制梯度与随机扰动，让近期高效用方向变化较小、低效用方向更活跃。

#### 证据

主体证据包括未知边界的非平稳流式监督任务；另外包含长时间 PPO 实验。两类证据应分别理解，不能把监督任务数量写成 RL 任务覆盖。

#### 条件与限制

近期分布上的效用不保证稀有旧知识的重要性；一阶和二阶近似、权重级和特征级版本不同。PPO 仍使用 rollout 与重复更新，不因 optimizer 在线就成为严格流式 RL。

#### 阅读与实验

用可精确消融的小网络检查效用估计，再拆开保护梯度、保护噪声和 weight decay 三种作用；独立报告新学习与旧功能。

#### 原文与相关入口

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/file/8e5f0591943d8dae5702af12dcdcd2f6-Paper-Conference.pdf)：效用定义、近似、不同 UPGD 变体与 PPO 实验。
- [作者预印本](https://arxiv.org/abs/2404.00781)：流式监督协议与 RL 证据范围。

#### 作者代码

[论文首页明确链接的作者仓库；README 的短实现是一个指定变体。](https://github.com/mohmdelsayed/upgd)

权重/特征效用实验、流式任务及 PPO 实现。

### Parseval Regularization for Continual Reinforcement Learning

Wesley Chung, Lynn Cherif, David Meger, Doina Precup

NeurIPS 2024 · 2024 · 直接研究持续学习

#### 研究问题

仅在初始化时保持良好的权重几何，是否足以让很晚出现的新任务仍容易学习？

#### 关键机制

在选定隐藏层加入 $\lambda\|WW^\top-sI\|_F^2$，持续约束行向量的范数与角度；输出层及额外尺度设计保留表达能力。它维护学习的几何条件，并不直接保存旧任务标签或预测。

#### 证据

作者在 Gridworld、CARL、MetaWorld 任务序列中检验，并拆分范数与角度约束。稳定秩、Jacobian 与熵属于诊断量，不单独构成可塑性或保留的因果证明。

#### 条件与限制

约束会限制函数类；输出行数大于输入维度时，全部行正交不可实现。非线性门控仍能切断梯度。有限任务序列的结果不保证无限生命内有效，也不是无任务信息的万能机制。

#### 阅读与实验

同预算比较仅初始化正交、持续范数约束、持续角度约束和完整正则；同时记录新目标拟合、真实回报、旧功能与额外计算。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/e6df4efa20adf8ef9acb80e94072a429-Paper-Conference.pdf)：目标函数、容量限制、角度/范数消融及持续任务协议。
- [作者版本记录](https://arxiv.org/abs/2412.07224)：正式会议年份为 2024。

#### 作者代码

[仓库明确标为 NeurIPS 2024 官方实现。](https://github.com/wechu/parseval_reg)

PPO、任务序列、正则化与网络结构消融。


<a id="chapter-code"></a>

## 下载与运行

存储策略、目标项和 trace 数值实验；完整 CLEAR 训练还需要神经网络、序列收集与环境配置。

[下载 lifelong_algorithms_lab.py](https://yingwen.io/zh/continual-rl/download/lifelong_algorithms_lab.py)

```sh
python lifelong_algorithms_lab.py retention
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction · §9.3–9.4](http://incompleteideas.net/book/the-book-2nd.html)：固定目标的梯度更新；线性函数逼近中的“线性”指相对于待学习权重。本章双线性参数算例为原创。

- [Lyle et al. · Understanding Plasticity in Neural Networks · §2–3](https://proceedings.mlr.press/v202/lyle23b.html)：区分旧任务遗忘与后续任务学习能力；明确探针目标、优化器和固定更新预算。

- [Rolnick et al. · Experience Replay for Continual Learning (CLEAR)](https://papers.nips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html)：新旧经验混合、off-policy 学习与行为/价值克隆的原论文。

- [Espeholt et al. · IMPALA](https://proceedings.mlr.press/v80/espeholt18a.html)：V-trace 原始推导及裁剪系数所控制的偏差、方差与目标策略。

- [Kirkpatrick et al. · Overcoming catastrophic forgetting in neural networks](https://doi.org/10.1073/pnas.1611835114)：EWC 的概率解释与 Fisher 近似；注意不同 Fisher 估计器的语义。

- [Schwarz et al. · Progress & Compress](https://proceedings.mlr.press/v80/schwarz18a.html)：固定架构大小的 active/knowledge 双系统与 online EWC 压缩。

- [Fu et al. · Knowledge Retention in Continual Model-Based RL · ICML 2025](https://proceedings.mlr.press/v267/fu25f.html)：DRAGO 的生成回放与真实记忆恢复；共享动力学、已知任务边界，当前任务数据仍可重放。

- [DRAGO 作者实现](https://github.com/YixiangSun/drago)：对应生成器、冻结旧动力学模型和任务/回访行为；与仅用 reservoir 的模型自由方法有不同预算。

- [AGI-Labs continual_rl](https://github.com/AGI-Labs/continual_rl)：可扩展 CRL 比较框架，包含统一接口下的方法复现；实现来源与原论文作者仓库分别标识。

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/5b76d77e7095c6480ed827b85f0c2878-Paper-Conference.pdf)：Algorithm 1–2、正则化联系与持续实验。

- [作者论文 v3](https://arxiv.org/html/2405.16642v3)：区分凸理论、RL 经验结果与初期表现限制。

- [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning · 作者实现](https://github.com/ComputationalRobotics/TRAC)：trac.py、PyTorch/JAX optimizer 包与控制/视觉实验。 作者项目页与仓库均明确标为官方实现。

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/file/8e5f0591943d8dae5702af12dcdcd2f6-Paper-Conference.pdf)：效用定义、近似、不同 UPGD 变体与 PPO 实验。

- [作者预印本](https://arxiv.org/abs/2404.00781)：流式监督协议与 RL 证据范围。

- [Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning · 作者实现](https://github.com/mohmdelsayed/upgd)：权重/特征效用实验、流式任务及 PPO 实现。 论文首页明确链接的作者仓库；README 的短实现是一个指定变体。

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/e6df4efa20adf8ef9acb80e94072a429-Paper-Conference.pdf)：目标函数、容量限制、角度/范数消融及持续任务协议。

- [作者版本记录](https://arxiv.org/abs/2412.07224)：正式会议年份为 2024。

- [Parseval Regularization for Continual Reinforcement Learning · 作者实现](https://github.com/wechu/parseval_reg)：PPO、任务序列、正则化与网络结构消融。 仓库明确标为 NeurIPS 2024 官方实现。
