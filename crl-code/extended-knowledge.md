# 扩展知识：从单步公式到独立、可运行的机制实验

这一组实验沿着四个问题展开：探索什么；怎样提出目标；怎样把动作组织成选项；怎样学习奖励与选项模型。每个主题都有问题设定、更新公式、学习循环和对照实验。先分清学习对象，再比较具体更新。

这里实现的是限定子机制。表格Option-Critic、表格最大熵技能发现、16轨迹枚举IRL和三状态中心化都是小型特例，不能代替原论文完整系统。消融基线复用对应实验，并明确关闭哪一个组件；它们不是新的完整算法。

## 运行与结果解释

在教程根目录使用Python 3.10+，不需第三方依赖：

```bash
python3 -m unittest discover -s tests -p test_extended_knowledge.py -v
python3 -m implementations.extended_knowledge.her_replay \
  --steps 1200 --seeds 11 23 --out results/her-component-v1
python3 implementations/runtime.py option_critic option_critic_fixed_beta \
  --steps 1200 --seeds 11 23 --out results/option-critic-component-v1
```

输出目录必须尚不存在。模块命令自动包含声明的baseline；通用runtime命令显式列出candidate与baseline。输出保存阶段事件、每seed CSV/config、源码快照/摘要、完整人口manifest和实测曲线。共同纵轴必须同时匹配family、task、metric、unit、higher_better和budget。不要把标签、梯度更新、恢复trial、模型备份当作环境交互；也不要将不同任务的1200steps直接横比。

也可直接执行文件，例如 `python3 implementations/extended_knowledge/her_replay.py --steps 1200 --seeds 11 23 --out results/her-direct-v1`。全部30个入口在不设置PYTHONPATH的仓库外工作目录接受独立文件调用；真实子进程测试覆盖数值工具、跨文件评价/模型/feature和baseline共享训练。baseline的 `ablation_of` 标明主文件，baseline页也会同时运行其主方法作为对照。

每个运行只在step1、每20steps及真实预算末记录，末条严格等于steps。日志phase区分机制训练、模型规划、偏好拟合、奖励推断和元训练。环境任务的评价从当前参数冻结计算，不更新策略/预测器/分类器；RND覆盖与恢复净奖励是明确的在线指标，不伪称冻结回报。

外层只比较META定义的最后指标。算法内评价有时精确遍历有限状态/轨迹，它不是额外训练，也不是跨任务效能证据。RND、option、HER的额外预测或回放备份均与环境交互分开标注；同环境预算不代表同计算成本。停止状态本身的奖励建模与辅助stopping bonus保持分离。

## 入口与真正完成的层次

| 主题与ID | 对照ID | 实现层次与时钟 |
|---|---|---|
| RND `rnd_exploration` | `rnd_no_bonus` | 固定随机MLP目标、可训练MLP预测器、内奖Q控制；环境转移 |
| learning progress `learning_progress` | `uniform_curriculum` | 三任务实际回归训练与双窗调度；训练样本 |
| recovery `recovery_filter` | `recovery_unfiltered` | 从恢复结果估计概率再过滤；reset trials |
| UVFA `uvfa_shared` | `uvfa_tabular` | 状态/目标共同输入的共享MLP半梯度TD；环境转移 |
| HER `her_replay` | `goal_replay_plain` | 完整事实轨迹、future goal重标、表格回放控制；环境转移 |
| stopping `subtask_stopping` | `subtask_no_bonus` | STOMP奖励尊重子任务TD、内部策略与停止规则；环境转移 |
| intra-option `intra_option_q` | `option_on_policy_only` | 固定随机option的多option片段价值学习；环境转移 |
| Option-Critic `option_critic` | `option_critic_fixed_beta` | 表格内部actor、action/option critic、arrival termination联合更新；环境转移 |
| DIAYN `diayn_tabular` | `diayn_no_discriminator_reward` | 表格soft actor-critic与状态分类器联合训练；环境转移 |
| option model `option_model` | `option_model_monte_carlo` | 真实reward及折扣endpoint模型TD；环境转移 |
| option VI `option_value_iteration` | `primitive_value_iteration` | 已知有限option模型的真实逐状态Bellman规划；模型备份 |
| preference `preference_reward` | `preference_single_feature` | 随机片段偏好标签的线性BT奖励训练；偏好标签 |
| MaxEnt IRL `maxent_irl` | `maxent_single_feature` | 确定性有限DAG的全轨迹配分函数与奖励学习；梯度更新 |
| intrinsic meta `intrinsic_meta_gradient` | `intrinsic_fixed_reward` | 展开真实一次策略更新的解析外奖元梯度；内外更新 |
| Reward Centering `reward_centering` | `reward_uncentered` | TD驱动参照的continuing折扣预测；环境转移 |

## 1. RND：学习的是预测器，不是随机目标

[Burda等原论文](https://arxiv.org/abs/1810.12894)与[作者工程](https://github.com/openai/random-network-distillation)对应 `rnd_exploration.py`。

设置为12格反射链，无外奖，Q控制以预测误差探索。两个独立随机初始化的12→12→3网络采用tanh隐藏层；目标网络从不更新。先执行真实转移，再计算
`b(s')=mean((predictor(s')-target(s'))²)`，此误差必须在预测器SGD前读取。随后只更新预测器，Q按内奖和下一状态最大Q进行半梯度备份。记录已访状态比例、即时/累计新奇误差和预测器更新数。

对照保留同样的预测器训练，只不给Q内奖；由此分离“会预测”与“驱动探索”。单状态重复拟合应使误差下降，但不能要求所有未见状态总是高误差：共享网络会泛化，预测误差也不是校准不确定性。缺口是PPO/Atari、双价值头、观测/奖励归一化与原工程重训。

## 2. Learning progress：学习进展不等于奖励或误差大小

[Oudeyer等IAC原论文](https://doi.org/10.1109/TEVC.2006.890271)提供学习进展动机，本文件只做固定三个任务的调度组件。

各任务真的训练独立斜率预测器，第三任务含不可约高斯噪声。完成两个各10条已见误差窗口后，调度分数为两窗均值差的绝对值；未满窗口先收集数据，之后以epsilon .2维持全任务支持。调度器不能读取生成斜率或未来误差。SGD后以均匀输入的解析期望MSE冻结评价，包括噪声方差。

均匀任务baseline使用相同学习器和噪声过程。误差上升也会产生绝对progress，噪声可能吸引调度器，单次优势不是课程有效性证据。缺口是IAC自适应区域划分、复杂目标生成与机器人闭环。

## 3. Recovery filter：学到的概率不构成安全证明

[Leave No Trace原论文](https://arxiv.org/abs/1711.06782)对应“前进行为须可恢复”的机制动机，而不是声称完整双策略实现。

三个前进行为的恢复成功率只在环境中用于采样。前30trial收集全部行为的真实恢复结果，并计入失败和预算。之后以Beta(1,1)先验平滑的经验成功率过滤提议行为；若没有达到.75阈值的行为，只选最高估计项，不把它叫“安全”。每次选择都执行真实随机恢复trial，更新估计。在线净奖励包含forward收益减失败成本；另记失败率与干预次数。

未过滤baseline保留相同forward价值学习与恢复结果采集。缺口是训练reset policy、reset critic、真实状态空间恢复区和置信安全证书。坏初始化、未知状态、有限样本和恢复分布漂移都可能失效。

## 4. UVFA：共享参数是真实目标条件逼近

[Schaul等UVFA原文](https://proceedings.mlr.press/v37/schaul15.html)提出状态与目标同时泛化。此处共享MLP输入为归一化的s、g、s-g和距离，输出两个动作Q。真实goal-conditioned TD目标为
`r + gamma*(1-done)*max Q(s',g,a)`，只对当前Q求导；到达goal或物理终止不bootstrap，12步训练截断仍bootstrap。

训练随机起点与目标；评价遍历42个不同起点/目标，冻结贪心策略最多走12步。对照逐目标表格Q使用同任务/环境预算，但参数化和学习率不同，不能当作同计算量模型选择结论。缺口是原论文低秩embedding预训练、未见目标测试和大状态泛化；当前评价目标都可能被训练访问。

## 5. HER：重提问题，不改变发生过的事实

[Andrychowicz等HER原文](https://arxiv.org/abs/1707.01495)对应完整future replay教学实例。每条事实保存s、a、s'与physical terminal。在完成的真实轨迹中，对每个时刻从该时刻及之后的已达后继状态随机取goal；仅替换goal，重新计算奖励与成功done。旧目标成功不等于物理终止；物理终止则在任何重标目标下保留。环境动力学必须与goal无关。

原目标及重标样本进入回放，每真实交互额外两次回放备份。对照有相同回放量但不重标。当前预算末未完成轨迹不伪造未来goal，已经进行的在线TD记录保留。缺口是神经UVFA+DDPG、机器人连续控制及随机动力学下的HER偏差处理。

## 6. Subtask stopping：终止option不等于终止环境

[STOMP原文](https://arxiv.org/abs/2202.03466)的到达停止约定为
`r + beta(s')*z(s') + gamma*(1-beta(s'))*V(s')`。
stopping bonus z项不再额外乘gamma。六步链的真实环境奖励始终保留；状态3给辅助bonus .8，内部动作价值按真实转移TD学习，停止规则由z(s')与继续价值比较。均匀可重置起点的一步模拟转移提供全支持数据，不冒充自然自主episode采样。

冻结评价执行每个条件自身学到的停止规则，再按同一客观子任务目标计分。无bonus基线不能在评价时凭空换上候选的终止策略。物理终点强制停止，主任务reward model不会混入z。缺口是完整STOMP子任务生成、option模型学习和主任务规划闭环。

## 7. Intra-option Q：片段可以更新多个option

[Sutton、Precup、Singh原文](https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/sutton%2Bal-1999.pdf)对应固定option价值组件。两option在同一状态均有左右动作支持，行为真正执行一个active option；每转移用
`rho = pi_option(a|s)/behavior(a|s)`
更新所有option。到达值为继续旧option与终止后重新选择的混合。

所有右端都使用旧快照，尤其自环不能让先更新option影响另一个目标。重新选择受到initiation mask限制，continuation不受该mask限制；物理terminal到达值为零。on-policy baseline只更新实际执行option，不作额外多option更新。冻结评价精确计算call-and-return策略回报。缺口是内部策略与终止学习、大状态逼近和收敛的递减学习率条件。

## 8. Option-Critic：是联合训练，但只在小表格里

[Bacon等Option-Critic原文](https://arxiv.org/abs/1609.05140)给出内部策略与termination梯度。此处每一步真的更新action critic、option critic、softmax内部动作策略及sigmoid termination。actor方向为
`(Q_U-baseline)*(one_hot(action)-pi)`；
arrival termination方向为
`-beta*(1-beta)*(Q_option-V_select)`。

训练时option选择epsilon .2，因此continuation和termination中的选择价值采用同一epsilon-soft期望；这是一种明确的on-policy小型变体，不暗称论文的greedy off-policy目标。actor与beta读取旧critic，物理terminal无bootstrap也不更新beta。冻结报告以贪心option选择、随机内部策略作精确有限状态求值。

固定beta=.5 baseline仍训练critic/actor。缺口是神经网络、原Atari/房间实验、deliberation cost、额外正则与长期稳定性；本例允许选项相似或退化，不强制“发现技能”。

## 9. DIAYN：分类器与技能策略都必须真的学习

[Eysenbach等DIAYN原文](https://arxiv.org/abs/1802.06070)使用固定技能先验、状态分类器与最大熵技能策略。这里两技能等先验，episode开始抽z并保持8步；策略/critic包含时间索引，因而8步结束是真正有限时域任务结束而不是错误地截断无限价值。分类器只见position；内奖读取更新前的
`log q_phi(z|s') - log p(z)`。

每交互更新分类器CE、soft Q，以及最大化`sum pi*(Q-temperature*log pi)`的精确表格actor梯度。无内奖baseline仍训练分类器和动作熵策略。冻结评价精确传播每技能各时刻状态占用，从真实joint distribution计算MI，不使用训练分类器准确率冒充技能多样性。

缺口是神经SAC、重放/target networks、连续技能控制和下游迁移；本例是有限表格目标实例，不是原DIAYN全工程。低MI或塌缩就是实测结果，不人为分离技能或画高信息曲线。

## 10. Option reward/endpoint model：时间折扣不应被归一化掉

同[Sutton等options原文](https://people.eecs.berkeley.edu/~russell/classes/cs294/f05/papers/sutton%2Bal-1999.pdf)：
`R_o(s)=E sum gamma^k r_k`，
`P_o_gamma(s,j)=E gamma^tau 1(endpoint=j)`。
固定随机option的每真实primitive转移进行TD模型学习，端点分布目标为
`gamma*(beta*one_hot(s')+(1-beta)*P(s'))`。
物理terminal强制beta=1；显式terminal端点质量保留，规划时该端点V=0。

评价对照精确解的reward/endpoint联合RMSE；真解只用于诊断，不用于学习target。完整option MC baseline只在真正option结束后更新起点，不把预算末未结束前缀当完成rollout。P行和一般小于1，反映时长折扣，不是漏掉概率。缺口是学习内部option、神经模型和模型误差进入主任务规划。

## 11. Option Value Iteration：已经含gamma的模型只乘一次

同options原文，Bellman备份为
`V(s)=max_o[R_o(s)+sum_j P_o_gamma(s,j)*V(j)]`。
已知小型option模型与primitive模型进入真正的随机逐状态规划更新。每step一个状态备份；另列primitive/option候选项备份数，不能把它说成相同总算力。精确最优右移价值仅用于冻结误差。

primitive-only baseline有同样状态备份时钟。两个都可能最终达到零误差：option不必改善这个小任务。模型已知/精确求解是预处理，不将它冒充在线环境模型训练；缺口是学习模型、未知MDP和大尺度规划。

## 12. Bradley–Terry preference reward：真实标签拟合而非手算一次

[Christiano等偏好RL原文](https://arxiv.org/abs/1706.03741)中的片段比较以线性回报和实现：
`P(A>B)=sigmoid(w*(Phi_A-Phi_B))`。
真实随机比较标签驱动cross-entropy SGD；两个features为右动作数和转向数。生成权重只用于标签环境及冻结期望CE，不提供给训练更新。单特征baseline真的训练受限奖励模型。

冻结指标含标签噪声，不应强求降到零；偏好不能辨识共同reward offset。缺口是人类标注、主动查询、神经奖励拟合和奖励模型—策略优化循环。

## 13. MaxEnt IRL：全局枚举准确，但仍是小特例

[Ziebart等原论文与修正版入口](https://www.cs.cmu.edu/~bziebart/publications/maximum-entropy-inverse-reinforcement-learning.html)对应16条合法轨迹的确定性四层DAG。采样64条专家轨迹作为固定训练数据，按全局
`P_theta(tau)=exp(theta*Phi_tau)/Z_theta`
拟合；梯度为模型feature期望减经验feature期望。配分函数使用全部完整合法path，而不是错误地对每步分别做无未来价值的softmax。

每step一次真实全批梯度更新；64演示的生成成本另外明确记录。冻结指标是真实专家分布下的轨迹CE，含有限演示误差。单特征baseline限定reward容量。缺口是大图动态规划、随机动力学maximum causal entropy、deep IRL和在线控制；枚举准确不代表工程可扩展或原驾驶实验复现。

## 14. Intrinsic reward meta-gradient：外层目标必须是外奖

[Zheng等原论文](https://arxiv.org/abs/1804.06459)对应可微内奖设计组件。二动作bandit中，logit theta用外奖+内奖进行一次真实策略更新，
`theta'=theta+alpha*(1+eta)*p*(1-p)`；
outer根据更新后的外在回报`J= sigmoid(theta')`，通过该更新求`dJ/deta`。内层theta持续推进，eta从0学习；固定eta=0 baseline就是只用外奖，并非刻意给坏内奖的弱对照。

有限差分核验链式导数。此处用精确二动作期望梯度，不假称采样环境interaction；budget是meta_updates。缺口是原A2C/PPO、长时域、多步unroll、复杂状态分布/二阶项及实际延迟奖励问题。

## 15. Reward Centering：明确2024论文的value-based版本

[Naik等Reward Centering原文](https://arxiv.org/abs/2405.09999)的TD驱动参照（论文Eq7–8）在on-policy rho=1时：
`delta=r-c+gamma*V(s')-V(s)`，
`V(s)+=alpha*delta`，
`c+=eta*alpha*delta`。
本例三状态continuing环，无任何episode/done，gamma=.99。报告的是重构原折扣值`V+c/(1-gamma)`的RMSE，另外记中心化值尺度和参照c。不能只展示数值变小却把价值目标换掉。

原value-based参照不是行为奖励running mean，且gamma<1时参照与初始值存在耦合；表格不变量`c-eta*sum(V)`可用于检查更新时间，但不是深网守恒定律。后续[2025 Bellman Error Centering论文](https://arxiv.org/abs/2502.03104)对命名/概念作了再分析；本文件明确保留原2024版本，不混称simple reward centering或Differential TD。

无中心化baseline采用相同折扣TD和原值RMSE。缺口是off-policy逼近、深度控制、目标策略reward-rate估计及平均奖励控制；三状态解析真值只是评价标准，并没有将真值注入学习。

## 验证与结论边界

`tests/test_extended_knowledge.py`核验MLP全部参数组、BT/MaxEnt/分类器/最大熵actor/Option-Critic actor与termination/内奖unroll的有限差分；核验goal成功/physical terminal/截断、合法future HER事实、旧快照、自环、initiation与continuation、两步option折扣、终点bootstrap及continuing center不变量。全30入口用两seed检查预算末记录、emit/return一致、有限指标、确定性重跑和同任务baseline契约。

开发烟测以seeds11/23、每入口1200真实时钟完成60次运行，全部指标有限；这只核验可运行性。它没有挑选有利seed：RND两个条件最终均覆盖全部12格，intra-option两条件最终冻结回报相同，两个规划器均达到零误差，课程也没有一致显著改善。保留这些结果，不调参数把无差异变成“优势”。最终公开曲线与源码绑定manifest由统一冻结后的runtime批次生成；开发烟测不是原论文结果或统计效能证据。
