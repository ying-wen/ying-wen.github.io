# 从多智能体公式到学习循环

这组实验把反事实优势和单调价值混合接入完整的小型学习任务。它不复现 SMAC，也不把小型任务等同于持续多智能体学习。

## COMA：谁的动作应得到信用？

两智能体分别看到独立比特 \(o_0,o_1\)。各自输出二元动作。共享奖励为

\[
r=.2\mathbf1[a_0=o_0]+.2\mathbf1[a_1=o_1]+.6\mathbf1[a_0=o_0,a_1=o_1].
\]

每次联合动作后回合结束。critic 可以在训练时看到全局状态。actor 在训练与执行时均只输入自身观测。这是集中训练、分散执行的信息约束，不允许把对方观测悄悄放进 actor。

固定另一智能体的已采样动作，边际化自身动作：

\[
b_i(s,a_{-i})=\sum_u\pi_i(u\mid o_i)Q(s,(a_{-i},u)),\qquad
A_i=Q(s,a)-b_i(s,a_{-i}).
\]

对 softmax logits，\(\partial\log\pi_i(a_i)/\partial\theta_{i,k}
=\mathbf1[k=a_i]-\pi_i(k)\)。因此可以直接实现表格梯度上升。由于基线不依赖已采样的 \(a_i\)，其期望 score 项为零。测试枚举全部联合动作，并与回报的有限差分导数比较。这个恒等式不保证估计 critic 没有偏差，也不保证每种条件下都降低方差。

[coma.py](../implementations/multiagent/coma.py) 用采样前的 critic 和保存的旧策略概率计算两个反事实优势，再分别更新 critic 与 actor。若先用本次动作的奖励更新 critic，基线也可能依赖这次采样的动作，上述期望恒等式便不能直接套用。[joint_policy_gradient.py](../implementations/multiagent/joint_policy_gradient.py) 保留任务、critic 与步长，只把基线改为零。评价枚举冻结随机策略的期望奖励。它不是 argmax 策略回报，也不参与训练。

## QMIX：怎样保证联合贪心可分解？

第二个任务使用相同的合作奖励，但每回合持续三步。每个 agent 看到自身比特、时间及身份。局部网络输出两个 action utilities。状态条件 hypernetwork 产生非负混合权重：

\[
Q_{\rm tot}=w_2(s)^\top{\rm ELU}(W_1(s)^\top Q+b_1(s))+b_2(s).
\]

非负权重与单调激活使 \(\partial Q_{\rm tot}/\partial Q_i\geq0\)。固定状态时，将任意局部动作替换为该 agent 的最大 utility 动作，不会降低联合值。因此各自贪心的组合是联合最大值之一。这是结构性质，不是学习正确性保证。

[qmix.py](../implementations/multiagent/qmix.py) 实现局部神经网络、hypernetwork、回放、终止掩码与目标网络。目标为

\[
y=r+.9(1-d)Q_{{\rm tot},\bar\theta}(s',
  (\arg\max_{a'_i}Q_{i,\bar\theta}(o'_i,a'_i))_i).
\]

目标停止梯度。实际第三步才终止。训练预算结束不能额外制造终止。[vdn.py](../implementations/multiagent/vdn.py) 将混合网络替换为局部 utilities 求和，使用相同观测、回放批次和目标同步周期。它没有 QMIX 的额外参数，不能声称计算成本相同。

两者评价都是分散贪心策略在四种 context 上的精确折扣回报。mixer 不参与执行期选动作。原方法的循环状态、多步部分可观测长序列和大规模合作环境不在此例中。

## 测试与读图

- COMA：边际化维度、条件期望优势为零、梯度有限差分、训练与评价可复现。
- QMIX：偏导非负、枚举联合动作验证贪心一致、终止目标不 bootstrap、目标参数无梯度。
- 信息约束：改变另一 agent 的比特不能改变自身网络输入。
- 反例：支付表 \(\begin{bmatrix}1&0\\0&1\end{bmatrix}\) 的局部偏好随对方动作翻转，不能由固定 scalar utility 排序的单调函数精确表示。

曲线改善只能说明该小任务中的学习过程。更低 TD loss 不自动意味着更好的策略；更好的策略也不证明反事实基线普遍降低梯度方差。

## 原始资料

- [COMA 原文](https://arxiv.org/abs/1705.08926)。
- [QMIX 原文](https://proceedings.mlr.press/v80/rashid18a.html)。
- [作者 COMA learner](https://github.com/oxwhirl/pymarl/blob/master/src/learners/coma_learner.py) 与 [QMIX mixer](https://github.com/oxwhirl/pymarl/blob/master/src/modules/mixers/qmix.py)。

以上小型实现独立编写。作者工程入口用于检查算法语义；本例不包含其训练规模或全部工程设计。
