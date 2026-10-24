"""一个共享特征怎样改变 DQN 的预测、动作和旧经验标签。

运行：python3 dqn_shared_feature_walkthrough.py
检查：python3 dqn_shared_feature_walkthrough.py --test
仅标准库；输入与唯一一次 SGD 更新均明确给定，不采样环境、不运行随机训练。
两输出共享 h=ReLU(w*x)，Q(L)=v*h，Q(R)=3-v*h；这是一项受限的
机制算例，不是 Atari 网络或完整 DQN 复现。动作 0/1 对应 L/R。
对照 examples/deep_textbook_train.py：bootstrap_target 对应 dqn_update
的 no_grad 区块，frozen_target_step 对应当前动作损失与 optimizer.step，
copy 对应 train_dqn 的 load_state_dict。该完整工程另有 replay、Adam、
真实交互、评估与目标更新时钟，本脚本只隔离一次标签生成和求导。
来源：Sutton & Barto §9.3/§16.5；Mnih et al. (2015), Methods/Algorithm 1；
van Hasselt et al. (2015 arXiv v3 / AAAI 2016), Double DQN section。
"""
import argparse
import copy
import json
import math
import unittest


def q_values(theta, x):
    if len(theta) != 2 or not all(math.isfinite(v) for v in (*theta, x)):
        raise ValueError("theta=(w,v), x 必须有限")
    w, v = theta
    left = v * max(0, w*x)
    return [left, 3-left]


def q_gradient(theta, x, action):
    q_values(theta, x)
    if action not in (0, 1):
        raise ValueError("action 必须为 0 或 1")
    w, v = theta
    sign = 1 if action == 0 else -1
    # 与常用 ReLU 自动微分相同：零点的导数约定为零。
    return [sign*v*x, sign*w*x] if w*x > 0 else [0, 0]


def greedy(values):
    """并列时选 L；正文三个快照都没有并列。"""
    return 0 if values[0] >= values[1] else 1


def bootstrap_target(online, target, experience, gamma=.5, double=False):
    if not 0 <= gamma <= 1:
        raise ValueError("gamma 必须在 [0,1]")
    if not isinstance(experience['terminated'], bool):
        raise ValueError("terminated 必须是 bool")
    reward = experience['reward']
    if not math.isfinite(reward):
        raise ValueError("reward 必须有限")
    if experience['terminated']:
        return reward
    xp = experience['nextX']
    selected = greedy(q_values(online if double else target, xp))
    return reward + gamma*q_values(target, xp)[selected]


def frozen_target_step(theta, experience, label, alpha=.5):
    """label 先算好，再作为常数参与当前预测的求导；返回新参数。"""
    if not math.isfinite(label) or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("label、alpha 必须有限，alpha>=0")
    prediction = q_values(theta, experience['x'])[experience['action']]
    error = prediction-label
    gradient = [error*g for g in q_gradient(theta, experience['x'], experience['action'])]
    return dict(prediction=prediction, label=label, error=error, gradient=gradient,
                theta=[v-alpha*g for v, g in zip(theta, gradient)])


def walkthrough():
    e = dict(x=1, action=0, reward=0, nextX=2, terminated=False)
    initial = [.5, 1]
    target = copy.copy(initial)
    update = frozen_target_step(initial, e, bootstrap_target(initial, target, e))
    final = update['theta']

    def snapshot(phase, online, target):
        return dict(phase=phase, online=list(online), target=list(target),
                    current=q_values(online, 1), successor=q_values(online, 2),
                    targetSuccessor=q_values(target, 2), selected=greedy(q_values(online, 2)),
                    dqn=bootstrap_target(online, target, e),
                    double=bootstrap_target(online, target, e, double=True))

    return dict(experience=e, gamma=.5, alpha=.5, update=update,
                phases=[snapshot('before', initial, target),
                        snapshot('updated', final, target),
                        snapshot('copied', final, copy.copy(final))])


def finite_difference(function, theta, epsilon=1e-6):
    result = []
    for i in range(len(theta)):
        plus, minus = list(theta), list(theta)
        plus[i] += epsilon
        minus[i] -= epsilon
        result.append((function(plus)-function(minus))/(2*epsilon))
    return result


class WalkthroughTests(unittest.TestCase):
    def test_exact_update_and_shared_outputs(self):
        result = walkthrough()
        self.assertEqual(result['update']['gradient'], [-.5, -.25])
        self.assertEqual(result['update']['theta'], [3/4, 9/8])
        self.assertEqual(result['phases'][1]['current'], [27/32, 69/32])
        self.assertEqual(result['phases'][1]['successor'], [27/16, 21/16])

    def test_relabel_without_new_experience(self):
        phases = walkthrough()['phases']
        self.assertEqual([p['dqn'] for p in phases], [1, 1, 27/32])
        self.assertEqual([p['double'] for p in phases], [1, .5, 27/32])
        self.assertEqual([p['selected'] for p in phases], [1, 0, 0])

    def test_separate_parameters_are_not_mutated(self):
        initial = [.5, 1]
        target = initial.copy()
        e = walkthrough()['experience']
        old_e = e.copy()
        frozen_target_step(initial, e, bootstrap_target(initial, target, e))
        self.assertEqual(initial, [.5, 1])
        self.assertEqual(target, [.5, 1])
        self.assertEqual(e, old_e)

    def test_frozen_label_gradient_by_finite_difference(self):
        actual = finite_difference(lambda theta: .5*(q_values(theta, 1)[0]-1)**2, [.5, 1])
        for got, expected in zip(actual, [-.5, -.25]):
            self.assertAlmostEqual(got, expected, places=8)

    def test_full_shared_parameter_residual_has_another_gradient(self):
        actual = finite_difference(
            lambda t: .5*(q_values(t, 1)[0]-.5*max(q_values(t, 2)))**2, [.5, 1])
        for got, expected in zip(actual, [-1, -.5]):
            self.assertAlmostEqual(got, expected, places=8)

    def test_frozen_feature_exercise(self):
        # Freeze w=.5; only v is updated, using its original gradient -1/4.
        theta = [.5, 1-.5*(-.25)]
        self.assertEqual(q_values(theta, 1)[0], 9/16)
        self.assertEqual(q_values(theta, 2), [9/8, 15/8])
        self.assertEqual(greedy(q_values(theta, 2)), 1)

    def test_relu_inactive_and_tie_convention(self):
        self.assertEqual(q_gradient([-1, 2], 1, 0), [0, 0])
        self.assertEqual(q_values([-1, 2], 1), [0, 3])
        self.assertEqual(greedy([1, 1]), 0)

    def test_termination_preserves_reward(self):
        e = dict(walkthrough()['experience'], terminated=True, reward=2)
        for double in (False, True):
            self.assertEqual(bootstrap_target([.5, 1], [.75, 1.125], e, double=double), 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=['dqn_shared_feature_walkthrough.py'])
    else:
        print(json.dumps(walkthrough(), ensure_ascii=False, indent=2))
