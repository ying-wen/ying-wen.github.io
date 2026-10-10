"""可检查的持续闭环：子任务学习 → option → 多预测模型 → 平均奖励规划。

不是 OaK 完整实现。状态和两个子目标由设计者给定，内部策略从经验学习。
这里集中暴露一个接口问题：改变 option 的策略，也改变了它的模型和 Q 的语义。
所有算法入口调用相同系统；入口文件显式声明唯一改动。
"""
import random
from collections import deque

N = 7
GOALS = (1, 4)
CAP = 6


def greedy(xs):
    return max(range(len(xs)), key=lambda i: xs[i])


def transition(state, action, uniform):
    """环境随机数由环境提供；agent 不访问隐藏的奖励变化时间。"""
    return state if uniform < .1 else (state + (-1 if action == 0 else 1)) % N


def available(state, primitive_only=False):
    # 到达目标时不允许启动对应 option，避免零时长循环。
    return [0, 1] if primitive_only else [0, 1] + [2+i for i,g in enumerate(GOALS) if state != g]


def subtask_update(q, state, action, successor, goal, alpha=.25):
    """真实环境不终止；到达子目标只终止这个预测/控制问题。"""
    reached = successor == goal
    target = (1. if reached else -.01) + (0. if reached else .95*max(q[successor]))
    error = target-q[state][action]
    q[state][action] += alpha*error
    return error


def smdp_error(reward_sum, duration, rate, next_value, current_value):
    if duration < 1:
        raise ValueError('option duration must be positive')
    return reward_sum-rate*duration+next_value-current_value


class OptionModel:
    """三个同策略问题：累计外部奖励、累计单位时钟、终止位置指示量。

    按完整 option 样本更新，相当于这些终止 GVF 的 Monte Carlo 学习。
    不包含 subtask 人工奖励。不把 E[R/τ] 错当成 E[R]/E[τ]。
    固定 7×4 表；近期模式使用固定步长，累计模式使用样本均值。
    """
    def __init__(self, recent=True):
        self.recent = recent
        self.cells = {}
        self.updates = 0

    def update(self, state, option, reward_sum, duration, successor):
        if duration < 1:
            raise ValueError('duration must be positive')
        cell = self.cells.setdefault((state, option), [0, 0., 0., [0.]*N])
        count, old_reward, old_duration, endpoints = cell
        alpha = 1. if count == 0 else (.2 if self.recent else 1/(count+1))
        cell[0] += 1
        cell[1] += alpha*(reward_sum-old_reward)
        cell[2] += alpha*(duration-old_duration)
        for j in range(N):
            endpoints[j] += alpha*(float(j == successor)-endpoints[j])
        self.updates += 1

    def invalidate(self, option):
        for key in [key for key in self.cells if key[1] == option]:
            del self.cells[key]

    def target(self, state, option, q, rate, primitive_only=False):
        _, rewards, duration, endpoint = self.cells[state, option]
        return rewards-rate*duration+sum(p*max(q[s][a] for a in available(s, primitive_only))
                                        for s,p in enumerate(endpoint))


class Agent:
    def __init__(self, seed, planning=4, recent=True, versioned=True, primitive_only=False):
        self.rng = random.Random(seed)
        self.planning = planning
        self.versioned = versioned
        self.primitive_only = primitive_only
        self.q = [[0.]*4 for _ in range(N)]
        self.subq = [[[0., 0.] for _ in range(N)] for _ in GOALS]
        self.policies = [[0]*N for _ in GOALS]
        self.versions = [0]*len(GOALS)
        self.model = OptionModel(recent)
        self.rate = 0.
        self.real_backups = self.model_backups = self.invalidations = 0
        self.subtask_backups = 0
        self.option_finishes = self.cap_finishes = 0
        self.prediction_errors = deque(maxlen=100)

    def choose(self, state):
        choices = available(state, self.primitive_only)
        if self.rng.random() < .2:
            return self.rng.choice(choices)
        maximum = max(self.q[state][a] for a in choices)
        return self.rng.choice([a for a in choices if self.q[state][a] == maximum])

    def primitive(self, state, option):
        if option < 2:
            return option
        # Policy is committed only at macro boundaries, not during execution.
        return self.rng.randrange(2) if self.rng.random() < .1 else self.policies[option-2][state]

    def observe_primitive(self, state, action, successor):
        # Off-policy tabular Q-learning for each *given* subtask.
        if not self.primitive_only:
            for i, goal in enumerate(GOALS):
                subtask_update(self.subq[i], state, action, successor, goal)
                self.subtask_backups += 1

    def commit_policies(self):
        """策略坐标变化时，同时失效对应模型和高层价值，不只清模型。"""
        if self.primitive_only:
            return
        for i, goal in enumerate(GOALS):
            new = [greedy(row) for row in self.subq[i]]
            changed = any(new[s] != self.policies[i][s] for s in range(N) if s != goal)
            if changed:
                self.versions[i] += 1
                self.policies[i] = new
                if self.versioned:
                    self.model.invalidate(i+2)
                    for row in self.q:
                        row[i+2] = 0.
                    self.invalidations += 1

    def complete(self, start, option, reward_sum, duration, successor):
        # This target describes the policy actually executed. Plan before commit.
        nxt = max(self.q[successor][a] for a in available(successor, self.primitive_only))
        error = smdp_error(reward_sum, duration, self.rate, nxt, self.q[start][option])
        self.q[start][option] += .1*error
        self.rate += .002*error  # heuristic constant-step differential rate update
        self.real_backups += 1
        if (start, option) in self.model.cells:
            self.prediction_errors.append(abs(reward_sum-self.model.cells[start, option][1]))
        self.model.update(start, option, reward_sum, duration, successor)
        keys = sorted(self.model.cells)
        for _ in range(self.planning):
            s, o = self.rng.choice(keys)
            target = self.model.target(s, o, self.q, self.rate, self.primitive_only)
            self.q[s][o] += .1*(target-self.q[s][o])
            self.model_backups += 1
        if option >= 2:
            self.option_finishes += 1
            self.cap_finishes += int(duration == CAP and successor != GOALS[option-2])
        self.commit_policies()


def run_system(seed=0, steps=1200, emit=None, *, planning=4, recent=True,
               versioned=True, primitive_only=False):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    env = random.Random(seed+10000)
    agent = Agent(seed, planning, recent, versioned, primitive_only)
    state = 0
    option = agent.choose(state)
    start, total, duration = state, 0., 0
    rewards = deque(maxlen=100)
    lifetime = 0.
    rows = []
    for t in range(1, steps+1):
        action = agent.primitive(state, option)
        successor = transition(state, action, env.random())
        # Only the environment and evaluator know the change, never Agent.
        hidden_goal = GOALS[0] if t <= steps//2 else GOALS[1]
        reward = float(successor == hidden_goal)-.02
        agent.observe_primitive(state, action, successor)
        total += reward
        duration += 1
        ended = option < 2 or successor == GOALS[option-2] or duration == CAP
        state = successor
        if ended:
            agent.complete(start, option, total, duration, successor)
            # The reward switch and log windows do not reset any agent state.
            option = agent.choose(state)
            start, total, duration = state, 0., 0
        rewards.append(reward)
        lifetime += reward
        if t == 1 or t % 20 == 0 or t == steps:
            row = dict(step=t, value=sum(rewards)/len(rewards),
                       phase='before-change' if t <= steps//2 else 'after-change',
                       lifetime_reward=lifetime, average_reward=lifetime/t,
                       estimated_rate=agent.rate, real_backups=agent.real_backups,
                       model_backups=agent.model_backups, subtask_backups=agent.subtask_backups,
                       model_cells=len(agent.model.cells), invalidations=agent.invalidations,
                       option_finishes=agent.option_finishes, cap_finishes=agent.cap_finishes,
                       pending_duration=duration,
                       model_reward_mae=(sum(agent.prediction_errors)/len(agent.prediction_errors)
                                         if agent.prediction_errors else 0.),
                       model_error_samples=len(agent.prediction_errors))
            rows.append(row)
            if emit:
                emit(row.copy())
    # An unfinished option at the budget boundary is censored: do not make a fake
    # termination or update its full-option model with a partial reward/duration.
    return rows
