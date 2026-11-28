"""Original CPU teaching implementations, not benchmark reproductions.

python3 deep_textbook_train.py dqn --steps 2000 --seed 0
python3 deep_textbook_train.py ppo --epochs 20 --seed 0
python3 deep_textbook_train.py test
The neighboring deep_textbook_lab.py is required. No Gym/GPU is used.
"""
import argparse
import copy
import json
import math
import random
import unittest
from collections import deque

import torch
from torch import nn
from torch.distributions import Categorical, Normal
from deep_textbook_lab import gae

torch.set_num_threads(1)


def mlp(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, 32), nn.Tanh(), nn.Linear(32, n_out))


# BEGIN environment
class DeadlineChain:
    """Five positions; right reaches reward 1. The observed deadline is 12.

    Goal or deadline is genuine task termination. A rollout boundary in PPO
    is only a sampling cutoff and does not reset this environment.
    """
    def reset(self):
        self.position, self.remaining = 0, 12
        return self.observe()

    def observe(self):
        return torch.tensor([float(i == self.position) for i in range(5)]
                            + [self.remaining/12.0], dtype=torch.float32)

    def step(self, action):
        self.position = min(4, max(0, self.position + (1 if action else -1)))
        self.remaining -= 1
        reached = self.position == 4
        return self.observe(), (1.0 if reached else -0.02), reached or self.remaining == 0


def evaluate(actor, episodes=10):
    scores = []
    with torch.no_grad():
        for _ in range(episodes):
            env, score, done = DeadlineChain(), 0.0, False
            observation = env.reset()
            while not done:
                action = int(actor(observation).argmax())
                observation, reward, done = env.step(action)
                score += reward
            scores.append(score)
    return sum(scores)/len(scores)
# END environment


# BEGIN dqn_update
def dqn_update(online, target, optimizer, batch, gamma=.99, double=True):
    observations, actions, rewards, next_observations, terminated = zip(*batch)
    x, xp = torch.stack(observations), torch.stack(next_observations)
    action = torch.tensor(actions, dtype=torch.long)
    reward, terminal = torch.tensor(rewards), torch.tensor(terminated, dtype=torch.float32)
    prediction = online(x).gather(1, action[:, None]).squeeze(1)
    with torch.no_grad():
        target_values = target(xp)
        if double:
            selected = online(xp).argmax(dim=1)
            tail = target_values.gather(1, selected[:, None]).squeeze(1)
        else:
            tail = target_values.max(dim=1).values
        y = reward + gamma*(1-terminal)*tail
    assert prediction.shape == y.shape == (len(batch),)
    loss = nn.functional.smooth_l1_loss(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
    optimizer.step()
    return float(loss.detach())
# END dqn_update


# BEGIN dqn_train
def train_dqn(steps=2000, seed=0, double=True):
    random.seed(seed)
    torch.manual_seed(seed)
    online = mlp(6, 2)
    target = copy.deepcopy(online).requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=.003)
    replay = deque(maxlen=4000)
    env, initial_score = DeadlineChain(), evaluate(online)
    observation, losses = env.reset(), []
    for t in range(steps):
        epsilon = max(.05, 1-t/max(1, steps*.7))
        with torch.no_grad():
            action = random.randrange(2) if random.random() < epsilon else int(online(observation).argmax())
        next_observation, reward, terminated = env.step(action)
        replay.append((observation, action, reward, next_observation, terminated))
        observation = env.reset() if terminated else next_observation
        if len(replay) >= 32 and t % 2 == 0:
            batch = random.sample(list(replay), 32)
            losses.append(dqn_update(online, target, optimizer, batch, double=double))
        if (t+1) % 100 == 0:
            target.load_state_dict(online.state_dict())
    return {'algorithm': 'Double DQN' if double else 'DQN', 'seed': seed,
            'environment_steps': steps, 'updates': len(losses),
            'initial_greedy_return': initial_score, 'final_greedy_return': evaluate(online),
            'last_loss': losses[-1] if losses else None, 'scope': 'DeadlineChain only'}
# END dqn_train


# BEGIN vpg_update
def vpg_update(actor, critic, actor_optimizer, critic_optimizer, x, actions,
               advantages, returns):
    # One update with fresh on-policy data; gamma=1 episodic convention.
    distribution = Categorical(logits=actor(x))
    loss_actor = -(distribution.log_prob(actions)*advantages.detach()).mean()
    actor_optimizer.zero_grad()
    loss_actor.backward()
    actor_optimizer.step()
    prediction = critic(x).squeeze(-1)
    assert prediction.shape == returns.shape
    loss_critic = ((prediction-returns.detach())**2).mean()
    critic_optimizer.zero_grad()
    loss_critic.backward()
    critic_optimizer.step()
    return float(loss_actor.detach()), float(loss_critic.detach())
# END vpg_update


# BEGIN ppo_update
def ppo_update(actor, critic, actor_optimizer, critic_optimizer, x, actions,
               old_logp, advantages, returns, epochs=4, clip=.2, target_kl=.03):
    # Old log-probabilities, advantages and targets stay fixed for all epochs.
    old_logp, returns = old_logp.detach(), returns.detach()
    advantages = advantages.detach()
    advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    updates, final_kl = 0, 0.0
    for _ in range(epochs):
        distribution = Categorical(logits=actor(x))
        logp = distribution.log_prob(actions)
        ratio = (logp-old_logp).exp()
        kl = ((ratio-1)-(logp-old_logp)).mean()
        if float(kl.detach()) > target_kl:
            break
        surrogate = torch.minimum(ratio*advantages,
                                  ratio.clamp(1-clip, 1+clip)*advantages)
        loss_actor = -surrogate.mean()
        actor_optimizer.zero_grad()
        loss_actor.backward()
        actor_optimizer.step()
        updates += 1
    for _ in range(epochs):
        prediction = critic(x).squeeze(-1)
        assert prediction.shape == returns.shape
        loss_critic = ((prediction-returns)**2).mean()
        critic_optimizer.zero_grad()
        loss_critic.backward()
        critic_optimizer.step()
    with torch.no_grad():
        difference = Categorical(logits=actor(x)).log_prob(actions)-old_logp
        final_kl = float((difference.exp()-1-difference).mean())
    return {'actor_updates': updates, 'sample_kl': final_kl,
            'value_loss': float(loss_critic.detach())}
# END ppo_update


# BEGIN ppo_train
def train_ppo(epochs=20, seed=0, batch_steps=128):
    torch.manual_seed(seed)
    actor, critic = mlp(6, 2), mlp(6, 1)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=.003)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=.01)
    env, initial_score = DeadlineChain(), evaluate(actor)
    observation = env.reset()
    log = {}
    for _ in range(epochs):
        xs, actions, logps, rewards, values, next_values, terminals = [], [], [], [], [], [], []
        for _ in range(batch_steps):
            with torch.no_grad():
                distribution = Categorical(logits=actor(observation))
                action = distribution.sample()
                logp, value = distribution.log_prob(action), float(critic(observation).item())
            xp, reward, terminal = env.step(int(action))
            with torch.no_grad():
                next_value = float(critic(xp).item())
            xs.append(observation); actions.append(action); logps.append(logp)
            rewards.append(reward); values.append(value); next_values.append(next_value); terminals.append(terminal)
            observation = env.reset() if terminal else xp
        # gamma=1 matches this finite-horizon, undiscounted task objective.
        advantages, returns = gae(rewards, values, next_values, terminals, terminals, gamma=1., lam=.95)
        log = ppo_update(actor, critic, actor_opt, critic_opt, torch.stack(xs),
                         torch.stack(actions), torch.stack(logps),
                         torch.tensor(advantages), torch.tensor(returns))
    return {'algorithm': 'PPO-Clip', 'seed': seed, 'environment_steps': epochs*batch_steps,
            'initial_greedy_return': initial_score, 'final_greedy_return': evaluate(actor),
            **log, 'scope': 'DeadlineChain only; full-batch updates'}
# END ppo_train


class Critic(nn.Module):
    def __init__(self, observation_dim=3, action_dim=2):
        super().__init__()
        self.net = mlp(observation_dim+action_dim, 1)

    def forward(self, observation, action):
        return self.net(torch.cat([observation, action], -1)).squeeze(-1)


def freeze(modules, frozen):
    for module in modules:
        module.requires_grad_(not frozen)


def update_targets(online_modules, target_modules, tau):
    with torch.no_grad():
        for online, target in zip(online_modules, target_modules):
            for source_parameter, target_parameter in zip(online.parameters(), target.parameters()):
                target_parameter.lerp_(source_parameter, tau)


# BEGIN ddpg_update
def ddpg_update(actor, critic, target_actor, target_critic,
                actor_opt, critic_opt, batch, gamma=.99, tau=.005):
    x, action, reward, xp, terminal = batch
    with torch.no_grad():
        y = reward+gamma*(1-terminal)*target_critic(xp, target_actor(xp))
    prediction = critic(x, action)
    assert prediction.shape == y.shape == reward.shape
    loss_q = ((prediction-y)**2).mean()
    critic_opt.zero_grad(); loss_q.backward(); critic_opt.step()
    freeze([critic], True)
    loss_actor = -critic(x, actor(x)).mean()
    actor_opt.zero_grad(); loss_actor.backward(); actor_opt.step()
    freeze([critic], False)
    update_targets([actor, critic], [target_actor, target_critic], tau)
    return float(loss_q.detach()), float(loss_actor.detach())
# END ddpg_update


# BEGIN td3_update
def td3_update(actor, q1, q2, target_actor, target_q1, target_q2,
               actor_opt, critic_opt, batch, update_index, delay=2, gamma=.99, tau=.005):
    x, action, reward, xp, terminal = batch
    with torch.no_grad():
        next_action = target_actor(xp)
        noise = (.2*torch.randn_like(next_action)).clamp(-.5, .5)
        next_action = (next_action+noise).clamp(-1, 1)
        y = reward+gamma*(1-terminal)*torch.minimum(target_q1(xp, next_action), target_q2(xp, next_action))
    loss_q = ((q1(x, action)-y)**2+(q2(x, action)-y)**2).mean()
    critic_opt.zero_grad(); loss_q.backward(); critic_opt.step()
    policy_loss = None
    if update_index % delay == 0:
        freeze([q1, q2], True)
        # Freeze critic parameters, not the path from action to critic output.
        loss_actor = -q1(x, actor(x)).mean()
        actor_opt.zero_grad(); loss_actor.backward(); actor_opt.step()
        freeze([q1, q2], False)
        update_targets([actor, q1, q2], [target_actor, target_q1, target_q2], tau)
        policy_loss = float(loss_actor.detach())
    return float(loss_q.detach()), policy_loss
# END td3_update


# BEGIN squashed_policy
class SquashedGaussian(nn.Module):
    def __init__(self, observation_dim=3, action_dim=2):
        super().__init__()
        self.net = mlp(observation_dim, 2*action_dim)

    def forward(self, observation):
        mean, log_std = self.net(observation).chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)
        normal = Normal(mean, log_std.exp())
        u = normal.rsample()
        action = u.tanh()
        log_jacobian = 2*(math.log(2)-u-nn.functional.softplus(-2*u))
        log_prob = (normal.log_prob(u)-log_jacobian).sum(dim=-1)
        return action, log_prob
# END squashed_policy


# BEGIN sac_update
def sac_update(actor, q1, q2, target_q1, target_q2, log_alpha,
               actor_opt, critic_opt, alpha_opt, batch, target_entropy=-2., gamma=.99, tau=.005):
    x, action, reward, xp, terminal = batch
    alpha = log_alpha.exp().detach()
    with torch.no_grad():
        next_action, next_logp = actor(xp)
        soft_value = torch.minimum(target_q1(xp, next_action), target_q2(xp, next_action))-alpha*next_logp
        y = reward+gamma*(1-terminal)*soft_value
    loss_q = ((q1(x, action)-y)**2+(q2(x, action)-y)**2).mean()
    critic_opt.zero_grad(); loss_q.backward(); critic_opt.step()
    freeze([q1, q2], True)
    sampled_action, logp = actor(x)
    assert logp.shape == reward.shape
    loss_actor = (alpha*logp-torch.minimum(q1(x, sampled_action), q2(x, sampled_action))).mean()
    actor_opt.zero_grad(); loss_actor.backward(); actor_opt.step()
    freeze([q1, q2], False)
    # Exact log-alpha parameterization of the dual loss, not its common proxy.
    loss_alpha = -(log_alpha.exp()*(logp.detach()+target_entropy)).mean()
    alpha_opt.zero_grad(); loss_alpha.backward(); alpha_opt.step()
    update_targets([q1, q2], [target_q1, target_q2], tau)
    return {'critic_loss': float(loss_q.detach()), 'actor_loss': float(loss_actor.detach()),
            'alpha': float(log_alpha.exp().detach())}
# END sac_update


# BEGIN fisher
def fisher_vector_product(logits, vector, damping=0.):
    old_prob = logits.detach().softmax(-1)
    old_logp = logits.detach().log_softmax(-1)
    kl = (old_prob*(old_logp-logits.log_softmax(-1))).sum(-1).mean()
    gradient = torch.autograd.grad(kl, logits, create_graph=True)[0]
    product = torch.autograd.grad((gradient*vector).sum(), logits)[0]
    return product+damping*vector
# END fisher


class Checks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)

    def test_dqn_stopped_target(self):
        online, target = mlp(6, 2), mlp(6, 2)
        optimizer = torch.optim.SGD(online.parameters(), .01)
        batch = [(torch.zeros(6), 1, 1., torch.ones(6), False)]*4
        self.assertTrue(math.isfinite(dqn_update(online, target, optimizer, batch)))
        self.assertTrue(all(p.grad is None for p in target.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in online.parameters()))

    def test_squashed_policy_pathwise_gradient(self):
        actor = SquashedGaussian()
        action, logp = actor(torch.randn(7, 3))
        self.assertEqual(action.shape, (7, 2)); self.assertEqual(logp.shape, (7,))
        self.assertTrue((action.abs() <= 1).all())
        (action.square().sum()+logp.sum()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.parameters()))

    def test_deterministic_actor_finite_difference(self):
        theta = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        value = -(theta-.8)**2
        gradient = torch.autograd.grad(value, theta)[0].item()
        eps = 1e-6
        value_at = lambda x: -(x-.8)**2
        finite_difference = (value_at(eps)-value_at(-eps))/(2*eps)
        self.assertAlmostEqual(gradient, 1.6)
        self.assertAlmostEqual(gradient, finite_difference, places=8)

    def test_sac_pathwise_finite_difference(self):
        # Keep base noise fixed: compare the same reparameterized sample path.
        def objective(mean):
            normal = Normal(mean, torch.tensor(.7, dtype=torch.float64))
            u = mean + .7*.3
            log_jacobian = 2*(math.log(2)-u-nn.functional.softplus(-2*u))
            logp = normal.log_prob(u)-log_jacobian
            return .2*logp+(u.tanh()-.4)**2
        mean = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
        gradient = torch.autograd.grad(objective(mean), mean)[0].item()
        eps = 1e-6
        plus = objective(torch.tensor(.1+eps, dtype=torch.float64)).item()
        minus = objective(torch.tensor(.1-eps, dtype=torch.float64)).item()
        self.assertAlmostEqual(gradient, (plus-minus)/(2*eps), places=8)

    def test_observed_deadline_and_greedy_optimum(self):
        env = DeadlineChain()
        env.reset()
        for t in range(12):
            observation, reward, terminal = env.step(0)
            self.assertEqual(terminal, t == 11)
            self.assertAlmostEqual(observation[-1].item(), (11-t)/12, places=6)
        env.reset()
        total = 0.
        for _ in range(4):
            _, reward, terminal = env.step(1)
            total += reward
        self.assertTrue(terminal)
        self.assertAlmostEqual(total, .94)

    def test_fisher_matches_closed_form(self):
        logits = torch.tensor([.2, -.1], dtype=torch.float64, requires_grad=True)
        vector = torch.tensor([.4, -.3], dtype=torch.float64)
        p = logits.detach().softmax(-1)
        expected = p*vector-p*(p*vector).sum()
        torch.testing.assert_close(fisher_vector_product(logits, vector), expected)

    def test_td3_delayed_actor(self):
        actor = nn.Sequential(mlp(3, 2), nn.Tanh())
        q1, q2 = Critic(), Critic()
        ta, tq1, tq2 = [copy.deepcopy(x).requires_grad_(False) for x in (actor, q1, q2)]
        ao = torch.optim.Adam(actor.parameters(), lr=.001)
        qo = torch.optim.Adam(list(q1.parameters())+list(q2.parameters()), lr=.001)
        batch = (torch.randn(8, 3), torch.randn(8, 2).tanh(), torch.randn(8), torch.randn(8, 3), torch.zeros(8))
        before = [p.detach().clone() for p in actor.parameters()]
        _, loss = td3_update(actor, q1, q2, ta, tq1, tq2, ao, qo, batch, 1)
        self.assertIsNone(loss)
        for a, b in zip(before, actor.parameters()): torch.testing.assert_close(a, b)
        _, loss = td3_update(actor, q1, q2, ta, tq1, tq2, ao, qo, batch, 2)
        self.assertTrue(math.isfinite(loss))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, actor.parameters())))
        self.assertTrue(all(p.grad is None for p in tq1.parameters()))

    def test_ddpg_target_and_actor(self):
        actor = nn.Sequential(mlp(3, 2), nn.Tanh())
        critic = Critic()
        ta, tq = [copy.deepcopy(x).requires_grad_(False) for x in (actor, critic)]
        ao = torch.optim.Adam(actor.parameters(), lr=.001)
        qo = torch.optim.Adam(critic.parameters(), lr=.001)
        batch = (torch.randn(8, 3), torch.randn(8, 2).tanh(), torch.randn(8), torch.randn(8, 3), torch.zeros(8))
        before = [p.detach().clone() for p in actor.parameters()]
        losses = ddpg_update(actor, critic, ta, tq, ao, qo, batch)
        self.assertTrue(all(math.isfinite(v) for v in losses))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, actor.parameters())))
        self.assertTrue(all(p.grad is None for p in tq.parameters()))

    def test_sac_update_shapes(self):
        actor, q1, q2 = SquashedGaussian(), Critic(), Critic()
        tq1, tq2 = [copy.deepcopy(x).requires_grad_(False) for x in (q1, q2)]
        log_alpha = nn.Parameter(torch.tensor(math.log(.2)))
        ao = torch.optim.Adam(actor.parameters(), lr=.001)
        qo = torch.optim.Adam(list(q1.parameters())+list(q2.parameters()), lr=.001)
        alpha_opt = torch.optim.Adam([log_alpha], lr=.001)
        batch = (torch.randn(8, 3), torch.randn(8, 2).tanh(), torch.randn(8), torch.randn(8, 3), torch.zeros(8))
        result = sac_update(actor, q1, q2, tq1, tq2, log_alpha, ao, qo, alpha_opt, batch)
        self.assertTrue(all(math.isfinite(v) for v in result.values()))
        self.assertTrue(all(p.grad is None for p in tq1.parameters()))

    def test_ppo_fixed_targets(self):
        actor, critic = mlp(6, 2), mlp(6, 1)
        ao, co = torch.optim.Adam(actor.parameters(), .001), torch.optim.Adam(critic.parameters(), .001)
        x = torch.randn(16, 6)
        with torch.no_grad():
            dist = Categorical(logits=actor(x)); actions = dist.sample(); old = dist.log_prob(actions)
        old_copy = old.clone()
        result = ppo_update(actor, critic, ao, co, x, actions, old, torch.randn(16), torch.randn(16))
        torch.testing.assert_close(old_copy, old)
        self.assertTrue(math.isfinite(result['value_loss']))

    def test_vpg_detached_advantages(self):
        actor, critic = mlp(6, 2), mlp(6, 1)
        ao, co = torch.optim.Adam(actor.parameters(), .001), torch.optim.Adam(critic.parameters(), .001)
        x = torch.randn(16, 6)
        with torch.no_grad():
            actions = Categorical(logits=actor(x)).sample()
        advantages = torch.randn(16, requires_grad=True)
        returns = torch.randn(16, requires_grad=True)
        losses = vpg_update(actor, critic, ao, co, x, actions, advantages, returns)
        self.assertTrue(all(math.isfinite(v) for v in losses))
        self.assertIsNone(advantages.grad)
        self.assertIsNone(returns.grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['dqn', 'ppo', 'test'])
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ordinary-dqn', action='store_true')
    args = parser.parse_args()
    if args.command == 'test':
        unittest.main(argv=['deep_textbook_train.py'], verbosity=2)
    else:
        if args.steps < 32 or args.epochs < 1:
            parser.error('steps >= 32 and epochs >= 1 are required')
        result = train_dqn(args.steps, args.seed, not args.ordinary_dqn) if args.command == 'dqn' else train_ppo(args.epochs, args.seed)
        print(json.dumps(result, indent=2))
