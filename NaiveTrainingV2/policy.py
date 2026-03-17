import torch
import torch.nn as nn
from torch.distributions import Categorical


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=[256, 256]):
        super().__init__()

        def mlp(dims):
            layers = []
            for i in range(len(dims) - 1):
                layers += [nn.Linear(dims[i], dims[i+1]), nn.Tanh()]
            return nn.Sequential(*layers[:-1])  # no activation on last layer

        self.actor  = mlp([obs_dim] + hidden + [action_dim])
        self.critic = mlp([obs_dim] + hidden + [1])

    def forward(self, obs):
        logits = self.actor(obs)
        value  = self.critic(obs).squeeze(-1)
        return logits, value

    def get_action(self, obs, deterministic=False):
        logits, value = self(obs)
        dist   = Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, obs, actions):
        logits, value = self(obs)
        dist     = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return log_prob, entropy, value
