import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, action_space_low=None, action_space_high=None):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)

        # Action scaling
        if action_space_low is None or action_space_high is None:
            self.action_scale = torch.tensor(1.0)
            self.action_bias = torch.tensor(0.0)
        else:
            self.action_scale = torch.tensor((action_space_high - action_space_low) / 2.0, dtype=torch.float32)
            self.action_bias = torch.tensor((action_space_high + action_space_low) / 2.0, dtype=torch.float32)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (used in SAC update)
        y_t = torch.tanh(x_t)   # Squash distribution to (-1, 1)

        # Scale to action space
        action = y_t * self.action_scale.to(obs.device) + self.action_bias.to(obs.device)

        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound: Correcting log probability for Tanh squashing
        # Derivation: log p(y) = log p(x) - log(1 - tanh(x)^2)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True) # Sum across action dimensions

        mean = torch.tanh(mean) * self.action_scale.to(obs.device) + self.action_bias.to(obs.device) # Squashed mean

        return action, log_prob, mean