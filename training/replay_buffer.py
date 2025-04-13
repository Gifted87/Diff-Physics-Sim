import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, capacity, device='cuda'):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs_buf = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.next_obs_buf = torch.zeros((capacity, obs_dim), dtype=torch.float32, device=device)
        self.actions_buf = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards_buf = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.dones_buf = torch.zeros((capacity, 1), dtype=torch.float32, device=device) # Use float for easy multiplication

    def add(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs.to(self.device) # Ensure data is on the correct device
        self.actions_buf[self.ptr] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.rewards_buf[self.ptr] = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        self.next_obs_buf[self.ptr] = next_obs.to(self.device)
        self.dones_buf[self.ptr] = torch.tensor([[float(done)]], dtype=torch.float32, device=self.device)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     next_obs=self.next_obs_buf[idxs],
                     actions=self.actions_buf[idxs],
                     rewards=self.rewards_buf[idxs],
                     dones=self.dones_buf[idxs])
        return batch # Tensors are already on the correct device