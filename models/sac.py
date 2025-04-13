import torch
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy

from .policy import Actor
from .critic import Critic
from warp_kernels.simulation_step import differentiable_physics_step # Import the autograd function
from warp_kernels.core import state_to_torch, torch_to_state # Utilities

class SAC:
    def __init__(self, obs_dim, action_dim, action_space_low, action_space_high,
                 actor_lr=3e-4, critic_lr=1e-3, alpha_lr=3e-4, gamma=0.99, tau=0.005,
                 alpha=0.2, # Initial alpha or 'auto'
                 target_update_interval=1,
                 hidden_dim=256,
                 physics_grad_weight=0.2, # Weight for physics-based gradient loss
                 device='cuda'):

        self.gamma = gamma
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.physics_grad_weight = physics_grad_weight
        self.device = device

        # Actor network
        self.actor = Actor(obs_dim, action_dim, hidden_dim, action_space_low, action_space_high).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic network
        self.critic = Critic(obs_dim, action_dim, hidden_dim).to(device)
        self.critic_target = deepcopy(self.critic) # Target network
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Alpha (entropy regularization coefficient)
        self.auto_tune_alpha = isinstance(alpha, str) and alpha.lower() == 'auto'
        if self.auto_tune_alpha:
            self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp().item() # Initial value
        else:
            self.alpha = alpha

        self.total_it = 0

    def select_action(self, obs, deterministic=False):
        # Assume obs is already a tensor on the correct device
        with torch.no_grad():
            mean, log_std = self.actor(obs.unsqueeze(0)) # Add batch dimension
            if deterministic:
                # Use mean for deterministic action
                action = torch.tanh(mean) * self.actor.action_scale.to(self.device) + self.actor.action_bias.to(self.device)
            else:
                # Sample action using reparameterization trick result
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                x_t = normal.sample() # Can use sample here as no grad needed for selection
                y_t = torch.tanh(x_t)
                action = y_t * self.actor.action_scale.to(self.device) + self.actor.action_bias.to(self.device)
        return action.squeeze(0).cpu().numpy() # Remove batch dim and move to CPU

    def update_rl(self, batch):
        """ Standard SAC update using replay buffer batch. """
        self.total_it += 1

        obs = batch['obs']
        actions = batch['actions']
        rewards = batch['rewards']
        next_obs = batch['next_obs']
        dones = batch['dones']

        # --- Critic Update ---
        with torch.no_grad():
            # Get next actions and log probs from current policy
            next_actions, next_log_pi, _ = self.actor.sample(next_obs)
            # Compute target Q values
            q1_next_target, q2_next_target = self.critic_target(next_obs, next_actions)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_pi
            # Bellman equation for target Q
            next_q_value = rewards + (1.0 - dones) * self.gamma * min_q_next_target

        # Get current Q estimates
        q1, q2 = self.critic(obs, actions)

        # Compute critic loss (MSE Bellman error)
        q1_loss = F.mse_loss(q1, next_q_value)
        q2_loss = F.mse_loss(q2, next_q_value)
        critic_loss = q1_loss + q2_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor Update ---
        # Freeze critic gradients
        for p in self.critic.parameters():
            p.requires_grad = False

        # Compute actor loss
        pi, log_pi, _ = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_pi - min_q_pi).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic gradients
        for p in self.critic.parameters():
            p.requires_grad = True

        # --- Alpha Update (Optional) ---
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # --- Target Network Update ---
        if self.total_it % self.target_update_interval == 0:
            self._update_target_network(self.critic, self.critic_target)

        return critic_loss.item(), actor_loss.item()


    def update_physics_gradient(self, batch, model, state_template):
        """ Update policy using gradients flowed through the differentiable physics simulator. """
        obs = batch['obs'] # Observations often map directly to part of the physics state

        current_state_flat = obs.clone().detach().requires_grad_(True)

        # Get actions from policy for the current state
        # Use rsample() for policy gradient, but we need forces/torques
        # Let's assume the policy directly outputs forces/torques for now
        # This needs refinement based on env action space definition.
        # Option 1: Policy outputs joint torques/forces directly.
        # Option 2: Policy outputs target positions/velocities, a PD controller calculates forces. (More common)

        # --- Assuming Option 1: Policy outputs forces/torques ---
        # Requires actor output dim = force_dim + torque_dim
        # Let's fake this for now: assume action is split into force/torque components
        n_bodies = state_template.body_pos.shape[0]
        # This mapping is ENV specific! Placeholder:
        action_forces_flat, action_torques_flat = self.actor.sample_forces_torques(current_state_flat) # Needs implementation in Actor

        # --- Enable gradient computation through physics ---
        with torch.enable_grad(): # Ensure PyTorch tracks operations
            # Simulate one step using the differentiable physics function
            predicted_next_state_flat = differentiable_physics_step(
                current_state_flat,
                action_forces_flat.reshape(n_bodies, 3), # Reshape action parts
                action_torques_flat.reshape(n_bodies, 3),
                model,
                state_template
            )

            # --- Define loss based on the *predicted* next state ---
            # Goal: Maximize the expected future return from the predicted state.
            # Use the target critic to estimate the value (Q-value) of the predicted state.
            # Need an action for the predicted state from the policy.
            predicted_next_obs = predicted_next_state_flat # Again, assuming state = obs
            with torch.no_grad(): # Don't need gradients for next action in this loss term itself
                 predicted_next_action, _, _ = self.actor.sample(predicted_next_obs)

            # Use target critic to get Q-value of predicted next state/action
            q1_pred_next, q2_pred_next = self.critic_target(predicted_next_obs, predicted_next_action)
            min_q_pred_next = torch.min(q1_pred_next, q2_pred_next)

            # Physics gradient loss: We want to maximize this value. Loss is negative value.
            physics_loss = -min_q_pred_next.mean() * self.physics_grad_weight

        # --- Backpropagate through physics and update actor ---
        self.actor_optimizer.zero_grad()
        physics_loss.backward() # Gradients flow back through differentiable_physics_step
        self.actor_optimizer.step()

        return physics_loss.item() / self.physics_grad_weight # Return unweighted loss

    def _update_target_network(self, main_net, target_net):
        """ Polyak averaging for target network update. """
        for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save({
            'critic_state_dict': self.critic.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_tune_alpha else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.auto_tune_alpha else None
        }, filename)
        print(f"Saved SAC model to {filename}")

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target = deepcopy(self.critic)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        if self.auto_tune_alpha and checkpoint['log_alpha'] is not None:
            self.log_alpha.data.copy_(checkpoint['log_alpha'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.alpha = self.log_alpha.exp().item()
        print(f"Loaded SAC model from {filename}")