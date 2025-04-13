from abc import ABC, abstractmethod
import numpy as np
import torch
import warp as wp

from warp_kernels.core import Model, RigidBodyState, state_to_torch, torch_to_state, device as warp_device
from warp_kernels.simulation_step import differentiable_physics_step

class DiffEnv(ABC):
    def __init__(self, device='cuda'):
        self.device = device
        self.model: Model = None
        self.state: RigidBodyState = None
        self.state_template: RigidBodyState = None # Keep template for reconstruction
        self.action_space = None # Define in subclass (e.g., gym.spaces.Box)
        self.observation_space = None # Define in subclass

        self._init_sim() # Subclass should implement this to setup model and initial state

    @abstractmethod
    def _init_sim(self):
        """ Initialize Warp model (Model) and initial state (RigidBodyState)."""
        pass

    @abstractmethod
    def _get_observation(self) -> torch.Tensor:
        """ Extract observation tensor from current simulation state. """
        pass

    @abstractmethod
    def _get_reward(self) -> float:
        """ Calculate reward based on current simulation state. """
        pass

    @abstractmethod
    def _is_done(self) -> bool:
        """ Check termination conditions based on current simulation state. """
        pass

    @abstractmethod
    def _compute_action_forces(self, action: np.ndarray) -> (torch.Tensor, torch.Tensor):
        """ Convert policy action np.ndarray into force and torque tensors for simulation step. """
        pass

    @abstractmethod
    def _reset_state(self):
        """ Reset the simulation state (Warp arrays) to initial conditions. """
        pass

    def reset(self) -> torch.Tensor:
        """ Resets the environment to an initial state and returns the initial observation. """
        self._reset_state()
        obs = self._get_observation()
        return obs.to(self.device)

    def step(self, action: np.ndarray) -> (torch.Tensor, float, bool, dict):
        """
        Run one timestep of the environment's dynamics.
        Accepts an action, returns observation, reward, done, info.
        """
        # 1. Convert agent action to simulation forces/torques
        action_forces_tensor, action_torques_tensor = self._compute_action_forces(action)
        action_forces_tensor = action_forces_tensor.to(warp_device)
        action_torques_tensor = action_torques_tensor.to(warp_device)

        # 2. Get current state as flat tensor (for differentiable step input)
        # Ensure requires_grad is False unless we need gradients w.r.t previous state itself
        current_state_flat = state_to_torch(self.state, requires_grad=False).to(warp_device)

        # 3. Perform differentiable physics step
        # Note: If used inside RL update with torch.enable_grad(), gradients will flow.
        # For standard env interaction, no grad calculation is needed here.
        with torch.no_grad(): # Typically no grad needed for standard env stepping
             next_state_flat = differentiable_physics_step(
                 current_state_flat,
                 action_forces_tensor,
                 action_torques_tensor,
                 self.model,
                 self.state # Pass template for reconstruction
             )

        # 4. Update self.state using the resulting tensor (Warp arrays updated in torch_to_state)
        self.state = torch_to_state(next_state_flat, self.state)

        # 5. Calculate results
        obs = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()
        info = {} # Optional auxiliary diagnostic information

        return obs.to(self.device), reward, done, info

    def get_state_flat(self, requires_grad=False):
        """ Utility to get the current physics state as a flat tensor. """
        return state_to_torch(self.state, requires_grad=requires_grad).to(self.device)