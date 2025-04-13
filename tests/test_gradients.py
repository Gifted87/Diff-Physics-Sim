import torch
import unittest
import numpy as np
import warp as wp

# Need to initialize env to get model and state_template
from environments.ant.ant_env import AntEnv # Example env
from warp_kernels.simulation_step import DifferentiablePhysicsStep
from warp_kernels.core import device as warp_device, torch_to_state, state_to_torch

class TestGradients(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize a simple environment to get model and state structure
        cls.env = AntEnv(device=warp_device) # Use the same device Warp uses
        cls.model = cls.env.model
        cls.state_template = cls.env.state_template
        cls.n_bodies = cls.model.n_bodies
        cls.action_dim = cls.env.action_space.shape[0]

        # Dummy action mapping (replace with actual mapping if possible)
        cls.dummy_forces = torch.randn(cls.n_bodies, 3, device=warp_device, dtype=torch.float32)
        cls.dummy_torques = torch.randn(cls.n_bodies, 3, device=warp_device, dtype=torch.float32)


    def test_physics_step_gradients(self):
        print("\nTesting DifferentiablePhysicsStep gradients...")

        # Get an initial state tensor
        initial_state_flat = self.env.get_state_flat(requires_grad=True).detach() # Detach before setting requires_grad

        # Define the function to be checked: it takes state, forces, torques and returns next_state
        # gradcheck needs inputs to be tuple
        def func_to_check(state_flat, forces, torques):
             # Ensure inputs have grad enabled for gradcheck internal workings
             state_flat = state_flat.requires_grad_(True)
             forces = forces.requires_grad_(True)
             torques = torques.requires_grad_(True)
             # The apply function handles model and state_template internally
             return DifferentiablePhysicsStep.apply(state_flat, forces, torques, self.model, self.state_template)

        # Create dummy input tensors with requires_grad=True for gradcheck
        test_state = initial_state_flat.clone().detach().requires_grad_(True)
        test_forces = self.dummy_forces.clone().detach().requires_grad_(True)
        test_torques = self.dummy_torques.clone().detach().requires_grad_(True)

        # Inputs must be float64 for gradcheck by default
        test_state = test_state.to(torch.float64)
        test_forces = test_forces.to(torch.float64)
        test_torques = test_torques.to(torch.float64)

        # gradcheck verifies gradients
        # Use eps=1e-4 or 1e-5, atol might need adjustment based on physics complexity/smoothing
        # Check gradients w.r.t all inputs (state, forces, torques)
        try:
             is_correct = torch.autograd.gradcheck(func_to_check, (test_state, test_forces, test_torques), eps=1e-4, atol=5e-3, rtol=1e-2, raise_exception=True)
             self.assertTrue(is_correct)
             print("Gradient check PASSED!")
        except Exception as e:
             print(f"Gradient check FAILED: {e}")
             self.fail("Gradient check failed")


if __name__ == '__main__':
    unittest.main()