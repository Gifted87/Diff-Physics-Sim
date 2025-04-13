import torch
import warp as wp
from .core import RigidBodyState, Model, device, state_to_torch, torch_to_state, zero_forces
from .rigid_body import integrate_bodies
from .contact import compute_contacts # Assuming contact kernel applies forces directly

class DifferentiablePhysicsStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, current_state_flat: torch.Tensor, action_forces: torch.Tensor, action_torques: torch.Tensor, model: Model, state_template: RigidBodyState):
        """
        Performs one simulation step.
        Args:
            current_state_flat: Flat tensor representing the current RigidBodyState.
            action_forces: Tensor of forces applied by the agent/controller.
            action_torques: Tensor of torques applied by the agent/controller.
            model: Static Model parameters struct.
            state_template: A template RigidBodyState to reconstruct the state struct.
        Returns:
            next_state_flat: Flat tensor representing the state after one step.
        """
        # Ensure tensors are on the correct device
        current_state_flat = current_state_flat.to(device)
        action_forces = action_forces.to(device)
        action_torques = action_torques.to(device)

        # Create/update Warp state from input tensor
        current_state = torch_to_state(current_state_flat, state_template)

        # We need a copy of the state for the backward pass if using Warp's adjoint method directly
        # state_for_backward = RigidBodyState() # Need to properly allocate and copy

        # Apply action forces/torques (add to existing forces, e.g., gravity computed earlier)
        # Ensure zero_forces was called before this step
        wp.launch(kernel=apply_actions_kernel, dim=model.n_bodies,
                  inputs=[current_state, wp.from_torch(action_forces), wp.from_torch(action_torques)],
                  device=device)

        # --- Execute Physics Kernels ---
        # 1. Compute contact forces based on current state (adds to state.body_force/torque)
        wp.launch(kernel=compute_contacts, dim=model.n_bodies, inputs=[current_state, model], device=device)

        # 2. Integrate dynamics using accumulated forces (updates state pos/vel/rot/ang_vel)
        wp.launch(kernel=integrate_bodies, dim=model.n_bodies, inputs=[current_state, model], device=device)

        # --- Prepare for backward pass ---
        # Save necessary tensors for backward. Action forces/torques are inputs,
        # and the state *before* integration is needed to recompute intermediate steps.
        # However, Warp's autodiff through launch handles this implicitly if tape is active.
        # Let's rely on PyTorch's tape combined with Warp's kernel diff for now.
        # If using wp.Tape explicitly, we'd save the tape.
        ctx.save_for_backward(current_state_flat, action_forces, action_torques) # Save input state and actions
        ctx.model = model # Need model parameters in backward
        ctx.state_template = state_template # Need template to reconstruct state

        # --- Return next state ---
        next_state_flat = state_to_torch(current_state, requires_grad=False) # The state object was modified in-place

        return next_state_flat

    @staticmethod
    def backward(ctx, grad_output_flat: torch.Tensor):
        """
        Computes gradients w.r.t. forward inputs: current_state_flat, action_forces, action_torques.
        """
        # Ensure Warp is recording gradients
        if not wp.is_tape_active():
             wp.Tape() # Start a tape if not already active (e.g., within torch.enable_grad)

        with wp.Tape() as tape: # Use Warp's tape for gradient computation
             # Retrieve saved tensors and parameters
             current_state_flat, action_forces, action_torques = ctx.saved_tensors
             model = ctx.model
             state_template = ctx.state_template

             # --- Re-run forward pass under the tape ---
             # Detach inputs to avoid PyTorch tracking them within this backward pass
             current_state_flat = current_state_flat.detach().requires_grad_(True)
             action_forces = action_forces.detach().requires_grad_(True)
             action_torques = action_torques.detach().requires_grad_(True)

             # Reconstruct state and clear forces
             current_state = torch_to_state(current_state_flat, state_template)

             # Apply actions
             wp.launch(kernel=apply_actions_kernel, dim=model.n_bodies,
                       inputs=[current_state, wp.from_torch(action_forces), wp.from_torch(action_torques)],
                       device=device)
             # Compute contacts
             wp.launch(kernel=compute_contacts, dim=model.n_bodies, inputs=[current_state, model], device=device)
             # Integrate
             wp.launch(kernel=integrate_bodies, dim=model.n_bodies, inputs=[current_state, model], device=device)

             # Convert the *resulting* state (output of forward) to Torch for loss calculation
             next_state_flat_tape = state_to_torch(current_state, requires_grad=True) # Need grad=True for tape

             # --- Compute gradients using the tape ---
             # Apply the incoming gradient (from the loss function) to the output of the taped forward pass
             tape.backward(loss=next_state_flat_tape, grad_outputs={'loss': grad_output_flat.to(device)})

             # --- Extract gradients w.r.t. inputs ---
             grad_state = tape.gradients[current_state_flat] # Gradient w.r.t. initial state flat tensor
             grad_action_forces = tape.gradients[action_forces] # Gradient w.r.t. action forces
             grad_action_torques = tape.gradients[action_torques] # Gradient w.r.t. action torques

             # Return gradients in the order of inputs in forward (excluding model, state_template)
             # Return None for non-Tensor inputs (model, state_template)
             return grad_state, grad_action_forces, grad_action_torques, None, None

@wp.kernel
def apply_actions_kernel(state: RigidBodyState, action_forces: wp.array(dtype=wp.vec3), action_torques: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    # Assume action forces/torques are directly applicable in world frame
    wp.atomic_add(state.body_force, tid, action_forces[tid])
    wp.atomic_add(state.body_torque, tid, action_torques[tid])

# Convenience function to call the autograd Function
def differentiable_physics_step(current_state_flat, action_forces, action_torques, model, state_template):
    return DifferentiablePhysicsStep.apply(current_state_flat, action_forces, action_torques, model, state_template)