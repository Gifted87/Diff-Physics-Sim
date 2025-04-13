import warp as wp
import torch

# Define float and int types based on torch default
wp.set_preferred_device('cuda') # Ensure Warp uses GPU
dtype = wp.float32
device = 'cuda' # Assuming GPU usage

@wp.struct
class RigidBodyState:
    # Pose
    body_pos: wp.array(dtype=wp.vec3, ndim=1)  # Position (n_bodies)
    body_rot: wp.array(dtype=wp.quat, ndim=1)  # Orientation (n_bodies)
    # Velocity
    body_vel: wp.array(dtype=wp.vec3, ndim=1)  # Linear velocity (n_bodies)
    body_ang_vel: wp.array(dtype=wp.vec3, ndim=1) # Angular velocity (n_bodies)
    # Force accumulators
    body_force: wp.array(dtype=wp.vec3, ndim=1) # Forces (n_bodies)
    body_torque: wp.array(dtype=wp.vec3, ndim=1) # Torques (n_bodies)

@wp.struct
class Model:
    # Static properties of rigid bodies
    body_inv_mass: wp.array(dtype=dtype, ndim=1) # Inverse mass (n_bodies)
    body_inv_inertia: wp.array(dtype=wp.mat33, ndim=1) # Inverse inertia tensor (body frame) (n_bodies)

    # Contact properties (example: spheres)
    body_radius: wp.array(dtype=dtype, ndim=1) # Radius for collision (n_bodies)

    # Ground plane
    ground_plane_normal: wp.vec3
    ground_plane_offset: dtype

    # Simulation parameters
    gravity: wp.vec3
    dt: dtype # Simulation timestep

    # Contact parameters
    contact_ke: dtype # Contact elastic stiffness
    contact_kd: dtype # Contact damping
    contact_mu: dtype # Friction coefficient (Coulomb)

    # Total number of bodies
    n_bodies: int


def state_to_torch(state: RigidBodyState, requires_grad=False):
    """ Converts RigidBodyState Warp arrays to a flat Torch tensor. """
    tensors = [
        wp.to_torch(state.body_pos),
        wp.to_torch(state.body_rot), # Quaternions need careful handling for gradients if represented differently
        wp.to_torch(state.body_vel),
        wp.to_torch(state.body_ang_vel)
    ]
    flat_tensor = torch.cat([t.reshape(-1) for t in tensors])
    if requires_grad:
        flat_tensor.requires_grad_(True)
    return flat_tensor

def torch_to_state(flat_tensor: torch.Tensor, state_template: RigidBodyState):
    """ Converts a flat Torch tensor back into a RigidBodyState struct. """
    n_bodies = state_template.body_pos.shape[0]
    pos_size = n_bodies * 3
    rot_size = n_bodies * 4 # quat
    vel_size = n_bodies * 3
    ang_vel_size = n_bodies * 3

    current_idx = 0
    pos = flat_tensor[current_idx : current_idx + pos_size].reshape(n_bodies, 3)
    current_idx += pos_size
    rot = flat_tensor[current_idx : current_idx + rot_size].reshape(n_bodies, 4)
    current_idx += rot_size
    vel = flat_tensor[current_idx : current_idx + vel_size].reshape(n_bodies, 3)
    current_idx += vel_size
    ang_vel = flat_tensor[current_idx : current_idx + ang_vel_size].reshape(n_bodies, 3)

    # Create new state or update template? Let's update template arrays
    wp.copy(target=state_template.body_pos, source=wp.from_torch(pos, dtype=wp.vec3))
    wp.copy(target=state_template.body_rot, source=wp.from_torch(rot, dtype=wp.quat))
    wp.copy(target=state_template.body_vel, source=wp.from_torch(vel, dtype=wp.vec3))
    wp.copy(target=state_template.body_ang_vel, source=wp.from_torch(ang_vel, dtype=wp.vec3))

    # Important: Clear force/torque accumulators if this state represents a new step start
    wp.launch(kernel=zero_forces, dim=n_bodies, inputs=[state_template], device=device)

    return state_template # Return the updated state object

@wp.kernel
def zero_forces(state: RigidBodyState):
    tid = wp.tid()
    state.body_force[tid] = wp.vec3(0.0, 0.0, 0.0)
    state.body_torque[tid] = wp.vec3(0.0, 0.0, 0.0)