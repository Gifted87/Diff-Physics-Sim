import warp as wp
from .core import RigidBodyState, Model, dtype, device

@wp.kernel
def integrate_bodies(state: RigidBodyState, model: Model):
    tid = wp.tid()

    # Retrieve state and parameters for the current body
    pos = state.body_pos[tid]
    rot = state.body_rot[tid]
    vel = state.body_vel[tid]
    ang_vel = state.body_ang_vel[tid]
    force = state.body_force[tid]
    torque = state.body_torque[tid]

    inv_mass = model.body_inv_mass[tid]
    inv_inertia_local = model.body_inv_inertia[tid] # Stored in local body frame

    dt = model.dt
    gravity = model.gravity

    # Apply gravity (if mass > 0)
    if inv_mass > 0.0:
        force = force + gravity / inv_mass # Apply gravity as a force

    # --- Semi-implicit Euler integration ---

    # Update velocity
    vel = vel + force * inv_mass * dt
    # Update position
    pos = pos + vel * dt

    # Update angular velocity
    # Convert inertia tensor to world frame
    rot_mat = wp.mat33_from_quat(rot)
    inv_inertia_world = rot_mat * inv_inertia_local * wp.transpose(rot_mat)
    ang_vel = ang_vel + inv_inertia_world * torque * dt

    # Update orientation (integrate angular velocity)
    # Use small angle approximation: q_new = q_old * exp(0.5 * omega * dt)
    # exp(x) ≈ 1 + x for small x. quat_from_axis_angle ≈ (1, 0.5 * axis * angle)
    axis_angle = ang_vel * dt
    delta_rot = wp.quat_from_axis_angle(wp.normalize(axis_angle), wp.length(axis_angle))
    rot = wp.normalize(delta_rot * rot) # Use delta_rot * rot for world frame ang vel

    # Store updated state
    state.body_pos[tid] = pos
    state.body_rot[tid] = rot
    state.body_vel[tid] = vel
    state.body_ang_vel[tid] = ang_vel

    # Clear forces/torques for the next step accumulation (optional here, maybe better in main loop)
    # state.body_force[tid] = wp.vec3(0.0, 0.0, 0.0)
    # state.body_torque[tid] = wp.vec3(0.0, 0.0, 0.0)