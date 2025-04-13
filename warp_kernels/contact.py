import warp as wp
from .core import RigidBodyState, Model, dtype, device

@wp.kernel
def compute_contacts(state: RigidBodyState, model: Model):
    tid = wp.tid()

    pos = state.body_pos[tid]
    vel = state.body_vel[tid]
    rot = state.body_rot[tid]
    ang_vel = state.body_ang_vel[tid]
    radius = model.body_radius[tid]

    # Contact parameters
    ke = model.contact_ke # Stiffness
    kd = model.contact_kd # Damping
    mu = model.contact_mu # Friction coefficient

    # Accumulate forces/torques directly onto the state
    force_contact = wp.vec3(0.0, 0.0, 0.0)
    torque_contact = wp.vec3(0.0, 0.0, 0.0)

    # --- 1. Ground Contact (Plane y=offset) ---
    plane_normal = model.ground_plane_normal
    plane_offset = model.ground_plane_offset
    signed_dist = wp.dot(pos, plane_normal) - plane_offset - radius

    if signed_dist < 0.0:
        penetration_depth = -signed_dist
        contact_point_world = pos - plane_normal * radius # Point on sphere surface

        # Relative velocity at contact point (sphere surface velocity)
        point_vel_world = vel + wp.cross(ang_vel, contact_point_world - pos)

        # Normal velocity (towards the plane)
        normal_vel = wp.dot(point_vel_world, plane_normal)

        # --- Normal Force (Penalty + Damping) ---
        # Smoothed penalty force using pow for gradient stability near zero
        fn_magnitude = ke * wp.pow(penetration_depth, 1.5) - kd * normal_vel * wp.min(1.0, penetration_depth * 10.0) # Damping only when penetrating
        fn_magnitude = wp.max(fn_magnitude, 0.0) # Force is only repulsive
        normal_force = plane_normal * fn_magnitude

        force_contact += normal_force

        # --- Friction Force (Coulomb Approximation) ---
        # Tangential velocity
        vel_tangential = point_vel_world - plane_normal * normal_vel

        # Smoothed friction direction (opposite to tangential velocity)
        # Use a threshold to avoid division by zero and instability at low velocities
        vel_tang_norm = wp.length(vel_tangential)
        friction_dir = wp.select(vel_tang_norm > 1e-4, -vel_tangential / vel_tang_norm, wp.vec3(0.0))

        # Friction magnitude limit (Coulomb cone)
        # Smoothly ramp up friction force based on tangential velocity magnitude (tanh)
        friction_mag_limit = mu * fn_magnitude
        friction_mag = friction_mag_limit * wp.tanh(vel_tang_norm * 5.0) # Smooth factor 5.0

        friction_force = friction_dir * friction_mag
        force_contact += friction_force

        # Apply torque due to contact forces at the surface point
        torque_contact += wp.cross(contact_point_world - pos, normal_force + friction_force)

    # --- 2. Sphere-Sphere Contact (Example - simple, no broadphase) ---
    # Warning: O(n^2), needs optimization (hash grid) for many bodies
    # This is just a placeholder structure
    # for i in range(model.n_bodies):
    #     if i == tid: continue # Skip self-collision
    #     # Calculate distance, penetration, relative velocity, forces...
    #     # Add forces/torques similar to ground contact


    # Add accumulated contact forces/torques to the body's state
    # Use atomic add for safety if parallelizing contact pairs later
    wp.atomic_add(state.body_force, tid, force_contact)
    wp.atomic_add(state.body_torque, tid, torque_contact)