import numpy as np
import torch
import warp as wp
import gym # For spaces

from ..base_env import DiffEnv
from warp_kernels.core import Model, RigidBodyState, dtype, device as warp_device

# TODO: Define Ant morphology constants (num bodies, joint indices, link lengths etc.)
# These would typically be loaded from MJCF/URDF or hardcoded
NUM_ANT_BODIES = 9 # Example: 1 torso + 4*2 leg segments
TORSO_IDX = 0
# ... indices for leg parts ...
JOINT_INDICES = [(0, 1), (0, 3), (0, 5), (0, 7), # Hip joints
                 (1, 2), (3, 4), (5, 6), (7, 8)] # Knee joints
NUM_JOINTS = len(JOINT_INDICES) # 8 controllable joints


class AntEnv(DiffEnv):
    def __init__(self, device='cuda'):
        super().__init__(device)
        # Define action/observation spaces (adjust dims based on actual ant model)
        # Action: Torques for 8 joints
        action_dim = NUM_JOINTS
        action_high = np.ones(action_dim, dtype=np.float32) * 1.0 # Torque limits
        self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)

        # Observation: Torso orientation (quat), joint angles, joint velocities, torso velocity
        # Needs careful calculation from RigidBodyState
        obs_dim = 4 + NUM_JOINTS + NUM_JOINTS + 6 # Example dims
        obs_high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def _init_sim(self):
        n_bodies = NUM_ANT_BODIES

        # --- Create Model ---
        self.model = Model()
        self.model.n_bodies = n_bodies
        self.model.dt = 1.0 / 60.0 # Simulation frequency 60 Hz
        self.model.gravity = wp.vec3(0.0, -9.81, 0.0) # Gravity downwards in Y
        self.model.ground_plane_normal = wp.vec3(0.0, 1.0, 0.0)
        self.model.ground_plane_offset = 0.0 # Ground at y=0

        # Contact params
        self.model.contact_ke = 5000.0
        self.model.contact_kd = 100.0
        self.model.contact_mu = 0.7

        # Body properties (mass, inertia, shape) - Example values
        inv_mass = np.ones(n_bodies, dtype=np.float32) * 1.0 # 1kg mass bodies
        inv_mass[inv_mass > 0] = 1.0 / inv_mass[inv_mass > 0]
        inv_mass[TORSO_IDX] = 1.0 / 2.0 # Heavier torso
        self.model.body_inv_mass = wp.array(inv_mass, dtype=dtype, device=warp_device)

        # Simple sphere inertia: 2/5 * m * r^2 -> inv = 5/2 * inv_m / r^2
        radius = np.ones(n_bodies, dtype=np.float32) * 0.1 # 10cm radius spheres
        inertia_scalar = (2.0/5.0) * (1.0/inv_mass) * (radius**2) # Mass = 1/inv_mass
        inv_inertia_scalar = np.zeros_like(inertia_scalar)
        inv_inertia_scalar[inertia_scalar > 1e-9] = 1.0 / inertia_scalar[inertia_scalar > 1e-9]

        inv_inertia_diag = np.array([wp.mat33(inv_inertia_scalar[i], 0, 0, 0, inv_inertia_scalar[i], 0, 0, 0, inv_inertia_scalar[i]) for i in range(n_bodies)])
        self.model.body_inv_inertia = wp.array(inv_inertia_diag, dtype=wp.mat33, device=warp_device)
        self.model.body_radius = wp.array(radius, dtype=dtype, device=warp_device)

        # --- Create Initial State ---
        self.state = RigidBodyState()
        self.state.body_pos = wp.zeros((n_bodies, 3), dtype=wp.vec3, device=warp_device)
        self.state.body_rot = wp.zeros((n_bodies, 4), dtype=wp.quat, device=warp_device) # Identity quat = (0,0,0,1) in scalar-last convention? Warp uses scalar-last? Check docs. Let's assume xyzw
        self.state.body_vel = wp.zeros((n_bodies, 3), dtype=wp.vec3, device=warp_device)
        self.state.body_ang_vel = wp.zeros((n_bodies, 3), dtype=wp.vec3, device=warp_device)
        self.state.body_force = wp.zeros((n_bodies, 3), dtype=wp.vec3, device=warp_device)
        self.state.body_torque = wp.zeros((n_bodies, 3), dtype=wp.vec3, device=warp_device)

        # Keep a template for structure info
        self.state_template = RigidBodyState()
        self.state_template.body_pos = self.state.body_pos
        # ... copy structure for others

        self._reset_state() # Set initial positions/orientations


    def _reset_state(self):
        # Example: Place torso at (0, 0.5, 0), limbs folded
        n_bodies = self.model.n_bodies
        pos_np = np.zeros((n_bodies, 3), dtype=np.float32)
        rot_np = np.zeros((n_bodies, 4), dtype=np.float32) # xyzw
        vel_np = np.zeros((n_bodies, 3), dtype=np.float32)
        ang_vel_np = np.zeros((n_bodies, 3), dtype=np.float32)

        # Set initial torso pose
        pos_np[TORSO_IDX] = [0.0, 0.3, 0.0] # Lifted slightly off ground
        rot_np[TORSO_IDX] = [0.0, 0.0, 0.0, 1.0] # Identity quaternion (scalar last)

        # TODO: Set initial positions/rotations for limb segments based on joint angles

        # Copy to Warp arrays
        wp.copy(target=self.state.body_pos, source=wp.array(pos_np, dtype=wp.vec3, device=warp_device))
        wp.copy(target=self.state.body_rot, source=wp.array(rot_np, dtype=wp.quat, device=warp_device))
        wp.copy(target=self.state.body_vel, source=wp.array(vel_np, dtype=wp.vec3, device=warp_device))
        wp.copy(target=self.state.body_ang_vel, source=wp.array(ang_vel_np, dtype=wp.vec3, device=warp_device))
        wp.launch(kernel=wp.kernels.zero_vec, dim=n_bodies*3, inputs=[self.state.body_force.grad], device=warp_device) # Zero forces/torques
        wp.launch(kernel=wp.kernels.zero_vec, dim=n_bodies*3, inputs=[self.state.body_torque.grad], device=warp_device)


    def _get_observation(self) -> torch.Tensor:
        # --- Extract relevant info from self.state (Warp arrays) ---
        # Torso pose/velocity
        torso_pos = wp.to_torch(self.state.body_pos)[TORSO_IDX]
        torso_rot = wp.to_torch(self.state.body_rot)[TORSO_IDX] # xyzw
        torso_vel = wp.to_torch(self.state.body_vel)[TORSO_IDX]
        torso_ang_vel = wp.to_torch(self.state.body_ang_vel)[TORSO_IDX]

        # Joint angles/velocities (Requires calculating relative orientations/velocities)
        # This is complex and depends on the kinematic structure. Placeholder:
        joint_angles = torch.zeros(NUM_JOINTS, device=self.device)
        joint_velocities = torch.zeros(NUM_JOINTS, device=self.device)
        # TODO: Implement calculation based on self.state.body_rot and self.state.body_ang_vel

        # Concatenate into observation vector
        obs = torch.cat([
            torso_rot, # 4
            joint_angles, # 8
            joint_velocities, # 8
            torso_vel, # 3
            torso_ang_vel # 3
            # Maybe relative positions of feet, contact forces etc.
        ]).to(self.device)

        # Ensure obs dim matches self.observation_space.shape[0]
        return obs

    def _get_reward(self) -> float:
        # --- Calculate reward based on current state ---
        torso_pos = wp.to_torch(self.state.body_pos)[TORSO_IDX]
        torso_vel = wp.to_torch(self.state.body_vel)[TORSO_IDX]

        # Reward forward velocity (e.g., in x direction)
        forward_reward = torso_vel[0].item() * 1.0

        # Penalize control effort (requires action from previous step)
        # control_cost = np.sum(np.square(self.last_action)) * 0.01
        control_cost = 0 # Placeholder

        # Penalize deviation from upright (e.g., torso Z axis close to world Y axis)
        # torso_rot_mat = ... calculation from quat ...
        # upright_penalty = ...

        # Survival reward
        survival_reward = 0.1

        # TODO: Add contact cost?

        total_reward = forward_reward - control_cost + survival_reward
        return total_reward

    def _is_done(self) -> bool:
        # --- Check termination conditions ---
        torso_pos = wp.to_torch(self.state.body_pos)[TORSO_IDX]
        torso_rot = wp.to_torch(self.state.body_rot)[TORSO_IDX] # xyzw

        # Done if torso height is too low (fallen)
        height = torso_pos[1].item()
        if height < 0.1:
            return True

        # Done if torso orientation is too tilted (fallen)
        # Check angle between torso's local Y axis and world Y axis
        # q = torso_rot
        # up_vector_local = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        # R = ... quat_to_rot_mat(q) ...
        # up_vector_world = R @ up_vector_local
        # angle = torch.acos(torch.dot(up_vector_world, torch.tensor([0.0, 1.0, 0.0])))
        # if angle > np.pi / 3.0: # Fallen if tilted more than 60 degrees
        #      return True

        # TODO: Add other conditions (e.g., max episode length)

        return False

    def _compute_action_forces(self, action: np.ndarray) -> (torch.Tensor, torch.Tensor):
        """ Convert joint torques (action) into world-frame forces/torques on bodies. """
        n_bodies = self.model.n_bodies
        action_torques = torch.tensor(action, dtype=torch.float32, device=warp_device) # Action = joint torques

        # We need to apply these joint torques between the connected bodies
        # This requires knowing the joint axes in world frame and applying equal/opposite torques
        # This is non-trivial. Placeholder: Apply directly to child body (needs joint info)

        # Initialize world frame force/torque tensors
        forces_world = torch.zeros((n_bodies, 3), dtype=torch.float32, device=warp_device)
        torques_world = torch.zeros((n_bodies, 3), dtype=torch.float32, device=warp_device)

        # --- TODO: Implement proper joint torque application ---
        # For each joint `j` connecting body `p` (parent) and `c` (child):
        #   Get joint axis `a_w` in world frame (depends on parent orientation)
        #   Torque on child: `tau_c = action_torques[j] * a_w`
        #   Torque on parent: `tau_p = -action_torques[j] * a_w`
        #   Add these to torques_world[c] and torques_world[p]

        # Placeholder: Just apply torque to second body in joint pair for simplicity
        for j, (idx_p, idx_c) in enumerate(JOINT_INDICES):
             # Fake axis (needs proper calculation)
             axis = torch.tensor([0.0, 0.0, 1.0], device=warp_device)
             torque_on_child = action_torques[j] * axis
             # Apply torque to child body (this is incorrect physics but placeholder)
             torques_world[idx_c] += torque_on_child
             # Need to apply equal opposite torque to parent!
             torques_world[idx_p] -= torque_on_child


        return forces_world, torques_world