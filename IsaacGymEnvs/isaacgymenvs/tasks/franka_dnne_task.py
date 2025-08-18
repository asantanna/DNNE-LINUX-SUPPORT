"""
Franka DNNE Environment Module

This module provides the resolver function and specific subtask implementations
for the Franka DNNE environment. The base class is in franka_dnne/franka_dnne_base.py.
"""

from .franka_dnne.franka_dnne_base import Franka_DNNE_Base
import torch
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch, quat_mul, tensor_clamp
import os
import numpy as np


class Franka_DNNE_RandomTarget(Franka_DNNE_Base):
    """
    Franka DNNE Random Target Subtask
    
    Replaces the cube stacking task with a simple "touch the target" task
    using a single sphere as the target. The target position is randomized
    at each reset. This is designed for DNNE experimentation where rewards
    and termination are controlled externally.
    """
    
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        # Override the observation space to include target position instead of cube positions
        cfg["env"]["numObservations"] = 16 if cfg["env"]["controlType"] == "osc" else 23
        
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        
        # Target sphere properties
        self.target_radius = 0.03  # 3cm radius sphere
        self.target_height_range = [0.1, 0.5]  # Height range for target placement
        self.target_xy_range = 0.3  # Max distance from origin in x,y plane
        
    def _create_envs(self, num_envs, spacing, num_per_row):
        """Override to create sphere target instead of cubes."""
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # Load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        # Create sphere asset for target
        sphere_opts = gymapi.AssetOptions()
        sphere_opts.density = 0.001  # Very light so it doesn't affect physics much
        sphere_opts.disable_gravity = True  # Float in place
        sphere_opts.fix_base_link = True  # Fixed position until we move it
        target_radius = 0.05  # Default target radius
        target_asset = self.gym.create_sphere(self.sim, target_radius, sphere_opts)

        franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        # Get Franka DOF properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        
        # Use position control for the first 7 joints, effort control for grippers
        for i in range(7):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
            franka_dof_props['damping'][i] = franka_dof_damping[i]
        
        # Grippers
        franka_dof_props['driveMode'][7] = gymapi.DOF_MODE_EFFORT
        franka_dof_props['driveMode'][8] = gymapi.DOF_MODE_EFFORT
        franka_dof_props['stiffness'][7] = franka_dof_stiffness[7]
        franka_dof_props['stiffness'][8] = franka_dof_stiffness[8]
        franka_dof_props['damping'][7] = franka_dof_damping[7]
        franka_dof_props['damping'][8] = franka_dof_damping[8]

        # Create environments
        self.frankas = []
        self.envs = []
        self.targets = []  # Replace cubeA and cubeB with single target
        
        # Aggregate settings
        self.aggregate_mode = self.cfg["env"].get("aggregateMode", 0)
        max_agg_bodies = self.cfg["env"].get("maxAggBodies", 30)
        max_agg_shapes = self.cfg["env"].get("maxAggShapes", 30)
        
        for i in range(self.num_envs):
            # Create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            
            if self.aggregate_mode >= 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            
            # Create franka actor
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, gymapi.Transform(), "franka", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)
                
            # Create target sphere actor
            target_pose = gymapi.Transform()
            target_pose.p = gymapi.Vec3(0.0, 0.0, 0.3)  # Initial position
            target_actor = self.gym.create_actor(env_ptr, target_asset, target_pose, "target", i, 0, 0)
            
            # Set target color to red for visibility
            color = gymapi.Vec3(1.0, 0.0, 0.0)
            self.gym.set_rigid_body_color(env_ptr, target_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
            
            if self.aggregate_mode >= 2:
                self.gym.end_aggregate(env_ptr)
                
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.targets.append(target_actor)
            
        # Store number of DOFs
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)
        
        # Configure DOF properties
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        
        for i in range(self.num_franka_dofs):
            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)

    def _init_data(self):
        """Initialize data structures for target instead of cubes."""
        # Get initial target state
        target = self.gym.get_actor_rigid_body_states(self.envs[0], self.targets[0], gymapi.STATE_ALL)
        
        # Initialize target state tensors
        self._init_target_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_target_state[:, :3] = to_torch([0.0, 0.0, 0.3], device=self.device)  # Position
        self._init_target_state[:, 3:7] = to_torch([0.0, 0.0, 0.0, 1.0], device=self.device)  # Quaternion
        
        # Get indices for target
        self._target_id = self.gym.get_actor_rigid_body_handle(self.envs[0], self.targets[0], 0)
        
        # Current target state
        self._target_state = self._init_target_state.clone()
        
    def compute_observations(self):
        """Compute observations including target position and episode time."""
        # Check if target state is initialized
        if not hasattr(self, '_target_state') or self._target_state is None:
            # Return zeros if not initialized yet
            self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
            return self.obs_buf
            
        self._refresh()
        
        # Calculate episode elapsed time in seconds
        episode_elapsed_seconds = self.progress_buf.float() * self.dt
        
        # Observations: target_pos (3), eef_pos (3), eef_quat (4), gripper (2 or 7), episode_time (1)
        obs_list = [
            self._target_state[:, :3],  # Target position
            self._eef_state[:, :3],  # End effector position  
            self._eef_state[:, 3:7],  # End effector quaternion
        ]
        
        if self.control_type == "osc":
            obs_list.append(self._q[:, 7:9])  # Gripper joints only
        else:
            obs_list.append(self._q[:, :7])  # All arm joints
            
        obs_list.append(episode_elapsed_seconds.unsqueeze(-1))  # Episode time
        
        self.obs_buf = torch.cat(obs_list, dim=-1)
        
        return self.obs_buf
        
    def _reset_franka_arm(self, env_ids):
        """Reset just the franka arm position without cubes"""
        # Check if we're being called during initialization (before tensors exist)
        if not hasattr(self, '_dof_state') or self._dof_state is None:
            return
            
        # print(f"[DEBUG _reset_franka_arm] Resetting franka arm for env_ids={env_ids}")
        
        # The base class stores DOF state in self._dof_state tensor
        # For single environment, franka is the only actor with DOFs
        
        # Reset to default joint positions
        default_pos = torch.tensor([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04], 
                                  device=self.device)
        
        # Apply position to DOF state (pos is index 0, vel is index 1)
        self._dof_state[env_ids, :, 0] = default_pos
        self._dof_state[env_ids, :, 1] = 0  # Zero velocity
        
        # Apply the reset to simulation
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids.to(torch.int32)),
            len(env_ids)
        )

    def reset_idx(self, env_ids):
        """Reset specified environments with randomized target positions."""
        # Check if we're being called during initialization (before tensors exist)
        if not hasattr(self, '_root_state') or self._root_state is None:
            return
            
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        
        # Since DNNE uses single environment, we expect env_ids to be [0]
        if len(env_ids) != 1 or env_ids[0] != 0:
            raise ValueError(f"DNNE reset expects single environment, got env_ids={env_ids}")
        
        # Reset franka to default position WITHOUT cubes
        # Don't call super().reset_idx() as it tries to reset cubes we don't have
        # Instead, just reset the franka arm state directly
        self._reset_franka_arm(env_ids)
        
        # Randomize target position for the single environment
        target_pos = torch.zeros((1, 3), device=self.device)
        target_pos[0, 0] = (torch.rand(1, device=self.device) * 2 - 1) * self.target_xy_range
        target_pos[0, 1] = (torch.rand(1, device=self.device) * 2 - 1) * self.target_xy_range
        target_pos[0, 2] = torch.rand(1, device=self.device) * \
                          (self.target_height_range[1] - self.target_height_range[0]) + \
                          self.target_height_range[0]
        
        # Update target state for single environment
        self._target_state[0, :3] = target_pos[0]
        self._target_state[0, 3:7] = to_torch([0.0, 0.0, 0.0, 1.0], device=self.device)
        self._target_state[0, 7:] = 0  # Zero velocity
        
        # Apply state to simulation for the single target
        self._root_state[1] = self._target_state[0]  # Target is the second actor
        
        target_indices = torch.tensor([1], dtype=torch.int32, device=self.device)  # Target actor index
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(target_indices),
            1
        )
        
        # Reset progress buffer for single environment
        self.progress_buf[0] = 0
        self.reset_buf[0] = 0
        

def resolve_franka_dnne(cfg, *args, **kwargs):
    """
    Resolver function for Franka DNNE environments.
    
    Selects the appropriate subtask based on the configuration.
    """
    subtask_name = cfg.get("env", {}).get("subtask", "random_target")
    
    subtask_map = {
        "random_target": Franka_DNNE_RandomTarget,
        # Add more subtasks here as they are developed
        # "reach_pose": Franka_DNNE_ReachPose,
        # "trajectory_follow": Franka_DNNE_TrajectoryFollow,
    }
    
    if subtask_name not in subtask_map:
        raise NotImplementedError(f"Franka DNNE subtask '{subtask_name}' not implemented. Available: {list(subtask_map.keys())}")
    
    return subtask_map[subtask_name](cfg, *args, **kwargs)


# Export the resolver as the main class
FrankaDNNE = resolve_franka_dnne