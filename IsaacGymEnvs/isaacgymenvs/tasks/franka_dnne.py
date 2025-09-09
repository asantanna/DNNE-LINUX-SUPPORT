# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp  
from isaacgymenvs.tasks.base.vec_task import VecTask


@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class FrankaDNNE(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        
        # DNNE environments must use single environment
        if self.cfg["env"]["numEnvs"] != 1:
            raise ValueError(f"DNNE environments must use numEnvs=1, got {self.cfg['env']['numEnvs']}")

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.franka_position_noise = self.cfg["env"]["frankaPositionNoise"]
        self.franka_rotation_noise = self.cfg["env"]["frankaRotationNoise"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Create dicts to pass to reward function
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_align_scale": self.cfg["env"]["alignRewardScale"],
            "r_stack_scale": self.cfg["env"]["stackRewardScale"],
        }

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "Invalid control type specified. Must be one of: {osc, joint_tor}"

        # dimensions - trust values from YAML/dnne_cfg
        # DNNE passes the correct observation/action sizes via configuration
        # Don't hardcode these values

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed
        self._init_target_state = None          # Initial state of target for the current env
        self._target_state = None                # Current state of target for the current env
        self._target_id = None                   # Actor ID corresponding to target for a given env
        self._debug_sphere_id = None            # Actor ID for debug sphere
        self._debug_sphere_state = None         # Current state of debug sphere

        # Tensor placeholders
        self._root_state = None             # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None                     # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self._contact_forces = None     # Contact forces in sim
        self._eef_state = None  # end effector state (at grasping point)
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None            # Position actions
        self._effort_control = None         # Torque actions
        self._franka_effort_limits = None        # Actuator effort limits for franka
        self._global_indices = None         # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Franka defaults
        self.franka_default_dof_pos = to_torch(
            [0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854, 0.035, 0.035], device=self.device
        )

        # OSC Gains
        self.kp = to_torch([150.] * 6, device=self.device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = to_torch([10.] * 7, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        #self.cmd_limit = None                   # filled in later

        # Set control limits - will be properly initialized after _franka_effort_limits is populated
        if self.control_type == "osc":
            self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0)
        else:
            # For joint_tor, check if we have custom torque limits in the config
            joint_control_cfg = self.cfg["env"].get("dnne", {}).get("joint_control", None)
            if joint_control_cfg and "torque_limits" in joint_control_cfg:
                # Use configured torque limits
                torque_limits = joint_control_cfg["torque_limits"]
                cmd_limits = []
                for i in range(7):
                    if i in torque_limits:
                        cmd_limits.append(torque_limits[i])
                    else:
                        # Default safe limit for joints not specified
                        cmd_limits.append(0.5 if i > 3 else 1.0)
                self.cmd_limit = to_torch(cmd_limits, device=self.device).unsqueeze(0)
                print(f"Using configured torque limits: {cmd_limits}")
            else:
                # Default: use small limits to prevent unstable behavior with untrained networks
                # These are much smaller than actual robot limits but safe for training
                self.cmd_limit = to_torch([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        # Read joint control configuration from YAML if available
        joint_control_cfg = self.cfg["env"].get("dnne", {}).get("joint_control", None)
        
        if joint_control_cfg:
            # Use YAML configuration for joint freezing
            controlled_joints = joint_control_cfg.get("controlled_joints", [0, 1, 2, 3, 4, 5, 6])
            freeze_mode = joint_control_cfg.get("freeze_mode", "position")
            freeze_pos_stiffness = joint_control_cfg.get("freeze_position_stiffness", 5000.0)
            freeze_pos_damping = joint_control_cfg.get("freeze_position_damping", 100.0)
            freeze_effort_damping = joint_control_cfg.get("freeze_effort_damping", 50.0)
            
            # New: Add friction to controlled joints to prevent drift
            controlled_joint_friction = joint_control_cfg.get("controlled_joint_friction", 0.0)
            
            # Initialize arrays based on configuration
            franka_dof_stiffness = []
            franka_dof_damping = []
            
            for i in range(9):  # 7 arm joints + 2 gripper joints
                if i < 7:  # Arm joints
                    if i in controlled_joints:
                        # Controlled joint - use effort mode with optional friction damping
                        franka_dof_stiffness.append(0.0)
                        franka_dof_damping.append(controlled_joint_friction)
                    else:
                        # Frozen joint - apply freeze settings
                        if freeze_mode == "position":
                            franka_dof_stiffness.append(freeze_pos_stiffness)
                            franka_dof_damping.append(freeze_pos_damping)
                        else:  # damped_effort mode
                            franka_dof_stiffness.append(0.0)
                            franka_dof_damping.append(freeze_effort_damping)
                else:  # Gripper joints (7, 8)
                    franka_dof_stiffness.append(5000.0)
                    franka_dof_damping.append(100.0)
            
            franka_dof_stiffness = to_torch(franka_dof_stiffness, dtype=torch.float, device=self.device)
            franka_dof_damping = to_torch(franka_dof_damping, dtype=torch.float, device=self.device)
            
            # Store configuration for later use
            self.controlled_joints = controlled_joints
            self.freeze_mode = freeze_mode
            self.controlled_joint_friction = controlled_joint_friction
            print(f"Joint control config: controlled={controlled_joints}, freeze_mode={freeze_mode}, friction={controlled_joint_friction}")
        else:
            # Default behavior if no joint control config
            franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
            franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)
            self.controlled_joints = list(range(7))
            self.freeze_mode = None

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        # Create target sphere asset instead of cubes
        self.target_radius = 0.05
        target_opts = gymapi.AssetOptions()
        target_opts.density = 0.001
        target_opts.disable_gravity = True
        target_opts.fix_base_link = True
        target_asset = self.gym.create_sphere(self.sim, self.target_radius, target_opts)
        target_color = gymapi.Vec3(1.0, 0.0, 0.0)  # Red target

        # Create debug sphere asset (visual only, no physics interactions)
        self.debug_sphere_radius = 0.01  # Small red sphere
        debug_sphere_opts = gymapi.AssetOptions()
        debug_sphere_opts.density = 0.001  # Very light
        debug_sphere_opts.disable_gravity = True
        debug_sphere_opts.fix_base_link = True
        debug_sphere_asset = self.gym.create_sphere(self.sim, self.debug_sphere_radius, debug_sphere_opts)
        debug_sphere_color = gymapi.Vec3(1.0, 0.0, 0.0)  # Red

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            # Set drive mode based on joint control configuration
            if i < 7:  # Arm joints
                if hasattr(self, 'controlled_joints') and self.freeze_mode == "position":
                    # Use position control for frozen joints, effort for controlled joints
                    if i in self.controlled_joints:
                        franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
                    else:
                        franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
                else:
                    # Default: all arm joints in effort mode
                    franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            else:  # Gripper joints (7, 8)
                franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        
        # Keep the safe training limits rather than using full effort limits
        # This prevents instability with untrained networks
        # if self.control_type == "joint_tor":
        #     self.cmd_limit = self._franka_effort_limits[:7].unsqueeze(0)
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        # Define start pose for franka
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for table stand
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)


        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 4     # 1 for table, table stand, target, debug sphere
        max_agg_shapes = num_franka_shapes + 4     # 1 for table, table stand, target, debug sphere

        self.frankas = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            # Potentially randomize start pose
            if self.franka_position_noise > 0:
                rand_xy = self.franka_position_noise * (-1. + np.random.rand(2) * 2.0)
                franka_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
            if self.franka_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.franka_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                franka_start_pose.r = gymapi.Quat(*new_quat)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create target sphere
            target_start_pose = gymapi.Transform()
            target_start_pose.p = gymapi.Vec3(0.0, 0.0, 1.3)  # Start at center, above table
            self._target_id = self.gym.create_actor(env_ptr, target_asset, target_start_pose, "target", i, 2, 0)
            # Set color
            self.gym.set_rigid_body_color(env_ptr, self._target_id, 0, gymapi.MESH_VISUAL, target_color)

            # Create debug sphere (visual only, hidden initially)
            debug_start_pose = gymapi.Transform()
            debug_start_pose.p = gymapi.Vec3(0.0, 0.0, -10.0)  # Start hidden below the scene
            # Use unique collision filter to prevent all interactions
            debug_sphere_filter = 0b1000  # Unique bit mask that doesn't overlap with others
            debug_sphere_group = 0  # Collision group
            self._debug_sphere_id = self.gym.create_actor(env_ptr, debug_sphere_asset, debug_start_pose, 
                                                         "debug_sphere", debug_sphere_group, debug_sphere_filter)
            
            # Set color to red
            self.gym.set_rigid_body_color(env_ptr, self._debug_sphere_id, 0, gymapi.MESH_VISUAL, debug_sphere_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # Setup init state buffer
        self._init_target_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_leftfinger_tip"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_rightfinger_tip"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_grip_site"),
            # Target
            "target_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._target_id, "sphere0"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print(f"[DEBUG] Total DOFs per environment: {self.num_dofs}")

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['panda_hand_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]
        self._target_state = self._root_state[:, self._target_id, :]
        self._debug_sphere_state = self._root_state[:, self._debug_sphere_id, :]

        # Initialize states
        self.states.update({
            # No cube sizes needed for target task
        })

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]

        # Initialize indices (5 actors: franka, table, table_stand, target, debug_sphere)
        self._global_indices = torch.arange(self.num_envs * 5, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        self.states.update({
            # Franka
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
            # Target
            "target_pos": self._target_state[:, :3],
            "target_quat": self._target_state[:, 3:7],
            "target_to_eef": self._target_state[:, :3] - self._eef_state[:, :3],
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        # DNNE doesn't use rewards - handled externally
        # Check for target reached (gripper close to target)
        target_distance = torch.norm(self.states["target_to_eef"], dim=-1)
        target_reached = target_distance < 0.1  # Within 10cm of target
        
        # Check for gripper-table collision (gripper too close to table)
        gripper_z = self.states["eef_pos"][:, 2]  # Z position of gripper
        table_surface_z = self._table_surface_pos[2]  # Table surface height (set in _create_envs)
        gripper_safety_margin = 0.05  # 5cm safety margin above table
        gripper_too_low = gripper_z < (table_surface_z + gripper_safety_margin)
        
        # Debug output when target is reached
        if target_reached.any() and not hasattr(self, '_last_target_reached'):
            self._last_target_reached = False
        if target_reached.any() and not self._last_target_reached:
            print(f"[DNNE] Target reached! Distance: {target_distance[0]:.3f}m")
            self._last_target_reached = True
        elif not target_reached.any():
            self._last_target_reached = False
        
        # Debug output when gripper hits table
        if gripper_too_low.any() and not hasattr(self, '_last_gripper_low'):
            self._last_gripper_low = False
        if gripper_too_low.any() and not self._last_gripper_low:
            print(f"[DNNE] Gripper too close to table! Height: {gripper_z[0]:.3f}m, Table: {table_surface_z:.3f}m")
            self._last_gripper_low = True
        elif not gripper_too_low.any():
            self._last_gripper_low = False
        
        # Set reset for timeout OR target reached OR gripper-table collision
        self.reset_buf = torch.where(
            (self.progress_buf >= self.max_episode_length - 1) | target_reached | gripper_too_low,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )

    def compute_observations(self):
        self._refresh()
        
        # For DNNE: target_pos (3), eef_pos (3), eef_quat (4), joints/gripper (2 or 7), time (1)
        obs = ["target_pos", "eef_pos", "eef_quat"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        
        # Add episode time
        episode_time = (self.progress_buf.float() * self.dt).unsqueeze(-1)
        
        obs_list = [self.states[ob] for ob in obs]
        obs_list.append(episode_time)
        
        self.obs_buf = torch.cat(obs_list, dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Reset target to random position in reachable shell
        self._reset_target_state(env_ids)
        
        # Write the new init state to the sim state
        self._target_state[env_ids] = self._init_target_state[env_ids]

        # Reset agent
        reset_noise = torch.rand((len(env_ids), 9), device=self.device)
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0), self.franka_dof_upper_limits)

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # Update target states (target is actor 3, debug_sphere is actor 4)
        multi_env_ids_target_int32 = self._global_indices[env_ids, 3:4].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_target_int32), len(multi_env_ids_target_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _reset_target_state(self, env_ids):
        """
        Reset target to random position within reachable shell for specified environments.

        
        Args:
            env_ids (tensor): Specific environments to reset target for
        """
        num_resets = len(env_ids)
        
        # Sample random positions in a reachable shell
        # Shell parameters (similar to old code)
        shell_center = torch.tensor([0.0, 0.0, 1.3], device=self.device)  
        inner_radius = 0.3
        outer_radius = 0.6

        # Table surface height: table at z=1.0 with thickness 0.05 + safety margin
        min_z_above_table = 1.05 + 0.1  # 10cm above table surface
        
        # Sample spherical coordinates
        r = torch.rand(num_resets, device=self.device) * (outer_radius - inner_radius) + inner_radius
        
        # FIX: Limit theta to ensure target stays above table
        # Previously theta could go from 0 to pi, allowing targets below the table
        # Now we calculate max_theta to ensure z >= min_z_above_table
        # Math: z = r * cos(theta) + shell_center[2] >= min_z_above_table
        #       cos(theta) >= (min_z_above_table - shell_center[2]) / r
        # Use outer_radius for safety since r varies
        cos_theta_min = (min_z_above_table - shell_center[2]) / outer_radius
        # Clamp to valid cosine range
        cos_theta_min = max(min(cos_theta_min, 1.0), -1.0)  # Avoid tensor creation warning
        max_theta = torch.acos(torch.tensor(cos_theta_min, device=self.device))
        
        theta = torch.rand(num_resets, device=self.device) * max_theta  # 0 to max_theta for upper hemisphere
        phi = torch.rand(num_resets, device=self.device) * 2 * np.pi  # 0 to 2pi
        
        # Convert to Cartesian
        x = r * torch.sin(theta) * torch.cos(phi) + shell_center[0]
        y = r * torch.sin(theta) * torch.sin(phi) + shell_center[1]
        z = r * torch.cos(theta) + shell_center[2]
        
        # Set target positions
        self._init_target_state[env_ids, 0] = x
        self._init_target_state[env_ids, 1] = y
        self._init_target_state[env_ids, 2] = z
        
        # Set identity quaternion (no rotation)
        self._init_target_state[env_ids, 3:7] = torch.tensor([0., 0., 0., 1.], device=self.device)
        
        # Zero velocities
        self._init_target_state[env_ids, 7:] = 0
        
        # Apply the target state to the simulation
        self._target_state[env_ids] = self._init_target_state[env_ids]

    def _compute_osc_torques(self, dpose):
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :7], self._qd[:, :7]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"]).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, 7:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = tensor_clamp(u.squeeze(-1),
                         -self._franka_effort_limits[:7].unsqueeze(0), self._franka_effort_limits[:7].unsqueeze(0))

        return u

    def pre_physics_step(self, actions, extra_args=None):
        self.actions = actions.clone().to(self.device)
        
        # Handle debug sphere visualization if requested
        if extra_args and "debug_sphere_pos" in extra_args:
            pos = extra_args["debug_sphere_pos"]
            # Update debug sphere position
            self._debug_sphere_state[0, 0:3] = torch.tensor(pos, device=self.device, dtype=torch.float32)
            # Keep orientation as identity quaternion
            self._debug_sphere_state[0, 3:7] = torch.tensor([0., 0., 0., 1.], device=self.device)
            # Zero velocities
            self._debug_sphere_state[0, 7:] = 0
            
            # Apply state update to simulation (debug_sphere is actor 4)
            multi_env_ids_debug = self._global_indices[0, 4].unsqueeze(0).to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim, 
                gymtorch.unwrap_tensor(self._root_state),
                gymtorch.unwrap_tensor(multi_env_ids_debug), 
                1
            )

        # Handle different action formats based on control type and action size
        if self.actions.shape[1] == 8:
            # 8 actions: 7 arm + 1 gripper (original format)
            u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]
        elif self.actions.shape[1] == 7:
            if self.control_type == "osc":
                # 7 actions for OSC: 6 arm + 1 gripper
                u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]
            else:
                # 7 actions for joint_tor: all 7 are arm joints, no gripper
                u_arm = self.actions
                u_gripper = torch.zeros(self.actions.shape[0], device=self.device)
        else:
            raise ValueError(f"Unexpected action size: {self.actions.shape[1]}")

        # print(u_arm, u_gripper)
        # print(self.cmd_limit, self.action_scale)

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
        self._arm_control[:, :] = u_arm

        # Control gripper
        u_fingers = torch.zeros_like(self._gripper_control)
        u_fingers[:, 0] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-2].item(),
                                      self.franka_dof_lower_limits[-2].item())
        u_fingers[:, 1] = torch.where(u_gripper >= 0.0, self.franka_dof_upper_limits[-1].item(),
                                      self.franka_dof_lower_limits[-1].item())
        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:, :] = u_fingers

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def reset(self):
        """DNNE: Manual reset for all environments"""
        # Reset all environments
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)
        
        # Don't compute observations - they're meaningless after reset
        # The next step will provide proper observations
        
        # Return empty dict (observations will come from next step)
        return {"obs": torch.zeros_like(self.obs_buf)}

    def post_physics_step(self):
        self.progress_buf += 1

        # DNNE: Auto-reset disabled - environment will be reset manually via trigger
        # or by IsaacGymSim node when reset_when_done=True
        # env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(env_ids) > 0:
        #     self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            target_pos = self.states["target_pos"]
            target_rot = self.states["target_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, target_pos), (eef_rot, target_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

#####################################################################
###=========================jit functions=========================###
#####################################################################


# Removed compute_franka_reward - DNNE doesn't use rewards
