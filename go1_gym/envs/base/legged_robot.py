# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import os
from typing import Dict, List
from collections import deque
import wandb

from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

assert gymtorch
import torch

from go1_gym import MINI_GYM_ROOT_DIR
from go1_gym.envs.base.base_task import BaseTask
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from go1_gym.utils.terrain import Terrain, perlin
from go1_gym.utils.parkour_terrain import Terrain as ParkourTerrain
from .legged_robot_config import Cfg

TRANSFORM_BASE_ARM_X = 0.2
TRANSFORM_BASE_ARM_Z = 0.1585
DEFAULT_BASE_HEIGHT = 0.78
INDEX_EE_POS_RADIUS_CMD = 15
INDEX_EE_POS_PITCH_CMD = 16
INDEX_EE_POS_YAW_CMD = 17
INDEX_EE_POS_TIMING_CMD = 18

def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

class LeggedRobot(BaseTask):
    def __init__(self, cfg: Cfg, sim_params, physics_engine, sim_device, headless,
                 initial_dynamics_dict=None, terrain_props=None, custom_heightmap=None):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """

        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self.initial_dynamics_dict = initial_dynamics_dict
        self.terrain_props = terrain_props
        self.custom_heightmap = custom_heightmap
        self._parse_cfg(self.cfg)

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()

        self._prepare_reward_function()
        self.init_done = True
        self.record_now = False
        self.collecting_evaluation = False
        self.num_still_evaluating = 0
        self.global_counter = 0 # parkour
        
        self.torques_substeps_list = []
        self.velocities_substeps_list = []
        self.torques_steps_list = []
        self.velocities_steps_list = []
        
        
        self.unclipped_torques = torch.zeros((self.num_envs, 12), device=self.device) 

        self.past_joint_pos_errs = torch.zeros((2*self.cfg.control.decimation, self.num_envs, 12), device=self.device)
        self.past_joint_vels = torch.zeros((2*self.cfg.control.decimation, self.num_envs, 12), device=self.device)
        self.energies = torch.zeros((self.num_envs, 12, 3), device=self.device)

    def pre_physics_step(self):
        # self.initialize_reward_bufs()
        self.torques_substeps_list = []
        self.velocities_substeps_list = []
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()
        self.render_gui()

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        if self.cfg.domain_rand.action_delay:
            if self.global_counter % self.cfg.domain_rand.delay_update_global_steps == 0:
                if len(self.cfg.domain_rand.action_curr_step) != 0:
                    self.delay = torch.tensor(self.cfg.domain_rand.action_curr_step.pop(0), device=self.device, dtype=torch.float)
            if self.viewer:
                self.delay = torch.tensor(self.cfg.domain_rand.action_delay_view, device=self.device, dtype=torch.float)
            indices = -self.delay -1
            actions = self.action_history_buf[:, indices.long()] # delay for 1/50=20ms

        self.global_counter += 1

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame

        self.pre_physics_step()
        
        for _ in range(self.cfg.control.decimation):

            # Create an energies_list variable
            if self.cfg.commands.control_only_z1:
                self.actions[:, :12] = torch.zeros((self.num_envs, 12), dtype=torch.float32, device=self.device)
                # self.dof_vel[:, :12] = torch.zeros((self.num_envs, 12), dtype=torch.float32, device=self.device)
                # self.dof_pos[:, :12] = self.default_dof_pos[:, :12]
                self.actions[:, 18] = 0.0 # set gripper to 0
                # self.dof_vel[:, 18] = 0.0
                # self.dof_pos[:, 18] = self.default_dof_pos[:, 18]
                # self.actions[:, 17] = 0.0

            self.last_torques = self.torques.clone()
            self.torques[:, :self.num_actuated_dof] = self._compute_torques(self.actions).view(self.torques[:, :self.num_actuated_dof].shape)

            # self.torques[:, :12] = torch.zeros((self.num_envs, 12), dtype=torch.float32, device=self.device)
            # self.torques[:, 18] = 0.0

            if self.ball_force_feedback is not None:
                asset_torques, asset_forces = self.ball_force_feedback()
                if asset_torques is not None:
                    self.torques[:, self.num_actuated_dof:] = asset_torques
                if asset_forces is not None:
                    self.forces[:, self.num_bodies:self.num_bodies + self.num_object_bodies] = asset_forces

            if self.cfg.domain_rand.randomize_ball_drag:
                self._apply_drag_force(self.forces)

            # self.forces = self._apply_ext_forces(self.forces)
            
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.GLOBAL_SPACE)
            self.gym.simulate(self.sim)
            # if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            
            # Store torques and velocities at substeps
            # TODO (srinathm): Should these be clipped?
            clip_obs = self.cfg.normalization.clip_observations
            if self.cfg.env.save_torques:
                self.torques_substeps_list.append(self.torques.clone())
                # self.torques_substeps_list.append(torch.clip(self.torques, -clip_obs, clip_obs))
            if self.cfg.env.save_velocities:
                self.velocities_substeps_list.append(self.dof_vel.clone())
                # self.velocities_substeps_list.append(torch.clip(self.dof_vel, -clip_obs, clip_obs))
            # Add the reward value to the buffer
            # self.compute_substep_reward()
            
            self.base_pos[:] = self.root_states[self.robot_actor_idxs, 0:3]
            self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                          )[:, self.feet_indices, 7:10]
            self.foot_positions = self.rigid_body_state.view(self.num_envs,             self.num_bodies, 13)[:, self.feet_indices,
                                0:3]
            
        self.post_physics_step()

        if self.cfg.env.save_torques:
            self.torques_substeps_list = torch.stack(self.torques_substeps_list)
        if self.cfg.env.save_velocities:
            self.velocities_substeps_list = torch.stack(self.velocities_substeps_list)

        self.estimate_energies()
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _update_goals(self):
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
        self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0

        self.reached_goal_ids = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1) < self.cfg.env.next_goal_threshold
        self.reach_goal_timer[self.reached_goal_ids] += 1

        self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]

        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])

        norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
        self.next_target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
    
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.record_now:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        
        self._update_path_distance()

        # print(f" base position world = {self.rigid_body_state[0, 0, 0:3]}")
        # print(f" FL foot position world = {self.rigid_body_state[0, 4, 0:3]}")       
        # print(f" link1 position world = {self.rigid_body_state[0, 17, 0:3]}")

        # link1_pos_base = quat_rotate_inverse(self.base_quat, self.rigid_body_state[:, 17, 0:3] - self.rigid_body_state[:, 0, 0:3])
        # print(f" link1 position body = {link1_pos_base[0,:]}")

        # print(f" link1 position body = {self.rigid_body_state[0, 17, 0:3]}")

        # prepare quantities
        self.base_pos[:] = self.root_states[self.robot_actor_idxs, 0:3]
        self.base_quat[:] = self.root_states[self.robot_actor_idxs, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                          )[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs,             self.num_bodies, 13)[:, self.feet_indices,
                              0:3]
        
        if self.cfg.env.add_objects:
            # self.randomize_ball_state()

            # check for a big change
            too_big_change_envs = torch.norm(self.object_pos_world_frame - self.root_states[self.object_actor_idxs, 0:3], dim=1) > 0.5
            self.too_big_change_envs = torch.logical_and(too_big_change_envs, ~self.reset_buf)
            too_big_change_object_ids = self.object_actor_idxs[too_big_change_envs]
            too_big_change_robot_ids = self.robot_actor_idxs[too_big_change_envs]
            
            self.root_states[too_big_change_object_ids, :] = self.object_init_state
            new_pos = self.root_states[too_big_change_robot_ids, 0:3] + quat_apply_yaw(self.root_states[too_big_change_robot_ids,3:7], self.root_states[too_big_change_object_ids, 0:3])
            self.root_states[too_big_change_object_ids, :3] = new_pos
            
            too_big_change_object_ids_int32 = too_big_change_object_ids.to(dtype = torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(too_big_change_object_ids_int32), len(too_big_change_object_ids_int32))
            self.gym.refresh_actor_root_state_tensor(self.sim)
            

            self.object_pos_world_frame = self.root_states[self.object_actor_idxs, 0:3].clone() 
            robot_object_vec = self.asset.get_local_pos()
            true_object_local_pos = quat_rotate_inverse(self.base_quat, robot_object_vec)
            true_object_local_pos[:, 2] = 0.0*torch.ones(self.num_envs, dtype=torch.float,
                                       device=self.device, requires_grad=False)

            # simulate observation delay
            self.object_local_pos = self.simulate_ball_pos_delay(true_object_local_pos, self.object_local_pos)
            self.object_lin_vel = self.asset.get_lin_vel()
            self.object_ang_vel = self.asset.get_ang_vel()
        # print("object linear velocity: ", self.object_lin_vel)

        # parkour
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        self._update_goals() 

        self._post_physics_step_callback()

        # # update the diff history
        # self.x_vel_diff_history[:, 1:] = self.x_vel_diff_history[:, :-1]
        # self.x_vel_diff_history[:, 0] = self.commands[:, 0] - self.base_lin_vel[:, 0]
        # self.y_vel_diff_history[:, 1:] = self.y_vel_diff_history[:, :-1]
        # self.y_vel_diff_history[:, 0] = self.commands[:, 1] - self.base_lin_vel[:, 1]
        # self.yaw_vel_diff_history[:, 1:] = self.yaw_vel_diff_history[:, :-1]
        # self.yaw_vel_diff_history[:, 0] = self.commands[:, 2] - self.base_ang_vel[:, 2]

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        # self.finalize_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        # parkour
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

        self.compute_observations()

        if "z1" in self.cfg.robot.name:
            # Compute ee position commands for the next time step
            self.compute_intermediate_ee_pos_command(env_ids)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:, :] = self.actions[:, :self.num_actuated_dof]
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target[:]
        self.last_joint_pos_target[:] = self.joint_pos_target[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[self.robot_actor_idxs, 7:13]

        self._render_headless()

        # parkour
        if self.viewer:
            self._draw_goals()


    def get_measured_ee_pos_spherical(self) -> torch.Tensor:
        '''
        Get the current ee position in the arm frame in spherical coordinates 

        Returns:
            - radius, pitch, yaw (size: (num_envs, 3))
        '''
        # Get gripper cartesian coordinates in world frame
        ee_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_actor_handles[0], "gripperMover")
        ee_position_world = self.rigid_body_state[:, ee_idx, 0:3].view(self.num_envs, 3) # env.rigid_body_state.shape = (num_envs, num_rigid_bodies, 13) = (1, 24, 13)
        
        base_quat_world = self.base_quat.view(self.num_envs,4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        base_quat_world_indep = quat_from_euler_xyz(base_rpy_world[:, 0], base_rpy_world[:, 1], base_rpy_world[:, 2])

        # Measured ee position in the base frame
        ee_position_base = quat_rotate_inverse(base_quat_world_indep, ee_position_world - self.base_pos[:, :3].view(self.num_envs, 3)).view(self.num_envs,3)

        # Measured ee position in the arm frame in cartesian coordinates 
        ee_position_arm = torch.zeros_like(ee_position_base)
        ee_position_arm[:,0] = ee_position_base[:,0].add_(-TRANSFORM_BASE_ARM_X)
        ee_position_arm[:,1] = ee_position_base[:,1]
        ee_position_arm[:,2] = ee_position_base[:,2].add_(-TRANSFORM_BASE_ARM_Z)

        # Spherical to cartesian coordinates in the arm base frame 
        radius_cmd = torch.norm(ee_position_arm, dim=1).view(self.num_envs,1)
        pitch_cmd = -torch.asin(ee_position_arm[:,2].view(self.num_envs,1)/radius_cmd).view(self.num_envs,1)
        yaw_cmd = torch.atan2(ee_position_arm[:,1].view(self.num_envs,1), ee_position_arm[:,0].view(self.num_envs,1)).view(self.num_envs,1)
        ee_pos_sphe_arm = torch.cat((radius_cmd, pitch_cmd, yaw_cmd), dim=1).view(self.num_envs,3)


        return ee_pos_sphe_arm

    def set_teleop_value(self, value:float):
        self.teleop_value = value*torch.ones_like(self.teleop_value, device=self.device)

    def set_gripper_teleop_value(self, value:float):
        self.teleop_gripper_value = value*torch.ones_like(self.teleop_gripper_value)

    def set_joint6_teleop_value(self, value:float):
        self.teleop_joint6_value = value*torch.ones_like(self.teleop_joint6_value)

    def set_trajectory_time(self, value:float):
        self.trajectory_time = value*torch.ones_like(self.trajectory_time)

    def set_initial_ee_pos(self):
        self.initial_ee_pos = self.get_measured_ee_pos_spherical()

    def compute_intermediate_ee_pos_command(self, env_ids):
        '''
        According to Deepak's paper:
            - The ee position commands in spherical coordinates (radius, pitch, yaw)

        Args:
            env_ids (list[int]): List of environment ids which have been reset

        self.reset_buf starts with 1's -> every env is reset at the very beginning -> ee_target_pos_cmd is defined 
        '''

        # Init current and target ee positions only once for all envs
        if self.init_training:
            self.init_training = False

            # Define the new long term target ee position command 
            self.ee_target_pos_cmd = self.commands[:, INDEX_EE_POS_RADIUS_CMD:(INDEX_EE_POS_YAW_CMD+1)].view(self.num_envs, 3).clone() # radius, pitch, yaw

            # Define the first ee position
            self.initial_ee_pos = self.get_measured_ee_pos_spherical()
        
        # When some envs have been reset, a new target ee position command is resampled in reset_idx, need to update first ee pos and target ee pos  
        if len(env_ids) > 0:
            # print("--reset")
            # Reset trajectory time 
            self.trajectory_time[env_ids] = 0.0

            # Define the new long term target ee position command 
            self.ee_target_pos_cmd[env_ids] = self.commands[env_ids, INDEX_EE_POS_RADIUS_CMD:(INDEX_EE_POS_YAW_CMD+1)].view(len(env_ids), 3) # radius, pitch, yaw

            # Define the first ee position
            self.initial_ee_pos[env_ids] = self.get_measured_ee_pos_spherical()[env_ids, :]

        # Interpolate intermediary ee position commands 
        T_traj = self.commands[:, INDEX_EE_POS_TIMING_CMD] # size num_envs

        # Make sure that the interpolated target ee position saturates after T_traj
        
        env_ids_inter = (self.trajectory_time.view(self.num_envs) < T_traj).nonzero(as_tuple=False).flatten()

        self.commands[:, INDEX_EE_POS_RADIUS_CMD:(INDEX_EE_POS_YAW_CMD+1)] = self.ee_target_pos_cmd.view(self.num_envs,3)

        if self.cfg.commands.interpolate_ee_cmds:
            if len(env_ids_inter):
                new_command = self.trajectory_time.view(self.num_envs,1)/T_traj.view(self.num_envs,1)*self.ee_target_pos_cmd.view(self.num_envs,3) + (1 - self.trajectory_time.view(self.num_envs,1)/T_traj.view(self.num_envs,1))*self.initial_ee_pos.view(self.num_envs,3)
                self.commands[env_ids_inter, INDEX_EE_POS_RADIUS_CMD:(INDEX_EE_POS_YAW_CMD+1)] = new_command[env_ids_inter, :]

        # Increase the time 
        self.trajectory_time += self.dt

        # print("self.commands[0, 15] =", self.commands[0, INDEX_EE_POS_RADIUS_CMD])

    def randomize_ball_state(self):
        reset_ball_pos_mark = np.random.choice([True, False],self.num_envs, p=[self.cfg.object.pos_reset_prob,1-self.cfg.object.pos_reset_prob])
        reset_ball_pos_env_ids = torch.tensor(np.array(np.nonzero(reset_ball_pos_mark)), device = self.device).flatten()# reset_ball_pos_mark.nonzero(as_tuple=False).flatten()
        ball_pos_env_ids = self.object_actor_idxs[reset_ball_pos_env_ids].to(device=self.device)
        reset_ball_pos_env_ids_int32 = ball_pos_env_ids.to(dtype = torch.int32)
        self.root_states[ball_pos_env_ids,0:3] += 2*(torch.rand(len(ball_pos_env_ids), 3, dtype=torch.float, device=self.device,
                                                     requires_grad=False)-0.5) * torch.tensor(self.cfg.object.pos_reset_range,device=self.device,
                                                     requires_grad=False)
        # self.root_states[ball_pos_env_ids, 2] = 0.08* torch.ones(len(ball_pos_env_ids), device = self.device, requires_grad=False)
        # self.root_states[ball_pos_env_ids, :3] += self.env_origins[reset_ball_pos_env_ids]

        reset_ball_vel_mark = np.random.choice([True, False],self.num_envs, p=[self.cfg.object.vel_reset_prob,1-self.cfg.object.vel_reset_prob])
        reset_ball_vel_env_ids = torch.tensor(np.array(np.nonzero(reset_ball_vel_mark)), device = self.device).flatten()
        ball_vel_env_ids = self.object_actor_idxs[reset_ball_vel_env_ids].to(device=self.device)
        self.root_states[ball_vel_env_ids,7:10] = 2*(torch.rand(len(ball_vel_env_ids), 3, dtype=torch.float, device=self.device,
                                                     requires_grad=False)-0.5) * torch.tensor(self.cfg.object.vel_reset_range,device=self.device,
                                                     requires_grad=False)
                                            
        reset_ball_vel_env_ids_int32 = ball_vel_env_ids.to(dtype = torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(torch.cat((reset_ball_pos_env_ids_int32, reset_ball_vel_env_ids_int32))), len(reset_ball_pos_env_ids_int32) + len(reset_ball_vel_env_ids_int32))
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def simulate_ball_pos_delay(self, new_ball_pos, last_ball_pos):
        receive_mark = np.random.choice([True, False],self.num_envs, p=[self.cfg.object.vision_receive_prob,1-self.cfg.object.vision_receive_prob])
        last_ball_pos[receive_mark,:] = new_ball_pos[receive_mark,:]

        return last_ball_pos

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # parkour
        self.reset_buf = torch.zeros((self.num_envs, ), dtype=torch.bool, device=self.device)
        roll_cutoff = torch.abs(self.roll) > 1.5
        pitch_cutoff = torch.abs(self.pitch) > 1.5
        reach_goal_cutoff = self.cur_goal_idx >= self.cfg.terrain.num_goals
        height_cutoff = self.root_states[:, 2] < -0.25

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.time_out_buf |= reach_goal_cutoff

        self.reset_buf |= self.time_out_buf
        self.reset_buf |= roll_cutoff
        self.reset_buf |= pitch_cutoff
        self.reset_buf |= height_cutoff
        
        # self.contact_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
        #                            dim=1)

        # self.reset_buf = torch.clone(self.contact_buf)
        # # print(f'1. contact: {torch.any(self.reset_buf)}')
        # self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length  # no terminal reward for time-outs


        # self.reset_buf |= self.time_out_buf
        # # print(f'2. timeout: {torch.any(self.reset_buf)}')

        # if self.cfg.rewards.use_terminal_body_height:
        #     self.body_height_buf = torch.mean(self.root_states[self.robot_actor_idxs, 2].unsqueeze(1) - self.measured_heights, dim=1) \
        #                            < self.cfg.rewards.terminal_body_height

        #     self.reset_buf = torch.logical_or(self.body_height_buf, self.reset_buf)

        # if self.cfg.rewards.use_terminal_roll_pitch:
        #     self.body_ori_buf = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) \
        #                         > self.cfg.rewards.terminal_body_ori
        #     # print('resetting?')

        #     self.reset_buf = torch.logical_or(self.body_ori_buf, self.reset_buf)

        # if self.cfg.rewards.use_terminal_time_since_last_obs:
        #     # if there is an ObjectSensor, check its time_since_last_obs property for termination
        #     if "ObjectSensor" in self.cfg.sensors.sensor_names:
        #         object_sensor_idx = self.cfg.sensors.sensor_names.index("ObjectSensor")
        #         object_sensor = self.sensors[object_sensor_idx]
        #         time_since_last_obs = object_sensor.time_since_last_obs
        #         self.time_since_last_obs_buf = time_since_last_obs > self.cfg.rewards.terminal_time_since_last_obs
        #         self.reset_buf = torch.logical_or(self.time_since_last_obs_buf, self.reset_buf)

            
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """


        if len(env_ids) == 0:
            return

        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)

        # reset robot states
        self._resample_commands(env_ids)
        self._randomize_dof_props(env_ids, self.cfg)
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._randomize_rigid_body_props(env_ids, self.cfg)
            self.refresh_actor_rigid_shape_props(env_ids, self.cfg)

        self._reset_dofs(env_ids, self.cfg)
        self._reset_root_states(env_ids, self.cfg)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.path_distance[env_ids] = 0.
        self.past_base_pos[env_ids] = self.base_pos.clone()[env_ids]
        self.contact_buf[env_ids] = 0.
        self.action_history_buf[env_ids] = 0.
        self.cur_goal_idx[env_ids] = 0.
        self.reach_goal_timer[env_ids] = 0.

        self.reset_buf = self.reset_buf.clone()
        self.reset_buf[env_ids] = 1

        self.x_vel_diff_history[env_ids] = 0.
        self.y_vel_diff_history[env_ids] = 0.
        self.yaw_vel_diff_history[env_ids] = 0.
        
        # reset history buffers
        if hasattr(self, "obs_history"):
            self.obs_history_buf[env_ids, :] = 0
            self.obs_history[env_ids, :] = 0

        self.extras = self.logger.populate_log(env_ids)
        self.episode_length_buf[env_ids] = 0

        self.gait_indices = self.gait_indices.clone()
        self.gait_indices[env_ids] = 0

        for i in range(len(self.lag_buffer)):
            self.lag_buffer[i] = self.lag_buffer[i].clone()
            self.lag_buffer[i][env_ids, :] = 0

    def set_idx_pose(self, env_ids, dof_pos, base_state):
        if len(env_ids) == 0:
            return

        env_ids_long = env_ids.to(dtype=torch.long, device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32, device=self.device)
        robot_actor_idxs_int32 = self.robot_actor_idxs.to(dtype=torch.int32)

        # joints
        if dof_pos is not None:
            self.dof_pos[env_ids] = dof_pos
            self.dof_vel[env_ids] = 0.

            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(robot_actor_idxs_int32[env_ids_long]), len(env_ids_long))

        # base position
        self.root_states[self.robot_actor_idxs[env_ids_long]] = base_state.to(self.device)
        # self.root_states[self.object_actor_idxs[env_ids_long]] = ball_init_state.to(self.device)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(torch.cat((self.robot_actor_idxs[env_ids_long], self.object_actor_idxs[env_ids_long]))), 2*len(env_ids_int32))
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def compute_torque_uncertainty(self):
        if not hasattr(self, "num_actuator_networks"):
            return torch.zeros(self.num_envs, device=self.device)
        # Compute the uncertainty in the actuator network via model disagreement
        all_torques = [self.actuator_network(self.joint_pos_err, self.past_joint_pos_errs[self.cfg.control.decimation], self.past_joint_pos_errs[0],
                                        self.joint_vel, self.past_joint_vels[self.cfg.control.decimation], self.past_joint_vels[0], i)
                                        for i in range(self.num_actuator_networks)]
        # all_torques has shape (num_actuator_networks, num_envs, num_actuated_dof) before stacking
        # Stack all_torques
        all_torques = torch.stack(all_torques, dim=0)
        # Compute the variance of all_torques in dim=0
        torque_uncertainty = torch.var(all_torques, dim=0)
        # Take the mean over the actuated DOFs
        torque_uncertainty = torch.mean(torque_uncertainty, dim=1)
        return torque_uncertainty
    
    def initialize_reward_bufs(self):
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.

    def compute_substep_reward(self):
        """ Compute rewards at each substep
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        # Start with previous rewards (keep rew_bufs from previous substep)
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew
            self.episode_sums[name] += rew
            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += self.reward_scales[name] + rew
            else:
                self.command_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        elif self.cfg.rewards.only_positive_rewards_ji22_style: #TODO: update
            self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.rewards.sigma_rew_neg)
    
    def finalize_reward(self):
        """ Takes the total reward to be the average of the reward at substeps. Adds this to the episode sums
        """
        # Average the rewards over the substeps and add to episode_sums
        self.rew_buf[:] = self.rew_buf[:] / self.cfg.control.decimation
        self.episode_sums["total"] += self.rew_buf
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self.reward_container._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            self.command_sums["termination"] += rew

        self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
        self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
        self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2
        self.command_sums["ep_timesteps"] += 1

    def estimate_energies(self):
        """
        Provides an estimate of the energy consumption over the past timestep
        """
        if not hasattr(self, "energy_network"):
            return torch.zeros(self.num_envs, 12, 3, device=self.device)

        energies = self.energy_network(self.joint_pos_err, self.joint_pos_err_last_timestep, self.joint_pos_err_last_last_timestep,
                                self.joint_vel, self.joint_vel_last_timestep, self.joint_vel_last_last_timestep)
        self.joint_pos_err_last_last_timestep = torch.clone(self.joint_pos_err_last_timestep)
        self.joint_pos_err_last_timestep = torch.clone(self.joint_pos_err)
        self.joint_vel_last_last_timestep = torch.clone(self.joint_vel_last_timestep)
        self.joint_vel_last_timestep = torch.clone(self.joint_vel)
        self.energies = energies
        return energies
    
    def compute_energy(self):
        joint_energies = self.energies.clone().detach() # (num_envs, 12, 3)

        mech_work = joint_energies[:,:,0].sum(dim=1) + joint_energies[:,:,1].sum(dim=1)
        torque_squareds = joint_energies[:,:,2] # (num_envs, 12)

        # gear ratios
        gear_ratios = torch.tensor([1.0, 1.0, 1./1.5,
                            1.0, 1.0, 1./1.5,
                            1.0, 1.0, 1./1.5,
                            1.0, 1.0, 1./1.5,], device=self.device,
                            ).unsqueeze(dim=0) # knee has extra gearing
        joule_heating = torch.sum(torque_squareds * torch.square(gear_ratios), dim=1) * 0.65

        return mech_work + joule_heating

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        # Define tensor of shape (num_envs, 1) to store the energy reward
        self.energy_rew = torch.zeros((self.num_envs,), device=self.device)
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if 'energy' in name:
                # print(rew.shape)
                # print(self.energy_rew.shape)
                self.energy_rew += rew.clone()
            else:
                self.rew_buf += rew
                if torch.sum(rew) >= 0:
                    self.rew_buf_pos += rew
                elif torch.sum(rew) <= 0:
                    self.rew_buf_neg += rew
            self.episode_sums[name] += rew
            if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                self.command_sums[name] += self.reward_scales[name] + rew
            else:
                self.command_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        elif self.cfg.rewards.only_positive_rewards_ji22_style: #TODO: update
            self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.cfg.rewards.sigma_rew_neg)
        self.episode_sums["total"] += self.rew_buf
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self.reward_container._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            self.command_sums["termination"] += rew

        self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
        self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
        self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2
        self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2
        self.command_sums["ep_timesteps"] += 1

    def initialize_sensors(self):
        """ Initializes sensors
        """
        from go1_gym.sensors import ALL_SENSORS
        self.sensors = []
        for sensor_name in self.cfg.sensors.sensor_names:
            if sensor_name in ALL_SENSORS.keys():
                self.sensors.append(ALL_SENSORS[sensor_name](self, **self.cfg.sensors.sensor_args[sensor_name]))
            else:
                raise ValueError(f"Sensor {sensor_name} not found.")

        # privileged sensors
        self.privileged_sensors = []
        for privileged_sensor_name in self.cfg.sensors.privileged_sensor_names:
            if privileged_sensor_name in ALL_SENSORS.keys():
                # print(privileged_sensor_name)
                self.privileged_sensors.append(ALL_SENSORS[privileged_sensor_name](self, **self.cfg.sensors.privileged_sensor_args[privileged_sensor_name]))
            else:
                raise ValueError(f"Sensor {privileged_sensor_name} not found.")
        

        # initialize noise vec
        self.add_noise = self.cfg.noise.add_noise
        noise_vec = []
        for sensor in self.sensors:
            noise_vec.append(sensor.get_noise_vec())

        self.noise_scale_vec = torch.cat(noise_vec, dim=-1).to(self.device)

    def compute_observations(self):
        """ Computes observations
        """
        if self.cfg.terrain.parkour:
            imu_obs = torch.stack((self.roll, self.pitch), dim=1)
            if self.global_counter % 5 == 0:
                self.delta_yaw = self.target_yaw - self.yaw
                self.delta_next_yaw = self.next_target_yaw - self.yaw
            obs_buf = torch.cat((
                self.base_ang_vel * self.obs_scales.ang_vel,
                imu_obs,
                0*self.delta_yaw[:, None],
                self.delta_yaw[:, None],
                self.delta_next_yaw[:, None],
                0*self.commands[:, 0:2],
                self.commands[:, 0:1],
                (self.env_class != 17).float()[:, None],
                (self.env_class == 17).float()[:, None],

                # TODO: seld.default_actuated_dof_pos might be different
                (self.dof_pos - self.default_actuated_dof_pos)* self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.action_history_buf[:, -1],
                self.contact_filt.float()-0.5
            ), dim=-1)
            priv_explicit = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                    0 * self.base_lin_vel,
                                    0 * self.base_lin_vel), dim=-1)
            # TODO: mass_params_tensor could be wrong
            priv_latent = torch.cat((
                self.mass_params_tensor,
                self.friction_coeffs,
                self.motor_strengths[0]-1,
                self.motor_strengths[1]-1
            ),dim=-1)
            
            if self.cfg.terrain.measure_heights:
                heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
                self.obs_buf = torch.cat([obs_buf, heights, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
            else:
                self.obs_buf = torch.cat([obs_buf, priv_explicit, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
            obs_buf[:, 6:8] = 0  # mask yaw in proprioceptive history
            self.obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None], 
                torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
                torch.cat([
                    self.obs_history_buf[:, 1:],
                    obs_buf.unsqueeze(1)
                ], dim=1)
            )

            self.contact_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None], 
                torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
                torch.cat([
                    self.contact_buf[:, 1:],
                    self.contact_filt.float().unsqueeze(1)
                ], dim=1)
            )
        else:
            # aggregate the sensor data
            self.pre_obs_buf = []
            for sensor in self.sensors:
                obs_tmp = sensor.get_observation()
                # if sensor.name == "JointPositionSensor" or sensor.name == "JointVelocitySensor":
                #     obs_tmp[:, -1]=0
                self.pre_obs_buf += [obs_tmp]
                # print("sensor type: ", sensor) #, " observation: ", obs_tmp)
                

            # print(torch.cat(self.pre_obs_buf, dim=-1).shape)
            
            self.pre_obs_buf = torch.reshape(torch.cat(self.pre_obs_buf, dim=-1), (self.num_envs, -1))
            self.obs_buf = self.pre_obs_buf.clone()
            if self.cfg.env.random_mask_input:
                self.mask_num = 10
                self.mask_input = torch.randint(0, self.obs_buf.shape[1], (self.num_envs, self.mask_num), device=self.device)
                self.obs_buf[:, self.mask_input] = 0.0

            self.privileged_obs_buf_list = []
            # aggregate the privileged observations
            for sensor in self.privileged_sensors:
                self.privileged_obs_buf_list += [sensor.get_observation()]
            # print("privileged_obs_buf: ", self.privileged_obs_buf)
            if len(self.privileged_obs_buf_list):
                self.privileged_obs_buf = torch.reshape(torch.cat(self.privileged_obs_buf_list, dim=-1), (self.num_envs, -1))
            # print("self.privileged_obs_buf: ", self.privileged_obs_buf)
            # add noise if needed
            if self.cfg.noise.add_noise:
                self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

            

            assert self.privileged_obs_buf.shape[
                    1] == self.cfg.env.num_privileged_obs, f"num_privileged_obs ({self.cfg.env.num_privileged_obs}) != the number of privileged observations ({self.privileged_obs_buf.shape[1]}), you will discard data from the student!"

    
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)

        mesh_type = self.cfg.terrain.mesh_type

        from go1_gym.terrains import ALL_TERRAINS
        if mesh_type not in ALL_TERRAINS.keys():
            raise ValueError(f"Terrain mesh type {mesh_type} not recognised. Allowed types are {ALL_TERRAINS.keys()}")
        
        if self.cfg.terrain.parkour:
            self.terrain = ParkourTerrain(self.cfg.terrain, self.num_envs)
        else:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)

        self.terrain_obj = ALL_TERRAINS[mesh_type](self)

        self._create_envs()
        
        self.terrain_obj.initialize()

        self.set_lighting()

    def set_lighting(self):
        light_index = 0
        intensity = gymapi.Vec3(0.5, 0.5, 0.5)
        ambient = gymapi.Vec3(0.2, 0.2, 0.2)
        direction = gymapi.Vec3(0.01, 0.01, 1.0)
        self.gym.set_light_parameters(self.sim, light_index, intensity, ambient, direction)

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def add_actor_critic(self, actor_critic):
        self.reward_container.actor_critic = actor_critic

    def add_teacher_actor_critic(self, teacher_actor_critic):
        self.reward_container.teacher_actor_critic = teacher_actor_critic

    def set_main_agent_pose(self, loc, quat):
        agent_id = 0

        self.root_states[self.robot_actor_idxs[agent_id], 0:3] = torch.Tensor(loc)
        self.root_states[self.robot_actor_idxs[agent_id], 3:7] = torch.Tensor(quat)


        robot_env_ids = self.robot_actor_idxs[agent_id].to(device=self.device)
        if self.cfg.env.add_objects:
            self.root_states[self.object_actor_idxs[agent_id], 0:2] = torch.Tensor(loc[0:2]) + torch.Tensor([2.5, 0.0])
            self.root_states[self.object_actor_idxs[agent_id], 3] = 0.0
            self.root_states[self.object_actor_idxs[agent_id], 3:7] = torch.Tensor(quat)
            print(self.root_states)
            
            object_env_ids = self.object_actor_idxs[agent_id].to(device=self.device)
            all_subject_env_ids = torch.tensor([robot_env_ids, object_env_ids]).to(device=self.device)
        else:
            all_subject_env_ids = torch.tensor([robot_env_ids]).to(device=self.device)
        all_subject_env_ids_int32 = all_subject_env_ids.to(dtype = torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
                                                gymtorch.unwrap_tensor(all_subject_env_ids_int32), len(all_subject_env_ids_int32))
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def _randomize_gravity(self, external_force = None):

        if external_force is not None:
            self.gravities[:, :] = external_force.unsqueeze(0)
        elif self.cfg.domain_rand.randomize_gravity:
            min_gravity, max_gravity = self.cfg.domain_rand.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity

            self.gravities[:, :] = external_force.unsqueeze(0)

        sim_params = self.gym.get_sim_params(self.sim)
        gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.8]).to(self.device)
        self.gravity_vec[:, :] = gravity.unsqueeze(0) / torch.norm(gravity)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)
        
    def _randomize_ball_drag(self):
        if self.cfg.domain_rand.randomize_ball_drag:
            min_drag, max_drag = self.cfg.domain_rand.drag_range
            ball_drags = torch.rand(self.num_envs, dtype=torch.float, device=self.device,
                                    requires_grad=False) * (max_drag - min_drag) + min_drag
            self.ball_drags[:, :]  = ball_drags.unsqueeze(1)

    def _randomize_feet_forces(self, env_ids):
        self.prev_target_feet_positions[env_ids, :] = self.target_feet_positions[env_ids, :]

        for fax in range(3):
            self.target_feet_positions[env_ids, fax] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (self.cfg.domain_rand.foot_height_forced_range[1][fax] - self.cfg.domain_rand.foot_height_forced_range[0][fax]) + self.cfg.domain_rand.foot_height_forced_range[0][fax]
        
        self.freed_envs[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                            requires_grad=False) < self.cfg.domain_rand.foot_height_forced_prob
        self.target_feet_positions[self.freed_envs, :3] = 0.

    def _apply_drag_force(self, force_tensor):
        if self.cfg.domain_rand.randomize_ball_drag:
            # force_tensor = torch.zeros((self.num_envs, self.num_bodies + 1, 3), dtype=torch.float32, device=self.device)
            force_tensor[:, self.num_bodies, :2] = - self.ball_drags * torch.square(self.object_lin_vel[:, :2]) * torch.sign(self.object_lin_vel[:, :2])
            # self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(force_tensor), None, gymapi.GLOBAL_SPACE)

    def _apply_ext_forces(self, forces):
        # apply the foot external force
        if self.cfg.domain_rand.randomize_foot_height_forced:
            progress =  (self.episode_length_buf % int(self.cfg.domain_rand.foot_height_forced_rand_interval)) / (int(self.cfg.domain_rand.foot_height_forced_rand_interval) * self.cfg.domain_rand.foot_motion_duration)
            progress = torch.clamp(progress, 0., 1.)

            self.cur_targets = self.target_feet_positions * (progress.unsqueeze(1)) + self.prev_target_feet_positions * (1 - progress.unsqueeze(1))

            # foot_positions = self.foot_positions[:, 0, :]
            cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
            cur_foot_vels = self.foot_velocities
            footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
            feet_vels_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
            for i in range(4):
                footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
                                                                cur_footsteps_translated[:, i, :])
                feet_vels_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
                                                                cur_foot_vels[:, i, :])
            footsteps_in_body_frame[:, :, 2] = self.foot_positions[:, :, 2]

            adjusted_foot_positions = footsteps_in_body_frame[:, 0, :]
            adjusted_foot_positions[:, 0] -= 0.25
            adjusted_foot_positions[:, 1] -= 0.20

            # self.feet_forces[:, 0] = torch.clamp((self.cur_targets[:, 0] - adjusted_foot_positions[:, 0]) * self.cfg.domain_rand.foot_force_kp, -self.cfg.domain_rand.max_foot_force, self.cfg.domain_rand.max_foot_force)
            # self.feet_forces[:, 1] = torch.clamp((self.cur_targets[:, 1] - adjusted_foot_positions[:, 1]) * self.cfg.domain_rand.foot_force_kp, -self.cfg.domain_rand.max_foot_force, self.cfg.domain_rand.max_foot_force)
            # self.feet_forces[:, 2] = torch.clamp((self.cur_targets[:, 2] - adjusted_foot_positions[:, 2]) * self.cfg.domain_rand.foot_force_kp, -self.cfg.domain_rand.max_foot_force, self.cfg.domain_rand.max_foot_force)

            pos_err = self.cur_targets[:, 0:3] - adjusted_foot_positions
            # margin = 0.02 # meters
            # pos_err[torch.abs(pos_err)<margin] = 0
            
            vel_err = 0 - feet_vels_in_body_frame[:, 0, :]
            
            self.feet_forces[:, :3] = pos_err[:, :3] * self.cfg.domain_rand.foot_force_kp + vel_err * self.cfg.domain_rand.foot_force_kd
            self.feet_forces[:, :3] = torch.clamp(self.feet_forces[:, :3], -self.cfg.domain_rand.max_foot_force, self.cfg.domain_rand.max_foot_force)
            
            self.feet_forces[self.freed_envs, :3] = 0.

            # transform the forces back into the world frame
            self.feet_forces[:, :3] = quat_apply_yaw(self.base_quat, self.feet_forces[:, :3])

            forces[:, self.feet_indices[0], :3] = self.feet_forces[:, :3]

            
        return forces

    def get_ground_frictions(self, env_ids):
        # get terrain cell indices
        # optimize later
        positions = self.root_states[env_ids, 0:3]
        terrain_cell_indices = torch.zeros((self.num_envs, 2), device=self.device)
        terrain_cell_indices[:, 0] = torch.clamp(positions[:, 0] / self.terrain.cfg.env_width, 0,
                                                 self.terrain.cfg.num_rows - 1)
        terrain_cell_indices[:, 1] = torch.clamp(positions[:, 1] / self.terrain.cfg.env_length, 0,
                                                 self.terrain.cfg.num_cols - 1)

        # get frictions
        ground_frictions = torch.zeros(self.num_envs, device=self.device)
        ground_frictions[:] = self.terrain_obj.terrain_cell_frictions[
            terrain_cell_indices[:, 0].long(), terrain_cell_indices[:, 1].long()]

        return ground_frictions

    def get_ground_restitutions(self, env_ids):
        # get terrain cell indices
        # optimize later
        positions = self.root_states[env_ids, 0:3]
        terrain_cell_indices = torch.zeros((self.num_envs, 2), device=self.device)
        terrain_cell_indices[:, 0] = torch.clamp(positions[:, 0] / self.terrain.cfg.env_width, 0,
                                                 self.terrain.cfg.num_rows - 1)
        terrain_cell_indices[:, 1] = torch.clamp(positions[:, 1] / self.terrain.cfg.env_length, 0,
                                                 self.terrain.cfg.num_cols - 1)

        # get frictions
        restitutions = torch.zeros(self.num_envs, device=self.device)
        restitutions[:] = self.terrain_obj.terrain_cell_restitutions[
                terrain_cell_indices[:, 0].long(), terrain_cell_indices[:, 1].long()]

        return restitutions

    def get_ground_roughness(self, env_ids):
        # get terrain cell indices
        # optimize later
        positions = self.base_pos
        terrain_cell_indices = torch.zeros((self.num_envs, 1, 2), device=self.device)
        terrain_cell_indices[:, 0, 0] = torch.clamp(positions[:, 0] / self.terrain.cfg.env_width, 0,
                                                       self.terrain.cfg.num_rows - 1)
        terrain_cell_indices[:, 0, 1] = torch.clamp(positions[:, 1] / self.terrain.cfg.env_length, 0,
                                                       self.terrain.cfg.num_cols - 1)

        # get roughnesses
        roughnesses = torch.zeros(self.num_envs, device=self.device)
        roughnesses[:] = self.terrain_obj.terrain_cell_roughnesses[
            terrain_cell_indices[:, 0, 0].long(), terrain_cell_indices[:, 0, 1].long()]

        return roughnesses

    def get_stair_height(self, env_ids):
        points = self.base_pos
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, 0]
        py = points[:, 1]
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        stair_heights = self.stair_heights_samples[px, py]
        return stair_heights

    def get_stair_run(self, env_ids):
        points = self.base_pos
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, 0]
        py = points[:, 1]
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        stair_runs = self.stair_runs_samples[px, py]
        return stair_runs

    def get_stair_ori(self, env_ids):
        points = self.base_pos
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, 0]
        py = points[:, 1]
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        stair_oris = self.stair_oris_samples[px, py]
        return stair_oris

    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        for s in range(len(props)):
            props[s].friction = self.friction_coeffs[env_id, 0]
            props[s].restitution = self.restitutions[env_id, 0]

        return props
    
    def _process_ball_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        for s in range(len(props)):
            props[s].friction = self.ball_friction_coeffs[env_id]
            props[s].restitution = self.ball_restitutions[env_id]

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()

                # Torque limits
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        return props

    def _randomize_rigid_body_props(self, env_ids, cfg):
        if cfg.domain_rand.randomize_base_mass:
            min_payload, max_payload = cfg.domain_rand.added_mass_range
            # self.payloads[env_ids] = -1.0
            self.payloads[env_ids] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                requires_grad=False) * (max_payload - min_payload) + min_payload
        if cfg.domain_rand.randomize_com_displacement:
            min_com_displacement, max_com_displacement = cfg.domain_rand.com_displacement_range
            self.com_displacements[env_ids, :] = torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
                                                            requires_grad=False) * (
                                                         max_com_displacement - min_com_displacement) + min_com_displacement

        if cfg.domain_rand.randomize_friction:
            min_friction, max_friction = cfg.domain_rand.friction_range
            self.friction_coeffs[env_ids, :] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                          requires_grad=False) * (
                                                       max_friction - min_friction) + min_friction

        if cfg.domain_rand.randomize_restitution:
            min_restitution, max_restitution = cfg.domain_rand.restitution_range
            self.restitutions[env_ids] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                    requires_grad=False) * (
                                                 max_restitution - min_restitution) + min_restitution

        if cfg.env.add_objects:
            if cfg.domain_rand.randomize_ball_restitution:
                min_restitution, max_restitution = cfg.domain_rand.ball_restitution_range
                self.ball_restitutions[env_ids] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                        requires_grad=False) * (
                                                    max_restitution - min_restitution) + min_restitution

            if cfg.domain_rand.randomize_ball_friction:
                min_friction, max_friction = cfg.domain_rand.ball_friction_range
                self.ball_friction_coeffs[env_ids] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                        requires_grad=False) * (
                                                    max_friction - min_friction) + min_friction

    def refresh_actor_rigid_shape_props(self, env_ids, cfg):
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(self.num_dofs):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitutions[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _randomize_dof_props(self, env_ids, cfg):
        if cfg.domain_rand.randomize_motor_strength:
            min_strength, max_strength = cfg.domain_rand.motor_strength_range
            self.motor_strengths[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_strength - min_strength) + min_strength
        if cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids, :] = torch.rand(len(env_ids), self.num_dofs, dtype=torch.float,
                                                        device=self.device, requires_grad=False) * (
                                                     max_offset - min_offset) + min_offset
        if cfg.domain_rand.randomize_Kp_factor:
            min_Kp_factor, max_Kp_factor = cfg.domain_rand.Kp_factor_range
            self.Kp_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kp_factor - min_Kp_factor) + min_Kp_factor
        if cfg.domain_rand.randomize_Kd_factor:
            min_Kd_factor, max_Kd_factor = cfg.domain_rand.Kd_factor_range
            self.Kd_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kd_factor - min_Kd_factor) + min_Kd_factor

    def _process_rigid_body_props(self, props, env_id):
        self.default_body_mass = props[0].mass

        props[0].mass = self.default_body_mass + self.payloads[env_id]
        props[0].com = gymapi.Vec3(self.com_displacements[env_id, 0], self.com_displacements[env_id, 1],
                                   self.com_displacements[env_id, 2])
        
        mass_params = np.array([ self.payloads[env_id].cpu(),self.com_displacements[env_id, 0].cpu(),self.com_displacements[env_id, 1].cpu(),self.com_displacements[env_id, 2].cpu()])
        return props, mass_params

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """

        # teleport robots to prevent falling off the edge
        self._teleport_robots(torch.arange(self.num_envs, device=self.device), self.cfg)

        # resample commands
        sample_interval = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % sample_interval == 0).nonzero(as_tuple=False).flatten()
        # print(self.episode_length_buf, sample_interval, env_ids)
        
        self._resample_commands(env_ids)
        # if len(env_ids) > 0: print(self.commands[0, 0:3])
        self._step_contact_targets()
        
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0]) - self.heading_offsets
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.heading_commands - heading), -1., 1.)

        # measure terrain heights
        if self.cfg.perception.measure_heights:
            self.measured_heights = self.heightmap_sensor.get_observation()

        # push robots
        self._push_robots(torch.arange(self.num_envs, device=self.device), self.cfg)

        # randomize dof properties
        env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.rand_interval) == 0).nonzero(
            as_tuple=False).flatten()
        self._randomize_dof_props(env_ids, self.cfg)

        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()
        if self.common_step_counter % int(self.cfg.domain_rand.ball_drag_rand_interval) == 0:
            self._randomize_ball_drag()
        if int(self.common_step_counter - self.cfg.domain_rand.gravity_rand_duration) % int(
                self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity(torch.tensor([0, 0, 0]))
        if self.cfg.domain_rand.randomize_rigids_after_start:
            self._randomize_rigid_body_props(env_ids, self.cfg)
            self.refresh_actor_rigid_shape_props(env_ids, self.cfg)

        env_ids = (self.episode_length_buf % int(self.cfg.domain_rand.foot_height_forced_rand_interval) == 0).nonzero(
            as_tuple=False).flatten()
        self._randomize_feet_forces(env_ids)

    def _gather_cur_goals(self, future=0):
        return self.env_goals.gather(1, (self.cur_goal_idx[:, None, None]+future).expand(-1, -1, self.env_goals.shape[-1])).squeeze(1)

    def _resample_commands(self, env_ids):

        if len(env_ids) == 0 or not self.cfg.commands.resample_command: 
            return

        timesteps = int(self.cfg.commands.resampling_time / self.dt)
        ep_len = min(self.cfg.env.max_episode_length, timesteps)


        # update curricula based on terminated environment bins and categories
        for i, (category, curriculum) in enumerate(zip(self.category_names, self.curricula)):
            env_ids_in_category = self.env_command_categories[env_ids.cpu()] == i
            if isinstance(env_ids_in_category, np.bool_) or len(env_ids_in_category) == 1:
                env_ids_in_category = torch.tensor([env_ids_in_category], dtype=torch.bool)
            elif len(env_ids_in_category) == 0:
                continue

            env_ids_in_category = env_ids[env_ids_in_category]

            task_rewards, success_thresholds = [], []
            for key in ["tracking_lin_vel", "tracking_ang_vel", "tracking_lin_vel_balanced", "tracking_contacts_shaped_force",
                        "tracking_contacts_shaped_vel"]:
                if key in self.command_sums.keys():
                    task_rewards.append(self.command_sums[key][env_ids_in_category] / ep_len)
                    success_thresholds.append(self.curriculum_thresholds[key] * self.reward_scales[key])

            old_bins = self.env_command_bins[env_ids_in_category.cpu().numpy()]
            if len(success_thresholds) > 0:
                if not self.cfg.commands.inverse_IK_door_opening:
                    curriculum.update(old_bins, task_rewards, success_thresholds,
                                    local_range=np.array(
                                        [0.55, 0.55, 0.55, 0.55, 0.35, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
                else:
                    # curriculum.update(old_bins, task_rewards, success_thresholds,
                    #                 local_range=np.array(
                    #                     [0.55, 0.55, 0.55, 0.55, 0.35, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                    #                     0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.1]))
                    curriculum.update(old_bins, task_rewards, success_thresholds,
                                    local_range=np.array(
                                        [0.55, 0.55, 0.55, 0.55, 0.35, 0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                                        0.5, 0.5, 0.5, 1.0]))
        # assign resampled environments to new categories
        random_env_floats = torch.rand(len(env_ids), device=self.device)
        probability_per_category = 1. / len(self.category_names)
        category_env_ids = [env_ids[torch.logical_and(probability_per_category * i <= random_env_floats,
                                                      random_env_floats < probability_per_category * (i + 1))] for i in
                            range(len(self.category_names))]

        # sample from new category curricula
        for i, (category, env_ids_in_category, curriculum) in enumerate(
                zip(self.category_names, category_env_ids, self.curricula)):

            batch_size = len(env_ids_in_category)
            if batch_size == 0: continue

            new_commands, new_bin_inds = curriculum.sample(batch_size=batch_size)

            self.env_command_bins[env_ids_in_category.cpu().numpy()] = new_bin_inds
            self.env_command_categories[env_ids_in_category.cpu().numpy()] = i

            self.commands[env_ids_in_category, :] = torch.Tensor(new_commands[:, :self.cfg.commands.num_commands]).to(
                self.device)

            # print(self.commands[0, 0:3])



        if self.cfg.commands.num_commands > 5:
            if self.cfg.commands.gaitwise_curricula:
                for i, (category, env_ids_in_category) in enumerate(zip(self.category_names, category_env_ids)):
                    if category == "pronk":  # pronking
                        self.commands[env_ids_in_category, 5] = (self.commands[env_ids_in_category, 5] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 6] = (self.commands[env_ids_in_category, 6] / 2 - 0.25) % 1
                        self.commands[env_ids_in_category, 7] = (self.commands[env_ids_in_category, 7] / 2 - 0.25) % 1
                    elif category == "trot":  # trotting
                        self.commands[env_ids_in_category, 5] = self.commands[env_ids_in_category, 5] / 2 + 0.25
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "pace":  # pacing
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = self.commands[env_ids_in_category, 6] / 2 + 0.25
                        self.commands[env_ids_in_category, 7] = 0
                    elif category == "bound":  # bounding
                        self.commands[env_ids_in_category, 5] = 0
                        self.commands[env_ids_in_category, 6] = 0
                        self.commands[env_ids_in_category, 7] = self.commands[env_ids_in_category, 7] / 2 + 0.25

            elif self.cfg.commands.exclusive_phase_offset:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                trotting_envs = env_ids[random_env_floats < 0.34]
                pacing_envs = env_ids[torch.logical_and(0.34 <= random_env_floats, random_env_floats < 0.67)]
                bounding_envs = env_ids[0.67 <= random_env_floats]
                self.commands[pacing_envs, 5] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[trotting_envs, 6] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 7] = 0

            elif self.cfg.commands.balance_gait_distribution:
                random_env_floats = torch.rand(len(env_ids), device=self.device)
                pronking_envs = env_ids[random_env_floats <= 0.25]
                trotting_envs = env_ids[torch.logical_and(0.25 <= random_env_floats, random_env_floats < 0.50)]
                pacing_envs = env_ids[torch.logical_and(0.50 <= random_env_floats, random_env_floats < 0.75)]
                bounding_envs = env_ids[0.75 <= random_env_floats]
                self.commands[pronking_envs, 5] = (self.commands[pronking_envs, 5] / 2 - 0.25) % 1
                self.commands[pronking_envs, 6] = (self.commands[pronking_envs, 6] / 2 - 0.25) % 1
                self.commands[pronking_envs, 7] = (self.commands[pronking_envs, 7] / 2 - 0.25) % 1
                self.commands[trotting_envs, 6] = 0
                self.commands[trotting_envs, 7] = 0
                self.commands[pacing_envs, 5] = 0
                self.commands[pacing_envs, 7] = 0
                self.commands[bounding_envs, 5] = 0
                self.commands[bounding_envs, 6] = 0
                self.commands[trotting_envs, 5] = self.commands[trotting_envs, 5] / 2 + 0.25
                self.commands[pacing_envs, 6] = self.commands[pacing_envs, 6] / 2 + 0.25
                self.commands[bounding_envs, 7] = self.commands[bounding_envs, 7] / 2 + 0.25

            if self.cfg.commands.binary_phases:
                self.commands[env_ids, 5] = (torch.round(2 * self.commands[env_ids, 5])) / 2.0 % 1
                self.commands[env_ids, 6] = (torch.round(2 * self.commands[env_ids, 6])) / 2.0 % 1
                self.commands[env_ids, 7] = (torch.round(2 * self.commands[env_ids, 7])) / 2.0 % 1

        # # setting the smaller commands to zero
        # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

        # reset command sums
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.
            
        # respect command constriction
        self._update_command_ranges(env_ids)
            
        # heading commands
        if self.cfg.commands.heading_command:
            self.heading_commands[env_ids] = torch_rand_float(self.cfg.commands.heading[0],
                                                         self.cfg.commands.heading[1], (len(env_ids), 1),
                                                         device=self.device).squeeze(1)

    def _step_contact_targets(self):
        # if self.cfg.env.observe_gait_commands:
        frequencies = self.commands[:, 4]
        phases = self.commands[:, 5]
        offsets = self.commands[:, 6]
        bounds = self.commands[:, 7]
        durations = self.commands[:, 8]
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        if self.cfg.commands.pacing_offset:
            foot_indices = [self.gait_indices + phases + offsets + bounds,
                            self.gait_indices + bounds,
                            self.gait_indices + offsets,
                            self.gait_indices + phases]
        else:
            foot_indices = [self.gait_indices + phases + offsets + bounds,
                            self.gait_indices + offsets,
                            self.gait_indices + bounds,
                            self.gait_indices + phases]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                        0.5 / (1 - durations[swing_idxs]))

        # if self.cfg.commands.durations_warp_clock_inputs:

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        self.doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[0])
        self.doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[1])
        self.doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[2])
        self.doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[3])

        self.halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[0])
        self.halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[1])
        self.halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[2])
        self.halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[3])

        # von mises distribution
        kappa = self.cfg.rewards.kappa_gait_probs
        smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
        smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR
        # print("self.foot_indices", self.foot_indices, " self.desired_contact_states", self.desired_contact_states, " self.clock_inputs", self.clock_inputs)

        if self.cfg.commands.num_commands > 9:
            self.desired_footswing_height = self.commands[:, 9]

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller

        actions_scaled = torch.zeros(self.num_envs, self.num_dofs, device = self.device)
        actions_scaled[:, :self.num_actuated_dof] = actions[:, :self.num_actuated_dof] * self.cfg.control.action_scale
        if self.num_actions >= 12:
            actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction  # scale down hip flexion range

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
            self.joint_pos_target = self.lag_buffer[0] + self.default_actuated_dof_pos
        else:
            self.joint_pos_target = actions_scaled + self.default_actuated_dof_pos

        control_type = self.cfg.control.control_type

        #self.joint_pos_target[:, 12] = -70/180*3.14
        # Force gripper state to 0
        if self.num_dofs > 12:
            if self.cfg.commands.teleop_occulus:
                self.joint_pos_target[:, 18] = self.teleop_value
                self.joint_pos_target[:, 18] = self.teleop_gripper_value
                self.joint_pos_target[:, 17] = self.teleop_joint6_value
            else:
                self.joint_pos_target[:, 18] = 0.0
        if self.cfg.commands.only_test_loco:
            self.joint_pos_target[:, :7] = 0.0
            # self.joint_pos_target[:, 17] = 0.0

        if control_type == "actuator_net":
            self.joint_pos_err = self.actuated_dof_pos - self.joint_pos_target + self.motor_offsets
            self.joint_vel = self.dof_vel
            torques = self.actuator_network(self.joint_pos_err, self.past_joint_pos_errs[self.cfg.control.decimation], self.past_joint_pos_errs[0],
                                            self.joint_vel, self.past_joint_vels[self.cfg.control.decimation], self.past_joint_vels[0])
            # Save the most recent joint pos err and most recent joint vel, delete oldest joint pos err and oldest joint vel
            self.past_joint_pos_errs = torch.cat((self.past_joint_pos_errs[1:], self.joint_pos_err.unsqueeze(dim=0)), dim=0)
            self.past_joint_vels = torch.cat((self.past_joint_vels[1:], self.joint_vel.unsqueeze(dim=0)), dim=0)

            # self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
            # self.joint_pos_err_last = torch.clone(self.joint_pos_err)
            # self.joint_vel_last_last = torch.clone(self.joint_vel_last)
            # self.joint_vel_last = torch.clone(self.joint_vel)
        elif control_type == "P":
            torques = self.p_gains * self.Kp_factors * (
                    self.joint_pos_target - self.actuated_dof_pos + self.motor_offsets) - self.d_gains * self.Kd_factors * self.actuated_dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        torques = torques * self.motor_strengths
        # Shape of self.dof_vel_limits: (num_dofs, )
        torque_lim_constant = 33
        torque_limits = torque_lim_constant/(self.dof_vel_limits/(self.dof_vel_limits - torch.abs(self.dof_vel)))
        # torque_limits = torque_lim_constant
        self.unclipped_torques = torques
        torques = torch.clip(torques, -torque_limits, torque_limits)
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids, cfg):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof_all),
                                                                        device=self.device)
        self.dof_vel[env_ids] = 0.

        all_subject_env_ids = self.robot_actor_idxs[env_ids].to(device=self.device)
        if self.cfg.env.add_objects and self.cfg.object.asset == "door": 
            object_env_ids = self.object_actor_idxs[env_ids].to(device=self.device)
            all_subject_env_ids = torch.cat((all_subject_env_ids, object_env_ids))
        all_subject_env_ids_int32 = all_subject_env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_subject_env_ids_int32), len(all_subject_env_ids_int32))

    def _reset_root_states(self, env_ids, cfg):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        robot_env_ids = self.robot_actor_idxs[env_ids].to(device=self.device)

        ### Reset robots
        # base origins
        if self.custom_origins:
            self.root_states[robot_env_ids] = self.base_init_state
            self.root_states[robot_env_ids, :3] += self.env_origins[env_ids]
            # self.root_states[robot_env_ids, 0:1] += torch_rand_float(-cfg.terrain.x_init_range,
            #                                                    cfg.terrain.x_init_range, (len(robot_env_ids), 1),
            #                                                    device=self.device)
            # self.root_states[robot_env_ids, 1:2] += torch_rand_float(-cfg.terrain.y_init_range,
            #                                                    cfg.terrain.y_init_range, (len(robot_env_ids), 1),
            #                                                    device=self.device)
            # self.root_states[robot_env_ids, 0] += cfg.terrain.x_init_offset
            # self.root_states[robot_env_ids, 1] += cfg.terrain.y_init_offset
        else:
            self.root_states[robot_env_ids] = self.base_init_state
            self.root_states[robot_env_ids, :3] += self.env_origins[env_ids]

        # base yaws
        # init_yaws = torch_rand_float(-cfg.terrain.yaw_init_range,
        #                              cfg.terrain.yaw_init_range, (len(robot_env_ids), 1),
        #                              device=self.device)
        # print(init_yaws.shape, len(robot_env_ids))
        # quat = quat_from_angle_axis(init_yaws, torch.Tensor([0, 0, 1]).to(self.device))[:, 0, :]
        # print(quat.shape)
        # self.root_states[robot_env_ids, 3:7] = quat
        
        # random_yaw_angle = 2*(torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
        #                                              requires_grad=False)-0.5)*torch.tensor([0, 0, cfg.terrain.yaw_init_range], device=self.device)
        # self.root_states[robot_env_ids,3:7] = quat_from_euler_xyz(random_yaw_angle[:,0], random_yaw_angle[:,1], random_yaw_angle[:,2])
            

        if self.cfg.env.offset_yaw_obs:
            self.heading_offsets[env_ids] = torch_rand_float(-cfg.terrain.yaw_init_range,
                                     cfg.terrain.yaw_init_range, (len(env_ids), 1),
                                     device=self.device).flatten()

        # base velocities
        self.root_states[robot_env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(robot_env_ids), 6),
                                                           device=self.device)  # [7:10]: lin vel, [10:13]: ang vel


        ### Reset objects
        if self.cfg.env.add_objects:
            object_env_ids = self.object_actor_idxs[env_ids].to(device=self.device)
            # base origins
            self.root_states[object_env_ids, :] = self.object_init_state
            # transform to world frame
            # self.root_states[object_env_ids, :3] += self.env_origins[env_ids]
            object_init_local = self.root_states[object_env_ids, :3]
            object_init_local += 2*(torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
                                                     requires_grad=False)-0.5) * torch.tensor(Cfg.object.init_pos_range,device=self.device,
                                                     requires_grad=False)
            object_init_world = self.root_states[robot_env_ids, :3] + quat_apply_yaw(self.root_states[robot_env_ids,3:7], object_init_local)

            self.root_states[object_env_ids, 0:3] = object_init_world

            # self.root_states[object_env_ids,0:3] 
            self.root_states[object_env_ids,7:10] += 2*(torch.rand(len(env_ids), 3, dtype=torch.float, device=self.device,
                                                     requires_grad=False)-0.5) * torch.tensor(Cfg.object.init_vel_range,device=self.device,
                                                     requires_grad=False)
                                                     

        # apply reset states
        all_subject_env_ids = robot_env_ids
        if self.cfg.env.add_objects: 
            all_subject_env_ids = torch.cat((robot_env_ids, object_env_ids))
        all_subject_env_ids_int32 = all_subject_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(all_subject_env_ids_int32), len(all_subject_env_ids_int32))
        self.gym.refresh_actor_root_state_tensor(self.sim)

        if cfg.env.record_video and 0 in env_ids:
            if self.complete_video_frames is None:
                self.complete_video_frames = []
            else:
                self.complete_video_frames = self.video_frames[:]
            self.video_frames = []
        
    def _push_robots(self, env_ids, cfg):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        if cfg.domain_rand.push_robots:
            env_ids = env_ids[self.episode_length_buf[env_ids] % int(cfg.domain_rand.push_interval) == 0]

            max_vel = cfg.domain_rand.max_push_vel_xy
            self.root_states[self.robot_actor_idxs[env_ids], 7:9] = torch_rand_float(-max_vel, max_vel, (len(env_ids), 2),
                                                              device=self.device)  # lin vel x/y
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self.gym.refresh_actor_root_state_tensor(self.sim)
            
    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.
        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        robot_env_ids = self.robot_actor_idxs[env_ids]
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[robot_env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.cfg.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        # move_down = (distance < torch.norm(self.commands[env_ids, :2],
        #                                    dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        move_down = (self.path_distance[env_ids] < torch.norm(self.commands[env_ids, :2],
                                           dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random xfone
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids],
                                                                      low=self.min_terrain_level,
                                                                      high=self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids],
                                                              self.min_terrain_level))  # (the minumum level is zero)
        # self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.env_origins[env_ids] = self.cfg.terrain.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        
    def _update_path_distance(self):
        path_distance_interval = 25
        env_ids = (self.episode_length_buf % path_distance_interval == 0).nonzero(as_tuple=False).flatten()
        distance_traversed = torch.linalg.norm(self.base_pos[env_ids, 0:2] - self.past_base_pos[env_ids, 0:2], dim=1)
        self.path_distance[env_ids] += distance_traversed
        self.past_base_pos[env_ids] = self.base_pos.clone()[env_ids]

    def _update_command_ranges(self, env_ids):
        constrict_indices = self.cfg.rewards.constrict_indices
        constrict_ranges = self.cfg.rewards.constrict_ranges

        if self.cfg.rewards.constrict and self.common_step_counter >= self.cfg.rewards.constrict_after:
            for idx, range in zip(constrict_indices, constrict_ranges):
                self.commands[env_ids, idx] = range[0]

    def _teleport_robots(self, env_ids, cfg):
        """ Teleports any robots that are too close to the edge to the other side
        """
        robot_env_ids = self.robot_actor_idxs[env_ids].to(device=self.device)
        # object_env_ids = self.object_actor_idxs[env_ids].to(device=self.device)
        if cfg.terrain.teleport_robots:
            thresh = cfg.terrain.teleport_thresh
            x_offset = int(cfg.terrain.x_offset * cfg.terrain.horizontal_scale)

            # print(self.root_states[robot_env_ids, 0], thresh + x_offset, cfg.terrain.terrain_length * (cfg.terrain.num_rows) - thresh + x_offset)
            # print(self.root_states[robot_env_ids, 1], thresh, cfg.terrain.terrain_width * (cfg.terrain.num_cols) - thresh)

            low_x_ids = robot_env_ids[self.root_states[robot_env_ids, 0] < thresh + x_offset]

            high_x_ids = robot_env_ids[
                self.root_states[robot_env_ids, 0] > cfg.terrain.terrain_length * (cfg.terrain.num_rows) - thresh + x_offset]

            low_y_ids = robot_env_ids[self.root_states[robot_env_ids, 1] < thresh]

            high_y_ids = robot_env_ids[
                self.root_states[robot_env_ids, 1] > cfg.terrain.terrain_width * (cfg.terrain.num_cols) - thresh]
            
            self.root_states[low_x_ids, 0] += cfg.terrain.terrain_length * (cfg.terrain.num_rows - 2)
            self.root_states[high_x_ids, 0] -= cfg.terrain.terrain_length * (cfg.terrain.num_rows - 2)
            self.root_states[low_y_ids, 1] += cfg.terrain.terrain_width * (cfg.terrain.num_cols - 2)
            self.root_states[high_y_ids, 1] -= cfg.terrain.terrain_width * (cfg.terrain.num_cols - 2)

            # print(low_x_ids, high_x_ids, low_y_ids, high_y_ids)

            # print("after")
            # print(self.root_states[robot_env_ids, 0], thresh + x_offset, cfg.terrain.terrain_length * (cfg.terrain.num_rows - 1))
            # print(self.root_states[robot_env_ids, 1], thresh, cfg.terrain.terrain_width * (cfg.terrain.num_cols - 1))


            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self.gym.refresh_actor_root_state_tensor(self.sim)

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof_all, 2)[..., 0]
        self.actuated_dof_pos = self.dof_state.view(self.num_envs, self.num_dof_all, 2)[:, :self.num_dofs, 0]
        self.base_pos = self.root_states[self.robot_actor_idxs, 0:3]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof_all, 2)[..., 1]
        self.actuated_dof_vel = self.dof_state.view(self.num_envs, self.num_dof_all, 2)[:, :self.num_dofs, 1]
        self.base_quat = self.root_states[self.robot_actor_idxs, 3:7]
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs*self.total_rigid_body_num,:].view(self.num_envs,self.total_rigid_body_num,13)[:,0:self.num_bodies, :]#.contiguous().view(self.num_envs*self.num_bodies,13)
        self.rigid_body_state_object = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs*self.total_rigid_body_num,:].view(self.num_envs,self.total_rigid_body_num,13)[:,self.num_bodies:self.num_bodies + self.num_object_bodies, :]#.contiguous().view(self.num_envs*self.num_bodies,13)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, -1, 13)[:,
                               self.feet_indices,
                               7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, -1, 13)[:, self.feet_indices,
                              0:3]
        self.prev_base_pos = self.base_pos.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()

        self.init_training = False

        self.lag_buffer = [torch.zeros_like(self.actuated_dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps+1)]

        self.x_vel_diff_history = torch.zeros(self.num_envs, self.cfg.rewards.integral_history_len, device=self.device)
        self.y_vel_diff_history = torch.zeros(self.num_envs, self.cfg.rewards.integral_history_len, device=self.device)
        self.yaw_vel_diff_history = torch.zeros(self.num_envs, self.cfg.rewards.integral_history_len, device=self.device)

        # self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs * self.num_bodies, :].view(self.num_envs, -1,
        #                                                                     3)  # shape: num_envs, num_bodies, xyz axis
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)[:self.num_envs*self.total_rigid_body_num,:].view(self.num_envs,self.total_rigid_body_num,3)[:,0:self.num_bodies, :]

        if self.cfg.env.add_objects:
            self.object_pos_world_frame = self.root_states[self.object_actor_idxs, 0:3].clone()
            # if self.cfg.object.asset == "door":
            #     handle_idx = self.gym.find_actor_rigid_body_handle(self.envs[0], self.object_actor_handles[0], "handle") - self.num_bodies
            #     handle_pos_global = self.rigid_body_state_object.view(self.num_envs, -1, 13)[:,handle_idx,0:3].view(self.num_envs,3)
            #     robot_object_vec = handle_pos_global - self.base_pos
            # else:
            #     robot_object_vec = self.root_states[self.object_actor_idxs, 0:3] - self.base_pos
            robot_object_vec = self.asset.get_local_pos()
            self.object_local_pos = quat_rotate_inverse(self.base_quat, robot_object_vec)
            self.object_local_pos[:, 2] = 0.0*torch.ones(self.num_envs, dtype=torch.float,
                                       device=self.device, requires_grad=False)

            self.last_object_local_pos = torch.clone(self.object_local_pos)
            self.object_lin_vel = self.asset.get_lin_vel()
            self.object_ang_vel = self.asset.get_ang_vel()
         

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}

        # if self.cfg.perception.measure_heights:
        self.height_points = self._init_height_points(torch.arange(self.num_envs, device=self.device), self.cfg)
        self.measured_heights = 0

        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof_all, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_torques = torch.zeros(self.num_envs, self.num_dof_all, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.forces = torch.zeros(self.num_envs, self.total_rigid_body_num, 3, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.p_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float,
                                            device=self.device,
                                            requires_grad=False)
        self.last_joint_pos_target = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.last_last_joint_pos_target = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float,
                                                      device=self.device,
                                                      requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[self.robot_actor_idxs, 7:13])
        self.path_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.past_base_pos = self.base_pos.clone()


        self.commands_value = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                          device=self.device, requires_grad=False)
        self.commands = torch.zeros_like(self.commands_value)  # x vel, y vel, yaw vel, heading
        self.heading_commands = torch.zeros(self.num_envs, dtype=torch.float,
                                          device=self.device, requires_grad=False)  # heading
        self.heading_offsets = torch.zeros(self.num_envs, dtype=torch.float,
                                            device=self.device, requires_grad=False)  # heading offset
        
        if self.cfg.commands.inverse_IK_door_opening:
            # self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
            #                                     self.obs_scales.body_height_cmd, self.obs_scales.gait_freq_cmd,
            #                                     self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
            #                                     self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
            #                                     self.obs_scales.footswing_height_cmd, self.obs_scales.body_pitch_cmd,
            #                                     self.obs_scales.body_roll_cmd, self.obs_scales.stance_width_cmd,
            #                                     self.obs_scales.stance_length_cmd, self.obs_scales.aux_reward_cmd, 
            #                                     self.obs_scales.end_effector_pos_x_cmd, self.obs_scales.end_effector_pos_y_cmd,
            #                                     self.obs_scales.end_effector_pos_z_cmd, self.obs_scales.end_effector_roll_cmd,
            #                                     self.obs_scales.end_effector_pitch_cmd, self.obs_scales.end_effector_yaw_cmd,
            #                                     self.obs_scales.end_effector_gripper_cmd],
            #                                     device=self.device, requires_grad=False, )[:self.cfg.commands.num_commands]

            self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
                                                self.obs_scales.body_height_cmd, self.obs_scales.gait_freq_cmd,
                                                self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                                self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                                self.obs_scales.footswing_height_cmd, self.obs_scales.body_pitch_cmd,
                                                self.obs_scales.body_roll_cmd, self.obs_scales.stance_width_cmd,
                                                self.obs_scales.stance_length_cmd, self.obs_scales.aux_reward_cmd, 
                                                self.obs_scales.ee_sphe_radius_cmd, self.obs_scales.ee_sphe_pitch_cmd,
                                                self.obs_scales.ee_sphe_yaw_cmd, self.obs_scales.ee_timing_cmd],
                                                device=self.device, requires_grad=False, )[:self.cfg.commands.num_commands]
        else:
            self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
                                                self.obs_scales.body_height_cmd, self.obs_scales.gait_freq_cmd,
                                                self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                                self.obs_scales.gait_phase_cmd, self.obs_scales.gait_phase_cmd,
                                                self.obs_scales.footswing_height_cmd, self.obs_scales.body_pitch_cmd,
                                                self.obs_scales.body_roll_cmd, self.obs_scales.stance_width_cmd,
                                                self.obs_scales.stance_length_cmd, self.obs_scales.aux_reward_cmd],
                                                device=self.device, requires_grad=False, )[:self.cfg.commands.num_commands]            
            
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.time_since_takeoff = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.time_since_touchdown = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float,
                                         device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device,
                                         requires_grad=False)
        self.last_contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool,
                                             device=self.device,
                                             requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[self.robot_actor_idxs, 10:13])
       
        
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # joint positions offsets and PD gains
        self.default_actuated_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = torch.zeros(self.num_dof_all, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_actuated_dof_pos[i] = angle
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                # print("dof_name: ", dof_name)
                if dof_name in name:

                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    # print("dof_name is in name: ", dof_name, " name: ", name, self.cfg.control.stiffness)
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        for i in range(self.num_dof_all-self.num_dofs):
            self.default_dof_pos[i+self.num_dofs] = 0.0
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_actuated_dof_pos = self.default_actuated_dof_pos.unsqueeze(0)

        if self.cfg.control.control_type == "actuator_net":
            # actuator_path = f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/../../resources/actuator_nets/unitree_go1_8_1-6.pt'
            # actuator_network = torch.jit.load(actuator_path, map_location=self.device)

            # actuator_paths = [f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/../../resources/actuator_nets/unitree_go1_new2.pt']
            actuator_paths = [f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/../../resources/actuator_nets/unitree_go1_8_16-{i}.pt'
                    for i in range(0,6)]
            self.num_actuator_networks = len(actuator_paths)
            actuator_networks = [torch.jit.load(actuator_path, map_location=self.device) for actuator_path in actuator_paths]

            def eval_actuator_network(joint_pos, joint_pos_last, joint_pos_last_last, joint_vel, joint_vel_last,
                                      joint_vel_last_last, model_idx=0):
                actuator_network = actuator_networks[model_idx]
                with torch.no_grad():
                    xs = torch.cat((joint_pos.unsqueeze(-1),
                                    joint_pos_last.unsqueeze(-1),
                                    joint_pos_last_last.unsqueeze(-1),
                                    joint_vel.unsqueeze(-1),
                                    joint_vel_last.unsqueeze(-1),
                                    joint_vel_last_last.unsqueeze(-1)), dim=-1)
                    torques = actuator_network(xs.view(self.num_envs * 12, 6))
                return torques.view(self.num_envs, 12)

            self.actuator_network = eval_actuator_network

            # self.joint_pos_err_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_pos_err_last_last_timestep = torch.zeros((self.num_envs, 12), device=self.device)
            # self.joint_pos_err_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_pos_err_last_timestep = torch.zeros((self.num_envs, 12), device=self.device)
            # self.joint_vel_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last_last_timestep = torch.zeros((self.num_envs, 12), device=self.device)
            # self.joint_vel_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last_timestep = torch.zeros((self.num_envs, 12), device=self.device)

            # energy_path = f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/../../resources/energy_nets/unitree_go1_independent_07_31_01.pt'
            energy_path = f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/../../resources/energy_nets/unitree_go1_independent_08_02_01.pt'
            energy_network = torch.jit.load(energy_path, map_location=self.device)

            def eval_energy_network(joint_pos, joint_pos_last, joint_pos_last_last, joint_vel, joint_vel_last,
                                      joint_vel_last_last):
                with torch.no_grad():
                    xs = torch.stack((joint_pos,
                                    joint_pos_last,
                                    joint_pos_last_last,
                                    joint_vel,
                                    joint_vel_last,
                                    joint_vel_last_last), dim=-1)
                    return energy_network(xs) # Shape: (num_envs, 12, 2)

            self.energy_network = eval_energy_network

    def _init_custom_buffers__(self):
        # domain randomization properties
        self.friction_coeffs = self.default_friction * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.restitutions = self.default_restitution * torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.Kp_factors = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.ball_drags = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.ball_restitutions = self.default_restitution * torch.ones(self.num_envs, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.ball_friction_coeffs = self.default_friction * torch.ones(self.num_envs, dtype=torch.float,
                                                                  device=self.device,
                                                                  requires_grad=False)
        self.feet_forces = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device,
                                       requires_grad=False)
        self.target_feet_positions = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device,
                                                  requires_grad=False)
        self.prev_target_feet_positions = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device,
                                                    requires_grad=False)
        self.freed_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        # if custom initialization values were passed in, set them here
        dynamics_params = ["friction_coeffs", "restitutions", "payloads", "com_displacements", "motor_strengths",
                           "Kp_factors", "Kd_factors"]
        if self.initial_dynamics_dict is not None:
            for k, v in self.initial_dynamics_dict.items():
                if k in dynamics_params:
                    setattr(self, k, v.to(self.device))

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.doubletime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
        self.halftime_clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        
        # Deepak commands curriculum 
        if self.cfg.commands.inverse_IK_door_opening:
            self.initial_ee_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) # 3 = (radius, pitch, yaw)
            self.ee_target_pos_cmd = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) 
            self.trajectory_time = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

            # To init the tensors only once at the beginning of a training 
            self.init_training = True

        if self.cfg.commands.teleop_occulus:
            self.teleop_gripper_value = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
            self.teleop_joint6_value = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
            self.teleop_value = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.commands.sample_feasible_commands:
            self.target_joint_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # parkour
        self.reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def randomize_ep_len_buf(self):
        self.episode_length_buf = torch.randint_like(self.episode_length_buf,
                                                             high=int(self.max_episode_length))

    def _init_command_distribution(self, env_ids):
        # new style curriculum
        self.category_names = ['nominal']
        if self.cfg.commands.gaitwise_curricula:
            # self.category_names = ['pronk', 'trot', 'pace', 'bound']
            self.category_names = ['trot']

        if self.cfg.commands.curriculum_type == "RewardThresholdCurriculum":
            from .curriculum import RewardThresholdCurriculum
            CurriculumClass = RewardThresholdCurriculum
        self.curricula = []
        for category in self.category_names:
            if self.cfg.commands.inverse_IK_door_opening:
                self.curricula += [CurriculumClass(seed=self.cfg.commands.curriculum_seed,
                                                    x_vel=(self.cfg.commands.limit_vel_x[0],
                                                            self.cfg.commands.limit_vel_x[1],
                                                            self.cfg.commands.num_bins_vel_x),
                                                    y_vel=(self.cfg.commands.limit_vel_y[0],
                                                            self.cfg.commands.limit_vel_y[1],
                                                            self.cfg.commands.num_bins_vel_y),
                                                    yaw_vel=(self.cfg.commands.limit_vel_yaw[0],
                                                                self.cfg.commands.limit_vel_yaw[1],
                                                                self.cfg.commands.num_bins_vel_yaw),
                                                    body_height=(self.cfg.commands.limit_body_height[0],
                                                                    self.cfg.commands.limit_body_height[1],
                                                                    self.cfg.commands.num_bins_body_height),
                                                    gait_frequency=(self.cfg.commands.limit_gait_frequency[0],
                                                                    self.cfg.commands.limit_gait_frequency[1],
                                                                    self.cfg.commands.num_bins_gait_frequency),
                                                    gait_phase=(self.cfg.commands.limit_gait_phase[0],
                                                                self.cfg.commands.limit_gait_phase[1],
                                                                self.cfg.commands.num_bins_gait_phase),
                                                    gait_offset=(self.cfg.commands.limit_gait_offset[0],
                                                                    self.cfg.commands.limit_gait_offset[1],
                                                                    self.cfg.commands.num_bins_gait_offset),
                                                    gait_bounds=(self.cfg.commands.limit_gait_bound[0],
                                                                    self.cfg.commands.limit_gait_bound[1],
                                                                    self.cfg.commands.num_bins_gait_bound),
                                                    gait_duration=(self.cfg.commands.limit_gait_duration[0],
                                                                    self.cfg.commands.limit_gait_duration[1],
                                                                    self.cfg.commands.num_bins_gait_duration),
                                                    footswing_height=(self.cfg.commands.limit_footswing_height[0],
                                                                        self.cfg.commands.limit_footswing_height[1],
                                                                        self.cfg.commands.num_bins_footswing_height),
                                                    body_pitch=(self.cfg.commands.limit_body_pitch[0],
                                                                self.cfg.commands.limit_body_pitch[1],
                                                                self.cfg.commands.num_bins_body_pitch),
                                                    body_roll=(self.cfg.commands.limit_body_roll[0],
                                                                self.cfg.commands.limit_body_roll[1],
                                                                self.cfg.commands.num_bins_body_roll),
                                                    stance_width=(self.cfg.commands.limit_stance_width[0],
                                                                    self.cfg.commands.limit_stance_width[1],
                                                                    self.cfg.commands.num_bins_stance_width),
                                                    stance_length=(self.cfg.commands.limit_stance_length[0],
                                                                        self.cfg.commands.limit_stance_length[1],
                                                                        self.cfg.commands.num_bins_stance_length),
                                                    aux_reward_coef=(self.cfg.commands.limit_aux_reward_coef[0],
                                                                self.cfg.commands.limit_aux_reward_coef[1],
                                                                self.cfg.commands.num_bins_aux_reward_coef),
                                                    ee_sphe_radius=(self.cfg.commands.limit_ee_sphe_radius[0],
                                                                self.cfg.commands.limit_ee_sphe_radius[1],
                                                                self.cfg.commands.num_bins_ee_sphe_radius),   
                                                    ee_sphe_pitch=(self.cfg.commands.limit_ee_sphe_pitch[0],
                                                                self.cfg.commands.limit_ee_sphe_pitch[1],
                                                                self.cfg.commands.num_bins_ee_sphe_pitch),   
                                                    ee_sphe_yaw=(self.cfg.commands.limit_ee_sphe_yaw[0],
                                                                self.cfg.commands.limit_ee_sphe_yaw[1],
                                                                self.cfg.commands.num_bins_ee_sphe_yaw),    
                                                    ee_timing=(self.cfg.commands.limit_ee_timing[0],
                                                                self.cfg.commands.limit_ee_timing[1],
                                                                self.cfg.commands.num_bins_ee_timing),                                                                                                              
                                                    # end_effector_pos_x=(self.cfg.commands.limit_end_effector_pos_x[0],
                                                    #                     self.cfg.commands.limit_end_effector_pos_x[1],
                                                    #                     self.cfg.commands.num_bins_end_effector_pos_x),
                                                    # end_effector_pos_y=(self.cfg.commands.limit_end_effector_pos_y[0],
                                                    #                     self.cfg.commands.limit_end_effector_pos_y[1],
                                                    #                     self.cfg.commands.num_bins_end_effector_pos_y),    
                                                    # end_effector_pos_z=(self.cfg.commands.limit_end_effector_pos_z[0],
                                                    #                     self.cfg.commands.limit_end_effector_pos_z[1],
                                                    #                     self.cfg.commands.num_bins_end_effector_pos_z),      
                                                    # end_effector_roll=(self.cfg.commands.limit_end_effector_roll[0],
                                                    #                     self.cfg.commands.limit_end_effector_roll[1],
                                                    #                     self.cfg.commands.num_bins_end_effector_roll),      
                                                    # end_effector_pitch=(self.cfg.commands.limit_end_effector_pitch[0],
                                                    #                     self.cfg.commands.limit_end_effector_pitch[1],
                                                    #                     self.cfg.commands.num_bins_end_effector_pitch),   
                                                    # end_effector_yaw=(self.cfg.commands.limit_end_effector_yaw[0],
                                                    #                     self.cfg.commands.limit_end_effector_yaw[1],
                                                    #                     self.cfg.commands.num_bins_end_effector_yaw),   
                                                    # end_effector_gripper=(self.cfg.commands.limit_end_effector_gripper[0],
                                                    #                     self.cfg.commands.limit_end_effector_gripper[1],
                                                    #                     self.cfg.commands.num_bins_end_effector_gripper),                                                                                                                                                                                                                                                                                                                                                              
                                )]
            else:
                self.curricula += [CurriculumClass(seed=self.cfg.commands.curriculum_seed,
                                                x_vel=(self.cfg.commands.limit_vel_x[0],
                                                        self.cfg.commands.limit_vel_x[1],
                                                        self.cfg.commands.num_bins_vel_x),
                                                y_vel=(self.cfg.commands.limit_vel_y[0],
                                                        self.cfg.commands.limit_vel_y[1],
                                                        self.cfg.commands.num_bins_vel_y),
                                                yaw_vel=(self.cfg.commands.limit_vel_yaw[0],
                                                            self.cfg.commands.limit_vel_yaw[1],
                                                            self.cfg.commands.num_bins_vel_yaw),
                                                body_height=(self.cfg.commands.limit_body_height[0],
                                                                self.cfg.commands.limit_body_height[1],
                                                                self.cfg.commands.num_bins_body_height),
                                                gait_frequency=(self.cfg.commands.limit_gait_frequency[0],
                                                                self.cfg.commands.limit_gait_frequency[1],
                                                                self.cfg.commands.num_bins_gait_frequency),
                                                gait_phase=(self.cfg.commands.limit_gait_phase[0],
                                                            self.cfg.commands.limit_gait_phase[1],
                                                            self.cfg.commands.num_bins_gait_phase),
                                                gait_offset=(self.cfg.commands.limit_gait_offset[0],
                                                                self.cfg.commands.limit_gait_offset[1],
                                                                self.cfg.commands.num_bins_gait_offset),
                                                gait_bounds=(self.cfg.commands.limit_gait_bound[0],
                                                                self.cfg.commands.limit_gait_bound[1],
                                                                self.cfg.commands.num_bins_gait_bound),
                                                gait_duration=(self.cfg.commands.limit_gait_duration[0],
                                                                self.cfg.commands.limit_gait_duration[1],
                                                                self.cfg.commands.num_bins_gait_duration),
                                                footswing_height=(self.cfg.commands.limit_footswing_height[0],
                                                                    self.cfg.commands.limit_footswing_height[1],
                                                                    self.cfg.commands.num_bins_footswing_height),
                                                body_pitch=(self.cfg.commands.limit_body_pitch[0],
                                                            self.cfg.commands.limit_body_pitch[1],
                                                            self.cfg.commands.num_bins_body_pitch),
                                                body_roll=(self.cfg.commands.limit_body_roll[0],
                                                            self.cfg.commands.limit_body_roll[1],
                                                            self.cfg.commands.num_bins_body_roll),
                                                stance_width=(self.cfg.commands.limit_stance_width[0],
                                                                self.cfg.commands.limit_stance_width[1],
                                                                self.cfg.commands.num_bins_stance_width),
                                                stance_length=(self.cfg.commands.limit_stance_length[0],
                                                                    self.cfg.commands.limit_stance_length[1],
                                                                    self.cfg.commands.num_bins_stance_length),
                                                aux_reward_coef=(self.cfg.commands.limit_aux_reward_coef[0],
                                                            self.cfg.commands.limit_aux_reward_coef[1],
                                                            self.cfg.commands.num_bins_aux_reward_coef),
                                                )]
        if self.cfg.commands.curriculum_type == "LipschitzCurriculum":
            for curriculum in self.curricula:
                curriculum.set_params(lipschitz_threshold=self.cfg.commands.lipschitz_threshold,
                                      binary_phases=self.cfg.commands.binary_phases)
        self.env_command_bins = np.zeros(len(env_ids), dtype=np.int32)
        self.env_command_categories = np.zeros(len(env_ids), dtype=np.int32)

        if self.cfg.commands.inverse_IK_door_opening:
            low = np.array(
                [self.cfg.commands.lin_vel_x[0], self.cfg.commands.lin_vel_y[0],
                self.cfg.commands.ang_vel_yaw[0], self.cfg.commands.body_height_cmd[0],
                self.cfg.commands.gait_frequency_cmd_range[0],
                self.cfg.commands.gait_phase_cmd_range[0], self.cfg.commands.gait_offset_cmd_range[0],
                self.cfg.commands.gait_bound_cmd_range[0], self.cfg.commands.gait_duration_cmd_range[0],
                self.cfg.commands.footswing_height_range[0], self.cfg.commands.body_pitch_range[0],
                self.cfg.commands.body_roll_range[0],self.cfg.commands.stance_width_range[0],
                self.cfg.commands.stance_length_range[0], self.cfg.commands.aux_reward_coef_range[0], 
                self.cfg.commands.ee_sphe_radius[0], self.cfg.commands.ee_sphe_pitch[0], 
                self.cfg.commands.ee_sphe_yaw[0], self.cfg.commands.ee_timing[0],])
            high = np.array(
                [self.cfg.commands.lin_vel_x[1], self.cfg.commands.lin_vel_y[1],
                self.cfg.commands.ang_vel_yaw[1], self.cfg.commands.body_height_cmd[1],
                self.cfg.commands.gait_frequency_cmd_range[1],
                self.cfg.commands.gait_phase_cmd_range[1], self.cfg.commands.gait_offset_cmd_range[1],
                self.cfg.commands.gait_bound_cmd_range[1], self.cfg.commands.gait_duration_cmd_range[1],
                self.cfg.commands.footswing_height_range[1], self.cfg.commands.body_pitch_range[1],
                self.cfg.commands.body_roll_range[1],self.cfg.commands.stance_width_range[1],
                self.cfg.commands.stance_length_range[1], self.cfg.commands.aux_reward_coef_range[1], 
                self.cfg.commands.ee_sphe_radius[1], self.cfg.commands.ee_sphe_pitch[1], 
                self.cfg.commands.ee_sphe_yaw[1], self.cfg.commands.ee_timing[1],])
            # low = np.array(
            #     [self.cfg.commands.lin_vel_x[0], self.cfg.commands.lin_vel_y[0],
            #     self.cfg.commands.ang_vel_yaw[0], self.cfg.commands.body_height_cmd[0],
            #     self.cfg.commands.gait_frequency_cmd_range[0],
            #     self.cfg.commands.gait_phase_cmd_range[0], self.cfg.commands.gait_offset_cmd_range[0],
            #     self.cfg.commands.gait_bound_cmd_range[0], self.cfg.commands.gait_duration_cmd_range[0],
            #     self.cfg.commands.footswing_height_range[0], self.cfg.commands.body_pitch_range[0],
            #     self.cfg.commands.body_roll_range[0],self.cfg.commands.stance_width_range[0],
            #     self.cfg.commands.stance_length_range[0], self.cfg.commands.aux_reward_coef_range[0], 
            #     self.cfg.commands.end_effector_pos_x[0], self.cfg.commands.end_effector_pos_y[0], 
            #     self.cfg.commands.end_effector_pos_z[0], self.cfg.commands.end_effector_roll[0], 
            #     self.cfg.commands.end_effector_pitch[0], self.cfg.commands.end_effector_yaw[0],
            #     self.cfg.commands.end_effector_gripper[0],])
            # high = np.array(
            #     [self.cfg.commands.lin_vel_x[1], self.cfg.commands.lin_vel_y[1],
            #     self.cfg.commands.ang_vel_yaw[1], self.cfg.commands.body_height_cmd[1],
            #     self.cfg.commands.gait_frequency_cmd_range[1],
            #     self.cfg.commands.gait_phase_cmd_range[1], self.cfg.commands.gait_offset_cmd_range[1],
            #     self.cfg.commands.gait_bound_cmd_range[1], self.cfg.commands.gait_duration_cmd_range[1],
            #     self.cfg.commands.footswing_height_range[1], self.cfg.commands.body_pitch_range[1],
            #     self.cfg.commands.body_roll_range[1],self.cfg.commands.stance_width_range[1],
            #     self.cfg.commands.stance_length_range[1], self.cfg.commands.aux_reward_coef_range[1], 
            #     self.cfg.commands.end_effector_pos_x[1], self.cfg.commands.end_effector_pos_y[1], 
            #     self.cfg.commands.end_effector_pos_z[1], self.cfg.commands.end_effector_roll[1], 
            #     self.cfg.commands.end_effector_pitch[1], self.cfg.commands.end_effector_yaw[1],
            #     self.cfg.commands.end_effector_gripper[1],])
        else:
            low = np.array(
                [self.cfg.commands.lin_vel_x[0], self.cfg.commands.lin_vel_y[0],
                self.cfg.commands.ang_vel_yaw[0], self.cfg.commands.body_height_cmd[0],
                self.cfg.commands.gait_frequency_cmd_range[0],
                self.cfg.commands.gait_phase_cmd_range[0], self.cfg.commands.gait_offset_cmd_range[0],
                self.cfg.commands.gait_bound_cmd_range[0], self.cfg.commands.gait_duration_cmd_range[0],
                self.cfg.commands.footswing_height_range[0], self.cfg.commands.body_pitch_range[0],
                self.cfg.commands.body_roll_range[0],self.cfg.commands.stance_width_range[0],
                self.cfg.commands.stance_length_range[0], self.cfg.commands.aux_reward_coef_range[0], ])
            high = np.array(
                [self.cfg.commands.lin_vel_x[1], self.cfg.commands.lin_vel_y[1],
                self.cfg.commands.ang_vel_yaw[1], self.cfg.commands.body_height_cmd[1],
                self.cfg.commands.gait_frequency_cmd_range[1],
                self.cfg.commands.gait_phase_cmd_range[1], self.cfg.commands.gait_offset_cmd_range[1],
                self.cfg.commands.gait_bound_cmd_range[1], self.cfg.commands.gait_duration_cmd_range[1],
                self.cfg.commands.footswing_height_range[1], self.cfg.commands.body_pitch_range[1],
                self.cfg.commands.body_roll_range[1],self.cfg.commands.stance_width_range[1],
                self.cfg.commands.stance_length_range[1], self.cfg.commands.aux_reward_coef_range[1], ])            
        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # reward containers
        from go1_gym.rewards.corl_rewards import CoRLRewards
        from go1_gym.rewards.energy_efficiency_rewards import EnergyEfficiencyRewards
        from go1_gym.rewards.stair_rewards import StairRewards
        from go1_gym.rewards.soccer_rewards import SoccerRewards
        from go1_gym.rewards.door_opening_rewards import DoorOpeningRewards
        from go1_gym.rewards.inverse_kinematics_rewards import InverseKinematicsRewards
        from go1_gym.rewards.vanilla_inverse_kinematics_rewards import VanillaInverseKinematicsRewards
        from go1_gym.rewards.bc_estimation_rewards import BCEstimationRewards
        from go1_gym.rewards.parkour_rewards import ParkourRewards
        from go1_gym.rewards.parkour_dribbling_rewards import ParkourDribblingRewards
        reward_containers = {"CoRLRewards": CoRLRewards, "EnergyEfficiencyRewards": EnergyEfficiencyRewards, "StairRewards": StairRewards, "SoccerRewards": SoccerRewards, "DoorOpeningRewards": DoorOpeningRewards, 
                             "BCEstimationRewards": BCEstimationRewards, "InverseKinematicsRewards": InverseKinematicsRewards,
                             "VanillaInverseKinematicsRewardlin_vels": VanillaInverseKinematicsRewards,
                             "ParkourRewards": ParkourRewards, "ParkourDribblingRewards": ParkourDribblingRewards}
        

        self.reward_container = reward_containers[self.cfg.rewards.reward_container_name](self)

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue

            if not hasattr(self.reward_container, '_reward_' + name):
                print(f"Warning: reward {'_reward_' + name} has nonzero coefficient but was not found!")
            else:
                self.reward_names.append(name)
                self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()}
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in
            list(self.reward_scales.keys()) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual",
                                               "ep_timesteps"]}

    def _create_envs(self):

        all_assets = []

        # create robot

        from go1_gym.robots.go1 import Go1
        from go1_gym.robots.b1 import B1
        from go1_gym.robots.b1_plus_z1 import B1PlusZ1
        from go1_gym.robots.z1 import Z1
        from go1_gym.robots.z1_3dof import Z1_3DOF

        robot_classes = {
            'go1': Go1,
            'b1': B1,
            'b1_plus_z1': B1PlusZ1,
            'z1': Z1,
            'z1_3dof': Z1_3DOF,
        }

        self.robot = robot_classes[self.cfg.robot.name](self)
        all_assets.append(self.robot)
        self.robot_asset, dof_props_asset, rigid_shape_props_asset = self.robot.initialize()
        

        object_init_state_list = self.cfg.object.ball_init_pos + self.cfg.object.ball_init_rot + self.cfg.object.ball_init_lin_vel + self.cfg.object.ball_init_ang_vel
        self.object_init_state = to_torch(object_init_state_list, device=self.device, requires_grad=False)

        # create objects

        # from go1_gym.assets.ball import Ball
        # from go1_gym.assets.cube import Cube
        # from go1_gym.assets.door import Door
        # from go1_gym.assets.chair import Chair
        # from go1_gym.assets.bucket_of_balls import BallBucket

        # asset_classes = {
        #     "ball": Ball,
        #     "cube": Cube,
        #     "door": Door,
        #     "chair": Chair,
        #     "ballbucket": BallBucket,
        # }

        if self.cfg.env.add_objects:
            # if there is a list of assets
            self.asset = asset_classes[self.cfg.object.asset](self)
            all_assets.append(self.asset)
            self.ball_asset, ball_rigid_shape_props_asset = self.asset.initialize()
            self.ball_force_feedback = self.asset.get_force_feedback()
            self.num_object_bodies = self.gym.get_asset_rigid_body_count(self.ball_asset)
        else:
            self.ball_force_feedback = None
            self.num_object_bodies = 0

        # aggregate the asset properties
        self.total_rigid_body_num = sum([asset.get_num_bodies() for asset in 
                                        all_assets])
        self.num_dof_all = sum([asset.get_num_dof() for asset in
                            all_assets])
        # self.num_actuated_dof = sum([asset.get_num_actuated_dof() for asset in
        #                                 all_assets])
        self.num_actuated_dof = self.robot.get_num_actuated_dof()

        if self.cfg.terrain.mesh_type == "boxes":
            self.total_rigid_body_num += self.cfg.terrain.num_cols * self.cfg.terrain.num_rows


        self.ball_init_pose = gymapi.Transform()
        self.ball_init_pose.p = gymapi.Vec3(*self.object_init_state[:3])

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.terrain_levels = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self.terrain_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.terrain_types = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
        self._get_env_origins(torch.arange(self.num_envs, device=self.device), self.cfg)
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.robot_actor_handles = []
        self.object_actor_handles = []
        self.imu_sensor_handles = []
        self.envs = []
        self.robot_actor_idxs = []
        self.object_actor_idxs = []

        self.object_rigid_body_idxs = []
        self.feet_rigid_body_idxs = []
        self.robot_rigid_body_idxs = []

        self.default_friction = rigid_shape_props_asset[1].friction
        self.default_restitution = rigid_shape_props_asset[1].restitution
        self._init_custom_buffers__()
        self._randomize_rigid_body_props(torch.arange(self.num_envs, device=self.device), self.cfg)
        self._randomize_gravity()
        self._randomize_ball_drag()
        # self._randomize_feet_forces()

        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        if self.cfg.env.all_agents_share:
            shared_env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

        for i in range(self.num_envs):
            # create env instance
            if self.cfg.env.all_agents_share:
                env_handle = shared_env
            else:
                env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            # env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[0:1] += torch_rand_float(-self.cfg.terrain.x_init_range, self.cfg.terrain.x_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            pos[1:2] += torch_rand_float(-self.cfg.terrain.y_init_range, self.cfg.terrain.y_init_range, (1, 1),
                                         device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            # add robots
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)
            robot_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, "robot", i,
                                                  self.cfg.asset.self_collisions, 0)
            for bi in body_names:
                self.robot_rigid_body_idxs.append(self.gym.find_actor_rigid_body_handle(env_handle, robot_handle, bi))
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, robot_handle, body_props, recomputeInertia=True)
            
            self.robot_actor_handles.append(robot_handle)
            self.robot_actor_idxs.append(self.gym.get_actor_index(env_handle, robot_handle, gymapi.DOMAIN_SIM))

            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)

            # add objects
            if self.cfg.env.add_objects:
                ball_rigid_shape_props = self._process_ball_rigid_shape_props(ball_rigid_shape_props_asset, i)
                self.gym.set_asset_rigid_shape_properties(self.ball_asset, ball_rigid_shape_props)
                ball_handle = self.gym.create_actor(env_handle, self.ball_asset, self.ball_init_pose, "ball", i, 0)
                color = gymapi.Vec3(1, 1, 0)
                # self.gym.set_rigid_body_color(env_handle, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                texture_path = f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/../../resources/textures/soccer_ball_texture.jpeg'
                # texture_path = f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/../../resources/textures/snow.jpg'
                texture = self.gym.create_texture_from_file(self.sim, texture_path)
                self.gym.set_rigid_body_texture(env_handle, ball_handle, 0, gymapi.MeshType.MESH_VISUAL_AND_COLLISION, texture)
                ball_idx = self.gym.get_actor_rigid_body_index(env_handle, ball_handle, 0, gymapi.DOMAIN_SIM)
                ball_body_props = self.gym.get_actor_rigid_body_properties(env_handle, ball_handle)
                ball_body_props[0].mass = self.cfg.object.mass*(np.random.rand()*0.3+0.5)
                self.gym.set_actor_rigid_body_properties(env_handle, ball_handle, ball_body_props, recomputeInertia=True)
                # self.gym.set_actor_rigid_shape_properties(env_handle, ball_handle, ball_shape_props)
                self.object_actor_handles.append(ball_handle)
                self.object_rigid_body_idxs.append(ball_idx)
                self.object_actor_idxs.append(self.gym.get_actor_index(env_handle, ball_handle, gymapi.DOMAIN_SIM))
                

            self.envs.append(env_handle)

        self.robot_actor_idxs = torch.Tensor(self.robot_actor_idxs).to(device=self.device,dtype=torch.long)
        self.object_actor_idxs = torch.Tensor(self.object_actor_idxs).to(device=self.device,dtype=torch.long)
        self.object_rigid_body_idxs = torch.Tensor(self.object_rigid_body_idxs).to(device=self.device,dtype=torch.long)
            
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.robot_actor_handles[0],
                                                                         feet_names[i])

        

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.robot_actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.robot_actor_handles[0],
                                                                                        termination_contact_names[i])
        ################
        ### Add sensors
        ################

        self.initialize_sensors()
        
        # if perception is on, set up camera
        if self.cfg.perception.compute_segmentation or self.cfg.perception.compute_rgb or self.cfg.perception.compute_depth:
            self.initialize_cameras(range(self.num_envs))

        if self.cfg.perception.measure_heights:
            from go1_gym.sensors.heightmap_sensor import HeightmapSensor
            self.heightmap_sensor = HeightmapSensor(self)

        # if recording video, set up camera
        if self.cfg.env.record_video:
            from go1_gym.sensors.floating_camera_sensor import FloatingCameraSensor
            self.rendering_camera = FloatingCameraSensor(self)
            

        ################
        ### Initialize Logging
        ################

        from go1_gym.utils.logger import Logger
        self.logger = Logger(self)
        
        self.video_writer = None
        self.video_frames = []
        self.complete_video_frames = []


    def render(self, mode="rgb_array", target_loc=None, cam_distance=None):
        self.rendering_camera.set_position(target_loc, cam_distance)
        return self.rendering_camera.get_observation()

    def _render_headless(self):
        if self.record_now and self.complete_video_frames is not None and len(self.complete_video_frames) == 0:
            bx, by, bz = self.root_states[self.robot_actor_idxs[0], 0], self.root_states[self.robot_actor_idxs[0], 1], self.root_states[self.robot_actor_idxs[0], 2]
            target_loc = [bx, by , bz]
            cam_distance = [0, -1.2, 1.2]
            self.rendering_camera.set_position(target_loc, cam_distance)
            self.video_frame = self.rendering_camera.get_observation()
            self.video_frames.append(self.video_frame)

    def start_recording(self):
        self.complete_video_frames = None
        self.record_now = True

    def pause_recording(self):
        self.complete_video_frames = []
        self.video_frames = []
        self.record_now = False

    def get_complete_frames(self):
        if self.complete_video_frames is None:
            return []
        return self.complete_video_frames

    def _get_env_origins(self, env_ids, cfg):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            # put robots at the origins defined by the terrain
            max_init_level = cfg.terrain.max_init_terrain_level
            min_init_level = cfg.terrain.min_init_terrain_level
            if not cfg.terrain.curriculum: max_init_level = cfg.terrain.num_rows - 1
            if not cfg.terrain.curriculum: min_init_level = 0
            if cfg.terrain.center_robots:
                min_terrain_level = cfg.terrain.num_rows // 2 - cfg.terrain.center_span
                max_terrain_level = cfg.terrain.num_rows // 2 + cfg.terrain.center_span - 1
                min_terrain_type = cfg.terrain.num_cols // 2 - cfg.terrain.center_span
                max_terrain_type = cfg.terrain.num_cols // 2 + cfg.terrain.center_span - 1
                self.terrain_levels[env_ids] = torch.randint(min_terrain_level, max_terrain_level + 1, (len(env_ids),),
                                                             device=self.device)
                self.terrain_types[env_ids] = torch.randint(min_terrain_type, max_terrain_type + 1, (len(env_ids),),
                                                            device=self.device)
            else:
                self.terrain_levels[env_ids] = torch.randint(min_init_level, max_init_level + 1, (len(env_ids),),
                                                            device=self.device)
                self.terrain_types[env_ids] = torch.div(torch.arange(len(env_ids), device=self.device),
                                                    (len(env_ids) / cfg.terrain.num_cols), rounding_mode='floor').to(
                    torch.long)
            cfg.terrain.max_terrain_level = cfg.terrain.num_rows
            self.max_terrain_level = cfg.terrain.num_rows
            self.min_terrain_level = cfg.terrain.num_border_boxes
            cfg.terrain.terrain_origins = torch.from_numpy(cfg.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[env_ids] = cfg.terrain.terrain_origins[
                self.terrain_levels[env_ids], self.terrain_types[env_ids]]
            
            # parkour specific
            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]
            self.terrain_goals = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)
            self.env_goals = torch.zeros(self.num_envs, self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs, 3, device=self.device, requires_grad=False)
            self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, requires_grad=False, dtype=torch.long)
            temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
            last_col = temp[:, -1].unsqueeze(1)
            self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)), dim=1)[:]
            self.cur_goals = self._gather_cur_goals()
            self.next_goals = self._gather_cur_goals(future=1)


        elif cfg.terrain.mesh_type in ["boxes", "boxes_tm"]:
            self.custom_origins = True
            # put robots at the origins defined by the terrain
            max_init_level = int(cfg.terrain.max_init_terrain_level + cfg.terrain.num_border_boxes)
            min_init_level = int(cfg.terrain.min_init_terrain_level + cfg.terrain.num_border_boxes)
            if not cfg.terrain.curriculum: max_init_level = int(cfg.terrain.num_rows - 1 - cfg.terrain.num_border_boxes)
            if not cfg.terrain.curriculum: min_init_level = int(0 + cfg.terrain.num_border_boxes)

            if cfg.terrain.center_robots:
                self.min_terrain_level = cfg.terrain.num_rows // 2 - cfg.terrain.center_span
                self.max_terrain_level = cfg.terrain.num_rows // 2 + cfg.terrain.center_span - 1
                min_terrain_type = cfg.terrain.num_cols // 2 - cfg.terrain.center_span
                max_terrain_type = cfg.terrain.num_cols // 2 + cfg.terrain.center_span - 1
                self.terrain_levels[env_ids] = torch.randint(self.min_terrain_level, self.max_terrain_level + 1, (len(env_ids),),
                                                             device=self.device)
                self.terrain_types[env_ids] = torch.randint(min_terrain_type, max_terrain_type + 1, (len(env_ids),),
                                                            device=self.device)
            else:
                self.terrain_levels[env_ids] = torch.randint(min_init_level, max_init_level + 1, (len(env_ids),),
                                                             device=self.device)
                self.terrain_types[env_ids] = (torch.div(torch.arange(len(env_ids), device=self.device),
                                                        (len(env_ids) / (cfg.terrain.num_cols - 2 * cfg.terrain.num_border_boxes)),
                                                        rounding_mode='floor') + cfg.terrain.num_border_boxes).to(torch.long)
                self.min_terrain_level = int(cfg.terrain.num_border_boxes)
                self.max_terrain_level = int(cfg.terrain.num_rows - cfg.terrain.num_border_boxes)
            cfg.terrain.env_origins[:, :, 2] = self.terrain_obj.terrain_cell_center_heights.cpu().numpy()
            cfg.terrain.terrain_origins = torch.from_numpy(cfg.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[env_ids] = cfg.terrain.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
            # self.env_origins[env_ids, 2] = self.terrain_cell_center_heights[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        else:
            self.custom_origins = False
            # create a grid of robots
            num_cols = np.floor(np.sqrt(len(env_ids)))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = cfg.env.env_spacing
            self.env_origins[env_ids, 0] = spacing * xx.flatten()[:len(env_ids)]
            self.env_origins[env_ids, 1] = spacing * yy.flatten()[:len(env_ids)]
            self.env_origins[env_ids, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.obs_scales
        self.reward_scales = vars(self.cfg.reward_scales)
        self.curriculum_thresholds = vars(self.cfg.curriculum_thresholds)
        cfg.command_ranges = vars(cfg.commands)
        if cfg.terrain.mesh_type not in ['heightfield', 'trimesh', 'boxes', 'boxes_tm']:
            cfg.terrain.curriculum = False
        self.max_episode_length_s = cfg.env.episode_length_s
        cfg.env.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.max_episode_length = cfg.env.max_episode_length

        cfg.domain_rand.push_interval = np.ceil(cfg.domain_rand.push_interval_s / self.dt)
        cfg.domain_rand.rand_interval = np.ceil(cfg.domain_rand.rand_interval_s / self.dt)
        cfg.domain_rand.gravity_rand_interval = np.ceil(cfg.domain_rand.gravity_rand_interval_s / self.dt)
        cfg.domain_rand.ball_drag_rand_interval = np.ceil(cfg.domain_rand.ball_drag_rand_interval_s / self.dt)
        cfg.domain_rand.gravity_rand_duration = np.ceil(
            cfg.domain_rand.gravity_rand_interval * cfg.domain_rand.gravity_impulse_duration)
        cfg.domain_rand.foot_height_forced_rand_interval = np.ceil(cfg.domain_rand.foot_height_forced_rand_interval_s / self.dt)
    
    def _draw_goals(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(1, 0, 0))
        sphere_geom_cur = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0, 0, 1))
        sphere_geom_reached = gymutil.WireframeSphereGeometry(self.cfg.env.next_goal_threshold, 32, 32, None, color=(0, 1, 0))
        goals = self.terrain_goals[self.terrain_levels[self.lookat_id], self.terrain_types[self.lookat_id]].cpu().numpy()
        for i, goal in enumerate(goals):
            goal_xy = goal[:2] + self.terrain.cfg.border_size
            pts = (goal_xy/self.terrain.cfg.horizontal_scale).astype(int)
            goal_z = self.height_samples[pts[0], pts[1]].cpu().item() * self.terrain.cfg.vertical_scale
            pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal_z), r=None)
            if i == self.cur_goal_idx[self.lookat_id].cpu().item():
                gymutil.draw_lines(sphere_geom_cur, self.gym, self.viewer, self.envs[self.lookat_id], pose)
                if self.reached_goal_ids[self.lookat_id]:
                    gymutil.draw_lines(sphere_geom_reached, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            else:
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose)
        
        if not self.cfg.perception.compute_depth:
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))
            pose_robot = self.root_states[self.lookat_id, :3].cpu().numpy()
            for i in range(5):
                norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.target_pos_rel / (norm + 1e-5)
                pose_arrow = pose_robot[:2] + 0.1*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)
            
            sphere_geom_arrow = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 1, 0.5))
            for i in range(5):
                norm = torch.norm(self.next_target_pos_rel, dim=-1, keepdim=True)
                target_vec_norm = self.next_target_pos_rel / (norm + 1e-5)
                pose_arrow = pose_robot[:2] + 0.2*(i+3) * target_vec_norm[self.lookat_id, :2].cpu().numpy()
                pose = gymapi.Transform(gymapi.Vec3(pose_arrow[0], pose_arrow[1], pose_robot[2]), r=None)
                gymutil.draw_lines(sphere_geom_arrow, self.gym, self.viewer, self.envs[self.lookat_id], pose)

    def _init_height_points(self, env_ids, cfg):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(cfg.perception.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(cfg.perception.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        cfg.perception.num_height_points = grid_x.numel()
        points = torch.zeros(len(env_ids), cfg.perception.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids, cfg):
        if cfg.terrain.mesh_type == 'plane':
            return torch.zeros(len(env_ids), cfg.perception.num_height_points, device=self.device, requires_grad=False)
        elif cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, cfg.perception.num_height_points),
                                self.height_points[env_ids]) + (self.root_states[self.robot_actor_idxs[env_ids], :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.terrain_obj.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.terrain_obj.height_samples.shape[1] - 2)

        heights1 = self.terrain_obj.height_samples[px, py]
        heights2 = self.terrain_obj.height_samples[px + 1, py]
        heights3 = self.terrain_obj.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(len(env_ids), -1) * self.terrain.cfg.vertical_scale

    def get_heights_points(self, global_positions):
        points = global_positions + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, 0]
        py = points[:, 1]
        px = torch.clip(px, 0, self.terrain_obj.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.terrain_obj.height_samples.shape[1] - 2)
        heights = self.terrain_obj.height_samples[px, py]
        return heights * self.terrain.cfg.vertical_scale
    
    def get_frictions(self, env_ids, cfg):
        if cfg.terrain.mesh_type == 'plane':
            return torch.zeros(len(env_ids), cfg.perception.num_height_points, device=self.device, requires_grad=False)
        elif cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure friction field with terrain mesh type 'none'")

        points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, cfg.perception.num_height_points),
                                self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)

        # points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.terrain_obj.friction_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.terrain_obj.friction_samples.shape[1] - 2)

        frictions = self.terrain_obj.friction_samples[px, py]

        return frictions.view(len(env_ids), -1)
    
    def get_frictions_points(self, global_positions):
        points = global_positions + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, 0]
        py = points[:, 1]
        px = torch.clip(px, 0, self.terrain_obj.friction_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.terrain_obj.friction_samples.shape[1] - 2)
        frictions = self.terrain_obj.friction_samples[px, py]
        return frictions
    
    def initialize_cameras(self, env_ids):
        self.cams = {label: [] for label in self.cfg.perception.camera_names}
        self.camera_sensors = {}

        from go1_gym.sensors.attached_camera_sensor import AttachedCameraSensor

        for camera_label, camera_pose, camera_rpy, camera_gimbal in zip(self.cfg.perception.camera_names,
                                                             self.cfg.perception.camera_poses,
                                                             self.cfg.perception.camera_rpys,
                                                             self.cfg.perception.camera_gimbals):
            self.camera_sensors[camera_label] = AttachedCameraSensor(self)
            self.camera_sensors[camera_label].initialize(camera_label, camera_pose, camera_rpy, camera_gimbal, env_ids=env_ids)
        
    def get_segmentation_images(self, env_ids):
        segmentation_images = []
        for camera_name in self.cfg.perception.camera_names:
            segmentation_images = self.camera_sensors[camera_name].get_segmentation_images(env_ids)
        return segmentation_images

    def get_rgb_images(self, env_ids):
        rgb_images = {}
        for camera_name in self.cfg.perception.camera_names:
            rgb_images[camera_name] = self.camera_sensors[camera_name].get_rgb_images(env_ids)
        return rgb_images

    def get_depth_images(self, env_ids):
        depth_images = {}
        for camera_name in self.cfg.perception.camera_names:
            depth_images[camera_name] = self.camera_sensors[camera_name].get_depth_images(env_ids)
        return depth_images