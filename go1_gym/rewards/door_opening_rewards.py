import torch
import numpy as np
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from .rewards import Rewards

class DoorOpeningRewards(Rewards):
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.env.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        # k_qd = -6e-4
        return torch.sum(torch.square(self.env.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel) / self.env.dt), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions - self.env.actions), dim=1)
    
    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_contact = self.env.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
        return reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
        return reward / 4

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.actuated_dof_pos[:, :self.env.num_actuated_dof] - self.env.dof_pos_limits[:self.env.num_actuated_dof, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.actuated_dof_pos[:, :self.env.num_actuated_dof] - self.env.dof_pos_limits[:self.env.num_actuated_dof, 1]).clip(min=0.)
        # print(self.env.dof_pos[:, :self.env.num_actuated_dof], self.env.dof_pos_limits[:self.env.num_actuated_dof])
        return torch.sum(out_of_limits, dim=1)

    def _reward_feet_clearance_cmd_linear(self):
        reference_points = self.env.foot_positions[:, :, 0:2].view(-1, 2)
        reference_heights = self.env.get_heights_points(reference_points).view(-1, 4)
        phases = 1 - torch.abs(1.0 - torch.clip((self.env.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1) - reference_heights
        target_height = self.env.commands[:, 9].unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.env.desired_contact_states)
        return torch.sum(rew_foot_clearance, dim=1)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.env.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)
    
    def _reward_jump(self):
        reference_heights = 0
        body_height = self.env.base_pos[:, 2] - reference_heights
        jump_height_target = self.env.commands[:, 3] + self.env.cfg.rewards.base_height_target
        reward = - torch.square(body_height - jump_height_target)
        return reward
    
    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        roll_pitch_commands = self.env.commands[:, 10:12]
        quat_roll = quat_from_angle_axis(-roll_pitch_commands[:, 1],
                                         torch.tensor([1, 0, 0], device=self.env.device, dtype=torch.float))
        quat_pitch = quat_from_angle_axis(-roll_pitch_commands[:, 0],
                                          torch.tensor([0, 1, 0], device=self.env.device, dtype=torch.float))

        desired_base_quat = quat_mul(quat_roll, quat_pitch)
        desired_projected_gravity = quat_rotate_inverse(desired_base_quat, self.env.gravity_vec)

        return torch.sum(torch.square(self.env.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.env.foot_positions - self.env.base_pos.unsqueeze(1)
        
        # print(self.env.foot_positions[0, :])
        # print(self.env.base_pos[0, :])
        
        footsteps_in_body_frame = torch.zeros(self.env.num_envs, 4, 3, device=self.env.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.env.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        if self.env.cfg.commands.num_commands >= 13:
            desired_stance_width = self.env.commands[:, 12:13]
            desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], dim=1)
        else:
            desired_stance_width = 0.3
            desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.env.device).unsqueeze(0)

        if self.env.cfg.commands.num_commands >= 14:
            desired_stance_length = self.env.commands[:, 13:14]
            desired_xs_nom = torch.cat([desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], dim=1)
        else:
            desired_stance_length = 0.45
            desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.env.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.env.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = self.env.commands[:, 4]
        x_vel_des = self.env.commands[:, 0:1]
        yaw_vel_des = self.env.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward

    def _reward_dof_pos(self):
        # Penalize dof positions
        # k_q = -0.75
        # print("actuated: ", self.env.actuated_dof_pos[:, :self.env.num_actuated_dof], " default_dof_pos: ", self.env.default_actuated_dof_pos[:, :self.env.num_actuated_dof])
        return torch.sum(torch.square(self.env.actuated_dof_pos[:, :self.env.num_actuated_dof] - self.env.default_actuated_dof_pos[:, :self.env.num_actuated_dof]), dim=1)

    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        # k_s1 =-2.5
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - self.env.last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:,:self.env.num_actuated_dof] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        # k_s2 = -1.2
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - 2 * self.env.last_joint_pos_target[:, :self.env.num_actuated_dof] + self.env.last_last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:,:self.env.num_actuated_dof] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:,:self.env.num_actuated_dof] != 0)  # ignore second step
        return torch.sum(diff, dim=1)
    
    def _reward_robot_door_pos(self):
        # reward robot walking to the door
        handle_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.object_actor_handles[0], "handle") - self.env.num_bodies
        handle_pos_global = self.env.rigid_body_state_object.view(self.env.num_envs, -1, 13)[:,handle_idx,0:3].view(self.env.num_envs,3)
        handle_pos_body = quat_rotate_inverse(self.env.base_quat, handle_pos_global - self.env.base_pos)
        # robot_body_pos_global = self.env.base_pos

        target_handle_pos_body = torch.tensor([0.8, 0.0, 0.0], device=self.env.device).unsqueeze(0)
        robot_handle_pos_error = torch.norm(handle_pos_body[:, 0:2] - target_handle_pos_body[:, 0:2], dim=1)

        # if error is large, reward velocity towards door
        far_envs = robot_handle_pos_error > 0.2
        body_velocity_target = torch.zeros_like(handle_pos_body[:, 0:2])
        body_velocity_target[far_envs] = handle_pos_body[far_envs, 0:2] - target_handle_pos_body[:, 0:2]
        # if robot_handle_pos_error > 0.2:
        #     body_velocity_target = 
        #     body_velocity_target = body_velocity_target / torch.norm(body_velocity_target, dim=1)
        # # else:
        #     body_velocity_target = 0
        body_velocity = self.env.base_lin_vel[:, 0:2]
        body_velocity_error = torch.norm(body_velocity_target - body_velocity, dim=1)
        
        # robot_handle_distance = torch.norm(robot_body_pos_global - handle_pos_global, dim=1)
        # margin = 1.1
        # robot_handle_distance_clip = torch.clip(robot_handle_distance - margin, 0.0, 10.0)
        
        # return torch.exp(-robot_handle_pos_error)
        return torch.exp(-body_velocity_error**2)
    
    def _reward_robot_door_ori(self):
        # reward robot walking to the door
        
        handle_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.object_actor_handles[0], "handle") - self.env.num_bodies
        handle_pos_global = self.env.rigid_body_state_object.view(self.env.num_envs, -1, 13)[:,handle_idx,0:3].view(self.env.num_envs,3)
        robot_body_pos_global = self.env.base_pos
        
        body_handle_vec_global = handle_pos_global - robot_body_pos_global
        body_handle_vec_body = quat_rotate_inverse(self.env.base_quat, body_handle_vec_global)
        body_handle_vec_body = body_handle_vec_body / torch.norm(body_handle_vec_body, dim=1).unsqueeze(1)
        
        heading = torch.atan2(body_handle_vec_body[:, 1], body_handle_vec_body[:, 0])
        
        return torch.exp(-heading**2)
    
    def _reward_gripper_handle_pos(self):
        # reward gripper being near the handle
        gripper_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "gripperMover")
        gripper_pos_body = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,gripper_idx,0:3].view(self.env.num_envs,3)-self.env.base_pos)
        
        # print(self.env.gym.get_asset_rigid_body_names(self.env.ball_asset))

        handle_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.object_actor_handles[0], "handle") - self.env.num_bodies
        handle_pos_body = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state_object.view(self.env.num_envs, -1, 13)[:,handle_idx,0:3].view(self.env.num_envs,3)-self.env.base_pos)
        
        gripper_handle_dist = torch.norm(gripper_pos_body - handle_pos_body, dim=1)
        
        return torch.exp(-5 * gripper_handle_dist**2)
    
    def _reward_gripper_handle_height(self):
        # reward gripper being near the handle
        gripper_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "gripperMover")
        gripper_pos_body = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,gripper_idx,0:3].view(self.env.num_envs,3)-self.env.base_pos)
        
        # print(self.env.gym.get_asset_rigid_body_names(self.env.ball_asset))

        handle_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.object_actor_handles[0], "handle") - self.env.num_bodies
        handle_pos_body = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state_object.view(self.env.num_envs, -1, 13)[:,handle_idx,0:3].view(self.env.num_envs,3)-self.env.base_pos)
        
        gripper_handle_dist = gripper_pos_body[:, 2] - handle_pos_body[:, 2]
        
        return torch.exp(-gripper_handle_dist**2)
 
    def _reward_turn_handle(self):
        # reward the handle turning
        handle_angle = self.env.dof_pos[:, -1]
        # print(handle_angle)
        return torch.abs(handle_angle)
    
    def _reward_open_door(self):
        # reward the handle turning
        door_angle = self.env.dof_pos[:, -2]
        # print(door_angle)
        return torch.abs(door_angle)

    # # encourage robot velocity align vector from robot body to ball
    # # r_cv
    # def _reward_dribbling_robot_ball_vel(self):
    #     FR_shoulder_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "FR_thigh_shoulder")
    #     FR_HIP_positions = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,FR_shoulder_idx,0:3].view(self.env.num_envs,3)-self.env.base_pos)
    #     FR_HIP_velocities = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,FR_shoulder_idx,7:10].view(self.env.num_envs,3))
        
    #     # front_target = torch.tensor([[0.05, 0.0, 0.0]], device=self.env.device)
    #     # front_target = torch.tensor(self.env.cfg.rewards.front_target, device=self.env.device)
    #     # robot_velocity = self.env.base_lin_vel[:, :2]

    #     delta_dribbling_robot_ball_vel = 1.0
    #     robot_ball_vec = self.env.object_local_pos[:,0:2] - FR_HIP_positions[:,0:2]
    #     d_robot_ball=robot_ball_vec / torch.norm(robot_ball_vec, dim=-1).unsqueeze(dim=-1)
    #     ball_robot_velocity_projection = torch.norm(self.env.commands[:,:2], dim=-1) - torch.sum(d_robot_ball * FR_HIP_velocities[:,0:2], dim=-1) # set approaching speed to velocity command
    #     velocity_concatenation = torch.cat((torch.zeros(self.env.num_envs,1, device=self.env.device), ball_robot_velocity_projection.unsqueeze(dim=-1)), dim=-1)
    #     rew_dribbling_robot_ball_vel=torch.exp(-delta_dribbling_robot_ball_vel* torch.pow(torch.max(velocity_concatenation,dim=-1).values, 2) )
    #     return rew_dribbling_robot_ball_vel

    # # encourage robot near ball
    # # r_cp
    # def _reward_dribbling_robot_ball_pos(self):
    #     # body_names = self.env.gym.get_asset_rigid_body_names(self.env.robot_asset)
    #     # print(body_names)
    #     # body_names = self.env.gym.get_actor_rigid_body_names(self.env.envs[0], self.env.robot_actor_handles[0])
    #     # print(body_names)
    #     FR_shoulder_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "FR_thigh_shoulder")
    #     FR_HIP_positions = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,FR_shoulder_idx,0:3].view(self.env.num_envs,3)-self.env.base_pos)
        
    #     # front_target = torch.tensor([[0.20, 0.0, 0.0]], device=self.env.device)
    #     # front_target = torch.tensor(self.env.cfg.rewards.front_target, device=self.env.device)
    #     # print(FR_shoulder_idx)
    #     # print(FR_HIP_positions)
    #     # print(self.env.object_local_pos)

    #     delta_dribbling_robot_ball_pos = 4.0
    #     rew_dribbling_robot_ball_pos = torch.exp(-delta_dribbling_robot_ball_pos * torch.pow(torch.norm(self.env.object_local_pos - FR_HIP_positions, dim=-1), 2) )
    #     return rew_dribbling_robot_ball_pos 

    # # encourage ball vel align with unit vector between ball target and ball current position
    # # r^bv
    # def _reward_dribbling_ball_vel(self):
    #     # target velocity is command input
    #     lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.object_lin_vel[:, :2]), dim=1)
    #     rew_dribbling_ball_vel = torch.exp(-lin_vel_error / (self.env.cfg.rewards.tracking_sigma*2))
    #     return torch.exp(-lin_vel_error / (self.env.cfg.rewards.tracking_sigma*2))
        
    # def _reward_dribbling_robot_ball_yaw(self):
    #     robot_ball_vec = self.env.object_pos_world_frame[:,0:2] - self.env.base_pos[:,0:2]
    #     d_robot_ball=robot_ball_vec / torch.norm(robot_ball_vec, dim=-1).unsqueeze(dim=-1)

    #     unit_command_vel = self.env.commands[:,:2] / torch.norm(self.env.commands[:,:2], dim=-1).unsqueeze(dim=-1)
    #     robot_ball_cmd_yaw_error = torch.norm(unit_command_vel, dim=-1) - torch.sum(d_robot_ball * unit_command_vel, dim=-1)

    #     # robot ball vector align with body yaw angle
    #     roll, pitch, yaw = get_euler_xyz(self.env.base_quat)
    #     body_yaw_vec = torch.zeros(self.env.num_envs, 2, device=self.env.device)
    #     body_yaw_vec[:,0] = torch.cos(yaw)
    #     body_yaw_vec[:,1] = torch.sin(yaw)
    #     robot_ball_body_yaw_error = torch.norm(body_yaw_vec, dim=-1) - torch.sum(d_robot_ball * body_yaw_vec, dim=-1)
    #     delta_dribbling_robot_ball_cmd_yaw = 2.0
    #     rew_dribbling_robot_ball_yaw = torch.exp(-delta_dribbling_robot_ball_cmd_yaw * (robot_ball_cmd_yaw_error+robot_ball_body_yaw_error))
    #     return rew_dribbling_robot_ball_yaw
    
    # def _reward_dribbling_ball_vel_norm(self):
    #     # target velocity is command input
    #     vel_norm_diff = torch.pow(torch.norm(self.env.commands[:, :2], dim=-1) - torch.norm(self.env.object_lin_vel[:, :2], dim=-1), 2)
    #     delta_vel_norm = 2.0
    #     rew_vel_norm_tracking = torch.exp(-delta_vel_norm * vel_norm_diff)
    #     return rew_vel_norm_tracking

    # def _reward_dribbling_ball_vel_angle(self):
    #     angle_diff = torch.atan2(self.env.commands[:,1], self.env.commands[:,0]) - torch.atan2(self.env.object_lin_vel[:,1], self.env.object_lin_vel[:,0])
    #     angle_diff_in_pi = torch.pow(wrap_to_pi(angle_diff), 2)
    #     rew_vel_angle_tracking = 1.0 - angle_diff_in_pi/(torch.pi**2)
    #     return rew_vel_angle_tracking