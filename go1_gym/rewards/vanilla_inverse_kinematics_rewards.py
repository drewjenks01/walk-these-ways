import torch
import numpy as np
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *

TRANSFORM_BASE_ARM_X = 0.2
TRANSFORM_BASE_ARM_Z = 0.1585
DEFAULT_BASE_HEIGHT = 0.78

INDEX_EE_POS_RADIUS_CMD = 15
INDEX_EE_POS_PITCH_CMD = 16
INDEX_EE_POS_YAW_CMD = 17
INDEX_EE_TIMING_CMD = 18
INDEX_EE_ORI_ROLL_CMD = 19
INDEX_EE_ORI_PITCH_CMD = 20
INDEX_EE_ORI_YAW_CMD = 21

class VanillaInverseKinematicsRewards:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env
    
    def _reward_manip_pos_tracking(self):
        '''
        Reward for manipulation tracking (EE positon) from Deepak's paper 
        '''
        # Commands in spherical coordinates in the arm base frame 
        radius_cmd = self.env.commands[:, INDEX_EE_POS_RADIUS_CMD].view(self.env.num_envs, 1) 
        pitch_cmd = self.env.commands[:, INDEX_EE_POS_PITCH_CMD].view(self.env.num_envs, 1) 
        yaw_cmd = self.env.commands[:, INDEX_EE_POS_YAW_CMD].view(self.env.num_envs, 1) 

        # Spherical to cartesian coordinates in the arm base frame 
        x_cmd_arm = radius_cmd*torch.cos(pitch_cmd)*torch.cos(yaw_cmd)
        y_cmd_arm = radius_cmd*torch.cos(pitch_cmd)*torch.sin(yaw_cmd)
        z_cmd_arm = - radius_cmd*torch.sin(pitch_cmd)

        # Cartesian coordinates in the base frame
        x_cmd_base = x_cmd_arm.add_(TRANSFORM_BASE_ARM_X)
        y_cmd_base = y_cmd_arm
        z_cmd_base = z_cmd_arm.add_(TRANSFORM_BASE_ARM_Z)
        ee_position_cmd_base = torch.cat((x_cmd_base, y_cmd_base, z_cmd_base), dim=1)

        # Commands in world frame
        base_quat_world = self.env.base_quat.view(self.env.num_envs,4)
        base_rpy_world = torch.stack(get_euler_xyz(base_quat_world), dim=1)
        # Make the commands roll and pitch independent 
        base_rpy_world[:, 0] = 0.0
        base_rpy_world[:, 1] = 0.0
        base_quat_world_indep = quat_from_euler_xyz(base_rpy_world[:, 0], base_rpy_world[:, 1], base_rpy_world[:, 2]).view(self.env.num_envs,4)

        # Make the commands independent from base height 
        x_base_pos_world = self.env.base_pos[:, 0].view(self.env.num_envs, 1) 
        y_base_pos_world = self.env.base_pos[:, 1].view(self.env.num_envs, 1) 
        z_base_pos_world = torch.ones_like(self.env.base_pos[:, 2].view(self.env.num_envs, 1))*DEFAULT_BASE_HEIGHT
        base_position_world = torch.cat((x_base_pos_world, y_base_pos_world, z_base_pos_world), dim=1)

        # Command in cartesian coordinates in world frame 
        ee_position_cmd_world = quat_rotate_inverse(quat_conjugate(base_quat_world_indep), ee_position_cmd_base) + base_position_world

        # Get current ee position in world frame 
        ee_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "gripperStator")
        ee_pos_world = self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,ee_idx,0:3].view(self.env.num_envs,3)

        ee_position_error = torch.sum(torch.abs(ee_position_cmd_world - ee_pos_world), dim=1)

        ee_position_coeff = 2.0
        # print("eeposiitno error: ",torch.exp(-ee_position_coeff*ee_position_error)) 
        return torch.exp(-ee_position_coeff*ee_position_error) 

    # def _reward_tracking_lin_vel(self):
    #     # Tracking of linear velocity commands (xy axes)
    #     lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
    #     return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)

    # def _reward_tracking_ang_vel(self):
    #     # Tracking of angular velocity commands (yaw)
    #     ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
    #     return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)
        
    # def _reward_manip_energy(self):
    #     '''
    #     Energy of the arm (i.e for joints: 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')
    #     - 'jointGripper' is not included for now 
        
    #     '''
    #     arm_joints_torques = self.env.torques[:, :] 
    #     arm_joints_vel = self.env.dof_vel[:, :]

    #     return torch.sum(torch.abs(arm_joints_torques*arm_joints_vel), dim=1)

    # def _reward_loco_energy(self):
    #     '''
    #     Energy of the legs (12 leg joints)
    #     '''
    #     leg_joints_torques = self.env.torques[:, :12] 
    #     leg_joints_vel = self.env.dof_vel[:, :12]

    #     return torch.sum(torch.square(leg_joints_torques*leg_joints_vel), dim=1)

    # def _reward_tracking_lin_vel_x(self):
    #     '''
    #     Track v_x
    #     '''
    #     lin_vel_error = torch.abs(self.env.commands[:, 0] - self.env.base_lin_vel[:, 0])
    #     return lin_vel_error 

    # def _reward_tracking_ang_vel_yaw(self):
    #     '''
    #     Track w_yaw
    #     '''
    #     ang_vel_error = torch.abs(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
    #     return torch.exp(-ang_vel_error)

    # def _reward_alive(self):
    #     return torch.ones(self.env.num_envs, device=self.env.device)

    # def _reward_orientation(self):
    #     # Penalize non flat base orientation
    #     return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)
    
    # def _reward_torques(self):
    #     # Penalize torques
    #     return torch.sum(torch.square(self.env.torques[:,:12]), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        # k_qd = -6e-4
        return torch.sum(torch.square(self.env.dof_vel[:,:]), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.env.last_dof_vel[:,:] - self.env.dof_vel[:,:]) / self.env.dt), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        # print('self.env.penalised_contact_indices: ', self.env.penalised_contact_indices)
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)


    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.env.last_actions[:,:] - self.env.actions[:,:]), dim=1)

    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        # k_s1 =-2.5
        diff = torch.square(self.env.joint_pos_target[:, :] - self.env.last_joint_pos_target[:, :])
        diff = diff * (self.env.last_actions[:,:] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        # k_s2 = -1.2
        diff = torch.square(self.env.joint_pos_target[:, :] - 2 * self.env.last_joint_pos_target[:, :] + self.env.last_last_joint_pos_target[:, :])
        diff = diff * (self.env.last_actions[:, :] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, :] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

    # def _reward_tracking_contacts_shaped_force(self):
    #     foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
    #     desired_contact = self.env.desired_contact_states

    #     reward = 0
    #     for i in range(4):
    #         reward += - (1 - desired_contact[:, i]) * (
    #                     1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
    #     return reward / 4

    # def _reward_tracking_contacts_shaped_vel(self):
    #     foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
    #     desired_contact = self.env.desired_contact_states
    #     reward = 0
    #     for i in range(4):
    #         reward += - (desired_contact[:, i] * (
    #                     1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
    #     return reward / 4

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.env.dof_pos[:, :] - self.env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos[:, :] - self.env.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    # def _reward_dof_pos(self):
    #     # Penalize dof positions
    #     # k_q = -0.75
    #     return torch.sum(torch.square(self.env.dof_pos[:, :12] - self.env.default_dof_pos[:, :12]), dim=1)

    # def _reward_end_effector_position_tracking(self):
    #     # Tracking the target end-effector position
    #     gripper_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "gripperMover")
    #     gripper_pos_body = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,gripper_idx,0:3].view(self.env.num_envs,3)-self.env.base_pos)

    #     gripper_target_dist = torch.sum(torch.square(gripper_pos_body - self.env.commands[:, 15:18]), dim=1)

    #     return torch.exp(-gripper_target_dist)

    # def _reward_body_height_tracking(self):
    #     # Range: 0 - 0.5 -> exp(-range): 1 - 0.6
    #     reference_height = 0.5
    #     tracking_error = self.env.base_pos[:, 2] - reference_height
    #     return torch.exp(-torch.square(tracking_error))
    
    # def _reward_end_effector_pos_x_tracking(self):
    #     # Range: 0 - 1 -> exp(-range): 1 - 0.36
    #     # Tracking the target end-effector position x
    #     gripper_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "gripperMover")
    #     gripper_pos_body = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,gripper_idx,0:3].view(self.env.num_envs,3)-self.env.base_pos)

    #     gripper_target_dist = torch.square(gripper_pos_body[:, 0] - self.env.commands[:, 15])

    #     return torch.exp(-gripper_target_dist)
    
    # def _reward_end_effector_pos_y_tracking(self):
    #     # Range: 0 - 1 -> exp(-range): 1 - 0.36
    #     # Tracking the target end-effector position y
    #     gripper_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "gripperMover")
    #     gripper_pos_body = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,gripper_idx,0:3].view(self.env.num_envs,3)-self.env.base_pos)

    #     gripper_target_dist = torch.square(gripper_pos_body[:, 1] - self.env.commands[:, 16])

    #     return torch.exp(-gripper_target_dist)    

    # def _reward_end_effector_pos_z_tracking(self):
    #     # Range: 0 - 1 -> exp(-range): 1 - 0.36
    #     # Tracking the target end-effector position z
    #     gripper_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "gripperMover")
    #     gripper_pos_body = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,gripper_idx,0:3].view(self.env.num_envs,3)-self.env.base_pos)

    #     gripper_target_dist = torch.square(gripper_pos_body[:, 2] - self.env.commands[:, 17])

    #     return torch.exp(-gripper_target_dist)        

    # def _reward_end_effector_ori_roll_tracking(self):
    #     # Range: 0 - pi -> exp(-range): 1 - 0.04
    #     gripper_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "gripperMover")

    #     # Gripper orientation in body frame 
    #     gripper_quat_world = self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,gripper_idx,3:7].view(self.env.num_envs,4)
    #     gripper_quat_body = quat_mul(quat_conjugate(self.env.base_quat.view(self.env.num_envs,4)), gripper_quat_world)
    #     gripper_rpy_body = torch.stack(get_euler_xyz(gripper_quat_body), dim=1) 
        
    #     # current_roll in 0 - 2pi
    #     # target_roll in -pi - pi

    #     # If absolute difference is greater than π, subtract from 2π
    #     angle_diff = abs(gripper_rpy_body[:, 0] - self.env.commands[:, 18])
    #     mask = angle_diff > torch.pi
    #     angle_diff[mask] = 2*torch.pi - angle_diff[mask]


    #     return torch.exp(-torch.square(angle_diff)) 

    # def _reward_end_effector_ori_pitch_tracking(self):
    #     # Range: 0 - pi -> exp(-range): 1 - 0.04
    #     gripper_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "gripperMover")

    #     # Gripper orientation in body frame 
    #     gripper_quat_world = self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,gripper_idx,3:7].view(self.env.num_envs,4)
    #     gripper_quat_body = quat_mul(quat_conjugate(self.env.base_quat.view(self.env.num_envs,4)), gripper_quat_world)
    #     gripper_rpy_body = torch.stack(get_euler_xyz(gripper_quat_body), dim=1)

    #     # If absolute difference is greater than π, subtract from 2π
    #     angle_diff = abs(gripper_rpy_body[:, 1] - self.env.commands[:, 19])
    #     mask = angle_diff > torch.pi
    #     angle_diff[mask] = 2*torch.pi - angle_diff[mask]

    #     return torch.exp(-torch.square(angle_diff)) 
    
    # def _reward_end_effector_ori_yaw_tracking(self):
    #     # Range: 0 - pi -> exp(-range): 1 - 0.04
    #     gripper_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "gripperMover")

    #     # Gripper orientation in body frame 
    #     gripper_quat_world = self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,gripper_idx,3:7].view(self.env.num_envs,4)
    #     gripper_quat_body = quat_mul(quat_conjugate(self.env.base_quat.view(self.env.num_envs,4)), gripper_quat_world)
    #     gripper_rpy_body = torch.stack(get_euler_xyz(gripper_quat_body), dim=1)

    #     # If absolute difference is greater than π, subtract from 2π
    #     angle_diff = abs(gripper_rpy_body[:, 2] - self.env.commands[:, 20])
    #     mask = angle_diff > torch.pi
    #     angle_diff[mask] = 2*torch.pi - angle_diff[mask]

    #     return torch.exp(-torch.square(angle_diff)) 
    

    # def _reward_end_effector_orientation_tracking(self):
    #     # Tracking the target end-effector orientation 
    #     gripper_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "gripperMover")
    #     gripper_quat_world = self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,gripper_idx,3:7].view(self.env.num_envs,4)

    #     gripper_quat_body = quat_mul(quat_conjugate(self.env.base_quat), gripper_quat_world)

    #     target_end_effector_roll_body = self.env.commands[:, 18]
    #     target_end_effector_pitch_body = self.env.commands[:, 19]
    #     target_end_effector_yaw_body = self.env.commands[:, 20]

    #     target_end_effector_quat_body = quat_from_euler_xyz(target_end_effector_roll_body, target_end_effector_pitch_body, target_end_effector_yaw_body)

    #     # Normalize the quaternions
    #     gripper_quat_body = quat_unit(gripper_quat_body)
    #     target_end_effector_quat_body = quat_unit(target_end_effector_quat_body)

    #     # Compute the error 
    #     quat_error = quat_mul(target_end_effector_quat_body, quat_conjugate(gripper_quat_body))

    #     #scalar_error = torch.sum(quat_error[:, 0:3] * torch.sign(quat_error[:, 3]).unsqueeze(-1))
    #     #scalar_error = torch.square(quat_error, dim=1)

    #     return torch.exp(-torch.norm(quat_error, dim=-1))

