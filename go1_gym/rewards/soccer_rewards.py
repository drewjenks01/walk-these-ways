import torch
import numpy as np
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from .rewards import Rewards

class SoccerRewards(Rewards):
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    # def _reward_orientation(self):
    #     # Penalize non flat base orientation
    #     return torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)

    def _reward_tracking_goal_vel(self):
        # norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        # target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        # cur_vel = self.env.root_states[:, 7:9]
        cur_vel = self.env.base_lin_vel[:, :1]
        rew = torch.minimum(torch.sum(1.0 * cur_vel, dim=-1), self.env.commands[:, 0]) / (self.env.commands[:, 0] + 1e-5)
        return rew
    
    def _reward_tracking_goal_vel_xy(self):
        # norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        # target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        cur_world_vel = self.env.root_states[self.env.object_actor_idxs, 7:9]
        target_world_vel = self.env.commands[:, :2]
        target_vel_magnitude = torch.norm(target_world_vel, dim=-1)

        # forward = quat_apply(self.env.base_quat, self.env.forward_vec)
        # heading = torch.atan2(forward[:, 1], forward[:, 0])
        
        # robot_ball_vec = self.env.object_pos_world_frame[:,0:2] - self.env.base_pos[:,0:2]
        # desired_heading = torch.atan2(robot_ball_vec[:, 1], robot_ball_vec[:, 0])
        
        # heading_error = heading - desired_heading
        # heading_error = heading_error - 2*np.pi*(heading_error > np.pi)
        # heading_error = heading_error + 2*np.pi*(heading_error < -np.pi)

        # target_vel_magnitude = target_vel_magnitude * (1 - heading_error / np.pi)
        target_world_vel_norm = target_world_vel / (target_vel_magnitude.unsqueeze(dim=-1) + 1e-5)

        # cur_vel = self.env.base_lin_vel[:, :1]
        rew = torch.maximum(torch.minimum(torch.sum(target_world_vel_norm * cur_world_vel, dim=-1), target_vel_magnitude), -target_vel_magnitude) / (target_vel_magnitude + 1e-5)
        return rew
    
    def _reward_tracking_yaw(self):
        forward = quat_apply(self.env.base_quat, self.env.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        
        FR_shoulder_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "FR_thigh_shoulder")
        FR_HIP_positions = self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,FR_shoulder_idx,0:3].view(self.env.num_envs,3)
        robot_ball_vec = self.env.object_pos_world_frame[:,0:2] - FR_HIP_positions[:,0:2]
        robot_ball_heading = torch.atan2(robot_ball_vec[:, 1], robot_ball_vec[:, 0])
        
        heading_error = wrap_to_pi(robot_ball_heading - heading)

        # forward = quat_apply(self.env.base_quat, self.env.forward_vec)
        # heading = torch.atan2(forward[:, 1], forward[:, 0])
        # robot_ball_vec = self.env.object_pos_world_frame[:,0:2] - self.env.base_pos[:,0:2]
        # robot_ball_heading = torch.atan2(robot_ball_vec[:, 1], robot_ball_vec[:, 0])
        # ball_heading_error = wrap_to_pi(heading - robot_ball_heading)

        rew = torch.exp(-torch.abs(heading_error))# - torch.abs(ball_heading_error))
        return rew
    
    def _reward_tracking_goal_yaw(self):
        target_heading = self.env.commands[:, 2] - self.env.heading_offsets
        forward = quat_apply(self.env.base_quat, self.env.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0]) - self.env.heading_offsets
        rew = torch.exp(-torch.abs(heading))
        return rew
    
    def _reward_dribbling_robot_ball_vel(self):
        FR_shoulder_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "FR_thigh_shoulder")
        FR_HIP_positions = self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,FR_shoulder_idx,0:3].view(self.env.num_envs,3)
        robot_ball_vec = self.env.object_pos_world_frame[:,0:2] - FR_HIP_positions[:,0:2]
       
        cur_vel = self.env.root_states[self.env.robot_actor_idxs, 7:9]

        target_world_vel = self.env.commands[:, :2]
        target_vel_magnitude = torch.norm(target_world_vel, dim=-1).unsqueeze(dim=-1)
        
        target_vel_norm = robot_ball_vec / (torch.norm(robot_ball_vec, dim=-1).unsqueeze(dim=-1) + 1e-5)
        rew = torch.minimum(torch.sum(target_vel_norm * cur_vel, dim=-1), target_vel_magnitude[:, 0]) / (target_vel_magnitude[:, 0] + 1e-5)
        
        return rew
    
    # # encourage robot velocity align vector from robot body to ball
    # # r_cv
    # def _reward_dribbling_robot_ball_vel(self):
    #     FR_shoulder_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "FR_thigh_shoulder")
    #     FR_HIP_positions = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,FR_shoulder_idx,0:3].view(self.env.num_envs,3)-self.env.base_pos)
    #     FR_HIP_velocities = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,FR_shoulder_idx,7:10].view(self.env.num_envs,3))
        
    #     # front_target = torch.tensor([[0.05, 0.0, 0.0]], device=self.env.device)
    #     # front_target = torch.tensor(self.env.cfg.rewards.front_target, device=self.env.device)
    #     robot_velocity = self.env.base_lin_vel[:, :2]

    #     delta_dribbling_robot_ball_vel = 1.0
    #     robot_ball_vec = self.env.object_local_pos[:,0:2] - FR_HIP_positions[:,0:2]
    #     d_robot_ball=robot_ball_vec / torch.norm(robot_ball_vec, dim=-1).unsqueeze(dim=-1)
    #     # ball_robot_velocity_projection = torch.norm(self.env.commands[:,:2], dim=-1) - torch.sum(d_robot_ball * FR_HIP_velocities[:,0:2], dim=-1) # set approaching speed to velocity command
    #     # velocity_concatenation = torch.cat((torch.zeros(self.env.num_envs,1, device=self.env.device), ball_robot_velocity_projection.unsqueeze(dim=-1)), dim=-1)
    #     # rew_dribbling_robot_ball_vel=torch.exp(-delta_dribbling_robot_ball_vel* torch.pow(torch.max(velocity_concatenation,dim=-1).values, 2) )
        
    #     robot_velocity_projection = torch.sum(d_robot_ball * robot_velocity, dim=-1)
        
    #     rew_dribbling_robot_ball_vel = torch.clip(robot_velocity_projection, max=1.0)
        
    #     return rew_dribbling_robot_ball_vel
    
    def _reward_dribbling_robot_ball_yaw(self):
        forward = quat_apply(self.env.base_quat, self.env.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        
        # robot_ball_vec = self.env.object_pos_world_frame[:,0:2] - self.env.base_pos[:,0:2]
        # FR_shoulder_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "FR_thigh_shoulder")
        # FR_HIP_positions = self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,FR_shoulder_idx,0:3].view(self.env.num_envs,3)
        # robot_ball_vec = self.env.object_pos_world_frame[:,0:2] - FR_HIP_positions[:,0:2]
       
        desired_heading = torch.atan2(self.env.commands[:, 1], self.env.commands[:, 0])
        
        heading_error = wrap_to_pi(desired_heading - heading)
        # heading_error = heading_error - 2*np.pi*(heading_error > np.pi)
        # heading_error = heading_error + 2*np.pi*(heading_error < -np.pi)
        
        rew = torch.exp(-torch.abs(heading_error))
        return rew
    
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
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    # def _reward_feet_clearance_linear(self):
    #     swing_height_target = 0.12
    #     reference_points = self.env.foot_positions[:, :, 0:2].view(-1, 2)
    #     reference_heights = self.env.get_heights_points(reference_points).view(-1, 4)
    #     phases = 1 - torch.abs(1.0 - torch.clip((self.env.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
    #     foot_height = (self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1) - reference_heights
    #     target_height = swing_height_target * phases + 0.02 # offset for foot radius 2cm
    #     rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.env.desired_contact_states)
    #     return torch.sum(rew_foot_clearance, dim=1)

    def _reward_feet_clearance_cmd_linear(self):
        reference_points = self.env.foot_positions[:, :, 0:2].view(-1, 2)
        reference_heights = self.env.get_heights_points(reference_points).view(-1, 4)
        phases = 1 - torch.abs(1.0 - torch.clip((self.env.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.env.foot_positions[:, :, 2]).view(self.env.num_envs, -1) - reference_heights
        target_height = self.env.commands[:, 9].unsqueeze(1) * phases + 0.02 # offset for foot radius 2cm
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.env.desired_contact_states)
        return torch.sum(rew_foot_clearance, dim=1)

    def _reward_dof_pos(self):
        # Penalize dof positions
        # k_q = -0.75
        return torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)

    def _reward_action_smoothness_1(self):
        # Penalize changes in actions
        # k_s1 =-2.5
        diff = torch.square(self.env.joint_pos_target - self.env.last_joint_pos_target)
        diff = diff * (self.env.last_actions[:,:12] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_action_smoothness_2(self):
        # Penalize changes in actions
        # k_s2 = -1.2
        diff = torch.square(self.env.joint_pos_target - 2 * self.env.last_joint_pos_target + self.env.last_last_joint_pos_target)
        diff = diff * (self.env.last_actions[:,:12] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:,:12] != 0)  # ignore second step
        return torch.sum(diff, dim=1)

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
    
    # # encourage robot velocity align vector from robot body to ball
    # # r_cv
    # def _reward_dribbling_robot_ball_vel(self):
    #     FR_shoulder_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "FR_thigh_shoulder")
    #     FR_HIP_positions = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,FR_shoulder_idx,0:3].view(self.env.num_envs,3)-self.env.base_pos)
    #     FR_HIP_velocities = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,FR_shoulder_idx,7:10].view(self.env.num_envs,3))
        
    #     # front_target = torch.tensor([[0.05, 0.0, 0.0]], device=self.env.device)
    #     # front_target = torch.tensor(self.env.cfg.rewards.front_target, device=self.env.device)
    #     robot_velocity = self.env.base_lin_vel[:, :2]

    #     delta_dribbling_robot_ball_vel = 1.0
    #     robot_ball_vec = self.env.object_local_pos[:,0:2] - FR_HIP_positions[:,0:2]
    #     d_robot_ball=robot_ball_vec / torch.norm(robot_ball_vec, dim=-1).unsqueeze(dim=-1)
    #     # ball_robot_velocity_projection = torch.norm(self.env.commands[:,:2], dim=-1) - torch.sum(d_robot_ball * FR_HIP_velocities[:,0:2], dim=-1) # set approaching speed to velocity command
    #     # velocity_concatenation = torch.cat((torch.zeros(self.env.num_envs,1, device=self.env.device), ball_robot_velocity_projection.unsqueeze(dim=-1)), dim=-1)
    #     # rew_dribbling_robot_ball_vel=torch.exp(-delta_dribbling_robot_ball_vel* torch.pow(torch.max(velocity_concatenation,dim=-1).values, 2) )
        
    #     robot_velocity_projection = torch.sum(d_robot_ball * robot_velocity, dim=-1)
        
    #     rew_dribbling_robot_ball_vel = torch.clip(robot_velocity_projection, max=1.0)
        
    #     return rew_dribbling_robot_ball_vel

    # encourage robot near ball
    # r_cp
    def _reward_dribbling_robot_ball_pos(self):
        # body_names = self.env.gym.get_asset_rigid_body_names(self.env.robot_asset)
        # print(body_names)
        # body_names = self.env.gym.get_actor_rigid_body_names(self.env.envs[0], self.env.robot_actor_handles[0])
        # print(body_names)
        FR_shoulder_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "FR_thigh_shoulder")
        FR_HIP_positions = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,FR_shoulder_idx,0:3].view(self.env.num_envs,3)-self.env.base_pos)
        
        # front_target = torch.tensor([[0.20, 0.0, 0.0]], device=self.env.device)
        # front_target = torch.tensor(self.env.cfg.rewards.front_target, device=self.env.device)
        # print(FR_shoulder_idx)
        # print(FR_HIP_positions)
        # print(self.env.object_local_pos)

        delta_dribbling_robot_ball_pos = 4.0
        rew_dribbling_robot_ball_pos = torch.exp(-delta_dribbling_robot_ball_pos * torch.pow(torch.norm(self.env.object_local_pos - FR_HIP_positions, dim=-1), 2) )
        # print("rew_dribbling_robot_ball_pos", rew_dribbling_robot_ball_pos, " distance norm: ", torch.norm(self.env.object_local_pos - FR_HIP_positions, dim=-1), " hip position: ", FR_HIP_positions, " ball position: ", self.env.object_local_pos)
        return rew_dribbling_robot_ball_pos 

    # encourage ball vel align with unit vector between ball target and ball current position
    # r^bv
    def _reward_dribbling_ball_vel(self):
        # target velocity is command input
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.object_lin_vel[:, :2]), dim=1)
        # rew_dribbling_ball_vel = torch.exp(-lin_vel_error / (self.env.cfg.rewards.tracking_sigma*2))
        return torch.exp(-lin_vel_error / (self.env.cfg.rewards.tracking_sigma*2))
        
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
    
    def _reward_dribbling_ball_vel_norm(self):
        # target velocity is command input
        vel_norm_diff = torch.pow(torch.norm(self.env.commands[:, :2], dim=-1) - torch.norm(self.env.object_lin_vel[:, :2], dim=-1), 2)
        delta_vel_norm = 2.0
        rew_vel_norm_tracking = torch.exp(-delta_vel_norm * vel_norm_diff)
        # print("vel_norm_diff", vel_norm_diff, " rew_vel_norm_tracking", rew_vel_norm_tracking, " commands", self.env.commands[:, :2], " object_lin_vel", self.env.object_lin_vel[:, :2],
        #       " norm commands", torch.norm(self.env.commands[:, :2], dim=-1), " norm object_lin_vel", torch.norm(self.env.object_lin_vel[:, :2], dim=-1))
        return rew_vel_norm_tracking

    # def _reward_dribbling_ball_vel_angle(self):
    #     angle_diff = torch.atan2(self.env.commands[:,1], self.env.commands[:,0]) - torch.atan2(self.env.object_lin_vel[:,1], self.env.object_lin_vel[:,0])
    #     angle_diff_in_pi = torch.pow(wrap_to_pi(angle_diff), 2)
    #     rew_vel_angle_tracking = torch.exp(-5.0*angle_diff_in_pi/(torch.pi**2))
    #     # print("angle_diff", angle_diff, " angle_diff_in_pi: ", angle_diff_in_pi, " rew_vel_angle_tracking", rew_vel_angle_tracking, " commands", self.env.commands[:, :2], " object_lin_vel", self.env.object_lin_vel[:, :2])
    #     return rew_vel_angle_tracking

    def _reward_dribbling_ball_vel_angle(self):
        angle_diff = torch.atan2(self.env.commands[:,1], self.env.commands[:,0]) - torch.atan2(self.env.object_lin_vel[:,1], self.env.object_lin_vel[:,0])
        angle_diff_in_pi = torch.pow(wrap_to_pi(angle_diff), 2)
        # print(angle_diff_in_pi)
        rew_vel_angle_tracking = 1.0 - angle_diff_in_pi/(torch.pi**2)
        # print("rew_vel_angle_tracking: ", rew_vel_angle_tracking, " angle_diff_in_pi: ", angle_diff_in_pi)
        # print("rew_dribbling_ball_vel: ",rew_dribbling_ball_vel, " command: ", self.env.commands[:, :2], " ball velocity: ", self.env.ball_lin_vel[:, :2])
        return rew_vel_angle_tracking
    
    def _reward_energy_analytic(self):
        # Penalize the energy usage while staying in distribution of the actuator network
        torques = self.env.torques # (num_envs, 12)
        joint_vels = self.env.dof_vel # (num_envs, 12)

        gear_ratios = torch.tensor([1.0, 1.0, 1./1.5,
                            1.0, 1.0, 1./1.5,
                            1.0, 1.0, 1./1.5,
                            1.0, 1.0, 1./1.5,], device=self.env.device,
                            ) # knee has extra gearing

        power_joule = torch.sum((torques * gear_ratios)**2 * 0.7, dim=1)
        power_mechanical = torch.sum(torch.clip(torques, -3, 10000) * joint_vels, dim=1)
        power_battery = 42.0

        return power_joule + power_mechanical + power_battery - 200
    
    def _reward_trot_symmetry(self):
        # Get the contacts
        feet_forces = self.env.contact_forces[:, self.env.feet_indices, 2]
        feet_contacts = feet_forces > 1.
        
        # # penalize if non-diagonal feet contact together
        # num_ill_feet_contacts = torch.logical_and(feet_contacts[:, 0], feet_contacts[:, 1]).float() + \
        #                         torch.logical_and(feet_contacts[:, 0], feet_contacts[:, 2]).float() + \
        #                         torch.logical_and(feet_contacts[:, 1], feet_contacts[:, 3]).float() + \
        #                         torch.logical_and(feet_contacts[:, 2], feet_contacts[:, 3]).float() + \
        #                         torch.logical_and(~feet_contacts[:, 0], ~feet_contacts[:, 1]).float() + \
        #                         torch.logical_and(~feet_contacts[:, 0], ~feet_contacts[:, 2]).float() + \
        #                         torch.logical_and(~feet_contacts[:, 1], ~feet_contacts[:, 3]).float() + \
        #                         torch.logical_and(~feet_contacts[:, 2], ~feet_contacts[:, 3]).float()

        # return -1. * num_ill_feet_contacts
                                
        # # give a bonus for the desired contact patterns
        # phase_1_contacts = torch.logical_and(
        #     torch.logical_and(feet_contacts[:, 0], feet_contacts[:, 3]),
        #     torch.logical_and(~feet_contacts[:, 1], ~feet_contacts[:, 2]),
        # ).float()
        # phase_2_contacts = torch.logical_and(
        #     torch.logical_and(feet_contacts[:, 1], feet_contacts[:, 2]),
        #     torch.logical_and(~feet_contacts[:, 0], ~feet_contacts[:, 3]),
        # ).float()
        
        # num_good_feet_contacts = -0.8 + phase_1_contacts + phase_2_contacts
                                    
        # return num_good_feet_contacts
        
        
        phase_1_force = feet_forces[:, 0] + feet_forces[:, 3]
        phase_2_force = feet_forces[:, 1] + feet_forces[:, 2]
        
        phase_1_envs = phase_1_force > phase_2_force 
        phase_2_envs = ~phase_1_envs
        
        # min_switch_time = 10 # timesteps
        
        # first_contact = (self.env.feet_air_time > 0.) * feet_contacts
        # self.env.feet_air_time += self.env.dt
        
        # rew_airTime = torch.sum((self.env.feet_air_time - 0.5) * first_contact, dim=1)
        # self.env.feet_air_time *= ~feet_contacts
        
        corrected_forces = feet_forces.clone()
        corrected_forces[phase_1_envs, 0] = 0.
        corrected_forces[phase_1_envs, 3] = 0.
        corrected_forces[phase_2_envs, 1] = 0.
        corrected_forces[phase_2_envs, 2] = 0.
        
        # penalise the contact force on the off feet
        force_penalty = torch.sum(corrected_forces, dim=1)
        
        return 10.0 - 0.1 * force_penalty
    
    def _reward_feet_air_time(self):

        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.env.last_contacts)
        
        self.env.last_contacts = contact[:]
        self.env.last_contact_filt = contact_filt[:]
        # k_airTime = 0.3

        first_contact = (self.env.feet_air_time > 0.) * contact_filt
        self.env.feet_air_time += self.env.dt

        rew_airTime = torch.sum((self.env.feet_air_time - 0.5) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.env.commands[:, :2], dim=1) > 0.1
        self.env.feet_air_time *= ~contact_filt

        return rew_airTime

    def _reward_lin_vel_z(self):
        rew = torch.square(self.env.base_lin_vel[:, 2])
        # rew[self.env_class != 17] *= 0.5
        rew *= 0.5
        return rew
    
    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)
     
    def _reward_orientation(self):
        rew = torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)
        # rew[self.env_class != 17] = 0.
        rew[:] = 0.
        return rew
        
    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)
        return dof_error
    
    def _reward_action_rate(self):
        return torch.norm(self.env.last_actions - self.env.actions, dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.env.torques - self.env.last_torques), dim=1)
    
    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel) / self.env.dt), dim=1)
