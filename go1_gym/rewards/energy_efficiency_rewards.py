import torch
import numpy as np
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi
from .rewards import Rewards

class EnergyEfficiencyRewards(Rewards):
    def __init__(self, env):
        self.env = env
        self.actor_critic = None

    def load_env(self, env):
        self.env = env

    # ------------ reward functions----------------
    def _reward_tracking_x_vel(self):
        # Tracking of x velocity commands
        return torch.abs(self.env.commands[:,0] - self.env.base_lin_vel[:,0])

    def _reward_tracking_other_vels(self):
        # Tracking of y velocity and yaw velocity commands
        return torch.square(self.env.commands[:,1] - self.env.base_lin_vel[:,1]) + torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])

    def _reward_survival(self):
        # Survival bonus
        return self.env.commands[:,0]

    def _reward_vel(self):
        # Velocity reward
        y_vel_error = torch.square(self.env.commands[:, 1] - self.env.base_lin_vel[:, 1])
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return self.env.base_lin_vel[:,0] * torch.exp((-y_vel_error - ang_vel_error)/self.env.cfg.rewards.tracking_sigma)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        x_vel_diff = self.env.commands[:, 0] - self.env.base_lin_vel[:, 0]
        # When x_vel_diff is less than 0, then the robot is moving faster than the commanded velocity
        # Clip x_vel_diff to be above 0. It is ok for the robot to move faster than the commanded velocity
        x_vel_diff = torch.clamp(x_vel_diff, min=0)
        x_vel_error = torch.square(x_vel_diff)
        y_vel_error = torch.square(self.env.commands[:, 1] - self.env.base_lin_vel[:, 1])
        lin_vel_error = x_vel_error + y_vel_error
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma) # Shape: (num_envs,)
    
    def _reward_tracking_lin_vel_balanced(self):
        x_vel_diff = self.env.commands[:, 0] - self.env.base_lin_vel[:, 0]
        # x_vel_diff = torch.clamp(x_vel_diff, min=0)
        x_vel_error = torch.square(x_vel_diff)
        y_vel_error = torch.square(self.env.commands[:, 1] - self.env.base_lin_vel[:, 1])
        lin_vel_reward = torch.exp(-x_vel_error / self.env.cfg.rewards.tracking_sigma) + 0.05 * torch.exp(-y_vel_error / self.env.cfg.rewards.tracking_sigma)
        return lin_vel_reward

    def _reward_lin_vel_z(self):
        z_vel = self.env.base_lin_vel[:, 2]
        z_vel_reward = -torch.square(z_vel)
        return z_vel_reward

    def _reward_tracking_lin_vel_integral(self):
        # Tracking of linear velocity commands (xy axes)
        x_vel_diff_history = self.env.x_vel_diff_history
        # x_vel_diff_history = torch.clamp(x_vel_diff_history, min=0)
        x_vel_error_integral = torch.square(torch.mean(x_vel_diff_history, dim=1))
        y_vel_diff_history = self.env.y_vel_diff_history
        y_vel_error_integral = torch.square(torch.mean(y_vel_diff_history, dim=1))
        lin_vel_error = x_vel_error_integral + y_vel_error_integral
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)
    
    def _reward_tracking_ang_vel_integral(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(torch.mean(self.env.yaw_vel_diff_history, dim=1))
        return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)
    
    def _reward_energy(self):
        # Penalize the energy usage while staying in distribution of the actuator network
        joint_energies = self.env.energies # (num_envs, 12, 3)
        mech_work = joint_energies[:,:,0].sum(dim=1) + joint_energies[:,:,1].sum(dim=1)
        torque_squareds = joint_energies[:,:,2] # (num_envs, 12)

        # gear ratios
        gear_ratios = torch.tensor([1.0, 1.0, 1./1.5,
                            1.0, 1.0, 1./1.5,
                            1.0, 1.0, 1./1.5,
                            1.0, 1.0, 1./1.5,], device=self.env.device,
                            ).unsqueeze(dim=0) # knee has extra gearing
        joule_heating = torch.sum(torque_squareds * torch.square(gear_ratios), dim=1) * 0.65

        energy_estimate = mech_work + joule_heating

        torque_uncertainty = self.env.compute_torque_uncertainty()
        torque_uncertainty_reward = torch.exp(-torque_uncertainty/self.env.cfg.rewards.torque_uncertainty_sigma)
        return (energy_estimate - 200) * torque_uncertainty_reward
    
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
    
    def _reward_energy_action_smoothness_1(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - self.env.last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:, :self.env.num_actuated_dof] != 0)  # ignore first step
        return torch.sum(diff, dim=1)

    def _reward_energy_action_smoothness_2(self):
        # Penalize changes in actions
        diff = torch.square(self.env.joint_pos_target[:, :self.env.num_actuated_dof] - 2 * self.env.last_joint_pos_target[:, :self.env.num_actuated_dof] + self.env.last_last_joint_pos_target[:, :self.env.num_actuated_dof])
        diff = diff * (self.env.last_actions[:, :self.env.num_actuated_dof] != 0)  # ignore first step
        diff = diff * (self.env.last_last_actions[:, :self.env.num_actuated_dof] != 0)  # ignore second step
        return torch.sum(diff, dim=1)
    
    def _reward_energy_footswing_bonus(self):

        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.env.contact_forces[:, self.env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.env.last_contacts)
        #self.env.time_since_takeoff += self.env.dt
        #self.env.time_since_touchdown += self.env.dt
        #self.env.time_since_takeoff *= ~(self.env.last_contact_filt * ~contact_filt)
        #self.env.time_since_touchdown *= ~(~self.env.last_contact_filt * contact_filt)

        #desired_contact = self.env.desired_contact_states > 0.5

        #stance_rew = desired_contact * torch.clamp(self.env.time_since_touchdown - self.env.time_since_takeoff, -0.3, 0.3)
        #t_max = torch.max(self.env.time_since_touchdown, self.env.time_since_takeoff)
        #swing_rew = ~desired_contact * torch.clamp(t_max, max=0.2) * (t_max < 0.25)

        #rew_airTime = torch.sum(stance_rew + swing_rew, dim=1)

        self.env.last_contacts = contact[:]
        self.env.last_contact_filt = contact_filt[:]
        # k_airTime = 0.3

        first_contact = (self.env.feet_air_time > 0.) * contact_filt
        self.env.feet_air_time += self.env.dt

        rew_airTime = torch.sum((self.env.feet_air_time - 0.5) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.env.commands[:, :2], dim=1) > 0.1
        self.env.feet_air_time *= ~contact_filt

        return rew_airTime
