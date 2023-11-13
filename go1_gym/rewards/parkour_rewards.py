import torch
import numpy as np
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift, quat_apply
from isaacgym.torch_utils import *
from isaacgym import gymapi
from .rewards import Rewards

class ParkourRewards(Rewards):
    def __init__(self, env):
        self.env = env
        self.actor_critic = None

    def load_env(self, env):
        self.env = env

    # ------------ reward functions----------------
    def _reward_tracking_goal_vel(self):
        norm = torch.norm(self.env.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.env.target_pos_rel / (norm + 1e-5)
        cur_vel = self.env.root_states[:, 7:9]
        rew = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), self.env.commands[:, 0]) / (self.env.commands[:, 0] + 1e-5)
        return rew

    def _reward_tracking_yaw(self):
        rew = torch.exp(-torch.abs(self.env.target_yaw - self.env.yaw))
        return rew
    
    def _reward_lin_vel_z(self):
        rew = torch.square(self.env.base_lin_vel[:, 2])
        rew[self.env.env_class != 17] *= 0.5
        return rew
    
    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)
     
    def _reward_orientation(self):
        rew = torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)
        rew[self.env.env_class != 17] = 0.
        return rew

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel) / self.env.dt), dim=1)

    def _reward_collision(self):
        return torch.sum(1.*(torch.norm(self.env.contact_forces[:, self.env.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_action_rate(self):
        return torch.norm(self.env.last_actions - self.env.actions, dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.env.torques - self.env.last_torques), dim=1)
    
    def _reward_torques(self):
        return torch.sum(torch.square(self.env.torques), dim=1)

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.env.dof_pos[:, [0, 3, 6, 9]] - self.env.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)

    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos), dim=1)
        return dof_error
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.env.contact_forces[:, self.env.feet_indices, :2], dim=2) >\
             4 *torch.abs(self.env.contact_forces[:, self.env.feet_indices, 2]), dim=1)
        return rew.float()

    def _reward_feet_edge(self):
        feet_pos_xy = ((self.env.rigid_body_state[:, self.env.feet_indices, :2] + self.env.cfg.terrain.border_size) / self.env.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
        feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.env.x_edge_mask.shape[0]-1)
        feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.env.x_edge_mask.shape[1]-1)
        feet_at_edge = self.env.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
    
        self.feet_at_edge = self.env.contact_filt & feet_at_edge
        rew = (self.env.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
        return rew

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
