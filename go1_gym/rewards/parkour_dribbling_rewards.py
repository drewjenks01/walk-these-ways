import torch
import numpy as np
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift, quat_apply
from isaacgym.torch_utils import *
from isaacgym import gymapi
from .rewards import Rewards

class ParkourDribblingRewards(Rewards):
    def __init__(self, env):
        self.env = env
        self.actor_critic = None

    def load_env(self, env):
        self.env = env

    # ------------ reward functions----------------
    def _reward_tracking_goal_vel(self):
        cur_vel = self.env.base_lin_vel[:, :2]

        FR_shoulder_idx = self.env.gym.find_actor_rigid_body_handle(self.env.envs[0], self.env.robot_actor_handles[0], "FR_thigh_shoulder")
        FR_HIP_positions = quat_rotate_inverse(self.env.base_quat, self.env.rigid_body_state.view(self.env.num_envs, -1, 13)[:,FR_shoulder_idx,0:3].view(self.env.num_envs,3)-self.env.base_pos)
        robot_ball_vec = self.env.object_local_pos[:,0:2] - FR_HIP_positions[:,0:2]

        robot_velocity_projection = torch.sum(cur_vel * robot_ball_vec, dim=1)

        # rew = torch.minimum(torch.sum(1.0 * cur_vel, dim=-1), self.env.commands[:, 0]) / (self.env.commands[:, 0] + 1e-5)
        rew = torch.minimum(robot_velocity_projection, self.env.commands[:, 0]) / (self.env.commands[:, 0] + 1e-5)
        return rew

    def _reward_tracking_yaw(self):
        forward = quat_apply(self.env.base_quat, self.env.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0]) - self.env.heading_offsets
        rew = torch.exp(-torch.abs(heading))
        return rew
    
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

    # def _reward_feet_edge(self):
    #     feet_pos_xy = ((self.rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
    #     feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
    #     feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
    #     feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
    
    #     self.feet_at_edge = self.contact_filt & feet_at_edge
    #     rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
    #     return rew

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
