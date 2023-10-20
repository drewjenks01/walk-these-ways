import torch
import numpy as np
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *
from isaacgym import gymapi
from .rewards import Rewards

class BCEstimationRewards(Rewards):
    def __init__(self, env):
        self.env = env
        self.actor_critic = None
        self.teacher_actor_critic = None

    def load_env(self, env):
        self.env = env

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.env.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.env.cfg.rewards.tracking_sigma_yaw)

    def _reward_bc(self):
        if self.teacher_actor_critic is None:
            print("Warning: BC reward is active but no teacher was loaded to the env!")
            return torch.zeros(self.env.num_envs).to(self.env.device)
        if self.actor_critic is None:
            print("Warning: BC reward is active but no adaptation module was loaded to the env!")
            return torch.zeros(self.env.num_envs).to(self.env.device)
        
        obs = self.env.obs_history
        teacher_actions = self.teacher_actor_critic.get_action(obs)
        student_actions = self.actor_critic.get_action(obs)

        return -torch.norm(teacher_actions - student_actions, dim=1)

    def _reward_estimation_bonus(self):
        if self.actor_critic is None:
            print("Warning: estimation bonus reward is active but no adaptation module was loaded to the env!")
            return torch.zeros(self.env.num_envs).to(self.env.device)
        
        with torch.no_grad():
            obs = self.env.obs_history
            target = self.env.privileged_obs_buf
            estimation_errors = self.actor_critic.estimation_module.compute_losses(target, obs, reduce=False)

            estimation_errors = torch.cat([estimation_errors[idx].reshape(self.env.num_envs, -1) * self.env.cfg.rewards.estimation_bonus_weights[idx] for idx in range(len(estimation_errors))], dim=1)
            
            estimation_reward = torch.sum(estimation_errors, dim=1).to(self.env.device)

        return estimation_reward