from .sensor import Sensor

from isaacgym.torch_utils import *
import torch
from go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift

class FootForceSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset
        self.name = "FootForceSensor"

    def get_observation(self, env_ids = None):
        feet_forces = self.env.feet_forces
        return feet_forces
    
    def get_noise_vec(self):
        return torch.zeros(12, device=self.env.device)

    def get_dim(self):
        return 12