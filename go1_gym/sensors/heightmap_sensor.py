from .sensor import Sensor
import torch
from go1_gym.utils.math_utils import quat_apply_yaw

class HeightmapSensor(Sensor):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.name = "HeightmapSensor"

    def get_observation(self, env_ids = None):

        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=self.env.device)

        if self.env.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.env.num_envs, self.env.cfg.perception.num_height_points, device=self.device, requires_grad=False)
        elif self.env.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = quat_apply_yaw(self.env.base_quat[env_ids].repeat(1, self.env.num_height_points),
                                self.env.height_points[env_ids]) + (self.env.root_states[env_ids, :3]).unsqueeze(1)

        points += self.env.cfg.terrain.border_size
        points = (points / self.env.cfg.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.env.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.env.height_samples.shape[1] - 2)

        heights1 = self.env.height_samples[px, py]
        heights2 = self.env.height_samples[px + 1, py]
        heights3 = self.env.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(len(env_ids), -1) * self.env.cfg.terrain.vertical_scale