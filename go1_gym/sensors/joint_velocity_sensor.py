from .sensor import Sensor

class JointVelocitySensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset
        self.name = "JointVelocitySensor"

    def get_observation(self, env_ids = None):
        if self.env.cfg.commands.only_test_loco:
            self.env.dof_vel[:, self.env.num_dofs-7:self.env.num_dofs] = 0.0

        if self.env.cfg.commands.control_only_z1:
            self.env.dof_vel[:, :12] = 0.0 # legs obs to 0
            self.env.dof_vel[:, 18] = 0.0 # gripper obs to 0

        # print("joint vel ",self.env.dof_vel[0, :self.env.num_actuated_dof])

        return self.env.dof_vel[:, :self.env.num_actuated_dof] * \
                    self.env.cfg.obs_scales.dof_vel
    
    def get_noise_vec(self):
        import torch
        return torch.ones(self.env.num_actuated_dof, device=self.env.device) * \
            self.env.cfg.noise_scales.dof_vel * \
            self.env.cfg.noise.noise_level * \
            self.env.cfg.obs_scales.dof_vel
    
    def get_dim(self):
        return self.env.num_actuated_dof