from .sensor import Sensor

class JointPositionSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset
        self.name = "JointPositionSensor"

    def get_observation(self, env_ids = None):
        # self.env.dof_pos[:, -1] = 0.0
        # self.env.dof_pos[:, -2] = 0.0
        # print("dof_pos: ", self.env.dof_pos)
        if self.env.cfg.commands.only_test_loco:
            # print(self.env.actuated_dof_pos)
            # print(int(self.env.actuated_dof_pos))
            self.env.dof_pos[:, (self.env.num_dofs-7):self.env.num_dofs] = self.env.default_actuated_dof_pos[:, (self.env.num_dofs-7):self.env.num_dofs]
        if self.env.cfg.commands.control_only_z1:
            self.env.dof_pos[:, :12] = self.env.default_dof_pos[:, :12] # legs obs to 0
            self.env.dof_pos[:, 18] = self.env.default_dof_pos[:, 18] # gripper obs to 0

        # print("joint pos delta: ", (self.env.dof_pos[0, :self.env.num_actuated_dof] - \
        #         self.env.default_dof_pos[0, :self.env.num_actuated_dof]))
        
        
        return (self.env.dof_pos[:, :self.env.num_actuated_dof] - \
                self.env.default_dof_pos[:, :self.env.num_actuated_dof]) * \
                    self.env.cfg.obs_scales.dof_pos
    
    def get_noise_vec(self):
        import torch
        return torch.ones(self.env.num_actuated_dof, device=self.env.device) * \
            self.env.cfg.noise_scales.dof_pos * \
            self.env.cfg.noise.noise_level * \
            self.env.cfg.obs_scales.dof_pos
    
    def get_dim(self):
        return self.env.num_actuated_dof