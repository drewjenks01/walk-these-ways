from .sensor import Sensor

class ActionSensor(Sensor):
    def __init__(self, env, attached_robot_asset=None, delay=0):
        super().__init__(env)
        self.env = env
        self.attached_robot_asset = attached_robot_asset
        self.delay = delay
        self.name = "ActionSensor"

    def get_observation(self, env_ids = None):
        if self.delay == 0:
            action_to_return = self.env.actions
            if self.env.cfg.commands.only_test_loco:
                action_to_return[:, (self.env.num_dofs-7):self.env.num_dofs] = self.env.default_actuated_dof_pos[:, (self.env.num_dofs-7):self.env.num_dofs]
            return action_to_return
        elif self.delay == 1:
            return self.env.last_actions
        else:
            raise NotImplementedError("Action delay of {} not implemented".format(self.delay))
    
    def get_noise_vec(self):
        import torch
        return torch.zeros(self.env.num_actions, device=self.env.device)
    
    def get_dim(self):
        return self.env.num_actions