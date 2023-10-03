class Sensor:
    def __init__(self, env):
        self.env = env
        self.name = "Sensor"

    def get_observation(self):
        raise NotImplementedError

    def get_noise_vec(self):
        raise NotImplementedError
    
    def get_dim(self):
        raise NotImplementedError