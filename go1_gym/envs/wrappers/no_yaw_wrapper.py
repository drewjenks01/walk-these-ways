from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
import torch

class NoYawWrapper(HistoryWrapper):

    def __init__(self,env,yaw_bool):
        super().__init__(env)

        self.yaw_bool=yaw_bool

        self.env=env



    def step(self, action):

        obs,rew,done,info = super().step(action)
       
        if not self.yaw_bool:
            obs_history = torch.reshape(obs['obs_history'],(-1,self.env.num_obs))

            no_yaw_obs_history = obs_history[:,:-1]
            no_yaw_obs_history = torch.reshape(no_yaw_obs_history,(self.env.num_envs,-1))
            obs['obs_history']=no_yaw_obs_history
    
        return obs, rew, done, info



    def reset(self):
        ret = super().reset()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history[:, :] = 0
        if not self.yaw_bool:
            print(self.obs_history.shape)
            return {"obs": ret, "privileged_obs": privileged_obs, "obs_history": self.obs_history[:,:-30]}

        return {"obs": ret, "privileged_obs": privileged_obs, "obs_history": self.obs_history}