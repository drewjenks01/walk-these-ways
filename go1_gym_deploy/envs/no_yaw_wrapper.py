from go1_gym_deploy.envs.history_wrapper import HistoryWrapper
import torch

class NoYawWrapper(HistoryWrapper):

    def __init__(self,env):
        super().__init__(env)

        self.yaw_bool=False

        self.env=env



    def step(self, action):

        obs,rew,done,info = super().step(action)
       
        # if not self.yaw_bool:
        #     obs_history = torch.reshape(obs['obs_history'],(-1,self.env.num_obs))

        #     no_yaw_obs_history = obs_history[:,:-1]
        #     no_yaw_obs_history = torch.reshape(no_yaw_obs_history,(1,-1))
        #     obs['obs_history']=no_yaw_obs_history
    
        return obs, rew, done, info

    
    def get_post_obs(self, obs, policy):
        #print('size', obs['obs_history'].size)

        if not self.yaw_bool and obs['obs_history'].shape[1]==2130:
            obs_history = torch.reshape(obs['obs_history'],(-1,self.env.num_obs))

            no_yaw_obs_history = obs_history[:,:-1]
            no_yaw_obs_history = torch.reshape(no_yaw_obs_history,(1,-1))
            obs['obs_history']=no_yaw_obs_history
        
        elif self.yaw_bool and obs['obs_history'].shape[1]==2100:
            self.yaw_bool=False
            policy = 'walk'


        return obs, policy




    def reset(self):
        ret = super().reset()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history[:, :] = 0
        if not self.yaw_bool:
            print('Obs history shape:',self.obs_history.shape)
            return {"obs": ret, "privileged_obs": privileged_obs, "obs_history": self.obs_history[:,:-30]}

        return {"obs": ret, "privileged_obs": privileged_obs, "obs_history": self.obs_history}