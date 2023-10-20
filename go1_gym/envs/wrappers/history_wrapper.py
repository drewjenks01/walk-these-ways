import isaacgym
assert isaacgym
import torch
import gym

class HistoryWrapper(gym.Wrapper):
    def __init__(self, env, args):
        super().__init__(env)
        self.env = env
        self.exp = args.exp

        self.obs_history_length = self.env.cfg.env.num_observation_history
        self.history_frame_skip = self.env.cfg.env.history_frame_skip

        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history_buf = torch.zeros(self.env.num_envs, self.obs_history_length * self.history_frame_skip, self.num_obs, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        self.obs_history = torch.zeros(self.env.num_envs, self.num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        self.num_privileged_obs = self.num_privileged_obs

        self.reward_container.load_env(self)
        
    def step(self, action):
        # privileged information and observation history are stored in info
        obs, rew, done, info = self.env.step(action)
        privileged_obs = info["privileged_obs"]

        self.obs_history_buf = torch.cat((self.obs_history_buf[:, 1:, :], obs.unsqueeze(1)), dim=1)
        self.obs_history = self.obs_history_buf[:, self.history_frame_skip-1::self.history_frame_skip, :].reshape(self.env.num_envs, -1)
        assert self.obs_history[:, -self.num_obs:].allclose(obs[:, :]), "obs_history does not end with obs"

        rew_enrg = self.env.energy_rew
        bsz = self.env.num_envs // 2
            
        if 'enrg' in self.exp:
            if 'eipo' in self.exp:
                rew_eipo_ext = rew[:bsz]
                rew_ext = rew[bsz:]
                rew_int = rew_enrg[:bsz]
                rew_ext_int = rew_enrg[bsz:]
                rew = {'eipo_ext': rew_eipo_ext, 'ext': rew_ext, 'int': rew_int, 'ext_int': rew_ext_int}
            elif 'const' in self.exp:
                if 'enrg_orig' in self.exp or 'enrg_trkv' in self.exp:
                    rew_ext = rew_enrg
                    rew_int = rew
                else:
                    rew_ext = rew
                    rew_int = rew_enrg
                rew = {'ext': rew_ext, 'int': rew_int}
            else:
                rew_ext = rew_enrg
                rew = {'ext': rew_ext}
        else:
            if self.exp == 'eipo_task_axlr':
                rew_tsk_all = rew['task'][:]
                rew_aux_all = rew['axlr'][:]
                rew_eipo_ext = rew_tsk_all[:bsz]
                rew_int = rew_aux_all[:bsz]
                rew_ext = rew_tsk_all[bsz:]
                rew_ext_int = rew_aux_all[bsz:]
                rew = {'eipo_ext': rew_eipo_ext, 'ext': rew_ext, 'int': rew_int, 'ext_int': rew_ext_int}
            else:
                rew = rew + rew_enrg
                rew = {'ext': rew}
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}, rew, done, info

    def get_observations(self):
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        self.obs_history_buf = torch.cat((self.obs_history_buf[:, 1:, :], obs.unsqueeze(1)), dim=1)
        self.obs_history = self.obs_history_buf[:, self.history_frame_skip-1::self.history_frame_skip, :].reshape(self.env.num_envs, -1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}

    def reset_idx(self, env_ids):  # it might be a problem that this isn't getting called!!
        # print("RESETIDX")
        ret = super().reset_idx(env_ids)
        self.obs_history_buf[env_ids, :, :] = 0
        self.obs_history[env_ids, :] = 0
        return ret

    def reset(self):
        ret = super().reset()
        privileged_obs = self.env.get_privileged_observations()
        with torch.inference_mode():
            self.obs_history[:, :] = 0
        return {"obs": ret, "privileged_obs": privileged_obs, "obs_history": self.obs_history}


if __name__ == "__main__":
    from tqdm import trange
    import matplotlib.pyplot as plt

    import ml_logger as logger

    from go1_gym_learn.ppo import Runner
    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from go1_gym_learn.ppo.actor_critic import AC_Args

    from go1_gym.envs.base.legged_robot_config import Cfg
    from go1_gym.envs.mini_cheetah.mini_cheetah_config import config_mini_cheetah
    config_mini_cheetah(Cfg)

    test_env = gym.make("VelocityTrackingEasyEnv-v0", cfg=Cfg)
    env = HistoryWrapper(test_env)

    env.reset()
    action = torch.zeros(test_env.num_envs, 12)
    for i in trange(3):
        obs, rew, done, info = env.step(action)
        print(obs.keys())
        print(f"obs: {obs['obs']}")
        print(f"privileged obs: {obs['privileged_obs']}")
        print(f"obs_history: {obs['obs_history']}")

        img = env.render('rgb_array')
        plt.imshow(img)
        plt.show()