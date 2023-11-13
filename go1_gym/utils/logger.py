import torch
import numpy as np

class Logger:
    def __init__(self, env):
        self.env = env
        self.is_eipo = False

    def populate_log(self, env_ids):

        extras = {}

        # fill extras
        if len(env_ids) > 0:
            extras["train/episode"] = {}
            for key in self.env.episode_sums.keys():
                if self.is_eipo:
                    mixed_env_ids = []
                    ext_env_ids = []
                    for env_id in env_ids:
                        if env_id < self.env.num_envs // 2:
                            mixed_env_ids.append(env_id)
                        else:
                            ext_env_ids.append(env_id)
                    mixed_env_ids = torch.LongTensor(mixed_env_ids).to(self.env.episode_sums[key].device)
                    ext_env_ids = torch.LongTensor(ext_env_ids).to(self.env.episode_sums[key].device)
                    extras["train/episode"]['rew_mixed_' + key] = \
                        self.env.episode_sums[key][mixed_env_ids].tolist()
                    extras["train/episode"]['rew_ext_' + key] = \
                        self.env.episode_sums[key][ext_env_ids].tolist()
                else:
                    extras["train/episode"]['rew_' + key] = \
                        self.env.episode_sums[key][env_ids].tolist()
                self.env.episode_sums[key][env_ids] = 0.

        # log additional curriculum info
        if "ObjectSensor" in self.env.cfg.sensors.sensor_names:
            object_sensor_idx = self.env.cfg.sensors.sensor_names.index("ObjectSensor")
            extras["train/episode"]["Number of env. with ball visible"] = self.env.sensors[object_sensor_idx].visible_envs
        if self.env.cfg.terrain.curriculum:
            extras["train/episode"]["terrain_level"] = torch.mean(
                self.env.terrain_levels.float()).item()
        if self.env.cfg.commands.command_curriculum:
            commands = self.env.commands
            extras["env_bins"] = torch.Tensor(self.env.env_command_bins)
            # extras["train/episode"]["min_command_duration"] = torch.min(commands[:, 8])
            # extras["train/episode"]["max_command_duration"] = torch.max(commands[:, 8])
            # extras["train/episode"]["min_command_bound"] = torch.min(commands[:, 7])
            # extras["train/episode"]["max_command_bound"] = torch.max(commands[:, 7])
            # extras["train/episode"]["min_command_offset"] = torch.min(commands[:, 6])
            # extras["train/episode"]["max_command_offset"] = torch.max(commands[:, 6])
            # extras["train/episode"]["min_command_phase"] = torch.min(commands[:, 5])
            # extras["train/episode"]["max_command_phase"] = torch.max(commands[:, 5])
            # extras["train/episode"]["min_command_freq"] = torch.min(commands[:, 4])
            # extras["train/episode"]["max_command_freq"] = torch.max(commands[:, 4])
            extras["train/episode"]["min_command_x_vel"] = torch.min(commands[:, 0])
            extras["train/episode"]["max_command_x_vel"] = torch.max(commands[:, 0])
            # extras["train/episode"]["min_command_y_vel"] = torch.min(commands[:, 1])
            # extras["train/episode"]["max_command_y_vel"] = torch.max(commands[:, 1])
            # extras["train/episode"]["min_command_yaw_vel"] = torch.min(commands[:, 2])
            # extras["train/episode"]["max_command_yaw_vel"] = torch.max(commands[:, 2])
            # extras["train/episode"]["min_command_body_height"] = torch.min(commands[:, 3])
            # extras["train/episode"]["max_command_body_height"] = torch.max(commands[:, 3])
            # extras["train/episode"]["min_command_swing_height"] = torch.min(commands[:, 9])
            # extras["train/episode"]["max_command_swing_height"] = torch.max(commands[:, 9])
            # extras["train/episode"]["min_command_body_pitch"] = torch.min(commands[:, 10])
            # extras["train/episode"]["max_command_body_pitch"] = torch.max(commands[:, 10])
            # extras["train/episode"]["min_command_body_roll"] = torch.min(commands[:, 11])
            # extras["train/episode"]["max_command_body_roll"] = torch.max(commands[:, 11])
            # extras["train/episode"]["min_command_stance_width"] = torch.min(commands[:, 12])
            # extras["train/episode"]["max_command_stance_width"] = torch.max(commands[:, 12])
            # extras["train/episode"]["min_command_stance_length"] = torch.min(commands[:, 13])
            # extras["train/episode"]["max_command_stance_length"] = torch.max(commands[:, 13])
            # extras["train/episode"]["min_command_aux_reward"] = torch.min(commands[:, 14])
            # extras["train/episode"]["max_command_aux_reward"] = torch.max(commands[:, 14])

            if self.env.cfg.commands.inverse_IK_door_opening:
                extras["train/episode"]["min_command_ee_sphe_radius"] = torch.min(commands[:, 15])
                extras["train/episode"]["max_command_ee_sphe_radius"] = torch.max(commands[:, 15])
                extras["train/episode"]["min_command_ee_sphe_pitch"] = torch.min(commands[:, 16])
                extras["train/episode"]["max_command_ee_sphe_pitch"] = torch.max(commands[:, 16])
                extras["train/episode"]["min_command_ee_sphe_yaw"] = torch.min(commands[:, 17])
                extras["train/episode"]["max_command_ee_sphe_yaw"] = torch.max(commands[:, 17])            
                # extras["train/episode"]["min_command_ee_timing"] = torch.min(commands[:, 18])
                # extras["train/episode"]["max_command_ee_timing"] = torch.max(commands[:, 18])  
                # extras["train/episode"]["min_command_EE_pos_x"] = torch.min(commands[:, 15])
                # extras["train/episode"]["max_command_EE_pos_x"] = torch.max(commands[:, 15])                
                # extras["train/episode"]["min_command_EE_pos_y"] = torch.min(commands[:, 16])
                # extras["train/episode"]["max_command_EE_pos_y"] = torch.max(commands[:, 16])
                # extras["train/episode"]["min_command_EE_pos_z"] = torch.min(commands[:, 17])
                # extras["train/episode"]["max_command_EE_pos_z"] = torch.max(commands[:, 17])
                # extras["train/episode"]["min_command_EE_roll"] = torch.min(commands[:, 18])
                # extras["train/episode"]["max_command_EE_roll"] = torch.max(commands[:, 18])
                # extras["train/episode"]["min_command_EE_pitch"] = torch.min(commands[:, 19])
                # extras["train/episode"]["max_command_EE_pitch"] = torch.max(commands[:, 19])
                # extras["train/episode"]["min_command_EE_yaw"] = torch.min(commands[:, 20])
                # extras["train/episode"]["max_command_EE_yaw"] = torch.max(commands[:, 20])
                # extras["train/episode"]["min_command_EE_gripper"] = torch.min(commands[:, 21])
                # extras["train/episode"]["max_command_EE_gripper"] = torch.max(commands[:, 21])


            # for curriculum, category in zip(self.env.curricula, self.env.category_names):
            #     extras["train/episode"][f"command_area_{category}"] = np.sum(curriculum.weights) / \
            #                                                                curriculum.weights.shape[0]

            # extras["train/episode"]["min_action"] = torch.min(self.env.actions)
            # extras["train/episode"]["max_action"] = torch.max(self.env.actions)

            extras["curriculum/distribution"] = {}
            for curriculum, category in zip(self.env.curricula, self.env.category_names):
                extras[f"curriculum/distribution"][f"weights_{category}"] = curriculum.weights
                extras[f"curriculum/distribution"][f"grid_{category}"] = curriculum.grid
        # if self.env.cfg.env.send_timeouts:
        extras["train/episode"]["Number of env. terminated on time_outs"] = len(self.env.time_out_buf.nonzero(as_tuple=False).flatten())
        extras["train/episode"]["Number of env. terminated on contacts"] = len(self.env.contact_buf.nonzero(as_tuple=False).flatten())    
        if self.env.cfg.rewards.use_terminal_body_height:    
            extras["train/episode"]["Number of env. terminated on body_height"] = len(self.env.body_height_buf.nonzero(as_tuple=False).flatten()) 
        if self.env.cfg.rewards.use_terminal_roll_pitch:        
            extras["train/episode"]["Number of env. terminated on body_roll_pitch"] = len(self.env.body_ori_buf.nonzero(as_tuple=False).flatten()) 
        if self.env.cfg.rewards.use_terminal_time_since_last_obs:
            extras["train/episode"]["Number of env. terminated on time_since_last_obs"] = len(self.env.time_since_last_obs_buf.nonzero(as_tuple=False).flatten())

        extras["train/episode"]['Number of env. reset'] = len(env_ids)
        extras["train/episode"]['Average episode length'] = torch.mean(self.env.episode_length_buf[env_ids]*self.env.dt).item()

        extras["train/episode"]["too_big_change_envs"] = self.env.too_big_change_envs.sum()


        return extras
