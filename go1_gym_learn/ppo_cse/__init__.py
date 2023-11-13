import time
from collections import deque
import copy
import os

import torch
# from ml_logger import logger
import wandb
from wandb_osh.hooks import TriggerWandbSyncHook

from params_proto import PrefixProto

from .actor_critic import ActorCritic
from .rollout_storage import RolloutStorage


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class DataCaches:
    def __init__(self, curriculum_bins):
        from go1_gym_learn.ppo_cse.metrics_caches import SlotCache, DistCache

        self.slot_cache = SlotCache(curriculum_bins)
        self.dist_cache = DistCache()


caches = DataCaches(1)


class RunnerArgs(PrefixProto, cli=False):
    # runner
    algorithm_class_name = 'RMA'
    num_steps_per_env = 24  # per iteration
    max_iterations = 50000  # number of policy updates

    # logging
    save_interval = 400  # check for potential saves every this many iterations
    save_video_interval = 100
    log_freq = 10

    # load and resume
    resume = False
    resume_supercloud = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt
    resume_curriculum = True
    resume_checkpoint = 'ac_weights_last.pt'



class Runner:

    def __init__(self, env, args, device='cpu'):
        from .ppo import PPO

        self.device = device
        self.env = env
        self.exp = args.exp

        self.randint = torch.randint(0, 1000000, (1,)).item()

        actor_critic = ActorCritic(self.env.num_obs,
                                      self.env.num_privileged_obs,
                                      self.env.num_obs_history,
                                      self.env.num_actions,
                                      self.exp,
                                      args.init_path).to(self.device)

        from go1_gym import MINI_GYM_ROOT_DIR
        # Load weights from checkpoint 
        if RunnerArgs.resume:
            body = wandb.restore(RunnerArgs.resume_checkpoint, run_path=RunnerArgs.resume_path)            
            actor_critic.load_state_dict(torch.load(body.name))
            print(f"Successfully loaded weights from checkpoint ({RunnerArgs.resume_checkpoint}) and run path ({RunnerArgs.resume_path})")
        elif RunnerArgs.resume_supercloud:
            print(f"Loading weights from checkpoint ({RunnerArgs.resume_checkpoint}) and run path ({RunnerArgs.resume_path} and {MINI_GYM_ROOT_DIR})")
            path = MINI_GYM_ROOT_DIR+ "/resume_runs/" + RunnerArgs.resume_path + "/" + RunnerArgs.resume_checkpoint
            print("path: ", path)
            actor_critic.load_state_dict(torch.load(path))

        self.alg = PPO(actor_critic, args.alpha, args.lmbd, device=self.device)
        self.num_steps_per_env = RunnerArgs.num_steps_per_env

        # init storage and model
        num_train_envs = self.env.num_train_envs
        if 'eipo' in self.alg.actor_critic.exp:
            num_train_envs = num_train_envs // 2
        self.alg.init_storage(num_train_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_obs_history], [self.env.num_actions], args)

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.last_recording_it = -RunnerArgs.save_video_interval

        self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, eval_freq=100, curriculum_dump_freq=500, eval_expert=False):
        # from ml_logger import logger
        # initialize writer
        # assert logger.prefix, "you will overwrite the entire instrument server"

        # logger.start('start', 'epoch', 'episode', 'run', 'step')

        trigger_sync = TriggerWandbSyncHook()
        wandb.watch(self.alg.actor_critic, log=None, log_freq=RunnerArgs.log_freq)

        if init_at_random_ep_len:
            self.env.episode_length_buf[:] = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # split train and test envs
        num_train_envs = self.env.num_train_envs
        num_envs = self.env.num_envs
        if 'eipo' in self.alg.actor_critic.exp:
            num_envs = num_envs // 2
        obs_dict = self.env.get_observations()  # TODO: check, is this correct on the first step?
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(num_envs, dtype=torch.float, device=self.device)

        record_log = {}
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            energies = []
            torque_uncertainty_reward = []
            velocities = []
            std_norm = []
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs],
                                                 obs_history[:num_train_envs])
                    
                    if 'eipo' in self.alg.actor_critic.exp:
                        actions = torch.cat([actions['mixed'], actions['ext']])
                    else:
                        actions = actions['ext']
                    ret = self.env.step(actions)
                    obs_dict, rewards, dones, infos = ret

                    ## Logging metrics
                    energies.append(self.env.compute_energy().mean().item())
                    torque_uncertainty_reward.append(torch.exp(-self.env.compute_torque_uncertainty()).mean().item())
                    velocities.append(self.env.base_lin_vel[:,0].mean().item())
                    ## Logging norm of action std
                    action_stds = self.alg.actor_critic.action_std
                    std_norms = {n: v.norm(dim=-1).mean().item() for (n, v) in action_stds.items()}
                    # take mean over dictionary
                    # find number of keys in dictionary std_norms
                    std_norm.append(sum(std_norms.values()) / len(std_norms))

                    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"]

                    obs, privileged_obs, obs_history, dones = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), dones.to(self.device)
                    rewards = {n: v.to(self.device) for (n, v) in rewards.items()}
                    self.alg.process_env_step({n: v[:num_train_envs] for (n, v) in rewards.items()}
                            , dones[:num_train_envs], infos)
                    if 'eipo' in self.alg.actor_critic.exp:
                        self.alg.lgrgn_mtpr.compute_alpha_values(self.env.commands[:,0])
                    if 'train/episode' in infos:
                        for (k, v) in infos['train/episode'].items():
                            if 'rew_' in k and k in record_log:
                                record_log[k] = record_log[k] + v
                            else:
                                record_log[k] = v
         
                        # with logger.Prefix(metrics="train/episode"):
                        #     logger.store_metrics(**infos['train/episode'])
                        # wandb.log(infos['train/episode'], step=it)
                        


                    if 'curriculum' in infos:
                        rewards_record = rewards['ext']
                        if 'eipo' in self.exp:
                            rewards_record = rewards_record['eipo_ext']
                        cur_reward_sum += rewards_record
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        new_ids_train = new_ids[new_ids < num_train_envs]
                        rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_train] = 0
                        cur_episode_length[new_ids_train] = 0

                    if 'curriculum/distribution' in infos:
                        distribution = infos['curriculum/distribution']

                # stop = time.time()
                # collection_time = stop - start

                # Learning step
                # start = stop
                self.alg.compute_returns(obs_history[:num_train_envs], privileged_obs[:num_train_envs])

                # if it % curriculum_dump_freq == 0:
                #     logger.save_pkl({"iteration": it,
                #                      **caches.slot_cache.get_summary(),
                #                      **caches.dist_cache.get_summary()},
                #                     path=f"curriculum/info.pkl", append=True)

                #     if 'curriculum/distribution' in infos:
                #         logger.save_pkl({"iteration": it,
                #                          "distribution": distribution},
                #                          path=f"curriculum/distribution.pkl", append=True)

            mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student, mean_adaptation_losses_dict = self.alg.update(it)
            if 'frz' in self.alg.actor_critic.exp and (it + 1) % 500 == 0:
                ext_state = self.alg.actor_critic.a2c_models['ext'].state_dict()
                mixed_state = self.alg.actor_critic.a2c_models['mixed'].state_dict()
                new_ext_state = {name: param.clone() for name, param in mixed_state.items() if name in ext_state}
                self.alg.actor_critic.a2c_models['ext'].load_state_dict(new_ext_state)
                for para in self.alg.actor_critic.a2c_models['ext'].parameters():
                    para.requires_grad = False
            stop = time.time()
            learn_time = stop - start

            # logger.store_metrics(
            #     # total_time=learn_time - collection_time,
            #     time_elapsed=logger.since('start'),
            #     time_iter=logger.split('epoch'),
            #     adaptation_loss=mean_adaptation_module_loss,
            #     mean_value_loss=mean_value_loss,
            #     mean_surrogate_loss=mean_surrogate_loss,
            #     mean_decoder_loss=mean_decoder_loss,
            #     mean_decoder_loss_student=mean_decoder_loss_student,
            #     mean_decoder_test_loss=mean_decoder_test_loss,
            #     mean_decoder_test_loss_student=mean_decoder_test_loss_student,
            #     mean_adaptation_module_test_loss=mean_adaptation_module_test_loss
            # )
            for k in record_log:
                if 'rew_' in k:
                    record_log[k] = sum(record_log[k]) / len(record_log[k])
            record_log.update({
                "time_iter": learn_time,
                "adaptation_loss": mean_adaptation_module_loss,
                "mean_value_loss": mean_value_loss,
                "mean_surrogate_loss": mean_surrogate_loss,
                "mean_decoder_loss": mean_decoder_loss,
                "mean_decoder_loss_student": mean_decoder_loss_student,
                "mean_decoder_test_loss": mean_decoder_test_loss,
                "mean_decoder_test_loss_student": mean_decoder_test_loss_student,
                "mean_adaptation_module_test_loss": mean_adaptation_module_test_loss
            })
            if 'eipo' in self.alg.actor_critic.exp:
                if isinstance(self.alg.lgrgn_mtpr.alpha, float):
                    record_log['alpha_value'] = self.alg.lgrgn_mtpr.alpha
                else:
                    for idx, v in enumerate(self.alg.lgrgn_mtpr.alpha.weight.data):
                        record_log[f'alpha_value_{idx}'] = v.item()
                record_log['alpha_grad'] = self.alg.lgrgn_mtpr.alpha_grad
            record_log['avg_energy_consumption'] = sum(energies) / len(energies)
            record_log['torque_uncertainty_reward'] = sum(torque_uncertainty_reward) / len(torque_uncertainty_reward)
            record_log['std_norm'] = sum(std_norm) / len(std_norm)
            record_log['avg_velocity'] = sum(velocities) / len(velocities)

            wandb_record_log = {}
            for k in record_log:
                if k.startswith('rew_ext'):
                    wandb_record_log[k.replace('rew_ext_', 'return_ext/')] = record_log[k]
                elif k.startswith('rew_mixed'):
                    wandb_record_log[k.replace('rew_mixed_', 'return_mixed/')] = record_log[k]
                elif k.startswith('rew'):
                    wandb_record_log[k.replace('rew_', 'return/')] = record_log[k]
                else:
                    wandb_record_log[f'metrics/{k}'] = record_log[k]
                
            wandb.log(wandb_record_log, step=it)

            record_log = {}

            
            # logger.store_metrics(**mean_adaptation_losses_dict)
            wandb.log(mean_adaptation_losses_dict, step=it)

            if RunnerArgs.save_video_interval:
                self.log_video(it)

            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            # if logger.every(RunnerArgs.log_freq, "iteration", start_on=1):
            # if it % Config.log_freq == 0:
            # logger.log_metrics_summary(key_values={"timesteps": self.tot_timesteps, "iterations": it})
            # logger.job_running()
            wandb.log({"timesteps": self.tot_timesteps, "iterations": it}, step=it)
            trigger_sync()

            if it % RunnerArgs.save_interval == 0:
                    print(f"Saving model at iteration {it}")
                # with logger.Sync():
                #     logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
                #     logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")
                    
                    path = f'./tmp{self.randint}/legged_data-{self.alg.actor_critic.exp}'
                    if 'const' in path:
                        path = path + '-alpha{:.1f}'.format(self.alg.alpha)
                    for n in self.alg.actor_critic.a2c_models:
                        os.makedirs(f'{path}-{n}', exist_ok=True)

                        ac_weight_path = f'{path}-{n}/ac_weights_{it}.pt'
                        torch.save(self.alg.actor_critic.a2c_models[n].state_dict(), ac_weight_path)
                        wandb.save(ac_weight_path)

                        ac_weight_path = f'{path}-{n}/ac_weights_latest.pt'
                        torch.save(self.alg.actor_critic.a2c_models[n].state_dict(), ac_weight_path)
                        wandb.save(ac_weight_path)

                        adaptation_module_path = f'{path}-{n}/adaptation_module_{it}.jit'
                        adaptation_module = copy.deepcopy(self.alg.actor_critic.a2c_models[n].adaptation_module).to('cpu')
                        traced_script_adaptation_module = torch.jit.script(adaptation_module)
                        traced_script_adaptation_module.save(adaptation_module_path)
                        wandb.save(adaptation_module_path)

                        adaptation_module_path = f'{path}-{n}/adaptation_module_latest.jit'
                        traced_script_adaptation_module.save(adaptation_module_path)
                        wandb.save(adaptation_module_path)

                        body_path = f'{path}-{n}/body_{it}.jit'
                        body_model = copy.deepcopy(self.alg.actor_critic.a2c_models[n].actor_body).to('cpu')
                        traced_script_body_module = torch.jit.script(body_model)
                        traced_script_body_module.save(body_path)
                        wandb.save(body_path)

                        body_path = f'{path}-{n}/body_latest.jit'
                        traced_script_body_module.save(body_path)
                        wandb.save(body_path)

                        # logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
                        # logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)

                        ac_weights_path = f"{path}-{n}/ac_weights_{it}.pt"
                        torch.save(self.alg.actor_critic.a2c_models[n].state_dict(), ac_weights_path)
                        ac_weights_path = f"{path}-{n}/ac_weights_latest.pt"
                        torch.save(self.alg.actor_critic.a2c_models[n].state_dict(), ac_weights_path)
                    
                    wandb.save(f"./tmp{self.randint}/legged_data/adaptation_module_{it}.jit")
                    wandb.save(f"./tmp{self.randint}/legged_data/body_{it}.jit")
                    wandb.save(f"./tmp{self.randint}/legged_data/ac_weights_{it}.pt")
                    wandb.save(f"./tmp{self.randint}/legged_data/adaptation_module_latest.jit")
                    wandb.save(f"./tmp{self.randint}/legged_data/body_latest.jit")
                    wandb.save(f"./tmp{self.randint}/legged_data/ac_weights_latest.pt")
                    
            self.current_learning_iteration += num_learning_iterations

        # torch.save(self.alg.actor_critic.state_dict(), f"./tmp/legged_data/ac_weights_last.pt")
        # wandb.save(f"./tmp/legged_data/ac_weights_last.pt")

        path = f'./tmp{self.randint}/legged_data'
        for n in self.alg.actor_critic.a2c_models:
            os.makedirs(f'{path}-{n}', exist_ok=True)

            adaptation_module_path = f'{path}-{n}/adaptation_module_latest.jit'
            adaptation_module = copy.deepcopy(self.alg.actor_critic.a2c_models[n].adaptation_module).to('cpu')
            traced_script_adaptation_module = torch.jit.script(adaptation_module)
            traced_script_adaptation_module.save(adaptation_module_path)

            body_path = f'{path}-{n}/body_latest.jit'
            body_model = copy.deepcopy(self.alg.actor_critic.a2c_models[n].actor_body).to('cpu')
            traced_script_body_module = torch.jit.script(body_model)
            traced_script_body_module.save(body_path)

        # logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/", once=False)
        # logger.upload_file(file_path=body_path, target_path=f"checkpoints/", once=False)
        wandb.save(f"./tmp{self.randint}/legged_data/adaptation_module_latest.jit")
        wandb.save(f"./tmp{self.randint}/legged_data/body_latest.jit")

    def log_video(self, it):
        if it - self.last_recording_it >= RunnerArgs.save_video_interval:
            self.env.start_recording()
            print("START RECORDING")
            self.last_recording_it = it

        frames = self.env.get_complete_frames()
        if len(frames) > 0:
            self.env.pause_recording()
            print("LOGGING VIDEO")
            import numpy as np
            video_array = np.concatenate([np.expand_dims(frame, axis=0) for frame in frames ], axis=0).swapaxes(1, 3).swapaxes(2, 3)
            print(video_array.shape)
            # logger.save_video(frames, f"videos/{it:05d}.mp4", fps=1 / self.env.dt)
            wandb.log({"video": wandb.Video(video_array, fps=1 / self.env.dt)}, step=it)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert