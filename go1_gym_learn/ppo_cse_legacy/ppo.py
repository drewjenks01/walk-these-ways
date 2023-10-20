import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from params_proto import PrefixProto

from go1_gym_learn.ppo_cse import ActorCritic
from go1_gym_learn.ppo_cse import RolloutStorage
from go1_gym_learn.ppo_cse import caches


class PPO_Args(PrefixProto):
    # algorithm
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
    learning_rate = 1.e-3  # 5.e-4
    adaptation_module_learning_rate = 1.e-3
    num_adaptation_module_substeps = 1
    adaptation_batch_size = 64
    schedule = 'adaptive'  # could be adaptive, fixed
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.

    selective_adaptation_module_loss = False


class EIPO_Args(PrefixProto):
    alpha_lr = 0.01
    alpha_g_clip = 1.0
    alpha_clip = 10
    alpha_bsz = 8


class PPO:
    actor_critic: ActorCritic

    def __init__(self, actor_critic, alpha, lmbd, device='cpu'):

        self.device = device

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(device)
        self.alpha, self.lmbd = alpha, lmbd
        self.alpha_grad = 0.1

        PPO_Args.adaptation_labels = self.actor_critic.adaptation_labels
        PPO_Args.adaptation_dims = self.actor_critic.adaptation_dims
        PPO_Args.adaptation_weights = self.actor_critic.adaptation_weights
        
        self.storage = None  # initialized later
        self.optimizer = {}
        self.adaptation_module_optimizer = {}
        self.learning_rate = {}
        for n in self.actor_critic.a2c_models:
            self.optimizer[n] = optim.Adam(self.actor_critic.a2c_models[n].parameters(), lr=PPO_Args.learning_rate)
            self.adaptation_module_optimizer[n] = optim.Adam(self.actor_critic.a2c_models[n].parameters(),
                                                        lr=PPO_Args.adaptation_module_learning_rate)
            self.learning_rate[n] = PPO_Args.learning_rate
        if self.actor_critic.decoder:
            self.decoder_optimizer = optim.Adam(self.actor_critic.parameters(),
                                             lr=PPO_Args.adaptation_module_learning_rate)
        self.alpha_derivatives = []
        self.alpha_grad = 1.0
        self.transition = RolloutStorage.Transition()

        

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape,
                     action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape,
                                      obs_history_shape, action_shape, self.actor_critic.exp, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, privileged_obs, obs_history):
        # Compute the actions and values
        self.transition.actions = {n: v.detach() 
                                   for (n, v) in self.actor_critic.act(obs_history).items()}
        self.transition.values = {n: v.detach() 
                                   for (n, v) in self.actor_critic.evaluate(obs_history, privileged_obs).items()}
        self.transition.actions_log_prob = {n: v.detach() 
                                            for (n, v) in self.actor_critic.get_actions_log_prob(self.transition.actions).items()}
        self.transition.action_mean = {n: v.detach() 
                                        for (n, v) in self.actor_critic.action_mean.items()}
        self.transition.action_sigma = {n: v.detach() 
                                        for (n, v) in self.actor_critic.action_std.items()}
        # need to record obs and critic_obs before env.step()
        if 'eipo' in self.actor_critic.exp:
            bsz = len(obs_history) // 2
            self.transition.observations = {'mixed': obs[:bsz], 'ext': obs[bsz:]}
            self.transition.critic_observations = {'mixed': obs[:bsz], 'ext': obs[bsz:]}
            self.transition.privileged_observations = {'mixed': privileged_obs[:bsz], \
                                                    'ext': privileged_obs[bsz:]}
            self.transition.observation_histories = {'mixed': obs_history[:bsz], \
                                                    'ext': obs_history[bsz:]}
        else:
            self.transition.observations = {'ext': obs}
            self.transition.critic_observations = {'ext': obs}
            self.transition.privileged_observations = {'ext': privileged_obs}
            self.transition.observation_histories = {'ext': obs_history}
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = {n: v.clone() for (n, v) in rewards.items()}
        if 'eipo' in self.actor_critic.exp:
            bsz = len(dones) // 2
            self.transition.dones = {'mixed': dones[:bsz], 'ext': dones[bsz:]}
        else:
            self.transition.dones = {'ext': dones}
        # self.transition.env_bins = infos["env_bins"]
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            for n in self.transition.rewards:
                self.transition.rewards[n] += PPO_Args.gamma * torch.squeeze(
                    self.transition.values[n] * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_critic_privileged_obs):
        last_values = {n: v.detach() 
                       for (n, v) in self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs).items()}
        self.storage.compute_returns(last_values, PPO_Args.gamma, PPO_Args.lam)

    @torch.no_grad()
    def process_advantages(self, advantages_batch, rewards_batch):
        if 'const' in self.actor_critic.exp:
            advantages_batch['const_mixed'] = (1 + self.alpha) * \
                advantages_batch['ext'] + self.lmbd * advantages_batch['int']
        
        if 'eipo' in self.actor_critic.exp:
            advantages_batch['eipo_mixed'] = (1 + self.alpha) * \
                advantages_batch['eipo_ext'] + self.lmbd * advantages_batch['int']
            advantages_batch['U_max'] = rewards_batch['ext'] + \
                self.lmbd * rewards_batch['ext_int'] + self.alpha * advantages_batch['ext']
            advantages_batch['U_min'] = (1 + self.alpha) * advantages_batch['eipo_ext'] + \
            advantages_batch['int'] - (rewards_batch['eipo_ext'] + \
                self.lmbd * rewards_batch['int'])

            
    def compute_value_loss(self, value_batch, target_values_batch, returns_batch):
        if PPO_Args.use_clipped_value_loss:
            value_clipped = target_values_batch + \
                            (value_batch - target_values_batch).clamp(-PPO_Args.clip_param,
                                                                        PPO_Args.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()
        return value_loss
    
    def compute_surrogate_loss(self, actions_log_prob_batch, old_actions_log_prob_batch, advantages_batch):
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - PPO_Args.clip_param,
                                                                            1.0 + PPO_Args.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        return surrogate_loss

    def update(self, it):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0
        mean_decoder_loss = 0
        mean_decoder_loss_student = 0
        mean_adaptation_module_test_loss = 0
        mean_decoder_test_loss = 0
        mean_decoder_test_loss_student = 0
        
        mean_adaptation_losses = {}
        label_start_end = {}
        si = 0
        for idx, (label, length) in enumerate(zip(PPO_Args.adaptation_labels, PPO_Args.adaptation_dims)):
            label_start_end[label] = (si, si + length)
            si = si + length
            mean_adaptation_losses[label] = 0
        
        generator = self.storage.mini_batch_generator(PPO_Args.num_mini_batches, PPO_Args.num_learning_epochs)
        for obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, \
            rewards_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, masks_batch, env_bins_batch in generator:

            if 'eipo' in self.actor_critic.exp:
                current_obs_history_batch = torch.cat([obs_history_batch['mixed'], 
                                                       obs_history_batch['ext']])
                current_privileged_obs_batch = torch.cat([privileged_obs_batch['mixed'],
                                                          privileged_obs_batch['ext']])
            else:
                current_obs_history_batch = obs_history_batch['ext']
                current_privileged_obs_batch = privileged_obs_batch['ext']
            self.actor_critic.act(current_obs_history_batch, masks=masks_batch)
            value_batch = self.actor_critic.evaluate(current_obs_history_batch, 
                    current_privileged_obs_batch, masks=masks_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            if 'eipo' in self.actor_critic.exp:
                actions_log_prob_batch['mixed-ext'] = \
                    self.actor_critic.distributions['mixed-ext'].log_prob(actions_batch['ext']).sum(dim=-1)
                actions_log_prob_batch['ext-mixed'] = \
                    self.actor_critic.distributions['ext-mixed'].log_prob(actions_batch['mixed']).sum(dim=-1)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy
            
            # KL
            for n in self.optimizer:
                if PPO_Args.desired_kl != None and PPO_Args.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch[n] / \
                                      old_sigma_batch[n] + 1.e-5)
                              + (torch.square(old_sigma_batch[n]) 
                                 + torch.square(old_mu_batch[n] 
                                                - mu_batch[n])) / (
                                    2.0 * torch.square(sigma_batch[n])) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > PPO_Args.desired_kl * 2.0:
                            self.learning_rate[n] = max(1e-5, self.learning_rate[n] / 1.5)
                        elif kl_mean < PPO_Args.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate[n] = min(1e-2, self.learning_rate[n] * 1.5)

                        for param_group in self.optimizer[n].param_groups:
                            param_group['lr'] = self.learning_rate[n]

            # Surrogate loss
            self.process_advantages(advantages_batch, rewards_batch)
            if self.actor_critic.exp in ['orig', 'trkv', 'enrg']:
                surrogate_loss = self.compute_surrogate_loss(actions_log_prob_batch['ext'], \
                                                            old_actions_log_prob_batch['ext'], \
                                                            advantages_batch['ext'])
            elif 'const' in self.actor_critic.exp:
                surrogate_loss = self.compute_surrogate_loss(actions_log_prob_batch['ext'], \
                                                            old_actions_log_prob_batch['ext'], \
                                                            advantages_batch['const_mixed'])
            elif 'eipo' in self.actor_critic.exp:
                eipo_po = self.compute_surrogate_loss(actions_log_prob_batch['mixed-ext'], \
                                                        old_actions_log_prob_batch['ext'], \
                                                        advantages_batch['U_max'])
                ext_po = self.compute_surrogate_loss(actions_log_prob_batch['ext-mixed'], \
                                                        old_actions_log_prob_batch['mixed'], \
                                                        advantages_batch['U_min'])
                eipo_ao = self.compute_surrogate_loss(actions_log_prob_batch['mixed'], \
                                                        old_actions_log_prob_batch['mixed'], \
                                                        advantages_batch['eipo_mixed'])
                ext_ao = self.compute_surrogate_loss(actions_log_prob_batch['ext'], \
                                                        old_actions_log_prob_batch['ext'], \
                                                        advantages_batch['ext'])
                
                surrogate_loss = eipo_ao + ext_ao + eipo_po + ext_po
                
                with torch.no_grad():
                    # alpha_derivative = self.compute_surrogate_loss(actions_log_prob_batch['mixed-ext'], \
                    #                                         old_actions_log_prob_batch['ext'], \
                    #                                         advantages_batch['ext'])
                    # self.alpha_derivatives.append(alpha_derivative.mean().detach().cpu().item())
                    alpha_derivative = advantages_batch['eipo_ext'].sum().detach().cpu().item() - \
                        advantages_batch['ext'].sum().detach().cpu().item()
                    self.alpha_derivatives.append(alpha_derivative)
                    if len(self.alpha_derivatives) == EIPO_Args.alpha_bsz:
                        self.alpha_grad = np.mean(self.alpha_derivatives) * 1e3
                        self.alpha = self.alpha - EIPO_Args.alpha_lr * np.clip(self.alpha_grad, \
                                                                               -EIPO_Args.alpha_g_clip, EIPO_Args.alpha_g_clip)
                        self.alpha = np.clip(self.alpha, -EIPO_Args.alpha_clip, EIPO_Args.alpha_clip)
                        self.alpha_derivatives = []

            # Value function loss
            value_loss_dict = {}
            for n in value_batch:
                value_loss_dict[n] = self.compute_value_loss(value_batch[n], \
                                                            target_values_batch[n], \
                                                            returns_batch[n])
            total_value_loss = sum(value_loss_dict.values())
            loss = surrogate_loss + PPO_Args.value_loss_coef * total_value_loss - PPO_Args.entropy_coef * entropy_batch.mean()

            # Gradient step
            for n in self.optimizer:
                self.optimizer[n].zero_grad()
            loss.backward()
                
            for n in self.optimizer:
                nn.utils.clip_grad_norm_(self.actor_critic.a2c_models[n].parameters(), 
                                         PPO_Args.max_grad_norm)
                self.optimizer[n].step()

            mean_value_loss += total_value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            data_size = privileged_obs_batch['ext'].shape[0]
            num_train = int(data_size // 5 * 4)

            # Adaptation module gradient step, only update concurrent state estimation module, not policy network
            if len(PPO_Args.adaptation_labels) > 0:

                for epoch in range(PPO_Args.num_adaptation_module_substeps):
                    for n in self.actor_critic.a2c_models:
                        adaptation_pred = self.actor_critic.a2c_models[n].get_student_latent(obs_history_batch[n])
                        with torch.no_grad():
                            adaptation_target = privileged_obs_batch[n]
                        adaptation_loss = 0
                        for idx, (label, length, weight) in enumerate(zip(PPO_Args.adaptation_labels, PPO_Args.adaptation_dims, PPO_Args.adaptation_weights)):

                            start, end = label_start_end[label]
                            selection_indices = torch.linspace(start, end - 1, steps=end - start, dtype=torch.long)

                            idx_adaptation_loss = F.mse_loss(adaptation_pred[:, selection_indices] * weight,
                                                            adaptation_target[:, selection_indices] * weight)
                            mean_adaptation_losses[label] += idx_adaptation_loss.item()

                            adaptation_loss += idx_adaptation_loss

                    if adaptation_loss.requires_grad:
                        self.adaptation_module_optimizer[n].zero_grad()
                        adaptation_loss.backward()
                        self.adaptation_module_optimizer[n].step()

                        mean_adaptation_module_loss += adaptation_loss.item() / len(self.actor_critic.a2c_models)
                        mean_adaptation_module_test_loss += 0  # adaptation_test_loss.item()

        num_updates = PPO_Args.num_learning_epochs * PPO_Args.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_adaptation_module_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps * PPO_Args.adaptation_batch_size)
        mean_decoder_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_loss_student /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_adaptation_module_test_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_test_loss /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        mean_decoder_test_loss_student /= (num_updates * PPO_Args.num_adaptation_module_substeps)
        for label in PPO_Args.adaptation_labels:
            mean_adaptation_losses[label] /= (num_updates * PPO_Args.num_adaptation_module_substeps * PPO_Args.adaptation_batch_size)
        self.storage.clear()
        # if 'eipo' in self.actor_critic.exp and it == 1500:
        #     for p in self.actor_critic.a2c_models['ext'].parameters():
        #         p.requires_grad_(False)
        return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss, mean_decoder_loss, mean_decoder_loss_student, mean_adaptation_module_test_loss, mean_decoder_test_loss, mean_decoder_test_loss_student, mean_adaptation_losses