

import torch

from go1_gym_learn.utils import split_and_pad_trajectories

class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.privileged_observations = None
            self.observation_histories = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.env_bins = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, obs_history_shape, actions_shape, exp, device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.obs_history_shape = obs_history_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = {'ext': torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)}
        self.privileged_observations = {'ext': torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)}
        self.observation_histories = {'ext': torch.zeros(num_transitions_per_env, num_envs, *obs_history_shape, device=self.device)}
        self.rewards = {'ext': torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)}
        self.actions = {'ext': torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)}
        self.dones = {'ext': torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()}

        # For PPO
        self.actions_log_prob = {'ext': torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)}
        self.values = {'ext': torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)}
        self.returns = {'ext': torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)}
        self.advantages = {'ext': torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)}
        self.mu = {'ext': torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)}
        self.sigma = {'ext': torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)}
        self.env_bins = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0
        self.exp = exp

        if 'const' in self.exp:
            self.rewards['int'] = self.rewards['ext'].clone()
            self.values['int'] = self.values['ext'].clone()
            self.returns['int'] = self.returns['ext'].clone()
            self.advantages['int'] = self.advantages['ext'].clone()
        elif 'eipo' in self.exp:
            self.actions['mixed'] = self.actions['ext'].clone()
            self.actions['mixed-ext'] = self.actions['ext'].clone()
            self.actions['ext-mixed'] = self.actions['ext'].clone()
            self.mu['mixed'] = self.mu['ext'].clone()
            self.sigma['mixed'] = self.sigma['ext'].clone()
            self.actions_log_prob['mixed'] = self.actions_log_prob['ext'].clone()
            self.actions_log_prob['mixed-ext'] = self.actions_log_prob['ext'].clone()
            self.actions_log_prob['ext-mixed'] = self.actions_log_prob['ext'].clone()
            self.rewards['eipo_ext'] = self.rewards['ext'].clone()
            self.rewards['int'] = self.rewards['ext'].clone()
            self.rewards['ext_int'] = self.rewards['ext'].clone()
            self.values['eipo_ext'] = self.values['ext'].clone()
            self.values['int'] = self.values['ext'].clone()
            self.returns['eipo_ext'] = self.returns['ext'].clone()
            self.returns['int'] = self.returns['ext'].clone()
            self.advantages['eipo_ext'] = self.advantages['ext'].clone()
            self.advantages['int'] = self.advantages['ext'].clone()
            self.observations['mixed'] = self.observations['ext'].clone()
            self.privileged_observations['mixed'] = self.privileged_observations['ext'].clone()
            self.observation_histories['mixed'] = self.observation_histories['ext'].clone()
            self.dones['mixed'] = self.dones['ext'].clone()

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        for n in self.observations:
            self.observations[n][self.step].copy_(transition.observations[n])
            self.privileged_observations[n][self.step].copy_(transition.privileged_observations[n])
            self.observation_histories[n][self.step].copy_(transition.observation_histories[n])
            self.dones[n][self.step].copy_(transition.dones[n].view(-1, 1))
        for n in transition.actions:
            self.actions[n][self.step].copy_(transition.actions[n])
            self.actions_log_prob[n][self.step].copy_(transition.actions_log_prob[n].view(-1, 1))
        for n in transition.rewards:
            self.rewards[n][self.step].copy_(transition.rewards[n].view(-1, 1))
        for n in transition.values:
            self.values[n][self.step].copy_(transition.values[n])    
        for n in self.mu:
            self.mu[n][self.step].copy_(transition.action_mean[n])
            self.sigma[n][self.step].copy_(transition.action_sigma[n])
        # self.env_bins[self.step].copy_(transition.env_bins.view(-1, 1))
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        for n in last_values:
            advantage = 0
            for step in reversed(range(self.num_transitions_per_env)):
                if step == self.num_transitions_per_env - 1:
                    next_values = last_values[n]
                else:
                    next_values = self.values[n][step + 1]
                dn = 'mixed' if 'eipo' in n else 'ext'
                next_is_not_terminal = 1.0 - self.dones[dn][step].float()
                delta = self.rewards[n][step] + next_is_not_terminal * gamma * next_values - self.values[n][step]
                advantage = delta + next_is_not_terminal * gamma * lam * advantage
                self.returns[n][step] = advantage + self.values[n][step]

            # Compute and normalize the advantages
            self.advantages[n] = self.returns[n] - self.values[n]
            self.advantages[n] = (self.advantages[n] - self.advantages[n].mean()) / (self.advantages[n].std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards['ext'].mean()

    def generate_permutation_properties(self, num_mini_batches):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, 
                                    requires_grad=False, device=self.device)
        return mini_batch_size, indices

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        mini_batch_size, indices = self.generate_permutation_properties(num_mini_batches)
        observations = {n: v.flatten(0, 1) for (n, v) in self.observations.items()}
        privileged_obs = {n: v.flatten(0, 1) for (n, v) in self.privileged_observations.items()}
        obs_history = {n: v.flatten(0, 1) for (n, v) in self.observation_histories.items()}
        critic_observations = observations

        actions = {n: v.flatten(0, 1) for (n, v) in self.actions.items()}
        values = {n: v.flatten(0, 1) for (n, v) in self.values.items()}
        rewards = {n: v.flatten(0, 1) for (n, v) in self.rewards.items()}
        returns = {n: v.flatten(0, 1) for (n, v) in self.returns.items()}
        old_actions_log_prob = {n: v.flatten(0, 1) for (n, v) in self.actions_log_prob.items()}
        advantages = {n: v.flatten(0, 1) for (n, v) in self.advantages.items()}
        old_mu = {n: v.flatten(0, 1) for (n, v) in self.mu.items()}
        old_sigma = {n: v.flatten(0, 1) for (n, v) in self.sigma.items()}
        # old_env_bins = self.env_bins.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):

                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = {n: v[batch_idx] \
                                 for (n, v) in observations.items()}
                critic_observations_batch = {n: v[batch_idx] \
                                 for (n, v) in critic_observations.items()}
                privileged_obs_batch = {n: v[batch_idx] \
                                 for (n, v) in privileged_obs.items()}
                obs_history_batch = {n: v[batch_idx] \
                                 for (n, v) in obs_history.items()}
                actions_batch = {n: v[batch_idx] \
                                 for (n, v) in actions.items()}
                target_values_batch = {n: v[batch_idx] \
                                 for (n, v) in values.items()}
                rewards_batch = {n: v[batch_idx] \
                                 for (n, v) in rewards.items()}
                returns_batch = {n: v[batch_idx] \
                                 for (n, v) in returns.items()}
                old_actions_log_prob_batch = {n: v[batch_idx] \
                                              for (n, v) in old_actions_log_prob.items()}
                advantages_batch = {n: v[batch_idx] \
                                    for (n, v) in advantages.items()}
                old_mu_batch = {n: v[batch_idx] \
                                    for (n, v) in old_mu.items()}
                old_sigma_batch = {n: v[batch_idx] \
                                    for (n, v) in old_sigma.items()}
                # env_bins_batch = old_env_bins[batch_idx]
                env_bins_batch = None
                yield obs_batch, critic_observations_batch, privileged_obs_batch, obs_history_batch, actions_batch, \
                    target_values_batch, advantages_batch, rewards_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, None, env_bins_batch

    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):

        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        padded_privileged_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.privileged_observations, self.dones)
        padded_obs_history_trajectories, trajectory_masks = split_and_pad_trajectories(self.observation_histories, self.dones)
        padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i*mini_batch_size
                stop = (i+1)*mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size
                
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]
                privileged_obs_batch = padded_privileged_obs_trajectories[:, first_traj:last_traj]
                obs_history_batch = padded_obs_history_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                yield obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, masks_batch
                
                first_traj = last_traj