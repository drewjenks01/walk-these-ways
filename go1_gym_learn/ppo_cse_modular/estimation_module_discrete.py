import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from torch.distributions import Normal
import numpy as np

from go1_gym_learn.ppo_cse_modular.estimation_module import Estimation_Args


class EstimationModuleDiscrete(nn.Module):
    is_recurrent = False

    def __init__(self,
                 num_privileged_obs,
                 num_obs_history,
                 **kwargs):
        if kwargs:
            print("Estimator_Args.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super().__init__()
        
        self.labels = Estimation_Args.labels
        self.dims = Estimation_Args.dims
        self.weights = Estimation_Args.weights
        self.discrete_labels = Estimation_Args.discrete_labels
        self.discrete_num_bins = Estimation_Args.discrete_num_bins
        self.discrete_bin_ranges = torch.tensor(Estimation_Args.discrete_bin_ranges, dtype=torch.float)

        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs

        activation = get_activation(Estimation_Args.activation)

        # Estimation module
        # self.estimation_modules = []
        self.estimation_modules = nn.ModuleList()
        self.optimizers = []
        self.criteria = []
        self.elementwise_criteria = []

        for (label, dim, weight, is_discrete, num_bins, bins_range) in zip(self.labels, self.dims, self.weights, self.discrete_labels, self.discrete_num_bins, self.discrete_bin_ranges):
            # setattr(self, label + '_estimation_module', EstimationModuleBranch(dim, weight))
            if is_discrete:
                estimation_module_layers = []
                estimation_module_layers.append(nn.Linear(self.num_obs_history, Estimation_Args.branch_hidden_dims[0]))
                estimation_module_layers.append(activation)
                for l in range(len(Estimation_Args.branch_hidden_dims)):
                    if l == len(Estimation_Args.branch_hidden_dims) - 1:
                        estimation_module_layers.append(
                            nn.Linear(Estimation_Args.branch_hidden_dims[l], dim * num_bins))
                        # estimation_module_layers.append(nn.Softmax(dim=-1))
                        # estimation_module_layers.append(activation)
                    else:
                        estimation_module_layers.append(
                            nn.Linear(Estimation_Args.branch_hidden_dims[l],
                                    Estimation_Args.branch_hidden_dims[l + 1]))
                        estimation_module_layers.append(activation)
                self.criteria += [nn.CrossEntropyLoss()]
                self.elementwise_criteria += [nn.CrossEntropyLoss(reduction='none')]
                    
            else:
                estimation_module_layers = []
                estimation_module_layers.append(nn.Linear(self.num_obs_history, Estimation_Args.branch_hidden_dims[0]))
                estimation_module_layers.append(activation)
                for l in range(len(Estimation_Args.branch_hidden_dims)):
                    if l == len(Estimation_Args.branch_hidden_dims) - 1:
                        estimation_module_layers.append(
                            nn.Linear(Estimation_Args.branch_hidden_dims[l], dim))
                    else:
                        estimation_module_layers.append(
                            nn.Linear(Estimation_Args.branch_hidden_dims[l],
                                    Estimation_Args.branch_hidden_dims[l + 1]))
                        estimation_module_layers.append(activation)
                self.criteria += [nn.MSELoss()]
                self.elementwise_criteria += [nn.MSELoss(reduction='none')]
            
            estimation_module = nn.Sequential(*estimation_module_layers)

            self.estimation_modules.append(estimation_module)

            self.optimizers += [optim.Adam(estimation_module.parameters(),
                                        lr=Estimation_Args.learning_rate)]
        
        self.label_start_end = {}
        si = 0
        for idx, (label, length) in enumerate(zip(Estimation_Args.labels, Estimation_Args.dims)):
            self.label_start_end[label] = (si, si + length)
            si = si + length

        # if Estimation_Args.use_buffer:
        self.obs_history_buffer = torch.zeros((Estimation_Args.buffer_size, self.num_obs_history))
        self.privileged_obs_buffer = torch.zeros((Estimation_Args.buffer_size, self.num_privileged_obs))

    def binarize(self, obs):
        ret = []
        for idx, (label, length, bins_range, num_bins) in enumerate(zip(Estimation_Args.labels, Estimation_Args.dims, Estimation_Args.discrete_bin_ranges, Estimation_Args.discrete_num_bins)):
            if self.discrete_labels[idx]:
                adaptation_target = obs[:, self.label_start_end[label][0]:self.label_start_end[label][1]]
                adaptation_target = torch.bucketize(adaptation_target, torch.linspace(bins_range[0], bins_range[1], steps=num_bins).to(adaptation_target.device)).clip(0, num_bins - 1)
                adaptation_target = torch.eye(num_bins, device=adaptation_target.device)[adaptation_target.long().squeeze(1)]
                ret.append(adaptation_target)
            else:
                ret.append(obs[:, self.label_start_end[label][0]:self.label_start_end[label][1]])
        ret = torch.cat(ret, dim=-1)
        return ret

    def to(self, device):
        if Estimation_Args.use_buffer:
            self.obs_history_buffer = self.obs_history_buffer.to(device)
            self.privileged_obs_buffer = self.privileged_obs_buffer.to(device)
        super().to(device)
        return self
            
    def forward(self, obs):
        ret = []
        
        for idx, module in enumerate(self.estimation_modules):
            is_discrete = self.discrete_labels[idx]
            this_ret = module(obs)
            if is_discrete:
                this_ret = torch.argmax(this_ret, dim=-1).float()
                this_ret = this_ret / (self.discrete_num_bins[idx] - 1) * (self.discrete_bin_ranges[idx][1] - self.discrete_bin_ranges[idx][0]) + self.discrete_bin_ranges[idx][0]
                this_ret = this_ret.unsqueeze(-1)
            ret.append(this_ret)
        ret = torch.cat(ret, dim=-1)
        return ret
    
    def update_buffer(self, privileged_obs_batch, obs_history_batch):
        # replace oldest data
        subsample_idx = torch.randperm(obs_history_batch.shape[0])[:int(Estimation_Args.buffer_size * Estimation_Args.buffer_replace_rate)]
        self.obs_history_buffer = torch.cat((self.obs_history_buffer, obs_history_batch[subsample_idx]), dim=0)[-Estimation_Args.buffer_size:]
        self.privileged_obs_buffer = torch.cat((self.privileged_obs_buffer, privileged_obs_batch[subsample_idx]), dim=0)[-Estimation_Args.buffer_size:]

    def compute_losses(self, privileged_obs_batch, obs_history_batch, reduce=True):
        if reduce:
            criteria = self.criteria
        else:
            criteria = self.elementwise_criteria

        adaptation_losses = []
        for idx, (label, dim, weight, is_discrete, num_bins, bins_range, module, optimizer, criterion) in enumerate(zip(self.labels, self.dims, self.weights, self.discrete_labels, self.discrete_num_bins, self.discrete_bin_ranges, self.estimation_modules, self.optimizers, criteria)):
            selection_indices = torch.linspace(self.label_start_end[label][0], self.label_start_end[label][1] - 1, steps=dim, dtype=torch.long)
            adaptation_target = privileged_obs_batch[:, selection_indices]

            adaptation_pred = module(obs_history_batch)

            # binarize
            if is_discrete:
                adaptation_target = torch.bucketize(adaptation_target, torch.linspace(bins_range[0], bins_range[1], steps=num_bins).to(adaptation_target.device)).clip(0, num_bins - 1).squeeze(1)
                # epsilon = 1e-8
                # log_probabilities = torch.log(adaptation_pred + epsilon)
                idx_adaptation_loss = criterion(adaptation_pred,
                                                    adaptation_target)
                
            else:
                idx_adaptation_loss = criterion(adaptation_pred * weight,
                                                    adaptation_target * weight)
                
            adaptation_losses += [idx_adaptation_loss]
                
        return adaptation_losses
                            

    def update(self, privileged_obs_batch, obs_history_batch):
        data_size = privileged_obs_batch.shape[0]

        mean_adaptation_losses = {label: 0 for label in Estimation_Args.labels}
            
        # buffer training
        num_buffer_batches = 0
        if Estimation_Args.use_buffer:
            num_buffer_batches = Estimation_Args.buffer_size // Estimation_Args.batch_size
            total_samples = Estimation_Args.batch_size * num_buffer_batches

            for epoch in range(Estimation_Args.num_substeps):
                for batch in range(num_buffer_batches):
                    batch_indices = np.linspace(batch * Estimation_Args.batch_size, (batch + 1) * Estimation_Args.batch_size - 1, Estimation_Args.batch_size, dtype=int)  
                    adaptation_losses = self.compute_losses(self.privileged_obs_buffer[batch_indices], self.obs_history_buffer[batch_indices])
                    
                    for (label, adaptation_loss, optimizer) in zip(self.labels, adaptation_losses, self.optimizers):
                        optimizer.zero_grad()
                        adaptation_loss.backward()
                        optimizer.step()
                        mean_adaptation_losses[label] += adaptation_loss.item()
        else:
            self.obs_history_buffer = obs_history_batch
            self.privileged_obs_buffer = privileged_obs_batch

        num_batches = Estimation_Args.num_batches
        total_samples = Estimation_Args.batch_size * num_batches

        assert(total_samples <= data_size, "there is not enough data to train the estimation module! decrease the batch size or number of batches")

        # on policy training
        for epoch in range(Estimation_Args.num_substeps):
            random_indices = np.random.choice(data_size, total_samples, replace=False)
            # batches
            for batch in range(num_batches):
                batch_indices = random_indices[batch * Estimation_Args.batch_size: (batch + 1) * Estimation_Args.batch_size]
                adaptation_losses = self.compute_losses(privileged_obs_batch[batch_indices], obs_history_batch[batch_indices])
                    
                for (label, adaptation_loss, optimizer) in zip(self.labels, adaptation_losses, self.optimizers):
                    optimizer.zero_grad()
                    adaptation_loss.backward()
                    optimizer.step()
                    mean_adaptation_losses[label] += adaptation_loss.item()

        if Estimation_Args.use_buffer:
            self.update_buffer(privileged_obs_batch, obs_history_batch)

        for label in Estimation_Args.labels:
            mean_adaptation_losses[label] /= (Estimation_Args.num_substeps * (num_batches + num_buffer_batches))

        return mean_adaptation_losses

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
