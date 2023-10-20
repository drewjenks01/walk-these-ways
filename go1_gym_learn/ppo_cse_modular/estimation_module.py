import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from params_proto import PrefixProto
from torch.distributions import Normal
import numpy as np

class Estimation_Args(PrefixProto, cli=False):
    activation = 'elu'

    learning_rate = 1e-4
    
    branch_hidden_dims = [256, 128]

    batch_size = 250
    num_batches = 4

    use_buffer = False
    buffer_size = 100000
    buffer_replace_rate = 0.05
    
    labels = []
    dims = []
    weights = []

    discrete_labels = []
    discrete_num_bins = []
    discrete_bin_ranges = []

    num_substeps = 1


class EstimationModule(nn.Module):
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

        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs

        activation = get_activation(Estimation_Args.activation)

        # Estimation module
        estimation_module_layers = []
        estimation_module_layers.append(nn.Linear(self.num_obs_history, Estimation_Args.branch_hidden_dims[0]))
        estimation_module_layers.append(activation)
        for l in range(len(Estimation_Args.branch_hidden_dims)):
            if l == len(Estimation_Args.branch_hidden_dims) - 1:
                estimation_module_layers.append(
                    nn.Linear(Estimation_Args.branch_hidden_dims[l], self.num_privileged_obs))
            else:
                estimation_module_layers.append(
                    nn.Linear(Estimation_Args.branch_hidden_dims[l],
                              Estimation_Args.branch_hidden_dims[l + 1]))
                estimation_module_layers.append(activation)
        self.estimation_module = nn.Sequential(*estimation_module_layers)


        self.optimizer = optim.Adam(self.parameters(),
                                    lr=Estimation_Args.learning_rate)
        
        self.label_start_end = {}
        si = 0
        for idx, (label, length) in enumerate(zip(Estimation_Args.labels, Estimation_Args.dims)):
            self.label_start_end[label] = (si, si + length)
            si = si + length
            
    def forward(self, obs):
        return self.estimation_module(obs)
    
    def update(self, privileged_obs_batch, obs_history_batch):
        data_size = privileged_obs_batch.shape[0]
        # num_train = int(data_size // 5 * 4)

        mean_adaptation_losses = {label: 0 for label in Estimation_Args.labels}
            
        num_random_samples = 1000
        num_batches = num_random_samples // Estimation_Args.batch_size
        
        for epoch in range(Estimation_Args.num_substeps):
            random_indices = np.random.choice(data_size, num_random_samples, replace=False)
            # batches
            for batch in range(num_batches):
                batch_indices = random_indices[batch * Estimation_Args.batch_size: (batch + 1) * Estimation_Args.batch_size]

                adaptation_pred = self.forward(obs_history_batch[batch_indices])
                with torch.no_grad():
                    adaptation_target = privileged_obs_batch[batch_indices]
                adaptation_loss = 0
                for idx, (label, length, weight) in enumerate(zip(Estimation_Args.labels, Estimation_Args.dims, Estimation_Args.weights)):
                    selection_indices = torch.linspace(self.label_start_end[label][0], self.label_start_end[label][1] - 1, steps=length, dtype=torch.long)
                    idx_adaptation_loss = F.mse_loss(adaptation_pred[:, selection_indices] * weight,
                                                        adaptation_target[:, selection_indices] * weight)
                    adaptation_loss += idx_adaptation_loss
                    mean_adaptation_losses[label] += idx_adaptation_loss.item()
                

                self.optimizer.zero_grad()
                adaptation_loss.backward()
                self.optimizer.step()

        for label in Estimation_Args.labels:
            mean_adaptation_losses[label] /= (Estimation_Args.num_substeps * num_batches)

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
