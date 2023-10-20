import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal


class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    adaptation_module_branch_hidden_dims = [256, 128]
    
    adaptation_labels = []
    adaptation_dims = []
    adaptation_weights = []

    use_decoder = False

class ActorCriticModel(nn.Module):
    def __init__(self,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 is_use_int=False,
                 output_tuple=False,
                 **kwargs):
        super().__init__()

        self.activation = get_activation(AC_Args.activation)
        self.is_use_int = is_use_int
        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs
        
        # Adaptation module
        self.adaptation_module = self.build_adaptation_layers()

        # Policy
        self.actor_body = self.build_policy_layers(num_actions)

        # Value function
        self.critic_body = self.build_critic_layers()
        self.critic_body_enrg = None
        self.output_tuple = output_tuple
        if is_use_int:
            self.critic_body_int = self.build_critic_layers()
        
        # Action noise
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        
    def build_adaptation_layers(self):
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(self.num_obs_history, AC_Args.adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(self.activation)
        for l in range(len(AC_Args.adaptation_module_branch_hidden_dims)):
            if l == len(AC_Args.adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l], self.num_privileged_obs))
            else:
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l],
                              AC_Args.adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(self.activation)
        adaptation_module = nn.Sequential(*adaptation_module_layers)
        return adaptation_module

    def build_policy_layers(self, num_actions):
        actor_layers = []
        actor_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs_history, AC_Args.actor_hidden_dims[0]))
        actor_layers.append(self.activation)
        for l in range(len(AC_Args.actor_hidden_dims)):
            if l == len(AC_Args.actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], AC_Args.actor_hidden_dims[l + 1]))
                actor_layers.append(self.activation)
        actor_body = nn.Sequential(*actor_layers)
        return actor_body

    def build_critic_layers(self):
        critic_layers = []
        critic_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs_history, AC_Args.critic_hidden_dims[0]))
        critic_layers.append(self.activation)
        for l in range(len(AC_Args.critic_hidden_dims)):
            if l == len(AC_Args.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], AC_Args.critic_hidden_dims[l + 1]))
                critic_layers.append(self.activation)
        critic_body = nn.Sequential(*critic_layers)
        return critic_body

    def update_distribution(self, observation_history):
        latent = self.adaptation_module(observation_history)
        mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def get_current_actions_log_prob(self, observation_history):
        latent = self.adaptation_module(observation_history)
        mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))
        dis = Normal(mean, mean * 0. + self.std)
        return dis.log_prob(dis.sample()).sum(dim=-1)

    def act(self, observation_history, **kwargs):
        self.update_distribution(observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs_history"], ob["privileged_obs"])

    def act_inference(self, ob, policy_info={}):
        return self.act_student(ob["obs_history"], policy_info=policy_info)

    def act_student(self, observation_history, policy_info={}):
        latent = self.adaptation_module(observation_history)
        actions_mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, observation_history, privileged_info, policy_info={}):
        actions_mean = self.actor_body(torch.cat((observation_history, privileged_info), dim=-1))
        policy_info["latents"] = privileged_info
        return actions_mean

    def evaluate(self, observation_history, privileged_observations, **kwargs):
        if self.is_use_int:
            ext_value = self.critic_body(torch.cat((observation_history, privileged_observations), dim=-1))
            int_value = self.critic_body_int(torch.cat((observation_history, privileged_observations), dim=-1))  
            if not self.critic_body_enrg is None:
                ext_value = ext_value + int_value
                int_value = self.critic_body_enrg(torch.cat((observation_history, privileged_observations), dim=-1))
            if self.output_tuple:
                return ext_value, int_value
            else:
                return ext_value + int_value
        else:
            if self.output_tuple:
                ext_value = self.critic_body(torch.cat((observation_history, privileged_observations), dim=-1))
                int_value = self.critic_body_int(torch.cat((observation_history, privileged_observations), dim=-1))  
                return ext_value, int_value
            else:
                value = self.critic_body(torch.cat((observation_history, privileged_observations), dim=-1))    
                return value

    def get_student_latent(self, observation_history):
        return self.adaptation_module(observation_history)
    

class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 exp,
                 ptrn_path=None,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        self.decoder = AC_Args.use_decoder
        self.exp = exp

        self.adaptation_labels = AC_Args.adaptation_labels
        self.adaptation_dims = AC_Args.adaptation_dims
        self.adaptation_weights = AC_Args.adaptation_weights

        if len(self.adaptation_weights) < len(self.adaptation_labels):
            # pad
            self.adaptation_weights += [1.0] * (len(self.adaptation_labels) - len(self.adaptation_weights))

        super().__init__()

        if 'const' in self.exp:
            self.a2c_models = nn.ModuleDict({'ext': ActorCriticModel(num_privileged_obs,
                 num_obs_history,
                 num_actions, True, True)})
        elif 'eipo' in self.exp:
            if not ptrn_path is None:
                self.a2c_models = nn.ModuleDict({'ext': ActorCriticModel(num_privileged_obs,
                    num_obs_history,
                    num_actions, True, False), 'mixed': ActorCriticModel(num_privileged_obs,
                    num_obs_history,
                    num_actions, True, True)})
                self.a2c_models['ext'].load_state_dict(torch.load(ptrn_path))
                if 'frz' in self.exp:
                    for para in self.a2c_models['ext'].parameters():
                        para.requires_grad = False
                self.a2c_models['mixed'].load_state_dict(torch.load(ptrn_path))
                self.a2c_models['mixed'].critic_body_enrg = \
                    self.a2c_models['mixed'].build_critic_layers()
            else:
                self.a2c_models = nn.ModuleDict({'ext': ActorCriticModel(num_privileged_obs,
                    num_obs_history,
                    num_actions), 'mixed': ActorCriticModel(num_privileged_obs,
                    num_obs_history,
                    num_actions, True, True)})

        else:
            self.a2c_models = nn.ModuleDict({'ext': ActorCriticModel(num_privileged_obs,
                 num_obs_history,
                 num_actions)})
        
        print(f"Adaptation Module: {self.a2c_models['ext'].adaptation_module}")
        print(f"Actor MLP: {self.a2c_models['ext'].actor_body}")
        print(f"Critic MLP: {self.a2c_models['ext'].critic_body}")

        self.distributions = {}
        # disable args validation for speedup
        Normal.set_default_validate_args = False


    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass
    
    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return {n: self.distributions[n].mean for n in self.distributions}       
     
    @property
    def action_std(self):
        return {n: self.distributions[n].stddev for n in self.distributions}
      
    @property
    def entropy(self):
        if 'eipo' in self.exp:
            mixed_entropy = self.distributions['mixed'].entropy()
            ext_entropy = self.distributions['ext'].entropy()
            return mixed_entropy.sum(dim=-1) + \
                ext_entropy.sum(dim=-1)
        else:
            return self.distributions['ext'].entropy().sum(dim=-1)
       
    def update_distribution(self, observation_history):
        if 'eipo' in self.exp:
            bsz = len(observation_history) // 2
            self.a2c_models['mixed'].update_distribution(observation_history[bsz:])
            self.distributions['mixed-ext'] = self.a2c_models['mixed'].distribution
            self.a2c_models['mixed'].update_distribution(observation_history[:bsz])
            self.distributions['mixed'] = self.a2c_models['mixed'].distribution

            self.a2c_models['ext'].update_distribution(observation_history[:bsz])
            self.distributions['ext-mixed'] = self.a2c_models['ext'].distribution
            self.a2c_models['ext'].update_distribution(observation_history[bsz:])
            self.distributions['ext'] = self.a2c_models['ext'].distribution
        else:
            self.a2c_models['ext'].update_distribution(observation_history)
            self.distributions['ext'] = self.a2c_models['ext'].distribution
        
    def act(self, observation_history, **kwargs):
        self.update_distribution(observation_history)
        sampled_actions = {n: self.distributions[n].sample() for n in self.distributions}
        return sampled_actions

    def get_actions_log_prob(self, actions):
        log_probs = {n: self.distributions[n].log_prob(actions[n]).sum(dim=-1) for n in self.distributions}
        return log_probs

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs_history"], ob["privileged_obs"])

    def act_inference(self, ob, policy_info={}):
        return self.act_student(ob["obs_history"], policy_info=policy_info)

    def act_student(self, observation_history, policy_infos={}):
        policy_infos["latents"] = {}
        actions_means = {}
        for n in self.a2c_models:
            policy_info = {}
            actions_means[n] = self.a2c_models[n].act_student(observation_history, policy_info)
            policy_infos["latents"][n] = policy_info["latents"]
        return actions_means

    def act_teacher(self, observation_history, privileged_info, policy_infos={}):
        policy_infos["latents"] = {}
        actions_means = {}
        for n in self.a2c_models:
            policy_info = {}
            actions_means[n] = self.a2c_models[n].act_teacher(observation_history, policy_info)
            policy_info["latents"][n] = privileged_info.detach().cpu().numpy() 
        return actions_means

    def evaluate(self, observation_history, privileged_observations, **kwargs):
        values = {}
        if 'const' in self.exp:
            values['ext'], values['int'] = self.a2c_models['ext'].evaluate(observation_history, 
                                            privileged_observations)
        elif 'eipo' in self.exp:
            bsz = len(observation_history) // 2
            values['eipo_ext'], values['int'] = self.a2c_models['mixed'].evaluate(observation_history[:bsz], 
                                            privileged_observations[:bsz])
            values['ext'] = self.a2c_models['ext'].evaluate(observation_history[bsz:], 
                                            privileged_observations[bsz:])
        else:
            values['ext'] = self.a2c_models['ext'].evaluate(observation_history, 
                                            privileged_observations)
        return values

    def get_student_latent(self, observation_history):
        return self.adaptation_module(observation_history)

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