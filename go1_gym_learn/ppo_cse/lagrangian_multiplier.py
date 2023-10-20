import torch
import bisect
import numpy as np
from params_proto import PrefixProto

class EIPO_Args(PrefixProto):
    alpha_lr = 0.01
    alpha_g_clip = 1.0
    alpha_max_clip = 10
    alpha_min_clip = -1
    alpha_bsz = 8

class LagrangianMultiplier:
    def __init__(self, min_vel, max_vel, num_vel_itvl, num_envs, 
                 num_transitions_per_env, alpha, lmbd, storage, device='cpu'):
        self.device = device
        self.lmbd = lmbd
        self.num_vel_itvl = num_vel_itvl
        if self.num_vel_itvl == 1:
            self.alpha = alpha
            self.alpha_record = {'mixed': self.alpha, 
                           'ext': self.alpha}
        else:
            self.intervals = torch.linspace(min_vel, max_vel, 
                                            steps=num_vel_itvl+1)[1:-1].tolist()
            self.alpha = torch.nn.Embedding(num_vel_itvl, 1)
            # self.alpha.weight.data.copy_(torch.linspace(0.2, 0.8, 
            #     num_vel_itvl).view(self.alpha.weight.data.shape))
            self.alpha.weight.data.fill_(alpha)
            self.alpha = self.alpha.to(self.device)
            self.alpha_record = {'mixed': torch.zeros(num_transitions_per_env, 
                                                num_envs, 1).to(self.device), 
                           'ext': torch.zeros(num_transitions_per_env, 
                                              num_envs, 1).to(self.device)}
            self.index_record = {'mixed': torch.zeros(num_transitions_per_env, 
                                                    num_envs).to(self.device), 
                            'ext': torch.zeros(num_transitions_per_env, 
                                                num_envs).to(self.device)}
        
        self.alpha_grad = 0.1
        self.step = 0
        self.num_transitions_per_env = num_transitions_per_env
        self.storage = storage

    @ torch.no_grad()
    def compute_alpha_values(self, vels):
        if self.num_vel_itvl > 1:
            for idx, v in enumerate(vels):
                policy = 'mixed' if idx < len(vels) // 2 else 'ext'
                idx = idx % (len(vels) // 2)
                alpha_idx = bisect.bisect_left(self.intervals, v.item())
                self.index_record[policy][self.step, idx] = alpha_idx
                self.alpha_record[policy][self.step, idx] = self.alpha.weight.data[alpha_idx]
            self.step = (self.step + 1) % self.num_transitions_per_env
        else:
            self.alpha_record = {'mixed': self.alpha, 
                           'ext': self.alpha}

    @ torch.no_grad()
    def compute_advantages(self):
        self.storage.advantages['eipo_mixed'] = (1 + self.alpha_record['mixed']) * \
            self.storage.advantages['eipo_ext'] + self.lmbd * self.storage.advantages['int']
        self.storage.advantages['U_max'] = self.storage.rewards['ext'] + \
            self.lmbd * self.storage.rewards['ext_int'] + \
                self.alpha_record['ext'] * self.storage.advantages['ext']
        self.storage.advantages['U_min'] = (1 + self.alpha_record['mixed']) * \
            self.storage.advantages['eipo_ext'] + \
                self.storage.advantages['int'] - (self.storage.rewards['eipo_ext'] + \
                    self.lmbd * self.storage.rewards['int'])

    def update_alpha_values(self):
        if self.num_vel_itvl > 1:
            self.alpha.zero_grad()
            cnt = {'mixed': torch.zeros(self.num_vel_itvl).to(self.device),
                   'ext': torch.zeros(self.num_vel_itvl).to(self.device)}
            for i in range(self.num_vel_itvl):
                cnt['mixed'][i] = (self.index_record['mixed'] == i).sum()
                cnt['ext'][i] = (self.index_record['ext'] == i).sum()
            mixed_alpha = self.alpha(self.index_record['mixed'].long())
            ext_alpha = self.alpha(self.index_record['ext'].long())
            loss = (mixed_alpha * torch.empty_like(\
                self.storage.advantages['eipo_ext']).copy_(\
                self.storage.advantages['eipo_ext']) / cnt['mixed'] - \
                ext_alpha * torch.empty_like(\
                self.storage.advantages['ext']).copy_(\
                self.storage.advantages['ext']) / cnt['ext']).sum()
            loss.backward()
            alpha_data = self.alpha.weight.data
            alpha_grad = self.alpha.weight.grad 
            # print(alpha_grad.max(), alpha_grad.min(), cnt['ext'].max(), cnt['ext'].min())
            alpha_data = alpha_data - EIPO_Args.alpha_lr * torch.clamp(alpha_grad, 
                                                                    min=-EIPO_Args.alpha_g_clip,
                                                                    max=EIPO_Args.alpha_g_clip)
            self.alpha.weight.data.copy_(torch.clamp(alpha_data, 
                                                min=EIPO_Args.alpha_min_clip, 
                                                max=EIPO_Args.alpha_max_clip))
            self.alpha_grad = alpha_grad.max().item()
        else:
            self.alpha_grad = (self.storage.advantages['eipo_ext'] - \
                self.storage.advantages['ext']).sum().item() * 1e2
            self.alpha = self.alpha - EIPO_Args.alpha_lr * np.clip(self.alpha_grad, 
                                                            -EIPO_Args.alpha_g_clip, 
                                                            EIPO_Args.alpha_g_clip)
            self.alpha = np.clip(self.alpha, EIPO_Args.alpha_min_clip, EIPO_Args.alpha_max_clip)
                        