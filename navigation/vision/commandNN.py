# %%
import torch
import torch.nn as nn
from torch import optim
# from tqdm import tqdm
import os
import numpy as np
import pickle as pkl
# import pandas as pd
from torch.utils.data import DataLoader, Dataset
# from torchinfo import summary
import cv2
from torchvision import transforms
from copy import deepcopy
from navigation.vision.utils.image_processing import process_batch, horiz_flip_img, process_image, normalize_image, augment_image, process_deployed
# from torchview import draw_graph
import gc
gc.collect()
torch.cuda.empty_cache()
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
# print(torch.cuda.memory_summary(device='cuda', abbreviated=False))
import shutil

"""
CNN that takes in images as input and control commands as output

"""


class CustomDataset(Dataset):
    def __init__(self, data, data_type,use_memory, num_classes):
        self.data = data
        self.use_memory = use_memory
        self.num_classes = num_classes
        self.data_type = data_type

    def __getitem__(self, index):
        row = self.data[index]
        commands = row[0]
        image = row[1]

        policy = np.array([0]*self.num_classes)
        policy[int(commands[-1])] = 1
        
        if self.data_type =='both':
            image_two = row[2]
            if self.use_memory:
                memory = row[3]
                return image, image_two, commands, policy, memory
            
            else:
                return image, image_two, commands, policy

        else:

            if self.use_memory:
                return image, commands, policy, memory
            else:
                return image, commands, policy

    def __len__(self):
        return len(self.data)


class CommandNet(nn.Module):
    def __init__(self, model_name, demo_folder, multi_command, data_type, predict_commands=None,
                 demo_type=None, deploy=False, scaled_commands=False, finetune=False,
                 use_memory=False, use_flipped=False, name_extra=''):
        super().__init__()
        self.model_name = model_name
        self.demo_type = demo_type
        self.demo_folder = demo_folder
        self.deploy = deploy
        self.use_flipped = use_flipped
        self.data_type = data_type

        # --------------------------------
        # Filepaths
        # --------------------------------
        command_type = 'multi_comm' if multi_command else 'single_comm'
        train_type = 'finetuned' if finetune else 'trained'

        self.root_dir = 'navigation/commandnet/runs/run_recent'
        self.model_path = f'{self.root_dir}/{command_type}/{self.demo_folder}/{self.model_name}'
        # seperate folder if using memory
        if use_memory:
            self.model_path += '_memory'
        if name_extra:
            self.model_path += f'_{name_extra}'

        self.model_save_path = f'{self.model_path}/{train_type}.pth'
        self.model_load_path = f'{self.model_path}/trained.pth'
        self.model_deploy_load_path = f'{self.model_path}/finetuned.pth' if finetune else f'{self.model_path}/trained.pth'
        if demo_type:
            self.demo_load_path = f'navigation/robot_demos/{self.demo_folder}' if demo_type == 'sim' else f'navigation/robot_demos/jenkins_experiment/{self.demo_folder}'
        self.rescale_path = f'{self.model_path}/rescales.pkl'
        self.config_path = f'{self.model_path}/config.pkl'

        # load config if deployed
        if self.deploy:
            with open(self.config_path, 'rb') as f:
                self.config = pkl.load(f)

        # make config if training
        else:
            self.config = {'use_memory': use_memory, 'multi_command': multi_command, 'scale_commands': scaled_commands,
                           'finetune': finetune, 'predict_commands': predict_commands}

            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)


            # update files
            if os.path.exists(self.model_path+f'/{self.model_name}_prev'):     # delete earliest run if exists
                print('Removing previous run')
                shutil.rmtree(self.model_path+f'/{self.model_name}_prev')
            
            if os.listdir(self.model_path):    # make it the previous run if it does exist
                print('Shuffling model dirs')
                os.rename(self.model_path, self.model_path+'_prev')
                os.makedirs(self.model_path)
                shutil.move(self.model_path+'_prev',self.model_path)



        # --------------------------------
        # DATA PARAMS
        # --------------------------------

        self.train_percent = 0.8
        self.val_percent = 0.2

        self.trainloader = None
        self.testloader = None
        self.testdemoloader = None

        self.batch_size = 256
        if self.deploy:
            self.batch_size = 1
        self.input_h = 224
        self.input_w = 224

        self.data_rescales = []
        if deploy and self.config['scale_commands']:
            with open(self.rescale_path, 'rb') as f:
                self.data_rescales = pkl.load(f)

        self.visualize_data = True
        self.visualize_model = False
        if self.deploy:
            self.visualize_data = False
            self.visualize_model = False

        # --------------------------------
        # DEFINE TRAINING PARAMS
        # --------------------------------
        self.loss_func = nn.MSELoss()
        self.policy_loss = nn.CrossEntropyLoss()

        if self.config['finetune']:
            self.lr = 8e-5
            self.epochs = 9
            self.batch_size = 64
            self.weight_decay = 1e-3
        else:
            self.lr = 2e-3
            self.main_lr = 2e-4
            self.epochs = 10
            self.weight_decay = 2e-3

        # --------------------------------
        # DEFINE NEURAL NETS
        # --------------------------------
        self.models = {}

        # define number of gaits we are using
        self.config['num_classes'] = 3
        print(f'Using {self.config["num_classes"]} classes.')

        # conv model

        if self.model_name == 'resnet18':
            from torchvision.models import resnet18, ResNet
            self.commandnet = resnet18(pretrained=not self.deploy)
            self.commandnet.fc = nn.Identity()
            self.fc_input_shape = 512
        elif self.model_name == 'resnet34':
            from torchvision.models import resnet34
            self.commandnet = resnet34(pretrained=not self.deploy)
            self.commandnet.fc = nn.Identity()
            self.fc_input_shape = 512
        elif self.model_name == 'resnet50':
            from torchvision.models import resnet50
            self.commandnet = resnet50(pretrained=not self.deploy)
            self.commandnet.fc = nn.Identity()
            self.fc_input_shape = 2048
        elif self.model_name == 'mnv3s':
            from torchvision.models import mobilenet_v3_small
            self.commandnet = mobilenet_v3_small(pretrained=not self.deploy)
            self.commandnet.classifier = nn.Identity()
            # print(self.commandnet)
            self.fc_input_shape = 576
        elif self.model_name == 'mnv3l':
            from torchvision.models import mobilenet_v3_large
            self.commandnet = mobilenet_v3_large(pretrained=not self.deploy)
            self.commandnet.classifier = nn.Identity()
            self.fc_input_shape = 960
        elif self.model_name == 'enb0':
            from torchvision.models import efficientnet_b0
            self.commandnet = efficientnet_b0(pretrained=not self.deploy)
            self.commandnet.classifier = nn.Identity()
            self.fc_input_shape = 1280
        elif self.model_name == 'regnet':
            from torchvision.models import regnet_y_400mf
            self.commandnet = regnet_y_400mf(pretrained=not self.deploy)
            self.commandnet.fc = nn.Identity()
            self.fc_input_shape = 440

        elif self.model_name == 'shuffle':
            from torchvision.models import shufflenet_v2_x1_0
            self.commandnet = shufflenet_v2_x1_0(pretrained=not self.deploy)
            self.commandnet.fc = nn.Identity()
            self.fc_input_shape = 440

        elif self.model_name == 'dino':
            if self.deploy:
                self.commandnet = torch.hub.load(
                    '/home/unitree/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_vits14', source='local')
            else:
                self.commandnet = torch.hub.load(
                    'facebookresearch/dinov2', 'dinov2_vits14')
            self.fc_input_shape = 384

        if self.data_type =='both':
            self.fc_input_shape*=2

            # print(self.commandnet)

         # memory
        if self.config['use_memory']:
            self.batch_memory = None
            self.memory_filled = False
            self.memory_size = 9
            # adapt fc_input_size based on memory size -> conv embedding * mem size + num comms * mem size
            self.mem_output_shape = 50
            self.fill_curr = 0

            self.models['memory'] = nn.Sequential(
                nn.Linear(self.fc_input_shape, 250),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(250, self.mem_output_shape)
            )
            self.fc_input_shape = self.mem_output_shape * \
                (self.memory_size+1)+ 3*(self.memory_size)
            self._reset_memory()

        # add all models
        self.models['main'] = self.commandnet

        if self.config['predict_commands']:
            self.gaits = ['x', 'yaw', 'policy']
        else:
            self.gaits = ['policy']

        if self.config['multi_command']:
            
            self.shared_out = 25

            # make final layers
            self.shared_layer = nn.Sequential(
                nn.Linear(self.fc_input_shape, 150),
                nn.GELU(),
                nn.Linear(150, self.shared_out),
                nn.GELU()
            )

            if self.config['predict_commands']:
                x = nn.Sequential(
                    self.shared_layer,
                    nn.Linear(self.shared_out,10),
                    nn.GELU(),
                    nn.Linear(10,1)
                )
                yaw = nn.Sequential(
                    self.shared_layer,
                    nn.Linear(self.shared_out,10),
                    nn.GELU(),
                    nn.Linear(10,1)

                )

                self.models['x'] = x
                self.models['yaw'] = yaw

            policy = nn.Sequential(
                    self.shared_layer,
                    nn.Dropout(),
                    nn.Linear(self.shared_out,25),
                    nn.GELU(),
                    nn.Linear(25,self.config['num_classes'])
                )
            self.models['policy'] = policy

        else:
            # make final layers
            self.command_layer = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.fc_input_shape, 3),
            )

            comm = deepcopy(self.command_layer)
            pol = deepcopy(self.command_layer)
            self.models['commands'] = comm
            self.models['policy'] = nn.Sequential(
                nn.Linear(self.fc_input_shape, 1),
                nn.Sigmoid()
            )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0

        for model_type in self.models:
            self.models[model_type].to(self.device)

        # freeze main NN
        if not self.deploy:
            for p in self.models['main'].parameters():
                p.requires_grad = False

            # for p in self.models['main'].layer3.parameters():
            #         p.requires_grad = True

            # for p in self.models['main'].layer4.parameters():
            #     p.requires_grad = True

            # unfreeze some layers if finetuning. TODO: freeze BatchNorms
            if self.config['finetune']:
                print('Finetuning!')
                self.load_trained()
                # for p in self.models['main'].parameters():
                #     p.requires_grad = True

            #     if self.mode
            #     for p in self.models['main'].layer2.parameters():
            #         p.requires_grad = True

            #    # print(self.models['main'].layer2)

                # for p in self.models['main'].layer3.parameters():
                #     p.requires_grad = True

                # for p in self.models['main'].layer4.parameters():
                #     p.requires_grad = True

                chil = list(self.models['main'].children())

                conv_chil = list(chil[0].children())

                # print(conv_chil[-3:])

                for c in conv_chil[-6:]:
                    for p in c.parameters():
                        p.requires_grad = True

            # defines full model and opts
            self.opts = []
            # create optimizer for model(s)
            for model_type in self.models:
                print(model_type)
                self.opts.append(optim.Adam(
                    params=self.models[model_type].parameters(), lr=self.lr, weight_decay=self.weight_decay))

            self.prepare_data()

        else:

            self.load_trained()

        # --------------------------------
        # DEFINE OUTPUTS
        # --------------------------------
        # summary(self, input_size=(self.batch_size, 3, self.input_h, self.input_w),depth = 4,device=self.device)
        # summary(self.models['fc'][0], input_size=(self.batch_size,self.fc_model_input),depth = 1,device=self.device)
        # # draw_graph(self, input_size=(self.batch_size,self.input_channels, self.input_h, self.input_w), device=self.device,
        #                 save_graph=True, filename='CommandNet_graph',directory='navigation/commandnet/graph/')

        print(f'PARAMETER SUMMARY \n \
                Batch size: {self.batch_size}\n \
                Model Name: {self.model_name}\n \
                Scaled: {self.config["scale_commands"]}\n \
                Multi: {self.config["multi_command"]}\n \
                LR: {self.lr}         \n \
                Epochs: {self.epochs} \n \
                Demo type: {self.demo_type}    \n \
                Demo folder: {self.demo_folder}\n \
                Num Classes: {self.config["num_classes"]}\n \
                Data type: {self.data_type}\n \
                -----------------------------------------')

    def forward(self, input, input_two=None, mem_comms=[]):
        with torch.set_grad_enabled(not self.deploy):

            # images through main layer
            output = self.models['main'](input)

            if input_two != None:
                output_two = self.models['main'](input_two)
                output = torch.concat((output,output_two), dim=1)

            if self.config['use_memory']:

                # compress conv embeddding
                memory_embedding = self.models['memory'](output)

                # if memory not full, add train commands or temp 0's
                if not self.deploy and not self.memory_filled:

                    # add commands to mem embedding
                    comb_embedding = torch.concat(
                        (memory_embedding, mem_comms), dim=-1).reshape(self.batch_size, 1, -1)

                    # add full embedding to memory
                    self._add_to_memory(comb_embedding)

                    if self.batch_memory.shape[1] == self.memory_size:
                        self.memory_filled = True

                    # return None
                    return None

                elif self.deploy and self.fill_curr != self.memory_size:

                    # add commands to mem embedding
                    if self.fill_curr == 0:
                        mem_comms = self.get_memory_zero_comm(
                            tens=True).float()
                        comb_embedding = torch.concat(
                            (memory_embedding, mem_comms), dim=-1).reshape(self.batch_size, 1, -1)

                        # add full embedding to memory
                        self._add_to_memory(comb_embedding)

                    # fill memory
                    self._fill_memory()

                # get flattened memory
                flat_memory = self.batch_memory.flatten(1)

                # add memory embedding
                flat_memory = torch.concat(
                    (flat_memory, memory_embedding), dim=1)
                output = flat_memory

            # pass through dropout
            # if not self.deploy: output = nn.Dropout(0.25)(output)

            # output through command regression layer
            if self.config['multi_command']:
                policy = self.models['policy'](output)
                if self.config['predict_commands']:
                    x = self.models['x'](output)
                    yaw = self.models['yaw'](output)
                    commands = [x, yaw]

                    return commands, policy

                else:
                    return [], policy

            else:
                commands = self.models['commands'](output)
                policy = self.models['policy'](output)
                # policy = commands[:,-1].reshape(-1,1)
                # commands = commands[:,:-1]

            # remove first element from memory and add comms
            if self.deploy and self.config['use_memory']:

                command_concat = torch.concat((x, y, yaw), dim=1)
                _, policy = torch.max(policy, 1)
                policy = policy.reshape(-1, 1)
                # add comms to flat memory embedding
                comb_embedding = torch.concat(
                    (memory_embedding, command_concat, policy), dim=1).reshape(self.batch_size, 1, -1)

                self._add_to_memory(comb_embedding)

            return commands, policy

    def train_model(self):
        from tqdm import tqdm
        trainloader = self.trainloader
        testloader = self.testloader

        for model_type in self.models:
            self.models[model_type].train()

        loss_log = {g: [] for g in self.gaits}
        loss_log['total'] = []

        test_loss = {g: [] for g in self.gaits}
        test_loss['total'] = []

        str_log = {g: [0, 0] for g in self.gaits}
        str_log['total'] = [0, 0]

        print('START TRAINING')
        for epoch in range(self.epochs):
            torch.cuda.empty_cache()

            total_train_loss = {g: [] for g in self.gaits}
            total_train_loss['total'] = []

            for idx, info in tqdm(enumerate(trainloader)):

                image = info[0]
                image = image.to(self.device)
                image  = image.float()

                image_two = None

                if self.data_type =='both':
                    image_two = info[1]
                    image_two = image_two.to(self.device)
                    image_two  = image_two.float()

                    command_targets = info[2]
                    command_targets = command_targets.to(self.device)
                    command_targets  = command_targets.float()

                    policy_targets = info[3]
                    policy_targets = policy_targets.to(self.device)
                    policy_targets  = policy_targets.float()

                    if self.config['use_memory']:
                        memory = info[4]
                        

                else:
                    command_targets = info[1]
                    command_targets = command_targets.to(self.device)
                    command_targets  = command_targets.float()

                    policy_targets = info[2]
                    policy_targets = policy_targets.to(self.device)
                    policy_targets  = policy_targets.float()

                    if self.config['use_memory']:
                        memory = info[3]

                if self.config['use_memory']:

                    # reset batch mem
                    self._reset_memory()

                    # fill memory
                    for i in range(self.memory_size):
                        mem_comms, mem_image = memory[i]
                        mem_comms = mem_comms.to(self.device)
                        mem_comms = mem_comms.float()
                        mem_image = mem_image.to(self.device)
                        mem_image = mem_image.float()
                        self.forward(mem_image, mem_comms)

                    if not self.memory_filled:

                        self._fill_memory()

                    assert self.memory_filled == True, 'Memory not filled'

                for opt in self.opts:
                    opt.zero_grad(set_to_none=True)

                commands, policy = self.forward(image, image_two)

                if self.config['multi_command']:

                    ind_loss = [self.loss_func(
                        commands[i], command_targets[:, i].reshape(-1, 1)) for i in range(len(commands))]

                    policy_loss = self.policy_loss(policy, policy_targets)
                    ind_loss.append(policy_loss)

                    loss = sum(ind_loss)

                else:
                    ind_loss = [self.loss_func(
                        commands[:, i], command_targets[:, i]) for i in range(commands.shape[1])]

                    policy_loss = self.policy_loss(policy, policy_targets)
                    ind_loss.append(policy_loss)

                    comm_loss = self.loss_func(commands, command_targets)

                    loss = comm_loss+policy_loss

                loss.backward()
                for opt in self.opts:
                    opt.step()

                for ind, g in enumerate(self.gaits):
                    total_train_loss[g].append(ind_loss[ind].item())

                total_train_loss['total'].append(loss.item())

                del image, command_targets, policy_targets, ind_loss, loss

            for l in total_train_loss:
                loss_log[l].append(np.mean(total_train_loss[l]))

            val_loss = self.evaluate(testloader)

            for l in val_loss:
                test_loss[l].append(np.mean(val_loss[l]))
                str_log[l][1] = val_loss[l][-1]-val_loss[l][0]
                str_log[l][0] = val_loss[l][-1]

            # if epoch % 5==0:
            #     self._save_model_plots(loss_log, test_loss)
            for model_type in self.models:
                self.models[model_type].train()

            print("\n [INFO] EPOCH: {}/{}".format(epoch + 1, self.epochs))
            print(
                f"Train loss: {round(loss_log['total'][-1],4)} Val loss: {round(test_loss['total'][-1],4)}\nTest Commands: {[round(test_loss[l][-1],4) for l in str_log]}")
           # print(f"Train loss: {round(loss_log['total'][-1],4)} Val loss: {round(test_loss['total'][-1],4)}")

        self._save_all_models_()
        self._save_model_plots(loss_log, test_loss)

        if self.visualize_model:
            self._visualize_model_()

        print('Checking if all models in train mode...')
        for model_type in self.models:
            if self.models[model_type].training:
                self.models[model_type].eval()

            print(self.models[model_type].training)

    def evaluate(self, data):
        import time
        for model_type in self.models:
            self.models[model_type].eval()

        with torch.no_grad():

            eval_log = {g: [] for g in self.gaits}
            eval_log['total'] = []

            for iter, info in enumerate(data):

                image = info[0]
                image = image.to(self.device)
                image  = image.float()

                image_two = None

                if self.data_type =='both':
                    image_two = info[1]
                    image_two = image_two.to(self.device)
                    image_two  = image_two.float()

                    command_targets = info[2]
                    command_targets = command_targets.to(self.device)
                    command_targets  = command_targets.float()

                    policy_targets = info[3]
                    policy_targets = policy_targets.to(self.device)
                    policy_targets  = policy_targets.float()

                    if self.config['use_memory']:
                        memory = info[4]
                        

                else:
                    command_targets = info[1]
                    command_targets = command_targets.to(self.device)
                    command_targets  = command_targets.float()

                    policy_targets = info[2]
                    policy_targets = policy_targets.to(self.device)
                    policy_targets  = policy_targets.float()

                    if self.config['use_memory']:
                        memory = info[3]

                if self.config['use_memory']:

                    # reset batch mem
                    self._reset_memory()

                    # fill memory
                    for i in range(self.memory_size):
                        mem_comms, mem_image = memory[i]
                        mem_comms = mem_comms.to(self.device)
                        mem_comms = mem_comms.float()
                        mem_image = mem_image.to(self.device)
                        mem_image = mem_image.float()
                        self.forward(mem_image, mem_comms)

                    if not self.memory_filled:

                        self._fill_memory()

                    assert self.memory_filled == True, 'Memory not filled'

                commands, policy = self.forward(image, image_two)

                if self.config['multi_command']:
                    ind_loss = [self.loss_func(
                        commands[i], command_targets[:, i].reshape(-1, 1)) for i in range(len(commands))]

                    policy_loss = self.policy_loss(policy, policy_targets)
                    ind_loss.append(policy_loss)

                    loss = sum(ind_loss)

                else:
                    ind_loss = [self.loss_func(
                        commands[:, i], command_targets[:, i]) for i in range(commands.shape[1])]

                    policy_loss = self.policy_loss(policy, policy_targets)
                    ind_loss.append(policy_loss)

                    comm_loss = self.loss_func(commands, command_targets)

                    loss = comm_loss+policy_loss

                for ind, g in enumerate(self.gaits):
                    eval_log[g].append(ind_loss[ind].item())

                eval_log['total'].append(loss.item())

                del commands, loss, ind_loss

        return eval_log

    def evaluate_full_demo(self):
        from tqdm import tqdm
        data = self.testdemoloader
        self.batch_size = data.batch_size
        self.deploy = True
        for model_type in self.models:
            self.models[model_type].eval()

        # reset batch mem
        if self.config['use_memory']:
            self._reset_memory()

        print('Evaluating full demo')
        with torch.no_grad():

            preds = {g: [] for g in self.gaits}
            labels = {g: [] for g in self.gaits}

            for iter,info in tqdm(enumerate(data)):

                image = info[0]
                image = image.to(self.device)
                image  = image.float()

                image_two = None

                if self.data_type =='both':
                    image_two = info[1]
                    image_two = image_two.to(self.device)
                    image_two  = image_two.float()

                    command_targets = info[2]
                    command_targets = command_targets.to(self.device)
                    command_targets  = command_targets.float()

                    policy_targets = info[3]
                    policy_targets = policy_targets.to(self.device)
                    policy_targets  = policy_targets.float()

                        

                else:
                    command_targets = info[1]
                    command_targets = command_targets.to(self.device)
                    command_targets  = command_targets.float()

                    policy_targets = info[2]
                    policy_targets = policy_targets.to(self.device)
                    policy_targets  = policy_targets.float()

                command_targets = command_targets.reshape(-1).detach().tolist()

                # if memory not filled yet then fill and continue
                if self.config['use_memory'] and not self.memory_filled:

                    self.forward(image)
                    # self._fill_memory()
                    continue
                else:
                    commands, policy = self.forward(image, image_two)

                _, policy = torch.max(policy, 1)

                # commands,policy = self._data_rescale(commands, policy)
                if self.config['multi_command']:
                    if self.config['predict_commands']:
                        for i in range(len(commands)):
                            commands[i] = commands[i].detach().item()
                    else:
                        commands = []
                else:
                    commands = commands.reshape(-1).detach().tolist()
                policy = policy.detach().item()

                commands = list(commands)
                commands.append(policy)

                if self.config['predict_commands']:
                    for ind, g in enumerate(self.gaits):
                        preds[g].append(commands[ind])
                        labels[g].append(command_targets[ind])
                else:
                    preds['policy'].append(commands[0])
                    labels['policy'].append(command_targets[-1])

        return preds, labels

    def prepare_data(self):
        from tqdm import tqdm
        from pathlib import Path
        import pickle
        import gzip
        print('Preparing Data...')

        data = {'Commands':[], 'Commands_test':[]}

        if self.data_type in {'rgb', 'both'}:
            data['Image1st']=[]
            data['Image1st_test']=[]
            data['Image1st_aug']=[]
            data['Image1st_processed']=[]
        
        if self.data_type in {'depth', 'both'}:
            data['DepthImg']=[]
            data['DepthImg_test']=[]
            data['DepthImg_processed']=[]


        num_runs = len([p for p in Path(self.demo_load_path).glob('*')])
        print('Demo path:', self.demo_load_path)
        print('Num runs', num_runs)

        print('Extracted demos...')

        # ------------------------
        # DATA EXTRACTION
        # ------------------------
        for i in tqdm(range(num_runs)):
            log_files = sorted([str(p) for p in Path(self.demo_load_path+f'/run{i+1}').glob(
                "*.pkl")], key=lambda x: int(x.split('/')[-1].split('.')[0][3:]))

            if i == 0:
                print(log_files)

            for log in log_files:
                with gzip.open(log, 'rb') as f:
                    p = pickle.Unpickler(f)
                    demo = p.load()
                    for k in demo:
                        
                        if k in data:
                            if i ==0:
                                data[k+'_test'] +=demo[k]
                            else:
                                data[k] += demo[k]


        print('Original train & test length:', len(data['Commands']), len(data['Commands_test']))
        # ------------------------
        # Horiz Flipping
        # ------------------------
        if self.use_flipped:
            print('Flipping images...')

            for i in tqdm(range(len(data['Commands']))):

                if abs(data['Commands'][i][2]) >= 0.5:
                    c = data['Commands'][i]
                    flip_y = -1*c[1]
                    flip_yaw = -1*c[2]
                    data['Commands'].append([c[0], flip_y, flip_yaw, c[3]])
                    
                    if self.data_type in {'rgb', 'both'}:
                        horiz_rgb = horiz_flip_img(data['Image1st'][i])
                        data['Image1st'].append(horiz_rgb)
                    elif self.data_type in {'rgb', 'both'}:
                        horiz_depth = horiz_flip_img(data['DepthImg'][i])
                        data['DepthImg'].append(horiz_depth)

            print('Flipped train & test length:', len(data['Commands']))

        # remove y vals
        print('Removing y_cmd vals')
        for i in range(len(data['Commands'])):
            data['Commands'][i].pop(1)

        for i in range(len(data['Commands_test'])):
            data['Commands_test'][i].pop(1)
        
        data['Commands'] = np.array(data['Commands'])
        data['Commands_test'] = np.array(data['Commands_test'])
        if self.data_type in {'rgb','both'}:
            data['Image1st'] = np.array(data['Image1st'])
        if self.data_type in {'depth','both'}:
            data['DepthImg'] = np.array(data['DepthImg'])

        print('Removed y_cmd shapes:',data['Commands'].shape, data['Commands_test'].shape)

        original_comms = deepcopy(data['Commands'])
        original_test_comms = deepcopy(data['Commands_test'])

        



        # ------------------------
        # DATA SCALING
        # ------------------------

        if self.config['scale_commands']:
            print('Scaling commands...')

            

            all_comms = np.concatenate((data['Commands'], data['Commands_test']), axis=0)
            print('all combs:', all_comms.shape)

            mini = all_comms.min(axis=0)
            ptp = all_comms.ptp(axis=0)

            mini = mini[:-1]
            ptp = ptp[:-1]
            self.data_rescales = [mini, ptp]
            print('rescales:', self.data_rescales)

            for i in range(len(self.data_rescales[0])):
                print('before',max(data['Commands'][:, i].flatten()), min(data['Commands'][:, i].flatten()))
                print('before',max(data['Commands_test'][:, i].flatten()), min(data['Commands_test'][:, i].flatten()))

                data['Commands'][:, i] = (data['Commands'][:, i]-mini[i])/ptp[i]
                data['Commands_test'][:, i] = (data['Commands_test'][:, i]-mini[i])/ptp[i] 

                if self.data_rescales[1][i] == 0.0:
                    data['Commands'][:, i] = original_comms[:, i]
                    data['Commands_test'][:, i] = original_test_comms[:, i]

                print('after',max(data['Commands'][:, i].flatten()), min(data['Commands'][:, i].flatten()))
                print('after',max(data['Commands_test'][:, i].flatten()), min(data['Commands_test'][:, i].flatten()))
                print('NaN search:', np.argwhere(
                    np.isnan(data['Commands'][:, i])), np.argwhere(np.isnan(data['Commands_test'][:, i])))

        print('Original dataset')
        for k in data:
            if type(data[k])!=list:
                print(k, data[k].shape)

        policy_counts = {p:0 for p in range(self.config['num_classes'])}

        for i in range(len(data['Commands'])):
            policy_counts[int(data['Commands'][i][-1])] += 1

        print('Train policy counts:', policy_counts)

        # ------------------------
        # IMAGE PROCESSING
        # ------------------------
        print('Processing train images...')
        
        if self.data_type in {'rgb', 'both'}:
            for i in tqdm(range(len(data['Image1st']))):
                processed_img = process_image(data['Image1st'][i])
                data['Image1st_processed'].append(processed_img)

            del data['Image1st']

        if self.data_type in {'depth','both'}:
            for i in tqdm(range(len(data['DepthImg']))):
                processed_img = process_image(data['DepthImg'][i])
                data['DepthImg_processed'].append(processed_img)

            del data['DepthImg']

        # self.plot_data(deepcopy(processed_train[0]), 'processed train', pil=True)

        print('Creating test data...')
        test_unbatched =[]
        for i in tqdm(range(len(data['Commands_test']))):
            batch = []
            batch.append(data['Commands_test'][i])

            if self.data_type in {'rgb','both'}:
                proc_img = process_image(data['Image1st_test'][i])
                norm_img = normalize_image(proc_img)
                batch.append(norm_img)

            if self.data_type in {'depth', 'both'}:
                proc_img = process_image(data['DepthImg_test'][i])
                norm_img = normalize_image(proc_img)
                batch.append(norm_img)

            test_unbatched.append(batch)

        
        del data['Commands_test']
        if 'Image1st_test' in data:
            del data['Image1st_test']
        if 'DepthImg_test' in data:
            del data['DepthImg_test']



        # ------------------------
        # IMAGE AUGMENTATION & Normalization
        # ------------------------
        print('Augmenting and Normalizing train images...')
        for i in tqdm(range(len(data['Commands']))):
            batch =[]

            comm = data['Commands'][i]
            
            
            if self.data_type in {'depth', 'both'}:
                norm_img = normalize_image(data['DepthImg_processed'][i])
                data['DepthImg_processed'][i] = norm_img

            if self.data_type in {'rgb','both'}:
                augmented_imgs = augment_image(data['Image1st_processed'][i])

                for im in augmented_imgs:
                    norm_img = normalize_image(im)
                    batch.append([comm, norm_img])

                data['Image1st_aug'].append(batch)

        
        if 'Image1st_processed' in data:
            del data['Image1st_processed']

        print(
            f'Augmented lengths: {len(data["Image1st_aug"])}')

        # self.plot_data(deepcopy(augmented_train[0][0]), 'aug train', pil=True)

        # ------------------------
        # Add memory to data
        # ------------------------
        if self.config['use_memory']:
            print('Adding memory info to train data...')
            data['Image1st_mem']=[]

            for i in tqdm(range(len(augmented_train[0]))):

                for j in range(1, len(augmented_train)):

                    memory_info = []

                    for k in range(j-self.memory_size, j):
                        if k > 0:
                            memory_info.append(
                                [augmented_train[k][i][0], augmented_train[k][i][1]])
                        elif k == 0:
                            zero_comm = self.get_memory_zero_comm(tens=False)
                            memory_info.append(
                                [zero_comm, augmented_train[k][i][1]])

                    while len(memory_info) != self.memory_size:
                        cop = deepcopy(memory_info[-1])
                        memory_info.append(cop)

                    data_point = [augmented_train[j][i][0],
                                  augmented_train[j][i][1], memory_info]
                    train_memory.append(data_point)

            del augmented_train

            print('Adding memory info to test data...')
            test_memory = []
            for i in tqdm(range(len(augmented_test[0]))):

                for j in range(1, len(augmented_test)):

                    memory_info = []

                    for k in range(j-self.memory_size, j):
                        if k > 0:
                            memory_info.append(
                                [augmented_test[k][i][0], augmented_test[k][i][1]])
                        elif k == 0:
                            zero_comm = self.get_memory_zero_comm(tens=False)
                            memory_info.append(
                                [zero_comm, augmented_test[k][i][1]])

                    while len(memory_info) != self.memory_size:
                        cop = deepcopy(memory_info[-1])
                        memory_info.append(cop)

                    data_point = [augmented_test[j][i][0],
                                  augmented_test[j][i][1], memory_info]
                    test_memory.append(data_point)

            del augmented_test

        else:
            print('Adding info to train data...')
            train_memory = []
            for i in tqdm(range(len(data['Commands']))):

                for j in range(len(data['Image1st_aug'][0])):
                    data_point = []


                    if self.data_type in {'rgb', 'both'}:
                        data_point .append(data['Image1st_aug'][i][j][0])
                        data_point.append(data['Image1st_aug'][i][j][1])

                    if self.data_type in {'depth','both'}:
                        if not data_point:
                            data_point = [data['Commands'][i]]
                        data_point.append(data['DepthImg_processed'][i])

                    if i==0 and j==0:
                        print('Data point:', data_point)

                    train_memory.append(data_point)

        train = train_memory

        # if self.visualize_data:
        #     self._visualize_robot_dataset_(processed_train,train_memory)

        torch.cuda.empty_cache()

        print(
            f'Final shapes: train = {len(train)} test unbatched = {len(test_unbatched)}')

        np.random.shuffle(train)

        print('Image shape:', train[0][1].shape)
        if self.config['use_memory']:
            print('Memory shape:', len(train[0][2]))

        # train data in batches of 10
        train_data = CustomDataset(
            train, use_memory=self.config['use_memory'], num_classes=self.config['num_classes'], data_type=self.data_type)
        trainloader = DataLoader(train_data, batch_size=self.batch_size,
                                 shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

        # final test data in batches of 1...memory will have to fill
        testdemo_data = CustomDataset(
            test_unbatched, use_memory=False, num_classes=self.config['num_classes'], data_type=self.data_type)
        testdemoloader = DataLoader(
            testdemo_data, batch_size=1, num_workers=2, pin_memory=True)

        self.trainloader = trainloader
        self.testloader = testdemoloader
        self.testdemoloader = testdemoloader

    def _data_rescale(self, commands, policy):

        # prepare commands
        if self.config['multi_command']:
            for i in range(len(commands)):
                commands[i] = commands[i].detach().item()
        else:
            commands = commands.reshape(-1).detach().tolist()

        _, policy = torch.max(policy, 1)
        policy = policy.detach().item()

        if self.config['scale_commands']:
            # commands.append(policy)

            commands = np.array(commands)
            commands = np.clip(commands, 0.0, 1.0)
            commands = (commands)*self.data_rescales[1]+self.data_rescales[0]

            commands = list(commands)

        return commands, policy

    def _visualize_model_(self):
        import matplotlib.pyplot as plt
        import pandas as pd
        model = self.models['conv'][0]
        model_weights = []
        conv_layers = []

        model_children = list(model.modules())
        # counter to keep count of the conv layers
        counter = 0
        # append all the conv layers and their respective weights to the list
        for i in range(len(model_children)):
            # print(model_children[i])
            if type(model_children[i]) == nn.Conv2d:
                counter += 1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == nn.Conv2d:
                            counter += 1
                            model_weights.append(child.weight)
                            conv_layers.append(child)
        print(f"Total convolutional layers: {counter}")

        # visualize the first conv layer filters
        plt.figure(figsize=(20, 17))
        for i, filter in enumerate(model_weights[0]):
            # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
            plt.subplot(10, 10, i+1)
            plt.imshow(filter[0, :, :].detach().cpu(), cmap='gray')
            plt.axis('off')
        plt.show()

        plt.figure(figsize=(20, 17))
        for i, filter in enumerate(model_weights[1]):
            # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
            plt.subplot(10, 10, i+1)
            plt.imshow(filter[0, :, :].detach().cpu(), cmap='gray')
            plt.axis('off')
        plt.show()

        # example image
        df = pd.read_pickle('navigation/robot_demos/demosDF.pkl')
        extracted_data = []
        for r in range(1):
            row = df.iloc[r].to_numpy()
            commands = row[0]
            first_person = row[1]
            third_person = row[2]

            for e in range(30, 31):
                imgs, _ = process_image(
                    first=first_person[e], third=third_person[e], image_mode=self.image_mode)
                extracted_data.append([commands[e], imgs])

        # x= self.models['x'](output)True
        # y=self.models['y'] (output)
        # yaw=self.models['yaw'] (output)
        # leg=self.models['leg'] (output)
        # step=self.models['step'] (output)
        # commands=torch.concat((x,y,yaw,leg,step),dim=1)
        img = img_to_tensor_norm(
            extracted_data[0][1], self.data_mean, self.data_std)
        print(img.shape)
        assert img.shape == (self.image_channels, self.input_h, self.input_w)
        img = img[None, ...].cuda().float()

        # pass the image through all the layers
        results = [conv_layers[0](img)]
        for i in range(1, len(conv_layers)):
            # pass the result from the last layer to the next layer
            results.append(conv_layers[i](results[-1]))
        # make a copy of the `results`
        outputs = results

        # visualize 64 features from each layer
        # (although there are more feature maps in the upper layers)
        for num_layer in range(len(outputs)):
            plt.figure(figsize=(30, 30))
            layer_viz = outputs[num_layer][0, :, :, :]
            layer_viz = layer_viz.data
            print(layer_viz.size())
            for i, filter in enumerate(layer_viz):
                if i == 64:  # we will visualize only 8x8 blocks from each layer
                    break
                plt.subplot(8, 8, i + 1)
                plt.imshow(filter.detach().cpu(), cmap='gray')
                plt.axis("off")
            plt.show()
            plt.close()

    def _visualize_dataset_(self, extracted, df):
        import matplotlib.pyplot as plt
        to_pil = transforms.Compose([
            transforms.ToPILImage()]
        )

        # INPUTS

        # extract 36 random images
        # imgs=extracted[:,1]

        rand = []
        for i in range(len(extracted)):
            if np.random.choice(2, 1) == 1:
                rand.append(to_pil(extracted[i][1]))
            if len(rand) == 36:
                break

        # rand=np.array(rand)

        print('Num of visualized inputs:', len(rand))

        if self.image_mode == 'comb':
            plt.figure(figsize=(20, 17))
            plt.title('Input examples')
            for i in range(len(rand)//2):
                # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
                plt.subplot(6, 6, 2*i+1)
                plt.imshow(rand[i][:, :, :3])
                # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
                plt.subplot(6, 6, 2*i+2)
                plt.imshow(rand[i][:, :, 3:])
                plt.axis('off')
            plt.show()

            plt.figure(figsize=(20, 17))
            plt.title('Input examples histogram')
            for channel_id, color in enumerate(('red', 'green', 'blue')):
                plt.subplot(1, 2, 1)
                histogram, bin_edges = np.histogram(
                    np.squeeze(rand[:, :, :, channel_id]), bins=50, range=(0, 1)
                )
                plt.plot(bin_edges[0:-1], histogram, color=color, alpha=0.4)

                # third person
                plt.subplot(1, 2, 2)
                histogram, bin_edges = np.histogram(
                    np.squeeze(rand[:, :, :, 3+channel_id]), bins=50, range=(0, 1)
                )
                plt.plot(bin_edges[0:-1], histogram, color=color, alpha=0.4)
            plt.show()

        else:

            plt.figure(figsize=(20, 17))
            plt.title('Input examples')
            for i in range(len(rand)):
                # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
                plt.subplot(6, 6, i+1)
                plt.imshow(rand[i])
                plt.axis('off')
            plt.show()

            plt.figure(figsize=(20, 17))
            plt.title('Input examples histogram')
            for channel_id, color in enumerate(('red', 'green', 'blue')):
                histogram, bin_edges = np.histogram(
                    np.squeeze(rand[:, :, :, channel_id]), bins=50, range=(0, 1)
                )
                plt.plot(bin_edges[0:-1], histogram, color=color, alpha=0.4)

            plt.show()

        # OUTPUTS
        dems = np.array(df['Commands'])
        df_commands = []

        for dem in dems:
            for move in dem:
                df_commands.append(move)

        df_commands = np.array(df_commands)

        x_vels_d = df_commands[:, 0]
        y_vels_d = df_commands[:, 1]
        yaws_d = df_commands[:, 2]
        height_d = df_commands[:, 3]
        freq_d = df_commands[:, 4]

        x_vels = []
        y_vels = []
        yaws = []
        height = []
        freq = []

        for x, y, yaw, h, f in extracted[:, 0]:
            x_vels.append(x)
            y_vels.append(y)
            yaws.append(yaw)
            height.append(h)
            freq.append(f)

        fig, axes = plt.subplots(2, 5)
        axes[0, 0].hist(x_vels_d)
        axes[0, 0].set_title('x_vels')
        axes[0, 1].hist(y_vels_d)
        axes[0, 1].set_title('y_vels')
        axes[0, 2].hist(yaws_d)
        axes[0, 2].set_title('yaws')
        axes[0, 3].hist(height_d)
        axes[0, 3].set_title('foot height')
        axes[0, 4].hist(freq_d)
        axes[0, 4].set_title('step freq')

        axes[1, 0].hist(x_vels)
        axes[1, 0].set_title('x_vels')
        axes[1, 1].hist(y_vels)
        axes[1, 1].set_title('y_vels')
        axes[1, 2].hist(yaws)
        axes[1, 2].set_title('yaws')
        axes[1, 3].hist(height)
        axes[1, 3].set_title('foot height')
        axes[1, 4].hist(freq)
        axes[1, 4].set_title('step freq')
        fig.tight_layout()
        plt.show()

    def _visualize_robot_dataset_(self, original, extracted):
        import matplotlib.pyplot as plt
        to_pil = transforms.Compose([
            transforms.ToPILImage()]
        )

        # INPUTS

        # extract 36 random images
        # imgs=extracted[:,1]

        # rand = []
        # for i in range(len(extracted)):
        #     if np.random.choice(2, 1) == 1:
        #         rand.append(to_pil(extracted[i][1]))
        #     if len(rand) == 36:
        #         break

        # rand = np.array(rand, dtype=object)

        # print('Num of visualized inputs:', len(rand))

        # plt.figure(figsize=(20, 17))
        # plt.title('Input examples')
        # for i in range(len(rand)):
        #     plt.subplot(6, 6, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        #     plt.imshow(rand[i])
        #     plt.axis('off')
        # plt.show()

        # plt.figure(figsize=(20, 17))
        # plt.title('Input examples histogram')
        # for channel_id, color in enumerate(('red','green','blue')):
        #     histogram, bin_edges = np.histogram(
        #         np.squeeze(rand[:,:, :, channel_id]), bins=50, range=(0,1)
        #     )
        #     plt.plot(bin_edges[0:-1], histogram, color=color,alpha=0.4)

        # plt.show()

        # OUTPUTS
        x_vels = []
        y_vels = []

        policy = []

       # print(extracted, original)
        for comm, _ in extracted:
            x, y, yaw, p = comm
            x_vels.append(x)
            y_vels.append(y)
            yaws.append(yaw)
            policy.append(p)

        origx = []
        origy = []
        origyaw = []
        origp = []

        for comm, _ in original:
            x, y, yaw, p = comm
            origx.append(x)
            origy.append(y)
            origyaw.append(yaw)
            origp.append(p)

        fig, axes = plt.subplots(2, 4)

        axes[0, 0].hist(x_vels)
        axes[0, 0].set_title('x_vels')
        axes[1, 0].hist(origx)
        axes[1, 0].set_title(' orig x_vels')

        axes[0, 1].hist(y_vels)
        axes[0, 1].set_title('y_vels')
        axes[1, 1].hist(origy)
        axes[1, 1].set_title('orig y_vels')

        axes[0, 2].hist(yaws)
        axes[0, 2].set_title('yaws')
        axes[1, 2].hist(origyaw)
        axes[1, 2].set_title('orig yaws')

        axes[0, 3].hist(policy)
        axes[0, 3].set_title('policy')
        axes[1, 3].hist(origp)
        axes[1, 3].set_title('orig policy')

        fig.tight_layout()
        plt.show()

    def _save_all_models_(self):
        import pickle as pkl
        from datetime import datetime
        
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print("date and time =", dt_string)

        with open(f'{self.model_path}/date_time.txt', 'w') as f:
            f.write(dt_string)

        all_models = {}
        for model_type in self.models:
            print(f'Saving model type: {model_type}')
            all_models[model_type] = self.models[model_type].state_dict()

        torch.save(all_models, self.model_save_path)
        print('Saved to:', self.model_save_path)

        if self.config['scale_commands']:
            print('saving rescale data...')
            with open(self.rescale_path, 'wb') as f:
                pkl.dump(self.data_rescales, f)

        print('Saving config parameters')
        with open(self.config_path, 'wb') as f:
            pkl.dump(self.config, f)

    def _save_model_plots(self, loss_log, test_loss):
        import matplotlib.pyplot as plt

        log_keys = ['total']

        log_keys += self.gaits

        num_rows = int(np.ceil(len(log_keys)/2))

        fig, ax = plt.subplots(num_rows+1, 2, sharey=False)
        fig.suptitle(
            f'Final train loss: {round(loss_log["total"][-1],4)}    Final eval loss: {round(test_loss["total"][-1],4)} \
                \n  LR: {self.lr}   Batch size: {self.batch_size}')
        count = 0
        for i in range(num_rows):
            for j in range(2):
                if i*2+j+1 > len(log_keys):
                    continue

                ax[i, j].plot(range(len(loss_log['total'])),
                              loss_log[log_keys[count]], label='train')
                ax[i, j].plot(range(len(test_loss['total'])),
                              test_loss[log_keys[count]], label='test')
                ax[i, j].set_title(log_keys[count])
                count += 1
        handles, labels = ax[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plt.ylim(0.0, 0.6)
        fig.tight_layout()
        extra = 'finetuned' if self.config['finetune'] else ''
        plt.savefig(f'{self.model_path}/losses_{extra}.png')
        # plt.show()

        eval_preds, eval_labels = self.evaluate_full_demo()
        eval_keys = log_keys[1:]
        fig, ax = plt.subplots(2, 2, sharey=False, figsize=(15, 15))
        fig.suptitle(f'Full test demo')
        count = 0
        for i in range(num_rows):
            for j in range(2):
                if i*2+j+1 > len(eval_keys):
                    continue

                ax[i, j].plot(range(len(eval_preds[eval_keys[count]])),
                              eval_preds[eval_keys[count]], label='Pred')
                ax[i, j].plot(range(len(eval_labels[eval_keys[count]])),
                              eval_labels[eval_keys[count]], label='Truth')
                ax[i, j].set_title(eval_keys[count])
                count += 1
            handles, labels = ax[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right')
            fig.tight_layout()
            plt.savefig(f'{self.model_path}/preds_{extra}.png')

        print('Plots saved to:', self.model_path)

        # plt.show()

    def load_trained(self):
        if self.deploy:
            load_model = torch.load(self.model_deploy_load_path)
            print('Load model path:', self.model_deploy_load_path)
        else:
            load_model = torch.load(self.model_load_path)
            print('Load model path:', self.model_load_path)

        # load all models
        for model_type in self.models:
            print(f'Loading model type: {model_type}')
            if model_type == 'main' and self.model_name == 'dino':
                self.models['main'].cuda()
                self.models['main'].eval()
                continue
            else:
                self.models[model_type].load_state_dict(load_model[model_type])
            self.models[model_type].cuda()

            if self.deploy:
                # set into eval mode
                self.models[model_type].eval()
                print('Eval mode')

        print('Eval check:')
        for model_type in self.models:
            print('Training?:', self.models[model_type].training)

    def _reset_memory(self):

        # reset memory
        self.batch_memory = torch.empty(
            size=(self.batch_size, 0, self.mem_output_shape+3), requires_grad=False).cuda()

        # reset filled boolean
        self.memory_filled = False

    def _reset_fill_count(self):
        self.fill_curr = 0

    def _fill_memory(self):

        self.fill_curr += 1
        # print('Filling memory, fill count:', self.fill_curr)

        first = self.batch_memory[:, :self.fill_curr].reshape(
            self.batch_size, self.fill_curr, -1)

        self.fill_memory = first
        # print('pre-fill', first.shape)
        fill = self.batch_memory[:, -1]

        self._reset_memory()
        self._add_to_memory(first)

        while self.batch_memory.shape[1] < self.memory_size:
            new = deepcopy(fill).reshape(self.batch_size, 1, -1)
            self.batch_memory = torch.concat((self.batch_memory, new), dim=1)

        # print(self.batch_memory)
        self.memory_filled = True

    def _add_to_memory(self, embedding):
        with torch.no_grad():

            self.batch_memory = torch.concat(
                (self.batch_memory, embedding), dim=1)

            if self.memory_filled:
                # remove first element
                new_memory = self.batch_memory[:, 1:]
                del self.batch_memory
                self.batch_memory = new_memory
                assert self.batch_memory.shape == (
                    self.batch_size, self.memory_size, self.mem_output_shape+4), f'{self.batch_memory.shape}'

    def get_memory_zero_comm(self, tens):

        zero_comm = np.array([0.0, 0.0])

        if self.config['scale_commands']:
            zero_comm = (
                zero_comm - self.data_rescales[0])/self.data_rescales[1]

        zero_comm = np.concatenate((zero_comm, [0]))

        if tens:
            zero_comm = torch.tensor(
                zero_comm, device=self.device).reshape(self.batch_size, 3)
            zero_comm = zero_comm.repeat(self.batch_size, 1)
            assert zero_comm.shape == (self.batch_size, 3), zero_comm.shape

        return zero_comm

    def plot_data(self, sample, title, pil=False):
        if pil:
            to_pil = transforms.Compose([
                transforms.ToPILImage(), ]
            )
            sample[1] = to_pil(sample[1])
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title(f'{title}:{sample[0]}')
        plt.imshow(sample[1])
        plt.show()


if __name__ == '__main__':

    demo_type = 'robot'
    demo_folder = 'stata'
    deploy = False
    scaled_commands = True
    use_memory = False
    use_flipped = False
    multi_command = True
    predict_commands = True
    name_extra = ''
    data_type = 'rgb'

    model = ['dino']
    for m in model:

        finetune = False

        cnn = CommandNet(demo_type=demo_type,
                         model_name=m,
                         demo_folder=demo_folder,
                         deploy=deploy,
                         scaled_commands=scaled_commands,
                         finetune=finetune,
                         use_memory=use_memory,
                         use_flipped=use_flipped,
                         multi_command=multi_command,
                         predict_commands=predict_commands,
                         name_extra=name_extra,
                         data_type=data_type)

        cnn.train_model()

        finetune = True

        # cnn = CommandNet(demo_type=demo_type,
        #                  model_name=m,
        #                  demo_folder=demo_folder,
        #                  deploy=deploy,
        #                  scaled_commands=scaled_commands,
        #                  finetune=finetune,
        #                  use_memory=use_memory,
        #                  use_flipped=use_flipped,
        #                  multi_command=multi_command,
        #                  predict_commands=predict_commands,
        #                  data_type=data_type)

        # cnn.train_model()


# %%
# fre
