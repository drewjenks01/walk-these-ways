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
from navigation.utils.image_processing import process_image, img_to_tensor_norm, process_realsense
# from torchview import draw_graph
import gc
gc.collect()
torch.cuda.empty_cache()
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
# print(torch.cuda.memory_summary(device='cuda', abbreviated=False))

"""
CNN that takes in images as input and control commands as output

"""


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        row = self.data[index]
        memory = row[:-1]
        inp = row[-1]
        commands = inp[0]
        image = inp[1]
        return image, commands, memory

    def __len__(self):
        return len(self.data)


class CommandNet(nn.Module):
    def __init__(self, model_name, demo_type=None, demo_folder=None, deploy=False, scaled_commands=False, finetune=False, multi_command=False):
        super().__init__()
        self.model_name = model_name
        self.demo_type = demo_type
        self.demo_folder = demo_folder
        self.deploy = deploy
        self.finetune = finetune
        self.multi_command = multi_command

        # --------------------------------
        # Filepaths
        # --------------------------------
        command_type = 'multi_comm' if self.multi_command else 'single_comm'
        train_type = 'finetuned' if self.finetune else 'trained'

        self.root_dir = 'navigation/commandnet/runs/run_recent'
        self.model_path = self.model_save_path = f'{self.root_dir}/{command_type}/{self.demo_folder}/{self.model_name}'
        self.model_save_path = f'{self.model_path}/{train_type}.pth'
        self.model_load_path = f'{self.model_path}/trained.pth'
        if demo_type: self.demo_load_path = f'navigation/robot_demos/{self.demo_folder}/demos.pkl' if demo_type == 'sim' else f'navigation/robot_demos/jenkins_experiment/{self.demo_folder}/runs'
        self.rescale_path = f'{self.root_dir}/{command_type}/{self.demo_folder}/{self.model_name}/rescales.pkl'

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # --------------------------------
        # DATA PARAMS
        # --------------------------------

        self.train_percent = 0.8
        self.val_percent = 0.2

        self.trainloader = None
        self.testloader = None
        self.testdemoloader = None

        self.batch_size = 256
        self.input_h = 240
        self.input_w = 200

        self.scale_commands = scaled_commands
        self.data_rescales = []
        if deploy and scaled_commands:
            with open(self.rescale_path, 'rb') as f:
                self.data_rescales = pkl.load(f)

        self.visualize_data = False
        self.visualize_model = False
        if self.deploy:
            self.visualize_data = False
            self.visualize_model = False

        # --------------------------------
        # DEFINE TRAINING PARAMS
        # --------------------------------
        self.loss_func = nn.MSELoss()
        self.policy_loss = nn.BCELoss()

        if self.finetune:
            self.lr = 2e-4
        else:
            self.lr = 5e-3
        self.epochs = 30

        # --------------------------------
        # DEFINE NEURAL NETS
        # --------------------------------
        self.models = {}

        # conv model

        if self.model_name == 'resnet18':
            from torchvision.models import resnet18, ResNet
            self.commandnet = resnet18(pretrained=not self.deploy)
            self.commandnet.fc = nn.Identity()
            self.memory_input_shape = 512
        elif self.model_name == 'resnet34':
            from torchvision.models import resnet34
            self.commandnet = resnet34(pretrained=not self.deploy)
            self.commandnet.fc = nn.Identity()
            self.memory_input_shape = 512
        elif self.model_name == 'resnet50':
            from torchvision.models import resnet50
            self.commandnet = resnet50(pretrained=not self.deploy)
            self.commandnet.fc = nn.Identity()
            self.memory_input_shape = 2048
        elif self.model_name == 'mnv3s':
            from torchvision.models import mobilenet_v3_small
            self.commandnet = mobilenet_v3_small(pretrained=not self.deploy)
            del self.commandnet.classifier[3]
            self.memory_input_shape = 1024
        elif self.model_name == 'mnv3l':
            from torchvision.models import mobilenet_v3_large
            self.commandnet = mobilenet_v3_large(pretrained=not self.deploy)
            del self.commandnet.classifier[3]
            self.memory_input_shape = 1280
        elif self.model_name == 'enb3':
            from torchvision.models import efficientnet_b3
            self.commandnet = efficientnet_b3(pretrained=not self.deploy)
            self.commandnet.classifier = nn.Identity()
            print(self.commandnet)
            self.memory_input_shape = 1536

        # memory layer to reduce dimensionality
        self.memory_output_shape = 250
        self.models['memory'] = nn.Sequential([
            nn.Linear(self.memory_input_shape, self.memory_output_shape),
            nn.ReLU()
        ])

         # memory
        self.batch_memory=torch.empty(shape=(0,self.memory_output_shape+5))
        self.memory_filled=False
        self.memory_size = 9

        # adapt fc_input_size based on memory size -> conv embedding * mem size + num comms * mem size
        self.fc_input_shape = self.memory_output_shape*self.memory_size + 5*self.memory_size


        self.policy_layer = nn.Sequential(
            nn.Linear(self.fc_input_shape, 1, bias=True),
            nn.Sigmoid()
        )


        # add all models
        self.models['main'] = self.commandnet

        if self.multi_command:

            # make final layers
            self.command_layer = nn.Sequential(
                nn.Linear(self.fc_input_shape, 1, bias=True)
            )

            x = deepcopy(self.command_layer)
            y = deepcopy(self.command_layer)
            yaw = deepcopy(self.command_layer)
            leg = deepcopy(self.command_layer)
            step = deepcopy(self.command_layer)

            self.models['x'] = x
            self.models['y'] = y
            self.models['yaw'] = yaw
            self.models['leg'] = leg
            self.models['step'] = step

        else:
            # make final layers
            self.command_layer = nn.Sequential(
                nn.Linear(self.fc_input_shape, 5, bias=True)
            )
            self.models['commands'] = self.command_layer

        self.models['policy'] = self.policy_layer

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0

        for model_type in self.models:
            self.models[model_type].to(self.device)


        # freeze main NN
        if not self.deploy:
            for p in self.models['main'].parameters():
                p.requires_grad = False

            # unfreeze some layers if finetuning. TODO: freeze BatchNorms
            if self.finetune:
                print('Finetuning!')
                self.load_trained()

                for p in self.models['main'].layer2.parameters():
                    p.requires_grad = True

                for p in self.models['main'].layer3.parameters():
                    p.requires_grad = True

                for p in self.models['main'].layer4.parameters():
                    p.requires_grad = True

            # defines full model and opts
            self.opts = []
            # create optimizer for model(s)
            for model_type in self.models:
                print(model_type)
                self.opts.append(optim.Adam(
                    params=self.models[model_type].parameters(), lr=self.lr))

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


        print(f'PARAMETER SUMMARY\n \
                Batch size: {self.batch_size}     Model Name: {self.model_name}\n \
                LR: {self.lr}         Epochs: {self.epochs} \
                Demo type: {self.demo_type}    Demo folder: {self.demo_folder}\n \
                -----------------------------------------')

    def forward(self, input):

        # images through main layer
        output = self.models['main'](input)

        # put output through mem layer
        output = self.models['memory'] (output)

        # add data to memory
        mem = torch.cat((output, input)).flatten()
        self.batch_memory = torch.cat((self.batch_memory,mem))

        # check if memory not filled
        if not self.memory_filled:

            # check if memory at capacity
            if len(self.batch_memory) == self.memory_size:
                print('Memory filled! Beginning inference.')
                self.memory_filled = True

            # return nothing
            return None, None

        # make new output the flatten conv embeddings
        output = self.batch_memory.flatten()

        #output = nn.Dropout(0.5)(output)

        # output through command regression layer
        if self.multi_command:
            x = self.models['x'](output)
            y = self.models['y'](output)
            yaw = self.models['yaw'](output)

            commands = torch.concat((x, y, yaw), dim=1)

        else:
            commands = self.models['commands'](output)


        # output through policy classification layer
        policy = self.models['policy'](output)

        # remove first element from memory
        self.batch_memory = self.batch_memory[1:] 

        return commands, policy

    def train_model(self):
        from tqdm import tqdm
        trainloader = self.trainloader
        testloader = self.testloader

        for model_type in self.models:
            self.models[model_type].train()

        loss_log = {'x_vel': [], 'y_vel': [], 'yaw': [],
                    'height': [], 'freq': [], 'policy': [], 'total': []}
        test_loss = {'x_vel': [], 'y_vel': [], 'yaw': [],
                     'height': [], 'freq': [], 'policy': [], 'total': []}
        str_log = {'x_vel': [0, 0], 'y_vel': [0, 0], 'yaw': [0, 0], 'height': [
            0, 0], 'freq': [0, 0], 'policy': [0, 0], 'total': [0, 0]}

        print('START TRAINING')
        for epoch in range(self.epochs):

            total_train_loss = {'x_vel': [], 'y_vel': [], 'yaw': [
            ], 'height': [], 'freq': [], 'policy': [], 'total': []}

            for idx, (image, targets,memory) in tqdm(enumerate(trainloader)):

                targets = targets.to(self.device)
                image = image.float()
                targets = targets.float()

                # reset batch mem
                self._reset_memory()

                # fill memory
                for idx, (mem_image, mem_targets) in tqdm(enumerate(memory)):
                    mem_image = mem_image.float()
                    self.forward(mem_image)


                command_targets = targets[:, :-1]
                policy_targets = targets[:, -1].reshape(-1, 1)

                for opt in self.opts:
                    opt.zero_grad(set_to_none=True)

                commands, policy = self.forward(image)

                # calc individual loss to be able to track progress
                ind_loss = [self.loss_func(
                    commands[:, i], command_targets[:, i]) for i in range(commands.shape[1])]
                policy_loss = self.policy_loss(policy, policy_targets)

                # calc true loss all together
                loss = self.loss_func(commands, command_targets)+policy_loss

                total_train_loss['x_vel'].append(ind_loss[0].item())
                total_train_loss['y_vel'].append(ind_loss[1].item())
                total_train_loss['yaw'].append(ind_loss[2].item())
                total_train_loss['height'].append(ind_loss[3].item())
                total_train_loss['freq'].append(ind_loss[4].item())
                total_train_loss['policy'].append(policy_loss.item())

                total_train_loss['total'].append(loss.item())

                loss.backward()
                for opt in self.opts:
                    opt.step()

                del image, targets, command_targets, policy_targets, ind_loss, policy_loss, loss

            for l in total_train_loss:
                loss_log[l].append(np.mean(total_train_loss[l]))

            val_loss = self.evaluate(testloader)
            for l in val_loss:
                test_loss[l].append(np.mean(val_loss[l]))
                str_log[l][1] = val_loss[l][-1]-val_loss[l][0]
                str_log[l][0] = val_loss[l][-1]

            for model_type in self.models:
                self.models[model_type].train()

            print("\n [INFO] EPOCH: {}/{}".format(epoch + 1, self.epochs))
            print(
                f"Train loss: {round(loss_log['total'][-1],4)} Val loss: {round(test_loss['total'][-1],4)}\nTestCommands: {[round(str_log[l][0],4) for l in str_log]}")
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

            eval_log = {'x_vel': [], 'y_vel': [], 'yaw': [],
                        'height': [], 'freq': [], 'policy': [], 'total': []}

            for iter, (image, targets, memory) in enumerate(data):

                # reset batch mem
                self._reset_memory()

                # fill memory
                for idx, (mem_image, mem_targets) in enumerate(memory):
                    mem_image = mem_image.float()
                    self.forward(mem_image)

                image = image.float()
                targets = targets.to(self.device)
                targets = targets.float()
                command_targets = targets[:, :-1]
                policy_targets = targets[:, -1].reshape(-1, 1)

                commands, policy = self.forward(image)

                # calc individual loss to be able to track progress
                ind_loss = [self.loss_func(
                    commands[:, i], command_targets[:, i]) for i in range(commands.shape[1])]
                policy_loss = self.policy_loss(policy, policy_targets)

                # calc true loss all together
                loss = self.loss_func(commands, command_targets)+policy_loss

                eval_log['x_vel'].append(ind_loss[0].item())
                eval_log['y_vel'].append(ind_loss[1].item())
                eval_log['yaw'].append(ind_loss[2].item())
                eval_log['height'].append(ind_loss[3].item())
                eval_log['freq'].append(ind_loss[4].item())
                eval_log['policy'].append(policy_loss.item())

                eval_log['total'].append(loss.item())

                del commands, policy, loss, ind_loss

        return eval_log

    def evaluate_full_demo(self):
        data = self.testdemoloader
        for model_type in self.models:
            self.models[model_type].eval()

        # reset batch mem
        self._reset_memory()

        with torch.no_grad():

            preds = {'x_vel': [], 'y_vel': [], 'yaw': [],
                     'height': [], 'freq': [], 'policy': []}
            labels = {'x_vel': [], 'y_vel': [], 'yaw': [],
                      'height': [], 'freq': [], 'policy': []}

            for iter, (image, targets) in enumerate(data):

                image = image.float()
                targets = targets.to(self.device)
                targets = targets.float()

                targets = targets.reshape(-1).detach().tolist()

                command_targets = targets[:-1]
                policy_targets = targets[-1]

                # if memory not filled yet then fill and continue
                if not self.memory_filled:
                    self.forward(image)
                    continue
                else:
                    commands, policy = self.forward(image)


                commands = commands.reshape(-1).detach().tolist()
                policy = policy.detach().item()

                preds['x_vel'].append(commands[0])
                preds['y_vel'].append(commands[1])
                preds['yaw'].append(commands[2])
                preds['height'].append(commands[3])
                preds['freq'].append(commands[4])
                preds['policy'].append(policy)

                labels['x_vel'].append(command_targets[0])
                labels['y_vel'].append(command_targets[1])
                labels['yaw'].append(command_targets[2])
                labels['height'].append(command_targets[3])
                labels['freq'].append(command_targets[4])
                labels['policy'].append(policy_targets)

        return preds, labels

    def prepare_data(self):
        from tqdm import tqdm
        from pathlib import Path
        import pickle
        import gzip
        print('Preparing Data...')

        num_runs = len([p for p in Path(self.demo_load_path).glob('*')])
        print('Num runs', num_runs)

        comms = []
        test_comms = []
        images = []
        test_images = []
        print('Extracted demos...')

        if self.demo_type == 'robot':
            for i in tqdm(range(num_runs)):
                log_files = sorted([str(p) for p in Path(self.demo_load_path+f'/run{i+1}').glob(
                    "*.pkl")], key=lambda x: int(x.split('/')[-1].split('.')[0][3:]))

                for log in log_files:
                    with gzip.open(log, 'rb') as f:
                        p = pickle.Unpickler(f)
                        demo = p.load()
                        for k in demo:

                            if k == 'Commands':
                                if i == 0:
                                    test_comms += demo[k]
                                else:
                                    comms += demo[k]
                            elif k == 'Image1st':
                                if i == 0:
                                    test_images += demo[k]
                                else:
                                    images += demo[k]

        elif self.demo_type == 'sim':
            with gzip.open(self.demo_load_path, 'rb') as f:
                p = pickle.Unpickler(f)
                df = p.load()

            print(f'Training with {len(df)} demos')

            for r in range(len(df)):
                row = df.iloc[r].to_numpy()

                if r == 0:
                    for i in range(len(row[0])):
                        test_comms.append(row[0][i])
                        test_images.append(row[1][i][:,:,:3])

                else:
                    for i in range(len(row[0])):
                        comms.append(row[0][i])
                        images.append(row[1][i][:,:,:3])

        comms = np.array(comms)
        original_comms = deepcopy(comms)

        images = np.array(images)
        test_images = np.array(test_images)

        test_comms = np.array(test_comms)
        original_test_comms = deepcopy(test_comms)

        print('Shapes', comms.shape, images.shape,
              test_comms.shape, test_images.shape)

        if self.scale_commands:

            min = comms.min(axis=0)
            ptp = comms.ptp(axis=0)
            self.data_rescales = [min, ptp]

            comms = (comms-min)/ptp
            test_comms = (test_comms-min)/ptp

            for i in range(len(self.data_rescales[1])):
                if self.data_rescales[1][i] == 0.0:
                    comms[:, i] = original_comms[:, i]
                    test_comms[:, 1] = original_test_comms[:, 1]

        print('Orignal dataset:', comms.shape, images.shape,
              test_comms.shape, test_images.shape)

        # new full numpy array with everything extracted
        print('Augmenting and processing images...')
        extracted_data = []
        checked = False

        for e in tqdm(range(len(comms))):
            original_img = images[e]

            flipped_img = deepcopy(original_img)
            rev_comms = deepcopy(comms[e])
            rev_comms[1] *= -1
            rev_comms[2] *= -1

        # original image
            if not checked:
                augmented = process_realsense(
                    img=original_img, check=self.visualize_data, augment=True)
                rev_augmented = process_realsense(
                    img=flipped_img, check=self.visualize_data, augment=True, flipped=True)
                checked = True
            else:
                augmented = process_realsense(img=original_img, augment=True)
                rev_augmented = process_realsense(
                    img=flipped_img, augment=True, flipped=True)

            for i in range(len(augmented)):
                extracted_data.append([comms[e], augmented[i]])
                extracted_data.append([rev_comms,rev_augmented[i]])

            torch.cuda.empty_cache()

        del comms, images
        if self.visualize_data:
            self._visualize_robot_dataset_(original_comms, extracted_data)

        extracted_test = []
        for e in range(len(test_comms)):
            if e == 0:
                augmented = process_realsense(img=test_images[e], test=True)
            else:
                augmented = process_realsense(img=test_images[e], test=True)

            for im in augmented:
                extracted_test.append([test_comms[e], im])

        del test_comms, test_images


        # create batches of 10 -> (N, 10, comm, im)
        batched_extracted_data = []
        batch=[]
        print('Batching...')
        for i in range(len( extracted_data)):
            batch.append(extracted_data[i])

            while len(batch)==10:
                batch_copy = deepcopy(batch)
                batched_extracted_data.append(batch_copy)
                batch.pop(0)

        del batch, batch_copy, extracted_data

        batched_extracted_test = []
        batch_test=[]
        for i in range(len( extracted_test)):
            batch_test.append(extracted_test[i])

            while len(batch_test)==10:
                batch_copy = deepcopy(batch_test)
                batched_extracted_test.append(batch_copy)
                batch_test.pop(0)


        np.random.shuffle(batched_extracted_data)
        np.random.shuffle(batched_extracted_test)

        train = batched_extracted_data
        test = batched_extracted_test

        print('Train samples:', len(train))
        print('Test samples:', len(test))
        print('Data, Test shape:', batched_extracted_data.shape, batched_extracted_test.shape)

        print('Image shape:', batched_extracted_data[0][0][1].shape)

        # test data in batches of 10
        train_data = CustomDataset(train)
        trainloader = DataLoader(train_data, batch_size=self.batch_size,
                                 shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

        # eval data in batches of 10
        test_data = CustomDataset(test)
        testloader = DataLoader(
            test_data, batch_size=self.batch_size, num_workers=2, pin_memory=True, shuffle=True)

        # final test data in batches of 1...memory will have to fill
        testdemoloader = DataLoader(
            extracted_test, batch_size=1, num_workers=2, pin_memory=True)
            

        self.trainloader = trainloader
        self.testloader = testloader
        self.testdemoloader = testdemoloader

    def _data_rescale(self, commands, policy):

        # prepare commands
        commands = commands.reshape(-1).detach().cpu().tolist()

        # prepare policy
        policy = round(policy.detach().cpu().item())

        if self.scale_commands:
            commands.append(policy)

            commands = np.array(commands)
            commands = commands*self.data_rescales[1]+self.data_rescales[0]

            policy = commands[-1]
            commands = commands[:-1]

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

        rand = []
        for i in range(len(extracted)):
            if np.random.choice(2, 1) == 1:
                rand.append(to_pil(extracted[i][1]))
            if len(rand) == 36:
                break

        rand = np.array(rand, dtype=object)

        print('Num of visualized inputs:', len(rand))

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
        yaws = []
        height = []
        freq = []
        policy = []

        for [(x, y, yaw, h, f, p), _] in extracted:
            x_vels.append(x)
            y_vels.append(y)
            yaws.append(yaw)
            height.append(h)
            freq.append(f)
            policy.append(p)

        origx = []
        origy = []
        origyaw = []
        origh = []
        origf = []
        origp = []

        for x, y, yaw, h, f, p in original:
            origx.append(x)
            origy.append(y)
            origyaw.append(yaw)
            origh.append(h)
            origf.append(f)
            origp.append(p)

        fig, axes = plt.subplots(2, 6)

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

        axes[0, 3].hist(height)
        axes[0, 3].set_title('foot height')
        axes[1, 3].hist(origh)
        axes[1, 3].set_title('orig foot height')

        axes[0, 4].hist(freq)
        axes[0, 4].set_title('step freq')
        axes[1, 4].hist(origf)
        axes[1, 4].set_title('orig step freq')

        axes[0, 5].hist(policy)
        axes[0, 5].set_title('policy')
        axes[1, 5].hist(origp)
        axes[1, 5].set_title('orig policy')

        fig.tight_layout()
        plt.show()

    def _save_all_models_(self):
        import pickle as pkl
        all_models = {}

        for model_type in self.models:
            print(f'Saving model type: {model_type}')
            all_models[model_type] = self.models[model_type].state_dict()

        torch.save(all_models, self.model_save_path)

        if self.scale_commands:
            print('saving rescale data...')
            with open(self.rescale_path, 'wb') as f:
                pkl.dump(self.data_rescales, f)

    def _save_model_plots(self, loss_log, test_loss):
        import matplotlib.pyplot as plt

        log_keys = ['total', 'x_vel', 'y_vel',
                    'yaw', 'height', 'freq', 'policy']

        fig, ax = plt.subplots(2, 4, sharey=True)
        fig.suptitle(
            f'Loss per Epoch\n Final train loss: {round(loss_log["total"][-1],4)}    Final eval loss: {round(test_loss["total"][-1],4)} ')
        count = 0
        for i in range(2):
            for j in range(4):
                if i == 1 and j == 3:
                    #  ax[i,j].plot(range(len(test_loss['time'])),test_loss['time'],label='test')
                    #  ax[i,j].set_title('Time')
                    #  ax[i,j].tick_params(axis='y')
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
        plt.savefig(f'{self.model_path}/losses.png')
        plt.show()

        if self.demo_type == 'robot':
            eval_preds, eval_labels = self.evaluate_full_demo()
            eval_keys = log_keys[1:]
            fig, ax = plt.subplots(2, 3, sharey=True)
            fig.suptitle(f'Full test demo')
            count = 0
            for i in range(2):
                for j in range(3):

                    ax[i, j].plot(range(len(eval_preds[eval_keys[count]])),
                                  eval_preds[eval_keys[count]], label='Pred')
                    ax[i, j].plot(range(len(eval_labels[eval_keys[count]])),
                                  eval_labels[eval_keys[count]], label='Truth')
                    ax[i, j].set_title(eval_keys[count])
                    count += 1
            handles, labels = ax[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right')
            fig.tight_layout()
            plt.savefig(f'{self.model_path}/preds.png')
            plt.show()

    def load_trained(self):

        load_model = torch.load(self.model_load_path)

        # load all models
        for model_type in self.models:
            print(f'Loading model type: {model_type}')
            self.models[model_type].load_state_dict(load_model[model_type])
            self.models[model_type].cuda()

            if self.deploy:
                # set into eval mode
                self.models[model_type].eval()

        
        if self.deploy:
            # dummy pass to cache
            fake_data=torch.zeros(size=(1,3,224,224)).cuda()

            self.forward(fake_data)

            print('Model ready for inference.')

    def _reset_memory(self):

        # reset memory
        self.batch_memory=torch.empty(shape=(0,self.memory_output_shape+5))

        # reset filled boolean
        self.memory_filled = False

if __name__ == '__main__':

    demo_type = 'robot'
    demo_folder = 'simple'
    deploy = 'False'
    scaled_commands = False
    finetune = False

    model = ['resnet18']
    for m in model:
        cnn = CommandNet(demo_type=demo_type,
                         model_name=m,
                         demo_folder=demo_folder,
                         deploy=deploy,
                         scaled_commands=scaled_commands,
                         finetune=finetune)

        cnn.train_model()


# %%
# fre
