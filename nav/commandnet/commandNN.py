import torch.nn as nn
from torch import optim
from tqdm import tqdm
import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms 
from copy import deepcopy
from nav.commandnet.commandNN_utils import process_image, img_to_tensor_norm
from torchview import draw_graph
import gc
gc.collect()
torch.cuda.empty_cache()

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
"""
CNN that takes in images as input and control commands as output

"""
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        row = self.data[index]
        commands = row[0]
        images = row[1]
        return images, commands

    def __len__(self):
        return len(self.data)


class CommandNet(nn.Module):
    def __init__(self,image_mode=None,deploy=False):
        super().__init__()

        # --------------------------------
        # DATA PARAMS
        # --------------------------------
        if not image_mode:
            self.image_mode = 'comb'        # 'comb', 'first', or 'third'
        else:
            self.image_mode = image_mode
        self.train_percent = 0.8
        self.val_percent = 0.2
        self.batch_size= 32
        self.input_h = 240
        self.input_w = 200

        if self.image_mode == 'comb':
            self.input_channels = 6
        elif self.image_mode in ['first','third']:
            self.input_channels = 3

        self.data_mean=0
        self.data_std=0
        self.data_rescales = []
        self.deploy=deploy
        self.visualize_data=False
        self.visualize_model = False
        if self.deploy:
            self.visualize_data=False
            self.visualize_model = False
        

        # --------------------------------
        # DEFINE NEURAL NETS
        # --------------------------------
        self.main_model = nn.Sequential(

            nn.Conv2d(self.input_channels,32,kernel_size=(3,3),padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(32),

            nn.Conv2d(32,32,kernel_size=(3,3),padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(32),

            nn.Conv2d(32,32,kernel_size=(3,3),padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(32),

            nn.Conv2d(32,32,kernel_size=(3,3),padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(32),

            nn.Conv2d(32,32,kernel_size=(3,3),padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(32),

            nn.Flatten(),
            nn.Dropout(0.5),
        )

        self.control_model_input = 960
        self.control_model=nn.Sequential(


            nn.Linear(self.control_model_input, 50),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.BatchNorm1d(50),


            # nn.Linear(100, 10),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.BatchNorm1d(10),

            nn.Linear(50, 10),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.BatchNorm1d(10),

            nn.Linear(10,1)
            )
        self.x_vel_model = deepcopy(self.control_model)
        self.y_vel_model = deepcopy(self.control_model)
        self.yaw_model = deepcopy(self.control_model)
        self.leg_height_model = deepcopy(self.control_model)
        self.step_freq_model = deepcopy(self.control_model)
        self.models={'main':self.main_model, 'x_vel':self.x_vel_model, 'y_vel':self.y_vel_model, 
                        'yaw':self.yaw_model, 'leg_height':self.leg_height_model, 'step_freq':self.step_freq_model}
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
        for model in self.models:self.models[model].to(self.device)
        
        # --------------------------------
        # TRAINING PARAMS
        # --------------------------------
        self.loss_func = nn.MSELoss()
        self.lr= 2e-3
        self.epochs= 25
        self.output_dir=os.path.join('nav/commandnet/runs', 'run_recent')
        self.opts ={'main':optim.Adam(params=self.models['main'].parameters(),lr=self.lr), 
                    'x_vel':optim.Adam(params=self.models['x_vel'].parameters(),lr=self.lr),
                    'y_vel':optim.Adam(params=self.models['y_vel'].parameters(),lr=self.lr),
                    'yaw':optim.Adam(params=self.models['yaw'].parameters(),lr=self.lr),
                    'leg_height':optim.Adam(params=self.models['leg_height'].parameters(),lr=self.lr),
                    'step_freq':optim.Adam(params=self.models['step_freq'].parameters(),lr=self.lr)}

        summary(self, input_size=(self.batch_size,self.input_channels, self.input_h, self.input_w),depth = 2,device=self.device)
        # draw_graph(self, input_size=(self.batch_size,self.input_channels, self.input_h, self.input_w), device=self.device,
        #                 save_graph=True, filename='CommandNet_graph',directory='nav/commandnet/graph/')

    def forward(self,input):
        #input= input.float()
        x=self.models['main'](input)
        x_vel=self.models['x_vel'](x)
        y_vel=self.models['y_vel'](x)
        yaw=self.models['yaw'](x)
        leg_height=self.models['leg_height'](x)
        step_freq=self.models['step_freq'](x)

        return x_vel,y_vel,yaw,leg_height,step_freq


    def train_model(self, trainloader, testloader):
        for model in self.models:self.models[model].train()
        loss_log={'x_vel':[],'y_vel':[],'yaw':[],'height':[], 'freq':[],'total':[]}

        print('START TRAINING')
        for epoch in range(self.epochs):

            total_train_loss={'x_vel':[],'y_vel':[],'yaw':[],'height':[], 'freq':[],'total':[]}

            for idx, (image, targets) in enumerate(trainloader):

                image, targets= image.to(self.device), targets.to(self.device)
                image=image.float()
                targets=targets.float()
                for k in self.opts: self.opts[k].zero_grad()

                x_vel,y_vel,yaw,leg_height,step_freq=self.forward(image)


                x_loss=self.loss_func(targets[:,0].unsqueeze(1), x_vel)
                y_loss=self.loss_func(targets[:,1].unsqueeze(1), y_vel)
                yaw_loss=self.loss_func(targets[:,2].unsqueeze(1), yaw) # increased magnitude
                height_loss=self.loss_func(targets[:,3].unsqueeze(1), leg_height)
                step_loss=self.loss_func(targets[:,4].unsqueeze(1), step_freq)

                loss = x_loss+y_loss+yaw_loss+height_loss+step_loss

                total_train_loss['x_vel'].append(x_loss.item())
                total_train_loss['y_vel'].append(y_loss.item())
                total_train_loss['yaw'].append(yaw_loss.item())
                total_train_loss['height'].append(height_loss.item())
                total_train_loss['freq'].append(step_loss.item())
                total_train_loss['total'].append(loss.item())


                loss.backward()
                for k in self.opts: self.opts[k].step()

            for l in total_train_loss:
                loss_log[l].append(np.mean(total_train_loss[l]))

            
            val_loss = self.evaluate(testloader)
            for model in self.models:self.models[model].train()
            
            print("\n [INFO] EPOCH: {}/{}".format(epoch + 1, self.epochs))
            print("Train loss: {:.6f}  Val loss: {:.4f}".format(loss_log['total'][-1], val_loss))

        print('------TEST------')
        test_loss = self.evaluate(testloader)
        print('LOSS Test:',test_loss)


        for model in self.models:
            torch.save(self.models[model].state_dict(),
                    os.path.join(self.output_dir, model+'_'+self.image_mode+'_weights_final.pth'))

        fig, ax = plt.subplots(2,3,sharey=True)
        fig.suptitle('Loss per Epoch')
        ax[0,0].plot(range(len(loss_log['total'])),loss_log['total'])
        ax[0,0].set_title('Train total')
        ax[0,1].plot(range(len(loss_log['total'])),loss_log['x_vel'])
        ax[0,1].set_title('Train x_vel')
        ax[0,2].plot(range(len(loss_log['total'])),loss_log['y_vel'])
        ax[0,2].set_title('Train y_vel')
        ax[1,0].plot(range(len(loss_log['total'])),loss_log['yaw'])
        ax[1,0].set_title('Train yaw')
        ax[1,1].plot(range(len(loss_log['total'])),loss_log['height'])
        ax[1,1].set_title('Train height')
        ax[1,2].plot(range(len(loss_log['total'])),loss_log['freq'])
        ax[1,2].set_title('Train freq')

        plt.show()

        if self.visualize_model: self._visualize_model_()



    def evaluate(self, data):
        for model in self.models: self.models[model].eval()
        with torch.no_grad():

            loss_log = {'x_vel':[],'y_vel':[],'yaw':[],'height':[], 'freq':[],'total':[]}

            for iter, (images, targets) in enumerate(data):

                images=images.to(self.device)
                targets = targets.to(self.device)
                x_vel,y_vel,yaw,leg_height,step_freq=self.forward(images)

                x_loss=self.loss_func(targets[:,0].unsqueeze(1), x_vel)
                y_loss=self.loss_func(targets[:,1].unsqueeze(1), y_vel)
                yaw_loss=self.loss_func(targets[:,2].unsqueeze(1), yaw) # increased magnitude
                height_loss=self.loss_func(targets[:,3].unsqueeze(1), leg_height)
                step_loss=self.loss_func(targets[:,4].unsqueeze(1), step_freq)

                loss = x_loss+y_loss+yaw_loss+height_loss+step_loss

                loss_log['x_vel'].append(x_loss.item())
                loss_log['y_vel'].append(y_loss.item())
                loss_log['yaw'].append(yaw_loss.item())
                loss_log['height'].append(height_loss.item())
                loss_log['freq'].append(step_loss.item())
                loss_log['total'].append(loss.item())
        

        avg_total_loss = np.mean(loss_log['total'])
        return avg_total_loss


    def prepare_data(self):
        print('Preparing Data')

        df = pd.read_pickle('nav/robot_demos/demosDF.pkl')
 
        comms=[]
        first_person=[]
        third_person=[]
        for r in range(len(df)):
            row=df.iloc[r].to_numpy()
            for i in range(len(row[0])):
                comms.append(row[0][i])
                first_person.append(row[1][i])
                third_person.append(row[2][i])

        comms=np.array(comms)
        first_person=np.array(first_person)
        third_person=np.array(third_person)

        print(comms.shape, first_person.shape,third_person.shape)

        # normalize commands [-1,1]
        for i in tqdm(range(len(comms[0]))):
            mul = np.amax(comms[:,i])-np.amin(comms[:,i])
            add = np.amin(comms[:,i])
            self.data_rescales.append([mul, add])
            comms[:,i]=(comms[:,i]-add)/(mul)


        # new full numpy array with everything extracted
        extracted_data = []
        for e in tqdm(range(len(comms))):
            if e==0:
                imgs, rev_imgs= process_image(first_person[e], third_person[e],self.image_mode,check=self.visualize_data)
            else:
                imgs, rev_imgs= process_image(first_person[e], third_person[e], self.image_mode)
            assert imgs.shape == rev_imgs.shape

            extracted_data.append([comms[e],imgs])
            rev_commands=np.array([comms[e][0],-1*comms[e][1],-1*comms[e][2], comms[e][3], comms[e][4]])
            extracted_data.append([rev_commands,rev_imgs])

        extracted_data=np.array(extracted_data,dtype=object)

        self._dataset_mean_std_(extracted_data)

        # stop here when used a deployed model
        if self.deploy:
            return

        for i in range(len(extracted_data)):
            extracted_data[i][1]=img_to_tensor_norm(extracted_data[i][1],self.data_mean,self.data_std)

        if self.visualize_data:
            self._visualize_dataset_(extracted_data,df)


        print('Extracted data shape:',extracted_data.shape)
        print('Image size:',extracted_data[0][1].shape)
        np.random.shuffle(extracted_data)

        num_samples = len(extracted_data)
        num_train_samples = int(num_samples*self.train_percent)


        train = extracted_data[:num_train_samples]
        test = extracted_data[num_train_samples:]

        print(train.shape,test.shape)

        print('Train samples:', len(train))
        print('Test samples:', len(test))

        train_data = CustomDataset(train)
        test_data = CustomDataset(test)

        trainloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True,drop_last=True)
        testloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True, drop_last=True)

        return trainloader, testloader

    def _dataset_mean_std_(self,extracted_data):
        transform = transforms.Compose([transforms.ToTensor()])

        # placeholders used for mean and std of dataset
        psum    = torch.tensor([0.0]*self.input_channels)
        psum_sq = torch.tensor([0.0]*self.input_channels)
        
        for c,i in extracted_data:
            inputs = transform(i)

            psum    += inputs.sum(axis        = [1,2])
            psum_sq += (inputs ** 2).sum(axis = [1,2])  

        count = len(extracted_data) * self.input_h * self.input_w

        # mean and std
        total_mean = psum / count
        total_var  = (psum_sq / count) - (total_mean ** 2)
        total_std  = torch.sqrt(total_var)

        self.data_mean=total_mean
        self.data_std=total_std

        return
    
    
    def _data_rescale(self, inp):
        out=[]
        for i in range(len(inp)):
            inp_scalar = inp[i].detach().cpu().item()
            out.append(inp_scalar*self.data_rescales[i][0]+self.data_rescales[i][1])

        return out

    
    def _visualize_model_(self):
        model=self.models['main']
        model_weights=[]
        conv_layers=[]

        model_children = list(model.modules())
        # counter to keep count of the conv layers
        counter = 0 
        # append all the conv layers and their respective weights to the list
        for i in range(len(model_children)):
            #print(model_children[i])
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
            plt.subplot(10, 10, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
            plt.imshow(filter[0, :, :].detach().cpu(), cmap='gray')
            plt.axis('off')
        plt.show()

        plt.figure(figsize=(20, 17))
        for i, filter in enumerate(model_weights[1]):
            plt.subplot(10, 10, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
            plt.imshow(filter[0, :, :].detach().cpu(), cmap='gray')
            plt.axis('off')
        plt.show()


        # example image
        df = pd.read_pickle('nav/robot_demos/demosDF.pkl')
        extracted_data = []
        for r in range(1):
            row = df.iloc[r].to_numpy()
            commands = row[0]
            first_person = row[1]
            third_person = row[2]

            for e in range(30,31):
                imgs ,_= process_image(first_person[e], third_person[e],self.image_mode)
                extracted_data.append([commands[e],imgs])


        img = img_to_tensor_norm( extracted_data[0][1],self.data_mean,self.data_std)
        print(img.shape)
        assert img.shape == (self.input_channels,self.input_h, self.input_w)
        img = img[None,...].cuda().float()

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
                if i == 64: # we will visualize only 8x8 blocks from each layer
                    break
                plt.subplot(8, 8, i + 1)
                plt.imshow(filter.detach().cpu(), cmap='gray')
                plt.axis("off")
            plt.show()
            plt.close()

    def _visualize_dataset_(self,extracted,df):
        
        # INPUTS

        # extract 36 random images
        imgs=extracted[:,1]

        rand=[]
        for i in range(len(imgs)):
            if np.random.choice(2,1)==1:
                rand.append(np.moveaxis(imgs[i].cpu().numpy(),0,-1))
            if len(rand)==36:
                break

        rand=np.array(rand)


        print('Num of visualized inputs:', len(rand))

        if self.image_mode=='comb':
            plt.figure(figsize=(20, 17))
            plt.title('Input examples')
            for i in range(len(rand)//2):
                plt.subplot(6, 6, 2*i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
                plt.imshow(rand[i][:,:,:3])
                plt.subplot(6, 6, 2*i+2) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
                plt.imshow(rand[i][:,:,3:])
                plt.axis('off')
            plt.show()

            plt.figure(figsize=(20, 17))
            plt.title('Input examples histogram')
            for channel_id, color in enumerate(('red','green','blue')):
                plt.subplot(1,2,1)
                histogram, bin_edges = np.histogram(
                    np.squeeze(rand[:,:, :, channel_id]), bins=50, range=(0,1)
                )
                plt.plot(bin_edges[0:-1], histogram, color=color,alpha=0.4)
                
                # third person
                plt.subplot(1,2,2)
                histogram, bin_edges = np.histogram(
                    np.squeeze(rand[:,:, :, 3+channel_id]), bins=50, range=(0,1)
                )
                plt.plot(bin_edges[0:-1], histogram, color=color,alpha=0.4)
            plt.show()


        else:

            plt.figure(figsize=(20, 17))
            plt.title('Input examples')
            for i in range(len(rand)):
                plt.subplot(6, 6, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
                plt.imshow(rand[i])
                plt.axis('off')
            plt.show()

            plt.figure(figsize=(20, 17))
            plt.title('Input examples histogram')
            for channel_id, color in enumerate(('red','green','blue')):
                histogram, bin_edges = np.histogram(
                    np.squeeze(rand[:,:, :, channel_id]), bins=50, range=(0,1)
                )
                plt.plot(bin_edges[0:-1], histogram, color=color,alpha=0.4)

            plt.show()




        # OUTPUTS
        dems=np.array(df['Commands'])
        df_commands=[]

        for dem in dems:
            for move in dem:
                df_commands.append(move)

        df_commands=np.array(df_commands)

        x_vels_d = df_commands[:,0]
        y_vels_d =df_commands[:,1]
        yaws_d = df_commands[:,2]
        height_d = df_commands[:,3]
        freq_d= df_commands[:,4]

        x_vels=[]
        y_vels=[]
        yaws=[]
        height=[]
        freq=[]

        for x,y,yaw,h,f in extracted[:,0]:
            x_vels.append(x)
            y_vels.append(y)
            yaws.append(yaw)
            height.append(h)
            freq.append(f)


        fig, axes = plt.subplots(2,5)
        axes[0,0].hist(x_vels_d)
        axes[0,0].set_title('x_vels')
        axes[0,1].hist(y_vels_d)
        axes[0,1].set_title('y_vels')
        axes[0,2].hist(yaws_d)
        axes[0,2].set_title('yaws')
        axes[0,3].hist(height_d)
        axes[0,3].set_title('foot height')
        axes[0,4].hist(freq_d)
        axes[0,4].set_title('step freq')

        axes[1,0].hist(x_vels)
        axes[1,0].set_title('x_vels')
        axes[1,1].hist(y_vels)
        axes[1,1].set_title('y_vels')
        axes[1,2].hist(yaws)
        axes[1,2].set_title('yaws')
        axes[1,3].hist(height)
        axes[1,3].set_title('foot height')
        axes[1,4].hist(freq)
        axes[1,4].set_title('step freq')
        fig.tight_layout()
        plt.show()
        




if __name__ == '__main__':
    cnn = CommandNet()

    trainloader,testloader = cnn.prepare_data()

    cnn.train_model(trainloader, testloader)







