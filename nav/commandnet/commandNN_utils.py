import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision import transforms
import glob
import torch


torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


def process_image(first, third,image_mode,check=False):

        # remove 4th channel
        first=first[:,:,:3]
        third=third[:,:,:3]

        # crop images
        first = first[:,75:275,:]
        third = third[:,75:275,:]

        # first = cv2.resize(first, (200,66))
        # third = cv2.resize(third, (200,66))

        rev_first = cv2.flip(first,1)
        rev_third = cv2.flip(third,1)

        if check:
            fig, ax = plt.subplots(2,2)
            ax[0,0].imshow(first)
            ax[0,1].imshow(rev_first)
            ax[1,0].imshow(third)
            ax[1,1].imshow(rev_third)
            plt.show()

        if image_mode == 'first':
            assert first.shape[2] == 3
            return first, rev_first

        elif image_mode == 'third':
            assert third.shape[2]==3
            return third, rev_third


        elif image_mode == 'comb':

            #concat first and third person images together
            imgs = np.concatenate([first, third],axis=-1)
            rev_imgs = np.concatenate([rev_first,rev_third],axis=-1)
            assert imgs.shape[2] ==6

            return imgs, rev_imgs


def img_to_tensor_norm(img,mean,std):
    # global positive standardization

    transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
        ])

    img_norm = transform_norm(img)

    # img_norm=torch.clip(img_norm,-1.0,1.0)
    # img_norm=torch.add(img_norm,1.0)
    # img_norm=torch.div(img_norm,2.0)

    #img_norm =( img_norm-torch.amin(img_norm))/(torch.amax(img_norm) - torch.amin(img_norm))

    #print(torch.amax(img_norm), torch.amin(img_norm))

    # assert torch.amax(img_norm)<=1.0 and torch.amin(img_norm)>=0.0


    return img_norm

def load_trained(mode):   # mode = ['comb', 'first','third]
    from nav.commandnet.commandNN import CommandNet
    model = CommandNet(mode)
    model.visualize_data=False
    model.visualize_model=False
    model.deploy=True
    model.image_mode=mode

    # prepare data to extract mean
    model.prepare_data()
    
    logdir = 'nav/commandnet/runs/run_recent/'

    if mode=='comb':
        for m in model.models:
            model.models[m].load_state_dict(torch.load(logdir+m+'_comb_weights_final.pth'),strict=False)
            model.models[m].cuda()
            model.models[m].eval()
    elif mode=='first':
        for m in model.models:
            model.models[m].load_state_dict(torch.load(logdir+m+'_first_weights_final.pth'),strict=False)
            model.models[m].cuda()
            model.models[m].eval()

    elif mode=='third':
        for m in model.models:
            model.models[m].load_state_dict(torch.load(logdir+m+'_third_weights_final.pth'),strict=False)
            model.models[m].cuda()
            model.models[m].eval()
    return model, model.data_mean, model.data_std