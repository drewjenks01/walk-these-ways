import torch
import cv2
import numpy as np
from torchvision import transforms


def process_realsense(img, check=False, deploy=False, test=False, augment=False, flipped=False):

    if not flipped:
        # transform to tensor and center crop
        to_tens =  transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(240),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])
    else:
        # flip and transform to tensor and center crop
        to_tens =  transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(1.0) ,
            transforms.Resize(240),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            ])


    to_norm= transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    )

    img_tens=to_tens(img).cuda()

    if deploy:

        final_img = to_norm(img_tens)
        return final_img[None,...]

    elif test or not augment:
        augments = [img_tens]

    else:
        augments, augment_names = augment_img(img_tens)

    final_imgs = []
    for im in augments:
        final_imgs.append(to_norm(im))

    if check:
        to_pil=transforms.Compose([
        transforms.ToPILImage()]
         )

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,len(final_imgs))
        fig.suptitle('Image Augmentations')

        for i in range(len(final_imgs)):
            ax[i].imshow(to_pil(final_imgs[i]))
            ax[i].set_title(augment_names[i])
        plt.show()

    del img_tens, img

    return final_imgs


def augment_img(img,first=False):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    to_pil=transforms.Compose([
        transforms.ToPILImage()]
    )

    rotate_r = transforms.Compose([
        transforms.RandomRotation(degrees=(5,8))]
    )

    rotate_l = transforms.Compose([
        transforms.RandomRotation(degrees=(-8,-5))]
    )

    gauss_blur = transforms.Compose([
        transforms.GaussianBlur(5,1)]
    )


    augments = [img, rotate_r(img), rotate_l(img)]
    augment_names = ['original','rot_r','rot_l']

    return augments, augment_names

    