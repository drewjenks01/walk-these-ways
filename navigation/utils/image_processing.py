import torch
import cv2
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional


def process_image(img):

    #print(img.shape)
   # img = img[120:,90:270,:]

    # transform to tensor and center crop
    to_tens =  transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        ])

    processed = to_tens(img)

    return processed


def normalize_image(img):
    to_norm= transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    norm_img = to_norm(img)

    return norm_img


def process_deployed(img):

    processed = process_image(img).cuda()
    normalized = normalize_image(processed)[None, ...]

    return normalized

def process_depth(img):
    cv_image_norm = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX)

    return cv_image_norm



def augment_image(img,check=False):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    rotate_r = transforms.Compose([
        transforms.RandomRotation(degrees=(5,8))]
    )

    rotate_l = transforms.Compose([
        transforms.RandomRotation(degrees=(-8,-5))]
    )

    gauss_blur = transforms.Compose([
        transforms.GaussianBlur(5,1)]
    )

    # rand_crop = transforms.Compose([
    #     transforms.RandomResizedCrop(size=(224,224))]
    # )

    vert = transforms.Compose([
        transforms.RandomVerticalFlip(1.0)]
    )

    bright = functional.adjust_brightness(img,1.5)

    dark =  functional.adjust_brightness(img,0.5)


    augments = [gauss_blur]
    augment_names = ['original', 'rl','rr', 'blur','brihgt','dark']

    augmented_images = [img]

    for aug in augments:
        augmented_images.append(aug(img))

    augmented_images+=[bright,dark]


    # if check:
    #     to_pil=transforms.Compose([
    #     transforms.ToPILImage()]
    #      )

    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots(1,len(augmented_images))
    #     fig.suptitle('Image Augmentations')

    #     for b in range(len(augmented_images)):
    #         ax[b].imshow(to_pil(augmented_images[b]))
    #         ax[b].set_title(augment_names[b])
    #     #plt.tight_layout()
    #     plt.show()
        

    return augmented_images

def horiz_flip_img(img):

    horiz_flip = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=1.0)]
    )

    hor_img = np.array(horiz_flip(img))

    return hor_img


def process_batch(batch, augment, check=False):

    # process each image
    processed = [[c,process_image(im)] for c,im in batch]

    # augment each image of each batch
    if augment:
        aug_processed, augment_names = augment_batch(processed)

        processed += aug_processed

        # normalize each image of each batch
        normalized = [[[c,normalize_image(im)] for c,im in batch] for batch in processed]

    else:
         # normalize each image of each batch
        normalized = [[c,normalize_image(im)] for c,im in processed]



    if check and augment:
        to_pil=transforms.Compose([
        transforms.ToPILImage()]
         )

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(normalized),len(normalized[0]))
        fig.suptitle('Image Augmentations')

        for b in range(len(normalized)):
            for i in range(len(normalized[b])):
                ax[b,i].imshow(to_pil(normalized[b][i][1]))
                ax[b,i].set_title(augment_names[b])
        #plt.tight_layout()
        plt.show()

    elif check:
        to_pil=transforms.Compose([
        transforms.ToPILImage()]
         )

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,len(normalized))
        fig.suptitle('Image Augmentations')

        for b in range(len(normalized)):
            ax[b].imshow(to_pil(normalized[b][1]))
        #plt.tight_layout()
        plt.show()


    return normalized
