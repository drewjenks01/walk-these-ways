import torch.nn as nn
import torch
from PIL import Image as PILImage
from torchvision.transforms import functional as TF
from torchvision import transforms
from typing import List
import numpy as np

from navigation import constants


class BaseModel(nn.Module):
    def __init__(self):
       super(BaseModel, self).__init__()

       self.context_queue = []

    def forward(self, images):
        raise NotImplementedError
    
    def transform_images(self, pil_imgs: List[PILImage.Image], image_size: List[int], center_crop: bool = False) -> torch.Tensor:
        """Transforms a list of PIL image to a torch tensor."""
        transform_type = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                        0.229, 0.224, 0.225]),
            ]
        )
        if type(pil_imgs) != list:
            pil_imgs = [pil_imgs]
        transf_imgs = []
        for pil_img in pil_imgs:
            w, h = pil_img.size
            if center_crop:
                if w > h:
                    pil_img = TF.center_crop(pil_img, (h, int(h * constants.IMAGE_ASPECT_RATIO)))  # crop to the right ratio
                else:
                    pil_img = TF.center_crop(pil_img, (int(w / constants.IMAGE_ASPECT_RATIO), w))
            pil_img = pil_img.resize(image_size) 
            transf_img = transform_type(pil_img)
            transf_img = torch.unsqueeze(transf_img, 0)
            transf_imgs.append(transf_img)
        return torch.cat(transf_imgs, dim=1)
    
    def update_context_queue(self, pil_img):
        if len(self.context_queue) < constants.CONTEXT_SIZE + 1:
            self.context_queue.append(pil_img)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(pil_img)
    
    @staticmethod
    def to_numpy(tensor: torch.tensor):
        return tensor.cpu().detach().numpy()

    @staticmethod
    def numpy_to_pil(numpy_img):
        return PILImage.fromarray(numpy_img)
