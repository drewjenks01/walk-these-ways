from navigation import constants
from navigation.vision.utils.data_processing_utils import load_demo_data
from navigation.vision.utils.model_utils import get_predicted_class
from navigation.vision.train import confusion_matrix

import torch
import clip
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import pathlib
from PIL import Image
from sklearn import svm


class ClipClassifier(nn.Module):
    def __init__(self):
        super(ClipClassifier, self).__init__()
        labels_for_classification = [
            "walk",
            "climb",
            #"low",
        ]
        # self.map_labels_to_class_indx = {'walk':0,'climb':1,'low':2}
        self.model, self.preprocess_img = clip.load("ViT-B/32", device=constants.DEVICE)
        for param in self.model.parameters():
            param.requires_grad = False
        self.text = clip.tokenize(labels_for_classification).to(constants.DEVICE)

        # checkpoint = "openai/clip-vit-large-patch14"
        # self.classifier = pipeline(model=checkpoint, task="zero-shot-image-classification", device=constants.DEVICE)

    def forward(self, image):
        image = self.preprocess_img(image).unsqueeze(0).to(constants.DEVICE)
        with torch.inference_mode():
            logits_per_image, logits_per_text = self.model(image, self.text)
            probs = logits_per_image.softmax(dim=-1)

        return probs

    def test_on_dataset(self, demo_path):
        print("testing")
        demo_data = load_demo_data(demo_path)
        paired = self.maked_paired_data(demo_data)

        predicted, actual = [], []

        for img, gait in tqdm(paired):
            pred = int(get_predicted_class(self.forward(img)).flatten().cpu()[0])
            predicted.append(pred)
            actual.append(int(gait))

        print(confusion_matrix(actual, predicted))

    def maked_paired_data(self, demo_data: dict):
        rgb_imgs = demo_data[constants.DEMO_RGB_KEY]
        all_commands = np.array(demo_data[constants.DEMO_COMMAND_KEY])

        paired_data = []

        for rgb_img, command in tqdm(zip(rgb_imgs, all_commands), total=len(rgb_imgs)):
            # processed_img = image_processor(images=rgb_img, return_tensors="pt")['pixel_values'][0]
            processed_img = Image.fromarray(rgb_img)
            gait_label = command[-1]
            data_point = [processed_img, gait_label]
            paired_data.append(data_point)

        return paired_data

class SVMClassifier(nn.Module):
    def __init__(self):
        super(SVMClassifier, self).__init__()
        labels_for_classification = [
            "walk",
            "climb",
            #"low",
        ]
        # self.map_labels_to_class_indx = {'walk':0,'climb':1,'low':2}
        self.model = svm.SVC(gamma='scale')
        self.vision_backbone = Visi

    def forward(self, image):
        image = self.preprocess_img(image).unsqueeze(0).to(constants.DEVICE)
        probs = self.predict(np.array(image.cpu()).reshape(1, -1))
        return probs

    def test_on_dataset(self, demo_path):
        print("testing")
        demo_data = load_demo_data(demo_path)
        paired = self.maked_paired_data(demo_data)

        predicted, actual = [], []

        for img, gait in tqdm(paired):
            pred = int(get_predicted_class(self.forward(img)).flatten().cpu()[0])
            predicted.append(pred)
            actual.append(int(gait))

        print(confusion_matrix(actual, predicted))

    def maked_paired_data(self, demo_data: dict):
        rgb_imgs = demo_data[constants.DEMO_RGB_KEY]
        all_commands = np.array(demo_data[constants.DEMO_COMMAND_KEY])

        paired_data = []

        for rgb_img, command in tqdm(zip(rgb_imgs, all_commands), total=len(rgb_imgs)):
            # processed_img = image_processor(images=rgb_img, return_tensors="pt")['pixel_values'][0]
            processed_img = Image.fromarray(rgb_img)
            gait_label = command[-1]
            data_point = [processed_img, gait_label]
            paired_data.append(data_point)

        return paired_data





if __name__ == "__main__":
    demo_path = constants.DEMO_BASE_PATH / "icra_trials" / "combo"
    cc = ClipClassifier()
    cc.test_on_dataset(demo_path)
