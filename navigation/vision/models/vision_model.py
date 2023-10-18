from navigation import constants
from navigation.vision.utils.model_utils import get_output_shape

import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformers import pipeline
import logging
from sklearn import svm


class VisionModel(nn.Module):
    """Creates full vision model from all components.
    """

    def __init__(self, inference_type: str, num_cameras: int, classifier_type: str, regressor_type: str):
        """
        Args:
            inference_type (str) in {"online", "offline"}: whether 
                inference is online (on-robot) or offline. 
            num_cameras (int): num of cameras (images) being used
                for input
        """
        super(VisionModel, self).__init__()
        self.inference_type = inference_type
        self.num_cameras = num_cameras
        self.classifier_type = classifier_type
        self.regressor_type = regressor_type

        self.backbone, self.backbone_output_shape = VisionModel.build_backbone(inference_type=self.inference_type)

        self.gait_head = self.build_gait_head()
        self.command_head = self.build_command_head()

    def forward(self, images: dict):
        with torch.no_grad():
            # process each camera image seperately and then combine features
            forward_features = self.backbone(images['forward'])
            downward_features = self.backbone(images['downward'])
            features = torch.cat((forward_features, downward_features), dim=1)

        class_logits = self.gait_head(features)
        regression_outputs = self.command_head(features)

        return class_logits, regression_outputs
    
    @staticmethod
    def build_backbone(inference_type: str):
        #from transformers import AutoImageProcessor, Dinov2ForImageClassification
        # if doing inference on-robot, need to load local model
        if inference_type == 'online':
                backbone_model = torch.hub.load(
                    '/home/unitree/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_vits14', source='local')
        elif inference_type == 'offline':
            backbone_model = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vits14', ffn_layer = "identity")
   
        else:
            raise AssertionError(f'Invalid inference type. Should be "offline" or "online", not {inference_type}')

        # freeze params
        for param in backbone_model.parameters():
            param.requires_grad = False
        
        # input shape to the model (batch size == 1, num_channels == 3)
        image_width = 224
        image_height = 224
        input_shape = (1, 3, image_height, image_width)

        # get output shape of backbone model
        backbone_output_shape = get_output_shape(backbone_model, input_shape)

        backbone_model.to(constants.DEVICE)

        return backbone_model, backbone_output_shape

    def build_gait_head(self):
        """Defines layers of the classification model and returns full model.
        """
        valid_classifiers = {'svc', 'mlp'}
        assert self.classifier_type in valid_classifiers

        if self.classifier_type == 'svc':
            model = svm.SVC(gamma='scale')

        elif self.classifier_type == 'mlp':
            model = nn.Linear(self.backbone_output_shape[0], constants.NUM_GAITS)

        else:
            raise AssertionError('No valid classifier')
        
        return model
    
    def build_command_head(self,):
        """Defines layers of the regression model and returns full model.
        """
        valid_regressors = {'svr', 'mlp'}
        assert self.regressor_type in valid_regressors

        if self.regressor_type == 'mlp': 
            lin1 = nn.Linear(self.backbone_output_shape[0], 128)
            lin2 = nn.Linear(128, constants.NUM_COMMANDS-1)

            return nn.Sequential(
                lin1,
                nn.ReLU(),
                lin2,
            )
        
        elif self.regressor_type == 'svr':
            return svm.SVR()