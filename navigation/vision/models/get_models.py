
import logging
from networkx import radius
import torch
from typing import List
from PIL import Image as PILImage
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
import torch.nn as nn
import yaml
import numpy as np
from navigation.vision.models.base_model import BaseModel
from navigation.vision.models.navigators.vint_navigator import ViNTNavigator

from navigation.vision.models.navigators.vint_utils.vint import ViNT
from navigation import constants


def get_models(navigator = '', gait_classifier = '', **kwargs):
    assert navigator or gait_classifier, 'Pick a type of model to load'
    models = {'navigator':None, 'gait_classifier': None, 'combo': None}

    # if the same, then that means they are using the same backbone
    # we dont need to load them both
    if navigator == gait_classifier:
        key = 'combo'
    elif navigator:
        key = 'navigator'
    elif gait_classifier:
        key = 'gait_classifier'

    # navigators
    if navigator == 'vint':
        models[key] = ViNTNavigator(**kwargs)

    # gait classifiers

    return models