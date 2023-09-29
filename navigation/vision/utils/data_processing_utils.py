from navigation import constants
from navigation.demo import utils
from navigation.vision.utils.image_processing import process_image
from navigation.vision.vision_model import VisionModel

import os
import logging
import pickle as pkl
import gzip
import pathlib
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


def load_single_demo_run(demo_run_path: pathlib.Path):
    run_data_dict = utils.get_empty_demo_data()

    num_partial_logs = 0
    for filename in os.listdir(demo_run_path):
        if constants.DEMO_PARTIAL_RUN_LABEL in filename:
            num_partial_logs += 1

    for i in range(1, num_partial_logs + 1):
        with gzip.open(
            demo_run_path / utils.make_partial_run_label(i), "rb"
        ) as file:
            p = pkl.Unpickler(file)
            partial_log = p.load()
            for key in run_data_dict:
                run_data_dict[key] += partial_log[key]

    return run_data_dict


def load_demo_data(demo_path: pathlib.Path):
    demo_data_dict = utils.get_empty_demo_data()

    run_list = os.listdir(demo_path)
    for run_name in run_list:
        assert constants.DEMO_RUN_LABEL == run_name[:len(constants.DEMO_RUN_LABEL)]
    

    # loop through runs chronologically
    logging.info('Loading demo runs.')
    for i in tqdm(range(1, len(run_list)+1)):
        curr_run_name = utils.make_run_label(i)
        curr_run_data = load_single_demo_run(demo_path / curr_run_name)

        for key in curr_run_data:
            demo_data_dict[key] += curr_run_data[key]

    return demo_data_dict


def make_dataloaders_from_demo(demo_data: dict, batch_size: int, train_perc: float):
    logging.info('Making dataloaders')

    rgb_imgs = demo_data[constants.DEMO_RGB_KEY]
    all_commands = np.array(demo_data[constants.DEMO_COMMAND_KEY])
    
    paired_data = []

    logging.info('Processing images and making data pairs')
    for rgb_img, command in tqdm(zip(rgb_imgs, all_commands), total=len(rgb_imgs)):
        #processed_img = image_processor(images=rgb_img, return_tensors="pt")['pixel_values'][0]
        processed_img = process_image(rgb_img)
        gait_label = command[-1]
        comms = command[:-1]
        data_point = [processed_img, gait_label, comms]
        paired_data.append(data_point)

    train_size = int(train_perc*len(rgb_imgs))
    test_size = len(rgb_imgs)-train_size

    train_split, test_split = random_split(paired_data, [train_size, test_size])

    trainloader = DataLoader(train_split, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True)
    testloader = DataLoader(test_split, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=True)

    return trainloader, testloader


def make_dino_feature_video():
    vision_model = VisionModel(inference_type='offline', num_cameras=constants.NUM_CAMERAS)
    

