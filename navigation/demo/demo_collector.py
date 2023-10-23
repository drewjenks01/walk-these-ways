from navigation.demo import utils
from navigation import constants

import logging
import os
import pickle as pkl
import shutil
import time
from PIL import Image


class DemoCollector:
    def __init__(self, demo_folder: str, demo_name: str):
        self.demo_name = demo_name

        # determine demo folder, create if doesnt exists
        self.save_path_base = constants.RAW_DEMOS_PATH / demo_folder / self.demo_name
        if not self.save_path_base.exists():
            self.run_count = 1
            self.save_path_base.mkdir(parents=True)
        else:
            self.run_count = len(os.listdir(self.save_path_base)) + 1

        # create the demo run folder
        self.save_path = self.save_path_base / utils.make_run_label(self.run_count)
    
        # define inital partial run file
        self.run_image_count = 0

        # initialize data to store during demo
        self.demo_command_data = utils.get_empty_demo_command_data()

        self.fps = constants.FPS
        self.how_often_capture_data = 1
        self.timer = 0.0

        self.currently_collecting = False

    def create_demo_folders(self):
        # create run folder
        if not self.save_path.exists():
            self.save_path.mkdir()

        # create image folders
        for img_suffix in constants.CAMERA_IMAGE_NAMES:
            if not (self.save_path / img_suffix).exists():
                (self.save_path / img_suffix).mkdir()

    def start_collecting(self):
        self.currently_collecting = True
        self.create_demo_folders()
        self._reset_timer()

    def _reset_timer(self):
        self.timer= time.time()

    @staticmethod
    def save_commands_to_file(data, filepath):
        with filepath.open(mode="wb") as file:
                pkl.dump(data, file)

    def save_image_to_file(data, filepath):
        img = Image.fromarray(data)
        img.save(filepath)


    def add_data_to_run(self, command_data: dict, image_data: dict):
        assert self.currently_collecting, 'Make sure to call start_collecting()'

        # save rgb and depth image (if exists)
        for camera_image_name in constants.CAMERA_IMAGE_NAMES:
            if image_data[camera_image_name] is not None:
                DemoCollector.save_image_to_file(
                    data=image_data[camera_image_name],
                    filepath=self.save_path / camera_image_name / f'{self.run_image_count}.jpg'
                )
        # increment image count -- assumes theres always at least 1 image
        self.run_image_count += 1

        if self.run_image_count % 50 == 0:
            logging.info(f'{self.run_image_count} images saved.')

        # add command data to running list
        for command_key in command_data:
            self.demo_command_data[command_key].append(command_data[command_key])

        self._reset_timer() # reset timer

    def end_and_save_demo(self):
        # save the command data to file
        DemoCollector.save_commands_to_file(
            data=self.demo_command_data,
            filepath=self.save_path / 'command_data.pkl'
            )

        # reset demo and prepare for the next
        self.reset_demo()

    def reset_demo(self, reset_current=False):
        logging.info('Resetting demo')

        # turn collecting off
        self.currently_collecting = False
        
        # delete any data saved so far and dont change save path
        if reset_current:
            logging.info('Deleting current demo')
            shutil.rmtree(self.save_path)

        else:
            self.run_count+=1
            self.save_path = self.save_path.parent / utils.make_run_label(self.run_count) 


        # reset command data
        self.demo_command_data = utils.get_empty_demo_command_data()

        logging.info(f'Current demo folder: {self.save_path}')