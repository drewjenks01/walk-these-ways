from navigation.demo import utils
from navigation import constants

import logging
import os
import pickle as pkl
import shutil
import time


class DemoCollector:
    def __init__(self, demo_folder: str, demo_name: str):
        self.demo_name = demo_name

        # determine demo folder, create if doesnt exists
        self.save_path_base = constants.DEMO_BASE_PATH / demo_folder / self.demo_name
        if not self.save_path_base.exists():
            self.run_count = 1
            self.save_path_base.mkdir(parents=True)
        else:
            self.run_count = len(os.listdir(self.save_path_base)) + 1

        # create the demo run folder
        self.save_path = self.save_path_base / utils.make_run_label(self.run_count)
        self.save_path.mkdir()

        # define inital partial run file
        self.partial_run_count = 1
        self.save_path = self.save_path / utils.make_partial_run_label(
            self.partial_run_count
        )
        self.timestep_to_save_partial = 20
        self.partial_demo_timestep = 0

        # initialize data to store during demo
        self.demo_data = utils.get_empty_demo_data()

        self.fps = constants.FPS
        self.how_often_capture_data = 1/self.fps
        self.timer = 0.0

        self.currently_collecting = False

    def start_collecting(self):
        self.currently_collecting = True
        self.timer = time.time()

    def add_data_to_partial_run(self, data: dict) -> None:
        assert self.currently_collecting, 'Make sure to call start_collecting()'
        assert data.keys() == self.get_current_demo_data().keys(), f'{data.keys()} != {self.get_current_demo_data().keys()} '
        for key, val in data.items():
            self.demo_data[key].append(val)
        
        self.partial_demo_timestep += 1

        if self.partial_demo_timestep % self.timestep_to_save_partial == 0:
            self.save_partial_demo()
        
        self.start_collecting() # reset timer

    def save_partial_demo(self) -> None:
        self._reset_demo(partial_save=True)

    def end_and_save_full_demo(self) -> None:
        # save current demo
        self._reset_demo(partial_save=True, soft_reset=True)

    def get_current_demo_data(self) -> dict:
        return self.demo_data

    def hard_reset(self):
        self._reset_demo(hard_reset=True)

    def _reset_demo(
        self,
        partial_save: bool = False,
        soft_reset: bool = False,
        hard_reset: bool = False,
    ) -> None:
        assert (
            partial_save or hard_reset or soft_reset
        ), "One of the inputs must be true."

        if partial_save:
            # store the current demo data as a partial run
            if not self.save_path.parent.exists():
                self.save_path.parent.mkdir()
            logging.info(f"Storing partial demo of length={len(self.demo_data[constants.COMMAND_KEY])} to {self.save_path}")
            with self.save_path.open(mode="wb") as file:
                pkl.dump(self.demo_data, file)

            self.demo_data = utils.get_empty_demo_data()

            # update demo file name
            self.partial_run_count += 1

        if hard_reset or soft_reset:
            if hard_reset:
                # delete current run folder
                self._delete_partial_saves()
            else:
                self.run_count+=1

            # reset demo data
            self.demo_data = utils.get_empty_demo_data()

            # reset partial run label
            self.partial_run_count = 1

            self.currently_collecting=False

        self.save_path = self.save_path_base / utils.make_run_label(self.run_count) / utils.make_partial_run_label(
            self.partial_run_count
        )
        self.partial_demo_timestep = 0

    def _delete_partial_saves(self):
        # delete current run folder
        logging.info(f'Deleting {self.save_path.parent}')
        shutil.rmtree(self.save_path.parent)

        # re-create run folder
        self.save_path.parent.mkdir()
