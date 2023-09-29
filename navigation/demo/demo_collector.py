from navigation.demo import constants, utils

import logging
import os
import pickle as pkl
import shutil

class DemoCollector:

    def __init__(self, demo_name: str):
        self.demo_name = demo_name

        # determine demo folder, create if doesnt exists
        self.save_path = constants.BASE_PATH / self.demo_name
        if not self.save_path.exists():
            run_count = 1
            self.save_path.mkdir()
        else:
            run_count = len(os.listdir(self.save_path)) + 1

        # create the demo run folder
        self.save_path = self.save_path / utils.make_run_label(run_count)
        self.save_path.mkdir()

        # define inital partial run file
        self.partial_run_count = 1
        self.save_path = self.save_path / utils.make_partial_run_label(self.partial_run_count)
        
        # initialize data to store during demo
        self.demo_data = utils.get_empty_demo_data()


    def add_data_to_partial_run(self, data: dict) -> None:
        assert data.keys() == self.get_current_demo_data().keys()
        for key, val in data:
            self.demo_data[key].append(val)

    def end_partial_demo(self, save: bool) -> None:
        self._reset_partial_demo(save=save)

    def get_current_demo_data(self) -> dict:
        return self.demo_data

    def hard_reset(self):
       self._reset_partial_demo(save=False, hard_reset=True)

    def _reset_partial_demo(self, save: bool, hard_reset: bool) -> None:
        assert save or hard_reset, 'One of the inputs must be true.'
        
        if save:
            # store the current demo data as a partial run
            logging.info(f'Storing partial demo: {self.save_path}')
            with self.save_path.open(mode='w') as file:
                pkl.dump(self.demo_data, file)

            # update demo file name
            self.partial_run_count += 1

        elif hard_reset:
            # delete current run folder
            self._delete_partial_saves()

            # reset demo data
            self.demo_data = utils.get_empty_demo_data()

            # reset partial run label
            self.partial_run_count = 1

        else:
            raise AssertionError('Reset partial demo called, but not doing anything.\
                                Check function inputs.')
        
        self.save_path = self.save_path / utils.make_partial_run_label(self.partial_run_count)

    def _delete_partial_saves(self):

        # delete current run folder
        shutil.rmtree(self.save_path.parent)

        # re-create run folder
        self.save_path.parent.mkdir()