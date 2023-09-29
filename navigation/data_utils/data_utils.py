from navigation import constants, utils

import os
import logging
import pickle as pkl
import gzip
import pathlib


def load_single_demo_run(demo_run_path: pathlib.Path, empty_data_dict: dict):
    num_partial_logs = 0
    for filename in os.listdir(demo_run_path):
        if constants.DEMO_PARTIAL_RUN_LABEL in filename:
            num_partial_logs += 1

    logging.info(f"Num partial logs: {num_partial_logs}")
    for i in range(1, num_partial_logs + 1):
        with gzip.open(
            demo_run_path / utils.make_partial_run_label(i), "rb"
        ) as file:
            p = pkl.Unpickler(file)
            partial_log = p.load()
            for key in empty_data_dict:
                empty_data_dict[key] += partial_log[key]


def load_demo_data(demo_path: pathlib.Path):
    run_list = os.listdir(demo_path)

    for run_name in run_list:

        # skip folder if it isnt a run
        if constants.DEMO_RUN_LABEL not in run_name:
            continue

        
