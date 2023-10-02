# %%
from gym import make
from httpx import head
import isaacgym
import torch
import numpy as np
import random
import time
import pathlib
import logging
import pickle as pkl
import os
import argparse

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
from go1_gym.envs.wrappers.no_yaw_wrapper import NoYawWrapper

from navigation.demo.demo_collector import DemoCollector
from navigation import constants
from navigation.vision.utils.image_processing import process_deployed
from navigation.sim.sim_utils import (
    create_xbox_controller,
    update_sim_view,
    render_forward_depth_rgb,
)
import gc

gc.collect()
torch.cuda.empty_cache()


def load_policy(logdir):
    body = torch.jit.load(logdir + "/checkpoints/body_latest.jit")
    import os

    adaptation_module = torch.jit.load(
        logdir + "/checkpoints/adaptation_module_latest.jit"
    )

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to("cpu"))
        action = body.forward(torch.cat((obs["obs_history"].to("cpu"), latent), dim=-1))
        info["latent"] = latent
        return action

    return policy


def load_config_from_policy(policy_path: pathlib.Path):
    with (policy_path / "parameters.pkl").open(mode="rb") as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    # if key2=='command_curriculum':
                    #     continue
                    setattr(getattr(Cfg, key), key2, value2)


def load_env(headless=False):
    load_config_from_policy(constants.WALK_GAIT_PATH)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.camera.use_camera = True

    # stair policy
    Cfg.env.num_observations = 71
    Cfg.env.num_scalar_observations = 71
    Cfg.env.observe_yaw = True

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    # episode lasts math.ciel(episode_length_s/sim_params.dt), dt=0.01999
    Cfg.env.episode_length_s = 1000
    Cfg.init_state.pos = [1.0, 1.0, 0.5]
    Cfg.terrain.num_rows = 6
    Cfg.terrain.num_cols = 6
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.mesh_type = "trimesh"

    Cfg.terrain.generated = False
    Cfg.terrain.generated_name = "0014"
    Cfg.terrain.generated_diff = "icra"
    Cfg.terrain.icra = True
    Cfg.terrain.maze_terrain = Cfg.terrain.generated

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    env_vel = VelocityTrackingEasyEnv(sim_device="cuda:0", headless=False, cfg=Cfg)
    env = NoYawWrapper(env_vel, yaw_bool=False)

    walk_policy = load_policy(constants.WALK_GAIT_PATH)
    climb_policy = load_policy(constants.CLIMB_GAIT_PATH)

    return env, env_vel, walk_policy, climb_policy


def play_go1(make_demo:bool, demo_folder: str, demo_name: str, headless: bool):
    if make_demo:
        assert demo_folder and demo_name, 'Need to specifiy a folder and name to use demo.'
        demo_collector = DemoCollector(demo_folder, demo_name)

    env, env_vel, walk_policy, climb_policy = load_env(headless=headless)

    obs = env.reset()
    joy = create_xbox_controller()

    curr_policy = constants.WALK_GAIT_NAME

    while True:
        with torch.no_grad():
            if curr_policy == constants.WALK_GAIT_NAME:
                actions = walk_policy(obs)
            elif curr_policy == constants.CLIMB_GAIT_NAME:
                actions = climb_policy(obs)

        update_sim_view(env)

        controls = joy.read()

        if controls['r_dpad']:
            curr_policy = constants.WALK_GAIT_NAME
            curr_policy_params = constants.WALK_GAIT_PARAMS
        elif controls['up_dpad']:
            curr_policy = constants.CLIMB_GAIT_NAME
            curr_policy_params = constants.CLIMB_GAIT_PARAMS
        elif controls['down_dpad']:
            curr_policy = constants.DUCK_GAIT_NAME
            curr_policy_params = constants.DUCK_GAIT_PARAMS

        # climb policy is not trained to go backward
        if controls['y_vel'] < 0 and curr_policy == constants.CLIMB_GAIT_NAME:
            curr_policy = constants.WALK_GAIT_NAME
            curr_policy_params = constants.WALK_GAIT_PARAMS

        # if 'x' button pressed on xbox, reset demo
        if controls['x_but'] != 0:
            logging.info("Resetting demo")
            env.reset()
            demo_collector.hard_reset()

        # read and update demo collect bool
        if controls['y_but'] != 0:
            if demo_collector.currently_collecting:
                logging.info('Saving demo')
                demo_collector.end_and_save_full_demo()
            else:
                logging.info("Starting log.")
                logging.info(f'Logging to: {demo_collector.save_path}')
                demo_collector.start_collecting()

        # update NN control
        if controls['l_trig'] != 0 and demo_collector.using_NN:
            logging.info("NN policy: off")
            demo_collector.using_NN = False

        if controls['r_trig'] != 0:
            if demo_collector.NN_ready:
                logging.info('NN policy: on')
                demo_collector.using_NN = True
                # TODO: turn nn on
            else:
                logging.info('NN not ready to be used')
            

        if demo_collector.using_NN:
            pass
            # TODO: implement NN

            # first, _ = render_first_third_imgs(env)
            # img = process_deployed(first)

            # # check if model memory is filled yet
            # if model.use_memory and not model.memory_filled:
            #     # if not, add to memory and get processed command

            #     # add to memory
            #     model.forward(img)

            # else:
            #     # if memory is filled, get predicted commands from NN
            #     commands, policy = model.forward(img)
            #     commands, policy = model._data_rescale(commands, policy)

            #     x_vel_cmd, y_vel_cmd, yaw_vel_cmd = commands

            #     print(
            #         "x_vel:",
            #         round(x_vel_cmd, 2),
            #         "y_vel:",
            #         round(y_vel_cmd, 2),
            #         "yaw:",
            #         round(yaw_vel_cmd, 2),
            #         "policy:",
            #         curr_policy,
            #     )

            # if policy == 1:
            #     curr_policy = "stairs"
            #     env.yaw_bool = True
            #     step_frequency_cmd = 2.0
            #     footswing_height_cmd = 0.30
            # elif policy == 0:
            #     curr_policy = "walk"
            #     env.yaw_bool = False
            #     step_frequency_cmd = 3.0
            #     footswing_height_cmd = 0.08

        # TODO: scale x_vel if using wtw


        env.commands[:, 0] = 0.0
        env.commands[:, 1] = controls['y_vel']
        env.commands[:, 2] = controls['yaw']
        env.commands[:, 3] = curr_policy_params['body_height_cmd']
        env.commands[:, 4] = curr_policy_params['step_frequency_cmd']
        env.commands[:, 5:8] = curr_policy_params['gait']
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = curr_policy_params['footswing_height_cmd']
        env.commands[:, 10] = curr_policy_params['pitch_cmd']
        env.commands[:, 11] = curr_policy_params['roll_cmd']
        env.commands[:, 12] = curr_policy_params['stance_width_cmd']
        obs, rew, done, info = env.step(actions)

        # collect commands every timestep -> will be averages
        if demo_collector.currently_collecting and time.time() - demo_collector.timer >= demo_collector.how_often_capture_data:
            rgb_img, depth_img = render_forward_depth_rgb(env)
            frame_data = {
                "y_vel": controls['y_vel'],
                "yaw": controls['yaw'],
                "gait": curr_policy_params['gait'],
                "forward_rgb": rgb_img,
                "forward_depth": depth_img

            }
            demo_collector.add_data_to_partial_run(frame_data)

        i += 1


def parse_args():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train a full vision model.")

    parser.add_argument(
        "--make_demo",
        action="store_true",
        help="Whether or not you will record a demo.",
    )
    parser.add_argument("--demo_folder", type=str)
    parser.add_argument("--demo_name", type=str)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Call if you dont want to render the visualization",
    )

    args = parser.parse_args()
    return args


def main(make_demo, demo_folder, demo_name, headless):
    play_go1(make_demo, demo_folder, demo_name, headless=headless)


if __name__ == "__main__":
    args = parse_args()
    make_demo = args.make_demo
    demo_name = args.demo_name
    demo_folder = args.demo_folder
    headless = args.headless
    main(make_demo, demo_folder, demo_name, headless)
