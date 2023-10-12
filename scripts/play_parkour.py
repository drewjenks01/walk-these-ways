# %%
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
from go1_gym.envs.wrappers.multi_gait_wrapper import MultiGaitWrapper

from navigation.demo.demo_collector import DemoCollector
from navigation.demo.utils import get_empty_demo_data
from navigation import constants
from navigation.vision.utils.image_processing import process_deployed
from navigation.sim.sim_utils import (
    create_xbox_controller,
    update_sim_view,
    
)
import gc

from parkour.utils import task_registry

gc.collect()
torch.cuda.empty_cache()


def load_policy(logdir, parkour: bool = False):
    body = torch.jit.load(logdir / "checkpoints/body_latest.jit")
    import os

    adaptation_module = torch.jit.load(
        logdir / "checkpoints/adaptation_module_latest.jit"
    )

    def policy(obs, info={}):
        latent = adaptation_module.forward(obs["obs_history"].to("cpu"))
        action = body.forward(torch.cat((obs["obs_history"].to("cpu"), latent), dim=-1))
        info["latent"] = latent
        return action
    
    def parkour_policy(obs, depth_img):
        obs_student = obs[:,:53].clone()[:, 6:8]
        depth_latent_and_yaw = adaptation_module.forward(depth_img.to("cpu"), obs_student)
        depth_latent = depth_latent_and_yaw[:,:-2]
        yaw = depth_latent_and_yaw[:,-2:]
        action = body.forward(obs.detach(), hist_encoding=True, scandots_latent = depth_latent)
        return action

    if parkour:
        return parkour_policy
    return policy


def load_env(headless=False):
    env_cfg, train_cfg = task_registry.get_cfgs(name='a1')

    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
                                    "rough slope up": 0.0,
                                    "rough slope down": 0.0,
                                    "rough stairs up": 0., 
                                    "rough stairs down": 0., 
                                    "discrete": 0., 
                                    "stepping stones": 0.0,
                                    "gaps": 0., 
                                    "smooth flat": 0,
                                    "pit": 0.0,
                                    "wall": 0.0,
                                    "platform": 0.,
                                    "large stairs up": 0.,
                                    "large stairs down": 0.,
                                    "parkour": 0.2,
                                    "parkour_hurdle": 0.2,
                                    "parkour_flat": 0.,
                                    "parkour_step": 0.2,
                                    "parkour_gap": 0.2, 
                                    "demo": 0.2}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    # prepare env
    depth_latent_buffer = []
    env, _ = task_registry.make_env(name='a1', env_cfg=env_cfg)

    # load policy
    train_cfg.runner.resume = True

    return env, env_cfg, train_cfg

def play_go1(headless: bool):

    env, env_cfg, train_cfgs = load_env(headless=headless)
    obs = env.get_observations()

    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name='a1', train_cfg=train_cfg, return_log_dir=True)
    policy_jit = torch.jit.load('navigation/data_and_models/trained_controllers/parkour_depth/checkpoints/051-42-14000-base_jit.pt', device=env.device)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    joy = create_xbox_controller()

    while True:

        if infos["depth"] is not None:
            depth_latent = torch.ones((env_cfg.env.num_envs, 32), device=env.device)
            actions, depth_latent = policy_jit(obs.detach(), True, infos["depth"], depth_latent)
        else:
            depth_buffer = torch.ones((env_cfg.env.num_envs, 58, 87), device=env.device)
            actions, depth_latent = policy_jit(obs.detach(), False, depth_buffer, depth_latent)

        actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
        obs, _, rews, dones, infos = env.step(actions.detach())
        print("time:", env.episode_length_buf[env.lookat_id].item() / 50, 
              "cmd vx", env.commands[env.lookat_id, 0].item(),
              "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), )
        
        id = env.lookat_id

        update_sim_view(env)

        controls = joy.read()

        # # climb policy is not trained to go backward
        # if controls['y_vel'] < 0 and curr_policy == constants.CLIMB_GAIT_NAME:
        #     curr_policy = constants.WALK_GAIT_NAME
        #     curr_policy_params = constants.WALK_GAIT_PARAMS

        # # if 'x' button pressed on xbox, reset demo
        # if controls['x_but'] != 0 and make_demo:
        #     logging.info("Resetting demo")
        #     env.reset()
        #     if make_demo:
        #         demo_collector.hard_reset()

        # # read and update demo collect bool
        # if controls['y_but'] != 0 and make_demo:
        #     if demo_collector.currently_collecting:
        #         logging.info('Saving demo')
        #         demo_collector.end_and_save_full_demo()
        #     else:
        #         logging.info("Starting log.")
        #         logging.info(f'Logging to: {demo_collector.save_path}')
        #         demo_collector.start_collecting()

        # # update NN control
        # if make_demo and controls['l_trig'] != 0 and demo_collector.using_NN:
        #     logging.info("NN policy: off")
        #     demo_collector.using_NN = False

        # if controls['r_trig'] != 0 and make_demo:
        #     if demo_collector.NN_ready:
        #         logging.info('NN policy: on')
        #         demo_collector.using_NN = True
        #         # TODO: turn nn on
        #     else:
        #         logging.info('NN not ready to be used')

        # # collect commands every timestep -> will be averages

        # if make_demo and demo_collector.currently_collecting and time.time() - demo_collector.timer >= demo_collector.how_often_capture_data :

        #     frame_data = get_empty_demo_data()
        #     frame_data["Commands"]= [controls['y_vel'],controls['yaw'],curr_policy_params['gait']]
        #     frame_data["forward_rgb"]= rgb_img
        #     frame_data['forward_depth'] = depth_img
        #     demo_collector.add_data_to_partial_run(frame_data)
            

        # if using_nn:
        #     pass
        #     # TODO: implement NN

        #     # first, _ = render_first_third_imgs(env)
        #     # img = process_deployed(first)

        #     # # check if model memory is filled yet
        #     # if model.use_memory and not model.memory_filled:
        #     #     # if not, add to memory and get processed command

        #     #     # add to memory
        #     #     model.forward(img)

        #     # else:
        #     #     # if memory is filled, get predicted commands from NN
        #     #     commands, policy = model.forward(img)
        #     #     commands, policy = model._data_rescale(commands, policy)

        #     #     x_vel_cmd, y_vel_cmd, yaw_vel_cmd = commands

        #     #     print(
        #     #         "x_vel:",
        #     #         round(x_vel_cmd, 2),
        #     #         "y_vel:",
        #     #         round(y_vel_cmd, 2),
        #     #         "yaw:",
        #     #         round(yaw_vel_cmd, 2),
        #     #         "policy:",
        #     #         curr_policy,
        #     #     )

        #     # if policy == 1:
        #     #     curr_policy = "stairs"
        #     #     env.yaw_bool = True
        #     #     step_frequency_cmd = 2.0
        #     #     footswing_height_cmd = 0.30
        #     # elif policy == 0:
        #     #     curr_policy = "walk"
        #     #     env.yaw_bool = False
        #     #     step_frequency_cmd = 3.0
        #     #     footswing_height_cmd = 0.08

        # # TODO: scale x_vel if using wtw

        # if curr_policy != 'parkour':
        #     env.commands[:, 0] = controls['y_vel']*1.5
        #     env.commands[:, 1] = 0.0
        #     env.commands[:, 2] = controls['yaw']*1.5
        #     env.commands[:, 3] = curr_policy_params['body_height_cmd']
        #     env.commands[:, 4] = curr_policy_params['step_frequency_cmd']
        #     env.commands[:, 5:8] = curr_policy_params['gait']
        #     env.commands[:, 8] = 0.5
        #     env.commands[:, 9] = curr_policy_params['footswing_height_cmd']
        #     env.commands[:, 10] = curr_policy_params['pitch_cmd']
        #     env.commands[:, 11] = curr_policy_params['roll_cmd']
        #     env.commands[:, 12] = curr_policy_params['stance_width_cmd']
        #     env.yaw_bool = curr_policy_params['yaw_obs_bool']
        # obs, rew, done, info = env.step(actions)

def parse_args():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train a full vision model.")

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Call if you dont want to render the visualization",
    )

    args = parser.parse_args()
    return args


def main(headless):
    play_go1(headless=headless)


if __name__ == "__main__":
    args = parse_args()
    headless = args.headless
    main(headless)
