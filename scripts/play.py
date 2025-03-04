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
from navigation.demo.utils import get_empty_demo_command_data, get_empty_demo_image_data
from navigation import constants
from navigation.vision.models.get_models import get_models
from navigation.sim.sim_utils import (
    create_xbox_controller,
    update_sim_view,
    
)
import gc

gc.collect()
torch.cuda.empty_cache()
import inspect

def load_policy(logdir):
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
    return policy

def load_parkour_policy(logdir):

    policy_jit = torch.jit.load('navigation/data_and_models/trained_controllers/parkour_depth/checkpoints/body_latest.jit', map_location=constants.DEVICE)
    vision_backbone = torch.jit.load('navigation/data_and_models/trained_controllers/parkour_depth/checkpoints/adaptation_module_latest.jit', map_location=constants.DEVICE)
    print(vision_backbone.code)

    def policy(obs, depth_img):
        print(obs)
        proprio = obs['obs'][:, :Cfg.env.n_proprio]
        depth_latent_and_yaw = vision_backbone(depth_img, proprio)
        depth_latent  = depth_latent_and_yaw[:, :-2]
        print(obs['obs'].shape, depth_latent.shape)
        return policy_jit(obs['obs'], depth_latent)

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

    Cfg.terrain.generated = True
    Cfg.terrain.generated_name = "0014"
    Cfg.terrain.generated_diff = "easy"
    Cfg.terrain.icra = False
    Cfg.terrain.maze_terrain = Cfg.terrain.generated

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    Cfg.perception.image_horizontal_fov = 110
    Cfg.perception.image_height = 160
    Cfg.perception.image_width = 220
    Cfg.perception.camera_names = ["forward", "downward"]
    Cfg.perception.camera_poses = [[0.3, 0, 0], [0.3, 0, -0.08]]
    Cfg.perception.camera_rpys = [[0.0, 0, 0], [0, -3.14 / 2, 0]]
    Cfg.perception.compute_depth = True
    Cfg.perception.compute_rgb = True

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = NoYawWrapper(env, False)

    walk_policy = load_policy(constants.WALK_GAIT_PATH)
    climb_policy = load_policy(constants.CLIMB_GAIT_PATH)

    return env, walk_policy, climb_policy

def play_go1(demo_folder: str, demo_name: str, headless: bool, model_name: str):
    if demo_folder and demo_name:
        make_demo = True
        demo_collector = DemoCollector(demo_folder, demo_name)
    else:
        make_demo = False
        logging.warning('No demo folder or name provided. Demo collection will not work.')

    env, walk_policy, climb_policy = load_env(headless=headless)

    obs = env.reset()
    joy = create_xbox_controller()

    curr_policy = constants.WALK_GAIT_NAME
    curr_policy_params = constants.WALK_GAIT_PARAMS

    fps_timer = time.time()
    init_img=True

    using_nn = False
    if model_name:
        models = get_models(navigator='vint', pretrained=True, topomap_folder='sim_maze')
        navigator = models['navigator']


    while True:
        env.render()
        
        if fps_timer >= 1/constants.FPS or init_img:
            rgb_imgs = env.get_rgb_images(env_ids = [0])
            depth_imgs = env.get_depth_images(env_ids = [0])
            rgb_img = rgb_imgs['forward']
            depth_img = depth_imgs['forward']

            if init_img:
                init_img=False

        with torch.no_grad():
            if curr_policy == constants.WALK_GAIT_NAME:
                # env.change_current_controller(constants.WALK_GAIT_NAME)
                env.yaw_bool = False
                actions = walk_policy(obs)
            elif curr_policy == constants.CLIMB_GAIT_NAME:
                # env.change_current_controller(constants.WALK_GAIT_NAME)
                env.yaw_bool = True
                actions = climb_policy(obs)
            elif curr_policy == constants.DUCK_GAIT_NAME:
                # env.change_current_controller(constants.WALK_GAIT_NAME)
                env.yaw_bool = False
                actions = walk_policy(obs)


        update_sim_view(env)
        #update_viewer_cam(env)

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
        if controls['x_but'] != 0 and make_demo:
            logging.info("Resetting demo")
            env.reset()
            if make_demo:
                demo_collector.reset_demo(reset_current=True)

        # read and update demo collect bool
        if controls['y_but'] != 0 and make_demo:
            if demo_collector.currently_collecting:
                logging.info('Saving demo')
                demo_collector.end_and_save_demo()
            else:
                logging.info("Starting log.")
                logging.info(f'Logging to: {demo_collector.save_path}')
                demo_collector.start_collecting()

        # update NN control
        if make_demo and controls['l_trig'] != 0 and using_nn:
            logging.info("NN policy: off")
            using_nn = False

        if controls['r_trig'] != 0 and not using_nn:
            using_nn = True
            logging.info('NN policy: on')


        # collect commands every timestep -> will be averages

        if make_demo and demo_collector.currently_collecting and time.time() - demo_collector.timer >= demo_collector.how_often_capture_data :

            command_data = get_empty_demo_command_data()
            command_data['y_vel'] = controls['y_vel']
            command_data['yaw'] = controls['yaw']
            command_data['gait'] = curr_policy_params['gait']

            image_data = get_empty_demo_image_data()
            image_data[constants.FORWARD_RGB_CAMERA] = rgb_img.squeeze(0).detach().cpu().numpy().astype(np.uint8)
            image_data[constants.FORWARD_DEPTH_CAMERA] = depth_img.squeeze(0).detach().cpu().numpy().astype(np.uint8)
            
            demo_collector.add_data_to_run(command_data, image_data)
            

        if using_nn:
            outs = navigator(rgb_img.squeeze(0).detach().cpu().numpy().astype(np.uint8))
            controls['y_vel'] = outs['y_vel']
            controls['yaw'] = outs['yaw']

            logging.info(f'y_vel: {controls["y_vel"]}, yaw: {controls["yaw"]}')
            
            # TODO: add gait control

        if curr_policy != 'parkour':
            env.commands[:, 0] = controls['y_vel']
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = controls['yaw']
            env.commands[:, 3] = curr_policy_params['body_height_cmd']
            env.commands[:, 4] = curr_policy_params['step_frequency_cmd']
            env.commands[:, 5:8] = curr_policy_params['gait']
            env.commands[:, 8] = 0.5
            env.commands[:, 9] = curr_policy_params['footswing_height_cmd']
            env.commands[:, 10] = curr_policy_params['pitch_cmd']
            env.commands[:, 11] = curr_policy_params['roll_cmd']
            env.commands[:, 12] = curr_policy_params['stance_width_cmd']
            env.yaw_bool = curr_policy_params['yaw_obs_bool']
        obs, rew, done, info = env.step(actions)

def parse_args():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train a full vision model.")

    parser.add_argument("--demo_folder", type=str)
    parser.add_argument("--demo_name", type=str)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Call if you dont want to render the visualization",
    )
    parser.add_argument("--model_name", type=str)

    args = parser.parse_args()
    return args


def main(args):
    play_go1(
        args.demo_folder, 
        args.demo_name,
        args.headless,
        args.model_name
        )
def main(args):
    play_go1(
        args.demo_folder, 
        args.demo_name,
        args.headless,
        args.model_name
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
    main(args)

# %%
