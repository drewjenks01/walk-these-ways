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

from navigation.sim.sim_utils import (
    create_xbox_controller,
    update_sim_view,
    
)
import gc

from parkour.utils import task_registry, get_args

gc.collect()
torch.cuda.empty_cache()

def load_env():
    env_cfg, train_cfg = task_registry.get_cfgs(name='a1')
    print(env_cfg.sim)

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

    env = MultiGaitWrapper(Cfg)

    walk_policy = load_policy(constants.WALK_GAIT_PATH)
    climb_policy = load_policy(constants.CLIMB_GAIT_PATH)
    parkour_depth_policy = load_policy(constants.PARKOUR_DEPTH_GAIT_PATH)

    return env, walk_policy, climb_policy, parkour_depth_policy

def play_go1(args):

    env, env_cfg, train_cfgs = load_env()
    obs = env.get_observations()

    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name='a1', train_cfg=train_cfg, return_log_dir=True, args=args)
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


def main(args):
    play_go1(args)


if __name__ == "__main__":
    args = get_args()
    main(args)
