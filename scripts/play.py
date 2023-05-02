#%%
import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl
import os

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
from go1_gym.envs.wrappers.no_yaw_wrapper import NoYawWrapper

import pandas

from navigation.commandnet.commandNN import CommandNet
from navigation.utils.image_processing import process_realsense
from navigation.utils.sim_utils import create_xbox_controller, update_sim_view, render_first_third_imgs
from navigation.utils.demo import Demo
from navigation.utils.video_recorder import VideoRecorder
import gc
gc.collect()
torch.cuda.empty_cache()


def load_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs, info={}):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy


def load_env(label, headless=False):
    import pathlib
    this_file_path = pathlib.Path(__file__).parent.parent.resolve()
    dirs = glob.glob(f"{this_file_path}/../runs/{label}/*")

    walk_dir = f"/home/andrewjenkins/walk-these-ways/runs/{label}/pretrain-v0"
    stair_dir = f"/home/andrewjenkins/walk-these-ways/runs/{label}/pretrain-stairs"

    # print(dirs, label)
    # logdir = sorted(dirs)[0]

    with open(walk_dir + "/parameters.pkl", 'rb') as file:
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
    Cfg.env.num_observations=71
    Cfg.env.num_scalar_observations=71
    Cfg.env.observe_yaw=True

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    # episode lasts math.ciel(episode_length_s/sim_params.dt), dt=0.01999
    Cfg.env.episode_length_s = 1000
    Cfg.init_state.pos=[1.,1.,0.5]
    Cfg.terrain.num_rows = 6
    Cfg.terrain.num_cols = 6
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.mesh_type='trimesh'

    Cfg.terrain.generated = False
    Cfg.terrain.generated_name = '0007'
    Cfg.terrain.generated_diff = 'icra'
    Cfg.terrain.icra = True
    Cfg.terrain.maze_terrain=Cfg.terrain.generated

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    env_vel = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = NoYawWrapper(env_vel,yaw_bool=False)
    

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    walk_policy = load_policy(walk_dir)
    stair_policy = load_policy(stair_dir)

    return env,env_vel, walk_policy, stair_policy

def auto_test(test_sucess, test_time, dist_travel, model_name):
    log_root=f'/home/andrewjenkins/walk-these-ways/navigation/robot_demos/sim_multi_results/{model_name}/{Cfg.terrain.generated_diff}.{Cfg.terrain.generated_name}'
     # get list of files
    num_runs = len(os.listdir(log_root))
    curr_run = num_runs+1
    os.makedirs(log_root+f'/run{curr_run}')
    log_filename = log_root+f'/run{curr_run}/log.pkl'

    log={'Success':test_sucess,'Time':test_time,'Distance':dist_travel}
    print('Auto test log:',log)

    # Store data (serialize)
    with open(log_filename, 'wb') as handle:
        pkl.dump(log, handle, protocol=pkl.HIGHEST_PROTOCOL)



def manual_test(map_type, gait_type, time, energy):
    print('\n')
    inp = input('Test success?')

    # chekc if test is sucess
    if inp=='y':
        success=1
    else:
        success=0

    # read in previous tests
    
    tests= pandas.read_pickle(f'navigation/robot_demos/sim_trials/{map_type}.pkl')
    new={'Success':success,'Time':time,'Energy': energy, 'Gait':gait_type}
    new_test=pandas.DataFrame([new])
    tests=pandas.concat([tests,new_test])

    print('Logged new test:\n',tests.iloc[-1])
    tests.to_pickle('navigation/commandnet/manual_tests.pkl')
    from collections import Counter
    counts = Counter(list(tests['Mode']))
    print('Tests counts:', counts)


def play_go1(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os
    from matplotlib import pyplot as plt
    from isaacgym import gymapi, torch_utils
    

    label = "gait-conditioned-agility"

    env, env_vel, walk_policy, stair_policy = load_env(label, headless=headless)

    num_eval_steps = 25000
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}


    body_height_cmd = 0.0

    # decrease to 2 or 2.5
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.35

    obs = env.reset()
    print('obs shape',obs['obs_history'].shape)

    joy = create_xbox_controller()


    collect_data=False  # starts when Y pressed on controller
    execute_script=False
    record_video=False

    demo_folder=f'sim_trials/{env.cfg.terrain.generated_diff}'



   #
   # 
   #  new_video = VideoRecorder()

    i=0

    demo_folder = 'sim_multi'
    model_name = 'resnet18'
    model = CommandNet(model_name=model_name,demo_folder=demo_folder)
    model.load_trained()

    test_started=False
    test_time=0
    
    curr_policy = 'walk'
    policy=0

    import time

    while True:

        with torch.no_grad():

            if curr_policy=='walk':
                actions = walk_policy(obs)
            elif curr_policy=='stairs':
                actions=stair_policy(obs)

        update_sim_view(env)

        x_vel_cmd, y_vel_cmd, yaw_vel_cmd, a_btn, foot_down, foot_up, y_cmd, x_cmd, ltrig, rtrig, b,thumbs, xdpad, ydpad = joy.read()

        if x_vel_cmd<0 and curr_policy=='stairs':
            thumbs=-1

        # read in X and terminate if pressed
        if x_cmd!=0:
            env.reset()
            print('Resetting demo')
            collect_data=False
            new_demo=Demo(folder=demo_folder)
            
     
        # read and update demo collect bool
        if y_cmd!=0:
            new_demo = Demo(folder=demo_folder)
            demo_type = 'manual'        # manual or demo

            collect_data=True

            if demo_type == 'demo':

                 # demo collection
                new_demo.prepare_demo(terrain = f'{env.cfg.terrain.generated_diff}/{env.cfg.terrain.generated_name}')
          
            elif demo_type=='manual':
                manual_gait = 'stair'     # wtw, stair, multi
                print(f'Starting manual test for : Gait={manual_gait}, env={env.cfg.terrain.generated_diff}')

                # manual testing
                new_demo.prepare_gait_test()
                start_time=time.time()
        
        if a_btn!=0:
            print('Saving demo...')
            collect_data=False

            if demo_type=='demo':
                new_demo.save_demo()

            elif demo_type=='manual':
                end_time=time.time()
            
                collect_data=False

                test_time = end_time-start_time
                print('\n')
                inp = input('Test success?')

                # check if test is sucess
                if inp=='y':
                    success=1
                else:
                    success=0
                        
                new_demo.save_gait_test(time=test_time,success=success)
            del new_demo
            new_demo=Demo(folder=demo_folder)
            

        # turn off NN control
        if ltrig!=0:
            print('Action script off')
            execute_script=False

        if b!=0:
            test_started=True
            test_start_time=time.time()
            execute_script=True

        if thumbs>0:
            curr_policy='stairs'
            policy=1

            env.yaw_bool = True


            step_frequency_cmd = 2.0
            footswing_height_cmd = 0.30
            stance_width_cmd = 0.35

            print(f'Switch policy to {curr_policy}')
        
        if thumbs<0:
            curr_policy='walk'
            policy=0

            env.yaw_bool = False

            step_frequency_cmd = 3.0
            footswing_height_cmd = 0.08
            stance_width_cmd = 0.35

            print(f'Switch policy to {curr_policy}')
            #test_started=True


        if not execute_script:

            # read and update footswing height
            if foot_up!=0:
                #body_height_cmd = 0.0
                footswing_height_cmd = 0.25
                #stance_width_cmd = 0.35
                step_frequency_cmd = 2.0
                stance_width_cmd = 0.35

                # footswing_height_cmd=0.27
                # step_frequency_cmd=2.0
                # body_height_cmd = 0.25
                # stance_width_cmd = 0.5

            elif foot_down!=0 :
               # body_height_cmd = 0.0
                footswing_height_cmd = 0.08
                #stance_width_cmd = 0.25
                step_frequency_cmd = 3.0


                # footswing_height_cmd=0.08
                # step_frequency_cmd=3.0
                # body_height_cmd = 0.0
                stance_width_cmd = 0.35

            if xdpad>0:
                gait = torch.tensor(gaits["trotting"])

            if xdpad<0:
                gait = torch.tensor(gaits["pacing"])

            if ydpad>0:
                gait = torch.tensor(gaits["pronking"])

            if ydpad<0:
                gait = torch.tensor(gaits["bounding"])

            # read and update action script execute
            if rtrig!=0:
                print('Executing action script')
                execute_script=True
            

        else:
            first, _= render_first_third_imgs(env)
            img= process_realsense(image=first, deploy=True)

            # check if model memory is filled yet
            if not model.memory_filled:
                # if not, add to memory and get processed command

                # add to memory
                model(img)

            else:

                # if memory is filled, get predicted commands from NN
                commands, policy = model(img)
                commands, policy = model._data_rescale(commands, policy)

                x_vel_cmd,y_vel_cmd,yaw_vel_cmd,footswing_height_cmd,step_frequency_cmd = commands

                print('x_vel:',round(x_vel_cmd,2),'y_vel:',round(y_vel_cmd,2),
                'yaw:',round(yaw_vel_cmd,2),'height:',round(footswing_height_cmd,2), 
                'freq:',round(step_frequency_cmd,2), 'policy:',curr_policy)

            if policy==1:
                curr_policy='stairs'
                env.yaw_bool = True
                stance_width_cmd = 0.35
            elif policy==0:
                curr_policy='walk'
                env.yaw_bool = False
                stance_width_cmd = 0.25

        
        
        env.commands[:, 0] = x_vel_cmd
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)


        # extract demo data
        if i%10==0 and collect_data:

            if demo_type=='demo':

                comms = [x_vel_cmd,y_vel_cmd,yaw_vel_cmd,footswing_height_cmd,step_frequency_cmd, policy]
                
                new_demo.collect_demo_data(comms, env)
            
            elif demo_type=='manual':
                 
                torques = info['torques'].flatten()
                joint_vels = info['joint_vel'].flatten()


                new_demo.collect_gait_test_data(torque=torques ,vel=joint_vels, gait = manual_gait)

        # record video
        if record_video:
            new_video.record_img(env)

        i+=1



    # save demo
    if collect_data:
        new_demo.save_demo()

    if record_video:
        new_video.save_video()



if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)

# %%
