#%%
import isaacgym

assert isaacgym
import torch
import numpy as np
import random
import time

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

from navigation.vision.commandNN import CommandNet
from navigation.vision.utils.image_processing import process_deployed
from navigation.sim.sim_utils import create_xbox_controller, update_sim_view, render_first_third_imgs
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
    Cfg.terrain.generated_name = '0014'
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
    domain_random = False
    if domain_random:
        rand_y_drift = None
        rand_yaw_drift = None
        domain_change_timer = time.time()

    #demo_folder=f'sim_trials/{env.cfg.terrain.generated_diff}'



   #
   # 
   #  new_video = VideoRecorder()

    i=0

    demo_save_folder = 'navigation/robot_demos/icra_trials'
    demo_folder = 'icra_trials/combo'
    model_name = 'mnv3s'
    use_memory=True
    multi_command = True
    scale_commands = True
    finetune= True
    model = CommandNet(model_name=model_name,
                        demo_folder=demo_folder, 
                        deploy=True, 
                        use_memory=use_memory, 
                        multi_command=multi_command, 
                        scaled_commands=scale_commands,
                        finetune=finetune)

    test_started=False
    test_time=0
    
    curr_policy = 'walk'
    policy=0



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
            if collect_data:
                new_demo.undo_log()
            collect_data=False

            
     
        # read and update demo collect bool
        if y_cmd!=0 and not collect_data:
            print('Starting log...')
            new_demo = Demo(log_root=demo_save_folder)

            collect_data=True

            fps_logging = time.time()
            curr_loc = np.array([env.root_states[0, 0].item(), env.root_states[0, 1].item()])

            # manual testing
            new_demo.init_log(start_iter=i)
        
        if a_btn!=0:
            print('Saving demo...')
            collect_data=False

            print('\n')
            inp = input('Test success?')

            # check if test is sucess
            if inp=='y':
                success=1
            else:
                success=0

            addit = {'Success':success}
            new_demo.log.update(addit)
            new_demo.end_log(end_iter=i)
            del new_demo
            new_demo=Demo(log_root=demo_folder)
            

        # turn off NN control
        if ltrig!=0:
            print('Action script off')
            execute_script=False
            if model.use_memory:
                model._reset_memory()
                model._reset_fill_count()

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


            print(f'Switch policy to {curr_policy}')
        
        if thumbs<0:
            curr_policy='walk'
            policy=0

            env.yaw_bool = False

            step_frequency_cmd = 3.0
            footswing_height_cmd = 0.08

            print(f'Switch policy to {curr_policy}')
            #test_started=True


        if not execute_script:

            # read and update footswing height
            if foot_up!=0:
                #body_height_cmd = 0.0
                footswing_height_cmd = 0.3
                #stance_width_cmd = 0.35
                step_frequency_cmd = 2.0

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
            first,_ = render_first_third_imgs(env)
            img= process_deployed(first)

            # check if model memory is filled yet
            if model.use_memory and not model.memory_filled:
                # if not, add to memory and get processed command

                # add to memory
                model.forward(img)

            else:

                # if memory is filled, get predicted commands from NN
                commands, policy = model.forward(img)
                commands, policy= model._data_rescale(commands, policy)

                x_vel_cmd,y_vel_cmd,yaw_vel_cmd= commands

                print('x_vel:',round(x_vel_cmd,2),'y_vel:',round(y_vel_cmd,2),
                'yaw:',round(yaw_vel_cmd,2), 'policy:',curr_policy)

            if policy==1:
                curr_policy='stairs'
                env.yaw_bool = True
                step_frequency_cmd = 2.0
                footswing_height_cmd = 0.30
            elif policy==0:
                curr_policy='walk'
                env.yaw_bool = False
                step_frequency_cmd = 3.0
                footswing_height_cmd = 0.08




        # add drift if using domain_randomization
        if domain_random:
            
            # update drifts after 5 sec
            if time.time()-domain_change_timer>=5.0 or not rand_yaw_drift:
               # rand_y_drift = random.uniform(0.1,0.3)*random.choice([1,-1])
                rand_yaw_drift = random.uniform(0.1,0.3)*random.choice([1,-1])

                print('New drift:',rand_yaw_drift)
                domain_change_timer = time.time()

           # y_vel_cmd += rand_y_drift
            #yaw_vel_cmd_drift = yaw_vel_cmd+rand_yaw_drift

            # todo: 

        else:
            rand_yaw_drift = 0.0



        # scale x_vel if using wtw
        if curr_policy=='walk':
            x_vel_cmd_scaled = x_vel_cmd*1.5
            # scale yaw
            yaw_vel_cmd_scaled = yaw_vel_cmd*1.5+ rand_yaw_drift

        elif curr_policy=='stair':

            x_vel_cmd_scaled = x_vel_cmd
            yaw_vel_cmd_scaled = yaw_vel_cmd*1.5 + rand_yaw_drift

        else:
            x_vel_cmd_scaled = x_vel_cmd
            yaw_vel_cmd_scaled = yaw_vel_cmd + rand_yaw_drift

           

       
        env.commands[:, 0] = x_vel_cmd_scaled
        env.commands[:, 1] = y_vel_cmd
        env.commands[:, 2] = yaw_vel_cmd_scaled
        env.commands[:, 3] = body_height_cmd
        env.commands[:, 4] = step_frequency_cmd
        env.commands[:, 5:8] = gait
        env.commands[:, 8] = 0.5
        env.commands[:, 9] = footswing_height_cmd
        env.commands[:, 10] = pitch_cmd
        env.commands[:, 11] = roll_cmd
        env.commands[:, 12] = stance_width_cmd
        obs, rew, done, info = env.step(actions)

        # collect commands every timestep -> will be averages
        if collect_data:

            frame_comms = {'x':x_vel_cmd,'y':y_vel_cmd,'yaw':yaw_vel_cmd,'policy':policy}
            new_demo.collect_frame_commands(frame_comms)



        # extract demo data ->
        if collect_data and time.time()-fps_logging>=1/new_demo.fps:

            first,_ = render_first_third_imgs(env)

            torques = info['torques'].flatten()
            joint_vels = info['joint_vel'].flatten()

            new_loc = np.array([env.root_states[0, 0].item(), env.root_states[0, 1].item()])

            distance = np.linalg.norm(new_loc- curr_loc)

            curr_loc = new_loc

            data = {'Image1st':first, 'Torque':torques, 'Joint_Vel':joint_vels, 'Distance':distance,'Drift':rand_yaw_drift}
            
            new_demo.collect_demo_data(data)

            fps_logging = time.time()

        
            
     

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
