#%%
import isaacgym

assert isaacgym
import torch
import numpy as np

import glob
import pickle as pkl

from go1_gym.envs import *
from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm
from pynput.keyboard import Key, Listener, KeyCode
from pynput import keyboard
import pandas
import imageio

from nav.commandnet.commandNN_utils import process_image, img_to_tensor_norm, load_trained


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
    this_file_path = pathlib.Path(__file__).parent.resolve()
    dirs = glob.glob(f"{this_file_path}/../runs/{label}/*")
    print(dirs, label)
    logdir = sorted(dirs)[0]

    with open(logdir + "/parameters.pkl", 'rb') as file:
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

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    # episode lasts math.ciel(episode_length_s/sim_params.dt), dt=0.01999
    Cfg.env.episode_length_s = 1000
    Cfg.init_state.pos=[1.,1.,1.]
    Cfg.terrain.num_rows = 3
    Cfg.terrain.num_cols = 3
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.mesh_type='trimesh'

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=False, cfg=Cfg)
    env = HistoryWrapper(env)

    # load policy
    from ml_logger import logger
    from go1_gym_learn.ppo_cse.actor_critic import ActorCritic

    policy = load_policy(logdir)

    return env, policy


def manual_test(time,mode):
    print('\n')
    inp = input('Test success?')
    if inp=='y':
        success=True
    else:
        success=False
    tests= pandas.read_pickle('nav/commandnet/manual_tests.pkl')
    new={'Success':success,'Time':time,'Mode':mode}
    new_test=pandas.DataFrame([new])
    tests=pandas.concat([tests,new_test])
    print('Logged new test:\n',tests.iloc[-1])
    tests.to_pickle('nav/commandnet/manual_tests.pkl')
    from collections import Counter
    counts = Counter(list(tests['Mode']))
    print('Tests counts:', counts)


class KeyCheck:

    def __init__(self):
        self.command=[0.0,0.0,0.0]
        self.end=False

    def on_press(self,key):
        #print('{0} pressed'.format(
            #key))
        self.check_key(key)

    def on_release(self,key):
        #print('{0} release'.format(
        # key))
        if key == Key.esc:
            # Stop listener
            self.end=True
            return False

    def check_key(self,key):
        if key==Key.up:
            self.command[0]+=1.0
        elif key==Key.down:
            self.command[0]-=1.0
        elif key==Key.left:
            self.command[1]+=1.0
        elif key==Key.right:
            self.command[1]-=1.0
        elif key==Key.esc:
            self.on_release(key)
        elif key.char=='a':
            self.command[2]+=1.0
        elif key.char=='s':
            self.command[2]-=1.0

        print(self.command)
    

    def start_listening(self):
        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        listener.start()



import math
from inputs import get_gamepad
import threading
class XboxController(object):
    
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()


    def read(self): # return the buttons/triggers that you care about in this methode
        x = -1 * self.LeftJoystickX  # x vel
        y = -1*self.LeftJoystickY # y vel
        yaw = -2*self.RightJoystickX # yaw vel
        a = self.A # switch gaits
        lb = self.LeftBumper
        rb = self.RightBumper
        y_cmd = self.Y
        x_cmd= self.X
        ltrig=self.LeftTrigger
        rtrig=self.RightTrigger
        b=self.B
        return [y, x, yaw, a,lb,rb,y_cmd, x_cmd,ltrig,rtrig,b]


    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                # up/down
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1

                # left/right
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RY':
                    self.RightJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                
                # yaw left/right
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
               
                # record action script
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1

                # execute action script
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1

                # decrease foot swing
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                
                # increase foot swing
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state 

                # starts test collection
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state

                # terminates demo
                elif event.code == 'BTN_NORTH':
                    self.X = event.state #previously switched with X
                
                # starts data collection
                elif event.code == 'BTN_WEST':
                    self.Y = event.state #previously switched with Y
                
                # starts video recording
                elif event.code == 'BTN_EAST':
                    self.B = event.state

                elif event.code == 'BTN_THUMBL':
                    self.LeftThumb = event.state
                elif event.code == 'BTN_THUMBR':
                    self.RightThumb = event.state
                elif event.code == 'BTN_SELECT':
                    self.Back = event.state
                elif event.code == 'BTN_START':
                    self.Start = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY1':
                    self.LeftDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY2':
                    self.RightDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY3':
                    self.UpDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY4':
                    self.DownDPad = event.state

def play_go1(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os
    from matplotlib import pyplot as plt
    from isaacgym import gymapi, torch_utils
    

    label = "gait-conditioned-agility/pretrain-v0/train"

    env, policy = load_env(label, headless=headless)

    num_eval_steps = 25000
    gaits = {"pronking": [0, 0, 0],
             "trotting": [0.5, 0, 0],
             "bounding": [0, 0.5, 0],
             "pacing": [0, 0, 0.5]}

    # key_checker=KeyCheck()
    # x_vel_cmd, y_vel_cmd, yaw_vel_cmd = key_checker.command
    body_height_cmd = 0.0

    # decrease to 2 or 2.5
    step_frequency_cmd = 3.0
    gait = torch.tensor(gaits["trotting"])
    footswing_height_cmd = 0.08
    pitch_cmd = 0.0
    roll_cmd = 0.0
    stance_width_cmd = 0.25

    # measured_x_vels = np.zeros(num_eval_steps)
    # target_x_vels = np.ones(num_eval_steps) * x_vel_cmd
    # joint_positions = np.zeros((num_eval_steps, 12))

    obs = env.reset()


    #key_checker.start_listening()
    joy = XboxController()


    def update_sim_view(offset=[-1,-1,1]):
        bx, by, bz = env.root_states[0, 0], env.root_states[0, 1], env.root_states[0, 2]
        forward = torch_utils.quat_apply(env.base_quat, env.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
        # print('Robot loc: ',[bx.item(),by.item(),bz.item()], 'Heading:',heading.item(),'forward:',list(forward))
        env.set_camera([bx.item()-np.cos(heading.item()), by.item()-np.sin(heading.item()), bz+offset[2]],[bx.item()+np.cos(heading.item()), by.item()+np.sin(heading.item()), bz])

    def render_cam():
        bx, by, bz = env.root_states[0, 0], env.root_states[0, 1], env.root_states[0, 2]
        forward = torch_utils.quat_apply(env.base_quat, env.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
        offset=[bx.item()-np.cos(heading.item()), by.item()-np.sin(heading.item()), bz+1]

        # collect 1st person view
        env.gym.set_camera_location(env.rendering_camera, env.envs[0], gymapi.Vec3(bx,by,bz+0.5),
                                    gymapi.Vec3(bx.item()+1.5*np.cos(heading.item()), by.item()+ 1.5*np.sin(heading.item()), bz))
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        img = env.gym.get_camera_image(env.sim, env.envs[0], env.rendering_camera, gymapi.IMAGE_COLOR)
        w, h = img.shape
        first_person= img.reshape([w, h // 4, 4])

        # plt.figure()
        # plt.imshow(img)
        # plt.show()

        # collect 3rd person view
        env.gym.set_camera_location(env.rendering_camera, env.envs[0], gymapi.Vec3(bx.item()-1.2*np.cos(heading.item()), by.item()-1.2*np.sin(heading.item()), bz+1),gymapi.Vec3(bx.item()+np.cos(heading.item()), by.item()+np.sin(heading.item()), bz))
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        img = env.gym.get_camera_image(env.sim, env.envs[0], env.rendering_camera, gymapi.IMAGE_COLOR)
        w, h = img.shape
        third_person= img.reshape([w, h // 4, 4])

        return first_person, third_person
    
    collect_data=False  # starts when Y pressed on controller
    execute_script=False
    data_count=0
    record_video=False

    # holds data collected during demo
    demo_data={'Commands':[], 'Image1st':[],
        'Image3rd':[],'Terrain':''}
    
    i=0
    video_imgs=[]
    model_mode='comb'
    model, data_mean, data_std= load_trained(mode=model_mode)

    test_started=False
    test_time=0
    
    # help auto robot with snags
    static=False
    while True:
        with torch.no_grad():
            actions = policy(obs)

        update_sim_view()

        # if i%100==0 and i!=0:
        #     im1,im2 =render_cam()

        #     f,ax = plt.subplots(2,2)
        #     ax[0,0].imshow(im1[:,:,:3])
        #     ax[0,1].imshow(im1[125:,75:275,:3])
        #     ax[1,0].imshow(im2[:,:,:3])
        #     ax[1,1].imshow(im2[125:,100:250,:3])
        #     plt.show()

        x_vel_cmd, y_vel_cmd, yaw_vel_cmd, test_read, foot_down, foot_up, y_cmd, x_cmd, ltrig, rtrig, b=joy.read()
        #print(x_vel_cmd, y_vel_cmd, yaw_vel_cmd, '  \r',)

        # read in X and terminate if pressed
        if x_cmd!=0:
            break
     

        # read and update demo collect bool
        if y_cmd!=0:
            print('Starting data collection')
            collect_data=True

        # turn off NN control
        if ltrig!=0:
            print('Action script off')
            execute_script=False

        if b!=0:
            print('Recording video')
            record_video=True

        if test_read!=0:
            print('Starting test')
            test_started=True


        if not execute_script:

            # read and update footswing height
            if foot_up!=0:
                footswing_height_cmd=0.25
                step_frequency_cmd=2.0
            elif foot_down!=0 :
                footswing_height_cmd=0.08
                step_frequency_cmd=3.0

            # read and update action script execute
            if rtrig!=0:
                print('Executing action script')
                execute_script=True


        
        elif static:
            x_vel_cmd=0.0
            y_vel_cmd=0.0
            yaw_vel_cmd=0.0
            

        else:
            first, third= render_cam()
            img,_ = process_image(first, third,model.image_mode)
            img= img_to_tensor_norm(img,data_mean,data_std)[None,...].cuda()
            outputs = model(img)
            x_vel_cmd,y_vel_cmd,yaw_vel_cmd,footswing_height_cmd,step_frequency_cmd = model._data_rescale(outputs)

            print('x_vel:',round(x_vel_cmd,2),'y_vel:',round(y_vel_cmd,2),
                'yaw:',round(yaw_vel_cmd,2),'height:',round(footswing_height_cmd,2), 
                'freq:',round(step_frequency_cmd,2), end='\r')

        # check pos deltas
        # if i%200==0 and i>=500 and execute_script:
        #     if static:
        #         static=False
        #     else:
        #         pos_curr=np.array([env.root_states[0, 0].cpu(), env.root_states[0, 1].cpu()])
        #         dist = np.linalg.norm(pos_past-pos_curr)
        #         print('\ndist:',dist,)

        #         if dist<=0.5:
        #             static=True

        #     pos_past=np.array([env.root_states[0, 0].cpu(), env.root_states[0, 1].cpu()])
        
        # elif i==0:
        #     pos_past=np.array([env.root_states[0, 0].cpu(), env.root_states[0, 1].cpu()])


        


        #print(footswing_height_cmd)
            
        #x_vel_cmd, y_vel_cmd, yaw_vel_cmd = key_checker.command
        
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

        # measured_x_vels[i] = env.base_lin_vel[0, 0]
        # joint_positions[i] = env.dof_pos[0, :].cpu()

        # extract demo data
        if i%10==0 and collect_data:
            first_person, third_person = render_cam()
            cont=[x_vel_cmd,y_vel_cmd,yaw_vel_cmd,footswing_height_cmd,step_frequency_cmd]

            demo_data['Commands'].append(cont)
            demo_data['Image1st'].append(first_person)
            demo_data['Image3rd'].append(third_person)

            #print(np.array(demo_data['Commands']).shape,np.array(demo_data['Image1st']).shape)

            data_count+=1
            print('Collected data: ', data_count)

        # record video
        if record_video:
            vid_img = env.render(mode="rgb_array")
            video_imgs.append(vid_img)

        i+=1
        if test_started: test_time+=1

    # change to numpy
    if collect_data:
        demo_data['Commands']=np.array(demo_data['Commands'])
        demo_data['Image1st']=np.array(demo_data['Image1st'])
        demo_data['Image3rd']=np.array(demo_data['Image3rd'])

        print('Demo shape (commands):',demo_data['Commands'].shape)
        print('Demo shape (image1):',demo_data['Image1st'].shape)
        print('avg commands:', np.mean(demo_data['Commands'],axis=0))

        # output example data
        # print('EXAMPLE DATA')
        # action=demo_data['Commands'][5]
        # pics=[np.array(demo_data['Image1st'][5],np.int32),np.array(demo_data['Image3rd'][5],np.int32)]
        # print('Action at 5:', action)
        # plt.figure()
        # plt.imshow(pics[0])
        # plt.show()
        # plt.figure()
        # plt.imshow(pics[1])
        # plt.show()


        # dump demo to folder
        print('Dumping demo!')
        
        path='nav/robot_demos/demosDF.pkl'
        demos= pandas.read_pickle(path)
        print('Old DF size:',demos.shape)

        terrain = 'med-0033'
        demo_data['Terrain']=terrain
        demo_data=pandas.DataFrame([demo_data])
        demos=pandas.concat([demos,demo_data])
        print('New DF size:',demos.shape)

        demos.to_pickle('nav/robot_demos/demosDF.pkl')

    if record_video:
        with imageio.get_writer('nav/play_video.mp4', mode='I',fps=35) as writer:
            for img in video_imgs:
                writer.append_data(img)


    if test_started:
        manual_test(test_time, model_mode)




        
        

    # for i in tqdm(range(num_eval_steps)):
    #     with torch.no_grad():
    #         actions = policy(obs)
    #     env.commands[:, 0] = x_vel_cmd
    #     env.commands[:, 1] = y_vel_cmd
    #     env.commands[:, 2] = yaw_vel_cmd
    #     env.commands[:, 3] = body_height_cmd
    #     env.commands[:, 4] = step_frequency_cmd
    #     env.commands[:, 5:8] = gait
    #     env.commands[:, 8] = 0.5
    #     env.commands[:, 9] = footswing_height_cmd
    #     env.commands[:, 10] = pitch_cmd
    #     env.commands[:, 11] = roll_cmd
    #     env.commands[:, 12] = stance_width_cmd
    #     obs, rew, done, info = env.step(actions)

    #     measured_x_vels[i] = env.base_lin_vel[0, 0]
    #     joint_positions[i] = env.dof_pos[0, :].cpu()

    # # plot target and measured forward velocity
    # from matplotlib import pyplot as plt
    # fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    # axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), measured_x_vels, color='black', linestyle="-", label="Measured")
    # axs[0].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), target_x_vels, color='black', linestyle="--", label="Desired")
    # axs[0].legend()
    # axs[0].set_title("Forward Linear Velocity")
    # axs[0].set_xlabel("Time (s)")
    # axs[0].set_ylabel("Velocity (m/s)")

    # axs[1].plot(np.linspace(0, num_eval_steps * env.dt, num_eval_steps), joint_positions, linestyle="-", label="Measured")
    # axs[1].set_title("Joint Positions")
    # axs[1].set_xlabel("Time (s)")
    # axs[1].set_ylabel("Joint Position (rad)")

    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)

# %%
