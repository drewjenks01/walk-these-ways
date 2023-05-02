import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import pickle as pkl
import lcm
import pathlib
import threading
import select
import math
import torch

from lcm_types.leg_control_data_lcmt import leg_control_data_lcmt
from lcm_types.state_estimator_lcmt import state_estimator_lcmt
from lcm_types.rc_command_lcmt import rc_command_lcmt
from lcm_types.realsense_lcmt import realsense_lcmt

from navigation.utils.image_processing import process_realsense
from navigation.commandnet.commandNN import CommandNet
from navigation.utils.demo import Demo

import gzip, pickletools

"""
To do
 - capture deth as well if no slow down
 - test slow down with timer -> every 10 iters, 10/time = freq

"""
import cv2
import torch
import numpy as np


class RealSense:

    def __init__(self, camera_type, display_camera=False):
        self.display_camera_img = display_camera
        self.camera_type = camera_type      # 'realsense' or '360'

        if self.display_camera_img:
            print('Displaying img')
            self.display_camera()

        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

        self.imu_subscription = self.lc.subscribe("state_estimator_data", self._imu_cb)
        self.legdata_state_subscription = self.lc.subscribe("leg_control_data", self._legdata_cb)
        self.rc_command_subscription = self.lc.subscribe("rc_command", self._rc_command_cb)
        # self.realsense_subscription = self.lc.subscribe("realsense_command_data", self._i_cb)

        self.received_first_legdata=False

        self.rc_msg=None
        self.legdata_msg=None
        self.imu_msg=None
        self.processed_msg=None

        self.use_commandnet = False

        self.go = True
        self.policy=0

        self.rs_pipeline = None
        self.rs_align = None
        self.camera=None

        # DEFINE LOG ROOT
        self.log_root = '/home/unitree/go1_gym/logs/jenkins_experiment'
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)
        self.log_filename = None
        self.log_count = 1
        self.logging = False
        self.log_start_time=0
        self.log_iter_count =1

        # reverse legs
        self.joint_idxs = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        self.contact_idxs = [1, 0, 3, 2]
        # self.joint_idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)
        self.tau_est = np.zeros(12)
        self.world_lin_vel = np.zeros(3)
        self.world_ang_vel = np.zeros(3)
        self.euler = np.zeros(3)
        self.R = np.eye(3)
        self.buf_idx = 0

        self.smoothing_length = 12
        self.deuler_history = np.zeros((self.smoothing_length, 3))
        self.dt_history = np.zeros((self.smoothing_length, 1))
        self.euler_prev = np.zeros(3)
        self.timuprev = time.time()

        self.body_lin_vel = np.zeros(3)
        self.body_ang_vel = np.zeros(3)
        self.smoothing_ratio = 0.2

        self.contact_state = np.ones(4)

        self.mode = 0
        self.ctrlmode_left = 0
        self.ctrlmode_right = 0
        self.left_stick = [0, 0]
        self.right_stick = [0, 0]
        self.left_upper_switch = 0
        self.left_lower_left_switch = 0
        self.left_lower_right_switch = 0
        self.right_upper_switch = 0
        self.right_lower_left_switch = 0
        self.right_lower_right_switch = 0
        self.left_upper_switch_pressed = 0
        self.left_lower_left_switch_pressed = 0
        self.left_lower_right_switch_pressed = 0
        self.right_upper_switch_pressed = 0
        self.right_lower_left_switch_pressed = 0
        self.right_lower_right_switch_pressed = 0

        # default trotting gait
        self.cmd_freq = 3.0
        self.cmd_phase = 0.5
        self.cmd_offset = 0.0
        self.cmd_duration = 0.5

        self.timestep=0


        self.init_time = time.time()
        self.received_first_legdata = False

        self.body_loc = np.array([0, 0, 0])
        self.body_quat = np.array([0, 0, 0, 1])


        #commandnet info
        model_name = 'resnet18'
        demo_folder = 'simple'
        scaled_commands = False
        deploy = True
        self.model = CommandNet(model_name=model_name, demo_folder=demo_folder, scaled_commands=scaled_commands, deploy=deploy)
        self.model.load_trained()

        # fake inference data to cache model
        fake_data=torch.zeros(size=(1,3,224,224)).cuda()
        self.model(fake_data)
        print('NN is ready!')
        #self._legdata_cb(np.zeros((5)))

        os.system(f'sudo chown -R $USER {self.log_root}')
        self.spin()
        self.realsense_log()


    
    def realsense_log(self):
        print('Starting Realsense...')
        # initialize realsense
        self.initialize_realsense()
        self.timestep=0
        #self.time = time.time()

        # demo
        demo = Demo(log_root=self.log_root)
        while True:
            # if self.timestep % 10 == 0 and self.timestep!=0: 
            #     print(f'frq: {10 / (time.time() - self.time)} Hz');
            #     self.time = time.time()

            if self.right_lower_right_switch_pressed:
                print('Resetting...')
                self.right_lower_right_switch_pressed=False
                demo.end_log(no_save=True)
                

            # check rc_command to start log
            if self.left_lower_left_switch_pressed and not self.logging:
                demo.init_log()
                self.left_lower_left_switch_pressed = False


            elif self.left_lower_left_switch_pressed and self.logging:
                print('Ending log...')
                demo.end_log()
                self.left_lower_left_switch_pressed = False


            camera_imgs = self.capture_realsense_img()
            rs_img_rgb = camera_imgs['Image1st']
            

            if self.use_commandnet:

                # check if model memory is filled yet
                if not self.model.memory_filled:
                    # if not, add to memory and get processed command

                    # add to memory
                    comms = self.nn_commands(rs_img_rgb)

                    # processed command
                    comms = self.get_processed_command()

                else:

                    # if memory is filled, get predicted commands from NN
                    comms = self.nn_commands(rs_img_rgb)

            else:
                comms = self.get_processed_command()


            # log commands and image
            if self.logging:
                if demo.log_iter_count%1==0:

                    torques = 
                    joint_vels = self.joint_vel

                    demo_data = {'Commands':comms, 'Image1st':rs_img_rgb, 'DepthImg':camera_imgs['DepthImg']}

                    demo.collect_demo_data(data = demo_data)


                # save partial log every 5 seconds
                if time.time()-self.log_start_time>=5:
                    fps = len(demo.log['Image1st'])/5.0
                    print(f'FPS: {fps}')
                    demo.save_partial_log()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            self.timestep+=1


        # terminate realsense
        self.terminate_realsense()


    def initialize_realsense(self):
        if self.camera_type == '360':
            index=1

            # Open the default camera (0). Change the number if you have multiple cameras.
            cap = cv2.VideoCapture(index)
            # To set the resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 3)
            print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT),cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FPS))
            self.camera = cap

            if not cap.isOpened():
                print("Error: Could not open the camera.")
                return
        
        elif self.camera_type == 'realsense':
            self.rs_pipeline = rs.pipeline()
            config = rs.config()

            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 3)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 3)

            self.rs_pipeline.start(config)

            align_to = rs.stream.color
            self.rs_align = rs.rs_align(align_to)

            print('RS FPS:', )




        print('Beginning log')


    def terminate_realsense(self):
        if self.camera_type == 'realsense':
            self.rs_pipeline.stop()

        elif self.camera_type == '360':

            self.camera.release()
            cv2.destroyAllWindows()

        
    def capture_realsense_img(self):
        imgs={}
        if self.camera_type == '360':
            # Capture frame-by-frame
            ret, frame = self.camera.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            imgs['Image1st'] = rgb


        elif self.camera_type == 'realsense':

            frames = self.rs_pipeline.wait_for_frames()
            aligned_framed = self.rs_align.process(frames)

            depth_frame = aligned_framed.get_depth_frame()
            color_frame = aligned_framed.get_color_frame()

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            imgs['Image1st'] = color_image
            imgs['DepthImg'] = depth_image

        return imgs

    def get_processed_command(self):
        MODES_LEFT = ["body_height", "lat_vel", "stance_width"]
        MODES_RIGHT = ["step_frequency", "footswing_height", "body_pitch"]

        if self.left_upper_switch_pressed:
            self.ctrlmode_left = (self.ctrlmode_left + 1) % 3
            self.left_upper_switch_pressed = False
        if self.right_upper_switch_pressed:
            self.ctrlmode_right = (self.ctrlmode_right + 1) % 3
            self.right_upper_switch_pressed = False

        MODE_LEFT = MODES_LEFT[self.ctrlmode_left]
        MODE_RIGHT = MODES_RIGHT[self.ctrlmode_right]

        # always in use
        cmd_x = 1 * self.left_stick[1]
        cmd_yaw = -1 * self.right_stick[0]

        # default values
        cmd_y = 0.  # -1 * self.left_stick[0]
        cmd_height = 0.
        cmd_footswing = 0.08
        cmd_stance_width = 0.33
        cmd_stance_length = 0.40
        cmd_ori_pitch = 0.
        cmd_ori_roll = 0.
        cmd_freq = 3.0

        # joystick commands
        if MODE_LEFT == "body_height":
            cmd_height = 0.3 * self.left_stick[0]
        elif MODE_LEFT == "lat_vel":
            cmd_y = 0.6 * self.left_stick[0]
        elif MODE_LEFT == "stance_width":
            cmd_stance_width = 0.275 + 0.175 * self.left_stick[0]
        if MODE_RIGHT == "step_frequency":
            min_freq = 2.0
            max_freq = 4.0
            cmd_freq = (1 + self.right_stick[1]) / 2 * (max_freq - min_freq) + min_freq
        elif MODE_RIGHT == "footswing_height":
            cmd_footswing = max(0, self.right_stick[1]) * 0.32 + 0.03
        elif MODE_RIGHT == "body_pitch":
            cmd_ori_pitch = -0.4 * self.right_stick[1]

        # gait buttons
        if self.mode == 0:
            self.cmd_phase = 0.5
            self.cmd_offset = 0.0
            self.cmd_bound = 0.0
            self.cmd_duration = 0.5
        elif self.mode == 1:
            self.cmd_phase = 0.0
            self.cmd_offset = 0.0
            self.cmd_bound = 0.0
            self.cmd_duration = 0.5
        elif self.mode == 2:
            self.cmd_phase = 0.0
            self.cmd_offset = 0.5
            self.cmd_bound = 0.0
            self.cmd_duration = 0.5
        elif self.mode == 3:
            self.cmd_phase = 0.0
            self.cmd_offset = 0.0
            self.cmd_bound = 0.5
            self.cmd_duration = 0.5
        else:
            self.cmd_phase = 0.5
            self.cmd_offset = 0.0
            self.cmd_bound = 0.0
            self.cmd_duration = 0.5

        # up dpad
        if self.mode == 4:
            self.policy=1    # stairs

            cmd_freq = 2.0
            cmd_footswing = 0.30

        elif self.mode == 6:
            self.policy=0    # walk

            cmd_freq = 3.0
            cmd_footswing = 0.08

        elif self.mode==5: #right dpad
            self.use_commandnet=True
            print('Using CommandNet')

        elif self.mode==7: #left dpad
            self.use_commandnet=False
        

        comms = np.array([cmd_x, cmd_y, cmd_yaw, cmd_footswing, cmd_freq, self.policy])
        return comms

    def nn_commands(self,img):
        if self.mode == 7:
            print('Stopping NN...')
            self.use_commandnet = False

        img= process_realsense(img, deploy=True)
        commands, policy = self.model(img)
        commands, policy = self.model._data_rescale(commands, policy)

        commands.append(policy)
        commands = np.array(commands)

        self.rs_commanddata_cb(commands)
        return commands


    def _legdata_cb(self, channel, data):

        msg = leg_control_data_lcmt.decode(data)

        # print(msg.q)
        self.joint_pos = np.array(msg.q)
        self.joint_vel = np.array(msg.qd)
        self.tau_est = np.array(msg.tau_est)
        # print(f"update legdata {msg.id}")


    def _imu_cb(self, channel, data):
        # print("update imu")
        msg = state_estimator_lcmt.decode(data)

        self.euler = np.array(msg.rpy)

        self.R = self.get_rotation_matrix_from_rpy(self.euler)

        self.contact_state = 1.0 * (np.array(msg.contact_estimate) > 200)

        self.deuler_history[self.buf_idx % self.smoothing_length, :] = msg.rpy - self.euler_prev
        self.dt_history[self.buf_idx % self.smoothing_length] = time.time() - self.timuprev

        self.timuprev = time.time()

        self.buf_idx += 1
        self.euler_prev = np.array(msg.rpy)



    def _rc_command_cb(self, channel, data):

        msg = rc_command_lcmt.decode(data)

        self.left_upper_switch_pressed = ((msg.left_upper_switch and not self.left_upper_switch) or self.left_upper_switch_pressed)
        self.left_lower_left_switch_pressed = ((msg.left_lower_left_switch and not self.left_lower_left_switch) or self.left_lower_left_switch_pressed)
        self.left_lower_right_switch_pressed = ((msg.left_lower_right_switch and not self.left_lower_right_switch) or self.left_lower_right_switch_pressed)
        self.right_upper_switch_pressed = ((msg.right_upper_switch and not self.right_upper_switch) or self.right_upper_switch_pressed)
        self.right_lower_left_switch_pressed = ((msg.right_lower_left_switch and not self.right_lower_left_switch) or self.right_lower_left_switch_pressed)
        self.right_lower_right_switch_pressed = ((msg.right_lower_right_switch and not self.right_lower_right_switch) or self.right_lower_right_switch_pressed)

        self.mode = msg.mode
        self.right_stick = msg.right_stick
        self.left_stick = msg.left_stick
        self.left_upper_switch = msg.left_upper_switch
        self.left_lower_left_switch = msg.left_lower_left_switch
        self.left_lower_right_switch = msg.left_lower_right_switch
        self.right_upper_switch = msg.right_upper_switch
        self.right_lower_left_switch = msg.right_lower_left_switch
        self.right_lower_right_switch = msg.right_lower_right_switch

        # print(self.right_stick, self.left_stick)


    def rs_commanddata_cb(self, data):

        rs = realsense_lcmt()

        rs.commands = data

        self.lc.publish("realsense_command_data", rs.encode())


    def get_rotation_matrix_from_rpy(self,rpy):
        """
        Get rotation matrix from the given quaternion.
        Args:
            q (np.array[float[4]]): quaternion [w,x,y,z]
        Returns:
            np.array[float[3,3]]: rotation matrix.
        """
        r, p, y = rpy
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(r), -math.sin(r)],
                        [0, math.sin(r), math.cos(r)]
                        ])

        R_y = np.array([[math.cos(p), 0, math.sin(p)],
                        [0, 1, 0],
                        [-math.sin(p), 0, math.cos(p)]
                        ])

        R_z = np.array([[math.cos(y), -math.sin(y), 0],
                        [math.sin(y), math.cos(y), 0],
                        [0, 0, 1]
                        ])

        rot = np.dot(R_z, np.dot(R_y, R_x))
        return rot

    def poll(self, cb=None):
        t = time.time()
        try:
            while True:
                timeout = 0.01
                rfds, wfds, efds = select.select([self.lc.fileno()], [], [], timeout)
                if rfds:
                    # print("message received!")
                    self.lc.handle()
                    # print(f'Freq {1. / (time.time() - t)} Hz'); t = time.time()
                else:
                    continue
                    # print(f'waiting for message... Freq {1. / (time.time() - t)} Hz'); t = time.time()
                #    if cb is not None:
                #        cb()
        except KeyboardInterrupt:
            pass

    def spin(self):
        print('Starting subscriptions...')
        self.run_thread = threading.Thread(target=self.poll, daemon=False)
        self.run_thread.start()



if __name__ == '__main__':

    rs = RealSense(time_logs=False, display_camera=False)