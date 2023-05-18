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

from navigation.utils.image_processing import process_deployed
from navigation.commandnet.commandNN import CommandNet
from navigation.utils.demo import Demo

import gzip, pickletools
import cv2
import torch
import numpy as np


class RealSense:

    def __init__(self, camera_type, image_type):
        self.camera_type = camera_type      # 'realsense' or '360'
        self.image_type = image_type    # 'rgb' or 'depth'
        print('Image type:', self.image_type)

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
        self.logging = False

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
        model_name = 'mnv3s'
        demo_folder = 'stata'
        use_memory=False
        multi_command = True
        scale_commands = True
        finetune= True
        deploy=True
        num_classes = 3
        predict_commands = False
        # self.model = CommandNet(model_name=model_name,
        #                 demo_folder=demo_folder, 
        #                 deploy=deploy, 
        #                 use_memory=use_memory, 
        #                 multi_command=multi_command, 
        #                 scaled_commands=scale_commands,
        #                 finetune=finetune,
        #                 num_classes=num_classes,
        #                 predict_commands=predict_commands)

        # if not self.model.use_memory:
        #     # fake inference data to cache model
        #     fake_data=torch.zeros(size=(1,3,224,224)).cuda()
        #     self.model(fake_data)
        #     print('NN is ready!')


        os.system(f'sudo chown -R $USER {self.log_root}')
        self.spin()
        self.realsense_log()


    
    def realsense_log(self):
        print('Starting Realsense...')
        # initialize realsense
        self.initialize_realsense()
        self.timestep=0
        #self.time = time.time()

        while True:
            # if self.timestep % 10 == 0 and self.timestep!=0: 
            #     print(f'frq: {10 / (time.time() - self.time)} Hz');
            #     self.time = time.time()

            if self.right_lower_right_switch_pressed:
                print('Resetting...')
                self.right_lower_right_switch_pressed=False
                if self.logging:
                    demo.undo_log()
                    del demo
                self.logging = False
                

            # check rc_command to start log
            if self.left_lower_left_switch_pressed and not self.logging:
                demo = Demo(log_root=self.log_root)
                fps_logging = time.time()
                self.left_lower_left_switch_pressed = False
                self.logging=True
                demo.init_log(start_iter=self.timestep)
                print('demo initialized')


            elif self.left_lower_left_switch_pressed and self.logging:
                print('Ending log...')
                demo.end_log(end_iter=self.timestep)
                del demo
                print('log ended and saved')
                self.left_lower_left_switch_pressed = False
                self.logging = False


            camera_imgs = self.capture_realsense_img()
            #print(len(camera_imgs))
            

            if self.use_commandnet:
                rs_img_rgb = camera_imgs['Image1st']

                # check if model memory is filled yet
                if self.model.use_memory and not self.model.memory_filled:
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
                frame_comms = {'x':comms[0],'y':comms[1],'yaw':comms[2],'policy':comms[3]}
                demo.collect_frame_commands(frame_comms)

            if self.logging and time.time()-fps_logging>=1/demo.fps and len(comms)>0:

                torques = self.tau_est
                joint_vels = self.joint_vel

                demo_data = {'Torque':torques, 'Joint_Vel':joint_vels}

                if self.image_type=='rgb':
                    demo_data['Image1st'] = camera_imgs['Image1st']
                
                elif self.image_type=='depth':
                    demo_data['DepthImg'] = camera_imgs['DepthImg']

                elif self.image_type == 'both':
                    demo_data['Image1st'] = camera_imgs['Image1st']
                    demo_data['DepthImg'] = camera_imgs['DepthImg']


                demo.collect_demo_data(data = demo_data)


                fps_logging = time.time()
                        

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
            
            if self.image_type == 'rgb':
                config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                align_to = rs.stream.color

            elif self.image_type == 'depth':
                self.colorizer = rs.colorizer()
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                align_to = rs.stream.depth

            elif self.image_type == 'both':
                self.colorizer = rs.colorizer()
                config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                align_to = rs.stream.color

            
            profile = self.rs_pipeline.start(config)

            align_to = rs.stream.color
            self.rs_align = rs.align(align_to)

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
            
            if self.image_type=='rgb':
                color_frame = aligned_framed.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())
                color_image = color_image.copy()
                imgs['Image1st'] = color_image
            
            elif self.image_type == 'depth':
                depth_frame = aligned_framed.get_depth_frame()
                depth_image = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
                depth_image = depth_image.copy()
                imgs['DepthImg'] = depth_image

            elif self.image_type == 'both':
                color_frame = aligned_framed.get_color_frame()
                depth_frame = aligned_framed.get_depth_frame()

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())

                color_image = color_image.copy()
                depth_image = depth_image.copy()

                imgs['Image1st'] = color_image
                imgs['DepthImg'] = depth_image


        return imgs

    def get_processed_command(self):
        # always in use
        cmd_x = 1 * self.left_stick[1]
        cmd_yaw = -1 * self.right_stick[0]

        # default values
        cmd_y = 0.  # -1 * self.left_stick[0]
        
        if self.mode == 5:
            self.policy=0    # walk

        elif self.mode == 4:
            self.policy=1    # stair


        elif self.mode == 6:
            self.policy= 2   # duck

        # elif self.mode == 5:
        #     print('Using NN')
        #     self.use_commandnet=True
    
        

        comms = np.array([cmd_x, cmd_y, cmd_yaw, self.policy])
        #print(comms)

        return comms

    def nn_commands(self,img):
        if self.mode==7:
            print('Stopping NN...')
            self.use_commandnet = False

        img= process_deployed(img)

        if self.model.use_memory:
            self.model.forward(img)
        else:
            commands, policy = self.model.forward(img)
            commands, policy = self.model._data_rescale(commands, policy)

        commands.append(policy)
        commands = np.array(commands)
        print(commands)

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

    camera_type = 'realsense'
    image_type = 'both'
    rs = RealSense(camera_type=camera_type, image_type=image_type)