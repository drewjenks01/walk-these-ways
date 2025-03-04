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

from navigation.vision.utils.image_processing import process_deployed
from navigation.vision.commandNN import CommandNet
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

        self.realsense_camera = -1

        # boolean for whether or not to load vision policy
        self.use_nn = True
        self.nn_timing = 0
        
        if self.use_nn:
            
            #commandnet info
            model_name = 'dino'
            demo_folder = 'stata'
            multi_command = True
            deploy = True
            name_extra = ''
            data_type = 'rgb'

            self.model = CommandNet(
                            model_name=model_name,
                            demo_folder=demo_folder, 
                            deploy=deploy, 
                            multi_command=multi_command,
                            name_extra=name_extra,
                            data_type=data_type)

            if not self.model.config['use_memory']:
                # fake inference data to cache model
                fake_data=torch.zeros(size=(1,3,224,224)).cuda()
                self.model(fake_data)
                self.rs_commanddata_cb(self.realsense_camera,[0.0,0.0,0.0,0])
                print('NN is ready!')


        os.system(f'sudo chown -R $USER {self.log_root}')
        self.spin()
        self.realsense_log()

    def initialize_cameras(self):
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 30  # The framerate is lowered to avoid any USB3 bandwidth issues
        
        if is_jetson:
            cameras = sl.Camera.get_device_list()
        else:
            cameras = sl.Camera.get_streaming_device_list()
        
        index = 0
        for cam in cameras:
            if is_jetson:
                init.set_from_serial_number(cam.serial_number)
            else:
                init.set_from_stream(cam.ip, cam.port)

            init.depth_mode = sl.DEPTH_MODE.NEURAL
            init.coordinate_units = sl.UNIT.METER
            init.depth_maximum_distance = 3.0
            init.depth_minimum_distance = 0.1

            name_list.append("ZED {}".format(cam.serial_number))
            print("Opening {}".format(name_list[index]))
            zed_list.append(sl.Camera())
            left_list.append(sl.Mat())
            depth_list.append(sl.Mat())
            timestamp_list.append(0)
            last_ts_list.append(0)
            status = zed_list[index].open(init)
            if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
                zed_list[index].close()

            camera_configuration = zed_list[index].get_camera_information().camera_configuration
            seg_list.append(np.full((camera_configuration.resolution.height, camera_configuration.resolution.width, 4), [245, 239, 239,255], np.uint8))
            image_left_ocv_list.append(np.zeros((camera_configuration.resolution.height, camera_configuration.resolution.width, 4), dtype=np.uint8))

            index = index +1

        disp_scale = 4

        # run inference

        display_resolution = sl.Resolution(camera_configuration.resolution.width, camera_configuration.resolution.height)


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

        
        if self.mode == 5:
            self.policy=0    # walk

        elif self.mode == 4:
            self.policy=1    # stair


        elif self.mode == 6:
            self.policy= 2   # duck

        elif self.mode == 7:    # NN
            print('Using NN')
            self.use_commandnet=True
    
        

        comms = np.array([cmd_x, cmd_yaw, self.policy])
        #print(comms)

        return comms

    def nn_commands(self,images):
        if self.mode!=7:
            print('Stopping NN...')
            self.use_commandnet = False

        imgs =[]
        if self.model.data_type in {'rgb', 'both'}:
            imgs.append(images['Image1st'])
        if self.model.data_type in {'depth', 'both'}:
            imgs.append(images['DepthImg'])


        for i in range(len(imgs)):
            imgs[i] = process_deployed(imgs[i])

        if self.model.data_type in {'rgb', 'depth'}:
            assert len(imgs)==1, f'Number of images going to NN is {len(imgs)} but should only be 1.'

        if self.model.config['use_memory']:
            self.model.forward(*imgs)
        else:
            commands, policy = self.model.forward(*imgs)
            commands, policy = self.model._data_rescale(commands, policy)

        if not self.model.config['predict_commands']:
            commands = [-1,-1]

        commands.append(policy)
        commands = np.array(commands)
        print(commands)
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


    def rs_commanddata_cb(self, camera, commands = [-1.0,-1.0,0]):

        rs = realsense_lcmt()

        rs.commands = commands
        rs.camera = camera

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
    image_type = 'rgb'
    rs = RealSense(camera_type=camera_type, image_type=image_type)