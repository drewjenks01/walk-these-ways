import cv2
import pyrealsense2 as rs
import os
import numpy as np
import select
import threading
import lcm
import pickle as pkl
import pathlib
import lmdb
import imageio
from leg_control_data_lcmt import leg_control_data_lcmt
from pd_tau_targets_lcmt import pd_tau_targets_lcmt


import time

class Camera(object):
    def __init__(self):

        self.MOVE_FREELY = True

        kill_cmd = "kill -9 $(ps aux |grep rear_* | awk '{print $2}')"
        import subprocess
        subprocess.call(kill_cmd, shell=True)


        self.fps = 15
        print("Set cap")
        ctx = rs.context()
        devices = ctx.query_devices()
        
        # w, l = 1280, 720
        w, l = 640, 480
        # w, l = 424, 240

        self.pipelines = []
        configs = []
        
        for device in devices:
            sn = device.get_info(rs.camera_info.serial_number)
            # Create a context object. This object owns the handles to all connected realsense devices
            pipeline = rs.pipeline()

            # Configure streams
            config = rs.config()
            config.enable_device(sn)
            # config.enable_stream(rs.stream.depth, w, l, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, w, l, rs.format.bgr8, self.fps)


            self.pipelines += [pipeline]
            configs += [config]

        for pipeline, config in zip(self.pipelines, configs):
            # Start streaming
            pipeline.start(config)

        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        # self.log_subscription = self.lc.subscribe("loginfo", self._log_cb)
        self.received_first_legdata = False
        self.joint_pos = np.zeros(12)
        self.legdata_state_subscription = self.lc.subscribe("leg_control_data", self._legdata_cb)
        self.db = None

        self.aligned_time = 0.0
        self.log_signal = False
        self.previous_log_signal = False
        self.log_num = 0
        self.logfile_name = "tmp"

        self.mp4_writer = imageio.get_writer(f'./rs_viz.mp4', fps=self.fps)

        print("Start thread")
        
        self.log_thread = threading.Thread(target=self.poll, daemon=False)
        self.log_thread.start()
        # self.datetime = time.strftime("%Y_%m_%d_%H_%M_%S")

    def _legdata_cb(self, channel, data):
        # print("update legdata")
        if not self.received_first_legdata:
            self.received_first_legdata = True
            print(f"First legdata")

        msg = leg_control_data_lcmt.decode(data)
        # print(msg.q)
        self.joint_pos = np.array(msg.q)
        self.joint_vel = np.array(msg.qd)
        self.tau_est = np.array(msg.tau_est)
        # print(f"update legdata {msg.id}")

        if self.MOVE_FREELY:
            self._send_cmd()

    def start(self):

        self.idx = 0
        self.ts = time.time()

        images = []
        foot_positions = []
        
        print("read")

        #while self.cap.isOpened():
        for i in range(30 * self.fps):

#            print("receiving img")
            start = time.time()
            viz_frame = []

            for cam_id, pipeline in enumerate(self.pipelines):
                frames = pipeline.wait_for_frames()
                # depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                # print(depth_frame)
                if not color_frame: continue
            

                #depth_frame = spatial.process(depth_frame)
                #depth_frame = temporal.process(depth_frame)
                # depth_frame = hole_filling.process(depth_frame)

                # depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                #print(image.min(), image.max())
                #image = np.clip(image, 0.1, 1.0)
            
                # depth_colormap = cv2.applyColorMap(
                #         cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)

                # viz_frame += [depth_colormap]
                viz_frame += [color_image]

            frame = np.concatenate(viz_frame, axis=0)
            
            # out_dim = 200
            # image = cv2.resize(frame.astype(np.uint8), (out_dim, out_dim))
            img = frame.astype(np.uint8)
            # rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Convert the image to HSV color space
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Define the lower and upper bounds of the pink color
            lower_pink = np.array([150, 50, 50])
            upper_pink = np.array([180, 255, 255])

            # Create a binary mask of the pink color
            mask = cv2.inRange(hsv, lower_pink, upper_pink)

            # Apply a morphological opening operation to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find contours in the binary mask
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort the contours in descending order based on their area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            # Loop through the sorted contours and draw bounding boxes around the largest 4 contours
            for i, contour in enumerate(contours[:4]):
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # convert to rgb
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # write framerate and metadata
            cv2.putText(rgb_img, f'Freq {1. / (time.time() - self.ts)} Hz', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(rgb_img, f'Frame {self.idx}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(rgb_img, f"jpos_FR: {np.array2string(self.joint_pos[0:3], precision=3)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(rgb_img, f"jpos_FL: {np.array2string(self.joint_pos[3:6], precision=3)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(rgb_img, f"jpos_HR: {np.array2string(self.joint_pos[6:9], precision=3)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(rgb_img, f"jpos_HL: {np.array2string(self.joint_pos[9:12], precision=3)}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            self.mp4_writer.append_data(rgb_img)

            self.idx += 1
            # self.data[channel_name].append(frame)
            # self.data["log_timestep"].append(self.aligned_time)

            print(f'Freq {1. / (time.time() - self.ts)} Hz'); self.ts = time.time()

        self.mp4_writer.close()

    def _send_cmd(self):

        if not self.received_first_legdata:
            return

        command_for_robot = pd_tau_targets_lcmt()
        
        AVERAGE = True
        CONTROLLING_LEG = 2

        if AVERAGE:
            tmp_joint_pos = np.copy(self.joint_pos)
            tmp_joint_pos[0] = -tmp_joint_pos[0]
            tmp_joint_pos[6] = -tmp_joint_pos[6]
            # joint_pos_target = np.tile(np.mean(tmp_joint_pos.reshape(4, 3), axis=0), 4).flatten()
            joint_pos_target = np.tile(tmp_joint_pos[CONTROLLING_LEG*3:(CONTROLLING_LEG+1)*3], 4).flatten()
            joint_pos_target[0] = -joint_pos_target[0]
            joint_pos_target[6] = -joint_pos_target[6]
        else:
            joint_pos_target = self.joint_pos

        command_for_robot.q_des = joint_pos_target
        command_for_robot.qd_des = np.zeros(12)
        command_for_robot.kp = 10 * np.ones(12)
        command_for_robot.kd = 1 * np.ones(12)
        command_for_robot.tau_ff = np.zeros(12)
        command_for_robot.se_contactState = np.zeros(4)
        command_for_robot.timestamp_us = int(time.time() * 10 ** 6)
        command_for_robot.id = 0

        self.lc.publish("pd_plustau_targets", command_for_robot.encode())
    
    def poll(self, cb=None):
        t = time.time()
        try:
            while True:
                timeout = 0.02
                rfds, wfds, efds = select.select([self.lc.fileno()], [], [], timeout)
                # print("trying")
                if rfds:
                    # print("message received!")
                    self.lc.handle()
                    # print(f'Freq {1. / (time.time() - t)} Hz'); t = time.time()
                else:
                    continue
        except KeyboardInterrupt:
            self.mp4_writer.close()
            pass


if __name__ == '__main__':

    camera_nd = Camera()
    camera_nd.start()
