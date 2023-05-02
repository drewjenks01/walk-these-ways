import cv2
import os
import numpy as np
import select
import threading
import lcm
import pickle as pkl
from lcm_types.loginfo_lcmt import loginfo_lcmt
from lcm_types.state_estimator_lcmt import state_estimator_lcmt
import pathlib
import lmdb

import time

class CameraNode(object):
    def __init__(self, cpuid="BODY"):

        self.cpuid = cpuid

        width = 480
        height = 400
        self.requested_fps = 50

        if self.cpuid == "BODY":
            self.cap_right = cv2.VideoCapture(0)
            self.cap_left = cv2.VideoCapture(1)
            self.cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap_right.set(cv2.CAP_PROP_FPS,self.requested_fps)
            self.cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap_left.set(cv2.CAP_PROP_FPS,self.requested_fps)

            self.lmdbfile_name = "body.lmdb"

        elif self.cpuid == "HEAD":
            self.cap_chin = cv2.VideoCapture(0)
            self.cap_front = cv2.VideoCapture(1)
            self.cap_chin.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.cap_front.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            self.cap_chin.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap_chin.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap_chin.set(cv2.CAP_PROP_FPS,self.requested_fps)
            self.cap_front.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap_front.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap_front.set(cv2.CAP_PROP_FPS,self.requested_fps)

            self.lmdbfile_name = "head.lmdb"

        elif self.cpuid == "REAR":
            self.cap_rear = cv2.VideoCapture(0)
            self.cap_rear.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap_rear.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap_rear.set(cv2.CAP_PROP_FPS,self.requested_fps)

            self.lmdbfile_name = "rear.lmdb"

        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

        self.log_subscription = self.lc.subscribe("loginfo", self._log_cb)
        self.imu_subscription = self.lc.subscribe("state_estimator_data_relay", self._imu_cb)
        
        self.aligned_time = 0.0
        self.euler = np.zeros(3)
        self.log_signal = False
        self.logging = False
        self.log_num = 0
        self.logfile_name = "tmp"
        
        self.log_thread = threading.Thread(target=self.poll, daemon=False)
        self.log_thread.start()
        # self.datetime = time.strftime("%Y_%m_%d_%H_%M_%S")

    def start(self):

        channel_name_right = "cam_right"
        channel_name_left = "cam_left"
        channel_name_rear = "cam_rear"
        channel_name_chin = "cam_chin"
        channel_name_front = "cam_front"

        frame_skip = 5

             

        while True:
            if self.cpuid == "BODY":
                condition = self.cap_right.isOpened() and self.cap_left.isOpened()
            elif self.cpuid == "HEAD":
                condition = self.cap_chin.isOpened() and self.cap_front.isOpened()
            elif self.cpuid == "REAR":
                condition = self.cap_rear.isOpened()

            if not condition:
                print("camera not opened")
                break

            

            if not self.logging and self.log_signal:
                self.logging = True
                from pathlib import Path
                root = f"{pathlib.Path(__file__).parent.resolve()}/../logs"
                log_path = f'{root}/{self.logfile_name}/{self.log_num:05d}/'
                if not os.path.exists(log_path):
                    os.makedirs(log_path)
                lmdb_path = Path(log_path + self.lmdbfile_name)
                self.db = lmdb.open(str(lmdb_path), 
                                    map_size=1900000000,
                                    subdir=False,
                                    meminit=False,
                                    map_async=True
                            )  

                self.idx = 0
                ts = time.time()   
                print("start logging")
                time.sleep(1. / self.requested_fps)
            elif self.logging and self.log_signal:
                # out_dim_w, out_dim_h = 200, 200
                out_dim_w, out_dim_h = 480, 400
                if self.idx % frame_skip == 0:
                    t_delay = time.time()
                    info = {}
                    info["euler_pre_cap"] = self.euler
                    info["log_timestep_pre_cap"] = self.aligned_time

                    if self.cpuid == "BODY":
                        ret_left, frame_left = self.cap_left.read()
                        ret_right, frame_right = self.cap_right.read()
                        # info = {channel_name_left: cv2.resize(frame_left.astype(np.uint8), (out_dim_w, out_dim_h)).astype(np.uint8),
                        #         channel_name_right: cv2.resize(frame_right.astype(np.uint8), (out_dim_w, out_dim_h)).astype(np.uint8),}
                        info[channel_name_left] = frame_left.astype(np.uint8)
                        info[channel_name_right] = frame_right.astype(np.uint8)
                    elif self.cpuid == "HEAD":
                        ret_chin, frame_chin = self.cap_chin.read()
                        ret_front, frame_front = self.cap_front.read()
                        # info = {channel_name_chin: cv2.resize(frame_chin.astype(np.uint8), (out_dim_w, out_dim_h)).astype(np.uint8),
                        #         channel_name_front: cv2.resize(frame_front.astype(np.uint8), (out_dim_w, out_dim_h)).astype(np.uint8),}
                        info[channel_name_chin] = frame_chin.astype(np.uint8)
                        info[channel_name_front] = frame_front.astype(np.uint8)
                    elif self.cpuid == "REAR":
                        ret_rear, frame_rear = self.cap_rear.read()
                        # info = {channel_name_rear: cv2.resize(frame_rear.astype(np.uint8), (out_dim_w, out_dim_h)).astype(np.uint8),}
                        info[channel_name_rear] = frame_rear.astype(np.uint8)

                    info["log_timestep_post_cap"] = self.aligned_time
                    info["euler_post_cap"] = self.euler

                    # legacy
                    info["log_timestep"] = self.aligned_time
                    info["euler"] = self.euler

                    # print(f"Frame capture: {time.time() - t_delay}")

                    t_write = time.time()

                    txn = self.db.begin(write=True)
                    txn.put(f"{self.idx // frame_skip}".encode("ascii"), pkl.dumps(info, protocol=-1))
                    txn.commit()

                    # print(f"Frame write: {time.time() - t_write}")
                    # print(f"Frequency: {1 / (time.time() - ts)}")
                    ts = time.time()
                else:
                    time.sleep(1. / self.requested_fps)
                self.idx += 1
            elif self.logging and not self.log_signal:
                txn = self.db.begin(write=True)
                txn.put("length".encode("ascii"), pkl.dumps(self.idx // frame_skip - 1, protocol=-1))
                txn.commit()
                self.db.sync()
                self.db.close()
                print("lmdb saved!!")
                time.sleep(1. / self.requested_fps)
                self.logging = False

            

    def _log_cb(self, channel, data):
        msg = loginfo_lcmt.decode(data)
        self.aligned_time = msg.timestep
        self.log_signal = msg.log_signal
        self.log_num = msg.log_num
        self.logfile_name = msg.logfile_name
        # print("log signal received: ", self.log_signal, "log num: ", self.log_num, "aligned time: ", self.aligned_time)

    def _imu_cb(self, channel, data):
        # print("update imu")
        msg = state_estimator_lcmt.decode(data)
        self.euler = np.array(msg.rpy)
        # self.R = get_rotation_matrix_from_rpy(self.euler)


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
            pass