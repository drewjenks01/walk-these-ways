import cv2
# import pyrealsense2 as rs
import os
import numpy as np
import select
import threading
import lcm
import pickle as pkl
import pathlib
import lmdb
# import imageio
from leg_control_data_lcmt import leg_control_data_lcmt
from state_estimator_lcmt import state_estimator_lcmt
from pd_tau_targets_lcmt import pd_tau_targets_lcmt


import time

class IMUReader(object):
    def __init__(self):

        self.MOVE_FREELY = True

        kill_cmd = "kill -9 $(ps aux |grep rear_* | awk '{print $2}')"
        import subprocess
        subprocess.call(kill_cmd, shell=True)

        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        # self.log_subscription = self.lc.subscribe("loginfo", self._log_cb)
        self.received_first_legdata = False
        self.joint_pos = np.zeros(12)
        self.euler = np.zeros(3)
        self.legdata_state_subscription = self.lc.subscribe("leg_control_data_relay", self._legdata_cb)
        self.imu_subscription = self.lc.subscribe("state_estimator_data_relay", self._imu_cb)
        
        self.db = None

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

        # if self.MOVE_FREELY:
        #     self._send_cmd()

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
            self.mp4_writer.close()
            pass


if __name__ == '__main__':

    imu_reader = IMUReader()
    # camera_nd.spin()
    while True:
        print(imu_reader.euler)
        time.sleep(0.1)
