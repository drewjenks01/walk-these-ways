import cv2
import os
import numpy as np
import select
import threading
import lcm
import pickle as pkl
import pathlib
import lmdb
import imageio

import time

class Camera(object):
    def __init__(self):

        kill_cmd = "kill -9 $(ps aux |grep rear_* | awk '{print $2}')"
        import subprocess
        subprocess.call(kill_cmd, shell=True)

        self.cap = cv2.VideoCapture(0)
        # width, height = 480, 400
        width, height = 960, 800
        self.fps = 15
        print("Set cap")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,self.fps)
        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        # self.log_subscription = self.lc.subscribe("loginfo", self._log_cb)
        self.db = None

        self.aligned_time = 0.0
        self.log_signal = False
        self.previous_log_signal = False
        self.log_num = 0
        self.logfile_name = "tmp"

        self.mp4_writer = imageio.get_writer(f'./fish_viz.mp4', fps=self.fps)

        print("Start thread")
        
        self.log_thread = threading.Thread(target=self.poll, daemon=False)
        self.log_thread.start()
        # self.datetime = time.strftime("%Y_%m_%d_%H_%M_%S")

    def start(self):

        self.idx = 0
        self.ts = time.time()

        
        print("read")

        #while self.cap.isOpened():
        for i in range(3 * self.fps):

#            print("receiving img")
            start = time.time()
            ret, frame = self.cap.read()

            
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

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.mp4_writer.append_data(rgb_img)

            self.idx += 1
            # self.data[channel_name].append(frame)
            # self.data["log_timestep"].append(self.aligned_time)

            print(f'Freq {1. / (time.time() - self.ts)} Hz'); self.ts = time.time()

        self.mp4_writer.close()


    
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
