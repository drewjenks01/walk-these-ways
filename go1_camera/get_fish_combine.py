import cv2
import os
import numpy as np

import time

class Camera(object):
    def __init__(self):

        self.cap0 = cv2.VideoCapture(0)
        self.cap1 = cv2.VideoCapture(1)
        # width = 952
        # height = 800
        # fps = 30
        # self.cap0.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # self.cap0.set(cv2.CAP_PROP_FPS,fps)
        # self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # self.cap1.set(cv2.CAP_PROP_FPS,fps)

    def start(self):

        # start = time.time()
        # end = time.time()
        while self.cap0.isOpened() or self.cap1.isOpened():

            start = time.time()
            ret0, frame0 = self.cap0.read()
            ret1, frame1 = self.cap1.read()
            print("ret0: ", ret0, " ret1: ", ret1)

            end = time.time()
            true_fps = 1.0/(end-start)
            desired_fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print("desired fps: ", desired_fps, " actual fps: ", true_fps, " width: ",width, " height: ", height)

if __name__ == '__main__':

    camera_nd = Camera()
    camera_nd.start()