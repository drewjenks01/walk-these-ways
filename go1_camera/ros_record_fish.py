import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

import time

class Camera(object):
    def __init__(self):
        # Params
        self.image = None
        self.br = CvBridge()
        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(30)

        self.cap = cv2.VideoCapture(1)
        width = 952
        height = 800
        fps = 30
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,fps)

        # Publishers
        self.pub = rospy.Publisher('/front_camera/color/image_raw', Image,queue_size=1)

    def start(self):

        start = time.time()
        end = time.time()
        while self.cap.isOpened() and not rospy.is_shutdown():

            start = time.time()
            ret, frame = self.cap.read()
            if ret is not None:
                self.pub.publish(self.br.cv2_to_imgmsg(frame, "bgr8"))
            # print(frame)
            self.loop_rate.sleep()
            end = time.time()
            true_fps = 1.0/(end-start)
            desired_fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print("desired fps: ", desired_fps, " actual fps: ", true_fps, " width: ",width, " height: ", height)

if __name__ == '__main__':
    rospy.init_node("camera_ros_fisheye", anonymous=True)
    
    camera_nd = Camera()
    camera_nd.start()