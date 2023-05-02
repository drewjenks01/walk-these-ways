from camera_node import CameraNode
from imu_relay_node import IMURelay
import time

if __name__ == '__main__':

    imu_reader = IMURelay()

    while True:
        time.sleep(0.01)

    # camera_nd = CameraNode(cpuid="REAR")
    # camera_nd.start()