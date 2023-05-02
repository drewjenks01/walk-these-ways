from camera_node import CameraNode

if __name__ == '__main__':

    camera_nd = CameraNode(cpuid="HEAD")
    camera_nd.start()