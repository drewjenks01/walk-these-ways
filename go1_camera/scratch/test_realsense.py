# First import the library
import pyrealsense2 as rs
import time
import cv2
import numpy as np
import imageio

try:
    ctx = rs.context()
    devices = ctx.query_devices()
    
    fps = 30
    #w, l = 640, 480
    w, l = 424, 240

    pipelines = []
    configs = []
    
    for device in devices:
        sn = device.get_info(rs.camera_info.serial_number)
        # Create a context object. This object owns the handles to all connected realsense devices
        pipeline = rs.pipeline()

        # Configure streams
        config = rs.config()
        config.enable_device(sn)
        config.enable_stream(rs.stream.depth, w, l, rs.format.z16, fps)
        #config.enable_stream(rs.stream.color, w, l, rs.format.bgr8, fps)


        pipelines += [pipeline]
        configs += [config]

    for pipeline, config in zip(pipelines, configs):
        # Start streaming
        pipeline.start(config)

    ts = time.time()

    print("START")
    
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    mp4_writer = imageio.get_writer(f'./rs_viz.mp4', fps=fps)

    while True:
        # This call waits until a new coherent set of frames is available on a device
        # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        
        viz_frame = []

        for cam_id, (device, pipeline) in enumerate(zip(devices, pipelines)):
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            #color_frame = frames.get_color_frame()
            print(depth_frame)
            if not depth_frame: continue
            elif cam_id == 0: 
                print(f"frequency: {1./(time.time() - ts)}")
                ts = time.time()
        

            #depth_frame = spatial.process(depth_frame)
            #depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)

            depth_image = np.asanyarray(depth_frame.get_data())
            #color_image = np.asanyarray(color_frame.get_data())

            #print(image.min(), image.max())
            #image = np.clip(image, 0.1, 1.0)
        
            depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.09), cv2.COLORMAP_JET)

            viz_frame += [depth_colormap]
            #viz_frame += [color_image]

        #cv2.imwrite("last_image.png", depth_colormap)

        if len(viz_frame) > 0:
            viz_frame = np.concatenate(viz_frame, axis=0)
            mp4_writer.append_data(viz_frame)

        # Print a simple text-based representation of the image, by breaking it into 10x20 pixel regions and approximating the coverage of pixels within one meter
        #coverage = [0]*64
        #for y in range(480):
        #    for x in range(640):
        #        dist = depth.get_distance(x, y)
        #        if 0 < dist and dist < 1:
        #            coverage[x//10] += 1
        #    
        #    if y%20 is 19:
        #        line = ""
        #        for c in coverage:
        #            line += " .:nhBXWW"[c//25]
        #        coverage = [0]*64
        #        print(line)
    mp4_writer.close()
    exit(0)
#except rs.error as e:
#    # Method calls agaisnt librealsense objects may throw exceptions of type pylibrs.error
#    print("pylibrs.error was thrown when calling %s(%s):\n", % (e.get_failed_function(), e.get_failed_args()))
#    print("    %s\n", e.what())
#    exit(1)
except Exception as e:
    print(e)
    pass