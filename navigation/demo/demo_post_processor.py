from navigation import constants
from navigation.demo import utils
from navigation.vision.models.vision_model import VisionModel

import os
import pickle as pkl
import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
from io import BytesIO
import argparse
import gzip
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from torchvision import transforms
import torch
import torch.nn as nn


class DemoPostProcessor:
    def __init__(self, demo_folder: str, demo_name: str, run_number: int, demo_compressed: bool):
        self.demo_folder = demo_folder
        self.demo_name = demo_name
        self.run_number = run_number
        self.demo_compressed = demo_compressed
        logging.info(f"Demo name: {self.demo_name}, run_num: {self.run_number}")

        self.demo_folder = (
            constants.DEMO_BASE_PATH
            / self.demo_folder
            / self.demo_name
            / utils.make_run_label(self.run_number)
        )
        logging.info(f"Demo path: {self.demo_folder}")
        assert self.demo_folder.exists()

        # make post-process info folder within run folder
        self.demo_post_folder = self.demo_folder / "post_process_info"
        if not self.demo_post_folder.exists():
            self.demo_post_folder.mkdir()

        self.demo_data = utils.get_empty_demo_data()
        self._load_all_run_data()

    
    def make_vint_topomap(self):
        vint_topomap_dir = constants.DEMO_BASE_PATH.parent / 'vint_topomaps' / self.demo_folder / self.demo_name / utils.make_run_label(self.run_number)
        if not vint_topomap_dir.exists():
            vint_topomap_dir.mkdir()
        
        rgb_images = self.demo_data[constants.FORWARD_RGB_CAMERA]
        depth_images = self.demo_data[constants.FORWARD_DEPTH_CAMERA]
        commands = self.demo_data[constants.RGB]

        

    def get_dino_attention(image):
        model = VisionModel.build_backbone(inference_type='offline')
        patch_size = 14
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        img = transform(img)

        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size

        attentions = model.get_last_selfattention(img.to(constants.DEVICE))

        nh = attentions.shape[1] # number of head

        # we keep only the output patch attention
        # for every patch
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
        # weird: one pixel gets high attention over all heads?
        print(torch.max(attentions, dim=1)) 
        attentions[:, 283] = 0 

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

        # save attentions heatmaps
        os.makedirs(output_dir, exist_ok=True)

        for j in range(nh):
            fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")
            plt.imsave(fname=fname, arr=attentions[j], format='png')
            print(f"{fname} saved.")



    def make_video(self, model = None):
        logging.info("Making video")
        if model is not None:
            logging.info("Using model predictions")
            
        video_filepath = str(self.demo_post_folder / "run_video.mp4")

        rgb_images = self.demo_data[constants.FORWARD_RGB_CAMERA]
        depth_images = self.demo_data[constants.FORWARD_DEPTH_CAMERA]
        commands = self.demo_data[constants.COMMAND_KEY]

        depth_exists = len(depth_images) > 0

        # Set the number of frames (assumes all lists have the same length)
        num_frames = len(rgb_images)

        # Create a figure and subplots for the RGB, depth, and command plots
        fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle(f'Demo folder: {self.demo_folder.parent.parent.name} Demo name: {self.demo_name}, Run: {self.run_number}')

        # Function to update the frames in the animation
        pbar = tqdm(total=num_frames)
        skip_frames = False

        if skip_frames:
            skip_rate = 2
            logging.info(f'Skipping every {skip_rate} frame while making video.')


        def update(frame):
            pbar.update(1)

            # skip frames?
            if skip_frames and frame % skip_rate == 0:
                return

            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()

            # Display the RGB and depth images
            ax1.imshow(rgb_images[frame])
            ax1.set_title("RGB Image")
            
            if depth_exists:
                ax2.imshow(depth_images[frame], cmap="viridis")
                ax2.set_title("Depth Image")
            else:
                ax2.imshow(dino_attention(rgb_images[frame]))
                ax2.set_title("Dino Attention")

            # excludes gait
            ax3.bar(range(constants.NUM_COMMANDS-1), commands[frame][:-1])
            ax3.set_title("Commands")
            ax3.set_ylim(-1, 1)
            ax3.set_xticks(range(constants.NUM_COMMANDS-1))
            ax3.set_xticklabels(constants.COMMAND_NAMES[:-1])

            # gait only
            ax4.bar(0, commands[frame][-1])
            ax4.set_title("Gait")
            ax4.set_ylim(0, 2)
            ax4.set_yticks(range(constants.NUM_GAITS))
            ax4.set_yticklabels(constants.GAIT_NAMES)

            # Adjust subplot layouts and spacing
            plt.tight_layout()

        # Create an animation using FuncAnimation
        ani = FuncAnimation(fig, update, frames=num_frames, repeat=False)

        # Save the animation as a video
        ani.save(video_filepath, writer="ffmpeg", fps=10)

        logging.info(f"Video creation complete. Saved to: {video_filepath}")

    def single_example(self):
        logging.info("Making single example")
        random_timestep = np.random.randint(0, len(self.demo_data[constants.FORWARD_RGB_CAMERA]))

        rgb_image = self.demo_data[constants.FORWARD_RGB_CAMERA][random_timestep]
        depth_image = self.demo_data[constants.FORWARD_DEPTH_CAMERA][random_timestep]
        command = self.demo_data[constants.COMMAND_KEY][random_timestep]

        # Stack the RGB and depth images side by side
        merged_image = np.hstack((rgb_image, depth_image))

        # Convert to uint8 format if necessary (assuming depth is in a different format)
        merged_image = merged_image.astype(np.uint8)

        # Add the robot command as text to the merged image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)
        font_thickness = 2
        text_position = (10, 30)  # Adjust text position as needed
        cv2.putText(
            merged_image,
            f'{command}',
            text_position,
            font,
            font_scale,
            font_color,
            font_thickness,
        )

        # Save the merged image as a file
        output_image = str(self.demo_post_folder / f"single_example_indx_{random_timestep}.png")
        cv2.imwrite(output_image, merged_image)

        print("Image creation complete.")

    def compare_preds_with_truth(self):
        pass
        

    def _load_all_run_data(self):
        num_partial_logs = 0
        for filename in os.listdir(self.demo_folder):
            if constants.DEMO_PARTIAL_RUN_LABEL in filename:
                num_partial_logs += 1

        logging.info(f"Num partial logs: {num_partial_logs}")
        for i in range(1, num_partial_logs + 1):
            if self.demo_compressed:
                with gzip.open(
                    self.demo_folder / utils.make_partial_run_label(i), "rb"
                ) as file:
                    p = pkl.Unpickler(file)
                    partial_log = p.load()
                    for key in self.demo_data:
                        self.demo_data[key] += partial_log[key]
            else:
                 with (self.demo_folder / utils.make_partial_run_label(i)).open(mode='rb') as file:
                    p = pkl.Unpickler(file)
                    partial_log = p.load()
                    for key in self.demo_data:
                        self.demo_data[key] += partial_log[key]


def parse_args():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Post Process a Demo")
    parser.add_argument("--demo_folder", type=str, required=True)
    parser.add_argument("--demo_name", type=str, required=True)
    parser.add_argument("--run_num", type=int, required=True)
    parser.add_argument("--demo_compressed", action='store_true')

    # define actions to do
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--single_example", action="store_true")

    args = parser.parse_args()
    return args


def main():
    inputs = parse_args()
    logging.info(inputs)

    demo_processor = DemoPostProcessor(
        demo_folder=inputs.demo_folder,
        demo_name=inputs.demo_name,
        run_number=inputs.run_num,
        demo_compressed=inputs.demo_compressed
    )

    if inputs.video:
        demo_processor.make_video()

    if inputs.single_example:
        demo_processor.single_example()


if __name__ == "__main__":
    main()
