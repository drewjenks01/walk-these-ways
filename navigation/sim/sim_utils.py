from isaacgym import gymapi, torch_utils, gymtorch
import torch
import numpy as np
import matplotlib.pyplot as plt

from navigation import constants


def render_forward_depth_rgb(env):
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    depth_image_ = env.gym.get_camera_image_gpu_tensor(
        env.sim, env.envs[0], env.cam_handles[0], gymapi.IMAGE_DEPTH
    )
    depth_image = gymtorch.wrap_tensor(depth_image_)
    rgb_image = env.gym.get_camera_image_gpu_tensor(
        env.sim, env.envs[0], env.cam_handles[0], gymapi.IMAGE_COLOR
    )
    return rgb_image, depth_image


def render_first_third_imgs(env, view_imgs=False):
    """
    Renders and returns a first person and third person viewpoint image of robot.
    """
    bx, by, bz = env.root_states[0, 0], env.root_states[0, 1], env.root_states[0, 2]
    forward = torch_utils.quat_apply(env.base_quat, env.forward_vec)
    heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)

    # collect 1st person view
    env.gym.set_camera_location(
        env.rendering_camera,
        env.envs[0],
        gymapi.Vec3(bx, by, bz + 0.5),
        gymapi.Vec3(
            bx.item() + 1.5 * np.cos(heading.item()),
            by.item() + 1.5 * np.sin(heading.item()),
            bz,
        ),
    )
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    img = env.gym.get_camera_image(
        env.sim, env.envs[0], env.rendering_camera, gymapi.IMAGE_COLOR
    )
    w, h = img.shape
    first_person = img.reshape([w, h // 4, 4])

    # collect 3rd person view
    env.gym.set_camera_location(
        env.rendering_camera,
        env.envs[0],
        gymapi.Vec3(
            bx.item() - 1.2 * np.cos(heading.item()),
            by.item() - 1.2 * np.sin(heading.item()),
            bz + 1,
        ),
        gymapi.Vec3(
            bx.item() + np.cos(heading.item()), by.item() + np.sin(heading.item()), bz
        ),
    )
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    img = env.gym.get_camera_image(
        env.sim, env.envs[0], env.rendering_camera, gymapi.IMAGE_COLOR
    )
    w, h = img.shape
    third_person = img.reshape([w, h // 4, 4])

    if view_imgs:
        fig, ax = plt.subplots(2, 1)
        ax[0].imshow(first_person)
        ax[0].set_title("First Person")
        # ax[1].imshow(third_person)
        # ax[1].set_title('Third Person')
        plt.show()

    return first_person[:, :, :3], third_person[:, :, :3]


def update_sim_view(env, offset=[-1, -1, 1]):
    """
    Updates the camera view in sim to follow heading of robot.
    """
    bx, by, bz = env.root_states[0, 0], env.root_states[0, 1], env.root_states[0, 2]
    # print('robot loc:', bx,by,bz)
    forward = torch_utils.quat_apply(env.base_quat, env.forward_vec)
    heading = torch.atan2(forward[:, 1], forward[:, 0]).unsqueeze(1)
    # print('Robot loc: ',[bx.item(),by.item(),bz.item()], 'Heading:',heading.item(),'forward:',list(forward))
    env.set_camera(
        [
            bx.item() - np.cos(heading.item()),
            by.item() - np.sin(heading.item()),
            bz + offset[2],
        ],
        [bx.item() + np.cos(heading.item()), by.item() + np.sin(heading.item()), bz],
    )


def create_xbox_controller():
    """
    Creates a listener that reads inputs from xbox controller to control robot and various sim functions.

    To use:

        xbox = create_xbox_controller()

        for #steps:
            commands = xbox.read()
    """
    from navigation.sim.controllers.xbox_controller import XboxController

    xbox = XboxController()
    return xbox


def create_keyboard_controller():
    """
    Creates a listener that reads inputs from keyboard to control robot and various sim functions.

    To use:

        keyboard = create_keyboard_controller()

        for #steps:
            commands = keyboard.commands
    """
    from navigation.sim.controllers.keyboard_controller import KeyboardController

    keyboard = KeyboardController()
    return keyboard
