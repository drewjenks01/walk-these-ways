import yaml
import numpy as np
import torch
import logging
import os
from PIL import Image as PILImage

from navigation.vision.models.base_model import BaseModel
from navigation import constants
from navigation.vision.models.navigators.vint_utils.vint import ViNT


class ViNTNavigator(BaseModel):
    MODEL_WEIGHTS = 'navigation/vision/models/navigators/vint/vint.pth'
    TOPOMAP_DIR = 'navigation/data/topomaps/'
    MODEL_CONFIG_PATH = 'navigation/vision/models/navigators/vint/vint.yaml'
    RADIUS = 4
    GOAL_NODE = -1
    CLOSE_THRESHOLD = 3
    WAYPOINT = 2
    CONTEXT_SIZE = 5
    EPS = 1e-8
    DT = 1/constants.FPS
    
    def __init__(self, pretrained: bool, topomap_folder:str):
        if pretrained:
            self.model = torch.load(ViNTNavigator.MODEL_WEIGHTS, device=constants.DEVICE)
            self.model.to(constants.DEVICE)
            self.model.eval()
        else:
            self.model = ViNT().to(constants.DEVICE)

        self.topomap_info = self.load_topomap(topomap_folder)

        with open(ViNTNavigator.MODEL_CONFIG_PATH, "r") as f:
            self.model_params = yaml.safe_load(f)

        self.context_queue = []

    def forward(self, numpy_img):
        pil_img = self.numpy_to_pil(numpy_img)
        self.update_context_queue(pil_img)
        chosen_waypoint = self.get_chosen_waypoint()
        lin_v, ang_v = self.get_vels_from_waypoint(chosen_waypoint)
        outs = {
            'y_vel': lin_v,
            'yaw': ang_v,
            'gait': None
        }
        return outs

    def get_chosen_waypoint(self):
        chosen_waypoint = np.zeros(4)
        start = max(self.topomap_info['closest_node'] - ViNTNavigator.RADIUS, 0)
        end = min(self.topomap_info['closest_node'] + ViNTNavigator.RADIUS + 1, ViNTNavigator.GOAL_NODE)
        distances = []
        waypoints = []
        batch_obs_imgs = []
        batch_goal_data = []
        for i, sg_img in enumerate(self.topomap_info['topomap'][start: end + 1]):
            transf_obs_img = self.transform_images(self.context_queue, self.model_params["image_size"])
            goal_data = self.transform_images(sg_img, self.model_params["image_size"])
            batch_obs_imgs.append(transf_obs_img)
            batch_goal_data.append(goal_data)
            
        # predict distances and waypoints
        batch_obs_imgs = torch.cat(batch_obs_imgs, dim=0).to(constants.DEVICE)
        batch_goal_data = torch.cat(batch_goal_data, dim=0).to(constants.DEVICE)

        distances, waypoints = self.model(batch_obs_imgs, batch_goal_data)
        distances = self.to_numpy(distances)
        waypoints = self.to_numpy(waypoints)
        # look for closest node
        self.topomap_info['closest_node'] = np.argmin(distances)
        # chose subgoal and output waypoints
        if distances[self.topomap_info['closest_node']] > ViNTNavigator.CLOSE_THRESHOLD:
            chosen_waypoint = waypoints[self.topomap_info['closest_node']][ViNTNavigator.WAYPOINT]
            sg_img = self.topomap_info['topomap'][start + self.topomap_info['closest_node']]
        else:
            chosen_waypoint = waypoints[min(
                self.topomap_info['closest_node'] + 1, len(waypoints) - 1)][ViNTNavigator.WAYPOINT]
            sg_img = self.topomap_info['topomap'][start + min(self.topomap_info['closest_node'] + 1, len(waypoints) - 1)]     
        # RECOVERY MODE
        if self.model_params["normalize"]:
            chosen_waypoint[:2] *= (ViNTNavigator.MAX_V / ViNTNavigator.RATE)  

        self.topomap_info['reached_goal'] = self.topomap_info['closest_node'] == self.topomap_info['goal_node'] 
        return chosen_waypoint

    def get_vels_from_waypoint(self, waypoint):
       
        def clip_angle(theta) -> float:
            """Clip angle to [-pi, pi]"""
            theta %= 2 * np.pi
            if -np.pi < theta < np.pi:
                return theta
            return theta - 2 * np.pi

        # if reached goal then stop
        if self.topomap_info['reached_goal']:
            logging.info('Goal has been reached. Stopping navigation...')
            return 0, 0
        
        assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
        if len(waypoint) == 2:
            dx, dy = waypoint
        else:
            dx, dy, hx, hy = waypoint
        # this controller only uses the predicted heading if dx and dy near zero
        if len(waypoint) == 4 and np.abs(dx) < ViNTNavigator.EPS and np.abs(dy) < ViNTNavigator.EPS:
            v = 0
            w = clip_angle(np.arctan2(hy, hx))/ViNTNavigator.DT		
        elif np.abs(dx) < ViNTNavigator.EPS:
            v =  0
            w = np.sign(dy) * np.pi/(2*ViNTNavigator.DT)
        else:
            v = dx / ViNTNavigator.DT
            w = np.arctan(dy/dx) / ViNTNavigator.DT
        v = np.clip(v, 0, ViNTNavigator.MAX_V)
        w = np.clip(w, -ViNTNavigator.MAX_W,ViNTNavigator.MAX_W)
        return v, w

    def load_topomap(self, topomap_folder):
        # load topomap
        topomap_path = ViNTNavigator.TOPOMAP_DIR + topomap_folder
        topomap_filenames = sorted(os.listdir(topomap_path), key=lambda x: int(x.split(".")[0]))
        num_nodes = len(os.listdir(topomap_path))
        topomap = []
        for i in range(num_nodes):
            image_path = os.path.join(topomap_path, topomap_filenames[i])
            topomap.append(PILImage.open(image_path))

        goal_node = -1 # last node in topomap
        closest_node = 0
        assert -1 <= goal_node < len(topomap), "Invalid goal index"
        if goal_node == -1:
            goal_node = len(topomap) - 1
        else:
            goal_node = goal_node
        reached_goal = False

        topomap_info = {
            'topomap': topomap,
            'closest_node': closest_node,
            'reached_goal': reached_goal,
            'goal_node': goal_node
        }
        return topomap_info