# License: see [LICENSE, LICENSES/legged_gym/LICENSE]

import math

import numpy as np
from isaacgym import terrain_utils
from numpy.random import choice
import pandas as pd
from scipy.interpolate import interp2d
from tensorflow.keras.models import load_model
import cv2
import torch

from go1_gym.envs.base.legged_robot_config import Cfg


class Terrain:
    def __init__(self, cfg: Cfg.terrain, num_robots, eval_cfg=None, num_eval_robots=0, use_generated_terrain=False) -> None:

        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.train_rows, self.train_cols, self.eval_rows, self.eval_cols = self.load_cfgs()
        self.tot_rows = len(self.train_rows) #+ len(self.eval_rows)
        self.tot_cols = len(self.train_cols)#, len(self.eval_cols))
        self.cfg.env_length = cfg.terrain_length
        self.cfg.env_width = cfg.terrain_width

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        self.initialize_terrains()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(self.height_field_raw,
                                                                                         self.cfg.horizontal_scale,
                                                                                         self.cfg.vertical_scale,
                                                                                         self.cfg.slope_treshold)



    def load_cfgs(self):
        self._load_cfg(self.cfg)
        self.cfg.row_indices = np.arange(0, self.cfg.tot_rows)
        self.cfg.col_indices = np.arange(0, self.cfg.tot_cols)
        self.cfg.x_offset = 0
        self.cfg.rows_offset = 0
        # if self.eval_cfg is None:
        return self.cfg.row_indices, self.cfg.col_indices, [], []
        # else:
        #     self._load_cfg(self.eval_cfg)
        #     self.eval_cfg.row_indices = np.arange(self.cfg.tot_rows, self.cfg.tot_rows + self.eval_cfg.tot_rows)
        #     self.eval_cfg.col_indices = np.arange(0, self.eval_cfg.tot_cols)
        #     self.eval_cfg.x_offset = self.cfg.tot_rows
        #     self.eval_cfg.rows_offset = self.cfg.num_rows
        #     return self.cfg.row_indices, self.cfg.col_indices, self.eval_cfg.row_indices, self.eval_cfg.col_indices

    def _load_cfg(self, cfg):
        cfg.proportions = [np.sum(cfg.terrain_proportions[:i + 1]) for i in range(len(cfg.terrain_proportions))]

        cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        cfg.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        cfg.width_per_env_pixels = int(cfg.terrain_length / cfg.horizontal_scale)
        cfg.length_per_env_pixels = int(cfg.terrain_width / cfg.horizontal_scale)

        cfg.border = int(cfg.border_size / cfg.horizontal_scale)
        cfg.tot_cols = int(cfg.num_cols * cfg.width_per_env_pixels) + 2 * cfg.border
        cfg.tot_rows = int(cfg.num_rows * cfg.length_per_env_pixels) + 2 * cfg.border

        print(cfg.tot_cols, cfg.tot_rows)

    def initialize_terrains(self):
        self._initialize_terrain(self.cfg)
        if self.eval_cfg is not None:
            self._initialize_terrain(self.eval_cfg)

    def _initialize_terrain(self, cfg):
        if cfg.generated:
            self.generated_terrain(cfg)
        elif cfg.curriculum:
            self.curriculum(cfg)
        elif cfg.selected:
            self.selected_terrain(cfg)
        else:
            self.randomized_terrain(cfg)

    def randomized_terrain(self, cfg):
        for k in range(cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(cfg, choice, difficulty, cfg.proportions)
            self.add_terrain_to_map(cfg, terrain, i, j)

    def generated_terrain(self,cfg,map_file=True):

        ########################
        # Generate terrain and downsample or upsample to cfg.length_per_env_pixels x cfg.length_per_env_pixels

        # envWidth, downsampleRes, myRes = cfg.length_per_env_pixels, 0.035 / 50.0 * 225, 0.035
        envWidth, downsampleRes, myRes = cfg.length_per_env_pixels, 0.035 / 50.0 * 225, 0.035
        # myCorStart, mySize = int(-112*myRes/downsampleRes), int(225*myRes/downsampleRes)
        myCorStart, mySize = -envWidth // 2, envWidth #int(-112*myRes/downsampleRes), int(225*myRes/downsampleRes)
        myGrids = np.linspace(-112*myRes, 112*myRes, num=225)
        interpGrids = np.linspace(myCorStart*downsampleRes,(myCorStart+mySize-1)*downsampleRes, num=mySize)
        algoHeightScale, myHeightScale = 0.005, myRes * 15

        if map_file:

            #TODO: MAKE SURE TO CHANGE TERRAIN VAR IN DEMO IF THIS FILENAME IS CHANGED

            myMap = pd.read_csv(f'scripts/terrain_benchmark-main/hard/elevation{cfg.generated_name}.txt', sep=' ', header=None).values[::-1,:].T
        else:
            w_noise = np.random.normal(0, 1, (1, 14, 14, 1024))
            source1 = np.random.rand(225,225,1)<25/225/225
            source2 = np.random.rand(225,225,1)<25/225/225
            source = np.concatenate([source1,source2],axis=-1)[None]
            
            generator = load_model('scripts/terrain_benchmark-main/terrain_generation/terrain_generator50000.h5')

            myMap = generator([source, w_noise])[0][:,:,0]


        myDS = interp2d(myGrids, myGrids, myMap)
        interpHeight = myDS(interpGrids, interpGrids)* myHeightScale / algoHeightScale


        terrain = terrain_utils.SubTerrain("terrain",
                                               width=cfg.width_per_env_pixels,
                                               length=cfg.width_per_env_pixels,
                                               vertical_scale=cfg.vertical_scale,
                                               horizontal_scale=cfg.horizontal_scale)
        #################################

        for j in range(cfg.num_cols):
            for i in range(cfg.num_rows):

                # map coordinate system
                start_x = cfg.border + i * cfg.length_per_env_pixels + cfg.x_offset
                end_x = cfg.border + (i + 1) * cfg.length_per_env_pixels + cfg.x_offset
                start_y = cfg.border + j * cfg.width_per_env_pixels
                end_y = cfg.border + (j + 1) * cfg.width_per_env_pixels

                
                self.height_field_raw[start_x:end_x,
                        start_y:end_y]=interpHeight
            

                env_origin_x = (i + 0.5) * cfg.terrain_length + cfg.x_offset * terrain.horizontal_scale
                env_origin_y = int((j + 0.5) * cfg.terrain_width)
                env_origin_z = np.max(self.height_field_raw[start_x: end_x, start_y:end_y]) * terrain.vertical_scale

                cfg.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


        if self.cfg.generated_diff=='easy':
            scale_fac=3.5
        elif self.cfg.generated_diff=='medium':
            scale_fac=2.5
        else:
            scale_fac=2.0

        # start pad
        self.height_field_raw[2:40,2:30]=np.floor_divide(self.height_field_raw[2:40,2:30],3.5)

        # right
        self.height_field_raw[10:15,30:75] =np.floor_divide(self.height_field_raw[10:15,30:75],scale_fac)

        # down
        self.height_field_raw[14:55,69:75]=np.floor_divide(self.height_field_raw[14:55,69:75],scale_fac)

        # left
        self.height_field_raw[49:55,74:120]=np.floor_divide(self.height_field_raw[49:55,74:120],scale_fac)

        # down
        self.height_field_raw[54:120,114:120]=np.floor_divide(self.height_field_raw[54:120,114:120],scale_fac)

        # left
        self.height_field_raw[114:120,119:130]=np.floor_divide(self.height_field_raw[114:120,119:130],scale_fac)

        # down
        self.height_field_raw[119:130,124:130]=np.floor_divide(self.height_field_raw[119:130,124:130],scale_fac)

        # end
        self.height_field_raw[130:149,124:149]=np.floor_divide(self.height_field_raw[130:149,124:149],3.5)



    def curriculum(self, cfg):
        for j in range(cfg.num_cols):
            for i in range(cfg.num_rows):
                difficulty = i / cfg.num_rows * cfg.difficulty_scale
                choice = j / cfg.num_cols + 0.001

                terrain = self.make_terrain(cfg, choice, difficulty, cfg.proportions)
                self.add_terrain_to_map(cfg, terrain, i, j)

    def selected_terrain(self, cfg):
        terrain_type = cfg.terrain_kwargs.pop('type')
        for k in range(cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (cfg.num_rows, cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                                               width=cfg.width_per_env_pixels,
                                               length=cfg.width_per_env_pixels,
                                               vertical_scale=cfg.vertical_scale,
                                               horizontal_scale=cfg.horizontal_scale)

            eval(terrain_type)(terrain, **cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(cfg, terrain, i, j)

    def make_terrain(self, cfg, choice, difficulty, proportions):
        terrain = terrain_utils.SubTerrain("terrain",
                                           width=cfg.width_per_env_pixels,
                                           length=cfg.width_per_env_pixels,
                                           vertical_scale=cfg.vertical_scale,
                                           horizontal_scale=cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * (cfg.max_platform_height - 0.05)
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        if choice < proportions[0]:
            if choice < proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05,
                                                 step=self.cfg.terrain_smoothness, downsampled_scale=0.2)
        elif choice < proportions[3]:
            if choice < proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size,
                                                     rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size,
                                                  stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < proportions[6]:
            pass
        elif choice < proportions[7]:
            pass
        elif choice < proportions[8]:
            terrain_utils.random_uniform_terrain(terrain, min_height=-cfg.terrain_noise_magnitude,
                                                 max_height=cfg.terrain_noise_magnitude, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < proportions[9]:
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05,
                                                 step=self.cfg.terrain_smoothness, downsampled_scale=0.2)
            terrain.height_field_raw[0:terrain.length // 2, :] = 0

        return terrain

    def add_terrain_to_map(self, cfg, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = cfg.border + i * cfg.length_per_env_pixels + cfg.x_offset
        end_x = cfg.border + (i + 1) * cfg.length_per_env_pixels + cfg.x_offset
        start_y = cfg.border + j * cfg.width_per_env_pixels
        end_y = cfg.border + (j + 1) * cfg.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * cfg.terrain_length + cfg.x_offset * terrain.horizontal_scale
        env_origin_y = (j + 0.5) * cfg.terrain_width
        x1 = int((cfg.terrain_length / 2. - 1) / terrain.horizontal_scale) + cfg.x_offset
        x2 = int((cfg.terrain_length / 2. + 1) / terrain.horizontal_scale) + cfg.x_offset
        y1 = int((cfg.terrain_width / 2. - 1) / terrain.horizontal_scale)
        y2 = int((cfg.terrain_width / 2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(self.height_field_raw[start_x: end_x, start_y:end_y]) * terrain.vertical_scale

        cfg.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def perlin(x, y, octaves, lacunarity, gain, seed=0):
    np.random.seed(seed)
    total = 0
    amplitude = 1.0
    for _ in range(octaves):
        total += perlin_layer(x, y) * amplitude
        x *= lacunarity
        y *= lacunarity
        amplitude *= gain
    return total
    
def perlin_layer(x, y):
    # permutation table
    p = np.arange(10000, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi, yi = x.astype(int), y.astype(int)
    # internal coordinates
    xf, yf = x - xi, y - yi
    # fade factors
    u, v = fade(xf), fade(yf)
    # noise components
    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
    return lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here

def lerp(a, b, x):
    "linear interpolation"
    return a + x * (b - a)

def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h, x, y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y
