from go1_gym.envs.base.legged_robot_config import Cfg as walk_cfg
from parkour.envs import parkour_legged_robot
from parkour.envs.parkour_legged_robot_config import LeggedRobotCfg as parkour_cfg
from go1_gym.envs.base import legged_robot
from go1_gym.envs.wrappers import no_yaw_wrapper
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv
from go1_gym.utils import *
from parkour.utils import task_registry
from navigation import constants

class MultiGaitWrapper:

    VALID_POLICIES = {'walk', 'parkour'}

    def __init__(self, walk_cfg, parkour_cfg):
    
        self.climb_bool = False     # if climb, then yaw bool matters for walk controller
        self.walk_env = no_yaw_wrapper.NoYawWrapper(VelocityTrackingEasyEnv(sim_device="cuda:0", headless=False, cfg=walk_cfg), self.climb_bool)
        self.parkour_env, _ = task_registry.make_env(name='a1', env_cfg=parkour_cfg)

        self.current_policy = 'walk'    # walk, parkour -- duck and climb within walk


    def step(self):
        if self.current_policy == 'walk':
            return self.walk_env.step()
        elif self.current_policy == 'parkour':
            return self.parkour_env.step()
        
    def reset(self):
        if self.current_policy == 'walk':
            return self.walk_env.reset()
        elif self.current_policy == 'parkour':
            return self.parkour_env.reset()
    
    def render(self):
        if self.current_policy == 'walk':
            return self.walk_env.render()
        elif self.current_policy == 'parkour':
            return self.parkour_env.render()
        

    def prepare_parkour_policy(self):
        parkour_cfg, parkour_train_cfg = task_registry.get_cfgs(name='a1')
        self.parkour_env, _ = task_registry.make_env(name='a1', env_cfg=self.parkour_cfg)
        parkour_train_cfg.runner.resume = True
        ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=self.parkour_env, name='a1', args=None, train_cfg=train_cfg, return_log_dir=True)
        policy = torch.jit.load(constants.PARKOUR_DEPTH_GAIT_PATH / )

    
    def change_current_controller(self, controller_name: str):
        assert controller_name in MultiGaitWrapper.VALID_CONTROLLERS, 'Not a valid controller name'
        self.current_controller = controller_name

