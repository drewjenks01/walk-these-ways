from params_proto import PrefixProto, ParamsProto
import isaacgym
assert isaacgym
class RunCfg(PrefixProto, cli=False):
    experiment_group = "example_sweep"
    experiment_job_type= "default"


import torch

from go1_gym.envs.base.legged_robot_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from go1_gym_learn.ppo_cse import Runner
from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
from go1_gym_learn.ppo_cse.actor_critic import AC_Args
from go1_gym_learn.ppo_cse.ppo import PPO_Args
from go1_gym_learn.ppo_cse import RunnerArgs

from collections import namedtuple

def configure_env():

    config_go1(Cfg)

    Cfg.env.num_envs = 1000
    Cfg.robot.name = "go1"
    Cfg.sensors.sensor_names = [
                        "OrientationSensor",
                        "RCSensor",
                        "JointPositionSensor",
                        "JointVelocitySensor",
                        "ActionSensor",
                        # "JointPositionErrorSensor",
                        # "LastActionSensor",
                        # "ClockSensor",
                        # "YawSensor",
                        ]
    Cfg.sensors.sensor_args = {
                        "OrientationSensor": {},
                        "RCSensor": {},
                        "JointPositionSensor": {},
                        "JointVelocitySensor": {},
                        "ActionSensor": {},
                        # "JointPositionErrorSensor": {},
                        # "LastActionSensor": {"delay": 1},
                        # "ClockSensor": {},
                        # "YawSensor": {},
                        }
    
    Cfg.sensors.privileged_sensor_names = [
                        # "FrictionSensor",
                        # "RestitutionSensor",
                        "BodyVelocitySensor",
                        # "BodyHeightSensor",
                        # "HeightmapSensor",
    ]
    Cfg.sensors.privileged_sensor_args = {
                        # "FrictionSensor": {},
                        # "RestitutionSensor": {},
                        "BodyVelocitySensor": {},
                        # "BodyHeightSensor": {},
                        # "HeightmapSensor": {},
                        
    }

    Cfg.control.decimation = 4
    Cfg.sim.dt = 0.005 * 4. / Cfg.control.decimation

    Cfg.commands.num_lin_vel_bins = 30
    Cfg.commands.num_ang_vel_bins = 30
    Cfg.curriculum_thresholds.tracking_ang_vel = 0.7
    Cfg.curriculum_thresholds.tracking_lin_vel = 0.8
    Cfg.curriculum_thresholds.tracking_lin_vel_balanced = 0.6
    Cfg.curriculum_thresholds.tracking_contacts_shaped_vel = 0.90
    Cfg.curriculum_thresholds.tracking_contacts_shaped_force = 0.90

    Cfg.commands.distributional_commands = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    # Cfg.control.control_type = "actuator_net"
    Cfg.control.control_type = "P"

    Cfg.domain_rand.randomize_rigids_after_start = False
    # Cfg.env.priv_observe_motion = False
    # Cfg.env.priv_observe_gravity_transformed_motion = False
    Cfg.domain_rand.randomize_friction_indep = False
    # Cfg.env.priv_observe_friction_indep = False
    Cfg.domain_rand.randomize_friction = True
    # Cfg.env.priv_observe_friction = True
    Cfg.domain_rand.friction_range = [1.2, 4.0] # originally 0.1 to 3.0
    Cfg.domain_rand.randomize_restitution = True
    # Cfg.env.priv_observe_restitution = True
    Cfg.domain_rand.restitution_range = [1.0, 1.0] # originally to 0.4
    Cfg.domain_rand.randomize_base_mass = True
    # Cfg.env.priv_observe_base_mass = False
    Cfg.domain_rand.added_mass_range = [-0.0, 3.0]
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.gravity_range = [-0.0, 0.0]
    Cfg.domain_rand.gravity_rand_interval_s = 8.0
    Cfg.domain_rand.gravity_impulse_duration = 0.5
    # Cfg.env.priv_observe_gravity = False
    Cfg.domain_rand.randomize_com_displacement = True
    Cfg.domain_rand.com_displacement_range = [-0.2, 0.2]
    # Cfg.env.priv_observe_com_displacement = False
    Cfg.domain_rand.randomize_ground_friction = True
    # Cfg.env.priv_observe_ground_friction = False
    # Cfg.env.priv_observe_ground_friction_per_foot = False
    Cfg.domain_rand.ground_friction_range = [0.0, 0.01]
    Cfg.domain_rand.randomize_motor_strength = True
    Cfg.domain_rand.motor_strength_range = [0.8, 1.2]
    # Cfg.env.priv_observe_motor_strength = False
    Cfg.domain_rand.randomize_motor_offset = True
    Cfg.domain_rand.motor_offset_range = [-0.001, 0.001]
    # Cfg.env.priv_observe_motor_offset = False
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_Kp_factor = False
    # Cfg.env.priv_observe_Kp_factor = False
    Cfg.domain_rand.randomize_Kd_factor = False
    # Cfg.env.priv_observe_Kd_factor = False
    # Cfg.env.priv_observe_body_velocity = False
    # Cfg.env.priv_observe_body_height = False
    # Cfg.env.priv_observe_desired_contact_states = False
    # Cfg.env.priv_observe_contact_forces = False
    # Cfg.env.priv_observe_foot_displacement = False
    # Cfg.env.priv_observe_gravity_transformed_foot_displacement = False

    Cfg.env.num_privileged_obs = 3 #+ 187
    Cfg.env.num_observation_history = 10

    Cfg.init_state.pos = [0.0, 0.0, 0.40]

    Cfg.domain_rand.rand_interval_s = 4
    Cfg.commands.num_commands = 15
    # Cfg.env.observe_two_prev_actions = True
    # Cfg.env.observe_yaw = False
    Cfg.env.num_observations = 54
    Cfg.env.num_scalar_observations = 54
    Cfg.env.observe_gait_commands = True
    # Cfg.env.observe_timing_parameter = False
    # Cfg.env.observe_clock_inputs = True
    # Cfg.env.episode_length_s = 5.0
    Cfg.commands.heading_command = False

    Cfg.domain_rand.tile_height_range = [-0.0, 0.0]
    Cfg.domain_rand.tile_height_curriculum = False
    Cfg.domain_rand.tile_height_update_interval = 1000000
    Cfg.domain_rand.tile_height_curriculum_step = 0.01

    # Roughness
    Cfg.domain_rand.randomize_tile_roughness = True
    Cfg.domain_rand.tile_roughness_range = [0.03, 0.08] # [0.02, 0.1]

    Cfg.terrain.parkour = True
    Cfg.terrain.perlin_octaves = 5
    Cfg.terrain.perlin_lacunarity = 3.0
    Cfg.terrain.perlin_gain = 0.45

    Cfg.terrain.mesh_type = "trimesh" #originally "plane"
    # Cfg.domain_rand.ground_friction_range = [2.0, 2.01] # originally 0 to 1
    # Cfg.domain_rand.ground_restitution_range = [0, 0.1] # originally 0 to 1

    Cfg.terrain.border_size = 25
    # Cfg.terrain.mesh_type = "trimesh"
    Cfg.terrain.num_cols = 40
    Cfg.terrain.num_rows = 10
    Cfg.terrain.terrain_width = 4.0
    Cfg.terrain.terrain_length = 18.0
    Cfg.terrain.x_init_range = 0.2
    Cfg.terrain.y_init_range = 0.2
    Cfg.terrain.teleport_thresh = 0.3
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.center_robots = False
    Cfg.terrain.center_span = 4
    Cfg.terrain.horizontal_scale = 0.05
    Cfg.terrain.vertical_scale = 0.005
    Cfg.terrain.max_init_terrain_level = 5
    Cfg.terrain.num_border_boxes = 2
    Cfg.terrain.static_friction = 1.0
    Cfg.terrain.dynamic_friction = 1.0
    Cfg.terrain.restitution = 0.0
    Cfg.terrain.measure_heights = True
    Cfg.terrain.slope_treshold = 1.5
    Cfg.terrain.selected = False
    Cfg.terrain.terrain_kwargs = None
    Cfg.terrain.max_init_terrain_level = 5
    Cfg.terrain.terrain_proportions = list(Cfg.terrain.terrain_dict.values())

    Cfg.terrain.curriculum = True
    Cfg.terrain.max_step_height = 0.16
    Cfg.terrain.min_step_height = 0.02
    Cfg.terrain.min_init_terrain_level = 0
    Cfg.terrain.max_init_terrain_level = 5
    Cfg.terrain.platform_size = 0.7
    Cfg.terrain.difficulty_scale = 0.4

    # Cfg.terrain.curriculum = False
    # Cfg.terrain.terrain_proportions = [0.4, 0.2, 0.2, 0.0, 0.0, 0.2, 0.0]
    # Cfg.terrain.max_step_height = 0.16
    # Cfg.terrain.min_step_height = 0.02
    # Cfg.terrain.min_init_terrain_level = 0
    # Cfg.terrain.max_init_terrain_level = 0
    # Cfg.terrain.platform_size = 0.7
    # Cfg.terrain.difficulty_scale = 0.4

    Cfg.rewards.reward_container_name = 'ParkourRewards'
    Cfg.rewards.use_terminal_foot_height = False
    Cfg.rewards.use_terminal_body_height = False
    Cfg.rewards.terminal_body_height = 0.05
    Cfg.rewards.use_terminal_roll_pitch = True
    Cfg.rewards.terminal_body_ori = 0.9
    Cfg.rewards.only_positive_rewards = True

    Cfg.commands.resampling_time = 10


    Cfg.rewards.base_height_target = 0.30
    Cfg.rewards.kappa_gait_probs = 0.07
    Cfg.rewards.gait_force_sigma = 100.
    Cfg.rewards.gait_vel_sigma = 10.

    # Cfg.rewards.only_positive_rewards = False
    Cfg.rewards.only_positive_rewards_ji22_style = False
    Cfg.rewards.sigma_rew_neg = 0.02

    # Cfg.reward_scales.tracking_x_vel = -20.0
    # Cfg.reward_scales.tracking_other_vels = -1.0
    # Cfg.reward_scales.survival = 20.0

    # Cfg.reward_scales.energy = args.energy_coef
    # Cfg.reward_scales.energy_analytic = 0.0

    Cfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_v2.urdf'
    Cfg.asset.penalize_contacts_on = ["thigh", "calf"]
    Cfg.asset.terminate_after_contacts_on = ["base"]

    # Cfg.reward_scales.energy = 0.0
    Cfg.reward_scales.energy_analytic = -0.0002
    # Cfg.reward_scales.energy_footswing_bonus = 0.0

    Cfg.reward_scales.tracking_lin_vel = 0.0 #1.0
    # Cfg.reward_scales.tracking_lin_vel_balanced = 1.0 #0.0
    Cfg.reward_scales.tracking_ang_vel = 0.0 #0.5
    # Cfg.reward_scales.vel = 0.0 #1.0
    # # Cfg.reward_scales.tracking_lin_vel_integral = 1.0
    # # Cfg.reward_scales.tracking_ang_vel_integral = 0.5

    Cfg.reward_scales.tracking_goal_vel = 1.5
    Cfg.reward_scales.tracking_yaw = 0.0 #0.5
    # regularization rewards
    Cfg.reward_scales.lin_vel_z = -1.0
    Cfg.reward_scales.ang_vel_xy = -0.05
    Cfg.reward_scales.orientation = -1.
    Cfg.reward_scales.dof_acc = -2.5e-7
    Cfg.reward_scales.collision = -10.
    Cfg.reward_scales.action_rate = -0.1
    Cfg.reward_scales.delta_torques = -1.0e-7
    Cfg.reward_scales.torques = -0.00001
    Cfg.reward_scales.hip_pos = -0.5
    Cfg.reward_scales.dof_error = -0.04
    Cfg.reward_scales.feet_stumble = -1
    # Cfg.reward_scales.feet_edge = -1

    # Cfg.reward_scales.lin_vel_z = -0.0
    # Cfg.reward_scales.ang_vel_xy = -0.0
    # Cfg.reward_scales.orientation = -0.0
    # Cfg.reward_scales.torques = -0.0
    Cfg.reward_scales.dof_vel = -0.0
    Cfg.reward_scales.dof_pos_limits = 0.0
    Cfg.reward_scales.arm_dof_vel = 0.0
    # Cfg.reward_scales.dof_acc = -0.0
    Cfg.reward_scales.arm_dof_acc = 0.0
    Cfg.reward_scales.base_height = -0.0
    Cfg.reward_scales.feet_air_time = 0.0
    Cfg.reward_scales.collision = -0.0
    # Cfg.reward_scales.feet_stumble = -0.0
    # Cfg.reward_scales.action_rate = -0.0
    # Cfg.reward_scales.energy_action_smoothness_1 = -0.001
    # Cfg.reward_scales.energy_action_smoothness_2 = -0.001

    AC_Args.adaptation_labels = ["body_velocity_loss"]
    AC_Args.adaptation_dims = [3]
    AC_Args.activation = 'lrelu'
    AC_Args.init_noise_std = 1.0
    PPO_Args.learning_rate = 3e-4
    PPO_Args.entropy_coef = 0.01
    PPO_Args.lam = 0.97

    Cfg.commands.lin_vel_x = [0., 1.]
    Cfg.commands.lin_vel_y = [0., 0.]
    Cfg.commands.ang_vel_yaw = [-1., 1.]

    Cfg.commands.body_height_cmd = [0.0, 0.0]
    Cfg.commands.gait_frequency_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_phase_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_offset_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_bound_cmd_range = [0.0, 0.0]
    Cfg.commands.gait_duration_cmd_range = [0.0, 0.0]
    Cfg.commands.footswing_height_range = [0.0, 0.0]
    Cfg.commands.body_pitch_range = [-0.0, 0.0]
    Cfg.commands.body_roll_range = [-0.0, 0.0]
    Cfg.commands.stance_width_range = [0.0, 0.0]
    Cfg.commands.stance_length_range = [0.0, 0.0]

    Cfg.commands.limit_vel_x = [0., 1.]
    Cfg.commands.limit_vel_y = [-0., 0.]
    Cfg.commands.limit_vel_yaw = [-1., 1.]

    Cfg.commands.limit_body_height = [-0.0, 0.0]
    Cfg.commands.limit_gait_frequency = [0.0, 0.0]
    Cfg.commands.limit_gait_phase = [0.0, 0.0]
    Cfg.commands.limit_gait_offset = [0.0, 0.0]
    Cfg.commands.limit_gait_bound = [0.0, 0.0]
    Cfg.commands.limit_gait_duration = [0.0, 0.0]
    Cfg.commands.limit_footswing_height = [0.0, 0.0]
    Cfg.commands.limit_body_pitch = [-0.0, 0.0]
    Cfg.commands.limit_body_roll = [-0.0, 0.0]
    Cfg.commands.limit_stance_width = [0.0, 0.0]
    Cfg.commands.limit_stance_length = [0.0, 0.0]

    Cfg.commands.num_bins_vel_x = 1
    Cfg.commands.num_bins_vel_y = 1
    Cfg.commands.num_bins_vel_yaw = 1
    Cfg.commands.num_bins_body_height = 1
    Cfg.commands.num_bins_gait_frequency = 1
    Cfg.commands.num_bins_gait_phase = 1
    Cfg.commands.num_bins_gait_offset = 1
    Cfg.commands.num_bins_gait_bound = 1
    Cfg.commands.num_bins_gait_duration = 1
    Cfg.commands.num_bins_footswing_height = 1
    Cfg.commands.num_bins_body_roll = 1
    Cfg.commands.num_bins_body_pitch = 1
    Cfg.commands.num_bins_stance_width = 1

    Cfg.normalization.friction_range = [0, 1]
    Cfg.normalization.ground_friction_range = [0, 1]
    Cfg.terrain.yaw_init_range = 3.14
    Cfg.normalization.clip_actions = 10.0

    Cfg.commands.exclusive_phase_offset = False
    Cfg.commands.pacing_offset = False
    Cfg.commands.balance_gait_distribution = False
    Cfg.commands.binary_phases = False
    Cfg.commands.gaitwise_curricula = False

    return Cfg

def train_go1(headless=True, **deps):
    # configure
    Cfg = configure_env()

    AC_Args._update(deps)
    PPO_Args._update(deps)
    RunnerArgs._update(deps)
    RunCfg._update(deps)
    Cfg.terrain._update(deps)
    Cfg.commands._update(deps)
    Cfg.domain_rand._update(deps)
    Cfg.env._update(deps)
    Cfg.reward_scales._update(deps)
    Cfg.rewards._update(deps)
    Cfg.curriculum_thresholds._update(deps)
    Cfg.normalization._update(deps)
    Cfg.init_state._update(deps)
    Cfg.control._update(deps)
    Cfg.asset._update(deps)
    Cfg.noise_scales._update(deps)
    Cfg.perception._update(deps)

    # create the environment
    gpu_id = 0
    env = VelocityTrackingEasyEnv(sim_device=f'cuda:{gpu_id}', headless=headless, cfg=Cfg)
    Args = namedtuple('args', ['exp', 'alpha', 'init_path', 'lmbd', 'min_vel', 'max_vel', 'num_vel_itvl', 'conditional'])
    exp = 'eipo_trkv_enrg'
    env.logger.is_eipo = True
    alpha = 0.5
    init_path = None
    lmbd = 1.0
    min_vel = 0.0
    max_vel = 1.0
    num_vel_itvl = 11
    args = Args(exp=exp, alpha=alpha, init_path=init_path, lmbd=lmbd,
                min_vel=min_vel, max_vel=max_vel, num_vel_itvl=num_vel_itvl,
                conditional=False)
    env = HistoryWrapper(env, args)

    # log the experiment parameters
    import wandb
    wandb.init(
      # set the wandb project where this run will be logged
      project="walk-these-ways",
      group=RunCfg.experiment_group,
      job_type=RunCfg.experiment_job_type,
      # track hyperparameters and run metadata
      config={
      "AC_Args": vars(AC_Args),
      "PPO_Args": vars(PPO_Args),
      "RunnerArgs": vars(RunnerArgs),
      "Cfg": vars(Cfg),
      },
      mode="online"
    )

    # train
    runner = Runner(env, args, device=f"cuda:{gpu_id}")
    runner.learn(num_learning_iterations=10000, init_at_random_ep_len=True, eval_freq=100)

def play_go1(headless=True):
    # configure
    Cfg = configure_env()
    Cfg.env.num_envs = 1

    # create the environment
    env = VelocityTrackingEasyEnv(sim_device=f'cuda:{gpu_id}', headless=headless, cfg=Cfg)
    Args = namedtuple('args', ['exp'])
    exp = 'eipo_trkv_enrg'
    env.logger.is_eipo = True
    args = Args(exp=exp)
    env = HistoryWrapper(env, args)

    # simulate 100 steps
    obs = env.reset()
    for i in range(500):
        actions = torch.zeros((Cfg.env.num_envs, Cfg.env.num_actions), device=env.device)
        obs, reward, done, info = env.step(actions)

    # close the environment
    env.close()


if __name__ == '__main__':
    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR

    stem = Path(__file__).stem

    # to see the environment rendering, set headless=False

    train_go1(headless=False)
