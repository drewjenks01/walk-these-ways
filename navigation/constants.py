import pathlib
import torch

# PATHS
NAVIGATION_PATH = pathlib.Path('navigation')
MODELS_PATH = NAVIGATION_PATH / 'models'
DATA_PATH = NAVIGATION_PATH / 'data'
DEMO_PATH = NAVIGATION_PATH / 'demo'

RAW_DEMOS_PATH = DATA_PATH / 'raw_demos'
DEMO_TOPOMAPS_PATH = DATA_PATH / 'vint_topomaps'

TRAINED_VISION_MODEL_PATH = MODELS_PATH / 'trained_models'
TRAINED_CONTROLLERS_PATH = MODELS_PATH / 'trained_controllers'

CLIMB_GAIT_PATH = TRAINED_CONTROLLERS_PATH / 'climb'
WALK_GAIT_PATH = TRAINED_CONTROLLERS_PATH / 'walk'
PARKOUR_DEPTH_GAIT_PATH =TRAINED_CONTROLLERS_PATH / 'parkour_depth'


# ROBOT
MAX_V = 1
MAX_W = 1

# DEMO
DEMO_RUN_LABEL = "run"
DEMO_PARTIAL_RUN_LABEL = "log"

# VISION
NUM_CAMERAS = 3
NUM_GAITS = 3
COMMAND_NAMES = ["y_vel", "yaw", "gait"]
COMMAND_KEY = "Commands"
NUM_COMMANDS = len(COMMAND_NAMES)

FORWARD_RGB_CAMERA = 'forward_rgb'
FORWARD_DEPTH_CAMERA = 'forward_depth'
DOWNWARD_RGB_CAMERA = 'downward_rgb'
DOWNWARD_DEPTH_CAMERA = 'downward_depth'
CAMERA_IMAGE_NAMES = [FORWARD_RGB_CAMERA, FORWARD_DEPTH_CAMERA, DOWNWARD_RGB_CAMERA, DOWNWARD_DEPTH_CAMERA]
CAMERA_FORWARD_RELATIVE_LOC = [0.0 , 0.0, 0.5]
CAMERA_DOWNWARD_RELATIVE_LOC = []
CAMERA_HEIGHT_WIDTH = []
FPS = 4
IMAGE_ASPECT_RATIO = (4 / 3) 
CONTEXT_SIZE = 5

# GAITS
WALK_GAIT_NAME = "walk"
WALK_GAIT_PARAMS = {
    "step_frequency_cmd": 3.0,
    "gait": torch.tensor([0.5, 0, 0]),
    "footswing_height_cmd": 0.08,
    "pitch_cmd": 0.0,
    "roll_cmd": 0.0,
    "stance_width_cmd": 0.35,
    "body_height_cmd": 0.1,
    'yaw_obs_bool': 0
}
CLIMB_GAIT_NAME = "climb"
CLIMB_GAIT_PARAMS = {
    "step_frequency_cmd": 2.0,
    "gait": torch.tensor([0.5, 0, 0]),
    "footswing_height_cmd": 0.30,
    "pitch_cmd": 0.0,
    "roll_cmd": 0.0,
    "stance_width_cmd": 0.35,
    "body_height_cmd": 0.1,
    'yaw_obs_bool': 1
}
DUCK_GAIT_NAME = "duck"
DUCK_GAIT_PARAMS = {
    "step_frequency_cmd": 3.0,
    "gait": torch.tensor([0.5, 0, 0]),
    "footswing_height_cmd": 0.08,
    "pitch_cmd": 0.0,
    "roll_cmd": 0.0,
    "stance_width_cmd": 0.35,
    "body_height_cmd": -0.2,
    'yaw_obs_bool': 0
}
GAIT_NAMES = [WALK_GAIT_NAME, CLIMB_GAIT_NAME, DUCK_GAIT_NAME]

# MISC
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
