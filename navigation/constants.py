import pathlib
import torch

# DEMO
DEMO_BASE_SAVE_DIR = "navigation/data/raw_demos"
DEMO_BASE_PATH = pathlib.Path(DEMO_BASE_SAVE_DIR)
DEMO_RUN_LABEL = "run"
DEMO_PARTIAL_RUN_LABEL = "log"

# VISION
NUM_CAMERAS = 3
NUM_GAITS = 3
COMMAND_NAMES = ["y", "yaw", "gait"]
COMMAND_KEY = "Commands"
NUM_COMMANDS = len(COMMAND_NAMES)
TRAINED_VISION_MODEL_PATH = pathlib.Path("navigation/data/trained_vision_models")

FORWARD_RGB_CAMERA = 'forward_rgb'
FORWARD_DEPTH_CAMERA = 'forward_depth'
DOWNWARD_RGB_CAMERA = 'downward_rgb'
DOWNWARD_DEPTH_CAMERA = 'downward_depth'
CAMERA_IMAGE_NAMES = [FORWARD_RGB_CAMERA, FORWARD_DEPTH_CAMERA, DOWNWARD_RGB_CAMERA, DOWNWARD_DEPTH_CAMERA]
CAMERA_FORWARD_RELATIVE_LOC = [0.0 , 0.0, 0.5]
CAMERA_DOWNWARD_RELATIVE_LOC = []
CAMERA_HEIGHT_WIDTH = []

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
CLIMB_GAIT_PATH = pathlib.Path("navigation/data/trained_controllers/climb")
WALK_GAIT_PATH = pathlib.Path("navigation/data/trained_controllers/walk")

# MISC
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
