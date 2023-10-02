import pathlib
import torch

# DEMO
DEMO_BASE_SAVE_DIR = "navigation/demos_models/saved_demos"
DEMO_BASE_PATH = pathlib.Path(DEMO_BASE_SAVE_DIR)
DEMO_RUN_LABEL = "run"
DEMO_PARTIAL_RUN_LABEL = "log"

DEMO_RGB_KEY = "Image1st"
DEMO_DEPTH_KEY = "DepthImg"
DEMO_COMMAND_KEY = "Commands"

# VISION
NUM_CAMERAS = 3
NUM_GAITS = 3
COMMAND_NAMES = ["x", "y", "yaw", "gait"]
NUM_COMMANDS = len(COMMAND_NAMES)
TRAINED_MODEL_PATH = pathlib.Path("navigation/demos_models/trained_vision_models")

VISION_RGB_KEY = "Image1st"
VISION_DEPTH_KEY = "DepthImg"
VISION_COMMAND_KEY = "Commands"

CAMERA_NAMES = ['forward_rgb', 'forward_depth', 'downward_rgb', 'downward_depth']
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
}
GAIT_NAMES = [WALK_GAIT_NAME, CLIMB_GAIT_NAME, DUCK_GAIT_NAME]
CLIMB_GAIT_PATH = pathlib.Path("navigation/demos_models/trained_gait_policies/climb")
WALK_GAIT_PATH = pathlib.Path("navigation/demos_models/trained_gait_policies/walk")

# MISC
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
