import pathlib
import torch

# DEMO
DEMO_BASE_SAVE_DIR = "navigation/saved_demos"
DEMO_BASE_PATH = pathlib.Path(DEMO_BASE_SAVE_DIR)
DEMO_RUN_LABEL = "run"
DEMO_PARTIAL_RUN_LABEL = "log"

DEMO_RGB_KEY = "Image1st"
DEMO_DEPTH_KEY = "DepthImg"
DEMO_COMMAND_KEY = "Commands"

# VISION
NUM_CAMERAS = 3
NUM_GAITS = 3
GAIT_NAMES = ["walk", "stair", "duck"]
COMMAND_NAMES = ["x", "y", "yaw", "gait"]
NUM_COMMANDS = len(COMMAND_NAMES)
TRAINED_MODEL_PATH = pathlib.Path("navigation/vision/trained_models")

VISION_RGB_KEY = "Image1st"
VISION_DEPTH_KEY = "DepthImg"
VISION_COMMAND_KEY = "Commands"

# MISC
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
