from navigation import constants

def make_run_label(run_count: int) -> str:
    return constants.DEMO_RUN_LABEL+str(run_count)

def make_partial_run_label(partial_run_count: int) -> str:
    return constants.DEMO_PARTIAL_RUN_LABEL+str(partial_run_count)+'.pkl'

def get_empty_demo_data() -> None:
    demo_data = {}

    # add command key
    demo_data[constants.COMMAND_KEY] = []

    # add image keys
    for cam_name in constants.CAMERA_IMAGE_NAMES:
        demo_data[cam_name] = []
        
    return demo_data