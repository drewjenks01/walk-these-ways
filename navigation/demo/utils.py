from navigation import constants

def make_run_label(run_count: int) -> str:
    return constants.DEMO_RUN_LABEL+str(run_count)

def make_partial_run_label(partial_run_count: int) -> str:
    return constants.DEMO_PARTIAL_RUN_LABEL+str(partial_run_count)+'.pkl'

def get_empty_demo_command_data() -> None:
    demo_data = {}

    # add command key
    for key in constants.COMMAND_NAMES:
        demo_data[key] = []
        
    return demo_data

def get_empty_demo_image_data() -> None:
    demo_data = {}

    # add command key
    for key in constants.CAMERA_IMAGE_NAMES:
        demo_data[key] = []
        
    return demo_data