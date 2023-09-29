from navigation import constants

def make_run_label(run_count: int) -> str:
    return constants.DEMO_RUN_LABEL+str(run_count)

def make_partial_run_label(partial_run_count: int) -> str:
    return constants.DEMO_PARTIAL_RUN_LABEL+str(partial_run_count)+'.pkl'

def get_empty_demo_data() -> None:
    return {constants.DEMO_RGB_KEY: [], constants.DEMO_DEPTH_KEY: [], constants.DEMO_COMMAND_KEY: []}