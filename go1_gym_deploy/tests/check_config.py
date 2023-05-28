def check_config():
    import pickle as pkl

    config_filename = "../../navigation/commandnet/runs/run_recent/multi_comm/stata/dino/config.pkl"

    with open(config_filename, "rb") as f:
        config = pkl.load(f)

    print("Config:")
    print(config)

if __name__ == "__main__":
    check_config()