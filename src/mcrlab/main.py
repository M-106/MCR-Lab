# -----------
# > Imports <
# -----------

# config
import argparse
from mcrlab.config.config import load_config

from mcrlab.point_cloud.data import preprocess_data, get_preprocessing_transform



# --------------
# > Main Logic <
# --------------
def main():
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="./configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # print(config.mode)
    # print(config.model)
    # print(config.train.batch_size)
    # print(config.data.path)
    print("Configuration:")
    print(config)
    print("\n")



    # do something
    if config.mode == "train":
        from mcrlab.execution.train import train
        train(config)

    elif config.mode == "test":
        from mcrlab.execution.test import test
        test(config)

    elif config.mode == "test":
        from mcrlab.execution.inference import inference
        inference(config)

    elif config.mode == "preprocessing":
        preprocess_data(config.data.name, config.data.path, 
                        testdata=False,
                        device="cpu",
                        bev_tile_size=15.0, bev_resolution=0.01, bev_overlap=0.5)  # bev_tile_size=35.0, bev_resolution=0.05)
        
    elif config.mode == "tryout":
        from mcrlab.execution.tryout import tryout
        tryout(config)
        
    else:
        raise ValueError(f"'{config.mode}' is not an available mode for mcrlab.")

