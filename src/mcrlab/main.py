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
    if config.mode == "custom_train":
        from mcrlab.execution.custom_train import train
        train(config)

    elif config.mode == "train":
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
                        type=config.data.type,  # maybe also define over config!
                        device="cpu",
                        bev_tile_size=5.0, bev_resolution=0.01, bev_overlap=0.5,  # bev_tile_size=35.0, bev_resolution=0.05)
                        file_ending=config.preprocessing.file_ending)
        
    elif config.mode == "tryout":
        from mcrlab.execution.tryout import tryout
        tryout(config)
    
    elif config.mode == "eval_extraction":
        if config.eval_extraction.generate_2d_gt_maps:
            from mcrlab.execution.eval_extraction import ground_truth_extraction_2d
            ground_truth_extraction_2d(config)
        else:
            from mcrlab.execution.eval_extraction import ground_truth_extraction
            ground_truth_extraction(config)
        
    else:
        raise ValueError(f"'{config.mode}' is not an available mode for mcrlab.")




