

# config
import argparse
from mcrlab.config.config import Config, load_config
from mcrlab.train import train
from mcrlab.test import test
from mcrlab.point_cloud.data import ParisLille3DDataset, get_data_loader, get_basic_transform, \
                                    preprocess_data, get_preprocessing_transform
from mcrlab.point_cloud.inspect import print_pc, visualize

import open3d as o3d

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
        from mcrlab.train import train
        train(config)

    elif config.mode == "test":
        from mcrlab.test import test
        test(config)
    elif config.mode == "preprocessing":
        preprocess_data(config.data.name, config.data.path, 
                        testdata=False, transform=get_preprocessing_transform(), 
                        device="cpu")
    elif config.mode == "tryout":
        
        if config.data.name == "paris":
            dataset = ParisLille3DDataset(path=config.data.path, testdata=False, transform=None)
        point_cloud = next(iter(dataset))
        print_pc(point_cloud)
        # visualize(point_cloud, color_mode="class")

        # PyTorch Dataset try out
        if config.data.name == "paris":
            data_loader = get_data_loader(config.data.name, config.data.path, 
                                          testdata=False, 
                                          transform=get_basic_transform(num_points=-1),
                                          batch_size=4, shuffle=False, num_workers=1,
                                          preprocessed=True)
            
            # data_loader = get_paris_data_loader(config.data.path, testdata=False, 
            #                                     transform=None,
            #                                     batch_size=4, shuffle=False, num_workers=1)

        for batch in data_loader:
            point_cloud = batch[0]
            print_pc(point_cloud)
            visualize(point_cloud)
            break
    else:
        raise ValueError(f"'{config.mode}' is not an available mode for mcrlab.")

