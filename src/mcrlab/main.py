

# config
import argparse
from mcrlab.config.config import Config, load_config
from mcrlab.train import train
from mcrlab.test import test
from mcrlab.point_cloud.io import ParisLille3DDataset, get_paris_data_loader, get_basic_transform
from mcrlab.point_cloud.inspect import get_info, get_metrics, visualize

import open3d as o3d

def main():
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="./configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    print(config.mode)
    print(config.model)
    print(config.train.batch_size)

    # do something
    if config.mode == "train":
        from mcrlab.train import train
        train(config)

    elif config.mode == "test":
        from mcrlab.test import test
        test(config)
    elif config.mode == "tryout":
        # r"C:/Users/tippolito/Data/Benchmark"
        # "../data/paris_lille_3d"
        dataset = ParisLille3DDataset(path=r"C:/Users/tippolito/Data/Benchmark", testdata=True, transform=None)
        point_cloud = next(iter(dataset))
        get_info(point_cloud)
        get_metrics(point_cloud)
        visualize(point_cloud)

        # PyTorch Dataset try out
        data_loader = get_paris_data_loader(r"C:/Users/tippolito/Data/Benchmark", testdata=True, 
                                            transform=get_basic_transform(),
                                            batch_size=1, shuffle=False, num_workers=1)

        for batch in data_loader:
            point_cloud = batch[0]
            get_info(point_cloud)
            get_metrics(point_cloud)
            visualize(point_cloud)
            break
    else:
        raise ValueError(f"'{config.mode}' is not an available mode for mcrlab.")

