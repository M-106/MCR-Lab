# -----------
# > Imports <
# -----------

# config
import argparse
from mcrlab.config.config import Config, load_config

from mcrlab.train import train
from mcrlab.test import test
from mcrlab.point_cloud.data import ParisLille3DDataset, get_data_loader, get_basic_transform, \
                                    preprocess_data, get_preprocessing_transform
from mcrlab.point_cloud.inspect import print_pc, visualize
from mcrlab.image.utils import bev_projection, bev_projection_mapping, normalize_img
from mcrlab.image.io import save_bev_tiles_as_images

import open3d as o3d

import matplotlib.pyplot as plt
import numpy as np



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
        
        # if config.data.name == "paris":
        #     dataset = ParisLille3DDataset(path=config.data.path, testdata=False, transform=None, preprocessed=True)
        # point_cloud = next(iter(dataset))
        # print_pc(point_cloud)
        # visualize(point_cloud, color_mode="class")

        # PyTorch Dataset try out
        data_loader = get_data_loader(config.data.name, config.data.path, 
                                        testdata=False, 
                                        transform=get_basic_transform(num_points=-1), # get_basic_transform(num_points=-1),
                                        batch_size=4, shuffle=False, num_workers=1,
                                        preprocessed=True)
        
        # data_loader = get_paris_data_loader(config.data.path, testdata=False, 
        #                                     transform=None,
        #                                     batch_size=4, shuffle=False, num_workers=1)

        for batch in data_loader:
            point_cloud = batch[0]
            # print_pc(point_cloud)
            # visualize(point_cloud, color_mode="class")

            tiles, meta = bev_projection(point_cloud, tile_size=50.0, resolution=0.1)
            print("Tile 1 Shape:", tiles[0].shape)

            tile_1_img = np.transpose(tiles[0], (1, 2, 0))
            tile_1_img = normalize_img(tile_1_img)
            plt.imshow(tile_1_img[:, :, 2])
            plt.show()
            plt.imshow(tile_1_img[:, :, 1])
            plt.show()
            save_bev_tiles_as_images(tiles, folder="./test_bev_images")

            for cur_x in np.arange(0, tile_1_img.shape[0], dtype=int):
                for cur_y in np.arange(0, tile_1_img.shape[1], dtype=int):
                    if tile_1_img[cur_x][cur_y][1] != 0:
                        points = bev_projection_mapping(point_cloud, meta, tile_id=0, pixel=(cur_x, cur_y))
                        print(points)
                        print(type(points))
                        # break
                        return 0
    else:
        raise ValueError(f"'{config.mode}' is not an available mode for mcrlab.")

