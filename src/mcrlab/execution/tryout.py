# -----------
# > Imports <
# -----------
import numpy as np
import matplotlib.pyplot as plt
import torch

# get secrets
import os
from dotenv import load_dotenv

from mcrlab.point_cloud.data import ParisLille3DDataset, get_data_loader, get_basic_transform, \
                                    preprocess_data, get_preprocessing_transform
from mcrlab.point_cloud.inspect import print_pc, visualize
from mcrlab.projection import bev_projection_numba, bev_projection_mapping
from mcrlab.image.utils import normalize_img_per_channel
from mcrlab.image.io import save_bev_tiles_as_images
from mcrlab.models.segmentation import SegFormer, SAM2, SAM3, DinoMask2Former



# --------------
# > Playground <
# --------------
def tryout(config):
    # load all variables from .env file into os.environ
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    # if config.data.name == "paris":
    #     dataset = ParisLille3DDataset(path=config.data.path, testdata=False, transform=None, preprocessed=True)
    # point_cloud = next(iter(dataset))
    # print_pc(point_cloud)
    # visualize(point_cloud, color_mode="class")

    # PyTorch Dataset try out
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    testdata=False, 
                                    transform=get_basic_transform(num_points=-1), # get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=1,
                                    preprocessed=True, return_train_format=False)
    
    # data_loader = get_paris_data_loader(config.data.path, testdata=False, 
    #                                     transform=None,
    #                                     batch_size=4, shuffle=False, num_workers=1)

    for batch in data_loader:
        point_cloud = batch[0]
        print_pc(point_cloud)
        # visualize(point_cloud, color_mode="class")

        if point_cloud.bevs is None:
            print("Starting BEV projection...")
            tiles, meta = bev_projection_numba(point_cloud, tile_size=35.0, resolution=0.05)  #  tile_size=100.0/50.0, resolution=0.2/0.1
        else:
            tiles = point_cloud.bevs
            meta = point_cloud.meta

        print("Tile 1 Shape:", tiles[0].shape)

        tile_1_img = np.transpose(tiles[0], (1, 2, 0))
        tile_1_img = normalize_img_per_channel(tile_1_img, skip_already_normalized_channels=True)

        tile_1_intensity_channel = tile_1_img[:, :, 2]
        print("Intensity Channel:\n  Min:", tile_1_intensity_channel.min())
        print("  Max:", tile_1_intensity_channel.max())
        print("  Std:", tile_1_intensity_channel.std())

        # plt.imshow(tile_1_img[:, :, 2])
        # plt.show()
        # plt.imshow(tile_1_img[:, :, 1])
        # plt.show()
        save_bev_tiles_as_images(tiles, folder="./test_bev_images")

        # break_ = False
        # for cur_x in np.arange(0, tile_1_img.shape[0], dtype=int):
        #     for cur_y in np.arange(0, tile_1_img.shape[1], dtype=int):
        #         if tile_1_img[cur_x][cur_y][1] != 0:
        #             points = bev_projection_mapping(point_cloud, meta, tile_id=0, pixel=(cur_x, cur_y))
        #             print(points)
        #             print(type(points))
        #             break_ = True
        #             break
        #     if break_:
        #         break

        # show back propagated point 
        # tile_1_img[:, :, 1] = 0
        # tile_1_img[cur_x, cur_y, 1] = 255
        # plt.imshow(tile_1_img[:, :, 1])
        # plt.show()
        # point_cloud.coordinates = torch.cat((point_cloud.coordinates, torch.tensor([[points[0][0], points[0][1], points[0][2]]])), dim=0)
        # point_cloud.colors = torch.zeros((point_cloud.coordinates.shape[0], 3), dtype=torch.uint8)
        # point_cloud.colors[point_cloud.coordinates.shape[0]-1] = torch.Tensor([0, 255, 0])
        # visualize(point_cloud, color_mode=None)

        # try segmentation
        print("Try making a segmentation on BEV images...")

        # model = SegFormer(device=-1)
        model = SAM2(hf_token=hf_token, device=-1)
        # model = SAM3(hf_token=hf_token, device=-1)
        # model = DinoMask2Former(device=-1)

        with torch.inference_mode():
            results = model.predict(tile_1_intensity_channel)
        # mask_data = list(results.items())[0]
        # x, y, w, h = mask_data['bbox']
        # ax.text(x, y, f"{mask_data['predicted_iou']:.2f}", color='white', fontsize=8)
        plt.imshow(tile_1_intensity_channel, cmap="nipy_spectral")  #"gnuplot2", "nipy_spectral", "gist_rainbow", "rainbow"
        plt.show()
        model.visualize(tile_1_intensity_channel, results)

        break