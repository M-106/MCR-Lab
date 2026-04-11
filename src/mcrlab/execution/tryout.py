# -----------
# > Imports <
# -----------
import numpy as np
import matplotlib.pyplot as plt
import torch
import open3d as o3d

from tqdm import tqdm

# get secrets
import os
from dotenv import load_dotenv

from mcrlab.point_cloud.data import ParisLille3DDataset, get_data_loader, get_basic_transform, \
                                    preprocess_data, get_preprocessing_transform, \
                                    bev_gen_wrapper, extract_tiles_metas
from mcrlab.point_cloud.inspect import print_pc, visualize
from mcrlab.point_cloud.tensor_wrapper import PointCloudTensor
from mcrlab.projection import bev_projection_numba, bev_back_projection, bev_back_projection_testing
from mcrlab.image.utils import normalize_img_per_channel
from mcrlab.image.io import save_bev_tiles_as_images
from mcrlab.models.segmentation import SegFormer, SAM2, SAM3, DinoMask2Former
from mcrlab.point_cloud.utils import get_coordinate_attribute, get_intensity_attribute, \
                                     get_class_attribute



# -----------------------
# > Different Scenarios <
# -----------------------
def simple_viusalize_point_cloud(config):
    if config.data.name == "paris":
        dataset = ParisLille3DDataset(path=config.data.path, testdata=False, transform=None, 
                                      preprocessed=True, return_train_format=False)

    point_cloud = next(iter(dataset))

    print_pc(point_cloud)
    visualize(point_cloud, color_mode="class")



def torch_tensor_loading(config):
    # PyTorch Dataset try out
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    testdata=False, 
                                    transform=get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=1,
                                    preprocessed=True, return_train_format=False)

    for batch in data_loader:
        point_cloud = batch[0]

        print_pc(point_cloud)

        visualize(point_cloud, color_mode="class")

        break



def bev_trying(config):
    # PyTorch Dataset try out
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    testdata=False, 
                                    transform=get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=1,
                                    preprocessed=True, return_train_format=False)

    for batch in data_loader:
        point_cloud = batch[0]
        print_pc(point_cloud)

        if point_cloud.bev_data is None:
            print("Starting BEV projection...")
            tiles, metas = bev_projection_numba(point_cloud, tile_size=35.0, resolution=0.05)  #  tile_size=100.0/50.0, resolution=0.2/0.1
            # bev_gen = bev_gen_wrapper(tiles, metas)
        else:
            bev_gen = point_cloud.get_bev()
            tiles, metas = extract_tiles_metas(bev_gen, amount=5, as_numpy=True)

        print("Tile 1 Shape:", tiles[0].shape)

        tile_1_img = np.transpose(tiles[0], (1, 2, 0))
        tile_1_img = normalize_img_per_channel(tile_1_img, skip_already_normalized_channels=True)

        tile_1_intensity_channel = tile_1_img[:, :, 2]
        print("Intensity Channel:\n  Min:", tile_1_intensity_channel.min())
        print("  Max:", tile_1_intensity_channel.max())
        print("  Std:", tile_1_intensity_channel.std())
        plt.imshow(tile_1_intensity_channel, cmap="nipy_spectral")  #"gnuplot2", "nipy_spectral", "gist_rainbow", "rainbow"
        plt.show()

        # plt.imshow(tile_1_img[:, :, 2])
        # plt.show()
        # plt.imshow(tile_1_img[:, :, 1])
        # plt.show()
        save_bev_tiles_as_images(tiles, folder="./test_bev_images")

        break_ = False
        for cur_x in np.arange(0, tile_1_img.shape[0], dtype=int):
            for cur_y in np.arange(0, tile_1_img.shape[1], dtype=int):
                if tile_1_img[cur_y][cur_x][1] != 0:
                    remapping = bev_back_projection(point_cloud, meta, tile_id=0, pixel_x=cur_x, pixel_y=cur_y)
                    points = remapping["points"]
                    print(points)
                    print(type(points))
                    break_ = True
                    break
            if break_:
                break

        # show back propagated point -> hard to see ...
        tile_1_img[:, :, 1] = 0
        tile_1_img[cur_y, cur_x, 1] = 255
        plt.imshow(tile_1_img[:, :, 1])
        plt.show()
        point_cloud.coordinates = torch.cat((point_cloud.coordinates, torch.tensor([[points[0][0], points[0][1], points[0][2]]])), dim=0)
        point_cloud.colors = torch.zeros((point_cloud.coordinates.shape[0], 3), dtype=torch.uint8)
        point_cloud.colors[point_cloud.coordinates.shape[0]-1] = torch.Tensor([0, 255, 0])
        visualize(point_cloud, color_mode=None)

        break



def bev_segmentation_trying(config):
    # load all variables from .env file into os.environ
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    # PyTorch Dataset try out
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    testdata=False, 
                                    transform=get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=1,
                                    preprocessed=True, return_train_format=False)

    for batch in data_loader:
        point_cloud = batch[0]
        print_pc(point_cloud)

        if point_cloud.bev_data is None:
            print("Starting BEV projection...")
            tiles, metas = bev_projection_numba(point_cloud, tile_size=35.0, resolution=0.05)  #  tile_size=100.0/50.0, resolution=0.2/0.1
            # bev_gen = bev_gen_wrapper(tiles, metas)
        else:
            bev_gen = point_cloud.get_bev()
            tiles, metas = extract_tiles_metas(bev_gen, amount=5, as_numpy=True)

        print("Tile 1 Shape:", tiles[0].shape)

        tile_1_img = np.transpose(tiles[0], (1, 2, 0))
        tile_1_img = normalize_img_per_channel(tile_1_img, skip_already_normalized_channels=True)

        tile_1_intensity_channel = tile_1_img[:, :, 2]
        print("Intensity Channel:\n  Min:", tile_1_intensity_channel.min())
        print("  Max:", tile_1_intensity_channel.max())
        print("  Std:", tile_1_intensity_channel.std())
        
        # just for debugging
        save_bev_tiles_as_images(tiles, folder="./test_bev_images")

        # try segmentation
        print("Try making a segmentation on BEV images...")

        model_name = config.model.name.lower()
        if model_name == "segmformer":
            model = SegFormer(device=-1)
        elif model_name == "sam2":
            model = SAM2(hf_token=hf_token, device=-1)
        elif model_name == "sam3":
            model = SAM3(hf_token=hf_token, device=-1)
        elif model_name == "dinomask2former":
            model = DinoMask2Former(device=-1)

        with torch.inference_mode():
            results = model.predict(tile_1_intensity_channel)
        
        # visualize
        model.visualize(tile_1_intensity_channel, results)

        break



def bev_working_testing(config):
    # LOAD POINT CLOUD
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    testdata=False, 
                                    transform=None,  # get_basic_transform(num_points=-1), 
                                    batch_size=1, shuffle=False, num_workers=1,
                                    preprocessed=True, return_train_format=False)

    for batch in data_loader:
        point_cloud = batch[0]

        # point_cloud = point_cloud.get_as_o3d()
        if not isinstance(point_cloud, o3d.t.geometry.PointCloud):
            raise TypeError(f"Point Cloud should be get as Open3D Tensor, but got '{type(point_cloud)}'")
        print_pc(point_cloud)

        print("Starting BEV projection...")
        # tiles, meta = bev_projection_numba_and_open3d(point_cloud, tile_size=35.0, resolution=0.05, include_class=True)
        tiles, metas = bev_projection_numba(point_cloud, tile_size=35.0, resolution=0.05, include_class=True)  #  tile_size=100.0/50.0, resolution=0.2/0.1
        bev_gen = bev_gen_wrapper(tiles, metas)
        
        # if point_cloud.bevs is None:
        #     print("Starting BEV projection...")
        #     tiles, meta = bev_projection_numba(point_cloud, tile_size=35.0, resolution=0.05)  #  tile_size=100.0/50.0, resolution=0.2/0.1
        # else:
        #     print("Loaded Bevs from file...")
        #     tiles = point_cloud.bevs
        #     meta = point_cloud.meta

        bev_back_projection_testing(point_cloud, bev_gen)

        # do not end after one testset?
        break



def bev_preprocessed_loading_working_testing(config):
    # LOAD POINT CLOUD
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    testdata=False, 
                                    transform=get_basic_transform(num_points=-1), 
                                    batch_size=1, shuffle=False, num_workers=1,
                                    preprocessed=True, return_train_format=False)

    for batch in data_loader:
        point_cloud = batch[0]

        assert isinstance(point_cloud, PointCloudTensor)
        print_pc(point_cloud)

        print("Starting BEV projection...")
        if point_cloud.bev_data is None:
            raise ValueError("Preprocessed BEVs did not loaded.")
            print("Starting BEV projection...")
            tiles, metas = bev_projection_numba(point_cloud, tile_size=35.0, resolution=0.05)  #  tile_size=100.0/50.0, resolution=0.2/0.1
        else:
            print("Loaded Bevs from file...")
            bev_gen = point_cloud.get_bev()

        bev_back_projection_testing(point_cloud.get_as_o3d(), bev_gen)

        # do not end after one testset?
        break



def train_data_testing(config):
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    testdata=False, 
                                    transform=get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=1,
                                    preprocessed=True, return_train_format=True)

    for idx, (x_batch, y_batch) in enumerate(data_loader):
        print(f"Data check:")
        print(f"X:\n    Type: {type(x_batch)}")
        print(f"        -> {x_batch.dtype}") if hasattr(x_batch, "dtype") else "nothing"
        print(f"    Shape: {x_batch.shape}") if hasattr(x_batch, "shape") else "nothing"

        print(f"Y:\n    Type: {type(y_batch)}")
        print(f"        -> {y_batch.dtype}") if hasattr(y_batch, "dtype") else "nothing"
        print(f"    Shape: {y_batch.shape}") if hasattr(y_batch, "shape") else "nothing"

        break

    # FIXME -> continue



def train_testing(config):
    pass



# --------------
# > Playground <
# --------------
def tryout(config):
    # simple_viusalize_point_cloud(config)
    # torch_tensor_loading(config)
    # bev_trying(config)
    # bev_segmentation_trying(config)
    # bev_working_testing(config)
    bev_preprocessed_loading_working_testing(config)
    # train_data_testing(config)
    # train_testing(config)

    

    