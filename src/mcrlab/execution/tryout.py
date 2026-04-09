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
                                    preprocess_data, get_preprocessing_transform
from mcrlab.point_cloud.inspect import print_pc, visualize
from mcrlab.projection import bev_projection_numba, bev_projection_numba_and_open3d, bev_back_projection
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
        tiles, meta = bev_projection_numba(point_cloud, tile_size=35.0, resolution=0.05, include_class=True)  #  tile_size=100.0/50.0, resolution=0.2/0.1
        # if point_cloud.bevs is None:
        #     print("Starting BEV projection...")
        #     tiles, meta = bev_projection_numba(point_cloud, tile_size=35.0, resolution=0.05)  #  tile_size=100.0/50.0, resolution=0.2/0.1
        # else:
        #     print("Loaded Bevs from file...")
        #     tiles = point_cloud.bevs
        #     meta = point_cloud.meta

        # TEST START
        print("Starting BEV test...")
        points = point_cloud.point[get_coordinate_attribute(point_cloud)].numpy()
        intensities = point_cloud.point[get_intensity_attribute(point_cloud)].numpy().ravel()
        labels = point_cloud.point[get_class_attribute(point_cloud)].numpy().astype(np.int32)
        num_classes = int(labels.max()) + 1

        total_pixels = 0
        non_empty_pixels = 0
        correct_intensities = 0
        intensity_difference = 0
        total_classes = 0
        correct_class = 0
        total_empty_pixels = 0
        empty_pixels_correct = 0

        for tile_id, bev in tqdm(enumerate(tiles), total=len(tiles), desc="Tile Testing"):
            height, width = bev.shape[1], bev.shape[2]

            for cur_x in range(width):
                for cur_y in range(height):
                    total_pixels += 1

                    remapping = bev_back_projection(point_cloud, meta, tile_id, 
                                                    pixel_x=cur_x, pixel_y=cur_y, 
                                                    try_use_saved_local_points=False)
                    points_idx = remapping["global_indices"]

                    # empty pixel
                    if len(points_idx) == 0:
                        total_empty_pixels += 1
                        # if bev[3, cur_x, cur_y] == -1:
                        if bev[3, cur_y, cur_x] == -1:
                            empty_pixels_correct += 1
                        continue

                    non_empty_pixels += 1

                    points_idx = np.array(points_idx).astype(np.int32)

                    # intensity
                    y_mean_intensity = intensities[points_idx].mean()
                    y_mean_intensity /= intensities.max()  # apply same normalization
                    # bev_intensity = bev[2, cur_x, cur_y]
                    bev_intensity = bev[2, cur_y, cur_x]

                    # print(f"Intensity Ground Truth: {y_mean_intensity}, predicted: {bev_intensity}")

                    # same order? -> first closest sort or bad?

                    intensity_difference += np.sum(np.abs(bev_intensity - y_mean_intensity))
                    if np.isclose(y_mean_intensity, bev_intensity, atol=1e-4):
                        correct_intensities += 1

                    # classes
                    pixel_labels = labels[points_idx].ravel()
                    # print(f"Pixel label aount {pixel_labels.shape[0]} -> {pixel_labels}")
                    total_classes += pixel_labels.shape[0]

                    # print(f"Pixel Labels Shape: {pixel_labels.shape} -> {pixel_labels}")
                    # print(f"  -> Num Classes: {num_classes}")
                    bincount = np.bincount(pixel_labels, minlength=num_classes)
                    y_class = np.argmax(bincount)

                    # bev_class = int(bev[3, cur_x, cur_y])
                    bev_class = int(bev[3, cur_y, cur_x])

                    if bev_class == y_class:
                        correct_class += 1

        print("\n===== BEV TEST RESULTS =====")
        print(f"Total pixels checked: {total_pixels}")
        print(f"Correct intensities: {correct_intensities} ({(correct_intensities/non_empty_pixels)*100:.2f}%)")
        print(f"    -> absolute error sum: {intensity_difference}")
        print(f"Correct classes: {correct_class} ({(correct_class/non_empty_pixels)*100:.2f}%)")
        print(f"Correct empty pixels: {empty_pixels_correct} ({(empty_pixels_correct/total_empty_pixels)*100:.2f}%)")

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
    bev_working_testing(config)
    # train_data_testing(config)
    # train_testing(config)

    

    