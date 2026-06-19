# -----------
# > Imports <
# -----------
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import open3d as o3d

from tqdm import tqdm

# get secrets
import os
from dotenv import load_dotenv

from mcrlab.point_cloud.data import ParisLille3DDataset, get_data_loader, get_basic_transform, \
                                    preprocess_data, get_preprocessing_transform, \
                                    bev_gen_wrapper, extract_tiles_metas, \
                                    BEVDataset
from mcrlab.point_cloud.inspect import print_pc, visualize, visualize_intensity_in_2d, \
                                       analyze_point_distribution
from mcrlab.point_cloud.tensor_wrapper import PointCloudTensor
from mcrlab.projection import bev_projection, bev_projection_testing
from mcrlab.image.utils import normalize_img_per_channel
from mcrlab.image.io import save_bev_tiles_as_images
from mcrlab.models.segmentation import SegFormer, SAM2, SAM3, DinoMask2Former
from mcrlab.point_cloud.utils import get_coordinate_attribute, get_intensity_attribute, \
                                     get_class_attribute, get_instance_attribute, \
                                     extract_manhole, add_random_dense_manipulation_point_cloud
from mcrlab.classic.shape_fit import use_label_candidates_and_extract_center_point, \
                                      use_points_and_extract_center_point, \
                                      classic_manhole_prediction_pipeline
from mcrlab.classic.utils import visualize_circle_fit, visualize_circle_shape_and_center_prediction, \
                                 visualize_ransac_inliers
from mcrlab.point_cloud.shape_check import circle_shape_check



# -----------------------
# > Different Scenarios <
# -----------------------
def simple_viusalize_point_cloud(config):
    # if config.data.name == "paris":
    #     dataset = ParisLille3DDataset(path=config.data.path, type="train", transform=None, 
    #                                   preprocessed=config.data.preprocessed, return_train_format=False)
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                  type=config.data.type, 
                                  transform=get_basic_transform(num_points=-1),
                                  batch_size=1, shuffle=False, num_workers=1,
                                  preprocessed=config.data.preprocessed, return_train_format=False)

    point_cloud = next(iter(data_loader))[0]

    print_pc(point_cloud)
    visualize(point_cloud, color_mode="class")



def torch_tensor_loading(config):
    # PyTorch Dataset try out
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=1,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    for batch in data_loader:
        point_cloud = batch[0]

        print_pc(point_cloud)

        visualize(point_cloud, color_mode="class")

        break



# def bev_trying(config):
#     # PyTorch Dataset try out
#     data_loader = get_data_loader(config.data.name, config.data.path, 
#                                     type=config.data.type, 
#                                     transform=get_basic_transform(num_points=-1),
#                                     batch_size=1, shuffle=False, num_workers=1,
#                                     preprocessed=config.data.preprocessed, return_train_format=False)

#     for batch in data_loader:
#         point_cloud = batch[0]
#         print_pc(point_cloud)

#         if point_cloud.bev_data is None:
#             print("Starting BEV projection...")
#             tiles, metas = bev_projection(point_cloud, tile_size=35.0, resolution=0.05, overlap=0.0,
#                                           include_class=False, direct_single_saving=False)  #  tile_size=100.0/50.0, resolution=0.2/0.1
#             # bev_gen = bev_gen_wrapper(tiles, metas)
#         else:
#             bev_gen = point_cloud.get_bev()
#             tiles, metas = extract_tiles_metas(bev_gen, amount=5, as_numpy=True)

#         print("Tile 1 Shape:", tiles[0].shape)

#         tile_1_img = np.transpose(tiles[0], (1, 2, 0))
#         tile_1_img = normalize_img_per_channel(tile_1_img, skip_already_normalized_channels=True)

#         tile_1_intensity_channel = tile_1_img[:, :, 2]
#         print("Intensity Channel:\n  Min:", tile_1_intensity_channel.min())
#         print("  Max:", tile_1_intensity_channel.max())
#         print("  Std:", tile_1_intensity_channel.std())
#         plt.imshow(tile_1_intensity_channel, cmap="nipy_spectral")  #"gnuplot2", "nipy_spectral", "gist_rainbow", "rainbow"
#         plt.show()

#         # plt.imshow(tile_1_img[:, :, 2])
#         # plt.show()
#         # plt.imshow(tile_1_img[:, :, 1])
#         # plt.show()
#         save_bev_tiles_as_images(tiles, folder="./test_bev_images")

#         break_ = False
#         for cur_x in np.arange(0, tile_1_img.shape[0], dtype=int):
#             for cur_y in np.arange(0, tile_1_img.shape[1], dtype=int):
#                 if tile_1_img[cur_y][cur_x][1] != 0:
#                     remapping = bev_back_projection(point_cloud, metas, tile_id=0, pixel_x=cur_x, pixel_y=cur_y)
#                     points = remapping["points"]
#                     print(points)
#                     print(type(points))
#                     break_ = True
#                     break
#             if break_:
#                 break

#         # show back propagated point -> hard to see ...
#         tile_1_img[:, :, 1] = 0
#         tile_1_img[cur_y, cur_x, 1] = 255
#         plt.imshow(tile_1_img[:, :, 1])
#         plt.show()
#         point_cloud.coordinates = torch.cat((point_cloud.coordinates, torch.tensor([[points[0][0], points[0][1], points[0][2]]])), dim=0)
#         point_cloud.colors = torch.zeros((point_cloud.coordinates.shape[0], 3), dtype=torch.uint8)
#         point_cloud.colors[point_cloud.coordinates.shape[0]-1] = torch.Tensor([0, 255, 0])
#         visualize(point_cloud, color_mode=None)

#         break



def bev_segmentation_trying(config):
    # load all variables from .env file into os.environ
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    # PyTorch Dataset try out
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=1,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    for batch in data_loader:
        point_cloud = batch[0]
        print_pc(point_cloud)

        if point_cloud.bev_data is None:
            print("Starting BEV projection...")
            tiles, metas = bev_projection(point_cloud, tile_size=35.0, resolution=0.05, overlap=0.0,
                                          include_class=False, direct_single_saving=False)  #  tile_size=100.0/50.0, resolution=0.2/0.1
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



# def bev_working_testing(config):
#     # LOAD POINT CLOUD
#     data_loader = get_data_loader(config.data.name, config.data.path, 
#                                     type=config.data.type, 
#                                     transform=None,  # get_basic_transform(num_points=-1), 
#                                     batch_size=1, shuffle=False, num_workers=1,
#                                     preprocessed=config.data.preprocessed, return_train_format=False)

#     for batch in data_loader:
#         point_cloud = batch[0]

#         # point_cloud = point_cloud.get_as_o3d()
#         if not isinstance(point_cloud, o3d.t.geometry.PointCloud):
#             raise TypeError(f"Point Cloud should be get as Open3D Tensor, but got '{type(point_cloud)}'")
#         print_pc(point_cloud)

#         print("Starting BEV projection...")
#         # tiles, meta = bev_projection_numba_and_open3d(point_cloud, tile_size=35.0, resolution=0.05, include_class=True)
#         tiles, metas = bev_projection(point_cloud, tile_size=35.0, resolution=0.05, overlap=0.0,
#                                           include_class=True, direct_single_saving=False)  #  tile_size=100.0/50.0, resolution=0.2/0.1
#         bev_gen = bev_gen_wrapper(tiles, metas)
        
#         # if point_cloud.bevs is None:
#         #     print("Starting BEV projection...")
#         #     tiles, meta = bev_projection_numba(point_cloud, tile_size=35.0, resolution=0.05)  #  tile_size=100.0/50.0, resolution=0.2/0.1
#         # else:
#         #     print("Loaded Bevs from file...")
#         #     tiles = point_cloud.bevs
#         #     meta = point_cloud.meta

#         bev_back_projection_testing(point_cloud, bev_gen, bev_amount=len(tiles))

#         # do not end after one testset?
#         break



def bev_preprocessed_loading_working_testing(config):
    if not config.data.preprocessed:
        raise ValueError("'Preprocessing' must be True! (config.data.preprocessed)")

    # LOAD POINT CLOUD
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=get_basic_transform(num_points=-1), 
                                    batch_size=1, shuffle=False, num_workers=1,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    bev_projection_testing(patch_gen=data_loader, atol=1e-4, dataset_name=config.data.name, save_path=f"./output/bev_projection_test_{config.data.name}.txt")

    # for batch in data_loader:
    #     point_cloud = batch[0]

    #     assert isinstance(point_cloud, PointCloudTensor)
    #     print_pc(point_cloud)

    #     print("Starting BEV projection...")
    #     if point_cloud.bev_data is None:
    #         raise ValueError("Preprocessed BEVs did not loaded.")
    #         print("Starting BEV projection...")
    #         tiles, metas = bev_projection(point_cloud, tile_size=35.0, resolution=0.05)  #  tile_size=100.0/50.0, resolution=0.2/0.1
    #     else:
    #         print("Loaded Bevs from file...")
    #         bev_gen = point_cloud.get_bev()

    #     bev_back_projection_testing(point_cloud, bev_gen, bev_amount=point_cloud.bev_amount)

    #     # do not end after one testset?
    #     break



def train_data_testing(config):
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=1,
                                    preprocessed=config.data.preprocessed, return_train_format=True)

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



def manhole_intensity_test(config):
    print("\n --- Manhole Intensity Check ---")

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=None,  # get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    # clear save path
    path = f"./output/manhole_intensity_{config.data.name}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    cur_pc = 0

    for batch in data_loader:
        point_cloud = batch[0]
        # point_cloud = point_cloud.get_as_o3d()
        print_pc(point_cloud)

        print("\n> Manhole Intensity Check <\n")
        if config.data.name == "sud":
            label_value = (1, 255) if config.data.preprocessed else 3
        else:
            label_value = (1, 255) if config.data.preprocessed else 104002
        manholes = extract_manhole(point_cloud, label_value=label_value, points_around_dist=2)

        for cur_vis, cur_manhole in enumerate(manholes):
            # Visualize Manhole

            # 2D
            plot_name = f"pc_{cur_pc}_manhole_{cur_vis}.png"
            points = cur_manhole.point[get_coordinate_attribute(cur_manhole)].numpy()
            color = cur_manhole.point[get_intensity_attribute(cur_manhole)].numpy()
            visualize_intensity_in_2d(points, color, should_plot=False, save_path=os.path.join(path, plot_name))

            # 3D
            # visualize(cur_manhole, color_mode="intensity")

            # break
    
        cur_pc += 1
        # break



def manhole_BEV_intensity_test(config):
    if config.data.name == "sud":
        # label_value = (1, 255) if config.data.preprocessed else 3
        label_value = 1 if config.data.preprocessed else 3
    else:
        # label_value = (1, 255) if config.data.preprocessed else 104002
        label_value = 1 if config.data.preprocessed else 104002
    # FIXME -> go through BEV images and if it have the label than plot the image/save image 
    #                   -> have already a method right (but maybe use normalization if not visible)
    print("\n --- Manhole BEV Check ---")

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=get_basic_transform(),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    path = f"./output/bev_image_manhole_investigation_{config.data.name}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    cur_pc = 0
    for batch in data_loader:
        cur_pc += 1
        point_cloud = batch[0]
        # point_cloud = point_cloud.get_as_o3d()
        print_pc(point_cloud)

        # get BEV images
        print("Starting BEV projection...")
        if point_cloud.bev_data is None:
            raise ValueError("Preprocessed BEVs did not loaded.")
            print("Starting BEV projection...")
            tiles, metas = bev_projection(point_cloud, tile_size=35.0, resolution=0.05)  #  tile_size=100.0/50.0, resolution=0.2/0.1
            bev_gen = bev_gen_wrapper(tiles, metas)
        else:
            print("Loaded Bevs from file...")
            bev_gen = point_cloud.get_bev()

        for idx, bev_item in enumerate(bev_gen):
            img = bev_item["pixel_values"].detach().cpu().numpy()
            labels = bev_item["labels"].detach().cpu().numpy()
            meta = bev_item["meta"] 

            # extracting manholes? -> get all manhole points + clustering

            # print(labels.shape)
            if not isinstance(label_value, (tuple, list)):
                label_value = [label_value]
            if np.any(np.isin(labels, label_value)):
                H, W = labels.shape
                # colored_img = np.full((H, W, 3), 0.0, dtype=np.float32)
                # colored_img[np.isin(labels, label_value)] = [1.0, 1.0, 0.0]

                # fix img shape -> C, H, W -> H, W, C
                img_t = np.transpose(img[:4, :, :], (1, 2, 0))

                fig, ax = plt.subplots(figsize=(15,7), ncols=3, nrows=1)

                ax[0].imshow(img_t[:, :, 1], cmap="viridis")
                
                shifted_img = img_t[:, :, 2] + abs(img_t[:, :, 3].min())
                final_img = shifted_img / shifted_img.max()
                # # 1. set 1. and 99. percentile (for removing extreme outliers)
                # p_low, p_high = np.percentile(img_t, (1, 5))
                # # 2. clipping
                # clipped_img = np.clip(img_t, p_low, p_high)
                # # 3. normalize it
                # normalized_img = (clipped_img - p_low) / (p_high - p_low)
                # ax[1].imshow(final_img, cmap="viridis")
                ax[1].imshow(final_img, cmap="grey")
                # (img_t - np.min(img_t))/(np.max(img_t) - np.min(img_t))

                cmap = mcolors.ListedColormap(['black', 'yellow', 'blue'])
                mapping = {0:0, 1:1, 255:2}

                labels_mapped = np.vectorize(mapping.get)(labels)
                labels_mapped = labels_mapped.reshape(H, W)
                ax[2].imshow(labels_mapped, cmap=cmap, vmin=0, vmax=2)
                # ax[2].imshow(colored_img)

                ax[0].axis("off")
                ax[1].axis("off")
                ax[2].axis("off")

                ax[0].set_title("Height", fontsize=14, fontweight='bold')
                # ax[0].set_title("Labeling", fontsize=14, fontweight='bold')
                ax[1].set_title("Intensity", fontsize=14, fontweight='bold')
                ax[2].set_title("Manhole Marked BEV Image", fontsize=14, fontweight='bold')

                current_name = f"pc_{cur_pc}_bevimg_{idx}.png"
                plt.savefig(os.path.join(path, current_name))

                # plt.show()

                plt.close(fig)
                # break
    
        # break



def BEV_investigation(config):
    if config.data.name == "sud":
        # label_value = (1, 255) if config.data.preprocessed else 3
        label_value = 1 if config.data.preprocessed else 3
    else:
        # label_value = (1, 255) if config.data.preprocessed else 104002
        label_value = 1 if config.data.preprocessed else 104002
    
    print("\n --- Manhole BEV Check ---")

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=get_basic_transform(),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    path = f"./output/bev_channel_investigation_{config.data.name}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    cur_pc = 0
    for batch in data_loader:
        cur_pc += 1
        point_cloud = batch[0]
        # point_cloud = point_cloud.get_as_o3d()
        print_pc(point_cloud)

        # get BEV images
        print("Starting BEV projection...")
        if point_cloud.bev_data is None:
            raise ValueError("Preprocessed BEVs did not loaded.")
            print("Starting BEV projection...")
            tiles, metas = bev_projection(point_cloud, tile_size=35.0, resolution=0.05)  #  tile_size=100.0/50.0, resolution=0.2/0.1
            bev_gen = bev_gen_wrapper(tiles, metas)
        else:
            print("Loaded Bevs from file...")
            bev_gen = point_cloud.get_bev()

        for idx, bev_item in enumerate(bev_gen):
            img = bev_item["pixel_values"].detach().cpu().numpy()
            labels = bev_item["labels"].detach().cpu().numpy()
            meta = bev_item["meta"] 

            # extracting manholes? -> get all manhole points + clustering

            # print(labels.shape)
            if not isinstance(label_value, (tuple, list)):
                label_value = [label_value]
            if np.any(np.isin(labels, label_value)):
                H, W = labels.shape
                # colored_img = np.full((H, W, 3), 0.0, dtype=np.float32)
                # colored_img[np.isin(labels, label_value)] = [1.0, 1.0, 0.0]

                # fix img shape -> C, H, W -> H, W, C
                img_t = np.transpose(img[:, :, :], (1, 2, 0))

                fig, ax = plt.subplots(figsize=(20,12), ncols=3, nrows=2)

                # Max Height
                ax[0][0].imshow(img_t[:, :, 0], cmap="viridis")

                # Delta Height
                ax[0][1].imshow(img_t[:, :, 1], cmap="viridis")
                
                # Intensity
                # shifted_img = img_t[:, :, 2] + abs(img_t[:, :, 3].min())
                # final_img = shifted_img / shifted_img.max()
                # # 1. set 1. and 99. percentile (for removing extreme outliers)
                # p_low, p_high = np.percentile(img_t, (1, 5))
                # # 2. clipping
                # clipped_img = np.clip(img_t, p_low, p_high)
                # # 3. normalize it
                # normalized_img = (clipped_img - p_low) / (p_high - p_low)
                # ax[1].imshow(final_img, cmap="viridis")
                ax[1][0].imshow(img_t[:, :, 2], cmap="grey")
                # (img_t - np.min(img_t))/(np.max(img_t) - np.min(img_t))

                # Density
                ax[1][1].imshow(img_t[:, :, 3], cmap="viridis")

                # Labels
                cmap = mcolors.ListedColormap(['black', 'yellow', 'blue'])
                mapping = {0:0, 1:1, 255:2}

                labels_mapped = np.vectorize(mapping.get)(labels)
                labels_mapped = labels_mapped.reshape(H, W)
                ax[0][2].imshow(labels_mapped, cmap=cmap, vmin=0, vmax=2)

                ax[0][0].axis("off")
                ax[0][1].axis("off")
                ax[1][0].axis("off")
                ax[1][1].axis("off")
                ax[0][2].axis("off")
                ax[1][2].axis("off")

                ax[0][0].set_title("Max Height", fontsize=14, fontweight='bold')
                ax[0][1].set_title("Delta Height", fontsize=14, fontweight='bold')
                ax[1][0].set_title("Intensity", fontsize=14, fontweight='bold')
                ax[1][1].set_title("Density", fontsize=14, fontweight='bold')
                ax[0][2].set_title("Labels", fontsize=14, fontweight='bold')

                # current_name = f"pc_{cur_pc}_bevimg_{idx}.png"
                plot_name = f"pc_{meta['pc_id']}_x_{meta['origin_x']}_y_{meta['origin_y']}.png"
                plt.savefig(os.path.join(path, plot_name))

                # plt.show()

                plt.close(fig)



def BEV_Density_investigation(config):
    if config.data.name == "sud":
        # label_value = (1, 255) if config.data.preprocessed else 3
        label_value = 1 if config.data.preprocessed else 3
    else:
        # label_value = (1, 255) if config.data.preprocessed else 104002
        label_value = 1 if config.data.preprocessed else 104002
    
    print("\n --- BEV Density Check ---")

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=get_basic_transform(),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    path = f"./output/bev_density_{config.data.name}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    total_max = []
    total_mean = []
    cur_pc = 0
    for batch in data_loader:
        cur_pc += 1
        point_cloud = batch[0]
        # point_cloud = point_cloud.get_as_o3d()
        print_pc(point_cloud)

        # get BEV images
        print("Starting BEV projection...")
        if point_cloud.bev_data is None:
            raise ValueError("Preprocessed BEVs did not loaded.")
            print("Starting BEV projection...")
            tiles, metas = bev_projection(point_cloud, tile_size=35.0, resolution=0.05)  #  tile_size=100.0/50.0, resolution=0.2/0.1
            bev_gen = bev_gen_wrapper(tiles, metas)
        else:
            print("Loaded Bevs from file...")
            bev_gen = point_cloud.get_bev()

        for idx, bev_item in enumerate(bev_gen):
            img = bev_item["pixel_values"].detach().cpu().numpy()
            labels = bev_item["labels"].detach().cpu().numpy()
            meta = bev_item["meta"] 

            # print(labels.shape)
            if not isinstance(label_value, (tuple, list)):
                label_value = [label_value]

            if np.any(np.isin(labels, label_value)):
                H, W = labels.shape

                # fix img shape -> C, H, W -> H, W, C
                img_t = np.transpose(img[:, :, :], (1, 2, 0))
                channel = img_t[:, :, 3]

                # hist, bins = np.histogram(channel, bins=256, range=(0, 255))
                total_mean.append(np.mean(channel))
                total_max.append(np.max(channel))

                fig, ax = plt.subplots(figsize=(20,12), ncols=1, nrows=1)

                ax.hist(channel.ravel(), bins=256, range=(0, 255))
                ax.set_title("Histogram of Density (Point Amount)")
                ax.set_xlabel("Pixel Value")
                ax.set_ylabel("Frequency")

                # current_name = f"pc_{cur_pc}_bevimg_{idx}.png"
                plot_name = f"pc_{meta['pc_id']}_x_{meta['origin_x']}_y_{meta['origin_y']}.png"
                plt.savefig(os.path.join(path, plot_name))

                plt.close(fig)
        
    total_max = np.array(total_max)
    total_mean = np.array(total_mean)

    print(f"Max Density:")
    print(f"    - Mean: {total_max.mean()}")
    print(f"    - Max: {total_max.max()}")
    print(f"    - Min: {total_max.min()}")
    print(f"    - Std: {total_max.std()}")

    print(f"\nMean Density:")
    print(f"    - Mean: {total_mean.mean()}")
    print(f"    - Max: {total_mean.max()}")
    print(f"    - Min: {total_mean.min()}")
    print(f"    - Std: {total_mean.std()}")



def bev_dataset_stat_investigation(config):
    def compute_dataset_stats(dataset):
        H = W = None
        means, stds = [], []
        for i in range(len(dataset)):
            x = dataset[i]["pixel_values"]  # (C, H, W)
            means.append(x.mean(dim=[1, 2]))
            stds.append(x.std(dim=[1, 2]))
            if H is None:
                H = x.shape[1]
            if W is None:
                W = x.shape[2]
        mean = torch.stack(means).mean(dim=0).tolist()
        std  = torch.stack(stds).mean(dim=0).tolist()
        
        return mean, std, H, W
    
    if config.data.type != "train":
        print("[HINT] Changed Type of train because data stats are needed most likely from train data.")

    train_dataset = get_data_loader(config.data.name, 
                                   config.data.path, 
                                   type="train", 
                                   transform=get_basic_transform(),
                                   batch_size=1, 
                                   shuffle=False, 
                                   num_workers=1,
                                   preprocessed=True, 
                                   return_train_format=True,
                                   return_dataset=True)
    all_train_paths = train_dataset.point_cloud_paths
    train_dataset = BEVDataset(path=all_train_paths, file_paths=[], has_labels=True, image_training=True, preprocessor=None)

    mean, std, H, W = compute_dataset_stats(train_dataset)

    result = f"{config.data.name} Stats:\n    - Mean: {mean}\n    - STD: {std}\n    - Height: {H}\n    - Width: {W}"

    save_path = f"./output/{config.data.name}_bev_data_stats.txt"
    with open(save_path, "w") as file_:
        file_.write(result)

    print(result)
    print(f"\n[INFO] Saved to '{save_path}'")



def manhole_3d_and_2d_intensity_test(config):
    if config.data.name == "sud":
        # label_value = (1, 255) if config.data.preprocessed else 3
        label_value = 1 if config.data.preprocessed else 3
    else:
        # label_value = (1, 255) if config.data.preprocessed else 104002
        label_value = 1 if config.data.preprocessed else 104002
    
    print("\n --- Intensity 3D and 2D Check ---")

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=get_basic_transform(),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    path = f"./output/intensity_3d_2d_investigation_{config.data.name}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    cur_pc = 0
    for batch in data_loader:
        cur_pc += 1
        point_cloud = batch[0]
        # point_cloud = point_cloud.get_as_o3d()
        print_pc(point_cloud)

        # get BEV images
        print("Starting BEV projection...")
        if point_cloud.bev_data is None:
            raise ValueError("Preprocessed BEVs did not loaded.")
            print("Starting BEV projection...")
            tiles, metas = bev_projection(point_cloud, tile_size=35.0, resolution=0.05)  #  tile_size=100.0/50.0, resolution=0.2/0.1
            bev_gen = bev_gen_wrapper(tiles, metas)
        else:
            print("Loaded Bevs from file...")
            bev_gen = point_cloud.get_bev()

        for idx, bev_item in enumerate(bev_gen):
            img = bev_item["pixel_values"].detach().cpu().numpy()
            labels = bev_item["labels"].detach().cpu().numpy()
            meta = bev_item["meta"] 

            # extracting manholes? -> get all manhole points + clustering

            # print(labels.shape)
            if not isinstance(label_value, (tuple, list)):
                label_value = [label_value]
            if np.any(np.isin(labels, label_value)):
                H, W = labels.shape
                
                plot_name = f"pc_{meta['pc_id']}_x_{meta['origin_x']}_y_{meta['origin_y']}.png"

                # fix img shape -> C, H, W -> H, W, C
                img_t = np.transpose(img[:4, :, :], (1, 2, 0))

                fig, ax = plt.subplots(figsize=(15,7), ncols=2, nrows=1)

                pc_numpy = point_cloud.to_numpy(as_copy=True)
                points = pc_numpy.coordinates
                color = pc_numpy.intensities
                color = (color - np.min(color)) / (np.max(color) - np.min(color))
                color = np.repeat(color[:, np.newaxis], 3, axis=1).squeeze()
                x = points[:, 0]
                y = points[:, 1]
                ax[0].scatter(x, y, s=5, c=color[:, 0], alpha=1.0, cmap="viridis", edgecolors="none")
                # alpha=0.4, marker="o", linewidths=0
                # for size, alpha in [(40, 0.03), (20, 0.08), (8, 0.2)]:
                #     ax[0].scatter(
                #         x,
                #         y,
                #         s=size,
                #         c=color[:, 0],
                #         alpha=alpha,
                #         edgecolors="none",
                #         cmap="viridis"
                #     )

                # shifted_img = img_t[:, :, 3] + abs(img_t[:, :, 3].min())
                # final_img = shifted_img / shifted_img.max()
                # ax[1].imshow(final_img, cmap="gray")
                h, w = img_t.shape[:2]
                x, y = np.meshgrid(
                    np.arange(w),
                    np.arange(h)
                )
                color = img_t[:, :, 2]
                color = (color - np.min(color)) / (np.max(color) - np.min(color))
                ax[1].scatter(x.ravel(), y.ravel(), s=5, c=color.ravel(), alpha=1.0, cmap="viridis", edgecolors="none")
                # for size, alpha in [(40, 0.03), (20, 0.08), (8, 0.2)]:
                #     ax[1].scatter(
                #         x.ravel(),
                #         y.ravel(),
                #         s=size,
                #         c=color.ravel(),
                #         alpha=alpha,
                #         edgecolors="none",
                #         cmap="viridis"
                #     )

                ax[0].axis("off")
                ax[1].axis("off")

                ax[0].set_title("3D Intensity", fontsize=14, fontweight='bold')
                ax[1].set_title("2D Intensity", fontsize=14, fontweight='bold')

                ax[0].set_aspect("equal")
                ax[1].set_aspect("equal")

                ax[0].grid(alpha=0.3)
                ax[1].grid(alpha=0.3)

                plt.savefig(os.path.join(path, plot_name))

                plt.close(fig)



def manhole_density_test(config):
    print("\n --- Manhole Density Check ---")

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=None,  # get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    total_result = dict()

    print("\n> Manhole Density Check <\n")

    for batch in tqdm(data_loader, total=len(data_loader), desc="Density Check"):
        point_cloud = batch[0]
        # point_cloud = point_cloud.get_as_o3d()
        # print_pc(point_cloud)

        if config.data.name == "sud":
            # label_value = (1, 255) if config.data.preprocessed else 3
            label_value = 1 if config.data.preprocessed else 3
        else:
            # label_value = (1, 255) if config.data.preprocessed else 104002
            label_value = 1 if config.data.preprocessed else 104002
        manholes = extract_manhole(point_cloud, label_value=label_value, points_around_dist=0)

        for cur_manhole in manholes:

            points = cur_manhole.point[get_coordinate_attribute(cur_manhole)].numpy()
            result = analyze_point_distribution(points, num_angle_bins=36)

            for key, item in result.items():
                if key in total_result.keys():
                    total_result[key] += [item]
                else:
                    total_result[key] = [item]

            # direction_strength.append(result["direction_strength"])
            # print(f"DEBUGGING, added element: {result["direction_strength"]}")

    # print(f"DEBUGGING, elements: {len(direction_strength)}")
    # direction_strength = np.array(direction_strength)
    # print(f"DEBUGGING, numpy elements: {direction_strength.shape}")

    for key, values in total_result.items():
        values = np.array(values)
        print(f"\n{key}")
        print(f"    ▷ Mean: {values.mean():.4f}")
        print(f"    ▷ Max: {values.max():.4f}")
        print(f"    ▷ Min: {values.min():.4f}")
        print(f"    ▷ Std: {values.std():.4f}")



def preprocessing_speed_test(config):
    pass



def circular_manhole_classification_test(config):
    print("\n --- Center Shape Check ---")

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=None,  # get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    # clear save path
    path = f"./output/center_shape_check_{config.data.name}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    cur_pc = 0
    total_result = {"Circle": 0, "No Circle": 0}
    for batch in tqdm(data_loader, total=len(data_loader), desc="Manhole Circular Test"):
        point_cloud = batch[0]
        # point_cloud = point_cloud.get_as_o3d()
        # print_pc(point_cloud)

        # get maholes
        if config.data.name == "sud":
            label_value = (1, 255) if config.data.preprocessed else 3
        else:
            label_value = (1, 255) if config.data.preprocessed else 104002
        manholes = extract_manhole(point_cloud, label_value=label_value, points_around_dist=0)

        for cur_vis, cur_manhole in enumerate(manholes):
            plot_name = f"pc_{cur_pc}_manhole_{cur_vis}.png"
            is_circle, _ = circle_shape_check(cur_manhole, save_path=os.path.join(path, plot_name), should_plot=False)
            if is_circle:
                total_result["Circle"] += 1
            else:
                total_result["No Circle"] += 1

        cur_pc += 1
        
    print(total_result)



def center_robustnest_test(config):
    print("\n --- Stresstest Center Estimation (with labels) ---")

    if config.data.name == "sud":
        # label_value = (1, 255) if config.data.preprocessed else 3
        label_value = 1 if config.data.preprocessed else 3
    else:
        # label_value = (1, 255) if config.data.preprocessed else 104002
        label_value = 1 if config.data.preprocessed else 104002

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=None,  # get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    # clear save path
    path = f"./output/center_estimation_stresstest_{config.data.name}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    cur_pc = 0
    least_square_errors = []
    ransac_errors = []
    for batch in tqdm(data_loader, total=len(data_loader), desc="Center Estimation Stresstest"):
        point_cloud = batch[0]
        # point_cloud = point_cloud.get_as_o3d()
        # print_pc(point_cloud)

        # get cluster and manupilate it
        _, _, _, original_cluster_pcs, _, _, _ = center_estimation_3d_pipeline_debugging(point_cloud, method="least_square", extended_return=True, should_visualize=False, label_value=(1, 255) if config.data.preprocessed else 104002)
        
        if original_cluster_pcs is None:
            cur_pc += 1
            continue

        manipulated_clusters = []
        for cur_cluster in original_cluster_pcs:
            manipulated_clusters.append(
                add_random_dense_manipulation_point_cloud(cur_cluster, n=np.random.randint(20, max(20, 1000)))
            )

        print("\n> Least Square Circle Fit Check <\n")
        center_coordinates_square, radius_squares, points_square, cluster_point_clouds, _, error_s, _ = center_estimation_3d_pipeline_debugging(None, method="least_square", extended_return=True, should_visualize=False, clusters=manipulated_clusters, label_value=label_value)

        print("\n> RANSAC Fit Check <\n")
        center_coordinates_ransac, radius_ransac, points_ransac, cluster_point_clouds, _, error_r, _ = center_estimation_3d_pipeline_debugging(None, method="ransac", extended_return=True, should_visualize=False, clusters=manipulated_clusters, label_value=label_value)

        print("\n> RANSAC Downsampled Fit Check <\n")
        center_coordinates_ransac_downsampled, radius_ransac_downsampled, points_ransac_downsampled, _, _, error_r, _ = center_estimation_3d_pipeline_debugging(None, method="ransac", extended_return=True, should_visualize=False, clusters=manipulated_clusters, label_value=label_value, apply_downsampling=True)

        # compare similarity
        for cur_vis in range(len(points_ransac)):
            plot_name = f"pc_{cur_pc}_manhole_{cur_vis}.png"

            # save them -> don't show
            visualize_circle_fit(points=points_ransac[cur_vis], 
                                 center_pred=center_coordinates_square[cur_vis], 
                                 radius=radius_squares[cur_vis], 
                                 error=error_s[cur_vis], 
                                 name="Least-Squares", 
                                 additional_center_pred=center_coordinates_ransac[cur_vis], 
                                 additional_radius_pred=radius_ransac[cur_vis], 
                                 additional_name="RANSAC",
                                 should_plot=False,
                                 save_path=os.path.join(path, plot_name))
            
            plot_name = f"pc_{cur_pc}_manhole_{cur_vis}_downsampled.png"
            visualize_circle_fit(points=points_ransac_downsampled[cur_vis], 
                                 center_pred=center_coordinates_square[cur_vis], 
                                 radius=radius_squares[cur_vis], 
                                 error=error_s[cur_vis], 
                                 name="Least-Squares", 
                                 additional_center_pred=center_coordinates_ransac_downsampled[cur_vis], 
                                 additional_radius_pred=radius_ransac_downsampled[cur_vis], 
                                 additional_name="RANSAC (Downsampled)",
                                 should_plot=False,
                                 save_path=os.path.join(path, plot_name))

        cur_pc += 1
        least_square_errors.append(np.array(error_s).mean())
        ransac_errors.append(np.array(error_r).mean())


    # sum error
    least_square_errors = np.array(least_square_errors)
    print(f"Least Square Error")
    print(f"  - mean: {least_square_errors.mean():.4f}")
    print(f"  - min: {least_square_errors.min():.4f}")
    print(f"  - max: {least_square_errors.max():.4f}")
    print(f"  - std: {least_square_errors.std():.4f}")

    ransac_errors = np.array(ransac_errors)
    print(f"RANSAC Error")
    print(f"  - mean: {ransac_errors.mean():.4f}")
    print(f"  - min: {ransac_errors.min():.4f}")
    print(f"  - max: {ransac_errors.max():.4f}")
    print(f"  - std: {ransac_errors.std():.4f}")

    
    # stress test for least squares if the data is not euqually distributed (in live system relevant)
    # But for data annotation only relevant if the data is not always equal distributed 



def ransac_inlier_test(config):
    """
    RANSAC classifies all points as inliers or outliers 
    depending on the distance to the circle 
    which is build with 3 random points over 
    multiple iterations and a optimization 
    alogrithm like Least Squares. 
    """
    if config.data.name == "sud":
        # label_value = (1, 255) if config.data.preprocessed else 3
        label_value = 1 if config.data.preprocessed else 3
    else:
        # label_value = (1, 255) if config.data.preprocessed else 104002
        label_value = 1 if config.data.preprocessed else 104002

    # better understanding of why it chooses that
    print("\n --- RANSAC Center Estimation Investigation ---")

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=None,  # get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    # clear save path
    path = f"./output/ransac_inlier_test_{config.data.name}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    cur_pc = 0

    for batch in tqdm(data_loader, total=len(data_loader), desc="RANSAC Inlier Test"):
        point_cloud = batch[0]
        # print_pc(point_cloud)

        # get cluster
        print("\n> RANSAC Fit Check <\n")
        center_coordinates_ransac, radius_ransac, points, cluster_point_clouds, inliers, error, _ = center_estimation_3d_pipeline_debugging(point_cloud, method="ransac", extended_return=True, clusters=None, should_visualize=False, label_value=label_value)
        
        if points is None:
            cur_pc += 1
            continue

        # # visualize error
        for cur_vis in range(len(points)):

            plot_name = f"pc_{cur_pc}_manhole_{cur_vis}.png"
            cur_path = os.path.join(path, plot_name)
            
            visualize_ransac_inliers(points_2d=points[cur_vis][:, :2], 
                                     center_pred=center_coordinates_ransac[cur_vis, :2], 
                                     radius=radius_ransac[cur_vis], 
                                     inliers=inliers[cur_vis],
                                     should_plot=False, save_path=cur_path)

        cur_pc += 1

    print("Successfull finished!")



def ransac_downsampling_test(config):
    # baristisches downsampling
    """
    RANSAC samples from more dense regions and therefore a 
    centroid based downsampling (barycentric) could help RANSAC.
    """
    if config.data.name == "sud":
        # label_value = (1, 255) if config.data.preprocessed else 3
        label_value = 1 if config.data.preprocessed else 3
    else:
        # label_value = (1, 255) if config.data.preprocessed else 104002
        label_value = 1 if config.data.preprocessed else 104002

    # better understanding of why it chooses that
    print("\n --- RANSAC Center Estimation Downsampling Investigation ---")

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=None,  # get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    # clear save path
    path = f"./output/ransac_downsampling_test_{config.data.name}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    cur_pc = 0

    for batch in tqdm(data_loader, total=len(data_loader), desc="RANSAC Inlier Test"):
        point_cloud = batch[0]

        # get cluster and manupilate it
        _, _, _, original_cluster_pcs, _, _, _ = center_estimation_3d_pipeline_debugging(point_cloud, method="least_square", extended_return=True, should_visualize=False, label_value=label_value)
        
        if original_cluster_pcs is None:
            cur_pc += 1
            continue

        # normal RANSAC
        print("\n> RANSAC Fit Check <\n")
        center_coordinates_ransac, radius_ransac, _, cluster_point_clouds, inliers, error, _ = center_estimation_3d_pipeline_debugging(None, method="ransac", extended_return=True, clusters=original_cluster_pcs, 
                                                                                                                                                       should_visualize=False, 
                                                                                                                                                       label_value=label_value,
                                                                                                                                                       apply_downsampling=False)
        
        # downsampled RANSAC
        print("\n> Downsampled RANSAC Fit Check <\n")
        center_coordinates_ransac_downsampled, radius_ransac_downsampled, points, cluster_point_clouds, inliers, _, input_points = center_estimation_3d_pipeline_debugging(None, method="ransac", extended_return=True, clusters=original_cluster_pcs, 
                                                                                                                                                       should_visualize=False, 
                                                                                                                                                       label_value=label_value,
                                                                                                                                                       apply_downsampling=True)

        # # visualize error
        for cur_vis in range(len(points)):

            plot_name = f"pc_{cur_pc}_manhole_{cur_vis}.png"
            cur_path = os.path.join(path, plot_name)
            
            visualize_circle_fit(points=points[cur_vis], 
                                 center_pred=center_coordinates_ransac[cur_vis], 
                                 radius=radius_ransac[cur_vis], 
                                 error=error[cur_vis], 
                                 name="RANSAC", 
                                 additional_center_pred=center_coordinates_ransac_downsampled[cur_vis], 
                                 additional_radius_pred=radius_ransac_downsampled[cur_vis], 
                                 additional_name="Downsampled RANSAC",
                                 additional_points=input_points[cur_vis], 
                                 additional_points_label="Downsampled Points",
                                 hide_mean=True,
                                 save_path=cur_path, should_plot=False)

        cur_pc += 1

    print("Successfull finished!")



def point_amount_check(config):
    print("\n --- Point Amount Check ---")

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    # clear save path
    path = f"./output/center_estimation_stresstest_{config.data.name}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    points = []
    for batch in tqdm(data_loader, total=len(data_loader), desc="Point Amount Check"):
        point_cloud = batch[0]
        points.append(len(point_cloud.coordinates))

    points = np.array(points)

    print(f"Point Amount in Point Clouds from {config.data.name}")
    print(f"- mean: {points.mean():.2f}")
    print(f"- min: {points.min():.2f}")
    print(f"- max: {points.max():.2f}")
    print(f"- std: {points.std():.2f}")



def center_estimation_3d_pipeline_debugging(point_cloud, method, extended_return=False, should_visualize=True, clusters=None, label_value=1, apply_downsampling=False):
    """
    Helper Function
    """
    print("Compute centers...")
    if point_cloud is not None:
        center_points = use_label_candidates_and_extract_center_point(points=point_cloud, 
                                                                      use_2d_version=False, 
                                                                      label_value=label_value,  # 104002, 
                                                                      method=method, 
                                                                      use_projection=True, 
                                                                      cluster_if_needed=True,
                                                                      apply_downsampling=apply_downsampling)
    else:
        center_points = use_points_and_extract_center_point(clusters=clusters, 
                                                            method=method, 
                                                            use_projection=True,
                                                            apply_downsampling=apply_downsampling)
        
    if len(center_points) <= 0:
        if extended_return:
            return center_points, None, None, None, None, None, None
        else:
            return center_points
    
    # visualize -> all points in black, manholes in yellow and center point in red
    if should_visualize:
        if point_cloud is not None:
            print("Compute loss and error + preprare visualization...")
            pcd_vis = point_cloud.clone()

            class_key = get_class_attribute(point_cloud)
            classes = point_cloud.point[class_key].cpu().numpy()

            color = np.full([classes.shape[0], 3], 0.0, dtype=np.float32)
            # color[classes != label_value] = [0.0, 0.0, 0.0]
            if not isinstance(label_value, (list, tuple)):
                label_value = [label_value]
            for cur_label_value in label_value:
                color[(classes == cur_label_value).flatten()] = [1.0, 0.95, 0.0]
            pcd_vis.point["colors"] = o3d.core.Tensor(color, dtype=o3d.core.Dtype.Float32)

    # visualize found points (and show losses!)
    center_coordinates = np.full((len(center_points), 3), 0.0, dtype=np.float32)
    total_error = np.full((len(center_points),), -99.0, dtype=np.float32)
    error_exists = True
    total_loss = np.full((len(center_points),), -99.0, dtype=np.float32)
    loss_exists = True

    if extended_return:
        all_radius = np.full((len(center_points), 1), 0.0, dtype=np.float32)
        cluster_points = []
        cluster_point_clouds = []
        all_inliers = []
        all_input_points = []

    for idx, item in enumerate(center_points):
        center, radius, cluster, inliers, error, loss, input_points = item
        center_coordinates[idx] = center

        if error_exists is True and error is not None:
            total_error[idx] = error
        else:
            error_exists = False

        if loss_exists is True and loss is not None:
            total_loss[idx] = loss
        else:
            loss_exists = False

        if extended_return:
            all_radius[idx] = radius
            cluster_points.append(cluster.point[get_coordinate_attribute(cluster)].numpy())
            cluster_point_clouds.append(cluster)
            all_inliers.append(inliers)
            all_input_points.append(input_points)

    if len(center_points) == 0:
        raise ValueError("No center points found!")
    if np.any(total_error == -99) and error_exists:
        raise ValueError("Found a not set error value!")
    if np.any(total_loss == -99) and loss_exists:
        raise ValueError("Found a not set loss value!")

    # Residuals → “per-point mistake”
    # Error → “average mistake”
    # Loss → “how much we care about mistakes (with punishment for big ones)”

    if error_exists:
        print(f"\nError (mean absolute geometric distance to the circle):\n    - Mean: {total_error.mean()}\n    - STD: {total_error.std()}\n    - Min: {total_error.min()}\n    - Max: {total_error.max()}")
    else:
        print("\nNo Error available.")

    # also good: loss='soft_l1'
    if loss_exists:
        # penalizes outliers
        # smooth function, good for back-propagation/gradients
        print(f"\nLoss (sum of square geometric distance to the circle):\n    - Mean: {total_loss.mean()}\n    - STD: {total_loss.std()}\n    - Min: {total_loss.min()}\n    - Max: {total_loss.max()}\n")
    else:
        print("\nNo Loss available.")

    if should_visualize:
        if point_cloud is not None:
            print("Visualize (yellow are manhole and red the predicted center points)")
            new_colors = np.full((center_coordinates.shape[0], 3), [1.0, 0.0, 0.1], dtype=np.float32)
            new_colors = o3d.core.Tensor(new_colors, dtype=o3d.core.Dtype.Float32)

            coordinate_key = get_coordinate_attribute(point_cloud)
            new_point = o3d.core.Tensor(center_coordinates, dtype=pcd_vis.point[coordinate_key].dtype)
            pcd_vis.point[coordinate_key] = o3d.core.concatenate([pcd_vis.point[coordinate_key], new_point], axis=0)
            pcd_vis.point["colors"] = o3d.core.concatenate([pcd_vis.point["colors"], new_colors], axis=0)

            visualize(pcd_vis, color_mode=None)

    if extended_return:
        return center_coordinates, all_radius, cluster_points, cluster_point_clouds, all_inliers, total_error, all_input_points
    else:
        return center_coordinates



def center_prediction_use_labels_as_candidates_test(config):
    print("\n --- Center Estimation (with labels) ---")

    if config.data.name == "sud":
        # label_value = (1, 255) if config.data.preprocessed else 3
        label_value = 1 if config.data.preprocessed else 3
    else:
        # label_value = (1, 255) if config.data.preprocessed else 104002
        label_value = 1 if config.data.preprocessed else 104002

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=None,  # get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    # clear save path
    path = f"./output/center_estimation_{config.data.name}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    cur_pc = 0

    for batch in data_loader:
        point_cloud = batch[0]
        # point_cloud = point_cloud.get_as_o3d()
        print_pc(point_cloud)

        # get cluster
        _, _, _, original_cluster_pcs, _, _, _ = center_estimation_3d_pipeline_debugging(point_cloud, method="least_square", extended_return=True, should_visualize=False, label_value=label_value)

        if original_cluster_pcs is None:
            cur_pc += 1
            continue

        print("\n> Least Square Circle Fit Check <\n")
        center_coordinates_square, radius_squares, points_square, cluster_point_clouds, _, error, _ = center_estimation_3d_pipeline_debugging(None, method="least_square", extended_return=True, should_visualize=False, clusters=original_cluster_pcs, label_value=label_value)

        # # visualize error
        # for cur_vis in range(len(points_square)):
        #     visualize_circle_fit(points=points_square[cur_vis], 
        #                          center_pred=center_coordinates_square[cur_vis], 
        #                          radius=radius_squares[cur_vis], 
        #                          error=error[cur_vis])

        print("\n> RANSAC Fit Check <\n")
        center_coordinates_ransac, radius_ransac, points, cluster_point_clouds, _, error, _ = center_estimation_3d_pipeline_debugging(None, method="ransac", extended_return=True, clusters=original_cluster_pcs, should_visualize=False, label_value=label_value)
        # print(f"Center Coordinates, RANSAC: {center_coordinates_ransac}")
        # # visualize error
        # for cur_vis in range(len(points)):
        #     visualize_circle_fit(points=points[cur_vis], 
        #                          center_pred=center_coordinates_ransac[cur_vis], 
        #                          radius=radius_ransac[cur_vis], 
        #                          error=error[cur_vis])

        # DEBUGGING
        # print(points[0].shape)
        # print(points_square[0].shape)
        # print(points == points_square)

        print("\n> RANSAC Downsampled Fit Check <\n")
        center_coordinates_ransac_downsampled, radius_ransac_downsampled, points_ransac_downsampled, _, _, error_r, _ = center_estimation_3d_pipeline_debugging(None, method="ransac", extended_return=True, should_visualize=False, clusters=original_cluster_pcs, label_value=label_value, apply_downsampling=True)


        # compare similarity?
        print(f"len(points) = {len(points)}\nlen(points_square) = {len(points_square)}")
        assert len(points) == len(points_square)
        clusters_are_equal = True
        for cur_cluster_idx in range(len(points)):
            if points[cur_cluster_idx].shape != points_square[cur_cluster_idx].shape:
                clusters_are_equal = False
                raise ValueError("Shapes does not match")
                break
        print("Cluster Point arrangment is equal.")

        # approach_2_to_1_mapping = {}
        # for cur_idx_approach_2 in range(len(points)):
        #     mapping_found = False
        #     for cur_idx_approach_1 in range(len(points_square)):
        #         if points[cur_idx_approach_1].shape == points_square[cur_idx_approach_2].shape and \
        #             points[cur_idx_approach_1] == points_square[cur_idx_approach_2]:
        #             approach_2_to_1_mapping[cur_idx_approach_1] = cur_idx_approach_2
        #             mapping_found = True
        #             break

        #     if not mapping_found:
        #         raise RuntimeError("Mapping could not be completed.")
            
        # raise RuntimeError("Debugging Stop.")


        for cur_vis in range(len(points)):
            plot_name = f"pc_{cur_pc}_manhole_{cur_vis}.png"
            visualize_circle_fit(points=points[cur_vis], 
                                 center_pred=center_coordinates_square[cur_vis], 
                                 radius=radius_squares[cur_vis], 
                                 error=error[cur_vis], 
                                 name="Least-Squares", 
                                 additional_center_pred=center_coordinates_ransac[cur_vis], 
                                 additional_radius_pred=radius_ransac[cur_vis], 
                                 additional_name="RANSAC",
                                 save_path=os.path.join(path, plot_name), 
                                 should_plot=False)
            

            plot_name = f"pc_{cur_pc}_manhole_{cur_vis}_downsampled.png"
            visualize_circle_fit(points=points_ransac_downsampled[cur_vis], 
                                 center_pred=center_coordinates_square[cur_vis], 
                                 radius=radius_squares[cur_vis], 
                                 error=error[cur_vis], 
                                 name="Least-Squares", 
                                 additional_center_pred=center_coordinates_ransac_downsampled[cur_vis], 
                                 additional_radius_pred=radius_ransac_downsampled[cur_vis], 
                                 additional_name="RANSAC (Downsampled)",
                                 should_plot=False,
                                 save_path=os.path.join(path, plot_name))

        cur_pc += 1

    print("Successfull finished!")



def squares_circle_shape_test(config):
    print("\n --- Center Estimation Squares Shape Test ---")

    if config.data.name == "sud":
        # label_value = (1, 255) if config.data.preprocessed else 3
        label_value = 1 if config.data.preprocessed else 3
    else:
        # label_value = (1, 255) if config.data.preprocessed else 104002
        label_value = 1 if config.data.preprocessed else 104002

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=None,  # get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    # clear save path
    path = f"./output/squares_circle_shape_{config.data.name}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    cur_pc = 0

    for batch in data_loader:
        point_cloud = batch[0]
        # point_cloud = point_cloud.get_as_o3d()
        print_pc(point_cloud)

        # get cluster
        _, _, _, original_cluster_pcs, _, _, _ = center_estimation_3d_pipeline_debugging(point_cloud, method="least_square", extended_return=True, should_visualize=False, label_value=label_value)

        if original_cluster_pcs is None:
            cur_pc += 1
            continue

        print("\n> Least Square Circle Fit Check <\n")
        center_coordinates_square, radius_squares, points_square, cluster_point_clouds, _, error, _ = center_estimation_3d_pipeline_debugging(None, method="least_square", extended_return=True, should_visualize=False, clusters=original_cluster_pcs, label_value=label_value)

        # visualize error
        for cur_vis in range(len(points_square)):
            plot_name = f"pc_{cur_pc}_manhole_{cur_vis}_squares.png"

            is_circle_, _ = circle_shape_check(points_square[cur_vis], save_path=None, should_plot=False, threshold=0.6)

            points_2d = points_square[cur_vis][:, :2]

            title = f"Circle Shape Check (Is Circle = {is_circle_})"
            sub_title = f"Least-Squares (LS Error: Mean={error[cur_vis].mean():.2f}, Min={error[cur_vis].min():.2f}, Max={error[cur_vis].max():.2f})"
            visualize_circle_shape_and_center_prediction(points_2d=points_2d, 
                                                         center_pred=center_coordinates_square[cur_vis], 
                                                         radius=radius_squares[cur_vis], 
                                                         title=title, sub_title=sub_title,
                                                         should_plot=False, 
                                                         save_path=os.path.join(path, plot_name))

        cur_pc += 1

    print("Successfull finished!")



def center_prediction_use_labels_as_candidates_without_instances_test(config):
    pass



def center_prediction_without_labels_test(config):
    pass



def center_2D_prediction_use_labels_as_candidates_test(config):
    pass



def center_2D_prediction_use_labels_as_candidates_without_instances_test(config):
    pass



def center_2D_prediction_without_labels_test(config):
    pass



def clustering_tryout(config):
    pass



def classic_2D_pipeline_test(config):
    print("\n --- Center Estimation with 2D Classic Pipeline ---")

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    # clear save path
    path = f"./output/center_estimation_2d_geomtry_{config.data.name}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    cur_pc = 0

    print("Geometry processing...")
    for batch in tqdm(data_loader, total=len(data_loader), desc="Center Estimation Geometry 2D"):
        point_cloud = batch[0]
        classic_manhole_prediction_pipeline(point_cloud, type=config.data.type, plot_path=path)



def make_split(config, test_size=0.2, val_size=0.1):
    print("\n --- Make Split ---")

    if config.data.name == "sud":
        # label_value = (1, 255) if config.data.preprocessed else 3
        label_value = 1 if config.data.preprocessed else 3
    else:
        # label_value = (1, 255) if config.data.preprocessed else 104002
        label_value = 1 if config.data.preprocessed else 104002

    if not config.data.preprocessed:
        print("[Hint] Changed 'Preprocessed' to True.")

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type="all", 
                                    transform=None,  # get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=True, return_train_format=False)

    dataset = data_loader.dataset
    paths = dataset.point_cloud_paths

    # countering which files have manholes
    cur_pc = 0
    pc_with_manholes = []
    pc_without_manholes = []
    for idx, batch in enumerate(data_loader):
        path = paths[idx]
        point_cloud = batch[0]

        # get cluster
        _, _, _, original_cluster_pcs, _, _, _ = center_estimation_3d_pipeline_debugging(point_cloud, method="least_square", extended_return=True, should_visualize=False, label_value=label_value)

        cur_pc += 1

        # extract pc id
        _, file_name = os.path.split(path)
        id_ = file_name.replace("preprocessed_patch_", "").split("_")[0]

        if id_ in pc_with_manholes or id_ in pc_without_manholes:
            continue

        if original_cluster_pcs is None:
            pc_without_manholes.append(id_)
        else:
            pc_with_manholes.append(id_)

    # making the split (but first only with pc with manholes to ensure enogh manhole sin every set)
    pc_with_manholes = np.array(pc_with_manholes)
    np.random.shuffle(pc_with_manholes)
    
    num_total = len(pc_with_manholes)
    num_test = int(num_total * test_size)
    num_val = int(num_total * val_size)

    test_set = pc_with_manholes[:num_test]
    val_set = pc_with_manholes[num_test : num_test + num_val]
    train_set = pc_with_manholes[num_test + num_val:]

    # now add point clouds which do not have manholes
    pc_without_manholes = np.array(pc_without_manholes)
    np.random.shuffle(pc_without_manholes)

    num_total = len(pc_without_manholes)
    num_test = int(num_total * test_size)
    num_val = int(num_total * val_size)

    # combine results
    test_set = np.concatenate((test_set, pc_without_manholes[:num_test]))
    val_set = np.concatenate((val_set, pc_without_manholes[num_test : num_test + num_val])) 
    train_set = np.concatenate((train_set, pc_without_manholes[num_test + num_val:]))

    # summerize result
    complete = len(pc_with_manholes) + len(pc_without_manholes)
    # split_text = f"Used {complete}/{cur_pc} ({(complete/cur_pc)*100:.2f}%) Point Clouds because of missing manhole labels."
    split_text = f"Found {len(pc_with_manholes)}/{complete} ({(len(pc_with_manholes)/complete)*100:.2f}%) Point Clouds with manhole labels (but uses all point clouds for split)."
    split_text += f"\n\nSplit complete:\n    Train={len(train_set)} ({(len(train_set)/complete)*100:.2f}%)\n    Val={len(val_set)} ({(len(val_set)/complete)*100:.2f}%)\n    Test={len(test_set)} ({(len(test_set)/complete)*100:.2f}%)"
 
    split_text += "\n\n--- File-Paths ---"
    split_text += f"\n\nTrain Samples:\n{train_set.tolist()}"
    split_text += f"\n\nVal Samples:\n{val_set.tolist()}"
    split_text += f"\n\nTest Samples:\n{test_set.tolist()}"

    with open(f"./output/{config.data.name}_data_split.txt", "w") as file_:
        file_.write(split_text)

    print(split_text)
    
    return train_set, val_set, test_set


def ground_truth_2d_map_test(config):
    print("\n --- Center Estimation (with labels) ---")

    if config.data.name == "sud":
        # label_value = (1, 255) if config.data.preprocessed else 3
        label_value = 1 if config.data.preprocessed else 3
    else:
        # label_value = (1, 255) if config.data.preprocessed else 104002
        label_value = 1 if config.data.preprocessed else 104002

    print("Loading Data...")
    train_dataset = get_data_loader(config.data.name, 
                                   config.data.path, 
                                   type="train", 
                                   transform=get_basic_transform(),
                                   batch_size=1, 
                                   shuffle=False, 
                                   num_workers=1,
                                   preprocessed=True, 
                                   return_train_format=True,
                                   return_dataset=True)
    all_train_paths = train_dataset.point_cloud_paths
    train_dataset = BEVDataset(path=all_train_paths, file_paths=[], has_labels=True, image_training=True, preprocessor=None)

    all_file_paths = train_dataset.file_paths

    path = f"./output/mcr_gt_2d_map_test_{config.data.name}"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    for idx, batch in enumerate(train_dataset):
        
        x = batch["pixel_values"].detach().cpu().permute(1, 2, 0).numpy()
        # x = np.permute_dims(x, (1, 2, 0))
        y = batch["labels"].detach().cpu().numpy()
        # y[y == 255] = 0
        y = np.ma.masked_where(y == 255, y)
        # find the created gt map (if there is one)
        file_name = os.path.split(all_file_paths[idx])[-1]
        pc_id, x_start, y_start = train_dataset.extract_grid_identifier(file_name)

        gt_path = f"./2d_gt_patches/{config.data.name}_{pc_id}_{x_start}_{y_start}.npy"
        if not os.path.exists(gt_path):
            continue

        gt_2d_map = np.load(gt_path)

        fig, axes = plt.subplots(1, 5, figsize=(8*5, 7))

        axes[0].imshow(x[:, :, 1], cmap="viridis")
        axes[0].axis("off")
        axes[0].set_title("Intensity")

        axes[1].imshow(y, cmap="viridis")
        axes[1].axis("off")
        axes[1].set_title("Labels")

        axes[2].imshow(gt_2d_map[:, :, 0], cmap="viridis")
        axes[2].axis("off")
        axes[2].set_title("GT Binary Map")

        axes[3].imshow(gt_2d_map[:, :, 1], cmap="viridis")
        axes[3].axis("off")
        axes[3].set_title("GT Heatmap (sigma 5)")

        axes[4].imshow(gt_2d_map[:, :, 2], cmap="viridis")
        axes[4].axis("off")
        axes[4].set_title("GT Heatmap (sigma 60)")

        plt.tight_layout()

        current_name = f"comparison_{config.data.name}_{pc_id}_{x_start}_{y_start}.png"
        plt.savefig(os.path.join(path, current_name))

        plt.close(fig)
        

    print("Successfull finished!")


# --------------
# > Playground <
# --------------
def tryout(config):
    # simple_viusalize_point_cloud(config)
    # torch_tensor_loading(config)

    # bev_segmentation_trying(config)
    # bev_preprocessed_loading_working_testing(config)

    # train_data_testing(config)
    # train_testing(config)
    
    # manhole_intensity_test(config)
    # manhole_density_test(config)
    # manhole_BEV_intensity_test(config)
    # BEV_investigation(config)
    # BEV_Density_investigation(config)
    # bev_dataset_stat_investigation(config)
    # manhole_3d_and_2d_intensity_test(config)  
    # circular_manhole_classification_test(config)
    # center_robustnest_test(config)  # stresstest
    # ransac_inlier_test(config)
    # ransac_downsampling_test(config)
    # point_amount_check(config)
    # squares_circle_shape_test(config)

    # center_prediction_use_labels_as_candidates_test(config)
    # classic_2D_pipeline_test(config)

    # Not done
        # center_prediction_use_labels_as_candidates_without_instances_test(config)
        # center_prediction_without_labels_test(config)
        # center_2D_prediction_use_labels_as_candidates_test(config)
        # center_2D_prediction_use_labels_as_candidates_without_instances_test(config)
        # center_2D_prediction_without_labels_test(config)

    # clustering_tryout(config)
    # make_split(config)

    ground_truth_2d_map_test(config)
    






    