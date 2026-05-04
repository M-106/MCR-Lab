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
from mcrlab.projection import bev_projection, bev_back_projection, bev_back_projection_testing
from mcrlab.image.utils import normalize_img_per_channel
from mcrlab.image.io import save_bev_tiles_as_images
from mcrlab.models.segmentation import SegFormer, SAM2, SAM3, DinoMask2Former
from mcrlab.point_cloud.utils import get_coordinate_attribute, get_intensity_attribute, \
                                     get_class_attribute
from mcrlab.geometry.shape_fit import use_label_candidates_and_extract_center_point
from mcrlab.geometry.utils import visualize_circle_fit



# -----------------------
# > Different Scenarios <
# -----------------------
def simple_viusalize_point_cloud(config):
    # if config.data.name == "paris":
    #     dataset = ParisLille3DDataset(path=config.data.path, testdata=False, transform=None, 
    #                                   preprocessed=config.data.preprocessed, return_train_format=False)
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                  testdata=False, 
                                  transform=get_basic_transform(num_points=-1),
                                  batch_size=1, shuffle=False, num_workers=1,
                                  preprocessed=config.data.preprocessed, return_train_format=False)

    point_cloud = next(iter(data_loader))[0]

    print_pc(point_cloud)
    visualize(point_cloud, color_mode="class")



def torch_tensor_loading(config):
    # PyTorch Dataset try out
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    testdata=False, 
                                    transform=get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=1,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

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
                    remapping = bev_back_projection(point_cloud, metas, tile_id=0, pixel_x=cur_x, pixel_y=cur_y)
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



def bev_working_testing(config):
    # LOAD POINT CLOUD
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    testdata=False, 
                                    transform=None,  # get_basic_transform(num_points=-1), 
                                    batch_size=1, shuffle=False, num_workers=1,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    for batch in data_loader:
        point_cloud = batch[0]

        # point_cloud = point_cloud.get_as_o3d()
        if not isinstance(point_cloud, o3d.t.geometry.PointCloud):
            raise TypeError(f"Point Cloud should be get as Open3D Tensor, but got '{type(point_cloud)}'")
        print_pc(point_cloud)

        print("Starting BEV projection...")
        # tiles, meta = bev_projection_numba_and_open3d(point_cloud, tile_size=35.0, resolution=0.05, include_class=True)
        tiles, metas = bev_projection(point_cloud, tile_size=35.0, resolution=0.05, overlap=0.0,
                                          include_class=True, direct_single_saving=False)  #  tile_size=100.0/50.0, resolution=0.2/0.1
        bev_gen = bev_gen_wrapper(tiles, metas)
        
        # if point_cloud.bevs is None:
        #     print("Starting BEV projection...")
        #     tiles, meta = bev_projection_numba(point_cloud, tile_size=35.0, resolution=0.05)  #  tile_size=100.0/50.0, resolution=0.2/0.1
        # else:
        #     print("Loaded Bevs from file...")
        #     tiles = point_cloud.bevs
        #     meta = point_cloud.meta

        bev_back_projection_testing(point_cloud, bev_gen, bev_amount=len(tiles))

        # do not end after one testset?
        break



def bev_preprocessed_loading_working_testing(config):
    if not config.data.preprocessed:
        raise ValueError("'Preprocessing' must be True! (config.data.preprocessed)")

    # LOAD POINT CLOUD
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    testdata=False, 
                                    transform=get_basic_transform(num_points=-1), 
                                    batch_size=1, shuffle=False, num_workers=1,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    for batch in data_loader:
        point_cloud = batch[0]

        assert isinstance(point_cloud, PointCloudTensor)
        print_pc(point_cloud)

        print("Starting BEV projection...")
        if point_cloud.bev_data is None:
            raise ValueError("Preprocessed BEVs did not loaded.")
            print("Starting BEV projection...")
            tiles, metas = bev_projection(point_cloud, tile_size=35.0, resolution=0.05)  #  tile_size=100.0/50.0, resolution=0.2/0.1
        else:
            print("Loaded Bevs from file...")
            bev_gen = point_cloud.get_bev()

        bev_back_projection_testing(point_cloud, bev_gen, bev_amount=point_cloud.bev_amount)

        # do not end after one testset?
        break



def train_data_testing(config):
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    testdata=False, 
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


def center_estimation_3d_pipeline_debugging(point_cloud, method, extended_return=False):
    """
    Helper Function
    """
    print("Compute centers...")
    center_points = use_label_candidates_and_extract_center_point(points=point_cloud, 
                                                                use_2d_version=False, 
                                                                label_value=104002, 
                                                                method=method, 
                                                                use_projection=True, 
                                                                cluster_if_needed=True)
    
    # visualize -> all points in black, manholes in yellow and center point in red
    print("Compute loss and error + preprare visualization...")
    pcd_vis = point_cloud.clone()

    class_key = get_class_attribute(point_cloud)
    classes = point_cloud.point[class_key].cpu().numpy()

    color = np.full([classes.shape[0], 3], 0.0, dtype=np.float32)
    # color[classes != 104002] = [0.0, 0.0, 0.0]
    color[(classes == 104002).flatten()] = [1.0, 0.95, 0.0]
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

    for idx, item in enumerate(center_points):
        center, radius, cluster, error, loss = item
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

    print("Visualize (yellow are manhole and red the predicted center points)")
    new_colors = np.full((center_coordinates.shape[0], 3), [1.0, 0.0, 0.1], dtype=np.float32)
    new_colors = o3d.core.Tensor(new_colors, dtype=o3d.core.Dtype.Float32)

    coordinate_key = get_coordinate_attribute(point_cloud)
    new_point = o3d.core.Tensor(center_coordinates, dtype=pcd_vis.point[coordinate_key].dtype)
    pcd_vis.point[coordinate_key] = o3d.core.concatenate([pcd_vis.point[coordinate_key], new_point], axis=0)
    pcd_vis.point["colors"] = o3d.core.concatenate([pcd_vis.point["colors"], new_colors], axis=0)

    visualize(pcd_vis, color_mode=None)

    if extended_return:
        return center_coordinates, all_radius, cluster_points, total_error
    else:
        return center_coordinates



def center_prediction_use_labels_as_candidates_test(config):
    print("\n --- Center Estimation (with labels) ---")

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    testdata=False, 
                                    transform=None,  # get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    for batch in data_loader:
        point_cloud = batch[0]
        # point_cloud = point_cloud.get_as_o3d()
        print_pc(point_cloud)

        print("\n> Least Square Circle Fit Check <\n")
        center_coordinates_square, radius, points, error = center_estimation_3d_pipeline_debugging(point_cloud, method="least_square", extended_return=True)

        # visualize error
        for cur_vis in range(len(points)):
            visualize_circle_fit(points=points[cur_vis], 
                                 center_pred=center_coordinates_square[cur_vis], 
                                 radius=radius[cur_vis], 
                                 error=error[cur_vis])

        # print("\n> RANSAC Fit Check <\n")
        # center_coordinates_ransac = center_estimation_3d_pipeline_debugging(point_cloud, method="ransac")
        # FIXME -> RANSAC check, look at the outputed points
        # print(f"Center Coordinates, RANSAC: {center_coordinates_ransac}")

        # compare similarity?
        # FIXME -> check the difference of the center points! Should have the same order

        # Visualize Error
        # FIXME

        break



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



# --------------
# > Playground <
# --------------
def tryout(config):
    # simple_viusalize_point_cloud(config)
    # torch_tensor_loading(config)

    # bev_trying(config)
    # bev_segmentation_trying(config)
    # bev_working_testing(config)
    # bev_preprocessed_loading_working_testing(config)  # still try this again!

    # train_data_testing(config)
    # train_testing(config)
    
    center_prediction_use_labels_as_candidates_test(config)
    # center_prediction_use_labels_as_candidates_without_instances_test(config)
    # center_prediction_without_labels_test(config)
    # center_2D_prediction_use_labels_as_candidates_test(config)
    # center_2D_prediction_use_labels_as_candidates_without_instances_test(config)
    # center_2D_prediction_without_labels_test(config)
    

    