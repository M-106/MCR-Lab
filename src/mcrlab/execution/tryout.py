# -----------
# > Imports <
# -----------
import shutil

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
from mcrlab.point_cloud.inspect import print_pc, visualize, visualize_intensity_in_2d, \
                                       analyze_point_distribution
from mcrlab.point_cloud.tensor_wrapper import PointCloudTensor
from mcrlab.projection import bev_projection, bev_back_projection, bev_back_projection_testing
from mcrlab.image.utils import normalize_img_per_channel
from mcrlab.image.io import save_bev_tiles_as_images
from mcrlab.models.segmentation import SegFormer, SAM2, SAM3, DinoMask2Former
from mcrlab.point_cloud.utils import get_coordinate_attribute, get_intensity_attribute, \
                                     get_class_attribute, get_instance_attribute, \
                                     extract_manhole, add_random_dense_manipulation_point_cloud, \
                                     classify_manhole
from mcrlab.geometry.shape_fit import use_label_candidates_and_extract_center_point, \
                                      use_points_and_extract_center_point
from mcrlab.geometry.utils import visualize_circle_fit



# -----------------------
# > Different Scenarios <
# -----------------------
def simple_viusalize_point_cloud(config):
    # if config.data.name == "paris":
    #     dataset = ParisLille3DDataset(path=config.data.path, type="train", transform=None, 
    #                                   preprocessed=config.data.preprocessed, return_train_format=False)
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                  type="train", 
                                  transform=get_basic_transform(num_points=-1),
                                  batch_size=1, shuffle=False, num_workers=1,
                                  preprocessed=config.data.preprocessed, return_train_format=False)

    point_cloud = next(iter(data_loader))[0]

    print_pc(point_cloud)
    visualize(point_cloud, color_mode="class")



def torch_tensor_loading(config):
    # PyTorch Dataset try out
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type="train", 
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
                                    type="train", 
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
                                    type="train", 
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
                                    type="train", 
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
                                    type="train", 
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
                                    type="train", 
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
                                    type="train", 
                                    transform=None,  # get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    # clear save path
    path = "./output/manhole_intensity"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    cur_pc = 0

    for batch in data_loader:
        point_cloud = batch[0]
        # point_cloud = point_cloud.get_as_o3d()
        print_pc(point_cloud)

        print("\n> Manhole Intensity Check <\n")
        manholes = extract_manhole(point_cloud, label_value=104002, points_around_dist=2)

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



def manhole_BEV_intensity_test(config, label_value=104002):
    # FIXME -> go through BEV images and if it have the label than plot the image/save image 
    #                   -> have already a method right (but maybe use normalization if not visible)
    print("\n --- Manhole BEV Check ---")

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type="train", 
                                    transform=get_basic_transform(),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    for batch in data_loader:
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

        for bev_item in bev_gen:
            img = bev_item["pixel_values"].detach().cpu().numpy()
            labels = bev_item["labels"].detach().cpu().numpy()
            meta = bev_item["meta"] 

            # extracting manholes? -> get all manhole points + clustering

            # print(labels.shape)
            if np.any(labels == label_value):
                H, W = labels.shape
                colored_img = np.full((H, W, 3), 0.0, dtype=np.float32)
                colored_img[labels == label_value] = [255.0, 255.00, 0.0]

                # fix img shape -> C, H, W -> H, W, C
                img_t = np.transpose(img[:3, :, :], (1, 2, 0))

                fig, ax = plt.subplots(figsize=(15,7), ncols=3, nrows=1)

                ax[0].imshow(img_t, cmap="viridis")
                ax[1].imshow((img_t - np.min(img_t))/(np.max(img_t) - np.min(img_t)), cmap="viridis")
                ax[2].imshow(colored_img)

                ax[0].axis("off")
                ax[1].axis("off")
                ax[2].axis("off")

                ax[0].set_title("BEV Image", fontsize=14, fontweight='bold')
                ax[1].set_title("Normalized BEV Image", fontsize=14, fontweight='bold')
                ax[2].set_title("Manhole MArked BEV Image", fontsize=14, fontweight='bold')

                plt.show()
                # break
    
        # break



def manhole_density_test(config):
    print("\n --- Manhole Density Check ---")

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type="train", 
                                    transform=None,  # get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    total_result = dict()

    print("\n> Manhole Density Check <\n")

    for batch in tqdm(data_loader, total=len(data_loader), desc="Density Check"):
        point_cloud = batch[0]
        # point_cloud = point_cloud.get_as_o3d()
        # print_pc(point_cloud)

        manholes = extract_manhole(point_cloud, label_value=104002, points_around_dist=0)

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
                                    type="train", 
                                    transform=None,  # get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    # clear save path
    path = "./output/center_shape_check"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    cur_pc = 0
    total_result = {"Circle": 0, "No Circle": 0}
    for batch in tqdm(data_loader, total=len(data_loader), desc="Center Estimation Stresstest"):
        point_cloud = batch[0]
        # point_cloud = point_cloud.get_as_o3d()
        # print_pc(point_cloud)

        # get maholes
        manholes = extract_manhole(point_cloud, label_value=104002, points_around_dist=0)

        for cur_vis, cur_manhole in enumerate(manholes):
            plot_name = f"pc_{cur_pc}_manhole_{cur_vis}.png"
            is_circle = classify_manhole(cur_manhole, save_path=os.path.join(path, plot_name), should_plot=False)
            if is_circle:
                total_result["Circle"] += 1
            else:
                total_result["No Circle"] += 1

        cur_pc += 1
        
    print(total_result)



def center_robustnest_test(config):
    print("\n --- Stresstest Center Estimation (with labels) ---")

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type="train", 
                                    transform=None,  # get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    # clear save path
    path = "./output/center_estimation_stresstest"
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
        _, _, _, original_cluster_pcs, _ = center_estimation_3d_pipeline_debugging(point_cloud, method="least_square", extended_return=True, should_visualize=False)
        
        if original_cluster_pcs is None:
            cur_pc += 1
            continue

        manipulated_clusters = []
        for cur_cluster in original_cluster_pcs:
            manipulated_clusters.append(
                add_random_dense_manipulation_point_cloud(cur_cluster, n=np.random.randint(1, max(10, 10*cur_pc)))
            )

        print("\n> Least Square Circle Fit Check <\n")
        center_coordinates_square, radius_squares, points_square, cluster_point_clouds, error_s = center_estimation_3d_pipeline_debugging(None, method="least_square", extended_return=True, should_visualize=False, clusters=manipulated_clusters)

        print("\n> RANSAC Fit Check <\n")
        center_coordinates_ransac, radius_ransac, points_ransac, cluster_point_clouds, error_r = center_estimation_3d_pipeline_debugging(None, method="ransac", extended_return=True, should_visualize=False, clusters=manipulated_clusters)

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



def center_estimation_3d_pipeline_debugging(point_cloud, method, extended_return=False, should_visualize=True, clusters=None):
    """
    Helper Function
    """
    print("Compute centers...")
    if point_cloud is not None:
        center_points = use_label_candidates_and_extract_center_point(points=point_cloud, 
                                                                    use_2d_version=False, 
                                                                    label_value=104002, 
                                                                    method=method, 
                                                                    use_projection=True, 
                                                                    cluster_if_needed=True)
    else:
        center_points = use_points_and_extract_center_point(clusters=clusters, 
                                                            method=method, 
                                                            use_projection=True)
        
    if len(center_points) <= 0:
        if extended_return:
            return center_points, None, None, None, None
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
        cluster_point_clouds = []

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
            cluster_point_clouds.append(cluster)

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
        return center_coordinates, all_radius, cluster_points, cluster_point_clouds, total_error
    else:
        return center_coordinates



def center_prediction_use_labels_as_candidates_test(config):
    print("\n --- Center Estimation (with labels) ---")

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type="train", 
                                    transform=None,  # get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    # clear save path
    path = "./output/center_estimation"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    cur_pc = 0

    for batch in data_loader:
        point_cloud = batch[0]
        # point_cloud = point_cloud.get_as_o3d()
        print_pc(point_cloud)

        # get cluster
        _, _, _, original_cluster_pcs, _ = center_estimation_3d_pipeline_debugging(point_cloud, method="least_square", extended_return=True, should_visualize=False)

        if original_cluster_pcs is None:
            cur_pc += 1
            continue

        print("\n> Least Square Circle Fit Check <\n")
        center_coordinates_square, radius_squares, points_square, cluster_point_clouds, error = center_estimation_3d_pipeline_debugging(None, method="least_square", extended_return=True, should_visualize=False, clusters=original_cluster_pcs)

        # # visualize error
        # for cur_vis in range(len(points_square)):
        #     visualize_circle_fit(points=points_square[cur_vis], 
        #                          center_pred=center_coordinates_square[cur_vis], 
        #                          radius=radius_squares[cur_vis], 
        #                          error=error[cur_vis])

        print("\n> RANSAC Fit Check <\n")
        center_coordinates_ransac, radius_ransac, points, cluster_point_clouds, error = center_estimation_3d_pipeline_debugging(None, method="ransac", extended_return=True, clusters=original_cluster_pcs, should_visualize=False)
        # print(f"Center Coordinates, RANSAC: {center_coordinates_ransac}")
        # # visualize error
        # for cur_vis in range(len(points)):
        #     visualize_circle_fit(points=points[cur_vis], 
        #                          center_pred=center_coordinates_ransac[cur_vis], 
        #                          radius=radius_ransac[cur_vis], 
        #                          error=error[cur_vis])

        # DEBUGGING
        print(points[0].shape)
        print(points_square[0].shape)
        # print(points == points_square)

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
    
    # manhole_intensity_test(config)
    # manhole_density_test(config)
    # manhole_BEV_intensity_test(config, label_value=104002)
    # circular_manhole_classification_test(config)

    # center_robustnest_test(config)
    center_prediction_use_labels_as_candidates_test(config)

    # Not done
        # center_prediction_use_labels_as_candidates_without_instances_test(config)
        # center_prediction_without_labels_test(config)
        # center_2D_prediction_use_labels_as_candidates_test(config)
        # center_2D_prediction_use_labels_as_candidates_without_instances_test(config)
        # center_2D_prediction_without_labels_test(config)
    

    