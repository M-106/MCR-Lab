# -----------
# > Imports <
# -----------
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import open3d as o3d
import scipy.ndimage
import scipy

from tqdm import tqdm

import os
import json

from mcrlab.point_cloud.data import get_data_loader
from mcrlab.point_cloud.inspect import print_pc, visualize
from mcrlab.point_cloud.utils import get_coordinate_attribute, \
                                     get_class_attribute
from mcrlab.classic.shape_fit import use_label_candidates_and_extract_center_point, \
                                      use_points_and_extract_center_point
from mcrlab.classic.utils import visualize_circle_fit
from mcrlab.execution.tryout import center_estimation_3d_pipeline_debugging



# ------------------
# > Execution Code <
# ------------------
def add_entry(json_data, cur_point, cur_dataset, cur_pc_id):
    found_entry = False
    for idx_, cur_entry in enumerate(json_data):
        if cur_entry.get("dataset", "-999") == cur_dataset and cur_entry.get("pointcloud-id", "-999") == cur_pc_id:
            if cur_point:
                if hasattr(cur_entry, "centers"):
                    json_data[idx_]["centers"].append(cur_point)
                else:
                    json_data[idx_]["centers"] = [cur_point]

            found_entry = True
            break
    
    if not found_entry:
        if cur_point:
            json_data.append({
                "dataset": cur_dataset,
                "pointcloud-id": cur_pc_id,
                "centers": [cur_point]
            })
        else:
            json_data.append({
            "dataset": cur_dataset,
            "pointcloud-id": cur_pc_id,
            "centers": []
            })

    return json_data

def ground_truth_extraction(config):
    print("\n --- Center Ground Truth Extraction (for Evaluation) ---")

    label_value = 1

    for cur_idx in range(len(config.eval_extraction.data_paths)):
        json_data = list()

        cur_dataset = config.eval_extraction.names[cur_idx]

        data_loader = get_data_loader(config.eval_extraction.names[cur_idx], 
                                      config.eval_extraction.data_paths[cur_idx], 
                                      type=config.eval_extraction.type, 
                                      transform=None,  # get_basic_transform(num_points=-1),
                                      batch_size=1, shuffle=False, num_workers=0,
                                      preprocessed=config.eval_extraction.preprocessed, 
                                      return_train_format=False)

        point_cloud_paths = data_loader.dataset.point_cloud_paths

        for idx, batch in enumerate(data_loader):
            point_cloud = batch[0]
            # point_cloud = point_cloud.get_as_o3d()
            # print_pc(point_cloud)

            _, cur_pc_name = os.path.split(point_cloud_paths[idx])
            cur_pc_name = ".".join(cur_pc_name.replace("preprocessed_patch_", "").split(".")[:-1])
            cur_pc_id = cur_pc_name.split("_")[0]

            # get cluster
            _, _, _, original_cluster_pcs, _, _, _ = center_estimation_3d_pipeline_debugging(point_cloud, method="least_square", extended_return=True, should_visualize=False, label_value=label_value)

            if original_cluster_pcs is None:
                json_data = add_entry(
                    json_data=json_data, 
                    cur_point=None, 
                    cur_dataset=cur_dataset, 
                    cur_pc_id=cur_pc_id
                )
                continue

            print("\n> Least Square Circle Fit Check <\n")
            center_coordinates_square, radius_squares, points_square, cluster_point_clouds, _, error, _ = center_estimation_3d_pipeline_debugging(None, method="least_square", extended_return=True, should_visualize=False, clusters=original_cluster_pcs, label_value=label_value)
            center_coordinates_ransac, _, _, _, _, _, _ = center_estimation_3d_pipeline_debugging(None, method="ransac", extended_return=True, should_visualize=False, clusters=original_cluster_pcs, label_value=label_value)

            for cur_manhole_idx in range(len(points_square)):
                # cur_points = points_square[cur_manhole_idx]
                if config.eval_extraction.center_algorithm == "squares":
                    cur_center = center_coordinates_square[cur_manhole_idx]
                elif config.eval_extraction.center_algorithm == "mean":
                    points_ = points_square[cur_manhole_idx]
                    cur_center = np.array([np.mean(points_[:, 0]), np.mean(points_[:, 1]), np.mean(points_[:, 2])])
                elif config.eval_extraction.center_algorithm == "ransac":
                    cur_center = center_coordinates_ransac[cur_manhole_idx]
                elif config.eval_extraction.center_algorithm == "mesqra":
                    cur_center_squares = center_coordinates_square[cur_manhole_idx]
                    
                    cur_center_ransac = center_coordinates_ransac[cur_manhole_idx]
                    
                    points_ = points_square[cur_manhole_idx]
                    cur_center_mean = np.array([np.mean(points_[:, 0]), np.mean(points_[:, 1]), np.mean(points_[:, 2])])
                
                    # cur_center = np.mean(np.concatenate([cur_center_squares, cur_center_ransac, cur_center_mean], axis=0).reshape(3, -1), axis=0)
                    cur_center = np.mean(np.stack([cur_center_squares, cur_center_ransac, cur_center_mean], axis=0), axis=0)
                else:
                    raise ValueError(f"Unknown center extraction algorith: '{config.eval_extraction.center_algorithm}'")

                # cur_radius = radius_squares[cur_manhole_idx]
                cur_point = {
                    "x": float(cur_center[0]), 
                    "y": float(cur_center[1]), 
                    "z": float(cur_center[2])
                }
                
                # check if there is already an entry where we just can add the center
                json_data = add_entry(
                    json_data=json_data, 
                    cur_point=cur_point, 
                    cur_dataset=cur_dataset, 
                    cur_pc_id=cur_pc_id
                )

        save_path = os.path.join(config.eval_extraction.save_path, f"{cur_dataset}_eval_ground_truths_{config.eval_extraction.center_algorithm}.json")
    
        with open(save_path, "w", encoding="utf-8") as json_file:
            json.dump(json_data, json_file, indent=4, ensure_ascii=False)

        print("Successfull finished!")



# --------------------------
# > Training GT Generation <
# --------------------------
# generates 2D binary center maps and heatmaps (blurred binary center maps)


# Generates a 2D Gaussian heatmap centered at the given pixel coordinates.
def generate_gaussian_heatmap(shape, center, sigma=3):
    heatmap = np.zeros(shape, dtype=np.float32)
    y, x = center
    if 0 <= x < shape[1] and 0 <= y < shape[0]:
        heatmap[y, x] = 1.0
        heatmap = scipy.ndimage.gaussian_filter(heatmap, sigma=sigma)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
    return heatmap



def generate_gaussian_heatmap_3d(shape, center, sigma=(3, 3, 3)):
    """
    shape: (depth, height, width) -> (z_dim, y_dim, x_dim)
    center: (z, y, x) pixel coordinates
    sigma: (sigma_z, sigma_y, sigma_x)
    """
    heatmap = np.zeros(shape, dtype=np.float32)
    z, y, x = center
    
    # Check if the center is within the defined 3D volume
    if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
        heatmap[z, y, x] = 1.0
        # Apply 3D Gaussian blur
        heatmap = scipy.ndimage.gaussian_filter(heatmap, sigma=sigma)
        # Normalize
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
    return heatmap



def get_heatmap_values_for_points(point_cloud, volume, z_min, xstart, ystart, resolution):
    # point coordinates to indices
    coords = point_cloud[:, :3]
    indices = np.floor((coords - np.array([xstart, ystart, z_min])) / resolution).astype(int)
    
    # clipping agaisnt IndexOutOfBounds errors
    indices[:, 0] = np.clip(indices[:, 0], 0, patch_width - 1)  # X
    indices[:, 1] = np.clip(indices[:, 1], 0, patch_height - 1) # Y
    indices[:, 2] = np.clip(indices[:, 2], 0, depth_size - 1)   # Z
    
    # get heatmap value
    point_heatmaps = volume[indices[:, 2], indices[:, 1], indices[:, 0], 0] 
    
    return point_heatmaps



def ground_truth_extraction_heatmap(config):
    print("\n --- Aligned Heatmap Center Ground Truth Extraction ---")

    output_dir = os.path.join(
        os.path.dirname(config.eval_extraction.save_path), "2d_gt_patches"
    )
    os.makedirs(output_dir, exist_ok=True)
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    generate_3d_heatmaps = config.eval_extraction.generate_also_3d_gt_maps

    label_value = 1

    # using exactly the same value as in projection
    tile_size = 5.0  # exactly like bev_tile_size
    resolution = 0.01  # exactly like bev_resolution

    # Exactly the dimension in projection
    patch_height = int(tile_size / resolution)  # 5.0 / 0.01 = 500
    patch_width = int(tile_size / resolution)  # 5.0 / 0.01 = 500

    for cur_idx in range(len(config.eval_extraction.data_paths)):
        dataset_name = config.eval_extraction.names[cur_idx]

        data_loader = get_data_loader(
            dataset_name,
            config.eval_extraction.data_paths[cur_idx],
            type=config.eval_extraction.type,
            transform=None,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            preprocessed=config.eval_extraction.preprocessed,
            return_train_format=False,
        )

        point_cloud_paths = data_loader.dataset.point_cloud_paths

        for idx, batch in enumerate(data_loader):
            point_cloud = batch[0]

            if generate_3d_heatmaps:
                point_cloud_numpy = point_cloud.point["positions"].numpy()
                z_min = np.min(point_cloud_numpy[:, 2])
                z_max = np.max(point_cloud_numpy[:, 2])
                z_range = z_max - z_min
                z_resolution = 0.01  # Matches your XY resolution
                depth_size = int(z_range / z_resolution) + 1

                gt_volume_3d = np.zeros((depth_size, patch_height, patch_width, 3), dtype=np.float32)

            # Filename-Parsing (Example: preprocessed_patch_pc123_150.0_230.0.h5)
            _, cur_pc_name = os.path.split(point_cloud_paths[idx])
            cur_pc_name = ".".join(
                cur_pc_name.replace("preprocessed_patch_", "").split(".")[
                    :-1
                ]
            )

            parts = cur_pc_name.split("_")
            cur_pc_id = parts[0]

            try:
                xstart = float(parts[1])
                ystart = float(parts[2])
            except (IndexError, ValueError):
                print(
                    f"Warning: Could not parse coordinates from {cur_pc_name}."
                )
                continue

            # Cluster & Pipelines request
            (_, _, _, original_cluster_pcs, _, _, _) = center_estimation_3d_pipeline_debugging(
                point_cloud,
                method="least_square",
                extended_return=True,
                should_visualize=False,
                label_value=label_value,
            )

            if original_cluster_pcs is None:
                gt_channels = np.zeros(
                    (patch_height, patch_width, 3), dtype=np.float32
                )
            else:
                (center_coordinates_square, _, points_square, _, _, _, _) = center_estimation_3d_pipeline_debugging(
                    None,
                    method="least_square",
                    extended_return=True,
                    should_visualize=False,
                    clusters=original_cluster_pcs,
                    label_value=label_value,
                )

                # Init mask (Format HxWxC for the saving/training)
                # Channel 0: Binary, Channel 1: Heatmap
                gt_channels = np.zeros(
                    (patch_height, patch_width, 3), dtype=np.float32
                )

                for cur_manhole_idx in range(len(points_square)):
                    if config.eval_extraction.center_algorithm == "squares":
                        cur_center = center_coordinates_square[cur_manhole_idx]
                    else:
                        points_ = points_square[cur_manhole_idx]
                        cur_center = np.array(
                            [
                                np.mean(points_[:, 0]),
                                np.mean(points_[:, 1]),
                                np.mean(points_[:, 2]),
                            ]
                        )

                    # important remapping logic from projection of input images
                    # use np.floor() exactly like in `bev_projection` function!
                    pixel_x = int(np.floor((cur_center[0] - xstart) / resolution))
                    pixel_y = int(np.floor((cur_center[1] - ystart) / resolution))

                    # make sure the values really lay inside of the image/map
                    pixel_x = np.clip(pixel_x, 0, patch_width - 1)
                    pixel_y = np.clip(pixel_y, 0, patch_height - 1)

                    # Channel 0: set binary logic (careful: indeces y, x same to Numba prjection)
                    # gt_channels[pixel_y, pixel_x, 0] = 1.0
                    heatmap = generate_gaussian_heatmap(
                        (patch_height, patch_width), (pixel_y, pixel_x), sigma=10
                    )
                    gt_channels[:, :, 0] = np.maximum(
                        gt_channels[:, :, 0], heatmap
                    )

                    # Channel 1: generate and accumulate Heatmap
                    # Sigma=3 means at Res=0.01 a radius from round about 3cm around the center
                    heatmap = generate_gaussian_heatmap(
                        (patch_height, patch_width), (pixel_y, pixel_x), sigma=20
                    )
                    gt_channels[:, :, 1] = np.maximum(
                        gt_channels[:, :, 1], heatmap
                    )
                    # Channel 2: generate and accumulate Heatmap (but greater -> 60 for 60 cm average size)
                    # Sigma=3 means at Res=0.01 a radius from round about 3cm around the center
                    heatmap = generate_gaussian_heatmap(
                        (patch_height, patch_width), (pixel_y, pixel_x), sigma=60
                    )
                    gt_channels[:, :, 2] = np.maximum(
                        gt_channels[:, :, 2], heatmap
                    )

                    # compute 3D Heatmap
                    if generate_3d_heatmaps:
                        # # Calculate Z index exactly like X and Y
                        pixel_z = int(np.floor((cur_center[2] - z_min) / z_resolution))
                        pixel_z = np.clip(pixel_z, 0, depth_size - 1)
                        # pixel_x = cur_center[0]
                        # pixel_y = cur_center[1]
                        # pixel_z = cur_center[2]

                        # generate and accumulate for each channel
                        for i, sigma in enumerate([10, 20, 60]):
                            heatmap_3d = generate_gaussian_heatmap_3d(
                                (depth_size, patch_height, patch_width), 
                                (pixel_z, pixel_y, pixel_x), 
                                sigma=(sigma, sigma, sigma) # Adjust Z-sigma if needed
                            )
                            gt_volume_3d[:, :, :, i] = np.maximum(gt_volume_3d[:, :, :, i], heatmap_3d)

                # point_heatmaps = get_heatmap_values_for_points(point_cloud, gt_volume_3d,  z_min, xstart, ystart, resolution)
                # point_cloud_with_heatmap = np.hstack([point_cloud, point_heatmaps.reshape(-1, 1)])

            # save as .npy (Dataset_PCID_X_Y.npy)
            file_name = f"{dataset_name}_{cur_pc_id}_{xstart}_{ystart}.npy"
            save_file_path = os.path.join(output_dir, file_name)
            # shutil.rmtree(output_dir)
            np.save(save_file_path, gt_channels)

            if generate_3d_heatmaps:
                np.save(save_file_path.replace(".npy", "_3d.npy"), gt_volume_3d)

    print("Finished succefully! The GT Maps are now pixel-precise synchron.")






