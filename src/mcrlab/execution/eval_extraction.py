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
def ground_truth_extraction(config):
    print("\n --- Center Ground Truth Extraction (for Evaluation) ---")

    json_data = list()

    label_value = 1

    for cur_idx in range(len(config.eval_extraction.data_paths)):
        json_data.append({
            "dataset": config.eval_extraction.names[cur_idx],
            "data": list()
        })

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

            pc_idx = None
            for idx_, cur_data in enumerate(json_data[cur_idx]["data"]):
                if cur_data["pointcloud-id"] == cur_pc_id:
                    pc_idx = idx_
                
            if pc_idx is None:
                json_data[cur_idx]["data"].append(
                    {
                        "pointcloud-id": cur_pc_id,
                        "centers": list()
                    }
                ) 
                pc_idx = -1

            # get cluster
            _, _, _, original_cluster_pcs, _, _, _ = center_estimation_3d_pipeline_debugging(point_cloud, method="least_square", extended_return=True, should_visualize=False, label_value=label_value)

            if original_cluster_pcs is None:
                continue

            print("\n> Least Square Circle Fit Check <\n")
            center_coordinates_square, radius_squares, points_square, cluster_point_clouds, _, error, _ = center_estimation_3d_pipeline_debugging(None, method="least_square", extended_return=True, should_visualize=False, clusters=original_cluster_pcs, label_value=label_value)

            for cur_manhole_idx in range(len(points_square)):
                # cur_points = points_square[cur_manhole_idx]
                if config.eval_extraction.center_algorithm == "squares":
                    cur_center = center_coordinates_square[cur_manhole_idx]
                else:
                    points_ = points_square[cur_manhole_idx]
                    cur_center = np.array([np.mean(points_[:, 0]), np.mean(points_[:, 1]), np.mean(points_[:, 2])])
                
                # cur_radius = radius_squares[cur_manhole_idx]
                
                # save center -> cur_pc_id "pointcloud-id"
                assert json_data[cur_idx]["data"][pc_idx]["pointcloud-id"] == cur_pc_id

                json_data[cur_idx]["data"][pc_idx]["centers"].append(
                    {
                        "x": float(cur_center[0]), 
                        "y": float(cur_center[1]), 
                        "z": float(cur_center[2])
                    }
                ) 

    
    with open(config.eval_extraction.save_path, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)

    # FIXME call a check function here to verify the found data is correct? Or not?

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

def ground_truth_extraction_2d(config):
    print("\n --- Aligned 2D Center Ground Truth Extraction ---")

    output_dir = os.path.join(
        os.path.dirname(config.eval_extraction.save_path), "2d_gt_patches"
    )
    os.makedirs(output_dir, exist_ok=True)

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
                    gt_channels[pixel_y, pixel_x, 0] = 1.0

                    # Channel 1: generate and accumulate Heatmap
                    # Sigma=3 means at Res=0.01 a radius from round about 3cm around the center
                    heatmap = generate_gaussian_heatmap(
                        (patch_height, patch_width), (pixel_y, pixel_x), sigma=5
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

            # save as .npy (Dataset_PCID_X_Y.npy)
            file_name = f"{dataset_name}_{cur_pc_id}_{xstart}_{ystart}.npy"
            save_file_path = os.path.join(output_dir, file_name)
            np.save(save_file_path, gt_channels)

    print("Finished succefully! The GT Maps are now pixel-precise synchron.")






