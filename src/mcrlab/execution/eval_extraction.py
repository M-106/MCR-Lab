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

