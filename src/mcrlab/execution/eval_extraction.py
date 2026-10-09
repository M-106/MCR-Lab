# -----------
# > Imports <
# -----------
import shutil
import os

import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import open3d as o3d
import scipy.ndimage
import scipy

from tqdm import tqdm

from mcrlab.point_cloud.data import get_data_loader, get_filtered_raw_transform
from mcrlab.point_cloud.inspect import print_pc, visualize
from mcrlab.point_cloud.utils import get_coordinate_attribute, \
                                     get_class_attribute
from mcrlab.classic.shape_fit import use_label_candidates_and_extract_center_point, \
                                      use_points_and_extract_center_point
from mcrlab.classic.utils import visualize_circle_fit
from mcrlab.execution.tryout import center_estimation_3d_pipeline_debugging
# from mcrlab.helper import save_dir_creation



# ------------------
# > Execution Code <
# ------------------
def add_entry(json_data, cur_point, cur_dataset, cur_pc_id):
    found_entry = False
    for idx_, cur_entry in enumerate(json_data):
        if cur_entry.get("dataset", "-999") == cur_dataset and cur_entry.get("pointcloud-id", "-999") == cur_pc_id:
            if cur_point:
                if "centers" in cur_entry and isinstance(cur_entry["centers"], list):
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

        # Create output directory for image plots
        plot_save_dir = os.path.join(
            config.eval_extraction.save_path,   # "/out/center_gt_extraction",
            "center_gt/plots", 
            f"{cur_dataset}_{config.eval_extraction.center_algorithm}"
        )

        # save_dir_creation(plot_save_dir)
        os.makedirs(plot_save_dir, exist_ok=True)
        shutil.rmtree(plot_save_dir)
        # save_dir_creation(plot_save_dir)
        os.makedirs(plot_save_dir, exist_ok=True)

        data_loader = get_data_loader(config.eval_extraction.names[cur_idx], 
                                      config.eval_extraction.data_paths[cur_idx], 
                                      type="test",   # config.eval_extraction.type, 
                                      transform=get_filtered_raw_transform(manhole_label=3 if "sud" in cur_dataset.lower() else 104002),  # get_basic_transform(num_points=-1),
                                      batch_size=1, shuffle=False, num_workers=0,
                                      preprocessed=False, 
                                      return_train_format=False, 
                                      bev_normalized=False,  # config.data.normalization, 
                                      bev_normalize_mode=config.data.normalization_mode)

        point_cloud_paths = data_loader.dataset.point_cloud_paths

        for idx, batch in enumerate(data_loader):
            point_cloud = batch[0]
            # point_cloud = point_cloud.get_as_o3d()
            # print_pc(point_cloud)

            _, cur_pc_name = os.path.split(point_cloud_paths[idx])
            # cur_pc_name = ".".join(cur_pc_name.replace("preprocessed_patch_", "").split(".")[:-1])
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

            extracted_centers = []

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

                extracted_centers.append(cur_center)
                
                # check if there is already an entry where we just can add the center
                json_data = add_entry(
                    json_data=json_data, 
                    cur_point=cur_point, 
                    cur_dataset=cur_dataset, 
                    cur_pc_id=cur_pc_id
                )

            # Save visual inspection plot if points exist
            print("Points Amount:", len(points_square))
            if len(points_square) > 0:
                for manhole_idx in range(len(points_square)):
                    all_cluster_pts = points_square[manhole_idx]  # np.vstack(points_square)
                    centers_arr = extracted_centers[manhole_idx]  # np.array(extracted_centers)
                
                    fig = plt.figure(figsize=(12, 5))
                    
                    # 2D Top-Down Projection (XY)
                    ax1 = fig.add_subplot(1, 2, 1)
                    ax1.scatter(all_cluster_pts[:, 0], all_cluster_pts[:, 1], c='gray', s=1, alpha=0.5, label='Points')
                    # ax1.scatter(centers_arr[:, 0], centers_arr[:, 1], c='red', marker='X', s=80, label='Predicted Centers')
                    ax1.scatter(centers_arr[0], centers_arr[1], c='red', marker='X', s=80, label='Predicted Centers')
                    ax1.set_title(f"Top-Down (XY): {cur_pc_id}")
                    ax1.set_xlabel("X")
                    ax1.set_ylabel("Y")
                    ax1.set_aspect('equal', 'datalim')
                    ax1.legend()
                    ax1.grid(True)

                    # 3D Orthographic View
                    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
                    ax2.scatter(all_cluster_pts[:, 0], all_cluster_pts[:, 1], all_cluster_pts[:, 2], c=all_cluster_pts[:, 2], cmap='viridis', s=1, alpha=0.5)
                    # ax2.scatter(centers_arr[:, 0], centers_arr[:, 1], centers_arr[:, 2], c='red', marker='X', s=100, label='Centers')
                    ax2.scatter(centers_arr[0], centers_arr[1], centers_arr[2], c='red', marker='X', s=100, label='Centers')
                    ax2.set_title(f"3D View: {cur_pc_id}")
                    ax2.set_xlabel("X")
                    ax2.set_ylabel("Y")
                    ax2.set_zlabel("Z")

                    plt.suptitle(f"Dataset: {cur_dataset} | ID: {cur_pc_id} | Algo: {config.eval_extraction.center_algorithm}")
                    # plt.tight_layout()

                    # Save to disk and clear RAM
                    fig_path = os.path.join(plot_save_dir, f"{cur_pc_id}_centers_000.png")
                    counter_ = 1
                    while os.path.exists(fig_path):
                        fig_path = os.path.join(plot_save_dir, f"{cur_pc_id}_centers_{counter_:03}.png")
                        counter_ += 1
                    plt.savefig(fig_path, dpi=150)
                    plt.close(fig)

                    print(f"  → Saved fig at '{fig_path}'", flush=True)

        save_path = os.path.join(config.eval_extraction.save_path, f"{cur_dataset}_eval_ground_truths_{config.eval_extraction.center_algorithm}.json")
    
        with open(save_path, "w", encoding="utf-8") as json_file:
            json.dump(json_data, json_file, indent=4, ensure_ascii=False)

        print(f"Saved extracted ground truth centers for dataset '{cur_dataset}' at '{save_path}'", flush=True)

        print("Successfull finished!")



# --------------------------
# > Training GT Generation <
# --------------------------
# generates 2D binary center maps and heatmaps (blurred binary center maps)


# Generates a 2D Gaussian heatmap centered at the given pixel coordinates.
def generate_gaussian_heatmap(shape, center, sigma=3):
    """
    Generates an un-truncated Gaussian heatmap.
    `center` can be float values and lie outside [0, shape] bounds.
    """
    height, width = shape
    center_y, center_x = center

    # 1. Create pixel coordinate grid
    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # 2. Compute continuous squared radial distance
    dist_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2

    # 3. Un-normalized Gaussian (Peak amplitude is exactly 1.0 at center_x, center_y)
    heatmap = np.exp(-dist_sq / (2.0 * (sigma ** 2)))

    return heatmap



def generate_gaussian_heatmap_3d(shape, center, sigma=(3, 3, 3)):
    """
    shape: (depth, height, width) -> (z_dim, y_dim, x_dim)
    center: (center_z, center_y, center_x) in continuous float coordinates
    sigma: (sigma_z, sigma_y, sigma_x) or single float/int
    """
    depth, height, width = shape
    center_z, center_y, center_x = center

    if isinstance(sigma, (int, float)):
        sigma_z = sigma_y = sigma_x = float(sigma)
    else:
        sigma_z, sigma_y, sigma_x = sigma

    z = np.arange(0, depth, dtype=np.float32)
    y = np.arange(0, height, dtype=np.float32)
    x = np.arange(0, width, dtype=np.float32)

    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')

    dist_sq = (
        ((xx - center_x) / sigma_x) ** 2 +
        ((yy - center_y) / sigma_y) ** 2 +
        ((zz - center_z) / sigma_z) ** 2
    )

    return np.exp(-0.5 * dist_sq)



# def get_heatmap_values_for_points(point_cloud, volume, z_min, xstart, ystart, resolution):
#     depth_size, patch_height, patch_width, _ = volume.shape
#     coords = point_cloud[:, :3]
#     indices = np.floor((coords - np.array([xstart, ystart, z_min])) / resolution).astype(int)
    
#     indices[:, 0] = np.clip(indices[:, 0], 0, patch_width - 1)   # X
#     indices[:, 1] = np.clip(indices[:, 1], 0, patch_height - 1)  # Y
#     indices[:, 2] = np.clip(indices[:, 2], 0, depth_size - 1)    # Z
    
#     point_heatmaps = volume[indices[:, 2], indices[:, 1], indices[:, 0], 0] 
#     return point_heatmaps



def extract_raw_center_gt_from_patch(
    raw_point_cloud,          # Pass the loaded raw point cloud directly!
    patch_points,             # The preprocessed 5x5m patch point cloud
    label_value: int,
    method: str = "least_square",
    search_radius: float = 0.8 # Radius in meters to query around the patch cluster center
):
    # Step 1: Detect clusters on the local patch
    _, _, _, original_cluster_pcs, _, _, _ = center_estimation_3d_pipeline_debugging(
        patch_points,
        method=method,
        extended_return=True,
        should_visualize=False,
        label_value=label_value,
    )

    if original_cluster_pcs is None or len(original_cluster_pcs) == 0:
        return None, None

    # Get preliminary patch center estimates
    patch_centers, _, _, _, _, _, _ = center_estimation_3d_pipeline_debugging(
        None,
        method=method,
        extended_return=True,
        should_visualize=False,
        clusters=original_cluster_pcs,
        label_value=label_value,
    )

    raw_positions = raw_point_cloud.point[get_coordinate_attribute(raw_point_cloud)].numpy()
    
    # If raw point cloud has labels/features, extract them accordingly
    # Assuming standard PyTorch Geometric / Open3D structure:
    raw_labels = raw_point_cloud.point["labels"].numpy() if "labels" in raw_point_cloud.point else None

    raw_clusters = []

    # Step 2: Query the RAW point cloud around each patch center estimate
    for approx_center in patch_centers:
        
        # Quick 3D Bounding Box Crop (2m x 2m x 2m cube around center)
        # else memory error, because full point cloud neighbor computation
        in_box_mask = (
            (np.abs(raw_positions[:, 0] - approx_center[0]) <= search_radius) &
            (np.abs(raw_positions[:, 1] - approx_center[1]) <= search_radius) &
            (np.abs(raw_positions[:, 2] - approx_center[2]) <= search_radius)
        )

        # Filter candidate labels if available
        if raw_labels is not None:
            in_box_mask = in_box_mask & (raw_labels == label_value)

        full_cluster_points = raw_positions[in_box_mask]

        # # Precise Radius Filter on the candidate subset
        # dists_3d = np.linalg.norm(candidate_points - approx_center, axis=1)
        # full_cluster_points = candidate_points[dists_3d <= search_radius]

        if len(full_cluster_points) > 0:
            # Create Open3D Tensor PointCloud to match raw_point_cloud structure
            pcd = o3d.t.geometry.PointCloud()
            pcd.point["positions"] = o3d.core.Tensor(full_cluster_points)
            raw_clusters.append(pcd)

    if len(raw_clusters) == 0:
        return None, None

    # Step 3: Re-estimate high-precision centers on the un-cropped raw clusters
    refined_centers, _, refined_points, _, _, _, _ = center_estimation_3d_pipeline_debugging(
        None,
        method=method,
        extended_return=True,
        should_visualize=False,
        clusters=raw_clusters,
        label_value=label_value,
    )

    return refined_centers, refined_points



def ground_truth_extraction_heatmap(config):
    print("\n --- Aligned Heatmap Center Ground Truth Extraction ---")

    output_dir = os.path.join(
        os.path.dirname(config.eval_extraction.save_path), "2d_gt_patches_on_raw_pc"
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

    cur_raw_pc_loaded = None

    for cur_idx in range(len(config.eval_extraction.data_paths)):
        dataset_name = config.eval_extraction.names[cur_idx]

        # load PATCH data (preprocessed data)
        data_loader = get_data_loader(
            dataset_name,
            config.eval_extraction.data_paths[cur_idx],
            type="all",  # config.eval_extraction.type,
            transform=None,  # get_filtered_raw_transform(manhole_label=3 if "sud" in dataset_name.lower() else 104002),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            preprocessed=True,  # fetch preprocessed patches
            return_train_format=False,
            bev_normalized=False,
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

            # load right RAW point cloud for this patch
            if cur_raw_pc_loaded is None or cur_raw_pc_loaded[0] != cur_pc_id:
                # load RAW data
                raw_dataloader = get_data_loader(
                    dataset_name,
                    config.eval_extraction.data_paths[cur_idx],
                    type="all",  # config.eval_extraction.type,
                    transform=get_filtered_raw_transform(manhole_label=3 if "sud" in dataset_name.lower() else 104002),
                    batch_size=1,
                    shuffle=False,
                    preprocessed=False, # Fetch raw full cloud
                    return_train_format=False,
                )

                raw_pc_idx = raw_dataloader.dataset.get_idx_by_pc_id(cur_pc_id)
                raw_point_cloud = raw_dataloader.dataset[raw_pc_idx]
                # raw_point_cloud = next(iter(raw_dataloader))
                # if isinstance(raw_point_cloud, tuple) and len(raw_point_cloud) == 1:
                #     raw_point_cloud = raw_point_cloud[0]
                cur_raw_pc_loaded = (cur_pc_id, raw_point_cloud)

            # Extract accurate center using raw point cloud context
            center_coordinates_square, points_square = extract_raw_center_gt_from_patch(
                raw_point_cloud=cur_raw_pc_loaded[1],
                patch_points=point_cloud,
                label_value=label_value,
                method="least_square",
            )

            
            gt_channels = np.zeros(
                (patch_height, patch_width, 3), dtype=np.float32
            )
            if points_square is not None:
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
                    # Calculate continuous (float) pixel coordinates of the true 3D center relative to this patch
                    center_pixel_x = (cur_center[0] - xstart) / resolution
                    center_pixel_y = (cur_center[1] - ystart) / resolution

                    # Define maximum influence radius based on your largest sigma (sigma=60 -> ~3*sigma = 180 pixels = 1.8m)
                    max_sigma = 60
                    margin = 3 * max_sigma  # 180 pixels buffer zone around the patch

                    # Check if the center is close enough to affect this patch at all
                    if (-margin <= center_pixel_x < patch_width + margin) and \
                    (-margin <= center_pixel_y < patch_height + margin):

                        # Generate heatmaps using floating-point center coordinates directly
                        # (Ensure your generate_gaussian_heatmap supports float centers, or uses meshgrid)
                        heatmap_10 = generate_gaussian_heatmap(
                            (patch_height, patch_width), (center_pixel_y, center_pixel_x), sigma=10
                        )
                        heatmap_20 = generate_gaussian_heatmap(
                            (patch_height, patch_width), (center_pixel_y, center_pixel_x), sigma=20
                        )
                        heatmap_60 = generate_gaussian_heatmap(
                            (patch_height, patch_width), (center_pixel_y, center_pixel_x), sigma=60
                        )

                        gt_channels[:, :, 0] = np.maximum(gt_channels[:, :, 0], heatmap_10)
                        gt_channels[:, :, 1] = np.maximum(gt_channels[:, :, 1], heatmap_20)
                        gt_channels[:, :, 2] = np.maximum(gt_channels[:, :, 2], heatmap_60)

                    # compute 3D Heatmap
                    if generate_3d_heatmaps:
                        center_pixel_z = (cur_center[2] - z_min) / z_resolution

                        for i, s in enumerate([10, 20, 60]):
                            heatmap_3d = generate_gaussian_heatmap_3d(
                                (depth_size, patch_height, patch_width), 
                                (center_pixel_z, center_pixel_y, center_pixel_x), 
                                sigma=(s, s, s)
                            )
                            gt_volume_3d[:, :, :, i] = np.maximum(gt_volume_3d[:, :, :, i], heatmap_3d)

                # point_heatmaps = get_heatmap_values_for_points(point_cloud, gt_volume_3d,  z_min, xstart, ystart, resolution)
                # point_cloud_with_heatmap = np.hstack([point_cloud, point_heatmaps.reshape(-1, 1)])

            # save as .npy (Dataset_PCID_X_Y.npy)
            file_name = f"{dataset_name}_{cur_pc_id}_{xstart}_{ystart}.npy"
            save_file_path = os.path.join(output_dir, file_name)
            np.save(save_file_path, gt_channels)

            if generate_3d_heatmaps:
                np.save(save_file_path.replace(".npy", "_3d.npy"), gt_volume_3d)

    print("Finished succefully! TheGT Heatmaps are now pixel-precise synchron.\n   -> Saved to ", output_dir)






