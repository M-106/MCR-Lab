# -----------
# > Imports <
# -----------
import shutil
import os

import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import torch
import open3d as o3d
import scipy.ndimage
import scipy
from sklearn.cluster import DBSCAN

from tqdm import tqdm

from mcrlab.point_cloud.data import get_data_loader, get_filtered_raw_transform, get_basic_transform
from mcrlab.point_cloud.inspect import print_pc, visualize
from mcrlab.point_cloud.utils import get_coordinate_attribute, \
                                     get_class_attribute
from mcrlab.traditional.shape_fit import use_label_candidates_and_extract_center_point, \
                                      use_points_and_extract_center_point
from mcrlab.traditional.utils import visualize_circle_fit
from mcrlab.execution.tryout import center_estimation_3d_pipeline_debugging
# from mcrlab.helper import save_dir_creation



# ------------------
# > Execution Code <
# ------------------
def center_gt_debugging_plot(
        point_cloud, 
        points_square, 
        final_valid_centers,  # Updated parameter name to emphasize they passed validation
        final_valid_centers_patches,
        label_value, 
        cur_pc_id,
        dataset_name,
        config,
        plot_save_dir,
        all_original_centers, # New Parameter: List of centers before validation
        center_validation_status, # New Parameter: List of booleans (True=Valid, False=Invalid)
    ): 
    """
    Publication-quality 3-panel plot for thesis, analyzing local clusters, 
    multi-algo fusion, and global context with strict validation checking.
    """
    # Modern style setup
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # fig, axes = plt.subplots(1, 3, figsize=(16, 10), dpi=300)

    layout = """
    ABC
    ABD
    """

    fig, axes = plt.subplot_mosaic(layout, figsize=(20, 10), dpi=300)

    pc_positions = point_cloud.point[get_coordinate_attribute(point_cloud)].numpy()
    pc_labels = np.squeeze(point_cloud.point["classes"].numpy()) if "classes" in point_cloud.point else None

    # Use consistent color map for cluster points
    cmap = plt.get_cmap('tab20b', max(2, len(points_square)))

    # --- Panel 1: Local Patch & Cluster Breakdown ---
    if pc_labels is not None:
        mask_lbl = (pc_labels == label_value)
        axes['A'].scatter(pc_positions[~mask_lbl, 0], pc_positions[~mask_lbl, 1], c='gray', s=1, alpha=0.1, label="Background")
        # axes['A'].scatter(pc_positions[mask_lbl, 0], pc_positions[mask_lbl, 1], c='red', s=5, label="Raw Manhole Points")
    else:
        axes['A'].scatter(pc_positions[:, 0], pc_positions[:, 1], c='blue', s=1, label="Patch Points")

    for manhole_idx in range(len(points_square)):
        cluster_pts = points_square[manhole_idx]
        axes['A'].scatter(cluster_pts[:, 0], cluster_pts[:, 1], color=cmap(manhole_idx), s=50, alpha=1.0, # edgecolors='red', # linewidth=0.2,
                        label=f"Cluster {manhole_idx+1}" if manhole_idx < 3 else None)
        
    # Standard modern formatting
    axes['A'].set_title(f"Detected Manholes", fontsize=11, fontweight='bold')
    axes['A'].set_xlabel("X (m)", fontsize=9); axes['A'].set_ylabel("Y (m)", fontsize=9)
    axes['A'].tick_params(labelsize=8); axes['A'].grid(True, linestyle=':', alpha=0.6)
    axes['A'].set_aspect('equal')

    # --- Panel 2: Multi-Algo Fusion & Patch Validation ---
    # mask_lbl = (patch_labels == label_value)
    # axes['B'].scatter(pc_positions[~mask_lbl, 0], pc_positions[~mask_lbl, 1], c='gray', s=1, alpha=0.05, label="Raw Cloud Context")
    for manhole_idx in range(len(points_square)):
        cluster_pts = points_square[manhole_idx]
        final_c = all_original_centers[manhole_idx]
        is_valid = center_validation_status[manhole_idx]

        axes['B'].scatter(cluster_pts[:, 0], cluster_pts[:, 1], color=cmap(manhole_idx), s=180, alpha=1.0)
        
        # Determine color/shape based on validity
        if is_valid:
            c_color = '#2ca02c' # Green
            c_label = "Valid GT Center" if manhole_idx == 0 else None
            axes['B'].scatter(final_c[0], final_c[1], c=c_color, marker='X', s=120, edgecolors='black', label=c_label)
        else:
            c_color = '#d62728' # Red
            c_label = "Invalid Center (Rejected)" if manhole_idx == 0 else None
            axes['B'].scatter(final_c[0], final_c[1], c=c_color, marker='x', s=100, label=c_label)

    # MODERN ADDITION: Show the patch boundaries in Panel 2
    patch_colors = [
        '#1f77b4', # (Muted Blue / Matplotlib Blue)
        '#ff7f0e', # (Muted Orange)
        "#9367bda6", # (Muted Purple)
        '#e377c2', # (Pink)
        '#17becf', # (Cyan / Light Blue)
        "#8c564bc0", # (Brown)
        '#ffd700', # (Gold / Yellow)
        "#9d30eb", # (Indigo / Dark Purple)
        "#e97269", # (Iron Reds)
        "#caf777"  # (Light Green)
    ]
    # patch_colors = [
    #     '#ff0000', # (Pure Red)
    #     '#8b0000', # (Dark Red)
    #     '#b22222', # (Firebrick Red)
    #     '#dc143c', # (Crimson)
    #     '#cd5c5c', # (Indian Red / Muted Red)
    #     '#fa8072', # (Salmon / Light Red)
    #     '#ff6347', # (Tomato Red / Orange-Red)
    #     '#800000', # (Maroon / Deep Red)
    #     '#e34234', # (Cinnabar / Vermilion)
    #     '#cb4154'  # (Brick Red)
    # ]
    for cur_patches in final_valid_centers_patches:
        for i_, cur_patch in enumerate(cur_patches):
            xstart = cur_patch["meta"]["origin_x"].item()
            ystart = cur_patch["meta"]["origin_y"].item()
            tile_rect = patches.Rectangle((xstart, ystart), 5.0, 5.0, linewidth=1.5, edgecolor=patch_colors[i_], facecolor='none', linestyle='--', label="Patch Bounds (Valid Area)")
            axes['B'].add_patch(tile_rect)

    axes['B'].set_title(f"Verification of Center Inclusion", fontsize=11, fontweight='bold')
    axes['B'].set_xlabel("X (m)", fontsize=9); axes['B'].set_ylabel("Y (m)", fontsize=9)
    axes['B'].tick_params(labelsize=8); axes['B'].grid(True, linestyle=':', alpha=0.6)
    axes['B'].set_aspect('equal')

    # --- Panel 3: Sample Zoomed in ---
    for plot_i, plot_name in enumerate(['C', 'D']):
        for manhole_idx in range(plot_i, len(points_square)):
            cluster_pts = points_square[manhole_idx]
            final_c = all_original_centers[manhole_idx]
            is_valid = center_validation_status[manhole_idx]

            axes[plot_name].scatter(cluster_pts[:, 0], cluster_pts[:, 1], color=cmap(manhole_idx), s=25*(len(final_valid_centers_patches[plot_i])*2), alpha=1.0)
            
            # Determine color/shape based on validity
            if is_valid:
                c_color = '#2ca02c' # Green
                c_label = "Valid GT Center" if manhole_idx == 0 else None
                axes[plot_name].scatter(final_c[0], final_c[1], c=c_color, marker='X', s=120, edgecolors='black', label=c_label)
            else:
                c_color = '#d62728' # Red
                c_label = "Invalid Center (Rejected)" if manhole_idx == 0 else None
                axes[plot_name].scatter(final_c[0], final_c[1], c=c_color, marker='x', s=100, label=c_label)

            break  # only plot one sample

        for cur_patches in final_valid_centers_patches[plot_i:]:
            for i_, cur_patch in enumerate(cur_patches):
                xstart = cur_patch["meta"]["origin_x"].item()
                ystart = cur_patch["meta"]["origin_y"].item()
                tile_rect = patches.Rectangle((xstart, ystart), 5.0, 5.0, linewidth=1.5, edgecolor=patch_colors[i_], facecolor='none', linestyle='--', label="Patch Bounds (Valid Area)")
                axes[plot_name].add_patch(tile_rect)

            break  # plot only one sample

        axes[plot_name].set_title(f"Zoomed Sample", fontsize=11, fontweight='bold')
        axes[plot_name].set_xlabel("X (m)", fontsize=9); axes[plot_name].set_ylabel("Y (m)", fontsize=9)
        axes[plot_name].tick_params(labelsize=8); axes[plot_name].grid(True, linestyle=':', alpha=0.6)
        axes[plot_name].set_aspect('equal')

    # Consolidated, modern legend layout
    # handles, labels = axes['B'].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper right", ncol=3, bbox_to_anchor=(0.9, 1.05), fontsize=9, frameon=True, shadow=True)
    legend_elements = [
        # Gray point
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Raw background point'),

        # Green cross
        Line2D([0], [0], marker='x', color='w', markeredgecolor='green', markersize=8, markeredgewidth=2, label='Validated'),
        
        # Red point
        # Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Raw manhole'),

        # # Point with red border (Manhole)
        # Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='red', markersize=8, markeredgewidth=2, label='Manhole'),
        
        # Komplett farbiger Punkt (Manhole)
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#dc143c', markeredgecolor='none', markersize=8, label='Manhole'),

        # Red cross
        Line2D([0], [0], marker='x', color='w', markeredgecolor='red', markersize=8, markeredgewidth=2, label='Invalid (no patch found)'),

        # Dashed box (using a Patch with a dashed edge)
        Patch(facecolor='none', edgecolor='black', linestyle='--', linewidth=1.5, label='Patch data'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', ncol=3, bbox_to_anchor=(0.96, 1.0), fontsize=9, frameon=True, shadow=True)

    fig.suptitle("Center GT Extraction & Validation", y=1.03, fontsize=18, fontweight='bold')
    fig.text(0.5, 0.99, f"Dataset-Name: {dataset_name.upper()}, PC-ID: {cur_pc_id}, Algo: {config.gt_extraction.center_algorithm}", 
            ha='center', fontsize=12, color='dimgray')

    # Make sure on top is enough space
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # oder alternativ: plt.subplots_adjust(top=0.9)

    # plt.tight_layout()
    
    # Save cleanly
    fig_path = os.path.join(plot_save_dir, f"{cur_pc_id}_center_debug_validated.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → Saved thesis debugging plot at '{fig_path}'", flush=True)


def validate_centers_against_patches(extracted_centers, data_name, data_path, pc_id, tile_size=5.0, tolerance=0.01):
    """
    Checks, if every extracted center exists inside at least one patch.

    Can return multiple patches for one center point.

    Returns:
    - Valid-Centers: [[1.2, 1.8, 3.4], [-4.0, 0.23, 42.0], ...]
    - Valid-Center-Patches: [[dict, dict, ...], [], ...]
    """
    # 1. Load Patch Data
    data_loader = get_data_loader(data_name, 
                                  data_path, 
                                  type="all",
                                  transform=get_basic_transform(),  # get_basic_transform(num_points=-1),
                                  batch_size=1, shuffle=False, num_workers=0,
                                  preprocessed=True, 
                                  return_train_format=False, 
                                  bev_normalized=True,
                                  bev_normalize_mode="global_standard",
                                  only_these_pc_ids=[pc_id])

    # to get BEV, we can use the paths and create a BEV Dataset or using the Point Cloud Wrapper Class

    # 2. Check if centers are in at least one patch
    tile_size += tolerance
    valid_centers = []
    valid_centers_patches = []
    for center in extracted_centers:
        is_covered = False
        cur_center_patches = []

        # Iterate over all patches and compute with the coordinates if the center is physically inside the patch
        for batch in data_loader:
            # get bev
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to_numpy(as_copy=True).get_bev()
            batch = next(iter(batch))

            # Prüfe, ob Center im Bereich [xstart, xstart+tile_size] und [ystart, ystart+tile_size] liegt
            xstart = batch["meta"]["origin_x"].item() - tolerance
            ystart = batch["meta"]["origin_y"].item() - tolerance
            
            if (xstart <= center[0] < xstart + tile_size) and (ystart <= center[1] < ystart + tile_size):
                is_covered = True
                cur_center_patches.append(batch)
                # break
                
        if is_covered:
            valid_centers.append(center)
            valid_centers_patches.append(cur_center_patches)
        else:
            print(f"[WARNING] Center {center} does not exist in any patch!")
            
    return valid_centers, valid_centers_patches


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

    for cur_idx in range(len(config.gt_extraction.data_paths)):
        json_data = list()

        cur_dataset = config.gt_extraction.names[cur_idx]

        # Create output directory for image plots
        plot_save_dir = os.path.join(
            config.gt_extraction.save_path,   # "/out/center_gt_extraction",
            "center_gt/plots", 
            f"{cur_dataset}_{config.gt_extraction.center_algorithm}"
        )

        # save_dir_creation(plot_save_dir)
        os.makedirs(plot_save_dir, exist_ok=True)
        shutil.rmtree(plot_save_dir)
        # save_dir_creation(plot_save_dir)
        os.makedirs(plot_save_dir, exist_ok=True)

        data_loader = get_data_loader(config.gt_extraction.names[cur_idx], 
                                      config.gt_extraction.data_paths[cur_idx], 
                                      type="test",   # config.gt_extraction.type, 
                                      transform=get_filtered_raw_transform(manhole_label=3 if "sud" in cur_dataset.lower() else 104002),  # get_basic_transform(num_points=-1),
                                      batch_size=1, shuffle=False, num_workers=0,
                                      preprocessed=False, 
                                      return_train_format=False, 
                                      bev_normalized=False,  # config.data.normalization, 
                                      bev_normalize_mode=config.data.normalization_mode,
                                      invert_z=True)  # FIXME, works with that?? Must invert for SUD

        point_cloud_paths = data_loader.dataset.point_cloud_paths

        for idx, batch in enumerate(data_loader):
            point_cloud = batch[0]
            # point_cloud = point_cloud.get_as_o3d()
            # print_pc(point_cloud)

            _, cur_pc_name = os.path.split(point_cloud_paths[idx])
            cur_pc_name = ".".join(cur_pc_name.split(".")[:-1])
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
                center_gt_debugging_plot(
                    point_cloud=point_cloud, 
                    points_square=np.array([]), 
                    final_valid_centers=[], 
                    final_valid_centers_patches=[],
                    label_value=label_value, 
                    cur_pc_id=cur_pc_id,
                    dataset_name=cur_dataset,
                    config=config,
                    plot_save_dir=plot_save_dir,
                    all_original_centers=[],
                    center_validation_status=[]
                )
                continue

            print("\n> Least Square Circle Fit Check <\n")
            center_coordinates_square, radius_squares, points_square, cluster_point_clouds, _, error, _ = center_estimation_3d_pipeline_debugging(None, method="least_square", extended_return=True, should_visualize=False, clusters=original_cluster_pcs, label_value=label_value)
            center_coordinates_ransac, _, _, _, _, _, _ = center_estimation_3d_pipeline_debugging(None, method="ransac", extended_return=True, should_visualize=False, clusters=original_cluster_pcs, label_value=label_value)

            extracted_centers = []
            extracted_centers_patches = []
            all_original_centers = []
            center_validation_status = []

            for cur_manhole_idx in range(len(points_square)):
                # cur_points = points_square[cur_manhole_idx]
                if config.gt_extraction.center_algorithm == "squares":
                    cur_center = center_coordinates_square[cur_manhole_idx]
                elif config.gt_extraction.center_algorithm == "mean":
                    points_ = points_square[cur_manhole_idx]
                    cur_center = np.array([np.mean(points_[:, 0]), np.mean(points_[:, 1]), np.mean(points_[:, 2])])
                elif config.gt_extraction.center_algorithm == "ransac":
                    cur_center = center_coordinates_ransac[cur_manhole_idx]
                elif config.gt_extraction.center_algorithm == "mesqra":
                    cur_center_squares = center_coordinates_square[cur_manhole_idx]
                    
                    cur_center_ransac = center_coordinates_ransac[cur_manhole_idx]
                    
                    points_ = points_square[cur_manhole_idx]
                    cur_center_mean = np.array([np.mean(points_[:, 0]), np.mean(points_[:, 1]), np.mean(points_[:, 2])])
                
                    # cur_center = np.mean(np.concatenate([cur_center_squares, cur_center_ransac, cur_center_mean], axis=0).reshape(3, -1), axis=0)
                    cur_center = np.mean(np.stack([cur_center_squares, cur_center_ransac, cur_center_mean], axis=0), axis=0)
                else:
                    raise ValueError(f"Unknown center extraction algorith: '{config.gt_extraction.center_algorithm}'")

                all_original_centers.append(cur_center)

                # Post valid check 
                valid_centers, valid_centers_patches = validate_centers_against_patches(
                    extracted_centers=[cur_center], 
                    data_name=config.gt_extraction.names[cur_idx], 
                    data_path=config.gt_extraction.data_paths[cur_idx],
                    pc_id=cur_pc_id, 
                    tile_size=5.0, tolerance=0.01
                )

                is_valid = len(valid_centers) > 0
                center_validation_status.append(is_valid)
                
                if is_valid:
                    if len(valid_centers) > 1:
                        raise ValueError("Unexpected amount of centers! Expected only one or none center after validation.")
                    valid_center = valid_centers[0]
                    valid_center_patches = valid_centers_patches[0]

                    extracted_centers.append(valid_center)
                    extracted_centers_patches.append(valid_center_patches)

                    cur_point = {
                        "x": float(valid_center[0]), 
                        "y": float(valid_center[1]), 
                        "z": float(valid_center[2])
                    }
                
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
                # extract patch info
                # patch_meta = extracted_centers_patches[0]["meta"] if len(extracted_centers_patches) > 0 else {"origin_x": 0.0, "origin_y": 0.0}
                # xstart_val = patch_meta["origin_x"].item() if hasattr(patch_meta["origin_x"], "item") else patch_meta["origin_x"]
                # ystart_val = patch_meta["origin_y"].item() if hasattr(patch_meta["origin_y"], "item") else patch_meta["origin_y"]

                # debugging plot
                center_gt_debugging_plot(
                    point_cloud=point_cloud, 
                    points_square=points_square, 
                    final_valid_centers=extracted_centers, 
                    final_valid_centers_patches=extracted_centers_patches,
                    label_value=label_value, 
                    cur_pc_id=cur_pc_id,
                    dataset_name=cur_dataset,
                    config=config,
                    plot_save_dir=plot_save_dir,
                    all_original_centers=all_original_centers,
                    center_validation_status=center_validation_status
                )

                # prediction plot
                for manhole_idx in range(len(extracted_centers)):
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

                    plt.suptitle(f"Dataset: {cur_dataset} | ID: {cur_pc_id} | Algo: {config.gt_extraction.center_algorithm}")
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

        save_path = os.path.join(config.gt_extraction.save_path, "center_gt", f"{cur_dataset}_eval_ground_truths_{config.gt_extraction.center_algorithm}.json")
    
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

def debugging_center_extraction(patch_points, patch_centers, refined_centers, raw_point_cloud, label_value, search_radius, cur_pc_id, xstart, ystart, dataset_name):
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    # Plot 1: Patch points and the fallback/cluster center
    patch_positions = patch_points.point[get_coordinate_attribute(patch_points)].numpy()
    patch_labels = np.squeeze(patch_points.point["classes"].numpy()) if "classes" in patch_points.point else None

    if patch_labels is not None:
        mask_lbl = (patch_labels == label_value)
        axes[0].scatter(patch_positions[~mask_lbl, 0], patch_positions[~mask_lbl, 1], c='gray', s=1, alpha=0.3, label="Background")
        axes[0].scatter(patch_positions[mask_lbl, 0], patch_positions[mask_lbl, 1], c='red', s=5, label="Labeled Manhole/Fragment")
    else:
        axes[0].scatter(patch_positions[:, 0], patch_positions[:, 1], c='blue', s=1, label="Patch Points")

    for ac in patch_centers:
        axes[0].scatter(ac[0], ac[1], c='yellow', marker='X', s=70, edgecolors='black', label="Approx Center")

    axes[0].set_title("Local Patch & Fragment Center")
    axes[0].set_aspect('equal')
    # axes[0].legend(loc='upper right', fontsize=8)

    # Plot 2: Raw point cloud query box
    raw_positions = raw_point_cloud.point[get_coordinate_attribute(raw_point_cloud)].numpy()
    raw_labels = np.squeeze(raw_point_cloud.point["classes"].numpy()) if "classes" in raw_point_cloud.point else None
    # print("(In Debugging) label_value =", label_value)
    # print("(In Debugging) Plot raw labels unique:", np.unique(raw_labels))
    # print("(In Debugging) count label=1:", np.sum(raw_labels == 1))
    # print("(In Debugging) count label=255:", np.sum(raw_labels == 255))
    for counter, ac in enumerate(patch_centers):
        if counter == 0:
            label_bg = "Raw Query Box"
            label_mn = "Raw Full Labeled Fragment"
            label_ac = "Approx. Center from Patch"
        else:
            label_bg = None
            label_mn = None
            label_ac = None

        in_box = (
            (np.abs(raw_positions[:, 0] - ac[0]) <= search_radius) &
            (np.abs(raw_positions[:, 1] - ac[1]) <= search_radius)
        )
        box_pts = raw_positions[in_box]
        
        # axes[1].scatter(box_pts[:, 0], box_pts[:, 1], c='cyan', s=2, label="Raw Query Box")
        if raw_labels is not None:
            box_labels = raw_labels[in_box]
            mask_lbl = (box_labels == label_value)
            axes[1].scatter(box_pts[~mask_lbl, 0], box_pts[~mask_lbl, 1], c='gray', s=1, alpha=0.3, label=label_bg)
            axes[1].scatter(box_pts[mask_lbl, 0], box_pts[mask_lbl, 1], c='red', s=5, label=label_mn)
        else:
            axes[1].scatter(box_pts[:, 0], box_pts[:, 1], c='blue', s=1, label="Patch Points")
        axes[1].scatter(ac[0], ac[1], c='yellow', marker='X', s=70, edgecolors='black', label=label_ac)
    for counter, c in enumerate(refined_centers):
        axes[1].scatter(c[0], c[1], c='green', marker='X', s=70, edgecolors='black', label="Refined Center" if counter == 0 else None)

    axes[1].set_title(f"Raw Points in Search Box (Radius={search_radius}m)")
    axes[1].set_aspect('equal')
    # axes[1].legend(loc='upper right', fontsize=8)

    # Plot 3: Position in Raw Point Cloud
    raw_positions = raw_point_cloud.point[get_coordinate_attribute(raw_point_cloud)].numpy()
    raw_labels = np.squeeze(raw_point_cloud.point["classes"].numpy()) if "classes" in raw_point_cloud.point else None
    for ac in patch_centers:
        if raw_labels is not None:
            mask_lbl = (raw_labels == label_value)
            axes[2].scatter(raw_positions[~mask_lbl, 0], raw_positions[~mask_lbl, 1], c='gray', s=1, alpha=0.3, label="Raw Query Box")
            axes[2].scatter(raw_positions[mask_lbl, 0], raw_positions[mask_lbl, 1], c='red', s=5, label="Raw Full Labeled Manholes")
        else:
            axes[2].scatter(raw_positions[:, 0], raw_positions[:, 1], c='blue', s=1, label="Patch Points")
        axes[2].scatter(ac[0], ac[1], c='yellow', marker='X', s=20, edgecolors='black')
    for counter, c in enumerate(refined_centers):
            axes[1].scatter(c[0], c[1], c='green', marker='X', s=70, edgecolors='black')

    axes[2].set_title(f"Complete Raw Point Cloud")
    axes[2].set_aspect('equal')
    # axes[2].legend(loc='upper right', fontsize=8)

    # legend
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=2) # bbox_to_anchor=(1.02, 1.0),

    plt.savefig(f"./output/heatmap_extraction_debugging/hm_creation_{dataset_name}_{cur_pc_id}_{xstart}_{ystart}.png", dpi=300)
    plt.close(fig)



def extract_raw_center_gt_from_patch(
    raw_point_cloud,          # Pass the loaded raw point cloud directly!
    patch_points,             # The preprocessed 5x5m patch point cloud
    label_value: int,
    method: str = "least_square",
    search_radius: float = 1.2, # Radius in meters to query around the patch cluster center
    cur_pc_id=None, xstart=None, ystart=None, dataset_name=""
):
    # Step 1: Detect clusters on the local patch
    _, _, _, original_cluster_pcs, _, _, _ = center_estimation_3d_pipeline_debugging(
        patch_points,
        method=method,
        extended_return=True,
        should_visualize=False,
        label_value=label_value,
    )

    patch_centers = []

    if original_cluster_pcs is not None and len(original_cluster_pcs) > 0:
        # Get preliminary patch center estimates from valid clusters
        patch_centers, _, _, _, _, _, _ = center_estimation_3d_pipeline_debugging(
            None,
            method=method,
            extended_return=True,
            should_visualize=False,
            clusters=original_cluster_pcs,
            label_value=label_value,
        )
    else:
        # FALLBACK: If clustering failed because only a tiny fragment is visible,
        # find patch points matching the label and use their centroid as a fallback center!
        patch_positions = patch_points.point[get_coordinate_attribute(patch_points)].numpy()
        patch_labels = np.squeeze(patch_points.point["classes"].numpy()) if "classes" in patch_points.point else None
        
        if patch_labels is not None:
            mask = (patch_labels == label_value)
            fragment_points = patch_positions[mask]
        else:
            print("[WARNING] No Labels from Patch Point Cloud founded in 'extract_raw_center_gt_from_patch'-function.")
            fragment_points = patch_positions  # fallback if no labels attribute

        if len(fragment_points) > 0:
            # Use the center of the visible fragment as our approximate search anchor
            fallback_center = np.mean(fragment_points, axis=0)
            patch_centers = [fallback_center]

    if len(patch_centers) == 0:
            return None, None

    n_approx_center = len(patch_centers)

    # Step 2: Query the RAW point cloud around each patch center estimate

    raw_positions = raw_point_cloud.point[get_coordinate_attribute(raw_point_cloud)].numpy()
    raw_labels = np.squeeze(raw_point_cloud.point["classes"].numpy()) if "classes" in raw_point_cloud.point else None

    # print("Unique raw labels:", np.unique(raw_point_cloud.point["classes"].numpy()))
    # print("Count label=1:", np.sum(raw_labels == 1))
    # print("Count label=255:", np.sum(raw_labels == 255))
    # print("Min:", np.min(raw_point_cloud.point["classes"].numpy()))
    # print("Max:", np.max(raw_point_cloud.point["classes"].numpy()))

    if raw_labels is None:
        print("[WARNING] No Labels from Raw Point Cloud founded in 'extract_raw_center_gt_from_patch'-function.")

    raw_clusters = []
    direct_centers = []  # Fallback list for clusters with too few points for least squares

    for approx_center in patch_centers:
        
        # Quick 3D Bounding Box Crop (2m x 2m x 2m cube around center)
        # else memory error, because full point cloud neighbor computation
        in_box_mask = (
            (np.abs(raw_positions[:, 0] - approx_center[0]) <= search_radius) &
            (np.abs(raw_positions[:, 1] - approx_center[1]) <= search_radius) # &
            # (np.abs(raw_positions[:, 2] - approx_center[2]) <= search_radius)
            # must ignore Z-value because SUD raw is inverted, so you can invert the z value or ignore it
            # though we only have ground-extracted point clouds, ignoring Z is fine
        )
        # total_points_in_box = np.sum(in_box_mask)

        # Filter candidate labels if available
        in_box_mask = in_box_mask & (raw_labels == label_value)

        # Refine candidate labels (in_box_mask) -> neighbor removal
        candidate_points = raw_positions[in_box_mask]
        
        if len(candidate_points) > 0:
            # find objects via clustering
            # eps=0.04 (4cm) handles 0.01m resolution and separates manholes sitting close together
            clustering = DBSCAN(eps=0.3, min_samples=20).fit(candidate_points)
            labels = clustering.labels_
            unique_labels = set(labels) - {-1} # Exclude noise (-1)
            
            if len(unique_labels) > 0:
                best_cluster_label = -1
                min_dist = float('inf')
                
                # find right manhole -> via distance to cur patch center
                for lbl in unique_labels:
                    cluster_pts = candidate_points[labels == lbl]
                    cluster_center = np.mean(cluster_pts, axis=0)
                    dist = np.linalg.norm(cluster_center - approx_center)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster_label = lbl
                
                # apply removal/filter (keep only the closest valid manhole cluster)
                if best_cluster_label != -1:
                    full_cluster_points = candidate_points[labels == best_cluster_label]
                else:
                    full_cluster_points = np.empty((0, 3))
            else:
                full_cluster_points = np.empty((0, 3))
        else:
            full_cluster_points = np.empty((0, 3))


        # # Precise Radius Filter on the candidate subset
        # dists_3d = np.linalg.norm(candidate_points - approx_center, axis=1)
        # full_cluster_points = candidate_points[dists_3d <= search_radius]

        if len(full_cluster_points) > 0:
            if len(full_cluster_points) >= 3:
                # Enough points for high-precision circle fitting
                # Create Open3D Tensor PointCloud to match raw_point_cloud structure
                pcd = o3d.t.geometry.PointCloud()
                pcd.point["positions"] = o3d.core.Tensor(full_cluster_points)
                raw_clusters.append(pcd)
            else:
                # Too few points for least-squares, use mean directly
                direct_centers.append(np.mean(full_cluster_points, axis=0))

    # Step 3: Re-estimate high-precision centers on the un-cropped raw clusters
    refined_centers_list = []
    refined_points_list = []

    if len(raw_clusters) > 0:
        refined_centers, _, refined_points, _, _, _, _ = center_estimation_3d_pipeline_debugging(
            None,
            method=method,
            extended_return=True,
            should_visualize=False,
            clusters=raw_clusters,
            label_value=label_value,
        )
        if refined_centers is not None and len(refined_centers) > 0:
            refined_centers_list.append(refined_centers)
            refined_points_list.append(refined_points)
        else:
            # FALLBACK, try again with mean
            refined_centers, _, refined_points, _, _, _, _ = center_estimation_3d_pipeline_debugging(
                None,
                method="mean",
                extended_return=True,
                should_visualize=False,
                clusters=raw_clusters,
                label_value=label_value,
            )
            if refined_centers is not None and len(refined_centers) > 0:
                refined_centers_list.append(refined_centers)
                refined_points_list.append(refined_points)

    # Append any direct mean-based centers from extremely sparse fragments
    if len(direct_centers) > 0:
        direct_centers_arr = np.array(direct_centers, dtype=np.float32)
        refined_centers_list.append(direct_centers_arr)
        # Store the points themselves as surrogate points for visualization/processing
        for dc in direct_centers:
            refined_points_list.append(raw_positions[np.linalg.norm(raw_positions - dc, axis=1) <= search_radius])

    if len(refined_centers_list) > 0:
        final_centers = np.concatenate(refined_centers_list, axis=0)
    else:
        final_centers = []

    n_final_center = len(final_centers)

    # # Fallback, if refinement made things worse
    # validated_final_centers = []
    # if n_approx_center > n_final_center:
    #     print(f"[WARNING] Loses {n_approx_center-n_final_center} Centers during refinement.")
    #     for cur_approx_center in patch_centers:
    #         have_a_refined_version = False
    #         for cur_refine_center in refined_centers_list:
    #             if abs(cur_refine_center[0]-cur_approx_center[0]) + abs(cur_refine_center[1]-cur_approx_center[1]) < 0.6:
    #                 have_a_refined_version = True
    #                 validated_final_centers.append(cur_refine_center)
    #                 break

    #         if not have_a_refined_version:
    #             validated_final_centers.append(cur_approx_center)

    # final_centers = np.concatenate(validated_final_centers, axis=0)

    # debugging
    debugging_center_extraction(patch_points, patch_centers, final_centers, raw_point_cloud, label_value, search_radius, cur_pc_id, xstart, ystart, dataset_name)

    if len(refined_centers_list) == 0:
        return None, None

    # Handle list of arrays or concatenate depending on structure
    final_points = refined_points_list[0] if len(refined_points_list) == 1 else refined_points_list

    return final_centers, final_points



def ground_truth_extraction_heatmap(config):
    print("\n --- Aligned Heatmap Center Ground Truth Extraction ---")

    output_dir = os.path.join(
        os.path.dirname(config.gt_extraction.save_path), "2d_gt_patches_on_raw_pc"
    )
    os.makedirs(output_dir, exist_ok=True)
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    os.makedirs(f"./output/heatmap_extraction_debugging", exist_ok=True)
    shutil.rmtree(f"./output/heatmap_extraction_debugging")
    os.makedirs(f"./output/heatmap_extraction_debugging", exist_ok=True)

    generate_3d_heatmaps = config.gt_extraction.generate_also_3d_gt_maps

    label_value = 1

    # using exactly the same value as in projection
    tile_size = 5.0  # exactly like bev_tile_size
    resolution = 0.01  # exactly like bev_resolution

    # Exactly the dimension in projection
    patch_height = int(tile_size / resolution)  # 5.0 / 0.01 = 500
    patch_width = int(tile_size / resolution)  # 5.0 / 0.01 = 500

    cur_raw_pc_loaded = None

    for cur_idx in range(len(config.gt_extraction.data_paths)):
        dataset_name = config.gt_extraction.names[cur_idx]

        # load PATCH data (preprocessed data)
        data_loader = get_data_loader(
            dataset_name,
            config.gt_extraction.data_paths[cur_idx],
            type="all",  # config.gt_extraction.type,
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
                    config.gt_extraction.data_paths[cur_idx],
                    type="all",  # config.gt_extraction.type,
                    transform=get_filtered_raw_transform(manhole_label=3 if "sud" in dataset_name.lower() else 104002),
                    batch_size=1,
                    shuffle=False,
                    preprocessed=False, # Fetch raw full cloud
                    return_train_format=False,
                    invert_z=True
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
                method="least_square",  # FIXME: least_square, mean
                search_radius=1.0,
                cur_pc_id=cur_pc_id, 
                xstart=xstart, 
                ystart=ystart,
                dataset_name=dataset_name
            )

            
            gt_channels = np.zeros(
                (patch_height, patch_width, 3), dtype=np.float32
            )
            if points_square is not None:
                for cur_manhole_idx in range(len(points_square)):
                    if config.gt_extraction.center_algorithm == "squares":
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
                    margin = 3 * max_sigma * 2  # 180 pixels buffer zone around the patch

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






