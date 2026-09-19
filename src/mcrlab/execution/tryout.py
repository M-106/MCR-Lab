# -----------
# > Imports <
# -----------
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import torch
import open3d as o3d

from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression

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
from mcrlab.projection import bev_projection, bev_projection_testing, bev_back_projection_testing
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
from mcrlab.point_cloud.monte_carlo_simulation import eval_center_robustness, eval_ellipse_precision
# from mcrlab.helper import save_dir_creation



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


# def plot_3d_point_cloud_overview(point_cloud, target_labels=[1, 255], save_path="pc_overview.png"):
#     """
#     1. Full 3D Point Cloud Overview: Renders the entire scene with manholes highlighted.
#     """
#     # Extract coordinates & attributes
#     coords = point_cloud.point["positions"].numpy()
    
#     # Base grayscale color based on intensity
#     if "intensity" in point_cloud.point:
#         intensities = point_cloud.point["intensity"].numpy().flatten()
#         i_min, i_max = intensities.min(), intensities.max()
#         norm_i = (intensities - i_min) / (i_max - i_min + 1e-8) if i_max > i_min else np.zeros_like(intensities)
#         colors = np.tile(norm_i[:, None], (1, 3)) * 0.6  # Dim gray
#     else:
#         colors = np.full((len(coords), 3), 0.5)

#     # Highlight manhole points in Magenta
#     if "classes" in point_cloud.point:
#         semantics = point_cloud.point["classes"].numpy().flatten()
#         mask = np.isin(semantics, target_labels)
#         colors[mask] = [1.0, 0.0, 1.0]

#     legacy_pcd = o3d.geometry.PointCloud()
#     legacy_pcd.points = o3d.utility.Vector3dVector(coords)
#     legacy_pcd.colors = o3d.utility.Vector3dVector(colors)

#     # Render off-screen snapshot or launch viewer
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(visible=True, width=1920, height=1080)
#     vis.add_geometry(legacy_pcd)
    
#     opt = vis.get_render_option()
#     opt.point_size = 2.0
#     opt.background_color = np.array([0.05, 0.05, 0.05])
    
#     print(f"[3D Overview] Displaying point cloud. Press 'Q' or close window to proceed...")
#     vis.run()
#     if save_path:
#         vis.capture_screen_float_buffer(True)
#         vis.capture_screen_image(save_path)
#         print(f"[Saved] Overview image saved to {save_path}")
#     vis.destroy_window()



def plot_3d_point_cloud_overview(
    point_cloud, 
    target_labels=[1, 255], 
    save_path="pc_overview.png"
):
    """1. Full 3D Point Cloud Overview: Renders the entire scene with manholes highlighted using Matplotlib."""
    # Extract coordinates & attributes
    coords = point_cloud.point["positions"].numpy()

    # Base grayscale color based on intensity
    if "intensity" in point_cloud.point:
        intensities = point_cloud.point["intensity"].numpy().flatten()
        i_min, i_max = intensities.min(), intensities.max()
        norm_i = (
            (intensities - i_min) / (i_max - i_min + 1e-8)
            if i_max > i_min
            else np.zeros_like(intensities)
        )
        colors = np.tile(norm_i[:, None], (1, 3)) * 0.6  # Dim gray
    else:
        colors = np.full((len(coords), 3), 0.5)

    # Highlight manhole points in Magenta
    if "classes" in point_cloud.point:
        semantics = point_cloud.point["classes"].numpy().flatten()
        mask = np.isin(semantics, target_labels)
        colors[mask] = [1.0, 0.0, 1.0]

    # Initialize 1920x1080 figure with dark background
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#0d0d0d")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0d0d0d")

    # Scatter plot matching Open3D point size & colors
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=colors,
        s=2.0,  # Match point size
        depthshade=False,  # Keep exact RGB colors without artificial depth shading
    )

    # Clean plot appearance (remove axes and grid for viewer look)
    ax.axis("off")
    ax.set_box_aspect(
        [
            np.ptp(coords[:, 0]),
            np.ptp(coords[:, 1]),
            np.ptp(coords[:, 2]),
        ]
    )  # Keep 1:1 aspect ratio

    # plt.tight_layout()

    # save_dir_creation(os.path.dirname(save_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if save_path:
        plt.savefig(
            save_path,
            bbox_inches="tight",
            pad_inches=0,
            facecolor=fig.get_facecolor(),
        )
        print(f"[Saved] Overview image saved to {save_path}")

    # print(
    #     "[3D Overview] Displaying point cloud. Close the window to proceed..."
    # )
    # plt.show()
    plt.close(fig)



def plot_manhole_3d_crops(point_cloud, target_labels=[1, 255], crop_radius=2.0, max_crops=3, save_dir="./output/pc_viz/manhole_3d_crops"):
    """
    2. Local Manhole 3D Crops: Finds manholes, crops surrounding points, and saves 2D scatter plots.
    """
    # save_dir_creation(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    coords = point_cloud.point["positions"].numpy()
    semantics = point_cloud.point["classes"].numpy().flatten()
    
    # Extract intensity if present
    if "intensity" in point_cloud.point:
        intensities = point_cloud.point["intensity"].numpy().flatten()
    else:
        intensities = np.ones(len(coords))

    manhole_indices = np.where(np.isin(semantics, target_labels))[0]
    if len(manhole_indices) == 0:
        print("[Warning] No manholes found in this point cloud for cropping.")
        return

    # Simple spatial clustering to find unique manhole centers
    manhole_coords = coords[manhole_indices]
    
    # Take up to max_crops manholes
    for idx, center in enumerate(manhole_coords[::max(1, len(manhole_coords) // max_crops)][:max_crops]):
        dists = np.linalg.norm(coords[:, :2] - center[:2], axis=1)
        crop_mask = dists <= crop_radius
        
        crop_pts = coords[crop_mask]
        crop_intensity = intensities[crop_mask]
        crop_sem = semantics[crop_mask]

        fig = plt.figure(figsize=(12, 5))
        
        # 2D Overhead Scatter plot
        ax1 = fig.add_subplot(121)
        sc = ax1.scatter(crop_pts[:, 0] - center[0], crop_pts[:, 1] - center[1], 
                         c=crop_intensity, cmap="gray", s=12)
        plt.colorbar(sc, ax=ax1, label="Intensity")
        ax1.set_title("Local Overhead Intensity")
        ax1.set_xlabel("X Offset (m)")
        ax1.set_ylabel("Y Offset (m)")
        ax1.set_aspect("equal")

        # 2D Semantic Mask plot
        ax2 = fig.add_subplot(122)
        is_mh = np.isin(crop_sem, target_labels)
        ax2.scatter(crop_pts[~is_mh, 0] - center[0], crop_pts[~is_mh, 1] - center[1], c="gray", s=8, label="Background")
        ax2.scatter(crop_pts[is_mh, 0] - center[0], crop_pts[is_mh, 1] - center[1], c="magenta", s=20, label="Manhole")
        ax2.set_title("Semantic Label Segmentation")
        ax2.set_xlabel("X Offset (m)")
        ax2.set_ylabel("Y Offset (m)")
        ax2.legend()
        ax2.set_aspect("equal")

        plt.suptitle(f"Manhole Instance Crop #{idx+1} (Radius: {crop_radius}m)", fontsize=14, fontweight="bold")
        out_path = os.path.join(save_dir, f"manhole_crop_{idx+1}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] Manhole crop plot saved to {out_path}")


def plot_bev_rasterization_panel(bev_item, save_path="bev_rasterization_panel.png"):
    """
    3. BEV Rasterization Breakdown: Creates a multi-channel visualization grid for presentation figures.
    """
    img = bev_item["pixel_values"].detach().cpu().numpy()  # Channels x H x W
    labels = bev_item["labels"].detach().cpu().numpy()      # H x W
    meta = bev_item["meta"]

    # Transpose image channels (C, H, W -> H, W, C)
    img_t = np.transpose(img, (1, 2, 0))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    
    # 1. Max Height Channel
    axes[0, 0].imshow(img_t[:, :, 0], cmap="viridis")
    axes[0, 0].set_title("1. Max Height Channel", fontsize=12, fontweight="bold")

    # 2. Delta Height Channel
    axes[0, 1].imshow(img_t[:, :, 1], cmap="plasma")
    axes[0, 1].set_title("2. Height Delta Channel", fontsize=12, fontweight="bold")

    # 3. Intensity Channel (Normalized)
    i_chan = img_t[:, :, 2]
    i_norm = (i_chan - i_chan.min()) / (i_chan.ptp() + 1e-8) if np.ptp(i_chan) > 0 else i_chan
    axes[1, 0].imshow(i_norm, cmap="gray")
    axes[1, 0].set_title("3. Intensity Channel", fontsize=12, fontweight="bold")

    # 4. Density Channel
    axes[1, 1].imshow(img_t[:, :, 3], cmap="cividis")
    axes[1, 1].set_title("4. Density / Point Count Channel", fontsize=12, fontweight="bold")

    # 5. Semantic Ground Truth Overlay
    cmap_label = mcolors.ListedColormap(["black", "magenta", "cyan"])
    norm_bounds = mcolors.BoundaryNorm([0, 0.5, 1.5, 256], cmap_label.N)
    
    axes[0, 2].imshow(labels, cmap=cmap_label, norm=norm_bounds)
    axes[0, 2].set_title("5. Target Manhole Mask", fontsize=12, fontweight="bold")

    # 6. Composite RGB Preview (Intensity + Mask Overlay)
    rgb_composite = np.stack([i_norm]*3, axis=-1)
    mh_mask = np.isin(labels, [1, 255])
    rgb_composite[mh_mask] = [1.0, 0.0, 1.0]  # Highlight manhole in pink
    axes[1, 2].imshow(rgb_composite)
    axes[1, 2].set_title("6. Composite BEV Patch", fontsize=12, fontweight="bold")

    # Turn off axes and format titles
    for ax in axes.flat:
        ax.axis("off")

    pc_id = meta.get("pc_id", "Unknown")
    plt.suptitle(f"BEV Patch Rasterization Breakdown (Patch ID: {pc_id})", fontsize=16, fontweight="bold", y=0.95)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] Rasterization panel figure saved to {save_path}")



def generate_presentation_plots(config):
    data_loader = get_data_loader(
        config.data.name, 
        config.data.path, 
        type=config.data.type, 
        transform=None,  # get_basic_transform(),
        batch_size=1, shuffle=False, num_workers=0,
        preprocessed=config.data.preprocessed, return_train_format=False
    )

    target_labels = [1, 255] if config.data.preprocessed else ([3] if config.data.name == "sud" else [104002])

    for batch in data_loader:
        point_cloud = batch[0]
        
        # Plot 1: Full 3D Scene Overview
        plot_3d_point_cloud_overview(point_cloud, target_labels=target_labels, save_path="./output/pc_viz/full_scene_3d.png")
        
        # Plot 2: Local 3D Manhole Crops
        plot_manhole_3d_crops(point_cloud, target_labels=target_labels, crop_radius=2.5, max_crops=3)

        # Plot 3: BEV Patch Rasterization Multi-Channel Breakdown
        if point_cloud.bev_data is not None:
            bev_gen = point_cloud.get_bev()
            for idx, bev_item in enumerate(bev_gen):
                if np.any(np.isin(bev_item["labels"].numpy(), target_labels)):
                    plot_bev_rasterization_panel(bev_item, save_path=f"./output/pc_viz/bev_panel_patch_{idx}.png")
                    break  # Stop after showing the first patch containing a manhole
        break



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



def bev_back_preprocessed_loading_working_testing(config):
    if not config.data.preprocessed:
        raise ValueError("'Preprocessing' must be True! (config.data.preprocessed)")

    result = {}

    for cur_dataset in ["whu", "sud"]:
        result[cur_dataset] = []

        test_3d_dataset = get_data_loader(
            cur_dataset, 
            config.data.path if cur_dataset == "whu" else config.data.path_2, 
            type="test",
            transform=get_basic_transform(),
            batch_size=1, 
            shuffle=False, 
            num_workers=4,
            preprocessed=True, 
            return_train_format=True,
            return_dataset=True,
        )
        
        all_test_paths = test_3d_dataset.point_cloud_paths

        test_bev_dataset = BEVDataset(
            path=all_test_paths, 
            file_paths=[], 
            has_labels=True, 
            image_training=True, 
            preprocessor=None,
            augment=False,
            pass_label_in_preprocessor=False,
            heatmap_gt_path=None,
            used_heatmap_channel=False
        )

        for idx, cur_data_path in enumerate(all_test_paths):
            pc_id, x_start, y_start = test_bev_dataset.extract_grid_identifier(cur_data_path)

            bev_dict = next(test_bev_dataset.get_patch_via_identifier(pc_id, x_start, y_start, return_generator=True))
            meta = bev_dict["meta"]
            pc = test_3d_dataset[idx]

            result[cur_dataset].append(bev_back_projection_testing(pc=pc, meta=meta, num_samples=100))

    print(f"Back Projection Result:")
    for dataset_name, passed_list in result.items():
        total = len(passed_list)
        passed = len([x for x in passed_list if x])
        not_passed = total - passed

        print(f"\n{dataset_name}:\n  - passed: {round((passed/total)*100, 2)}%  ({passed})")
        print(f"  - not passed: {round((not_passed/total)*100, 2)}%  ({not_passed})")



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
    # save_dir_creation(path)
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
    # save_dir_creation(path)
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
    # save_dir_creation(path)
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
    # save_dir_creation(path)
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
    # save_dir_creation(path)
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



def manhole_3d_and_2d_density_test(config):
    """
    Manhole Density per Sample Check (2D & 3D)

    3D:
        - Mean: 590.67
        - Median: 406.00
        - Max: 3261.00
        - Min: 1.00
        - 5% Percentile: 20.00 (= 5% are under this value)
        - 10% Percentile: 56.50
        - 25% Percentile: 153.75
        - 50% Percentile: 406.00
        - 95% Percentile: 1562.50

    2D:
        - Mean: 550.23
        - Median: 406.00
        - Max: 2711.00
        - Min: 1.00
        - 5% Percentile: 19.00 (= 5% are under this value)
        - 10% Percentile: 55.50
        - 25% Percentile: 153.00
        - 50% Percentile: 406.00
        - 95% Percentile: 1368.50

    The 10% percentile serves as an empirical threshold to exclude 
    samples that have been so heavily decimated by sensors or 
    edge sections that they can no longer be reliably 
    considered 'complete' topologically.

    If you had taken the 5% percentile (~20 points), you would have kept even more samples, but you risk heavily noisy or extremely garbled edge objects distorting your evaluation.

    If you had taken the 25% percentile (~150 points), you would have already thrown away a quarter of your real test data (survivorship bias - you then only test on the "best" 75% of the data).

    The 10% percentile is the classic statistical standard for capping the bottom 10% (the obvious outliers/errors) without pruning the majority of the dataset.
    """
    if config.data.name == "sud":
        # label_value = (1, 255) if config.data.preprocessed else 3
        label_value = 1 if config.data.preprocessed else 3
    else:
        # label_value = (1, 255) if config.data.preprocessed else 104002
        label_value = 1 if config.data.preprocessed else 104002
    
    print("\n --- Intensity 3D and 2D Check ---")

    print("Loading Data...")
    data_loader = get_data_loader("whu", config.data.path, 
                                    type="train", 
                                    transform=get_basic_transform(),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    n_manhole_pixels = []
    n_manhole_points = []

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
                pc_numpy = point_cloud.to_numpy(as_copy=True)
                points = pc_numpy.labels

                n_manhole_pixels.append(np.sum(labels == 1))
                n_manhole_points.append(np.sum(points == 1))
                
    n_manhole_pixels = np.array(n_manhole_pixels)
    n_manhole_points = np.array(n_manhole_points)

    result = "Manhole Density per Sample Check (2D & 3D)"
    result += f"\n\n3D:"
    result += f"\n    - Mean: {np.mean(n_manhole_points):.2f}"
    result += f"\n    - Median: {np.median(n_manhole_points):.2f}"
    result += f"\n    - Max: {np.max(n_manhole_points):.2f}"
    result += f"\n    - Min: {np.min(n_manhole_points):.2f}"
    result += f"\n    - 5% Percentile: {np.percentile(n_manhole_points, 5):.2f} (= 5% are under this value)"
    result += f"\n    - 10% Percentile: {np.percentile(n_manhole_points, 10):.2f}"
    result += f"\n    - 25% Percentile: {np.percentile(n_manhole_points, 25):.2f}"
    result += f"\n    - 50% Percentile: {np.percentile(n_manhole_points, 50):.2f}"
    result += f"\n    - 95% Percentile: {np.percentile(n_manhole_points, 95):.2f}"

    result += f"\n\n2D:"
    result += f"\n    - Mean: {np.mean(n_manhole_pixels):.2f}"
    result += f"\n    - Median: {np.median(n_manhole_pixels):.2f}"
    result += f"\n    - Max: {np.max(n_manhole_pixels):.2f}"
    result += f"\n    - Min: {np.min(n_manhole_pixels):.2f}"
    result += f"\n    - 5% Percentile: {np.percentile(n_manhole_pixels, 5):.2f} (= 5% are under this value)"
    result += f"\n    - 10% Percentile: {np.percentile(n_manhole_pixels, 10):.2f}"
    result += f"\n    - 25% Percentile: {np.percentile(n_manhole_pixels, 25):.2f}"
    result += f"\n    - 50% Percentile: {np.percentile(n_manhole_pixels, 50):.2f}"
    result += f"\n    - 95% Percentile: {np.percentile(n_manhole_pixels, 95):.2f}"

    print(result)
    return result



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
    # save_dir_creation(path)
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
    # save_dir_creation(path)
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
    # save_dir_creation(path)
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
    # save_dir_creation(path)
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
    # save_dir_creation(path)
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
    # save_dir_creation(path)
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
    # save_dir_creation(path)
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
    # save_dir_creation(path)
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
    # save_dir_creation(path)
    os.makedirs(path, exist_ok=True)

    not_found_files = []

    for idx, batch in enumerate(train_dataset):
        
        x = batch["pixel_values"].detach().cpu().permute(1, 2, 0).numpy()
        # x = np.permute_dims(x, (1, 2, 0))
        y = batch["labels"].detach().cpu().numpy()
        # y[y == 255] = 0
        y = np.ma.masked_where(y == 255, y)
        # find the created gt map (if there is one)
        file_name = os.path.split(all_file_paths[idx])[-1]
        pc_id, x_start, y_start = train_dataset.extract_grid_identifier(file_name)

        # gt_path = f"./2d_gt_patches/{config.data.name}_{pc_id}_{x_start}_{y_start}.npy"
        gt_path = f"/data/2d_gt_patches/{config.data.name}_{pc_id}_{x_start}_{y_start}.npy"
        
        if not os.path.exists(gt_path):
            not_found_files.append(gt_path)
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
        axes[2].set_title("GT Heatmap (sigma 10)")

        axes[3].imshow(gt_2d_map[:, :, 1], cmap="viridis")
        axes[3].axis("off")
        axes[3].set_title("GT Heatmap (sigma 20)")

        axes[4].imshow(gt_2d_map[:, :, 2], cmap="viridis")
        axes[4].axis("off")
        axes[4].set_title("GT Heatmap (sigma 60)")

        plt.tight_layout()

        current_name = f"comparison_{config.data.name}_{pc_id}_{x_start}_{y_start}.png"
        plt.savefig(os.path.join(path, current_name))
        # print(f"Saved at {os.path.join(path, current_name)}")

        plt.close(fig)
        
    print(f"Did not found {len(not_found_files)} files.")
    print("Successfull finished!")



def ground_truth_2d_map_full_check(config):
    print("\n --- Heatmap Check ---")

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

    not_found_files = []

    for idx, batch in enumerate(train_dataset):
        
        x = batch["pixel_values"].detach().cpu().permute(1, 2, 0).numpy()
        y = batch["labels"].detach().cpu().numpy()
        y = np.ma.masked_where(y == 255, y)

        # search the created gt map (if there is one)
        file_name = os.path.split(all_file_paths[idx])[-1]
        pc_id, x_start, y_start = train_dataset.extract_grid_identifier(file_name)

        # gt_path = f"./2d_gt_patches/{config.data.name}_{pc_id}_{x_start}_{y_start}.npy"
        gt_path = f"/data/2d_gt_patches/{config.data.name}_{pc_id}_{x_start}_{y_start}.npy"
        
        if not os.path.exists(gt_path):
            not_found_files.append(gt_path)
            continue
        
    if len(not_found_files) <= 0:
        print("Successfull passed Heatmap Check. Found all Heatmap GTs!!!")
    else:
        print(f"Heatmap Check NOT passed!\nDid not found {len(not_found_files)} files.")



def ground_truth_2d_and_3d_map_test(config):
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
    # save_dir_creation(path)
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

        gt_path = f"./2d_gt_patches/{config.data.name}_{pc_id}_{x_start}_{y_start}_3d.npy"
        if not os.path.exists(gt_path):
            raise ValueError(f"Can find 2D Heatmap but not 3D heatmap: '{gt_path}'")

        gt_3d_map = np.load(gt_path)
        gt_3d_map = np.mean(gt_3d_map[:, :]).reshape(gt_3d_map.shape[:-2])

        fig, axes = plt.subplots(2, 5, figsize=(8*5, 7*2))

        axes[0][0].imshow(x[:, :, 1], cmap="viridis")
        axes[0][0].axis("off")
        axes[0][0].set_title("Intensity")

        axes[0][1].imshow(y, cmap="viridis")
        axes[0][1].axis("off")
        axes[0][1].set_title("Labels")

        axes[0][2].imshow(gt_2d_map[:, :, 0], cmap="viridis")
        axes[0][2].axis("off")
        axes[0][2].set_title("2D GT Heatmap (sigma 10)")

        axes[0][3].imshow(gt_2d_map[:, :, 1], cmap="viridis")
        axes[0][3].axis("off")
        axes[0][3].set_title("2D GT Heatmap (sigma 20)")

        axes[0][4].imshow(gt_2d_map[:, :, 2], cmap="viridis")
        axes[0][4].axis("off")
        axes[0][4].set_title("2D GT Heatmap (sigma 60)")

        # 3D
        print(f"DEBUGGING: shape {gt_3d_map.shape}")
        axes[1][2].imshow(gt_3d_map[:, :], cmap="viridis")
        color = gt_3d_map[:, :, :, 2]
        color = (color - np.min(color)) / (np.max(color) - np.min(color))
        ax[1].scatter(gt_3d_map[:, :, 0].ravel(), y.ravel(), s=5, c=color.ravel(), alpha=1.0, cmap="viridis", edgecolors="none")
        axes[1][2].axis("off")
        axes[1][2].set_title("3D GT Heatmap (sigma 10)")

        axes[1][3].imshow(gt_3d_map[:, :, 1], cmap="viridis")
        axes[1][3].axis("off")
        axes[1][3].set_title("3D GT Heatmap (sigma 20)")

        axes[1][4].imshow(gt_3d_map[:, :, 2], cmap="viridis")
        axes[1][4].axis("off")
        axes[1][4].set_title("3D GT Heatmap (sigma 60)")

        plt.tight_layout()

        current_name = f"comparison_{config.data.name}_{pc_id}_{x_start}_{y_start}.png"
        plt.savefig(os.path.join(path, current_name))

        plt.close(fig)
        

    print("Successfull finished!")


def eval_center_gt(config):
    
    def center_func(manhole_point_cloud, method="mesqra"):
        # (center_coordinates_square, _, points_square, _, _, _, _) = center_estimation_3d_pipeline_debugging(
        #     None,
        #     method="least_square",
        #     extended_return=True,
        #     should_visualize=False,
        #     clusters=manhole_point_cloud,
        #     label_value=1,
        # )
        results = use_points_and_extract_center_point(
            clusters=[manhole_point_cloud], 
            method=method, 
            use_projection=True,
            apply_downsampling=False
        )

        center_points = []
        for result in results:
            center, radius, cluster, inliers, error, loss, input_points = result
            center_points.append(center)

        # print(f"Debugging Type: {type(center_points)}")
        # if isinstance(center_points, (list, tuple)):
        #     print(f"Amount: {len(center_points)}")
        #     (print(f"  - {sub_array.shape}") for sub_array in center_points)
        # else:
        #     print(f"Debugging Shape: {center_points.shape}")
        return center_points


    methods = ["mesqra", "mean", "ransac", "least_square"]
    # sigmas = [0.00, 0.01, 0.02, 0.05, 0.10]
    sigmas = [0.00, 0.05, 0.5]
    point_amounts = [50, 100, 1000]
    first_run = True

    summary_fig, summary_axes = plt.subplots(nrows=len(methods), ncols=1, figsize=(15, 10))
    summary_axes = np.atleast_1d(summary_axes)
    summary_points_fig, summary_points_axes = plt.subplots(nrows=len(methods), ncols=1, figsize=(15, 10))
    sigma_samples_fig, sigma_samples_axes = plt.subplots(nrows=len(methods), ncols=len(sigmas), figsize=(4 * len(sigmas), 3 * len(methods)))
    n_points_samples_fig, n_points_samples_axes = plt.subplots(nrows=len(methods), ncols=len(point_amounts), figsize=(4 * len(point_amounts), 3 * len(methods)))

    summary_fig.suptitle("Center GT Noise Robustness", fontsize=14)
    summary_points_fig.suptitle("Center GT Points Robustness", fontsize=14)

    summary_fig.subplots_adjust(hspace=0.5)
    summary_points_fig.subplots_adjust(hspace=0.5)

    sigma_samples_fig.subplots_adjust(
        hspace=0.7,   # vertical spacing
        wspace=0.3    # horizontal spacing
    )

    n_points_samples_fig.subplots_adjust(
        hspace=0.7,   # vertical spacing
        wspace=0.3    # horizontal spacing
    )

    for i, method in enumerate(methods):
        eval_center_robustness(
            center_func,
            n_samples_per_test=1000,
            n_points_min=56,   # 10 percentile
            n_points_max=1562, # 90 percentile
            n_points=point_amounts,
            center_min=0.0,
            center_max=0.0,
            radius_min=0.2,
            radius_max=1.2, 
            sigma_min=0.0,
            sigma_max=0.5,
            sigmas=sigmas,
            sigma_n_values=3,
            reset_dir=first_run,
            method=method,
            summary_sigma_ax=summary_axes[i],
            summary_points_ax=summary_points_axes[i],
            single_sigma_sample_ax=sigma_samples_axes[i],
            single_n_points_sample_ax=n_points_samples_axes[i]
        )

        first_run = False

        # eval_ellipse_precision(
        #     center_func,
        #     n_samples_per_test=1000,
        #     n_points_min=56,   # 10 percentile
        #     n_points_max=1562, # 90 percentile
        #     center_min=0.0,
        #     center_max=0.0,
        #     radius_min=0.2,
        #     radius_max=1.2, 
        #     sigma=0.2,
        #     method=method
        # )

    # summary_fig.tight_layout()
    # sigma_samples_fig.tight_layout()

    for i, method in enumerate(methods):
        sigma_samples_axes[i, 0].set_ylabel(
            method.upper(),
            fontsize=14,
            rotation=90,
            labelpad=25
        )
    for i, sigma in enumerate(sigmas):
        # sigma_samples_axes[0, i].set_xlabel(
        #     f"δ {sigma}".upper(),
        #     fontsize=14,
        #     rotation=0,
        #     labelpad=25
        # )
        sigma_samples_axes[0, i].set_title(
            f"\u03C3 = {sigma}",  # \u03C3
            fontsize=14,
            pad=25
        )

    handles, labels = sigma_samples_axes[0, 0].get_legend_handles_labels()
    sigma_samples_fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.00)
    )

    for i, method in enumerate(methods):
        n_points_samples_axes[i, 0].set_ylabel(
            method.upper(),
            fontsize=14,
            rotation=90,
            labelpad=25
        )
    for i, n_point in enumerate(point_amounts):
        # n_points_samples_axes[0, i].set_ylabel(
        #     f"{n_point} Points".upper(),
        #     fontsize=14,
        #     rotation=0,
        #     labelpad=25
        # )
        n_points_samples_axes[0, i].set_title(
            f"{n_point} Points".upper(),
            fontsize=14,
            pad=25
        )

    handles, labels = n_points_samples_axes[0, 0].get_legend_handles_labels()
    n_points_samples_fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.00)
    )

    # Sigma - setting equal aspect and autoscale
    for ax in summary_axes.flat:
        ax.set_aspect('equal', adjustable='box')

    for ax in summary_axes:
        ax.relim()
        ax.autoscale_view()

    # xlim = summary_axes[0].get_xlim()
    # ylim = summary_axes[0].get_ylim()

    # for ax in summary_axes:
    #     ax.set_xlim(xlim)
    #     ax.set_ylim(ylim)

    # N-Points - setting equal aspect and autoscale
    for ax in summary_points_axes.flat:
        ax.set_aspect('equal', adjustable='box')

    for ax in summary_points_axes:
        ax.relim()
        ax.autoscale_view()

    # xlim = summary_points_axes[0].get_xlim()
    # ylim = summary_points_axes[0].get_ylim()

    # for ax in summary_points_axes:
    #     ax.set_xlim(xlim)
    #     ax.set_ylim(ylim)

    
    summary_fig.savefig(os.path.join("./output/monte-carlo-gt-check", f"center_summary_sigma.png"), dpi=300)  # , bbox_inches='tight'
    summary_points_fig.savefig(os.path.join("./output/monte-carlo-gt-check", f"center_summary_points.png"), dpi=300)
    sigma_samples_fig.savefig(os.path.join("./output/monte-carlo-gt-check", f"center_sigma_samples.png"), dpi=300)  # , bbox_inches='tight'
    n_points_samples_fig.savefig(os.path.join("./output/monte-carlo-gt-check", f"center_n_points_samples.png"), dpi=300)

    plt.close(summary_fig)
    plt.close(summary_points_fig)
    plt.close(sigma_samples_fig)
    plt.close(n_points_samples_fig)


def manhole_intensity_range_test(config):
    label_value = 1
    
    print("\n --- Intensity Range Check ---")

    def get_intensity_values(data_loader):
        intensities = []

        cur_pc = 0
        for batch in data_loader:
            cur_pc += 1
            point_cloud = batch[0]
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

                print(f"Image Shape: {img.shape}")

                intensity_channel = img[3]

                intensities += [intensity_channel.flatten()]

        return np.array(intensities)
                    

    print("Loading Data...")
    
    data_loader = get_data_loader("whu", config.data.path, 
                                    type="train", 
                                    transform=get_basic_transform(),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    whu_intensities = get_intensity_values(data_loader)

    data_loader = get_data_loader("sud", config.data.path_2, 
                                    type="train", 
                                    transform=get_basic_transform(),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    sud_intensities = get_intensity_values(data_loader)

    print(f"Intensities Shape: {sud_intensities.shape}")

    fig, ax = plt.subplots(figsize=(8, 5))

    def mean_histogram(dataset, bins=256, value_range=None):
        hists = []
        for img in dataset:
            h, bin_edges = np.histogram(np.ravel(img), bins=bins, range=value_range, density=True)
            hists.append(h)
        hists = np.array(hists)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return centers, hists.mean(axis=0), hists.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset, label, color in zip((whu_intensities, sud_intensities), ('WHU', 'SUD'), ('tab:blue', 'tab:orange')):
        centers, mean_h, std_h = mean_histogram(dataset, bins=256)
        ax.plot(centers, mean_h, label=label, color=color)
        ax.fill_between(centers, mean_h - std_h, mean_h + std_h, alpha=0.2, color=color)
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Density')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig(f"./output/intensity_range.png", dpi=300)
    # plt.tight_layout()
    # plt.show()


    # def pooled_histogram(dataset, bins=256, value_range=None):
    #     all_pixels = np.concatenate([np.ravel(img) for img in dataset])
    #     hist, bin_edges = np.histogram(all_pixels, bins=bins, range=value_range, density=True)
    #     centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #     return centers, hist

    # fig, ax = plt.subplots(figsize=(8, 5))
    # for dataset, label in zip((whu_intensities, sud_intensities), ('WHU', 'SUD')):
    #     centers, hist = pooled_histogram(dataset, bins=256)
    #     ax.plot(centers, hist, label=label, linewidth=1.5)
    # ax.set_xlabel('Intensity')
    # ax.set_ylabel('Density')
    # ax.legend()
    # plt.tight_layout()
    # plt.show()


def manhole_sample_counting(config):
    """
     --- Manhole Sample Counting ---
    Loading Data...
    Found 11260 bev images (orthogonal images).
    Found 11260 point clouds.
    Found 11260 bev images (orthogonal images).
    Reduced from 11260 to 106 (filtered by manhole points -> min manhole points: 50).
    Found 1531 bev images (orthogonal images).
    Found 1531 point clouds.
    Found 1531 bev images (orthogonal images).
    Reduced from 1531 to 32 (filtered by manhole points -> min manhole points: 50).
    Found 2907 bev images (orthogonal images).
    Found 2907 point clouds.
    Found 2907 bev images (orthogonal images).
    Reduced from 2907 to 34 (filtered by manhole points -> min manhole points: 50).
    Found 982 bev images (orthogonal images).
    Found 982 point clouds.
    Found 982 bev images (orthogonal images).
    Reduced from 982 to 172 (filtered by manhole points -> min manhole points: 50).
    Found 37 bev images (orthogonal images).
    Found 37 point clouds.
    Found 37 bev images (orthogonal images).
    Reduced from 37 to 11 (filtered by manhole points -> min manhole points: 50).
    Found 160 bev images (orthogonal images).
    Found 160 point clouds.
    Found 160 bev images (orthogonal images).
    Reduced from 160 to 44 (filtered by manhole points -> min manhole points: 50).

    WHU:
        - train: 106
        - val: 32
        - test: 34

    SUD:
        - train: 172
        - val: 11
        - test: 44
    --- Finish ---
    """

    if config.data.name == "sud":
        # label_value = (1, 255) if config.data.preprocessed else 3
        label_value = 1 if config.data.preprocessed else 3
    else:
        # label_value = (1, 255) if config.data.preprocessed else 104002
        label_value = 1 if config.data.preprocessed else 104002
    
    print("\n --- Manhole Sample Counting ---")

    print("Loading Data...")
    result = {}
    for dataset_name in ["whu", "sud"]:
        result[dataset_name] = {}
        for mode in ["train", "val", "test"]:
            result[dataset_name][mode] = 0
            dataset = get_data_loader(dataset_name, config.data.path if dataset_name == "whu" else config.data.path_2, 
                                            type=mode, 
                                            transform=get_basic_transform(),
                                            batch_size=1, shuffle=False, num_workers=0,
                                            preprocessed=config.data.preprocessed, 
                                            return_train_format=False,
                                            return_dataset=True)
            all_paths = dataset.point_cloud_paths
            dataset = BEVDataset(path=all_paths, 
                                    file_paths=[], 
                                    has_labels=True, 
                                    image_training=True, 
                                    preprocessor=None,
                                    augment=False,
                                    pass_label_in_preprocessor=False,
                                    heatmap_gt_path=False,
                                    used_heatmap_channel=False)
            dataset.manhole_filter(required_manhole_points=50, amount_non_manhole_samples=10)
            
            result[dataset_name][mode] += len(dataset)
                
    for dataset_name, modes in result.items():
        print(f"\n{dataset_name.upper()}:")
        for mode_name, sample_amount in modes.items():
            print(f"    - {mode_name}: {sample_amount}")



# def calculate_intensity_percentiles(config):
#     """
#     Computes the 1st and 99th percentiles of intensity values 
#     across all point clouds for specified datasets.

#     Raw Point Cloud Data:
#     [WHU] Intensity 1st Percentile : 1034.0000
#     [WHU] Intensity 99th Percentile: 34802.0000
#     [SUD] Intensity 1st Percentile : 1910.0000
#     [SUD] Intensity 99th Percentile: 3329.0000

#     Patch Intensity Data:
#     [WHU] Intensity 1st Percentile : 1141.0000
#     [WHU] Intensity 99th Percentile: 25715.0000
#     [SUD] Intensity 1st Percentile : 1912.0000
#     [SUD] Intensity 99th Percentile: 3297.0000

#     2D Results? (should be the same? / very similiar)
#     """
#     print("\n --- Calculating Intensity Percentiles (1st & 99th) ---")

#     datasets_to_check = [
#         ("whu", config.data.path),
#         ("sud", config.data.path_2)
#     ]

#     percentiles_result = {}

#     for name, path in datasets_to_check:
#         print(f"\nProcessing dataset: {name.upper()}...")

#         for is_processed in [False, True]:
#             # Load full point cloud dataset (without BEV pre-processing or transforms)
#             data_loader = get_data_loader(
#                 name, 
#                 path, 
#                 type="train", 
#                 transform=get_basic_transform(),
#                 batch_size=1, 
#                 shuffle=False, 
#                 num_workers=0,
#                 preprocessed=is_processed, 
#                 return_train_format=False
#             )

#             all_intensities = []

#             for idx, batch in enumerate(data_loader):
#                 point_cloud = batch[0]

#                 # Extract raw intensity values from the point cloud object
#                 # (assuming points are structured as [N, 4] with (X, Y, Z, Intensity) or accessible as an attribute)
#                 if hasattr(point_cloud, 'intensities'):
#                     intensities = point_cloud.intensities
#                 elif hasattr(point_cloud, 'points'):
#                     intensities = point_cloud.points[:, 3]  # standard 4th channel
#                 else:
#                     raise AttributeError("Could not find point intensity attributes on PointCloud object.")

#                 # print("Debug, intesnity: ", intensities)

#                 # Flatten and cast
#                 ints = np.asarray(intensities, dtype=np.float32).ravel()

#                 # Filter NaN / Inf directly per cloud
#                 valid_ints = ints[np.isfinite(ints)]

#                 if valid_ints.size > 0:
#                     all_intensities.append(valid_ints)

#                 # # Flatten and cast to float32 to save memory
#                 # all_intensities.append(np.asarray(intensities, dtype=np.float32).ravel())

#                 # if (idx + 1) % 50 == 0:
#                 #     print(f"  Processed {idx + 1} point clouds...")

#             if not all_intensities:
#                 print(f"Warning: No intensity values found for {name}!")
#                 continue

#             # Concatenate all point cloud intensity values into a single vector
#             global_intensities = np.concatenate(all_intensities, axis=0)

#             # Compute percentiles
#             p1 = np.percentile(global_intensities, 1)
#             p99 = np.percentile(global_intensities, 99)
#             mean = np.mean(global_intensities)
#             std = np.std(global_intensities)

#             percentiles_result[name] = {
#                 "p1": p1,
#                 "p99": p99,
#                 "mean": mean,
#                 "std": std
#             }

#             processed_str = "processed" if is_processed else "raw"
#             print(f"[{name.upper()} - {processed_str}] Intensity 1st Percentile : {p1:.4f}")
#             print(f"[{name.upper()} - {processed_str}] Intensity 99th Percentile: {p99:.4f}")
#             print(f"[{name.upper()} - {processed_str}] Intensity Mean           : {mean:.4f}")
#             print(f"[{name.upper()} - {processed_str}] Intensity Std            : {std:.4f}")

#     return percentiles_result



def calculate_intensity_statistics(config):
    """
    Computes intensity statistics for both 3D point clouds and 2D BEV images.

    Statistics:
        - 1st percentile
        - 99th percentile
        - mean
        - standard deviation

    Results are calculated separately for WHU and SUD.

    WHU:
    [3D] P1   : 1034.0000
    [3D] P99  : 34802.0000
    [3D] Mean : 6674.9087
    [3D] Std  : 6627.9756

    [2D] P1   : 0.0000
    [2D] P99  : 2224.0000
    [2D] Mean : 59.1831
    [2D] Std  : 711.9660


    SUD:
    [3D] P1   : 1910.0000
    [3D] P99  : 3329.0000
    [3D] Mean : 2493.8530
    [3D] Std  : 285.8570

    [2D] P1   : 0.0000
    [2D] P99  : 2642.0000
    [2D] Mean : 104.6962
    [2D] Std  : 502.2997
    """

    print("\n--- Calculating Intensity Statistics (2D & 3D) ---")

    datasets_to_check = [
        ("whu", config.data.path),
        ("sud", config.data.path_2)
    ]

    statistics_result = {}

    for name, path in datasets_to_check:
        print(f"\nProcessing dataset: {name.upper()}...")

        statistics_result[name] = {}

        # =========================
        # 3D POINT CLOUD STATISTICS

        print(f"  Processing 3D point clouds...")

        data_loader = get_data_loader(
            name,
            path,
            type="train",
            transform=get_basic_transform(),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            preprocessed=False,
            return_train_format=False
        )

        all_intensities_3d = []

        for idx, batch in enumerate(data_loader):

            point_cloud = batch[0]

            # Extract intensity values from point cloud
            if hasattr(point_cloud, 'intensities'):
                intensities = point_cloud.intensities

            elif hasattr(point_cloud, 'points'):
                intensities = point_cloud.points[:, 3]

            else:
                raise AttributeError(
                    "Could not find point intensity attributes "
                    "on PointCloud object."
                )

            # Convert to numpy and flatten
            ints = np.asarray(
                intensities,
                dtype=np.float32
            ).ravel()

            # Remove NaN and Inf
            valid_ints = ints[np.isfinite(ints)]

            if valid_ints.size > 0:
                all_intensities_3d.append(valid_ints)

            if (idx + 1) % 50 == 0:
                print(
                    f"    Processed {idx + 1} point clouds..."
                )

        if not all_intensities_3d:
            print(f"  Warning: No 3D intensity values found!")
        else:

            global_intensities_3d = np.concatenate(
                all_intensities_3d,
                axis=0
            )

            statistics_result[name]["3d"] = {
                "p1": np.percentile(global_intensities_3d, 1),
                "p99": np.percentile(global_intensities_3d, 99),
                "mean": np.mean(global_intensities_3d),
                "std": np.std(global_intensities_3d),
            }

            stats = statistics_result[name]["3d"]

            print(f"    [3D] P1   : {stats['p1']:.4f}")
            print(f"    [3D] P99  : {stats['p99']:.4f}")
            print(f"    [3D] Mean : {stats['mean']:.4f}")
            print(f"    [3D] Std  : {stats['std']:.4f}")


        # =================
        # 2D BEV STATISTICS

        print(f"  Processing 2D BEV images...")

        data_loader = get_data_loader(
            name,
            path,
            type="train",
            transform=get_basic_transform(),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            preprocessed=True,
            return_train_format=False
        )

        all_intensities_2d = []

        for idx, batch in enumerate(data_loader):

            point_cloud = batch[0]

            # Get BEV representation
            if point_cloud.bev_data is None:

                raise ValueError("Should not generate BEV.")

                # Generate BEV if it is not already available
                tiles, metas = bev_projection(
                    point_cloud,
                    tile_size=35.0,
                    resolution=0.05
                )

                bev_gen = bev_gen_wrapper(
                    tiles,
                    metas
                )

            else:

                bev_gen = point_cloud.get_bev()

            # Iterate over BEV tiles
            for bev_item in bev_gen:

                img = (
                    bev_item["pixel_values"]
                    .detach()
                    .cpu()
                    .numpy()
                )

                # Intensity is channel 3
                intensity_channel = img[3]

                # Flatten
                intensities = intensity_channel.ravel()

                # Remove NaN and Inf
                valid_intensities = intensities[
                    np.isfinite(intensities)
                ]

                if valid_intensities.size > 0:
                    all_intensities_2d.append(
                        valid_intensities
                    )

        if not all_intensities_2d:
            print(f"  Warning: No 2D intensity values found!")

        else:

            global_intensities_2d = np.concatenate(
                all_intensities_2d,
                axis=0
            )

            statistics_result[name]["2d"] = {
                "p1": np.percentile(global_intensities_2d, 1),
                "p99": np.percentile(global_intensities_2d, 99),
                "mean": np.mean(global_intensities_2d),
                "std": np.std(global_intensities_2d),
            }

            stats = statistics_result[name]["2d"]

            print(f"    [2D] P1   : {stats['p1']:.4f}")
            print(f"    [2D] P99  : {stats['p99']:.4f}")
            print(f"    [2D] Mean : {stats['mean']:.4f}")
            print(f"    [2D] Std  : {stats['std']:.4f}")

    return statistics_result



def calculate_dataset_statistics(config, max_percentile_samples=500_000):
    """
    Computes global statistics (min, max, mean, std, p1, p99) for 3D point cloud 
    channels (x, y, z, intensity) and 2D BEV image channels (Max Height, Delta Z, 
    Mean Intensity, Density).

    Results:

    WHU:
        [3D X        ] Min: 5538.3770 | Max: 7460.0439 | Mean: 6268.0885 | Std: 513.1016 | P1: 5566.9609 | P99: 7425.2817
        [3D Y        ] Min: 1849.6196 | Max: 3678.2065 | Mean: 3072.3206 | Std: 410.1066 | P1: 1881.0437 | P99: 3640.0256
        [3D Z        ] Min: 11.5335 | Max: 23.0701 | Mean: 14.6740 | Std: 1.3005 | P1: 13.5522 | P99: 21.3371
        [3D INTENSITY] Min: 800.0000 | Max: 65534.0000 | Mean: 5006.8666 | Std: 4917.5458 | P1: 1100.0000 | P99: 44403.0000
    
        [2D MAX_HEIGHT    ] Min: 11.5335 | Max: 23.0701 | Mean: 14.7305 | Std: 1.2948 | P1: 13.5511 | P99: 21.3451
        [2D DELTA_Z       ] Min: 0.0000 | Max: 1.5090 | Mean: 0.0575 | Std: 0.1290 | P1: 0.0010 | P99: 0.6840
        [2D MEAN_INTENSITY] Min: 800.0000 | Max: 65534.0000 | Mean: 4786.7349 | Std: 4348.1123 | P1: 1105.8550 | P99: 40865.0000
        [2D DENSITY       ] Min: 0.6931 | Max: 9.1638 | Mean: 0.7386 | Std: 0.1784 | P1: 0.6931 | P99: 1.3863


    SUD:
        [3D X        ] Min: -352.8900 | Max: 15.5900 | Mean: -205.0103 | Std: 91.9485 | P1: -345.6700 | P99: 5.3300
        [3D Y        ] Min: -275.0950 | Max: 135.9750 | Mean: -72.0568 | Std: 105.6210 | P1: -260.9350 | P99: 130.7450
        [3D Z        ] Min: 1.6750 | Max: 27.5950 | Mean: 15.4685 | Std: 6.0902 | P1: 1.8050 | P99: 27.1650
        [3D INTENSITY] Min: 57.0000 | Max: 5155.0000 | Mean: 2487.2689 | Std: 279.8789 | P1: 1935.0000 | P99: 3384.0000
    
        [2D MAX_HEIGHT    ] Min: 1.6950 | Max: 27.5950 | Mean: 15.4651 | Std: 6.3341 | P1: 1.8050 | P99: 27.1650
        [2D DELTA_Z       ] Min: 0.0100 | Max: 0.7100 | Mean: 0.0101 | Std: 0.0034 | P1: 0.0100 | P99: 0.0100
        [2D MEAN_INTENSITY] Min: 70.0000 | Max: 4314.5000 | Mean: 2499.8788 | Std: 284.4588 | P1: 1944.0000 | P99: 3395.5000
        [2D DENSITY       ] Min: 0.6931 | Max: 3.1355 | Mean: 0.8837 | Std: 0.2619 | P1: 0.6931 | P99: 1.6094
        
    """
    print("\n--- Calculating Extended Statistics (2D & 3D) ---")

    datasets_to_check = [
        ("whu", config.data.path),
        ("sud", config.data.path_2)
    ]

    # Map BEV channel indices schema/format
    bev_channel_names = {
        0: "max_height",
        1: "delta_z",
        2: "mean_intensity",
        3: "density"
    }

    statistics_result = {}

    for name, path in datasets_to_check:
        print(f"\nProcessing dataset: {name.upper()}...")
        statistics_result[name] = {"3d": {}, "2d": {}}

        # =========================================================================
        # 1. 3D POINT CLOUD STATISTICS (x, y, z, intensity)
        # =========================================================================
        print(f"  Processing 3D point clouds...")
        data_loader_3d = get_data_loader(
            name,
            path,
            type="train",
            transform=get_basic_transform(),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            preprocessed=True,
            return_train_format=False
        )

        # Accumulators for 4 features: [X, Y, Z, Intensity]
        counts_3d = np.zeros(4, dtype=np.int64)
        sums_3d = np.zeros(4, dtype=np.float64)
        sq_sums_3d = np.zeros(4, dtype=np.float64)
        mins_3d = np.full(4, np.inf, dtype=np.float64)
        maxs_3d = np.full(4, -np.inf, dtype=np.float64)
        samples_3d = [[] for _ in range(4)]

        for idx, batch in enumerate(data_loader_3d):
            point_cloud = batch[0]

            pts = point_cloud.coordinates
            # Ensure shape is (N, 4) -> X, Y, Z, Intensity
            pts = np.asarray(pts, dtype=np.float32)
    
            ints = np.asarray(point_cloud.intensities, dtype=np.float32).reshape(-1, 1)
            pts = np.hstack((pts[:, :3], ints))

            # Filter non-finite values
            valid_mask = np.all(np.isfinite(pts[:, :4]), axis=1)
            valid_pts = pts[valid_mask, :4]

            if valid_pts.shape[0] > 0:
                counts_3d += valid_pts.shape[0]
                sums_3d += np.sum(valid_pts, axis=0)
                sq_sums_3d += np.sum(valid_pts ** 2, axis=0)
                mins_3d = np.minimum(mins_3d, np.min(valid_pts, axis=0))
                maxs_3d = np.maximum(maxs_3d, np.max(valid_pts, axis=0))

                # Subsample for percentiles to prevent memory explosion
                if valid_pts.shape[0] > 1000:
                    sub_idx = np.random.choice(valid_pts.shape[0], size=1000, replace=False)
                    sub_pts = valid_pts[sub_idx]
                else:
                    sub_pts = valid_pts

                for c in range(4):
                    samples_3d[c].append(sub_pts[:, c])

            if (idx + 1) % 50 == 0:
                print(f"    Processed {idx + 1} 3D point clouds...")

        # Process aggregated 3D stats
        ch_names_3d = ["x", "y", "z", "intensity"]
        for c, ch_name in enumerate(ch_names_3d):
            if counts_3d[c] > 0:
                mean = sums_3d[c] / counts_3d[c]
                var = (sq_sums_3d[c] / counts_3d[c]) - (mean ** 2)
                std = np.sqrt(np.maximum(0.0, var))
                
                cat_samples = np.concatenate(samples_3d[c]) if samples_3d[c] else np.array([])
                p1 = np.percentile(cat_samples, 1) if cat_samples.size > 0 else np.nan
                p99 = np.percentile(cat_samples, 99) if cat_samples.size > 0 else np.nan

                statistics_result[name]["3d"][ch_name] = {
                    "min": float(mins_3d[c]),
                    "max": float(maxs_3d[c]),
                    "mean": float(mean),
                    "std": float(std),
                    "p1": float(p1),
                    "p99": float(p99)
                }

                print(f"    [3D {ch_name.upper():<9}] Min: {mins_3d[c]:.4f} | Max: {maxs_3d[c]:.4f} | Mean: {mean:.4f} | Std: {std:.4f} | P1: {p1:.4f} | P99: {p99:.4f}")

        # =========================================================================
        # 2. 2D BEV STATISTICS (Channels 0, 1, 2, 3)
        # =========================================================================
        print(f"  Processing 2D BEV images...")
        data_loader_2d = get_data_loader(
            name,
            path,
            type="train",
            transform=get_basic_transform(),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            preprocessed=True,
            return_train_format=False,
            bev_normalized=False
        )

        counts_2d = np.zeros(4, dtype=np.int64)
        sums_2d = np.zeros(4, dtype=np.float64)
        sq_sums_2d = np.zeros(4, dtype=np.float64)
        mins_2d = np.full(4, np.inf, dtype=np.float64)
        maxs_2d = np.full(4, -np.inf, dtype=np.float64)
        samples_2d = [[] for _ in range(4)]

        for idx, batch in enumerate(data_loader_2d):
            point_cloud = batch[0]
            
            bev_gen = point_cloud.get_bev() if point_cloud.bev_data is not None else []

            for bev_item in bev_gen:
                img = bev_item["pixel_values"].detach().cpu().numpy()  # Expected shape (C, H, W) or (H, W, C)
                
                # Align channel dimension to first axis if shape is (H, W, C)
                if img.ndim == 3 and img.shape[2] in [4, 5]:
                    img = np.transpose(img, (2, 0, 1))

                for c_idx in range(4):
                    ch_data = img[c_idx].ravel()
                    # we also mask zero because most zeros should come from missing sensor values, but be aware that statistics might be biased from this
                    valid_mask = np.isfinite(ch_data) & (ch_data != 0.0)
                    valid_ch = ch_data[valid_mask]

                    if valid_ch.size > 0:
                        counts_2d[c_idx] += valid_ch.size
                        sums_2d[c_idx] += np.sum(valid_ch)
                        sq_sums_2d[c_idx] += np.sum(valid_ch ** 2)
                        mins_2d[c_idx] = min(mins_2d[c_idx], np.min(valid_ch))
                        maxs_2d[c_idx] = max(maxs_2d[c_idx], np.max(valid_ch))

                        if valid_ch.size > 1000:
                            sub = np.random.choice(valid_ch, size=1000, replace=False)
                        else:
                            sub = valid_ch
                        samples_2d[c_idx].append(sub)

        # Process aggregated 2D stats
        for c_idx, ch_name in bev_channel_names.items():
            if counts_2d[c_idx] > 0:
                mean = sums_2d[c_idx] / counts_2d[c_idx]
                var = (sq_sums_2d[c_idx] / counts_2d[c_idx]) - (mean ** 2)
                std = np.sqrt(np.maximum(0.0, var))

                cat_samples = np.concatenate(samples_2d[c_idx]) if samples_2d[c_idx] else np.array([])
                p1 = np.percentile(cat_samples, 1) if cat_samples.size > 0 else np.nan
                p99 = np.percentile(cat_samples, 99) if cat_samples.size > 0 else np.nan

                statistics_result[name]["2d"][ch_name] = {
                    "min": float(mins_2d[c_idx]),
                    "max": float(maxs_2d[c_idx]),
                    "mean": float(mean),
                    "std": float(std),
                    "p1": float(p1),
                    "p99": float(p99)
                }

                print(f"    [2D {ch_name.upper():<14}] Min: {mins_2d[c_idx]:.4f} | Max: {maxs_2d[c_idx]:.4f} | Mean: {mean:.4f} | Std: {std:.4f} | P1: {p1:.4f} | P99: {p99:.4f}")

    return statistics_result



def calculate_max_density(config):
    """
    Compute maximum point density per BEV-Pixel 
    over all patches.

    Results:
    Processing dataset: WHU...
    Found 11260 bev images (orthogonal images).
    Found 11260 point clouds.
    [WHU] Absolute Max Points/Pixel : 1.0
    [WHU] 95th Percentile Density   : 1.00
    [WHU] 99th Percentile Density   : 1.00
    [WHU] 99.9th Percentile Density : 1.00

    Processing dataset: SUD...
    Found 982 bev images (orthogonal images).
    Found 982 point clouds.
    [SUD] Absolute Max Points/Pixel : 1.0
    [SUD] 95th Percentile Density   : 1.00
    [SUD] 99th Percentile Density   : 1.00
    [SUD] 99.9th Percentile Density : 1.00
    """
    print("\n --- Calculating Max BEV Pixel Density ---")

    datasets_to_check = [
        ("whu", config.data.path),
        ("sud", config.data.path_2)
    ]

    for name, path in datasets_to_check:
        print(f"\nProcessing dataset: {name.upper()}...")

        data_loader = get_data_loader(
            name, 
            path, 
            type="train", 
            transform=get_basic_transform(),
            batch_size=1, 
            shuffle=False, 
            num_workers=0,
            preprocessed=True, 
            return_train_format=False
        )

        pixel_counts_list = []

        for idx, batch in enumerate(data_loader):
            point_cloud = batch[0]
            
            # 1. Get the BEV-Tiles / Patches
            bev_gen = point_cloud.get_bev()

            for bev_item in bev_gen:
                img = bev_item["pixel_values"].detach().cpu().numpy()
                
                density_channel = img[2]
                
                # Only pixels with at least 1 point
                nonzero_counts = density_channel[density_channel > 0]
                
                if nonzero_counts.size > 0:
                    pixel_counts_list.append(nonzero_counts.ravel())

        if not pixel_counts_list:
            print(f"Warning: No valid points found for {name}!")
            continue

        # Aggregate all Pixel-Counts
        all_counts = np.concatenate(pixel_counts_list, axis=0)

        # Calc statistics
        absolute_max = np.max(all_counts)
        p95_density = np.percentile(all_counts, 95)
        p99_density = np.percentile(all_counts, 99)
        p99_9_density = np.percentile(all_counts, 99.9)

        print(f"[{name.upper()}] Absolute Max Points/Pixel : {absolute_max}")
        print(f"[{name.upper()}] 95th Percentile Density   : {p95_density:.2f}")
        print(f"[{name.upper()}] 99th Percentile Density   : {p99_density:.2f}")
        print(f"[{name.upper()}] 99.9th Percentile Density : {p99_9_density:.2f}")              
    


def analyze_point_cloud_resolution(config, num_patches=3, patch_size_m=0.5):
    # init Data Loader
    data_loader = get_data_loader(
        config.data.name, 
        config.data.path, 
        type=config.data.type, 
        transform=get_basic_transform(num_points=-1),
        batch_size=1, 
        shuffle=False, 
        num_workers=1,
        preprocessed=config.data.preprocessed, 
        return_train_format=False
    )

    # Reset Output-Folder
    path = f"./output/pc_resolution_{config.data.name}_{patch_size_m}"
    if os.path.exists(path):
        shutil.rmtree(path)
    # save_dir_creation(path)
    os.makedirs(path, exist_ok=True)

    for batch_idx, batch in enumerate(data_loader):
        point_cloud = batch[0]
        # print_pc(point_cloud)

        # Extraction of 2D coordiantes
        points = point_cloud.coordinates
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
            
        coords_2d = points[:, :2] # only X and Y for 
        half_size = patch_size_m / 2.0

        for patch_idx in range(num_patches):
            # Pick random center point
            random_idx = np.random.randint(0, len(coords_2d))
            center = coords_2d[random_idx]

            # 50cm x 50cm Bounding Box filtering
            mask = (
                (np.abs(coords_2d[:, 0] - center[0]) <= half_size) & 
                (np.abs(coords_2d[:, 1] - center[1]) <= half_size)
            )
            patch_points = coords_2d[mask]

            if len(patch_points) < 10:
                print(f"Patch {patch_idx + 1}: Skipping...not enough points found...")
                continue

            # 2. Compute statistical metrics
            area_sqm = patch_size_m * patch_size_m
            density_per_sqm = len(patch_points) / area_sqm

            tree = KDTree(patch_points)
            distances, _ = tree.query(patch_points, k=2)  # k=2, because k=1 is the point itself
            mean_dist_cm = np.mean(distances[:, 1]) * 100.0  # from meter in cm

            print(f"\n[Batch {batch_idx} | Patch {patch_idx + 1}]")
            print(f"  Points in Patch: {len(patch_points)}")
            print(f"  Pointdensity:     {density_per_sqm:.1f} Pkt/m²")
            print(f"  Mean Distance:  {mean_dist_cm:.2f} cm")

            # 3. Visualization and plotting
            plt.figure(figsize=(8, 8))
            plt.scatter(patch_points[:, 0], patch_points[:, 1], s=12, c='blue', alpha=0.7, edgecolors='none')

            # Set axislimit to 50cm x 50cm
            x_min, x_max = center[0] - half_size, center[0] + half_size
            y_min, y_max = center[1] - half_size, center[1] + half_size
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            # Grid-Ticks in 1-cm-steps
            plt.xticks(np.arange(x_min, x_max + 0.01, 0.01))
            plt.yticks(np.arange(y_min, y_max + 0.01, 0.01))
            plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)

            # Description and Infobox
            plt.title(f"Patch {patch_idx + 1} ({patch_size_m*100:.0f}x{patch_size_m*100:.0f}cm)\n"
                      f"Dichte: {density_per_sqm:.1f} Pkt/m² | Mittl. Abstand: {mean_dist_cm:.2f} cm")
            plt.xlabel("X (Meter)")
            plt.ylabel("Y (Meter)", rotation=60)
            plt.axis('equal')

            # save image
            save_filename = os.path.join(path, f"batch_{batch_idx}_patch_{patch_idx + 1}.png")
            plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Saved on '{save_filename}'")

        # end of first point cloud reached, should be already enough
        break



def analyze_point_cloud_resolution_upgraded(config, raster_size=0.8, grid_size=0.01):
    print("\n --- Center Shape Check ---")

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=None,  # get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    # clear save path
    path = f"./output/pc_resolution_check"
    if os.path.exists(path):
        shutil.rmtree(path)
    # save_dir_creation(path)
    os.makedirs(path, exist_ok=True)

    print("Make resolution check")

    cur_pc = 0
    for batch in tqdm(data_loader, total=len(data_loader), desc="Resolution Check"):
        point_cloud = batch[0]

        # get maholes
        if config.data.name == "sud":
            label_value = (1, 255) if config.data.preprocessed else 3
        else:
            label_value = (1, 255) if config.data.preprocessed else 104002
        manholes = extract_manhole(point_cloud, label_value=label_value, points_around_dist=0)

        for cur_vis, cur_manhole in enumerate(manholes):
            # Extract 2D coordinates
            if isinstance(cur_manhole, o3d.t.geometry.PointCloud):
                points = cur_manhole.point[get_coordinate_attribute(cur_manhole)].numpy()
            points_2d = points[:, :2]

            # (not really important, because of line scan distance) 
            # Calculate mean distance to nearest neighbor for each point
            tree = KDTree(points_2d)
            distances, _ = tree.query(points_2d, k=2)  # k=2 because k=1 is the point itself
            avg_point_dist = np.mean(distances[:, 1])
            
            plot_name = f"pc_{cur_pc}_manhole_{cur_vis}_resolution.png"
            
            plt.style.use("seaborn-v0_8-whitegrid")

            fig, ax = plt.subplots(figsize=(7,7))

            ax.scatter(points_2d[:, 0], points_2d[:, 1], s=5, alpha=0.3, label="Points")

            ax.set_aspect("equal")

            # Force tick intervals to exactly 0.1 on both axes
            ax.xaxis.set_major_locator(ticker.MultipleLocator(grid_size))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(grid_size))
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(grid_size*5.0))
            # ax.yaxis.set_major_locator(ticker.MultipleLocator(grid_size*5.0))

            # ax.xaxis.set_minor_locator(ticker.MultipleLocator(grid_size))
            # ax.yaxis.set_minor_locator(ticker.MultipleLocator(grid_size))

            # ax.grid(True, which='both', linestyle='--', alpha=0.4)

            # X-Axis rotation
            ax.tick_params(axis='x', rotation=90)

            # Make axis font smaller
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)

            # Fix specific bounds around the manhole center -> radius of 1m
            center_x, center_y = np.mean(points_2d[:, 0]), np.mean(points_2d[:, 1])
            ax.set_xlim(center_x - raster_size, center_x + raster_size)
            ax.set_ylim(center_y - raster_size, center_y + raster_size)

            # Make grid visible at every tick step
            ax.grid(True, which='major', linestyle='--', alpha=0.6)
            ax.legend()

            stats_text = (
                f"Points Count : {len(points_2d)}\n"
                f"Avg Pt Dist  : {avg_point_dist:.4f} m\n"
                f"Grid Raster  : {grid_size} m\n"
                f"Raster Size  : {raster_size*2} m"
            )

            ax.text(
                0.05, 0.95,           # X, Y coordinates (5% from left, 95% from bottom)
                stats_text, 
                transform=ax.transAxes,
                fontsize=10, 
                fontfamily='monospace',  # Keeps things aligned nicely like code
                verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
            )

            ax.set_title(f"Point Cloud Resolution Check - PC: {cur_pc} | Manhole: {cur_vis}", fontsize=12, pad=10)

            save_file = os.path.join(path, plot_name)
            plt.savefig(save_file, bbox_inches='tight', dpi=300)
            print(f"Successfull saved to: '{save_file}'")

            plt.close(fig)

        cur_pc += 1

    print("successfull finish resolution check!")


from sklearn.linear_model import RANSACRegressor, LinearRegression

def extract_scanlines_ransac(points_2d, residual_threshold=0.005, min_points=10):
    remaining_points = points_2d.copy()
    labels = np.full(len(points_2d), -1)
    current_label = 0
    
    indices = np.arange(len(points_2d))
    
    while len(remaining_points) >= min_points:
        X = remaining_points[:, 0].reshape(-1, 1)
        y = remaining_points[:, 1]
        
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=residual_threshold,
            min_samples=2,
            max_trials=1000
        )
        
        try:
            ransac.fit(X, y)
        except ValueError:
            break
            
        inlier_mask = ransac.inlier_mask_
        if np.sum(inlier_mask) < min_points:
            break
            
        # Assign cluster label to found line
        original_indices = indices[inlier_mask]
        labels[original_indices] = current_label
        
        # Remove inliers and repeat
        remaining_points = remaining_points[~inlier_mask]
        indices = indices[~inlier_mask]
        current_label += 1
        
    return labels


# FIXME -> remove noise or lines/points which not fit
#       -> add stats (per patch and overall)
def auto_point_cloud_resolution_finder(config, max_distance=0.01):
    print("\n --- Center Shape Check ---")

    print("Loading Data...")
    data_loader = get_data_loader(config.data.name, config.data.path, 
                                    type=config.data.type, 
                                    transform=None,  # get_basic_transform(num_points=-1),
                                    batch_size=1, shuffle=False, num_workers=0,
                                    preprocessed=config.data.preprocessed, return_train_format=False)

    # clear save path
    path = f"./output/pc_resolution_auto_compute"
    if os.path.exists(path):
        shutil.rmtree(path)
    # save_dir_creation(path)
    os.makedirs(path, exist_ok=True)

    print("Make resolution check")

    stats = {
        "total_mean_sl_distance": [],
        "total_min_sl_distance": [],
        "total_max_sl_distance": [],
        "total_scanline_count": [],
        "total_avg_scanline_spacing": [],
        "per_patch_mean_sl_distances": [],
        "per_patch_min_sl_distances": [],
        "per_patch_max_sl_distances": [],
        "per_patch_scanline_counts": [],
        "per_patch_avg_scanline_spacings": []
    }

    cur_pc = 0
    for batch in tqdm(data_loader, total=len(data_loader), desc="Auto Resolution Check"):
        point_cloud = batch[0]

        # get maholes
        if config.data.name == "sud":
            label_value = (1, 255) if config.data.preprocessed else 3
        else:
            label_value = (1, 255) if config.data.preprocessed else 104002
        manholes = extract_manhole(point_cloud, label_value=label_value, points_around_dist=0)

        for cur_vis, cur_manhole in enumerate(manholes):
            # 1. Extract 2D coordinates
            if isinstance(cur_manhole, o3d.t.geometry.PointCloud):
                points = cur_manhole.point[get_coordinate_attribute(cur_manhole)].numpy()
            points_2d = points[:, :2]
            center_x, center_y = np.mean(points_2d[:, 0]), np.mean(points_2d[:, 1])

            # 2. Cluster Points - Get Scanlines
            # clustering = DBSCAN(eps=max_distance, min_samples=10).fit(points_2d)
            # labels = clustering.labels_
            labels = extract_scanlines_ransac(points_2d, residual_threshold=0.01, min_points=20)

            # Cluster to Lines
            unique_labels = [l for l in np.unique(labels) if l != -1]  # Noise (-1) filtern

            if len(unique_labels) < 4:
                print(f"Found not enough Scanlines. Found only {len(unique_labels)} scanlines.")
                continue

            slopes = []
            intercepts = {}

            for label in unique_labels:
                cluster_points = points_2d[labels == label]
                reg = LinearRegression().fit(cluster_points[:, 0].reshape(-1, 1), cluster_points[:, 1])
                slopes.append(reg.coef_[0])
                intercepts[label] = reg.intercept_

            # Use mean slope for all lines, because they should be parallel
            m_avg = np.mean(slopes)

            # std warning, if std is too big
            slope_std = np.std(slopes)

            if slope_std > 0.1:
                print(
                    f"WARNING: Scanlines not sufficiently parallel "
                    f"(slope std = {slope_std:.4f})"
                )

            # Get distance from scaline to each other
            def line_distance(b1, b2, m):
                """
                Compute orthogonal distance between 2 parallel lines.
                """
                return abs(b2 - b1) / np.sqrt(1 + m**2)

            print(f"Mittlere Steigung (m): {m_avg:.4f}\n")

            # 4. Sort Lines (the lines are parallel but can have any rotation)
            # 4.1 Compute cluster centroids and normal vector perpendicular to lines
            centroids = {}
            for label in unique_labels:
                cluster_pts = points_2d[labels == label]
                centroids[label] = np.mean(cluster_pts, axis=0)

            # Normal vector perpendicular to line direction vector (1, m_avg)
            # Depending on orientation preference, use (-m_avg, 1) or (m_avg, -1)
            normal = np.array([-m_avg, 1.0])
            normal = normal / np.linalg.norm(normal)  # Normalize to unit vector

            # 4.2 Project centroids onto normal vector to order them spatially
            sorted_labels = sorted(
                unique_labels,
                key=lambda l: np.dot(centroids[l], normal),
                reverse=True  # True for highest-to-lowest / top-left
            )
            

            # 5. Distance from only neighbor lines
            # Compute distances between the sorted lines 
            # we want the direct orthogonal distance between them
            # 5.1 Store projected distances for sorted labels
            projections = [np.dot(centroids[cur_label], normal) for cur_label in sorted_labels]

            # 5.2 Compute distances between adjacent lines
            neighbor_distances = []
            for i in range(len(sorted_labels) - 1):
                line_1 = sorted_labels[i]
                line_2 = sorted_labels[i + 1]
                
                # Orthogonal distance along the normal vector
                dist = abs(projections[i] - projections[i + 1])
                neighbor_distances.append({
                    "line_pair": (line_1, line_2),
                    "distance": dist
                })

                # print(f"Distance between Line {line_1} and Line {line_2}: {dist:.6f} m")

            # 5.3. Overall average resolution (step size) between adjacent scanlines
            if not neighbor_distances:
                print("[WARNING] No Neighbor Distances found, skipping this manhole...")
                continue

            avg_line_spacing = np.mean([d["distance"] for d in neighbor_distances])
            # print(f"Average Scanline Spacing: {avg_line_spacing:.6f} m")
                
            # 6. Save Stats
            stats["total_mean_sl_distance"].append(avg_line_spacing)
            stats["total_min_sl_distance"].append(np.min([d["distance"] for d in neighbor_distances]))
            stats["total_max_sl_distance"].append(np.max([d["distance"] for d in neighbor_distances]))
            stats["total_scanline_count"].append(len(unique_labels))
            stats["total_avg_scanline_spacing"].append(avg_line_spacing)
            stats["per_patch_mean_sl_distances"].append([avg_line_spacing])
            stats["per_patch_min_sl_distances"].append([np.min([d["distance"] for d in neighbor_distances])])
            stats["per_patch_max_sl_distances"].append([np.max([d["distance"] for d in neighbor_distances])])
            stats["per_patch_scanline_counts"].append([len(unique_labels)])
            stats["per_patch_avg_scanline_spacings"].append([avg_line_spacing])

            # 7.Debug visualization
            # Plot Clustering + Plot extracted lines + Plot line distances
            fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(18, 6))

            ax[0].scatter(points_2d[:, 0], points_2d[:, 1], s=5, alpha=0.3, label="Points")
            ax[0].legend()
            ax[0].set_title("Original Point Cloud")

            # filter for only point sinside a cluster (not noise)
            valid_mask = labels != -1

            ax[1].scatter(points_2d[valid_mask, 0], points_2d[valid_mask, 1], c=labels[valid_mask], cmap='Set3', s=5, alpha=0.7)
            # for label in unique_labels:
            #     cluster_points = points_2d[labels == label]
            #     ax[1].plot(cluster_points[:, 0], cluster_points[:, 1], 'o', markersize=5)
            ax[1].set_title("DBSCAN Clustering of Scanlines")

            ax[2].scatter(points_2d[valid_mask, 0], points_2d[valid_mask, 1], c=labels[valid_mask], cmap='Set3', s=5, alpha=0.7)
            for label in unique_labels:
                cluster_points = points_2d[labels == label]
                centroid = centroids[label]
                # ax[2].plot(cluster_points[:, 0], cluster_points[:, 1], 'o', markersize=5)
                # Plot the fitted line
                b_corr = centroid[1] - m_avg * centroid[0]
    
                x_vals = np.array([np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0])])
                y_vals = m_avg * x_vals + b_corr
                ax[2].plot(x_vals, y_vals, '--', color='red', linewidth=1.5)

            ax[2].set_title(f"Fitted Lines with Mean Slope (m_avg={m_avg:.4f})")

            ax[3].scatter(points_2d[valid_mask, 0], points_2d[valid_mask, 1], c=labels[valid_mask], cmap='Set3', s=5, alpha=0.7)
            for i, dist_info in enumerate(neighbor_distances):
                line_1, line_2 = dist_info["line_pair"]
                dist = dist_info["distance"]
                centroid_1 = centroids[line_1]
                centroid_2 = centroids[line_2]
                mid_point = (centroid_1 + centroid_2) / 2.0

                # plot distance line between centroids
                ax[3].plot(
                    [centroid_1[0], centroid_2[0]], 
                    [centroid_1[1], centroid_2[1]], 
                    color='blue', linestyle='--', linewidth=1.2, alpha=0.8
                )
                
                # compute direction vector and perpendicular offset for text placement
                mid_point = (centroid_1 + centroid_2) / 2.0
                vec = centroid_2 - centroid_1
                norm = np.linalg.norm(vec)
                
                if norm > 0:
                    # perpendicular vector to the line connecting the two centroids
                    perp_vec = np.array([-vec[1], vec[0]]) / norm
                else:
                    perp_vec = np.array([0.0, 1.0])
                
                # change per run the side of the text placement to avoid overlap
                side = 1 if i % 2 == 0 else -1
                offset_dist = 0.01
                text_pos = mid_point + side * offset_dist * perp_vec
                
                # put text
                if i%3 == 0:
                    ax[3].text(
                        text_pos[0], text_pos[1],
                        f"{dist:.4f} m",
                        fontsize=8,
                        color='blue',
                        ha='center',
                        va='center',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.7) # optional für bessere Lesbarkeit
                    )

                # ax[3].annotate(
                #     f"{dist:.4f} m",
                #     xy=mid_point,
                #     xytext=(mid_point[0] + 0.02, mid_point[1] + 0.02),
                #     arrowprops=dict(arrowstyle='->', color='blue'),
                #     fontsize=8,
                #     color='blue'
                # ) 
            # for cur_label in sorted_labels:
            #     cur_centroid = centroids[cur_label]
            #     ax[3].scatter(cur_centroid[0], cur_centroid[1], color='black', s=50, marker='x')
            #     ax[3].text(cur_centroid[0], cur_centroid[1], f"Line {cur_label}", fontsize=8, color='black', ha='right')
            ax[3].set_title(f"Orthogonal Distances Between Neighbor Lines")

            dist_vals = [d["distance"] for d in neighbor_distances]
            info_text = f"Lines found: {len(unique_labels)}\n" \
                        f"Avg Spacing: {np.mean(dist_vals):.4f} m\n" \
                        f"Min Spacing: {np.min(dist_vals):.4f} m\n" \
                        f"Max Spacing: {np.max(dist_vals):.4f} m"

            ax[3].text(0.03, 0.95, info_text, transform=ax[3].transAxes, fontsize=9,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            for a in ax:
                a.set_aspect("equal")
                # a.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
                # a.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
                a.grid(True, which='major', linestyle='--', alpha=0.6)

            plt.savefig(os.path.join(path, f"pc_{cur_pc}_manhole_{cur_vis}_resolution_debug.png"), dpi=300)
            plt.close(fig)

        cur_pc += 1

    # Stats Summarization
    if stats["total_mean_sl_distance"]:
        overall_mean = np.mean(stats["total_mean_sl_distance"])
        overall_min = np.min(stats["total_min_sl_distance"])
        overall_max = np.max(stats["total_max_sl_distance"])
        overall_scanline_count = np.mean(stats["total_scanline_count"])
        overall_avg_spacing = np.mean(stats["total_avg_scanline_spacing"])

        result_str = "\n--- Overall Statistics ---"
        result_str += f"\nMean Scanline Distance: {overall_mean:.4f} m"
        result_str += f"\nMin Scanline Distance: {overall_min:.4f} m"
        result_str += f"\nMax Scanline Distance: {overall_max:.4f} m"
        result_str += f"\nAverage Scanline Count: {overall_scanline_count:.2f}"
        result_str += f"\nAverage Scanline Spacing: {overall_avg_spacing:.4f} m"

    if stats["per_patch_mean_sl_distances"]:
        per_patch_mean = np.mean([np.mean(distances) for distances in stats["per_patch_mean_sl_distances"]])
        per_patch_min = np.mean([np.min(distances) for distances in stats["per_patch_min_sl_distances"]])
        per_patch_max = np.mean([np.max(distances) for distances in stats["per_patch_max_sl_distances"]])
        per_patch_scanline_count = np.mean([len(distances) for distances in stats["per_patch_scanline_counts"]])
        per_patch_avg_spacing = np.mean([np.mean(distances) for distances in stats["per_patch_avg_scanline_spacings"]])

        result_str += "\n--- Per Patch Statistics ---"
        result_str += f"\nMean Scanline Distance (Per Patch): {per_patch_mean:.4f} m"
        result_str += f"\nMin Scanline Distance (Per Patch): {per_patch_min:.4f} m"
        result_str += f"\nMax Scanline Distance (Per Patch): {per_patch_max:.4f} m"
        result_str += f"\nAverage Scanline Count (Per Patch): {per_patch_scanline_count:.2f}"
        result_str += f"\nAverage Scanline Spacing (Per Patch): {per_patch_avg_spacing:.4f} m"

    print(result_str)

    result_path = os.path.join(path, "resolution_stats.txt")
    with open(result_path, "w") as f:
        f.write(result_str)

    print(f"successfull finish resolution check!\nSaved result: '{result_path}'")



# --------------
# > Playground <
# --------------
def tryout(config):
    # simple_viusalize_point_cloud(config)
    # torch_tensor_loading(config)

    # bev_segmentation_trying(config)
    # bev_preprocessed_loading_working_testing(config)
    # bev_back_preprocessed_loading_working_testing(config)

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

    # ground_truth_2d_map_test(config)
    # ground_truth_2d_and_3d_map_test(config)

    # eval_center_gt(config)

    # manhole_3d_and_2d_density_test(config)
    # manhole_intensity_range_test(config)

    # manhole_sample_counting(config)

    # calculate_intensity_statistics(config)
    # calculate_dataset_statistics(config, max_percentile_samples=1_000_000)
    # calculate_max_density(config)

    # not working, maybe on local work:
    # generate_presentation_plots(config)
    
    # analyze_point_cloud_resolution(config, num_patches=200, patch_size_m=0.5)
    # analyze_point_cloud_resolution(config, num_patches=200, patch_size_m=1.0)
    # analyze_point_cloud_resolution_upgraded(config, raster_size=0.1, grid_size=0.01)
    auto_point_cloud_resolution_finder(config, max_distance=0.05)

    # ground_truth_2d_map_full_check(config)




    