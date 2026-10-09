# -----------
# > Imports <
# -----------
import os
import shutil
import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import label, binary_closing
from scipy.ndimage.morphology import generate_binary_structure # or skimage.morphology?
from scipy.ndimage import center_of_mass
from skimage.feature import peak_local_max
from tqdm import tqdm

import matplotlib.pyplot as plt

from mcrlab.execution.train import get_model_and_processor
from mcrlab.execution.test import predict_single_sample
from mcrlab.classic.least_squares import fit_circle_least_squares
from mcrlab.point_cloud.shape_check import circle_shape_check
from mcrlab.point_cloud.data import get_data_loader, get_basic_transform, BEVDataset
from mcrlab.projection import bev_projection, bev_pixel_to_3d
from mcrlab.metrices import mask_to_polygon
from mcrlab.classic.traditional_methods import predict_manhole_centers
# from mcrlab.helper import save_dir_creation



# --------------------
# > Debugging helper <
# --------------------
def plot_center_prediction_debug(
    pixel_values, labels, labeled_preds, preds_prob, preds_binary, preds_closed, 
    extracted_centers, gt_centers, pc_points, meta, pc_id, idx, save_dir,
    num_preds
):
    """
    Generates a 6-panel visual diagnostic grid to isolate failures between 
    segmentation predictions, morphology, center fitting, and 3D projection.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract Intensity/BEV image background
    if isinstance(pixel_values, torch.Tensor):
        img_bg = pixel_values.detach().cpu().numpy().squeeze()
    else:
        img_bg = np.squeeze(pixel_values)
        
    if img_bg.ndim == 3:
        img_bg = img_bg[0]  # Take 1st channel

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Preparation for coloring GT & preds
    # masking so that background is transparent in the overlay
    preds_colored = np.where(labeled_preds > 0, labeled_preds, np.nan)
    # remove ignore index
    labels_clean = np.where(labels == 255, 0, labels) if np.any(labels == 255) else labels
    labels_clean = np.squeeze(labels_clean)
    labeled_gt, num_gt = label((labels_clean > 0).astype(int))
    gt_colored = np.where(labeled_gt > 0, labeled_gt, np.nan)
    # colormaps (we want different colors)
    cmap_pred = plt.colormaps.get_cmap("tab10").resampled(max(num_preds, 1))
    cmap_gt = plt.colormaps.get_cmap("Set1").resampled(max(num_preds, 1))
    
    # Plotting
    # Panel 1: Ground Truth Mask / Heatmap
    axes[0, 0].imshow(img_bg, cmap="gray", alpha=0.4)
    axes[0, 0].imshow(gt_colored, cmap=cmap_gt, alpha=0.6, interpolation="none")  # labels, "jet"
    axes[0, 0].set_title("1. GT Mask / Heatmap Overlay")
    
    # Panel 2: Model Output Probability/Heatmap
    im2 = axes[0, 1].imshow(preds_prob, cmap="magma")
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    axes[0, 1].set_title("2. Raw Model Probability")

    # Panel 3: Thresholded Binary Mask (Seg Check)
    axes[0, 2].imshow(preds_binary, cmap="binary")
    axes[0, 2].set_title("3. Thresholded Mask (>= 0.5 -> FIXME right?)")  # FIXME

    # Panel 4: Post-Processed Binary Mask (Morphology Check)
    # axes[1, 0].imshow(preds_closed, cmap="binary")
    # axes[1, 0].set_title("4. Morphologically Closed Mask")
    axes[1, 0].imshow(img_bg, cmap="gray", alpha=0.3)
    axes[1, 0].imshow(preds_colored, cmap=cmap_pred, alpha=0.8, interpolation="none")
    axes[1, 0].set_title(f"4. Morphologically Closed Pred Objects (Count: {num_preds})")

    # Panel 5: Extracted Centers on 2D BEV
    # axes[1, 1].imshow(img_bg, cmap="gray")
    # axes[1, 1].imshow(preds_closed, cmap="Purples", alpha=0.4)
    axes[1, 1].imshow(img_bg, cmap="gray")
    axes[1, 1].imshow(preds_colored, cmap=cmap_pred, alpha=0.3, interpolation="none")
    
    # Plot GT centers
    for gx, gy in gt_centers:
        axes[1, 1].scatter(gx, gy, edgecolors="lime", facecolors="none", marker="o", s=100, linewidths=2.5, label="GT")
    
    # Plot predicted centers
    for cx, cy in extracted_centers:
        axes[1, 1].scatter(cx, cy, c="red", marker="x", s=120, linewidths=2.5, label="Pred")
        
    axes[1, 1].set_title("5. Extracted Centers (GT=Green O, Pred=Red X)")

    # Panel 6: 3D Point Cloud Local Overhead Check - Fixed 1D/2D array indexing crash
    pts = None
    if pc_points is not None:
        if isinstance(pc_points, torch.Tensor):
            pts = pc_points.detach().cpu().numpy()
        elif isinstance(pc_points, np.ndarray):
            pts = pc_points
        elif isinstance(pc_points, (list, tuple)) and len(pc_points) > 0:
            p0 = pc_points[0]
            pts = p0.detach().cpu().numpy() if isinstance(p0, torch.Tensor) else np.array(p0)

    if pts is not None and pts.ndim == 2 and pts.shape[1] >= 3 and len(pts) > 0:
        axes[1, 2].scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap="viridis", s=2)
        axes[1, 2].set_title("6. 3D Point Cloud Patch (X/Y)")
        axes[1, 2].set_aspect("equal")
        axes[1, 2].invert_yaxis()  # FIXME invertation is right? -> maybe just change x and y?
    else:
        axes[1, 2].text(0.5, 0.5, "No/Invalid 3D PC Data", ha="center", va="center")
        axes[1, 2].set_title("6. 3D Point Cloud Patch")

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle(f"Debug Analysis - Sample #{idx} | PC ID: {pc_id}", fontsize=14, fontweight="bold")
    out_file = save_dir / f"debug_{pc_id}_idx_{idx}.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)



# ---------------
# > Center Eval <
# ---------------
def add_to_result(result, cur_dataset, pc_id, cur_center_point):
    found_entry = False
    for cur_res in result:
        if cur_res.get("dataset", "-999") == cur_dataset and \
            cur_res.get("pointcloud-id", "-999") == pc_id:
            if "centers" in cur_res:  # hasattr(cur_res, "centers"):
                cur_res["centers"].append(cur_center_point)
            else:
                cur_res["centers"] = [cur_center_point]

            found_entry = True
            break
    if not found_entry:
        result.append({
            "dataset": cur_dataset,
            "pointcloud-id": pc_id,
            "centers": [cur_center_point]
        })
    
    return result



def extract_center_polygon_centroid(pred_mask, **kwargs):
    """
    Extracts the center using the geometric centroid of the polygon mask.
    """
    mask_to_polygon_fn = kwargs.get("mask_to_polygon_fn")
    if mask_to_polygon_fn is None:
        raise ValueError("'mask_to_polygon_fn' must be provided in kwargs.")
        # return None

    pred_poly = mask_to_polygon_fn(pred_mask)

    if pred_poly is None or pred_poly.is_empty:
        return None

    # if pred_poly is None:
    #     return None

    # row = y, col = x in BEV
    # return pred_poly.centroid.x, pred_poly.centroid.y

    centroid = pred_poly.centroid
    return float(centroid.x), float(centroid.y)



def extract_center_circle_least_squares(pred_mask, **kwargs):
    """
    Extracts the center by fitting a circle via least squares on the polygon exterior.

    Gets one potential manhole points as input (not the whole patch).
    """
    mask_to_polygon_fn = kwargs.get("mask_to_polygon_fn")
    fit_circle_fn = kwargs.get("fit_circle_fn")

    if mask_to_polygon_fn is None or fit_circle_fn is None:
        raise ValueError("Both 'mask_to_polygon_fn' and 'fit_circle_fn' must be provided in kwargs.")
    
    pred_poly = mask_to_polygon_fn(pred_mask)
    if pred_poly is None or pred_poly.is_empty:
        return None
    
    polys_to_plot = pred_poly.geoms if hasattr(pred_poly, 'geoms') else [pred_poly]
    cur_manhole_pred_x, cur_manhole_pred_y = [], []
    
    for p in polys_to_plot:
        if p.is_empty or p.exterior is None:
            continue
        x, y = p.exterior.xy
        cur_manhole_pred_x.extend(x)
        cur_manhole_pred_y.extend(y)

    # filter FIXME
    if len(cur_manhole_pred_x) < 20:
        return None
        
    cur_manhole_pred_x = np.array(cur_manhole_pred_x)
    cur_manhole_pred_y = np.array(cur_manhole_pred_y)
    
    fit_fn = kwargs.get("fit_circle_fn")
    center_x, center_y, _, _, _ = fit_fn(cur_manhole_pred_x, cur_manhole_pred_y)
    return center_x, center_y



def extract_center_peak_maxima(prediction, **kwargs):
    """
    Placeholder for extracting centers directly from heatmap peaks (e.g., using skimage.feature.peak_local_max).
    """
    min_confidence_peak = kwargs.get("min_confidence_peak", 0.3)
    min_distance = kwargs.get("min_distance", 3)
    num_peaks = kwargs.get("num_peaks", 1)

    if prediction is None or np.max(prediction) < min_confidence_peak:
        return None

    # Find peak coordinates (returned as array of shape [N, 2] in [row, col] format)
    peaks = peak_local_max(
        prediction,
        min_distance=min_distance,
        threshold_abs=min_confidence_peak,
        num_peaks=num_peaks,
        exclude_border=False,
    )

    if len(peaks) == 0:
        return None

    # Extract the highest probability peak if multiple are returned
    if num_peaks == 1:
        row, col = peaks[0]
        return float(col), float(row)  # Return (x, y) = (col, row)

    # Return list of tuples if searching for multiple manholes
    return [(float(col), float(row)) for row, col in peaks]
    # return np.array([(float(col), float(row)) for row, col in peaks]).mean()



def extract_center_peak_maxima(pred_mask, prediction, **kwargs):
    """
    Finds heatmap-maximum inside a specific cadidate mask (proposal).
    
    Args:
        pred_mask (np.ndarray): Binary mask from the single Candidate-Object.
        prediction (np.ndarray): Continuous Heatmap/Probability Map [0..1].
    """
    min_confidence_peak = kwargs.get("min_confidence_peak", 0.3)
    min_distance = kwargs.get("min_distance", 3)

    if prediction is None or pred_mask is None:
        return None

    # Masking the prediction, so only heatmap values of the candidate are visible
    masked_pred = np.where(pred_mask, prediction, 0.0)

    # Break if the max probability is under the given threshold
    if np.max(masked_pred) < min_confidence_peak:
        return None

    # search the peak (hopefully center) inside the instance
    # we want only one peak, because it is one proposal manhole
    peaks = peak_local_max(
        masked_pred,
        min_distance=min_distance,
        threshold_abs=min_confidence_peak,
        num_peaks=1,
        exclude_border=False,
    )

    if len(peaks) == 0:
        # Fallback: if peak_local_max cause of min_distancefinds nothing 
        # then just take the argmax value
        row, col = np.unravel_index(np.argmax(masked_pred), masked_pred.shape)
        return float(col), float(row)

    if len(peaks) > 1:
        print(f"[Warning] {len(peaks)} Peaks in one Proposal found (we take simply the first).")

    row, col = peaks[0]
    return float(col), float(row)  # Return (x, y) = (col, row)



def extract_center_intensity_weighted(pred_mask, prediction, **kwargs):
    """
    Extracts center using intensity-weighted Center of Mass (useful for smooth heatmaps).
    """
    min_confidence_peak = kwargs.get("min_confidence_peak", 0.1)
    
    if prediction is None or pred_mask is None:
        return None

    # check min confiedence on only the current object
    masked_heatmap = np.where(pred_mask & (prediction >= min_confidence_peak), prediction, 0.0)
    if np.sum(masked_heatmap) == 0:
        return None

    # scipy returns (row, col)
    row_center, col_center = center_of_mass(masked_heatmap)
    if np.isnan(row_center) or np.isnan(col_center):
        return None
    return float(col_center), float(row_center)


# Registry for center extraction methods
CENTER_EXTRACTION_STRATEGIES = {
    "polygon_centroid": extract_center_polygon_centroid,
    "circle_least_squares": extract_center_circle_least_squares,
    "peak_maxima": extract_center_peak_maxima,
    "intensity_weighted_center": extract_center_intensity_weighted
}

# Merging
def merge_nearby_centers(centers, dist_threshold=0.8):
    """
    Merges predicted (x, y, z) centers that are closer than dist_threshold pixels.
    """
    if len(centers) <= 1:
        return centers

    merged_centers = []
    used = [False] * len(centers)

    for i in range(len(centers)):
        if used[i]:
            continue

        cluster = [centers[i]]
        used[i] = True

        for j in range(i + 1, len(centers)):
            if used[j]:
                continue

            # calculate 3d euclidean distance
            dist = np.sqrt(
                (centers[i][0] - centers[j][0])**2 +
                (centers[i][1] - centers[j][1])**2 +
                (centers[i][2] - centers[j][2])**2
            )

            if dist < dist_threshold:
                cluster.append(centers[j])
                used[j] = True

        avg_x = sum(pt[0] for pt in cluster) / len(cluster)
        avg_y = sum(pt[1] for pt in cluster) / len(cluster)
        avg_z = sum(pt[2] for pt in cluster) / len(cluster)
        merged_centers.append((avg_x, avg_y, avg_z))
    return merged_centers



def make_prediction(
    pixel_values, 
    model, 
    model_name, 
    labels, 
    min_confidence, 
    ignore_index, 
    using_heatmap_as_gt,
    as_prob=True, 
    manhole_class_idx=1
):
    if isinstance(pixel_values, np.ndarray):
        pixel_values = torch.from_numpy(pixel_values)

    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values.float().to(model.device)

    # print(f"Pixel Value Shape: {pixel_values.shape}")
    # [1, 3, 500, 500]
    # make center prediction
    preds = predict_single_sample(
        model, 
        model_name, 
        None, 
        pixel_values,
        using_heatmap_as_gt=using_heatmap_as_gt, 
        as_prob=True, 
        manhole_class_idx=1)
    # print(f"DEBUGGING 1, shape: {preds.shape}")
    # [1, 500, 500]

    # apply closing + clustering
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    valid_mask = (labels != ignore_index)

    # set confidence
    preds_binary = ((preds >= min_confidence) & valid_mask).astype(np.uint8)
    preds_binary = np.squeeze(preds_binary)
    preds_prob = np.squeeze(preds)
    preds_prob *= preds_binary

    return preds_prob, preds_binary, valid_mask, preds, labels, pixel_values



def filter_ignored_manholes(
    results,
    dataloader,
    box_size=0.6,
    ignore_label=255,
    ignore_threshold=0.20,
    check_z_axis=False
):
    """
    Filters predicted manhole centers if the region around them contains
    >= ignore_threshold (20%) points labeled with ignore_label (255).

    Args:
        results (list): List of dicts, each formatted as:
            {
                "dataset": str,
                "pointcloud-id": str/int,
                "centers": [{"x": float, "y": float, "z": float, "confidence": float}, ...]
            }
        dataloader: PyTorch DataLoader for fetching full raw point clouds.
        box_size (float): Side length of the square bounding box in meters (default: 0.6m).
        ignore_label (int): Label value representing ignored areas (default: 255).
        ignore_threshold (float): Ratio threshold to reject predictions (default: 0.20).
        check_z_axis (bool): If True, also bounds Z within box_size/2. If False, checks XY bounding box only.

    Returns:
        list: Filtered results with predictions in ignored regions removed.
    """
    half_box = box_size / 2.0
    filtered_results = []

    for entry in results:
        pc_id = entry["pointcloud-id"]
        centers = entry.get("centers", [])

        if not centers:
            continue

        # 1. Fetch full raw point cloud for current pc_id
        try:
            raw_pc_idx = dataloader.dataset.get_idx_by_pc_id(pc_id)
            raw_point_cloud = dataloader.dataset[raw_pc_idx]
        except Exception as e:
            print(f"Warning: Could not load point cloud ID '{pc_id}': {e}. Skipping filtering for these centers.")
            filtered_results.append(entry)
            continue

        # Extract coordinates and labels
        raw_positions = raw_point_cloud.point[get_coordinate_attribute(raw_point_cloud)].numpy()
        
        if "labels" not in raw_point_cloud.point:
            # If no labels present, keep all centers
            filtered_results.append(entry)
            continue

        raw_labels = raw_point_cloud.point["labels"].numpy()

        valid_centers = []

        # 2. Process each center point
        for center in centers:
            cx, cy, cz = center["x"], center["y"], center["z"]

            # Crop box in XY plane (and optionally Z)
            in_box_mask = (
                (np.abs(raw_positions[:, 0] - cx) <= half_box) &
                (np.abs(raw_positions[:, 1] - cy) <= half_box)
            )

            if check_z_axis:
                in_box_mask = in_box_mask & (np.abs(raw_positions[:, 2] - cz) <= half_box)

            box_labels = raw_labels[in_box_mask]

            # If no points are found in the box, keep the prediction
            if len(box_labels) == 0:
                valid_centers.append(center)
                continue

            # Calculate ignore label ratio
            ignore_ratio = np.mean(box_labels == ignore_label)

            # Keep only if ignore label coverage is strictly below threshold
            if ignore_ratio < ignore_threshold:
                valid_centers.append(center)

        if valid_centers:
            # Create a shallow copy with updated centers list
            filtered_entry = dict(entry)
            filtered_entry["centers"] = valid_centers
            filtered_results.append(filtered_entry)

    return filtered_results



def center_eval(config):

    model_name = config.model.name.lower()

    enable_debug = getattr(config.center_eval, "save_debug_plots", False)
    min_confidences = getattr(config.center_eval, "min_confidences")
    min_confidence_peak = getattr(config.center_eval, "min_confidence_peak")
    candidate_min_points = getattr(config.center_eval, "candidate_min_points")

    if model_name.startswith("traditional_"):
        min_confidences = [-999]

    print(f"Confidences: {min_confidences}")
    
    # extract params
    heatmap_path = config.data.heatmap_path
    used_heatmap_channel = config.data.used_heatmap_channel
    if heatmap_path is None or heatmap_path == "None":
        using_heatmap_as_gt = False
        heatmap_path = None
    else:
        using_heatmap_as_gt = True

    ignore_index = 255
    num_labels = 2
    if using_heatmap_as_gt:
        ignore_index = -999
        num_labels = 1
        using_heatmap_as_gt = True

    # Load the TRAINED model checkpoint
    encoder_name = config.model.encoder
    checkpoint_path = config.model.check_point_path
    if not model_name.startswith("traditional_"):
        if (not checkpoint_path or checkpoint_path == "None"):
            raise ValueError("Please provide the path to your trained checkpoint in config.model.check_point_path")

        print(f"Loading trained model and processor from: {checkpoint_path}")
        model, processor = get_model_and_processor(model_name, encoder_name, checkpoint_path, mode="test", num_labels=num_labels, ignore_index=ignore_index, heatmap_is_gt=using_heatmap_as_gt)
        model.eval().to("cuda")

        parts = Path(checkpoint_path).parts
        exp_name = parts[-2]
    else:
        exp_name = model_name
        processor = None
        model = None

    # Load Test Data
    heatmap_path = config.data.heatmap_path
    used_heatmap_channel = config.data.used_heatmap_channel
    pass_label_in_preprocessor = model_name in ["mask2former", "oneformer"]
    normalization = config.data.normalization
    normalization_mode = config.data.normalization_mode

    if heatmap_path is None or heatmap_path == "None":
        using_heatmap_as_gt = False
        heatmap_path = None
    else:
        using_heatmap_as_gt = True
    
    extraction_method_name = getattr(config.center_eval, "center_extraction_method", "polygon_centroid")
    if extraction_method_name not in CENTER_EXTRACTION_STRATEGIES:
        raise ValueError(f"Unknown extraction method: {extraction_method_name}. Choose from {list(CENTER_EXTRACTION_STRATEGIES.keys())}")
    
    extract_center_fn = CENTER_EXTRACTION_STRATEGIES[extraction_method_name]
    print(f"Using center extraction strategy: {extraction_method_name}")

    for cur_dataset in ["whu", "sud"]:
        # result list over all thresholds of one dataset
        all_dataset_results = []

        for min_confidence in min_confidences:
            raw_threshold_results = []

            debug_save_dir = Path(f"./output/center_eval/debug_plots_{exp_name}_{cur_dataset}_{min_confidence}")
            # save_dir_creation(str(debug_save_dir))
            os.makedirs(str(debug_save_dir), exist_ok=True)
            shutil.rmtree(str(debug_save_dir))
            # save_dir_creation(str(debug_save_dir))
            os.makedirs(str(debug_save_dir), exist_ok=True)


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
                bev_normalized=config.data.normalization, 
                bev_normalize_mode=config.data.normalization_mode
            )
            
            all_test_paths = test_3d_dataset.point_cloud_paths

            test_bev_dataset = BEVDataset(
                path=all_test_paths, 
                file_paths=[], 
                has_labels=True, 
                image_training=True, 
                preprocessor=processor,
                augment=False,
                pass_label_in_preprocessor=pass_label_in_preprocessor,
                heatmap_gt_path=heatmap_path,
                used_heatmap_channel=used_heatmap_channel,
                normalize=normalization, 
                normalization_mode=normalization_mode
            )

            all_pc_ids = set()

            for idx, cur_data_path in tqdm(enumerate(all_test_paths), total=len(all_test_paths), desc="2D Center Pipe"):
                pc_id, x_start, y_start = test_bev_dataset.extract_grid_identifier(cur_data_path)

                all_pc_ids.add(pc_id)

                # get data
                bev_dict = next(test_bev_dataset.get_patch_via_identifier(pc_id, x_start, y_start, return_generator=True))
                meta = bev_dict["meta"]
                pixel_values = bev_dict["pixel_values"]
                labels = bev_dict["labels"]
                
                # pc = test_3d_dataset[idx]  # -> wrong patch?
                pc, pc_labels = test_3d_dataset.get_patch_via_identifier(pc_id, x_start, y_start)
                # [type(x) for x in pc]=[<class 'torch.Tensor'>, <class 'torch.Tensor'>]
                # [x.shape for x in pc]=[torch.Size([424, 4]), torch.Size([424, 1])]
                # print(f"{type(pc)=}")
                # print(f"{[type(x) for x in pc]=}")
                # print(f"{[x.shape for x in pc]=}")

                pixel_values = pixel_values.unsqueeze(0)

                # -----------------
                # Manhole Search
                if model_name.startswith("traditional_"):
                    traditional_method = "_".join(model_name.split("_")[1:])

                    # Call unified prediction method using direct PyTorch Tensors
                    detected_centers = predict_manhole_centers(
                        pts_3d=pc,
                        bev_img=pixel_values,
                        method=traditional_method,
                        meta=meta
                    )

                    # Process predictions depending on 2D vs 3D method output
                    for det in detected_centers:
                        if traditional_method.startswith("2d"):
                            # det is [pixel_x, pixel_y, radius] -> transform to 3D point
                            center_3d = bev_pixel_to_3d(
                                patch_points=pc,
                                pixel_x=det[0],
                                pixel_y=det[1],
                                origin_x=meta["origin_x"],
                                origin_y=meta["origin_y"],
                                resolution=meta["resolution"],
                                search_radius=None,
                                tile_size=meta["tile_size"],
                                invert_y=False,
                                invert_x=False
                            )
                        else:
                            # det is already 3D coordinates [x, y, z]
                            center_3d = det

                        if center_3d is not None and len(center_3d) >= 3:
                            cur_center_point = {
                                "x": float(center_3d[0]),
                                "y": float(center_3d[1]),
                                "z": float(center_3d[2]),
                                "confidence": min_confidence
                            }
                            raw_threshold_results = add_to_result(raw_threshold_results, cur_dataset, pc_id, cur_center_point)
                else:
                    preds_prob, preds_binary, valid_mask, preds, gt_labels_numpy, pixel_values = make_prediction(
                        pixel_values, 
                        model, 
                        model_name, 
                        labels, 
                        min_confidence, 
                        ignore_index,
                        using_heatmap_as_gt,
                        as_prob=True, 
                        manhole_class_idx=1)

                    orig_h, orig_w = int(meta["tile_size"] / meta["resolution"]), int(meta["tile_size"] / meta["resolution"])
                    pred_h, pred_w = preds_binary.shape
                        
                    struct = generate_binary_structure(2, 2)  # 8-Nachbarschaft
                    # bigger neighborhood against a problem, where multiple predictions are made due to little riffles
                    preds_closed = binary_closing(preds_binary, structure=struct, iterations=8).astype(np.uint8)
                    labeled_preds, num_pred_objects = label(preds_closed)

                    extracted_pixel_centers = []
                    gt_pixel_centers = []

                    # ONLY FOR DEBUGGING:
                    # extract GT center from labels
                    # labels[labels == ignore_index] = 0
                    labels_clean = np.where(gt_labels_numpy == ignore_index, 0, gt_labels_numpy)
                    # close gaps
                    labels_binary = (labels_clean >= min_confidence).astype(np.uint8)
                    labels_binary = np.squeeze(labels_binary)
                    struct = generate_binary_structure(2, 2)
                    labels_closed = binary_closing(labels_binary, structure=struct, iterations=8).astype(np.uint8)

                    if np.any(labels_closed > 0):
                        gt_labeled, num_gt = label(labels_closed)
                        for g_i in range(1, num_gt + 1):
                            gt_mask = (gt_labeled == g_i)
                            
                            # Skip small noise components with less than 4 pixels
                            if np.count_nonzero(gt_mask) < 20:
                                continue
                                
                            g_coords = extract_center_fn(
                                pred_mask=gt_mask,
                                prediction=labels_closed if not using_heatmap_as_gt else labels_clean.astype(np.float32),
                                mask_to_polygon_fn=mask_to_polygon,
                                fit_circle_fn=fit_circle_least_squares
                            )
                            if g_coords is not None:
                                gt_pixel_centers.append(g_coords)
                    
                    # -----------------
                    # Prediction Center Extraction
                    for p_idx in range(1, num_pred_objects + 1):
                        pred_mask = (labeled_preds == p_idx)

                        if np.sum(pred_mask) < candidate_min_points:
                            continue
                        
                        # Extract center using the selected strategy function
                        center_coords = extract_center_fn(
                            pred_mask=pred_mask,
                            prediction=preds_prob,
                            mask_to_polygon_fn=mask_to_polygon,
                            fit_circle_fn=fit_circle_least_squares,
                            min_confidence_peak=min_confidence_peak
                        )

                        if center_coords is None:
                            continue

                        center_x, center_y = center_coords
                        extracted_pixel_centers.append((center_x, center_y))

                        # Transform 2D pixel center to 3D point
                        center = bev_pixel_to_3d(
                            patch_points=pc,
                            pixel_x=center_x,
                            pixel_y=center_y,
                            origin_x=meta["origin_x"],
                            origin_y=meta["origin_y"],
                            resolution=meta["resolution"],
                            search_radius=None,
                            tile_size=meta["tile_size"],
                            invert_y=False,
                            invert_x=False
                        )

                        # print(f"Center shape: {center.shape}")
                        if center is None or len(center) < 3:
                            print(f"[Warning] Skipped a center prediction -> pred: {center}")
                            continue
                        
                        cur_center_point = {
                            "x": center[0],
                            "y": center[1],
                            "z": center[2],
                            "confidence": min_confidence
                        }
                        
                        raw_threshold_results = add_to_result(raw_threshold_results, cur_dataset, pc_id, cur_center_point)

                    # --- Execute Debug Plotting ---
                    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
                    if enable_debug and (num_pred_objects > 0 or np.any(labels_np > 0)):
                        plot_center_prediction_debug(
                            pixel_values=pixel_values,
                            labels=labels_np,
                            labeled_preds=labeled_preds,
                            preds_prob=preds_prob,
                            preds_binary=preds_binary,
                            preds_closed=preds_closed,
                            extracted_centers=extracted_pixel_centers,
                            gt_centers=gt_pixel_centers,
                            pc_points=pc,
                            meta=meta,
                            pc_id=pc_id,
                            idx=idx,
                            save_dir=debug_save_dir,
                            num_preds=num_pred_objects
                        )

            # -----------------
            # Post Processing
            # Merge nearby predictions
            print(f"Post-processing center predictions for {cur_dataset} (merging duplicates from patch overlaps).")
            
            threshold_merged_results = []

            for cur_result in raw_threshold_results:
                target_pc_id = cur_result["pointcloud-id"]
                target_confidence = min_confidence
                centers = cur_result.get("centers", [])

                if not centers:
                    continue

                for pt in centers:
                    if pt.get("confidence", -999) != target_confidence:
                        raise ValueError(f"Found another confidence inside ofthe current result.")

                # extract points & confidence
                pc_points = [
                    (pt["x"], pt["y"], pt["z"])
                    for pt in centers
                ]

                # merging
                merged_pixel_centers = merge_nearby_centers(
                    pc_points, 
                    dist_threshold=0.8
                )

                # rebuild merged version
                for cur_new_pixel_center in merged_pixel_centers:
                    cur_center_point = {
                        "x": cur_new_pixel_center[0],
                        "y": cur_new_pixel_center[1],
                        "z": cur_new_pixel_center[2],
                        "confidence": min_confidence
                    }
                    
                    threshold_merged_results = add_to_result(threshold_merged_results, cur_dataset, target_pc_id, cur_center_point)
            
            print(f"Conf {min_confidence}: Reduced results from {len(raw_threshold_results)} to {len(threshold_merged_results)} ({len(raw_threshold_results)-len(threshold_merged_results)}).")

            # add new found manhole centers of this threshold to the overall results
            # all_dataset_results.extend(threshold_merged_results)
            for res in threshold_merged_results:
                target_pc_id = res["pointcloud-id"]
                new_centers = res.get("centers", [])

                # Find if an entry for this pc_id already exists in all_dataset_results
                existing_entry = next((e for e in all_dataset_results if e.get("pointcloud-id") == target_pc_id), None)

                if existing_entry is not None:
                    existing_entry["centers"].extend(new_centers)
                else:
                    all_dataset_results.append(res)

        # Check if found center in ignored manhole
        # if prediction is in 255 label area of the 3D 
        # -> if 0.6m box around is 20% or more ignore index, then not pass the prediction
        # Check if found center is in ignored manhole area
        all_dataset_results = filter_ignored_manholes(
            results=all_dataset_results,
            dataloader=raw_dataloader,
            box_size=0.6,
            ignore_label=255,
            ignore_threshold=0.20,
            check_z_axis=False  # Set to True if Z coordinate should also be bounded by 0.6m
        )

        # -----------------
        # Saving
        # Save evaluation results per dataset (and over all thresholds)
        output_path = Path(f"./output/{cur_dataset}_eval_{exp_name}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as file_:
            json.dump(all_dataset_results, file_, indent=4)
        print(f"Saved Eval Center Results in: '{output_path}'")



def main(config):
    # model 
    center_eval(config)






