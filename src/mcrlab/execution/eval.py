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
# from mcrlab.helper import save_dir_creation



# --------------------
# > Debugging helper <
# --------------------
def plot_center_prediction_debug(
    pixel_values, labels, preds_prob, preds_binary, preds_closed, 
    extracted_centers, gt_centers, pc_points, meta, pc_id, idx, save_dir
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
    cmap_pred = plt.cm.get_cmap("tab10", max(num_preds, 1))
    cmap_gt = plt.cm.get_cmap("Set1", max(num_gt, 1))
    
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

    # filter 
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



def make_prediction(pixel_values, model, model_name, labels, min_confidence, ignore_index):
    if isinstance(pixel_values, np.ndarray):
        pixel_values = torch.from_numpy(pixel_values)

    if isinstance(pixel_values, torch.Tensor):
        pixel_values = pixel_values.float().to(model.device)

    # print(f"Pixel Value Shape: {pixel_values.shape}")
    # [1, 3, 500, 500]
    # make center prediction
    preds = predict_single_sample(model, model_name, None, pixel_values)
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



def center_eval(config):

    model_name = config.model.name.lower()

    enable_debug = getattr(config.center_eval, "save_debug_plots", False)
    min_confidence = getattr(config.center_eval, "min_confidence")
    min_confidence_peak = getattr(config.center_eval, "min_confidence_peak")
    candidate_min_points = getattr(config.center_eval, "candidate_min_points")
    
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
    if not checkpoint_path or checkpoint_path == "None":
        raise ValueError("Please provide the path to your trained checkpoint in config.model.check_point_path")

    print(f"Loading trained model and processor from: {checkpoint_path}")
    model, processor = get_model_and_processor(model_name, encoder_name, checkpoint_path, mode="test", num_labels=num_labels, ignore_index=ignore_index, heatmap_is_gt=using_heatmap_as_gt)
    model.eval().to("cuda")

    parts = Path(checkpoint_path).parts
    exp_name = parts[-2]

    # Load Test Data
    heatmap_path = config.data.heatmap_path
    used_heatmap_channel = config.data.used_heatmap_channel
    pass_label_in_preprocessor = model_name in ["mask2former", "oneformer"]
    normalization = config.data.normalization
    normalization_mode = config.data.normalization_mode
    
    extraction_method_name = getattr(config.center_eval, "center_extraction_method", "polygon_centroid")
    if extraction_method_name not in CENTER_EXTRACTION_STRATEGIES:
        raise ValueError(f"Unknown extraction method: {extraction_method_name}. Choose from {list(CENTER_EXTRACTION_STRATEGIES.keys())}")
    
    extract_center_fn = CENTER_EXTRACTION_STRATEGIES[extraction_method_name]
    print(f"Using center extraction strategy: {extraction_method_name}")

    for cur_dataset in ["whu", "sud"]:
        result = []
        debug_save_dir = Path(f"./output/center_eval/debug_plots_{exp_name}_{cur_dataset}")

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
            if model_name == "traditional":
                # print(f"Shape check, should be [C, W, H]: {pixel_values.shape}")
                centers = get_manhole_candidates_hough(bev_image=pixel_values, resolution=0.01)
                
                for elem in centers:
                    center_x, center_y = elem["center_px"][0], elem["center_px"][1]
                    center = transform_pixel_to_3d(pc, center_x, center_y, meta)
                    result = add_to_result(result, cur_dataset, pc_id, {"x": center[0], "y": center[1], "z": center[2]})
            else:
                preds_prob, preds_binary, valid_mask, preds, gt_labels_numpy, pixel_values = make_prediction(pixel_values, model, model_name, labels, min_confidence, ignore_index)

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
                        "z": center[2]
                    }
                    
                    result = add_to_result(result, cur_dataset, pc_id, cur_center_point)

                # --- Execute Debug Plotting ---
                if enable_debug and (num_pred_objects > 0 or np.any(labels > 0)):
                    plot_center_prediction_debug(
                        pixel_values=pixel_values,
                        labels=labels,
                        preds_prob=preds_prob,
                        preds_binary=preds_binary,
                        preds_closed=preds_closed,
                        extracted_centers=extracted_pixel_centers,
                        gt_centers=gt_pixel_centers,
                        pc_points=pc,
                        meta=meta,
                        pc_id=pc_id,
                        idx=idx,
                        save_dir=debug_save_dir
                    )

        # -----------------
        # Post Processing
        # Merge nearby predictions
        print(f"Post-processing center predictions for {cur_dataset} (merging duplicates from patch overlaps).")
        
        cur_all_pred_centers = []
        final_dataset_results = []

        for target_pc_id in all_pc_ids:
            # get all 3d centers
            pc_3d_centers = [
                (item["center"]["x"], item["center"]["y"], item["center"]["z"]) \
                for item in result if item["pointcloud-id"] == target_pc_id
            ]

            if not pc_3d_centers:
                continue

            # add the merged version to the new results
            merged_pixel_centers = merge_nearby_centers(pc_3d_centers, dist_threshold=0.8)
            for cur_new_pixel_center in merged_pixel_centers:
                cur_center_point = {
                    "x": cur_new_pixel_center[0],
                    "y": cur_new_pixel_center[1],
                    "z": cur_new_pixel_center[2]
                }
                
                final_dataset_results = add_to_result(final_dataset_results, cur_dataset, target_pc_id, cur_center_point)
        
        print(f"Reduced results from {len(result)} to {len(final_dataset_results)} ({len(result)-len(final_dataset_results)}).")
        result = final_dataset_results

       

        # -----------------
        # Saving
        # Save evaluation results per dataset
        output_path = Path(f"./output/{cur_dataset}_eval_{exp_name}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as file_:
            json.dump(result, file_, indent=4)
        print(f"Saved Eval Center Results in: '{output_path}'")


def main(config):
    # model 
    center_eval(config)






