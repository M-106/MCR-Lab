# -----------
# > Imports <
# -----------
import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import label, binary_closing
from scipy.ndimage.morphology import generate_binary_structure # or skimage.morphology?

from mcrlab.execution.train import get_model_and_processor
from mcrlab.execution.test import predict_single_sample
from mcrlab.classic.least_squares import fit_circle_least_squares
from mcrlab.point_cloud.shape_check import circle_shape_check
from mcrlab.point_cloud.data import get_data_loader, get_basic_transform, BEVDataset
from mcrlab.projection import bev_projection, bev_pixel_to_3d
from mcrlab.metrices import mask_to_polygon





# ---------------
# > Center Eval <
# ---------------
def add_to_result(result, cur_dataset, pc_id, cur_center_point):
    found_entry = False
    for cur_res in result:
        if cur_res.get("dataset", "-999") == cur_dataset and \
            cur_res.get("pointcloud-id", "-999") == pc_id:
            if hasattr(cur_res, "centers"):
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
    pred_poly = kwargs.get("mask_to_polygon_fn")(pred_mask)
    if pred_poly is None:
        return None
    return pred_poly.centroid.x, pred_poly.centroid.y



def extract_center_circle_least_squares(pred_mask, **kwargs):
    """
    Extracts the center by fitting a circle via least squares on the polygon exterior.
    """
    pred_poly = kwargs.get("mask_to_polygon_fn")(pred_mask)
    if pred_poly is None:
        return None
    
    polys_to_plot = pred_poly.geoms if hasattr(pred_poly, 'geoms') else [pred_poly]
    cur_manhole_pred_x, cur_manhole_pred_y = [], []
    
    for p in polys_to_plot:
        x, y = p.exterior.xy
        cur_manhole_pred_x.extend(x)
        cur_manhole_pred_y.extend(y)
        
    cur_manhole_pred_x = np.array(cur_manhole_pred_x)
    cur_manhole_pred_y = np.array(cur_manhole_pred_y)
    
    fit_fn = kwargs.get("fit_circle_fn")
    center_x, center_y, _, _, _ = fit_fn(cur_manhole_pred_x, cur_manhole_pred_y)
    return center_x, center_y



def extract_center_peak_maxima(prediction, **kwargs):
    """
    Placeholder for extracting centers directly from heatmap peaks (e.g., using skimage.feature.peak_local_max).
    """
    # Implement peak local maxima extraction for heatmaps here
    raise NotImplementedError("Peak local maxima extraction not implemented yet.")



# Registry for center extraction methods
CENTER_EXTRACTION_STRATEGIES = {
    "polygon_centroid": extract_center_polygon_centroid,
    "circle_least_squares": extract_center_circle_least_squares,
    "peak_maxima": extract_center_peak_maxima,
}

def center_eval(config):

    model_name = config.model.name.lower()

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
    
    extraction_method_name = getattr(config.center_eval, "center_extraction_method", "polygon_centroid")
    if extraction_method_name not in CENTER_EXTRACTION_STRATEGIES:
        raise ValueError(f"Unknown extraction method: {extraction_method_name}. Choose from {list(CENTER_EXTRACTION_STRATEGIES.keys())}")
    
    extract_center_fn = CENTER_EXTRACTION_STRATEGIES[extraction_method_name]
    print(f"Using center extraction strategy: {extraction_method_name}")

    result = []

    for cur_dataset in ["whu", "sud"]:
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
            used_heatmap_channel=used_heatmap_channel
        )

        for idx, cur_data_path in enumerate(all_test_paths):
            pc_id, x_start, y_start = test_bev_dataset.extract_grid_identifier(cur_data_path)

            # get data
            bev_dict = next(test_bev_dataset.get_patch_via_identifier(pc_id, x_start, y_start, return_generator=True))
            meta = bev_dict["meta"]
            pixel_values = bev_dict["pixel_values"]
            labels = bev_dict["labels"]
            pc = test_3d_dataset[idx]

            pixel_values = pixel_values.unsqueeze(0)

            # --- Manhole Search ---
            if model_name == "traditional":
                # print(f"Shape check, should be [C, W, H]: {pixel_values.shape}")
                centers = get_manhole_candidates_hough(bev_image=pixel_values, resolution=0.01)
                
                for elem in centers:
                    center_x, center_y = elem["center_px"][0], elem["center_px"][1]
                    center = transform_pixel_to_3d(pc, center_x, center_y, meta)
                    result = add_to_result(result, cur_dataset, pc_id, {"x": center[0], "y": center[1], "z": center[2]})
            else:
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
                preds_binary = ((preds >= 0.5) & valid_mask).astype(np.uint8)
                preds_binary = np.squeeze(preds_binary)
                preds = np.squeeze(preds)
                
                struct = generate_binary_structure(2, 2)  # 8-Nachbarschaft
                preds_closed = binary_closing(preds_binary, structure=struct, iterations=4).astype(np.uint8)
                labeled_preds, num_pred_objects = label(preds_closed)

                # for_range = range(1, num_pred_objects + 1)

                for p_idx in range(1, num_pred_objects + 1):
                    pred_mask = (labeled_preds == p_idx)
                    
                    # Extract center using the selected strategy function
                    center_coords = extract_center_fn(
                        pred_mask=pred_mask,
                        prediction=preds,
                        mask_to_polygon_fn=mask_to_polygon,
                        fit_circle_fn=fit_circle_least_squares
                    )
                    
                    if center_coords is None:
                        continue
                        
                    center_x, center_y = center_coords

                    # Transform 2D pixel center to 3D point
                    center = bev_pixel_to_3d(
                        patch_points=pc,
                        pixel_x=center_x,
                        pixel_y=center_y,
                        origin_x=meta["origin_x"],
                        origin_y=meta["origin_y"],
                        resolution=meta["resolution"],
                        search_radius=None
                    )

                    print(f"Center shape: {center.shape}")
                    
                    cur_center_point = {
                        "x": center[0],
                        "y": center[1],
                        "z": center[2]
                    }
                    
                    result = add_to_result(result, cur_dataset, pc_id, cur_center_point)


        # Save evaluation results per dataset
        output_path = Path(f"./output/{cur_dataset}_eval_{exp_name}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as file_:
            json.dump(result, file_, indent=4)


def main(config):
    # model 
    center_eval(config)






