# -----------
# > Imports <
# -----------
import os

import matplotlib.pyplot as plt

# import evaluate
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
from scipy.ndimage import label, binary_closing, generate_binary_structure
from scipy.optimize import linear_sum_assignment
import torch

from skimage import measure
from shapely.geometry import Polygon

# mean_iou_metric = evaluate.load("mean_iou")



def mask_to_polygon(mask):
    contours = measure.find_contours(mask.astype(np.uint8), 0.5)

    if len(contours) == 0:
        return None

    # take largest contour
    largest_contour = max(contours, key=lambda x: len(x))

    # (row, col) → (x, y)
    coords = [(p[1], p[0]) for p in largest_contour]

    # create Polygon
    poly = Polygon(coords)

    # if broken (self-intersection etc.)
    if not poly.is_valid:
        poly = poly.buffer(0)

    return poly




def polygon_iou(poly1, poly2):
    if poly1 is None or poly2 is None:
        return 0.0

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    return intersection / union if union > 0 else 0.0



# ------------
# > Metrices <
# ------------

def evaluate_object_wise(preds, labels, iou_threshold, ignore_index=255, debug_plot_path=None):
    valid_mask = (labels != ignore_index)
    
    preds_binary = ((preds == 1) & valid_mask).astype(np.uint8)
    labels_binary = ((labels == 1) & valid_mask).astype(np.uint8)

    struct = generate_binary_structure(2, 2)  # 8-Nachbarschaft
    preds_closed = binary_closing(preds_binary, structure=struct, iterations=4).astype(np.uint8)
    labels_closed = binary_closing(labels_binary, structure=struct, iterations=4).astype(np.uint8)

    labeled_preds, num_pred_objects = label(preds_closed)
    labeled_labels, num_true_objects = label(labels_closed)

    # filter of small islands
    # FIXME -> ok?
    # temp_labeled_preds, num_temp_preds = label(preds_closed)
    # temp_labeled_labels, num_temp_trues = label(labels_closed)

    # clean_preds = np.zeros_like(preds_closed)
    # for i in range(1, num_temp_preds + 1):
    #     if np.sum(temp_labeled_preds == i) >= min_pixel_size:
    #         clean_preds[temp_labeled_preds == i] = 1
            
    # clean_labels = np.zeros_like(labels_closed)
    # for i in range(1, num_temp_trues + 1):
    #     if np.sum(temp_labeled_labels == i) >= min_pixel_size:
    #         clean_labels[temp_labeled_labels == i] = 1

    # labeled_preds, num_pred_objects = label(clean_preds)
    # labeled_labels, num_true_objects = label(clean_labels)

    # debug plot
    if debug_plot_path:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Live Metric Debug (GT Objs: {num_true_objects}, Pred Objs: {num_pred_objects})", fontsize=14)

        # col 1: original image masks comparison
        overlay_raw = np.zeros((*preds_binary.shape, 3), dtype=np.uint8)
        overlay_raw[labels_binary == 1] = [255, 0, 0]  # GT = red
        overlay_raw[preds_binary == 1] = [0, 255, 0]  # preds = green
        overlay_raw[(labels_binary == 1) & (preds_binary == 1)] = [255, 255, 0]  # intersection = yellow
        axes[0].imshow(overlay_raw)
        axes[0].set_title("Original Masks (red=GT, green=pred)")

        # col 2: with closing operation + poylgon (come a bit later)
        axes[1].imshow(labels_closed, cmap='gray')
        axes[1].set_title("GT after Closing + Polygone")
        axes[2].imshow(preds_closed, cmap='gray')
        axes[2].set_title("Pred after Closing + Polygone")

    # make iou calc
    tp_objects = 0
    all_obj_ious = []
    
    # if no obj exist, break
    if num_true_objects == 0 and num_pred_objects == 0:
        return {"tp_objects_count": 0, "true_objects_count": 0, "pred_objects_count": 0, "object_mean_iou": 0.0}
    
    if num_true_objects == 0 or num_pred_objects == 0:
        # every pred is false positive and IoU 0
        return {
            "tp_objects_count": 0,
            "true_objects_count": num_true_objects,
            "pred_objects_count": num_pred_objects,
            "object_mean_iou": 0.0
        }

    # 1. create IoU costmatrix (form: True Objects x Pred objects)
    # because linear_sum_assignment (hungary_algorithm) minimizes, we take (1-IoU) as cost
    cost_matrix = np.ones((num_true_objects, num_pred_objects), dtype=np.float32)

    for t_idx in range(1, num_true_objects + 1):
        true_mask = (labeled_labels == t_idx)
        true_poly = mask_to_polygon(true_mask)

        for p_idx in range(1, num_pred_objects + 1):
            pred_mask = (labeled_preds == p_idx)
            pred_poly = mask_to_polygon(pred_mask)

            # debug plot
            if debug_plot_path and t_idx == 1:
                if true_poly is not None:
                    polys_to_plot = true_poly.geoms if hasattr(true_poly, 'geoms') else [true_poly]
                    for p in polys_to_plot:
                        x, y = p.exterior.xy
                        axes[1].plot(x, y, color='cyan', linewidth=2)
                    axes[1].text(true_poly.centroid.x, true_poly.centroid.y, f"GT_{t_idx}", color='cyan', weight='bold')
                    # x, y = true_poly.exterior.xy
                    # axes[1].plot(x, y, color="cyan", linewidth=2)
                    # axes[1].text(np.mean(x), np.mean(y), f"GT_{t_idx}", color='cyan', weight='bold')

                if pred_poly is not None:
                    polys_to_plot = pred_poly.geoms if hasattr(pred_poly, 'geoms') else [pred_poly]
                    for p in polys_to_plot:
                        x, y = p.exterior.xy
                        axes[2].plot(x, y, color='magenta', linewidth=2)
                    axes[2].text(pred_poly.centroid.x, pred_poly.centroid.y, f"Pred_{p_idx}", color='magenta', weight='bold')
                    # x, y = pred_poly.exterior.xy
                    # axes[2].plot(x, y, color='magenta', linewidth=2)
                    # axes[2].text(np.mean(x), np.mean(y), f"Pred_{p_idx}", color='magenta', weight='bold')

            iou = polygon_iou(true_poly, pred_poly)

            cost_matrix[t_idx - 1, p_idx - 1] = 1.0 - iou

            # fast Pixel-IoU instead of slow Shapely Polygone
            # maybe also try with polygons, because the precise pixel does not matter so much
            # intersection = np.logical_and(true_mask, pred_mask).sum()
            # if intersection > 0:
            #     union = np.logical_or(true_mask, pred_mask).sum()
            #     iou = intersection / union if union > 0 else 0.0
            #     cost_matrix[t_idx - 1, p_idx - 1] = 1.0 - iou

    # saving the debug plot
    if debug_plot_path:
        plt.tight_layout()
        os.makedirs(os.path.dirname(debug_plot_path), exist_ok=True)
        plt.savefig(debug_plot_path, bbox_inches='tight', dpi=150)
        plt.close()

    # 2. make global matching
    true_ind, pred_ind = linear_sum_assignment(cost_matrix)

    # 3. evaluate the matched pairs
    matched_preds = set()
    matched_gts = set()

    for t_i, p_i in zip(true_ind, pred_ind):
        iou = 1.0 - cost_matrix[t_i, p_i]

        # a real amtching only makes sense if there is any overlap
        if iou > 0:
            all_obj_ious.append(iou)
            matched_gts.add(t_i + 1)
            matched_preds.add(p_i + 1)
            
            if iou >= iou_threshold:
                tp_objects += 1

    # 4. unmachted GTs (False NEgatives= gets 0 IoU)
    num_fn_objects = num_true_objects - len(matched_gts)
    all_obj_ious.extend([0.0] * num_fn_objects)

    # 5. Unmachted Preds (False Positives = gets 0 IoU)
    num_fp_objects = num_pred_objects - len(matched_preds)
    all_obj_ious.extend([0.0] * num_fp_objects)

    mean_obj_iou = np.mean(all_obj_ious) if len(all_obj_ious) > 0 else 0.0

    return {
        "tp_objects_count": tp_objects,
        "true_objects_count": num_true_objects,
        "pred_objects_count": num_pred_objects,
        "object_mean_iou": float(mean_obj_iou)
    }



def compute_metrics(preds, labels, iou_threshold_start=0.0, iou_threshold_end=0.95, iou_threshold_step=0.05):
    # logits = eval_pred.predictions
    # labels = eval_pred.label_ids

    # convert to numpy
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # print("DEBUGGING PRINT:")
    # print(f"labels: labels\nDtype: {type(labels)}, len: {len(labels)}")
    # print(f"Shape: {labels.shape}") if hasattr(labels, "shape") else ""

    # print(f"labels: {type(labels)}, len: {len(labels)}")
    # for i_, x in enumerate(labels):
    #     if isinstance(x, (list, tuple)):
    #         print(f"    - {i_}: {type(x)}, len: {len(x)}")
    #         for i_2, x_2 in enumerate(x):
    #             print(f"        - {i_2} shape: {x_2.shape}")
    #     else:
    #         print(f"  - {i_} shape: {x.shape}")

    if preds.shape != labels.shape:
        raise ValueError(f"Shape mismatch! Preds shape is {preds.shape}, but Labels shape is {labels.shape}.")

    # print(f"\nDEBUG INFO:\n  - preds shape (eval): {preds.shape}\n  - labels shape: {labels.shape}")

    batch_size = preds.shape[0]

    obj_results = []

    for cur_iou_threshold in np.arange(iou_threshold_start, iou_threshold_end+iou_threshold_step, iou_threshold_step):

        total_tp = 0
        total_true = 0
        total_pred = 0
        total_obj_iou = 0.0

        # aggregated_obj_metrics = {
        #     "true_objects_count": 0, "pred_objects_count": 0,
        #     "object_recall": 0.0, "object_precision": 0.0,
        #     "object_f1": 0.0, "object_mean_iou": 0.0
        # }

        for batch_idx in range(batch_size):

            # if 0.46 < cur_iou_threshold < 0.54:
            #     metrics_idx = 0
            #     plot_path = f"./debug_outputs/{metrics_idx:04}_batch_{batch_idx}.png"
            #     while os.path.exists(plot_path):
            #         metrics_idx += 1
            #         plot_path = f"./debug_outputs/{metrics_idx:04}_batch_{batch_idx}.png"
            # else:
            #     plot_path = None
            plot_path = None

            # >>> compute object metrics <<<
            obj_metrics = evaluate_object_wise(
                preds=preds[batch_idx], 
                labels=labels[batch_idx], 
                iou_threshold=cur_iou_threshold,
                ignore_index=255,
                debug_plot_path=plot_path
            )
            total_tp += obj_metrics["tp_objects_count"]
            total_true += obj_metrics["true_objects_count"]
            total_pred += obj_metrics["pred_objects_count"]
            total_obj_iou += obj_metrics["object_mean_iou"]
            # for key in aggregated_obj_metrics:
            #     aggregated_obj_metrics[key] += obj_metrics[key]

        obj_recall = total_tp / total_true if total_true > 0 else 0.0
        obj_precision = total_tp / total_pred if total_pred > 0 else 0.0
        obj_f1 = (2 * obj_precision * obj_recall) / (obj_precision + obj_recall) if (obj_precision + obj_recall) > 0 else 0.0
        avg_obj_iou = total_obj_iou / batch_size

        obj_results.append({
            "iou_threshold": cur_iou_threshold,
            "avg_obj_recall": float(obj_recall),
            "avg_obj_precision": float(obj_precision),
            "avg_f1": float(obj_f1),
            "avg_obj_iou": float(avg_obj_iou)
        })

    # get mA results
    mAP = np.array([cur_obj_result["avg_obj_precision"] for cur_obj_result in obj_results], dtype=np.float32).mean()
    mAR = np.array([cur_obj_result["avg_obj_recall"] for cur_obj_result in obj_results], dtype=np.float32).mean()
    mAF1 = np.array([cur_obj_result["avg_f1"] for cur_obj_result in obj_results], dtype=np.float32).mean()
    mAIOU = np.array([cur_obj_result["avg_obj_iou"] for cur_obj_result in obj_results], dtype=np.float32).mean()

    # >>> compute pixel metrics <<<
    # create mask for ignroe index
    mask = labels != 255
    preds_flat = preds[mask].flatten()
    labels_flat = labels[mask].flatten()
    # -> Flatten for sklearn metrics (exclude ignore_index)

    f1 = f1_score(labels_flat, preds_flat, pos_label=1, zero_division=0)
    precision = precision_score(labels_flat, preds_flat, pos_label=1, zero_division=0)
    recall = recall_score(labels_flat, preds_flat, pos_label=1, zero_division=0)

    # calc iou via sklearn
    per_class_iou = jaccard_score(labels_flat, preds_flat, average=None, labels=[0, 1], zero_division=0)
    manhole_iou = per_class_iou[1]
    mean_iou = np.mean(per_class_iou)
    # # IoU
    # iou_result = mean_iou_metric.compute(
    #     predictions=preds,
    #     references=labels,
    #     num_labels=2,
    #     ignore_index=255,
    # )
    # manhole_iou = iou_result["per_category_iou"][1]  # class 1 = manhole

    return {
        "manhole_iou": float(manhole_iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "mean_iou": float(mean_iou),
        "obj_mA_f1": float(mAF1),
        "obj_mA_recall": float(mAR),
        "obj_mA_precision": float(mAP),
        "obj_mA_iou": float(mAIOU),
        "obj_results": obj_results,
        "avg_true_objects_per_img": float(total_true / batch_size),
        "avg_pred_objects_per_img": float(total_pred / batch_size)
    }







