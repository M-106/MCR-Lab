# -----------
# > Imports <
# -----------
import traceback
import sys
import os

import matplotlib.pyplot as plt

# import evaluate
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
from sklearn.metrics import auc
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

def evaluate_object_wise(preds, labels, confident_threshold, iou_threshold, ignore_index=255, debug_plot_path=None, using_heatmap_as_gt=False):
    valid_mask = (labels != ignore_index)

    # add here confidence threshold? -> at binarization?
    
    if using_heatmap_as_gt:
        preds_binary = ((preds >= confident_threshold) & valid_mask).astype(np.uint8)
        labels_binary = ((labels >= 0.5) & valid_mask).astype(np.uint8)
    else:
        # preds_binary = ((preds == 1) & valid_mask).astype(np.uint8)
        # labels_binary = ((labels == 1) & valid_mask).astype(np.uint8)
        # FIXME -> preds müssen nicht binarisiert übergeben werden!
        preds_binary = ((preds >= confident_threshold) & valid_mask).astype(np.uint8)
        labels_binary = ((labels >= 0.5) & valid_mask).astype(np.uint8)

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
    
    # if no obj exist, break
    if num_true_objects == 0 and num_pred_objects == 0:
        return {
            # "object_precision": 1.0,
            # "object_recall": 1.0,
            "tp_objects_count": 0,
            "fp_objects_count": 0,
            "fn_objects_count": 0, 
            "true_objects_count": 0, 
            "pred_objects_count": 0, 
            # "object_mean_iou": 1.0,
            # "object_pred_mean_iou": None
            "summed_object_iou": 0.0
        }
    elif num_true_objects == 0 and num_pred_objects >= 1:
        # every pred is false positive and IoU 0
        return {
            # "object_precision": 0.0,  # TP / (TP+FP) = 0/1
            # "object_recall": 1.0,    # TP / (TP+FN) -> 0/0 = 1.0
            "tp_objects_count": 0,
            "fp_objects_count": num_pred_objects,
            "fn_objects_count": 0,
            "true_objects_count": num_true_objects,
            "pred_objects_count": num_pred_objects,
            # "object_mean_iou": 0.0,
            # "object_pred_mean_iou": 0.0
            "summed_object_iou": 0.0
        }
    elif num_true_objects >= 1 and num_pred_objects == 0:
        # every pred is false negative and IoU 0
        return {
            # "object_precision": 1.0,  # TP / (TP+FP) -> 0/0 = 1.0
            # "object_recall": 0.0,     # TP / (TP+FN) -> 0/1 = 0.0
            "tp_objects_count": 0,
            "fp_objects_count": 0,
            "fn_objects_count": num_true_objects,
            "true_objects_count": num_true_objects,
            "pred_objects_count": num_pred_objects,
            # "object_mean_iou": 0.0,
            # "object_pred_mean_iou": 1.0  # not found FN, does not influence only prediction iou
            "summed_object_iou": 0.0
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
    # all_obj_ious = []
    # pred_ious = []
    summed_object_iou = 0.0
    
    for t_i, p_i in zip(true_ind, pred_ind):
        iou = 1.0 - cost_matrix[t_i, p_i]

        # a real amtching only makes sense if there is any overlap
        if iou > iou_threshold:  # NEW -> add also match threshold?
            # all_obj_ious.append(iou)
            matched_gts.add(t_i + 1)
            matched_preds.add(p_i + 1)
            
            # if iou >= iou_tp_threshold:
            tp_objects += 1
            # pred_ious.append(iou)
            summed_object_iou += iou

    # 4. unmachted GTs (False NEgatives= gets 0 IoU)
    num_fn_objects = num_true_objects - len(matched_gts)
    # FIXME -> is that right?? Or should be removed? -> IoU is localization not Detection!
    # all_obj_ious.extend([0.0] * num_fn_objects)

    # 5. Unmachted Preds (False Positives = gets 0 IoU)
    num_fp_objects = num_pred_objects - len(matched_preds)
    # all_obj_ious.extend([0.0] * num_fp_objects)
    # pred_ious.extend([0.0] * num_fp_objects)

    # mean_obj_iou = np.mean(all_obj_ious) if len(all_obj_ious) > 0 else 0.0
    # mean_obj_pred_iou = np.mean(pred_ious) if len(pred_ious) > 0 else 0.0

    # obj_precision = tp_objects / max(tp_objects+num_fp_objects, 1)
    # obj_recall = tp_objects / max(tp_objects+num_fn_objects, 1)

    return {
        # "object_precision": obj_precision,
        # "object_recall": obj_recall,
        "tp_objects_count": tp_objects,
        "fp_objects_count": num_fp_objects,
        "fn_objects_count": num_fn_objects,
        "true_objects_count": num_true_objects,
        "pred_objects_count": num_pred_objects,
        "summed_object_iou": summed_object_iou
        # "object_mean_iou": float(mean_obj_iou),
        # "object_pred_mean_iou": float(mean_obj_pred_iou)
    }


def get_mean_from_multiple_results(value_name, obj_results, choosen_iou_threshold=0.5, confident_threshold_min=0.5):
    
    # create accumulation vars
    acc_tp = 0
    acc_fp = 0
    acc_fn = 0
    
    acc_matches_for_iou = 0
    acc_summed_iou = 0

    acc_other_metrics = []

    # fill the accumulator
    for cur_obj_result in obj_results:
        iou_threshold = cur_obj_result["iou_threshold"]
        confident_threshold = cur_obj_result["confident_threshold"]
        total_iou_sum = cur_obj_result["total_iou_sum"]
        total_tp = cur_obj_result["total_tp"]
        total_fp = cur_obj_result["total_fp"]
        total_fn = cur_obj_result["total_fn"]

        if iou_threshold == choosen_iou_threshold and confident_threshold >= confident_threshold_min:
            acc_tp += total_tp
            acc_fp += total_fp
            acc_fn += total_fn

            if total_tp > 0:
                acc_summed_iou += total_iou_sum
                acc_matches_for_iou += total_tp
                
            if value_name not in ["avg_obj_precision", "avg_obj_recall", "avg_obj_iou", "avg_f1"]:
                acc_other_metrics.append(cur_obj_result[value_name])

    # return result
    if value_name == "avg_obj_precision":
        return acc_tp / max(acc_tp + acc_fp, 1)
    elif value_name == "avg_obj_recall":
        return acc_tp / max(acc_tp + acc_fn, 1)
    elif value_name == "avg_f1":
        precision = acc_tp / max(acc_tp + acc_fp, 1)
        recall = acc_tp / max(acc_tp + acc_fn, 1)
        return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    elif value_name == "avg_obj_iou":
        if acc_matches_for_iou == 0:
            return np.nan
        return acc_summed_iou / acc_matches_for_iou
        
    else:
        return np.nanmean(np.array(other_metrics, dtype=np.float32))


def compute_metrics(preds, labels, 
                    confident_threshold_start=0.0, confident_threshold_end=0.95, confident_threshold_step=0.05, 
                    iou_threshold_start=0.5, iou_threshold_end=0.55, iou_threshold_step=0.05, 
                    ignore_index=255, using_heatmap_as_gt=False):
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
    avg_true_objects_per_img = 0.0
    avg_pred_objects_per_img = 0.0

    for cur_confident_threshold in np.arange(confident_threshold_start, confident_threshold_end+confident_threshold_step, confident_threshold_step):
        for cur_iou_threshold in np.arange(iou_threshold_start, iou_threshold_end+iou_threshold_step, iou_threshold_step):
            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_true = 0
            total_pred = 0
            total_iou_sum = 0

            for batch_idx in range(batch_size):
                plot_path = None

                # >>> compute object metrics <<<
                obj_metrics = evaluate_object_wise(
                    preds=preds[batch_idx], 
                    labels=labels[batch_idx], 
                    confident_threshold=cur_confident_threshold,
                    iou_threshold=cur_iou_threshold,
                    ignore_index=ignore_index,
                    debug_plot_path=plot_path,
                    using_heatmap_as_gt=using_heatmap_as_gt
                )
                
                total_tp += obj_metrics["tp_objects_count"]
                total_fp += obj_metrics["fp_objects_count"]
                total_fn += obj_metrics["fn_objects_count"]
                total_true += obj_metrics["true_objects_count"]
                total_pred += obj_metrics["pred_objects_count"]
                total_iou_sum += obj_metrics["summed_object_iou"]
            
            obj_results.append({
                "confident_threshold": cur_confident_threshold,
                "iou_threshold": cur_iou_threshold,
                "total_iou_sum": total_iou_sum,
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn
            })

            if len(obj_results) == 1:
                # we only want to execute this code once
                avg_true_objects_per_img += float(total_true / batch_size)  
                avg_pred_objects_per_img += float(total_pred / batch_size) 

    # get mA results
    mAP = get_mean_from_multiple_results("avg_obj_precision", obj_results, choosen_iou_threshold=0.5, confident_threshold_min=0.5)
    mAR = get_mean_from_multiple_results("avg_obj_recall", obj_results, choosen_iou_threshold=0.5, confident_threshold_min=0.5)
    mAF1 = get_mean_from_multiple_results("avg_f1", obj_results, choosen_iou_threshold=0.5, confident_threshold_min=0.5)
    mAIOU = get_mean_from_multiple_results("avg_obj_iou", obj_results, choosen_iou_threshold=0.5, confident_threshold_min=0.5)

    # >>> compute pixel metrics <<<
    # create mask for ignroe index
    mask = labels != ignore_index

    print(f"Debug preds shape: {preds.shape}")
    print(f"Debug labels shape: {labels.shape}")

    preds_binarized = (preds >= 0.5).astype(np.uint8)
    labels_binarized = (labels >= 0.5).astype(np.uint8)
    # if using_heatmap_as_gt:
    preds_flat = preds_binarized[mask].flatten()
    labels_flat = labels_binarized[mask].flatten()
    # else:
    #     preds_flat = preds[mask].flatten()
    #     labels_flat = labels[mask].flatten()
    # -> Flatten for sklearn metrics (exclude ignore_index)

    print(f"Debug preds_flat shape: {preds_flat.shape}")
    print(f"Debug labels_flat shape: {labels_flat.shape}")

    f1 = f1_score(labels_flat, preds_flat, pos_label=1, zero_division=0)
    precision = precision_score(labels_flat, preds_flat, pos_label=1, zero_division=0)
    recall = recall_score(labels_flat, preds_flat, pos_label=1, zero_division=0)

    # calc iou via sklearn
    per_class_iou = jaccard_score(labels_flat, preds_flat, average=None, labels=[0, 1], zero_division=0)
    manhole_iou = per_class_iou[1] if len(per_class_iou) > 1 else per_class_iou[0]
    mean_iou = np.mean(per_class_iou)

    # most likely for heatmap - mean absolute error
    mae_score = np.abs(preds[mask] - labels[mask]).mean()

    return {
        "mae_score": float(mae_score),
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
        "avg_true_objects_per_img": avg_true_objects_per_img,
        "avg_pred_objects_per_img": avg_pred_objects_per_img
    }






