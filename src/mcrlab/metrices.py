# -----------
# > Imports <
# -----------
# import evaluate
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
from scipy.ndimage import label, binary_closing, generate_binary_structure
import torch

# mean_iou_metric = evaluate.load("mean_iou")



# ------------
# > Metrices <
# ------------
def evaluate_object_wise(preds, labels, iou_threshold=0.3, ignore_index=255):
    valid_mask = (labels != ignore_index)
    
    # only use/view manholes
    preds_binary = ((preds == 1) & valid_mask).astype(np.uint8)
    labels_binary = ((labels == 1) & valid_mask).astype(np.uint8)

    # get label clusters
    #    1. morphological closing -> close empty space between points in obj
    struct = generate_binary_structure(2, 2)  # 8 neighborhood structure

    preds_closed = binary_closing(preds_binary, structure=struct, iterations=4).astype(np.uint8)
    labels_closed = binary_closing(labels_binary, structure=struct, iterations=4).astype(np.uint8)

    labeled_preds, num_pred_objects = label(preds_closed)
    labeled_labels, num_true_objects = label(labels_closed)

    tp_objects = 0
    all_obj_ious = []
    used_pred_ids = set()  # tracking of every prediction which got matched with a GT
    matched_pred_ids = set()  # tracking obj ids we already matched
    # for preventing double counting
    # 2 obj first get matched and then it get decide whether they are good/bad iou -> tp/fp

    # go through every true object and check if there is a prediction for that
    for obj_idx in range(1, num_true_objects+1):
        true_mask = (labeled_labels == obj_idx)

        # intersection with preds
        pred_ids_in_target = labeled_preds[true_mask]
        # remove background
        pred_ids_in_target = pred_ids_in_target[pred_ids_in_target > 0]

        if len(pred_ids_in_target) > 0:
            # fitting -> if multiple pred objects cut the gt object, take the one with the most pixels intersection
            counts = np.bincount(pred_ids_in_target)
            best_pred_id = np.argmax(counts)

            # skip if already used this pred-id (OR then use the second obj intersection if available?)
            if best_pred_id in used_pred_ids:  # in matched_pred_ids:
                all_obj_ious.append(0)
                continue

            used_pred_ids.add(best_pred_id)

            # calc IoU
            pred_mask = (labeled_preds == best_pred_id)
            union = (true_mask | pred_mask).sum()
            intersection = (true_mask & pred_mask).sum()
            obj_iou = intersection / union if union > 0 else 0

            all_obj_ious.append(obj_iou)

            if obj_iou >= iou_threshold:
                tp_objects += 1
                matched_pred_ids.add(best_pred_id)
        else:
            all_obj_ious.append(0.0)

    num_fp_objects = num_pred_objects - len(used_pred_ids)
    all_obj_ious.extend([0.0] * max(0, num_fp_objects))

    mean_obj_iou = np.mean(all_obj_ious) if len(all_obj_ious) > 0 else 0.0

    # # calc classical obj metrices
    # mean_obj_iou = np.mean(all_obj_ious) if len(all_obj_ious) > 0 else 0.0

    # # fn_objects = num_true_objects - tp_objects
    # # fp_objects = max(0, num_pred_objects - tp_objects)

    # obj_recall = tp_objects / num_true_objects if num_true_objects > 0 else 0
    # obj_precision = tp_objects / num_pred_objects if num_pred_objects > 0 else 0
    # if (obj_precision + obj_recall) > 0:
    #     obj_f1 = (2 * obj_precision * obj_recall) / (obj_precision + obj_recall)
    # else:
    #     obj_f1 = 0.0

    return {
        "tp_objects_count": tp_objects,
        "true_objects_count": num_true_objects,
        "pred_objects_count": num_pred_objects,
        # "object_recall": float(obj_recall),     # How many of all manholes got found
        # "object_precision": float(obj_precision), # How many predicted manholes were really manholes
        # "object_f1": float(obj_f1),  # value which scores if all manholes got found and predicted manholes are really manholes          
        "object_mean_iou": float(mean_obj_iou)
    }



def compute_metrics(preds, labels):
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

    if preds.shape != labels.shape:
        raise ValueError(f"Shape mismatch! Preds shape is {preds.shape}, but Labels shape is {labels.shape}.")

    # print(f"\nDEBUG INFO:\n  - preds shape (eval): {preds.shape}\n  - labels shape: {labels.shape}")

    batch_size = preds.shape[0]

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
        # >>> compute object metrics <<<
        obj_metrics = evaluate_object_wise(
            preds=preds[batch_idx], 
            labels=labels[batch_idx], 
            iou_threshold=0.3,
            ignore_index=255
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
    mean_obj_iou = total_obj_iou / batch_size

    # Average the sample metrics over the batch
    # for key in aggregated_obj_metrics:
    #     aggregated_obj_metrics[key] /= batch_size

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
        "obj_f1": float(obj_f1),
        "obj_recall": float(obj_recall),
        "obj_precision": float(obj_precision),
        "obj_mean_iou": float(mean_obj_iou),
        "avg_true_objects_per_img": float(total_true / batch_size),
        "avg_pred_objects_per_img": float(total_pred / batch_size)
        # "obj_f1": aggregated_obj_metrics["object_f1"],
        # "obj_recall": aggregated_obj_metrics["object_recall"],
        # "obj_precision": aggregated_obj_metrics["object_precision"],
        # "obj_mean_iou": aggregated_obj_metrics["object_mean_iou"] 
    }









