# -----------
# > Imports <
# -----------
import evaluate
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

mean_iou_metric = evaluate.load("mean_iou")



# ------------
# > Metrices <
# ------------
def compute_metrics(preds, labels):
    # logits = eval_pred.predictions
    # labels = eval_pred.label_ids

    # preds = np.argmax(logits, axis=1)

    # IoU
    iou_result = mean_iou_metric.compute(
        predictions=preds,
        references=labels,
        num_labels=2,
        ignore_index=255,
    )
    manhole_iou = iou_result["per_category_iou"][1]  # class 1 = manhole

    # Flatten for sklearn metrics (exclude ignore_index)
    mask = labels != 255
    preds_flat = preds[mask].flatten()
    labels_flat = labels[mask].flatten()

    f1 = f1_score(labels_flat, preds_flat, pos_label=1, zero_division=0)
    precision = precision_score(labels_flat, preds_flat, pos_label=1, zero_division=0)
    recall = recall_score(labels_flat, preds_flat, pos_label=1, zero_division=0)

    return {
        "manhole_iou": manhole_iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mean_iou": iou_result["mean_iou"]
    }









