# -----------
# > Imports <
# -----------
import os
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sklearn.metrics import roc_curve, auc  # , f1_score, precision_score, recall_score, jaccard_score

from transformers import (Trainer as HFTrainer, 
                         TrainingArguments as HFTrainingArguments)

from mcrlab.config.config import Config
# from mcrlab.model_utils import get_model, get_device, get_criterion, \
#                                TorchModelWrapper, \
#                                compute_loss, match_with_thresholding, \
#                                compute_metrics
from mcrlab.execution.train import get_model_and_processor, get_segmentation_prediction
from mcrlab.metrices import compute_metrics
from mcrlab.point_cloud.data import get_data_loader, get_basic_transform, BEVDataset
        


# -----------
# > Testing <
# -----------
# class Tester:
#     def __init__(self, model, dataloader, loss_fns, device=None):
#         self.model = model
#         self.dataloader = dataloader
#         self.loss_fns = loss_fns
#         self.device = get_device(device)

#     def evaluate(self):
#         self.model.eval()

#         total_losses = [0.0 for _ in self.loss_fns]
#         # maybe other metrices?

#         batches = len(self.dataloader)

#         # with torch.no_grad():
#         with torch.inference_mode():
#             for batch_idx, (x_batch, y_batch) in tqdm(enumerate(self.dataloader), desc="Testing", total=batches):
#                 # moving data to device is not done
#                 # because of geometry models
#                 predictions = self.model.predict(x_batch)

#                 matches, unmatched_pred, unmatched_gt = match_with_thresholding(pred=predictions,
#                                                                                 target=y_batch,
#                                                                                 max_dist=0.5)

#                 # calc loss 
#                 for loss_idx, loss_fn in enumerate(self.loss_fns):
#                     if isinstance(loss_fn, str) and loss_fn in ["recall", "precision", "f1", "f2"]:
#                         cur_loss = compute_metrics(matches, unmatched_pred, unmatched_gt)[loss_fn]
#                     else:
#                         cur_loss = compute_loss(loss_fn=loss_fn,
#                                                 pred=predictions,
#                                                 target=y_batch,
#                                                 matches=matches,
#                                                 unmatched_gt=unmatched_gt,
#                                                 unmatched_pred=unmatched_pred,
#                                                 lambda_fn=1.0,
#                                                 lambda_fp=0.5)
#                     total_losses[loss_idx] += cur_loss.item()

#         # calc mean
#         mean_losses = [cur_loss / max(batches, 1) for cur_loss in total_losses]
#         return mean_losses



def get_data_for_plot(obj_results, 
                      fix_metric_name="iou_threshold", fix_value=0.5, 
                      variable_metric_name="confident_threshold",
):
    """
    Filtert die Daten so, dass eine Metrik fixiert bleibt (z.B. iou_threshold = 0.5)
    und die andere Metrik über die X-Achse läuft.
    """
    # filter after threshols
    filtered = [r for r in obj_results if r[fix_metric_name] == fix_value]

    # sort for clean line
    filtered.sort(key=lambda x: x[variable_metric_name])

    x_thresholds = [r[variable_metric_name] for r in filtered]
    
    # directly calc here, we only have the micro aggregations here
    AP = []
    AR = []
    AIOU = []
    FP = []

    for r in filtered:
        tp, fp, fn = r["total_tp"], r["total_fp"], r["total_fn"]
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        iou = r["total_iou_sum"] / max(tp, 1) if tp > 0 else 0.0
        
        AP.append(precision)
        AR.append(recall)
        AIOU.append(iou)
        FP.append(fp)

    return np.array(x_thresholds), np.array(AP), np.array(AR), np.array(AIOU), np.array(FP)



def plot_pixel_roc_and_save(fpr, tpr, auc_score, title, save_path):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))

    # Plot ROC line
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'Pixel ROC (AUC = {auc_score:.4f})')
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')

    # Set titles and labels
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR / Sensitivity)', fontsize=12)

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', fontsize=11)

    # plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)



def plot_froc_and_save(fp, recall, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(fp, recall, label='Instance Detection', color='darkorange', linewidth=2, marker='o')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('False Positive Instances (FP)', fontsize=12)
    ax.set_ylabel('Recall / TPR (TP / Ground Truth)', fontsize=12)

    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', fontsize=11)

    # plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)



def plot_and_save(iou_thresholds, AR, AP, AIOU, title, xlabel, save_path):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))

    # plot values
    ax.plot(iou_thresholds, AR, label='Avg Recall (AR)', marker='o', linewidth=2)
    ax.plot(iou_thresholds, AP, label='Avg Precision (AP)', marker='s', linewidth=2)
    # ax.plot(iou_thresholds, AF1, label='Avg F1-Score (AF1)', marker='^', linewidth=2)
    if AIOU is not None:
        ax.plot(iou_thresholds, AIOU, label='Avg Object IoU', marker='d', linewidth=2)

    # set titles and labels
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)

    # set range for these metrics
    ax.set_ylim(0, 1.05)

    # add grid and legend
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower left', fontsize=11)

    # save and close fig
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)



def plot_mean_average(object_results, root_save_path, save_name):
    # 1. Confidence Curve (Fixed IoU Threshold on 0.5, variable Confident Threshold)
    save_path1 = os.path.join(root_save_path, f"confidence_curve_{save_name}.png")
    
    conf_thresholds, AP, AR, AIOU, _ = get_data_for_plot(
        object_results, 
        fix_metric_name="iou_threshold", 
        fix_value=0.5, 
        variable_metric_name="confident_threshold"
    )

    plot_and_save(
        iou_thresholds=conf_thresholds, 
        AR=AR, 
        AP=AP, 
        AIOU=AIOU,
        title="Metrics vs. Confidence Threshold (Fix IoU @ 0.5)",
        xlabel='Confidence Threshold',
        save_path=save_path1
    )

    # 2. IoU Matching Sensitivity Curve (Fixed Confident Threshold on 0.5, variable IoU Threshold)
    save_path2 = os.path.join(root_save_path, f"matching_sensitivity_curve_{save_name}.png")
    
    iou_thresholds, AP, AR, AIOU, _ = get_data_for_plot(
        object_results, 
        fix_metric_name="confident_threshold", 
        fix_value=0.5, 
        variable_metric_name="iou_threshold"
    )

    plot_and_save(
        iou_thresholds=iou_thresholds, 
        AR=AR, 
        AP=AP, 
        AIOU=AIOU,
        title="Metrics vs. IoU Matching Threshold (Fix Conf @ 0.5)",
        xlabel='IoU Matching Threshold',
        save_path=save_path2
    )

    # # 3.F ROC Curve (Fix IoU @ 0.5)
    # save_path3 = os.path.join(root_save_path, f"froc_curve_{save_name}.png")

    # conf_thresholds, AP, AR, AIOU, FP = get_data_for_plot(
    #     object_results, 
    #     fix_metric_name="iou_threshold", 
    #     fix_value=0.5, 
    #     variable_metric_name="confident_threshold"
    # )

    # plot_froc_and_save(
    #     fp=FP, 
    #     recall=AR, 
    #     title="FROC Curve (Fix IoU @ 0.5)", 
    #     save_path=save_path3
    # )





def evaluate_hf_pipeline(config, use_all_test_data, use_testset_1=True, print_out_results=True):
    print("Evaluating on device:", "GPU" if torch.cuda.is_available() else "CPU")
    
    model_name = config.model.name.lower()
    encoder_name = config.model.encoder
    batch_size = config.test.batch_size  # or maybe set a specific eval_batch_size?

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
    checkpoint_path = config.model.check_point_path
    if not checkpoint_path or checkpoint_path == "None":
        raise ValueError("Please provide the path to your trained checkpoint in config.model.check_point_path")

    print(f"Loading trained model and processor from: {checkpoint_path}")
    model, processor = get_model_and_processor(model_name, encoder_name, checkpoint_path, mode="test", num_labels=num_labels, ignore_index=ignore_index, heatmap_is_gt=using_heatmap_as_gt)

    if use_testset_1:
        data_name = config.data.name
        data_path = config.data.path
    else:
        data_name = config.data.name_2
        data_path = config.data.path_2

    parts = Path(checkpoint_path).parts
    exp_name = parts[-2]
    exp_name += f"_on_{data_name}"
    if use_all_test_data:
        exp_name += "_all"

    output_dir = f"./tmp_eval/{exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load Test Data
    heatmap_path = config.data.heatmap_path
    used_heatmap_channel = config.data.used_heatmap_channel
    pass_label_in_preprocessor = model_name in ["mask2former", "oneformer"]
    normalization = config.data.normalization
    normalization_mode = config.data.normalization_mode
    
    if config.data.name == "merged_whu_sud":
        test_loader_raw = get_data_loader(
            "whu", 
            config.data.path, 
            type="test",
            transform=get_basic_transform(),
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            preprocessed=True, 
            return_train_format=True,
            return_dataset=True,
        )
        all_test_paths = test_loader_raw.point_cloud_paths

        test_loader_raw = get_data_loader(
            config.data.name_2, 
            config.data.path_2, 
            type="test",
            transform=get_basic_transform(),
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            preprocessed=True, 
            return_train_format=True,
            return_dataset=True,
        )
        all_test_paths.extend(test_loader_raw.point_cloud_paths)
    else:
        test_loader_raw = get_data_loader(
            data_name, 
            data_path, 
            type="test",
            transform=get_basic_transform(),
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            preprocessed=True, 
            return_train_format=True,
            return_dataset=True,
        )
        
        all_test_paths = test_loader_raw.point_cloud_paths

    test_dataset = BEVDataset(
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
    
    # Filter identical to training to keep metric comparisons fair
    if not use_all_test_data:
        test_dataset.manhole_filter(required_manhole_points=55.5)
    print(f"Loaded {len(test_dataset)} testing samples.")

    # Define Helper Functions (reused from your training codebase)
    def collate_fn(batch):
        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        if model_name in ["mask2former", "oneformer"]:
            return {
                "pixel_values": pixel_values,
                "mask_labels": [x["mask_labels"] for x in batch],
                "class_labels": [x["class_labels"] for x in batch]
            }
        else:
            return {
                "pixel_values": pixel_values, 
                "labels": torch.stack([x["labels"] for x in batch])
            }

    # def compute_metrics_fn(eval_pred, model_name, processor, batch_size):
    #     if hasattr(eval_pred, "predictions") and hasattr(eval_pred, "label_ids"):
    #         outputs = eval_pred.predictions
    #         labels = eval_pred.label_ids
    #         batch_size = outputs[0].shape[0] if isinstance(outputs, tuple) else outputs.shape[0]
    #         target_sizes = [(500, 500)] * batch_size
    #     else:
    #         outputs, labels = eval_pred
    #         target_sizes = None

    #     preds = get_segmentation_prediction(
    #         outputs,
    #         model_name=model_name,
    #         processor=processor,
    #         target_sizes=target_sizes
    #     )
    #     return compute_metrics(preds=preds, labels=labels)

    def compute_metrics_fn(eval_pred, model_name, processor, batch_size, ignore_index=255, using_heatmap_as_gt=False, as_prob=True):
        if hasattr(eval_pred, "predictions") and hasattr(eval_pred, "label_ids"):
            outputs = eval_pred.predictions
            labels = eval_pred.label_ids
            
            # dynamic batch size calc
            aggregated_batch_size = outputs[0].shape[0] if isinstance(outputs, tuple) else outputs.shape[0]
            target_sizes = [(500, 500)] * aggregated_batch_size
            
            # generate prediction
            preds = get_segmentation_prediction(
                outputs,
                model_name=model_name,
                processor=processor,
                target_sizes=target_sizes,
                as_prob=as_prob
            )
            
            # --- MASK2FORMER / ONEFORMER SPECIFIC LABEL-PROCESSING ---
            if model_name in ["mask2former", "oneformer"]:
                mask_labels_list = labels[0]
                class_labels_list = labels[1]
                num_real_images = len(mask_labels_list)
                
                # slicing preds if padding-dummies are there
                # FIXME -> is that right? or do they mean something else?
                preds = preds[:num_real_images]
                
                # retransformation from lists from bianry masks to semantic 2D images
                # the same as in train
                semantic_labels = []
                for masks, classes in zip(mask_labels_list, class_labels_list):
                    # masks Form: (Obj-Amount, H, W)
                    H, W = masks.shape[1], masks.shape[2]
                    sem = np.zeros((H, W), dtype=np.int64)
                    for mask, cls in zip(masks, classes):
                        # if as_prob:
                        #     # FIXME -> how to do?
                        #     sem[mask > 0.5] = mask
                        # else:
                        sem[mask > 0.5] = cls
                    semantic_labels.append(sem)
                
                labels = np.stack(semantic_labels)
                
        else:
            # Fallback if directly a tuple is given
            outputs, labels = eval_pred
            target_sizes = [(500, 500)] * labels.shape[0] if hasattr(labels, "shape") else None
            
            preds = get_segmentation_prediction(
                outputs,
                model_name=model_name,
                processor=processor,
                target_sizes=target_sizes,
                as_prob=as_prob
            )

        # convert to numpy & cpu
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        return compute_metrics(
            preds=preds, 
            labels=labels, 
            ignore_index=ignore_index, 
            using_heatmap_as_gt=using_heatmap_as_gt,
            confident_threshold_start=0.0, confident_threshold_end=0.96, confident_threshold_step=0.05, 
            iou_threshold_start=0.0, iou_threshold_end=0.96, iou_threshold_step=0.05, 
        )

    # Initialize Trainer for Evaluation Only
    eval_args = HFTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=batch_size,
        remove_unused_columns=False,
        report_to="none" # Turn off mlflow/tensorboard for clean terminal outputs
    )

    trainer = HFTrainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        compute_metrics=partial(
            compute_metrics_fn,
            model_name=model_name,
            processor=processor,
            batch_size=batch_size,
            ignore_index=ignore_index, 
            using_heatmap_as_gt=using_heatmap_as_gt
        ),
        preprocess_logits_for_metrics=lambda logits, labels: logits[:2] if model_name in ["mask2former", "oneformer"] else logits,
        # preprocess_logits_for_metrics=lambda logits, labels: logits[:2] if model_name in ["mask2former", "oneformer"] else None,
    )

    # Run Quantitative Evaluation
    print("Running quantitative evaluation...")
    # results = trainer.predict(test_dataset)
    results = trainer.evaluate(test_dataset)
    
    output_lines = []
    output_lines.append("="*40)
    output_lines.append(f"QUANTITATIVE TEST RESULTS ({model_name.upper()})")
    output_lines.append("="*40)
    
    for metric_name, value in results.items():
        # remove prefix 'test_' or 'eval_'
        if isinstance(value, list) and not any([name_ in metric_name for name_ in ["pixel_fpr", "pixel_tpr"]]):
            for cur_obj_result in value:
                line = f"Object Mean Average Values - IoU Threshold: {cur_obj_result["iou_threshold"]:.2f} & Confident Threshold: {cur_obj_result["confident_threshold"]:.2f}"
                output_lines.append(line)
                for cur_obj_result_name, cur_obj_result_value in cur_obj_result.items():
                    clean_name = cur_obj_result_name.replace("test_", "").replace("eval_", "")
                    line = f"      - {clean_name:<30}: {cur_obj_result_value:.4f}" if isinstance(cur_obj_result_value, float) else f"      - {clean_name:<30}: {cur_obj_result_value}"
                    output_lines.append(line)
        else:
            clean_name = metric_name.replace("test_", "").replace("eval_", "")
            line = f"{clean_name:<30}: {value:.4f}" if isinstance(value, float) else f"{clean_name:<30}: {value}"
            output_lines.append(line)
        
    output_lines.append("="*40)
    
    # print
    text_content = "\n".join(output_lines)
    if print_out_results:
        print(text_content)

    # save in file
    # now = datetime.now()
    # year = now.year
    # month = now.month
    # day = now.day
    # hour = now.hour
    # minute = now.minute

    # Save Text Metrics
    save_name = exp_name  # f"{year}_{month:02}_{day:02}_{hour:02}_{minute:02}_{model_name}"
    
    os.makedirs(eval_args.output_dir, exist_ok=True)
    txt_path = os.path.join(eval_args.output_dir, f"test_metrics_{save_name}.txt")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_content)
        
    print(f"Metrics successfully saved to: {txt_path}")

    # Plot Object-Level Mean Average Curves & FROC
    plot_mean_average(results["eval_obj_results"], root_save_path=eval_args.output_dir, save_name=save_name)
    
    # Plot Pixel-Level ROC Curve
    if "eval_pixel_fpr" in results and "eval_pixel_tpr" in results:
        pixel_roc_path = os.path.join(eval_args.output_dir, f"pixel_roc_curve_{save_name}.png")
        plot_pixel_roc_and_save(
            fpr=results["eval_pixel_fpr"],
            tpr=results["eval_pixel_tpr"],
            auc_score=results["eval_pixel_auc"],
            title=f"Pixel-Level ROC Curve ({model_name.upper()})",
            save_path=pixel_roc_path
        )

    return results



def predict_single_sample(model, model_name, processor, image, device="cuda"):

    model.to(device)
    model.eval()

    # if isinstance(image, np.ndarray):
    #     image = torch.from_numpy(image)
    
    # if isinstance(image, torch.Tensor):
    #     image = image.float()
    #     # Add batch dimension if missing (e.g., [C, H, W] -> [1, C, H, W])
    #     if image.ndim == 3:
    #         image = image.unsqueeze(0)
    #     image = image.to(model.device)

    # print(f"Input Image Type: {image.dtype}")

    # Preprocess
    # Adjust inputs based on your specific model requirements
    if processor is not None:
        inputs = processor(images=image, return_tensors="pt").to(device)
    else:
        inputs = image.to(device)

    # print(f"Input Processed Image Type: {inputs.dtype}")

    # Inference
    with torch.no_grad():
        outputs = model(inputs)

    # Post-process (using your existing helper)
    if isinstance(image, torch.Tensor):
        h, w = image.shape[-2], image.shape[-1]
        target_sizes = [(h, w)]
    else:
        target_sizes = [image.size[::-1]]

    prediction = get_segmentation_prediction(
        outputs, 
        model_name=model_name,
        processor=processor, 
        target_sizes=target_sizes
    )

    return prediction


def test(config):

    encoder_to_short = {
        "timm-efficientnet-b7": "teb7", 
        "mit_b5": "mb5", 
        "resnet101": "r101", 
        "resnext101_32x32d": "r101_32x32d", 
        "densenet161": "d161", 
        "mobileone_s4": "mos4",
        "tu-samvit_huge_patch16.sa1b": "tshp16s", 
        "tu-maxvit_xlarge_tf_512": "tmxt512", 
        "tu-beit_large_patch16_512.in22k_ft_in22k_in1k": "tblp16512"
    }

    names = [
        "unet",
        # "fpn", 
        # "deeplabv3", 
        # "deeplabv3plus", 
        #"dpt"
    ]
    encoders = [
        ["timm-efficientnet-b7"]
        # ["resnext101_32x32d"], 
        # ["timm-efficientnet-b7", "mit_b5", "resnet101", "resnext101_32x32d", "densenet161", "mobileone_s4"], 
        # ["timm-efficientnet-b7", "resnet101", "resnext101_32x32d", "mobileone_s4"],
        # ["timm-efficientnet-b7", "mit_b5", "resnet101", "resnext101_32x32d", "mobileone_s4"],
        # ["tu-samvit_huge_patch16.sa1b", "tu-maxvit_xlarge_tf_512", "tu-beit_large_patch16_512.in22k_ft_in22k_in1k"]
    ]
    checkpoint_paths = [
        ["./output/checkpoints/2026_07_17_22_10_unet_whu_comparison_1_teb7/checkpoint-702"],
        # ["./output/checkpoints/2026_08_11_09_41_unet_sud_comparison_1_teb7_domainfinetuned/checkpoint-196"],
        # ["./output/checkpoints/2026_08_11_18_37_unet_merged_whu_sud_comparison_1_teb7_domainfinetuned_merged/checkpoint-308"],
        # ["./output/checkpoints/2026_08_11_21_32_unet_whu_comparison_1_teb7_improved_norm/checkpoint-1411"],
        # ["./output/checkpoints/2026_08_12_11_05_unet_whu_comparison_1_teb7_improved_norm_other_channels_minmax/checkpoint-1394"],
        # ["./output/checkpoints/2026_07_18_11_59_unet_whu_comparison_1_r101_32x32d/checkpoint-962"],
        # ["./output/checkpoints/2026_07_19_12_06_fpn_whu_comparison_1_teb7/checkpoint-819",
        #  "./output/checkpoints/2026_07_19_13_13_fpn_whu_comparison_1_mb5/checkpoint-546",
        #  "./output/checkpoints/2026_07_19_14_44_fpn_whu_comparison_1_r101/checkpoint-637",
        #  "./output/checkpoints/2026_07_19_15_28_fpn_whu_comparison_1_r101_32x32d/checkpoint-1027",
        #  "./output/checkpoints/2026_07_19_18_11_fpn_whu_comparison_1_d161/checkpoint-1066",
        #  "./output/checkpoints/2026_07_19_19_07_fpn_whu_comparison_1_mos4/checkpoint-1066"],
        # ["./output/checkpoints/2026_07_19_20_08_deeplabv3_whu_comparison_1_teb7/checkpoint-5141",
        #  "./output/checkpoints/2026_07_20_05_37_deeplabv3_whu_comparison_1_r101/checkpoint-1479",
        #  "./output/checkpoints/2026_07_20_10_43_deeplabv3_whu_comparison_1_r101_32x32d/checkpoint-3869",
        #  "./output/checkpoints/2026_07_20_17_59_deeplabv3_whu_comparison_1_mos4/checkpoint-1428"],
        #  ["./output/checkpoints/2026_07_20_20_23_deeplabv3plus_whu_comparison_1_teb7/checkpoint-4240",
        #  "./output/checkpoints/2026_07_21_04_34_deeplabv3plus_whu_comparison_1_mb5/checkpoint-1190",
        #  "./output/checkpoints/2026_07_21_06_33_deeplabv3plus_whu_comparison_1_r101/checkpoint-1003",
        #  "./output/checkpoints/2026_07_21_08_27_deeplabv3plus_whu_comparison_1_r101_32x32d/checkpoint-1898",
        #  "./output/checkpoints/2026_07_21_12_16_deeplabv3plus_whu_comparison_1_mos4/checkpoint-663"],
        #  ["./output/checkpoints/2026_07_21_13_59_dpt_whu_comparison_1_tshp16s/checkpoint-1248",
        #  "./output/checkpoints/2026_07_22_16_38_dpt_whu_comparison_1_tmxt512/checkpoint-4134",
        #  "./output/checkpoints/2026_07_23_07_15_dpt_whu_comparison_1_tblp16512/checkpoint-510"],
    ]

    model_entries = zip(names, encoders, checkpoint_paths)

    # first check, if every configuration is right
    print("Start Testing")
    for name, encoders, ckpt_paths in model_entries:
        backbone_and_weights = zip(encoders, ckpt_paths)
        for encoder, path in backbone_and_weights:

            # sanity check
            if name not in path:
                raise ValueError(f"Model-Name ('{name}') not in ckpt path ('{path}') ")

            if encoder_to_short[encoder] not in path:
                raise ValueError(f"Encoder-Name ('{encoder_to_short[encoder]}') not in ckpt path ('{path}') ")

            if not os.path.exists(path):
                raise ValueError(f"Path does not exists: {path}")
    print("Passed all Tests. Start Evaluating now.")

    model_entries = zip(names, encoders, checkpoint_paths)
    for name, encoders, ckpt_paths in model_entries:
        backbone_and_weights = zip(encoders, ckpt_paths)
        for encoder, path in backbone_and_weights:

            config.model.name = name
            config.model.encoder = encoder
            config.model.check_point_path = path


            evaluate_hf_pipeline(config, use_all_test_data=False, use_testset_1=True, print_out_results=False)
            # evaluate_hf_pipeline(config, use_all_test_data=True, use_testset_1=True)

            evaluate_hf_pipeline(config, use_all_test_data=False, use_testset_1=False, print_out_results=False)
            # evaluate_hf_pipeline(config, use_all_test_data=True, use_testset_1=False)

    # # load model
    # model = get_model(config.model.name)

    # if isinstance(model, torch.nn.Module):
    #     model = TorchModelWrapper(model, config.device)

    # data_loader = get_data_loader(config.data.name, config.data.path, 
    #                                 testdata=True, 
    #                                 transform=get_basic_transform(num_points=-1), # get_basic_transform(num_points=-1),
    #                                 batch_size=config.test.batch_size, shuffle=False, num_workers=1,
    #                                 preprocessed=True,
    #                                 return_train_format=True)  # because we still want x and y
    
    # metrices = []
    # for metric_name in config.test.metrices:
    #     metrices.append(get_criterion(metric_name))

    # tester = Tester(model=model, 
    #                 device=config.device,
    #                 dataloader=data_loader,
    #                 loss_fns=metrices)
    # tester.evaluate()











