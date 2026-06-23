# -----------
# > Imports <
# -----------
import os
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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


def evaluate_hf_pipeline(config):
    print("Evaluating on device:", "GPU" if torch.cuda.is_available() else "CPU")
    
    model_name = config.model.name.lower()
    batch_size = config.test.batch_size  # or maybe set a specific eval_batch_size?

    # Load the TRAINED model checkpoint
    checkpoint_path = config.model.check_point_path
    if not checkpoint_path or checkpoint_path == "None":
        raise ValueError("Please provide the path to your trained checkpoint in config.model.check_point_path")

    print(f"Loading trained model and processor from: {checkpoint_path}")
    model, processor = get_model_and_processor(model_name, checkpoint_path)

    parts = Path(checkpoint_path).parts
    exp_name = parts[-2]

    # Load Test Data
    heatmap_path = config.data.heatmap_path
    used_heatmap_channel = config.data.used_heatmap_channel
    pass_label_in_preprocessor = model_name in ["mask2former", "oneformer"]
    
    test_loader_raw = get_data_loader(
        config.data.name, 
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
    test_dataset = BEVDataset(
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
    
    # Filter identical to training to keep metric comparisons fair
    test_dataset.manhole_filter(required_manhole_points=50)
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

    def compute_metrics_fn(eval_pred, model_name, processor, batch_size):
        if hasattr(eval_pred, "predictions") and hasattr(eval_pred, "label_ids"):
            outputs = eval_pred.predictions
            labels = eval_pred.label_ids
            batch_size = outputs[0].shape[0] if isinstance(outputs, tuple) else outputs.shape[0]
            target_sizes = [(500, 500)] * batch_size
        else:
            outputs, labels = eval_pred
            target_sizes = None

        preds = get_segmentation_prediction(
            outputs,
            model_name=model_name,
            processor=processor,
            target_sizes=target_sizes
        )
        return compute_metrics(preds=preds, labels=labels)

    # Initialize Trainer for Evaluation Only
    eval_args = HFTrainingArguments(
        output_dir="./tmp_eval",
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
            batch_size=batch_size
        ),
        preprocess_logits_for_metrics=lambda logits, labels: logits[:2] if model_name in ["mask2former", "oneformer"] else logits,
        # preprocess_logits_for_metrics=lambda logits, labels: logits[:2] if model_name in ["mask2former", "oneformer"] else None,  # only keep class + mask logits
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
        clean_name = metric_name.replace("test_", "").replace("eval_", "")
        line = f"{clean_name:<30}: {value:.4f}" if isinstance(value, float) else f"{clean_name:<30}: {value}"
        output_lines.append(line)
        
    output_lines.append("="*40)
    
    # print
    text_content = "\n".join(output_lines)
    print(text_content)

    # save in file
    # now = datetime.now()
    # year = now.year
    # month = now.month
    # day = now.day
    # hour = now.hour
    # minute = now.minute

    save_name = exp_name  # f"{year}_{month:02}_{day:02}_{hour:02}_{minute:02}_{model_name}"
    
    os.makedirs(eval_args.output_dir, exist_ok=True)
    txt_path = os.path.join(eval_args.output_dir, f"test_metrics_{save_name}.txt")
    
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text_content)
        
    print(f"Metrics successfully saved to: {txt_path}")

    return results


def test(config):

    evaluate_hf_pipeline(config)

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











