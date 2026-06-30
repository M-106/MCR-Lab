# -----------
# > Imports <
# -----------
import os
import shutil
from datetime import datetime
from functools import partial

from tqdm import tqdm
from transformers import (Trainer as HFTrainer, 
                         TrainingArguments as HFTrainingArguments, 
                         SegformerForSemanticSegmentation, 
                         Mask2FormerForUniversalSegmentation, 
                         OneFormerForUniversalSegmentation,
                         # DeepLabV3ForSemanticSegmentation,
                         SegformerImageProcessor,
                         AutoImageProcessor)
                         #TrainerCallBack as HFTrainerCallBack
from transformers import get_cosine_schedule_with_warmup
from transformers.modeling_outputs import SemanticSegmenterOutput

import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, AdamW

# for plotting
import numpy as np
import matplotlib.pyplot as plt
from transformers import TrainerCallback, TrainerState, TrainerControl

from mcrlab.config.config import Config
from mcrlab.log import get_logger, LoggerPrinter
from mcrlab.point_cloud.data import get_data_loader, get_basic_transform, BEVDataset
from mcrlab.model_utils import get_model, get_device, get_criterion
from mcrlab.metrices import compute_metrics
from mcrlab.custom_hf.unet import UnetForSemanticSegmentation, UnetConfig



# ----------------------
# > HuggingFace Helper <
# ----------------------

class ImagePlottingCallback(TrainerCallback):
    def __init__(self, val_dataset, model_name, processor, config, save_name, num_samples=1, pre_name="", clear_path=True, batch_size=5, ignore_index=255, is_heatmap_gt=False):
        super().__init__()
        self.val_dataset = val_dataset
        self.model_name = model_name.lower()
        self.processor = processor
        self.config = config
        self.num_samples = num_samples
        self.pre_name = pre_name
        self.batch_size = batch_size
        self.ignore_index = ignore_index
        self.is_heatmap_gt = is_heatmap_gt
        
        # create folderfor saving

        self.plot_dir = f"./output/plots/{save_name}"
        # if save_post_dir_name is not None:
        #     self.plot_dir += f"_{save_post_dir_name}"
        os.makedirs(self.plot_dir, exist_ok=True)
        if clear_path:
            shutil.rmtree(self.plot_dir)
            os.makedirs(self.plot_dir, exist_ok=True)

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """
        Calls after every evaluation
        """
        if model is None:
            return

        was_training = model.training
        model.eval()
        device = next(model.parameters()).device

        plotted_samples = 0

        # go through the first x samples
        with torch.no_grad():
            for i in range(len(self.val_dataset)):
                sample = self.val_dataset[i]
                
                # Prepare inputs -> add batch dim + move to device
                pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
                labels = sample["labels"]  # Ground Truth (bereits als Tensor im Dataset)

                # make prediction
                outputs = model(pixel_values=pixel_values)
                
                # extract target_sizes if needed (for Mask2Former/OneFormer?)
                # print(f"EVAL:\nlabels:\n   type: {type(labels)}\n    len: {len(labels)}\n    shape: {len(labels.shape)}")
                # print(f"Label [0]:\n   type: {type(labels[0])}\n    len: {len(labels[0])}\n    shape: {len(labels[0].shape)}")
                # # [print(f"\n    - Output {idx}: {cur_label.shape}, dtype={cur_label.dtype}") for idx, cur_label in enumerate(labels[0])]
                # print(f"Label [1]:\n   type: {type(labels[1])}\n    len: {len(labels[1])}\n    shape: {len(labels[1].shape)}")
                # # [print(f"\n    - Output {idx}: {cur_label.shape}, dtype={cur_label.dtype}") for idx, cur_label in enumerate(labels[1])]
                # # print(f"{labels}")
                # print(f"\nOutputs:\n   type: {type(outputs)}\n    len: {len(outputs)}")

                if self.model_name in ["mask2former", "oneformer"]:
                    # size = np.array(labels[0][1].shape[-2:])
                    # target_sizes = np.repeat(size*4, self.batch_size).reshape((-1, 2))
                    target_sizes = [(500, 500)] # * self.batch_size
                else:
                    # target_sizes = None
                    target_sizes = [(500, 500)] # * self.batch_size
                # target_sizes = [labels.shape[-2:]] if self.model_name in ["mask2former", "oneformer"] else None

                # if target_sizes is not None:
                #     print(f"target len: {len(target_sizes)}")
                
                preds = get_segmentation_prediction(
                    outputs,
                    model_name=self.model_name,
                    processor=self.processor,
                    target_sizes=target_sizes,
                    is_heatmap_gt=self.is_heatmap_gt
                )
                
                input_img = pixel_values[0].cpu().numpy().transpose(1, 2, 0)
                
                if input_img.shape[-1] == 3:
                    # if input_img.max() > 1:
                    #     print(f"[WARNING] Found value bigger than 1 ({input_img.max()}), will clip it away for visualization.")
                    # if input_img.min() < 0:
                    #     print(f"[WARNING] Found value smaller than 0 ({input_img.min()}), will clip it away for visualization.")
                    
                    img_min = input_img.min()
                    img_max = input_img.max()
                    if img_max > img_min:
                        input_to_show = (input_img - img_min) / (img_max - img_min)
                    else:
                        input_to_show = np.zeros_like(input_img)
                    input_to_show = np.clip(input_to_show, 0, 1)
                else:
                    # raise ValueError("Debuggign Stop, expected image to have 3 channels")
                    input_to_show = input_img[:, :, 0]

                gt_mask = labels.cpu().numpy().squeeze()
                # torch.as_tensor(labels)
                # print(f"preds before: {preds.shape}")
                pred_mask = preds[0].cpu().numpy().squeeze()

                # values, counts = np.unique(gt_mask, return_counts=True)
                # print("\n--- Unique Values and Frequency ---")
                # for val, count in zip(values, counts):
                #     print(f"Value: {val:<5} | Pixel Amount: {count:,}")

                heatmap_threshold = 0.5 if gt_mask.max() <= 1.0 else 127.5
                if self.is_heatmap_gt:
                    heatmap_threshold = 0.5 if gt_mask.max() <= 1.0 else 127.5 # Falls Werte 0-255 sind
                
                    if np.sum(gt_mask >= heatmap_threshold) < 50:
                        continue
                else:
                    if np.sum(gt_mask == 1) < 25:
                        continue

                # "remove" ignore label, so it does not hinder the plot
                missing_point_idx_mask = gt_mask == self.ignore_index
                gt_mask = np.where(gt_mask == self.ignore_index, 0, gt_mask)

                # print(f"[DEBUG] GT unique values: {np.unique(gt_mask)} | Pred unique values: {np.unique(pred_mask)}")

                # # scale values
                # if gt_mask.max() <= 1:
                #     gt_mask *= 255

                # if pred_mask.max() <= 1:
                #     pred_mask *= 255

                # print("\n=== DEBUGGING PRINT ===")
                # print("pixel_values:", pixel_values.shape)
                # print("outputs.logits:", outputs.logits.shape)
                # print("gt_mask:", gt_mask.shape)
                # print("pred_mask:", pred_mask.shape)
                # preds before: torch.Size([1, 125, 125])      

                # === DEBUGGING PRINT ===
                # pixel_values: torch.Size([1, 3, 500, 500])
                # outputs.logits: torch.Size([1, 2, 125, 125])
                # gt_mask: (500, 500)
                # pred_mask: (125, 125) 

                # create plot
                if self.is_heatmap_gt:
                    cols = 5
                else:
                    cols = 6
                fig, axes = plt.subplots(1, cols, figsize=(5*cols, 5))
                
                cur_idx = 0
                # Image 1 - Input Channel Max Height
                axes[cur_idx].imshow(input_to_show[:, :, 0], cmap="viridis")
                axes[cur_idx].set_title("Input Max height")
                axes[cur_idx].axis("off")

                # Image 2 - Input Channel Intensity
                cur_idx += 1
                axes[cur_idx].imshow(input_to_show[:, :, 1], cmap="viridis")
                axes[cur_idx].set_title("Input Intensity")
                axes[cur_idx].axis("off")

                # Image 3 - Input Channel Density
                cur_idx += 1
                axes[cur_idx].imshow(input_to_show[:, :, 2], cmap="viridis")
                axes[cur_idx].set_title("Input Density")
                axes[cur_idx].axis("off")

                # Image 4 - Ground Truth
                cur_idx += 1
                if self.is_heatmap_gt:
                    vmax = 1  # 255
                else:
                    vmax = 1
                axes[cur_idx].imshow(gt_mask, cmap='viridis', vmin=0, vmax=vmax)
                axes[cur_idx].set_title("Ground Truth")
                axes[cur_idx].axis("off")

                # Image 5 - Prediction
                cur_idx += 1
                # print(f"DEBUGGING preds shape: {pred_mask.shape}")
                pred_mask[missing_point_idx_mask] = 0
                axes[cur_idx].imshow(pred_mask, cmap='viridis', vmin=0, vmax=vmax)
                axes[cur_idx].set_title(f"Prediction")
                axes[cur_idx].axis("off")

                # FIXME -> also difference? -> be careful when substracting because of dtype

                # Image 5 - Difference
                if not self.is_heatmap_gt:
                    cur_idx += 1
                    diff = (gt_mask != pred_mask).astype(np.uint8)
                    # diff = np.abs(gt_mask.astype(np.int64) - pred_mask.astype(np.int64))
                    axes[cur_idx].imshow(diff, cmap='viridis', vmin=0, vmax=1)
                    axes[cur_idx].set_title(f"Difference")
                    axes[cur_idx].axis("off")

                fig.suptitle(f"Sample Prediction Epoch {state.epoch:.1f}")

                save_path = os.path.join(self.plot_dir, f"{self.pre_name}_epoch_{state.epoch:03}_step_{state.global_step:03}_sample_{i:03}.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)

                plotted_samples += 1

                if plotted_samples >= self.num_samples:
                    break
    
        if was_training:
            model.train()



# ----------
# > Helper <
# ----------
def save_model(model, dir_path, name="model"):
    os.makedirs(dir_path, exist_ok=True)

    if not name.endswith(".pt"):
        name += ".pt"

    torch.save(model.state_dict(), os.path.join(dir_path, name))

def get_optimizer(optimizer_name, model, lr):
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_scaler(name):
    if name:
        return torch.amp.GradScaler()
    return None

def get_scheduler(name, optimizer):
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return None



# ------------------------------
# > HuggingFace Train Pipeline <
# ------------------------------
class LogKeysCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print("\n" + "="*50)
            print("Available metrics for the trainer:")
            for key in sorted(metrics.keys()):
                print(f"   - {key}")
            print("="*50 + "\n")

class CustomSegmentationTrainer(HFTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Setup Dice + Focal Loss for binary segmentation (sigmoid=True)
        # focal_weight can be adjusted if you want to balance class weights further
        # self.loss_fn = DiceFocalLoss(
        #     sigmoid=True, 
        #     squared_pred=True, 
        #     reduction="mean"
        # )

        self.dice_loss = smp.losses.DiceLoss(mode="multiclass", ignore_index=255)
        self.focal_loss = smp.losses.FocalLoss(mode="multiclass", ignore_index=255)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forward pass
        outputs = model(**inputs)
        
        # Hugging Face models typically output logits shape: [batch_size, num_classes, height, width]
        # For binary segmentation, num_classes is usually 1 or 2 depending on your setup.
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        labels = inputs.get("labels")

        # Ensure labels have the same floating type and shape matches logits
        # Monai expects shapes like [B, C, H, W]. 
        # If your labels are [B, H, W], unsqueeze the channel dimension:
        if len(labels.shape) == 3:
            labels = labels.unsqueeze(1)
            
        loss = 0.5 * self.focal_loss(logits, labels) + \
               0.5 * self.dice_loss(logits, labels)

        # return (loss, outputs) if return_outputs else loss
        return SemanticSegmenterOutput(loss=loss, logits=logits)



# Model registry: name -> (model_class, default_checkpoint)
MODEL_REGISTRY = {
    "segformer":   (SegformerForSemanticSegmentation,        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"),
    "mask2former": (Mask2FormerForUniversalSegmentation,     "facebook/mask2former-swin-large-cityscapes-semantic"),
    "oneformer":   (OneFormerForUniversalSegmentation,       "shi-labs/oneformer_cityscapes_swin-l_160k"),
    "unet":        (UnetForSemanticSegmentation,             "resnet34"),
    # "deeplabv3":   (DeepLabV3ForSemanticSegmentation,        "microsoft/deeplabv3-resnet-101"),
    # -> model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
}

# processor size config per model
PROCESSOR_SIZE = {
    "segformer":   {"height": 500, "width": 500},
    "mask2former": {"height": 500, "width": 500},  # {"shortest_edge": 500},
    "oneformer":   {"height": 500, "width": 500},  # {"shortest_edge": 500},
    "unet":        {"height": 500, "width": 500},
    # "deeplabv3":   {"height": 500, "width": 500},
}



def get_model_and_processor(model_name, check_point_path=None, num_labels=2,
                                image_mean=[0.485, 0.456, 0.406],
                                image_std=[0.229, 0.224, 0.225],
                                mode="train",
                                ignore_index=255,
                                heatmap_is_gt=False):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}")

    # MODEL EXTRACTION
    # ------------
    # extract model class and checkpoitn/default loading
    model_class, default_checkpoint = MODEL_REGISTRY[model_name]
    # checkpoint = check_point_path or default_checkpoint
    checkpoint = check_point_path if check_point_path is not None else default_checkpoint

    # MODEL LOADING
    # ------------
    if model_name == "unet":
        # use wrapper
        config = UnetConfig(num_labels=num_labels, ignore_index=ignore_index, heatmap_is_gt=heatmap_is_gt)
        model = model_class(config, encoder_weights="imagenet" if not check_point_path else None)
    
    if not (mode == "train" and model_name == "unet"):
        model = model_class.from_pretrained(
            checkpoint,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
    model.config.ignore_index = ignore_index
    model.config.num_labels = num_labels

    # PROCESSOR
    # ------------
    if model_name == "unet":
        processor_source = "nvidia/mit-b0"
    else:
        processor_source = default_checkpoint if check_point_path else checkpoint
    processor = AutoImageProcessor.from_pretrained(
        processor_source,
        do_resize=True,
        size=PROCESSOR_SIZE[model_name],
        size_divisibility=0,  # important, else it will round to the next vielfache
        do_rescale=False,
        do_normalize=False,  # FIXME check if that change something
        image_mean=image_mean,
        image_std=image_std,
    )

    return model, processor



def get_segmentation_prediction(outputs, model_name, processor=None, target_sizes=None, is_heatmap_gt=False):
    model_name = model_name.lower()

    is_numpy_input = isinstance(outputs, np.ndarray)
    is_torch_input = isinstance(outputs, torch.Tensor)

    if model_name in ["segformer", "deeplabv3", "unet"]:
        if not is_numpy_input and hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs
        
        # raise RuntimeError(f"DEBUGGING STOP, logits/pred shape: {logits.shape}\nMin-Max ({logits.min()} - {logits.max()})")
        # if isinstance(logits, np.ndarray):
        #     preds = logits.argmax(axis=1)
        # else:
        #     preds = logits.argmax(dim=1)

        if isinstance(logits, np.ndarray):
            logits_tensor = torch.from_numpy(logits)
        else:
            logits_tensor = logits.detach().cpu()

        

        # upscaling because: SegFormer logits are 1/4 of input size
        if model_name in ["segformer", "unet"] and hasattr(outputs, "logits") and processor is not None and not is_heatmap_gt:
            preds_list = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
            preds = torch.stack(preds_list)
        else:
            if is_heatmap_gt:
                preds = torch.sigmoid(logits_tensor)
            else:
                preds = logits_tensor.argmax(dim=1) # (B, W, H)

            if target_sizes is not None:
                size = tuple(int(x) for x in target_sizes[0])
        
                # Interpolate braucht 4D: (B, 1, H, W)
                preds = preds if is_heatmap_gt else preds.unsqueeze(1).float()
                mode = "bilinear" if is_heatmap_gt else "nearest"
                preds = F.interpolate(preds, size=size, mode=mode)
                if not is_heatmap_gt:
                    preds = preds.squeeze(1).long()
                else:
                    preds = preds.squeeze(1)

    elif model_name in ["mask2former", "oneformer"]:

        # print(f"\noutputs Len: {len(outputs)} Dtype: {type(outputs)}\nSub-Element type: {type(outputs[0])}\nShapes:")
        # for cur_idx, cur_elem in enumerate(outputs):
        #     if hasattr(cur_elem, "shape"):
        #         shape_str = f"{cur_elem.shape}"
        #     else:
        #         shape_str = "none"
        #     print(f"  - {cur_idx:02}:\n      dtype={type(cur_elem)}\n      shape={shape_str}")
        # raise ValueError("DEBUGGING STOP")

        if is_numpy_input or is_torch_input:
            if is_numpy_input:
                logits_tensor = torch.from_numpy(outputs)
            else:
                logits_tensor = outputs

            if is_heatmap_gt:
                preds = torch.sigmoid(logits_tensor)
            else:
                preds = logits_tensor.argmax(dim=1)
        
            if target_sizes is not None:
                size = tuple(int(x) for x in target_sizes[0])
                preds = preds if is_heatmap_gt else preds.unsqueeze(1).float()
                mode = "bilinear" if is_heatmap_gt else "nearest"
                preds = F.interpolate(preds, size=size, mode=mode)
                if not is_heatmap_gt:
                    preds = preds.squeeze(1).long()
                else:
                    preds = preds.squeeze(1)
        elif isinstance(outputs, tuple):
            # debugging
            # print(type(outputs), len(outputs) if isinstance(outputs, tuple) else outputs.shape)
            # for i, o in enumerate(outputs):
            #     print(f"  outputs[{i}]: shape={o.shape}, dtype={o.dtype}")

            # raise ValueError("DEBUGING STOP")
            # <class 'tuple'> 52/3 [00:01<00:00,  1.18it/s]
            # outputs[0]: shape=(22, 100, 3), dtype=float32
            # outputs[1]: shape=(22, 100, 128, 128), dtype=float32
            # outputs[2]: shape=(22, 1536, 16, 16), dtype=float32
            # outputs[3]: shape=(22, 256, 128, 128), dtype=float32
            # outputs[4]: shape=(22, 100, 256), dtype=float32

            from transformers.models.mask2former.modeling_mask2former import Mask2FormerForUniversalSegmentationOutput
            outputs_obj = Mask2FormerForUniversalSegmentationOutput(
                class_queries_logits=torch.from_numpy(outputs[0]),
                masks_queries_logits=torch.from_numpy(outputs[1]),
            )

            # Post-processing liefert eine Liste von PyTorch-Tensoren
            preds_list = processor.post_process_semantic_segmentation(
                outputs_obj,
                target_sizes=target_sizes
            )
            preds = torch.stack(preds_list)
        else:
            if processor is None:
                raise ValueError(f"Processor muss für {model_name} übergeben werden! (maybe activate code below)")

            if target_sizes is None or len(target_sizes) != outputs.class_queries_logits.shape[0]:
                current_batch_size = outputs.class_queries_logits.shape[0]
                target_sizes = [(500, 500)] * current_batch_size

            preds_list = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=target_sizes
            )
            preds = torch.stack(preds_list)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # if not right size
    # pred_mask_resized = cv2.resize(pred_mask, (500, 500), interpolation=cv2.INTER_NEAREST)

    # preds = torch.argmax(outputs.logits, dim=1)
    # preds = preds.unsqueeze(1).float() 
    # preds_upsampled = F.interpolate(preds, size=(500, 500), mode="nearest").long()
    # pred_mask = preds_upsampled.squeeze().cpu().numpy()

    return preds
    # return preds if is_numpy_input else preds.numpy()


def train_hf_pipeline(config):
    print("GPU available:", torch.cuda.is_available())
    
    if not torch.cuda.is_available():
        raise RuntimeError("Does not find GPU accelerator!")

    batch_size = config.train.batch_size

    model_name = config.model.name.lower()

    heatmap_path = config.data.heatmap_path
    used_heatmap_channel = config.data.used_heatmap_channel
    using_heatmap_as_gt = not(heatmap_path is None)

    ignore_index = 255
    num_labels = 2
    metric_for_best_model = "eval_manhole_iou"
    greater_is_better = True
    if using_heatmap_as_gt:
        ignore_index = -999
        num_labels = 1
        metric_for_best_model = "eval_mae_score"
        greater_is_better = False
        is_heatmap_gt = True

    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute

    heatmap_gt_str = f"heatmapGT{used_heatmap_channel}_" if heatmap_path else ""
    save_name = f"{year}_{month:02}_{day:02}_{hour:02}_{minute:02}_{model_name}_{config.data.name}_{heatmap_gt_str}{config.train.exp_name}"


    # FIXME
    # for cur_file in os.listdir("./output/"):
    #     cur_file_path = os.path.join("./output/", cur_file)
    #     if os.path.isdir(cur_file_path):
    # try:
    #     shutil.rmtree("./output/plots")
    #     shutil.rmtree("./output/checkpoints")
    # except Exception as e:
    #     print(e)

    checkpoint_path = config.model.check_point_path
    if checkpoint_path == "None":
        checkpoint_path = None
    model, processor = get_model_and_processor(model_name, checkpoint_path, num_labels=num_labels, mode="train", ignore_index=ignore_index, heatmap_is_gt=using_heatmap_as_gt)

    # Load Data

    pass_label_in_preprocessor = model_name in ["mask2former", "oneformer"]
    if config.data.name == "merged_whu_sud":
        train_dataset = get_data_loader(
            "whu", 
            config.data.path, 
            type="train", 
            transform=get_basic_transform(),
            batch_size=config.train.batch_size, 
            shuffle=True, 
            num_workers=4,
            preprocessed=True, 
            return_train_format=True,
            return_dataset=True
        )
        all_train_paths = train_dataset.point_cloud_paths

        train_dataset = get_data_loader(
            "sud", 
            config.data.path_2, 
            type="train", 
            transform=get_basic_transform(),
            batch_size=config.train.batch_size, 
            shuffle=True, 
            num_workers=4,
            preprocessed=True, 
            return_train_format=True,
            return_dataset=True
        )
        all_train_paths.extend(train_dataset.point_cloud_paths)
    else:
        train_dataset = get_data_loader(
            config.data.name, 
            config.data.path, 
            type="train", 
            transform=get_basic_transform(),
            batch_size=config.train.batch_size, 
            shuffle=True, 
            num_workers=4,
            preprocessed=True, 
            return_train_format=True,
            return_dataset=True
        )
        all_train_paths = train_dataset.point_cloud_paths
    train_dataset = BEVDataset(path=all_train_paths, 
                               file_paths=[], 
                               has_labels=True, 
                               image_training=True, 
                               preprocessor=processor,
                               augment=True,
                               pass_label_in_preprocessor=pass_label_in_preprocessor,
                               heatmap_gt_path=heatmap_path,
                               used_heatmap_channel=used_heatmap_channel)
    train_dataset.manhole_filter(required_manhole_points=50, amount_non_manhole_samples=10)
    
    if config.data.name == "merged_whu_sud":
        val_dataset = get_data_loader(
            "whu", 
            config.data.path, 
            type="val", 
            transform=get_basic_transform(),
            batch_size=config.train.batch_size, 
            shuffle=False, 
            num_workers=4,
            preprocessed=True, 
            return_train_format=True,
            return_dataset=True
        )
        all_val_paths = val_dataset.point_cloud_paths

        val_dataset = get_data_loader(
            "sud", 
            config.data.path_2, 
            type="val", 
            transform=get_basic_transform(),
            batch_size=config.train.batch_size, 
            shuffle=False, 
            num_workers=4,
            preprocessed=True, 
            return_train_format=True,
            return_dataset=True
        )
        all_val_paths.extend(val_dataset.point_cloud_paths)
    else:
        val_dataset = get_data_loader(config.data.name, 
                                    config.data.path, 
                                    type="val", 
                                    transform=get_basic_transform(),
                                    batch_size=config.train.batch_size, 
                                    shuffle=False, 
                                    num_workers=4,
                                    preprocessed=True, 
                                    return_train_format=True,
                                    return_dataset=True)
        all_val_paths = val_dataset.point_cloud_paths
    val_dataset = BEVDataset(path=all_val_paths, 
                             file_paths=[], 
                             has_labels=True, 
                             image_training=True, 
                             preprocessor=processor,
                             augment=False,
                             pass_label_in_preprocessor=pass_label_in_preprocessor,
                             heatmap_gt_path=heatmap_path,
                             used_heatmap_channel=used_heatmap_channel)
    val_dataset.manhole_filter(required_manhole_points=50)
    # config.data.preprocessed

    # Callback for in-between sample plotting
    plotting_callback = ImagePlottingCallback(
        val_dataset=val_dataset,
        save_name=save_name,
        processor=processor,
        model_name=model_name,
        config=config,
        num_samples=5,
        pre_name="finetuning",
        batch_size=batch_size,
        ignore_index=ignore_index,
        is_heatmap_gt=using_heatmap_as_gt
    )

    # Helper Functions
    def collate_fn(batch, model_name):
        pixel_values = torch.stack([x["pixel_values"] for x in batch])
        
        if model_name in ["mask2former", "oneformer"]:
            return {
                "pixel_values": pixel_values,
                "mask_labels": [x["mask_labels"] for x in batch],
                "class_labels": [x["class_labels"] for x in batch]
            }
        else:
            labels = torch.stack([x["labels"] for x in batch])
            # print(f"\n\n=== DEBUGGING PRINT === (collate_fn)\nLabels Shape: {labels.shape}")
            return {
                "pixel_values": pixel_values, 
                "labels": labels
            }

    def compute_metrics_fn(eval_pred, model_name, processor, batch_size, ignore_index=255, is_heatmap_gt=False):
        if hasattr(eval_pred, "predictions") and hasattr(eval_pred, "label_ids"):
            # raise ValueError("This path is unexpected might need to revert? or use model name to handle!")
            outputs = eval_pred.predictions
            labels = eval_pred.label_ids

            # print(f"\n\n=== DEBUGGING PRINT === (compute_metrics_fn)\nLabels Shape: {labels.shape}")
            # FIXME labels already have 500, 500

            # print("Output/Preds:", type(outputs), len(outputs) if isinstance(outputs, tuple) else outputs.shape)
            # for i, o in enumerate(outputs):
            #     print(f"  outputs[{i}]: shape={o.shape}, dtype={o.dtype}")


            # print(f"Labels:\n   type: {type(labels)}\n    len: {len(labels)}")
            # print(f"Label [0]:\n   type: {type(labels[0])}\n    len: {len(labels[0])}")
            # [print(f"    - Output {idx}: {cur_label.shape}, dtype={cur_label.dtype}") for idx, cur_label in enumerate(labels[0])]
            # print(f"Label [1]:\n   type: {type(labels[1])}\n    len: {len(labels[1])}")
            # [print(f"    - Output {idx}: {cur_label.shape}, dtype={cur_label.dtype})") for idx, cur_label in enumerate(labels[1])]
            # print(f"{labels}")
            # print(labels[1])

            # print(f"Pred:\n   type: {type(outputs)}\n    len: {len(outputs)}")
            # print(f"Label [0]:\n   type: {type(outputs[0])}\n    len: {len(outputs[0])}")
            # [print(f"    - Output {idx}: {cur_out.shape}, dtype={cur_out.dtype}") for idx, cur_out in enumerate(outputs[0])]
            # print(f"Label [1]:\n   type: {type(outputs[1])}\n    len: {len(outputs[1])}")
            # [print(f"    - Output {idx}: {cur_out.shape}, dtype={cur_out.dtype})") for idx, cur_out in enumerate(outputs[1])]

                # Labels:
                # type: <class 'tuple'>
                #     len: 2
                # Label [0]:
                # type: <class 'list'>
                #     len: 8
                #     - Output 0: (7, 512, 512), dtype=float32
                #     - Output 1: (7, 512, 512), dtype=float32
                #     - Output 2: (7, 512, 512), dtype=float32
                #     - Output 3: (7, 512, 512), dtype=float32
                #     - Output 4: (7, 512, 512), dtype=float32
                #     - Output 5: (7, 512, 512), dtype=float32
                #     - Output 6: (6, 512, 512), dtype=float32
                #     - Output 7: (6, 512, 512), dtype=float32
                # Label [1]:
                # type: <class 'list'>
                #     len: 8
                #     - Output 0: (7,), dtype=int64)
                #     - Output 1: (7,), dtype=int64)
                #     - Output 2: (7,), dtype=int64)
                #     - Output 3: (7,), dtype=int64)
                #     - Output 4: (7,), dtype=int64)
                #     - Output 5: (7,), dtype=int64)
                #     - Output 6: (6,), dtype=int64)
                #     - Output 7: (6,), dtype=int64)
                # Preds:
                # type: <class 'tuple'>
                #     len: 2
                # Pred [0]:
                # type: <class 'numpy.ndarray'>
                #     len: 32
                #     - Output 0: (100, 3), dtype=float32
                #     - Output 1: (100, 3), dtype=float32
                #     - ...
                # Pred [1]:
                # type: <class 'numpy.ndarray'>
                #     len: 32
                #     - Output 0: (100, 128, 128), dtype=float32)
                #     - Output 1: (100, 128, 128), dtype=float32)
                #     - ...

            if model_name in ["segformer", "unet"]:
                # target_sizes = [labels.shape[-2:]] * labels.shape[0]
                target_sizes = [(500, 500)] * labels.shape[0]  # batch_size  # [label.shape[-2:] for label in labels]

                preds = get_segmentation_prediction(
                    outputs,
                    model_name=model_name,
                    processor=processor,
                    target_sizes=target_sizes,
                    is_heatmap_gt=is_heatmap_gt
                )
            else:
                # raise ValueError("DEBUGGING STOP: did not expect to go here...")
                # unpack labels and get true number of pictures
                mask_labels_list = labels[0]
                class_labels_list = labels[1]
                num_real_images = len(mask_labels_list)

                aggregated_batch_size = outputs[0].shape[0] if isinstance(outputs, tuple) else outputs.shape[0]
                target_sizes = [(500, 500)] * aggregated_batch_size

                # slicing preds to match labels
                # if isinstance(outputs, (list, tuple)):
                #     cls_logits = outputs[0][:num_images]
                #     mask_logits = outputs[1][:num_images]
                #     outputs = (cls_logits, mask_logits)
                # else:
                #     outputs = outputs[:num_images]

                preds = get_segmentation_prediction(
                    outputs,
                    model_name=model_name,
                    processor=processor,
                    target_sizes=target_sizes,
                    is_heatmap_gt=is_heatmap_gt
                )

                # drop padded elements, right??
                preds = preds[:num_real_images]

                semantic_labels = []
                for masks, classes in zip(mask_labels_list, class_labels_list):
                    H, W = masks.shape[1], masks.shape[2]
                    sem = np.zeros((H, W), dtype=np.int64)
                    for mask, cls in zip(masks, classes):
                        sem[mask > 0.5] = cls
                    semantic_labels.append(sem)
                labels = np.stack(semantic_labels)

        else:
            outputs, labels = eval_pred

            # target_sizes = None
            target_sizes = [(500, 500)] * batch_size

            preds = get_segmentation_prediction(
                outputs,
                model_name=model_name,
                processor=processor,
                target_sizes=target_sizes,
                is_heatmap_gt=is_heatmap_gt
            )

        return compute_metrics(preds=preds, labels=labels, ignore_index=ignore_index, is_heatmap_gt=is_heatmap_gt)

    # FIXME -> make many of the settings adjustable via config
    training_args = HFTrainingArguments(
        output_dir=f"./output/checkpoints/{save_name}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=1e-4,  # 6e-5,           # 6e-5     
        lr_scheduler_type="cosine",   # cosine
        warmup_steps=200,             # 0.1
        fp16=False,                    # faster training
        gradient_accumulation_steps=4,
        num_train_epochs=400,   
        dataloader_num_workers=4,

        eval_strategy="steps",
        eval_steps=int( len(train_dataset)/batch_size ),
        save_strategy="steps",
        save_steps=int( len(train_dataset)/batch_size ),

        save_total_limit=2,

        load_best_model_at_end=True,  # aadd the end saves the best ckpt
        metric_for_best_model=metric_for_best_model, # also possible "eval_iou" / "eval_f1"
        greater_is_better=greater_is_better,

        logging_steps=10,
        remove_unused_columns=False,   # important for SAM
        push_to_hub=False,
        report_to=["tensorboard", "mlflow"],  # "none"
        use_cpu=False
    )

    optimizer = SGD(
        model.parameters(), 
        lr=1e-3, 
        momentum=0.9, 
        weight_decay=1e-4
    )

    # optimizer = AdamW(
    #     model.parameters(), 
    #     lr=1e-3,
    #     betas=(0.9, 0.999),
    #     eps=1e-8,
    #     weight_decay=0.01
    # )

    total_train_steps = int((len(train_dataset) / batch_size) * training_args.num_train_epochs)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=200,
        num_training_steps=total_train_steps
    )

    trainer = HFTrainer(    # CustomSegmentationTrainer(  # HFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # load bev/meta files and extract in right format -> use BEV Dataset
        eval_dataset=val_dataset,
        # train_dataset=train_dataset.select(range(2)) if hasattr(train_dataset, 'select') else train_dataset,
        # eval_dataset=val_dataset.select(range(2)) if hasattr(val_dataset, 'select') else val_dataset,
        data_collator=partial(collate_fn, model_name=model_name),
        optimizers=(optimizer, lr_scheduler),
        compute_metrics=partial(
            compute_metrics_fn,
            model_name=model_name,
            processor=processor,
            batch_size=batch_size,
            ignore_index=ignore_index, 
            is_heatmap_gt=is_heatmap_gt
        ),
        callbacks=[plotting_callback],  # , LogKeysCallback()
        # preprocess_logits_for_metrics=lambda logits, labels: logits[:2] if model_name in ["mask2former", "oneformer"] else None,  # only keep class + mask logits
        preprocess_logits_for_metrics=lambda logits, labels: logits[:2] if model_name in ["mask2former", "oneformer"] else logits,
    )

    trainer.train()
    print("Success! Your Training is finish and your pipeline works.")

    # ------------------------------------
    # --> POST TRAINING <--

    # robustness finetuning
    # danger: Catastrophic Forgetting

    # print("Start post training.")

    # train_dataset = BEVDataset(path=all_train_paths, 
    #                            file_paths=[], 
    #                            has_labels=True, 
    #                            image_training=True, 
    #                            preprocessor=processor,
    #                            augment=True)
    # val_dataset = BEVDataset(path=all_val_paths, 
    #                          file_paths=[], 
    #                          has_labels=True, 
    #                          image_training=True, 
    #                          preprocessor=processor,
    #                          augment=False)

    # plotting_callback = ImagePlottingCallback(
    #     val_dataset=val_dataset,
    #     model_name=model_name,
    #     processor=processor,
    #     config=config,
    #     num_samples=5,
    #     pre_name="post_training", 
    #     clear_path=True,
    #     save_post_dir_name="post_training",
    #     batch_size=batch_size
    # )

    # training_args = HFTrainingArguments(
    #     output_dir=f"./output/checkpoints/{config.model.name}_post_training",
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=batch_size,
    #     learning_rate=5e-6,           # 6e-5     
    #     lr_scheduler_type="cosine",   # cosine
    #     warmup_steps=0,             # 0.1
    #     fp16=False,                    # faster training
    #     gradient_accumulation_steps=4,
    #     num_train_epochs=100,   
    #     dataloader_num_workers=4,
    #     eval_strategy="steps",
    #     eval_steps=int( len(train_dataset)/batch_size ) * 2,
    #     save_strategy="steps",  # or best?
    #     save_steps=int( len(train_dataset)/batch_size ) * 2,
    #     save_total_limit=2,
    #     logging_steps=10,
    #     remove_unused_columns=False,   # important for SAM
    #     push_to_hub=False,
    #     report_to=["tensorboard", "mlflow"],  # "none"
    #     use_cpu=False
    # )

    # trainer = HFTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,  # load bev/meta files and extract in right format -> use BEV Dataset
    #     eval_dataset=val_dataset,
    #     # train_dataset=train_dataset.select(range(2)) if hasattr(train_dataset, 'select') else train_dataset,
    #     # eval_dataset=val_dataset.select(range(2)) if hasattr(val_dataset, 'select') else val_dataset,
    #     data_collator=collate_fn,
    #     compute_metrics=partial(
    #         compute_metrics_fn,
    #         model_name=model_name,
    #         processor=processor
    #     ),
    #     callbacks=[plotting_callback]
    # )

    # trainer.train()

    # print("Post Training is finish!")



# -----------------------
# > Main Train Function <
# -----------------------
def train(config):
    # model_name = config.model.name

    train_hf_pipeline(config)

    # if model_name.lower() in ["sam2", "sam3", "segformer", "dinomask2former"]:
    #     train_hf_pipeline(config)
    # else:
    #     raise RuntimeError("Only Huggingface Pipeline is available right now")
    #     train_pipeline(config)











