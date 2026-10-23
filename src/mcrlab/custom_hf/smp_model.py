# -----------
# > Imports <
# -----------
import os

import numpy as np
import torch
import torch.nn.functional as F

from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers import PreTrainedModel, PretrainedConfig
from safetensors import safe_open

import segmentation_models_pytorch as smp


# ----------------
# > Count Params <
# ----------------
def count_parameters_meta(model_fn, *args, **kwargs):
    """
    Instantiates a PyTorch/HF model on the 'meta' device.
    RAM & VRAM usage: ~0 MB.
    """
    with torch.device("meta"):
        model = model_fn(*args, **kwargs)
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params



def count_params_from_checkpoint(checkpoint_dir, frozen_prefixes=("encoder.", "backbone.")):
    """
    Counts total, trainable, and non-trainable parameters from disk with 0 MB RAM.
    
    Args:
        checkpoint_dir (str): Path containing model.safetensors or pytorch_model.bin.
        frozen_prefixes (tuple): Parameter prefixes that were frozen during training.
                                 Pass empty tuple () if the full model was fine-tuned.
    """
    sf_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    # Common non-trainable buffer suffixes across PyTorch/Hugging Face
    buffer_suffixes = ("running_mean", "running_var", "num_batches_tracked")

    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    if os.path.isfile(sf_path):
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                shape = f.get_slice(k).get_shape()
                numel = 1
                for d in shape:
                    numel *= d
                
                total_params += numel

                # Check if tensor is a persistent buffer or part of a frozen module
                is_buffer = k.endswith(buffer_suffixes)
                is_frozen = any(k.startswith(pfx) for pfx in frozen_prefixes)

                if is_buffer or is_frozen:
                    non_trainable_params += numel
                else:
                    trainable_params += numel

        # return {
        #     "total": total_params,
        #     "trainable": trainable_params,
        #     "non_trainable": non_trainable_params
        # }
        return (total_params, trainable_params, non_trainable_params)

    elif os.path.isfile(bin_path):
        state_dict = torch.load(bin_path, map_location="meta", mmap=True)
        for k, t in state_dict.items():
            numel = t.numel()
            total_params += numel

            is_buffer = k.endswith(buffer_suffixes)
            is_frozen = any(k.startswith(pfx) for pfx in frozen_prefixes)

            if is_buffer or is_frozen:
                non_trainable_params += numel
            else:
                trainable_params += numel

        # return {
        #     "total": total_params,
        #     "trainable": trainable_params,
        #     "non_trainable": non_trainable_params
        # }

        return (total_params, trainable_params, non_trainable_params)

    else:
        raise FileNotFoundError(f"No checkpoint file found in {checkpoint_dir}")


def run_count(config):
    heatmap_path = config.data.heatmap_path
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

    check_point_path = config.model.check_point_path
    checkpoint = check_point_path if check_point_path is not None else None
    
    if checkpoint is None: # mode == "train" or mode == "val":
        # raise ValueError("DEBUGGING ERROR: SHOULD GO into other branch!")
        config = ModelConfig(
            num_labels=num_labels, 
            ignore_index=ignore_index, 
            heatmap_is_gt=using_heatmap_as_gt, 
            model_name=config.model.name, 
            encoder_name=config.model.encoder
        )

        total, trainable = count_parameters_meta(
            ModelForSemanticSegmentation, 
            config=config, 
            encoder_weights=None
        )
    else:
        config = ModelConfig.from_pretrained(checkpoint)
        config.num_labels = num_labels
        config.ignore_index = ignore_index
        config.heatmap_is_gt = using_heatmap_as_gt

        total, trainable, not_trainable = count_params_from_checkpoint(checkpoint)
        # total, trainable = count_parameters_meta(
        #     ModelForSemanticSegmentation.from_pretrained, 
        #     pretrained_model_name_or_path=checkpoint,
        #     config=config,
        #     ignore_mismatched_sizes=True
        # )


    print(f"{checkpoint}")
    print(f"Total Params: {total:,} ({total / 1e6:.2f}M)")
    print(f"    -> Trainable Params: {trainable:,} ({trainable / 1e6:.2f}M)")
    print(f"    -> Not Trainable Params: {not_trainable:,} ({not_trainable / 1e6:.2f}M)\n")


# -------------
# > Get Model <
# -------------
def get_smp_model_builder(model_name, encoder_name, encoder_weights, num_labels, interpolation):
    if model_name == "unet":
        return smp.Unet(
            encoder_name=encoder_name,  # "resnet50", "efficientnet-b3" or "efficientnet-b4", "mit_b2" or "mit_b3"
            encoder_weights=encoder_weights, 
            classes=num_labels,
            decoder_interpolation=interpolation,
            activation=None,
            in_channels=3
        )
    elif model_name == "unetplusplus":
        return smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights, 
            classes=num_labels,
            decoder_interpolation=interpolation,
            activation=None,
            in_channels=3
        )
    elif model_name == "fpn":
        return smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights, 
            classes=num_labels,
            decoder_interpolation=interpolation,
            activation=None,
            in_channels=3
        )
    elif model_name == "pspnet":
        return smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights, 
            classes=num_labels,
            activation=None,
            in_channels=3
        )
    elif model_name == "deeplabv3":
        return smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights, 
            classes=num_labels,
            activation=None,
            in_channels=3
        )
    elif model_name == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights, 
            classes=num_labels,
            activation=None,
            in_channels=3
        )
    elif model_name == "linknet":
        return smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights, 
            classes=num_labels,
            activation=None,
            in_channels=3
        )
    elif model_name == "manet":
        return smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights, 
            classes=num_labels,
            decoder_interpolation=interpolation,
            activation=None,
            in_channels=3
        )
    elif model_name == "pan":
        return smp.PAN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights, 
            classes=num_labels,
            decoder_interpolation=interpolation,
            activation=None,
            in_channels=3
        )
    elif model_name == "upernet":
        return smp.UPerNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights, 
            classes=num_labels,
            activation=None,
            in_channels=3
        )
    elif model_name == "segformer":
        return smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights, 
            classes=num_labels,
            activation=None,
            in_channels=3
        )
    elif model_name == "dpt":
        return smp.DPT(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights, 
            classes=num_labels,
            activation=None,
            in_channels=3,
            #dynamic_img_size=True
        )
    else:
        raise ValueError(f"Unknown SMP Model Name '{model_name}'")



# ---------
# > Model <
# ---------
class ModelConfig(PretrainedConfig):
    model_type = "smp_model"
    def __init__(self, num_labels=2, ignore_index=255, heatmap_is_gt=False, model_name="unet", encoder_name="resnet34", **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.heatmap_is_gt = heatmap_is_gt
        self.model_name = model_name
        self.encoder_name = encoder_name

        # ModelConfig.model_type = model_name



class ModelForSemanticSegmentation(PreTrainedModel):
    config_class = ModelConfig
    models_with_512_io = [
        "unetplusplus", 
        "fpn", 
        "deeplabv3", 
        "deeplabv3plus", 
        "dpt"
    ]
    
    def __init__(self, config, encoder_weights=None):
        super().__init__(config)

        print("DEBUGGING, NUM_LABELS:", config.num_labels)

        if encoder_weights is None:
            if config.encoder_name in ['resnext101_32x48d', 'resnext101_32x32d']:
                encoder_name = 'instagram'
            else:
                encoder_name = 'imagenet'

        self.model = get_smp_model_builder(
            model_name=config.model_name, 
            encoder_name=config.encoder_name, 
            encoder_weights=encoder_weights, 
            num_labels=config.num_labels,
            interpolation="bilinear" if config.heatmap_is_gt else "nearest"
        )

        print(f"Successfull loaded {config.model_name} with encoder: '{config.encoder_name}'")

        self.config = config

        if self.config.heatmap_is_gt:
            # self.heatmap_loss = torch.nn.MSELoss()
            self.heatmap_loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.dice_loss = smp.losses.DiceLoss(mode="multiclass", ignore_index=self.config.ignore_index)
            # class_weights = torch.tensor([1.0, 100.0]) 
            # self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=self.config.ignore_index)
            self.focal_loss = smp.losses.FocalLoss(mode="multiclass", ignore_index=self.config.ignore_index)
        
    def forward(self, pixel_values, labels=None, **kwargs):
        if self.config.model_name in ModelForSemanticSegmentation.models_with_512_io:
            pixel_values = F.pad(pixel_values, (0, 12, 0, 12))
        logits = self.model(pixel_values)
        if self.config.model_name in ModelForSemanticSegmentation.models_with_512_io:
            logits = logits[:, :, :500, :500]
        # logits = F.interpolate(logits, size=(500, 500), mode='bilinear', align_corners=False)
        
        loss = None
        if labels is not None:

            # handle wrong dimensionality
            if labels.dim() == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1)

            if self.config.heatmap_is_gt:
                if isinstance(self.heatmap_loss, torch.nn.BCEWithLogitsLoss):
                    labels = labels.float()
            
                    # Reduce the logits to a single channel if necessary
                    heatmap_logits = logits
                    if heatmap_logits.dim() == 4 and heatmap_logits.shape[1] == 1:
                        heatmap_logits = heatmap_logits.squeeze(1)

                    # directly pass the logits to the loss function (NO torch.sigmoid here!)
                    loss = self.heatmap_loss(heatmap_logits, labels)
                else:
                    # we want the output in range: 0 - 1
                    preds = torch.sigmoid(logits)
                    if preds.dim() == 4 and preds.shape[1] == 1:
                        preds = preds.squeeze(1)

                    labels = labels.float()

                    # print(f"Labels shape: {labels.shape}")
                    # print(f"Preds shape: {preds.shape}")

                    loss = self.heatmap_loss(preds, labels)
            else:
                
                # print("pixel_values:", pixel_values.shape)
                # print("logits:", logits.shape)
                # print("labels:", labels.shape)

                # loss_fct = nn.CrossEntropyLoss(ignore_index=255)
                # loss = loss_fct(logits, labels.long())

                # loss = 0.5 * self.ce_loss(logits, labels.long()) + \
                #        0.5 * self.dice_loss(logits, labels.long())
                loss = 0.5 * self.focal_loss(logits, labels.long()) + \
                    0.5 * self.dice_loss(logits, labels.long())
        # elif self.config.heatmap_is_gt and isinstance(self.heatmap_loss, torch.nn.BCEWithLogitsLoss):
        #     logits = torch.sigmoid(logits)


        # HF Trainer wants an object with 'loss' and 'logits' attributes
        return SemanticSegmenterOutput(loss=loss, logits=logits)





