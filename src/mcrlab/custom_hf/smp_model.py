# -----------
# > Imports <
# -----------
import numpy as np
import torch
import torch.nn.functional as F

from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers import PreTrainedModel, PretrainedConfig

import segmentation_models_pytorch as smp



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
    
    def __init__(self, config, encoder_weights="imagenet"):
        super().__init__(config)

        if encoder_weights is None:
            if config.encoder_name in ['resnext101_32x48d']:
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
            self.heatmap_loss = torch.nn.MSELoss()
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


        # HF Trainer wants an object with 'loss' and 'logits' attributes
        return SemanticSegmenterOutput(loss=loss, logits=logits)





