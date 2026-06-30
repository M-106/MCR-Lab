# -----------
# > Imports <
# -----------
import numpy as np
import torch

from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers import PreTrainedModel, PretrainedConfig

import segmentation_models_pytorch as smp




# ---------
# > Model <
# ---------
class UnetConfig(PretrainedConfig):
    model_type = "unet"
    def __init__(self, num_labels=2, ignore_index=255, heatmap_is_gt=False, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.heatmap_is_gt = heatmap_is_gt

class UnetForSemanticSegmentation(PreTrainedModel):
    config_class = UnetConfig
    
    def __init__(self, config, encoder_weights="imagenet"):
        super().__init__(config)
        # using segmentation_models_pytorch
        self.unet = smp.Unet(
            encoder_name="resnet34",  # "resnet50", "efficientnet-b3" or "efficientnet-b4", "mit_b2" or "mit_b3"
            encoder_weights=encoder_weights, 
            classes=config.num_labels
        )
        self.config = config


        if self.config.heatmap_is_gt:
            self.heatmap_loss = torch.nn.MSELoss()
        else:
            self.dice_loss = smp.losses.DiceLoss(mode="multiclass", ignore_index=self.config.ignore_index)
            # class_weights = torch.tensor([1.0, 100.0]) 
            # self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=self.config.ignore_index)
            self.focal_loss = smp.losses.FocalLoss(mode="multiclass", ignore_index=self.config.ignore_index)
        
    def forward(self, pixel_values, labels=None, **kwargs):
        logits = self.unet(pixel_values)
        
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





