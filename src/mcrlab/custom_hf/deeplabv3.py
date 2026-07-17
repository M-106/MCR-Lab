# -----------
# > Imports <
# -----------
import numpy as np

import torch
import torch.nn as nn
from torchvision.models.segmentation import (deeplabv3_resnet50, DeepLabV3_ResNet50_Weights,
                                             deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights,
                                             deeplabv3_resnet101, DeepLabV3_ResNet101_Weights)

from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers import PreTrainedModel, PretrainedConfig

import segmentation_models_pytorch as smp


# ----------
# > Models <
# ----------
class DeepLabV3Config(PretrainedConfig):
    model_type = "deeplabv3"
    def __init__(self, num_labels=2, ignore_index=255, heatmap_is_gt=False, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.heatmap_is_gt = heatmap_is_gt



class DeepLabV3Wrapper(PreTrainedModel):
    config_class = DeepLabV3Config

    def __init__(self, config):
        super().__init__(config)

        # Load torchvision model
        self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        # self.model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
        # self.model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)

        print("DeepLabV3 Architecture:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if hasattr(param.data, "shape"):
                    print(f"  - {name}: {param.data.shape}")
                else:
                    print(f"  - {name}")
        print(self.model)

        # Replace the head to match your number of labels
        self.model.classifier[4] = nn.Conv2d(256, config.num_labels, kernel_size=(1, 1))

        self.config = config

        if self.config.heatmap_is_gt:
            self.heatmap_loss = torch.nn.MSELoss()
        else:
            self.dice_loss = smp.losses.DiceLoss(mode="multiclass", ignore_index=self.config.ignore_index)
            
            # self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=self.config.ignore_index)
            self.focal_loss = smp.losses.FocalLoss(mode="multiclass", ignore_index=self.config.ignore_index)
        
        
    def forward(self, pixel_values, labels=None, **kwargs):
        # Torchvision models return a dict with an 'out' key
        output = self.model(pixel_values)
        logits = output['out']

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

                # loss = 0.5 * self.ce_loss(logits, labels.long()) + \
                #        0.5 * self.dice_loss(logits, labels.long())
                loss = 0.5 * self.focal_loss(logits, labels.long()) + \
                    0.5 * self.dice_loss(logits, labels.long())
        
        # # Return a simple object that has a .logits attribute
        # class Output:
        #     def __init__(self, logits, loss):
        #         self.logits = logits
        #         self.loss = loss
        return SemanticSegmenterOutput(loss=loss, logits=logits)





