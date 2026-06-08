# -----------
# > Imports <
# -----------
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F

from transformers import pipeline
# from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
# from transformers import Sam3Processor, Sam3Model
# from transformers import Sam2Processor, Sam2Model
from transformers import AutoModel, AutoProcessor

from mcrlab.image.utils import one_channel_img_to_pil_rgb_img
from mcrlab.models.base import BaseModel



# ----------
# > Helper <
# ----------
def postprocess(segmentation, model):
    results = []
    for class_id in np.unique(segmentation):
        mask = segmentation == class_id
        label = model.config.id2label[class_id]

        results.append({
            "label": label,
            "mask": mask.astype(np.uint8)
        })

    return results



# they are for inference
# --------------------------
# > 2D Segmentation Models <
# --------------------------
class SegFormer(BaseModel):
    # nvidia/segformer-b0-finetuned-cityscapes-512-512
    # chase-geigle/segformer-b0-finetuned-sidewalk
    # nvidia/segformer-b0-finetuned-ade-512-512
    def __init__(self, device=0):  # facebook/sam3-base
        # device = 0 is first GPU, -1 = cpu

        # self.pipe = pipeline(
        #     "image-segmentation",  # "mask-generation", 
        #     model="nvidia/segformer-b5-finetuned-cityscapes-1024-1024",  # "nvidia/segformer-b0-finetuned-ade-512-512", 
        #     device=device
        # )

        self.device = torch.device(f"cuda:{device}" if device >= 0 else "cpu")

        model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"

        # Load preprocessor + model separately
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, x):
        # x can be a path, PIL image or numpy array
        # return self.generator(x, points_per_side=32)
        image = one_channel_img_to_pil_rgb_img(x)

        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Postprocess (resize to original image size)
        logits = outputs.logits  # (1, num_classes, H, W)
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False,
        )

        predicted_segmentation = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        return postprocess(predicted_segmentation, self.model)
        # return self.pipe(image)

    def predict(self, x):
        return self(x)
    
    def get_model(self):
        return self.model
    
    def visualize(self, image_input, masks):
        image = one_channel_img_to_pil_rgb_img(image_input)

        # width, height = image.size

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        ax = plt.gca()

        # SegFormer/MaskFormer returns a list
        for cur_mask in masks:
            label = cur_mask['label']
            mask_pil = cur_mask['mask']
            
            # pil mask to numpy
            mask = np.array(mask_pil) > 0
            
            # random colors
            color = np.random.rand(3)
            
            # mask overlay on image
            img_overlay = np.zeros((*mask.shape, 4)) # RGBA
            img_overlay[mask, :3] = color
            img_overlay[mask, 3] = 0.5  # 50% Transparenz
            
            ax.imshow(img_overlay)
            
            # write label as text
            if mask.any():
                y, x = np.argwhere(mask).mean(axis=0)
                ax.text(x, y, label, color='white', fontsize=12, 
                        fontweight='bold', backgroundcolor='black')

        plt.axis("off")
        plt.show()


        

class SAM2(BaseModel):
    def __init__(self, hf_token=None, device=0): 
        # device = 0 is first GPU, -1 = cpu

        self.device = device

        model_name = "facebook/sam2.1-hiera-base-plus"

        # Load processor + model
        # self.processor = Sam2Processor.from_pretrained(
        #     model_name,
        #     token=hf_token,
        #     trust_remote_code=True
        # )

        # self.model = Sam2Model.from_pretrained(
        #     model_name,
        #     token=hf_token,
        #     trust_remote_code=True
        # )

        # self.model.to(self.device)
        # self.model.eval()

        self.pipe = pipeline(
            "mask-generation",  # for zero shot instance segmentation
            model=model_name,  # "facebook/sam2.1-hiera-base-plus",  # "facebook/sam3",
            # image_processor=self.processor,
            token=hf_token,
            device=device,
            trust_remote_code=True
        )

        # self.device = torch.device(f"cuda:{device}" if device >= 0 else "cpu")

    def __call__(self, x):
        # 1 channel to pseudo RGB
        image = one_channel_img_to_pil_rgb_img(x)

        # # Preprocess
        # inputs = self.processor(images=image, return_tensors="pt")
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # # Forward
        # outputs = self.model(**inputs)

        # masks = self.processor.post_process_masks(
        #     outputs.pred_masks,
        #     inputs["original_sizes"],
        #     # inputs["reshaped_input_sizes"]
        # )

        # return masks

        return self.pipe(image, points_per_side=32)  # bigger value = more details

    def predict(self, x):
        return self(x)
    
    def get_model(self):
        return self.model
    
    # def visualize(self, image_input, results):
    #     image = one_channel_img_to_pil_rgb_img(image_input)

    #     plt.figure(figsize=(12, 12))
    #     plt.imshow(image)
    #     ax = plt.gca()

    #     # SAM output is a dict with key "masks"
    #     # Each mask is a dict with 'segmentation' (bool numpy array)
    #     print(results.keys())
    #     masks = results.get('masks', results)

    #     # sort after plane (smaller masks on top)
    #     # print(type(masks))  # list
    #     # print(type(masks[0]))  # torch tensor
    #     print(masks[0].dtype)  # bool
    #     print(masks[0].shape)  # tile shape torch.Size([600, 600])
    #     print(masks[0])  # False and True
    #     print(masks[0].keys())
    #     print(masks[0]["area"])
    #     masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

    #     for i, mask_data in enumerate(masks):
    #         mask = mask_data['segmentation']
            
    #         # random color
    #         color = np.concatenate([np.random.rand(3), [0.5]]) # RGBA (50% Alpha)
            
    #         # plot masks -> using RGBA
    #         overlay = np.zeros((*mask.shape, 4))
    #         overlay[mask] = color
            
    #         ax.imshow(overlay)
            
    #         # write ID of class/instance
    #         if mask.any():
    #             y, x = np.argwhere(mask).mean(axis=0)
    #             ax.text(x, y, f"ID {i}", color='white', fontsize=8, 
    #                     bbox=dict(facecolor='black', alpha=0.5, pad=0))

    #     plt.axis("off")
    #     plt.title(f"SAM 2: {len(masks)} Instanzen gefunden")
    #     plt.show()

    def visualize(self, image_input, results):
        image = one_channel_img_to_pil_rgb_img(image_input)

        plt.figure(figsize=(12, 12))
        plt.imshow(image)
        ax = plt.gca()

        # Determine if we have a list of tensors or the dictionary-style output
        if isinstance(results, dict) and 'masks' in results:
            masks_list = results['masks']
        else:
            masks_list = results

        # Convert to a list of dicts if they are raw tensors
        # This ensures consistency for the sorting and plotting logic below
        formatted_masks = []
        for m in masks_list:
            # If it's a tensor, move to CPU and convert to numpy bool array
            mask_np = m.cpu().numpy() if torch.is_tensor(m) else m
            
            # If input was just a mask, we create the dict structure manually
            if isinstance(mask_np, np.ndarray):
                formatted_masks.append({
                    'segmentation': mask_np,
                    'area': np.sum(mask_np) # Calculate area for sorting
                })
            else:
                formatted_masks.append(m)

        # Sort: Larger masks first so smaller masks are drawn ON TOP
        formatted_masks = sorted(formatted_masks, key=(lambda x: x['area']), reverse=True)

        for i, mask_data in enumerate(formatted_masks):
            mask = mask_data['segmentation']
            
            # Create an RGBA overlay for the mask
            color = np.concatenate([np.random.rand(3), [0.5]]) 
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask] = color
            
            ax.imshow(overlay)
            
            # Centroid calculation for the ID label
            if mask.any():
                coords = np.argwhere(mask)
                y, x = coords.mean(axis=0)
                ax.text(x, y, f"ID {i}", color='white', fontsize=8, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

        plt.axis("off")
        plt.title(f"SAM 2: {len(formatted_masks)} Instances Found")
        plt.show()


# IMPORTANT: first run 'hf auth login'
class SAM3(BaseModel):
    def __init__(self, hf_token=None, device=-1):

        self.device = torch.device("cuda" if device >= 0 and torch.cuda.is_available() else "cpu")
        print(f"Load SAM3 on {self.device}...")
        
        # try:
        #     self.pipe = pipeline("mask-generation", model="facebook/sam3")
        # except Exception as e:
        #     print(f"IMPORTANT: maybe first run 'hf auth login' and then try again.")
        #     raise e

        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False

        self.processor = Sam3Processor.from_pretrained("facebook/sam3")

    def __call__(self, x, text_query="manhole"):
        # 1-Channel BEV zu Pseudo-RGB (für den ViT Encoder notwendig)
        image = one_channel_img_to_pil_rgb_img(x)
        
        inputs = self.processor(images=image, text=text_query, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)

        # Post-process results
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        return results

    def predict(self, x):
        return self(x)
    
    def get_model(self):
        return self.model
    
    def visualize(self, image_input, results, threshold=0.3):
        """
        Code from https://huggingface.co/facebook/sam3
        """
        image = one_channel_img_to_pil_rgb_img(image_input)
        image = image.convert("RGBA")

        print(f"Found {len(results['masks'])} objects")
        masks = 255 * results['masks'].cpu().numpy().astype(np.uint8)
        
        n_masks = masks.shape[0]
        cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
        colors = [
            tuple(int(c * 255) for c in cmap(i)[:3])
            for i in range(n_masks)
        ]

        for mask, color in zip(masks, colors):
            mask = Image.fromarray(mask)
            overlay = Image.new("RGBA", image.size, color + (0,))
            alpha = mask.point(lambda v: int(v * 0.5))
            overlay.putalpha(alpha)
            image = Image.alpha_composite(image, overlay)
        return image



class DinoMask2Former(BaseModel):
    def __init__(self, device=-1):
        self.device = torch.device("cuda" if device >= 0 and torch.cuda.is_available() else "cpu")
        print(f"Load Mask2Former on {self.device}...")
        
        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-coco-instance").to(self.device)

    def __call__(self, x):
        # 1 channel to pseudo RGB
        image = one_channel_img_to_pil_rgb_img(x)

        # preprocessing
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # inference
        # with torch.no_grad():
        outputs = self.model(**inputs)

        # post processing
        result = self.processor.post_process_instance_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        
        return result

    def predict(self, x):
        return self(x)
    
    def get_model(self):
        return self.model

    def visualize(self, image, result):
        image = one_channel_img_to_pil_rgb_img(image)

        plt.figure(figsize=(12, 12))
        plt.imshow(image)
        ax = plt.gca()

        # map with ids
        # print("Result Type:", type(result))
        # print(f"Keys: {result.keys()}")
        segmentation = result["segmentation"].cpu().numpy()
        segments_info = result["segments_info"]

        for info in segments_info:
            mask = (segmentation == info["id"])
            
            # color for instance
            color = np.concatenate([np.random.rand(3), [0.5]])
            
            # overlay mask
            mask_overlay = np.zeros((*mask.shape, 4))
            mask_overlay[mask] = color
            ax.imshow(mask_overlay)
            
            # write label
            label_id = info["class_id"]
            label_name = self.model.config.id2label[label_id]
            
            if mask.any():
                y, x = np.argwhere(mask).mean(axis=0)
                ax.text(x, y, label_name, color='white', fontsize=9, 
                        bbox=dict(facecolor='black', alpha=0.5, pad=1))

        plt.axis("off")
        plt.show()









# Output:
# masks: A list of dictionaries. Each dictionary contains:
#       segmentation: A Boolean matrix (binary mask) at the original image size.
#       area: The number of pixels in the instance.
#       bbox: The bounding box [x, y, w, h].
#       Predicted_iou: A quality score for the model.

# class HuggingFacePipelineWrapper:
#     # nvidia/segformer-b0-finetuned-cityscapes-512-512
#     # chase-geigle/segformer-b0-finetuned-sidewalk
#     # nvidia/segformer-b0-finetuned-ade-512-512
#     def __init__(self, model_id="facebook/sam3-huge", device=0):  # facebook/sam3-base
#         # device = 0 is first GPU
#         self.pipe = pipeline(
#             "image-segmentation",  # "mask-generation", 
#             model=model_id, 
#             device=device
#         )

#     def __call__(self, x):
#         # x can be a path, PIL image or numpy array
#         # return self.generator(x, points_per_side=32)
#         if isinstance(x, np.ndarray):
#             if x.max() <= 1.0:
#                 x = (x * 255).astype(np.uint8)
#             image = Image.fromarray(x.squeeze())
#         else:
#             image = Image.open(x)

#         rgb_image = image.convert("RGB")

#         return self.pipe(rgb_image)

#     def predict(self, x):
#         return self(x)
    
#     def visualize(self, image_input, masks):
#         if isinstance(image_input, str):
#             image = Image.open(image_input).convert("RGB")
#         else:
#             # Falls es ein numpy array ist
#             image = Image.fromarray(image_input).convert("RGB")

#         # width, height = image.size

#         plt.figure(figsize=(10, 10))
#         plt.imshow(image)
#         ax = plt.gca()

#         # # (B, C, H/4, W/4) -> (B, C, H, W)
#         # upsampled_logits = F.interpolate(
#         #     masks.logits,
#         #     size=(height, width),  # image.shape[:2],  # (H, W)
#         #     mode="bilinear",
#         #     align_corners=False
#         # )
#         # # (B, C, H, W) -> (B, H, W)
#         # seg_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

#         # # Plot
#         # plt.figure(figsize=(10, 10))
#         # plt.imshow(image)
#         # plt.imshow(seg_map, alpha=0.5, cmap="viridis")


#         # SegFormer/MaskFormer returns a list
#         for obj in masks:
#             label = obj['label']
#             mask_pil = obj['mask']
            
#             # PIL Maske in numpy array umwandeln (0 oder 255)
#             mask = np.array(mask_pil) > 0
            
#             # Zufällige Farbe generieren
#             color = np.random.rand(3)
            
#             # Maske über das Bild legen
#             img_overlay = np.zeros((*mask.shape, 4)) # RGBA
#             img_overlay[mask, :3] = color
#             img_overlay[mask, 3] = 0.5  # 50% Transparenz
            
#             ax.imshow(img_overlay)
            
#             # write label as text
#             if mask.any():
#                 y, x = np.argwhere(mask).mean(axis=0)
#                 ax.text(x, y, label, color='white', fontsize=12, 
#                         fontweight='bold', backgroundcolor='black')
        
#         # # sort masks after plane (small masks on top -> else cant be seen)
#         # sorted_masks = sorted(masks["masks"], key=(lambda x: x['area']), reverse=True)
        
#         # # show masks
#         # ax = plt.gca()
#         # ax.set_autoscale_on(False)

#         # for mask_data in sorted_masks:
#         #     m = mask_data['segmentation']
#         #     img = np.ones((m.shape[0], m.shape[1], 3))

#         #     # random color for each instance
#         #     color_mask = np.random.random((1, 3)).tolist()[0]
#         #     for i in range(3):
#         #         img[:,:,i] = color_mask[i]

#         #     # show mask with 40% transparency
#         #     ax.imshow(np.dstack((img, m * 0.4)))

#         plt.axis("off")
#         plt.show()








