# -----------
# > Imports <
# -----------
import json
import os

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from mcrlab.config.config import Config
from mcrlab.model_utils import get_model, get_device, TorchModelWrapper
from mcrlab.point_cloud.data import get_data_loader, get_basic_transform
from mcrlab.execution.train import get_model_and_preprocessor, get_segmentation_prediction
from mcrlab.image.utils import normalize_bev



# ----------------
# > Inference <
# ----------------
# class Inferencer:
#     def __init__(self, model, dataloader, device=None):
#         self.model = model
#         self.dataloader = dataloader
#         self.device = get_device(device)

#     def run(self):
#         results = []

#         for batch_idx, x_batch in tqdm(enumerate(self.dataloader),
#                                        desc="Inference",
#                                        total=len(self.dataloader)
#                                     ):
#             # dataloader might return (x, y)
#             if isinstance(x_batch, (list, tuple)):
#                 x_batch = x_batch[0]

#             preds = self.model.predict(x_batch)

#             results.append(preds)

#         return results

def run_2d_inference(model_name, checkpoint_path, image_path, do_processing, device="cuda"):
    # load device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # load model
    model, preprocessor = get_model_and_preprocessor(
        model_name=model_name, 
        check_point_path=checkpoint_path, 
        num_labels=2 
    )
    model.to(device)
    model.eval()

    # load + preprocess image
    if isinstance(image_path, str):
        x = Image.open(image_path).convert("RGB")
        original_size = [x.size[::-1]]
        x = np.asarray(x)
    elif isinstance(image_path, np.ndarray):
        x = image_path
    elif isinstance(image_path, torch.Tensor):
        x = image_path.detach().cpu().numpy()

    if do_processing:
        x = normalize_bev(x)
        x = x[[0,2,3]]     # drop channel

        x = Image.fromarray(x)
        x = processor(image=x, return_tensors="pt")
        x = x["pixel_values"].to(device)
    else:
        # if "pixel_values" in x:
        #     x = x["pixel_values"].to(device)
        # else:
        x.to(device)

    # model prediction
    with torch.no_grad():
        outputs = model(pixel_values=x)

    # post process
    predicted_mask = get_segmentation_prediction(
        outputs=outputs,
        model_name=model_name,
        processor=processor,
        target_sizes=original_size
    )

    if isinstance(predicted_mask, torch.Tensor):
        predicted_mask = predicted_mask.squeeze().cpu().numpy()
    else:
        predicted_mask = np.squeeze(predicted_mask)

    return predicted_mask



# FIXME
def run_3d_center_pipeline(patch_points,
                           meta,
                           model_name, 
                           checkpoint_path, 
                           image_path, 
                           do_processing, 
                           device="cuda"):
    """
    Takes Tile and Meta and calling run_2d_inference to make a segmentation.

    Then runs a shape-check and center-regression with least-squares.
    """
    origin_x = meta["origin_x"]
    origin_y = meta["origin_y"]
    resolution = meta["resolution"]

    if isinstance(patch_points, PointCloudTensor):
        patch_points = patch_points.get_as_o3d()

    points = patch_points.point[
        get_coordinate_attribute(patch_points)
    ].numpy()

    # center prediction
    centers = run_2d_inference(model_name=model_name, 
                               checkpoint_path=checkpoint_path, 
                               image_path=image_path, 
                               do_processing=do_processing, 
                               device=device)
    
    centers_3d = []
    for center in centers:
        # FIXME -> which channel order?
        pixel_x = center[0]
        pixel_y = center[1]

        # center to 2d
        cur_center_3d = bev_pixel_to_3d(
            patch_points,
            pixel_x,
            pixel_y,
            origin_x=origin_x,    # in meta
            origin_y=origin_y,    # in meta
            resolution=resolution,  # in meta
            search_radius=None
        )

        centers_3d.append(cur_center_3d)

    return centers_3d



# ----------------
# > Main
# ----------------
def inference(config):

    device = get_device(config.device)

    model = get_model(config.model.name)

    checkpoint_path = config.inference.checkpoint_path

    dataset = get_data_loader(config.data.name, 
                                   config.data.path, 
                                   type=config.data.type, 
                                   transform=get_basic_transform(),
                                   batch_size=1, 
                                   shuffle=False, 
                                   num_workers=1,
                                   preprocessed=True, 
                                   return_train_format=False,
                                   return_dataset=False)
    all_paths = dataset.point_cloud_paths
    bev_dataset = BEVDataset(
        path=all_paths, 
        file_paths=[], 
        has_labels=True, 
        image_training=True, 
        preprocessor=processor,
        augment=True
    )


    json_data = list()
    json_data.append({
        "dataset": config.data.name,
        "data": list()
    })
    for tile_id, patch_points in tqdm(enumerate(patch_gen),
                                      total=len(patch_gen),
                                      desc="Inference"):
        _, cur_pc_name = os.path.split(all_paths[tile_id])
        cur_pc_name = ".".join(cur_pc_name.replace("preprocessed_patch_", "").split(".")[:-1])
        cur_pc_id = cur_pc_name.split("_")[0]
        
        pc_idx = None
        for idx_, cur_data in enumerate(json_data[0]["data"]):
            if "pointcloud-id" in cur_data and cur_data["pointcloud-id"] == cur_pc_id:
                pc_idx = idx_
            
        if pc_idx is None:
            json_data[0]["data"].append(
                {
                    "pointcloud-id": cur_pc_id,
                    "centers": list()
                }
            ) 
            pc_idx = -1

        if isinstance(patch_points, (list, tuple)):
            patch_points = patch_points[0]

        if not isinstance(patch_points, PointCloudTensor):
            raise TypeError(f"Expected patch points to be 'PointCloudTensor' but got '{type(patch_points)}'")
        
        # bev_gen = patch_points.get_bev()
        bev_gen = iter(bev_dataset)
        bev_dict = next(bev_gen)

        bev = bev_dict["pixel_values"]

        if isinstance(bev, torch.Tensor):
            bev = bev.detach().cpu().numpy()

        # if "labels" in bev_dict and bev_dict["labels"] is not None:
        #     labels_bev = bev_dict["labels"]
        #     if isinstance(labels_bev, torch.Tensor):
        #         labels_bev = labels_bev.detach().cpu().numpy()

        #     bev = np.concatenate([bev, labels_bev[None]], axis=0)

        # back propagation
        meta = bev_dict["meta"]

        # prediction -> get center
        cur_center_points = run_3d_center_pipeline(
            pixel_values,
            meta,
            model_name=config.model.name, 
            checkpoint_path=config.inference.checkpoint_path, 
            image_path=bev, 
            do_processing=False,  # right? 
            device="cuda"
        )

        for cur_center in cur_center_points:
            
            # save center -> cur_pc_id "pointcloud-id"
            assert json_data[0]["data"][pc_idx]["pointcloud-id"] == cur_pc_id

            json_data[0]["data"][pc_idx]["centers"].append(
                {
                    "x": float(cur_center[0]), 
                    "y": float(cur_center[1]), 
                    "z": float(cur_center[2])
                }
            ) 

    with open(config.inference.save_path, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)

    print("Successfull finished!")

        












