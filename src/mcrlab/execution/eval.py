# -----------
# > Imports <
# -----------
import json

from mcrlab.execution.train import get_model_and_processor
from mcrlab.execution.test import predict_single_sample
from mctlab.classic.least_squares import fit_circle_least_squares
from mcrlab.point_cloud.shape_check import circle_shape_check





# ---------------
# > Center Eval <
# ---------------
def add_to_result(result, cur_dataset, pc_id, cur_center_point):
    found_entry = False
    for cur_res in result:
        if cur_res.get("dataset", "-999") == cur_dataset and \
            cur_res.get("pointcloud-id", "-999") == pc_id:
            if hasattr(cur_res, "centers"):
                cur_res["centers"].append(cur_center_point)
            else:
                cur_res["centers"] = [cur_center_point]

            found_entry = True
            break
    if not found_entry:
        result.append({
            "dataset": cur_dataset,
            "pointcloud-id": pc_id,
            "centers": [cur_center_point]
        })
    
    return result

def center_eval(config):

    model_name = config.model.name.lower()

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
    model, processor = get_model_and_processor(model_name, checkpoint_path, mode="test", num_labels=num_labels, ignore_index=ignore_index, heatmap_is_gt=using_heatmap_as_gt)
    model.eval().to("cuda")

    parts = Path(checkpoint_path).parts
    exp_name = parts[-2]

    # Load Test Data
    heatmap_path = config.data.heatmap_path
    used_heatmap_channel = config.data.used_heatmap_channel
    pass_label_in_preprocessor = model_name in ["mask2former", "oneformer"]
    
    result = []

    for cur_dataset in ["whu", "sud"]
        test_3d_dataset = get_data_loader(
            cur_dataset, 
            config.data.path if cur_dataset == "whu" else config.data.path_2, 
            type="test",
            transform=get_basic_transform(),
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            preprocessed=True, 
            return_train_format=True,
            return_dataset=True,
        )
        
        all_test_paths = test_3d_dataset.point_cloud_paths

        test_bev_dataset = BEVDataset(
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

        for idx, cur_data_path in enumerate(all_test_paths():
            pc_id, x_start, y_start = test_bev_dataset.extract_grid_identifier(cur_data_path)

            # get data
            bev_dict = next(test_bev_dataset.get_patch_via_identifier(pc_id, x_start, y_start, return_generator=True))
            meta = bev_dict["meta"]
            pixel_values = bev_dict["pixel_values"]
            labels = bev_dict["labels"]

            pc = test_3d_dataset[idx]

            # make center prediction
            prediction = predict_single_sample(model, processor, pixel_values)
            print(f"DEBUGGING 1, shape: {prediction.shape}")

            # apply closing + clustering
            if isinstance(prediction, torch.Tensor):
                prediction = preds.detach().cpu().numpy()
            valid_mask = (labels != ignore_index)
    
            if using_heatmap_as_gt:
                preds_binary = ((preds >= 0.5) & valid_mask).astype(np.uint8)
            else:
                preds_binary = ((preds == 1) & valid_mask).astype(np.uint8)

            struct = generate_binary_structure(2, 2)  # 8-Nachbarschaft
            preds_closed = binary_closing(preds_binary, structure=struct, iterations=4).astype(np.uint8)
            # FIXME -> custering

            # go thrpugh every cluster:
            # could also transform everything into 3D
            a, b, abs(r), mean_distance_error, loss = fit_circle_least_squares(x, y)

            # transform to 3d
            cluster_in_3d = bev_pixel_to_3d(
                patch_points=pc,
                pixel_x=,
                pixel_y=,
                origin_x=meta["origin_x"],    # in meta
                origin_y=meta["origin_y"],    # in meta
                resolution=meta["resolution"],  # in meta
                search_radius=None
            )
            

            # shape check
            is_circle_, _ = circle_shape_check(points_square[cur_vis], save_path=None, should_plot=False, threshold=0.6)

            # get center i 3D
            center = bev_pixel_to_3d(
                patch_points=pc,
                pixel_x=a,
                pixel_y=b,
                origin_x=meta["origin_x"],    # in meta
                origin_y=meta["origin_y"],    # in meta
                resolution=meta["resolution"],  # in meta
                search_radius=None
            )

            # 
            cur_center_point = {
                "x": ...,
                "y": ...,
                "z": ...
            }
            # cur_point = None

            # add result
            result = add_to_result(result, cur_dataset, pc_id, cur_center_point)

    with open(f"./output/eval_{exp_name}.json", "w") as file_:
        json.dump(result, indent=4)


def main(config):
    model 
    center_eval(config)






