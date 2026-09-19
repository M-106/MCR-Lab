# -----------
# > Imports <
# -----------

# config
import argparse
from mcrlab.config.config import load_config

from mcrlab.point_cloud.data import preprocess_data, get_preprocessing_transform



# --------------
# > Main Logic <
# --------------
def main():
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="./configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # print(config.mode)
    # print(config.model)
    # print(config.train.batch_size)
    # print(config.data.path)
    print("Configuration:")
    print(config)
    print("\n")



    # do something
    if config.mode == "custom_train":
        from mcrlab.execution.custom_train import train
        train(config)

    elif config.mode == "train":
        from mcrlab.execution.train import train
        train(config)

    elif config.mode == "train_all":
        from mcrlab.execution.train import train

        processes = dict()
        processes_errors = dict()

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
            "fpn", 
            "deeplabv3", 
            "deeplabv3plus", 
            "dpt"
        ]
        encoders = [ 
            ["timm-efficientnet-b7", "mit_b5", "resnet101", "resnext101_32x32d", "densenet161", "mobileone_s4"], 
            ["timm-efficientnet-b7", "resnet101", "resnext101_32x32d", "mobileone_s4"],
            ["timm-efficientnet-b7", "mit_b5", "resnet101", "resnext101_32x32d", "mobileone_s4"],
            ["tu-samvit_huge_patch16.sa1b", "tu-maxvit_xlarge_tf_512", "tu-beit_large_patch16_512.in22k_ft_in22k_in1k"]
        ]

        original_exp_name = config.train.exp_name

        for name, cur_encoders in zip(names, encoders):
            for encoder in cur_encoders:

                config.model.name = name
                config.model.encoder = encoder
                config.train.exp_name = f"{name}_{encoder_to_short[encoder]}_{original_exp_name}"
        
                try:
                    train(config)
                    processes[(name, encoder)] = True
                except Exception as e:
                    processes[(name, encoder)] = False
                    processes_errors[(name, encoder)] = e

        print("\nAll trainings done. Results:")
        for key, value in processes.items():
            print(f"  - ({key[0]}, {key[1]}) = {str(value)}")
            print(f"      -> {str(processes_errors[key])}")

    elif config.mode == "custom":
        from mcrlab.execution.train import train

        experiments = dict()

        for cur_idx in range(10):
            if cur_idx == 0:
                config.train.exp_name = "seg_norm_check_unet_teb7_no_norm"
                config.data.normalization = False
                config.data.normalization_mode = "local_minmax"
                config.data.heatmap_path = None
            elif cur_idx == 1:
                config.train.exp_name = "seg_norm_check_unet_teb7_local_minmax"
                config.data.normalization = True
                config.data.normalization_mode = "local_minmax"
                config.data.heatmap_path = None
            elif cur_idx == 2:
                config.train.exp_name = "seg_norm_check_unet_teb7_global_minmax"
                config.data.normalization = True
                config.data.normalization_mode = "global_minmax"
                config.data.heatmap_path = None 
            elif cur_idx == 3:
                config.train.exp_name = "seg_norm_check_unet_teb7_local_standard"
                config.data.normalization = True
                config.data.normalization_mode = "global_standard"
                config.data.heatmap_path = None
            elif cur_idx == 4:
                config.train.exp_name = "heatmap_unet_teb7_sigma_10"
                config.data.normalization = True
                config.data.normalization_mode = "global_minmax"
                config.data.heatmap_path = "/data/2d_gt_patches"
                config.data.used_heatmap_channel = 0
            elif cur_idx == 5:
                config.train.exp_name = "heatmap_unet_teb7_sigma_20"
                config.data.normalization = True
                config.data.normalization_mode = "global_minmax"
                config.data.heatmap_path = "/data/2d_gt_patches"
                config.data.used_heatmap_channel = 1
            elif cur_idx == 6:
                config.train.exp_name = "heatmap_unet_teb7_sigma_60"
                config.data.normalization = True
                config.data.normalization_mode = "global_minmax"
                config.data.heatmap_path = "/data/2d_gt_patches"
                config.data.used_heatmap_channel = 2
            elif cur_idx == 7:
                config.train.exp_name = "heatmap_dlv3_teb7_sigma_10"
                config.model.name = "deeplabv3"
                config.data.normalization = True
                config.data.normalization_mode = "global_minmax"
                config.data.heatmap_path = "/data/2d_gt_patches"
                config.data.used_heatmap_channel = 0
            elif cur_idx == 8:
                config.train.exp_name = "heatmap_dlv3_teb7_sigma_20"
                config.model.name = "deeplabv3"
                config.data.normalization = True
                config.data.normalization_mode = "global_minmax"
                config.data.heatmap_path = "/data/2d_gt_patches"
                config.data.used_heatmap_channel = 1
            elif cur_idx == 9:
                config.train.exp_name = "heatmap_dlv3_teb7_sigma_60"
                config.model.name = "deeplabv3"
                config.data.normalization = True
                config.data.normalization_mode = "global_minmax"
                config.data.heatmap_path = "/data/2d_gt_patches"
                config.data.used_heatmap_channel = 2
            
            try:
                train(config)
                experiments[config.train.exp_name] = "Passed"
            except Exception as e:
                experiments[config.train.exp_name] = str(e)
        
        print("\n\nCustom Run Finished!\nProcess Info:")
        for key, value in experiments.items():
            print(f"  - {key}: {value}")

    elif config.mode == "test":
        from mcrlab.execution.test import test
        test(config)

    # elif config.mode == "test":
    #     from mcrlab.execution.inference import inference
    #     inference(config)

    elif config.mode == "preprocessing":
        preprocess_data(config.data.name, config.data.path, 
                        type=config.data.type,  # maybe also define over config!
                        device="cpu",
                        bev_tile_size=5.0, bev_resolution=0.01, bev_overlap=0.5,  # bev_tile_size=35.0, bev_resolution=0.05)
                        file_ending=config.preprocessing.file_ending)
        
    elif config.mode == "tryout":
        from mcrlab.execution.tryout import tryout
        tryout(config)
    
    elif config.mode == "eval_extraction":
        if config.eval_extraction.generate_2d_gt_maps:
            from mcrlab.execution.eval_extraction import ground_truth_extraction_heatmap
            ground_truth_extraction_heatmap(config)
        else:
            from mcrlab.execution.eval_extraction import ground_truth_extraction
            ground_truth_extraction(config)

    elif config.mode == "center_eval":
        from mcrlab.execution.eval import center_eval
        center_eval(config)
    else:
        raise ValueError(f"'{config.mode}' is not an available mode for mcrlab.")




