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
            "timm-efficientnet-b7": "efficientnet", 
            "mit_b5": "segformer", 
            "tu-convnext_large.fb_in22k_ft_in1k_384": "convnext",
            "tu-swinv2_base_window12to16_192to256.ms_in22k_ft_in1k": "swinv2",
            "tu-maxvit_xlarge_tf_512": "vit", 
            "tu-maxvit_base_tf_512.in21k_ft_in1k": "vit-pretrained",
            "tu-swinv2_base_window16_256.ms_in1k": "swinv2",
            
            "resnet101": "r101", 
            "resnext101_32x32d": "r101_32x32d", 
            "densenet161": "d161", 
            "mobileone_s4": "mos4",
            "tu-samvit_huge_patch16.sa1b": "tshp16s", 
            "tu-beit_large_patch16_512.in22k_ft_in22k_in1k": "tblp16512",
        }

#         - (unet, tu-swinv2_base_window12to16_192to256.ms_in22k_ft_in1k) = False
#       -> Input height (500) doesn't match model (256).
#   - (deeplabv3, timm-efficientnet-b7) = True
#   - (deeplabv3, tu-convnext_large.fb_in22k_ft_in1k_384) = True
#   - (deeplabv3, mit_b5) = False
#       -> NVML_SUCCESS == r INTERNAL ASSERT FAILED at "../c10/cuda/CUDACachingAllocator.cpp":838, please report a bug to PyTorch. 
#   - (deeplabv3, tu-swinv2_base_window12to16_192to256.ms_in22k_ft_in1k) = False
#       -> Input height (512) doesn't match model (256).
#   - (dpt, tu-maxvit_xlarge_tf_512) = False
#       -> NVML_SUCCESS == r INTERNAL ASSERT FAILED at "../c10/cuda/CUDACachingAllocator.cpp":838, please report a bug to PyTorch. 
#   - (dpt, tu-maxvit_base_tf_512.in21k_ft_in1k) = False
#       -> NVML_SUCCESS == r INTERNAL ASSERT FAILED at "../c10/cuda/CUDACachingAllocator.cpp":838, please report a bug to PyTorch. 
#   - (dpt, tu-swinv2_base_window16_256.ms_in1k) = False
#       -> NVML_SUCCESS == r INTERNAL ASSERT FAILED at "../c10/cuda/CUDACachingAllocator.cpp":838, please report a bug to PyTorch. 


        names = [
            # "unet",
            # "fpn", 
            "deeplabv3", 
            # "deeplabv3plus", 
            "dpt"
        ]
        encoders = [ 
            # [
            #     "timm-efficientnet-b7",

            #     # 1. Kanten- & Präzision: ConvNet SOTA  
            #     "tu-convnext_large.fb_in22k_ft_in1k_384",  
                
            #     # 2. Multi-Scale Transformer: SegFormer-Encoder 
            #     "mit_b5",                                    
                
            #     # 3. Hybrid: Swin-Attention + CNN Stem
            #     # "tu-swinv2_base_window12to16_192to256.ms_in22k_ft_in1k"  
            # ],
            [
                # # 0. EfficientNet (Strong CNN)
                # "timm-efficientnet-b7",

                # # 1. Kanten- & Präzision: ConvNet SOTA  
                # "tu-convnext_large.fb_in22k_ft_in1k_384",  
                
                # 2. Multi-Scale Transformer: SegFormer-Encoder 
                "mit_b5",                                    
                
                # 3. Hybrid: Swin-Attention + CNN Stem
                # "tu-swinv2_base_window12to16_192to256.ms_in22k_ft_in1k"  
            ],
            # ["timm-efficientnet-b7", "mit_b5", "resnet101", "resnext101_32x32d", "densenet161", "mobileone_s4"], 
            # ["timm-efficientnet-b7", "resnet101", "resnext101_32x32d", "mobileone_s4"],
            # ["timm-efficientnet-b7", "mit_b5", "resnet101", "resnext101_32x32d", "mobileone_s4"],
            [
                "tu-maxvit_xlarge_tf_512",

                # 1. Hybrid: (Multi-Axis Attention + CNN Stem -> extrem gut für kleine Details)
                "tu-maxvit_base_tf_512.in21k_ft_in1k", 

                # 2. Hierarchical Transformer (Standard für DPT, exzellente Kanten)
                "tu-swinv2_base_window16_256.ms_in1k"  
            ]
        ]  #"tu-samvit_huge_patch16.sa1b", "tu-beit_large_patch16_512.in22k_ft_in22k_in1k"

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
            if processes_errors.get(key) is not None:
                print(f"      -> {str(processes_errors[key])}")

    elif config.mode == "custom":
        # from mcrlab.execution.train import train
        from mcrlab.execution.eval import center_eval

        experiments = dict()

        config.data.normalization = True
        config.data.normalization_mode = "global_standard"
        config.data.heatmap_path = "/data/2d_gt_patches"
        config.data.used_heatmap_channel = 2

        config.center_eval.center_extraction_method = "peak_maxima"  # "polygon_centroid", "circle_least_squares", "peak_maxima"
        config.center_eval.save_debug_plots = True
        config.center_eval.min_confidences = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.75, 0.8, 0.85, 0.9, 0.95]
        config.center_eval.min_confidence_peak = 0.9
        config.center_eval.candidate_min_points = 30

        for cur_idx in range(0, 11):
            if cur_idx == 0:
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_01_11_20_unet_whu_heatmapGT2_unet_teb7_sigma_60_comp_2/checkpoint-6731"
            elif cur_idx == 1:
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_01_15_33_unet_whu_heatmapGT2_unet_mb5_sigma_60_comp_2/checkpoint-1855"
            elif cur_idx == 2:
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_01_20_40_unet_whu_heatmapGT2_unet_r101_sigma_60_comp_2/checkpoint-6837"
            elif cur_idx == 3:
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_01_22_15_unet_whu_heatmapGT2_unet_r101_32x32d_sigma_60_comp_2/checkpoint-7367"
            elif cur_idx == 4:
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_02_03_13_unet_whu_heatmapGT2_unet_d161_sigma_60_comp_2/checkpoint-5883"
            elif cur_idx == 5:
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_02_07_32_unet_whu_heatmapGT2_unet_mos4_sigma_60_comp_2/checkpoint-7685"
            elif cur_idx == 6:
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_02_10_27_fpn_whu_heatmapGT2_fpn_teb7_sigma_60_comp_2/checkpoint-6148"
            elif cur_idx == 7:
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_02_17_02_fpn_whu_heatmapGT2_fpn_r101_sigma_60_comp_2/checkpoint-7155"
            elif cur_idx == 8:
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_03_01_36_fpn_whu_heatmapGT2_fpn_r101_32x32d_sigma_60_comp_2/checkpoint-7314"
            elif cur_idx == 9:
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_03_11_31_fpn_whu_heatmapGT2_fpn_mos4_sigma_60_comp_2/checkpoint-3975"
            elif cur_idx == 10:
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_03_15_44_deeplabv3_whu_heatmapGT2_deeplabv3_teb7_sigma_60_comp_2/checkpoint-5565"
            
            try:
                # train(config)

                center_eval(config)
                experiments[config.train.exp_name] = "Passed"
            except Exception as e:
                raise e
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




