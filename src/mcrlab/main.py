# -----------
# > Imports <
# -----------
import os

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
    if config.mode == "train":
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
            "unet",
            # "fpn", 
            "deeplabv3", 
            # "deeplabv3plus", 
            "dpt"
        ]
        encoders = [ 
            # [
                # # 0. EfficientNet (Strong CNN)
                "timm-efficientnet-b7",

                # 1. Kanten- & Präzision: ConvNet SOTA  
                "tu-convnext_large.fb_in22k_ft_in1k_384",  
                
                # 2. Multi-Scale Transformer: SegFormer-Encoder 
                "mit_b5",                                    
                
            #     # 3. Hybrid: Swin-Attention + CNN Stem
            #     # "tu-swinv2_base_window12to16_192to256.ms_in22k_ft_in1k"  
            # ],
            [
                # # 0. EfficientNet (Strong CNN)
                "timm-efficientnet-b7",

                # # 1. Kanten- & Präzision: ConvNet SOTA  
                "tu-convnext_large.fb_in22k_ft_in1k_384",  
                
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
                # "tu-swinv2_base_window16_256.ms_in1k"  
            ]
        ]  #"tu-samvit_huge_patch16.sa1b", "tu-beit_large_patch16_512.in22k_ft_in22k_in1k"

        original_exp_name = config.train.exp_name

        for name, cur_encoders in zip(names, encoders):
            for encoder in cur_encoders:

                if name == "dpt":
                    config.train.batch_size = 1
                elif name == "deeplabv3" and encoder == "mit_b5":
                    config.train.batch_size = 1
                else:
                    config.train.batch_size = 2

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
        from mcrlab.execution.train import train, finetune_for_precision
        from mcrlab.execution.center_prediction_pipeline import center_eval
        # from mcrlab.execution.test import test
        # from mcrlab.custom_hf.smp_model import run_count

        experiments = dict()

        # # cleaning
        # shutil.rmtree(config.center_eval.pred_save_load_mode_saving_path)


        config.data.normalization = True
        config.data.normalization_mode = "global_standard"
        
        for cur_idx in range(0, 15):
            if cur_idx == 0:
                config.data.heatmap_path = "/data/heatmap"
                config.data.used_heatmap_channel = 2
                config.center_eval.pred_save_load_mode_saving_name = "HEATMAP_DPT_VIT_PRETRAINED_FINETUNED"
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_18_08_35_dpt_whu_heatmapGT2_dpt_vit-pretrained_final_comp_std_norm/checkpoint-6254"
            elif cur_idx == 1:
                config.data.heatmap_path = "/data/heatmap"
                config.data.used_heatmap_channel = 2
                config.center_eval.pred_save_load_mode_saving_name = "HEATMAP_DPT_VIT_FINETUNED"
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_17_21_07_dpt_whu_heatmapGT2_dpt_vit_final_comp_std_norm/checkpoint-9116"
            elif cur_idx == 2:
                config.data.heatmap_path = "/data/heatmap"
                config.data.used_heatmap_channel = 2
                config.center_eval.pred_save_load_mode_saving_name = "HEATMAP_DEEPLABV3_CONVNEXT_FINETUNED"
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_17_03_10_deeplabv3_whu_heatmapGT2_deeplabv3_convnext_final_comp_std_norm/checkpoint-2173"
            elif cur_idx == 3:
                config.data.heatmap_path = "/data/heatmap"
                config.data.used_heatmap_channel = 2
                config.center_eval.pred_save_load_mode_saving_name = "HEATMAP_DEEPLABV3_EFFICIENTNET_FINETUNED"
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_16_21_42_deeplabv3_whu_heatmapGT2_deeplabv3_efficientnet_final_comp_std_norm/checkpoint-4982"
            elif cur_idx == 4:
                config.data.heatmap_path = "/data/heatmap"
                config.data.used_heatmap_channel = 2
                config.center_eval.pred_save_load_mode_saving_name = "HEATMAP_UNET_SEGFORMER_FINETUNED"
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_16_18_42_unet_whu_heatmapGT2_unet_segformer_final_comp_std_norm/checkpoint-4611"
            elif cur_idx == 5:
                config.data.heatmap_path = "/data/heatmap"
                config.data.used_heatmap_channel = 2
                config.center_eval.pred_save_load_mode_saving_name = "HEATMAP_UNET_CONVNEXT_FINETUNED"
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_16_14_07_unet_whu_heatmapGT2_unet_convnext_final_comp_std_norm/checkpoint-7738"
            elif cur_idx == 6:
                config.data.heatmap_path = "/data/heatmap"
                config.data.used_heatmap_channel = 2
                config.center_eval.pred_save_load_mode_saving_name = "HEATMAP_UNET_EFFICIENTNET_FINETUNED"
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_16_11_56_unet_whu_heatmapGT2_unet_efficientnet_final_comp_std_norm/checkpoint-7261"
            elif cur_idx == 7:
                continue
                # config.data.heatmap_path = "/data/heatmap"
                # config.data.used_heatmap_channel = 2
                # config.center_eval.pred_save_load_mode_saving_name = "HEATMAP_UNET_EFFICIENTNET"
                # config.model.check_point_path = "/out/checkpoints/2d/2026_10_16_11_39_unet_whu_heatmapGT2_unet_efficientnet_seg_final_comp_std_norm/checkpoint-212"
            elif cur_idx == 8:
                config.data.heatmap_path = None
                config.center_eval.pred_save_load_mode_saving_name = "SEG_DPT_VIT_PRETRAINED_FINETUNED"
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_13_21_41_dpt_whu_dpt_vit-pretrained_seg_final_comp_std_norm/checkpoint-15264"
            elif cur_idx == 9:
                config.data.heatmap_path = None
                config.center_eval.pred_save_load_mode_saving_name = "SEG_DPT_VIT_FINETUNED"
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_13_10_19_dpt_whu_dpt_vit_seg_final_comp_std_norm/checkpoint-4452"
            elif cur_idx == 10:
                config.data.heatmap_path = None
                config.center_eval.pred_save_load_mode_saving_name = "SEG_DEEPLABV3_CONVNEXT_FINETUNED"
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_08_10_40_deeplabv3_whu_deeplabv3_convnext_seg_final_comp_std_norm/checkpoint-7314"
            elif cur_idx == 11:
                config.data.heatmap_path = None
                config.center_eval.pred_save_load_mode_saving_name = "SEG_DEEPLABV3_EFFICIENTNET_FINETUNED"
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_08_05_07_deeplabv3_whu_deeplabv3_efficientnet_seg_final_comp_std_norm/checkpoint-7208"
            elif cur_idx == 12:
                config.data.heatmap_path = None
                config.center_eval.pred_save_load_mode_saving_name = "SEG_UNET_SEGFORMER_FINETUNED"
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_08_02_01_unet_whu_unet_segformer_seg_final_comp_std_norm/checkpoint-7685"
            elif cur_idx == 13:
                config.data.heatmap_path = None
                config.center_eval.pred_save_load_mode_saving_name = "SEG_UNET_CONVNEXT_FINETUNED"
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_07_21_20_unet_whu_unet_convnext_seg_final_comp_std_norm/checkpoint-7473"
            elif cur_idx == 14:
                config.data.heatmap_path = None
                config.center_eval.pred_save_load_mode_saving_name = "SEG_UNET_EFFICIENTNET_FINETUNED"
                config.model.check_point_path = "/out/checkpoints/2d/2026_10_07_19_05_unet_whu_unet_efficientnet_seg_final_comp_std_norm/checkpoint-7526"
            
            config.center_eval.run_mode = "save"
            config.center_eval.pred_save_load_mode_saving_path = "/out/saved_preds"

            try:
                # train(config)
                # test(config)

                print("Finetuning")

                best_ckpt = finetune_for_precision(
                    config=config,
                    checkpoint_path=config.model.check_point_path, # Dein bisheriger Checkpoint-Ordner
                    epochs=1.0,                                    # 1 Epoche reicht völlig
                    lr=1e-5,                                       # Kleine Lernrate (war vorher versehentlich 1e-3)
                    non_manhole_train_samples=800,                 # Statt 10 jetzt 800 leere Patches im Training!
                    non_manhole_val_samples=200,                   # Auch im Val-Set leere Patches prüfen!
                    freeze_encoder=True                            # Spart 50% Zeit und schützt den Recall
                )

                config.model.check_point_path = best_ckpt

                print("Save predictions")
                center_eval(config)

                # run_count(config)
                # experiments[config.train.exp_name] = "Passed"
                experiments[os.path.split(config.model.check_point_path)[-1]] = "Passed"
            except Exception as e:
                raise e
                # experiments[config.train.exp_name] = str(e)
                experiments[os.path.split(config.model.check_point_path)[-1]] = str(e)
        
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
    
    elif config.mode == "gt_extraction":
        if config.gt_extraction.generate_2d_gt_maps:
            from mcrlab.execution.gt_extraction import ground_truth_extraction_heatmap
            ground_truth_extraction_heatmap(config)
        else:
            from mcrlab.execution.gt_extraction import ground_truth_extraction
            ground_truth_extraction(config)

    elif config.mode == "center_eval":
        from mcrlab.execution.center_prediction_pipeline import center_eval
        center_eval(config)
    
    else:
        raise ValueError(f"'{config.mode}' is not an available mode for mcrlab.")




