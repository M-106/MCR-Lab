# -----------
# > Imports <
# -----------
import torch.nn as nn
from torch.utils.data import DataLoader, default_collate



# ----------
# > Getter <
# ----------
def get_data(name:str, path:str, type:str, load_2d:bool,
             pass_label_in_preprocessor:bool, 
             heatmap_path=None,
             used_heatmap_channel=2,
             sample_required_manhole_points=50,
             amount_non_manhole_samples=10,
             return_as_dataloader=True,
             dataloader_batchsize=1,
             dataloader_shuffle=False,
             dataloader_num_workers=4,
             dataloader_collate_fn=None):
    if not load_2d:
        raise RuntimeError(f"Custom NN Pipeline is currently only available for 2D not 3D.")

    name = name.lower()

    dataset = get_data_loader(
        name, 
        path, 
        type="train", 
        transform=get_basic_transform(),
        batch_size=config.train.batch_size, 
        shuffle=True, 
        num_workers=4,
        preprocessed=True, 
        return_train_format=True,
        return_dataset=True if load_2d else False
    )

    if load_2d:
        # FIXME: also add Heatmap paramter 
        all_paths = dataset.point_cloud_paths
        dataset = BEVDataset(path=all_paths, 
                                file_paths=[], 
                                has_labels=True, 
                                image_training=True, 
                                preprocessor=processor,
                                augment=True,
                                pass_label_in_preprocessor=pass_label_in_preprocessor,
                                heatmap_gt_path=heatmap_path,
                                used_heatmap_channel=used_heatmap_channel)
        dataset.manhole_filter(required_manhole_points=50, amount_non_manhole_samples=10)
    else:
        pass
        # FIXME: add filtering from my code in LIDARLearn

   
    if dataset is None:
        raise ValueError(f"Could not create Criterion/Loss with name '{name}'.")

    # make dataloader
    if return_as_dataloader:
        if dataloader_collate_fn is None:
            dataloader_collate_fn = default_collate

        dataset = DataLoader(
            dataset, 
            batch_size=dataloader_batchsize, 
            shuffle=dataloader_shuffle, 
            num_workers=dataloader_num_workers,
            collate_fn=dataloader_collate_fn
        )

    return dataset







