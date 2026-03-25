# -----------
# > Imports <
# -----------
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from mcrlab.point_cloud.utils import filter_ground_with_height, filter_ground_with_RANSAC, \
                                     get_coordinate_attribute, get_class_attribute, \
                                     get_intensity_attribute, get_color_attribute, \
                                     get_normal_attribute
from mcrlab.point_cloud.io import load_point_cloud
from mcrlab.point_cloud.core import PointCloudTensor



# ----------------------
# > Data-Augmentations <
# ----------------------
class RoadExtractionTransform:
    def __init__(self, mode="height"):  # "height", "ransac", None
        self.mode = mode

    def __call__(self, point_cloud):
        coordinate_idx = get_coordinate_attribute(point_cloud)
        point_before = len(point_cloud.point[coordinate_idx])

        if self.mode is None:
            pass
        elif self.mode == "height":
            point_cloud = filter_ground_with_height(point_cloud, threshold=10.0)
        elif self.mode == "ransac":
            point_cloud = filter_ground_with_RANSAC(point_cloud, distance_threshold=1.0, ransac_n=3, num_iterations=3000)
        else:
            raise ValueError(f"No `{self.mode}` mode for Road Extraction.")
        point_after = len(point_cloud.point[coordinate_idx])
        print(f"Road Extraction filtered {point_before-point_after} points.")
        return point_cloud

class OutlierRemovalTransform:
    def __init__(self):
        pass

    def __call__(self, point_cloud):
        coordinate_idx = get_coordinate_attribute(point_cloud)
        point_before = len(point_cloud.point[coordinate_idx])

        
        point_after = len(point_cloud.point[coordinate_idx])
        print(f"Road Extraction filtered {point_before-point_after} points.")
        return point_cloud


# add more augmentations! -> random rotate, jitter



class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, point_cloud):
        for cur_transformation in self.transforms:
            point_cloud = cur_transformation(point_cloud)
        return point_cloud



# ----------------
# > Data-Helpers <
# ----------------
class ToPointCloudTensorTransform:
    def __init__(self, num_points=-1):
        self.num_points = num_points

    def __call__(self, point_cloud):
        # convert to numpy
        points = point_cloud.point[get_coordinate_attribute(point_cloud)].numpy()

        # process additional information
        # data = points
        data = PointCloudTensor(points)

        color_idx = get_color_attribute(point_cloud)
        if color_idx:
            colors = point_cloud.point[color_idx].numpy()
            data.colors = colors
            # data = np.hstack([data, colors])
            # we use hstack because we want more columns not more rows
            # [x y z r g b intensity nx ny nz]
            # not [[x y z]
            #      [r g b] 
            #        ...  ]
            # also could use concatenate: np.concatenate([data, colors], axis=1)

        intensity_idx = get_intensity_attribute(point_cloud)
        if intensity_idx:
            intensities = point_cloud.point[intensity_idx].numpy()
            data.intensities = intensities
            # data = np.hstack([data, intensities])

        normal_idx = get_normal_attribute(point_cloud)
        if normal_idx:
            normals = point_cloud.point[normal_idx].numpy()
            data.normals = normals
            # data = np.hstack([data, normals])

        data.labels = extract_labels_as_tensor(point_cloud)

        # sample or pad points to fixed size
        if self.num_points > 0:
            cur_number_of_points = data.coordinates.shape[0]
            if cur_number_of_points > self.num_points:
                # idxs = np.random.choice(cur_number_of_points, self.num_points, replace=False)
                
                z = data.coordinates[:, 2]
                idx_sorted = np.argsort(z)     # sorted low to high

                # remove top N
                N_remove = 500
                idxs = idx_sorted[:-N_remove]
            else:
                extra_idxs = np.random.choice(cur_number_of_points, (self.num_points-cur_number_of_points), replace=True)
                idxs = np.hstack((np.arange(cur_number_of_points), extra_idxs))  # result also 1 dimensional

            np.random.shuffle(idxs)

            # apply mask/filtered points
            data.coordinates = data.coordinates[idxs]
            if data.intensities is not None:
                data.intensities = data.intensities[idxs]
            if data.colors is not None:
                data.colors = data.colors[idxs]
            if data.normals is not None:
                data.normals = data.normals[idxs]
            if data.labels is not None:
                data.labels = data.labels[idxs]

        # to torch tensor
        data.to_torch()

        return data



def extract_labels_as_tensor(point_cloud):
    label_idx = get_class_attribute(point_cloud)
    
    if label_idx:
        labels = torch.tensor(point_cloud.point[label_idx].numpy())
    else:
        labels = None

    return labels



def collate_point_clouds(batch):
    """
    Either just returns the incoming as it is 
    or returning
    """
    # return batch
    # if type(batch) in [list, tuple] and len(batch[0]) == 2:
    #     pcs = [item[0] for item in batch]
    #     labels = [item[1] for item in batch]
    #     return pcs, labels
    # else:
    return batch



def get_basic_transform(num_points=-1, road_extraction_mode="height"):
    transform = Compose([
        # RandomRotate(axis='z'),
        # Jitter(sigma=0.01, clip=0.05),
        RoadExtractionTransform(mode=road_extraction_mode),
        ToPointCloudTensorTransform(num_points=num_points)
    ])
    return transform



# -----------
# > Dataset <
# -----------
class ParisLille3DDataset(Dataset):
    def __init__(self, path, testdata=False, transform=None):
        self.path = path
        self.testdata = testdata
        self.transform = transform

        self.POINT_LIMIT = 1024

        if self.testdata:
            self.path = os.path.join(self.path, "test_10_classes")
        else:
            self.path = os.path.join(self.path, "training_10_classes")

        self.point_cloud_paths = []
        for cur_file in os.listdir(self.path):
            if any([cur_file.endswith(ending) for ending in [".las", ".laz", ".ply"]]):
                self.point_cloud_paths.append(os.path.join(self.path, cur_file))

        print(f"Uses {len(self.point_cloud_paths)} point clouds.")

    def __len__(self):
        return len(self.point_cloud_paths)

    def __getitem__(self, idx):
        point_cloud = load_point_cloud(self.point_cloud_paths[idx])
        # point_cloud = point_cloud.voxel_down_sample(voxel_size=0.05)

        if self.transform:
            point_cloud = self.transform(point_cloud)

        return point_cloud  # PointCloudTensor or o3d.t.geometry.Tensor

        # labels = extract_labels_as_tensor(point_cloud)

        # if labels:
        #     return (point_cloud, labels)
        # else:
        #     return point_cloud  # shape (1024, 3)


# also add data_loader which return 3 data-loader or 2
# -> train, val, eval
# -> val is simply a random split from train

def get_data_loader(data_name, path, testdata=False, transform=None,
                    batch_size=32, shuffle=True, num_workers=4):
    if data_name == "paris":
        data_loader = get_paris_data_loader(path, testdata=testdata, transform=transform,
                                            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
        raise ValueError(f"No Dataset with the name '{data_name}' founded. Try 'paris'.")

    return data_loader


def get_paris_data_loader(path, testdata=False, transform=None,
                          batch_size=32, shuffle=True, num_workers=4):
    dataset = ParisLille3DDataset(path=path, testdata=testdata, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_point_clouds)




