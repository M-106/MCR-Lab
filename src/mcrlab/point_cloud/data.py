# -----------
# > Imports <
# -----------
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import open3d as o3d

from tqdm import tqdm

from mcrlab.point_cloud.utils import filter_ground_with_height, filter_ground_with_RANSAC, \
                                     get_coordinate_attribute, get_class_attribute, \
                                     get_intensity_attribute, get_color_attribute, \
                                     get_normal_attribute
from mcrlab.point_cloud.io import load_point_cloud, save_point_cloud
from mcrlab.point_cloud.core import PointCloudTensor, map_torch_device_to_o3d



# ----------------------
# > Data-Augmentations <
# ----------------------
class HeightFilterTransform:
    def __init__(self, min_height=-2.0, max_height=2.0):
        self.min_height = min_height
        self.max_height = max_height

    def __call__(self, point_cloud):
        if not isinstance(point_cloud, o3d.t.geometry.PointCloud):
            raise ValueError(f"Can't apply ToPointCloudTensorTransform on '{type(point_cloud)}'")

        coordinate_idx = get_coordinate_attribute(point_cloud)
        point_before = len(point_cloud.point[coordinate_idx])

        z = point_cloud.point[get_coordinate_attribute(point_cloud)][:, 2]

        if self.min_height is not None and self.max_height is not None:
            mask = (z > self.min_height) & (z < self.max_height)
        elif self.min_height is not None and self.max_height is None:
            mask = (z > self.min_height)
        elif self.min_height is None and self.max_height is not None:
            mask = (z < self.max_height)
        else:
            raise ValueError("HeightFilter does not work with no min and no max height,\
                              no of them or both are required.")
        

        point_cloud = point_cloud.select_by_mask(mask)
        
        point_after = len(point_cloud.point[coordinate_idx])
        # DEBUGGING -> comment when go productive FIXME
        print(f"Height-Filter filtered {point_before-point_after} points.")
        return point_cloud
    


class NaivMinHistoGroundKeepFilterTransform:
    def __init__(self):
        pass

    def __call__(self, point_cloud):
        if not isinstance(point_cloud, o3d.t.geometry.PointCloud):
            raise ValueError(f"Can't apply ToPointCloudTensorTransform on '{type(point_cloud)}'")

        coordinate_idx = get_coordinate_attribute(point_cloud)
        point_before = len(point_cloud.point[coordinate_idx])

        z = point_cloud.point[get_coordinate_attribute(point_cloud)][:, 2]

        # value setup for histogram
        bin_size = 0.05  # 5cm

        z_min = z.min().item()
        z_max = z.max().item()

        # calc histogram
        num_bins = int((z_max - z_min) / bin_size) + 1

        hist = o3d.core.Tensor.zeros((num_bins), dtype=o3d.core.int32)

        # bin indexes
        bin_idx = ((z - z_min) / bin_size).floor().to(o3d.core.int32)

        # count
        for cur_bin_idx in range(bin_idx.shape[0]):
            hist[bin_idx[cur_bin_idx]] += 1

        # biggest bin = ground
        ground_bin = hist.argmax().item()

        # have to upscale from bin_idx to normal value
        z_ground = z_min + ground_bin * bin_size

        # remove everything under and on top of the ground
        mask = (z > (z_ground - 0.5)) & (z < (z_ground + 0.5))
        point_cloud = point_cloud.select_by_mask(mask)
        
        point_after = len(point_cloud.point[coordinate_idx])
        # DEBUGGING -> comment when go productive FIXME
        print(f"NaivMinHistoGroundKeepFilter filtered {point_before-point_after} points.")
        return point_cloud



class RoadExtractionTransform:
    def __init__(self, mode="height"):  # "height", "ransac", None
        self.mode = mode

    def __call__(self, point_cloud):
        if not isinstance(point_cloud, o3d.t.geometry.PointCloud):
            raise ValueError(f"Can't apply ToPointCloudTensorTransform on '{type(point_cloud)}'")

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
        # DEBUGGING -> comment when go productive FIXME
        print(f"Road Extraction filtered {point_before-point_after} points.")
        return point_cloud



class OutlierRemovalTransform:
    def __init__(self, mode="radius", nb_points=4, radius=0.2, std_ratio=2.0):
        """
        Modes:
            - None
            - "radius" (quick, less precise)
            - "statistical" (slow, more precise)
        """
        self.mode = mode
        self.nb_points = nb_points
        self.radius = radius
        self.std_ratio = std_ratio

    def __call__(self, point_cloud):
        if not isinstance(point_cloud, o3d.t.geometry.PointCloud):
            raise ValueError(f"Can't apply ToPointCloudTensorTransform on '{type(point_cloud)}'")

        coordinate_idx = get_coordinate_attribute(point_cloud)
        point_before = len(point_cloud.point[coordinate_idx])

        if self.mode == "radius":
            point_cloud, mask = point_cloud.remove_radius_outliers(
                nb_points=self.nb_points,
                search_radius=self.radius
            )
        elif self.mode == "statistical":
            point_cloud, mask = point_cloud.remove_statistical_outliers(
                nb_neighbors=self.nb_points,
                std_ratio=self.std_ratio
            )
        elif self.mode is None:
            pass
            # just pass
        else:
            raise ValueError(f"Unknown mode for outlier removal '{self.mode}'")
        
        point_after = len(point_cloud.point[coordinate_idx])
        # DEBUGGING -> comment when go productive FIXME
        print(f"Outlier-Removal filtered {point_before-point_after} points.")
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
        if not isinstance(point_cloud, o3d.t.geometry.PointCloud):
            raise ValueError(f"Can't apply ToPointCloudTensorTransform on '{type(point_cloud)}'")

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



class ToDevice:
    def __init__(self, device="cuda"):
        self.device = device
        self.o3d_device = map_torch_device_to_o3d(device)
        
    def __call__(self, point_cloud):
        if isinstance(point_cloud, PointCloudTensor):
            point_cloud = point_cloud.to(self.device)
        elif isinstance(point_cloud, o3d.t.geoemtry.PointCloud):
            point_cloud = point_cloud.to(self.o3d_device)
        else:
            raise ValueError(f"Can't handle type of point cloud: '{type(point_cloud)}'")



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



def get_basic_transform(num_points=-1):
    transform = Compose([
        # RandomRotate(axis='z'),
        # Jitter(sigma=0.01, clip=0.05),
        # NaivMinHistoGroundKeepFilterTransform(),
        # OutlierRemovalTransform(mode="radius", nb_points=4, radius=0.2, std_ratio=2.0),
        # RoadExtractionTransform(mode="height"),
        ToPointCloudTensorTransform(num_points=num_points)
    ])
    return transform



def get_preprocessing_transform():
    transform = Compose([
        # NaivMinHistoGroundKeepFilterTransform(),
        OutlierRemovalTransform(mode="radius", nb_points=4, radius=0.2, std_ratio=2.0),
        RoadExtractionTransform(mode="height")
    ])
    return transform



# -----------
# > Dataset <
# -----------
class ParisLille3DDataset(Dataset):
    def __init__(self, path, testdata=False, transform=None, 
                 preprocessed=False):
        self.path = path
        self.testdata = testdata
        self.transform = transform
        self.preprocessed = preprocessed

        self.POINT_LIMIT = 1024

        if self.testdata:
            self.path = os.path.join(self.path, "test_10_classes")
        else:
            self.path = os.path.join(self.path, "training_10_classes")

        self.point_cloud_paths = []
        for cur_file in os.listdir(self.path):
            if any([cur_file.endswith(ending) for ending in [".las", ".laz", ".ply"]]):
                if preprocessed:
                    cur_file = "preprocessed_" + cur_file
                
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
                    batch_size=32, shuffle=True, num_workers=4,
                    preprocessed=False):
    if data_name == "paris":
        data_loader = get_paris_data_loader(path, testdata=testdata, transform=transform,
                                            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                            preprocessed=preprocessed)
    else:
        raise ValueError(f"No Dataset with the name '{data_name}' founded. Try 'paris'.")

    return data_loader



def get_paris_data_loader(path, testdata=False, transform=None,
                          batch_size=32, shuffle=True, num_workers=4,
                          preprocessed=False):
    dataset = ParisLille3DDataset(path=path, testdata=testdata, transform=transform,
                                  preprocessed=preprocessed)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_point_clouds)



def preprocess_data(data_name, path, testdata=False, transform=None, device="cpu"):
    print("--- Data Preprocessing ---")
    data_loader = get_data_loader(data_name, path, testdata=testdata, transform=transform,
                                  batch_size=1, shuffle=False, num_workers=1)
    
    # to_device = ToDevice(device)
    dataset = data_loader.dataset

    print("Start Preprocessing your dataset...")

    for idx, batch in tqdm(enumerate(data_loader)):
        # batch = to_device(batch)

        # adjust path
        cur_path = dataset.point_cloud_paths[idx]
        cur_root_path, cur_file_name = os.path.split(cur_path)
        new_file_path = os.path.join(cur_root_path, "preprocessed_"+cur_file_name)

        if len(batch) > 1:
            raise ValueError("Not expected bigger Batch.")
        
        save_point_cloud(path=new_file_path, point_cloud=batch[0])

    print("Congratelations, your preprocessing is finish!")



