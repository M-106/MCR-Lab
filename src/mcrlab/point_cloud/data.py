# -----------
# > Imports <
# -----------
import os
import shutil
import gc
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import open3d as o3d
import CSF  # package: cloth-simulation-filter

from tqdm import tqdm

from mcrlab.point_cloud.utils import filter_ground_with_height, filter_ground_with_RANSAC, \
                                     get_coordinate_attribute, get_class_attribute, \
                                     get_intensity_attribute, get_color_attribute, \
                                     get_normal_attribute
from mcrlab.point_cloud.io import load_point_cloud, save_point_cloud
from mcrlab.point_cloud.semantic_kitti_utils import load_semantic_kitti_as_o3d
from mcrlab.point_cloud.tensor_wrapper import PointCloudTensor, map_torch_device_to_o3d
from mcrlab.projection import bev_projection_numba
from mcrlab.image.io import save_bev_tiles_as_pickle, load_bev_tiles_as_pickle, \
                            load_single_bev_tile_as_pickle
# from mcrlab.point_cloud.inspect import print_pc



# ----------------------
# > Data-Augmentations <
# ----------------------
class ToFixPointsTransform:
    def __init__(self, num_points=-1, allow_padding=False, reduction_by_height=True):
        """
        Reduced or pads an point cloud to have one fix point set.
        Reduction is done by taking the highest points.

        -> It will not be recued **by** num_points, it will be reduced **to** this amount
        """
        self.num_points = num_points
        self.allow_padding = allow_padding
        self.reduction_by_height = reduction_by_height

    def __call__(self, point_cloud):
        if not isinstance(point_cloud, o3d.t.geometry.PointCloud):
            raise ValueError(f"Can't apply ToPointCloudTensorTransform on '{type(point_cloud)}'")
        
        if self.num_points <= 0:
            return point_cloud

        # convert to numpy
        coordinates_key = get_coordinate_attribute(point_cloud)
        coordinates = point_cloud.point[coordinates_key].numpy()
        coordinate_amount = coordinates.shape[0]

        points_to_remove = coordinate_amount - self.num_points

        # sample or pad points to fixed size
        if points_to_remove > 0:
            z = coordinates[:, 2]
            idx_sorted = np.argsort(z)     # sorted low to high

            if self.reduction_by_height:
                idxs = idx_sorted[:-points_to_remove]
            else:
                idxs = np.random.choice(coordinate_amount, self.num_points, replace=False)
            idxs = o3d.core.Tensor(idxs, dtype=o3d.core.Dtype.Int64)

            # apply mask/filtered points
            point_cloud = point_cloud.select_by_index(idxs)
        else:
            if self.allow_padding:
                extra_idxs = np.random.choice(coordinate_amount, (self.num_points - coordinate_amount), replace=True)
                idxs = np.hstack((np.arange(coordinate_amount), extra_idxs))  # result also 1 dimensional

                np.random.shuffle(idxs)

                idxs = o3d.core.Tensor(idxs, dtype=o3d.core.Dtype.Int64)
                # apply mask/filtered points
                point_cloud = point_cloud.select_by_index(idxs)

        return point_cloud
    
        # dtype = point_cloud.point[coordinates_idx].dtype
        # coordinates = coordinates[idxs]
        # point_cloud.point[coordinates_idx] = o3d.core.Tensor(coordinates, dtype=dtype)

        # intensities_idx = get_intensity_attribute(point_cloud)
        # if intensities_idx is not None:
        #     dtype = point_cloud.point[intensities_idx].dtype
        #     intensities = point_cloud.point[intensities_idx].numpy()
        #     intensities = intensities[idxs]
        #     point_cloud.point[intensities_idx] = o3d.core.Tensor(intensities, dtype=dtype)

        # colors_idx = get_color_attribute(point_cloud)
        # if colors_idx is not None:
        #     dtype = point_cloud.point[colors_idx].dtype
        #     colors = point_cloud.point[colors_idx].numpy()
        #     colors = colors[idxs]
        #     point_cloud.point[colors_idx] = o3d.core.Tensor(colors, dtype=dtype)

        # normals_idx = get_color_attribute(point_cloud)
        # if normals_idx is not None:
        #     dtype = point_cloud.point[normals_idx].dtype
        #     normals = point_cloud.point[normals_idx].numpy()
        #     normals = normals[idxs]
        #     point_cloud.point[normals_idx] = o3d.core.Tensor(normals, dtype=dtype)
        
        # labels_idx = get_color_attribute(point_cloud)
        # if labels_idx is not None:
        #     dtype = point_cloud.point[labels_idx].dtype
        #     labels = point_cloud.point[labels_idx].numpy()
        #     labels = labels[idxs]
        #     point_cloud.point[labels_idx] = o3d.core.Tensor(labels, dtype=dtype)



class HeightFilterTransform:
    def __init__(self, min_height=-2.0, max_height=2.0):
        self.min_height = min_height
        self.max_height = max_height

    def __call__(self, point_cloud):
        if not isinstance(point_cloud, o3d.t.geometry.PointCloud):
            raise ValueError(f"Can't apply ToPointCloudTensorTransform on '{type(point_cloud)}'")

        coordinate_idx = get_coordinate_attribute(point_cloud)
        # point_before = len(point_cloud.point[coordinate_idx])

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
        
        # point_after = len(point_cloud.point[coordinate_idx])
        # # DEBUGGING -> comment when go productive FIXME
        # print(f"Height-Filter filtered {point_before-point_after} points.")
        return point_cloud
    


# Plane-Fitting
class RANSACGroundKeepFilterTransform:
    def __init__(self, dist_threshold=0.5, ransac_tries=3):
        self.dist_threshold = dist_threshold
        self.ransac_tries = ransac_tries

    def __call__(self, point_cloud):
        if not isinstance(point_cloud, o3d.t.geometry.PointCloud):
            raise ValueError(f"Can't apply RANSACGroundKeepFilterTransform on '{type(point_cloud)}'")

        coordinate_idx = get_coordinate_attribute(point_cloud)
        # point_before = len(point_cloud.point[coordinate_idx])

        # use RANSAC to get ground plane
        best_inliers = []
        best_plane = None

        for _ in range(self.ransac_tries):  # mehrere Versuche
            plane_model, inliers = point_cloud.segment_plane(
                distance_threshold=0.02,
                ransac_n=5,
                num_iterations=20000
            )

            plane_model = plane_model.to(o3d.core.Dtype.Float32)
            a, b, c, d = plane_model.numpy()
            normal = o3d.core.Tensor([a, b, c], dtype=o3d.core.Dtype.Float32)
            # normalize the normal
            normal = (normal - normal.max()) / (normal.max() - normal.min())

            # nur fast horizontale Ebenen akzeptieren
            if abs(normal[2].item()) > 0.9:
                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_plane = plane_model

        # fallback
        if best_plane is None:
            best_plane, best_inliers = plane_model, inliers

        [a, b, c, d] = best_plane
        points = point_cloud.point[coordinate_idx]

        # calc distance to the plane
        # ax + by + cz + d
        # dist = (points[:, 0] * a +
        #         points[:, 1] * b +
        #         points[:, 2] * c + d).abs()
        # -> dist = abs(a*x + b*y + c*z + d) / sqrt(a²+b²+c²)
        dist = (points[:, 0]*a + points[:, 1]*b + points[:, 2]*c + d).abs()
        dist = dist / (a*a + b*b + c*c).sqrt()  # elementwise sqrt equals ** 0.5

        mask = dist < self.dist_threshold
        point_cloud = point_cloud.select_by_mask(mask)
        
        # point_after = len(point_cloud.point[coordinate_idx])
        # # DEBUGGING -> comment when go productive FIXME
        # print(f"RANSACGroundKeepFilterTransform filtered {point_before-point_after} points.")
        return point_cloud



class RoadExtractionTransform:
    def __init__(self, mode="height"):  # "height", "ransac", None
        self.mode = mode

    def __call__(self, point_cloud):
        if not isinstance(point_cloud, o3d.t.geometry.PointCloud):
            raise ValueError(f"Can't apply ToPointCloudTensorTransform on '{type(point_cloud)}'")

        coordinate_idx = get_coordinate_attribute(point_cloud)
        # point_before = len(point_cloud.point[coordinate_idx])

        if self.mode is None:
            pass
        elif self.mode == "height":
            point_cloud = filter_ground_with_height(point_cloud, threshold=1.0)
        elif self.mode == "ransac":
            point_cloud = filter_ground_with_RANSAC(point_cloud, distance_threshold=1.0, ransac_n=3, num_iterations=3000)
        else:
            raise ValueError(f"No `{self.mode}` mode for Road Extraction.")
        
        # point_after = len(point_cloud.point[coordinate_idx])
        # # DEBUGGING -> comment when go productive FIXME
        # print(f"Road Extraction filtered {point_before-point_after} points.")
        return point_cloud



class CSFGroundFilterTransform:
    """
    Cloth-Simulation-Filter

    Idea:
    1. inverse Point-Cloud
    2. Let a cloth fall down on the point cloud -> ground, because we inversed it
    3. now you have exact height differences of the ground and just need a bit height difference
    """
    def __init__(self, invert_z=False):  # , threshold=1.0):
        self.invert_z = invert_z
        # self.threshold = threshold

    def __call__(self, point_cloud):
        if not isinstance(point_cloud, o3d.t.geometry.PointCloud):
            raise ValueError(f"Can't apply ToPointCloudTensorTransform on '{type(point_cloud)}'")

        coordinate_idx = get_coordinate_attribute(point_cloud)
        points = point_cloud.point[coordinate_idx].numpy()
        if self.invert_z:
            points[:, 2] *= -1

        csf = CSF.CSF()

        # parameters
        csf.params.bSloopSmooth = True
        csf.params.cloth_resolution = 0.3   # smaller = more details
        csf.params.rigidness = 3            # 1-3 typically -> how stable the cloth should be
        csf.params.time_step = 0.65
        csf.params.class_threshold = 0.7    # more points = more ground

        csf.setPointCloud(points)

        # results
        ground = CSF.VecInt()
        non_ground = CSF.VecInt()

        # run filter
        csf.do_filtering(ground, non_ground)

        idx = np.array([int(x) for x in ground], dtype=int)
        o3d_idx = o3d.core.Tensor(idx, dtype=o3d.core.int64)
        # ground_points = points[idx]  # points[list(ground)]
        ground_points = point_cloud.select_by_index(o3d_idx)

        # point_cloud.select_by_index(o3d.core.Tensor(ground_points))

        return ground_points



class OutlierRemovalTransform:
    def __init__(self, mode="radius", nb_points=4, radius=0.2, std_ratio=2.0):
        """
        Modes:
            - None
            - "radius" 
            - "statistical" 
        """
        self.mode = mode
        self.nb_points = nb_points
        self.radius = radius
        self.std_ratio = std_ratio

    def __call__(self, point_cloud):
        if not isinstance(point_cloud, o3d.t.geometry.PointCloud):
            raise ValueError(f"Can't apply OutlierRemovalTransform on '{type(point_cloud)}'")

        coordinate_idx = get_coordinate_attribute(point_cloud)
        # point_before = len(point_cloud.point[coordinate_idx])
    
        # debugging
        # pts = point_cloud.point[coordinate_idx].numpy()
        # print("NaN:", np.isnan(pts).any())
        # print("Inf:", np.isinf(pts).any())
        # print("Min:", pts.min(axis=0))
        # print("Max:", pts.max(axis=0))

        if self.mode == "radius":
            point_cloud, mask = point_cloud.remove_radius_outliers(
                nb_points=self.nb_points,
                search_radius=self.radius
            )
        elif self.mode == "statistical":
            # print("Checkpoint 2:", point_cloud.point[coordinate_idx].shape)
            point_cloud, mask = point_cloud.remove_statistical_outliers(
                nb_neighbors=self.nb_points,
                std_ratio=self.std_ratio
            )
        elif self.mode is None:
            pass
            # just pass
        else:
            raise ValueError(f"Unknown mode for outlier removal '{self.mode}'")
        
        # point_after = len(point_cloud.point[coordinate_idx])
        # # DEBUGGING -> comment when go productive FIXME
        # print(f"Outlier-Removal filtered {point_before-point_after} points.")
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
        # FIXME -> numpy conversion directly have right datatype, right?
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



def collate_bev_images(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return {"pixel_values": pixel_values, "labels": labels}



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
        # ToFixPointsTransform(num_points=50000, allow_padding=False, reduction_by_height=True),  # 7250451 -> 5000000
        # NaivMinHistoGroundKeepFilterTransform(),
        OutlierRemovalTransform(mode="statistical", nb_points=8, radius=0.2, std_ratio=2.0),
        # RANSACGroundKeepFilterTransform(dist_threshold=0.5, ransac_tries=3),
        # RoadExtractionTransform(mode="height")
        CSFGroundFilterTransform(invert_z=False)
    ])
    return transform



# -----------
# > Dataset <
# -----------
class ParisLille3DDataset(Dataset):
    def __init__(self, path, testdata=False, transform=None, 
                 preprocessed=False, return_train_format=False):
        self.path = path
        self.testdata = testdata
        self.transform = transform
        self.preprocessed = preprocessed
        self.return_train_format = return_train_format

        self.POINT_LIMIT = 1024

        if self.testdata:
            self.path = os.path.join(self.path, "test_10_classes")
        else:
            self.path = os.path.join(self.path, "training_10_classes")

        self.point_cloud_paths = []
        self.bev_paths = dict()
        preprocesed_path = os.path.join(self.path, "preprocessed")
        path = preprocesed_path if preprocessed else self.path
        for cur_file in os.listdir(path):
            # if any([cur_file.endswith(ending) for ending in [".las", ".laz", ".ply"]]):
            if cur_file.endswith((".las", ".laz", ".ply")):
                if preprocessed and not cur_file.startswith("preprocessed_"):
                    continue
                elif not preprocessed and cur_file.startswith("preprocessed_"):
                    continue

                self.point_cloud_paths.append(os.path.join(path, cur_file))

        if preprocessed:
            self.bev_gen = BEVDataset(path=self.point_cloud_paths, search_files=False, mode="linear")

        print(f"Found {len(self.point_cloud_paths)} point clouds.")

    def __len__(self):
        return len(self.point_cloud_paths)

    def __getitem__(self, idx):
        point_cloud = load_point_cloud(self.point_cloud_paths[idx])
        # point_cloud = point_cloud.voxel_down_sample(voxel_size=0.05)

        if self.transform:
            point_cloud = self.transform(point_cloud)

        # add BEV information
        if isinstance(point_cloud, PointCloudTensor) and self.preprocessed:
            # general_file_name = ".".join(self.point_cloud_paths[idx].split(".")[:-1])
            # root, filename = os.path.split(self.point_cloud_paths[idx])
            # bev_file_name = filename.replace("preprocessed_", "preprocessed_bev_").replace(".ply", ".pkl")
            # bev_path = self.bev_paths[bev_file_name]
            # bevs, meta = load_bev_tiles_as_pickle(bev_path)  
            # bevs, meta = load_bev_tiles_as_pt(bev_path)
            # point_cloud.bevs = bevs
            # point_cloud.meta = meta

            point_cloud.set_bev(self.bev_gen, self.point_cloud_paths[idx])
            
            # cur_bev_gen = self.bev_gen.get_via_bev_filename(self.point_cloud_paths[idx], extract_from_full_ply_path=True)

        # use labels
        if isinstance(point_cloud, PointCloudTensor):
            if point_cloud.labels is not None:
                y = point_cloud.labels
            else:
                y = None

        if self.return_train_format:

            if isinstance(point_cloud, PointCloudTensor):
                if y is None:
                    raise ValueError(f"Can't find labels in PointCloudTensor!")
            else:
                raise ValueError(f"Can't handle Data type `{type(point_cloud)}`")
            return point_cloud.get_as_one_tensor(include_intensity=True), y
        else:
            return point_cloud  # PointCloudTensor or o3d.t.geometry.Tensor

        # labels = extract_labels_as_tensor(point_cloud)

        # if labels:
        #     return (point_cloud, labels)
        # else:
        #     return point_cloud  # shape (1024, 3)


class WHUUrban3DDataset(Dataset):
    def __init__(self, path, testdata=False, transform=None, 
                 preprocessed=False, return_train_format=False):
        self.path = os.path.join(path, "mls", "h5")
        self.testdata = testdata  # see in https://pypi.org/project/pywhu3d/ which scenes are train/val/test split
        self.transform = transform
        self.preprocessed = preprocessed
        self.return_train_format = return_train_format

        self.point_cloud_paths = []
        self.bev_paths = dict()
        preprocesed_path = os.path.join(self.path, "preprocessed")
        path = preprocesed_path if preprocessed else self.path
        for cur_file in os.listdir(path):
            # if any([cur_file.endswith(ending) for ending in [".las", ".laz", ".ply"]]):
            if cur_file.endswith((".h5", ".ply")):
                if preprocessed and not cur_file.startswith("preprocessed_"):
                    continue
                elif not preprocessed and cur_file.startswith("preprocessed_"):
                    continue

                self.point_cloud_paths.append(os.path.join(path, cur_file))

        if preprocessed:
            self.bev_gen = BEVDataset(path=self.point_cloud_paths, search_files=False, mode="linear")

        print(f"Found {len(self.point_cloud_paths)} point clouds.")

    def __len__(self):
        return len(self.point_cloud_paths)

    def __getitem__(self, idx):
        point_cloud = load_point_cloud(self.point_cloud_paths[idx])
        # point_cloud = point_cloud.voxel_down_sample(voxel_size=0.05)

        # print("Empty? ->", point_cloud.is_empty())
        # print(type(point_cloud))
        # print("Checkpoint 1:", point_cloud.point[get_coordinate_attribute(point_cloud)].shape)

        if self.transform:
            point_cloud = self.transform(point_cloud)

        # add BEV information
        if isinstance(point_cloud, PointCloudTensor) and self.preprocessed:
            # general_file_name = ".".join(self.point_cloud_paths[idx].split(".")[:-1])
            # root, filename = os.path.split(self.point_cloud_paths[idx])
            # bev_file_name = filename.replace("preprocessed_", "preprocessed_bev_").replace(".ply", ".pkl")
            # bev_path = self.bev_paths[bev_file_name]
            # bevs, meta = load_bev_tiles_as_pickle(bev_path)  
            # bevs, meta = load_bev_tiles_as_pt(bev_path)
            # point_cloud.bevs = bevs
            # point_cloud.meta = meta

            point_cloud.set_bev(self.bev_gen, self.point_cloud_paths[idx])
            
            # cur_bev_gen = self.bev_gen.get_via_bev_filename(self.point_cloud_paths[idx], extract_from_full_ply_path=True)

        # use labels
        if isinstance(point_cloud, PointCloudTensor):
            if point_cloud.labels is not None:
                y = point_cloud.labels
            else:
                y = None

        if self.return_train_format:

            if isinstance(point_cloud, PointCloudTensor):
                if y is None:
                    raise ValueError(f"Can't find labels in PointCloudTensor!")
            else:
                raise ValueError(f"Can't handle Data type `{type(point_cloud)}`")
            return point_cloud.get_as_one_tensor(include_intensity=True), y
        else:
            print(point_cloud.point[get_coordinate_attribute(point_cloud)].shape)
            print(f"Any nans: {np.isnan(np.asarray(point_cloud.point[get_coordinate_attribute(point_cloud)])).any()}")
            return point_cloud  # PointCloudTensor or o3d.t.geometry.Tensor



class SemanticKittiDataset(Dataset):
    """
    FIXME -> NOT TESTED YET!

    To do:
    - need label mapping?
    - add preprocessing 
    """
    def __init__(self, path, transform=None, 
                 preprocessed=False, return_train_format=False):
        self.path = path
        self.sequences_path = os.path.join(self.path, "sequences")
        self.transform = transform
        self.preprocessed = preprocessed
        self.return_train_format = return_train_format

        self.frames = []
        for cur_sequence in os.listdir(self.sequences_path):
            data_dir = os.path.join(self.sequences_path, cur_sequence, "velodyne")
            labels_dir = os.path.join(self.sequences_path, cur_sequence, "labels")

            for cur_frame_name in os.listdir(data_dir):
                if cur_frame_name.endswith(".bin"):
                    cur_frame_data_path = os.path.join(data_dir, cur_frame_name)
                    cur_frame_labels_path = os.path.join(labels_dir, cur_frame_name)

                    if os.path.exists(cur_frame_labels_path):
                        self.frames.append(
                            (cur_frame_data_path,
                             cur_frame_labels_path)
                        )
                    else:
                        self.frames.append(
                            (cur_frame_data_path, None)
                        )
        
        print(f"Loaded {len(self.frames)} frames")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        data_path, labels_path = self.frames[idx]
        point_cloud = load_semantic_kitti_as_o3d(data_path, labels_path)
        
        if self.transform:
            point_cloud = self.transform(point_cloud)

        if self.return_train_format:
            if not isinstance(point_cloud, PointCloudTensor):
                raise ValueError(f"Can't handle Data type `{type(point_cloud)}`")
            if point_cloud.labels is None:
                raise ValueError("Labels missing!")

            return point_cloud.get_as_one_tensor(include_intensity=True), point_cloud.labels

        return point_cloud


class BEVDataset(Dataset):
    """
    Dataset for loading BEV image datasets. 
    Must preprocess the bev images.

    Independent of the dataset, because it works 
    on top of our own preprocessing creation.

    Uses the collate_bev_images function for the dataloader.

    1 Sample = 1 Tile
    """
    def __init__(self, path=None, search_files=False, mode="linear", file_paths=[]):
        """
        path is a list of point cloud file or a list of paths to search the bev images.

        It is designed to be used on before preprocessed data!

        FIXME -> also make working on not preprocessed data and create BEV on the fly -> good idea?

        mode can be "linear" or "dataset"
            - in "linear" every tile will be exported one after another
            - in "dataset" a generator of tiles in one point_cloud will be returned
        """
        if isinstance(path, str):
            path = [path]

        self.path = path
        self.mode = mode
        self.file_paths = file_paths

        # find pkl files
        all_bev_paths = []
        # search all files
        if search_files:
            for cur_path in self.path:
                for cur_file in os.listdir(cur_path):
                    if cur_file.endswith(".pkl") and cur_file.startswith("preprocessed_bev_"):
                        cur_file_path = os.path.join(cur_path, cur_file)
                        all_bev_paths.append(cur_file_path)            
        else:
            for cur_path in self.path:
                root, filename = os.path.split(cur_path)
                bev_file_name = filename.replace("preprocessed_", "preprocessed_bev_").replace(".ply", ".pkl")
                all_bev_paths.append(os.path.join(root, bev_file_name))
                
        # merge the found files
        #    -> the files only contain the file-paths
        self.bev_paths = dict()
        self.bev_tile_mapping = []  # every tile (one image) gets file, id in file info
        self.idx_to_fileid = dict()
        cur_pc_idx = 0

        for cur_path in all_bev_paths:
            cur_root, cur_file = os.path.split(cur_path)

            with open(cur_path, "rb") as file_:
                all_paths = pickle.load(file_)
            self.bev_paths[cur_file] = all_paths
            
            for file_idx in all_paths:
                self.bev_tile_mapping.append((cur_pc_idx, file_idx))
                self.idx_to_fileid[(cur_pc_idx, file_idx)] = cur_file
            cur_pc_idx += 1

        print(f"Found {len(self.bev_tile_mapping)} bev images (orthogonal images) in {len(self.bev_paths)} files.")

    def __getitem__(self, idx):
        if self.mode == "linear":
            file_idx, tile_id = self.bev_tile_mapping[idx]
            file_name = self.idx_to_fileid[(file_idx, tile_id)]
            bevs, meta = load_single_bev_tile_as_pickle(self.bev_paths[file_name])
            bev = bevs[tile_id]

            x = torch.from_numpy(bev[:-1]).float()
            y = torch.from_numpy(bev[-1]).long()
            assert y.ndim == 2

            return {
                "pixel_values": x,   # (C, H, W)
                "labels": y,  # .reshape((bev.shape[1], bev.shape[2]))          # (H, W)
                "meta": meta
            }
        else:
            cur_tile_ids = []
            for file_idx, tile_id in self.bev_tile_mapping:
                if file_idx == idx:
                    cur_tile_ids.append(tile_id)
                else:
                    break

            return self.generator(cur_tile_ids)
        
    def get_via_bev_filename(self, file_name, extract_from_full_ply_path=False):
        if extract_from_full_ply_path:
            bev_file_name = self.extract_from_full_ply_path(file_name)
        else:
            bev_file_name = file_name
        # bev_file_name = file_name.replace("preprocessed_", "preprocessed_bev_").replace(".ply", ".pkl")
        return self.generator(self.bev_paths[bev_file_name])

    def generator(self, file_paths=None):
        if file_paths is None:
            if self.file_paths is None:
                raise ValueError("No filenames set and no filenames given")
            else:
                file_paths = self.file_paths

        for cur_file_path in file_paths:
            tile, meta = load_single_bev_tile_as_pickle(cur_file_path)
            x = torch.from_numpy(tile[:-1]).float()
            y = torch.from_numpy(tile[-1]).long()
            assert y.ndim == 2
            yield {
                "pixel_values": x,   # (C, H, W)
                "labels": y,  # .reshape((bev.shape[1], bev.shape[2]))          # (H, W)
                "meta": meta
            }

    def extract_from_full_ply_path(self, path):
        _, bev_file_name = os.path.split(path)
        bev_file_name = bev_file_name.replace("preprocessed_", "preprocessed_bev_").replace(".ply", ".pkl")
        return bev_file_name

    def __len__(self):
        return len(self.bev_tile_mapping)
    
def bev_gen_wrapper(self, tiles, metas):
        for idx in range(len(tiles)):
            tile = tiles[idx]
            meta = metas[idx]
            x = torch.from_numpy(tile[:-1]).float()
            y = torch.from_numpy(tile[-1]).long()
            assert y.ndim == 2
            yield {
                "pixel_values": x,   # (C, H, W)
                "labels": y,  # .reshape((bev.shape[1], bev.shape[2]))          # (H, W)
                "meta": meta
            }

def extract_tiles_metas(bev_gen, amount=5, as_numpy=True):
    tiles = []
    metas = []
    cur_amount = 0
    for bev_data in bev_gen:
        if cur_amount == amount:
            break

        if bev_data["labels"] is not None and bev_data["labels"].shape[0] == bev_data["pixel_values"].shape[1]:
            bev = torch.cat((bev_data["pixel_values"], 
                            bev_data["labels"].unsqueeze(0)),
                            dim=0
                            )
        else:
            bev = bev_data["pixel_values"]
        if as_numpy:
            bev = bev.cpu().detach().numpy()
        tiles.append(bev)

        meta = bev_data["meta"]
        metas.append(meta)

        cur_amount += 1

    return (tiles, metas)
    

# also add data_loader which return 3 data-loader or 2
# -> train, val, eval
# -> val is simply a random split from train



def get_data_loader(data_name, path, testdata=False, transform=None,
                    batch_size=32, shuffle=True, num_workers=4,
                    preprocessed=False, return_train_format=False):
    if data_name == "paris":
        data_loader = get_paris_data_loader(path, testdata=testdata, transform=transform,
                                            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                            preprocessed=preprocessed, return_train_format=return_train_format)
    elif data_name == "whu":
        data_loader = get_whu_data_loader(path, testdata=testdata, transform=transform,
                                          batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                          preprocessed=preprocessed, return_train_format=return_train_format)
    else:
        raise ValueError(f"No Dataset with the name '{data_name}' founded. Try 'paris'.")

    return data_loader



def get_paris_data_loader(path, testdata=False, transform=None,
                          batch_size=32, shuffle=True, num_workers=4,
                          preprocessed=False, return_train_format=False):
    dataset = ParisLille3DDataset(path=path, testdata=testdata, transform=transform,
                                  preprocessed=preprocessed, return_train_format=return_train_format)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_point_clouds)



def get_whu_data_loader(path, testdata=False, transform=None,
                          batch_size=32, shuffle=True, num_workers=4,
                          preprocessed=False, return_train_format=False):
    dataset = WHUUrban3DDataset(path=path, testdata=testdata, transform=transform,
                                preprocessed=preprocessed, return_train_format=return_train_format)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_point_clouds)



def preprocess_data(data_name, path, testdata=False, transform=None, device="cpu",
                    bev_tile_size=15.0, bev_resolution=0.01):
    print("--- Data Preprocessing ---")
    data_loader = get_data_loader(data_name, path, testdata=testdata, transform=transform,
                                  batch_size=1, shuffle=False, num_workers=0, preprocessed=False,
                                  return_train_format=False)
    
    # to_device = ToDevice(device)
    dataset = data_loader.dataset

    print("Start Preprocessing your dataset...")

    preprocessed_path_cleaned = False
    for idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Preprocessing"):
        # batch = to_device(batch)

        # adjust path
        cur_path = dataset.point_cloud_paths[idx]
        cur_root_path, cur_file_name = os.path.split(cur_path)
        cur_file_name = ".".join(cur_file_name.split(".")[:-1])

        cur_root_path = os.path.join(cur_root_path, "preprocessed")
        if not preprocessed_path_cleaned:
            os.makedirs(cur_root_path, exist_ok=True)
            shutil.rmtree(cur_root_path)
            os.makedirs(cur_root_path, exist_ok=True)
            preprocessed_path_cleaned = True

        new_file_path = os.path.join(cur_root_path, "preprocessed_"+cur_file_name +".ply")

        if len(batch) > 1:
            raise ValueError("Not expected bigger Batch.")
        
        # print(f"Data-Type: {type(batch)}")
        # print(f"Data-Shape: {batch.shape}") if hasattr(batch, "shape") else ""
        # print(f"Data-Type (batch 0): {type(batch[0])}")
        # print_pc(batch[0])

        save_point_cloud(path=new_file_path, point_cloud=batch[0])

        # save BEVs
        print("Generating BEV images...")
        bev_file_path = os.path.join(cur_root_path, "preprocessed_bev_"+cur_file_name +".pkl")  # ".pkl"
        tiles, meta = bev_projection_numba(batch[0], tile_size=bev_tile_size, resolution=bev_resolution, include_class=True)
        save_bev_tiles_as_pickle(tiles, meta, bev_file_path)
        # save_bev_tiles_as_pt(tiles, meta, bev_file_path)
        print(f"Saving BEVs to '{bev_file_path}'\n  Found: {os.path.isfile(bev_file_path)}")

        del tiles, meta, batch
        gc.collect()

    print("\nCongratelations, your preprocessing is finish!")







