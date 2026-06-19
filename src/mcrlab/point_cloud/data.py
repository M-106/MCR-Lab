# -----------
# > Imports <
# -----------
import os
import shutil
import gc
import pickle
import random
import copy

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# from skimage.ndimage import label as ski_label
from sklearn.cluster import DBSCAN
import cv2

import open3d as o3d
import CSF  # package: cloth-simulation-filter

import albumentations as A

from tqdm import tqdm

from mcrlab.point_cloud.utils import filter_ground_with_height, filter_ground_with_RANSAC, \
                                     get_coordinate_attribute, get_class_attribute, \
                                     get_intensity_attribute, get_color_attribute, \
                                     get_normal_attribute, get_instance_attribute, \
                                     split_point_cloud_into_multiple
from mcrlab.point_cloud.io import load_point_cloud, save_point_cloud
from mcrlab.point_cloud.semantic_kitti_utils import load_semantic_kitti_as_o3d
from mcrlab.point_cloud.tensor_wrapper import PointCloudTensor, map_torch_device_to_o3d
from mcrlab.projection import bev_projection
from mcrlab.image.io import load_single_bev_tile_as_pickle
from mcrlab.image.utils import normalize_img_per_channel, normalize_bev
from mcrlab.point_cloud.shape_check import circle_shape_check



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



class SegmentationGroundKeepFilterTransform:
    def __init__(self):
        pass

    def __call__(self, point_cloud):
        if not isinstance(point_cloud, o3d.t.geometry.PointCloud):
            raise ValueError(f"Can't apply ToPointCloudTensorTransform on '{type(point_cloud)}'")

        plane_model, inliers = point_cloud.segment_plane(
            distance_threshold=0.02,
            ransac_n=3,
            num_iterations=1000
        )

        road_cloud = point_cloud.select_by_index(inliers)

        return road_cloud



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



class VoxelDownsamplerTransform:
    """
    Voxel downsampling summarizes all points in the same grid.
    grid-size is in meters -> 0,01 means 1cm grids 
    
    if creating BEV, you have to be carefully 
    do not dwnsampling more than you want to show in the BEV
    """
    def __init__(self, grid_size=0.01):
        self.grid_size = grid_size

    def __call__(self, point_cloud):
        if not isinstance(point_cloud, o3d.t.geometry.PointCloud):
            raise ValueError(f"Can't apply OutlierRemovalTransform on '{type(point_cloud)}'")

        
        return point_cloud.voxel_down_sample(voxel_size=self.grid_size)



class FilterAndRelabelingTransform:
    """
    Filters manhole labels and relabels to:
    - 0: no manholes
    - 1: manholes
    - 255: ignore -> special cases (not round manholes)
    """
    def __init__(self, manhole_label=104002):
        self.manhole_label = manhole_label

    def __call__(self, point_cloud, ):
        if not isinstance(point_cloud, o3d.t.geometry.PointCloud):
            raise ValueError(f"Can't apply FilterAndRelabelingTransform on '{type(point_cloud)}'")

        semantic_key = get_class_attribute(point_cloud)
        semantics = point_cloud.point[semantic_key].cpu().numpy()

        positions = point_cloud.point[get_coordinate_attribute(point_cloud)].cpu().numpy()

        semantic_out = np.zeros(len(semantics), dtype=np.int32)

        # filter labels -> only instances with label 104002
        mask = semantics == self.manhole_label
        mask_indices = np.where(mask)[0]
        # manhole_points = positions[mask]
        
        if len(mask_indices) == 0:
            point_cloud.point[semantic_key] = o3d.core.Tensor(
                semantic_out, dtype=o3d.core.Dtype.Int32
            )
            return point_cloud
        
        manhole_points = positions[mask_indices]

        instance_key = get_instance_attribute(point_cloud)
        if instance_key is not None:
            # use existing instance labels only for manhole points
            all_instances = point_cloud.point[instance_key].cpu().numpy().squeeze()

            # keep only points that are semantic manholes
            clusters = all_instances[mask_indices]

            # ignore invalid/noise instances if present
            # valid = clusters >= 0
            valid = np.full(clusters.shape, True, dtype=np.bool)
        else:
            # get manhole instances via clustering
            clustering = DBSCAN(eps=0.3, min_samples=20).fit(manhole_points)
            clusters = clustering.labels_

            # remove noise (-1)
            valid = clusters >= 0

        # process each cluster
        for cluster_id in np.unique(clusters[valid]):
            # get manhole with current cluster id
            local_mask = clusters == cluster_id
            global_indices = mask_indices[local_mask]
            instance_points = positions[global_indices]
  
            is_circle, _ = circle_shape_check(
                            instance_points,
                            save_path=None,
                            should_plot=False,
                            threshold=0.6
                        )
            
            if is_circle:
                semantic_out[global_indices] = 1
            else:
                semantic_out[global_indices] = 255

            # everything else stays 0

        # ignore noisy clusters/manholes
        noise_mask = clusters == -1
        noise_indices = np.where(mask)[0][noise_mask]
        semantic_out[noise_indices] = 255

        # add to point-cloud again
        # point_cloud.point[instance_key] = o3d.core.Tensor(instances, dtype=o3d.core.Dtype.Int32)
        point_cloud.point[semantic_key] = o3d.core.Tensor(semantic_out, dtype=o3d.core.Dtype.Int32)

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

        data.instances = extract_instances_as_tensor(point_cloud)

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
            if data.instances is not None:
                data.instances = data.instances[idxs]

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



def point_cloud_tensor_to_image_dataset(point_cloud):
    """
    This function is needed? FIXME-> maybe remve it

    Wanted format:
    {
        "pixel_values": tensor(C, H, W),
        "labels": tensor(H, W)  # class ids per pixel
    }
    """
    tiles, meta = bev_projection(point_cloud, tile_size=35.0, resolution=0.05, overlap=0.5)

    items = []
    for cur_tile in tiles:
        cur_img = np.transpose(cur_tile, (1, 2, 0))
        cur_img = normalize_img_per_channel(cur_img, skip_already_normalized_channels=True)

        cur_labels = cur_img[:, :, 3]
        cur_img = cur_img[:, :, :-1]
        
        items.append({
            "pixel_values": cur_img,
            "labels": cur_labels
        })
        
    return items



def extract_labels_as_tensor(point_cloud):
    label_idx = get_class_attribute(point_cloud)
    
    if label_idx:
        labels = torch.tensor(point_cloud.point[label_idx].numpy())
    else:
        labels = None

    return labels



def extract_instances_as_tensor(point_cloud):
    label_idx = get_instance_attribute(point_cloud)
    
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



def get_preprocessing_transform(grid_size=0.01, do_voxelation=True, manhole_label=104002):
    transformations = [
        # ToFixPointsTransform(num_points=1000000, allow_padding=False, reduction_by_height=True),  # 7250451 -> 5000000
        # NaivMinHistoGroundKeepFilterTransform(),
        OutlierRemovalTransform(mode="statistical", nb_points=8, radius=0.2, std_ratio=2.0),
        # RANSACGroundKeepFilterTransform(dist_threshold=0.5, ransac_tries=3),
        # RoadExtractionTransform(mode="height")
        CSFGroundFilterTransform(invert_z=False),
        # SegmentationGroundKeepFilterTransform
        FilterAndRelabelingTransform(manhole_label=manhole_label)
    ]

    if do_voxelation:
        transformations.append(VoxelDownsamplerTransform(grid_size=grid_size))
    
    transform = Compose(transformations)
    return transform



# -----------
# > Dataset <
# -----------
class ParisLille3DDataset(Dataset):
    def __init__(self, path, type="train", transform=None, 
                 preprocessed=False, return_train_format=False):
        self.path = path
        self.type = type
        self.transform = transform
        self.preprocessed = preprocessed
        self.return_train_format = return_train_format

        self.POINT_LIMIT = 1024

        if self.type == "test":
            self.path = os.path.join(self.path, "test_10_classes")
        elif self.type == "train":
            self.path = os.path.join(self.path, "training_10_classes")
        else:
            raise ValueError(f"Got unknown dataset type: {self.type}")

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
            self.bev_gen = BEVDataset(path=self.point_cloud_paths, type=self.type, search_files=False, mode="linear", has_labels=False if self.type == "inference" else True)

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


# FIXME why only find Found 28 point clouds??? -> because of preprocessed flag, but where is it used?
class WHUUrban3DDataset(Dataset):
    def __init__(self, path, type="train", transform=None, 
                 preprocessed=False, return_train_format=False):
        self.path = os.path.join(path, "mls", "h5")
        self.type = type  # see in https://pypi.org/project/pywhu3d/ which scenes are train/val/test split
        self.transform = transform
        self.preprocessed = preprocessed
        self.return_train_format = return_train_format

        self.train_ids = ['8018', '4938', '0414', '2002', '0444', '1046', '5642', '4333', '4629', '0424', '2421', '0947', '0434', '2022', '2719', '2810', '8048', '2423', '2522', '8008', '0502', '6017', '3918', '2422', '2322', '3405', '2323', '8038']
        self.val_ids = ['0404', '6027', '3648']
        self.test_ids = ['0940', '2447', '6037', '2321', '8028', '5627', '2521']

        if preprocessed:
            self.train_ids = ['preprocessed_patch_'+x for x in self.train_ids]
            self.val_ids = ['preprocessed_patch_'+x for x in self.val_ids]
            self.test_ids = ['preprocessed_patch_'+x for x in self.test_ids]

        self.point_cloud_paths = []
        self.bev_paths = dict()
        preprocesed_path = os.path.join(self.path, "preprocessed")
        path = preprocesed_path if preprocessed else self.path
        for cur_file in os.listdir(path):
            # if any([cur_file.endswith(ending) for ending in [".las", ".laz", ".ply"]]):
            if cur_file.endswith((".h5", ".ply")):
                if preprocessed and not cur_file.startswith("preprocessed_patch"):
                    continue
                elif not preprocessed and cur_file.startswith("preprocessed_patch"):
                    continue

                if self.type == "train" and not any([cur_file.startswith(cur_id) for cur_id in self.train_ids]):
                    continue
                elif self.type == "test" and not any([cur_file.startswith(cur_id) for cur_id in self.test_ids]):
                    continue
                elif self.type == "val" and not any([cur_file.startswith(cur_id) for cur_id in self.val_ids]):
                    continue

                self.point_cloud_paths.append(os.path.join(path, cur_file))

        if preprocessed:
            self.bev_gen = BEVDataset(path=self.point_cloud_paths, has_labels=False if self.type == "inference" else True)

        print(f"Found {len(self.point_cloud_paths)} point clouds.")

    def __len__(self):
        return len(self.point_cloud_paths)

    def __getitem__(self, idx):
        point_cloud = load_point_cloud(self.point_cloud_paths[idx])

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

        # FIXME check if complete?

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
            # print(point_cloud.point[get_coordinate_attribute(point_cloud)].shape)
            return point_cloud  # PointCloudTensor or o3d.t.geometry.Tensor



class SUDROADDataset(Dataset):
    def __init__(self, path, type="train", transform=None, 
                 preprocessed=False, return_train_format=False):
        self.path = path
        self.type = type  # have the additional special type not_splitted
        self.transform = transform
        self.preprocessed = preprocessed
        self.return_train_format = return_train_format

        self.train_ids = ['8', '10', '1', '11', '3', '4', '6', '5', '7']
        self.val_ids = ['0']
        self.test_ids = ['2', '9']

        if preprocessed:
            self.train_ids = ['preprocessed_patch_'+x for x in self.train_ids]
            self.val_ids = ['preprocessed_patch_'+x for x in self.val_ids]
            self.test_ids = ['preprocessed_patch_'+x for x in self.test_ids]

        self.point_cloud_paths = []
        self.bev_paths = dict()

        if self.type == "load_unsplitted":
            self.point_cloud_paths.append(os.path.join(path, "SUD_road.las"))
        else:
            preprocesed_path = os.path.join(self.path, "preprocessed")
            path = preprocesed_path if preprocessed else self.path

            for cur_file in os.listdir(path):
                # if any([cur_file.endswith(ending) for ending in [".las", ".laz", ".ply"]]):
                if cur_file.endswith((".ply", ".h5")):
                    if cur_file == "SUD_road.las":
                        continue

                    if preprocessed and not cur_file.startswith("preprocessed_patch"):
                        continue
                    elif not preprocessed and cur_file.startswith("preprocessed_patch"):
                        continue

                    if self.type == "train" and not any([cur_file.startswith(cur_id) for cur_id in self.train_ids]):
                        continue
                    elif self.type == "test" and not any([cur_file.startswith(cur_id) for cur_id in self.test_ids]):
                        continue
                    elif self.type == "val" and not any([cur_file.startswith(cur_id) for cur_id in self.val_ids]):
                        continue

                    self.point_cloud_paths.append(os.path.join(path, cur_file))

        if preprocessed:
            self.bev_gen = BEVDataset(path=self.point_cloud_paths, has_labels=False if self.type == "inference" else True)

        print(f"Found {len(self.point_cloud_paths)} point clouds.")

    def __len__(self):
        return len(self.point_cloud_paths)

    def __getitem__(self, idx):
        point_cloud = load_point_cloud(self.point_cloud_paths[idx])

        if self.type == "load_unsplitted":
            return point_cloud

        if self.transform:
            point_cloud = self.transform(point_cloud)

        # add BEV information
        if isinstance(point_cloud, PointCloudTensor) and self.preprocessed:
            point_cloud.set_bev(self.bev_gen, self.point_cloud_paths[idx])
            
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



def custom_augmentation(image, augmentation_type="noise", dropout=0.1, **kwargs):
    """
    Custom function to apply augmentations ONLY to Channel 1 (Intensity).

    Make sure that your data is in 0-1 range when using this augmentation!

    Expected image shape: (H, W, 3) -> [Max Height, Intensity, Density]
    """
    # Create a copy so we don't overwrite the original data in-place
    img = image.copy()
    # n_channels = img.shape[-1]

    # for cur_channel_idx in range(n_channels): 
    cur_channel_idx = 1
    channel = img[:, :, cur_channel_idx] # Extract the channel
    
    if augmentation_type == "shift":
        shift = np.random.uniform(-0.1, 0.1)
        channel = np.clip(channel + shift, 0.0, 1.0)
        
    elif augmentation_type == "dropout":
        dropout_p = dropout
        mask = np.random.random(channel.shape) < dropout_p
        channel[mask] = 0.0
        
    elif augmentation_type == "noise":
        noise = np.random.normal(0, 0.02, size=channel.shape)
        channel = np.clip(channel + noise, 0.0, 1.0)

    elif augmentation_type == "local_contrast":
        alpha = np.random.uniform(0.7, 1.3)
        mean = np.mean(channel)
        channel = np.clip((channel - mean) * alpha + mean, 0.0, 1.0)
    
    elif augmentation_type == "blur":
        k = np.random.choice([3, 5, 7])
        channel = cv2.GaussianBlur(channel, (k, k), 0)

    else:
        raise ValueError(f"Did not found augmentation type: '{augmentation_type}'")

    # assign augmented channel
    img[:, :, cur_channel_idx] = channel
    return img



def get_bev_augmentations():
    return A.Compose([
        # 1. Rotations (90, 180, 270) are "lossless" for grid data
        A.RandomRotate90(p=0.15),
        
        # 2. Horizontal and Vertical Flips
        A.HorizontalFlip(p=0.15),
        A.VerticalFlip(p=0.15),
        
        # 3. Fine-grained rotation for circular manhole variety
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=45, p=0.15),
        
        # A.RandomBrightnessContrast(p=0.2),
        # A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

        # 4. Custom Intensity Shifting (Only Channel 1)
        A.Lambda(
            name="Shift",
            image=lambda img, **kwargs: custom_augmentation(img, "shift"),
            p=0.15
        ),

        # 5. Custom Local Intensity Shifting (Only Channel 1)
        A.Lambda(
            name="LocalShift",
            image=lambda img, **kwargs: custom_augmentation(img, "local_contrast"),
            p=0.15
        ),

        # 6. Custom Local Intensity Shifting (Only Channel 1)
        A.Lambda(
            name="Blur",
            image=lambda img, **kwargs: custom_augmentation(img, "blur"),
            p=0.15
        ),
        
        # 7. Custom Dropout (Only Channel 1)
        # A.Lambda(
        #     name="Dropout",
        #     image=lambda img, **kwargs: custom_augmentation(img, "dropout", 0.2),
        #     p=0.1
        # ),
        
        # # 8. Custom Noise (Only Channel 1)
        # A.Lambda(
        #     name="Noise",
        #     image=lambda img, **kwargs: custom_augmentation(img, "noise"),
        #     p=0.1
        # ),
    ])

class BEVDataset(Dataset):
    """
    Dataset for loading BEV image datasets. 
    Must preprocess the bev images.

    Independent of the dataset, because it works 
    on top of our own preprocessing creation.

    Uses the collate_bev_images function for the dataloader.

    1 Sample = 1 Tile
    """
    def __init__(self, path=None, 
                 file_paths=[], has_labels=False,
                 image_training=False, preprocessor=None,
                 augment=False, pass_label_in_preprocessor=False):
        """
        path is a list of point cloud file or a list of paths to search the bev images.

        It is designed to be used on before preprocessed data!

        mode can be "linear" or "dataset"
            - in "linear" every tile will be exported one after another
            - in "dataset" a generator of tiles in one point_cloud will be returned
        """
        if isinstance(path, str):
            path = [path]

        self.path = path
        self.file_paths = file_paths
        self.has_labels = has_labels
        self.image_training = image_training
        self.preprocessor = preprocessor
        self.augment = augment
        self.aug_pipeline = get_bev_augmentations() if augment else None
        self.pass_label_in_preprocessor = pass_label_in_preprocessor

        # find pkl files
        all_bev_paths = []
                  
        if self.path is not None:
            for cur_path in self.path:
                root, filename = os.path.split(cur_path)

                filename = ".".join(filename.split(".")[:-1]) + ".pkl"
                bev_file_name = filename.replace("preprocessed_patch", "preprocessed_patch_bev") 
                all_bev_paths.append(os.path.join(root, bev_file_name))

        self.file_paths = copy.deepcopy(all_bev_paths)

        if file_paths is not None:
            self.file_paths += file_paths


        # get unique grid naming (for accessing tiles/patches via identifier = pcid_xstart_ystart)
        self.file_paths_dict = dict()
        for cur_file_path in self.file_paths:
            cur_file_root, cur_file_name = os.path.split(cur_file_path)

            pc_id, x_start, y_start = self.extract_grid_identifier(cur_file_name)  # cur_file_name.replace("preprocessed_patch_bev_", "").split("_")[:3]

            self.file_paths_dict[f"{pc_id}_{x_start}_{y_start}"] = cur_file_path

        print(f"Found {len(self.file_paths)} bev images (orthogonal images).")

    def __getitem__(self, idx):
        cur_file_path = self.file_paths[idx]

        pc_id, x_start, y_start = self.extract_grid_identifier(cur_file_path)

        tile, meta = load_single_bev_tile_as_pickle(cur_file_path)

        if self.has_labels:

            x_np = tile[:-1].transpose(1, 2, 0)
            y_np = tile[-1]

            if self.augment:
                augmented = self.aug_pipeline(image=x_np, mask=y_np)
                x_np = augmented['image']
                y_np = augmented['mask']

            # point_exist_mask = np.where(y_np == 255, 0.0, 1.0).astype(np.float32)

            x = normalize_bev(x_np.transpose(2, 0, 1))
            x = torch.from_numpy(x).float()
            if self.image_training:
                x = x[[0,2,3]]     # drop channel
                # x = x[[1,2,3]]
                # x[2] = torch.from_numpy(point_exist_mask)

                # raise ValueError(f"DEBUGGING STOP -> Shape x: {x_np.shape} -> Shape y: {y_np.shape}")

                if self.preprocessor is not None:
                    # HF preprocessors expect numpy (H, W, C) or PIL
                    x_np = x.permute(1, 2, 0).numpy()  # (H, W, C)
                    if self.pass_label_in_preprocessor:
                        y_np = y_np.astype(np.int32)
                        processed = self.preprocessor(
                            images=x_np,
                            segmentation_maps=y_np,
                            return_tensors="pt"
                        )
                        x = processed["pixel_values"].squeeze(0)  # (C, H, W)
                        return {
                            "pixel_values": x,
                            "mask_labels": processed["mask_labels"][0],    # squeeze batch dim
                            "class_labels": processed["class_labels"][0],
                            "labels": torch.from_numpy(y_np).long(),
                            "meta": meta
                        }
                    else:
                        processed = self.preprocessor(
                            images=x_np,
                            return_tensors="pt"
                        )
                    # FIXME, OneFromer need: task_inputs=["semantic"]?
                    x = processed["pixel_values"].squeeze(0)  # (C, H, W)
                else:
                    # x = (x - x.mean()) / (x.std() + 1e-6)
                    # Update me!
                    x[:2] = (x[:2] - x[:2].mean()) / (x[:2].std() + 1e-6)
                assert x.ndim == 3

            y = torch.from_numpy(y_np).long()
            assert y.ndim == 2

            return {
                "pixel_values": x,   # (C, H, W)
                "labels": y,  # .reshape((bev.shape[1], bev.shape[2]))          # (H, W)
                "meta": meta
            }
        else:
            x = normalize_bev(tile)
            x = torch.from_numpy(x).float()
            return {
                "pixel_values": x,   # (C, H, W)
                "labels": None,
                "meta": meta
            }

    def manhole_filter(self, required_manhole_points=200, amount_non_manhole_samples=10):
        new_file_paths = []

        all_non_manhole_paths = []

        for idx in range(len(self.file_paths)):
            cur_file_path = self.file_paths[idx]

            tile, meta = load_single_bev_tile_as_pickle(cur_file_path)

            labels = tile[-1]

            manhole_points = np.sum(labels == 1)

            if manhole_points >= required_manhole_points:
                new_file_paths.append(self.file_paths[idx])
            
            if manhole_points == 0:
                all_non_manhole_paths.append(self.file_paths[idx])


        # add non manhole samples
        samples_to_take = min(len(all_non_manhole_paths), amount_non_manhole_samples)
        if samples_to_take > 0:
            non_manhole_paths_to_add = random.sample(all_non_manhole_paths, samples_to_take)
            new_file_paths.extend(non_manhole_paths_to_add)

        # info and update
        print(f"Reduced from {len(self.file_paths)} to {len(new_file_paths)} (filtered by manhole points -> min manhole points: {required_manhole_points}).")
        self.file_paths = new_file_paths
        
    def get_patch_via_identifier(self, pc_id, x_start, y_start, return_generator=False):
        file_path = self.file_paths_dict[f"{pc_id}_{x_start}_{y_start}"]

        if return_generator:
            return self.generator(file_paths=[file_path])
        else:
            return file_path

    def generator(self, file_paths=None):
        if file_paths is None:
            if self.file_paths is None:
                raise ValueError("No filenames set and no filenames given")
            else:
                file_paths = self.file_paths

        for cur_file_path in file_paths:
            tile, meta = load_single_bev_tile_as_pickle(cur_file_path)

            if self.has_labels:
                x = torch.from_numpy(tile[:-1]).float()
                y = torch.from_numpy(tile[-1]).long()
                assert x.ndim == 3
                assert y.ndim == 2
                yield {
                    "pixel_values": x,   # (C, H, W)
                    "labels": y,  # .reshape((bev.shape[1], bev.shape[2]))          # (H, W)
                    "meta": meta
                }
            else:
                x = torch.from_numpy(tile).float()
                yield {
                    "pixel_values": x,   # (C, H, W)
                    "labels": None,
                    "meta": meta
                }

    def extract_bev_path_from_full_path(self, path):
        _, bev_file_name = os.path.split(path)
        bev_file_name = ".".join(bev_file_name.split(".")[:-1]) + ".pkl"
        bev_file_name = bev_file_name.replace("preprocessed_patch", "preprocessed_patch_bev_")
        return bev_file_name

    def __len__(self):
        return len(self.file_paths)
    
    def extract_grid_identifier(self, input_string):
        _, input_string = os.path.split(input_string)
        input_string = ".".join(input_string.split(".")[:-1])
        input_string = input_string.replace("preprocessed_patch_bev_", "").replace("preprocessed_patch_", "")
        pc_id, x_start, y_start = input_string.split("_")[:3]
        return pc_id, x_start, y_start

def bev_gen_wrapper(tiles, metas, has_labels=False):
    for idx in range(len(tiles)):
        tile = tiles[idx]
        meta = metas[idx]

        if has_labels:
            x = torch.from_numpy(tile[:-1]).float()
            y = torch.from_numpy(tile[-1]).long()
            assert y.ndim == 2
            yield {
                "pixel_values": x,   # (C, H, W)
                "labels": y,  # .reshape((bev.shape[1], bev.shape[2]))          # (H, W)
                "meta": meta
            }
        else:
            x = torch.from_numpy(tile).float()
            yield {
                "pixel_values": x,   # (C, H, W)
                "labels": None,
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



def get_data_loader(data_name, path, type="train", transform=None,
                    batch_size=32, shuffle=True, num_workers=4,
                    preprocessed=False, return_train_format=False,
                    return_dataset=False):
    if data_name == "paris":
        data_loader = get_paris_data_loader(path, type=type, transform=transform,
                                            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                            preprocessed=preprocessed, return_train_format=return_train_format,
                                            return_dataset=return_dataset)
    elif data_name == "whu":
        data_loader = get_whu_data_loader(path, type=type, transform=transform,
                                          batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                          preprocessed=preprocessed, return_train_format=return_train_format,
                                          return_dataset=return_dataset)
    elif data_name == "sud":
        data_loader = get_sud_data_loader(path, type=type, transform=transform,
                                          batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                          preprocessed=preprocessed, return_train_format=return_train_format,
                                          return_dataset=return_dataset)
    else:
        raise ValueError(f"No Dataset with the name '{data_name}' founded. Try 'paris'.")

    return data_loader



def get_paris_data_loader(path, type="train", transform=None,
                          batch_size=32, shuffle=True, num_workers=4,
                          preprocessed=False, return_train_format=False,
                          return_dataset=False):
    dataset = ParisLille3DDataset(path=path, type=type, transform=transform,
                                  preprocessed=preprocessed, return_train_format=return_train_format)

    if return_dataset:
        return dataset

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_point_clouds)



def get_whu_data_loader(path, type="train", transform=None,
                          batch_size=32, shuffle=True, num_workers=4,
                          preprocessed=False, return_train_format=False,
                          return_dataset=False):
    dataset = WHUUrban3DDataset(path=path, type=type, transform=transform,
                                preprocessed=preprocessed, return_train_format=return_train_format)

    if return_dataset:
        return dataset

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_point_clouds)



def get_sud_data_loader(path, type="train", transform=None,
                          batch_size=32, shuffle=True, num_workers=4,
                          preprocessed=False, return_train_format=False,
                          return_dataset=False):
    dataset = SUDROADDataset(path=path, type=type, transform=transform,
                             preprocessed=preprocessed, return_train_format=return_train_format)

    if return_dataset:
        return dataset

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_point_clouds)



def preprocess_data(data_name, path, type="train", device="cpu",
                    bev_tile_size=15.0, bev_resolution=0.01, bev_overlap=0.5,
                    file_ending=".ply"):
    print("--- Data Preprocessing ---")

    print("setting type to 'all'")
    type = "all"

    data_loader = get_data_loader(data_name, path, type="load_unsplitted" if data_name=="sud"  else type, 
                                    transform=None if data_name=="sud"  else get_preprocessing_transform(grid_size=bev_resolution, do_voxelation=False),
                                    batch_size=1, shuffle=False, num_workers=0, preprocessed=False,
                                    return_train_format=False)
    
    # to_device = ToDevice(device)
    dataset = data_loader.dataset

    # reset sample folders
    sample_root_path = f"./bev_samples"
    for dir in os.listdir(sample_root_path):
        dir_path = os.path.join(sample_root_path, dir)
        if os.path.isdir(dir_path) and dir.startswith(f"{data_name}_"):
            shutil.rmtree(dir_path)

    # the sud dataset first have to get saved into 
    # multiple subclouds, without preprocessing
    if data_name == "sud":
        print("Splitting Dataset into subdatasets...")
        # first cleaning
        for cur_file in os.listdir(path):
            cur_file_path = os.path.join(path, cur_file)
            if os.path.isfile(cur_file_path) and cur_file != "SUD_road.las":
                os.remove(cur_file_path)

        point_cloud = next(iter(data_loader))[0]
        # split_point_cloud_into_multiple(point_cloud, path,  init_tile_size=50.0, overlap=3.0)
        for cur_idx, sub_cloud in point_cloud.items():
            save_point_cloud(path=os.path.join(path, f"{cur_idx}.h5"), 
                             point_cloud=sub_cloud)

        # load new splitted point clouds and apply preprocessing
        data_loader = get_data_loader(data_name, path, type="all", transform=get_preprocessing_transform(grid_size=bev_resolution, do_voxelation=False, manhole_label=3 if data_name=="sud"  else 104002),
                                      batch_size=1, shuffle=False, num_workers=0, preprocessed=False,
                                      return_train_format=False)

        # to_device = ToDevice(device)
        dataset = data_loader.dataset

    print("Start Preprocessing your dataset...")

    preprocessed_path_cleaned = False
    for idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc="Preprocessing"):
        # batch = to_device(batch)
        # print_pc(batch[0])

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

        pc_id = cur_file_name
        new_file_path = os.path.join(cur_root_path, "preprocessed_complete_"+cur_file_name +file_ending)

        if not isinstance(batch, list):  # Dataset
            raise TypeError(f"Batch is not a list, else it is '{type(batch)}'.")

        if len(batch) > 1:
            raise ValueError("Not expected bigger Batch.")
        
        # print(f"Data-Type: {type(batch)}")
        # print(f"Data-Shape: {batch.shape}") if hasattr(batch, "shape") else ""
        # print(f"Data-Type (batch 0): {type(batch[0])}")
        # print_pc(batch[0])

        # print(f"Semantic Label Shape: {batch[0].point[get_class_attribute(batch[0])].shape}")

        # # create preprocessed pc -> just to work on it, don't get directly used later on!!
        # save_point_cloud(path=new_file_path, point_cloud=batch[0])

        # save BEVs
        print("Generating BEV images and 3D Point Cloud Patches...")
        patch_info = bev_projection(batch[0], pc_id=pc_id,
                                    tile_size=bev_tile_size, resolution=bev_resolution, overlap=bev_overlap,
                                    include_class=True,
                                    direct_single_saving=True, single_saving_path=cur_root_path,
                                    save_3d_patches=True,
                                    sample_path=os.path.join(sample_root_path, f"{data_name}_{idx}"))
        # save_bev_tiles_as_pickle(tiles, meta, bev_file_path)
        # save_bev_tiles_as_pt(tiles, meta, bev_file_path)
        print(f"Saving BEVs to '{cur_root_path}'")

        # # create 3D Patches
        # print(f"Saving 3D Point Cloud as Patches to '{cur_root_path}'")
        # pc = batch[0]
        # for cur_pc_id, cur_x_start, cur_y_start, cur_tile_size in patch_info:
        #     # get positions for bounding box
        #     positions = pc.point[get_coordinate_attribute(pc)]
            
        #     # create bounding box data
        #     x_min, x_max = cur_x_start, cur_x_start + cur_tile_size
        #     y_min, y_max = cur_y_start, cur_y_start + cur_tile_size
            
        #     # create mask for the patch
        #     mask_x = (positions[:, 0] >= x_min) & (positions[:, 0] < x_max)
        #     mask_y = (positions[:, 1] >= y_min) & (positions[:, 1] < y_max)
        #     mask = mask_x & mask_y
        #     indices = mask.nonzero()[0]
            
        #     # filter/slice the pc via the created mask
        #     pc_patch = pc.select_by_index(indices)

        #     pc_patch_name = f"preprocessed_patch_{cur_pc_id}_{cur_x_start}_{cur_y_start}.h5"
        #     save_point_cloud(path=os.path.join(cur_root_path, pc_patch_name), 
        #                      point_cloud=pc_patch)

        # del tiles, meta, batch
        del batch
        gc.collect()

    print("\nCongratelations, your preprocessing is finish!")







