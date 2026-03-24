# -----------
# > Imports <
# -----------
import os

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import laspy
import open3d as o3d
import open3d.core as ocore

from mcrlab.point_cloud.utils import add_color_from_intensity, add_color_from_height, \
                                     filter_ground_with_height, filter_ground_with_RANSAC



# ---------
# > Utils <
# ---------
def get_device(device):
    if device:
        return device
    else:
        return ocore.Device("CPU:0")



def load_point_cloud(path):
 
    if path.endswith(".las") or path.endswith(".laz"):
        las = laspy.read(path)
        points = np.vstack((las.x, las.y, las.z)).transpose()

        # Tensor-based loading
        point_cloud = o3d.t.geometry.PointCloud()
        point_cloud.point["positions"] = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
        
        if hasattr(las, "classification"):
            labels = np.array(las.classification, dtype=np.int32)
            point_cloud.point["labels"] = o3d.core.Tensor(labels, dtype=o3d.core.Dtype.Int32)
        
        if hasattr(las, "red"):
            colors = np.vstack((las.red, las.green, las.blue)).T / 65535.0
            point_cloud.point["colors"] = o3d.core.Tensor(colors, dtype=o3d.core.Dtype.Float32)

        if hasattr(las, "intensity"):
            intensities = np.array(las.intensity, dtype=(np.float32))
            point_cloud.point["intensity"] = o3d.core.Tensor(intensities, dtype=o3d.core.Dtype.Float32)

        return point_cloud
    elif path.endswith(".ply"):
            # point_cloud = o3d.io.read_triangle_mesh(path)
            # point_cloud.compute_vertex_normals()
            # point_cloud = o3d.t.geometry.PointCloud.from_legacy(point_cloud)
    
        try:
            point_cloud = o3d.t.io.read_point_cloud(path)
        except TypeError:
            point_cloud = o3d.io.read_point_cloud(path)
            point_cloud = o3d.t.geometry.PointCloud.from_legacy(point_cloud)

        if "reflectance" in point_cloud.point:
            point_cloud = add_color_from_intensity(point_cloud)
            # add_color_from_height
    
        return point_cloud
    else:
        _, file_ = os.path.split(path)
        raise ValueError(f"Can't load '{file_}' as point-cloud.")


def save_point_cloud(path, point_cloud):
    if path.endswith(".las") or path.endswith(".laz"):
        header = laspy.LasHeader(point_format=3, version="1.2")
        export_las = laspy.LasData(header)

        export_las.x = np.asarray(point_cloud.point["positions"])[:, 0]
        export_las.y = np.asarray(point_cloud.point["positions"])[:, 1]
        export_las.z = np.asarray(point_cloud.point["positions"])[:, 2]

        if "colors" in point_cloud.point:
            colors = np.asarray(point_cloud.point["colors"]) * 65535
            export_las.red = colors[:, 0].astype(np.uint16)
            export_las.green = colors[:, 1].astype(np.uint16)
            export_las.blue = colors[:, 2].astype(np.uint16)

        if "labels" in point_cloud.point:
            export_las.classification = np.asarray(point_cloud.point["labels"], dtype=np.uint8)

        # FIXME -> and other informations?!

        export_las.write(path)
    elif path.endswith(".ply"):
        o3d.t.io.write_point_cloud(path, point_cloud)
    else:
        _, file_ = os.path.split(path)
        raise ValueError(f"Can't save '{file_}' as point-cloud.") 



class ToTensorTransform:
    def __init__(self, num_points=16384):
        self.num_points = num_points

    def __call__(self, point_cloud):
        # convert to numpy
        points = point_cloud.point["positions"].numpy()

        # process additional information
        data = points
        if "colors" in point_cloud.point:
            colors = point_cloud.point["colors"].numpy()
            data = np.hstack([data, colors])
        if "normals" in point_cloud.point:
            normals = point_cloud.point["normals"].numpy()
            data = np.hstack([data, normals])

        # sample or pad points to fixed size
        cur_number_of_points = data.shape[0]
        if cur_number_of_points > self.num_points:
            # idxs = np.random.choice(cur_number_of_points, self.num_points, replace=False)
            
            z = data[:, 2]
            idx_sorted = np.argsort(z)     # sorted low to high

            # remove top N
            N_remove = 500
            idxs = idx_sorted[:-N_remove]
        else:
            extra_idxs = np.random.choice(cur_number_of_points, (self.num_points-cur_number_of_points), replace=True)
            idxs = np.hstack((np.arange(cur_number_of_points), extra_idxs))  # result also 1 dimensional

        np.random.shuffle(idxs)
        data = data[idxs]

        return torch.from_numpy(data).float()
    

class ExtractLabelsAsTensorTransform:
    def __init__(self):
        pass

    def __call__(self, point_cloud):
        # FIXME -> implement me -> first have to know how labels are saved
        return point_cloud


class RoadExtractionTransform:
    def __init__(self, mode="height"):  # height, ransac
        self.mode = mode

    def __call__(self, point_cloud):
        point_before = len(point_cloud.point["positions"])

        if self.mode == "height":
            point_cloud = filter_ground_with_height(point_cloud, threshold=10.0)
        elif self.mode == "ransac":
            point_cloud = filter_ground_with_RANSAC(point_cloud, distance_threshold=1.0, ransac_n=3, num_iterations=3000)
        else:
            raise ValueError(f"No `{self.mode}` mode for Road Extraction.")
        point_after = len(point_cloud.point["positions"])
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

        return point_cloud  # shape (1024, 3)



def collate_point_clouds(batch):
    """
    Just returns the point-clouds as a list,
    not a stacked tensor.
    """
    return batch



def get_paris_data_loader(path, testdata=False, transform=None,
                          batch_size=32, shuffle=True, num_workers=4):
    dataset = ParisLille3DDataset(path=path, testdata=testdata, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_point_clouds)



def get_basic_transform(num_points=16384):
    transform = Compose([
        # RandomRotate(axis='z'),
        # Jitter(sigma=0.01, clip=0.05),
        RoadExtractionTransform(),
        ToTensorTransform(num_points=num_points)
    ])
    return transform










