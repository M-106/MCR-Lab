# -----------
# > Imports <
# -----------
import os
from pathlib import Path

import numpy as np
from PIL import Image
import h5py

import open3d as o3d
import open3d.core as ocore

from mcrlab.point_cloud.las_utils import las_to_o3d, save_as_las
from mcrlab.point_cloud.utils import get_intensity_attribute, set_color
from mcrlab.point_cloud.tensor_wrapper import PointCloudTensor
from mcrlab.point_cloud.inspect import print_pc
# from mcrlab.point_cloud.data import ToPointCloudTensorTransform



# ---------
# > Utils <
# ---------
def get_device(device):
    if device is not None:
        return device
    else:
        return ocore.Device("CPU:0")



def load_point_cloud(path):
    if path.endswith(".las") or path.endswith(".laz"):
        point_cloud = las_to_o3d(path)
        # if as_point_cloud_tensor:
        #     point_cloud = ToPointCloudTensorTransform()(point_cloud)
        return point_cloud
    elif path.endswith(".ply"):
            # point_cloud = o3d.io.read_triangle_mesh(path)
            # point_cloud.compute_vertex_normals()
            # point_cloud = o3d.t.geometry.PointCloud.from_legacy(point_cloud)
    
        point_cloud = o3d.t.io.read_point_cloud(path)

        # except TypeError:
        #     point_cloud = o3d.io.read_point_cloud(path)
        #     point_cloud = o3d.t.geometry.PointCloud.from_legacy(point_cloud)

        intensity_idx = get_intensity_attribute(point_cloud)
        if intensity_idx:
            point_cloud.point["intensity"] = point_cloud.point[intensity_idx]
            if "intensity" != intensity_idx:
                del point_cloud.point[intensity_idx]

        # if as_point_cloud_tensor:
        #     point_cloud = ToPointCloudTensorTransform()(point_cloud)
    
        return point_cloud
    elif path.endswith(".h5"):
        with h5py.File(path, "r") as file_:
            # view data
            # file_.visititems(lambda name, obj: print(name, obj))
            # results:
            # coords <HDF5 dataset "coords": shape (7250451, 3), type "<f8">
            # instances <HDF5 dataset "instances": shape (7250451,), type "<i8">
            # intensity <HDF5 dataset "intensity": shape (7250451,), type "<f8">
            # number_returns <HDF5 dataset "number_returns": shape (7250451,), type "<f8">
            # semantics <HDF5 dataset "semantics": shape (7250451,), type "<i8">
            # for key in file_.keys():
            #     print(f"Key: {key}")
            #     print(f"    - Shape: {file_[key].shape}")
            #     print(f"    - Dtype: {file_[key].dtype}")

            # extract data
            coordinates = file_["coords"][:]
            instances = file_["instances"][:]
            intensities = file_["intensity"][:]
            semantics = file_["semantics"][:]
            number_returns = file_["number_returns"][:]

        if intensities.max() <= 1.0:
            intensities *= 255

        # create open3d version
        point_cloud = o3d.t.geometry.PointCloud()
        # use key positions and not coordinates!
        point_cloud.point["positions"] = o3d.core.Tensor(coordinates, dtype=o3d.core.Dtype.Float32)
        # point_cloud.point["classes"] = o3d.core.Tensor(np.hstack([instances.reshape(-1, 1), 
        #                                                           semantics.reshape(-1, 1)], axis=0), 
        #                                                dtype=o3d.core.Dtype.Int32)
        # point_cloud.point["classes"] = o3d.core.Tensor(semantics.reshape(-1, 1), dtype=o3d.core.Dtype.Int32)
        point_cloud.point["classes"] = o3d.core.Tensor(instances.reshape(-1, 1), dtype=o3d.core.Dtype.Int32)
        point_cloud.point["intensities"] = o3d.core.Tensor(intensities.reshape(-1, 1), dtype=o3d.core.Dtype.Float32)

        return point_cloud
    else:
        _, file_ = os.path.split(path)
        raise ValueError(f"Can't load '{file_}' as point-cloud.")



def save_point_cloud(path, point_cloud):
    if isinstance(point_cloud, PointCloudTensor):
        point_cloud = point_cloud.get_as_o3d()

    if not isinstance(point_cloud, o3d.t.geometry.PointCloud):
        raise ValueError(f"Expected 'o3d.t.geometry.PointCloud' but got {type(point_cloud)}")

    if path.endswith(".las") or path.endswith(".laz"):
        save_as_las(path=path, point_cloud=point_cloud)
    elif path.endswith(".ply"):
        # print_pc(point_cloud)
        o3d.t.io.write_point_cloud(filename=path, pointcloud=point_cloud)  # compressed=True

        # loaded_pcd = o3d.t.io.read_point_cloud(path)
        # print_pc(loaded_pcd)

        # # Check if the intensity attribute exists and print its shape
        # if 'intensity' in loaded_pcd.point:
        #     print(f"Successfully loaded point cloud with {loaded_pcd.point['intensity'].shape} intensity values.")
        # else:
        #     print("Loading failed: Intensity attribute not found.")
    else:
        _, file_ = os.path.split(path)
        raise ValueError(f"Can't save '{file_}' as point-cloud.") 
















