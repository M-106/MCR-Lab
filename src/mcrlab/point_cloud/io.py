# -----------
# > Imports <
# -----------
import os

import numpy as np

import open3d as o3d
import open3d.core as ocore

from mcrlab.point_cloud.las_utils import las_to_o3d, save_as_las
from mcrlab.point_cloud.utils import get_intensity_attribute
from mcrlab.point_cloud.core import PointCloudTensor
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
            del point_cloud.point[intensity_idx]

        # if as_point_cloud_tensor:
        #     point_cloud = ToPointCloudTensorTransform()(point_cloud)
    
        return point_cloud
    else:
        _, file_ = os.path.split(path)
        raise ValueError(f"Can't load '{file_}' as point-cloud.")



def save_point_cloud(path, point_cloud):
    if isinstance(point_cloud, PointCloudTensor):
        point_cloud = point_cloud.get_as_o3d()

    if path.endswith(".las") or path.endswith(".laz"):
        save_as_las(path=path, point_cloud=point_cloud)
    elif path.endswith(".ply"):
        o3d.t.io.write_point_cloud(path=path, point_cloud=point_cloud)
    else:
        _, file_ = os.path.split(path)
        raise ValueError(f"Can't save '{file_}' as point-cloud.") 














