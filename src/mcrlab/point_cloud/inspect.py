# -----------
# > Imports <
# -----------
import numpy as np
import open3d as o3d
import torch

# from mcrlab.point_cloud.io import get_device
from mcrlab.point_cloud.utils import set_color
from mcrlab.point_cloud.utils import get_coordinate_attribute, get_class_attribute, \
                                     get_intensity_attribute, get_color_attribute, \
                                     get_normal_attribute
from mcrlab.point_cloud.core import PointCloudTensor



# --------------------
# > Inspection Utils <
# --------------------
def print_pc(point_cloud):
    print("\n---------------------------")
    print_info(point_cloud)
    print_metrics(point_cloud)
    print("\n---------------------------\n")



def print_info(point_cloud):
    print("\n--- Point Cloud Info ---")
    if isinstance(point_cloud, torch.Tensor):
        raise ValueError("Torch Tensor should be replaced by PointCloudTensor")
        print("Type: Torch Tensor")
        print(f"Shape: {point_cloud.shape}")
        print(f"First 5 points:\n{point_cloud[:5]}")

    elif isinstance(point_cloud, PointCloudTensor):
        print("Type: PointCloudTensor")
        print(f"Shape: {point_cloud.coordinates.shape}")
        print(f"First 5 points:\n{point_cloud.coordinates[:5]}")
    
    elif isinstance(point_cloud, o3d.t.geometry.PointCloud):
        print("Type: Tensor-based Open3D PointCloud")
        print(f"Primary Key: {point_cloud.point.primary_key}")
        print(f"Number of points: {len(point_cloud.point[get_coordinate_attribute(point_cloud)])}")
    
    elif isinstance(point_cloud, o3d.geometry.PointCloud):
        print("Type: Legacy Open3D PointCloud")
        print(f"Number of points: {len(point_cloud.points)}")
    
    else:
        print("Unknown type:", type(point_cloud))



def print_metrics(point_cloud):
    print("\n--- Point Cloud Metrics ---")

    # Check for Tensor-based PointCloud
    # FIXME -> change from torch.Tensor to your own Tensor
    if isinstance(point_cloud, torch.Tensor):
        raise ValueError("Torch Tensor should be replaced by PointCloudTensor")
        print("Type: Torch Tensor")
        # print(f" - Shape: {point_cloud.shape}")
        points = point_cloud[:, :3].numpy()
        num_points = points.shape[0]
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        has_colors = "unknown"
        has_intensity = "unknown"
        has_normals = "unknown"
        labels = None

    elif isinstance(point_cloud, PointCloudTensor):
        print("Type: PointCloudTensor")
        # print(f" - Shape: {point_cloud.coordinates.shape}")
        points = point_cloud.coordinates[:, :3].numpy()
        num_points = points.shape[0]
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        has_colors = point_cloud.colors is not None
        has_intensity = point_cloud.intensities is not None
        has_normals = point_cloud.normals is not None
        labels = point_cloud.labels if point_cloud.labels is not None else None

    elif isinstance(point_cloud, o3d.t.geometry.PointCloud):
        print("Type: Tensor-based (o3d.t)")
        
        points = point_cloud.point[get_coordinate_attribute(point_cloud)].numpy()
        num_points = points.shape[0]
        
        min_bound = point_cloud.get_min_bound().numpy()
        max_bound = point_cloud.get_max_bound().numpy()
        
        has_colors = get_color_attribute(point_cloud) is not None
        has_normals = get_normal_attribute(point_cloud) is not None
        has_intensity = get_intensity_attribute(point_cloud) is not None
        
        label_idx = get_class_attribute(point_cloud)
        
        if label_idx:
            labels = point_cloud.point[label_idx].numpy()
        else:
            labels = None

        print(point_cloud)

    # Check for Legacy PointCloud
    elif isinstance(point_cloud, o3d.geometry.PointCloud):
        print("Type: Legacy (o3d.geometry)")
        
        points = np.asarray(point_cloud.points)
        num_points = points.shape[0]
        
        min_bound = point_cloud.get_min_bound()
        max_bound = point_cloud.get_max_bound()
        
        has_colors = point_cloud.has_colors()
        has_normals = point_cloud.has_normals()
        has_intensity = point_cloud.has_intensity()
        
        # Legacy clouds don't have a .point attribute or built-in labels
        labels = None 

        print(point_cloud)
    
    else:
        print("Error: Provided object is not an Open3D PointCloud.")
        return

    print(f" - Total Points: {num_points}")
    print(f" - Bounding Box: Min {min_bound}, Max {max_bound}")
    print(f" - Has Colors:   {str(has_colors):6}")  # ^6, <6, 6, .>6
    print(f" - Has Normals:  {str(has_normals):6}")
    print(f" - Has Intensity:{str(has_intensity):6}")
    print(f" - Has Classes:  {str(labels is not None):6}")

    if labels is not None:
        class_distribution = []
        unique, counts = np.unique(labels, return_counts=True)
        print("\nClass Distribution:")
        for cls, count in zip(unique, counts):
            percentage = (count / num_points) * 100
            class_distribution.append((cls, count, percentage))
        # sort and print sorted
        class_distribution.sort(key=lambda x: x[2], reverse=True)
        for cls, count, percentage in class_distribution:
            print(f"   Class {cls:<4}: {count:8} points ({percentage:5.2f}%)")
    else:
        print(" - Labels: No 'labels' attribute found.")



def visualize(point_cloud, color_mode=None):
    """
    Visualizes a point cloud via Open3D.

    Available color_modes: class, height, intensity
    """

    if isinstance(point_cloud, torch.Tensor):
        raise ValueError("Torch Tensor should be replaced by PointCloudTensor")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3].numpy())
        o3d.visualization.draw_geometries([pcd])

    elif isinstance(point_cloud, PointCloudTensor):
        pcd = point_cloud.get_as_o3d()
        if color_mode:
            pcd = set_color(pcd, color_mode)
        o3d.visualization.draw([pcd])
    elif isinstance(point_cloud, o3d.t.geometry.PointCloud):
        if color_mode:
            point_cloud = set_color(point_cloud, color_mode)
        o3d.visualization.draw([point_cloud])  # new tensor-based API
    
    elif isinstance(point_cloud, o3d.geometry.PointCloud):
        o3d.visualization.draw_geometries([point_cloud])
    
    else:
        print("Cannot visualize type:", type(point_cloud))







