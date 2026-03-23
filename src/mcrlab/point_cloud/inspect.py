# -----------
# > Imports <
# -----------
import numpy as np
import open3d as o3d
import torch

# from mcrlab.point_cloud.io import get_device



# --------------------
# > Inspection Utils <
# --------------------
def get_info(point_cloud):
    print("\n--- Point Cloud Info ---")
    if isinstance(point_cloud, torch.Tensor):
        print("Type: Torch Tensor")
        print(f"Shape: {point_cloud.shape}")
        print(f"First 5 points:\n{point_cloud[:5]}")
    
    elif isinstance(point_cloud, o3d.t.geometry.PointCloud):
        print("Type: Tensor-based Open3D PointCloud")
        print(f"Properties: {point_cloud.point.primary_key}")
        print(f"Number of points: {len(point_cloud.point[point_cloud.point.primary_key])}")
    
    elif isinstance(point_cloud, o3d.geometry.PointCloud):
        print("Type: Legacy Open3D PointCloud")
        print(f"Number of points: {len(point_cloud.points)}")
    
    else:
        print("Unknown type:", type(point_cloud))



def get_metrics(point_cloud):
    print("\n--- Point Cloud Metrics ---")

    # Check for Tensor-based PointCloud
    if isinstance(point_cloud, torch.Tensor):
        print("Type: Torch Tensor")
        points = point_cloud[:, :3].numpy()
        num_points = points.shape[0]
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        has_colors = point_cloud.shape[1] >= 6
        has_normals = point_cloud.shape[1] >= 9
        labels = None  # assume labels not stored in tensor
    elif isinstance(point_cloud, o3d.t.geometry.PointCloud):
        print("Type: Tensor-based (o3d.t)")
        
        points = point_cloud.point["positions"].numpy()
        num_points = points.shape[0]
        
        min_bound = point_cloud.get_min_bound().numpy()
        max_bound = point_cloud.get_max_bound().numpy()
        
        has_colors = "colors" in point_cloud.point
        has_normals = "normals" in point_cloud.point
        
        labels = point_cloud.point["labels"].numpy() if "labels" in point_cloud.point else None

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
        
        # Legacy clouds don't have a .point attribute or built-in labels
        labels = None 

        print(point_cloud)
    
    else:
        print("Error: Provided object is not an Open3D PointCloud.")
        return

    print(f" - Total Points: {num_points}")
    print(f" - Bounding Box: Min {min_bound}, Max {max_bound}")
    print(f" - Has Colors:   {has_colors}")
    print(f" - Has Normals:  {has_normals}")

    if labels is not None:
        unique, counts = np.unique(labels, return_counts=True)
        print("\nClass Distribution:")
        for cls, count in zip(unique, counts):
            percentage = (count / num_points) * 100
            print(f"   Class {cls}: {count} points ({percentage:.2f}%)")
    else:
        print(" - Labels: No 'labels' attribute found.")
    
    print("---------------------------\n")



def visualize(point_cloud):
    if isinstance(point_cloud, torch.Tensor):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3].numpy())
        o3d.visualization.draw_geometries([pcd])
    
    elif isinstance(point_cloud, o3d.t.geometry.PointCloud):
        o3d.visualization.draw([point_cloud])  # new tensor-based API
    
    elif isinstance(point_cloud, o3d.geometry.PointCloud):
        o3d.visualization.draw_geometries([point_cloud])
    
    else:
        print("Cannot visualize type:", type(point_cloud))







