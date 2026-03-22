# -----------
# > Imports <
# -----------
import numpy as np
import open3d as o3d

from point_cloud_io import get_device



# --------------------
# > Inspection Utils <
# --------------------
def point_cloud_info(point_cloud):
    print(point_cloud)



def point_cloud_metrics(pcd):
    print("\n--- Point Cloud Metrics ---")

    # Check for Tensor-based PointCloud
    if isinstance(pcd, o3d.t.geometry.PointCloud):
        print("Type: Tensor-based (o3d.t)")
        
        points = pcd.point["positions"].numpy()
        num_points = points.shape[0]
        
        min_bound = pcd.get_min_bound().numpy()
        max_bound = pcd.get_max_bound().numpy()
        
        has_colors = "colors" in pcd.point
        has_normals = "normals" in pcd.point
        
        labels = pcd.point["labels"].numpy() if "labels" in pcd.point else None

    # Check for Legacy PointCloud
    elif isinstance(pcd, o3d.geometry.PointCloud):
        print("Type: Legacy (o3d.geometry)")
        
        points = np.asarray(pcd.points)
        num_points = points.shape[0]
        
        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        
        has_colors = pcd.has_colors()
        has_normals = pcd.has_normals()
        
        # Legacy clouds don't have a .point attribute or built-in labels
        labels = None 
    
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





def visualize_point_cloud(point_cloud):
    o3d.visualization.draw_geometries([point_cloud])







