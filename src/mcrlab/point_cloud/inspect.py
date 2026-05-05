# -----------
# > Imports <
# -----------
import numpy as np
import open3d as o3d
import torch
import matplotlib.pyplot as plt

# from mcrlab.point_cloud.io import get_device
from mcrlab.point_cloud.utils import set_color
from mcrlab.point_cloud.utils import get_coordinate_attribute, get_class_attribute, \
                                     get_intensity_attribute, get_color_attribute, \
                                     get_normal_attribute, get_instance_attribute
from mcrlab.point_cloud.tensor_wrapper import PointCloudTensor



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
        points_type = point_cloud.coordinates.dtype
        points_shape = point_cloud.coordinates.shape
        num_points = points.shape[0]

        # min_bound = points.min(axis=0)
        # max_bound = points.max(axis=0)

        has_colors = point_cloud.colors is not None
        colors_type = point_cloud.colors.dtype if point_cloud.colors is not None else None
        colors_shape = point_cloud.colors.shape if point_cloud.colors is not None else None

        has_intensity = point_cloud.intensities is not None
        intensity_type = point_cloud.intensities.dtype if point_cloud.intensities is not None else None
        intensity_shape = point_cloud.intensities.shape if point_cloud.intensities is not None else None

        has_normals = point_cloud.normals is not None
        normals_type = point_cloud.normals.dtype if point_cloud.normals is not None else None
        normals_shape = point_cloud.normals.shape if point_cloud.normals is not None else None

        labels = point_cloud.labels if point_cloud.labels is not None else None
        labels_type = point_cloud.labels.dtype if point_cloud.labels is not None else None
        labels_shape = point_cloud.labels.shape if point_cloud.labels is not None else None

        instances = point_cloud.instances if point_cloud.instances is not None else None
        instances_type = point_cloud.instances.dtype if point_cloud.instances is not None else None
        instances_shape = point_cloud.instances.shape if point_cloud.instances is not None else None

    #   or isinstance(point_cloud, o3d.cpu.pybind.t.geometry.PointCloud)
    elif isinstance(point_cloud, o3d.t.geometry.PointCloud):
        print("Type: Tensor-based (o3d.t)")
        
        points_idx = get_coordinate_attribute(point_cloud)
        points = point_cloud.point[points_idx].numpy()
        points_type = point_cloud.point[points_idx].dtype
        points_shape = point_cloud.point[points_idx].shape
        num_points = points.shape[0]
        
        # min_bound = point_cloud.get_min_bound().numpy()
        # max_bound = point_cloud.get_max_bound().numpy()
        
        colors_idx = get_color_attribute(point_cloud)
        has_colors = colors_idx is not None
        colors_type = point_cloud.point[colors_idx].dtype if has_colors else None
        colors_shape = point_cloud.point[colors_idx].shape if has_colors else None

        normals_idx = get_normal_attribute(point_cloud)
        has_normals = normals_idx is not None
        normals_type = point_cloud.point[normals_idx].dtype if has_normals else None
        normals_shape = point_cloud.point[normals_idx].shape if has_normals else None

        intensity_idx = get_intensity_attribute(point_cloud)
        has_intensity = intensity_idx is not None
        intensity_type = point_cloud.point[intensity_idx].dtype if has_intensity else None
        intensity_shape = point_cloud.point[intensity_idx].shape if has_intensity else None
        
        label_idx = get_class_attribute(point_cloud)
        
        if label_idx:
            labels = point_cloud.point[label_idx].numpy()
            labels_type = point_cloud.point[label_idx].dtype
            labels_shape = point_cloud.point[label_idx].shape
        else:
            labels = None
            labels_type = None
            labels_shape = None

        instance_idx = get_instance_attribute(point_cloud)
        if instance_idx:
            instances = point_cloud.point[instance_idx].numpy()
            instances_type = point_cloud.point[instance_idx].dtype
            instances_shape = point_cloud.point[instance_idx].shape
        else:
            instances = None
            instances_type = None
            instances_shape = None

        print(point_cloud)

    # # Check for Legacy PointCloud
    # elif isinstance(point_cloud, o3d.geometry.PointCloud):
    #     print("Type: Legacy (o3d.geometry)")
        
    #     points = np.asarray(point_cloud.points)
    #     num_points = points.shape[0]
        
    #     min_bound = point_cloud.get_min_bound()
    #     max_bound = point_cloud.get_max_bound()
        
    #     has_colors = point_cloud.has_colors()
    #     has_normals = point_cloud.has_normals()
    #     has_intensity = point_cloud.has_intensity()
        
    #     # Legacy clouds don't have a .point attribute or built-in labels
    #     labels = None 

    #     print(point_cloud)
    
    else:
        print("Error: Provided object is not an Open3D PointCloud.")
        return

    # Points
    points_str = f"{num_points:,}".replace(',', '.')
    print(f" ◉ Points: {points_str}")
    if points_type is not None:
        print(f"       ⨀ type '{points_type}'")
    if points_shape is not None:
        print(f"       ⨀ shape '{points_shape}'")
    # print(f" - Bounding Box: Min {min_bound}, Max {max_bound}")

    # Colors
    print(f" ◉ Colors:   {str(has_colors):6}")  # ^6, <6, 6, .>6
    if colors_type is not None:
        print(f"       ⨀ type '{colors_type}'")
    if colors_shape is not None:
        print(f"       ⨀ shape '{colors_shape}'")

    # Normals
    print(f" ◉ Normals:  {str(has_normals):6}")
    if normals_type is not None:
        print(f"       ⨀ type '{normals_type}'")
    if normals_shape is not None:
        print(f"       ⨀ shape '{normals_shape}'")

    # Intensity
    print(f" ◉ Intensity:{str(has_intensity):6}")
    if intensity_type is not None:
        print(f"       ⨀ type '{intensity_type}'")
    if intensity_shape is not None:
        print(f"       ⨀ shape '{intensity_shape}'")
    
    # Classes
    print(f" ◉ Classes:  {str(labels is not None):6}")
    if labels_type is not None:
        print(f"       ⨀ type '{labels_type}'")
    if labels_shape is not None:
        print(f"       ⨀ shape '{labels_shape}'")

    if labels is not None:
        class_distribution = []
        unique, counts = np.unique(labels, return_counts=True)
        print("       ⨀ class distribution:")
        for cls, count in zip(unique, counts):
            percentage = (count / num_points) * 100
            class_distribution.append((cls, count, percentage))
        # sort and print sorted
        class_distribution.sort(key=lambda x: x[2], reverse=True)

        max_counter = 0
        for cls, count, percentage in class_distribution:
            if max_counter <= 5:
                print(f"           Class {cls:<4}: {count:8} points ({percentage:5.2f}%)")
                max_counter += 1
            else:
                print(f"           ... ({int(len(unique)-max_counter)} other classes)")
                
                break

    # Instance Labels
    print(f" ◉ Instances (labels):  {str(instances is not None):6}")
    if instances_type is not None:
        print(f"       ⨀ type '{instances_type}'")
    if instances_shape is not None:
        print(f"       ⨀ shape '{instances_shape}'")

    if instances is not None:
        class_distribution = []
        unique, counts = np.unique(instances, return_counts=True)
        print("       ⨀ instance distribution:")
        for cls, count in zip(unique, counts):
            percentage = (count / num_points) * 100
            class_distribution.append((cls, count, percentage))
        # sort and print sorted
        class_distribution.sort(key=lambda x: x[2], reverse=True)

        max_counter = 0
        for cls, count, percentage in class_distribution:
            if max_counter <= 5:
                print(f"           Instance {cls:<4}: {count:8} points ({percentage:5.2f}%)")
                max_counter += 1
            else:
                print(f"           ... ({int(len(unique)-max_counter)} other instances)")
                
                break
    
    # BEV Images
    if isinstance(point_cloud, PointCloudTensor):
        if point_cloud.bev_data is not None:
            # print(f" ◉ BEV Images:  {str(point_cloud.bev_amount):6}")
            # print(f"       ⨀ type '{str(type(point_cloud.bevs))}'")
            # if len(point_cloud.bevs) > 0:
            #     print(f"       ⨀ element-type '{str(type(point_cloud.bevs[0]))}'")
            #     print(f"            → '{str(point_cloud.bevs[0].dtype)}'") if hasattr(point_cloud.bevs[0], "dtype") else ""
            #     print(f"       ⨀ shape '{str(point_cloud.bevs[0].shape)}'") if hasattr(point_cloud.bevs[0], "shape") else ""
            print(f" ◉ BEV Images:  {str(point_cloud.bev_amount):6}")
            print(f"       ⨀ type '{str(point_cloud.bev_img_type)}'")
            print(f"       ⨀ data-type '{str(point_cloud.bev_img_dtype)}'")
            print(f"       ⨀ shape '{str(point_cloud.bev_img_shape)}'")
            print(f" ◉ BEV Labels:")
            print(f"       ⨀ type '{str(point_cloud.bev_labels_type)}'")
            print(f"       ⨀ data-type '{str(point_cloud.bev_labels_dtype)}'")
            print(f"       ⨀ shape '{str(point_cloud.bev_labels_shape)}'")




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



def visualize_intensity_in_2d(points, color):
    if not isinstance(points, np.ndarray):
        raise ValueError(f"Points must be a numpy array, but got: {type(points)}")
    
    if not isinstance(color, np.ndarray):
        raise ValueError(f"Color must be a numpy array, but got: {type(color)}")

    # matplotlib settings
    # plt.style.use("seaborn-v0_8-whitegrid")
    plt.style.use("default")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    # extract data
    x = points[:, 0]
    y = points[:, 1]

    # prepare color
    # if color.max() > 1.0:
    #     color /= 255
    color = (color - np.min(color)) / (np.max(color) - np.min(color))
    color = np.repeat(color[:, np.newaxis], 3, axis=1).squeeze()
    # print(f"Color Shape: {color.shape}")

    # plotting
    plt.figure(figsize=(7, 7))

    # points
    plt.scatter(x, y, s=15, c=color, alpha=1.0)  # , label="Manhole Points")

    # annotation
    plt.title("Point Cloud Intensity (Orthogonal View)", fontsize=14, weight="bold", y=0.98)

    plt.grid(alpha=0.3)
    plt.axis("equal")
    plt.margins(0.1)

    plt.show()




