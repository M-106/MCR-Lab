# -----------
# > Imports <
# -----------
import numpy as np
import laspy
import open3d as o3d
import torch

from mcrlab.point_cloud.utils import get_coordinate_attribute, get_class_attribute, \
                                     get_intensity_attribute, get_color_attribute, \
                                     get_normal_attribute
from mcrlab.point_cloud.tensor_wrapper import PointCloudTensor



# ---------
# > Utils <
# ---------
def las_to_o3d(path):
    las = laspy.read(path)

    # print("LAS READING Point Cloud")
    # print("X:", las.x.min(), "bis", las.x.max())
    # print("Y:", las.y.min(), "bis", las.y.max())
    # print("Z:", las.z.min(), "bis", las.z.max())
    # print("Scale:", las.header.scales)
    # print("Offset:", las.header.offsets)

    index_attr_name = None
    candidates = ["user_data", "point_source_id", "cluster", "id", "index", "subcloud"]
    for cur_candidate in candidates:
        if hasattr(las, cur_candidate):
            index_attr_name = cur_candidate
            break

    if index_attr_name is None:
        extra_attr = list(las.point_format.extra_dimension_names)
        standard_attr = list(las.point_format.dimension_names)
        raise ValueError(
            "No Index for LAS file found.\n"
            f"Available Standard Attributes: {standard_attr} \n"
            f"Extra-Attribute (Extra Bytes): {extra_attr}"
        )

    points = np.vstack((las.x, las.y, las.z)).transpose()

    # add local offset
    offset = np.array([las.header.offsets])
    points_local = points - offset

    # extract sub-point cloud indices
    subcloud_indices = np.array(getattr(las, index_attr_name), dtype=np.int32)
    unique_indices = np.unique(subcloud_indices)

    # # Tensor-based loading
    # point_cloud = o3d.t.geometry.PointCloud()
    # point_cloud.point["positions"] = o3d.core.Tensor(points_local.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    
    labels = None
    for label_idx in ["classification", "classes", "class", "labels", "label"]:
        if hasattr(las, label_idx):
            labels = np.array(getattr(las, label_idx), dtype=np.int32)
            break
    
    colors = None
    if hasattr(las, "red"):
        colors = np.vstack((las.red, las.green, las.blue)).T / 65535.0

    intensities = None
    for intensity_idx in ["intensity", "intensities", "reflectance", "reflection", "reflections"]:
        if hasattr(las, intensity_idx):
            intensities = np.array(getattr(las, intensity_idx), dtype=np.float32)
            break

    normals = None
    for normal_idx in ["normals", "normal"]:
        if hasattr(las, normal_idx):
            normals = np.array(getattr(las, normal_idx), dtype=np.float32)
            break

    # collect every sub-point cloud
    subclouds_dict = dict()

    for cur_idx in unique_indices:
        mask = (subcloud_indices == cur_idx)

        sub_point_cloud = o3d.t.geometry.PointCloud()

        sub_points = points_local[mask]
        sub_point_cloud.point["positions"] = o3d.core.Tensor(sub_points.astype(np.float32), dtype=o3d.core.Dtype.Float32)

        if labels is not None:
            sub_point_cloud.point["labels"] = o3d.core.Tensor(labels[mask], dtype=o3d.core.Dtype.Int32)

        if colors is not None:
            sub_point_cloud.point["colors"] = o3d.core.Tensor(colors[mask], dtype=o3d.core.Dtype.Float32)

        if intensities is not None:
            sub_point_cloud.point["intensity"] = o3d.core.Tensor(intensities[mask], dtype=o3d.core.Dtype.Float32)

        if normals is not None:
            sub_point_cloud.point["normals"] = o3d.core.Tensor(normals[mask], dtype=o3d.core.Dtype.Float32)

        subclouds_dict[cur_idx] = sub_point_cloud

    return subclouds_dict



def save_as_las(path, point_cloud):
    header = laspy.LasHeader(point_format=3, version="1.2")
    # add little offset?
    # header.scales = np.array([0.001, 0.001, 0.001])   # 1 mm
    # header.offsets = np.array([x_min, y_min, z_min]
    export_las = laspy.LasData(header)

    if isinstance(point_cloud, PointCloudTensor):
        point_cloud = point_cloud.get_as_o3d()

    if isinstance(point_cloud, o3d.t.geometry.PointCloud):
        coordinate_idx = get_coordinate_attribute(point_cloud)
        export_las.x = np.asarray(point_cloud.point[coordinate_idx])[:, 0]
        export_las.y = np.asarray(point_cloud.point[coordinate_idx])[:, 1]
        export_las.z = np.asarray(point_cloud.point[coordinate_idx])[:, 2]

        label_idx = get_class_attribute(point_cloud)
        if label_idx:
            export_las.classification = np.asarray(point_cloud.point[label_idx], dtype=np.uint8)

        intensity_idx = get_intensity_attribute(point_cloud)
        if intensity_idx:
            # intensities = np.asarray(point_cloud.point[intensity_idx])
            # if intensities.ndim == 2 and intensities.shape[1] == 1:
            #     intensities = intensities[:, 0]
            # export_las.intensity = intensities.astype(np.uint16)

            intensities = np.asarray(point_cloud.point[intensity_idx]).reshape(-1)
            export_las.intensity = np.clip(intensities, 0, 65535).astype(np.uint16)

        color_idx = get_color_attribute(point_cloud)
        if color_idx:
            colors = np.asarray(point_cloud.point[color_idx])
            
            if np.issubdtype(colors.dtype, np.floating):
                colors16 = np.clip(colors, 0.0, 1.0) * 65535.0
                colors16 = colors16.astype(np.uint16)
            elif colors.max() <= 255:
                colors16 = (colors.astype(np.uint16) * 257)   # 255 -> 65535
            else:
                colors16 = colors.astype(np.uint16)

            export_las.red = colors16[:, 0].astype(np.uint16)
            export_las.green = colors16[:, 1].astype(np.uint16)
            export_las.blue = colors16[:, 2].astype(np.uint16)

        # not available in las
        # normal_idx = get_normal_attribute(point_cloud)
        # if normal_idx:
        #      normals = np.asarray(point_cloud.point[normal_idx])
        #      export_las.normals = normals
    else:
        raise ValueError(f"Exporting as .las/.laz can't handle point cloud type '{type(point_cloud)}'.")

    export_las.write(path)








