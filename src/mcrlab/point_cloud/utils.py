import numpy as np
import open3d as o3d



def add_color_from_intensity(point_cloud):
    refl = point_cloud.point["reflectance"].numpy().squeeze()

    # normalize to [0,1]
    refl = (refl - refl.min()) / (refl.max() - refl.min() + 1e-8)

    # grayscale mapping
    colors = np.stack([refl, refl, refl], axis=1)  # (N,3)

    point_cloud.point["colors"] = o3d.core.Tensor(colors, dtype=o3d.core.Dtype.Float32)
    return point_cloud



def add_color_from_height(point_cloud):
    points = point_cloud.point["positions"].numpy()
    z = points[:, 2]

    z = (z - z.min()) / (z.max() - z.min() + 1e-8)

    # blue -> red gradient
    colors = np.stack([z, 0*z, 1-z], axis=1)

    point_cloud.point["colors"] = o3d.core.Tensor(colors, dtype=o3d.core.Dtype.Float32)
    return point_cloud



# def filter_ground(point_cloud, distance_threshold=0.2, ransac_n=3, num_iterations=1000):
#     # convert to legacy (needed for RANSAC)
#     pc_legacy = point_cloud.to_legacy()

#     plane_model, inliers = pc_legacy.segment_plane(
#         distance_threshold=distance_threshold,
#         ransac_n=ransac_n,
#         num_iterations=num_iterations
#     )

#     # KEEP ground (inliers)
#     pc_ground = pc_legacy.select_by_index(inliers)

#     # convert back to tensor
#     pc_tensor = o3d.t.geometry.PointCloud.from_legacy(pc_ground)

    # for key in ["positions", "colors", "reflectance", "labels"]:
    #         try:
    #             data = point_cloud.point[key].numpy()
    #             pc_filtered.point[key] = o3d.core.Tensor(data[mask])
    #         except KeyError:
    #             pass

#     return pc_tensor

def filter_ground_with_RANSAC(point_cloud, distance_threshold=0.2, ransac_n=3, num_iterations=1000):
    # get points
    points = point_cloud.point["positions"].numpy()

    # legacy for plane fitting
    pc_legacy = o3d.geometry.PointCloud()
    pc_legacy.points = o3d.utility.Vector3dVector(points)

    plane_model, inliers = pc_legacy.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    mask = np.zeros(len(points), dtype=bool)
    mask[inliers] = True  # ground = True

    # build new tensor point cloud with ALL attributes
    pc_ground = o3d.t.geometry.PointCloud()

    for key in point_cloud.point:
        data = point_cloud.point[key].numpy()
        pc_ground.point[key] = o3d.core.Tensor(data[mask])

    return pc_ground



def filter_ground_with_height(point_cloud, threshold=0.3):
    points = point_cloud.point["positions"].numpy()

    # naive approximation: subtract global minimum
    z = points[:, 2]
    z_norm = z - z.min()

    mask = z_norm < threshold
    filtered_points = points[mask]

    pc_filtered = o3d.t.geometry.PointCloud()

    for key in ["positions", "colors", "reflectance", "labels"]:
        try:
            data = point_cloud.point[key].numpy()
            pc_filtered.point[key] = o3d.core.Tensor(data[mask])
        except KeyError:
            pass

    return pc_filtered



def bev_projection(point_cloud):
    # ... -> tile based bev images from pc
    return point_cloud



def bev_back_projection(image_number, pixel):
    # ... -> back-projection from from one specific pixel tile based bev images from pc
    return point_cloud


