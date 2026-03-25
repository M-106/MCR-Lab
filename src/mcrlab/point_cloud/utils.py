# -----------
# > Imports <
# -----------
import numpy as np
import open3d as o3d



# ---------------------
# > Get Attribute Key <
# ---------------------
def get_coordinate_attribute(point_cloud):
    if "positions" in point_cloud.point:
        coordinate_idx = "positions"
    elif "position" in point_cloud.point:
        coordinate_idx = "position"
    elif "coordinate" in point_cloud.point:
        coordinate_idx = "coordinate"
    elif "pos" in point_cloud.point:
        coordinate_idx = "pos"
    else:
        coordinate_idx = None
        raise ValueError(f"No coordinate attribute found, \
                         please add your key in the `get_coordinate_attribute` function.\
                         \nPoint-Cloud Info:\n{point_cloud}")

    return coordinate_idx



def get_class_attribute(point_cloud):
    if "class" in point_cloud.point:
        label_idx = "class"
    elif "classes" in point_cloud.point:
        label_idx = "classes"
    elif "classification" in point_cloud.point:
        label_idx = "classification"
    elif "label" in point_cloud.point:
        label_idx = "label"
    elif "labels" in point_cloud.point:
        label_idx = "labels"
    else:
        label_idx = None

    return label_idx



def get_intensity_attribute(point_cloud):
    if "intensity" in point_cloud.point:
        intensity_idx = "intensity"
    elif "reflectance" in point_cloud.point:
        intensity_idx = "reflectance"
    elif "reflection" in point_cloud.point:
        intensity_idx = "reflection"
    else:
        intensity_idx = None

    return intensity_idx



def get_color_attribute(point_cloud):
    # FIXME -> is color not the intensity in point clouds?
    # if yes -> add color to the get intensity and remove this function
    if "colors" in point_cloud.point:
        color_idx = "colors"
    elif "color" in point_cloud.point:
        color_idx = "color"
    else:
        color_idx = None

    return color_idx



def get_normal_attribute(point_cloud):
    if "normals" in point_cloud.point:
        normal_idx = "normals"
    elif "normal" in point_cloud.point:
        normal_idx = "normal"
    else:
        normal_idx = None

    return normal_idx



def get_attributes(point_cloud):
    coordinate_idx = get_coordinate_attribute(point_cloud)
    label_idx = get_class_attribute(point_cloud)
    intensity_idx = get_intensity_attribute(point_cloud)
    color_idx = get_color_attribute(point_cloud)
    normal_idx = get_normal_attribute(point_cloud)

    return coordinate_idx, label_idx, intensity_idx, color_idx, normal_idx



# ------------------------------
# > Extract Attribute as Color <
# ------------------------------
def get_color_from_intensity(point_cloud):
    intensity_idx = get_intensity_attribute(point_cloud)
    if intensity_idx is not None:
        refl = point_cloud.point[intensity_idx].numpy().squeeze()

        # normalize to [0,1]
        refl = (refl - refl.min()) / (refl.max() - refl.min() + 1e-8)

        # grayscale mapping
        colors = np.stack([refl, refl, refl], axis=1)  # (N,3)
    else:
        colors = None

    return colors



def get_color_from_height(point_cloud):
    coordinate_idx = get_coordinate_attribute(point_cloud)
    if coordinate_idx is not None:
        points = point_cloud.point[coordinate_idx].numpy()
        z = points[:, 2]

        z = (z - z.min()) / (z.max() - z.min() + 1e-8)

        # blue -> red gradient
        colors = np.stack([z, 0*z, 1-z], axis=1)
    else:
        colors = None

    return colors



def get_color_from_class(point_cloud):
    label_idx = get_class_attribute(point_cloud)
    if label_idx is not None:
        points = point_cloud.point[label_idx].numpy()

        classes = np.unique(points)

        # print(points)
        # print(points.shape)

        num_class = classes.shape[0]
        
        color_mapping = np.random.rand(num_class, 3).astype(np.float32)

        class_to_idx = {class_: idx for idx, class_ in enumerate(classes)}

        colors = np.array([
            color_mapping[class_to_idx[class_[0]]] for class_ in points
        ], dtype=np.float32)
    else:
        colors = None

    return colors



def set_color(point_cloud, mode):
    if mode == "height":
        colors = get_color_from_height(point_cloud)
    elif mode == "intensity":
        colors = get_color_from_intensity(point_cloud)
    elif mode == "class":
        colors = get_color_from_class(point_cloud)

    if colors is not None:
        color_idx = get_color_attribute(point_cloud)
        color_idx = color_idx if color_idx else "colors"
        point_cloud.point[color_idx] = o3d.core.Tensor(colors, dtype=o3d.core.Dtype.Float32)
    return point_cloud



# -------------------------
# > Keep Ground Filtering <
# -------------------------
def filter_ground_with_RANSAC(point_cloud, distance_threshold=0.2, ransac_n=3, num_iterations=1000):
    # get points
    points = point_cloud.point[get_coordinate_attribute(point_cloud)].numpy()

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
    points = point_cloud.point[get_coordinate_attribute(point_cloud)].numpy()

    # naive approximation: subtract global minimum
    z = points[:, 2]
    z_norm = z - z.min()

    mask = z_norm < threshold
    # filtered_points = points[mask]

    pc_filtered = o3d.t.geometry.PointCloud()

    # for key in get_attributes(point_cloud):
    #     if key:
    #         data = point_cloud.point[key].numpy()
    #         pc_filtered.point[key] = o3d.core.Tensor(data[mask])

    for key in point_cloud.point:
        data = point_cloud.point[key].numpy()
        pc_filtered.point[key] = o3d.core.Tensor(data[mask])

    return pc_filtered



# ------------------
# > BEV Projection <
# ------------------
# BEV (Bird-Eye-View) are images tile-based over the point cloud
# and grid based (one grid-point is one pixel) but with the difference
# that one grid-point can contain multiple values (because multiple points
# on the same cooridnate) + no RGB values but height, intensity and
# sometimes other values, like normal.
def bev_projection(point_cloud):
    # ... -> tile based bev images from pc
    return point_cloud



def bev_projection_mapping(images, image_number, pixel, point_cloud):
    # ... -> back-projection from from one specific pixel tile based bev images from pc
    return point_cloud





