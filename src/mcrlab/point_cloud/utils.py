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
        raise ValueError(f"No coordinate attribute found, please add your key in the `get_coordinate_attribute` function.\nPoint-Cloud Info:\n{point_cloud}")

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
    elif mode in ["class", "classes", "label", "labels"] :
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
    coordinate_idx = get_coordinate_attribute(point_cloud)
    points = point_cloud.point[coordinate_idx].numpy()
    x = points[:, 0]
    y = points[:, 1]

    # for every cell
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min

    y_min = y.min()
    y_max = y.max()
    y_range = y_max - y_min
    
    # cell_size = 2.0  # 200 cm
    cell_size = min(y_range, x_range) * 0.25
    x_n_steps = int(np.ceil(x_range / cell_size))
    y_n_steps = int(np.ceil(y_range / cell_size))

    ground = None  # o3d.t.geometry.PointCloud()
    for cur_x_cell in range(x_n_steps):  # np.arange(x_min, x_max, x_step):
        cur_x_min = x_min + (cur_x_cell * cell_size)
        cur_x_max = cur_x_min + cell_size
        for cur_y_cell in range(y_n_steps):  # np.arange(y_min, y_max, y_step):
            cur_y_min = y_min + (cur_y_cell * cell_size)
            cur_y_max = cur_y_min + cell_size

            # get only cell
            if cur_x_max >= x_max:
                x_mask = (x >= cur_x_min) & (x <= cur_x_max)
            else:
                x_mask = (x >= cur_x_min) & (x < cur_x_max)
            if cur_y_max >= y_max:
                y_mask =  (y >= cur_y_min) & (y <= cur_y_max)
            else:
                y_mask =  (y >= cur_y_min) & (y < cur_y_max)
            mask = x_mask & y_mask

            cell_pc = point_cloud.select_by_mask(mask)  # no sideffect, makes copy
            # all attributes filtered!

            cell_points = cell_pc.point[coordinate_idx].numpy()
            if len(cell_points) == 0:  # < 5
                continue

            z = cell_points[:, 2]

            # naive approximation: subtract cell minimum
            # works better with previous outlier removal
            # z_norm = z - z.min()
            z_norm = z - np.percentile(z, 5)

            cell_mask = z_norm < threshold
            # filtered_points = points[mask]

            if ground is None:
                ground = cell_pc.select_by_mask(cell_mask)
            else:
                ground += cell_pc.select_by_mask(cell_mask)

            # for key in get_attributes(point_cloud):
            #     if key:
            #         data = point_cloud.point[key].numpy()
            #         pc_filtered.point[key] = o3d.core.Tensor(data[mask])

            # for key in point_cloud.point:
            #     data = point_cloud.point[key].numpy()
            #     pc_filtered.point[key] = o3d.core.Tensor(data[mask])

    return ground









