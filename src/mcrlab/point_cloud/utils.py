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
# BEV = point cloud -> discretization -> aggregation -> image
# Idea: 3D -> BEV -> CNN/Transformer -> predictions ((tile_id, px, py), ...)
# -> corresponding 3D points (projection-mapping)

# FIXME -> add saving and loading of BEV in preprocessing

def bev_projection(point_cloud, tile_size=50.0, resolution=2.0):
    """
    *Put explanation here
    """
    # if isinstance(point_cloud, PointCloudTensor):
    #     point_cloud = point_cloud.get_as_o3d()

    points  = point_cloud.point[get_coordinate_attribute()].numpy()
    n_points = points.shape[0]

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # global boundings
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    tiles = []
    meta = []

    tile_id = 0

    for cur_x in np.arange(x_min, x_max, tile_size):
        for cur_y in np.arange(y_min, y_max, tile_size):
            # select points inside of tile
            mask = (
                (x >= cur_x) & (x < cur_x + tile_size) &
                (y >= cur_y) & (y < cur_y + tile_size)
            )

            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue

            points_tile = points[idxs]

            # convert to pixels coordinates
            points_x = ((points_tile[:, 0] - cur_x) / resolution).astype(int)
            points_y = ((points_tile[:, 1] - cur_y) / resolution).astype(int)

            height = int(tile_size / resolution)
            width = int(tile_size / resolution)

            # create empty BEV grid
            bev = np.zeros((height, width), dtype=np.float32)

            # fill bev grid
            #    aggregate max height per pixel
            np.maximum.at(bev, (points_x, points_y), points_tile[:, 2])

            # + store indices per pixel
            # pixel_to_point_idxs = {}

            # for point_idx in range(len(points_x)):
            #     key = (points_x[point_idx], points_y[point_idx])

            #     # height aggregation
            #     bev[points_x[point_idx], points_y[point_idx]] = max(
            #         bev[points_x[point_idx], points_y[point_idx]],
            #         points_tile[point_idx, 2]
            #     )

            # build mapping: pixel -> point indices
            pixel_ids = points_x * width + points_y  # flatten 2D to 1D pixel index

            #  group points by pixel
            unique_pixels, inverse = np.unique(pixel_ids, return_inverse=True)

            #  store mapping
            pixel_to_indices = [np.where(inverse == idx)[0] \
                                    for idx in range(len(unique_pixels))]

            #     if key not in pixel_to_point_idxs.keys():
            #         pixel_to_point_idxs[key] = []
            #     pixel_to_point_idxs.append(idxs[point_idx])

            tiles.append(bev)

            meta.append({
                "tile_id": tile_id,
                "cur_x": cur_x,
                "cur_y": cur_y,
                # "pixel_map": pixel_to_point_idxs
                "unique_pixels": unique_pixels,
                "pixel_to_indices": pixel_to_indices,
                "global_indices": idxs
            })

            tile_id += 1

    return tiles, meta



def bev_projection_mapping(point_cloud, meta, tile_id, pixel):
    """
    *Put explanation here
    """
    tile = meta[tile_id]

    point_x, point_y = pixel
    height = width = int(np.sqrt(tile["unique_pixels"].max() + 1))

    pixel_id = point_x * width + point_y

    # find pixel
    matches = np.where(tile["unique_pixels"] == pixel_id)[0]
    if len(matches) == 0:
        return None
    
    group_idx = matches[0]

    local_indices = tile["pixel_to_indices"][group_idx]
    globla_indices = tile["global_indices"][local_indices] 

    points = point_cloud.point[get_coordinate_attribute()].numpy()[globla_indices]

    return points





