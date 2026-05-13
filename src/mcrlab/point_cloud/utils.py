# -----------
# > Imports <
# -----------
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm

from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt



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
    elif "coordinates" in point_cloud.point:
        coordinate_idx = "coordinates"
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



def get_instance_attribute(point_cloud):
    if "instance" in point_cloud.point:
        instance_idx = "instance"
    elif "instances" in point_cloud.point:
        instance_idx = "instances"
    else:
        instance_idx = None

    return instance_idx



def get_intensity_attribute(point_cloud):
    if "intensity" in point_cloud.point:
        intensity_idx = "intensity"
    elif "intensities" in point_cloud.point:
        intensity_idx = "intensities"
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
    elif "rgb" in point_cloud.point:
        color_idx = "rgb"
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

        colors *= 255
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
    if label_idx is None:
        return None

    labels = point_cloud.point[label_idx].numpy().reshape(-1)

    unique_labels = np.unique(labels)

    np.random.seed(42)
    color_map = {
        label: np.random.rand(3).astype(np.float32)
        for label in unique_labels
    }

    colors = np.zeros((labels.shape[0], 3), dtype=np.float32)

    for label, color in color_map.items():
        # only debugging:
        # if label != 104002:  # 105800, 106200, >104002< -> manhole, 101100, 101701
        #     continue

        colors[labels == label] = color

    # Debug
    # print("Class mapping:")
    # for k, v in color_map.items():
    #     print(f"  ID {k} → color {v}")

    # # debugging only, find id via color
    # target = np.array([208, 231, 86]) / 255.0    # via color picker from visualization, only possible because of seed!

    # best_id = [None] * 5
    # best_dist = [float("inf")] * 5

    # for k, v in color_map.items():
    #     cur_dist = np.linalg.norm(v - target)
    #     cur_k = k
    #     for idx in range(len(best_dist)):
    #         if cur_dist < best_dist[idx]:
    #             previously_best_dist = best_dist[idx]
    #             previously_best_id = best_id[idx]
    #             best_dist[idx] = cur_dist
    #             best_id[idx] = cur_k
    #             cur_dist = previously_best_dist
    #             cur_k = previously_best_id

    #             # no 'continue'! -> other have to get updated too!

    # print("Closest ID:", best_id, "distance:", best_dist)

    return colors



def get_color_from_instance(point_cloud):
    label_idx = get_instance_attribute(point_cloud)
    if label_idx is None:
        return None

    labels = point_cloud.point[label_idx].numpy().reshape(-1)

    unique_labels = np.unique(labels)

    np.random.seed(42)
    color_map = {
        label: np.random.rand(3).astype(np.float32)
        for label in unique_labels
    }

    colors = np.zeros((labels.shape[0], 3), dtype=np.float32)

    for label, color in color_map.items():
        colors[labels == label] = color

    return colors



def set_color(point_cloud, mode, normalize=True):
    if mode == "height":
        colors = get_color_from_height(point_cloud)
    elif mode == "intensity":
        colors = get_color_from_intensity(point_cloud)
    elif mode in ["class", "classes", "label", "labels"] :
        colors = get_color_from_class(point_cloud)
    elif mode in ["instance", "instances"] :
        colors = get_color_from_instance(point_cloud)

    if colors is not None:
        color_idx = get_color_attribute(point_cloud)
        color_idx = color_idx if color_idx else "colors"

        if normalize:
            # dtype = colors.dtype
            if isinstance(colors, o3d.core.Tensor):
                colors = colors.numpy()
            colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))

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



# ---------------
# > Other Utils <
# ---------------
def extract_manhole(points, label_value=104002, points_around_dist=5):
    if not isinstance(points, o3d.t.geometry.PointCloud):
        raise ValueError(f"Points must be o3d.t.geometry.PointCloud, but got: {type(points)}")

    manholes = []

    semantic_class_idx = get_class_attribute(points)
    instance_idx = get_instance_attribute(points)

    if instance_idx is not None:
        instance_ids = points.point[instance_idx].numpy().ravel()
        item_ = np.unique(instance_ids)
        labels = points.point[semantic_class_idx].numpy().ravel()
    else:
        # raise RuntimeError("PointCloud does not have instance Label! But is needed.")
        labels = points.point[semantic_class_idx].numpy().ravel()

        # Filter to only manhole points before clustering
        manhole_mask = labels == label_value
        manhole_indices = np.where(manhole_mask)[0]

        if len(manhole_indices) == 0:
            return manholes

        all_coords = points.point[get_coordinate_attribute(points)].numpy()
        manhole_coords = all_coords[manhole_indices]

        dbscan = DBSCAN(eps=3, min_samples=2)
        dbscan.fit(manhole_coords)

        cluster_labels = dbscan.labels_           # shape: (n_manhole_points,)
        item_ = np.unique(cluster_labels)

    for cur_item in item_:
        
        if instance_idx is not None:
            cur_instance_id = cur_item
            if cur_instance_id < 0:
                continue
            indices = np.where(instance_ids == cur_instance_id)[0]
        else:
            if cur_item < 0:                      # skip noise points
                continue
            # indices back into the *original* point cloud
            indices = manhole_indices[cluster_labels == cur_item]

        cluster = points.select_by_index(indices)

        if len(indices) < 30:
            continue

        # check if manhole class
        if np.all(labels[indices] != label_value):
            continue

        cluster_points = cluster
        cluster_points_arr = cluster.point[get_coordinate_attribute(cluster)].numpy()

        # add other additional points from around
        if points_around_dist > 0 :
            min_bound = cluster_points_arr.min(axis=0)
            max_bound = cluster_points_arr.max(axis=0)

            # expand by your margin
            min_bound[:2] -= points_around_dist
            max_bound[:2] += points_around_dist

            all_points = points.point[get_coordinate_attribute(points)].numpy()

            mask = (
                (all_points[:, 0] >= min_bound[0]) & (all_points[:, 0] <= max_bound[0]) &
                (all_points[:, 1] >= min_bound[1]) & (all_points[:, 1] <= max_bound[1])
            )

            expanded_indices = np.where(mask)[0]
            final_indices = np.union1d(indices, expanded_indices)
            cluster_points = points.select_by_index(final_indices)
            del all_points

        manholes.append(cluster_points)
    return manholes



# ----------------
# > Circle Utils <
# ----------------
def sample_uniform_circle(n, max_radius=1.0, add_z=False):
    theta = np.random.uniform(0, 2*np.pi, n)
    r = max_radius * np.sqrt(np.random.uniform(0, 1, n))

    # convert to x, y coordiantes from circle coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    if add_z:
        return np.stack([x, y, np.zeros(x.shape)], axis=1)  
    else:  
        return np.stack([x, y], axis=1)



def sample_biased_circle(n, radius=1.0, bias_strength=1.0):
    points = []

    while len(points) < n:
        # candidate
        p = sample_uniform_circle(1, radius)[0]
        x, y = p

        # Bias -> right/left side more dense?
        weight = 0.5 + 0.5 * (1 + bias_strength * x / radius)

        if np.random.rand() < weight:
            points.append(p)

    return np.array(points)



def add_random_dense_manipulation(points, n):
    # Choose a random point
    idx = np.random.randint(0, len(points))
    chosen_point = points[idx] # [x, y]

    # calc max distance for point changing range
    max_dist = np.max(pdist(points))
    radius = max_dist * 0.01

    # create new points
    offsets = np.random.uniform(-radius, radius, size=(n, 2))
    new_points = chosen_point + offsets

    # merge to the old points
    return np.concatenate([points, new_points], axis=0)



def add_random_dense_manipulation_point_cloud(points, n):
    points = points.clone()
    points_arr = points.point[get_coordinate_attribute(points)].numpy()

    # Choose a random point
    idx = np.random.randint(0, len(points_arr))
    chosen_point = points_arr[idx] # [x, y, z]

    # calc max distance for point changing range
    max_dist = np.max(pdist(points_arr))
    radius = max_dist * 0.1

    # create new points
    offsets = np.random.uniform(-radius, radius, size=(n, 3))
    new_points = chosen_point + offsets

    # merge to the old points
    merged = np.concatenate([points_arr, new_points], axis=0)

    points.point[get_coordinate_attribute(points)] = o3d.core.Tensor(merged, o3d.core.Dtype.Float32)

    return points



# def classify_manhole(points,
#                      num_angle_bins=180,
#                      radial_percentile=95,
#                      residual_threshold=0.03,
#                      min_coverage=0.7,
#                      save_path=None,
#                      should_plot=False):

#     """
#     Detect whether a 2D point cloud represents a circular structure.

#     This method is designed for sparse and noisy LiDAR point clouds where
#     the interior point distribution is unreliable due to scanlines,
#     occlusions, varying density, or incomplete coverage.

#     Instead of analyzing the full point distribution, the algorithm focuses
#     on the OUTER BOUNDARY of the point cloud, because the geometric shape
#     of a circle is primarily encoded in its border.

#     Algorithm Overview
#     ------------------
#     1. Estimate an initial center using the mean of all points.

#     2. Convert all points into polar coordinates:
#            radius r
#            angle theta

#     3. Split the point cloud into angular bins.

#     4. For each angular bin:
#            select only the outermost points
#            (using a high radial percentile)

#        This extracts a robust approximation of the boundary even when:
#            - scanlines are visible
#            - point density is anisotropic
#            - the interior is noisy
#            - parts of the circle are missing

#     5. Fit a circle to the extracted boundary points using least squares.

#     6. Measure:
#            - radial residual error
#            - angular coverage

#     7. Decide whether the shape is circular based on:
#            - normalized fitting error
#            - sufficient angular coverage
#     """

#     if not isinstance(points, np.ndarray):
#         raise ValueError("points must be numpy array")

#     x = points[:, 0]
#     y = points[:, 1]

#     # 1. estimate center
#     cx = np.mean(x)
#     cy = np.mean(y)

#     x0 = x - cx
#     y0 = y - cy

#     theta = np.arctan2(y0, x0)
#     r = np.sqrt(x0**2 + y0**2)

#     # 2. extract boundary points
#     bins = np.linspace(-np.pi, np.pi, num_angle_bins + 1)

#     boundary_points = []

#     for i in range(num_angle_bins):

#         mask = (theta >= bins[i]) & (theta < bins[i+1])

#         if np.sum(mask) == 0:
#             continue

#         r_bin = r[mask]

#         # take outer percentile instead of max
#         r_thresh = np.percentile(r_bin, radial_percentile)

#         idx = np.where(mask)[0]

#         selected = idx[r_bin >= r_thresh]

#         boundary_points.extend(selected.tolist())

#     boundary_points = np.unique(boundary_points)

#     bx = x[boundary_points]
#     by = y[boundary_points]

#     # 3. circle fit
#     A = np.column_stack([
#         2*bx,
#         2*by,
#         np.ones(len(bx))
#     ])

#     b = bx**2 + by**2

#     params, *_ = np.linalg.lstsq(A, b, rcond=None)

#     cx_fit, cy_fit, c = params

#     radius = np.sqrt(c + cx_fit**2 + cy_fit**2)

#     # 4. residuals
#     distances = np.sqrt(
#         (bx - cx_fit)**2 +
#         (by - cy_fit)**2
#     )

#     residuals = np.abs(distances - radius)

#     mean_residual = np.mean(residuals)

#     # normalized residual
#     normalized_error = mean_residual / radius

#     # 5. angular coverage
#     theta_boundary = np.arctan2(by - cy_fit, bx - cx_fit)

#     hist, _ = np.histogram(
#         theta_boundary,
#         bins=num_angle_bins
#     )

#     coverage = np.sum(hist > 0) / num_angle_bins

    
#     # 6. make final decision
#     is_circle = (
#         normalized_error < residual_threshold
#         and coverage > min_coverage
#     )

#     result = {
#         "is_circle": is_circle,
#         "center": np.array([cx_fit, cy_fit]),
#         "radius": radius,
#         "normalized_error": normalized_error,
#         "coverage": coverage,
#         "boundary_points": boundary_points
#     }

    
#     # 7. visualize
#     if should_plot or save_path is not None:
#         fig, ax = plt.subplots(figsize=(7,7))

#         ax.scatter(x, y, s=5, alpha=0.3)

#         ax.scatter(
#             bx,
#             by,
#             s=12,
#             label="boundary"
#         )

#         circle = plt.Circle(
#             (cx_fit, cy_fit),
#             radius,
#             fill=False,
#             linewidth=2
#         )

#         ax.add_patch(circle)

#         ax.set_aspect("equal")
#         ax.legend()

#         if is_circle:
#             ax.set_title("Shape Check (✅ Is a Circle)")
#         else:
#             ax.set_title("Shape Check (❌ Is not a Circle)")

#         if save_path is not None:
#             plt.savefig(save_path)

#         if should_plot:
#             plt.show()

#     return result



def classify_manhole(points, save_path=None, should_plot=False):
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. check data format
    if isinstance(points, o3d.t.geometry.PointCloud):
        points = points.point[get_coordinate_attribute(points)].numpy()
    
    # 2. Project to 2D
    points_2d = points[:, :2]
    
    # 3. Compute the Convex Hull -> the "envelope" of the points
    # This gives us a clean shape even if the LIDAR points are sparse inside
    hull = ConvexHull(points_2d)
    
    # 4. Extract Area and Perimeter (length) from the hull
    area = hull.volume  # In 2D, 'volume' is the area
    perimeter = hull.area  # In 2D, 'area' is the perimeter
    
    # 5. Calculate Circularity
    # -> https://en.wikipedia.org/wiki/Isoperimetric_inequality
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    
    # 6. Classification Logic
    # Threshold is usually around 0.88 - 0.90
    if circularity > 0.89:
        # result = ("Circular", circularity)
        is_circle = True
    else:
        # result = ("Square/Rectangular", circularity)
        is_circle = False
    

    # 7. visualize
    if should_plot or save_path is not None:
        fig, ax = plt.subplots(figsize=(7,7))

        ax.scatter(points_2d[:, 0], points_2d[:, 1], s=5, alpha=0.3, label="Points")

        # hull vertices in correct order
        hull_points = points_2d[hull.vertices]

        # close the polygon by repeating the first point
        hull_points = np.vstack([hull_points, hull_points[0]])

        # plot convex hull
        ax.plot(hull_points[:, 0],
                hull_points[:, 1],
                'r-', lw=2, label="Convex Hull")

        # # highlight hull vertices
        # ax.scatter(hull_points[:, 0],
        #         hull_points[:, 1],
        #         color='red', s=30)

        ax.set_aspect("equal")
        ax.legend()

        if is_circle:
            ax.set_title("Shape Check (✅ Is a Circle)")
        else:
            ax.set_title("Shape Check (❌ Is not a Circle)")

        if save_path is not None:
            plt.savefig(save_path)

        if should_plot:
            plt.show()

        plt.close(fig)


    return is_circle



# -------------------
# > Splitting Utils <
# -------------------
def split_point_cloud_into_multiple(point_cloud, path,  init_tile_size=500.0, overlap=3.0):
    if not isinstance(point_cloud, o3d.t.geometry.PointCloud):
        raise TypeError(f"Expected point cloud as o3d.t.geometry.PointCloud but got: `{type(point_cloud)}`")

    # print(f"Point Cloud have following attributes: {point_cloud.point.keys()}")

    points = point_cloud.point[get_coordinate_attribute(point_cloud)].numpy()  # np.asarray(pc_legacy.points)

    print(f"Point Amount: {points.shape}")

    x_min, y_min = points[:, :2].min(axis=0)
    x_max, y_max = points[:, :2].max(axis=0)

    print(f"Create Chunks from ({x_min}, {y_min}) to ({x_max}, {y_max})")

    tile_size = init_tile_size
    step = tile_size - overlap

    x_values = np.arange(x_min, x_max + step, step)
    y_values = np.arange(y_min, y_max + step, step)

    chunk_id = 0

    sizes = []

    for x in tqdm(x_values, total=len(x_values), desc="Creating Chunks"):
        for y in y_values:
            mask = (
                (points[:, 0] >= x) &
                (points[:, 0] <= (x+tile_size)) &
                (points[:, 1] >= y) &
                (points[:, 1] <= (y+tile_size))
            )

            idx = np.where(mask)[0]

            if len(idx) == 0:
                continue

            sizes.append(len(idx))

            idx_tensor = o3d.core.Tensor(
                idx,
                dtype=o3d.core.Dtype.Int64
            )

            tile = point_cloud.select_by_index(idx_tensor)

            filename = f"sud_splitted_chunk_{chunk_id}.ply"

            # unsqueeze one dimensional attributes
            for key, tensor in tile.point.items():
                if len(tensor.shape) == 1:
                    tile.point[key] = tensor.reshape((-1, 1))

            o3d.t.io.write_point_cloud(os.path.join(path, filename), 
                                     tile)

            chunk_id += 1

    print(f"Splitted Point Cloud into {len(sizes)} chunks.")
    sizes = np.array(sizes)
    print(f"Mean Points: {sizes.mean():.2f}")
    print(f"Min Points: {sizes.min():.2f}")
    print(f"Max Points: {sizes.max():.2f}")
    print(f"STD Points: {sizes.std():.2f}\n")
                









