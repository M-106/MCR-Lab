# ----------
# > Import <
# ----------
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from sklearn.preprocessing import StandardScaler
from skimage.measure import ransac, CircleModel
import cv2
import pyransac3d as pyrsc

import open3d as o3d
import torch

from mcrlab.classic.utils import fit_plane, plane_basis, project_to_plane, \
                                  visualize_circle_fit
from mcrlab.point_cloud.utils import get_coordinate_attribute, get_intensity_attribute, get_class_attribute, get_instance_attribute, \
                                     barycentric_downsample_manhole
from mcrlab.point_cloud.data import CSFGroundFilterTransform, bev_gen_wrapper
from mcrlab.projection import bev_projection, bev_back_projection

from mcrlab.classic.least_squares import fit_circle_least_squares, fit_circle_least_squares_3D
from mcrlab.point_cloud.shape_check import circle_shape_check



# ------------------
# > RANSAC 2D & 3D <
# ------------------
# Short explanation: 
#           1. Randomly pick 3 points (minimal sample for a circle)
#           2. Fit a circle to those 3 points
#               (either exact geometry OR solving equations)
#           3. Compute residuals
#               (distance of all points to the circle boundary)
#           4. Count points that are close enough (inliers)
#           5. Repeat and keep the model with most inliers
#       Therefore RANSAC is not a fitting method by itself but
#       a wrapper for fitting methods.
#   > Fitting = choosing model parameters so the model matches the data as closely as possible
#   Least squares → minimize squared distances
#       Just predicts the Circle from the 3 Points here, without Backpropagation, just simple minimization via numpy
#   RANSAC → maximize number of inliers (not minimize directly!)
#   RANSAC always need a fitting method but it does not have to be optimization-based
#   It tries different candidate shapes and takes the one which matches the most.
#   RANSAC takes a sample and test the circle against all points to make the distance to the circle boundary as close as possbile
# Cite: FIXME

# notice that ransac can be applied on
# the whole point cloud or on candidates

# class CircleModel:
#     """
#     SkLearn/SkImage Model.

#     Least Square Fit Model for RANSAC.

#     Fits a circle to 2D points: (x - cx)^2 + (y - cy)^2 = r^2
#     """
#     def __init__(self):
#         self.params = None

#     @classmethod
#     def from_estimate(cls, data, *args, **kwargs):
#         instance = cls()
#         success = instance.estimate(data, *args, **kwargs)
#         return instance if success else None

#     def estimate(self, data):
#         """
#         Fit circle to minimal sample (3 points) using algebraic method.

#         Called every iteration of RANSAC with 3 random points.
#         """
#         x, y = data[:, 0], data[:, 1]

#         # Check for collinearity (area of triangle ~ 0)
#         # print(f"X Shape: {x.shape}")
#         # print(f"Y Shape: {y.shape}")
#         # print(f"All original Shape: {data.shape}")
#         mat = np.column_stack((x, y, np.ones(len(x))))
#         if len(x) < 3:
#             return False

#         if len(x) == 3:
#             mat = np.column_stack((x, y, np.ones(3)))
#             if abs(np.linalg.det(mat)) < 1e-6:
#                 return False

#         # Build linear system: 2x*cx + 2y*cy + r^2 - cx^2 - cy^2 = x^2 + y^2
#         A = np.column_stack((2 * x, 2 * y, np.ones(len(x))))
#         b = x**2 + y**2

#         try:
#             result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
#             # This is also LEast Square Fit BUT:
#             # - solves a linear system
#             # - one shot (no iterations)
#             # - no gradients
#             # - no "learning"
#         except np.linalg.LinAlgError:
#             return False
        
#         cx, cy, c = result

#         r_sq = c + cx**2 + cy**2
#         if r_sq <= 0:
#             return False
        
#         self.params = (cx, cy, np.sqrt(r_sq))
#         return True

#     def residuals(self, data):
#         """
#         Euclidean distance from each point to the circle boundary.
#         """
#         if self.params is None:
#             raise ValueError("Got no params, unexpected!")
#             return np.full(len(data), np.inf)
        
#         cx, cy, r = self.params
#         x, y = data[:, 0], data[:, 1]
        
#         # return np.abs( np.sqrt( (x - cx)**2 + (y - cy)**2 ) - r )
    
#         # distances = np.sqrt((x - cx)**2 + (y - cy)**2)
#         # return (distance - r) ** 2

#         # return np.abs((x - cx)**2 + (y - cy)**2 - r**2)

#         f = (x - cx)**2 + (y - cy)**2 - r**2

#         grad_sq = 4 * ((x - cx)**2 + (y - cy)**2)

#         return np.abs(f) / np.sqrt(grad_sq + 1e-12)

#     def predict_xy(self, t):
#         cx, cy, r = self.params
#         return np.column_stack((cx + r * np.cos(t), cy + r * np.sin(t)))



def fit_circle_ransac(x, y, method="sklearn"):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError(f"Points must be a numpy array, but got '{type(x)}, {type(y)}'")
    
    if method == "sklearn":
        data = np.column_stack((x, y))

        model, inliers = ransac(
            data,
            CircleModel,
            min_samples=3,
            residual_threshold=0.02,  #  * np.std(data),  # 1 cm tolerance (same as before) + dynamic thresholding
            max_trials=10000,
        )

        if model is None:
            raise RuntimeError("RANSAC failed to find a valid circle")
        
        # refinment using all inliers:
        inlier_points = data[inliers]
        model.from_estimate(inlier_points)

        # Residuals for all points
        all_residuals = model.residuals(data)

        # Residuals for inliers only
        inlier_residuals = all_residuals[inliers]

        # Common error metrics
        error = np.mean(inlier_residuals)
        # rmse = np.sqrt(np.mean(inlier_residuals**2))
        # median_error = np.median(inlier_residuals)
        # max_error = np.max(inlier_residuals)

        # RANSAC quality
        # inlier_ratio = np.sum(inliers) / len(inliers)

        # cx, cy, radius = model.params
        cx, cy = model.center
        radius = model.radius
        
        center = np.array([cx, cy])
        axis = np.array([0, 0, 1])  # Normal axis (z), matches 3D RANSAC convention
    elif method == "pyransac3d":
        points_3d = np.column_stack((x, y, np.zeros_like(x)))
        circle = pyrsc.Circle()
        center, axis, radius, inliers = circle.fit(points_3d, thresh=0.01)  # 5 cm tolerance -> maybe increase if not working because of scanline-artefacts (gaps)
        center = center[:2]
        
        distances = np.linalg.norm(
            np.column_stack((x, y)) - center,
            axis=1
        )

        # Residuals = distance to circle boundary
        residuals = np.abs(distances - radius)

        # Inlier residuals
        inlier_residuals = residuals[inliers]

        # Metrics
        error = np.mean(inlier_residuals)
        # rmse = np.sqrt(np.mean(inlier_residuals**2))
        # median_error = np.median(inlier_residuals)
        # max_error = np.max(inlier_residuals)

        # inlier_ratio = len(inliers) / len(points_3d)
    else:
        raise ValueError(f"Got unknown method: '{method}'")
    
    # FIXME -> or own implementation? First get others work

    return center, axis, radius, inliers, error



def fit_circle_ransac_3D(points, use_projection=True):
    if not isinstance(points, np.ndarray):
        raise TypeError(f"Points must be a numpy array, but got '{type(points)}'")
    
    if use_projection:
        # prepraration for projection
        centroid, normal = fit_plane(points)
        basis_x, basis_y = plane_basis(normal)

        # projection into 2D
        x, y = project_to_plane(centroid=centroid, points=points, basis_x=basis_x, basis_y=basis_y)

        # optimize circle shape
        center, axis, radius, inliers, error = fit_circle_ransac(x, y)
        axis = normal

        # back projection
        center = centroid + (center[0] * basis_x) + (center[1] * basis_y)
    else:
        circle = pyrsc.Circle()
        center, axis, radius, inliers = circle.fit(points, thresh=10.0, maxIteration=1000)

        print(center)

        distances = np.linalg.norm(
            points - center,
            axis=1
        )

        # Residuals = distance to circle boundary
        residuals = np.abs(distances - radius)

        # Inlier residuals
        inlier_residuals = residuals[inliers]

        # Metric
        error = np.mean(inlier_residuals)

    return center, axis, radius, inliers, error



# --------------------------------
# > Hough Transformation 2D & 3D <
# --------------------------------
# Short explanation: FIXME
# Cite: FIXME

# FIXME



# -------------
# > Pipelines <
# -------------
# FIXME -> Except Refinemethod and method, example pipeline: RANSAC -> find inliers -> scipy.optimize.least_squares (refinement)
def extract_center_point(points, method, use_2d_version, use_projection=False, apply_downsampling=False):
    """
    method: "least_square", "ransac"
    """
    if apply_downsampling:
        points = barycentric_downsample_manhole(points, radius=0.05, min_neighbors=1)

    if use_2d_version:
        x, y = points

    if method == "least_square":
        if use_2d_version:
            center_x, center_y, r, mean_distance_error, loss = fit_circle_least_squares(x, y)
            return {
                "center": np.array([center_x, center_y]),
                "radius": r,
                "inliers": None,
                "error": mean_distance_error,
                "loss": loss,
                "input_points": points
            }
        else:
            center_3D, normal, r, mean_distance_error, loss = fit_circle_least_squares_3D(points)
            return {
                "center": center_3D,
                "radius": r,
                "inliers": None,
                "error": mean_distance_error,
                "loss": loss,
                "input_points": points
            }
    elif method == "ransac":
        if use_2d_version:
            center, axis, r, inliers, error = fit_circle_ransac(x, y, method="sklearn")
        else:
            center, axis, r, inliers, error = fit_circle_ransac_3D(points, use_projection=use_projection)

        return {
            "center": center,
            "radius": r,
            "inliers": inliers,
            "error": error,
            "loss": None,
            "input_points": points
        }
    else:
        raise ValueError(f"Center Point Extraction Method not Found: {method}")



def use_points_and_extract_center_point(clusters, method, use_projection=True, apply_downsampling=False):
    candidates = []

    for cluster in clusters:

        if isinstance(cluster, np.ndarray):
            cluster_points = cluster
        else:
            cluster_points = cluster.point[get_coordinate_attribute(cluster)].numpy()

        # print("Cluster-Point Shape", cluster_points.shape)

        result = extract_center_point(
            points=cluster_points,
            method=method,
            use_2d_version=False,
            use_projection=use_projection,
            apply_downsampling=apply_downsampling
        )

        if result is None:
            continue

        center = result["center"]
        radius = result["radius"]
        inliers = result["inliers"]
        error = result["error"]
        loss = result["loss"]
        input_points = result["input_points"]

        # if 0.20 < radius < 0.50 and (inliers is None or len(inliers) > 30):
        candidates.append((center, radius, cluster, inliers, error, loss, input_points))

    return candidates



def use_label_candidates_and_extract_center_point(points, use_2d_version, label_value, 
                                                  method="least_square", use_projection=True, cluster_if_needed=True,
                                                  apply_downsampling=False):
    candidates = []
    
    # 2D CASE
    if use_2d_version:
        if not isinstance(points, (list, tuple)):
            raise ValueError(f"Points must be list or tuple, but got: {type(points)}")

        x, y = points

        raise NotImplementedError("2D labeling logic not implemented yet")

    # 3D CASE
    else:
        if not isinstance(points, o3d.t.geometry.PointCloud):
            raise ValueError(f"Points must be o3d.t.geometry.PointCloud, but got: {type(points)}")

        semantic_class_idx = get_class_attribute(points)
        instance_idx = get_instance_attribute(points)

        if instance_idx is not None:
            instance_ids = points.point[instance_idx].numpy().ravel()
            unique_ids = np.unique(instance_ids)
            labels = points.point[semantic_class_idx].numpy().ravel()

            for cur_instance_id in unique_ids:
                if cur_instance_id < 0:
                    continue

                indices = np.where(instance_ids == cur_instance_id)[0]
                cluster = points.select_by_index(indices)

                if len(indices) < 30:
                    continue

                # check if manhole class
                if not isinstance(label_value, (list, tuple)):
                    label_value = [label_value]

                # not_have_manhole = True
                # for cur_label_value in label_value:
                #     not_have_manhole = not_have_manhole and np.all(labels[indices] != cur_label_value)
                
                # if not_have_manhole:
                #     continue
                if not np.any(np.isin(labels[indices], label_value)):
                    continue

                cluster_points = cluster.point[get_coordinate_attribute(cluster)].numpy()

                result = extract_center_point(
                    points=cluster_points,
                    method=method,
                    use_2d_version=False,
                    use_projection=use_projection,
                    apply_downsampling=apply_downsampling
                )

                if result is None:
                    continue

                center = result["center"]
                radius = result["radius"]
                inliers = result["inliers"]
                error = result["error"]
                loss = result["loss"]
                input_points = result["input_points"]

                if 0.20 < radius < 0.50 and (inliers is None or len(inliers) > 30):
                    candidates.append((center, radius, cluster, inliers, error, loss, input_points))
        
            return candidates
        else:
            if semantic_class_idx is not None:
                labels = points.point[semantic_class_idx].numpy().ravel()

                # filter by desired class
                if not isinstance(label_value, (list, tuple)):
                    label_value = [label_value]
                indices = np.where(np.isin(labels, label_value))[0]

                if len(indices) == 0:
                    return []

                filtered_points = points.select_by_index(indices)
            else:
                filtered_points = points

            if not cluster_if_needed:
                return []
            
            points = filtered_points.point[get_coordinate_attribute(filtered_points)].numpy()

            # or use this for clustering:
            # features = np.column_stack((points, intensity))
            db = DBSCAN(eps=0.15, min_samples=20).fit(points)
            cluster_labels = db.labels_

            unique_labels = np.unique(cluster_labels)

            for cur_label in unique_labels:
                if cur_label == -1:
                    continue

                indices = np.where(cluster_labels == cur_label)[0]
                cluster = filtered_points.select_by_index(indices)

                if len(indices) < 30 or len(indices) > 10000:
                    continue

                cluster_points = cluster.point[get_coordinate_attribute(cluster)].numpy()

                result = extract_center_point(
                    points=cluster_points,
                    method=method,
                    use_2d_version=False,
                    use_projection=use_projection,
                    apply_downsampling=apply_downsampling
                )

                if result is None:
                    continue

                center = result["center"]
                radius = result["radius"]
                inliers = result["inliers"]
                error = result["error"]
                loss = result["loss"]
                input_points = result["input_points"]

                if 0.20 < radius < 0.50 and (inliers is None or len(inliers) > 30):
                    candidates.append((center, radius, cluster, inliers, error, loss, input_points))
        
            return candidates



def find_candidates_and_extract_center_point(point_cloud, method="least_square", cluster_method="sklearn", 
                                             extract_ground=False, ground_extraction_method="csf",
                                             use_projection=True, apply_downsampling=False):
    """
    - method: "least_square", "ransac"
    - cluster_method: "sklearn", "o3d"
    - ground_extraction_method: "segmentation", "csf"
    """
    if extract_ground:
        if ground_extraction_method == "segmentation":
            # segment the plane
            plane_model, inliers_indices = point_cloud.segment_plane(
                distance_threshold=0.01,
                ransac_n=3,
                num_iterations=500
            )

            # extract the ground
            point_cloud = point_cloud.select_by_index(inliers_indices)
            # FIXME -> does this work?
        elif ground_extraction_method == "csf":
            point_cloud = CSFGroundFilterTransform(invert_z=False)(point_cloud)
        else:
            raise ValueError(f"Ground Extraction method not found: {ground_extraction_method}")

    # clustering with DBSCAN -> tensor of labels
    if cluster_method == "o3d":
        labels = point_cloud.cluster_dbscan(  # FIXME -> uses every channel, also intesnity?
            eps=0.15, 
            min_points=20, 
            print_progress=False
        ).numpy()
    elif cluster_method == "sklearn":
        points = point_cloud.point[get_coordinate_attribute(point_cloud)].numpy()
        intensity = point_cloud.point[get_intensity_attribute(point_cloud)].numpy().ravel()

        # combine features, to [N, 4]
        features = np.column_stack((points, intensity))

        features = StandardScaler().fit_transform(features)

        # # run meanshift
        # bandwidth = estimate_bandwidth(features, quantile=0.2, n_samples=500)
        # if bandwidth <= 0:
        #     bandwidth = 0.15
        # ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        # ms.fit(features)
        # labels = ms.labels_

        # run dbscan
        db = DBSCAN(eps=0.15, min_samples=20).fit(features)
        labels = db.labels_
        # FIXME, ok?
    else:
        raise ValueError(f"Cluster method not found: {cluster_method}")

    unique_labels = np.unique(labels)

    # get unique labels
    clusters = []
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise
        
        # create a boolean mask for the current cluster
        indices = np.where(labels == label)[0]
        
        # select points belonging to this cluster
        cluster = point_cloud.select_by_index(indices)
        clusters.append(cluster)

    # apply RANSAC
    detected_manhole_candidates = []

    for cluster in clusters:
        points_numpy = cluster.point[get_coordinate_attribute(cluster)].numpy()

        # skip too small point clouds
        xy = points_numpy[:, :2]
        if len(xy) < 30:
            continue

        # FIXME -> also make 2D available?
        result = extract_center_point(points=points_numpy, method=method, use_2d_version=False, use_projection=use_projection, apply_downsampling=apply_downsampling)
        center = result["center"]
        radius = result["radius"]
        inliers = result["inliers"]
        input_points = result["input_points"]

        # Realistic Manhole-Radius: 20–40 cm
        if 0.20 < radius < 0.50 and (inliers is not None or len(inliers) > 30):
            detected_manhole_candidates.append((center, radius, cluster, input_points))
            print("Candidate:", center, radius)

    return detected_manhole_candidates



# FIXME -> make a 3D version with maybe features via Open3D
def classic_manhole_prediction_pipeline(point_cloud, type, plot_path):

    # points = point_cloud.point[get_coordinate_attribute(point_cloud)].cpu().numpy()
    points = point_cloud.to_numpy(as_copy=True).coordinates

    if point_cloud.bev_data is None:
        print("Starting BEV projection...")
        tiles, metas = bev_projection(point_cloud, tile_size=35.0, resolution=0.01)  #  tile_size=100.0/50.0, resolution=0.2/0.1
        bev_gen = bev_gen_wrapper(tiles, metas, has_labels=False if type == "inference" else True)
    else:
        print("Loaded Bevs from file...")
        bev_gen = point_cloud.get_bev()

    center_points = []
    for tile_id, bev_dict in enumerate(bev_gen):
        bev = bev_dict["pixel_values"].cpu().detach().numpy()
        meta = bev_dict["meta"]
        height, width = bev.shape[1], bev.shape[2]
        
        manholes = get_manhole_candidates_from_2d_img(bev)
        manholes_3d = []
        for cur_manhole in manholes:
            # "cluster"
            # "center_px"
            # "diameter_m"
            # "circularity"
            # "axis_ratio"
            cluster = cur_manhole["cluster"]
            # center_px = cur_manhole["center_px"]
            # x = center_px[0]
            # y = center_px[1]

            # or compute circle
            # a, b, r, mean_distance_error, loss = fit_circle_least_squares(x, y)

            # get candidate in 3D
            cluster_3d = []
            for point in cluster:
                # print("Point Shape", point.shape)
                x = point[0]
                y = point[1]
                remapping = bev_back_projection(point_cloud, meta, tile_id, 
                                                pixel_x=x, pixel_y=y, 
                                                try_use_saved_local_points=False)
                point_idx = remapping["global_indices"]

                # empty pixel
                if len(point_idx) == 0:
                    continue

                point_idx = np.array(point_idx).astype(np.int32)

                # access looks like:
                point_3d = points[point_idx]
                cluster_3d.append(point_3d)

            # find center
            # cluster_3d = np.array(cluster_3d, dtype=np.float64)
            if len(cluster_3d) > 5:
                cluster_3d = np.concatenate(cluster_3d, axis=0)
            else:
                continue
                cluster_3d = np.empty((0, 3)) # Handle empty case
            # print("cluster_3d shape:", cluster_3d)
            manholes_3d.append(cluster_3d)  # .squeeze()

        center_points += use_points_and_extract_center_point(clusters=manholes_3d, 
                                                             method="least_square", 
                                                             use_projection=True)
        
    for idx, (center, radius, cluster, inlier, error, loss, input_points) in enumerate(center_points):
        filename = f"2D_Geomtry_Pipeline_Center_Prediction_Tile_{tile_id}_Manhole_{idx}.png"
        visualize_circle_fit(points=cluster, 
                                center_pred=center, 
                                radius=radius, 
                                error=error,
                                should_plot=False,
                                save_path=os.path.join(plot_path, filename))



# --------------------
# > Candidate Search <
# --------------------
def get_manhole_candidates_from_2d_img(bev_image):
    # print("BEV Image Shape:", bev_image.shape)

    intensity_map = bev_image[2, :, :]

    # find local features / edges
    blur = cv2.GaussianBlur(intensity_map, (5,5), 0)
    lap = cv2.Laplacian(
        blur.astype(np.float32),
        cv2.CV_32F
    )
    threshold = np.mean(np.abs(lap)) + 2*np.std(np.abs(lap))
    edges = np.abs(lap) > threshold
    # edges = cv2.Canny(img8, 50, 150)

    # morphological closing
    kernel = np.ones((3,3), np.uint8)

    edges = cv2.morphologyEx(
        edges.astype(np.uint8),
        cv2.MORPH_CLOSE,
        kernel
    )

    candidate_points = np.column_stack(np.where(edges))

    if candidate_points.shape[0] == 0:
        return []

    clustering = DBSCAN(
        eps=3,
        min_samples=20
    ).fit(candidate_points)

    labels = clustering.labels_

    unique_labels = np.unique(labels)

    final_manholes = []

    for label in unique_labels:

        if label == -1:
            continue

        cluster = candidate_points[labels == label]

        center = cluster.mean(axis=0)

        dists = np.linalg.norm(
            cluster - center,
            axis=1
        )

        diameter_px = 2 * dists.max()

        diameter_m = diameter_px * 0.01

        if not (0.4 < diameter_m < 1.2):
            continue

        # circles = cv2.HoughCircles(
        #     bev_image,
        #     cv2.HOUGH_GRADIENT,
        #     dp=1,
        #     minDist=20,
        #     param1=50,
        #     param2=20,
        #     minRadius=5,
        #     maxRadius=20
        # )

        # model, inliers = ransac(
        #     cluster_xy,
        #     CircleModel,
        #     min_samples=3,
        #     residual_threshold=0.02,
        #     max_trials=100
        # )

        if len(cluster) < 20:
            continue

        # Shape Check
        is_circle, shape_check_res = circle_shape_check(points=cluster, save_path=None, should_plot=False, threshold=0.6)

        # Classification Logic
        # Threshold is usually around 0.88 - 0.90
        if is_circle:
            final_manholes.append({
                "cluster": cluster,
                "center_px": center,
                "diameter_m": diameter_m,
                "circularity": shape_check_res["circularity"],
                "pca_score": shape_check_res["pca_score"],
                "radial_var": shape_check_res["radial_var"],
                "least_squares_error": shape_check_res["least_squares_error"],
                "score": shape_check_res["score"]
            })
        else:
            continue

    return final_manholes





