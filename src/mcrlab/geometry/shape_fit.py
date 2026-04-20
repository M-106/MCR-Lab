# ----------
# > Import <
# ----------
import numpy as np
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from skimage.measure import ransac
import pyransac3d as pyrsc

from mcrlab.geometry.utils import fit_plane, plane_basis, project_to_plane
from mcrlab.point_cloud.utils import get_coordinate_attribute, get_intensity_attribute, get_class_attribute
from mcrlab.point_cloud.data import CSFGroundFilterTransform



# ----------------------------
# > Least Square Fit 2D & 3D <
# ----------------------------
# Short explanation: Iteratively minimize residuals -> perfect circle compared to current circle.
#                    Optimized via gradients (the direction of where we want to go to complete the circle function), 
#                    we have the ground truth, it is a perfect circle.
#                    We don't know where the center is, but we improve the circle via gradients
#                    so that the circle function gets minimized/approximal optimized.
#                    - It minimized a nonlinear function
#                    - uses gradients (kinda like backpropagation but a bit different)
# Cite: FIXME

def circle_residuals(params, x, y):
    """
    Computes distance error for each point.

    > Residual is that what is left after a difference or the rest part.
    > Here it is the rest part which is not optimal for a perfect circle.

    Center: a, b
    Radius: r
    """
    a, b, r = params

    distances = np.sqrt( (x - a)**2 + (y - b)**2 )

    return distances - r



def fit_circle_least_squares(x, y):
    """
    Fits a circle to 2D points using nonlinear least squares.

    Iteratively minimize circle function and optimize via back-propagation.
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError(f"Points must be a numpy array, but got '{type(x)}, {type(y)}'")

    # init guess
    a0 = np.mean(x)
    b0 = np.mean(y)
    r0 = np.mean( np.sqrt((x - a0)**2 + (y - b0)**2) )
    init_guess = [a0, b0, r0]

    # optimize
    result = least_squares(
        circle_residuals,
        init_guess,
        args=(x, y),
        method='lm',  # Levenberg-Marquardt (should be good for small/medium problems)
        loss='linear'
    )

    a, b, r = result.x
    # result.fun contains the residuals of every point, with the current optimized params
    mean_distance_error = np.mean(np.abs(result.fun))
    loss = result.cost
    return a, b, abs(r), mean_distance_error, loss



def fit_circle_least_squares_3D(points):
    if not isinstance(points, np.ndarray):
        raise TypeError(f"Points must be a numpy array, but got '{type(points)}'")

    # prepraration for projection
    centroid, normal = fit_plane(points)
    basis_x, basis_y = plane_basis(normal)

    # projection into 2D
    x, y = project_to_plane(centroid=centroid, points=points, basis_x=basis_x, basis_y=basis_y)

    # optimize circle shape
    a, b, r, mean_distance_error, loss = fit_circle_least_squares(x, y)

    # back projection
    center_3D = centroid + (a * basis_x) + (b *basis_y)

    return center_3D, normal, r, mean_distance_error, loss



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

class CircleModel:
    """
    SkLearn/SkImage Model.

    Least Square Fit Model for RANSAC.

    Fits a circle to 2D points: (x - cx)^2 + (y - cy)^2 = r^2
    """
    def estimate(self, data):
        """
        Fit circle to minimal sample (3 points) using algebraic method.
        """
        x, y = data[:, 0], data[:, 1]

        # Check for collinearity (area of triangle ~ 0)
        mat = np.column_stack((x, y, np.ones(3)))
        if abs(np.linalg.det(mat)) < 1e-6:
            return False

        # Build linear system: 2x*cx + 2y*cy + r^2 - cx^2 - cy^2 = x^2 + y^2
        A = np.column_stack((2 * x, 2 * y, np.ones(len(x))))
        b = x**2 + y**2

        try:
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            # This is also LEast Square Fit BUT:
            # - solves a linear system
            # - one shot (no iterations)
            # - no gradients
            # - no "learning"
        except np.linalg.LinAlgError:
            return False
        cx, cy, c = result
        r_sq = c + cx**2 + cy**2
        if r_sq <= 0:
            return False
        self.params = (cx, cy, np.sqrt(r_sq))
        return True

    def residuals(self, data):
        """
        Euclidean distance from each point to the circle boundary.
        """
        cx, cy, r = self.params
        x, y = data[:, 0], data[:, 1]
        return np.abs( np.sqrt( (x - cx)**2 + (y - cy)**2 ) - r )

    def predict_xy(self, t):
        cx, cy, r = self.params
        return np.column_stack((cx + r * np.cos(t), cy + r * np.sin(t)))



def fit_circle_ransac(x, y, method="sklearn"):
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError(f"Points must be a numpy array, but got '{type(x)}, {type(y)}'")
    
    if method == "sklearn":
        data = np.column_stack((x, y))

        model, inliers = ransac(
            data,
            CircleModel,
            min_samples=3,
            residual_threshold=0.01 * np.std(data),  # 1 cm tolerance (same as before) + dynamic thresholding
            max_trials=1000,
        )

        if model is None:
            raise RuntimeError("RANSAC failed to find a valid circle")
        
        # refinment using all inliers:
        inlier_points = data[inliers]
        model.estimate(inlier_points)

        cx, cy, radius = model.params
        center = np.array([cx, cy])
        axis = np.array([0, 0, 1])  # Normal axis (z), matches 3D RANSAC convention
    elif method == "pyransac3d":
        points_3d = np.column_stack((x, y, np.zeros_like(x)))
        circle = pyrsc.Circle()
        center, axis, radius, inliers = circle.fit(points_3d, thresh=0.01)  # 5 cm tolerance -> maybe increase if not working because of scanline-artefacts (gaps)
        center = center[:2]
    else:
        raise ValueError(f"Got unknown method: '{method}'")
    
    # FIXME -> or own implementation? First get others work

    return center, axis, radius, inliers



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
        center, axis, radius, inliers = fit_circle_ransac(x, y)
        axis = normal

        # back projection
        center = centroid + (center[0] * basis_x) + (center[1] * basis_y)
    else:
        circle = pyrsc.Circle()
        center, axis, radius, inliers = circle.fit(points, thresh=0.01)

    return center, axis, radius, inliers



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
def extract_center_point(FIXME):
    pass



def use_label_candidates_and_extract_center_point(FIXME):
    pass



def find_candidates_and_extract_center_point(point_cloud, method="least_square", cluster_method="sklearn", extract_ground=False, ground_extraction_method="csf"):
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

        # normalization
        intensity_scaled = intensity * 0.1 

        # combine features, to [N, 4]
        features = np.column_stack((points, intensity_scaled))

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
        indices = np.where(labels.numpy() == label)[0]
        
        # select points belonging to this cluster
        cluster = point_cloud.select_by_index(indices)
        clusters.append(cluster)

    # apply RANSAC
    detected_manhole_candidates = []

    for cluster in clusters:
        points_numpy = point_cloud.point[get_coordinate_attribute(point_cloud)].numpy()

        # skip too small point clouds
        xy = points_numpy[:, :2]
        if len(xy) < 30:
            continue

        result = extract_center_point()
        # center, radius, inliers = fit_circle_ransac_3D(points_numpy)

        # Realistic Manhole-Radius: 20–40 cm
        if 0.20 < radius < 0.50 and len(inliers) > 30:
            detected_manhole_candidates.append((center, radius, cluster))
            print("Candidate:", center, radius)

    return detected_manhole_candidates








