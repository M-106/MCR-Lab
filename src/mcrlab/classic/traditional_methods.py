# -----------
# > Imports <
# -----------
import numpy as np
import cv2

import torch



# --------------
# > 2D METHODS <
# --------------

def method_2d_hough(bev_img, resolution=0.01, origin=(0.0, 0.0)):
    """
    2D Detection via Bilateral Filtering & Hough Circle Transform on Intensity.
    """
    intensity = bev_img[1].copy()
    if intensity.max() <= intensity.min():
        return np.empty((0, 3))

    norm_int = ((intensity - intensity.min()) / (intensity.max() - intensity.min()) * 255).astype(np.uint8)
    filtered = cv2.bilateralFilter(norm_int, d=5, sigmaColor=75, sigmaSpace=75)

    min_r_px = int(0.20 / resolution)
    max_r_px = int(0.60 / resolution)
    min_dist_px = int(0.8 / resolution)

    circles = cv2.HoughCircles(
        filtered, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist_px,
        param1=50, param2=28, minRadius=min_r_px, maxRadius=max_r_px
    )

    if circles is None:
        return np.empty((0, 3))

    results = []
    for px, py, r_px in np.uint16(np.around(circles[0])):
        if 0 <= py < bev_img.shape[1] and 0 <= px < bev_img.shape[2]:
            x_m = origin[0] + px * resolution
            y_m = origin[1] + py * resolution
            results.append([x_m, y_m, r_px * resolution])

    return np.array(results) if len(results) > 0 else np.empty((0, 3))


def method_2d_contours(bev_img, resolution=0.01, origin=(0.0, 0.0)):
    """
    2D Detection via Adaptive Thresholding & Contour Circularity Filtering.
    """
    intensity = bev_img[1].copy()
    height = bev_img[0].copy()

    norm_int = cv2.normalize(intensity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thresh = cv2.adaptiveThreshold(
        norm_int, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -5
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter ** 2))
        (px, py), r_px = cv2.minEnclosingCircle(cnt)
        diameter_m = (2 * r_px) * resolution

        if 0.65 < circularity <= 1.2 and 0.35 <= diameter_m <= 1.2:
            mask = np.zeros_like(height, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            if np.std(height[mask == 255]) < 0.08:  # Height variance check
                x_m = origin[0] + px * resolution
                y_m = origin[1] + py * resolution
                results.append([x_m, y_m, r_px * resolution])

    return np.array(results) if len(results) > 0 else np.empty((0, 3))



# --------------
# > 3D METHODS <
# --------------

def method_3d_ransac_circle(pts_3d, intensity_percentile=85):
    """
    3D Detection via Ground Plane Fitting + 2D RANSAC Circle Optimization.
    """
    if len(pts_3d) < 50:
        return np.empty((0, 3))

    coords = pts_3d[:, :3]
    intensities = pts_3d[:, 3]

    # Select high-intensity ground points
    thresh = np.percentile(intensities, intensity_percentile)
    high_int_pts = coords[intensities >= thresh]

    if len(high_int_pts) < 15:
        return np.empty((0, 3))

    xy_pts = high_int_pts[:, :2]
    best_center, best_r, max_inliers = None, 0, 0
    inlier_thresh = 0.03  # 3cm radius error tolerance

    for _ in range(250):
        idx = np.random.choice(len(xy_pts), 3, replace=False)
        p1, p2, p3 = xy_pts[idx]

        d = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
        if abs(d) < 1e-6:
            continue

        cx = ((p1[0]**2 + p1[1]**2)*(p2[1] - p3[1]) + (p2[0]**2 + p2[1]**2)*(p3[1] - p1[1]) + (p3[0]**2 + p3[1]**2)*(p1[0] - p2[0])) / d
        cy = ((p1[0]**2 + p1[1]**2)*(p3[0] - p2[0]) + (p2[0]**2 + p2[1]**2)*(p1[0] - p3[0]) + (p3[0]**2 + p3[1]**2)*(p2[0] - p1[0])) / d
        r = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)

        if 0.20 <= r <= 0.55:
            dists = np.abs(np.linalg.norm(xy_pts - np.array([cx, cy]), axis=1) - r)
            num_inliers = np.sum(dists < inlier_thresh)

            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_center = [cx, cy]
                best_r = r

    if max_inliers >= 12 and best_center is not None:
        return np.array([[best_center[0], best_center[1], best_r]])

    return np.empty((0, 3))


def method_3d_dbscan_pca(pts_3d, intensity_percentile=80):
    """
    3D Detection via Density Clustering (DBSCAN) + Shape Eigenvalue Filtering (PCA).
    """
    from sklearn.cluster import DBSCAN

    if len(pts_3d) < 50:
        return np.empty((0, 3))

    coords = pts_3d[:, :3]
    intensities = pts_3d[:, 3]

    high_int_mask = intensities > np.percentile(intensities, intensity_percentile)
    filtered_pts = coords[high_int_mask]

    if len(filtered_pts) < 15:
        return np.empty((0, 3))

    clustering = DBSCAN(eps=0.15, min_samples=10).fit(filtered_pts[:, :2])
    labels = clustering.labels_
    results = []

    for label in set(labels):
        if label == -1:
            continue

        cluster = filtered_pts[labels == label]
        if len(cluster) < 15:
            continue

        xy = cluster[:, :2]
        center = np.mean(xy, axis=0)

        # Evaluate symmetry using covariance eigenvalues
        cov = np.cov(xy.T)
        eigenvalues = np.sort(np.linalg.eigvals(cov))[::-1]
        axis_ratio = eigenvalues[0] / (eigenvalues[1] + 1e-8)

        radii = np.linalg.norm(xy - center, axis=1)
        mean_radius = np.mean(radii)

        if axis_ratio < 1.6 and 0.20 <= mean_radius <= 0.55:
            results.append([center[0], center[1], mean_radius])

    return np.array(results) if len(results) > 0 else np.empty((0, 3))



# ---------
# > Usage <
# ---------

METHOD_REGISTRY = {
    "2d_hough": method_2d_hough,
    "2d_contours": method_2d_contours,
    "3d_ransac": method_3d_ransac_circle,
    "3d_dbscan": method_3d_dbscan_pca
}

def predict_manhole_centers(
    pts_3d=None, 
    bev_img=None, 
    method="2d_hough", 
    meta=None
):
    """
    Predicts manhole centers directly from PyTorch Tensors or NumPy arrays.

    Parameters:
        pts_3d (torch.Tensor | np.ndarray, optional): Tensor of shape (N, 4) -> [x, y, z, intensity]
        bev_img (torch.Tensor | np.ndarray, optional): Tensor of shape (3, H, W) or (1, 3, H, W)
        method (str): Key matching METHOD_REGISTRY ('2d_hough', '2d_contours', '3d_ransac', '3d_dbscan', etc.)
        meta (dict, optional): Dict containing 'resolution', 'origin_x', 'origin_y'

    Returns:
        np.ndarray (M, 3): Array of detected centers [[x, y, z], ...] in 3D, or [[px, py, radius], ...] in 2D
    """
    if method not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method '{method}'. Available options: {list(METHOD_REGISTRY.keys())}")

    if meta is None:
        meta = {"resolution": 0.01, "origin_x": 0.0, "origin_y": 0.0}

    # Handle 2D methods
    if method.startswith("2d"):
        if bev_img is None:
            print(f"[Warning] Method '{method}' requires `bev_img`, but None was provided.")
            return np.empty((0, 3))

        # Tensor to NumPy conversion
        if isinstance(bev_img, torch.Tensor):
            bev_img_np = bev_img.detach().cpu().numpy()
        else:
            bev_img_np = np.array(bev_img)

        # Squeeze batch dimension if present: (1, C, H, W) -> (C, H, W)
        if bev_img_np.ndim == 4 and bev_img_np.shape[0] == 1:
            bev_img_np = bev_img_np.squeeze(0)

        origin = (meta.get("origin_x", 0.0), meta.get("origin_y", 0.0))
        res = meta.get("resolution", 0.01)

        return METHOD_REGISTRY[method](bev_img_np, resolution=res, origin=origin)

    # Handle 3D methods
    elif method.startswith("3d"):
        if pts_3d is None:
            print(f"[Warning] Method '{method}' requires `pts_3d`, but None was provided.")
            return np.empty((0, 3))

        # Handle tuple output from get_patch_via_identifier: (pts, labels)
        if isinstance(pts_3d, (tuple, list)):
            pts_3d = pts_3d[0]

        # Tensor to NumPy conversion
        if isinstance(pts_3d, torch.Tensor):
            pts_3d_np = pts_3d.detach().cpu().numpy()
        else:
            pts_3d_np = np.array(pts_3d)

        return METHOD_REGISTRY[method](pts_3d_np)

    else:
        raise ValueError(f"Method prefix must start with '2d' or '3d', got '{method}'")




















