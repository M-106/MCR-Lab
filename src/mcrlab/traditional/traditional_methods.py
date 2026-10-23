# -----------
# > Imports <
# -----------
import numpy as np
import cv2

import torch



# --------------
# > 2D METHODS <
# --------------
def suppress_overlapping_circles(circles, intensity_img, resolution, overlap_thresh=0.4):
    """
    Applies Non-Maximum Suppression to a list of detected circles [x_m, y_m, r_m].
    """
    if len(circles) == 0:
        return np.empty((0, 3))

    circles = np.array(circles)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    # Score each circle by measuring the mean intensity inside its boundary
    scores = []
    for cx, cy, cr in circles:
        # Convert meters back to pixels for local masking
        px = int((cx) / resolution)
        py = int((cy) / resolution)
        pr = int(cr / resolution)
        
        # Simple score: average pixel intensity within the circle region
        mask = np.zeros_like(intensity_img, dtype=np.uint8)
        cv2.circle(mask, (px, py), pr, 255, -1)
        mean_val = np.mean(intensity_img[mask == 255]) if np.any(mask == 255) else 0
        scores.append(mean_val)

    scores = np.array(scores)
    
    # Sort circles by score in descending order
    idxs = np.argsort(scores)[::-1]
    keep = []

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)

        if len(idxs) == 1:
            break

        # Calculate distances between the current best circle and the rest
        current_center = circles[current, :2]
        other_centers = circles[idxs[1:], :2]
        distances = np.linalg.norm(other_centers - current_center, axis=1)

        # Suppress circles whose centers are closer than the overlap threshold (e.g., 40% of radius)
        radius_sum = circles[current, 2] + circles[idxs[1:], 2]
        overlap_mask = distances < (overlap_thresh * radius_sum)

        # Keep only indices that are NOT heavily overlapping
        idxs = idxs[1:][~overlap_mask]

    return circles[keep]


def method_2d_hough(bev_img, resolution=0.01, origin=(0.0, 0.0)):
    """
    2D Detection via Bilateral Filtering & Hough Circle Transform on Intensity.

    Hough Circles does detect edges via canny and then Hough Transform votes for center points.
    If a circle gets enough votes, it is considers a hit.

    In Hough Transform every edge point votes for potential center points. 
    Hough Transform uses for every edge pixel the direction of the strongest brightness increase (a gradient) 
    and this gradient shows in direction of the center-point. Sometimes the center also can be on the opposite direction,
    when the object is brighter than the background and because of this, OpenCV creates votes for both directions. 
    The edge point votes now only for all points in the min-max radius in the gradient direction (both directions). 
    Every edge point does that and in the end we get a vote-map which the accumulator uses.

    > The gradients reduces the search-area massively.

    Image
     ↓
    Gaussian Blur
     ↓
    Canny Edge Detector
     ↓
    Gradient Direction (via edge direction)
     ↓
    Hough Voting
     ↓
    Circles
    """
    intensity = bev_img[1].copy()
    if intensity.max() <= intensity.min():
        return (np.empty((0, 3)), None)

    # Normalize to 0-255 uint8
    norm_int = cv2.normalize(intensity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # norm_int = ((intensity - intensity.min()) / (intensity.max() - intensity.min()) * 255).astype(np.uint8)
    # filtered = cv2.bilateralFilter(norm_int, d=5, sigmaColor=75, sigmaSpace=75)

    # Use Median Blur to remove speckle noise on asphalt
    blurred = cv2.medianBlur(norm_int, 5)
    # blurred = cv2.GaussianBlur(norm_int, (3, 3), 1.0)

    min_r_px = int(0.25 / resolution)  # 25 px for 0.5m diameter
    max_r_px = int(0.45 / resolution)  # 45 px for 0.9m diameter
    min_dist_px = int(1.0 / resolution)  # distance from circles

    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.0,  # 1.5,   # decides resolution of accumulator relative to current resolution
        minDist=min_dist_px,
        param1=60,  # canny edge value threshold -> how strong the edge have to be, as gradient value not pixel value -> maybe 1020 max
        param2=20,  # threshold of voting for hit
        minRadius=min_r_px, 
        maxRadius=max_r_px
    )

    if circles is None:
        return (np.empty((0, 3)), None)

    # Collect Debug Info
    debug_dict = None
    # Compute canny for visualization -> hough transform use it
    edges = cv2.Canny(blurred, 20, 20) 
    
    extra_info = []
    if circles is not None:
        for px, py, r_px in np.uint16(np.around(circles[0])):
            extra_info.append({
                'type': 'circle',
                'x': px,
                'y': py,
                'r': r_px
            })
    
    debug_dict = {
        'prob_map': blurred,  # Panel 2 shows the graues/verschwommenes Bild
        'binary_mask': edges, # Panel 3 shows the Canny-Edges
        'geometries': extra_info
    }

    # Compute Final Results
    results = []
    for px, py, r_px in np.uint16(np.around(circles[0])):
        if 0 <= py < bev_img.shape[1] and 0 <= px < bev_img.shape[2]:
            # --- SCENARIO A: Standard Image Mapping ---
            # Use this if World-X goes RIGHT and World-Y goes DOWN
            # x_m = origin[0] + (px * resolution)
            # y_m = origin[1] + (py * resolution)

            # --- SCENARIO B: Typical Automotive BEV Mapping ---
            # Use this if World-X goes FORWARD (vertical image axis) 
            # and World-Y goes LATERAL (horizontal image axis)
            # x_m = origin[0] + (py * resolution) 
            # y_m = origin[1] + (px * resolution)

            # results.append([x_m, y_m, r_px * resolution])

            # should return pixel coordinate system not meters
            results.append([px, py, r_px])

    # Apply Maximum Suppression to clean up redundant overlapping circles
    filtered_results = suppress_overlapping_circles(results, intensity, resolution)

    return (np.array(filtered_results), debug_dict) if len(filtered_results) > 0 else (np.empty((0, 3)), debug_dict)



def method_2d_contours(bev_img, resolution=0.01, origin=(0.0, 0.0)):
    """
    2D Detection via Adaptive Thresholding & Contour Circularity Filtering.
    """
    intensity = bev_img[1].copy()
    height = bev_img[0].copy()

    norm_int = cv2.normalize(intensity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # thresh = cv2.adaptiveThreshold(
    #     norm_int, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -5
    # )
    # Otsu's thresholding for global high-contrast manholes handling
    _, thresh = cv2.threshold(norm_int, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # (5, 5)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

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

        if 0.5 < circularity <= 1.3 and 0.3 <= diameter_m <= 1.1:
            # mask = np.zeros_like(height, dtype=np.uint8)
            # cv2.drawContours(mask, [cnt], -1, 255, -1)
            # if np.std(height[mask == 255]) < 0.08:  # Height variance check
            results.append([px, py])

    return np.array(results) if len(results) > 0 else np.empty((0, 2))

def _fit_circle_3pts(p1, p2, p3):
    """
    Fits a circle through 3 distinct 2D points (x, y).
    Returns (cx, cy, r) or None if collinear / degenerate.
    """
    # Determinant of the perpendicular bisector linear system
    d = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    if abs(d) < 1e-6:
        return None

    p1_sq = p1[0] ** 2 + p1[1] ** 2
    p2_sq = p2[0] ** 2 + p2[1] ** 2
    p3_sq = p3[0] ** 2 + p3[1] ** 2

    cx = (p1_sq * (p2[1] - p3[1]) + p2_sq * (p3[1] - p1[1]) + p3_sq * (p1[1] - p2[1])) / d
    cy = (p1_sq * (p3[0] - p2[0]) + p2_sq * (p1[0] - p3[0]) + p3_sq * (p2[0] - p1[0])) / d
    r = np.hypot(p1[0] - cx, p1[1] - cy)

    return cx, cy, r


def _ransac_single_circle(
    points,
    min_r,
    max_r,
    distance_threshold=1.5,
    max_iterations=1200,
    min_inliers=35,
):
    """
    Finds the single best circle in a 2D point cloud (N, 2) using RANSAC.
    Returns (cx, cy, r), inlier_mask.
    """
    n_points = len(points)
    if n_points < 3:
        return None, None

    best_inlier_count = 0
    best_circle = None
    best_inliers = None

    for _ in range(max_iterations):
        # Sample 3 random points
        idx = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[idx]

        # Reject degenerate/collinear samples early if pairwise distances are too small
        if np.linalg.norm(p1 - p2) < 2.0 or np.linalg.norm(p2 - p3) < 2.0:
            continue

        model = _fit_circle_3pts(p1, p2, p3)
        if model is None:
            continue

        cx, cy, r = model
        if not (min_r <= r <= max_r):
            continue

        # Distance from each edge point to circle perimeter: |sqrt((x-cx)^2 + (y-cy)^2) - r|
        distances = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - r)
        inlier_mask = distances < distance_threshold
        inlier_count = np.count_nonzero(inlier_mask)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_circle = (cx, cy, r)
            best_inliers = inlier_mask

    if best_inlier_count < min_inliers:
        return None, None

    # Refine circle fit using all inliers via Kasa least-squares algebraic circle fit
    inlier_pts = points[best_inliers]
    x, y = inlier_pts[:, 0], inlier_pts[:, 1]
    a_mat = np.column_stack([x, y, np.ones_like(x)])
    b_vec = x**2 + y**2
    try:
        sol, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        refined_cx = sol[0] / 2.0
        refined_cy = sol[1] / 2.0
        rad_sq = sol[2] + refined_cx**2 + refined_cy**2
        if rad_sq > 0:
            refined_r = np.sqrt(rad_sq)
            if min_r <= refined_r <= max_r:
                best_circle = (refined_cx, refined_cy, refined_r)
    except np.linalg.LinAlgError:
        pass

    return best_circle, best_inliers



def method_2d_ransac(
    bev_img,
    resolution=0.01,
    origin=(0.0, 0.0),
    max_circles=5,
    ransac_iters=1200,
    dist_thresh_px=1.5,
    min_inlier_ratio=0.35,
):
    """
    2D Detection via Canny Edge Extraction & Sequential Circle RANSAC.

    Pipeline:
      1. Normalize & blur intensity image.
      2. Canny edge detector to extract manhole edge candidates.
      3. Sequential RANSAC loop:
         - Fit 3-point circle candidate within [min_radius, max_radius].
         - Count inliers within distance tolerance.
         - Least-squares refinement over candidate inliers.
         - Suppress detected inliers and repeat for next circle.
      4. Convert pixel coordinates (px, py, r_px) to metric space (x_m, y_m, r_m).
    """
    intensity = bev_img[1].copy()
    if intensity.max() <= intensity.min():
        return np.empty((0, 3))

    # Normalize to 0-255 uint8
    norm_int = cv2.normalize(intensity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred = cv2.GaussianBlur(norm_int, (3, 3), 1.0)

    # Edge detection
    edges = cv2.Canny(blurred, threshold1=60, threshold2=120)

    # Extract edge coordinates: Canny yields (row, col) = (y, x)
    edge_y, edge_x = np.nonzero(edges)
    if len(edge_x) < 30:
        return np.empty((0, 3))

    # Working set of points in (x, y) format
    pts = np.column_stack([edge_x, edge_y]).astype(np.float64)

    min_r_px = 0.25 / resolution   # 25 px for 0.5m diameter
    max_r_px = 0.45 / resolution   # 45 px for 0.9m diameter
    min_dist_px = 1.0 / resolution # Min distance separation between circles

    # Expected circumference in pixels at median radius: 2 * pi * r
    # Minimum required inlier points to avoid false-positive noise loops
    min_inliers = int(2.0 * np.pi * min_r_px * min_inlier_ratio)

    detected_circles = []

    for _ in range(max_circles):
        if len(pts) < min_inliers:
            break

        circle, inlier_mask = _ransac_single_circle(
            pts,
            min_r=min_r_px,
            max_r=max_r_px,
            distance_threshold=dist_thresh_px,
            max_iterations=ransac_iters,
            min_inliers=min_inliers,
        )

        if circle is None:
            break

        cx, cy, r = circle

        # Check minimum center-to-center distance against already found circles
        too_close = False
        for ex_cx, ex_cy, _ in detected_circles:
            if np.hypot(cx - ex_cx, cy - ex_cy) < min_dist_px:
                too_close = True
                break

        if not too_close:
            detected_circles.append((cx, cy, r))

        # Remove inliers and nearby points so subsequent iterations don't re-detect this circle
        remaining_distances = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
        pts = pts[remaining_distances >= (dist_thresh_px * 2.0)]

    if not detected_circles:
        return np.empty((0, 3))

    # Convert back to ground plane metric coordinates
    results = []
    h, w = bev_img.shape[1], bev_img.shape[2]
    for px, py, r_px in detected_circles:
        if 0 <= py < h and 0 <= px < w:
            results.append([px, py])

    return np.array(results) if len(results) > 0 else np.empty((0, 2))


def _generate_ring_template(radius_px, thickness_px=3):
    """
    Creates a synthetic binary/float edge template of a circular ring.
    The circle boundary is set to 1.0 and background to 0.0.
    """
    pad = int(thickness_px + 2)
    side = int(2 * (radius_px + pad) + 1)
    center = (side // 2, side // 2)

    template = np.zeros((side, side), dtype=np.float32)
    cv2.circle(
        template,
        center,
        int(round(radius_px)),
        color=1.0,
        thickness=int(thickness_px),
    )
    return template


def _non_max_suppression_circles(candidates, min_dist_px):
    """
    Filters dense multi-scale detections using greedy 2D Non-Maximum Suppression.
    Candidates: list of (x, y, r, score)
    """
    if len(candidates) == 0:
        return []

    # Sort descending by correlation score
    candidates = sorted(candidates, key=lambda c: c[3], reverse=True)
    kept = []

    for cand in candidates:
        cx, cy, r, score = cand
        overlap = False
        for k_cx, k_cy, _, _ in kept:
            dist = np.hypot(cx - k_cx, cy - k_cy)
            if dist < min_dist_px:
                overlap = True
                break

        if not overlap:
            kept.append(cand)

    return kept


def method_2d_template_matching(
    bev_img,
    resolution=0.01,
    match_threshold=0.35,
    radius_step_m=0.04,
    ring_thickness_m=0.04,
):
    """
    2D Detection via Multi-Scale Edge Ring Template Matching and Non-Max Suppression.

    Parameters:
      - bev_img: Array where bev_img[1] is the intensity channel.
      - resolution: Spatial resolution in m/pixel (default: 0.01 m/px).
      - origin: (x, y) real-world offset of pixel (0, 0).
      - match_threshold: Min TM_CCOEFF_NORMED score [0.0 - 1.0] to consider a candidate.
      - radius_step_m: Step size to scale circular templates between min/max radius.
      - ring_thickness_m: Physical thickness of the synthetic ring border.

    Returns:
      - np.ndarray of shape (N, 3) containing [x_m, y_m, r_m].
    """
    intensity = bev_img[1].copy()
    if intensity.max() <= intensity.min():
        return np.empty((0, 3))

    # Normalize intensity to uint8
    norm_int = cv2.normalize(intensity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred = cv2.GaussianBlur(norm_int, (3, 3), 1.0)

    # Edge map of the scene
    # Matching edge-to-edge avoids lighting variation issues on road surfaces
    edges = cv2.Canny(blurred, threshold1=50, threshold2=120).astype(np.float32) / 255.0

    min_r_px = int(0.25 / resolution)  # ~25 px for 0.5m diameter
    max_r_px = int(0.45 / resolution)  # ~45 px for 0.9m diameter
    step_px = max(1, int(radius_step_m / resolution))
    thickness_px = max(1, int(ring_thickness_m / resolution))
    min_dist_px = 1.0 / resolution      # 1 meter clearance between centers

    candidates = []

    # Iterate over multiple potential radii
    for r_px in range(min_r_px, max_r_px + 1, step_px):
        tpl = _generate_ring_template(r_px, thickness_px=thickness_px)
        th, tw = tpl.shape

        if edges.shape[0] < th or edges.shape[1] < tw:
            continue

        # TM_CCOEFF_NORMED outputs scores in [-1.0, 1.0]
        corr_map = cv2.matchTemplate(edges, tpl, cv2.TM_CCOEFF_NORMED)

        # Threshold to gather raw local peaks
        match_ys, match_xs = np.where(corr_map >= match_threshold)

        for top_left_x, top_left_y in zip(match_xs, match_ys):
            score = float(corr_map[top_left_y, top_left_x])
            # Shift top-left template position to true circle center
            center_x = top_left_x + tw / 2.0
            center_y = top_left_y + th / 2.0
            candidates.append((center_x, center_y, float(r_px), score))

    if not candidates:
        return []

    # Suppress overlapping hits across different scales and nearby locations
    filtered_circles = _non_max_suppression_circles(candidates, min_dist_px=min_dist_px)

    # Map back to real-world ground coordinates
    results = []
    h, w = bev_img.shape[1], bev_img.shape[2]

    for cx_px, cy_px, r_px, _ in filtered_circles:
        if 0 <= cy_px < h and 0 <= cx_px < w:
            results.append([cx_px, cy_px])

    return np.array(results) if len(results) > 0 else []




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
    "2d_template": method_2d_template_matching,
    "2d_ransac": method_2d_ransac,
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

        return METHOD_REGISTRY[method](bev_img_np, origin=origin, resolution=res)

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




















