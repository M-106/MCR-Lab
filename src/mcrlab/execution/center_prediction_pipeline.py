# -----------
# > Imports <
# -----------
import os
import shutil
import json
from pathlib import Path
import time
import math

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.ndimage import label, binary_closing, center_of_mass, gaussian_filter, distance_transform_edt
from scipy.ndimage.morphology import generate_binary_structure # or skimage.morphology?
from scipy.optimize import least_squares
from scipy.signal import correlate2d
from skimage.feature import peak_local_max
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.measure import EllipseModel
from sklearn.cluster import DBSCAN

from tqdm import tqdm

from mcrlab.execution.train import get_model_and_processor
from mcrlab.execution.test import predict_single_sample
from mcrlab.traditional.least_squares import fit_circle_least_squares
from mcrlab.point_cloud.shape_check import circle_shape_check
from mcrlab.point_cloud.data import get_data_loader, get_basic_transform, BEVDataset
from mcrlab.projection import bev_projection, bev_pixel_to_3d
from mcrlab.metrices import mask_to_polygon
from mcrlab.traditional.traditional_methods import predict_manhole_centers
from mcrlab.point_cloud.utils import get_coordinate_attribute, get_intensity_attribute, \
                                     get_class_attribute



# --------------------------------------------
# > Center Localization/Extraction Functions <
# --------------------------------------------
def extract_center_polygon_centroid(pred_mask, **kwargs):
    """
    Extracts the center using the geometric centroid of the polygon mask.
    """
    mask_to_polygon_fn = kwargs.get("mask_to_polygon_fn")
    if mask_to_polygon_fn is None:
        raise ValueError("'mask_to_polygon_fn' must be provided in kwargs.")
        # return None

    pred_poly = mask_to_polygon_fn(pred_mask)

    if pred_poly is None or pred_poly.is_empty:
        return None

    # if pred_poly is None:
    #     return None

    # row = y, col = x in BEV
    # return pred_poly.centroid.x, pred_poly.centroid.y

    centroid = pred_poly.centroid
    return float(centroid.x), float(centroid.y)



def extract_center_circle_least_squares(pred_mask, **kwargs):
    """
    Extracts the center by fitting a circle via least squares on the polygon exterior.

    Gets one potential manhole points as input (not the whole patch).
    """
    mask_to_polygon_fn = kwargs.get("mask_to_polygon_fn")
    fit_circle_fn = fit_circle_least_squares

    if mask_to_polygon_fn is None or fit_circle_fn is None:
        raise ValueError("Both 'mask_to_polygon_fn' and 'fit_circle_fn' must be provided in kwargs.")
    
    pred_poly = mask_to_polygon_fn(pred_mask)
    if pred_poly is None or pred_poly.is_empty:
        return None
    
    polys_to_plot = pred_poly.geoms if hasattr(pred_poly, 'geoms') else [pred_poly]
    cur_manhole_pred_x, cur_manhole_pred_y = [], []
    
    for p in polys_to_plot:
        if p.is_empty or p.exterior is None:
            continue
        x, y = p.exterior.xy
        cur_manhole_pred_x.extend(x)
        cur_manhole_pred_y.extend(y)

    # filter FIXME
    if len(cur_manhole_pred_x) < 20:
        return None
        
    cur_manhole_pred_x = np.array(cur_manhole_pred_x)
    cur_manhole_pred_y = np.array(cur_manhole_pred_y)
    
    fit_fn = fit_circle_least_squares
    center_x, center_y, _, _, _ = fit_fn(cur_manhole_pred_x, cur_manhole_pred_y)
    return center_x, center_y




def extract_center_least_square_radius(pred_mask, **kwargs):
    """
    Finds the center by using ONLY valid boundary points (not touching the image edge) 
    and fitting them to a known, fixed physical radius.
    """
    mask_to_polygon_fn = kwargs.get("mask_to_polygon_fn")
    
    # 1. Konvertiere deinen echten Radius in Pixel-Radius!
    # z.B. 0.3m bei einer Auflösung von 0.02m/pixel = 15 Pixel
    known_radius_px = kwargs.get("known_radius_px") 
    
    if known_radius_px is None or mask_to_polygon_fn is None:
        raise ValueError("'known_radius_px' and 'mask_to_polygon_fn' are required.")

    h, w = pred_mask.shape
    pred_poly = mask_to_polygon_fn(pred_mask)
    if pred_poly is None or pred_poly.is_empty:
        return None

    polys_to_plot = pred_poly.geoms if hasattr(pred_poly, 'geoms') else [pred_poly]
    valid_x, valid_y = [], []

    # 2. Randpunkte extrahieren und filtern
    for p in polys_to_plot:
        if p.is_empty or p.exterior is None:
            continue
        x_coords, y_coords = p.exterior.xy
        
        for x, y in zip(x_coords, y_coords):
            # Filtere Punkte heraus, die exakt auf dem Bildrand liegen!
            # Toleranz von ~1 Pixel ist meistens gut wegen Rundungen bei der Polygon-Erstellung
            if x <= 1 or x >= w - 2 or y <= 1 or y >= h - 2:
                continue # Dies ist ein künstlicher, abgeschnittener Rand
                
            valid_x.append(x)
            valid_y.append(y)

    # Wir brauchen mindestens 3 gültige Randpunkte für einen stabilen Fit
    if len(valid_x) < 3:
        return None

    valid_x = np.array(valid_x)
    valid_y = np.array(valid_y)

    # 3. Least Squares mit FIXEM Radius
    def calc_distances(params, x, y, r):
        cx, cy = params
        return np.sqrt((x - cx)**2 + (y - cy)**2) - r

    # Initialer Schätzwert (Mittelwert der gültigen Randpunkte)
    initial_guess = [np.mean(valid_x), np.mean(valid_y)]
    
    # Optimiere NUR das Zentrum, der Radius ist fix
    result = least_squares(calc_distances, initial_guess, args=(valid_x, valid_y, known_radius_px))
    
    return float(result.x[0]), float(result.x[1])



def refine_center_least_squares(x, y, initial_cx, initial_cy, fixed_r):
    """
    Refines the center using non-linear least squares, keeping the radius tightly bounded.
    """
    def calc_distances(params, x, y, fixed_r):
        cx, cy = params
        # Calculate the geometric distance from points to the circle perimeter
        return np.sqrt((x - cx)**2 + (y - cy)**2) - fixed_r

    # Initial guess from Hough
    initial_guess = [initial_cx, initial_cy]
    
    # Run optimization (Levenberg-Marquardt)
    result = least_squares(calc_distances, initial_guess, args=(x, y, fixed_r))
    
    return result.x[0], result.x[1]

def extract_center_hough_least_squares(pred_mask, **kwargs):
    mask_to_polygon_fn = kwargs.get("mask_to_polygon_fn")
    
    # You MUST provide the expected radius of the manhole in pixels
    # You can provide an array of a few close radii if it varies slightly, e.g., [18, 19, 20, 21, 22]
    expected_radii = kwargs.get("expected_radii", np.arange(18, 23)) 

    pred_poly = mask_to_polygon_fn(pred_mask)
    if pred_poly is None or pred_poly.is_empty:
        return None

    polys_to_plot = pred_poly.geoms if hasattr(pred_poly, 'geoms') else [pred_poly]
    x_coords, y_coords = [], []

    for p in polys_to_plot:
        if p.is_empty or p.exterior is None:
            continue
        x, y = p.exterior.xy
        x_coords.extend(x)
        y_coords.extend(y)

    if len(x_coords) < 5: # Even 5 points is enough for Hough!
        return None

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    # 1. HOUGH TRANSFORM
    # To use skimage Hough, we need an image grid. 
    # Create a blank image covering the bounds of our polygon.
    min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
    min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
    
    # Add padding so the center can be found even if it's far outside the 10% arc bounding box
    pad = int(np.max(expected_radii)) + 5 
    img_shape = (max_y - min_y + 2*pad, max_x - min_x + 2*pad)
    grid = np.zeros(img_shape, dtype=np.uint8)
    
    # Map points to the local grid
    local_x = np.round(x_coords - min_x + pad).astype(int)
    local_y = np.round(y_coords - min_y + pad).astype(int)
    grid[local_y, local_x] = 1

    # Accumulate votes
    hough_res = hough_circle(grid, expected_radii)
    
    # Find the single strongest peak (highest vote count)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, expected_radii, total_num_peaks=1)
    
    if len(cx) == 0:
        return None
        
    # Convert local grid coordinates back to global coordinates
    global_cx = cx[0] + min_x - pad
    global_cy = cy[0] + min_y - pad
    best_radius = radii[0]

    # 2. LEAST SQUARES REFINEMENT
    # Now that we have a highly robust estimate, refine it precisely to sub-pixel accuracy
    final_cx, final_cy = refine_center_least_squares(
        x_coords, y_coords, 
        initial_cx=global_cx, 
        initial_cy=global_cy, 
        fixed_r=best_radius
    )

    return final_cx, final_cy



def extract_center_distance_transform(pred_mask, **kwargs):
    """
    Finds the center by computing the distance transform of the mask 
    and finding the point furthest from the background.
    """
    if pred_mask is None or not np.any(pred_mask):
        return None
        
    # Calculate distance from every foreground pixel to the nearest background pixel
    dist_map = distance_transform_edt(pred_mask)
    
    # Find the pixel with the maximum distance
    row, col = np.unravel_index(np.argmax(dist_map), dist_map.shape)
    return float(col), float(row)



def extract_center_ellipse_fit(pred_mask, **kwargs):
    """
    Extracts the center by fitting an Ellipse to the polygon exterior.
    """
    mask_to_polygon_fn = kwargs.get("mask_to_polygon_fn")
    
    pred_poly = mask_to_polygon_fn(pred_mask)
    if pred_poly is None or pred_poly.is_empty:
        return None
        
    polys_to_plot = pred_poly.geoms if hasattr(pred_poly, 'geoms') else [pred_poly]
    x_coords, y_coords = [], []
    
    for p in polys_to_plot:
        if p.is_empty or p.exterior is None:
            continue
        x, y = p.exterior.xy
        x_coords.extend(x)
        y_coords.extend(y)

    if len(x_coords) < 6: # Ellipse needs at least 5 points, 6 is safer
        return None
        
    points = np.column_stack([x_coords, y_coords])
    
    ellipse = EllipseModel()
    try:
        success = ellipse.estimate(points)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        # Catch complex number TypeErrors and linear algebra errors from degenerate points
        success = False
    
    if success:
        # xc, yc, a, b, theta
        xc, yc, _, _, _ = ellipse.params
        return float(xc), float(yc)
    return None



# --- Heatmaps ---

def extract_center_intensity_weighted(pred_mask, prediction, **kwargs):
    """
    Extracts center using intensity-weighted Center of Mass (useful for smooth heatmaps).
    """
    min_confidence_peak = kwargs.get("min_confidence_peak", 0.1)
    
    if prediction is None or pred_mask is None:
        return None

    # check min confiedence on only the current object
    masked_heatmap = np.where(pred_mask & (prediction >= min_confidence_peak), prediction, 0.0)
    if np.sum(masked_heatmap) == 0:
        return None

    # scipy returns (row, col)
    row_center, col_center = center_of_mass(masked_heatmap)
    if np.isnan(row_center) or np.isnan(col_center):
        return None
    return float(col_center), float(row_center)


def extract_center_peak_maxima(pred_mask, prediction, **kwargs):
    """
    Finds heatmap-maximum inside a specific cadidate mask (proposal).
    
    Args:
        pred_mask (np.ndarray): Binary mask from the single Candidate-Object.
        prediction (np.ndarray): Continuous Heatmap/Probability Map [0..1].
    """
    min_confidence_peak = kwargs.get("min_confidence_peak", 0.3)
    min_distance = kwargs.get("min_distance", 3)

    if prediction is None or pred_mask is None:
        return None

    # Masking the prediction, so only heatmap values of the candidate are visible
    masked_pred = np.where(pred_mask, prediction, 0.0)

    # Break if the max probability is under the given threshold
    if np.max(masked_pred) < min_confidence_peak:
        return None

    # search the peak (hopefully center) inside the instance
    # we want only one peak, because it is one proposal manhole
    peaks = peak_local_max(
        masked_pred,
        min_distance=min_distance,
        threshold_abs=min_confidence_peak,
        num_peaks=1,
        exclude_border=False,
    )

    if len(peaks) == 0:
        # Fallback: if peak_local_max cause of min_distancefinds nothing 
        # then just take the argmax value
        row, col = np.unravel_index(np.argmax(masked_pred), masked_pred.shape)
        return float(col), float(row)

    if len(peaks) > 1:
        print(f"[Warning] {len(peaks)} Peaks in one Proposal found (we take simply the first).")

    row, col = peaks[0]
    return float(col), float(row)  # Return (x, y) = (col, row)



def extract_edge_aware_maxima_peak(pred_mask, prediction, **kwargs):
    """
    Finds the heatmap center. Uses standard peak detection for fully visible 
    objects, and Gradient Vector Intersection for out-of-bounds edge cases.
    
    Args:
        pred_mask (np.ndarray): Binary mask from the single Candidate-Object.
        prediction (np.ndarray): Continuous Heatmap/Probability Map [0..1].
    """
    min_confidence_peak = kwargs.get("min_confidence_peak", 0.3)
    min_distance = kwargs.get("min_distance", 3)
    
    if prediction is None or pred_mask is None:
        return None

    # Mask the prediction so only the candidate's heatmap values are evaluated
    masked_pred = np.where(pred_mask, prediction, 0.0)
    
    # 1. Break if the max probability is under the given threshold
    if np.max(masked_pred) < min_confidence_peak:
        return None

    # 2. Check where the current maximum is located
    row_max, col_max = np.unravel_index(np.argmax(masked_pred), masked_pred.shape)
    h, w = masked_pred.shape
    
    # If the maximum is exactly on the image border, the true center is likely outside
    is_truncated = (row_max == 0 or row_max == h - 1 or col_max == 0 or col_max == w - 1)

    if not is_truncated:
        # CASE A: Normal Case (Center is inside the image)

        peaks = peak_local_max(
            masked_pred,
            min_distance=min_distance,
            threshold_abs=min_confidence_peak,
            num_peaks=1,
            exclude_border=False,
        )
        if len(peaks) > 0:
            return float(peaks[0][1]), float(peaks[0][0])  # col, row
        else:
            return float(col_max), float(row_max)

    else:
        # CASE B: Edge Case (Center is outside the image)
        # Gradient Vector Intersection

        # Slight smoothing to reduce noise in the gradients
        smoothed = gaussian_filter(prediction, sigma=1.0)
        
        # Calculate gradients (slopes) in Y and X directions
        dy, dx = np.gradient(smoothed)
        
        # Only consider pixels inside the mask that have a noticeable slope
        # (This ignores flat noise areas and flattened peaks)
        grad_mag = np.sqrt(dx**2 + dy**2)
        valid_pixels = pred_mask & (grad_mag > 0.05 * np.max(grad_mag))
        
        y_coords, x_coords = np.where(valid_pixels)
        
        if len(x_coords) < 3:
            # Not enough sloping edge pixels to calculate an intersection
            return float(col_max), float(row_max)
            
        dx_valid = dx[valid_pixels]
        dy_valid = dy[valid_pixels]
        
        # We define lines along the gradient for each pixel.
        # Equation: Normal_Vector * Center = Normal_Vector * Pixel_Position
        # The normal vector to a gradient (dx, dy) is (dy, -dx)
        A = np.vstack((dy_valid, -dx_valid)).T
        b = dy_valid * x_coords - dx_valid * y_coords
        
        # Solve the linear system using Least Squares
        try:
            center_x, center_y = np.linalg.lstsq(A, b, rcond=None)[0]
            return float(center_x), float(center_y)
        except np.linalg.LinAlgError:
            # Fallback if the matrix is singular
            return float(col_max), float(row_max)



def extract_center_template_matching(pred_mask, prediction, **kwargs):
    """
    Finds the center by correlating the prediction with an ideal Gaussian template.
    """
    sigma = kwargs.get("sigma", 3.0)
    
    if prediction is None or pred_mask is None:
        return None
        
    masked_pred = np.where(pred_mask, prediction, 0.0)
    if np.max(masked_pred) < 0.1:
        return None

    # Create a small, perfect Gaussian template
    size = int(sigma * 6)
    size = size + 1 if size % 2 == 0 else size # make it odd
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    center = size // 2
    template = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    
    # Cross-correlate template with the masked prediction
    correlation = correlate2d(masked_pred, template, mode='same', boundary='fill')
    
    # The highest correlation is our center
    row, col = np.unravel_index(np.argmax(correlation), correlation.shape)
    return float(col), float(row)



# Registry for center extraction methods
CENTER_2D_EXTRACTION_STRATEGIES = {
    # for segmentations
    "polygon_centroid": extract_center_polygon_centroid,
    "circle_least_squares": extract_center_circle_least_squares,
    "hough_least_squares": extract_center_hough_least_squares,
    "ellipse_fit": extract_center_ellipse_fit,
    "distance_transform": extract_center_distance_transform,
    "least_square_radius": extract_center_least_square_radius,

    # for heatmaps
    "intensity_weighted_center": extract_center_intensity_weighted,
    "peak_maxima": extract_center_peak_maxima,
    "edge_peak_maxima":extract_edge_aware_maxima_peak,
    "template_matching":extract_center_template_matching,
}



# -----------------------------------------------
# > 3D Center Localization/Extraction Functions <
# -----------------------------------------------
def extract_3d_center_polygon_centroid(pred_mask, **kwargs):
    """
    Extracts the 3D center using the geometric mean (centroid) of the 3D cluster points.
    """
    pts_3d = kwargs.get("pts_3d")
    
    if pts_3d is None or pred_mask is None:
        return None

    # Get the points belonging to this specific predicted object
    cluster_pts = pts_3d[pred_mask]
    
    if len(cluster_pts) == 0:
        return None
        
    # Calculate geometric centroid
    centroid = np.mean(cluster_pts, axis=0)
    
    # Return local x, y, z
    return float(centroid[0]), float(centroid[1]), float(centroid[2])



def extract_3d_center_bbox_median(pred_mask, **kwargs):
    """
    Extracts the 3D center using the XY bounding box center and Median Z.
    Immune to point density bias and vertical noise outliers.
    """
    pts_3d = kwargs.get("pts_3d")
    
    if pts_3d is None or pred_mask is None:
        return None

    cluster_pts = pts_3d[pred_mask]
    if len(cluster_pts) == 0:
        return None
        
    # X and Y: Use min/max to ignore density bias (points concentrated on one side)
    min_x, min_y = np.min(cluster_pts[:, 0]), np.min(cluster_pts[:, 1])
    max_x, max_y = np.max(cluster_pts[:, 0]), np.max(cluster_pts[:, 1])
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    
    # Z: Use median to completely ignore noise outliers (e.g. grass, reflections)
    center_z = np.median(cluster_pts[:, 2])
    
    return float(center_x), float(center_y), float(center_z)



def extract_3d_center_pca_plane_circle(pred_mask, **kwargs):
    """
    Fits a 3D plane to the points (handling sloped roads) using PCA, 
    projects points to a local 2D system, fits a circle, and maps back to 3D.
    """
    pts_3d = kwargs.get("pts_3d")
    
    if pts_3d is None or pred_mask is None:
        return None

    cluster_pts = pts_3d[pred_mask]
    
    # We need at least a few points to define a plane and a circle
    if len(cluster_pts) < 10:
        return None

    # 1. PCA to find the 3D plane of the manhole
    centroid_3d = np.mean(cluster_pts, axis=0)
    centered_pts = cluster_pts - centroid_3d
    
    # Covariance matrix to find principal directions
    cov_matrix = np.dot(centered_pts.T, centered_pts) / len(centered_pts)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors by eigenvalues descending
    # The smallest eigenvalue corresponds to the normal of the plane
    # The two largest correspond to the X and Y axes ON the manhole plane
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    local_x_axis = eigenvectors[:, 0]
    local_y_axis = eigenvectors[:, 1]
    
    # 2. Project 3D points onto the local 2D plane
    # Dot product of centered points with the local axes
    local_2d_x = np.dot(centered_pts, local_x_axis)
    local_2d_y = np.dot(centered_pts, local_y_axis)
    
    # 3. Least Squares Circle Fit in the local 2D plane
    def calc_distances(params, x, y):
        cx, cy, r = params
        return np.sqrt((x - cx)**2 + (y - cy)**2) - r

    # Initial guess for local center is (0,0) because we centered the points earlier
    # Initial guess for radius can be estimated from the variance
    initial_r = np.sqrt(np.var(local_2d_x) + np.var(local_2d_y))
    initial_guess = [0.0, 0.0, initial_r]
    
    try:
        result = least_squares(calc_distances, initial_guess, args=(local_2d_x, local_2d_y))
        local_cx, local_cy, best_r = result.x
    except ValueError:
        return None

    # 4. Transform the local 2D center back to global 3D coordinates
    # global_center = centroid_3d + (local_cx * local_x_axis) + (local_cy * local_y_axis)
    global_center_x = centroid_3d[0] + local_cx * local_x_axis[0] + local_cy * local_y_axis[0]
    global_center_y = centroid_3d[1] + local_cx * local_x_axis[1] + local_cy * local_y_axis[1]
    global_center_z = centroid_3d[2] + local_cx * local_x_axis[2] + local_cy * local_y_axis[2]
    
    return float(global_center_x), float(global_center_y), float(global_center_z)



# Registry for 3D center extraction methods
CENTER_3D_EXTRACTION_STRATEGIES = {
    "polygon_centroid_3d": extract_3d_center_polygon_centroid,               # Basic Mean
    "bbox_median_z": extract_3d_center_bbox_median,                       # Better: ignores density bias & Z-noise
    "pca_plane_circle_least_squares": extract_3d_center_pca_plane_circle 
}



# --------------------
# > Debugging helper <
# --------------------

def plot_center_prediction_debug(
    pixel_values, labels, labeled_preds, preds_prob, preds_binary, preds_closed, 
    extracted_centers, gt_centers, pc_points, meta, pc_id, idx, save_dir,
    num_preds, is_heatmap=False, extra_debug_info=None
):
    """
    Generates a visual diagnostic grid to isolate failures between 
    segmentation predictions, morphology, center fitting, and 3D projection.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Extract & Normalize Image Channels
    # -------------------------------------------------------------------------
    if isinstance(pixel_values, torch.Tensor):
        pv_np = pixel_values.detach().cpu().numpy().squeeze()
    else:
        pv_np = np.squeeze(pixel_values)
        
    img_ht, img_int = None, None
    if pv_np.ndim == 2:
        img_ht = pv_np
    elif pv_np.ndim == 3:
        img_ht = pv_np[0] if pv_np.shape[0] > 0 else None
        img_int = pv_np[1] if pv_np.shape[0] > 1 else None

    # Helper function for visibility normalization
    def normalize_img(img):
        if img is None: return None
        img_min, img_max = np.min(img), np.max(img)
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img

    img_ht_norm = normalize_img(img_ht)
    img_int_norm = normalize_img(img_int)
    
    # Use the height map as the background for the bottom row (fallback to zeros if missing)
    img_bg = img_ht_norm if img_ht_norm is not None else np.zeros((256, 256))

    # -------------------------------------------------------------------------
    # Extract 3D Point Cloud Data
    # -------------------------------------------------------------------------
    pts = None
    if pc_points is not None:
        if isinstance(pc_points, torch.Tensor):
            pts = pc_points.detach().cpu().numpy()
        elif isinstance(pc_points, np.ndarray):
            pts = pc_points
        elif isinstance(pc_points, (list, tuple)) and len(pc_points) > 0:
            p0 = pc_points[0]
            pts = p0.detach().cpu().numpy() if isinstance(p0, torch.Tensor) else np.array(p0)
            
    if pts is not None and pts.ndim == 3:
        pts = np.squeeze(pts) # Handle batch dim if present

    x, y, z, intensity = None, None, None, None
    if pts is not None and pts.ndim == 2 and len(pts) > 0:
        if pts.shape[1] >= 3:
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        if pts.shape[1] >= 4:
            intensity = pts[:, 3]

    # Handle missing conventional outputs
    if preds_prob is None: preds_prob = np.zeros_like(img_bg)
    if preds_binary is None: preds_binary = np.zeros_like(img_bg)
    if labeled_preds is None: labeled_preds = np.zeros_like(img_bg)

    # Label processing
    preds_colored = np.where(labeled_preds > 0, labeled_preds, np.nan)
    labels_clean = np.where(labels == 255, 0, labels) if labels is not None and np.any(labels == 255) else labels
    labels_clean = np.squeeze(labels_clean) if labels_clean is not None else np.zeros_like(img_bg)
    
    # labeled_gt, num_gt = label((labels_clean > 0).astype(int))
    # gt_colored = np.where(labeled_gt > 0, labeled_gt, np.nan)
    
    cmap_pred = plt.colormaps.get_cmap("tab10").resampled(max(num_preds, 1))
    # cmap_gt = plt.colormaps.get_cmap("Set1").resampled(max(num_gt, 1))

    # -------------------------------------------------------------------------
    # Setup Figure with GridSpec (2 rows: Top=4 cols, Bottom=5 cols)
    # Using 20 slots to perfectly divide 4 plots (span 5) and 5 plots (span 4)
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(26, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 20)
    
    # Row 1: Inputs (Span 5 each)
    ax_2d_int = fig.add_subplot(gs[0, 0:5])
    ax_3d_int = fig.add_subplot(gs[0, 5:10], projection='3d')
    ax_2d_ht  = fig.add_subplot(gs[0, 10:15])
    ax_3d_ht  = fig.add_subplot(gs[0, 15:20], projection='3d')

    # Row 2: Pipeline (Span 4 each)
    # ax_gt    = fig.add_subplot(gs[1, 0:4])
    ax_prob  = fig.add_subplot(gs[1, 0:5])
    ax_bin   = fig.add_subplot(gs[1, 5:10])
    ax_morph = fig.add_subplot(gs[1, 10:15])
    ax_ext   = fig.add_subplot(gs[1, 15:20])

    # --- Helper to format 3D plots ---
    def format_3d_ax(ax, title):
        ax.set_title(title)
        ax.set_xlabel("X", labelpad=10)
        ax.set_ylabel("Y", labelpad=10)
        ax.set_zlabel("Z", labelpad=15) # Increased padding to prevent overlay
        # ax.tick_params(axis='z', pad=10) # Push Z ticks further out
        # for tick in ax.get_zticklabels():
        #     tick.set_rotation(0)
        ax.zaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis='z', labelsize=8)
        try:
            ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
        except Exception:
            pass

    def text_missing(ax, title, is_3d=False):
        if is_3d:
            ax.text2D(0.5, 0.5, "Data not available for this plot", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Data not available for this plot", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis('off')

    # -------------------------------------------------------------------------
    # Row 1 (Inputs)
    # -------------------------------------------------------------------------
    # 1. 2D Intensity
    if img_int_norm is not None:
        alpha = np.ones_like(img_int_norm, dtype=float)
        alpha[img_int_norm == 0] = 0.1
        ax_2d_int.imshow(img_int_norm, cmap="plasma", alpha=alpha)
        # ax_2d_int.imshow(img_int_norm, cmap="plasma", origin="lower")
        ax_2d_int.set_title("1. Input 2D Intensity (Normalized)")
        ax_2d_int.axis("off")
    else:
        text_missing(ax_2d_int, "1. Input 2D Intensity")

    # 2. 3D Intensity
    if intensity is not None and x is not None:
        sc = ax_3d_int.scatter(x, y, z, c=intensity, cmap="plasma", s=1, alpha=0.5)
        ax_3d_int.invert_yaxis()
        # fig.colorbar(sc, ax=ax_3d_int, fraction=0.046, pad=0.1, label="Intensity")
        cax = inset_axes(
            ax_3d_int,
            width="60%",
            height="3%",
            loc="upper center"
        )

        cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cbar.set_label("Intensity")
        format_3d_ax(ax_3d_int, "2. Input 3D Intensity")
    else:
        text_missing(ax_3d_int, "2. Input 3D Intensity", is_3d=True)

    # 3. 2D Height
    if img_ht_norm is not None:
        alpha = np.ones_like(img_ht_norm, dtype=float)
        alpha[img_ht_norm == 0] = 0.1
        ax_2d_ht.imshow(img_ht_norm, cmap="plasma")
        # ax_2d_ht.imshow(img_ht_norm, cmap="plasma", origin="lower")
        ax_2d_ht.set_title("3. Input 2D Height (Normalized)")
        ax_2d_ht.axis("off")
    else:
        text_missing(ax_2d_ht, "3. Input 2D Height")

    # 4. 3D Height
    if z is not None and x is not None:
        sc = ax_3d_ht.scatter(x, y, z, c=z, cmap="plasma", s=1, alpha=0.5)
        ax_3d_ht.invert_yaxis()
        # fig.colorbar(sc, ax=ax_3d_ht, fraction=0.046, pad=0.1, label="Z (Height)")
        cax = inset_axes(
            ax_3d_ht,
            width="60%",
            height="3%",
            loc="upper center"
        )

        cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cbar.set_label("Height")
        format_3d_ax(ax_3d_ht, "4. Input 3D Height")
    else:
        text_missing(ax_3d_ht, "4. Input 3D Height", is_3d=True)

    # -------------------------------------------------------------------------
    # Row 2 (Pipeline outputs)
    # -------------------------------------------------------------------------
    # # 5. GT Mask
    # ax_gt.imshow(img_bg, cmap="gray", alpha=0.4)
    # if num_gt > 0:
    #     ax_gt.imshow(gt_colored, cmap=cmap_gt, alpha=0.6, interpolation="none")
    # ax_gt.set_title("5. GT Mask Overlay")
    # ax_gt.axis("off")
    
    # 6. Raw Model Probability
    im_prob = ax_prob.imshow(preds_prob, cmap="magma")
    fig.colorbar(im_prob, ax=ax_prob, fraction=0.046, pad=0.04)
    ax_prob.set_title("6. Raw Model Probability")
    if is_heatmap:
        for cx, cy in extracted_centers:
            ax_prob.scatter(cx, cy, c="cyan", marker="+", s=150, linewidths=2, label="Peak")
    ax_prob.axis("off")

    # 7. Thresholded Mask
    ax_bin.imshow(preds_binary, cmap="binary")
    ax_bin.set_title("7. Thresholded Mask (Conf-Filter)")
    ax_bin.axis("off")

    # 8. Post-Processed Closed Mask
    ax_morph.imshow(img_bg, cmap="gray", alpha=0.3)
    ax_morph.imshow(preds_colored, cmap=cmap_pred, alpha=0.8, interpolation="none")
    ax_morph.set_title(f"8. Closed Pred Objects (Count: {num_preds})")
    ax_morph.axis("off")

    # 9. Extracted Centers
    ax_ext.imshow(img_bg, cmap="gray")
    ax_ext.imshow(preds_colored, cmap=cmap_pred, alpha=0.3, interpolation="none")
    # for gx, gy in gt_centers:
    #     ax_ext.scatter(gx, gy, edgecolors="lime", facecolors="none", marker="o", s=50, linewidths=2.5, label="GT")
    for cx, cy in extracted_centers:
        ax_ext.scatter(cx, cy, c="red", marker="x", s=120, linewidths=2.5, label="Pred")
        
    if extra_debug_info is not None:
        for info in extra_debug_info:
            if info.get('type') == 'circle':
                circ = patches.Circle((info['x'], info['y']), info['r'], fill=False, edgecolor='blue', linestyle='--', lw=2)
                ax_ext.add_patch(circ)
            elif info.get('type') == 'ellipse':
                ell = patches.Ellipse((info['x'], info['y']), info['width'], info['height'], angle=info['angle'], fill=False, edgecolor='orange', lw=2)
                ax_ext.add_patch(ell)
            elif info.get('type') == 'polygon':
                ax_ext.plot(info['poly_x'], info['poly_y'], color='cyan', lw=1)
                
    ax_ext.set_title("9. Extracted Centers & Shapes")
    handles, labels_leg = ax_ext.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    if by_label:
        ax_ext.legend(by_label.values(), by_label.keys(), loc='upper right')
    ax_ext.axis("off")

    plt.suptitle(f"Debug Analysis - Sample #{idx} | PC ID: {pc_id}", fontsize=18, fontweight="bold")
    out_file = save_dir / f"debug_{pc_id}_idx_{idx}.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)



# --------------------
# > Important Helper <
# --------------------
def add_to_result(result, cur_dataset, pc_id, cur_center_point):
    found_entry = False
    for cur_res in result:
        if cur_res.get("dataset", "-999") == cur_dataset and \
            cur_res.get("pointcloud-id", "-999") == pc_id:
            if "centers" in cur_res:  # hasattr(cur_res, "centers"):
                cur_res["centers"].append(cur_center_point)
            else:
                cur_res["centers"] = [cur_center_point]

            found_entry = True
            break
    if not found_entry:
        result.append({
            "dataset": cur_dataset,
            "pointcloud-id": pc_id,
            "centers": [cur_center_point]
        })
    
    return result



# Merging
def merge_nearby_centers(centers, dist_threshold=0.8):
    """
    Merges predicted (x, y, z) centers that are closer than dist_threshold pixels.
    """
    if len(centers) <= 1:
        return centers

    merged_centers = []
    used = [False] * len(centers)

    for i in range(len(centers)):
        if used[i]:
            continue

        cluster = [centers[i]]
        used[i] = True

        for j in range(i + 1, len(centers)):
            if used[j]:
                continue

            # calculate 3d euclidean distance
            dist = np.sqrt(
                (centers[i][0] - centers[j][0])**2 +
                (centers[i][1] - centers[j][1])**2 +
                (centers[i][2] - centers[j][2])**2
            )

            if dist < dist_threshold:
                cluster.append(centers[j])
                used[j] = True

        avg_x = sum(pt[0] for pt in cluster) / len(cluster)
        avg_y = sum(pt[1] for pt in cluster) / len(cluster)
        avg_z = sum(pt[2] for pt in cluster) / len(cluster)
        # avg_conf = sum(pt["confidence"] for pt in cluster) / len(cluster)  # only with same confidence merging!
        avg_time = sum(pt[3] for pt in cluster) / len(cluster)
        merged_centers.append((avg_x, avg_y, avg_z, avg_time))
    return merged_centers



def filter_ignored_manholes(
    results,
    raw_dataset,
    box_size=0.6,
    ignore_label=255,
    ignore_threshold=0.20,
    check_z_axis=False
):
    """
    Filters predicted manhole centers if the region around them contains
    >= ignore_threshold (20%) points labeled with ignore_label (255).

    Args:
        results (list): List of dicts, each formatted as:
            {
                "dataset": str,
                "pointcloud-id": str/int,
                "centers": [{"x": float, "y": float, "z": float, "confidence": float}, ...]
            }
        dataloader: PyTorch DataLoader for fetching full raw point clouds.
        box_size (float): Side length of the square bounding box in meters (default: 0.6m).
        ignore_label (int): Label value representing ignored areas (default: 255).
        ignore_threshold (float): Ratio threshold to reject predictions (default: 0.20).
        check_z_axis (bool): If True, also bounds Z within box_size/2. If False, checks XY bounding box only.

    Returns:
        list: Filtered results with predictions in ignored regions removed.
    """
    half_box = box_size / 2.0
    filtered_results = []
    pc_cache = {}

    for entry in results:
        pc_id = entry["pointcloud-id"]
        centers = entry.get("centers", [])

        if not centers:
            continue

        # 1. Fetch full raw point cloud for current pc_id
        if pc_id not in pc_cache:
            try:
                raw_pc_idx = raw_dataset.get_idx_by_pc_id(pc_id)
                raw_point_cloud = raw_dataset[raw_pc_idx]
                raw_positions = raw_point_cloud.point[get_coordinate_attribute(raw_point_cloud)].numpy()
                raw_labels = raw_point_cloud.point[get_class_attribute(raw_point_cloud)].numpy()
                pc_cache[pc_id] = (raw_positions, raw_labels, None)
            except Exception as e:
                pc_cache[pc_id] = (None, None, e)
                # filtered_results.append(entry)
                continue

        # Extract coordinates and labels
        raw_positions, raw_labels, err = pc_cache[pc_id]
        if err is not None:
            print(f"Warning: Could not load point cloud ID '{pc_id}': {err}. Skipping filtering for these centers.")
            filtered_results.append(entry)
            continue
        # raw_positions = raw_point_cloud.point[get_coordinate_attribute(raw_point_cloud)].numpy()
        
        # # if "labels" not in raw_point_cloud.point:
        # #     # If no labels present, keep all centers
        # #     filtered_results.append(entry)
        # #     continue

        # raw_labels = raw_point_cloud.point[get_class_attribute(raw_point_cloud)].numpy()

        valid_centers = []

        # Pre-filter raw_positions to the min/max extent of all centers in this pc_id for 10x faster masking
        cx_all = np.array([c["x"] for c in centers])
        cy_all = np.array([c["y"] for c in centers])
        global_mask = (
            (raw_positions[:, 0] >= cx_all.min() - half_box) &
            (raw_positions[:, 0] <= cx_all.max() + half_box) &
            (raw_positions[:, 1] >= cy_all.min() - half_box) &
            (raw_positions[:, 1] <= cy_all.max() + half_box)
        )
        sub_positions = raw_positions[global_mask]
        sub_labels = raw_labels[global_mask]

        # 2. Process each center point
        for center in centers:
            cx, cy, cz = center["x"], center["y"], center["z"]

            # Crop box in XY plane (and optionally Z)
            in_box_mask = (
                (np.abs(sub_positions[:, 0] - cx) <= half_box) &
                (np.abs(sub_positions[:, 1] - cy) <= half_box)
            )

            if check_z_axis:
                in_box_mask = in_box_mask & (np.abs(sub_positions[:, 2] - cz) <= half_box)

            box_labels = sub_labels[in_box_mask]

            # If no points are found in the box, keep the prediction
            if len(box_labels) == 0:
                valid_centers.append(center)
                continue

            # Calculate ignore label ratio
            ignore_ratio = np.mean(box_labels == ignore_label)

            # Keep only if ignore label coverage is strictly below threshold
            if ignore_ratio < ignore_threshold:
                valid_centers.append(center)
                continue

            print(f"[INFO] Prediction was filtered, thus it is probably a prediction on a ignored manhole. You might want to check that:\n    Point:{center}\n    PC-ID: {pc_id}")

        if valid_centers:
            # Create a shallow copy with updated centers list
            filtered_entry = dict(entry)
            filtered_entry["centers"] = valid_centers
            filtered_results.append(filtered_entry)

    return filtered_results



def make_prediction(
    pixel_values=None, 
    model=None, 
    model_name=None, 
    labels=None, 
    ignore_index=255, 
    using_heatmap_as_gt=False,
    as_prob=True, 
    manhole_class_idx=1,
    save_raw_pred=False,
    save_filepath=None,
    load_filepath=None
):
    # --- 1. LOAD OR PREDICT ---
    if load_filepath is not None:
        # Bypass the model and load the raw prediction (e.g., from a 3D model run)
        if not os.path.exists(load_filepath):
            raise FileNotFoundError(f"Prediction file not found at: {load_filepath}")

        preds = np.load(load_filepath)
        
        # Format pixel_values if they are still passed for downstream use
        if pixel_values is not None:
            if isinstance(pixel_values, np.ndarray):
                pixel_values = torch.from_numpy(pixel_values)
            if isinstance(pixel_values, torch.Tensor) and model is not None:
                pixel_values = pixel_values.float().to(model.device)
                
    else:
        # Standard model inference
        if model is None:
            raise ValueError("You must provide either a 'model' or a 'load_filepath'.")
            
        if isinstance(pixel_values, np.ndarray):
            pixel_values = torch.from_numpy(pixel_values)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.float().to(model.device)

        preds = predict_single_sample(
            model, 
            model_name, 
            None, 
            pixel_values,
            using_heatmap_as_gt=using_heatmap_as_gt, 
            as_prob=as_prob, 
            manhole_class_idx=manhole_class_idx
        )

    # --- 2. SAVE RAW PREDICTION ---
    if save_raw_pred and save_filepath is not None:
        save_dir = os.path.dirname(save_filepath)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            os.chmod(save_dir, 0o777)
            
        # Convert tensor to numpy for framework-agnostic storage
        preds_to_save = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else preds
        np.save(save_filepath, preds_to_save)
        os.chmod(save_filepath, 0o666)

    # --- 3. POST-PROCESSING ---
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
        
    valid_mask = (labels != ignore_index) if labels is not None else None

    preds_squeezed = np.squeeze(preds)

    return preds_squeezed, valid_mask, preds, labels, pixel_values

    # # Apply closing + clustering / confidence thresholds
    # if valid_mask is not None:
    #     preds_binary = ((preds >= min_confidence) & valid_mask).astype(np.uint8)
    # else:
    #     preds_binary = (preds >= min_confidence).astype(np.uint8)
        
    # preds_binary = np.squeeze(preds_binary)
    # preds_prob = np.squeeze(preds)
    # preds_prob *= preds_binary

    # return preds_prob, preds_binary, valid_mask, preds, labels, pixel_values



def apply_confidence_threshold(preds_squeezed, valid_mask, min_confidence):
    """
    Step 3 of make_prediction: Applies a specific min_confidence threshold 
    to the already-loaded prediction without modifying the original array in-place.
    """
    if valid_mask is not None:
        preds_binary = ((preds_squeezed >= min_confidence) & np.squeeze(valid_mask)).astype(np.uint8)
    else:
        preds_binary = (preds_squeezed >= min_confidence).astype(np.uint8)
        
    preds_binary = np.squeeze(preds_binary)
    # Important: Use multiplication (creates a copy) instead of *= so preds_squeezed isn't mutated across confidences!
    preds_prob = preds_squeezed * preds_binary
    return preds_prob, preds_binary



def extract_and_match_centers(dataset_name, pc_id, gt_json_file_paths, queried_points_3d=None, gt_patch_labels=None, meta=None, pc=None, min_points=20):
    """
    Extracts GT centers from JSON for a given dataset and pc-id, and finds the closest 
    GT center for pasted 3D points. Optionally extracts candidates from 2D GT patch labels.
    
    Args:
        dataset_name (str): e.g., "whu" or "sud".
        pc_id (str): Pointcloud ID, e.g., "0940".
        queried_points_3d (list of dicts): Explicit 3D points to match [{'x':.., 'y':.., 'z':..}].
        gt_patch_labels (np.ndarray): Optional 2D array of GT labels to extract candidates from.
        meta (dict): Optional metadata for 2D->3D conversion (requires pc).
        pc: Optional point cloud tensor/array for 2D->3D conversion.
        min_points (int): Minimum cluster size for manhole candidates.
        
    Returns:
        list of dicts containing the queried point, the nearest GT center, and the Euclidean distance.
    """

    # 0. Load JSON GT Data
    json_data = None
    for cur_dataset_json_path in gt_json_file_paths:
        if dataset_name in os.path.split(cur_dataset_json_path)[-1]:
            with open(cur_dataset_json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

    if not json_data:
        print("[WARNING] Ground Truth for debugging not found.")
        return None
    
    # 1. Extract Ground Truth Centers from JSON
    gt_centers = []
    for entry in json_data:
        if entry.get("dataset") == dataset_name and entry.get("pointcloud-id") == pc_id:
            gt_centers = entry.get("centers", [])
            break
            
    # 2. Optional Preprocessing: Extract centers from 2D patch labels via Morph Closing & Clustering
    if queried_points_3d is None and gt_patch_labels is not None:
        queried_points_3d = []
        
        # Binary mask (assuming >0 is manhole label)
        binary_mask = (np.squeeze(gt_patch_labels) > 0).astype(np.uint8)
        
        # Morphological Closing
        struct = generate_binary_structure(2, 2)
        closed_mask = binary_closing(binary_mask, structure=struct, iterations=8).astype(np.uint8)
        
        # Clustering (Labeling)
        labeled_preds, num_objects = label(closed_mask)
        
        for p_idx in range(1, num_objects + 1):
            mask = labeled_preds == p_idx
            if np.sum(mask) < min_points:
                continue
                
            # Extract 2D Center of Mass
            cy, cx = center_of_mass(mask)

            queried_points_3d.append({"x": cx, "y": cy})
            
            # # Transform to 3D if meta and pc are provided (using your pipeline's function)
            # if meta is not None and pc is not None:
            #     center_3d = bev_pixel_to_3d(
            #         patch_points=pc,
            #         pixel_x=cx,
            #         pixel_y=cy,
            #         origin_x=meta["origin_x"],
            #         origin_y=meta["origin_y"],
            #         resolution=meta["resolution"],
            #         search_radius=None,
            #         tile_size=meta["tile_size"],
            #         invert_y=False,
            #         invert_x=False
            #     )
            #     if center_3d is not None and len(center_3d) >= 3:
            #         queried_points_3d.append({"x": float(center_3d[0]), "y": float(center_3d[1]), "z": float(center_3d[2])})
            # else:
            #     # Fallback if no 3D conversion data is passed
            #     queried_points_3d.append({"x": cx, "y": cy, "z": 0.0})

    # 3. Match each queried point to the nearest Ground Truth Center
    results = []
    if not queried_points_3d:
        return results

    for q_pt in queried_points_3d:
        nearest_gt = None
        min_dist = float('inf')
        
        for gt in gt_centers:
            # Euclidean Distance
            dist = math.sqrt(
                (q_pt["x"] - gt["x"])**2 + 
                (q_pt["y"] - gt["y"])**2 
                # (q_pt["z"] - gt["z"])**2
            )
            if dist < min_dist and dist < 10.0:
                min_dist = dist
                nearest_gt = gt
                
        # results.append({
        #     "query_point": q_pt,
        #     "nearest_gt_center": nearest_gt,
        #     "distance": min_dist
        # })
        if nearest_gt:
            results.append((nearest_gt['x'], nearest_gt['y']))
        
    return results



# -------------------------------
# > Center Eval - Main Pipeline <
# -------------------------------
# Main Pipeline to predict Center Points via runtime models OR loaded predictions.
def center_eval(config):

    model_name = config.model.name.lower()

    enable_debug = getattr(config.center_eval, "save_debug_plots", False)
    # gt_json_file_paths = getattr(config.center_eval, "gt_json_file_paths")
    min_input_points = getattr(config.center_eval, "min_input_points")
    min_confidences = getattr(config.center_eval, "min_confidences")
    min_confidence_peak = getattr(config.center_eval, "min_confidence_peak")
    candidate_min_points = getattr(config.center_eval, "candidate_min_points")
    # run_in_only_pred_save_mode = getattr(config.center_eval, "run_in_only_pred_save_mode")
    # pred_save_mode_saving_name = getattr(config.center_eval, "pred_save_mode_saving_name")
    run_mode = getattr(config.center_eval, "run_mode")  # save, load, normal (everything else)
    pred_save_load_mode_saving_name = getattr(config.center_eval, "pred_save_load_mode_saving_name")
    pred_save_load_mode_saving_path = getattr(config.center_eval, "pred_save_load_mode_saving_path")
    clear_pred_save_load_only_model_folder = getattr(config.center_eval, "clear_pred_save_load_folder")
    clear_pred_save_load_folder = getattr(config.center_eval, "clear_pred_save_load_folder")
    process_3d_mode = getattr(config.center_eval, "process_3d_mode")

    if run_mode.lower() in ["load", "save"]:
        model_name = pred_save_load_mode_saving_name

    if model_name.lower().startswith("traditional_"):
        min_confidences = [-999]
        is_3D_model = False
    elif model_name.lower().startswith("3d"):
        is_3D_model = True
    else:
        is_3D_model = False
    
    if run_mode == "save":
        min_confidences = [min_confidences[int((len(min_confidences)-1)//2)]]
        if clear_pred_save_load_folder and os.path.exists(pred_save_load_mode_saving_path):
            shutil.rmtree(pred_save_load_mode_saving_path)
        model_save_folder = os.path.join(pred_save_load_mode_saving_path, pred_save_load_mode_saving_name)
        if clear_pred_save_load_only_model_folder and os.path.exists(model_save_folder):
            shutil.rmtree(model_save_folder)

    print(f"Confidences: {min_confidences}")
    
    # extract params
    heatmap_path = config.data.heatmap_path
    used_heatmap_channel = config.data.used_heatmap_channel
    if heatmap_path is None or heatmap_path == "None":
        using_heatmap_as_gt = False
        heatmap_path = None
    else:
        using_heatmap_as_gt = True

    ignore_index = 255
    num_labels = 2
    if using_heatmap_as_gt:
        ignore_index = -999
        num_labels = 1
        using_heatmap_as_gt = True

    # Load the TRAINED model checkpoint
    encoder_name = config.model.encoder
    checkpoint_path = config.model.check_point_path
    if not model_name.lower().startswith("traditional_") and run_mode.lower() != "load":
        if (not checkpoint_path or checkpoint_path == "None"):
            raise ValueError("Please provide the path to your trained checkpoint in config.model.check_point_path")

        print(f"Loading trained model and processor from: {checkpoint_path}")
        model, processor = get_model_and_processor(
            model_name, 
            encoder_name, 
            checkpoint_path, 
            mode="test", 
            num_labels=num_labels, 
            ignore_index=ignore_index, 
            heatmap_is_gt=using_heatmap_as_gt
        )
        model.eval().to("cuda")

        parts = Path(checkpoint_path).parts
        exp_name = parts[-2]
    else:
        exp_name = model_name
        processor = None
        model = None

        # if run_mode.lower() in ["load"]:  # "save", 
        #     raise ValueError("Running Save/Load mode with traditional method does not work. Use 2D dep-learning model instead.")

    # Load Test Data
    pass_label_in_preprocessor = model_name in ["mask2former", "oneformer"]
    normalization = config.data.normalization
    normalization_mode = config.data.normalization_mode
    data_split_type = config.data.type

    if data_split_type != "test":
        print("[WARNING] You currently use not 'test' as split. Make sure you really don't want to use the 'test' split.")
    
    extraction_method_name = getattr(config.center_eval, "center_extraction_method")
    is_2d_extractrion_method = extraction_method_name in CENTER_2D_EXTRACTION_STRATEGIES.keys()
    is_3d_extractrion_method = extraction_method_name in CENTER_3D_EXTRACTION_STRATEGIES.keys()
    if not is_2d_extractrion_method and not is_3d_extractrion_method:
        raise ValueError(f"Unknown extraction method: {extraction_method_name}. Choose from {list(CENTER_2D_EXTRACTION_STRATEGIES.keys())}")

    if not model_name.lower().startswith("traditional_"):
        if is_2d_extractrion_method and is_3D_model and process_3d_mode.lower() == "3d":
            raise ValueError("If using 3D Model with 3D processing, you must use a 3d center localization/extraction strategy, see 'CENTER_3D_EXTRACTION_STRATEGIES' var.")

        if is_3d_extractrion_method and ((is_3D_model and process_3d_mode.lower() == "2d") or (not is_3D_model)):
                raise ValueError("If using a 2D model (or a 3D Model with 2D processing), you must use a 2d center localization/extraction strategy, see 'CENTER_2D_EXTRACTION_STRATEGIES' var.")

    if not model_name.lower().startswith("traditional_"):
        exp_name += f"_localization-method_{extraction_method_name}"

    if is_2d_extractrion_method:
        extract_center_fn = CENTER_2D_EXTRACTION_STRATEGIES[extraction_method_name]
    else:
        extract_center_fn = CENTER_3D_EXTRACTION_STRATEGIES[extraction_method_name]
    print(f"Using center extraction strategy: {extraction_method_name}")

    # Pre-create the binary closing structure once
    closing_struct = generate_binary_structure(2, 2)

    for cur_dataset in ["whu", "sud"]:
        # # 1. Pre-load GT JSON ONCE per dataset (only needed if enable_debug is True)
        # gt_json_data = None
        # if enable_debug and gt_json_file_paths:
        #     for cur_dataset_json_path in gt_json_file_paths:
        #         if cur_dataset in os.path.split(cur_dataset_json_path)[-1]:
        #             with open(cur_dataset_json_path, 'r', encoding='utf-8') as f:
        #                 gt_json_data = json.load(f)

        # 2. Prepare debug directories and result containers for ALL confidences upfront
        debug_save_dirs = {}
        raw_threshold_results_by_conf = {conf: [] for conf in min_confidences}

        if enable_debug:
            for min_confidence in min_confidences:
                debug_save_dir = Path(f"{config.center_eval.center_pred_path}/debug_plots/center_pred_{cur_dataset}_{data_split_type}_{exp_name}_{min_confidence}")
                os.makedirs(str(debug_save_dir), exist_ok=True)
                shutil.rmtree(str(debug_save_dir))
                os.makedirs(str(debug_save_dir), exist_ok=True)
                debug_save_dirs[min_confidence] = debug_save_dir

        mid_conf = min_confidences[len(min_confidences) // 2]

        # 3. Initialize datasets ONCE per dataset (outside confidence loop!)
        test_3d_dataset = get_data_loader(
            cur_dataset, 
            config.data.path if cur_dataset == "whu" else config.data.path_2, 
            type=data_split_type,
            transform=get_basic_transform(),
            batch_size=1, 
            shuffle=False, 
            num_workers=4,
            preprocessed=True, 
            return_train_format=True,
            return_dataset=True,
            bev_normalized=config.data.normalization, 
            bev_normalize_mode=config.data.normalization_mode
        )
        
        all_test_paths = test_3d_dataset.point_cloud_paths

        test_bev_dataset = BEVDataset(
            path=all_test_paths, 
            file_paths=[], 
            has_labels=True, 
            image_training=True, 
            preprocessor=processor,
            augment=False,
            pass_label_in_preprocessor=pass_label_in_preprocessor,
            heatmap_gt_path=heatmap_path,
            used_heatmap_channel=used_heatmap_channel,
            normalize=normalization, 
            normalization_mode=normalization_mode
        )

        # 4. Iterate over all patches ONCE
        for idx, cur_data_path in tqdm(enumerate(all_test_paths), total=len(all_test_paths), desc=f"Center Pipe ({cur_dataset})"):
            pc_id, x_start, y_start = test_bev_dataset.extract_grid_identifier(cur_data_path)

            pred_file_path = os.path.join(pred_save_load_mode_saving_path, pred_save_load_mode_saving_name, f"{cur_dataset}_{pc_id}_{x_start}_{y_start}.npy")
            pred_time_file_path = os.path.join(pred_save_load_mode_saving_path, pred_save_load_mode_saving_name, f"{cur_dataset}_{pc_id}_{x_start}_{y_start}_TIME.npy")

            # Load patch data ONCE
            bev_dict = next(test_bev_dataset.get_patch_via_identifier(pc_id, x_start, y_start, return_generator=True))
            meta = bev_dict["meta"]
            pixel_values = bev_dict["pixel_values"].unsqueeze(0)
            labels = bev_dict["labels"]
            gt_labels_numpy = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)

            # Only load 3D patch if actually needed (2D extraction, traditional, or debug plotting)
            need_pc_patch = (
                model_name.lower().startswith("traditional_")
                or not (is_3D_model and process_3d_mode.lower() == "3d")
                or enable_debug
            )
            pc, _ = test_3d_dataset.get_patch_via_identifier(pc_id, x_start, y_start) if need_pc_patch else (None, None)

            # Only compute GT centers if debug plotting is enabled!
            gt_centers = None
            # if enable_debug:
            #     gt_centers = extract_and_match_centers(
            #         dataset_name=cur_dataset, 
            #         pc_id=pc_id, 
            #         gt_json_data=gt_json_data, 
            #         queried_points_3d=None, 
            #         gt_patch_labels=gt_labels_numpy, 
            #         meta=meta, 
            #         pc=pc, 
            #         min_points=10
            #     )

            # Min Input Points -> Optional
            if pc is not None:
                if len(pc) < min_input_points:
                    continue

            # =================================================================
            # CASE A: Traditional Methods
            # =================================================================
            if model_name.lower().startswith("traditional_"):
                start_time = time.perf_counter()
                traditional_method = "_".join(model_name.split("_")[1:])
                min_confidence = min_confidences[0]

                detected_centers, debug_dict = predict_manhole_centers(
                    pts_3d=pc,
                    bev_img=pixel_values,
                    method=traditional_method,
                    meta=meta
                )

                if enable_debug and debug_dict is not None:
                    plot_center_prediction_debug(
                        pixel_values=pixel_values,
                        labels=gt_labels_numpy,
                        labeled_preds=None,
                        preds_prob=debug_dict['prob_map'],
                        preds_binary=debug_dict['binary_mask'],
                        preds_closed=None, 
                        extracted_centers=detected_centers[:, :2],
                        gt_centers=gt_centers,
                        pc_points=pc,
                        meta=meta,
                        pc_id=pc_id,
                        idx=idx,
                        save_dir=debug_save_dirs[min_confidence],
                        num_preds=len(detected_centers),
                        is_heatmap=False,
                        extra_debug_info=debug_dict['geometries']
                    )

                for cur_det_center in detected_centers:
                    if traditional_method.startswith("2d"):
                        center_3d = bev_pixel_to_3d(
                            patch_points=pc,
                            pixel_x=cur_det_center[0],
                            pixel_y=cur_det_center[1],
                            origin_x=meta["origin_x"],
                            origin_y=meta["origin_y"],
                            resolution=meta["resolution"],
                            search_radius=None,
                            tile_size=meta["tile_size"],
                            invert_y=False,
                            invert_x=False
                        )
                    else:
                        center_3d = cur_det_center

                    if center_3d is not None and len(center_3d) >= 3:
                        elapsed_time = time.perf_counter() - start_time
                        cur_center_point = {
                            "x": float(center_3d[0]),
                            "y": float(center_3d[1]),
                            "z": float(center_3d[2]),
                            "time": elapsed_time,
                            "confidence": min_confidence
                        }
                        raw_threshold_results_by_conf[min_confidence] = add_to_result(
                            raw_threshold_results_by_conf[min_confidence], cur_dataset, pc_id, cur_center_point
                        )

            # =================================================================
            # CASE B: 3D Models
            # =================================================================
            elif is_3D_model:
                # Load 3D predictions ONCE outside the confidence loop
                loaded_preds = np.load(pred_file_path, allow_pickle=True)
                loaded_dict = loaded_preds.item() if isinstance(loaded_preds, np.ndarray) else loaded_preds
                
                pts_3d_local = loaded_dict["points"]
                probs_1d = loaded_dict["probs"]
                centroid = loaded_dict["centroid"]
                scale = loaded_dict["scale"]

                xyz_local = pts_3d_local[:, :3]
                pts_3d_world = (xyz_local * scale) + centroid

                # Preload time file once if needed
                loaded_pred_time = np.load(pred_time_file_path).item() if (
                    process_3d_mode.lower() == "3d" or run_mode.lower() == "load"
                ) else None

                # Precompute 2D projection indices ONCE if in 2D mode
                if process_3d_mode.lower() == "2d":
                    orig_h = int(meta["tile_size"] / meta["resolution"])
                    orig_w = orig_h
                    center_x_pixel = orig_w / 2.0
                    center_y_pixel = orig_h / 2.0

                    offset_x = pts_3d_world[:, 0] - meta["origin_x"]
                    offset_y = pts_3d_world[:, 1] - meta["origin_y"]
                    
                    col_indices = np.floor((offset_x / meta["resolution"]) + center_x_pixel).astype(int)
                    row_indices = np.floor((offset_y / meta["resolution"]) + center_y_pixel).astype(int)
                    
                    valid_idx = (col_indices >= 0) & (col_indices < orig_w) & \
                                (row_indices >= 0) & (row_indices < orig_h)
                    
                    valid_rows = row_indices[valid_idx]
                    valid_cols = col_indices[valid_idx]
                    valid_probs = probs_1d[valid_idx]

                    base_grid_prob = np.zeros((orig_h, orig_w), dtype=np.float32)
                    np.maximum.at(base_grid_prob, (valid_rows, valid_cols), valid_probs)

                # Now iterate over all min_confidences in memory!
                for min_confidence in min_confidences:
                    if process_3d_mode.lower() == "3d":
                        start_time = time.perf_counter() - loaded_pred_time

                        mask = probs_1d >= min_confidence
                        valid_pts_world = pts_3d_world[mask]
                        labeled_preds = np.zeros(len(pts_3d_world), dtype=int)

                        if len(valid_pts_world) > 0:
                            dbscan = DBSCAN(eps=0.3, min_samples=candidate_min_points)
                            cluster_labels = dbscan.fit_predict(valid_pts_world)
                            valid_cluster_mask = cluster_labels != -1
                            labeled_preds[mask] = np.where(valid_cluster_mask, cluster_labels + 1, 0)
                            num_pred_objects = cluster_labels.max() + 1
                        else:
                            num_pred_objects = 0

                        preds_prob = probs_1d
                        total_prediction_time = time.perf_counter() - start_time
                        
                        for p_idx in range(1, num_pred_objects + 1):
                            additional_post_processing_time_start = time.perf_counter()
                            pred_mask = (labeled_preds == p_idx)
        
                            if np.sum(pred_mask) < candidate_min_points:
                                continue
                            
                            center_coords = extract_center_fn(
                                pred_mask=pred_mask,
                                prediction=preds_prob,
                                pts_3d=pts_3d_world,
                                known_radius_px=0.6
                            )
                            if center_coords is None:
                                continue

                            total_localization_time = time.perf_counter() - additional_post_processing_time_start
                            final_elapsed_time = total_prediction_time + total_localization_time
                            
                            cur_center_point = {
                                "x": float(center_coords[0]),
                                "y": float(center_coords[1]),
                                "z": float(center_coords[2]),
                                "time": final_elapsed_time,
                                "confidence": min_confidence
                            }
                            raw_threshold_results_by_conf[min_confidence] = add_to_result(
                                raw_threshold_results_by_conf[min_confidence], cur_dataset, pc_id, cur_center_point
                            )

                    elif process_3d_mode.lower() == "2d":
                        start_time = time.perf_counter()
                        grid_binary = np.zeros((orig_h, orig_w), dtype=np.uint8)
                        bin_mask_1d = (valid_probs >= min_confidence).astype(np.uint8)
                        np.maximum.at(grid_binary, (valid_rows, valid_cols), bin_mask_1d)
                        
                        preds_prob = base_grid_prob
                        preds_binary = grid_binary

                        if run_mode.lower() == "load":
                            start_time = time.perf_counter() - loaded_pred_time
                        elif run_mode.lower() == "save":
                            elapsed_prediction_time = time.perf_counter() - start_time
                            np.save(pred_time_file_path, np.array([elapsed_prediction_time]))
                            os.chmod(pred_time_file_path, 0o666)
                            continue

                        preds_closed = binary_closing(preds_binary, structure=closing_struct, iterations=8).astype(np.uint8)
                        labeled_preds, num_pred_objects = label(preds_closed)
                        extracted_pixel_centers = []
                        total_prediction_time = time.perf_counter() - start_time

                        for p_idx in range(1, num_pred_objects + 1):
                            additional_post_processing_time_start = time.perf_counter()
                            pred_mask = (labeled_preds == p_idx)

                            if np.sum(pred_mask) < candidate_min_points:
                                continue
                            
                            center_coords = extract_center_fn(
                                pred_mask=pred_mask,
                                prediction=preds_prob,
                                mask_to_polygon_fn=mask_to_polygon,
                                min_confidence_peak=min_confidence_peak,
                                known_radius_px=0.6
                            )
                            if center_coords is None:
                                continue

                            center_x, center_y = center_coords
                            extracted_pixel_centers.append((center_x, center_y))

                            center = bev_pixel_to_3d(
                                patch_points=pc,
                                pixel_x=center_x,
                                pixel_y=center_y,
                                origin_x=meta["origin_x"],
                                origin_y=meta["origin_y"],
                                resolution=meta["resolution"],
                                search_radius=None,
                                tile_size=meta["tile_size"],
                                invert_y=False,
                                invert_x=False
                            )
                            if center is None or len(center) < 3:
                                print(f"[Warning] Skipped a center prediction -> pred: {center}")
                                continue

                            total_localization_time = time.perf_counter() - additional_post_processing_time_start
                            final_elapsed_time = total_prediction_time + total_localization_time
                            
                            cur_center_point = {
                                "x": center[0],
                                "y": center[1],
                                "z": center[2],
                                "time": final_elapsed_time,
                                "confidence": min_confidence
                            }
                            raw_threshold_results_by_conf[min_confidence] = add_to_result(
                                raw_threshold_results_by_conf[min_confidence], cur_dataset, pc_id, cur_center_point
                            )

                        if enable_debug and min_confidence == mid_conf and (num_pred_objects > 0 or np.any(gt_labels_numpy > 0)):
                            plot_center_prediction_debug(
                                pixel_values=pixel_values,
                                labels=gt_labels_numpy,
                                labeled_preds=labeled_preds,
                                preds_prob=preds_prob,
                                preds_binary=preds_binary,
                                preds_closed=preds_closed,
                                extracted_centers=extracted_pixel_centers,
                                gt_centers=gt_centers,
                                pc_points=pc,
                                meta=meta,
                                pc_id=pc_id,
                                idx=idx,
                                save_dir=debug_save_dirs[min_confidence],
                                num_preds=num_pred_objects,
                                is_heatmap=False,
                            )
                    else:
                        raise ValueError(f"Unknown Processing Mode for 3D loaded predictions: '{process_3d_mode}'")

            # =================================================================
            # CASE C: 2D Deep Learning Models
            # =================================================================
            else:
                raw_start_time = time.perf_counter()

                # 1. Load or Predict ONCE per patch
                preds_squeezed, valid_mask, _, _, pixel_values = make_prediction(
                    pixel_values=pixel_values, 
                    model=model, 
                    model_name=model_name, 
                    labels=labels, 
                    ignore_index=ignore_index, 
                    using_heatmap_as_gt=using_heatmap_as_gt,
                    as_prob=True, 
                    manhole_class_idx=1,
                    save_raw_pred=run_mode.lower() == "save",
                    save_filepath=pred_file_path,
                    load_filepath=pred_file_path if run_mode.lower() == "load" else None
                )

                if run_mode.lower() == "save":
                    elapsed_prediction_time = time.perf_counter() - raw_start_time
                    np.save(pred_time_file_path, np.array([elapsed_prediction_time]))
                    os.chmod(pred_time_file_path, 0o666)
                    continue

                if run_mode.lower() == "load":
                    base_prediction_time = np.load(pred_time_file_path).item()
                else:
                    base_prediction_time = time.perf_counter() - raw_start_time

                # 2. Iterate over all confidence thresholds in memory
                for min_confidence in min_confidences:
                    
                    thresh_start_time = time.perf_counter()
                    preds_prob, preds_binary = apply_confidence_threshold(preds_squeezed, valid_mask, min_confidence)

                    preds_closed = binary_closing(preds_binary, structure=closing_struct, iterations=8).astype(np.uint8)
                    labeled_preds, num_pred_objects = label(preds_closed)

                    extracted_pixel_centers = []
                    total_prediction_time = base_prediction_time + (time.perf_counter() - thresh_start_time)

                    for p_idx in range(1, num_pred_objects + 1):
                        additional_post_processing_time_start = time.perf_counter()
                        pred_mask = (labeled_preds == p_idx)

                        if np.sum(pred_mask) < candidate_min_points:
                            continue
                        
                        center_coords = extract_center_fn(
                            pred_mask=pred_mask,
                            prediction=preds_prob,
                            mask_to_polygon_fn=mask_to_polygon,
                            min_confidence_peak=min_confidence_peak,
                            known_radius_px=0.6
                        )

                        if center_coords is None:
                            continue

                        center_x, center_y = center_coords
                        extracted_pixel_centers.append((center_x, center_y))

                        center = bev_pixel_to_3d(
                            patch_points=pc,
                            pixel_x=center_x,
                            pixel_y=center_y,
                            origin_x=meta["origin_x"],
                            origin_y=meta["origin_y"],
                            resolution=meta["resolution"],
                            search_radius=None,
                            tile_size=meta["tile_size"],
                            invert_y=False,
                            invert_x=False
                        )

                        if center is None or len(center) < 3:
                            print(f"[Warning] Skipped a center prediction -> pred: {center}")
                            continue

                        total_localization_time = time.perf_counter() - additional_post_processing_time_start
                        final_elapsed_time = total_prediction_time + total_localization_time
                        
                        cur_center_point = {
                            "x": center[0],
                            "y": center[1],
                            "z": center[2],
                            "time": final_elapsed_time,
                            "confidence": min_confidence
                        }
                        
                        raw_threshold_results_by_conf[min_confidence] = add_to_result(
                            raw_threshold_results_by_conf[min_confidence], cur_dataset, pc_id, cur_center_point
                        )

                    if enable_debug and min_confidence == mid_conf and (num_pred_objects > 0 or np.any(gt_labels_numpy > 0)):
                        plot_center_prediction_debug(
                            pixel_values=pixel_values,
                            labels=gt_labels_numpy,
                            labeled_preds=labeled_preds,
                            preds_prob=preds_prob,
                            preds_binary=preds_binary,
                            preds_closed=preds_closed,
                            extracted_centers=extracted_pixel_centers,
                            gt_centers=gt_centers,
                            pc_points=pc,
                            meta=meta,
                            pc_id=pc_id,
                            idx=idx,
                            save_dir=debug_save_dirs[min_confidence],
                            num_preds=num_pred_objects,
                            is_heatmap=False,
                        )

        if run_mode.lower() == "save":
            continue

        # 5. Merge nearby predictions per confidence (exactly as before)
        all_dataset_results = []
        for min_confidence in min_confidences:
            raw_threshold_results = raw_threshold_results_by_conf[min_confidence]
            print(f"Post-processing center predictions for {cur_dataset} (merging duplicates from patch overlaps).")
            
            threshold_merged_results = []

            for cur_result in raw_threshold_results:
                target_pc_id = cur_result["pointcloud-id"]
                target_confidence = min_confidence
                centers = cur_result.get("centers", [])

                if not centers:
                    continue

                for pt in centers:
                    if pt.get("confidence", -999) != target_confidence:
                        raise ValueError("Found another confidence inside of the current result.")

                pc_points = [
                    (pt["x"], pt["y"], pt["z"], pt["time"])
                    for pt in centers
                ]

                merged_pixel_centers = merge_nearby_centers(
                    pc_points, 
                    dist_threshold=0.8
                )

                for cur_new_pixel_center in merged_pixel_centers:
                    cur_center_point = {
                        "x": cur_new_pixel_center[0],
                        "y": cur_new_pixel_center[1],
                        "z": cur_new_pixel_center[2],
                        "time": cur_new_pixel_center[3],
                        "confidence": min_confidence
                    }
                    threshold_merged_results = add_to_result(
                        threshold_merged_results, cur_dataset, target_pc_id, cur_center_point
                    )
            
            print(f"Conf {min_confidence}: Reduced results from {len(raw_threshold_results)} to {len(threshold_merged_results)} ({len(raw_threshold_results)-len(threshold_merged_results)}).")

            for res in threshold_merged_results:
                target_pc_id = res["pointcloud-id"]
                new_centers = res.get("centers", [])

                # Fixed typo: was e.get("pointcloud_id") which never matched "pointcloud-id"
                existing_entry = next((e for e in all_dataset_results if e.get("pointcloud-id") == target_pc_id), None)

                if existing_entry is not None:
                    existing_entry["centers"].extend(new_centers)
                else:
                    all_dataset_results.append(res)

        # 6. Check if found center is in ignored manhole area
        test_raw_3d_dataset = get_data_loader(
            cur_dataset, 
            config.data.path if cur_dataset == "whu" else config.data.path_2, 
            type=data_split_type,
            transform=None,
            batch_size=1, 
            shuffle=False, 
            num_workers=4,
            preprocessed=False, 
            return_train_format=False,
            return_dataset=True,
            bev_normalized=config.data.normalization, 
            bev_normalize_mode=config.data.normalization_mode
        )
        all_dataset_results = filter_ignored_manholes(
            results=all_dataset_results,
            raw_dataset=test_raw_3d_dataset,
            box_size=0.9,
            ignore_label=255,
            ignore_threshold=0.80,
            check_z_axis=False
        )

        # 7. Save evaluation results per dataset
        output_path = Path(f"{config.center_eval.center_pred_path}/center_pred_{cur_dataset}_{data_split_type}_{exp_name}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as file_:
            json.dump(all_dataset_results, file_, indent=4)
        print(f"Saved Eval Center Results in: '{output_path}'")

        # # result list over all thresholds of one dataset
        # all_dataset_results = []

        # for min_confidence in min_confidences:
        #     raw_threshold_results = []

        #     debug_save_dir = Path(f"{config.center_eval.center_pred_path}/debug_plots/center_pred_{cur_dataset}_{data_split_type}_{exp_name}_{min_confidence}")
        #     # save_dir_creation(str(debug_save_dir))
        #     os.makedirs(str(debug_save_dir), exist_ok=True)
        #     shutil.rmtree(str(debug_save_dir))
        #     # save_dir_creation(str(debug_save_dir))
        #     os.makedirs(str(debug_save_dir), exist_ok=True)


        #     test_3d_dataset = get_data_loader(
        #         cur_dataset, 
        #         config.data.path if cur_dataset == "whu" else config.data.path_2, 
        #         type=data_split_type,
        #         transform=get_basic_transform(),
        #         batch_size=1, 
        #         shuffle=False, 
        #         num_workers=4,
        #         preprocessed=True, 
        #         return_train_format=True,
        #         return_dataset=True,
        #         bev_normalized=config.data.normalization, 
        #         bev_normalize_mode=config.data.normalization_mode
        #     )
            
        #     all_test_paths = test_3d_dataset.point_cloud_paths

        #     test_bev_dataset = BEVDataset(
        #         path=all_test_paths, 
        #         file_paths=[], 
        #         has_labels=True, 
        #         image_training=True, 
        #         preprocessor=processor,
        #         augment=False,
        #         pass_label_in_preprocessor=pass_label_in_preprocessor,
        #         heatmap_gt_path=heatmap_path,
        #         used_heatmap_channel=used_heatmap_channel,
        #         normalize=normalization, 
        #         normalization_mode=normalization_mode
        #     )

        #     all_pc_ids = set()

        #     for idx, cur_data_path in tqdm(enumerate(all_test_paths), total=len(all_test_paths), desc="2D Center Pipe"):
        #         pc_id, x_start, y_start = test_bev_dataset.extract_grid_identifier(cur_data_path)

        #         pred_file_path = os.path.join(pred_save_load_mode_saving_path, pred_save_load_mode_saving_name, f"{cur_dataset}_{pc_id}_{x_start}_{y_start}.npy")
        #         pred_time_file_path = os.path.join(pred_save_load_mode_saving_path, pred_save_load_mode_saving_name, f"{cur_dataset}_{pc_id}_{x_start}_{y_start}_TIME.npy")

        #         all_pc_ids.add(pc_id)

        #         # get data
        #         bev_dict = next(test_bev_dataset.get_patch_via_identifier(pc_id, x_start, y_start, return_generator=True))
        #         meta = bev_dict["meta"]
        #         pixel_values = bev_dict["pixel_values"]
        #         labels = bev_dict["labels"]

        #         gt_labels_numpy = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
                
        #         # pc = test_3d_dataset[idx]  # -> wrong patch?
        #         pc, pc_labels = test_3d_dataset.get_patch_via_identifier(pc_id, x_start, y_start)
        #         # [type(x) for x in pc]=[<class 'torch.Tensor'>, <class 'torch.Tensor'>]
        #         # [x.shape for x in pc]=[torch.Size([424, 4]), torch.Size([424, 1])]
        #         # print(f"{type(pc)=}")
        #         # print(f"{[type(x) for x in pc]=}")
        #         # print(f"{[x.shape for x in pc]=}")

        #         pixel_values = pixel_values.unsqueeze(0)

        #         # if enable_debug:
        #         #     gt_centers = extract_and_match_centers(
        #         #         dataset_name=cur_dataset, 
        #         #         pc_id=pc_id, 
        #         #         gt_json_file_paths=gt_json_file_paths, 
        #         #         queried_points_3d=None, 
        #         #         gt_patch_labels=gt_labels_numpy, 
        #         #         meta=meta, 
        #         #         pc=pc,  # pixel_values, 
        #         #         min_points=10
        #         #     )
        #         gt_centers = None

        #         # Start timing the prediction/extraction for this sample
        #         start_time = time.perf_counter()

        #         # -----------------
        #         # Manhole Search
        #         extra_debug_info = []

        #         if model_name.lower().startswith("traditional_"):
        #             traditional_method = "_".join(model_name.split("_")[1:])

        #             # Call unified prediction method using direct PyTorch Tensors
        #             detected_centers, debug_dict = predict_manhole_centers(
        #                 pts_3d=pc,
        #                 bev_img=pixel_values,
        #                 method=traditional_method,
        #                 meta=meta
        #             )

        #             # Debuggin
        #             if enable_debug and debug_dict is not None:
        #                 plot_center_prediction_debug(
        #                     pixel_values=pixel_values,
        #                     labels=gt_labels_numpy,
        #                     labeled_preds=None,                         # np instance masks
        #                     preds_prob=debug_dict['prob_map'],          # e.g., Heatmap/Blur/Intensity
        #                     preds_binary=debug_dict['binary_mask'],     # e.g., Canny Edges / Otsu
        #                     preds_closed=None, 
        #                     extracted_centers=detected_centers[:, :2],  # Found centers
        #                     gt_centers=gt_centers,
        #                     pc_points=pc,
        #                     meta=meta,
        #                     pc_id=pc_id,
        #                     idx=idx,
        #                     save_dir=debug_save_dir,
        #                     num_preds=len(detected_centers),
        #                     is_heatmap=False,                            # Creates crosses for centers
        #                     extra_debug_info=debug_dict['geometries']   # Creates Circles and Polygones in Panel 5
        #                 )

        #             # Process predictions depending on 2D vs 3D method output
        #             for cur_det_center in detected_centers:
        #                 if traditional_method.startswith("2d"):
        #                     # det is [pixel_x, pixel_y, radius] -> transform to 3D point
        #                     center_3d = bev_pixel_to_3d(
        #                         patch_points=pc,
        #                         pixel_x=cur_det_center[0],
        #                         pixel_y=cur_det_center[1],
        #                         origin_x=meta["origin_x"],
        #                         origin_y=meta["origin_y"],
        #                         resolution=meta["resolution"],
        #                         search_radius=None,
        #                         tile_size=meta["tile_size"],
        #                         invert_y=False,
        #                         invert_x=False
        #                     )
        #                 else:
        #                     # det is already 3D coordinates [x, y, z]
        #                     center_3d = cur_det_center

        #                 if center_3d is not None and len(center_3d) >= 3:
        #                     elapsed_time = time.perf_counter() - start_time
        #                     cur_center_point = {
        #                         "x": float(center_3d[0]),
        #                         "y": float(center_3d[1]),
        #                         "z": float(center_3d[2]),
        #                         "time": elapsed_time,
        #                         "confidence": min_confidence
        #                     }
        #                     raw_threshold_results = add_to_result(raw_threshold_results, cur_dataset, pc_id, cur_center_point)
        #         else:
                    
        #             # 3D Model handling
        #             if is_3D_model:
        #                 loaded_preds = np.load(pred_file_path, allow_pickle=True)
        #                 loaded_dict = loaded_preds.item() if isinstance(loaded_preds, np.ndarray) else loaded_preds
                        
        #                 pts_3d_local = loaded_dict["points"]  # The exact points used in the model [N, 3]
        #                 probs_1d = loaded_dict["probs"] # The probabilities [N]
        #                 centroid = loaded_dict["centroid"]  # .get("centroid", np.zeros(3))
        #                 scale = loaded_dict["scale"]

        #                 # restore absolut world-coordinates
        #                 # WICHTIG: Falls pts_3d_local 4 Kanäle hat (X, Y, Z, Intensity), 
        #                 # dürfen wir NUR X, Y und Z skalieren!
        #                 xyz_local = pts_3d_local[:, :3]
                        
        #                 # Restore absolut world-coordinates (DENORMALIZATION)
        #                 # 1. Multiplikation mit scale (Rückgängig machen der Einheitskugel)
        #                 # 2. Addition des Centroids (Rückgängig machen der Zentrierung)
        #                 pts_3d_world_xyz = (xyz_local * scale) + centroid

        #                 # Falls du Intensity oder andere Features für deine Extraction brauchst, 
        #                 # füge sie hier wieder an. Da deine extract_3d_center-Funktionen 
        #                 # aber nur Koordinaten brauchen, reicht xyz:
        #                 pts_3d_world = pts_3d_world_xyz

        #                 if process_3d_mode.lower() == "2d":
        #                     # --- Projection into 2D ---
        #                     orig_h = int(meta["tile_size"] / meta["resolution"])
        #                     orig_w = orig_h
                            
        #                     grid_prob = np.zeros((orig_h, orig_w), dtype=np.float32)
        #                     grid_binary = np.zeros((orig_h, orig_w), dtype=np.uint8)
                            
        #                     # Note: Since our points were normalized via mean-centering in the dataset,
        #                     # we have to map them to the 2D grid relative to the patch center.
        #                     # Patch center in BEV coordinates is exactly at (orig_w/2, orig_h/2).
        #                     center_x_pixel = orig_w / 2.0
        #                     center_y_pixel = orig_h / 2.0

        #                     offset_x = pts_3d_world[:, 0] - meta["origin_x"]
        #                     offset_y = pts_3d_world[:, 1] - meta["origin_y"]
                            
        #                     # Convert local centered coordinates to pixels
        #                     col_indices = np.floor((offset_x / meta["resolution"]) + center_x_pixel).astype(int)
        #                     row_indices = np.floor((offset_y / meta["resolution"]) + center_y_pixel).astype(int)
                            
        #                     valid_idx = (col_indices >= 0) & (col_indices < orig_w) & \
        #                                 (row_indices >= 0) & (row_indices < orig_h)
                            
        #                     # Thresholding for binary mask
        #                     bin_mask_1d = (probs_1d >= min_confidence).astype(np.uint8)
                            
        #                     # Project into 2D grids using maximum values to handle overlapping points
        #                     np.maximum.at(grid_prob, (row_indices[valid_idx], col_indices[valid_idx]), probs_1d[valid_idx])
        #                     np.maximum.at(grid_binary, (row_indices[valid_idx], col_indices[valid_idx]), bin_mask_1d[valid_idx])
                            
        #                     preds_prob = grid_prob
        #                     preds_binary = grid_binary
                            
        #                     # The rest of our existing 2D pipeline (binary_closing, label, extract_center_fn)
        #                     # will seamlessly take over from here :-)
        #                 elif process_3d_mode.lower() == "3d":
        #                     # --- Native 3D Processing ---
        #                     # can't use the 2D pipeline, so have to make a own
        #                     # start the timer
        #                     elapsed_prediction_time = np.load(pred_time_file_path).item()
        #                     start_time = time.perf_counter() - elapsed_prediction_time

        #                     # Bypass the 2D morphology. We extract the 3D center directly.
                            
        #                     # 1. Filter points by confidence threshold
        #                     mask = probs_1d >= min_confidence
        #                     valid_pts_world = pts_3d_world[mask]

        #                     # 2. Extract manhole candidates using DBSCAN
        #                     # Initialize array with 0 (background/ignore) to match 2D morphology output
        #                     labeled_preds = np.zeros(len(pts_3d_world), dtype=int)

        #                     if len(valid_pts_world) > 0:
        #                         dbscan = DBSCAN(eps=0.3, min_samples=candidate_min_points)  # 20
        #                         cluster_labels = dbscan.fit_predict(valid_pts_world)
                                
        #                         # remove noise data
        #                         valid_cluster_mask = cluster_labels != -1
                                
        #                         # Place the 1-indexed cluster IDs back into the full-size array
        #                         # (Only for the valid points that made it past the threshold mask)
        #                         labeled_preds[mask] = np.where(
        #                             valid_cluster_mask, 
        #                             cluster_labels + 1, 
        #                             0
        #                         )
                                
        #                         num_pred_objects = cluster_labels.max() + 1
        #                     else:
        #                         num_pred_objects = 0

        #                     preds_prob = probs_1d
        #                     total_prediction_time = time.perf_counter() - start_time
                            
        #                     # 3. Extract Centers
        #                     for p_idx in range(1, num_pred_objects + 1):
        #                         additional_post_processing_time_start = time.perf_counter()
                                
        #                         # bool mask for current candidate / cluster
        #                         pred_mask = (labeled_preds == p_idx)
        
        #                         if np.sum(pred_mask) < candidate_min_points:
        #                             continue
                                
        #                         # Extract center using the selected strategy function
        #                         center_coords = extract_center_fn(
        #                             pred_mask=pred_mask,
        #                             prediction=preds_prob,
        #                             pts_3d=pts_3d_world
        #                         )
        
        #                         if center_coords is None:
        #                             continue

        #                         global_cx = center_coords[0]
        #                         global_cy = center_coords[1]
        #                         global_cz = center_coords[2]

        #                         # # Convert local (zero-centered) coordinates back to global world coordinates
        #                         # global_patch_center_x = meta["origin_x"] + (meta["tile_size"] / 2.0)
        #                         # global_patch_center_y = meta["origin_y"] + (meta["tile_size"] / 2.0)
                                
        #                         # global_cx = global_patch_center_x + local_cx
        #                         # global_cy = global_patch_center_y + local_cy
        #                         # global_cz = local_cz 
                                
        #                         total_localization_time = time.perf_counter() - additional_post_processing_time_start
        #                         final_elapsed_time = total_prediction_time + total_localization_time
                                
        #                         cur_center_point = {
        #                             "x": float(global_cx),
        #                             "y": float(global_cy),
        #                             "z": float(global_cz),
        #                             "time": final_elapsed_time,
        #                             "confidence": min_confidence
        #                         }
        #                         raw_threshold_results = add_to_result(raw_threshold_results, cur_dataset, pc_id, cur_center_point)

        #                     # must skip the 2D processing!
        #                     continue
        #                 else:
        #                     raise ValueError(f"Unknown Processing Mode for 3D loaded predictions: '{process_3d_mode}'")
        #             else:
        #                 # _ -> gt_labels_numpy
        #                 preds_prob, preds_binary, valid_mask, preds, _, pixel_values = make_prediction(
        #                     pixel_values=pixel_values, 
        #                     model=model, 
        #                     model_name=model_name, 
        #                     labels=labels, 
        #                     min_confidence=min_confidence, 
        #                     ignore_index=ignore_index, 
        #                     using_heatmap_as_gt=using_heatmap_as_gt,
        #                     as_prob=True, 
        #                     manhole_class_idx=1,
        #                     save_raw_pred=run_mode.lower() == "save",
        #                     save_filepath=pred_file_path,
        #                     load_filepath=pred_file_path if run_mode.lower() == "load" else None
        #                 )

    
        #             # use time loading if using loaded models or save time if in saving mode
        #             # correct time if loading the model
        #             if run_mode.lower() == "load":
        #                 elapsed_prediction_time = np.load(pred_time_file_path).item()
        #                 start_time = time.perf_counter() - elapsed_prediction_time
        #             elif run_mode.lower() == "save":
        #                 elapsed_prediction_time = time.perf_counter() - start_time
        #                 np.save(pred_time_file_path, np.array([elapsed_prediction_time]))
        #                 os.chmod(pred_time_file_path, 0o666)
        #                 continue

        #             orig_h, orig_w = int(meta["tile_size"] / meta["resolution"]), int(meta["tile_size"] / meta["resolution"])
        #             pred_h, pred_w = preds_binary.shape
                        
        #             struct = generate_binary_structure(2, 2)  # 8-Nachbarschaft
        #             # bigger neighborhood against a problem, where multiple predictions are made due to little riffles
        #             preds_closed = binary_closing(preds_binary, structure=struct, iterations=8).astype(np.uint8)
        #             labeled_preds, num_pred_objects = label(preds_closed)

        #             extracted_pixel_centers = []
        #             # gt_pixel_centers = []

        #             # ONLY FOR DEBUGGING:
        #             # # extract GT center from labels
        #             # # labels[labels == ignore_index] = 0
        #             # labels_clean = np.where(gt_labels_numpy == ignore_index, 0, gt_labels_numpy)
        #             # # close gaps
        #             # labels_binary = (labels_clean >= min_confidence).astype(np.uint8)
        #             # labels_binary = np.squeeze(labels_binary)
        #             # struct = generate_binary_structure(2, 2)
        #             # labels_closed = binary_closing(labels_binary, structure=struct, iterations=8).astype(np.uint8)

        #             # if np.any(labels_closed > 0):
        #             #     gt_labeled, num_gt = label(labels_closed)
        #             #     for g_i in range(1, num_gt + 1):
        #             #         gt_mask = (gt_labeled == g_i)
                            
        #             #         # Skip small noise components with less than 4 pixels
        #             #         if np.count_nonzero(gt_mask) < 20:
        #             #             continue
                                
        #             #         g_coords = extract_center_fn(
        #             #             pred_mask=gt_mask,
        #             #             prediction=labels_closed if not using_heatmap_as_gt else labels_clean.astype(np.float32),
        #             #             mask_to_polygon_fn=mask_to_polygon,
        #             #         )
        #             #         if g_coords is not None:
        #             #             gt_pixel_centers.append(g_coords)
                    
        #             # -----------------
        #             # Prediction Center Extraction
                    
        #             total_prediction_time = time.perf_counter() - start_time

        #             for p_idx in range(1, num_pred_objects + 1):
        #                 additional_post_processing_time_start = time.perf_counter()
        #                 pred_mask = (labeled_preds == p_idx)

        #                 if np.sum(pred_mask) < candidate_min_points:
        #                     continue
                        
        #                 # Extract center using the selected strategy function
        #                 center_coords = extract_center_fn(
        #                     pred_mask=pred_mask,
        #                     prediction=preds_prob,
        #                     mask_to_polygon_fn=mask_to_polygon,
        #                     min_confidence_peak=min_confidence_peak
        #                 )

        #                 if center_coords is None:
        #                     continue

        #                 center_x, center_y = center_coords
        #                 extracted_pixel_centers.append((center_x, center_y))

        #                 # Transform 2D pixel center to 3D point
        #                 center = bev_pixel_to_3d(
        #                     patch_points=pc,
        #                     pixel_x=center_x,
        #                     pixel_y=center_y,
        #                     origin_x=meta["origin_x"],
        #                     origin_y=meta["origin_y"],
        #                     resolution=meta["resolution"],
        #                     search_radius=None,
        #                     tile_size=meta["tile_size"],
        #                     invert_y=False,
        #                     invert_x=False
        #                 )

        #                 # print(f"Center shape: {center.shape}")
        #                 if center is None or len(center) < 3:
        #                     print(f"[Warning] Skipped a center prediction -> pred: {center}")
        #                     continue

        #                 total_localization_time = time.perf_counter() - additional_post_processing_time_start
        #                 final_elapsed_time = total_prediction_time + total_localization_time
                        
        #                 cur_center_point = {
        #                     "x": center[0],
        #                     "y": center[1],
        #                     "z": center[2],
        #                     "time": final_elapsed_time,
        #                     "confidence": min_confidence
        #                 }
                        
        #                 raw_threshold_results = add_to_result(raw_threshold_results, cur_dataset, pc_id, cur_center_point)

        #             # --- Execute Debug Plotting ---
        #             if run_mode.lower() != "save":
        #                 if enable_debug and (num_pred_objects > 0 or np.any(gt_labels_numpy > 0)):
        #                     plot_center_prediction_debug(
        #                         pixel_values=pixel_values,
        #                         labels=gt_labels_numpy,
        #                         labeled_preds=labeled_preds,
        #                         preds_prob=preds_prob,
        #                         preds_binary=preds_binary,
        #                         preds_closed=preds_closed,
        #                         extracted_centers=extracted_pixel_centers,
        #                         gt_centers=gt_centers,
        #                         pc_points=pc,
        #                         meta=meta,
        #                         pc_id=pc_id,
        #                         idx=idx,
        #                         save_dir=debug_save_dir,
        #                         num_preds=num_pred_objects,
        #                         is_heatmap=False,                            # Creates crosses for centers
        #                         # extra_debug_info=debug_dict['geometries']   # Creates Circles and Polygones in Panel 5
        #                     )

        #     # early stop, if only making savings
        #     if run_mode.lower() == "save":
        #         continue

        #     # -----------------
        #     # Post Processing
        #     # Merge nearby predictions
        #     print(f"Post-processing center predictions for {cur_dataset} (merging duplicates from patch overlaps).")
            
        #     threshold_merged_results = []

        #     for cur_result in raw_threshold_results:
        #         target_pc_id = cur_result["pointcloud-id"]
        #         target_confidence = min_confidence
        #         centers = cur_result.get("centers", [])

        #         if not centers:
        #             continue

        #         for pt in centers:
        #             if pt.get("confidence", -999) != target_confidence:
        #                 raise ValueError(f"Found another confidence inside of the current result.")

        #         # extract points & confidence
        #         pc_points = [
        #             (pt["x"], pt["y"], pt["z"], pt["time"])
        #             for pt in centers
        #         ]

        #         # merging
        #         merged_pixel_centers = merge_nearby_centers(
        #             pc_points, 
        #             dist_threshold=0.8
        #         )

        #         # rebuild merged version
        #         for cur_new_pixel_center in merged_pixel_centers:
        #             cur_center_point = {
        #                 "x": cur_new_pixel_center[0],
        #                 "y": cur_new_pixel_center[1],
        #                 "z": cur_new_pixel_center[2],
        #                 "time": cur_new_pixel_center[3],
        #                 "confidence": min_confidence
        #             }
                    
        #             threshold_merged_results = add_to_result(threshold_merged_results, cur_dataset, target_pc_id, cur_center_point)
            
        #     print(f"Conf {min_confidence}: Reduced results from {len(raw_threshold_results)} to {len(threshold_merged_results)} ({len(raw_threshold_results)-len(threshold_merged_results)}).")

        #     # add new found manhole centers of this threshold to the overall results
        #     # all_dataset_results.extend(threshold_merged_results)
        #     for res in threshold_merged_results:
        #         target_pc_id = res["pointcloud-id"]
        #         new_centers = res.get("centers", [])

        #         # Find if an entry for this pc_id already exists in all_dataset_results
        #         existing_entry = next((e for e in all_dataset_results if e.get("pointcloud-id") == target_pc_id), None)

        #         if existing_entry is not None:
        #             existing_entry["centers"].extend(new_centers)
        #         else:
        #             all_dataset_results.append(res)

        # if run_mode.lower() == "save":
        #     continue

        # # Check if found center in ignored manhole
        # # if prediction is in 255 label area of the 3D 
        # # -> if 0.45m box around is 80% or more ignore index, then not pass the prediction
        # # Check if found center is in ignored manhole area
        # test_raw_3d_dataset = get_data_loader(
        #     cur_dataset, 
        #     config.data.path if cur_dataset == "whu" else config.data.path_2, 
        #     type=data_split_type,
        #     transform=None,  # get_basic_transform(),
        #     batch_size=1, 
        #     shuffle=False, 
        #     num_workers=4,
        #     preprocessed=False, 
        #     return_train_format=False,
        #     return_dataset=True,
        #     bev_normalized=config.data.normalization, 
        #     bev_normalize_mode=config.data.normalization_mode
        # )
        # all_dataset_results = filter_ignored_manholes(
        #     results=all_dataset_results,
        #     raw_dataset=test_raw_3d_dataset,
        #     box_size=0.9,
        #     ignore_label=255,
        #     ignore_threshold=0.80,
        #     check_z_axis=False  # Set to True if Z coordinate should also be bounded by 0.6m
        # )

        # # -----------------
        # # Saving
        # # Save evaluation results per dataset (and over all thresholds)
        # output_path = Path(f"{config.center_eval.center_pred_path}/center_pred_{cur_dataset}_{data_split_type}_{exp_name}.json")
        # output_path.parent.mkdir(parents=True, exist_ok=True)
        # with open(output_path, "w") as file_:
        #     json.dump(all_dataset_results, file_, indent=4)
        # print(f"Saved Eval Center Results in: '{output_path}'")



def main(config):
    # model 
    center_eval(config)






