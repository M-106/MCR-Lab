# -----------
# > Imports <
# -----------
import numpy as np
import cv2

import torch

from scipy.ndimage import generate_binary_structure, grey_closing, iterate_structure, maximum_filter

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.morphology import disk, opening, closing, remove_small_objects

import matplotlib.pyplot as plt



# --------------
# > 2D METHODS <
# --------------
def method_2d_contour_circularity(bev_img, resolution=0.01, origin=(0.0, 0.0)):
    """
    Robuste Schachtdeckel-Erkennung via Grauwert-Morphologie, 
    Schwellenwertbildung, Kontursuche und Zirkularitätsfilterung.
    """
    intensity = bev_img[1].copy()
    if intensity.max() <= intensity.min():
        return (np.empty((0, 3)), None)

    # 1. Normalisierung auf 0-255 uint8
    p_low, p_high = np.percentile(intensity, (2, 98))
    if p_high > p_low:
        clipped = np.clip(intensity, p_low, p_high)
        norm_int = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    else:
        norm_int = np.zeros_like(intensity, dtype=np.uint8)

    # 2. Strukturelement für Schließen (Gaps in Deckeln schließen)
    struct = generate_binary_structure(2, 2)
    large_footprint = iterate_structure(struct, 4) # z.B. passend zur Größe skalieren
    closed_image = grey_closing(norm_int, footprint=large_footprint)

    # 3. Leichtes Weichzeichnen gegen Rauschen
    blurred = cv2.GaussianBlur(closed_image, (5, 5), 1.5)

    # 4. Binarisierung (Otsu-Thresholding oder Adaptiv)
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Konturen finden
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_r_px = int(0.25 / resolution)  # 25 px für 0.5m Durchmesser
    max_r_px = int(0.45 / resolution)  # 45 px für 0.9m Durchmesser
    min_area = np.pi * (min_r_px ** 2)
    max_area = np.pi * (max_r_px ** 2)

    results = []
    extra_info = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Grobe Vorfilterung über die Fläche
        if not (min_area <= area <= max_area):
            continue

        # Zirkularität prüfen: 4 * pi * Area / (Perimeter^2)
        # Ein perfekter Kreis hat eine Zirkularität von 1.0. 
        # Unrunde Objekte haben deutlich kleinere Werte (z.B. < 0.7).
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # Nur runde Objekte zulassen (z.B. Zirkularität > 0.75)
        if circularity < 0.75:
            continue

        # Exakten minimalen umschriebenen Kreis berechnen
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        px, py, r_px = int(x), int(y), int(radius)

        # Radius-Check nach dem Einpassen
        if not (min_r_px <= r_px <= max_r_px):
            continue

        results.append([px, py, r_px])
        extra_info.append({
            'type': 'circle',
            'x': px,
            'y': py,
            'r': r_px
        })

    debug_dict = {
        'prob_map': blurred,
        'binary_mask': binary_mask,
        'geometries': extra_info
    }

    if len(results) == 0:
        return (np.empty((0, 3)), debug_dict)

    # Optional: Maximale Unterdrückung / Überlappungsbereinigung beibehalten
    filtered_results = suppress_overlapping_circles(results, intensity, resolution)

    return (np.array(filtered_results) if len(filtered_results) > 0 else np.empty((0, 3)), debug_dict)


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
    # norm_int = cv2.normalize(intensity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # norm_int = ((intensity - intensity.min()) / (intensity.max() - intensity.min()) * 255).astype(np.uint8)
    # filtered = cv2.bilateralFilter(norm_int, d=5, sigmaColor=75, sigmaSpace=75)
    p_low, p_high = np.percentile(intensity, (2, 98))
    if p_high > p_low:
        clipped = np.clip(intensity, p_low, p_high)
        norm_int = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    else:
        norm_int = np.zeros_like(intensity, dtype=np.uint8)

    # plt.imshow(norm_int, cmap="plasma")
    # plt.title("Input Image (Normalized)")
    # plt.axis("off")
    # plt.show()
    # plt.close()

    # 1. Footprint/Structure (connectivity)
    # struct = generate_binary_structure(2, 2)
    # large_footprint = iterate_structure(struct, 8)
    # from skimage.morphology import closing, disk
    # closed_image = closing(image, footprint=disk(8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # # 2. Grayscale closing (dilation followed by erosion using local max/min)
    # closed_image = grey_closing(norm_int, footprint=struct)
    # remove noise
    opened = cv2.morphologyEx(norm_int, cv2.MORPH_OPEN, kernel)
    # close gaps on the circle edge
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # plt.imshow(closed_image, cmap="plasma")
    # plt.title("Closed Image")
    # plt.axis("off")
    # plt.show()
    # plt.close()

    # enhance contrast
    # clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    # enhanced = clahe.apply(closed_image)

    # plt.imshow(enhanced, cmap="plasma")
    # plt.title("Enhanced Image")
    # plt.axis("off")
    # plt.show()
    # plt.close()

    # Use Median Blur to remove speckle noise on asphalt -> smooth gradients for canny
    # blurred = cv2.medianBlur(enhanced, 5)
    # blurred = cv2.GaussianBlur(norm_int, (3, 3), 1.0)
    # blurred = cv2.GaussianBlur(closed_image, (3, 3), 1.2)
    # blurred = enhanced
    blurred = cv2.GaussianBlur(closed, (5, 5), 1.5)

    # plt.imshow(blurred, cmap="plasma")
    # plt.title("Blurred Image")
    # plt.axis("off")
    # plt.show()
    # plt.close()

    min_r_px = int(0.25 / resolution)  # 25 px for 0.5m diameter
    max_r_px = int(0.45 / resolution)  # 45 px for 0.9m diameter
    min_dist_px = int(1.0 / resolution)  # distance from circles

    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.0,  # 1.5,   # decides resolution of accumulator relative to current resolution
        minDist=min_dist_px,
        param1=50,   # canny edge value threshold -> how strong the edge have to be, as gradient value not pixel value -> maybe 1020 max
        param2=40,  # threshold of voting for hit
        minRadius=min_r_px, 
        maxRadius=max_r_px
    )

    if circles is None:
        return (np.empty((0, 3)), None)

    print("Circles: ", circles)

    # Collect Debug Info
    debug_dict = None
    # Compute canny for visualization -> hough transform use it
    edges = cv2.Canny(blurred, 50, 60) 

    # plt.imshow(edges, cmap="plasma")
    # plt.title("Canny Edges")
    # plt.axis("off")
    # plt.show()
    # plt.close()
    
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




# def method_2d_hough_skimage(bev_img, resolution=0.01, origin=(0.0, 0.0)):
#     """
#     2D Detection via Bilateral Filtering & Hough Circle Transform on Intensity.

#     Hough Circles does detect edges via canny and then Hough Transform votes for center points.
#     If a circle gets enough votes, it is considers a hit.

#     In Hough Transform every edge point votes for potential center points. 
#     Hough Transform uses for every edge pixel the direction of the strongest brightness increase (a gradient) 
#     and this gradient shows in direction of the center-point. Sometimes the center also can be on the opposite direction,
#     when the object is brighter than the background and because of this, OpenCV creates votes for both directions. 
#     The edge point votes now only for all points in the min-max radius in the gradient direction (both directions). 
#     Every edge point does that and in the end we get a vote-map which the accumulator uses.

#     > The gradients reduces the search-area massively.

#     Image
#      ↓
#     Gaussian Blur
#      ↓
#     Canny Edge Detector
#      ↓
#     Gradient Direction (via edge direction)
#      ↓
#     Hough Voting
#      ↓
#     Circles
#     """
#     intensity = bev_img[1].copy()
#     if intensity.max() <= intensity.min():
#         return (np.empty((0, 3)), None)

#     # Normalize to 0-255 uint8
#     # norm_int = cv2.normalize(intensity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     # norm_int = ((intensity - intensity.min()) / (intensity.max() - intensity.min()) * 255).astype(np.uint8)
#     # filtered = cv2.bilateralFilter(norm_int, d=5, sigmaColor=75, sigmaSpace=75)
#     p_low, p_high = np.percentile(intensity, (2, 98))
#     if p_high > p_low:
#         clipped = np.clip(intensity, p_low, p_high)
#         norm_int = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
#     else:
#         norm_int = np.zeros_like(intensity, dtype=np.uint8)

#     # blurring
#     blurred = cv2.GaussianBlur(norm_int, (5, 5), 1.5)

#     # morph operations to clean input image -> remove noise + close gaps
#     binary_or_edges = canny(blurred, sigma=2.0)
#     # Oder du schließt Lücken im Kantenbild direkt:
#     closed_edges = closing(binary_or_edges, disk(3))

#     # set circle settings
#     min_r_px = int(0.25 / resolution)
#     max_r_px = int(0.45 / resolution)
#     hough_radii = np.arange(min_r_px, max_r_px, 1)

#     # Apply Hough-Kreis-Transformation on the edge-image
#     hough_res = hough_circle(closed_edges, hough_radii)

#     # Find peaks
#     # num_peaks controls how many max finds
#     accums, x_centers, y_centers, radii = hough_circle_peaks(
#         hough_res, hough_radii, total_num_peaks=2, threshold=0.4
#     )

#     if len(x_centers) == 0:
#         return (np.empty((0, 3)), None)

#     # print("Detected Circles (skimage):", list(zip(x_centers, y_centers, radii)))

#     # Collect Debug Info
#     extra_info = []
#     for px, py, r_px in zip(x_centers, y_centers, radii):
#         extra_info.append({
#             'type': 'circle',
#             'x': int(px),
#             'y': int(py),
#             'r': int(r_px)
#         })

#     # Edge-Image for debugging
#     edges_uint8 = (closed_edges * 255).astype(np.uint8)

#     debug_dict = {
#         'prob_map': blurred,        # Verschwommenes Grauwertbild
#         'binary_mask': edges_uint8, # Geschlossene Kanten
#         'geometries': extra_info
#     }

#     # Compute Final Results
#     results = []
#     img_h, img_w = bev_img.shape[1], bev_img.shape[2]

#     for px, py, r_px in zip(x_centers, y_centers, radii):
#         if 0 <= py < img_h and 0 <= px < img_w:
#             results.append([int(px), int(py), int(r_px)])

#     if len(results) == 0:
#         return (np.empty((0, 3)), debug_dict)

#     # Apply Maximum Suppression to clean up redundant overlapping circles
#     filtered_results = suppress_overlapping_circles(results, intensity, resolution)

#     return (np.array(filtered_results), debug_dict) if len(filtered_results) > 0 else (np.empty((0, 3)), debug_dict)





def method_2d_hough_skimage(bev_img, resolution=0.01, origin=(0.0, 0.0)):
    """
    High-Precision 2D Detection via Canny Edge Filtering, Skimage Hough Transform,
    and Gradient/Intensity Post-Verification to eliminate False Positives.
    """
    intensity = bev_img[1].copy()
    if intensity.max() <= intensity.min():
        return (np.empty((0, 3)), None)

    img_h, img_w = intensity.shape

    # 1. Normalize to 0-255 uint8
    p_low, p_high = np.percentile(intensity, (2, 98))
    if p_high > p_low:
        clipped = np.clip(intensity, p_low, p_high)
        norm_int = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    else:
        norm_int = np.zeros_like(intensity, dtype=np.uint8)

    # 2. Close small gaps in GRAYSCALE before Canny (instead of thickening binary edges!)
    # This connects sparse LiDAR points without turning thin Canny edges into thick blobs.
    morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed_gray = cv2.morphologyEx(norm_int, cv2.MORPH_CLOSE, morph_k)
    blurred = cv2.GaussianBlur(closed_gray, (5, 5), 1.5)

    # 3. Strict Canny Edge Detection (only keep strong intensity transitions)
    # Using quantile-based hysteresis thresholds prevents noisy road texture from becoming edges
    edges = canny(
        blurred, 
        sigma=2.0, 
        low_threshold=0.15, 
        high_threshold=0.35, 
        use_quantiles=False
    )

    # Remove tiny isolated speckle edges (< 15 pixels long) so they cannot vote in Hough space
    clean_edges = remove_small_objects(edges, min_size=15, connectivity=2)

    if not np.any(clean_edges):
        return (np.empty((0, 3)), None)

    # 4. Set circle radius settings
    min_r_px = int(0.25 / resolution)  # 25 px
    max_r_px = int(0.45 / resolution)  # 45 px
    hough_radii = np.arange(min_r_px, max_r_px + 1, 2)  # Step=2 speeds up & reduces redundant scale hits

    # 5. Apply Hough Circle Transform on thin, cleaned edges
    hough_res = hough_circle(clean_edges, hough_radii, normalize=True)

    # Enforce spatial separation between peaks (~0.8m) so peaks don't cluster on the same spot
    min_sep_px = max(5, int(0.8 / resolution))

    accums, x_centers, y_centers, radii = hough_circle_peaks(
        hough_res, 
        hough_radii, 
        min_xdistance=min_sep_px,
        min_ydistance=min_sep_px,
        threshold=0.42,       # Stricter accumulator vote ratio on thin edges
        total_num_peaks=5     # Gather top 5 candidates, then verify them strictly below
    )

    if len(x_centers) == 0:
        return (np.empty((0, 3)), None)

    # Precompute image gradients to verify that edges actually curve around the circle center
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

    results = []
    extra_info = []

    # 6. Post-Hough False Positive Filter (Radial Gradient & Interior Contrast Verification)
    for accum_score, px, py, r_px in zip(accums, x_centers, y_centers, radii):
        px, py, r_px = int(px), int(py), int(r_px)
        if not (0 <= py < img_h and 0 <= px < img_w):
            continue

        # Extract local patch around candidate
        pad = int(r_px * 1.6)
        y1, y2 = max(0, py - pad), min(img_h, py + pad + 1)
        x1, x2 = max(0, px - pad), min(img_w, px + pad + 1)

        y_grid, x_grid = np.ogrid[y1:y2, x1:x2]
        dx = (x_grid - px).astype(np.float32)
        dy = (y_grid - py).astype(np.float32)
        dist = np.sqrt(dx**2 + dy**2) + 1e-6

        # Mask A: Perimeter band where the circle edge sits
        rim_mask = (dist >= r_px - 2.5) & (dist <= r_px + 2.5)
        # Mask B: Inner disc vs Outer surrounding asphalt ring
        inner_mask = dist <= (r_px * 0.85)
        outer_mask = (dist >= (r_px * 1.15)) & (dist <= (r_px * 1.50))

        if np.sum(inner_mask) == 0 or np.sum(outer_mask) == 0 or np.sum(rim_mask) == 0:
            continue

        # Check 1: Interior vs Background Contrast
        # Rejects accidental intersections of random edges where the inside looks identical to the outside
        patch_gray = blurred[y1:y2, x1:x2]
        mean_inner = float(np.mean(patch_gray[inner_mask]))
        mean_outer = float(np.mean(patch_gray[outer_mask]))
        
        if mean_inner <= mean_outer * 1.08 or (mean_inner - mean_outer) < 8.0:
            continue

        # Check 2: Angular Coverage & Radial Gradient Alignment
        # For a true circle, gradients along the rim point radially inward/outward (collinear with dx, dy).
        # Straight lines passing through the circle fail this test.
        patch_edges = clean_edges[y1:y2, x1:x2]
        active_rim = rim_mask & patch_edges

        if np.sum(active_rim) < (0.30 * 2 * np.pi * r_px):
            continue

        gx_rim = grad_x[y1:y2, x1:x2][active_rim]
        gy_rim = grad_y[y1:y2, x1:x2][active_rim]
        mag_rim = np.sqrt(gx_rim**2 + gy_rim**2) + 1e-6

        rx_rim = (dx / dist)[active_rim]
        ry_rim = (dy / dist)[active_rim]

        # Cosine similarity between edge gradient and radial vector: |cos(theta)| -> 1.0 for true circle
        radial_alignment = np.abs((gx_rim * rx_rim + gy_rim * ry_rim) / mag_rim)
        if np.mean(radial_alignment) < 0.72:
            continue

        results.append([px, py, r_px])
        extra_info.append({
            'type': 'circle',
            'x': px,
            'y': py,
            'r': r_px
        })

    edges_uint8 = (clean_edges * 255).astype(np.uint8)
    debug_dict = {
        'prob_map': blurred,
        'binary_mask': edges_uint8,
        'geometries': extra_info
    }

    if len(results) == 0:
        return (np.empty((0, 3)), debug_dict)

    filtered_results = suppress_overlapping_circles(results, intensity, resolution)

    return (np.array(filtered_results), debug_dict) if len(filtered_results) > 0 else (np.empty((0, 3)), debug_dict)


# def method_2d_dist_transform(bev_img, resolution=0.01, origin=(0.0, 0.0)):
#     """
#     Robust 2D Manhole Detection for sparse LiDAR BEV data using 
#     Morphological Closing, Distance Transform, and Local Maxima Peak Extraction.
#     """
#     intensity = bev_img[1].copy()
#     if intensity.max() <= intensity.min():
#         return (np.empty((0, 3)), None)

#     # 1. Robust intensity normalization using percentiles
#     p_low, p_high = np.percentile(intensity, (2, 98))
#     if p_high > p_low:
#         clipped = np.clip(intensity, p_low, p_high)
#         norm_int = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
#     else:
#         norm_int = np.zeros_like(intensity, dtype=np.uint8)

#     # 2. Otsu thresholding to isolate high-intensity reflective manhole returns
#     _, binary = cv2.threshold(norm_int, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # 3. Heavy morphological closing to bridge gaps between sparse LiDAR points
#     # Manholes in LiDAR often appear as a ring of discrete points; closing connects them into solid blobs.
#     kernel_size = int(0.15 / resolution)  # ~15cm kernel to bridge point gaps
#     kernel_size = max(3, kernel_size | 1) # Ensure kernel size is odd
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
#     closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
#     # Remove small speckle noise with opening
#     opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

#     # 4. Distance Transform
#     # Computes distance to the closest zero pixel for each foreground pixel.
#     # For a circular blob, the peak of the distance transform is exactly the center, 
#     # and the peak value corresponds directly to the radius!
#     dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)

#     # 5. Define valid physical radius bounds in pixels
#     min_r_px = int(0.25 / resolution)  # 0.25m min radius -> 0.5m diameter
#     max_r_px = int(0.45 / resolution)  # 0.45m max radius -> 0.9m diameter

#     # 6. Find local peaks in the distance transform map
#     neighborhood_size = int(0.8 / resolution)  # Min distance expected between distinct manholes (~0.8m)
#     neighborhood_size = max(5, neighborhood_size | 1)
    
#     # Extract local maxima peaks that fit our radius criteria
#     local_max = maximum_filter(dist_transform, size=neighborhood_size) == dist_transform
#     peak_mask = local_max & (dist_transform >= min_r_px) & (dist_transform <= max_r_px) & (opened > 0)

#     py_indices, px_indices = np.where(peak_mask)

#     results = []
#     extra_info = []
#     h, w = bev_img.shape[1], bev_img.shape[2]

#     for py, px in zip(py_indices, px_indices):
#         radius = float(dist_transform[py, px])
#         if min_r_px <= radius <= max_r_px:
#             if 0 <= py < h and 0 <= px < w:
#                 r_int = int(round(radius))
#                 results.append([int(px), int(py), r_int])
#                 extra_info.append({
#                     'type': 'circle',
#                     'x': int(px),
#                     'y': int(py),
#                     'r': r_int
#                 })

#     # Normalize distance transform to uint8 for safe debugging visualization
#     dt_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#     debug_dict = {
#         'prob_map': dt_vis,      # Panel 2 shows the distance field (peaks represent centers)
#         'binary_mask': opened,   # Panel 3 shows the closed LiDAR point blobs
#         'geometries': extra_info
#     }

#     return (np.array(results), debug_dict) if len(results) > 0 else (np.empty((0, 3)), debug_dict)


# def method_2d_dist_transform(bev_img, resolution=0.01, origin=(0.0, 0.0)):
#     """
#     Robust 2D Manhole Detection for sparse LiDAR BEV data using 
#     Percentile Thresholding, Distance Transform, and Intensity Contrast Validation.
#     """
#     intensity = bev_img[1].copy()
#     if intensity.max() <= intensity.min():
#         return (np.empty((0, 3)), None)

#     # 1. Robust intensity normalization using percentiles
#     p_low, p_high = np.percentile(intensity, (2, 98))
#     if p_high > p_low:
#         clipped = np.clip(intensity, p_low, p_high)
#         norm_int = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
#     else:
#         norm_int = np.zeros_like(intensity, dtype=np.uint8)

#     # 2. FIX: Replace Otsu with High-Percentile Thresholding
#     # Manholes are highly reflective/distinct. Otsu fails on roads because asphalt dominates.
#     # We dynamically isolate only the top 3% brightest intensity pixels.
#     thresh_val = np.percentile(norm_int, 97)
#     _, binary = cv2.threshold(norm_int, thresh_val, 255, cv2.THRESH_BINARY)

#     # 3. Morphological closing to bridge gaps between sparse LiDAR points
#     kernel_size = int(0.15 / resolution)  # ~15cm kernel to bridge point gaps
#     kernel_size = max(3, kernel_size | 1) # Ensure odd kernel size
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
#     closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
#     # Remove small speckle noise with opening
#     opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

#     # 4. Distance Transform to extract center and radius simultaneously
#     dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)

#     # 5. Define valid physical radius bounds in pixels
#     min_r_px = int(0.25 / resolution)  # 0.25m min radius -> 0.5m diameter
#     max_r_px = int(0.45 / resolution)  # 0.45m max radius -> 0.9m diameter

#     # 6. Find local peaks in the distance transform map
#     neighborhood_size = int(0.8 / resolution)  # Min distance expected between manholes (~0.8m)
#     neighborhood_size = max(5, neighborhood_size | 1)
    
#     local_max = maximum_filter(dist_transform, size=neighborhood_size) == dist_transform
#     peak_mask = local_max & (dist_transform >= min_r_px) & (dist_transform <= max_r_px) & (opened > 0)

#     py_indices, px_indices = np.where(peak_mask)

#     results = []
#     extra_info = []
#     h, w = bev_img.shape[1], bev_img.shape[2]

#     # 7. Intensity Validation (Filter out false positive asphalt blobs)
#     for py, px in zip(py_indices, px_indices):
#         radius = float(dist_transform[py, px])
#         if min_r_px <= radius <= max_r_px:
#             if 0 <= py < h and 0 <= px < w:
#                 # Create a mask for the candidate circle to check inner intensity vs background
#                 y_grid, x_grid = np.ogrid[:h, :w]
#                 dist_from_center = np.sqrt((x_grid - px)**2 + (y_grid - py)**2)
                
#                 inner_mask = dist_from_center <= radius
#                 ring_mask = (dist_from_center > radius) & (dist_from_center <= radius * 1.5)
                
#                 # Ensure masks are valid
#                 if np.sum(inner_mask) > 0 and np.sum(ring_mask) > 0:
#                     mean_inner = np.mean(norm_int[inner_mask])
#                     mean_ring = np.mean(norm_int[ring_mask])
                    
#                     # A true manhole must be significantly brighter than its immediate asphalt surroundings
#                     # (Adjust contrast threshold if necessary, e.g., mean_inner > mean_ring * 1.15)
#                     if mean_inner > mean_ring * 1.1:
#                         r_int = int(round(radius))
#                         results.append([int(px), int(py), r_int])
#                         extra_info.append({
#                             'type': 'circle',
#                             'x': int(px),
#                             'y': int(py),
#                             'r': r_int
#                         })

#     # Normalize distance transform for debugging visualization
#     dt_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#     debug_dict = {
#         'prob_map': dt_vis,      # Panel 2 shows the distance field peaks
#         'binary_mask': opened,   # Panel 3 shows filtered LiDAR blobs
#         'geometries': extra_info
#     }

#     return (np.array(results), debug_dict) if len(results) > 0 else (np.empty((0, 3)), debug_dict)


def method_2d_dist_transform(bev_img, resolution=0.01, origin=(0.0, 0.0)):
    """
    Highly Robust 2D Manhole Detection for sparse LiDAR BEV data using 
    Percentile Thresholding, Morphological Closing, Contour Shape Filtering, 
    and Distance Transform Peak Extraction.
    """
    intensity = bev_img[1].copy()
    if intensity.max() <= intensity.min():
        return (np.empty((0, 3)), None)

    # 1. Robust intensity normalization using percentiles
    p_low, p_high = np.percentile(intensity, (2, 98))
    if p_high > p_low:
        clipped = np.clip(intensity, p_low, p_high)
        norm_int = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    else:
        norm_int = np.zeros_like(intensity, dtype=np.uint8)

    # 2. Isolate top 3% brightest intensity pixels (target reflective manhole covers)
    thresh_val = np.percentile(norm_int, 97)
    _, binary = cv2.threshold(norm_int, thresh_val, 255, cv2.THRESH_BINARY)

    # 3. Morphological closing to bridge gaps between sparse LiDAR points
    kernel_size = int(0.15 / resolution)  # ~15cm kernel to bridge point gaps
    kernel_size = max(3, kernel_size | 1) # Ensure odd kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # 4. PRE-FILTER BLOBS VIA CONTOURS (Crucial for high Precision)
    # Filter out non-circular objects, linear road markings, or irregular debris 
    # BEFORE running distance transform.
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_r_px = int(0.25 / resolution)  # 0.25m min radius -> 0.5m diameter
    max_r_px = int(0.45 / resolution)  # 0.45m max radius -> 0.9m diameter
    
    min_area = np.pi * (min_r_px * 0.7) ** 2
    max_area = np.pi * (max_r_px * 1.3) ** 2

    filtered_binary = np.zeros_like(opened)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
            
        # Circularity check: 4 * pi * Area / (Perimeter^2). Perfect circle = 1.0
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < 0.65:  # Strict filter against elongated or random shapes
            continue

        # Extent check: Area / Bounding Box Area. For a circle, ratio is approx pi / 4 (~0.785)
        _, _, w_box, h_box = cv2.boundingRect(cnt)
        box_area = w_box * h_box
        if box_area > 0:
            extent = area / box_area
            if not (0.5 <= extent <= 0.95):  # Validates circle-like square filling
                continue

        # If the blob passes geometric tests, keep it in the working mask
        cv2.drawContours(filtered_binary, [cnt], -1, 255, thickness=cv2.FILLED)

    # 5. Distance Transform on pre-filtered valid circular blobs only
    dist_transform = cv2.distanceTransform(filtered_binary, cv2.DIST_L2, 5)

    # 6. Find local peaks in the distance transform map
    neighborhood_size = int(0.8 / resolution)  # Min distance expected between manholes (~0.8m)
    neighborhood_size = max(5, neighborhood_size | 1)
    
    local_max = maximum_filter(dist_transform, size=neighborhood_size) == dist_transform
    peak_mask = local_max & (dist_transform >= min_r_px) & (dist_transform <= max_r_px) & (filtered_binary > 0)

    py_indices, px_indices = np.where(peak_mask)

    results = []
    extra_info = []
    h, w = bev_img.shape[1], bev_img.shape[2]

    # 7. Final Intensity Contrast Validation
    for py, px in zip(py_indices, px_indices):
        radius = float(dist_transform[py, px])
        if min_r_px <= radius <= max_r_px:
            if 0 <= py < h and 0 <= px < w:
                y_grid, x_grid = np.ogrid[:h, :w]
                dist_from_center = np.sqrt((x_grid - px)**2 + (y_grid - py)**2)
                
                inner_mask = dist_from_center <= radius
                ring_mask = (dist_from_center > radius) & (dist_from_center <= radius * 1.5)
                
                if np.sum(inner_mask) > 0 and np.sum(ring_mask) > 0:
                    mean_inner = np.mean(norm_int[inner_mask])
                    mean_ring = np.mean(norm_int[ring_mask])
                    
                    # Core must be brighter than immediate surrounding road background
                    if mean_inner > mean_ring * 1.05:
                        r_int = int(round(radius))
                        results.append([int(px), int(py), r_int])
                        extra_info.append({
                            'type': 'circle',
                            'x': int(px),
                            'y': int(py),
                            'r': r_int
                        })

    dt_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    debug_dict = {
        'prob_map': dt_vis,
        'binary_mask': filtered_binary,
        'geometries': extra_info
    }

    return (np.array(results), debug_dict) if len(results) > 0 else (np.empty((0, 3)), debug_dict)



# def method_2d_lidar_robust(bev_img, resolution=0.01, origin=(0.0, 0.0)):
#     """
#     Highly Robust 2D Manhole Detection for sparse LiDAR BEV data.
#     Builds directly on the proven Distance-Transform + Contour pipeline,
#     upgraded with Multi-Percentile Scanning and 4-Quadrant Surround Verification.
#     """
#     intensity = bev_img[1].copy()
#     if intensity.max() <= intensity.min():
#         return (np.empty((0, 3)), None)

#     h, w = bev_img.shape[1], bev_img.shape[2]

#     # 1. Proven robust intensity normalization using percentiles
#     p_low, p_high = np.percentile(intensity, (2, 98))
#     if p_high > p_low:
#         clipped = np.clip(intensity, p_low, p_high)
#         norm_int = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
#     else:
#         norm_int = np.zeros_like(intensity, dtype=np.uint8)

#     # Physical radius bounds in pixels
#     min_r_px = int(0.25 / resolution)  # 25 px (0.5m diameter)
#     max_r_px = int(0.45 / resolution)  # 45 px (0.9m diameter)
    
#     min_area = np.pi * (min_r_px * 0.72) ** 2
#     max_area = np.pi * (max_r_px * 1.28) ** 2

#     # Morphological kernel (~13cm to bridge sparse LiDAR gaps without over-merging)
#     kernel_size = max(3, int(0.13 / resolution) | 1)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

#     # 2. Multi-Percentile Contour Extraction
#     # Testing 3 levels catches dimmer manholes (95) while keeping bright ones (98.5) sharp
#     combined_valid_blobs = np.zeros_like(norm_int)

#     for pct in [98.0, 96.8, 95.5]:
#         thresh_val = np.percentile(norm_int, pct)
#         if thresh_val <= 5:
#             continue
            
#         _, binary = cv2.threshold(norm_int, thresh_val, 255, cv2.THRESH_BINARY)
#         closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
#         opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

#         contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             if not (min_area <= area <= max_area):
#                 continue

#             perimeter = cv2.arcLength(cnt, True)
#             if perimeter == 0:
#                 continue

#             # Circularity check: 4 * pi * Area / (Perimeter^2)
#             circularity = (4 * np.pi * area) / (perimeter ** 2)
#             if circularity < 0.68:
#                 continue

#             # Bounding Box Aspect Ratio & Extent check
#             _, _, w_box, h_box = cv2.boundingRect(cnt)
#             if h_box == 0 or w_box == 0:
#                 continue
                
#             aspect_ratio = float(w_box) / float(h_box)
#             if not (0.72 <= aspect_ratio <= 1.38):
#                 continue

#             extent = area / float(w_box * h_box)
#             if not (0.55 <= extent <= 0.92):
#                 continue

#             # Convex Hull Solidity check (rejects crescent/hollow or jagged road lines)
#             hull = cv2.convexHull(cnt)
#             hull_area = cv2.contourArea(hull)
#             if hull_area == 0 or (area / hull_area) < 0.82:
#                 continue

#             # Blob passed all geometric tests -> add to verified mask
#             cv2.drawContours(combined_valid_blobs, [cnt], -1, 255, thickness=cv2.FILLED)

#     # 3. Distance Transform on verified circular blobs
#     dist_transform = cv2.distanceTransform(combined_valid_blobs, cv2.DIST_L2, 5)

#     # 4. Find local peaks in the distance transform map
#     neighborhood_size = max(5, int(0.8 / resolution) | 1)
#     local_max = maximum_filter(dist_transform, size=neighborhood_size) == dist_transform
#     peak_mask = (
#         local_max 
#         & (dist_transform >= min_r_px * 0.9) 
#         & (dist_transform <= max_r_px * 1.1) 
#         & (combined_valid_blobs > 0)
#     )

#     py_indices, px_indices = np.where(peak_mask)

#     candidates = []

#     # 5. Strict 4-Quadrant Surround & Contrast Verification
#     # Eliminates road markings/stripes by checking that the circle is surrounded on ALL sides
#     for py, px in zip(py_indices, px_indices):
#         radius = float(dist_transform[py, px])
#         r_clamped = np.clip(radius, min_r_px, max_r_px)

#         if not (0 <= py < h and 0 <= px < w):
#             continue

#         # Crop a fast local patch around the candidate
#         pad = int(r_clamped * 1.7)
#         y1, y2 = max(0, py - pad), min(h, py + pad + 1)
#         x1, x2 = max(0, px - pad), min(w, px + pad + 1)

#         y_grid, x_grid = np.ogrid[y1:y2, x1:x2]
#         dy = y_grid - py
#         dx = x_grid - px
#         dist_from_center = np.sqrt(dx**2 + dy**2)

#         inner_mask = dist_from_center <= (r_clamped * 0.9)
#         ring_mask = (dist_from_center > (r_clamped * 1.1)) & (dist_from_center <= (r_clamped * 1.55))

#         if np.sum(inner_mask) == 0 or np.sum(ring_mask) == 0:
#             continue

#         patch = norm_int[y1:y2, x1:x2]
#         mean_inner = float(np.mean(patch[inner_mask]))
#         mean_ring = float(np.mean(patch[ring_mask]))

#         # Overall contrast check
#         if mean_inner <= mean_ring * 1.12 or (mean_inner - mean_ring) < 10.0:
#             continue

#         # 4-Quadrant Check: A true manhole is surrounded by darker asphalt on ALL 4 sides.
#         # A white road stripe continues in 2 directions, so 2 quadrants will be bright!
#         q1 = ring_mask & (dy < 0)  # Top
#         q2 = ring_mask & (dy > 0)  # Bottom
#         q3 = ring_mask & (dx < 0)  # Left
#         q4 = ring_mask & (dx > 0)  # Right

#         quadrants_darker = 0
#         for q_mask in (q1, q2, q3, q4):
#             if np.sum(q_mask) > 0:
#                 if mean_inner > float(np.mean(patch[q_mask])) * 1.06:
#                     quadrants_darker += 1

#         # Require at least 3 out of 4 surrounding sides to be significantly darker
#         if quadrants_darker < 3:
#             continue

#         score = (mean_inner - mean_ring) * (quadrants_darker / 4.0)
#         r_int = int(round(r_clamped))
#         candidates.append((int(px), int(py), r_int, score))

#     # 6. Non-Maximum Suppression (keep strongest non-overlapping candidates)
#     candidates.sort(key=lambda x: x[3], reverse=True)
#     min_dist_px = 0.8 / resolution

#     results = []
#     extra_info = []

#     for px, py, r_int, score in candidates:
#         if all(np.hypot(px - kx, py - ky) >= min_dist_px for kx, ky, _ in results):
#             results.append([px, py, r_int])
#             extra_info.append({
#                 'type': 'circle',
#                 'x': px,
#                 'y': py,
#                 'r': r_int
#             })

#     dt_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#     debug_dict = {
#         'prob_map': dt_vis,
#         'binary_mask': combined_valid_blobs,
#         'geometries': extra_info
#     }

#     return (np.array(results), debug_dict) if len(results) > 0 else (np.empty((0, 3)), debug_dict)


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


# def _generate_ring_template(radius_px, thickness_px=3):
#     """
#     Creates a synthetic binary/float edge template of a circular ring.
#     The circle boundary is set to 1.0 and background to 0.0.
#     """
#     pad = int(thickness_px + 2)
#     side = int(2 * (radius_px + pad) + 1)
#     center = (side // 2, side // 2)

#     template = np.zeros((side, side), dtype=np.float32)
#     cv2.circle(
#         template,
#         center,
#         int(round(radius_px)),
#         color=1.0,
#         thickness=int(thickness_px),
#     )
#     return template


# def _non_max_suppression_circles(candidates, min_dist_px):
#     """
#     Filters dense multi-scale detections using greedy 2D Non-Maximum Suppression.
#     Candidates: list of (x, y, r, score)
#     """
#     if len(candidates) == 0:
#         return []

#     # Sort descending by correlation score
#     candidates = sorted(candidates, key=lambda c: c[3], reverse=True)
#     kept = []

#     for cand in candidates:
#         cx, cy, r, score = cand
#         overlap = False
#         for k_cx, k_cy, _, _ in kept:
#             dist = np.hypot(cx - k_cx, cy - k_cy)
#             if dist < min_dist_px:
#                 overlap = True
#                 break

#         if not overlap:
#             kept.append(cand)

#     return kept


# def method_2d_template_matching(
#     bev_img,
#     resolution=0.01,
#     match_threshold=0.35,
#     radius_step_m=0.04,
#     ring_thickness_m=0.04,
# ):
#     """
#     2D Detection via Multi-Scale Edge Ring Template Matching and Non-Max Suppression.

#     Parameters:
#       - bev_img: Array where bev_img[1] is the intensity channel.
#       - resolution: Spatial resolution in m/pixel (default: 0.01 m/px).
#       - origin: (x, y) real-world offset of pixel (0, 0).
#       - match_threshold: Min TM_CCOEFF_NORMED score [0.0 - 1.0] to consider a candidate.
#       - radius_step_m: Step size to scale circular templates between min/max radius.
#       - ring_thickness_m: Physical thickness of the synthetic ring border.

#     Returns:
#       - np.ndarray of shape (N, 3) containing [x_m, y_m, r_m].
#     """
#     intensity = bev_img[1].copy()
#     if intensity.max() <= intensity.min():
#         return np.empty((0, 3))

#     # Normalize intensity to uint8
#     norm_int = cv2.normalize(intensity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     blurred = cv2.GaussianBlur(norm_int, (3, 3), 1.0)

#     # Edge map of the scene
#     # Matching edge-to-edge avoids lighting variation issues on road surfaces
#     edges = cv2.Canny(blurred, threshold1=50, threshold2=120).astype(np.float32) / 255.0

#     min_r_px = int(0.25 / resolution)  # ~25 px for 0.5m diameter
#     max_r_px = int(0.45 / resolution)  # ~45 px for 0.9m diameter
#     step_px = max(1, int(radius_step_m / resolution))
#     thickness_px = max(1, int(ring_thickness_m / resolution))
#     min_dist_px = 1.0 / resolution      # 1 meter clearance between centers

#     candidates = []

#     # Iterate over multiple potential radii
#     for r_px in range(min_r_px, max_r_px + 1, step_px):
#         tpl = _generate_ring_template(r_px, thickness_px=thickness_px)
#         th, tw = tpl.shape

#         if edges.shape[0] < th or edges.shape[1] < tw:
#             continue

#         # TM_CCOEFF_NORMED outputs scores in [-1.0, 1.0]
#         corr_map = cv2.matchTemplate(edges, tpl, cv2.TM_CCOEFF_NORMED)

#         # Threshold to gather raw local peaks
#         match_ys, match_xs = np.where(corr_map >= match_threshold)

#         for top_left_x, top_left_y in zip(match_xs, match_ys):
#             score = float(corr_map[top_left_y, top_left_x])
#             # Shift top-left template position to true circle center
#             center_x = top_left_x + tw / 2.0
#             center_y = top_left_y + th / 2.0
#             candidates.append((center_x, center_y, float(r_px), score))

#     if not candidates:
#         return []

#     # Suppress overlapping hits across different scales and nearby locations
#     filtered_circles = _non_max_suppression_circles(candidates, min_dist_px=min_dist_px)

#     # Map back to real-world ground coordinates
#     results = []
#     h, w = bev_img.shape[1], bev_img.shape[2]

#     for cx_px, cy_px, r_px, _ in filtered_circles:
#         if 0 <= cy_px < h and 0 <= cx_px < w:
#             results.append([cx_px, cy_px])

#     return np.array(results) if len(results) > 0 else []


def _generate_ring_template(r_px, thickness_px):
    """
    Generates a synthetic binary edge ring template of a given radius and thickness.
    """
    # Canvas size must accommodate the radius plus thickness/border
    size = 2 * (r_px + thickness_px) + 1
    center = size // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
    
    # Define ring mask: pixels falling onto the circle perimeter border
    inner_r = max(0, r_px - thickness_px / 2.0)
    outer_r = r_px + thickness_px / 2.0
    
    ring_mask = (dist_from_center >= inner_r) & (dist_from_center <= outer_r)
    
    # Convert to float32 template [0.0, 1.0] to match edge map
    template = ring_mask.astype(np.float32)
    return template

def _non_max_suppression_circles(candidates, min_dist_px):
    """
    Suppresses overlapping circle candidates based on score and minimum distance.
    Candidates format: (center_x, center_y, radius, score)
    """
    if not candidates:
        return []

    # Sort candidates by matching score in descending order (highest score first)
    sorted_candidates = sorted(candidates, key=lambda x: x[3], reverse=True)
    
    kept_circles = []
    for cand in sorted_candidates:
        cx, cy, r, score = cand
        suppress = False
        
        for kept in kept_circles:
            kcx, kcy, kr, _ = kept
            # Compute Euclidean distance between centers
            dist = np.sqrt((cx - kcx)**2 + (cy - kcy)**2)
            # If centers are too close, suppress the lower-scoring candidate
            if dist < min_dist_px:
                suppress = True
                break
                
        if not suppress:
            kept_circles.append(cand)
            
    return kept_circles

def method_2d_template_matching(
    bev_img,
    resolution=0.01,
    origin=(0.0, 0.0),
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
      - Tuple of (np.ndarray of shape (N, 3) containing [px, py, r_px], debug_dict)
    """
    intensity = bev_img[1].copy()
    if intensity.max() <= intensity.min():
        return (np.empty((0, 3)), None)

    # Normalize intensity to 0-255 uint8
    p_low, p_high = np.percentile(intensity, (2, 98))
    if p_high > p_low:
        clipped = np.clip(intensity, p_low, p_high)
        norm_int = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    else:
        norm_int = np.zeros_like(intensity, dtype=np.uint8)

    # Blur to smooth out high-frequency asphalt noise
    blurred = cv2.GaussianBlur(norm_int, (5, 5), 1.5)

    # Edge map of the scene using Canny
    # Matching edge-to-edge avoids lighting variation issues on road surfaces
    edges = cv2.Canny(blurred, threshold1=50, threshold2=120).astype(np.float32) / 255.0

    min_r_px = int(0.25 / resolution)  # ~25 px for 0.5m diameter
    max_r_px = int(0.45 / resolution)  # ~45 px for 0.9m diameter
    step_px = max(1, int(radius_step_m / resolution))
    thickness_px = max(1, int(ring_thickness_m / resolution))
    min_dist_px = 1.0 / resolution     # 1 meter clearance between centers

    candidates = []

    # Iterate over multiple potential radii scales
    for r_px in range(min_r_px, max_r_px + 1, step_px):
        tpl = _generate_ring_template(r_px, thickness_px=thickness_px)
        th, tw = tpl.shape

        if edges.shape[0] < th or edges.shape[1] < tw:
            continue

        # TM_CCOEFF_NORMED outputs scores in [-1.0, 1.0]
        corr_map = cv2.matchTemplate(edges, tpl, cv2.TM_CCOEFF_NORMED)

        # Threshold to gather raw local matches
        match_ys, match_xs = np.where(corr_map >= match_threshold)

        for top_left_x, top_left_y in zip(match_xs, match_ys):
            score = float(corr_map[top_left_y, top_left_x])
            # Shift top-left template position to true circle center
            center_x = top_left_x + tw / 2.0
            center_y = top_left_y + th / 2.0
            candidates.append((center_x, center_y, float(r_px), score))

    if not candidates:
        return (np.empty((0, 3)), None)

    # Suppress overlapping hits across different scales and nearby locations
    filtered_circles = _non_max_suppression_circles(candidates, min_dist_px=min_dist_px)

    # Collect Debug Info and Format Results
    results = []
    extra_info = []
    h, w = bev_img.shape[1], bev_img.shape[2]

    for cx_px, cy_px, r_px, score in filtered_circles:
        if 0 <= cy_px < h and 0 <= cx_px < w:
            px, py, pr = int(round(cx_px)), int(round(cy_px)), int(round(r_px))
            results.append([px, py, pr])
            extra_info.append({
                'type': 'circle',
                'x': px,
                'y': py,
                'r': pr
            })

    edges_uint8 = (edges * 255).astype(np.uint8)
    debug_dict = {
        'prob_map': blurred,
        'binary_mask': edges_uint8,
        'geometries': extra_info
    }

    return (np.array(results), debug_dict) if len(results) > 0 else (np.empty((0, 3)), debug_dict)




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
    # "2d_hough": method_2d_hough,
    "2d_hough": method_2d_hough_skimage,              # Works
    "2d_contours": method_2d_contours,
    "2d_dist_transform": method_2d_dist_transform,    # Works, but maybe improve
    "2d_template": method_2d_template_matching,
    "2d_ransac": method_2d_ransac,
    "2d_circularity": method_2d_contour_circularity,  # Works, but maybe improve
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




















