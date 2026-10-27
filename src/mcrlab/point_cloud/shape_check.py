# ----------
# > Import <
# ----------
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial import ConvexHull

from mcrlab.traditional.least_squares import fit_circle_least_squares_3D
from mcrlab.point_cloud.utils import get_coordinate_attribute



# ---------------
# > Shape Check <
# ---------------
def circle_shape_check(points, save_path=None, should_plot=False, threshold=0.6):
    # 1. check data format
    if isinstance(points, o3d.t.geometry.PointCloud):
        points = points.point[get_coordinate_attribute(points)].numpy()
    
    # 2. Project to 2D
    points_2d = points[:, :2]
    
    # 3. Compute the Convex Hull -> the "envelope" of the points
    # This gives us a clean shape even if the LIDAR points are sparse inside
    hull = ConvexHull(points_2d)

    # Extract Area and Perimeter (length) from the hull
    area = hull.volume  # In 2D, 'volume' is the area
    perimeter = hull.area  # In 2D, 'area' is the perimeter
    
    # Calculate Circularity
    # -> https://en.wikipedia.org/wiki/Isoperimetric_inequality
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-8)

    # # 4. PCA shape check
    # cov = np.cov(points_2d.T)
    # # eigvals, _ = np.linalg.eigh(cov)
    # eigvals = np.linalg.eigvalsh(cov)
    # eigvals = np.sort(eigvals)
    # # anisotropy = 1 - (eigvals[0] / eigvals[1])
    # pca_score = eigvals[0] / (eigvals[1] + 1e-8)

    # # 5. Radial Variance
    # center = points_2d.mean(axis=0)
    # center_norm = np.linalg.norm(points_2d - center, axis=1)

    # radial_var = np.std(center_norm) / (np.mean(center_norm) + 1e-8)

    # 6. Least Square Fit Shape Check
    center_3D, normal, r, mean_distance_error, loss = fit_circle_least_squares_3D(points)

    # 7. Voting
    vote_circle = 0
    vote_circle += circularity > 0.85
    # vote_circle += pca_score > 0.5
    # vote_circle += radial_var < 0.25
    vote_circle += mean_distance_error < 0.1

    score = vote_circle / 2.0

    is_circle_ = score >= threshold
    

    # 8. visualize
    if should_plot or save_path is not None:
        plt.style.use("seaborn-v0_8-whitegrid")

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

        if is_circle_:
            ax.set_title("Shape Check (Is a Circle)")
        else:
            ax.set_title("Shape Check (Is NOT a Circle)")

        stats_text = (
            f"circularity: {circularity:.3f} ({circularity > 0.85})\n"
            # f"pca_score: {pca_score:.3f} ({pca_score > 0.5})\n"
            # f"radial_var: {radial_var:.3f} ({radial_var < 0.25})\n"
            f"least_squares_error: {mean_distance_error:.3f} ({mean_distance_error < 0.1})\n"
            f"score: {score:.3f} ({score >= threshold})"
        )

        ax.text(
            0.05, 0.95,           # X, Y coordinates (5% from left, 95% from bottom)
            stats_text, 
            transform=ax.transAxes,
            fontsize=10, 
            fontfamily='monospace',  # Keeps things aligned nicely like code
            verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
        )

        if save_path is not None:
            plt.savefig(save_path)

        if should_plot:
            plt.show()

        plt.close(fig)


    return is_circle_, {
        "circularity": circularity,
        # "pca_score": pca_score,
        # "radial_var": radial_var,
        "least_squares_error": mean_distance_error,
        "score": score
    }

# 2D Shape Check
def fit_circle_least_squares_2d(points_2d):
    """Algebraic circle fit (Kasa method)."""
    x = points_2d[:, 0]
    y = points_2d[:, 1]
    
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    xc = c[0] / 2.0
    yc = c[1] / 2.0
    val = c[2] + xc**2 + yc**2
    if val <= 0:
        return np.array([xc, yc]), 1.0, 999.0
    r = np.sqrt(val)
    
    distances = np.abs(np.sqrt((x - xc)**2 + (y - yc)**2) - r)
    mean_error = np.mean(distances)
    return np.array([xc, yc]), r, mean_error

def check_angular_coverage(points_2d, center):
    """
    Computes angular coverage (in radians) of boundary points around the center.
    A full circle covers ~2*pi (6.28), a half circle covers ~pi (3.14).
    """
    angles = np.arctan2(points_2d[:, 1] - center[1], points_2d[:, 0] - center[0])
    angles = np.sort(angles)
    
    # Calculate angular differences between consecutive sorted points
    diffs = np.diff(angles)
    # Wrap-around gap from last to first point
    diffs = np.append(diffs, (angles[0] + 2 * np.pi) - angles[-1])
    
    max_gap = np.max(diffs)
    span = 2 * np.pi - max_gap
    return span

def circle_shape_check_2d(points, save_path=None, should_plot=False, threshold=0.5):
    if 'o3d' in globals() and isinstance(points, o3d.t.geometry.PointCloud):
        points = points.point[get_coordinate_attribute(points)].numpy()

    points = np.asarray(points)
    points_2d = points[:, :2]

    if len(points_2d) < 8:
        return False, {"score": 0.0, "shape_type": "none", "reason": "Too few points"}

    # 1. Convex Hull & Circularity
    try:
        hull = ConvexHull(points_2d)
        area = hull.volume
        perimeter = hull.area
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-8)
        hull_pts = points_2d[hull.vertices]
    except QhullError:
        return False, {"score": 0.0, "shape_type": "none", "reason": "Degenerate hull"}

    # 2. Fit Circle to Hull Boundary
    center_2d, r, mean_dist_error = fit_circle_least_squares_2d(hull_pts)
    rel_fit_error = mean_dist_error / (r + 1e-8)
    
    # 3. Check Angular Span around the center
    angular_span = check_angular_coverage(hull_pts, center_2d)

    # 4. Separate Tests for Full Circle vs. Half Circle
    # Full circle: high circularity, low fit error, span near 2*pi (e.g. > 5.0 rad / ~286 deg)
    is_full = (circularity > 0.72) and (rel_fit_error < 0.18) and (angular_span > 4.8)

    # Half circle:
    # - Circularity of ideal semicircle is ~0.74, discretized on pixel grid ~0.50-0.72
    # - Angular span around center is near pi (between 2.2 rad (~125 deg) and 4.2 rad (~240 deg))
    # - Relative fit error on arc points: allow slightly higher tolerance (<= 0.25)
    is_half = (0.48 <= circularity <= 0.75) and (rel_fit_error < 0.25) and (2.2 <= angular_span <= 4.2)

    detected_shape = "none"
    score = 0.0

    if is_full:
        detected_shape = "full_circle"
        score = 1.0
    elif is_half:
        detected_shape = "half_circle"
        score = 0.85
    else:
        # Partial soft score
        score = max(0.0, 1.0 - rel_fit_error) * (circularity / 0.8)

    is_circle_ = (is_full or is_half) and (score >= threshold)

    # 5. Visualization
    if should_plot or save_path is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(points_2d[:, 1], points_2d[:, 0], s=8, alpha=0.3, label="Mask Points")

        hull_plot = np.vstack([hull_pts, hull_pts[0]])
        ax.plot(hull_plot[:, 1], hull_plot[:, 0], 'r-', lw=2, label="Convex Hull")

        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = center_2d[0] + r * np.cos(theta)
        circle_y = center_2d[1] + r * np.sin(theta)
        ax.plot(circle_y, circle_x, 'b--', lw=1.5, label="Fitted Circle")

        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.legend()
        ax.set_title(f"Shape Check: {detected_shape.upper()} ({'PASS' if is_circle_ else 'FAIL'})")

        stats_text = (
            f"Type: {detected_shape}\n"
            f"Circularity: {circularity:.3f}\n"
            f"Rel Fit Err: {rel_fit_error:.3f}\n"
            f"Angular Span: {np.degrees(angular_span):.1f}°\n"
            f"Radius (px): {r:.1f}\n"
            f"Score: {score:.2f}"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if should_plot:
            plt.show()
        plt.close(fig)

    return is_circle_, {
        "is_circle": is_circle_,
        "shape_type": detected_shape,
        "circularity": circularity,
        "angular_span_deg": np.degrees(angular_span),
        "rel_fit_error": rel_fit_error,
        "radius": r,
        "score": score
    }














