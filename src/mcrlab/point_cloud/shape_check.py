# ----------
# > Import <
# ----------
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial import ConvexHull

from mcrlab.classic.least_squares import fit_circle_least_squares_3D
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

    # 4. PCA shape check
    cov = np.cov(points_2d.T)
    # eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)
    # anisotropy = 1 - (eigvals[0] / eigvals[1])
    pca_score = eigvals[0] / (eigvals[1] + 1e-8)

    # 5. Radial Variance
    center = points_2d.mean(axis=0)
    center_norm = np.linalg.norm(points_2d - center, axis=1)

    radial_var = np.std(center_norm) / (np.mean(center_norm) + 1e-8)

    # 6. Least Square Fit Shape Check
    center_3D, normal, r, mean_distance_error, loss = fit_circle_least_squares_3D(points)

    # 7. Voting
    vote_circle = 0
    vote_circle += circularity > 0.85
    vote_circle += pca_score > 0.5
    vote_circle += radial_var < 0.25
    vote_circle += mean_distance_error < 0.1

    score = vote_circle / 4.0

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
            ax.set_title("Shape Check (✅ Is a Circle)")
        else:
            ax.set_title("Shape Check (❌ Is not a Circle)")

        if save_path is not None:
            plt.savefig(save_path)

        if should_plot:
            plt.show()

        plt.close(fig)


    return is_circle_, {
        "circularity": circularity,
        "pca_score": pca_score,
        "radial_var": radial_var,
        "least_squares_error": mean_distance_error,
        "score": score
    }







