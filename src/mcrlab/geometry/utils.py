# ----------
# > Import <
# ----------
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt



# ---------
# > Utils <
# ---------
def fit_plane(points):
    """
    Fit/get plane via finding the centroid, 
    normalizing the point cloud and 
    finding the normal via PCA, normal where the plane spreads at least.
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # Singular Value Decomposition -> https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
    # it is basically applying Principal Component Analysis (PCA)
    # Principle Axes:
    #   The First Component: The direction in which the points are most stretched out (the length).
    #   The Second Component: The direction of the second-most spread, perpendicular to the first (the width).
    #   The Third Component (we use this): The direction of the least spread (the thickness).
    # Function Outputs:
    #   Left Singular Vectors (u): These represent the coordinates of your points projected onto the principal axes.
    #   Singular Values (s): A vector of 3 sizes. Large values mean the points spread far in that direction.
    #                                             Small values mean the cloud is "flat" in that direction.
    #   Right Singular Vectors (vh): The Principal Axes. These are the unit vectors defining the orientation of your point cloud.
    _, _, vh = np.linalg.svd(centered)

    # normal = last singular vector
    normal = vh[-1]

    return centroid, normal



def plane_basis(normal):
    """
    Create a local 2D coordinate system.
    """
    normal = normal / np.linalg.norm(normal)

    # pick arbitrary vector not parallel to normal
    if abs(normal[0]) < 0.9:
        v = np.array([1, 0, 0])
    else:
        v = np.array([0, 1, 0])

    normal = np.asarray(normal).reshape(-1)
    basis_x = np.cross(normal, v)
    basis_x /= np.linalg.norm(basis_x)

    basis_y = np.cross(normal, basis_x)

    return basis_x, basis_y



def project_to_plane(points, centroid, basis_x, basis_y):
    diff = points - centroid
    x = diff @ basis_x
    y = diff @ basis_y
    return x, y



# -----------------
# > Visualization <
# -----------------
# def visualize_center_points_in_point_cloud(point_cloud, center_points):
#     o3d.visualization.draw_geometries([point_cloud] + [center_points])
def visualize_circle_fit(points, center_pred, radius, error):
    """
    Visualizes circle fit quality.

    Parameters:
    - points: (N,2) manhole points
    - center_pred: (2,) predicted circle center
    - radius: predicted radius
    - error: mean radial error (from least squares)
    """
    # matplotlib settings
    # plt.style.use("seaborn-v0_8-whitegrid")
    plt.style.use("default")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    # color palette
    POINT_COLOR = "#6c757d"
    CIRCLE_COLOR = "#0077b6"
    ERROR_COLOR = "#90e0ef"
    MEAN_COLOR = "#52b788"
    CENTER_COLOR = "#d62828"

    # extract the data
    x = points[:, 0]
    y = points[:, 1]

    # mean center of points
    center_mean = np.array([np.mean(x), np.mean(y)])

    # reduce prediction to 2D
    center_pred = center_pred[:2]

    # L1 distance between centers
    l1_center_error = np.sum(np.abs(center_pred - center_mean))

    # angles for circle drawing
    t = np.linspace(0, 2*np.pi, 400)

    # predicted circle
    cx, cy = center_pred
    x_c = cx + radius * np.cos(t)
    y_c = cy + radius * np.sin(t)

    # error band (inner + outer circle)
    x_outer = cx + (radius + error) * np.cos(t)
    y_outer = cy + (radius + error) * np.sin(t)

    x_inner = cx + max(radius - error, 0) * np.cos(t)
    y_inner = cy + max(radius - error, 0) * np.sin(t)

    # radius farest point
    distances = np.sqrt((x - center_mean[0])**2 + (y - center_mean[1])**2)
    radius_farest = np.max(distances)

    # mean-center circle (for comparison)
    x_mean = center_mean[0] + radius_farest * np.cos(t)
    y_mean = center_mean[1] + radius_farest * np.sin(t)

    # plotting
    plt.figure(figsize=(7, 7))

    # points
    plt.scatter(x, y, s=15, color=POINT_COLOR, alpha=0.6, label="Manhole Points")

    # predicted circle
    plt.plot(x_c, y_c, color=CIRCLE_COLOR, linewidth=2.5, label="Predicted circle")

    # error band -> area 
    # plt.plot(x_outer, y_outer, "--", label="+ error band")
    # plt.plot(x_inner, y_inner, "--", label="- error band")
    # plt.fill_between(x_outer, y_outer, y_inner, color=ERROR_COLOR, alpha=0.3, label="Error band")
    # plt.fill(x_outer, y_outer, color=ERROR_COLOR, alpha=0.2)
    # plt.fill(x_inner, y_inner, color="white")

    # mean-center circle
    plt.plot(x_mean, y_mean, "g:", label="Mean-center circle")

    # centers
    plt.scatter(*center_pred, color=CENTER_COLOR, s=80, edgecolor="white", zorder=5, label="Predicted center")
    plt.scatter(*center_mean, color=MEAN_COLOR, s=60, edgecolor="white", zorder=5, label="Mean center")

    # center difference
    plt.plot(
        [center_pred[0], center_mean[0]],
        [center_pred[1], center_mean[1]],
        color="black",
        linestyle=":",
        linewidth=1
    )

    # annotation
    plt.title("Circle Fit Evaluation", fontsize=14, weight="bold", y=0.98)
    plt.suptitle(f"Center error = {error:.4f}, L1 Dist to mean center = {l1_center_error:.4f}", fontsize=10, y=0.93)

    # plt.legend(frameon=False, loc="upper right")
    plt.legend(frameon=False, bbox_to_anchor=(0.78, 1.00), loc="upper left")

    plt.grid(alpha=0.3)
    plt.axis("equal")
    plt.margins(0.1)

    plt.show()













