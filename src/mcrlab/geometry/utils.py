# ----------
# > Import <
# ----------
import numpy as np
import open3d as o3d



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














