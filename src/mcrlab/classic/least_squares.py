# ----------
# > Import <
# ----------
import numpy as np
from scipy.optimize import least_squares

from mcrlab.classic.utils import fit_plane, plane_basis, project_to_plane



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
    # print("Normal", normal.shape)
    basis_x, basis_y = plane_basis(normal)

    # projection into 2D
    x, y = project_to_plane(centroid=centroid, points=points, basis_x=basis_x, basis_y=basis_y)

    # optimize circle shape
    a, b, r, mean_distance_error, loss = fit_circle_least_squares(x, y)

    # back projection
    center_3D = centroid + (a * basis_x) + (b *basis_y)

    return center_3D, normal, r, mean_distance_error, loss







