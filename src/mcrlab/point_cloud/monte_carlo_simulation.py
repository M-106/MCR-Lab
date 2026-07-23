# -----------
# > Imports <
# -----------
import os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from matplotlib.transforms as transforms



# -----------------
# > MC Simulation <
# -----------------
# Monte Carlo Simulation is used to generate
# data in form of an knonw/unknown probability
# 1. Define the Input
# 2. Choose a probability distribution
# 3. Simulate the input by sampling from probability distribution
# 4. Perform a deterministic calculation of the simulated input
# 5. Summarize the results
# def monte_carlo_generated_circle(
#     points_min=10000, 
#     points_max=1000000,
#     radius_min=0.3, 
#     raidus_max=0.8,
#     origin_min=-300,
#     origin_max=300,
#     dense_points_min=0,
#     dense_points_max=3
#     ):

#     radius = np.random.uniform(radius_min, raidus_max)
#     origin_x = np.random.uniform(origin_min, origin_max)
#     origin_y = np.random.uniform(origin_min, origin_max)
#     origin_z = np.random.uniform(origin_min, origin_max)

#     points = []

#     for _ in range(np.random.randint(points_min, points_max)):

#         x = np.random.uniform(-radius, radius) + origin_x
#         y = np.random.uniform(-radius, radius) + origin_y
#         z = np.random.uniform(-radius, radius) + origin_z

#         points.append((x, y, z))

#         # dist_from_origin = np.sqrt((x-origin_x)**2 + (y-origin_y)**2 + (z-origin_z)**2)
#     points = np.array(points)
    
#     for _ in range(np.random.randint(dense_points_min, dense_points_max)):
#         choosen_point = np.random.choice(points)

#         x = np.random.uniform(-radius/4, radius/4) + choosen_point[0]
#         y = np.random.uniform(-radius/4, radius/4) + choosen_point[1]
#         z = np.random.uniform(-radius/4, radius/4) + choosen_point[2]

#         points = np.concat([points, np.array([x, y, z])], axis=0)

#     return points, np.array([origin_x, origin_y, origin_z, radius])
def generate_manhole(
    n_points_min=5000, 
    n_points_max=100000, 
    radius_min=0.2,
    radius_max=1.2, 
    center_min=0.0,
    center_max=0.0,
    sigma=0.01
):
    n_points = np.random.randint(n_points_min, n_points_max)
    center = np.random.uniform(center_min, center_max, size=3)
    radius = np.random.uniform(radius_min, radius_max, size=3)
    
    # Creates points on a circle line
    theta = np.random.uniform(0, 2*np.pi, n_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full(n_points, center[2]) # Flat Manhole
    
    # Add Noise (simulated measurement errors)
    noise = np.random.normal(0, sigma, (n_points, 3)) 
    points = np.stack([x, y, z], axis=1) + noise
    return points, center, radius



def eval_center_robustness(
    center_func,
    n_samples_per_test=10000,
    n_points_min=5000, 
    n_points_max=100000,
    center_min=-1000.0,
    center_max=1000.0,
    radius_min=0.2,
    radius_max=1.2, 
):

    os.makedirs("./output", exist_ok=True)

    # --- Run Experiments ---
    # - With Data Generation
    # - L2 Calculation
    results = dict()

    sigmas = np.vectorize(lambda x: round(x, 2))(np.linspace(0.0, 0.5, 20))

    for cur_sigma in sigmas:
        results[cur_sigma] = []

        for sample_idx in range(n_samples_per_test):
            points, center, radius = generate_manhole(
                n_points_min=n_points_min, 
                n_points_max=n_points_max,  
                center_min=center_min,
                center_max=center_max,
                radius_min=radius_min,
                radius_max=radius_max, 
                sigma=cur_sigma
            )

            pred_x, pred_y, pred_z = center_func(points)

            # L2 Error
            pred = np.array([pred_x, pred_y, pred_z])
            # error_distance = np.sqrt((pred_x-center[0])**2 + \
            #                          (pred_y-center[1])**2 + \
            #                          (pred_z-center[2])**2)
            error_distance = np.linalg.norm(pred - center)

            results[cur_sigma].append(error_distance)

    # --- Result Evaluation/Plotting ---
    plot_results = []
    for key, value in results.items():
        value = np.array(value)
        mean = np.mean(value)
        std = np.std(value)
        plot_results.append([key, mean, std])
    sorted(plot_results, key=lambda x: x[0])
    sigmas, means, stds = zip(*plot_results)

    plt.style.use('ggplot')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

    ax.errorbar(sigmas, means, yerr=stds, fmt='-o', capsize=5, label="Mean L2 Error ± Std")
    ax.set_xlabel("Input Noise (Sigma)")
    ax.set_ylabel("Center Error (L2)")
    ax.set_title("Robustness Anylsis of Center Detection")
    ax.legend()
    ax.grid(True)
    
    plt.savefig("./output/center_robustness.png")
    plt.close(fig)
    

def eval_ellipse_precision(
    center_func,
    n_samples_per_test=10000,
    n_points_min=50000, 
    n_points_max=50000,
    center_min=-1000.0,
    center_max=1000.0,
    radius_min=0.2,
    radius_max=1.2, 
    sigma=0.2
):
    samples = n_samples_per_test
    # preds = []
    # centers = []
    errors = []

    for _ in range(samples):
        points, center, radius = generate_manhole(
                n_points_min=n_points_min, 
                n_points_max=n_points_max,  
                center_min=center_min,
                center_max=center_max,
                radius_min=radius_min,
                radius_max=radius_max, 
                sigma=sigma
            )
        cur_pred = center_func(points)
        # preds.append([cur_pred[0], cur_pred[1]])
        # centers.append([center[0], center[1]])
        error_vec = np.array([cur_pred[0] - center[0], cur_pred[1] - center[1]])
        errors.append(error_vec)

    # preds = np.array(preds)
    # centers = np.array(centers)
    errors = np.array(errors)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    ax.scatter(preds[:,0], preds[:, 1], s=1, alpha=0.3, label="Detections")
    plot_ellipse_precision(errors, np.array([0.0, 0.0]), ax)

    ax.set_title(f"Spatial Precision (Error Distribution) at Sigma={sigma}")
    ax.legend()
    plt.grid(True)
    plt.savefig("./output/center_precision.png")
    plt.close(fig)



def plot_ellipse_precision(
    detected_centers, 
    true_center, 
    ax, 
    n_std=2.0
):
    """
    Creates a Confidence Ellipse based on Errordistribution.
    """
    errors = detected_centers - true_center

    # covariance matrix
    cov = np.cov(errors.T)

    # Eigenvalues and Eigenvectors for Ellipse setup
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[order]

    # Angle of Ellipse
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # Width and Height of Ellipse
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    # Draw Ellipse
    ellipse = Ellipse(
        xy=true_center,
        width=width,
        height=height,
        angle=angle,
        edgecolor='red',
        fc='none',
        lw=2,
        label=f'{n_std}σ Confidence'
    )
    ax.add_patch(ellipse)
    ax.scatter(true_center[0], true_center[1], c='red', marker='+', s=100, label='Truth')
    return ax











