# -----------
# > Imports <
# -----------
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms



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
    radius = np.random.uniform(radius_min, radius_max, size=None)
    
    # Creates points on a circle line -> Uniform distribution in
    r = radius * np.sqrt(np.random.uniform(0, 1, n_points))
    theta = np.random.uniform(0, 2*np.pi, n_points)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
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
    n_points=None,
    center_min=-1000.0,
    center_max=1000.0,
    radius_min=0.2,
    radius_max=1.2, 
    sigma_min=0.0,
    sigma_max=0.5,
    sigmas=None,
    sigma_n_values=5,
    reset_dir=False,
    method="mesqra",
    summary_sigma_ax=None,
    summary_points_ax=None,
    single_sigma_sample_ax=None,
    single_n_points_sample_ax=None,
):

    path = "./output/monte-carlo-gt-check"
    if reset_dir:
        os.makedirs(path, exist_ok=True)
        shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


    print("Starting with Center Robustness Evaluation")

    # --- Run Experiments ---
    # - With Data Generation
    # - L2 Calculation
    # results = dict()
    sample_inputs = {}
    sample_inputs_points = {}
    sigma_results = {}
    point_results = {}

    if not sigmas:
        # sigmas = np.vectorize(lambda x: round(x, 2))(np.linspace(sigma_min, sigma_max, sigma_n_values))
        sigmas = np.linspace(sigma_min, sigma_max, sigma_n_values).round(2)

    print("Beginning Looping...")

    for cur_sigma in tqdm(sigmas, total=len(sigmas), desc="Monte-Carlo-Center-GT Testing"):
        sigma_results[cur_sigma] = []
        sample_inputs[cur_sigma] = []

        for sample_idx in tqdm(range(n_samples_per_test), total=n_samples_per_test):
            if n_points is not None:
                n_points_min = 406  # 50 percentile
                n_points_max = 406+1  # 50 percentile
            points, center, radius = generate_manhole(
                n_points_min=n_points_min, 
                n_points_max=n_points_max,  
                center_min=center_min,
                center_max=center_max,
                radius_min=radius_min,
                radius_max=radius_max, 
                sigma=cur_sigma
            )

            centers = center_func(points, method=method)

            if len(centers) > 1 or len(centers) < 0:
                raise ValueError(f"Only 1 result expected but got {len(centers)}")

            # for cur_center in centers:
            cur_center = centers[0]
            pred_x, pred_y, pred_z = cur_center

            # L2 Error
            pred = np.array([pred_x, pred_y, pred_z])
            # error_distance = np.sqrt((pred_x-center[0])**2 + \
            #                          (pred_y-center[1])**2 + \
            #                          (pred_z-center[2])**2)
            error_distance = np.linalg.norm(pred - center)

            pred_rel = pred - center

            points_rel = points - center

            if len(sample_inputs[cur_sigma]) < 6:
                sample_inputs[cur_sigma].append({
                    "points": points_rel.copy(),
                    "center": np.zeros_like(center),  # center.copy(),
                    "pred": pred_rel.copy(),  # pred.copy()
                    "radius": radius
                })

            # results[cur_sigma].append(error_distance)
            # FIXME: What to do, if no manhole was found? -> currently catched by error
            sigma_results[cur_sigma].append({
                "pred": pred_rel,
                "error": error_distance
            })

    # points loop (collect data)
    for cur_n_points in n_points:
        point_results[cur_n_points] = []
        sample_inputs_points[cur_n_points] = []

        for sample_idx in range(n_samples_per_test):

            points, center, radius = generate_manhole(
                n_points_min=cur_n_points, 
                n_points_max=cur_n_points+1,  
                center_min=center_min,
                center_max=center_max,
                radius_min=radius_min,
                radius_max=radius_max, 
                sigma=0.0
            )

            centers = center_func(points, method=method)

            if len(centers) > 1 or len(centers) < 0:
                raise ValueError(f"Only 1 result expected but got {len(centers)}")

            # for cur_center in centers:
            cur_center = centers[0]
            pred_x, pred_y, pred_z = cur_center

            # L2 Error
            pred = np.array([pred_x, pred_y, pred_z])
            error_distance = np.linalg.norm(pred - center)

            pred_rel = pred - center
            points_rel = points - center

            if len(sample_inputs_points[cur_n_points]) < 6:
                sample_inputs_points[cur_n_points].append({
                    "points": points_rel.copy(),
                    "center": np.zeros_like(center),  # center.copy(),
                    "pred": pred_rel.copy(),  # pred.copy()
                    "radius": radius
                })

            point_results[cur_n_points].append({
                "pred": pred_rel,
                "error": error_distance
            })

    # --- Result Evaluation/Plotting ---
    # Summary Plot
    plot_limit = 0.1
    plot_results = []

    for sigma, values in sigma_results.items():

        errors = np.array([v["error"] for v in values])
        preds = np.array([v["pred"] for v in values])

        plot_results.append({
            "sigma": sigma,
            "mean": errors.mean(),
            "std": errors.std(),
            "mean": errors.mean(),
            "min": errors.min(),
            "sum": errors.sum(),
            "preds": preds
        })

    plot_results = sorted(plot_results, key=lambda x: x["sigma"])
    # sorted(plot_results, key=lambda x: x[0])
    # sigmas, means, stds, preds = zip(*plot_results)

    # also collect the results from points
    plot_point_results = []

    for n_pts, values in point_results.items():

        errors = np.array([v["error"] for v in values])
        preds  = np.array([v["pred"] for v in values])

        plot_point_results.append({
            "n_points": n_pts,
            "mean": errors.mean(),
            "std": errors.std(),
            "min": errors.min(),
            "sum": errors.sum(),
            "preds": preds,
        })

    plot_point_results = sorted(plot_point_results,
                                key=lambda x: x["n_points"])


    # get values for plotting
    sigmas = [r["sigma"] for r in plot_results]
    means = [r["mean"] for r in plot_results]
    stds = [r["std"] for r in plot_results]
    mins = [r["min"] for r in plot_results]
    sums = [r["sum"] for r in plot_results]

    # Summary Plot
    summary_plot(
        summary_sigma_ax, 
        sigmas, 
        means, 
        stds, 
        mins, 
        sums, 
        method, 
        plot_limit,
        "Noise \u03C3",
        save_plot=False
    )

    # get values for plotting
    n_points_ = [r["n_points"] for r in plot_point_results]
    means = [r["mean"] for r in plot_point_results]
    stds = [r["std"] for r in plot_point_results]
    mins = [r["min"] for r in plot_point_results]
    sums = [r["sum"] for r in plot_point_results]

    summary_plot(
        summary_points_ax, 
        n_points_, 
        means, 
        stds, 
        mins, 
        sums, 
        method, 
        plot_limit,
        "n-Points",
        save_plot=False
    )

    # Detail Plot
    plot_single_value_sample(
        single_sigma_sample_ax, 
        plot_results, 
        plot_limit,
        value_idx="sigma",
        save_plot=False
    )

    plot_single_value_sample(
        single_n_points_sample_ax, 
        plot_point_results, 
        plot_limit,
        value_idx="n_points",
        save_plot=False
    )


    # Plotting Input Samples
    plot_sample_input(sample_inputs, is_sigma=True, method=method)
    plot_sample_input(sample_inputs_points, is_sigma=False, method=method)
    


def summary_plot(ax, values, means, stds, mins, sums, method, plot_limit, x_label, save_plot=True):
    plt.style.use("ggplot")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,5))
    
    ax.errorbar(
        values,
        means,
        yerr=stds,
        fmt="-o",
        capsize=4
    )

    for value, mean, min_val, sum_val in zip(values, means, mins, sums):
        text = (
            f"μ={mean:.3f}\n"
            f"min={min_val:.3f}\n"
            f"Σ={sum_val:.3f}"
        )

        ax.annotate(
            text,
            (value, mean),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(
                facecolor="white",
                alpha=0.8,
                edgecolor="gray"
            )
        )

    # stats_text = (
    #     f"Mean: {np.mean(means):.3f}\n"
    #     f"Min : {np.min(mins):.3f}\n"
    #     f"Sum : {np.sum(sums):.3f}"
    # )

    # ax.text(
    #     0.98,
    #     0.98,
    #     stats_text,
    #     transform=ax.transAxes,
    #     ha="right",
    #     va="top",
    #     fontsize=9,
    #     bbox=dict(
    #         facecolor="white",
    #         alpha=0.8,
    #         edgecolor="gray"
    #     )
    # )

    ax.set_xlabel(x_label)
    ax.set_ylabel("L2 Error")
    # ax.set_title("Center Detection Robustness")
    # ax.axis("equal")

    # plt.tight_layout()
    if save_plot:
        plt.savefig(os.path.join(path, f"{method}_center_summary.png"))
        plt.close()

    return ax



def plot_single_value_sample(ax, plot_results, plot_limit, value_idx, save_plot=True):
    for idx, res in enumerate(plot_results):

        value = res[value_idx]
        preds = res["preds"]

        if ax is None:
            fig, cur_ax = plt.subplots(figsize=(6,6))
        else:
            cur_ax = ax[idx]

        xy = preds[:, :2]

        inside_mask = (
            (np.abs(xy[:, 0]) <= plot_limit) &
            (np.abs(xy[:, 1]) <= plot_limit)
        )

        inside = xy[inside_mask]
        outside = xy[~inside_mask]

        cur_ax.scatter(
            inside[:, 0],
            inside[:, 1],
            s=5,
            alpha=0.35,
            label="Predictions"
        )

        outside_clipped = outside.copy()

        outside_clipped[:, 0] = np.clip(
            outside_clipped[:, 0],
            -plot_limit,
            plot_limit
        )

        outside_clipped[:, 1] = np.clip(
            outside_clipped[:, 1],
            -plot_limit,
            plot_limit
        )

        cur_ax.scatter(
            outside_clipped[:, 0],
            outside_clipped[:, 1],
            marker="^",
            color="red",
            s=35,
            label="Clipped outlier"
        )

        cur_ax.scatter(
            0,
            0,
            color="red",
            marker="x",
            s=120,
            label="Ground Truth"
        )

        if len(outside) == 0:
            errors_in = np.linalg.norm(inside, axis=1)
            text = (
                "Outliers : 0\n"
                f"Sum error: {errors_in.sum():.3f}"
            )
        else:
            errors_in = np.linalg.norm(inside, axis=1)
            errors_out = np.linalg.norm(outside, axis=1)
            total_error = errors_in.sum() + errors_out.sum()
            text = (
                f"Outliers : {len(outside)}\n"
                f"Total sum error: {total_error:.3f}\n"
                f"Out sum error: {errors_out.sum():.3f}"
            )

        cur_ax.text(
            0.98,
            0.98,
            text,
            transform=cur_ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(
                facecolor="white",
                alpha=0.8,
                edgecolor="gray"
            )
        )

        cur_ax.axhline(0, color="gray", ls="--", lw=1)
        cur_ax.axvline(0, color="gray", ls="--", lw=1)

        cur_ax.set_xlim(-plot_limit, plot_limit)
        cur_ax.set_ylim(-plot_limit, plot_limit)
        cur_ax.set_aspect("equal")

        cur_ax.set_xlabel("x error")
        cur_ax.set_ylabel("y error")
        # cur_ax.set_title(f"Prediction Scatter (σ={sigma})")

        # plot_limit = 0,1  # 0.10  # 10 cm
        # all_preds = np.vstack([r["preds"] for r in plot_results])
        # plot_limit = np.max(np.abs(all_preds)) * 1.05
        # plot_limit = np.percentile(np.abs(all_preds), 99) * 1.1

        cur_ax.set_xlim(-plot_limit, plot_limit)
        cur_ax.set_ylim(-plot_limit, plot_limit)
        cur_ax.set_aspect("equal", adjustable="box")

        # cur_ax.axis("equal")
        cur_ax.grid(True)
        
        # cur_ax.legend()
        # if sigma_idx == 0:
        #     cur_ax.legend(loc="upper right")

        # plt.tight_layout()
        if save_plot:
            plt.savefig(os.path.join(path, f"{method}_center_sigma_{value_idx:.2f}.png"))
            plt.close()

    return ax


def plot_sample_input(sample_inputs, is_sigma, method):
    for value, samples in sample_inputs.items():

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(12,8)
        )

        axes = axes.ravel()

        for ax, sample in zip(axes, samples):

            pts = sample["points"]

            ax.scatter(
                pts[:,0],
                pts[:,1],
                s=0.5,
                alpha=1.0,
                color="red",
                label="Manhole Points"
            )

            ax.scatter(
                sample["center"][0],
                sample["center"][1],
                color="green",
                marker="x",
                s=60,
                label="GT"
            )

            # ax.scatter(
            #     sample["pred"][0],
            #     sample["pred"][1],
            #     color="red",
            #     marker="+",
            #     s=60,
            #     label="Prediction"
            # )

            # sample_limit = sample["radius"]  # 0.8 * 1.3
            # ax.set_xlim(-sample_limit, sample_limit)
            # ax.set_ylim(-sample_limit, sample_limit)
            # ax.axis("equal")
            # ax.set_xticks([])
            # ax.set_yticks([])
            ax.legend()

        path = "./output/monte-carlo-gt-check"

        if is_sigma:
            plt.suptitle(f"Example Inputs (σ={value})")
            # plt.tight_layout()

            plt.savefig(os.path.join(path, f"{method}_input_examples_sigma_{value:.2f}.png"))
        else:
            plt.suptitle(f"Example Inputs (n-Points={value})")
            # plt.tight_layout()

            plt.savefig(os.path.join(path, f"{method}_input_examples_npoints_{value:.2f}.png"))

        plt.close()



def eval_ellipse_precision(
    center_func,
    n_samples_per_test=10000,
    n_points_min=5000, 
    n_points_max=5001,
    center_min=-1000.0,
    center_max=1000.0,
    radius_min=0.2,
    radius_max=1.2, 
    sigma=0.2,
    method="mesqra"
):
    samples = n_samples_per_test
    preds = []
    # centers = []
    errors = []

    for _ in tqdm(range(samples), total=samples, desc="Error Ellipse Precision"):
        points, center, radius = generate_manhole(
                n_points_min=n_points_min, 
                n_points_max=n_points_max,  
                center_min=center_min,
                center_max=center_max,
                radius_min=radius_min,
                radius_max=radius_max, 
                sigma=sigma
            )
        cur_pred = center_func(points, method=method)

        if len(cur_pred) > 1 or len(cur_pred) < 0:
            raise ValueError(f"Only 1 result expected but got {len(cur_pred)}")

        # for cur_center in centers:
        cur_pred = cur_pred[0]

        preds.append([cur_pred[0], cur_pred[1]])
        # centers.append([center[0], center[1]])
        error_vec = np.array([cur_pred[0] - center[0], cur_pred[1] - center[1]])
        errors.append(error_vec)

    preds = np.array(preds)
    # centers = np.array(centers)
    errors = np.array(errors)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    ax.scatter(preds[:,0], preds[:, 1], s=1, alpha=0.3, label="Detections")
    plot_ellipse_precision(errors, np.array([0.0, 0.0]), ax)

    ax.set_title(f"Spatial Precision (Error Distribution) at Sigma={sigma}")
    ax.legend()
    plt.grid(True)
    plt.savefig(f"./output/monte-carlo-gt-check/{method}_center_precision.png")
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











