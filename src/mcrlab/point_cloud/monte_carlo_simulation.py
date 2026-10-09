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

# from mcrlab.helper import save_dir_creation



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



def generate_realistic_manhole(
    center_min=0.0,
    center_max=0.0,
    radius_min=0.2,
    radius_max=1.2,
    target_n_points=1000,
    # Line Scan Parameters
    n_lines=(5, 20),          # Number of scan lines crossing the manhole
    line_spacing_jitter=0.2,          # Random variation in line-to-line spacing (0.0 to 0.5)
    # point_density_per_meter=300,      # In-line point sampling density
    # Waviness Parameters
    wave_amplitude=0.0,             # Max wave distortion perpendicular to line (in meters)
    wave_frequency=25.0,              # Frequency of waves along the line
    # Noise & Measurement Error
    sigma=0.005,                      # Gaussian measurement noise along scan lines
    # Outliers
    outlier_ratio=0.10,               # Fractional amount of total points that are outliers
    near_edge_outlier_prob=0.8,       # Fraction of outliers spawned near the rim vs background
    # Real-World Missingness / Occlusion Parameter
    occlusion_severity=0.25           # Control amount of structural dropout [0.0 = none, 1.0 = heavy]
):
    """
    Generates a realistic 3D point cloud of a manhole incorporating mobile LiDAR scan lines,
    line waviness, non-uniform line spacing, and near-boundary/background outliers.
    """
    # 1. Base Geometry Parameters
    center = np.random.uniform(center_min, center_max, size=3)
    radius = np.random.uniform(radius_min, radius_max)
    
    # Random orientation angle of scan lines across the manhole plane (0 to pi)
    scan_angle = np.random.uniform(0, np.pi)
    cos_a, sin_a = np.cos(scan_angle), np.sin(scan_angle)

    # 2. Determine Non-Uniform Line Positions across diameter (-radius to +radius)
    if isinstance(n_lines, (list, tuple)):
        n_lines = np.random.randint(n_lines[0], n_lines[1] + 1)
    else:
        n_lines = int(n_lines)

    # Base uniform line offsets relative to center
    base_offsets = np.linspace(-radius * 0.95, radius * 0.95, n_lines)
    
    # Add random jitter to line spacing
    spacing_step = (2 * radius) / n_lines
    offsets = base_offsets + np.random.uniform(
        -spacing_step * line_spacing_jitter, 
        spacing_step * line_spacing_jitter, 
        size=n_lines
    )
    # Clip to keep lines within disk boundary
    offsets = np.clip(offsets, -radius * 0.98, radius * 0.98)

    # 3. Compute length of each line
    chord_lengths = 2 * np.sqrt(radius**2 - offsets**2)
    total_length = np.sum(chord_lengths)

    # Over-sample candidate points initially to allow clean masking
    sample_multiplier = 3.0 if occlusion_severity > 0 else 1.2
    raw_n_inliers = int((target_n_points * (1.0 - outlier_ratio)) * sample_multiplier)
    
    pts_per_line = np.round((chord_lengths / np.sum(chord_lengths)) * raw_n_inliers).astype(int)

    # 4. Point distribution (inlier, outlier) based on target n points
    # n_outliers = int(target_n_points * outlier_ratio)
    # n_inliers = target_n_points - n_outliers

    # # compute points per line
    # pts_per_line = np.round((chord_lengths / total_length) * n_inliers).astype(int)
    
    # # correction of rounding errors to ensure the sum equals n_inliers
    # diff = n_inliers - np.sum(pts_per_line)
    # if diff != 0:
    #     pts_per_line[np.argmax(chord_lengths)] += diff  # adjust the line with the longest chord to compensate

    # 5. Generate Points along each circle line
    inlier_points_list = []
    
    for offset, chord_len, n_pts in zip(offsets, chord_lengths, pts_per_line):
        if n_pts <= 0:
            continue

        # Half length of chord at this offset
        # -> length of line in circle at this offset: 2 * sqrt(radius^2 - offset^2)
        chord_half_len = np.sqrt(radius**2 - offset**2)
        # coordinates along scan line
        s = np.linspace(-chord_half_len, chord_half_len, n_pts)
        
        # Add sinusoidal waviness along the scan path
        phase = np.random.uniform(0, 2 * np.pi)
        d_perp = wave_amplitude * np.sin(wave_frequency * s + phase)
        
        # Coordinates in scan-line local system (u: along line, v: perpendicular)
        u = s
        v = offset + d_perp
        
        # Rotate back to world system
        x_local = u * cos_a - v * sin_a
        y_local = u * sin_a + v * cos_a
        z_local = np.zeros_like(x_local)
        
        # Stack local points
        line_pts = np.stack([x_local, y_local, z_local], axis=1)
        inlier_points_list.append(line_pts)
        
    inliers = np.vstack(inlier_points_list)
    n_inliers = len(inliers)

    # 6. Add Realistic Asynchronous Masking / Occlusions
    if occlusion_severity > 0.0:
        keep_mask = np.ones(len(inliers), dtype=bool)
        
        # A. Sector dropout (Missing slice/side)
        if np.random.rand() < (0.5 * occlusion_severity + 0.3):
            start_angle = np.random.uniform(0, 2 * np.pi)
            angle_span = np.random.uniform(np.pi / 6, np.pi / 2) * occlusion_severity
            pts_angles = np.arctan2(inliers[:, 1], inliers[:, 0]) % (2 * np.pi)
            
            end_angle = (start_angle + angle_span) % (2 * np.pi)
            if start_angle < end_angle:
                sector_mask = (pts_angles >= start_angle) & (pts_angles <= end_angle)
            else:
                sector_mask = (pts_angles >= start_angle) | (pts_angles <= end_angle)
            
            keep_mask &= ~sector_mask

        # B. Linear Dropout / Stripe (Line through manhole without points)
        if np.random.rand() < (0.6 * occlusion_severity + 0.2):
            stripe_angle = np.random.uniform(0, np.pi)
            stripe_width = radius * np.random.uniform(0.05, 0.25) * occlusion_severity
            stripe_dist = np.abs(
                inliers[:, 0] * np.sin(stripe_angle) - inliers[:, 1] * np.cos(stripe_angle)
            )
            keep_mask &= (stripe_dist > stripe_width)

        # C. Localized Circular Pockets (Wear / Potholes / Damage)
        n_pockets = np.random.randint(1, max(2, int(4 * occlusion_severity)))
        for _ in range(n_pockets):
            pocket_r = np.random.uniform(radius * 0.1, radius * 0.35) * occlusion_severity
            pocket_center = np.random.uniform(-radius * 0.7, radius * 0.7, size=2)
            dist_to_pocket = np.linalg.norm(inliers[:, :2] - pocket_center, axis=1)
            keep_mask &= (dist_to_pocket > pocket_r)

        inliers = inliers[keep_mask]

    # 7. Resample Inliers to Target Point Count
    n_outliers = int(target_n_points * outlier_ratio)
    desired_inliers = max(1, target_n_points - n_outliers)
    
    if len(inliers) > desired_inliers:
        indices = np.random.choice(len(inliers), size=desired_inliers, replace=False)
        inliers = inliers[indices]

    # 8. Add Measurement Gaussian Noise (Sigma)
    inliers += np.random.normal(0, sigma, size=inliers.shape)

    # 9. Generate Outliers
    if n_outliers > 0:
        n_edge_outliers = int(n_outliers * near_edge_outlier_prob)
        n_bg_outliers = n_outliers - n_edge_outliers
        
        outliers_list = []

        # Edge Outliers: Concentrated around the manhole perimeter (r ≈ radius)
        if n_edge_outliers > 0:
            edge_angles = np.random.uniform(0, 2 * np.pi, n_edge_outliers)
            # Distance centered around rim with small deviation
            edge_r = radius + np.random.normal(0, radius * 0.15, n_edge_outliers)
            edge_x = edge_r * np.cos(edge_angles)
            edge_y = edge_r * np.sin(edge_angles)
            edge_z = np.random.normal(0, sigma * 3, n_edge_outliers)  # vertical displacement
            outliers_list.append(np.stack([edge_x, edge_y, edge_z], axis=1))
            
        # Background Outliers: Uniformly scattered in bounding box around manhole
        if n_bg_outliers > 0:
            bg_x = np.random.uniform(-radius * 1.5, radius * 1.5, n_bg_outliers)
            bg_y = np.random.uniform(-radius * 1.5, radius * 1.5, n_bg_outliers)
            bg_z = np.random.uniform(-0.05, 0.05, n_bg_outliers)
            outliers_list.append(np.stack([bg_x, bg_y, bg_z], axis=1))
            
        outliers = np.vstack(outliers_list)
        all_points = np.vstack([inliers, outliers])
    else:
        all_points = inliers

    # Shuffle points so algorithms cannot exploit scan order
    np.random.shuffle(all_points)
    
    # Shift to target center
    all_points += center

    return all_points, center, radius



def eval_center_robustness(
    center_func,
    n_samples_per_test=10000,
    n_points_min=5000, 
    n_points_max=100000,
    n_points=None,
    center_min=0.0,
    center_max=0.0,
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

    sample_inputs = {}
    sample_inputs_points = {}
    sigma_results = {}
    point_results = {}

    if not sigmas:
        # sigmas = np.vectorize(lambda x: round(x, 2))(np.linspace(sigma_min, sigma_max, sigma_n_values))
        sigmas = np.linspace(sigma_min, sigma_max, sigma_n_values).round(2)

    if n_points is None:
        n_points = [50, 100, 1000]

    print("Beginning Occlusion-Robustness Loop...")

    # --- Sigma Eval Loop ---
    for cur_sigma in tqdm(sigmas, total=len(sigmas), desc=f"Testing Sigma ({method})"):
        sigma_results[cur_sigma] = []
        sample_inputs[cur_sigma] = []

        for sample_idx in range(n_samples_per_test):
            
            # points, center, radius = generate_manhole(
            #     n_points_min=n_points_min, 
            #     n_points_max=n_points_max,  
            #     center_min=center_min,
            #     center_max=center_max,
            #     radius_min=radius_min,
            #     radius_max=radius_max, 
            #     sigma=cur_sigma
            # )

            target_n_points = 406  # 406 is the 50 percentile of the SUD dataset
            # FIXME: from which experiment to know 406 is the mean amount?
            #        -> manhole stats where computed?

            points, center, radius = generate_realistic_manhole(
                center_min=center_min,
                center_max=center_max,
                radius_min=radius_min,
                radius_max=radius_max,
                target_n_points=target_n_points,
                n_lines=np.random.randint(5, 20),
                line_spacing_jitter=np.random.uniform(0.01, 0.3),  
                wave_amplitude=0.0,             # 3mm to 10mm wave distortion
                wave_frequency=25.0,
                sigma=0.0, # cur_sigma,  # 0.005,
                outlier_ratio=np.random.uniform(0.00, 0.03),  # 0-3% outliers
                near_edge_outlier_prob=0.9,
                occlusion_severity=cur_sigma,  # np.random.uniform(0.0, 0.5)
            )

            centers = center_func(points, method=method)

            if len(centers) > 1:
                raise ValueError(f"Only 1 result expected but got {len(centers)}")

            # for cur_center in centers:
            cur_center = centers[0]
            if cur_center is None:
                pred_x, pred_y, pred_z = np.nan, np.nan, np.nan
            else:
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


    print("Beginning Point-Amount Robustness Loop...")

    # -- n-points eval loop ---
    for cur_n_points in tqdm(n_points, total=len(n_points), desc=f"Testing Points ({method})"):
        point_results[cur_n_points] = []
        sample_inputs_points[cur_n_points] = []

        for sample_idx in range(n_samples_per_test):

            # points, center, radius = generate_manhole(
            #     n_points_min=cur_n_points, 
            #     n_points_max=cur_n_points+1,  
            #     center_min=center_min,
            #     center_max=center_max,
            #     radius_min=radius_min,
            #     radius_max=radius_max, 
            #     sigma=0.0
            # )

            points, center, radius = generate_realistic_manhole(
                center_min=center_min,
                center_max=center_max,
                radius_min=radius_min,
                radius_max=radius_max,
                target_n_points=cur_n_points,
                n_lines=np.random.randint(5, 20),
                line_spacing_jitter=np.random.uniform(0.01, 0.3),  
                wave_amplitude=0.0,             # 3mm to 10mm wave distortion
                wave_frequency=25.0,
                sigma=0.0,  # 0.005,
                outlier_ratio=np.random.uniform(0.00, 0.03),  # 0-3% outliers
                near_edge_outlier_prob=0.9,
                occlusion_severity=0.0  # np.random.uniform(0.0, 0.2)
            )

            centers = center_func(points, method=method)

            if len(centers) > 1 or len(centers) < 0:
                raise ValueError(f"Only 1 result expected but got {len(centers)}")

            # for cur_center in centers:
            cur_center = centers[0]
            if cur_center is None:
                pred_x, pred_y, pred_z = np.nan, np.nan, np.nan
            else:
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
        "Occlusion Severity", # "Noise \u03C3",
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
            plt.savefig(os.path.join(path, f"{method}_center_{value_idx:.2f}.png"))
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
            plt.suptitle(f"Example Inputs (Occlusion Severity={value})")  # σ
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











