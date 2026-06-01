# -----------
# > Imports <
# -----------
import os
import psutil

import numpy as np
from PIL import Image
import torch

import open3d as o3d

import numba

from tqdm import tqdm

# dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mcrlab.point_cloud.utils import get_coordinate_attribute, get_intensity_attribute, get_class_attribute
from mcrlab.point_cloud.tensor_wrapper import PointCloudTensor, torch_tensor_to_numpy, torch_tensor_type_to_numpy_type
from mcrlab.image.io import save_single_bev_tile_as_pickle, save_bev_tiles_as_images
from mcrlab.point_cloud.io import save_point_cloud



# ----------------------------
# > Dimensionality Reduction <
# ----------------------------
def pca_projection(point_cloud):
    if isinstance(point_cloud, PointCloudTensor):
        point_cloud = point_cloud.get_as_o3d()

    data_3d = point_cloud.point[get_coordinate_attribute(point_cloud)]
    
    scaler = StandardScaler()
    data_rescaled = scaler.fit_transform(data_3d)

    pca = PCA(n_components=2)
    return pca.fit_transform(data_rescaled)



# ------------------
# > BEV Projection <
# ------------------
# BEV (Bird-Eye-View) are images tile-based over the point cloud
# and grid based (one grid-point is one pixel) but with the difference
# that one grid-point can contain multiple values (because multiple points
# on the same cooridnate) + no RGB values but height, intensity and
# sometimes other values, like normal.
# BEV = point cloud -> discretization -> aggregation -> image
# Idea: 3D -> BEV -> CNN/Transformer -> predictions ((tile_id, px, py), ...)
# -> corresponding 3D points (projection-mapping)

@numba.njit
def _numba_aggregate(px, py, z, intensity, height, width, labels=None, num_classes=0):
    """
    Numba-accelerated per-pixel aggregation for Bird's Eye View (BEV) tiles.

    This function takes a set of points within a tile and computes a 3-channel
    BEV grid, where each pixel contains:
        1. Maximum height (Z) of points falling into that pixel
        2. Minimum height (Z) of points falling into that pixel
        3. Mean intensity of points falling into that pixel
        4. Density
        5. Class/label

    The function is optimized using Numba, which compiles Python loops into
    fast machine code, providing significant speedup over pure Python/Numpy
    aggregation for large point clouds.

    Steps:

    1. **Initialize BEV grid and counters**:
       - `bev` is a 3xHxW array:
           - Channel 0: max height
           - Channel 1: min height (initialized to a very large value)
           - Channel 2: sum of intensities
       - `counts` keeps track of how many points fall into each pixel.

    2. **Loop over all points**:
       - For each point at `(px[i], py[i])`:
           - Update max height: `bev[0, x, y] = max(current, z[i])`
           - Update min height: `bev[1, x, y] = min(current, z[i])`
           - Accumulate intensity: `bev[2, x, y] += intensity[i]`
           - Increment counts for this pixel: `counts[x, y] += 1`

    3. **Finalize mean intensity and handle empty pixels**:
       - Loop over all pixels `(i, j)`:
           - If at least one point falls into the pixel:
               - Divide accumulated intensity by the count to get the mean
           - If no points fall into the pixel:
               - Set min height and intensity to 0
               - Prevents uninitialized or invalid values

    4. **Return BEV grid**:
       - The output `bev` is a 3xHxW numpy array representing the tile,
         ready to be normalized or combined with other tiles.

    Args:
        px (np.ndarray): X pixel coordinates of points within the tile (int32)
        py (np.ndarray): Y pixel coordinates of points within the tile (int32)
        z (np.ndarray): Z coordinates (heights) of points
        intensity (np.ndarray): intensity values of points
        height (int): number of pixels along Y-axis of the tile
        width (int): number of pixels along X-axis of the tile

    Returns:
        np.ndarray: 3xHxW BEV grid with max height, min height, and mean intensity
    """
    if labels is None:  
        bev = np.zeros((4, height, width), dtype=np.float32)
    else:
        bev = np.zeros((5, height, width), dtype=np.float32)
    # print("Checkpoint 1", bev.nbytes / 1024**2, "MB")
    counts = np.zeros((height, width), dtype=np.int32)
    # print("Checkpoint 2", counts.nbytes / 1024**2, "MB")
    
    # for class voting
    if labels is not None:  
        class_counts = np.zeros((height, width, num_classes), dtype=np.int32)
        # print("Checkpoint 3", class_counts.nbytes / 1024**2, "MB")


    bev[1, :, :] = 1e6  # initialize min height
    
    for i in range(len(px)):
        x = px[i]
        y = py[i]

        # bev[0, x, y] = max(bev[0, x, y], z[i])        # max height
        # bev[1, x, y] = min(bev[1, x, y], z[i])        # min height
        # bev[2, x, y] += intensity[i]                  # sum intensity
        
        # counts[x, y] += 1

        # # class voting
        # if labels is not None: 
        #     cls = labels[i]
        #     class_counts[x, y, cls] += 1

        # Swap x and y when indexing BEV
        # x => width / columns
        # y => height / rows
        bev[0, y, x] = max(bev[0, y, x], z[i])  # max z
        bev[1, y, x] = min(bev[1, y, x], z[i])  # min z
        bev[2, y, x] += intensity[i]
        counts[y, x] += 1

        # class voting
        if labels is not None:
            cls = labels[i]
            class_counts[y, x, cls] += 1
    
    # finalize mean intensity
    for cur_y in range(height):        # row
        for cur_x in range(width):     # column
            if counts[cur_y, cur_x] > 0:
                # finalize delta z -> height difference
                delta_z = bev[0, cur_y, cur_x] - bev[1, cur_y, cur_x]
                bev[1, cur_y, cur_x] = delta_z

                # finalize mean intensity
                bev[2, cur_y, cur_x] /= counts[cur_y, cur_x]

                # create density (point amounts)
                bev[3, cur_y, cur_x] = np.log1p(counts[cur_y, cur_x])
                # fix: bev[3, cur_y, cur_x] = np.log(1.0 + counts[cur_y, cur_x])

                # majority class
                if labels is not None:
                    best_class = 255
                    best_count = -1
                    for c in range(num_classes):
                        if class_counts[cur_y, cur_x, c] > best_count:
                            best_count = class_counts[cur_y, cur_x, c]
                            best_class = c
                    bev[4, cur_y, cur_x] = best_class
            else:
                bev[1, cur_y, cur_x] = 0.0
                bev[2, cur_y, cur_x] = 0.0
                bev[3, cur_y, cur_x] = 0.0
                if labels is not None:
                    bev[4, cur_y, cur_x] = 255  # empty pixel = ignore index
    
    return bev



def bev_projection(point_cloud, pc_id,
                   tile_size=10.0, resolution=0.5, overlap=0.0,
                   include_class=False,
                   direct_single_saving=True, single_saving_path=None,
                   save_3d_patches=False,
                   sample_path=None):
    """
    Projects a 3D point cloud into Bird's Eye View (BEV) tiles using Numba for fast per-pixel aggregation.

    This function converts a 3D point cloud into a set of 2D BEV grids by dividing the XY-plane into 
    square tiles of a given size. Each tile is discretized into pixels, and for each pixel, the following
    features are computed:

        1. Maximum height (Z coordinate) of points in the pixel
        2. Delta Z (Chang of height)
        3. Density (How many points)
        4. Mean intensity of points in the pixel
        5. Class Labels (optional)

    The BEV projection is done efficiently using Numba-accelerated loops for pixel aggregation,
    which provides significant speedup compared to standard Python or NumPy operations.

    Steps performed by the function:

    1. **Extract points and intensity**:
       - Convert the point cloud object to a numpy array if needed.
       - Extract X, Y, Z coordinates and intensity values from the point cloud.

    2. **Compute global bounds**:
       - Determine the minimum and maximum X and Y coordinates of the point cloud.
       - These bounds define the total area to be covered by tiles.

    3. **Initialize containers**:
       - Prepare empty lists for `tiles` and `meta`.
       - Initialize a `tile_id` counter for assigning unique IDs to tiles.

    4. **Iterate over tiles**:
       - Loop over the XY-plane in steps of `tile_size`.
       - `cur_x` and `cur_y` represent the bottom-left corner of the current tile.

    5. **Select points inside the tile**:
       - Create a boolean mask to select all points that lie within the current tile.
       - Skip the tile if no points are found.

    6. **Compute pixel coordinates within the tile**:
       - Convert the tile-local XY coordinates to discrete pixel indices using the `resolution`.
       - Calculate the height and width of the grid (number of pixels per tile).

    7. **Aggregate per-pixel values using Numba**:
       - Call `_numba_aggregate` to compute:
           - Maximum height per pixel
           - Minimum height per pixel
           - Sum and then mean of intensity per pixel
       - Numba accelerates the aggregation loops for large point clouds.

    8. **Normalize features**:
       - Normalize maximum height by the global maximum Z value.
       - Normalize mean intensity by the global maximum intensity.

    9. **Store tile and metadata**:
       - Append the BEV grid to the `tiles` list.
       - Store metadata for each tile, including:
           - `tile_id`: unique identifier for the tile
           - `cur_x`, `cur_y`: tile's bottom-left coordinates
           - `global_indices`: original point indices that fall inside this tile

    10. **Return results**:
        - `tiles`: list of 3xHxW BEV grids for all tiles
        - `meta`: list of metadata dictionaries, one per tile

    Args:
        point_cloud: Point cloud object or Nx4 numpy array with fields (x, y, z, intensity)
        tile_size: float, size of each square tile in meters (default: 10.0)
        resolution: float, size of each pixel in meters (default: 0.5)

    Returns:
        tiles: list of 3xHxW numpy arrays representing BEV grids
        meta: list of dictionaries containing metadata for each tile
    """
    if direct_single_saving and single_saving_path is None:
        raise ValueError("If using 'direct_single_saving', 'single_saving_path' must have an string value and can't be None.")

    # extract points and intensity
    # if hasattr(point_cloud, "get_as_o3d"):
    if isinstance(point_cloud, PointCloudTensor):
        point_cloud = point_cloud.get_as_o3d()
    
    points  = point_cloud.point[get_coordinate_attribute(point_cloud)].numpy()
    intensities = point_cloud.point[get_intensity_attribute(point_cloud)].numpy().ravel()

    if include_class:
        labels = point_cloud.point[get_class_attribute(point_cloud)].numpy().astype(np.int32).reshape(-1)

        # num_classes = int(labels.max()) + 1
        # print("Num_classes:", str(num_classes))

        unique_classes = np.unique(labels)
        class_map = {c: i for i, c in enumerate(unique_classes)}
        labels = np.array([class_map[c] for c in labels], dtype=np.int32)
        num_classes = len(unique_classes)
        # print("Num_classes:", str(num_classes))
    else:
        labels = None
        num_classes = 0
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    if not direct_single_saving:
        tiles = []
        meta = []
    else:
        patch_info = []

    tile_id = 0

    # calc step-size
    #    normally just tile_size, but
    #    if you want to get overlapped tiles 
    #    then the stepsize must be smaller
    stride = tile_size - overlap

    grid_x_min = float(np.floor(x.min() / stride) * stride)
    grid_x_max = float(np.ceil(x.max() / stride) * stride)

    grid_y_min = float(np.floor(y.min() / stride) * stride)
    grid_y_max = float(np.ceil(y.max() / stride) * stride)

    if sample_path:
        max_sample_amount = 5
        sample_tiles = []
        sample_saving_completed = False
        nx = int(np.ceil((x_max - x_min) / stride))
        ny = int(np.ceil((y_max - y_min) / stride))
        max_steps = nx * ny
        start_step = np.random.randint(0, max(0, max_steps - max_sample_amount)+1)

    # iterate over tiles
    cur_step = -1
    for cur_x in np.arange(grid_x_min, grid_x_max, stride):
        for cur_y in np.arange(grid_y_min, grid_y_max, stride):
            cur_step += 1

            is_last_x = (cur_x + stride >= grid_x_max)
            is_last_y = (cur_y + stride >= grid_y_max)
            
            # include points on the border in last grids/tiles
            x_upper_bound = cur_x + tile_size
            y_upper_bound = cur_y + tile_size
            
            # select points inside this tile
            mask = (
                (x >= cur_x) & (x <= x_upper_bound if is_last_x else x < x_upper_bound) &
                (y >= cur_y) & (y <= y_upper_bound if is_last_y else y < y_upper_bound)
            )

            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            
            points_tile = points[idxs]
            intensities_tile = intensities[idxs]

            if labels is not None:
                labels_tile = labels[idxs]
            else:
                labels_tile = None
            
            # pixel coordinates
            # points_x = ((points_tile[:, 0] - cur_x) / resolution).astype(np.int32)
            # points_y = ((points_tile[:, 1] - cur_y) / resolution).astype(np.int32)
            points_x = np.floor((points_tile[:, 0] - cur_x) / resolution).astype(np.int32)
            points_y = np.floor((points_tile[:, 1] - cur_y) / resolution).astype(np.int32)

            height = int(tile_size / resolution)
            width = int(tile_size / resolution)

            # clipping against out of bounds and such values
            points_x = np.clip(points_x, 0, width - 1)
            points_y = np.clip(points_y, 0, height - 1)

            # # for remapping
            # pixel_to_points = [[[] for _ in range(width)] for _ in range(height)]
            # for local_idx in range(len(points_x)):
            #     x_ = points_x[local_idx]
            #     y_ = points_y[local_idx]
            #     pixel_to_points[y_][x_].append(local_idx)

            # process = psutil.Process(os.getpid())
            # mem_mb = process.memory_info().rss / 1024**2
            # print(f"RAM usage: {mem_mb:.2f} MB")
            
            # aggregate using Numba
            bev = _numba_aggregate(points_x, points_y, points_tile[:, 2], intensities_tile, height, width, labels_tile, num_classes)
            
            # normalize
            # IMPORTANT -> normalizing looses the real world absolute values
            # FIXME maybe remove normalizing
            # FIXME maybe add also other informations -> max intrensity for metal?
            # bev[0] /= z.max()
            # bev[2] /= intensities.max()

            # class/label back mapping
            if include_class:
                inverse_class_map = {new_: origin for origin, new_ in class_map.items()}
                inverse_array = np.array([inverse_class_map[i] for i in range(len(inverse_class_map))])

                mapped = bev[4].astype(np.int32)
                valid_mask = mapped != 255

                bev[4][:] = 255
                bev[4][valid_mask] = inverse_array[mapped[valid_mask]]
                # here we use the class channel as indexing for the indexing array
                # position in this indexing array = new_mapping
                # value = original value
                # works, because the new mapping is from 0-n in a linear order


            cur_meta = {
                    # "tile_id": tile_id,
                    "origin_x": cur_x,  # origin x -> start x
                    "origin_y": cur_y,  # origin y
                    # "global_indices": idxs,
                    # "pixel_to_points": pixel_to_points,
                    "resolution": resolution,
                    "tile_size": tile_size,
                    "pc_id": pc_id,
                    # "tile_points_local": points_tile  # can causes memory error during saving
                }
            
            if sample_path is not None and len(sample_tiles) == max_sample_amount and not sample_saving_completed:
                save_bev_tiles_as_images(sample_tiles, folder=sample_path)
                sample_saving_completed = True
            
            if sample_path is not None and not sample_saving_completed and cur_step >= start_step:
                sample_tiles.append(bev)
            
            if direct_single_saving:
                save_single_bev_tile_as_pickle(tile=bev, 
                                               meta=cur_meta, 
                                               pc_id=pc_id, 
                                               path=single_saving_path)
                patch_info.append((pc_id, cur_x, cur_y, tile_size))
            else:
                tiles.append(bev)
                meta.append(cur_meta)

            # save also 3D Point Cloud
            if save_3d_patches:
                # idxs are the globalen indices, which we already calculated
                pc_patch = point_cloud.select_by_index(idxs) 
                
                pc_patch_name = f"preprocessed_patch_{pc_id}_{cur_x}_{cur_y}.h5"
                save_point_cloud(path=os.path.join(single_saving_path, pc_patch_name), 
                                 point_cloud=pc_patch)
            
            tile_id += 1
    
    if direct_single_saving:
        return patch_info
    else:
        return tiles, meta



def bev_pixel_to_3d(
        patch_points,
        pixel_x,
        pixel_y,
        origin_x,    # in meta
        origin_y,    # in meta
        resolution,  # in meta
        search_radius
):
    """
    Converts a BEV pixel coordinate into 3D world coordinate.
    """
    if isinstance(patch_points, o3d.t.geometry.PointCloud):
        patch_points = patch_points.point[get_coordinate_attribute(patch_points)].cpu().numpy()
    elif isinstance(patch_points, PointCloudTensor):
        patch_points = patch_points.to_numpy(as_copy=True).coordinates

    # pixel center to world xy
    world_x = origin_x + (pixel_x + 0.5) * resolution
    world_y = origin_y + (pixel_y + 0.5) * resolution

    if search_radius is None:
        search_radius = resolution * 1.5

    # search nearby points for z
    mask = (
        (patch_points[:,0] >= world_x - search_radius) &
        (patch_points[:,0] <= world_x + search_radius) &
        (patch_points[:,1] >= world_y - search_radius) &
        (patch_points[:,1] <= world_y + search_radius)
    )
    nearby_points = patch_points[mask]

    # no nearby points
    if len(nearby_points) == 0:
        return np.array([world_x, world_y, np.nan])  # or 0?
    
    # robust z estimation
    world_z = np.median(nearby_points[:,2])

    return np.array([world_x, world_y, world_z])



def bev_pixel_to_world_area(pixel_x, pixel_y,
                            origin_x, origin_y,
                            resolution):
    """
    Convert BEV pixel into world-space bounds.
    """

    x_min = origin_x + pixel_x * resolution
    x_max = x_min + resolution

    y_min = origin_y + pixel_y * resolution
    y_max = y_min + resolution

    return x_min, x_max, y_min, y_max



# @numba.njit(parallel=True)
def bev_projection_testing(patch_gen, atol=1e-4, dataset_name=None, save_path=None):
    """
    Validates BEV channels using geometric reprojection.

    Checks:
        - max height
        - delta z
        - mean intensity
        - density
        - majority class
    """

    print("Starting geometric BEV validation...")

    # init all needed vars
    total_pixels = 0
    total_non_empty_pixels = 0
    total_empty_pixels = 0

    correct_max_height = 0
    correct_delta_z = 0
    correct_intensity = 0
    correct_density = 0
    correct_class = 0
    # correct_empty_pixels = 0

    max_height_error = 0.0
    delta_z_error = 0.0
    intensity_error = 0.0
    density_error = 0.0

    for tile_id, patch_points in tqdm(enumerate(patch_gen),
                                      total=len(patch_gen),
                                      desc="BEV Validation"):
        if isinstance(patch_points, (list, tuple)):
            patch_points = patch_points[0]

        if not isinstance(patch_points, PointCloudTensor):
            raise TypeError(f"Expected patch points to be 'PointCloudTensor' but got '{type(patch_points)}'")
        
        # LOAD BEV
        
        # if point_cloud.bev_data is None:
        #     print("Starting BEV projection...")
        #     tiles, metas = bev_projection(patch_points, tile_size=35.0, resolution=0.05, overlap=0.0,
        #                                   include_class=False, direct_single_saving=False)
        #     # bev_gen = bev_gen_wrapper(tiles, metas)
        # else:
        # bev_gen = patch_points.get_bev()
        # tiles, metas = extract_tiles_metas(bev_gen, amount=5, as_numpy=True)
        bev_gen = patch_points.get_bev()
        bev_dict = next(bev_gen)

        bev = bev_dict["pixel_values"]

        if isinstance(bev, torch.Tensor):
            bev = bev.detach().cpu().numpy()

        if "labels" in bev_dict and bev_dict["labels"] is not None:
            labels_bev = bev_dict["labels"]
            if isinstance(labels_bev, torch.Tensor):
                labels_bev = labels_bev.detach().cpu().numpy()

            bev = np.concatenate([bev, labels_bev[None]], axis=0)

        meta = bev_dict["meta"]

        # LOAD PATCH POINT CLOUD

        if isinstance(patch_points, PointCloudTensor):
            patch_points = patch_points.get_as_o3d()

        points = patch_points.point[
            get_coordinate_attribute(patch_points)
        ].numpy()

        intensities = patch_points.point[
            get_intensity_attribute(patch_points)
        ].numpy().ravel().astype(np.float32)

        labels = patch_points.point[get_class_attribute(patch_points)].numpy()
        labels = np.asarray(labels).reshape(-1).astype(np.int32)

        num_classes = int(labels.max()) + 1

        # META

        origin_x = meta["origin_x"]
        origin_y = meta["origin_y"]
        resolution = meta["resolution"]

        height, width = bev.shape[1], bev.shape[2]

        # global normalization references
        max_intensity_global = intensities.max()

        # PIXEL LOOP

        # px = ((points[:, 0] - origin_x) / resolution).astype(np.int32)
        px = np.floor(
                (points[:, 0] - origin_x) / resolution
            ).astype(np.int32)
        # py = ((points[:, 1] - origin_y) / resolution).astype(np.int32)
        py = np.floor(
                (points[:, 1] - origin_y) / resolution
            ).astype(np.int32)

        valid = (
            (px >= 0) & (px < width) &
            (py >= 0) & (py < height)
        )

        px = px[valid]
        py = py[valid]

        z = points[:, 2][valid]
        intensities_valid = intensities[valid]
        labels_valid = labels[valid]

        pixel_idx = py * width + px

        # grouping
        order = np.argsort(pixel_idx)
        pixel_idx = pixel_idx[order]
        z = z[order]
        intensities_valid = intensities_valid[order]
        labels_valid = labels_valid[order]

        unique_pixels, start_idx, counts = np.unique(
            pixel_idx,
            return_index=True,
            return_counts=True
        )

        # empty pixels
        # occupied = np.zeros(height * width, dtype=np.bool_)
        # occupied[unique_pixels] = True
        # empty_pixels = (~occupied).sum()

        gt_occupied = np.zeros(height * width, dtype=np.bool_)
        gt_occupied[unique_pixels] = True

        gt_empty = ~gt_occupied
        # pred_empty = ~occupied

        # correct_empty_pixels += np.sum(pred_empty & gt_empty)
        cur_empty_pixels = np.sum(gt_empty)

        total_non_empty_pixels += ((height * width - cur_empty_pixels))
        # total_empty_pixels += cur_empty_pixels

        total_pixels += height * width

        # compute other values (intensities, ..)
        for pix, start, count in zip(unique_pixels, start_idx, counts):
            indices = slice(start, start+count)

            pixel_z = z[indices]
            pixel_intensity = intensities_valid[indices]
            pixel_labels = labels_valid[indices]

            gt_max_height = pixel_z.max()
            gt_delta_z = pixel_z.max() - pixel_z.min()
            gt_mean_intensity = pixel_intensity.mean()
            # FIXME
            # gt_mean_intensity = pixel_intensity.mean() / pixel_intensity.max()
            # gt_mean_intensity = (pixel_intensity.mean() - pixel_intensity.min()) / (pixel_intensity.max() - pixel_intensity.min())
            gt_density = np.log1p(count)

            gt_class = np.bincount(pixel_labels).argmax()
            # counts = np.zeros(num_classes, dtype=np.int32)

            # for l in pixel_labels:
            #     counts[l] += 1

            py = pix // width
            px = pix % width

            # BEV VALUES

            bev_max_height = bev[0, py, px]
            bev_delta_z = bev[1, py, px]
            bev_intensity = bev[2, py, px]
            bev_density = bev[3, py, px]
            bev_class = int(bev[4, py, px])

            # ERROR METRICS

            max_height_error += abs(
                bev_max_height - gt_max_height
            )

            delta_z_error += abs(
                bev_delta_z - gt_delta_z
            )

            intensity_error += abs(
                bev_intensity - gt_mean_intensity
            )

            density_error += abs(
                bev_density - gt_density
            )

            # ACCURACY

            # if np.isclose(
            #     bev_max_height,
            #     gt_max_height,
            #     atol=atol
            # ):
            #     correct_max_height += 1
            if abs(bev_max_height - gt_max_height) <= atol:
                correct_max_height += 1

            # if np.isclose(
            #     bev_delta_z,
            #     gt_delta_z,
            #     atol=atol
            # ):
            #     correct_delta_z += 1
            if abs(bev_delta_z - gt_delta_z) <= atol:
                correct_delta_z += 1

            # if np.isclose(
            #     bev_intensity,
            #     gt_mean_intensity,
            #     atol=atol
            # ):
            #     correct_intensity += 1
            
            if abs(bev_intensity - gt_mean_intensity) <= atol:
                correct_intensity += 1

            # if np.isclose(
            #     bev_density,
            #     gt_density,
            #     atol=atol
            # ):
            #     correct_density += 1
            if abs(bev_density - gt_density) <= atol:
                correct_density += 1

            if bev_class == gt_class:
                correct_class += 1

    coverage = total_non_empty_pixels / total_pixels
    coverage_percent = coverage * 100

    # FINAL REPORT

    final_report = "\n\n===== BEV VALIDATION RESULTS ====="

    if dataset_name is not None:
        final_report += f"\n\nTested Dataset: {dataset_name}"

    final_report += f"\n\nTotal pixels: {total_pixels}"
    final_report += f"\nNon-empty pixels: {total_non_empty_pixels}"
    final_report += f"\nPixel Value Coverage: {coverage_percent:.2f} %"

    final_report += "\n\n--- Max Height ---"
    final_report += f"\nAccuracy: {(correct_max_height/total_non_empty_pixels)*100:.2f}%"
    final_report += f"\nAbsolute Error Sum: {max_height_error:.6f}"

    final_report += "\n\n--- Delta Z ---"
    final_report += f"\nAccuracy: {(correct_delta_z/total_non_empty_pixels)*100:.2f}%"
    final_report += f"\nAbsolute Error Sum: {delta_z_error:.6f}"

    final_report += "\n\n--- Intensity ---"
    final_report += f"\nAccuracy: {(correct_intensity/total_non_empty_pixels)*100:.2f}%"
    final_report += f"\nAbsolute Error Sum: {intensity_error:.6f}"

    final_report += "\n\n--- Density ---"
    final_report += f"\nAccuracy: {(correct_density/total_non_empty_pixels)*100:.2f}%"
    final_report += f"\nAbsolute Error Sum: {density_error:.6f}"

    final_report += "\n\n--- Class ---"
    final_report += f"\nAccuracy: {(correct_class/total_non_empty_pixels)*100:.2f}%\n\n\n"

    print(final_report)

    if save_path is not None:
        with open(save_path, "w") as file_:
            file_.write(final_report)

        print(f"\nSaved the report at: {save_path}")

    















