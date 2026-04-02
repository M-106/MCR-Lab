# -----------
# > Imports <
# -----------
import numpy as np
from PIL import Image

import open3d as o3d

import numba

# dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mcrlab.point_cloud.utils import get_coordinate_attribute, get_intensity_attribute
from mcrlab.point_cloud.tensor_wrapper import PointCloudTensor, torch_tensor_to_numpy, torch_tensor_type_to_numpy_type



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

# FIXME -> add saving and loading of BEV in preprocessing
# FIXME -> add compilation with numba!

def bev_projection(point_cloud, tile_size=10.0, resolution=0.5):
    """
    Projects a 3D point cloud into a Bird's Eye View (BEV) representation.

    The algorithm divides the point cloud into square tiles in the XY-plane 
    (top-down view). Each tile is then converted into a 2D grid (image), where 
    each pixel represents a small area of space defined by the given resolution.

    For every tile:
    - Points are filtered to only include those within the tile boundaries.
    - Their XY coordinates are discretized into pixel indices.
    - A 3-channel BEV image is created:
        1. Maximum height (Z) per pixel
        2. Minimum height (Z) per pixel
        3. Mean intensity per pixel

    If multiple points fall into the same pixel, their values are aggregated:
    - Max height → highest Z value
    - Min height → lowest Z value
    - Intensity → average intensity

    Additionally, the function stores metadata that links each BEV pixel back 
    to the original point indices. This allows tracing predictions or features 
    from the 2D BEV representation back to the 3D point cloud.
    """
    if isinstance(point_cloud, PointCloudTensor):
        point_cloud = point_cloud.get_as_o3d()

    points  = point_cloud.point[get_coordinate_attribute(point_cloud)].numpy()
    intensities = point_cloud.point[get_intensity_attribute(point_cloud)].numpy()
    # n_points = points.shape[0]

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # global boundings
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    tiles = []
    meta = []

    tile_id = 0

    for cur_x in np.arange(x_min, x_max, tile_size):
        for cur_y in np.arange(y_min, y_max, tile_size):
            # select points inside of tile
            mask = (
                (x >= cur_x) & (x < cur_x + tile_size) &
                (y >= cur_y) & (y < cur_y + tile_size)
            )

            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue

            points_tile = points[idxs]
            intensities_tile = intensities[idxs].ravel()

            # convert to pixels coordinates
            points_x = ((points_tile[:, 0] - cur_x) / resolution).astype(int)
            points_y = ((points_tile[:, 1] - cur_y) / resolution).astype(int)

            height = int(tile_size / resolution)
            width = int(tile_size / resolution)

            # create empty BEV grid
            bev = np.zeros((3, height, width), dtype=np.float32)

            CHANNEL_MAX_HEIGHT = 0
            CHANNEL_MIN_HEIGHT = 1
            CHANNEL_INTENSITY = 2

            # fill bev grid
            #  -> using `at` to handle duplicate values correctly
            # example -> np.add.at(intensity_sum, (points_x, points_y), intensity_tile)
            # is like: 
            # for i in range(len(points_x)):
            #     x = points_x[i]
            #     y = points_y[i]
            #     intensity_sum[x, y] += intensity_tile[i]

            #    aggregate max height per pixel
            np.maximum.at(bev[CHANNEL_MAX_HEIGHT], (points_x, points_y), points_tile[:, 2])

            bev[CHANNEL_MIN_HEIGHT].fill(np.inf)  # 0 not optimal when searching minimum
            np.minimum.at(bev[CHANNEL_MIN_HEIGHT], (points_x, points_y), points_tile[:, 2])
            bev[CHANNEL_MIN_HEIGHT][bev[CHANNEL_MIN_HEIGHT] == np.inf] = 0

            intensities_sum = np.zeros((height, width), dtype=np.float32)
            counts = np.zeros((height, width), dtype=np.int32)
            np.add.at(intensities_sum, (points_x, points_y), intensities_tile)
            np.add.at(counts, (points_x, points_y), 1)
            bev[CHANNEL_INTENSITY] = intensities_sum / np.maximum(counts, 1)

            # normalization
            bev[CHANNEL_MAX_HEIGHT] /= z.max()  # points_tile[:, 2].max()
            bev[CHANNEL_INTENSITY] /= intensities.max()  # intensities_tile.max()

            # print("BEV shape:", bev.shape)

            # build mapping: pixel -> point indices
            pixel_ids = points_x * width + points_y  # flatten 2D to 1D pixel index

            #  group points by pixel
            unique_pixels, inverse = np.unique(pixel_ids, return_inverse=True)

            #  store mapping
            pixel_to_indices = [np.where(inverse == idx)[0] \
                                    for idx in range(len(unique_pixels))]

            #     if key not in pixel_to_point_idxs.keys():
            #         pixel_to_point_idxs[key] = []
            #     pixel_to_point_idxs.append(idxs[point_idx])

            tiles.append(bev)

            meta.append({
                "tile_id": tile_id,
                "cur_x": cur_x,
                "cur_y": cur_y,
                # "pixel_map": pixel_to_point_idxs
                "unique_pixels": unique_pixels,
                "pixel_to_indices": pixel_to_indices,
                "global_indices": idxs
            })

            tile_id += 1

    return tiles, meta



@numba.njit
def _numba_aggregate(px, py, z, intensity, height, width):
    """
    Numba-accelerated per-pixel aggregation for Bird's Eye View (BEV) tiles.

    This function takes a set of points within a tile and computes a 3-channel
    BEV grid, where each pixel contains:
        1. Maximum height (Z) of points falling into that pixel
        2. Minimum height (Z) of points falling into that pixel
        3. Mean intensity of points falling into that pixel

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
    bev = np.zeros((3, height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.int32)
    bev[1, :, :] = 1e6  # initialize min height
    
    for i in range(len(px)):
        x = px[i]
        y = py[i]
        bev[0, x, y] = max(bev[0, x, y], z[i])        # max height
        bev[1, x, y] = min(bev[1, x, y], z[i])        # min height
        bev[2, x, y] += intensity[i]                  # sum intensity
        counts[x, y] += 1
    
    # finalize mean intensity
    for i in range(height):
        for j in range(width):
            if counts[i, j] > 0:
                bev[2, i, j] /= counts[i, j]
            else:
                bev[1, i, j] = 0.0
                bev[2, i, j] = 0.0
    
    return bev



def bev_projection_numba(point_cloud, tile_size=10.0, resolution=0.5):
    """
    Projects a 3D point cloud into Bird's Eye View (BEV) tiles using Numba for fast per-pixel aggregation.

    This function converts a 3D point cloud into a set of 2D BEV grids by dividing the XY-plane into 
    square tiles of a given size. Each tile is discretized into pixels, and for each pixel, the following
    features are computed:

        1. Maximum height (Z coordinate) of points in the pixel
        2. Minimum height (Z coordinate) of points in the pixel
        3. Mean intensity of points in the pixel

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
    # extract points and intensity
    if hasattr(point_cloud, "get_as_o3d"):
        point_cloud = point_cloud.get_as_o3d()
    
    points  = point_cloud.point[get_coordinate_attribute(point_cloud)].numpy()
    intensities = point_cloud.point[get_intensity_attribute(point_cloud)].numpy().ravel()
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    tiles = []
    meta = []
    tile_id = 0
    
    # iterate over tiles
    for cur_x in np.arange(x_min, x_max, tile_size):
        for cur_y in np.arange(y_min, y_max, tile_size):
            # select points inside this tile
            mask = (
                (x >= cur_x) & (x < cur_x + tile_size) &
                (y >= cur_y) & (y < cur_y + tile_size)
            )
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            
            points_tile = points[idxs]
            intensities_tile = intensities[idxs]
            
            # pixel coordinates
            points_x = ((points_tile[:, 0] - cur_x) / resolution).astype(np.int32)
            points_y = ((points_tile[:, 1] - cur_y) / resolution).astype(np.int32)
            
            height = int(tile_size / resolution)
            width = int(tile_size / resolution)
            
            # aggregate using Numba
            bev = _numba_aggregate(points_x, points_y, points_tile[:, 2], intensities_tile, height, width)
            
            # normalize
            bev[0] /= z.max()
            bev[2] /= intensities.max()
            
            tiles.append(bev)
            meta.append({
                "tile_id": tile_id,
                "cur_x": cur_x,
                "cur_y": cur_y,
                "global_indices": idxs
            })
            
            tile_id += 1
    
    return tiles, meta



def bev_projection_mapping(point_cloud, meta, tile_id, pixel):
    """
    Maps a BEV pixel back to its corresponding 3D points in the original point cloud.

    This function uses the metadata generated during the BEV projection to retrieve 
    all 3D points that contributed to a specific pixel in a given tile.

    How it works:
    - Each BEV pixel corresponds to a small area in the XY-plane.
    - During projection, points falling into the same pixel were grouped together.
    - The metadata stores this mapping (pixel → point indices).

    Given a tile ID and a pixel coordinate:
    - The pixel is converted into a flattened pixel index.
    - The function looks up which group of points belongs to that pixel.
    - It retrieves the corresponding indices of the original point cloud.
    - Finally, it returns the actual 3D points for that pixel.

    If no points exist for the given pixel, the function returns None.
    """
    tile = meta[tile_id]

    point_x, point_y = pixel
    height = width = int(np.sqrt(tile["unique_pixels"].max() + 1))

    pixel_id = point_x * width + point_y

    # find pixel
    matches = np.where(tile["unique_pixels"] == pixel_id)[0]
    if len(matches) == 0:
        return None
    
    group_idx = matches[0]

    local_indices = tile["pixel_to_indices"][group_idx]
    globla_indices = tile["global_indices"][local_indices] 

    if isinstance(point_cloud, PointCloudTensor):
        points = torch_tensor_to_numpy(point_cloud.coordinates, 
                                       dtype=torch_tensor_type_to_numpy_type(point_cloud.coordinates))
    elif isinstance(point_cloud, o3d.t.geometry.PointCloud):
        points = point_cloud.point[get_coordinate_attribute(point_cloud)].numpy()
    
    # apply indeces to get points
    points = points[globla_indices]

    return points






