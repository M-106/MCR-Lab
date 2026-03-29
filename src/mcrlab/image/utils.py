# -----------
# > Imports <
# -----------
import numpy as np
from PIL import Image

from mcrlab.point_cloud.utils import get_coordinate_attribute, get_intensity_attribute
from mcrlab.point_cloud.core import PointCloudTensor


# ---------
# > Utils <
# ---------
def normalize_img(img: np.ndarray):
    # Normalize to 0-255 for image
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img_norm = (img - img_min) / (img_max - img_min)
    else:
        img_norm = img * 0

    return (img_norm * 255).astype(np.uint8)



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
    *Put explanation here
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



def bev_projection_mapping(point_cloud, meta, tile_id, pixel):
    """
    *Put explanation here
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

    points = point_cloud.point[get_coordinate_attribute(point_cloud)].numpy()[globla_indices]

    return points








