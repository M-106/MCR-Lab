# -----------
# > Imports <
# -----------
import os
import shutil
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

from mcrlab.image.utils import normalize_img_per_channel



# -------------
# > Functions <
# -------------
def save_bev_tiles_as_images(tiles, folder="./bev_images"):
    os.makedirs(folder, exist_ok=True)
    shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)


    for i, bev in enumerate(tiles):
        # Normalize value range
        # bev_img = normalize_img(bev)
        bev_img = normalize_img_per_channel(bev, skip_already_normalized_channels=True)

        bev_img *= 255

        bev_img = np.transpose(bev_img, (1, 2, 0))  # H, W, C

        # Convert to PIL image
        img = Image.fromarray(bev_img)
        img.save(os.path.join(folder, f"tile_{i:03d}_all_channels.png"))
        Image.fromarray(bev_img[:, :, 0]).save(os.path.join(folder, f"tile_{i:03d}_max_height_channel.png"))
        Image.fromarray(bev_img[:, :, 1]).save(os.path.join(folder, f"tile_{i:03d}_min_height_channel.png"))
        Image.fromarray(bev_img[:, :, 2]).save(os.path.join(folder, f"tile_{i:03d}_intensity_channel.png"))

        # plt.imshow(bev_img[:, :, 2], cmap="nipy_spectral")  #"gnuplot2", "nipy_spectral", "gist_rainbow", "rainbow"
        # plt.savefig(os.path.join(folder, f"tile_{i:03d}_intensity_channel_v2.png"))
        # plt.clf()

    # print(f"Saved {len(tiles)} BEV images to '{folder}'")



def save_bev_tiles_as_pickle_together(tiles, metas, path):
    data = (tiles, metas)

    if not path.endswith(".pkl"):
        path += ".pkl"

    with open(path, "wb") as file_:
        pickle.dump(data, file_)



def load_bev_tiles_as_pickle_together(path):
    if not path.endswith(".pkl"):
        path += ".pkl"

    with open(path, "rb") as file_:
        data = pickle.load(file_)

    return data



def save_bev_tiles_as_pickle(tiles, metas, path):
    if not path.endswith(".pkl"):
        path += ".pkl"

    # adjust file names
    root_path, file_name = os.path.split(path)
    file_name_without_ending = ".".join(file_name.split(".")[:-1])

    all_paths = []

    n_tiles = len(tiles)
    n_metas = len(metas)
    assert n_tiles == n_metas
    for cur_bev_idx in range(n_tiles):
        cur_bev_data_file_name = "single_bev_" + file_name_without_ending + f"_{cur_bev_idx:02}.pkl"
        cur_bev_data_path = os.path.join(root_path, cur_bev_data_file_name)

        data = (tiles[cur_bev_idx], metas[cur_bev_idx])

        # saving
        with open(cur_bev_data_path, "wb") as file_:
            pickle.dump(data, file_)

        all_paths.append(cur_bev_data_path)

    # save paths
    with open(path, "wb") as file_:
        pickle.dump(all_paths, file_)



def load_single_bev_tile_as_pickle(path):
    if not path.endswith(".pkl"):
        path += ".pkl"

    # load the bev file (tile/image + meta)
    with open(path, "rb") as file_:
        tiles, metas = pickle.load(file_)

    return (tiles, metas)


def load_bev_tiles_as_pickle(path):
    if not path.endswith(".pkl"):
        path += ".pkl"

    # load file which saves the paths
    with open(path, "rb") as file_:
        all_paths = pickle.load(file_)

    # load every bev file (tile/image + meta)
    tiles = []
    metas = []
    for cur_file_path in all_paths:
        with open(cur_file_path, "rb") as file_:
            cur_tile, cur_meta = pickle.load(file_)
        tiles.append(cur_tile)
        metas.append(cur_meta)

    return (tiles, metas)



def save_bev_tiles_as_pt_together(tiles, metas, path):
    if not path.endswith(".pt"):
        path += ".pt"

    # convert list -> tensor explizit
    tiles_tensor = torch.stack(
        [torch.from_numpy(cur_tile) for cur_tile in tiles]
    ).float()   # (N, C+1, H, W)

    torch.save(
        {
            "tiles": tiles_tensor,
            "meta": metas
        },
        path
    )



def load_bev_tiles_as_pt_together(path, return_tiles_as_list_numpy_array=False):
    if not path.endswith(".pt"):
        path += ".pt"

    data = torch.load(path, map_location="cpu")

    tiles_tensor = data["tiles"]
    meta = data["meta"]

    # back to List[np.ndarray], if wished
    if return_tiles_as_list_numpy_array:
        tiles = [cur_tile.numpy() for cur_tile in tiles_tensor]

    return tiles, meta



# def save_bev_tiles_as_pt(tiles, metas, path):
#     raise ValueError("Adjust this method as the pickle file.")
#     if not path.endswith(".pt"):
#         path += ".pt"

#     # adjust file names
#     root_path, file_name = os.path.split(path)
#     tiles_file_name = "tiles_"+file_name
#     metas_file_name = "metas_"+file_name
#     tiles_path = os.path.join(root_path, tiles_file_name)
#     metas_path = os.path.join(root_path, metas_file_name)

#     # convert list -> tensor explizit
#     tiles_tensor = torch.stack(
#         [torch.from_numpy(cur_tile) for cur_tile in tiles]
#     ).float()   # (N, C+1, H, W)

#     # save both seperatly
#     torch.save(
#         tiles_tensor,
#         tiles_path
#     )

#     torch.save(
#         metas,
#         metas_path
#     )



# def load_bev_tiles_as_pt(path, return_tiles_as_list_numpy_array=False, only_tiles=False):
#     raise ValueError("Adjust this method as the pickle file.")
#     if not path.endswith(".pt"):
#         path += ".pt"

#     root_path, file_name = os.path.split(path)
#     if file_name.startswith("tiles_"):
#         tiles_file_name = file_name
#         metas_file_name = "metas_" + "_".join(file_name.split("_")[1:])
#     elif file_name.startswith("metas_"):
#         tiles_file_name = "tiles_" + "_".join(file_name.split("_")[1:])
#         metas_file_name = file_name
#     else:
#         tiles_file_name = "tiles_" + file_name
#         metas_file_name = "metas_" + file_name

#     tiles_path = os.path.join(root_path, tiles_file_name)
#     metas_path = os.path.join(root_path, metas_file_name)

#     tiles = torch.load(tiles_path, map_location="cpu")
#     if only_tiles:
#         metas = None
#     else:
#         metas = torch.load(metas_path, map_location="cpu")

#     # back to List[np.ndarray], if wished
#     if return_tiles_as_list_numpy_array:
#         tiles = [cur_tile.numpy() for cur_tile in tiles]

#     return tiles, metas

    



