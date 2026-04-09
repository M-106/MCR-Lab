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



def save_bev_tiles_as_pickle(tiles, metas, path):
    data = (tiles, metas)

    if not path.endswith(".pkl"):
        path += ".pkl"

    with open(path, "wb") as file_:
        pickle.dump(data, file_)



def load_bev_tiles_as_pickle(path):
    if not path.endswith(".pkl"):
        path += ".pkl"

    with open(path, "rb") as file_:
        data = pickle.load(file_)

    return data



def save_bev_tiles_as_pt(tiles, metas, path):
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



def load_bev_tiles_as_pt(path, return_tiles_as_list_numpy_array=False):
    if not path.endswith(".pt"):
        path += ".pt"

    data = torch.load(path, map_location="cpu")

    tiles_tensor = data["tiles"]
    meta = data["meta"]

    # back to List[np.ndarray], if wished
    if return_tiles_as_list_numpy_array:
        tiles = [cur_tile.numpy() for cur_tile in tiles_tensor]

    return tiles, meta







