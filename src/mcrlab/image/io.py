# -----------
# > Imports <
# -----------
import os
import shutil
import numpy as np
from PIL import Image

from mcrlab.image.utils import normalize_img



# -------------
# > Functions <
# -------------
def save_bev_tiles_as_images(tiles, folder="./bev_images"):
    os.makedirs(folder, exist_ok=True)
    shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)


    for i, bev in enumerate(tiles):
        # Normalize value range
        bev_img = normalize_img(bev)

        bev_img = np.transpose(bev_img, (1, 2, 0))  # H, W, C

        # Convert to PIL image
        img = Image.fromarray(bev_img)
        img.save(os.path.join(folder, f"tile_{i:03d}_all_channels.png"))
        Image.fromarray(bev_img[:, 0]).save(os.path.join(folder, f"tile_{i:03d}_max_height_channel.png"))
        Image.fromarray(bev_img[:, 1]).save(os.path.join(folder, f"tile_{i:03d}_min_height_channel.png"))
        Image.fromarray(bev_img[:, 2]).save(os.path.join(folder, f"tile_{i:03d}_intensity_channel.png"))

    # print(f"Saved {len(tiles)} BEV images to '{folder}'")







