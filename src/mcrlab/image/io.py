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

from mcrlab.image.utils import normalize_img_per_channel, apply_colormap, random_colorize



# -------------
# > Functions <
# -------------
def save_bev_tiles_as_images(tiles, folder="./bev_images"):
    os.makedirs(folder, exist_ok=True)
    shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)


    for i, bev in enumerate(tiles):
    #     print(f"\nSample Saving look inside (before normalizing):")
    #     print(f"  - Dtype: {bev.dtype}")
    #     print(f"  - Shape: {bev.shape}")
    #     print(f"  - Min/Max: ({bev.min()}, {bev.max()})")
    #     for channel in range(bev.shape[0]):
    #         print(f"      - Channel {channel} -> Min/Max: ({bev[channel, :, :].min()}, {bev[channel, :, :].max()}")


        # Normalize value range
        # bev_img = normalize_img(bev)
        bev_img = np.transpose(bev, (1, 2, 0))  # [C, H, W] -> [H, W, C]
        bev_img_3 = normalize_img_per_channel(bev_img[:, :, :3], skip_already_normalized_channels=True)

        # upscale
        bev_img_3 *= 255

        # type conversion for PIL image
        # bev_img = bev_img.clip(0, 255).astype(np.uint8)


        # print(f"\n  After processing:")
        # print(f"  - Dtype: {bev_img.dtype}")
        # print(f"  - Shape: {bev_img.shape}")
        # print(f"  - Min/Max: ({bev_img.min()}, {bev_img.max()})")
        # for channel in range(bev_img.shape[-1]):
        #     print(f"      - Channel {channel} -> Min/Max: ({bev_img[:, :, channel].min()}, {bev_img[:, :, channel].max()}")

        # print("Debug View End\n")

        # Convert to PIL image
        img = Image.fromarray(bev_img_3)
        img.save(os.path.join(folder, f"tile_{i:03d}_all_channels.png"))
        Image.fromarray(apply_colormap(bev_img_3[:, :, 0], cmap_name="nipy_spectral")).save(os.path.join(folder, f"tile_{i:03d}_max_height_channel_nipy.png"))
        Image.fromarray(apply_colormap(bev_img_3[:, :, 1], cmap_name="nipy_spectral")).save(os.path.join(folder, f"tile_{i:03d}_min_height_channel_nipy.png"))
        Image.fromarray(apply_colormap(bev_img_3[:, :, 2], cmap_name="nipy_spectral")).save(os.path.join(folder, f"tile_{i:03d}_intensity_channel_nipy.png"))
        Image.fromarray(apply_colormap(bev_img_3[:, :, 0], cmap_name="viridis")).save(os.path.join(folder, f"tile_{i:03d}_max_height_channel_viridis.png"))
        Image.fromarray(apply_colormap(bev_img_3[:, :, 1], cmap_name="viridis")).save(os.path.join(folder, f"tile_{i:03d}_min_height_channel_viridis.png"))
        Image.fromarray(apply_colormap(bev_img_3[:, :, 2], cmap_name="viridis")).save(os.path.join(folder, f"tile_{i:03d}_intensity_channel_viridis.png"))

        if bev_img.shape[-1] > 3:
            Image.fromarray(random_colorize(bev_img[:, :, 3])).save(os.path.join(folder, f"tile_{i:03d}_label_channel.png"))

        # plt.imshow(bev_img[:, :, 2], cmap="nipy_spectral")  #"gnuplot2", "nipy_spectral", "gist_rainbow", "rainbow"
        # plt.savefig(os.path.join(folder, f"tile_{i:03d}_intensity_channel_v2.png"))
        # plt.clf()

    print(f"Samples saved in '{folder}'")
    # print(f"Saved {len(tiles)} BEV images to '{folder}'")



def save_single_bev_tile_as_pickle(tile, meta, pc_id, path):
    # adjust file names
    cur_bev_data_file_name = "preprocessed_patch_bev_" + f"{pc_id}_{meta['origin_x']}_{meta['origin_y']}.pkl"
    cur_bev_data_path = os.path.join(path, cur_bev_data_file_name)

    data = (tile, meta)

    # saving
    with open(cur_bev_data_path, "wb") as file_:
        pickle.dump(data, file_)



def load_single_bev_tile_as_pickle(path):
    if not path.endswith(".pkl"):
        path += ".pkl"

    # load the bev file (tile/image + meta)
    with open(path, "rb") as file_:
        tile, meta = pickle.load(file_)

    return (tile, meta)





    






