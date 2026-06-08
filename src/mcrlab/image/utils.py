# -----------
# > Imports <
# -----------
import numpy as np
from PIL import Image
import matplotlib.cm as cm



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

    img_result = (img_norm * 255).astype(np.uint8)

    # if img.ndim >= 3:
    #     for channel in range(img.shape[2]):
    #         if img_result[:, :, channel].max() <= 1.01:
    #             img_result[:, :, channel] *= 255

    return img_result



def normalize_bev(bev):
    bev = bev.astype(np.float32)

    # C, H, W -> H, W, C
    # FIXME prüfen ob ok so!
    # Channel order prüfen
    transposed = False
    if bev.shape[0] <= 5:
        transposed = True
        bev = np.transpose(bev, [1, 2, 0])

    for cur_channel_idx in range(bev.shape[-1]):
        channel = bev[:, :, cur_channel_idx]

        # valid = channel[channel > 0]

        # if valid.size == 0:
        #     continue

        # max height = 0
        # delta height = 1
        # intensity = 2
        # density = 3
        if cur_channel_idx == 2:
            # percentile normalization 
            #  -> outlier min-max normalization
            p2, p98 = np.percentile(channel, [2, 98])
            denom = p98 - p2
            if denom > 1e-6:
                channel = np.clip((channel-p2) / denom, 0.0, 1.0)
            else:
                channel = np.zeros_like(channel, dtype=np.float32)
        else:
            # min-max normalization
            vmin, vmax = channel.min(), channel.max()
            if vmax - vmin > 1e-6:
                channel = np.clip((channel - vmin) / (vmax - vmin), 0.0, 1.0)
            else:
                channel = np.zeros_like(channel, dtype=np.float32)
            
        bev[:, :, cur_channel_idx] = channel

    if transposed:
        bev = np.transpose(bev, [2, 0, 1])

    return bev



def normalize_img_per_channel(img: np.ndarray, skip_already_normalized_channels=True):
    img = img.astype(np.float32)
    img_norm = np.zeros_like(img)

    for c_idx in range(img.shape[2]):
        channel = img[:, :, c_idx]
        min_, max_ = channel.min(), channel.max()

        if max_ > min_:
            if max_ > 1.0 and not skip_already_normalized_channels:  # only normalize if the channel is not already normalized
                img_norm[:, :, c_idx] = (channel - min_) / (max_ - min_)
            else:
                img_norm[:, :, c_idx] = channel
        else:
            img_norm[:, :, c_idx] = 0

    return (img_norm * 255).astype(np.uint8)



def one_channel_img_to_pil_rgb_img(image_input, return_numpy=False):
    # right now expect a one channel input

    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.max() <= 1.0:
            image_input = (image_input * 255).astype(np.uint8)
        image = Image.fromarray(image_input).convert("RGB")
    else:
        image = image_input

    if return_numpy:
        return np.array(image)
    else:
        return image



def apply_colormap(channel, cmap_name="viridis"):
    channel = channel.astype(np.float32)

    # normalisieren auf [0,1]
    channel -= channel.min()
    if channel.max() > 0:
        channel /= channel.max()

    cmap = cm.get_cmap(cmap_name)
    colored = cmap(channel)  # -> RGBA (H, W, 4)

    colored = (colored[:, :, :3] * 255).astype(np.uint8)  # .astype(np.uint8)  # RGB

    # print(f"  - Dtype: {colored.dtype}")
    # print(f"  - Shape: {colored.shape}")
    # print(f"  - Min/Max: ({colored.min()}, {colored.max()})")

    return colored



def random_colorize(arr, seed=0):
    """
    Converts a 2D array (H, W) with discrete values (e.g. labels)
    into a colored RGB image (H, W, 3).

    Each unique value in the array is assigned a random color.
    """
    # Find all unique values in the array (e.g. labels)
    # and create an "inverse" mapping:
    # - unique_vals: sorted list of unique values
    # - inverse: for each pixel, the index of its value in unique_vals
    unique_vals, inverse = np.unique(arr, return_inverse=True)

    # Create a random number generator with a fixed seed
    # so colors are reproducible across runs
    rng = np.random.default_rng(seed)

    # Generate a random RGB color for each unique value
    # shape: (number of unique values, 3)
    colors = rng.integers(0, 256, size=(len(unique_vals), 3), dtype=np.uint8)

    # Map each pixel to its corresponding RGB color:
    # - inverse contains, for each pixel, the index into unique_vals
    # - colors[inverse] maps that index to an RGB color
    # Then reshape back to image format (H, W, 3)
    colored = colors[inverse].reshape(arr.shape[0], arr.shape[1], 3)

    # Return the colored RGB image
    return colored








