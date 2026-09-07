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


# values come from global 2D train dataset
SENSOR_CONFIGS = {
    'as-900hl': {  # WHU
        'intensity_p1': 0.0, 
        'intensity_p99': 6069.0,
        'intensity_mean': 236.1677, 
        'intensity_std': 1416.8572, 
        'intensity_min': 0.0,
        'intensity_max': 65534.0,
        # 'max_density': 1.0,
        'height_min': 0.0,
        'height_max': 23.0701,
        'delta_height_min': 0.0,
        'delta_height_max': 1.5090,
        'density_min': 0.0,
        'density_max': 9.1638
    },
    'riegl_vux-1ha_mls': {  # sud
        'intensity_p1': 0.0, 
        'intensity_p99': 3012.0, 
        'intensity_mean': 418.6283, 
        'intensity_std': 940.646,
        'intensity_min': 0.0,
        'intensity_max': 4314.5,
        #'max_density': 1.0,
        'height_min': 0.0,
        'height_max': 0.0,
        'delta_height_min': 0.0,
        'delta_height_max': 27.595,
        'density_min': 0.0,
        'density_max': 3.1355
    },
    # 'velodyne_hdl64': {'intensity_p1': 0.0, 'intensity_p99': 255.0, 'max_density': 100},
    # 'hesai_pandar64': {'intensity_p1': 0.0, 'intensity_p99': 255.0, 'max_density': 150},
    # 'ouster_os1_128': {'intensity_p1': 0.0, 'intensity_p99': 65535.0, 'max_density': 200},
    'default': {'intensity_p1': 0.0, 'intensity_p99': 255.0, 'max_density': 100}
}

def safe_minmax(channel, min_value=None, max_value=None, eps=1e-6, clipping=True):
    """
    Robust min-max normalization to [0, 1].

    Handles:
        - NaN
        - +/- Inf
        - constant channels
        - empty valid masks
    """

    channel = channel.astype(np.float32, copy=False)

    # Only use valid values
    valid = np.isfinite(channel)

    # If everything is inf/nan
    if not np.any(valid):
        return np.zeros_like(channel, dtype=np.float32)

    valid_values = channel[valid]

    if min_value is None:
        min_value = valid_values.min()
    if max_value is None:
        max_value = valid_values.max()

    denom = max_value - min_value

    out = np.zeros_like(channel, dtype=np.float32)

    # Constant Channel
    if denom < eps:
        return out

    out[valid] = (channel[valid] - min_value) / denom

    if clipping:
        out = np.clip(out, 0.0, 1.0)

    return out

def safe_standard_norm(channel, mean, std, eps=1e-6, clip_range=None):
    """
    Standardizes a channel using mean and standard deviation: (x - mean) / std.
    
    Optionally clips extreme values (e.g. clip_range=(-3.0, 3.0)).
    """
    channel = channel.astype(np.float32, copy=False)
    
    # Avoid division by zero or tiny std
    std = max(std, eps)
    
    # Normalize
    out = (channel - mean) / std
    
    # Fill non-finite values (if present) with zeros
    invalid = ~np.isfinite(out)
    if np.any(invalid):
        out[invalid] = 0.0

    if clip_range is not None:
        out = np.clip(out, clip_range[0], clip_range[1])
        
    return out

def normalize_bev_robust(
    bev,
    channel_names,
    sensor_type='default', 
    intensity_norm_mode='standard',  # Options: 'standard', 'minmax'
    intensity_clip_range=(-3.0, 3.0) # Used only if norm_mode is 'standard'
):
    """
    Robust BEV-data normalization via list of channel-names.
    """
    is_chw = bev.shape[0] <= 10 and bev.shape[0] == len(channel_names)
    if is_chw:
        bev = np.transpose(bev, (1, 2, 0))

    if bev.shape[-1] != len(channel_names):
        raise ValueError(f"Amount of channel names ({len(channel_names)}) "
                         f"does not fit to BEV shape ({bev.shape}).")

    bev_norm = bev.copy()
    cfg = SENSOR_CONFIGS.get(sensor_type, SENSOR_CONFIGS['default'])

    for idx, name in enumerate(channel_names):
        channel = bev_norm[:, :, idx]
        name_clean = name.lower().strip()

        if name_clean in ['max_height', 'z_max', 'height']:
            min_, max_ = cfg['height_min'], cfg['height_max']
            channel = safe_minmax(channel, min_value=min_, max_value=max_, eps=1e-6, clipping=False)

            # channel = safe_minmax(channel, min_value=z_bounds[0], max_value=z_bounds[1], eps=1e-6, clipping=True)

        elif name_clean in ['delta_z', 'height_diff']:
            min_, max_ = cfg['delta_height_min'], cfg['delta_height_max']
            channel = safe_minmax(channel, min_value=min_, max_value=max_, eps=1e-6, clipping=False)

            # channel = safe_minmax(channel, min_value=max_delta_bounds[0], max_value=max_delta_bounds[1], eps=1e-6, clipping=True)

        elif name_clean in ['intensity', 'mean_intensity']:
            if intensity_norm_mode == 'standard':
                # Standard Image Normalization: (x - mean) / std
                mean = cfg.get('intensity_mean', 127.5)
                std = cfg.get('intensity_std', 73.5)
                channel = safe_standard_norm(
                    channel, 
                    mean=mean, 
                    std=std, 
                    clip_range=intensity_clip_range
                )
            elif intensity_norm_mode == 'minmax':
                # Percentile Min-Max Normalization: [0.0, 1.0]
                p1, p99 = cfg['intensity_p1'], cfg['intensity_p99']
                channel = safe_minmax(channel, min_value=p1, max_value=p99, eps=1e-6, clipping=True)
            else:
                raise ValueError(f"Unknown intensity_norm_mode: {intensity_norm_mode}")

        elif name_clean in ['density', 'point_count']:
            # channel = safe_minmax(channel, min_value=None, max_value=None, eps=1e-6, clipping=True)
            min_, max_ = cfg['density_min'], cfg['density_max']
            channel = safe_minmax(channel, min_value=min_, max_value=max_, eps=1e-6, clipping=False)

        elif name_clean in ['class', 'label', 'category', 'ignore']:
            pass

        else:
            raise ValueError(f"Unknown channel name: '{name}'")

        bev_norm[:, :, idx] = channel

    if is_chw:
        bev_norm = np.transpose(bev_norm, (2, 0, 1))

    return bev_norm



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








