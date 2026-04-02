# -----------
# > Imports <
# -----------
import numpy as np
from PIL import Image



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


def normalize_img_per_channel(img: np.ndarray, skip_already_normalized_channels=True):
    img = img.astype(np.float32)
    img_norm = np.zeros_like(img)

    for c_idx in range(img.shape[2]):
        channel = img[:, :, c_idx]
        min_, max_ = channel.min(), channel.max()

        if max_ > min_:
            if max_ > 1.0:  # only normalize if the channel is not already normalized
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











