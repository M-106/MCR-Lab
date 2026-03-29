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

    return (img_norm * 255).astype(np.uint8)








