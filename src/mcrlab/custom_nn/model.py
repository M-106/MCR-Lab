# -----------
# > Imports <
# -----------
from typing import Union

import torch
import torch.optim as optim



# ----------
# > Getter <
# ----------
def get_model(name:str, check_point_path:Union[None, str], device="cuda", check_valid=True):
    name = name.lower()

    # load model
    model = None

    # smp and custom unset! AND heatmap unet
    if name == "unet":
        model = # FIXME
    # ...

    # optional check
    if check_valid:
        if model is None:
            raise ValueError(f"Could not create Optimizer with name '{name}'.")

    # load checkpoints
    if check_point_path is not None and model is not None:
        # FIXME -> how to load the right way?
        #  -> maybe load for every model on its own, maybe with smp different?
        model = model.load_state_dict(torch.load(check_point_path, weights_only=True))

    # move to device
    if model is not None and device is not None:
        model = model.to(device)

    return model







